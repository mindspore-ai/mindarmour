'''
inversion model
'''

import copy
import logging
from typing import Dict, Optional

import mindspore as ms
import mindspore.ops as ops
from mindnlp.transformers import PreTrainedModel, PreTrainedTokenizer, AutoModelForSeq2SeqLM, RobertaModel

from .config import InversionConfig
from .model_utils import (
    FREEZE_STRATEGIES,
    disable_dropout,
    freeze_params,
    load_embedder_and_tokenizer,
    load_encoder_decoder,
    load_tokenizer,
    mean_pool,
)


logger = logging.getLogger(__name__)


class InversionModel(PreTrainedModel):
    """A class of model that conditions on embeddings from a pre-trained sentence embedding model
    to decode text autoregressively.
    """

    config_class = InversionConfig
    embedder: ms.nn.Cell
    embedder_tokenizer: PreTrainedTokenizer  # embedder's tokenizer
    encoder_decoder: AutoModelForSeq2SeqLM
    encoder_decoder_lora: bool  # Whether to use LoRA for the encoder-decoder model
    tokenizer: PreTrainedTokenizer  # encoder_decoder's tokenizer
    embedding_transform: ms.nn.Cell  # Module that transformers embedder output into encoder-decoder input
    bottleneck_dim: int  # Bottleneck dimension for embedding_transform
    num_repeat_tokens: int  # Sequence length for repeating embedder embedding for encoder-decoder input
    embedder_dim: int  # Hidden dimension of embedding model
    embedder_no_grad: bool  # Disable gradients for embedding model
    embedder_fake_with_zeros: bool  # Whether to just provide zeros as input for encoder-decoder (unconditional)
    embedding_transform_strategy: str  # Way to transform bottleneck embedding into input for encoder-decoder
    use_frozen_embeddings_as_input: bool  # Whether to train/evaluate on frozen embeddings
    embedded_tokens: ms.Tensor  # used for decoding
    embedder_model_api: Optional[str]

    def __init__(self, config: InversionConfig):
        super().__init__(config=config)

        embedder_model_api = config.embedder_model_api
        embedder_fake_with_zeros = config.embedder_fake_with_zeros
        use_frozen_embeddings_as_input = config.use_frozen_embeddings_as_input
        encoder_dropout_disabled = config.encoder_dropout_disabled
        decoder_dropout_disabled = config.decoder_dropout_disabled
        embeddings_from_layer_n = config.embeddings_from_layer_n


        encoder_decoder = load_encoder_decoder(
            model_name=config.model_name_or_path,
            lora=config.use_lora,
        )


        embedder, embedder_tokenizer = load_embedder_and_tokenizer(
            name=config.embedder_model_name, torch_dtype=config.embedder_torch_dtype
        )

        tokenizer = load_tokenizer(
            config.model_name_or_path,
            max_length=config.max_seq_length,
        )
        num_repeat_tokens = config.num_repeat_tokens
        embedder_no_grad = config.embedder_no_grad

        self.encoder_decoder = encoder_decoder  # .to_bettertransformer()
        self.num_repeat_tokens = num_repeat_tokens

        self.embedder_is_decoder = False

        encoder_hidden_dim = self.encoder_decoder.config.hidden_size
        if embedder_model_api:
            assert use_frozen_embeddings_as_input, "must precompute embeddings w/ api"
            # Hard-code OpenAI embedding dim
            self.embedder_dim = 1536
            bottleneck_dim = self.embedder_dim
        # elif isinstance(embedder, mindnlp.transformers.models.t5.modeling_t5.T5ForConditionalGeneration):
        #     self.embedder_dim = embedder.get_sentence_embedding_dimension()
        #     bottleneck_dim = self.embedder_dim
        else:
            self.embedder_dim = embedder.config.hidden_size
            bottleneck_dim = self.embedder_dim
        self.embedder_no_grad = embedder_no_grad
        self.use_frozen_embeddings_as_input = use_frozen_embeddings_as_input
        self.bottleneck_dim = bottleneck_dim

        self.embedding_transform = ms.nn.SequentialCell(
            ms.nn.Dense(self.embedder_dim, bottleneck_dim),
            ms.nn.Dropout(self.encoder_decoder.config.dropout_rate),
            ms.nn.GELU(),  # TODO consider dropout or normalization here.
            ms.nn.Dense(bottleneck_dim, encoder_hidden_dim * num_repeat_tokens),
        )
        if encoder_dropout_disabled:
            disable_dropout(self.encoder_decoder.encoder)
        if decoder_dropout_disabled:
            disable_dropout(self.encoder_decoder.decoder)
            disable_dropout(self.encoder_decoder.lm_head)
        ######################################################
        self.tokenizer = tokenizer
        self.embedder = embedder
        if self.embedder_no_grad:
            for param in self.embedder.parameters():
                param.requires_grad = False

            self.embedder.eval()

        self.embedder_tokenizer = embedder_tokenizer
        self.embedder_model_api = embedder_model_api
        # self.freeze(freeze_strategy=config.freeze_strategy)
        self.embedder_fake_with_zeros = embedder_fake_with_zeros

        self.embedding_transform_strategy = "repeat"  # "none" # "repeat"
        self.embeddings_from_layer_n = embeddings_from_layer_n
        self.noise_level = 0

    def _freeze_encoder(self):
        freeze_params(self.encoder_decoder.encoder)

    def _freeze_decoder(self):
        # github.com/huggingface/transformers/blob/master/src/transformers/models/t5/modeling_t5.py#L1229-L1231
        freeze_params(self.encoder_decoder.decoder)
        freeze_params(self.encoder_decoder.lm_head)

    def freeze(self, freeze_strategy: str):
        '''maybe freeze module of encoder_decoder for subsequent training'''

        assert freeze_strategy in FREEZE_STRATEGIES

        if freeze_strategy == "decoder":
            self._freeze_decoder()
        elif freeze_strategy == "encoder":
            self._freeze_encoder()
        elif freeze_strategy == "encoder_and_decoder":
            self._freeze_encoder()
            self._freeze_decoder()
            # in this case, freeze embeddings too
            freeze_params(self.encoder_decoder.shared)
        elif freeze_strategy == "none":
            pass
        else:
            raise ValueError(f"invalid freezing strategy {freeze_strategy}")

    def _process_embedder_output(self, outputs, attention_mask: ms.Tensor):

        '''process_embedder_output'''

        if hasattr(outputs, "pooler_output") and (outputs.pooler_output is not None):
            return outputs.pooler_output
        if self.embeddings_from_layer_n is not None:
            assert hasattr(
                outputs, "hidden_states"
            ), "output missing hidden states - remember to initialize the model with output_hidden_states=True?"
            hidden_state = outputs.hidden_states[self.embeddings_from_layer_n]
            embeddings = mean_pool(hidden_state, attention_mask)
        else:
            hidden_state = outputs.last_hidden_state
            embeddings = mean_pool(hidden_state, attention_mask)
        return embeddings

    def call_embedding_model(self, input_ids: ms.Tensor, attention_mask: ms.Tensor,
                             token_type_ids: Optional[ms.Tensor] = None):
        '''
        call_embedding_model
        '''
        embedder = self.embedder
        # print("** call_embedding_model")
        if self.embedder_no_grad:
            embedder.eval()
        # pylint: disable=R1705
        if self.embedder_fake_with_zeros:
            batch_size = input_ids.shape[0]
            return ops.zeros(
                (batch_size, self.embedder_dim),
                dtype=ms.float32
            )

        elif isinstance(self.embedder, RobertaModel):
            #before : mindnlp.transformers.models.t5.modeling_t5.T5ForConditionalGeneration
            #self.embedder : RobertaModel
            # sentence-transformers is kind of really annoying
            model_inputs = {"input_ids": input_ids, "attention_mask": attention_mask}

            if token_type_ids is not None:
                model_inputs["token_type_ids"] = token_type_ids
            # print(model_inputs)
            # print(model_inputs['input_ids'].shape)


            model_output = embedder(model_inputs['input_ids'])

            embeddings = self._process_embedder_output(model_output, attention_mask)

        else:
            model_output = embedder(input_ids=input_ids, attention_mask=attention_mask)
            embeddings = self._process_embedder_output(model_output, attention_mask)

        if self.noise_level > 0:
            embeddings += self.noise_level * ops.randn(
                embeddings.shape
            )
        return embeddings

    def embed_and_project(self, embedder_input_ids: Optional[ms.Tensor],
                          embedder_attention_mask,
                          frozen_embeddings: Optional[ms.Tensor] = None,):
        '''
        embed_and_project
        '''
        assert not ((embedder_input_ids is None) and (frozen_embeddings is None))
        if frozen_embeddings is not None:
            embeddings = frozen_embeddings
            assert len(embeddings.shape) == 2  # batch by d
        elif self.embedder_no_grad:
            embeddings = self.call_embedding_model(input_ids=embedder_input_ids,
                                                   attention_mask=embedder_attention_mask,)
        else:
            embeddings = self.call_embedding_model(
                input_ids=embedder_input_ids,
                attention_mask=embedder_attention_mask,
            )
        if self.embedding_transform_strategy == "repeat":
            if embeddings.dtype != self.dtype:
                embeddings = embeddings.astype(self.dtype)
            repeated_embeddings = self.embedding_transform(embeddings)
            # linear outputs a big embedding, reshape into a sequence of regular size embeddings.
            embeddings = repeated_embeddings.reshape(
                (*repeated_embeddings.shape[:-1], self.num_repeat_tokens, -1)
            )
        elif self.embedding_transform_strategy == "nearest_neighbors":
            # TODO
            raise NotImplementedError()
        else:
            raise ValueError(
                f"unknown embedding transformation strategy {self.embedding_transform_strategy}"
            )
        attention_mask = ops.ones(
            (embeddings.shape[0], embeddings.shape[1]), dtype=ms.float32)
        return embeddings, attention_mask

    def generate(self, inputs: Dict[str, ms.Tensor], generation_kwargs: Dict[str, ms.Tensor],):
        '''
        generate embedding
        '''
        generation_kwargs = copy.copy(generation_kwargs)  # make a copy so we can edit
        inputs_embeds, attention_mask = self.embed_and_project(
            embedder_input_ids=inputs.get("embedder_input_ids"),
            embedder_attention_mask=inputs.get("embedder_attention_mask"),
            # frozen_embeddings=inputs.get("frozen_embeddings"),
            # embedder_input_ids=inputs[4],
            # embedder_attention_mask=inputs[5],

        )

        if "decoder_input_ids" in inputs:
            return self.encoder_decoder.generate(
                # required: input embeddings
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                # optional: input IDs (for starting generation).
                # typically not set unless generating prefixes for
                # reranking.
                decoder_input_ids=inputs["decoder_input_ids"],
                **generation_kwargs,
            )

        return self.encoder_decoder.generate(
            # required: input embeddings
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            # optional: input IDs (for starting generation).
            # typically not set unless generating prefixes for
            # reranking.
            **generation_kwargs,
        )


    def generate_corrector(self, inputs: Dict[str, ms.Tensor], generation_kwargs: Dict[str, ms.Tensor],):
        '''
        因为数据格式不一样，所以corrector中的generate改成这个名字了generate_corrector
        '''
        generation_kwargs = copy.copy(generation_kwargs)  # make a copy so we can edit
        print(inputs)
        inputs_embeds, attention_mask = self.embed_and_project(
            embedder_input_ids=inputs.get("embedder_input_ids"),
            embedder_attention_mask=inputs.get("embedder_attention_mask"),
            frozen_embeddings=inputs.get("frozen_embeddings"),
        )

        return self.encoder_decoder.generate(inputs_embeds=inputs_embeds,
                                             attention_mask=attention_mask, **generation_kwargs,)


    def forward(self, embedder_input_ids: ms.Tensor, embedder_attention_mask: ms.Tensor,
                labels: Optional[ms.Tensor] = None,
                frozen_embeddings: Optional[ms.Tensor] = None, decoder_input_ids: Optional[ms.Tensor] = None,):
        '''
        forward function
        '''
        # Unused: input_ids, attention_mask
        inputs_embeds, attention_mask = self.embed_and_project(
            embedder_input_ids=embedder_input_ids,
            embedder_attention_mask=embedder_attention_mask,
            frozen_embeddings=frozen_embeddings,
        )
        return self.encoder_decoder(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=labels,
            decoder_input_ids=decoder_input_ids,
        )
