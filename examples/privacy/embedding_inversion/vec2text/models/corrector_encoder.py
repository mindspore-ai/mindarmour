'''
model to inverse embedding with the inversion model
'''

import copy
from typing import Dict, Optional

import mindspore as ms
import mindspore.ops as ops
from mindnlp.transformers import PreTrainedModel, AutoModelForSeq2SeqLM

from .config import InversionConfig


class CorrectorEncoderModel(PreTrainedModel):
    """Embeds text and concats with a provided embedding.

    TODO improve comment here.
    """

    config_class = InversionConfig
    encoder_decoder: PreTrainedModel

    def __init__(self, config: InversionConfig,):
        super().__init__(config=config)
        if config.embedder_model_api:
            embedder_dim = 1536
        else:
            embedder_dim = 768
        bottleneck_dim = embedder_dim

        num_repeat_tokens = config.num_repeat_tokens
        ignore_hypothesis_embedding = config.corrector_ignore_hypothesis_embedding
        self.use_ff_dropout = False

        encoder_decoder = AutoModelForSeq2SeqLM.from_pretrained(
            config.model_name_or_path
        )
        self.encoder_decoder = encoder_decoder  # .to_bettertransformer()
        self.embedder_dim = embedder_dim
        self.num_repeat_tokens = num_repeat_tokens
        self.encoder_hidden_dim = self.encoder_decoder.config.hidden_size
        self.embedding_transform_1 = ms.nn.SequentialCell(
            ms.nn.Dense(self.embedder_dim, bottleneck_dim),
            # ms.nn.Dropout(
            #     self.encoder_decoder.config.dropout_rate if self.use_ff_dropout else 0.0
            # ),
            ms.nn.GELU(),
            ms.nn.Dense(bottleneck_dim, self.encoder_hidden_dim * num_repeat_tokens),
        )
        self.embedding_transform_2 = ms.nn.SequentialCell(
            ms.nn.Dense(self.embedder_dim, bottleneck_dim),
            # ms.nn.Dropout(
            #     self.encoder_decoder.config.dropout_rate if self.use_ff_dropout else 0.0
            # ),
            ms.nn.GELU(),
            ms.nn.Dense(bottleneck_dim, self.encoder_hidden_dim * num_repeat_tokens),
        )
        self.embedding_transform_3 = ms.nn.SequentialCell(
            ms.nn.Dense(self.embedder_dim, bottleneck_dim),
            # ms.nn.Dropout(
            #     self.encoder_decoder.config.dropout_rate if self.use_ff_dropout else 0.0
            # ),
            ms.nn.GELU(),
            ms.nn.Dense(bottleneck_dim, self.encoder_hidden_dim * num_repeat_tokens),
        )
        self.ignore_hypothesis_embedding = ignore_hypothesis_embedding
        # TODO argparse; default to 0?
        self.training_embedding_noise_level = 0
        # self.training_embedding_noise_level = 1e-5  # adding for openai...
        self.use_ln = True
        if self.use_ln:
            self.layernorm = ms.nn.LayerNorm([self.encoder_hidden_dim])
        # print(f"Corrector encoder noise level {self.training_embedding_noise_level}")

    def get_encoder_embedding(self, embedding: ms.Tensor, hypothesis_embedding: ms.Tensor,
                              hypothesis_input_ids: ms.Tensor, hypothesis_attention_mask: ms.Tensor):
        '''
        get encoder embedding
        '''

        batch_size, _ = embedding.shape
        assert embedding.shape == (batch_size, self.embedder_dim)
        assert hypothesis_embedding.shape == (batch_size, self.embedder_dim)

        if (self.training) and (self.training_embedding_noise_level > 0):
            embedding += self.training_embedding_noise_level * ops.randn(
                embedding.shape
            )
            hypothesis_embedding += self.training_embedding_noise_levelA * ops.randn(
                hypothesis_embedding.shape
            )

        if self.ignore_hypothesis_embedding:
            # For "No Feedback" ablation
            hypothesis_embedding = embedding

        diff_embedding = embedding - hypothesis_embedding

        embedding = self.embedding_transform_1(embedding)
        embedding = embedding.reshape(
            (batch_size, self.num_repeat_tokens, self.encoder_hidden_dim)
        )
        #
        diff_embedding = self.embedding_transform_2(diff_embedding)
        diff_embedding = diff_embedding.reshape(
            (batch_size, self.num_repeat_tokens, self.encoder_hidden_dim)
        )
        #
        hypothesis_embedding = self.embedding_transform_3(hypothesis_embedding)
        hypothesis_embedding = hypothesis_embedding.reshape(
            (batch_size, self.num_repeat_tokens, self.encoder_hidden_dim)
        )
        inputs_embeds = self.encoder_decoder.encoder.embed_tokens(hypothesis_input_ids)
        #
        ones = ops.ones(
            (batch_size, 1), dtype=ms.bool_)
        # TODO: pad_token_id or eos_token_id? Or does it not matter?
        sep_token = ones * self.encoder_decoder.config.eos_token_id
        sep_token = self.encoder_decoder.encoder.embed_tokens(sep_token)
        # inputs_embeds = ops.cat((sep_token, embedding, sep_token, hypothesis_embedding, inputs_embeds), dim=1)
        inputs_embeds = ops.cat(
            (
                sep_token,
                embedding,
                sep_token,
                hypothesis_embedding,
                sep_token,
                diff_embedding,
                sep_token,
                inputs_embeds,
            ),
            axis=1,
        )
        if self.use_ln:
            inputs_embeds = self.layernorm(inputs_embeds)
        # attention_mask = ops.cat(
        #     (ones.tile(1, 4 + 3 * self.num_repeat_tokens), hypothesis_attention_mask),
        #     axis=1,
        # )
        ones_repeated = ones.tile((1, 4 + 3 * self.num_repeat_tokens))



        attention_mask = ops.cat(
            (ones_repeated, hypothesis_attention_mask),
            axis=1,
        )
        # ones_repeated = ops.Tile()(ms.Tensor([1], dtype=ms.float32), (1, 4 + 3 * self.num_repeat_tokens))
        # attention_mask = ops.Concat(axis=1)((ones_repeated, hypothesis_attention_mask))
        return (inputs_embeds, attention_mask)

    def generate(self, inputs: Dict[str, ms.Tensor], generation_kwargs: Dict[str, ms.Tensor],
                 return_dict_in_generate: bool = False) -> ms.Tensor:
        '''generate inversion embedding'''

        if "max_length" not in generation_kwargs:
            generation_kwargs = copy.copy(
                generation_kwargs
            )  # make a copy so we can edit
            generation_kwargs["max_length"] = inputs.get(
                "input_ids", inputs["embedder_input_ids"]
            ).shape[1]

        inputs_embeds, attention_mask = self.get_encoder_embedding(
            embedding=inputs["frozen_embeddings"],
            hypothesis_input_ids=inputs["hypothesis_input_ids"],
            hypothesis_attention_mask=inputs["hypothesis_attention_mask"],
            hypothesis_embedding=inputs["hypothesis_embedding"],
        )

        if "decoder_input_ids" in inputs:
            return self.encoder_decoder.generate(
                # required: input embeddings
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                return_dict_in_generate=return_dict_in_generate,
                output_scores=return_dict_in_generate,
                # optional: input IDs (for starting generation).
                # typically not set unless generating prefixes for
                # reranking.
                decoder_input_ids=inputs["decoder_input_ids"],
                # decoder_attention_mask=inputs["decoder_attention_mask"],
                **generation_kwargs,
            )

        return self.encoder_decoder.generate(
            # required: input embeddings
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            return_dict_in_generate=return_dict_in_generate,
            output_scores=return_dict_in_generate,
            # optional: input IDs (for starting generation).
            # typically not set unless generating prefixes for
            # reranking.
            **generation_kwargs,
        )

    def forward(self, embedding: ms.Tensor, hypothesis_embedding, hypothesis_input_ids: ms.Tensor,
                hypothesis_attention_mask: ms.Tensor, labels: Optional[ms.Tensor] = None,):
        '''
        forward function
        '''
        inputs_embeds, attention_mask = self.get_encoder_embedding(
            embedding=embedding,
            hypothesis_embedding=hypothesis_embedding,
            hypothesis_input_ids=hypothesis_input_ids,
            hypothesis_attention_mask=hypothesis_attention_mask,
        )
        return self.encoder_decoder(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=labels,
        )
