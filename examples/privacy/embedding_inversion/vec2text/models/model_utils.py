'''
model utils for training
'''

from typing import Any, Dict

import mindspore as ms
from mindnlp.sentence import SentenceTransformer
from mindnlp.transformers import AutoModel, AutoTokenizer, AutoModelForCausalLM, AutoModelForSeq2SeqLM, \
    PreTrainedTokenizer

EMBEDDER_MODEL_NAMES = [
    "bert",
    "bert__random_init",
    "contriever",
    "dpr",
    "gtr_base",
    "gtr_base__random_init",
    "medicalai/ClinicalBERT",
    "gtr_large",
    "ance_tele",
    "dpr_st",
    "gtr_base_st",
    "paraphrase-distilroberta",
    "sentence-transformers/all-MiniLM-L6-v2",
    "meta-llama/Llama-2-7b-hf",
    "meta-llama/Llama-2-13b-hf",
    "meta-llama/Llama-2-7b-chat-hf",
    "meta-llama/Llama-2-13b-chat-hf",
    "nomic-ai/nomic-embed-text-v1",
    "gpt2",
    "gpt2-medium",
    "gpt2-large",
    "gpt2-xl",
]


FREEZE_STRATEGIES = ["decoder", "encoder_and_decoder", "encoder", "none"]
EMBEDDING_TRANSFORM_STRATEGIES = ["repeat"]


device = ms.get_context("device_target")


def disable_dropout(model: ms.nn.Cell):
    dropout_modules = [m for m in model.modules() if isinstance(m, ms.Dropout)]
    for m in dropout_modules:
        m.p = 0.0
    print(
        f"Disabled {len(dropout_modules)} dropout modules from model type {type(model)}"
    )


def freeze_params(model: ms.nn.Cell):
    total_num_params = 0
    for _, params in model.named_parameters():
        params.requires_grad = False
        total_num_params += params.numel()
    # print(f"Froze {total_num_params} params from model type {type(model)}")


def mean_pool(hidden_states: ms.Tensor, attention_mask: ms.Tensor):
    b, _, d = hidden_states.shape
    unmasked_outputs = hidden_states * attention_mask[..., None]
    pooled_outputs = unmasked_outputs.sum(axis=1) / attention_mask.sum(axis=1)[:, None]
    assert pooled_outputs.shape == (b, d)
    return pooled_outputs


def max_pool(hidden_states: ms.Tensor, attention_mask: ms.Tensor) -> ms.Tensor:
    b, _, d = hidden_states.shape
    unmasked_outputs = hidden_states * attention_mask[..., None]
    pooled_outputs = unmasked_outputs.max(axis=1)
    assert pooled_outputs.shape == (b, d)
    return pooled_outputs


def stack_pool(hidden_states: ms.Tensor, attention_mask: ms.Tensor):
    b, s, d = hidden_states.shape
    unmasked_outputs = hidden_states * attention_mask[..., None]
    pooled_outputs = unmasked_outputs.reshape((b, s * d))  # stack along seq length
    assert pooled_outputs.shape == (b, s * d)
    return pooled_outputs


def load_embedder_and_tokenizer(name: str, torch_dtype: str):# pylint: disable=W0613

    '''
        TODO make abstract/argparse for it etc.
        name = "gpt2" #### <--- TEMP. For debugging. Delete!
    '''
    model_kwargs = {
        #"low_cpu_mem_usage": True,  # Not compatible with DeepSpeed
        "output_hidden_states": False,
    }

    if name == "gtr_base":
        print("gtr-t5-base is regarded as embedder model......")
        model = AutoModel.from_pretrained(
            "sentence-transformers/gtr-t5-base", **model_kwargs
        ).encoder
        tokenizer = AutoTokenizer.from_pretrained(
            "sentence-transformers/gtr-t5-base"
        )
    elif name == "paraphrase-distilroberta":
        model = AutoModel.from_pretrained(
            "sentence-transformers/paraphrase-distilroberta-base-v1", **model_kwargs
        )
        tokenizer = AutoTokenizer.from_pretrained(
            "sentence-transformers/paraphrase-distilroberta-base-v1"
        )
    # elif name == "paraphrase-distilroberta":
    #     tokenizer = AutoTokenizer.from_pretrained("google-t5/t5-base")
    #     model = AutoModelForSeq2SeqLM.from_pretrained("google-t5/t5-base")
    elif name == "medicalai/ClinicalBERT":
        model = AutoModel.from_pretrained(
            "medicalai/ClinicalBERT", **model_kwargs
        )
        tokenizer = AutoTokenizer.from_pretrained("medicalai/ClinicalBERT")
    elif name.startswith("gpt2"):
        model = AutoModelForCausalLM.from_pretrained(
            name,
            **model_kwargs,
        )
        # model.to_bettertransformer()
        tokenizer = AutoTokenizer.from_pretrained(name)
        tokenizer.pad_token = tokenizer.eos_token

    elif name.startswith("sentence-transformers/"):
        model = SentenceTransformer(name)
        tokenizer = model.tokenizer

    else:
        print(f"WARNING: Trying to initialize from unknown embedder {name}")
        model = AutoModel.from_pretrained(name, **model_kwargs)
        tokenizer = AutoTokenizer.from_pretrained(name)

    # model = torch.compile(model)
    return model, tokenizer

# pylint: disable=W0613
def load_encoder_decoder(model_name: str, lora: bool = False):
    model_kwargs: Dict[str, Any] = {
        #"low_cpu_mem_usage": True,z
    }
    return AutoModelForSeq2SeqLM.from_pretrained(
        model_name, **model_kwargs
    )


def load_tokenizer(name: str, max_length: int) -> PreTrainedTokenizer:
    '''
    load tokenizer
    '''
    tokenizer = AutoTokenizer.from_pretrained(
        name,
        padding="max_length",
        truncation="max_length",
        max_length=max_length,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Disable super annoying warning:
    # https://github.com/huggingface/transformers/issues/22638
    tokenizer.deprecation_warnings["Asking-to-pad-a-fast-tokenizer"] = True
    return tokenizer
