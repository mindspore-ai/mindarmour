'''
utiliaztions for training
'''

import multiprocessing
import os
from typing import Callable

import tqdm
import datasets
import mindspore as ms
from mindnlp.transformers import AutoTokenizer

datasets.disable_caching()

def emb(model: ms.nn.Cell, input_ids: ms.Tensor, attention_mask: ms.Tensor) -> ms.Tensor:
    model.set_train(False)
    embedding = model.call_embedding_model(
        input_ids=input_ids, attention_mask=attention_mask
    )
    model.set_train(True)
    return embedding

def get_world_size() -> int:
    try:
        return os.environ.get("WORLD_SIZE", 1)
    except (RuntimeError, ValueError):
        return 1


def get_num_proc() -> int:
    world_size: int = get_world_size()
    try:
        # os.sched_getaffinity respects schedulers, unlike cpu_count(), but it's only available
        # on some Unix platforms, so we support both!
        return len(os.sched_getaffinity(0)) // world_size  # type: ignore[attr-defined]
    except AttributeError:
        return multiprocessing.cpu_count() // world_size

#pylint: disable=C0103
def embed_all_tokens(model: ms.nn.Cell, tokenizer: AutoTokenizer):
    """Generates embeddings for all tokens in tokenizer vocab."""
    i = 0
    model.embedder.eval()
    batch_size = 1024
    all_token_embeddings = []
    v = tokenizer.vocab_size
    #
    # DPR has CLS and SEP.
    # GTR has no CLS or start token at all, and has EOS at the end.
    CLS = tokenizer.cls_token_id
    SEP = (tokenizer.sep_token_id) or (tokenizer.eos_token_id)
    assert SEP is not None
    #
    # device = next(model.parameters()).device
    pbar = tqdm.tqdm(
        desc="generating token embeddings", colour="#008080", total=v, leave=False
    )
    while i < v:
        #
        minibatch_size = min(v - i, batch_size)
        inputs = ms.arange(i, min(i + minibatch_size, v))
        #
        if CLS is not None:
            input_ids = ms.stack(
                [
                    ms.tensor([CLS]).repeat(len(inputs)),
                    inputs,
                    ms.tensor([SEP]).repeat(len(inputs)),
                ]
            ).T
        else:
            input_ids = ms.stack([inputs, ms.tensor([SEP]).repeat(len(inputs))]).T
        # input_ids = input_ids.to(device)
        #
        attention_mask = ms.ones_like(input_ids)
        #
        model.set_train(False)
        token_embeddings = emb(model, input_ids, attention_mask)
        model.set_train(True)
        all_token_embeddings.extend(token_embeddings)
        i += batch_size
        pbar.update(batch_size)
    #
    all_token_embeddings_tensor: ms.Tensor = ms.stack(all_token_embeddings)
    assert all_token_embeddings_tensor.shape == (tokenizer.vocab_size, 768)

    all_token_embeddings_tensor /= all_token_embeddings_tensor.norm(
        p=2, dim=1, keepdim=True
    )
    return all_token_embeddings_tensor


def convert_to_tensor(data):
    return ms.Tensor(data, ms.int64)
def add_index(data, idx):
    #
    data["idx"] = idx

    return data

def dataset_map_single_worker(dataset, map_fn: Callable, *args, **kwargs) -> datasets.Dataset:
    # kwargs["num_proc"] = kwargs.get("num_proc", 1)

    das = dataset.map(map_fn, *args, **kwargs)
    return das

manifest_object = None
