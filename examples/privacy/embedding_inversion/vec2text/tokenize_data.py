'''
tokenize data
'''

from dataclasses import dataclass
from typing import Callable, Dict, Optional, Union, Any

import numpy as np
import mindspore as ms
import mindspore.numpy as mnp
from mindnlp.transformers import PreTrainedTokenizerBase
from mindnlp.utils import PaddingStrategy
from mindnlp.transformers.tokenization_utils_fast import PreTrainedTokenizer
from mindnlp import transformers

from models import InversionModel



# pylint disable: C0330
def tokenize_function(tokenizer: PreTrainedTokenizer, embedder_tokenizer: PreTrainedTokenizer,
                      max_seq_length: int, padding: bool = False,) -> Callable[[Dict], Dict]:
    '''
    tokenize_function
    '''
    def tokenize_function_inner(examples) -> Dict[str, ms.Tensor]:
        try:
            texts = examples
            output = tokenizer(
                texts,
                padding=padding,
                truncation=True,
                max_length=max_seq_length,
                return_tensors='np'
            )

            output['input_ids'] = output['input_ids'][0]
            output['attention_mask'] = output['attention_mask'][0]
            output_labels_list = [
                (-100 if token_id == tokenizer.pad_token_id else token_id) for token_id in output["input_ids"]
            ]
            output["labels"] = np.array(output_labels_list)

            # 计算有效长度并生成 length 数组
            count_of_ones = sum(output["attention_mask"])
            output["length"] = np.array([count_of_ones])
            # mask = output["input_ids"] == tokenizer.pad_token_id
            # labels = mnp.where(mask, ms.tensor(-100), output["input_ids"])
            # output["labels"] = labels

            # count_of_ones = mnp.sum(output["attention_mask"])
            # output["length"] = count_of_ones
            # print(count_of_ones.asnumpy().item())



            embedder_output = embedder_tokenizer(
                texts,
                padding="max_length",
                truncation=True,
                max_length=max_seq_length,
                return_tensors="np",
            )
            embedder_output['input_ids'] = embedder_output['input_ids'][0]
            embedder_output['attention_mask'] = embedder_output['attention_mask'][0]



            # embedder_output = {f"embedder_{key}": value.asnumpy().tolist() for key, value in embedder_output.items()}
            embedder_output = {f"embedder_{key}": value for key, value in embedder_output.items()}
            # print("--------------------------------------------------------------------")
            # print({**output, **embedder_output})
            return {**output, **embedder_output}
        except Exception as e:
            print(f"Error during processing: {e}")
            raise  # Re-throw the exception after logging

    return tokenize_function_inner


def tokenize_function_(tokenizer: PreTrainedTokenizer, embedder_tokenizer: PreTrainedTokenizer,
                       max_seq_length: int, padding: bool = False,) -> Callable[[Dict], Dict]:
    '''tokenize_function'''
    def tokenize_function_inner(examples) -> Dict[str, ms.Tensor]:
        try:
            texts = examples

            output = tokenizer(
                texts,
                padding=padding,
                truncation=True,
                max_length=max_seq_length,
                return_tensors='ms'
            )

            # print("output的值是：")
            # print(output)
            # print("------------end---------------")
            output['input_ids'] = output['input_ids'][0]
            output['attention_mask'] = output['attention_mask'][0]
            output_labels_list = [
                (-100 if token_id == tokenizer.pad_token_id else token_id) for token_id in output["input_ids"]
            ]
            output["labels"] = np.array(output_labels_list)

            # 计算有效长度并生成 length 数组
            # count_of_ones = sum(output["attention_mask"])
            # output["length"] = np.array([count_of_ones])
            mask = output["input_ids"] == tokenizer.pad_token_id
            labels = mnp.where(mask, ms.tensor(-100), output["input_ids"])
            output["labels"] = labels

            count_of_ones = mnp.sum(output["attention_mask"])
            output["length"] = [count_of_ones.asnumpy().item()]
            # print("++++++++++++++++++++++++++++++++++++++++")
            # print(count_of_ones.asnumpy().item())
            # print("-----------------------------------------")


            embedder_output = embedder_tokenizer(
                texts,
                padding="max_length",
                truncation=True,
                max_length=max_seq_length,
                return_tensors="ms",
            )
            embedder_output['input_ids'] = embedder_output['input_ids'][0]
            embedder_output['attention_mask'] = embedder_output['attention_mask'][0]



            # embedder_output = {f"embedder_{key}": value.asnumpy().tolist() for key, value in embedder_output.items()}
            embedder_output = {f"embedder_{key}": value for key, value in embedder_output.items()}
            # print("--------------------------------------------------------------------")
            # print({**output, **embedder_output})
            return {**output, **embedder_output}
        except Exception as e:
            print(f"Error during processing: {e}")
            raise  # Re-throw the exception after logging

    return tokenize_function_inner


def embed_dataset_batch(model: InversionModel, batch: Dict) -> Dict:
    '''
    embed_dataset_batch
    '''
    assert "input_ids" in batch.keys(), f"invalid keys {batch.keys()}"
    assert hasattr(model, "call_embedding_model")

    input_ids = batch["input_ids"]
    inputs_str = model.tokenizer.batch_decode(input_ids, skip_special_tokens=True)
    emb_input_ids = model.embedder_tokenizer(
        inputs_str,
        max_length=model.config.max_seq_length,
        truncation=True,
        padding="max_length",
        return_tensors="ms",
    )

    model.set_train(False)
    batch["frozen_embeddings"] = model.call_embedding_model(**emb_input_ids)
    model.set_train(True)
    return batch

# pylint: disable=W0613
def get_tokenizer_mapping(lm: str, inverter: str, inverter_vocab_size: int) -> ms.Tensor:
    """Computes the mapping from token outputs in `lm`'s vocabulary to those in `inverter's
    vocabulary. Makes some assumptions about spacing.
    """
    lm_tokenizer = transformers.AutoTokenizer.from_pretrained(lm)
    inverter_tokenizer = transformers.AutoTokenizer.from_pretrained(inverter)

    lm_vocab = lm_tokenizer.vocab
    mapping = ms.ops.zeros(len(lm_vocab), dtype=ms.int64)
    for k, idx in lm_tokenizer.vocab.items():
        # We replace space tokens with nothing and allow the call to
        # inverter_tokenizer.decode to determine this. We also
        # filter out 2 and 3 as first tokens which are extremely common
        # when the T5 tokenizer processes unicode. (These are hacks
        # specific to the LLAMA-T5 lm-inverter pairing, and it would
        # be better to find an automated wa to do this later.)
        mapping[idx] = inverter_tokenizer.encode(k.replace("▁", " "))[0]
        if mapping[idx] in [2, 3]:
            mapping[idx] = inverter_tokenizer.encode(k.replace("▁", " "))[1]

    preservation = len(set(mapping.tolist())) / len(lm_vocab)
    print(
        f"Mapped tokenizer {lm} to {inverter}. Preserved {preservation*100:.1f}% of unique tokens."
    )
    return mapping

def pad_without_fast_tokenizer_warning(tokenizer, *pad_args, **pad_kwargs):
    """
    Pads without triggering the warning about how using the pad function is sub-optimal when using a fast tokenizer.
    """

    # To avoid errors when using Feature extractors
    if not hasattr(tokenizer, "deprecation_warnings"):
        return tokenizer.pad(*pad_args, **pad_kwargs)

    # Save the state of the warning, then disable it
    warning_state = tokenizer.deprecation_warnings.get("Asking-to-pad-a-fast-tokenizer", False)
    tokenizer.deprecation_warnings["Asking-to-pad-a-fast-tokenizer"] = True

    try:
        padded = tokenizer.pad(*pad_args, **pad_kwargs)
    finally:
        # Restore the state of the warning.
        tokenizer.deprecation_warnings["Asking-to-pad-a-fast-tokenizer"] = warning_state

    return padded


# convert to current assignment without too much change from transformer library of huggingface
# lizard: ignore=CYCLOMATIC_COMPLEXITY
@dataclass
class DataCollatorForSeq2Seq:
    """
    Data collator that will dynamically pad the inputs received, as well as the labels.

    Args:
        tokenizer ([`PreTrainedTokenizer`] or [`PreTrainedTokenizerFast`]):
            The tokenizer used for encoding the data.
        model ([`PreTrainedModel`], *optional*):
            The model that is being trained. If set and has the *prepare_decoder_input_ids_from_labels*, use it to
            prepare the *decoder_input_ids*

            This is useful when using *label_smoothing* to avoid calculating loss twice.
        padding (`bool`, `str` or [`~utils.PaddingStrategy`], *optional*, defaults to `True`):
            Select a strategy to pad the returned sequences (according to the model's padding side and padding index)
            among:

            - `True` or `'longest'` (default): Pad to the longest sequence in the batch (or no padding if only a single
              sequence is provided).
            - `'max_length'`: Pad to a maximum length specified with the argument `max_length` or to the maximum
              acceptable input length for the model if that argument is not provided.
            - `False` or `'do_not_pad'`: No padding (i.e., can output a batch with sequences of different lengths).
        max_length (`int`, *optional*):
            Maximum length of the returned list and optionally padding length (see above).
        pad_to_multiple_of (`int`, *optional*):
            If set will pad the sequence to a multiple of the provided value.

            This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability >=
            7.5 (Volta).
        label_pad_token_id (`int`, *optional*, defaults to -100):
            The id to use when padding the labels (-100 will be automatically ignored by PyTorch loss functions).
        return_tensors (`str`, *optional*, defaults to `"pt"`):
            The type of Tensor to return. Allowable values are "np", "pt" and "tf".
    """

    tokenizer: PreTrainedTokenizerBase
    model: Optional[Any] = None
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    label_pad_token_id: int = -100
    return_tensors: str = "ms"

    def __call__(self, features, return_tensors=None):
        '''call func, reconstruct to form a func to meet with CCN restriction'''
        # 确定返回的 tensor 类型
        return_tensors = return_tensors or self.return_tensors

        # 获取 labels 键名
        label_name = self._get_label_name(features)

        # 提取 labels 和非 labels 特征
        labels, non_labels_features = self._extract_labels_and_features(features, label_name)

        # 使用 tokenizer 对非标签特征进行处理
        batch = self._process_features(non_labels_features, return_tensors)

        # 手动填充 labels
        if labels is not None:
            batch["labels"] = self._process_labels(labels, features, label_name)

        # 处理返回 tensor 类型
        batch = self._convert_labels_to_tensor(batch, return_tensors)

        # 准备 decoder_input_ids
        if self._requires_decoder_input_ids(labels):
            batch["decoder_input_ids"] = self.model.prepare_decoder_input_ids_from_labels(labels=batch["labels"])

        return batch

    def _get_label_name(self, features):
        """获取标签名称，如果有 'label' 则使用 'label' 否则使用 'labels'"""
        return "label" if "label" in features[0].keys() else "labels"

    def _extract_labels_and_features(self, features, label_name):
        """提取标签和非标签特征"""
        labels = [feature[label_name] for feature in features] if label_name in features[0].keys() else None
        # 将 [None] 转换为 None
        if labels and all(label is None for label in labels):
            labels = None
        non_labels_features = [{k: v for k, v in feature.items() if k != label_name} for feature in features]
        return labels, non_labels_features

    def _process_features(self, non_labels_features, return_tensors):
        """使用 tokenizer 处理特征"""
        return pad_without_fast_tokenizer_warning(
            self.tokenizer,
            non_labels_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=return_tensors,
        )

    def _process_labels(self, labels, features, label_name):
        """手动填充 labels"""
        no_padding = self.padding is False or self.padding == PaddingStrategy.DO_NOT_PAD
        if no_padding:
            return self._handle_no_padding(labels, features, label_name)

        return self._handle_padding(labels)

    def _handle_no_padding(self, labels, features, label_name):
        """处理没有填充的标签"""
        if isinstance(features[0][label_name], list):
            return list(labels)

        return [np.concatenate([label, []]) for label in labels]

    def _handle_padding(self, labels):
        """处理需要填充的标签"""
        max_label_length = self._get_max_label_length(labels)
        return [
            self._pad_label(label, max_label_length) for label in labels
        ]

    def _get_max_label_length(self, labels):
        """获取最大标签长度"""
        max_padding = self.padding == PaddingStrategy.MAX_LENGTH and self.max_length is not None
        if max_padding:
            return self.max_length
        return max(len(l) for l in labels)

    def _pad_label(self, label, max_label_length):
        """对标签进行填充"""
        padding_side = self.tokenizer.padding_side
        pad_length = max_label_length - len(label)
        padding = [self.label_pad_token_id] * pad_length

        if padding_side == "right":
            return label + padding

        return padding + label

    def _convert_labels_to_tensor(self, batch, return_tensors):
        """根据指定的返回类型转换 labels 为 tensor"""
        if batch.get("labels") is not None:
            if return_tensors == "pt":
                import torch
                batch["labels"] = torch.tensor(batch["labels"], dtype=torch.int64)
            elif return_tensors == "tf":
                import tensorflow as tf
                batch["labels"] = tf.constant(batch["labels"], dtype=tf.int64)
            else:
                batch["labels"] = ms.tensor(batch["labels"], dtype=ms.int64)
        else:
            batch["labels"] = None
        return batch

    def _requires_decoder_input_ids(self, labels):
        """检查是否需要生成 decoder_input_ids"""
        return (
            labels is not None and
            self.model is not None and
            hasattr(self.model, "prepare_decoder_input_ids_from_labels")
        )
