from typing import Dict, List, Optional
import numpy as np

try:
    from mindformers import AutoTokenizer, AutoModel
    import mindspore as ms

    MINDFORMERS_AVAILABLE = True
except Exception:
    MINDFORMERS_AVAILABLE = False


class MindFormersLlamaAdapter:
    """
    本地 LLaMA (MindFormers) 适配器
    - 提供 full logits for next token
    - 提供基于 logits + bias 的模拟 top-k logprobs
    使用前请确保模型与分词器可用（名称需与你的权重一致）
    """

    def __init__(self, model_name: str = "llama_7b", device_target: str = "Ascend"):
        if not MINDFORMERS_AVAILABLE:
            raise ImportError("mindformers / mindspore 未安装或不可用。")
        ms.set_context(mode=ms.GRAPH_MODE, device_target=device_target)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name, dtype=ms.float16)
        self._hidden_size = getattr(getattr(self.model, "config", None), "hidden_size", None)

    @property
    def vocab_size(self) -> int:
        return self.tokenizer.vocab_size

    @property
    def hidden_size(self) -> Optional[int]:
        return self._hidden_size

    def encode(self, text: str) -> List[int]:
        return self.tokenizer(text)["input_ids"]

    def batch_encode(self, texts: List[str]) -> List[List[int]]:
        return [self.encode(t) for t in texts]

    def decode(self, token_ids: List[int]) -> str:
        return self.tokenizer.decode(token_ids)

    def batch_next_token_logits(
        self, batch_input_ids: List[List[int]], pad_token_id: int = 0
    ) -> np.ndarray:
        """
        批量获取下一 token 的 logits
        batch_input_ids: List[List[int]]，批量的 token id 序列
        返回: np.ndarray，形状 [batch_size, vocab_size]
        """
        max_len = max(len(ids) for ids in batch_input_ids)
        padded_ids = [ids + [pad_token_id] * (max_len - len(ids)) for ids in batch_input_ids]
        input_ids_ms = ms.Tensor(padded_ids, dtype=ms.int64)
        outputs = self.model.generate(
            input_ids=input_ids_ms,
            max_length=max_len + 1,
            return_dict_in_generate=True,
            output_scores=True,
        )
        step_logits = outputs.scores[0]  # shape: [batch_size, vocab_size]
        return step_logits.astype(np.float64)

    def next_token_logits(self, input_ids: List[int]) -> np.ndarray:
        """
        获取单条输入的下一 token logits
        """
        logits_batch = self.batch_next_token_logits(
            [input_ids], pad_token_id=self.tokenizer.pad_token_id
        )
        return logits_batch[0]

    def get_W_true(self) -> Optional[np.ndarray]:
        """
        获取输出头权重（通常与嵌入权重 tied）
        """
        lm_head = getattr(self.model, "lm_head", None)
        if lm_head is None or not hasattr(lm_head, "weight"):
            return None
        W = lm_head.weight.asnumpy()
        return W
