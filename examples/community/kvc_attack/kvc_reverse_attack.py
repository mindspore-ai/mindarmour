import numpy as np
import hashlib
import json
from typing import Dict, Iterable, List, Optional, Union
import hnswlib

def load_array_from_file(file_name):
    data = np.load(file_name)
    array_list = [data[key] for key in data.files]
    return array_list


ZEROS_KVC_HASH = "60cacbf3d72e1e7834203da608037b1bf83b40e8"

class KVCHashTable:
    """build hash of KVC"""
    def __init__(self):
        """ hash -> index """
        self.hash_table = {} # hash->index
        self.token_kvc_dict = {} # index->per_token_tensor
        self.total_hash_num = 0
        self.exist_hash_num = 0
        
        # 新增：存放 flatten 后的向量矩阵和 ANN 索引
        self.token_kvc_matrix = None      # shape = (N, D)
        self.ann_index = None

    def build_from_kvc(self, kvc_list: list[np.ndarray]):
        base_index = 0
        vecs = []  # 用于 ANN 建索引
        for kvc in kvc_list:
            a, b, _, _ = kvc.shape
            for i in range(a):
                for j in range(b):
                    kvc_per_token = kvc[i, j]  # shape [c, d]
                    h = hashlib.sha1(kvc_per_token.tobytes()).hexdigest()
                    global_index = base_index + i * b + j
                    
                    #  NOTE: may have the same hash for different kvc_per_token
                    self.hash_table[h] = global_index
                    self.token_kvc_dict[global_index] = kvc_per_token
                    self.total_hash_num += 1
                    vecs.append(kvc_per_token.flatten())
            base_index += a * b
        
        # 构建矩阵（float32 更快）
        self.token_kvc_matrix = np.stack(vecs).astype(np.float32)

        # 建 HNSW 索引
        dim = self.token_kvc_matrix.shape[1]
        p = hnswlib.Index(space='l2', dim=dim)
        p.init_index(max_elements=self.token_kvc_matrix.shape[0],
                     ef_construction=60, M=8)
        p.add_items(self.token_kvc_matrix, list(self.token_kvc_dict.keys()))
        p.set_ef(30)  # 查询精度与速度平衡
        self.ann_index = p
        
        print(f"build hash of KVC finished")
    
    def find_idx_of_kvc(self, kvc: np.ndarray, progress_callback=None) -> list:
        """查找 kvc 中每个 token 的 kvc 在 hash_table 中的索引"""
        a, b, c, d = kvc.shape
        indices = []
        total_token_num = a * b
        for i in range(a):
            for j in range(b):
                current_token_num = i * b + j
                kvc_per_token = kvc[i, j]
                h = hashlib.sha1(kvc_per_token.tobytes()).hexdigest()
                if h == ZEROS_KVC_HASH:
                    continue
                idx = self.hash_table.get(h)
                if idx is None:
                    print(f"not found, i: {i}, j: {j}, idx: {idx}")
                    idx, _, miscount_dict = self.try_to_match(kvc_per_token)
                    if idx is None:
                        # give a "?"
                        idx = 30
                        print(f"kvc hash not found in hash_table, index: {i * b + j}")
                    else:
                        print(f"mached idx: {idx}")
                print(f"total: {total_token_num}, current: {current_token_num + 1}, get token id: {idx}")
                if progress_callback is not None:
                    progress_callback(current_token_num, total_token_num)
                indices.append(idx)
        return indices

    def try_to_match(self, token_kvc: np.ndarray, tau = 0, k = 5, ann_topk=5):
        """
        先做容差匹配，查找与 token_kvc 最接近的，允许容差 tau 和错配上限个数 k。
        再做ANN最近邻匹配
        
        返回：
        - best: 最佳匹配的索引（错配最少，且 <= k），若无则为 None
        - candidates: 所有错配 <= k 的索引列表
        - miscount_dict: 每个候选的错配计数
        """
        token_kvc_int = token_kvc.astype(np.int32)
        miscount_dict = {}
        # ==== 第二档：容差匹配 ====
        for idx, arr in self.token_kvc_dict.items():
            arr_int = arr.astype(np.int32)
            diff = np.abs(arr_int - token_kvc_int)
            miscnt = np.count_nonzero(diff > tau)
            if miscnt <= k:
                miscount_dict[idx] = miscnt
            
        if miscount_dict:
            # 找到错配最少的那个
            best = min(miscount_dict, key=lambda x: miscount_dict[x])
            candidates = list(miscount_dict.keys())
            return best, candidates, miscount_dict
        
        # ==== 第三档：ANN 最近邻搜索 ====
        q_vec = token_kvc.flatten().astype(np.float32)
        labels, distances = self.ann_index.knn_query(q_vec, k=ann_topk)
        labels = labels[0].tolist()
        best_label = None
        best_dist = float('inf')
        for idx in labels:
            cand_vec = self.token_kvc_matrix[idx]
            dist = np.sum(np.abs(cand_vec - q_vec))  # L1 精排
            if dist < best_dist:
                best_dist = dist
                best_label = idx
        return best_label, labels, {}  # 第三档不返回容差计数


class QwenDeTokenizer:
    """
    仅基于 vocab.json 的 Qwen/GPT-2 风格字节级 BPE 解码器。
    - 输入: token_ids（int 列表）
    - 输出: 原始 UTF-8 文本
    - 不加载模型，不使用第三方 tokenizer
    """

    def __init__(
        self,
        vocab_json_path: str,
        special_token_policy: str = "keep",
        utf8_errors: str = "replace",
        known_special_tokens: Optional[Iterable[str]] = None,
    ):
        """
        :param vocab_json_path: vocab.json 路径（键为“可见化字节”的 token 字符串，值为 id）
        :param special_token_policy: 遇到特殊 token 的处理方式:
            - "keep": 原样保留到输出（以 UTF-8 文本形式拼接）
            - "skip": 跳过
            - "error": 抛出异常
        :param utf8_errors: 最终 bytes→str 的 UTF-8 解码策略（"strict" | "replace" | "ignore"...）
        :param known_special_tokens: 可显式指定特殊 token 列表（如 <|endoftext|>, <|im_start|>, <|im_end|> 等）
        """
        self.vocab_json_path = vocab_json_path
        self.special_token_policy = special_token_policy
        self.utf8_errors = utf8_errors

        # 构建 GPT-2/Qwen 的字节→可见Unicode 映射及其反向映射
        self._byte_decoder = self._build_byte_decoder()

        # 加载 id->token 字符串（可见化字节序列）
        self._id_to_token = self._load_id_to_token(vocab_json_path)

        # 特殊 token 判定
        self._known_special = set(known_special_tokens or [])
        # 常见占位（若 vocab 中包含则加入）
        self._known_special.update({
            "<|endoftext|>", "<|im_start|>", "<|im_end|>", "<|system|>",
            "<|user|>", "<|assistant|>", "<|bos|>", "<|eos|>", "<|sep|>",
        } & set(self._id_to_token))

    # -------------------------
    # 公共 API
    # -------------------------

    def de_tokenize(self, token_ids: Iterable[int]) -> str:
        """
        将 token id 序列解码为原始 UTF-8 文本。
        """
        blob = self.decode_to_bytes(token_ids)
        return blob.decode("utf-8", errors=self.utf8_errors)

    def decode_to_bytes(self, token_ids: Iterable[int]) -> bytes:
        """
        将 token id 序列解码为原始字节流（不进行 UTF-8 解码）。
        """
        out = bytearray()
        for tid in token_ids:
            tok = self._lookup_token(tid)
            if self._is_special(tok):
                self._handle_special_append(tok, out)
            else:
                out.extend(self._token_str_to_bytes(tok))
        return bytes(out)

    def id_to_token(self, token_id: int) -> str:
        """
        返回词表中的原始 token 字符串（注意：这是“可见化字节”的表示，不是原始文本）。
        """
        return self._lookup_token(token_id)

    def token_to_bytes(self, token: Union[int, str]) -> bytes:
        """
        将单个 token（id 或可见化字符串）转换为原始字节。
        """
        tok = self._lookup_token(token) if isinstance(token, int) else token
        if self._is_special(tok):
            return tok.encode("utf-8")
        return self._token_str_to_bytes(tok)

    # -------------------------
    # 内部方法
    # -------------------------

    @staticmethod
    def _bytes_to_unicode() -> Dict[int, str]:
        # 与 GPT-2 相同：将 0..255 映射为可见 Unicode 字符，避免不可见/控制字符
        bs = list(range(33, 127)) + list(range(161, 173)) + list(range(174, 256))
        cs = bs[:]
        n = 0
        for b in range(256):
            if b not in bs:
                bs.append(b)
                cs.append(256 + n)
                n += 1
        cs = [chr(c) for c in cs]
        return dict(zip(bs, cs))  # byte(int) -> unicode char(str)

    @classmethod
    def _build_byte_decoder(cls) -> Dict[str, int]:
        b2u = cls._bytes_to_unicode()
        return {u: b for b, u in b2u.items()}  # 可见字符 -> 原始单字节

    @staticmethod
    def _load_id_to_token(vocab_json_path: str) -> List[str]:
        with open(vocab_json_path, "r", encoding="utf-8") as f:
            vocab = json.load(f)  # {token_str: id}
        max_id = max(int(i) for i in vocab.values())
        id_to_tok = [""] * (max_id + 1)
        for tok_str, idx in vocab.items():
            id_to_tok[int(idx)] = tok_str
        return id_to_tok

    def _lookup_token(self, token: Union[int, str]) -> str:
        """改为越界安全返回"""
        if isinstance(token, int):
            if token < 0 or token >= len(self._id_to_token):
                return f"<UNK:{token}>"
            return self._id_to_token[token]
        return token

    def _is_special(self, tok: str) -> bool:
        # 规则：
        # 1) 显式列入 known_special
        # 2) 形如 <| ... |> 的占位
        # 3) 含有任意一个不在映射表中的字符（防御性兜底）
        if tok in self._known_special:
            return True
        if tok.startswith("<|") and tok.endswith("|>"):
            return True
        for ch in tok:
            if ch not in self._byte_decoder:
                return True
        return False

    def _handle_special_append(self, tok: str, out: bytearray) -> None:
        out.extend(tok.encode("utf-8", errors="replace"))

    def _token_str_to_bytes(self, tok_str: str) -> bytes:
        # 将“可见化字节序列”逐字符映射回原始单字节
        out = bytearray()
        for ch in tok_str:
            out.append(self._byte_decoder.get(ch, 0x3F))  # 未知字符用'?'字节
        return bytes(out)


class SafeQwenDeTokenizer(QwenDeTokenizer):
    def __init__(self, vocab_json_path: str):
        super().__init__(
            vocab_json_path,
            special_token_policy="keep",   # 永远保留特殊 token
            utf8_errors="replace",         # UTF-8 解码容错
            known_special_tokens=None
        )

    def _lookup_token(self, token: int) -> str:
        """改为越界安全返回"""
        if isinstance(token, int):
            if token < 0 or token >= len(self._id_to_token):
                return f"<UNK:{token}>"
            return self._id_to_token[token]
        return token

    def _token_str_to_bytes(self, tok_str: str) -> bytes:
        """改为单字节容错映射"""
        out = bytearray()
        for ch in tok_str:
            out.append(self._byte_decoder.get(ch, 0x3F))  # 未知字符用'?'字节
        return bytes(out)

    def _is_special(self, tok: str) -> bool:
        """保持原定义，但不会触发报错"""
        return super()._is_special(tok)

    def _handle_special_append(self, tok: str, out: bytearray) -> None:
        """统一保留特殊 token"""
        out.extend(tok.encode("utf-8", errors="replace"))


class KVCProcessor:
    def __init__(self, kvc_list: list[np.ndarray], vocab_path):
        self.hash_table = KVCHashTable()
        self.hash_table.build_from_kvc(kvc_list)
        self.de_tokenizer = QwenDeTokenizer(vocab_path)
    
    def inversion(self, kvc_array, progress_callback=None):
        token_ids = self.hash_table.find_idx_of_kvc(kvc_array, progress_callback)
        print(f"attack token ids: {token_ids}")
        return self.de_tokenizer.de_tokenize(token_ids)