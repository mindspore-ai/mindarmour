import numpy as np
from typing import Tuple


def recover_hidden_matrix(
    logits_matrix: np.ndarray, gap_threshold: float = 2.0
) -> Tuple[int, np.ndarray, np.ndarray]:
    """
    从logits矩阵中估计隐藏层维度 h并复原W_hat
    Args:
        logits_matrix: shape [num_queries, vocab_size]
        gap_threshold: 判断奇异值落差的阈值（log域差值）
    Returns:
        h: 估计的隐藏层维度
        singular_values: 奇异值数组
        W: 复原的矩阵
    """
    Q = logits_matrix.T.astype(np.float64) # 转置为[l,n]
    U, S, _ = np.linalg.svd(Q, full_matrices=False)
    log_s = np.log(np.abs(S))
    gaps = - np.diff(log_s)[1:-1]
    h = int(np.argmax(gaps) + 1)
    if gaps[h - 1] < np.log(gap_threshold):
        print("[WARN] 最大gap低于阈值，结果可能不稳定")
    W_hat = U[:,:h] @ np.diag(S[:h])
    return h, S, W_hat
