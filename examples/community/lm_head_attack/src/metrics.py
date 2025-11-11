import numpy as np
from typing import Dict


def rms_error(W_true: np.ndarray, W_est: np.ndarray) -> float:
    """
    计算均方根差
    """
    return np.sqrt(np.mean((W_true - W_est) ** 2))


def bit_agreement(logits_true: np.ndarray, logits_est: np.ndarray) -> float:
    """
    估算logits的位精度（假设已归一化）
    """
    diff = np.abs(logits_true - logits_est)
    return -np.log2(np.mean(diff) + 1e-12)


def summary_report(results: Dict) -> str:
    """
    生成简要评估报告
    """
    lines = ["=== 攻击实验评估报告 ==="]
    for k, v in results.items():
        lines.append(f"{k}:{v}")
    return "\n".join(lines)
