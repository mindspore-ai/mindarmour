import numpy as np


def center_matrix(X: np.ndarray) -> np.ndarray:
    """
    按行中心化矩阵
    """
    return X - X.mean(axis=0, keepdims=True)


def normalize_rows(X: np.ndarray) -> np.ndarray:
    """
    按行归一化
    """
    norms = np.linalg.norm(X, axis=1, keepdims=True) + 1e-12
    return X / norms
