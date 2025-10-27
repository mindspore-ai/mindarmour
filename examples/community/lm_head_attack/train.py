import os

os.environ["GLOG_v"] = "3"
from src.dim_svd import recover_hidden_matrix
from src.metrics import rms_error, summary_report
from src.svd_plots import plot_singular_values
from src.numerics import center_matrix
import numpy as np


def collect_full_logits_matrix(
    adapter,
    num_queries: int,
    prompt_len: int,
    vocab_subset: int = None,
    seed: int = 0,
    batch_size: int = 1,
):
    """
    批量采集多次查询的“最后一个位置”的全量 logits，形成矩阵：
    返回:
        X: [num_queries, vocab_or_subset]
        subset_indices: Optional[np.ndarray]
    """
    rng = np.random.default_rng(seed)
    vsize = adapter.tokenizer.vocab_size

    # 如果指定了子词表
    subset_idx = None
    if vocab_subset is not None:
        subset_idx = rng.choice(vsize, size=vocab_subset, replace=False)

    all_logits = []

    for start in range(0, num_queries, batch_size):
        current_bs = min(batch_size, num_queries - start)

        # 随机生成 token id 序列
        batch_prompts = rng.integers(
            low=0, high=vsize, size=(current_bs, prompt_len), endpoint=False
        ).tolist()

        # 批量获取 logits
        logits_batch = adapter.batch_next_token_logits(
            batch_prompts, pad_token_id=adapter.tokenizer.pad_token_id
        )  # shape: [batch, vocab]

        if subset_idx is not None:
            logits_batch = logits_batch[:, subset_idx]

        all_logits.append(logits_batch)

    X = np.vstack(all_logits)  # shape: [num_queries, vocab or subset]
    return X, subset_idx


def align_and_eval(W_hat: np.ndarray, W_true: np.ndarray):
    """
    通过最小二乘在右侧对齐（求解 G：W_hat G ≈ W_true），并计算 RMS
    """
    # 解 G 的最小二乘：对每列一起解，lstsq(W_hat, W_true)
    G, *_ = np.linalg.lstsq(W_hat.astype(np.float64), W_true.astype(np.float64), rcond=None)
    W_aligned = (W_hat.astype(np.float64) @ G).astype(np.float64)
    err = rms_error(W_true, W_aligned)
    return err, W_aligned, G


def train(adapter, num_queries=5000, prompt_len=16, vocab_subset=None, seed=0, batch_size=8):
    import matplotlib.pyplot as plt

    X, subset_idx = collect_full_logits_matrix(
        adapter,
        num_queries=num_queries,
        prompt_len=prompt_len,
        vocab_subset=vocab_subset,
        seed=seed,
        batch_size=batch_size,
    )
    Xc = center_matrix(X)
    Q = Xc.T.astype(np.float64)
    U, S, _ = np.linalg.svd(Q, full_matrices=False)
    log_s = np.log(np.abs(S))
    gaps = log_s[:-1] - log_s[1:]

    plt.plot(gaps[:-1])
    plt.yscale("log")
    plt.savefig("./outputs/gap_plot.png")
    h_exp = np.argmax(gaps[1:-1]) + 1
    W_hat = U[:, :h_exp] @ np.diag(S[:h_exp])
    W_true_full = adapter.get_W_true().astype(np.float64)
    err, W_aligned, G = align_and_eval(W_hat, W_true_full)
    # 简要报告
    report = {
        "num_queries": num_queries,
        "prompt_len": prompt_len,
        "vocab_used": X.shape[1],
        "h_exp": h_exp,
        "rms_aligned": err,
    }
    print()
    print("----- Summary -----")
    print(summary_report(report))
    os.makedirs("./outputs", exist_ok=True)
    with open("./outputs/summary_report.txt", "w") as f:
        f.write("----- Summary -----\n")
        f.write(summary_report(report))
    print("-------------------")
