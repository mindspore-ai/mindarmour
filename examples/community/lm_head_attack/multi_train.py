# 构建完整代码，复现不同查询次数和不同种子下的h值复现结果和W_aligned对应的rms复现结果
import os

os.environ["GLOG_v"] = "3"
from src.dim_svd import recover_hidden_matrix
from src.metrics import rms_error, summary_report
from src.svd_plots import plot_singular_values
from src.numerics import center_matrix
from src.adapters.local_mindformers_llama import MindFormersLlamaAdapter
import numpy as np
from train import collect_full_logits_matrix, align_and_eval
import matplotlib.pyplot as plt
import pandas as pd
import argparse


def main(model_name, num_queries, prompt_len, vocab_subset, seed, batch_size):
    adapter = MindFormersLlamaAdapter(model_name)
    X, subset_idx = collect_full_logits_matrix(
        adapter,
        num_queries=num_queries,
        prompt_len=prompt_len,
        vocab_subset=vocab_subset,
        seed=seed,
        batch_size=batch_size,
    )
    # 行中心化以提升数值稳定性
    Xc = center_matrix(X)

    # 提取隐藏维度 h
    h_est, S, W_hat = recover_hidden_matrix(Xc, gap_threshold=2.0)
    print(f"[RESULT] Estimated hidden dimension h = {h_est}")

    # 读取真实 W（若本地模型可得）
    W_true_full = adapter.get_W_true().astype(np.float64)
    if W_true_full is None:
        print("[WARN] 无法读取真实 W，跳过对齐评估。")
        plot_singular_values(S, h_est)
        return h_est, S, None

    # 如使用了子词表，仅对相同行做评估
    if subset_idx is not None:
        W_true = W_true_full[subset_idx, :]
    else:
        W_true = W_true_full

    rms, W_aligned, G = align_and_eval(W_hat, W_true)
    print(f"[RESULT] RMS after alignment: {rms:.6e}")

    # 简要报告
    report = {
        "model": model_name,
        "num_queries": num_queries,
        "prompt_len": prompt_len,
        "vocab_used": X.shape[1],
        "h_est": h_est,
        "rms_aligned": rms,
    }
    print()
    print("----- Summary -----")
    print(summary_report(report))
    os.makedirs("./outputs", exist_ok=True)
    with open("./outputs/summary_report.txt", "a") as f:
        f.write("----- Summary -----\n")
        f.write(summary_report(report) + "\n")
    return h_est, S, rms


if __name__ == "__main__":
    h_list, rms_list = [], []
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="llama_7b")
    parser.add_argument("--prompt_len", type=int, default=16)
    parser.add_argument("--vocab_subset", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument(
        "--num_queries_list",
        type=int,
        nargs="+",
        default=[1024, 2048, 4000, 5000, 6000],
        required=True,
    )
    args = parser.parse_args()

    h_list, rms_list = [], []

    for num in args.num_queries_list:
        h_est, S, rms = main(
            args.model_name, num, args.prompt_len, args.vocab_subset, args.seed, args.batch_size
        )
        h_list.append(h_est)
        rms_list.append(rms)
        pd.DataFrame(S).to_csv(f"./temp_results/{num}.csv")

    for num in args.num_queries_list:
        S = pd.read_csv(f"./temp_results/{num}.csv", index_col=0)
        plt.plot(S[:-1], label=f"{num}")
    plt.yscale("log")
    plt.legend()
    plt.savefig("./outputs/singular_values_comparison.png")
