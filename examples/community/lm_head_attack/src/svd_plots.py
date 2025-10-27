import numpy as np
import matplotlib.pyplot as plt


def plot_singular_values(S: np.ndarray, h_est: int):
    plt.figure()
    plt.semilogy(S, marker="o")
    plt.axvline(h_est - 1, color="r", linestyle="--", label=f"estimated h={h_est}")
    plt.xlabel("Sorted Singular Values")
    plt.ylabel("Magnitude(log)")
    plt.legend()
    plt.title("Singular Values Distribution")
    plt.show()
