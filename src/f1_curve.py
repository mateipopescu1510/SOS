import matplotlib.pyplot as plt
import numpy as np

from src.data import prepare_data
from src.evaluate import anomaly_scores, flag
from src.ga import run_ga
from src.pso import run_pso

KMEANS_F1 = 0.356


def f1_only(scores, mask):
    flags = flag(scores)
    tp = (flags & mask).sum()
    fp = (flags & ~mask).sum()
    fn = (~flags & mask).sum()
    if tp == 0:
        return 0.0
    p = tp / (tp + fp)
    r = tp / (tp + fn)
    return 2 * p * r / (p + r)


def f1_series(history, X, mask):
    return np.array([f1_only(anomaly_scores(X, c), mask) for c in history])


def main():
    d = prepare_data()
    pso = run_pso(d.X)
    ga = run_ga(d.X)

    pso_f1 = f1_series(pso.centroids_history, d.X, d.mask)
    ga_f1 = f1_series(ga.centroids_history, d.X, d.mask)

    plt.figure(figsize=(8, 4.5))
    plt.plot(pso_f1, color="C0", label="PSO")
    plt.plot(ga_f1, color="C1", label="GA")
    plt.axhline(KMEANS_F1, color="gray", ls="--", lw=0.8, label=f"k-means ({KMEANS_F1:.3f})")
    plt.xlabel("iteration / generation")
    plt.ylabel("F1")
    plt.legend()
    plt.tight_layout()
    plt.savefig("plots/f1_vs_iter.png", dpi=120)

    print(f"PSO  peak F1 = {pso_f1.max():.3f} at iter {pso_f1.argmax()}, final = {pso_f1[-1]:.3f}")
    print(f"GA   peak F1 = {ga_f1.max():.3f} at iter {ga_f1.argmax()}, final = {ga_f1[-1]:.3f}")
    print("saved plots/f1_vs_iter.png")


if __name__ == "__main__":
    main()
