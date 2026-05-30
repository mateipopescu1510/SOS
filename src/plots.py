import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

import src.config as config
from src.data import prepare_data
from src.evaluate import anomaly_scores, flag
from src.ga import run_ga
from src.pso import run_pso


KINDS = ("occlusion", "ood", "flip", "swap")
COLORS = {"occlusion": "C2", "ood": "C3", "flip": "C4", "swap": "C5"}


def score_histogram(scores, mask, kind_col, k, ax, bins, cutoff):
    ax.hist(scores[~mask], bins=bins, density=True, alpha=0.55, color="lightgray", label="clean")
    ax.hist(scores[kind_col == k], bins=bins, density=True, alpha=0.8, color=COLORS[k], label=k)
    ax.axvline(cutoff, color="black", ls="--", lw=0.9, label="flag cutoff")
    ax.set_yscale("log")
    ax.legend(fontsize=7, loc="upper right")


def per_kind_bars(d, kmeans_c, pso_c, ga_c, fraction):
    def recalls(centroids):
        flags = flag(anomaly_scores(d.X, centroids))
        return [(flags & (d.kind == k)).sum() / (d.kind == k).sum() for k in KINDS]

    x = np.arange(len(KINDS))
    w = 0.27
    plt.figure(figsize=(7, 4))
    plt.bar(x - w, recalls(kmeans_c), w, label="k-means")
    plt.bar(x, recalls(pso_c), w, label="PSO")
    plt.bar(x + w, recalls(ga_c), w, label="GA")
    plt.axhline(fraction, color="gray", ls="--", lw=0.8, label="base rate (random flagging)")
    plt.xticks(x, KINDS)
    plt.ylabel("recall")
    plt.legend()
    plt.tight_layout()
    plt.savefig("plots/per_kind_bars.png", dpi=120)


def scatter_2d(Y, kind, mask, C2, name, ax, sample):
    ax.set_title(name)
    clean_idx = np.where(~mask)[0]
    clean_idx = np.random.default_rng(0).choice(clean_idx, sample, replace=False)
    ax.scatter(Y[clean_idx, 0], Y[clean_idx, 1], s=3, alpha=0.3, color="lightgray", label="clean")
    for k in KINDS:
        m = kind == k
        ax.scatter(Y[m, 0], Y[m, 1], s=8, alpha=0.7, color=COLORS[k], label=k)
    ax.scatter(C2[:, 0], C2[:, 1], s=120, marker="x", color="black", linewidths=2, label="centroids")
    ax.legend(fontsize=8, markerscale=1.5, loc="best")


def main():
    d = prepare_data()

    print("running PSO...", flush=True)
    pso = run_pso(d.X)
    print(f"  PSO SSE={pso.fitness:.0f}", flush=True)

    print("running GA...", flush=True)
    ga = run_ga(d.X)
    print(f"  GA  SSE={ga.fitness:.0f}", flush=True)

    print("running k-means...", flush=True)
    km = KMeans(n_clusters=config.K, n_init=10, random_state=config.SEED).fit(d.X)
    print(f"  k-means SSE={km.inertia_:.0f}", flush=True)

    print("plotting score histograms...", flush=True)
    algo_scores = [
        ("k-means", anomaly_scores(d.X, km.cluster_centers_)),
        ("PSO", anomaly_scores(d.X, pso.centroids)),
        ("GA", anomaly_scores(d.X, ga.centroids)),
    ]
    lo = min(s.min() for _, s in algo_scores)
    hi = max(s.max() for _, s in algo_scores)
    bins = np.linspace(lo, hi, 50)

    cutoffs = [np.quantile(s, 1 - config.FLAG_FRACTION) for _, s in algo_scores]

    fig, axes = plt.subplots(len(KINDS), 3, figsize=(12, 9), sharex=True, sharey=True)
    for i, k in enumerate(KINDS):
        for j, (name, s) in enumerate(algo_scores):
            score_histogram(s, d.mask, d.kind, k, axes[i, j], bins, cutoffs[j])
            if i == 0:
                axes[i, j].set_title(name)
            if j == 0:
                axes[i, j].set_ylabel(f"{k}\ndensity")
    for ax in axes[-1, :]:
        ax.set_xlabel("anomaly score")
    plt.tight_layout()
    plt.savefig("plots/score_histograms.png", dpi=120)

    print("plotting per-kind bars...", flush=True)
    per_kind_bars(d, km.cluster_centers_, pso.centroids, ga.centroids, config.FLAG_FRACTION)

    print("plotting 2D scatter...", flush=True)
    pca2 = PCA(n_components=2, random_state=config.SEED).fit(d.X)
    Y = pca2.transform(d.X)
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharex=True, sharey=True)
    scatter_2d(Y, d.kind, d.mask, pca2.transform(km.cluster_centers_), "k-means", axes[0], sample=2000)
    scatter_2d(Y, d.kind, d.mask, pca2.transform(pso.centroids), "PSO", axes[1], sample=2000)
    scatter_2d(Y, d.kind, d.mask, pca2.transform(ga.centroids), "GA", axes[2], sample=2000)
    plt.tight_layout()
    plt.savefig("plots/scatter_2d.png", dpi=120)

    print("done.", flush=True)


if __name__ == "__main__":
    main()
