import matplotlib.pyplot as plt
import numpy as np

import src.config as config
from src.data import prepare_data
from src.evaluate import anomaly_scores, metrics
from src.ga import run_ga
from src.pso import run_pso


def f1_for(algo, X, mask, kind, seed, **overrides):
    saved = {k: getattr(config, k) for k in overrides}
    saved_seed = config.SEED
    config.SEED = seed
    for k, v in overrides.items():
        setattr(config, k, v)
    result = algo(X)
    for k, v in saved.items():
        setattr(config, k, v)
    config.SEED = saved_seed
    return metrics(anomaly_scores(X, result.centroids), mask, kind)["f1"]


def main():
    d = prepare_data()
    X, mask, kind = d.X, d.mask, d.kind
    seeds = (0, 1)
    sizes = [10, 30, 50, 80]

    pso = np.empty((len(sizes), len(seeds)))
    ga = np.empty((len(sizes), len(seeds)))

    for i, sz in enumerate(sizes):
        for j, s in enumerate(seeds):
            pso[i, j] = f1_for(run_pso, X, mask, kind, s, SWARM_SIZE=sz, ITERATIONS=200)
            print(f"  PSO size={sz:>2} seed={s}  F1={pso[i, j]:.3f}", flush=True)
            ga[i, j] = f1_for(run_ga, X, mask, kind, s, POP_SIZE=sz, GENERATIONS=200)
            print(f"  GA  size={sz:>2} seed={s}  F1={ga[i, j]:.3f}", flush=True)

    plt.figure(figsize=(7, 4))
    for scores, label, color in [(pso, "PSO", "C0"), (ga, "GA", "C1")]:
        m, s = scores.mean(axis=1), scores.std(axis=1)
        plt.plot(sizes, m, marker="o", color=color, label=label)
        plt.fill_between(sizes, m - s, m + s, color=color, alpha=0.2)
    plt.xlabel("swarm / population size")
    plt.ylabel("F1")
    plt.legend()
    plt.tight_layout()
    plt.savefig("plots/sweep_size.png", dpi=120)
    print("saved plots/sweep_size.png", flush=True)


if __name__ == "__main__":
    main()
