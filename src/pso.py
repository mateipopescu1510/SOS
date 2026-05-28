from dataclasses import dataclass

import numpy as np

import src.config as config


@dataclass
class PSOResult:
    centroids: np.ndarray   # (K, D) best centroids found
    fitness: float          # SSE of those centroids
    history: np.ndarray     # gbest fitness per iteration


def sse(X, centroids):
    d2 = (X**2).sum(1)[:, None] + (centroids**2).sum(1) - 2 * X @ centroids.T
    return d2.min(axis=1).sum()


def run_pso(X, rng=None):
    rng = rng or np.random.default_rng(config.SEED)
    P, K, D = config.SWARM_SIZE, config.K, X.shape[1]
    iters = config.ITERATIONS

    pos = X[rng.choice(len(X), (P, K), replace=True)]
    vel = np.zeros_like(pos)

    pbest_pos = pos.copy()
    pbest_fit = np.array([sse(X, p) for p in pos])
    g = pbest_fit.argmin()
    gbest_pos = pbest_pos[g].copy()
    gbest_fit = float(pbest_fit[g])

    history = np.empty(iters)
    for t in range(iters):
        w = config.W_MAX - (config.W_MAX - config.W_MIN) * t / (iters - 1)
        r1 = rng.random(pos.shape)
        r2 = rng.random(pos.shape)
        vel = w * vel + config.C1 * r1 * (pbest_pos - pos) + config.C2 * r2 * (gbest_pos - pos)
        pos = pos + vel

        fit = np.array([sse(X, p) for p in pos])
        better = fit < pbest_fit
        pbest_pos[better] = pos[better]
        pbest_fit[better] = fit[better]

        g = pbest_fit.argmin()
        if pbest_fit[g] < gbest_fit:
            gbest_fit = float(pbest_fit[g])
            gbest_pos = pbest_pos[g].copy()
        history[t] = gbest_fit

    return PSOResult(gbest_pos, gbest_fit, history)


if __name__ == "__main__":
    from src.data import prepare_data
    d = prepare_data()
    result = run_pso(d.X)
    print(f"final SSE: {result.fitness:.1f}")
    print(f"iter   0: {result.history[0]:.1f}")
    print(f"iter  50: {result.history[50]:.1f}")
    print(f"iter -1: {result.history[-1]:.1f}")
