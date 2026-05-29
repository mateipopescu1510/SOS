from dataclasses import dataclass

import numpy as np

import src.config as config
from src.pso import sse


@dataclass
class GAResult:
    centroids: np.ndarray   # (K, D) best centroids found
    fitness: float          # SSE of those centroids
    history: np.ndarray     # best fitness per generation


def tournament(fits, size, rng):
    picks = rng.choice(len(fits), size, replace=False)
    return picks[np.argmin(fits[picks])]


def crossover(a, b, rng):
    mask = rng.random(a.shape[0]) < 0.5
    return np.where(mask[:, None], a, b)


def mutate(child, rng):
    mask = rng.random(child.shape) < config.MUTATION_RATE
    return child + mask * rng.normal(0, config.MUTATION_STD, child.shape)


def run_ga(X, rng=None):
    rng = rng or np.random.default_rng(config.SEED)
    P, K, D = config.POP_SIZE, config.K, X.shape[1]
    gens = config.GENERATIONS

    pop = X[rng.choice(len(X), (P, K), replace=True)]
    fits = np.array([sse(X, ind) for ind in pop])

    history = np.empty(gens)
    for t in range(gens):
        order = fits.argsort()
        new_pop = [pop[i] for i in order[:config.ELITES]]
        while len(new_pop) < P:
            a = pop[tournament(fits, config.TOURNAMENT_SIZE, rng)]
            b = pop[tournament(fits, config.TOURNAMENT_SIZE, rng)]
            new_pop.append(mutate(crossover(a, b, rng), rng))
        pop = np.array(new_pop)
        fits = np.array([sse(X, ind) for ind in pop])
        history[t] = fits.min()

    best = fits.argmin()
    return GAResult(pop[best], float(fits[best]), history)


if __name__ == "__main__":
    from src.data import prepare_data
    d = prepare_data()
    result = run_ga(d.X)
    print(f"final SSE: {result.fitness:.1f}")
    print(f"gen   0: {result.history[0]:.1f}")
    print(f"gen 100: {result.history[100]:.1f}")
    print(f"gen -1: {result.history[-1]:.1f}")
