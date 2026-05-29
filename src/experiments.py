import src.config as config
from src.data import prepare_data
from src.evaluate import anomaly_scores, metrics
from src.ga import run_ga
from src.pso import run_pso


def run(name, algo, X, mask, kind, **overrides):
    saved = {k: getattr(config, k) for k in overrides}
    for k, v in overrides.items():
        setattr(config, k, v)
    result = algo(X)
    for k, v in saved.items():
        setattr(config, k, v)

    m = metrics(anomaly_scores(X, result.centroids), mask, kind)
    print(
        f"{name:<32} F1={m['f1']:.3f}  AUC={m['auc']:.3f}  "
        f"occ={m['per_kind']['occlusion']:.2f}  ood={m['per_kind']['ood']:.2f}  "
        f"flip={m['per_kind']['flip']:.2f}  swap={m['per_kind']['swap']:.2f}"
    )
    return m


def main():
    d = prepare_data()
    X, mask, kind = d.X, d.mask, d.kind

    print("--- PSO ---")
    run("baseline", run_pso, X, mask, kind)
    run("W_MAX=0.7", run_pso, X, mask, kind, W_MAX=0.7)
    run("iters=400", run_pso, X, mask, kind, ITERATIONS=400)
    run("W_MAX=0.7, iters=400", run_pso, X, mask, kind, W_MAX=0.7, ITERATIONS=400)
    run("swarm=50, iters=400", run_pso, X, mask, kind, SWARM_SIZE=50, ITERATIONS=400)

    print("\n--- GA ---")
    run("baseline (T=3)", run_ga, X, mask, kind)
    run("T=5", run_ga, X, mask, kind, TOURNAMENT_SIZE=5)
    run("T=7", run_ga, X, mask, kind, TOURNAMENT_SIZE=7)
    run("pop=50, gens=400", run_ga, X, mask, kind, POP_SIZE=50, GENERATIONS=400)
    run("pop=50, gens=400, T=5", run_ga, X, mask, kind,
        POP_SIZE=50, GENERATIONS=400, TOURNAMENT_SIZE=5)


if __name__ == "__main__":
    main()
