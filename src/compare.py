import matplotlib.pyplot as plt

from src.data import prepare_data
from src.evaluate import anomaly_scores, metrics
from src.ga import run_ga
from src.pso import run_pso


def main():
    d = prepare_data()
    pso = run_pso(d.X)
    ga = run_ga(d.X)

    pso_f1 = metrics(anomaly_scores(d.X, pso.centroids), d.mask, d.kind)["f1"]
    ga_f1 = metrics(anomaly_scores(d.X, ga.centroids), d.mask, d.kind)["f1"]

    n = len(d.X)
    plt.figure(figsize=(8, 4.5))
    plt.plot(pso.history / n, label=f"PSO  F1={pso_f1:.3f}  final={pso.fitness/n:.2f}")
    plt.plot(ga.history / n, label=f"GA   F1={ga_f1:.3f}  final={ga.fitness/n:.2f}")
    plt.xlabel("iteration / generation")
    plt.ylabel("mean squared distance per point")
    plt.legend()
    plt.tight_layout()
    plt.savefig("plots/convergence.png", dpi=120)

    print(f"PSO  F1={pso_f1:.3f}  mean sq dist={pso.fitness/n:.2f}")
    print(f"GA   F1={ga_f1:.3f}  mean sq dist={ga.fitness/n:.2f}")
    print("saved plots/convergence.png")


if __name__ == "__main__":
    main()
