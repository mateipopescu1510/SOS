import numpy as np
from sklearn.metrics import roc_auc_score

import src.config as config


def anomaly_scores(X, centroids):
    d2 = (X**2).sum(1)[:, None] + (centroids**2).sum(1) - 2 * X @ centroids.T
    return np.sqrt(np.maximum(d2.min(axis=1), 0))


def flag(scores, fraction=config.FLAG_FRACTION):
    cutoff = np.quantile(scores, 1 - fraction)
    return scores >= cutoff


def metrics(scores, mask, kind):
    flags = flag(scores)
    tp = int((flags & mask).sum())
    fp = int((flags & ~mask).sum())
    fn = int((~flags & mask).sum())

    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    auc = roc_auc_score(mask, scores)

    per_kind = {}
    for k in ("occlusion", "ood", "flip", "swap"):
        is_k = kind == k
        per_kind[k] = float((flags & is_k).sum() / is_k.sum())

    return dict(precision=precision, recall=recall, f1=f1, auc=auc, per_kind=per_kind)


def report(name, m):
    print(f"\n{name}")
    print(f"  precision: {m['precision']:.3f}")
    print(f"  recall:    {m['recall']:.3f}")
    print(f"  F1:        {m['f1']:.3f}")
    print(f"  AUC:       {m['auc']:.3f}")
    print(f"  recall by kind:")
    for k, v in m["per_kind"].items():
        print(f"    {k:<6} {v:.3f}")


if __name__ == "__main__":
    from src.data import prepare_data
    from src.pso import run_pso
    from src.ga import run_ga

    d = prepare_data()
    pso = run_pso(d.X)
    ga = run_ga(d.X)

    report("PSO", metrics(anomaly_scores(d.X, pso.centroids), d.mask, d.kind))
    report("GA",  metrics(anomaly_scores(d.X, ga.centroids),  d.mask, d.kind))
