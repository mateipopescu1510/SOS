import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA

import src.config as config
from src.data import load_fashion_mnist, load_mnist_images, inject_errors


def main():
    rng = np.random.default_rng(config.SEED)
    images, labels = load_fashion_mnist(config.N_SAMPLES, rng)
    mnist = load_mnist_images(rng)
    images, _, _, _ = inject_errors(images, labels, mnist, rng)

    cumulative = np.cumsum(PCA().fit(images).explained_variance_ratio_)

    plt.figure(figsize=(7, 4))
    plt.plot(range(1, len(cumulative) + 1), cumulative)
    for frac in (0.85, 0.90, 0.95):
        dims = int(np.argmax(cumulative >= frac)) + 1
        plt.axhline(frac, color="gray", ls="--", lw=0.6)
        plt.annotate(f"{int(frac * 100)}% @ {dims}d", (dims, frac))
    plt.axvline(config.PCA_DIMS, color="red", lw=0.8, label=f"current: {config.PCA_DIMS}d")
    plt.xlabel("components")
    plt.ylabel("cumulative variance")
    plt.legend()
    plt.tight_layout()
    plt.savefig("plots/variance.png", dpi=120)
    print(f"{config.PCA_DIMS} dims keep {cumulative[config.PCA_DIMS - 1]:.1%} of variance")


if __name__ == "__main__":
    main()
