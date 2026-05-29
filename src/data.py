from dataclasses import dataclass

import numpy as np
from sklearn.decomposition import PCA
from torchvision.datasets import FashionMNIST, MNIST

import src.config as config


@dataclass
class Dataset:
    X: np.ndarray       # PCA features, shape (n, dims)
    labels: np.ndarray  # labels after corruption
    mask: np.ndarray    # True where we injected an error
    kind: np.ndarray    # error type per sample ("clean", "flip", ...)


def load_fashion_mnist(n, rng):
    ds = FashionMNIST(config.DATA_DIR, train=True, download=True)
    images = ds.data.numpy().reshape(-1, 784) / 255.0
    labels = ds.targets.numpy()
    idx = rng.choice(len(images), n, replace=False)
    return images[idx], labels[idx]


def load_mnist_images(rng):
    ds = MNIST(config.DATA_DIR, train=True, download=True)
    return ds.data.numpy().reshape(-1, 784) / 255.0


def inject_errors(images, labels, mnist, rng):
    n = len(images)
    images, labels = images.copy(), labels.copy()
    mask = np.zeros(n, bool)
    kind = np.full(n, "clean", dtype=object)
    free = list(rng.permutation(n))

    def take(count, where=lambda i: True):
        picked = [i for i in free if where(i)][:count]
        for i in picked:
            free.remove(i)
        return picked

    swap = {a: b for x, y in config.SWAP_PAIRS for a, b in ((x, y), (y, x))}
    for i in take(int(config.CLASS_SWAP_FRAC * n), lambda i: labels[i] in swap):
        labels[i] = swap[labels[i]]
        mask[i], kind[i] = True, "swap"

    for i in take(int(config.LABEL_FLIP_FRAC * n)):
        labels[i] = rng.choice([c for c in range(10) if c != labels[i]])
        mask[i], kind[i] = True, "flip"

    s = config.OCCLUSION_SIZE
    for i in take(int(config.OCCLUSION_FRAC * n)):
        y, x = rng.integers(0, 28 - s + 1, 2)
        images[i].reshape(28, 28)[y:y + s, x:x + s] = 0
        mask[i], kind[i] = True, "occlusion"

    ood = take(int(config.OOD_FRAC * n))
    picks = mnist[rng.choice(len(mnist), len(ood), replace=False)]
    for i, d in zip(ood, picks):
        images[i] = d
        mask[i], kind[i] = True, "ood"

    return images, labels, mask, kind


def reduce_pca(X, dims):
    return PCA(n_components=dims, random_state=config.SEED).fit_transform(X)


def prepare_data():
    rng = np.random.default_rng(config.SEED)
    images, labels = load_fashion_mnist(config.N_SAMPLES, rng)
    mnist = load_mnist_images(rng)
    images, labels, mask, kind = inject_errors(images, labels, mnist, rng)
    X = reduce_pca(images, config.PCA_DIMS)
    return Dataset(X, labels, mask, kind)
