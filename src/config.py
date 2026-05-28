SEED = 42
N_SAMPLES = 10_000
PCA_DIMS = 50

LABEL_FLIP_FRAC = 0.04
CLASS_SWAP_FRAC = 0.04
NOISE_FRAC = 0.04
OOD_FRAC = 0.04
NOISE_STD = 0.5

# similar-looking classes to swap: T-shirt/Shirt, Pullover/Coat, Sandal/Sneaker
SWAP_PAIRS = [(0, 6), (2, 4), (5, 7)]

DATA_DIR = "data"
