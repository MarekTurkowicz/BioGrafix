"""5×5 Gaussian blur."""

import numpy as np

from ._common import convolve_rgb

GAUSSIAN_KERNEL = [
    [1, 4, 6, 4, 1],
    [4, 16, 24, 16, 4],
    [6, 24, 36, 24, 6],
    [4, 16, 24, 16, 4],
    [1, 4, 6, 4, 1],
]


def filter_gaussian(image: np.ndarray) -> np.ndarray:
    """5×5 Gaussian blur (binomial weights, sum 256)."""
    norm = 1.0 / 256.0
    return convolve_rgb(image, [[v * norm for v in row] for row in GAUSSIAN_KERNEL])
