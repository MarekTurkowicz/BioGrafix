"""High-pass sharpening filter."""

import numpy as np

from ._common import convolve_rgb

SHARPEN_KERNEL = [
    [0, -1, 0],
    [-1, 5, -1],
    [0, -1, 0],
]


def filter_sharpen(image: np.ndarray) -> np.ndarray:
    """High-pass sharpening filter."""
    return convolve_rgb(image, SHARPEN_KERNEL)
