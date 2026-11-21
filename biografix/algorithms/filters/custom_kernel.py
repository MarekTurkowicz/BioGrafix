"""Convolution with a user-given kernel (replicated border)."""

import numpy as np

from ._common import convolve_rgb


def filter_custom(image: np.ndarray, kernel) -> np.ndarray:
    """Convolve with an arbitrary kernel given as a list of rows."""
    return convolve_rgb(image, kernel)
