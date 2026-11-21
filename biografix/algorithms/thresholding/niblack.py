"""Niblack local threshold: T = m + k·s."""

import numpy as np

from ._common import _gray_average, apply_threshold, local_mean_std


def niblack_threshold(image: np.ndarray, window_size: int = 15, k: float = -0.2) -> np.ndarray:
    """Niblack local threshold: T = m + k·s."""
    gray = _gray_average(image).astype(np.float64)
    mean, std_dev = local_mean_std(gray, window_size)
    threshold = mean + k * std_dev
    return apply_threshold(gray, threshold)
