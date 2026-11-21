"""Sauvola local threshold: T = m · (1 + k·(s/R − 1))."""

import numpy as np

from ._common import _gray_average, apply_threshold, local_mean_std


def sauvola_threshold(image: np.ndarray, window_size: int = 15, k: float = 0.2, R: float = 128.0) -> np.ndarray:
    """Sauvola local threshold: T = m · (1 + k·(s/R − 1))."""
    gray = _gray_average(image).astype(np.float64)
    mean, std_dev = local_mean_std(gray, window_size)
    threshold = mean * (1 + k * (std_dev / R - 1))
    return apply_threshold(gray, threshold)
