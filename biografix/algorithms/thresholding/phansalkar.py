"""Phansalkar local threshold: T = m · (1 + p·exp(−q·m) + k·(s/R − 1))."""

import numpy as np

from ._common import _gray_average, apply_threshold, local_mean_std


def phansalkar_threshold(image: np.ndarray, window_size: int = 15, k: float = 0.25,
                         p: float = 2.0, q: float = 10.0, R: float = 0.5) -> np.ndarray:
    """Phansalkar local threshold: T = m · (1 + p·exp(−q·m) + k·(s/R − 1)).

    Computed on the image normalised to [0, 1], hence R = 0.5.
    """
    gray = _gray_average(image).astype(np.float64) / 255.0
    mean, std_dev = local_mean_std(gray, window_size)
    threshold = mean * (1 + p * np.exp(-q * mean) + k * (std_dev / R - 1))
    return apply_threshold(gray, threshold)
