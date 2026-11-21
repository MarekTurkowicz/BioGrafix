"""Manual threshold: the user chooses T."""

import numpy as np

from ._common import _gray_average, apply_threshold, luminance_levels, threshold_luminance


def threshold_manual(image: np.ndarray, T: int) -> np.ndarray:
    """Threshold the luminance at a user-given T (below T becomes black)."""
    return threshold_luminance(luminance_levels(image), max(0, min(255, T)))


def manual_threshold(image: np.ndarray, threshold: int) -> np.ndarray:
    """Threshold the channel average at a user-given T."""
    grayscale = _gray_average(image)
    return apply_threshold(grayscale, threshold)
