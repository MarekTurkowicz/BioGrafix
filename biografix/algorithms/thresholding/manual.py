"""Manual threshold: the user chooses T."""

import numpy as np

from ._common import _apply_threshold_to_pixels, _gray_average, apply_threshold


def threshold_manual(pixels, T):
    """Threshold at a user-given T."""
    if T < 0:
        T = 0
    elif T > 255:
        T = 255
    return _apply_threshold_to_pixels(pixels, T)


def manual_threshold(image: np.ndarray, threshold: int) -> np.ndarray:
    """Threshold the channel average at a user-given T."""
    grayscale = _gray_average(image)
    return apply_threshold(grayscale, threshold)
