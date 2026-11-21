"""Percent Black: T chosen so that a given share of pixels becomes black."""

import numpy as np

from ._common import luminance_levels, threshold_luminance


def threshold_percent_black(image: np.ndarray, percent_black: float) -> np.ndarray:
    """Choose T as the first level whose cumulative histogram reaches ``percent_black`` % of the pixels."""
    percent_black = max(0.0, min(100.0, percent_black))
    levels = luminance_levels(image)
    cumulative = np.cumsum(np.bincount(levels.ravel(), minlength=256))
    target = levels.size * (percent_black / 100.0)
    T = int(np.argmax(cumulative >= target))
    return threshold_luminance(levels, T)
