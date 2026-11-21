"""Mean Iterative Selection."""

import numpy as np

from ._common import luminance_levels, threshold_luminance


def threshold_mean_iterative(image: np.ndarray, max_iter: int = 100, eps: float = 0.5) -> np.ndarray:
    """Mean Iterative Selection: iterate T towards the mean of both class means."""
    levels = luminance_levels(image)
    T = int(levels.sum()) / levels.size  # start from the global mean

    for _ in range(max_iter):
        lower = levels <= T
        lower_count = int(lower.sum())
        if lower_count in (0, levels.size):
            break
        mean_lower = int(levels[lower].sum()) / lower_count
        mean_upper = int(levels[~lower].sum()) / (levels.size - lower_count)
        new_T = (mean_lower + mean_upper) / 2.0
        if abs(new_T - T) < eps:
            T = new_T
            break
        T = new_T

    return threshold_luminance(levels, int(round(T)))
