"""Otsu's method: maximise the between-class variance."""

import numpy as np

from ._common import _gray_average, apply_threshold


def otsu_threshold_value(grayscale: np.ndarray) -> int:
    """Otsu threshold: maximise the between-class variance ω0·ω1·(μ0−μ1)²."""
    histogram, _ = np.histogram(grayscale, bins=256, range=(0, 256))

    total_pixels = grayscale.size
    histogram_normalized = histogram / total_pixels

    cumsum = np.cumsum(histogram_normalized)
    cumulative_mean = np.cumsum(histogram_normalized * np.arange(256))

    global_mean = cumulative_mean[-1]

    max_between_class_variance = 0
    best_threshold = 0

    for t in range(256):
        weight_background = cumsum[t]
        weight_foreground = 1 - weight_background

        if weight_background == 0 or weight_foreground == 0:
            continue

        mean_background = cumulative_mean[t] / weight_background
        mean_foreground = (global_mean - cumulative_mean[t]) / weight_foreground

        between_class_variance = weight_background * weight_foreground * (mean_background - mean_foreground) ** 2

        if between_class_variance > max_between_class_variance:
            max_between_class_variance = between_class_variance
            best_threshold = t

    return best_threshold


def otsu_threshold(image: np.ndarray) -> np.ndarray:
    """Threshold with Otsu's method."""
    grayscale = _gray_average(image)
    return apply_threshold(grayscale, otsu_threshold_value(grayscale))
