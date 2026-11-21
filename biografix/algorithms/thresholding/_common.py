"""Helpers shared by the thresholding methods: grayscale conversion, applying a threshold, local statistics."""

import numpy as np
from scipy.ndimage import uniform_filter

from ..point_ops import luminance_levels  # noqa: F401  (used by the luminance methods)


def threshold_luminance(levels: np.ndarray, threshold: int) -> np.ndarray:
    """Luminance below ``threshold`` becomes black, the rest white (RGB image)."""
    return binary_to_rgb(np.where(levels < threshold, 0, 255).astype(np.uint8))


def _gray_average(image: np.ndarray) -> np.ndarray:
    """Channel average (R+G+B)/3 as uint8."""
    return np.clip(np.asarray(image, dtype=np.float32).mean(axis=2), 0, 255).astype(np.uint8)


def binary_to_rgb(binary_image: np.ndarray) -> np.ndarray:
    """Expand a binary (H, W) image to RGB for display."""
    return np.stack([binary_image] * 3, axis=-1)


def apply_threshold(gray: np.ndarray, threshold) -> np.ndarray:
    """Threshold with a scalar or per-pixel threshold: values above it become 255."""
    binary_image = np.where(gray > threshold, 255, 0).astype(np.uint8)
    return binary_to_rgb(binary_image)


def local_mean_std(gray: np.ndarray, window_size: int):
    """Local mean ``m`` and standard deviation ``s`` in a square window.

    Uses s² = E[x²] − E[x]²; the input is converted to float so squares do not overflow.
    """
    gray = np.asarray(gray, dtype=np.float64)
    mean = uniform_filter(gray, size=window_size, mode="reflect")
    squared_mean = uniform_filter(gray ** 2, size=window_size, mode="reflect")
    std_dev = np.sqrt(np.maximum(squared_mean - mean ** 2, 0))
    return mean, std_dev
