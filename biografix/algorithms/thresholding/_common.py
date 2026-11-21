"""Helpers shared by the thresholding methods: grayscale conversion, applying a threshold, local statistics."""

import numpy as np
from scipy.ndimage import uniform_filter


def _luma(r, g, b):
    y = 0.299 * r + 0.587 * g + 0.114 * b
    if y < 0:
        y = 0
    elif y > 255:
        y = 255
    return int(round(y))


def _to_gray_list(pixels):
    """Return the luminance (0..255) of every pixel."""
    return [_luma(r, g, b) for (r, g, b) in pixels]


def _apply_threshold_to_pixels(pixels, T):
    """Apply threshold T: luminance below T becomes black, the rest white."""
    out = []
    for r, g, b in pixels:
        y = _luma(r, g, b)
        v = 0 if y < T else 255
        out.append((v, v, v))
    return out


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
