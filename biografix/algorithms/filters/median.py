"""Median filter (numpy and scipy versions)."""

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from scipy.signal import medfilt

MAX_WINDOW_ELEMENTS = 4_000_000  # rows are processed in chunks so the window copies stay small


def filter_median(image: np.ndarray, size: int = 3) -> np.ndarray:
    """Median of the size × size neighbourhood of each RGB channel (replicated border)."""
    if size % 2 == 0 or size < 1:
        raise ValueError("The kernel size must be odd and >= 1.")
    rgb = np.asarray(image)[:, :, :3]
    height, width = rgb.shape[:2]
    radius = size // 2
    middle = size * size // 2
    padded = np.pad(rgb, ((radius, radius), (radius, radius), (0, 0)), mode="edge")
    windows = sliding_window_view(padded, (size, size), axis=(0, 1))  # (H, W, 3, size, size), no copy
    out = np.empty_like(rgb)
    rows_per_chunk = max(1, MAX_WINDOW_ELEMENTS // (width * 3 * size * size))
    for top in range(0, height, rows_per_chunk):
        chunk = windows[top:top + rows_per_chunk].reshape(-1, width, 3, size * size)
        out[top:top + rows_per_chunk] = np.partition(chunk, middle, axis=-1)[..., middle]
    return out


def median_scipy(image: np.ndarray, size: int = 5) -> np.ndarray:
    """Median filter on each RGB channel (``scipy.signal.medfilt``)."""
    image_array = np.asarray(image)
    median_filtered = np.zeros_like(image_array)
    for i in range(3):
        median_filtered[:, :, i] = medfilt(image_array[:, :, i], kernel_size=size)
    return median_filtered
