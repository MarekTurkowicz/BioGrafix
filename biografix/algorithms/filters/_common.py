"""Helpers shared by the filters: convolution with a replicated border."""

import numpy as np

from ..point_ops import round_to_byte  # noqa: F401  (used by the filter modules)


def correlate_replicated(channels: np.ndarray, kernel) -> np.ndarray:
    """Weighted sum of the neighbourhood of every pixel, as float.

    ``channels`` is (H, W) or (H, W, C); ``kernel`` is a list of rows whose centre
    is element [kh // 2][kw // 2]. Pixels beyond the border repeat the edge pixel.
    The kernel is applied without flipping, like the original per-pixel loop, and
    the terms are added in the same order, so the result is identical to it.
    """
    kernel = [list(row) for row in kernel]
    kh, kw = len(kernel), len(kernel[0])
    ky, kx = kh // 2, kw // 2
    data = np.asarray(channels, dtype=np.float64)
    height, width = data.shape[:2]
    pad = [(ky, kh - 1 - ky), (kx, kw - 1 - kx)] + [(0, 0)] * (data.ndim - 2)
    padded = np.pad(data, pad, mode="edge")
    total = np.zeros_like(data)
    for j in range(kh):
        for i in range(kw):
            total += padded[j:j + height, i:i + width] * kernel[j][i]
    return total


def convolve_rgb(image: np.ndarray, kernel) -> np.ndarray:
    """Apply ``kernel`` to each RGB channel (replicated border) and round to uint8."""
    return round_to_byte(correlate_replicated(np.asarray(image)[:, :, :3], kernel))
