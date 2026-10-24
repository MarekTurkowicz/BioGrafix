"""Median filter (pure Python and scipy versions)."""

import numpy as np
from scipy.signal import medfilt

from ._common import _to_1d, _to_2d


def filter_median(pixels, w, h, size=3):
    """Median filter."""
    if size % 2 == 0 or size < 1:
        raise ValueError("The kernel size must be odd and >= 1.")

    src = _to_2d(pixels, w, h)
    r = size // 2
    out = [[(0, 0, 0) for _ in range(w)] for _ in range(h)]

    for y in range(h):
        for x in range(w):
            neigh_r = []
            neigh_g = []
            neigh_b = []
            for j in range(-r, r + 1):
                for i in range(-r, r + 1):
                    yy = y + j
                    xx = x + i
                    if yy < 0:
                        yy = 0
                    if yy >= h:
                        yy = h - 1
                    if xx < 0:
                        xx = 0
                    if xx >= w:
                        xx = w - 1
                    rr, gg, bb = src[yy][xx]
                    neigh_r.append(rr)
                    neigh_g.append(gg)
                    neigh_b.append(bb)
            neigh_r.sort()
            neigh_g.sort()
            neigh_b.sort()
            m_idx = len(neigh_r) // 2
            out[y][x] = (neigh_r[m_idx], neigh_g[m_idx], neigh_b[m_idx])
    return _to_1d(out)


def median_scipy(image: np.ndarray, size: int = 5) -> np.ndarray:
    """Median filter on each RGB channel (``scipy.signal.medfilt``)."""
    image_array = np.asarray(image)
    median_filtered = np.zeros_like(image_array)
    for i in range(3):
        median_filtered[:, :, i] = medfilt(image_array[:, :, i], kernel_size=size)
    return median_filtered
