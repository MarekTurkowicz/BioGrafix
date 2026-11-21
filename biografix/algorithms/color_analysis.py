"""Colour analysis: share of pixels within an HSV range."""

import numpy as np


def rgb_to_hsv(image: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """H (0..1), S, V of every pixel, with the same formulas as ``colorsys.rgb_to_hsv``."""
    rgb = np.asarray(image)[:, :, :3].astype(np.float64) / 255.0
    r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
    maxc = np.maximum(np.maximum(r, g), b)
    minc = np.minimum(np.minimum(r, g), b)
    rangec = maxc - minc
    gray = rangec == 0
    with np.errstate(divide="ignore", invalid="ignore"):
        s = np.where(gray, 0.0, rangec / maxc)
        rc = (maxc - r) / rangec
        gc = (maxc - g) / rangec
        bc = (maxc - b) / rangec
        h = np.where(r == maxc, bc - gc, np.where(g == maxc, 2.0 + rc - bc, 4.0 + gc - rc))
    h = np.where(gray, 0.0, np.mod(h / 6.0, 1.0))
    return h, s, maxc


def color_coverage(image: np.ndarray, h_min=70.0, h_max=160.0, s_min=0.2, v_min=0.2):
    """Return the percentage of pixels inside the HSV range (green by default).

    Returns ``(percent, h_min, h_max)`` with the hue range normalised.
    """
    # clamp the hue range to [0, 360] and make sure min <= max
    h_min = max(0.0, min(360.0, h_min))
    h_max = max(0.0, min(360.0, h_max))
    if h_max < h_min:
        h_min, h_max = h_max, h_min

    h, s, v = rgb_to_hsv(image)
    if h.size == 0:
        raise ValueError("The image is empty.")
    h_deg = h * 360.0
    inside = (h_deg >= h_min) & (h_deg <= h_max) & (s >= s_min) & (v >= v_min)
    return 100.0 * int(inside.sum()) / h.size, h_min, h_max
