"""Histogram equalisation (luminance and per RGB channel)."""

import numpy as np

from ._common import _luma
from .compute import compute_histogram


def histogram_equalize(pixels):
    """Equalise the luminance histogram; returns grayscale."""
    hist = compute_histogram(pixels)
    total = sum(hist)
    if total == 0:
        return pixels[:]

    # cumulative distribution
    cdf = [0] * 256
    cumsum = 0
    for i in range(256):
        cumsum += hist[i]
        cdf[i] = cumsum

    # smallest non-zero CDF value
    cdf_min = next((c for c in cdf if c > 0), 0)
    if cdf_min == 0 or cdf[-1] == cdf_min:
        # a single-level image cannot be equalised
        out = []
        for r, g, b in pixels:
            y = _luma(r, g, b)
            out.append((y, y, y))
        return out

    denom = total - cdf_min
    mapping = [0] * 256
    for i in range(256):
        v = (cdf[i] - cdf_min) / denom
        if v < 0:
            v = 0.0
        elif v > 1:
            v = 1.0
        val = int(round(v * 255))
        if val < 0:
            val = 0
        elif val > 255:
            val = 255
        mapping[i] = val

    out = []
    for r, g, b in pixels:
        y = _luma(r, g, b)
        v = mapping[y]
        out.append((v, v, v))
    return out


def equalize_histogram_rgb(image: np.ndarray) -> np.ndarray:
    """Equalise each RGB channel independently using a lookup table."""
    image_array = np.asarray(image, dtype=np.uint8)

    equalized = np.zeros_like(image_array, dtype=np.uint8)

    for i in range(3):
        channel = image_array[:, :, i]

        histogram, _ = np.histogram(channel, bins=256, range=(0, 256))
        cdf = histogram.cumsum()
        cdf_normalized = cdf / cdf[-1]

        S0 = cdf_normalized[cdf > 0][0]
        if S0 >= 1:
            # a single-level channel cannot be equalised
            equalized[:, :, i] = channel
            continue
        LUT = ((cdf_normalized - S0) / (1 - S0) * 255).clip(0, 255).astype(np.uint8)

        equalized[:, :, i] = LUT[channel]

    return equalized
