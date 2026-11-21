"""Histogram stretching (luminance and per RGB channel)."""

import numpy as np

from ._common import _luma
from .compute import compute_histogram


def histogram_stretch(pixels):
    """Stretch luminance linearly so that min → 0 and max → 255; returns grayscale."""
    hist = compute_histogram(pixels)
    total = sum(hist)
    if total == 0:
        return pixels[:]

    imin = 0
    while imin < 256 and hist[imin] == 0:
        imin += 1
    imax = 255
    while imax >= 0 and hist[imax] == 0:
        imax -= 1

    if imin >= imax:
        # a uniform image cannot be stretched
        gray = imin
        return [(gray, gray, gray) for _ in pixels]

    mapping = [0] * 256
    scale = 255.0 / (imax - imin)
    for i in range(256):
        if i <= imin:
            v = 0
        elif i >= imax:
            v = 255
        else:
            v = int(round((i - imin) * scale))
        if v < 0:
            v = 0
        elif v > 255:
            v = 255
        mapping[i] = v

    out = []
    for r, g, b in pixels:
        y = _luma(r, g, b)
        v = mapping[y]
        out.append((v, v, v))
    return out


def stretch_histogram_rgb(image: np.ndarray) -> np.ndarray:
    """Stretch the histogram of each RGB channel independently."""
    image_array = np.asarray(image, dtype=np.float32)

    stretched = np.zeros_like(image_array, dtype=np.uint8)

    for i in range(3):
        channel = image_array[:, :, i]
        k_min = channel.min()
        k_max = channel.max()

        if k_max > k_min:
            stretched[:, :, i] = ((channel - k_min) / (k_max - k_min) * 255).clip(0, 255).astype(np.uint8)
        else:
            stretched[:, :, i] = channel.astype(np.uint8)

    return stretched
