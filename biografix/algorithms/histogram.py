"""Histogram computation, stretching and equalisation.

- Pixel-list functions work on luminance and return grayscale images.
- numpy functions process each RGB channel independently.
"""

import numpy as np


# ============================================================ pixel lists ==

def _luma(r, g, b):
    """Luminance in the range 0..255."""
    y = 0.299 * r + 0.587 * g + 0.114 * b
    if y < 0:
        y = 0
    elif y > 255:
        y = 255
    return int(round(y))


def compute_histogram(pixels):
    """Return a 256-bin luminance histogram of ``pixels``."""
    hist = [0] * 256
    for r, g, b in pixels:
        y = _luma(r, g, b)
        hist[y] += 1
    return hist


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


# ============================================================ numpy arrays ==

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
