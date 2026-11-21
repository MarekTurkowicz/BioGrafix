"""Histogram equalisation (luminance and per RGB channel)."""

import numpy as np

from ._common import gray_image, luminance_levels


def histogram_equalize(image: np.ndarray) -> np.ndarray:
    """Equalise the luminance histogram; returns a grayscale image."""
    levels = luminance_levels(image)
    cdf = np.cumsum(np.bincount(levels.ravel(), minlength=256))
    total = int(cdf[-1])
    cdf_min = int(cdf[cdf > 0][0])  # smallest non-zero value of the cumulative distribution
    if cdf_min == total:
        return gray_image(levels)  # a single-level image cannot be equalised

    share = np.clip((cdf - cdf_min) / (total - cdf_min), 0.0, 1.0)
    lookup = np.clip(np.round(share * 255), 0, 255)
    return gray_image(lookup[levels])


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
