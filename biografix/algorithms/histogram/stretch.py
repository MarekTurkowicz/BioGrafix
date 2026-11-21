"""Histogram stretching (luminance and per RGB channel)."""

import numpy as np

from ._common import gray_image, luminance_levels


def histogram_stretch(image: np.ndarray) -> np.ndarray:
    """Stretch luminance linearly so that min → 0 and max → 255; returns a grayscale image."""
    levels = luminance_levels(image)
    level_min, level_max = int(levels.min()), int(levels.max())
    if level_min >= level_max:
        return gray_image(levels)  # a uniform image cannot be stretched

    level = np.arange(256)
    scale = 255.0 / (level_max - level_min)
    lookup = np.clip(np.round((level - level_min) * scale), 0, 255)
    lookup[level <= level_min] = 0
    lookup[level >= level_max] = 255
    return gray_image(lookup[levels])


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
