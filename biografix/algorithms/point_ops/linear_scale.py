"""Levels: linear stretch of a value range to 0..255."""

import numpy as np

from ._common import rgb_channels, round_to_byte


def linear_color_scale(image: np.ndarray, in_min: int, in_max: int) -> np.ndarray:
    """Levels: stretch [in_min, in_max] linearly to [0, 255]."""
    in_min = max(0, min(255, int(in_min)))
    in_max = max(0, min(255, int(in_max)))
    rgb = rgb_channels(image)
    if in_max <= in_min:
        return rgb.copy()  # empty input range
    k = 255.0 / (in_max - in_min)
    return round_to_byte((rgb.astype(np.int32) - in_min) * k)
