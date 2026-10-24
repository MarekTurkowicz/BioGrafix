"""Conversion to grayscale."""

import numpy as np

from ._common import _clamp_byte


def to_grayscale_avg(pixels):
    """Grayscale as the channel average (R+G+B)/3."""
    out = []
    for r, g, b in pixels:
        gval = _clamp_byte((r + g + b) / 3.0)
        out.append((gval, gval, gval))
    return out


def to_grayscale_luma(pixels):
    """Grayscale as weighted luminance 0.299R + 0.587G + 0.114B."""
    out = []
    for r, g, b in pixels:
        gval = _clamp_byte(0.299 * r + 0.587 * g + 0.114 * b)
        out.append((gval, gval, gval))
    return out


def to_grayscale(image_array: np.ndarray, mode: str = "average") -> np.ndarray:
    """Convert an image to a (H, W) uint8 grayscale array using ``mode``."""
    image_array = np.asarray(image_array, dtype=np.float32)

    if mode == "red":
        grayscale = image_array[:, :, 0]
    elif mode == "green":
        grayscale = image_array[:, :, 1]
    elif mode == "blue":
        grayscale = image_array[:, :, 2]
    elif mode == "average":
        grayscale = image_array.mean(axis=2)
    else:
        raise ValueError(f"Unknown grayscale mode: {mode}")

    return np.clip(grayscale, 0, 255).astype(np.uint8)
