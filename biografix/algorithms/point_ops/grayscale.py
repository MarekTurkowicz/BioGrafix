"""Conversion to grayscale."""

import numpy as np

from ._common import rgb_channels, round_to_byte


def _gray_to_rgb(gray: np.ndarray) -> np.ndarray:
    return np.stack([gray] * 3, axis=-1)


def to_grayscale_avg(image: np.ndarray) -> np.ndarray:
    """Grayscale RGB image from the channel average (R+G+B)/3."""
    rgb = rgb_channels(image).astype(np.int32)
    return _gray_to_rgb(round_to_byte(rgb.sum(axis=2) / 3.0))


def luminance(image: np.ndarray) -> np.ndarray:
    """Weighted luminance 0.299R + 0.587G + 0.114B as float."""
    rgb = rgb_channels(image).astype(np.float64)
    return 0.299 * rgb[:, :, 0] + 0.587 * rgb[:, :, 1] + 0.114 * rgb[:, :, 2]


def luminance_levels(image: np.ndarray) -> np.ndarray:
    """Integer luminance 0..255 of every pixel (rounded, halves to even)."""
    return np.round(np.clip(luminance(image), 0, 255)).astype(np.intp)


def to_grayscale_luma(image: np.ndarray) -> np.ndarray:
    """Grayscale RGB image from the weighted luminance."""
    return _gray_to_rgb(round_to_byte(luminance(image)))


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
