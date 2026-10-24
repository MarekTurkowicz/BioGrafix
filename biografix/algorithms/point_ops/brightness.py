"""Brightness changes: constant offset, power and logarithmic transforms."""

import numpy as np

from .arithmetic import add_constant


def change_brightness(pixels, delta):
    """Change brightness by adding a constant."""
    return add_constant(pixels, delta)


def brighten(image: np.ndarray) -> np.ndarray:
    """Brighten with a power (square root) transform."""
    image_array = np.asarray(image, dtype=np.float32) / 255.0
    return np.clip((image_array ** 0.5) * 255, 0, 255).astype(np.uint8)


def darken(image: np.ndarray) -> np.ndarray:
    """Darken with a logarithmic transform."""
    image_array = np.asarray(image, dtype=np.float32) / 255.0
    return np.clip(np.log1p(image_array) * 255, 0, 255).astype(np.uint8)
