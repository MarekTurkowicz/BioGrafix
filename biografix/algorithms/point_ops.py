"""Point operations on pixel lists and numpy arrays."""

from typing import List, Tuple

import numpy as np

Color = Tuple[int, int, int]


# ============================================================ pixel lists ==

def clamp(v: int) -> int:
    return 0 if v < 0 else 255 if v > 255 else v


def linear_color_scale(pixels: List[Color], in_min: int, in_max: int) -> List[Color]:
    """Levels: stretch [in_min, in_max] linearly to [0, 255]."""
    in_min = max(0, min(255, int(in_min)))
    in_max = max(0, min(255, int(in_max)))
    if in_max <= in_min:
        return pixels[:]  # empty input range
    k = 255.0 / (in_max - in_min)
    out: List[Color] = []
    for r, g, b in pixels:
        rr = clamp(int(round((r - in_min) * k)))
        gg = clamp(int(round((g - in_min) * k)))
        bb = clamp(int(round((b - in_min) * k)))
        out.append((rr, gg, bb))
    return out


def _clamp_byte(v: float) -> int:
    v = int(round(v))
    if v < 0:
        return 0
    if v > 255:
        return 255
    return v


def add_constant(pixels, value):
    """Add a constant to every channel, clipped to 0..255."""
    out = []
    for r, g, b in pixels:
        out.append((_clamp_byte(r + value), _clamp_byte(g + value), _clamp_byte(b + value)))
    return out


def mul_constant(pixels, value):
    """Multiply every channel by a constant."""
    out = []
    for r, g, b in pixels:
        out.append((_clamp_byte(r * value), _clamp_byte(g * value), _clamp_byte(b * value)))
    return out


def div_constant(pixels, value):
    """Divide every channel by a non-zero constant."""
    if value == 0:
        raise ValueError("Division by zero is not allowed.")
    out = []
    for r, g, b in pixels:
        out.append((_clamp_byte(r / value), _clamp_byte(g / value), _clamp_byte(b / value)))
    return out


def change_brightness(pixels, delta):
    """Change brightness by adding a constant."""
    return add_constant(pixels, delta)


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


# ============================================================ numpy arrays ==

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


def brighten(image: np.ndarray) -> np.ndarray:
    """Brighten with a power (square root) transform."""
    image_array = np.asarray(image, dtype=np.float32) / 255.0
    return np.clip((image_array ** 0.5) * 255, 0, 255).astype(np.uint8)


def darken(image: np.ndarray) -> np.ndarray:
    """Darken with a logarithmic transform."""
    image_array = np.asarray(image, dtype=np.float32) / 255.0
    return np.clip(np.log1p(image_array) * 255, 0, 255).astype(np.uint8)


def fill_color(image: np.ndarray, r: int, g: int, b: int) -> np.ndarray:
    """Fill the image with a single colour."""
    out = np.empty_like(image)
    out[:, :] = (r, g, b)
    return out
