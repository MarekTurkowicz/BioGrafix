"""High-pass sharpening filter."""

from ._common import _conv_rgb


def filter_sharpen(pixels, w, h):
    """High-pass sharpening filter."""
    kernel = [
        [0, -1, 0],
        [-1, 5, -1],
        [0, -1, 0],
    ]
    return _conv_rgb(pixels, w, h, kernel)
