"""Mean (box) filter."""

from ._common import _conv_rgb


def filter_box_blur(pixels, w, h, size=3):
    """Mean (box) filter."""
    if size % 2 == 0 or size < 1:
        raise ValueError("The kernel size must be odd and >= 1.")
    n = size * size
    k = [[1.0 / n for _ in range(size)] for _ in range(size)]
    return _conv_rgb(pixels, w, h, k)
