"""5×5 Gaussian blur."""

from ._common import _conv_rgb


def filter_gaussian(pixels, w, h):
    """5×5 Gaussian blur."""
    kernel = [
        [1, 4, 6, 4, 1],
        [4, 16, 24, 16, 4],
        [6, 24, 36, 24, 6],
        [4, 16, 24, 16, 4],
        [1, 4, 6, 4, 1],
    ]
    norm = 1.0 / 256.0
    kernel = [[v * norm for v in row] for row in kernel]
    return _conv_rgb(pixels, w, h, kernel)
