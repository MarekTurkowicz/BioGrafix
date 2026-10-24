"""Convolution with a user-given kernel (pixel list)."""

from ._common import _conv_rgb


def filter_custom(pixels, w, h, kernel):
    """Convolve with an arbitrary kernel given as a list of rows."""
    return _conv_rgb(pixels, w, h, kernel)
