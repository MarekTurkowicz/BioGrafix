"""Adapters that run pixel-list algorithms on numpy images."""

from ..core.imaging import array_to_pixels, pixels_to_array


def run_on_pixels(func, image, *args, **kwargs):
    """Call ``f(pixels, *args)`` and convert the result back to an array."""
    pixels, w, h = array_to_pixels(image)
    return pixels_to_array(func(pixels, *args, **kwargs), w, h)


def run_on_pixels_wh(func, image, *args, **kwargs):
    """Call ``f(pixels, w, h, *args)`` and convert the result back to an array."""
    pixels, w, h = array_to_pixels(image)
    return pixels_to_array(func(pixels, w, h, *args, **kwargs), w, h)
