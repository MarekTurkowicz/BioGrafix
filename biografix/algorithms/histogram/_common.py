"""Helpers shared by the histogram operations."""


def _luma(r, g, b):
    """Luminance in the range 0..255."""
    y = 0.299 * r + 0.587 * g + 0.114 * b
    if y < 0:
        y = 0
    elif y > 255:
        y = 255
    return int(round(y))
