"""Percent Black: T chosen so that a given share of pixels becomes black."""

from ._common import _apply_threshold_to_pixels, _to_gray_list


def threshold_percent_black(pixels, percent_black):
    """Percent Black: choose T so that about ``percent_black`` % of pixels become black."""
    if percent_black < 0:
        percent_black = 0.0
    elif percent_black > 100:
        percent_black = 100.0

    vals = _to_gray_list(pixels)
    n = len(vals)
    if n == 0:
        return pixels[:]

    # cumulative histogram
    hist = [0] * 256
    for v in vals:
        hist[v] += 1

    target = n * (percent_black / 100.0)
    cumsum = 0
    T = 0
    for i in range(256):
        cumsum += hist[i]
        if cumsum >= target:
            T = i
            break

    return _apply_threshold_to_pixels(pixels, T)
