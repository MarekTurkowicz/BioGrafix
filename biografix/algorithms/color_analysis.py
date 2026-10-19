"""Colour analysis: share of pixels within an HSV range."""

import colorsys


def color_coverage(pixels, h_min=70.0, h_max=160.0, s_min=0.2, v_min=0.2):
    """Return the percentage of pixels inside the HSV range (green by default).

    Returns ``(percent, h_min, h_max)`` with the hue range normalised.
    """
    # clamp the hue range to [0, 360] and make sure min <= max
    h_min = max(0.0, min(360.0, h_min))
    h_max = max(0.0, min(360.0, h_max))
    if h_max < h_min:
        h_min, h_max = h_max, h_min

    total = len(pixels)
    if total == 0:
        raise ValueError("The image is empty.")

    count = 0
    for r, g, b in pixels:
        rn = r / 255.0
        gn = g / 255.0
        bn = b / 255.0
        h, s, v = colorsys.rgb_to_hsv(rn, gn, bn)
        h_deg = h * 360.0

        if h_min <= h_deg <= h_max and s >= s_min and v >= v_min:
            count += 1

    percent = 100.0 * count / total
    return percent, h_min, h_max
