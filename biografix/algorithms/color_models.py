"""RGB <-> CMYK colour model conversions."""


def _clamp(value, min_value, max_value):
    return max(min_value, min(max_value, value))


def rgb_to_cmyk(r, g, b):
    """Convert RGB (0–255) to CMYK, each channel in [0, 1]."""
    r = _clamp(int(r), 0, 255)
    g = _clamp(int(g), 0, 255)
    b = _clamp(int(b), 0, 255)

    if r == 0 and g == 0 and b == 0:
        # pure black
        return 0.0, 0.0, 0.0, 1.0

    r_f = r / 255.0
    g_f = g / 255.0
    b_f = b / 255.0

    k = 1.0 - max(r_f, g_f, b_f)
    if k >= 1.0:
        return 0.0, 0.0, 0.0, 1.0

    c = (1.0 - r_f - k) / (1.0 - k)
    m = (1.0 - g_f - k) / (1.0 - k)
    y = (1.0 - b_f - k) / (1.0 - k)

    c = _clamp(c, 0.0, 1.0)
    m = _clamp(m, 0.0, 1.0)
    y = _clamp(y, 0.0, 1.0)
    k = _clamp(k, 0.0, 1.0)

    return c, m, y, k


def cmyk_to_rgb(c, m, y, k):
    """Convert CMYK (each channel in [0, 1]) to RGB (0–255)."""
    c = _clamp(float(c), 0.0, 1.0)
    m = _clamp(float(m), 0.0, 1.0)
    y = _clamp(float(y), 0.0, 1.0)
    k = _clamp(float(k), 0.0, 1.0)

    r_f = (1.0 - c) * (1.0 - k)
    g_f = (1.0 - m) * (1.0 - k)
    b_f = (1.0 - y) * (1.0 - k)

    r = _clamp(int(round(r_f * 255.0)), 0, 255)
    g = _clamp(int(round(g_f * 255.0)), 0, 255)
    b = _clamp(int(round(b_f * 255.0)), 0, 255)

    return r, g, b
