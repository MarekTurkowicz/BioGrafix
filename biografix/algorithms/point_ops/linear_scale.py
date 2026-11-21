"""Levels: linear stretch of a value range to 0..255."""


from ._common import Color, clamp


def linear_color_scale(pixels: list[Color], in_min: int, in_max: int) -> list[Color]:
    """Levels: stretch [in_min, in_max] linearly to [0, 255]."""
    in_min = max(0, min(255, int(in_min)))
    in_max = max(0, min(255, int(in_max)))
    if in_max <= in_min:
        return pixels[:]  # empty input range
    k = 255.0 / (in_max - in_min)
    out: list[Color] = []
    for r, g, b in pixels:
        rr = clamp(int(round((r - in_min) * k)))
        gg = clamp(int(round((g - in_min) * k)))
        bb = clamp(int(round((b - in_min) * k)))
        out.append((rr, gg, bb))
    return out
