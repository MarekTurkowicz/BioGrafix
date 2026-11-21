"""Helpers shared by the point operations."""


Color = tuple[int, int, int]


def clamp(v: int) -> int:
    return 0 if v < 0 else 255 if v > 255 else v


def _clamp_byte(v: float) -> int:
    v = int(round(v))
    if v < 0:
        return 0
    if v > 255:
        return 255
    return v
