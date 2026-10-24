"""Adding, multiplying and dividing by a constant."""

from ._common import _clamp_byte


def add_constant(pixels, value):
    """Add a constant to every channel, clipped to 0..255."""
    out = []
    for r, g, b in pixels:
        out.append((_clamp_byte(r + value), _clamp_byte(g + value), _clamp_byte(b + value)))
    return out


def mul_constant(pixels, value):
    """Multiply every channel by a constant."""
    out = []
    for r, g, b in pixels:
        out.append((_clamp_byte(r * value), _clamp_byte(g * value), _clamp_byte(b * value)))
    return out


def div_constant(pixels, value):
    """Divide every channel by a non-zero constant."""
    if value == 0:
        raise ValueError("Division by zero is not allowed.")
    out = []
    for r, g, b in pixels:
        out.append((_clamp_byte(r / value), _clamp_byte(g / value), _clamp_byte(b / value)))
    return out
