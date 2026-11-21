"""Highlighting minutiae on an image: one colour and marker shape per type."""

from collections.abc import Sequence

import numpy as np

from .crossing_number import Minutia, MinutiaType

MARKER_RADIUS = 4

# RGB colours chosen to stay distinct from each other and from black / white ridges
MINUTIA_COLOURS: dict[MinutiaType, tuple[int, int, int]] = {
    MinutiaType.ENDING: (230, 30, 30),          # red circle
    MinutiaType.BIFURCATION: (20, 160, 40),     # green square
    MinutiaType.CROSSING: (30, 90, 230),        # blue diamond
    MinutiaType.ISOLATED_POINT: (240, 150, 0),  # orange cross
}
REMOVED_COLOUR = (175, 175, 175)  # grey: removed at the fingerprint border


def marker_offsets(kind: MinutiaType, radius: int = MARKER_RADIUS) -> tuple[np.ndarray, np.ndarray]:
    """(dy, dx) offsets of the marker outline pixels around the minutia."""
    dy, dx = np.mgrid[-radius:radius + 1, -radius:radius + 1]
    match kind:
        case MinutiaType.ENDING:
            distance = np.hypot(dx, dy)
            outline = (distance <= radius + 0.5) & (distance > radius - 1)
        case MinutiaType.BIFURCATION:
            outline = np.maximum(abs(dx), abs(dy)) == radius
        case MinutiaType.CROSSING:
            outline = abs(dx) + abs(dy) == radius
        case MinutiaType.ISOLATED_POINT:
            outline = (dx == 0) | (dy == 0)
    return dy[outline], dx[outline]


def _draw_markers(out: np.ndarray, minutiae: Sequence[Minutia], colour_of, radius: int) -> None:
    """Draw the markers of ``minutiae`` into ``out`` in place; ``colour_of(minutia)`` gives the RGB colour."""
    height, width = out.shape[:2]
    offsets = {kind: marker_offsets(kind, radius) for kind in MinutiaType}  # 4 shapes, computed once
    for m in minutiae:
        dy, dx = offsets[m.type]
        ys, xs = dy + m.y, dx + m.x
        inside = (ys >= 0) & (ys < height) & (xs >= 0) & (xs < width)
        out[ys[inside], xs[inside]] = colour_of(m)


def draw_minutiae(image: np.ndarray, minutiae: Sequence[Minutia], removed: Sequence[Minutia] = (),
                  radius: int = MARKER_RADIUS) -> np.ndarray:
    """Copy of the RGB ``image`` with a marker around every minutia; ``removed`` ones are grey and drawn first."""
    out = np.array(image[:, :, :3], dtype=np.uint8, copy=True)
    _draw_markers(out, removed, lambda _m: REMOVED_COLOUR, radius)
    _draw_markers(out, minutiae, lambda m: MINUTIA_COLOURS[m.type], radius)
    return out
