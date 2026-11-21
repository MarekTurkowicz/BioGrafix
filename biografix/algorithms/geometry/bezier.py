"""Bézier curves (de Casteljau's algorithm)."""

from collections.abc import Sequence

from ._common import Point


def de_casteljau(points: Sequence[Point], t: float) -> Point:
    """Evaluate a Bézier curve at ``t`` with de Casteljau's algorithm."""
    pts = [(float(x), float(y)) for (x, y) in points]
    m = len(pts)
    for r in range(1, m):
        for i in range(m - r):
            x1, y1 = pts[i]
            x2, y2 = pts[i + 1]
            x = (1 - t) * x1 + t * x2
            y = (1 - t) * y1 + t * y2
            pts[i] = (x, y)
    return pts[0]


def bezier_points(points: Sequence[Point], steps: int = 200) -> list[Point]:
    """Sample a Bézier curve into a list of (x, y) points."""
    if len(points) < 2:
        return []
    return [de_casteljau(points, i / steps) for i in range(steps + 1)]
