"""Geometry: Bresenham rasterisation, Bézier curves, homogeneous transforms and 3D rotation."""

import math
from typing import List, Sequence, Tuple

Point = Tuple[float, float]
Matrix3 = List[List[float]]


# ---------------------------------------------------------------- Bresenham --

def bresenham_line(x1: int, y1: int, x2: int, y2: int) -> List[Tuple[int, int]]:
    points = []
    dx, dy = abs(x2 - x1), abs(y2 - y1)
    sx, sy = (1 if x1 < x2 else -1), (1 if y1 < y2 else -1)
    err = dx - dy
    while True:
        points.append((x1, y1))
        if x1 == x2 and y1 == y2:
            break
        e2 = err << 1
        if e2 > -dy:
            err -= dy
            x1 += sx
        if e2 < dx:
            err += dx
            y1 += sy
    return points


def bresenham_circle(cx: int, cy: int, r: int) -> List[Tuple[int, int]]:
    if r <= 0:
        return []
    pts = []
    x, y = 0, r
    d = 3 - 2 * r

    def add8(cx, cy, x, y):
        pts.extend(
            [
                (cx + x, cy + y),
                (cx - x, cy + y),
                (cx + x, cy - y),
                (cx - x, cy - y),
                (cx + y, cy + x),
                (cx - y, cy + x),
                (cx + y, cy - x),
                (cx - y, cy - x),
            ]
        )

    while x <= y:
        add8(cx, cy, x, y)
        if d < 0:
            d += 4 * x + 6
        else:
            d += 4 * (x - y) + 10
            y -= 1
        x += 1

    # small radii produce duplicate points
    return list(dict.fromkeys(pts))


def rect_outline(x1: int, y1: int, x2: int, y2: int) -> List[Tuple[int, int]]:
    """Rectangle outline made of four Bresenham segments."""
    pts = []
    pts += bresenham_line(x1, y1, x2, y1)
    pts += bresenham_line(x2, y1, x2, y2)
    pts += bresenham_line(x2, y2, x1, y2)
    pts += bresenham_line(x1, y2, x1, y1)
    return list(dict.fromkeys(pts))


# ------------------------------------------------------------------- Bézier --

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


def bezier_points(points: Sequence[Point], steps: int = 200) -> List[Point]:
    """Sample a Bézier curve into a list of (x, y) points."""
    if len(points) < 2:
        return []
    return [de_casteljau(points, i / steps) for i in range(steps + 1)]


# ------------------------------------------- 2D homogeneous coordinates --
# A point (x, y) is the vector [x, y, 1]^T multiplied by a 3×3 matrix.

def apply_matrix(points: Sequence[Point], M: Matrix3) -> List[Point]:
    """Return the points transformed by a 3×3 matrix."""
    res = []
    for x, y in points:
        vx = M[0][0] * x + M[0][1] * y + M[0][2] * 1.0
        vy = M[1][0] * x + M[1][1] * y + M[1][2] * 1.0
        vw = M[2][0] * x + M[2][1] * y + M[2][2] * 1.0
        if abs(vw) < 1e-9:
            vw = 1.0
        res.append((vx / vw, vy / vw))
    return res


def matrix_translate(dx: float, dy: float) -> Matrix3:
    return [
        [1.0, 0.0, dx],
        [0.0, 1.0, dy],
        [0.0, 0.0, 1.0],
    ]


def matrix_rotate(cx: float, cy: float, angle_rad: float) -> Matrix3:
    c = math.cos(angle_rad)
    s = math.sin(angle_rad)
    # T(cx,cy) * R * T(-cx,-cy)
    return [
        [c, -s, cx - c * cx + s * cy],
        [s, c, cy - s * cx - c * cy],
        [0, 0, 1],
    ]


def matrix_scale(cx: float, cy: float, k: float) -> Matrix3:
    # T(cx,cy) * S(k,k) * T(-cx,-cy)
    return [
        [k, 0, cx - k * cx],
        [0, k, cy - k * cy],
        [0, 0, 1],
    ]


def point_in_polygon(pt: Point, points: Sequence[Point]) -> bool:
    """Point-in-polygon test (ray casting)."""
    x, y = pt
    inside = False
    n = len(points)
    for i in range(n):
        x1, y1 = points[i]
        x2, y2 = points[(i + 1) % n]
        if ((y1 > y) != (y2 > y)) and (x < (x2 - x1) * (y - y1) / (y2 - y1 + 1e-9) + x1):
            inside = not inside
    return inside


def distance_to_segment(p: Point, a: Point, b: Point) -> float:
    px, py = p
    ax, ay = a
    bx, by = b
    dx, dy = bx - ax, by - ay
    if dx == 0 and dy == 0:
        return math.hypot(px - ax, py - ay)
    t = max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / (dx * dx + dy * dy)))
    return math.hypot(px - (ax + t * dx), py - (ay + t * dy))


def centroid(points: Sequence[Point]) -> Point:
    if not points:
        return (0.0, 0.0)
    return (sum(p[0] for p in points) / len(points), sum(p[1] for p in points) / len(points))


# -------------------------------------------------------------- 3D rotation --

def rotate_point_3d(x, y, z, ax, ay, az):
    """Rotate a point about the X, Y and Z axes in turn (angles in radians)."""
    cosx, sinx = math.cos(ax), math.sin(ax)
    y1 = y * cosx - z * sinx
    z1 = y * sinx + z * cosx
    x1 = x

    cosy, siny = math.cos(ay), math.sin(ay)
    x2 = x1 * cosy + z1 * siny
    z2 = -x1 * siny + z1 * cosy
    y2 = y1

    cosz, sinz = math.cos(az), math.sin(az)
    x3 = x2 * cosz - y2 * sinz
    y3 = x2 * sinz + y2 * cosz
    z3 = z2

    return x3, y3, z3
