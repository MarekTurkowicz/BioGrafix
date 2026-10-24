"""2D homogeneous transforms.

A point (x, y) is the vector [x, y, 1]^T multiplied by a 3×3 matrix.
"""

import math
from typing import List, Sequence

from ._common import Matrix3, Point


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
