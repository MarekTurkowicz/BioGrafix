"""Geometry: shape transforms in homogeneous coordinates."""

import math

from ..algorithms import geometry as alg
from ..core.params import FloatParam, IntParam
from ..core.registry import TARGET_SHAPE, operation

CAT = "Geometry"
KINDS = ("polygon", "bezier")


def _pivot_defaults(obj, context):
    """Default pivot: the point set with the mouse, or the shape centre."""
    pivot = context.get("pivot")
    cx, cy = pivot if pivot else alg.centroid(obj.points)
    return {"cx": round(cx, 1), "cy": round(cy, 1)}


def _transform(obj, M):
    obj.points = alg.apply_matrix(obj.points, M)


@operation("geom.translate", CAT, "Translate", target=TARGET_SHAPE, shape_kinds=KINDS,
           params=[FloatParam("dx", "dx", 20.0, step=1.0, decimals=1), FloatParam("dy", "dy", 20.0, step=1.0, decimals=1)])
def translate(obj, dx, dy):
    _transform(obj, alg.matrix_translate(dx, dy))


@operation("geom.rotate", CAT, "Rotate about a point", target=TARGET_SHAPE, shape_kinds=KINDS,
           params=[FloatParam("cx", "Pivot x", 0.0, step=1.0, decimals=1),
                   FloatParam("cy", "Pivot y", 0.0, step=1.0, decimals=1),
                   FloatParam("angle", "Angle [°]", 30.0, -360.0, 360.0, step=5.0, decimals=1)],
           dynamic_defaults=_pivot_defaults,
           description="Default pivot: the point set with the Rotate/Scale tool (Shift+click) or the shape centre.")
def rotate(obj, cx, cy, angle):
    _transform(obj, alg.matrix_rotate(cx, cy, math.radians(angle)))


@operation("geom.scale", CAT, "Scale about a point", target=TARGET_SHAPE, shape_kinds=KINDS,
           params=[FloatParam("cx", "Pivot x", 0.0, step=1.0, decimals=1),
                   FloatParam("cy", "Pivot y", 0.0, step=1.0, decimals=1),
                   FloatParam("k", "Scale k", 1.5, 0.01, 100.0, step=0.1)],
           dynamic_defaults=_pivot_defaults)
def scale(obj, cx, cy, k):
    _transform(obj, alg.matrix_scale(cx, cy, k))


@operation("geom.bezier_degree", CAT, "Change Bézier curve degree", target=TARGET_SHAPE,
           shape_kinds=("bezier",),
           params=[IntParam("n", "Degree n (points = n+1)", 3, 1, 10)],
           dynamic_defaults=lambda obj, ctx: {"n": max(1, min(10, obj.degree))},
           description="Extra points are dropped; missing ones are added along a line from the last point.")
def bezier_degree(obj, n):
    needed = n + 1
    pts = list(obj.points)
    if len(pts) > needed:
        pts = pts[:needed]
    else:
        # extend along the direction of the last segment
        (x0, y0), (x1, y1) = pts[-2], pts[-1]
        dx, dy = (x1 - x0) or 40.0, (y1 - y0)
        while len(pts) < needed:
            x1, y1 = x1 + dx * 0.5, y1 + dy * 0.5
            pts.append((x1, y1))
    obj.points = pts
