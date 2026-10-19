"""Scene objects: vector shapes and raster images (no Qt dependency).

Every object can move, report its bounding box and handles, apply a handle
drag, hit-test a point, describe itself as editable text parameters and
serialise itself to a dict.

Image pixel arrays are read-only; operations always create new arrays, so the
undo history can share them without copying.
"""

from __future__ import annotations

import base64
import io
import math
import uuid
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np
from PIL import Image

from ..i18n import tr
from ..algorithms.geometry import (
    bezier_points,
    bresenham_circle,
    bresenham_line,
    distance_to_segment,
    point_in_polygon,
    rect_outline,
)
from ..constants import COL_CIRC, COL_LINE, COL_RECT

Point = Tuple[float, float]


# ---------------------------------------------------------------- helpers --

def new_oid() -> str:
    return uuid.uuid4().hex[:8]


def parts(txt: str, n: int):
    """Parse ``n`` comma- or semicolon-separated numbers."""
    ps = [p.strip() for p in txt.replace(";", ",").split(",") if p.strip()]
    if len(ps) != n:
        raise ValueError(tr("Enter {n} numbers", n=n))
    return ps


def parse_points(raw: str) -> List[Point]:
    """Parse points given as ``x1,y1; x2,y2; …`` or one ``x,y`` pair per line."""
    pts = []
    for line in raw.strip().splitlines():
        line = line.strip()
        if not line:
            continue
        for part in line.split(";"):
            part = part.strip()
            if not part:
                continue
            for ch in "()[]":
                part = part.replace(ch, "")
            part = part.replace(" ", ",")
            values = [s for s in part.split(",") if s]
            if len(values) != 2:
                raise ValueError(tr("Invalid point: \"{part}\" (expected x,y)", part=part))
            pts.append((float(values[0]), float(values[1])))
    return pts


def format_points(points) -> str:
    return "\n".join(f"{x:.1f},{y:.1f}" for (x, y) in points)


def freeze(arr: np.ndarray) -> np.ndarray:
    """Return a read-only uint8 array that can be shared by history snapshots."""
    arr = np.ascontiguousarray(arr, dtype=np.uint8)
    if arr.flags.writeable:
        arr = arr.copy()
        arr.flags.writeable = False
    return arr


# ------------------------------------------------------------- base class --

@dataclass
class SceneObject:
    """Base class of scene objects."""

    kind = "object"
    kind_label = "Object"
    params_hint = ""

    def __post_init__(self):
        if not getattr(self, "oid", ""):
            self.oid = new_oid()

    # --- implemented by subclasses ---
    def move(self, dx: float, dy: float) -> None: ...
    def bbox(self) -> Tuple[float, float, float, float]: ...
    def handles(self) -> List[Tuple[float, float, str]]: return []
    def apply_handle(self, kind: str, x: float, y: float) -> None: ...
    def hit_test(self, x: float, y: float, tol: float) -> bool: return False
    def params_text(self) -> str: return ""
    def set_params_text(self, txt: str) -> None: ...
    def to_dict(self) -> dict: ...

    def clone(self):
        """Snapshot copy: point lists are copied, pixel arrays are shared."""
        import copy

        return copy.copy(self)

    @property
    def display_name(self) -> str:
        return self.name or f"{tr(self.kind_label)} {self.oid[:4]}"


def _bbox_contains(b, x, y, margin=0.0) -> bool:
    return b[0] - margin <= x <= b[2] + margin and b[1] - margin <= y <= b[3] + margin


# ------------------------------------------------------------------- line --

@dataclass
class LineShape(SceneObject):
    x1: int = 0
    y1: int = 0
    x2: int = 0
    y2: int = 0
    color: str = COL_LINE
    width: int = 1           # stroke width in pixels
    name: str = ""
    oid: str = ""

    kind = "line"
    kind_label = "Line"
    params_hint = "x1,y1,x2,y2"

    def raster_points(self):
        return bresenham_line(int(self.x1), int(self.y1), int(self.x2), int(self.y2))

    def move(self, dx, dy):
        self.x1 += int(dx)
        self.y1 += int(dy)
        self.x2 += int(dx)
        self.y2 += int(dy)

    def bbox(self):
        x1, x2 = sorted([self.x1, self.x2])
        y1, y2 = sorted([self.y1, self.y2])
        return (x1, y1, x2 + 1, y2 + 1)

    def handles(self):
        return [(self.x1, self.y1, "p1"), (self.x2, self.y2, "p2")]

    def apply_handle(self, kind, x, y):
        if kind == "p1":
            self.x1, self.y1 = int(round(x)), int(round(y))
        elif kind == "p2":
            self.x2, self.y2 = int(round(x)), int(round(y))

    def hit_test(self, x, y, tol):
        return distance_to_segment((x, y), (self.x1 + 0.5, self.y1 + 0.5), (self.x2 + 0.5, self.y2 + 0.5)) <= tol

    def params_text(self):
        return f"{self.x1},{self.y1},{self.x2},{self.y2}"

    def set_params_text(self, txt):
        x1, y1, x2, y2 = map(int, parts(txt, 4))
        self.x1, self.y1, self.x2, self.y2 = x1, y1, x2, y2

    def to_dict(self):
        return {"type": "line", "name": self.name, "x1": self.x1, "y1": self.y1, "x2": self.x2, "y2": self.y2,
                "color": self.color, "width": self.width}


# -------------------------------------------------------------- rectangle --

@dataclass
class RectShape(SceneObject):
    x1: int = 0
    y1: int = 0
    x2: int = 0
    y2: int = 0
    color: str = COL_RECT
    width: int = 1           # stroke width in pixels
    name: str = ""
    oid: str = ""

    kind = "rect"
    kind_label = "Rectangle"
    params_hint = "x1,y1,x2,y2"

    def _norm(self):
        self.x1, self.x2 = sorted([self.x1, self.x2])
        self.y1, self.y2 = sorted([self.y1, self.y2])

    def raster_points(self):
        self._norm()
        return rect_outline(int(self.x1), int(self.y1), int(self.x2), int(self.y2))

    def move(self, dx, dy):
        self.x1 += int(dx)
        self.y1 += int(dy)
        self.x2 += int(dx)
        self.y2 += int(dy)

    def bbox(self):
        self._norm()
        return (self.x1, self.y1, self.x2 + 1, self.y2 + 1)

    def handles(self):
        self._norm()
        return [(self.x1, self.y1, "tl"), (self.x2, self.y1, "tr"), (self.x2, self.y2, "br"), (self.x1, self.y2, "bl")]

    def apply_handle(self, kind, x, y):
        x, y = int(round(x)), int(round(y))
        if kind == "tl":
            self.x1, self.y1 = x, y
        elif kind == "tr":
            self.x2, self.y1 = x, y
        elif kind == "br":
            self.x2, self.y2 = x, y
        elif kind == "bl":
            self.x1, self.y2 = x, y

    def hit_test(self, x, y, tol):
        b = self.bbox()
        inner = (b[0] + tol, b[1] + tol, b[2] - tol, b[3] - tol)
        inside_inner = inner[0] < x < inner[2] and inner[1] < y < inner[3]
        return _bbox_contains(b, x, y, tol) and not inside_inner

    def params_text(self):
        return f"{self.x1},{self.y1},{self.x2},{self.y2}"

    def set_params_text(self, txt):
        x1, y1, x2, y2 = map(int, parts(txt, 4))
        self.x1, self.y1, self.x2, self.y2 = x1, y1, x2, y2
        self._norm()

    def to_dict(self):
        return {"type": "rect", "name": self.name, "x1": self.x1, "y1": self.y1, "x2": self.x2, "y2": self.y2,
                "color": self.color, "width": self.width}


# ----------------------------------------------------------------- circle --

@dataclass
class CircleShape(SceneObject):
    cx: int = 0
    cy: int = 0
    r: int = 1
    color: str = COL_CIRC
    width: int = 1           # stroke width in pixels
    name: str = ""
    oid: str = ""

    kind = "circle"
    kind_label = "Circle"
    params_hint = "cx,cy,r"

    def raster_points(self):
        return bresenham_circle(int(self.cx), int(self.cy), int(self.r))

    def move(self, dx, dy):
        self.cx += int(dx)
        self.cy += int(dy)

    def bbox(self):
        return (self.cx - self.r, self.cy - self.r, self.cx + self.r + 1, self.cy + self.r + 1)

    def handles(self):
        return [(self.cx, self.cy, "center"), (self.cx + self.r, self.cy, "radius")]

    def apply_handle(self, kind, x, y):
        if kind == "center":
            self.cx, self.cy = int(round(x)), int(round(y))
        elif kind == "radius":
            self.r = max(1, int(round(math.hypot(x - self.cx, y - self.cy))))

    def hit_test(self, x, y, tol):
        return abs(math.hypot(x - self.cx - 0.5, y - self.cy - 0.5) - self.r) <= tol

    def params_text(self):
        return f"{self.cx},{self.cy},{self.r}"

    def set_params_text(self, txt):
        cx, cy, r = map(int, parts(txt, 3))
        if r <= 0:
            raise ValueError(tr("Radius must be > 0"))
        self.cx, self.cy, self.r = cx, cy, r

    def to_dict(self):
        return {"type": "circle", "name": self.name, "cx": self.cx, "cy": self.cy, "r": self.r, "color": self.color, "width": self.width}


# ---------------------------------------------------------- point shapes --

@dataclass
class _PointsShape(SceneObject):
    points: List[Point] = field(default_factory=list)
    color: str = ""          # empty: theme default
    width: int = 1           # stroke width in pixels
    name: str = ""
    oid: str = ""

    min_points = 2

    def clone(self):
        c = super().clone()
        c.points = list(self.points)
        return c

    def move(self, dx, dy):
        self.points = [(x + dx, y + dy) for (x, y) in self.points]

    def bbox(self):
        if not self.points:
            return (0, 0, 0, 0)
        xs = [p[0] for p in self.points]
        ys = [p[1] for p in self.points]
        return (min(xs), min(ys), max(xs), max(ys))

    def apply_handle(self, kind, x, y):
        i = int(kind[1:])
        if 0 <= i < len(self.points):
            self.points[i] = (x, y)

    def params_text(self):
        return format_points(self.points)

    def set_params_text(self, txt):
        pts = parse_points(txt)
        if len(pts) < self.min_points:
            raise ValueError(tr("At least {n} points are needed.", n=self.min_points))
        self.points = pts


@dataclass
class PolygonShape(_PointsShape):
    closed: bool = True

    kind = "polygon"
    kind_label = "Polygon"
    params_hint = "vertices: x,y on separate lines (or x1,y1; x2,y2; …)"
    min_points = 3

    def handles(self):
        return [(x, y, f"v{i}") for i, (x, y) in enumerate(self.points)]

    def edges(self):
        n = len(self.points)
        count = n if self.closed else n - 1
        return [(self.points[i], self.points[(i + 1) % n]) for i in range(max(0, count))]

    def hit_test(self, x, y, tol):
        if self.closed and len(self.points) >= 3 and point_in_polygon((x, y), self.points):
            return True
        return any(distance_to_segment((x, y), a, b) <= tol for a, b in self.edges())

    def to_dict(self):
        return {"type": "polygon", "name": self.name, "points": [list(p) for p in self.points], "closed": self.closed,
                "color": self.color, "width": self.width}


@dataclass
class BezierShape(_PointsShape):
    kind = "bezier"
    kind_label = "Bézier curve"
    params_hint = "control points: x,y on separate lines (degree = number of points − 1)"
    min_points = 2

    @property
    def degree(self) -> int:
        return len(self.points) - 1

    def curve(self, steps=200):
        return bezier_points(self.points, steps)

    def handles(self):
        return [(x, y, f"p{i}") for i, (x, y) in enumerate(self.points)]

    def hit_test(self, x, y, tol):
        pts = self.curve(100)
        if any(distance_to_segment((x, y), pts[i], pts[i + 1]) <= tol for i in range(len(pts) - 1)):
            return True
        return any(math.hypot(x - px, y - py) <= tol * 1.5 for px, py in self.points)

    def to_dict(self):
        return {"type": "bezier", "name": self.name, "points": [list(p) for p in self.points], "color": self.color, "width": self.width}


# ------------------------------------------------------------------ image --

@dataclass
class RasterImage(SceneObject):
    """A raster image placed on the scene.

    Attributes:
        pixels: current (H, W, 3) or (H, W, 4) array, read-only.
        original: pixels as loaded, used by "Restore original".
        w, h: displayed size (nearest-neighbour scaling).
        roi: selected region ``(x0, y0, x1, y1)`` in image pixels, or None.
    """

    x: float = 0
    y: float = 0
    pixels: Optional[np.ndarray] = None
    original: Optional[np.ndarray] = None
    w: Optional[int] = None
    h: Optional[int] = None
    src: Optional[str] = None
    roi: Optional[Tuple[int, int, int, int]] = None
    name: str = ""
    oid: str = ""

    kind = "image"
    kind_label = "Image"
    params_hint = "x,y,w,h (position and displayed size)"

    def __post_init__(self):
        super().__post_init__()
        self.pixels = freeze(self.pixels)
        self.original = self.pixels if self.original is None else freeze(self.original)
        if self.w is None:
            self.w = self.src_w
        if self.h is None:
            self.h = self.src_h

    @property
    def src_w(self) -> int:
        return int(self.pixels.shape[1])

    @property
    def src_h(self) -> int:
        return int(self.pixels.shape[0])

    def set_pixels(self, arr: np.ndarray) -> None:
        """Replace the pixels; a size change also resets the displayed size and the region."""
        old = (self.src_w, self.src_h)
        self.pixels = freeze(arr)
        if (self.src_w, self.src_h) != old:
            sx = self.w / old[0] if old[0] else 1
            sy = self.h / old[1] if old[1] else 1
            self.w = max(1, int(round(self.src_w * sx)))
            self.h = max(1, int(round(self.src_h * sy)))
            self.roi = None

    def scene_to_pixel(self, sx: float, sy: float) -> Tuple[int, int]:
        """Map scene coordinates to source pixel indices (may fall outside the image)."""
        px = math.floor((sx - self.x) * self.src_w / self.w)
        py = math.floor((sy - self.y) * self.src_h / self.h)
        return px, py

    def pixel_at(self, sx: float, sy: float):
        px, py = self.scene_to_pixel(sx, sy)
        if 0 <= px < self.src_w and 0 <= py < self.src_h:
            return px, py, tuple(int(v) for v in self.pixels[py, px])
        return None

    def _clamp_dims(self):
        self.w = max(1, int(self.w))
        self.h = max(1, int(self.h))

    def move(self, dx, dy):
        self.x += dx
        self.y += dy

    def bbox(self):
        return (self.x, self.y, self.x + self.w, self.y + self.h)

    def handles(self):
        x1, y1, x2, y2 = self.bbox()
        return [(x1, y1, "tl"), (x2, y1, "tr"), (x2, y2, "br"), (x1, y2, "bl")]

    def apply_handle(self, kind, mx, my):
        # free scaling, aspect ratio not preserved
        x1, y1, x2, y2 = self.bbox()
        if kind == "tl":
            self.x, self.y = mx, my
            self.w, self.h = x2 - self.x, y2 - self.y
        elif kind == "tr":
            self.y = my
            self.w, self.h = mx - self.x, y2 - self.y
        elif kind == "br":
            self.w, self.h = mx - self.x, my - self.y
        elif kind == "bl":
            self.x = mx
            self.w, self.h = x2 - self.x, my - self.y
        if self.w < 0:
            self.x += self.w
            self.w = -self.w
        if self.h < 0:
            self.y += self.h
            self.h = -self.h
        self.w, self.h = int(round(self.w)), int(round(self.h))
        self._clamp_dims()

    def hit_test(self, x, y, tol):
        return _bbox_contains(self.bbox(), x, y)

    def params_text(self):
        return f"{round(self.x)},{round(self.y)},{self.w},{self.h}"

    def set_params_text(self, txt):
        x, y, w, h = map(int, parts(txt, 4))
        self.x, self.y = x, y
        self.w, self.h = w, h
        self._clamp_dims()

    def to_dict(self):
        d = {
            "type": "image", "name": self.name, "x": self.x, "y": self.y, "w": self.w, "h": self.h,
            "pixels_png": _encode_png(self.pixels),
        }
        if self.original is not self.pixels:
            d["original_png"] = _encode_png(self.original)
        if self.src:
            d["src"] = self.src
        return d


def _encode_png(arr: np.ndarray) -> str:
    buf = io.BytesIO()
    Image.fromarray(np.asarray(arr)).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _decode_png(data: str) -> np.ndarray:
    img = Image.open(io.BytesIO(base64.b64decode(data)))
    return np.asarray(img.convert("RGBA" if img.mode in ("RGBA", "LA") else "RGB")).copy()


# ---------------------------------------------------------- JSON factory --

def object_from_dict(d: dict) -> SceneObject:
    t = d.get("type")
    name = d.get("name", "")
    color = {"color": d["color"]} if d.get("color") else {}
    if d.get("width"):
        color["width"] = int(d["width"])
    if t == "line":
        return LineShape(d["x1"], d["y1"], d["x2"], d["y2"], name=name, **color)
    if t == "rect":
        return RectShape(d["x1"], d["y1"], d["x2"], d["y2"], name=name, **color)
    if t == "circle":
        return CircleShape(d["cx"], d["cy"], d["r"], name=name, **color)
    if t == "polygon":
        return PolygonShape(points=[tuple(p) for p in d["points"]], closed=d.get("closed", True), name=name, **color)
    if t == "bezier":
        return BezierShape(points=[tuple(p) for p in d["points"]], name=name, **color)
    if t == "image":
        if "pixels_png" in d:
            pixels = _decode_png(d["pixels_png"])
        elif d.get("src"):
            # legacy format: file path only
            from ..io.images import load_image

            pixels = load_image(d["src"])
        else:
            pixels = np.full((int(d.get("h", 64)), int(d.get("w", 64)), 3), 200, np.uint8)
        original = _decode_png(d["original_png"]) if "original_png" in d else None
        return RasterImage(
            x=d.get("x", 0), y=d.get("y", 0), pixels=pixels, original=original,
            w=d.get("w"), h=d.get("h"), src=d.get("src"), name=name,
        )
    raise ValueError(f"Unknown object type: {t}")


SHAPE_TYPES = {
    "line": LineShape,
    "rect": RectShape,
    "circle": CircleShape,
    "polygon": PolygonShape,
    "bezier": BezierShape,
}
