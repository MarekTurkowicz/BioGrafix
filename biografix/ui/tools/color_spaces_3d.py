"""3D colour space views: RGB cube and HSV cone.

The projection is computed manually (rotation about X/Y/Z and depth sorting).
Clicking the solid shows the corresponding cross-section.
"""

from __future__ import annotations

import colorsys
import math

from PySide6.QtCore import QPointF, QRectF, Qt, Signal
from PySide6.QtGui import QColor, QPainter, QPen
from PySide6.QtWidgets import QGridLayout, QGroupBox, QHBoxLayout, QLabel, QSlider, QVBoxLayout, QWidget

from ...algorithms.geometry import rotate_point_3d
from ...i18n import tr


class _Canvas(QWidget):
    """Canvas that draws rectangles or dots and reports clicks."""

    clicked = Signal(float, float)

    def __init__(self, size: int):
        super().__init__()
        self.setFixedSize(size, size)
        self.rects = []      # (x1, y1, x2, y2, QColor)
        self.outline = None  # ("rect"|"ellipse", QRectF)

    def paintEvent(self, event):
        p = QPainter(self)
        p.fillRect(self.rect(), QColor("#ffffff"))
        for x1, y1, x2, y2, color in self.rects:
            p.fillRect(QRectF(x1, y1, x2 - x1, y2 - y1), color)
        if self.outline:
            kind, r = self.outline
            p.setPen(QPen(QColor("#444444")))
            p.setBrush(Qt.BrushStyle.NoBrush)
            if kind == "rect":
                p.drawRect(r)
            else:
                p.drawEllipse(r)
        p.setPen(QPen(QColor("#cccccc")))
        p.drawRect(self.rect().adjusted(0, 0, -1, -1))

    def mousePressEvent(self, event):
        self.clicked.emit(event.position().x(), event.position().y())


class _Projection3DWindow(QWidget):
    """Base window: rotatable 3D solid with a cross-section view."""

    canvas_size = 360
    slice_size = 320
    slice_prompt = ""

    def __init__(self, title: str, parent=None, **config):
        super().__init__(parent, Qt.WindowType.Window)
        for key, value in config.items():  # subclass configuration: sizes and sampling density
            setattr(self, key, value)
        self.setWindowTitle(tr(title))
        self._points = []  # (sx, sy, cross-section data)

        main = QHBoxLayout(self)
        left = QVBoxLayout()
        self.canvas = _Canvas(self.canvas_size)
        self.canvas.clicked.connect(self._on_click)
        left.addWidget(self.canvas)

        rot = QGridLayout()
        self.angles = {}
        for row, (axis, val) in enumerate([("X", 30), ("Y", -30), ("Z", 0)]):
            s = QSlider(Qt.Orientation.Horizontal)
            s.setRange(-180, 180)
            s.setValue(val)
            s.valueChanged.connect(self.redraw)
            rot.addWidget(QLabel(f"Rot {axis}:"), row, 0)
            rot.addWidget(s, row, 1)
            self.angles[axis] = s
        left.addLayout(rot)
        self._build_extra_controls(left)
        main.addLayout(left)

        right = QVBoxLayout()
        self.slice_info = QLabel(tr(self.slice_prompt))
        right.addWidget(self.slice_info)
        self.slice_canvas = _Canvas(self.slice_size)
        right.addWidget(self.slice_canvas)
        right.addStretch(1)
        main.addLayout(right)

        self.redraw()

    def _build_extra_controls(self, layout):
        pass

    def _rad(self):
        return tuple(math.radians(self.angles[a].value()) for a in "XYZ")

    def redraw(self, *_):
        voxels, size_pt = self.build_voxels()
        voxels.sort(key=lambda t: t[0])  # painter's algorithm: far to near
        rects, self._points = [], []
        for zr, sx, sy, data, color in voxels:
            rects.append((sx - size_pt / 2, sy - size_pt / 2, sx + size_pt / 2, sy + size_pt / 2, QColor(color)))
            self._points.append((sx, sy, data))
        self.canvas.rects = rects
        self.canvas.update()

    def _on_click(self, ex, ey):
        if not self._points:
            return
        best = min(self._points, key=lambda t: (t[0] - ex) ** 2 + (t[1] - ey) ** 2)
        self.draw_slice(best[2])
        self.slice_canvas.update()

    # implemented by subclasses
    def build_voxels(self):
        raise NotImplementedError

    def draw_slice(self, data):
        raise NotImplementedError


# =============================================================== RGB cube ==

class RGBCubeWindow(_Projection3DWindow):
    """RGB cube, either sparse points (8³) or sliceable (10³ with R/G/B cut sliders)."""

    slice_prompt = "Cross-section: click the cube to choose R"

    def __init__(self, sliced: bool = False, parent=None):
        super().__init__("3D RGB cube (slices)" if sliced else "3D RGB cube (points)", parent,
                         sliced=sliced, steps=10 if sliced else 8)

    def _build_extra_controls(self, layout):
        self.clips = {}
        if not self.sliced:
            return
        box = QGroupBox(tr("Cube slicing (R/G/B layers)"))
        g = QGridLayout(box)
        for row, axis in enumerate(["R (X)", "G (Y)", "B (Z)"]):
            s = QSlider(Qt.Orientation.Horizontal)
            s.setRange(0, self.steps)
            s.setValue(self.steps)
            s.valueChanged.connect(self.redraw)
            g.addWidget(QLabel(axis + ":"), row, 0)
            g.addWidget(s, row, 1)
            self.clips[row] = s
        layout.addWidget(box)

    def build_voxels(self):
        w = h = self.canvas_size
        cx, cy = w / 2, h / 2
        scale = min(w, h) * (0.45 if self.sliced else 0.4)
        ax, ay, az = self._rad()
        steps = self.steps
        if self.sliced and self.clips:
            max_i, max_j, max_k = (max(0, min(steps, self.clips[i].value())) for i in range(3))
        else:
            max_i = max_j = max_k = steps

        voxels = []
        for i in range(min(steps, max_i)):
            r = int(round(i / (steps - 1) * 255))
            x = i / (steps - 1) - 0.5
            for j in range(min(steps, max_j)):
                g = int(round(j / (steps - 1) * 255))
                y = j / (steps - 1) - 0.5
                for k in range(min(steps, max_k)):
                    b = int(round(k / (steps - 1) * 255))
                    z = k / (steps - 1) - 0.5
                    xr, yr, zr = rotate_point_3d(x, y, z, ax, ay, az)
                    voxels.append((zr, cx + xr * scale, cy - yr * scale, i, f"#{r:02x}{g:02x}{b:02x}"))

        if self.sliced:
            size_pt = max(4, scale / (steps - 1) * 1.3)
        else:
            size_pt = 5
        return voxels, size_pt

    def draw_slice(self, r_index):
        steps = self.steps
        w = h = self.slice_size
        size = min(w, h) - 20
        cell = size / steps
        ox, oy = (w - size) / 2, (h - size) / 2
        r_val = int(round(r_index / (steps - 1) * 255))
        self.slice_info.setText(tr("Cross-section at R = {value} (0–255)", value=r_val))
        rects = []
        for j in range(steps):
            g_val = int(round(j / (steps - 1) * 255))
            for k in range(steps):
                b_val = int(round(k / (steps - 1) * 255))
                x1, y1 = ox + k * cell, oy + j * cell
                rects.append((x1, y1, x1 + cell, y1 + cell, QColor(r_val, g_val, b_val)))
        self.slice_canvas.rects = rects
        self.slice_canvas.outline = ("rect", QRectF(ox, oy, size, size))


# =============================================================== HSV cone ==

def _hsv_color(h, s, v) -> QColor:
    r, g, b = colorsys.hsv_to_rgb(h, s, v)
    return QColor(int(r * 255), int(g * 255), int(b * 255))


class HSVConeWindow(_Projection3DWindow):
    """HSV cone, either sparse points or a dense solid."""

    slice_prompt = "Cross-section: click the cone to choose V"

    def __init__(self, full: bool = False, parent=None):
        if full:
            config = dict(steps_v=14, steps_s=12, steps_h=48, canvas_size=380, slice_size=320)
        else:
            config = dict(steps_v=8, steps_s=6, steps_h=24, canvas_size=360, slice_size=300)
        super().__init__("3D HSV cone (solid)" if full else "3D HSV cone (points)", parent,
                         full=full, **config)

    def build_voxels(self):
        w = h = self.canvas_size
        cx, cy = w / 2, h / 2
        scale = min(w, h) * (0.48 if self.full else 0.45)
        ax, ay, az = self._rad()
        voxels = []
        for vi in range(self.steps_v):
            v = vi / (self.steps_v - 1)
            z = v - 0.5  # cone centred on the Z axis
            for si in range(self.steps_s):
                s = si / (self.steps_s - 1)
                r_cone = s * v  # radius grows with S and V
                for hi in range(self.steps_h):
                    h_norm = hi / self.steps_h
                    angle = 2 * math.pi * h_norm
                    x, y = r_cone * math.cos(angle), r_cone * math.sin(angle)
                    xr, yr, zr = rotate_point_3d(x, y, z, ax, ay, az)
                    voxels.append((zr, cx + xr * scale, cy - yr * scale, vi, _hsv_color(h_norm, s, v)))
        if self.full:
            size_pt = max(4, scale / max(self.steps_v, self.steps_s, 1) * 1.8)
        else:
            size_pt = 5
        return voxels, size_pt

    def draw_slice(self, vi):
        v = vi / (self.steps_v - 1)
        self.slice_info.setText(tr("Cross-section at V = {value} (0–1)", value=f"{v:.2f}"))
        w = h = self.slice_size
        cx, cy = w / 2, h / 2
        radius = min(w, h) / 2 - 10
        samples_s, samples_h = self.steps_s * 2, self.steps_h * 2
        rects = []
        for si in range(samples_s):
            s = si / (samples_s - 1)
            for hi in range(samples_h):
                h_norm = hi / samples_h
                angle = 2 * math.pi * h_norm
                r = s * radius
                x, y = cx + r * math.cos(angle), cy + r * math.sin(angle)
                rects.append((x - 2, y - 2, x + 2, y + 2, _hsv_color(h_norm, s, v)))
        self.slice_canvas.rects = rects
        self.slice_canvas.outline = ("ellipse", QRectF(cx - radius, cy - radius, 2 * radius, 2 * radius))
