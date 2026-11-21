"""Scene view: rendering, zoom and pan, and mouse tools."""

from __future__ import annotations

import math

from PySide6.QtCore import QPointF, QRectF, Qt, Signal
from PySide6.QtGui import QColor, QFont, QPainter
from PySide6.QtWidgets import QGraphicsScene, QGraphicsView

from ...algorithms.geometry import apply_matrix, matrix_rotate, matrix_scale
from ...constants import COL_PREV, COL_ROI, COL_SEL, H_FILL, H_OUTLINE, HANDLE_PX, HIT_TOLERANCE_PX
from ...core.document import Document
from ...i18n import tr
from ...scene.objects import (
    BezierShape,
    CircleShape,
    LineShape,
    PolygonShape,
    RasterImage,
    RectShape,
)
from .. import theme
from .items import ObjectItem, cosmetic_pen

IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".ppm", ".pgm", ".pbm", ".gif", ".webp")

# (id, label, shortcut)
TOOLS = [
    ("select", "Select / move", "V"),
    ("line", "Line", "L"),
    ("rect", "Rectangle", "R"),
    ("circle", "Circle", "C"),
    ("polygon", "Polygon", "P"),
    ("bezier", "Bézier curve", "B"),
    ("roi", "Image region", "O"),
    ("rotate", "Rotate", "T"),
    ("scale", "Scale", "S"),
]

TOOL_HINTS = {
    "select": "Click an object to select it. Drag to move it; drag the handles to reshape it. Mouse wheel = zoom, middle button = pan.",
    "line": "Drag to draw a line (Bresenham's algorithm).",
    "rect": "Drag to draw a rectangle.",
    "circle": "Drag from the centre to draw a circle (Bresenham).",
    "polygon": "Click to add vertices. Double-click or Enter closes the polygon, Backspace removes a point, Esc cancels.",
    "bezier": "Click control points. Double-click or Enter finishes the curve, Backspace removes a point, Esc cancels.",
    "roi": "Drag on an image to select a region – operations will only affect it. Click without dragging to clear it.",
    "rotate": "Shift+click sets the pivot. Then drag a polygon or curve to rotate it.",
    "scale": "Shift+click sets the pivot. Then drag a polygon or curve to scale it.",
}

TRANSFORMABLE = ("polygon", "bezier")
MIN_SCENE_RECT = QRectF(0, 0, 1600, 1000)


class SceneView(QGraphicsView):
    cursor_info = Signal(str)
    zoom_changed = Signal(float)
    status_message = Signal(str)
    files_dropped = Signal(list)

    def __init__(self, doc: Document, parent=None):
        super().__init__(parent)
        self.doc = doc
        self.setScene(QGraphicsScene(self))
        self.setBackgroundBrush(QColor(theme.canvas()["bg"]))
        self.setAcceptDrops(True)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.ViewportAnchor.AnchorViewCenter)
        self.setViewportUpdateMode(QGraphicsView.ViewportUpdateMode.FullViewportUpdate)
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

        self.object_items: dict[str, ObjectItem] = {}
        self.tool = "select"
        self.show_pixel_values = True
        self.draw_color: str | None = None      # colour of new shapes; None means the default
        self.draw_width: int = 1                   # stroke width of new shapes
        self.pivot: tuple | None = None

        self._drag: dict | None = None
        self._start: tuple | None = None        # drag start
        self._hover: tuple | None = None        # current drag position
        self._path: list = []                      # polygon / curve points being drawn
        self._pan_last = None

        doc.changed.connect(self.rebuild)
        doc.object_updated.connect(self._on_object_updated)
        doc.selection_changed.connect(self.viewport().update)
        self.scene().setSceneRect(MIN_SCENE_RECT)
        self.rebuild()

    # ================================================================== scene ==
    def rebuild(self):
        """Recreate the graphics items from the document."""
        self.scene().clear()
        self.object_items = {}
        for z, obj in enumerate(self.doc.objects):
            item = ObjectItem(obj, self)
            item.setZValue(z)
            self.scene().addItem(item)
            self.object_items[obj.oid] = item
        self._grow_scene_rect()
        self.viewport().update()

    def _on_object_updated(self, oid):
        item = self.object_items.get(oid)
        if item:
            item.refresh()
        self._grow_scene_rect()
        self.viewport().update()

    def _grow_scene_rect(self):
        bounds = self.scene().itemsBoundingRect().adjusted(-300, -300, 300, 300)
        self.scene().setSceneRect(self.scene().sceneRect().united(bounds).united(MIN_SCENE_RECT))

    def apply_theme(self):
        self.setBackgroundBrush(QColor(theme.canvas()["bg"]))
        self.viewport().update()

    # ============================================================ drag and drop ==
    @staticmethod
    def _dropped_images(event) -> list:
        if not event.mimeData().hasUrls():
            return []
        paths = [u.toLocalFile() for u in event.mimeData().urls() if u.isLocalFile()]
        return [p for p in paths if p.lower().endswith(IMAGE_EXTENSIONS)]

    def dragEnterEvent(self, event):
        if self._dropped_images(event):
            event.acceptProposedAction()
        else:
            super().dragEnterEvent(event)

    def dragMoveEvent(self, event):
        if self._dropped_images(event):
            event.acceptProposedAction()
        else:
            super().dragMoveEvent(event)

    def dropEvent(self, event):
        paths = self._dropped_images(event)
        if paths:
            event.acceptProposedAction()
            self.files_dropped.emit(paths)
        else:
            super().dropEvent(event)

    # ================================================================ zoom ==
    @property
    def zoom(self) -> float:
        return self.transform().m11()

    def _apply_zoom(self, factor: float):
        new = self.zoom * factor
        if not 0.02 <= new <= 96:
            return
        self.scale(factor, factor)
        self.zoom_changed.emit(self.zoom)

    def wheelEvent(self, event):
        steps = event.angleDelta().y() / 120.0
        if steps:
            self._apply_zoom(1.25 ** steps)

    def zoom_in(self):
        self._zoom_centered(1.25)

    def zoom_out(self):
        self._zoom_centered(0.8)

    def _zoom_centered(self, factor):
        anchor = self.transformationAnchor()
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorViewCenter)
        self._apply_zoom(factor)
        self.setTransformationAnchor(anchor)

    def zoom_reset(self):
        self.resetTransform()
        self.zoom_changed.emit(self.zoom)

    def zoom_fit(self):
        target = self.scene().itemsBoundingRect()
        if target.isEmpty():
            return self.zoom_reset()
        self.fitInView(target.adjusted(-10, -10, 10, 10), Qt.AspectRatioMode.KeepAspectRatio)
        self.zoom_changed.emit(self.zoom)

    def focus_object(self, oid):
        item = self.object_items.get(oid)
        if item:
            self.ensureVisible(item.boundingRect(), 40, 40)

    # ================================================================== tools ==
    def set_tool(self, tool: str):
        self._cancel_in_progress()
        self.tool = tool
        cursor = Qt.CursorShape.ArrowCursor if tool == "select" else Qt.CursorShape.CrossCursor
        self.viewport().setCursor(cursor)
        self.status_message.emit(tr(TOOL_HINTS.get(tool, "")))
        self.viewport().update()

    def _cancel_in_progress(self):
        drag, self._drag = self._drag, None
        if drag and drag.get("interactive"):
            self.doc.cancel_interactive()  # restore the state from before the drag
        self._start = None
        self._path = []
        self.viewport().update()

    def _tol(self) -> float:
        return HIT_TOLERANCE_PX / max(self.zoom, 1e-6)

    def _scene_pos(self, event) -> QPointF:
        return self.mapToScene(event.position().toPoint())

    def _image_at(self, p: QPointF) -> RasterImage | None:
        for obj in reversed(self.doc.objects):
            if isinstance(obj, RasterImage) and obj.hit_test(p.x(), p.y(), 0):
                return obj
        return None

    # ------------------------------------------------------------------ mouse --
    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.MiddleButton:
            self._pan_last = event.position()
            self.viewport().setCursor(Qt.CursorShape.ClosedHandCursor)
            return
        if event.button() != Qt.MouseButton.LeftButton:
            return super().mousePressEvent(event)
        p = self._scene_pos(event)
        handler = getattr(self, f"_press_{self.tool}", None)
        if handler:
            handler(p, event)

    def mouseMoveEvent(self, event):
        if self._pan_last is not None:
            delta = event.position() - self._pan_last
            self._pan_last = event.position()
            self.horizontalScrollBar().setValue(int(self.horizontalScrollBar().value() - delta.x()))
            self.verticalScrollBar().setValue(int(self.verticalScrollBar().value() - delta.y()))
            return
        p = self._scene_pos(event)
        self._emit_cursor_info(p)
        self._hover = (p.x(), p.y())
        handler = getattr(self, f"_move_{self.tool}", None)
        if handler:
            handler(p, event)
        if self.tool in ("polygon", "bezier") and self._path:
            self.viewport().update()

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MouseButton.MiddleButton and self._pan_last is not None:
            self._pan_last = None
            self.set_cursor_for_tool()
            return
        if event.button() != Qt.MouseButton.LeftButton:
            return super().mouseReleaseEvent(event)
        p = self._scene_pos(event)
        handler = getattr(self, f"_release_{self.tool}", None)
        if handler:
            handler(p, event)

    def mouseDoubleClickEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton and self.tool in ("polygon", "bezier"):
            self.finish_path()
            return
        super().mouseDoubleClickEvent(event)

    def set_cursor_for_tool(self):
        cursor = Qt.CursorShape.ArrowCursor if self.tool == "select" else Qt.CursorShape.CrossCursor
        self.viewport().setCursor(cursor)

    def keyPressEvent(self, event):
        key = event.key()
        if key in (Qt.Key.Key_Return, Qt.Key.Key_Enter) and self._path:
            self.finish_path()
        elif key == Qt.Key.Key_Backspace and self._path:
            self._path.pop()
            self.viewport().update()
        elif key == Qt.Key.Key_Escape:
            if self._path or self._start or self._drag:
                self._cancel_in_progress()
            else:
                self.doc.select(None)
        else:
            super().keyPressEvent(event)

    def _emit_cursor_info(self, p: QPointF):
        text = f"x = {p.x():.0f}   y = {p.y():.0f}"
        img = self._image_at(p)
        if img is not None:
            info = img.pixel_at(p.x(), p.y())
            if info:
                px, py, val = info
                ch = "RGBA" if len(val) == 4 else "RGB"
                text += f"   |   {img.display_name}: " + tr("pixel ({x}, {y})", x=px, y=py) + f"  {ch}{val}"
        self.cursor_info.emit(text)

    # ------------------------------------------------------------ select tool --
    def _press_select(self, p, event):
        obj = self.doc.selected
        if obj is not None:
            hs = (HANDLE_PX + 2) / self.zoom
            for hx, hy, kind in obj.handles():
                if abs(p.x() - hx) <= hs and abs(p.y() - hy) <= hs:
                    self.doc.begin_interactive()
                    self._drag = {"mode": "handle", "obj": obj, "kind": kind, "moved": False, "interactive": True}
                    return
        hit = self.doc.object_at(p.x(), p.y(), self._tol())
        if hit is None:
            self.doc.select(None)
            return
        self.doc.select(hit.oid)
        self.doc.begin_interactive()
        self._drag = {"mode": "move", "obj": hit, "last": (round(p.x()), round(p.y())), "moved": False,
                      "interactive": True}

    def _move_select(self, p, event):
        d = self._drag
        if not d or not (event.buttons() & Qt.MouseButton.LeftButton):
            return
        obj = d["obj"]
        if d["mode"] == "handle":
            obj.apply_handle(d["kind"], round(p.x()), round(p.y()))
            d["moved"] = True
        else:
            cur = (round(p.x()), round(p.y()))
            dx, dy = cur[0] - d["last"][0], cur[1] - d["last"][1]
            if dx or dy:
                obj.move(dx, dy)
                d["last"] = cur
                d["moved"] = True
        self.doc.notify_updated(obj)

    def _release_select(self, p, event):
        d, self._drag = self._drag, None
        if not d:
            return
        verb = tr("Reshape") if d["mode"] == "handle" else tr("Move")
        self.doc.end_interactive(f"{verb}: {d['obj'].display_name}", d["moved"])

    # ---------------------------------------------- line, rectangle, circle --
    def _press_shape(self, p, event):
        self._start = (round(p.x()), round(p.y()))
        self._hover = (p.x(), p.y())

    _press_line = _press_rect = _press_circle = _press_shape

    def _move_shape(self, p, event):
        if self._start:
            self.viewport().update()

    _move_line = _move_rect = _move_circle = _move_shape

    def _release_shape(self, p, event):
        if not self._start:
            return
        x0, y0 = self._start
        x1, y1 = round(p.x()), round(p.y())
        self._start = None
        obj = None
        if self.tool == "line" and (x0, y0) != (x1, y1):
            obj = LineShape(x0, y0, x1, y1)
        elif self.tool == "rect":
            if (x0, y0) != (x1, y1):
                obj = RectShape(min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1))
        elif self.tool == "circle":
            r = int(round(math.hypot(x1 - x0, y1 - y0)))
            if r > 0:
                obj = CircleShape(x0, y0, r)
        if obj is not None:
            self._apply_draw_color(obj)
            self.doc.add_objects([obj], tr("Drawn with mouse: {kind}", kind=tr(obj.kind_label)))
        self.viewport().update()

    def _apply_draw_color(self, obj):
        if self.draw_color:
            obj.color = self.draw_color
        obj.width = self.draw_width

    _release_line = _release_rect = _release_circle = _release_shape

    # ------------------------------------------------------ polygon, Bézier --
    def _press_polygon(self, p, event):
        self._path.append((round(p.x()), round(p.y())))
        self.viewport().update()

    _press_bezier = _press_polygon

    def finish_path(self):
        pts = list(self._path)
        if self.tool == "polygon":
            if len(pts) < 3:
                self.status_message.emit(tr("Polygon: at least 3 points are needed."))
                return
            obj = PolygonShape(points=[(float(x), float(y)) for x, y in pts], closed=True)
        else:
            if len(pts) < 2:
                self.status_message.emit(tr("Bézier curve: at least 2 control points are needed."))
                return
            obj = BezierShape(points=[(float(x), float(y)) for x, y in pts])
        self._path = []
        self._apply_draw_color(obj)
        self.doc.add_objects([obj], tr("Drawn with mouse: {kind}", kind=tr(obj.kind_label)))

    # ------------------------------------------------------------ region tool --
    def _press_roi(self, p, event):
        img = self._image_at(p) or self.doc.target_image()
        if img is None:
            self.status_message.emit(tr("Region: click on an image."))
            return
        self.doc.select(img.oid)
        self._drag = {"mode": "roi", "obj": img, "start": (p.x(), p.y()), "screen": event.position()}

    def _move_roi(self, p, event):
        if self._drag:
            self.viewport().update()

    def _roi_from_drag(self, img: RasterImage, a, b):
        def clamp_px(sx, sy):
            px, py = img.scene_to_pixel(sx, sy)
            return min(max(px, 0), img.src_w - 1), min(max(py, 0), img.src_h - 1)

        ax, ay = clamp_px(*a)
        bx, by = clamp_px(*b)
        return (min(ax, bx), min(ay, by), max(ax, bx) + 1, max(ay, by) + 1)

    def _release_roi(self, p, event):
        d, self._drag = self._drag, None
        if not d:
            return
        img = d["obj"]
        moved = (event.position() - d["screen"]).manhattanLength() > 3
        roi = self._roi_from_drag(img, d["start"], (p.x(), p.y())) if moved else None
        self.doc.set_roi(img, roi)
        if roi:
            self.status_message.emit(tr("Selected a {w}×{h} px region – operations will only affect it.",
                                         w=roi[2] - roi[0], h=roi[3] - roi[1]))
        else:
            self.status_message.emit(tr("Region cleared."))
        self.viewport().update()

    # ---------------------------------------------------------- rotate, scale --
    def _press_rotate(self, p, event):
        if self.pivot is None or event.modifiers() & Qt.KeyboardModifier.ShiftModifier:
            self.pivot = (p.x(), p.y())
            self.status_message.emit(tr("Pivot: ({x}, {y})", x=f"{p.x():.0f}", y=f"{p.y():.0f}"))
            self.viewport().update()
            return
        hit = self.doc.object_at(p.x(), p.y(), self._tol(), kinds=TRANSFORMABLE)
        if hit is None:
            self.status_message.emit(tr("Drag a polygon or Bézier curve (Shift+click moves the pivot)."))
            return
        self.doc.select(hit.oid)
        self.doc.begin_interactive()
        px, py = self.pivot
        self._drag = {
            "mode": self.tool, "obj": hit, "orig": list(hit.points), "moved": False, "interactive": True,
            "a0": math.atan2(p.y() - py, p.x() - px),
            "d0": math.hypot(p.x() - px, p.y() - py) or 1.0,
        }

    _press_scale = _press_rotate

    def _move_rotate(self, p, event):
        d = self._drag
        if not d or not (event.buttons() & Qt.MouseButton.LeftButton):
            return
        px, py = self.pivot
        if d["mode"] == "rotate":
            delta = math.atan2(p.y() - py, p.x() - px) - d["a0"]
            M = matrix_rotate(px, py, delta)
            d["value"] = f"{math.degrees(delta):.1f}°"
        else:
            k = math.hypot(p.x() - px, p.y() - py) / d["d0"]
            M = matrix_scale(px, py, k)
            d["value"] = f"k = {k:.3f}"
        d["obj"].points = apply_matrix(d["orig"], M)
        d["moved"] = True
        self.status_message.emit(f"{tr("Rotate") if d['mode'] == 'rotate' else tr("Scale")}: {d['value']}")
        self.doc.notify_updated(d["obj"])

    _move_scale = _move_rotate

    def _release_rotate(self, p, event):
        d, self._drag = self._drag, None
        if not d:
            return
        verb = tr("Rotate with mouse") if d["mode"] == "rotate" else tr("Scale with mouse")
        self.doc.end_interactive(f"{verb} ({d.get('value', '')}): {d['obj'].display_name}", d["moved"])

    _release_scale = _release_rotate

    # =============================================================== overlays ==
    def drawForeground(self, painter: QPainter, rect):
        if not self.doc.objects:
            self._paint_empty_hint(painter)
        s = max(self.zoom, 1e-6)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, False)

        # selection and handles
        obj = self.doc.selected
        if obj is not None:
            x1, y1, x2, y2 = obj.bbox()
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.setPen(cosmetic_pen(COL_SEL, 1, Qt.PenStyle.DashLine))
            painter.drawRect(QRectF(x1, y1, x2 - x1, y2 - y1))
            hs = HANDLE_PX / s
            painter.setPen(cosmetic_pen(H_OUTLINE, 1))
            painter.setBrush(QColor(H_FILL))
            for hx, hy, _kind in obj.handles():
                painter.drawRect(QRectF(hx - hs, hy - hs, 2 * hs, 2 * hs))

        painter.setBrush(Qt.BrushStyle.NoBrush)
        prev = cosmetic_pen(self.draw_color or COL_PREV, 1, Qt.PenStyle.DashLine)

        # preview of the shape being drawn
        if self._start and self._hover:
            x0, y0 = self._start
            hx, hy = round(self._hover[0]), round(self._hover[1])
            painter.setPen(prev)
            if self.tool == "line":
                painter.drawLine(QPointF(x0 + 0.5, y0 + 0.5), QPointF(hx + 0.5, hy + 0.5))
            elif self.tool == "rect":
                painter.drawRect(QRectF(QPointF(min(x0, hx), min(y0, hy)), QPointF(max(x0, hx) + 1, max(y0, hy) + 1)))
            elif self.tool == "circle":
                r = math.hypot(hx - x0, hy - y0)
                painter.drawEllipse(QPointF(x0 + 0.5, y0 + 0.5), r, r)

        # polygon / curve being drawn
        if self._path:
            pts = [QPointF(x, y) for x, y in self._path]
            if self._hover:
                pts.append(QPointF(*self._hover))
            painter.setPen(prev)
            painter.drawPolyline(pts)
            painter.setBrush(QColor(theme.canvas()["fg"]))
            r = 3 / s
            for x, y in self._path:
                painter.drawEllipse(QPointF(x, y), r, r)
            painter.setBrush(Qt.BrushStyle.NoBrush)

        # region being selected
        if self._drag and self._drag.get("mode") == "roi" and self._hover:
            img = self._drag["obj"]
            x0, y0, x1, y1 = self._roi_from_drag(img, self._drag["start"], self._hover)
            sx, sy = img.w / img.src_w, img.h / img.src_h
            painter.setPen(cosmetic_pen(COL_ROI, 1, Qt.PenStyle.DashLine))
            painter.drawRect(QRectF(img.x + x0 * sx, img.y + y0 * sy, (x1 - x0) * sx, (y1 - y0) * sy))

        # rotation / scaling pivot
        if self.pivot and self.tool in ("rotate", "scale"):
            x, y = self.pivot
            d = 6 / s
            painter.setPen(cosmetic_pen(theme.canvas()["pivot"], 2))
            painter.drawLine(QPointF(x - d, y), QPointF(x + d, y))
            painter.drawLine(QPointF(x, y - d), QPointF(x, y + d))
            painter.drawEllipse(QPointF(x, y), 3 / s, 3 / s)

    def _paint_empty_hint(self, painter: QPainter):
        """Draw a getting-started hint on an empty scene."""
        painter.save()
        painter.resetTransform()
        font = QFont(self.font())
        font.setPointSizeF(self.font().pointSizeF() * 1.6)
        painter.setFont(font)
        painter.setPen(QColor(theme.canvas()["hint"]))
        painter.drawText(self.viewport().rect(), Qt.AlignmentFlag.AlignCenter,
                         tr("Open images (Ctrl+O) or drop files here\nSample images: Load demo (Ctrl+Shift+D)"))
        painter.restore()
