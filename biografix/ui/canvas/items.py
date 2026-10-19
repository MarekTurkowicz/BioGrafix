"""Graphics items that render scene objects."""

from __future__ import annotations

import math

from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QBrush, QColor, QFont, QImage, QPainter, QPen, QPolygonF
from PySide6.QtWidgets import QGraphicsItem, QStyleOptionGraphicsItem

from ...constants import COL_ROI
from .. import theme
from ...scene.objects import BezierShape, PolygonShape, RasterImage, SceneObject
from ..qt_image import array_to_qimage

PIXEL_VALUES_MIN_PX = 44   # on-screen pixel size above which pixel values are drawn
PIXEL_VALUES_MAX_COUNT = 2500


def cosmetic_pen(color, width=1.0, style=Qt.PenStyle.SolidLine) -> QPen:
    pen = QPen(QColor(color), width, style)
    pen.setCosmetic(True)
    return pen


class ObjectItem(QGraphicsItem):
    """Graphics item that renders one scene object."""

    def __init__(self, obj: SceneObject, view):
        super().__init__()
        self.obj = obj
        self.view = view
        self._qimage = None
        self._qimage_key = None
        self._points_key = None
        self._points_poly = None
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemUsesExtendedStyleOption, True)

    # --------------------------------------------------------------- geometry --
    def boundingRect(self) -> QRectF:
        x1, y1, x2, y2 = self.obj.bbox()
        m = 20.0 + getattr(self.obj, "width", 1)  # room for thick strokes, handles and point labels
        return QRectF(x1 - m, y1 - m, (x2 - x1) + 2 * m, (y2 - y1) + 2 * m)

    def refresh(self):
        self.prepareGeometryChange()
        self.update()

    # --------------------------------------------------------------- painting --
    def paint(self, painter: QPainter, option: QStyleOptionGraphicsItem, widget=None):
        obj = self.obj
        if isinstance(obj, RasterImage):
            self._paint_image(painter, option, obj)
        elif isinstance(obj, PolygonShape):
            self._paint_polygon(painter, obj)
        elif isinstance(obj, BezierShape):
            self._paint_bezier(painter, obj)
        elif hasattr(obj, "raster_points"):
            self._paint_raster_shape(painter, obj)

    def _lod(self, painter) -> float:
        return QStyleOptionGraphicsItem.levelOfDetailFromTransform(painter.worldTransform()) or 1.0

    # --- rasterised shapes (Bresenham) ---
    def _paint_raster_shape(self, painter, obj):
        width = max(1, int(getattr(obj, "width", 1)))
        key = (obj.params_text(), obj.color, width)
        if key != self._points_key:
            pts = [(int(round(x)), int(round(y))) for x, y in obj.raster_points()]
            self._points_poly = QPolygonF([QPointF(x + 0.5, y + 0.5) for x, y in pts])
            # cache the rasterised points as a small image so pixels stay sharp at any zoom;
            # with width > 1 every point becomes a width × width square
            lo = width // 2
            xs = [p[0] for p in pts] or [0]
            ys = [p[1] for p in pts] or [0]
            x0, y0 = min(xs) - lo, min(ys) - lo
            img = QImage(max(xs) - x0 + width, max(ys) - y0 + width, QImage.Format.Format_ARGB32)
            img.fill(Qt.GlobalColor.transparent)
            color = QColor(obj.color)
            if width == 1:
                rgb = color.rgba()
                for x, y in pts:
                    img.setPixel(x - x0, y - y0, rgb)
            else:
                p = QPainter(img)
                for x, y in pts:
                    p.fillRect(x - lo - x0, y - lo - y0, width, width, color)
                p.end()
            self._qimage = (x0, y0, img)
            self._points_key = key  # set last so that a failure does not leave an empty cache
        if self._lod(painter) * width >= 1.0:
            x0, y0, img = self._qimage
            painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, False)
            painter.drawImage(QPointF(x0, y0), img)
        else:
            # zoomed out: cosmetic points so the shape does not vanish
            painter.setPen(cosmetic_pen(obj.color, 1))
            painter.drawPoints(self._points_poly)

    # --- polygon ---
    def _paint_polygon(self, painter, obj: PolygonShape):
        if not obj.points:
            return
        selected = self.view.doc.selected_oid == obj.oid
        c = theme.canvas()
        color = obj.color or c["fg"]
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        w = max(1, getattr(obj, "width", 1)) + 1
        painter.setPen(cosmetic_pen(color, w + 1 if selected else w))
        painter.setBrush(Qt.BrushStyle.NoBrush)
        poly = QPolygonF([QPointF(x, y) for x, y in obj.points])
        if obj.closed and len(obj.points) >= 3:
            painter.drawPolygon(poly)
        else:
            painter.drawPolyline(poly)
        r = 3 / self._lod(painter)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor(color))
        for x, y in obj.points:
            painter.drawEllipse(QPointF(x, y), r, r)

    # --- Bézier curve ---
    def _paint_bezier(self, painter, obj: BezierShape):
        if len(obj.points) < 2:
            return
        lod = self._lod(painter)
        c = theme.canvas()
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        # control polygon
        painter.setPen(cosmetic_pen(c["bezier_ctrl"], 1, Qt.PenStyle.DashLine))
        painter.drawPolyline(QPolygonF([QPointF(x, y) for x, y in obj.points]))
        # curve (de Casteljau)
        painter.setPen(cosmetic_pen(obj.color or c["bezier"], max(1, getattr(obj, "width", 1)) + 1))
        painter.drawPolyline(QPolygonF([QPointF(x, y) for x, y in obj.curve(200)]))
        # control points
        r = 5 / lod
        painter.setPen(cosmetic_pen(c["fg"], 1))
        painter.setBrush(QColor(c["bezier_point"]))
        font = QFont()
        font.setPointSizeF(max(8, theme.current_font_pt() - 2))
        for i, (x, y) in enumerate(obj.points):
            painter.drawEllipse(QPointF(x, y), r, r)
        # labels in screen coordinates (constant font size)
        painter.save()
        t = painter.worldTransform()
        painter.resetTransform()
        painter.setFont(font)
        painter.setPen(QColor(c["fg"]))
        for i, (x, y) in enumerate(obj.points):
            p = t.map(QPointF(x, y))
            painter.drawText(p + QPointF(10, 4), f"P{i}")
        painter.restore()

    # --- raster image ---
    def _paint_image(self, painter, option, obj: RasterImage):
        key = id(obj.pixels)
        if key != self._qimage_key:
            self._qimage_key = key
            self._qimage = array_to_qimage(obj.pixels)
        target = QRectF(obj.x, obj.y, obj.w, obj.h)
        if obj.pixels.shape[2] == 4:
            painter.fillRect(target, QBrush(QColor("#cccccc"), Qt.BrushPattern.Dense4Pattern))
        # nearest-neighbour scaling shows pixels as squares
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, False)
        painter.drawImage(target, self._qimage)

        if self.view.show_pixel_values:
            self._paint_pixel_values(painter, option, obj)

        if obj.roi:
            x0, y0, x1, y1 = obj.roi
            sx, sy = obj.w / obj.src_w, obj.h / obj.src_h
            rect = QRectF(obj.x + x0 * sx, obj.y + y0 * sy, (x1 - x0) * sx, (y1 - y0) * sy)
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.setPen(cosmetic_pen("#ffffff", 1))
            painter.drawRect(rect)
            painter.setPen(cosmetic_pen(COL_ROI, 1, Qt.PenStyle.DashLine))
            painter.drawRect(rect)

    def _paint_pixel_values(self, painter, option, obj: RasterImage):
        """Draw the RGB value on every visible pixel at high zoom."""
        lod = self._lod(painter)
        px_w = obj.w / obj.src_w
        px_h = obj.h / obj.src_h
        if min(px_w, px_h) * lod < PIXEL_VALUES_MIN_PX:
            return
        visible = option.exposedRect.intersected(QRectF(obj.x, obj.y, obj.w, obj.h))
        if visible.isEmpty():
            return
        c0 = max(0, int(math.floor((visible.left() - obj.x) / px_w)))
        c1 = min(obj.src_w, int(math.ceil((visible.right() - obj.x) / px_w)))
        r0 = max(0, int(math.floor((visible.top() - obj.y) / px_h)))
        r1 = min(obj.src_h, int(math.ceil((visible.bottom() - obj.y) / px_h)))
        if (c1 - c0) * (r1 - r0) > PIXEL_VALUES_MAX_COUNT:
            return

        painter.save()
        t = painter.worldTransform()
        painter.resetTransform()
        font = QFont()
        font.setPixelSize(max(8, min(13, int(min(px_w, px_h) * lod / 5))))
        painter.setFont(font)
        for row in range(r0, r1):
            for col in range(c0, c1):
                px = obj.pixels[row, col]
                r, g, b = int(px[0]), int(px[1]), int(px[2])
                rect = t.mapRect(QRectF(obj.x + col * px_w, obj.y + row * px_h, px_w, px_h))
                light = 0.299 * r + 0.587 * g + 0.114 * b > 140
                painter.setPen(QColor("#000000" if light else "#ffffff"))
                text = str(r) if r == g == b else f"{r}\n{g}\n{b}"  # gray pixel: a single value
                painter.drawText(rect, Qt.AlignmentFlag.AlignCenter, text)
        painter.restore()
