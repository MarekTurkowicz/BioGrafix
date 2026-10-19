"""Document, history and user interface tests (Qt offscreen platform)."""

import os
import tempfile
import time
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np  # noqa: E402
from PySide6.QtCore import QEvent, QPointF, Qt  # noqa: E402
from PySide6.QtGui import QKeyEvent, QMouseEvent  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from biografix.core import registry  # noqa: E402
from biografix.scene.objects import BezierShape, LineShape, PolygonShape, RasterImage  # noqa: E402
from biografix.scene.serialization import load_scene, save_scene  # noqa: E402
from biografix.ui.main_window import MainWindow  # noqa: E402

app = QApplication.instance() or QApplication([])
L, NB = Qt.MouseButton.LeftButton, Qt.MouseButton.NoButton


def sample_image(h=60, w=80):
    return (np.random.default_rng(1).random((h, w, 3)) * 255).astype(np.uint8)


class GuiTestCase(unittest.TestCase):
    def setUp(self):
        self.w = MainWindow()
        self.w.resize(1200, 800)
        self.w.show()
        self.doc, self.v = self.w.doc, self.w.view
        self.v.zoom_reset()
        app.processEvents()

    def tearDown(self):
        self.doc.undo_stack.setClean()
        self.w.close()
        self.w.deleteLater()
        app.processEvents()
        app.sendPostedEvents(None, QEvent.Type.DeferredDelete)  # actually destroy the window

    # --- helpers ---
    def add_image(self, **kw):
        img = RasterImage(x=10, y=10, pixels=sample_image(), name="test.png", **kw)
        self.doc.add_objects([img], "image")
        return img

    def run_op(self, op_id, **values):
        op = registry.get(op_id)
        vals = {p.name: p.validate(p.default) for p in op.params}
        vals.update(values)
        self.w.runner.run(op, vals)
        deadline = time.time() + 30
        while self.w.runner.busy and time.time() < deadline:
            app.processEvents()
            time.sleep(0.005)
        app.processEvents()

    def mouse(self, kind, x, y, button, buttons, mods=Qt.KeyboardModifier.NoModifier):
        p = QPointF(self.v.mapFromScene(QPointF(x, y)))
        e = QMouseEvent(kind, p, self.v.viewport().mapToGlobal(p), button, buttons, mods)
        QApplication.sendEvent(self.v.viewport(), e)
        app.processEvents()

    def drag(self, x0, y0, x1, y1, mods=Qt.KeyboardModifier.NoModifier):
        self.mouse(QEvent.Type.MouseButtonPress, x0, y0, L, L, mods)
        for i in range(1, 6):
            self.mouse(QEvent.Type.MouseMove, x0 + (x1 - x0) * i / 5, y0 + (y1 - y0) * i / 5, NB, L, mods)
        self.mouse(QEvent.Type.MouseButtonRelease, x1, y1, L, NB, mods)

    def click(self, x, y, mods=Qt.KeyboardModifier.NoModifier):
        self.mouse(QEvent.Type.MouseButtonPress, x, y, L, L, mods)
        self.mouse(QEvent.Type.MouseButtonRelease, x, y, L, NB, mods)

    def key(self, k):
        QApplication.sendEvent(self.v, QKeyEvent(QEvent.Type.KeyPress, k, Qt.KeyboardModifier.NoModifier))
        app.processEvents()


class TestOperationsAndHistory(GuiTestCase):
    def test_background_operation_and_undo(self):
        img = self.add_image()
        self.run_op("bin.otsu")
        cur = self.doc.find(img.oid)
        self.assertTrue(set(np.unique(cur.pixels)) <= {0, 255})
        self.doc.undo_stack.undo()
        self.assertTrue(np.array_equal(self.doc.find(img.oid).pixels, img.original))
        self.doc.undo_stack.redo()
        self.assertTrue(set(np.unique(self.doc.find(img.oid).pixels)) <= {0, 255})

    def test_roi_limits_operation(self):
        img = self.add_image()
        self.doc.set_roi(img, (10, 10, 40, 30))
        self.run_op("point.fill", r=1, g=2, b=3)
        px = self.doc.find(img.oid).pixels
        self.assertTrue((px[10:30, 10:40] == (1, 2, 3)).all())
        self.assertTrue(np.array_equal(px[:10], img.original[:10]))

    def test_history_jump(self):
        img = self.add_image()
        self.run_op("point.brighten")
        self.run_op("point.darken")
        self.assertEqual(self.doc.undo_stack.count(), 3)
        self.doc.undo_stack.setIndex(1)  # state right after the image was added
        self.assertTrue(np.array_equal(self.doc.find(img.oid).pixels, img.original))

    def test_reset_image_is_undoable(self):
        img = self.add_image()
        self.run_op("point.darken")
        self.w.reset_image()
        self.assertIs(self.doc.find(img.oid).pixels, self.doc.find(img.oid).original)
        self.doc.undo_stack.undo()
        self.assertIsNot(self.doc.find(img.oid).pixels, self.doc.find(img.oid).original)

    def test_shape_operation(self):
        poly = PolygonShape(points=[(0, 0), (10, 0), (10, 10)])
        self.doc.add_objects([poly], "polygon")
        self.run_op("geom.translate", dx=5.0, dy=-5.0)
        self.assertEqual(self.doc.find(poly.oid).points[0], (5.0, -5.0))

    def test_scene_roundtrip(self):
        img = self.add_image()
        self.run_op("point.darken")
        self.doc.add_objects([LineShape(1, 2, 3, 4), BezierShape(points=[(0, 0), (5, 5), (9, 0)])], "shapes")
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "scena.json")
            save_scene(path, self.doc.objects)
            objs = load_scene(path)
        self.assertEqual([o.kind for o in objs], ["image", "line", "bezier"])
        self.assertTrue(np.array_equal(objs[0].pixels, self.doc.find(img.oid).pixels))
        self.assertTrue(np.array_equal(objs[0].original, img.original))

    def test_undo_redo_are_logged(self):
        self.add_image()
        self.run_op("point.darken")
        self.doc.undo_stack.undo()
        self.assertIn("Undone: ", self.w.log.toPlainText())
        self.doc.undo_stack.redo()
        self.assertIn("Redone: ", self.w.log.toPlainText())
        self.doc.undo_stack.setIndex(0)
        self.assertIn("Undone 2 steps", self.w.log.toPlainText())

    def test_polish_interface(self):
        from biografix import i18n

        self.w.close()
        i18n.set_language("pl")
        try:
            w = MainWindow()
            menus = [a.text() for a in w.menuBar().actions()]
            self.assertIn("&Plik", menus)
            self.assertEqual(w.ribbon.tabText(0), "Operacje punktowe")
            self.assertEqual(w.cancel_button.text(), "Przerwij")
            w.close()
            w.deleteLater()
        finally:
            i18n.set_language("en")

    def test_dialog_is_centered_on_window(self):
        from PySide6.QtWidgets import QDialog

        dlg = QDialog(self.w)
        dlg.resize(300, 200)
        dlg.show()
        app.processEvents()
        diff = dlg.frameGeometry().center() - self.w.frameGeometry().center()
        self.assertLessEqual(diff.manhattanLength(), 4)
        dlg.close()

    def _start_slow_op(self):
        """Start an operation that runs in the background for about 0.5 s."""
        from biografix.core.registry import Operation

        def slow(image):
            time.sleep(0.5)
            return 255 - image

        img = self.add_image()
        self.w.runner.run(Operation("test.slow", "Test", "Wolna", slow), {})
        self.assertTrue(self.w.runner.busy)
        return img

    def test_busy_image_cannot_be_removed_without_asking(self):
        img = self._start_slow_op()
        self.w._ask_busy = lambda *a: False          # user chooses "Keep running"
        self.doc.remove(img.oid)
        self.assertIsNotNone(self.doc.find(img.oid))
        while self.w.runner.busy:
            app.processEvents()
        app.processEvents()
        self.assertFalse(np.array_equal(self.doc.find(img.oid).pixels, img.original))

    def test_cancel_drops_result(self):
        img = self._start_slow_op()
        self.w.cancel_operation()
        self.assertFalse(self.w.runner.busy)
        self.w.runner.shutdown()
        app.processEvents()
        self.assertTrue(np.array_equal(self.doc.find(img.oid).pixels, img.original))
        self.assertEqual(self.w.time_label.text(), "Cancelled")


class TestMouseTools(GuiTestCase):
    def test_draw_move_and_resize_line(self):
        self.v.set_tool("line")
        self.drag(100, 100, 300, 160)
        line = self.doc.objects[-1]
        self.assertEqual(line.params_text(), "100,100,300,160")
        self.v.set_tool("select")
        self.drag(200, 130, 220, 150)
        self.assertEqual(self.doc.find(line.oid).params_text(), "120,120,320,180")
        self.drag(320, 180, 400, 400)  # line end handle
        self.assertEqual(self.doc.find(line.oid).params_text(), "120,120,400,400")

    def test_polygon_and_rotation(self):
        self.v.set_tool("polygon")
        for x, y in [(500, 100), (650, 120), (620, 260)]:
            self.click(x, y)
        self.mouse(QEvent.Type.MouseButtonDblClick, 620, 260, L, L)
        self.mouse(QEvent.Type.MouseButtonRelease, 620, 260, L, NB)
        poly = self.doc.objects[-1]
        self.assertEqual((poly.kind, len(poly.points)), ("polygon", 3))
        self.v.set_tool("rotate")
        self.click(590, 160, Qt.KeyboardModifier.ShiftModifier)
        self.assertEqual(self.v.pivot, (590.0, 160.0))
        before = list(poly.points)
        self.drag(620, 180, 600, 200)
        self.assertNotEqual(self.doc.find(poly.oid).points, before)

    def test_escape_cancels_drag(self):
        self.doc.add_objects([LineShape(100, 100, 200, 100)], "linia")
        line = self.doc.objects[-1]
        count = self.doc.undo_stack.count()
        self.v.set_tool("select")
        self.mouse(QEvent.Type.MouseButtonPress, 150, 100, L, L)
        self.mouse(QEvent.Type.MouseMove, 250, 300, NB, L)
        self.key(Qt.Key.Key_Escape)
        self.mouse(QEvent.Type.MouseButtonRelease, 250, 300, L, NB)
        self.assertEqual(self.doc.find(line.oid).params_text(), "100,100,200,100")
        self.assertEqual(self.doc.undo_stack.count(), count)

    def test_palette_color_for_new_and_selected_shapes(self):
        self.w.palette.choose("#ff9800")
        self.v.set_tool("line")
        self.drag(100, 100, 200, 150)
        line = self.doc.objects[-1]
        self.assertEqual(line.color, "#ff9800")
        self.doc.select(line.oid)
        self.w.palette.choose("#26c6da")                    # applies to the selected shape
        self.assertEqual(self.doc.find(line.oid).color, "#26c6da")
        self.doc.undo_stack.undo()
        self.assertEqual(self.doc.find(line.oid).color, "#ff9800")
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "s.json")
            save_scene(path, self.doc.objects)
            self.assertEqual(load_scene(path)[-1].color, "#ff9800")

    def test_width_slider_changes_selected_shape_once(self):
        self.v.set_tool("line")
        self.drag(100, 100, 200, 150)
        line = self.doc.objects[-1]
        self.doc.select(line.oid)
        count = self.doc.undo_stack.count()
        slider = self.w.palette.width_slider
        slider.setSliderDown(True)                          # dragging: preview only
        for value in (3, 5, 7):
            slider.setValue(value)
        self.assertEqual(self.doc.undo_stack.count(), count)
        slider.setSliderDown(False)
        slider.sliderReleased.emit()                        # release: one undo step
        self.assertEqual(self.doc.find(line.oid).width, 7)
        self.assertEqual(self.doc.undo_stack.count(), count + 1)
        self.assertEqual(self.v.draw_width, 7)

    def test_roi_tool(self):
        img = self.add_image()
        self.v.set_tool("roi")
        self.drag(img.x + 10, img.y + 20, img.x + 40, img.y + 50)
        self.assertEqual(self.doc.find(img.oid).roi, (10, 20, 41, 51))


if __name__ == "__main__":
    unittest.main()
