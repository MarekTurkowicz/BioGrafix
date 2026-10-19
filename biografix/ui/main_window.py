"""Main application window."""

from __future__ import annotations

import os
import sys
import time
from typing import Dict

import numpy as np
from PySide6.QtCore import QProcess, QSettings, Qt, QTimer
from PySide6.QtGui import QAction, QActionGroup, QKeySequence
from PySide6.QtWidgets import (
    QApplication,
    QPushButton,
    QDockWidget,
    QInputDialog,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QProgressBar,
    QToolBar,
    QUndoView,
    QVBoxLayout,
    QWidget,
)

from ..constants import APP_NAME, APP_TITLE, ORG_NAME
from .. import i18n
from ..core import registry
from ..core.document import Document
from ..core.registry import TARGET_SHAPE, Operation
from ..core.runner import OperationError, OperationRunner
from ..io.images import OPEN_FILTER, SAVE_FILTER, load_image, save_image
from ..operations import load_all
from ..scene.objects import RasterImage
from ..i18n import tr
from ..scene.serialization import load_scene, save_scene
from .canvas.view import TOOLS, SceneView
from .dialogs import FigureWindow, MarkdownDialog, ParamsDialog
from .panels import ObjectsPanel, PropertiesPanel
from . import file_dialogs, theme
from . import centering
from .palette import ColorPalette
from .ribbon import ActionButton, Ribbon, action_tooltip

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SCENE_FILTER = "BioGrafix scene (*.json)"
RESOURCES = os.path.join(PROJECT_ROOT, "biografix", "resources")
# sample images: a portrait (Unsplash, CC0) and a synthetic fingerprint
DEMO_IMAGES = [os.path.join(RESOURCES, name) for name in ("portrait.png", "fingerprint.png")]

# "More" tab: one group of tiles per course assignment
TASKS = [
    ("Task 1 – Thresholding", ["bin.manual", "bin.niblack", "bin.sauvola", "bin.phansalkar", "bin.otsu"]),
]


TOOL_WINDOWS = [
    ("cmyk", "RGB ↔ CMYK converter"),
    ("cube_points", "3D RGB cube (points)"),
    ("cube_sliced", "3D RGB cube (slices)"),
    ("cone_points", "3D HSV cone (points)"),
    ("cone_full", "3D HSV cone (solid)"),
]


# quick action bar: groups of (action key, button text)
QUICK_BAR = [
    [("load", "Open images"), ("demo", "Load demo"), ("open_scene", "Open scene"),
     ("save_scene", "Save scene"), ("save_image", "Save image")],
    [("undo", "↶ Undo"), ("redo", "↷ Redo"), ("reset", "Restore original")],
    [("roi_tool", "Select region"), ("clear_roi", "Clear region")],
    [("fit", "Fit"), ("zoom100", "100%")],
]

# shortcuts listed in the help window (F1) in addition to TOOLS
SHORTCUTS = [
    ("Ctrl+Z / Ctrl+Y", "Undo / redo (full list of steps: **History** panel)"),
    ("Ctrl+O", "Open images (several at once)"),
    ("Ctrl+S", "Save scene"),
    ("Ctrl+E", "Save the selected image"),
    ("Ctrl+D / Delete", "Duplicate / delete the selected object"),
    ("Ctrl+R", "Restore original image"),
    ("Ctrl+0 / Ctrl+1", "Fit to window / 100%"),
    ("Mouse wheel", "Zoom at the cursor"),
    ("Middle button", "Pan the view"),
    ("Esc", "Cancel drawing / deselect"),
    ("Ctrl+Alt+= / Ctrl+Alt+-", "Larger / smaller font"),
    ("F11", "Full screen"),
    ("F1", "This help"),
]
HELP_NOTES = [
    "Images can also be **dragged from the file manager** onto the canvas.",
    "**Operations** (ribbon tiles or the menu) apply to the selected image. If the image has a **region** selected (Image region tool), the operation only affects that region.",
    "Long computations run in the background – the window stays responsive and the status bar shows progress.",
]


def _fmt_seconds(seconds: float) -> str:
    if seconds < 1:
        return f"{seconds * 1000:.0f} ms"
    if seconds < 60:
        return f"{seconds:.2f} s"
    return f"{int(seconds // 60)} min {seconds % 60:.0f} s"


def _op_tooltip(op: Operation) -> str:
    lines = [f"<b>{tr(op.label)}</b>"]
    if op.description:
        lines.append(tr(op.description))
    if op.params:
        lines.append(tr("Opens a parameter dialog."))
    if op.slow:
        lines.append(tr("Slow on large images (can be limited to a region)."))
    if op.shortcut:
        lines.append(tr("Shortcut: {key}", key=op.shortcut))
    return "<br>".join(lines)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        load_all()
        self.settings = QSettings(ORG_NAME, APP_NAME)
        self.doc = Document(self)
        self.runner = OperationRunner(self.doc, self)
        self._param_memory: Dict[str, dict] = {}
        self._tool_windows: Dict[str, object] = {}
        self._figure_windows = []
        self._started_at = 0.0
        self.act: Dict[str, QAction] = {}

        # theme and font must be applied before any widget is created
        theme.apply(str(self.settings.value("theme", theme.DEFAULT_THEME)),
                    int(self.settings.value("font_pt", theme.DEFAULT_FONT_PT)))

        self.view = SceneView(self.doc, self)
        self.setCentralWidget(self.view)

        self._build_docks()
        self._build_status_bar()
        self._build_toolbar()
        self._build_menus()
        self._build_quick_bar()
        self._build_ribbon()
        self._connect()
        self.doc.remove_guard = self._guard_remove
        self._hist_index = self.doc.undo_stack.index()
        centering.install(QApplication.instance())
        self.apply_appearance()

        self._initial_geometry()
        self._update_title()
        self.view.set_tool("select")

    # ========================================================= UI construction ==
    def _build_docks(self):
        self.objects_panel = ObjectsPanel(self.doc)
        self.properties_panel = PropertiesPanel(self.doc)
        self.history_view = QUndoView(self.doc.undo_stack)
        self.history_view.setEmptyLabel(tr("Start (initial state)"))
        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setMaximumBlockCount(5000)

        self.dock_objects = self._add_dock("Objects", self.objects_panel, Qt.DockWidgetArea.RightDockWidgetArea)
        self.dock_props = self._add_dock("Properties", self.properties_panel, Qt.DockWidgetArea.RightDockWidgetArea)
        self.dock_history = self._add_dock("History", self.history_view, Qt.DockWidgetArea.RightDockWidgetArea)
        self.history_view.setToolTip(tr("Click a step to return to that state (Ctrl+Z / Ctrl+Y)"))
        self.dock_log = self._add_dock("Results", self.log, Qt.DockWidgetArea.BottomDockWidgetArea)
        self.tabifyDockWidget(self.dock_props, self.dock_history)
        self.dock_props.raise_()
        self.resizeDocks([self.dock_objects], [380], Qt.Orientation.Horizontal)
        self.resizeDocks([self.dock_log], [130], Qt.Orientation.Vertical)

    def _add_dock(self, title, widget, area) -> QDockWidget:
        dock = QDockWidget(tr(title), self)
        dock.setObjectName(title)
        dock.setWidget(widget)
        self.addDockWidget(area, dock)
        return dock

    def _build_status_bar(self):
        sb = self.statusBar()
        self.cursor_label = QLabel("")
        self.zoom_label = QLabel(tr("Zoom: {percent}%", percent=100))
        self.progress = QProgressBar()
        self.progress.setRange(0, 0)  # busy indicator
        self.progress.setMaximumWidth(160)
        self.progress.hide()
        self.time_label = QLabel("")
        self.time_label.setToolTip(tr("Duration of the current / last operation"))
        self.cancel_button = QPushButton(tr("Cancel"))
        self.cancel_button.setToolTip(tr("Cancel the running operation (its result is discarded)"))
        self.cancel_button.clicked.connect(self.cancel_operation)
        self.cancel_button.hide()
        self._tick = QTimer(self)
        self._tick.setInterval(100)
        self._tick.timeout.connect(self._update_elapsed)
        sb.addPermanentWidget(self.cursor_label)
        sb.addPermanentWidget(self.progress)
        sb.addPermanentWidget(self.time_label)
        sb.addPermanentWidget(self.cancel_button)
        sb.addPermanentWidget(self.zoom_label)

    def _build_toolbar(self):
        tb = QToolBar(tr("Tools"))
        tb.setObjectName("tools")
        tb.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextOnly)
        tb.setMovable(False)
        self.addToolBar(Qt.ToolBarArea.LeftToolBarArea, tb)
        self.tool_group = QActionGroup(self)
        self.tool_actions = {}
        for tool_id, label, key in TOOLS:
            act = QAction(tr(label), self, checkable=True)
            act.setShortcut(QKeySequence(key))
            act.setToolTip(f"{tr(label)} ({key})")
            act.triggered.connect(lambda _=False, t=tool_id: self.view.set_tool(t))
            self.tool_group.addAction(act)
            tb.addAction(act)
            self.tool_actions[tool_id] = act
        self.tool_actions["select"].setChecked(True)

        tb.addSeparator()
        self.palette = ColorPalette()
        self.palette.color_chosen.connect(self._on_color_chosen)
        self.palette.width_chosen.connect(self._on_width_chosen)
        tb.addWidget(self.palette)
        self.view.draw_color = self.palette.color
        self.view.draw_width = self.palette.pen_width
        self.properties_panel.color_provider = lambda: (self.palette.color, self.palette.pen_width)

    def _on_color_chosen(self, color: str):
        """Use the palette colour for new shapes and apply it to the selected shape."""
        self.view.draw_color = color
        self._restyle_selected("color", color, tr("Colour"))

    def _on_width_chosen(self, width: int):
        self.view.draw_width = width
        self._restyle_selected("width", width, tr("Width"))

    def _restyle_selected(self, attr, value, label):
        obj = self.doc.selected
        if obj is not None and not isinstance(obj, RasterImage) and getattr(obj, attr, None) != value:
            with self.doc.change(f"{label}: {obj.display_name}"):
                setattr(obj, attr, value)
            self.doc.notify_updated(obj)
        self.view.viewport().update()

    def _action(self, menu, text, slot, shortcut=None, tip="", key=None):
        act = QAction(tr(text), self)
        if key:
            self.act[key] = act
        if shortcut:
            if isinstance(shortcut, (list, tuple)):
                act.setShortcuts([QKeySequence(s) for s in shortcut])
            else:
                act.setShortcut(QKeySequence(shortcut))
        if tip:
            act.setStatusTip(tr(tip))
            act.setToolTip(tr(tip))
        act.triggered.connect(slot)
        menu.addAction(act)
        return act

    def _build_menus(self):
        mb = self.menuBar()

        # --- File ---
        m = mb.addMenu(tr("&File"))
        self._action(m, "New scene", self.new_scene, QKeySequence.StandardKey.New)
        self._action(m, "Open images…", self.load_images, QKeySequence.StandardKey.Open,
                     "PNG, JPEG, BMP, TIFF, PPM (P3/P6) – multiple files can be selected", key="load")
        self._action(m, "Load demo images", self.load_demo, "Ctrl+Shift+D", "Portrait and fingerprint bundled with the app",
                     key="demo")
        m.addSeparator()
        self._action(m, "Open scene (JSON)…", self.open_scene, "Ctrl+Shift+O", "Scene with images and shapes", key="open_scene")
        self._action(m, "Save scene", self.save_scene, QKeySequence.StandardKey.Save, "Save the whole scene as JSON", key="save_scene")
        self._action(m, "Save scene as…", self.save_scene_as, "Ctrl+Shift+S")
        m.addSeparator()
        self._action(m, "Save image as…", self.save_image_as, "Ctrl+E", "Save the selected image (PNG, JPEG, BMP…)", key="save_image")
        m.addSeparator()
        self._action(m, "Quit", self.close, QKeySequence.StandardKey.Quit)

        # --- Edit ---
        m = mb.addMenu(tr("&Edit"))
        undo = self.doc.undo_stack.createUndoAction(self, tr("Undo"))
        undo.setShortcut(QKeySequence.StandardKey.Undo)
        redo = self.doc.undo_stack.createRedoAction(self, tr("Redo"))
        redo.setShortcuts([QKeySequence("Ctrl+Y"), QKeySequence("Ctrl+Shift+Z")])
        undo.setToolTip(tr("Undo the last change"))
        redo.setToolTip(tr("Redo the undone change"))
        m.addAction(undo)
        m.addAction(redo)
        self.act["undo"], self.act["redo"] = undo, redo
        m.addSeparator()
        self._action(m, "Duplicate", self.duplicate_selected, "Ctrl+D")
        self._action(m, "Delete", self.delete_selected, QKeySequence.StandardKey.Delete)
        self._action(m, "Clear scene", self.doc.clear)
        m.addSeparator()
        self._action(m, "Restore original image", self.reset_image, "Ctrl+R",
                     "Restores the pixels as loaded (added to history, so it can be undone)", key="reset")
        self._action(m, "Clear region selection", self.clear_roi, "Ctrl+Shift+A", "Operations apply to the whole image again", key="clear_roi")

        # --- View ---
        m = mb.addMenu(tr("&View"))
        self._action(m, "Zoom in", self.view.zoom_in, QKeySequence.StandardKey.ZoomIn)
        self._action(m, "Zoom out", self.view.zoom_out, QKeySequence.StandardKey.ZoomOut)
        self._action(m, "Fit to window", self.view.zoom_fit, "Ctrl+0", "Show all objects", key="fit")
        self._action(m, "Actual size (100%)", self.view.zoom_reset, "Ctrl+1", "1 image pixel = 1 screen pixel", key="zoom100")
        m.addSeparator()
        pv = QAction(tr("Pixel values at high zoom"), self, checkable=True, checked=True)
        pv.toggled.connect(self._toggle_pixel_values)
        m.addAction(pv)
        self.fullscreen_action = self._action(m, "Full screen", self.toggle_fullscreen, "F11",
                                              "Frameless full-screen window (F11 to exit)")
        m.addSeparator()
        theme_menu = m.addMenu(tr("Theme"))
        self.theme_group = QActionGroup(self)
        for name, t in theme.THEMES.items():
            a = QAction(tr(t["label"]), self, checkable=True, checked=name == theme.current_name())
            a.triggered.connect(lambda _=False, n=name: self.apply_appearance(name=n))
            self.theme_group.addAction(a)
            theme_menu.addAction(a)
        language_menu = m.addMenu(tr("Language"))
        self.language_group = QActionGroup(self)
        for code, name in i18n.LANGUAGES.items():
            a = QAction(name, self, checkable=True, checked=code == i18n.language())
            a.triggered.connect(lambda _=False, c=code: self.change_language(c))
            self.language_group.addAction(a)
            language_menu.addAction(a)
        self._action(m, "Larger font", lambda: self.apply_appearance(pt=theme.current_font_pt() + 1), "Ctrl+Alt+=")
        self._action(m, "Smaller font", lambda: self.apply_appearance(pt=theme.current_font_pt() - 1), "Ctrl+Alt+-")
        self._action(m, "Default font", lambda: self.apply_appearance(pt=theme.DEFAULT_FONT_PT), "Ctrl+Alt+0")
        m.addSeparator()
        self.ribbon_bottom_action = QAction(tr("Algorithm ribbon at the bottom"), self, checkable=True)
        self.ribbon_bottom_action.setChecked(self.settings.value("ribbon_bottom", "false") in (True, "true"))
        self.ribbon_bottom_action.toggled.connect(self._place_ribbon)
        m.addAction(self.ribbon_bottom_action)
        m.addSeparator()
        for dock in (self.dock_objects, self.dock_props, self.dock_history, self.dock_log):
            m.addAction(dock.toggleViewAction())

        # --- registered operations, one submenu per category ---
        algorithms = mb.addMenu(tr("&Algorithms"))
        for category, ops in registry.operations_by_category().items():
            menu = algorithms.addMenu(tr(category))
            menu.setToolTipsVisible(True)
            for section, section_ops in registry.group_by_section(ops).items():
                if section:
                    menu.addSection(tr(section))
                for op in section_ops:
                    menu.addAction(self._operation_action(op))

        # --- tool windows ---
        m = mb.addMenu(tr("&Tools"))
        for key, text in TOOL_WINDOWS:
            if key == "cube_points":
                m.addSeparator()
            self._action(m, text, lambda _=False, k=key: self._open_tool(k), key=f"tool_{key}")

        # --- Help ---
        m = mb.addMenu(tr("&Help"))
        self._action(m, "Shortcuts and usage", self._show_shortcuts, "F1", "Keyboard shortcuts and mouse controls")
        self._action(m, "About", self._show_about)

    def _build_ribbon(self):
        """Build the ribbon: one tab per operation category, plus Tools and More."""
        self.ribbon = Ribbon()
        self.ribbon_bar = QToolBar(tr("Algorithm ribbon"))
        self.ribbon_bar.setObjectName("ribbonBar")
        self.ribbon_bar.setMovable(False)
        holder = QWidget()  # spacing between the quick action bar and the ribbon tabs
        holder.setObjectName("ribbonHolder")
        lay = QVBoxLayout(holder)
        lay.setContentsMargins(0, 12, 0, 0)
        lay.setSpacing(0)
        lay.addWidget(self.ribbon)
        self.ribbon_bar.addWidget(holder)
        rb = self.ribbon

        for accent, (category, ops) in enumerate(registry.operations_by_category().items()):
            page = rb.add_page(tr(category))
            for section, section_ops in registry.group_by_section(ops).items():
                group = page.add_group(tr(section or category))
                for op in section_ops:
                    rb.add_tile(group, tr(op.label), lambda oid=op.id: self.run_operation(oid), _op_tooltip(op),
                                accent % len(theme.CATEGORY_ACCENTS))

        page = rb.add_page(tr("Tools"))
        g = page.add_group(tr("Colour models"))
        for key, text in TOOL_WINDOWS:
            rb.add_action_tile(g, self.act[f"tool_{key}"], tr(text))

        if TASKS:
            self._build_more_page()

        self._place_ribbon(self.ribbon_bottom_action.isChecked())

    def _build_more_page(self):
        """Build the More tab with the methods required by each assignment."""
        rb = self.ribbon
        page = rb.add_page(tr("More"))
        for title, op_ids in TASKS:
            g = page.add_group(tr(title))
            for op_id in op_ids:
                op = registry.get(op_id)
                rb.add_tile(g, tr(op.label), lambda oid=op_id: self.run_operation(oid), _op_tooltip(op))

    def _build_quick_bar(self):
        """Always-visible bar with file, history, region and view actions."""
        bar = QToolBar(tr("Quick actions"))
        bar.setObjectName("quickBar")
        bar.setMovable(False)
        self.addToolBar(Qt.ToolBarArea.TopToolBarArea, bar)
        self.act["roi_tool"] = self.tool_actions["roi"]
        for i, group in enumerate(QUICK_BAR):
            if i:
                bar.addSeparator()
            for key, text in group:
                b = ActionButton()
                b.setObjectName("quick")
                b.setText(tr(text))
                b.setToolTip(action_tooltip(self.act[key], tr(text)))
                b.bind_action(self.act[key])
                bar.addWidget(b)
        self.quick_bar = bar

    def _initial_geometry(self):
        """Centre the window at 1920×1080, or the available screen size if smaller."""
        screen = (self.screen() or QApplication.primaryScreen()).availableGeometry()
        w, h = min(1920, screen.width()), min(1080, screen.height())
        self.resize(w, h)
        self.move(screen.x() + (screen.width() - w) // 2, screen.y() + (screen.height() - h) // 2)

    def toggle_fullscreen(self):
        if self.isFullScreen():
            self.showMaximized()
        else:
            self.showFullScreen()

    def _place_ribbon(self, bottom: bool):
        area = Qt.ToolBarArea.BottomToolBarArea if bottom else Qt.ToolBarArea.TopToolBarArea
        self.removeToolBarBreak(self.ribbon_bar)
        self.removeToolBar(self.ribbon_bar)
        self.addToolBar(area, self.ribbon_bar)
        if not bottom:
            self.insertToolBarBreak(self.ribbon_bar)  # below the quick action bar rather than next to it
        self.ribbon_bar.show()
        self.settings.setValue("ribbon_bottom", "true" if bottom else "false")

    def apply_appearance(self, name=None, pt=None):
        """Apply the theme and font size and store them in the settings."""
        theme.apply(name, pt)
        self.settings.setValue("theme", theme.current_name())
        self.settings.setValue("font_pt", theme.current_font_pt())
        for a in self.theme_group.actions():
            a.setChecked(a.text() == tr(theme.colors()["label"]))
        self.view.apply_theme()
        self.ribbon.refresh_sizes()
        theme.style_title_bar(self)
        self.statusBar().showMessage(tr("Theme: {theme}, font {pt} pt", theme=tr(theme.colors()["label"]),
                                        pt=theme.current_font_pt()), 3000)

    def change_language(self, code: str):
        """Save the language; it takes effect after a restart."""
        if code == i18n.language():
            return
        self.settings.setValue("language", code)
        i18n.set_language(code)  # ask in the newly chosen language
        ans = QMessageBox.question(self, tr("Language"), tr("The language changes after a restart. Restart now?"))
        if ans == QMessageBox.StandardButton.Yes and self.close():
            QProcess.startDetached(sys.executable, sys.argv)

    def _operation_action(self, op: Operation) -> QAction:
        act = QAction(tr(op.label) + ("…" if op.params else ""), self)
        tip = tr(op.description or op.label)
        if op.slow:
            tip += "\n" + tr("(slow on large images)")
        act.setToolTip(tip)
        act.setStatusTip(tip.replace("\n", " "))
        if op.shortcut:
            act.setShortcut(QKeySequence(op.shortcut))
        act.triggered.connect(lambda _=False, oid=op.id: self.run_operation(oid))
        return act

    def _connect(self):
        self.view.cursor_info.connect(self.cursor_label.setText)
        self.view.files_dropped.connect(self.load_images)
        self.view.zoom_changed.connect(lambda z: self.zoom_label.setText(tr("Zoom: {percent}%", percent=f"{z * 100:.0f}")))
        self.view.status_message.connect(lambda s: self.statusBar().showMessage(s, 8000))
        self.objects_panel.focus_requested.connect(self.view.focus_object)
        self.doc.undo_stack.cleanChanged.connect(self._update_title)
        self.doc.undo_stack.indexChanged.connect(self._update_title)
        self.doc.undo_stack.indexChanged.connect(self._on_history_moved)

        self.runner.started.connect(self._on_op_started)
        self.runner.finished.connect(self._on_op_finished)
        self.runner.failed.connect(self._on_op_failed)
        self.runner.report.connect(self._on_report)
        self.runner.figure.connect(self._on_figure)
        self.runner.cancelled.connect(self._on_op_cancelled)

    def _update_title(self, *_):
        title = APP_NAME
        if self.doc.file_path:
            title += f" — {os.path.basename(self.doc.file_path)}"
        self.setWindowTitle(title + ("" if self.doc.undo_stack.isClean() else " *"))

    def _on_history_moved(self, index: int):
        """Report undo / redo (including clicks in the History panel) in the status bar and log."""
        old, self._hist_index = self._hist_index, index
        if self.doc.recording or index == old:
            return
        stack = self.doc.undo_stack
        if index < old:
            steps = [stack.text(i) for i in range(old - 1, index - 1, -1)]
            verb = tr("Undone")
        else:
            steps = [stack.text(i) for i in range(old, index)]
            verb = tr("Redone")
        if len(steps) == 1:
            msg = f"{verb}: {steps[0]}"
        else:
            msg = tr("{verb} {n} steps: ", verb=verb, n=len(steps)) + "; ".join(steps)
        self.statusBar().showMessage(msg, 8000)
        self._log(msg)

    def _toggle_pixel_values(self, on):
        self.view.show_pixel_values = on
        self.view.viewport().update()

    # ============================================================== operations ==
    def run_operation(self, op_id: str):
        op: Operation = registry.get(op_id)
        if op.target == TARGET_SHAPE:
            target = self.doc.selected
            if target is None or (op.shape_kinds and target.kind not in op.shape_kinds):
                kinds = ", ".join(op.shape_kinds) or tr("a shape")
                QMessageBox.information(self, tr(op.label), tr("Select a shape ({kinds}) to apply the operation to.", kinds=kinds))
                return
        else:
            target = self.doc.target_image()
            if target is None:
                QMessageBox.information(self, tr(op.label), tr("Select an image to apply the operation to\n(load one with File → Open images…)."))
                return
            if self.runner.busy:
                self.statusBar().showMessage(tr("Please wait – the previous operation is still running."), 5000)
                return

        values = {}
        if op.params:
            initial = {p.name: p.default for p in op.params}
            initial.update(self._param_memory.get(op.id, {}))
            if op.dynamic_defaults:
                initial.update(op.dynamic_defaults(target, {"pivot": self.view.pivot}))
            dlg = ParamsDialog(op, initial, self)
            if dlg.exec() != ParamsDialog.DialogCode.Accepted:
                return
            self._param_memory[op.id] = dlg.raw
            values = dlg.values

        try:
            self.runner.run(op, values)
        except OperationError as e:
            QMessageBox.information(self, tr(op.label), str(e))

    def _on_op_started(self, label):
        self._started_at = time.perf_counter()
        self.progress.show()
        self.statusBar().showMessage(tr("Processing: {label}…", label=label))
        if self.runner.busy:  # background operation: show the timer and the cancel button
            self.cancel_button.show()
            self._update_elapsed()
            self._tick.start()
        QApplication.setOverrideCursor(Qt.CursorShape.BusyCursor)

    def _update_elapsed(self):
        self.time_label.setText(tr("Time: {s} s", s=f"{time.perf_counter() - self._started_at:.1f}"))

    def _end_busy(self):
        self._tick.stop()
        self.progress.hide()
        self.cancel_button.hide()
        while QApplication.overrideCursor() is not None:
            QApplication.restoreOverrideCursor()

    def _on_op_finished(self, label, seconds):
        self._end_busy()
        self.time_label.setText(tr("Last: {time}", time=_fmt_seconds(seconds)))
        self.statusBar().showMessage(tr("Done: {label} ({time})", label=label, time=_fmt_seconds(seconds)), 8000)
        self._log(f"{label} – {_fmt_seconds(seconds)}")

    def _on_op_cancelled(self, label):
        self._end_busy()
        self.time_label.setText(tr("Cancelled"))
        self.statusBar().showMessage(tr("Cancelled: {label}", label=label), 8000)
        self._log(tr("CANCELLED: {label} (after {s} s)", label=label, s=f"{time.perf_counter() - self._started_at:.1f}"))

    def cancel_operation(self):
        if self.runner.busy:
            self.runner.cancel()

    def _ask_busy(self, title: str, action_text: str) -> bool:
        """Ask what to do with a running operation; True means cancel it and proceed."""
        img = self.doc.find(self.runner.busy_oid) if self.runner.busy_oid else None
        name = f'"{img.display_name}"' if img else ""
        box = QMessageBox(self)
        box.setIcon(QMessageBox.Icon.Warning)
        box.setWindowTitle(tr(title))
        box.setText(tr("An operation is running on image {name}:\n{label}\n(running for {s} s)",
                       name=name, label=self.runner.busy_label, s=f"{time.perf_counter() - self._started_at:.1f}"))
        box.setInformativeText(tr("You can wait for it to finish or cancel it."))
        keep = box.addButton(tr("Keep running"), QMessageBox.ButtonRole.RejectRole)
        stop = box.addButton(tr(action_text), QMessageBox.ButtonRole.DestructiveRole)
        box.setDefaultButton(keep)
        box.exec()
        if box.clickedButton() is stop:
            self.runner.cancel()
            return True
        return False

    def _guard_remove(self, oids) -> bool:
        """Deletion veto: ask before deleting an image that is being processed."""
        busy = self.runner.busy_oid
        if busy is None or busy not in oids:
            return True
        return self._ask_busy("Image is being processed", "Cancel and delete")

    def _on_op_failed(self, label, text):
        self._end_busy()
        self._log(tr("ERROR: {label}", label=label) + f"\n{text}")
        last_line = text.strip().splitlines()[-1] if text.strip() else text
        if ": " in last_line:  # translate the message part of "ValueError: message"
            kind, message = last_line.split(": ", 1)
            last_line = f"{kind}: {tr(message)}"
        QMessageBox.warning(self, tr("Operation failed"), f"{label}\n\n{last_line}")

    def _on_report(self, title, text):
        self._log(f"[{title}] {text}")
        self.dock_log.show()
        self.dock_log.raise_()
        self.statusBar().showMessage(text, 15000)

    def _on_figure(self, fig, title):
        win = FigureWindow(fig, title, self)
        self._figure_windows = [w for w in self._figure_windows if w.isVisible()] + [win]
        win.show()

    def _log(self, text):
        self.log.appendPlainText(f"{time.strftime('%H:%M:%S')}  {text}")

    # ================================================================== files ==
    def _confirm_discard(self) -> bool:
        if self.doc.undo_stack.isClean() or not self.doc.objects:
            return True
        ans = QMessageBox.question(
            self, tr("Unsaved changes"), tr("The scene has unsaved changes. Save it before continuing?"),
            QMessageBox.StandardButton.Save | QMessageBox.StandardButton.Discard | QMessageBox.StandardButton.Cancel,
        )
        if ans == QMessageBox.StandardButton.Save:
            return self.save_scene()
        return ans == QMessageBox.StandardButton.Discard

    def new_scene(self):
        if self.doc.can_remove(o.oid for o in self.doc.objects) and self._confirm_discard():
            self.doc.reset([])

    def _last_dir(self, key="last_dir"):
        return self.settings.value(key, PROJECT_ROOT)

    def load_images(self, paths=None):
        if not paths:
            paths = file_dialogs.open_files(self, tr("Open images"), self._last_dir(), tr(OPEN_FILTER))
        if not paths:
            return
        self.settings.setValue("last_dir", os.path.dirname(paths[0]))
        objs, errors = [], []
        # place new images to the right of existing objects
        x = max((o.bbox()[2] for o in self.doc.objects), default=-10) + 20
        for path in paths:
            try:
                pixels = load_image(path)
            except Exception as e:
                errors.append(f"{os.path.basename(path)}: {tr(str(e))}")
                continue
            img = RasterImage(x=x, y=10, pixels=pixels, src=path, name=os.path.basename(path))
            objs.append(img)
            x += img.w + 20
        if errors:
            QMessageBox.warning(self, tr("Could not load"), "\n".join(errors))
        if objs:
            label = (tr("Loaded image: {name}", name=objs[0].name) if len(objs) == 1
                     else tr("Loaded images ({n})", n=len(objs)))
            self.doc.add_objects(objs, label)
            self.view.focus_object(objs[0].oid)

    def load_demo(self):
        """Load the bundled sample images."""
        missing = [p for p in DEMO_IMAGES if not os.path.exists(p)]
        if missing:
            QMessageBox.warning(self, tr("Demo images"), tr("File not found:") + "\n" + "\n".join(missing))
            return
        self.load_images(DEMO_IMAGES)

    def open_scene(self):
        if not self.doc.can_remove(o.oid for o in self.doc.objects) or not self._confirm_discard():
            return
        path = file_dialogs.open_file(self, tr("Open scene"), self._last_dir(), tr(SCENE_FILTER))
        if not path:
            return
        try:
            objects = load_scene(path)
        except Exception as e:
            QMessageBox.critical(self, tr("Open scene"), tr("Could not load the file:") + f"\n{e}")
            return
        self.settings.setValue("last_dir", os.path.dirname(path))
        self.doc.reset(objects, path)
        self.view.zoom_fit()

    def save_scene(self) -> bool:
        if not self.doc.file_path:
            return self.save_scene_as()
        try:
            save_scene(self.doc.file_path, self.doc.objects)
        except Exception as e:
            QMessageBox.critical(self, tr("Save scene"), tr("Could not save:") + f"\n{e}")
            return False
        self.doc.undo_stack.setClean()
        self.statusBar().showMessage(tr("Saved: {path}", path=self.doc.file_path), 5000)
        return True

    def save_scene_as(self) -> bool:
        path = file_dialogs.save_file(self, tr("Save scene"), self._last_dir(), tr(SCENE_FILTER))
        if not path:
            return False
        if not path.lower().endswith(".json"):
            path += ".json"
        self.doc.file_path = path
        self.settings.setValue("last_dir", os.path.dirname(path))
        return self.save_scene()

    def save_image_as(self):
        img = self.doc.target_image()
        if img is None:
            QMessageBox.information(self, tr("Save image"), tr("Select an image to save."))
            return
        base = os.path.splitext(img.name or tr("image"))[0]
        path = file_dialogs.save_file(self, tr("Save image"), os.path.join(self._last_dir(), base + ".png"), SAVE_FILTER)
        if not path:
            return
        if not os.path.splitext(path)[1]:
            path += ".png"
        pixels = img.pixels
        if (img.w, img.h) != (img.src_w, img.src_h):
            ans = QMessageBox.question(
                self, tr("Output size"),
                tr("Save at the displayed size {w}×{h}?\n(No = source size {sw}×{sh})",
                   w=img.w, h=img.h, sw=img.src_w, sh=img.src_h),
            )
            if ans == QMessageBox.StandardButton.Yes:
                # nearest-neighbour scaling
                ys = (np.arange(img.h) * img.src_h // img.h)
                xs = (np.arange(img.w) * img.src_w // img.w)
                pixels = img.pixels[ys][:, xs]
        quality = 90
        if path.lower().endswith((".jpg", ".jpeg")):
            quality, ok = QInputDialog.getInt(self, tr("JPEG quality"), tr("Quality (1–100):"), 90, 1, 100)
            if not ok:
                return
        try:
            save_image(path, pixels, quality)
        except Exception as e:
            QMessageBox.critical(self, tr("Save image"), tr("Could not save:") + f"\n{e}")
            return
        self.settings.setValue("last_dir", os.path.dirname(path))
        self.statusBar().showMessage(tr("Saved image: {path}", path=path), 5000)

    # ================================================================ editing ==
    def duplicate_selected(self):
        if self.doc.selected_oid:
            self.doc.duplicate(self.doc.selected_oid)

    def delete_selected(self):
        if self.doc.selected_oid:
            self.doc.remove(self.doc.selected_oid)

    def reset_image(self):
        img = self.doc.target_image()
        if img is None:
            return
        if img.pixels is img.original:
            self.statusBar().showMessage(tr("The image is already in its original state."), 4000)
            return
        with self.doc.change(tr("Restore original: {name}", name=img.display_name)):
            img.set_pixels(img.original)

    def clear_roi(self):
        img = self.doc.target_image()
        if img is not None and img.roi:
            self.doc.set_roi(img, None)

    # ================================================================== tools ==
    def _open_tool(self, key):
        from .tools.color_converter import ColorConverterWindow
        from .tools.color_spaces_3d import HSVConeWindow, RGBCubeWindow

        win = self._tool_windows.get(key)
        if win is None:
            factories = {
                "cmyk": lambda: ColorConverterWindow(self),
                "cube_points": lambda: RGBCubeWindow(False, self),
                "cube_sliced": lambda: RGBCubeWindow(True, self),
                "cone_points": lambda: HSVConeWindow(False, self),
                "cone_full": lambda: HSVConeWindow(True, self),
            }
            win = factories[key]()
            self._tool_windows[key] = win
        win.show()
        win.raise_()
        win.activateWindow()

    # =================================================================== help ==
    def _show_shortcuts(self):
        rows = [(key, tr(label)) for _id, label, key in TOOLS] + [(tr(key), tr(text)) for key, text in SHORTCUTS]
        table = "\n".join(f"| {key} | {text} |" for key, text in rows)
        text = (f"# {tr("Usage")}\n\n| {tr("Shortcut")} | {tr("Action")} |\n|---|---|\n{table}\n\n"
                + "\n\n".join(tr(p) for p in HELP_NOTES))
        MarkdownDialog(tr("Shortcuts and usage"), text, self).exec()

    def _show_about(self):
        n = len(registry.REGISTRY)
        QMessageBox.about(self, tr("About"), f"<b>{APP_TITLE}</b><br><br>"
                          + tr("Image processing and graphics.") + "<br>" + tr("Available operations: {n}.", n=n))

    # ================================================================ closing ==
    def closeEvent(self, event):
        if self.runner.busy and not self._ask_busy("Operation in progress", "Cancel and quit"):
            event.ignore()
            return
        if self._confirm_discard():
            QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
            self.runner.shutdown()  # cancelled workers must finish before the application exits
            QApplication.restoreOverrideCursor()
            event.accept()
        else:
            event.ignore()

