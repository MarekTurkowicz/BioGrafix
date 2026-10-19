"""Side panels: object list and properties of the selected object."""

from __future__ import annotations

import os

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QFontDatabase
from PySide6.QtWidgets import (
    QComboBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from ..core.document import Document
from ..i18n import tr
from ..scene.objects import SHAPE_TYPES, RasterImage


def _mono_edit(height=80) -> QPlainTextEdit:
    w = QPlainTextEdit()
    w.setFont(QFontDatabase.systemFont(QFontDatabase.SystemFont.FixedFont))
    w.setFixedHeight(height)
    return w


# buttons below the object list: (text, tooltip, action key)
OBJECT_BUTTONS = [
    ("▲", "Move up (to front)", "up"),
    ("▼", "Move down", "down"),
    ("Duplicate", "Duplicate (Ctrl+D)", "duplicate"),
    ("Delete", "Delete (Del)", "remove"),
]


class ObjectsPanel(QWidget):
    """Scene objects, topmost first."""

    focus_requested = Signal(str)

    def __init__(self, doc: Document, parent=None):
        super().__init__(parent)
        self.doc = doc
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        self.list = QListWidget()
        layout.addWidget(self.list)

        row = QHBoxLayout()
        slots = {"up": lambda: self._move(+1), "down": lambda: self._move(-1),
                 "duplicate": self._duplicate, "remove": self._remove}
        for text, tip, key in OBJECT_BUTTONS:
            b = QPushButton(tr(text))
            b.setToolTip(tr(tip))
            b.clicked.connect(slots[key])
            row.addWidget(b)
        layout.addLayout(row)

        self.list.currentItemChanged.connect(self._on_current_changed)
        self.list.itemDoubleClicked.connect(lambda it: self.focus_requested.emit(it.data(Qt.ItemDataRole.UserRole)))
        doc.changed.connect(self.refresh)
        doc.selection_changed.connect(self._sync_selection)
        self.refresh()

    def refresh(self):
        self.list.blockSignals(True)
        self.list.clear()
        for obj in reversed(self.doc.objects):
            text = f"{tr(obj.kind_label)}: {obj.display_name}"
            if isinstance(obj, RasterImage):
                text += f"  ({obj.src_w}×{obj.src_h})"
            item = QListWidgetItem(text)
            item.setData(Qt.ItemDataRole.UserRole, obj.oid)
            self.list.addItem(item)
        self.list.blockSignals(False)
        self._sync_selection()

    def _sync_selection(self):
        self.list.blockSignals(True)
        self.list.setCurrentRow(-1)
        for i in range(self.list.count()):
            if self.list.item(i).data(Qt.ItemDataRole.UserRole) == self.doc.selected_oid:
                self.list.setCurrentRow(i)
                break
        self.list.blockSignals(False)

    def _on_current_changed(self, current, _previous):
        self.doc.select(current.data(Qt.ItemDataRole.UserRole) if current else None)

    def _move(self, delta):
        if self.doc.selected_oid:
            self.doc.move_z(self.doc.selected_oid, delta)

    def _duplicate(self):
        if self.doc.selected_oid:
            self.doc.duplicate(self.doc.selected_oid)

    def _remove(self):
        if self.doc.selected_oid:
            self.doc.remove(self.doc.selected_oid)


NEW_SHAPE_DEFAULTS = {
    "line": "100,100,300,200",
    "rect": "100,100,300,200",
    "circle": "250,220,80",
    "polygon": "100,100\n250,80\n300,220\n150,260",
    "bezier": "80,250\n180,80\n320,80\n420,250",
}


class PropertiesPanel(QWidget):
    """Parameters of the selected object and creation of shapes from typed values."""

    def __init__(self, doc: Document, parent=None):
        super().__init__(parent)
        self.color_provider = None  # returns (colour, width) from the palette; set by the main window
        self.doc = doc
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)

        # --- selected object ---
        self.sel_box = QGroupBox(tr("Selected object"))
        sel = QFormLayout(self.sel_box)
        self.kind_label = QLabel("—")
        self.name_edit = QLineEdit()
        self.name_edit.setPlaceholderText(tr("name"))
        self.name_edit.editingFinished.connect(self._rename)
        self.hint_label = QLabel("")
        self.hint_label.setWordWrap(True)
        self.hint_label.setStyleSheet("color:#666")
        self.params_edit = _mono_edit(90)
        self.apply_btn = QPushButton(tr("Apply parameters"))
        self.apply_btn.clicked.connect(self._apply)
        self.info_label = QLabel("")
        self.info_label.setWordWrap(True)
        sel.addRow(tr("Type:"), self.kind_label)
        sel.addRow(tr("Name:"), self.name_edit)
        sel.addRow(self.hint_label)
        sel.addRow(self.params_edit)
        sel.addRow(self.apply_btn)
        sel.addRow(self.info_label)
        layout.addWidget(self.sel_box)

        # --- new shape from parameters ---
        new_box = QGroupBox(tr("New shape from parameters"))
        nl = QVBoxLayout(new_box)
        self.new_kind = QComboBox()
        for kind, cls in SHAPE_TYPES.items():
            self.new_kind.addItem(tr(cls.kind_label), kind)
        self.new_hint = QLabel("")
        self.new_hint.setWordWrap(True)
        self.new_hint.setObjectName("hint")
        self.new_params = _mono_edit(80)
        draw_btn = QPushButton(tr("Draw"))
        draw_btn.clicked.connect(self._draw_new)
        self.new_kind.currentIndexChanged.connect(self._on_new_kind)
        nl.addWidget(self.new_kind)
        nl.addWidget(self.new_hint)
        nl.addWidget(self.new_params)
        nl.addWidget(draw_btn)
        layout.addWidget(new_box)
        layout.addStretch(1)

        self._on_new_kind()
        doc.selection_changed.connect(self.refresh)
        doc.changed.connect(self.refresh)
        doc.object_updated.connect(self._on_object_updated)
        self.refresh()

    def refresh(self):
        obj = self.doc.selected
        enabled = obj is not None
        for w in (self.name_edit, self.params_edit, self.apply_btn):
            w.setEnabled(enabled)
        if obj is None:
            self.kind_label.setText(tr("— (nothing selected)"))
            self.name_edit.setText("")
            self.hint_label.setText("")
            self.params_edit.setPlainText("")
            self.info_label.setText("")
            return
        self.kind_label.setText(tr(obj.kind_label))
        self.name_edit.setText(obj.name)
        self.hint_label.setText(tr("Parameters: {hint}", hint=tr(obj.params_hint)))
        self.params_edit.setPlainText(obj.params_text())
        self._update_info(obj)

    def _update_info(self, obj):
        if isinstance(obj, RasterImage):
            ch = "RGBA" if obj.pixels.shape[2] == 4 else "RGB"
            text = tr("Source size: {w}×{h} px ({ch})<br>Displayed: {dw}×{dh}",
                      w=obj.src_w, h=obj.src_h, ch=ch, dw=obj.w, dh=obj.h)
            if obj.roi:
                x0, y0, x1, y1 = obj.roi
                text += "<br>" + tr("<b>Region:</b> ({x0}, {y0}) – ({x1}, {y1}), {w}×{h} px",
                                   x0=x0, y0=y0, x1=x1, y1=y1, w=x1 - x0, h=y1 - y0)
            if obj.src:
                text += "<br>" + tr("File: {name}", name=os.path.basename(obj.src))
            self.info_label.setText(text)
            self.info_label.setToolTip(obj.src or "")
        elif obj.kind == "bezier":
            self.info_label.setText(tr("Curve degree: {n}", n=obj.degree))
        else:
            self.info_label.setText("")

    def _on_object_updated(self, oid):
        obj = self.doc.selected
        if obj is not None and obj.oid == oid:
            if not self.params_edit.hasFocus():
                self.params_edit.setPlainText(obj.params_text())
            self._update_info(obj)

    def _rename(self):
        obj = self.doc.selected
        if obj is not None and self.name_edit.text() != obj.name:
            self.doc.rename(obj.oid, self.name_edit.text())

    def _apply(self):
        obj = self.doc.selected
        if obj is None:
            return
        try:
            with self.doc.change(tr("Apply parameters: {name}", name=obj.display_name)):
                obj.set_params_text(self.params_edit.toPlainText())
        except ValueError as e:
            QMessageBox.warning(self, tr("Parameters"), str(e))

    def _on_new_kind(self):
        kind = self.new_kind.currentData()
        self.new_hint.setText(tr("Parameters: {hint}", hint=tr(SHAPE_TYPES[kind].params_hint)))
        self.new_params.setPlainText(NEW_SHAPE_DEFAULTS[kind])

    def _draw_new(self):
        cls = SHAPE_TYPES[self.new_kind.currentData()]
        obj = cls()
        if self.color_provider:
            color, width = self.color_provider()
            obj.color, obj.width = color, width
        try:
            obj.set_params_text(self.new_params.toPlainText())
        except ValueError as e:
            QMessageBox.warning(self, tr("Parameters"), str(e))
            return
        self.doc.add_objects([obj], tr("Draw from fields: {kind}", kind=tr(obj.kind_label)))
