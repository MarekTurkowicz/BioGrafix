"""Magic wand panel: selection settings and actions on the current selection."""

from __future__ import annotations

from PySide6.QtCore import Signal
from PySide6.QtGui import QAction
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QLabel,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from ..core.document import Document
from ..i18n import tr
from .ribbon import ActionButton

DEFAULT_RANGE = 20


class WandPanel(QWidget):
    """Settings read by the magic wand tool; any change re-runs the last click."""

    settings_changed = Signal()

    def __init__(self, doc: Document, parent=None):
        super().__init__(parent)
        self.doc = doc
        layout = QVBoxLayout(self)

        box = QGroupBox(tr("Magic wand"))
        form = QFormLayout(box)

        self.mode = QComboBox()
        self.mode.addItem(tr("Local – connected pixels (flood fill)"), False)
        self.mode.addItem(tr("Global – whole image (global flood mode)"), True)
        form.addRow(tr("Mode"), self.mode)

        self.compare = QComboBox()
        self.compare.addItem(tr("Gray (R+G+B)/3"), "gray")
        self.compare.addItem(tr("Each RGB channel"), "rgb")
        form.addRow(tr("Compare"), self.compare)

        self.below = self._spin(0, 255, DEFAULT_RANGE)
        self.above = self._spin(0, 255, DEFAULT_RANGE)
        form.addRow(tr("Range below the clicked value"), self.below)
        form.addRow(tr("Range above the clicked value"), self.above)
        self.range_hint = QLabel()
        self.range_hint.setWordWrap(True)
        form.addRow(self.range_hint)

        self.max_pixels = self._spin(0, 100_000_000, 0)
        self.max_pixels.setSpecialValueText(tr("no limit"))
        self.max_pixels.setSingleStep(100)
        form.addRow(tr("Maximum number of pixels"), self.max_pixels)

        self.connectivity = QComboBox()
        self.connectivity.addItem(tr("4 neighbours"), 4)
        self.connectivity.addItem(tr("8 neighbours"), 8)
        form.addRow(tr("Connectivity"), self.connectivity)

        self.fill = QCheckBox(tr("Fill with the palette colour on click (like Paint)"))
        form.addRow(self.fill)
        layout.addWidget(box)

        self.actions_box = QGroupBox(tr("Selection"))
        self.actions_grid = QGridLayout(self.actions_box)
        layout.addWidget(self.actions_box)

        self.info = QLabel()
        self.info.setWordWrap(True)
        layout.addWidget(self.info)
        hint = QLabel(tr("Click: new selection · Shift+click: add · Ctrl+click: subtract"))
        hint.setWordWrap(True)
        layout.addWidget(hint)
        layout.addStretch(1)

        for combo in (self.mode, self.compare, self.connectivity):
            combo.currentIndexChanged.connect(self._changed)
        for spin in (self.below, self.above, self.max_pixels):
            spin.valueChanged.connect(self._changed)
        self.fill.toggled.connect(self._update_hints)
        self.doc.selection_changed.connect(self.update_info)
        self.doc.object_updated.connect(lambda _oid: self.update_info())
        self._update_hints()
        self.update_info()

    @staticmethod
    def _spin(lo: int, hi: int, value: int) -> QSpinBox:
        spin = QSpinBox()
        spin.setRange(lo, hi)
        spin.setValue(value)
        return spin

    def add_actions(self, actions: list[tuple[QAction, str]], columns: int = 2) -> None:
        """Add buttons that trigger ``actions`` (action, button text)."""
        for i, (action, text) in enumerate(actions):
            button = ActionButton()
            button.setObjectName("quick")  # same style as the quick action bar
            button.setText(text)
            button.setToolTip(action.toolTip() or text)
            button.bind_action(action)
            self.actions_grid.addWidget(button, i // columns, i % columns)

    def settings(self) -> dict:
        """Keyword arguments for the magic wand, plus ``fill``."""
        return {
            "below": self.below.value(),
            "above": self.above.value(),
            "compare": self.compare.currentData(),
            "max_pixels": self.max_pixels.value(),
            "connectivity": self.connectivity.currentData(),
            "global_mode": self.mode.currentData(),
            "fill": self.fill.isChecked(),
        }

    def _changed(self, *_):
        self._update_hints()
        self.settings_changed.emit()

    def _update_hints(self, *_):
        self.connectivity.setEnabled(not self.mode.currentData())
        self.range_hint.setText(tr("Accepted values: [v − {below}, v + {above}], v = value of the clicked pixel",
                                   below=self.below.value(), above=self.above.value()))

    def update_info(self):
        img = self.doc.target_image()
        if img is None or img.selection is None:
            self.info.setText(tr("No selection – operations apply to the whole image (or the region)."))
            return
        n = int(img.selection.sum())
        total = img.selection.size
        self.info.setText(tr("Selected: {n} px ({percent}% of the image) – operations apply only to the selection.",
                             n=n, percent=f"{100 * n / total:.1f}"))
