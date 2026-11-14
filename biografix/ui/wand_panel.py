"""Magic wand panel: selection settings and actions on the current selection."""

from __future__ import annotations

from dataclasses import dataclass

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

from ..algorithms.segmentation import CompareMode, Connectivity
from ..core.document import Document
from ..i18n import tr
from .ribbon import ActionButton

DEFAULT_VALUE_RANGE = 20


@dataclass(frozen=True)
class MagicWandSettings:
    """Values chosen in the panel; all but ``fill_on_click`` are arguments of ``magic_wand_selection``."""

    range_below: int = DEFAULT_VALUE_RANGE
    range_above: int = DEFAULT_VALUE_RANGE
    compare_mode: CompareMode = CompareMode.GRAY
    max_pixels: int = 0
    connectivity: Connectivity = Connectivity.FOUR
    global_mode: bool = False
    fill_on_click: bool = False

    def selection_arguments(self) -> dict:
        """Keyword arguments for ``magic_wand_selection``."""
        return {
            "range_below": self.range_below,
            "range_above": self.range_above,
            "compare_mode": self.compare_mode,
            "max_pixels": self.max_pixels,
            "connectivity": self.connectivity,
            "global_mode": self.global_mode,
        }


class WandPanel(QWidget):
    """Settings read by the magic wand tool; any change re-runs the last click."""

    settings_changed = Signal()

    def __init__(self, doc: Document, parent=None):
        super().__init__(parent)
        self.doc = doc
        layout = QVBoxLayout(self)

        box = QGroupBox(tr("Magic wand"))
        form = QFormLayout(box)

        self.mode_combo = QComboBox()
        self.mode_combo.addItem(tr("Local – connected pixels (flood fill)"), False)
        self.mode_combo.addItem(tr("Global – whole image (global flood mode)"), True)
        form.addRow(tr("Mode"), self.mode_combo)

        self.compare_mode_combo = QComboBox()
        self.compare_mode_combo.addItem(tr("Gray (R+G+B)/3"), CompareMode.GRAY.value)
        self.compare_mode_combo.addItem(tr("Each RGB channel"), CompareMode.RGB.value)
        form.addRow(tr("Compare"), self.compare_mode_combo)

        self.range_below_spin = self._spin_box(0, 255, DEFAULT_VALUE_RANGE)
        self.range_above_spin = self._spin_box(0, 255, DEFAULT_VALUE_RANGE)
        form.addRow(tr("Range below the clicked value"), self.range_below_spin)
        form.addRow(tr("Range above the clicked value"), self.range_above_spin)
        self.accepted_range_label = QLabel()
        self.accepted_range_label.setWordWrap(True)
        form.addRow(self.accepted_range_label)

        self.max_pixels_spin = self._spin_box(0, 100_000_000, 0)
        self.max_pixels_spin.setSpecialValueText(tr("no limit"))
        self.max_pixels_spin.setSingleStep(100)
        form.addRow(tr("Maximum number of pixels"), self.max_pixels_spin)

        self.connectivity_combo = QComboBox()
        self.connectivity_combo.addItem(tr("4 neighbours"), Connectivity.FOUR.value)
        self.connectivity_combo.addItem(tr("8 neighbours"), Connectivity.EIGHT.value)
        form.addRow(tr("Connectivity"), self.connectivity_combo)

        self.fill_on_click_checkbox = QCheckBox(tr("Fill with the palette colour on click (like Paint)"))
        form.addRow(self.fill_on_click_checkbox)
        layout.addWidget(box)

        self.selection_actions_box = QGroupBox(tr("Selection"))
        self.selection_actions_grid = QGridLayout(self.selection_actions_box)
        layout.addWidget(self.selection_actions_box)

        self.selection_size_label = QLabel()
        self.selection_size_label.setWordWrap(True)
        layout.addWidget(self.selection_size_label)
        modifier_hint = QLabel(tr("Click: new selection · Shift+click: add · Ctrl+click: subtract"))
        modifier_hint.setWordWrap(True)
        layout.addWidget(modifier_hint)
        layout.addStretch(1)

        for combo in (self.mode_combo, self.compare_mode_combo, self.connectivity_combo):
            combo.currentIndexChanged.connect(self._on_setting_changed)
        for spin in (self.range_below_spin, self.range_above_spin, self.max_pixels_spin):
            spin.valueChanged.connect(self._on_setting_changed)
        self.doc.selection_changed.connect(self.show_selection_size)
        self.doc.object_updated.connect(lambda _oid: self.show_selection_size())
        self._update_dependent_widgets()
        self.show_selection_size()

    @staticmethod
    def _spin_box(minimum: int, maximum: int, value: int) -> QSpinBox:
        spin = QSpinBox()
        spin.setRange(minimum, maximum)
        spin.setValue(value)
        return spin

    def add_selection_buttons(self, actions: list[tuple[QAction, str]], columns: int = 2) -> None:
        """Add buttons that trigger ``actions`` (action, button text)."""
        for i, (action, text) in enumerate(actions):
            button = ActionButton()
            button.setObjectName("quick")  # same style as the quick action bar
            button.setText(text)
            button.setToolTip(action.toolTip() or text)
            button.bind_action(action)
            self.selection_actions_grid.addWidget(button, i // columns, i % columns)

    def current_settings(self) -> MagicWandSettings:
        return MagicWandSettings(
            range_below=self.range_below_spin.value(),
            range_above=self.range_above_spin.value(),
            compare_mode=CompareMode(self.compare_mode_combo.currentData()),
            max_pixels=self.max_pixels_spin.value(),
            connectivity=Connectivity(self.connectivity_combo.currentData()),
            global_mode=bool(self.mode_combo.currentData()),
            fill_on_click=self.fill_on_click_checkbox.isChecked(),
        )

    def _on_setting_changed(self, *_):
        self._update_dependent_widgets()
        self.settings_changed.emit()

    def _update_dependent_widgets(self):
        self.connectivity_combo.setEnabled(not self.mode_combo.currentData())  # global mode ignores neighbours
        self.accepted_range_label.setText(
            tr("Accepted values: [v − {below}, v + {above}], v = value of the clicked pixel",
               below=self.range_below_spin.value(), above=self.range_above_spin.value()))

    def show_selection_size(self):
        img = self.doc.target_image()
        if img is None or img.selection is None:
            self.selection_size_label.setText(tr("No selection – operations apply to the whole image (or the region)."))
            return
        selected = int(img.selection.sum())
        self.selection_size_label.setText(
            tr("Selected: {n} px ({percent}% of the image) – operations apply only to the selection.",
               n=selected, percent=f"{100 * selected / img.selection.size:.1f}"))
