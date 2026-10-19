"""Drawing palette: colour swatches and stroke width.

The chosen colour and width apply to new shapes and, when a shape is selected,
to that shape as an undoable step.
"""

from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QColorDialog,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QSlider,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from ..i18n import tr

PALETTE = [
    "#ffffff", "#9e9e9e", "#000000",
    "#ff4d4d", "#ff9800", "#ffd43b",
    "#4caf50", "#26c6da", "#4a8fd4",
    "#3f51b5", "#ab47bc", "#ff6fb5",
]
DEFAULT_COLOR = "#ffd43b"
MIN_WIDTH, MAX_WIDTH = 1, 10
DEFAULT_WIDTH = 2
COLUMNS = 3


def _swatch_css(color: str, size: int, selected: bool = False) -> str:
    border = "2px solid #ffffff" if selected else "1px solid rgba(128,128,128,0.6)"
    return (f"QToolButton {{ background: {color}; border: {border}; border-radius: 4px; "
            f"min-width: {size}px; max-width: {size}px; min-height: {size}px; max-height: {size}px; padding: 0; }}"
            f"QToolButton:hover {{ border: 2px solid #4a8fd4; }}")


class ColorPalette(QWidget):
    color_chosen = Signal(str)   # #rrggbb
    width_chosen = Signal(int)   # pixels

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("palette")
        self._color = DEFAULT_COLOR
        self._width = DEFAULT_WIDTH
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 6, 8, 6)
        layout.setSpacing(6)

        title = QLabel(tr("COLOUR"))
        title.setObjectName("paletteTitle")
        layout.addWidget(title)

        self.current = QToolButton()
        self.current.setToolTip(tr("Current drawing colour – click to pick any colour"))
        self.current.setCursor(Qt.CursorShape.PointingHandCursor)
        self.current.clicked.connect(self.pick_custom)
        layout.addWidget(self.current, alignment=Qt.AlignmentFlag.AlignHCenter)

        grid = QGridLayout()
        grid.setSpacing(4)
        self._swatches = {}
        for i, color in enumerate(PALETTE):
            b = QToolButton()
            b.setToolTip(color)
            b.setCursor(Qt.CursorShape.PointingHandCursor)
            b.clicked.connect(lambda _=False, c=color: self.choose(c))
            grid.addWidget(b, i // COLUMNS, i % COLUMNS)
            self._swatches[color] = b
        layout.addLayout(grid)

        more = QToolButton()
        more.setText(tr("Other…"))
        more.setObjectName("quick")
        more.clicked.connect(self.pick_custom)
        layout.addWidget(more, alignment=Qt.AlignmentFlag.AlignHCenter)

        header = QHBoxLayout()
        title = QLabel(tr("WIDTH"))
        title.setObjectName("paletteTitle")
        header.addWidget(title)
        header.addStretch(1)
        self.width_label = QLabel()
        header.addWidget(self.width_label)
        layout.addLayout(header)

        self.width_slider = QSlider(Qt.Orientation.Horizontal)
        self.width_slider.setRange(MIN_WIDTH, MAX_WIDTH)
        self.width_slider.setPageStep(1)
        self.width_slider.setToolTip(tr("Stroke width of new shapes (and of the selected shape)"))
        self.width_slider.setCursor(Qt.CursorShape.PointingHandCursor)
        self.width_slider.setValue(self._width)
        self.width_slider.valueChanged.connect(self._on_slider_moved)
        self.width_slider.sliderReleased.connect(lambda: self.width_chosen.emit(self._width))
        layout.addWidget(self.width_slider)
        self._refresh()

    @property
    def color(self) -> str:
        return self._color

    @property
    def pen_width(self) -> int:
        return self._width

    def choose_width(self, width: int):
        """Set the width programmatically and emit it as if the slider was released."""
        self.width_slider.setValue(int(width))
        self.width_chosen.emit(self._width)

    def _on_slider_moved(self, value: int):
        self._width = value
        self._refresh()
        # while dragging only the preview changes; the selected shape is updated on release
        # so that a drag produces a single undo step
        if not self.width_slider.isSliderDown():
            self.width_chosen.emit(value)

    def choose(self, color: str):
        self._color = QColor(color).name()
        self._refresh()
        self.color_chosen.emit(self._color)

    def pick_custom(self):
        c = QColorDialog.getColor(QColor(self._color), self.window(), tr("Drawing colour"))
        if c.isValid():
            self.choose(c.name())

    def _refresh(self):
        self.width_label.setText(f"{self._width} px")
        size = max(18, self.fontMetrics().height())
        for color, b in self._swatches.items():
            b.setStyleSheet(_swatch_css(color, size, color == self._color))
        big = size * COLUMNS + 8
        self.current.setStyleSheet(
            f"QToolButton {{ background: {self._color}; border: 1px solid rgba(128,128,128,0.8); "
            f"border-radius: 6px; min-width: {big}px; max-width: {big}px; min-height: {size}px; max-height: {size}px; }}")
