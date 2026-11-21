"""Ribbon: category tabs with operation tiles grouped into sections.

The ribbon is generated from the operation registry. All tiles share one size
so that they line up in a regular grid.
"""

from __future__ import annotations

from collections.abc import Callable

from PySide6.QtCore import Qt
from PySide6.QtGui import QAction
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QScrollArea,
    QSizePolicy,
    QTabWidget,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

WRAP_AT = 13          # labels longer than this wrap onto two lines
TILE_LINES = 3.2      # tile height in text lines
TILE_MIN_EM = 8       # minimum tile width in em
TILE_MAX_EM = 13      # maximum tile width in em


def wrap_label(text: str) -> str:
    """Wrap a label onto two lines at the space closest to its middle."""
    if "\n" in text or len(text) <= WRAP_AT or " " not in text:
        return text
    mid = len(text) / 2
    cut = min((i for i, ch in enumerate(text) if ch == " "), key=lambda i: abs(i - mid))
    return text[:cut] + "\n" + text[cut + 1:]


def action_tooltip(action: QAction, fallback: str = "") -> str:
    tip = action.toolTip() or fallback
    if not action.shortcut().isEmpty():
        tip += f"  ({action.shortcut().toString()})"
    return tip


class ActionButton(QToolButton):
    """Button with its own text whose enabled state follows a QAction."""

    _action: QAction | None = None

    def bind_action(self, action: QAction):
        self._action = action
        self.clicked.connect(action.trigger)
        action.changed.connect(self._sync_enabled)
        self._sync_enabled()

    def _sync_enabled(self):
        try:
            self.setEnabled(self._action.isEnabled())
        except RuntimeError:  # the action is deleted while the window closes
            pass


class Tile(ActionButton):
    """Ribbon tile."""

    def __init__(self, text: str, tooltip: str = "", accent: int | None = None, parent=None):
        super().__init__(parent)
        self.setObjectName("tile")
        self.setText(wrap_label(text))
        self.setToolTip(tooltip or text)
        self.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextOnly)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        if accent is not None:
            self.setProperty("accent", str(accent))

    def needed_width(self) -> int:
        fm = self.fontMetrics()
        text_w = max(fm.horizontalAdvance(line) for line in self.text().split("\n"))
        em = fm.horizontalAdvance("M")
        return max(em * TILE_MIN_EM, min(em * TILE_MAX_EM, text_w + em * 2))


class RibbonGroup(QFrame):
    """Titled group of tiles."""

    def __init__(self, title: str, parent=None):
        super().__init__(parent)
        self.setObjectName("ribbonGroup")
        outer = QVBoxLayout(self)
        outer.setContentsMargins(12, 6, 12, 8)
        outer.setSpacing(5)
        label = QLabel(title.upper())
        label.setObjectName("ribbonGroupTitle")
        outer.addWidget(label)
        self.row = QHBoxLayout()
        self.row.setSpacing(6)
        self.row.setAlignment(Qt.AlignmentFlag.AlignLeft)
        outer.addLayout(self.row)


class RibbonPage(QScrollArea):
    """Ribbon tab that scrolls horizontally when the tiles do not fit."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("ribbonPage")
        self.setWidgetResizable(True)
        self.setFrameShape(QFrame.Shape.NoFrame)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        inner = QWidget()
        inner.setObjectName("ribbonInner")
        self.layout_ = QHBoxLayout(inner)
        self.layout_.setContentsMargins(0, 0, 0, 0)
        self.layout_.setSpacing(0)
        self.layout_.addStretch(1)
        self.setWidget(inner)

    def add_group(self, title: str) -> RibbonGroup:
        group = RibbonGroup(title)
        self.layout_.insertWidget(self.layout_.count() - 1, group)
        return group


class Ribbon(QTabWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("ribbon")
        self.setDocumentMode(True)
        self.setUsesScrollButtons(True)
        self.tabBar().setDrawBase(False)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.tiles: list[Tile] = []

    def add_page(self, title: str) -> RibbonPage:
        page = RibbonPage()
        self.addTab(page, title)
        return page

    def add_tile(self, group: RibbonGroup, text: str, slot: Callable, tooltip: str = "",
                 accent: int | None = None) -> Tile:
        tile = Tile(text, tooltip, accent)
        tile.clicked.connect(lambda _=False: slot())
        group.row.addWidget(tile)
        self.tiles.append(tile)
        return tile

    def add_action_tile(self, group: RibbonGroup, action: QAction, text: str, accent: int | None = None) -> Tile:
        """Add a tile that triggers ``action`` and follows its enabled state."""
        tile = Tile(text, action_tooltip(action, text), accent)
        tile.bind_action(action)
        group.row.addWidget(tile)
        self.tiles.append(tile)
        return tile

    def refresh_sizes(self):
        """Recompute the common tile size and ribbon height, e.g. after a font change."""
        if not self.tiles:
            return
        fm = self.tiles[0].fontMetrics()
        width = max(t.needed_width() for t in self.tiles)
        height = int(fm.height() * TILE_LINES)
        for tile in self.tiles:
            tile.setFixedSize(width, height)
        title_h = int(fm.height() * 0.9)
        self.setFixedHeight(self.tabBar().sizeHint().height() + height + title_h + 34)
