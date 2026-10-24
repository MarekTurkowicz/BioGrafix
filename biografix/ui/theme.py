"""Dark and light themes and the UI font size.

Interface colours feed the Qt palette and the generated style sheet; canvas
items read their colours from :func:`canvas` while painting. The accents are
the Python blue #3776AB and yellow #FFD43B.
"""

from __future__ import annotations

import sys
from typing import Any, cast

from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QFont, QPalette
from PySide6.QtWidgets import QApplication

DEFAULT_THEME = "dark"
DEFAULT_FONT_PT = 11
FONT_PT_RANGE = (8, 18)

PY_BLUE = "#3776ab"
PY_YELLOW = "#ffd43b"

THEMES: dict[str, dict[str, Any]] = {
    "dark": {
        "label": "Dark",
        "bg": "#1e1f22",         # window background
        "panel": "#25272b",      # panels and ribbon
        "base": "#2b2d31",       # text fields and lists
        "tile": "#2f3237",       # tiles
        "tile_hover": "#383c42",
        "border": "#3d4148",
        "text": "#dfe1e5",
        "muted": "#8b9099",
        "accent": "#4a8fd4",     # lighter Python blue, readable on dark backgrounds
        "accent2": PY_YELLOW,
        "warn": "#e8a33d",
        "canvas": {
            "bg": "#17181b",
            "fg": "#e6e6e6",          # polygons and point labels
            "poly_sel": "#ff5a5a",
            "bezier": "#5b9bff",
            "bezier_ctrl": "#6b7078",
            "bezier_point": "#ff5a5a",
            "pivot": "#7fb0ff",
            "hint": "#5c6068",
        },
    },
    "light": {
        "label": "Light",
        "bg": "#eef0f3",
        "panel": "#f7f8fa",
        "base": "#ffffff",
        "tile": "#ffffff",
        "tile_hover": "#eaf2fb",
        "border": "#c9ced6",
        "text": "#1f2328",
        "muted": "#656d76",
        "accent": PY_BLUE,
        "accent2": "#e0a800",
        "warn": "#a05a00",
        "canvas": {
            "bg": "#ffffff",
            "fg": "#000000",
            "poly_sel": "#ff0000",
            "bezier": "#0040ff",
            "bezier_ctrl": "#cccccc",
            "bezier_point": "#ff0000",
            "pivot": "#0000aa",
            "hint": "#a0a6ae",
        },
    },
}

# per-category accent colours, assigned to tabs in order
CATEGORY_ACCENTS = ["#4a8fd4", "#ffd43b", "#4fc08d", "#e06c75", "#c678dd", "#56b6c2", "#e5a04b", "#98c379", "#d19a66"]

_current: dict[str, Any] = {"name": DEFAULT_THEME, "pt": DEFAULT_FONT_PT}


def current_name() -> str:
    return _current["name"]


def current_font_pt() -> int:
    return _current["pt"]


def colors() -> dict:
    return THEMES[_current["name"]]


def canvas() -> dict:
    """Canvas colours of the current theme."""
    return THEMES[_current["name"]]["canvas"]


def _palette(t: dict) -> QPalette:
    p = QPalette()
    R = QPalette.ColorRole
    G = QPalette.ColorGroup
    p.setColor(R.Window, QColor(t["bg"]))
    p.setColor(R.WindowText, QColor(t["text"]))
    p.setColor(R.Base, QColor(t["base"]))
    p.setColor(R.AlternateBase, QColor(t["panel"]))
    p.setColor(R.ToolTipBase, QColor(t["panel"]))
    p.setColor(R.ToolTipText, QColor(t["text"]))
    p.setColor(R.PlaceholderText, QColor(t["muted"]))
    p.setColor(R.Text, QColor(t["text"]))
    p.setColor(R.Button, QColor(t["tile"]))
    p.setColor(R.ButtonText, QColor(t["text"]))
    p.setColor(R.BrightText, QColor(t["accent2"]))
    p.setColor(R.Highlight, QColor(t["accent"]))
    p.setColor(R.HighlightedText, QColor("#ffffff"))
    p.setColor(R.Link, QColor(t["accent"]))
    p.setColor(R.Mid, QColor(t["border"]))
    p.setColor(R.Dark, QColor(t["border"]))
    for role in (R.WindowText, R.Text, R.ButtonText):
        p.setColor(G.Disabled, role, QColor(t["muted"]))
    return p


def stylesheet(t: dict, pt: int) -> str:
    small = max(7, pt - 2)
    return f"""
QWidget {{ font-size: {pt}pt; }}
QMainWindow, QDialog {{ background: {t['bg']}; }}
QMainWindow::separator {{ background: {t['border']}; width: 1px; height: 1px; }}

QMenuBar {{ background: {t['bg']}; padding: 2px; }}
QMenuBar::item {{ padding: 4px 10px; border-radius: 4px; }}
QMenuBar::item:selected {{ background: {t['tile_hover']}; }}
QMenu {{ background: {t['panel']}; border: 1px solid {t['border']}; padding: 4px; }}
QMenu::item {{ padding: 5px 28px 5px 14px; border-radius: 4px; }}
QMenu::item:selected {{ background: {t['accent']}; color: #ffffff; }}
QMenu::separator {{ height: 1px; background: {t['border']}; margin: 4px 8px; }}

QDockWidget::title {{ background: {t['panel']}; padding: 6px 10px; border-bottom: 1px solid {t['border']}; }}

QToolTip {{ background: {t['panel']}; color: {t['text']}; border: 1px solid {t['accent']}; padding: 6px; }}
QStatusBar {{ background: {t['panel']}; color: {t['muted']}; border-top: 1px solid {t['border']}; }}
QStatusBar QLabel {{ color: {t['muted']}; padding: 0 8px; }}

QPushButton {{ background: {t['tile']}; border: 1px solid {t['border']}; border-radius: 6px; padding: 5px 14px; }}
QPushButton:hover {{ border-color: {t['accent']}; background: {t['tile_hover']}; }}
QPushButton:pressed {{ background: {t['accent']}; color: #ffffff; }}
QPushButton:default {{ border-color: {t['accent']}; }}

QLineEdit, QPlainTextEdit, QTextEdit, QListView, QListWidget, QTreeView, QUndoView {{
    background: {t['base']}; border: 1px solid {t['border']}; border-radius: 6px;
    selection-background-color: {t['accent']}; selection-color: #ffffff; }}
QLineEdit {{ padding: 3px 6px; }}
QLineEdit:focus, QPlainTextEdit:focus {{ border-color: {t['accent']}; }}
QListView::item, QListWidget::item {{ padding: 4px 6px; }}
QGroupBox {{ border: 1px solid {t['border']}; border-radius: 8px; margin-top: 14px; padding-top: 6px; }}
QGroupBox::title {{ subcontrol-origin: margin; left: 10px; padding: 0 4px; color: {t['muted']}; }}

QProgressBar {{ border: 1px solid {t['border']}; border-radius: 5px; background: {t['base']}; max-height: 12px; }}
QProgressBar::chunk {{ background: {t['accent2']}; border-radius: 4px; }}

QScrollBar:vertical {{ background: transparent; width: 12px; margin: 0; }}
QScrollBar:horizontal {{ background: transparent; height: 12px; margin: 0; }}
QScrollBar::handle {{ background: {t['border']}; border-radius: 5px; min-height: 30px; min-width: 30px; margin: 2px; }}
QScrollBar::handle:hover {{ background: {t['muted']}; }}
QScrollBar::add-line, QScrollBar::sub-line {{ width: 0; height: 0; }}
QScrollBar::add-page, QScrollBar::sub-page {{ background: transparent; }}

/* ---- mouse tool bar ---- */
QToolBar {{ background: {t['bg']}; border: none; spacing: 3px; padding: 4px; }}
QToolBar#tools QToolButton {{ padding: 6px 10px; border-radius: 6px; border: 1px solid transparent; }}
QToolBar#tools QToolButton:hover {{ background: {t['tile_hover']}; }}
QToolBar#tools QToolButton:checked {{ background: {t['accent']}; color: #ffffff; }}

/* ---- quick action bar ---- */
QToolBar#quickBar {{ background: {t['bg']}; padding: 5px 4px 7px 4px; spacing: 2px; border-bottom: 1px solid {t['border']}; }}
QToolBar#quickBar::separator {{ background: {t['border']}; width: 1px; margin: 4px 6px; }}
QToolButton#quick {{ padding: 4px 9px; border-radius: 6px; border: 1px solid transparent; }}
QToolButton#quick:hover {{ background: {t['tile_hover']}; border-color: {t['border']}; }}
QToolButton#quick:pressed {{ background: {t['accent']}; color: #ffffff; }}
QToolButton#quick:disabled {{ color: {t['muted']}; }}
QToolButton#quick:checked {{ background: {t['accent']}; color: #ffffff; }}

/* ---- ribbon ---- */
QToolBar#ribbonBar {{ background: {t['panel']}; padding: 0; border-bottom: 1px solid {t['border']}; }}
QWidget#ribbonHolder {{ background: {t['panel']}; }}
QTabWidget#ribbon::pane {{ border: none; background: {t['panel']}; }}
QTabWidget#ribbon QTabBar {{ qproperty-drawBase: 0; }}
QTabWidget#ribbon QTabBar::tab {{ background: transparent; color: {t['muted']}; padding: 8px 22px; margin-right: 6px;
    border: none; border-bottom: 3px solid transparent; font-weight: 600; }}
QTabWidget#ribbon QTabBar::tab:hover {{ color: {t['text']}; }}
QTabWidget#ribbon QTabBar::tab:selected {{ color: {t['text']}; border-bottom: 3px solid {t['accent2']}; }}
QScrollArea#ribbonPage, QWidget#ribbonInner {{ background: {t['panel']}; border: none; }}
QFrame#ribbonGroup {{ border-right: 1px solid {t['border']}; }}
QLabel#hint {{ color: {t['muted']}; }}
QLabel#paletteTitle {{ color: {t['muted']}; font-size: {small}pt; font-weight: 600; letter-spacing: 1px; }}
QSlider::groove:horizontal {{ height: 4px; background: {t['border']}; border-radius: 2px; }}
QSlider::sub-page:horizontal {{ background: {t['accent']}; border-radius: 2px; }}
QSlider::handle:horizontal {{ background: {t['accent2']}; width: 14px; height: 14px; margin: -6px 0; border-radius: 7px; }}
QSlider::handle:horizontal:hover {{ background: {t['text']}; }}
QLabel#ribbonGroupTitle {{ color: {t['muted']}; font-size: {small}pt; font-weight: 600; letter-spacing: 1px; }}

QToolButton#tile {{ background: {t['tile']}; color: {t['text']}; border: 1px solid {t['border']};
    border-bottom: 3px solid {t['border']}; border-radius: 8px; padding: 4px 8px; }}
QToolButton#tile:hover {{ background: {t['tile_hover']}; border-color: {t['accent']}; }}
QToolButton#tile:pressed {{ background: {t['accent']}; color: #ffffff; }}
QToolButton#tile:disabled {{ color: {t['muted']}; background: {t['panel']}; }}
""" + "".join(
        # coloured bottom border from the tile's "accent" property
        f'QToolButton#tile[accent="{i}"] {{ border-bottom-color: {c}; }}\n' for i, c in enumerate(CATEGORY_ACCENTS)
    )


def apply(name: str | None = None, pt: int | None = None) -> None:
    """Apply a theme and/or font size to the whole application."""
    app = cast(QApplication, QApplication.instance())
    if name in THEMES:
        _current["name"] = name
    if pt is not None:
        _current["pt"] = max(FONT_PT_RANGE[0], min(FONT_PT_RANGE[1], int(pt)))
    t = colors()
    try:  # Qt >= 6.8: matching window title bar on Windows
        app.styleHints().setColorScheme(Qt.ColorScheme.Dark if _current["name"] == "dark" else Qt.ColorScheme.Light)
    except AttributeError:
        pass
    app.setStyle("Fusion")
    app.setPalette(_palette(t))
    font = QFont(app.font())
    font.setPointSize(_current["pt"])
    app.setFont(font)
    app.setStyleSheet(stylesheet(t, _current["pt"]))


def style_title_bar(window) -> None:
    """Match the Windows title bar and frame to the theme (DWM); no-op elsewhere."""
    if sys.platform != "win32":
        return
    try:
        import ctypes

        hwnd = int(window.winId())
        value = ctypes.c_int(1 if _current["name"] == "dark" else 0)
        for attr in (20, 19):  # DWMWA_USE_IMMERSIVE_DARK_MODE (current / older Windows 10 builds)
            if ctypes.windll.dwmapi.DwmSetWindowAttribute(hwnd, attr, ctypes.byref(value), ctypes.sizeof(value)) == 0:
                break
    except Exception:
        pass
