"""Dialogs: generated parameter dialog, chart window and Markdown viewer."""

from __future__ import annotations

from typing import Any

from PySide6.QtCore import Qt
from PySide6.QtGui import QFontDatabase
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFormLayout,
    QLabel,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QSpinBox,
    QTextBrowser,
    QVBoxLayout,
    QWidget,
)

from ..core.params import BoolParam, ChoiceParam, FloatParam, IntParam, MatrixParam, Param
from ..core.registry import Operation
from ..i18n import tr
from . import theme


class ParamsDialog(QDialog):
    """Parameter dialog generated from an operation's parameter list.

    After acceptance ``values`` holds the validated values and ``raw`` the field contents.
    """

    def __init__(self, op: Operation, initial: dict[str, Any], parent=None):
        super().__init__(parent)
        self.op = op
        self.values: dict[str, Any] = {}
        self.raw: dict[str, Any] = {}
        self._widgets: dict[str, QWidget] = {}
        self.setWindowTitle(f"{tr(op.category)} – {tr(op.label)}")
        self.setMinimumWidth(420)

        muted = theme.colors()["muted"]
        layout = QVBoxLayout(self)
        if op.description:
            desc = QLabel(tr(op.description))
            desc.setWordWrap(True)
            layout.addWidget(desc)
        if op.slow:
            warn = QLabel("⚠ " + tr("On a large image this algorithm may take a while (the window stays responsive). Use the Region tool to limit it to a part of the image."))
            warn.setWordWrap(True)
            warn.setStyleSheet(f"color:{theme.colors()['warn']}")
            layout.addWidget(warn)

        form = QFormLayout()
        for p in op.params:
            w = self._make_widget(p)
            self._set_widget_value(p, w, initial.get(p.name, p.default))
            self._widgets[p.name] = w
            form.addRow(tr(p.label) + ":", w)
            if p.help:
                w.setToolTip(tr(p.help))
                hint = QLabel(f"<span style='color:{muted}'>{tr(p.help)}</span>")
                hint.setWordWrap(True)
                form.addRow("", hint)
        layout.addLayout(form)

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        reset = QPushButton(tr("Defaults"))
        buttons.addButton(reset, QDialogButtonBox.ButtonRole.ResetRole)
        reset.clicked.connect(self._reset_defaults)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    # ---------------------------------------------------------------- widgets --
    def _make_widget(self, p: Param) -> QWidget:
        if isinstance(p, IntParam):
            spin = QSpinBox()
            spin.setRange(p.minimum, p.maximum)
            spin.setSingleStep(2 if p.odd else p.step)
            return spin
        if isinstance(p, FloatParam):
            dspin = QDoubleSpinBox()
            dspin.setDecimals(p.decimals)
            dspin.setRange(p.minimum, p.maximum)
            dspin.setSingleStep(p.step)
            return dspin
        if isinstance(p, BoolParam):
            return QCheckBox()
        if isinstance(p, ChoiceParam):
            combo = QComboBox()
            for value, label in p.choices:
                combo.addItem(tr(label), value)
            return combo
        if isinstance(p, MatrixParam):
            edit = QPlainTextEdit()
            edit.setFont(QFontDatabase.systemFont(QFontDatabase.SystemFont.FixedFont))
            edit.setFixedHeight(110)
            return edit
        raise TypeError(f"Unsupported parameter type: {type(p).__name__}")

    def _set_widget_value(self, p, w, value):
        if isinstance(w, (QSpinBox, QDoubleSpinBox)):
            w.setValue(value)
        elif isinstance(w, QCheckBox):
            w.setChecked(bool(value))
        elif isinstance(w, QComboBox):
            i = w.findData(value)
            w.setCurrentIndex(max(0, i))
        elif isinstance(w, QPlainTextEdit):
            w.setPlainText(str(value))

    @staticmethod
    def _widget_value(w):
        if isinstance(w, (QSpinBox, QDoubleSpinBox)):
            return w.value()
        if isinstance(w, QCheckBox):
            return w.isChecked()
        if isinstance(w, QComboBox):
            return w.currentData()
        if isinstance(w, QPlainTextEdit):
            return w.toPlainText()
        return None

    def _reset_defaults(self):
        for p in self.op.params:
            self._set_widget_value(p, self._widgets[p.name], p.default)

    def accept(self):
        raw = {p.name: self._widget_value(self._widgets[p.name]) for p in self.op.params}
        try:
            values = {p.name: p.validate(raw[p.name]) for p in self.op.params}
        except (ValueError, TypeError) as e:
            QMessageBox.warning(self, tr("Invalid parameter"), str(e))
            return
        self.raw, self.values = raw, values
        super().accept()


class FigureWindow(QWidget):
    """Window that shows a matplotlib figure."""

    def __init__(self, figure, title: str, parent=None):
        super().__init__(parent, Qt.WindowType.Window)
        from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT

        self.setWindowTitle(title)
        self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)
        canvas = FigureCanvasQTAgg(figure)
        layout = QVBoxLayout(self)
        layout.addWidget(NavigationToolbar2QT(canvas, self))
        layout.addWidget(canvas)
        w, h = figure.get_size_inches() * figure.dpi
        self.resize(int(w) + 20, int(h) + 60)


class MarkdownDialog(QDialog):
    def __init__(self, title: str, markdown: str, parent=None):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.resize(900, 700)
        layout = QVBoxLayout(self)
        view = QTextBrowser()
        view.setOpenExternalLinks(True)
        view.setMarkdown(markdown)
        layout.addWidget(view)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
