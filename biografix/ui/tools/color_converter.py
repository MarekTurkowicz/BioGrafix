"""RGB ↔ CMYK colour converter."""

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QButtonGroup,
    QDoubleSpinBox,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QRadioButton,
    QSlider,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from ...algorithms.color_models import cmyk_to_rgb, rgb_to_cmyk
from ...i18n import tr


class ColorConverterWindow(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent, Qt.WindowType.Window)
        self.setWindowTitle(tr("RGB / CMYK colour conversion"))
        self._updating = False

        layout = QVBoxLayout(self)

        # input mode
        modes = QHBoxLayout()
        modes.addWidget(QLabel(tr("Input mode:")))
        self.rb_rgb = QRadioButton("RGB")
        self.rb_cmyk = QRadioButton("CMYK")
        self.rb_rgb.setChecked(True)
        group = QButtonGroup(self)
        group.addButton(self.rb_rgb)
        group.addButton(self.rb_cmyk)
        modes.addWidget(self.rb_rgb)
        modes.addWidget(self.rb_cmyk)
        modes.addStretch(1)
        layout.addLayout(modes)
        self.rb_rgb.toggled.connect(self._on_mode)

        # RGB inputs
        self.rgb_box = QGroupBox("RGB (0–255)")
        g = QGridLayout(self.rgb_box)
        self.rgb = {}
        for row, (ch, val) in enumerate([("R", 255), ("G", 0), ("B", 0)]):
            s = QSlider(Qt.Orientation.Horizontal)
            s.setRange(0, 255)
            sp = QSpinBox()
            sp.setRange(0, 255)
            s.setValue(val)
            sp.setValue(val)
            s.valueChanged.connect(sp.setValue)
            sp.valueChanged.connect(s.setValue)
            sp.valueChanged.connect(self._from_rgb)
            g.addWidget(QLabel(ch + ":"), row, 0)
            g.addWidget(s, row, 1)
            g.addWidget(sp, row, 2)
            self.rgb[ch] = sp
        layout.addWidget(self.rgb_box)

        # CMYK inputs in percent
        self.cmyk_box = QGroupBox("CMYK (0–100 %)")
        g = QGridLayout(self.cmyk_box)
        self.cmyk = {}
        for row, ch in enumerate("CMYK"):
            s = QSlider(Qt.Orientation.Horizontal)
            s.setRange(0, 1000)  # 0.1 %
            sp = QDoubleSpinBox()
            sp.setRange(0.0, 100.0)
            sp.setDecimals(1)
            s.valueChanged.connect(lambda v, sp=sp: sp.setValue(v / 10.0))
            sp.valueChanged.connect(lambda v, s=s: s.setValue(int(round(v * 10))))
            sp.valueChanged.connect(self._from_cmyk)
            g.addWidget(QLabel(f"{ch} (%):"), row, 0)
            g.addWidget(s, row, 1)
            g.addWidget(sp, row, 2)
            self.cmyk[ch] = sp
        layout.addWidget(self.cmyk_box)

        # preview and result
        prev = QHBoxLayout()
        prev.addWidget(QLabel(tr("Preview:")))
        self.preview = QFrame()
        self.preview.setFixedSize(68, 44)
        self.preview.setFrameShape(QFrame.Shape.Box)
        prev.addWidget(self.preview)
        res = QVBoxLayout()
        self.res_rgb = QLabel("RGB: -")
        self.res_cmyk = QLabel("CMYK: -")
        res.addWidget(self.res_rgb)
        res.addWidget(self.res_cmyk)
        prev.addLayout(res)
        prev.addStretch(1)
        layout.addLayout(prev)

        self._from_rgb()
        self._on_mode()

    def _on_mode(self):
        rgb_mode = self.rb_rgb.isChecked()
        self.rgb_box.setVisible(rgb_mode)
        self.cmyk_box.setVisible(not rgb_mode)
        self.adjustSize()

    def _from_rgb(self, *_):
        if self._updating:
            return
        self._updating = True
        r, g, b = (self.rgb[c].value() for c in "RGB")
        c, m, y, k = rgb_to_cmyk(r, g, b)
        for ch, v in zip("CMYK", (c, m, y, k)):
            self.cmyk[ch].setValue(v * 100.0)
        self._updating = False
        self._show(r, g, b, c, m, y, k)

    def _from_cmyk(self, *_):
        if self._updating:
            return
        self._updating = True
        c, m, y, k = (self.cmyk[ch].value() / 100.0 for ch in "CMYK")
        r, g, b = cmyk_to_rgb(c, m, y, k)
        for ch, v in zip("RGB", (r, g, b)):
            self.rgb[ch].setValue(v)
        self._updating = False
        self._show(r, g, b, c, m, y, k)

    def _show(self, r, g, b, c, m, y, k):
        self.preview.setStyleSheet(f"background:#{r:02x}{g:02x}{b:02x}")
        self.res_rgb.setText(f"RGB: {r}, {g}, {b}")
        self.res_cmyk.setText(f"CMYK: {c * 100:.1f}%, {m * 100:.1f}%, {y * 100:.1f}%, {k * 100:.1f}%")
