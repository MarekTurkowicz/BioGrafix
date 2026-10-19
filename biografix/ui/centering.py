"""Keep dialogs centred on the main window instead of where the window manager puts them."""

from __future__ import annotations

from PySide6.QtCore import QEvent, QObject, Qt, QTimer
from PySide6.QtWidgets import QApplication, QMainWindow, QWidget

_DONE = "_biografix_centered"


def center_on_parent(win: QWidget) -> None:
    parent = win.parentWidget()
    if parent is None:
        return
    host = parent.window().frameGeometry()
    rect = win.frameGeometry()
    rect.moveCenter(host.center())
    # keep the window on screen
    screen = (parent.window().screen() or win.screen()).availableGeometry()
    x = max(screen.left(), min(rect.left(), screen.right() - rect.width()))
    y = max(screen.top(), min(rect.top(), screen.bottom() - rect.height()))
    win.move(x, y)


class DialogCenterer(QObject):
    """Application event filter that centres each child window when it is first shown."""

    def eventFilter(self, obj, event):
        if event.type() == QEvent.Type.Show and isinstance(obj, QWidget) and obj.isWindow() \
                and not isinstance(obj, QMainWindow) \
                and obj.windowType() in (Qt.WindowType.Dialog, Qt.WindowType.Window, Qt.WindowType.Sheet) \
                and obj.parentWidget() is not None and not obj.property(_DONE):
            obj.setProperty(_DONE, True)
            center_on_parent(obj)
            # repeat after showing: some window managers (e.g. WSLg) move the window when it maps
            QTimer.singleShot(0, lambda w=obj: _safe_center(w))
        return False


def _safe_center(win):
    try:
        if win.isVisible():
            center_on_parent(win)
    except RuntimeError:  # the window has already been deleted
        pass


def install(app: QApplication) -> None:
    """Install the filter once per application; later calls do nothing."""
    if getattr(app, "_biografix_centerer", None) is None:
        app._biografix_centerer = DialogCenterer(app)
        app.installEventFilter(app._biografix_centerer)
