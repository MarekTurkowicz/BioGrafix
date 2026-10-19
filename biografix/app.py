"""Application start-up."""

import os
import sys
import traceback

from PySide6.QtCore import QLibraryInfo, QLocale, QSettings, QTranslator
from PySide6.QtWidgets import QApplication, QMessageBox

from . import i18n
from .constants import APP_NAME, ORG_NAME
from .i18n import tr


def _install_excepthook():
    """Show unhandled exceptions in a message box as well as on stderr."""

    def hook(exc_type, exc, tb):
        text = "".join(traceback.format_exception(exc_type, exc, tb))
        sys.stderr.write(text)
        if QApplication.instance() is not None:
            QMessageBox.critical(None, tr("Unexpected error"),
                                 f"{exc_type.__name__}: {exc}\n\n" + tr("Details in the console."))

    sys.excepthook = hook


def run(argv=None) -> int:
    argv = sys.argv if argv is None else argv
    from .ui.file_dialogs import is_wsl

    if is_wsl() and "QT_QPA_PLATFORM" not in os.environ:
        # Wayland is unreliable under WSL ("Failed to create wl_display"); use X11
        os.environ["QT_QPA_PLATFORM"] = "xcb"
    app = QApplication(argv)
    app.setApplicationName(APP_NAME)
    app.setOrganizationName(ORG_NAME)
    _install_excepthook()

    # saved language, or the system language on first run
    i18n.set_language(str(QSettings(ORG_NAME, APP_NAME).value("language", i18n.system_language())))
    if i18n.language() == "pl":
        # Polish labels on standard Qt buttons (Cancel, Save, …)
        translator = QTranslator(app)
        if translator.load(QLocale("pl_PL"), "qtbase", "_", QLibraryInfo.path(QLibraryInfo.LibraryPath.TranslationsPath)):
            app.installTranslator(translator)

    from .ui.main_window import MainWindow

    window = MainWindow()
    window.showMaximized()
    # images passed on the command line
    files = [a for a in argv[1:] if not a.startswith("-")]
    if files:
        window.load_images(files)
    return app.exec()
