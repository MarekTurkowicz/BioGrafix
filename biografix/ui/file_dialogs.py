"""File dialogs.

Native dialogs are used everywhere except WSL, where a Linux process cannot
open the Windows file explorer; there a Qt dialog is shown with shortcuts to the
Windows user folders.
"""

from __future__ import annotations

import glob
import os
import platform
import sys
from typing import List, Optional

from PySide6.QtCore import QUrl
from PySide6.QtWidgets import QFileDialog, QWidget


def is_wsl() -> bool:
    return sys.platform.startswith("linux") and "microsoft" in platform.release().lower()


def _windows_home() -> Optional[str]:
    """The Windows user directory as seen from WSL, e.g. ``/mnt/c/Users/name``."""
    here = os.path.abspath(__file__)
    if here.startswith("/mnt/c/Users/"):
        parts = here.split("/")
        return "/".join(parts[:5])
    users = [p for p in glob.glob("/mnt/c/Users/*") if os.path.basename(p) not in ("Public", "Default", "All Users", "Default User")
             and os.path.isdir(os.path.join(p, "Desktop"))]
    return users[0] if len(users) == 1 else None


def _sidebar() -> List[QUrl]:
    places = []
    home = _windows_home()
    if home:
        for sub in ("", "Desktop", "Documents", "Downloads", "Pictures", "OneDrive"):
            path = os.path.join(home, sub) if sub else home
            if os.path.isdir(path):
                places.append(path)
    for drive in sorted(glob.glob("/mnt/[a-z]")):
        places.append(drive)
    places.append(os.path.expanduser("~"))
    return [QUrl.fromLocalFile(p) for p in places]


def _dialog(parent, title, directory, filter_, mode, accept) -> QFileDialog:
    dlg = QFileDialog(parent, title, directory, filter_)
    dlg.setFileMode(mode)
    dlg.setAcceptMode(accept)
    dlg.setSidebarUrls(_sidebar())
    dlg.resize(1000, 640)
    return dlg


def open_files(parent: QWidget, title: str, directory: str, filter_: str) -> List[str]:
    if not is_wsl():
        return QFileDialog.getOpenFileNames(parent, title, directory, filter_)[0]
    dlg = _dialog(parent, title, directory, filter_, QFileDialog.FileMode.ExistingFiles, QFileDialog.AcceptMode.AcceptOpen)
    return dlg.selectedFiles() if dlg.exec() else []


def open_file(parent: QWidget, title: str, directory: str, filter_: str) -> str:
    if not is_wsl():
        return QFileDialog.getOpenFileName(parent, title, directory, filter_)[0]
    dlg = _dialog(parent, title, directory, filter_, QFileDialog.FileMode.ExistingFile, QFileDialog.AcceptMode.AcceptOpen)
    return dlg.selectedFiles()[0] if dlg.exec() and dlg.selectedFiles() else ""


def save_file(parent: QWidget, title: str, directory: str, filter_: str) -> str:
    if not is_wsl():
        return QFileDialog.getSaveFileName(parent, title, directory, filter_)[0]
    dlg = _dialog(parent, title, directory, filter_, QFileDialog.FileMode.AnyFile, QFileDialog.AcceptMode.AcceptSave)
    return dlg.selectedFiles()[0] if dlg.exec() and dlg.selectedFiles() else ""
