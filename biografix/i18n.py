"""User interface language.

Source strings are written in English and wrapped in ``tr()``. When Polish is
active, ``tr()`` looks the string up in ``i18n_pl.PL`` and falls back to the
English text if no translation exists. The language is read once at start-up;
changing it in View → Language takes effect after a restart.
"""

from __future__ import annotations

from .i18n_pl import PL

LANGUAGES = {"en": "English", "pl": "Polski"}
DEFAULT_LANGUAGE = "en"

_language = DEFAULT_LANGUAGE


def system_language() -> str:
    """Polish on a Polish system, English otherwise."""
    from PySide6.QtCore import QLocale

    return "pl" if QLocale.system().language() == QLocale.Language.Polish else "en"


def set_language(code: str) -> None:
    global _language
    _language = code if code in LANGUAGES else DEFAULT_LANGUAGE


def language() -> str:
    return _language


def tr(text: str, **values) -> str:
    """Translate ``text`` and fill its ``{name}`` placeholders from ``values``."""
    if _language == "pl" and text:
        text = PL.get(text, text)
    return text.format(**values) if values else text
