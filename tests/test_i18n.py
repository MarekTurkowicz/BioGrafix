"""Interface translation: every visible string has a Polish version with matching placeholders."""

import ast
import os
import re
import string
import unittest

from biografix import i18n
from biografix.core import registry
from biografix.core.params import ChoiceParam
from biografix.i18n_pl import PL
from biografix.operations import load_all
from biografix.scene.objects import SHAPE_TYPES, RasterImage, SceneObject

PACKAGE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "biografix")

# Raised by algorithms and file readers; translated when shown to the user.
ERROR_MESSAGES = [
    "The image is empty.",
    "The kernel size must be odd and >= 1.",
    "The structuring element is empty.",
    "The structuring element may contain only integers (0, 1 and optionally -1).",
    "All rows of the structuring element must have the same length.",
    "Could not read the structuring element.",
    "Division by zero is not allowed.",
    "Empty PPM file.",
    "Not a PPM P3 file (magic != P3).",
    "Incomplete P3 header.",
    "Invalid P3 header.",
    "Not a PPM P6 file (magic != P6).",
    "Invalid P6 header.",
    "Missing data separator after the P6 header.",
    "Not enough binary data in P6.",
    "Unknown PPM format (magic is not P3/P6).",
    # selected through a variable, so not visible as tr("…") literals
    "{label}: only integers are allowed.",
    "{label}: only numbers are allowed.",
]

# Method names and symbols that read the same in both languages.
NAMES = {"Otsu", "Niblack", "Sauvola", "Phansalkar", "Percent Black", "Mean Iterative Selection",
         "Histogram", "Prewitt", "Sobel", "Laplace", "Zoom: {percent}%",
         "H min [°]", "H max [°]", "S min", "V min"}

# Helpers that pass their string arguments through tr() themselves.
TRANSLATING_CALLS = {"tr", "_action", "_add_dock", "_ask_busy"}

SHORTCUT = re.compile(r"F\d+|((Ctrl|Shift|Alt)\+)+\S+")


def _literals_passed_to_tr():
    found = set()
    for root, _dirs, files in os.walk(PACKAGE):
        for name in files:
            if not name.endswith(".py") or name == "i18n_pl.py":
                continue
            with open(os.path.join(root, name), encoding="utf-8") as f:
                tree = ast.parse(f.read())
            for node in ast.walk(tree):
                if not isinstance(node, ast.Call):
                    continue
                func = getattr(node.func, "id", None) or getattr(node.func, "attr", None)
                if func not in TRANSLATING_CALLS:
                    continue
                args = node.args[:1] if func == "tr" else node.args
                found |= {a.value for a in args if isinstance(a, ast.Constant) and isinstance(a.value, str)
                          and not SHORTCUT.fullmatch(a.value)}
    return found


def visible_strings():
    """Every English string the user can see in the interface."""
    from biografix.io.images import OPEN_FILTER
    from biografix.ui import theme
    from biografix.ui.canvas.view import TOOL_HINTS, TOOLS
    from biografix.ui.main_window import HELP_NOTES, QUICK_BAR, SCENE_FILTER, SHORTCUTS, TASKS, TOOL_WINDOWS
    from biografix.ui.panels import OBJECT_BUTTONS
    from biografix.ui.tools.color_spaces_3d import HSVConeWindow, RGBCubeWindow

    load_all()
    keys = _literals_passed_to_tr() | set(ERROR_MESSAGES) | {OPEN_FILTER, SCENE_FILTER}
    for op in registry.REGISTRY.values():
        keys |= {op.category, op.label, op.section, op.description}
        for p in op.params:
            keys |= {p.label, p.help}
            if isinstance(p, ChoiceParam):
                keys |= {label for _value, label in p.choices}
    keys |= {label for _id, label, _key in TOOLS} | set(TOOL_HINTS.values())
    keys |= {text for _key, text in TOOL_WINDOWS} | {title for title, _ids in TASKS}
    keys |= {text for _key, text in SHORTCUTS} | set(HELP_NOTES)
    keys |= {text for group in QUICK_BAR for _key, text in group}
    keys |= {text for text, _tip, _key in OBJECT_BUTTONS} | {tip for _text, tip, _key in OBJECT_BUTTONS}
    for cls in [SceneObject, RasterImage, *SHAPE_TYPES.values()]:
        keys |= {cls.kind_label, cls.params_hint}
    keys |= {t["label"] for t in theme.THEMES.values()}
    keys |= {RGBCubeWindow.slice_prompt, HSVConeWindow.slice_prompt}
    # numbers and symbols such as "100%" or "▲" need no translation
    return {k for k in keys if re.search(r"[a-z]{3}", k, re.IGNORECASE)} - NAMES


def _fields(text):
    return {f for _, f, _, _ in string.Formatter().parse(text) if f}


class TestTranslations(unittest.TestCase):
    def test_every_string_has_polish_version(self):
        missing = sorted(visible_strings() - PL.keys())
        self.assertEqual(missing, [], "Missing translations:\n" + "\n".join(missing))

    def test_no_unused_translations(self):
        unused = sorted(PL.keys() - visible_strings())
        self.assertEqual(unused, [], "Unused translations:\n" + "\n".join(unused))

    def test_placeholders_match(self):
        for en, pl in PL.items():
            with self.subTest(en=en):
                self.assertEqual(_fields(en), _fields(pl))

    def test_tr_switches_language(self):
        try:
            self.assertEqual(i18n.tr("Loaded images ({n})", n=2), "Loaded images (2)")
            i18n.set_language("pl")
            self.assertEqual(i18n.tr("Undo"), "Cofnij")
            self.assertEqual(i18n.tr("Loaded images ({n})", n=2), "Wczytano obrazy (2)")
            self.assertEqual(i18n.tr("Text without translation"), "Text without translation")
        finally:
            i18n.set_language("en")


if __name__ == "__main__":
    unittest.main()
