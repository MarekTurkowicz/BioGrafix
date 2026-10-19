"""Operations exposed in the user interface.

Importing a module registers its operations; new modules are added to ``MODULES``.
"""

import importlib

MODULES = [
    "point_ops",
    "histogram",
    "filters",
    "thresholding",
    "morphology",
    "analysis",
    "geometry",
]

_loaded = False


def load_all() -> None:
    global _loaded
    if _loaded:
        return
    for name in MODULES:
        importlib.import_module(f"{__name__}.{name}")
    _loaded = True
