"""Operation registry.

An algorithm becomes available in the application by decorating a function
with :func:`operation`. Its ribbon tile, menu entry, parameter dialog,
background execution, region support and undo step are derived from that
declaration.

An image operation receives a (H, W, 3) uint8 RGB array and returns one of:

- ``np.ndarray``        – the new image, recorded as one undo step;
- :class:`Report`       – a text result shown in the Results panel;
- matplotlib ``Figure`` – a chart opened in a separate window;
- ``None``              – nothing changes.

A list or tuple may combine several of these.
"""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

from ..i18n import tr
from .params import Param

# tab and menu order; unlisted categories are appended
CATEGORY_ORDER = [
    "Point operations",
    "Histogram",
    "Filters",
    "Thresholding",
    "Morphology",
    "Segmentation",
    "Analysis",
    "Geometry",
]

TARGET_IMAGE = "image"   # operates on the selected image
TARGET_SHAPE = "shape"   # operates on the selected vector shape


@dataclass
class Report:
    """Text result of an operation, shown in the Results panel and the status bar."""

    text: str
    title: str = ""


@dataclass
class Operation:
    id: str
    category: str
    label: str
    func: Callable
    params: list[Param] = field(default_factory=list)
    description: str = ""
    section: str = ""                    # group title on the ribbon and section in the menu
    source: str = ""                     # optional note, not shown in the UI
    target: str = TARGET_IMAGE
    shape_kinds: Sequence[str] = ()      # TARGET_SHAPE only: accepted shape kinds
    supports_roi: bool = True            # can be limited to the selected region
    slow: bool = False                   # show a warning about large images
    shortcut: str = ""
    # defaults computed at run time, e.g. the shape centre: f(obj, context) -> dict
    dynamic_defaults: Callable[[Any, dict], dict[str, Any]] | None = None

    def history_label(self, values: dict[str, Any]) -> str:
        from .params import format_value

        title = f"{tr(self.category)}: {tr(self.label)}"
        if not self.params:
            return title
        shown = ", ".join(
            f"{tr(p.label).split(' (')[0]}={format_value(p, values[p.name])}" for p in self.params if p.name in values
        )
        return f"{title} ({shown})"


REGISTRY: OrderedDict[str, Operation] = OrderedDict()


def operation(
    id: str,
    category: str,
    label: str,
    params: Sequence[Param] = (),
    **kwargs,
):
    """Register the decorated function as an operation."""

    def decorator(func: Callable) -> Callable:
        if id in REGISTRY:
            raise ValueError(f"Operation id {id!r} is already registered.")
        REGISTRY[id] = Operation(id=id, category=category, label=label, func=func, params=list(params), **kwargs)
        return func

    return decorator


def group_by_section(ops: list[Operation]) -> OrderedDict[str, list[Operation]]:
    """Group operations by section, keeping the order of first appearance."""
    groups: OrderedDict[str, list[Operation]] = OrderedDict()
    for op in ops:
        groups.setdefault(op.section, []).append(op)
    return groups


def operations_by_category() -> OrderedDict[str, list[Operation]]:
    groups: OrderedDict[str, list[Operation]] = OrderedDict()
    for cat in CATEGORY_ORDER:
        groups[cat] = []
    for op in REGISTRY.values():
        groups.setdefault(op.category, []).append(op)
    return OrderedDict((k, v) for k, v in groups.items() if v)


def get(op_id: str) -> Operation:
    return REGISTRY[op_id]
