"""Operation parameter declarations.

The parameter dialog is generated from these declarations: integers become
spin boxes, choices become combo boxes, matrices become text fields, and so on.
"""

from dataclasses import dataclass, field
from typing import Any, List, Optional, Sequence, Tuple

from ..i18n import tr


@dataclass
class Param:
    name: str            # keyword argument of the operation function
    label: str           # label shown in the dialog
    default: Any = None
    help: str = field(default="", kw_only=True)  # hint shown below the field (keyword-only)

    def validate(self, value):
        return value


@dataclass
class IntParam(Param):
    default: int = 0
    minimum: int = -1_000_000
    maximum: int = 1_000_000
    step: int = 1
    odd: bool = False    # require an odd value (e.g. window size)

    def validate(self, value):
        value = int(value)
        if not self.minimum <= value <= self.maximum:
            raise ValueError(tr("{label}: value must be between {lo} and {hi}.",
                                label=tr(self.label), lo=self.minimum, hi=self.maximum))
        if self.odd and value % 2 == 0:
            raise ValueError(tr("{label}: value must be odd.", label=tr(self.label)))
        return value


@dataclass
class FloatParam(Param):
    default: float = 0.0
    minimum: float = -1e9
    maximum: float = 1e9
    step: float = 0.1
    decimals: int = 3

    def validate(self, value):
        value = float(value)
        if not self.minimum <= value <= self.maximum:
            raise ValueError(tr("{label}: value must be between {lo} and {hi}.",
                                label=tr(self.label), lo=self.minimum, hi=self.maximum))
        return value


@dataclass
class BoolParam(Param):
    default: bool = False

    def validate(self, value):
        return bool(value)


@dataclass
class ChoiceParam(Param):
    """One value from ``choices``, a list of ``(value, label)`` pairs."""

    choices: Sequence[Tuple[Any, str]] = field(default_factory=list)

    def __post_init__(self):
        if self.default is None and self.choices:
            self.default = self.choices[0][0]

    def validate(self, value):
        if value not in [c[0] for c in self.choices]:
            raise ValueError(tr("{label}: invalid value {value}.", label=tr(self.label), value=repr(value)))
        return value

    def label_for(self, value) -> str:
        return next((tr(lbl) for v, lbl in self.choices if v == value), str(value))


@dataclass
class MatrixParam(Param):
    """A numeric matrix entered as text: one row per line, values separated by spaces."""

    default: str = "0 1 0\n1 1 1\n0 1 0"
    integer: bool = False
    allowed: Optional[Sequence[float]] = None  # e.g. (0, 1, -1) for a structuring element
    odd_size: bool = False

    def validate(self, value) -> List[List[float]]:
        text = str(value).strip()
        if not text:
            raise ValueError(tr("{label}: the matrix is empty.", label=tr(self.label)))
        rows = []
        for line in text.splitlines():
            items = line.replace(",", " " if self.integer else ".").replace(";", " ").split()
            if not items:
                continue
            try:
                row = [int(v) if self.integer else float(v) for v in items]
            except ValueError:
                msg = "{label}: only integers are allowed." if self.integer else "{label}: only numbers are allowed."
                raise ValueError(tr(msg, label=tr(self.label)))
            rows.append(row)
        if not rows:
            raise ValueError(tr("{label}: the matrix is empty.", label=tr(self.label)))
        width = len(rows[0])
        if any(len(r) != width for r in rows):
            raise ValueError(tr("{label}: all rows must have the same number of elements.", label=tr(self.label)))
        if self.odd_size and (width % 2 == 0 or len(rows) % 2 == 0):
            raise ValueError(tr("{label}: the matrix size must be odd (e.g. 3×3, 5×5).", label=tr(self.label)))
        if self.allowed is not None:
            bad = {v for r in rows for v in r if v not in self.allowed}
            if bad:
                raise ValueError(tr("{label}: invalid values {bad}; allowed: {allowed}.",
                                     label=tr(self.label), bad=sorted(bad), allowed=list(self.allowed)))
        return rows


def format_value(param: Param, value) -> str:
    """Short representation of a value for the history label."""
    if isinstance(param, ChoiceParam):
        return param.label_for(value).split(" – ")[0]  # "Fast – scipy, …" → "Fast"
    if isinstance(param, MatrixParam):
        rows = value if isinstance(value, list) else []
        return f"{len(rows)}×{len(rows[0]) if rows else 0}"
    if isinstance(value, float):
        return f"{value:g}"
    return str(value)
