"""Scene model and JSON serialisation, independent of the GUI."""

from .objects import (
    BezierShape,
    CircleShape,
    LineShape,
    PolygonShape,
    RasterImage,
    RectShape,
    SceneObject,
    object_from_dict,
)

__all__ = [
    "SceneObject",
    "LineShape",
    "RectShape",
    "CircleShape",
    "PolygonShape",
    "BezierShape",
    "RasterImage",
    "object_from_dict",
]
