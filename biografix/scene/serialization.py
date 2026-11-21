"""Scene serialisation to and from JSON.

Images are embedded as base64 PNG so that edits survive a save. Files that
contain only a ``polygons`` list are also accepted.

The file layout is validated with pydantic before any object is built, so a
malformed file raises a ``ValueError`` that names the offending field.
"""

import json
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from .objects import PolygonShape, SceneObject, object_from_dict

FORMAT_VERSION = 2


class ObjectEntry(BaseModel):
    """One serialised scene object; fields beyond ``type`` depend on the object kind."""

    model_config = ConfigDict(extra="allow")

    type: str


class PolygonEntry(BaseModel):
    """A polygon from a polygons-only file."""

    points: list[tuple[float, float]] = Field(default_factory=list)
    closed: bool = True


class SceneFile(BaseModel):
    """Top-level layout of a scene file."""

    app: str = "BioGrafix"
    version: int = FORMAT_VERSION
    objects: list[ObjectEntry] = Field(default_factory=list)
    polygons: list[PolygonEntry] = Field(default_factory=list)


def scene_to_dict(objects: list[SceneObject]) -> dict:
    return {"app": "BioGrafix", "version": FORMAT_VERSION, "objects": [o.to_dict() for o in objects]}


def scene_from_dict(data: Any) -> list[SceneObject]:
    """Build scene objects from parsed JSON; raises ``ValueError`` if the layout is invalid."""
    try:
        scene = SceneFile.model_validate(data)
    except ValidationError as e:
        raise ValueError(f"Invalid scene file:\n{e}") from None
    objects = [object_from_dict(entry.model_dump()) for entry in scene.objects]
    for polygon in scene.polygons:
        if polygon.points:
            objects.append(PolygonShape(points=list(polygon.points), closed=polygon.closed))
    return objects


def save_scene(path: str, objects: list[SceneObject]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(scene_to_dict(objects), f, ensure_ascii=False, indent=1)


def load_scene(path: str) -> list[SceneObject]:
    with open(path, encoding="utf-8") as f:
        return scene_from_dict(json.load(f))
