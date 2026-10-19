"""Scene serialisation to and from JSON.

Images are embedded as base64 PNG so that edits survive a save. Files that
contain only a ``polygons`` list are also accepted.
"""

import json
from typing import List

from .objects import PolygonShape, SceneObject, object_from_dict

FORMAT_VERSION = 2


def scene_to_dict(objects: List[SceneObject]) -> dict:
    return {"app": "BioGrafix", "version": FORMAT_VERSION, "objects": [o.to_dict() for o in objects]}


def scene_from_dict(data: dict) -> List[SceneObject]:
    objects = [object_from_dict(d) for d in data.get("objects", [])]
    # polygons-only file
    for item in data.get("polygons", []):
        pts = [(float(x), float(y)) for (x, y) in item.get("points", [])]
        if pts:
            objects.append(PolygonShape(points=pts, closed=bool(item.get("closed", True))))
    return objects


def save_scene(path: str, objects: List[SceneObject]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(scene_to_dict(objects), f, ensure_ascii=False, indent=1)


def load_scene(path: str) -> List[SceneObject]:
    with open(path, "r", encoding="utf-8") as f:
        return scene_from_dict(json.load(f))
