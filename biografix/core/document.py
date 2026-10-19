"""Document model: scene objects, selection and undo history.

Every change goes through :meth:`Document.change` or the
:meth:`Document.begin_interactive` / :meth:`Document.end_interactive` pair
used for mouse drags. The history stores snapshots taken before and after each
change; they are cheap because shapes are small and pixel arrays are shared
read-only.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from typing import Callable, List, Optional, Sequence, Set, Tuple

from PySide6.QtCore import QObject, Signal
from PySide6.QtGui import QUndoCommand, QUndoStack

from ..constants import UNDO_LIMIT
from ..i18n import tr
from ..scene.objects import RasterImage, SceneObject, new_oid


@dataclass(frozen=True)
class SceneState:
    objects: Tuple[SceneObject, ...]
    selected_oid: Optional[str]


class SnapshotCommand(QUndoCommand):
    """Undo command that restores the snapshot taken before or after a change."""

    def __init__(self, doc: "Document", label: str, before: SceneState, after: SceneState):
        super().__init__(label)
        self._doc = doc
        self._before = before
        self._after = after
        self._skip_first_redo = True  # the change has already been applied when the command is pushed

    def redo(self):
        if self._skip_first_redo:
            self._skip_first_redo = False
            return
        self._doc._restore(self._after)

    def undo(self):
        self._doc._restore(self._before)


class Document(QObject):
    changed = Signal()                 # the object list changed
    selection_changed = Signal()       # the selection changed
    object_updated = Signal(str)       # one object changed in place (drag, region)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.objects: List[SceneObject] = []
        self.selected_oid: Optional[str] = None
        # veto for deletions: f(ids) -> bool, set by the main window to protect images being processed
        self.remove_guard: Optional[Callable[[Set[str]], bool]] = None
        self.file_path: Optional[str] = None
        self.undo_stack = QUndoStack(self)
        self.undo_stack.setUndoLimit(UNDO_LIMIT)
        self._interactive_before: Optional[SceneState] = None
        # True while the history changes because of a new step or a reset rather than undo / redo
        self.recording = False

    # ----------------------------------------------------------------- access --
    def find(self, oid: Optional[str]) -> Optional[SceneObject]:
        if oid is None:
            return None
        return next((o for o in self.objects if o.oid == oid), None)

    def index_of(self, oid: str) -> int:
        return next((i for i, o in enumerate(self.objects) if o.oid == oid), -1)

    @property
    def selected(self) -> Optional[SceneObject]:
        return self.find(self.selected_oid)

    def images(self) -> List[RasterImage]:
        return [o for o in self.objects if isinstance(o, RasterImage)]

    def target_image(self) -> Optional[RasterImage]:
        """The selected image, or the only image in the scene when nothing is selected."""
        sel = self.selected
        if isinstance(sel, RasterImage):
            return sel
        if sel is None:
            imgs = self.images()
            if len(imgs) == 1:
                return imgs[0]
        return None

    def object_at(self, x: float, y: float, tol: float, kinds: Sequence[str] = ()) -> Optional[SceneObject]:
        """The topmost object at (x, y)."""
        for obj in reversed(self.objects):
            if kinds and obj.kind not in kinds:
                continue
            if obj.hit_test(x, y, tol):
                return obj
        return None

    # -------------------------------------------------------------- snapshots --
    def snapshot(self) -> SceneState:
        return SceneState(tuple(o.clone() for o in self.objects), self.selected_oid)

    def _restore(self, state: SceneState) -> None:
        self.objects = [o.clone() for o in state.objects]
        self.selected_oid = state.selected_oid if self.find(state.selected_oid) else None
        self.changed.emit()
        self.selection_changed.emit()

    def _push(self, label: str, before: SceneState) -> None:
        after = self.snapshot()
        self.recording = True
        try:
            self.undo_stack.push(SnapshotCommand(self, label, before, after))
        finally:
            self.recording = False
        self.changed.emit()
        self.selection_changed.emit()

    @contextmanager
    def change(self, label: str):
        """Record the enclosed changes as one undo step; an exception rolls them back."""
        before = self.snapshot()
        try:
            yield
        except Exception:
            self._restore(before)
            raise
        self._push(label, before)

    def begin_interactive(self) -> None:
        self._interactive_before = self.snapshot()

    def end_interactive(self, label: str, modified: bool = True) -> None:
        before, self._interactive_before = self._interactive_before, None
        if before is not None and modified:
            self._push(label, before)

    def cancel_interactive(self) -> None:
        """Abort a drag (Esc) and restore the state from before it started."""
        before, self._interactive_before = self._interactive_before, None
        if before is not None:
            self._restore(before)

    def notify_updated(self, obj: SceneObject) -> None:
        self.object_updated.emit(obj.oid)

    # --------------------------------------------------------- scene editing --
    def reset(self, objects: List[SceneObject], file_path: Optional[str] = None) -> None:
        """Replace the scene (new or opened file) and clear the history."""
        self.objects = list(objects)
        self.selected_oid = None
        self.file_path = file_path
        self.recording = True
        try:
            self.undo_stack.clear()
            self.undo_stack.setClean()
        finally:
            self.recording = False
        self.changed.emit()
        self.selection_changed.emit()

    def add_objects(self, objs: List[SceneObject], label: str) -> None:
        if not objs:
            return
        with self.change(label):
            self.objects.extend(objs)
            self.selected_oid = objs[-1].oid

    def can_remove(self, oids) -> bool:
        """Ask the deletion veto whether the objects may be removed."""
        return self.remove_guard is None or self.remove_guard(set(oids))

    def remove(self, oid: str) -> None:
        obj = self.find(oid)
        if obj is None or not self.can_remove([oid]):
            return
        with self.change(f"{tr('Delete')}: {obj.display_name}"):
            self.objects = [o for o in self.objects if o.oid != oid]
            if self.selected_oid == oid:
                self.selected_oid = None

    def duplicate(self, oid: str) -> None:
        obj = self.find(oid)
        if obj is None:
            return
        dup = obj.clone()
        dup.oid = new_oid()
        dup.name = tr("{name} (copy)", name=obj.display_name)
        dup.move(15, 15)
        self.add_objects([dup], tr("Duplicate: {name}", name=obj.display_name))

    def move_z(self, oid: str, delta: int) -> None:
        i = self.index_of(oid)
        j = i + delta
        if i < 0 or not 0 <= j < len(self.objects):
            return
        with self.change(tr("Reorder layers")):
            self.objects[i], self.objects[j] = self.objects[j], self.objects[i]

    def clear(self) -> None:
        if self.objects and self.can_remove(o.oid for o in self.objects):
            with self.change(tr("Clear scene")):
                self.objects = []
                self.selected_oid = None

    def rename(self, oid: str, name: str) -> None:
        obj = self.find(oid)
        if obj is None or obj.name == name:
            return
        with self.change(tr("Rename: {name}", name=name)):
            obj.name = name

    def select(self, oid: Optional[str]) -> None:
        """Change the selection (not recorded in the history)."""
        if oid != self.selected_oid:
            self.selected_oid = oid
            self.selection_changed.emit()

    def set_roi(self, image: RasterImage, roi) -> None:
        """Set or clear an image region (not recorded in the history)."""
        image.roi = roi
        self.object_updated.emit(image.oid)
        self.selection_changed.emit()
