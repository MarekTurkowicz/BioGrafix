"""Runs registered operations on the document.

Image operations are executed in three steps:

1. copy the pixels of the target image (or of its selected region);
2. run the function in a worker thread so the UI stays responsive;
3. apply the result in the GUI thread as a single undo step.
"""

from __future__ import annotations

import time
import traceback
from collections.abc import Callable
from typing import Any

import numpy as np
from PySide6.QtCore import QObject, QThread, Signal

from ..i18n import tr
from .document import Document
from .imaging import bounding_box_of_mask, normalize_result, paste_region, paste_region_where_selected, to_rgb
from .registry import MIN_SCENE_IMAGES, TARGET_SCENE, TARGET_SHAPE, ImageResults, Operation, Report


class _Worker(QThread):
    done = Signal(object, float)
    failed = Signal(str)

    def __init__(self, fn: Callable[[], Any], parent=None):
        super().__init__(parent)
        self._fn = fn

    def run(self):
        t0 = time.perf_counter()
        try:
            result = self._fn()
        except Exception:
            self.failed.emit(traceback.format_exc())
            return
        self.done.emit(result, time.perf_counter() - t0)


class OperationError(Exception):
    """An error that is reported to the user, e.g. no image selected."""


class OperationRunner(QObject):
    started = Signal(str)                 # label
    finished = Signal(str, float)         # label, seconds
    failed = Signal(str, str)             # label, error text
    report = Signal(str, str)             # title, text
    figure = Signal(object, str)          # matplotlib Figure, title
    cancelled = Signal(str)               # label

    def __init__(self, doc: Document, parent=None, synchronous: bool = False):
        super().__init__(parent)
        self.doc = doc
        self.synchronous = synchronous    # run in the calling thread (tests)
        self._worker: _Worker | None = None
        self._job: dict | None = None
        self._abandoned: list[_Worker] = []   # cancelled workers that are still running

    @property
    def busy(self) -> bool:
        return self._worker is not None

    @property
    def busy_oids(self) -> list[str]:
        """Ids of the images being processed (several for a scene operation)."""
        return list(self._job["oids"]) if self._job and self._worker is not None else []

    @property
    def busy_oid(self) -> str | None:
        """Id of the (first) image being processed, or None."""
        oids = self.busy_oids
        return oids[0] if oids else None

    @property
    def busy_label(self) -> str:
        return self._job["label"] if self._job else ""

    def cancel(self) -> None:
        """Cancel the running operation and discard its result.

        Python threads cannot be stopped from outside, so the worker finishes in
        the background, but its result is ignored and a new operation can start
        immediately.
        """
        worker = self._worker
        if worker is None or self._job is None:
            return
        label = self._job["label"]
        worker.done.disconnect(self._on_done)
        worker.failed.disconnect(self._on_failed)
        self._abandoned.append(worker)
        worker.finished.connect(self._forget_abandoned)
        self._worker, self._job = None, None
        self.cancelled.emit(label)

    def _forget_abandoned(self):
        self._abandoned = [w for w in self._abandoned if not w.isFinished()]

    def shutdown(self) -> None:
        """Wait for all workers to finish; called when the application closes."""
        for w in self._abandoned + ([self._worker] if self._worker else []):
            w.wait()

    # --------------------------------------------------------------- start --
    def run(self, op: Operation, values: dict[str, Any]) -> None:
        if op.target == TARGET_SHAPE:
            self._run_on_shape(op, values)
        elif op.target == TARGET_SCENE:
            self._run_on_scene(op, values)
        else:
            self._run_on_image(op, values)

    def _run_on_shape(self, op: Operation, values):
        obj = self.doc.selected
        if obj is None or (op.shape_kinds and obj.kind not in op.shape_kinds):
            raise OperationError(tr("Select a shape to apply the operation to."))
        label = op.history_label(values)
        self.started.emit(label)
        t0 = time.perf_counter()
        with self.doc.change(label):
            result = op.func(obj, **values)
        self.doc.notify_updated(obj)
        self._emit_side_results(result, op)
        self.finished.emit(label, time.perf_counter() - t0)

    def _run_on_image(self, op: Operation, values):
        if self.busy:
            raise OperationError(tr("Please wait – the previous operation is still running."))
        image = self.doc.target_image()
        if image is None:
            raise OperationError(tr("Select an image to apply the operation to\n(load one with File → Open images…)."))

        roi = image.roi if op.supports_roi else None
        selection_in_roi = None
        if op.supports_roi and image.selection is not None:
            # magic-wand selection: process its bounding box, change only the selected pixels
            roi = bounding_box_of_mask(image.selection)
            selection_in_roi = image.selection[roi[1]:roi[3], roi[0]:roi[2]]
        rgb = to_rgb(image.pixels)
        region = rgb if roi is None else rgb[roi[1]:roi[3], roi[0]:roi[2]]
        region = np.array(region, copy=True)  # the function may modify its own copy
        suffix = " " + tr("[selection]") if selection_in_roi is not None else " " + tr("[region]") if roi else ""
        label = op.history_label(values) + suffix
        self._job = dict(op=op, label=label, oids=[image.oid], source_pixels=image.pixels,
                         roi=roi, selection_in_roi=selection_in_roi, region_shape=region.shape)

        def compute():
            return op.func(region, **values)

        self._start(label, compute)

    def _run_on_scene(self, op: Operation, values):
        """Run a comparison on every image in the scene (whole images, regions are ignored)."""
        if self.busy:
            raise OperationError(tr("Please wait – the previous operation is still running."))
        images = self.doc.images()
        if len(images) < MIN_SCENE_IMAGES:
            raise OperationError(tr("Load at least {n} images to compare (File → Open images…).", n=MIN_SCENE_IMAGES))
        arrays = [np.array(to_rgb(img.pixels), copy=True) for img in images]
        names = [img.display_name for img in images]
        label = op.history_label(values)
        self._job = dict(op=op, label=label, oids=[img.oid for img in images],
                         source_pixels=[img.pixels for img in images])

        def compute():
            return op.func(arrays, names, **values)

        self._start(label, compute)

    def _start(self, label: str, compute: Callable[[], Any]) -> None:
        """Run ``compute`` in a worker thread (or synchronously in tests) and apply its result."""
        if self.synchronous:
            self.started.emit(label)
            t0 = time.perf_counter()
            try:
                result = compute()
            except Exception:
                self._on_failed(traceback.format_exc())
                return
            self._on_done(result, time.perf_counter() - t0)
            return

        # slots of this object run in the GUI thread
        self._worker = _Worker(compute, self)
        self._worker.done.connect(self._on_done)
        self._worker.failed.connect(self._on_failed)
        self._worker.finished.connect(self._worker.deleteLater)
        self.started.emit(label)  # emitted after the worker is set, so busy is True
        self._worker.start()

    def _on_done(self, result, seconds):
        job, self._job, self._worker = self._job, None, None
        try:
            if job["op"].target == TARGET_SCENE:
                self._apply_scene_result(job["op"], job["label"], job["oids"], job["source_pixels"], result)
            else:
                self._apply_image_result(job["op"], job["label"], job["oids"][0], job["source_pixels"],
                                         job["roi"], job["region_shape"], result, job["selection_in_roi"])
        except Exception as e:
            self.failed.emit(job["label"], tr(str(e)))
            return
        self.finished.emit(job["label"], seconds)

    def _on_failed(self, text):
        job, self._job, self._worker = self._job, None, None
        self.failed.emit(job["label"], text)

    # ----------------------------------------------------------------- result --
    def _apply_image_result(self, op, label, oid, source_pixels, roi, region_shape, result, selection_in_roi=None):
        new_pixels = None
        for item in _as_list(result):
            if isinstance(item, np.ndarray):
                arr = normalize_result(item)
                if roi is not None:
                    if arr.shape[:2] != region_shape[:2]:
                        raise OperationError(tr("The operation changed the image size – it cannot be pasted into the selected region."))
                    arr = paste_region(source_pixels, arr, roi) if selection_in_roi is None \
                        else paste_region_where_selected(source_pixels, arr, roi, selection_in_roi)
                new_pixels = arr

        if new_pixels is not None:
            image = self.doc.find(oid)
            if image is None:
                raise OperationError(tr("The image was deleted during processing – result discarded."))
            if image.pixels is not source_pixels:
                raise OperationError(tr("The image changed during processing – result discarded."))
            with self.doc.change(label):
                image.set_pixels(new_pixels)

        self._emit_side_results(result, op)

    def _apply_scene_result(self, op, label, oids, source_pixels, result):
        """Replace the pixels of every image the scene operation changed, as one undo step."""
        changes = []
        for item in _as_list(result):
            if not isinstance(item, ImageResults):
                continue
            if len(item.pixels) != len(oids):
                raise OperationError(tr("The operation returned a result for a different number of images."))
            for oid, source, arr in zip(oids, source_pixels, item.pixels, strict=True):
                if arr is None:
                    continue
                image = self.doc.find(oid)
                if image is None:
                    raise OperationError(tr("The image was deleted during processing – result discarded."))
                if image.pixels is not source:
                    raise OperationError(tr("The image changed during processing – result discarded."))
                changes.append((image, normalize_result(arr)))
        if changes:
            with self.doc.change(label):
                for image, arr in changes:
                    image.set_pixels(arr)
        self._emit_side_results(result, op)

    def _emit_side_results(self, result, op):
        for item in _as_list(result):
            if isinstance(item, Report):
                self.report.emit(item.title or tr(op.label), item.text)
            elif _is_figure(item):
                self.figure.emit(item, tr(op.label))


def _as_list(result) -> list[Any]:
    if result is None:
        return []
    if isinstance(result, (list, tuple)):
        return list(result)
    return [result]


def _is_figure(obj) -> bool:
    try:
        from matplotlib.figure import Figure
    except ImportError:  # pragma: no cover
        return False
    return isinstance(obj, Figure)
