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
from .imaging import normalize_result, paste_region, to_rgb
from .registry import TARGET_SHAPE, Operation, Report


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
    def busy_oid(self) -> str | None:
        """Id of the image being processed, or None."""
        return self._job["oid"] if self._job and self._worker is not None else None

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
        rgb = to_rgb(image.pixels)
        region = rgb if roi is None else rgb[roi[1]:roi[3], roi[0]:roi[2]]
        region = np.array(region, copy=True)  # the function may modify its own copy
        label = op.history_label(values) + (" " + tr("[region]") if roi else "")
        self._job = dict(op=op, label=label, oid=image.oid, source_pixels=image.pixels,
                         roi=roi, region_shape=region.shape)

        def compute():
            return op.func(region, **values)

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
            self._apply_image_result(job["op"], job["label"], job["oid"], job["source_pixels"],
                                     job["roi"], job["region_shape"], result)
        except Exception as e:
            self.failed.emit(job["label"], tr(str(e)))
            return
        self.finished.emit(job["label"], seconds)

    def _on_failed(self, text):
        job, self._job, self._worker = self._job, None, None
        self.failed.emit(job["label"], text)

    # ----------------------------------------------------------------- result --
    def _apply_image_result(self, op, label, oid, source_pixels, roi, region_shape, result):
        new_pixels = None
        for item in _as_list(result):
            if isinstance(item, np.ndarray):
                arr = normalize_result(item)
                if roi is not None:
                    if arr.shape[:2] != region_shape[:2]:
                        raise OperationError(tr("The operation changed the image size – it cannot be pasted into the selected region."))
                    arr = paste_region(source_pixels, arr, roi)
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
