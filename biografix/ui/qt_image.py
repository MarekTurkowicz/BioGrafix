"""Conversion between numpy arrays and QImage."""

import numpy as np
from PySide6.QtGui import QImage


def array_to_qimage(arr: np.ndarray) -> QImage:
    arr = np.ascontiguousarray(arr, dtype=np.uint8)
    if arr.ndim == 2:
        arr = np.ascontiguousarray(np.stack([arr] * 3, axis=-1))
    h, w, c = arr.shape
    fmt = QImage.Format.Format_RGB888 if c == 3 else QImage.Format.Format_RGBA8888
    data = arr.tobytes()
    # copy() detaches the QImage from the numpy buffer
    return QImage(data, w, h, w * c, fmt).copy()
