"""GUI-independent image processing algorithms.

Functions work either on a flat pixel list ``[(r, g, b), ...]`` or on numpy
arrays of shape (H, W, 3), dtype uint8. They are exposed to the user interface
through :mod:`biografix.operations`.
"""
