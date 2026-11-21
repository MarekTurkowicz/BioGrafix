"""Synthetic fingerprints with minutiae at known places, for the matching tests.

Parallel ridges (period 9 px) get phase singularities: going once around a
singularity adds a full period to the phase, so one ridge forks or ends there.
The same seed gives the same "finger"; a rotation and shift give another
impression of it.
"""

import numpy as np


def synthetic_print(seed: int, size: int = 320, rotation_deg: float = 0.0, shift=(0.0, 0.0),
                    singularities: int = 10) -> np.ndarray:
    """(size, size) uint8 fingerprint-like image: dark ridges inside a disk on a white background."""
    rng = np.random.default_rng(seed)
    ridge_angle = rng.uniform(0, np.pi)
    centres = rng.uniform(70, size - 70, (singularities, 2))
    signs = rng.choice([-1, 1], singularities)
    y, x = np.mgrid[0:size, 0:size].astype(float)
    r = np.radians(rotation_deg)
    centre = size / 2
    # coordinates in the original impression: undo the shift and the rotation about the centre
    xs, ys = x - shift[0] - centre, y - shift[1] - centre
    u = np.cos(r) * xs + np.sin(r) * ys + centre
    v = -np.sin(r) * xs + np.cos(r) * ys + centre
    phase = 2 * np.pi / 9 * (u * np.cos(ridge_angle) + v * np.sin(ridge_angle))
    for (px, py), sign in zip(centres, signs, strict=True):
        phase += sign * np.arctan2(v - py, u - px)
    image = 128 + 100 * np.cos(phase)
    inside = (u - centre) ** 2 + (v - centre) ** 2 < (size * 0.42) ** 2
    return np.where(inside, image, 255).astype(np.uint8)
