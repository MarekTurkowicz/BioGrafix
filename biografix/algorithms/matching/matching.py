"""Minutiae matching with Hough-transform alignment (after Ratha et al., 1996).

Two impressions of the same finger differ by a rotation and a translation (and
some distortion). Every pair of minutiae of the same type, one from each print,
proposes the transform that would put the first onto the second:

- rotation Δθ = θ_B − θ_A (ridge orientations, limited to ±``MAX_ROTATION``),
- translation t = p_B − R(Δθ)·p_A.

The proposals are counted in bins of (Δθ, tx, ty); genuine pairs agree on one
bin, random pairs scatter. For the ``CANDIDATE_BINS`` most voted bins, print A
is transformed and minutiae are paired greedily, closest first, when they are
closer than ``DISTANCE_TOLERANCE`` px, their orientations differ by less than
``ANGLE_TOLERANCE`` and their types agree. The best alignment gives

    similarity = matched / sqrt(n_A · n_B)        (0 … 1)

and the prints are declared the same finger when the similarity reaches
``SAME_FINGER_THRESHOLD`` and at least ``MIN_PAIRED_MINUTIAE`` minutiae are
paired (a few chance pairs between prints with few minutiae are not enough).
"""

from dataclasses import dataclass, field

import numpy as np

from .template import FingerprintTemplate

MAX_ROTATION = np.radians(60)
ROTATION_BIN = np.radians(4)
TRANSLATION_BIN = 8.0          # px
CANDIDATE_BINS = 6
DISTANCE_TOLERANCE = 15.0      # px
ANGLE_TOLERANCE = np.radians(20)
MIN_MINUTIAE = 3

# Chosen on FVC2002 DB1_B + DB2_B (560 genuine, 90 impostor comparisons, see
# tools/evaluate_verification.py): FAR 2.2 %, FRR 15 %. Verification favours
# rejecting a genuine user now and then over accepting an impostor; the equal
# error rate (12.5 %) lies at a similarity of about 0.16.
SAME_FINGER_THRESHOLD = 0.18
MIN_PAIRED_MINUTIAE = 6


@dataclass(frozen=True)
class MatchResult:
    similarity: float                      # matched / sqrt(n_A · n_B)
    pairs: list[tuple[int, int]] = field(default_factory=list)  # (index in A, index in B)
    rotation: float = 0.0                  # radians, A → B
    translation: tuple[float, float] = (0.0, 0.0)

    @property
    def matched(self) -> int:
        return len(self.pairs)

    @property
    def same_finger(self) -> bool:
        return self.similarity >= SAME_FINGER_THRESHOLD and self.matched >= MIN_PAIRED_MINUTIAE


def orientation_difference(a, b):
    """Smallest difference between ridge orientations taken modulo π."""
    d = np.abs(a - b) % np.pi
    return np.minimum(d, np.pi - d)


def transform_votes(a: np.ndarray, b: np.ndarray):
    """Rotation and translation proposed by every compatible pair of minutiae, and their bins."""
    ia, ib = np.nonzero(a[:, None, 3] == b[None, :, 3])  # same type only
    rotation = (b[ib, 2] - a[ia, 2] + np.pi / 2) % np.pi - np.pi / 2
    plausible = np.abs(rotation) <= MAX_ROTATION
    ia, ib, rotation = ia[plausible], ib[plausible], rotation[plausible]
    cos, sin = np.cos(rotation), np.sin(rotation)
    tx = b[ib, 0] - (cos * a[ia, 0] - sin * a[ia, 1])
    ty = b[ib, 1] - (sin * a[ia, 0] + cos * a[ia, 1])
    bins = np.stack([np.round(rotation / ROTATION_BIN), np.round(tx / TRANSLATION_BIN),
                     np.round(ty / TRANSLATION_BIN)], axis=1).astype(int)
    return rotation, tx, ty, bins


def pair_minutiae(a: np.ndarray, b: np.ndarray, rotation: float, tx: float, ty: float) -> list[tuple[int, int]]:
    """Greedy one-to-one pairing of the transformed minutiae of A with those of B, closest first."""
    cos, sin = np.cos(rotation), np.sin(rotation)
    ax = cos * a[:, 0] - sin * a[:, 1] + tx
    ay = sin * a[:, 0] + cos * a[:, 1] + ty
    distance = np.hypot(ax[:, None] - b[None, :, 0], ay[:, None] - b[None, :, 1])
    compatible = ((distance < DISTANCE_TOLERANCE)
                  & (orientation_difference(a[:, None, 2] + rotation, b[None, :, 2]) < ANGLE_TOLERANCE)
                  & (a[:, None, 3] == b[None, :, 3]))
    candidates = np.argwhere(compatible)
    candidates = candidates[np.argsort(distance[compatible], kind="stable")]
    used_a, used_b, pairs = set(), set(), []
    for i, j in candidates:
        if i not in used_a and j not in used_b:
            used_a.add(int(i))
            used_b.add(int(j))
            pairs.append((int(i), int(j)))
    return pairs


def match_templates(template_a: FingerprintTemplate, template_b: FingerprintTemplate) -> MatchResult:
    """Best alignment of A onto B and the similarity of the two fingerprints."""
    a, b = template_a.points, template_b.points
    if len(a) < MIN_MINUTIAE or len(b) < MIN_MINUTIAE:
        return MatchResult(0.0)
    rotation, tx, ty, bins = transform_votes(a, b)
    if len(bins) == 0:
        return MatchResult(0.0)
    voted_bins, votes = np.unique(bins, axis=0, return_counts=True)
    best = MatchResult(0.0)
    for voted_bin in voted_bins[np.argsort(-votes, kind="stable")[:CANDIDATE_BINS]]:
        in_bin = (bins == voted_bin).all(axis=1)
        r, dx, dy = rotation[in_bin].mean(), tx[in_bin].mean(), ty[in_bin].mean()
        pairs = pair_minutiae(a, b, r, dx, dy)
        similarity = len(pairs) / np.sqrt(len(a) * len(b))
        if similarity > best.similarity:
            best = MatchResult(float(similarity), pairs, float(r), (float(dx), float(dy)))
    return best
