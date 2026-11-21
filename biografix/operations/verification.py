"""Verification and identification of fingerprints (task 5).

- Verify: two fingerprints in the scene – are they the same finger?
- Identify: the selected fingerprint is compared with every other image in the
  scene, which act as the database; the best match above the threshold is the
  identified person. With FVC-style names (``<person>_<impression>.tif``) the
  candidates are grouped by person.

Both draw the enhanced skeleton of the compared prints with the paired minutiae
in green and the unpaired ones in red.
"""

import os

import numpy as np

from ..algorithms.matching import (
    MIN_PAIRED_MINUTIAE,
    SAME_FINGER_THRESHOLD,
    FingerprintTemplate,
    MatchResult,
    cached_template,
    match_templates,
)
from ..algorithms.minutiae import draw_markers
from ..algorithms.skeletonization import mask_to_image
from ..core.registry import TARGET_SCENE, ImageResults, Report, operation
from ..i18n import tr

CAT = "Feature extraction"
SECTION = "Verification and identification"
PAIRED_COLOUR = (20, 160, 40)
UNPAIRED_COLOUR = (230, 30, 30)
RANKING_LENGTH = 5


def draw_comparison(template: FingerprintTemplate, paired: set[int]) -> np.ndarray:
    """Skeleton with paired minutiae in green and the others in red."""
    out = mask_to_image(template.skeleton)
    draw_markers(out, [m for i, m in enumerate(template.minutiae) if i not in paired], lambda _m: UNPAIRED_COLOUR)
    draw_markers(out, [m for i, m in enumerate(template.minutiae) if i in paired], lambda _m: PAIRED_COLOUR)
    return out


def percent(similarity: float) -> str:
    return f"{similarity * 100:.0f}%"


def match_summary(name_a: str, name_b: str, a: FingerprintTemplate, b: FingerprintTemplate, result: MatchResult) -> str:
    return tr("{a} ↔ {b}: similarity {similarity} ({matched} paired minutiae of {na} and {nb}), "
              "rotation {rotation}°, shift ({dx}, {dy}) px",
              a=name_a, b=name_b, similarity=percent(result.similarity), matched=result.matched, na=len(a), nb=len(b),
              rotation=f"{np.degrees(result.rotation):.0f}", dx=f"{result.translation[0]:.0f}",
              dy=f"{result.translation[1]:.0f}")


def person_of(name: str) -> str:
    """Person (finger) id from a file name like ``101_3.tif``; otherwise the name without extension."""
    stem = os.path.splitext(name)[0]
    return stem.split("_")[0] if "_" in stem else stem


@operation("verify.same_finger", CAT, "Verify: same finger?", section=SECTION, target=TARGET_SCENE,
           description="Compares the two fingerprints in the scene: Gabor enhancement, K3M skeleton, minutiae,\n"
                       "alignment by the Hough transform and pairing of minutiae. Decides whether they are the same finger.")
def verify(images, names, _query_index):
    if len(images) != 2:
        raise ValueError("Verification compares exactly two images – leave two fingerprints in the scene.")
    a, b = cached_template(images[0]), cached_template(images[1])
    result = match_templates(a, b)
    rule = tr("similarity ≥ {threshold} and at least {pairs} paired minutiae",
              threshold=percent(SAME_FINGER_THRESHOLD), pairs=MIN_PAIRED_MINUTIAE)
    decision = (tr("Decision: SAME FINGER ({rule})", rule=rule) if result.same_finger
                else tr("Decision: DIFFERENT FINGERS (needed: {rule})", rule=rule))
    drawn = [draw_comparison(a, {i for i, _ in result.pairs}), draw_comparison(b, {j for _, j in result.pairs})]
    text = match_summary(names[0], names[1], a, b, result) + "\n" + decision
    return ImageResults(drawn), Report(text, tr("Verification"))


@operation("verify.identify", CAT, "Identify the selected fingerprint", section=SECTION, target=TARGET_SCENE,
           description="The selected fingerprint is compared with every other image in the scene (the database).\n"
                       "Shows the ranking and the identified person when the best similarity reaches the threshold.")
def identify(images, names, query_index):
    query = cached_template(images[query_index])
    candidates = []
    for i, image in enumerate(images):
        if i != query_index:
            template = cached_template(image)
            candidates.append((match_templates(query, template), i, template))
    candidates.sort(key=lambda c: -c[0].similarity)

    lines = [tr("Query: {name} ({n} minutiae), database: {count} images",
                name=names[query_index], n=len(query), count=len(candidates))]
    for rank, (result, i, _template) in enumerate(candidates[:RANKING_LENGTH], start=1):
        lines.append(tr("{rank}. {name} (person {person}) – similarity {similarity}, {matched} paired minutiae",
                        rank=rank, name=names[i], person=person_of(names[i]),
                        similarity=percent(result.similarity), matched=result.matched))
    best_result, best_index, best_template = candidates[0]
    if best_result.same_finger:
        lines.append(tr("Identified: person {person} ({name})", person=person_of(names[best_index]), name=names[best_index]))
    else:
        lines.append(tr("Not identified: no image reaches similarity {threshold} with {pairs} paired minutiae",
                        threshold=percent(SAME_FINGER_THRESHOLD), pairs=MIN_PAIRED_MINUTIAE))

    drawn: list = [None] * len(images)
    drawn[query_index] = draw_comparison(query, {i for i, _ in best_result.pairs})
    drawn[best_index] = draw_comparison(best_template, {j for _, j in best_result.pairs})
    return ImageResults(drawn), Report("\n".join(lines), tr("Identification"))
