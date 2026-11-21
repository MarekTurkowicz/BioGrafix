"""Measure verification accuracy on an FVC-style fingerprint database.

Download e.g. FVC2002 DB1_B (http://bias.csr.unibo.it/fvc2002/) and run:

    python tools/evaluate_verification.py path/to/DB1_B [path/to/DB2_B ...]

Images are named <finger>_<impression>.tif. Following the FVC protocol, every
pair of impressions of the same finger is a genuine comparison and the first
impressions of every pair of different fingers are impostor comparisons. The
script prints the false rejection rate (FRR), the false acceptance rate (FAR) at
the current threshold and the equal error rate (EER), where FAR = FRR.
"""

import itertools
import os
import re
import sys
import time

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from biografix.algorithms.matching import MatchResult, extract_template, match_templates  # noqa: E402

NAME = re.compile(r"(\d+)_(\d+)\.(tif|tiff|bmp|png)$", re.IGNORECASE)


def evaluate(folder: str) -> tuple[list[MatchResult], list[MatchResult]]:
    files = {(int(m[1]), int(m[2])): os.path.join(folder, name)
             for name in sorted(os.listdir(folder)) if (m := NAME.match(name))}
    started = time.perf_counter()
    templates = {key: extract_template(np.asarray(Image.open(path).convert("L"))) for key, path in files.items()}
    extraction = (time.perf_counter() - started) / len(templates)
    fingers = sorted({finger for finger, _ in templates})
    impressions = sorted({impression for _, impression in templates})
    started = time.perf_counter()
    genuine = [match_templates(templates[(f, a)], templates[(f, b)])
               for f in fingers for a, b in itertools.combinations(impressions, 2)
               if (f, a) in templates and (f, b) in templates]
    impostor = [match_templates(templates[(a, impressions[0])], templates[(b, impressions[0])])
                for a, b in itertools.combinations(fingers, 2)]
    matching = (time.perf_counter() - started) / (len(genuine) + len(impostor))
    counts = [len(t) for t in templates.values()]
    print(f"{os.path.basename(folder.rstrip('/'))}: {len(templates)} images, median {int(np.median(counts))} minutiae, "
          f"{extraction * 1000:.0f} ms per template, {matching * 1000:.1f} ms per comparison")
    report(genuine, impostor)
    return genuine, impostor


def equal_error_rate(genuine: np.ndarray, impostor: np.ndarray) -> tuple[float, float]:
    """(EER, threshold) where the false rejection and false acceptance rates are closest."""
    best = (1.0, 0.0)
    for threshold in np.unique(np.concatenate([genuine, impostor])):
        error = max((genuine < threshold).mean(), (impostor >= threshold).mean())
        if error < best[0]:
            best = (error, float(threshold))
    return best


def report(genuine: list[MatchResult], impostor: list[MatchResult]) -> None:
    frr = np.mean([not r.same_finger for r in genuine])
    far = np.mean([r.same_finger for r in impostor])
    g = np.array([r.similarity for r in genuine])
    i = np.array([r.similarity for r in impostor])
    eer, eer_threshold = equal_error_rate(g, i)
    print(f"  genuine {len(g)}: mean similarity {g.mean():.3f} | impostor {len(i)}: mean {i.mean():.3f}, max {i.max():.3f}")
    print(f"  decision rule: FRR {frr * 100:.1f}%, FAR {far * 100:.1f}% | EER {eer * 100:.1f}% at similarity {eer_threshold:.3f}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    results = [evaluate(folder) for folder in sys.argv[1:]]
    if len(results) > 1:
        print("all databases:")
        report([r for g, _ in results for r in g], [r for _, i in results for r in i])
