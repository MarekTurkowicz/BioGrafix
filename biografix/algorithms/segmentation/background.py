"""Foreground segmentation with RemBG.

On first use rembg downloads its model (~170 MB) to the user's home directory.
"""

import numpy as np
from PIL import Image


def remove_background(image: np.ndarray) -> np.ndarray:
    """Return an RGBA image with a transparent background."""
    from rembg import remove  # imported lazily: loading rembg takes several seconds

    output_image = remove(Image.fromarray(np.asarray(image)))
    return np.asarray(output_image.convert("RGBA"))
