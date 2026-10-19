"""Binary morphology.

A binary image is a flat list of 0/1 values of length ``w * h``. A structuring
element is a list of integer rows: 1 = object, 0 = don't care, -1 = background
(hit-or-miss only).
"""


def parse_structuring_element(raw: str):
    """Parse a structuring element: one row per line, values separated by spaces."""
    raw = raw.strip()
    if not raw:
        raise ValueError("The structuring element is empty.")

    rows = []
    width = None
    for line in raw.splitlines():
        parts = line.strip().split()
        if not parts:
            continue
        try:
            row = [int(p) for p in parts]
        except ValueError:
            raise ValueError(
                "The structuring element may contain only integers (0, 1 and optionally -1)."
            )
        if width is None:
            width = len(row)
        elif len(row) != width:
            raise ValueError("All rows of the structuring element must have the same length.")
        rows.append(row)

    if not rows or width is None:
        raise ValueError("Could not read the structuring element.")

    return rows


def pixels_to_binary(pixels):
    """Convert RGB pixels to a binary 0/1 image."""
    bin_img = []
    for r, g, b in pixels:
        if (r + g + b) / 3 >= 128:
            bin_img.append(1)
        else:
            bin_img.append(0)
    return bin_img


def binary_to_pixels(bin_img):
    """Convert a binary 0/1 image to black and white RGB pixels."""
    pixels = []
    for v in bin_img:
        val = 255 if v else 0
        pixels.append((val, val, val))
    return pixels


def dilate(bin_img, w, h, se):
    kh = len(se)
    kw = len(se[0])
    cy = kh // 2
    cx = kw // 2
    out = [0] * (w * h)

    for y in range(h):
        for x in range(w):
            val = 0
            for j in range(kh):
                for i in range(kw):
                    if se[j][i] != 1:
                        continue
                    xx = x + i - cx
                    yy = y + j - cy
                    if 0 <= xx < w and 0 <= yy < h:
                        if bin_img[yy * w + xx] == 1:
                            val = 1
                            break
                if val:
                    break
            out[y * w + x] = val
    return out


def erode(bin_img, w, h, se):
    kh = len(se)
    kw = len(se[0])
    cy = kh // 2
    cx = kw // 2
    out = [0] * (w * h)

    for y in range(h):
        for x in range(w):
            val = 1
            for j in range(kh):
                for i in range(kw):
                    if se[j][i] != 1:
                        continue
                    xx = x + i - cx
                    yy = y + j - cy
                    if not (0 <= xx < w and 0 <= yy < h):
                        val = 0
                        break
                    if bin_img[yy * w + xx] == 0:
                        val = 0
                        break
                if val == 0:
                    break
            out[y * w + x] = val
    return out


def hit_or_miss(bin_img, w, h, se):
    """Hit-or-miss transform with 1 = object, -1 = background, 0 = don't care."""
    kh = len(se)
    kw = len(se[0])
    cy = kh // 2
    cx = kw // 2
    out = [0] * (w * h)

    for y in range(h):
        for x in range(w):
            match = True
            for j in range(kh):
                for i in range(kw):
                    v = se[j][i]
                    if v == 0:
                        continue
                    xx = x + i - cx
                    yy = y + j - cy

                    if v == 1:
                        if not (0 <= xx < w and 0 <= yy < h):
                            match = False
                            break
                        if bin_img[yy * w + xx] != 1:
                            match = False
                            break
                    elif v == -1:
                        # pixels outside the image count as background
                        if 0 <= xx < w and 0 <= yy < h:
                            if bin_img[yy * w + xx] != 0:
                                match = False
                                break
                if not match:
                    break
            out[y * w + x] = 1 if match else 0
    return out


MODES = ("dilate", "erode", "open", "close", "thin", "thicken")


def apply_morphology(pixels, w, h, se, mode):
    """Apply the morphological operation ``mode`` and return black and white pixels."""
    bin_img = pixels_to_binary(pixels)

    if mode == "dilate":
        out = dilate(bin_img, w, h, se)
    elif mode == "erode":
        out = erode(bin_img, w, h, se)
    elif mode == "open":
        tmp = erode(bin_img, w, h, se)
        out = dilate(tmp, w, h, se)
    elif mode == "close":
        tmp = dilate(bin_img, w, h, se)
        out = erode(tmp, w, h, se)
    elif mode == "thin":
        hm = hit_or_miss(bin_img, w, h, se)
        out = [1 if (bin_img[i] == 1 and hm[i] == 0) else 0 for i in range(len(bin_img))]
    elif mode == "thicken":
        hm = hit_or_miss(bin_img, w, h, se)
        out = [1 if (bin_img[i] == 1 or hm[i] == 1) else 0 for i in range(len(bin_img))]
    else:
        raise ValueError(f"Unknown morphology mode: {mode}")

    return binary_to_pixels(out)
