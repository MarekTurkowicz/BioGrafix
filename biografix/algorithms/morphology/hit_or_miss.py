"""Hit-or-miss transform."""


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
