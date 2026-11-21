"""Binary dilation."""


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
