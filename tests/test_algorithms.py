"""Algorithm and operation registry tests (no GUI)."""

import time
import unittest
from collections import deque

import numpy as np

from biografix.algorithms import geometry, segmentation, skeletonization, thresholding
from biografix.algorithms.color_models import cmyk_to_rgb, rgb_to_cmyk
from biografix.algorithms.skeletonization import k3m, kmm
from biografix.core.imaging import array_to_pixels, paste_region, paste_region_where_selected, pixels_to_array, to_rgb
from biografix.core.params import IntParam, MatrixParam
from biografix.core.registry import REGISTRY, TARGET_IMAGE
from biografix.operations import load_all
from biografix.scene.objects import LineShape, PolygonShape
from biografix.scene.serialization import scene_from_dict, scene_to_dict

load_all()


def sample_image(h=40, w=50, seed=0):
    return (np.random.default_rng(seed).random((h, w, 3)) * 255).astype(np.uint8)


class TestEveryOperation(unittest.TestCase):
    def test_all_image_operations_run_on_defaults(self):
        img = sample_image()
        for op in REGISTRY.values():
            if op.target != TARGET_IMAGE or op.id == "seg.rembg":  # rembg downloads a model
                continue
            with self.subTest(op=op.id):
                values = {p.name: p.validate(p.default) for p in op.params}
                result = op.func(img.copy(), **values)
                items = result if isinstance(result, tuple) else (result,)
                for item in items:
                    if isinstance(item, np.ndarray):
                        self.assertEqual(item.shape[:2], img.shape[:2])
                        self.assertEqual(item.dtype, np.uint8)


class TestThresholding(unittest.TestCase):
    def test_otsu_separates_two_levels(self):
        img = np.zeros((10, 10, 3), np.uint8)
        img[:, 5:] = 200
        img[:, :5] = 50
        gray = thresholding._gray_average(img)
        t = thresholding.otsu_threshold_value(gray)
        self.assertTrue(50 <= t < 200)
        out = thresholding.otsu_threshold(img)
        self.assertTrue((out[:, :5] == 0).all() and (out[:, 5:] == 255).all())

    def test_local_methods_handle_uneven_lighting(self):
        # ridge-like stripes on a background with a lighting gradient
        h, w = 60, 120
        bg = np.tile(np.linspace(60, 230, w), (h, 1))
        stripes = (np.sin(np.arange(w) / 3.0) > 0.3)[None, :].repeat(h, 0)
        gray = np.where(stripes, bg * 0.55, bg)
        img = np.stack([gray] * 3, -1).astype(np.uint8)
        truth = ~stripes
        for fn in (thresholding.niblack_threshold, thresholding.sauvola_threshold, thresholding.phansalkar_threshold):
            with self.subTest(fn=fn.__name__):
                acc = ((fn(img)[:, :, 0] == 255) == truth).mean()
                self.assertGreater(acc, 0.95)

    def test_local_mean_std_no_uint8_overflow(self):
        gray = np.full((20, 20), 200, np.uint8)
        m, s = thresholding.local_mean_std(gray, 5)
        self.assertTrue(np.allclose(m, 200))
        self.assertTrue(np.allclose(s, 0))


class TestGeometry(unittest.TestCase):
    def test_bresenham_endpoints(self):
        pts = geometry.bresenham_line(0, 0, 10, 4)
        self.assertEqual(pts[0], (0, 0))
        self.assertEqual(pts[-1], (10, 4))
        self.assertEqual(len(pts), 11)

    def test_rotation_matrix(self):
        (x, y), = geometry.apply_matrix([(10, 0)], geometry.matrix_rotate(0, 0, np.pi / 2))
        self.assertAlmostEqual(x, 0, places=6)
        self.assertAlmostEqual(y, 10, places=6)

    def test_bezier_passes_through_endpoints(self):
        pts = [(0, 0), (50, 100), (100, 0)]
        self.assertEqual(geometry.de_casteljau(pts, 0), (0, 0))
        self.assertEqual(geometry.de_casteljau(pts, 1), (100, 0))


class TestFormatsAndParams(unittest.TestCase):
    def test_pixels_roundtrip(self):
        img = sample_image(5, 7)
        pixels, w, h = array_to_pixels(img)
        self.assertTrue(np.array_equal(pixels_to_array(pixels, w, h), img))

    def test_rgba_composited_on_white(self):
        rgba = np.zeros((1, 1, 4), np.uint8)  # black, fully transparent
        self.assertTrue((to_rgb(rgba) == 255).all())

    def test_paste_region(self):
        full = np.zeros((10, 10, 3), np.uint8)
        out = paste_region(full, np.full((2, 3, 3), 9, np.uint8), (1, 2, 4, 4))
        self.assertEqual(int(out.sum()), 2 * 3 * 3 * 9)

    def test_cmyk_roundtrip(self):
        for rgb in [(255, 0, 0), (12, 200, 99), (0, 0, 0)]:
            self.assertEqual(cmyk_to_rgb(*rgb_to_cmyk(*rgb)), rgb)

    def test_param_validation(self):
        with self.assertRaises(ValueError):
            IntParam("w", "Window", 15, 3, 99, odd=True).validate(16)
        se = MatrixParam("se", "SE", integer=True, allowed=(0, 1, -1))
        self.assertEqual(se.validate("0 1\n-1 1"), [[0, 1], [-1, 1]])
        with self.assertRaises(ValueError):
            se.validate("0 2")


def stripes():
    """7×5 image: black left part, a white column at x=3, black right part."""
    img = np.zeros((5, 7, 3), np.uint8)
    img[:, 3] = 255
    return img


class TestMagicWand(unittest.TestCase):
    def test_local_mode_stops_at_other_values(self):
        mask = segmentation.magic_wand_selection(stripes(), 0, 0, 10, 10)
        self.assertEqual(mask.sum(), 15)
        self.assertFalse(mask[:, 3:].any())

    def test_global_mode_ignores_connectivity(self):
        mask = segmentation.magic_wand_selection(stripes(), 0, 0, 10, 10, global_mode=True)
        self.assertEqual(mask.sum(), 30)
        self.assertFalse(mask[:, 3].any())

    def test_range_is_an_interval_around_the_seed(self):
        img = np.zeros((1, 5, 3), np.uint8)
        img[0, :, :] = np.array([90, 95, 100, 120, 130])[:, None]
        # seed 100: accept [100 - 5, 100 + 20] = [95, 120]
        mask = segmentation.magic_wand_selection(img, 2, 0, range_below=5, range_above=20, global_mode=True)
        self.assertEqual(mask[0].tolist(), [False, True, True, True, False])

    def test_max_pixels(self):
        local = segmentation.magic_wand_selection(stripes(), 0, 0, 10, 10, max_pixels=4)
        self.assertEqual(local.sum(), 4)
        self.assertTrue(local[0, 0])
        glob = segmentation.magic_wand_selection(stripes(), 0, 0, 10, 10, max_pixels=4, global_mode=True)
        self.assertEqual(glob.sum(), 4)
        self.assertFalse(glob[:, 4:].any())  # nearest pixels to the click come first

    def test_connectivity(self):
        img = np.full((3, 3, 3), 255, np.uint8)
        img[0, 0] = img[1, 1] = img[2, 2] = 0  # diagonal line
        self.assertEqual(segmentation.magic_wand_selection(img, 0, 0, 0, 0, connectivity=segmentation.Connectivity.FOUR).sum(), 1)
        self.assertEqual(segmentation.magic_wand_selection(img, 0, 0, 0, 0, connectivity=segmentation.Connectivity.EIGHT).sum(), 3)

    def test_rgb_mode_compares_each_channel(self):
        img = np.zeros((1, 2, 3), np.uint8)
        img[0, 0] = (100, 100, 100)
        img[0, 1] = (70, 100, 130)  # same average, different channels
        self.assertEqual(segmentation.magic_wand_selection(img, 0, 0, 5, 5, segmentation.CompareMode.GRAY, global_mode=True).sum(), 2)
        self.assertEqual(segmentation.magic_wand_selection(img, 0, 0, 5, 5, segmentation.CompareMode.RGB, global_mode=True).sum(), 1)


class TestSelectionMorphology(unittest.TestCase):
    def test_dilate_and_erode(self):
        point = np.zeros((7, 7), bool)
        point[3, 3] = True
        self.assertEqual(segmentation.dilate_selection(point, 1, segmentation.StructuringElementShape.SQUARE).sum(), 9)
        self.assertEqual(segmentation.dilate_selection(point, 1, segmentation.StructuringElementShape.CROSS).sum(), 5)
        self.assertEqual(segmentation.dilate_selection(point, 2, segmentation.StructuringElementShape.DISK).sum(), 13)
        self.assertTrue(np.array_equal(segmentation.erode_selection(segmentation.dilate_selection(point, 2), 2), point))

    def test_erosion_shrinks_from_the_image_border(self):
        full = np.ones((5, 5), bool)
        self.assertEqual(segmentation.erode_selection(full, 1).sum(), 9)


class TestWatershed(unittest.TestCase):
    def test_two_flat_areas_give_two_regions(self):
        img = np.zeros((20, 30, 3), np.uint8)
        img[:, 15:] = 200
        labels, regions = segmentation.watershed_segmentation(img, smoothing_passes=1, marker_percent=50, min_marker_size=5)
        self.assertEqual(regions, 2)
        self.assertNotEqual(labels[10, 2], labels[10, 27])
        self.assertTrue((labels[:, 13:17] == segmentation.WATERSHED_LINE).any())  # line along the edge
        out = segmentation.paint_regions_with_mean_colour(img, labels)
        self.assertEqual(tuple(out[10, 2]), (0, 0, 0))
        self.assertEqual(tuple(out[10, 27]), (200, 200, 200))


class TestMaskedPaste(unittest.TestCase):
    def test_only_masked_pixels_change(self):
        full = np.zeros((4, 4, 3), np.uint8)
        region = np.full((2, 2, 3), 255, np.uint8)
        mask = np.array([[True, False], [False, True]])
        out = paste_region_where_selected(full, region, (1, 1, 3, 3), mask)
        self.assertEqual(int(out[..., 0].sum()) // 255, 2)
        self.assertEqual(out[1, 1, 0], 255)
        self.assertEqual(out[1, 2, 0], 0)


RING_WEIGHTS = (128, 1, 2, 4, 8, 16, 32, 64)  # neighbours clockwise from the top-left corner


def sticking_neighbours(weight_sum):
    """Number of object neighbours if they form one run around the ring, else 0."""
    bits = [(weight_sum & w) != 0 for w in RING_WEIGHTS]
    runs = sum(1 for i in range(8) if bits[i] and not bits[i - 1])
    return sum(bits) if runs == 1 and sum(bits) < 8 else 0


def components_8(mask):
    """Number of 8-connected components."""
    seen = np.zeros_like(mask)
    count = 0
    for y, x in zip(*np.nonzero(mask), strict=True):
        if seen[y, x]:
            continue
        count += 1
        seen[y, x] = True
        queue = deque([(y, x)])
        while queue:
            cy, cx = queue.popleft()
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    ny, nx = cy + dy, cx + dx
                    if 0 <= ny < mask.shape[0] and 0 <= nx < mask.shape[1] and mask[ny, nx] and not seen[ny, nx]:
                        seen[ny, nx] = True
                        queue.append((ny, nx))
    return count


THINNING = {"Zhang-Suen": skeletonization.zhang_suen_thinning, "K3M": k3m.k3m_thinning, "KMM": kmm.kmm_thinning}


class TestSkeletonization(unittest.TestCase):
    def test_lookup_tables_follow_the_sticking_neighbours_rule(self):
        def sums_with(counts):
            return tuple(c for c in range(256) if sticking_neighbours(c) in counts)

        self.assertEqual(sorted(k3m.A0), list(sums_with(range(2, 8))))
        self.assertEqual(sorted(k3m.A1), list(sums_with({3})))
        self.assertEqual(sorted(k3m.A2), list(sums_with({3, 4})))
        self.assertEqual(sorted(k3m.A3), list(sums_with({3, 4, 5})))
        self.assertEqual(sorted(k3m.A4), list(sums_with({3, 4, 5, 6})))
        # phase 5 adds 7 neighbours only when the missing one is an edge neighbour (N, E, S, W)
        self.assertEqual(sorted(set(k3m.A5) - set(k3m.A4)), [191, 239, 251, 254])
        self.assertEqual(sorted(kmm.FOUR_SUMS), list(sums_with({2, 3, 4})))

    def test_thick_bar_becomes_a_thin_line(self):
        bar = np.zeros((11, 40), bool)
        bar[3:8, 5:35] = True  # 5 px thick
        for name, thinning in THINNING.items():
            with self.subTest(method=name):
                skeleton, _iterations = thinning(bar)
                self.assertFalse((skeleton & ~bar).any())            # only object pixels remain
                self.assertEqual(components_8(skeleton), 1)          # still one piece
                self.assertLessEqual(int(skeleton.sum(axis=0).max()), 2)  # at most 2 px per column
                self.assertGreater(int(skeleton.sum()), 20)          # the line spans the bar

    def test_fingerprint_is_thinned_in_under_a_second(self):
        from PIL import Image

        image = np.asarray(Image.open("biografix/resources/fingerprint.png").convert("RGB"))
        mask = skeletonization.object_mask(thresholding.sauvola_threshold(image, 25, 0.2, 128.0))
        for name, thinning in THINNING.items():
            with self.subTest(method=name):
                thinning(mask[:8, :8])  # load the compiled code before timing
                started = time.perf_counter()
                skeleton, _iterations = thinning(mask)
                self.assertLess(time.perf_counter() - started, 1.0)
                blocks_2x2 = skeleton[:-1, :-1] & skeleton[1:, :-1] & skeleton[:-1, 1:] & skeleton[1:, 1:]
                self.assertLessEqual(int(blocks_2x2.sum()), 2)  # essentially one pixel wide


class TestSceneFile(unittest.TestCase):
    def test_roundtrip(self):
        objs = scene_from_dict(scene_to_dict([LineShape(x1=1, y1=2, x2=3, y2=4)]))
        self.assertIsInstance(objs[0], LineShape)
        self.assertEqual((objs[0].x1, objs[0].y2), (1, 4))

    def test_polygons_only_file(self):
        objs = scene_from_dict({"polygons": [{"points": [[0, 0], [5, 0], [5, 5]], "closed": False}]})
        self.assertIsInstance(objs[0], PolygonShape)
        self.assertFalse(objs[0].closed)

    def test_invalid_file_is_rejected(self):
        for bad in ([], {"objects": [{"name": "no type"}]}, {"polygons": [{"points": [["a", 0]]}]}):
            with self.assertRaises(ValueError):
                scene_from_dict(bad)


if __name__ == "__main__":
    unittest.main()
