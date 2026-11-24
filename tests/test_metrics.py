import math
import unittest

import numpy as np

from geograph2.metrics import compute_landscape_metrics


class TestLandscapeMetrics(unittest.TestCase):
    def test_exact_analytical_two_class_landscape(self):
        # 4x4 raster, cell_size=100m -> each pixel = 1 ha, TA = 16 ha
        # Class 1: top-left 2x2 block (4 px = 4 ha, PLAND = 25%) and bottom-right 1x1 (1 px = 1 ha)
        # Class 2: remaining 11 px (11 ha, PLAND = 68.75%)
        raster = np.full((4, 4), 2, dtype=int)
        raster[0:2, 0:2] = 1
        raster[3, 3] = 1

        res = compute_landscape_metrics(raster, cell_size_m=100.0, directions=8)
        self.assertAlmostEqual(res.ta_ha, 16.0)
        cls_map = {c.class_val: c for c in res.class_metrics}
        self.assertEqual(cls_map[1].np, 2)
        self.assertEqual(cls_map[2].np, 1)
        self.assertAlmostEqual(cls_map[1].ca_ha, 5.0)
        self.assertAlmostEqual(cls_map[1].pland, 31.25)
        self.assertAlmostEqual(cls_map[2].pland, 68.75)

        # Exact ENN between (1,1) of top-left patch and (3,3) of bottom-right patch:
        # distance = sqrt((3-1)^2 + (3-1)^2) * 100 = 200 * sqrt(2)
        self.assertAlmostEqual(cls_map[1].enn_mn_m, 200.0 * math.sqrt(2.0), places=5)

        # Exact SHDI = -(5/16 ln(5/16) + 11/16 ln(11/16))
        p1, p2 = 5.0 / 16.0, 11.0 / 16.0
        expected_shdi = -(p1 * math.log(p1) + p2 * math.log(p2))
        self.assertAlmostEqual(res.shdi, expected_shdi, places=9)

        # CONTAG must lie in [0, 100]
        self.assertTrue(0.0 <= res.contag <= 100.0)

    def test_single_class_landscape(self):
        raster = np.ones((5, 5), dtype=int)
        res = compute_landscape_metrics(raster, cell_size_m=10.0)
        self.assertAlmostEqual(res.shdi, 0.0)
        self.assertAlmostEqual(res.contag, 100.0)
        self.assertAlmostEqual(res.ed_m_per_ha, 0.0)
        self.assertEqual(res.np_total, 1)


if __name__ == "__main__":
    unittest.main()
