import unittest

import numpy as np
from shapely.geometry import box

from geograph2.algebra import (
    AlignmentMode,
    RasterLayer,
    SpatialExtent,
    VectorLayer,
    where,
)


class TestLazySpatialAlgebra(unittest.TestCase):
    def test_lazy_aoh_pipeline_with_vector_rasterization(self):
        ext = SpatialExtent(0.0, 0.0, 10.0, 10.0, pixel_size=1.0, crs="EPSG:3857")
        habitat = np.ones((10, 10), dtype=np.float64)
        habitat[:, :5] = 2.0  # code 2 on left half, code 1 on right half
        elev = np.tile(np.arange(10, dtype=np.float64) * 100.0, (10, 1))

        hab_layer = RasterLayer(habitat, ext, name="habitat")
        elev_layer = RasterLayer(elev, ext, name="elevation")
        range_poly = VectorLayer([box(2.0, 2.0, 8.0, 8.0)], ext, name="range")

        # Build expression tree: nothing should be evaluated yet!
        refined_hab = hab_layer.isin([2.0])
        refined_elev = (elev_layer >= 200.0) & (elev_layer <= 600.0)
        aoh = refined_hab * refined_elev * range_poly

        self.assertEqual(hab_layer.chunks_read, 0)
        self.assertEqual(elev_layer.chunks_read, 0)
        self.assertEqual(range_poly.chunks_rasterized, 0)

        # Trigger chunked evaluation (chunk_size=4 -> 3x3 = 9 chunks)
        area = aoh.sum(chunk_size=4)
        self.assertEqual(hab_layer.chunks_read, 9)
        self.assertEqual(range_poly.chunks_rasterized, 9)
        # x in [2, 5) -> cols 2,3,4 (3 cols); y in [2, 8) -> 6 rows => 18 pixels
        self.assertAlmostEqual(area, 18.0)

    def test_auto_intersection_and_union_alignment(self):
        ext_a = SpatialExtent(0.0, 0.0, 6.0, 6.0, pixel_size=1.0)
        ext_b = SpatialExtent(3.0, 0.0, 9.0, 6.0, pixel_size=1.0)
        la = RasterLayer(np.ones((6, 6)), ext_a)
        lb = RasterLayer(np.full((6, 6), 2.0), ext_b)

        # Multiplication infers INTERSECTION: x in [3, 6], y in [0, 6] -> 6x3 = 18 cells
        prod = la * lb
        self.assertEqual(prod.extent.shape, (6, 3))
        self.assertAlmostEqual(prod.sum(), 36.0)

        # Addition infers UNION: x in [0, 9], y in [0, 6] -> 6x9 = 54 cells
        # Left 3 cols = 1, middle 3 cols = 3, right 3 cols = 2 => 6*(3*1 + 3*3 + 3*2) = 108
        total = la + lb
        self.assertEqual(total.extent.shape, (6, 9))
        self.assertAlmostEqual(total.sum(), 108.0)

        # Explicit alignment override on addition -> INTERSECTION
        total_inter = (la + lb).with_alignment(AlignmentMode.INTERSECTION)
        self.assertEqual(total_inter.extent.shape, (6, 3))
        self.assertAlmostEqual(total_inter.sum(), 54.0)

    def test_where_expression(self):
        ext = SpatialExtent(0.0, 0.0, 4.0, 4.0, pixel_size=1.0)
        arr = np.arange(16, dtype=np.float64).reshape(4, 4)
        layer = RasterLayer(arr, ext)
        masked = where(layer >= 8.0, layer, 0.0)
        res = masked.to_array(chunk_size=2)
        expected = np.where(arr >= 8.0, arr, 0.0)
        np.testing.assert_allclose(res, expected)


if __name__ == "__main__":
    unittest.main()
