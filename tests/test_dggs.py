import math
import unittest

from geograph2.dggs import (
    EARTH_AUTHALIC_RADIUS_KM,
    SphericalHexDGGS,
    corrected_cell_densities,
    haversine_km,
    healpix_cell_area_km2,
    high_latitude_area_correction,
)


class TestSphericalHexDGGS(unittest.TestCase):
    def test_area_distortion_under_half_percent_equator_to_high_latitude(self):
        dggs = SphericalHexDGGS(hex_radius_km=25.0, resolution=6)
        # Test low-latitude (equatorial) and high-latitude (65-72N) strips
        eq_cells = dggs.generate_cells((0.0, -2.0, 4.0, 2.0))
        hi_cells = dggs.generate_cells((10.0, 65.0, 18.0, 72.0))
        self.assertGreater(len(eq_cells), 10)
        self.assertGreater(len(hi_cells), 10)

        dist_eq = dggs.area_distortion_pct(eq_cells)
        dist_hi = dggs.area_distortion_pct(hi_cells)
        self.assertLess(dist_eq, 0.5)
        self.assertLess(dist_hi, 0.5)

    def test_hex_graph_topology_and_haversine(self):
        dggs = SphericalHexDGGS(hex_radius_km=30.0, resolution=5)
        g = dggs.build_graph((0.0, 45.0, 3.0, 47.0), dispersal_scale_km=80.0)
        self.assertGreater(g.num_nodes, 5)
        self.assertGreater(g.num_edges, 10)
        self.assertEqual(g.metadata.dggs_type, "equal_area_hex_r5")
        # Every edge must have positive great-circle distance and valid probability
        for e in g.edges:
            self.assertGreater(e.distance, 0.0)
            self.assertTrue(0.0 <= e.probability <= 1.0)

    def test_area_correction_helpers(self):
        self.assertAlmostEqual(high_latitude_area_correction(100.0, 60.0), 50.0, places=5)
        # HEALPix area at Nside=1 should be 4*pi*R^2 / 12
        expected_nside1 = (4.0 * math.pi * EARTH_AUTHALIC_RADIUS_KM ** 2) / 12.0
        self.assertAlmostEqual(healpix_cell_area_km2(1), expected_nside1, places=5)

        res = corrected_cell_densities([10.0, 20.0], [2.0, 4.0], total_land_cells=4)
        self.assertEqual(res["densities"], [5.0, 5.0])
        self.assertAlmostEqual(res["mean_density"], 2.5)
        self.assertAlmostEqual(haversine_km(0.0, 0.0, 0.0, 1.0), 111.195, delta=0.1)


if __name__ == "__main__":
    unittest.main()
