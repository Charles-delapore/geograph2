import unittest

import numpy as np

from geograph2.ir import NodeKind
from geograph2.layers3d import build_stratified_25d_graph, build_voxel_3d_graph


class TestLayers3D(unittest.TestCase):
    def test_stratified_25d_asymmetric_vertical_weights(self):
        understory = np.ones((2, 2), dtype=np.float64)
        canopy = np.full((2, 2), 2.0, dtype=np.float64)
        g = build_stratified_25d_graph(
            strata_resistance={"understory": understory, "canopy": canopy},
            stratum_order=["understory", "canopy"],
            pixel_size=10.0,
            stratum_heights_m={"understory": 2.0, "canopy": 12.0},
            vertical_cost_factors={
                ("understory", "canopy"): 2.5,
                ("canopy", "understory"): 0.8,
            },
            connectivity=4,
        )
        self.assertEqual(g.num_nodes, 8)
        # Check asymmetric vertical edge cost between understory_r0_c0 and canopy_r0_c0
        up_edges = g.parallel_edges("understory_r0_c0", "canopy_r0_c0")
        down_edges = g.parallel_edges("canopy_r0_c0", "understory_r0_c0")
        self.assertEqual(len(up_edges), 1)
        self.assertEqual(len(down_edges), 1)
        # dz=10, mean_res=1.5 -> up=10*1.5*2.5=37.5, down=10*1.5*0.8=12.0
        self.assertAlmostEqual(up_edges[0].cost, 37.5)
        self.assertAlmostEqual(down_edges[0].cost, 12.0)
        self.assertEqual(up_edges[0].edge_class, "inter")

    def test_voxel_3d_neighborhoods(self):
        vox = np.ones((2, 2, 2), dtype=np.float64)
        g6 = build_voxel_3d_graph(vox, voxel_size=(10.0, 10.0, 5.0), neighborhood=6)
        g26 = build_voxel_3d_graph(vox, voxel_size=(10.0, 10.0, 5.0), neighborhood=26)
        self.assertEqual(g6.num_nodes, 8)
        self.assertTrue(all(n.kind == NodeKind.VOXEL for n in g6.iter_nodes()))
        # In a 2x2x2 cube, each corner has 3 neighbors in 6-conn -> 24 directed edges
        self.assertEqual(g6.num_edges, 24)
        # In 26-conn, every voxel connects to all other 7 voxels -> 56 directed edges
        self.assertEqual(g26.num_edges, 56)


if __name__ == "__main__":
    unittest.main()
