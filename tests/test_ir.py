import unittest
from dataclasses import FrozenInstanceError

import numpy as np

from geograph2.ir import (
    HabitatEdge,
    HabitatGraph,
    HabitatNode,
    NodeKind,
    SpatialMetadata,
)


class TestHabitatGraphIR(unittest.TestCase):
    def _sample_graph(self) -> HabitatGraph:
        nodes = [
            HabitatNode(
                node_id="P1",
                kind=NodeKind.POLYGON,
                coords=(10.0, 20.0),
                area=50.0,
                quality=0.9,
                habitat_type="forest",
                protected=True,
                geometry_wkt="POLYGON ((0 0, 10 0, 10 5, 0 5, 0 0))",
                attributes={"elevation_m": 450},
            ),
            HabitatNode(
                node_id="P2",
                kind=NodeKind.POINT,
                coords=(25.0, 20.0),
                area=25.0,
                quality=0.7,
                habitat_type="wetland",
            ),
            HabitatNode(
                node_id="V3",
                kind=NodeKind.VOXEL,
                coords=(25.0, 20.0, 15.0),
                area=10.0,
                quality=0.8,
                habitat_type="canopy",
                z_layer=2,
            ),
        ]
        edges = [
            HabitatEdge("E1", "P1", "P2", weight=0.8, cost=2.0, probability=0.6),
            HabitatEdge(
                "E2",
                "P1",
                "P2",
                weight=0.5,
                cost=1.2,
                probability=0.85,
                edge_class="inter",
                process="complementation",
            ),
            HabitatEdge("E3", "P2", "V3", weight=0.4, cost=3.0, probability=0.4),
        ]
        meta = SpatialMetadata(
            crs="EPSG:32632",
            bbox=(0.0, 0.0, 30.0, 30.0),
            resolution=(10.0, 10.0),
            vertical_datum="EGM2008",
        )
        return HabitatGraph(nodes, edges, meta)

    def test_immutability(self):
        g = self._sample_graph()
        with self.assertRaises(FrozenInstanceError):
            g.get_node("P1").area = 999.0  # type: ignore[misc]
        with self.assertRaises(TypeError):
            g.get_node("P1").attributes["elevation_m"] = 999  # type: ignore[index]
        with self.assertRaises(TypeError):
            g.nodes["P99"] = HabitatNode("P99")  # type: ignore[index]

    def test_multiedges_and_sparse_reduction(self):
        g = self._sample_graph()
        self.assertEqual(g.num_nodes, 3)
        self.assertEqual(g.num_edges, 3)
        self.assertEqual(len(g.parallel_edges("P1", "P2")), 2)

        w_sum = g.to_sparse_matrix("weight", reduce="sum").toarray()
        self.assertAlmostEqual(w_sum[0, 1], 1.3)

        c_min = g.to_sparse_matrix("cost", reduce="min").toarray()
        self.assertAlmostEqual(c_min[0, 1], 1.2)

        p_max = g.to_sparse_matrix("probability", reduce="max").toarray()
        self.assertAlmostEqual(p_max[0, 1], 0.85)

    def test_functional_updates_preserve_original(self):
        g = self._sample_graph()
        g2 = g.without_node("P2")
        self.assertEqual(g.num_nodes, 3)
        self.assertEqual(g2.num_nodes, 2)
        self.assertEqual(g2.num_edges, 0)

        g3 = g.without_edge("E1")
        self.assertEqual(g.num_edges, 3)
        self.assertEqual(g3.num_edges, 2)

        sub = g.subgraph_by_habitat(["forest", "wetland"])
        self.assertEqual(sub.node_ids, ("P1", "P2"))
        self.assertEqual(sub.num_edges, 2)

    def test_validation_errors(self):
        with self.assertRaises(ValueError):
            HabitatNode("bad", area=-1.0)
        with self.assertRaises(ValueError):
            HabitatEdge("e", "a", "b", probability=1.5)
        with self.assertRaises(ValueError):
            HabitatGraph([HabitatNode("a")], [HabitatEdge("e", "a", "missing")])
        with self.assertRaises(ValueError):
            HabitatGraph([HabitatNode("a"), HabitatNode("a")])

    def test_node_arrays(self):
        g = self._sample_graph()
        np.testing.assert_allclose(g.node_areas(), [50.0, 25.0, 10.0])
        np.testing.assert_allclose(g.node_qualities(), [0.9, 0.7, 0.8])


if __name__ == "__main__":
    unittest.main()
