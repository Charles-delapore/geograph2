import unittest

from geograph2.ir import HabitatEdge, HabitatGraph, HabitatNode, NodeKind
from geograph2.temporal import (
    LineageEventType,
    compute_temporal_deca_curve,
    track_patch_lineage,
)


class TestTemporalLineageAndDECA(unittest.TestCase):
    def test_split_merge_extinction_and_deca_curve(self):
        # Epoch 2000: Large patch A [0,10]x[0,4] and small patch B [20,22]x[0,2]
        g2000 = HabitatGraph(
            nodes=[
                HabitatNode(
                    "A_2000",
                    NodeKind.POLYGON,
                    coords=(5.0, 2.0),
                    area=40.0,
                    geometry_wkt="POLYGON ((0 0, 10 0, 10 4, 0 4, 0 0))",
                ),
                HabitatNode(
                    "B_2000",
                    NodeKind.POLYGON,
                    coords=(21.0, 1.0),
                    area=4.0,
                    geometry_wkt="POLYGON ((20 0, 22 0, 22 2, 20 2, 20 0))",
                ),
            ],
            edges=[
                HabitatEdge("e1", "A_2000", "B_2000", probability=0.4, cost=1.0),
                HabitatEdge("e2", "B_2000", "A_2000", probability=0.4, cost=1.0),
            ],
        )

        # Epoch 2010: A splits into A1 [0,4]x[0,4] and A2 [6,10]x[0,4]; B goes extinct!
        g2010 = HabitatGraph(
            nodes=[
                HabitatNode(
                    "A1_2010",
                    NodeKind.POLYGON,
                    coords=(2.0, 2.0),
                    area=16.0,
                    geometry_wkt="POLYGON ((0 0, 4 0, 4 4, 0 4, 0 0))",
                ),
                HabitatNode(
                    "A2_2010",
                    NodeKind.POLYGON,
                    coords=(8.0, 2.0),
                    area=16.0,
                    geometry_wkt="POLYGON ((6 0, 10 0, 10 4, 6 4, 6 0))",
                ),
            ],
            edges=[
                HabitatEdge("e3", "A1_2010", "A2_2010", probability=0.5, cost=1.0),
                HabitatEdge("e4", "A2_2010", "A1_2010", probability=0.5, cost=1.0),
            ],
        )

        # Epoch 2020: Restoration merges A1 and A2 back into A_restored [0,10]x[0,4]
        g2020 = HabitatGraph(
            nodes=[
                HabitatNode(
                    "A_2020",
                    NodeKind.POLYGON,
                    coords=(5.0, 2.0),
                    area=40.0,
                    geometry_wkt="POLYGON ((0 0, 10 0, 10 4, 0 4, 0 0))",
                ),
            ],
            edges=[],
        )

        epochs = [("2000", g2000), ("2010", g2010), ("2020", g2020)]
        events = track_patch_lineage(epochs)
        ev_types = {(e.epoch_from, e.epoch_to, e.event_type) for e in events}

        self.assertIn(("2000", "2010", LineageEventType.SPLIT), ev_types)
        self.assertIn(("2000", "2010", LineageEventType.EXTINCTION), ev_types)
        self.assertIn(("2010", "2020", LineageEventType.MERGE), ev_types)

        curve = compute_temporal_deca_curve(epochs, landscape_area=100.0)
        self.assertEqual(len(curve), 3)
        # Fragmentation + extinction from 2000 -> 2010 reduces ECA
        self.assertLess(curve[1].deca_step, 0.0)
        # Restoration merge from 2010 -> 2020 increases ECA
        self.assertGreater(curve[2].deca_step, 0.0)


if __name__ == "__main__":
    unittest.main()
