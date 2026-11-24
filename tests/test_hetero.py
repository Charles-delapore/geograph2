import unittest

from geograph2.hetero import (
    EcologicalProcess,
    analyze_hetero_processes,
    build_multi_habitat_graph,
)


class TestMultiHabitatHeteroGraph(unittest.TestCase):
    def test_amphibian_complementation_supplementation_and_source_sink(self):
        nodes_spec = [
            {"node_id": "W1", "habitat_type": "water", "area_ha": 2.0, "quality": 0.9, "season_use": "breeding"},
            {"node_id": "W2", "habitat_type": "water", "area_ha": 1.0, "quality": 0.4, "season_use": "breeding"},
            {"node_id": "F1", "habitat_type": "woodland", "area_ha": 5.0, "quality": 0.85, "season_use": "overwintering"},
            {"node_id": "F2", "habitat_type": "woodland", "area_ha": 3.0, "quality": 0.75, "season_use": "foraging"},
        ]
        edges_spec = [
            # Source-sink from high-quality W1 to low-quality W2
            {
                "edge_id": "E_ss",
                "from_node": "W1",
                "to_node": "W2",
                "process": "source_sink",
                "movement_type": "dispersal",
                "cost": 1.2,
                "probability": 0.6,
            },
            # Complementation from breeding water W1 -> overwintering woodland F1
            {
                "edge_id": "E_comp",
                "from_node": "W1",
                "to_node": "F1",
                "process": "complementation",
                "movement_type": "seasonal_migration",
                "cost": 1.5,
                "probability": 0.75,
            },
            # Supplementation between woodlands F1 <-> F2
            {
                "edge_id": "E_supp",
                "from_node": "F1",
                "to_node": "F2",
                "process": "supplementation",
                "movement_type": "foraging",
                "cost": 1.0,
                "probability": 0.8,
            },
        ]
        g = build_multi_habitat_graph(nodes_spec, edges_spec)
        summary = analyze_hetero_processes(g)

        self.assertEqual(summary.process_edge_counts[EcologicalProcess.SOURCE_SINK.value], 1)
        self.assertEqual(summary.process_edge_counts[EcologicalProcess.COMPLEMENTATION.value], 1)
        self.assertEqual(summary.process_edge_counts[EcologicalProcess.SUPPLEMENTATION.value], 1)

        # W1 is net source (<0), W2 is net sink (>0)
        self.assertLess(summary.source_sink_net_balance["W1"], 0.0)
        self.assertGreater(summary.source_sink_net_balance["W2"], 0.0)
        self.assertEqual(summary.complementation_pairs[0][:2], ("W1", "F1"))
        self.assertEqual(summary.supplementation_redundancy["F1"], 1)

    def test_complementation_rejects_same_habitat_intra_edge(self):
        nodes_spec = [
            {"node_id": "W1", "habitat_type": "water", "area_ha": 1.0},
            {"node_id": "W2", "habitat_type": "water", "area_ha": 1.0},
        ]
        edges_spec = [
            {"edge_id": "bad", "from_node": "W1", "to_node": "W2", "process": "complementation"}
        ]
        with self.assertRaises(ValueError):
            build_multi_habitat_graph(nodes_spec, edges_spec)


if __name__ == "__main__":
    unittest.main()
