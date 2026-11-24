import json
import tempfile
import unittest
from pathlib import Path

import pyarrow.parquet as pq

from geograph2.io import (
    load_geoparquet,
    load_graphjson,
    save_geoparquet,
    save_graphjson,
)
from geograph2.ir import (
    HabitatEdge,
    HabitatGraph,
    HabitatNode,
    NodeKind,
    SpatialMetadata,
)


class TestSerialization(unittest.TestCase):
    def _build_graph(self) -> HabitatGraph:
        nodes = [
            HabitatNode(
                "N1",
                NodeKind.POLYGON,
                (1.0, 2.0),
                area=12.5,
                quality=0.85,
                habitat_type="forest",
                protected=True,
                geometry_wkt="POLYGON ((0 0, 2 0, 2 2, 0 2, 0 0))",
                attributes={"biome": "temperate"},
            ),
            HabitatNode(
                "N2",
                NodeKind.VOXEL,
                (3.0, 4.0, 10.0),
                area=5.0,
                quality=0.6,
                habitat_type="canopy",
                z_layer=1,
            ),
        ]
        edges = [
            HabitatEdge(
                "E1",
                "N1",
                "N2",
                weight=0.75,
                cost=1.5,
                probability=0.8,
                distance=282.8,
                edge_class="inter",
                process="complementation",
                attributes={"season": "spring"},
            )
        ]
        meta = SpatialMetadata(
            crs="EPSG:4326",
            bbox=(0.0, 0.0, 5.0, 5.0),
            resolution=(1.0, 1.0),
            epoch="2026",
        )
        return HabitatGraph(nodes, edges, meta)

    def test_graphjson_roundtrip(self):
        g = self._build_graph()
        with tempfile.TemporaryDirectory() as tmp:
            p = save_graphjson(g, Path(tmp) / "graph.json")
            g2 = load_graphjson(p)
        self.assertEqual(g2.node_ids, g.node_ids)
        self.assertEqual(g2.metadata.crs, "EPSG:4326")
        self.assertEqual(g2.metadata.epoch, "2026")
        self.assertEqual(g2.get_node("N1").attributes["biome"], "temperate")
        self.assertEqual(g2.get_edge("E1").process, "complementation")
        self.assertAlmostEqual(g2.get_edge("E1").distance, 282.8)

    def test_geoparquet_roundtrip_and_geo_metadata(self):
        g = self._build_graph()
        with tempfile.TemporaryDirectory() as tmp:
            nodes_p, edges_p = save_geoparquet(g, tmp)
            self.assertTrue(nodes_p.exists())
            self.assertTrue(edges_p.exists())
            tbl = pq.read_table(nodes_p)
            geo = json.loads(tbl.schema.metadata[b"geo"].decode("utf-8"))
            self.assertEqual(geo["version"], "1.1.0")
            self.assertEqual(geo["primary_column"], "geometry")
            g2 = load_geoparquet(tmp)

        self.assertEqual(g2.num_nodes, 2)
        self.assertEqual(g2.num_edges, 1)
        self.assertEqual(g2.get_node("N2").z_layer, 1)
        self.assertEqual(g2.get_node("N1").protected, True)
        self.assertEqual(g2.get_edge("E1").attributes["season"], "spring")


if __name__ == "__main__":
    unittest.main()
