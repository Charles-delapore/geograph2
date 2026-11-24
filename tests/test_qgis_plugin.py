import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from geograph2.io import save_graphjson
from geograph2.ir import HabitatEdge, HabitatGraph, HabitatNode
from qgis_plugin.geograph2_provider import (
    GBF30x30AuditAlgorithm,
    GeoGraph2Provider,
    RSPAdjointSensitivityAlgorithm,
)


class TestQGISProcessingPlugin(unittest.TestCase):
    def test_provider_and_algorithms_offline(self):
        provider = GeoGraph2Provider()
        provider.loadAlgorithms()
        self.assertEqual(provider.id(), "geograph2")
        self.assertEqual(len(provider.algorithms()), 2)

        g = HabitatGraph(
            [
                HabitatNode("P1", area=30.0, quality=0.9, protected=True),
                HabitatNode("P2", area=20.0, quality=0.8, protected=True),
            ],
            [
                HabitatEdge("e1", "P1", "P2", weight=1.0, cost=1.0, probability=0.75),
                HabitatEdge("e2", "P2", "P1", weight=1.0, cost=1.0, probability=0.75),
            ],
        )
        with tempfile.TemporaryDirectory() as tmp:
            gp = save_graphjson(g, Path(tmp) / "input.json")

            audit_alg = GBF30x30AuditAlgorithm()
            audit_alg.initAlgorithm()
            self.assertEqual(len(audit_alg.parameterDefinitions()), 4)
            res = audit_alg.processAlgorithm(
                {
                    "INPUT_GRAPH": str(gp),
                    "REGION_AREA_HA": 100.0,
                    "SITE_NAME": "QGIS Test Reserve",
                    "OUTPUT_DIR": tmp,
                }
            )
            self.assertAlmostEqual(res["PROT_PCT"], 50.0)
            self.assertTrue(Path(res["MARKDOWN_REPORT"]).exists())

            rsp_alg = RSPAdjointSensitivityAlgorithm()
            rsp_alg.initAlgorithm()
            rsp_res = rsp_alg.processAlgorithm({"INPUT_GRAPH": str(gp), "BETA": 1.0, "ALPHA": 0.25})
            self.assertGreater(rsp_res["CONNECTIVITY_K"], 0.0)


if __name__ == "__main__":
    unittest.main()
