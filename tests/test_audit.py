import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from geograph2.audit import run_gbf_30x30_audit, verify_golden_baseline_files
from geograph2.ir import HabitatEdge, HabitatGraph, HabitatNode


class TestGBF30x30Audit(unittest.TestCase):
    def test_audit_export_markdown_and_json(self):
        g = HabitatGraph(
            nodes=[
                HabitatNode("PA_North", area=200.0, protected=True),
                HabitatNode("Corridor_Mid", area=50.0, protected=False),
                HabitatNode("PA_South", area=150.0, protected=True),
            ],
            edges=[
                HabitatEdge("e1", "PA_North", "Corridor_Mid", probability=0.7, cost=1.0),
                HabitatEdge("e2", "Corridor_Mid", "PA_North", probability=0.7, cost=1.0),
                HabitatEdge("e3", "Corridor_Mid", "PA_South", probability=0.6, cost=1.0),
                HabitatEdge("e4", "PA_South", "Corridor_Mid", probability=0.6, cost=1.0),
            ],
        )
        raster = np.array([[1, 1, 2, 2], [1, 2, 2, 3], [2, 2, 3, 3], [2, 3, 3, 3]], dtype=int)
        report = run_gbf_30x30_audit(
            g,
            region_area_ha=1000.0,
            site_name="Test Reserve",
            land_cover_raster=raster,
            cell_size_m=250.0,
        )
        self.assertTrue(report.prot_compliant)  # 350/1000 = 35% >= 30%
        self.assertTrue(report.protconn_compliant)

        with tempfile.TemporaryDirectory() as tmp:
            md_p, json_p = report.export(tmp, stem="reserve_audit")
            md_text = md_p.read_text(encoding="utf-8")
            parsed = json.loads(json_p.read_text(encoding="utf-8"))
            self.assertIn("GBF 30x30 Policy Audit Report: Test Reserve", md_text)
            self.assertIn("dPC_connector", md_text)
            self.assertEqual(parsed["site_name"], "Test Reserve")
            self.assertTrue(parsed["gbf_target_3"]["prot_compliant"])

    def test_golden_baseline_cross_verification(self):
        checks = verify_golden_baseline_files()
        ag = checks["analytical_golden_connectivity"]
        self.assertLess(ag["pc_abs_err"], 1e-12)
        self.assertLess(ag["eca_abs_err"], 1e-12)
        self.assertLess(ag["iic_abs_err"], 1e-12)
        self.assertLess(ag["protconn_abs_err"], 1e-12)
        self.assertLess(ag["dpc_decomp_max_err"], 1e-12)


if __name__ == "__main__":
    unittest.main()
