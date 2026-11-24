import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

from geograph2.cli import main
from geograph2.io import save_graphjson
from geograph2.ir import HabitatEdge, HabitatGraph, HabitatNode


class TestCLI(unittest.TestCase):
    def _run_cli(self, argv):
        buf = io.StringIO()
        with redirect_stdout(buf):
            code = main(argv)
        self.assertEqual(code, 0)
        return json.loads(buf.getvalue())

    def test_cli_dggs_audit_rsp_and_cases(self):
        dggs_out = self._run_cli(["dggs", "--bbox", "0", "60", "3", "62", "--radius-km", "25"])
        self.assertGreater(dggs_out["num_nodes"], 3)
        self.assertLess(dggs_out["max_area_distortion_pct"], 0.5)

        c1 = self._run_cli(["case", "protected-area-30x30"])
        self.assertTrue(c1["prot_compliant"])
        self.assertGreater(c1["protconn_pct"], 20.0)

        c2 = self._run_cli(["case", "urban-green-connectivity"])
        self.assertGreater(c2["tessera_vs_hard_rmse_reduction_pct"], 0.0)

        gold = self._run_cli(["verify-golden"])
        self.assertLess(gold["analytical_golden_connectivity"]["pc_abs_err"], 1e-12)

        g = HabitatGraph(
            [HabitatNode("A", area=20.0, protected=True), HabitatNode("B", area=20.0, protected=True)],
            [
                HabitatEdge("e1", "A", "B", weight=1.0, cost=1.0, probability=0.7),
                HabitatEdge("e2", "B", "A", weight=1.0, cost=1.0, probability=0.7),
            ],
        )
        with tempfile.TemporaryDirectory() as tmp:
            gp = save_graphjson(g, Path(tmp) / "g.json")
            audit_out = self._run_cli(["audit", str(gp), "--region-area", "100", "--output-dir", tmp])
            self.assertAlmostEqual(audit_out["protconn"]["prot_pct"], 40.0)
            self.assertTrue((Path(tmp) / "gbf_30x30_audit.md").exists())

            rsp_out = self._run_cli(["rsp", str(gp), "--beta", "1.0"])
            self.assertGreater(rsp_out["connectivity_K"], 0.0)


if __name__ == "__main__":
    unittest.main()
