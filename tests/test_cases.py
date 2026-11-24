import tempfile
import unittest
from pathlib import Path

from geograph2.cases import (
    run_case_protected_area_30x30,
    run_case_urban_green_connectivity,
)


class TestOfflineCaseStudies(unittest.TestCase):
    def test_case1_protected_area_30x30_end_to_end(self):
        with tempfile.TemporaryDirectory() as tmp:
            res = run_case_protected_area_30x30(output_dir=tmp)
            self.assertTrue(res["prot_compliant"])
            self.assertGreater(res["protconn_pct"], 25.0)
            self.assertEqual(len(res["exported_files"]), 3)
            for fp in res["exported_files"]:
                self.assertTrue(Path(fp).exists())

    def test_case2_urban_green_connectivity_end_to_end(self):
        with tempfile.TemporaryDirectory() as tmp:
            res = run_case_urban_green_connectivity(output_dir=tmp)
            self.assertEqual(res["stratified_25d_nodes"], 12)
            self.assertGreater(res["tessera_vs_hard_rmse_reduction_pct"], 0.0)
            self.assertEqual(len(res["deca_trajectory"]), 3)
            self.assertGreater(res["deca_trajectory"][-1]["deca_cumulative"], 0.0)
            self.assertTrue((Path(tmp) / "case2_urban_hetero.json").exists())


if __name__ == "__main__":
    unittest.main()
