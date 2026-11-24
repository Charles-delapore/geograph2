import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarks.run_benchmarks import run_all_benchmarks


class TestBenchmarkSuite(unittest.TestCase):
    def test_benchmarks_run_cleanly(self):
        res = run_all_benchmarks()
        self.assertIn("rsp", res)
        self.assertIn("connectivity_30_nodes", res)
        self.assertIn("spherical_hex_dggs", res)
        self.assertLess(res["spherical_hex_dggs"]["max_area_distortion_pct"], 0.5)
        self.assertIn("tessera_64x64x128", res)
        self.assertAlmostEqual(res["lazy_algebra_256x256"]["sum_value"], 256.0 * 256.0 * 3.0)


if __name__ == "__main__":
    unittest.main()
