import math
import unittest

from geograph2.connectivity import (
    compute_deca,
    compute_iic,
    compute_pc,
    compute_protconn,
)
from geograph2.ir import HabitatEdge, HabitatGraph, HabitatNode


def _three_node_chain() -> HabitatGraph:
    # A (area=10, protected) --p=0.5--> B (area=5, unprotected stepping stone) --p=0.5--> C (area=10, protected)
    nodes = [
        HabitatNode("A", area=10.0, protected=True),
        HabitatNode("B", area=5.0, protected=False),
        HabitatNode("C", area=10.0, protected=True),
    ]
    edges = [
        HabitatEdge("e_ab", "A", "B", probability=0.5, cost=1.0),
        HabitatEdge("e_ba", "B", "A", probability=0.5, cost=1.0),
        HabitatEdge("e_bc", "B", "C", probability=0.5, cost=1.0),
        HabitatEdge("e_cb", "C", "B", probability=0.5, cost=1.0),
    ]
    return HabitatGraph(nodes, edges)


class TestConnectivityIndices(unittest.TestCase):
    def test_exact_pc_and_three_component_sum(self):
        g = _three_node_chain()
        al = 50.0
        res = compute_pc(g, landscape_area=al)

        # Analytical PC numerator:
        # diag: 10^2 + 5^2 + 10^2 = 225
        # A<->B: 2 * 10 * 5 * 0.5 = 50
        # B<->C: 2 * 5 * 10 * 0.5 = 50
        # A<->C (via B): 2 * 10 * 10 * 0.25 = 50
        # Total = 375 -> PC = 375 / 2500 = 0.15, ECA = sqrt(375)
        self.assertAlmostEqual(res.numerator, 375.0, places=9)
        self.assertAlmostEqual(res.value, 0.15, places=9)
        self.assertAlmostEqual(res.eca, math.sqrt(375.0), places=9)

        # Verify exact dPC = dPCintra + dPCflux + dPCconnector for every node
        imp_map = {r.node_id: r for r in res.node_importance}
        for r in res.node_importance:
            self.assertAlmostEqual(r.d_total, r.d_intra + r.d_flux + r.d_connector, places=9)

        # B is the unique stepping stone between A and C -> dPCconnector(B) = 50/375 * 100
        self.assertAlmostEqual(imp_map["B"].d_connector, 100.0 * 50.0 / 375.0, places=9)
        self.assertAlmostEqual(imp_map["A"].d_connector, 0.0, places=9)
        self.assertAlmostEqual(imp_map["C"].d_connector, 0.0, places=9)

    def test_exact_iic_and_deca(self):
        g = _three_node_chain()
        al = 50.0
        res = compute_iic(g, landscape_area=al)
        # Analytical IIC numerator:
        # diag: 225; A<->B (1 hop): 2*50/2 = 50; B<->C (1 hop): 50; A<->C (2 hops): 2*100/3 = 66.6666667
        expected_num = 225.0 + 50.0 + 50.0 + 200.0 / 3.0
        self.assertAlmostEqual(res.numerator, expected_num, places=9)
        self.assertAlmostEqual(res.value, expected_num / 2500.0, places=9)

        deca = compute_deca(10.0, 12.5)
        self.assertAlmostEqual(deca["dECA"], 2.5)
        self.assertAlmostEqual(deca["rdECA_pct"], 25.0)

    def test_protconn_with_unprotected_stepping_stone(self):
        g = _three_node_chain()
        # Region area = 50; protected = A(10) + C(10) = 20 -> Prot = 40%
        # With stepping stone B: p*_AC = 0.25 -> ECA_PA = sqrt(10^2 + 10^2 + 2*10*10*0.25) = sqrt(250)
        res = compute_protconn(g, region_area=50.0)
        self.assertAlmostEqual(res.prot_pct, 40.0, places=9)
        self.assertAlmostEqual(res.eca_protected, math.sqrt(250.0), places=9)
        self.assertAlmostEqual(res.protconn_pct, 100.0 * math.sqrt(250.0) / 50.0, places=9)
        self.assertAlmostEqual(res.protconn_within_pct, 100.0 * math.sqrt(200.0) / 50.0, places=9)
        self.assertAlmostEqual(res.protconn_contig_pct, 0.0, places=9)
        self.assertGreater(res.protconn_unprot_pct, 0.0)
        self.assertIn("A", res.node_dprotconn)
        self.assertIn("C", res.node_dprotconn)
        self.assertNotIn("B", res.node_dprotconn)


if __name__ == "__main__":
    unittest.main()
