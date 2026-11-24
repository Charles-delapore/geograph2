import unittest

import numpy as np

from geograph2.ir import HabitatEdge, HabitatGraph, HabitatNode
from geograph2.rsp import (
    effective_resistance_matrix,
    least_cost_path_matrix,
    solve_rsp,
)


def _grid_graph(nrows: int, ncols: int, cost_fn=None) -> HabitatGraph:
    nodes = [
        HabitatNode(f"n_{r}_{c}", coords=(float(c), float(r)), area=1.0, quality=1.0)
        for r in range(nrows)
        for c in range(ncols)
    ]
    edges = []
    eid = 0
    for r in range(nrows):
        for c in range(ncols):
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                nr, nc = r + dr, c + dc
                if 0 <= nr < nrows and 0 <= nc < ncols:
                    eid += 1
                    cost = 1.0 if cost_fn is None else float(cost_fn(r, c, nr, nc))
                    edges.append(
                        HabitatEdge(
                            f"e_{eid}",
                            f"n_{r}_{c}",
                            f"n_{nr}_{nc}",
                            weight=1.0 / cost,
                            cost=cost,
                            probability=np.exp(-0.2 * cost),
                        )
                    )
    return HabitatGraph(nodes, edges)


class TestRSPSolver(unittest.TestCase):
    def test_large_beta_converges_to_lcp(self):
        # Grid with a high-cost barrier in the middle
        def cost_fn(r, c, nr, nc):
            if nr == 1 and nc == 1:
                return 5.0
            return 1.0

        g = _grid_graph(3, 3, cost_fn=cost_fn)
        lcp = least_cost_path_matrix(g)
        res_hi = solve_rsp(g, beta=15.0)
        # Expected cost and free-energy distance both approach LCP at high beta
        max_diff = np.max(np.abs(res_hi.expected_cost - lcp))
        self.assertLess(max_diff, 1e-3)
        self.assertTrue(np.all(res_hi.expected_cost >= lcp - 1e-9))

    def test_small_beta_commute_cost_proportional_to_effective_resistance(self):
        # On a uniform undirected graph (all costs = 1, weights = 1),
        # commute time C_bar(s,t) + C_bar(t,s) -> vol(G) * R_eff(s,t) as beta -> 0+
        g = _grid_graph(3, 3)
        res_lo = solve_rsp(g, beta=1e-5)
        commute = res_lo.expected_cost + res_lo.expected_cost.T
        r_eff = effective_resistance_matrix(g)
        vol = g.to_sparse_matrix("weight").sum()  # sum of degrees = 2 * |E_undir|
        np.testing.assert_allclose(commute, vol * r_eff, rtol=1e-3, atol=1e-3)

    def test_monotonic_expected_cost_in_beta(self):
        g = _grid_graph(3, 3)
        c_low = solve_rsp(g, beta=0.1).expected_cost
        c_mid = solve_rsp(g, beta=1.0).expected_cost
        c_high = solve_rsp(g, beta=5.0).expected_cost
        # Higher beta -> less randomness -> strictly lower or equal expected cost
        self.assertTrue(np.all(c_low >= c_mid - 1e-9))
        self.assertTrue(np.all(c_mid >= c_high - 1e-9))


if __name__ == "__main__":
    unittest.main()
