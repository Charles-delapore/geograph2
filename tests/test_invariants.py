"""45 ecological and graph-theoretic mathematical invariant property tests (D1-P7)."""

from __future__ import annotations

import math
import unittest

import numpy as np
from scipy import sparse

from geograph2.connectivity import compute_iic, compute_pc, compute_protconn
from geograph2.dggs import SphericalHexDGGS, haversine_km
from geograph2.embedding import TesseraCube, generate_synthetic_tessera_cube
from geograph2.hetero import analyze_hetero_processes, build_multi_habitat_graph
from geograph2.ir import HabitatEdge, HabitatGraph, HabitatNode, NodeKind
from geograph2.manifold import cosine_resistance_field, semantic_gradient_boundaries
from geograph2.metrics import compute_landscape_metrics
from geograph2.rsp import (
    build_reference_transition,
    effective_resistance_matrix,
    least_cost_path_matrix,
    solve_rsp,
    solve_rsp_matrices,
)
from geograph2.rsp_adjoint import rsp_adjoint_sensitivity
from geograph2.temporal import track_patch_lineage


def _sample_undirected_graph(scale_area: float = 1.0, scale_cost: float = 1.0) -> HabitatGraph:
    nodes = [
        HabitatNode("n0", area=10.0 * scale_area, quality=0.9, protected=True),
        HabitatNode("n1", area=6.0 * scale_area, quality=0.7, protected=False),
        HabitatNode("n2", area=12.0 * scale_area, quality=0.8, protected=True),
        HabitatNode("n3", area=8.0 * scale_area, quality=0.6, protected=False),
        HabitatNode("n4", area=14.0 * scale_area, quality=0.95, protected=True),
    ]
    raw_edges = [
        ("n0", "n1", 1.0, 0.7),
        ("n1", "n2", 1.5, 0.6),
        ("n2", "n3", 1.2, 0.65),
        ("n3", "n4", 1.0, 0.75),
        ("n1", "n3", 2.0, 0.5),
        ("n0", "n4", 3.0, 0.35),
    ]
    edges = []
    eid = 0
    for u, v, c, p in raw_edges:
        sc = c * scale_cost
        eid += 1
        edges.append(HabitatEdge(f"e{eid}", u, v, weight=1.0 / sc, cost=sc, probability=p))
        eid += 1
        edges.append(HabitatEdge(f"e{eid}", v, u, weight=1.0 / sc, cost=sc, probability=p))
    return HabitatGraph(nodes, edges)


def _bridge_graph() -> HabitatGraph:
    nodes = [
        HabitatNode("L1", area=10.0, protected=True),
        HabitatNode("BR", area=4.0, protected=False),
        HabitatNode("R1", area=10.0, protected=True),
    ]
    edges = [
        HabitatEdge("e1", "L1", "BR", weight=1.0, cost=1.0, probability=0.6),
        HabitatEdge("e2", "BR", "L1", weight=1.0, cost=1.0, probability=0.6),
        HabitatEdge("e3", "BR", "R1", weight=1.0, cost=1.0, probability=0.6),
        HabitatEdge("e4", "R1", "BR", weight=1.0, cost=1.0, probability=0.6),
    ]
    return HabitatGraph(nodes, edges)


class TestEcologicalAndGraphInvariants(unittest.TestCase):
    # --- Connectivity Monotonicity & Scaling (1-15) ---

    def test_inv_01_removing_positive_edge_never_increases_pc(self):
        g = _sample_undirected_graph()
        base = compute_pc(g, landscape_area=100.0, compute_node_importance=False, compute_edge_importance=False).value
        for e in g.edges:
            val = compute_pc(g.without_edge(e.edge_id), landscape_area=100.0, compute_node_importance=False, compute_edge_importance=False).value
            self.assertLessEqual(val, base + 1e-12)

    def test_inv_02_removing_positive_edge_never_increases_iic(self):
        g = _sample_undirected_graph()
        base = compute_iic(g, landscape_area=100.0, compute_node_importance=False, compute_edge_importance=False).value
        for e in g.edges:
            val = compute_iic(g.without_edge(e.edge_id), landscape_area=100.0, compute_node_importance=False, compute_edge_importance=False).value
            self.assertLessEqual(val, base + 1e-12)

    def test_inv_03_removing_positive_edge_never_increases_eca(self):
        g = _sample_undirected_graph()
        base = compute_pc(g, landscape_area=100.0, compute_node_importance=False, compute_edge_importance=False).eca
        val = compute_pc(g.without_edge("e1"), landscape_area=100.0, compute_node_importance=False, compute_edge_importance=False).eca
        self.assertLessEqual(val, base + 1e-12)

    def test_inv_04_removing_positive_edge_never_increases_protconn(self):
        g = _sample_undirected_graph()
        base = compute_protconn(g, region_area=100.0, compute_dprotconn=False).protconn_pct
        for e in g.edges:
            val = compute_protconn(g.without_edge(e.edge_id), region_area=100.0, compute_dprotconn=False).protconn_pct
            self.assertLessEqual(val, base + 1e-12)

    def test_inv_05_removing_node_never_increases_eca(self):
        g = _sample_undirected_graph()
        base = compute_pc(g, landscape_area=100.0, compute_node_importance=False, compute_edge_importance=False).eca
        for nid in g.node_ids:
            val = compute_pc(g.without_node(nid), landscape_area=100.0, compute_node_importance=False, compute_edge_importance=False).eca
            self.assertLessEqual(val, base + 1e-12)

    def test_inv_06_adding_positive_edge_never_decreases_pc(self):
        g = _bridge_graph()
        base = compute_pc(g, landscape_area=50.0, compute_node_importance=False, compute_edge_importance=False).value
        g2 = g.with_nodes_and_edges(
            edges=(*g.edges, HabitatEdge("direct", "L1", "R1", probability=0.8, cost=0.5))
        )
        after = compute_pc(g2, landscape_area=50.0, compute_node_importance=False, compute_edge_importance=False).value
        self.assertGreaterEqual(after, base - 1e-12)

    def test_inv_07_adding_positive_edge_never_decreases_iic(self):
        g = _bridge_graph()
        base = compute_iic(g, landscape_area=50.0, compute_node_importance=False, compute_edge_importance=False).value
        g2 = g.with_nodes_and_edges(
            edges=(*g.edges, HabitatEdge("direct", "L1", "R1", probability=0.8, cost=0.5))
        )
        after = compute_iic(g2, landscape_area=50.0, compute_node_importance=False, compute_edge_importance=False).value
        self.assertGreaterEqual(after, base - 1e-12)

    def test_inv_08_area_scaling_scales_eca_linearly(self):
        g1 = _sample_undirected_graph(scale_area=1.0)
        g3 = _sample_undirected_graph(scale_area=3.5)
        e1 = compute_pc(g1, landscape_area=100.0, compute_node_importance=False, compute_edge_importance=False).eca
        e3 = compute_pc(g3, landscape_area=350.0, compute_node_importance=False, compute_edge_importance=False).eca
        self.assertAlmostEqual(e3, 3.5 * e1, places=9)

    def test_inv_09_simultaneous_area_and_landscape_scaling_preserves_pc(self):
        g1 = _sample_undirected_graph(scale_area=1.0)
        g5 = _sample_undirected_graph(scale_area=5.0)
        p1 = compute_pc(g1, landscape_area=100.0, compute_node_importance=False, compute_edge_importance=False).value
        p5 = compute_pc(g5, landscape_area=500.0, compute_node_importance=False, compute_edge_importance=False).value
        self.assertAlmostEqual(p1, p5, places=12)

    def test_inv_10_simultaneous_area_and_landscape_scaling_preserves_iic(self):
        g1 = _sample_undirected_graph(scale_area=1.0)
        g5 = _sample_undirected_graph(scale_area=5.0)
        i1 = compute_iic(g1, landscape_area=100.0, compute_node_importance=False, compute_edge_importance=False).value
        i5 = compute_iic(g5, landscape_area=500.0, compute_node_importance=False, compute_edge_importance=False).value
        self.assertAlmostEqual(i1, i5, places=12)

    def test_inv_11_pc_bounded_in_unit_interval(self):
        g = _sample_undirected_graph()
        pc = compute_pc(g, landscape_area=100.0, compute_node_importance=False, compute_edge_importance=False).value
        self.assertTrue(0.0 <= pc <= 1.0)

    def test_inv_12_iic_bounded_in_unit_interval(self):
        g = _sample_undirected_graph()
        iic = compute_iic(g, landscape_area=100.0, compute_node_importance=False, compute_edge_importance=False).value
        self.assertTrue(0.0 <= iic <= 1.0)

    def test_inv_13_eca_bounded_between_l2_and_l1_area_norms(self):
        g = _sample_undirected_graph()
        areas = g.node_areas()
        eca = compute_pc(g, landscape_area=100.0, compute_node_importance=False, compute_edge_importance=False).eca
        self.assertGreaterEqual(eca, math.sqrt(float(np.sum(areas ** 2))) - 1e-12)
        self.assertLessEqual(eca, float(np.sum(areas)) + 1e-12)

    def test_inv_14_isolated_graph_pc_equals_iic_equals_sum_sq_area(self):
        nodes = [HabitatNode("a", area=3.0), HabitatNode("b", area=4.0)]
        g = HabitatGraph(nodes, [])
        pc = compute_pc(g, landscape_area=10.0, compute_node_importance=False, compute_edge_importance=False).value
        iic = compute_iic(g, landscape_area=10.0, compute_node_importance=False, compute_edge_importance=False).value
        self.assertAlmostEqual(pc, 0.25, places=12)
        self.assertAlmostEqual(iic, 0.25, places=12)

    def test_inv_15_complete_unit_prob_graph_eca_equals_total_area(self):
        nodes = [HabitatNode("a", area=3.0), HabitatNode("b", area=7.0)]
        edges = [
            HabitatEdge("e1", "a", "b", probability=1.0, cost=0.1),
            HabitatEdge("e2", "b", "a", probability=1.0, cost=0.1),
        ]
        g = HabitatGraph(nodes, edges)
        eca = compute_pc(g, landscape_area=10.0, compute_node_importance=False, compute_edge_importance=False).eca
        self.assertAlmostEqual(eca, 10.0, places=12)

    # --- Decomposition & ProtConn Invariants (16-23) ---

    def test_inv_16_dpc_three_component_exact_partition(self):
        res = compute_pc(_sample_undirected_graph(), landscape_area=100.0)
        for r in res.node_importance:
            self.assertAlmostEqual(r.d_total, r.d_intra + r.d_flux + r.d_connector, places=9)

    def test_inv_17_diic_three_component_exact_partition(self):
        res = compute_iic(_sample_undirected_graph(), landscape_area=100.0)
        for r in res.node_importance:
            self.assertAlmostEqual(r.d_total, r.d_intra + r.d_flux + r.d_connector, places=9)

    def test_inv_18_leaf_node_has_zero_connector_contribution(self):
        res = compute_pc(_bridge_graph(), landscape_area=50.0)
        m = {r.node_id: r for r in res.node_importance}
        self.assertAlmostEqual(m["L1"].d_connector, 0.0, places=12)
        self.assertAlmostEqual(m["R1"].d_connector, 0.0, places=12)

    def test_inv_19_bridge_node_has_strictly_positive_connector(self):
        res = compute_pc(_bridge_graph(), landscape_area=50.0)
        m = {r.node_id: r for r in res.node_importance}
        self.assertGreater(m["BR"].d_connector, 0.0)

    def test_inv_20_protconn_bounded_by_within_and_prot(self):
        r = compute_protconn(_sample_undirected_graph(), region_area=100.0)
        self.assertLessEqual(r.protconn_within_pct, r.protconn_pct + 1e-12)
        self.assertLessEqual(r.protconn_pct, r.prot_pct + 1e-12)

    def test_inv_21_protconn_plus_protunconn_equals_prot(self):
        r = compute_protconn(_sample_undirected_graph(), region_area=100.0)
        self.assertAlmostEqual(r.protconn_pct + r.protunconn_pct, r.prot_pct, places=9)

    def test_inv_22_relconn_in_0_100(self):
        r = compute_protconn(_sample_undirected_graph(), region_area=100.0)
        self.assertTrue(0.0 <= r.relconn_pct <= 100.0 + 1e-9)

    def test_inv_23_protconn_fractions_sum_to_protconn(self):
        r = compute_protconn(_sample_undirected_graph(), region_area=100.0)
        self.assertAlmostEqual(
            r.protconn_within_pct + r.protconn_contig_pct + r.protconn_unprot_pct,
            r.protconn_pct,
            places=9,
        )

    # --- RSP, Resistance & Adjoint Sensitivity Invariants (24-34) ---

    def test_inv_24_rsp_diagonal_is_zero(self):
        c = solve_rsp(_sample_undirected_graph(), beta=1.0).expected_cost
        np.testing.assert_allclose(np.diag(c), 0.0, atol=1e-12)

    def test_inv_25_rsp_expected_cost_non_negative(self):
        c = solve_rsp(_sample_undirected_graph(), beta=0.5).expected_cost
        self.assertTrue(np.all(c >= -1e-12))

    def test_inv_26_rsp_expected_cost_dominates_lcp(self):
        g = _sample_undirected_graph()
        lcp = least_cost_path_matrix(g)
        c = solve_rsp(g, beta=1.2).expected_cost
        self.assertTrue(np.all(c >= lcp - 1e-9))

    def test_inv_27_rsp_monotonically_nonincreasing_in_beta(self):
        g = _sample_undirected_graph()
        c1 = solve_rsp(g, beta=0.2).expected_cost
        c2 = solve_rsp(g, beta=2.0).expected_cost
        self.assertTrue(np.all(c1 >= c2 - 1e-9))

    def test_inv_28_resistance_constant_scaling_order_and_homogeneity(self):
        # Scaling edge costs by k > 0 and inverse temperature by 1/k scales expected cost by exact k
        g1 = _sample_undirected_graph(scale_cost=1.0)
        gk = _sample_undirected_graph(scale_cost=4.0)
        c1 = solve_rsp(g1, beta=1.0).expected_cost
        ck = solve_rsp(gk, beta=0.25).expected_cost
        np.testing.assert_allclose(ck, 4.0 * c1, rtol=1e-9, atol=1e-9)
        # Rank order of off-diagonal pairwise distances is preserved
        self.assertTrue(np.array_equal(np.argsort(c1.ravel()), np.argsort(ck.ravel())))

    def test_inv_29_symmetric_graph_resistance_is_symmetric(self):
        r = effective_resistance_matrix(_sample_undirected_graph())
        np.testing.assert_allclose(r, r.T, atol=1e-12)

    def test_inv_30_effective_resistance_triangle_inequality(self):
        r = effective_resistance_matrix(_sample_undirected_graph())
        n = r.shape[0]
        for i in range(n):
            for j in range(n):
                for k in range(n):
                    self.assertLessEqual(r[i, k], r[i, j] + r[j, k] + 1e-10)

    def test_inv_31_symmetric_regular_graph_rsp_expected_cost_symmetric(self):
        # 4-cycle ring (symmetric and degree-regular)
        nodes = [HabitatNode(f"r{i}") for i in range(4)]
        edges = []
        eid = 0
        for i in range(4):
            j = (i + 1) % 4
            eid += 1
            edges.append(HabitatEdge(f"e{eid}", f"r{i}", f"r{j}", weight=1.0, cost=1.0))
            eid += 1
            edges.append(HabitatEdge(f"e{eid}", f"r{j}", f"r{i}", weight=1.0, cost=1.0))
        c = solve_rsp(HabitatGraph(nodes, edges), beta=0.8).expected_cost
        np.testing.assert_allclose(c, c.T, atol=1e-10)

    def test_inv_32_adjoint_dK_dq_nonnegative(self):
        P, C = build_reference_transition(_sample_undirected_graph())
        q = np.array([1.0, 2.0, 0.5, 1.5, 1.0])
        adj = rsp_adjoint_sensitivity(P, C, beta=1.0, alpha=0.3, q=q)
        self.assertTrue(np.all(adj.dK_dq >= 0.0))

    def test_inv_33_adjoint_dK_dC_nonpositive_for_exp_kernel(self):
        P, C = build_reference_transition(_sample_undirected_graph())
        q = np.ones(5)
        adj = rsp_adjoint_sensitivity(P, C, beta=1.0, alpha=0.3, q=q, kernel="exponential")
        # Increasing edge cost cannot increase exponential connectivity on undirected/positive graph
        self.assertLess(float(np.sum(adj.dK_dC)), 0.0)

    def test_inv_34_adjoint_euler_homogeneity_degree_two(self):
        P, C = build_reference_transition(_sample_undirected_graph())
        q = np.array([1.2, 0.7, 2.1, 0.9, 1.4])
        adj = rsp_adjoint_sensitivity(P, C, beta=0.9, alpha=0.2, q=q)
        # Since K(q) = q^T K_mat q is homogeneous of degree 2: q . dK/dq == 2 K
        self.assertAlmostEqual(float(np.dot(q, adj.dK_dq)), 2.0 * adj.connectivity_K, places=10)

    # --- FRAGSTATS Pattern Invariants (35-39) ---

    def test_inv_35_pland_sums_to_100_percent(self):
        raster = np.array([[1, 1, 2], [2, 3, 3], [1, 3, 2]], dtype=int)
        res = compute_landscape_metrics(raster, cell_size_m=10.0)
        self.assertAlmostEqual(sum(c.pland for c in res.class_metrics), 100.0, places=9)

    def test_inv_36_homogeneous_raster_zero_shdi_full_contag(self):
        res = compute_landscape_metrics(np.full((6, 6), 7, dtype=int), cell_size_m=20.0)
        self.assertAlmostEqual(res.shdi, 0.0, places=12)
        self.assertAlmostEqual(res.contag, 100.0, places=12)

    def test_inv_37_equiprobable_classes_maximize_shdi(self):
        raster = np.array([[1, 2], [3, 4]], dtype=int)
        res = compute_landscape_metrics(raster, cell_size_m=10.0)
        self.assertAlmostEqual(res.shdi, math.log(4.0), places=12)

    def test_inv_38_frac_always_in_1_to_2(self):
        rng = np.random.default_rng(99)
        raster = rng.integers(1, 4, size=(12, 12))
        res = compute_landscape_metrics(raster, cell_size_m=10.0)
        for p in res.patch_metrics:
            self.assertTrue(1.0 <= p.frac <= 2.0)

    def test_inv_39_pixel_size_scaling_laws_for_ta_and_enn(self):
        raster = np.array([[1, 2, 2], [2, 2, 2], [2, 2, 1]], dtype=int)
        r1 = compute_landscape_metrics(raster, cell_size_m=10.0)
        r3 = compute_landscape_metrics(raster, cell_size_m=30.0)
        self.assertAlmostEqual(r3.ta_ha, 9.0 * r1.ta_ha, places=9)
        self.assertAlmostEqual(r3.class_metrics[0].enn_mn_m, 3.0 * r1.class_metrics[0].enn_mn_m, places=9)

    # --- DGGS, Embedding, Hetero & Temporal Invariants (40-45) ---

    def test_inv_40_dggs_area_distortion_under_half_percent(self):
        dggs = SphericalHexDGGS(hex_radius_km=20.0, resolution=6)
        cells = dggs.generate_cells((0.0, 0.0, 2.0, 2.0)) + dggs.generate_cells((0.0, 70.0, 5.0, 73.0))
        self.assertLess(dggs.area_distortion_pct(cells), 0.5)

    def test_inv_41_haversine_symmetry_and_triangle_inequality(self):
        a, b, c = (0.0, 0.0), (10.0, 20.0), (25.0, 55.0)
        self.assertAlmostEqual(haversine_km(*a, *b), haversine_km(*b, *a), places=12)
        self.assertLessEqual(
            haversine_km(*a, *c), haversine_km(*a, *b) + haversine_km(*b, *c) + 1e-9
        )

    def test_inv_42_cosine_resistance_scale_invariance(self):
        cube, _, _ = generate_synthetic_tessera_cube(8, 8, dim=32, seed=5)
        cube_scaled = TesseraCube(embeddings=cube.embeddings * 7.5)
        ref = cube.embeddings[0, 0]
        r1 = cosine_resistance_field(cube, ref)
        r2 = cosine_resistance_field(cube_scaled, ref * 3.0)
        np.testing.assert_allclose(r1, r2, rtol=1e-6, atol=1e-6)

    def test_inv_43_uniform_embedding_zero_semantic_gradient(self):
        vec = np.ones((8, 8, 16), dtype=np.float32)
        grad = semantic_gradient_boundaries(TesseraCube(embeddings=vec))
        np.testing.assert_allclose(grad, 0.0, atol=1e-12)

    def test_inv_44_source_sink_net_balance_zero_sum_conservation(self):
        nodes = [
            {"node_id": "S1", "habitat_type": "forest", "area_ha": 10.0, "quality": 0.9},
            {"node_id": "K1", "habitat_type": "forest", "area_ha": 4.0, "quality": 0.3},
            {"node_id": "K2", "habitat_type": "forest", "area_ha": 3.0, "quality": 0.2},
        ]
        edges = [
            {"edge_id": "e1", "from_node": "S1", "to_node": "K1", "process": "source_sink", "probability": 0.5},
            {"edge_id": "e2", "from_node": "S1", "to_node": "K2", "process": "source_sink", "probability": 0.4},
        ]
        summary = analyze_hetero_processes(build_multi_habitat_graph(nodes, edges))
        self.assertAlmostEqual(sum(summary.source_sink_net_balance.values()), 0.0, places=12)

    def test_inv_45_temporal_lineage_accounts_for_every_parent_patch(self):
        g1 = HabitatGraph([
            HabitatNode("P1", NodeKind.POLYGON, (1.0, 1.0), area=4.0, geometry_wkt="POLYGON ((0 0, 2 0, 2 2, 0 2, 0 0))"),
            HabitatNode("P2", NodeKind.POLYGON, (6.0, 1.0), area=4.0, geometry_wkt="POLYGON ((5 0, 7 0, 7 2, 5 2, 5 0))"),
        ])
        g2 = HabitatGraph([
            HabitatNode("Q1", NodeKind.POLYGON, (1.0, 1.0), area=4.0, geometry_wkt="POLYGON ((0 0, 2 0, 2 2, 0 2, 0 0))"),
        ])
        events = track_patch_lineage([("t1", g1), ("t2", g2)])
        seen_parents = {pid for ev in events for pid in ev.parent_ids}
        self.assertEqual(seen_parents, {"P1", "P2"})


if __name__ == "__main__":
    unittest.main()
