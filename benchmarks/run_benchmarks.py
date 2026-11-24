"""Reproducible offline performance benchmark suite for GeoGraph 2.0 (D1-P7)."""

from __future__ import annotations

import json
import time
from typing import Any

import numpy as np

from geograph2.algebra import RasterLayer, SpatialExtent
from geograph2.connectivity import compute_iic, compute_pc, compute_protconn
from geograph2.dggs import SphericalHexDGGS
from geograph2.embedding import generate_synthetic_tessera_cube
from geograph2.ir import HabitatEdge, HabitatGraph, HabitatNode
from geograph2.manifold import cosine_resistance_field, fit_few_shot_suitability
from geograph2.rsp import build_reference_transition, solve_rsp_matrices
from geograph2.rsp_adjoint import rsp_adjoint_sensitivity


def _make_grid_graph(side: int) -> HabitatGraph:
    nodes = [
        HabitatNode(f"g_{r}_{c}", coords=(float(c), float(r)), area=1.0, quality=1.0)
        for r in range(side)
        for c in range(side)
    ]
    edges = []
    eid = 0
    for r in range(side):
        for c in range(side):
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                nr, nc = r + dr, c + dc
                if 0 <= nr < side and 0 <= nc < side:
                    eid += 1
                    edges.append(
                        HabitatEdge(f"e_{eid}", f"g_{r}_{c}", f"g_{nr}_{nc}", weight=1.0, cost=1.0, probability=0.8)
                    )
    return HabitatGraph(nodes, edges)


def _make_pa_network(n_nodes: int, seed: int = 42) -> HabitatGraph:
    rng = np.random.default_rng(seed)
    nodes = [
        HabitatNode(
            f"pa_{i}",
            coords=(float(rng.uniform(0, 100)), float(rng.uniform(0, 100))),
            area=float(rng.uniform(5.0, 30.0)),
            protected=(i % 2 == 0),
        )
        for i in range(n_nodes)
    ]
    edges = []
    eid = 0
    for i in range(n_nodes):
        for j in range(i + 1, n_nodes):
            dx = nodes[i].coords[0] - nodes[j].coords[0]
            dy = nodes[i].coords[1] - nodes[j].coords[1]
            d = float(np.hypot(dx, dy))
            if d < 35.0:
                p = float(np.exp(-d / 20.0))
                eid += 1
                edges.append(HabitatEdge(f"e_{eid}", nodes[i].node_id, nodes[j].node_id, cost=d, probability=p))
                eid += 1
                edges.append(HabitatEdge(f"e_{eid}", nodes[j].node_id, nodes[i].node_id, cost=d, probability=p))
    return HabitatGraph(nodes, edges)


def run_all_benchmarks() -> dict[str, Any]:
    """Execute all offline benchmarks and return timing + accuracy metrics."""
    results: dict[str, Any] = {}

    # 1. RSP + Adjoint benchmarks on 10x10 and 20x20 grids
    rsp_bench: dict[str, Any] = {}
    for side in (10, 20):
        g = _make_grid_graph(side)
        P, C = build_reference_transition(g)
        t0 = time.perf_counter()
        solve_rsp_matrices(P, C, beta=0.5)
        t_rsp_ms = (time.perf_counter() - t0) * 1000.0

        t1 = time.perf_counter()
        rsp_adjoint_sensitivity(P, C, beta=0.5, alpha=0.2, q=np.ones(g.num_nodes))
        t_adj_ms = (time.perf_counter() - t1) * 1000.0
        rsp_bench[f"grid_{side}x{side}_n{g.num_nodes}"] = {
            "rsp_ms": round(t_rsp_ms, 2),
            "rsp_adjoint_ms": round(t_adj_ms, 2),
        }
    results["rsp"] = rsp_bench

    # 2. Connectivity (PC / IIC / ProtConn) on 30-node PA network
    pa_g = _make_pa_network(30)
    t0 = time.perf_counter()
    pc_res = compute_pc(pa_g, landscape_area=1000.0)
    t_pc_ms = (time.perf_counter() - t0) * 1000.0

    t1 = time.perf_counter()
    iic_res = compute_iic(pa_g, landscape_area=1000.0)
    t_iic_ms = (time.perf_counter() - t1) * 1000.0

    t2 = time.perf_counter()
    prot_res = compute_protconn(pa_g, region_area=1000.0)
    t_prot_ms = (time.perf_counter() - t2) * 1000.0

    results["connectivity_30_nodes"] = {
        "pc_full_ranking_ms": round(t_pc_ms, 2),
        "iic_full_ranking_ms": round(t_iic_ms, 2),
        "protconn_ms": round(t_prot_ms, 2),
        "pc": round(pc_res.value, 6),
        "iic": round(iic_res.value, 6),
        "protconn_pct": round(prot_res.protconn_pct, 2),
    }

    # 3. Spherical Hex DGGS
    dggs = SphericalHexDGGS(hex_radius_km=25.0, resolution=6)
    t0 = time.perf_counter()
    hex_cells = dggs.generate_cells((0.0, 55.0, 10.0, 65.0))
    t_cells_ms = (time.perf_counter() - t0) * 1000.0
    results["spherical_hex_dggs"] = {
        "n_cells": len(hex_cells),
        "build_ms": round(t_cells_ms, 2),
        "max_area_distortion_pct": round(dggs.area_distortion_pct(hex_cells), 4),
    }

    # 4. TESSERA 128D manifold (64x64x128)
    cube, true_suit, _ = generate_synthetic_tessera_cube(64, 64, dim=128, seed=42)
    coords = [(r, c) for r in (8, 24, 40, 56) for c in (8, 24, 40, 56)]
    targets = [float(true_suit[r, c]) for r, c in coords]
    t0 = time.perf_counter()
    model = fit_few_shot_suitability(cube, coords, targets, l2_reg=0.2)
    _ = model.predict(cube)
    t_fit_ms = (time.perf_counter() - t0) * 1000.0

    t1 = time.perf_counter()
    _ = cosine_resistance_field(cube, cube.embeddings[16, 16])
    t_cos_ms = (time.perf_counter() - t1) * 1000.0
    results["tessera_64x64x128"] = {
        "few_shot_fit_and_predict_ms": round(t_fit_ms, 2),
        "cosine_resistance_ms": round(t_cos_ms, 2),
    }

    # 5. Lazy spatial algebra chunked reduction (256x256)
    ext = SpatialExtent(0.0, 0.0, 256.0, 256.0, pixel_size=1.0)
    a = RasterLayer(np.ones((256, 256)), ext)
    b = RasterLayer(np.full((256, 256), 2.0), ext)
    t0 = time.perf_counter()
    val = (a * b + 1.0).sum(chunk_size=64)
    t_alg_ms = (time.perf_counter() - t0) * 1000.0
    results["lazy_algebra_256x256"] = {
        "chunked_sum_ms": round(t_alg_ms, 2),
        "sum_value": val,
    }

    return results


if __name__ == "__main__":
    print(json.dumps(run_all_benchmarks(), indent=2))
