"""End-to-end offline reproducible case studies for GeoGraph 2.0 (D1-P8).

Case 1: Protected Area GBF 30x30 Policy Audit (ProtConn, PC/IIC, RSP adjoint, FRAGSTATS).
Case 2: Urban Green Space Connectivity (Multi-habitat hetero, 2.5D canopy/understory,
        TESSERA 128D manifold vs hard classification, and 2015-2025 temporal dECA).
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np

from geograph2.audit import run_gbf_30x30_audit
from geograph2.embedding import generate_synthetic_tessera_cube
from geograph2.hetero import analyze_hetero_processes, build_multi_habitat_graph
from geograph2.io import save_graphjson
from geograph2.ir import HabitatEdge, HabitatGraph, HabitatNode, NodeKind, SpatialMetadata
from geograph2.layers3d import build_stratified_25d_graph
from geograph2.manifold import (
    compare_continuous_vs_hard_classification,
    cosine_resistance_field,
    fit_few_shot_suitability,
)
from geograph2.rsp import build_reference_transition
from geograph2.rsp_adjoint import rsp_adjoint_sensitivity
from geograph2.temporal import compute_temporal_deca_curve, track_patch_lineage


def run_case_protected_area_30x30(output_dir: str | Path | None = None) -> dict[str, Any]:
    """Case 1: Natural Protected Area Network GBF 30x30 Policy Audit."""
    nodes = [
        HabitatNode("PA_Alpine_N", NodeKind.POLYGON, (10.0, 40.0), area=140.0, quality=0.95, habitat_type="alpine_forest", protected=True),
        HabitatNode("PA_Valley_W", NodeKind.POLYGON, (5.0, 20.0), area=95.0, quality=0.88, habitat_type="riparian", protected=True),
        HabitatNode("PA_Ridge_E", NodeKind.POLYGON, (35.0, 25.0), area=110.0, quality=0.92, habitat_type="alpine_forest", protected=True),
        HabitatNode("PA_Lake_S", NodeKind.POLYGON, (20.0, 5.0), area=85.0, quality=0.90, habitat_type="wetland", protected=True),
        HabitatNode("Step_Bridge_C", NodeKind.POLYGON, (20.0, 22.0), area=30.0, quality=0.72, habitat_type="meadow", protected=False),
        HabitatNode("Step_Pass_NE", NodeKind.POLYGON, (24.0, 34.0), area=22.0, quality=0.68, habitat_type="shrub", protected=False),
        HabitatNode("Step_Corridor_SW", NodeKind.POLYGON, (12.0, 12.0), area=25.0, quality=0.70, habitat_type="riparian", protected=False),
        HabitatNode("Step_Foothill_SE", NodeKind.POLYGON, (30.0, 14.0), area=18.0, quality=0.65, habitat_type="meadow", protected=False),
    ]
    links = [
        ("PA_Alpine_N", "Step_Bridge_C", 1.8),
        ("PA_Alpine_N", "Step_Pass_NE", 1.4),
        ("Step_Pass_NE", "PA_Ridge_E", 1.5),
        ("PA_Valley_W", "Step_Bridge_C", 1.6),
        ("PA_Valley_W", "Step_Corridor_SW", 1.2),
        ("Step_Corridor_SW", "PA_Lake_S", 1.3),
        ("Step_Bridge_C", "PA_Ridge_E", 1.7),
        ("Step_Bridge_C", "PA_Lake_S", 1.9),
        ("PA_Ridge_E", "Step_Foothill_SE", 1.5),
        ("Step_Foothill_SE", "PA_Lake_S", 1.6),
    ]
    edges = []
    eid = 0
    for u, v, cost in links:
        p = math.exp(-0.35 * cost)
        eid += 1
        edges.append(HabitatEdge(f"c_{eid}", u, v, weight=1.0 / cost, cost=cost, probability=p))
        eid += 1
        edges.append(HabitatEdge(f"c_{eid}", v, u, weight=1.0 / cost, cost=cost, probability=p))

    graph = HabitatGraph(nodes, edges, SpatialMetadata(crs="EPSG:32632", epoch="2026"))

    # Synthetic 10x10 land-cover raster (1=forest, 2=wetland, 3=meadow, 4=matrix)
    lc = np.full((10, 10), 4, dtype=int)
    lc[0:4, 0:4] = 1
    lc[0:3, 6:10] = 1
    lc[6:10, 0:3] = 2
    lc[4:6, 4:7] = 3
    lc[7:10, 6:10] = 2

    region_area_ha = 1200.0
    audit = run_gbf_30x30_audit(
        graph,
        region_area_ha=region_area_ha,
        site_name="Alpine-Riparian Reserve Network",
        land_cover_raster=lc,
        cell_size_m=100.0,
    )

    P, C = build_reference_transition(graph)
    adj = rsp_adjoint_sensitivity(P, C, beta=0.8, alpha=0.25, q=graph.node_qualities())
    top_node_idx = int(np.argmax(adj.dK_dq))

    result: dict[str, Any] = {
        "case": "protected_area_30x30",
        "site_name": audit.site_name,
        "prot_pct": round(audit.protconn.prot_pct, 3),
        "protconn_pct": round(audit.protconn.protconn_pct, 3),
        "relconn_pct": round(audit.protconn.relconn_pct, 3),
        "prot_compliant": audit.prot_compliant,
        "protconn_compliant": audit.protconn_compliant,
        "pc": round(audit.pc.value, 6),
        "iic": round(audit.iic.value, 6),
        "eca_ha": round(audit.pc.eca, 3),
        "top_dpc_node": audit.pc.node_importance[0].node_id,
        "top_connector_node": max(audit.pc.node_importance, key=lambda r: r.d_connector).node_id,
        "rsp_connectivity_K": round(adj.connectivity_K, 4),
        "rsp_top_quality_sensitivity_node": graph.node_ids[top_node_idx],
    }

    if output_dir is not None:
        out = Path(output_dir)
        md_p, json_p = audit.export(out, stem="case1_protected_area_30x30")
        g_p = save_graphjson(graph, out / "case1_graph.json")
        result["exported_files"] = [str(md_p), str(json_p), str(g_p)]
    return result


def run_case_urban_green_connectivity(output_dir: str | Path | None = None) -> dict[str, Any]:
    """Case 2: Urban Green Infrastructure Connectivity (Hetero + 2.5D + TESSERA + Temporal)."""
    # 1. Multi-habitat urban park + wetland + street-tree network
    nodes_spec = [
        {"node_id": "Park_Central", "habitat_type": "urban_park", "area_ha": 45.0, "quality": 0.90, "season_use": "year_round"},
        {"node_id": "Wetland_River", "habitat_type": "water", "area_ha": 18.0, "quality": 0.85, "season_use": "breeding"},
        {"node_id": "Woodland_Hill", "habitat_type": "woodland", "area_ha": 32.0, "quality": 0.88, "season_use": "overwintering"},
        {"node_id": "Pocket_East", "habitat_type": "urban_park", "area_ha": 6.0, "quality": 0.45, "season_use": "foraging"},
    ]
    edges_spec = [
        {"edge_id": "u1", "from_node": "Park_Central", "to_node": "Pocket_East", "process": "source_sink", "movement_type": "dispersal", "cost": 1.8, "probability": 0.55},
        {"edge_id": "u2", "from_node": "Wetland_River", "to_node": "Woodland_Hill", "process": "complementation", "movement_type": "seasonal_migration", "cost": 1.4, "probability": 0.72},
        {"edge_id": "u3", "from_node": "Park_Central", "to_node": "Woodland_Hill", "process": "supplementation", "movement_type": "foraging", "cost": 1.2, "probability": 0.78},
    ]
    hetero_g = build_multi_habitat_graph(nodes_spec, edges_spec)
    hetero_summary = analyze_hetero_processes(hetero_g)

    # 2. 2.5D canopy/understory connectivity across an urban boulevard barrier
    understory = np.array([[1.0, 8.0, 1.0], [1.0, 8.0, 1.0]], dtype=np.float64)  # road barrier at col 1
    canopy = np.array([[1.2, 1.5, 1.2], [1.2, 1.5, 1.2]], dtype=np.float64)      # continuous tree canopy bridge
    g_25d = build_stratified_25d_graph(
        {"understory": understory, "canopy": canopy},
        stratum_order=["understory", "canopy"],
        pixel_size=20.0,
        stratum_heights_m={"understory": 1.5, "canopy": 12.0},
    )

    # 3. TESSERA 128D continuous embedding vs hard land-cover suitability
    cube, true_suit, hard_lc = generate_synthetic_tessera_cube(24, 24, dim=128, seed=2026)
    rng = np.random.default_rng(2026)
    samples = [(int(rng.integers(0, 24)), int(rng.integers(0, 24))) for _ in range(16)]
    targets = [float(true_suit[r, c]) for r, c in samples]
    model = fit_few_shot_suitability(cube, samples, targets, l2_reg=0.2)
    pred = model.predict(cube)
    comp = compare_continuous_vs_hard_classification(pred, hard_lc, true_suit)
    _ = cosine_resistance_field(cube, cube.embeddings[6, 6])

    # 4. Multi-epoch 2015 -> 2020 -> 2025 greenway restoration dECA trajectory
    g2015 = HabitatGraph(
        [
            HabitatNode("G_West", NodeKind.POLYGON, (2.0, 2.0), area=16.0, geometry_wkt="POLYGON ((0 0, 4 0, 4 4, 0 4, 0 0))"),
            HabitatNode("G_East", NodeKind.POLYGON, (12.0, 2.0), area=16.0, geometry_wkt="POLYGON ((10 0, 14 0, 14 4, 10 4, 10 0))"),
        ],
        [
            HabitatEdge("e1", "G_West", "G_East", probability=0.25, cost=3.0),
            HabitatEdge("e2", "G_East", "G_West", probability=0.25, cost=3.0),
        ],
    )
    g2020 = HabitatGraph(
        [
            HabitatNode("G_West_20", NodeKind.POLYGON, (2.0, 2.0), area=16.0, geometry_wkt="POLYGON ((0 0, 4 0, 4 4, 0 4, 0 0))"),
            HabitatNode("G_Step_20", NodeKind.POLYGON, (7.0, 2.0), area=6.0, geometry_wkt="POLYGON ((6 1, 8 1, 8 3, 6 3, 6 1))"),
            HabitatNode("G_East_20", NodeKind.POLYGON, (12.0, 2.0), area=16.0, geometry_wkt="POLYGON ((10 0, 14 0, 14 4, 10 4, 10 0))"),
        ],
        [
            HabitatEdge("e3", "G_West_20", "G_Step_20", probability=0.70, cost=1.0),
            HabitatEdge("e4", "G_Step_20", "G_West_20", probability=0.70, cost=1.0),
            HabitatEdge("e5", "G_Step_20", "G_East_20", probability=0.70, cost=1.0),
            HabitatEdge("e6", "G_East_20", "G_Step_20", probability=0.70, cost=1.0),
        ],
    )
    g2025 = HabitatGraph(
        [
            HabitatNode("G_Greenway_25", NodeKind.POLYGON, (7.0, 2.0), area=44.0, geometry_wkt="POLYGON ((0 0, 14 0, 14 4, 0 4, 0 0))"),
        ],
        [],
    )
    epochs = [("2015", g2015), ("2020", g2020), ("2025", g2025)]
    lineage = track_patch_lineage(epochs)
    deca_curve = compute_temporal_deca_curve(epochs, landscape_area=100.0)

    result: dict[str, Any] = {
        "case": "urban_green_connectivity",
        "hetero_process_counts": hetero_summary.process_edge_counts,
        "stratified_25d_nodes": g_25d.num_nodes,
        "stratified_25d_edges": g_25d.num_edges,
        "tessera_vs_hard_rmse_reduction_pct": round(comp["rmse_reduction_pct"], 2),
        "tessera_pearson": round(comp["pearson_continuous"], 4),
        "hard_lc_pearson": round(comp["pearson_hard_class"], 4),
        "lineage_events": [e.event_type.value for e in lineage],
        "deca_trajectory": [
            {"epoch": p.epoch, "eca": round(p.eca, 3), "deca_cumulative": round(p.deca_cumulative, 3)}
            for p in deca_curve
        ],
    }

    if output_dir is not None:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        save_graphjson(hetero_g, out / "case2_urban_hetero.json")
    return result
