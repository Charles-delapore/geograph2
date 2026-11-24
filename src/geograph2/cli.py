"""Modern CLI for GeoGraph 2.0 (``geograph2``)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

from geograph2 import __version__
from geograph2.audit import run_gbf_30x30_audit, verify_golden_baseline_files
from geograph2.cases import (
    run_case_protected_area_30x30,
    run_case_urban_green_connectivity,
)
from geograph2.dggs import SphericalHexDGGS
from geograph2.io import load_graphjson, save_graphjson
from geograph2.rsp import build_reference_transition
from geograph2.rsp_adjoint import rsp_adjoint_sensitivity


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="geograph2",
        description="GeoGraph 2.0: Clean-Room Spatial Habitat Topology, RSP Adjoint, and GBF 30x30 Audit CLI",
    )
    parser.add_argument("--version", action="version", version=f"geograph2 {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    # audit
    p_audit = sub.add_parser("audit", help="Run GBF 30x30 policy audit on a GraphJSON file")
    p_audit.add_argument("graph_json", type=Path, help="Input GraphJSON file")
    p_audit.add_argument("--region-area", type=float, required=True, help="Total region area (ha)")
    p_audit.add_argument("--site-name", type=str, default="Study Region")
    p_audit.add_argument("--output-dir", type=Path, default=None, help="Directory to export MD + JSON report")

    # rsp
    p_rsp = sub.add_parser("rsp", help="Run RSP expected cost and adjoint sensitivity on a GraphJSON file")
    p_rsp.add_argument("graph_json", type=Path, help="Input GraphJSON file")
    p_rsp.add_argument("--beta", type=float, default=1.0, help="Inverse temperature beta (> 0)")
    p_rsp.add_argument("--alpha", type=float, default=0.25, help="Dispersal decay alpha")

    # dggs
    p_dggs = sub.add_parser("dggs", help="Build an equal-area spherical hexagonal DGGS graph")
    p_dggs.add_argument("--bbox", type=float, nargs=4, required=True, metavar=("MINLON", "MINLAT", "MAXLON", "MAXLAT"))
    p_dggs.add_argument("--radius-km", type=float, default=25.0)
    p_dggs.add_argument("--resolution", type=int, default=6)
    p_dggs.add_argument("--output", type=Path, default=None, help="Optional GraphJSON output path")

    # case
    p_case = sub.add_parser("case", help="Run an offline reproducible end-to-end case study")
    p_case.add_argument("name", choices=["protected-area-30x30", "urban-green-connectivity"])
    p_case.add_argument("--output-dir", type=Path, default=None)

    # verify-golden
    p_gold = sub.add_parser("verify-golden", help="Cross-verify golden baseline metrics")
    p_gold.add_argument("--pylandstats-csv", type=Path, default=None)
    p_gold.add_argument("--landscapemetrics-csv", type=Path, default=None)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "audit":
        g = load_graphjson(args.graph_json)
        rep = run_gbf_30x30_audit(g, region_area_ha=args.region_area, site_name=args.site_name)
        if args.output_dir is not None:
            rep.export(args.output_dir, stem="gbf_30x30_audit")
        print(rep.to_json())
        return 0

    if args.command == "rsp":
        g = load_graphjson(args.graph_json)
        P, C = build_reference_transition(g)
        adj = rsp_adjoint_sensitivity(P, C, beta=args.beta, alpha=args.alpha, q=g.node_qualities())
        out = {
            "beta": args.beta,
            "alpha": args.alpha,
            "connectivity_K": adj.connectivity_K,
            "mean_expected_cost": float(adj.rsp.expected_cost.mean()),
            "dK_dq": {nid: float(val) for nid, val in zip(g.node_ids, adj.dK_dq)},
        }
        print(json.dumps(out, indent=2))
        return 0

    if args.command == "dggs":
        dggs = SphericalHexDGGS(hex_radius_km=args.radius_km, resolution=args.resolution)
        bbox = tuple(args.bbox)
        cells = dggs.generate_cells(bbox)  # type: ignore[arg-type]
        g = dggs.build_graph(bbox)  # type: ignore[arg-type]
        if args.output is not None:
            save_graphjson(g, args.output)
        out = {
            "num_nodes": g.num_nodes,
            "num_edges": g.num_edges,
            "max_area_distortion_pct": dggs.area_distortion_pct(cells),
        }
        print(json.dumps(out, indent=2))
        return 0

    if args.command == "case":
        if args.name == "protected-area-30x30":
            res = run_case_protected_area_30x30(args.output_dir)
        else:
            res = run_case_urban_green_connectivity(args.output_dir)
        print(json.dumps(res, indent=2))
        return 0

    if args.command == "verify-golden":
        res = verify_golden_baseline_files(args.pylandstats_csv, args.landscapemetrics_csv)
        print(json.dumps(res, indent=2))
        return 0

    return 1


if __name__ == "__main__":
    sys.exit(main())
