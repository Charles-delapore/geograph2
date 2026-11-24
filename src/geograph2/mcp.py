"""MCP Agent Skill Python dispatcher with geospatial contract validation (D1-P8)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import jsonschema

from geograph2.audit import run_gbf_30x30_audit
from geograph2.cases import (
    run_case_protected_area_30x30,
    run_case_urban_green_connectivity,
)
from geograph2.dggs import SphericalHexDGGS
from geograph2.io import load_graphjson
from geograph2.rsp import build_reference_transition
from geograph2.rsp_adjoint import rsp_adjoint_sensitivity


class GeoGraph2MCPDispatcher:
    """Validates MCP tool arguments against the declared JSON schema and executes tools."""

    def __init__(self, manifest_path: str | Path | None = None) -> None:
        if manifest_path is None:
            manifest_path = Path(__file__).resolve().parents[2] / "mcp_skill" / "geograph2_mcp_skill.json"
        self.manifest_path = Path(manifest_path)
        self.manifest: dict[str, Any] = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        self._tool_schemas: dict[str, dict[str, Any]] = {
            t["name"]: t["inputSchema"] for t in self.manifest["tools"]
        }

    @property
    def geo_contract(self) -> dict[str, Any]:
        return dict(self.manifest["geo_contract"])

    def list_tools(self) -> list[str]:
        return list(self._tool_schemas.keys())

    def invoke(self, tool_name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        if tool_name not in self._tool_schemas:
            raise KeyError(f"Unknown MCP tool: {tool_name}")
        jsonschema.validate(instance=arguments, schema=self._tool_schemas[tool_name])

        if tool_name == "geograph2.audit_30x30":
            g = load_graphjson(arguments["graph_json_path"])
            rep = run_gbf_30x30_audit(
                g,
                region_area_ha=float(arguments["region_area_ha"]),
                site_name=str(arguments.get("site_name", "Study Region")),
                target_prot_pct=float(arguments.get("target_prot_pct", 30.0)),
                target_protconn_pct=float(arguments.get("target_protconn_pct", 17.0)),
            )
            return rep.to_dict()

        if tool_name == "geograph2.solve_rsp_adjoint":
            g = load_graphjson(arguments["graph_json_path"])
            P, C = build_reference_transition(g)
            adj = rsp_adjoint_sensitivity(
                P,
                C,
                beta=float(arguments.get("beta", 1.0)),
                alpha=float(arguments.get("alpha", 0.25)),
                q=g.node_qualities(),
            )
            return {
                "connectivity_K": adj.connectivity_K,
                "mean_expected_cost": float(adj.rsp.expected_cost.mean()),
                "dK_dq": {nid: float(v) for nid, v in zip(g.node_ids, adj.dK_dq)},
            }

        if tool_name == "geograph2.build_spherical_dggs":
            dggs = SphericalHexDGGS(
                hex_radius_km=float(arguments.get("hex_radius_km", 25.0)),
                resolution=int(arguments.get("resolution", 6)),
            )
            bbox = tuple(float(x) for x in arguments["bbox_lonlat"])
            cells = dggs.generate_cells(bbox)  # type: ignore[arg-type]
            g = dggs.build_graph(bbox)  # type: ignore[arg-type]
            return {
                "crs": "EPSG:4326",
                "num_nodes": g.num_nodes,
                "num_edges": g.num_edges,
                "max_area_distortion_pct": dggs.area_distortion_pct(cells),
            }

        if tool_name == "geograph2.run_case_study":
            if arguments["case_name"] == "protected-area-30x30":
                return run_case_protected_area_30x30()
            return run_case_urban_green_connectivity()

        raise KeyError(tool_name)
