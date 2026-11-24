"""Multi-habitat heterogeneous graph modeling (Savary et al. 2024; Dunning et al. 1992).

Explicitly distinguishes directed ecological processes:
- ``source_sink``: population rescue from high-quality source patches to sink patches,
- ``supplementation``: intra-habitat resource supplementation across patches,
- ``complementation``: inter-habitat life-history complementation (e.g. amphibian
  breeding water -> overwintering woodland seasonal migration).
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Mapping, Sequence

from geograph2.ir import HabitatEdge, HabitatGraph, HabitatNode, NodeKind, SpatialMetadata


class EcologicalProcess(str, Enum):
    """Dunning et al. (1992) landscape ecological process categories."""

    SOURCE_SINK = "source_sink"
    SUPPLEMENTATION = "supplementation"
    COMPLEMENTATION = "complementation"


class EdgeClass(str, Enum):
    """Intra-habitat vs inter-habitat link classification (Graphab 3.0)."""

    INTRA = "intra"
    INTER = "inter"


@dataclass(frozen=True)
class HeteroProcessSummary:
    """Process-stratified connectivity summary for a multi-habitat graph."""

    process_edge_counts: dict[str, int]
    process_total_flux: dict[str, float]
    source_sink_net_balance: dict[str, float]
    complementation_pairs: tuple[tuple[str, str, float], ...]
    supplementation_redundancy: dict[str, int]


def build_multi_habitat_graph(
    nodes_spec: Sequence[Mapping[str, Any]],
    edges_spec: Sequence[Mapping[str, Any]],
    crs: str = "EPSG:3857",
) -> HabitatGraph:
    """Construct a validated multi-habitat HabitatGraph matching the Savary et al. schema."""
    valid_processes = {p.value for p in EcologicalProcess}
    nodes: list[HabitatNode] = []
    node_types: dict[str, str] = {}

    for row in nodes_spec:
        nid = str(row["node_id"])
        htype = str(row["habitat_type"])
        node_types[nid] = htype
        nodes.append(
            HabitatNode(
                node_id=nid,
                kind=NodeKind(row.get("kind", "polygon")),
                coords=tuple(row.get("coords", (0.0, 0.0))),
                area=float(row.get("area_ha", row.get("area", 1.0))),
                quality=float(row.get("quality", 1.0)),
                habitat_type=htype,
                protected=bool(row.get("protected", False)),
                season_use=str(row.get("season_use", "year_round")),
                geometry_wkt=row.get("geometry_wkt"),
                attributes={"source_id": row.get("source_id", "unknown")},
            )
        )

    edges: list[HabitatEdge] = []
    for row in edges_spec:
        eid = str(row["edge_id"])
        src = str(row["from_node"])
        dst = str(row["to_node"])
        process = str(row["process"])
        if process not in valid_processes:
            raise ValueError(f"Unsupported process '{process}'; expected one of {sorted(valid_processes)}")

        # Infer or validate intra vs inter edge_class
        inferred_class = "intra" if node_types[src] == node_types[dst] else "inter"
        eclass = str(row.get("edge_class", inferred_class))
        if process == EcologicalProcess.COMPLEMENTATION.value and eclass != "inter":
            raise ValueError(
                f"Complementation edge {eid} must connect distinct habitat types (inter-habitat)"
            )

        cost = float(row.get("cost", 1.0))
        prob = float(row.get("probability", 1.0))
        edges.append(
            HabitatEdge(
                edge_id=eid,
                source=src,
                target=dst,
                weight=prob / max(cost, 1e-6),
                cost=cost,
                probability=prob,
                distance=float(row["distance_m"]) if row.get("distance_m") is not None else None,
                edge_class=eclass,
                process=process,
                movement_type=str(row.get("movement_type", "dispersal")),
            )
        )

    return HabitatGraph(nodes, edges, SpatialMetadata(crs=crs, extra={"graph_type": "multi_habitat"}))


def analyze_hetero_processes(graph: HabitatGraph) -> HeteroProcessSummary:
    """Quantify process-specific ecological fluxes, source-sink balance, and complementation."""
    counts = {p.value: 0 for p in EcologicalProcess}
    fluxes = {p.value: 0.0 for p in EcologicalProcess}
    ss_balance = {nid: 0.0 for nid in graph.node_ids}
    comp_pairs: list[tuple[str, str, float]] = []
    supp_deg = {nid: 0 for nid in graph.node_ids}

    for e in graph.edges:
        src = graph.get_node(e.source)
        dst = graph.get_node(e.target)
        # Directed ecological flux weighted by source area * quality * probability
        flux = src.area * src.quality * e.probability

        if e.process in counts:
            counts[e.process] += 1
            fluxes[e.process] += flux

        if e.process == EcologicalProcess.SOURCE_SINK.value:
            # Net balance: negative for net exporter (source), positive for net importer (sink)
            ss_balance[e.source] -= flux
            ss_balance[e.target] += flux
        elif e.process == EcologicalProcess.COMPLEMENTATION.value:
            comp_pairs.append((e.source, e.target, flux))
        elif e.process == EcologicalProcess.SUPPLEMENTATION.value:
            supp_deg[e.source] += 1
            supp_deg[e.target] += 1

    comp_pairs.sort(key=lambda x: x[2], reverse=True)
    return HeteroProcessSummary(
        process_edge_counts=counts,
        process_total_flux=fluxes,
        source_sink_net_balance=ss_balance,
        complementation_pairs=tuple(comp_pairs),
        supplementation_redundancy=supp_deg,
    )
