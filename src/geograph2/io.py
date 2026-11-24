"""Serialization for HabitatGraph: GraphJSON and GeoParquet/Parquet."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
from shapely import wkb, wkt
from shapely.geometry import Point

from geograph2.ir import (
    HabitatEdge,
    HabitatGraph,
    HabitatNode,
    NodeKind,
    SpatialMetadata,
)

GRAPHJSON_SCHEMA_VERSION = "2.0.0"


def to_graphjson_dict(graph: HabitatGraph) -> dict[str, Any]:
    """Serialize a HabitatGraph into a GraphJSON-compatible Python dict."""
    meta = graph.metadata
    return {
        "schema": "geograph2.graphjson",
        "version": GRAPHJSON_SCHEMA_VERSION,
        "metadata": {
            "crs": meta.crs,
            "bbox": list(meta.bbox) if meta.bbox is not None else None,
            "resolution": list(meta.resolution) if meta.resolution is not None else None,
            "vertical_datum": meta.vertical_datum,
            "dggs_type": meta.dggs_type,
            "epoch": meta.epoch,
            "extra": dict(meta.extra),
        },
        "nodes": [
            {
                "node_id": n.node_id,
                "kind": n.kind.value,
                "coords": list(n.coords),
                "area": n.area,
                "quality": n.quality,
                "habitat_type": n.habitat_type,
                "z_layer": n.z_layer,
                "protected": n.protected,
                "season_use": n.season_use,
                "geometry_wkt": n.geometry_wkt,
                "attributes": dict(n.attributes),
            }
            for n in graph.iter_nodes()
        ],
        "edges": [
            {
                "edge_id": e.edge_id,
                "source": e.source,
                "target": e.target,
                "weight": e.weight,
                "cost": e.cost,
                "probability": e.probability,
                "distance": e.distance,
                "edge_class": e.edge_class,
                "process": e.process,
                "movement_type": e.movement_type,
                "attributes": dict(e.attributes),
            }
            for e in graph.edges
        ],
    }


def from_graphjson_dict(payload: dict[str, Any]) -> HabitatGraph:
    """Deserialize a GraphJSON dict into an immutable HabitatGraph."""
    if payload.get("schema") != "geograph2.graphjson":
        raise ValueError("Invalid GraphJSON schema header")
    m = payload.get("metadata", {})
    metadata = SpatialMetadata(
        crs=m.get("crs", "EPSG:4326"),
        bbox=tuple(m["bbox"]) if m.get("bbox") is not None else None,
        resolution=tuple(m["resolution"]) if m.get("resolution") is not None else None,
        vertical_datum=m.get("vertical_datum"),
        dggs_type=m.get("dggs_type"),
        epoch=m.get("epoch"),
        extra=m.get("extra", {}),
    )
    nodes = [
        HabitatNode(
            node_id=n["node_id"],
            kind=NodeKind(n["kind"]),
            coords=tuple(n["coords"]),
            area=float(n["area"]),
            quality=float(n["quality"]),
            habitat_type=n.get("habitat_type", "habitat"),
            z_layer=int(n.get("z_layer", 0)),
            protected=bool(n.get("protected", False)),
            season_use=n.get("season_use", "year_round"),
            geometry_wkt=n.get("geometry_wkt"),
            attributes=n.get("attributes", {}),
        )
        for n in payload.get("nodes", [])
    ]
    edges = [
        HabitatEdge(
            edge_id=e["edge_id"],
            source=e["source"],
            target=e["target"],
            weight=float(e["weight"]),
            cost=float(e["cost"]),
            probability=float(e["probability"]),
            distance=float(e["distance"]) if e.get("distance") is not None else None,
            edge_class=e.get("edge_class", "intra"),
            process=e.get("process", "dispersal"),
            movement_type=e.get("movement_type", "dispersal"),
            attributes=e.get("attributes", {}),
        )
        for e in payload.get("edges", [])
    ]
    return HabitatGraph(nodes, edges, metadata)


def save_graphjson(graph: HabitatGraph, path: str | Path) -> Path:
    """Write a HabitatGraph to a GraphJSON file."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(to_graphjson_dict(graph), indent=2, sort_keys=True), encoding="utf-8")
    return out


def load_graphjson(path: str | Path) -> HabitatGraph:
    """Read a HabitatGraph from a GraphJSON file."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return from_graphjson_dict(payload)


def _node_to_wkb(node: HabitatNode) -> bytes:
    if node.geometry_wkt:
        geom = wkt.loads(node.geometry_wkt)
    elif len(node.coords) >= 3:
        geom = Point(node.coords[0], node.coords[1], node.coords[2])
    else:
        geom = Point(node.coords[0], node.coords[1])
    return wkb.dumps(geom)


def save_geoparquet(
    graph: HabitatGraph,
    output_dir: str | Path,
) -> tuple[Path, Path]:
    """Save a HabitatGraph as GeoParquet 1.1.0 ``nodes.parquet`` + ``edges.parquet``."""
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    nodes_path = out_dir / "nodes.parquet"
    edges_path = out_dir / "edges.parquet"

    nodes_list = list(graph.iter_nodes())
    node_table = pa.table(
        {
            "node_id": pa.array([n.node_id for n in nodes_list], type=pa.string()),
            "kind": pa.array([n.kind.value for n in nodes_list], type=pa.string()),
            "coords_json": pa.array([json.dumps(list(n.coords)) for n in nodes_list], type=pa.string()),
            "area": pa.array([n.area for n in nodes_list], type=pa.float64()),
            "quality": pa.array([n.quality for n in nodes_list], type=pa.float64()),
            "habitat_type": pa.array([n.habitat_type for n in nodes_list], type=pa.string()),
            "z_layer": pa.array([n.z_layer for n in nodes_list], type=pa.int32()),
            "protected": pa.array([n.protected for n in nodes_list], type=pa.bool_()),
            "season_use": pa.array([n.season_use for n in nodes_list], type=pa.string()),
            "geometry_wkt": pa.array([n.geometry_wkt for n in nodes_list], type=pa.string()),
            "attributes_json": pa.array(
                [json.dumps(dict(n.attributes), sort_keys=True) for n in nodes_list],
                type=pa.string(),
            ),
            "geometry": pa.array([_node_to_wkb(n) for n in nodes_list], type=pa.binary()),
        }
    )

    geo_meta = {
        "version": "1.1.0",
        "primary_column": "geometry",
        "columns": {
            "geometry": {
                "encoding": "WKB",
                "geometry_types": ["Point", "Polygon", "Point Z"],
                "crs": graph.metadata.crs,
            }
        },
    }
    gg_meta = to_graphjson_dict(graph)["metadata"]
    existing_meta = dict(node_table.schema.metadata or {})
    existing_meta[b"geo"] = json.dumps(geo_meta).encode("utf-8")
    existing_meta[b"geograph2_metadata"] = json.dumps(gg_meta).encode("utf-8")
    node_table = node_table.replace_schema_metadata(existing_meta)
    pq.write_table(node_table, nodes_path)

    edge_table = pa.table(
        {
            "edge_id": pa.array([e.edge_id for e in graph.edges], type=pa.string()),
            "source": pa.array([e.source for e in graph.edges], type=pa.string()),
            "target": pa.array([e.target for e in graph.edges], type=pa.string()),
            "weight": pa.array([e.weight for e in graph.edges], type=pa.float64()),
            "cost": pa.array([e.cost for e in graph.edges], type=pa.float64()),
            "probability": pa.array([e.probability for e in graph.edges], type=pa.float64()),
            "distance": pa.array([e.distance for e in graph.edges], type=pa.float64()),
            "edge_class": pa.array([e.edge_class for e in graph.edges], type=pa.string()),
            "process": pa.array([e.process for e in graph.edges], type=pa.string()),
            "movement_type": pa.array([e.movement_type for e in graph.edges], type=pa.string()),
            "attributes_json": pa.array(
                [json.dumps(dict(e.attributes), sort_keys=True) for e in graph.edges],
                type=pa.string(),
            ),
        }
    )
    pq.write_table(edge_table, edges_path)
    return nodes_path, edges_path


def load_geoparquet(input_dir: str | Path) -> HabitatGraph:
    """Load a HabitatGraph from a GeoParquet directory (``nodes.parquet`` + ``edges.parquet``)."""
    in_dir = Path(input_dir)
    node_table = pq.read_table(in_dir / "nodes.parquet")
    edge_table = pq.read_table(in_dir / "edges.parquet")

    raw_meta = (node_table.schema.metadata or {}).get(b"geograph2_metadata")
    m = json.loads(raw_meta.decode("utf-8")) if raw_meta else {}
    metadata = SpatialMetadata(
        crs=m.get("crs", "EPSG:4326"),
        bbox=tuple(m["bbox"]) if m.get("bbox") is not None else None,
        resolution=tuple(m["resolution"]) if m.get("resolution") is not None else None,
        vertical_datum=m.get("vertical_datum"),
        dggs_type=m.get("dggs_type"),
        epoch=m.get("epoch"),
        extra=m.get("extra", {}),
    )

    nodes_py = node_table.to_pylist()
    nodes = [
        HabitatNode(
            node_id=row["node_id"],
            kind=NodeKind(row["kind"]),
            coords=tuple(json.loads(row["coords_json"])),
            area=float(row["area"]),
            quality=float(row["quality"]),
            habitat_type=row["habitat_type"],
            z_layer=int(row["z_layer"]),
            protected=bool(row["protected"]),
            season_use=row["season_use"],
            geometry_wkt=row["geometry_wkt"],
            attributes=json.loads(row["attributes_json"]),
        )
        for row in nodes_py
    ]

    edges_py = edge_table.to_pylist()
    edges = [
        HabitatEdge(
            edge_id=row["edge_id"],
            source=row["source"],
            target=row["target"],
            weight=float(row["weight"]),
            cost=float(row["cost"]),
            probability=float(row["probability"]),
            distance=float(row["distance"]) if row["distance"] is not None else None,
            edge_class=row["edge_class"],
            process=row["process"],
            movement_type=row["movement_type"],
            attributes=json.loads(row["attributes_json"]),
        )
        for row in edges_py
    ]
    return HabitatGraph(nodes, edges, metadata)
