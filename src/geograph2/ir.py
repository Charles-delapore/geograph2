"""Unified immutable spatial habitat topology Intermediate Representation (IR).

Provides point, polygon, and voxel node abstractions, directed weighted multi-edges,
spatial metadata, and sparse CSR matrix projections for downstream solvers.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType
from typing import Any, Iterator, Mapping, Sequence

import numpy as np
from scipy import sparse


class NodeKind(str, Enum):
    """Supported spatial geometry representations for habitat nodes."""

    POINT = "point"
    POLYGON = "polygon"
    VOXEL = "voxel"


def _freeze_mapping(data: Mapping[str, Any] | None) -> Mapping[str, Any]:
    if not data:
        return MappingProxyType({})
    return MappingProxyType(dict(data))


@dataclass(frozen=True)
class SpatialMetadata:
    """Coordinate reference system and spatial domain metadata for a HabitatGraph."""

    crs: str = "EPSG:4326"
    bbox: tuple[float, float, float, float] | None = None
    resolution: tuple[float, ...] | None = None
    vertical_datum: str | None = None
    dggs_type: str | None = None
    epoch: str | None = None
    extra: Mapping[str, Any] = field(default_factory=lambda: MappingProxyType({}))

    def __post_init__(self) -> None:
        if not isinstance(self.extra, MappingProxyType):
            object.__setattr__(self, "extra", _freeze_mapping(self.extra))
        if self.bbox is not None and len(self.bbox) != 4:
            raise ValueError("bbox must be a 4-tuple (minx, miny, maxx, maxy)")


@dataclass(frozen=True)
class HabitatNode:
    """Immutable habitat node representing a point, polygon patch, or 3D voxel."""

    node_id: str
    kind: NodeKind = NodeKind.POINT
    coords: tuple[float, ...] = (0.0, 0.0)
    area: float = 1.0
    quality: float = 1.0
    habitat_type: str = "habitat"
    z_layer: int = 0
    protected: bool = False
    season_use: str = "year_round"
    geometry_wkt: str | None = None
    attributes: Mapping[str, Any] = field(default_factory=lambda: MappingProxyType({}))

    def __post_init__(self) -> None:
        if not isinstance(self.kind, NodeKind):
            object.__setattr__(self, "kind", NodeKind(self.kind))
        if not isinstance(self.coords, tuple):
            object.__setattr__(self, "coords", tuple(float(c) for c in self.coords))
        if self.area < 0.0:
            raise ValueError(f"Node {self.node_id} area must be non-negative")
        if self.quality < 0.0:
            raise ValueError(f"Node {self.node_id} quality must be non-negative")
        if not isinstance(self.attributes, MappingProxyType):
            object.__setattr__(self, "attributes", _freeze_mapping(self.attributes))


@dataclass(frozen=True)
class HabitatEdge:
    """Immutable directed weighted multi-edge between two habitat nodes."""

    edge_id: str
    source: str
    target: str
    weight: float = 1.0
    cost: float = 1.0
    probability: float = 1.0
    distance: float | None = None
    edge_class: str = "intra"
    process: str = "dispersal"
    movement_type: str = "dispersal"
    attributes: Mapping[str, Any] = field(default_factory=lambda: MappingProxyType({}))

    def __post_init__(self) -> None:
        if self.cost < 0.0:
            raise ValueError(f"Edge {self.edge_id} cost must be non-negative")
        if not (0.0 <= self.probability <= 1.0):
            raise ValueError(f"Edge {self.edge_id} probability must be in [0, 1]")
        if not isinstance(self.attributes, MappingProxyType):
            object.__setattr__(self, "attributes", _freeze_mapping(self.attributes))


class HabitatGraph:
    """Immutable directed weighted multigraph IR for spatial ecology.

    Stores nodes in deterministic insertion order, supports parallel directed edges
    between the same node pair (multi-edges), and exposes cached sparse matrix views.
    """

    __slots__ = (
        "_nodes",
        "_node_index",
        "_node_ids",
        "_edges",
        "_edge_map",
        "_out_edges",
        "_in_edges",
        "_metadata",
    )

    def __init__(
        self,
        nodes: Sequence[HabitatNode],
        edges: Sequence[HabitatEdge] = (),
        metadata: SpatialMetadata | None = None,
    ) -> None:
        node_dict: dict[str, HabitatNode] = {}
        node_index: dict[str, int] = {}
        node_ids: list[str] = []
        for idx, node in enumerate(nodes):
            if node.node_id in node_dict:
                raise ValueError(f"Duplicate node_id: {node.node_id}")
            node_dict[node.node_id] = node
            node_index[node.node_id] = idx
            node_ids.append(node.node_id)

        edge_list: list[HabitatEdge] = []
        edge_map: dict[str, HabitatEdge] = {}
        out_edges: dict[str, list[HabitatEdge]] = {nid: [] for nid in node_ids}
        in_edges: dict[str, list[HabitatEdge]] = {nid: [] for nid in node_ids}

        for edge in edges:
            if edge.edge_id in edge_map:
                raise ValueError(f"Duplicate edge_id: {edge.edge_id}")
            if edge.source not in node_dict:
                raise ValueError(f"Edge {edge.edge_id} references unknown source: {edge.source}")
            if edge.target not in node_dict:
                raise ValueError(f"Edge {edge.edge_id} references unknown target: {edge.target}")
            edge_list.append(edge)
            edge_map[edge.edge_id] = edge
            out_edges[edge.source].append(edge)
            in_edges[edge.target].append(edge)

        self._nodes = MappingProxyType(node_dict)
        self._node_index = MappingProxyType(node_index)
        self._node_ids = tuple(node_ids)
        self._edges = tuple(edge_list)
        self._edge_map = MappingProxyType(edge_map)
        self._out_edges = MappingProxyType({k: tuple(v) for k, v in out_edges.items()})
        self._in_edges = MappingProxyType({k: tuple(v) for k, v in in_edges.items()})
        self._metadata = metadata if metadata is not None else SpatialMetadata()

    @property
    def metadata(self) -> SpatialMetadata:
        return self._metadata

    @property
    def node_ids(self) -> tuple[str, ...]:
        return self._node_ids

    @property
    def nodes(self) -> Mapping[str, HabitatNode]:
        return self._nodes

    @property
    def edges(self) -> tuple[HabitatEdge, ...]:
        return self._edges

    def __len__(self) -> int:
        return len(self._node_ids)

    @property
    def num_nodes(self) -> int:
        return len(self._node_ids)

    @property
    def num_edges(self) -> int:
        return len(self._edges)

    def node_index(self, node_id: str) -> int:
        return self._node_index[node_id]

    def get_node(self, node_id: str) -> HabitatNode:
        return self._nodes[node_id]

    def get_edge(self, edge_id: str) -> HabitatEdge:
        return self._edge_map[edge_id]

    def out_edges(self, node_id: str) -> tuple[HabitatEdge, ...]:
        return self._out_edges[node_id]

    def in_edges(self, node_id: str) -> tuple[HabitatEdge, ...]:
        return self._in_edges[node_id]

    def parallel_edges(self, source: str, target: str) -> tuple[HabitatEdge, ...]:
        """Return all directed multi-edges from source to target."""
        return tuple(e for e in self._out_edges[source] if e.target == target)

    def iter_nodes(self) -> Iterator[HabitatNode]:
        for nid in self._node_ids:
            yield self._nodes[nid]

    def with_nodes_and_edges(
        self,
        nodes: Sequence[HabitatNode] | None = None,
        edges: Sequence[HabitatEdge] | None = None,
        metadata: SpatialMetadata | None = None,
    ) -> HabitatGraph:
        """Functional persistent update returning a new immutable HabitatGraph."""
        return HabitatGraph(
            nodes=list(self.iter_nodes()) if nodes is None else nodes,
            edges=self._edges if edges is None else edges,
            metadata=self._metadata if metadata is None else metadata,
        )

    def without_node(self, node_id: str) -> HabitatGraph:
        """Return a new HabitatGraph with the specified node and incident edges removed."""
        if node_id not in self._nodes:
            raise KeyError(f"Unknown node_id: {node_id}")
        kept_nodes = [n for n in self.iter_nodes() if n.node_id != node_id]
        kept_edges = [e for e in self._edges if e.source != node_id and e.target != node_id]
        return HabitatGraph(kept_nodes, kept_edges, self._metadata)

    def without_edge(self, edge_id: str) -> HabitatGraph:
        """Return a new HabitatGraph with the specified edge removed."""
        if edge_id not in self._edge_map:
            raise KeyError(f"Unknown edge_id: {edge_id}")
        kept_edges = [e for e in self._edges if e.edge_id != edge_id]
        return HabitatGraph(list(self.iter_nodes()), kept_edges, self._metadata)

    def subgraph_by_habitat(self, habitat_types: Sequence[str]) -> HabitatGraph:
        """Filter graph to nodes matching the given habitat_types."""
        allowed = set(habitat_types)
        kept_nodes = [n for n in self.iter_nodes() if n.habitat_type in allowed]
        kept_ids = {n.node_id for n in kept_nodes}
        kept_edges = [
            e for e in self._edges if e.source in kept_ids and e.target in kept_ids
        ]
        return HabitatGraph(kept_nodes, kept_edges, self._metadata)

    def to_sparse_matrix(
        self,
        attr: str = "weight",
        reduce: str = "sum",
    ) -> sparse.csr_matrix:
        """Project directed multi-edges onto an (N, N) scipy.sparse.csr_matrix.

        Parameters
        ----------
        attr:
            Edge field to project: ``"weight"``, ``"cost"``, or ``"probability"``.
        reduce:
            Aggregation rule for parallel edges: ``"sum"``, ``"min"``, or ``"max"``.
        """
        if attr not in {"weight", "cost", "probability"}:
            raise ValueError(f"Unsupported edge attribute: {attr}")
        if reduce not in {"sum", "min", "max"}:
            raise ValueError(f"Unsupported reduction: {reduce}")

        n = self.num_nodes
        if n == 0 or self.num_edges == 0:
            return sparse.csr_matrix((n, n), dtype=np.float64)

        agg: dict[tuple[int, int], float] = {}
        for edge in self._edges:
            u = self._node_index[edge.source]
            v = self._node_index[edge.target]
            val = float(getattr(edge, attr))
            key = (u, v)
            if key not in agg:
                agg[key] = val
            elif reduce == "sum":
                agg[key] += val
            elif reduce == "min":
                agg[key] = min(agg[key], val)
            elif reduce == "max":
                agg[key] = max(agg[key], val)

        rows = np.fromiter((k[0] for k in agg), dtype=np.int32, count=len(agg))
        cols = np.fromiter((k[1] for k in agg), dtype=np.int32, count=len(agg))
        vals = np.fromiter(agg.values(), dtype=np.float64, count=len(agg))
        return sparse.csr_matrix((vals, (rows, cols)), shape=(n, n), dtype=np.float64)

    def node_areas(self) -> np.ndarray:
        """Return 1D float64 array of node areas aligned with ``node_ids``."""
        return np.array([self._nodes[nid].area for nid in self._node_ids], dtype=np.float64)

    def node_qualities(self) -> np.ndarray:
        """Return 1D float64 array of node qualities aligned with ``node_ids``."""
        return np.array([self._nodes[nid].quality for nid in self._node_ids], dtype=np.float64)
