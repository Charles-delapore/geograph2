"""Cross-epoch patch lineage tracking (split/merge/extinction/emergence) and temporal dECA curves."""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import Sequence

from shapely import wkt
from shapely.geometry.base import BaseGeometry

from geograph2.connectivity import compute_deca, compute_pc
from geograph2.ir import HabitatGraph, HabitatNode


class LineageEventType(str, Enum):
    """Patch topological evolution events between consecutive epochs."""

    PERSIST = "persist"
    SPLIT = "split"
    MERGE = "merge"
    EXTINCTION = "extinction"
    EMERGENCE = "emergence"


@dataclass(frozen=True)
class LineageEvent:
    """A single tracked patch lineage transition between epoch t_from and t_to."""

    epoch_from: str
    epoch_to: str
    event_type: LineageEventType
    parent_ids: tuple[str, ...]
    child_ids: tuple[str, ...]
    overlap_area: float


@dataclass(frozen=True)
class TemporalDECAPoint:
    """Connectivity trajectory point for a single epoch."""

    epoch: str
    num_patches: int
    total_habitat_area: float
    pc: float
    eca: float
    deca_step: float
    rdeca_step_pct: float
    deca_cumulative: float


def _node_geom(node: HabitatNode) -> BaseGeometry | None:
    if node.geometry_wkt:
        return wkt.loads(node.geometry_wkt)
    return None


def _spatial_overlap(n1: HabitatNode, n2: HabitatNode, radius_fallback: float) -> float:
    g1 = _node_geom(n1)
    g2 = _node_geom(n2)
    if g1 is not None and g2 is not None:
        inter = g1.intersection(g2).area
        return float(inter)
    # Fallback to Euclidean centroid proximity if polygons are not provided
    dist = math.hypot(n1.coords[0] - n2.coords[0], n1.coords[1] - n2.coords[1])
    if dist <= radius_fallback:
        return min(n1.area, n2.area) * max(0.0, 1.0 - dist / max(radius_fallback, 1e-6))
    return 0.0


def track_patch_lineage(
    epochs: Sequence[tuple[str, HabitatGraph]],
    min_overlap_ratio: float = 0.05,
    centroid_fallback_radius: float = 10.0,
) -> list[LineageEvent]:
    """Track patch split, merge, persist, extinction, and emergence events across epochs."""
    if len(epochs) < 2:
        return []

    events: list[LineageEvent] = []
    for idx in range(len(epochs) - 1):
        ep1, g1 = epochs[idx]
        ep2, g2 = epochs[idx + 1]

        parent_to_children: dict[str, list[tuple[str, float]]] = {nid: [] for nid in g1.node_ids}
        child_to_parents: dict[str, list[tuple[str, float]]] = {nid: [] for nid in g2.node_ids}

        for n1 in g1.iter_nodes():
            for n2 in g2.iter_nodes():
                ov = _spatial_overlap(n1, n2, centroid_fallback_radius)
                denom = max(min(n1.area, n2.area), 1e-9)
                if ov > 0.0 and (ov / denom) >= min_overlap_ratio:
                    parent_to_children[n1.node_id].append((n2.node_id, ov))
                    child_to_parents[n2.node_id].append((n1.node_id, ov))

        handled_children: set[str] = set()

        # 1. Extinctions (1 -> 0) and Splits (1 -> k)
        for pid, children in parent_to_children.items():
            if len(children) == 0:
                events.append(
                    LineageEvent(ep1, ep2, LineageEventType.EXTINCTION, (pid,), (), 0.0)
                )
            elif len(children) >= 2:
                c_ids = tuple(sorted(c[0] for c in children))
                ov_sum = sum(c[1] for c in children)
                events.append(
                    LineageEvent(ep1, ep2, LineageEventType.SPLIT, (pid,), c_ids, ov_sum)
                )
                handled_children.update(c_ids)

        # 2. Merges (k -> 1), Emergences (0 -> 1), and 1-to-1 Persists
        for cid, parents in child_to_parents.items():
            if cid in handled_children:
                continue
            if len(parents) == 0:
                events.append(
                    LineageEvent(ep1, ep2, LineageEventType.EMERGENCE, (), (cid,), 0.0)
                )
            elif len(parents) >= 2:
                p_ids = tuple(sorted(p[0] for p in parents))
                ov_sum = sum(p[1] for p in parents)
                events.append(
                    LineageEvent(ep1, ep2, LineageEventType.MERGE, p_ids, (cid,), ov_sum)
                )
            else:
                pid, ov = parents[0]
                if len(parent_to_children[pid]) == 1:
                    events.append(
                        LineageEvent(ep1, ep2, LineageEventType.PERSIST, (pid,), (cid,), ov)
                    )

    return events


def compute_temporal_deca_curve(
    epochs: Sequence[tuple[str, HabitatGraph]],
    landscape_area: float,
) -> list[TemporalDECAPoint]:
    """Compute time-series PC, ECA, step dECA/rdECA, and cumulative dECA across epochs."""
    curve: list[TemporalDECAPoint] = []
    first_eca: float | None = None
    prev_eca: float | None = None

    for epoch_label, graph in epochs:
        res = compute_pc(
            graph,
            landscape_area=landscape_area,
            compute_node_importance=False,
            compute_edge_importance=False,
        )
        if first_eca is None:
            first_eca = res.eca
            step_deca = 0.0
            step_rdeca = 0.0
        else:
            d = compute_deca(prev_eca if prev_eca is not None else res.eca, res.eca)
            step_deca = d["dECA"]
            step_rdeca = d["rdECA_pct"]

        curve.append(
            TemporalDECAPoint(
                epoch=epoch_label,
                num_patches=graph.num_nodes,
                total_habitat_area=float(graph.node_areas().sum()),
                pc=res.value,
                eca=res.eca,
                deca_step=step_deca,
                rdeca_step_pct=step_rdeca,
                deca_cumulative=res.eca - first_eca,
            )
        )
        prev_eca = res.eca
    return curve
