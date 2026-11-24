"""2.5D stratified and 3D voxel habitat adjacency graph builders.

Implements multi-layer vertical connectivity (e.g., understory <-> midstory <-> canopy)
following the three-paradigm 3D landscape connectivity framework (Frazier et al. 2026)
with explicit, parameterized vertical cross-layer edge weights.
"""

from __future__ import annotations

import math
from typing import Mapping, Sequence

import numpy as np

from geograph2.ir import HabitatEdge, HabitatGraph, HabitatNode, NodeKind, SpatialMetadata


def _2d_offsets(connectivity: int) -> tuple[tuple[int, int, float], ...]:
    if connectivity == 4:
        return ((-1, 0, 1.0), (1, 0, 1.0), (0, -1, 1.0), (0, 1, 1.0))
    if connectivity == 8:
        rt2 = math.sqrt(2.0)
        return (
            (-1, 0, 1.0), (1, 0, 1.0), (0, -1, 1.0), (0, 1, 1.0),
            (-1, -1, rt2), (-1, 1, rt2), (1, -1, rt2), (1, 1, rt2),
        )
    raise ValueError("connectivity must be 4 or 8")


def build_stratified_25d_graph(
    strata_resistance: Mapping[str, np.ndarray],
    stratum_order: Sequence[str],
    pixel_size: float = 10.0,
    vertical_cost_factors: Mapping[tuple[str, str], float] | None = None,
    stratum_heights_m: Mapping[str, float] | None = None,
    connectivity: int = 4,
    dispersal_scale: float = 100.0,
) -> HabitatGraph:
    """Build a 2.5D stratified habitat graph across vertical vegetation/structural strata.

    Parameters
    ----------
    strata_resistance:
        Mapping from stratum name (e.g. ``"understory"``, ``"canopy"``) to a 2D
        non-negative resistance raster (NaN or <= 0 treated as barrier / omitted).
    stratum_order:
        Ordered vertical stack from lowest to highest stratum.
    vertical_cost_factors:
        Optional directed cross-layer resistance multiplier ``(from_stratum, to_stratum) -> factor``
        allowing asymmetric climbing (understory -> canopy) vs descent (canopy -> understory) costs.
    """
    if not stratum_order:
        raise ValueError("stratum_order must be non-empty")
    ref_shape = np.asarray(strata_resistance[stratum_order[0]]).shape
    if len(ref_shape) != 2:
        raise ValueError("Each stratum raster must be 2D")
    nrows, ncols = ref_shape
    vfactors = dict(vertical_cost_factors or {})
    heights = dict(stratum_heights_m or {})

    nodes: list[HabitatNode] = []
    valid_nodes: dict[tuple[int, int, int], tuple[str, float]] = {}

    for z_idx, sname in enumerate(stratum_order):
        arr = np.asarray(strata_resistance[sname], dtype=np.float64)
        if arr.shape != ref_shape:
            raise ValueError(f"Stratum {sname} shape {arr.shape} != {ref_shape}")
        z_m = float(heights.get(sname, z_idx * pixel_size))
        for r in range(nrows):
            for c in range(ncols):
                res = float(arr[r, c])
                if not math.isfinite(res) or res <= 0.0:
                    continue
                nid = f"{sname}_r{r}_c{c}"
                valid_nodes[(z_idx, r, c)] = (nid, res)
                nodes.append(
                    HabitatNode(
                        node_id=nid,
                        kind=NodeKind.VOXEL,
                        coords=((c + 0.5) * pixel_size, (r + 0.5) * pixel_size, z_m),
                        area=pixel_size * pixel_size,
                        quality=1.0 / res,
                        habitat_type=sname,
                        z_layer=z_idx,
                        attributes={"stratum": sname, "row": r, "col": c, "resistance": res},
                    )
                )

    edges: list[HabitatEdge] = []
    eid = 0
    offsets = _2d_offsets(connectivity)

    # 1. Intra-stratum horizontal edges
    for (z_idx, r, c), (nid, res_u) in valid_nodes.items():
        sname = stratum_order[z_idx]
        for dr, dc, step_mult in offsets:
            nb = valid_nodes.get((z_idx, r + dr, c + dc))
            if nb is None:
                continue
            nb_id, res_v = nb
            eid += 1
            dist = pixel_size * step_mult
            cost = dist * 0.5 * (res_u + res_v)
            prob = math.exp(-cost / max(1e-6, dispersal_scale))
            edges.append(
                HabitatEdge(
                    edge_id=f"h_{eid}",
                    source=nid,
                    target=nb_id,
                    weight=1.0 / max(1e-6, cost),
                    cost=cost,
                    probability=prob,
                    distance=dist,
                    edge_class="intra",
                    process="dispersal",
                    movement_type="horizontal",
                    attributes={"stratum": sname},
                )
            )

    # 2. Inter-stratum vertical cross-layer edges (e.g. understory <-> canopy)
    for z_idx in range(len(stratum_order) - 1):
        low_name = stratum_order[z_idx]
        high_name = stratum_order[z_idx + 1]
        dz = abs(
            float(heights.get(high_name, (z_idx + 1) * pixel_size))
            - float(heights.get(low_name, z_idx * pixel_size))
        )
        dz = max(1e-6, dz)
        up_factor = float(vfactors.get((low_name, high_name), 1.5))
        down_factor = float(vfactors.get((high_name, low_name), 1.0))

        for r in range(nrows):
            for c in range(ncols):
                low = valid_nodes.get((z_idx, r, c))
                high = valid_nodes.get((z_idx + 1, r, c))
                if low is None or high is None:
                    continue
                low_id, res_low = low
                high_id, res_high = high
                mean_res = 0.5 * (res_low + res_high)

                # Upward edge (e.g. understory -> canopy)
                eid += 1
                up_cost = dz * mean_res * up_factor
                edges.append(
                    HabitatEdge(
                        edge_id=f"v_{eid}",
                        source=low_id,
                        target=high_id,
                        weight=1.0 / max(1e-6, up_cost),
                        cost=up_cost,
                        probability=math.exp(-up_cost / max(1e-6, dispersal_scale)),
                        distance=dz,
                        edge_class="inter",
                        process="vertical_transition",
                        movement_type="ascent",
                        attributes={"from_stratum": low_name, "to_stratum": high_name},
                    )
                )

                # Downward edge (e.g. canopy -> understory)
                eid += 1
                down_cost = dz * mean_res * down_factor
                edges.append(
                    HabitatEdge(
                        edge_id=f"v_{eid}",
                        source=high_id,
                        target=low_id,
                        weight=1.0 / max(1e-6, down_cost),
                        cost=down_cost,
                        probability=math.exp(-down_cost / max(1e-6, dispersal_scale)),
                        distance=dz,
                        edge_class="inter",
                        process="vertical_transition",
                        movement_type="descent",
                        attributes={"from_stratum": high_name, "to_stratum": low_name},
                    )
                )

    meta = SpatialMetadata(
        crs="EPSG:3857",
        bbox=(0.0, 0.0, ncols * pixel_size, nrows * pixel_size),
        resolution=(pixel_size, pixel_size),
        vertical_datum="AGL_meters",
        extra={"strata": list(stratum_order), "model": "2.5D_stratified"},
    )
    return HabitatGraph(nodes, edges, meta)


def build_voxel_3d_graph(
    voxel_resistance: np.ndarray,
    voxel_size: tuple[float, float, float] = (10.0, 10.0, 5.0),
    vertical_cost_factor: float = 1.5,
    neighborhood: int = 6,
    dispersal_scale: float = 100.0,
) -> HabitatGraph:
    """Build a full 3D voxel adjacency HabitatGraph from a (nz, ny, nx) resistance array."""
    arr = np.asarray(voxel_resistance, dtype=np.float64)
    if arr.ndim != 3:
        raise ValueError("voxel_resistance must be a 3D array of shape (nz, ny, nx)")
    if neighborhood not in {6, 18, 26}:
        raise ValueError("neighborhood must be 6, 18, or 26")

    nz, ny, nx = arr.shape
    dx, dy, dz = float(voxel_size[0]), float(voxel_size[1]), float(voxel_size[2])
    max_manhattan = {6: 1, 18: 2, 26: 3}[neighborhood]

    offsets: list[tuple[int, int, int, float]] = []
    for oz in (-1, 0, 1):
        for oy in (-1, 0, 1):
            for ox in (-1, 0, 1):
                m = abs(oz) + abs(oy) + abs(ox)
                if 1 <= m <= max_manhattan:
                    dist = math.sqrt((ox * dx) ** 2 + (oy * dy) ** 2 + (oz * dz) ** 2)
                    offsets.append((oz, oy, ox, dist))

    nodes: list[HabitatNode] = []
    valid: dict[tuple[int, int, int], tuple[str, float]] = {}
    for z in range(nz):
        for y in range(ny):
            for x in range(nx):
                res = float(arr[z, y, x])
                if not math.isfinite(res) or res <= 0.0:
                    continue
                nid = f"vox_z{z}_y{y}_x{x}"
                valid[(z, y, x)] = (nid, res)
                nodes.append(
                    HabitatNode(
                        node_id=nid,
                        kind=NodeKind.VOXEL,
                        coords=((x + 0.5) * dx, (y + 0.5) * dy, (z + 0.5) * dz),
                        area=dx * dy,
                        quality=1.0 / res,
                        habitat_type=f"voxel_z{z}",
                        z_layer=z,
                        attributes={"z": z, "y": y, "x": x, "volume": dx * dy * dz},
                    )
                )

    edges: list[HabitatEdge] = []
    eid = 0
    for (z, y, x), (nid, res_u) in valid.items():
        for oz, oy, ox, dist in offsets:
            nb = valid.get((z + oz, y + oy, x + ox))
            if nb is None:
                continue
            nb_id, res_v = nb
            eid += 1
            v_mult = vertical_cost_factor if oz != 0 else 1.0
            cost = dist * 0.5 * (res_u + res_v) * v_mult
            edges.append(
                HabitatEdge(
                    edge_id=f"ve_{eid}",
                    source=nid,
                    target=nb_id,
                    weight=1.0 / max(1e-6, cost),
                    cost=cost,
                    probability=math.exp(-cost / max(1e-6, dispersal_scale)),
                    distance=dist,
                    edge_class="inter" if oz != 0 else "intra",
                    process="vertical_transition" if oz != 0 else "dispersal",
                    movement_type="3d_step" if oz != 0 else "horizontal",
                )
            )

    meta = SpatialMetadata(
        crs="EPSG:3857",
        bbox=(0.0, 0.0, nx * dx, ny * dy),
        resolution=(dx, dy, dz),
        vertical_datum="AGL_meters",
        extra={"neighborhood": neighborhood, "model": "3D_voxel"},
    )
    return HabitatGraph(nodes, edges, meta)
