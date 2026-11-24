"""Core FRAGSTATS / landscapemetrics / PyLandStats landscape pattern indices.

Implements clean-room patch-, class-, and landscape-level pattern metrics on 2D
categorical rasters: NP, CA, TA, PLAND, ED, FRAC, ENN, CONTAG, and SHDI.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy import ndimage
from scipy.spatial.distance import cdist


@dataclass(frozen=True)
class PatchMetricRow:
    """Patch-level FRAGSTATS metrics."""

    class_val: int
    patch_id: int
    area_m2: float
    area_ha: float
    perimeter_m: float
    frac: float
    enn_m: float


@dataclass(frozen=True)
class ClassMetricRow:
    """Class-level FRAGSTATS metrics."""

    class_val: int
    np: int
    ca_ha: float
    pland: float
    ed_m_per_ha: float
    frac_mn: float
    enn_mn_m: float


@dataclass(frozen=True)
class LandscapeMetricsResult:
    """Full patch-, class-, and landscape-level FRAGSTATS summary."""

    ta_ha: float
    np_total: int
    ed_m_per_ha: float
    shdi: float
    contag: float
    class_metrics: tuple[ClassMetricRow, ...]
    patch_metrics: tuple[PatchMetricRow, ...]


def _patch_perimeter_pixels(mask: np.ndarray) -> int:
    """Exact orthogonal perimeter of a binary patch mask in pixel edge units."""
    # Pad with False so boundary edges count as perimeter
    p = np.pad(mask, 1, mode="constant", constant_values=False)
    horiz = np.sum(p[:-1, :] != p[1:, :])
    vert = np.sum(p[:, :-1] != p[:, 1:])
    return int(horiz + vert)


def _compute_enn_for_class(
    labeled: np.ndarray,
    num_patches: int,
    cell_size: float,
) -> dict[int, float]:
    """Compute edge-to-edge Euclidean nearest neighbor distance (FRAGSTATS/landscapemetrics).

    Uses boundary pixel centers + pixel-edge Euclidean distance (exact for 1-unit cells
    when comparing against landscapemetrics lsm_p_enn).
    """
    if num_patches <= 1:
        return {1: float("nan")} if num_patches == 1 else {}

    # Extract boundary coordinates of each patch
    coords_by_patch: dict[int, np.ndarray] = {}
    struct4 = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=bool)
    for pid in range(1, num_patches + 1):
        mask = labeled == pid
        eroded = ndimage.binary_erosion(mask, structure=struct4, border_value=0)
        boundary = mask & (~eroded)
        pts = np.argwhere(boundary).astype(np.float64) * cell_size
        coords_by_patch[pid] = pts

    enn: dict[int, float] = {}
    for pid in range(1, num_patches + 1):
        pts_i = coords_by_patch[pid]
        best = float("inf")
        for qid in range(1, num_patches + 1):
            if pid == qid:
                continue
            dmat = cdist(pts_i, coords_by_patch[qid], metric="euclidean")
            dmin = float(np.min(dmat))
            if dmin < best:
                best = dmin
        enn[pid] = best
    return enn


def compute_landscape_metrics(
    raster: np.ndarray,
    cell_size_m: float = 10.0,
    nodata: int | None = None,
    directions: int = 8,
) -> LandscapeMetricsResult:
    """Compute FRAGSTATS patch, class, and landscape metrics on a 2D integer raster.

    Parameters
    ----------
    raster:
        2D integer array of land-cover class codes.
    cell_size_m:
        Pixel resolution in meters (area in ha uses 1 ha = 10,000 m^2).
    nodata:
        Optional nodata class value to exclude from analysis.
    directions:
        Patch adjacency rule: 4 (rook) or 8 (queen, default in landscapemetrics).
    """
    arr = np.asarray(raster)
    if arr.ndim != 2:
        raise ValueError("raster must be a 2D array")
    if cell_size_m <= 0.0:
        raise ValueError("cell_size_m must be positive")
    if directions not in {4, 8}:
        raise ValueError("directions must be 4 or 8")

    structure = (
        np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=int)
        if directions == 4
        else np.ones((3, 3), dtype=int)
    )

    valid_mask = np.ones(arr.shape, dtype=bool) if nodata is None else (arr != nodata)
    total_pixels = int(np.sum(valid_mask))
    if total_pixels == 0:
        raise ValueError("Raster contains no valid pixels")

    pixel_area_m2 = cell_size_m * cell_size_m
    ta_m2 = total_pixels * pixel_area_m2
    ta_ha = ta_m2 / 10_000.0

    classes = sorted(int(c) for c in np.unique(arr[valid_mask]))
    patch_rows: list[PatchMetricRow] = []
    class_rows: list[ClassMetricRow] = []

    # Internal edge count between different valid classes for landscape ED
    diff_h = (arr[:-1, :] != arr[1:, :]) & valid_mask[:-1, :] & valid_mask[1:, :]
    diff_v = (arr[:, :-1] != arr[:, 1:]) & valid_mask[:, :-1] & valid_mask[:, 1:]
    landscape_edge_m = float(np.sum(diff_h) + np.sum(diff_v)) * cell_size_m
    landscape_ed = landscape_edge_m / ta_ha

    props: list[float] = []

    for cls in classes:
        cls_mask = (arr == cls) & valid_mask
        cls_pixels = int(np.sum(cls_mask))
        ca_ha = (cls_pixels * pixel_area_m2) / 10_000.0
        pland = 100.0 * cls_pixels / total_pixels
        props.append(cls_pixels / total_pixels)

        # Class internal edges (boundary with other valid classes, matching FRAGSTATS ED)
        ch = ((arr[:-1, :] == cls) ^ (arr[1:, :] == cls)) & valid_mask[:-1, :] & valid_mask[1:, :]
        cv = ((arr[:, :-1] == cls) ^ (arr[:, 1:] == cls)) & valid_mask[:, :-1] & valid_mask[:, 1:]
        cls_edge_m = float(np.sum(ch) + np.sum(cv)) * cell_size_m
        cls_ed = cls_edge_m / ta_ha

        labeled, n_patches = ndimage.label(cls_mask, structure=structure)
        enn_map = _compute_enn_for_class(labeled, n_patches, cell_size_m)

        cls_fracs: list[float] = []
        cls_enns: list[float] = []
        for pid in range(1, n_patches + 1):
            pmask = labeled == pid
            n_px = int(np.sum(pmask))
            area_m2 = n_px * pixel_area_m2
            area_ha = area_m2 / 10_000.0
            perim_m = _patch_perimeter_pixels(pmask) * cell_size_m
            if area_m2 > 1.0:
                frac = 2.0 * math.log(0.25 * perim_m) / math.log(area_m2)
            else:
                frac = 1.0
            frac = max(1.0, min(2.0, frac))
            enn_val = enn_map[pid]
            cls_fracs.append(frac)
            if math.isfinite(enn_val):
                cls_enns.append(enn_val)
            patch_rows.append(
                PatchMetricRow(
                    class_val=cls,
                    patch_id=pid,
                    area_m2=area_m2,
                    area_ha=area_ha,
                    perimeter_m=perim_m,
                    frac=frac,
                    enn_m=enn_val,
                )
            )

        class_rows.append(
            ClassMetricRow(
                class_val=cls,
                np=n_patches,
                ca_ha=ca_ha,
                pland=pland,
                ed_m_per_ha=cls_ed,
                frac_mn=float(np.mean(cls_fracs)) if cls_fracs else float("nan"),
                enn_mn_m=float(np.mean(cls_enns)) if cls_enns else float("nan"),
            )
        )

    # Shannon Diversity Index (SHDI = -sum(P_i * ln(P_i)))
    shdi = -sum(p * math.log(p) for p in props if p > 0.0)

    # Contagion Index (CONTAG, FRAGSTATS double-count adjacency)
    m = len(classes)
    if m <= 1:
        contag = 100.0
    else:
        cls_to_idx = {c: i for i, c in enumerate(classes)}
        adj = np.zeros((m, m), dtype=np.float64)
        # Horizontal pairs (both directions for double-count)
        r_h, c_h = np.where(valid_mask[:-1, :] & valid_mask[1:, :])
        for r, c in zip(r_h, c_h):
            i = cls_to_idx[int(arr[r, c])]
            j = cls_to_idx[int(arr[r + 1, c])]
            adj[i, j] += 1.0
            adj[j, i] += 1.0
        # Vertical pairs
        r_v, c_v = np.where(valid_mask[:, :-1] & valid_mask[:, 1:])
        for r, c in zip(r_v, c_v):
            i = cls_to_idx[int(arr[r, c])]
            j = cls_to_idx[int(arr[r, c + 1])]
            adj[i, j] += 1.0
            adj[j, i] += 1.0

        row_totals = adj.sum(axis=1)
        entropy_sum = 0.0
        for i in range(m):
            if row_totals[i] <= 0.0 or props[i] <= 0.0:
                continue
            for j in range(m):
                if adj[i, j] > 0.0:
                    p_ij = props[i] * (adj[i, j] / row_totals[i])
                    entropy_sum += p_ij * math.log(p_ij)
        contag = (1.0 + entropy_sum / (2.0 * math.log(m))) * 100.0
        contag = max(0.0, min(100.0, contag))

    return LandscapeMetricsResult(
        ta_ha=ta_ha,
        np_total=len(patch_rows),
        ed_m_per_ha=landscape_ed,
        shdi=shdi,
        contag=contag,
        class_metrics=tuple(class_rows),
        patch_metrics=tuple(patch_rows),
    )
