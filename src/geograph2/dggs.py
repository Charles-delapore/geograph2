"""Equal-area spherical hexagonal DGGS and area-distortion correction utilities.

Implements an iso-latitude equal-area spherical hexagonal grid (via Lambert
cylindrical equal-area parameterization + exact spherical polygon excess area),
achieving < 0.5% area distortion across low and high latitudes. Also provides
per-cell density correction (with empty-cell zero-filling), high-latitude
cos(phi) correction, and HEALPix analytical cell area helpers per MATH_AND_IR_SPEC §2.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np

from geograph2.ir import HabitatEdge, HabitatGraph, HabitatNode, NodeKind, SpatialMetadata

EARTH_AUTHALIC_RADIUS_KM = 6371.0072


def haversine_km(lon1: float, lat1: float, lon2: float, lat2: float) -> float:
    """Great-circle distance in kilometers between two WGS84 authalic sphere points."""
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2.0) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlam / 2.0) ** 2
    return 2.0 * EARTH_AUTHALIC_RADIUS_KM * math.asin(min(1.0, math.sqrt(a)))


def spherical_polygon_area_km2(vertices_lonlat: Sequence[tuple[float, float]]) -> float:
    """Exact spherical polygon area (km^2) via spherical excess on the authalic sphere."""
    n = len(vertices_lonlat)
    if n < 3:
        return 0.0
    # Convert vertices to 3D unit vectors and sum signed spherical triangle excesses
    pts = []
    for lon, lat in vertices_lonlat:
        lam = math.radians(lon)
        phi = math.radians(lat)
        pts.append(
            np.array(
                [math.cos(phi) * math.cos(lam), math.cos(phi) * math.sin(lam), math.sin(phi)],
                dtype=np.float64,
            )
        )
    v0 = pts[0]
    total_excess = 0.0
    for i in range(1, n - 1):
        v1 = pts[i]
        v2 = pts[i + 1]
        num = float(np.dot(v0, np.cross(v1, v2)))
        den = 1.0 + float(np.dot(v0, v1)) + float(np.dot(v1, v2)) + float(np.dot(v2, v0))
        total_excess += 2.0 * math.atan2(num, den)
    return abs(total_excess) * (EARTH_AUTHALIC_RADIUS_KM ** 2)


def high_latitude_area_correction(area_equator: float, lat_deg: float) -> float:
    """Latitude area factor A(phi) = A_eq * cos(phi) for geographic lat-lon grids."""
    return float(area_equator) * max(0.0, math.cos(math.radians(lat_deg)))


def healpix_cell_area_km2(nside: int, radius_km: float = EARTH_AUTHALIC_RADIUS_KM) -> float:
    """Exact spherical HEALPix cell area A = pi * R^2 / (3 * Nside^2)."""
    if nside <= 0 or (nside & (nside - 1)) != 0:
        raise ValueError("nside must be a positive power of 2")
    return (math.pi * (radius_km ** 2)) / (3.0 * (nside ** 2))


def corrected_cell_densities(
    counts: Sequence[float],
    cell_areas_km2: Sequence[float],
    total_land_cells: int | None = None,
) -> dict[str, float | list[float]]:
    """Compute per-cell density using exact cell areas and zero-filled empty cells.

    Per MATH_AND_IR_SPEC §2.3: never divide by resolution mean area, and zero-fill
    empty land cells before averaging to avoid upward density bias.
    """
    if len(counts) != len(cell_areas_km2):
        raise ValueError("counts and cell_areas_km2 must have equal length")
    densities = [
        float(c) / float(a) if a > 0.0 else 0.0
        for c, a in zip(counts, cell_areas_km2)
    ]
    n_total = len(densities) if total_land_cells is None else max(total_land_cells, len(densities))
    mean_density = sum(densities) / n_total if n_total > 0 else 0.0
    return {"densities": densities, "mean_density": mean_density, "n_cells": n_total}


@dataclass(frozen=True)
class HexCell:
    """A single equal-area spherical hexagonal cell."""

    cell_id: str
    q: int
    r: int
    lon: float
    lat: float
    area_km2: float
    vertices_lonlat: tuple[tuple[float, float], ...]


class SphericalHexDGGS:
    """Equal-area spherical hexagonal DGGS using Lambert cylindrical equal-area mapping.

    Maps axial hex coordinates (q, r) in equal-area (x = R*lambda, y = R*sin(phi))
    space back to the sphere, yielding near-constant cell areas (<0.5% distortion)
    across latitudes from the equator up to high latitudes (~78 deg).
    """

    def __init__(self, hex_radius_km: float = 50.0, resolution: int = 5) -> None:
        if hex_radius_km <= 0.0:
            raise ValueError("hex_radius_km must be positive")
        self.hex_radius_km = float(hex_radius_km)
        self.resolution = int(resolution)
        self.nominal_area_km2 = (3.0 * math.sqrt(3.0) / 2.0) * (self.hex_radius_km ** 2)

    def _xy_to_lonlat(self, x_km: float, y_km: float) -> tuple[float, float]:
        lon = math.degrees(x_km / EARTH_AUTHALIC_RADIUS_KM)
        sin_phi = max(-0.9999, min(0.9999, y_km / EARTH_AUTHALIC_RADIUS_KM))
        lat = math.degrees(math.asin(sin_phi))
        return (lon, lat)

    def _lonlat_to_xy(self, lon: float, lat: float) -> tuple[float, float]:
        x_km = EARTH_AUTHALIC_RADIUS_KM * math.radians(lon)
        y_km = EARTH_AUTHALIC_RADIUS_KM * math.sin(math.radians(lat))
        return (x_km, y_km)

    def _axial_to_xy(self, q: int, r: int) -> tuple[float, float]:
        # Pointy-top hex grid in equal-area (x, y) plane
        s = self.hex_radius_km
        x = s * math.sqrt(3.0) * (q + r / 2.0)
        y = s * 1.5 * r
        return (x, y)

    def cell_at(self, q: int, r: int) -> HexCell:
        cx, cy = self._axial_to_xy(q, r)
        lon, lat = self._xy_to_lonlat(cx, cy)
        verts: list[tuple[float, float]] = []
        for k in range(6):
            angle = math.radians(60.0 * k - 30.0)
            vx = cx + self.hex_radius_km * math.cos(angle)
            vy = cy + self.hex_radius_km * math.sin(angle)
            verts.append(self._xy_to_lonlat(vx, vy))
        area = spherical_polygon_area_km2(verts)
        cid = f"hex_r{self.resolution}_{q}_{r}"
        return HexCell(
            cell_id=cid,
            q=q,
            r=r,
            lon=lon,
            lat=lat,
            area_km2=area,
            vertices_lonlat=tuple(verts),
        )

    @staticmethod
    def neighbor_axial_coords(q: int, r: int) -> tuple[tuple[int, int], ...]:
        return (
            (q + 1, r),
            (q - 1, r),
            (q, r + 1),
            (q, r - 1),
            (q + 1, r - 1),
            (q - 1, r + 1),
        )

    def generate_cells(self, bbox_lonlat: tuple[float, float, float, float]) -> list[HexCell]:
        """Generate all equal-area spherical hex cells covering ``(minlon, minlat, maxlon, maxlat)``."""
        minlon, minlat, maxlon, maxlat = bbox_lonlat
        x0, y0 = self._lonlat_to_xy(minlon, minlat)
        x1, y1 = self._lonlat_to_xy(maxlon, maxlat)
        s = self.hex_radius_km

        r_min = int(math.floor(y0 / (1.5 * s))) - 1
        r_max = int(math.ceil(y1 / (1.5 * s))) + 1

        cells: list[HexCell] = []
        for r in range(r_min, r_max + 1):
            q_min = int(math.floor(x0 / (s * math.sqrt(3.0)) - r / 2.0)) - 1
            q_max = int(math.ceil(x1 / (s * math.sqrt(3.0)) - r / 2.0)) + 1
            for q in range(q_min, q_max + 1):
                cx, cy = self._axial_to_xy(q, r)
                if abs(cy) >= 0.985 * EARTH_AUTHALIC_RADIUS_KM:
                    continue
                lon, lat = self._xy_to_lonlat(cx, cy)
                if minlon <= lon <= maxlon and minlat <= lat <= maxlat:
                    cells.append(self.cell_at(q, r))
        return cells

    def area_distortion_pct(self, cells: Sequence[HexCell]) -> float:
        """Maximum relative area distortion (%) vs nominal hex area across cells."""
        if not cells:
            return 0.0
        return max(
            abs(c.area_km2 - self.nominal_area_km2) / self.nominal_area_km2 * 100.0
            for c in cells
        )

    def build_graph(
        self,
        bbox_lonlat: tuple[float, float, float, float],
        resistance_fn: Callable[[HexCell], float] | None = None,
        dispersal_scale_km: float = 100.0,
    ) -> HabitatGraph:
        """Build a spherical hexagonal adjacency HabitatGraph over a bounding box."""
        cells = self.generate_cells(bbox_lonlat)
        coord_map = {(c.q, c.r): c for c in cells}

        nodes: list[HabitatNode] = []
        res_map: dict[str, float] = {}
        for c in cells:
            res = float(resistance_fn(c)) if resistance_fn is not None else 1.0
            res_map[c.cell_id] = max(1e-6, res)
            ring = ", ".join(f"{lon:.6f} {lat:.6f}" for lon, lat in (*c.vertices_lonlat, c.vertices_lonlat[0]))
            nodes.append(
                HabitatNode(
                    node_id=c.cell_id,
                    kind=NodeKind.POLYGON,
                    coords=(c.lon, c.lat),
                    area=c.area_km2,
                    quality=1.0 / res_map[c.cell_id],
                    habitat_type="dggs_hex",
                    geometry_wkt=f"POLYGON (({ring}))",
                    attributes={"q": c.q, "r": c.r, "resolution": self.resolution},
                )
            )

        edges: list[HabitatEdge] = []
        edge_counter = 0
        for c in cells:
            for nq, nr in self.neighbor_axial_coords(c.q, c.r):
                nb = coord_map.get((nq, nr))
                if nb is None:
                    continue
                edge_counter += 1
                dist_km = haversine_km(c.lon, c.lat, nb.lon, nb.lat)
                cost = dist_km * 0.5 * (res_map[c.cell_id] + res_map[nb.cell_id])
                prob = math.exp(-cost / max(1e-6, dispersal_scale_km))
                edges.append(
                    HabitatEdge(
                        edge_id=f"de_{edge_counter}",
                        source=c.cell_id,
                        target=nb.cell_id,
                        weight=1.0 / max(1e-6, cost),
                        cost=cost,
                        probability=min(1.0, max(0.0, prob)),
                        distance=dist_km,
                        edge_class="intra",
                        process="dispersal",
                    )
                )

        meta = SpatialMetadata(
            crs="EPSG:4326",
            bbox=bbox_lonlat,
            resolution=(self.hex_radius_km,),
            dggs_type=f"equal_area_hex_r{self.resolution}",
            extra={"nominal_area_km2": self.nominal_area_km2},
        )
        return HabitatGraph(nodes, edges, meta)
