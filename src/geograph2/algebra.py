"""Declarative lazy spatial algebra inspired by Yirgacheffe (Dales et al. 2025).

Supports opaque RasterLayer and VectorLayer types, lazy expression trees that do not
materialize intermediate arrays until terminal reduction, automatic intersection/union
spatial alignment inference with explicit override, on-demand vector rasterization,
and chunked tile evaluation.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Callable, Iterable, Sequence

import numpy as np
from shapely.geometry.base import BaseGeometry
from shapely.prepared import prep


class AlignmentMode(str, Enum):
    """Spatial window alignment mode between layers."""

    INTERSECTION = "intersection"
    UNION = "union"


@dataclass(frozen=True)
class SpatialExtent:
    """Axis-aligned raster grid extent and pixel resolution."""

    minx: float
    miny: float
    maxx: float
    maxy: float
    pixel_size: float
    crs: str = "EPSG:3857"

    def __post_init__(self) -> None:
        if self.pixel_size <= 0:
            raise ValueError("pixel_size must be positive")
        if self.maxx <= self.minx or self.maxy <= self.miny:
            raise ValueError("Invalid extent bounds")

    @property
    def ncols(self) -> int:
        return int(round((self.maxx - self.minx) / self.pixel_size))

    @property
    def nrows(self) -> int:
        return int(round((self.maxy - self.miny) / self.pixel_size))

    @property
    def shape(self) -> tuple[int, int]:
        return (self.nrows, self.ncols)

    def intersect(self, other: SpatialExtent) -> SpatialExtent:
        if self.crs != other.crs:
            raise ValueError(f"CRS mismatch: {self.crs} vs {other.crs}")
        px = min(self.pixel_size, other.pixel_size)
        minx = max(self.minx, other.minx)
        miny = max(self.miny, other.miny)
        maxx = min(self.maxx, other.maxx)
        maxy = min(self.maxy, other.maxy)
        if maxx <= minx or maxy <= miny:
            raise ValueError("Spatial extents do not overlap")
        return SpatialExtent(minx, miny, maxx, maxy, px, self.crs)

    def union(self, other: SpatialExtent) -> SpatialExtent:
        if self.crs != other.crs:
            raise ValueError(f"CRS mismatch: {self.crs} vs {other.crs}")
        px = min(self.pixel_size, other.pixel_size)
        return SpatialExtent(
            min(self.minx, other.minx),
            min(self.miny, other.miny),
            max(self.maxx, other.maxx),
            max(self.maxy, other.maxy),
            px,
            self.crs,
        )


class SpatialExpr:
    """Base node in a lazy spatial algebra expression tree."""

    def __init__(self, extent: SpatialExtent) -> None:
        self._extent = extent
        self._explicit_alignment: AlignmentMode | None = None

    @property
    def extent(self) -> SpatialExtent:
        return self._extent

    def with_alignment(self, mode: str | AlignmentMode) -> SpatialExpr:
        """Override inferred alignment mode on a binary expression."""
        return AlignedExpr(self, AlignmentMode(mode))

    def eval_chunk(
        self,
        window: SpatialExtent,
        row_start: int,
        row_end: int,
        col_start: int,
        col_end: int,
    ) -> np.ndarray:
        raise NotImplementedError

    # --- Declarative operators (all lazy) ---

    def __mul__(self, other: SpatialExpr | float | int) -> SpatialExpr:
        return _make_binary(self, other, "mul", AlignmentMode.INTERSECTION)

    def __rmul__(self, other: float | int) -> SpatialExpr:
        return _make_binary(other, self, "mul", AlignmentMode.INTERSECTION)

    def __add__(self, other: SpatialExpr | float | int) -> SpatialExpr:
        return _make_binary(self, other, "add", AlignmentMode.UNION)

    def __radd__(self, other: float | int) -> SpatialExpr:
        return _make_binary(other, self, "add", AlignmentMode.UNION)

    def __sub__(self, other: SpatialExpr | float | int) -> SpatialExpr:
        return _make_binary(self, other, "sub", AlignmentMode.INTERSECTION)

    def __truediv__(self, other: SpatialExpr | float | int) -> SpatialExpr:
        return _make_binary(self, other, "div", AlignmentMode.INTERSECTION)

    def __and__(self, other: SpatialExpr) -> SpatialExpr:
        return _make_binary(self, other, "and", AlignmentMode.INTERSECTION)

    def __or__(self, other: SpatialExpr) -> SpatialExpr:
        return _make_binary(self, other, "or", AlignmentMode.UNION)

    def __ge__(self, other: SpatialExpr | float | int) -> SpatialExpr:
        return _make_binary(self, other, "ge", AlignmentMode.INTERSECTION)

    def __le__(self, other: SpatialExpr | float | int) -> SpatialExpr:
        return _make_binary(self, other, "le", AlignmentMode.INTERSECTION)

    def __gt__(self, other: SpatialExpr | float | int) -> SpatialExpr:
        return _make_binary(self, other, "gt", AlignmentMode.INTERSECTION)

    def __lt__(self, other: SpatialExpr | float | int) -> SpatialExpr:
        return _make_binary(self, other, "lt", AlignmentMode.INTERSECTION)

    def eq(self, value: float | int) -> SpatialExpr:
        return UnaryExpr(self, f"eq({value})", lambda a: (a == value).astype(np.float64))

    def isin(self, values: Iterable[int | float]) -> SpatialExpr:
        val_tuple = tuple(values)
        return UnaryExpr(
            self,
            f"isin({val_tuple})",
            lambda a: np.isin(a, val_tuple).astype(np.float64),
        )

    # --- Terminal reductions (trigger chunked evaluation) ---

    def _iter_chunks(
        self,
        chunk_size: int = 64,
        window: SpatialExtent | None = None,
    ) -> Iterable[tuple[int, int, int, int, np.ndarray]]:
        win = self.extent if window is None else window
        nrows, ncols = win.shape
        for r0 in range(0, nrows, chunk_size):
            r1 = min(r0 + chunk_size, nrows)
            for c0 in range(0, ncols, chunk_size):
                c1 = min(c0 + chunk_size, ncols)
                yield r0, r1, c0, c1, self.eval_chunk(win, r0, r1, c0, c1)

    def sum(self, chunk_size: int = 64, window: SpatialExtent | None = None) -> float:
        total = 0.0
        for _, _, _, _, arr in self._iter_chunks(chunk_size=chunk_size, window=window):
            total += float(np.nansum(arr))
        return total

    def mean(self, chunk_size: int = 64, window: SpatialExtent | None = None) -> float:
        total = 0.0
        count = 0
        for _, _, _, _, arr in self._iter_chunks(chunk_size=chunk_size, window=window):
            valid = ~np.isnan(arr)
            total += float(np.sum(arr[valid]))
            count += int(np.sum(valid))
        return total / count if count > 0 else float("nan")

    def to_array(self, chunk_size: int = 64, window: SpatialExtent | None = None) -> np.ndarray:
        win = self.extent if window is None else window
        out = np.empty(win.shape, dtype=np.float64)
        for r0, r1, c0, c1, arr in self._iter_chunks(chunk_size=chunk_size, window=window):
            out[r0:r1, c0:c1] = arr
        return out


class RasterLayer(SpatialExpr):
    """Opaque raster dataset evaluated on demand by spatial window chunks."""

    def __init__(
        self,
        data: np.ndarray,
        extent: SpatialExtent,
        nodata: float = 0.0,
        name: str = "raster",
    ) -> None:
        arr = np.asarray(data, dtype=np.float64)
        if arr.ndim != 2:
            raise ValueError("RasterLayer data must be 2D")
        if arr.shape != extent.shape:
            raise ValueError(f"Data shape {arr.shape} does not match extent shape {extent.shape}")
        super().__init__(extent)
        self._data = arr
        self._nodata = float(nodata)
        self.name = name
        self.chunks_read = 0

    def eval_chunk(
        self,
        window: SpatialExtent,
        row_start: int,
        row_end: int,
        col_start: int,
        col_end: int,
    ) -> np.ndarray:
        self.chunks_read += 1
        h = row_end - row_start
        w = col_end - col_start
        # Map chunk pixel centers in window coordinates to source raster indices
        xs = window.minx + (np.arange(col_start, col_end) + 0.5) * window.pixel_size
        ys = window.maxy - (np.arange(row_start, row_end) + 0.5) * window.pixel_size

        src_cols = np.floor((xs - self._extent.minx) / self._extent.pixel_size).astype(np.int64)
        src_rows = np.floor((self._extent.maxy - ys) / self._extent.pixel_size).astype(np.int64)

        valid_r = (src_rows >= 0) & (src_rows < self._data.shape[0])
        valid_c = (src_cols >= 0) & (src_cols < self._data.shape[1])

        out = np.full((h, w), self._nodata, dtype=np.float64)
        if np.any(valid_r) and np.any(valid_c):
            rr = np.where(valid_r)[0]
            cc = np.where(valid_c)[0]
            out[np.ix_(rr, cc)] = self._data[np.ix_(src_rows[valid_r], src_cols[valid_c])]
        return out


class VectorLayer(SpatialExpr):
    """Opaque vector polygon layer rasterized on demand per evaluated chunk."""

    def __init__(
        self,
        geometries: Sequence[BaseGeometry],
        extent: SpatialExtent,
        burn_value: float = 1.0,
        name: str = "vector",
    ) -> None:
        super().__init__(extent)
        self._geometries = tuple(geometries)
        self._prepared = tuple(prep(g) for g in self._geometries)
        self._burn_value = float(burn_value)
        self.name = name
        self.chunks_rasterized = 0

    def eval_chunk(
        self,
        window: SpatialExtent,
        row_start: int,
        row_end: int,
        col_start: int,
        col_end: int,
    ) -> np.ndarray:
        from shapely.geometry import Point

        self.chunks_rasterized += 1
        h = row_end - row_start
        w = col_end - col_start
        out = np.zeros((h, w), dtype=np.float64)
        xs = window.minx + (np.arange(col_start, col_end) + 0.5) * window.pixel_size
        ys = window.maxy - (np.arange(row_start, row_end) + 0.5) * window.pixel_size
        for r_idx, y in enumerate(ys):
            for c_idx, x in enumerate(xs):
                pt = Point(float(x), float(y))
                if any(pg.covers(pt) for pg in self._prepared):
                    out[r_idx, c_idx] = self._burn_value
        return out


class ScalarExpr(SpatialExpr):
    """Constant scalar lifted into a spatial expression."""

    def __init__(self, value: float, extent: SpatialExtent) -> None:
        super().__init__(extent)
        self._value = float(value)

    def eval_chunk(
        self,
        window: SpatialExtent,
        row_start: int,
        row_end: int,
        col_start: int,
        col_end: int,
    ) -> np.ndarray:
        return np.full((row_end - row_start, col_end - col_start), self._value, dtype=np.float64)


class UnaryExpr(SpatialExpr):
    """Lazy unary operation over a SpatialExpr."""

    def __init__(
        self,
        operand: SpatialExpr,
        op_name: str,
        fn: Callable[[np.ndarray], np.ndarray],
    ) -> None:
        super().__init__(operand.extent)
        self.operand = operand
        self.op_name = op_name
        self._fn = fn

    def eval_chunk(
        self,
        window: SpatialExtent,
        row_start: int,
        row_end: int,
        col_start: int,
        col_end: int,
    ) -> np.ndarray:
        arr = self.operand.eval_chunk(window, row_start, row_end, col_start, col_end)
        return np.asarray(self._fn(arr), dtype=np.float64)


class BinaryExpr(SpatialExpr):
    """Lazy binary operation with automatic intersection/union alignment."""

    _OPS: dict[str, Callable[[np.ndarray, np.ndarray], np.ndarray]] = {
        "mul": lambda a, b: a * b,
        "add": lambda a, b: a + b,
        "sub": lambda a, b: a - b,
        "div": lambda a, b: np.where(b != 0.0, a / b, np.nan),
        "and": lambda a, b: ((a != 0.0) & (b != 0.0)).astype(np.float64),
        "or": lambda a, b: ((a != 0.0) | (b != 0.0)).astype(np.float64),
        "ge": lambda a, b: (a >= b).astype(np.float64),
        "le": lambda a, b: (a <= b).astype(np.float64),
        "gt": lambda a, b: (a > b).astype(np.float64),
        "lt": lambda a, b: (a < b).astype(np.float64),
    }

    def __init__(
        self,
        left: SpatialExpr,
        right: SpatialExpr,
        op_name: str,
        alignment: AlignmentMode,
    ) -> None:
        self.left = left
        self.right = right
        self.op_name = op_name
        self.alignment = alignment
        extent = (
            left.extent.intersect(right.extent)
            if alignment == AlignmentMode.INTERSECTION
            else left.extent.union(right.extent)
        )
        super().__init__(extent)

    def eval_chunk(
        self,
        window: SpatialExtent,
        row_start: int,
        row_end: int,
        col_start: int,
        col_end: int,
    ) -> np.ndarray:
        a = self.left.eval_chunk(window, row_start, row_end, col_start, col_end)
        b = self.right.eval_chunk(window, row_start, row_end, col_start, col_end)
        return np.asarray(self._OPS[self.op_name](a, b), dtype=np.float64)


class AlignedExpr(SpatialExpr):
    """Explicit alignment override wrapper for a BinaryExpr."""

    def __init__(self, expr: SpatialExpr, mode: AlignmentMode) -> None:
        if isinstance(expr, BinaryExpr):
            rebuilt = BinaryExpr(expr.left, expr.right, expr.op_name, mode)
            self._inner: SpatialExpr = rebuilt
            super().__init__(rebuilt.extent)
        else:
            self._inner = expr
            super().__init__(expr.extent)

    def eval_chunk(
        self,
        window: SpatialExtent,
        row_start: int,
        row_end: int,
        col_start: int,
        col_end: int,
    ) -> np.ndarray:
        return self._inner.eval_chunk(window, row_start, row_end, col_start, col_end)


class WhereExpr(SpatialExpr):
    """Lazy element-wise conditional ``where(cond, x, y)``."""

    def __init__(
        self,
        cond: SpatialExpr,
        x: SpatialExpr | float,
        y: SpatialExpr | float,
    ) -> None:
        self.cond = cond
        self.x = x if isinstance(x, SpatialExpr) else ScalarExpr(float(x), cond.extent)
        self.y = y if isinstance(y, SpatialExpr) else ScalarExpr(float(y), cond.extent)
        ext = cond.extent.intersect(self.x.extent).intersect(self.y.extent)
        super().__init__(ext)

    def eval_chunk(
        self,
        window: SpatialExtent,
        row_start: int,
        row_end: int,
        col_start: int,
        col_end: int,
    ) -> np.ndarray:
        c = self.cond.eval_chunk(window, row_start, row_end, col_start, col_end)
        xv = self.x.eval_chunk(window, row_start, row_end, col_start, col_end)
        yv = self.y.eval_chunk(window, row_start, row_end, col_start, col_end)
        return np.where(c != 0.0, xv, yv).astype(np.float64)


def where(cond: SpatialExpr, x: SpatialExpr | float, y: SpatialExpr | float) -> SpatialExpr:
    """Declarative lazy conditional expression."""
    return WhereExpr(cond, x, y)


def _make_binary(
    left: SpatialExpr | float | int,
    right: SpatialExpr | float | int,
    op_name: str,
    alignment: AlignmentMode,
) -> BinaryExpr:
    if isinstance(left, SpatialExpr) and isinstance(right, SpatialExpr):
        return BinaryExpr(left, right, op_name, alignment)
    if isinstance(left, SpatialExpr):
        return BinaryExpr(left, ScalarExpr(float(right), left.extent), op_name, alignment)
    if isinstance(right, SpatialExpr):
        return BinaryExpr(ScalarExpr(float(left), right.extent), right, op_name, alignment)
    raise TypeError("At least one operand must be a SpatialExpr")
