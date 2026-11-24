"""TESSERA int8 + scale 128D continuous embedding cube reader/writer.

Provides a pure-numpy / zipfile / json Zarr-v3-compatible archive reader & writer
(for offline environments without the ``zarr`` package) alongside TESSERA NPY
``int8 + float32 scale`` dequantization (``embeddings = quantized.astype(float32) * scale``).
"""

from __future__ import annotations

import io
import json
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class TesseraCube:
    """128-dimensional (or D-dimensional) continuous earth-observation embedding cube."""

    embeddings: np.ndarray  # shape (H, W, D), float32
    pixel_size_m: float = 10.0
    crs: str = "EPSG:3857"
    version: str = "1.1"
    variant: str = "dclimate"
    year: int = 2024
    metadata: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        arr = np.asarray(self.embeddings, dtype=np.float32)
        if arr.ndim != 3:
            raise ValueError("embeddings must have shape (H, W, D)")
        object.__setattr__(self, "embeddings", arr)

    @property
    def height(self) -> int:
        return int(self.embeddings.shape[0])

    @property
    def width(self) -> int:
        return int(self.embeddings.shape[1])

    @property
    def dim(self) -> int:
        return int(self.embeddings.shape[2])


def quantize_tessera_int8(embeddings: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Quantize float32 (H, W, D) embeddings to TESSERA int8 + per-pixel float32 scale."""
    arr = np.asarray(embeddings, dtype=np.float32)
    max_abs = np.max(np.abs(arr), axis=-1, keepdims=True)
    scales = np.where(max_abs > 0.0, max_abs / 127.0, 1.0).astype(np.float32)
    q = np.clip(np.rint(arr / scales), -127, 127).astype(np.int8)
    return q, scales[..., 0]


def dequantize_tessera_int8(quantized: np.ndarray, scales: np.ndarray) -> np.ndarray:
    """Dequantize TESSERA int8 embeddings using per-pixel or broadcastable float32 scales."""
    q = np.asarray(quantized, dtype=np.int8)
    s = np.asarray(scales, dtype=np.float32)
    if s.ndim == 2 and q.ndim == 3:
        s = s[..., np.newaxis]
    return q.astype(np.float32) * s


def save_tessera_zarr_compat(cube: TesseraCube, archive_path: str | Path) -> Path:
    """Save a TesseraCube into a pure-stdlib Zarr-v3-compatible zip store (int8 + scale).

    Structure inside archive:
    - ``zarr.json`` (Zarr v3 root group metadata + geograph2/TESSERA attributes)
    - ``embeddings/zarr.json`` + ``embeddings/c/0/0/0`` (raw little-endian int8 bytes)
    - ``scales/zarr.json`` + ``scales/c/0/0`` (raw little-endian float32 bytes)
    """
    out = Path(archive_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    q, scales = quantize_tessera_int8(cube.embeddings)

    root_meta = {
        "zarr_format": 3,
        "node_type": "group",
        "attributes": {
            "provider": "geograph2.tessera_zarr_compat",
            "version": cube.version,
            "variant": cube.variant,
            "year": cube.year,
            "pixel_size_m": cube.pixel_size_m,
            "crs": cube.crs,
            "extra": cube.metadata or {},
        },
    }
    emb_meta = {
        "zarr_format": 3,
        "node_type": "array",
        "shape": list(q.shape),
        "data_type": "int8",
        "chunk_grid": {"name": "regular", "configuration": {"chunk_shape": list(q.shape)}},
        "codecs": [{"name": "bytes", "configuration": {"endian": "little"}}],
    }
    scale_meta = {
        "zarr_format": 3,
        "node_type": "array",
        "shape": list(scales.shape),
        "data_type": "float32",
        "chunk_grid": {"name": "regular", "configuration": {"chunk_shape": list(scales.shape)}},
        "codecs": [{"name": "bytes", "configuration": {"endian": "little"}}],
    }

    with zipfile.ZipFile(out, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("zarr.json", json.dumps(root_meta, indent=2))
        zf.writestr("embeddings/zarr.json", json.dumps(emb_meta, indent=2))
        zf.writestr("embeddings/c/0/0/0", np.ascontiguousarray(q, dtype="<i1").tobytes())
        zf.writestr("scales/zarr.json", json.dumps(scale_meta, indent=2))
        zf.writestr("scales/c/0/0", np.ascontiguousarray(scales, dtype="<f4").tobytes())
    return out


def load_tessera_zarr_compat(archive_path: str | Path) -> TesseraCube:
    """Read a Zarr-v3-compatible TESSERA int8 + scale zip store without requiring ``zarr``."""
    with zipfile.ZipFile(Path(archive_path), "r") as zf:
        root_meta = json.loads(zf.read("zarr.json").decode("utf-8"))
        emb_meta = json.loads(zf.read("embeddings/zarr.json").decode("utf-8"))
        scale_meta = json.loads(zf.read("scales/zarr.json").decode("utf-8"))

        q_bytes = zf.read("embeddings/c/0/0/0")
        s_bytes = zf.read("scales/c/0/0")

    q = np.frombuffer(q_bytes, dtype="<i1").reshape(tuple(emb_meta["shape"]))
    scales = np.frombuffer(s_bytes, dtype="<f4").reshape(tuple(scale_meta["shape"]))
    emb = dequantize_tessera_int8(q, scales)
    attrs = root_meta.get("attributes", {})
    return TesseraCube(
        embeddings=emb,
        pixel_size_m=float(attrs.get("pixel_size_m", 10.0)),
        crs=str(attrs.get("crs", "EPSG:3857")),
        version=str(attrs.get("version", "1.1")),
        variant=str(attrs.get("variant", "dclimate")),
        year=int(attrs.get("year", 2024)),
        metadata=attrs.get("extra", {}),
    )


def load_tessera_npy_pair(
    quantized_npy: str | Path,
    scales_npy: str | Path,
    pixel_size_m: float = 10.0,
) -> TesseraCube:
    """Load TESSERA legacy NPY tile pair (``grid.npy`` int8 + ``grid_scales.npy`` float32)."""
    q = np.load(Path(quantized_npy))
    s = np.load(Path(scales_npy))
    return TesseraCube(embeddings=dequantize_tessera_int8(q, s), pixel_size_m=pixel_size_m)


def generate_synthetic_tessera_cube(
    height: int = 24,
    width: int = 24,
    dim: int = 128,
    seed: int = 42,
) -> tuple[TesseraCube, np.ndarray, np.ndarray]:
    """Generate a reproducible synthetic 128D TESSERA landscape with soft ecotone gradients.

    Returns ``(cube, true_continuous_suitability, hard_landcover_classes)``.
    """
    rng = np.random.default_rng(seed)
    # Three distinct habitat prototypes in 128D unit sphere: core forest, ecotone, urban matrix
    proto_forest = rng.normal(size=dim).astype(np.float32)
    proto_forest /= np.linalg.norm(proto_forest)
    proto_urban = rng.normal(size=dim).astype(np.float32)
    proto_urban -= np.dot(proto_urban, proto_forest) * proto_forest
    proto_urban /= np.linalg.norm(proto_urban)

    yy, xx = np.mgrid[0:height, 0:width].astype(np.float32)
    # Continuous habitat suitability gradient with two core patches and a corridor
    g1 = np.exp(-((yy - height * 0.25) ** 2 + (xx - width * 0.25) ** 2) / (0.09 * height * width))
    g2 = np.exp(-((yy - height * 0.75) ** 2 + (xx - width * 0.75) ** 2) / (0.09 * height * width))
    corridor = 0.55 * np.exp(-((yy - xx) ** 2) / (0.02 * height * width))
    true_suit = np.clip(np.maximum(np.maximum(g1, g2), corridor), 0.0, 1.0).astype(np.float32)

    emb = (
        true_suit[..., None] * proto_forest[None, None, :]
        + (1.0 - true_suit[..., None]) * proto_urban[None, None, :]
        + 0.04 * rng.normal(size=(height, width, dim)).astype(np.float32)
    )
    # Hard 3-class discretization loses continuous ecotone gradation
    hard_lc = np.where(true_suit >= 0.65, 2, np.where(true_suit >= 0.30, 1, 0)).astype(np.int32)
    cube = TesseraCube(embeddings=emb, pixel_size_m=10.0, version="1.1", variant="dclimate", year=2024)
    return cube, true_suit, hard_lc
