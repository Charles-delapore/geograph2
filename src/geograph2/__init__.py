"""Clean-Room GeoGraph 2.0 (geograph2).

Unified spatial habitat topology IR, declarative lazy spatial algebra,
equal-area spherical hexagonal DGGS & 2.5D/3D voxel graphs, scipy.sparse RSP
expected cost & adjoint analytical sensitivity solver, FRAGSTATS & GBF 30x30
connectivity audit engine, TESSERA 128D continuous embedding manifold,
multi-habitat heterogeneous & temporal lineage dynamics, CLI, QGIS & MCP interfaces.
"""

from geograph2.algebra import AlignmentMode, RasterLayer, SpatialExtent, VectorLayer, where
from geograph2.audit import GBF30x30AuditReport, run_gbf_30x30_audit, verify_golden_baseline_files
from geograph2.connectivity import (
    compute_deca,
    compute_iic,
    compute_pc,
    compute_protconn,
)
from geograph2.dggs import SphericalHexDGGS, haversine_km, healpix_cell_area_km2
from geograph2.embedding import (
    TesseraCube,
    generate_synthetic_tessera_cube,
    load_tessera_zarr_compat,
    save_tessera_zarr_compat,
)
from geograph2.hetero import EcologicalProcess, analyze_hetero_processes, build_multi_habitat_graph
from geograph2.io import (
    load_geoparquet,
    load_graphjson,
    save_geoparquet,
    save_graphjson,
)
from geograph2.ir import HabitatEdge, HabitatGraph, HabitatNode, NodeKind, SpatialMetadata
from geograph2.layers3d import build_stratified_25d_graph, build_voxel_3d_graph
from geograph2.manifold import (
    compare_continuous_vs_hard_classification,
    cosine_resistance_field,
    fit_few_shot_suitability,
    semantic_gradient_boundaries,
)
from geograph2.metrics import compute_landscape_metrics
from geograph2.rsp import (
    build_reference_transition,
    effective_resistance_matrix,
    least_cost_path_matrix,
    solve_rsp,
)
from geograph2.rsp_adjoint import (
    available_rsp_backends,
    register_rsp_backend,
    rsp_adjoint_sensitivity,
    set_rsp_backend,
)
from geograph2.temporal import (
    LineageEventType,
    compute_temporal_deca_curve,
    track_patch_lineage,
)

__version__ = "0.1.0"

__all__ = [
    "__version__",
    "AlignmentMode",
    "EcologicalProcess",
    "GBF30x30AuditReport",
    "HabitatEdge",
    "HabitatGraph",
    "HabitatNode",
    "LineageEventType",
    "NodeKind",
    "RasterLayer",
    "SpatialExtent",
    "SpatialMetadata",
    "SphericalHexDGGS",
    "TesseraCube",
    "VectorLayer",
    "analyze_hetero_processes",
    "available_rsp_backends",
    "build_multi_habitat_graph",
    "build_reference_transition",
    "build_stratified_25d_graph",
    "build_voxel_3d_graph",
    "compare_continuous_vs_hard_classification",
    "compute_deca",
    "compute_iic",
    "compute_landscape_metrics",
    "compute_pc",
    "compute_protconn",
    "compute_temporal_deca_curve",
    "cosine_resistance_field",
    "effective_resistance_matrix",
    "fit_few_shot_suitability",
    "generate_synthetic_tessera_cube",
    "haversine_km",
    "healpix_cell_area_km2",
    "least_cost_path_matrix",
    "load_geoparquet",
    "load_graphjson",
    "load_tessera_zarr_compat",
    "register_rsp_backend",
    "rsp_adjoint_sensitivity",
    "run_gbf_30x30_audit",
    "save_geoparquet",
    "save_graphjson",
    "save_tessera_zarr_compat",
    "semantic_gradient_boundaries",
    "set_rsp_backend",
    "solve_rsp",
    "track_patch_lineage",
    "verify_golden_baseline_files",
    "where",
]
