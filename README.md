# GeoGraph 2.0 (`geograph2`)

Clean-room, first-principles spatial ecology and landscape connectivity engine (MIT License).

## Architecture Overview (D1-P1 to D1-P8)

| Module | Package | Capabilities |
|---|---|---|
| `geograph2.ir`, `algebra`, `io` | **D1-P1** | Immutable `HabitatGraph` IR (point/polygon/voxel nodes, directed weighted multi-edges), Yirgacheffe-inspired lazy spatial algebra (auto intersection/union alignment, chunked evaluation, on-demand vector rasterization), GraphJSON & GeoParquet 1.1.0 I/O |
| `geograph2.dggs`, `layers3d` | **D1-P2** | Equal-area spherical hexagonal DGGS (<0.5% area distortion across equator to high latitudes), HEALPix/lat-lon area helpers, 2.5D stratified & 3D voxel adjacency graph builders |
| `geograph2.rsp`, `rsp_adjoint` | **D1-P3** | `scipy.sparse` RSP expected cost & free-energy solver ($\beta \to \infty$ LCP, $\beta \to 0^+$ circuit random walk), exact reverse-mode adjoint sensitivities ($\partial K/\partial Q$, $\partial K/\partial P$, $\partial K/\partial C$) verified vs central finite differences ($< 10^{-5}$), pluggable D4 Rust backend registry |
| `geograph2.metrics`, `connectivity`, `audit` | **D1-P4** | FRAGSTATS metrics (`NP, CA, TA, PLAND, ED, FRAC, ENN, CONTAG, SHDI`), `PC`/`IIC` with `Intra/Flux/Connector` node & edge rankings, `ECA/dECA`, `ProtConn/dProtConn`, GBF 30×30 Markdown/JSON audit engine & golden baseline verifier |
| `geograph2.embedding`, `manifold` | **D1-P5** | TESSERA `int8 + scale` 128D pure-NumPy Zarr-v3-compatible reader/writer, cosine resistance fields, semantic gradient boundaries, CPU few-shot suitability fitting & quantitative comparison vs hard land-cover classification |
| `geograph2.hetero`, `temporal` | **D1-P6** | Multi-habitat heterogeneous graphs (`source_sink`, `supplementation`, `complementation`), cross-epoch patch lineage tracking (`persist`, `split`, `merge`, `extinction`, `emergence`), temporal `dECA` curves |
| `tests/test_invariants.py`, `benchmarks/` | **D1-P7** | 45 ecological & graph-theoretic mathematical invariant property tests and reproducible offline performance benchmarks |
| `geograph2.cli`, `qgis_plugin/`, `mcp_skill/`, `examples/` | **D1-P8** | Modern CLI, QGIS Processing Provider, schema-validated MCP Agent Skill with geospatial contract, and 2 offline end-to-end case studies |

## Quick Start (Offline)

```bash
# Run full test suite (unittest)
PYTHONPATH="src:$PYTHONPATH" python3 -m unittest discover -s tests -v

# Syntax compilation check
python3 -m compileall -q src tests benchmarks qgis_plugin examples

# Run CLI case studies
PYTHONPATH="src:$PYTHONPATH" python3 -m geograph2.cli case protected-area-30x30
PYTHONPATH="src:$PYTHONPATH" python3 -m geograph2.cli case urban-green-connectivity

# Run offline performance benchmarks
PYTHONPATH="src:$PYTHONPATH" python3 benchmarks/run_benchmarks.py
```

## License & Clean-Room Statement

`geograph2` is authored from first principles under the **MIT License**. No code was copied or derived from `ai4er-cdt/geograph`, `AnacletoLAB/grape`, `ensmallen`, or any GPL-licensed software. Mathematical formulations follow published open literature (Saura & Pascual-Hortal 2007; Saura & Rubio 2010; Saura et al. 2017; Saerens et al. 2009; Kivimäki et al. 2014, 2024; Dales et al. 2025; Savary et al. 2024; Frazier et al. 2026).
