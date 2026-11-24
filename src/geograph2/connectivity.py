"""Graph-based connectivity indices: PC, IIC, Intra/Flux/Connector, ECA/dECA, ProtConn/dProtConn.

Implements clean-room formulations following Saura & Pascual-Hortal (2007),
Pascual-Hortal & Saura (2006), Saura & Rubio (2010), Saura et al. (2011),
and Saura et al. (2017, ProtConn).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy import sparse
from scipy.sparse.csgraph import shortest_path

from geograph2.ir import HabitatGraph


@dataclass(frozen=True)
class NodeImportance:
    """Node-level connectivity contribution and three-component decomposition (%)."""

    node_id: str
    d_total: float
    d_intra: float
    d_flux: float
    d_connector: float


@dataclass(frozen=True)
class EdgeImportance:
    """Edge-level connectivity contribution (%)."""

    edge_id: str
    source: str
    target: str
    d_total: float


@dataclass(frozen=True)
class ConnectivityIndexResult:
    """Full PC or IIC evaluation with per-node and per-edge importance rankings."""

    metric: str
    value: float
    numerator: float
    landscape_area: float
    eca: float
    node_importance: tuple[NodeImportance, ...]
    edge_importance: tuple[EdgeImportance, ...]


@dataclass(frozen=True)
class ProtConnResult:
    """GBF 30x30 Protected Connected (ProtConn) evaluation and decomposition (%)."""

    region_area: float
    protected_area: float
    prot_pct: float
    protconn_pct: float
    protunconn_pct: float
    relconn_pct: float
    eca_protected: float
    protconn_within_pct: float
    protconn_contig_pct: float
    protconn_unprot_pct: float
    node_dprotconn: dict[str, float]


def _max_product_probability_matrix(graph: HabitatGraph) -> np.ndarray:
    """Compute exact all-pairs maximum product probability p*_ij via -log(p) shortest paths."""
    n = graph.num_nodes
    if n == 0:
        return np.zeros((0, 0), dtype=np.float64)

    P_direct = graph.to_sparse_matrix("probability", reduce="max").tocoo()
    # Convert positive probabilities to non-negative -log(p) weights
    mask = P_direct.data > 0.0
    rows = P_direct.row[mask]
    cols = P_direct.col[mask]
    neg_log = -np.log(np.clip(P_direct.data[mask], 1e-300, 1.0))
    W = sparse.csr_matrix((neg_log, (rows, cols)), shape=(n, n), dtype=np.float64)

    dist = shortest_path(W, directed=True, method="D")
    p_star = np.where(np.isfinite(dist), np.exp(-dist), 0.0)
    np.fill_diagonal(p_star, 1.0)
    return p_star


def _iic_hop_matrix(graph: HabitatGraph) -> np.ndarray:
    """Compute shortest topological hop count nl_ij on binary links (probability > 0)."""
    n = graph.num_nodes
    if n == 0:
        return np.zeros((0, 0), dtype=np.float64)
    P_direct = graph.to_sparse_matrix("probability", reduce="max").tocoo()
    mask = P_direct.data > 0.0
    ones = np.ones(int(np.sum(mask)), dtype=np.float64)
    B = sparse.csr_matrix((ones, (P_direct.row[mask], P_direct.col[mask])), shape=(n, n), dtype=np.float64)
    hops = shortest_path(B, directed=True, unweighted=True)
    np.fill_diagonal(hops, 0.0)
    return hops


def _pc_numerator(areas: np.ndarray, p_star: np.ndarray) -> float:
    return float(areas @ p_star @ areas)


def _iic_numerator(areas: np.ndarray, hops: np.ndarray) -> float:
    inv_term = np.where(np.isfinite(hops), 1.0 / (1.0 + hops), 0.0)
    return float(areas @ inv_term @ areas)


def compute_pc(
    graph: HabitatGraph,
    landscape_area: float | None = None,
    compute_node_importance: bool = True,
    compute_edge_importance: bool = True,
) -> ConnectivityIndexResult:
    """Compute Probability of Connectivity (PC), ECA, and Intra/Flux/Connector rankings."""
    areas = graph.node_areas()
    al = float(landscape_area) if landscape_area is not None else float(np.sum(areas))
    if al <= 0.0:
        raise ValueError("landscape_area must be positive")

    p_star = _max_product_probability_matrix(graph)
    pc_num = _pc_numerator(areas, p_star)
    pc_val = pc_num / (al * al)
    eca = math.sqrt(max(0.0, pc_num))

    node_imps: list[NodeImportance] = []
    if compute_node_importance and graph.num_nodes > 0:
        for idx, nid in enumerate(graph.node_ids):
            if pc_num <= 0.0:
                node_imps.append(NodeImportance(nid, 0.0, 0.0, 0.0, 0.0))
                continue
            g_minus = graph.without_node(nid)
            num_minus = _pc_numerator(g_minus.node_areas(), _max_product_probability_matrix(g_minus))
            d_total = 100.0 * (pc_num - num_minus) / pc_num

            a_k = float(areas[idx])
            intra_num = a_k * a_k
            # Flux: all paths starting or ending at k (excluding k->k)
            out_flux = float(a_k * np.sum(areas * p_star[idx, :]) - intra_num)
            in_flux = float(a_k * np.sum(areas * p_star[:, idx]) - intra_num)
            flux_num = out_flux + in_flux

            d_intra = 100.0 * intra_num / pc_num
            d_flux = 100.0 * flux_num / pc_num
            d_conn = max(0.0, d_total - d_intra - d_flux)
            node_imps.append(NodeImportance(nid, d_total, d_intra, d_flux, d_conn))

        node_imps.sort(key=lambda r: r.d_total, reverse=True)

    edge_imps: list[EdgeImportance] = []
    if compute_edge_importance and graph.num_edges > 0:
        for edge in graph.edges:
            if pc_num <= 0.0:
                edge_imps.append(EdgeImportance(edge.edge_id, edge.source, edge.target, 0.0))
                continue
            g_minus = graph.without_edge(edge.edge_id)
            num_minus = _pc_numerator(g_minus.node_areas(), _max_product_probability_matrix(g_minus))
            d_edge = 100.0 * (pc_num - num_minus) / pc_num
            edge_imps.append(EdgeImportance(edge.edge_id, edge.source, edge.target, max(0.0, d_edge)))

        edge_imps.sort(key=lambda r: r.d_total, reverse=True)

    return ConnectivityIndexResult(
        metric="PC",
        value=pc_val,
        numerator=pc_num,
        landscape_area=al,
        eca=eca,
        node_importance=tuple(node_imps),
        edge_importance=tuple(edge_imps),
    )


def compute_iic(
    graph: HabitatGraph,
    landscape_area: float | None = None,
    compute_node_importance: bool = True,
    compute_edge_importance: bool = True,
) -> ConnectivityIndexResult:
    """Compute Integral Index of Connectivity (IIC), ECA_IIC, and Intra/Flux/Connector rankings."""
    areas = graph.node_areas()
    al = float(landscape_area) if landscape_area is not None else float(np.sum(areas))
    if al <= 0.0:
        raise ValueError("landscape_area must be positive")

    hops = _iic_hop_matrix(graph)
    iic_num = _iic_numerator(areas, hops)
    iic_val = iic_num / (al * al)
    eca = math.sqrt(max(0.0, iic_num))
    inv_term = np.where(np.isfinite(hops), 1.0 / (1.0 + hops), 0.0)

    node_imps: list[NodeImportance] = []
    if compute_node_importance and graph.num_nodes > 0:
        for idx, nid in enumerate(graph.node_ids):
            if iic_num <= 0.0:
                node_imps.append(NodeImportance(nid, 0.0, 0.0, 0.0, 0.0))
                continue
            g_minus = graph.without_node(nid)
            num_minus = _iic_numerator(g_minus.node_areas(), _iic_hop_matrix(g_minus))
            d_total = 100.0 * (iic_num - num_minus) / iic_num

            a_k = float(areas[idx])
            intra_num = a_k * a_k
            out_flux = float(a_k * np.sum(areas * inv_term[idx, :]) - intra_num)
            in_flux = float(a_k * np.sum(areas * inv_term[:, idx]) - intra_num)
            flux_num = out_flux + in_flux

            d_intra = 100.0 * intra_num / iic_num
            d_flux = 100.0 * flux_num / iic_num
            d_conn = max(0.0, d_total - d_intra - d_flux)
            node_imps.append(NodeImportance(nid, d_total, d_intra, d_flux, d_conn))

        node_imps.sort(key=lambda r: r.d_total, reverse=True)

    edge_imps: list[EdgeImportance] = []
    if compute_edge_importance and graph.num_edges > 0:
        for edge in graph.edges:
            if iic_num <= 0.0:
                edge_imps.append(EdgeImportance(edge.edge_id, edge.source, edge.target, 0.0))
                continue
            g_minus = graph.without_edge(edge.edge_id)
            num_minus = _iic_numerator(g_minus.node_areas(), _iic_hop_matrix(g_minus))
            d_edge = 100.0 * (iic_num - num_minus) / iic_num
            edge_imps.append(EdgeImportance(edge.edge_id, edge.source, edge.target, max(0.0, d_edge)))

        edge_imps.sort(key=lambda r: r.d_total, reverse=True)

    return ConnectivityIndexResult(
        metric="IIC",
        value=iic_val,
        numerator=iic_num,
        landscape_area=al,
        eca=eca,
        node_importance=tuple(node_imps),
        edge_importance=tuple(edge_imps),
    )


def compute_deca(eca_t1: float, eca_t2: float) -> dict[str, float]:
    """Compute absolute and relative Equivalent Connected Area change (dECA, rdECA)."""
    deca = float(eca_t2) - float(eca_t1)
    rdeca = (100.0 * deca / float(eca_t1)) if eca_t1 > 0.0 else 0.0
    return {"eca_t1": float(eca_t1), "eca_t2": float(eca_t2), "dECA": deca, "rdECA_pct": rdeca}


def compute_protconn(
    graph: HabitatGraph,
    region_area: float,
    compute_dprotconn: bool = True,
) -> ProtConnResult:
    """Compute ProtConn, ProtUnconn, RelConn, fractions, and per-PA dProtConn (%).

    Protected patches are identified by ``node.protected == True``. Unprotected nodes
    in the graph act as stepping stones for ``ProtConn[Unprot]``.
    """
    if region_area <= 0.0:
        raise ValueError("region_area must be positive")

    prot_mask = np.array([n.protected for n in graph.iter_nodes()], dtype=bool)
    areas = graph.node_areas()
    prot_areas = np.where(prot_mask, areas, 0.0)
    total_prot_area = float(np.sum(prot_areas))
    prot_pct = 100.0 * total_prot_area / region_area

    if total_prot_area <= 0.0:
        return ProtConnResult(
            region_area=region_area,
            protected_area=0.0,
            prot_pct=0.0,
            protconn_pct=0.0,
            protunconn_pct=0.0,
            relconn_pct=0.0,
            eca_protected=0.0,
            protconn_within_pct=0.0,
            protconn_contig_pct=0.0,
            protconn_unprot_pct=0.0,
            node_dprotconn={},
        )

    # Full graph p* (allows unprotected stepping stones)
    p_star_full = _max_product_probability_matrix(graph)
    pc_num_full = float(prot_areas @ p_star_full @ prot_areas)
    eca_full = math.sqrt(max(0.0, pc_num_full))
    protconn_pct = 100.0 * eca_full / region_area
    protunconn_pct = max(0.0, prot_pct - protconn_pct)
    relconn_pct = 100.0 * protconn_pct / prot_pct if prot_pct > 0.0 else 0.0

    # Within-PA fraction (no inter-PA movement): sqrt(sum a_i^2) / A_region
    eca_within = math.sqrt(float(np.sum(prot_areas * prot_areas)))
    within_pct = 100.0 * eca_within / region_area

    # PA-only subgraph (no unprotected stepping stones)
    prot_nodes = [n for n in graph.iter_nodes() if n.protected]
    prot_ids = {n.node_id for n in prot_nodes}
    prot_edges = [e for e in graph.edges if e.source in prot_ids and e.target in prot_ids]
    g_prot_only = HabitatGraph(prot_nodes, prot_edges, graph.metadata)
    p_star_prot = _max_product_probability_matrix(g_prot_only)
    pa_only_areas = g_prot_only.node_areas()
    eca_prot_only = math.sqrt(max(0.0, float(pa_only_areas @ p_star_prot @ pa_only_areas)))
    prot_only_pct = 100.0 * eca_prot_only / region_area

    contig_pct = max(0.0, prot_only_pct - within_pct)
    unprot_step_pct = max(0.0, protconn_pct - prot_only_pct)

    dprotconn: dict[str, float] = {}
    if compute_dprotconn and protconn_pct > 0.0:
        for node in prot_nodes:
            g_minus = graph.without_node(node.node_id)
            m_mask = np.array([n.protected for n in g_minus.iter_nodes()], dtype=bool)
            m_areas = np.where(m_mask, g_minus.node_areas(), 0.0)
            if np.sum(m_areas) <= 0.0:
                eca_m = 0.0
            else:
                p_m = _max_product_probability_matrix(g_minus)
                eca_m = math.sqrt(max(0.0, float(m_areas @ p_m @ m_areas)))
            pc_m = 100.0 * eca_m / region_area
            dprotconn[node.node_id] = max(0.0, 100.0 * (protconn_pct - pc_m) / protconn_pct)

    return ProtConnResult(
        region_area=region_area,
        protected_area=total_prot_area,
        prot_pct=prot_pct,
        protconn_pct=protconn_pct,
        protunconn_pct=protunconn_pct,
        relconn_pct=relconn_pct,
        eca_protected=eca_full,
        protconn_within_pct=within_pct,
        protconn_contig_pct=contig_pct,
        protconn_unprot_pct=unprot_step_pct,
        node_dprotconn=dprotconn,
    )
