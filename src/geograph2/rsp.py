"""Randomized Shortest Paths (RSP) expected cost and free-energy distance solver.

Implements sparse matrix RSP inference (Saerens et al. 2009; Kivimaki et al. 2014;
Van Moorter et al. 2023) using scipy.sparse LU factorization. Supports both ``beta``
and ``theta`` naming for the inverse temperature parameter:
- beta -> inf converges to Least-Cost Path (LCP) distance.
- beta -> 0+ converges to pure reference random-walk expected hitting cost
  (and commute-time / effective resistance on symmetric graphs).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import sparse
from scipy.sparse.csgraph import shortest_path
from scipy.sparse.linalg import splu

from geograph2.ir import HabitatGraph


@dataclass(frozen=True)
class RSPResult:
    """Outputs of the RSP expected-cost solver."""

    beta: float
    expected_cost: np.ndarray
    free_energy_distance: np.ndarray
    fundamental_matrix: np.ndarray
    weighted_transition: sparse.csr_matrix
    transition_matrix: sparse.csr_matrix
    cost_matrix: sparse.csr_matrix
    node_ids: tuple[str, ...]


def build_reference_transition(
    graph: HabitatGraph,
    cost_matrix: sparse.csr_matrix | None = None,
) -> tuple[sparse.csr_matrix, sparse.csr_matrix]:
    """Extract (P, C) sparse matrices from a HabitatGraph.

    P is the row-stochastic reference random-walk transition matrix built from
    edge weights (or inverse costs if weights are uniform). C is the non-negative
    edge cost matrix.
    """
    C = graph.to_sparse_matrix("cost", reduce="min") if cost_matrix is None else cost_matrix.tocsr()
    W_raw = graph.to_sparse_matrix("weight", reduce="sum").tocsr()
    n = graph.num_nodes
    if W_raw.nnz == 0 and C.nnz > 0:
        W_raw = C.copy().astype(np.float64)
        W_raw.data = 1.0 / np.maximum(W_raw.data, 1e-12)

    row_sums = np.asarray(W_raw.sum(axis=1)).ravel()
    inv_row = np.where(row_sums > 0.0, 1.0 / row_sums, 0.0)
    P = sparse.diags(inv_row) @ W_raw
    return P.tocsr(), C.tocsr()


def solve_rsp_matrices(
    P: sparse.spmatrix,
    C: sparse.spmatrix,
    beta: float,
    node_ids: tuple[str, ...] | None = None,
) -> RSPResult:
    """Compute RSP expected hitting cost and free-energy distance from sparse (P, C).

    Parameters
    ----------
    P:
        (N, N) row-stochastic reference transition probability matrix.
    C:
        (N, N) non-negative edge cost matrix with matching sparsity pattern.
    beta:
        Inverse temperature parameter (beta > 0; denoted theta in Kivimaki/ConScape).
    """
    if beta <= 0.0:
        raise ValueError("beta must be strictly positive (beta > 0)")

    P_csr = sparse.csr_matrix(P, dtype=np.float64)
    C_csr = sparse.csr_matrix(C, dtype=np.float64)
    n = P_csr.shape[0]
    if P_csr.shape != (n, n) or C_csr.shape != (n, n):
        raise ValueError("P and C must be square matrices of identical shape")

    # Compute W = P * exp(-beta * C) on the sparsity support of P
    P_coo = P_csr.tocoo()
    c_vals = np.asarray(C_csr[P_coo.row, P_coo.col]).ravel()
    e_vals = np.exp(-beta * c_vals)
    w_vals = P_coo.data * e_vals
    wc_vals = w_vals * c_vals

    W = sparse.csr_matrix((w_vals, (P_coo.row, P_coo.col)), shape=(n, n), dtype=np.float64)
    Wc = sparse.csr_matrix((wc_vals, (P_coo.row, P_coo.col)), shape=(n, n), dtype=np.float64)

    # Solve (I - W) Z = I via sparse LU factorization
    I_minus_W = (sparse.eye(n, format="csc", dtype=np.float64) - W.tocsc())
    lu = splu(I_minus_W)
    Z = lu.solve(np.eye(n, dtype=np.float64))

    # Check numerical validity
    if not np.all(np.isfinite(Z)):
        raise ValueError("Fundamental matrix Z = (I - W)^-1 is not finite; check graph connectivity or beta")

    # S = Z @ Wc @ Z
    S = Z @ (Wc @ Z)

    # Hitting expected cost: C_bar[s, t] = S[s, t]/Z[s, t] - S[t, t]/Z[t, t]
    z_safe = np.where(np.abs(Z) > 1e-300, Z, np.nan)
    M = S / z_safe
    diag_M = np.diag(M)
    expected_cost = M - diag_M[np.newaxis, :]
    np.fill_diagonal(expected_cost, 0.0)

    # Free energy (potential) distance: d_FE[s, t] = -(1/beta) * log(Z[s, t] / Z[t, t])
    diag_Z = np.diag(Z)
    ratio = np.where((Z > 0.0) & (diag_Z[np.newaxis, :] > 0.0), Z / diag_Z[np.newaxis, :], np.nan)
    with np.errstate(divide="ignore", invalid="ignore"):
        fe_dist = -(1.0 / beta) * np.log(ratio)
    np.fill_diagonal(fe_dist, 0.0)

    ids = node_ids if node_ids is not None else tuple(str(i) for i in range(n))
    return RSPResult(
        beta=float(beta),
        expected_cost=expected_cost,
        free_energy_distance=fe_dist,
        fundamental_matrix=Z,
        weighted_transition=W,
        transition_matrix=P_csr,
        cost_matrix=C_csr,
        node_ids=ids,
    )


def solve_rsp(
    graph: HabitatGraph,
    beta: float | None = None,
    theta: float | None = None,
) -> RSPResult:
    """Solve RSP expected cost and free-energy distance on a HabitatGraph."""
    inv_temp = beta if beta is not None else theta
    if inv_temp is None:
        inv_temp = 1.0
    P, C = build_reference_transition(graph)
    return solve_rsp_matrices(P, C, beta=float(inv_temp), node_ids=graph.node_ids)


def least_cost_path_matrix(graph: HabitatGraph) -> np.ndarray:
    """Compute exact all-pairs Least-Cost Path (LCP) distance matrix via Dijkstra."""
    C = graph.to_sparse_matrix("cost", reduce="min")
    return shortest_path(C, directed=True, method="D")


def effective_resistance_matrix(graph: HabitatGraph) -> np.ndarray:
    """Compute exact circuit-theory effective resistance matrix for an undirected/symmetrized graph."""
    W = graph.to_sparse_matrix("weight", reduce="sum").toarray()
    A = 0.5 * (W + W.T)
    np.fill_diagonal(A, 0.0)
    L = np.diag(A.sum(axis=1)) - A
    L_pinv = np.linalg.pinv(L)
    diag = np.diag(L_pinv)
    R = diag[:, None] + diag[None, :] - 2.0 * L_pinv
    np.fill_diagonal(R, 0.0)
    return np.maximum(R, 0.0)
