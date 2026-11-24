"""Adjoint analytical sensitivity solver (dK/dQ, dK/dP, dK/dC) and D4 Rust backend hook.

Computes exact reverse-mode adjoint gradients of the RSP landscape connectivity
functional K(Q, P) = sum_{s,t} Q_{s,t} * k(C_bar_{s,t}) with respect to:
- habitat quality weights Q (both pair-matrix dK/dQ and node-vector dK/dq),
- reference transition matrix P (dK/dP),
- edge cost matrix C (dK/dC).

Also provides a clean pluggable backend registry for switching between the pure
Python/scipy.sparse kernel and a future D4 Rust extension kernel.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Protocol

import numpy as np
from scipy import sparse

from geograph2.rsp import RSPResult, solve_rsp_matrices


@dataclass(frozen=True)
class RSPAdjointResult:
    """Outputs of the RSP adjoint sensitivity calculation."""

    connectivity_K: float
    kernel_matrix: np.ndarray
    dK_dQ: np.ndarray
    dK_dq: np.ndarray | None
    dK_dP: np.ndarray
    dK_dC: np.ndarray
    rsp: RSPResult
    backend_name: str = "scipy"


class RSPSolverBackend(Protocol):
    """Pluggable solver backend interface reserved for D4 Rust native kernel."""

    name: str

    def solve_adjoint(
        self,
        P: sparse.spmatrix,
        C: sparse.spmatrix,
        beta: float,
        alpha: float,
        q: np.ndarray | None,
        Q: np.ndarray | None,
        kernel: str,
    ) -> RSPAdjointResult: ...


class ScipyRSPBackend:
    """Default pure-Python / scipy.sparse RSP adjoint solver backend."""

    name: str = "scipy"

    def solve_adjoint(
        self,
        P: sparse.spmatrix,
        C: sparse.spmatrix,
        beta: float,
        alpha: float = 0.25,
        q: np.ndarray | None = None,
        Q: np.ndarray | None = None,
        kernel: str = "exponential",
    ) -> RSPAdjointResult:
        rsp = solve_rsp_matrices(P, C, beta=beta)
        Z = rsp.fundamental_matrix
        P_csr = rsp.transition_matrix
        C_csr = rsp.cost_matrix
        W = rsp.weighted_transition
        n = Z.shape[0]

        if Q is not None:
            Q_mat = np.asarray(Q, dtype=np.float64)
            if Q_mat.shape != (n, n):
                raise ValueError(f"Q must have shape ({n}, {n})")
            q_vec = None
        elif q is not None:
            q_vec = np.asarray(q, dtype=np.float64).ravel()
            if q_vec.shape != (n,):
                raise ValueError(f"q must have shape ({n},)")
            Q_mat = np.outer(q_vec, q_vec)
        else:
            q_vec = np.ones(n, dtype=np.float64)
            Q_mat = np.ones((n, n), dtype=np.float64)

        C_bar = rsp.expected_cost
        if kernel == "exponential":
            K_mat = np.exp(-alpha * C_bar)
            dK_dCbar = -alpha * Q_mat * K_mat
        elif kernel == "linear":
            K_mat = C_bar.copy()
            dK_dCbar = Q_mat.copy()
        else:
            raise ValueError(f"Unsupported kernel: {kernel}")

        # Zero diagonal sensitivity because C_bar[t, t] == 0 identically
        np.fill_diagonal(dK_dCbar, 0.0)
        K_val = float(np.sum(Q_mat * K_mat))

        # Exact dK/dQ and dK/dq
        dK_dQ = K_mat.copy()
        dK_dq = (K_mat @ q_vec + K_mat.T @ q_vec) if q_vec is not None else None

        # Forward intermediates needed for reverse-mode adjoint
        P_coo = P_csr.tocoo()
        c_vals = np.asarray(C_csr[P_coo.row, P_coo.col]).ravel()
        e_vals = np.exp(-beta * c_vals)
        w_vals = P_coo.data * e_vals
        wc_vals = w_vals * c_vals
        Wc = sparse.csr_matrix((wc_vals, (P_coo.row, P_coo.col)), shape=(n, n), dtype=np.float64)
        S = Z @ (Wc @ Z)

        # Adjoint of C_bar[s, t] = M[s, t] - M[t, t]
        M_bar = dK_dCbar.copy()
        col_sums = np.sum(dK_dCbar, axis=0)
        M_bar[np.arange(n), np.arange(n)] -= col_sums

        # Adjoint of M[s, t] = S[s, t] / Z[s, t]
        S_bar = M_bar / Z
        Z_bar = -(M_bar * S) / (Z * Z)

        # Adjoint of S = Z @ Wc @ Z
        Wc_bar = Z.T @ S_bar @ Z.T
        Z_bar = Z_bar + S_bar @ (Wc @ Z).T + (Z @ Wc).T @ S_bar

        # Adjoint of Z = (I - W)^-1  =>  W_bar_1 = Z^T @ Z_bar @ Z^T
        W_bar_from_Z = Z.T @ Z_bar @ Z.T

        # Assemble dK/dP and dK/dC on the sparsity support of P
        dK_dP = np.zeros((n, n), dtype=np.float64)
        dK_dC = np.zeros((n, n), dtype=np.float64)

        w_bar_total_edge = (
            W_bar_from_Z[P_coo.row, P_coo.col]
            + Wc_bar[P_coo.row, P_coo.col] * c_vals
        )
        dK_dP[P_coo.row, P_coo.col] = w_bar_total_edge * e_vals

        # dW/dC = -beta * W; dWc/dC = W - beta * Wc
        dK_dC[P_coo.row, P_coo.col] = (
            W_bar_from_Z[P_coo.row, P_coo.col] * (-beta * w_vals)
            + Wc_bar[P_coo.row, P_coo.col] * (w_vals - beta * wc_vals)
        )

        return RSPAdjointResult(
            connectivity_K=K_val,
            kernel_matrix=K_mat,
            dK_dQ=dK_dQ,
            dK_dq=dK_dq,
            dK_dP=dK_dP,
            dK_dC=dK_dC,
            rsp=rsp,
            backend_name=self.name,
        )


_BACKENDS: dict[str, RSPSolverBackend] = {"scipy": ScipyRSPBackend()}
_ACTIVE_BACKEND: str = "scipy"


def register_rsp_backend(name: str, backend: RSPSolverBackend) -> None:
    """Register an external RSP solver backend (e.g. D4 Rust native extension)."""
    _BACKENDS[name] = backend


def set_rsp_backend(name: str) -> None:
    """Switch the active RSP solver backend."""
    global _ACTIVE_BACKEND
    if name not in _BACKENDS:
        raise KeyError(f"Unknown RSP backend '{name}'. Registered: {sorted(_BACKENDS)}")
    _ACTIVE_BACKEND = name


def get_rsp_backend(name: str | None = None) -> RSPSolverBackend:
    """Retrieve a registered RSP solver backend (defaults to active backend)."""
    key = _ACTIVE_BACKEND if name is None else name
    if key not in _BACKENDS:
        raise KeyError(f"Unknown RSP backend '{key}'. Registered: {sorted(_BACKENDS)}")
    return _BACKENDS[key]


def available_rsp_backends() -> tuple[str, ...]:
    """Return names of all registered RSP solver backends."""
    return tuple(sorted(_BACKENDS))


def rsp_adjoint_sensitivity(
    P: sparse.spmatrix,
    C: sparse.spmatrix,
    beta: float = 1.0,
    alpha: float = 0.25,
    q: np.ndarray | None = None,
    Q: np.ndarray | None = None,
    kernel: str = "exponential",
    backend: str | None = None,
) -> RSPAdjointResult:
    """Compute RSP connectivity K and analytical sensitivities dK/dQ, dK/dP, dK/dC."""
    solver = get_rsp_backend(backend)
    return solver.solve_adjoint(P=P, C=C, beta=beta, alpha=alpha, q=q, Q=Q, kernel=kernel)
