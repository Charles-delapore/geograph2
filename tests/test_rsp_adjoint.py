import unittest

import numpy as np
from scipy import sparse

from geograph2.rsp import solve_rsp_matrices
from geograph2.rsp_adjoint import (
    RSPAdjointResult,
    ScipyRSPBackend,
    available_rsp_backends,
    register_rsp_backend,
    rsp_adjoint_sensitivity,
    set_rsp_backend,
)


def _sample_pc(n: int = 5):
    # Strongly connected asymmetric directed ring + chords
    rows, cols, p_vals, c_vals = [], [], [], []
    rng = np.random.default_rng(42)
    A = np.zeros((n, n), dtype=np.float64)
    C = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        j1 = (i + 1) % n
        j2 = (i - 1) % n
        j3 = (i + 2) % n
        for j, w, cost in ((j1, 0.5, 1.0 + 0.2 * i), (j2, 0.3, 1.5), (j3, 0.2, 2.2)):
            A[i, j] = w
            C[i, j] = cost + 0.1 * float(rng.random())
    P = A / A.sum(axis=1, keepdims=True)
    return sparse.csr_matrix(P), sparse.csr_matrix(C)


class TestRSPAdjointSensitivity(unittest.TestCase):
    def test_adjoint_dK_dP_dK_dQ_dK_dC_match_finite_difference(self):
        P, C = _sample_pc(5)
        q = np.array([1.2, 0.8, 1.5, 0.6, 1.1], dtype=np.float64)
        beta = 0.8
        alpha = 0.3

        adj = rsp_adjoint_sensitivity(P, C, beta=beta, alpha=alpha, q=q, kernel="exponential")
        eps = 1e-6

        # 1. Verify dK/dq via central finite differences
        fd_dq = np.zeros_like(q)
        for i in range(len(q)):
            qp = q.copy()
            qm = q.copy()
            qp[i] += eps
            qm[i] -= eps
            kp = rsp_adjoint_sensitivity(P, C, beta=beta, alpha=alpha, q=qp).connectivity_K
            km = rsp_adjoint_sensitivity(P, C, beta=beta, alpha=alpha, q=qm).connectivity_K
            fd_dq[i] = (kp - km) / (2.0 * eps)
        rel_err_q = np.max(np.abs(adj.dK_dq - fd_dq) / np.maximum(np.abs(fd_dq), 1e-12))
        self.assertLess(rel_err_q, 1e-6)

        # 2. Verify dK/dP on all non-zero entries via central finite differences
        P_dense = P.toarray()
        max_rel_err_p = 0.0
        for i, j in zip(*P.nonzero()):
            Pp = P_dense.copy()
            Pm = P_dense.copy()
            Pp[i, j] += eps
            Pm[i, j] -= eps
            kp = float(np.sum(np.outer(q, q) * np.exp(-alpha * solve_rsp_matrices(sparse.csr_matrix(Pp), C, beta).expected_cost)))
            km = float(np.sum(np.outer(q, q) * np.exp(-alpha * solve_rsp_matrices(sparse.csr_matrix(Pm), C, beta).expected_cost)))
            fd = (kp - km) / (2.0 * eps)
            ana = adj.dK_dP[i, j]
            rel = abs(ana - fd) / max(abs(fd), 1e-12)
            max_rel_err_p = max(max_rel_err_p, rel)
        self.assertLess(max_rel_err_p, 1e-5)

        # 3. Verify dK/dC on all non-zero entries via central finite differences
        C_dense = C.toarray()
        max_rel_err_c = 0.0
        for i, j in zip(*P.nonzero()):
            Cp = C_dense.copy()
            Cm = C_dense.copy()
            Cp[i, j] += eps
            Cm[i, j] -= eps
            kp = float(np.sum(np.outer(q, q) * np.exp(-alpha * solve_rsp_matrices(P, sparse.csr_matrix(Cp), beta).expected_cost)))
            km = float(np.sum(np.outer(q, q) * np.exp(-alpha * solve_rsp_matrices(P, sparse.csr_matrix(Cm), beta).expected_cost)))
            fd = (kp - km) / (2.0 * eps)
            ana = adj.dK_dC[i, j]
            rel = abs(ana - fd) / max(abs(fd), 1e-12)
            max_rel_err_c = max(max_rel_err_c, rel)
        self.assertLess(max_rel_err_c, 1e-5)

    def test_d4_rust_backend_hook_registration_and_switching(self):
        class MockRustBackend(ScipyRSPBackend):
            name = "rust_d4_mock"

        register_rsp_backend("rust_d4_mock", MockRustBackend())
        self.assertIn("rust_d4_mock", available_rsp_backends())
        set_rsp_backend("rust_d4_mock")
        try:
            P, C = _sample_pc(4)
            res = rsp_adjoint_sensitivity(P, C, beta=1.0)
            self.assertEqual(res.backend_name, "rust_d4_mock")
        finally:
            set_rsp_backend("scipy")


if __name__ == "__main__":
    unittest.main()
