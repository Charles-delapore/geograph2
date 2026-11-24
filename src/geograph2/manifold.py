"""Continuous habitat manifold modeling on 128D TESSERA embeddings.

Provides:
- Cosine resistance field construction relative to target habitat prototypes,
- Semantic gradient boundary detection (ecotone & barrier mapping),
- Ultra-fast CPU few-shot suitability fitting (closed-form ridge probe in 128D),
- Quantitative comparison against discrete hard land-cover classification.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from scipy.stats import spearmanr

from geograph2.embedding import TesseraCube


def _l2_normalize(arr: np.ndarray, axis: int = -1, eps: float = 1e-12) -> np.ndarray:
    norms = np.linalg.norm(arr, axis=axis, keepdims=True)
    return arr / np.maximum(norms, eps)


def cosine_resistance_field(
    cube: TesseraCube,
    reference_embedding: np.ndarray,
    base_resistance: float = 1.0,
    gamma: float = 10.0,
    power: float = 2.0,
) -> np.ndarray:
    """Build a continuous resistance surface from cosine dissimilarity to a habitat prototype.

    ``R(y, x) = base_resistance + gamma * ((1 - cos(e(y, x), e_ref)) / 2)^power``
    """
    emb_u = _l2_normalize(cube.embeddings.astype(np.float64), axis=-1)
    ref = np.asarray(reference_embedding, dtype=np.float64).ravel()
    if ref.shape != (cube.dim,):
        raise ValueError(f"reference_embedding must have shape ({cube.dim},)")
    ref_u = ref / max(float(np.linalg.norm(ref)), 1e-12)
    cos_sim = np.clip(np.tensordot(emb_u, ref_u, axes=([-1], [0])), -1.0, 1.0)
    dissim = 0.5 * (1.0 - cos_sim)
    return base_resistance + gamma * (dissim ** power)


def semantic_gradient_boundaries(cube: TesseraCube) -> np.ndarray:
    """Detect continuous semantic boundaries via spatial cosine-distance gradients.

    Computes the RMS cosine dissimilarity to 4-connected neighbors at every pixel.
    """
    emb_u = _l2_normalize(cube.embeddings.astype(np.float64), axis=-1)
    h, w, _ = emb_u.shape
    grad_acc = np.zeros((h, w), dtype=np.float64)
    count = np.zeros((h, w), dtype=np.float64)

    # Vertical neighbor cosine distance
    if h > 1:
        dy = 1.0 - np.clip(np.sum(emb_u[:-1, :, :] * emb_u[1:, :, :], axis=-1), -1.0, 1.0)
        grad_acc[:-1, :] += dy * dy
        grad_acc[1:, :] += dy * dy
        count[:-1, :] += 1.0
        count[1:, :] += 1.0

    # Horizontal neighbor cosine distance
    if w > 1:
        dx = 1.0 - np.clip(np.sum(emb_u[:, :-1, :] * emb_u[:, 1:, :], axis=-1), -1.0, 1.0)
        grad_acc[:, :-1] += dx * dx
        grad_acc[:, 1:] += dx * dx
        count[:, :-1] += 1.0
        count[:, 1:] += 1.0

    return np.sqrt(grad_acc / np.maximum(count, 1.0))


@dataclass(frozen=True)
class FewShotSuitabilityModel:
    """Closed-form L2-regularized linear probe on 128D embeddings."""

    weights: np.ndarray
    bias: float
    l2_reg: float

    def predict(self, cube: TesseraCube) -> np.ndarray:
        emb_u = _l2_normalize(cube.embeddings.astype(np.float64), axis=-1)
        scores = np.tensordot(emb_u, self.weights, axes=([-1], [0])) + self.bias
        return np.clip(scores, 0.0, 1.0)


def fit_few_shot_suitability(
    cube: TesseraCube,
    sample_coords: Sequence[tuple[int, int]],
    sample_targets: Sequence[float],
    l2_reg: float = 0.5,
) -> FewShotSuitabilityModel:
    """CPU-fast closed-form ridge regression probe in embedding space (few-shot learning)."""
    if len(sample_coords) != len(sample_targets) or len(sample_coords) < 2:
        raise ValueError("Provide at least 2 (row, col) samples with matching target values")

    emb_u = _l2_normalize(cube.embeddings.astype(np.float64), axis=-1)
    X = np.array([emb_u[r, c] for r, c in sample_coords], dtype=np.float64)
    y = np.asarray(sample_targets, dtype=np.float64)

    # Augment with bias column (unregularized)
    n, d = X.shape
    X_aug = np.hstack([X, np.ones((n, 1), dtype=np.float64)])
    reg = np.eye(d + 1, dtype=np.float64) * float(l2_reg)
    reg[-1, -1] = 1e-8  # keep bias unregularized

    # Solve (X^T X + reg) theta = X^T y
    theta = np.linalg.solve(X_aug.T @ X_aug + reg, X_aug.T @ y)
    return FewShotSuitabilityModel(weights=theta[:-1], bias=float(theta[-1]), l2_reg=float(l2_reg))


def compare_continuous_vs_hard_classification(
    continuous_suitability: np.ndarray,
    hard_classes: np.ndarray,
    ground_truth: np.ndarray,
) -> dict[str, float]:
    """Quantitatively compare continuous embedding suitability against hard discrete classes."""
    cont = np.asarray(continuous_suitability, dtype=np.float64).ravel()
    hard = np.asarray(hard_classes, dtype=np.float64).ravel()
    gt = np.asarray(ground_truth, dtype=np.float64).ravel()

    # Calibrate hard classes to [0, 1] by class mean of ground truth (best-case hard lookup table)
    hard_mapped = np.zeros_like(gt)
    for cls in np.unique(hard):
        mask = hard == cls
        hard_mapped[mask] = float(np.mean(gt[mask]))

    rmse_cont = float(np.sqrt(np.mean((cont - gt) ** 2)))
    rmse_hard = float(np.sqrt(np.mean((hard_mapped - gt) ** 2)))

    r_cont = float(np.corrcoef(cont, gt)[0, 1]) if np.std(cont) > 0 and np.std(gt) > 0 else 0.0
    r_hard = float(np.corrcoef(hard_mapped, gt)[0, 1]) if np.std(hard_mapped) > 0 and np.std(gt) > 0 else 0.0

    rho_cont = float(spearmanr(cont, gt).statistic)
    rho_hard = float(spearmanr(hard_mapped, gt).statistic)

    return {
        "rmse_continuous": rmse_cont,
        "rmse_hard_class": rmse_hard,
        "rmse_reduction_pct": (100.0 * (rmse_hard - rmse_cont) / rmse_hard) if rmse_hard > 0 else 0.0,
        "pearson_continuous": r_cont,
        "pearson_hard_class": r_hard,
        "spearman_continuous": rho_cont,
        "spearman_hard_class": rho_hard,
    }
