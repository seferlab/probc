"""Proximal operators used by the ProbC ADMM solver.

Notation follows Sefer (2022), BMC Genomics 23:287, Eqs. (12)-(16).
"""
from __future__ import annotations

import numpy as np


def symmetrize(A: np.ndarray) -> np.ndarray:
    """Return the symmetric part (A + A^T) / 2."""
    return 0.5 * (A + A.T)


def singular_value_threshold(Q: np.ndarray, alpha: float) -> np.ndarray:
    """Singular value soft-thresholding S_alpha(Q) = U diag((sigma_i - alpha)_+) V^T.

    This is the closed-form proximal operator of ``alpha * ||.||_*`` (Eq. 13).
    """
    if alpha <= 0:
        return Q.copy()
    U, s, Vt = np.linalg.svd(Q, full_matrices=False)
    s_thr = np.maximum(s - alpha, 0.0)
    keep = s_thr > 0
    if not np.any(keep):
        return np.zeros_like(Q)
    return (U[:, keep] * s_thr[keep]) @ Vt[keep]


def soft_threshold(Q: np.ndarray, alpha: float) -> np.ndarray:
    """Element-wise soft-thresholding, the proximal operator of ``alpha * ||.||_1``.

    Implements Eqs. (15)-(16): entries with |Q_ij| < alpha are set to zero and the
    remaining ones are shrunk towards zero by alpha.
    """
    if alpha <= 0:
        return Q.copy()
    return np.sign(Q) * np.maximum(np.abs(Q) - alpha, 0.0)


def nuclear_norm(Q: np.ndarray) -> float:
    return float(np.linalg.svd(Q, compute_uv=False).sum())


def l1_norm(Q: np.ndarray) -> float:
    return float(np.abs(Q).sum())
