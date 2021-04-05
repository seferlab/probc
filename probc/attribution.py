"""Feature attribution for ProbC predictions.

Because ``s_uv = C_u^T Z C_v = sum_{m,n} c^u_m z_mn c^v_n`` is bilinear, the
log-survival ``-log P(no interaction)`` decomposes *exactly* into mark-pair
contributions. Splitting each pair term equally between its two marks gives a
per-mark attribution that sums to ``s_uv``; relative to an all-zero baseline this
coincides with the Shapley value of each mark for the game ``v(S) = s_uv``
restricted to marks in S (bilinear games split interaction terms equally).

The paper estimated SHAP values and averaged them over bins; :func:`bin_attribution`
does the same with these exact values. :func:`kernel_shap` is an optional
model-agnostic alternative using the ``shap`` package.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from .data import MarkMatrix


def pair_contributions(theta: np.ndarray, cu: np.ndarray, cv: np.ndarray) -> np.ndarray:
    """(m, m) matrix of mark-pair contributions to s_uv (symmetrised)."""
    M = np.outer(cu, cv) * theta
    return 0.5 * (M + M.T)


def mark_shapley(theta: np.ndarray, cu: np.ndarray, cv: np.ndarray) -> np.ndarray:
    """Exact per-mark Shapley values of ``s_uv`` when marks are switched on/off at
    both anchors jointly (baseline: no marks)."""
    P = pair_contributions(theta, cu, cv)
    # diagonal terms belong to one mark; off-diagonal terms are split equally,
    # which for the symmetrised matrix is simply its row sum.
    return P.sum(1)


def site_attribution(theta: np.ndarray, mm: MarkMatrix, u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Per-pair, per-mark attributions, shape (len(u), m). Rows sum to s_uv."""
    # mark m contributes: 0.5 * c^u_m (Z C_v)_m + 0.5 * c^v_m (Z C_u)_m
    return 0.5 * (mm.C[u] * (mm.C[v] @ theta)) + 0.5 * (mm.C[v] * (mm.C[u] @ theta))


def bin_attribution(theta: np.ndarray, mm: MarkMatrix, u: np.ndarray, v: np.ndarray, mark: str) -> np.ndarray:
    """Average attribution of ``mark`` per site over all pairs it participates in."""
    j = mm.marks.index(mark)
    att = site_attribution(theta, mm, u, v)[:, j]
    tot = np.zeros(mm.n_sites)
    cnt = np.zeros(mm.n_sites)
    np.add.at(tot, u, att)
    np.add.at(tot, v, att)
    np.add.at(cnt, u, 1)
    np.add.at(cnt, v, 1)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(cnt > 0, tot / cnt, 0.0)


def kernel_shap(model, mm: MarkMatrix, u: np.ndarray, v: np.ndarray, background: Optional[int] = 50, seed: int = 0):
    """Model-agnostic KernelSHAP on concatenated [C_u, C_v] features (needs ``shap``)."""
    import shap  # optional dependency

    rng = np.random.default_rng(seed)
    m = mm.n_marks
    X = np.hstack([mm.C[u], mm.C[v]])

    def f(Z):
        s = np.einsum("ij,ij->i", Z[:, :m] @ model.theta_, Z[:, m:])
        return -np.expm1(-np.maximum(s, 0))

    bg = X[rng.choice(len(X), size=min(background, len(X)), replace=False)]
    explainer = shap.KernelExplainer(f, bg)
    return explainer.shap_values(X)
