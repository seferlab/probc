"""Negative log-likelihoods of the ProbC model and their gradients.

Two likelihoods are implemented:

* ``bernoulli`` (unbinned restriction-site graph, Eqs. 1-4). Parameter ``Z >= 0``
  with ``x_mn = 1 - exp(-z_mn)`` so that

      P(E_uv | X) = 1 - prod_{m,n} (1 - x_mn)^{c_m c_n} = 1 - exp(-C_u^T Z C_v)

  and the NLL (Eq. 4, negated) is

      -sum_{(u,v) in E} log(1 - exp(-C_u^T Z C_v)) + sum_{(u,v) not in E} C_u^T Z C_v,

  which is convex in Z (Theorem 1).

* ``poisson`` (binned variant, Eq. 19). Parameter ``X`` with ``0 <= X <= 1`` and
  ``lambda_uv = C_u^T X C_v`` (Le Cam approximation of the Poisson-binomial):

      NLL = sum_{(u,v)} [lambda_uv - k_uv log lambda_uv] (+ const).

  Because lambda is linear in X this NLL is also convex in X.

Efficiency: the loss depends on a pair only through the mark *patterns*
(rows of C) at its two ends, so pairs are compressed into unique pattern pairs with
aggregated weights (:class:`CompiledPairs`). This is exact.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from .data import MarkMatrix, PairSet

EPS_RATE = 1e-8  # tiny additive rate to keep log() finite when s_uv == 0


@dataclass
class CompiledPairs:
    """Pairs compressed to unique (pattern_u, pattern_v) combinations.

    P : (k, m) unique mark-pattern rows.
    a, b : (q,) pattern indices of the two ends of each compressed pair (a <= b).
    w_pos : total weight of pairs with y > 0.
    w_neg : total weight of pairs with y == 0.
    wy : total weight * y (Poisson counts).
    w_all : total weight (w_pos + w_neg).
    W_implicit : (m, m) linear term of implicit zero-label pairs (or None).
    n_total : total number (weight) of pairs incl. implicit ones (for loss scaling).
    """

    P: np.ndarray
    a: np.ndarray
    b: np.ndarray
    w_pos: np.ndarray
    w_neg: np.ndarray
    wy: np.ndarray
    W_implicit: Optional[np.ndarray]
    n_total: float

    @property
    def w_all(self) -> np.ndarray:
        return self.w_pos + self.w_neg

    @property
    def n_marks(self) -> int:
        return self.P.shape[1]


def compile_pairs(C: np.ndarray, pairs: PairSet) -> CompiledPairs:
    C = np.asarray(C, dtype=np.float64)
    if len(pairs) == 0:
        P = np.zeros((1, C.shape[1]))
        z = np.zeros(0)
        return CompiledPairs(P, z.astype(int), z.astype(int), z, z, z, pairs.W_implicit, pairs.total_weight)
    used = np.unique(np.concatenate([pairs.u, pairs.v]))
    P, inv_used = np.unique(C[used], axis=0, return_inverse=True)
    inv_used = np.asarray(inv_used).ravel()
    lookup = np.full(C.shape[0], -1, dtype=np.int64)
    lookup[used] = inv_used
    pa = lookup[pairs.u]
    pb = lookup[pairs.v]
    a = np.minimum(pa, pb)
    b = np.maximum(pa, pb)
    k = np.int64(P.shape[0])
    keys = a * k + b
    ukeys, inv = np.unique(keys, return_inverse=True)
    inv = np.asarray(inv).ravel()
    q = len(ukeys)
    is_pos = pairs.y > 0
    w_pos = np.bincount(inv, weights=pairs.w * is_pos, minlength=q)
    w_neg = np.bincount(inv, weights=pairs.w * (~is_pos), minlength=q)
    wy = np.bincount(inv, weights=pairs.w * pairs.y, minlength=q)
    return CompiledPairs(
        P=P,
        a=(ukeys // k).astype(np.int64),
        b=(ukeys % k).astype(np.int64),
        w_pos=w_pos,
        w_neg=w_neg,
        wy=wy,
        W_implicit=None if pairs.W_implicit is None else 0.5 * (pairs.W_implicit + pairs.W_implicit.T),
        n_total=pairs.total_weight,
    )


def pair_scores(theta: np.ndarray, P: np.ndarray, a: np.ndarray, b: np.ndarray, chunk: int = 500_000) -> np.ndarray:
    """``s_e = P_a^T theta P_b`` for each compressed pair."""
    A = P @ theta  # (k, m)
    out = np.empty(len(a))
    for s in range(0, len(a), chunk):
        sl = slice(s, s + chunk)
        out[sl] = np.einsum("ij,ij->i", A[a[sl]], P[b[sl]])
    return out


def _weighted_outer(P: np.ndarray, a: np.ndarray, b: np.ndarray, g: np.ndarray, chunk: int = 500_000) -> np.ndarray:
    """``sum_e g_e P_a P_b^T`` symmetrised (gradient of sum_e f(s_e) wrt theta)."""
    m = P.shape[1]
    k = P.shape[0]
    G = np.zeros((m, m))
    if len(a) == 0:
        return G
    if len(a) > 4 * k:
        # aggregate into a k x k matrix first: G = P^T B P
        try:
            import scipy.sparse as sp

            B = sp.coo_matrix((g, (a, b)), shape=(k, k)).tocsr()
            G = P.T @ (B @ P)
            return 0.5 * (G + G.T)
        except Exception:  # pragma: no cover - scipy always present in practice
            pass
    for s in range(0, len(a), chunk):
        sl = slice(s, s + chunk)
        G += P[a[sl]].T @ (g[sl, None] * P[b[sl]])
    return 0.5 * (G + G.T)


# ----------------------------------------------------------------------------------
# Bernoulli (unbinned) NLL in Z
# ----------------------------------------------------------------------------------


def _neg_log1mexp(s: np.ndarray) -> np.ndarray:
    """-log(1 - exp(-s)) computed stably for s > 0."""
    with np.errstate(divide="ignore", invalid="ignore"):
        return -np.where(s < 0.693, np.log(-np.expm1(-s)), np.log1p(-np.exp(-s)))


def bernoulli_nll(Z: np.ndarray, cp: CompiledPairs, eps: float = EPS_RATE, with_grad: bool = True):
    s = pair_scores(Z, cp.P, cp.a, cp.b) + eps
    pos = cp.w_pos > 0  # only pairs with observed interactions enter the log term
    f = float(np.dot(cp.w_pos[pos], _neg_log1mexp(s[pos])) + np.dot(cp.w_neg, s))
    if cp.W_implicit is not None:
        f += float(np.sum(Z * cp.W_implicit))
    if not with_grad:
        return f
    # d/ds [-log(1-e^{-s})] = -1 / expm1(s)
    g = cp.w_neg.copy()
    with np.errstate(divide="ignore"):
        g[pos] -= cp.w_pos[pos] / np.expm1(s[pos])
    G = _weighted_outer(cp.P, cp.a, cp.b, g)
    if cp.W_implicit is not None:
        G = G + cp.W_implicit
    return f, G


def bernoulli_proba(Z: np.ndarray, Cu: np.ndarray, Cv: np.ndarray) -> np.ndarray:
    s = np.einsum("ij,ij->i", Cu @ Z, Cv)
    return -np.expm1(-np.maximum(s, 0.0))


# ----------------------------------------------------------------------------------
# Poisson (binned) NLL in X
# ----------------------------------------------------------------------------------


def poisson_nll(X: np.ndarray, cp: CompiledPairs, eps: float = EPS_RATE, with_grad: bool = True):
    lam = pair_scores(X, cp.P, cp.a, cp.b) + eps
    w_all = cp.w_all
    pos = cp.wy > 0
    with np.errstate(divide="ignore"):
        f = float(np.dot(w_all, lam) - np.dot(cp.wy[pos], np.log(lam[pos])))
    if cp.W_implicit is not None:
        f += float(np.sum(X * cp.W_implicit))
    if not with_grad:
        return f
    g = w_all.copy()
    with np.errstate(divide="ignore"):
        g[pos] -= cp.wy[pos] / lam[pos]
    G = _weighted_outer(cp.P, cp.a, cp.b, g)
    if cp.W_implicit is not None:
        G = G + cp.W_implicit
    return f, G


LIKELIHOODS = {
    "bernoulli": bernoulli_nll,
    "poisson": poisson_nll,
}


def nll(theta: np.ndarray, C: np.ndarray, pairs: PairSet, likelihood: str = "bernoulli") -> float:
    """Convenience: un-normalised NLL of ``pairs`` under parameter ``theta``."""
    cp = compile_pairs(C, pairs)
    return LIKELIHOODS[likelihood](theta, cp, with_grad=False)


def mark_matrix_scores(theta: np.ndarray, mm: MarkMatrix, u: np.ndarray, v: np.ndarray, chunk: int = 500_000) -> np.ndarray:
    """``C_u^T theta C_v`` for arbitrary site pairs of a MarkMatrix."""
    out = np.empty(len(u))
    A = mm.C @ theta
    for s in range(0, len(u), chunk):
        sl = slice(s, s + chunk)
        out[sl] = np.einsum("ij,ij->i", A[u[sl]], mm.C[v[sl]])
    return out
