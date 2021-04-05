"""Baselines compared against ProbC in the paper.

* :class:`DistanceDecay` -- P(u, v interact) = alpha * d(u, v)^(-beta) (Fig. 8),
  with alpha fitted on the training set and beta in {1, 1.5, 2}.
* :class:`HiCRegRF` -- a re-implementation of the HiC-Reg feature design
  (Zhang et al. 2019, Nat Commun): random-forest regression of (log) contact
  counts from mark signals at both anchors ("pair-concat"), the average signal in
  the window between them, and genomic distance (Fig. 7).

Rambutan (Schreiber et al.) requires the nucleotide sequence + DNase-seq and a
deep CNN; it is not re-implemented here (see README).
"""
from __future__ import annotations

from typing import Optional

import numpy as np
from scipy.optimize import minimize_scalar

from .data import MarkMatrix, PairSet


def pair_distance(mm: MarkMatrix, u: np.ndarray, v: np.ndarray) -> np.ndarray:
    d = np.abs(mm.pos[u] - mm.pos[v]).astype(float)
    d[mm.chrom[u] != mm.chrom[v]] = np.inf
    return d


class DistanceDecay:
    """Power-law distance-decay predictor ``alpha * d^-beta``."""

    def __init__(self, beta: float = 1.0, d0: float = 1.0):
        self.beta = beta
        self.d0 = d0

    def _base(self, d):
        return np.power(np.maximum(d / self.d0, 1.0), -self.beta)

    def fit(self, mm: MarkMatrix, pairs: PairSet) -> "DistanceDecay":
        d = pair_distance(mm, pairs.u, pairs.v)
        ok = np.isfinite(d)
        b = self._base(d[ok])
        y = (pairs.y[ok] > 0).astype(float)
        w = pairs.w[ok]
        hi = 1.0 / max(b.max(), 1e-300)

        def nll(alpha):
            p = np.clip(alpha * b, 1e-12, 1 - 1e-12)
            return -np.sum(w * (y * np.log(p) + (1 - y) * np.log1p(-p)))

        res = minimize_scalar(nll, bounds=(1e-12, hi), method="bounded")
        self.alpha_ = float(res.x)
        return self

    def predict_proba(self, mm: MarkMatrix, u: np.ndarray, v: np.ndarray) -> np.ndarray:
        d = pair_distance(mm, u, v)
        p = np.zeros(len(u))
        ok = np.isfinite(d)
        p[ok] = np.clip(getattr(self, "alpha_", 1.0) * self._base(d[ok]), 0, 1)
        return p


def _window_mean_features(mm: MarkMatrix, u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Average mark vector over sites strictly between u and v (same chromosome)."""
    out = np.zeros((len(u), mm.n_marks))
    order = np.lexsort((mm.pos, mm.chrom))
    rank = np.empty(mm.n_sites, dtype=np.int64)
    rank[order] = np.arange(mm.n_sites)
    Cs = mm.C[order]
    P = np.vstack([np.zeros((1, mm.n_marks)), np.cumsum(Cs, 0)])
    ru, rv = rank[u], rank[v]
    lo = np.minimum(ru, rv) + 1
    hi = np.maximum(ru, rv)
    same = mm.chrom[u] == mm.chrom[v]
    n = np.maximum(hi - lo, 0)
    ok = same & (n > 0)
    out[ok] = (P[hi[ok]] - P[lo[ok]]) / n[ok, None]
    return out


def hicreg_features(mm: MarkMatrix, u: np.ndarray, v: np.ndarray) -> np.ndarray:
    d = pair_distance(mm, u, v)
    d = np.where(np.isfinite(d), d, np.nanmax(np.where(np.isfinite(d), d, 0)) * 10 + 1)
    return np.hstack([mm.C[u], mm.C[v], _window_mean_features(mm, u, v), np.log10(d + 1)[:, None]])


def hicreg_feature_names(marks) -> list:
    return [f"{m}_u" for m in marks] + [f"{m}_v" for m in marks] + [f"{m}_win" for m in marks] + ["log10_distance"]


class HiCRegRF:
    """HiC-Reg style random forest regression on pair features."""

    def __init__(self, n_estimators: int = 100, max_features: float = 1 / 3, min_samples_leaf: int = 5,
                 log_target: bool = True, n_jobs: int = -1, random_state: Optional[int] = 0, oob_score: bool = False):
        self.n_estimators = n_estimators
        self.max_features = max_features
        self.min_samples_leaf = min_samples_leaf
        self.log_target = log_target
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.oob_score = oob_score

    def fit(self, mm: MarkMatrix, pairs: PairSet) -> "HiCRegRF":
        from sklearn.ensemble import RandomForestRegressor

        X = hicreg_features(mm, pairs.u, pairs.v)
        y = np.log1p(pairs.y) if self.log_target else pairs.y
        self.rf_ = RandomForestRegressor(
            n_estimators=self.n_estimators, max_features=self.max_features, min_samples_leaf=self.min_samples_leaf,
            n_jobs=self.n_jobs, random_state=self.random_state, oob_score=self.oob_score, bootstrap=True,
        ).fit(X, y)
        self.feature_names_ = hicreg_feature_names(mm.marks)
        return self

    def predict(self, mm: MarkMatrix, u: np.ndarray, v: np.ndarray) -> np.ndarray:
        return self.rf_.predict(hicreg_features(mm, u, v))

    def predict_proba(self, mm: MarkMatrix, u: np.ndarray, v: np.ndarray) -> np.ndarray:
        """Predicted counts divided by their sum (the paper's normalisation)."""
        pred = np.maximum(self.predict(mm, u, v), 0)
        if self.log_target:
            pred = np.expm1(pred)
        s = pred.sum()
        return pred / s if s > 0 else pred

    def permutation_importance(self, mm: MarkMatrix, pairs: PairSet, n_repeats: int = 5, random_state: int = 0):
        """Permutation importance (change in error when a feature is permuted)."""
        from sklearn.inspection import permutation_importance

        X = hicreg_features(mm, pairs.u, pairs.v)
        y = np.log1p(pairs.y) if self.log_target else pairs.y
        r = permutation_importance(self.rf_, X, y, n_repeats=n_repeats, random_state=random_state, n_jobs=self.n_jobs)
        import pandas as pd

        return pd.DataFrame({"feature": self.feature_names_, "importance": r.importances_mean,
                             "std": r.importances_std}).sort_values("importance", ascending=False)
