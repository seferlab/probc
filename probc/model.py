"""The ProbC estimator (scikit-learn-like interface)."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional, Sequence, Union

import numpy as np

from .admm import ADMMResult, admm
from .data import MarkMatrix, PairSet
from .objectives import (
    LIKELIHOODS,
    CompiledPairs,
    compile_pairs,
    mark_matrix_scores,
)

ArrayOrMM = Union[np.ndarray, MarkMatrix]


def _support_mask(support, m: int, marks: Optional[Sequence[str]] = None) -> np.ndarray:
    if support is None or (isinstance(support, str) and support == "full"):
        return np.ones((m, m), bool)
    if isinstance(support, str) and support == "diagonal":
        # "forbid the interactions between chromatin marks": only z_mm may be non-zero
        return np.eye(m, dtype=bool)
    mask = np.asarray(support, dtype=bool)
    if mask.shape != (m, m):
        raise ValueError(f"support mask must be {m}x{m}")
    return mask | mask.T


class ProbC:
    """Generative probabilistic decomposition of Hi-C / Micro-C interactions by marks.

    Parameters
    ----------
    lambda_nuc : float
        Weight of the nuclear-norm (low-rank / block structure) penalty. This is
        ``lambda_1`` in Eq. (3) of the paper.
    lambda_l1 : float
        Weight of the l1 (sparsity) penalty, ``lambda_2`` in Eq. (3).
    likelihood : {"bernoulli", "poisson"}
        "bernoulli": unbinned interaction graph (Eqs. 1-4), parameter Z >= 0 with
        x = 1 - exp(-z). "poisson": binned counts (Eq. 19), parameter 0 <= X <= 1.
    support : "full" | "diagonal" | (m, m) bool array
        Which mark pairs may interact. "diagonal" reproduces the paper's
        experiment where interactions *between* different marks are forbidden.
    loss_scale : "mean" | "sum" | float
        Multiplier of the NLL. "mean" divides by the total number of modelled pairs
        so that regularisation strengths are comparable across data sizes.
    rho, adaptive_rho, max_iter, tol_abs, tol_rel : ADMM settings.
    inner_solver : "lbfgsb" | "pgd"
        Solver for the smooth subproblem (Eq. 17).
    upper_bound : float or None
        Box upper bound for the parameter. Defaults to None for Bernoulli (z in
        [0, inf) <=> x in [0, 1)) and 1.0 for Poisson (x in [0, 1]).
    """

    def __init__(
        self,
        lambda_nuc: float = 1e-3,
        lambda_l1: float = 1e-3,
        likelihood: str = "bernoulli",
        support="full",
        loss_scale="mean",
        rho: float = 1.0,
        adaptive_rho: bool = True,
        max_iter: int = 300,
        tol_abs: float = 1e-6,
        tol_rel: float = 1e-4,
        inner_solver: str = "lbfgsb",
        inner_max_iter: int = 50,
        upper_bound: Optional[float] = None,
        eps: float = 1e-8,
        verbose: int = 0,
    ):
        self.lambda_nuc = lambda_nuc
        self.lambda_l1 = lambda_l1
        self.likelihood = likelihood
        self.support = support
        self.loss_scale = loss_scale
        self.rho = rho
        self.adaptive_rho = adaptive_rho
        self.max_iter = max_iter
        self.tol_abs = tol_abs
        self.tol_rel = tol_rel
        self.inner_solver = inner_solver
        self.inner_max_iter = inner_max_iter
        self.upper_bound = upper_bound
        self.eps = eps
        self.verbose = verbose

    # ------------------------------------------------------------------------------
    def get_params(self, deep: bool = True) -> dict:
        keys = [
            "lambda_nuc", "lambda_l1", "likelihood", "support", "loss_scale", "rho", "adaptive_rho",
            "max_iter", "tol_abs", "tol_rel", "inner_solver", "inner_max_iter", "upper_bound", "eps", "verbose",
        ]
        return {k: getattr(self, k) for k in keys}

    def set_params(self, **params) -> "ProbC":
        for k, v in params.items():
            setattr(self, k, v)
        return self

    def clone(self, **overrides) -> "ProbC":
        p = self.get_params()
        p.update(overrides)
        return ProbC(**p)

    # ------------------------------------------------------------------------------
    def _ub(self):
        if self.upper_bound is not None:
            return self.upper_bound
        return 1.0 if self.likelihood == "poisson" else None

    def _init_theta(self, cp: CompiledPairs, m: int) -> np.ndarray:
        ones = np.ones(m)
        t_pairs = (cp.P @ ones)[cp.a] * (cp.P @ ones)[cp.b]
        tot = float(np.dot(cp.w_all, t_pairs))
        if cp.W_implicit is not None:
            tot += float(cp.W_implicit.sum())
        tbar = max(tot / max(cp.n_total, 1e-12), 1e-12)
        if self.likelihood == "bernoulli":
            p = float(cp.w_pos.sum()) / max(cp.n_total, 1e-12)
            p = min(max(p, 1e-6), 1 - 1e-6)
            z0 = -np.log1p(-p) / tbar
        else:
            z0 = float(cp.wy.sum()) / max(cp.n_total, 1e-12) / tbar
            ub = self._ub()
            if ub is not None:
                z0 = min(z0, 0.5 * ub)
        return np.full((m, m), max(z0, 1e-6))

    def fit(self, C: ArrayOrMM, pairs: PairSet, theta0: Optional[np.ndarray] = None) -> "ProbC":
        """Estimate mark-pair interaction parameters from labelled pairs.

        ``pairs`` may contain explicit positives/negatives and/or an implicit
        non-interaction term (see :func:`probc.data.build_full_pairset`).
        """
        marks = None
        if isinstance(C, MarkMatrix):
            marks = C.marks
            C = C.C
        C = np.asarray(C, dtype=np.float64)
        m = C.shape[1]
        cp = compile_pairs(C, pairs)
        if self.loss_scale == "mean":
            scale = 1.0 / max(cp.n_total, 1e-12)
        elif self.loss_scale == "sum":
            scale = 1.0
        else:
            scale = float(self.loss_scale)
        lik = LIKELIHOODS[self.likelihood]

        def fun(theta):
            f, G = lik(theta, cp, eps=self.eps)
            return scale * f, scale * G

        support = _support_mask(self.support, m, marks)
        if theta0 is None:
            theta0 = self._init_theta(cp, m)
        res: ADMMResult = admm(
            fun, m, theta0, self.lambda_nuc, self.lambda_l1, lb=0.0, ub=self._ub(), support=support,
            rho=self.rho, adaptive_rho=self.adaptive_rho, max_iter=self.max_iter, tol_abs=self.tol_abs,
            tol_rel=self.tol_rel, inner_solver=self.inner_solver, inner_max_iter=self.inner_max_iter,
            verbose=self.verbose,
        )
        self.theta_ = res.theta
        self.theta_primal_ = res.theta_primal
        self.n_iter_ = res.n_iter
        self.converged_ = res.converged
        self.history_ = res.history
        self.marks_ = list(marks) if marks is not None else [f"mark{i}" for i in range(m)]
        self.scale_ = scale
        return self

    # ------------------------------------------------------------------------------
    @property
    def Z_(self) -> np.ndarray:
        """Transformed parameters (Bernoulli: z; Poisson: the rates x themselves)."""
        return self.theta_

    @property
    def X_(self) -> np.ndarray:
        """Interaction probabilities between mark pairs, x_mn in [0, 1]."""
        if self.likelihood == "bernoulli":
            return -np.expm1(-self.theta_)
        return np.clip(self.theta_, 0.0, 1.0)

    def _C(self, C: ArrayOrMM) -> np.ndarray:
        if isinstance(C, MarkMatrix):
            if list(C.marks) != list(self.marks_):
                C = C.select_marks(self.marks_)
            return C.C
        return np.asarray(C, dtype=np.float64)

    def decision_function(self, C: ArrayOrMM, u: np.ndarray, v: np.ndarray) -> np.ndarray:
        """``C_u^T theta C_v`` (Bernoulli: -log P(no interaction); Poisson: rate)."""
        Cm = self._C(C)
        mm = MarkMatrix(Cm, self.marks_, np.zeros(len(Cm), str), np.arange(len(Cm)))
        return mark_matrix_scores(self.theta_, mm, np.asarray(u), np.asarray(v))

    def predict_proba(self, C: ArrayOrMM, u: np.ndarray, v: np.ndarray) -> np.ndarray:
        """Probability of observing at least one interaction between u and v."""
        s = np.maximum(self.decision_function(C, u, v), 0.0)
        return -np.expm1(-s)

    def predict_rate(self, C: ArrayOrMM, u: np.ndarray, v: np.ndarray) -> np.ndarray:
        """Expected number of contacts (Poisson model) or Bernoulli score."""
        return self.decision_function(C, u, v)

    def predict_matrix(self, C_bins: np.ndarray) -> np.ndarray:
        """Bin-level predicted matrix ``M_a^T theta M_b`` from per-bin mark sums.

        For the Bernoulli model this is the expected number of interacting
        restriction-site pairs in the small-probability regime; for the Poisson
        model it is exactly the expected contact count (Eq. 19).
        """
        Cb = np.asarray(C_bins, dtype=np.float64)
        return Cb @ self.theta_ @ Cb.T

    def nll(self, C: ArrayOrMM, pairs: PairSet) -> float:
        cp = compile_pairs(self._C(C), pairs)
        return LIKELIHOODS[self.likelihood](self.theta_, cp, eps=self.eps, with_grad=False)

    def score(self, C: ArrayOrMM, pairs: PairSet, weighted: bool = False) -> float:
        """ROC AUC of predicted interaction probabilities on labelled pairs.

        By default every evaluation pair counts once (``weighted=False``), even if
        sampled negatives carry inverse-sampling weights for training.
        """
        from .metrics import roc_auc

        p = self.predict_proba(C, pairs.u, pairs.v)
        return roc_auc((pairs.y > 0).astype(int), p, sample_weight=pairs.w if weighted else None)

    # ------------------------------------------------------------------------------
    def active_marks(self, tol: float = 1e-6) -> list:
        """Marks with at least one non-zero interaction parameter."""
        nz = (np.abs(self.theta_) > tol).any(axis=1)
        return [m for m, k in zip(self.marks_, nz) if k]

    def interaction_table(self, tol: float = 0.0):
        import pandas as pd

        X = self.X_
        rows = []
        for i in range(len(self.marks_)):
            for j in range(i, len(self.marks_)):
                if X[i, j] > tol:
                    rows.append((self.marks_[i], self.marks_[j], X[i, j], self.theta_[i, j]))
        df = pd.DataFrame(rows, columns=["mark_1", "mark_2", "x", "theta"])
        return df.sort_values("x", ascending=False).reset_index(drop=True)

    def save(self, path: Union[str, Path]) -> None:
        path = Path(path)
        params = self.get_params()
        if not isinstance(params["support"], str):
            params["support"] = np.asarray(params["support"]).astype(int).tolist()
        np.savez_compressed(
            path, theta=self.theta_, marks=np.array(self.marks_), params=json.dumps(params),
            history=json.dumps({k: list(map(float, v)) for k, v in self.history_.items()}),
        )

    @classmethod
    def load(cls, path: Union[str, Path]) -> "ProbC":
        d = np.load(path, allow_pickle=False)
        params = json.loads(str(d["params"]))
        model = cls(**params)
        model.theta_ = d["theta"]
        model.marks_ = [str(m) for m in d["marks"]]
        model.history_ = json.loads(str(d["history"]))
        return model
