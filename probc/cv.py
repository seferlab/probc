"""Nested cross-validation as described in the paper.

"We follow fivefold nested cross-validation to prevent overfitting and optimize for
regularization hyperparameters. When we train and test on different chromosomes,
the outer step trains PROBC on all chromosomes except the chromosome to be
predicted ... Within each loop of outer step, we estimate the regularization
hyperparameters by utilizing fourfold inner cross-validation. When we train and
test on the same chromosome, we apply similar fivefold nested cross-validation on
the single chromosome."
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from typing import Callable, Optional, Sequence

import numpy as np

from .data import MarkMatrix, PairSet
from .metrics import roc_auc
from .model import ProbC


def stratified_folds(y: np.ndarray, k: int, rng: np.random.Generator) -> np.ndarray:
    """Fold id per sample, stratified on y > 0."""
    y = np.asarray(y) > 0
    folds = np.empty(len(y), dtype=np.int64)
    for cls in (True, False):
        idx = np.flatnonzero(y == cls)
        idx = idx[rng.permutation(len(idx))]
        folds[idx] = np.arange(len(idx)) % k
    return folds


def _evaluate(model: ProbC, mm: MarkMatrix, pairs: PairSet, criterion: str) -> float:
    if criterion == "auc":
        return model.score(mm, pairs)
    if criterion == "nll":
        # lower is better -> return negative mean NLL
        return -model.nll(mm, pairs) / max(pairs.total_weight, 1e-12)
    raise ValueError(criterion)


@dataclass
class GridSearchResult:
    best_params: dict
    scores: dict = field(default_factory=dict)


def grid_search(
    base: ProbC,
    mm: MarkMatrix,
    pairs: PairSet,
    grid: dict,
    n_folds: int = 4,
    criterion: str = "auc",
    rng: Optional[np.random.Generator] = None,
) -> GridSearchResult:
    """Inner k-fold CV over a hyper-parameter grid (default: fourfold)."""
    rng = np.random.default_rng(0) if rng is None else rng
    keys = list(grid.keys())
    combos = list(itertools.product(*[grid[k] for k in keys]))
    if len(combos) == 1:
        return GridSearchResult(dict(zip(keys, combos[0])), {combos[0]: float("nan")})
    folds = stratified_folds(pairs.y, n_folds, rng)
    scores = {}
    for combo in combos:
        params = dict(zip(keys, combo))
        vals = []
        for f in range(n_folds):
            tr = pairs.subset(folds != f)
            te = pairs.subset(folds == f)
            model = base.clone(**params).fit(mm, tr)
            vals.append(_evaluate(model, mm, te, criterion))
        scores[combo] = float(np.nanmean(vals))
    best = max(scores, key=lambda c: scores[c])
    return GridSearchResult(dict(zip(keys, best)), scores)


def kfold_auc(model: ProbC, mm: MarkMatrix, pairs: PairSet, n_folds: int = 5, seed: int = 0,
              return_predictions: bool = False):
    """Plain k-fold CV AUC for fixed hyper-parameters."""
    folds = stratified_folds(pairs.y, n_folds, np.random.default_rng(seed))
    aucs = []
    y_all, p_all = [], []
    for f in range(n_folds):
        m = model.clone().fit(mm, pairs.subset(folds != f))
        te = pairs.subset(folds == f)
        p = m.predict_proba(mm, te.u, te.v)
        aucs.append(roc_auc((te.y > 0).astype(int), p))
        y_all.append((te.y > 0).astype(int))
        p_all.append(p)
    if return_predictions:
        return float(np.nanmean(aucs)), np.concatenate(y_all), np.concatenate(p_all)
    return float(np.nanmean(aucs))


@dataclass
class NestedCVResult:
    outer_scores: list
    best_params: list
    models: list
    predictions: list  # (test PairSet, predicted probabilities) per outer fold

    @property
    def mean_auc(self) -> float:
        return float(np.nanmean(self.outer_scores))


def nested_cv(
    base: ProbC,
    mm: MarkMatrix,
    pairs: PairSet,
    grid: dict,
    outer_folds: int = 5,
    inner_folds: int = 4,
    criterion: str = "auc",
    seed: int = 0,
    test_mm: Optional[MarkMatrix] = None,
    test_pairs: Optional[PairSet] = None,
) -> NestedCVResult:
    """Fivefold (outer) x fourfold (inner) nested CV on a single pair set.

    If ``test_mm/test_pairs`` are given (cross-chromosome / cross-cell-type /
    cross-species prediction), the outer test fold ``f`` of the *test* pairs is
    scored with the model trained on the outer training folds of ``pairs``.
    """
    rng = np.random.default_rng(seed)
    folds = stratified_folds(pairs.y, outer_folds, rng)
    if test_pairs is not None:
        test_folds = stratified_folds(test_pairs.y, outer_folds, np.random.default_rng(seed + 1))
    scores, best, models, preds = [], [], [], []
    for f in range(outer_folds):
        tr = pairs.subset(folds != f)
        gs = grid_search(base, mm, tr, grid, inner_folds, criterion, np.random.default_rng(seed + 100 + f))
        model = base.clone(**gs.best_params).fit(mm, tr)
        if test_pairs is None:
            te_mm, te = mm, pairs.subset(folds == f)
        else:
            te_mm, te = test_mm, test_pairs.subset(test_folds == f)
        p = model.predict_proba(te_mm, te.u, te.v)
        scores.append(roc_auc((te.y > 0).astype(int), p))
        best.append(gs.best_params)
        models.append(model)
        preds.append((te, p))
    return NestedCVResult(scores, best, models, preds)


def leave_one_chromosome_out(
    base: ProbC,
    mm: MarkMatrix,
    pairs: PairSet,
    grid: dict,
    chromosomes: Optional[Sequence[str]] = None,
    inner_folds: int = 4,
    criterion: str = "auc",
    seed: int = 0,
) -> dict:
    """Outer loop over chromosomes: train on all others, predict the held-out one."""
    chroms = chromosomes or mm.chromosomes()
    pair_chrom = mm.chrom[pairs.u]
    out = {}
    for i, c in enumerate(chroms):
        tr = pairs.subset(pair_chrom != c)
        te = pairs.subset(pair_chrom == c)
        if len(te) == 0 or len(tr) == 0:
            continue
        gs = grid_search(base, mm, tr, grid, inner_folds, criterion, np.random.default_rng(seed + i))
        model = base.clone(**gs.best_params).fit(mm, tr)
        out[c] = dict(auc=model.score(mm, te), params=gs.best_params, model=model)
    return out


def chromosome_pair_auc(
    base: ProbC,
    mm: MarkMatrix,
    pairs: PairSet,
    grid: dict,
    chromosomes: Optional[Sequence[str]] = None,
    n_folds: int = 5,
    inner_folds: int = 4,
    criterion: str = "auc",
    seed: int = 0,
    test_mm: Optional[MarkMatrix] = None,
    test_pairs: Optional[PairSet] = None,
    test_chromosomes: Optional[Sequence[str]] = None,
    progress: Optional[Callable[[str], None]] = None,
) -> tuple[np.ndarray, list, list]:
    """AUC matrix A[i, j] = train on chromosome i, test on chromosome j (Fig. 4).

    Fivefold CV is applied to each pair of chromosomes; the diagonal is ordinary
    within-chromosome nested CV. With ``test_mm/test_pairs`` the columns refer to a
    different cell type or species (Figs. 10-11).
    """
    same_dataset = test_mm is None and test_pairs is None
    test_mm = mm if test_mm is None else test_mm
    test_pairs = pairs if test_pairs is None else test_pairs
    rows = list(chromosomes or mm.chromosomes())
    cols = list(test_chromosomes or (rows if test_mm is mm else test_mm.chromosomes()))
    A = np.full((len(rows), len(cols)), np.nan)
    tr_chrom = mm.chrom[pairs.u]
    te_chrom = test_mm.chrom[test_pairs.u]
    for i, ci in enumerate(rows):
        P_i = pairs.subset(tr_chrom == ci)
        if len(P_i) == 0:
            continue
        rng = np.random.default_rng(seed + i)
        folds_i = stratified_folds(P_i.y, n_folds, rng)
        models = []
        for f in range(n_folds):
            tr = P_i.subset(folds_i != f)
            gs = grid_search(base, mm, tr, grid, inner_folds, criterion, np.random.default_rng(seed + 31 * i + f))
            models.append(base.clone(**gs.best_params).fit(mm, tr))
        for j, cj in enumerate(cols):
            P_j = test_pairs.subset(te_chrom == cj)
            if len(P_j) == 0:
                continue
            if same_dataset and cj == ci:
                folds_j = folds_i  # within-chromosome: score the held-out training fold
            else:
                folds_j = stratified_folds(P_j.y, n_folds, np.random.default_rng(seed + 1000 + j))
            aucs = []
            for f in range(n_folds):
                te = P_j.subset(folds_j == f)
                aucs.append(models[f].score(test_mm, te))
            A[i, j] = float(np.nanmean(aucs))
        if progress:
            progress(f"trained on {ci}")
    return A, rows, cols
