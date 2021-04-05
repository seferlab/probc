"""Evaluation metrics used in the paper.

* ROC AUC / ROC curves for interaction prediction (Figs. 4, 7, 8, 10-12).
* Normalized Variation of Information (NVI, Meila 2007) between domain partitions
  (Fig. 9).
* TAD boundary confusion counts (Table 3).
* Spearman correlation of interaction-probability matrices (Fig. 13) and the
  stratum-adjusted correlation coefficient (SCC, HiCRep; Yang et al. 2017).
"""
from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
from scipy import stats


def roc_auc(y_true, y_score, sample_weight=None) -> float:
    from sklearn.metrics import roc_auc_score

    y_true = np.asarray(y_true)
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(roc_auc_score(y_true, y_score, sample_weight=sample_weight))


def roc_curve(y_true, y_score, sample_weight=None):
    """ROC curve obtained by sweeping the probability threshold from 1 to 0."""
    from sklearn.metrics import roc_curve as _rc

    fpr, tpr, thr = _rc(y_true, y_score, sample_weight=sample_weight)
    return fpr, tpr, thr


def average_precision(y_true, y_score) -> float:
    from sklearn.metrics import average_precision_score

    return float(average_precision_score(y_true, y_score))


# ----------------------------------------------------------------------------------
# Partitions and NVI
# ----------------------------------------------------------------------------------


def domains_to_labels(domains: Sequence[tuple], n_bins: int) -> np.ndarray:
    """Convert a list of (start, end) bin intervals (end exclusive) to cluster labels.

    Bins not covered by any domain get their own singleton label, following the
    convention used when comparing Armatus domain sets.
    """
    labels = -np.ones(n_bins, dtype=np.int64)
    for k, (s, e) in enumerate(domains):
        labels[int(s):int(e)] = k
    nxt = len(domains)
    for i in np.flatnonzero(labels < 0):
        labels[i] = nxt
        nxt += 1
    return labels


def variation_of_information(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a)
    b = np.asarray(b)
    n = len(a)
    _, ai = np.unique(a, return_inverse=True)
    _, bi = np.unique(b, return_inverse=True)
    joint = np.zeros((ai.max() + 1, bi.max() + 1))
    np.add.at(joint, (ai, bi), 1.0)
    p = joint / n
    pa = p.sum(1)
    pb = p.sum(0)
    nz = p > 0
    h_a = -np.sum(pa[pa > 0] * np.log(pa[pa > 0]))
    h_b = -np.sum(pb[pb > 0] * np.log(pb[pb > 0]))
    mi = np.sum(p[nz] * np.log(p[nz] / np.outer(pa, pb)[nz]))
    return float(h_a + h_b - 2 * mi)


def normalized_vi(a: np.ndarray, b: np.ndarray) -> float:
    """VI(a, b) / log(n) in [0, 1]; higher means more dissimilar partitions."""
    n = len(a)
    if n <= 1:
        return 0.0
    return variation_of_information(a, b) / np.log(n)


def nvi_domains(pred_domains, true_domains, n_bins: int) -> float:
    return normalized_vi(domains_to_labels(pred_domains, n_bins), domains_to_labels(true_domains, n_bins))


# ----------------------------------------------------------------------------------
# TAD boundaries
# ----------------------------------------------------------------------------------


def domain_boundaries(domains: Sequence[tuple]) -> np.ndarray:
    b = set()
    for s, e in domains:
        b.add(int(s))
        b.add(int(e))
    return np.array(sorted(b), dtype=np.int64)


def boundary_confusion(pred_boundaries, true_boundaries, tolerance: int = 0) -> dict:
    """Counts as in Table 3: TP, FN, FP, rates, #predicted, #true.

    A true boundary is recovered (TP) if a predicted boundary lies within
    ``tolerance`` bins of it.
    """
    pred = np.unique(np.asarray(pred_boundaries, dtype=np.int64))
    true = np.unique(np.asarray(true_boundaries, dtype=np.int64))
    if len(true) == 0:
        return dict(TP=0, FN=0, FP=len(pred), TPR=float("nan"), FNR=float("nan"), n_pred=len(pred), n_true=0)
    if len(pred) == 0:
        return dict(TP=0, FN=len(true), FP=0, TPR=0.0, FNR=1.0, n_pred=0, n_true=len(true))
    idx = np.clip(np.searchsorted(pred, true), 1, len(pred) - 1) if len(pred) > 1 else np.zeros(len(true), int)
    if len(pred) > 1:
        d = np.minimum(np.abs(pred[idx] - true), np.abs(pred[idx - 1] - true))
    else:
        d = np.abs(pred[0] - true)
    tp = int((d <= tolerance).sum())
    idx2 = np.clip(np.searchsorted(true, pred), 1, len(true) - 1) if len(true) > 1 else np.zeros(len(pred), int)
    if len(true) > 1:
        d2 = np.minimum(np.abs(true[idx2] - pred), np.abs(true[idx2 - 1] - pred))
    else:
        d2 = np.abs(true[0] - pred)
    fp = int((d2 > tolerance).sum())
    fn = len(true) - tp
    return dict(TP=tp, FN=fn, FP=fp, TPR=tp / len(true), FNR=fn / len(true), n_pred=len(pred), n_true=len(true))


# ----------------------------------------------------------------------------------
# Matrix similarity
# ----------------------------------------------------------------------------------


def upper_triangle(M: np.ndarray, k: int = 0) -> np.ndarray:
    iu = np.triu_indices(M.shape[0], k)
    return M[iu]


def spearman_matrices(A: np.ndarray, B: np.ndarray, k: int = 0) -> float:
    return float(stats.spearmanr(upper_triangle(A, k), upper_triangle(B, k)).statistic)


def pearson_matrices(A: np.ndarray, B: np.ndarray, k: int = 0) -> float:
    return float(np.corrcoef(upper_triangle(A, k), upper_triangle(B, k))[0, 1])


def _mean_filter(M: np.ndarray, h: int) -> np.ndarray:
    if h <= 0:
        return M
    from scipy.ndimage import uniform_filter

    return uniform_filter(M, size=2 * h + 1, mode="constant")


def scc(A: np.ndarray, B: np.ndarray, h: int = 1, max_diag: Optional[int] = None) -> float:
    """Stratum-adjusted correlation coefficient (HiCRep, Yang et al. 2017)."""
    A = _mean_filter(np.asarray(A, float), h)
    B = _mean_filter(np.asarray(B, float), h)
    n = A.shape[0]
    max_diag = n - 1 if max_diag is None else min(max_diag, n - 1)
    num, den = 0.0, 0.0
    for k in range(1, max_diag + 1):
        a = np.diagonal(A, k)
        b = np.diagonal(B, k)
        ok = (a > 0) | (b > 0)
        a, b = a[ok], b[ok]
        Nk = len(a)
        if Nk < 3:
            continue
        ra = stats.rankdata(a) / Nk
        rb = stats.rankdata(b) / Nk
        rk = np.sqrt(np.var(ra) * np.var(rb))
        if rk == 0 or a.std() == 0 or b.std() == 0:
            continue
        corr = np.corrcoef(a, b)[0, 1]
        num += Nk * rk * corr
        den += Nk * rk
    return float(num / den) if den > 0 else float("nan")
