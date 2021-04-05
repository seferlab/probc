"""A/B compartment calling (Lieberman-Aiden et al. 2009).

Observed/expected normalisation by genomic distance, Pearson correlation of rows,
sign of the first eigenvector. No extra parameters are needed, matching the
paper's description.
"""
from __future__ import annotations

from typing import Optional

import numpy as np


def observed_over_expected(A: np.ndarray) -> np.ndarray:
    A = np.asarray(A, dtype=float)
    n = A.shape[0]
    OE = np.zeros_like(A)
    for k in range(n):
        d = np.diagonal(A, k)
        mu = d.mean() if len(d) else 0.0
        val = d / mu if mu > 0 else np.zeros_like(d)
        idx = np.arange(n - k)
        OE[idx, idx + k] = val
        OE[idx + k, idx] = val
    return OE


def compartment_eigenvector(A: np.ndarray, orient_track: Optional[np.ndarray] = None) -> np.ndarray:
    OE = observed_over_expected(A)
    valid = OE.sum(1) > 0
    sub = OE[np.ix_(valid, valid)]
    with np.errstate(invalid="ignore", divide="ignore"):
        R = np.corrcoef(sub)
    R = np.nan_to_num(R)
    w, V = np.linalg.eigh(R)
    ev = V[:, np.argmax(w)]
    out = np.zeros(A.shape[0])
    out[valid] = ev
    if orient_track is not None:
        t = np.asarray(orient_track, float)
        if np.corrcoef(out[valid], t[valid])[0, 1] < 0:
            out = -out
    return out


def compartments_from_eigenvector(ev: np.ndarray) -> np.ndarray:
    """+1 (A) / -1 (B) / 0 (unassigned) labels."""
    return np.sign(ev).astype(int)


def compartment_domains(labels: np.ndarray) -> list:
    """Runs of identical compartment labels as (start, end) intervals."""
    labels = np.asarray(labels)
    if len(labels) == 0:
        return []
    change = np.flatnonzero(np.diff(labels) != 0) + 1
    starts = np.r_[0, change]
    ends = np.r_[change, len(labels)]
    return [(int(s), int(e)) for s, e in zip(starts, ends)]
