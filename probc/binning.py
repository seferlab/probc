"""Binning utilities: site-level data -> fixed-size genomic bins.

"For the binned scenario, we bin transcription factor binding sites, ChIP-Seq
histone modifications, Hi-C, and DNase-seq data at 1 kb resolution." The binned
mark vector of a bin is the sum of its sites' mark vectors
(``N[v'] = {(m, sum_k c^k_m)}``), and binned Hi-C edges carry total counts.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from .data import MarkMatrix, PairSet, canonical_pairs


def bin_ids(mm: MarkMatrix, resolution: int) -> tuple[np.ndarray, MarkMatrix]:
    """Assign each site to a bin and build the (dense, contiguous) binned MarkMatrix.

    Returns ``site_to_bin`` (index into the binned MarkMatrix) and the binned
    MarkMatrix whose ``C`` holds per-bin mark sums. Every bin between the first and
    last site of each chromosome is present (empty bins have zero marks).
    """
    site_to_bin = np.empty(mm.n_sites, dtype=np.int64)
    Cs, chroms, starts = [], [], []
    offset = 0
    for c in mm.chromosomes():
        idx = np.flatnonzero(mm.chrom == c)
        b = mm.pos[idx] // resolution
        bmin, bmax = int(b.min()), int(b.max())
        nb = bmax - bmin + 1
        Cb = np.zeros((nb, mm.n_marks))
        np.add.at(Cb, b - bmin, mm.C[idx])
        site_to_bin[idx] = offset + (b - bmin)
        Cs.append(Cb)
        chroms.append(np.full(nb, c))
        starts.append((np.arange(bmin, bmax + 1) * resolution).astype(np.int64))
        offset += nb
    starts = np.concatenate(starts)
    binned = MarkMatrix(np.vstack(Cs), mm.marks, np.concatenate(chroms), starts, starts + resolution,
                        dict(mm.meta, resolution=resolution))
    return site_to_bin, binned


def bin_marks(mm: MarkMatrix, resolution: int, binary: bool = False) -> MarkMatrix:
    _, b = bin_ids(mm, resolution)
    return b.binarize(0.0) if binary else b


def bin_pairs(pairs: PairSet, site_to_bin: np.ndarray, drop_diagonal: bool = True) -> PairSet:
    """Aggregate site-level interactions into bin-level counts."""
    u = site_to_bin[pairs.u]
    v = site_to_bin[pairs.v]
    u, v = canonical_pairs(u, v)
    if drop_diagonal:
        ok = u != v
        u, v, y = u[ok], v[ok], pairs.y[ok]
    else:
        y = pairs.y
    n = int(max(u.max(initial=0), v.max(initial=0)) + 1)
    keys = u * np.int64(n) + v
    uk, inv = np.unique(keys, return_inverse=True)
    counts = np.bincount(np.asarray(inv).ravel(), weights=y)
    return PairSet(uk // n, uk % n, counts)


def contact_matrix(binned: MarkMatrix, pairs: PairSet, chrom: str, symmetric: bool = True) -> np.ndarray:
    """Dense contact matrix of one chromosome from bin-level pairs."""
    idx = np.flatnonzero(binned.chrom == chrom)
    off = idx[0]
    n = len(idx)
    sel = (binned.chrom[pairs.u] == chrom) & (binned.chrom[pairs.v] == chrom)
    M = np.zeros((n, n))
    np.add.at(M, (pairs.u[sel] - off, pairs.v[sel] - off), pairs.y[sel])
    if symmetric:
        U = np.triu(M + np.tril(M, -1).T)
        M = U + np.triu(U, 1).T
    return M


def rebin_matrix(M: np.ndarray, factor: int) -> np.ndarray:
    """Sum-pool a square matrix by an integer factor."""
    n = M.shape[0]
    k = int(np.ceil(n / factor))
    pad = k * factor - n
    Mp = np.pad(M, ((0, pad), (0, pad)))
    return Mp.reshape(k, factor, k, factor).sum(axis=(1, 3))


def predicted_contact_matrix(theta: np.ndarray, binned: MarkMatrix, chrom: str,
                             distance_beta: Optional[float] = None, likelihood: str = "bernoulli") -> np.ndarray:
    """Predicted bin-level matrix ``M_a^T theta M_b`` for one chromosome.

    ``distance_beta`` optionally multiplies by a distance-decay prior ``d^-beta``
    (an extension, off by default; ProbC itself has no distance term).
    """
    idx = np.flatnonzero(binned.chrom == chrom)
    Cb = binned.C[idx]
    M = Cb @ theta @ Cb.T
    if likelihood == "bernoulli_prob":
        M = -np.expm1(-M)
    if distance_beta:
        n = len(idx)
        d = np.abs(np.arange(n)[:, None] - np.arange(n)[None, :]).astype(float)
        M = M * np.power(np.maximum(d, 1.0), -distance_beta)
    np.fill_diagonal(M, 0.0)
    return M
