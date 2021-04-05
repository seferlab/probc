"""Chromatin-mark inputs: peak files -> per-site / per-bin mark counts.

Unbinned (paper default): "we map each chromatin mark location to near Hi-C
restriction sites. A chromatin mark belongs to a Hi-C restriction site if the mark
is at most 100 away from the restriction site."

Binned: "we bin ... data at 1 kb resolution. Then, we estimate Reads Per Kilobase
per Million (RPKM) in each bin by logarithmic transformation ... When there are more
than one replicates ... averaging out the RPKM-level in each bin. Afterwards, we
normalize such values into binary values by simple thresholding at 0.5."
"""
from __future__ import annotations

from typing import Dict, Iterable, Optional, Sequence

import numpy as np
import pandas as pd

from ..data import MarkMatrix

BED_COLS = ["chrom", "start", "end", "name", "score", "strand", "signal", "p", "q", "summit"]


def read_bed(path, chroms: Optional[Iterable[str]] = None) -> pd.DataFrame:
    """Read BED / narrowPeak / broadPeak (gzipped or not)."""
    df = pd.read_csv(path, sep="\t", header=None, comment="#", compression="infer", low_memory=False)
    df = df[~df[0].astype(str).str.startswith(("track", "browser"))]
    ncol = min(df.shape[1], len(BED_COLS))
    df = df.iloc[:, :ncol]
    df.columns = BED_COLS[:ncol]
    df["chrom"] = df["chrom"].astype(str)
    df["start"] = df["start"].astype(np.int64)
    df["end"] = df["end"].astype(np.int64)
    if chroms is not None:
        df = df[df["chrom"].isin(set(chroms))]
    return df.reset_index(drop=True)


def peak_points(df: pd.DataFrame, mode: str = "interval") -> pd.DataFrame:
    """Return peak intervals used for site mapping.

    mode="interval": the full peak interval; mode="summit": the summit (narrowPeak
    column 10) or the midpoint if no summit is given.
    """
    if mode == "interval":
        return df[["chrom", "start", "end"]].copy()
    if "summit" in df.columns and (df["summit"] >= 0).all():
        s = df["start"] + df["summit"].astype(np.int64)
    else:
        s = (df["start"] + df["end"]) // 2
    return pd.DataFrame({"chrom": df["chrom"], "start": s, "end": s + 1})


def count_near_nodes(node_start: np.ndarray, node_end: np.ndarray, p_start: np.ndarray, p_end: np.ndarray,
                     max_dist: int) -> np.ndarray:
    """For sorted, non-overlapping nodes [start, end] count peaks within ``max_dist``.

    A peak [ps, pe) is near a node [ns, ne] if the gap between the two intervals is
    at most ``max_dist`` (0 if they overlap). Restriction sites are nodes with
    ``ns == ne``; bins are nodes with ``ne = ns + resolution - 1`` and ``max_dist=0``.
    """
    n = len(node_start)
    lo = np.searchsorted(node_end, p_start - max_dist, side="left")
    hi = np.searchsorted(node_start, p_end - 1 + max_dist, side="right")
    ok = hi > lo
    diff = np.zeros(n + 1, dtype=np.int64)
    np.add.at(diff, lo[ok], 1)
    np.add.at(diff, hi[ok], -1)
    return np.cumsum(diff)[:n]


def site_mark_matrix(
    sites: Dict[str, np.ndarray],
    mark_peaks: Dict[str, pd.DataFrame],
    max_dist: int = 100,
    mode: str = "interval",
    binary: bool = False,
    node_end: Optional[Dict[str, np.ndarray]] = None,
    drop_empty: bool = False,
) -> MarkMatrix:
    """Build the unbinned mark matrix C (sites x marks).

    ``sites``: chrom -> sorted restriction-site coordinates (or node starts).
    ``mark_peaks``: mark name -> peak DataFrame (``read_bed``).
    ``node_end``: optional chrom -> node ends (fragments / bins); defaults to points.
    """
    marks = list(mark_peaks.keys())
    Cs, chroms, poss = [], [], []
    for c, pos in sites.items():
        pos = np.asarray(pos, dtype=np.int64)
        order = np.argsort(pos, kind="stable")
        pos = pos[order]
        ends = pos if node_end is None else np.asarray(node_end[c], dtype=np.int64)[order]
        C = np.zeros((len(pos), len(marks)))
        for j, m in enumerate(marks):
            pk = peak_points(mark_peaks[m], mode)
            pk = pk[pk["chrom"] == c]
            if len(pk) == 0:
                continue
            C[:, j] = count_near_nodes(pos, ends, pk["start"].to_numpy(), pk["end"].to_numpy(), max_dist)
        Cs.append(C)
        chroms.append(np.full(len(pos), c))
        poss.append(pos)
    mm = MarkMatrix(np.vstack(Cs), marks, np.concatenate(chroms), np.concatenate(poss))
    if binary:
        mm = mm.binarize(0)
    if drop_empty:
        mm, _ = mm.drop_empty_sites()
    return mm


# ----------------------------------------------------------------------------------
# Binned signals
# ----------------------------------------------------------------------------------


def make_bins(chrom_sizes: Dict[str, int], resolution: int, chroms: Optional[Sequence[str]] = None) -> pd.DataFrame:
    rows = []
    for c, L in chrom_sizes.items():
        if chroms is not None and c not in chroms:
            continue
        st = np.arange(0, L, resolution, dtype=np.int64)
        rows.append(pd.DataFrame({"chrom": c, "start": st, "end": np.minimum(st + resolution, L)}))
    return pd.concat(rows, ignore_index=True)


def reads_rpkm(bins: pd.DataFrame, reads: pd.DataFrame, total_reads: Optional[int] = None) -> np.ndarray:
    """RPKM per bin from read (tagAlign / BED) 5' positions."""
    total = len(reads) if total_reads is None else total_reads
    out = np.zeros(len(bins))
    for c, sub in bins.groupby("chrom", sort=False):
        r = reads[reads["chrom"] == c]
        if len(r) == 0:
            continue
        five = np.where(r.get("strand", pd.Series("+", index=r.index)).to_numpy() == "-", r["end"] - 1, r["start"])
        b = np.searchsorted(sub["start"].to_numpy(), five, side="right") - 1
        b = b[(b >= 0) & (b < len(sub))]
        cnt = np.bincount(b, minlength=len(sub)).astype(float)
        length_kb = (sub["end"] - sub["start"]).to_numpy() / 1e3
        out[sub.index.to_numpy()] = cnt / length_kb / (total / 1e6)
    return out


def bigwig_bin_mean(bins: pd.DataFrame, bigwig_path) -> np.ndarray:
    """Mean bigWig signal per bin (requires pyBigWig)."""
    import pyBigWig

    bw = pyBigWig.open(str(bigwig_path))
    out = np.zeros(len(bins))
    chroms = bw.chroms()
    for c, sub in bins.groupby("chrom", sort=False):
        if c not in chroms:
            continue
        L = chroms[c]
        nb = len(sub)
        vals = bw.stats(c, 0, min(int(sub["end"].max()), L), type="mean", nBins=nb, exact=False)
        out[sub.index.to_numpy()] = np.nan_to_num(np.asarray(vals, dtype=float))
    bw.close()
    return out


def peak_coverage(bins: pd.DataFrame, peaks: pd.DataFrame) -> np.ndarray:
    """Number of peaks overlapping each bin (fallback when only peaks are available)."""
    out = np.zeros(len(bins))
    for c, sub in bins.groupby("chrom", sort=False):
        pk = peaks[peaks["chrom"] == c]
        if len(pk) == 0:
            continue
        s = sub["start"].to_numpy()
        e = sub["end"].to_numpy() - 1
        out[sub.index.to_numpy()] = count_near_nodes(s, e, pk["start"].to_numpy(), pk["end"].to_numpy(), 0)
    return out


def binarize_signal(values: np.ndarray, threshold: float = 0.5, transform: str = "log1p", scale: str = "none") -> np.ndarray:
    """log-transform RPKM, optionally rescale, and threshold (paper: 0.5).

    The paper does not specify whether values are rescaled before thresholding;
    ``scale`` = "none" thresholds log(1 + RPKM) directly, "minmax" rescales to
    [0, 1] first, "quantile" uses the empirical CDF.
    """
    v = np.asarray(values, float)
    if transform == "log1p":
        v = np.log1p(np.maximum(v, 0))
    elif transform == "log2p1":
        v = np.log2(1 + np.maximum(v, 0))
    if scale == "minmax":
        lo, hi = v.min(), v.max()
        v = (v - lo) / (hi - lo) if hi > lo else np.zeros_like(v)
    elif scale == "quantile":
        from scipy.stats import rankdata

        v = rankdata(v) / len(v)
    return (v > threshold).astype(float)


def binned_mark_matrix(bins: pd.DataFrame, signals: Dict[str, Sequence[np.ndarray]], threshold: float = 0.5,
                       transform: str = "log1p", scale: str = "none", binary: bool = True) -> MarkMatrix:
    """Average replicates (after transform) and binarize, per mark."""
    marks = list(signals.keys())
    C = np.zeros((len(bins), len(marks)))
    for j, m in enumerate(marks):
        reps = [np.asarray(r, float) for r in signals[m]]
        mean = np.mean(np.vstack(reps), axis=0)
        if binary:
            C[:, j] = binarize_signal(mean, threshold, transform, scale)
        else:
            C[:, j] = np.log1p(np.maximum(mean, 0)) if transform == "log1p" else mean
    return MarkMatrix(C, marks, bins["chrom"].to_numpy(), bins["start"].to_numpy(), bins["end"].to_numpy())
