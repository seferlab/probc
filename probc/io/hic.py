"""Hi-C / Micro-C contact readers.

* :func:`contacts_from_pairs` -- read-level contacts (4DN ``.pairs`` or Juicer
  ``merged_nodups.txt``), each read end assigned to its nearest node (restriction
  site). This gives the paper's unbinned restriction-site interaction graph.
* :func:`contacts_from_hic` -- Juicer ``.hic`` files via ``hicstraw`` (BP bins or
  FRAG = restriction-fragment resolution).
* :func:`contacts_from_cool` -- ``.cool`` / ``.mcool`` files via ``cooler``.

All functions return a :class:`~probc.data.PairSet` indexed into a
:class:`~probc.data.MarkMatrix` that shares the same nodes.
"""
from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
import pandas as pd

from ..data import MarkMatrix, PairSet, canonical_pairs

PAIRS_COLS = ["readID", "chr1", "pos1", "chr2", "pos2", "strand1", "strand2"]
# Juicer merged_nodups "long format": str1 chr1 pos1 frag1 str2 chr2 pos2 frag2 mapq1 cigar1 seq1 mapq2 ...
MERGED_NODUPS_COLS = {"chr1": 1, "pos1": 2, "chr2": 5, "pos2": 6, "mapq1": 8, "mapq2": 11}


def _node_lookup(mm: MarkMatrix):
    """chrom -> (sorted positions, global indices)."""
    out = {}
    for c in mm.chromosomes():
        idx = np.flatnonzero(mm.chrom == c)
        order = np.argsort(mm.pos[idx], kind="stable")
        out[c] = (mm.pos[idx][order], idx[order])
    return out


def nearest_node(lookup: dict, chrom: np.ndarray, pos: np.ndarray, max_dist: Optional[int] = None) -> np.ndarray:
    out = np.full(len(pos), -1, dtype=np.int64)
    for c in np.unique(chrom):
        if c not in lookup:
            continue
        sel = np.flatnonzero(chrom == c)
        P, G = lookup[c]
        j = np.clip(np.searchsorted(P, pos[sel]), 1, len(P) - 1) if len(P) > 1 else np.zeros(len(sel), int)
        if len(P) > 1:
            left = np.abs(pos[sel] - P[j - 1])
            right = np.abs(P[j] - pos[sel])
            best = np.where(left <= right, j - 1, j)
        else:
            best = j
        d = np.abs(P[best] - pos[sel])
        ok = np.ones(len(sel), bool) if max_dist is None else d <= max_dist
        out[sel[ok]] = G[best[ok]]
    return out


def contacts_from_pairs(
    path,
    mm: MarkMatrix,
    fmt: str = "pairs",
    min_mapq: int = 30,
    max_dist_to_site: Optional[int] = 1000,
    intra_only: bool = True,
    chunksize: int = 5_000_000,
) -> PairSet:
    """Restriction-site-level contact counts from read pairs.

    fmt="pairs": 4DN pairs (``#`` header lines; chr1 pos1 chr2 pos2 at cols 2-5).
    fmt="merged_nodups": Juicer long format (MAPQ filter applied).
    """
    lookup = _node_lookup(mm)
    n = np.int64(mm.n_sites)
    keys_all, cnt_all = [], []
    if fmt == "pairs":
        reader = pd.read_csv(path, sep="\t", comment="#", header=None, usecols=[1, 2, 3, 4],
                             names=["chr1", "pos1", "chr2", "pos2"], chunksize=chunksize,
                             dtype={"chr1": str, "pos1": np.int64, "chr2": str, "pos2": np.int64})
    elif fmt == "merged_nodups":
        cols = MERGED_NODUPS_COLS
        reader = pd.read_csv(path, sep=r"\s+", header=None, chunksize=chunksize,
                             usecols=list(cols.values()), names=None, engine="c")
    else:
        raise ValueError(fmt)
    for chunk in reader:
        if fmt == "merged_nodups":
            chunk = chunk.rename(columns={v: k for k, v in MERGED_NODUPS_COLS.items()})
            chunk = chunk[(chunk["mapq1"] >= min_mapq) & (chunk["mapq2"] >= min_mapq)]
            for k in ("chr1", "chr2"):
                chunk[k] = chunk[k].astype(str)
                chunk.loc[~chunk[k].str.startswith("chr"), k] = "chr" + chunk.loc[~chunk[k].str.startswith("chr"), k]
        if intra_only:
            chunk = chunk[chunk["chr1"] == chunk["chr2"]]
        u = nearest_node(lookup, chunk["chr1"].to_numpy(), chunk["pos1"].to_numpy(np.int64), max_dist_to_site)
        v = nearest_node(lookup, chunk["chr2"].to_numpy(), chunk["pos2"].to_numpy(np.int64), max_dist_to_site)
        ok = (u >= 0) & (v >= 0) & (u != v)
        u, v = canonical_pairs(u[ok], v[ok])
        k, c = np.unique(u * n + v, return_counts=True)
        keys_all.append(k)
        cnt_all.append(c)
    if not keys_all:
        return PairSet(np.zeros(0), np.zeros(0), np.zeros(0))
    keys = np.concatenate(keys_all)
    cnts = np.concatenate(cnt_all).astype(float)
    uk, inv = np.unique(keys, return_inverse=True)
    counts = np.bincount(np.asarray(inv).ravel(), weights=cnts)
    return PairSet(uk // n, uk % n, counts)


def contacts_from_hic(
    path,
    mm: MarkMatrix,
    chroms: Sequence[str],
    resolution: int,
    unit: str = "BP",
    normalization: str = "NONE",
) -> PairSet:
    """Intra-chromosomal contacts from a Juicer .hic file (needs ``hicstraw``).

    For ``unit="BP"`` ``mm`` must be a binned MarkMatrix at ``resolution``; for
    ``unit="FRAG"`` its nodes must be the restriction sites used to build the .hic
    (fragment i is mapped to node i of that chromosome).
    """
    import hicstraw

    hic = hicstraw.HiCFile(str(path))
    us, vs, ys = [], [], []
    lookup = _node_lookup(mm)
    for c in chroms:
        cname = c if c in [x.name for x in hic.getChromosomes()] else c.replace("chr", "")
        recs = hicstraw.straw("observed", normalization, str(path), cname, cname, unit, resolution)
        if not recs:
            continue
        a = np.fromiter((r.binX for r in recs), dtype=np.int64, count=len(recs))
        b = np.fromiter((r.binY for r in recs), dtype=np.int64, count=len(recs))
        y = np.fromiter((r.counts for r in recs), dtype=float, count=len(recs))
        P, G = lookup[c]
        if unit == "BP":
            ia = np.searchsorted(P, a)
            ib = np.searchsorted(P, b)
            ok = (ia < len(P)) & (ib < len(P))
            ok[ok] &= (P[ia[ok]] == a[ok]) & (P[ib[ok]] == b[ok])
            u, v = G[ia[ok]], G[ib[ok]]
        else:
            ok = (a < len(G)) & (b < len(G))
            u, v = G[a[ok]], G[b[ok]]
        y = y[ok]
        keep = u != v
        u, v = canonical_pairs(u[keep], v[keep])
        us.append(u)
        vs.append(v)
        ys.append(y[keep])
    return PairSet(np.concatenate(us), np.concatenate(vs), np.concatenate(ys))


def contacts_from_cool(path, mm: MarkMatrix, chroms: Sequence[str], balance: bool = False) -> PairSet:
    """Intra-chromosomal contacts from a .cool (or ``file.mcool::resolutions/N``)."""
    import cooler

    clr = cooler.Cooler(str(path))
    lookup = _node_lookup(mm)
    us, vs, ys = [], [], []
    for c in chroms:
        cname = c if c in clr.chromnames else c.replace("chr", "")
        pix = clr.matrix(balance=balance, as_pixels=True, join=True).fetch(cname)
        if len(pix) == 0:
            continue
        a = pix["start1"].to_numpy(np.int64)
        b = pix["start2"].to_numpy(np.int64)
        y = (pix["balanced"] if balance else pix["count"]).to_numpy(float)
        P, G = lookup[c]
        ia = np.searchsorted(P, a)
        ib = np.searchsorted(P, b)
        ok = (ia < len(P)) & (ib < len(P))
        ok[ok] &= (P[ia[ok]] == a[ok]) & (P[ib[ok]] == b[ok])
        u, v = G[ia[ok]], G[ib[ok]]
        y = np.nan_to_num(y[ok])
        keep = (u != v) & (y > 0)
        u, v = canonical_pairs(u[keep], v[keep])
        us.append(u)
        vs.append(v)
        ys.append(y[keep])
    return PairSet(np.concatenate(us), np.concatenate(vs), np.concatenate(ys))


def write_pairs(path, mm: MarkMatrix, pairs: PairSet) -> None:
    """Write a minimal 4DN-style pairs file (one line per contact count unit)."""
    with open(path, "w") as fh:
        fh.write("## pairs format v1.0\n#columns: readID chr1 pos1 chr2 pos2 strand1 strand2\n")
        k = 0
        for u, v, y in zip(pairs.u, pairs.v, pairs.y):
            for _ in range(int(round(y))):
                fh.write(f"r{k}\t{mm.chrom[u]}\t{mm.pos[u]}\t{mm.chrom[v]}\t{mm.pos[v]}\t+\t-\n")
                k += 1
