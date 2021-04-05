"""Core data containers for ProbC.

* :class:`MarkMatrix` -- the mark-count matrix ``C`` (sites/bins x marks), i.e. the
  vectors ``C_u = [c^u_m, m in M]`` of the paper, together with genomic coordinates.
* :class:`PairSet` -- a set of (u, v) site pairs with labels (interaction present /
  absent for the unbinned Bernoulli model, contact counts for the binned Poisson
  model), optional weights and an optional *implicit* linear term that represents
  all remaining non-interacting pairs exactly without enumerating them.

The implicit term is what makes the paper's likelihood (Eq. 1), a product over *all*
pairs of restriction sites, tractable: for the Bernoulli model the contribution of a
non-interacting pair is ``C_u^T Z C_v`` (linear in Z), so the sum over all
non-interacting pairs collapses into ``<Z, W>`` with ``W = sum_{(u,v) not in E} C_u C_v^T``.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Optional, Sequence

import numpy as np

# ----------------------------------------------------------------------------------
# Mark matrix
# ----------------------------------------------------------------------------------


@dataclass
class MarkMatrix:
    """Mark counts per genomic site (restriction fragment / bin).

    Attributes
    ----------
    C : (n_sites, n_marks) array
        ``C[u, m]`` = number of peaks of mark ``m`` near site ``u`` (or 0/1 presence).
    marks : list of str
        Mark names (columns of C).
    chrom : (n_sites,) array of str
        Chromosome of each site.
    pos : (n_sites,) int array
        Genomic coordinate of each site (restriction site position or bin start).
    end : (n_sites,) int array, optional
        Bin end (binned data only).
    """

    C: np.ndarray
    marks: list
    chrom: np.ndarray
    pos: np.ndarray
    end: Optional[np.ndarray] = None
    meta: dict = field(default_factory=dict)

    def __post_init__(self):
        self.C = np.asarray(self.C, dtype=np.float64)
        self.chrom = np.asarray(self.chrom).astype(str)
        self.pos = np.asarray(self.pos, dtype=np.int64)
        if self.end is not None:
            self.end = np.asarray(self.end, dtype=np.int64)
        self.marks = list(self.marks)
        if self.C.ndim != 2 or self.C.shape[0] != len(self.pos) or self.C.shape[1] != len(self.marks):
            raise ValueError(
                f"Inconsistent shapes: C={self.C.shape}, pos={len(self.pos)}, marks={len(self.marks)}"
            )

    # -- basic properties --------------------------------------------------------
    @property
    def n_sites(self) -> int:
        return self.C.shape[0]

    @property
    def n_marks(self) -> int:
        return self.C.shape[1]

    def chromosomes(self) -> list:
        _, idx = np.unique(self.chrom, return_index=True)
        return [str(c) for c in self.chrom[np.sort(idx)]]

    # -- manipulation --------------------------------------------------------------
    def subset_sites(self, mask_or_idx) -> "MarkMatrix":
        idx = _as_index(mask_or_idx, self.n_sites)
        return MarkMatrix(
            C=self.C[idx],
            marks=self.marks,
            chrom=self.chrom[idx],
            pos=self.pos[idx],
            end=None if self.end is None else self.end[idx],
            meta=dict(self.meta),
        )

    def subset_chrom(self, chroms: Iterable[str] | str) -> "MarkMatrix":
        if isinstance(chroms, str):
            chroms = [chroms]
        return self.subset_sites(np.isin(self.chrom, list(chroms)))

    def select_marks(self, marks: Sequence[str]) -> "MarkMatrix":
        missing = [m for m in marks if m not in self.marks]
        if missing:
            raise KeyError(f"Marks not present: {missing}")
        cols = [self.marks.index(m) for m in marks]
        return MarkMatrix(self.C[:, cols], list(marks), self.chrom, self.pos, self.end, dict(self.meta))

    def binarize(self, threshold: float = 0.0) -> "MarkMatrix":
        return MarkMatrix((self.C > threshold).astype(float), self.marks, self.chrom, self.pos, self.end, dict(self.meta))

    def with_background(self, name: str = "BACKGROUND") -> "MarkMatrix":
        """Add a pseudo-mark present at every site (an intercept / background rate).

        This is *not* part of the paper's model; it is offered as an option because
        sites with no marks otherwise have interaction probability exactly 0.
        """
        C = np.hstack([self.C, np.ones((self.n_sites, 1))])
        return MarkMatrix(C, self.marks + [name], self.chrom, self.pos, self.end, dict(self.meta))

    def drop_empty_sites(self) -> tuple["MarkMatrix", np.ndarray]:
        """Remove sites carrying no marks. Returns (new MarkMatrix, kept old indices)."""
        keep = np.flatnonzero(self.C.sum(1) > 0)
        return self.subset_sites(keep), keep

    def sort(self) -> tuple["MarkMatrix", np.ndarray]:
        order = np.lexsort((self.pos, self.chrom))
        return self.subset_sites(order), order

    def mark_index(self, names: Sequence[str]) -> list:
        return [self.marks.index(n) for n in names]


def align_marks(a: MarkMatrix, b: MarkMatrix) -> tuple[MarkMatrix, MarkMatrix, list]:
    """Restrict two mark matrices to their shared marks (in ``a``'s order).

    Used for cross-cell-type / cross-species prediction where the available mark
    panels differ (Table 1 of the paper).
    """
    shared = [m for m in a.marks if m in b.marks]
    if not shared:
        raise ValueError("No shared marks between the two datasets")
    return a.select_marks(shared), b.select_marks(shared), shared


# ----------------------------------------------------------------------------------
# Pair sets
# ----------------------------------------------------------------------------------


@dataclass
class PairSet:
    """Pairs of site indices with labels.

    Attributes
    ----------
    u, v : int arrays
        Site indices into a :class:`MarkMatrix` (``u < v`` by convention).
    y : float array
        Bernoulli model: 1 = interaction observed, 0 = not observed.
        Poisson model: observed contact count.
    w : float array
        Per-pair weights (e.g. inverse sampling rate of negatives).
    W_implicit : (m, m) array, optional
        ``sum C_u C_v^T`` over additional *implicit* pairs whose label is 0
        (non-interacting / zero count). These pairs contribute ``<theta, W>`` to the
        negative log-likelihood of both models.
    n_implicit : float
        Number (or total weight) of implicit pairs, used for loss normalisation.
    """

    u: np.ndarray
    v: np.ndarray
    y: np.ndarray
    w: Optional[np.ndarray] = None
    W_implicit: Optional[np.ndarray] = None
    n_implicit: float = 0.0

    def __post_init__(self):
        self.u = np.asarray(self.u, dtype=np.int64)
        self.v = np.asarray(self.v, dtype=np.int64)
        self.y = np.asarray(self.y, dtype=np.float64)
        if self.w is None:
            self.w = np.ones_like(self.y)
        else:
            self.w = np.asarray(self.w, dtype=np.float64)
        if not (len(self.u) == len(self.v) == len(self.y) == len(self.w)):
            raise ValueError("u, v, y, w must have equal length")

    def __len__(self) -> int:
        return len(self.u)

    @property
    def total_weight(self) -> float:
        return float(self.w.sum() + self.n_implicit)

    def subset(self, idx) -> "PairSet":
        idx = _as_index(idx, len(self))
        return PairSet(self.u[idx], self.v[idx], self.y[idx], self.w[idx])

    def binary(self, min_count: float = 1.0) -> "PairSet":
        return PairSet(self.u, self.v, (self.y >= min_count).astype(float), self.w, self.W_implicit, self.n_implicit)

    @staticmethod
    def concat(sets: Sequence["PairSet"]) -> "PairSet":
        W = None
        n_imp = 0.0
        for s in sets:
            if s.W_implicit is not None:
                W = s.W_implicit.copy() if W is None else W + s.W_implicit
                n_imp += s.n_implicit
        return PairSet(
            np.concatenate([s.u for s in sets]),
            np.concatenate([s.v for s in sets]),
            np.concatenate([s.y for s in sets]),
            np.concatenate([s.w for s in sets]),
            W,
            n_imp,
        )

    def keys(self, n_sites: int) -> np.ndarray:
        a = np.minimum(self.u, self.v)
        b = np.maximum(self.u, self.v)
        return a * np.int64(n_sites) + b


def _as_index(mask_or_idx, n: int) -> np.ndarray:
    arr = np.asarray(mask_or_idx)
    if arr.dtype == bool:
        if len(arr) != n:
            raise ValueError("Boolean mask has wrong length")
        return np.flatnonzero(arr)
    return arr.astype(np.int64)


def canonical_pairs(u: np.ndarray, v: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    u = np.asarray(u, dtype=np.int64)
    v = np.asarray(v, dtype=np.int64)
    return np.minimum(u, v), np.maximum(u, v)


# ----------------------------------------------------------------------------------
# Pair universe ("all pairs of restriction sites" in Eq. 1)
# ----------------------------------------------------------------------------------


@dataclass
class Universe:
    """Defines which site pairs are modelled.

    scope : "intra" (same chromosome only) or "genome" (all pairs).
    min_dist, max_dist : genomic-distance window for intra-chromosomal pairs
        (``max_dist=None`` means unbounded). The paper models all pairs; a finite
        ``max_dist`` is a practical option (commonly 1-2 Mb in Hi-C prediction work).
    """

    scope: str = "intra"
    min_dist: int = 1
    max_dist: Optional[int] = None

    def contains(self, mm: MarkMatrix, u: np.ndarray, v: np.ndarray) -> np.ndarray:
        same = mm.chrom[u] == mm.chrom[v]
        if self.scope == "genome":
            ok = u != v
            d = np.abs(mm.pos[u] - mm.pos[v])
            return ok & (~same | (d >= self.min_dist))
        d = np.abs(mm.pos[u] - mm.pos[v])
        ok = same & (d >= self.min_dist) & (u != v)
        if self.max_dist is not None:
            ok &= d <= self.max_dist
        return ok


def _window_bounds(pos_sorted: np.ndarray, min_dist: int, max_dist: Optional[int]) -> tuple[np.ndarray, np.ndarray]:
    """For each i (sorted positions) return [lo_i, hi_i) index range of partners j>i."""
    n = len(pos_sorted)
    idx = np.arange(n)
    lo = np.searchsorted(pos_sorted, pos_sorted + max(min_dist, 0), side="left")
    lo = np.maximum(lo, idx + 1)
    if max_dist is None:
        hi = np.full(n, n, dtype=np.int64)
    else:
        hi = np.searchsorted(pos_sorted, pos_sorted + max_dist, side="right")
    hi = np.maximum(hi, lo)
    return lo.astype(np.int64), hi.astype(np.int64)


def universe_linear_term(mm: MarkMatrix, universe: Universe) -> tuple[np.ndarray, float]:
    """Exact ``W_U = sum_{(u<v) in U} C_u C_v^T`` (symmetrised) and ``|U|``.

    Runs in O(n m^2) using prefix sums over position-sorted sites, so it never
    enumerates the O(n^2) pairs.
    """
    m = mm.n_marks
    W = np.zeros((m, m))
    count = 0.0
    if universe.scope == "genome":
        # all pairs within each chromosome restricted by min_dist, plus all inter pairs
        S_tot = mm.C.sum(0)
        inter = 0.5 * (np.outer(S_tot, S_tot))
        n_tot = mm.n_sites
        inter_count = 0.5 * n_tot * n_tot
        for c in mm.chromosomes():
            idx = np.flatnonzero(mm.chrom == c)
            S_c = mm.C[idx].sum(0)
            inter -= 0.5 * np.outer(S_c, S_c)
            inter_count -= 0.5 * len(idx) * len(idx)
        W += inter
        count += inter_count
        intra_universe = Universe("intra", universe.min_dist, None)
        W_in, n_in = universe_linear_term(mm, intra_universe)
        W += W_in
        count += n_in
        return 0.5 * (W + W.T), count

    for c in mm.chromosomes():
        idx = np.flatnonzero(mm.chrom == c)
        order = np.argsort(mm.pos[idx], kind="stable")
        idx = idx[order]
        Cc = mm.C[idx]
        pos = mm.pos[idx]
        lo, hi = _window_bounds(pos, universe.min_dist, universe.max_dist)
        P = np.vstack([np.zeros((1, m)), np.cumsum(Cc, axis=0)])
        window = P[hi] - P[lo]  # (n_c, m): sum of partners' mark vectors
        W += Cc.T @ window
        count += float((hi - lo).sum())
    return 0.5 * (W + W.T), count


def pair_outer_sum(mm: MarkMatrix, u: np.ndarray, v: np.ndarray, w: Optional[np.ndarray] = None,
                   chunk: int = 200_000) -> np.ndarray:
    """``sum_e w_e C_{u_e} C_{v_e}^T`` (symmetrised), chunked."""
    m = mm.n_marks
    G = np.zeros((m, m))
    for s in range(0, len(u), chunk):
        cu = mm.C[u[s:s + chunk]]
        cv = mm.C[v[s:s + chunk]]
        if w is not None:
            cv = cv * w[s:s + chunk, None]
        G += cu.T @ cv
    return 0.5 * (G + G.T)


def build_full_pairset(mm: MarkMatrix, edges: PairSet, universe: Universe) -> PairSet:
    """Training set representing *all* pairs of the universe exactly.

    Explicit pairs = observed interactions (within the universe); every other pair
    in the universe is a non-interaction represented implicitly through
    ``W_implicit``. This is the paper's genome-wide likelihood (Eqs. 1-4).
    """
    inside = universe.contains(mm, edges.u, edges.v)
    pos = edges.subset(np.flatnonzero(inside & (edges.y > 0)))
    W_U, n_U = universe_linear_term(mm, universe)
    W_E = pair_outer_sum(mm, pos.u, pos.v)
    pos.W_implicit = W_U - W_E
    pos.n_implicit = n_U - len(pos)
    return pos


def enumerate_universe_pairs(mm: MarkMatrix, universe: Universe, max_pairs: Optional[int] = None,
                             rng: Optional[np.random.Generator] = None) -> tuple[np.ndarray, np.ndarray]:
    """All intra-chromosomal universe pairs (u < v), optionally subsampled."""
    us, vs = [], []
    for c in mm.chromosomes():
        idx = np.flatnonzero(mm.chrom == c)
        order = np.argsort(mm.pos[idx], kind="stable")
        idx = idx[order]
        lo, hi = _window_bounds(mm.pos[idx], universe.min_dist, universe.max_dist)
        sizes = hi - lo
        i = np.repeat(np.arange(len(idx)), sizes)
        start = np.repeat(lo - np.r_[0, np.cumsum(sizes)[:-1]], sizes)
        j = start + np.arange(sizes.sum())
        u, v = canonical_pairs(idx[i], idx[j])
        us.append(u)
        vs.append(v)
    u = np.concatenate(us) if us else np.zeros(0, np.int64)
    v = np.concatenate(vs) if vs else np.zeros(0, np.int64)
    if max_pairs is not None and len(u) > max_pairs:
        rng = np.random.default_rng(0) if rng is None else rng
        sel = rng.choice(len(u), size=max_pairs, replace=False)
        u, v = u[sel], v[sel]
    return u, v


# ----------------------------------------------------------------------------------
# Negative sampling
# ----------------------------------------------------------------------------------


def sample_negatives(
    mm: MarkMatrix,
    positives: PairSet,
    universe: Universe,
    n_neg: int,
    rng: np.random.Generator,
    strategy: str = "uniform",
    max_rounds: int = 50,
) -> PairSet:
    """Sample non-interacting pairs from the universe.

    strategy="uniform": uniform over all universe pairs (faithful to Eq. 1 where all
    non-interacting pairs count equally). Returned weights are the inverse sampling
    rate, so the weighted sum is an unbiased estimate of the full non-edge term.

    strategy="distance_matched": negatives drawn to match the genomic-distance
    distribution of the positives (a harder, distance-controlled benchmark).
    """
    if universe.scope != "intra":
        raise NotImplementedError("Negative sampling implemented for intra-chromosomal universes")
    n = mm.n_sites
    pos_keys = np.unique(positives.keys(n))
    order = np.lexsort((mm.pos, mm.chrom))
    pos_sorted_all = mm.pos[order]
    chrom_sorted = mm.chrom[order]

    # per-chromosome windows over the globally sorted order
    lo = np.empty(n, dtype=np.int64)
    hi = np.empty(n, dtype=np.int64)
    for c in mm.chromosomes():
        sel = np.flatnonzero(chrom_sorted == c)
        lo_c, hi_c = _window_bounds(pos_sorted_all[sel], universe.min_dist, universe.max_dist)
        lo[sel] = lo_c + sel[0]
        hi[sel] = hi_c + sel[0]
    sizes = (hi - lo).astype(float)
    total_universe = sizes.sum()
    n_universe_neg = total_universe - len(pos_keys)

    out_u, out_v = [], []
    got = 0
    seen = set()
    for _ in range(max_rounds):
        need = n_neg - got
        if need <= 0:
            break
        k = int(need * 1.3) + 16
        if strategy == "uniform":
            if total_universe <= 0:
                break
            i = rng.choice(n, size=k, p=sizes / total_universe)
            j = lo[i] + (rng.random(k) * sizes[i]).astype(np.int64)
            j = np.minimum(j, hi[i] - 1)
        elif strategy == "distance_matched":
            d_pos = np.abs(mm.pos[positives.u] - mm.pos[positives.v])
            d = rng.choice(d_pos, size=k)
            i = rng.integers(0, n, size=k)
            target = pos_sorted_all[i] + d
            j = np.searchsorted(pos_sorted_all, target)
            j = np.clip(j, 0, n - 1)
            ok_same = chrom_sorted[j] == chrom_sorted[i]
            i, j = i[ok_same], j[ok_same]
        else:
            raise ValueError(f"Unknown strategy {strategy}")
        u = order[i]
        v = order[j]
        u, v = canonical_pairs(u, v)
        ok = universe.contains(mm, u, v)
        u, v = u[ok], v[ok]
        keys = u * np.int64(n) + v
        ok = ~np.isin(keys, pos_keys)
        u, v, keys = u[ok], v[ok], keys[ok]
        # de-duplicate
        _, first = np.unique(keys, return_index=True)
        u, v, keys = u[first], v[first], keys[first]
        new = np.array([kk not in seen for kk in keys.tolist()], dtype=bool) if seen else np.ones(len(keys), bool)
        u, v, keys = u[new], v[new], keys[new]
        seen.update(keys.tolist())
        take = min(need, len(u))
        perm = rng.permutation(len(u))[:take]
        out_u.append(u[perm])
        out_v.append(v[perm])
        got += take
    u = np.concatenate(out_u) if out_u else np.zeros(0, np.int64)
    v = np.concatenate(out_v) if out_v else np.zeros(0, np.int64)
    weight = n_universe_neg / max(len(u), 1) if strategy == "uniform" else 1.0
    return PairSet(u, v, np.zeros(len(u)), np.full(len(u), weight))


def make_labeled_pairs(
    mm: MarkMatrix,
    edges: PairSet,
    universe: Universe,
    neg_ratio: float = 1.0,
    rng: Optional[np.random.Generator] = None,
    strategy: str = "uniform",
    reweight: bool = False,
) -> PairSet:
    """Positives (observed interactions within the universe) + sampled negatives.

    ``reweight=True`` keeps the inverse-sampling weights on negatives so that the
    training loss is an unbiased estimate of the full likelihood; ``False`` gives
    unit weights (a balanced classification setting used for evaluation).
    """
    rng = np.random.default_rng() if rng is None else rng
    inside = universe.contains(mm, edges.u, edges.v)
    pos = edges.subset(np.flatnonzero(inside & (edges.y > 0)))
    n_neg = int(round(neg_ratio * len(pos)))
    neg = sample_negatives(mm, pos, universe, n_neg, rng, strategy)
    if not reweight:
        neg.w = np.ones(len(neg))
    return PairSet.concat([pos, neg])


def remap_pairs(pairs: PairSet, kept_old_idx: np.ndarray, n_old: int) -> PairSet:
    """Re-index pairs after sites were filtered; drops pairs touching removed sites."""
    new_index = np.full(n_old, -1, dtype=np.int64)
    new_index[kept_old_idx] = np.arange(len(kept_old_idx))
    u = new_index[pairs.u]
    v = new_index[pairs.v]
    ok = (u >= 0) & (v >= 0)
    return PairSet(u[ok], v[ok], pairs.y[ok], pairs.w[ok])
