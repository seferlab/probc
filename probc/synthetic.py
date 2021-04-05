"""Synthetic genomes drawn from the ProbC generative model.

Used for unit tests, for checking parameter recovery, and for smoke-testing every
experiment script without downloading data (``--synthetic``).

The simulator produces restriction-site-like positions, chromatin domains with
type-specific mark emission (so that TAD-/compartment-like structure emerges),
CTCF-like marks enriched at domain boundaries, a planted sparse/low-rank mark
interaction matrix, and interactions sampled from the model (optionally multiplied
by a genomic-distance decay to mimic real Hi-C).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np

from .data import MarkMatrix, PairSet, Universe, _window_bounds

HUMAN_HISTONES = [
    "H3K4me1", "H3K4me3", "H3K9me3", "H3K27ac", "H3K79me2", "H3K36me3", "H4K20me1", "H3K27me3",
    "H3K56ac", "H3K23ac", "H2AK5ac", "H2A.Z", "H3K9ac", "H3K4me2", "H4K8ac", "H3K18ac",
]
HUMAN_TFBS = [
    "CTCF", "SMC3", "MAFK", "CHD1", "POLR2A", "DNase", "RAD21", "CEBPB", "MAZ", "FOS", "USF2", "RCOR1",
    "RFX5", "ELK1", "MXI1", "NFE2L2",
]
MOUSE_HISTONES = ["H3K4me3", "H3K27ac", "H3K36me3", "H3K4me1", "H3K9me3", "H3K27me3", "H3K9ac"]
MOUSE_TFBS = ["CTCF", "POLR2A", "EP300", "MAFK", "CHD2", "HCFC1", "ZC3H11A", "ZNF384"]

# marks the paper finds most predictive; used to plant the true signal
DEFAULT_ACTIVE = ["H3K27ac", "H3K9me3", "H3K4me3", "H3K4me1", "CTCF"]


@dataclass
class SyntheticGenome:
    marks: MarkMatrix
    edges: PairSet            # observed interactions (y = 1 or counts)
    theta_true: np.ndarray     # planted Z (bernoulli) or X (poisson)
    domains: dict              # chrom -> array of (start_idx, end_idx) domain boundaries in site indices
    domain_type: dict          # chrom -> array of domain types
    universe: Universe


def planted_theta(
    marks: Sequence[str],
    active: Sequence[str] = DEFAULT_ACTIVE,
    rng: Optional[np.random.Generator] = None,
    scale: float = 1.0,
    n_blocks: int = 2,
    offdiag_density: float = 0.4,
) -> np.ndarray:
    """Sparse, approximately block-diagonal non-negative interaction matrix."""
    rng = np.random.default_rng(0) if rng is None else rng
    m = len(marks)
    T = np.zeros((m, m))
    act = [marks.index(a) for a in active if a in marks]
    if not act:
        act = list(range(min(5, m)))
    blocks = np.array_split(np.array(act), n_blocks)
    for blk in blocks:
        for i in blk:
            for j in blk:
                if i <= j and (i == j or rng.random() < 0.8):
                    T[i, j] = T[j, i] = scale * rng.uniform(0.5, 1.5)
    for i in act:
        for j in act:
            if i < j and T[i, j] == 0 and rng.random() < offdiag_density * 0.3:
                T[i, j] = T[j, i] = scale * rng.uniform(0.05, 0.3)
    return T


def simulate_genome(
    marks: Sequence[str] = tuple(HUMAN_HISTONES[:8] + ["CTCF", "RAD21", "POLR2A", "DNase"]),
    n_chroms: int = 3,
    sites_per_chrom: int = 2000,
    mean_gap: int = 1000,
    domain_size: tuple = (40, 160),
    n_domain_types: int = 4,
    mark_rate: float = 0.08,
    enrich: float = 6.0,
    boundary_marks: Sequence[str] = ("CTCF", "RAD21"),
    boundary_rate: float = 0.7,
    theta: Optional[np.ndarray] = None,
    theta_scale: float = 0.08,
    active: Sequence[str] = DEFAULT_ACTIVE,
    likelihood: str = "bernoulli",
    universe: Optional[Universe] = None,
    distance_decay: Optional[float] = None,
    poisson_depth: float = 1.0,
    emission: Optional[np.ndarray] = None,
    seed: int = 0,
    chrom_prefix: str = "chr",
) -> SyntheticGenome:
    """Simulate a genome, its marks and its interactions from the ProbC model.

    ``distance_decay=beta`` multiplies interaction rates by ``(d / d0)^-beta`` which
    makes the data *misspecified* for ProbC (as real Hi-C is); ``None`` samples
    exactly from the model (used for recovery tests).
    """
    rng = np.random.default_rng(seed)
    marks = list(marks)
    m = len(marks)
    if theta is None:
        theta = planted_theta(marks, active, rng, scale=theta_scale)
    universe = universe or Universe("intra", min_dist=1, max_dist=mean_gap * 150)
    if emission is None:
        emission = np.full((n_domain_types, m), mark_rate)
        for t in range(n_domain_types):
            hot = rng.choice(m, size=max(1, m // n_domain_types), replace=False)
            emission[t, hot] = min(0.9, mark_rate * enrich)
    bidx = [marks.index(b) for b in boundary_marks if b in marks]

    C_all, chrom_all, pos_all = [], [], []
    domains, dtypes = {}, {}
    for c in range(n_chroms):
        name = f"{chrom_prefix}{c + 1}"
        n = sites_per_chrom
        pos = np.cumsum(rng.integers(max(1, mean_gap // 4), mean_gap * 2, size=n)).astype(np.int64)
        # domains
        bounds = [0]
        while bounds[-1] < n:
            bounds.append(min(n, bounds[-1] + int(rng.integers(*domain_size))))
        bounds = np.array(bounds)
        types = rng.integers(0, n_domain_types, size=len(bounds) - 1)
        # alternate types between neighbours to create clear boundaries
        for i in range(1, len(types)):
            if types[i] == types[i - 1]:
                types[i] = (types[i] + 1) % n_domain_types
        C = np.zeros((n, m))
        for d in range(len(bounds) - 1):
            s, e = bounds[d], bounds[d + 1]
            C[s:e] = rng.random((e - s, m)) < emission[types[d]][None, :]
        for b in bounds[1:-1]:
            for j in bidx:
                for off in (-1, 0):
                    if 0 <= b + off < n and rng.random() < boundary_rate:
                        C[b + off, j] = 1
        C_all.append(C)
        chrom_all.append(np.full(n, name))
        pos_all.append(pos)
        domains[name] = np.stack([bounds[:-1], bounds[1:]], 1)
        dtypes[name] = types
    mm = MarkMatrix(np.vstack(C_all), marks, np.concatenate(chrom_all), np.concatenate(pos_all))

    # sample interactions within the universe
    us, vs, ys = [], [], []
    offset = 0
    A = mm.C @ theta
    for c in mm.chromosomes():
        idx = np.flatnonzero(mm.chrom == c)
        pos = mm.pos[idx]
        lo, hi = _window_bounds(pos, universe.min_dist, universe.max_dist)
        for i in range(len(idx)):
            if hi[i] <= lo[i]:
                continue
            j = np.arange(lo[i], hi[i])
            s = A[idx[i]] @ mm.C[idx[j]].T
            if distance_decay is not None:
                d = (pos[j] - pos[i]) / float(mean_gap)
                s = s * np.power(np.maximum(d, 1.0), -distance_decay) * 3.0
            if likelihood == "bernoulli":
                p = -np.expm1(-s)
                hit = rng.random(len(j)) < p
                us.append(np.full(hit.sum(), idx[i]))
                vs.append(idx[j[hit]])
                ys.append(np.ones(hit.sum()))
            else:
                k = rng.poisson(s * poisson_depth)
                nz = k > 0
                us.append(np.full(nz.sum(), idx[i]))
                vs.append(idx[j[nz]])
                ys.append(k[nz].astype(float))
        offset += len(idx)
    edges = PairSet(np.concatenate(us), np.concatenate(vs), np.concatenate(ys))
    return SyntheticGenome(mm, edges, theta, domains, dtypes, universe)


def perturb_theta(theta: np.ndarray, rng: np.random.Generator, noise: float = 0.15) -> np.ndarray:
    """A related interaction matrix (e.g. another cell type of the same species)."""
    T = theta * np.exp(rng.normal(0, noise, theta.shape))
    T = 0.5 * (T + T.T)
    return T


def remap_sites(genome: SyntheticGenome, mean_gap: int, seed: int) -> SyntheticGenome:
    """Same chromatin, different 'restriction enzyme': new site positions, marks
    transferred by nearest-site lookup, interactions resampled. Used for the
    enzyme-replicate stability experiment (Fig. 13a)."""
    rng = np.random.default_rng(seed)
    mm = genome.marks
    Cs, chroms, poss = [], [], []
    for c in mm.chromosomes():
        idx = np.flatnonzero(mm.chrom == c)
        pos = mm.pos[idx]
        n_new = int((pos[-1] - pos[0]) / mean_gap)
        new_pos = np.sort(rng.integers(pos[0], pos[-1], size=n_new))
        nearest = np.clip(np.searchsorted(pos, new_pos), 0, len(pos) - 1)
        Cs.append(mm.C[idx[nearest]])
        chroms.append(np.full(n_new, c))
        poss.append(new_pos)
    mm2 = MarkMatrix(np.vstack(Cs), mm.marks, np.concatenate(chroms), np.concatenate(poss))
    # resample edges from the same theta
    us, vs = [], []
    A = mm2.C @ genome.theta_true
    for c in mm2.chromosomes():
        idx = np.flatnonzero(mm2.chrom == c)
        pos = mm2.pos[idx]
        lo, hi = _window_bounds(pos, genome.universe.min_dist, genome.universe.max_dist)
        for i in range(len(idx)):
            if hi[i] <= lo[i]:
                continue
            j = np.arange(lo[i], hi[i])
            p = -np.expm1(-(A[idx[i]] @ mm2.C[idx[j]].T))
            hit = rng.random(len(j)) < p
            us.append(np.full(hit.sum(), idx[i]))
            vs.append(idx[j[hit]])
    edges = PairSet(np.concatenate(us), np.concatenate(vs), np.ones(sum(len(x) for x in us)))
    return SyntheticGenome(mm2, edges, genome.theta_true, genome.domains, genome.domain_type, genome.universe)
