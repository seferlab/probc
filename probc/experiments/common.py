"""Shared infrastructure for the experiment modules (one module per paper section)."""
from __future__ import annotations

import copy
import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional, Sequence

import numpy as np
import yaml

from ..data import PairSet, Universe, make_labeled_pairs, remap_pairs
from ..model import ProbC
from ..pipeline import Dataset, build_dataset, dataset_marks
from .. import synthetic as syn

log = logging.getLogger("probc")

DEFAULTS = {
    "seed": 0,
    "datasets_config": "configs/datasets.yaml",
    "universe": {"scope": "intra", "min_dist": 1, "max_dist": 2_000_000},
    "negatives": {"ratio": 1.0, "strategy": "uniform"},
    "cv": {"outer_folds": 5, "inner_folds": 4, "criterion": "auc"},
    "grid": {"lambda_nuc": [1e-4, 1e-3, 1e-2], "lambda_l1": [1e-4, 1e-3, 1e-2]},
    "model": {"likelihood": "bernoulli", "loss_scale": "mean", "max_iter": 300, "inner_solver": "lbfgsb"},
    "genome_wide": {"lambda_nuc": 1e-3, "lambda_l1": 1e-3, "select_lambda": True},
    "chromosomes": None,
    "primary": "hESC",
}


def deep_update(base: dict, upd: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in (upd or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_update(out[k], v)
        else:
            out[k] = v
    return out


@dataclass
class Context:
    cfg: dict
    synthetic: bool
    outdir: Path
    quick: bool = False
    _cache: dict = field(default_factory=dict)

    @property
    def seed(self) -> int:
        return int(self.cfg.get("seed", 0))

    def rng(self, offset: int = 0) -> np.random.Generator:
        return np.random.default_rng(self.seed + offset)

    def section(self, name: str) -> dict:
        return self.cfg.get(name, {}) or {}

    # -- model / CV settings ----------------------------------------------------
    def base_model(self, **overrides) -> ProbC:
        p = dict(self.cfg["model"])
        p.update(overrides)
        return ProbC(**p)

    def grid(self) -> dict:
        g = self.cfg["grid"]
        if self.quick:
            return {k: [v[len(v) // 2]] for k, v in g.items()}
        return g

    def universe(self) -> Universe:
        u = self.cfg["universe"]
        return Universe(u.get("scope", "intra"), int(u.get("min_dist", 1)),
                        None if u.get("max_dist") is None else int(u["max_dist"]))

    # -- datasets --------------------------------------------------------------
    def dataset(self, name: str, mode: str = "unbinned", resolution: Optional[int] = None,
                include: Sequence[str] = ("histone", "tf", "dnase")) -> Dataset:
        key = (name, mode, resolution, tuple(include))
        if key in self._cache:
            return self._cache[key]
        if self.synthetic:
            ds = synthetic_dataset(self, name)
            ds = restrict_kinds(ds, self.mark_kinds(name), include)
            if mode == "binned":
                ds = bin_dataset(ds, resolution)
        else:
            dcfg = load_yaml(self.cfg["datasets_config"])
            dcfg["universe"] = self.cfg["universe"]
            chroms = self.cfg.get("chromosomes") or dcfg["datasets"][name].get("chromosomes")
            ds = build_dataset(dcfg, name, chroms, mode=mode, resolution=resolution, include=include)
            ds.universe = self.universe() if mode == "unbinned" else Universe(
                "intra", int(resolution), self.universe().max_dist)
        self._cache[key] = ds
        return ds

    def mark_kinds(self, name: str) -> Dict[str, str]:
        if self.synthetic:
            mouse = name.lower().startswith("m")
            his = syn.MOUSE_HISTONES if mouse else syn.HUMAN_HISTONES
            tf = syn.MOUSE_TFBS if mouse else syn.HUMAN_TFBS
            out = {m: "histone" for m in his}
            out.update({m: ("dnase" if m == "DNase" else "tf") for m in tf})
            return out
        dcfg = load_yaml(self.cfg["datasets_config"])
        return dataset_marks(dcfg["datasets"][name])

    def chromosomes(self, ds: Dataset) -> list:
        chroms = ds.marks.chromosomes()
        if self.quick:
            return chroms[:2]
        return chroms

    def labeled_pairs(self, ds: Dataset, offset: int = 0, reweight: bool = True) -> PairSet:
        n = self.cfg["negatives"]
        return make_labeled_pairs(ds.marks, ds.edges, ds.universe, neg_ratio=float(n.get("ratio", 1.0)),
                                  rng=self.rng(1000 + offset), strategy=n.get("strategy", "uniform"),
                                  reweight=reweight)

    # -- outputs ------------------------------------------------------------------
    def path(self, name: str) -> Path:
        self.outdir.mkdir(parents=True, exist_ok=True)
        return self.outdir / name

    def save_json(self, obj, name: str) -> Path:
        p = self.path(name)
        with open(p, "w") as fh:
            json.dump(obj, fh, indent=2, default=_json_default)
        return p


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


def load_yaml(path) -> dict:
    with open(path) as fh:
        return yaml.safe_load(fh)


def make_context(config: Optional[str], synthetic: bool, outdir: str, quick: bool = False,
                 overrides: Optional[dict] = None) -> Context:
    cfg = copy.deepcopy(DEFAULTS)
    if config:
        cfg = deep_update(cfg, load_yaml(config))
    if synthetic:
        cfg = deep_update(cfg, cfg.get("synthetic_overrides", {}))
    cfg = deep_update(cfg, overrides or {})
    return Context(cfg=cfg, synthetic=synthetic, outdir=Path(outdir), quick=quick)


# ----------------------------------------------------------------------------------
# dataset helpers
# ----------------------------------------------------------------------------------


def restrict_marks(ds: Dataset, marks: Sequence[str]) -> Dataset:
    """Keep only some marks; sites left without any mark are dropped."""
    mm = ds.marks.select_marks([m for m in marks if m in ds.marks.marks])
    n_before = mm.n_sites
    mm2, kept = mm.drop_empty_sites()
    edges = remap_pairs(ds.edges, kept, n_before)
    return Dataset(ds.name, mm2, edges, ds.universe, dict(ds.meta, marks=list(mm2.marks)))


def chrom_dataset(ds: Dataset, chroms) -> Dataset:
    """Restrict a dataset to some chromosomes (edges are re-indexed)."""
    if isinstance(chroms, str):
        chroms = [chroms]
    idx = np.flatnonzero(np.isin(ds.marks.chrom, list(chroms)))
    mm = ds.marks.subset_sites(idx)
    edges = remap_pairs(ds.edges, idx, ds.marks.n_sites)
    return Dataset(ds.name, mm, edges, ds.universe, ds.meta)


def restrict_kinds(ds: Dataset, kinds: Dict[str, str], include: Sequence[str]) -> Dataset:
    keep = [m for m in ds.marks.marks if kinds.get(m, "tf") in include]
    if len(keep) == ds.marks.n_marks:
        return ds
    return restrict_marks(ds, keep)


def bin_dataset(ds: Dataset, resolution: int) -> Dataset:
    from ..binning import bin_ids, bin_pairs

    s2b, binned = bin_ids(ds.marks, resolution)
    edges = bin_pairs(ds.edges, s2b)
    n_before = binned.n_sites
    binned, kept = binned.drop_empty_sites()
    edges = remap_pairs(edges, kept, n_before)
    uni = Universe("intra", int(resolution), ds.universe.max_dist)
    return Dataset(ds.name, binned, edges, uni, dict(ds.meta, mode="binned", resolution=resolution))


# ----------------------------------------------------------------------------------
# synthetic stand-ins for the paper's datasets
# ----------------------------------------------------------------------------------

SYN_ACTIVE_HUMAN = ["H3K27ac", "H3K9me3", "H3K4me3", "H3K4me1", "CTCF"]


def _human_theta(rng: np.random.Generator, marks: list, scale: float) -> np.ndarray:
    T = syn.planted_theta(marks, SYN_ACTIVE_HUMAN, rng, scale=scale, n_blocks=2)
    # weaker secondary signal (cohesin / polycomb-like pairs), as discussed in the paper
    for a, b, w in [("RAD21", "SMC3", 0.5), ("CTCF", "MAFK", 0.3), ("H3K27me3", "H3K27me3", 0.4),
                    ("H3K9ac", "H3K4me2", 0.3), ("H3K27me3", "H3K36me3", 0.25)]:
        if a in marks and b in marks:
            i, j = marks.index(a), marks.index(b)
            T[i, j] = T[j, i] = scale * w
    return T


def synthetic_dataset(ctx: Context, name: str) -> Dataset:
    """Deterministic synthetic analogues of hESC, IMR90, IMR90 (HindIII), mESC."""
    key = ("_synthetic", name)
    if key in ctx._cache:
        return ctx._cache[key]
    sc = ctx.section("synthetic")
    n_chroms = int(sc.get("n_chroms", 4 if not ctx.quick else 2))
    sites = int(sc.get("sites_per_chrom", 1500 if not ctx.quick else 800))
    gap = int(sc.get("mean_gap", 1000))
    scale = float(sc.get("theta_scale", 0.08))
    decay = sc.get("distance_decay", None)
    universe = Universe("intra", 1, gap * int(sc.get("window_sites", 150)))
    human = syn.HUMAN_HISTONES + syn.HUMAN_TFBS
    mouse = syn.MOUSE_HISTONES + syn.MOUSE_TFBS
    base_rng = np.random.default_rng(ctx.seed + 7)
    theta_h = _human_theta(base_rng, human, scale)

    def emission(seed, marks):
        r = np.random.default_rng(seed)
        E = np.full((4, len(marks)), 0.05)
        for t in range(4):
            hot = r.choice(len(marks), size=max(2, len(marks) // 4), replace=False)
            E[t, hot] = 0.35
        return E

    if name in ("hESC", "IMR90", "IMR90_MboI"):
        theta = theta_h if name == "hESC" else syn.perturb_theta(theta_h, np.random.default_rng(ctx.seed + 11), 0.15)
        g = syn.simulate_genome(marks=human, n_chroms=n_chroms, sites_per_chrom=sites, mean_gap=gap, theta=theta,
                                universe=universe, distance_decay=decay, emission=emission(ctx.seed + (1 if name == "hESC" else 2), human),
                                seed=ctx.seed + (101 if name == "hESC" else 202))
    elif name == "IMR90_HindIII":
        base = synthetic_dataset(ctx, "IMR90")
        g0 = syn.SyntheticGenome(base.marks, base.edges, base.meta["theta_true"], {}, {}, base.universe)
        g = syn.remap_sites(g0, mean_gap=gap * 3, seed=ctx.seed + 303)
    elif name in ("mESC", "mESC_microC"):
        theta_m = np.zeros((len(mouse), len(mouse)))
        for i, a in enumerate(mouse):
            for j, b in enumerate(mouse):
                if a in human and b in human:
                    theta_m[i, j] = theta_h[human.index(a), human.index(b)]
        theta_m = syn.perturb_theta(theta_m, np.random.default_rng(ctx.seed + 13), 0.25)
        g = syn.simulate_genome(marks=mouse, n_chroms=n_chroms, sites_per_chrom=sites, mean_gap=gap // 2,
                                theta=theta_m, universe=Universe("intra", 1, (gap // 2) * int(sc.get("window_sites", 150))),
                                distance_decay=decay, emission=emission(ctx.seed + 3, mouse), seed=ctx.seed + 404)
    else:
        raise KeyError(f"No synthetic analogue for dataset {name!r}")
    # planted domains in genomic coordinates (site indices change when empty sites are dropped)
    domains_bp = {}
    for c, dom in (g.domains or {}).items():
        pos = g.marks.pos[g.marks.chrom == c]
        domains_bp[c] = [(int(pos[s]), int(pos[e - 1]) + 1) for s, e in dom]
    n_before = g.marks.n_sites
    mm, kept = g.marks.drop_empty_sites()
    edges = remap_pairs(g.edges, kept, n_before)
    ds = Dataset(name, mm, edges, g.universe,
                 dict(theta_true=g.theta_true, theta_marks=list(g.marks.marks), domains_bp=domains_bp,
                      synthetic=True))
    ctx._cache[key] = ds
    return ds


class Timer:
    def __init__(self, label: str):
        self.label = label

    def __enter__(self):
        self.t = time.time()
        return self

    def __exit__(self, *a):
        log.info("%s done in %.1fs", self.label, time.time() - self.t)
