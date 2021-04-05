"""Section "PROBC is stable to enzyme replicates, robust across resolution
parameters" (Fig. 13).

(a) Spearman correlation between histone-pair interaction probabilities inferred
    from two restriction-enzyme replicates (IMR90 MboI vs HindIII), per chromosome.
(b) Spearman correlation between interaction probabilities inferred from binned
    data at different resolutions (5 kb, 50 kb, 250 kb), per chromosome.
"""
from __future__ import annotations

import itertools

import numpy as np
import pandas as pd

from .. import plotting
from ..data import build_full_pairset
from ..metrics import pearson_matrices, spearman_matrices
from .common import Context, Timer, chrom_dataset, log


def _fit_chrom(ctx: Context, ds, chrom, **kw):
    dc = chrom_dataset(ds, chrom)
    full = build_full_pairset(dc.marks, dc.edges, dc.universe)
    gw = ctx.cfg["genome_wide"]
    params = dict(lambda_nuc=gw["lambda_nuc"], lambda_l1=gw["lambda_l1"])
    params.update(kw)
    return ctx.base_model(**params).fit(dc.marks, full)


def _aligned_X(a, b):
    shared = [m for m in a.marks_ if m in b.marks_]
    ia = [a.marks_.index(m) for m in shared]
    ib = [b.marks_.index(m) for m in shared]
    return a.X_[np.ix_(ia, ia)], b.X_[np.ix_(ib, ib)]


def run(ctx: Context) -> dict:
    sc = ctx.section("stability")
    out = {}
    with Timer("Fig.13a enzyme replicates"):
        r1, r2 = sc.get("replicates", ["IMR90", "IMR90_HindIII"])
        A = ctx.dataset(r1, include=("histone",))
        B = ctx.dataset(r2, include=("histone",))
        rows, pear = {}, {}
        for c in ctx.chromosomes(A):
            if c not in B.marks.chromosomes():
                continue
            Xa, Xb = _aligned_X(_fit_chrom(ctx, A, c), _fit_chrom(ctx, B, c))
            rows[c] = spearman_matrices(Xa, Xb)
            pear[c] = pearson_matrices(Xa, Xb)
        pd.DataFrame({"chrom": list(rows), "spearman": list(rows.values()), "pearson": list(pear.values())}).to_csv(
            ctx.path("fig13a_enzyme_replicates.csv"), index=False)
        out["enzyme_pearson"] = pear
        plotting.bar(rows, f"Fig. 13a: {r1} vs {r2}", "Spearman correlation", ctx.path("fig13a_enzyme_replicates.png"), ylim=(0, 1))
        out["enzyme"] = rows
        log.info("enzyme replicate Spearman: %s", {k: round(v, 3) for k, v in rows.items()})

    with Timer("Fig.13b resolutions"):
        name = sc.get("dataset", "IMR90")
        resolutions = [int(r) for r in sc.get("resolutions", [5_000, 50_000, 250_000])]
        likelihood = sc.get("binned_likelihood", "poisson")
        fits = {}
        for r in resolutions:
            ds = ctx.dataset(name, mode="binned", resolution=r, include=("histone",))
            for c in ctx.chromosomes(ds):
                fits[(r, c)] = _fit_chrom(ctx, ds, c, likelihood=likelihood)
        rows = []
        chroms = sorted({c for _, c in fits}, key=lambda s: (len(s), s))
        for r1_, r2_ in itertools.combinations(resolutions, 2):
            for c in chroms:
                if (r1_, c) in fits and (r2_, c) in fits:
                    Xa, Xb = _aligned_X(fits[(r1_, c)], fits[(r2_, c)])
                    rows.append({"pair": f"{r1_ // 1000}kb-{r2_ // 1000}kb", "chrom": c,
                                 "spearman": spearman_matrices(Xa, Xb), "pearson": pearson_matrices(Xa, Xb)})
        df = pd.DataFrame(rows)
        df.to_csv(ctx.path("fig13b_resolutions.csv"), index=False)
        piv = df.pivot(index="chrom", columns="pair", values="spearman").reindex(chroms)
        plotting.grouped_bar({p: piv[p].tolist() for p in piv.columns}, list(piv.index),
                             "Fig. 13b: stability across resolutions", "Spearman correlation",
                             ctx.path("fig13b_resolutions.png"), ylim=(-0.2, 1))
        out["resolution"] = df.groupby("pair")["spearman"].mean().to_dict()
        out["resolution_pearson"] = df.groupby("pair")["pearson"].mean().to_dict()
        log.info("resolution Spearman: %s", out["resolution"])
    ctx.save_json(out, "stability.json")
    return out
