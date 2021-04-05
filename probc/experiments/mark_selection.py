"""Section "CTCF and a small subset of histone modifications are critical for
predicting Hi-C interactions" and "PROBC finds a small subset of transcription factor
binding sites predictive of Hi-C interactions".

Outputs
-------
fig1_mark_frequency.{png,csv}   probability of each histone mark appearing in the
                                per-chromosome ProbC solution (Fig. 1)
diagonal_only_ranking.csv       marks ranked by z_mm when interactions *between*
                                marks are forbidden (support="diagonal")
coverage_curve.{png,csv}        held-out AUC vs number of top-ranked marks, histone
                                modifications vs TFBSs ("~12 TFBSs give the same
                                performance as 4 histone modifications")
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .. import plotting
from ..cv import kfold_auc
from ..data import build_full_pairset
from .common import Context, Timer, chrom_dataset, log


def _active(theta: np.ndarray, marks, rel_tol: float) -> list:
    if theta.max() <= 0:
        return []
    row = np.abs(theta).max(1)
    return [m for m, r in zip(marks, row) if r > rel_tol * theta.max()]


def run(ctx: Context) -> dict:
    sc = ctx.section("mark_selection")
    lam_nuc = float(sc.get("lambda_nuc", ctx.cfg["genome_wide"]["lambda_nuc"]))
    lam_l1 = float(sc.get("lambda_l1", 1e-2))
    rel_tol = float(sc.get("active_rel_tol", 0.1))
    support = sc.get("support", "full")
    name = ctx.cfg["primary"]
    results = {}

    # ---- Fig. 1: per-chromosome solutions, histone modifications only ------------
    with Timer("Fig.1 per-chromosome mark frequency"):
        ds_h = ctx.dataset(name, include=("histone",))
        chroms = ctx.chromosomes(ds_h)
        counts = {m: 0 for m in ds_h.marks.marks}
        per_chrom = {}
        for c in chroms:
            dc = chrom_dataset(ds_h, c)
            full = build_full_pairset(dc.marks, dc.edges, dc.universe)
            model = ctx.base_model(lambda_nuc=lam_nuc, lambda_l1=lam_l1, support=support).fit(dc.marks, full)
            act = _active(model.theta_, model.marks_, rel_tol)
            per_chrom[c] = act
            for m in act:
                counts[m] += 1
            log.info("%s active: %s", c, act)
        freq = {m: counts[m] / len(chroms) for m in ds_h.marks.marks}
        pd.DataFrame({"mark": list(freq), "frequency": list(freq.values())}).to_csv(ctx.path("fig1_mark_frequency.csv"), index=False)
        plotting.bar(dict(sorted(freq.items(), key=lambda kv: -kv[1])),
                     "Fig. 1: histone marks in per-chromosome ProbC solutions", "fraction of chromosomes",
                     ctx.path("fig1_mark_frequency.png"), ylim=(0, 1.05))
        results["fig1"] = {"frequency": freq, "per_chromosome": per_chrom}

    # ---- interactions between marks forbidden (diagonal support), all marks -------
    with Timer("diagonal-only ranking"):
        ds = ctx.dataset(name)
        full = build_full_pairset(ds.marks, ds.edges, ds.universe)
        model = ctx.base_model(lambda_nuc=0.0, lambda_l1=lam_l1, support="diagonal").fit(ds.marks, full)
        diag = np.diag(model.theta_)
        kinds = ctx.mark_kinds(name)
        rank = pd.DataFrame({"mark": model.marks_, "z_mm": diag, "x_mm": -np.expm1(-diag),
                             "kind": [kinds.get(m, "tf") for m in model.marks_]}).sort_values("z_mm", ascending=False)
        rank.to_csv(ctx.path("diagonal_only_ranking.csv"), index=False)
        results["diagonal_top"] = rank.head(int(sc.get("top_k", 5)))["mark"].tolist()
        log.info("top marks with diagonal-only support: %s", results["diagonal_top"])

    # ---- coverage curve: AUC vs number of marks, histones vs TFBSs ----------------
    with Timer("coverage curve"):
        pairs = ctx.labeled_pairs(ds)
        kmax = int(sc.get("coverage_max_marks", 16 if not ctx.quick else 6))
        curves = {}
        for kind_set, label in ((("histone",), "histone modifications"), (("tf", "dnase"), "TFBS (+DNase)")):
            ranked = [m for m in rank["mark"] if kinds.get(m, "tf") in kind_set]
            aucs = []
            for k in range(1, min(kmax, len(ranked)) + 1):
                mm_k = ds.marks.select_marks(ranked[:k])
                aucs.append(kfold_auc(ctx.base_model(lambda_nuc=lam_nuc, lambda_l1=ctx.cfg["genome_wide"]["lambda_l1"]),
                                      mm_k, pairs, n_folds=ctx.cfg["cv"]["outer_folds"], seed=ctx.seed))
            curves[label] = aucs
        L = max(len(v) for v in curves.values())
        df = pd.DataFrame({"n_marks": np.arange(1, L + 1)})
        for k, v in curves.items():
            df[k] = v + [np.nan] * (L - len(v))
        df.to_csv(ctx.path("coverage_curve.csv"), index=False)
        plotting.line_plot(df["n_marks"], {k: df[k] for k in curves}, "number of top-ranked marks", "held-out ROC AUC",
                           "Prediction performance vs number of marks", ctx.path("coverage_curve.png"))
        results["coverage"] = curves
    ctx.save_json(results, "mark_selection.json")
    return results
