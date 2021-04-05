"""Genome-wide interaction probabilities between marks (Figs. 2-3, Table 2).

Outputs
-------
fig2_histone_interactions_<ds>.{png,csv}   clustered heat map of x_mn (histones)
fig3_tfbs_interactions_<ds>.{png,csv}      clustered heat map of x_mn (TFBSs)
table2_es_vs_imr90.csv                     top-5 mark pairs by |x_ES - x_IMR90|
interaction_correlation.json               correlation between the two cell types
models/*.npz                               fitted models
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .. import plotting
from ..cv import grid_search
from ..data import build_full_pairset
from ..metrics import pearson_matrices, spearman_matrices
from .common import Context, Timer, log


def fit_genome_wide(ctx: Context, ds, offset: int = 0):
    """Select (lambda_nuc, lambda_l1) by inner CV (optional) and refit on all pairs."""
    gw = ctx.cfg["genome_wide"]
    params = {"lambda_nuc": float(gw["lambda_nuc"]), "lambda_l1": float(gw["lambda_l1"])}
    if gw.get("select_lambda", True) and not ctx.quick:
        pairs = ctx.labeled_pairs(ds, offset)
        gs = grid_search(ctx.base_model(), ds.marks, pairs, ctx.grid(), ctx.cfg["cv"]["inner_folds"],
                         ctx.cfg["cv"]["criterion"], ctx.rng(offset))
        params = gs.best_params
        log.info("%s: selected %s", ds.name, params)
    full = build_full_pairset(ds.marks, ds.edges, ds.universe)
    model = ctx.base_model(**params).fit(ds.marks, full)
    return model, params


def _save_matrix(X, marks, path):
    pd.DataFrame(X, index=marks, columns=marks).to_csv(path)


def run(ctx: Context) -> dict:
    sc = ctx.section("heatmaps")
    primary = ctx.cfg["primary"]
    other = sc.get("compare_with", "IMR90")
    out = {}
    models_dir = ctx.path("models")
    models_dir.mkdir(exist_ok=True)

    with Timer("Fig.2 histone interactions"):
        fits = {}
        for i, name in enumerate([primary, other]):
            ds = ctx.dataset(name, include=("histone",))
            model, params = fit_genome_wide(ctx, ds, offset=i)
            model.save(models_dir / f"{name}_histone.npz")
            fits[name] = model
            _save_matrix(model.X_, model.marks_, ctx.path(f"fig2_histone_interactions_{name}.csv"))
            plotting.clustered_heatmap(model.X_, model.marks_, f"Fig. 2: ProbC histone interactions ({name})",
                                       ctx.path(f"fig2_histone_interactions_{name}.png"))
            model.interaction_table(tol=1e-6).to_csv(ctx.path(f"interactions_histone_{name}.csv"), index=False)
            out[f"params_{name}"] = params

    with Timer("Table 2 / correlation"):
        a, b = fits[primary], fits[other]
        shared = [m for m in a.marks_ if m in b.marks_]
        ia = [a.marks_.index(m) for m in shared]
        ib = [b.marks_.index(m) for m in shared]
        Xa, Xb = a.X_[np.ix_(ia, ia)], b.X_[np.ix_(ib, ib)]
        out["pearson"] = pearson_matrices(Xa, Xb)
        out["spearman"] = spearman_matrices(Xa, Xb)
        rows = []
        for i in range(len(shared)):
            for j in range(i, len(shared)):
                rows.append((f"{shared[i]} - {shared[j]}", Xa[i, j], Xb[i, j], abs(Xa[i, j] - Xb[i, j])))
        t2 = pd.DataFrame(rows, columns=["interaction", primary, other, "abs_diff"]).sort_values("abs_diff", ascending=False)
        t2.head(int(sc.get("top_k", 5))).to_csv(ctx.path("table2_es_vs_imr90.csv"), index=False)
        out["table2"] = t2.head(int(sc.get("top_k", 5))).to_dict(orient="records")
        log.info("correlation %s vs %s: pearson=%.3f spearman=%.3f", primary, other, out["pearson"], out["spearman"])

    with Timer("Fig.3 TFBS interactions"):
        ds = ctx.dataset(primary, include=("tf", "dnase"))
        model, params = fit_genome_wide(ctx, ds, offset=7)
        model.save(models_dir / f"{primary}_tf.npz")
        _save_matrix(model.X_, model.marks_, ctx.path(f"fig3_tfbs_interactions_{primary}.csv"))
        plotting.clustered_heatmap(model.X_, model.marks_, f"Fig. 3: ProbC TFBS interactions ({primary})",
                                   ctx.path(f"fig3_tfbs_interactions_{primary}.png"))
        model.interaction_table(tol=1e-6).to_csv(ctx.path(f"interactions_tf_{primary}.csv"), index=False)
        out["params_tf"] = params
        out["top_tf_pairs"] = model.interaction_table(tol=1e-6).head(10).to_dict(orient="records")

    if ctx.synthetic:
        ds = ctx.dataset(primary)
        T = ds.meta["theta_true"]
        tm = ds.meta["theta_marks"]
        ia = [tm.index(m) for m in fits[primary].marks_]
        out["synthetic_recovery_corr"] = float(np.corrcoef(fits[primary].theta_.ravel(), T[np.ix_(ia, ia)].ravel())[0, 1])
    ctx.save_json(out, "heatmaps.json")
    return out
