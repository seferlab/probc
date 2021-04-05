"""Section "PROBC is robust to changes in lambda_1 and lambda_2" (Fig. 12).

Note on naming: Eq. (3) of the paper attaches lambda_1 to the nuclear norm and
lambda_2 to the l1 norm, while the Results text calls lambda_1 the sparsity and
lambda_2 the low-rank parameter. We therefore sweep the two penalties under explicit
names (``lambda_l1`` = sparsity, ``lambda_nuc`` = low rank).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .. import plotting
from ..cv import kfold_auc
from ..data import build_full_pairset
from .common import Context, Timer, log


def run(ctx: Context) -> dict:
    sc = ctx.section("robustness")
    name = ctx.cfg["primary"]
    ds = ctx.dataset(name)
    pairs = ctx.labeled_pairs(ds)
    full = build_full_pairset(ds.marks, ds.edges, ds.universe)
    gw = ctx.cfg["genome_wide"]
    l1_grid = [float(x) for x in sc.get("lambda_l1", [0, 1e-4, 1e-3, 1e-2, 3e-2, 1e-1])]
    nuc_grid = [float(x) for x in sc.get("lambda_nuc", [0, 1e-4, 1e-3, 1e-2, 3e-2, 1e-1])]
    rel_tol = float(sc.get("active_rel_tol", 0.1))
    if ctx.quick:
        l1_grid, nuc_grid = l1_grid[::2], nuc_grid[::2]
    out = {}
    for label, grid, fixed in (("lambda_l1 (sparsity)", l1_grid, {"lambda_nuc": gw["lambda_nuc"]}),
                               ("lambda_nuc (low rank)", nuc_grid, {"lambda_l1": gw["lambda_l1"]})):
        key = "lambda_l1" if label.startswith("lambda_l1") else "lambda_nuc"
        rows = []
        with Timer(f"Fig.12 sweep {key}"):
            for lam in grid:
                model = ctx.base_model(**fixed, **{key: lam})
                auc = kfold_auc(model, ds.marks, pairs, ctx.cfg["cv"]["outer_folds"], ctx.seed)
                fit = model.clone().fit(ds.marks, full)
                th = fit.theta_
                active = int((np.abs(th).max(1) > rel_tol * th.max()).sum()) if th.max() > 0 else 0
                sv = np.linalg.svd(th, compute_uv=False)
                rank = int((sv > 1e-3 * sv[0]).sum()) if sv[0] > 0 else 0
                rows.append({key: lam, "auc": auc, "active_marks": active, "effective_rank": rank,
                             "active": ",".join(m for m, r in zip(fit.marks_, np.abs(th).max(1)) if th.max() > 0 and r > rel_tol * th.max())})
                log.info("%s=%g auc=%.3f active=%d rank=%d", key, lam, auc, active, rank)
        df = pd.DataFrame(rows)
        df.to_csv(ctx.path(f"fig12_{key}.csv"), index=False)
        x = np.array(grid)
        xplot = np.where(x > 0, x, x[x > 0].min() / 10 if np.any(x > 0) else 1e-6)
        plotting.line_plot(xplot, {"held-out AUC": df["auc"]}, f"{key} (0 plotted at left edge)", "ROC AUC",
                           f"Fig. 12: robustness to {label}", ctx.path(f"fig12_{key}.png"), logx=True, ylim=(0.5, 1.0))
        out[key] = rows
    ctx.save_json(out, "robustness.json")
    return out
