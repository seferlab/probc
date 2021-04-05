"""Section "PROBC predicts Hi-C interactions and genomic structures from chromatin
marks" (Figs. 4-8).

Outputs
-------
fig4_chromosome_pair_auc.{png,csv}   AUC training on chromosome i, testing on j
fig5_venn_<chrom>.png                 real vs predicted interactions (held-out folds)
fig6_matrix_<chrom>.png               real vs predicted contact maps at 10 kb
fig7_method_comparison.{png,csv}      per-chromosome AUC: ProbC vs HiC-Reg-style RF
fig8_roc_distance_decay_<chrom>.png   ROC: ProbC vs distance decay (beta = 1, 1.5, 2)
prediction.json                       summary numbers (incl. SCC / Spearman between methods)
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from .. import plotting
from ..baselines import DistanceDecay, HiCRegRF
from ..binning import bin_ids, bin_pairs, contact_matrix, predicted_contact_matrix
from ..cv import chromosome_pair_auc, nested_cv, stratified_folds
from ..data import build_full_pairset
from ..metrics import roc_auc, roc_curve, scc, spearman_matrices
from .common import Context, Timer, chrom_dataset, log


def _marks_subset(ctx: Context):
    sc = ctx.section("prediction")
    return sc.get("marks", ["H3K27ac", "H3K9me3", "H3K4me3", "H3K4me1", "CTCF"])


def run(ctx: Context) -> dict:
    sc = ctx.section("prediction")
    name = ctx.cfg["primary"]
    ds_all = ctx.dataset(name)
    marks = [m for m in _marks_subset(ctx) if m in ds_all.marks.marks] if sc.get("use_top_marks", True) else ds_all.marks.marks
    from .common import restrict_marks

    ds = restrict_marks(ds_all, marks)
    log.info("prediction marks: %s", marks)
    pairs = ctx.labeled_pairs(ds)
    chroms = ctx.chromosomes(ds)
    cv = ctx.cfg["cv"]
    out = {"marks": marks}

    # ---- Fig. 4: chromosome-pair AUC matrix -----------------------------------------
    with Timer("Fig.4 chromosome-pair AUC"):
        A, rows, cols = chromosome_pair_auc(ctx.base_model(), ds.marks, pairs, ctx.grid(), chroms,
                                            cv["outer_folds"], cv["inner_folds"], cv["criterion"], ctx.seed,
                                            progress=lambda s: log.info(s))
        pd.DataFrame(A, index=rows, columns=cols).to_csv(ctx.path("fig4_chromosome_pair_auc.csv"))
        plotting.heatmap(A, rows, cols, "Fig. 4: AUC (train row, test column)", ctx.path("fig4_chromosome_pair_auc.png"),
                         vmin=0.5, vmax=1.0, annotate=len(rows) <= 8, cbar_label="ROC AUC")
        out["fig4_diag_mean"] = float(np.nanmean(np.diag(A)))
        out["fig4_offdiag_mean"] = float(np.nanmean(A[~np.eye(len(rows), dtype=bool)])) if len(rows) > 1 else None
        best = int(np.nanargmax(np.diag(A)))
        out["fig4_best_chrom"] = rows[best]

    # ---- Figs. 5, 7, 8: per-chromosome nested CV, baselines -----------------------------
    with Timer("Figs.5/7/8 per-chromosome CV and baselines"):
        comp = []
        betas = sc.get("distance_betas", [1.0, 1.5, 2.0])
        rf_kw = sc.get("hicreg", {})
        roc_chrom = sc.get("roc_chromosome", chroms[0])
        venn_chrom = sc.get("venn_chromosome", out["fig4_best_chrom"])
        corr_rows = []
        for i, c in enumerate(chroms):
            dc = chrom_dataset(ds, c)
            pc = ctx.labeled_pairs(dc, offset=10 + i)
            res = nested_cv(ctx.base_model(), dc.marks, pc, ctx.grid(), cv["outer_folds"], cv["inner_folds"],
                            cv["criterion"], ctx.seed + i)
            row = {"chrom": c, "ProbC": res.mean_auc}
            # baselines on identical folds
            folds = stratified_folds(pc.y, cv["outer_folds"], np.random.default_rng(ctx.seed + i))
            p_rf = np.zeros(len(pc))
            p_dd = {b: np.zeros(len(pc)) for b in betas}
            y_cv = (pc.y > 0).astype(int)
            rf_auc, dd_auc = [], {b: [] for b in betas}
            for f in range(cv["outer_folds"]):
                tr, te = pc.subset(folds != f), pc.subset(folds == f)
                te_idx = np.flatnonzero(folds == f)
                rf = HiCRegRF(**rf_kw).fit(dc.marks, tr)
                p_rf[te_idx] = rf.predict_proba(dc.marks, te.u, te.v)
                rf_auc.append(roc_auc(y_cv[te_idx], p_rf[te_idx]))
                for b in betas:
                    dd = DistanceDecay(beta=b).fit(dc.marks, tr)
                    p_dd[b][te_idx] = dd.predict_proba(dc.marks, te.u, te.v)
                    dd_auc[b].append(roc_auc(y_cv[te_idx], p_dd[b][te_idx]))
            row["HiC-Reg (RF)"] = float(np.nanmean(rf_auc))
            for b in betas:
                row[f"distance decay b={b}"] = float(np.nanmean(dd_auc[b]))
            comp.append(row)
            # correlation between ProbC and RF predictions on the same pairs (nested folds use the same seed)
            probc_full = np.zeros(len(pc))
            folds_n = stratified_folds(pc.y, cv["outer_folds"], np.random.default_rng(ctx.seed + i))
            for f, (te, p) in enumerate(res.predictions):
                probc_full[np.flatnonzero(folds_n == f)] = p
            corr_rows.append({"chrom": c, "spearman_probc_rf": float(stats.spearmanr(probc_full, p_rf).statistic)})
            if c == roc_chrom:
                curves = {}
                fpr, tpr, _ = roc_curve(y_cv, probc_full)
                curves["ProbC"] = (fpr, tpr, roc_auc(y_cv, probc_full))
                for b in betas:
                    fpr, tpr, _ = roc_curve(y_cv, p_dd[b])
                    curves[f"distance decay β={b}"] = (fpr, tpr, roc_auc(y_cv, p_dd[b]))
                plotting.roc_plot(curves, f"Fig. 8: ROC on {c}", ctx.path(f"fig8_roc_distance_decay_{c}.png"))
            if c == venn_chrom:
                n_real = int(y_cv.sum())
                top = np.argsort(-probc_full)[:n_real]
                pred_pos = np.zeros(len(pc), bool)
                pred_pos[top] = True
                n_ab = int((pred_pos & (y_cv == 1)).sum())
                plotting.venn2(n_real, int(pred_pos.sum()), n_ab, title=f"Fig. 5: {c} (held-out pairs)",
                               path=ctx.path(f"fig5_venn_{c}.png"))
                out["fig5"] = {"chrom": c, "real": n_real, "predicted": int(pred_pos.sum()), "overlap": n_ab}
            log.info("%s: %s", c, {k: round(v, 3) for k, v in row.items() if k != "chrom"})
        df = pd.DataFrame(comp)
        df.to_csv(ctx.path("fig7_method_comparison.csv"), index=False)
        pd.DataFrame(corr_rows).to_csv(ctx.path("probc_vs_rf_correlation.csv"), index=False)
        plotting.grouped_bar({k: df[k].tolist() for k in df.columns if k != "chrom"}, df["chrom"].tolist(),
                             "Fig. 7: Hi-C interaction prediction (nested CV)", "ROC AUC",
                             ctx.path("fig7_method_comparison.png"), ylim=(0.4, 1.0))
        out["fig7_mean"] = {k: float(df[k].mean()) for k in df.columns if k != "chrom"}

    # ---- Fig. 6: real vs predicted 10 kb matrices ----------------------------------------
    with Timer("Fig.6 matrices"):
        res_bp = int(sc.get("matrix_resolution", 10_000))
        c = sc.get("matrix_chromosome", chroms[0])
        dc = chrom_dataset(ds, c)
        full = build_full_pairset(dc.marks, dc.edges, dc.universe)
        gw = ctx.cfg["genome_wide"]
        model = ctx.base_model(lambda_nuc=gw["lambda_nuc"], lambda_l1=gw["lambda_l1"]).fit(dc.marks, full)
        s2b, binned = bin_ids(dc.marks, res_bp)
        real = contact_matrix(binned, bin_pairs(dc.edges, s2b), c)
        pred = predicted_contact_matrix(model.theta_, binned, c)
        band = int(np.ceil((ds.universe.max_dist or 10 ** 12) / res_bp))
        idx = np.arange(real.shape[0])
        mask = np.abs(idx[:, None] - idx[None, :]) <= band
        pred = pred * mask
        nmax = int(sc.get("matrix_max_bins", 400))
        plotting.matrix_pair(real[:nmax, :nmax], pred[:nmax, :nmax],
                             (f"Real ({c}, {res_bp // 1000} kb)", "ProbC predicted"), ctx.path(f"fig6_matrix_{c}.png"))
        out["fig6"] = {"chrom": c, "scc": scc(real, pred, h=1, max_diag=min(band, real.shape[0] - 1)),
                       "spearman": spearman_matrices(real * mask, pred, k=1)}
    ctx.save_json(out, "prediction.json")
    return out
