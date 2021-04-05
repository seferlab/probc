"""Section "PROBC can accurately predict cross-cell types and cross-species
interactions" (Figs. 10-11).

For each (training chromosome of dataset A, test chromosome of dataset B) a model
is trained with nested CV on A and scored on held-out folds of B, using the marks
shared by both datasets.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .. import plotting
from ..cv import chromosome_pair_auc
from .common import Context, Timer, log, restrict_marks

DEFAULT_PAIRS = [
    ("mESC", "hESC", "fig10a_human_from_mouse"),
    ("hESC", "mESC", "fig10b_mouse_from_human"),
    ("IMR90", "hESC", "fig11_hESC_from_IMR90"),
    ("hESC", "IMR90", "fig11b_IMR90_from_hESC"),
]


def run(ctx: Context) -> dict:
    sc = ctx.section("transfer")
    wanted = sc.get("marks", ["H3K27ac", "H3K9me3", "H3K4me3", "H3K4me1", "CTCF"])
    pairs_cfg = [tuple(p) for p in sc.get("pairs", DEFAULT_PAIRS)]
    cv = ctx.cfg["cv"]
    out = {}
    for train_name, test_name, tag in pairs_cfg:
        with Timer(f"{tag}: {train_name} -> {test_name}"):
            A = ctx.dataset(train_name)
            B = ctx.dataset(test_name)
            shared = [m for m in A.marks.marks if m in B.marks.marks]
            marks = [m for m in wanted if m in shared] if wanted else shared
            if not marks:
                log.warning("no shared marks for %s", tag)
                continue
            A2, B2 = restrict_marks(A, marks), restrict_marks(B, marks)
            pa = ctx.labeled_pairs(A2, offset=1)
            pb = ctx.labeled_pairs(B2, offset=2)
            M, rows, cols = chromosome_pair_auc(ctx.base_model(), A2.marks, pa, ctx.grid(), ctx.chromosomes(A2),
                                                cv["outer_folds"], cv["inner_folds"], cv["criterion"], ctx.seed,
                                                test_mm=B2.marks, test_pairs=pb, test_chromosomes=ctx.chromosomes(B2))
            pd.DataFrame(M, index=[f"{train_name}:{r}" for r in rows], columns=[f"{test_name}:{c}" for c in cols]).to_csv(
                ctx.path(f"{tag}.csv"))
            plotting.heatmap(M, rows, cols, f"AUC: train {train_name} (rows) -> test {test_name} (cols)",
                             ctx.path(f"{tag}.png"), vmin=0.5, vmax=1.0, annotate=len(rows) <= 8, cbar_label="ROC AUC")
            per_test = dict(zip(cols, np.nanmean(M, axis=0)))
            plotting.bar(per_test, f"{test_name} predicted from {train_name}", "mean ROC AUC",
                         ctx.path(f"{tag}_per_chrom.png"), ylim=(0.4, 1.0))
            out[tag] = {"marks": marks, "mean_auc": float(np.nanmean(M)), "per_test_chrom": per_test}
            log.info("%s mean AUC %.3f", tag, out[tag]["mean_auc"])
    ctx.save_json(out, "transfer.json")
    return out
