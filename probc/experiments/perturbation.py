"""Section "PROBC can predict perturbations in 3D architecture".

(1) CTCF masking at the Sox9-Kcnj2 locus (Despang et al. 2019): train on human ES,
    predict mouse ES, mask the CTCF sites at the TAD boundary and then all CTCF sites
    between the two genes, and check whether the two TADs fuse.
(2) DS2 deletion at 13q12.2 (Yang et al. 2020): cut DS2 out of all mark inputs, join
    the flanks, and check whether the FLT3 (DS1) and PAN3 (DS3) TADs fuse and their
    predicted interaction increases.

Coordinates are taken from ``configs/experiments.yaml`` (``perturbation`` section).
In synthetic mode a planted domain boundary is used instead.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .. import plotting
from ..attribution import bin_attribution
from ..binning import bin_ids, predicted_contact_matrix
from ..data import MarkMatrix, align_marks, enumerate_universe_pairs
from ..perturb import delete_region, mask_marks
from ..tads import boundary_between, call_tads
from .common import Context, Timer, log
from .heatmaps import fit_genome_wide


def _window(mm: MarkMatrix, chrom: str, start: int, end: int) -> MarkMatrix:
    return mm.subset_sites(np.flatnonzero((mm.chrom == chrom) & (mm.pos >= start) & (mm.pos < end)))


def _predict(model, mm_win: MarkMatrix, res: int, band_bp):
    _, binned = bin_ids(mm_win, res)
    c = mm_win.chromosomes()[0]
    M = predicted_contact_matrix(model.theta_, binned, c)
    if band_bp:
        b = int(np.ceil(band_bp / res))
        idx = np.arange(M.shape[0])
        M = M * (np.abs(idx[:, None] - idx[None, :]) <= b)
    return M, binned


def _bin_of(binned: MarkMatrix, pos: int) -> int:
    return int(np.clip(np.searchsorted(binned.pos, pos, side="right") - 1, 0, binned.n_sites - 1))


def _anchor_contact(M, i, j, w=1):
    return float(M[max(i - w, 0): i + w + 1, max(j - w, 0): j + w + 1].sum())


def _synthetic_locus(ds, res):
    """Pick a planted domain boundary flanked by two domains as a test locus."""
    c = ds.marks.chromosomes()[0]
    dom = ds.meta["domains_bp"][c]
    k = max(1, len(dom) // 2)
    (s1, e1), (s2, e2) = dom[k - 1], dom[k]
    a = (s1 + e1) // 2
    b = (s2 + e2) // 2
    return c, s1 - 5 * res, e2 + 5 * res, a, b, (e1 - 2 * res, e1 + 2 * res)


def run(ctx: Context) -> dict:
    sc = ctx.section("perturbation")
    out = {}
    gamma = float(sc.get("gamma", 0.5))

    # -------------------------------------------------------------------------------
    with Timer("CTCF masking"):
        cm = sc.get("ctcf_masking", {})
        res = int(cm.get("resolution", 10_000))
        train = ctx.dataset(cm.get("train", ctx.cfg["primary"]))
        target = ctx.dataset(cm.get("target", "mESC"))
        tr_mm, tg_mm, shared = align_marks(train.marks, target.marks)
        from .common import restrict_marks

        train_s = restrict_marks(train, shared)
        target_s = restrict_marks(target, shared)
        model, _ = fit_genome_wide(ctx, train_s, offset=21)
        mask_names = [m for m in cm.get("marks", ["CTCF"]) if m in shared]
        if ctx.synthetic:
            c, ws, we, a, b, boundary = _synthetic_locus(target_s, res)
            between = (min(a, b), max(a, b))
        else:
            c = cm["chrom"]
            ws, we = cm["window"]
            a, b = cm["anchors"].values()
            boundary = tuple(cm["boundary_region"])
            between = tuple(cm.get("between_region", (min(a, b), max(a, b))))
        win = _window(target_s.marks, c, ws, we)
        variants = {
            "unperturbed": win,
            "boundary CTCF masked": mask_marks(win, c, boundary[0], boundary[1], mask_names),
            "all CTCF between masked": mask_marks(win, c, between[0], between[1], mask_names),
        }
        rows, mats = [], {}
        band = target_s.universe.max_dist
        for label, mmv in variants.items():
            M, binned = _predict(model, mmv, res, band)
            doms = call_tads(M, gamma=gamma, max_size=min(200, M.shape[0]))
            ia, ib = _bin_of(binned, a), _bin_of(binned, b)
            rows.append({"variant": label, "n_tads": len(doms), "separate_tads": boundary_between(doms, ia, ib),
                         "anchor_contact": _anchor_contact(M, ia, ib)})
            mats[label] = (M, doms)
        df = pd.DataFrame(rows)
        df.to_csv(ctx.path("perturbation_ctcf_masking.csv"), index=False)
        out["ctcf_masking"] = df.to_dict(orient="records")
        log.info("\n%s", df)
        # attribution of CTCF before / after masking (exact Shapley decomposition)
        if mask_names:
            att = {}
            for label in ("unperturbed", "all CTCF between masked"):
                mmv = variants[label]
                i, j = enumerate_universe_pairs(mmv, target_s.universe, max_pairs=int(sc.get("attribution_max_pairs", 5_000_000)))
                att[label] = bin_attribution(model.theta_, mmv.select_marks(model.marks_), i, j, mask_names[0])
            pd.DataFrame({"pos": win.pos, **att}).to_csv(ctx.path("perturbation_ctcf_attribution.csv"), index=False)
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 3, figsize=(11, 3.8))
        for ax, (label, (M, doms)) in zip(axes, mats.items()):
            X = np.log1p(M)
            ax.imshow(X, cmap="Reds", vmin=0, vmax=np.percentile(X[X > 0], 98) if np.any(X > 0) else 1)
            for s, e in doms:
                ax.plot([s - 0.5, e - 0.5, e - 0.5], [s - 0.5, s - 0.5, e - 0.5], color="#1f77b4", lw=0.8)
            ax.set_title(label, fontsize=9)
            ax.set_xticks([])
            ax.set_yticks([])
        plotting.savefig(fig, ctx.path("perturbation_ctcf_masking.png"))

    # -------------------------------------------------------------------------------
    with Timer("region deletion"):
        dl = sc.get("deletion", {})
        res = int(dl.get("resolution", 5_000))
        ds = ctx.dataset(dl.get("dataset", ctx.cfg["primary"]))
        model, _ = fit_genome_wide(ctx, ds, offset=22)
        if ctx.synthetic:
            c, ws, we, a, b, boundary = _synthetic_locus(ds, res)
            ds1, ds3 = a, b
            del_s, del_e = boundary
        else:
            c = dl["chrom"]
            ws, we = dl["window"]
            ds1 = int(np.mean(dl["DS1"]))
            ds3 = int(np.mean(dl["DS3"]))
            del_s, del_e = dl.get("delete", dl["DS2"])
        win = _window(ds.marks, c, ws, we)
        deleted, _ = delete_region(win, c, del_s, del_e)
        shift = del_e - del_s
        rows = []
        for label, mmv, p3 in (("wild type", win, ds3), ("deletion", deleted, ds3 - shift if ds3 >= del_e else ds3)):
            M, binned = _predict(model, mmv, res, ds.universe.max_dist)
            doms = call_tads(M, gamma=gamma, max_size=min(200, M.shape[0]))
            i1, i3 = _bin_of(binned, ds1), _bin_of(binned, p3)
            rows.append({"variant": label, "n_tads": len(doms), "separate_tads": boundary_between(doms, i1, i3),
                         "DS1_DS3_contact": _anchor_contact(M, i1, i3)})
        df = pd.DataFrame(rows)
        df.to_csv(ctx.path("perturbation_deletion.csv"), index=False)
        out["deletion"] = df.to_dict(orient="records")
        log.info("\n%s", df)
    ctx.save_json(out, "perturbation.json")
    return out
