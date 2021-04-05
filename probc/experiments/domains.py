"""TAD and compartment prediction from ProbC-predicted interactions (Fig. 9, Table 3).

TADs are called with the Armatus objective (``probc.tads``) on both the observed
binned contact matrix and the ProbC-predicted matrix; compartments with the
first-eigenvector method. Agreement is measured by the Normalized Variation of
Information (lower is better) and by TAD-boundary confusion counts.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .. import plotting
from ..binning import bin_ids, bin_pairs, contact_matrix, predicted_contact_matrix
from ..compartments import compartment_domains, compartment_eigenvector, compartments_from_eigenvector
from ..data import build_full_pairset
from ..metrics import boundary_confusion, domain_boundaries, nvi_domains
from ..tads import call_tads
from .common import Context, Timer, chrom_dataset, log

MARK_SETS = {"histone": ("histone",), "tfbs": ("tf", "dnase"), "histone+tfbs": ("histone", "tf", "dnase")}


def _band(n: int, max_dist, res: int) -> np.ndarray:
    band = int(np.ceil((max_dist or 10 ** 12) / res))
    idx = np.arange(n)
    return np.abs(idx[:, None] - idx[None, :]) <= band


def _matrices(model, dc, res: int, distance_beta=None):
    c = dc.marks.chromosomes()[0]
    s2b, binned = bin_ids(dc.marks, res)
    real = contact_matrix(binned, bin_pairs(dc.edges, s2b), c)
    pred = predicted_contact_matrix(model.theta_, binned, c, distance_beta=distance_beta)
    mask = _band(real.shape[0], dc.universe.max_dist, res)
    return real * mask, pred * mask


def run(ctx: Context) -> dict:
    sc = ctx.section("domains")
    name = ctx.cfg["primary"]
    gamma = float(sc.get("gamma", 0.5))
    tad_method = sc.get("tad_method", "dp")
    resolutions = [int(r) for r in sc.get("nvi_resolutions", [10_000, 40_000])]
    table3_res = [int(r) for r in sc.get("table3_resolutions", [1_000, 5_000, 40_000, 100_000])]
    comp_res = int(sc.get("compartment_resolution", resolutions[0]))
    tol = int(sc.get("boundary_tolerance", 0))
    beta = sc.get("distance_beta", None)
    gw = ctx.cfg["genome_wide"]
    max_bins = int(sc.get("max_domain_bins", 200))
    out = {"nvi": [], "table3": {}}

    def tads(M, res):
        kw = {"gamma": gamma}
        if tad_method == "dp":
            kw["max_size"] = min(max_bins, max(2, int(sc.get("max_domain_bp", 2_000_000) // res)))
        return call_tads(M, method=tad_method, **kw)

    with Timer("Fig.9 NVI by mark set"):
        rows = []
        for set_name, kinds in MARK_SETS.items():
            ds = ctx.dataset(name, include=kinds)
            for c in ctx.chromosomes(ds):
                dc = chrom_dataset(ds, c)
                full = build_full_pairset(dc.marks, dc.edges, dc.universe)
                model = ctx.base_model(lambda_nuc=gw["lambda_nuc"], lambda_l1=gw["lambda_l1"]).fit(dc.marks, full)
                for res in resolutions:
                    real, pred = _matrices(model, dc, res, beta)
                    t_real, t_pred = tads(real, res), tads(pred, res)
                    rows.append(dict(marks=set_name, chrom=c, resolution=res, kind="TAD",
                                     nvi=nvi_domains(t_pred, t_real, real.shape[0])))
                real, pred = _matrices(model, dc, comp_res, beta)
                ev_r = compartment_eigenvector(real)
                ev_p = compartment_eigenvector(pred)
                cr = compartment_domains(compartments_from_eigenvector(ev_r))
                cp = compartment_domains(compartments_from_eigenvector(ev_p))
                rows.append(dict(marks=set_name, chrom=c, resolution=comp_res, kind="compartment",
                                 nvi=nvi_domains(cp, cr, real.shape[0])))
        df = pd.DataFrame(rows)
        df.to_csv(ctx.path("fig9_nvi.csv"), index=False)
        summary = df.groupby(["kind", "resolution", "marks"])["nvi"].mean().unstack("marks")
        summary.to_csv(ctx.path("fig9_nvi_summary.csv"))
        groups = [f"{k} {r // 1000}kb" for k, r in summary.index]
        plotting.grouped_bar({m: summary[m].tolist() for m in summary.columns}, groups,
                             "Fig. 9: domain prediction (lower NVI is better)", "NVI", ctx.path("fig9_nvi.png"))
        out["nvi"] = summary.reset_index().to_dict(orient="records")
        log.info("\n%s", summary)

    with Timer("Table 3 TAD boundaries"):
        ds = ctx.dataset(name)
        totals = {r: dict(TP=0, FN=0, FP=0, n_pred=0, n_true=0) for r in table3_res}
        for c in ctx.chromosomes(ds):
            dc = chrom_dataset(ds, c)
            full = build_full_pairset(dc.marks, dc.edges, dc.universe)
            model = ctx.base_model(lambda_nuc=gw["lambda_nuc"], lambda_l1=gw["lambda_l1"]).fit(dc.marks, full)
            for res in table3_res:
                real, pred = _matrices(model, dc, res, beta)
                if real.shape[0] < 4:
                    continue
                bc = boundary_confusion(domain_boundaries(tads(pred, res)), domain_boundaries(tads(real, res)), tol)
                for k in totals[res]:
                    totals[res][k] += bc[k]
        t3 = []
        for r, t in totals.items():
            tpr = t["TP"] / t["n_true"] if t["n_true"] else float("nan")
            t3.append({"resolution": r, "True Positives": t["TP"], "True Positive Rate": tpr,
                       "False Negatives": t["FN"], "False Negative Rate": 1 - tpr if t["n_true"] else float("nan"),
                       "False Positives": t["FP"], "Predicted TAD Boundaries": t["n_pred"],
                       "True TAD Boundaries": t["n_true"]})
        t3 = pd.DataFrame(t3).set_index("resolution").T
        t3.to_csv(ctx.path("table3_tad_boundaries.csv"))
        out["table3"] = t3.to_dict()
        log.info("\n%s", t3)
    ctx.save_json(out, "domains.json")
    return out
