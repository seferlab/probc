"""Collect the outputs of `probc run all` into one Markdown report.

    python scripts/summarize_results.py results/synthetic docs/synthetic_results.md [--copy-figures docs/img]
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import pandas as pd


def _json(p: Path):
    return json.loads(p.read_text()) if p.exists() else {}


def _table(df: pd.DataFrame, floatfmt: str = "{:.3f}") -> str:
    df = df.copy()
    for c in df.columns:
        if pd.api.types.is_float_dtype(df[c]):
            df[c] = df[c].map(lambda x: floatfmt.format(x) if pd.notna(x) else "")
    cols = [str(c) for c in df.columns]
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        lines.append("| " + " | ".join(str(v) for v in r.values) + " |")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("results")
    ap.add_argument("output")
    ap.add_argument("--copy-figures", default=None)
    a = ap.parse_args()
    R = Path(a.results)
    out = Path(a.output)
    figdir = Path(a.copy_figures) if a.copy_figures else None
    if figdir:
        figdir.mkdir(parents=True, exist_ok=True)

    def fig(rel: str, caption: str) -> str:
        p = R / rel
        if not p.exists():
            return ""
        if figdir:
            dst = figdir / p.name
            shutil.copy(p, dst)
            link = Path("img") / p.name if figdir.parent == out.parent else dst
        else:
            link = p
        return f"![{caption}]({link})\n\n*{caption}*\n"

    md = [f"# ProbC results: `{R}`\n"]
    if "synthetic" in str(R):
        md.append("> These numbers come from `python -m probc run all --synthetic`, i.e. from genomes simulated from the "
                  "ProbC model (4 chromosomes x 1,500 sites, the paper's 32 marks, planted signal on H3K27ac, H3K9me3, "
                  "H3K4me3, H3K4me1 and CTCF). They check that every analysis runs and behaves sensibly; they are not "
                  "estimates of the paper's real-data results.\n")
    t = _json(R / "timings.json")
    if t:
        md.append("Run time per experiment (s): " + ", ".join(f"{k} {v}" for k, v in t.items()) + "\n")

    ms = _json(R / "mark_selection" / "mark_selection.json")
    if ms:
        md.append("## Mark selection (Fig. 1, diagonal-only ranking, coverage)\n")
        md.append(f"Top marks with interactions between different marks forbidden: **{', '.join(ms.get('diagonal_top', []))}**\n")
        cov = pd.read_csv(R / "mark_selection" / "coverage_curve.csv")
        md.append(_table(cov) + "\n")
        md.append(fig("mark_selection/fig1_mark_frequency.png", "Fig. 1 analogue: frequency of histone marks in per-chromosome solutions"))
        md.append(fig("mark_selection/coverage_curve.png", "Held-out AUC vs number of top-ranked marks"))

    hm = _json(R / "heatmaps" / "heatmaps.json")
    if hm:
        md.append("## Interaction probabilities (Figs. 2-3, Table 2)\n")
        md.append(f"Pearson correlation of histone interaction matrices between cell types: **{hm.get('pearson', float('nan')):.3f}** "
                  f"(Spearman {hm.get('spearman', float('nan')):.3f}).\n")
        if "synthetic_recovery_corr" in hm:
            md.append(f"Correlation with planted parameters (histone-only model): {hm['synthetic_recovery_corr']:.3f}\n")
        md.append(_table(pd.read_csv(R / "heatmaps" / "table2_es_vs_imr90.csv")) + "\n")
        for p in sorted((R / "heatmaps").glob("fig2_*.png")) + sorted((R / "heatmaps").glob("fig3_*.png")):
            md.append(fig(f"heatmaps/{p.name}", p.stem.replace("_", " ")))

    pr = _json(R / "prediction" / "prediction.json")
    if pr:
        md.append("## Interaction prediction (Figs. 4-8)\n")
        md.append(f"Marks: {', '.join(pr['marks'])}. Fig. 4 mean AUC: within-chromosome "
                  f"{pr['fig4_diag_mean']:.3f}, cross-chromosome {pr.get('fig4_offdiag_mean') or float('nan'):.3f}.\n")
        md.append(_table(pd.read_csv(R / "prediction" / "fig7_method_comparison.csv")) + "\n")
        if "fig6" in pr:
            md.append(f"Fig. 6 ({pr['fig6']['chrom']}): SCC = {pr['fig6']['scc']:.3f}, Spearman = {pr['fig6']['spearman']:.3f}\n")
        md.append(fig("prediction/fig4_chromosome_pair_auc.png", "Fig. 4 analogue"))
        md.append(fig("prediction/fig7_method_comparison.png", "Fig. 7 analogue"))
        for p in sorted((R / "prediction").glob("fig8_*.png")) + sorted((R / "prediction").glob("fig6_*.png")) + \
                sorted((R / "prediction").glob("fig5_*.png")):
            md.append(fig(f"prediction/{p.name}", p.stem.replace("_", " ")))

    if (R / "domains" / "fig9_nvi_summary.csv").exists():
        md.append("## Domains (Fig. 9, Table 3)\n")
        md.append(_table(pd.read_csv(R / "domains" / "fig9_nvi_summary.csv")) + "\n")
        t3 = pd.read_csv(R / "domains" / "table3_tad_boundaries.csv", index_col=0)
        t3.index.name = "metric / resolution (bp)"
        t3 = t3.reset_index()
        md.append(_table(t3, "{:.3g}") + "\n")
        md.append(fig("domains/fig9_nvi.png", "Fig. 9 analogue (NVI, lower is better)"))

    pt = _json(R / "perturbation" / "perturbation.json")
    if pt:
        md.append("## Perturbations\n")
        for k, rows in pt.items():
            md.append(f"**{k}**\n\n" + _table(pd.DataFrame(rows)) + "\n")
        md.append(fig("perturbation/perturbation_ctcf_masking.png", "Predicted maps after CTCF masking"))

    tr = _json(R / "transfer" / "transfer.json")
    if tr:
        md.append("## Cross-species / cross-cell-type (Figs. 10-11)\n")
        md.append(_table(pd.DataFrame([{"experiment": k, "mean AUC": v["mean_auc"]} for k, v in tr.items()])) + "\n")
        md.append(fig("transfer/fig10a_human_from_mouse.png", "Fig. 10a analogue"))

    rb = _json(R / "robustness" / "robustness.json")
    if rb:
        md.append("## Robustness (Fig. 12)\n")
        for k, rows in rb.items():
            md.append(f"**{k}**\n\n" + _table(pd.DataFrame(rows).drop(columns=["active"], errors="ignore")) + "\n")

    st = _json(R / "stability" / "stability.json")
    if st:
        md.append("## Stability (Fig. 13)\n")
        md.append("Enzyme replicates, Spearman per chromosome: " + ", ".join(f"{k} {v:.3f}" for k, v in st["enzyme"].items()) + "\n")
        if "enzyme_pearson" in st:
            md.append("Enzyme replicates, Pearson per chromosome: " + ", ".join(f"{k} {v:.3f}" for k, v in st["enzyme_pearson"].items()) + "\n")
            md.append("Resolutions, mean Pearson: " + ", ".join(f"{k} {v:.3f}" for k, v in st["resolution_pearson"].items()) + "\n")
        md.append("Resolutions, mean Spearman: " + ", ".join(f"{k} {v:.3f}" for k, v in st["resolution"].items()) + "\n")
        md.append(fig("stability/fig13b_resolutions.png", "Fig. 13b analogue"))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(md))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
