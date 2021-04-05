"""Matplotlib figures mirroring the paper's figures."""
from __future__ import annotations

from typing import Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

plt.rcParams.update({"figure.dpi": 120, "savefig.dpi": 200, "axes.spines.top": False, "axes.spines.right": False,
                     "font.size": 9})

SEQ_CMAP = "viridis"
DIV_CMAP = "RdBu_r"
CAT = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3", "#937860", "#DA8BC3", "#8C8C8C"]


def savefig(fig, path):
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def bar(values: dict, title: str, ylabel: str, path, ylim=None, color=CAT[0], rotate: int = 60):
    fig, ax = plt.subplots(figsize=(max(4, 0.35 * len(values) + 1), 3.2))
    ax.bar(list(values.keys()), list(values.values()), color=color)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    if ylim:
        ax.set_ylim(*ylim)
    plt.setp(ax.get_xticklabels(), rotation=rotate, ha="right")
    savefig(fig, path)


def grouped_bar(table: dict, groups: Sequence[str], title: str, ylabel: str, path, ylim=None):
    """table: series name -> list of values aligned with ``groups``."""
    fig, ax = plt.subplots(figsize=(max(5, 0.5 * len(groups) * len(table) / 2 + 2), 3.4))
    k = len(table)
    w = 0.8 / k
    x = np.arange(len(groups))
    for i, (name, vals) in enumerate(table.items()):
        ax.bar(x + (i - (k - 1) / 2) * w, vals, w, label=name, color=CAT[i % len(CAT)])
    ax.set_xticks(x)
    ax.set_xticklabels(groups, rotation=60, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    if ylim:
        ax.set_ylim(*ylim)
    ax.legend(frameon=False, fontsize=8)
    savefig(fig, path)


def heatmap(M: np.ndarray, rows: Sequence[str], cols: Sequence[str], title: str, path, cmap=SEQ_CMAP,
            vmin=None, vmax=None, annotate: bool = False, cbar_label: str = ""):
    fig, ax = plt.subplots(figsize=(0.32 * len(cols) + 2.5, 0.32 * len(rows) + 1.6))
    im = ax.imshow(M, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
    ax.set_xticks(range(len(cols)))
    ax.set_yticks(range(len(rows)))
    ax.set_xticklabels(cols, rotation=90)
    ax.set_yticklabels(rows)
    ax.set_title(title)
    if annotate:
        for i in range(M.shape[0]):
            for j in range(M.shape[1]):
                if np.isfinite(M[i, j]):
                    ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=6,
                            color="white" if M[i, j] < (np.nanmax(M) + np.nanmin(M)) / 2 else "black")
    cb = fig.colorbar(im, ax=ax, fraction=0.04)
    cb.set_label(cbar_label)
    savefig(fig, path)


def clustered_heatmap(M: np.ndarray, labels: Sequence[str], title: str, path, cbar_label: str = "interaction probability"):
    """Heat map with a dendrogram on top (Figs. 2-3)."""
    from scipy.cluster.hierarchy import dendrogram, linkage
    from scipy.spatial.distance import pdist

    n = len(labels)
    if n > 2:
        Z = linkage(pdist(M, metric="euclidean"), method="average")
        order = dendrogram(Z, no_plot=True)["leaves"]
    else:
        Z, order = None, list(range(n))
    Mo = M[np.ix_(order, order)]
    lab = [labels[i] for i in order]
    fig = plt.figure(figsize=(0.32 * n + 3, 0.32 * n + 2.6))
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 4], width_ratios=[20, 1], hspace=0.02, wspace=0.03)
    axd = fig.add_subplot(gs[0, 0])
    if Z is not None:
        dendrogram(Z, ax=axd, color_threshold=0, above_threshold_color="#555555", no_labels=True)
    axd.axis("off")
    ax = fig.add_subplot(gs[1, 0])
    im = ax.imshow(Mo, cmap=SEQ_CMAP, aspect="auto", vmin=0)
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(lab, rotation=90)
    ax.set_yticklabels(lab)
    cax = fig.add_subplot(gs[1, 1])
    cb = fig.colorbar(im, cax=cax)
    cb.set_label(cbar_label)
    axd.set_title(title)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def roc_plot(curves: dict, title: str, path):
    fig, ax = plt.subplots(figsize=(4, 4))
    for i, (name, (fpr, tpr, auc)) in enumerate(curves.items()):
        ax.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})", color=CAT[i % len(CAT)], lw=1.5)
    ax.plot([0, 1], [0, 1], ls="--", color="#999999", lw=0.8)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title(title)
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    savefig(fig, path)


def line_plot(x, series: dict, xlabel: str, ylabel: str, title: str, path, logx: bool = False, ylim=None):
    fig, ax = plt.subplots(figsize=(4.5, 3.2))
    for i, (name, y) in enumerate(series.items()):
        ax.plot(x, y, marker="o", ms=3, label=name, color=CAT[i % len(CAT)])
    if logx:
        ax.set_xscale("log")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    if ylim:
        ax.set_ylim(*ylim)
    ax.legend(frameon=False, fontsize=8)
    savefig(fig, path)


def venn2(n_a: int, n_b: int, n_ab: int, labels=("Real", "Predicted"), title: str = "", path=None):
    """Two-set Venn diagram drawn with plain matplotlib (Fig. 5)."""
    from matplotlib.patches import Circle

    fig, ax = plt.subplots(figsize=(4, 3))
    ax.add_patch(Circle((-0.45, 0), 1, alpha=0.45, color=CAT[0]))
    ax.add_patch(Circle((0.45, 0), 1, alpha=0.45, color=CAT[1]))
    ax.text(-0.95, 0, f"{n_a - n_ab:,}", ha="center", va="center")
    ax.text(0.95, 0, f"{n_b - n_ab:,}", ha="center", va="center")
    ax.text(0, 0, f"{n_ab:,}", ha="center", va="center", fontweight="bold")
    ax.text(-0.45, 1.12, labels[0], ha="center")
    ax.text(0.45, 1.12, labels[1], ha="center")
    ax.set_xlim(-1.6, 1.6)
    ax.set_ylim(-1.2, 1.35)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(title)
    savefig(fig, path)


def matrix_pair(A: np.ndarray, B: np.ndarray, titles=("Real", "Predicted"), path=None, log: bool = True,
                domains_a=None, domains_b=None):
    """Side-by-side contact maps (Fig. 6)."""
    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    for ax, M, t, doms in zip(axes, (A, B), titles, (domains_a, domains_b)):
        X = np.log1p(M) if log else M
        vmax = np.percentile(X[X > 0], 98) if np.any(X > 0) else 1
        ax.imshow(X, cmap="Reds", vmin=0, vmax=vmax, interpolation="nearest")
        if doms:
            for s, e in doms:
                ax.plot([s - 0.5, e - 0.5, e - 0.5], [s - 0.5, s - 0.5, e - 0.5], color="#1f77b4", lw=0.7)
        ax.set_title(t)
        ax.set_xticks([])
        ax.set_yticks([])
    savefig(fig, path)
