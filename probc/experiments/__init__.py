"""Experiment modules, one per section of the paper's Results.

=================  ==========================================================
name               paper content
=================  ==========================================================
mark_selection     Fig. 1, diagonal-only ("forbidden interactions") ranking,
                   histone vs TFBS coverage
heatmaps           Figs. 2-3, Table 2, ES vs IMR90 correlation
prediction         Figs. 4-8 (chromosome-pair AUC, Venn, matrices, HiC-Reg
                   baseline, distance-decay baselines)
domains            Fig. 9, Table 3 (TADs / compartments, NVI, boundaries)
perturbation       CTCF masking (Sox9-Kcnj2) and DS2 deletion (FLT3)
transfer           Figs. 10-11 (cross-species / cross-cell-type)
robustness         Fig. 12 (lambda sweeps)
stability          Fig. 13 (enzyme replicates, resolutions)
=================  ==========================================================
"""
from importlib import import_module

EXPERIMENTS = ["mark_selection", "heatmaps", "prediction", "domains", "perturbation", "transfer", "robustness",
               "stability"]


def get(name: str):
    if name not in EXPERIMENTS:
        raise KeyError(f"unknown experiment {name!r}; choose from {EXPERIMENTS}")
    return import_module(f"{__name__}.{name}")
