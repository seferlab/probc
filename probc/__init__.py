"""ProbC: joint modeling of epigenome and transcriptome effects in 3D genome.

Reimplementation of Sefer E., BMC Genomics 23:287 (2022).
"""
from .data import MarkMatrix, PairSet, Universe, align_marks, build_full_pairset, make_labeled_pairs
from .model import ProbC

__version__ = "1.0.0"
__all__ = [
    "MarkMatrix", "PairSet", "Universe", "ProbC", "align_marks", "build_full_pairset", "make_labeled_pairs",
]
