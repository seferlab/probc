"""In-silico perturbations of the chromatin-mark input (paper section
"PROBC can predict perturbations in 3D architecture").

* :func:`mask_marks` -- remove selected marks (e.g. CTCF) in a genomic region, as
  in the Sox9-Kcnj2 CTCF-deletion experiment (Despang et al. 2019).
* :func:`delete_region` -- cut a region out of all mark inputs and join the two
  flanks ("the remaining two partitions were added together"), as in the 13q12.2
  / DS2 deletion near FLT3 (Yang et al. 2020).
"""
from __future__ import annotations

from typing import Sequence

import numpy as np

from .data import MarkMatrix


def mask_marks(mm: MarkMatrix, chrom: str, start: int, end: int, marks: Sequence[str]) -> MarkMatrix:
    C = mm.C.copy()
    sel = (mm.chrom == chrom) & (mm.pos >= start) & (mm.pos < end)
    cols = [mm.marks.index(m) for m in marks if m in mm.marks]
    C[np.ix_(np.flatnonzero(sel), cols)] = 0
    return MarkMatrix(C, mm.marks, mm.chrom, mm.pos, mm.end, dict(mm.meta))


def delete_region(mm: MarkMatrix, chrom: str, start: int, end: int) -> tuple[MarkMatrix, np.ndarray]:
    """Remove sites in [start, end) and shift downstream coordinates left.

    Returns the new MarkMatrix and the indices (into the old one) of kept sites.
    """
    on = mm.chrom == chrom
    inside = on & (mm.pos >= start) & (mm.pos < end)
    keep = np.flatnonzero(~inside)
    pos = mm.pos.copy()
    shift = on & (mm.pos >= end)
    pos[shift] -= end - start
    endv = None
    if mm.end is not None:
        endv = mm.end.copy()
        endv[shift] -= end - start
    new = MarkMatrix(mm.C[keep], mm.marks, mm.chrom[keep], pos[keep], None if endv is None else endv[keep], dict(mm.meta))
    return new, keep


def region_bins(mm: MarkMatrix, chrom: str, start: int, end: int) -> np.ndarray:
    return np.flatnonzero((mm.chrom == chrom) & (mm.pos >= start) & (mm.pos < end))
