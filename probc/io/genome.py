"""Restriction sites and genome utilities."""
from __future__ import annotations

import gzip
import re
from typing import Iterable, Optional

import numpy as np

# motif, cut offset within the motif
ENZYMES = {
    "MboI": ("GATC", 0),
    "DpnII": ("GATC", 0),
    "Sau3AI": ("GATC", 0),
    "HindIII": ("AAGCTT", 1),
    "NcoI": ("CCATGG", 1),
    "Arima": (r"GATC|GA[ACGT]TC", 0),
}


def _open(path):
    path = str(path)
    return gzip.open(path, "rt") if path.endswith(".gz") else open(path)


def iter_fasta(path, chroms: Optional[Iterable[str]] = None):
    """Yield (name, sequence) from a (gzipped) FASTA, upper-cased."""
    wanted = None if chroms is None else set(chroms)
    name, buf = None, []
    with _open(path) as fh:
        for line in fh:
            if line.startswith(">"):
                if name is not None and (wanted is None or name in wanted):
                    yield name, "".join(buf).upper()
                name = line[1:].split()[0]
                buf = []
            elif wanted is None or name in wanted:
                buf.append(line.strip())
    if name is not None and (wanted is None or name in wanted):
        yield name, "".join(buf).upper()


def restriction_sites_from_fasta(fasta, enzyme: str = "MboI", chroms: Optional[Iterable[str]] = None) -> dict:
    """Cut-site coordinates per chromosome (0-based)."""
    motif, offset = ENZYMES[enzyme] if enzyme in ENZYMES else (enzyme, 0)
    pat = re.compile(motif)
    out = {}
    for name, seq in iter_fasta(fasta, chroms):
        out[name] = np.fromiter((m.start() + offset for m in pat.finditer(seq)), dtype=np.int64)
    return out


def read_juicer_site_file(path, chroms: Optional[Iterable[str]] = None) -> dict:
    """Read a Juicer restriction-site file (``chrom pos1 pos2 ...`` per line)."""
    wanted = None if chroms is None else set(chroms)
    out = {}
    with _open(path) as fh:
        for line in fh:
            parts = line.split()
            if not parts:
                continue
            c = parts[0]
            if wanted is not None and c not in wanted and f"chr{c}" not in wanted:
                continue
            c = c if c.startswith("chr") else f"chr{c}"
            out[c] = np.asarray(parts[1:], dtype=np.int64)
    return out


def write_juicer_site_file(sites: dict, path) -> None:
    with open(path, "w") as fh:
        for c, pos in sites.items():
            fh.write(c + " " + " ".join(map(str, pos.tolist())) + "\n")


def fixed_bins(chrom_sizes: dict, size: int) -> dict:
    """Fixed-size bins (used as nodes for Micro-C, which has no restriction sites)."""
    return {c: np.arange(0, L, size, dtype=np.int64) for c, L in chrom_sizes.items()}


def read_chrom_sizes(path) -> dict:
    out = {}
    with _open(path) as fh:
        for line in fh:
            p = line.split()
            if len(p) >= 2:
                out[p[0]] = int(p[1])
    return out
