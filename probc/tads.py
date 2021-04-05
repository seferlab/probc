"""TAD calling on observed / predicted interaction matrices.

The paper uses Armatus (Filippova et al. 2014) to call TADs on predicted
interactions. We provide

* :func:`armatus_dp` -- a pure-Python implementation of the Armatus objective
  (resolution-specific domain quality with mean-centering per domain size and an
  O(n * L) dynamic program), and
* :func:`run_armatus_binary` -- a thin wrapper around the original C++ binary if it
  is installed (``armatus`` on PATH).
"""
from __future__ import annotations

import gzip
import os
import shutil
import subprocess
import tempfile
from typing import Optional

import numpy as np


def _band_domain_sums(A: np.ndarray, max_size: int) -> list:
    """S[t][k] = sum_{k <= i < j <= k+t} A[i, j] for t = 0..max_size-1."""
    n = A.shape[0]
    S = [np.zeros(n)]
    if max_size <= 1:
        return S
    S.append(np.diagonal(A, 1).astype(float).copy())
    for t in range(2, max_size):
        if t >= n:
            break
        prev, prev2 = S[t - 1], S[t - 2]
        n_t = n - t
        cur = prev[:n_t] + prev[1:n_t + 1] - prev2[1:n_t + 1] + np.diagonal(A, t)[:n_t]
        S.append(cur)
    return S


def armatus_dp(A: np.ndarray, gamma: float = 0.5, max_size: Optional[int] = None, min_size: int = 2) -> list:
    """Armatus-style optimal domain segmentation.

    Parameters
    ----------
    A : (n, n) symmetric interaction matrix.
    gamma : resolution parameter; larger gamma gives smaller domains.
    max_size : maximal domain size in bins (default: min(n, 200)).
    min_size : minimal domain size in bins.

    Returns
    -------
    list of (start, end) bin intervals, end exclusive.
    """
    A = np.asarray(A, dtype=float)
    A = np.nan_to_num(A)
    n = A.shape[0]
    L = min(n, max_size or 200)
    S = _band_domain_sums(A, L)
    # quality q_t[k] of domain [k, k+t] (size t+1 bins)
    Q = []
    for t in range(len(S)):
        if t + 1 < min_size or t == 0:
            Q.append(None)
            continue
        dens = S[t] / (t ** gamma)
        Q.append(dens - dens.mean())
    T = len(S)
    Qmat = np.full((T, n), -np.inf)
    for t in range(T):
        if Q[t] is not None:
            Qmat[t, : len(Q[t])] = Q[t]
    OPT = np.zeros(n + 1)  # OPT[l+1] = best score for bins 0..l
    back = np.full(n + 1, -1, dtype=np.int64)  # start bin of domain ending at l, -1 = gap
    for end in range(n):
        best = OPT[end]
        arg = -1
        tmax = min(T, end + 1)
        if tmax > 1:
            ts = np.arange(1, tmax)
            ks = end - ts
            vals = OPT[ks] + Qmat[ts, ks]
            j = int(np.argmax(vals))
            if vals[j] > best:
                best = float(vals[j])
                arg = int(ks[j])
        OPT[end + 1] = best
        back[end + 1] = arg
    domains = []
    pos = n
    while pos > 0:
        k = back[pos]
        if k < 0:
            pos -= 1
        else:
            domains.append((int(k), int(pos)))
            pos = k
    return domains[::-1]


def run_armatus_binary(A: np.ndarray, gamma: float = 0.5, chrom: str = "chr1", resolution: int = 40000,
                       binary: str = "armatus") -> list:
    """Call the original Armatus implementation (if installed).

    Writes the matrix in Armatus' gzipped dense format (chrom, start, end, values...)
    and parses the ``.consensus.txt`` / gamma output. Check ``armatus --help`` for
    the flags of your version; this wrapper uses ``-i -g -o -r -c``.
    """
    exe = shutil.which(binary)
    if exe is None:
        raise FileNotFoundError("armatus binary not found; use armatus_dp() instead")
    n = A.shape[0]
    with tempfile.TemporaryDirectory() as tmp:
        inp = os.path.join(tmp, "matrix.txt.gz")
        with gzip.open(inp, "wt") as fh:
            for i in range(n):
                fh.write(f"{chrom}\t{i * resolution}\t{(i + 1) * resolution}\t" + "\t".join(f"{x:.6g}" for x in A[i]) + "\n")
        out = os.path.join(tmp, "out")
        subprocess.run([exe, "-i", inp, "-g", str(gamma), "-o", out, "-r", str(resolution), "-c", chrom],
                       check=True, capture_output=True)
        cand = [f for f in os.listdir(tmp) if f.startswith("out") and f.endswith(".txt")]
        if not cand:
            raise RuntimeError("Armatus produced no output")
        domains = []
        with open(os.path.join(tmp, sorted(cand)[0])) as fh:
            for line in fh:
                parts = line.split()
                if len(parts) >= 3:
                    s, e = int(parts[1]) // resolution, int(parts[2]) // resolution + 1
                    domains.append((s, min(e, n)))
    return domains


def call_tads(A: np.ndarray, method: str = "dp", **kw) -> list:
    if method == "dp":
        return armatus_dp(A, **kw)
    if method == "armatus":
        return run_armatus_binary(A, **kw)
    raise ValueError(method)


def boundary_between(domains: list, i: int, j: int) -> bool:
    """True if bins i and j fall in different domains (or one is outside any)."""
    lab = {}
    for k, (s, e) in enumerate(domains):
        for b in (i, j):
            if s <= b < e:
                lab[b] = k
    return lab.get(i, -1) != lab.get(j, -2)
