"""Data containers, universes, negative sampling, IO and binning."""
import gzip

import numpy as np
import pandas as pd
import pytest

from probc.binning import bin_ids, bin_pairs, contact_matrix
from probc.data import (MarkMatrix, PairSet, Universe, align_marks, build_full_pairset, enumerate_universe_pairs,
                        pair_outer_sum, sample_negatives, universe_linear_term)
from probc.io.genome import read_juicer_site_file, restriction_sites_from_fasta, write_juicer_site_file
from probc.io.hic import contacts_from_pairs, write_pairs
from probc.io.peaks import binarize_signal, count_near_nodes, read_bed, site_mark_matrix
from probc.perturb import delete_region, mask_marks
from probc.synthetic import simulate_genome


@pytest.fixture(scope="module")
def g():
    return simulate_genome(n_chroms=2, sites_per_chrom=400, seed=0)


def _brute(mm, universe):
    m = mm.n_marks
    W = np.zeros((m, m))
    n = 0
    for i in range(mm.n_sites):
        for j in range(i + 1, mm.n_sites):
            if universe.contains(mm, np.array([i]), np.array([j]))[0]:
                W += np.outer(mm.C[i], mm.C[j])
                n += 1
    return 0.5 * (W + W.T), n


@pytest.mark.parametrize("universe", [Universe("intra", 1, 50_000), Universe("intra", 2000, None),
                                      Universe("genome", 1, None)])
def test_universe_linear_term_exact(g, universe):
    sub = g.marks.subset_sites(np.r_[np.arange(120), np.arange(400, 500)])
    W, n = universe_linear_term(sub, universe)
    Wb, nb = _brute(sub, universe)
    assert n == nb
    assert np.allclose(W, Wb)


def test_enumerate_universe_pairs(g):
    u, v = enumerate_universe_pairs(g.marks, g.universe)
    W, n = universe_linear_term(g.marks, g.universe)
    assert len(u) == n and np.all(u < v)
    assert np.allclose(pair_outer_sum(g.marks, u, v), W)


def test_full_pairset_counts(g):
    full = build_full_pairset(g.marks, g.edges, g.universe)
    _, n = universe_linear_term(g.marks, g.universe)
    assert len(full) + full.n_implicit == n


def test_negative_sampling(g):
    rng = np.random.default_rng(0)
    neg = sample_negatives(g.marks, g.edges, g.universe, 500, rng)
    assert len(neg) == 500
    assert np.all(g.universe.contains(g.marks, neg.u, neg.v))
    edge_keys = set(g.edges.keys(g.marks.n_sites).tolist())
    assert not (set(neg.keys(g.marks.n_sites).tolist()) & edge_keys)
    assert len(np.unique(neg.keys(g.marks.n_sites))) == 500
    dm = sample_negatives(g.marks, g.edges, g.universe, 300, rng, strategy="distance_matched")
    assert len(dm) > 0 and np.all(g.universe.contains(g.marks, dm.u, dm.v))


def test_align_marks():
    a = MarkMatrix(np.eye(3), ["A", "B", "C"], ["chr1"] * 3, [1, 2, 3])
    b = MarkMatrix(np.ones((2, 2)), ["C", "A"], ["chr1"] * 2, [1, 2])
    a2, b2, shared = align_marks(a, b)
    assert shared == ["A", "C"] and b2.marks == ["A", "C"]


def test_count_near_nodes():
    sites = np.array([100, 300, 1000])
    # half-open peak [150, 201) covers bases 150..200: 50 bp from site 100, 100 bp from site 300
    c = count_near_nodes(sites, sites, np.array([150, 2000]), np.array([201, 2100]), 100)
    assert c.tolist() == [1, 1, 0]
    c = count_near_nodes(sites, sites, np.array([150]), np.array([200]), 100)  # last base 199 -> 101 bp from 300
    assert c.tolist() == [1, 0, 0]


def test_peaks_and_sites_io(tmp_path):
    fa = tmp_path / "g.fa.gz"
    with gzip.open(fa, "wt") as fh:
        fh.write(">chr1\nAAGATCTTTTTTTTTTAAGCTTGGGATC\n>chr2\nGATCAAAA\n")
    sites = restriction_sites_from_fasta(fa, "MboI")
    assert sites["chr1"].tolist() == [2, 24] and sites["chr2"].tolist() == [0]
    hind = restriction_sites_from_fasta(fa, "HindIII")
    assert hind["chr1"].tolist() == [17]
    write_juicer_site_file(sites, tmp_path / "s.txt")
    back = read_juicer_site_file(tmp_path / "s.txt")
    assert back["chr1"].tolist() == [2, 24]
    bed = tmp_path / "p.bed"
    pd.DataFrame([["chr1", 0, 10, ".", 0, ".", 1, 1, 1, 5]]).to_csv(bed, sep="\t", header=False, index=False)
    peaks = read_bed(bed)
    mm = site_mark_matrix({"chr1": sites["chr1"]}, {"M": peaks}, max_dist=5)
    assert mm.C[:, 0].tolist() == [1, 0]
    mm2 = site_mark_matrix({"chr1": sites["chr1"]}, {"M": peaks}, max_dist=100, mode="summit")
    assert mm2.C[:, 0].tolist() == [1, 1]


def test_pairs_roundtrip(tmp_path, g):
    sub = PairSet(g.edges.u[:50], g.edges.v[:50], np.full(50, 2.0))
    write_pairs(tmp_path / "x.pairs", g.marks, sub)
    back = contacts_from_pairs(tmp_path / "x.pairs", g.marks, fmt="pairs", max_dist_to_site=0)
    k1 = dict(zip(sub.keys(g.marks.n_sites).tolist(), sub.y))
    k2 = dict(zip(back.keys(g.marks.n_sites).tolist(), back.y))
    assert k1 == k2


def test_binning(g):
    s2b, binned = bin_ids(g.marks, 10_000)
    assert np.allclose(binned.C.sum(0), g.marks.C.sum(0))
    bp = bin_pairs(g.edges, s2b)
    assert bp.y.sum() <= g.edges.y.sum()
    M = contact_matrix(binned, bp, "chr1")
    assert np.allclose(M, M.T)


def test_binarize_signal():
    assert binarize_signal(np.array([0.0, 0.5, 1.0, 10.0]), 0.5).tolist() == [0, 0, 1, 1]


def test_perturbations(g):
    c = "chr1"
    pos = g.marks.pos[g.marks.chrom == c]
    s, e = int(pos[100]), int(pos[150])
    masked = mask_marks(g.marks, c, s, e, ["CTCF"])
    j = g.marks.marks.index("CTCF")
    sel = (g.marks.chrom == c) & (g.marks.pos >= s) & (g.marks.pos < e)
    assert masked.C[sel, j].sum() == 0 and masked.C[~sel].sum() == g.marks.C[~sel].sum()
    deleted, keep = delete_region(g.marks, c, s, e)
    assert deleted.n_sites == g.marks.n_sites - sel.sum()
    after = deleted.pos[(deleted.chrom == c)]
    assert np.all(np.diff(after) > 0)
