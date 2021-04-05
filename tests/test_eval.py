"""Metrics, TAD/compartment calling, baselines, attribution, CV and experiments."""
import numpy as np
import pytest

from probc import ProbC
from probc.attribution import mark_shapley, site_attribution
from probc.baselines import DistanceDecay, HiCRegRF
from probc.compartments import compartment_domains, compartment_eigenvector, compartments_from_eigenvector
from probc.cv import chromosome_pair_auc, grid_search, kfold_auc, nested_cv, stratified_folds
from probc.data import make_labeled_pairs
from probc.metrics import boundary_confusion, domains_to_labels, normalized_vi, roc_auc, scc
from probc.synthetic import simulate_genome
from probc.tads import armatus_dp


@pytest.fixture(scope="module")
def g():
    return simulate_genome(n_chroms=2, sites_per_chrom=600, seed=4)


def block_matrix(sizes, inside=5.0, outside=0.5, seed=0):
    n = sum(sizes)
    rng = np.random.default_rng(seed)
    M = np.full((n, n), outside)
    s = 0
    for k in sizes:
        M[s:s + k, s:s + k] = inside
        s += k
    M = M + rng.random((n, n)) * 0.1
    return 0.5 * (M + M.T)


def test_armatus_dp_recovers_blocks():
    M = block_matrix([10, 15, 8, 12])
    doms = armatus_dp(M, gamma=0.5)
    assert [d for d in doms if d[1] - d[0] > 2] == [(0, 10), (10, 25), (25, 33), (33, 45)]


def test_compartments_two_types():
    lab = np.array([1] * 10 + [-1] * 10 + [1] * 10 + [-1] * 10)
    M = 1.0 + 0.8 * np.outer(lab, lab)
    ev = compartment_eigenvector(M, orient_track=(lab > 0).astype(float))
    assert np.all(compartments_from_eigenvector(ev) == lab)
    assert compartment_domains(lab) == [(0, 10), (10, 20), (20, 30), (30, 40)]


def test_nvi_properties():
    a = np.array([0, 0, 1, 1, 2, 2])
    assert normalized_vi(a, a) == 0
    assert normalized_vi(a, np.array([5, 5, 7, 7, 9, 9])) == 0
    assert 0 < normalized_vi(a, np.arange(6)) <= 1
    assert domains_to_labels([(0, 2)], 4).tolist() == [0, 0, 1, 2]


def test_boundary_confusion():
    r = boundary_confusion([0, 10, 21], [0, 10, 20, 30], tolerance=0)
    assert (r["TP"], r["FN"], r["FP"]) == (2, 2, 1)
    r = boundary_confusion([0, 10, 21], [0, 10, 20, 30], tolerance=1)
    assert (r["TP"], r["FN"], r["FP"]) == (3, 1, 0)


def test_scc_identity():
    M = block_matrix([10, 10])
    assert scc(M, M, h=0) == pytest.approx(1.0)


def test_attribution_sums(g):
    rng = np.random.default_rng(0)
    m = g.marks.n_marks
    Z = np.abs(rng.normal(size=(m, m)))
    Z = Z + Z.T
    u, v = g.edges.u[:20], g.edges.v[:20]
    att = site_attribution(Z, g.marks, u, v)
    s = np.einsum("ij,ij->i", g.marks.C[u] @ Z, g.marks.C[v])
    assert np.allclose(att.sum(1), s)
    assert np.allclose(mark_shapley(Z, g.marks.C[u[0]], g.marks.C[v[0]]), att[0])


def test_baselines(g):
    pairs = make_labeled_pairs(g.marks, g.edges, g.universe, rng=np.random.default_rng(0))
    dd = DistanceDecay(1.5).fit(g.marks, pairs)
    p = dd.predict_proba(g.marks, pairs.u, pairs.v)
    assert np.all((p >= 0) & (p <= 1)) and 0 < dd.alpha_
    rf = HiCRegRF(n_estimators=20, n_jobs=1).fit(g.marks, pairs)
    assert roc_auc(pairs.y, rf.predict_proba(g.marks, pairs.u, pairs.v)) > 0.7


def test_cv_routines(g):
    pairs = make_labeled_pairs(g.marks, g.edges, g.universe, rng=np.random.default_rng(0), reweight=True)
    f = stratified_folds(pairs.y, 5, np.random.default_rng(0))
    assert set(np.unique(f)) == set(range(5))
    base = ProbC(max_iter=100)
    grid = {"lambda_nuc": [1e-4, 1e-2], "lambda_l1": [1e-4]}
    gs = grid_search(base, g.marks, pairs, grid, n_folds=3)
    assert gs.best_params["lambda_nuc"] in (1e-4, 1e-2)
    res = nested_cv(base, g.marks, pairs, grid, outer_folds=3, inner_folds=2)
    assert len(res.outer_scores) == 3 and res.mean_auc > 0.7
    assert kfold_auc(base, g.marks, pairs, 3) > 0.7
    A, rows, cols = chromosome_pair_auc(base, g.marks, pairs, {"lambda_nuc": [1e-3], "lambda_l1": [1e-3]}, n_folds=3)
    assert A.shape == (2, 2) and np.all(A > 0.6)


def test_experiments_quick(tmp_path):
    """Every experiment module runs end-to-end on synthetic data."""
    from probc import experiments
    from probc.experiments.common import make_context

    overrides = {"synthetic": {"n_chroms": 2, "sites_per_chrom": 500}, "cv": {"outer_folds": 3, "inner_folds": 2},
                 "mark_selection": {"coverage_max_marks": 3},
                 "robustness": {"lambda_l1": [0, 1e-2], "lambda_nuc": [0, 1e-2]},
                 "prediction": {"hicreg": {"n_estimators": 10, "n_jobs": 1}}}
    for name in experiments.EXPERIMENTS:
        ctx = make_context("configs/experiments.yaml", synthetic=True, outdir=str(tmp_path / name), quick=True,
                           overrides=overrides)
        experiments.get(name).run(ctx)
        assert any((tmp_path / name).glob("*.json"))
