"""Core model tests: gradients, convexity, proximal operators, ADMM, recovery."""
import numpy as np
import pytest

from probc import ProbC
from probc.data import build_full_pairset, make_labeled_pairs
from probc.objectives import bernoulli_nll, compile_pairs, poisson_nll
from probc.prox import singular_value_threshold, soft_threshold
from probc.synthetic import simulate_genome


@pytest.fixture(scope="module")
def genome():
    return simulate_genome(n_chroms=2, sites_per_chrom=800, seed=1)


@pytest.fixture(scope="module")
def genome_poisson():
    return simulate_genome(n_chroms=2, sites_per_chrom=800, seed=2, likelihood="poisson", theta_scale=0.1)


def _sym(rng, m, loc=0.05, scale=0.02):
    A = np.abs(rng.normal(loc, scale, (m, m)))
    return 0.5 * (A + A.T)


@pytest.mark.parametrize("lik", ["bernoulli", "poisson"])
def test_gradient_finite_differences(genome, genome_poisson, lik):
    g = genome if lik == "bernoulli" else genome_poisson
    full = build_full_pairset(g.marks, g.edges, g.universe)
    cp = compile_pairs(g.marks.C, full)
    fun = bernoulli_nll if lik == "bernoulli" else poisson_nll
    rng = np.random.default_rng(0)
    m = g.marks.n_marks
    Z = _sym(rng, m)
    f, G = fun(Z, cp)
    for _ in range(3):
        D = rng.normal(size=(m, m))
        D = 0.5 * (D + D.T)
        h = 1e-6
        fd = (fun(Z + h * D, cp, with_grad=False) - fun(Z - h * D, cp, with_grad=False)) / (2 * h)
        assert np.isclose(fd, np.sum(G * D), rtol=1e-5, atol=1e-6)


def test_compiled_pairs_match_bruteforce(genome):
    """Pattern compression must give exactly the per-pair NLL."""
    g = genome
    pairs = make_labeled_pairs(g.marks, g.edges, g.universe, neg_ratio=2, rng=np.random.default_rng(0), reweight=True)
    rng = np.random.default_rng(1)
    Z = _sym(rng, g.marks.n_marks)
    cp = compile_pairs(g.marks.C, pairs)
    f = bernoulli_nll(Z, cp, eps=0.0, with_grad=False)
    s = np.einsum("ij,ij->i", g.marks.C[pairs.u] @ Z, g.marks.C[pairs.v])
    pos = pairs.y > 0
    with np.errstate(divide="ignore"):
        brute = np.sum(pairs.w[pos] * -np.log(1 - np.exp(-s[pos]))) + np.sum(pairs.w[~pos] * s[~pos])
    assert np.isclose(f, brute, rtol=1e-10)


def test_convexity_midpoint(genome):
    """Theorem 1: the Bernoulli NLL is convex in Z >= 0."""
    full = build_full_pairset(genome.marks, genome.edges, genome.universe)
    cp = compile_pairs(genome.marks.C, full)
    rng = np.random.default_rng(3)
    for _ in range(20):
        A, B = _sym(rng, genome.marks.n_marks, 0.05, 0.05), _sym(rng, genome.marks.n_marks, 0.05, 0.05)
        t = rng.random()
        mid = bernoulli_nll(t * A + (1 - t) * B, cp, with_grad=False)
        assert mid <= t * bernoulli_nll(A, cp, with_grad=False) + (1 - t) * bernoulli_nll(B, cp, with_grad=False) + 1e-8


def test_prox_operators():
    rng = np.random.default_rng(0)
    Q = rng.normal(size=(6, 6))
    Q = Q + Q.T
    S = singular_value_threshold(Q, 1.0)
    s0 = np.linalg.svd(Q, compute_uv=False)
    s1 = np.linalg.svd(S, compute_uv=False)
    assert np.allclose(np.sort(s1)[::-1][: (s0 > 1).sum()], (s0 - 1)[s0 > 1])
    assert np.allclose(soft_threshold(np.array([-2.0, -0.5, 0.3, 1.5]), 1.0), [-1.0, 0, 0, 0.5])
    # prox optimality: minimiser of a*|x| + 0.5 (x - q)^2
    q = rng.normal(size=100)
    x = soft_threshold(q, 0.4)
    grid = np.linspace(-4, 4, 4001)
    for qi, xi in zip(q[:5], x[:5]):
        assert abs(grid[np.argmin(0.4 * np.abs(grid) + 0.5 * (grid - qi) ** 2)] - xi) < 5e-3


def test_admm_matches_unregularised_lbfgs(genome):
    """With zero penalties ADMM must reach the unpenalised MLE."""
    from scipy.optimize import minimize

    full = build_full_pairset(genome.marks, genome.edges, genome.universe)
    cp = compile_pairs(genome.marks.C, full)
    m = genome.marks.n_marks
    iu = np.triu_indices(m)

    def obj(x):
        Z = np.zeros((m, m))
        Z[iu] = x
        Z = Z + np.triu(Z, 1).T
        f, G = bernoulli_nll(Z, cp)
        g = G[iu] * np.where(iu[0] == iu[1], 1.0, 2.0)
        return f / cp.n_total, g / cp.n_total

    x0 = np.full(len(iu[0]), 0.01)
    ref = minimize(obj, x0, jac=True, method="L-BFGS-B", bounds=[(0, None)] * len(x0),
                   options={"maxiter": 2000, "gtol": 1e-12, "ftol": 1e-15})
    model = ProbC(lambda_nuc=0, lambda_l1=0, tol_rel=1e-6, tol_abs=1e-9, max_iter=500).fit(genome.marks, full)
    f_admm = obj(model.theta_[iu])[0]
    assert f_admm <= ref.fun + 1e-6


def test_parameter_recovery(genome):
    full = build_full_pairset(genome.marks, genome.edges, genome.universe)
    model = ProbC(lambda_nuc=1e-4, lambda_l1=1e-4).fit(genome.marks, full)
    corr = np.corrcoef(model.theta_.ravel(), genome.theta_true.ravel())[0, 1]
    assert corr > 0.98
    true_active = {genome.marks.marks[i] for i in np.flatnonzero(genome.theta_true.max(1) > 0)}
    assert set(model.active_marks(1e-3 * model.theta_.max())) == true_active


def test_poisson_recovery(genome_poisson):
    g = genome_poisson
    full = build_full_pairset(g.marks, g.edges, g.universe)
    model = ProbC(likelihood="poisson", lambda_nuc=1e-4, lambda_l1=1e-4).fit(g.marks, full)
    assert np.corrcoef(model.theta_.ravel(), g.theta_true.ravel())[0, 1] > 0.98
    assert model.X_.max() <= 1.0


def test_pgd_and_lbfgsb_agree(genome):
    full = build_full_pairset(genome.marks, genome.edges, genome.universe)
    a = ProbC(lambda_nuc=1e-4, lambda_l1=1e-4, inner_solver="pgd").fit(genome.marks, full)
    b = ProbC(lambda_nuc=1e-4, lambda_l1=1e-4, inner_solver="lbfgsb").fit(genome.marks, full)
    assert np.abs(a.theta_ - b.theta_).max() < 1e-3 * max(1.0, b.theta_.max()) + 1e-4


def test_regularisation_effects(genome):
    full = build_full_pairset(genome.marks, genome.edges, genome.universe)
    lo = ProbC(lambda_nuc=0, lambda_l1=1e-4).fit(genome.marks, full)
    hi = ProbC(lambda_nuc=0, lambda_l1=1e-1).fit(genome.marks, full)
    assert np.abs(hi.theta_).sum() < np.abs(lo.theta_).sum()
    r_lo = ProbC(lambda_nuc=1e-4, lambda_l1=0).fit(genome.marks, full)
    r_hi = ProbC(lambda_nuc=1e-1, lambda_l1=0).fit(genome.marks, full)
    assert np.linalg.svd(r_hi.theta_, compute_uv=False).sum() < np.linalg.svd(r_lo.theta_, compute_uv=False).sum()
    d = ProbC(lambda_nuc=0, lambda_l1=1e-3, support="diagonal").fit(genome.marks, full)
    assert np.allclose(d.theta_, np.diag(np.diag(d.theta_)))


def test_sampled_negatives_unbiased(genome):
    """Inverse-probability weighted negatives approximate the exact likelihood."""
    full = build_full_pairset(genome.marks, genome.edges, genome.universe)
    lab = make_labeled_pairs(genome.marks, genome.edges, genome.universe, neg_ratio=10,
                             rng=np.random.default_rng(0), reweight=True)
    a = ProbC(lambda_nuc=1e-4, lambda_l1=1e-4).fit(genome.marks, full)
    b = ProbC(lambda_nuc=1e-4, lambda_l1=1e-4).fit(genome.marks, lab)
    assert np.corrcoef(a.theta_.ravel(), b.theta_.ravel())[0, 1] > 0.97


def test_save_load(tmp_path, genome):
    full = build_full_pairset(genome.marks, genome.edges, genome.universe)
    m = ProbC(lambda_nuc=1e-3, lambda_l1=1e-3).fit(genome.marks, full)
    m.save(tmp_path / "m.npz")
    m2 = ProbC.load(tmp_path / "m.npz")
    assert np.allclose(m.theta_, m2.theta_) and m.marks_ == m2.marks_
    u, v = genome.edges.u[:10], genome.edges.v[:10]
    assert np.allclose(m.predict_proba(genome.marks, u, v), m2.predict_proba(genome.marks, u, v))
