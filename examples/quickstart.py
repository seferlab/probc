"""Minimal ProbC example on a synthetic genome.

    python examples/quickstart.py
"""
import numpy as np

from probc import ProbC, build_full_pairset, make_labeled_pairs
from probc.cv import kfold_auc
from probc.synthetic import simulate_genome

# 1) data: mark counts per restriction site (C) and observed interactions (edges)
g = simulate_genome(n_chroms=2, sites_per_chrom=1500, seed=0)
mm, edges, universe = g.marks, g.edges, g.universe
print(f"{mm.n_sites} sites, {mm.n_marks} marks, {len(edges)} interacting pairs")

# 2) exact likelihood over *all* pairs (non-interactions handled implicitly)
full = build_full_pairset(mm, edges, universe)
model = ProbC(lambda_nuc=1e-3, lambda_l1=1e-3).fit(mm, full)
print(f"ADMM converged={model.converged_} after {model.n_iter_} iterations")
print("strongest mark-pair interaction probabilities:")
print(model.interaction_table().head(8).to_string(index=False))
print("correlation with the planted parameters:",
      round(np.corrcoef(model.theta_.ravel(), g.theta_true.ravel())[0, 1], 4))

# 3) predict interactions for held-out pairs (5-fold CV ROC AUC)
pairs = make_labeled_pairs(mm, edges, universe, neg_ratio=1.0, rng=np.random.default_rng(0), reweight=True)
print("5-fold CV ROC AUC:", round(kfold_auc(ProbC(lambda_nuc=1e-3, lambda_l1=1e-3), mm, pairs), 3))

# 4) forbid interactions between different marks (paper: diagonal-only analysis)
diag = ProbC(lambda_nuc=0, lambda_l1=1e-3, support="diagonal").fit(mm, full)
order = np.argsort(-np.diag(diag.theta_))
print("marks ranked by self-interaction:", [mm.marks[i] for i in order[:5]])
