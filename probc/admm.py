"""ADMM solver for the penalised ProbC objective (Eqs. 3, 5-17).

    argmin_{theta in box}  scale * NLL(theta) + lambda_nuc ||theta||_* + lambda_l1 ||theta||_1

is split as ``theta = Z1 = Z2`` (Eq. 5) and solved with scaled-form ADMM:

    theta^{k+1} = argmin_{theta in box} scale*NLL(theta)
                    + rho/2 ||theta - Z1^k + U1^k||^2 + rho/2 ||theta - Z2^k + U2^k||^2   (Eq. 17)
    Z1^{k+1}    = SVT_{lambda_nuc/rho}(theta^{k+1} + U1^k)                                (Eq. 13)
    Z2^{k+1}    = SoftThreshold_{lambda_l1/rho}(theta^{k+1} + U2^k)                       (Eqs. 15-16)
    U1^{k+1}    = U1^k + theta^{k+1} - Z1^{k+1}                                            (Eq. 10)
    U2^{k+1}    = U2^k + theta^{k+1} - Z2^{k+1}                                            (Eq. 11)

The smooth theta-subproblem is solved either by L-BFGS-B (the paper: "we solve
several parts of optimization by L-BFGS") or by projected gradient descent with
Armijo backtracking line search (the paper's description of the Z-step).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np
from scipy.optimize import minimize

from .prox import l1_norm, nuclear_norm, singular_value_threshold, soft_threshold, symmetrize


@dataclass
class ADMMResult:
    theta: np.ndarray          # final estimate (sparse copy Z2 projected on the box)
    theta_primal: np.ndarray   # the constrained primal iterate
    Z1: np.ndarray
    Z2: np.ndarray
    n_iter: int
    converged: bool
    history: dict = field(default_factory=dict)


class _Packer:
    """Maps between a symmetric matrix with a fixed support and a free vector."""

    def __init__(self, support: np.ndarray):
        support = np.asarray(support, dtype=bool)
        support = support | support.T
        iu = np.triu_indices(support.shape[0])
        keep = support[iu]
        self.rows = iu[0][keep]
        self.cols = iu[1][keep]
        self.offdiag = self.rows != self.cols
        self.m = support.shape[0]
        self.support = support

    def pack(self, M: np.ndarray) -> np.ndarray:
        return M[self.rows, self.cols].copy()

    def unpack(self, x: np.ndarray) -> np.ndarray:
        M = np.zeros((self.m, self.m))
        M[self.rows, self.cols] = x
        M[self.cols, self.rows] = x
        return M

    def grad_pack(self, G_sym: np.ndarray) -> np.ndarray:
        g = G_sym[self.rows, self.cols].copy()
        g[self.offdiag] *= 2.0
        return g


def _theta_step_lbfgsb(fun, theta0, V1, V2, rho, packer, lb, ub, max_iter):
    def obj(x):
        T = packer.unpack(x)
        f, G = fun(T)
        D1 = T - V1
        D2 = T - V2
        f += 0.5 * rho * (np.sum(D1 * D1) + np.sum(D2 * D2))
        G = G + rho * (D1 + D2)
        return f, packer.grad_pack(G)

    x0 = np.clip(packer.pack(theta0), lb, ub if ub is not None else np.inf)
    bounds = [(lb, ub)] * len(x0)
    res = minimize(obj, x0, jac=True, method="L-BFGS-B", bounds=bounds,
                   options={"maxiter": max_iter, "gtol": 1e-10, "ftol": 1e-13})
    return packer.unpack(res.x)


def _theta_step_pgd(fun, theta0, V1, V2, rho, packer, lb, ub, max_iter, t0=1.0, beta=0.5):
    support = packer.support

    def F(T):
        f, G = fun(T)
        D1 = T - V1
        D2 = T - V2
        f += 0.5 * rho * (np.sum(D1 * D1) + np.sum(D2 * D2))
        return f, (G + rho * (D1 + D2)) * support

    def proj(T):
        T = symmetrize(T) * support
        return np.clip(T, lb, ub if ub is not None else np.inf)

    T = proj(theta0)
    f, G = F(T)
    t = t0
    for _ in range(max_iter):
        while True:
            T_new = proj(T - t * G)
            D = T_new - T
            f_new, G_new = F(T_new)
            # Armijo-Goldstein sufficient decrease for projected gradient steps
            if f_new <= f + np.sum(G * D) + np.sum(D * D) / (2 * t) or t < 1e-14:
                break
            t *= beta
        if np.sum(D * D) < 1e-20:
            T, f, G = T_new, f_new, G_new
            break
        T, f, G = T_new, f_new, G_new
        t = min(t / beta, 1e6)  # allow the step to grow again
    return T


def admm(
    fun: Callable[[np.ndarray], tuple],
    m: int,
    theta0: np.ndarray,
    lambda_nuc: float,
    lambda_l1: float,
    lb: float = 0.0,
    ub: Optional[float] = None,
    support: Optional[np.ndarray] = None,
    rho: float = 1.0,
    adaptive_rho: bool = True,
    max_iter: int = 300,
    tol_abs: float = 1e-6,
    tol_rel: float = 1e-4,
    inner_solver: str = "lbfgsb",
    inner_max_iter: int = 50,
    verbose: int = 0,
    callback: Optional[Callable] = None,
) -> ADMMResult:
    """Run ADMM. ``fun(theta) -> (value, symmetric gradient)`` is the scaled NLL."""
    support = np.ones((m, m), bool) if support is None else np.asarray(support, bool)
    packer = _Packer(support)
    theta = np.clip(symmetrize(theta0) * packer.support, lb, ub if ub is not None else np.inf)
    Z1 = theta.copy()
    Z2 = theta.copy()
    U1 = np.zeros((m, m))
    U2 = np.zeros((m, m))
    hist = {"objective": [], "nll": [], "r_primal": [], "r_dual": [], "rho": []}
    converged = False
    step = _theta_step_lbfgsb if inner_solver == "lbfgsb" else _theta_step_pgd
    if inner_solver not in ("lbfgsb", "pgd"):
        raise ValueError("inner_solver must be 'lbfgsb' or 'pgd'")

    k = 0
    for k in range(1, max_iter + 1):
        theta = step(fun, theta, Z1 - U1, Z2 - U2, rho, packer, lb, ub, inner_max_iter)
        Z1_old, Z2_old = Z1, Z2
        Z1 = symmetrize(singular_value_threshold(theta + U1, lambda_nuc / rho))
        Z2 = soft_threshold(theta + U2, lambda_l1 / rho) * packer.support
        U1 = U1 + theta - Z1
        U2 = U2 + theta - Z2

        r = np.sqrt(np.sum((theta - Z1) ** 2) + np.sum((theta - Z2) ** 2))
        s = rho * np.sqrt(np.sum((Z1 - Z1_old + Z2 - Z2_old) ** 2))
        eps_pri = np.sqrt(2 * m * m) * tol_abs + tol_rel * max(
            np.sqrt(2) * np.linalg.norm(theta), np.sqrt(np.sum(Z1 ** 2) + np.sum(Z2 ** 2))
        )
        eps_dual = m * tol_abs + tol_rel * rho * np.linalg.norm(U1 + U2)

        f_val = fun(theta)[0]
        obj = f_val + lambda_nuc * nuclear_norm(theta) + lambda_l1 * l1_norm(theta)
        hist["objective"].append(obj)
        hist["nll"].append(f_val)
        hist["r_primal"].append(r)
        hist["r_dual"].append(s)
        hist["rho"].append(rho)
        if verbose and (k % verbose == 0 or k == 1):
            print(f"[ADMM] it={k:4d} obj={obj:.6g} nll={f_val:.6g} r={r:.2e} s={s:.2e} rho={rho:.3g}")
        if callback is not None:
            callback(k, theta, Z1, Z2)
        if r <= eps_pri and s <= eps_dual and k > 1:
            converged = True
            break
        if adaptive_rho:
            if r > 10 * s:
                rho *= 2.0
                U1 /= 2.0
                U2 /= 2.0
            elif s > 10 * r:
                rho /= 2.0
                U1 *= 2.0
                U2 *= 2.0

    # Report the split variable that carries the imposed structure exactly: Z2 has
    # exact zeros (l1), Z1 has exact low rank (nuclear). At convergence all three
    # copies agree up to the tolerance.
    if lambda_nuc > 0 and lambda_l1 <= 0:
        base = Z1
    elif lambda_l1 > 0:
        base = Z2
    else:
        base = theta
    final = np.clip(symmetrize(base) * packer.support, lb, ub if ub is not None else np.inf)
    return ADMMResult(theta=final, theta_primal=theta, Z1=Z1, Z2=Z2, n_iter=k, converged=converged, history=hist)
