"""Check the worked solutions to the exercises of the f-DP and Gaussian differential privacy chapter.

    uv run python book/part04-privacy-accounting/code/solutions_gaussian_dp.py

Prints every number quoted in solutions/04-gaussian-dp.md (seed 0; nothing here is random, the
seed is accepted for a uniform interface). Makes no figures; --out is accepted and ignored.

Conventions as in gaussian_dp.py: type I error a, type II error b, α only for Rényi orders.
"""

import argparse
import math
from pathlib import Path

import numpy as np
from gaussian_dp import (
    SGD_DELTA,
    SGD_Q,
    bisect_decreasing,
    clt_mu_homogeneous,
    clt_mu_subsampled_gaussian,
    eps_delta_tradeoff,
    gaussian_tradeoff,
    gdp_delta,
    gdp_eps,
    gdp_mu_for_eps,
    laplace_tradeoff,
    normal_isf,
    piecewise_linear,
    product_pmf,
    randomized_response_pmfs,
    rr_composed_delta,
    rr_functionals,
    tradeoff_breakpoints,
)

FIGURES = Path(__file__).resolve().parent.parent / "figures"

A_GRID = [i / 2_000 for i in range(1, 2_000)]

# --- Exercise 1: the Laplace mechanism is μ-GDP -------------------------------------------------


def worst_gap(mu: float, eps: float) -> float:
    """max_a (G_μ(a) - T_Lap(a)): positive when G_μ rises above the Laplace curve somewhere."""
    return max(gaussian_tradeoff(mu, a) - laplace_tradeoff(eps, a) for a in A_GRID)


def laplace_gdp_mu(eps: float) -> float:
    """The smallest μ with G_μ <= T(Lap(0, 1/ε), Lap(1, 1/ε)) on the grid (worst_gap decreases in μ)."""
    return bisect_decreasing(lambda m: worst_gap(m, eps), 0.0, 0.0, 5.0, iters=40)


# --- Exercise 2: group privacy ---------------------------------------------------------------------


def gdp_power(mu: float, a: float) -> float:
    """h(a) = 1 - G_μ(a) = Φ(Φ⁻¹(a) + μ): the largest power at type I error a under μ-GDP."""
    return 1 - gaussian_tradeoff(mu, a)


def group_tradeoff(mu: float, k: int, a: float) -> float:
    """1 - h^{∘k}(a): the group-of-k curve of a μ-GDP mechanism (DRS Theorem 2.14)."""
    for _ in range(k):
        a = gdp_power(mu, a)
    return 1 - a


def group_delta_from_point(eps: float, delta: float, k: int) -> tuple[float, float]:
    """Part 3's group property: (ε, δ)-DP gives (kε, δ (e^{kε} - 1)/(e^ε - 1)) for groups of k."""
    return k * eps, delta * math.expm1(k * eps) / math.expm1(eps)


# --- Exercise 3: a tensor product by hand ---------------------------------------------------------


def rr_pair_profile(eps0: float, eps: float) -> float:
    """δ(ε) of f_{ε0,0} ⊗ f_{ε0,0} read off its corners: max over corners of 1 - b - e^ε a (and 0)."""
    p, q = randomized_response_pmfs(eps0)
    corners = tradeoff_breakpoints(product_pmf(p, 2), product_pmf(q, 2))
    return max(0.0, max(1 - b - math.exp(eps) * a for a, b in corners))


# --- Exercise 4: inverting the duality -------------------------------------------------------------


def max_full_batch_steps(sigma: float, eps: float, delta: float) -> tuple[float, int]:
    """μ* with δ_μ*(ε) = δ, and the largest k with sqrt(k)/σ <= μ*."""
    mu_star = gdp_mu_for_eps(eps, delta)
    return mu_star, math.floor((sigma * mu_star) ** 2)


def max_sgd_steps(q: float, sigma: float, eps: float, delta: float) -> int:
    """The largest k whose CLT μ = q sqrt(k(e^{1/σ²} - 1)) stays at most μ*."""
    mu_star = gdp_mu_for_eps(eps, delta)
    return math.floor((mu_star / q) ** 2 / math.expm1(1 / sigma**2))


def print_table(rng: np.random.Generator) -> None:
    del rng  # nothing is random here
    print("== Exercise 1: Laplace is mu-GDP ==")
    for eps in (0.5, 1.0):
        mu = laplace_gdp_mu(eps)
        closed = 2 * normal_isf(math.exp(-eps / 2) / 2)
        at = max(A_GRID, key=lambda a: gaussian_tradeoff(mu, a) - laplace_tradeoff(eps, a))
        print(
            f"[Laplace eps={eps:g}] smallest mu={mu:.4f}  closed form 2 Phi^-1(1 - e^(-eps/2)/2)={closed:.4f}  "
            f"gap at that mu={worst_gap(mu, eps):.1e} attained at a={at:.4f} (diagonal point e^(-eps/2)/2={math.exp(-eps / 2) / 2:.4f})  "
            f"GDP eps at delta=1e-5={gdp_eps(mu, SGD_DELTA):.4f}"
        )

    print("== Exercise 2: group privacy ==")
    mu = 0.5
    gap = max(abs(group_tradeoff(mu, 2, a) - gaussian_tradeoff(2 * mu, a)) for a in A_GRID)
    print(f"[mu={mu:g}] max |1 - h(h(a)) - G_2mu(a)| on a grid={gap:.1e}")
    eps_point = gdp_eps(mu, SGD_DELTA)
    print(f"[mu={mu:g} single point] eps at delta=1e-5={eps_point:.4f}")
    for k in (2, 4):
        k_eps, k_delta = group_delta_from_point(eps_point, SGD_DELTA, k)
        curve_delta = gdp_delta(k * mu, k_eps)
        print(
            f"[group k={k}] GDP mu={k * mu:g}: eps at delta=1e-5={gdp_eps(k * mu, SGD_DELTA):.4f}  "
            f"from the point: ({k_eps:.4f}, {k_delta:.2e})  GDP delta({k_eps:.4f})={curve_delta:.2e}  "
            f"ratio={k_delta / curve_delta:.1f}"
        )

    print("== Exercise 3: RR x RR ==")
    eps0 = 1.0
    p, q = randomized_response_pmfs(eps0)
    corners = tradeoff_breakpoints(product_pmf(p, 2), product_pmf(q, 2))
    gap = max(
        abs(
            piecewise_linear(corners, a)
            - max(eps_delta_tradeoff(2 * eps0, 0.0, a), eps_delta_tradeoff(0.0, math.tanh(eps0 / 2), a))
        )
        for a in A_GRID
    )
    print(f"[eps0=1] tanh(1/2)={math.tanh(0.5):.4f}  max |RRxRR - max(f_2,0, f_0,tanh)| on a grid={gap:.1e}")
    keep = p[0]
    print(
        f"[corners] pi^2={keep**2:.4f}  (1-pi)^2={(1 - keep) ** 2:.4f}  "
        f"corners={[(round(x, 4), round(y, 4)) for x, y in corners]}"
    )
    for e in (0.0, 0.5, 1.0, 1.5, 2.0):
        print(f"[delta({e:g})] from corners={rr_pair_profile(eps0, e):.4f}  exact={rr_composed_delta(eps0, 2, e):.4f}")

    print("== Exercise 4: inverting the duality ==")
    mu_star, k = max_full_batch_steps(20.0, 1.0, SGD_DELTA)
    print(
        f"[eps=1 delta=1e-5] mu*={mu_star:.4f}  delta at mu*={gdp_delta(mu_star, 1.0):.2e}  full-batch sigma=20: k={k}"
    )
    for target in (1.0, 2.0, 4.0):
        k_sgd = max_sgd_steps(SGD_Q, 1.0, target, SGD_DELTA)
        print(
            f"[DP-SGD q=0.01 sigma=1 eps={target:g}] mu*={gdp_mu_for_eps(target, SGD_DELTA):.4f}  max k={k_sgd}  "
            f"eps at that k={gdp_eps(clt_mu_subsampled_gaussian(SGD_Q, 1.0, k_sgd), SGD_DELTA):.4f}"
        )

    print(
        f"[steps ratio eps 4 vs 2] {max_sgd_steps(SGD_Q, 1.0, 4.0, SGD_DELTA) / max_sgd_steps(SGD_Q, 1.0, 2.0, SGD_DELTA):.2f}"
    )
    print("== Exercise 5: Berry-Esseen for randomized response ==")
    mu100, _ = clt_mu_homogeneous(rr_functionals(0.1), 100)
    print(
        f"[k=100 delta(1)] exact={rr_composed_delta(0.1, 100, 1.0):.4f}  CLT={gdp_delta(mu100, 1.0):.4f}  "
        f"difference={gdp_delta(mu100, 1.0) - rr_composed_delta(0.1, 100, 1.0):.4f}"
    )
    for k, eps0 in ((100, 0.1), (10, 1 / math.sqrt(10)), (400, 0.05)):
        kl, kappa2, kappa3_bar = rr_functionals(eps0)
        mu, gamma = clt_mu_homogeneous((kl, kappa2, kappa3_bar), k)
        print(
            f"[k={k} eps0={eps0:.4f}] kl={kl:.6f}  eps0 tanh(eps0/2)={eps0 * math.tanh(eps0 / 2):.6f}  "
            f"mu={mu:.4f}  2 sqrt(k) sinh(eps0/2)={2 * math.sqrt(k) * math.sinh(eps0 / 2):.4f}  gamma={gamma:.4f}  "
            f"0.56(1+t^2)cosh(eps0/2)/sqrt(k)={0.56 * (1 + math.tanh(eps0 / 2) ** 2) * math.cosh(eps0 / 2) / math.sqrt(k):.4f}"
        )

    print("== Exercise 6: the subsampled CLT at q = 1 ==")
    for sigma in (1.0, 2.0, 5.0):
        k = 100
        clt = clt_mu_subsampled_gaussian(1.0, sigma, k)
        exact = math.sqrt(k) / sigma
        print(f"[q=1 sigma={sigma:g} k={k}] CLT mu={clt:.4f}  exact mu={exact:.4f}  ratio={clt / exact:.4f}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=FIGURES, help="ignored: this script makes no figures")
    args = parser.parse_args()
    seed: int = args.seed
    print_table(np.random.default_rng(seed))


if __name__ == "__main__":
    main()
