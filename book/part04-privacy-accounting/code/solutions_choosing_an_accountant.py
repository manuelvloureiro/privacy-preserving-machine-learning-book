"""Check the worked solutions to the exercises of the choosing-an-accountant chapter.

    uv run python book/part04-privacy-accounting/code/solutions_choosing_an_accountant.py

Prints every number quoted in solutions/09-choosing-an-accountant.md (seed 0; nothing here is random,
the seed is accepted for a uniform interface). Makes no figures; --out is accepted and ignored.
Takes about a minute, mostly PLD compositions.

Conventions as in choosing_an_accountant.py: q is the sampling rate, σ the noise multiplier, k the
number of steps; add/remove neighbours and Poisson sampling.
"""

import argparse
import math
from collections.abc import Callable
from pathlib import Path

import numpy as np
from choosing_an_accountant import (
    DELTA_RUN,
    DIRECTIONS,
    Q_RUN,
    SIGMA_RUN,
    H,
    accountant_table,
    composed_plds,
    epsilon,
    pld_eps,
)
from gaussian_dp import gdp_mu_for_eps
from numerical_accounting import compose_pld, convolve, discretize_subsampled_gaussian_pld, pld_epsilon
from rdp_conversions import rdp_to_dp_improved
from subsampling import ORDERS, dpsgd_eps_rdp, subsampled_gaussian_rdp_curve

FIGURES = Path(__file__).resolve().parent.parent / "figures"
KS_EX = (1, 100, 1000, 10_000)


# --- Exercise 2: how many steps fit in a budget ---------------------------------------------


def max_steps(eps_of_k: Callable[[int], float], budget: float, hi: int = 20_000) -> int:
    """The largest k with eps_of_k(k) <= budget, for eps_of_k increasing in k (binary search on integers)."""
    lo = 0  # eps_of_k(0) = 0 <= budget
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if eps_of_k(mid) <= budget:
            lo = mid
        else:
            hi = mid
    return lo


def clt_max_steps(budget: float, q: float, sigma: float, delta: float) -> int:
    """CLT: μ = q√(k(e^{1/σ²} - 1)) must not exceed the μ whose ε at δ is the budget."""
    mu = gdp_mu_for_eps(budget, delta)
    return math.floor((mu / q) ** 2 / math.expm1(1 / sigma**2))


# --- Exercise 4: two phases with different noise ---------------------------------------------


def two_phase_rdp(q: float, sigmas: tuple[float, float], steps: tuple[int, int], delta: float) -> float:
    """RDP curves add across the phases; convert the sum once."""
    c1, c2 = subsampled_gaussian_rdp_curve(q, sigmas[0]), subsampled_gaussian_rdp_curve(q, sigmas[1])
    total = [steps[0] * a + steps[1] * b for a, b in zip(c1, c2, strict=True)]
    return rdp_to_dp_improved(ORDERS, total, delta)[0]


def two_phase_pld(
    q: float, sigmas: tuple[float, float], steps: tuple[int, int], delta: float, h: float, pessimistic: bool
) -> float:
    """Convolve the two phases' composed PLDs, direction by direction, and take the worse direction."""
    eps: list[float] = []
    for d in DIRECTIONS:
        a = compose_pld(discretize_subsampled_gaussian_pld(q, sigmas[0], h, pessimistic, d), steps[0])
        b = compose_pld(discretize_subsampled_gaussian_pld(q, sigmas[1], h, pessimistic, d), steps[1])
        eps.append(pld_epsilon(convolve(a, b), delta))
    return max(eps)


def print_table() -> None:
    q, sigma, delta = Q_RUN, SIGMA_RUN, DELTA_RUN
    rows = {r.steps: r for r in accountant_table(q, sigma, delta, KS_EX)}

    print("[Exercise 1: where RDP's slack comes from]")
    for k in (1, 100, 10_000):
        r = rows[k]
        alpha = dpsgd_eps_rdp(q, sigma, k, delta)[1]
        print(
            f"  k={k}: RDP improved={r.rdp_improved:.4f} (alpha={alpha:g})  RDP optimal={r.rdp_optimal:.4f}  "
            f"PLD upper={r.pld_upper:.4f}  conversion slack={r.rdp_improved - r.rdp_optimal:.4f}  "
            f"curve slack={r.rdp_optimal - r.pld_upper:.4f}"
        )
    k1_term = math.log(1 / delta) / (dpsgd_eps_rdp(q, sigma, 1, delta)[1] - 1)
    print(f"  k=1: ln(1/delta)/(alpha-1) at the best order = {k1_term:.4f}")

    print("[Exercise 2: steps that fit in eps=3 at delta=1e-5]")
    budget = 3.0
    k_rdp = max_steps(lambda k: dpsgd_eps_rdp(q, sigma, k, delta)[0], budget)
    k_pld = max_steps(lambda k: epsilon("pld-upper", q, sigma, k, delta), budget)
    k_clt = clt_max_steps(budget, q, sigma, delta)
    print(
        f"  RDP improved: k={k_rdp} (eps={dpsgd_eps_rdp(q, sigma, k_rdp, delta)[0]:.4f}, "
        f"at k+1 {dpsgd_eps_rdp(q, sigma, k_rdp + 1, delta)[0]:.4f})"
    )
    print(
        f"  PLD upper: k={k_pld} (eps={epsilon('pld-upper', q, sigma, k_pld, delta):.4f}, "
        f"at k+1 {epsilon('pld-upper', q, sigma, k_pld + 1, delta):.4f})"
    )
    print(f"  CLT: k={k_clt}  PLD lower at that k={epsilon('pld-lower', q, sigma, k_clt, delta):.4f}")
    print(
        f"  epochs: RDP {k_rdp * q:.2f}  PLD {k_pld * q:.2f}  extra training from PLD {100 * (k_pld / k_rdp - 1):.0f}%"
    )

    print("[Exercise 3: reporting at a smaller delta]")
    upper = composed_plds(q, sigma, KS_EX, H, True)[10_000]
    for d in (1e-5, 1e-6, 1e-8):
        print(
            f"  delta={d:g}: PLD upper={pld_eps(upper, d):.4f}  RDP improved={dpsgd_eps_rdp(q, sigma, 10_000, d)[0]:.4f}"
        )

    print("[Exercise 4: 5000 steps at sigma=1, then 5000 at sigma=2]")
    sig, st = (1.0, 2.0), (5000, 5000)
    rdp2 = two_phase_rdp(q, sig, st, delta)
    up2 = two_phase_pld(q, sig, st, delta, H, True)
    dn2 = two_phase_pld(q, sig, st, delta, H, False)
    first = epsilon("pld-upper", q, 1.0, 5000, delta)
    second = epsilon("pld-upper", q, 2.0, 5000, delta)
    print(f"  RDP improved={rdp2:.4f}  PLD upper={up2:.4f}  lower={dn2:.4f}")
    print(f"  phases alone (PLD upper): sigma=1 {first:.4f}  sigma=2 {second:.4f}  sum {first + second:.4f}")

    print("[Exercise 5: whose number is it? k=1000]")
    r = rows[1000]
    print(
        f"  CLT={r.clt:.4f}  PLD lower={r.pld_lower:.4f}  PLD upper={r.pld_upper:.4f}  "
        f"RDP improved={r.rdp_improved:.4f}  RDP Mironov={r.rdp_mironov:.4f}  advanced={r.advanced:.2f}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=FIGURES, help="ignored: this script makes no figures")
    args = parser.parse_args()
    _ = np.random.default_rng(args.seed)
    print_table()


if __name__ == "__main__":
    main()
