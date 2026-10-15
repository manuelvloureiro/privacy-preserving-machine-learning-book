"""Check the worked solutions to the exercises of the adaptive-composition chapter.

    uv run python code/part04-privacy-accounting/solutions_adaptive_composition.py

Prints every number quoted in book-exercises/part04-privacy-accounting/08-adaptive-composition.md (seed 0; nothing here is
random, the seed is accepted for a uniform interface). Makes no figures; --out is accepted and
ignored. Conventions as in adaptive_composition.py.
"""

import argparse
import math
from pathlib import Path

import numpy as np
from adaptive_composition import (
    ORDERS,
    best_dp,
    brownian_line_crossing,
    gaussian_individual_rdp,
    normal_cdf,
    rdp_to_dp,
    rr_keep_probability,
    rr_rdp,
    ville_odometer,
    whitehouse_filter_epsilon,
    zcdp_budget_for,
    zcdp_to_dp,
)

# Figures are build output: _build/figures/<part>/, which the build copies into each book.
FIGURES = Path(__file__).resolve().parents[2] / "_build" / "figures" / Path(__file__).parent.name

# --- Exercise 1: stopping after one step ------------------------------------------------------


def stop_or_continue_rows(alpha: float, eps: float) -> list[tuple[str, float, float, float]]:
    """Randomized response with eps on one bit; a second identical step only if the first output is 1.

    Returns (path, P, Q, RDP spent along the path)."""
    p = rr_keep_probability(eps)
    rho = rr_rdp(alpha, eps)
    return [
        ("(1,1)", p * p, (1 - p) * (1 - p), 2 * rho),
        ("(1,0)", p * (1 - p), (1 - p) * p, 2 * rho),
        ("(0,stop)", 1 - p, p, rho),
    ]


# --- Exercise 3: crossing a line ----------------------------------------------------------------


def worst_fixed_time_probability(delta: float) -> float:
    """max over fixed s of P(B_s > a + b s) with ab = ln(1/δ)/2: Φ(-sqrt(2 ln(1/δ)))."""
    return normal_cdf(-math.sqrt(2 * math.log(1 / delta)))


# --- Exercise 5: the price of choosing the order late -----------------------------------------

FEW_ORDERS: list[float] = [1.5, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0, 16.0, 32.0, 64.0]


def print_table(rng: np.random.Generator) -> None:
    del rng  # nothing is random here

    print("[Exercise 1: stop after a 0, alpha=2, eps=ln 3]")
    alpha, eps = 2.0, math.log(3)
    rows = stop_or_continue_rows(alpha, eps)
    total = 0.0
    for name, p, q, spent in rows:
        n_t = math.exp((alpha - 1) * (math.log(p / q) - spent))
        total += p * n_t
        print(f"  {name}: P={p:.4f}  Q={q:.4f}  L={math.log(p / q):+.4f}  spent={spent:.4f}  N_T={n_t:.4f}")
    d2 = math.log(sum(p**alpha * q ** (1 - alpha) for _, p, q, _ in rows)) / (alpha - 1)
    print(
        f"  E_P[N_T]={total:.4f}  joint D_2={d2:.4f}  budget B={2 * rr_rdp(alpha, eps):.4f}  "
        f"one step={rr_rdp(alpha, eps):.4f}"
    )

    print("[Exercise 2: the zCDP route to the DP filter]")
    eps_g, delta = 1.0, 1e-6
    rho_star = zcdp_budget_for(eps_g, delta)
    print(f"  ln(1/delta)={math.log(1 / delta):.4f}  rho*={rho_star:.6f}  check={zcdp_to_dp(rho_star, delta):.4f}")
    eps0 = 0.01
    k = math.floor(2 * rho_star / eps0**2)
    print(
        f"  eps0={eps0:g}: k_max=floor(2 rho*/eps0^2)={k}  Whitehouse statistic at k={k}: "
        f"{whitehouse_filter_epsilon([eps0] * k, delta):.4f}  at k={k + 1}: "
        f"{whitehouse_filter_epsilon([eps0] * (k + 1), delta):.4f}"
    )
    a_star = 1 + math.sqrt(math.log(1 / delta) / rho_star)
    print(f"  optimal order 1 + sqrt(ln(1/delta)/rho*)={a_star:.2f}")

    print("[Exercise 3: Brownian crossing of the fixed-order Ville boundary]")
    for d in (0.05, 1e-5):
        a, b = math.log(1 / d) / (2 - 1), (2 - 1) / 2
        print(
            f"  delta={d:g}: infinite horizon={math.exp(-2 * a * b):.6g}  horizon s=1000: "
            f"{brownian_line_crossing(a, b, 1000.0):.6g}  worst fixed time: {worst_fixed_time_probability(d):.3g}  "
            f"ratio delta/worst={d / worst_fixed_time_probability(d):.1f}"
        )

    print("[Exercise 4: a record whose clipped gradient is half the clip, sigma=10, k0=100]")
    sigma, k0, d = 10.0, 100, 1e-5
    worst, alpha_w = best_dp(lambda a: a * k0 / (2 * sigma**2), d)
    budget = gaussian_individual_rdp(alpha_w, [1.0] * k0, sigma, 1.0)
    half = gaussian_individual_rdp(alpha_w, [0.5] * k0, sigma, 1.0)
    eps_best, a_best = best_dp(lambda a: gaussian_individual_rdp(a, [0.5] * k0, sigma, 1.0), d)
    per_step = gaussian_individual_rdp(alpha_w, [0.5], sigma, 1.0)
    print(f"  worst case eps={worst:.3f} at alpha*={alpha_w:g}  B={budget:.3f}")
    print(
        f"  half-norm record: spend at alpha*={half:.4f}  eps at alpha*={rdp_to_dp(half, alpha_w, d):.3f}  "
        f"eps optimized={eps_best:.3f} at alpha={a_best:g}"
    )
    print(f"  zCDP slope of the half-norm record rho={k0 / (8 * sigma**2):.3f}")
    print(
        f"  per-step spend={per_step:.6f}  steps until the filter excludes it: {math.floor(budget / per_step + 1e-9)}"
    )

    print("[Exercise 5: rho=1, delta=1e-5]")
    rho, d = 1.0, 1e-5

    def curve(a: float) -> float:
        return a * rho

    fixed, a_fixed = best_dp(curve, d)
    print(f"  fixed time over ORDERS: {fixed:.3f} at alpha={a_fixed:g}  continuous: {zcdp_to_dp(rho, d):.3f}")
    print(
        f"  Ville, |orders|={len(ORDERS)}: {ville_odometer(curve, d):.3f}  "
        f"continuous with ln(|orders|/delta): {zcdp_to_dp(rho, d / len(ORDERS)):.3f}"
    )
    print(f"  Ville, {len(FEW_ORDERS)} orders: {ville_odometer(curve, d, FEW_ORDERS):.3f}")
    a_small = best_dp(lambda a: a * 0.25, d)[1]
    print(f"  Ville at the single order {a_small:g} (optimal for rho=0.25): {ville_odometer(curve, d, [a_small]):.3f}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=FIGURES, help="accepted for a uniform interface; no figures")
    args = parser.parse_args()
    seed: int = args.seed
    print_table(np.random.default_rng(seed))


if __name__ == "__main__":
    main()
