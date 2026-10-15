"""Check the worked solutions to the exercises of the private prediction chapter.

    uv run python code/part05-dp-ml/solutions_private_prediction.py

Prints every number quoted in book-exercises/part05-dp-ml/05-private-prediction.md (seed 0). Makes no
figures; --out is accepted and ignored.
"""

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")

import argparse
import math
from pathlib import Path

import numpy as np
from dp_sgd import bisect
from private_prediction import (
    confident_epsilon,
    gnmax_dd_rdp,
    gnmax_epsilon,
    gnmax_sigma,
    icl_epsilon,
    lnmax_scale,
    pure_dd_rdp,
)

# Figures are build output: _build/figures/<part>/, which the build copies into each book.
FIGURES = Path(__file__).resolve().parents[2] / "_build" / "figures" / Path(__file__).parent.name


def normal_cdf(x: float) -> float:
    return 0.5 * math.erfc(-x / math.sqrt(2))


def normal_quantile(p: float) -> float:
    return bisect(lambda x: normal_cdf(x) >= p, -10.0, 10.0)


# --- Exercise 1: the Laplace version of the failure probability ---------------------------------------


def laplace_difference_tail(t: float, b: float) -> float:
    """Pr[Z1 - Z2 > t] for independent Lap(b), t ≥ 0: (2 + t/b) e^{-t/b} / 4."""
    return (2 + t / b) * math.exp(-t / b) / 4


def simulated_laplace_tail(rng: np.random.Generator, t: float, b: float, draws: int) -> float:
    z = rng.laplace(0.0, b, (draws, 2))
    return float(np.mean(z[:, 0] - z[:, 1] > t))


def gaussian_difference_tail(t: float, sigma: float) -> float:
    return normal_cdf(-t / (math.sqrt(2) * sigma))


# --- Exercise 2: the gap a reliable answer needs -----------------------------------------------------


def gap_for(prob: float, sigma: float) -> float:
    """The two-label gap g with Φ(g/(√2σ)) = prob."""
    return normal_quantile(prob) * math.sqrt(2) * sigma


# --- Exercise 3: where Gaussian noise overtakes Laplace noise ----------------------------------------


def crossover_queries(eps: float, q_max: int = 200) -> int:
    """The smallest Q at which GNMax's σ is below LNMax's standard deviation √2 b."""
    for q in range(1, q_max + 1):
        if gnmax_sigma(q, eps) < math.sqrt(2) * lnmax_scale(q, eps):
            return q
    return -1


# --- Exercise 4: the data-dependent cost for small failure probabilities -----------------------------


def small_q_slope(alpha: float, eps0: float) -> float:
    """d/dq of the pure-DP data-dependent bound at q = 0: (e^{(α-1)ε0} + (α-1)e^{ε0} - α)/(α - 1)."""
    return (math.exp((alpha - 1) * eps0) + (alpha - 1) * math.exp(eps0) - alpha) / (alpha - 1)


# --- Exercise 6: one teacher, every query -----------------------------------------------------------


def theorem6_cost(gap: float, sigma: float, alpha: float) -> float:
    return gnmax_dd_rdp(alpha, 0.5 * math.erfc(gap / (2 * sigma)), sigma)


def print_table(rng: np.random.Generator) -> None:
    print("== Exercise 1")
    exact = laplace_difference_tail(10.0, 5.0)
    sim = simulated_laplace_tail(rng, 10.0, 5.0, 1_000_000)
    print(f"gap=10, b=5: Pr[runner-up wins]={exact:.4f}  simulated {sim:.4f}")
    print(
        f"gap=10, Gaussian with the same sd (sigma={math.sqrt(2) * 5:.2f}): {gaussian_difference_tail(10.0, math.sqrt(2) * 5):.4f}"
    )
    print(
        f"gap=30, b=5: Laplace {laplace_difference_tail(30.0, 5.0):.2e}  Gaussian {gaussian_difference_tail(30.0, math.sqrt(2) * 5):.2e}"
    )

    print("== Exercise 2")
    sigma = gnmax_sigma(1000, 4.0)
    for prob in (0.9, 0.99):
        g = gap_for(prob, sigma)
        print(f"sigma={sigma:.1f}, Pr={prob:g}: gap={g:.1f}  fraction of 250 teachers agreeing={(250 + g) / 500:.3f}")

    print("== Exercise 3")
    for eps in (1.0, 4.0):
        print(f"eps={eps:g}: GNMax noise is smaller from Q={crossover_queries(eps)} answers")

    print("== Exercise 4")
    for alpha in (8.0, 32.0):
        slope = small_q_slope(alpha, 0.1)
        print(
            f"eps0=0.1, alpha={alpha:g}: slope={slope:.4f}  q=0.001: slope*q={slope * 0.001:.2e}  "
            f"bound={pure_dd_rdp(alpha, 0.001, 0.1):.2e}"
        )

    print("== Exercise 5")
    rho_check = 1000 / (2 * 150**2)
    rho_answers = 459 / 40**2
    print(
        f"check slope={rho_check:.4f}  answers slope={rho_answers:.4f}  total eps={confident_epsilon(1000, 459, 150.0, 40.0):.2f}"
    )
    print(f"plain GNMax on all 1000 queries, sigma=40: eps={gnmax_epsilon(1000, 40.0):.2f}")
    s1_equal = math.sqrt(1000 / (2 * rho_answers))
    print(f"sigma1 at which the check costs as much as the answers: {s1_equal:.1f}")

    print("== Exercise 6")
    for gap in (150, 200, 250):
        here = theorem6_cost(gap, 40.0, 8.0)
        there = theorem6_cost(gap - 2, 40.0, 8.0)
        print(f"gap={gap}: cost {here:.3g}  at gap-2: {there:.3g}  ratio {there / here:.3f}")

    print("== Exercise 7")
    s = 7.39
    print(
        f"Q=1000, sigma={s}: rate 0.04 eps={icl_epsilon(1000, 0.04, s):.2f}  no subsampling eps={gnmax_epsilon(1000, s):.1f}"
    )
    print(f"no subsampling, eps=1, Q=1000: sigma={gnmax_sigma(1000, 1.0):.1f}")
    print(f"expected number of queries that use one record: {1000 * 0.04:.0f}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=FIGURES, help="ignored: this script makes no figures")
    args = parser.parse_args()
    print_table(np.random.default_rng(args.seed))


if __name__ == "__main__":
    main()
