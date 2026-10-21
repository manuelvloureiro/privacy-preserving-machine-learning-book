"""Check the worked solutions to the exercises of the discrete-mechanisms chapter.

    uv run python code/part03-dp-foundations/solutions_discrete_mechanisms.py

Prints every number quoted in book-exercises/part03-dp-foundations/08-discrete-mechanisms.md (seed 0). Makes no figures;
--out is accepted and ignored.
"""

import argparse
import math
from fractions import Fraction
from pathlib import Path

import numpy as np
from discrete_mechanisms import (
    bernoulli_exp,
    bernoulli_exp_cost,
    discrete_gaussian_pmf,
    discrete_gaussian_variance,
    discrete_laplace_pmf,
    discrete_laplace_variance,
    float_laplace_pmf,
    floor_sqrt,
    gaussian_acceptance,
    mass_outside,
    seeded_source,
)

# Figures are build output: _build/figures/<part>/, which the build copies into each book.
FIGURES = Path(__file__).resolve().parents[2] / "_build" / "figures" / Path(__file__).parent.name

# --- Exercise 2: discrete Laplace as a difference of geometrics ----------------------------------


def geometric_difference_pmf(z: int, r: float, terms: int = 2000) -> float:
    """Pr[G1 - G2 = z] for independent G with Pr[G = g] = (1 - r) r^g, by summing over G2."""
    return sum((1 - r) * r ** (g + z) * (1 - r) * r**g for g in range(max(0, -z), terms))


# --- Exercise 3: variance of the discrete Laplace --------------------------------------------------


def discrete_laplace_variance_sinh(scale: float) -> float:
    """2r/(1 - r)² = 1/(2 sinh²(1/(2 scale)))."""
    return 1 / (2 * math.sinh(1 / (2 * scale)) ** 2)


# --- Exercise 4: tails of the geometric mechanism --------------------------------------------------


def discrete_laplace_tail(k: int, scale: float) -> float:
    """Pr[|Z| >= k] = 2 r^k / (1 + r) for k >= 1."""
    r = math.exp(-1 / scale)
    return 2 * r**k / (1 + r)


def smallest_radius(scale: float, beta: float) -> int:
    """The smallest k with Pr[|Z| >= k + 1] <= beta, so that |Z| <= k with probability >= 1 - beta."""
    k = 0
    while discrete_laplace_tail(k + 1, scale) > beta:
        k += 1
    return k


# --- Exercise 5: acceptance probability of Algorithm 3 ---------------------------------------------


def acceptance_closed_form(sigma2: Fraction) -> float:
    """(1 - r)/(1 + r) e^(-σ²/(2t²)) Σ_y e^(-y²/(2σ²)) with t = floor(σ) + 1 and r = e^(-1/t)."""
    t = floor_sqrt(sigma2) + 1
    s2 = float(sigma2)
    r = math.exp(-1 / t)
    theta = sum(math.exp(-(y * y) / (2 * s2)) for y in range(-int(40 * math.sqrt(s2)) - 5, int(40 * math.sqrt(s2)) + 6))
    return (1 - r) / (1 + r) * math.exp(-s2 / (2 * t * t)) * theta


# --- Exercise 6: sums of discrete Gaussians -----------------------------------------------------------


def convolve(p: dict[int, float], q: dict[int, float]) -> dict[int, float]:
    out: dict[int, float] = {}
    for x, wx in p.items():
        for y, wy in q.items():
            out[x + y] = out.get(x + y, 0.0) + wx * wy
    return out


# --- Table ----------------------------------------------------------------------------------------------


def print_table(rng: np.random.Generator) -> None:
    src = seeded_source(rng)

    print("[Exercise 1: cost of Algorithm 1]")
    for gamma in (Fraction(1, 2), Fraction(1), Fraction(5, 2)):
        n = 20_000
        src.draws = 0
        for _ in range(n):
            bernoulli_exp(gamma, src)
        print(f"  gamma={gamma}: expected draws={bernoulli_exp_cost(float(gamma)):.4f}  simulated={src.draws / n:.4f}")
    print(f"  just below 1: {bernoulli_exp_cost(1 - 1e-12):.4f}  at 1: {bernoulli_exp_cost(1.0):.4f}")

    print("[Exercise 2: difference of two geometrics]")
    r = math.exp(-1)
    gaps = max(abs(geometric_difference_pmf(z, r) - discrete_laplace_pmf(z, 1.0)) for z in range(-10, 11))
    print(
        f"  r=e^-1: Pr[G1-G2=0]={geometric_difference_pmf(0, r):.4f}  (1-r)/(1+r)={(1 - r) / (1 + r):.4f}  "
        f"largest gap over |z|<=10={gaps:.1e}"
    )

    print("[Exercise 3: variance of the discrete Laplace]")
    for scale in (0.5, 1.0, 10.0, 100.0):
        var = discrete_laplace_variance(scale)
        print(
            f"  scale={scale:g}: 2r/(1-r)^2={var:.4f}  sinh form={discrete_laplace_variance_sinh(scale):.4f}  "
            f"2 scale^2={2 * scale**2:.4f}  gap={2 * scale**2 - var:.4f}"
        )

    print("[Exercise 4: tails of the geometric mechanism, eps = 1]")
    print(
        f"  Pr[|Z|>=3]={discrete_laplace_tail(3, 1.0):.4f}  continuous Pr[|X|>=3]={math.exp(-3):.4f}  "
        f"Pr[|Z|>=4]={discrete_laplace_tail(4, 1.0):.4f}"
    )
    print(
        f"  95% radius: eps=1: {smallest_radius(1.0, 0.05)}  eps=0.1: {smallest_radius(10.0, 0.05)}  "
        f"continuous eps=0.1: {10 * math.log(1 / 0.05):.2f}"
    )

    print("[Exercise 5: acceptance probability of Algorithm 3]")
    for sigma2 in (Fraction(1, 4), Fraction(1), Fraction(4), Fraction(100), Fraction(10_000)):
        print(
            f"  sigma^2={sigma2}: closed form={acceptance_closed_form(sigma2):.4f}  "
            f"by summation={gaussian_acceptance(sigma2):.4f}"
        )
    print(f"  limit sqrt(pi/2) e^-1/2={math.sqrt(math.pi / 2) * math.exp(-0.5):.4f}")

    print("[Exercise 6: sum of two discrete Gaussians]")
    for s2 in (0.25, 1.0, 4.0):
        sigma = math.sqrt(s2)
        p = discrete_gaussian_pmf(sigma)
        total = convolve(p, p)
        target = discrete_gaussian_pmf(math.sqrt(2 * s2))
        tv = 0.5 * sum(abs(total.get(x, 0.0) - target.get(x, 0.0)) for x in set(total) | set(target))
        var = sum(x * x * w for x, w in total.items())
        print(
            f"  sigma^2={s2:g} each: Pr[sum=0]={total[0]:.4f}  N_Z(0,{2 * s2:g}) at 0={target[0]:.4f}  "
            f"var of sum={var:.4f}  var of N_Z(0,{2 * s2:g})={discrete_gaussian_variance(math.sqrt(2 * s2)):.4f}  "
            f"TV={tv:.2e}"
        )

    print("[Exercise 7: the support argument]")
    m0, m1 = float_laplace_pmf(0.0, 1.0, 12), float_laplace_pmf(1.0, 1.0, 12)
    print(f"  p=12: P(S)={mass_outside(m0, m1):.4f}  reverse={mass_outside(m1, m0):.4f}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=FIGURES, help="accepted for a uniform interface; no figures")
    args = parser.parse_args()
    seed: int = args.seed
    print_table(np.random.default_rng(seed))


if __name__ == "__main__":
    main()
