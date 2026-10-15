"""Compute the numbers quoted in the solutions to the distributions exercises.

    uv run python code/part01-prerequisites/solutions_distributions.py
    uv run python code/part01-prerequisites/solutions_distributions.py --seed 1

Prints every number in book-exercises/part01-prerequisites/01-distributions.md (they match exactly with the default seed).
No figures: --out is accepted for a uniform interface and ignored.
"""

import argparse
import math
from pathlib import Path

import numpy as np
from distributions import binomial_pmf, poisson_pmf, sample_moments
from numpy.typing import NDArray

# Figures are build output: _build/figures/<part>/, which the build copies into each book.
FIGURES = Path(__file__).resolve().parents[2] / "_build" / "figures" / Path(__file__).parent.name

# --- Exercise 1: binomial variance from the PMF ---------------------------------------------


def binomial_factorial_moment(n: int, p: float) -> float:
    """E[S(S - 1)] for S ~ Binomial(n, p), summed term by term over the PMF."""
    return sum(k * (k - 1) * binomial_pmf(k, n, p) for k in range(n + 1))


def binomial_variance_from_pmf(n: int, p: float) -> float:
    """Var S = E[S(S - 1)] + E S - (E S)², with E S also summed over the PMF."""
    mean = sum(k * binomial_pmf(k, n, p) for k in range(n + 1))
    return binomial_factorial_moment(n, p) + mean - mean**2


# --- Exercise 2: Poisson thinning ------------------------------------------------------------


def thinned_poisson(rng: np.random.Generator, mu: float, q: float, m: int) -> NDArray[np.float64]:
    """m draws of K: draw N ~ Poisson(mu), then keep each of the N events with probability q."""
    counts = rng.poisson(mu, m)
    return rng.binomial(counts, q).astype(np.float64)


# --- Exercise 3: quality of the Poisson approximation -----------------------------------------


def binomial_poisson_tv(n: int, mu: float) -> float:
    """Total variation distance between Binomial(n, mu / n) and Poisson(mu).

    The binomial lives on 0..n; the Poisson mass above n counts in full, once.
    """
    p = mu / n
    inside = sum(abs(binomial_pmf(k, n, p) - poisson_pmf(k, mu)) for k in range(n + 1))
    poisson_above_n = 1 - sum(poisson_pmf(k, mu) for k in range(n + 1))
    return 0.5 * (inside + max(poisson_above_n, 0.0))


TV_MU, TV_NS = 5.0, [10, 100, 1000]

# --- Exercise 6: maximum of normals ----------------------------------------------------------


def max_normal_bound(k: int) -> float:
    """The bound √(2 ln k) on E max of k standard normals."""
    return math.sqrt(2 * math.log(k))


def mean_max_independent_normals(rng: np.random.Generator, k: int, m: int) -> float:
    """Sample mean of max(Z_1, ..., Z_k) over m draws of k independent N(0, 1)."""
    return float(np.mean(rng.normal(0.0, 1.0, size=(m, k)).max(axis=1)))


MAX_KS = [2, 10, 100, 1000]

# --- Exercise 7: random vectors are nearly orthogonal ------------------------------------------


def gaussian_cosines(rng: np.random.Generator, d: int, m: int) -> NDArray[np.float64]:
    """m draws of <X, Y> / (‖X‖ ‖Y‖) for independent X, Y ~ N(0, I_d)."""
    x = rng.normal(0.0, 1.0, size=(m, d))
    y = rng.normal(0.0, 1.0, size=(m, d))
    return np.sum(x * y, axis=1) / (np.linalg.norm(x, axis=1) * np.linalg.norm(y, axis=1))


COS_DS = [10, 1000]

# --- Table ------------------------------------------------------------------------------------


def print_table(rng: np.random.Generator) -> None:
    print("[Ex 1] Var S from the PMF against np(1 - p)")
    for n, p in [(20, 0.3), (1000, 0.05)]:
        print(
            f"  n={n:<5} p={p:<5} E[S(S-1)] {binomial_factorial_moment(n, p):.4f} "
            f"(n(n-1)p² = {n * (n - 1) * p**2:.4f})   var {binomial_variance_from_pmf(n, p):.4f} "
            f"(np(1-p) = {n * p * (1 - p):.4f})"
        )

    mu, q = 4.0, 0.3
    mean, var = sample_moments(thinned_poisson(rng, mu, q, 1_000_000))
    print(
        f"[Ex 2] Poisson({mu:g}) thinned with q = {q:g}, 1,000,000 draws: "
        f"mean {mean:.3f}   var {var:.3f}   (q mu = {q * mu:.3f})"
    )

    print(f"[Ex 3] TV(Binomial(n, {TV_MU:g}/n), Poisson({TV_MU:g}))")
    for n in TV_NS:
        tv = binomial_poisson_tv(n, TV_MU)
        print(f"  n={n:<6} TV {tv:.6f}   n * TV {n * tv:.4f}")

    print("[Ex 6] E max of k independent N(0, 1), 10,000 draws, against √(2 ln k)")
    for k in MAX_KS:
        simulated, bound = mean_max_independent_normals(rng, k, 10_000), max_normal_bound(k)
        print(f"  k={k:<5} simulated {simulated:.3f}   bound {bound:.3f}   ratio {simulated / bound:.2f}")
    print(f"  k=2 exact 1/√π = {1 / math.sqrt(math.pi):.3f}")

    print("[Ex 7] cosine of independent X, Y ~ N(0, I_d), 10,000 draws")
    for d in COS_DS:
        c = gaussian_cosines(rng, d, 10_000)
        std = float(np.std(c))
        print(
            f"  d={d:<5} mean {float(np.mean(c)):.4f}   std {std:.4f}   "
            f"1/√d {1 / math.sqrt(d):.4f}   √d * std {math.sqrt(d) * std:.3f}   "
            f"max |cos| {(top := float(np.max(np.abs(c)))):.3f} "
            f"(within {math.degrees(math.asin(top)):.1f}° of a right angle)"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=FIGURES, help="unused: this script draws no figures")
    args = parser.parse_args()
    seed: int = args.seed
    print_table(np.random.default_rng(seed))


if __name__ == "__main__":
    main()
