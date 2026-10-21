"""Check the worked solutions to the exercises of the linkage and re-identification chapter.

    uv run python code/part02-attacks/solutions_linkage.py

Prints every number quoted in book-exercises/part02-attacks/07-linkage-reidentification.md (seed 0). Makes no
figures; --out is accepted and ignored.
"""

import argparse
import math
from pathlib import Path

import numpy as np
from linkage import (
    CITY_KM2,
    GRID_KM2,
    MINUTES_PER_YEAR,
    TAXI_TRIPS,
    WORLD,
    bits_for_unique_fraction,
    cell_counts,
    correct_match_poisson,
    correct_match_probability,
    linkage_success,
    mcdiarmid_halfwidth,
    uniform_unique_fraction,
)

# Figures are build output: _build/figures/<part>/, which the build copies into each book.
FIGURES = Path(__file__).resolve().parents[2] / "_build" / "figures" / Path(__file__).parent.name

# --- Exercise 3: collisions and the birthday bound -------------------------------------------


def expected_collision_pairs(n: int, cells: int) -> float:
    """C(n, 2) Σ p² for uniform cells."""
    return n * (n - 1) / 2 / cells


def collision_probability_exact(n: int, cells: int) -> float:
    """Pr[some two of n people share one of `cells` equally likely cells] = 1 - Π (1 - i/M)."""
    return 1 - math.prod(1 - i / cells for i in range(n))


# --- Exercise 4: the variance of the number of uniques ---------------------------------------


def unique_count_variance(n: int, cells: int) -> tuple[float, float]:
    """Mean and variance of U for n people in M uniform cells, from pairwise indicators."""
    mean = n * (1 - 1 / cells) ** (n - 1)
    both = (1 - 1 / cells) * (1 - 2 / cells) ** (n - 2)
    second_factorial = n * (n - 1) * both  # E[U(U - 1)]
    return mean, second_factorial + mean - mean**2


def simulated_unique_counts(rng: np.random.Generator, n: int, cells: int, trials: int) -> list[int]:
    return [int(np.count_nonzero(np.equal(cell_counts(rng.integers(0, cells, size=n)), 1))) for _ in range(trials)]


# --- Exercise 5: how many rare items ---------------------------------------------------------


def items_needed(n: int, s: float, target: float) -> int:
    """Smallest m with n s^m < target."""
    m = 0
    while n * s**m >= target:
        m += 1
    return m


def posterior_in_release(lr: float, n: int, prior: float) -> float:
    """q LR / (q LR + (1 - q) n): the posterior that the one agreeing record is the target (ε = 0)."""
    return prior * lr / (prior * lr + (1 - prior) * n)


# --- Exercise 7: eccentricity under exponential and normal scores ----------------------------


def false_match_rate(rng: np.random.Generator, n: int, phi: float, trials: int, kind: str) -> float:
    """Fraction of trials in which n independent null scores have eccentricity at least φ."""
    hits = 0
    for _ in range(trials):
        s = rng.exponential(size=n) if kind == "exponential" else rng.standard_normal(n)
        top2 = np.partition(s, -2)[-2:]
        hits += int((top2[1] - top2[0]) / float(np.std(s)) >= phi)
    return hits / trials


def print_table(rng: np.random.Generator) -> None:
    print("[Exercise 1]")
    for sizes in ([2, 2, 4], [1, 1, 1, 2, 3], [3, 3, 3]):
        print(f"  class sizes {sizes}: success={linkage_success(sizes):.3f}  1/min size={1 / min(sizes):.3f}")

    print("[Exercise 2]")
    for f in (0.5, 0.99):
        print(
            f"  f={f:g}: bits={bits_for_unique_fraction(WORLD, f):.1f}  beyond log2 n="
            f"{bits_for_unique_fraction(WORLD, f) - math.log2(WORLD):.2f}"
        )
    for n in (10, 100, 10_000, WORLD):
        print(f"  n={n:.0e}: (1 - 1/n)^(n-1)={uniform_unique_fraction(n, n):.4f}")
    print(f"  e^-1={math.exp(-1):.4f}")

    print("[Exercise 3]")
    n, cells = 30, 730
    pairs = expected_collision_pairs(n, cells)
    print(
        f"  n={n} M={cells}: expected pairs={pairs:.4f}  union bound={pairs:.3f}  "
        f"exact={collision_probability_exact(n, cells):.4f}  lower 1-e^-pairs={1 - math.exp(-pairs):.4f}"
    )
    print(
        f"  birthday only (M=365): union bound={expected_collision_pairs(n, 365):.3f}  "
        f"exact={collision_probability_exact(n, 365):.4f}"
    )

    print("[Exercise 4]")
    n = 10_000
    for cells in (10_000, 2_500):
        mean, var = unique_count_variance(n, cells)
        sims = simulated_unique_counts(rng, n, cells, 2000)
        var_independent = n * (mean / n) * (1 - mean / n)
        print(
            f"  n={n} M={cells}: E[U]/n={mean / n:.4f}  sd(U/n)={math.sqrt(var) / n:.5f}  "
            f"simulated mean={np.mean(sims) / n:.4f} sd={np.std(sims) / n:.5f}  1.96 sd={1.96 * math.sqrt(var) / n:.4f}"
        )
        print(
            f"    Var U={var:.1f}  sum of indicator variances={var_independent:.1f}  "
            f"covariance part={var - var_independent:.1f}"
        )
    print(f"  McDiarmid 95% half-width={mcdiarmid_halfwidth(n, 0.05):.4f}")

    print("[Exercise 5]")
    users, s = 480_189, 0.01
    m = items_needed(users, s, 0.01)
    print(f"  m={m}: expected false matches={users * s**m:.4f}  (m={m - 1}: {users * s ** (m - 1):.2f})")
    lr = (1 / s) ** m
    print(
        f"  LR={lr:.0e}  posterior at prior 1/2={posterior_in_release(lr, users, 0.5):.4f}  "
        f"at prior 0.01={posterior_in_release(lr, users, 0.01):.4f}"
    )

    print("[Exercise 6]")
    print(f"  lambda for kappa=1/2 at pi=0.01: {1 / 0.99:.4f}")
    for lam in (0.1, 1.0):
        n_pop = 100_000
        p = lam / (n_pop - 1)
        print(
            f"  lambda={lam:g} pi=0.01: exact binomial={correct_match_probability(p, n_pop, 0.01):.5f}  "
            f"Poisson={correct_match_poisson(lam, 0.01):.5f}  floor 1/(1+lambda)={1 / (1 + lam):.5f}"
        )

    print("[Exercise 7]")
    n, phi = 1000, 1.5
    print(f"  e^-phi={math.exp(-phi):.4f}")
    print(
        f"  simulated, n={n}: exponential={false_match_rate(rng, n, phi, 20_000, 'exponential'):.4f}  "
        f"normal={false_match_rate(rng, n, phi, 20_000, 'normal'):.4f}"
    )
    print(f"  normal approximation exp(-phi sqrt(2 ln n))={math.exp(-phi * math.sqrt(2 * math.log(n))):.4f}")

    print("== Exercise 8: hashed identifiers and taxi trips ==")
    rate = 1e7  # hashes per second, the exercise's assumption
    licences = 10**6 + 10**6  # six-digit numbers, and seven-digit numbers starting with 5
    passwords = 62**10
    print(
        f"  licences={licences} ({math.log2(licences):.1f} bits)  seconds at 1e7/s={licences / rate:.1f}  "
        f"passwords 62^10={passwords:.3g} ({math.log2(passwords):.1f} bits)  "
        f"years={passwords / rate / (365 * 24 * 3600):.0f}"
    )
    cells = MINUTES_PER_YEAR * CITY_KM2 / GRID_KM2
    print(
        f"  trips: cells={cells:.3g}  n/M={TAXI_TRIPS / cells:.5f}  "
        f"unique fraction under uniform cells={uniform_unique_fraction(TAXI_TRIPS, cells):.4f}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=FIGURES, help="accepted for a uniform interface; no figures")
    args = parser.parse_args()
    seed: int = args.seed
    print_table(np.random.default_rng(seed))


if __name__ == "__main__":
    main()
