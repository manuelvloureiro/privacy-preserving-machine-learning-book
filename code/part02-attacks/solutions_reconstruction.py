"""Check the worked solutions to the exercises of the reconstruction chapter.

    uv run python code/part02-attacks/solutions_reconstruction.py

Prints every number quoted in book-exercises/part02-attacks/09-reconstruction.md (seed 0). Makes no figures; --out is
accepted and ignored.

Conventions as in reconstruction.py: d ∈ {0,1}^n is the secret, a query is a 0/1 row q, and the
curator answers ⟨q, d⟩ plus an error.
"""

import argparse
import math
from pathlib import Path

import numpy as np
from reconstruction import (
    BLOCK,
    GROUPS,
    IntArray,
    all_subsets,
    consistent_blocks,
    escape_threshold,
    exponential_attack_worst_case,
    fraction_agreeing,
    hamming,
    hoeffding_max_error,
    information_min_queries,
    least_squares_decode,
    lp_queries_needed,
    ls_predicted_fraction,
    ls_recovery,
    paley_zygmund_escape,
    pattern_summary,
    publish,
    queries_for_recovery,
    rademacher_fourth_moment,
    rademacher_fourth_moment_formula,
    random_bits,
    random_queries,
    rounded_answer_values,
    rounding_curator_attack,
    sampling_curator,
    withhold,
)

# Figures are build output: _build/figures/<part>/, which the build copies into each book.
FIGURES = Path(__file__).resolve().parents[2] / "_build" / "figures" / Path(__file__).parent.name

# --- Exercise 1: a rounding curator ----------------------------------------------------------


def rounded(x: IntArray, width: int) -> IntArray:
    """Round each entry to the nearest multiple of `width` (halves round up)."""
    return (np.floor(x / width + 0.5) * width).astype(np.int64)


def separating_subsets(d: IntArray, c: IntArray, width: int) -> int:
    """How many of the 2^n subsets get different published (rounded) answers under d and under c."""
    subsets = all_subsets(len(d))
    return hamming(rounded(subsets @ d, width), rounded(subsets @ c, width))


def min_separating_over_flips(d: IntArray, width: int) -> int:
    """The fewest separating subsets between d and any database that differs from d in one bit."""
    counts: list[int] = []
    for i in range(len(d)):
        c = d.copy()
        c[i] = 1 - c[i]
        counts.append(separating_subsets(d, c, width))
    return min(counts)


# --- Exercise 2: another Paley–Zygmund threshold ---------------------------------------------


def lp_error_tolerance_theta(n: int, beta: float, theta: float) -> float:
    """Largest E with 2E + 1/2 ≤ √θ·√(βn)/6."""
    return escape_threshold(n, beta, theta) / 2 - 0.25


# --- Exercise 4: the inverse of I + b 11ᵀ -----------------------------------------------------


def sherman_morrison_error(n: int, b: float) -> float:
    """max |(I + b11ᵀ)⁻¹ − (I − b/(1 + nb) 11ᵀ)| over the entries."""
    ones = np.ones((n, n))
    direct = np.linalg.inv(np.eye(n) + b * ones)
    formula = np.eye(n) - b / (1 + n * b) * ones
    return float(np.max(np.abs(direct - formula)))


def print_table(rng: np.random.Generator) -> None:
    print("== Exercise 1: rounding curator ==")
    d10 = random_bits(rng, 10)  # the same database as in reconstruction.py with seed 0
    print(f"  d={d10.tolist()}")
    for width in (2, 4):
        count, dist = rounding_curator_attack(d10, width)
        print(
            f"  width {width}: consistent candidates={count}  max distance={dist}  "
            f"fewest separating subsets over one-bit flips={min_separating_over_flips(d10, width)}"
        )
    count, dist = exponential_attack_worst_case(d10, 2.0)
    print(f"  worst-case curator with E=2: consistent candidates={count}  max distance={dist}")

    print("== Exercise 2: theta = 1/9 ==")
    n, beta = 1000, 0.1
    for theta in (0.25, 1 / 9):
        esc = paley_zygmund_escape(theta)
        print(
            f"  theta={theta:.4f}: escape={esc:.4f}  threshold={escape_threshold(n, beta, theta):.4f}  "
            f"tolerance E <= {lp_error_tolerance_theta(n, beta, theta):.4f}  "
            f"queries m >= {lp_queries_needed(n, 0.01, esc):.0f}"
        )

    print("== Exercise 3: fourth moment ==")
    for a in ([1.0, 1.0, 1.0], [1.0, 2.0], [0.5, 0.5, 0.5, 0.5]):
        print(
            f"  a={a}: enumeration={rademacher_fourth_moment(a):.4f}  formula={rademacher_fourth_moment_formula(a):.4f}"
        )

    print("== Exercise 4: inverse of I + b 11^T ==")
    print(f"  n=5 b=1: max entry error={sherman_morrison_error(5, 1.0):.1e}")
    print(f"  diagonal factor n/(n+1) at n=500: {500 / 501:.4f}")

    print("== Exercise 5: queries to beat noise ==")
    for c in (2.0, 5.0):
        print(f"  sigma={c:g} sqrt(n): m/n={queries_for_recovery(0.9, c):.1f}")
    n = 100
    m = int(round(queries_for_recovery(0.9, 2.0) * n))
    sim = ls_recovery(rng, n, [m], [2.0 * math.sqrt(n)], reps=4)[0][0]
    print(f"  n={n}, m={m}, sigma=2 sqrt(n): simulated={sim:.4f}  predicted={ls_predicted_fraction(n, m, 20.0):.4f}")

    print("== Exercise 6: information bound ==")
    v = rounded_answer_values(1000, 10)
    print(
        f"  n=1000 beta=0.01 width 10: values={v}  log2={math.log2(v):.4f}  "
        f"m >= {information_min_queries(1000, 0.01, v):.1f}"
    )

    print("== Exercise 7: sampling curator with keep = 0.9 ==")
    n, m, keep = 500, 2000, 0.9
    d = random_bits(rng, n)
    a_matrix = random_queries(rng, m, n)
    answers, kept = sampling_curator(rng, a_matrix, d, keep)
    err = float(np.max(np.abs(answers - a_matrix @ d)))
    recovered = fraction_agreeing(least_squares_decode(a_matrix, answers), d)
    print(
        f"  Hoeffding bound={hoeffding_max_error(n, m, 0.05, keep):.2f}  observed max error={err:.2f}  "
        f"kept={int(kept.sum())}  recovered={recovered:.4f}  ceiling={keep + (1 - keep) / 2:.2f}"
    )

    print("== Exercise 8: redundant statistics ==")
    table = publish(BLOCK)
    total, female, male = table["total"], table["female"], table["male"]
    assert total.total is not None and female.total is not None and male.total is not None
    print(
        f"  sums: total={total.total}  female={female.total}  male={male.total}  "
        f"female + male={female.total + male.total}"
    )
    for group in GROUPS:
        found = consistent_blocks(withhold(table, group, "mean"))
        patterns, share = pattern_summary(found, BLOCK)
        print(f"  withhold {group} mean: {len(found)} blocks, {patterns} patterns, true pattern share {share:.2f}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=FIGURES, help="ignored: this script makes no figures")
    args = parser.parse_args()
    seed: int = args.seed
    print_table(np.random.default_rng(seed))


if __name__ == "__main__":
    main()
