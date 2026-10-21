"""Check the worked solutions to the exercises of the query-release chapter.

    uv run python code/part03-dp-foundations/solutions_query_release.py

Prints every number quoted in book-exercises/part03-dp-foundations/06-query-release.md (seed 0). Makes no figures;
--out is accepted and ignored.

Conventions as in query_release.py: a universe of N records, histograms, linear queries with values in [0, 1].
"""

import argparse
import math
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from query_release import (
    BETA,
    DELTA,
    EPS,
    best_rounds,
    halving_mistakes,
    marginal_queries,
    mistake_bound,
    n_gaussian,
    pmw_needed,
    sample_histogram,
    smalldb_size,
)

# Figures are build output: _build/figures/<part>/, which the build copies into each book.
FIGURES = Path(__file__).resolve().parents[2] / "_build" / "figures" / Path(__file__).parent.name

# --- Exercise 2: weighted majority ---------------------------------------------------------------


def weighted_majority_mistakes(rng: np.random.Generator, n_experts: int, rounds: int, best_errors: int) -> int:
    """Weighted majority (halve the weight of every wrong expert) against random experts.

    One expert is wrong on exactly `best_errors` rounds chosen at random; the others guess at random.
    """
    w = np.ones(n_experts)
    best = int(rng.integers(n_experts))
    wrong_rounds = set(rng.choice(rounds, size=best_errors, replace=False).tolist())
    mistakes = 0
    for t in range(rounds):
        truth = int(rng.integers(0, 2))
        votes = rng.integers(0, 2, n_experts)
        votes[best] = 1 - truth if t in wrong_rounds else truth
        up = float(w[votes == 1].sum())
        guess = 1 if up >= float(w.sum()) - up else 0
        mistakes += int(guess != truth)
        w[votes != truth] *= 0.5
    return mistakes


def weighted_majority_bound(n_experts: int, best_errors: int) -> float:
    """(OPT + log2 N)/log2(4/3)."""
    return (best_errors + math.log2(n_experts)) / math.log2(4 / 3)


# --- Exercise 3: the step size in the mistake bound -------------------------------------------


def mistake_bound_gamma(n_experts: int, alpha: float, gamma: float) -> float:
    """U < ln N/(γ(α - γ)) for step size γ < α."""
    return math.log(n_experts) / (gamma * (alpha - gamma))


# --- Exercise 4: the pure-DP sample size ------------------------------------------------------


def pure_records(n_experts: int, alpha: float, eps: float, log_term: float) -> float:
    """n = 108 L ln N/(α³ ε)."""
    return 108 * log_term * math.log(n_experts) / (alpha**3 * eps)


# --- Exercise 5: SmallDB's small dataset, by sampling --------------------------------------------


def sample_errors(
    rng: np.random.Generator, queries: NDArray[np.float64], p: NDArray[np.float64], m: int, trials: int
) -> NDArray[np.float64]:
    """Largest error over the queries of m records drawn from p, for each of `trials` samples."""
    out = np.empty(trials)
    truth = queries @ p
    for i in range(trials):
        counts = rng.multinomial(m, p)
        out[i] = float(np.max(np.abs(queries @ (counts / m) - truth)))
    return out


# --- Exercise 6: where private MW overtakes Gaussian noise -------------------------------------


def crossover_queries(n_experts: int, alpha: float) -> float:
    """Smallest |Q| (to 5%) at which the PMW bound needs fewer records than Gaussian noise on every answer."""
    lo, hi = 1e6, 1e12
    while hi / lo > 1.05:
        mid = math.sqrt(lo * hi)
        q = int(mid)
        if pmw_needed(n_experts, q, alpha, EPS, DELTA, BETA) < n_gaussian(q, alpha, EPS, DELTA, BETA):
            hi = mid
        else:
            lo = mid
    return hi


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=FIGURES, help="ignored: this script makes no figures")
    args = parser.parse_args()
    rng = np.random.default_rng(args.seed)

    print("[Exercise 1: halving]")
    runs = [halving_mistakes(rng, 1024, 200) for _ in range(200)]
    print(f"  1024 experts, 200 runs: largest mistakes={max(runs)}  mean={np.mean(runs):.2f}  bound=10")

    print("[Exercise 2: weighted majority]")
    print(f"  1/log2(4/3)={1 / math.log2(4 / 3):.4f}")
    wm = [weighted_majority_mistakes(rng, 1024, 200, 5) for _ in range(200)]
    print(
        f"  1024 experts, best wrong 5 times, 200 runs: largest mistakes={max(wm)}  "
        f"bound={weighted_majority_bound(1024, 5):.2f}"
    )

    print("[Exercise 3: the step size]")
    for g in (0.025, 0.0125, 0.0375):
        print(f"  N=256 alpha=0.05 gamma={g:g}: bound={mistake_bound_gamma(256, 0.05, g):.0f}")
    print(f"  check 4 ln N/alpha^2={mistake_bound(256, 0.05):.0f}")

    print("[Exercise 4: the pure-DP sample size]")
    big_n = 2**20
    n_pure = pmw_needed(big_n, 10**6, 0.05, EPS, 0.0, BETA)
    t_pure = best_rounds(big_n, 10**6, int(n_pure), EPS, BETA)
    log_term = math.log(10**6) + 2 * math.log(2 * t_pure / BETA)
    print(
        f"  N=2^20 |Q|=1e6 alpha=0.05: search n={n_pure:.3g}  best T={t_pure}  L={log_term:.2f}  "
        f"closed form={pure_records(big_n, 0.05, EPS, log_term):.3g}"
    )
    print(f"  approximate DP (chapter): {pmw_needed(big_n, 10**6, 0.05, EPS, DELTA, BETA):.3g}")

    print("[Exercise 5: a small dataset by sampling]")
    queries, _ = marginal_queries()
    p = sample_histogram(np.random.default_rng(0), 10_000)  # the chapter's running example
    m = smalldb_size(queries.shape[0], 0.1)
    one = 2 * math.exp(-m * 0.1**2 / 2)
    print(f"  Hoeffding per query={one:.2e}  union bound over {queries.shape[0]} queries={queries.shape[0] * one:.3f}")
    errs = sample_errors(rng, queries, p, m, 200)
    print(
        f"  m={m}: largest error of a sample, 200 samples: mean={errs.mean():.4f}  max={errs.max():.4f}  "
        f"fraction <= 0.05: {float(np.mean(errs <= 0.05)):.2f}"
    )
    small = sample_errors(rng, queries, p, 200, 200)
    print(f"  m=200: mean={small.mean():.4f}  fraction <= 0.05: {float(np.mean(small <= 0.05)):.2f}")

    print("[Exercise 6: crossover with Gaussian noise]")
    print(f"  N=2^20 alpha=0.05: private MW needs fewer records from |Q| = {crossover_queries(big_n, 0.05):.2g}")
    print(f"  N=2^10 alpha=0.05: from |Q| = {crossover_queries(2**10, 0.05):.2g}")


if __name__ == "__main__":
    main()
