"""Check the worked solutions to the exercises of the lower-bounds chapter.

    uv run python book/part03-dp-foundations/code/solutions_lower_bounds.py

Prints every number quoted in solutions/09-lower-bounds.md with seed 0 (only exercise 6 is random).
Makes no figures; --out is accepted and ignored.

Conventions as in lower_bounds.py: records in {-1, +1}^d with attribute means p ~ U[-1, 1]^d,
replace neighbours on datasets of fixed size n unless stated otherwise.
"""

import argparse
import math
from pathlib import Path

import numpy as np
from lower_bounds import (
    BETA,
    DELTA,
    EPS,
    TRACE_D,
    TRACE_N,
    approx_packing_min_k,
    best_advantage,
    bisect,
    dn_eps_lower,
    dp_bit_accuracy_bound,
    fingerprint_best_alpha0,
    fingerprint_lower_constant,
    fingerprint_terms,
    histogram_error_lower,
    laplace_histogram_error,
    marginals_fingerprint_lower,
    normal_cdf,
    tracing_mean_sd,
)
from numpy.typing import NDArray

FIGURES = Path(__file__).resolve().parent.parent / "figures"

# --- Exercise 1: histograms under replace neighbours -----------------------------------------


def histogram_error_lower_replace(n_bins: int, eps: float, beta: float) -> float:
    """D_0 = n copies of x_0 and D_x with k of them replaced by x, x ≠ x_0: N - 1 candidates."""
    return max(0.0, ((math.log(n_bins - 1) + math.log1p(-beta)) / eps - 1) / 2)


def laplace_histogram_error_replace(n_bins: int, eps: float, beta: float) -> float:
    """Replace moves two counts by 1 each: Δ₁ = 2, so the Laplace scale and the error double."""
    return 2 * laplace_histogram_error(n_bins, eps, beta)


# --- Exercise 2: what packing can force under approximate DP --------------------------------


def approx_packing_limit(eps: float, delta: float, beta: float) -> float:
    """The k at which the floor e^{-kε}(1 - β) - k e^{-ε} δ reaches 0 (the limit m → ∞)."""
    return bisect(lambda k: math.exp(-k * eps) * (1 - beta) - k * math.exp(-eps) * delta, 0.0, 100.0)


# --- Exercise 3: the fingerprinting lemma for a shrunk mean ---------------------------------


def expected_abs_mean(n: int, nodes: int = 200) -> float:
    """E|x̄| for p ~ U[-1, 1] and n i.i.d. ±1 coordinates with mean p (exact sum over the count)."""
    ps, ws = np.polynomial.legendre.leggauss(nodes)
    total = 0.0
    for p, w in zip(ps, ws, strict=True):
        q = (1 + float(p)) / 2
        for ones in range(n + 1):
            prob = math.comb(n, ones) * q**ones * (1 - q) ** (n - ones)
            total += float(w) / 2 * prob * abs(2 * ones - n) / n
    return total


class ShrunkMean:
    """f(x) = c x̄, a biased estimator of p."""

    def __init__(self, c: float) -> None:
        self.c = c

    def __call__(self, xs: NDArray[np.float64]) -> NDArray[np.float64]:
        return self.c * xs.mean(axis=1)


# --- Exercise 5: how many attributes does tracing need? -------------------------------------


def tracing_advantage_no_noise(n: int, d: float) -> float:
    m = 2 * d / (3 * n)
    s = math.sqrt(d * (2 / 15 + 8 / (15 * n)))
    return 2 * normal_cdf(m / (2 * s)) - 1


def attributes_for_advantage(n: int, target: float) -> float:
    return bisect(lambda d: tracing_advantage_no_noise(n, d) - target, 1.0, 1e9)


# --- Exercise 6: tracing with a reference sample --------------------------------------------


def reference_tracing_scores(
    rng: np.random.Generator, n: int, d: int, m: int, trials: int
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Scores ⟨x̄, y - p̂⟩ with p̂ the mean of m fresh reference records (no noise on x̄)."""
    members = np.empty(trials * n)
    others = np.empty(trials * n)
    for t in range(trials):
        p = rng.uniform(-1.0, 1.0, size=d)
        x = np.where(rng.random((n, d)) < (1 + p) / 2, 1.0, -1.0)
        y = np.where(rng.random((n, d)) < (1 + p) / 2, 1.0, -1.0)
        ref = np.where(rng.random((m, d)) < (1 + p) / 2, 1.0, -1.0)
        p_hat = ref.mean(axis=0)
        release = x.mean(axis=0)
        members[t * n : (t + 1) * n] = (x - p_hat) @ release
        others[t * n : (t + 1) * n] = (y - p_hat) @ release
    return members, others


def reference_predicted_advantage(n: int, d: int, m: int) -> float:
    """Same mean gap; the variance grows by the factor (1 + 1/m)."""
    gap, sd = tracing_mean_sd(n, d, 0.0)
    return 2 * normal_cdf(gap / (2 * sd * math.sqrt(1 + 1 / m))) - 1


# --- The page's numbers ---------------------------------------------------------------------


def print_table(rng: np.random.Generator) -> None:
    n_bins = 10**6
    print("[Exercise 1: histograms under replace neighbours]")
    print(
        f"  N=1e6: add/remove lower {histogram_error_lower(n_bins, EPS, BETA):.4f}  "
        f"replace lower {histogram_error_lower_replace(n_bins, EPS, BETA):.4f}  "
        f"Laplace add/remove {laplace_histogram_error(n_bins, EPS, BETA):.2f}  "
        f"Laplace replace {laplace_histogram_error_replace(n_bins, EPS, BETA):.2f}"
    )

    print("[Exercise 2: packing under approximate DP]")
    for d in (20, 1000):
        print(f"  m=2^{d}: forced k >= {approx_packing_min_k(d * math.log(2), EPS, DELTA, BETA)}")
    print(
        f"  limit m -> inf: k = {approx_packing_limit(EPS, DELTA, BETA):.2f}  "
        f"ln(1/delta)/eps = {math.log(1 / DELTA) / EPS:.2f}"
    )

    print("[Exercise 3: fingerprinting lemma for f = c * mean, n = 5]")
    for c in (0.25, 0.5, 0.75, 1.0):
        corr, two_pf, dev = fingerprint_terms(ShrunkMean(c), 5)
        print(
            f"  c={c:g}: E[f sum(x-p)]={corr:.4f}  2c/3={2 * c / 3:.4f}  2E[pf]={two_pf:.4f}  "
            f"E|f-mean|={dev:.4f}  bound={2 / 3 - 2 * dev:.4f}"
        )
    print(
        f"  E|mean|: n=1 {expected_abs_mean(1):.4f}  n=5 {expected_abs_mean(5):.4f}  "
        f"n=100 {expected_abs_mean(100):.4f}  limit 0.5"
    )

    print("[Exercise 4: padding]")
    a0 = fingerprint_best_alpha0(0.01, BETA)
    print(
        f"  best alpha0={a0:.4f}  padded constant={a0 * fingerprint_lower_constant(a0, BETA):.4f}  "
        f"(2/3-4beta)^2/8={(2 / 3 - 4 * BETA) ** 2 / 8:.4f}"
    )
    for alpha in (0.1, 0.01):
        print(f"  d=1e4, alpha={alpha:g}: n >= {marginals_fingerprint_lower(10_000, EPS, DELTA, alpha, BETA):.1f}")

    print("[Exercise 5: attributes needed for tracing advantage 0.5]")
    for n in (100, 200):
        d = attributes_for_advantage(n, 0.5)
        print(f"  n={n}: d = {d:.0f}  d/n^2 = {d / n**2:.4f}")

    n, d, m = TRACE_N, TRACE_D, 50
    members, others = reference_tracing_scores(rng, n, d, m, 200)
    print(f"[Exercise 6: tracing with a reference sample, n={n}, d={d}, m={m}]")
    print(
        f"  member mean {members.mean():.2f} (predicted {tracing_mean_sd(n, d, 0.0)[0]:.2f})  "
        f"non-member sd {others.std():.2f} (predicted {tracing_mean_sd(n, d, 0.0)[1] * math.sqrt(1 + 1 / m):.2f})"
    )
    print(f"  advantage {best_advantage(members, others):.3f} (predicted {reference_predicted_advantage(n, d, m):.3f})")

    print("[Exercise 7: reconstruction implies epsilon]")
    print(f"  r=0.9, delta=1e-6: eps >= {dn_eps_lower(0.9, DELTA):.4f}")
    print(f"  r=0.99: eps >= {dn_eps_lower(0.99, DELTA):.4f}")
    print(f"  eps=1 allows at most r = {dp_bit_accuracy_bound(1.0, DELTA):.4f}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=FIGURES, help="ignored: this script makes no figures")
    args = parser.parse_args()
    seed: int = args.seed
    print_table(np.random.default_rng(seed))


if __name__ == "__main__":
    main()
