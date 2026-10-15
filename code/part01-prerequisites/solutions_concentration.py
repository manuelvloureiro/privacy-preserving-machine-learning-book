"""Compute the numbers quoted in the solutions to the concentration exercises.

    uv run python code/part01-prerequisites/solutions_concentration.py
    uv run python code/part01-prerequisites/solutions_concentration.py --seed 1

Prints every number in book-exercises/part01-prerequisites/02-concentration.md (it matches exactly with the default seed).
The exercises ask for no figures, so --out is accepted but unused.
"""

import argparse
import math
from collections.abc import Callable
from pathlib import Path

import numpy as np
from concentration import azuma_bound, binomial_tail, laplace_max_bound, normal_sf, tail_at
from numpy.typing import NDArray

# Figures are build output: _build/figures/<part>/, which the build copies into each book.
FIGURES = Path(__file__).resolve().parents[2] / "_build" / "figures" / Path(__file__).parent.name


def bisect_increasing(f: Callable[[float], float], lo: float, hi: float, iters: int = 200) -> float:
    """The root of f on [lo, hi], assuming f(lo) <= 0 <= f(hi) and f increasing."""
    for _ in range(iters):
        mid = (lo + hi) / 2
        if f(mid) <= 0:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


# --- Exercise 4: sample size for a mean -----------------------------------------------------


def hoeffding_sample_size(t: float, beta: float, width: float = 1.0) -> int:
    """Smallest n with the two-sided Hoeffding bound 2 exp(-2nt²/(b-a)²) <= β."""
    return math.ceil(width**2 * math.log(2 / beta) / (2 * t * t))


# --- Exercise 5: noisy argmax ---------------------------------------------------------------


def noisy_argmax_gap_bound(k: int, b: float, beta: float) -> float:
    """2b ln(k/β): every |noise| is below half of it with probability >= 1 - β."""
    return 2 * laplace_max_bound(k, b, beta)


# --- Exercise 6: Chernoff for a Poisson -----------------------------------------------------


def poisson_upper_tail(mu: float, k: int) -> float:
    """Exact Pr[N >= k] for N ~ Poisson(μ)."""
    term = math.exp(-mu)
    below = 0.0
    for j in range(k):
        below += term
        term *= mu / (j + 1)
    return 1 - below


def chernoff_upper_exact_exponent(mu: float, delta: float) -> float:
    """exp(-μ[(1+δ)ln(1+δ) - δ]), the optimized Chernoff bound before simplifying."""
    return math.exp(-mu * ((1 + delta) * math.log(1 + delta) - delta))


def chernoff_upper_simplified(mu: float, delta: float) -> float:
    """exp(-δ²μ/(2+δ))."""
    return math.exp(-delta * delta * mu / (2 + delta))


# --- Exercise 7: Azuma for other adaptive rules ---------------------------------------------

BoolArray = NDArray[np.bool_]
FloatArray = NDArray[np.float64]
# A rule sees the last flip, the flip before it, and the running martingale, and returns
# True where the high coin should be used for the next flip.
Rule = Callable[[BoolArray, BoolArray, FloatArray], BoolArray]


def rule_after_heads(last: BoolArray, before: BoolArray, m: FloatArray) -> BoolArray:
    """The chapter's rule: high coin after a head."""
    return last


def rule_after_two_heads(last: BoolArray, before: BoolArray, m: FloatArray) -> BoolArray:
    """High coin only after two heads in a row."""
    return last & before


def rule_when_behind(last: BoolArray, before: BoolArray, m: FloatArray) -> BoolArray:
    """High coin whenever the martingale is negative, trying to push it up."""
    return m < 0


def rule_always_high(last: BoolArray, before: BoolArray, m: FloatArray) -> BoolArray:
    """No adaptivity: always the high coin, so the flips are independent."""
    return np.ones_like(last)


RULES: dict[str, Rule] = {
    "after one head": rule_after_heads,
    "after two heads": rule_after_two_heads,
    "when behind": rule_when_behind,
    "always 0.7": rule_always_high,
}


def adaptive_martingale(
    rng: np.random.Generator, rule: Rule, n: int, p_low: float, p_high: float, trials: int
) -> FloatArray:
    """M_n = Σ (X_k - p_k), where the coin for flip k is chosen by `rule` from the past."""
    m = np.zeros(trials)
    last = np.zeros(trials, dtype=bool)
    before = np.zeros(trials, dtype=bool)
    for _ in range(n):
        p = np.where(rule(last, before, m), p_high, p_low)
        x = rng.random(trials) < p
        m += x - p
        before, last = last, x
    return m


def martingale_normal_approx(t: float, n: int, step_variance: float) -> float:
    """Normal approximation with variance n · p(1-p): both coins have the same p(1-p)."""
    return normal_sf(t / math.sqrt(n * step_variance))


# --- Exercise 8: upper bounds on a low rate -------------------------------------------------


def hoeffding_upper(successes: int, n: int, beta: float) -> float:
    """p̂ + sqrt(ln(1/β)/(2n)), one-sided."""
    return successes / n + math.sqrt(math.log(1 / beta) / (2 * n))


def bernstein_plugin_upper(successes: int, n: int, beta: float, m: float = 1.0) -> float:
    """p̂ + sqrt(2σ̂² ln(1/β)/n) + 2M ln(1/β)/(3n), with σ̂² = p̂(1-p̂) (the chapter's heuristic)."""
    p_hat = successes / n
    lg = math.log(1 / beta)
    return p_hat + math.sqrt(2 * p_hat * (1 - p_hat) * lg / n) + 2 * m * lg / (3 * n)


def bernstein_upper(successes: int, n: int, beta: float, m: float = 1.0) -> float:
    """Largest p that Bernstein, with the true variance p(1-p), does not reject at level β.

    p is rejected when exp(-n s²/(2p(1-p) + 2Ms/3)) < β, where s = p - p̂.
    """
    p_hat = successes / n
    lg = math.log(1 / beta)

    def excess(p: float) -> float:
        s = p - p_hat
        return n * s * s - lg * (2 * p * (1 - p) + 2 * m * s / 3)

    return bisect_increasing(excess, p_hat, 0.5)


def chernoff_simplified_upper(successes: int, n: int, beta: float) -> float:
    """Largest p with exp(-(μ - s)²/(2μ)) >= β, μ = np: the lower-tail bound exp(-δ²μ/2)."""
    lg = math.log(1 / beta)
    s = successes
    mu = s + lg + math.sqrt(lg * lg + 2 * s * lg)  # larger root of (μ - s)² = 2μ ln(1/β)
    return mu / n


def chernoff_exact_exponent_upper(successes: int, n: int, beta: float) -> float:
    """Same, with the unsimplified exponent μ[δ + (1-δ)ln(1-δ)] = μ - s + s ln(s/μ)."""
    lg = math.log(1 / beta)
    s = successes

    def excess(mu: float) -> float:
        return mu - s + s * math.log(s / mu) - lg

    return bisect_increasing(excess, float(s), 10.0 * (s + lg)) / n


def binomial_cdf(n: int, p: float, k: int) -> float:
    """Pr[S <= k] for S ~ Binomial(n, p), summed in log space for stability."""
    return sum(
        math.exp(
            math.lgamma(n + 1)
            - math.lgamma(j + 1)
            - math.lgamma(n - j + 1)
            + j * math.log(p)
            + (n - j) * math.log1p(-p)
        )
        for j in range(k + 1)
    )


def clopper_pearson_upper(successes: int, n: int, beta: float) -> float:
    """One-sided exact upper bound: the p with Pr[S <= successes; p] = β (by bisection)."""
    return bisect_increasing(lambda p: beta - binomial_cdf(n, p, successes), successes / n, 0.5)


# --- The solutions' numbers -----------------------------------------------------------------


def print_table(rng: np.random.Generator) -> None:
    n4 = hoeffding_sample_size(0.01, 0.01)
    print(f"[Ex4] ln(200)/(2e-4)={math.log(200) / 2e-4:.1f}  n={n4}")

    print(f"[Ex5] k=100 b=1 beta=0.05 gap bound={noisy_argmax_gap_bound(100, 1.0, 0.05):.2f}")

    mu, delta = 10.0, 1.0
    print(
        f"[Ex6] Poisson mu={mu:g} delta={delta:g}: exact={poisson_upper_tail(mu, 20):.2e}  "
        f"Chernoff={chernoff_upper_exact_exponent(mu, delta):.2e}  "
        f"simplified={chernoff_upper_simplified(mu, delta):.2e}"
    )

    n, t = 200, 20.0
    print(
        f"[Ex7] n={n} t={t:g}: Azuma={azuma_bound(t, n):.2e}  "
        f"normal approx={martingale_normal_approx(t, n, 0.21):.2e}  "
        f"exact for always 0.7={binomial_tail(n, 0.7, 140 + int(t)):.2e}"
    )
    for name, rule in RULES.items():
        m = adaptive_martingale(rng, rule, n, 0.3, 0.7, trials=1_000_000)
        print(f"[Ex7] rule={name}: empirical={tail_at(m, t):.2e}")

    s, n, beta = 3, 5000, 0.05
    print(
        f"[Ex8] {s}/{n} beta={beta}: p_hat={s / n:.4f}  "
        f"Hoeffding={hoeffding_upper(s, n, beta):.4f}  "
        f"Bernstein plug-in={bernstein_plugin_upper(s, n, beta):.5f}  "
        f"Bernstein={bernstein_upper(s, n, beta):.5f}  "
        f"Chernoff={chernoff_simplified_upper(s, n, beta):.5f}  "
        f"Chernoff exact exponent={chernoff_exact_exponent_upper(s, n, beta):.5f}  "
        f"Clopper-Pearson={clopper_pearson_upper(s, n, beta):.5f}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=FIGURES, help="unused: the solutions draw no figures")
    args = parser.parse_args()
    seed: int = args.seed
    print_table(np.random.default_rng(seed))


if __name__ == "__main__":
    main()
