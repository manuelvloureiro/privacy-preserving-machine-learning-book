"""Check the worked solutions to the exercises of the hyperparameter-tuning chapter.

    uv run python code/part05-dp-ml/solutions_hyperparameter_tuning.py

Prints every number quoted in book-exercises/part05-dp-ml/03-hyperparameter-tuning.md (seed 0). Makes no
figures; --out is accepted and ignored.
"""

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")

import argparse
import math
from pathlib import Path

from hyperparameter_tuning import (
    Pgf,
    best_of_pmf,
    fixed_k_renyi_lower,
    fixed_pgf,
    geometric_pgf,
    max_loss,
    poisson_pgf,
    renyi,
    rr_pair,
    sigma_for_tuning,
    tnb_gamma,
    tnb_pgf,
)

# Figures are build output: _build/figures/<part>/, which the build copies into each book.
FIGURES = Path(__file__).resolve().parents[2] / "_build" / "figures" / Path(__file__).parent.name

# --- Exercise 1: the best of k randomized responses in Rényi divergence ----------------------------


def best_of_k_renyi(eps: float, k: int, alpha: float) -> tuple[float, float, float]:
    """(exact D_α of the best of k, the lower bound kε - k ln(1 + e^-ε)/(α - 1), composition k·D_α(one run))."""
    p, pp = rr_pair(eps)
    a, aa = best_of_pmf(p, fixed_pgf(k)), best_of_pmf(pp, fixed_pgf(k))
    return renyi(a, aa, alpha), fixed_k_renyi_lower(eps, k, alpha), k * renyi(p, pp, alpha)


# --- Exercise 2: geometric stopping on randomized response -----------------------------------------


def geometric_rr_bad_ratio(eps: float, gamma: float) -> float:
    """ln of P_D[bad]/P_D'[bad] for the best of a geometric number of runs: ε + ln((γ + (1-γ)s)/(γ + (1-γ)(1-s)))."""
    s = math.exp(eps) / (1 + math.exp(eps))
    return eps + math.log((gamma + (1 - gamma) * s) / (gamma + (1 - gamma) * (1 - s)))


def geometric_rr_exact(eps: float, gamma: float) -> float:
    p, pp = rr_pair(eps)
    return max_loss(best_of_pmf(p, geometric_pgf(gamma)), best_of_pmf(pp, geometric_pgf(gamma)))


# --- Exercise 3: the expected quantile of the best run ---------------------------------------------


def expected_quantile(pgf_integral: float) -> float:
    """E[K/(K+1)] = 1 - ∫_0^1 f(x) dx."""
    return 1 - pgf_integral


def integrate(f: Pgf, steps: int = 200_000) -> float:
    """Midpoint rule on [0, 1]."""
    h = 1 / steps
    return h * sum(f((i + 0.5) * h) for i in range(steps))


# --- Exercise 4: the Poisson theorem for a pure-DP run ---------------------------------------------


def rr_rdp(eps: float, alpha: float) -> float:
    p, pp = rr_pair(eps)
    return renyi(p, pp, alpha)


def poisson_pure_dp(eps: float, mu: float) -> tuple[float, float, float, float]:
    """(λ* = 1 + 1/(e^ε - 1), bound ε(λ*) + ln μ/(λ* - 1), exact D_λ* of the best of Poisson(μ) runs,
    composition μ·ε(λ*)), for randomized response with parameter ε."""
    lam = 1 + 1 / math.expm1(eps)
    p, pp = rr_pair(eps)
    a, aa = best_of_pmf(p, poisson_pgf(mu)), best_of_pmf(pp, poisson_pgf(mu))
    exact = max(renyi(a, aa, lam), renyi(aa, a, lam))
    one = rr_rdp(eps, lam)
    return lam, one + math.log(mu) / (lam - 1), exact, mu * one


# --- Exercise 6: how large a validation set ---------------------------------------------------------


def validation_size(k: int, eps: float, gap: float, beta: float) -> int:
    """Smallest m with 2(ln k + ln(1/β))/(ε m) ≤ gap: the exponential mechanism's accuracy gap stays below `gap`
    with probability at least 1 - β."""
    return math.ceil(2 * (math.log(k) + math.log(1 / beta)) / (eps * gap))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=FIGURES, help="ignored: this script makes no figures")
    parser.parse_args()

    print("== Exercise 1: best of k randomized responses, eps=0.5, k=5")
    for alpha in (2.0, 10.0, 100.0):
        exact, lower, comp = best_of_k_renyi(0.5, 5, alpha)
        print(f"order {alpha:g}: exact D={exact:.4f}  lower bound={lower:.4f}  composition={comp:.4f}")

    print("== Exercise 2: geometric stopping on randomized response, eps=0.5")
    for gamma in (0.5, 0.1, 0.01, 0.001):
        print(
            f"gamma={gamma:g}: ln ratio of 'bad'={geometric_rr_bad_ratio(0.5, gamma):.4f}  "
            f"exact eps={geometric_rr_exact(0.5, gamma):.4f}"
        )

    print("== Exercise 3: expected quantile with mean 8")
    g = tnb_gamma(0, 8)
    rows = (
        ("fixed 8", fixed_pgf(8)),
        ("Poisson 8", poisson_pgf(8)),
        ("geometric mean 8", geometric_pgf(1 / 8)),
        ("logarithmic mean 8", tnb_pgf(0, g)),
    )
    for name, f in rows:
        print(f"{name}: E[K/(K+1)]={expected_quantile(integrate(f)):.4f}")
    print(f"Poisson closed form: {1 - (1 - math.exp(-8)) / 8:.4f}")

    print("== Exercise 4: Poisson theorem for randomized response")
    for eps, mu in ((0.1, 100.0), (0.5, 10.0)):
        lam, bound, exact, comp = poisson_pure_dp(eps, mu)
        print(
            f"eps={eps:g}, mu={mu:g}: order={lam:.4f}  one run={rr_rdp(eps, lam):.4f}  bound={bound:.4f}  "
            f"exact={exact:.4f}  composition={comp:.4f}"
        )

    print("== Exercise 5: noise for a search of 10 runs at a total eps=6.71 (q=0.01, 10000 steps)")
    for name, strategy in (
        ("composition", ("fixed", 10.0)),
        ("Poisson", ("poisson", 10.0)),
        ("geometric", ("tnb1", 10.0)),
        ("logarithmic", ("tnb0", 10.0)),
    ):
        print(f"{name}: sigma={sigma_for_tuning(0.01, 10_000, 6.71, strategy):.3g}")

    print("== Exercise 6: validation set size")
    for k in (8, 1000):
        print(f"k={k}, eps=0.5, gap 0.005, beta=0.05: m={validation_size(k, 0.5, 0.005, 0.05)}")


if __name__ == "__main__":
    main()
