"""Check the worked solutions to the exercises of the adaptive data analysis chapter.

    uv run python code/part05-dp-ml/solutions_adaptive_data_analysis.py

Prints every number quoted in book-exercises/part05-dp-ml/11-adaptive-data-analysis.md (seed 0). Makes no
figures; --out is accepted and ignored.
"""

import argparse
import math
from pathlib import Path

import numpy as np
from adaptive_data_analysis import (
    boosting_approximation,
    boosting_attack,
    gaussian_error,
    gaussian_error_value,
    point_query_population,
    required_n,
    splitting_error,
    thresholdout_epsilon,
    transfer_lemma_bound,
)

# Figures are build output: _build/figures/<part>/, which the build copies into each book.
FIGURES = Path(__file__).resolve().parents[2] / "_build" / "figures" / Path(__file__).parent.name


# --- Exercise 1: how many point queries break a sample of n records -------------------------------


def smallest_breaking_domain(n: int) -> int:
    """Smallest domain size N for which the final query's expected error 1 - E[population] exceeds 1/2."""
    n_dom = 1
    while 1 - point_query_population(n, n_dom) <= 0.5:
        n_dom += 1
    return n_dom


# --- Exercise 2: randomized-response selection from one record ------------------------------------


def rr_selection_bias(eps: float) -> float:
    """Bias of choosing q(x) = 1[x = b] where b is the single record's bit, kept with probability e^ε/(1 + e^ε).

    The empirical value of the chosen query is P(b = x_1) = e^ε/(1 + e^ε); its population value is 1/2
    for a uniform bit. The difference is tanh(ε/2)/2.
    """
    return math.exp(eps) / (1 + math.exp(eps)) - 0.5


# --- Exercise 4: the balancing formula for the Gaussian mechanism ---------------------------------


def gaussian_n_formula(k: int, alpha: float, delta: float) -> float:
    """n = 4 sqrt(4 k ln(2k) ln(1/δ)) / α², from balancing the noise against ε ≈ 2 sqrt(ρ ln(1/δ))."""
    return 4 * math.sqrt(4 * k * math.log(2 * k) * math.log(1 / delta)) / alpha**2


# --- Exercise 5: splitting by rounds ---------------------------------------------------------------


def rounds_n(k: int, rounds: int, alpha: float) -> float:
    """Records for an expected largest error alpha when k queries come in `rounds` rounds: r ln(2k)/(2α²)."""
    return rounds * math.log(2 * k) / (2 * alpha**2)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=FIGURES, help="ignored: the exercises have no figures")
    args = parser.parse_args()
    rng = np.random.default_rng(args.seed)

    print("== Exercise 1: the point-query attack")
    for n_dom in (2000, 10_000):
        pop = point_query_population(1000, n_dom)
        print(f"[Ex1 n=1000 N={n_dom}] E population={pop:.4f}  n/N={1000 / n_dom:.4f}  error={1 - pop:.4f}")
    n_break = smallest_breaking_domain(1000)
    print(
        f"[Ex1 n=1000] smallest N with error > 1/2: {n_break}  queries={n_break + 1}  "
        f"approximation n/ln 2={1000 / math.log(2):.1f}"
    )

    print("== Exercise 2: the reverse direction and a selection from one record")
    for eps in (0.1, 1.0):
        b = rr_selection_bias(eps)
        print(
            f"[Ex2 eps={eps:g}] bias={b:.5f}  tanh(eps/2)/2={math.tanh(eps / 2) / 2:.5f}  eps/4={eps / 4:.5f}  "
            f"sharper (e^eps-1)/2={transfer_lemma_bound(eps) / 2:.5f}  lemma={transfer_lemma_bound(eps):.5f}"
        )

    print("== Exercise 4: Gaussian noise for k = 1000 queries")
    n_opt = required_n(gaussian_error_value, 1000, 0.05)
    g = gaussian_error(n_opt, 1000)
    formula = gaussian_n_formula(1000, 0.05, g.delta)
    print(f"[Ex4 k=1000 alpha=0.05] optimum delta={g.delta:.4f}  formula n={formula:.0f}  numerical n={n_opt}")
    rho = 1000 / (2 * n_opt**2 * g.sigma**2)
    print(
        f"[Ex4 n={n_opt}] sigma={g.sigma:.5f}  rho={rho:.3e}  eps={g.eps:.4f}  noise={g.sample_error:.4f}  "
        f"transfer={g.transfer:.4f}"
    )

    print("== Exercise 5: rounds of adaptivity")
    for rounds in (1, 10, 1000):
        print(f"[Ex5 k=1000 r={rounds}] n={math.ceil(rounds_n(1000, rounds, 0.05))}")
    print(f"[Ex5 k=1000 r=k] splitting n={required_n(splitting_error, 1000, 0.05)}")

    print("== Exercise 6: the boosting attack's approximation")
    for n, k in ((1000, 100), (4000, 400)):
        mean, se = boosting_attack(rng, n, k, trials=40)
        print(
            f"[Ex6 n={n} k={k}] simulated overfit={mean - 0.5:.4f}±{se:.4f}  approx={boosting_approximation(n, k):.4f}"
        )

    print("== Exercise 7: Thresholdout's budget")
    per = thresholdout_epsilon(1, 2000, 1 / math.sqrt(2000))
    print(f"[Ex7 n=2000] eps per detection={per:.4f}  detections within eps=1: {math.floor(1 / per)}")
    sqrt_n = 4 * 300 / 0.1  # ε = 4B/sqrt(n) for correlations with σ = 1/sqrt(n)
    print(f"[Ex7 B=300 eps=0.1] sqrt(n)={sqrt_n:.0f}  n={sqrt_n**2:.3e}")


if __name__ == "__main__":
    main()
