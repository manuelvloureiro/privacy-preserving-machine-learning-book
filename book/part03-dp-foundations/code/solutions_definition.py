"""Check the worked solutions to the exercises of the definition-of-differential-privacy chapter.

    uv run python book/part03-dp-foundations/code/solutions_definition.py

Prints every number quoted in solutions/01-definition.md (seed 0; nothing here is random, the
seed is accepted for a uniform interface). Makes no figures; --out is accepted and ignored.

Conventions as in definition.py: P = M(D), Q = M(D'), logs are natural.
"""

import argparse
import math
from pathlib import Path

import numpy as np
from definition import (
    LN3,
    advantage_bound,
    laplace_tv,
    max_divergence,
    product,
    release_records_neighbour,
    rr_delta,
    rr_output,
    rr_truth_prob,
    smallest_delta,
    smallest_epsilon,
)

FIGURES = Path(__file__).resolve().parent.parent / "figures"

# --- Exercise 1: randomized response with k answers -----------------------------------------------


def k_rr_output(value: int, k: int, eps: float) -> list[float]:
    """Output distribution of k-ary randomized response for a true value in {0, ..., k-1}."""
    denom = math.exp(eps) + k - 1
    return [math.exp(eps) / denom if v == value else 1.0 / denom for v in range(k)]


def k_rr_epsilon(k: int, eps: float) -> float:
    """The smallest ε over all pairs of true values, computed from the output distributions."""
    return max(
        smallest_epsilon(k_rr_output(a, k, eps), k_rr_output(b, k, eps)) for a in range(k) for b in range(k) if a != b
    )


# --- Exercise 2: from add/remove to replace with δ ------------------------------------------------


def replace_parameters(eps: float, delta: float) -> tuple[float, float]:
    """(2ε, (1 + e^ε)δ): the replace guarantee implied by (ε, δ)-DP under add/remove."""
    return 2.0 * eps, (1.0 + math.exp(eps)) * delta


def release_records_replace(n: int, delta: float) -> tuple[list[float], list[float]]:
    """Release-each-record outputs on D and on D with its last record replaced by a new one.

    Outcomes are subsets of the n + 1 records involved: the n - 1 shared ones, the old and the new.
    """
    shared: list[list[float]] = [[1.0 - delta, delta]] * (n - 1)
    on_d = product(shared + [[1.0 - delta, delta], [1.0, 0.0]])
    on_d_replaced = product(shared + [[1.0, 0.0], [1.0 - delta, delta]])
    return on_d, on_d_replaced


# --- Exercise 3: δ(ε) of randomized response -----------------------------------------------------


def rr_delta_formula(eps: float, eps0: float) -> float:
    """(e^ε0 - e^ε)_+/(1 + e^ε0)."""
    return max(math.exp(eps0) - math.exp(eps), 0.0) / (1.0 + math.exp(eps0))


# --- Exercise 4: asking twice ---------------------------------------------------------------------


def repeated_rr_loss(k: int, eps0: float) -> float:
    """Largest privacy loss of k independent randomized-response reports of the same bit."""
    yes = product([rr_output(1, eps0)] * k)
    no = product([rr_output(0, eps0)] * k)
    return max_divergence(yes, no)


def posterior_after_yeses(k: int, eps0: float, prior: float) -> float:
    """Pr[true answer yes | k reports of yes], by Bayes' rule."""
    t = rr_truth_prob(eps0)
    num = prior * t**k
    return num / (num + (1.0 - prior) * (1.0 - t) ** k)


# --- Exercise 5: choosing δ -----------------------------------------------------------------------


def largest_safe_delta(n: int, risk: float) -> float:
    """Largest δ with 1 - (1 - δ)^n ≤ risk: δ = 1 - (1 - risk)^{1/n}."""
    return -math.expm1(math.log1p(-risk) / n)


# --- Exercise 7: the Laplace count against the bound ---------------------------------------------


def laplace_over_bound(eps: float) -> float:
    return laplace_tv(eps) / advantage_bound(eps)


def print_table(rng: np.random.Generator) -> None:
    del rng  # nothing random here
    print("== Exercise 1")
    for k, eps in ((2, LN3), (4, 1.0), (10, 1.0)):
        out = k_rr_output(0, k, eps)
        print(
            f"k={k} eps={eps:.4f}: truth prob={out[0]:.4f}  other prob={out[1]:.4f}  "
            f"smallest eps over all pairs={k_rr_epsilon(k, eps):.4f}"
        )
    print(f"k=2 agrees with randomized response: {k_rr_output(1, 2, LN3)[1]:.4f} vs {rr_truth_prob(LN3):.4f}")

    print("== Exercise 2")
    e2, d2 = replace_parameters(1.0, 1e-6)
    print(f"eps=1 delta=1e-6 -> replace eps={e2:g} delta={d2:.4e}")
    on_d, on_r = release_records_replace(3, 0.1)
    on_d_add, on_d_prime = release_records_neighbour(3, 0.1)
    print(
        f"release w.p. 0.1, n=3: add/remove delta(0)={smallest_delta(on_d_add, on_d_prime, 0.0):.4f}  "
        f"replace delta(0)={smallest_delta(on_d, on_r, 0.0):.4f}  "
        f"conversion bound={replace_parameters(0.0, 0.1)[1]:.4f}"
    )

    print("== Exercise 3")
    for eps in (0.0, 0.5, 1.0):
        print(f"eps0=ln3 eps={eps:g}: formula={rr_delta_formula(eps, LN3):.4f}  hockey-stick={rr_delta(eps, LN3):.4f}")

    print("== Exercise 4")
    for k in (1, 2, 3):
        print(
            f"k={k}: largest loss={repeated_rr_loss(k, LN3):.4f}  k ln3={k * LN3:.4f}  "
            f"posterior after k yes={posterior_after_yeses(k, LN3, 0.5):.4f}"
        )

    print("== Exercise 5")
    d = largest_safe_delta(1_000_000, 0.01)
    print(f"n=1e6 risk 0.01: delta={d:.4e}  approx risk/n={0.01 / 1_000_000:.4e}")

    print("== Exercise 6")
    print(f"bound at eps=0: delta=1e-5 -> {advantage_bound(0.0, 1e-5):.4e}")
    print(
        f"release w.p. 0.1, n=3: TV={smallest_delta(on_d_add, on_d_prime, 0.0):.4f}  bound={advantage_bound(0.0, 0.1):.4f}"
    )

    print("== Exercise 7")
    for eps in (0.1, 1.0, 5.0, 10.0):
        print(
            f"eps={eps:g}: Laplace={laplace_tv(eps):.6f}  bound={advantage_bound(eps):.6f}  "
            f"ratio={laplace_over_bound(eps):.4f}  1-bound={1 - advantage_bound(eps):.4e}  "
            f"1-Laplace={1 - laplace_tv(eps):.4e}"
        )
    grid = [i / 1000 for i in range(1, 10_001)]
    worst = min(grid, key=laplace_over_bound)
    print(f"smallest ratio={laplace_over_bound(worst):.4f} at eps={worst:.2f}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=FIGURES, help="accepted for a uniform interface; no figures")
    args = parser.parse_args()
    seed: int = args.seed
    print_table(np.random.default_rng(seed))


if __name__ == "__main__":
    main()
