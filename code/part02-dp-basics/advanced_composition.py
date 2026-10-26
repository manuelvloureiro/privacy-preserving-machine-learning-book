"""Check the numbers in the course's advanced-composition chapter and draw its figure.

    uv run python code/part02-dp-basics/advanced_composition.py
    uv run python code/part02-dp-basics/advanced_composition.py --seed 1 --out /tmp/figures

Prints the numbers quoted in book-course/part02-dp-basics/04-advanced-composition.md, then saves
the chapter's figure. The k-fold composition of (ε, δ)-DP mechanisms is (kε, kδ)-DP by basic
composition and (ε√(2k ln(1/δ')) + kε tanh(ε/2), kδ + δ')-DP by advanced composition. For δ = 0
the exact value is that of k independent randomized responses, the worst case.
"""

import argparse
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # for code/shared
from shared.figures import configure

configure("advanced_composition")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

# Figures are build output: _build/figures/<part>/, which the build copies into each book.
FIGURES = Path(__file__).resolve().parents[2] / "_build" / "figures" / Path(__file__).parent.name


# --- The theorem -----------------------------------------------------------------------------


def basic_epsilon(eps: float, k: int) -> float:
    """Basic composition: kε."""
    return k * eps


def advanced_epsilon(eps: float, k: int, delta_prime: float) -> float:
    """Advanced composition: ε√(2k ln(1/δ')) + kε (e^ε - 1)/(e^ε + 1)."""
    return eps * math.sqrt(2.0 * k * math.log(1.0 / delta_prime)) + k * eps * math.tanh(eps / 2.0)


def per_mechanism_epsilon(eps_total: float, k: int, delta_prime: float) -> float:
    """The per-mechanism budget ε_0/(2√(2k ln(1/δ'))) of the corollary."""
    return eps_total / (2.0 * math.sqrt(2.0 * k * math.log(1.0 / delta_prime)))


def largest_per_mechanism_epsilon(eps_total: float, k: int, delta_prime: float) -> float:
    """The largest ε with advanced_epsilon(ε, k, δ') ≤ ε_total, by bisection (it increases with ε)."""
    lo, hi = 0.0, eps_total
    for _ in range(100):
        mid = 0.5 * (lo + hi)
        if advanced_epsilon(mid, k, delta_prime) <= eps_total:
            lo = mid
        else:
            hi = mid
    return lo


def crossover(eps: float, delta_prime: float) -> int:
    """The smallest k at which advanced composition beats basic composition."""
    k = 1
    while advanced_epsilon(eps, k, delta_prime) >= basic_epsilon(eps, k):
        k += 1
    return k


# --- The exact value for pure DP -------------------------------------------------------------


def log_binomial_pmf(k: int, q: float) -> list[float]:
    """ln Pr[Bin(k, q) = b] for b = 0, ..., k."""
    return [
        math.lgamma(k + 1) - math.lgamma(b + 1) - math.lgamma(k - b + 1) + b * math.log(q) + (k - b) * math.log1p(-q)
        for b in range(k + 1)
    ]


def exact_delta(eps: float, k: int, eps_total: float) -> float:
    """sup_T Pr[U^k ∈ T] - e^ε̃ Pr[V^k ∈ T] for k randomized responses (the pair U, V with δ = 0).

    With b ones among the k outcomes, Pr[U^k = z] = e^{ε(k - 2b)} Pr[V^k = z], and the number of
    ones under U is Bin(k, 1/(1 + e^ε)).
    """
    total = 0.0
    for b, log_p in enumerate(log_binomial_pmf(k, 1.0 / (1.0 + math.exp(eps)))):
        loss = eps * (k - 2 * b)
        if loss > eps_total:
            total += math.exp(log_p) * (1.0 - math.exp(eps_total - loss))
    return total


def exact_epsilon(eps: float, k: int, delta_prime: float) -> float:
    """The smallest ε̃ with exact_delta(ε, k, ε̃) ≤ δ', by bisection."""
    lo, hi = 0.0, k * eps
    for _ in range(100):
        mid = 0.5 * (lo + hi)
        if exact_delta(eps, k, mid) <= delta_prime:
            hi = mid
        else:
            lo = mid
    return hi


# --- Mean estimation -------------------------------------------------------------------------


def laplace_l2_error_basic(d: int, n: int, eps: float) -> float:
    """√(E‖Y‖²) for Laplace noise of scale d/(nε) in each of d coordinates: √2 d^{3/2}/(nε)."""
    return math.sqrt(2.0 * d) * d / (n * eps)


def laplace_l2_error_advanced(d: int, n: int, eps: float, delta_prime: float) -> float:
    """√(E‖Y‖²) for Laplace noise of scale 1/(nε_0) per coordinate, with ε_0 from the corollary."""
    return math.sqrt(2.0 * d) / (n * per_mechanism_epsilon(eps, d, delta_prime))


def laplace_l2_error_tuned(d: int, n: int, eps: float, delta_prime: float) -> float:
    """The same, with the largest ε_0 that advanced composition allows."""
    return math.sqrt(2.0 * d) / (n * largest_per_mechanism_epsilon(eps, d, delta_prime))


# --- Table and figure ------------------------------------------------------------------------


def print_table(rng: np.random.Generator) -> None:
    del rng  # nothing random in this chapter
    print("== Advanced composition")
    eps, dp = 0.1, 1e-6
    for k in (100, 1000):
        lead = eps * math.sqrt(2 * k * math.log(1 / dp))
        second = k * eps * math.tanh(eps / 2)
        print(
            f"eps={eps:g} k={k} delta'={dp:g}: basic={basic_epsilon(eps, k):.2f}  lead={lead:.3f}  "
            f"second={second:.3f}  advanced={advanced_epsilon(eps, k, dp):.3f}  exact={exact_epsilon(eps, k, dp):.3f}"
        )
    print(f"ln(1/delta')={math.log(1 / dp):.4f}  tanh(eps/2)={math.tanh(eps / 2):.5f}")
    print(f"crossover eps={eps:g} delta'={dp:g}: k={crossover(eps, dp)}")
    # Every k up to 300, where the ratio peaks, then a logarithmic grid up to 10,000.
    ks = sorted(set(range(100, 301)) | {int(k) for k in np.logspace(2, 4, 41)})
    ratios = [advanced_epsilon(eps, k, dp) / exact_epsilon(eps, k, dp) for k in ks]
    print(f"advanced/exact for k in [100, 10000]: max={max(ratios):.3f}  at k=10000={ratios[-1]:.3f}")
    print("== Budget")
    eps_total, k = 1.0, 1000
    e0 = per_mechanism_epsilon(eps_total, k, dp)
    e1 = largest_per_mechanism_epsilon(eps_total, k, dp)
    print(f"eps0={eps_total:g} k={k} delta'={dp:g}: per mechanism={e0:.5f}  advanced={advanced_epsilon(e0, k, dp):.4f}")
    print(f"largest per mechanism={e1:.5f}  basic would give={basic_epsilon(e1, k):.3f}")
    print("== Mean estimation")
    n = 100_000
    for d in (100, 1000, 10_000):
        print(
            f"d={d} n={n} eps={eps_total:g}: basic={laplace_l2_error_basic(d, n, eps_total):.4f}  "
            f"corollary={laplace_l2_error_advanced(d, n, eps_total, dp):.4f}  "
            f"tuned={laplace_l2_error_tuned(d, n, eps_total, dp):.4f}"
        )


def plot_composition() -> Figure:
    eps, dp = 0.1, 1e-6
    ks = np.unique(np.logspace(0, 4, 41).astype(int))
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.plot(ks, [basic_epsilon(eps, int(k)) for k in ks], color="C0", label=r"basic, $k\varepsilon$")
    ax.plot(ks, [advanced_epsilon(eps, int(k), dp) for k in ks], "--", color="C1", label="advanced composition")
    ax.plot(ks, [exact_epsilon(eps, int(k), dp) for k in ks], ":", color="C2", lw=2.0, label="exact (worst case)")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"number of mechanisms $k$")
    ax.set_ylabel(r"total $\varepsilon$ of the composition")
    ax.legend(fontsize=9)
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    return fig


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=FIGURES)
    args = parser.parse_args()
    print_table(np.random.default_rng(args.seed))
    args.out.mkdir(parents=True, exist_ok=True)
    fig = plot_composition()
    fig.savefig(args.out / "ac-composition.svg", metadata={"Date": None})
    plt.close(fig)


if __name__ == "__main__":
    main()
