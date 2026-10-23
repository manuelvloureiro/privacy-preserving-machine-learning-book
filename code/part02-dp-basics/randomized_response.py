"""Check the numbers in the course's randomized-response chapter and draw its figure.

    uv run python code/part02-dp-basics/randomized_response.py
    uv run python code/part02-dp-basics/randomized_response.py --seed 1 --out /tmp/figures

Prints the numbers quoted in book-course/part02-dp-basics/01-randomized-response.md, then saves
the chapter's figure. Each person reports their bit with probability 1/2 + γ and flips it
otherwise; ε-DP needs 1/2 + γ = e^ε/(1 + e^ε). The analyst debiases the reports.
"""

import argparse
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
matplotlib.rcParams["svg.hashsalt"] = "randomized_response"  # stable element ids across reruns
matplotlib.rcParams["svg.fonttype"] = "none"  # keep text as text

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from numpy.typing import NDArray

# Figures are build output: _build/figures/<part>/, which the build copies into each book.
FIGURES = Path(__file__).resolve().parents[2] / "_build" / "figures" / Path(__file__).parent.name


# --- Randomized response ---------------------------------------------------------------------


def truth_probability(eps: float) -> float:
    """1/2 + γ = e^ε/(1 + e^ε), the probability of reporting the true bit under ε-DP."""
    return math.exp(eps) / (1.0 + math.exp(eps))


def gamma_for(eps: float) -> float:
    """The bias γ = tanh(ε/2)/2."""
    return 0.5 * math.tanh(eps / 2.0)


def epsilon_for(gamma: float) -> float:
    """ε = ln((1/2 + γ)/(1/2 - γ))."""
    return math.log((0.5 + gamma) / (0.5 - gamma))


def estimator_variance(eps: float, n: int) -> float:
    """Var[p̃] = (1 - 4γ²)/(16γ² n) = e^ε/((e^ε - 1)² n)."""
    gamma = gamma_for(eps)
    return (1.0 - 4.0 * gamma**2) / (16.0 * gamma**2 * n)


def hoeffding_error(eps: float, n: int, beta: float) -> float:
    """coth(ε/2) √(ln(2/β)/(2n)), exceeded with probability at most β."""
    return math.sqrt(math.log(2.0 / beta) / (2.0 * n)) / math.tanh(eps / 2.0)


def simulate_errors(rng: np.random.Generator, eps: float, n: int, p: float, reps: int) -> NDArray[np.float64]:
    """|p̃ - p| over `reps` runs, on a dataset with round(p n) ones."""
    ones = round(p * n)
    truth = truth_probability(eps)
    gamma = gamma_for(eps)
    # The number of reported ones: truthful ones among the ones, flipped zeros among the zeros.
    reported = rng.binomial(ones, truth, size=reps) + rng.binomial(n - ones, 1.0 - truth, size=reps)
    estimate = (reported / n - 0.5 + gamma) / (2.0 * gamma)
    return np.abs(estimate - ones / n)


def error_quantile(rng: np.random.Generator, eps: float, n: int, p: float, reps: int, q: float) -> float:
    return float(np.quantile(simulate_errors(rng, eps, n, p, reps), q))


# --- Table and figure ------------------------------------------------------------------------


def print_table(rng: np.random.Generator) -> None:
    print("== Randomized response revisited")
    eps, n, beta = 1.0, 10_000, 0.05
    print(f"eps={eps:g}: truth prob={truth_probability(eps):.4f}  2 gamma={2 * gamma_for(eps):.4f}")
    print(f"eps round trip={epsilon_for(gamma_for(eps)):.4f}")
    print(
        f"n={n} beta={beta:g}: coth(eps/2)={1 / math.tanh(eps / 2):.4f}  "
        f"sqrt(ln(2/beta)/(2n))={math.sqrt(math.log(2 / beta) / (2 * n)):.5f}  "
        f"bound={hoeffding_error(eps, n, beta):.4f}"
    )
    print(f"sd={math.sqrt(estimator_variance(eps, n)):.4f}")
    for e in (0.5, 1.0, 2.0):
        lo, hi = 2 / e, 2 / e + e / 6
        print(f"eps={e:g}: 2/eps={lo:.4f}  coth(eps/2)={1 / math.tanh(e / 2):.4f}  2/eps+eps/6={hi:.4f}")
    for e in (0.5, 1.0, 2.0):
        sim = error_quantile(rng, e, n, 0.3, 2000, 0.95)
        print(
            f"eps={e:g} n={n}: simulated 95th percentile={sim:.4f}  bound/simulated={hoeffding_error(e, n, 0.05) / sim:.2f}"
        )


def plot_error(rng: np.random.Generator) -> Figure:
    ns = np.unique(np.logspace(2, 5, 13).astype(int))
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    for i, eps in enumerate((0.5, 1.0, 2.0)):
        color = f"C{i}"
        sim = [error_quantile(rng, eps, int(n), 0.3, 2000, 0.95) for n in ns]
        bound = [hoeffding_error(eps, int(n), 0.05) for n in ns]
        ax.plot(ns, sim, "-o", color=color, markersize=3, label=rf"$\varepsilon = {eps:g}$, simulated")
        ax.plot(ns, bound, "--", color=color, label=rf"$\varepsilon = {eps:g}$, bound")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"number of people $n$")
    ax.set_ylabel(r"error $|\tilde p - p|$ (95th percentile)")
    ax.legend(fontsize=8, ncol=2)
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
    fig = plot_error(np.random.default_rng(args.seed + 1))
    fig.savefig(args.out / "rr-error.svg", metadata={"Date": None})
    plt.close(fig)


if __name__ == "__main__":
    main()
