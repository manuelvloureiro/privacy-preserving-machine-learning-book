"""Check the numbers in the course's exponential-mechanism chapter and draw its figure.

    uv run python code/part02-dp-basics/exponential_mechanism.py
    uv run python code/part02-dp-basics/exponential_mechanism.py --seed 1 --out /tmp/figures

Prints the numbers quoted in book-course/part02-dp-basics/05-exponential-mechanism.md, then saves
the chapter's figure. The exponential mechanism outputs an object h with probability proportional
to exp(ε s(X, h)/(2Δ)). Here the objects are prices on a grid, the score is the revenue
p |{i : v_i ≥ p}|, and its sensitivity is at most 1 for valuations in [0, 1].
"""

import argparse
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
matplotlib.rcParams["svg.hashsalt"] = "exponential_mechanism"  # stable element ids across reruns
matplotlib.rcParams["svg.fonttype"] = "none"  # keep text as text

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from numpy.typing import NDArray

# Figures are build output: _build/figures/<part>/, which the build copies into each book.
FIGURES = Path(__file__).resolve().parents[2] / "_build" / "figures" / Path(__file__).parent.name


# --- The mechanism ---------------------------------------------------------------------------


def exponential_probabilities(scores: NDArray[np.float64], eps: float, sensitivity: float) -> NDArray[np.float64]:
    """Pr[M(X) = h] ∝ exp(ε s(X, h)/(2Δ)), computed stably by subtracting the largest score."""
    logits = eps * (scores - scores.max()) / (2.0 * sensitivity)
    weights = np.exp(logits)
    return weights / weights.sum()


def utility_loss(sensitivity: float, eps: float, size: int, t: float) -> float:
    """(2Δ/ε)(ln|H| + t): the score falls this far below OPT with probability at most e^{-t}."""
    return 2.0 * sensitivity / eps * (math.log(size) + t)


# --- Pricing a digital good ------------------------------------------------------------------


def revenue(valuations: NDArray[np.float64], prices: NDArray[np.float64]) -> NDArray[np.float64]:
    """p |{i : v_i ≥ p}| for each price p."""
    sorted_v = np.sort(valuations)
    buyers = len(sorted_v) - np.searchsorted(sorted_v, prices, side="left")
    return prices * buyers


def best_revenue(valuations: NDArray[np.float64]) -> float:
    """γ = max_p p |{i : v_i ≥ p}|, attained at one of the valuations."""
    return float(revenue(valuations, valuations).max())


def price_grid(m: int) -> NDArray[np.float64]:
    """The prices α, 2α, ..., 1 with α = 1/m."""
    return np.arange(1, m + 1, dtype=np.float64) / m


def score_quantile(scores: NDArray[np.float64], probs: NDArray[np.float64], q: float) -> float:
    """The q-quantile of the score of the mechanism's output."""
    order = np.argsort(scores)
    cdf = np.cumsum(probs[order])
    return float(scores[order][np.searchsorted(cdf, q)])


# --- PAC learning ----------------------------------------------------------------------------


def private_pac_samples(log_class_size: float, alpha: float, beta: float, eps: float) -> tuple[float, float]:
    """The two terms 2 ln(2|C|/β)/α² and 4 ln(2|C|/β)/(αε) of the sample bound."""
    log_term = math.log(2.0 / beta) + log_class_size
    return 2.0 * log_term / alpha**2, 4.0 * log_term / (alpha * eps)


# --- Table and figure ------------------------------------------------------------------------


def print_table(rng: np.random.Generator) -> None:
    print("== Pricing a digital good")
    n, eps = 1000, 0.5
    valuations = rng.uniform(0.0, 1.0, size=n)
    m = math.ceil(eps * n)
    prices = price_grid(m)
    scores = revenue(valuations, prices)
    probs = exponential_probabilities(scores, eps, 1.0)
    gamma = best_revenue(valuations)
    t = math.log(20.0)
    guarantee = gamma - (1.0 + 2.0 * math.log(m) + 2.0 * t) / eps
    print(f"n={n} eps={eps:g} m={m}: gamma={gamma:.2f}  OPT={scores.max():.2f}")
    print(f"guarantee at 95%={guarantee:.2f}  loss allowed={gamma - guarantee:.2f}")
    fifth = score_quantile(scores, probs, 0.05)
    print(f"mechanism: mean revenue={float(probs @ scores):.2f}  5th percentile={fifth:.2f}")
    print(
        f"loss at 5th percentile={gamma - fifth:.2f}  guarantee/actual loss={(gamma - guarantee) / (gamma - fifth):.1f}"
    )
    print("== PAC learning")
    alpha, beta, eps_pac = 0.05, 0.05, 1.0
    statistical, private = private_pac_samples(20 * math.log(2.0), alpha, beta, eps_pac)
    print(
        f"|C|=2^20 alpha={alpha:g} beta={beta:g} eps={eps_pac:g}: ln(2|C|/beta)={math.log(2 / beta) + 20 * math.log(2):.3f}"
    )
    print(f"sampling term={statistical:.0f}  privacy term={private:.0f}")


def plot_pricing(rng: np.random.Generator) -> Figure:
    n = 200
    valuations = rng.uniform(0.0, 1.0, size=n)
    prices = price_grid(200)
    scores = revenue(valuations, prices)
    fig, (top, bottom) = plt.subplots(2, 1, figsize=(7.5, 5.5), sharex=True)
    top.plot(prices, scores, color="C0")
    top.set_ylabel(r"revenue $s(v, p)$")
    top.grid(True, alpha=0.3)
    for i, eps in enumerate((0.2, 1.0)):
        probs = exponential_probabilities(scores, eps, 1.0)
        bottom.plot(prices, probs, color=f"C{i + 1}", label=rf"$\varepsilon = {eps:g}$")
    bottom.set_xlabel(r"price $p$")
    bottom.set_ylabel("probability of output")
    bottom.legend(fontsize=9)
    bottom.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=FIGURES)
    args = parser.parse_args()
    print_table(np.random.default_rng(args.seed))
    args.out.mkdir(parents=True, exist_ok=True)
    fig = plot_pricing(np.random.default_rng(args.seed + 1))
    fig.savefig(args.out / "em-pricing.svg", metadata={"Date": None})
    plt.close(fig)


if __name__ == "__main__":
    main()
