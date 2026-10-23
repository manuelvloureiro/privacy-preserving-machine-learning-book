"""Check the numbers in the course's Laplace-mechanism chapter and draw its figure.

    uv run python code/part02-dp-basics/laplace_mechanism.py
    uv run python code/part02-dp-basics/laplace_mechanism.py --seed 1 --out /tmp/figures

Prints the numbers quoted in book-course/part02-dp-basics/02-laplace-mechanism.md, then saves the
chapter's figure. The Laplace mechanism adds Lap(Δ₁/ε) noise to each coordinate of a statistic
whose ℓ1 sensitivity is Δ₁. Since Pr[|Y| ≥ tb] = e^{-t} for Y ~ Lap(b), the error of one answer
exceeds b ln(1/β) with probability β, and a union bound over k answers gives b ln(k/β).
"""

import argparse
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
matplotlib.rcParams["svg.hashsalt"] = "laplace_mechanism"  # stable element ids across reruns
matplotlib.rcParams["svg.fonttype"] = "none"  # keep text as text

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from numpy.typing import NDArray

# Figures are build output: _build/figures/<part>/, which the build copies into each book.
FIGURES = Path(__file__).resolve().parents[2] / "_build" / "figures" / Path(__file__).parent.name


# --- The Laplace mechanism -------------------------------------------------------------------


def laplace_density(x: NDArray[np.float64], center: float, b: float) -> NDArray[np.float64]:
    """The Lap(b) density shifted to `center`: e^{-|x - center|/b}/(2b)."""
    return np.exp(-np.abs(x - center) / b) / (2.0 * b)


def error_bound(b: float, beta: float, k: int = 1) -> float:
    """b ln(k/β): with probability at least 1 - β, all k Lap(b) noises are at most this in absolute value."""
    return b * math.log(k / beta)


def rr_error_bound(eps: float, n: int, beta: float) -> float:
    """coth(ε/2) √(ln(2/β)/(2n)), the randomized-response bound of the previous chapter."""
    return math.sqrt(math.log(2.0 / beta) / (2.0 * n)) / math.tanh(eps / 2.0)


def simulate_rr_errors(rng: np.random.Generator, eps: float, n: int, p: float, reps: int) -> NDArray[np.float64]:
    """|p̃ - p| for randomized response with truth probability e^ε/(1 + e^ε), on round(p n) ones."""
    ones = round(p * n)
    truth = math.exp(eps) / (1.0 + math.exp(eps))
    gamma = truth - 0.5
    reported = rng.binomial(ones, truth, size=reps) + rng.binomial(n - ones, 1.0 - truth, size=reps)
    return np.abs((reported / n - 0.5 + gamma) / (2.0 * gamma) - ones / n)


def simulate_max_errors(rng: np.random.Generator, b: float, k: int, reps: int) -> NDArray[np.float64]:
    """The largest of k independent |Lap(b)| noises, over `reps` runs."""
    return np.max(np.abs(rng.laplace(0.0, b, size=(reps, k))), axis=1)


# --- Table and figure ------------------------------------------------------------------------


def print_table(rng: np.random.Generator) -> None:
    eps, n, beta = 1.0, 10_000, 0.05
    print("== Mean of bits")
    lap = error_bound(1.0 / (eps * n), beta)
    rr = rr_error_bound(eps, n, beta)
    print(f"eps={eps:g} n={n} beta={beta:g}: laplace bound={lap:.2e}  sd={math.sqrt(2) / (eps * n):.2e}")
    print(f"randomized response bound={rr:.4f}  ratio={rr / lap:.0f}")
    print("== Counting queries and histograms")
    k = 100
    print(
        f"k={k}: counting bound={error_bound(k / eps, beta, k):.1f}  histogram bound={error_bound(2 / eps, beta, k):.2f}"
    )
    sim = float(np.quantile(simulate_max_errors(rng, 2.0 / eps, k, 20_000), 1 - beta))
    print(f"histogram, simulated 95th percentile of the largest error={sim:.2f}")


def plot_mechanism(rng: np.random.Generator) -> Figure:
    fig, (left, right) = plt.subplots(1, 2, figsize=(9.0, 3.8))
    eps = 1.0
    xs = np.linspace(-5.0, 6.0, 600)
    left.plot(xs, laplace_density(xs, 0.0, 1.0 / eps), color="C0", label="$M(D)$, $f(D) = 0$")
    left.plot(xs, laplace_density(xs, 1.0, 1.0 / eps), color="C1", label="$M(D')$, $f(D') = 1$")
    left.set_xlabel("output $y$")
    left.set_ylabel("density")
    left.set_title("Neighbouring outputs, $\\Delta_1 f = 1$, $\\varepsilon = 1$", fontsize=10)
    left.legend(fontsize=8, loc="upper left")
    ns = np.unique(np.logspace(2, 5, 13).astype(int))
    beta = 0.05
    rr = [float(np.quantile(simulate_rr_errors(rng, eps, int(n), 0.3, 2000), 1 - beta)) for n in ns]
    lap = [float(np.quantile(np.abs(rng.laplace(0.0, 1.0 / (eps * n), size=2000)), 1 - beta)) for n in ns]
    right.plot(ns, rr, "-o", color="C2", markersize=3, label="randomized response")
    right.plot(ns, lap, "-o", color="C0", markersize=3, label="Laplace mechanism")
    right.plot(
        ns,
        [error_bound(1.0 / (eps * n), beta) for n in ns],
        "--",
        color="C0",
        label="$\\ln(1/\\beta)/(\\varepsilon n)$",
    )
    right.set_xscale("log")
    right.set_yscale("log")
    right.set_xlabel("number of people $n$")
    right.set_ylabel("error (95th percentile)")
    right.set_title("Estimating a fraction, $\\varepsilon = 1$", fontsize=10)
    right.legend(fontsize=8)
    for ax in (left, right):
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
    fig = plot_mechanism(np.random.default_rng(args.seed + 1))
    fig.savefig(args.out / "lap-mechanism.svg", metadata={"Date": None})
    plt.close(fig)


if __name__ == "__main__":
    main()
