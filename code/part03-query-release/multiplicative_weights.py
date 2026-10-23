"""Check the numbers in the course's multiplicative-weights chapter and draw its figure.

    uv run python code/part03-query-release/multiplicative_weights.py
    uv run python code/part03-query-release/multiplicative_weights.py --seed 1 --out /tmp/figures

Prints the numbers quoted in book-course/part03-query-release/01-multiplicative-weights.md, then
saves the chapter's figure. The sample sizes are the sufficient conditions the chapter proves for
answering a set Q of linear queries to accuracy α: the Laplace mechanism with basic or advanced
composition, and private multiplicative weights. The figure runs the non-private multiplicative
weights algorithm for queries on a small domain.
"""

import argparse
import itertools
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
matplotlib.rcParams["svg.hashsalt"] = "multiplicative_weights"  # stable element ids across reruns
matplotlib.rcParams["svg.fonttype"] = "none"  # keep text as text

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from numpy.typing import NDArray

# Figures are build output: _build/figures/<part>/, which the build copies into each book.
FIGURES = Path(__file__).resolve().parents[2] / "_build" / "figures" / Path(__file__).parent.name


# --- Sufficient sample sizes -----------------------------------------------------------------


def laplace_basic_n(queries: float, alpha: float, eps: float, beta: float) -> float:
    """n ≥ |Q| ln(|Q|/β)/(αε): Laplace noise of scale |Q|/(εn) on every query, ε-DP."""
    return queries * math.log(queries / beta) / (alpha * eps)


def laplace_advanced_n(queries: float, alpha: float, eps: float, delta: float, beta: float) -> float:
    """n ≥ 2√(2|Q| ln(1/δ)) ln(|Q|/β)/(αε): Laplace noise with advanced composition, (ε, δ)-DP."""
    return 2.0 * math.sqrt(2.0 * queries * math.log(1.0 / delta)) * math.log(queries / beta) / (alpha * eps)


def pmw_rounds(domain: float, alpha: float) -> int:
    """T = ⌈64 ln|X|/α²⌉, the number of rounds of private multiplicative weights."""
    return math.ceil(64.0 * math.log(domain) / alpha**2)


def pmw_n(domain: float, queries: float, alpha: float, eps: float, delta: float, beta: float) -> float:
    """n ≥ 32 √(T ln(1/δ)) (ln|Q| + ln(2T/β))/(αε), the condition of the accuracy theorem."""
    rounds = pmw_rounds(domain, alpha)
    return (
        32.0
        * math.sqrt(rounds * math.log(1.0 / delta))
        * (math.log(queries) + math.log(2.0 * rounds / beta))
        / (alpha * eps)
    )


# --- Multiplicative weights for queries, without privacy -------------------------------------


def hypercube(d: int) -> NDArray[np.int64]:
    """All 2^d points of {0,1}^d, one per row."""
    return np.array(list(itertools.product((0, 1), repeat=d)), dtype=np.int64)


def marginal_queries(points: NDArray[np.int64]) -> NDArray[np.float64]:
    """One-way and two-way marginals: q(x) = 1[x_j = a] and q(x) = 1[x_j = a, x_k = b]."""
    d = points.shape[1]
    rows: list[NDArray[np.float64]] = []
    for j in range(d):
        for a in (0, 1):
            rows.append((points[:, j] == a).astype(np.float64))
    for j, k in itertools.combinations(range(d), 2):
        for a, b in itertools.product((0, 1), repeat=2):
            rows.append(((points[:, j] == a) & (points[:, k] == b)).astype(np.float64))
    return np.array(rows)


def correlated_histogram(rng: np.random.Generator, points: NDArray[np.int64], n: int) -> NDArray[np.float64]:
    """The histogram of n records: a hidden fair bit, copied into each feature with probability 0.8."""
    d = points.shape[1]
    hidden = rng.integers(0, 2, size=n)
    copy = rng.random((n, d)) < 0.8
    noise = rng.integers(0, 2, size=(n, d))
    records = np.where(copy, hidden[:, None], noise)
    place: NDArray[np.int64] = 2 ** np.arange(d - 1, -1, -1, dtype=np.int64)  # `points` lists the top bit first
    index: NDArray[np.int64] = records @ place  # each record's row number in `points`
    return np.bincount(index, minlength=len(points)) / n


def mw_for_queries(queries: NDArray[np.float64], p: NDArray[np.float64], alpha: float, max_updates: int) -> list[float]:
    """Run the algorithm with γ = α/2 and return the largest error before each update and at the end."""
    gamma = alpha / 2.0
    p_hat = np.full(len(p), 1.0 / len(p))
    truth = queries @ p
    errors: list[float] = []
    for _ in range(max_updates + 1):
        err = queries @ p_hat - truth
        j = int(np.argmax(np.abs(err)))
        errors.append(float(abs(err[j])))
        if abs(err[j]) <= alpha:
            break
        sign = 1.0 if err[j] >= 0 else -1.0
        w = p_hat * (1.0 - gamma * sign * queries[j])
        p_hat = w / w.sum()
    return errors


# --- Table and figure ------------------------------------------------------------------------


def print_table(rng: np.random.Generator) -> None:
    print("== Sample sizes")
    domain, alpha, eps, delta, beta = 2.0**20, 0.05, 1.0, 1e-6, 0.05
    print(f"T={pmw_rounds(domain, alpha)}  ln(1/delta)={math.log(1 / delta):.4f}")
    for queries in (1e6, 1e9):
        print(
            f"|Q|={queries:.0e}: basic={laplace_basic_n(queries, alpha, eps, beta):.3e}  "
            f"advanced={laplace_advanced_n(queries, alpha, eps, delta, beta):.3e}  "
            f"pmw={pmw_n(domain, queries, alpha, eps, delta, beta):.3e}"
        )
    print("== Multiplicative weights for queries")
    points = hypercube(8)
    queries = marginal_queries(points)
    p = correlated_histogram(rng, points, 2000)
    for a in (0.05, 0.02):
        errors = mw_for_queries(queries, p, a, 100_000)
        bound = 4 * math.log(len(points)) / a**2
        print(
            f"|Q|={len(queries)} alpha={a:g}: uniform error={errors[0]:.3f}  updates={len(errors) - 1}  bound={bound:.0f}"
        )


def plot_convergence(rng: np.random.Generator) -> Figure:
    points = hypercube(8)
    queries = marginal_queries(points)
    p = correlated_histogram(rng, points, 2000)
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    for i, a in enumerate((0.05, 0.02)):
        errors = mw_for_queries(queries, p, a, 100_000)
        color = f"C{i}"
        ax.plot(range(len(errors)), errors, "-", color=color, label=rf"$\alpha = {a:g}$, largest error")
        ax.axhline(a, color=color, linestyle="--", linewidth=1, label=rf"target $\alpha = {a:g}$")
    ax.set_xlabel("number of updates")
    ax.set_ylabel(r"$\max_{q \in \mathcal{Q}} |\langle q, p^t\rangle - q(D)|$")
    ax.set_ylim(0, None)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=FIGURES)
    args = parser.parse_args()
    print_table(np.random.default_rng(args.seed))
    args.out.mkdir(parents=True, exist_ok=True)
    fig = plot_convergence(np.random.default_rng(args.seed + 1))
    fig.savefig(args.out / "pmw-convergence.svg", metadata={"Date": None})
    plt.close(fig)


if __name__ == "__main__":
    main()
