"""Check the numbers in the course's reconstruction chapter and draw its figure.

    uv run python code/part01-privacy-failures/reconstruction.py
    uv run python code/part01-privacy-failures/reconstruction.py --seed 1 --out /tmp/figures

Prints the numbers quoted in book-course/part01-privacy-failures/02-reconstruction.md, then saves
the chapter's figure. A curator holds secret bits d ∈ {0,1}^n and answers subset queries with error
at most E. The attacker asks random subsets, finds any x ∈ [0,1]^n consistent with the answers (a
linear feasibility problem, solved here by accelerated projected gradient) and rounds it at 1/2.
"""

import argparse
import math
import os
import sys
from pathlib import Path

# One BLAS thread unless the caller sets more: the figure scripts run in parallel under make -j.
for _var in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # for code/shared
from shared.figures import configure

configure("reconstruction")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from numpy.typing import NDArray

# Figures are build output: _build/figures/<part>/, which the build copies into each book.
FIGURES = Path(__file__).resolve().parents[2] / "_build" / "figures" / Path(__file__).parent.name


# --- Reconstruction from exact statistics ----------------------------------------------------


def age_triples(median: int, mean: int, max_age: int) -> list[tuple[int, int, int]]:
    """All whole-number ages a ≤ b ≤ c in [0, max_age] with the given median and mean."""
    return [(a, median, 3 * mean - median - a) for a in range(median + 1) if median <= 3 * mean - median - a <= max_age]


def count_sorted_triples(max_age: int) -> int:
    """Number of triples a ≤ b ≤ c in [0, max_age]: C(max_age + 3, 3)."""
    return math.comb(max_age + 3, 3)


def count_with_median(median: int, max_age: int) -> int:
    """Triples a ≤ b ≤ c in [0, max_age] with b = median."""
    return (median + 1) * (max_age - median + 1)


# --- Query counts in the theorems ------------------------------------------------------------


def queries_linear(n: int, beta: float) -> int:
    """m = ⌈(32/3)(n ln 2 + ln(1/β))⌉ random subset queries for the consistent-database attack."""
    return math.ceil(32 / 3 * (n * math.log(2) + math.log(1 / beta)))


def queries_lp(n: int, noise: float, beta: float) -> int:
    """m = ⌈(32/3)(n ln(2K + 1) + ln(1/β))⌉ with K = ⌈n/(2E)⌉, for the linear-programming attack."""
    k = math.ceil(n / (2 * noise))
    return math.ceil(32 / 3 * (n * math.log(2 * k + 1) + math.log(1 / beta)))


def queries_robust(n: int, beta: float) -> int:
    """m = ⌈(2048/9)(n ln 2 + ln(1/β))⌉ for the attack that tolerates a few arbitrary answers."""
    return math.ceil(2048 / 9 * (n * math.log(2) + math.log(1 / beta)))


# --- The pseudo-random queries ---------------------------------------------------------------


def decimal_digit(value: float, place: int) -> int:
    """The digit in the given decimal place after the point (place 1 is tenths)."""
    return math.floor(value * 10.0**place) % 10


def selected(identifier: int) -> bool:
    """floor(100 y + 0.5) = floor(100 y) for y = (2 · identifier)^0.7: the third decimal of y is at most 4."""
    y = (2 * identifier) ** 0.7
    return math.floor(100 * y + 0.5) == math.floor(100 * y)


# --- The linear-programming attack, simulated ------------------------------------------------


TOLERANCE = 0.01  # largest constraint violation accepted as feasible


def feasible_points(
    a: NDArray[np.float64], r: NDArray[np.float64], noise: NDArray[np.float64], iters: int, tol: float = TOLERANCE
) -> NDArray[np.float64]:
    """Column k: a point x ∈ [0,1]^n with |a x - r[:, k]| ≤ noise[k] + tol, by accelerated projected gradient.

    All columns share the query matrix a, so each step is one matrix product instead of one
    matrix-vector product per noise bound. Stops early once every column violates its constraints
    by at most tol, and after iters steps in any case.
    """
    step = 1.0 / (2.0 * float(np.linalg.norm(a, 2)) ** 2)
    # Single precision halves the cost; the answers are at most n, so rounding stays far below tol.
    a32, r32, noise32 = a.astype(np.float32), r.astype(np.float32), noise.astype(np.float32)
    x = np.full((a.shape[1], r.shape[1]), 0.5, dtype=np.float32)
    y = x.copy()
    t = 1.0
    for i in range(iters):
        if i % 25 == 0 and np.max(np.abs(a32 @ x - r32) - noise32) <= tol:
            break
        residual = a32 @ y - r32
        excess = np.sign(residual) * np.maximum(np.abs(residual) - noise32, 0.0)
        x_next = np.clip(y - step * 2.0 * (a32.T @ excess), 0.0, 1.0)
        t_next = (1.0 + math.sqrt(1.0 + 4.0 * t * t)) / 2.0
        y = x_next + (t - 1.0) / t_next * (x_next - x)
        x, t = x_next, t_next
    return x.astype(np.float64)


def attack_errors(
    rng: np.random.Generator, n: int, m: int, noises: NDArray[np.float64], iters: int = 1500
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """One dataset and one set of m random queries, attacked at each noise bound E in noises.

    The curator's error on query j is noise · u_j with u_j uniform on [-1, 1], so it is uniform on
    [-E, E] for each E. Returns, per noise bound, the fraction of bits the rounded point gets wrong
    and the largest constraint violation left.
    """
    d = rng.integers(0, 2, n).astype(np.float64)
    a = rng.integers(0, 2, (m, n)).astype(np.float64)
    u = rng.uniform(-1.0, 1.0, m)
    r = (a @ d)[:, None] + u[:, None] * noises[None, :]
    x = feasible_points(a, r, noises, iters)
    violation = np.max(np.abs(a @ x - r), axis=0) - noises
    c = (x >= 0.5).astype(np.float64)  # round at 1/2
    wrong = np.count_nonzero(c - d[:, None], axis=0) / n
    return wrong.astype(np.float64), violation.astype(np.float64)


# --- Table and figure ------------------------------------------------------------------------

SIZES = (100, 200, 400)
RATIOS = np.linspace(0.0, 3.0, 13)


def print_table(rng: np.random.Generator) -> None:
    print("== Reconstruction from exact statistics")
    triples = age_triples(35, 47, 125)
    print(
        f"sorted triples={count_sorted_triples(125)}  with median 35={count_with_median(35, 125)}  with mean 47 too={len(triples)}"
    )
    print(f"first={triples[0]}  last={triples[-1]}")
    print("== Asking every query")
    print(f"4/401={4 / 401:.5f}")
    print("== Random queries")
    print(f"(32/3) 2 ln 2={32 / 3 * 2 * math.log(2):.2f}")
    for n in (1000, 10_000):
        print(f"n={n} beta=0.01: linear attack m={queries_linear(n, 0.01)}  ratio={queries_linear(n, 0.01) / n:.2f}")
    print(
        f"n=10000 E=10 beta=0.01: LP attack m={queries_lp(10_000, 10.0, 0.01)}  ratio={queries_lp(10_000, 10.0, 0.01) / 10_000:.1f}"
    )
    print(f"robust: 3/128={3 / 128:.4f}  n=1000 beta=0.01 m={queries_robust(1000, 0.01)}")
    ratios = np.array([0.5, 2.0])
    for n in SIZES:
        wrong, violation = attack_errors(rng, n, 8 * n, ratios * math.sqrt(n))
        for ratio, w, v in zip(ratios, wrong, violation):
            print(f"n={n} m={8 * n} E/sqrt(n)={ratio:g}: wrong={w:.4f}  violation={v:.4f}")
    print("== Reconstruction in practice")
    y = (3 * 7) ** 0.5
    print(f"(3*7)^0.5={y:.8f}  second decimal={decimal_digit(y, 2)}")
    for identifier in (2000, 2001, 2002):
        y = (2 * identifier) ** 0.7
        print(
            f"id={identifier}: (2 id)^0.7={y:.5f}  third decimal={decimal_digit(y, 3)}  selected={selected(identifier)}"
        )
    kept = sum(selected(i) for i in range(2000, 3001))
    print(f"ids 2000..3000: kept={kept} of 1001  share={kept / 1001:.3f}")


def plot_attack(rng: np.random.Generator) -> Figure:
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    for i, n in enumerate(SIZES):
        reps = 1600 // n  # average over more runs for small n, where one run is noisy
        runs = [attack_errors(rng, n, 8 * n, RATIOS * math.sqrt(n)) for _ in range(reps)]
        wrong = np.mean(np.stack([run[0] for run in runs]), axis=0)
        worst = max(float(np.max(run[1])) for run in runs)
        print(f"figure n={n}: {reps} runs, largest violation={worst:.4f}  tolerance={TOLERANCE}")
        ax.plot(RATIOS, wrong, "-o", color=f"C{i}", markersize=3, label=rf"$n = {n}$, $m = {8 * n}$ ({reps} runs)")
    ax.axhline(0.5, color="0.5", linestyle=":", linewidth=1)
    ax.text(0.05, 0.47, "random guessing", color="0.4", fontsize=8, va="top")
    ax.set_xlabel(r"noise bound relative to $\sqrt{n}$, $E/\sqrt{n}$")
    ax.set_ylabel("fraction of secret bits wrong")
    ax.set_ylim(-0.02, 0.55)
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
    fig = plot_attack(np.random.default_rng(args.seed + 1))
    fig.savefig(args.out / "rec-attack.svg", metadata={"Date": None})
    plt.close(fig)


if __name__ == "__main__":
    main()
