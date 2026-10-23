"""Check the numbers in the course's packing-lower-bounds chapter and draw its figure.

    uv run python code/part03-query-release/packing_lower_bounds.py
    uv run python code/part03-query-release/packing_lower_bounds.py --seed 1 --out /tmp/figures

Prints the numbers quoted in book-course/part03-query-release/04-packing-lower-bounds.md, then
saves the chapter's figure. Under ε-DP, the Laplace mechanism answers d one-way marginals to
ℓ∞ error α with probability 1/2 once n ≥ d ln(2d)/(αε), and a k-bin histogram once
n ≥ 2 ln(2k)/(αε). The packing argument shows that n ≥ ((d - 1) ln 2 - ε)/(2αε) and
n ≥ (ln(k/2) - ε)/(2αε) are necessary.
"""

import argparse
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
matplotlib.rcParams["svg.hashsalt"] = "packing_lower_bounds"  # stable element ids across reruns
matplotlib.rcParams["svg.fonttype"] = "none"  # keep text as text

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from numpy.typing import NDArray

# Figures are build output: _build/figures/<part>/, which the build copies into each book.
FIGURES = Path(__file__).resolve().parents[2] / "_build" / "figures" / Path(__file__).parent.name


# --- Upper bounds: the Laplace mechanism -----------------------------------------------------


def marginals_upper(d: int, alpha: float, eps: float, beta: float = 0.5) -> float:
    """n ≥ d ln(d/β)/(αε) makes every one-way marginal α-accurate with probability 1 - β."""
    return d * math.log(d / beta) / (alpha * eps)


def histogram_upper(k: int, alpha: float, eps: float, beta: float = 0.5) -> float:
    """n ≥ 2 ln(k/β)/(αε) makes every normalized count α-accurate with probability 1 - β."""
    return 2.0 * math.log(k / beta) / (alpha * eps)


def laplace_max_error(rng: np.random.Generator, k: int, scale: float, reps: int) -> NDArray[np.float64]:
    """max_j |Z_j| over `reps` draws of k independent Laplace(scale) variables."""
    return np.abs(rng.laplace(0.0, scale, size=(reps, k))).max(axis=1)


# --- Lower bounds: the packing argument ------------------------------------------------------


def mode_lower(k: int, eps: float) -> float:
    """n ≥ ln(k/2)/ε for a private mode that is correct with probability 1/2."""
    return math.log(k / 2.0) / eps


def marginals_lower(d: int, alpha: float, eps: float) -> float:
    """n ≥ ((d - 1) ln 2 - ε)/(2αε), for error at most α < 1/2 with probability 1/2."""
    return ((d - 1) * math.log(2.0) - eps) / (2.0 * alpha * eps)


def histogram_lower(k: int, alpha: float, eps: float) -> float:
    """n ≥ (ln(k/2) - ε)/(2αε), for error at most α < 1/2 with probability 1/2."""
    return (math.log(k / 2.0) - eps) / (2.0 * alpha * eps)


def packing_histograms(k: int, n: int, alpha: float) -> NDArray[np.float64]:
    """Row ℓ is h(D_ℓ): t = ⌊2αn⌋ + 1 copies of symbol ℓ and n - t copies of symbol 1 (row 0 is D_1)."""
    t = math.floor(2 * alpha * n) + 1
    h = np.zeros((k, k))
    h[0, 0] = 1.0
    for ell in range(1, k):
        h[ell, ell] = t / n
        h[ell, 0] = (n - t) / n
    return h


def balls_disjoint(centres: NDArray[np.float64], radius: float) -> bool:
    """Closed ℓ∞ balls of this radius are pairwise disjoint iff centres are > 2·radius apart."""
    m = centres.shape[0]
    return all(float(np.abs(centres[a] - centres[b]).max()) > 2 * radius for a in range(m) for b in range(a + 1, m))


# --- Table and figure ------------------------------------------------------------------------


def print_table(rng: np.random.Generator) -> None:
    print("== Mode")
    print(f"k=1000 eps=1: n >= {mode_lower(1000, 1.0):.2f}")
    print("== One-way marginals")
    d, alpha, eps = 1000, 0.1, 1.0
    lo, hi = marginals_lower(d, alpha, eps), marginals_upper(d, alpha, eps)
    print(f"d={d} alpha={alpha:g} eps={eps:g}: lower={math.ceil(lo)}  upper={math.ceil(hi)}  ratio={hi / lo:.1f}")
    print("== Histograms")
    k, alpha, eps = 10**6, 0.01, 1.0
    lo, hi = histogram_lower(k, alpha, eps), histogram_upper(k, alpha, eps)
    print(f"k={k} alpha={alpha:g} eps={eps:g}: lower={math.ceil(lo)}  upper={math.ceil(hi)}  ratio={hi / lo:.1f}")
    for big_k in (10**6, 10**12):
        ratio = histogram_upper(big_k, alpha, eps) / histogram_lower(big_k, alpha, eps)
        print(f"k={big_k:.0e}: upper/lower={ratio:.2f}")
    # The Laplace histogram at its upper bound: noise of scale 2/(nε) on each of k bins.
    small_k = 10**4
    n = math.ceil(histogram_upper(small_k, alpha, eps))
    errors = laplace_max_error(rng, small_k, 2.0 / (n * eps), 400)
    hits = int(np.sum(errors <= alpha))
    print(f"Laplace k={small_k} n={n}: max error <= alpha in {hits} of {errors.size} runs ({hits / errors.size:.0%})")
    h = packing_histograms(5, 20, 0.1)
    print(f"packing k=5 n=20 alpha=0.1: t/n={h[1, 1]:.2f}  disjoint={balls_disjoint(h, 0.1)}")


def draw_histogram(ax: Axes, h: NDArray[np.float64], alpha: float, title: str, color: str) -> None:
    k = h.shape[0]
    xs = list(range(1, k + 1))
    heights = [float(v) for v in h]
    ax.bar(xs, heights, width=0.35, color=color, alpha=0.8)
    for x, y in zip(xs, heights, strict=True):
        lo, hi = max(y - alpha, 0.0), min(y + alpha, 1.0)
        ax.fill_between([x - 0.3, x + 0.3], lo, hi, color=color, alpha=0.2, linewidth=0)
        ax.plot([x - 0.3, x + 0.3], [lo, lo], color=color, linewidth=1)
        ax.plot([x - 0.3, x + 0.3], [hi, hi], color=color, linewidth=1)
    ax.axhline(alpha, color="0.4", linestyle=":", linewidth=1)
    ax.set_xticks(xs)
    ax.set_xlabel("symbol $j$")
    ax.set_title(title, fontsize=10)
    ax.set_ylim(0, 1.05)


def plot_packing() -> Figure:
    k, n, alpha = 5, 20, 0.1
    h = packing_histograms(k, n, alpha)
    fig, axes = plt.subplots(1, 3, figsize=(9.0, 3.4), sharey=True)
    for ax, ell, color in zip(axes, (1, 3, 5), ("C0", "C1", "C2"), strict=True):
        draw_histogram(ax, h[ell - 1], alpha, rf"$D_{ell}$", color)
    axes[0].set_ylabel(r"fraction of records, with $\pm\alpha$ band")
    axes[0].text(4.6, alpha + 0.02, r"$\alpha$", color="0.3", fontsize=9)
    fig.tight_layout()
    return fig


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=FIGURES)
    args = parser.parse_args()
    print_table(np.random.default_rng(args.seed))
    args.out.mkdir(parents=True, exist_ok=True)
    fig = plot_packing()
    fig.savefig(args.out / "pack-histograms.svg", metadata={"Date": None})
    plt.close(fig)


if __name__ == "__main__":
    main()
