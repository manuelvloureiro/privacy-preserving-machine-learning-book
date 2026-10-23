"""Check the numbers in the course's mean-estimation chapter and draw its figures.

    uv run python code/part04-learning-statistics/mean_estimation.py
    uv run python code/part04-learning-statistics/mean_estimation.py --seed 1 --out /tmp/figures

Prints the numbers quoted in book-course/part04-learning-statistics/04-mean-estimation.md, then
saves the chapter's two figures. Three ε-DP estimators of the mean of N(μ, 1) with |μ| ≤ R: the
clipped Laplace estimator, a histogram coarse step followed by it, and shrinking confidence
intervals followed by it. Then the clipped estimator on heavy-tailed data, as the clipping
norm C varies.
"""

import argparse
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
matplotlib.rcParams["svg.hashsalt"] = "mean_estimation"  # stable element ids across reruns
matplotlib.rcParams["svg.fonttype"] = "none"  # keep text as text

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from numpy.typing import NDArray

# Figures are build output: _build/figures/<part>/, which the build copies into each book.
FIGURES = Path(__file__).resolve().parents[2] / "_build" / "figures" / Path(__file__).parent.name

Array = NDArray[np.float64]


# --- Bounds from the chapter -----------------------------------------------------------------


def tail_radius(n: int, beta: float) -> float:
    """τ = √(2 ln(2n/β)): all n Gaussian points lie within τ of μ with probability ≥ 1 - β."""
    return math.sqrt(2.0 * math.log(2.0 * n / beta))


def sampling_term(n: int, beta: float) -> float:
    """√(2 ln(2/β)/n), the sampling error of the empirical mean of N(μ, 1)."""
    return math.sqrt(2.0 * math.log(2.0 / beta) / n)


def naive_width(n: int, eps: float, beta: float, r: float) -> float:
    """s(r) = √(2 ln(2/β)/n) + 2(r + τ) ln(1/β)/(εn), the error bound of the clipped estimator."""
    tau = tail_radius(n, beta)
    return sampling_term(n, beta) + 2.0 * (r + tau) * math.log(1.0 / beta) / (eps * n)


def shrink_steps(n: int, beta: float, big_r: float) -> int:
    """T = ⌈log2(R/τ)⌉, or 0 when R ≤ τ."""
    tau = tail_radius(n, beta)
    return max(0, math.ceil(math.log2(big_r / tau)))


def shrink_radii(n: int, eps: float, beta: float, big_r: float) -> list[float]:
    """The radii r_0 = R, r_t = s(r_{t-1}) computed with ε/(2T) per step."""
    steps = shrink_steps(n, beta, big_r)
    radii = [big_r]
    for _ in range(steps):
        radii.append(naive_width(n, eps / (2 * steps), beta, radii[-1]))
    return radii


def histogram_bins(n: int, beta: float, big_r: float) -> int:
    """m = ⌊R/τ⌋ + 2 bins of width 2τ covering [-R - τ, R + τ]."""
    return math.floor(big_r / tail_radius(n, beta)) + 2


def fine_bound(n: int, eps: float, beta: float, radius_over_tau: float) -> float:
    """Final error bound with half the budget, on an interval of radius c τ: s(cτ) at ε/2."""
    return naive_width(n, eps / 2.0, beta, radius_over_tau * tail_radius(n, beta))


# --- Estimators ------------------------------------------------------------------------------


def clipped_mean(rng: np.random.Generator, x: Array, centre: float, radius: float, eps: float, beta: float) -> float:
    """Clip to [c - r - τ, c + r + τ] and add Laplace noise of scale 2(r + τ)/(εn)."""
    n = x.size
    tau = tail_radius(n, beta)
    lo, hi = centre - radius - tau, centre + radius + tau
    return float(np.clip(x, lo, hi).mean() + rng.laplace(0.0, 2.0 * (radius + tau) / (eps * n)))


def naive_estimate(rng: np.random.Generator, x: Array, big_r: float, eps: float, beta: float) -> float:
    return clipped_mean(rng, x, 0.0, big_r, eps, beta)


def histogram_estimate(rng: np.random.Generator, x: Array, big_r: float, eps: float, beta: float) -> float:
    """Noisy argmax of a histogram with ε/2, then the clipped mean on the chosen interval with ε/2."""
    n = x.size
    tau = tail_radius(n, beta)
    m = histogram_bins(n, beta, big_r)
    left = -big_r - tau
    idx = np.floor((x - left) / (2.0 * tau)).astype(np.int64)
    idx = idx[(idx >= 0) & (idx < m)]
    counts = np.bincount(idx, minlength=m).astype(np.float64)
    noisy = counts + rng.laplace(0.0, 4.0 / eps, size=m)  # L1 sensitivity 2, budget ε/2
    j = int(np.argmax(noisy))
    centre = left + 2.0 * tau * j + tau
    return clipped_mean(rng, x, centre, 2.0 * tau, eps / 2.0, beta)


def shrinking_estimate(rng: np.random.Generator, x: Array, big_r: float, eps: float, beta: float) -> float:
    """T steps of ε/(2T) that recentre and shrink the interval, then the clipped mean with ε/2."""
    n = x.size
    radii = shrink_radii(n, eps, beta, big_r)
    steps = len(radii) - 1
    centre = 0.0
    for t in range(steps):
        centre = clipped_mean(rng, x, centre, radii[t], eps / (2 * steps), beta)
    return clipped_mean(rng, x, centre, radii[-1], eps / 2.0, beta)


def error_quantile(
    rng: np.random.Generator, method: str, n: int, big_r: float, eps: float, beta: float, reps: int, q: float
) -> float:
    mu = big_r / 3.0
    errors = np.empty(reps)
    for k in range(reps):
        x = rng.normal(mu, 1.0, size=n)
        if method == "naive":
            est = naive_estimate(rng, x, big_r, eps, beta)
        elif method == "histogram":
            est = histogram_estimate(rng, x, big_r, eps, beta)
        elif method == "shrinking":
            est = shrinking_estimate(rng, x, big_r, eps, beta)
        else:
            est = float(x.mean())
        errors[k] = abs(est - mu)
    return float(np.quantile(errors, q))


# --- Heavy tails -----------------------------------------------------------------------------


def heavy_sample(rng: np.random.Generator, mu: float, n: int) -> Array:
    """Student t with 3 degrees of freedom, scaled to variance 1, shifted to mean μ."""
    return mu + rng.standard_t(3.0, size=n) / math.sqrt(3.0)


def heavy_bound(n: int, eps: float, beta: float, tau: float) -> float:
    """1/τ + 1/√(βn) + 2(1 + τ) ln(1/β)/(εn), the error bound with probability ≥ 1 - 2β."""
    return 1.0 / tau + 1.0 / math.sqrt(beta * n) + 2.0 * (1.0 + tau) * math.log(1.0 / beta) / (eps * n)


def heavy_error_quantile(
    rng: np.random.Generator, n: int, eps: float, tau: float, reps: int, q: float, mu: float = 0.5
) -> float:
    errors = np.empty(reps)
    for k in range(reps):
        x = np.clip(heavy_sample(rng, mu, n), -1.0 - tau, 1.0 + tau)
        est = float(x.mean() + rng.laplace(0.0, 2.0 * (1.0 + tau) / (eps * n)))
        errors[k] = abs(est - mu)
    return float(np.quantile(errors, q))


# --- Table and figures -----------------------------------------------------------------------

N, EPS = 4000, 1.0
BETA = 0.002  # per event in the Gaussian theorems: at most T + 3 <= 25 events fail, so <= 5% overall
BETA_HEAVY = 0.05  # the heavy-tailed bound has two events
R_VALUES = (1.0, 10.0, 1e2, 1e3, 1e4, 1e5, 1e6)


def print_table(rng: np.random.Generator) -> None:
    print("== Gaussian mean, constants")
    tau = tail_radius(N, BETA)
    print(f"n={N} eps={EPS:g} beta={BETA:g}: tau={tau:.2f}  sampling={sampling_term(N, BETA):.4f}")
    big_r = 1e6
    print(f"R=1e6: naive bound={naive_width(N, EPS, BETA, big_r):.0f}")
    steps = shrink_steps(N, BETA, big_r)
    radii = shrink_radii(N, EPS, BETA, big_r)
    print(
        f"R=1e6: T={steps}  needs n>={32 * steps * math.log(1 / BETA) / EPS:.0f}  final radius/tau={radii[-1] / tau:.3f}"
    )
    print(f"R=1e6: shrinking bound={fine_bound(N, EPS, BETA, radii[-1] / tau):.3f}")
    m = histogram_bins(N, BETA, big_r)
    print(
        f"R=1e6: m={m}  needs n>{16 / EPS * math.log(m / BETA):.0f}  histogram bound={fine_bound(N, EPS, BETA, 2.0):.3f}"
    )
    print("== Gaussian mean, simulated 90th percentile of the error")
    for big_r in (10.0, 1e6):
        row = "  ".join(
            f"{meth}={error_quantile(rng, meth, N, big_r, EPS, BETA, 300, 0.9):.4f}"
            for meth in ("nonprivate", "naive", "histogram", "shrinking")
        )
        print(f"R={big_r:g}: {row}")
    print("== Heavy tails")
    best = math.sqrt(EPS * N / (2 * math.log(1 / BETA_HEAVY)))
    print(f"best tau=sqrt(eps n/(2 ln(1/beta)))={best:.1f}  bound={heavy_bound(N, EPS, BETA_HEAVY, best):.3f}")
    for tau in (3.0, best, math.sqrt(N / BETA_HEAVY)):
        sim = heavy_error_quantile(rng, N, EPS, tau, 1000, 0.9)
        print(f"tau={tau:.1f}: simulated 90th percentile={sim:.4f}")


def plot_gaussian(rng: np.random.Generator) -> Figure:
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    styles = {
        "naive": ("C0", "clipped Laplace on $[-R, R]$"),
        "histogram": ("C1", "histogram, then clipped Laplace"),
        "shrinking": ("C2", "shrinking intervals, then clipped Laplace"),
        "nonprivate": ("0.4", "empirical mean (not private)"),
    }
    for meth, (color, label) in styles.items():
        errs = [error_quantile(rng, meth, N, big_r, EPS, BETA, 100, 0.9) for big_r in R_VALUES]
        ax.plot(R_VALUES, errs, "-o", color=color, markersize=3, label=label)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"a priori bound $R$ on $|\mu|$")
    ax.set_ylabel(r"error $|\hat\mu - \mu|$ (90th percentile)")
    ax.legend(fontsize=8)
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    return fig


def plot_heavy(rng: np.random.Generator) -> Figure:
    taus = np.logspace(0, math.log10(math.sqrt(N / BETA_HEAVY)), 12)
    sim = [heavy_error_quantile(rng, N, EPS, float(t), 300, 0.9) for t in taus]
    bound = [heavy_bound(N, EPS, BETA_HEAVY, float(t)) for t in taus]
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.plot(taus, sim, "-o", color="C0", markersize=3, label="simulated, 90th percentile")
    ax.plot(taus, bound, "--", color="C1", label=r"bound, $\beta = 0.05$")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"clipping norm $C$")
    ax.set_ylabel(r"error $|\hat\mu - \mu|$")
    ax.legend(fontsize=8)
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
    rng = np.random.default_rng(args.seed + 1)
    for name, fig in (("mean-gaussian.svg", plot_gaussian(rng)), ("mean-heavy.svg", plot_heavy(rng))):
        fig.savefig(args.out / name, metadata={"Date": None})
        plt.close(fig)


if __name__ == "__main__":
    main()
