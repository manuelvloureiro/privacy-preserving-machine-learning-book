"""Check the facts in the distributions chapter and draw its figures.

    uv run python book/part01-prerequisites/code/distributions.py
    uv run python book/part01-prerequisites/code/distributions.py --seed 1 --out /tmp/figures

Prints the tables in 01-distributions.md (they match exactly with the default seed),
then saves the chapter's figures.
"""

import argparse
import math
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
matplotlib.rcParams["svg.hashsalt"] = "distributions"  # stable element ids, so reruns don't churn the SVGs
matplotlib.rcParams["svg.fonttype"] = "none"  # keep text as text: smaller, diffable files

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from numpy.typing import NDArray

FIGURES = Path(__file__).resolve().parent.parent / "figures"

Sampler = Callable[[np.random.Generator, int], NDArray[np.float64]]

# --- Probability functions ---------------------------------------------------------------


def binomial_pmf(k: int, n: int, p: float) -> float:
    return math.comb(n, k) * p**k * (1 - p) ** (n - k)


def poisson_pmf(k: int, mu: float) -> float:
    return math.exp(-mu + k * math.log(mu) - math.lgamma(k + 1))


def slotted_wait_survival(t: float, n: int, b: float) -> float:
    """Pr[T_n > t] for T_n = G / n, with G ~ Geometric(1 / (b n)): the first ⌊n t⌋ slots are empty."""
    return (1 - 1 / (b * n)) ** math.floor(n * t)


def exponential_survival(t: float, b: float) -> float:
    return math.exp(-t / b)


def normal_pdf(x: float, mu: float = 0.0, sigma: float = 1.0) -> float:
    return math.exp(-((x - mu) ** 2) / (2 * sigma**2)) / (sigma * math.sqrt(2 * math.pi))


def normal_cdf(x: float, mu: float = 0.0, sigma: float = 1.0) -> float:
    return 0.5 * math.erfc(-(x - mu) / (sigma * math.sqrt(2)))


def laplace_pdf(x: float, mu: float = 0.0, b: float = 1.0) -> float:
    return math.exp(-abs(x - mu) / b) / (2 * b)


def laplace_cdf(x: float, mu: float = 0.0, b: float = 1.0) -> float:
    z = (x - mu) / b
    return 0.5 * math.exp(z) if z < 0 else 1 - 0.5 * math.exp(-z)


def normal_log_ratio(x: float, shift: float, sigma: float) -> float:
    """ln f(x) / f(x - shift) for f the N(0, σ²) density: linear in x, unbounded."""
    return (shift**2 - 2 * shift * x) / (2 * sigma**2)


def laplace_log_ratio(x: float, shift: float, b: float) -> float:
    """ln f(x) / f(x - shift) for f the Lap(0, b) density: bounded by |shift| / b."""
    return (abs(x - shift) - abs(x)) / b


# --- Moments: formula vs sample -----------------------------------------------------------


@dataclass(frozen=True)
class Family:
    name: str
    mean: float
    var: float
    sample: Sampler


FAMILIES = [
    Family("Bernoulli(0.3)", 0.3, 0.3 * 0.7, lambda rng, m: rng.binomial(1, 0.3, m).astype(np.float64)),
    Family("Binomial(20, 0.3)", 6.0, 20 * 0.3 * 0.7, lambda rng, m: rng.binomial(20, 0.3, m).astype(np.float64)),
    Family("Poisson(4)", 4.0, 4.0, lambda rng, m: rng.poisson(4.0, m).astype(np.float64)),
    Family("Exponential(b=2)", 2.0, 4.0, lambda rng, m: rng.exponential(2.0, m)),
    Family("Laplace(0, b=2)", 0.0, 8.0, lambda rng, m: rng.laplace(0.0, 2.0, m)),
    Family("Normal(1, 3²)", 1.0, 9.0, lambda rng, m: rng.normal(1.0, 3.0, m)),
]


def sample_moments(x: NDArray[np.float64]) -> tuple[float, float]:
    return float(np.mean(x)), float(np.var(x))


# --- Exponential as the limit of slotted waiting times --------------------------------------

WAIT_B, WAIT_T, WAIT_NS = 2.0, 4.0, [1, 5, 100]


def slotted_wait_samples(rng: np.random.Generator, n: int, b: float, m: int) -> NDArray[np.float64]:
    """m draws of T_n = G / n: the time of the first event when each 1/n-slot holds one with probability 1/(b n)."""
    return rng.geometric(1 / (b * n), m).astype(np.float64) / n


# --- Laplace as a difference of exponentials ----------------------------------------------


def difference_of_exponentials(rng: np.random.Generator, b: float, m: int) -> NDArray[np.float64]:
    return rng.exponential(b, m) - rng.exponential(b, m)


def max_cdf_gap(x: NDArray[np.float64], cdf: Callable[[float], float], grid: list[float]) -> float:
    """Largest gap between the empirical CDF of x and cdf, over the grid points."""
    xs = np.sort(x)
    return max(abs(float(np.searchsorted(xs, g, side="right")) / len(xs) - cdf(g)) for g in grid)


# --- Norm of an isotropic Gaussian vector ----------------------------------------------------


def scaled_gaussian_norms(rng: np.random.Generator, d: int, m: int) -> NDArray[np.float64]:
    """‖X‖ / √d for m draws of X ~ N(0, I_d)."""
    x = rng.normal(0.0, 1.0, size=(m, d))
    return np.linalg.norm(x, axis=1) / math.sqrt(d)


DIMS = [1, 10, 100, 1000, 10000]


# --- Central limit theorem: standardized sums of exponentials ---------------------------------

CLT_NS = [1, 2, 5, 30]


def standardized_erlang_pdf(z: float, n: int) -> float:
    """Density of (S - n) / √n for S = sum of n Exp(1) variables, which is Gamma(n, 1)."""
    x = n + z * math.sqrt(n)
    if x <= 0:
        return 0.0
    return math.sqrt(n) * math.exp((n - 1) * math.log(x) - x - math.lgamma(n))


def standardized_erlang_cdf(z: float, n: int) -> float:
    """Pr[(S - n) / √n ≤ z] = 1 - Σ_{k<n} e^{-x} x^k / k! at x = n + z√n."""
    x = n + z * math.sqrt(n)
    if x <= 0:
        return 0.0
    return 1 - sum(math.exp(k * math.log(x) - x - math.lgamma(k + 1)) for k in range(n))


def clt_kolmogorov_gap(n: int, grid: list[float]) -> float:
    """Largest gap between the CDF of the standardized sum and the N(0, 1) CDF, over the grid."""
    return max(abs(standardized_erlang_cdf(z, n) - normal_cdf(z)) for z in grid)


def standardized_exponential_sums(rng: np.random.Generator, n: int, m: int) -> NDArray[np.float64]:
    """m draws of (X_1 + ... + X_n - n) / √n with X_i ~ Exp(1)."""
    return (rng.exponential(1.0, size=(m, n)).sum(axis=1) - n) / math.sqrt(n)


# --- Tables --------------------------------------------------------------------------------


def print_tables(rng: np.random.Generator) -> None:
    print("[Moments, 1,000,000 samples]")
    for f in FAMILIES:
        mean, var = sample_moments(f.sample(rng, 1_000_000))
        print(f"  {f.name:<18} mean {f.mean:6.3f} vs {mean:6.3f}   var {f.var:6.3f} vs {var:6.3f}")

    gap = max_cdf_gap(difference_of_exponentials(rng, 1.0, 1_000_000), laplace_cdf, [i / 10 for i in range(-50, 51)])
    print(f"[Exp(1) - Exp(1) vs Lap(0, 1)] max CDF gap = {gap:.4f}")

    print("[‖X‖ / √d for X ~ N(0, I_d), 2,000 draws]")
    for d in DIMS:
        norms = scaled_gaussian_norms(rng, d, 2_000)
        print(f"  d={d:<6} mean {float(np.mean(norms)):.3f}   std {float(np.std(norms)):.3f}")

    print(f"[Slotted waiting time, b = {WAIT_B:g}: Pr[T_n > {WAIT_T:g}], and mean / variance of 1,000,000 draws]")
    for n in WAIT_NS:
        mean, var = sample_moments(slotted_wait_samples(rng, n, WAIT_B, 1_000_000))
        print(f"  n={n:<4} {slotted_wait_survival(WAIT_T, n, WAIT_B):.4f}   mean {mean:.3f}   var {var:.3f}")
    print(f"  Exp    {exponential_survival(WAIT_T, WAIT_B):.4f}")

    print("[CLT: (Exp(1) + ... + Exp(1) - n) / √n against N(0, 1)]")
    grid = [i / 100 for i in range(-500, 501)]
    for n in CLT_NS:
        print(f"  n={n:<3} skewness {2 / math.sqrt(n):.3f}   max CDF gap {clt_kolmogorov_gap(n, grid):.4f}")


# --- Figures --------------------------------------------------------------------------------


def plot_poisson_limit(mu: float = 5.0) -> Figure:
    ks = list(range(16))
    fig, ax = plt.subplots(figsize=(6, 4))
    for n, marker in [(10, "s"), (100, "^")]:
        ax.plot(
            ks, [binomial_pmf(k, n, mu / n) for k in ks], marker, linestyle="--", label=f"Binomial({n}, {mu / n:g})"
        )
    ax.plot(ks, [poisson_pmf(k, mu) for k in ks], "ko-", label=f"Poisson({mu:g})")
    xs = [i / 20 for i in range(15 * 20 + 1)]
    ax.plot(xs, [normal_pdf(x, mu, math.sqrt(mu)) for x in xs], color="gray", label=f"Normal({mu:g}, {mu:g})")
    ax.set(xlabel="k", ylabel="probability", title=f"Binomial(n, {mu:g}/n) → Poisson({mu:g})")
    ax.legend()
    return fig


def plot_exponential_limit(b: float = WAIT_B, ns: tuple[int, ...] = (1, 5)) -> Figure:
    ts = [i / 100 for i in range(10 * 100 + 1)]
    fig, ax = plt.subplots(figsize=(6, 4))
    for n in ns:
        ax.plot(
            ts,
            [slotted_wait_survival(t, n, b) for t in ts],
            linestyle="--",
            label=f"n = {n} slots per unit time, p = {1 / (b * n):g}",
        )
    ax.plot(ts, [exponential_survival(t, b) for t in ts], "k", label=f"$e^{{-t/{b:g}}}$")
    ax.set(xlabel="t", ylabel=r"$\Pr[T > t]$", title=f"Waiting time for the first event: slots → Exp(b = {b:g})")
    ax.legend()
    return fig


def plot_laplace_vs_normal(var: float = 2.0) -> Figure:
    sigma, b = math.sqrt(var), math.sqrt(var / 2)
    xs = [i / 50 for i in range(-400, 401)]
    normal = [normal_pdf(x, 0, sigma) for x in xs]
    laplace = [laplace_pdf(x, 0, b) for x in xs]

    fig, (left, right) = plt.subplots(1, 2, figsize=(10, 4))
    for ax in (left, right):
        ax.plot(xs, normal, label=f"Normal(0, {var:g})")
        ax.plot(xs, laplace, label=f"Laplace(0, b={b:g})")
        ax.set(xlabel="x")
    left.set(ylabel="density", title="Same variance: Laplace is more peaked")
    right.set(yscale="log", ylabel="density (log scale)", title="Log density: parabola vs straight lines")
    left.legend()
    return fig


def plot_difference_of_exponentials(rng: np.random.Generator, b: float = 1.0) -> Figure:
    x = difference_of_exponentials(rng, b, 200_000)
    xs = [i / 50 for i in range(-300, 301)]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(x, bins=120, range=(-6, 6), density=True, alpha=0.5, label="Exp(1) − Exp(1), samples")
    ax.plot(xs, [laplace_pdf(v, 0, b) for v in xs], "k", label="Laplace(0, 1) density")
    ax.set(xlabel="x", ylabel="density", title="A Laplace variable is a difference of exponentials")
    ax.legend()
    return fig


def plot_gaussian_norms(rng: np.random.Generator) -> Figure:
    fig, ax = plt.subplots(figsize=(6, 4))
    for d in DIMS:
        ax.hist(
            scaled_gaussian_norms(rng, d, 2_000),
            bins=80,
            range=(0, 2.5),
            density=True,
            histtype="step",
            label=f"d = {d}",
        )
    ax.set(xlabel="‖X‖ / √d", ylabel="density", title="The norm of X ~ N(0, I_d) concentrates at √d")
    ax.legend()
    return fig


def plot_clt_exponential(rng: np.random.Generator) -> Figure:
    zs = [i / 50 for i in range(-200, 251)]
    fig, axes = plt.subplots(2, 2, figsize=(8, 6), sharex=True, sharey=True)
    for ax, n in zip(axes.flat, CLT_NS):
        ax.hist(
            standardized_exponential_sums(rng, n, 50_000),
            bins=90,
            range=(-4, 5),
            density=True,
            alpha=0.4,
            label="samples",
        )
        ax.plot(zs, [standardized_erlang_pdf(z, n) for z in zs], label="exact density")
        ax.plot(zs, [normal_pdf(z) for z in zs], "k--", label="N(0, 1)")
        ax.set(ylim=(0, 1.05), title=f"n = {n}")
    for ax in axes[1]:
        ax.set(xlabel="z")
    for ax in axes[:, 0]:
        ax.set(ylabel="density")
    axes[0, 0].legend()
    fig.suptitle(r"Standardized sum $(X_1 + \cdots + X_n - n)/\sqrt{n}$ of $X_i \sim$ Exp(1)")
    return fig


def plot_shifted_log_ratio(shift: float = 1.0) -> Figure:
    xs = [i / 20 for i in range(-80, 81)]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(xs, [normal_log_ratio(x, shift, 1.0) for x in xs], label="Normal(0, 1)")
    ax.plot(xs, [laplace_log_ratio(x, shift, 1.0) for x in xs], label="Laplace(0, 1)")
    for level in (shift, -shift):
        ax.axhline(level, color="gray", linestyle=":")
    ax.set(
        xlabel="x",
        ylabel=f"ln f(x) / f(x − {shift:g})",
        title=f"Log ratio of a density and its copy shifted by {shift:g}",
    )
    ax.legend()
    return fig


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=FIGURES, help="directory for the figures")
    args = parser.parse_args()
    seed: int = args.seed
    out: Path = args.out

    print_tables(np.random.default_rng(seed))

    rng = np.random.default_rng(seed + 1)  # separate stream, so the tables stay reproducible
    out.mkdir(parents=True, exist_ok=True)
    figures = {
        "poisson_limit.svg": plot_poisson_limit(),
        "exponential_limit.svg": plot_exponential_limit(),
        "laplace_vs_normal.svg": plot_laplace_vs_normal(),
        "difference_of_exponentials.svg": plot_difference_of_exponentials(rng),
        "gaussian_norms.svg": plot_gaussian_norms(rng),
        "shifted_log_ratio.svg": plot_shifted_log_ratio(),
        "clt_exponential.svg": plot_clt_exponential(rng),
    }
    for name, fig in figures.items():
        fig.tight_layout()
        fig.savefig(out / name, metadata={"Date": None})  # no timestamp, so output is reproducible
        plt.close(fig)
        print(f"saved {out / name}")


if __name__ == "__main__":
    main()
