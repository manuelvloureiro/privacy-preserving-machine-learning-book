"""Check the bounds in the concentration chapter and draw its figures.

    uv run python book/part01-prerequisites/code/concentration.py
    uv run python book/part01-prerequisites/code/concentration.py --seed 1 --out /tmp/figures

Prints the table in 02-concentration.md (it matches exactly with the default seed),
then saves the chapter's figures.
"""

import argparse
import math
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
matplotlib.rcParams["svg.hashsalt"] = "concentration"  # stable element ids, so reruns don't churn the SVGs
matplotlib.rcParams["svg.fonttype"] = "none"  # keep text as text: smaller, diffable files

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from numpy.typing import NDArray

FIGURES = Path(__file__).resolve().parent.parent / "figures"


def normal_sf(t: float) -> float:
    """Pr[Z >= t] for Z ~ N(0, 1); erfc keeps full relative precision in the tail."""
    return 0.5 * math.erfc(t / math.sqrt(2))


def normal_pdf(t: float) -> float:
    return math.exp(-t * t / 2) / math.sqrt(2 * math.pi)


# --- Markov on a die roll -------------------------------------------------------------------


def markov_die(a: int) -> tuple[float, float]:
    """Exact Pr[X >= a] for a fair die, and the Markov bound E[X] / a."""
    return (7 - a) / 6, 3.5 / a


# --- Chernoff for a binomial: the bound as a function of λ ----------------------------------


def binomial_tail(n: int, p: float, k: int) -> float:
    """Exact Pr[S >= k] for S ~ Binomial(n, p)."""
    return sum(math.comb(n, j) * p**j * (1 - p) ** (n - j) for j in range(k, n + 1))


def chernoff_log_bound(lam: float, n: int, p: float, t: float) -> float:
    """ln of e^{-λt} E[e^{λS}] with the exact binomial MGF."""
    return -lam * t + n * math.log(1 + p * (math.exp(lam) - 1))


def chernoff_log_bound_relaxed(lam: float, n: int, p: float, t: float) -> float:
    """Same, after 1 + x <= e^x: the MGF is replaced by exp(μ(e^λ - 1))."""
    return -lam * t + n * p * (math.exp(lam) - 1)


# --- Mean of n Bernoulli(p): empirical tail vs the bounds -----------------------------------


@dataclass(frozen=True)
class TailBounds:
    markov: float
    chebyshev: float
    hoeffding: float
    chernoff: float


def bernoulli_mean_tail_bounds(n: int, p: float, t: float) -> TailBounds:
    """Bounds on Pr[mean - p >= t] for the mean of n independent Bernoulli(p)."""
    d = t / p  # multiplicative deviation for Chernoff
    return TailBounds(
        markov=p / (p + t),  # mean >= 0
        chebyshev=p * (1 - p) / (n * t**2),  # two-sided, so loose
        hoeffding=math.exp(-2 * n * t**2),  # range [0, 1]
        chernoff=math.exp(-(d**2) * n * p / (2 + d)),
    )


def empirical_bernoulli_mean_tail(rng: np.random.Generator, n: int, p: float, t: float, trials: int) -> float:
    means = rng.binomial(n, p, size=trials) / n
    return float(np.mean(means - p >= t))


# --- Gaussian tail bounds -------------------------------------------------------------------


@dataclass(frozen=True)
class GaussianTail:
    exact: float
    chebyshev: float  # two-sided bound 1/t², so also a one-sided one
    chernoff: float
    half_chernoff: float
    mills: float


def gaussian_tail(t: float) -> GaussianTail:
    return GaussianTail(
        exact=normal_sf(t),
        chebyshev=1 / t**2,
        chernoff=math.exp(-t * t / 2),
        half_chernoff=0.5 * math.exp(-t * t / 2),
        mills=normal_pdf(t) / t,
    )


# --- Maximum of k Laplace draws: max |X_i| <= b ln(k/β) w.p. >= 1-β ---------------------------


def laplace_max_bound(k: int, b: float, beta: float) -> float:
    return b * math.log(k / beta)


def empirical_laplace_max_failure(rng: np.random.Generator, k: int, b: float, beta: float, trials: int) -> float:
    """Fraction of trials in which max_i |X_i| exceeds the union bound."""
    x = rng.laplace(0, b, size=(trials, k))
    return float(np.mean(np.max(np.abs(x), axis=1) > laplace_max_bound(k, b, beta)))


# --- Sum of k Laplace draws: two tail regimes -----------------------------------------------


def laplace_sum_bound(t: float, k: int, b: float) -> float:
    """Sub-exponential bound exp(-min(t²/(8kb²), t/(4b)))."""
    return math.exp(-min(t * t / (8 * k * b * b), t / (4 * b)))


def laplace_sum_normal_approx(t: float, k: int, b: float) -> float:
    """CLT approximation: the sum has variance 2kb²."""
    return normal_sf(t / math.sqrt(2 * k * b * b))


def laplace_sums(rng: np.random.Generator, k: int, b: float, trials: int) -> NDArray[np.float64]:
    return rng.laplace(0, b, size=(trials, k)).sum(axis=1)


def tail_at(samples: NDArray[np.float64], t: float) -> float:
    return float(np.mean(samples >= t))


# --- Azuma: an adaptive coin game -----------------------------------------------------------


def adaptive_coin_martingale(
    rng: np.random.Generator, n: int, p_low: float, p_high: float, trials: int
) -> NDArray[np.float64]:
    """M_n = Σ (X_t - p_t), where coin t is the high coin iff flip t-1 came up heads.

    The flips are not independent and their sum is not binomial, but each increment
    has conditional mean 0 and lies in an interval of length 1, so Azuma applies.
    """
    m = np.zeros(trials)
    last = np.zeros(trials, dtype=bool)
    for _ in range(n):
        p = np.where(last, p_high, p_low)
        x = rng.random(trials) < p
        m += x - p
        last = x
    return m


def azuma_bound(t: float, n: int) -> float:
    return math.exp(-2 * t * t / n)


# --- The chapter's table --------------------------------------------------------------------


def print_table(rng: np.random.Generator) -> None:
    n, p, t = 200, 0.1, 0.05
    emp = empirical_bernoulli_mean_tail(rng, n, p, t, trials=200_000)
    b = bernoulli_mean_tail_bounds(n, p, t)
    print(
        f"[Bernoulli n={n} p={p} t={t}] empirical={emp:.2e}  "
        f"Markov={b.markov:.2e}  Chebyshev={b.chebyshev:.2e}  "
        f"Hoeffding={b.hoeffding:.2e}  Chernoff={b.chernoff:.2e}"
    )

    k, beta = 100, 0.05
    fail = empirical_laplace_max_failure(rng, k, 1.0, beta, trials=20_000)
    print(
        f"[Laplace max k={k} b=1] bound={laplace_max_bound(k, 1.0, beta):.2f}  "
        f"empirical failure={fail:.4f} (target <= {beta})"
    )

    g = gaussian_tail(3.0)
    print(
        f"[Gaussian t=3] exact={g.exact:.2e}  Chebyshev={g.chebyshev:.2e}  "
        f"Chernoff={g.chernoff:.2e}  half={g.half_chernoff:.2e}  Mills={g.mills:.2e}"
    )

    k = 10
    sums = laplace_sums(rng, k, 1.0, trials=1_000_000)
    for t in (10.0, 20.0):
        print(
            f"[Laplace sum k={k} b=1 t={t:g}] empirical={tail_at(sums, t):.2e}  "
            f"bound={laplace_sum_bound(t, k, 1.0):.2e}  normal approx={laplace_sum_normal_approx(t, k, 1.0):.2e}"
        )

    n, t = 200, 20.0
    m = adaptive_coin_martingale(rng, n, 0.3, 0.7, trials=100_000)
    print(f"[Adaptive coins n={n} t={t:g}] empirical={tail_at(m, t):.2e}  Azuma={azuma_bound(t, n):.2e}")


# --- Figures --------------------------------------------------------------------------------


def plot_markov_die(a: int = 5) -> Figure:
    rolls = list(range(1, 7))
    scaled_indicator = [a if r >= a else 0 for r in rolls]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar([r - 0.2 for r in rolls], rolls, width=0.4, label=f"X (average {sum(rolls) / 6:.2f})")
    ax.bar(
        [r + 0.2 for r in rolls],
        scaled_indicator,
        width=0.4,
        label=f"{a}·1[X ≥ {a}] (average {sum(scaled_indicator) / 6:.2f})",
    )
    ax.set(xticks=rolls, xlabel="die roll", ylabel="value", title="Pointwise ≤, so the averages are ordered too")
    ax.legend(loc="upper left")
    return fig


def plot_chernoff_lambda(n: int = 100, p: float = 0.1, t: int = 20) -> Figure:
    lams = [i / 100 for i in range(1, 251)]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(
        lams,
        [math.exp(chernoff_log_bound(lam, n, p, t)) for lam in lams],
        label=r"$e^{-\lambda t}\,\mathbb{E}[e^{\lambda S}]$",
    )
    ax.plot(lams, [math.exp(chernoff_log_bound_relaxed(lam, n, p, t)) for lam in lams], label=r"after $1 + x \leq e^x$")
    ax.axhline(binomial_tail(n, p, t), color="k", linestyle="--", label=f"exact Pr[S ≥ {t}]")
    ax.axvline(math.log(t / (n * p)), color="gray", linestyle=":", label="λ = ln(1 + δ)")
    ax.set(
        yscale="log",
        ylim=(1e-5, 2),
        xlabel="λ",
        ylabel="bound on Pr[S ≥ t]",
        title=f"Chernoff for S ~ Binomial({n}, {p}), t = {t}",
    )
    ax.legend()
    return fig


def plot_bernoulli(rng: np.random.Generator, n: int = 200, p: float = 0.1) -> Figure:
    ts = [0.01 * i for i in range(1, 11)]
    bounds = [bernoulli_mean_tail_bounds(n, p, t) for t in ts]
    empirical = [empirical_bernoulli_mean_tail(rng, n, p, t, trials=200_000) for t in ts]

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(ts, [b.markov for b in bounds], label="Markov")
    ax.plot(ts, [b.chebyshev for b in bounds], label="Chebyshev")
    ax.plot(ts, [b.hoeffding for b in bounds], label="Hoeffding")
    ax.plot(ts, [b.chernoff for b in bounds], label="Chernoff")
    ax.plot(ts, [e if e > 0 else math.nan for e in empirical], "ko", label="empirical")
    ax.set(
        yscale="log",
        ylim=(1e-6, 1.5),
        xlabel="deviation t",
        ylabel="Pr[mean - p ≥ t]",
        title=f"Mean of {n} Bernoulli({p})",
    )
    ax.legend()
    return fig


def plot_gaussian_tails() -> Figure:
    ts = [0.25 + i / 20 for i in range(116)]
    tails = [gaussian_tail(t) for t in ts]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(ts, [g.chebyshev for g in tails], label=r"Chebyshev $1/t^2$")
    ax.plot(ts, [g.chernoff for g in tails], label=r"Chernoff $e^{-t^2/2}$")
    ax.plot(ts, [g.half_chernoff for g in tails], label=r"$\frac{1}{2}e^{-t^2/2}$")
    ax.plot(ts, [g.mills for g in tails], label=r"Mills $\varphi(t)/t$")
    ax.plot(ts, [g.exact for g in tails], "k", linewidth=2, label="exact Pr[Z ≥ t]")
    ax.set(yscale="log", ylim=(1e-10, 2), xlabel="t", ylabel="Pr[Z ≥ t]", title="Gaussian tail bounds")
    ax.legend()
    return fig


def plot_laplace_max(rng: np.random.Generator, b: float = 1.0, beta: float = 0.05) -> Figure:
    ks = [1, 3, 10, 30, 100, 300, 1000]
    bounds = [laplace_max_bound(k, b, beta) for k in ks]
    failures = [empirical_laplace_max_failure(rng, k, b, beta, trials=20_000) for k in ks]

    fig, (left, right) = plt.subplots(1, 2, figsize=(10, 4))
    left.plot(ks, bounds, "o-")
    left.set(xscale="log", xlabel="draws k", ylabel="bound b ln(k/β)", title="The bound grows like ln k")
    right.plot(ks, failures, "o-", label="empirical failure rate")
    right.axhline(beta, color="k", linestyle="--", label=f"β = {beta}")
    right.set(
        xscale="log",
        ylim=(0, 2 * beta),
        xlabel="draws k",
        ylabel="Pr[max |X_i| > bound]",
        title="The union bound is nearly exact",
    )
    right.legend()
    return fig


def plot_laplace_sum(rng: np.random.Generator, k: int = 10, b: float = 1.0) -> Figure:
    sums = laplace_sums(rng, k, b, trials=1_000_000)
    ts = [i / 2 for i in range(1, 61)]
    empirical = [tail_at(sums, t) for t in ts]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(ts, [e if e > 0 else math.nan for e in empirical], "k", linewidth=2, label="empirical")
    ax.plot(ts, [laplace_sum_bound(t, k, b) for t in ts], label="sub-exponential bound")
    ax.plot(ts, [laplace_sum_normal_approx(t, k, b) for t in ts], label="normal approximation (CLT)")
    ax.axvline(2 * k * b, color="gray", linestyle=":", label="regime change t = 2kb")
    ax.set(yscale="log", ylim=(1e-7, 2), xlabel="t", ylabel="Pr[S ≥ t]", title=f"Sum of {k} independent Laplace({b:g})")
    ax.legend()
    return fig


def plot_azuma(rng: np.random.Generator, n: int = 200) -> Figure:
    m = adaptive_coin_martingale(rng, n, 0.3, 0.7, trials=100_000)
    ts = [float(i) for i in range(41)]
    empirical = [tail_at(m, t) for t in ts]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(ts, [e if e > 0 else math.nan for e in empirical], "k", linewidth=2, label="empirical")
    ax.plot(ts, [azuma_bound(t, n) for t in ts], label=r"Azuma $e^{-2t^2/n}$")
    ax.set(
        yscale="log", ylim=(1e-6, 2), xlabel="t", ylabel=r"Pr[$M_n \geq t$]", title=f"Adaptive coin game, n = {n} flips"
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

    print_table(np.random.default_rng(seed))

    rng = np.random.default_rng(seed + 1)  # separate stream, so the table stays reproducible
    out.mkdir(parents=True, exist_ok=True)
    figures = {
        "markov_die.svg": plot_markov_die(),
        "chernoff_lambda.svg": plot_chernoff_lambda(),
        "bernoulli_tails.svg": plot_bernoulli(rng),
        "gaussian_tails.svg": plot_gaussian_tails(),
        "laplace_max.svg": plot_laplace_max(rng),
        "laplace_sum.svg": plot_laplace_sum(rng),
        "azuma_adaptive.svg": plot_azuma(rng),
    }
    for name, fig in figures.items():
        fig.tight_layout()
        fig.savefig(out / name, metadata={"Date": None})  # no timestamp, so output is reproducible
        plt.close(fig)
        print(f"saved {out / name}")


if __name__ == "__main__":
    main()
