"""Check the Laplace and Gaussian mechanisms of the chapter and draw its figures.

    uv run python code/part03-dp-foundations/laplace_gaussian.py
    uv run python code/part03-dp-foundations/laplace_gaussian.py --seed 1 --out /tmp/figures

Prints the numbers quoted in 03-laplace-gaussian.md (they match exactly with the default seed),
then saves the chapter's figures.

Conventions: a mechanism adds noise to a query answer, M(D) = f(D) + Z. P is the output
distribution on D, Q the one on the neighbour D'. For one-dimensional pairs P is centred at 0 and
Q at the shift s = f(D') - f(D). For the Gaussian mechanism only μ = Δ₂/σ matters, for the Laplace
mechanism only t = Δ₁/b. δ(ε) is the hockey-stick divergence E_{e^ε}(P‖Q).
"""

import argparse
import itertools
import math
from collections.abc import Callable, Sequence
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
matplotlib.rcParams["svg.hashsalt"] = "laplace_gaussian"  # stable element ids, so reruns don't churn the SVGs
matplotlib.rcParams["svg.fonttype"] = "none"  # keep text as text: smaller, diffable files

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from numpy.typing import NDArray

# Figures are build output: _build/figures/<part>/, which the build copies into each book.
FIGURES = Path(__file__).resolve().parents[2] / "_build" / "figures" / Path(__file__).parent.name

Dataset = tuple[int, ...]  # a dataset is a tuple of records, each record an integer in a small domain
Query = Callable[[Dataset], list[float]]


def normal_cdf(x: float) -> float:
    """Φ(x); erfc keeps full relative precision in the lower tail."""
    return 0.5 * math.erfc(-x / math.sqrt(2))


def normal_pdf(x: float, mu: float = 0.0, sigma: float = 1.0) -> float:
    return math.exp(-((x - mu) ** 2) / (2 * sigma**2)) / (sigma * math.sqrt(2 * math.pi))


def laplace_pdf(x: float, mu: float = 0.0, b: float = 1.0) -> float:
    return math.exp(-abs(x - mu) / b) / (2 * b)


def bisect(f: Callable[[float], float], lo: float, hi: float, steps: int = 200) -> float:
    """A root of f in [lo, hi], given that f changes sign there."""
    f_lo = f(lo)
    for _ in range(steps):
        mid = 0.5 * (lo + hi)
        if (f(mid) > 0) == (f_lo > 0):
            lo, f_lo = mid, f(mid)
        else:
            hi = mid
    return 0.5 * (lo + hi)


def normal_quantile(p: float) -> float:
    """Φ⁻¹(p), by bisection on Φ."""
    return bisect(lambda x: normal_cdf(x) - p, -40.0, 40.0)


# --- Sensitivity ----------------------------------------------------------------------------


def count(data: Dataset) -> list[float]:
    return [float(len(data))]


def total(data: Dataset) -> list[float]:
    """Sum of the records; records are integers in [0, B]."""
    return [float(sum(data))]


def histogram(data: Dataset, bins: int) -> list[float]:
    return [float(sum(1 for x in data if x == j)) for j in range(bins)]


def threshold_counts(data: Dataset, d: int) -> list[float]:
    """The d counts #{x ≥ j} for j = 1..d: an empirical CDF, read off at d thresholds."""
    return [float(sum(1 for x in data if x >= j)) for j in range(1, d + 1)]


def datasets(domain: int, sizes: Sequence[int]) -> list[Dataset]:
    """Every dataset with records in {0, ..., domain-1} and a size in `sizes` (order kept, so with repeats)."""
    return [tuple(c) for n in sizes for c in itertools.product(range(domain), repeat=n)]


def neighbours(data: Dataset, domain: int, relation: str) -> list[Dataset]:
    """Add/remove: drop one record. Replace: change one record to another value."""
    if relation == "add/remove":
        return [data[:i] + data[i + 1 :] for i in range(len(data))]
    return [data[:i] + (v,) + data[i + 1 :] for i in range(len(data)) for v in range(domain) if v != data[i]]


def lp_norm(v: Sequence[float], p: float) -> float:
    return sum(abs(x) ** p for x in v) ** (1 / p)


def brute_force_sensitivity(f: Query, domain: int, sizes: Sequence[int], relation: str, p: float) -> float:
    """Δ_p f = max ‖f(D) - f(D')‖_p over every neighbouring pair with sizes in `sizes`."""
    best = 0.0
    for data in datasets(domain, sizes):
        fd = f(data)
        for other in neighbours(data, domain, relation):
            best = max(best, lp_norm([a - b for a, b in zip(fd, f(other), strict=True)], p))
    return best


# --- The Laplace mechanism ------------------------------------------------------------------


def laplace_scale(sens1: float, eps: float) -> float:
    """b = Δ₁/ε."""
    return sens1 / eps


def laplace_mechanism(
    rng: np.random.Generator, answer: NDArray[np.float64], sens1: float, eps: float
) -> NDArray[np.float64]:
    return answer + rng.laplace(0.0, laplace_scale(sens1, eps), size=answer.shape)


def laplace_privacy_loss(y: float, shift: float, b: float) -> float:
    """L(y) = ln p(y)/q(y) for P = Lap(0, b), Q = Lap(shift, b): (|y - shift| - |y|)/b."""
    return (abs(y - shift) - abs(y)) / b


def laplace_width(b: float, beta: float, d: int = 1) -> float:
    """With probability 1 - β, all d coordinates of Lap(b) noise are within b ln(d/β)."""
    return b * math.log(d / beta)


def laplace_delta(eps: float, t: float) -> float:
    """E_{e^ε}(Lap(0, 1)‖Lap(t, 1)) = (1 - e^{-(t - ε)/2})_+: the δ(ε) curve of a Laplace mechanism with t = Δ₁/b."""
    return max(0.0, 1.0 - math.exp(-(t - eps) / 2))


def event_ratio(rng: np.random.Generator, f_d: float, f_d2: float, b: float, cut: float, m: int) -> tuple[float, float]:
    """Simulated Pr[M(D) ≤ cut] and Pr[M(D') ≤ cut] for a one-dimensional Laplace mechanism."""
    a = float(np.mean(f_d + rng.laplace(0.0, b, size=m) <= cut))
    c = float(np.mean(f_d2 + rng.laplace(0.0, b, size=m) <= cut))
    return a, c


# --- The Gaussian mechanism: classic calibration --------------------------------------------


def classic_sigma(eps: float, delta: float, sens2: float = 1.0) -> float:
    """σ = Δ₂ √(2 ln(1.25/δ)) / ε (Dwork & Roth, Theorem A.1, stated for ε < 1)."""
    return sens2 * math.sqrt(2 * math.log(1.25 / delta)) / eps


def loss_tail(eps: float, mu: float) -> float:
    """P[L > ε] = Φ(μ/2 - ε/μ) for the Gaussian pair with μ = Δ₂/σ: the bad-event bound on δ."""
    return normal_cdf(mu / 2 - eps / mu)


def classic_proof_chain(eps: float, delta: float) -> tuple[float, float, float, float]:
    """The steps of the classic proof at σ = classic_sigma(ε, δ) and Δ₂ = 1.

    Returns (t, exact δ(ε), P[L > ε] = Φ(-t), Mills bound φ(t)/t). The proof shows Mills ≤ δ.
    """
    mu = 1 / classic_sigma(eps, delta)
    t = eps / mu - mu / 2
    return t, gaussian_delta(eps, mu), normal_cdf(-t), normal_pdf(t) / t


def classic_min_t(eps_max: float = 1.0) -> float:
    """Smallest t for which e^{ε/2}/(1.25 t √(2π)) ≤ 1 at ε = eps_max: the condition the proof needs."""
    return math.exp(eps_max / 2) / (1.25 * math.sqrt(2 * math.pi))


def classic_max_delta(eps_max: float = 1.0) -> float:
    """Largest δ for which the classic σ makes t = c - ε/(2c) ≥ classic_min_t at every ε ≤ eps_max."""
    t_min = classic_min_t(eps_max)
    c = (t_min + math.sqrt(t_min**2 + 2 * eps_max)) / 2  # root of c - ε/(2c) = t_min
    return 1.25 * math.exp(-(c**2) / 2)


# --- The Gaussian mechanism: the exact privacy curve ----------------------------------------


def gaussian_delta(eps: float, mu: float) -> float:
    """δ(ε) = Φ(μ/2 - ε/μ) - e^ε Φ(-μ/2 - ε/μ) with μ = Δ₂/σ (Balle & Wang 2018, Theorem 8)."""
    value = normal_cdf(mu / 2 - eps / mu) - math.exp(eps) * normal_cdf(-mu / 2 - eps / mu)
    return max(value, 0.0)


def gaussian_delta_integrated(eps: float, mu: float, h: float = 1e-3, half_width: float = 40.0) -> float:
    """δ(ε) = ∫ (p - e^ε q)_+ dy by Simpson's rule for p = N(0, 1), q = N(μ, 1): no closed form used."""
    n = int(2 * half_width / h) // 2 * 2  # an even number of intervals
    y = np.linspace(-half_width, half_width, n + 1)
    p = np.exp(-(y**2) / 2) / math.sqrt(2 * math.pi)
    q = np.exp(-((y - mu) ** 2) / 2) / math.sqrt(2 * math.pi)
    g = np.maximum(p - math.exp(eps) * q, 0.0)
    step = float(y[1] - y[0])
    return float(step / 3 * (g[0] + g[-1] + 4 * g[1:-1:2].sum() + 2 * g[2:-1:2].sum()))


def gaussian_delta_monte_carlo(rng: np.random.Generator, eps: float, mu: float, m: int) -> float:
    """δ(ε) = E_P[(1 - e^{ε - L})_+] with L ~ N(μ²/2, μ²), the privacy loss under P."""
    loss = rng.normal(mu**2 / 2, mu, size=m)
    return float(np.mean(np.maximum(1 - np.exp(eps - loss), 0.0)))


def analytic_mu(eps: float, delta: float) -> float:
    """The largest μ = Δ₂/σ with δ_μ(ε) ≤ δ, by bisection on ln μ (δ_μ(ε) increases with μ)."""
    log_mu = bisect(lambda lm: math.log(max(gaussian_delta(eps, math.exp(lm)), 1e-300)) - math.log(delta), -12.0, 8.0)
    return math.exp(log_mu)


def analytic_sigma(eps: float, delta: float, sens2: float = 1.0) -> float:
    """The smallest σ that makes the Gaussian mechanism (ε, δ)-DP (Balle & Wang's analytic calibration)."""
    return sens2 / analytic_mu(eps, delta)


def gaussian_epsilon(delta: float, mu: float) -> float:
    """The smallest ε with δ_μ(ε) ≤ δ: the curve read the other way."""
    if gaussian_delta(0.0, mu) <= delta:
        return 0.0
    return bisect(lambda e: gaussian_delta(e, mu) - delta, 0.0, 10 * mu * mu + 50.0)


# --- Laplace against Gaussian in many dimensions --------------------------------------------


def laplace_coordinate_rms(sens1: float, eps: float) -> float:
    """Root-mean-square error per coordinate of the Laplace mechanism: √2 Δ₁/ε."""
    return math.sqrt(2) * laplace_scale(sens1, eps)


def gaussian_coordinate_rms(sens2: float, eps: float, delta: float) -> float:
    """Root-mean-square error per coordinate of the analytically calibrated Gaussian mechanism: σ."""
    return analytic_sigma(eps, delta, sens2)


def crossover_dimension(eps: float, delta: float) -> float:
    """For d threshold counts (Δ₁ = d, Δ₂ = √d), Laplace is more accurate while d < (ε σ₁)²/2."""
    return (eps * analytic_sigma(eps, delta)) ** 2 / 2


def simulated_coordinate_rms(rng: np.random.Generator, noise: str, scale: float, d: int, trials: int) -> float:
    z = rng.laplace(0.0, scale, size=(trials, d)) if noise == "laplace" else rng.normal(0.0, scale, size=(trials, d))
    return float(np.sqrt(np.mean(z**2)))


# --- The chapter's numbers ------------------------------------------------------------------

EPS_TABLE: list[float] = [0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0]
DELTA = 1e-5


def print_table(rng: np.random.Generator) -> None:
    print("[Sensitivity, brute force over every dataset of up to 3 records]")
    bins = 4
    queries: list[tuple[str, Query, int]] = [
        ("count", count, 4),
        ("sum, records in [0, 3]", total, 4),
        ("histogram, 4 bins", lambda d: histogram(d, bins), 4),
        ("threshold counts, d = 3", lambda d: threshold_counts(d, 3), 4),
    ]
    for name, f, domain in queries:
        cells: list[str] = []
        for relation in ("add/remove", "replace"):
            l1 = brute_force_sensitivity(f, domain, (1, 2, 3), relation, 1)
            l2 = brute_force_sensitivity(f, domain, (1, 2, 3), relation, 2)
            cells.append(f"{relation}: l1={l1:.4f} l2={l2:.4f}")
        print(f"  {name}: " + "  ".join(cells))
    mean_sens = brute_force_sensitivity(lambda d: [sum(d) / len(d)], 4, (3,), "replace", 1)
    print(f"  mean of 3 records in [0, 3], replace: l1={mean_sens:.4f}")

    print("[Laplace mechanism]")
    eps, s = 1.0, 1.0
    grid = [i / 100 for i in range(-1000, 1001)]
    losses = [laplace_privacy_loss(y, s, laplace_scale(s, eps)) for y in grid]
    print(f"  count, eps=1: privacy loss ranges over [{min(losses):.4f}, {max(losses):.4f}]")
    a, c = event_ratio(rng, 0.0, 1.0, 1.0, 0.0, 1_000_000)
    print(f"  Pr[M(D) <= 0]={a:.4f}  Pr[M(D') <= 0]={c:.4f}  ratio={a / c:.4f}  e^eps={math.e:.4f}")
    b_small = 0.9 * laplace_scale(1.0, eps)
    print(f"  scale 0.9/eps: worst ratio e^(1/0.9)={math.exp(1 / b_small):.4f}, privacy loss {1 / b_small:.4f} > eps")
    n, true_count, eps = 10_000, 1234.0, 0.1
    b = laplace_scale(1.0, eps)
    noisy = laplace_mechanism(rng, np.full(1_000_000, true_count), 1.0, eps)
    err = np.abs(noisy - true_count)
    print(
        f"  count of {true_count:g} among n={n}, eps={eps:g}: b={b:g}  E|err|={b:g} (simulated {float(err.mean()):.3f})  "
        f"95% width b ln 20={laplace_width(b, 0.05):.2f} (simulated {float(np.quantile(err, 0.95)):.2f})  "
        f"relative error {laplace_width(b, 0.05) / true_count:.4f}"
    )
    b = laplace_scale(1.0, 1.0)
    noise = rng.laplace(0.0, b, size=(20_000, 20))
    fail = float(np.mean(np.max(np.abs(noise), axis=1) > laplace_width(b, 0.05, 20)))
    print(
        f"  histogram, 20 bins, eps=1: all bins within b ln(20/0.05)={laplace_width(b, 0.05, 20):.2f} "
        f"(simulated failure rate {fail:.4f})"
    )
    print(
        f"  mean age, n=1000, ages in [0, 100], replace: sensitivity 0.1000  b={laplace_scale(0.1, 1.0):.4f}  "
        f"95% width={laplace_width(laplace_scale(0.1, 1.0), 0.05):.4f}"
    )

    print("[Gaussian mechanism: classic calibration]")
    for e in (0.5, 1.0):
        t, exact, tail, mills = classic_proof_chain(e, DELTA)
        print(
            f"  eps={e:g} delta=1e-05: sigma={classic_sigma(e, DELTA):.4f}  t={t:.4f}  "
            f"exact delta={exact:.3e}  P[L>eps]={tail:.3e}  Mills={mills:.3e}"
        )
    print(
        f"  proof needs t >= {classic_min_t():.4f}, which holds for every eps <= 1 when delta <= {classic_max_delta():.4f}"
    )

    print("[Gaussian mechanism: the exact privacy curve]")
    worst = 0.0
    for e in (0.0, 0.5, 1.0, 2.0, 4.0, 8.0):
        for mu in (0.25, 0.5, 1.0, 2.0, 4.0):
            exact = gaussian_delta(e, mu)
            if exact > 1e-12:
                worst = max(worst, abs(gaussian_delta_integrated(e, mu) - exact) / exact)
    print(
        f"  closed form vs Simpson integration, 30 (eps, mu) pairs with delta > 1e-12: max relative error {worst:.1e}"
    )
    print(
        f"  mu=1: delta(0)={gaussian_delta(0.0, 1.0):.4f} (TV)  delta(1)={gaussian_delta(1.0, 1.0):.4f} "
        f"(integrated {gaussian_delta_integrated(1.0, 1.0):.4f}, Monte Carlo {gaussian_delta_monte_carlo(rng, 1.0, 1.0, 1_000_000):.4f})  "
        f"P[L>1]={loss_tail(1.0, 1.0):.4f}  e Q[L>1]={math.e * normal_cdf(-1.5):.4f}  delta(2)={gaussian_delta(2.0, 1.0):.4f}  delta(4)={gaussian_delta(4.0, 1.0):.2e}"
    )
    print(f"  sigma=2 (mu=0.5): P[L>1]={loss_tail(1.0, 0.5):.4f}  delta(1)={gaussian_delta(1.0, 0.5):.4f}")
    print(f"  mu=1: eps at delta=1e-5 is {gaussian_epsilon(DELTA, 1.0):.4f}")
    print("  calibration table, delta=1e-05, Delta_2=1:")
    for e in EPS_TABLE:
        cs, an = classic_sigma(e, DELTA), analytic_sigma(e, DELTA)
        print(
            f"    eps={e:g}: classic={cs:.4f}  analytic={an:.4f}  ratio={cs / an:.3f}  "
            f"true delta at classic={gaussian_delta(e, 1 / cs):.2e}"
        )
    e = 1.0
    sig = analytic_sigma(e, DELTA)
    print(f"  bisection check: delta at analytic sigma={gaussian_delta(e, 1 / sig):.3e}")
    big = bisect(lambda e: gaussian_delta(e, 1 / classic_sigma(e, DELTA)) - DELTA, 1.0, 50.0)
    print(f"  classic sigma first violates delta=1e-05 at eps={big:.2f}")
    print(
        f"  count, eps=0.1: analytic sigma={analytic_sigma(0.1, DELTA):.2f}  95% width={1.96 * analytic_sigma(0.1, DELTA):.2f}"
    )

    print("[Laplace against Gaussian, eps=1, delta=1e-05]")
    eps = 1.0
    sigma1 = analytic_sigma(eps, DELTA)
    print(
        f"  per coordinate, sensitivity 1: Laplace rms={laplace_coordinate_rms(1, eps):.4f}  Gaussian rms={sigma1:.4f}"
    )
    print(f"  threshold counts: Laplace better while d < {crossover_dimension(eps, DELTA):.2f}")
    for d in (1, 10, 100, 1000):
        lap = laplace_coordinate_rms(d, eps)
        gau = gaussian_coordinate_rms(math.sqrt(d), eps, DELTA)
        lap_sim = simulated_coordinate_rms(rng, "laplace", laplace_scale(d, eps), d, max(1, 200_000 // d))
        gau_sim = simulated_coordinate_rms(
            rng, "gauss", analytic_sigma(eps, DELTA, math.sqrt(d)), d, max(1, 200_000 // d)
        )
        print(
            f"    d={d}: Laplace rms={lap:.2f} (simulated {lap_sim:.2f})  Gaussian rms={gau:.2f} (simulated {gau_sim:.2f})  "
            f"ratio={lap / gau:.2f}"
        )


# --- Figures --------------------------------------------------------------------------------


def plot_privacy_loss(eps: float = 1.0, sigma: float = 2.0) -> Figure:
    """Densities of M(D) and M(D') and the privacy loss L(y), for a count with Laplace and Gaussian noise."""
    ys = [i / 100 for i in range(-500, 601)]
    b = laplace_scale(1.0, eps)
    mu = 1 / sigma
    fig, axes = plt.subplots(2, 2, figsize=(11, 6.5), sharex="col")
    lap_loss = [laplace_privacy_loss(y, 1.0, b) for y in ys]
    axes[0, 0].plot(ys, [laplace_pdf(y, 0.0, b) for y in ys], "C0", label="M(D) = 0 + Lap(1)")
    axes[0, 0].plot(ys, [laplace_pdf(y, 1.0, b) for y in ys], "C1", label="M(D') = 1 + Lap(1)")
    axes[0, 0].set(ylabel="density", title=rf"Laplace mechanism, $\varepsilon$ = {eps:g}, b = {b:g}")
    axes[1, 0].plot(ys, lap_loss, "k")
    axes[1, 0].set(xlabel="output y", ylabel=r"privacy loss $L(y)$", ylim=(-2.5, 2.5))
    gys = [y * sigma for y in ys]
    loss = [(mu * mu / 2 - mu * (y / sigma)) for y in gys]  # L(y) = (Δ² - 2Δy)/(2σ²) with Δ = 1
    axes[0, 1].plot(gys, [normal_pdf(y, 0.0, sigma) for y in gys], "C0", label=f"M(D) = 0 + N(0, {sigma:g}²)")
    axes[0, 1].plot(gys, [normal_pdf(y, 1.0, sigma) for y in gys], "C1", label=f"M(D') = 1 + N(0, {sigma:g}²)")
    cut = (mu * mu / 2 - eps) / mu * sigma  # L(y) > ε exactly when y < cut
    axes[0, 1].fill_between(
        gys,
        [normal_pdf(y, 0.0, sigma) for y in gys],
        where=[y < cut for y in gys],
        color="C3",
        alpha=0.4,
        label=rf"$L > \varepsilon$, probability {loss_tail(eps, mu):.4f}",
    )
    axes[0, 1].set(
        title=rf"Gaussian mechanism, $\sigma$ = {sigma:g}: $\delta({eps:g})$ = {gaussian_delta(eps, mu):.4f}"
    )
    axes[1, 1].plot(gys, loss, "k")
    axes[1, 1].fill_between(gys, eps, loss, where=[v > eps for v in loss], color="C3", alpha=0.4)
    axes[1, 1].set(xlabel="output y", ylim=(-2.5, 2.5))
    for ax in axes[1]:
        ax.axhline(eps, color="C3", linestyle="--", linewidth=1, label=r"$\pm\varepsilon$")
        ax.axhline(-eps, color="C3", linestyle="--", linewidth=1)
        ax.legend(fontsize=9, loc="upper right")
    for ax in axes[0]:
        ax.legend(fontsize=9, loc="upper right")
    return fig


def plot_delta_curves() -> Figure:
    """δ(ε) for the Gaussian mechanism at several σ, against the Laplace mechanism's curve."""
    eps_grid = [i / 50 for i in range(401)]
    fig, ax = plt.subplots(figsize=(10, 4.6))
    for k, sigma in enumerate((0.5, 1.0, 2.0, 4.0)):
        mu = 1 / sigma
        curve = [gaussian_delta(e, mu) for e in eps_grid]
        ax.plot(eps_grid, curve, f"C{k}", label=rf"Gaussian, $\sigma/\Delta_2$ = {sigma:g}")
        marks = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]
        ax.plot(marks, [gaussian_delta_integrated(e, mu) for e in marks], f"C{k}x", markersize=6)
    ax.plot(eps_grid, [laplace_delta(e, 1.0) for e in eps_grid], "k--", label=r"Laplace, $b/\Delta_1$ = 1")
    ax.plot([], [], "kx", label="numerical integration")
    ax.axhline(DELTA, color="0.5", linestyle=":", linewidth=1)
    ax.text(7.2, DELTA * 1.6, r"$\delta = 10^{-5}$", fontsize=9, color="0.4")
    ax.set(
        yscale="log",
        ylim=(1e-12, 1.2),
        xlim=(0, 8),
        xlabel=r"$\varepsilon$",
        ylabel=r"$\delta(\varepsilon)$",
        title=r"Each mechanism is $(\varepsilon, \delta(\varepsilon))$-DP for every $\varepsilon$ at once",
    )
    ax.legend(fontsize=9, loc="upper left", bbox_to_anchor=(1.02, 1))
    return fig


def plot_calibration() -> Figure:
    """Classic against analytic σ across ε at δ = 1e-5, and the true δ that the classic σ achieves."""
    eps_grid = [10 ** (i / 40) for i in range(-80, 41)]  # 0.01 .. 10
    fig, (left, right) = plt.subplots(1, 2, figsize=(12, 4.4))
    left.plot(eps_grid, [classic_sigma(e, DELTA) for e in eps_grid], "C1", label="classic (Dwork–Roth)")
    left.plot(eps_grid, [analytic_sigma(e, DELTA) for e in eps_grid], "C0", label="analytic (exact)")
    left.axvspan(1, 10, color="0.9", zorder=0)
    left.text(1.3, 300, "classic theorem\nassumes " + r"$\varepsilon < 1$", fontsize=9, color="0.35")
    left.set(
        xscale="log",
        yscale="log",
        xlabel=r"$\varepsilon$",
        ylabel=r"$\sigma / \Delta_2$",
        title=r"Noise needed for $(\varepsilon, 10^{-5})$-DP",
    )
    left.legend(fontsize=9, loc="lower left")
    right.plot(
        eps_grid,
        [gaussian_delta(e, 1 / classic_sigma(e, DELTA)) for e in eps_grid],
        "C1",
        label=r"true $\delta(\varepsilon)$ at the classic $\sigma$",
    )
    right.axhline(DELTA, color="k", linestyle="--", linewidth=1, label=r"target $\delta = 10^{-5}$")
    right.axvspan(1, 10, color="0.9", zorder=0)
    right.set(
        xscale="log",
        yscale="log",
        ylim=(1e-11, 1e-3),
        xlabel=r"$\varepsilon$",
        ylabel=r"$\delta$",
        title="What the classic calibration actually delivers",
    )
    right.legend(fontsize=9, loc="lower right")
    return fig


def plot_high_dim(rng: np.random.Generator, eps: float = 1.0) -> Figure:
    """Per-coordinate error of Laplace and Gaussian mechanisms as the number of released counts grows."""
    ds = [int(round(10 ** (i / 10))) for i in range(31)]
    ds = sorted(set(ds))
    sim_ds = [1, 10, 100, 1000]
    fig, (left, right) = plt.subplots(1, 2, figsize=(12, 4.4), sharey=True)

    def one(d: int) -> float:
        return 1.0

    panels: list[tuple[Axes, str, Callable[[int], float], Callable[[int], float]]] = [
        (left, "Histogram: each record in one bin", one, one),
        (right, "Threshold counts: each record in up to d counts", float, math.sqrt),
    ]
    for ax, title, sens1, sens2 in panels:
        ax.plot(ds, [laplace_coordinate_rms(sens1(d), eps) for d in ds], "C0", label=r"Laplace, $\Delta_1$")
        ax.plot(
            ds,
            [gaussian_coordinate_rms(sens2(d), eps, DELTA) for d in ds],
            "C1",
            label=r"Gaussian, $\Delta_2$, $\delta = 10^{-5}$",
        )
        ax.plot(
            sim_ds,
            [
                simulated_coordinate_rms(rng, "laplace", laplace_scale(sens1(d), eps), d, 20_000 // d + 1)
                for d in sim_ds
            ],
            "C0o",
            label="simulated",
        )
        ax.plot(
            sim_ds,
            [
                simulated_coordinate_rms(rng, "gauss", analytic_sigma(eps, DELTA, sens2(d)), d, 20_000 // d + 1)
                for d in sim_ds
            ],
            "C1o",
        )
        ax.set(xscale="log", yscale="log", xlabel="number of counts d", title=title)
        ax.legend(fontsize=9, loc="upper left")
    left.set_ylabel(rf"root-mean-square error per count, $\varepsilon$ = {eps:g}")
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
        "lapgauss_privacy_loss.svg": plot_privacy_loss(),
        "lapgauss_delta_curves.svg": plot_delta_curves(),
        "lapgauss_calibration.svg": plot_calibration(),
        "lapgauss_high_dim.svg": plot_high_dim(rng),
    }
    for name, fig in figures.items():
        fig.tight_layout()
        fig.savefig(out / name, metadata={"Date": None})  # no timestamp, so output is reproducible
        plt.close(fig)
        print(f"saved {out / name}")


if __name__ == "__main__":
    main()
