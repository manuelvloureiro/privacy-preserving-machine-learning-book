"""Check the divergences in the divergences chapter and draw its figures.

    uv run python code/part01-prerequisites/divergences.py
    uv run python code/part01-prerequisites/divergences.py --seed 1 --out /tmp/figures

Prints the numbers quoted in 03-divergences.md (they match exactly with the default seed),
then saves the chapter's figures.

Conventions: P is the first argument, Q the second; logs are natural (nats). For the continuous
pairs, P is a density and Q is its copy shifted by s: P = N(0, σ²), Q = N(s, σ²), and
P = Lap(0, b), Q = Lap(s, b). Only the standardized shift μ = s/σ or t = s/b matters.
"""

import argparse
import math
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from functools import cache
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # for code/shared
from shared.figures import configure
from shared.stats import laplace_pdf, normal_cdf, normal_pdf

configure("divergences")

import itertools

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from numpy.typing import NDArray

# Figures are build output: _build/figures/<part>/, which the build copies into each book.
FIGURES = Path(__file__).resolve().parents[2] / "_build" / "figures" / Path(__file__).parent.name

Dist = Sequence[float]  # a distribution on a finite set, as a list of probabilities

# The running example: a die loaded so that a six comes up half the time, against a fair die.
LOADED: list[float] = [0.1, 0.1, 0.1, 0.1, 0.1, 0.5]
FAIR: list[float] = [1 / 6] * 6


# --- Discrete divergences (the die) ---------------------------------------------------------


def kl(p: Dist, q: Dist) -> float:
    """D_KL(P‖Q) = Σ p ln(p/q), with 0 ln 0 = 0 and +∞ if P puts mass where Q does not."""
    total = 0.0
    for pi, qi in zip(p, q, strict=True):
        if pi == 0:
            continue
        if qi == 0:
            return math.inf
        total += pi * math.log(pi / qi)
    return total


def log_ratio_log_mgf(p: Dist, q: Dist, lam: float) -> float:
    """ψ(λ) = ln E_P[e^{λL}] = ln Σ p (p/q)^λ, where L = ln(p/q) is the log-likelihood ratio."""
    return math.log(sum(pi * (pi / qi) ** lam for pi, qi in zip(p, q, strict=True) if pi > 0))


def max_divergence(p: Dist, q: Dist) -> float:
    """D_∞(P‖Q) = ln max p/q over the support of P."""
    return math.log(max(pi / qi if qi > 0 else math.inf for pi, qi in zip(p, q, strict=True) if pi > 0))


def renyi(p: Dist, q: Dist, alpha: float) -> float:
    """D_α(P‖Q) = ψ(α - 1) / (α - 1); α = 1 is KL and α = ∞ the max-divergence."""
    if alpha == 1:
        return kl(p, q)
    if math.isinf(alpha):
        return max_divergence(p, q)
    return log_ratio_log_mgf(p, q, alpha - 1) / (alpha - 1)


def power_mean_ratio(p: Dist, q: Dist, s: float) -> float:
    """M_s = (E_P[r^s])^{1/s}, the power mean of order s of r = p/q under P; s = 0 is the geometric mean.

    D_α(P‖Q) = ln M_{α-1}: every Rényi divergence is the log of a power mean of the likelihood ratio.
    """
    if s == 0:
        return math.exp(kl(p, q))
    if math.isinf(s):
        return math.exp(max_divergence(p, q))
    return math.exp(log_ratio_log_mgf(p, q, s) / s)


def log_ratio_mean_sd(p: Dist, q: Dist) -> tuple[float, float]:
    """Mean and standard deviation of L = ln(p/q) under P: the drift and the spread of the evidence per draw."""
    pairs = [(pi, math.log(pi / qi)) for pi, qi in zip(p, q, strict=True) if pi > 0]
    mean = sum(pi * li for pi, li in pairs)
    return mean, math.sqrt(sum(pi * (li - mean) ** 2 for pi, li in pairs))


def evidence_paths(rng: np.random.Generator, source: Dist, p: Dist, q: Dist, n: int, paths: int) -> NDArray[np.float64]:
    """Cumulative evidence for P over Q, Σ_{i≤k} ln(p/q)(X_i) for k = 1..n, with X_i drawn i.i.d. from `source`.

    One row per path. With source = P the rows drift up at rate KL(P‖Q); with source = Q, down at rate KL(Q‖P).
    """
    log_r = np.log(np.asarray(p) / np.asarray(q))
    rolls = rng.choice(len(source), size=(paths, n), p=np.asarray(source))
    return np.cumsum(log_r[rolls], axis=1)


def f_divergence(p: Dist, q: Dist, f: Callable[[float], float]) -> float:
    """D_f(P‖Q) = Σ q f(p/q), for Q > 0 everywhere."""
    return sum(qi * f(pi / qi) for pi, qi in zip(p, q, strict=True))


def tv(p: Dist, q: Dist) -> float:
    return 0.5 * sum(abs(pi - qi) for pi, qi in zip(p, q, strict=True))


def hockey_stick(p: Dist, q: Dist, gamma: float) -> float:
    """E_γ(P‖Q) = Σ (p - γq)_+."""
    return sum(max(pi - gamma * qi, 0.0) for pi, qi in zip(p, q, strict=True))


def chi2(p: Dist, q: Dist) -> float:
    return f_divergence(p, q, lambda t: (t - 1) ** 2)


def hellinger_sq(p: Dist, q: Dist) -> float:
    """H² = Σ (√p - √q)², the f-divergence of (√t - 1)²."""
    return f_divergence(p, q, lambda t: (math.sqrt(t) - 1) ** 2)


def push_forward(p: Dist, labels: Sequence[int]) -> list[float]:
    """Distribution of Y = labels[X]: a deterministic processing of the outcome."""
    out = [0.0] * (max(labels) + 1)
    for pi, y in zip(p, labels, strict=True):
        out[y] += pi
    return out


def kl_from_hockey_sticks(p: Dist, q: Dist, steps: int = 200_000) -> float:
    """KL rebuilt from the E_γ curves: ∫_1^∞ E_γ(P‖Q)/γ dγ + ∫_0^1 E_{1/γ}(Q‖P) dγ (f'' = 1/t).

    Midpoint rule; E_γ(P‖Q) vanishes beyond γ = max p/q, which bounds the first integral.
    """
    top = max(pi / qi for pi, qi in zip(p, q, strict=True))
    h1 = (top - 1) / steps
    first = sum(hockey_stick(p, q, 1 + (i + 0.5) * h1) / (1 + (i + 0.5) * h1) for i in range(steps)) * h1
    h2 = 1 / steps
    second = sum(hockey_stick(q, p, 1 / ((i + 0.5) * h2)) for i in range(steps)) * h2
    return first + second


# --- Shifted Gaussians: P = N(0, σ²), Q = N(s, σ²), μ = s/σ ----------------------------------


def gaussian_kl(mu: float) -> float:
    return mu * mu / 2


def gaussian_renyi(alpha: float, mu: float) -> float:
    return alpha * mu * mu / 2


def gaussian_tv(mu: float) -> float:
    """The best event is {x < s/2}: TV = Φ(μ/2) - Φ(-μ/2) = 2Φ(μ/2) - 1."""
    return 2 * normal_cdf(abs(mu) / 2) - 1


def gaussian_hockey_stick(gamma: float, mu: float) -> float:
    """E_γ = Φ(μ/2 - c/μ) - γ Φ(-μ/2 - c/μ) with c = ln γ."""
    c = math.log(gamma)
    return normal_cdf(mu / 2 - c / mu) - gamma * normal_cdf(-mu / 2 - c / mu)


def gaussian_log_ratio(x: NDArray[np.float64], mu: float) -> NDArray[np.float64]:
    """L(x) = ln p(x)/q(x) for P = N(0, 1), Q = N(μ, 1)."""
    return (mu * mu - 2 * mu * x) / 2


def gaussian_upper_tail_exact(c: float, mu: float) -> float:
    """Q[L ≥ c] for the Gaussian pair: L ≥ c iff x ≤ μ/2 - c/μ, and x ~ N(μ, 1) under Q."""
    return normal_cdf(-mu / 2 - c / mu)


def gaussian_upper_tail_bound(c: float, mu: float) -> tuple[float, float]:
    """Best Chernoff bound Q[L ≥ c] ≤ exp(-αc + (α - 1) D_α) over α > 0, and the optimal α."""
    alpha = c / (mu * mu) + 0.5
    return math.exp(-alpha * c + (alpha - 1) * gaussian_renyi(alpha, mu)), alpha


def gaussian_tilted_pdf(x: float, alpha: float, s: float, sigma: float) -> float:
    """p^α q^{1-α} / ∫ p^α q^{1-α} for P = N(0, σ²), Q = N(s, σ²): the normal N((1 - α)s, σ²).

    This is where E_P[r^{α-1}] gets its mass, so it shows which x the Rényi divergence of order α looks at.
    """
    return normal_pdf(x, (1 - alpha) * s, sigma)


# --- Shifted Laplaces: P = Lap(0, b), Q = Lap(s, b), t = s/b ---------------------------------


def laplace_kl(t: float) -> float:
    return t + math.exp(-t) - 1


def laplace_renyi(alpha: float, t: float) -> float:
    """D_α from ∫ p^α q^{1-α} = α/(2α-1) e^{(α-1)t} + (α-1)/(2α-1) e^{-αt}  (α ≠ 1/2, 1)."""
    if alpha == 1:
        return laplace_kl(t)
    if math.isinf(alpha):
        return t
    if alpha == 0.5:
        return -2 * math.log((1 + t / 2) * math.exp(-t / 2))
    integral = alpha / (2 * alpha - 1) * math.exp((alpha - 1) * t) + (alpha - 1) / (2 * alpha - 1) * math.exp(
        -alpha * t
    )
    return math.log(integral) / (alpha - 1)


def laplace_tv(t: float) -> float:
    return 1 - math.exp(-t / 2)


def laplace_hockey_stick(gamma: float, t: float) -> float:
    """E_γ = 1 - e^{-(t - ln γ)/2} for 1 ≤ γ ≤ e^t, and 0 beyond."""
    return max(1 - math.exp(-(t - math.log(gamma)) / 2), 0.0)


def laplace_tilted_pdf(x: float, alpha: float, t: float) -> float:
    """p^α q^{1-α} / ∫ p^α q^{1-α} for P = Lap(0, 1), Q = Lap(t, 1), with the integral from the closed form."""
    integral = math.exp((alpha - 1) * laplace_renyi(alpha, t)) if alpha != 1 else 1.0
    return math.exp(-alpha * abs(x) - (1 - alpha) * abs(x - t)) / 2 / integral


def laplace_log_ratio(x: NDArray[np.float64], t: float) -> NDArray[np.float64]:
    """L(x) = |x - t| - |x| for P = Lap(0, 1), Q = Lap(t, 1)."""
    return np.abs(x - t) - np.abs(x)


# --- Monte Carlo checks ---------------------------------------------------------------------


def mc_mean(x: NDArray[np.float64]) -> float:
    return float(np.mean(x))


def mc_gaussian_kl(rng: np.random.Generator, mu: float, m: int) -> float:
    """KL = E_P[L], estimated from m draws of P = N(0, 1)."""
    return mc_mean(gaussian_log_ratio(rng.normal(0.0, 1.0, m), mu))


def mc_laplace_kl(rng: np.random.Generator, t: float, m: int) -> float:
    return mc_mean(laplace_log_ratio(rng.laplace(0.0, 1.0, m), t))


def mc_gaussian_hockey_stick(rng: np.random.Generator, gamma: float, mu: float, m: int) -> float:
    """E_γ = E_Q[(r - γ)_+], estimated from m draws of Q = N(μ, 1)."""
    r = np.exp(gaussian_log_ratio(rng.normal(mu, 1.0, m), mu))
    return mc_mean(np.maximum(r - gamma, 0.0))


def mean_test_accuracy(rng: np.random.Generator, mu: float, n: int, trials: int) -> float:
    """Draw n points from P = N(0, 1) or Q = N(μ, 1) with equal odds; guess P iff the sample mean < μ/2.

    That test thresholds the likelihood ratio of the whole sample, so it is the best one.
    """
    from_p = rng.normal(0.0, 1.0, size=(trials, n)).mean(axis=1) < mu / 2
    from_q = rng.normal(mu, 1.0, size=(trials, n)).mean(axis=1) >= mu / 2
    return 0.5 * mc_mean(from_p.astype(np.float64)) + 0.5 * mc_mean(from_q.astype(np.float64))


# --- Data processing: binning a Gaussian ----------------------------------------------------


def binned_gaussian(mu: float, width: float) -> tuple[list[float], list[float], list[float]]:
    """Bin edges and the bin probabilities under N(0, 1) and N(μ, 1) when x is rounded into bins of a given width.

    The bin edges are μ/2 + k·width, so width = ∞ is the one-bit test "x < μ/2".
    """
    if math.isinf(width):
        edges = [-math.inf, mu / 2, math.inf]
    else:
        k = math.ceil(12 / width)
        edges = [-math.inf] + [mu / 2 + j * width for j in range(-k, k + 1)] + [math.inf]
    p = [normal_cdf(hi) - normal_cdf(lo) for lo, hi in itertools.pairwise(edges)]
    q = [normal_cdf(hi - mu) - normal_cdf(lo - mu) for lo, hi in itertools.pairwise(edges)]
    return edges, p, q


def binned_gaussian_kl(mu: float, width: float) -> float:
    """KL between N(0, 1) and N(μ, 1) after rounding x into bins of the given width."""
    _, p, q = binned_gaussian(mu, width)
    return kl(p, q)


# --- Pinsker and Bretagnolle–Huber ----------------------------------------------------------


def pinsker_bound(kl_value: float) -> float:
    return math.sqrt(kl_value / 2)


def bretagnolle_huber_bound(kl_value: float) -> float:
    return math.sqrt(1 - math.exp(-kl_value))


# --- Forward and reverse KL: fitting one Gaussian to two bumps ------------------------------

MIX_CENTRE, MIX_SD = 2.0, 0.5
GRID = np.linspace(-10.0, 10.0, 4001)


def gaussian_log_pdf(x: NDArray[np.float64], mu: float, sigma: float) -> NDArray[np.float64]:
    return -((x - mu) ** 2) / (2 * sigma**2) - math.log(sigma * math.sqrt(2 * math.pi))


def mixture_log_pdf(x: NDArray[np.float64]) -> NDArray[np.float64]:
    """ln p for p = ½ N(-2, 0.5²) + ½ N(2, 0.5²), computed in logs so the tails don't underflow."""
    return np.logaddexp(gaussian_log_pdf(x, -MIX_CENTRE, MIX_SD), gaussian_log_pdf(x, MIX_CENTRE, MIX_SD)) - math.log(2)


def mixture_pdf(x: NDArray[np.float64]) -> NDArray[np.float64]:
    return np.exp(mixture_log_pdf(x))


def gaussian_pdf_array(x: NDArray[np.float64], mu: float, sigma: float) -> NDArray[np.float64]:
    return np.exp(gaussian_log_pdf(x, mu, sigma))


def integrate(y: NDArray[np.float64]) -> float:
    return float(np.trapezoid(y, GRID))


def forward_kl(mu: float, sigma: float) -> float:
    """KL(p‖q) for q = N(μ, σ²): what maximum likelihood minimizes."""
    log_p = mixture_log_pdf(GRID)
    return integrate(np.exp(log_p) * (log_p - gaussian_log_pdf(GRID, mu, sigma)))


def reverse_kl(mu: float, sigma: float) -> float:
    """KL(q‖p) for q = N(μ, σ²): what variational inference minimizes."""
    log_q = gaussian_log_pdf(GRID, mu, sigma)
    return integrate(np.exp(log_q) * (log_q - mixture_log_pdf(GRID)))


def forward_kl_fit() -> tuple[float, float]:
    """Moment matching: μ = E_p X = 0 and σ² = Var_p X = 2² + 0.5²."""
    return 0.0, math.sqrt(MIX_CENTRE**2 + MIX_SD**2)


@cache
def reverse_kl_fit() -> tuple[float, float]:
    """Grid search over μ ∈ [0, 3] (the problem is symmetric) and σ ∈ [0.1, 3], in steps of 0.01.

    Same integral as reverse_kl, vectorized over σ.
    """
    sigma_list = [j / 100 for j in range(10, 301)]
    sigmas: NDArray[np.float64] = np.array(sigma_list).reshape(-1, 1)
    log_p = mixture_log_pdf(GRID)
    best = (math.inf, 0.0, 0.0)
    for mu in [i / 100 for i in range(301)]:
        log_q = -((GRID - mu) ** 2) / (2 * sigmas**2) - np.log(sigmas * math.sqrt(2 * math.pi))
        integrals = np.atleast_1d(np.trapezoid(np.exp(log_q) * (log_q - log_p), GRID, axis=1))
        values = [float(v) for v in integrals]
        j = int(np.argmin(values))
        if values[j] < best[0]:
            best = (values[j], mu, sigma_list[j])
    return best[1], best[2]


def bisect(f: Callable[[float], float], lo: float, hi: float, steps: int = 60) -> float:
    """A root of f in [lo, hi], given that f changes sign there."""
    for _ in range(steps):
        mid = (lo + hi) / 2
        if (f(lo) < 0) == (f(mid) < 0):
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


# --- The chapter's numbers ------------------------------------------------------------------


@dataclass(frozen=True)
class DiePair:
    name: str
    p: list[float]
    q: list[float]


def print_table(rng: np.random.Generator) -> None:
    pq, qp = DiePair("P‖Q", LOADED, FAIR), DiePair("Q‖P", FAIR, LOADED)

    print("[Die: P = loaded (six w.p. 0.5), Q = fair]")
    print(f"  likelihood ratio p/q = {[round(pi / qi, 3) for pi, qi in zip(LOADED, FAIR)]}")
    for d in (pq, qp):
        orders = "  ".join(f"D_{a:g}={renyi(d.p, d.q, a):.4f}" for a in (0.5, 1, 2, 5, 10, math.inf))
        print(f"  {d.name}: KL={kl(d.p, d.q):.4f}  {orders}")
    print(
        f"  TV={tv(LOADED, FAIR):.4f}  chi2={chi2(LOADED, FAIR):.4f}  "
        f"Hellinger^2={hellinger_sq(LOADED, FAIR):.4f}  "
        f"E_2={hockey_stick(LOADED, FAIR, 2):.4f}  E_3={hockey_stick(LOADED, FAIR, 3):.4f}  "
        f"Pinsker={pinsker_bound(kl(LOADED, FAIR)):.4f}"
    )
    means = "  ".join(f"M_{s:g}={power_mean_ratio(LOADED, FAIR, s):.4f}" for s in (-1, 0, 1, math.inf))
    print(f"  power means of r under P: {means}")
    for d in (pq, qp):
        drift, spread = log_ratio_mean_sd(d.p, d.q)
        print(
            f"  evidence per roll, {d.name}: mean={drift:.4f}  sd={spread:.4f}  (sd/mean)^2={(spread / drift) ** 2:.2f}"
        )
    print(f"  KL rebuilt from hockey-stick curves = {kl_from_hockey_sticks(LOADED, FAIR):.4f}")
    parity = [1, 0, 1, 0, 1, 0]  # 1 = odd
    six = [0, 0, 0, 0, 0, 1]
    print(
        f"  after processing: parity KL={kl(push_forward(LOADED, parity), push_forward(FAIR, parity)):.4f}  "
        f"is-six KL={kl(push_forward(LOADED, six), push_forward(FAIR, six)):.4f}"
    )

    mu, m = 1.0, 1_000_000
    print(f"[Gaussian pair, μ = s/σ = {mu:g}]")
    print(
        f"  KL={gaussian_kl(mu):.4f} (Monte Carlo {mc_gaussian_kl(rng, mu, m):.4f})  "
        f"D_2={gaussian_renyi(2, mu):.4f}  D_10={gaussian_renyi(10, mu):.4f}"
    )
    print(
        f"  TV={gaussian_tv(mu):.4f}  best single-shot accuracy={0.5 + 0.5 * gaussian_tv(mu):.4f}  "
        f"Pinsker={pinsker_bound(gaussian_kl(mu)):.4f}"
    )
    print(
        f"  E_2={gaussian_hockey_stick(2, mu):.4f} (Monte Carlo {mc_gaussian_hockey_stick(rng, 2, mu, m):.4f})  "
        f"E_4={gaussian_hockey_stick(4, mu):.4f}"
    )
    c = 2.0
    bound, alpha = gaussian_upper_tail_bound(c, mu)
    print(
        f"  Q[L >= {c:g}]: exact={gaussian_upper_tail_exact(c, mu):.4f}  Markov e^-c={math.exp(-c):.4f}  "
        f"Renyi-Chernoff={bound:.4f} at alpha={alpha:g}"
    )
    print(
        f"  binned KL: width 0.5={binned_gaussian_kl(mu, 0.5):.4f}  width 1={binned_gaussian_kl(mu, 1):.4f}  "
        f"width 2={binned_gaussian_kl(mu, 2):.4f}  one bit={binned_gaussian_kl(mu, math.inf):.4f}"
    )

    t = 1.0
    print(f"[Laplace pair, t = s/b = {t:g}; Gaussian with the same variance has μ = {t / math.sqrt(2):.4f}]")
    g = t / math.sqrt(2)
    print(
        f"  Laplace:  KL={laplace_kl(t):.4f} (Monte Carlo {mc_laplace_kl(rng, t, m):.4f})  "
        f"D_0.5={laplace_renyi(0.5, t):.4f}  D_2={laplace_renyi(2, t):.4f}  D_10={laplace_renyi(10, t):.4f}  "
        f"D_inf={laplace_renyi(math.inf, t):.4f}  TV={laplace_tv(t):.4f}  E_2={laplace_hockey_stick(2, t):.4f}  "
        f"Pinsker={pinsker_bound(laplace_kl(t)):.4f}"
    )
    print(
        f"  Gaussian: KL={gaussian_kl(g):.4f}  D_0.5={gaussian_renyi(0.5, g):.4f}  D_2={gaussian_renyi(2, g):.4f}  "
        f"D_10={gaussian_renyi(10, g):.4f}  D_inf=inf  TV={gaussian_tv(g):.4f}  E_2={gaussian_hockey_stick(2, g):.4f}"
    )

    crossing = bisect(lambda a: gaussian_renyi(a, g) - laplace_renyi(a, t), 1.5, 10.0)
    print(f"  Gaussian D_alpha overtakes Laplace D_alpha at alpha={crossing:.2f}")

    mu, trials = 0.1, 20_000
    print(f"[n samples, μ = {mu:g} per sample]  one sample: TV={gaussian_tv(mu):.4f}")
    for n in (100, 400):
        kl_n = n * gaussian_kl(mu)
        exact = gaussian_tv(mu * math.sqrt(n))
        print(
            f"  n={n}: KL={kl_n:.2f}  TV={exact:.4f}  accuracy={0.5 + 0.5 * exact:.4f} "
            f"(simulated {mean_test_accuracy(rng, mu, n, trials):.4f})  "
            f"Pinsker={pinsker_bound(kl_n):.4f}  BH={bretagnolle_huber_bound(kl_n):.4f}"
        )

    crossing = bisect(lambda k: bretagnolle_huber_bound(k) - pinsker_bound(k), 0.5, 3.0)
    print(f"  Bretagnolle-Huber beats Pinsker for KL > {crossing:.2f}")

    fwd, rev = forward_kl_fit(), reverse_kl_fit()
    print("[Fitting N(μ, σ²) to ½N(-2, 0.5²) + ½N(2, 0.5²)]")
    print(
        f"  forward KL fit: mu={fwd[0]:.2f} sigma={fwd[1]:.2f}  KL(p‖q)={forward_kl(*fwd):.3f}  KL(q‖p)={reverse_kl(*fwd):.3f}"
    )
    print(
        f"  reverse KL fit: mu={rev[0]:.2f} sigma={rev[1]:.2f}  KL(p‖q)={forward_kl(*rev):.3f}  KL(q‖p)={reverse_kl(*rev):.3f}"
    )


# --- Figures --------------------------------------------------------------------------------


def plot_forward_reverse_kl() -> Figure:
    fwd, rev = forward_kl_fit(), reverse_kl_fit()
    x = GRID[(GRID >= -6) & (GRID <= 6)]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.fill_between(x, mixture_pdf(x), color="0.85", label="data p: two bumps")
    ax.plot(
        x,
        gaussian_pdf_array(x, *fwd),
        label=rf"min KL(p‖q): $\mu$ = {fwd[0]:.2f}, $\sigma$ = {fwd[1]:.2f} (covers both)",
    )
    ax.plot(
        x, gaussian_pdf_array(x, *rev), label=rf"min KL(q‖p): $\mu$ = {rev[0]:.2f}, $\sigma$ = {rev[1]:.2f} (picks one)"
    )
    ax.set(xlabel="x", ylabel="density", ylim=(0, 0.95), title="Fitting one Gaussian q to two bumps")
    ax.legend(loc="upper left", fontsize=9)
    return fig


def plot_evidence_walk(rng: np.random.Generator, n: int = 60, paths: int = 8) -> Figure:
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ks = np.arange(1, n + 1)
    runs = (
        (LOADED, kl(LOADED, FAIR), "C0", "rolls of the loaded die"),
        (FAIR, -kl(FAIR, LOADED), "C1", "rolls of the fair die"),
    )
    for source, drift, color, name in runs:
        walks = evidence_paths(rng, source, LOADED, FAIR, n, paths)
        for j in range(paths):
            ax.plot(ks, walks[j], color=color, alpha=0.35, linewidth=1, label=name if j == 0 else None)
        ax.plot(
            ks, drift * ks, color=color, linestyle="--", linewidth=2, label=rf"average: ${drift:+.3f}$ nats per roll"
        )
    ax.axhline(0, color="0.6", linewidth=0.8)
    ax.set(
        xlabel="number of rolls n",
        ylabel=r"evidence for loaded: $\sum_i \ln r(x_i)$ (nats)",
        title="Evidence accumulates at rate KL(loaded‖fair), or falls at rate KL(fair‖loaded)",
    )
    ax.legend(fontsize=9, loc="upper left")
    return fig


def plot_renyi_tilt(t: float = 1.0) -> Figure:
    sigma = math.sqrt(2.0)  # same variance as Lap(0, 1)
    xs = [i / 50 for i in range(-500, 251)]
    alphas = (1.0, 2.0, 4.0, 8.0)
    fig, (left, right) = plt.subplots(1, 2, figsize=(11, 4))
    for k, a in enumerate(alphas):
        left.plot(xs, [gaussian_tilted_pdf(x, a, t, sigma) for x in xs], color=f"C{k}", label=rf"$\alpha$ = {a:g}")
        right.plot(xs, [laplace_tilted_pdf(x, a, t) for x in xs], color=f"C{k}", label=rf"$\alpha$ = {a:g}")
    right.axvspan(-10, 0, color="0.92", zorder=0)
    right.text(-9.6, 0.95, r"here $\ln r = t$, its maximum", fontsize=9, color="0.35")
    left.set(
        xlabel="x",
        ylabel=r"weight $\propto p^\alpha q^{1-\alpha}$",
        xlim=(-10, 5),
        ylim=(0, 0.35),
        title=f"Gaussian pair N(0, 2) vs N({t:g}, 2): runs off to the tail",
    )
    right.set(
        xlabel="x",
        ylabel=r"weight $\propto p^\alpha q^{1-\alpha}$",
        xlim=(-10, 5),
        ylim=(0, 1.05),
        title=f"Laplace pair Lap(0, 1) vs Lap({t:g}, 1): stops at x ≤ 0",
    )
    left.legend(fontsize=9, loc="upper left")
    right.legend(fontsize=9, loc="upper right")
    return fig


def plot_renyi_chords() -> Figure:
    lams = [-1 + i / 100 for i in range(401)]
    psi = [log_ratio_log_mgf(LOADED, FAIR, lam) for lam in lams]
    fig, (left, right) = plt.subplots(1, 2, figsize=(11, 4.2))
    left.plot(lams, psi, "k", linewidth=2, label=r"$K_L(\lambda) = \ln \mathbb{E}_P[e^{\lambda L}]$")
    kl_value = kl(LOADED, FAIR)
    left.plot([-1, 3], [-kl_value, 3 * kl_value], ":", color="gray", label=f"tangent at 0, slope KL = {kl_value:.3f}")
    for lam in (-0.5, 1.0, 2.5):
        val = log_ratio_log_mgf(LOADED, FAIR, lam)
        left.plot(
            [0, lam],
            [0, val],
            "o-",
            label=rf"chord to $\lambda$ = {lam:g}, slope $D_{{{lam + 1:g}}}$ = {val / lam:.3f}",
        )
    left.axhline(0, color="0.8", linewidth=0.8)
    left.axvline(0, color="0.8", linewidth=0.8)
    left.set(
        xlabel=r"$\lambda = \alpha - 1$", ylabel=r"$K_L(\lambda)$", title="Rényi divergence = slope of a chord of ψ"
    )
    left.legend(fontsize=8, loc="upper left")

    alphas = [0.05 * i for i in range(1, 401)]
    for p, q, name, color in (
        (LOADED, FAIR, r"$D_\alpha$(loaded ‖ fair)", "C0"),
        (FAIR, LOADED, r"$D_\alpha$(fair ‖ loaded)", "C1"),
    ):
        right.plot(alphas, [renyi(p, q, a) for a in alphas], color=color, label=name)
        right.axhline(
            max_divergence(p, q),
            color=color,
            linestyle="--",
            linewidth=1,
            label=rf"$D_\infty$ = ln {math.exp(max_divergence(p, q)):.3g} = {max_divergence(p, q):.3f}",
        )
        right.plot([1], [kl(p, q)], "o", color=color)
    right.set(
        xlabel=r"order $\alpha$",
        ylabel="nats",
        ylim=(0, 1.2),
        title="Non-decreasing in α, bounded by the max-divergence",
    )
    right.legend(fontsize=8, loc="lower right")
    return fig


def plot_gauss_laplace_renyi(t: float = 1.0) -> Figure:
    g = t / math.sqrt(2)  # same variance: σ² = 2b²
    alphas = [0.05 * i for i in range(1, 201)]
    fig, (left, right) = plt.subplots(1, 2, figsize=(11, 4))
    xs = [i / 50 for i in range(-300, 351)]
    left.plot(xs, [laplace_pdf(x, 0, 1) for x in xs], "C0", label="Lap(0, 1)")
    left.plot(xs, [laplace_pdf(x, t, 1) for x in xs], "C0", linestyle="--", label=f"Lap({t:g}, 1)")
    left.plot(xs, [normal_pdf(x, 0, math.sqrt(2)) for x in xs], "C1", label="N(0, 2)")
    left.plot(xs, [normal_pdf(x, t, math.sqrt(2)) for x in xs], "C1", linestyle="--", label=f"N({t:g}, 2)")
    left.set(xlabel="x", ylabel="density", title=f"Two pairs: shift {t:g}, variance 2")
    left.legend(fontsize=9)
    right.plot(alphas, [laplace_renyi(a, t) for a in alphas], "C0", label=r"Laplace pair (finite $D_\infty$)")
    right.plot(
        alphas,
        [gaussian_renyi(a, g) for a in alphas],
        "C1",
        label=r"Gaussian pair: $\alpha s^2/(2\sigma^2)$, $D_\infty = \infty$",
    )
    right.axhline(t, color="C0", linestyle=":", label=rf"Laplace $D_\infty$ = s/b = {t:g}")
    right.set(
        xlabel=r"order $\alpha$",
        ylabel=r"$D_\alpha$ (nats)",
        ylim=(0, 3),
        title="Laplace saturates, Gaussian grows without bound",
    )
    right.legend(fontsize=9)
    return fig


def plot_tv_gaussians(rng: np.random.Generator, mu: float = 1.0, per_sample: float = 0.1) -> Figure:
    xs = np.linspace(-4.0, 5.0, 901)
    p = np.array([normal_pdf(float(x)) for x in xs])
    q = np.array([normal_pdf(float(x), mu) for x in xs])
    fig, (left, right) = plt.subplots(1, 2, figsize=(11, 4))
    left.plot(xs, p, "C0", label="P = N(0, 1)")
    left.plot(xs, q, "C1", label=f"Q = N({mu:g}, 1)")
    left.fill_between(
        xs, q, p, where=p > q, color="C0", alpha=0.3, label=f"area of (p − q)₊ = TV = {gaussian_tv(mu):.3f}"
    )
    left.axvline(mu / 2, color="k", linestyle=":", label="best test: say P iff x < μ/2")
    left.set(xlabel="x", ylabel="density", ylim=(0, 0.55), title="TV is the area where p exceeds q")
    left.legend(fontsize=8, loc="upper left")

    ns = [round(10 ** (k / 20)) for k in range(81)]
    exact = [0.5 + 0.5 * gaussian_tv(per_sample * math.sqrt(n)) for n in ns]
    pinsker = [0.5 + 0.5 * min(1.0, pinsker_bound(n * gaussian_kl(per_sample))) for n in ns]
    sim_ns = [10, 100, 1000, 10000]
    simulated = [mean_test_accuracy(rng, per_sample, n, trials=4000) for n in sim_ns]
    right.plot(ns, exact, "k", linewidth=2, label="exact: ½ + ½ TV")
    right.plot(ns, pinsker, "C2", linestyle="--", label="via Pinsker: ½ + ½ √(KL/2)")
    right.plot(sim_ns, simulated, "ko", label="simulated best test")
    right.set(
        xscale="log",
        xlabel="samples n",
        ylabel="P(correct)",
        ylim=(0.45, 1.02),
        title=f"Telling N(0, 1) from N({per_sample:g}, 1) with n samples",
    )
    right.legend(fontsize=9, loc="upper left")
    return fig


def plot_hockey_area(mu: float = 1.0) -> Figure:
    xs = np.linspace(-4.0, 5.0, 901)
    p = np.array([normal_pdf(float(x)) for x in xs])
    q = np.array([normal_pdf(float(x), mu) for x in xs])
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6), sharey=True)
    for ax, gamma in zip(axes, (1.0, 2.0, 4.0)):
        ax.plot(xs, p, "C0", label="p = N(0, 1)")
        ax.plot(xs, gamma * q, "C1", label=rf"$\gamma q$, q = N({mu:g}, 1)")
        ax.fill_between(xs, gamma * q, p, where=p > gamma * q, color="C0", alpha=0.35, label=r"$(p - \gamma q)_+$")
        ax.set(
            xlabel="x",
            xlim=(-4, 5),
            ylim=(0, 1.0),
            title=rf"$\gamma$ = {gamma:g}: $E_\gamma$ = {gaussian_hockey_stick(gamma, mu):.4f}",
        )
    axes[0].set_ylabel("density")
    axes[0].legend(fontsize=8, loc="upper left")
    return fig


def plot_hockey_stick(t: float = 1.0) -> Figure:
    g = t / math.sqrt(2)
    cs = [i / 200 for i in range(401)]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(cs, [hockey_stick(LOADED, FAIR, math.exp(c)) for c in cs], label="die: loaded ‖ fair")
    ax.plot(cs, [laplace_hockey_stick(math.exp(c), t) for c in cs], label=f"Lap(0, 1) ‖ Lap({t:g}, 1)")
    ax.plot(cs, [gaussian_hockey_stick(math.exp(c), g) for c in cs], label=f"N(0, 2) ‖ N({t:g}, 2)")
    for x, color in ((math.log(3), "C0"), (t, "C1")):
        ax.axvline(x, color=color, linestyle=":", linewidth=1)
    ax.set(
        xlabel=r"$\ln \gamma$",
        ylabel=r"$E_\gamma(P \| Q)$",
        ylim=(0, 0.45),
        title=r"Hockey-stick curves: zero once $\gamma$ passes max p/q",
    )
    ax.legend()
    return fig


def plot_f_generators() -> Figure:
    ts = [i / 200 for i in range(1, 801)]
    gens: list[tuple[str, Callable[[float], float]]] = [
        (r"KL: $t\ln t$", lambda t: t * math.log(t)),
        (r"reverse KL: $-\ln t$", lambda t: -math.log(t)),
        (r"TV: $\frac{1}{2}|t-1|$", lambda t: 0.5 * abs(t - 1)),
        (r"$E_2$: $(t-2)_+$", lambda t: max(t - 2, 0.0)),
        (r"$\chi^2$: $(t-1)^2$", lambda t: (t - 1) ** 2),
        (r"Hellinger: $(\sqrt{t}-1)^2$", lambda t: (math.sqrt(t) - 1) ** 2),
    ]
    fig, ax = plt.subplots(figsize=(9, 4.5))
    for label, f in gens:
        ax.plot(ts, [f(t) for t in ts], label=label)
    for r in (0.6, 3.0):
        ax.axvline(r, color="0.6", linestyle=":", linewidth=1)
    ax.text(0.62, 2.75, "die: p/q = 0.6", fontsize=8, color="0.4")
    ax.text(3.02, 2.75, "die: p/q = 3", fontsize=8, color="0.4")
    ax.set(
        xlabel="likelihood ratio t = p/q",
        ylabel="f(t)",
        ylim=(-0.5, 3),
        xlim=(0, 4),
        title="Each f-divergence averages its own f of the likelihood ratio",
    )
    ax.legend(fontsize=9, loc="upper left", bbox_to_anchor=(1.02, 1))
    return fig


def plot_dpi_binning(mu: float = 1.0) -> Figure:
    widths = [0.05 * i for i in range(1, 121)]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(widths, [binned_gaussian_kl(mu, w) for w in widths], label="KL after binning x into width-w bins")
    ax.axhline(gaussian_kl(mu), color="k", linestyle="--", label=f"KL of the raw data = {gaussian_kl(mu):g}")
    ax.axhline(
        binned_gaussian_kl(mu, math.inf),
        color="C3",
        linestyle=":",
        label=f"one bit, x < μ/2: {binned_gaussian_kl(mu, math.inf):.3f}",
    )
    ax.set(
        xlabel="bin width w",
        ylabel="KL (nats)",
        ylim=(0, 0.55),
        title=f"Processing loses information: N(0, 1) vs N({mu:g}, 1)",
    )
    ax.legend(loc="lower left")
    return fig


def plot_dpi_staircase(mu: float = 1.0) -> Figure:
    xs = [i / 100 for i in range(-400, 501)]
    fig, ax = plt.subplots(figsize=(8, 4.2))
    ax.plot(xs, [(mu * mu - 2 * mu * x) / 2 for x in xs], "k", linewidth=2, label=r"raw: $\ln r(x) = \mu^2/2 - \mu x$")
    for width, color in ((1.0, "C0"), (2.0, "C1")):
        edges, p, q = binned_gaussian(mu, width)
        label: str | None = f"binned, width {width:g}"
        for lo, hi, pj, qj in zip(edges, edges[1:], p, q):
            lo_c, hi_c = max(lo, -4.0), min(hi, 5.0)
            if lo_c < hi_c:
                level = math.log(pj / qj)
                ax.plot([lo_c, hi_c], [level, level], color=color, linewidth=2.5, label=label)
                label = None
    ax.axhline(0, color="0.7", linewidth=0.8)
    ax.set(
        xlabel="x",
        ylabel="log-likelihood ratio (nats)",
        xlim=(-4, 5),
        ylim=(-5, 5),
        title=f"Binning N(0, 1) vs N({mu:g}, 1) flattens the log-likelihood ratio into steps",
    )
    ax.legend(fontsize=9, loc="upper right")
    return fig


def plot_pinsker() -> Figure:
    kls = [i / 100 for i in range(401)]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(kls, [min(pinsker_bound(k), 1.0) for k in kls], "k--", label=r"Pinsker $\sqrt{D_{\mathrm{KL}}/2}$")
    ax.plot(
        kls,
        [bretagnolle_huber_bound(k) for k in kls],
        "k:",
        label=r"Bretagnolle–Huber $\sqrt{1 - e^{-D_{\mathrm{KL}}}}$",
    )
    mus = [i / 100 for i in range(284)]
    ax.plot([gaussian_kl(m) for m in mus], [gaussian_tv(m) for m in mus], label="shifted Gaussians")
    ts = [i / 100 for i in range(501)]
    ax.plot(
        [laplace_kl(t) for t in ts if laplace_kl(t) <= 4],
        [laplace_tv(t) for t in ts if laplace_kl(t) <= 4],
        label="shifted Laplaces",
    )
    ax.plot([kl(LOADED, FAIR)], [tv(LOADED, FAIR)], "s", label="die: loaded vs fair")
    ax.set(
        xlabel="KL(P‖Q)",
        ylabel="TV(P, Q)",
        ylim=(0, 1.05),
        xlim=(0, 4),
        title="TV against KL: both bounds hold, Pinsker is tight near 0",
    )
    ax.legend(loc="lower right", fontsize=9)
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
        "divergences_evidence_walk.svg": plot_evidence_walk(rng),
        "divergences_forward_reverse_kl.svg": plot_forward_reverse_kl(),
        "divergences_renyi_tilt.svg": plot_renyi_tilt(),
        "divergences_hockey_area.svg": plot_hockey_area(),
        "divergences_dpi_staircase.svg": plot_dpi_staircase(),
        "divergences_renyi_chords.svg": plot_renyi_chords(),
        "divergences_gauss_laplace_renyi.svg": plot_gauss_laplace_renyi(),
        "divergences_tv_gaussians.svg": plot_tv_gaussians(rng),
        "divergences_hockey_stick.svg": plot_hockey_stick(),
        "divergences_f_generators.svg": plot_f_generators(),
        "divergences_dpi_binning.svg": plot_dpi_binning(),
        "divergences_pinsker.svg": plot_pinsker(),
    }
    for name, fig in figures.items():
        fig.tight_layout()
        fig.savefig(out / name, metadata={"Date": None})  # no timestamp, so output is reproducible
        plt.close(fig)
        print(f"saved {out / name}")


if __name__ == "__main__":
    main()
