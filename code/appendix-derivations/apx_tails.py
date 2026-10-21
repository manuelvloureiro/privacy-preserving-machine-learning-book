"""Check the derivations of the appendix on tail bounds, tests and divergences, and draw its figures.

    uv run python code/appendix-derivations/apx_tails.py
    uv run python code/appendix-derivations/apx_tails.py --seed 1 --out /tmp/figures

Prints every number quoted in book-full/appendix-derivations/06-tails-tests-divergences.md
(seed 0), then saves the page's figures.
"""

import argparse
import math
from collections.abc import Callable, Sequence
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
matplotlib.rcParams["svg.hashsalt"] = "apx_tails"  # stable element ids, so reruns don't churn the SVGs
matplotlib.rcParams["svg.fonttype"] = "none"  # keep text as text: smaller, diffable files

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from numpy.typing import NDArray

# Figures are build output: _build/figures/<part>/, which the build copies into each book.
FIGURES = Path(__file__).resolve().parents[2] / "_build" / "figures" / Path(__file__).parent.name

FloatArray = NDArray[np.float64]


# --- Shared helpers ---------------------------------------------------------------------------


def log_binom_pmf(n: int, k: int, p: float) -> float:
    """ln Pr[Bin(n, p) = k] for 0 < p < 1."""
    return math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1) + k * math.log(p) + (n - k) * math.log1p(-p)


def logsumexp(values: Sequence[float]) -> float:
    top = max(values)
    if top == -math.inf:
        return -math.inf
    return top + math.log(sum(math.exp(v - top) for v in values))


def binom_cdf(n: int, p: float, k: int) -> float:
    """F_p(k) = Pr[Bin(n, p) <= k]."""
    if k < 0:
        return 0.0
    if k >= n or p <= 0.0:
        return 1.0
    if p >= 1.0:
        return 0.0
    return min(1.0, math.exp(logsumexp([log_binom_pmf(n, j, p) for j in range(k + 1)])))


def log_binom_upper_tail(n: int, p: float, k: int) -> float:
    """ln Pr[Bin(n, p) >= k], accurate far into the tail."""
    return logsumexp([log_binom_pmf(n, j, p) for j in range(max(k, 0), n + 1)])


def kl_bernoulli(a: float, b: float) -> float:
    """Binary KL kl(a, b) in nats."""
    total = 0.0
    if a > 0:
        total += a * math.log(a / b)
    if a < 1:
        total += (1 - a) * math.log((1 - a) / (1 - b))
    return total


def normal_cdf(x: float) -> float:
    return 0.5 * math.erfc(-x / math.sqrt(2))


def bisect(fn: Callable[[float], float], lo: float, hi: float, iters: int = 200) -> float:
    """Root of a function with fn(lo) and fn(hi) of opposite signs."""
    f_lo = fn(lo)
    for _ in range(iters):
        mid = (lo + hi) / 2
        f_mid = fn(mid)
        if (f_mid > 0) == (f_lo > 0):
            lo, f_lo = mid, f_mid
        else:
            hi = mid
    return (lo + hi) / 2


# --- Bernstein's inequality -------------------------------------------------------------------


def centred_bernoulli_log_mgf(lam: float, p: float) -> float:
    """psi(λ) = ln E[e^{λ(X - p)}] for X ~ Bernoulli(p)."""
    return math.log1p(p * math.expm1(lam)) - lam * p


def centred_bernoulli_moment(k: int, p: float) -> float:
    """E[(X - p)^k] for X ~ Bernoulli(p)."""
    return p * (1 - p) ** k + (1 - p) * (-p) ** k


def hoeffding_log_mgf(lam: float, width: float) -> float:
    """Hoeffding's lemma: λ² (b - a)² / 8."""
    return lam * lam * width * width / 8


def bernstein_log_mgf_series(lam: float, var: float, m: float) -> float:
    """Step 2 of the derivation: (σ²/M²)(e^{λM} - 1 - λM)."""
    u = lam * m
    return var / (m * m) * (math.expm1(u) - u)


def bernstein_log_mgf_simple(lam: float, var: float, m: float) -> float:
    """Step 3: σ²λ² / (2(1 - λM/3)), valid for 0 <= λ < 3/M."""
    if lam * m >= 3:
        return math.inf
    return var * lam * lam / (2 * (1 - lam * m / 3))


def series_tail_bound_ok(u: float) -> bool:
    """e^u - 1 - u <= u² / (2(1 - u/3)) for 0 <= u < 3."""
    return math.expm1(u) - u <= u * u / (2 * (1 - u / 3)) * (1 + 1e-12)


def bernstein_tail(n: int, t: float, var: float, m: float) -> float:
    return math.exp(-n * t * t / (2 * var + 2 * m * t / 3))


def hoeffding_tail(n: int, t: float, width: float) -> float:
    return math.exp(-2 * n * t * t / width**2)


def chernoff_tail(n: int, p: float, t: float) -> float:
    """The optimized Chernoff bound for a Bernoulli mean, exp(-n kl(p + t, p))."""
    return math.exp(-n * kl_bernoulli(p + t, p))


def bernoulli_mean_tail(n: int, p: float, t: float) -> float:
    """Exact Pr[mean of n Bernoulli(p) - p >= t]."""
    k = math.ceil(n * (p + t) - 1e-9)
    return math.exp(log_binom_upper_tail(n, p, k))


def bernstein_width(n: int, var: float, m: float, beta: float) -> float:
    """The t at which the Bernstein bound equals β: the root of n t² - (2/3) M L t - 2 σ² L = 0."""
    big_l = math.log(1 / beta)
    return m * big_l / (3 * n) + math.sqrt((m * big_l / (3 * n)) ** 2 + 2 * var * big_l / n)


def bernstein_width_simple(n: int, var: float, m: float, beta: float) -> float:
    big_l = math.log(1 / beta)
    return math.sqrt(2 * var * big_l / n) + 2 * m * big_l / (3 * n)


# --- Cramér's lower bound ---------------------------------------------------------------------


def binomial_rate(n: int, p: float, t: float) -> float:
    """-(1/n) ln Pr[Bin(n, p) >= n t]."""
    return -log_binom_upper_tail(n, p, math.ceil(n * t - 1e-9)) / n


def tilted_bernoulli(p: float, lam: float) -> float:
    """Success probability of Bernoulli(p) tilted by e^{λx}."""
    return p * math.exp(lam) / (1 - p + p * math.exp(lam))


def tilt_for_mean(p: float, mean: float) -> float:
    """The λ whose tilted Bernoulli(p) has the given mean: ψ'(λ) = mean."""
    return math.log(mean * (1 - p) / (p * (1 - mean)))


def bernoulli_log_mgf(lam: float, p: float) -> float:
    return math.log1p(p * math.expm1(lam))


def cramer_rate_upper(n: int, p: float, t: float, eps: float) -> float:
    """The tilting argument: -(1/n) ln Pr[S_n >= n t] is at most this.

    Tilt so the mean is t + ε/2; then Pr >= e^{-n(λ(t+ε) - ψ(λ))} Q_λ[window], and Chebyshev
    gives Q_λ[window] >= 1 - Var_λ / (n (ε/2)²).
    """
    mean = t + eps / 2
    lam = tilt_for_mean(p, mean)
    window = 1 - mean * (1 - mean) / (n * (eps / 2) ** 2)
    if window <= 0:
        return math.inf
    return lam * (t + eps) - bernoulli_log_mgf(lam, p) - math.log(window) / n


def cramer_eps(n: int, t: float) -> float:
    """The window width used in the figure: wide enough that Chebyshev leaves about 3/4 of the mass."""
    return 4 * math.sqrt(t * (1 - t) / n)


# --- The Chernoff–Stein lemma on the spam pair ------------------------------------------------

HAM = (0.4, 0.3, 0.2, 0.1)  # P, the null hypothesis
SPAM = (0.1, 0.2, 0.3, 0.4)  # Q, the alternative
# The evidence L = ln(p/q) is ln 4, ln 1.5, -ln 1.5, -ln 4 on the counts 0..3, so the evidence of a
# sample is a ln 4 + b ln 1.5 with a = #0 - #3 and b = #1 - #2.
STEPS = ((1, 0), (0, 1), (0, -1), (-1, 0))


def spam_kl() -> float:
    return sum(p * math.log(p / q) for p, q in zip(HAM, SPAM))


def spam_evidence_variance() -> float:
    d = spam_kl()
    return sum(p * (math.log(p / q) - d) ** 2 for p, q in zip(HAM, SPAM))


def _step(mass: FloatArray, weights: Sequence[float]) -> FloatArray:
    new = np.zeros_like(mass)
    for (da, db), w in zip(STEPS, weights):
        src = mass[max(0, -da) : mass.shape[0] - max(0, da), max(0, -db) : mass.shape[1] - max(0, db)]
        new[max(0, da) : mass.shape[0] - max(0, -da), max(0, db) : mass.shape[1] - max(0, -db)] += w * src
    return new


def optimal_beta(p_mass: FloatArray, q_mass: FloatArray, n_max: int, size: float) -> float:
    """Type II error of the most powerful test of size exactly `size` (Neyman–Pearson)."""
    a_idx, b_idx = np.nonzero((p_mass > 0) | (q_mass > 0))
    evidence = (a_idx - n_max) * math.log(4) + (b_idx - n_max) * math.log(1.5)
    order = np.argsort(evidence, kind="stable")  # reject the most spam-like (smallest evidence) first
    p_sorted = p_mass[a_idx, b_idx][order]
    q_sorted = q_mass[a_idx, b_idx][order]
    cum = np.cumsum(p_sorted)
    i = int(np.searchsorted(cum, size))
    before = float(cum[i - 1]) if i > 0 else 0.0
    gamma = (size - before) / float(p_sorted[i])
    return float(q_sorted[i + 1 :].sum()) + (1 - gamma) * float(q_sorted[i])


def stein_betas(ns: Sequence[int], size: float) -> dict[int, float]:
    """β_n of the optimal size-`size` test between n draws of ham and n draws of spam, for each n."""
    n_max = max(ns)
    width = 2 * n_max + 1
    p_mass = np.zeros((width, width))
    q_mass = np.zeros((width, width))
    p_mass[n_max, n_max] = q_mass[n_max, n_max] = 1.0
    out: dict[int, float] = {}
    for n in range(1, n_max + 1):
        p_mass = _step(p_mass, HAM)
        q_mass = _step(q_mass, SPAM)
        if n in ns:
            out[n] = optimal_beta(p_mass, q_mass, n_max, size)
    return out


def stein_rate_lower(n: int, size: float) -> float:
    """Achievability: a threshold test at D - ε with ε = sqrt(V/(size n)) has size <= `size` by Chebyshev."""
    return spam_kl() - math.sqrt(spam_evidence_variance() / (size * n))


def stein_rate_upper(n: int, size: float) -> float:
    """Converse with ε = sqrt(2V/((1 - size) n)): β_n >= e^{-n(D+ε)}(1 - size)/2."""
    eps = math.sqrt(2 * spam_evidence_variance() / ((1 - size) * n))
    return spam_kl() + eps - math.log((1 - size) / 2) / n


# --- Clopper–Pearson --------------------------------------------------------------------------


def cdf_slope(n: int, p: float, k: int) -> float:
    """d/dp F_p(k) = -n C(n-1, k) p^k (1-p)^{n-1-k}, for k <= n - 1."""
    return -n * math.comb(n - 1, k) * p**k * (1 - p) ** (n - 1 - k)


def cdf_by_beta_integral(n: int, p: float, k: int, steps: int = 20_000) -> float:
    """∫_p^1 n C(n-1, k) s^k (1-s)^{n-1-k} ds by Simpson's rule: the Beta form of F_p(k)."""
    h = (1 - p) / steps
    total = 0.0
    for i in range(steps + 1):
        s = p + i * h
        weight = 1 if i in (0, steps) else (4 if i % 2 else 2)
        total += weight * -cdf_slope(n, s, k)
    return total * h / 3


def cp_upper(n: int, k: int, level: float) -> float:
    """u(k): the rate with Pr[Bin(n, u) <= k] = level (1 if k = n)."""
    if k >= n:
        return 1.0
    return bisect(lambda p: binom_cdf(n, p, k) - level, 0.0, 1.0)


def cp_lower(n: int, k: int, level: float) -> float:
    """l(k): the rate with Pr[Bin(n, l) >= k] = level (0 if k = 0)."""
    if k <= 0:
        return 0.0
    return bisect(lambda p: 1 - binom_cdf(n, p, k - 1) - level, 0.0, 1.0)


def wald_interval(n: int, k: int, z: float = 1.959963984540054) -> tuple[float, float]:
    phat = k / n
    half = z * math.sqrt(phat * (1 - phat) / n)
    return phat - half, phat + half


def coverage(n: int, p: float, intervals: Sequence[tuple[float, float]]) -> float:
    """Exact probability that the interval built from K ~ Bin(n, p) contains p."""
    return sum(math.exp(log_binom_pmf(n, k, p)) for k, (lo, hi) in enumerate(intervals) if lo <= p <= hi)


def cp_intervals(n: int, conf_miss: float) -> list[tuple[float, float]]:
    return [(cp_lower(n, k, conf_miss / 2), cp_upper(n, k, conf_miss / 2)) for k in range(n + 1)]


# --- Rényi divergence tends to KL -------------------------------------------------------------

DIE_P = (0.1, 0.1, 0.1, 0.1, 0.1, 0.5)  # loaded
DIE_Q = (1 / 6,) * 6  # fair


def die_psi(lam: float) -> float:
    """ψ(λ) = ln E_P[r^λ] for the loaded die against the fair die."""
    return math.log(sum(p * (p / q) ** lam for p, q in zip(DIE_P, DIE_Q)))


def die_sandwich(lam: float) -> float:
    """E_P[(r^λ - 1)/λ], the bound from ln m <= m - 1."""
    return sum(p * ((p / q) ** lam - 1) / lam for p, q in zip(DIE_P, DIE_Q))


def die_kl() -> float:
    return sum(p * math.log(p / q) for p, q in zip(DIE_P, DIE_Q))


def gauss_psi(lam: float, mu: float) -> float:
    """ψ(λ) for N(0, 1) against N(μ, 1): λ(λ + 1)μ²/2."""
    return lam * (lam + 1) * mu * mu / 2


def gauss_sandwich(lam: float, mu: float) -> float:
    return math.expm1(gauss_psi(lam, mu)) / lam


def power_slope(x: float, lam: float) -> float:
    """h_λ(x) = (x^λ - 1)/λ, with h_0 = ln x."""
    return math.log(x) if lam == 0 else math.expm1(lam * math.log(x)) / lam


def renyi_entropy_bits(probs: Sequence[float], counts: Sequence[int], alpha: float) -> float:
    """H_α in bits for a distribution given as probability levels with their multiplicities."""
    if math.isinf(alpha):
        return -math.log2(max(probs))
    if alpha == 1:
        return -sum(c * p * math.log2(p) for p, c in zip(probs, counts))
    log_sum = logsumexp([math.log(c) + alpha * math.log(p) for p, c in zip(probs, counts)])
    return log_sum / ((1 - alpha) * math.log(2))


def renyi_div_from_uniform_bits(probs: Sequence[float], counts: Sequence[int], alpha: float) -> float:
    """D_α(P || U) in bits, computed from its definition."""
    size = sum(counts)
    log_sum = logsumexp(
        [math.log(c) + alpha * math.log(p) + (1 - alpha) * math.log(1 / size) for p, c in zip(probs, counts)]
    )
    return log_sum / ((alpha - 1) * math.log(2))


PIN_PROBS = (0.1, 0.9 / 9_999)
PIN_COUNTS = (1, 9_999)


# --- Perspective functions --------------------------------------------------------------------


def renyi_term(x: float, y: float, alpha: float) -> float:
    """x^α y^{1-α}, the perspective of t^α, with 0 at x = 0 and ∞ at y = 0 < x."""
    if x == 0:
        return 0.0
    if y == 0:
        return math.inf
    return x**alpha * y ** (1 - alpha)


def joint_convexity_violations(rng: np.random.Generator, alpha: float, trials: int) -> int:
    """Random pairs of points and weights where g(mix) > mix of g, beyond rounding."""
    pts = rng.uniform(0.01, 1.0, size=(trials, 4))
    lam = rng.uniform(0.0, 1.0, size=trials)
    bad = 0
    for (x1, y1, x2, y2), w in zip(pts.tolist(), lam.tolist()):
        mix = renyi_term(w * x1 + (1 - w) * x2, w * y1 + (1 - w) * y2, alpha)
        avg = w * renyi_term(x1, y1, alpha) + (1 - w) * renyi_term(x2, y2, alpha)
        bad += mix > avg * (1 + 1e-12)
    return bad


def subadditivity_violations(rng: np.random.Generator, alpha: float, trials: int) -> int:
    pts = rng.uniform(0.0, 1.0, size=(trials, 4))
    bad = 0
    for x1, y1, x2, y2 in pts.tolist():
        merged = renyi_term(x1 + x2, y1 + y2, alpha)
        bad += merged > (renyi_term(x1, y1, alpha) + renyi_term(x2, y2, alpha)) * (1 + 1e-12)
    return bad


def renyi_div(p: FloatArray, q: FloatArray, alpha: float) -> float:
    return math.log(float(np.sum(p**alpha * q ** (1 - alpha)))) / (alpha - 1)


def quasi_convexity_violations(rng: np.random.Generator, alpha: float, trials: int, outputs: int = 5) -> int:
    """Mixtures of 3 pairs with shared weights whose D_α exceeds the largest D_α of the parts."""
    bad = 0
    for _ in range(trials):
        ps = rng.dirichlet(np.ones(outputs), size=3)
        qs = rng.dirichlet(np.ones(outputs), size=3)
        w = rng.dirichlet(np.ones(3))
        mixed = renyi_div(w @ ps, w @ qs, alpha)
        worst = max(renyi_div(ps[i], qs[i], alpha) for i in range(3))
        bad += mixed > worst + 1e-12
    return bad


def randomized_test_divergence(p: Sequence[float], q: Sequence[float], phi: Sequence[float], alpha: float) -> float:
    """D_α between the reject/accept coins of a randomized test φ."""
    a = sum(pi * f for pi, f in zip(p, phi))
    one_minus_b = sum(qi * f for qi, f in zip(q, phi))
    s = renyi_term(a, one_minus_b, alpha) + renyi_term(1 - a, 1 - one_minus_b, alpha)
    return math.log(s) / (alpha - 1)


def subsampled_reverse_moment(q: float, sigma: float, alpha: float) -> float:
    """e^{(α-1) D_α(N(0,σ²) || (1-q)N(0,σ²) + q N(1,σ²))}, by a fine Riemann sum."""
    y = np.linspace(-14 * sigma, 1 + 14 * sigma, 400_001)
    dy = float(y[1] - y[0])
    log_p0 = -(y**2) / (2 * sigma**2) - math.log(sigma * math.sqrt(2 * math.pi))
    log_ratio = (2 * y - 1) / (2 * sigma**2)  # ln(P1/P0)
    log_mix = log_p0 + np.logaddexp(math.log(1 - q), math.log(q) + log_ratio)
    integrand = np.exp(alpha * log_p0 + (1 - alpha) * log_mix)
    return float(integrand.sum() * dy)


def rdp_constraints(a: float, b: float, alpha: float) -> tuple[float, float]:
    """The two left sides of the RDP test constraint at order α for a test with errors (a, b)."""
    return (
        renyi_term(a, 1 - b, alpha) + renyi_term(1 - a, b, alpha),
        renyi_term(1 - b, a, alpha) + renyi_term(b, 1 - a, alpha),
    )


def f_rdp(a: float, alpha: float, eps_alpha: float) -> float:
    """Smallest b with both constraints at most e^{(α-1)ε(α)}; both decrease in b on [0, 1 - a]."""
    cap = math.exp((alpha - 1) * eps_alpha)
    if max(rdp_constraints(a, 0.0, alpha)) <= cap:
        return 0.0
    return bisect(lambda b: max(rdp_constraints(a, b, alpha)) - cap, 0.0, 1 - a)


# --- The noisy count --------------------------------------------------------------------------


def binomial_log_weights(m: int) -> FloatArray:
    return np.array(
        [math.lgamma(m + 1) - math.lgamma(k + 1) - math.lgamma(m - k + 1) - m * math.log(2) for k in range(m + 1)]
    )


def log_count_density(y: FloatArray, log_w: FloatArray, sigma: float) -> FloatArray:
    """ln f(y) for f = Σ_k w_k φ_σ(y - k), by log-sum-exp over k."""
    k = np.arange(log_w.size, dtype=np.float64)
    terms = log_w[None, :] - (y[:, None] - k[None, :]) ** 2 / (2 * sigma**2)
    top = terms.max(axis=1)
    return top + np.log(np.exp(terms - top[:, None]).sum(axis=1)) - math.log(sigma * math.sqrt(2 * math.pi))


def log_shift_ratio(y: FloatArray, log_w: FloatArray, sigma: float) -> FloatArray:
    """ln q(y)/p(y) with p = f and q = f(· - 1): world 1 against world 0."""
    return log_count_density(y - 1, log_w, sigma) - log_count_density(y, log_w, sigma)


def count_modes(values: FloatArray) -> int:
    inner = values[1:-1]
    return int(np.sum((inner > values[:-2]) & (inner > values[2:])))


def pairing_identity_error(y: float, y2: float, m: int, sigma: float) -> float:
    """|q(y')p(y) - q(y)p(y') - Σ_{j>k} (w_{j-1}w_k - w_{k-1}w_j)(φ(y'-j)φ(y-k) - φ(y-j)φ(y'-k))|."""
    w = [math.comb(m, k) / 2**m for k in range(m + 1)]

    def wt(i: int) -> float:
        return w[i] if 0 <= i <= m else 0.0

    def phi(x: float) -> float:
        return math.exp(-x * x / (2 * sigma**2)) / (sigma * math.sqrt(2 * math.pi))

    def p(x: float) -> float:
        return sum(wt(k) * phi(x - k) for k in range(m + 1))

    def q(x: float) -> float:
        return p(x - 1)

    lhs = q(y2) * p(y) - q(y) * p(y2)
    rhs = sum(
        (wt(j - 1) * wt(k) - wt(k - 1) * wt(j)) * (phi(y2 - j) * phi(y - k) - phi(y - j) * phi(y2 - k))
        for j in range(m + 2)
        for k in range(j)
    )
    return abs(lhs - rhs)


def count_advantage(m: int, sigma: float, threshold: float) -> float:
    """Pr[K + 1 + Z > t] - Pr[K + Z > t] with K ~ Bin(m, 1/2), Z ~ N(0, σ²)."""
    w = [math.comb(m, k) / 2**m for k in range(m + 1)]
    world1 = sum(wk * (1 - normal_cdf((threshold - k - 1) / sigma)) for k, wk in enumerate(w))
    world0 = sum(wk * (1 - normal_cdf((threshold - k) / sigma)) for k, wk in enumerate(w))
    return world1 - world0


def max_second_difference(values: FloatArray) -> float:
    return float(np.max(values[2:] - 2 * values[1:-1] + values[:-2]))


# --- The page's table -------------------------------------------------------------------------


def print_table(rng: np.random.Generator) -> None:
    # Bernstein
    p, n, t = 0.01, 1000, 0.01
    var, m = p * (1 - p), 1 - p
    lams = [i / 100 for i in range(1, 300)]
    gap = max(centred_bernoulli_log_mgf(lam, p) - bernstein_log_mgf_series(lam, var, m) for lam in lams)
    gap2 = max(bernstein_log_mgf_series(lam, var, m) - bernstein_log_mgf_simple(lam, var, m) for lam in lams)
    series_ok = all(series_tail_bound_ok(u / 1000) for u in range(3000))
    print(
        f"[Bernstein MGF p={p}] max(psi - series)={gap:.1e}  max(series - simple)={gap2:.1e}  "
        f"series bound holds on [0,3): {series_ok}"
    )
    moments = "  ".join(
        f"k={k}: E[Y^k]={centred_bernoulli_moment(k, p):.7f} bound={var * m ** (k - 2):.7f}" for k in (2, 3, 4)
    )
    print(f"[Bernstein moments p={p}] {moments}")
    print(
        f"[Bernstein tail n={n} p={p} t={t}] exact={bernoulli_mean_tail(n, p, t):.2e}  "
        f"Hoeffding={hoeffding_tail(n, t, 1.0):.2e}  Bernstein={bernstein_tail(n, t, var, m):.2e}  "
        f"Chernoff={chernoff_tail(n, p, t):.2e}"
    )
    print(
        f"[Bernstein width n={n} p={p} beta=0.05] root={bernstein_width(n, var, m, 0.05):.5f}  "
        f"simple={bernstein_width_simple(n, var, m, 0.05):.5f}  "
        f"Hoeffding={math.sqrt(math.log(20) / (2 * n)):.5f}"
    )

    # Cramér
    p, t = 0.1, 0.2
    lam = tilt_for_mean(p, t)
    print(f"[Cramer p={p} t={t}] kl={kl_bernoulli(t, p):.5f}  lambda={lam:.4f}  psi={bernoulli_log_mgf(lam, p):.5f}")
    for n in (100, 1000, 10000):
        print(
            f"[Cramer n={n}] exact rate={binomial_rate(n, p, t):.5f}  "
            f"tilting bound={cramer_rate_upper(n, p, t, cramer_eps(n, t)):.5f}"
        )

    # Stein
    size = 0.05
    betas = stein_betas([10, 50, 200], size)
    print(f"[Stein spam] KL={spam_kl():.4f}  V={spam_evidence_variance():.4f}")
    for n, beta in betas.items():
        print(
            f"[Stein n={n}] beta={beta:.3e}  rate={-math.log(beta) / n:.4f}  "
            f"proof bounds=[{stein_rate_lower(n, size):.4f}, {stein_rate_upper(n, size):.4f}]"
        )

    # Clopper–Pearson
    n = 1000
    slope_err = max(
        abs(cdf_slope(n, s, k) - (binom_cdf(n, s + 1e-7, k) - binom_cdf(n, s - 1e-7, k)) / 2e-7)
        / abs(cdf_slope(n, s, k))
        for s in (0.001, 0.005, 0.02)
        for k in (0, 1, 10)
    )
    beta_err = max(
        abs(cdf_by_beta_integral(n, s, k) - binom_cdf(n, s, k)) for s in (0.001, 0.005, 0.02) for k in (0, 1, 10)
    )
    print(
        f"[Clopper slope] max relative error of the slope formula={slope_err:.1e}  max |F - Beta integral|={beta_err:.1e}"
    )
    print(
        f"[Clopper n={n} level 0.025] u(0)={cp_upper(n, 0, 0.025):.6f}  u(1)={cp_upper(n, 1, 0.025):.6f}  "
        f"u(10)={cp_upper(n, 10, 0.025):.6f}"
    )
    print(
        f"[Clopper checks] k=10 n=10000: [{cp_lower(10_000, 10, 0.025):.5f}, {cp_upper(10_000, 10, 0.025):.5f}]  "
        f"k=1 n=1000 one-sided 0.05: {cp_upper(1000, 1, 0.05):.4f}  k=0 n=3000: {cp_upper(3000, 0, 0.05):.4e}"
    )
    n = 100
    cp = cp_intervals(n, 0.05)
    wald = [wald_interval(n, k) for k in range(n + 1)]
    grid = [i / 2000 for i in range(1, 2000)]
    cov_cp = [coverage(n, s, cp) for s in grid]
    cov_wald = [coverage(n, s, wald) for s in grid]
    print(
        f"[Coverage n={n}] Clopper-Pearson min={min(cov_cp):.4f}  mean={sum(cov_cp) / len(cov_cp):.4f}  "
        f"Wald min={min(cov_wald):.4f}  Wald at p=0.01: {coverage(n, 0.01, wald):.4f}  "
        f"CP at p=0.01: {coverage(n, 0.01, cp):.4f}"
    )

    # Rényi to KL
    print(f"[Renyi die] KL={die_kl():.4f}")
    for lam in (-0.5, -0.1, -0.01, 0.01, 0.1, 0.5):
        print(f"[Renyi die lambda={lam:+g}] D={die_psi(lam) / lam:.4f}  sandwich={die_sandwich(lam):.4f}")
    mono = all(
        power_slope(x, a) <= power_slope(x, a + 0.01) + 1e-12
        for x in (0.01, 0.6, 3.0, 50.0)
        for a in [i / 100 for i in range(-200, 200)]
    )
    print(f"[Renyi slopes] h_lambda(x) non-decreasing in lambda: {mono}")
    for x in (3.0, 0.6):
        row = "  ".join(f"{lam:+g}: {power_slope(x, lam):.4f}" for lam in (-0.5, -0.1, 0.0, 0.1, 0.5))
        print(f"[Renyi slopes x={x:g}] {row}")
    print(
        f"[Renyi entropy PIN bits] H={renyi_entropy_bits(PIN_PROBS, PIN_COUNTS, 1):.4f}  "
        f"H_0.999={renyi_entropy_bits(PIN_PROBS, PIN_COUNTS, 0.999):.4f}  "
        f"H_1.001={renyi_entropy_bits(PIN_PROBS, PIN_COUNTS, 1.001):.4f}  "
        f"H_2={renyi_entropy_bits(PIN_PROBS, PIN_COUNTS, 2):.4f}  "
        f"H_100={renyi_entropy_bits(PIN_PROBS, PIN_COUNTS, 100):.4f}  "
        f"H_inf={renyi_entropy_bits(PIN_PROBS, PIN_COUNTS, math.inf):.4f}"
    )
    log_n = math.log2(10_000)
    ident = abs(
        renyi_entropy_bits(PIN_PROBS, PIN_COUNTS, 2) - (log_n - renyi_div_from_uniform_bits(PIN_PROBS, PIN_COUNTS, 2))
    )
    print(f"[Renyi entropy identity] |H_2 - (log N - D_2(P||U))|={ident:.1e}")

    # Perspective
    print(
        f"[Perspective alpha=2] joint convexity violations={joint_convexity_violations(rng, 2.0, 100_000)}  "
        f"subadditivity violations={subadditivity_violations(rng, 2.0, 100_000)}  "
        f"quasi-convexity violations={quasi_convexity_violations(rng, 2.0, 20_000)}"
    )
    die_p, die_q = list(DIE_P), list(DIE_Q)
    phi_test = [0.0, 0.0, 0.0, 0.0, 0.5, 1.0]
    print(
        f"[Perspective die alpha=2] D_2(P||Q)={renyi_div(np.array(die_p), np.array(die_q), 2.0):.4f}  "
        f"test coin={randomized_test_divergence(die_p, die_q, phi_test, 2.0):.4f}"
    )
    q, sigma, alpha = 0.1, 1.0, 2.0
    rho0 = 1 / (2 * sigma**2)
    print(
        f"[Perspective subsampled q={q} sigma={sigma} alpha={alpha:g}] "
        f"moment={subsampled_reverse_moment(q, sigma, alpha):.4f}  "
        f"bound={(1 - q) + q * math.exp((alpha - 1) * alpha * rho0):.4f}"
    )
    a_grid = [i / 200 for i in range(1, 200)]
    f_vals = np.array([f_rdp(a, 2.0, 1.0) for a in a_grid])
    print(
        f"[RDP region alpha=2 eps=1] f_RDP(0.1)={f_rdp(0.1, 2.0, 1.0):.4f}  power={1 - f_rdp(0.1, 2.0, 1.0):.4f}  "
        f"min second difference={float(np.min(f_vals[2:] - 2 * f_vals[1:-1] + f_vals[:-2])):.1e}"
    )

    # Noisy count
    y = np.linspace(-1.5, 2.5, 4001)
    lw1 = binomial_log_weights(1)
    dens = np.exp(log_count_density(y, lw1, 0.2))
    lr1 = log_shift_ratio(y, lw1, 0.2)
    print(
        f"[Noisy count Bernoulli+N(0,0.2^2)] modes={count_modes(dens)}  "
        f"min increment of log ratio={float(np.min(np.diff(lr1))):.2e}  "
        f"pairing identity error={pairing_identity_error(-0.3, 0.8, 1, 0.2):.1e}"
    )
    lw100 = binomial_log_weights(100)
    y100 = np.linspace(20.0, 81.0, 6101)
    lr100 = log_shift_ratio(y100, lw100, 1.0)
    crossing = bisect(lambda s: float(log_shift_ratio(np.array([s]), lw100, 1.0)[0]), 40.0, 60.0)
    print(
        f"[Noisy count m=100 sigma=1] crossing={crossing:.4f}  advantage={count_advantage(100, 1.0, crossing):.4f}  "
        f"min increment of log ratio={float(np.min(np.diff(lr100))):.2e}  "
        f"pairing identity error={pairing_identity_error(48.2, 51.7, 100, 1.0):.1e}"
    )
    for s in (1.0, 0.5):
        lf = log_count_density(y100, lw100, s)
        print(f"[Noisy count m=100 sigma={s}] max second difference of ln f={max_second_difference(lf):+.1e}")


# --- Figures ----------------------------------------------------------------------------------


def plot_bernstein_mgf(p: float = 0.01) -> Figure:
    var, m = p * (1 - p), 1 - p
    lams = [i / 100 for i in range(1, 291)]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(lams, [centred_bernoulli_log_mgf(lam, p) for lam in lams], "k", linewidth=2, label=r"exact $\psi(\lambda)$")
    ax.plot(lams, [hoeffding_log_mgf(lam, 1.0) for lam in lams], label=r"Hoeffding $\lambda^2/8$")
    ax.plot(
        lams,
        [bernstein_log_mgf_series(lam, var, m) for lam in lams],
        label=r"Bernstein, step 2: $\frac{\sigma^2}{M^2}(e^{\lambda M}-1-\lambda M)$",
    )
    ax.plot(
        lams,
        [bernstein_log_mgf_simple(lam, var, m) for lam in lams],
        "--",
        label=r"Bernstein, step 3: $\frac{\sigma^2\lambda^2}{2(1-\lambda M/3)}$",
    )
    ax.set(
        yscale="log",
        ylim=(1e-5, 3),
        xlabel=r"$\lambda$",
        ylabel="log-MGF of the centred variable",
        title=f"Centred Bernoulli({p}): variance {var:.4f}, range bound M = {m}",
    )
    ax.legend(fontsize=8, loc="lower right")
    return fig


def plot_bernstein_tail(n: int = 1000, p: float = 0.01) -> Figure:
    var, m = p * (1 - p), 1 - p
    ts = [i / 2000 for i in range(1, 81)]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(ts, [bernoulli_mean_tail(n, p, t) for t in ts], "k", linewidth=2, label="exact (binomial)")
    ax.plot(ts, [hoeffding_tail(n, t, 1.0) for t in ts], label="Hoeffding")
    ax.plot(ts, [bernstein_tail(n, t, var, m) for t in ts], label="Bernstein")
    ax.plot(ts, [chernoff_tail(n, p, t) for t in ts], label="Chernoff (optimized)")
    ax.set(
        yscale="log",
        ylim=(1e-12, 2),
        xlabel="deviation t of the mean",
        ylabel=r"$\Pr[\overline{X}_n - p \geq t]$",
        title=f"Mean of n = {n} Bernoulli({p})",
    )
    ax.legend()
    return fig


def plot_cramer(p: float = 0.1, t: float = 0.2) -> Figure:
    fig, (left, right) = plt.subplots(1, 2, figsize=(11, 4))
    n = 100
    eps = cramer_eps(n, t)
    lam = tilt_for_mean(p, t + eps / 2)
    p_tilt = tilted_bernoulli(p, lam)
    ks = list(range(46))
    left.bar(ks, [math.exp(log_binom_pmf(n, k, p)) for k in ks], width=0.8, alpha=0.7, label=f"P: Bin({n}, {p})")
    left.bar(
        ks,
        [math.exp(log_binom_pmf(n, k, p_tilt)) for k in ks],
        width=0.8,
        alpha=0.6,
        label=f"tilted: Bin({n}, {p_tilt:.3f})",
    )
    left.axvspan(n * t, n * (t + eps), color="gray", alpha=0.2, label=r"window $[nt,\ n(t+\varepsilon)]$")
    left.set(xlabel="count S", ylabel="probability", title=f"Tilting moves the mass into the window (n = {n})")
    left.legend(fontsize=8)

    ns = [20 * i for i in range(1, 151)]
    right.plot(
        ns, [binomial_rate(k, p, t) for k in ns], "k", linewidth=2, label=r"exact $-\frac{1}{n}\ln\Pr[S_n \geq nt]$"
    )
    right.plot(
        ns, [cramer_rate_upper(k, p, t, cramer_eps(k, t)) for k in ns], label="tilting bound (lower bound on Pr)"
    )
    right.axhline(kl_bernoulli(t, p), color="C3", linestyle="--", label=f"Chernoff exponent kl({t}, {p})")
    right.set(
        ylim=(0, 0.25),
        xlabel="n",
        ylabel="exponent per sample",
        title=f"Pr[Bin(n, {p}) ≥ {t}n]: both bounds close in on kl",
    )
    right.legend(fontsize=8)
    return fig


def plot_stein(size: float = 0.05) -> Figure:
    ns = list(range(5, 201, 5))
    betas = stein_betas(ns, size)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(
        ns, [-math.log(betas[n]) / n for n in ns], "ko-", markersize=3, label=r"optimal test: $-\frac{1}{n}\ln\beta_n$"
    )
    ax.plot(ns, [stein_rate_lower(n, size) for n in ns], label="achievability bound (threshold test)")
    ax.plot(ns, [stein_rate_upper(n, size) for n in ns], label="converse bound (any test)")
    ax.axhline(spam_kl(), color="C3", linestyle="--", label=f"KL(P‖Q) = {spam_kl():.4f}")
    ax.set(
        ylim=(-0.3, 1.5),
        xlabel="number of observations n",
        ylabel="type II exponent per observation",
        title=f"Spam example, size {size}",
    )
    ax.legend(fontsize=8)
    return fig


def plot_clopper_cdf(n: int = 1000, level: float = 0.025) -> Figure:
    ps = [i / 20000 for i in range(501)]
    fig, ax = plt.subplots(figsize=(6, 4))
    for k, offset in ((0, (-10, 40)), (1, (6, 20)), (10, (4, 8))):
        line = ax.plot(ps, [binom_cdf(n, s, k) for s in ps], label=f"k = {k}")
        u = cp_upper(n, k, level)
        ax.plot([u], [level], "o", color=line[0].get_color())
        ax.annotate(
            f"u({k}) = {u:.4f}",
            (u, level),
            textcoords="offset points",
            xytext=offset,
            fontsize=8,
            arrowprops={"arrowstyle": "-", "linewidth": 0.5},
        )
    ax.axhline(level, color="k", linestyle=":", label=f"level a/2 = {level}")
    ax.set(xlabel="true rate p", ylabel=r"$F_p(k) = \Pr[\mathrm{Bin}(n, p) \leq k]$", title=f"n = {n}")
    ax.legend()
    return fig


def plot_clopper_coverage(n: int = 100) -> Figure:
    cp = cp_intervals(n, 0.05)
    wald = [wald_interval(n, k) for k in range(n + 1)]
    grid = [i / 2000 for i in range(1, 2000)]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(grid, [coverage(n, s, wald) for s in grid], color="C1", linewidth=0.8, label="Wald")
    ax.plot(grid, [coverage(n, s, cp) for s in grid], color="C0", linewidth=0.8, label="Clopper–Pearson")
    ax.axhline(0.95, color="k", linestyle="--", label="nominal 0.95")
    ax.set(ylim=(0.5, 1.005), xlabel="true rate p", ylabel="exact coverage", title=f"95% intervals for a rate, n = {n}")
    ax.legend(loc="lower center")
    return fig


def plot_renyi_sandwich(mu: float = 1.0) -> Figure:
    lams = [i / 100 for i in range(-100, 101) if i != 0]
    fig, (left, right) = plt.subplots(1, 2, figsize=(11, 4))

    def g_psi(lam: float) -> float:
        return gauss_psi(lam, mu)

    def g_sandwich(lam: float) -> float:
        return gauss_sandwich(lam, mu)

    panels = [
        (left, die_psi, die_sandwich, die_kl(), "Loaded die against fair die"),
        (right, g_psi, g_sandwich, mu * mu / 2, f"N(0, 1) against N({mu:g}, 1)"),
    ]
    for ax, psi, sandwich, kl, title in panels:
        neg = [lam for lam in lams if lam < 0]
        pos = [lam for lam in lams if lam > 0]
        for part in (neg, pos):
            ax.plot(part, [psi(lam) / lam for lam in part], "k", linewidth=2)
            ax.plot(part, [sandwich(lam) for lam in part], color="C0")
        ax.plot([], [], "k", linewidth=2, label=r"$D_{1+\lambda} = \psi(\lambda)/\lambda$")
        ax.plot([], [], color="C0", label=r"$\mathrm{E}_P[(r^\lambda - 1)/\lambda]$")
        ax.axhline(kl, color="C3", linestyle="--", label=f"KL = {kl:.4f}")
        ax.axvline(0, color="gray", linewidth=0.5)
        ax.set(xlabel=r"$\lambda = \alpha - 1$", ylabel="nats", title=title)
        ax.legend(fontsize=8, loc="upper left")
    return fig


def plot_perspective(alpha: float = 2.0, eps_alpha: float = 1.0) -> Figure:
    fig, (left, right) = plt.subplots(1, 2, figsize=(11, 4.5))
    xs = np.linspace(0.0, 1.0, 201)
    ys = np.linspace(0.05, 1.0, 191)
    gx, gy = np.meshgrid(xs, ys)
    values = gx**alpha * gy ** (1 - alpha)
    contours = left.contour(gx, gy, values, levels=[0.05, 0.1, 0.2, 0.4, 0.8, 1.6, 3.2], cmap="viridis")
    left.clabel(contours, fontsize=7)
    p1, p2 = (0.9, 0.3), (0.1, 0.9)
    mid = ((p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2)
    g1, g2, gm = (renyi_term(x, y, alpha) for x, y in (p1, p2, mid))
    left.plot([p1[0], p2[0]], [p1[1], p2[1]], "k--", linewidth=1)
    for (x, y), label in (
        (p1, f"g = {g1:.2f}"),
        (p2, f"g = {g2:.2f}"),
        (mid, f"midpoint: g = {gm:.2f} ≤ {(g1 + g2) / 2:.2f}"),
    ):
        left.plot([x], [y], "ko")
        left.annotate(label, (x, y), textcoords="offset points", xytext=(5, 5), fontsize=8)
    left.set(xlabel="x", ylabel="y", title=r"Contours of $g(x, y) = x^2/y$ (order $\alpha = 2$)")

    a_grid = [i / 400 for i in range(1, 400)]
    b_vals = [f_rdp(a, alpha, eps_alpha) for a in a_grid]
    right.fill_between(a_grid, b_vals, [1 - a for a in a_grid], color="C0", alpha=0.25, label="allowed (a, b)")
    right.plot(a_grid, b_vals, "C0", label=r"$f_{\mathrm{RDP}}(a)$")
    right.plot([0, 1], [1, 0], "k:", linewidth=1, label="b = 1 - a (coin flip)")
    a1, a2 = 0.02, 0.4
    b1, b2 = f_rdp(a1, alpha, eps_alpha), f_rdp(a2, alpha, eps_alpha)
    right.plot([a1, a2], [b1, b2], "k--", linewidth=1, label="chord between two boundary points")
    right.plot([a1, a2], [b1, b2], "ko")
    right.set(
        xlim=(0, 1),
        ylim=(0, 1),
        xlabel="type I error a",
        ylabel="type II error b",
        title=rf"Tests allowed by order $\alpha = {alpha:g}$, $\varepsilon(\alpha) = {eps_alpha:g}$",
    )
    right.legend(fontsize=8)
    return fig


def plot_noisy_count() -> Figure:
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    y = np.linspace(-1.0, 2.0, 1201)
    lw1 = binomial_log_weights(1)
    axes[0].plot(y, np.exp(log_count_density(y, lw1, 0.2)), label="world 0: f(y)")
    axes[0].plot(y, np.exp(log_count_density(y - 1, lw1, 0.2)), label="world 1: f(y - 1)")
    axes[0].set(xlabel="observed count y", ylabel="density", title=r"Bernoulli($\frac{1}{2}$) + N(0, 0.2²): bimodal")
    axes[0].legend(fontsize=8)
    axes[1].plot(y, log_shift_ratio(y, lw1, 0.2), "k")
    axes[1].set(
        xlabel="observed count y", ylabel="ln f(y - 1)/f(y)", title="Its shift likelihood ratio still increases"
    )
    y100 = np.linspace(35.0, 66.0, 1241)
    lw100 = binomial_log_weights(100)
    for s, style in ((1.0, "k"), (0.5, "C0"), (0.3, "C1")):
        axes[2].plot(y100, log_shift_ratio(y100, lw100, s), style, linewidth=1.2, label=f"σ = {s:g}")
    axes[2].axvline(50.5, color="gray", linestyle=":", label="threshold 50.5")
    axes[2].axhline(0, color="gray", linewidth=0.5)
    axes[2].set(
        xlabel="observed count y", ylabel="ln f(y - 1)/f(y)", title="Bin(100, ½) + N(0, σ²): increasing for every σ"
    )
    axes[2].legend(fontsize=8)
    return fig


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=FIGURES, help="directory for the figures")
    args = parser.parse_args()
    seed: int = args.seed
    out: Path = args.out

    print_table(np.random.default_rng(seed))

    out.mkdir(parents=True, exist_ok=True)
    figures = {
        "tail_bernstein_mgf.svg": plot_bernstein_mgf(),
        "tail_bernstein_tail.svg": plot_bernstein_tail(),
        "tail_cramer.svg": plot_cramer(),
        "tail_stein.svg": plot_stein(),
        "tail_clopper_cdf.svg": plot_clopper_cdf(),
        "tail_clopper_coverage.svg": plot_clopper_coverage(),
        "tail_renyi_sandwich.svg": plot_renyi_sandwich(),
        "tail_perspective.svg": plot_perspective(),
        "tail_noisy_count.svg": plot_noisy_count(),
    }
    for name, fig in figures.items():
        fig.tight_layout()
        fig.savefig(out / name, metadata={"Date": None})  # no timestamp, so output is reproducible
        plt.close(fig)
        print(f"saved {out / name}")


if __name__ == "__main__":
    main()
