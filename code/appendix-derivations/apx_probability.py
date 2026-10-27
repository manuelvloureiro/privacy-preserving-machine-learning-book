"""Check the derivations of Appendix A (sums, limits and counting) and draw its figures.

    uv run python code/appendix-derivations/apx_probability.py
    uv run python code/appendix-derivations/apx_probability.py --seed 1 --out /tmp/figures

Prints the numbers quoted in book-full/appendix-derivations/05-probability.md (they match exactly
with the default seed), then saves the appendix's figures.
"""

import argparse
import cmath
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # for code/shared
from shared.figures import configure
from shared.stats import laplace_pdf, normal_cdf, normal_pdf

configure("apx_probability")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from numpy.typing import NDArray

# Figures are build output: _build/figures/<part>/, which the build copies into each book.
FIGURES = Path(__file__).resolve().parents[2] / "_build" / "figures" / Path(__file__).parent.name

FloatArray = NDArray[np.float64]


def trapezoid(values: FloatArray, step: float) -> float:
    """Trapezoid rule on an evenly spaced grid."""
    return float(step * (values.sum() - 0.5 * (values[0] + values[-1])))


# --- Sums by convolution, and rotation invariance --------------------------------------------


def poisson_pmf(mu: float, k: int) -> float:
    return math.exp(-mu + k * math.log(mu) - math.lgamma(k + 1))


def poisson_convolution(mu1: float, mu2: float, k: int) -> float:
    """Pr[N1 + N2 = k] by summing over the ways to split k, for independent Poissons."""
    return sum(poisson_pmf(mu1, j) * poisson_pmf(mu2, k - j) for j in range(k + 1))


def exponential_pdf(y: FloatArray, b: float = 1.0) -> FloatArray:
    return np.where(y >= 0, np.exp(-np.maximum(y, 0.0) / b) / b, 0.0)


def exp_difference_density(x: float, b: float = 1.0, upper: float = 60.0, steps: int = 200_001) -> float:
    """Density of Y1 - Y2 for independent Exp(b), as the overlap integral of f(y) and f(y - x)."""
    y = np.linspace(min(0.0, x), upper * b + abs(x), steps)
    values = exponential_pdf(y, b) * exponential_pdf(y - x, b)
    return trapezoid(values, float(y[1] - y[0]))


def gamma_pdf(x: float, n: int) -> float:
    """The Gamma(n, 1) density x^{n-1} e^{-x} / (n-1)!."""
    if x <= 0:
        return 0.0
    return math.exp((n - 1) * math.log(x) - x - math.lgamma(n))


def exp_sum_density_grid(n: int, step: float = 2e-3, upper: float = 30.0) -> tuple[FloatArray, FloatArray]:
    """Density of a sum of n independent Exp(1), by n - 1 numerical convolutions on a grid."""
    x = np.arange(0.0, upper, step)
    one = np.exp(-x)
    density = one.copy()
    for _ in range(n - 1):
        density = np.convolve(density, one)[: x.size] * step
        density[0] = 0.0  # the trapezoid correction at the left end, where the sum is 0
    return x, density


def normal_sum_density(s: float, sigma1: float, sigma2: float, steps: int = 200_001) -> float:
    """Density of X1 + X2 at s for independent N(0, sigma1^2), N(0, sigma2^2), by numerical convolution."""
    width = 12 * max(sigma1, sigma2)
    y = np.linspace(-width, width, steps)
    values = np.exp(-(y**2) / (2 * sigma1**2) - (s - y) ** 2 / (2 * sigma2**2)) / (2 * math.pi * sigma1 * sigma2)
    return trapezoid(values, float(y[1] - y[0]))


def rotation_projections(rng: np.random.Generator, theta: float, sigma: float, trials: int) -> FloatArray:
    """Coordinates of N(0, sigma^2 I_2) draws along u = (cos θ, sin θ) and v = (-sin θ, cos θ)."""
    x = rng.normal(0.0, sigma, size=(trials, 2))
    c, s = math.cos(theta), math.sin(theta)
    return np.stack([c * x[:, 0] + s * x[:, 1], -s * x[:, 0] + c * x[:, 1]], axis=1)


def laplace_difference_equal_density(z: float, eps: float) -> float:
    """Density of the difference of two independent Lap(1/ε): (ε/4)(1 + ε|z|) e^{-ε|z|}."""
    return eps / 4 * (1 + eps * abs(z)) * math.exp(-eps * abs(z))


def laplace_difference_density_numeric(z: float, b1: float, b2: float, steps: int = 400_001) -> float:
    """Density of ν + ρ at z for ν ~ Lap(b1), ρ ~ Lap(b2), by numerical convolution."""
    width = 60 * max(b1, b2) + abs(z)
    y = np.linspace(-width, width, steps)
    values = np.exp(-np.abs(y) / b1 - np.abs(z - y) / b2) / (4 * b1 * b2)
    return trapezoid(values, float(y[1] - y[0]))


# --- Stirling, the binomial mode and the peak height -----------------------------------------


def binomial_pmf(n: int, p: float) -> list[float]:
    """Pr[X = k] for k = 0..n, X ~ Bin(n, p), computed in log space so large n is safe."""
    q = 1 - p
    return [
        math.exp(
            math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1) + k * math.log(p) + (n - k) * math.log(q)
        )
        for k in range(n + 1)
    ]


def binomial_mode_matches(n_max: int, ps: list[float]) -> tuple[int, int]:
    """How many (n, p) pairs have floor((n+1)p) among the most likely counts, out of how many."""
    hits, total = 0, 0
    for n in range(1, n_max + 1):
        for p in ps:
            pmf = binomial_pmf(n, p)
            top = max(pmf)
            mode = math.floor((n + 1) * p)
            hits += pmf[mode] >= top * (1 - 1e-12)
            total += 1
    return hits, total


def stirling_remainder(n: int) -> float:
    """d_n - C = ln n! - (n + 1/2) ln n + n - ln sqrt(2π), which lies in (0, 1/(12n)]."""
    return math.lgamma(n + 1) - (n + 0.5) * math.log(n) + n - 0.5 * math.log(2 * math.pi)


def crude_stirling_bounds(n: int) -> tuple[float, float]:
    """Lower and upper bounds on ln n! from comparing the sum of ln k with the integral of ln x."""
    lower = n * math.log(n) - n + 1
    upper = (n + 1) * math.log(n + 1) - n
    return lower, upper


def wallis_ratio(j: int) -> float:
    """C(2j, j) 4^{-j} sqrt(πj), which tends to 1."""
    log_central = math.lgamma(2 * j + 1) - 2 * math.lgamma(j + 1) - 2 * j * math.log(2)
    return math.exp(log_central) * math.sqrt(math.pi * j)


def wallis_integral(n: int) -> float:
    """I_n = ∫_0^{π/2} sin^n x dx by the recursion I_n = (n-1)/n I_{n-2}."""
    value = math.pi / 2 if n % 2 == 0 else 1.0
    for k in range(2 if n % 2 == 0 else 3, n + 1, 2):
        value *= (k - 1) / k
    return value


def reconstruction_fraction(m: int) -> float:
    """Exact expected fraction of m fair bits recovered from their sum: 1/2 + E|K - m/2| / m."""
    pmf = binomial_pmf(m, 0.5)
    return 0.5 + sum(w * abs(k - m / 2) for k, w in enumerate(pmf)) / m


def reconstruction_fraction_closed(m: int) -> float:
    """The same through the telescoping identity: 1/2 + C(2j, j) 4^{-j} / 2 with j = floor(m/2), for any m."""
    j = m // 2
    log_central = math.lgamma(2 * j + 1) - 2 * math.lgamma(j + 1) - 2 * j * math.log(2)
    return 0.5 + 0.5 * math.exp(log_central)


def shift_tv(n: int, p: float) -> float:
    """TV distance between X and X + 1 for X ~ Bin(n, p): half the sum of |p_k - p_{k-1}|."""
    pmf = [0.0, *binomial_pmf(n, p), 0.0]
    return 0.5 * sum(abs(pmf[i] - pmf[i - 1]) for i in range(1, len(pmf)))


def binomial_mode_pmf(n: int, p: float) -> float:
    """Pr[X = floor((n+1)p)] for X ~ Bin(n, p), the height of the mode."""
    k = math.floor((n + 1) * p)
    log_pmf = (
        math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1) + k * math.log(p) + (n - k) * math.log(1 - p)
    )
    return math.exp(log_pmf)


def peak_ratio(n: int, p: float) -> float:
    """max_k Pr[X = k] times sqrt(2π n p q), which tends to 1. The mode is floor((n+1)p)."""
    return binomial_mode_pmf(n, p) * math.sqrt(2 * math.pi * n * p * (1 - p))


# --- The tail of a Poisson count -------------------------------------------------------------


def poisson_pmf_bound(mu: float, k: int) -> float:
    """The bound (eμ/k)^k on Pr[N = k], from k! >= (k/e)^k."""
    return (math.e * mu / k) ** k


def poisson_tail(mu: float, k: int) -> float:
    """Pr[N >= k] for N ~ Poisson(μ), which equals the regularized lower incomplete gamma P(k, μ)."""
    return regularized_gamma_p(k, mu)


def poisson_tail_bound(mu: float, k: int) -> float:
    """The geometric-series bound q^k / (1 - q) on Pr[N >= k], with q = eμ/k < 1."""
    q = math.e * mu / k
    return q**k / (1 - q)


def poisson_stirling_ratio(mu: float, k: int) -> tuple[float, float]:
    """The ratio bound / Pr[N = k] split by Stirling: e^μ sqrt(2πk) alone, and times e^{r_k}."""
    head = math.exp(mu) * math.sqrt(2 * math.pi * k)
    return head, head * math.exp(stirling_remainder(k))


def normal_upper_tail(x: float, mean: float, sd: float) -> float:
    """Pr[X > x] for X ~ N(mean, sd^2), through erfc so that far tails keep their precision."""
    return 0.5 * math.erfc((x - mean) / (sd * math.sqrt(2)))


def poisson_chernoff(mu: float, k: int) -> float:
    """The Chernoff bound e^{-μ} (eμ/k)^k on Pr[N >= k], for k > μ."""
    return math.exp(-mu) * poisson_pmf_bound(mu, k)


# --- Central limit theorem by Lindeberg's replacement ----------------------------------------

EXP_THIRD = 12 / math.e - 2  # E|X - 1|^3 for X ~ Exp(1)
GAUSS_THIRD = 2 * math.sqrt(2 / math.pi)  # E|G|^3 for G ~ N(0, 1)
SMOOTHSTEP_THIRD = 60.0  # largest |s'''| of s(u) = 6u^5 - 15u^4 + 10u^3 on [0, 1]


def hybrid_sin_mean(k: int, n: int) -> float:
    """E sin(H_k) for H_k = (X_1 + ... + X_k - k + G_{k+1} + ... + G_n) / sqrt(n), X_i ~ Exp(1), G_i ~ N(0, 1).

    Exact: E e^{iuH_k} = e^{-iuk/√n} (1 - iu/√n)^{-k} e^{-u²(n-k)/(2n)}, and E sin is its imaginary part at u = 1.
    """
    r = math.sqrt(n)
    char = cmath.exp(-1j * k / r) * (1 - 1j / r) ** (-k) * math.exp(-(n - k) / (2 * n))
    return char.imag


def lindeberg_swap_bound(n: int, h_third: float, third: float = EXP_THIRD) -> float:
    """Bound on |E h(H_k) - E h(H_{k-1})| for one swap: ‖h'''‖ (E|Y|^3 + E|G|^3) / (6 n^{3/2})."""
    return h_third * (third + GAUSS_THIRD) / (6 * n**1.5)


def smoothstep(u: FloatArray) -> FloatArray:
    """s(u) = 6u^5 - 15u^4 + 10u^3 on [0, 1], 0 below and 1 above: a ramp with bounded third derivative."""
    c = np.clip(u, 0.0, 1.0)
    return c**3 * (10 - 15 * c + 6 * c**2)


def lyapunov_ratio(n: int, third: float = EXP_THIRD) -> float:
    """L = sum of E|Y_i|^3 over s^3, for n identical summands of variance 1."""
    return third / math.sqrt(n)


def lindeberg_cdf_bound(lyapunov: float) -> tuple[float, float]:
    """Best width ε and the CDF-gap bound ε/√(2π) + 60 c L / ε^3 at that width, c = (1 + E|G|^3) / 6."""
    c = (1 + GAUSS_THIRD) / 6
    eps = (3 * SMOOTHSTEP_THIRD * c * lyapunov * math.sqrt(2 * math.pi)) ** 0.25
    return eps, eps / math.sqrt(2 * math.pi) + SMOOTHSTEP_THIRD * c * lyapunov / eps**3


def regularized_gamma_p(a: float, x: float) -> float:
    """P(a, x) = γ(a, x) / Γ(a), the regularized lower incomplete gamma function."""
    if x <= 0:
        return 0.0
    if x < a + 1:
        term = total = 1.0 / a
        ap = a
        for _ in range(100_000):
            ap += 1
            term *= x / ap
            total += term
            if abs(term) < abs(total) * 1e-16:
                break
        return total * math.exp(-x + a * math.log(x) - math.lgamma(a))
    return 1.0 - regularized_gamma_q(a, x)


def regularized_gamma_q(a: float, x: float) -> float:
    """Q(a, x) = 1 - P(a, x); a continued fraction (modified Lentz) in the upper range keeps tail precision."""
    if x < a + 1:
        return 1.0 - regularized_gamma_p(a, x)
    tiny = 1e-300
    b = x + 1 - a
    c = 1 / tiny
    d = 1 / b
    h = d
    for i in range(1, 100_000):
        an = -i * (i - a)
        b += 2
        d = an * d + b
        d = tiny if abs(d) < tiny else d
        c = b + an / c
        c = tiny if abs(c) < tiny else c
        d = 1 / d
        delta = d * c
        h *= delta
        if abs(delta - 1) < 1e-16:
            break
    return math.exp(-x + a * math.log(x) - math.lgamma(a)) * h


def clt_exp_cdf_gap(n: int, grid_step: float = 0.01) -> float:
    """Largest |F_n(z) - Φ(z)| over z in [-5, 5] for the standardized sum of n Exp(1), from the Gamma CDF."""
    gap = 0.0
    for i in range(round(10 / grid_step) + 1):
        z = -5 + i * grid_step
        f = regularized_gamma_p(n, n + z * math.sqrt(n))
        gap = max(gap, abs(f - normal_cdf(z)))
    return gap


# --- The norm of a Gaussian vector -----------------------------------------------------------


def chi_tail(d: int, r: float) -> float:
    """Pr[‖Z‖ >= r] for Z ~ N(0, I_d): ‖Z‖² is chi-squared with d degrees of freedom, a Gamma(d/2, 2)."""
    return regularized_gamma_q(d / 2, r * r / 2)


def chernoff_chi_square(d: int, x: float) -> float:
    """The Laurent–Massart bound on Pr[‖Z‖² - d >= 2 sqrt(dx) + 2x]: e^{-x}."""
    return math.exp(-x)


def chi_square_chernoff_exponent(lam: float, d: int, u: float) -> float:
    """-λu + d(-λ - ln(1 - 2λ)/2), the exact log-MGF form of the Chernoff exponent for S - d."""
    return -lam * u + d * (-lam - 0.5 * math.log(1 - 2 * lam))


def chi_square_relaxed_exponent(lam: float, d: int, u: float) -> float:
    """-λu + dλ²/(1 - 2λ), after the series bound on the log-MGF."""
    return -lam * u + d * lam * lam / (1 - 2 * lam)


def norm_offset_density(d: int, w: float) -> float:
    """Density of ‖Z‖ - sqrt(d) at w for Z ~ N(0, I_d): the chi density at r = sqrt(d) + w."""
    r = math.sqrt(d) + w
    if r <= 0:
        return 0.0
    log_f = (d - 1) * math.log(r) - r * r / 2 - (d / 2 - 1) * math.log(2) - math.lgamma(d / 2)
    return math.exp(log_f)


# --- Difference of two Laplace variables -----------------------------------------------------


def laplace_difference_tail(t: float, b1: float, b2: float) -> float:
    """Pr[ν - ρ > t] for t >= 0, ν ~ Lap(b1), ρ ~ Lap(b2) independent, b1 != b2."""
    return (b1**2 * math.exp(-t / b1) - b2**2 * math.exp(-t / b2)) / (2 * (b1**2 - b2**2))


def laplace_difference_tail_equal(t: float, b: float) -> float:
    """The same with b1 = b2 = b: (2 + t/b) e^{-t/b} / 4."""
    return (2 + t / b) * math.exp(-t / b) / 4


def laplace_difference_tail_numeric(t: float, b1: float, b2: float, steps: int = 400_001) -> float:
    """The tail by integrating the convolution density f(s) from t to far out, on a grid."""
    s = np.linspace(t, t + 80 * max(b1, b2), steps)
    density = (b1 * np.exp(-s / b1) - b2 * np.exp(-s / b2)) / (2 * (b1**2 - b2**2))
    return trapezoid(density, float(s[1] - s[0]))


def laplace_difference_samples(rng: np.random.Generator, b1: float, b2: float, trials: int) -> FloatArray:
    return rng.laplace(0.0, b1, trials) - rng.laplace(0.0, b2, trials)


def stable_threshold(pool: int, delta_bad: float, b1: float, b2: float) -> float:
    """Smallest t with pool * Pr[ν - ρ > t] <= delta_bad, by bisection (the tail decreases in t)."""
    lo, hi = 0.0, 1000.0 * max(b1, b2)
    for _ in range(200):
        mid = (lo + hi) / 2
        if pool * laplace_difference_tail(mid, b1, b2) > delta_bad:
            lo = mid
        else:
            hi = mid
    return hi


# --- The appendix's table --------------------------------------------------------------------


def print_table(rng: np.random.Generator) -> None:
    # Sums by convolution.
    gap = max(abs(poisson_convolution(2.0, 3.0, k) - poisson_pmf(5.0, k)) for k in range(40))
    print(f"[Poisson 2 + 3] largest |convolution - Poisson(5)| over k < 40 = {gap:.1e}")
    gap = max(abs(exp_difference_density(x) - laplace_pdf(x)) for x in (-3.0, -1.0, -0.2, 0.0, 0.5, 1.5, 4.0))
    print(f"[Exp - Exp] largest |overlap integral - Lap(1) density| = {gap:.1e}")
    x, density = exp_sum_density_grid(5)
    idx = np.arange(0, x.size, 250)
    gap = max(abs(float(density[i]) - gamma_pdf(float(x[i]), 5)) for i in idx)
    print(f"[Gamma] five Exp(1) convolved on a grid of step 0.002: largest gap to Gamma(5, 1) = {gap:.1e}")
    gap = max(
        abs(normal_sum_density(s, 1.0, 2.0) - normal_pdf(s / math.sqrt(5)) / math.sqrt(5))
        for s in (-4.0, 0.0, 1.0, 3.0)
    )
    print(f"[Normal 1 + 4] largest |convolution - N(0, 5) density| = {gap:.1e}")
    proj = rotation_projections(rng, math.pi / 6, 1.0, 1_000_000)
    var_u, var_v = float(proj[:, 0].var()), float(proj[:, 1].var())
    corr = float(np.mean(proj[:, 0] * proj[:, 1])) / math.sqrt(var_u * var_v)
    joint = float(np.mean((proj[:, 0] > 1) & (proj[:, 1] > 1)))
    single = 1 - normal_cdf(1.0)
    print(
        f"[Rotation θ=π/6] var u={var_u:.4f}  var v={var_v:.4f}  corr={corr:+.4f}  "
        f"Pr[both > 1]={joint:.4f}  product={single**2:.4f}"
    )
    gap = max(
        abs(laplace_difference_density_numeric(z, 1.0, 1.0) - laplace_difference_equal_density(z, 1.0))
        for z in (0.0, 0.7, 2.0, 5.0)
    )
    print(f"[Lap - Lap, equal scales] largest |convolution - (ε/4)(1 + ε|z|)e^(-ε|z|)| = {gap:.1e}")

    # Stirling and the binomial.
    hits, total = binomial_mode_matches(60, [0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9, 1 / 3])
    print(f"[Mode] floor((n+1)p) is a most likely count in {hits} of {total} pairs (n <= 60)")
    for n in (1, 10, 100, 1000):
        lo, hi = crude_stirling_bounds(n)
        print(
            f"[Stirling n={n}] ln n! = {math.lgamma(n + 1):.4f}  crude [{lo:.4f}, {hi:.4f}]  "
            f"sharp remainder={stirling_remainder(n):.3e}  1/(12n)={1 / (12 * n):.3e}"
        )
    for j in (5, 50, 500):
        print(
            f"[Wallis j={j}] C(2j,j) 4^-j sqrt(πj) = {wallis_ratio(j):.5f}  I_2j/I_2j+1 = {wallis_integral(2 * j) / wallis_integral(2 * j + 1):.5f}"
        )
    for m in (10, 11, 100, 1000):
        print(
            f"[Reconstruction m={m}] exact={reconstruction_fraction(m):.5f}  "
            f"closed={reconstruction_fraction_closed(m):.5f}  approx={0.5 + 1 / math.sqrt(2 * math.pi * m):.5f}"
        )
    for n, p in ((4, 0.5), (10, 0.5), (100, 0.5)):
        print(f"[Shift TV Bin({n},{p})] TV={shift_tv(n, p):.4f}  mode height={max(binomial_pmf(n, p)):.4f}")
    worst_peak = max(
        n * p * (1 - p) * abs(math.log(peak_ratio(n, p)))
        for p in (0.5, 0.3, 0.2, 0.1, 0.05, 0.01)
        for n in range(1, 20001)
        if n * p * (1 - p) >= 4
    )
    print(f"[Peak] largest npq |ln ratio| over npq >= 4, n <= 20000 = {worst_peak:.3f}  (proved: <= 3.5)")
    for p in (0.5, 0.2, 0.05):
        ratios = "  ".join(f"n={n}: {peak_ratio(n, p):.4f}" for n in (10, 100, 1000, 10000))
        print(f"[Peak p={p}] max_k p_k sqrt(2πnpq)  {ratios}")

    # The tail of a Poisson count.
    mu = 4.0
    for k in (20, 40):
        exact, bound = poisson_pmf(mu, k), poisson_pmf_bound(mu, k)
        tail, tail_bound = poisson_tail(mu, k), poisson_tail_bound(mu, k)
        head, full = poisson_stirling_ratio(mu, k)
        print(
            f"[Poisson μ=4 k={k}] Pr[N=k]={exact:.2e}  bound={bound:.2e}  ratio={bound / exact:.0f}  "
            f"Pr[N>=k]={tail:.2e}  bound={tail_bound:.2e}  ratio={tail_bound / tail:.0f}"
        )
        print(
            f"[Poisson μ=4 k={k}] e^μ sqrt(2πk)={head:.1f}  times e^(r_k)={full:.1f}  "
            f"1/(1 - q)={1 / (1 - math.e * mu / k):.2f}  normal N(4, 4) tail={normal_upper_tail(k, mu, 2.0):.2e}  "
            f"Chernoff={poisson_chernoff(mu, k):.2e} ({poisson_chernoff(mu, k) / tail:.1f} times the tail)"
        )

    # Central limit theorem.
    n = 30
    final = abs(hybrid_sin_mean(n, n) - hybrid_sin_mean(0, n))
    steps = [abs(hybrid_sin_mean(k, n) - hybrid_sin_mean(k - 1, n)) for k in range(1, n + 1)]
    swap = lindeberg_swap_bound(n, 1.0)
    print(
        f"[Lindeberg n={n}, h=sin] |E h(S) - E h(Z)| = {final:.4f}  bound = {n * swap:.4f}  "
        f"largest step = {max(steps):.5f}  per-swap bound = {swap:.5f}"
    )
    print(f"[Exp(1)] E|X - 1|^3 = {EXP_THIRD:.4f}  E|G|^3 = {GAUSS_THIRD:.4f}")
    for n in (30, 10**4, 10**8):
        lyap = lyapunov_ratio(n)
        eps, bound = lindeberg_cdf_bound(lyap)
        print(
            f"[CDF gap n={n}] L={lyap:.3e}  ε*={eps:.3f}  Lindeberg bound={bound:.4f}  Berry–Esseen 0.48 L={0.48 * lyap:.4f}"
        )
    print(f"[CDF gap n=30] exact largest gap = {clt_exp_cdf_gap(30):.4f}")

    # The norm of a Gaussian vector.
    for d in (1, 10, 100, 1000):
        cells = "  ".join(f"t={t:g}: {chi_tail(d, math.sqrt(d) + t):.2e}" for t in (1.0, 2.0, 3.0))
        print(f"[Norm d={d}] Pr[‖Z‖ >= √d + t]  {cells}")
    print("[Norm bound] e^(-t²/2)  " + "  ".join(f"t={t:g}: {math.exp(-t * t / 2):.2e}" for t in (1.0, 2.0, 3.0)))
    worst = max(
        chi_tail(d, math.sqrt(d) + t) / math.exp(-t * t / 2)
        for d in (1, 2, 5, 10, 100, 1000, 10000)
        for t in np.linspace(0, 6, 61)
    )
    print(f"[Norm] largest ratio exact / bound over d <= 10000, t in [0, 6] = {worst:.3f}")
    lam_gap = min(
        chi_square_relaxed_exponent(lam, 10, 5.0) - chi_square_chernoff_exponent(lam, 10, 5.0)
        for lam in np.linspace(0.0, 0.49, 50)
    )
    print(f"[Norm] smallest (relaxed - exact) Chernoff exponent over λ in [0, 0.49] = {lam_gap:.2e}")

    # Difference of two Laplace variables.
    eps0 = 1.0
    b1, b2 = 4 / eps0, 2 / eps0
    samples = laplace_difference_samples(rng, b1, b2, 1_000_000)
    t = 8.0
    print(
        f"[Laplace diff t={t:g}] formula={laplace_difference_tail(t, b1, b2):.4f}  "
        f"integral={laplace_difference_tail_numeric(t, b1, b2):.4f}  simulated={float(np.mean(samples > t)):.4f}"
    )
    for pool in (1000, 10000):
        print(f"[Threshold P={pool}] t = {stable_threshold(pool, 5e-6, b1, b2):.1f}")
    print(
        f"[Laplace diff equal b=1, t=3] tail={laplace_difference_tail_equal(3.0, 1.0):.4f}  "
        f"near-equal b1=1.001={laplace_difference_tail(3.0, 1.001, 1.0):.4f}"
    )


# --- Figures ---------------------------------------------------------------------------------


def plot_convolution_overlap() -> Figure:
    y = np.linspace(-3.0, 6.0, 1801)
    shifts = [-1.0, 0.0, 1.5]
    fig, axes = plt.subplots(1, 4, figsize=(13, 3.4))
    for ax, x in zip(axes[:3], shifts, strict=True):
        f = exponential_pdf(y)
        g = exponential_pdf(y - x)
        ax.plot(y, f, color="C0", label="f(y)")
        ax.plot(y, g, color="C1", label=f"f(y − x), x = {x:g}")
        ax.fill_between(y, f * g, color="C2", alpha=0.9, label="f(y) f(y − x)")
        area = exp_difference_density(x)
        ax.set(xlabel="y", ylim=(0, 1.15), title=f"x = {x:g}: area = {area:.3f}")
        ax.legend(loc="upper right", fontsize=8)
    xs = np.linspace(-4, 4, 401)
    right = axes[3]
    right.plot(xs, [laplace_pdf(float(v)) for v in xs], color="k", label="Lap(1) density")
    right.plot(shifts, [exp_difference_density(x) for x in shifts], "o", color="C2", label="areas at left")
    right.set(xlabel="x", ylim=(0, 0.6), title="Density of Y₁ − Y₂")
    right.legend(loc="upper right", fontsize=8)
    return fig


def plot_rotation(rng: np.random.Generator, theta: float = math.pi / 6) -> Figure:
    fig, (left, right) = plt.subplots(1, 2, figsize=(10, 4.2))
    grid = np.linspace(-3.2, 3.2, 201)
    gx, gy = np.meshgrid(grid, grid)
    left.contour(gx, gy, np.exp(-(gx**2 + gy**2) / 2) / (2 * math.pi), levels=6, colors="gray", linewidths=0.8)
    points = rng.normal(size=(400, 2))
    left.plot(points[:, 0], points[:, 1], ".", color="gray", alpha=0.4, markersize=3)
    u = (math.cos(theta), math.sin(theta))
    v = (-math.sin(theta), math.cos(theta))
    left.annotate(
        "", xy=(2.5 * u[0], 2.5 * u[1]), xytext=(0, 0), arrowprops={"arrowstyle": "->", "color": "C0", "lw": 2}
    )
    left.annotate(
        "", xy=(2.5 * v[0], 2.5 * v[1]), xytext=(0, 0), arrowprops={"arrowstyle": "->", "color": "C1", "lw": 2}
    )
    left.text(2.6 * u[0], 2.6 * u[1], "u", color="C0", fontsize=12)
    left.text(2.6 * v[0] - 0.3, 2.6 * v[1], "v", color="C1", fontsize=12)
    left.set(
        aspect="equal",
        xlim=(-3.2, 3.2),
        ylim=(-3.2, 3.2),
        xlabel="x₁",
        ylabel="x₂",
        title="Contours of N(0, I₂) are circles",
    )
    proj = rotation_projections(rng, theta, 1.0, 200_000)
    bins = np.linspace(-4.5, 4.5, 61)
    right.hist(proj[:, 0], bins=bins, density=True, histtype="step", color="C0", lw=1.5, label="⟨u, X⟩")
    right.hist(proj[:, 1], bins=bins, density=True, histtype="step", color="C1", lw=1.5, label="⟨v, X⟩")
    right.plot(bins, [normal_pdf(float(b)) for b in bins], "k--", label="N(0, 1) density")
    right.set(xlabel="coordinate", ylabel="density", title="Both projections are N(0, 1)")
    right.legend(fontsize=8)
    return fig


def plot_peak_height() -> Figure:
    ns = [round(10 ** (k / 8)) for k in range(8, 33)]
    fig, (left, right) = plt.subplots(1, 2, figsize=(10, 4))
    for p in (0.5, 0.2, 0.05):
        ratios = [peak_ratio(n, p) for n in ns]
        left.plot(ns, ratios, "o-", markersize=3, label=f"p = {p}")
        right.plot(
            [n * p * (1 - p) for n in ns], [abs(math.log(r)) for r in ratios], "o-", markersize=3, label=f"p = {p}"
        )
    left.axhline(1.0, color="k", linestyle="--", linewidth=1)
    left.set(
        xscale="log", xlabel="n", ylabel="mode height × √(2πnp(1 − p))", title="Mode height over normal peak height"
    )
    left.legend()
    npq = np.logspace(math.log10(4), 3.5, 50)
    right.plot(npq, 3.5 / npq, "k--", label="proved bound 3.5/(np(1 − p)), np(1 − p) ≥ 4")
    right.set(
        xscale="log", yscale="log", xlabel="np(1 − p)", ylabel="|ln ratio|", title="The error falls like 1/(np(1 − p))"
    )
    right.legend(fontsize=8)
    return fig


def plot_stirling_error() -> Figure:
    ns = [round(10 ** (k / 10)) for k in range(41)]
    ns = sorted(set(ns))
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(ns, [stirling_remainder(n) for n in ns], "o-", markersize=3, label="ln n! minus Stirling's formula")
    ax.plot(ns, [1 / (12 * n) for n in ns], "k--", label=r"$1/(12n)$")
    crude = [crude_stirling_bounds(n)[1] - crude_stirling_bounds(n)[0] for n in ns]
    ax.plot(ns, crude, ":", color="C1", label="gap between the crude bounds")
    ax.set(
        xscale="log", yscale="log", xlabel="n", ylabel="error in ln n!", title="Stirling's formula: error below 1/(12n)"
    )
    ax.legend()
    return fig


def plot_shift_tv(n: int = 10) -> Figure:
    pmf = binomial_pmf(n, 0.5)
    ks = list(range(n + 2))
    p = [*pmf, 0.0]
    shifted = [0.0, *pmf]
    fig, (top, bottom) = plt.subplots(2, 1, figsize=(7, 5.5), sharex=True)
    top.bar([k - 0.2 for k in ks], p, width=0.4, label="Pr[X = k]")
    top.bar([k + 0.2 for k in ks], shifted, width=0.4, label="Pr[X + 1 = k]")
    top.set(ylabel="probability", title=f"X ~ Bin({n}, 1/2) and its copy moved one step right")
    top.legend()
    diffs = [s - q for s, q in zip(shifted, p, strict=True)]
    colors = ["C2" if d > 0 else "lightgray" for d in diffs]
    bottom.bar(ks, diffs, color=colors)
    bottom.axhline(0, color="k", linewidth=0.8)
    total = sum(d for d in diffs if d > 0)
    bottom.set(
        xticks=ks,
        xlabel="k",
        ylabel="Pr[X + 1 = k] − Pr[X = k]",
        title=f"Green bars add up to {total:.4f}, the mode height Pr[X = {n // 2}]",
    )
    return fig


def plot_poisson_tail(mu: float = 4.0) -> Figure:
    ks = list(range(1, 46))
    far = [k for k in ks if k > math.e * mu]
    above = [k for k in ks if k > mu]
    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.plot(ks, [poisson_tail(mu, k) for k in ks], "o-", color="k", markersize=3, label=r"exact $\Pr[N \geq k]$")
    ax.plot(ks, [poisson_pmf_bound(mu, k) for k in ks], "--", color="C0", label=r"bound $(e\mu/k)^k$ on $\Pr[N = k]$")
    ax.plot(
        far, [poisson_tail_bound(mu, k) for k in far], "-", color="C1", label=r"tail bound $q^k/(1 - q)$, $k > e\mu$"
    )
    ax.plot(above, [poisson_chernoff(mu, k) for k in above], ":", color="C2", label=r"Chernoff $e^{-\mu}(e\mu/k)^k$")
    ax.plot(
        ks,
        [normal_upper_tail(k, mu, math.sqrt(mu)) for k in ks],
        "-.",
        color="C3",
        label="normal tail, same mean and variance",
    )
    ax.set(
        yscale="log",
        ylim=(1e-80, 10),
        xlabel="k",
        ylabel="probability",
        title=f"The tail of Poisson({mu:g}) against its bounds and the normal tail",
    )
    ax.legend(fontsize=8, loc="lower left")
    return fig


def plot_lindeberg_hybrids(n: int = 30) -> Figure:
    ks = list(range(n + 1))
    base = hybrid_sin_mean(0, n)
    errors = [abs(hybrid_sin_mean(k, n) - base) for k in ks]
    swap = lindeberg_swap_bound(n, 1.0)
    fig, ax = plt.subplots(figsize=(6.5, 4))
    ax.plot(ks, errors, "o-", markersize=4, label="|E h(H_k) − E h(Z)|, exact")
    ax.plot(ks, [k * swap for k in ks], "k--", label="sum of the per-swap bounds")
    ax.set(
        xlabel="k, the number of summands swapped in",
        ylabel="error",
        title=f"Replacing normals by Exp(1) one at a time, n = {n}, h = sin",
    )
    ax.legend()
    return fig


def plot_smoothing() -> Figure:
    x = np.linspace(-2.5, 2.5, 1001)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(x, [normal_pdf(float(v)) for v in x], color="gray", label="N(0, 1) density")
    ax.fill_between(
        x,
        [normal_pdf(float(v)) for v in x],
        where=(x >= 0) & (x <= 0.5),
        color="C1",
        alpha=0.25,
        label="Φ(κ) − Φ(0), κ = 0.5",
    )
    ax.step(x, (x <= 0).astype(float), where="post", color="k", lw=2, label="1[x ≤ 0]")
    for eps, color in ((0.5, "C1"), (0.2, "C0")):
        ax.plot(x, 1 - smoothstep(x / eps), color=color, label=f"upper ramp, κ = {eps}")
        ax.plot(x, 1 - smoothstep((x + eps) / eps), color=color, linestyle="--", label=f"lower ramp, κ = {eps}")
    ax.set(xlabel="x", ylim=(-0.05, 1.1), title="Smooth ramps sandwich the indicator")
    ax.legend(fontsize=8, loc="upper right")
    return fig


def plot_norm_tails() -> Figure:
    ts = np.linspace(0.0, 4.0, 81)
    fig, ax = plt.subplots(figsize=(6.5, 4))
    for d in (1, 10, 100, 1000):
        ax.plot(ts, [chi_tail(d, math.sqrt(d) + float(t)) for t in ts], label=f"d = {d}")
    ax.plot(ts, np.exp(-(ts**2) / 2), "k--", lw=2, label=r"bound $e^{-t^2/2}$")
    ax.set(
        yscale="log",
        ylim=(1e-6, 1.5),
        xlabel="t",
        ylabel="Pr[‖Z‖ ≥ √d + t]",
        title="The bound holds in every dimension",
    )
    ax.legend()
    return fig


def plot_norm_density() -> Figure:
    ws = np.linspace(-4.0, 4.0, 801)
    fig, ax = plt.subplots(figsize=(6.5, 4))
    for d in (1, 10, 100, 1000):
        ax.plot(ws, [norm_offset_density(d, float(w)) for w in ws], label=f"d = {d}, √d = {math.sqrt(d):.1f}")
    ax.plot(ws, [normal_pdf(float(w) * math.sqrt(2)) * math.sqrt(2) for w in ws], "k:", label="normal, variance 1/2")
    ax.set(xlabel="‖Z‖ − √d", ylabel="density", title="The length sits within O(1) of √d")
    ax.legend(fontsize=8)
    return fig


def plot_laplace_difference(rng: np.random.Generator, eps0: float = 1.0) -> Figure:
    b1, b2 = 4 / eps0, 2 / eps0
    samples = laplace_difference_samples(rng, b1, b2, 1_000_000)
    ts = np.linspace(0.0, 90.0, 361)
    threshold = stable_threshold(10_000, 5e-6, b1, b2)
    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.plot(
        ts, [laplace_difference_tail(float(t), b1, b2) for t in ts], color="k", lw=2, label="formula, Pr[ν − ρ > τ]"
    )
    points = np.arange(0.0, 56.0, 4.0)
    sim = [float(np.mean(samples > t)) for t in points]
    ax.plot(points, [s if s > 0 else math.nan for s in sim], "o", color="C3", label="simulation, 10⁶ pairs")
    ax.plot(ts, 0.5 * np.exp(-ts / b1), "--", color="C0", label=f"Pr[ν > τ], ν ~ Lap({b1:g})")
    ax.plot(ts, 0.5 * np.exp(-ts / b2), "--", color="C1", label=f"Pr[ρ > τ], ρ ~ Lap({b2:g})")
    ax.axvline(threshold, color="gray", linestyle=":")
    ax.axhline(5e-6 / 10_000, color="gray", linestyle=":")
    ax.text(threshold - 1, 2e-8, f"τ = {threshold:.1f}\n(P = 10,000)", ha="right", fontsize=8)
    ax.set(
        yscale="log",
        ylim=(1e-11, 1),
        xlabel="τ",
        ylabel="tail probability",
        title=f"Difference of Laplace noises, ε₀ = {eps0:g}",
    )
    ax.legend(fontsize=8)
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
        "apx_prob_convolution.svg": plot_convolution_overlap(),
        "apx_prob_rotation.svg": plot_rotation(rng),
        "apx_prob_shift_tv.svg": plot_shift_tv(),
        "apx_prob_stirling.svg": plot_stirling_error(),
        "apx_prob_peak.svg": plot_peak_height(),
        "apx_prob_poisson_tail.svg": plot_poisson_tail(),
        "apx_prob_lindeberg.svg": plot_lindeberg_hybrids(),
        "apx_prob_smoothing.svg": plot_smoothing(),
        "apx_prob_norm_tails.svg": plot_norm_tails(),
        "apx_prob_norm_density.svg": plot_norm_density(),
        "apx_prob_laplace_diff.svg": plot_laplace_difference(rng),
    }
    for name, fig in figures.items():
        fig.tight_layout()
        fig.savefig(out / name, metadata={"Date": None})  # no timestamp, so output is reproducible
        plt.close(fig)
        print(f"saved {out / name}")


if __name__ == "__main__":
    main()
