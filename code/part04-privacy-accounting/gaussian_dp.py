"""Check the results in the f-DP and Gaussian differential privacy chapter and draw its figures.

    uv run python code/part04-privacy-accounting/gaussian_dp.py
    uv run python code/part04-privacy-accounting/gaussian_dp.py --seed 1 --out /tmp/figures

Prints every number quoted in 04-gaussian-dp.md (they match exactly with the default seed),
then saves the chapter's figures.

Conventions. P = M(D) and Q = M(D') are the output distributions on two neighbours. A test
rejects "the data is D" when it fires; its type I error is a = E_P[φ] and its type II error is
b = 1 - E_Q[φ], which the chapter writes FPR and FNR because it keeps α for Rényi orders.
The trade-off function T(P, Q)(a) is the smallest b at type I error at most a. The privacy loss
is L = ln(p/q), and the privacy loss random variable is L(Y) with Y ~ P.
"""

import argparse
import math
import sys
from collections.abc import Callable, Sequence
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # for code/shared
from shared.figures import configure
from shared.stats import normal_cdf

configure("gaussian_dp")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from numpy.typing import NDArray

# Figures are build output: _build/figures/<part>/, which the build copies into each book.
FIGURES = Path(__file__).resolve().parents[2] / "_build" / "figures" / Path(__file__).parent.name

# The RDP orders shared by every chapter of the part.
ORDERS: list[float] = [1 + x / 10 for x in range(1, 100)] + [float(x) for x in range(12, 64)] + [128.0, 256.0, 512.0]

# The running example of the part: DP-SGD with sampling rate q, noise multiplier σ, target δ.
SGD_Q = 0.01
SGD_SIGMA = 1.0
SGD_DELTA = 1e-5
SGD_STEPS: tuple[int, ...] = (100, 1000, 10_000)

# --- Normal distribution (no SciPy) ----------------------------------------------------------


def normal_isf(p: float) -> float:
    """Φ⁻¹(1 - p): the x with Pr[Z > x] = p, by bisection (precise even for tiny p)."""
    if not 0.0 < p < 1.0:
        raise ValueError("p must lie in (0, 1)")
    lo, hi = -40.0, 40.0
    for _ in range(200):
        mid = (lo + hi) / 2
        if mid in (lo, hi):
            break
        if normal_cdf(-mid) > p:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def bisect_decreasing(g: Callable[[float], float], target: float, lo: float, hi: float, iters: int = 200) -> float:
    """The x in [lo, hi] with g(x) = target, for g decreasing; returns the end where g <= target."""
    for _ in range(iters):
        mid = (lo + hi) / 2
        if mid in (lo, hi):
            break
        if g(mid) > target:
            lo = mid
        else:
            hi = mid
    return hi


# --- Section 30.3: f-DP, and (ε, δ)-DP as a trade-off function --------------------------------------


def eps_delta_tradeoff(eps: float, delta: float, a: float) -> float:
    """f_{ε,δ}(a) = max(0, 1 - δ - e^ε a, e^{-ε}(1 - δ - a)): the trade-off curve of (ε, δ)-DP."""
    return max(0.0, 1 - delta - math.exp(eps) * a, math.exp(-eps) * (1 - delta - a))


def laplace_tradeoff(eps: float, a: float) -> float:
    """T(Lap(0, 1/ε), Lap(1, 1/ε))(a): the Laplace mechanism with sensitivity 1 (Part 1, κ = ε)."""
    if a < 0.5 * math.exp(-eps):
        return 1 - math.exp(eps) * a
    if a <= 0.5:
        return math.exp(-eps) / (4 * a)
    return math.exp(-eps) * (1 - a)


def randomized_response_pmfs(eps: float) -> tuple[list[float], list[float]]:
    """Binary randomized response: report the true bit with probability e^ε/(1 + e^ε).

    P is the output distribution when the bit is 0, Q when it is 1, as pmfs on {0, 1}.
    """
    keep = math.exp(eps) / (1 + math.exp(eps))
    return [keep, 1 - keep], [1 - keep, keep]


def tradeoff_breakpoints(p: Sequence[float], q: Sequence[float]) -> list[tuple[float, float]]:
    """Corners (a, b) of T(P, Q) for pmfs on a finite set: add outcomes in decreasing order of q/p."""
    order = sorted(range(len(p)), key=lambda i: -(q[i] / p[i]) if p[i] > 0 else -math.inf)
    a, power = 0.0, 0.0
    points = [(0.0, 1.0)]
    for i in order:
        a += p[i]
        power += q[i]
        points.append((min(a, 1.0), max(1.0 - power, 0.0)))
    return points


def piecewise_linear(points: Sequence[tuple[float, float]], a: float) -> float:
    """Evaluate the polygon through the corners (sorted by first coordinate) at a."""
    for (a0, b0), (a1, b1) in zip(points, points[1:], strict=False):
        if a0 <= a <= a1:
            return b0 if a1 == a0 else b0 + (b1 - b0) * (a - a0) / (a1 - a0)
    return points[-1][1]


# --- Section 30.4: Gaussian differential privacy -----------------------------------------------------


def gaussian_tradeoff(mu: float, a: float) -> float:
    """G_μ(a) = Φ(Φ⁻¹(1 - a) - μ), the trade-off curve of N(0, 1) against N(μ, 1)."""
    if a <= 0.0:
        return 1.0
    if a >= 1.0:
        return 0.0
    return normal_cdf(normal_isf(a) - mu)


def gaussian_mechanism_mu(sigma: float, shift: float = 1.0) -> float:
    """μ of the Gaussian mechanism on a pair whose outputs differ by `shift` sensitivities: shift/σ."""
    return abs(shift) / sigma


def simulate_projected_test(rng: np.random.Generator, mus: Sequence[float], a: float, n: int) -> tuple[float, float]:
    """Errors of the projected likelihood-ratio test on N(0, I) against N(m, I), from n draws each.

    The test rejects when <y, m>/|m| > Φ⁻¹(1 - a); returns the simulated (type I, type II) errors.
    """
    m = np.asarray(mus, dtype=float)
    norm = float(np.linalg.norm(m))
    t = normal_isf(a)
    y0 = rng.standard_normal((n, len(mus)))
    y1 = rng.standard_normal((n, len(mus))) + m
    a_hat = float(np.mean(y0 @ m / norm > t))
    b_hat = float(np.mean(y1 @ m / norm <= t))
    return a_hat, b_hat


# --- Section 30.5: duality with (ε, δ) ----------------------------------------------------------------


def gdp_delta(mu: float, eps: float) -> float:
    """δ(ε) = Φ(-ε/μ + μ/2) - e^ε Φ(-ε/μ - μ/2): the (ε, δ) profile of μ-GDP (DRS Corollary 2.13)."""
    if mu <= 0.0:
        return 0.0
    return max(0.0, normal_cdf(-eps / mu + mu / 2) - math.exp(eps) * normal_cdf(-eps / mu - mu / 2))


def gdp_eps(mu: float, delta: float) -> float:
    """The smallest ε with δ(ε) <= δ for μ-GDP, by bisection on the decreasing profile."""
    if gdp_delta(mu, 0.0) <= delta:
        return 0.0
    hi = 1.0
    while gdp_delta(mu, hi) > delta:
        hi *= 2
    return bisect_decreasing(lambda e: gdp_delta(mu, e), delta, 0.0, hi)


def gdp_delta_hockey_stick(mu: float, eps: float, points: int = 400_001) -> float:
    """E_{e^ε}(P‖Q) = E_P[(1 - e^{ε - L})_+] with L ~ N(μ²/2, μ²), by the trapezoid rule."""
    loss = np.linspace(mu * mu / 2 - 40 * mu, mu * mu / 2 + 40 * mu, points)
    density = np.exp(-((loss - mu * mu / 2) ** 2) / (2 * mu * mu)) / (mu * math.sqrt(2 * math.pi))
    return float(np.trapezoid(density * np.maximum(0.0, 1 - np.exp(eps - loss)), loss))


def conjugate_delta(f: Callable[[float], float], eps: float, grid: Sequence[float]) -> tuple[float, float]:
    """1 + f*(-e^ε) = max_a (1 - f(a) - e^ε a), over a grid of type I errors; returns (δ, maximizing a)."""
    best, best_a = -math.inf, 0.0
    for a in grid:
        gap = 1 - f(a) - math.exp(eps) * a
        if gap > best:
            best, best_a = gap, a
    return best, best_a


def gdp_tangent_point(mu: float, eps: float) -> tuple[float, float]:
    """Where the line of slope -e^ε touches G_μ: a = Φ(-ε/μ - μ/2), b = Φ(ε/μ - μ/2)."""
    return normal_cdf(-eps / mu - mu / 2), normal_cdf(eps / mu - mu / 2)


def envelope_of_profile(mu: float, a: float, eps_grid: Sequence[float]) -> float:
    """sup over ε in the grid of f_{ε,δ(ε)}(a): the curve rebuilt from the (ε, δ) profile."""
    return max(eps_delta_tradeoff(e, gdp_delta(mu, e), a) for e in eps_grid)


def gdp_mu_for_eps(eps: float, delta: float) -> float:
    """The largest μ whose GDP profile gives δ(ε) <= δ at this ε, by bisection (δ(ε) grows with μ)."""
    return bisect_decreasing(lambda m: -gdp_delta(m, eps), -delta, 1e-9, 100.0)


# --- Section 30.6: composition --------------------------------------------------------------------


def compose_gdp(mus: Sequence[float]) -> float:
    """G_{μ1} ⊗ ... ⊗ G_{μk} = G_μ with μ = sqrt(μ1² + ... + μk²) (DRS Corollary 3.3)."""
    return math.sqrt(sum(m * m for m in mus))


def product_pmf(p: Sequence[float], k: int) -> list[float]:
    """The k-fold product distribution of a pmf, flattened over all outcome tuples."""
    out = [1.0]
    for _ in range(k):
        out = [x * y for x in out for y in p]
    return out


def naive_composition_eps(mu: float, k: int, delta: float) -> float:
    """Compose k copies of μ-GDP as (ε, δ/k) points with basic composition: k ε(δ/k)."""
    return k * gdp_eps(mu, delta / k)


# --- Section 30.7: the central limit theorem for composition ------------------------------------------


def log_binomial_pmf(k: int, p: float) -> NDArray[np.float64]:
    j = np.arange(k + 1)
    log_choose = np.array([math.lgamma(k + 1) - math.lgamma(i + 1) - math.lgamma(k - i + 1) for i in range(k + 1)])
    return log_choose + j * math.log(p) + (k - j) * math.log1p(-p)


def rr_composed_loss_pmf(eps0: float, k: int) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """The privacy loss of k-fold randomized response under P: L = ε0(2B - k), B ~ Bin(k, e^ε0/(1 + e^ε0))."""
    keep = math.exp(eps0) / (1 + math.exp(eps0))
    j = np.arange(k + 1, dtype=np.float64)
    loss: NDArray[np.float64] = eps0 * (2 * j - k)
    return loss, np.exp(log_binomial_pmf(k, keep))


def delta_from_loss(loss: NDArray[np.float64], prob: NDArray[np.float64], eps: float) -> float:
    """δ(ε) = E_P[(1 - e^{ε - L})_+]: the hockey-stick divergence from the law of L under P."""
    return float(np.sum(prob * np.maximum(0.0, -np.expm1(eps - loss))))


def rr_composed_delta(eps0: float, k: int, eps: float) -> float:
    """Exact δ(ε) of k-fold ε0-randomized response."""
    loss, prob = rr_composed_loss_pmf(eps0, k)
    return delta_from_loss(loss, prob, eps)


def rr_composed_tradeoff(eps0: float, k: int, a: float) -> float:
    """Exact f_{ε0,0}^{⊗k}(a): the LR test thresholds the number B of reported ones."""
    keep = math.exp(eps0) / (1 + math.exp(eps0))
    p = np.exp(log_binomial_pmf(k, keep))  # B under P
    q = p[::-1]  # B under Q is Bin(k, 1 - keep), the mirror image
    # Under P ones are likely, so q/p is largest at B = 0: reject B <= J.
    return piecewise_linear(tradeoff_breakpoints(p.tolist(), q.tolist()), a)


def laplace_loss_pmf(eps0: float, bins: int, round_up: bool) -> tuple[NDArray[np.float64], float]:
    """Privacy loss of the Laplace mechanism Lap(0, 1/ε0) vs Lap(1, 1/ε0) under P, on a grid.

    L = ε0(|y - 1| - |y|): an atom 1/2 at +ε0 (y <= 0), an atom e^{-ε0}/2 at -ε0 (y >= 1), and
    L = ε0(1 - 2y) for y in (0, 1). The grid has spacing h = ε0/bins and covers [-ε0, ε0]; the
    continuous part is rounded up (or down) to the grid, which bounds δ(ε) from above (below).
    Returns the probabilities on the 2 bins + 1 grid points and h.
    """
    h = eps0 / bins
    pmf = np.zeros(2 * bins + 1)
    pmf[-1] += 0.5
    pmf[0] += 0.5 * math.exp(-eps0)
    # Exact mass of each loss cell (l_i, l_{i+1}]: y ranges over an interval, y = (1 - l/ε0)/2.
    edges = np.linspace(-eps0, eps0, 2 * bins + 1)
    y_hi = (1 - edges[:-1] / eps0) / 2
    y_lo = (1 - edges[1:] / eps0) / 2
    mass = 0.5 * (np.exp(-eps0 * y_lo) - np.exp(-eps0 * y_hi))  # Pr_P[y_lo < Y < y_hi]
    if round_up:
        pmf[1:] += mass
    else:
        pmf[:-1] += mass
    return pmf, h


def compose_loss_pmf(pmf: NDArray[np.float64], k: int) -> NDArray[np.float64]:
    """The pmf of a sum of k independent copies of a lattice variable, by FFT (exact support, no wrap)."""
    size = k * (len(pmf) - 1) + 1
    n = 1 << (size - 1).bit_length()
    out = np.fft.irfft(np.fft.rfft(pmf, n) ** k, n)[:size]
    return np.maximum(out, 0.0)


def laplace_composed_delta(eps0: float, k: int, eps: float, bins: int = 400) -> tuple[float, float]:
    """Lower and upper bounds on δ(ε) for k-fold Laplace(1/ε0), from rounding the loss down and up."""
    out: list[float] = []
    for up in (False, True):
        pmf, h = laplace_loss_pmf(eps0, bins, up)
        composed = compose_loss_pmf(pmf, k)
        loss: NDArray[np.float64] = (np.arange(len(composed), dtype=np.float64) - k * bins) * h
        out.append(delta_from_loss(loss, composed, eps))
    return out[0], out[1]


def eps_from_profile(delta_of_eps: Callable[[float], float], delta: float, hi: float = 50.0) -> float:
    """The smallest ε with δ(ε) <= δ, for a decreasing privacy profile."""
    return bisect_decreasing(delta_of_eps, delta, 0.0, hi, iters=60)


def rr_functionals(eps0: float) -> tuple[float, float, float]:
    """(kl, κ2, κ̄3) of f_{ε0,0}: moments of the loss ±ε0 under P (DRS §3.2)."""
    keep = math.exp(eps0) / (1 + math.exp(eps0))
    kl = eps0 * (2 * keep - 1)
    kappa2 = eps0 * eps0
    kappa3_bar = keep * abs(eps0 - kl) ** 3 + (1 - keep) * abs(-eps0 - kl) ** 3
    return kl, kappa2, kappa3_bar


def laplace_functionals(eps0: float, points: int = 200_001) -> tuple[float, float, float]:
    """(kl, κ2, κ̄3) of the Laplace mechanism's trade-off curve, from the law of its loss under P."""
    y = np.linspace(0.0, 1.0, points)
    loss = eps0 * (1 - 2 * y)
    density = 0.5 * eps0 * np.exp(-eps0 * y)
    w_hi, w_lo = 0.5, 0.5 * math.exp(-eps0)

    def moment(g: Callable[[NDArray[np.float64]], NDArray[np.float64]]) -> float:
        atoms = w_hi * float(g(np.array([eps0]))[0]) + w_lo * float(g(np.array([-eps0]))[0])
        return atoms + float(np.trapezoid(g(loss) * density, y))

    kl = moment(lambda x: x)
    kappa2 = moment(lambda x: x * x)
    kappa3_bar = moment(lambda x: np.abs(x - kl) ** 3)
    return kl, kappa2, kappa3_bar


def clt_mu(kl_sum: float, kappa2_sum: float, kl_sq_sum: float) -> float:
    """DRS Theorem 3.4: μ = 2‖kl‖₁ / sqrt(‖κ2‖₁ - ‖kl‖₂²), twice the mean over the sd of the loss."""
    return 2 * kl_sum / math.sqrt(kappa2_sum - kl_sq_sum)


def berry_esseen_gamma(kappa3_bar_sum: float, kappa2_sum: float, kl_sq_sum: float) -> float:
    """DRS Theorem 3.4: γ = 0.56 ‖κ̄3‖₁ / (‖κ2‖₁ - ‖kl‖₂²)^{3/2}, the uniform error band."""
    return 0.56 * kappa3_bar_sum / (kappa2_sum - kl_sq_sum) ** 1.5


def clt_mu_homogeneous(functionals: tuple[float, float, float], k: int) -> tuple[float, float]:
    """(μ, γ) of Theorem 3.4 for k copies of one trade-off function with these functionals."""
    kl, kappa2, kappa3_bar = functionals
    return clt_mu(k * kl, k * kappa2, k * kl * kl), berry_esseen_gamma(k * kappa3_bar, k * kappa2, k * kl * kl)


# --- Section 30.8: the subsampled Gaussian CLT --------------------------------------------------------


def clt_mu_subsampled_gaussian(q: float, sigma: float, steps: int) -> float:
    """Bu, Dong, Long & Su (2020): k steps of Poisson-subsampled Gaussian are ≈ μ-GDP, μ = q sqrt(k(e^{1/σ²} - 1))."""
    return q * math.sqrt(steps * math.expm1(1 / sigma**2))


def drs_fixed_batch_mu(q: float, sigma: float, steps: int) -> float:
    """DRS Corollary 5.4 for their symmetrized subsampling: μ = sqrt(2) q sqrt(k) sqrt(χ²₊(G_{1/σ}))."""
    m = 1 / sigma
    chi2 = math.exp(m * m) * normal_cdf(1.5 * m) + 3 * normal_cdf(-0.5 * m) - 2
    return math.sqrt(2) * q * math.sqrt(steps) * math.sqrt(chi2)


def subsampled_loss_moments(q: float, sigma: float, remove: bool, points: int = 400_001) -> tuple[float, float]:
    """(mean, variance) of one step's privacy loss for the Poisson-subsampled Gaussian, by quadrature.

    The pair is N(0, σ²) and the mixture (1 - q)N(0, σ²) + qN(1, σ²). As in numerical_accounting.py,
    remove=True puts the record-containing dataset first, so P is the mixture; with remove=False
    (add), P is N(0, σ²).
    """
    y = np.linspace(-40 * sigma, 1 + 40 * sigma, points)
    log_base = -(y**2) / (2 * sigma**2)
    log_shift = -((y - 1) ** 2) / (2 * sigma**2)
    log_mix = np.logaddexp(math.log1p(-q) + log_base, math.log(q) + log_shift)
    norm = 1 / (sigma * math.sqrt(2 * math.pi))
    if remove:
        density, loss = norm * np.exp(log_mix), log_mix - log_base
    else:
        density, loss = norm * np.exp(log_base), log_base - log_mix
    mean = float(np.trapezoid(density * loss, y))
    var = float(np.trapezoid(density * (loss - mean) ** 2, y))
    return mean, var


def subsampled_loss_lattice(
    q: float, sigma: float, remove: bool, round_up: bool, h: float, size: int
) -> NDArray[np.float64]:
    """One step's privacy loss on the lattice hZ, stored circularly in `size` cells (index mod size).

    P(y) and Q(y) are the two output densities (the mixture and N(0, σ²), in the order set by
    `remove` as in subsampled_loss_moments). y is integrated on a fine grid; each loss value is
    rounded up (or down) to the lattice, which can only raise (or lower) δ(ε) of the composition.
    """
    y = np.linspace(-9 * sigma, 1 + 9 * sigma, 1_000_001)
    log_base = -(y**2) / (2 * sigma**2)
    log_mix = np.logaddexp(math.log1p(-q) + log_base, math.log(q) - ((y - 1) ** 2) / (2 * sigma**2))
    log_p, log_other = (log_mix, log_base) if remove else (log_base, log_mix)
    weight = np.exp(log_p - log_p.max())
    weight /= weight.sum()
    scaled = (log_p - log_other) / h
    rounded: NDArray[np.float64] = np.ceil(scaled) if round_up else np.floor(scaled)
    index: NDArray[np.int64] = rounded.astype(np.int64) % size
    pmf = np.zeros(size)
    np.add.at(pmf, index, weight)
    return pmf


def lattice_eps(pmf_one: NDArray[np.float64], steps: int, h: float, delta: float) -> float:
    """ε at δ for the k-fold composition of a circular lattice loss, by FFT and bisection."""
    size = len(pmf_one)
    composed = np.maximum(np.fft.irfft(np.fft.rfft(pmf_one) ** steps, size), 0.0)
    index = np.arange(size)
    loss = np.where(index >= size // 2, index - size, index) * h

    def profile(eps: float) -> float:
        tail = loss > eps
        return float(np.sum(composed[tail] * -np.expm1(eps - loss[tail])))

    return eps_from_profile(profile, delta, hi=40.0)


def subsampled_gaussian_eps_bracket(
    q: float, sigma: float, steps: Sequence[int], delta: float, h: float = 1e-5, width: float = 20.0
) -> list[tuple[float, float]]:
    """Numerical lower and upper values of ε at δ for k steps of the Poisson-subsampled Gaussian.

    Takes the worse of the two neighbouring directions. Losses beyond ±width are aliased, which is
    harmless here because the composed loss stays far inside. A preview of chapter 34's accountants.
    """
    size = int(2 * width / h)
    brackets: list[tuple[float, float]] = [(0.0, 0.0) for _ in steps]
    for remove in (True, False):
        low = subsampled_loss_lattice(q, sigma, remove, False, h, size)
        high = subsampled_loss_lattice(q, sigma, remove, True, h, size)
        for i, k in enumerate(steps):
            lo_eps, hi_eps = lattice_eps(low, k, h, delta), lattice_eps(high, k, h, delta)
            brackets[i] = (max(brackets[i][0], lo_eps), max(brackets[i][1], hi_eps))
    return brackets


# --- Section 30.9: RDP as a lossy summary of the trade-off curve ---------------------------------------


def binary_renyi(alpha: float, x: float, y: float) -> float:
    """D_α(Bern(x)‖Bern(y)) for α > 1, computed in logs; ∞ when y ∈ {0, 1} puts no mass where x does."""
    return float(binary_renyi_orders(np.array([alpha]), x, y)[0])


def binary_renyi_orders(alphas: NDArray[np.float64], x: float, y: float) -> NDArray[np.float64]:
    """D_α(Bern(x)‖Bern(y)) at every order in `alphas` (all > 1) at once."""
    terms: list[NDArray[np.float64]] = []
    for px, py in ((x, y), (1 - x, 1 - y)):
        if px <= 0.0:
            continue
        if py <= 0.0:
            return np.full(len(alphas), math.inf)
        terms.append(alphas * math.log(px) + (1 - alphas) * math.log(py))
    total = terms[0] if len(terms) == 1 else np.logaddexp(terms[0], terms[1])
    return total / (alphas - 1)


def rdp_feasible(a: float, b: float, rdp: Callable[[float], float], orders: Sequence[float]) -> bool:
    """Can a test have errors (a, b) under a mechanism with this RDP curve (both directions)?

    The test turns (P, Q) into (Bern(a), Bern(1 - b)); post-processing cannot raise D_α.
    """
    alphas = np.asarray(orders, dtype=float)
    bound = np.array([rdp(x) for x in orders])
    forward = binary_renyi_orders(alphas, a, 1 - b)
    backward = binary_renyi_orders(alphas, 1 - b, a)
    return bool(np.all(forward <= bound) and np.all(backward <= bound))


def rdp_tradeoff(rdp: Callable[[float], float], a: float, orders: Sequence[float] = ORDERS) -> float:
    """The lowest type II error at type I error a that the RDP curve allows: the best curve it implies.

    Every feasible (a, b) is attained by the Bernoulli pair (Bern(a), Bern(1 - b)), so this is exact.
    """
    if a >= 1.0:
        return 0.0
    lo, hi = 0.0, 1.0 - a  # b = 1 - a is always feasible (identical distributions)
    for _ in range(60):
        mid = (lo + hi) / 2
        if rdp_feasible(a, mid, rdp, orders):
            hi = mid
        else:
            lo = mid
    return hi


def rdp_implied_delta(
    rdp: Callable[[float], float], eps: float, orders: Sequence[float] = ORDERS
) -> tuple[float, float]:
    """1 + f_RDP*(-e^ε) = max_a (1 - f_RDP(a) - e^ε a): the best δ(ε) the RDP curve certifies.

    The objective is concave in a (f_RDP is convex), and for ε >= 0 its maximum lies left of the
    diagonal, so a golden-section search on [0, 1/2] finds it. Returns (δ, maximizing a).
    """

    def gain(a: float) -> float:
        return 1 - rdp_tradeoff(rdp, a, orders) - math.exp(eps) * a

    ratio = (math.sqrt(5) - 1) / 2
    lo, hi = 0.0, 0.5
    for _ in range(80):
        x1, x2 = hi - ratio * (hi - lo), lo + ratio * (hi - lo)
        if gain(x1) < gain(x2):
            lo = x1
        else:
            hi = x2
    a = (lo + hi) / 2
    return max(0.0, gain(a)), a


def rdp_diagonal_point(rdp: Callable[[float], float], orders: Sequence[float] = ORDERS) -> float:
    """The a with f_RDP(a) = a, where the symmetric curve f_RDP crosses the diagonal."""
    return bisect_decreasing(lambda a: rdp_tradeoff(rdp, a, orders) - a, 0.0, 0.0, 0.5, iters=60)


def gaussian_rdp(mu: float) -> Callable[[float], float]:
    """The RDP curve of μ-GDP's Gaussian pair: ε(α) = αμ²/2 (Part 1's D_α of two normals)."""
    return lambda alpha: alpha * mu * mu / 2


# --- The chapter's table -------------------------------------------------------------------------


def print_table(rng: np.random.Generator) -> None:
    print("== Part 1: f-DP and (eps, delta) ==")
    for eps, delta in ((1.0, 0.0), (1.0, 0.1)):
        vals = "  ".join(f"f({a:g})={eps_delta_tradeoff(eps, delta, a):.4f}" for a in (0.0, 0.1, 0.2, 0.5))
        print(f"[f_eps,delta eps={eps:g} delta={delta:g}] {vals}  kinks at a={(1 - delta) / (1 + math.exp(eps)):.4f}")
    p, q = randomized_response_pmfs(1.0)
    print(
        f"[randomized response eps=1] p=({p[0]:.4f}, {p[1]:.4f})  q=({q[0]:.4f}, {q[1]:.4f})  "
        f"corners={[(round(x, 4), round(y, 4)) for x, y in tradeoff_breakpoints(p, q)]}"
    )
    for a in (0.01, 0.1, 0.25):
        print(
            f"[eps=1 at a={a:g}] f_1,0={eps_delta_tradeoff(1.0, 0.0, a):.4f}  "
            f"Laplace={laplace_tradeoff(1.0, a):.4f}  G_1={gaussian_tradeoff(1.0, a):.4f}"
        )

    print("== Part 2: GDP and the Gaussian mechanism ==")
    for a in (0.01, 0.05, 0.1):
        print(f"[G_1({a:g})] {gaussian_tradeoff(1.0, a):.4f}  power={1 - gaussian_tradeoff(1.0, a):.4f}")
    print(
        f"[Gaussian mechanism] sigma=1 -> mu={gaussian_mechanism_mu(1.0):.4f}  sigma=4 -> mu={gaussian_mechanism_mu(4.0):.4f}  "
        f"pair 0.5 sensitivities apart, sigma=1 -> mu={gaussian_mechanism_mu(1.0, 0.5):.4f}"
    )
    a_hat, b_hat = simulate_projected_test(rng, [1.0], 0.05, 1_000_000)
    print(f"[simulated G_1 at a=0.05] a={a_hat:.4f}  b={b_hat:.4f}  exact b={gaussian_tradeoff(1.0, 0.05):.4f}")

    print("== Part 3: duality with (eps, delta) ==")
    for mu, eps in ((1.0, 0.0), (1.0, 1.0), (1.0, 2.0), (0.5, 1.0), (2.0, 1.0)):
        print(
            f"[mu={mu:g} eps={eps:g}] delta formula={gdp_delta(mu, eps):.6f}  "
            f"hockey-stick integral={gdp_delta_hockey_stick(mu, eps):.6f}"
        )
    grid = [i / 20_000 for i in range(20_001)]
    conj, at = conjugate_delta(lambda x: gaussian_tradeoff(1.0, x), 1.0, grid)
    print(f"[mu=1 eps=1] conjugate max_a(1 - G_1(a) - e a) on a grid of 20,001 points={conj:.6f} at a={at:.4f}")
    mu, eps = 1.0, 1.0
    t = eps / mu + mu / 2
    a_star, b_star = gdp_tangent_point(mu, eps)
    print(
        f"[walkthrough mu=1 eps=1] threshold t={t:.4f}  a*=Phi(-1.5)={a_star:.6f}  "
        f"1-b*=Phi(-0.5)={1 - b_star:.6f}  e*a*={math.e * a_star:.6f}  "
        f"delta=1-b*-e*a*={1 - b_star - math.e * a_star:.6f}"
    )
    for m in (0.5, 1.0, 2.0):
        print(f"[eps at delta=1e-5] mu={m:g}: eps={gdp_eps(m, 1e-5):.4f}")
    for m in (1.0, 2.0):
        e4, e5, e6 = (gdp_eps(m, 10.0**-j) for j in (4, 5, 6))
        print(
            f"[eps per decade of delta] mu={m:g}: delta=1e-4 -> {e4:.4f}, 1e-5 -> {e5:.4f}, 1e-6 -> {e6:.4f}  "
            f"steps {e5 - e4:.2f}, {e6 - e5:.2f}"
        )
    e_star = gdp_eps(1.0, 1e-5)
    print(
        f"[one line vs the curve] f_(4.3772,1e-5)(0.1)={eps_delta_tradeoff(e_star, 1e-5, 0.1):.4f}  G_1(0.1)={gaussian_tradeoff(1.0, 0.1):.4f}"
    )
    print(f"[delta at eps=0] mu=1: {gdp_delta(1.0, 0.0):.4f} = TV = 2Phi(1/2)-1 = {2 * normal_cdf(0.5) - 1:.4f}")
    eps_grid = [i / 100 for i in range(1001)]
    gap = max(
        abs(envelope_of_profile(1.0, a, eps_grid) - gaussian_tradeoff(1.0, a))
        for a in [i / 1000 for i in range(1, 1000)]
    )
    print(f"[envelope of 1001 lines, eps in [0,10]] max |sup f_eps,delta(eps) - G_1| on a grid = {gap:.1e}")

    print("== Part 4: composition ==")
    print(
        f"[compose] mu=(1,1)->{compose_gdp([1.0, 1.0]):.4f}  mu=(0.5,0.5,1)->{compose_gdp([0.5, 0.5, 1.0]):.4f}  "
        f"4 x mu=1 -> {compose_gdp([1.0] * 4):.4f}  100 x sigma=10 -> {compose_gdp([0.1] * 100):.4f}"
    )
    a_hat, b_hat = simulate_projected_test(rng, [1.0, 1.0], 0.05, 1_000_000)
    print(
        f"[simulated 2-D test mu=(1,1) at a=0.05] a={a_hat:.4f}  b={b_hat:.4f}  "
        f"G_sqrt2(0.05)={gaussian_tradeoff(math.sqrt(2), 0.05):.4f}  "
        f"G_1(0.05)={gaussian_tradeoff(1.0, 0.05):.4f}  G_2(0.05)={gaussian_tradeoff(2.0, 0.05):.4f}"
    )
    p, q = randomized_response_pmfs(1.0)
    pp, qq = product_pmf(p, 2), product_pmf(q, 2)
    print(
        "[RR x RR eps=1] outcomes 00,01,10,11: p="
        + ",".join(f"{x:.4f}" for x in pp)
        + "  q="
        + ",".join(f"{x:.4f}" for x in qq)
        + "  q/p="
        + ",".join(f"{y / x:.4f}" for x, y in zip(pp, qq, strict=True))
    )
    corners = tradeoff_breakpoints(pp, qq)
    print(f"[RR x RR corners] {[(round(x, 4), round(y, 4)) for x, y in corners]}  tanh(1/2)={math.tanh(0.5):.4f}")
    k = 4
    exact = gdp_eps(compose_gdp([1.0] * k), SGD_DELTA)
    naive = naive_composition_eps(1.0, k, SGD_DELTA)
    print(
        f"[4 x 1-GDP at delta=1e-5] exact via G_2: eps={exact:.4f}  per-step eps(delta/4)={gdp_eps(1.0, SGD_DELTA / k):.4f}  "
        f"basic composition of points: eps={naive:.4f}"
    )

    print("== Part 5: CLT for composition ==")
    for k in (1, 10, 100, 1000):
        eps0 = 1 / math.sqrt(k)
        mu_rr, g_rr = clt_mu_homogeneous(rr_functionals(eps0), k)
        mu_lap, g_lap = clt_mu_homogeneous(laplace_functionals(eps0), k)
        rr_d = rr_composed_delta(eps0, k, 1.0)
        lap_lo, lap_hi = laplace_composed_delta(eps0, k, 1.0)
        print(
            f"[k={k} eps0={eps0:.4f}] RR: exact delta(1)={rr_d:.4f}  Thm3.4 mu={mu_rr:.4f} gamma={g_rr:.4f} "
            f"-> delta(1)={gdp_delta(mu_rr, 1.0):.4f} | Laplace: delta(1) in [{lap_lo:.4f}, {lap_hi:.4f}]  "
            f"Thm3.4 mu={mu_lap:.4f} gamma={g_lap:.4f} -> delta(1)={gdp_delta(mu_lap, 1.0):.4f} | limit G_1: {gdp_delta(1.0, 1.0):.4f}"
        )
    for k in (10, 100, 1000):
        eps0 = 1 / math.sqrt(k)
        rr_e = eps_from_profile(lambda e: rr_composed_delta(eps0, k, e), SGD_DELTA)
        mu_rr, _ = clt_mu_homogeneous(rr_functionals(eps0), k)
        print(
            f"[k={k} eps at delta=1e-5] RR exact={rr_e:.4f}  GDP(Thm3.4 mu)={gdp_eps(mu_rr, SGD_DELTA):.4f}  "
            f"GDP(mu=1)={gdp_eps(1.0, SGD_DELTA):.4f}  basic composition k*eps0={k * eps0:.4f}"
        )
    print(f"[RR closed form] 2 sqrt(k) sinh(eps0/2) at k=100: {2 * 10 * math.sinh(0.05):.4f}")

    print("== Part 6: subsampled Gaussian CLT ==")
    print(
        f"[e^(1/sigma^2)-1] sigma=1: {math.expm1(1.0):.4f}  sqrt: {math.sqrt(math.expm1(1.0)):.4f}  "
        f"sigma=0.8: {math.expm1(1 / 0.64):.4f}  sigma=1.5: {math.expm1(1 / 2.25):.4f}"
    )
    for k in SGD_STEPS:
        mu = clt_mu_subsampled_gaussian(SGD_Q, SGD_SIGMA, k)
        print(
            f"[DP-SGD q=0.01 sigma=1 k={k}] mu={mu:.4f}  eps(1e-5)={gdp_eps(mu, SGD_DELTA):.4f}  "
            f"q sqrt(k)={SGD_Q * math.sqrt(k):.2f}  DRS Cor 5.4 mu={drs_fixed_batch_mu(SGD_Q, SGD_SIGMA, k):.4f}  "
            f"full-batch mu=sqrt(k)/sigma={math.sqrt(k) / SGD_SIGMA:.1f}"
        )
    brackets = subsampled_gaussian_eps_bracket(SGD_Q, SGD_SIGMA, SGD_STEPS, SGD_DELTA)
    for k, (lo, hi) in zip(SGD_STEPS, brackets, strict=True):
        clt = gdp_eps(clt_mu_subsampled_gaussian(SGD_Q, SGD_SIGMA, k), SGD_DELTA)
        print(
            f"[numerical accountant k={k}] eps(1e-5) in [{lo:.4f}, {hi:.4f}]  CLT eps={clt:.4f}  "
            f"CLT below the lower value: {clt < lo}  CLT/lower={clt / lo:.3f}  width={hi - lo:.4f} = k h={k * 1e-5:.4f}"
        )
    for remove in (True, False):
        mean, var = subsampled_loss_moments(SGD_Q, SGD_SIGMA, remove)
        k = 10_000
        mu = 2 * k * mean / math.sqrt(k * var)
        print(
            f"[one step, P={'mixture' if remove else 'N(0,1)'}] mean={mean:.4e}  var={var:.4e}  "
            f"heuristic q^2(e-1)/2={SGD_Q**2 * math.expm1(1.0) / 2:.4e}, q^2(e-1)={SGD_Q**2 * math.expm1(1.0):.4e}  "
            f"Thm3.4 mu at k=10000={mu:.4f} -> eps(1e-5)={gdp_eps(mu, SGD_DELTA):.4f}"
        )

    print("== Part 7: RDP as a lossy summary ==")
    rdp = gaussian_rdp(1.0)
    a_half = normal_cdf(-0.5)
    dense = [
        float(x)
        for x in np.concatenate([np.linspace(1.01, 1.1, 10), np.linspace(1.1, 64, 3000), np.linspace(64, 3000, 2000)])
    ]
    print(
        f"[mu=1 a=Phi(-1/2)={a_half:.4f}] G_1={gaussian_tradeoff(1.0, a_half):.4f}  "
        f"f_RDP on the grid of orders={rdp_tradeoff(rdp, a_half):.4f}  "
        f"f_RDP with {len(dense)} orders in [1.01, 3000]={rdp_tradeoff(rdp, a_half, dense):.4f}"
    )
    diag = rdp_diagonal_point(rdp)
    print(
        f"[mu=1 diagonal point] G_1: a={a_half:.4f} advantage={1 - 2 * a_half:.4f}  "
        f"f_RDP: a={diag:.4f} advantage={1 - 2 * diag:.4f}  delta_RDP(0)={rdp_implied_delta(rdp, 0.0)[0]:.4f}"
    )
    d_rdp, a_rdp = rdp_implied_delta(rdp, 1.0)
    print(
        f"[mu=1 eps=1] GDP delta={gdp_delta(1.0, 1.0):.4f}  delta certified by the RDP curve={d_rdp:.4f} at a={a_rdp:.4f}  "
        f"ratio={d_rdp / gdp_delta(1.0, 1.0):.2f}"
    )


# --- Figures --------------------------------------------------------------------------------------

GRID = [i / 1000 for i in range(1001)]


def plot_eps_delta_vs_gdp() -> Figure:
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(GRID, [1 - a for a in GRID], color="gray", linestyle=":", label=r"perfect privacy $1 - \mathrm{FPR}$")
    ax.plot(GRID, [eps_delta_tradeoff(1.0, 0.0, a) for a in GRID], label=r"$f_{1,0}$ (randomized response)")
    ax.plot(GRID, [eps_delta_tradeoff(1.0, 0.1, a) for a in GRID], label=r"$f_{1,0.1}$")
    ax.plot(
        GRID, [laplace_tradeoff(1.0, a) for a in GRID], linestyle="--", label=r"Laplace mechanism, $\varepsilon = 1$"
    )
    ax.plot(GRID, [gaussian_tradeoff(1.0, a) for a in GRID], color="k", label=r"$G_1$ (Gaussian, $\sigma = 1$)")
    ax.set(
        xlabel=r"type I error $\mathrm{FPR}$",
        ylabel=r"type II error $\mathrm{FNR}$",
        xlim=(0, 1),
        ylim=(0, 1),
        title="Privacy guarantees as trade-off curves",
    )
    ax.legend(loc="upper right", fontsize=9)
    return fig


def plot_duality(mu: float = 1.0) -> Figure:
    fig, (left, right) = plt.subplots(1, 2, figsize=(11, 4.5))
    epss = [i / 100 for i in range(601)]
    for m in (0.5, 1.0, 2.0):
        left.plot(epss, [max(gdp_delta(m, e), 1e-300) for e in epss], label=rf"$\mu$ = {m:g}")
    left.axhline(1e-5, color="gray", linestyle=":", label=r"$\delta = 10^{-5}$")
    left.set(
        yscale="log",
        ylim=(1e-12, 1),
        xlim=(0, 6),
        xlabel=r"$\varepsilon$",
        ylabel=r"$\delta(\varepsilon)$",
        title=r"Privacy profile of $\mu$-GDP",
    )
    left.legend(loc="lower left")
    right.plot(GRID, [gaussian_tradeoff(mu, a) for a in GRID], color="k", linewidth=2, label=rf"$G_{{{mu:g}}}$")
    for e in (0.0, 0.5, 1.0, 1.5, 2.0, 3.0):
        d = gdp_delta(mu, e)
        right.plot(
            GRID,
            [eps_delta_tradeoff(e, d, a) for a in GRID],
            linewidth=0.8,
            alpha=0.8,
            label=rf"$f_{{\varepsilon,\delta(\varepsilon)}}$, $\varepsilon$ = {e:g}",
        )
        a_t, b_t = gdp_tangent_point(mu, e)
        right.plot([a_t, b_t], [b_t, a_t], "k.", markersize=5)
    right.set(
        xlim=(0, 1),
        ylim=(0, 1),
        xlabel=r"type I error $\mathrm{FPR}$",
        ylabel=r"type II error $\mathrm{FNR}$",
        title=r"$G_1$ is the upper envelope of its $(\varepsilon,\delta)$ curves",
    )
    right.legend(loc="upper right", fontsize=8)
    return fig


def plot_composition(rng: np.random.Generator, mus: tuple[float, float] = (1.0, 1.0)) -> Figure:
    fig, (left, right) = plt.subplots(1, 2, figsize=(11, 5))
    n = 400
    y0 = rng.standard_normal((n, 2))
    y1 = rng.standard_normal((n, 2)) + np.array(mus)
    left.scatter(y0[:, 0], y0[:, 1], s=4, alpha=0.5, label=r"$P$: N(0, I)")
    left.scatter(y1[:, 0], y1[:, 1], s=4, alpha=0.5, label=r"$Q$: N(($\mu_1, \mu_2$), I)")
    mu = compose_gdp(mus)
    t = normal_isf(0.05)
    u = np.array(mus) / mu
    along = np.linspace(-4, 4, 2)
    base = t * u
    left.plot(base[0] + along * -u[1], base[1] + along * u[0], "k--", label=r"LR threshold at $\mathrm{FPR} = 0.05$")
    left.annotate("", xy=(mus[0], mus[1]), xytext=(0, 0), arrowprops={"arrowstyle": "->", "color": "k"})
    left.set(
        xlim=(-4, 5),
        ylim=(-4, 5),
        aspect="equal",
        xlabel="output of mechanism 1",
        ylabel="output of mechanism 2",
        title="Two Gaussian releases: test along the shift",
    )
    left.legend(loc="lower right", fontsize=8)
    for m, style in ((mus[0], ":"), (mu, "-")):
        right.plot(GRID, [gaussian_tradeoff(m, a) for a in GRID], linestyle=style, label=rf"$G_{{{m:.3g}}}$")
    sims = [simulate_projected_test(rng, mus, a, 20_000) for a in (0.01, 0.05, 0.1, 0.2, 0.4, 0.6, 0.8)]
    right.plot([s[0] for s in sims], [s[1] for s in sims], "kx", label="simulated 2-D tests")
    right.set(
        xlim=(0, 1),
        ylim=(0, 1),
        xlabel=r"type I error $\mathrm{FPR}$",
        ylabel=r"type II error $\mathrm{FNR}$",
        title=r"$G_1 \otimes G_1 = G_\mu$ with $\mu = 2^{1/2}$ = 1.41",
    )
    right.legend(loc="upper right")
    return fig


def plot_clt() -> Figure:
    fig, (left, right) = plt.subplots(1, 2, figsize=(11, 4.5))
    for k in (1, 4, 16, 64):
        eps0 = 1 / math.sqrt(k)
        left.plot(
            GRID,
            [rr_composed_tradeoff(eps0, k, a) for a in GRID],
            label=rf"$f_{{\varepsilon_0,0}}^{{\otimes {k}}}$, $\varepsilon_0$ = {eps0:.3g}",
        )
    left.plot(GRID, [gaussian_tradeoff(1.0, a) for a in GRID], "k--", linewidth=2, label=r"limit $G_1$")
    left.set(
        xlim=(0, 1),
        ylim=(0, 1),
        xlabel=r"type I error $\mathrm{FPR}$",
        ylabel=r"type II error $\mathrm{FNR}$",
        title=r"Composed randomized response, $k\varepsilon_0^2 = 1$",
    )
    left.legend(loc="upper right", fontsize=8)
    epss = [i / 50 for i in range(251)]
    for k in (1, 10, 100):
        eps0 = 1 / math.sqrt(k)
        right.plot(
            epss, [max(rr_composed_delta(eps0, k, e), 1e-300) for e in epss], label=f"randomized response, k = {k}"
        )
    right.plot(epss, [max(gdp_delta(1.0, e), 1e-300) for e in epss], "k--", linewidth=2, label=r"$G_1$")
    right.set(
        yscale="log",
        ylim=(1e-8, 1),
        xlim=(0, 5),
        xlabel=r"$\varepsilon$",
        ylabel=r"$\delta(\varepsilon)$",
        title="Privacy profiles approach the GDP profile",
    )
    right.legend(loc="lower left", fontsize=8)
    return fig


def plot_subsampled() -> Figure:
    fig, (left, right) = plt.subplots(1, 2, figsize=(11, 4.5))
    ks = [int(round(10 ** (1 + 3 * i / 150))) for i in range(151)]
    for sigma in (0.8, 1.0, 1.5):
        mus = [clt_mu_subsampled_gaussian(SGD_Q, sigma, k) for k in ks]
        left.plot(ks, mus, label=rf"$\sigma$ = {sigma:g}")
        right.plot(ks, [gdp_eps(m, SGD_DELTA) for m in mus], label=rf"$\sigma$ = {sigma:g}")
    for k in SGD_STEPS:
        mu = clt_mu_subsampled_gaussian(SGD_Q, SGD_SIGMA, k)
        left.plot([k], [mu], "ko")
        right.plot([k], [gdp_eps(mu, SGD_DELTA)], "ko")
    left.set(
        xscale="log", yscale="log", xlabel="steps $T$", ylabel=r"CLT $\mu$", title=r"CLT $\mu$ for DP-SGD, $q = 0.01$"
    )
    right.set(
        xscale="log",
        xlabel="steps $T$",
        ylabel=r"$\varepsilon$ at $\delta = 10^{-5}$",
        title="Reported ε via GDP duality (an approximation)",
    )
    left.legend()
    right.legend()
    return fig


def plot_rdp_lossy(mu: float = 1.0, eps: float = 1.0) -> Figure:
    fig, ax = plt.subplots(figsize=(6, 5))
    rdp = gaussian_rdp(mu)
    grid = [i / 200 for i in range(201)]
    ax.plot(grid, [1 - a for a in grid], color="gray", linestyle=":", linewidth=0.8)
    ax.plot(grid, [gaussian_tradeoff(mu, a) for a in grid], "k", linewidth=2, label=rf"$G_{{{mu:g}}}$ (exact)")
    ax.plot(
        grid,
        [rdp_tradeoff(rdp, a) for a in grid],
        "--",
        color="C3",
        linewidth=2,
        label=r"$f_{\mathrm{RDP}}$ from $\varepsilon(\alpha) = \alpha\mu^2/2$",
    )
    d_gdp = gdp_delta(mu, eps)
    a_gdp, b_gdp = gdp_tangent_point(mu, eps)
    d_rdp, a_rdp = rdp_implied_delta(rdp, eps)
    b_rdp = rdp_tradeoff(rdp, a_rdp)
    for d, (a_t, b_t), color, name in (
        (d_gdp, (a_gdp, b_gdp), "k", r"$G_1$"),
        (d_rdp, (a_rdp, b_rdp), "C3", r"$f_{\mathrm{RDP}}$"),
    ):
        ax.plot(
            grid,
            [eps_delta_tradeoff(eps, d, a) for a in grid],
            color=color,
            linewidth=0.8,
            alpha=0.8,
            label=rf"$f_{{{eps:g},\,{d:.4f}}}$, certified by {name}",
        )
        ax.plot([a_t, b_t], [b_t, a_t], ".", color=color, markersize=8)
    ax.set(
        xlim=(0, 1),
        ylim=(0, 1),
        xlabel=r"type I error $\mathrm{FPR}$",
        ylabel=r"type II error $\mathrm{FNR}$",
        title=rf"What the RDP curve keeps, and its $\varepsilon = {eps:g}$ line",
    )
    ax.legend(loc="upper right", fontsize=8)
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
        "gaussian_dp_eps_delta.svg": plot_eps_delta_vs_gdp(),
        "gaussian_dp_duality.svg": plot_duality(),
        "gaussian_dp_composition.svg": plot_composition(rng),
        "gaussian_dp_clt.svg": plot_clt(),
        "gaussian_dp_subsampled.svg": plot_subsampled(),
        "gaussian_dp_rdp_lossy.svg": plot_rdp_lossy(),
    }
    for name, fig in figures.items():
        fig.tight_layout()
        fig.savefig(out / name, metadata={"Date": None})  # no timestamp, so output is reproducible
        plt.close(fig)
        print(f"saved {out / name}")


if __name__ == "__main__":
    main()
