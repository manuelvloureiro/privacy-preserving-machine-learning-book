"""Check the derivations of the appendix on subsampling, shuffling and iteration, and draw its figures.

    uv run python code/appendix-derivations/apx_amplification.py
    uv run python code/appendix-derivations/apx_amplification.py --seed 1 --out /tmp/figures

Prints every number quoted in book-full/appendix-derivations/12-amplification.md (they match exactly
with the default seed), then saves the appendix's figures.

Conventions. q is the sampling rate and σ the noise multiplier, with sensitivity 1. P0 = N(0, σ²),
P1 = N(1, σ²) and Pq = (1 - q)P0 + qP1. The hockey-stick divergence is
E_γ(P‖Q) = max over events A of P(A) - γQ(A), for every real γ. A binary symmetric distribution
π_θ on the two outputs {+, -} has π_θ(±) = e^{±θ} / (2 cosh θ). In the shuffling section ε0 is the
local privacy level, p = e^ε0 / (e^ε0 + 1), and n is the number of users. In the iteration sections
L is the Lipschitz constant of the loss and σ the noise standard deviation. Logarithms are natural.
"""

import argparse
import itertools
import math
import sys
from collections.abc import Sequence
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # for code/shared
from shared.figures import configure
from shared.stats import normal_cdf

configure("apx_amplification")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from numpy.typing import NDArray

# Figures are build output: _build/figures/<part>/, which the build copies into each book.
FIGURES = Path(__file__).resolve().parents[2] / "_build" / "figures" / Path(__file__).parent.name

FloatArray = NDArray[np.float64]


def log_sum_exp(v: FloatArray) -> float:
    m = float(np.max(v))
    return m + math.log(float(np.sum(np.exp(v - m))))


# --- 1. The remove direction dominates in Rényi divergence ------------------------------------


def lncosh(x: FloatArray) -> FloatArray:
    """ln cosh x without overflow."""
    return np.logaddexp(x, -x) - math.log(2.0)


def mixture_parameter(a: FloatArray, q: float) -> FloatArray:
    """θ of (1 - q)π_a + qπ_{-a}: the mixture is again binary symmetric, with tanh θ = (1 - 2q) tanh a."""
    return np.arctanh((1 - 2 * q) * np.tanh(a))


def pair_log_moments(a: FloatArray, q: float, alpha: float) -> tuple[FloatArray, FloatArray]:
    """ln A' and ln B' for the binary pair P0 = π_a, Pq = (1 - q)π_a + qπ_{-a}.

    A' = Σ Pq^α P0^{1-α} (remove direction) and B' = Σ P0^α Pq^{1-α} (add direction). With
    Pq = π_b, both are ratios of cosh terms.
    """
    b = mixture_parameter(a, q)
    log_a = lncosh(alpha * b + (1 - alpha) * a) - alpha * lncosh(b) - (1 - alpha) * lncosh(a)
    log_b = lncosh(alpha * a + (1 - alpha) * b) - alpha * lncosh(a) - (1 - alpha) * lncosh(b)
    return log_a, log_b


def pair_log_moments_direct(a: float, q: float, alpha: float) -> tuple[float, float]:
    """The same two moments, summed over the two outputs."""
    p0 = np.array([math.exp(a), math.exp(-a)]) / (2 * math.cosh(a))
    pq = (1 - q) * p0 + q * p0[::-1]
    return (
        math.log(float(np.sum(pq**alpha * p0 ** (1 - alpha)))),
        math.log(float(np.sum(p0**alpha * pq ** (1 - alpha)))),
    )


def tanh_average(center: float, half_width: float) -> float:
    """Average of tanh over [center - h, center + h]: (ln cosh(c + h) - ln cosh(c - h)) / (2h)."""
    hi = lncosh(np.array([center + half_width]))[0]
    lo = lncosh(np.array([center - half_width]))[0]
    return float(hi - lo) / (2 * half_width)


def direction_grid_check(qs: Sequence[float], alphas: FloatArray, a_values: FloatArray) -> tuple[float, int, int]:
    """Smallest ln A' - ln B' over the grid, the number of grid points and the number below -1e-12."""
    smallest, points, violations = math.inf, 0, 0
    for q in qs:
        for alpha in alphas:
            log_a, log_b = pair_log_moments(a_values, q, float(alpha))
            gap = log_a - log_b
            smallest = min(smallest, float(np.min(gap)))
            points += gap.size
            violations += int(np.sum(gap < -1e-12))
    return smallest, points, violations


def gaussian_log_moment(q: float, sigma: float, alpha: float, power: float) -> float:
    """ln E_{P0}[(Pq/P0)^power] by integration over the output y on a fine grid."""
    h = sigma / 200
    y = np.arange(-12 * sigma, alpha + 12 * sigma, h)
    log_p0 = -(y**2) / (2 * sigma**2) - 0.5 * math.log(2 * math.pi * sigma**2)
    log_ratio = np.logaddexp(math.log(1 - q), math.log(q) + (2 * y - 1) / (2 * sigma**2))
    return log_sum_exp(log_p0 + power * log_ratio) + math.log(h)


def gaussian_renyi_both(q: float, sigma: float, alpha: float) -> tuple[float, float]:
    """D_α(Pq‖P0) and D_α(P0‖Pq) by integration over the output."""
    return (
        gaussian_log_moment(q, sigma, alpha, alpha) / (alpha - 1),
        gaussian_log_moment(q, sigma, alpha, 1 - alpha) / (alpha - 1),
    )


def gaussian_renyi_by_pairs(q: float, sigma: float, alpha: float) -> tuple[float, float]:
    """The same two divergences, as an average over reflected pairs {1/2 - z, 1/2 + z}.

    The pair at distance z is binary symmetric with a = z / (2σ²), and z has the density
    P0(1/2 - z) + P0(1/2 + z) under P0, P1 and Pq alike.
    """
    h = sigma / 200
    z: FloatArray = (np.arange(int((alpha + 12 * sigma) / h), dtype=np.float64) + 0.5) * h
    log_m = np.logaddexp(-((0.5 - z) ** 2) / (2 * sigma**2), -((0.5 + z) ** 2) / (2 * sigma**2))
    log_m = log_m - 0.5 * math.log(2 * math.pi * sigma**2)
    log_a, log_b = pair_log_moments(z / (2 * sigma**2), q, alpha)
    return (
        (log_sum_exp(log_m + log_a) + math.log(h)) / (alpha - 1),
        (log_sum_exp(log_m + log_b) + math.log(h)) / (alpha - 1),
    )


# --- 2. Dominating pairs for Poisson subsampling ----------------------------------------------


def gaussian_hockey(gamma: float, mu: float) -> float:
    """E_γ(N(μ, 1)‖N(0, 1)), the same in both orders; 1 - γ when γ <= 0."""
    if gamma <= 0:
        return 1 - gamma
    e = math.log(gamma)
    return normal_cdf(mu / 2 - e / mu) - gamma * normal_cdf(-mu / 2 - e / mu)


def remove_hockey_direct(eps: float, q: float, sigma: float) -> float:
    """E_{e^ε}(Pq‖P0), from the output threshold where the remove loss exceeds ε."""
    gamma = math.exp(eps)
    if eps <= math.log(1 - q):
        return 1 - gamma
    y = sigma**2 * math.log((gamma - 1 + q) / q) + 0.5
    tail_p = (1 - q) * normal_cdf(-y / sigma) + q * normal_cdf((1 - y) / sigma)
    return tail_p - gamma * normal_cdf(-y / sigma)


def add_hockey_direct(eps: float, q: float, sigma: float) -> float:
    """E_{e^ε}(P0‖Pq): the outputs where the remove loss is below -ε."""
    gamma = math.exp(eps)
    if -eps <= math.log(1 - q):
        return 0.0
    y = sigma**2 * math.log((math.exp(-eps) - 1 + q) / q) + 0.5
    mixture = (1 - q) * normal_cdf(y / sigma) + q * normal_cdf((y - 1) / sigma)
    return normal_cdf(y / sigma) - gamma * mixture


def remove_gamma_map(gamma: float, q: float) -> float:
    """γ' = 1 + (γ - 1)/q, the level at which the base pair is read in the remove identity."""
    return 1 + (gamma - 1) / q


def remove_hockey_identity(eps: float, q: float, sigma: float) -> float:
    """q E_{γ'}(P1‖P0) with γ' = 1 + (e^ε - 1)/q."""
    return q * gaussian_hockey(remove_gamma_map(math.exp(eps), q), 1 / sigma)


def add_hockey_identity(eps: float, q: float, sigma: float) -> float:
    """c E_{γq/c}(P0‖P1) with c = 1 - γ(1 - q), or 0 when c <= 0."""
    gamma = math.exp(eps)
    c = 1 - gamma * (1 - q)
    if c <= 0:
        return 0.0
    return c * gaussian_hockey(gamma * q / c, 1 / sigma)


def subsampled_sum_hockey(
    gradients: Sequence[float], extra: float, q: float, sigma: float, gamma: float, remove: bool
) -> float:
    """E_γ between a Poisson-subsampled noisy sum on D ∪ {x} and on D, in 1-D, by integration.

    remove=True compares M(D') against M(D), remove=False the reverse.
    """
    y = np.linspace(-14.0, 14.0, 56_001)
    h = float(y[1] - y[0])
    without = np.zeros_like(y)
    for chosen in itertools.product((0, 1), repeat=len(gradients)):
        weight = math.prod(q if c else 1 - q for c in chosen)
        centre = sum(g for g, c in zip(gradients, chosen, strict=True) if c)
        without += weight * np.exp(-((y - centre) ** 2) / (2 * sigma**2))
    with_x = np.zeros_like(y)
    for chosen in itertools.product((0, 1), repeat=len(gradients)):
        weight = math.prod(q if c else 1 - q for c in chosen)
        centre = sum(g for g, c in zip(gradients, chosen, strict=True) if c)
        with_x += weight * (
            (1 - q) * np.exp(-((y - centre) ** 2) / (2 * sigma**2))
            + q * np.exp(-((y - centre - extra) ** 2) / (2 * sigma**2))
        )
    norm = 1 / math.sqrt(2 * math.pi * sigma**2)
    p, other = (with_x, without) if remove else (without, with_x)
    return float(np.sum(np.maximum(p - gamma * other, 0.0))) * norm * h


# --- 3. The clone reduction for shuffling -----------------------------------------------------


def kary_rr(eps0: float, k: int) -> FloatArray:
    """k-ary randomized response: row x is R(x), the truth with weight e^ε0, every other value 1."""
    m = np.ones((k, k))
    np.fill_diagonal(m, math.exp(eps0))
    return m / m.sum(axis=1, keepdims=True)


def rr_components(r0: FloatArray, r1: FloatArray, eps0: float) -> tuple[FloatArray, FloatArray]:
    """Q0 = (e^ε0 R(x0) - R(x1)) / (e^ε0 - 1) and Q1 the same with x0 and x1 swapped."""
    e = math.exp(eps0)
    return (e * r0 - r1) / (e - 1), (e * r1 - r0) / (e - 1)


def clone_split(rx: FloatArray, r0: FloatArray, r1: FloatArray, eps0: float) -> tuple[FloatArray, FloatArray]:
    """R(x) = clone part + leftover part, with clone part e^{-ε0}(R(x0) + R(x1))/2."""
    clone = math.exp(-eps0) * (r0 + r1) / 2
    return clone, rx - clone


def random_randomizer(rng: np.random.Generator, eps0: float, inputs: int, outputs: int) -> FloatArray:
    """A random randomizer that is ε0-DP: each column's entries lie within a factor e^{ε0/2}
    before normalizing, and normalizing changes each row's scale by at most e^{ε0/2}."""
    m = np.exp(rng.uniform(0.0, eps0 / 2, size=(inputs, outputs)))
    return m / m.sum(axis=1, keepdims=True)


def decomposition_errors(rng: np.random.Generator, eps0: float, trials: int) -> tuple[float, float, float]:
    """Over random ε0-DP randomizers: the most negative entry of Q0, Q1 and of any leftover, the
    largest error of R(x0) = pQ0 + (1 - p)Q1, and the largest error of (R(x0) + R(x1))/2 = (Q0 + Q1)/2."""
    p = math.exp(eps0) / (math.exp(eps0) + 1)
    most_negative, mix_err, mean_err = 0.0, 0.0, 0.0
    for _ in range(trials):
        r = random_randomizer(rng, eps0, 4, 5)
        q0, q1 = rr_components(r[0], r[1], eps0)
        lowest = min(float(q0.min()), float(q1.min()))
        for x in range(4):
            lowest = min(lowest, float(clone_split(r[x], r[0], r[1], eps0)[1].min()))
        most_negative = min(most_negative, lowest)
        mix_err = max(mix_err, float(np.max(np.abs(p * q0 + (1 - p) * q1 - r[0]))))
        mix_err = max(mix_err, float(np.max(np.abs((1 - p) * q0 + p * q1 - r[1]))))
        mean_err = max(mean_err, float(np.max(np.abs((r[0] + r[1]) / 2 - (q0 + q1) / 2))))
    return most_negative, mix_err, mean_err


Histogram = tuple[int, ...]


def add_draw(dist: dict[Histogram, float], probs: FloatArray) -> dict[Histogram, float]:
    """Law of the histogram after one more independent report drawn from probs."""
    out: dict[Histogram, float] = {}
    for hist, w in dist.items():
        for y, py in enumerate(probs):
            new = tuple(c + (1 if i == y else 0) for i, c in enumerate(hist))
            out[new] = out.get(new, 0.0) + w * float(py)
    return out


def shuffled_histogram(rows: Sequence[FloatArray]) -> dict[Histogram, float]:
    """Exact law of the shuffled reports (equivalently their histogram) when user i draws from rows[i]."""
    k = len(rows[0])
    dist: dict[Histogram, float] = {tuple([0] * k): 1.0}
    for row in rows:
        dist = add_draw(dist, row)
    return dist


def clone_pair_pmf(n: int, eps0: float) -> dict[tuple[int, int], tuple[float, float]]:
    """P̃ and Q̃ of the clone pair as exact laws of (first count, second count)."""
    pc = math.exp(-eps0)
    p = math.exp(eps0) / (math.exp(eps0) + 1)
    out: dict[tuple[int, int], tuple[float, float]] = {}
    for c in range(n):
        wc = math.comb(n - 1, c) * pc**c * (1 - pc) ** (n - 1 - c)
        for a in range(c + 1):
            wa = wc * math.comb(c, a) / 2**c
            for delta, pd in ((1, p), (0, 1 - p)):
                key_p = (a + delta, c - a + 1 - delta)
                key_q = (a + 1 - delta, c - a + delta)
                old_p = out.get(key_p, (0.0, 0.0))
                out[key_p] = (old_p[0] + wa * pd, old_p[1])
                old_q = out.get(key_q, (0.0, 0.0))
                out[key_q] = (old_q[0], old_q[1] + wa * pd)
    return out


def reduction_check(eps0: float, others: Sequence[int], k: int) -> float:
    """Largest gap between the exact shuffled law and the post-processed clone pair, on D and D'.

    You hold value 0 on D and 1 on D'; the other users hold the values in `others`. The map draws
    n0 reports from Q0 and n1 from Q1, a leftover report for each of the other users not chosen
    as a clone (a uniformly random set of size n0 + n1 - 1), and forgets the order.
    """
    r = kary_rr(eps0, k)
    q0, q1 = rr_components(r[0], r[1], eps0)
    leftovers = [clone_split(r[x], r[0], r[1], eps0)[1] / (1 - math.exp(-eps0)) for x in others]
    n = len(others) + 1
    worst = 0.0
    pair = clone_pair_pmf(n, eps0)
    for side, you in ((0, 0), (1, 1)):
        exact = shuffled_histogram([r[you]] + [r[x] for x in others])
        built: dict[Histogram, float] = {}
        for (n0, n1), probs in pair.items():
            w = probs[side]
            if w == 0:
                continue
            clones = n0 + n1 - 1
            subsets = list(itertools.combinations(range(len(others)), clones))
            for subset in subsets:
                rows = [q0] * n0 + [q1] * n1 + [leftovers[i] for i in range(len(others)) if i not in subset]
                for hist, ph in shuffled_histogram(rows).items():
                    built[hist] = built.get(hist, 0.0) + w * ph / len(subsets)
        keys = set(exact) | set(built)
        worst = max(worst, max(abs(exact.get(h, 0.0) - built.get(h, 0.0)) for h in keys))
    return worst


def log_factorials(n: int) -> FloatArray:
    return np.array([math.lgamma(k + 1) for k in range(n + 1)])


def clone_pair_arrays(n: int, eps0: float, width: float = 12.0) -> tuple[FloatArray, FloatArray]:
    """The probabilities of P̃ and Q̃ on every point (a, c + 1 - a) that has non-negligible mass.

    Given C = c, the first count a determines the second, so each point is one (c, a).
    """
    pc = math.exp(-eps0)
    p = math.exp(eps0) / (math.exp(eps0) + 1)
    mu = (n - 1) * pc
    sd = math.sqrt(mu * (1 - pc))
    lf = log_factorials(n)
    cs = np.arange(max(0, int(mu - width * sd - 2)), min(n - 1, int(mu + width * sd + 2)) + 1)
    log_c = lf[n - 1] - lf[cs] - lf[n - 1 - cs] + cs * math.log(pc) + (n - 1 - cs) * math.log1p(-pc)
    half = int(width * math.sqrt(float(cs.max())) / 2 + 3)
    big_c = cs[:, None]
    a = big_c // 2 + np.arange(-half, half + 1)[None, :]

    def binom_half(k: NDArray[np.int64]) -> FloatArray:
        kk = np.clip(k, 0, big_c)
        logs = lf[big_c] - lf[kk] - lf[big_c - kk] - big_c * math.log(2)
        return np.where((k >= 0) & (k <= big_c), np.exp(logs), 0.0)

    b_prev, b_here = binom_half(a - 1), binom_half(a)
    w = np.exp(log_c)[:, None]
    pt = (w * (p * b_prev + (1 - p) * b_here)).ravel()
    qt = (w * ((1 - p) * b_prev + p * b_here)).ravel()
    keep = np.maximum(pt, qt) > 1e-30
    return pt[keep], qt[keep]


def pair_delta(pt: FloatArray, qt: FloatArray, eps: float) -> float:
    """δ(ε) of a discrete pair: the larger of the two hockey-stick sums."""
    g = math.exp(eps)
    return max(float(np.sum(np.maximum(pt - g * qt, 0.0))), float(np.sum(np.maximum(qt - g * pt, 0.0))))


def pair_epsilon(pt: FloatArray, qt: FloatArray, delta: float, hi: float) -> float:
    """Smallest ε with δ(ε) <= delta, by bisection."""
    lo = 0.0
    for _ in range(40):
        mid = (lo + hi) / 2
        if pair_delta(pt, qt, mid) > delta:
            lo = mid
        else:
            hi = mid
    return hi


def clone_loss(a: FloatArray, c: int, eps0: float) -> FloatArray:
    """Privacy loss ln P̃/Q̃ at first count a given C = c: ln((pa + (1-p)(c-a+1)) / ((1-p)a + p(c-a+1)))."""
    p = math.exp(eps0) / (math.exp(eps0) + 1)
    return np.log((p * a + (1 - p) * (c - a + 1)) / ((1 - p) * a + p * (c - a + 1)))


def clone_closed_form(n: int, eps0: float, delta: float) -> tuple[float, float, float, float]:
    """This appendix's closed form: (c_min, ε from c_min, simplified ε from μ alone, its
    ln(1 + 4x̄) form, which needs x̄ <= 1/2 as well).

    c_min = μ - sqrt(2μ ln(2/δ)) with μ = (n - 1)e^{-ε0}; x = b(sqrt(2 c ln(2/δ)) + 1)/(c + 1) at
    c = c_min, ε = ln((1 + x)/(1 - x)); the simplified x̄ = 2b(sqrt(ln(2/δ)/μ) + 1/μ), valid when
    μ >= 8 ln(2/δ). NaN where a condition fails.
    """
    mu = (n - 1) * math.exp(-eps0)
    big_l = math.log(2 / delta)
    b = (math.exp(eps0) - 1) / (math.exp(eps0) + 1)
    c_min = mu - math.sqrt(2 * mu * big_l)
    x = b * (math.sqrt(2 * c_min * big_l) + 1) / (c_min + 1) if c_min > 0 else math.inf
    xbar = 2 * b * (math.sqrt(big_l / mu) + 1 / mu) if mu >= 8 * big_l else math.inf
    eps = math.log((1 + x) / (1 - x)) if x < 1 else math.nan
    eps_bar = math.log((1 + xbar) / (1 - xbar)) if xbar < 1 else math.nan
    eps_lin = math.log(1 + 4 * xbar) if xbar <= 0.5 else math.nan
    return c_min, eps, eps_bar, eps_lin


def fmt_closed_form(n: int, eps0: float, delta: float) -> float:
    """Feldman, McMillan & Talwar's closed form, NaN where its condition fails."""
    if eps0 > math.log(n / (16 * math.log(2 / delta))):
        return math.nan
    b = (math.exp(eps0) - 1) / (math.exp(eps0) + 1)
    e = math.exp(eps0)
    return math.log(1 + b * (8 * math.sqrt(e * math.log(4 / delta)) / math.sqrt(n) + 8 * e / n))


# --- 4. Shift reduction ------------------------------------------------------------------------


def gaussian_cost(alpha: float, a: float, sigma: float) -> float:
    """R_α(a) = α a² / (2σ²)."""
    return alpha * a * a / (2 * sigma**2)


def half_mixture_renyi2(a: float, sigma: float) -> float:
    """D_2(½N(s - a, σ²) + ½N(s, σ²) ‖ N(s, σ²)) = ln(1 + ¼(e^{a²/σ²} - 1))."""
    return math.log(1 + 0.25 * (math.exp(a * a / sigma**2) - 1))


def half_mixture_renyi_numeric(a: float, sigma: float, alpha: float) -> float:
    """The same divergence at any order, by integration."""
    y = np.linspace(-12 * sigma - a, 12 * sigma + a, 40_001)
    h = float(y[1] - y[0])
    log_q = -(y**2) / (2 * sigma**2) - 0.5 * math.log(2 * math.pi * sigma**2)
    log_p = np.logaddexp(-((y + a) ** 2) / (2 * sigma**2), -(y**2) / (2 * sigma**2)) + math.log(0.5)
    log_p = log_p - 0.5 * math.log(2 * math.pi * sigma**2)
    return (log_sum_exp(alpha * log_p + (1 - alpha) * log_q) + math.log(h)) / (alpha - 1)


def split_displacement(w: FloatArray, a: float) -> tuple[FloatArray, FloatArray]:
    """Split each row w of displacements into W1 (carried, length (|w| - a)+) and W2 (paid, length <= a)."""
    norms = np.linalg.norm(w, axis=1, keepdims=True)
    scale = np.where(norms > a, a / np.maximum(norms, 1e-300), 1.0)
    w2 = w * scale
    return w - w2, w2


def split_check(rng: np.random.Generator, z: float, a: float, samples: int) -> tuple[float, float]:
    """Largest |W1| and |W2| over random displacements of length at most z + a in three dimensions."""
    direction = rng.normal(size=(samples, 3))
    direction /= np.linalg.norm(direction, axis=1, keepdims=True)
    w = direction * rng.uniform(0, z + a, size=(samples, 1))
    w1, w2 = split_displacement(w, a)
    return float(np.max(np.linalg.norm(w1, axis=1))), float(np.max(np.linalg.norm(w2, axis=1)))


# --- 5. Random stopping time --------------------------------------------------------------------


def fixed_stop_rdp(t: int, n: int, alpha: float, lip: float, sigma: float) -> float:
    """Theorem 23: record t of a one-pass run released after step n."""
    return 2 * alpha * lip**2 / (sigma**2 * (n - t + 1))


def random_stop_rdp(t: int, n: int, alpha: float, lip: float, sigma: float) -> float:
    """(1/(α-1)) ln of the average of e^{(α-1)D_T} over T uniform on {1..n}, D_T from Theorem 23."""
    k = np.arange(1, n - t + 2)
    x = (alpha - 1) * 2 * alpha * lip**2 / (sigma**2 * k)
    return math.log(((t - 1) + float(np.sum(np.exp(x)))) / n) / (alpha - 1)


def harmonic(m: int) -> float:
    return float(np.sum(1.0 / np.arange(1, m + 1)))


def random_stop_linearized(n: int, alpha: float, lip: float, sigma: float, slope: float) -> float:
    """The bound after e^x <= 1 + slope·x on [0, 1] and ln(1 + u) <= u: slope·2αL²H_n/(nσ²)."""
    return slope * 2 * alpha * lip**2 * harmonic(n) / (n * sigma**2)


def random_stop_stated(n: int, alpha: float, lip: float, sigma: float) -> float:
    """The chapter's statement of Theorem 26: 4αL² ln n / (nσ²)."""
    return 4 * alpha * lip**2 * math.log(n) / (n * sigma**2)


def harmonic_threshold(limit: int) -> int:
    """Smallest n0 such that 2(e - 1)H_n <= 4 ln n for every n from n0 to limit."""
    h = np.cumsum(1.0 / np.arange(1, limit + 1))
    n = np.arange(1, limit + 1)
    ok = 2 * (math.e - 1) * h <= 4 * np.log(n)
    bad = np.nonzero(~ok)[0]
    return int(bad[-1]) + 2


# --- The appendix's table -----------------------------------------------------------------------


def print_table(rng: np.random.Generator) -> None:
    # 1. Direction.
    a, q, alpha = 0.5, 0.1, 2.0
    p0 = np.array([math.exp(a), math.exp(-a)]) / (2 * math.cosh(a))
    pq = (1 - q) * p0 + q * p0[::-1]
    b = float(mixture_parameter(np.array([a]), q)[0])
    la, lb = pair_log_moments_direct(a, q, alpha)
    d = a - b
    centre = (a + b) / 2
    narrow = tanh_average(centre, d / 2)
    wide = tanh_average(centre, (2 * alpha - 1) * d / 2)
    print(
        f"[direction pair a={a} q={q} alpha={alpha:g}] P0=({p0[0]:.4f}, {p0[1]:.4f})  "
        f"Pq=({pq[0]:.4f}, {pq[1]:.4f})  b={b:.4f}  ln A'={la:.6f}  ln B'={lb:.6f}"
    )
    print(
        f"[direction tanh] centre={centre:.4f}  narrow average={narrow:.6f}  wide average={wide:.6f}  "
        f"(2a-1)d(narrow-wide)={(2 * alpha - 1) * d * (narrow - wide):.6f}"
    )
    qs = [0.001, 0.01, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9]
    alphas = np.concatenate([np.linspace(1.01, 3.0, 40), np.linspace(3.0, 256.0, 254)])
    a_values = np.linspace(0.0, 0.5 * math.log(1e6), 2001)
    smallest, points, violations = direction_grid_check(qs, alphas, a_values)
    print(f"[direction grid] points={points}  violations={violations}  smallest gap={smallest:.1e}")
    for order in (2.0, 8.0, 32.0):
        rem, add = gaussian_renyi_both(0.01, 1.0, order)
        rem_p, add_p = gaussian_renyi_by_pairs(0.01, 1.0, order)
        print(
            f"[direction Gaussian q=0.01 sigma=1 alpha={order:g}] remove={rem:.4e} add={add:.4e}  "
            f"by pairs: remove={rem_p:.4e} add={add_p:.4e}"
        )

    # 2. Poisson dominating pairs.
    q, sigma = 0.01, 1.0
    for eps in (0.0, 0.5, 1.0):
        print(
            f"[poisson eps={eps:g}] remove direct={remove_hockey_direct(eps, q, sigma):.6e}  "
            f"identity={remove_hockey_identity(eps, q, sigma):.6e}  "
            f"add direct={add_hockey_direct(eps, q, sigma):.6e}  identity={add_hockey_identity(eps, q, sigma):.6e}"
        )
    grid = np.linspace(-0.5, 3.0, 351)
    rel = max(
        abs(remove_hockey_direct(float(e), q, sigma) / remove_hockey_identity(float(e), q, sigma) - 1) for e in grid
    )
    rel_add = max(
        abs(add_hockey_direct(float(e), q, sigma) - add_hockey_identity(float(e), q, sigma))
        / max(add_hockey_identity(float(e), q, sigma), 1e-300)
        for e in grid
        if add_hockey_identity(float(e), q, sigma) > 0
    )
    print(
        f"[poisson identities] largest relative gap remove={rel:.1e}  add={rel_add:.1e}  "
        f"add delta vanishes from eps={-math.log(1 - q):.4f}"
    )
    eps_base = 1.0
    eps_sub = math.log(1 + q * (math.exp(eps_base) - 1))
    print(
        f"[poisson bonus] eps'={eps_sub:.6f}  gamma'={remove_gamma_map(math.exp(eps_sub), q):.6f}  "
        f"delta(eps')={remove_hockey_direct(eps_sub, q, sigma):.6e}  q*delta_base(1)={q * gaussian_hockey(math.e, 1.0):.6e}"
    )
    worst_ratio, worst_excess = 0.0, -math.inf
    for gamma in (0.8, 1.0, 1.2, 2.0, 5.0):
        for remove in (True, False):
            real = subsampled_sum_hockey([0.3, -0.5], 0.8, 0.3, 1.0, gamma, remove)
            eps_g = math.log(gamma)
            pair = remove_hockey_direct(eps_g, 0.3, 1.0) if remove else add_hockey_direct(eps_g, 0.3, 1.0)
            worst_excess = max(worst_excess, real - pair)
            if pair > 1e-12:
                worst_ratio = max(worst_ratio, real / pair)
    print(
        f"[poisson three records q=0.3] largest ratio real/pair={worst_ratio:.4f}  "
        f"largest excess real-pair={worst_excess:.1e}"
    )

    # 3. Clones.
    eps0 = math.log(4)
    r = kary_rr(eps0, 3)
    q0, q1 = rr_components(r[0], r[1], eps0)
    clone, left = clone_split(r[2], r[0], r[1], eps0)
    print(f"[clones 3-ary RR eps0=ln4] R(0)={np.round(r[0], 4)}  Q0={np.round(q0, 4)}  Q1={np.round(q1, 4)}")
    print(
        f"[clones split x=2] clone={np.round(clone, 4)}  leftover={np.round(left, 4)}  "
        f"LO={np.round(left / (1 - math.exp(-eps0)), 4)}"
    )
    neg, mix_err, mean_err = decomposition_errors(rng, 2.0, 2000)
    print(
        f"[clones random randomizers] most negative={neg:.1e}  mixture error={mix_err:.1e}  mean error={mean_err:.1e}"
    )
    print(f"[clones exact reduction n=4] largest gap={reduction_check(eps0, [2, 2, 0], 3):.1e}")
    for n, e0 in (
        (1_000, 1.0),
        (1_000, 2.0),
        (10_000, 1.0),
        (10_000, 2.0),
        (10_000, 4.0),
        (100_000, 2.0),
        (100_000, 4.0),
    ):
        pt, qt = clone_pair_arrays(n, e0)
        numeric = pair_epsilon(pt, qt, 1e-6, e0)
        c_min, mine, simple, linear = clone_closed_form(n, e0, 1e-6)
        print(
            f"[clones n={n} eps0={e0:g}] numeric={numeric:.4f}  this appendix={mine:.4f}  "
            f"simplified={simple:.4f}  ln(1+4x)={linear:.4f}  FMT={fmt_closed_form(n, e0, 1e-6):.4f}  c_min={c_min:.1f}  mu={(n - 1) * math.exp(-e0):.1f}  "
            f"delta at this appendix's eps={pair_delta(pt, qt, mine):.2e}  "
            f"ratios to numeric: this appendix={mine / numeric:.2f}  FMT={fmt_closed_form(n, e0, 1e-6) / numeric:.2f}"
        )

    # 4. Shift reduction.
    for a_pay in (0.25, 0.5, 1.0):
        print(
            f"[shift s=1 sigma=1 a={a_pay:g}] exact D2={half_mixture_renyi2(a_pay, 1.0):.4f}  "
            f"numeric={half_mixture_renyi_numeric(a_pay, 1.0, 2.0):.4f}  bound={gaussian_cost(2.0, a_pay, 1.0):.4f}"
        )
    w1max, w2max = split_check(rng, 0.6, 0.4, 100_000)
    print(f"[shift split z=0.6 a=0.4] largest |W1|={w1max:.4f}  largest |W2|={w2max:.4f}")

    # 5. Random stopping.
    n, alpha, lip, sigma = 1000, 2.0, 1.0, 2.0
    print(
        f"[stopping n={n} alpha={alpha:g} L={lip:g} sigma={sigma:g}] condition x1={2 * alpha * (alpha - 1) * lip**2 / sigma**2:.2f}  "
        f"fixed t=1: {fixed_stop_rdp(1, n, alpha, lip, sigma):.4f}  fixed t=n: {fixed_stop_rdp(n, n, alpha, lip, sigma):.4f}"
    )
    print(
        f"[stopping bounds] exact average t=1: {random_stop_rdp(1, n, alpha, lip, sigma):.6f}  "
        f"t=n: {random_stop_rdp(n, n, alpha, lip, sigma):.6f}  "
        f"(e-1) slope: {random_stop_linearized(n, alpha, lip, sigma, math.e - 1):.6f}  "
        f"slope 2: {random_stop_linearized(n, alpha, lip, sigma, 2.0):.6f}  "
        f"stated 4 ln n: {random_stop_stated(n, alpha, lip, sigma):.6f}  H_n={harmonic(n):.4f}"
    )
    analytic = math.exp(2 * (math.e - 1) / (4 - 2 * (math.e - 1)))
    print(
        f"[stopping threshold] 2(e-1)H_n <= 4 ln n for every n >= {harmonic_threshold(1_000_000)} "
        f"(checked to 10^6; 2(e-1)(1 + ln n) <= 4 ln n once n >= {analytic:.1f})"
    )


# --- Figures --------------------------------------------------------------------------------------


def plot_direction_pairs(q: float = 0.1, sigma: float = 1.0) -> Figure:
    y = np.linspace(-3.5, 4.5, 801)
    p0 = np.exp(-(y**2) / (2 * sigma**2)) / math.sqrt(2 * math.pi * sigma**2)
    p1 = np.exp(-((y - 1) ** 2) / (2 * sigma**2)) / math.sqrt(2 * math.pi * sigma**2)
    fig, (left, right) = plt.subplots(1, 2, figsize=(11, 4))
    left.plot(y, p0, color="C0", label="$P_0$ = N(0, 1)")
    left.plot(y, p1, color="C3", label="$P_1$ = N(1, 1)")
    left.plot(y, (1 - q) * p0 + q * p1, "k--", label=rf"$P_q$, $q = {q:g}$")
    z = 1.2
    for point, colour in ((0.5 - z, "C0"), (0.5 + z, "C3")):
        h0 = math.exp(-(point**2) / 2) / math.sqrt(2 * math.pi)
        h1 = math.exp(-((point - 1) ** 2) / 2) / math.sqrt(2 * math.pi)
        left.vlines(point, 0, max(h0, h1), color=colour, linestyle=":")
        left.plot([point, point], [h0, h1], "o", color=colour)
    left.axvline(0.5, color="gray", linewidth=0.8)
    left.annotate("mirror $y = 1/2$", (0.5, 0.42), (1.1, 0.43), fontsize=9, arrowprops={"arrowstyle": "->"})
    left.set(xlabel="output $y$", ylabel="density", title=rf"A reflected pair $y = 1/2 \pm {z}$", ylim=(0, 0.46))
    left.legend(loc="upper right", fontsize=8)

    a_values = np.linspace(0.01, 4.0, 400)
    for alpha, style in ((2.0, "-"), (8.0, "--"), (32.0, ":")):
        for q_i, colour in ((0.01, "C0"), (0.1, "C1"), (0.5, "C2")):
            log_a, log_b = pair_log_moments(a_values, q_i, alpha)
            right.plot(2 * a_values, log_a - log_b, style, color=colour, label=rf"$q = {q_i:g}$, $\alpha = {alpha:g}$")
    right.set(
        yscale="log",
        ylim=(1e-8, 100),
        xlabel=r"privacy loss of the pair, $2a = z/\sigma^2$",
        ylabel="ln A' - ln B'",
        title="Remove minus add, for every reflected pair",
    )
    right.legend(fontsize=7, ncol=3, loc="lower right")
    return fig


def plot_direction_tanh(a: float = 1.0, q: float = 0.1, alpha: float = 4.0) -> Figure:
    b = float(mixture_parameter(np.array([a]), q)[0])
    d = a - b
    centre = (a + b) / 2
    x = np.linspace(-2.5, 4.5, 701)
    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.plot(x, np.tanh(x), "k", label=r"$\tanh$")
    narrow = (b, a)
    wide = (a - alpha * d, a + (alpha - 1) * d)
    for (lo, hi), colour, name, offset in (
        (wide, "C1", r"$I_2$, width $(2\alpha - 1)d$", 0.0),
        (narrow, "C0", r"$I_1 = [b, a]$, width $d$", 0.0),
    ):
        xs = np.linspace(lo, hi, 100)
        ax.fill_between(xs, 0, np.tanh(xs), color=colour, alpha=0.3 + offset, label=name)
        avg = tanh_average(centre, (hi - lo) / 2)
        ax.hlines(avg, lo, hi, color=colour, linewidth=2)
    ax.axvline(centre, color="gray", linestyle=":", label=r"common centre $(a + b)/2$")
    ax.set(
        xlabel=r"$u$",
        ylabel=r"$\tanh u$",
        title=rf"Averages of tanh over two intervals ($a = {a:g}$, $q = {q:g}$, $\alpha = {alpha:g}$)",
    )
    ax.legend(fontsize=8, loc="lower right")
    return fig


def plot_poisson(q: float = 0.01, sigma: float = 1.0) -> Figure:
    fig, (left, right) = plt.subplots(1, 2, figsize=(11, 4))
    gammas = np.linspace(0.0, 3.0, 301)
    for q_i, colour in ((0.5, "C2"), (0.1, "C1"), (0.01, "C0")):
        left.plot(gammas, [remove_gamma_map(float(g), q_i) for g in gammas], color=colour, label=rf"$q = {q_i:g}$")
    left.axhline(0.0, color="k", linewidth=0.8)
    left.axvline(1.0, color="gray", linestyle=":")
    left.axhspan(-40, 0, color="gray", alpha=0.15)
    left.text(0.05, -25, "γ' < 0: both sides equal 1 - γ", fontsize=8)
    left.set(
        xlabel=r"level $\gamma$ of the subsampled pair",
        ylabel="level γ' = 1 + (γ - 1)/q of the base pair",
        ylim=(-40, 40),
        title="The remove identity reads the base pair at a stretched level",
    )
    left.legend(fontsize=8)

    eps = np.geomspace(1e-4, 2.0, 241)
    rem = [remove_hockey_direct(float(e), q, sigma) for e in eps]
    add = [add_hockey_direct(float(e), q, sigma) for e in eps]
    right.plot(eps, rem, color="C3", label=r"remove, $E_{e^\varepsilon}(P_q\|P_0)$")
    right.plot(
        eps[::12],
        [remove_hockey_identity(float(e), q, sigma) for e in eps[::12]],
        "o",
        color="C3",
        mfc="none",
        label="q E(γ') of the base pair",
    )
    eps_add = np.geomspace(1e-4, -math.log(1 - q) * 0.999, 40)
    right.plot(eps, [v if v > 0 else math.nan for v in add], color="C0", label=r"add, $E_{e^\varepsilon}(P_0\|P_q)$")
    right.plot(
        eps_add[::4],
        [add_hockey_identity(float(e), q, sigma) for e in eps_add[::4]],
        "s",
        color="C0",
        mfc="none",
        label="c E(γq/c) of the base pair",
    )
    right.axvline(-math.log(1 - q), color="C0", linestyle=":", linewidth=0.8)
    right.set(
        xscale="log",
        yscale="log",
        ylim=(1e-12, 1e-1),
        xlabel=r"$\varepsilon$",
        ylabel=r"$\delta(\varepsilon)$",
        title=rf"One step, $q = {q:g}$, $\sigma = {sigma:g}$: identities are exact",
    )
    right.legend(fontsize=8)
    return fig


def plot_clone_bars(eps0: float = math.log(4)) -> Figure:
    r = kary_rr(eps0, 3)
    q0, q1 = rr_components(r[0], r[1], eps0)
    clone, left_part = clone_split(r[2], r[0], r[1], eps0)
    fig, (left, right) = plt.subplots(1, 2, figsize=(11, 4))
    outs = np.arange(3)
    p = math.exp(eps0) / (math.exp(eps0) + 1)
    w = 0.38
    left.bar(outs - w / 2, p * q0, w, color="C0", label=r"$p\,Q_0$")
    left.bar(outs - w / 2, (1 - p) * q1, w, bottom=p * q0, color="C1", label=r"$(1-p)\,Q_1$")
    left.bar(outs + w / 2, (1 - p) * q0, w, color="C0", alpha=0.55, label=r"$(1-p)\,Q_0$")
    left.bar(outs + w / 2, p * q1, w, bottom=(1 - p) * q0, color="C1", alpha=0.55, label=r"$p\,Q_1$")
    left.set(
        xticks=outs,
        xticklabels=["report 0", "report 1", "report 2"],
        ylabel="probability",
        title=r"$R(x^0)$ (left bars) and $R(x^1)$ (right bars) from $Q_0, Q_1$",
    )
    left.legend(fontsize=8)
    right.bar(outs, clone, 0.6, color="C2", label=r"clone part $e^{-\varepsilon_0}\cdot\frac{1}{2}(Q_0 + Q_1)$")
    right.bar(
        outs, left_part, 0.6, bottom=clone, color="C7", label=r"leftover $(1 - e^{-\varepsilon_0})\,\mathrm{LO}(2)$"
    )
    right.set(
        xticks=outs,
        xticklabels=["report 0", "report 1", "report 2"],
        ylabel="probability",
        title=r"Another user's $R(2)$ = clone part + leftover",
    )
    right.legend(fontsize=8)
    return fig


def plot_clone_loss(n: int = 1000, eps0: float = 2.0, delta: float = 1e-6) -> Figure:
    mu = (n - 1) * math.exp(-eps0)
    c = round(mu)
    c_min, eps_mine, _, _ = clone_closed_form(n, eps0, delta)
    a = np.arange(0, c + 2).astype(float)
    fig, (left, right) = plt.subplots(1, 2, figsize=(11, 4))
    loss = clone_loss(a, c, eps0)
    left.plot(a, loss, "k.", markersize=3, label=rf"loss at $N_c = {c}$")
    half = math.sqrt(c * math.log(2 / delta) / 2) + 0.5
    left.axvspan((c + 1) / 2 - half, (c + 1) / 2 + half, color="C0", alpha=0.2, label="Hoeffding window")
    left.axhline(eps_mine, color="C3", linestyle="--", label=rf"$\varepsilon = {eps_mine:.3f}$ (closed form)")
    left.set(
        xlabel=r"first count $a$",
        ylabel="privacy loss of the clone pair",
        ylim=(-eps0 - 0.2, eps0 + 0.2),
        title=rf"$n = {n}$, $\varepsilon_0 = {eps0:g}$: loss given $N_c$",
    )
    left.legend(fontsize=8, loc="upper left")
    cs = np.arange(max(1, int(mu - 5 * math.sqrt(mu))), int(mu + 5 * math.sqrt(mu)))
    pc = math.exp(-eps0)
    pmf = [
        math.exp(
            math.lgamma(n) - math.lgamma(k + 1) - math.lgamma(n - k) + k * math.log(pc) + (n - 1 - k) * math.log1p(-pc)
        )
        for k in cs
    ]
    right.bar(cs, pmf, 1.0, color="C7", label=r"law of the clone count $N_c$")
    right.axvline(c_min, color="C3", linestyle="--", label=rf"$n_{{c,\min}} = {c_min:.1f}$")
    right.axvline(mu, color="k", linestyle=":", label=rf"$\mu = {mu:.1f}$")
    right.set(xlabel="number of clones $n_c$", ylabel="probability", title="Chernoff: fewer clones is rare")
    right.legend(fontsize=8)
    return fig


def plot_shift(sigma: float = 1.0, s: float = 1.0) -> Figure:
    a = 0.5
    y = np.linspace(-3.5, 4.5, 801)

    def dens(m: float) -> FloatArray:
        return np.exp(-((y - m) ** 2) / (2 * sigma**2)) / math.sqrt(2 * math.pi * sigma**2)

    fig, (left, right) = plt.subplots(1, 2, figsize=(11, 4.2))
    left.plot(y, 0.5 * dens(0.0) + 0.5 * dens(s), "C0", label=r"$X + Z$, $X$ uniform on $\{0, 1\}$")
    left.plot(y, 0.5 * dens(s - a) + 0.5 * dens(s), "C1--", label="Y' = Y + $W_2$ + Z")
    left.plot(y, dens(s), "k:", label="X' + Z, with X' = 1")
    top = 0.47
    left.annotate("", (s - a, top), (0.0, top), arrowprops={"arrowstyle": "->", "color": "C0", "lw": 2})
    left.annotate("", (s, top), (s - a, top), arrowprops={"arrowstyle": "->", "color": "C3", "lw": 2})
    left.text(s + 0.1, top + 0.01, rf"$W_1$, carried ($z = {s - a:g}$)", color="C0", fontsize=8)
    left.text(s + 0.1, top - 0.03, rf"$W_2$, paid ($a = {a:g}$)", color="C3", fontsize=8)
    left.set(xlabel="value", ylabel="density", ylim=(0, 0.53), title="Splitting the displacement from 0 to 1")
    left.legend(fontsize=8, loc="upper left")

    pays = np.linspace(0.0, s, 101)
    right.plot(pays, [gaussian_cost(2.0, float(p), sigma) for p in pays], "C3", label=r"bound $R_2(a) = a^2/\sigma^2$")
    right.plot(
        pays, [half_mixture_renyi2(float(p), sigma) for p in pays], "k", label="exact $D_2$ of Y' against X' + Z"
    )
    right.set(
        xlabel=r"shift paid by the noise, $a$ (carried shift $z = 1 - a$)",
        ylabel=r"$D_2$",
        title="What the constructed Y' costs, σ = 1",
    )
    right.legend(fontsize=8)
    return fig


def plot_stopping(n: int = 1000, alpha: float = 2.0, lip: float = 1.0, sigma: float = 2.0) -> Figure:
    ts = np.unique(np.round(np.geomspace(1, n, 80)).astype(int))
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    ax.plot(
        ts,
        [fixed_stop_rdp(int(t), n, alpha, lip, sigma) for t in ts],
        "C3",
        label=r"stop at $n$: $2\alpha L^2/(\sigma^2(n - t + 1))$",
    )
    ax.plot(ts, [random_stop_rdp(int(t), n, alpha, lip, sigma) for t in ts], "C0", label="random stop: convexity bound")
    ax.axhline(
        random_stop_linearized(n, alpha, lip, sigma, math.e - 1),
        color="C0",
        linestyle="--",
        label=r"after $e^x \leq 1 + (e - 1)x$",
    )
    ax.axhline(
        random_stop_stated(n, alpha, lip, sigma),
        color="k",
        linestyle=":",
        label=r"stated: $4\alpha L^2 \ln n/(n\sigma^2)$",
    )
    ax.set(
        xscale="log",
        yscale="log",
        xlabel="position $t$ of the record",
        ylabel=r"RDP bound at $\alpha = 2$",
        title=rf"One pass, $n = {n}$, $L = {lip:g}$, $\sigma = {sigma:g}$",
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

    out.mkdir(parents=True, exist_ok=True)
    figures = {
        "apx_amp_direction_pairs.svg": plot_direction_pairs(),
        "apx_amp_direction_tanh.svg": plot_direction_tanh(),
        "apx_amp_poisson.svg": plot_poisson(),
        "apx_amp_clone_bars.svg": plot_clone_bars(),
        "apx_amp_clone_loss.svg": plot_clone_loss(),
        "apx_amp_shift.svg": plot_shift(),
        "apx_amp_stopping.svg": plot_stopping(),
    }
    for name, fig in figures.items():
        fig.tight_layout()
        fig.savefig(out / name, metadata={"Date": None})  # no timestamp, so output is reproducible
        plt.close(fig)
        print(f"saved {out / name}")


if __name__ == "__main__":
    main()
