"""Check the derivations of the appendix "Mechanisms and their limits" and draw its figures.

    uv run python code/appendix-derivations/apx_mechanisms.py
    uv run python code/appendix-derivations/apx_mechanisms.py --seed 1 --out /tmp/figures

Prints every number quoted in book-full/appendix-derivations/10-mechanisms.md (they match exactly
with the default seed), then saves the appendix's figures.

Conventions: P and Q are the output distributions of a mechanism on two neighbouring datasets D and D',
with probabilities p(y) and q(y). E_γ(P‖Q) = Σ_y (p(y) - γ q(y))_+ is the hockey-stick divergence.
The four-point pair lists its outputs in the order ("flag D", 0, 1, "flag D'"). Laplace and Gumbel
noise are given by their scale. Logarithms are natural. Exact quantities are computed by sums or by
numerical integration on a fine grid; simulation is used only to check them.
"""

import argparse
import math
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from itertools import permutations
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # for code/shared
from shared.figures import configure
from shared.stats import normal_cdf

configure("apx_mechanisms")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from numpy.typing import NDArray

# Figures are build output: _build/figures/<part>/, which the build copies into each book.
FIGURES = Path(__file__).resolve().parents[2] / "_build" / "figures" / Path(__file__).parent.name

FloatArray = NDArray[np.float64]


# --- 1. Every (ε, δ) pair is a post-processing of the four-point pair -------------------------------

# The worked example of the page: three outputs, e^ε = 2.
SMALL_P: list[float] = [0.6, 0.3, 0.1]
SMALL_Q: list[float] = [0.2, 0.3, 0.5]
SMALL_EPS = math.log(2.0)

# The figure's pair: N(0, 1) on D and N(1, 0.8²) on D', reported in eight bins. The spreads differ,
# so the two hockey-stick divergences differ and the equalizing step has work to do.
GAUSS_EDGES: list[float] = [-1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5]
GAUSS_SD_Q = 0.8
GAUSS_EPS = 1.0


def hockey_stick(p: FloatArray, q: FloatArray, gamma: float) -> float:
    """E_γ(P‖Q) = Σ_y (p(y) - γ q(y))_+."""
    return float(np.maximum(p - gamma * q, 0.0).sum())


def exact_delta(p: FloatArray, q: FloatArray, eps: float) -> float:
    """The smallest δ for which the pair is (ε, δ)-DP in both directions."""
    return max(hockey_stick(p, q, math.exp(eps)), hockey_stick(q, p, math.exp(eps)))


def binned_gaussian_pair(
    edges: Sequence[float], shift: float = 1.0, sd_q: float = GAUSS_SD_Q
) -> tuple[FloatArray, FloatArray]:
    """Probabilities of the bins (-inf, e_1], (e_1, e_2], ..., (e_m, inf) under N(0, 1) and N(shift, sd_q²)."""
    cuts = [-math.inf, *edges, math.inf]

    def cdf(x: float, mean: float, sd: float) -> float:
        return 0.0 if x == -math.inf else 1.0 if x == math.inf else normal_cdf((x - mean) / sd)

    p = [cdf(cuts[i + 1], 0.0, 1.0) - cdf(cuts[i], 0.0, 1.0) for i in range(len(cuts) - 1)]
    q = [cdf(cuts[i + 1], shift, sd_q) - cdf(cuts[i], shift, sd_q) for i in range(len(cuts) - 1)]
    return np.array(p), np.array(q)


@dataclass(frozen=True)
class Decomposition:
    """P = (1 - δ) P̄ + δ R_D and Q = (1 - δ) Q̄ + δ R_D', with (P̄, Q̄) an ε-DP pair."""

    m_p: FloatArray  # min(p, e^ε q), before equalizing
    m_q: FloatArray  # min(q, e^ε p)
    mass_p: float  # A = 1 - E_{e^ε}(P‖Q)
    mass_q: float  # B = 1 - E_{e^ε}(Q‖P)
    theta: float  # the mixing weight that equalizes the masses (applied to the heavier side)
    pure_p: FloatArray  # m̂_P, mass 1 - δ
    pure_q: FloatArray  # m̂_Q, mass 1 - δ
    rest_p: FloatArray  # R_D = (p - m̂_P) / δ
    rest_q: FloatArray  # R_D' = (q - m̂_Q) / δ


def equalize(m_big: FloatArray, m_small: FloatArray, eps: float) -> tuple[FloatArray, float]:
    """Mix the heavier part with e^{-ε} times the lighter one until both have the lighter one's mass."""
    a, b = float(m_big.sum()), float(m_small.sum())
    theta = (a - b) / (a - math.exp(-eps) * b) if a > b else 0.0
    return (1 - theta) * m_big + theta * math.exp(-eps) * m_small, theta


def mass_after_mixing(m_big: FloatArray, m_small: FloatArray, eps: float, theta: float) -> float:
    """Mass of (1 - θ) m_big + θ e^{-ε} m_small: it falls linearly from A to e^{-ε} B."""
    return float((1 - theta) * m_big.sum() + theta * math.exp(-eps) * m_small.sum())


def decompose(p: FloatArray, q: FloatArray, eps: float, delta: float) -> Decomposition:
    """The construction of the derivation. Needs E_{e^ε}(P‖Q) <= δ, E_{e^ε}(Q‖P) <= δ and δ > 0."""
    g = math.exp(eps)
    m_p, m_q = np.minimum(p, g * q), np.minimum(q, g * p)
    a, b = float(m_p.sum()), float(m_q.sum())
    if a >= b:
        mixed_p, theta = equalize(m_p, m_q, eps)
        mixed_q = m_q
    else:
        mixed_q, theta = equalize(m_q, m_p, eps)
        mixed_p = m_p
    scale = (1 - delta) / float(mixed_p.sum())
    pure_p, pure_q = scale * mixed_p, scale * mixed_q
    return Decomposition(
        m_p=m_p,
        m_q=m_q,
        mass_p=a,
        mass_q=b,
        theta=theta,
        pure_p=pure_p,
        pure_q=pure_q,
        rest_p=(p - pure_p) / delta,
        rest_q=(q - pure_q) / delta,
    )


def four_point_pair(eps: float, delta: float) -> tuple[FloatArray, FloatArray]:
    """P* and Q* on the outputs ("flag D", 0, 1, "flag D'"), with t = e^ε / (1 + e^ε)."""
    t = math.exp(eps) / (1 + math.exp(eps))
    p_star = np.array([delta, (1 - delta) * t, (1 - delta) * (1 - t), 0.0])
    q_star = np.array([0.0, (1 - delta) * (1 - t), (1 - delta) * t, delta])
    return p_star, q_star


def channel(dec: Decomposition, eps: float, delta: float) -> FloatArray:
    """The 4 x n channel K: flags go to R_D and R_D', the reports 0 and 1 to the rows of the δ = 0 channel."""
    p_bar, q_bar = dec.pure_p / (1 - delta), dec.pure_q / (1 - delta)
    g = math.exp(eps)
    row0 = (g * p_bar - q_bar) / (g - 1)
    row1 = (g * q_bar - p_bar) / (g - 1)
    return np.vstack([dec.rest_p, row0, row1, dec.rest_q])


@dataclass(frozen=True)
class ChannelCheck:
    pairs: int
    worst_reproduction: float  # largest |P*K - P| or |Q*K - Q| over all outputs and pairs
    smallest_entry: float  # smallest entry of any channel (should be >= 0 up to rounding)
    worst_row_sum: float  # largest |row sum - 1|
    worst_ratio: float  # largest max(P̄/Q̄, Q̄/P̄) / e^ε (should be <= 1)
    worst_excess: float  # largest m̂ - p or m̂ - q (should be <= 0)


def random_dp_pair(rng: np.random.Generator, n: int, eps: float) -> tuple[FloatArray, FloatArray, float]:
    """A random pair on n outputs and its exact δ at ε; one draw in four has outputs only one side can produce."""
    p = rng.dirichlet(np.ones(n))
    q = rng.dirichlet(np.ones(n))
    if rng.random() < 0.25:
        p[0], q[-1] = p[0] + p[-1], q[-1] + q[0]  # move mass within each distribution, so both still sum to 1
        p[-1], q[0] = 0.0, 0.0
    return p, q, exact_delta(p, q, eps)


def check_random_channels(rng: np.random.Generator, pairs: int, n: int, eps: float) -> ChannelCheck:
    rep = neg = rows = ratio = excess = 0.0
    tiny = 1e-300
    for _ in range(pairs):
        p, q, delta = random_dp_pair(rng, n, eps)
        if delta <= 0 or delta >= 1:  # the construction needs 0 < δ < 1
            continue
        dec = decompose(p, q, eps, delta)
        k = channel(dec, eps, delta)
        p_star, q_star = four_point_pair(eps, delta)
        rep = max(rep, float(np.abs(p_star @ k - p).max()), float(np.abs(q_star @ k - q).max()))
        neg = min(neg, float(k.min()))
        rows = max(rows, float(np.abs(k.sum(axis=1) - 1).max()))
        both = (dec.pure_p > tiny) & (dec.pure_q > tiny)
        if not both.any():
            continue
        r = np.maximum(dec.pure_p[both] / dec.pure_q[both], dec.pure_q[both] / dec.pure_p[both])
        ratio = max(ratio, float(r.max()) / math.exp(eps))
        excess = max(excess, float((dec.pure_p - p).max()), float((dec.pure_q - q).max()))
    return ChannelCheck(pairs, rep, neg, rows, ratio, excess)


def tradeoff_vertices(p: FloatArray, q: FloatArray) -> tuple[FloatArray, FloatArray]:
    """Corners of the trade-off curve of (P, Q): add outputs to the rejection set in decreasing q/p."""
    with np.errstate(divide="ignore"):
        ratio = np.where(p > 0, q / np.where(p > 0, p, 1.0), np.inf)
    order = np.argsort(-ratio, kind="stable")
    alpha = np.concatenate([[0.0], np.cumsum(p[order])])
    beta = np.concatenate([[1.0], 1 - np.cumsum(q[order])])
    return alpha, np.maximum(beta, 0.0)


def tradeoff_at(p: FloatArray, q: FloatArray, alphas: FloatArray) -> FloatArray:
    """The trade-off function (smallest false negative rate at each false positive rate)."""
    a, b = tradeoff_vertices(p, q)
    return np.interp(alphas, a, b)


def tradeoff_boundary(alphas: FloatArray, eps: float, delta: float) -> FloatArray:
    """f_{ε,δ}(α) = max{0, 1 - δ - e^ε α, e^{-ε}(1 - δ - α)}."""
    g = math.exp(eps)
    return np.maximum(0.0, np.maximum(1 - delta - g * alphas, (1 - delta - alphas) / g))


# --- 2. (ε, δ)-DP implies δ-approximate zCDP ---------------------------------------------------------


def renyi(p: FloatArray, q: FloatArray, alpha: float) -> float:
    """D_α(P‖Q) = ln(Σ p^α q^{1-α}) / (α - 1), over outputs where p > 0 (q > 0 there for pure-DP pairs)."""
    keep = p > 0
    return math.log(float(np.sum(p[keep] ** alpha * q[keep] ** (1 - alpha)))) / (alpha - 1)


def worst_zcdp_fraction(p_bar: FloatArray, q_bar: FloatArray, eps: float, alphas: Sequence[float]) -> float:
    """Largest D_α / (α ε²/2) over the orders and both directions: the share of the zCDP line used."""
    return max(max(renyi(p_bar, q_bar, a), renyi(q_bar, p_bar, a)) / (a * eps**2 / 2) for a in alphas)


# --- 3. The stability-based histogram ------------------------------------------------------------


def stability_threshold(eps: float, delta: float) -> float:
    """T = 1 + ln(1/(2δ))/ε: a count of 1 plus Lap(1/ε) noise clears T with probability exactly δ."""
    return 1 + math.log(1 / (2 * delta)) / eps


def laplace_cdf(x: float, b: float) -> float:
    return 0.5 * math.exp(x / b) if x < 0 else 1 - 0.5 * math.exp(-x / b)


def release_probability(count: float, threshold: float, eps: float) -> float:
    """Pr[count + Z > T] for Z ~ Lap(1/ε)."""
    return 1 - laplace_cdf(threshold - count, 1 / eps)


def bin_error_cdf(count: int, a: float, threshold: float, eps: float) -> float:
    """Pr[error of one non-empty bin <= a]: the error is |Z| if the bin is released and the count if not."""
    b = 1 / eps
    lo = max(threshold - count, -a)  # released with |Z| <= a means max(T - c, -a) < Z <= a
    released_small = max(laplace_cdf(a, b) - laplace_cdf(lo, b), 0.0)
    suppressed = laplace_cdf(threshold - count, b) if count <= a else 0.0
    return released_small + suppressed


def stability_error_quantile(count: int, bins: int, eps: float, delta: float, beta: float) -> float:
    """Exact 1-β quantile of the ℓ∞ error when `bins` bins all hold `count` records (empty bins add 0)."""
    t = stability_threshold(eps, delta)
    lo, hi = 0.0, t + math.log(bins / beta) / eps + 1.0
    for _ in range(200):
        mid = (lo + hi) / 2
        if bin_error_cdf(count, mid, t, eps) ** bins >= 1 - beta:
            hi = mid
        else:
            lo = mid
    return hi


def stability_worst_quantile(bins: int, eps: float, delta: float, beta: float) -> tuple[float, int]:
    """Largest 1-β error quantile over equal counts c = 1, 2, ...; returns it and the worst count."""
    t = stability_threshold(eps, delta)
    top = int(t + math.log(bins / beta) / eps) + 10
    values = [(stability_error_quantile(c, bins, eps, delta, beta), c) for c in range(1, top)]
    return max(values)


def stability_error_bound(bins: int, eps: float, delta: float, beta: float) -> float:
    """T + ln(k/β)/ε: every non-empty bin's error is at most T + |Z|, and all k noises stay below ln(k/β)/ε."""
    return stability_threshold(eps, delta) + math.log(bins / beta) / eps


def laplace_histogram_error(n_bins: float, eps: float, beta: float) -> float:
    """Exact 1-β quantile of the ℓ∞ error of Lap(1/ε) on every one of N bins: -ln(1 - (1-β)^{1/N})/ε."""
    return -math.log(-math.expm1(math.log1p(-beta) / n_bins)) / eps


def packing_histogram_bound(n_bins: float, eps: float, beta: float) -> float:
    """The pure-DP lower bound of the lower-bounds chapter: ½(ln(N(1-β))/ε - 1)."""
    return 0.5 * (math.log(n_bins * (1 - beta)) / eps - 1)


def simulate_stability_error(
    rng: np.random.Generator, count: int, bins: int, eps: float, delta: float, trials: int
) -> FloatArray:
    """ℓ∞ errors of the stability histogram on `bins` bins of `count` records each."""
    t = stability_threshold(eps, delta)
    z = rng.laplace(0.0, 1 / eps, size=(trials, bins))
    err = np.where(count + z > t, np.abs(z), float(count))
    return err.max(axis=1)


def crossing(f: Callable[[float], float], level: float, lo: float, hi: float) -> float:
    """The N (on a log scale) at which an increasing curve f(N) reaches `level`, by bisection."""
    a, b = math.log(lo), math.log(hi)
    for _ in range(200):
        mid = (a + b) / 2
        if f(math.exp(mid)) < level:
            a = mid
        else:
            b = mid
    return math.exp(b)


# --- 4. Lower bound for heavy-tailed means -----------------------------------------------------------


def two_point(alpha: float, moment: float, k: int) -> tuple[float, float]:
    """(a, p) for P1 = (1-p)·δ_0 + p·δ_a with p a = 2α and p a^k = moment (needs 2α <= moment^{1/k})."""
    a = (moment / (2 * alpha)) ** (1 / (k - 1))
    return a, 2 * alpha / a


def central_moment(a: float, p: float, k: int) -> float:
    """E|X - μ|^k for X = a with probability p and 0 otherwise, μ = p a."""
    return (1 - p) * (p * a) ** k + p * (a * (1 - p)) ** k


def heavy_lower_bound(alpha: float, eps: float, beta: float, k: int, moment: float = 1.0) -> float:
    """Records needed by any ε-DP estimator with error below α w.p. 1-β: ln((1-β)/β) / ln(1 - p + p e^ε)."""
    _, p = two_point(alpha, moment, k)
    return math.log((1 - beta) / beta) / math.log1p(p * math.expm1(eps))


def heavy_lower_bound_simple(alpha: float, eps: float, beta: float, k: int, moment: float = 1.0) -> float:
    """The weaker closed form ln((1-β)/β) / (p (e^ε - 1)), from 1 + x <= e^x."""
    _, p = two_point(alpha, moment, k)
    return math.log((1 - beta) / beta) / (p * math.expm1(eps))


def bounded_lower_bound(alpha: float, eps: float, beta: float) -> float:
    """The same argument with the far point at a = 1 (data in [0, 1]): p = 2α."""
    return math.log((1 - beta) / beta) / math.log1p(2 * alpha * math.expm1(eps))


def heavy_upper_bound(alpha: float, eps: float, sigma: float = 1.0) -> float:
    """Records at which the chapter's clipped mean reaches RMSE α: σ/√n + 2σ√(2√2/(nε)) = α."""
    return (sigma * (1 + 2 * math.sqrt(2 * math.sqrt(2) / eps)) / alpha) ** 2


# --- 5. Sparse vector with c answers above --------------------------------------------------------


def laplace_log_cdf(x: FloatArray, b: float) -> FloatArray:
    """ln Pr[Z < x] for Z ~ Lap(b), accurate in both tails."""
    return np.where(x < 0, math.log(0.5) + np.minimum(x, 0.0) / b, np.log1p(-0.5 * np.exp(-np.maximum(x, 0.0) / b)))


def laplace_log_sf(x: FloatArray, b: float) -> FloatArray:
    """ln Pr[Z >= x] for Z ~ Lap(b)."""
    return laplace_log_cdf(-x, b)


def laplace_log_pdf(x: FloatArray, b: float) -> FloatArray:
    return -np.abs(x) / b - math.log(2 * b)


def logsumexp(x: FloatArray) -> float:
    m = float(np.max(x))
    return m + math.log(float(np.sum(np.exp(x - m))))


def pattern_log_prob(
    f: Sequence[float], above: Sequence[bool], threshold: float, b_rho: float, b_nu: float, points_per_scale: int = 200
) -> float:
    """ln Pr[the answers are exactly `above` (True = ⊤)], integrating over the threshold noise ρ ~ Lap(b_rho)."""
    r = np.linspace(-40 * b_rho, 40 * b_rho, 80 * points_per_scale + 1)
    total = laplace_log_pdf(r, b_rho)
    for fi, up in zip(f, above, strict=True):
        x = threshold + r - fi  # ⊥ needs ν < x, ⊤ needs ν >= x
        total = total + (laplace_log_sf(x, b_nu) if up else laplace_log_cdf(x, b_nu))
    return logsumexp(total) + math.log(float(r[1] - r[0]))


def svt_c_loss(m: int, c: int, eps1: float, eps2: float, noise_grows: bool = True) -> float:
    """Privacy loss of the output ⊥^m ⊤^c on f(D) = (0^m, 1^c), f(D') = (1^m, 0^c), Δ = 1, T = 0.

    The c-th ⊤ ends the run, so this is a complete output. With noise_grows the query noise is
    Lap(2c/ε₂) (Lyu, Su & Li's Algorithm 1); otherwise Lap(2/ε₂), as if c did not matter.
    """
    f_d = [0.0] * m + [1.0] * c
    f_d2 = [1.0] * m + [0.0] * c
    above = [False] * m + [True] * c
    b_nu = 2 * c / eps2 if noise_grows else 2 / eps2
    return pattern_log_prob(f_d, above, 0.0, 1 / eps1, b_nu) - pattern_log_prob(f_d2, above, 0.0, 1 / eps1, b_nu)


# --- 6. Gumbel top-c is the exponential mechanism without replacement --------------------------------

SIX_SCORES: list[float] = [4.0, 3.0, 3.0, 1.0, 0.0, 0.0]


def plackett_luce(weights: Sequence[float], order: Sequence[int]) -> float:
    """Pr[the first picks are `order`] when each round picks a remaining candidate in proportion to its weight."""
    remaining = float(sum(weights))
    prob = 1.0
    for i in order:
        prob *= weights[i] / remaining
        remaining -= weights[i]
    return prob


def ordered_pairs(k: int) -> list[tuple[int, int]]:
    return list(permutations(range(k), 2))


def gumbel_top2_frequencies(
    rng: np.random.Generator, scores: Sequence[float], scale: float, trials: int
) -> dict[tuple[int, int], float]:
    """Frequencies of the ordered top two of u_r + G_r, G_r ~ Gumbel(scale)."""
    u = np.asarray(scores, dtype=np.float64)
    noisy = u[None, :] + rng.gumbel(0.0, scale, size=(trials, len(scores)))
    top = np.argsort(-noisy, axis=1)[:, :2]
    codes = top[:, 0] * len(scores) + top[:, 1]
    counts = np.bincount(codes, minlength=len(scores) ** 2)
    return {(i, j): float(counts[i * len(scores) + j]) / trials for i, j in ordered_pairs(len(scores))}


def exponential_race_means(rng: np.random.Generator, scores: Sequence[float], scale: float, trials: int) -> FloatArray:
    """λ_r · mean of E_r = exp(-(u_r + G_r)/scale), which is 1 if E_r ~ Exp(rate λ_r = e^{u_r/scale})."""
    u = np.asarray(scores, dtype=np.float64)
    g = rng.gumbel(0.0, scale, size=(trials, len(scores)))
    e = np.exp(-(u[None, :] + g) / scale)
    return e.mean(axis=0) * np.exp(u / scale)


# --- 7. The alarm mechanism -----------------------------------------------------------------------


def alarm_probability(eps: float, delta: float, size: int, count: int) -> float:
    """a_c = δ Σ_{j < K - c} e^{jε} for c <= K, and 0 above K."""
    return delta * sum(math.exp(j * eps) for j in range(max(size - count, 0)))


def alarm_worst_slack(eps: float, delta: float, size: int) -> float:
    """Largest Pr[M(c) ∈ S] - e^ε Pr[M(c') ∈ S] - δ over neighbouring counts, both orders, every event.

    The events are ∅, {alarm}, {quiet} and both outputs; ∅ and both outputs never violate the bound.
    """
    g = math.exp(eps)
    worst = -math.inf
    for c in range(size + 1):
        a, b = alarm_probability(eps, delta, size, c), alarm_probability(eps, delta, size, c + 1)
        for x, y in ((a, b), (b, a)):
            worst = max(worst, x - g * y - delta, (1 - x) - g * (1 - y) - delta)
    return worst


def alarm_cap(eps: float, delta: float) -> float:
    """The condition of the derivation: a_0 <= (e^ε + δ)/(e^ε + 1)."""
    return (math.exp(eps) + delta) / (math.exp(eps) + 1)


def alarm_largest_size(eps: float, delta: float) -> int:
    """Largest K whose a_0 = δ(e^{Kε} - 1)/(e^ε - 1) stays within the cap."""
    size = 1
    while alarm_probability(eps, delta, size + 1, 0) <= alarm_cap(eps, delta):
        size += 1
    return size


# --- The table -----------------------------------------------------------------------------------


def print_table(rng: np.random.Generator) -> None:
    # 1. The small example.
    p, q = np.array(SMALL_P), np.array(SMALL_Q)
    eps = SMALL_EPS
    e_pq, e_qp = hockey_stick(p, q, 2.0), hockey_stick(q, p, 2.0)
    delta = max(e_pq, e_qp)
    dec = decompose(p, q, eps, delta)
    k = channel(dec, eps, delta)
    p_star, q_star = four_point_pair(eps, delta)
    print(f"[Four-point, small example, e^eps=2] E(P||Q)={e_pq:.4f}  E(Q||P)={e_qp:.4f}  delta={delta:.4f}")
    print(
        f"  m_P={np.round(dec.m_p, 4).tolist()}  A={dec.mass_p:.4f}  m_Q={np.round(dec.m_q, 4).tolist()}  B={dec.mass_q:.4f}"
    )
    print(f"  theta={dec.theta:.4f}  scale={(1 - delta) / dec.mass_q:.4f}")
    print(f"  pure_P={np.round(dec.pure_p, 4).tolist()}  pure_Q={np.round(dec.pure_q, 4).tolist()}")
    print(
        f"  Pbar={np.round(dec.pure_p / (1 - delta), 4).tolist()}  Qbar={np.round(dec.pure_q / (1 - delta), 4).tolist()}"
    )
    print(f"  R_D={np.round(dec.rest_p, 4).tolist()}  R_D'={np.round(dec.rest_q, 4).tolist()}")
    print(f"  K(.|0)={np.round(k[1], 4).tolist()}  K(.|1)={np.round(k[2], 4).tolist()}")
    print(f"  P*K={np.round(p_star @ k, 4).tolist()}  Q*K={np.round(q_star @ k, 4).tolist()}")

    # 1. The Gaussian pair of the figure.
    gp, gq = binned_gaussian_pair(GAUSS_EDGES)
    gd = exact_delta(gp, gq, GAUSS_EPS)
    gdec = decompose(gp, gq, GAUSS_EPS, gd)
    gk = channel(gdec, GAUSS_EPS, gd)
    gps, gqs = four_point_pair(GAUSS_EPS, gd)
    print(
        f"[Four-point, binned Gaussian, eps=1] E(P||Q)={hockey_stick(gp, gq, math.e):.4f}  "
        f"E(Q||P)={hockey_stick(gq, gp, math.e):.4f}  delta={gd:.4f}  A={gdec.mass_p:.4f}  "
        f"B={gdec.mass_q:.4f}  theta={gdec.theta:.4f}"
    )
    alphas = np.linspace(0.0, 1.0, 2001)
    pair_gap = float(np.min(tradeoff_at(gp, gq, alphas) - tradeoff_boundary(alphas, GAUSS_EPS, gd)))
    pure_gap = float(
        np.min(
            tradeoff_at(gdec.pure_p / (1 - gd), gdec.pure_q / (1 - gd), alphas)
            - tradeoff_boundary(alphas, GAUSS_EPS, 0)
        )
    )
    four_gap = float(np.max(np.abs(tradeoff_at(gps, gqs, alphas) - tradeoff_boundary(alphas, GAUSS_EPS, gd))))
    print(
        f"  reproduction error={max(float(np.abs(gps @ gk - gp).max()), float(np.abs(gqs @ gk - gq).max())):.1e}  "
        f"min trade-off gap: pair={pair_gap:.1e}  pure part={pure_gap:.1e}  four-point vs boundary={four_gap:.1e}"
    )

    # 1. Many random pairs.
    for n_out, eps_r in ((3, 0.5), (8, 1.0), (20, 2.0)):
        chk = check_random_channels(rng, 2000, n_out, eps_r)
        print(
            f"[Four-point, {chk.pairs} random pairs, {n_out} outputs, eps={eps_r:g}] "
            f"reproduction={chk.worst_reproduction:.1e}  smallest entry={chk.smallest_entry:.1e}  "
            f"row sums={chk.worst_row_sum:.1e}  ratio/e^eps-1={chk.worst_ratio - 1:.1e}  excess={chk.worst_excess:.1e}"
        )

    # 2. Approximate zCDP of the pure parts.
    orders = [1.001, *[1 + i / 10 for i in range(1, 200)]]
    small_frac = worst_zcdp_fraction(dec.pure_p / (1 - delta), dec.pure_q / (1 - delta), eps, orders)
    gauss_frac = worst_zcdp_fraction(gdec.pure_p / (1 - gd), gdec.pure_q / (1 - gd), GAUSS_EPS, orders)
    print(
        f"[zCDP of the pure parts] small example: rho=eps^2/2={eps**2 / 2:.4f}  largest D_a/(a rho)={small_frac:.4f}; "
        f"binned Gaussian: rho={GAUSS_EPS**2 / 2:.4f}  largest D_a/(a rho)={gauss_frac:.4f}"
    )

    # 3. The stability histogram.
    eps, delta, beta, bins = 1.0, 1e-6, 0.05, 10
    t = stability_threshold(eps, delta)
    worst_q, worst_c = stability_worst_quantile(bins, eps, delta, beta)
    sim = simulate_stability_error(rng, worst_c, bins, eps, delta, trials=200_000)
    print(
        f"[Stability histogram eps={eps:g} delta={delta:g}] T={t:.4f}  release at count 1={release_probability(1, t, eps):.3e}  "
        f"release at count T={release_probability(t, t, eps):.2f}"
    )
    print(
        f"  {bins} non-empty bins, beta={beta:g}: bound T+ln(k/beta)/eps={stability_error_bound(bins, eps, delta, beta):.4f}  "
        f"exact worst quantile={worst_q:.4f} at count {worst_c}  simulated={float(np.quantile(sim, 1 - beta)):.4f}"
    )
    for n_bins in (1e3, 1e6, 1e12):
        print(
            f"  N={n_bins:.0e}: Laplace exact={laplace_histogram_error(n_bins, eps, beta):.4f}  "
            f"packing lower bound={packing_histogram_bound(n_bins, eps, beta):.4f}  "
            f"stability bound={stability_error_bound(bins, eps, delta, beta):.4f}"
        )
    level = stability_error_bound(bins, eps, delta, beta)
    print(
        f"  stability bound beats Laplace from N={crossing(lambda n: laplace_histogram_error(n, eps, beta), level, 10, 1e30):.3e}  "
        f"and the pure-DP lower bound from N={crossing(lambda n: packing_histogram_bound(n, eps, beta), level, 10, 1e30):.3e}"
    )

    # 4. Heavy tails.
    eps, beta, alpha = 1.0, 0.05, 0.05
    a, pr = two_point(alpha, 1.0, 2)
    print(
        f"[Heavy tails eps={eps:g} beta={beta:g} alpha={alpha:g}, variance <= 1] a={a:.4f}  p={pr:.4f}  "
        f"variance={central_moment(a, pr, 2):.4f}  "
        f"n >= {heavy_lower_bound(alpha, eps, beta, 2):.1f} (simple form {heavy_lower_bound_simple(alpha, eps, beta, 2):.1f})  "
        f"clipped mean needs {heavy_upper_bound(alpha, eps):.0f}"
    )
    for kk in (3, 4):
        a, pr = two_point(alpha, 1.0, kk)
        print(
            f"  k={kk}: a={a:.4f}  p={pr:.4f}  moment={central_moment(a, pr, kk):.4f}  "
            f"n >= {heavy_lower_bound(alpha, eps, beta, kk):.1f}"
        )
    print(f"  bounded data in [0, 1]: n >= {bounded_lower_bound(alpha, eps, beta):.1f}")
    worst_moment = max(
        central_moment(*two_point(al, 1.0, kk), kk) for kk in (2, 3, 4, 6) for al in np.linspace(0.001, 0.5, 500)
    )
    print(f"  largest k-th central moment of P1 over alpha in [0.001, 0.5], k in (2, 3, 4, 6): {worst_moment:.4f}")

    # 5. Sparse vector with c answers.
    for c in (1, 2, 4):
        losses = "  ".join(f"m={m}: {svt_c_loss(m, c, 0.5, 0.5):.4f}" for m in (1, 10, 40))
        print(f"[Sparse vector c={c}, eps1=eps2=0.5] {losses}")
    print(f"  c=4 with noise Lap(2/eps2), m=40: {svt_c_loss(40, 4, 0.5, 0.5, noise_grows=False):.4f}")

    # 6. Gumbel top-2.
    scale = 2.0
    weights = [math.exp(u / scale) for u in SIX_SCORES]
    freq = gumbel_top2_frequencies(rng, SIX_SCORES, scale, trials=1_000_000)
    exact = {pair: plackett_luce(weights, pair) for pair in ordered_pairs(len(SIX_SCORES))}
    diff = max(abs(freq[pair] - exact[pair]) for pair in exact)
    tv = 0.5 * sum(abs(freq[pair] - exact[pair]) for pair in exact)
    print(
        f"[Gumbel top-2, scores {SIX_SCORES}, scale {scale:g}] Pr[(0,1)] exact={exact[(0, 1)]:.4f}  "
        f"simulated={freq[(0, 1)]:.4f}  largest difference={diff:.4f}  TV={tv:.4f}  sum exact={sum(exact.values()):.4f}"
    )
    means = exponential_race_means(rng, SIX_SCORES, scale, trials=1_000_000)
    print(f"  rate times mean arrival time: {np.round(means, 3).tolist()}")

    # 7. The alarm mechanism.
    eps, delta = 0.5, 1e-5
    print(
        f"[Alarm eps={eps:g} delta={delta:g}] cap (e^eps+delta)/(e^eps+1)={alarm_cap(eps, delta):.4f}  "
        f"a_0 at K=10={alarm_probability(eps, delta, 10, 0):.3e}  largest K={alarm_largest_size(eps, delta)}"
    )
    for size in (10, alarm_largest_size(eps, delta), alarm_largest_size(eps, delta) + 1):
        print(
            f"  K={size}: a_0={alarm_probability(eps, delta, size, 0):.4f}  a_1={alarm_probability(eps, delta, size, 1):.4f}  "
            f"largest slack={alarm_worst_slack(eps, delta, size):.3e}"
        )


# --- Figures --------------------------------------------------------------------------------------


def plot_fourpoint_split() -> Figure:
    p, q = binned_gaussian_pair(GAUSS_EDGES)
    delta = exact_delta(p, q, GAUSS_EPS)
    dec = decompose(p, q, GAUSS_EPS, delta)
    x = np.arange(len(p))
    labels = ["≤ −1.5", "−1", "−0.5", "0", "0.5", "1", "1.5", "> 1.5"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    for ax, total, pure, other, name, rest in (
        (axes[0], p, dec.pure_p, dec.pure_q, "P (on D)", r"$\delta R_D$"),
        (axes[1], q, dec.pure_q, dec.pure_p, "Q (on D')", "δ·R_D′"),
    ):
        ax.bar(
            x,
            pure,
            color="tab:blue",
            label=r"pure part, $(1-\delta)\overline{P}$" if name[0] == "P" else r"pure part, $(1-\delta)\overline{Q}$",
        )
        ax.bar(x, total - pure, bottom=pure, color="tab:red", label=f"leftover, {rest}")
        ax.step(
            np.concatenate([[x[0] - 0.5], x + 0.5]),
            np.concatenate([[math.exp(GAUSS_EPS) * other[0]], math.exp(GAUSS_EPS) * other]),
            where="pre",
            color="k",
            linestyle="--",
            label=r"$e^\varepsilon$ times the other pure part",
        )
        ax.set(
            xticks=x,
            ylim=(0, 0.75),
            xlabel="bin (upper edge)",
            title=f"{name}: {1 - delta:.3f} pure + {delta:.3f} leftover",
        )
        ax.set_xticklabels(labels)
        ax.legend(loc="upper left", fontsize=8)
    axes[0].set_ylabel("probability")
    fig.suptitle(rf"A binned Gaussian pair at $\varepsilon = 1$, $\delta = {delta:.4f}$", fontsize=11)
    return fig


def plot_fourpoint_tradeoff() -> Figure:
    p, q = binned_gaussian_pair(GAUSS_EDGES)
    delta = exact_delta(p, q, GAUSS_EPS)
    dec = decompose(p, q, GAUSS_EPS, delta)
    alphas = np.linspace(0.0, 1.0, 1001)
    fig, (left, right) = plt.subplots(1, 2, figsize=(11, 4.2))
    left.plot(
        alphas,
        tradeoff_boundary(alphas, GAUSS_EPS, delta),
        "k",
        linewidth=2,
        label=r"$f_{\varepsilon,\delta}$ = four-point pair",
    )
    left.plot(
        alphas, tradeoff_boundary(alphas, GAUSS_EPS, 0.0), "k:", label=r"$f_{\varepsilon,0}$ = randomized response"
    )
    left.plot(alphas, tradeoff_at(p, q, alphas), color="tab:orange", label="the pair (P, Q)")
    left.plot(
        alphas,
        tradeoff_at(dec.pure_p / (1 - delta), dec.pure_q / (1 - delta), alphas),
        color="tab:blue",
        linestyle="--",
        label=r"its pure part $(\overline{P}, \overline{Q})$",
    )
    left.set(
        xlim=(0, 1),
        ylim=(0, 1),
        xlabel="false positive rate",
        ylabel="false negative rate",
        title="Trade-off curves",
    )
    left.set_aspect("equal")
    left.legend(fontsize=8)

    big, small = (dec.m_p, dec.m_q) if dec.mass_p >= dec.mass_q else (dec.m_q, dec.m_p)
    thetas = np.linspace(0.0, 1.0, 201)
    right.plot(
        thetas,
        [mass_after_mixing(big, small, GAUSS_EPS, float(t)) for t in thetas],
        color="tab:blue",
        label="mass of the mixed heavier part",
    )
    right.axhline(
        float(small.sum()),
        color="tab:orange",
        linestyle="--",
        label=f"mass of the lighter part ({float(small.sum()):.4f})",
    )
    right.plot([dec.theta], [float(small.sum())], "ko")
    right.annotate(
        rf"$\theta^* = {dec.theta:.3f}$", (dec.theta, float(small.sum())), textcoords="offset points", xytext=(8, 8)
    )
    right.set(xlabel=r"mixing weight $\omega$", ylabel="total mass", title="Equalizing the two pure parts")
    right.legend(fontsize=8, loc="lower left")
    return fig


def plot_histogram_error() -> Figure:
    eps, delta, beta, bins = 1.0, 1e-6, 0.05, 10
    ns = [10 ** (i / 4) for i in range(8, 81)]
    worst_q, _ = stability_worst_quantile(bins, eps, delta, beta)
    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.plot(
        ns,
        [laplace_histogram_error(n, eps, beta) for n in ns],
        color="tab:orange",
        label="Laplace histogram (exact), pure DP",
    )
    lower = [packing_histogram_bound(n, eps, beta) for n in ns]
    ax.plot(ns, lower, color="tab:blue", label="packing lower bound for pure DP")
    ax.fill_between(ns, 0, lower, color="tab:blue", alpha=0.12)
    ax.axhline(
        stability_error_bound(bins, eps, delta, beta),
        color="tab:green",
        label=r"stability histogram, bound $\tau + \ln(k/\beta)/\varepsilon$",
    )
    ax.axhline(worst_q, color="tab:green", linestyle="--", label="stability histogram, exact worst case")
    ax.set(
        xscale="log",
        ylim=(0, 32),
        xlabel="number of bins N",
        ylabel=r"$\ell_\infty$ error at 95%",
        title=rf"$\varepsilon = 1$, $\delta = 10^{{-6}}$, {bins} non-empty bins",
    )
    ax.legend(fontsize=8, loc="lower right")
    return fig


def plot_histogram_release() -> Figure:
    eps, delta = 1.0, 1e-6
    t = stability_threshold(eps, delta)
    cs = np.linspace(0.0, 25.0, 501)
    fig, (left, right) = plt.subplots(1, 2, figsize=(11, 4))
    for ax, log in ((left, False), (right, True)):
        ax.plot(cs, [release_probability(float(c), t, eps) for c in cs], color="tab:blue")
        ints = list(range(1, 26))
        ax.plot(
            ints,
            [release_probability(c, t, eps) for c in ints],
            "o",
            color="tab:blue",
            markersize=4,
            label="integer counts",
        )
        ax.axvline(t, color="gray", linestyle=":", label=rf"threshold $\tau$ = {t:.2f}")
        ax.plot(
            [1],
            [release_probability(1, t, eps)],
            "o",
            color="tab:red",
            markersize=7,
            label=r"count 1: probability $\delta = 10^{-6}$",
        )
        ax.set(xlabel="true count c of the bin", ylabel="probability the bin is released")
        if log:
            ax.set(yscale="log", ylim=(1e-7, 2), title=r"Log scale: the tail below $\tau$ is exponential")
        else:
            ax.set(ylim=(-0.02, 1.05), title=r"Release probability $\Pr[c + Z > \tau]$")
        ax.legend(fontsize=8, loc="lower right" if log else "upper left")
    return fig


def plot_heavy_two_point() -> Figure:
    alpha, k = 0.15, 2
    a, p = two_point(alpha, 1.0, k)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.stem([0.0], [1.0], linefmt="tab:blue", markerfmt="o", basefmt=" ", label=r"$P_0$: all mass at 0, mean 0")
    ax.stem(
        [0.05, a],
        [1 - p, p],
        linefmt="tab:orange",
        markerfmt="s",
        basefmt=" ",
        label=rf"$P_1$: mass {p:.2f} at $v$ = {a:.2f}, mean $2a$ = {2 * alpha:.1f}",
    )
    ax.axvspan(-alpha, alpha, color="tab:blue", alpha=0.15, label=r"answers within $a$ of 0")
    ax.axvspan(alpha, 3 * alpha, color="tab:orange", alpha=0.15, label=r"answers within $a$ of $2a$")
    ax.axvline(alpha, color="k", linestyle=":", label=r"the test: is the estimate $\geq a$?")
    ax.set(
        xlim=(-0.4, a + 0.6),
        ylim=(0, 1.1),
        xlabel="record value",
        ylabel="probability",
        title=rf"Two distributions with variance at most 1, $a = {alpha}$",
    )
    ax.legend(fontsize=8, loc="center right")
    return fig


def plot_heavy_rates() -> Figure:
    eps, beta = 1.0, 0.05
    alphas = [10 ** (-3 + 2.69897 * i / 200) for i in range(201)]  # 0.001 to 0.5
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for k, color in ((2, "tab:red"), (3, "tab:purple"), (4, "tab:blue")):
        ax.plot(
            alphas, [heavy_lower_bound(a, eps, beta, k) for a in alphas], color=color, label=f"lower bound, k = {k}"
        )
    ax.plot(alphas, [bounded_lower_bound(a, eps, beta) for a in alphas], color="k", label="lower bound, data in [0, 1]")
    ax.plot(
        alphas,
        [heavy_upper_bound(a, eps) for a in alphas],
        color="tab:red",
        linestyle="--",
        label="clipped mean (upper bound), k = 2",
    )
    ax.set(
        xscale="log",
        yscale="log",
        xlabel=r"target error $a$",
        ylabel="records n",
        title=r"$\varepsilon = 1$, $\beta = 0.05$, k-th central moment at most 1",
    )
    ax.legend(fontsize=8)
    return fig


def plot_svt_losses() -> Figure:
    ms = [0, 1, 2, 3, 5, 7, 10, 15, 20, 30, 40, 60, 80]
    fig, ax = plt.subplots(figsize=(7, 4.2))
    for c, color in ((1, "tab:blue"), (2, "tab:orange"), (4, "tab:green")):
        ax.plot(
            ms,
            [svt_c_loss(m, c, 0.5, 0.5) for m in ms],
            "o-",
            color=color,
            markersize=4,
            label=rf"c = {c}, $\nu_i \sim$ Lap$(2c/\varepsilon_2)$",
        )
    ax.plot(
        ms,
        [svt_c_loss(m, 4, 0.5, 0.5, noise_grows=False) for m in ms],
        "s--",
        color="tab:green",
        markersize=4,
        label=r"c = 4, $\nu_i \sim$ Lap$(2/\varepsilon_2)$ (noise too small)",
    )
    ax.axhline(1.0, color="k", linestyle=":", label=r"$\varepsilon_1 + \varepsilon_2 = 1$")
    ax.axhline(0.5 + 4 * 0.5, color="gray", linestyle=":", label=r"$\varepsilon_1 + 4\varepsilon_2 = 2.5$")
    ax.set(
        xlabel=r"number m of $\bot$ answers before the c answers $\top$",
        ylabel="privacy loss of the output",
        title=r"Sparse vector with c answers above, $\varepsilon_1 = \varepsilon_2 = 0.5$",
    )
    ax.legend(fontsize=8, loc="center right", bbox_to_anchor=(1.0, 0.62))
    return fig


def plot_gumbel_top2(rng: np.random.Generator, scale: float = 2.0) -> Figure:
    weights = [math.exp(u / scale) for u in SIX_SCORES]
    pairs = ordered_pairs(len(SIX_SCORES))
    exact = [plackett_luce(weights, pair) for pair in pairs]
    freq = gumbel_top2_frequencies(rng, SIX_SCORES, scale, trials=200_000)
    order: list[int] = [int(i) for i in np.argsort(exact)[::-1]]
    x = np.arange(len(pairs))
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.bar(x, [exact[i] for i in order], color="0.75", label="two rounds of the exponential mechanism (exact)")
    ax.plot(
        x,
        [freq[pairs[i]] for i in order],
        "o",
        color="tab:red",
        markersize=4,
        label="top two of the Gumbel noisy scores (200,000 runs)",
    )
    ax.set_xticks(x)
    ax.set_xticklabels([f"{pairs[i][0] + 1},{pairs[i][1] + 1}" for i in order], rotation=90, fontsize=8)
    ax.set(
        xlabel="ordered pair (first, second)",
        ylabel="probability",
        title=f"Six candidates with scores {[int(u) for u in SIX_SCORES]}, noise scale {scale:g}",
    )
    ax.legend(fontsize=8)
    return fig


def plot_gumbel_race(rng: np.random.Generator, scale: float = 2.0) -> Figure:
    u = np.asarray(SIX_SCORES)
    rates = np.exp(u / scale)
    arrivals = np.exp(-(u + rng.gumbel(0.0, scale, size=len(u))) / scale)
    s = np.logspace(-3, 1, 400)
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    colors = [f"C{i}" for i in range(len(u))]
    rank = np.argsort(np.argsort(arrivals))
    for r, (rate, arrival) in enumerate(zip(rates, arrivals, strict=True)):
        style = "--" if r in (2, 5) else "-"  # the second of each tied pair of scores
        ax.plot(
            s, np.exp(-s * rate), style, color=colors[r], label=rf"candidate {r + 1}, score {u[r]:g}, rate {rate:.2f}"
        )
        ax.plot([arrival], [math.exp(-arrival * rate)], "o", color=colors[r], markersize=7)
        ax.annotate(
            f"{rank[r] + 1}",
            (arrival, math.exp(-arrival * rate)),
            textcoords="offset points",
            xytext=(5, 5),
            fontsize=9,
        )
    ax.set(
        xscale="log",
        xlabel=r"time $s$ (log scale)",
        ylabel=r"$\Pr[E_r > s] = e^{-s\lambda_r}$",
        title="Exponential race: survival curves and one draw of the arrival times",
    )
    ax.legend(fontsize=8, loc="lower left")
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
        "apx_mech_fourpoint_split.svg": plot_fourpoint_split(),
        "apx_mech_fourpoint_tradeoff.svg": plot_fourpoint_tradeoff(),
        "apx_mech_histogram_error.svg": plot_histogram_error(),
        "apx_mech_histogram_release.svg": plot_histogram_release(),
        "apx_mech_heavy_two_point.svg": plot_heavy_two_point(),
        "apx_mech_heavy_rates.svg": plot_heavy_rates(),
        "apx_mech_svt_losses.svg": plot_svt_losses(),
        "apx_mech_gumbel_top2.svg": plot_gumbel_top2(rng),
        "apx_mech_gumbel_race.svg": plot_gumbel_race(rng),
    }
    for name, fig in figures.items():
        fig.tight_layout()
        fig.savefig(out / name, metadata={"Date": None})  # no timestamp, so output is reproducible
        plt.close(fig)
        print(f"saved {out / name}")


if __name__ == "__main__":
    main()
