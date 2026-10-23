"""Check the amplification-by-shuffling and amplification-by-iteration chapter and draw its figures.

    uv run python code/part04-privacy-accounting/shuffling_iteration.py
    uv run python code/part04-privacy-accounting/shuffling_iteration.py --seed 1 --out /tmp/figures

Prints the numbers quoted in 07-shuffling-iteration.md (they match exactly with the default seed),
then saves the chapter's figures.

Conventions. Shuffling uses the replace-one relation: n users, and the two datasets differ in the
value of one user. Binary randomized response (RR) with local parameter ε0 reports the true bit with
probability θ = e^ε0 / (1 + e^ε0). Iteration uses one-dimensional noisy gradient steps
w_k = c·w_{k-1} + b_k + N(0, σ²), k = 1..T, from a fixed start w_0: c is the contraction factor, σ the
noise standard deviation per step (in parameter units) and s the amount by which the step of the run
on D differs from the step of the run on D' at a step that uses the differing record. The per-step
noise multiplier is σ/s. Logarithms are natural.
"""

import argparse
import math
from collections.abc import Callable, Sequence
from functools import cache
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
matplotlib.rcParams["svg.hashsalt"] = "shuffling_iteration"  # stable element ids, so reruns don't churn the SVGs
matplotlib.rcParams["svg.fonttype"] = "none"  # keep text as text: smaller, diffable files

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from numpy.typing import NDArray

# Figures are build output: _build/figures/<part>/, which the build copies into each book.
FIGURES = Path(__file__).resolve().parents[2] / "_build" / "figures" / Path(__file__).parent.name

FloatArray = NDArray[np.float64]

# The RDP orders shared by every chapter of the part.
ORDERS: list[float] = [1 + x / 10 for x in range(1, 100)] + [float(a) for a in range(12, 64)] + [128.0, 256.0, 512.0]

# The parameter grid of the shuffling comparison.
SHUFFLE_NS: tuple[int, ...] = (1_000, 10_000, 100_000)
SHUFFLE_EPS0: tuple[float, ...] = (1.0, 2.0, 4.0)
SHUFFLE_DELTA = 1e-6


# --- Toolkit: normal CDF, hockey-stick on log-probabilities, conversions ----------------------------


def normal_cdf(x: float) -> float:
    """Φ(x); erfc keeps full relative precision in the lower tail."""
    return 0.5 * math.erfc(-x / math.sqrt(2))


@cache
def log_factorials(n: int) -> FloatArray:
    """ln k! for k = 0..n, as a cumulative sum of logarithms."""
    return np.concatenate(([0.0], np.cumsum(np.log(np.arange(1, n + 1, dtype=np.float64)))))


def log_binomial_window(n: int, a: float, lo: int, hi: int) -> FloatArray:
    """ln Pr[Bin(n, a) = k] for k = lo..hi, in log space so nothing under- or overflows."""
    lf = log_factorials(1 << max(n, 1).bit_length())  # one table per power of two, reused
    k = np.arange(lo, hi + 1)
    return lf[n] - lf[k] - lf[n - k] + k * math.log(a) + (n - k) * math.log1p(-a)


def log_binomial_pmf(n: int, a: float) -> FloatArray:
    """ln Pr[Bin(n, a) = k] for k = 0..n."""
    return log_binomial_window(n, a, 0, n)


def hockey_stick_log(log_p: FloatArray, log_q: FloatArray, eps: float) -> float:
    """E_{e^ε}(P‖Q) = Σ (p - e^ε q)_+ = Σ p (1 - e^{ε + ln q - ln p})_+, from log-probabilities."""
    with np.errstate(invalid="ignore", over="ignore"):
        excess = -np.expm1(eps + log_q - log_p)  # 1 - e^{ε} q/p
    terms = np.where(excess > 0, np.exp(log_p) * excess, 0.0)
    return float(terms.sum())


class Profile:
    """δ(ε) = Σ (p - e^ε q)_+ for fixed P, Q, answered in O(log size) from sorted prefix sums.

    The terms with p > e^ε q are exactly those whose log-ratio exceeds ε; sort by log-ratio once,
    and δ(ε) is (P-mass) - e^ε (Q-mass) of the outcomes above ε.
    """

    def __init__(self, p: FloatArray, q: FloatArray, extra: float = 0.0) -> None:
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = np.log(p) - np.log(q)
        keep = p > 0
        order = np.argsort(-ratio[keep], kind="stable")
        self.neg_ratio: FloatArray = -ratio[keep][order]
        self.cum_p: FloatArray = np.cumsum(p[keep][order])
        self.cum_q: FloatArray = np.cumsum(q[keep][order])
        self.extra = extra  # mass dropped by a truncation, counted in full

    def __call__(self, eps: float) -> float:
        k = int(np.searchsorted(self.neg_ratio, -eps, side="left"))  # outcomes with ratio > ε
        if k == 0:
            return self.extra
        return max(float(self.cum_p[k - 1] - math.exp(eps) * self.cum_q[k - 1]), 0.0) + self.extra


def epsilon_for_delta(delta_of_eps: Callable[[float], float], delta: float, hi: float) -> float:
    """The smallest ε in [0, hi] with δ(ε) ≤ delta, by bisection (δ(ε) is non-increasing)."""
    if delta_of_eps(0.0) <= delta:
        return 0.0
    if delta_of_eps(hi) > delta:
        return math.inf
    lo = 0.0
    for _ in range(50):
        mid = (lo + hi) / 2
        if delta_of_eps(mid) > delta:
            lo = mid
        else:
            hi = mid
    return hi


def gdp_delta(eps: float, mu: float) -> float:
    """δ(ε) of μ-GDP: Φ(-ε/μ + μ/2) - e^ε Φ(-ε/μ - μ/2)."""
    if mu == 0:
        return 0.0
    tail = normal_cdf(-eps / mu - mu / 2)
    second = math.exp(eps + math.log(tail)) if tail > 0 else 0.0  # e^ε Φ(...) without overflow
    return normal_cdf(-eps / mu + mu / 2) - second


def gdp_epsilon(mu: float, delta: float) -> float:
    """The ε at which μ-GDP has δ(ε) = delta."""
    return epsilon_for_delta(lambda e: gdp_delta(e, mu), delta, 10 * mu * mu + 20 * mu + 10)


def rdp_to_epsilon(rdp: Callable[[float], float], delta: float) -> tuple[float, float]:
    """Mironov's conversion min_α ε(α) + ln(1/δ)/(α - 1) over ORDERS. Returns (ε, best α)."""
    best = min(ORDERS, key=lambda a: rdp(a) + math.log(1 / delta) / (a - 1))
    return rdp(best) + math.log(1 / delta) / (best - 1), best


# --- Section 33.3: shuffled binary randomized response, exactly ------------------------------------------


def rr_truth_prob(eps0: float) -> float:
    """θ = e^ε0 / (1 + e^ε0), the probability that ε0-RR reports the true bit."""
    return 1 / (1 + math.exp(-eps0))


def local_rr_delta(eps: float, eps0: float) -> float:
    """δ(ε) of one RR report: (θ - e^ε (1 - θ))_+. Zero for ε ≥ ε0."""
    theta = rr_truth_prob(eps0)
    return max(theta - math.exp(eps) * (1 - theta), 0.0)


@cache
def shuffled_rr_log_pmfs(eps0: float, n: int) -> tuple[FloatArray, FloatArray]:
    """ln-pmfs of the number Y of 1-reports, under D (you hold 0) and D' (you hold 1).

    The other n - 1 users hold 0, so their 1-reports S ~ Bin(n - 1, 1 - θ). Your own report adds
    Bern(1 - θ) under D and Bern(θ) under D':
        P(y) = θ·b(y) + (1 - θ)·b(y - 1),   Q(y) = (1 - θ)·b(y) + θ·b(y - 1),   b = pmf of S.
    """
    theta = rr_truth_prob(eps0)
    log_b = np.concatenate((log_binomial_pmf(n - 1, 1 - theta), [-np.inf]))  # b(y), y = 0..n
    log_b_prev = np.concatenate(([-np.inf], log_binomial_pmf(n - 1, 1 - theta)))  # b(y - 1)
    log_p = np.logaddexp(math.log(theta) + log_b, math.log1p(-theta) + log_b_prev)
    log_q = np.logaddexp(math.log1p(-theta) + log_b, math.log(theta) + log_b_prev)
    return log_p, log_q


def shuffled_rr_pmfs_general(eps0: float, n: int, ones: int) -> tuple[FloatArray, FloatArray]:
    """The same pair when `ones` of the other n - 1 users hold 1 (direct convolution; for checks)."""
    theta = rr_truth_prob(eps0)
    others = np.convolve(np.exp(log_binomial_pmf(ones, theta)), np.exp(log_binomial_pmf(n - 1 - ones, 1 - theta)))
    b = np.concatenate((others, [0.0]))
    b_prev = np.concatenate(([0.0], others))
    return theta * b + (1 - theta) * b_prev, (1 - theta) * b + theta * b_prev


def shuffled_rr_pair_epsilon(eps0: float, n: int, ones: int, delta: float) -> float:
    """The exact ε at δ of shuffled binary RR for the pair in which `ones` of the other n - 1 users hold 1."""
    p, q = shuffled_rr_pmfs_general(eps0, n, ones)
    forward, backward = Profile(p, q), Profile(q, p)
    return epsilon_for_delta(lambda e: max(forward(e), backward(e)), delta, eps0)


@cache
def shuffled_rr_profiles(eps0: float, n: int) -> tuple[Profile, Profile]:
    log_p, log_q = shuffled_rr_log_pmfs(eps0, n)
    p, q = np.exp(log_p), np.exp(log_q)
    return Profile(p, q), Profile(q, p)


def shuffled_rr_delta(eps: float, eps0: float, n: int) -> float:
    """Exact δ(ε) of n shuffled ε0-RR reports, for the pair in which the other users all hold 0.

    The analyst sees only the multiset of bits, that is the count of 1s. Both directions count, since
    (ε, δ)-DP must hold for (D, D') and for (D', D).
    """
    forward, backward = shuffled_rr_profiles(eps0, n)
    return max(forward(eps), backward(eps))


def shuffled_rr_epsilon(eps0: float, n: int, delta: float) -> float:
    """The smallest ε with shuffled_rr_delta(ε) ≤ δ: the exact ε of shuffled binary RR."""
    return epsilon_for_delta(lambda e: shuffled_rr_delta(e, eps0, n), delta, eps0)


def shuffled_rr_gaussian_mu(eps0: float, n: int) -> float:
    """CLT approximation: the count shifts by 2θ - 1 against a spread √(nθ(1-θ)), so μ ≈ 2 sinh(ε0/2)/√n."""
    return 2 * math.sinh(eps0 / 2) / math.sqrt(n)


# --- Section 33.4: the clone bound -----------------------------------------------------------------------


def clone_condition(eps0: float, n: int, delta: float) -> bool:
    """Feldman, McMillan & Talwar (2021), Theorem 3.1 needs ε0 ≤ ln(n / (16 ln(2/δ)))."""
    return eps0 <= math.log(n / (16 * math.log(2 / delta)))


def clones_epsilon(eps0: float, n: int, delta: float) -> float:
    """Theorem 3.1 of Feldman, McMillan & Talwar (2021), equation (1):

    ε ≤ ln(1 + (e^ε0 - 1)/(e^ε0 + 1) · (8 √(e^ε0 ln(4/δ)) / √n + 8 e^ε0 / n)).
    Returns inf when the theorem's condition on ε0 fails.
    """
    if not clone_condition(eps0, n, delta):
        return math.inf
    e0 = math.exp(eps0)
    inner = 8 * math.sqrt(e0 * math.log(4 / delta)) / math.sqrt(n) + 8 * e0 / n
    return math.log1p((e0 - 1) / (e0 + 1) * inner)


def clone_split(
    r_x: Sequence[float], r_x0: Sequence[float], r_x1: Sequence[float], eps0: float
) -> tuple[list[float], list[float]]:
    """R(x) = e^{-ε0}/2 · R(x0) + e^{-ε0}/2 · R(x1) + (1 - e^{-ε0}) · LO(x).

    Returns (the clone part, the leftover distribution LO(x)).
    """
    w = math.exp(-eps0) / 2
    clones = [w * (a + b) for a, b in zip(r_x0, r_x1, strict=True)]
    rest = 1 - math.exp(-eps0)
    return clones, [(r - c) / rest for r, c in zip(r_x, clones, strict=True)]


@cache
def clone_pair(eps0: float, n: int) -> tuple[FloatArray, FloatArray, FloatArray, float]:
    """The pair of Theorem 3.2 (Feldman, McMillan & Talwar 2021), flattened over (c, k).

    C ~ Bin(n - 1, e^{-ε0}), A ~ Bin(C, 1/2), Δ ~ Bern(e^ε0/(e^ε0 + 1)); P = (A + Δ, C - A + 1 - Δ),
    Q = (A + 1 - Δ, C - A + Δ). Given C = c both are fixed by their first coordinate k, so
    P(c, k) = Pr[C = c] (pd·b_c(k - 1) + (1 - pd)·b_c(k)) and Q swaps pd and 1 - pd, b_c = Bin(c, 1/2).
    Returns (Pr[C = c] per entry, P(k | c), Q(k | c), probability mass of C outside the window).
    """
    pd = rr_truth_prob(eps0)
    clone = math.exp(-eps0)
    mean, sd = (n - 1) * clone, math.sqrt((n - 1) * clone * (1 - clone))
    c_lo, c_hi = max(0, int(mean - 12 * sd - 5)), min(n - 1, int(mean + 12 * sd + 5))
    c_log = log_binomial_window(n - 1, clone, c_lo, c_hi)
    dropped = max(0.0, 1.0 - float(np.exp(c_log).sum()))
    weights: list[FloatArray] = []
    p_rows: list[FloatArray] = []
    q_rows: list[FloatArray] = []
    for c, lw in zip(range(c_lo, c_hi + 1), c_log, strict=True):
        if c == 0:
            b = np.array([1.0])
        else:
            half = int(8 * math.sqrt(c) + 5)
            k_lo, k_hi = max(0, c // 2 - half), min(c, c // 2 + half)
            b = np.zeros(c + 1)
            b[k_lo : k_hi + 1] = np.exp(log_binomial_window(c, 0.5, k_lo, k_hi))
        b_k = np.concatenate((b, [0.0]))
        b_km1 = np.concatenate(([0.0], b))
        keep = (b_k > 0) | (b_km1 > 0)
        weights.append(np.full(int(keep.sum()), math.exp(float(lw))))
        p_rows.append((pd * b_km1 + (1 - pd) * b_k)[keep])
        q_rows.append(((1 - pd) * b_km1 + pd * b_k)[keep])
    return np.concatenate(weights), np.concatenate(p_rows), np.concatenate(q_rows), dropped


@cache
def clone_profile(eps0: float, n: int) -> Profile:
    w, p, q, dropped = clone_pair(eps0, n)
    return Profile(w * p, w * q, dropped)


def clones_numeric_delta(eps: float, eps0: float, n: int) -> float:
    """δ(ε) of the Theorem 3.2 pair (symmetric, so one direction suffices), plus the truncated mass."""
    return clone_profile(eps0, n)(eps)


def clones_numeric_epsilon(eps0: float, n: int, delta: float) -> float:
    """The smallest ε that Theorem 3.2 certifies for every ε0-DP randomizer, by computing its δ(ε)."""
    return epsilon_for_delta(lambda e: clones_numeric_delta(e, eps0, n), delta, eps0)


def erlingsson_epsilon(eps0: float, n: int, delta: float) -> float:
    """Erlingsson et al. (2019), Theorem 7 (arXiv version): ε1 √(2n ln(1/δ)) + n ε1 (e^ε1 - 1),
    with ε1 = 2 e^{2ε0} (e^ε0 - 1) / n."""
    eps1 = 2 * math.exp(2 * eps0) * math.expm1(eps0) / n
    if eps1 > 50:
        return math.inf
    return eps1 * math.sqrt(2 * n * math.log(1 / delta)) + n * eps1 * math.expm1(eps1)


# --- Section 33.5: amplification by iteration ------------------------------------------------------------


def gaussian_rdp(alpha: float, shift: float, sigma: float) -> float:
    """D_α(N(0, σ²) ‖ N(shift, σ²)) = α shift² / (2σ²): the price R_α of hiding a shift with noise σ."""
    return alpha * shift * shift / (2 * sigma * sigma)


def even_shift_cost(alpha: float, shift: float, sigma: float, steps: int) -> float:
    """Shift-reduction cost when a shift z is paid off in `steps` equal parts a = z/steps."""
    return steps * gaussian_rdp(alpha, shift / steps, sigma)


def fmtt_rdp(alpha: float, noise_multiplier: float, steps_left: int) -> float:
    """Feldman, Mironov, Talwar & Thakurta (2018), Theorem 23, with σ̃ = σ/(2L_f) the noise multiplier:
    the record used at step t of n is (α, α / (2 σ̃² (n - t + 1)))-RDP. `steps_left` = n - t + 1."""
    return alpha / (2 * noise_multiplier**2 * steps_left)


def fmtt_stop_rdp(alpha: float, noise_multiplier: float, n: int) -> float:
    """Theorem 26 of the same paper (random stopping time), 4αL_f² ln n / (n σ²) = α ln n / (n σ̃²).
    Valid only when σ ≥ L_f √(2(α - 1)α), that is σ̃ ≥ √(2(α - 1)α) / 2."""
    if noise_multiplier < math.sqrt(2 * (alpha - 1) * alpha) / 2:
        return math.inf
    return alpha * math.log(n) / (n * noise_multiplier**2)


def last_iterate_mu(contraction: float, steps: int, sigma: float, shifts: Sequence[float]) -> float:
    """Exact GDP μ of the last iterate of w_k = c w_{k-1} + b_k + N(0, σ²), k = 1..T, from a fixed w_0.

    shifts[k - 1] is the difference between the two runs' offsets b_k. The last iterate is Gaussian on
    both datasets with the same variance σ² Σ_j c^{2j} and means that differ by Σ_k c^{T-k} shifts[k - 1],
    so the pair is a Gaussian shift pair with μ = |mean difference| / standard deviation.
    """
    c = contraction
    gap = sum(c ** (steps - k) * s for k, s in zip(range(1, steps + 1), shifts, strict=True))
    variance = sigma * sigma * sum(c ** (2 * j) for j in range(steps))
    return abs(gap) / math.sqrt(variance)


def noisy_gd_one_pass_mu(contraction: float, steps: int, sigma: float, shift: float, t: int) -> float:
    """μ of the last iterate when the differing record is used once, at step t (one pass, no sampling)."""
    return last_iterate_mu(contraction, steps, sigma, [shift if k == t else 0.0 for k in range(1, steps + 1)])


def reset_one_pass_mu(steps: int, sigma: float, shift: float, t: int) -> float:
    """μ when step t first maps every point to one point (a contraction), then translates (c = 1).

    The noise added before step t is erased, so only the T - t + 1 later draws hide the shift:
    μ = shift / (σ √(T - t + 1)). This is the case in which Theorem 23 is exact."""
    return shift / (sigma * math.sqrt(steps - t + 1))


# --- Sections 33.6 and 33.7: hidden state, plateau against composition ---------------------------------------


def noisy_gd_last_iterate_mu(contraction: float, steps: int, sigma: float, shift: float, t: int) -> float:
    """Exact μ of the last iterate of 1-D noisy GD when the differing record is in every step from t on.

    The model is w_k = c·w_{k-1} + b_k + N(0, σ²), k = 1..steps, from a fixed w_0, with c = contraction
    (c = 1 - ηλ for a quadratic loss with curvature λ and step size η), σ the noise standard deviation
    per step and `shift` the difference between the steps on D and D' at every step k ≥ t (for full-batch
    GD with the replace-one relation, shift = η·(gradient sensitivity)). t = 1 is full-batch GD.
    Releasing all iterates would cost composed_mu(steps - t + 1, sigma, shift) instead.
    """
    return last_iterate_mu(contraction, steps, sigma, [shift if k >= t else 0.0 for k in range(1, steps + 1)])


def composed_mu(steps: int, sigma: float, shift: float) -> float:
    """μ of releasing every iterate: `steps` Gaussian mechanisms with μ_step = shift/σ compose to √steps μ_step."""
    return math.sqrt(steps) * shift / sigma


def hidden_state_limit_mu(contraction: float, sigma: float, shift: float) -> float:
    """lim_{T→∞} of the full-batch last-iterate μ: (s/σ) √((1 + c)/(1 - c)) for c < 1."""
    return shift / sigma * math.sqrt((1 + contraction) / (1 - contraction))


def plateau_rdp_bound(alpha: float, steps: int, shift: float, sigma: float, diameter: float) -> float:
    """RDP of the last iterate of full-batch projected noisy GD on a convex set of diameter d_K.

    From Proposition 3.2 of Altschuler & Talwar (2022) with an even shift schedule over the last m
    steps: α (d_K + m s)² / (2 m σ²), minimized over integers 1 ≤ m ≤ T, and the plain bound α T s² / (2σ²)
    (m = T with no diameter term). Here s is the per-step shift and σ the per-step noise std.
    """
    best = steps * shift * shift
    m_star = diameter / shift
    for m in {1, steps, max(1, min(steps, math.floor(m_star))), max(1, min(steps, math.ceil(m_star)))}:
        best = min(best, (diameter + m * shift) ** 2 / m)
    return alpha * best / (2 * sigma * sigma)


def _normal_cdf_array(x: FloatArray) -> FloatArray:
    flat = [normal_cdf(float(v)) for v in x.ravel()]
    return np.array(flat, dtype=np.float64).reshape(x.shape)


def projected_walk_kernel(drift: float, sigma: float, diameter: float, cells: int) -> FloatArray:
    """Transition matrix of w → Π_[0, d_K](w + drift + N(0, σ²)) on a grid of cell centres.

    Row i is the distribution of the next cell from the centre of cell i. Mass that leaves [0, D]
    is projected onto the boundary, so it lands in the first or last cell.
    """
    h = diameter / cells
    centres = (np.arange(cells) + 0.5) * h
    edges = np.arange(cells + 1) * h
    z = (edges[None, :] - centres[:, None] - drift) / sigma
    cdf = _normal_cdf_array(z)
    cdf[:, 0], cdf[:, -1] = 0.0, 1.0  # projection: everything below 0 is in cell 0, above d_K in the last
    return np.diff(cdf, axis=1)


def projected_walk_renyi(
    alpha: float, steps: Sequence[int], shift: float, sigma: float, diameter: float, cells: int = 1000
) -> list[float]:
    """D_α between the last iterates of two projected random walks on [0, d_K], started at d_K/2.

    Run on D the walk drifts by +s/2 per step, on D' by -s/2 (a linear loss whose gradient differs by
    s/η between the two datasets): the lower-bound construction of Altschuler & Talwar (2022).
    This is one pair of datasets and one start, so it is a lower bound on the worst-case RDP.
    Computed on a grid of `cells` cells, so the values are a discretization, not exact.
    """
    k_p = projected_walk_kernel(shift / 2, sigma, diameter, cells)
    k_q = projected_walk_kernel(-shift / 2, sigma, diameter, cells)
    p = np.zeros(cells)
    p[cells // 2] = 1.0
    q = p.copy()
    out: list[float] = []
    targets = sorted(set(steps))
    done = 0
    for target in targets:
        p = p @ np.linalg.matrix_power(k_p, target - done)  # repeated squaring
        q = q @ np.linalg.matrix_power(k_q, target - done)
        done = target
        mask = p > 0
        out.append(float(np.log(np.sum(p[mask] ** alpha * q[mask] ** (1 - alpha))) / (alpha - 1)))
    by_step = dict(zip(targets, out, strict=True))
    return [by_step[s] for s in steps]


def chourasia_limit_rdp(alpha: float, contraction: float, sigma: float, shift: float) -> float:
    """Chourasia, Ye & Shokri (2021), Corollary 1, as K → ∞: α S_g² / (λ σ_C² n²).

    In this file's units (step θ ← (1 - ηλ)θ - (η/n) g + √(2η) σ_C Z): c = 1 - ηλ, s = η S_g / n,
    σ = √(2η) σ_C, so the limit is 2 α s² / ((1 - c) σ²).
    """
    return 2 * alpha * shift * shift / ((1 - contraction) * sigma * sigma)


def simulate_last_iterate(
    rng: np.random.Generator, contraction: float, steps: int, sigma: float, offsets: Sequence[float], runs: int
) -> FloatArray:
    """Monte Carlo draws of the last iterate of w_k = c w_{k-1} + offsets[k-1] + N(0, σ²), w_0 = 0."""
    w = np.zeros(runs)
    for b in offsets[:steps]:
        w = contraction * w + b + sigma * rng.standard_normal(runs)
    return w


# --- The numbers in the text -----------------------------------------------------------------------


def print_table(rng: np.random.Generator) -> None:
    print("[Part 1: n = 3 users, ε0 = ln 3, others hold 0]")
    eps0 = math.log(3)
    log_p, log_q = shuffled_rr_log_pmfs(eps0, 3)
    p, q = np.exp(log_p), np.exp(log_q)
    others = np.exp(log_binomial_pmf(2, 1 - rr_truth_prob(eps0)))
    print("  others' count b x 16 = " + "  ".join(f"{16 * v:.0f}" for v in others))
    print(
        "  P(count) x 64 = "
        + "  ".join(f"{64 * v:.0f}" for v in p)
        + "   Q(count) x 64 = "
        + "  ".join(f"{64 * v:.0f}" for v in q)
    )
    print("  ratios P/Q = " + "  ".join(f"{a / b:.4f}" for a, b in zip(p, q, strict=True)))
    for eps in (0.5, 1.0):
        print(
            f"  eps={eps:g}: shuffled delta={shuffled_rr_delta(eps, eps0, 3):.4f}  "
            f"one local report delta={local_rr_delta(eps, eps0):.4f}"
        )
    counts = rng.binomial(1, 0.25, size=(200_000, 2)).sum(axis=1) + rng.binomial(1, 0.25, size=200_000)
    freq = np.bincount(counts, minlength=4) / 200_000
    print(
        "  simulated shuffle under D, 200000 runs: "
        + "  ".join(f"{v:.4f}" for v in freq)
        + "   exact: "
        + "  ".join(f"{v:.4f}" for v in p)
        + f"   largest gap: {float(np.abs(freq - p).max()):.4f}"
    )
    for e0 in (2.0, 4.0):
        print(
            f"  eps0={e0:g}: truth probability theta={rr_truth_prob(e0):.4f}  "
            f"expected 1-reports among 999 others holding 0: {999 * (1 - rr_truth_prob(e0)):.1f}"
        )
    print(f"  eps0=4, n=1000: expected clones (n - 1) e^-eps0={999 * math.exp(-4.0):.1f}")
    theta4 = rr_truth_prob(4.0)
    print(
        f"  eps0=4: added variance per user of the debiased proportion theta(1-theta)/(2 theta-1)^2="
        f"{theta4 * (1 - theta4) / (2 * theta4 - 1) ** 2:.4f}"
    )

    print("[Part 1: which dataset pair is worst? delta = 1e-6, eps over the number m of others holding 1]")
    # Swapping 0 and 1 for everyone maps m to n - 1 - m and swaps D and D', so m <= (n - 1)/2 covers all pairs.
    for n_users, ms in ((1000, range(500)), (10_000, range(31)), (100_000, range(31))):
        scope = "all m" if n_users == 1000 else f"m <= {ms[-1]}"
        for e0 in SHUFFLE_EPS0:
            values = [shuffled_rr_pair_epsilon(e0, n_users, m, SHUFFLE_DELTA) for m in ms]
            worst = int(np.argmax(values))
            tail = f"  m={ms[-1]}: {values[-1]:.4f}" if n_users == 1000 else ""
            print(
                f"  n={n_users} eps0={e0:g}: m=0: {values[0]:.6f}  largest over {scope}: {values[worst]:.6f} "
                f"at m={ms[worst]}  excess {values[worst] - values[0]:.1e}" + tail
            )

    print("[Part 2: clone split of 2-ary RR, eps0 = ln 3, x = 0 against x0 = 0, x1 = 1]")
    theta = rr_truth_prob(eps0)
    r0, r1 = [theta, 1 - theta], [1 - theta, theta]
    clones, left = clone_split(r0, r0, r1, eps0)
    print(
        f"  R(0)={[round(v, 4) for v in r0]}  clone part={[round(v, 4) for v in clones]}  "
        f"LO(0)={[round(v, 4) for v in left]}"
    )

    print("[Part 2: epsilon at delta = 1e-6]")
    for n in SHUFFLE_NS:
        print(f"  n={n}: eps0 must be <= {math.log(n / (16 * math.log(2 / SHUFFLE_DELTA))):.3f}")
        for e0 in SHUFFLE_EPS0:
            exact = shuffled_rr_epsilon(e0, n, SHUFFLE_DELTA)
            closed = clones_epsilon(e0, n, SHUFFLE_DELTA)
            numeric = clones_numeric_epsilon(e0, n, SHUFFLE_DELTA)
            erl = erlingsson_epsilon(e0, n, SHUFFLE_DELTA)
            gauss = gdp_epsilon(shuffled_rr_gaussian_mu(e0, n), SHUFFLE_DELTA)
            closed_text = f"{closed:.4f}" if math.isfinite(closed) else "n/a"
            erl_text = f"{erl:.4f}" if erl < e0 else "none (>= eps0)"
            print(
                f"  n={n} eps0={e0:g}: exact RR={exact:.4f}  CLT approx={gauss:.4f}  "
                f"clones numeric={numeric:.4f}  clones closed form={closed_text}  "
                f"Erlingsson={erl_text}  sqrt(n)*exact={math.sqrt(n) * exact:.3f}  "
                f"numeric/exact={numeric / exact:.2f}"
                + (
                    f"  closed/exact={closed / exact:.2f}  closed/numeric={closed / numeric:.2f}"
                    if math.isfinite(closed)
                    else ""
                )
            )

    figure_ns = [int(round(10 ** (k / 4))) for k in range(10, 21)]
    print(
        f"  Erlingsson at eps0=4, smallest over the figure's n={figure_ns[0]}..{figure_ns[-1]}: "
        f"{min(erlingsson_epsilon(4.0, m, SHUFFLE_DELTA) for m in figure_ns):.3g}"
    )
    print("[Part 2: clone-count noise against binary RR noise, 2 / (1 + e^-eps0)]")
    print("  " + "  ".join(f"eps0={e0:g}: {2 / (1 + math.exp(-e0)):.2f}" for e0 in SHUFFLE_EPS0))

    print("[Part 3: shift reduction for a pure translation, alpha = 2, shift z = 1, sigma = 1]")
    for steps in (1, 2, 4, 10):
        exact = gaussian_rdp(2.0, 1.0, math.sqrt(steps))
        print(
            f"  T={steps}: even schedule cost={even_shift_cost(2.0, 1.0, 1.0, steps):.4f}  "
            f"exact D_2(N(0,T)||N(1,T))={exact:.4f}"
        )
    print(f"  uneven schedule a=(0.8, 0.2): cost={gaussian_rdp(2.0, 0.8, 1.0) + gaussian_rdp(2.0, 0.2, 1.0):.4f}")

    print("[Part 3: one pass over n = 10000 records, noise multiplier 1, delta = 1e-5]")
    n = 10_000
    for t in (1, 5_000, 9_901, 10_000):
        eps, alpha = rdp_to_epsilon(lambda a, t=t: fmtt_rdp(a, 1.0, n - t + 1), 1e-5)
        print(
            f"  t={t}: RDP eps(2)={fmtt_rdp(2.0, 1.0, n - t + 1):.6f}  (eps, delta)=({eps:.4f}, 1e-5) at alpha={alpha:g}"
        )
    eps_all, alpha_all = rdp_to_epsilon(lambda a: a / 2, 1e-5)
    print(f"  released all iterates (one Gaussian step per record): eps={eps_all:.4f} at alpha={alpha_all:g}")
    print(f"  one Gaussian mechanism, mu = 1, exact: eps={gdp_epsilon(1.0, 1e-5):.4f}")
    _, alpha_first = rdp_to_epsilon(lambda a: fmtt_rdp(a, 1.0, n), 1e-5)
    print(
        f"  t=1 at alpha={alpha_first:g}: RDP term={fmtt_rdp(alpha_first, 1.0, n):.4f}  "
        f"conversion term={math.log(1e5) / (alpha_first - 1):.4f}"
    )
    print(f"  random stopping (Thm 26), alpha=2, multiplier 1: {fmtt_stop_rdp(2.0, 1.0, n):.6f}")

    print("[Part 3: 1-D exact one pass, T = 100 steps, s/sigma = 1]")
    for t in (1, 50, 100):
        print(
            f"  t={t}: bound mu={1 / math.sqrt(100 - t + 1):.4f}  reset mu={reset_one_pass_mu(100, 1.0, 1.0, t):.4f}  "
            f"c=1 mu={noisy_gd_one_pass_mu(1.0, 100, 1.0, 1.0, t):.4f}  "
            f"c=0.95 mu={noisy_gd_one_pass_mu(0.95, 100, 1.0, 1.0, t):.4f}"
        )

    print("[Part 4: bounded domain, full batch, d_K = 1, s = 0.01, sigma = 0.1, alpha = 2]")
    for steps in (10, 100, 400, 1_000, 2_000):
        comp = gaussian_rdp(2.0, 0.01, 0.1) * steps
        bound = plateau_rdp_bound(2.0, steps, 0.01, 0.1, 1.0)
        print(f"  T={steps}: composition={comp:.4f}  plateau bound={bound:.4f}")
    walk_steps = [10, 100, 400, 1_000, 2_000]
    fine = projected_walk_renyi(2.0, walk_steps, 0.01, 0.1, 1.0, cells=1000)
    coarse = projected_walk_renyi(2.0, [2_000], 0.01, 0.1, 1.0, cells=500)
    print(
        "  projected walk D_2 (1000 cells): "
        + "  ".join(f"T={s}: {v:.4f}" for s, v in zip(walk_steps, fine, strict=True))
    )
    print(
        f"  projected walk D_2 at T=2000 with 500 cells: {coarse[0]:.4f}  "
        f"bound/walk at T=2000: {plateau_rdp_bound(2.0, 2_000, 0.01, 0.1, 1.0) / fine[-1]:.2f}"
    )

    print("[Part 5: 1-D full-batch noisy GD, s/sigma = 1 per step]")
    for c in (1.0, 0.99, 0.9):
        row = "  ".join(f"T={k}: {noisy_gd_last_iterate_mu(c, k, 1.0, 1.0, 1):.4f}" for k in (1, 10, 100, 1_000))
        print(f"  c={c:g}: {row}")
    for c in (0.99, 0.9):
        print(f"  c={c:g}: limit mu={hidden_state_limit_mu(c, 1.0, 1.0):.4f}")
    print("  composition: " + "  ".join(f"T={k}: {composed_mu(k, 1.0, 1.0):.4f}" for k in (1, 10, 100, 1_000)))
    k = 10_000
    mu_comp, mu_hidden = composed_mu(k, 10.0, 1.0), noisy_gd_last_iterate_mu(0.99, k, 10.0, 1.0, 1)
    print(
        f"  k={k}, noise multiplier 10: composition mu={mu_comp:.4f} eps={gdp_epsilon(mu_comp, 1e-5):.4f}; "
        f"last iterate c=0.99 mu={mu_hidden:.4f} eps={gdp_epsilon(mu_hidden, 1e-5):.4f} (delta = 1e-5)"
    )
    exact_rdp = 2.0 * hidden_state_limit_mu(0.99, 1.0, 1.0) ** 2 / 2
    print(
        f"  alpha=2, c=0.99: exact limit RDP={exact_rdp:.4f}  "
        f"Chourasia et al. limit={chourasia_limit_rdp(2.0, 0.99, 1.0, 1.0):.4f}  ratio={exact_rdp / chourasia_limit_rdp(2.0, 0.99, 1.0, 1.0):.4f}"
    )
    draws_d = simulate_last_iterate(rng, 0.9, 50, 1.0, [1.0] * 50, 200_000)
    draws_dp = simulate_last_iterate(rng, 0.9, 50, 1.0, [0.0] * 50, 200_000)
    gap = float(draws_d.mean() - draws_dp.mean())
    sd = float(np.sqrt(0.5 * (draws_d.var() + draws_dp.var())))
    print(
        f"  Monte Carlo c=0.9, T=50, 200000 runs: mu={gap / sd:.4f}  exact={noisy_gd_last_iterate_mu(0.9, 50, 1.0, 1.0, 1):.4f}"
    )
    print(
        f"  record joining at t=41 of T=50, c=0.9: mu={noisy_gd_last_iterate_mu(0.9, 50, 1.0, 1.0, 41):.4f}  "
        f"composition of its 10 steps={composed_mu(10, 1.0, 1.0):.4f}"
    )


# --- Figures --------------------------------------------------------------------------------------


def plot_shuffled_counts(eps0: float = 2.0) -> Figure:
    fig, (left, right) = plt.subplots(1, 2, figsize=(11, 4))
    n = 30
    log_p, log_q = shuffled_rr_log_pmfs(eps0, n)
    ys = np.arange(n + 1)
    left.bar(ys - 0.2, np.exp(log_p), width=0.4, color="C0", label="D: you hold 0")
    left.bar(ys + 0.2, np.exp(log_q), width=0.4, color="C1", label="D': you hold 1")
    left.set(
        xlabel="number of 1-reports the analyst sees",
        ylabel="probability",
        xlim=(-0.5, 16.5),
        title=rf"n = {n} shuffled RR reports, $\varepsilon_0$ = {eps0:g}, others hold 0",
    )
    left.legend()
    eps_grid = [i / 100 for i in range(201)]
    right.plot(eps_grid, [local_rr_delta(e, eps0) for e in eps_grid], "k", label="one local report (n = 1)")
    for i, m in enumerate((10, 100, 1000)):
        right.plot(
            eps_grid, [shuffled_rr_delta(e, eps0, m) for e in eps_grid], color=f"C{i}", label=f"shuffled, n = {m}"
        )
    right.set(
        yscale="log",
        ylim=(1e-8, 1),
        xlabel=r"$\varepsilon$",
        ylabel=r"$\delta(\varepsilon)$",
        title=r"Exact $\delta(\varepsilon)$ of shuffled binary RR, $\varepsilon_0$ = " + f"{eps0:g}",
    )
    right.legend(fontsize=9)
    return fig


def plot_shuffle_bounds(delta: float = SHUFFLE_DELTA) -> Figure:
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), sharey=True)
    ns = [int(round(10 ** (k / 4))) for k in range(10, 21)]  # the same grid as print_table's Erlingsson line
    for ax, eps0 in zip(axes, SHUFFLE_EPS0):
        closed = [clones_epsilon(eps0, n, delta) for n in ns]
        ax.loglog(
            [n for n, v in zip(ns, closed) if math.isfinite(v)],
            [v for v in closed if math.isfinite(v)],
            "C0",
            label="clones, closed form (Thm 3.1)",
        )
        ax.loglog(ns, [clones_numeric_epsilon(eps0, n, delta) for n in ns], "C0--", label="clones, numeric (Thm 3.2)")
        erl = [erlingsson_epsilon(eps0, n, delta) for n in ns]
        ax.loglog(
            [n for n, v in zip(ns, erl) if v < 50], [v for v in erl if v < 50], "C2:", label="Erlingsson et al. (2019)"
        )
        ax.loglog(ns, [shuffled_rr_epsilon(eps0, n, delta) for n in ns], "k", linewidth=2, label="exact, binary RR")
        ax.axhline(eps0, color="0.6", linewidth=1)
        ax.text(ns[0], eps0 * 1.15, r"local $\varepsilon_0$", fontsize=8, color="0.4")
        ax.set(xlabel="users n", ylim=(5e-3, 50), title=rf"$\varepsilon_0$ = {eps0:g}, $\delta$ = $10^{{-6}}$")
    axes[0].set_ylabel(r"central $\varepsilon$")
    axes[0].legend(fontsize=8, loc="lower left")
    return fig


def plot_iteration_positions(steps: int = 100) -> Figure:
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    ts = list(range(1, steps + 1))
    ax.plot(ts, [1.0] * steps, "k:", label="all iterates released: s/σ")
    ax.plot(
        ts,
        [1 / math.sqrt(steps - t + 1) for t in ts],
        "C3",
        linewidth=2,
        label="Theorem 23 bound (as μ), = exact after a reset",
    )
    ax.plot(ts, [noisy_gd_one_pass_mu(1.0, steps, 1.0, 1.0, t) for t in ts], "C0", label="exact, c = 1")
    ax.plot(ts, [noisy_gd_one_pass_mu(0.95, steps, 1.0, 1.0, t) for t in ts], "C1", label="exact, c = 0.95")
    ax.set(
        xlabel="step t at which your record is used",
        ylabel=r"$\mu$ of the last iterate",
        ylim=(0, 1.05),
        title=f"One pass, T = {steps} steps, s/σ = 1: early records are better hidden",
    )
    ax.legend(fontsize=9, loc="upper left")
    return fig


def plot_shift_schedule(shift: float = 1.0, steps: int = 4) -> Figure:
    """Two coupled runs that start a distance z apart; noise lets each step pay to shrink the shift."""
    fig, (left, right) = plt.subplots(1, 2, figsize=(11, 4))
    ks = list(range(steps + 1))
    remaining = [shift - k * shift / steps for k in ks]
    left.step(ks, remaining, where="post", color="C0", linewidth=2, label="shift still to hide, $z_k$")
    for k in range(steps):
        left.annotate(
            "",
            xy=(k + 1, remaining[k + 1]),
            xytext=(k + 1, remaining[k]),
            arrowprops={"arrowstyle": "->", "color": "C3"},
        )
    left.text(
        steps / 2,
        shift * 1.04,
        r"each red arrow: pay $\alpha a^2/(2\sigma^2)$ to cut the shift by $a = z/T$",
        ha="center",
        fontsize=9,
        color="C3",
    )
    left.set(
        xlabel="noisy step k",
        ylabel="shift",
        ylim=(-0.05, 1.15 * shift),
        title=r"Even schedule: total $T\cdot\alpha (z/T)^2/(2\sigma^2) = \alpha z^2/(2T\sigma^2)$",
    )
    left.legend(loc="lower left")
    alphas_t = list(range(1, 21))
    right.plot(alphas_t, [even_shift_cost(2.0, shift, 1.0, t) for t in alphas_t], "C3o", label="even shift schedule")
    right.plot(
        alphas_t,
        [gaussian_rdp(2.0, shift, math.sqrt(t)) for t in alphas_t],
        "k-",
        label=r"exact $D_2(N(0, T)\,\|\,N(1, T))$",
    )
    right.plot(alphas_t, [gaussian_rdp(2.0, shift, 1.0)] * len(alphas_t), "k:", label="pay the whole shift at once")
    right.set(
        xlabel="noisy steps T after the shift",
        ylabel=r"$D_2$",
        ylim=(0, 1.1),
        title=r"z = 1, $\sigma$ = 1, $\alpha$ = 2: the schedule is exact for translations",
    )
    right.legend(fontsize=9)
    return fig


def plot_hidden_state() -> Figure:
    fig, (left, right) = plt.subplots(1, 2, figsize=(12, 4.2))
    ks = sorted({int(round(10 ** (j / 10))) for j in range(41)})
    left.loglog(
        ks, [composed_mu(k, 1.0, 1.0) for k in ks], "k", linewidth=2, label="all iterates released (composition)"
    )
    for i, c in enumerate((1.0, 0.99, 0.9)):
        style = "--" if c == 1.0 else "-"
        left.loglog(
            ks,
            [noisy_gd_last_iterate_mu(c, k, 1.0, 1.0, 1) for k in ks],
            style,
            color=f"C{i}",
            label=f"last iterate only, c = {c:g}",
        )
    left.set(
        xlabel="steps T",
        ylabel=r"$\mu$",
        title=r"$\theta \leftarrow c\,\theta + b + \mathcal{N}(0, \sigma^2)$, full batch, s/σ = 1: exact μ",
    )
    left.legend(fontsize=9)
    steps = sorted({int(round(10 ** (j / 8))) for j in range(27)})
    walk = projected_walk_renyi(2.0, steps, 0.01, 0.1, 1.0, cells=1000)
    right.loglog(steps, [gaussian_rdp(2.0, 0.01, 0.1) * k for k in steps], "k", linewidth=2, label="composition")
    right.loglog(steps, [plateau_rdp_bound(2.0, k, 0.01, 0.1, 1.0) for k in steps], "C3", label="bounded-domain bound")
    right.loglog(steps, walk, "C0o-", markersize=3, label="projected walk (numerical)")
    right.set(
        xlabel="steps T", ylabel=r"$D_2$ of the last iterate", title="Linear loss on [0, 1], s = 0.01, σ = 0.1 per step"
    )
    right.legend(fontsize=9)
    return fig


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=FIGURES, help="directory for the figures")
    args = parser.parse_args()
    seed: int = args.seed
    out: Path = args.out

    print_table(np.random.default_rng(seed))

    out.mkdir(parents=True, exist_ok=True)  # nothing below is random; seed + 1 is reserved for figures
    figures = {
        "shuffling_iteration_counts.svg": plot_shuffled_counts(),
        "shuffling_iteration_bounds.svg": plot_shuffle_bounds(),
        "shuffling_iteration_shift.svg": plot_shift_schedule(),
        "shuffling_iteration_positions.svg": plot_iteration_positions(),
        "shuffling_iteration_hidden_state.svg": plot_hidden_state(),
    }
    for name, fig in figures.items():
        fig.tight_layout()
        fig.savefig(out / name, metadata={"Date": None})  # no timestamp, so output is reproducible
        plt.close(fig)
        print(f"saved {out / name}")


if __name__ == "__main__":
    main()
