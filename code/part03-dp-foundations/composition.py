"""Check the composition theorems in the advanced-and-optimal-composition chapter and draw its figures.

    uv run python code/part03-dp-foundations/composition.py
    uv run python code/part03-dp-foundations/composition.py --seed 1 --out /tmp/figures

Prints the numbers quoted in 04-composition.md (they match exactly with the default seed),
then saves the chapter's figures.

Conventions: P is the output distribution on D, Q the one on the neighbour D', and the privacy
loss is L = ln(p/q). An (ε, δ) pair satisfies E_{e^ε}(P‖Q) ≤ δ, where E_γ is the hockey-stick
divergence. Randomized response with parameter ε reports the true bit with probability
e^ε / (1 + e^ε); its privacy loss is +ε or -ε. Logs are natural.
"""

import argparse
import itertools
import math
import sys
from collections.abc import Sequence
from fractions import Fraction
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # for code/shared
from shared.figures import configure
from shared.stats import normal_cdf

configure("composition")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from numpy.typing import NDArray

# Figures are build output: _build/figures/<part>/, which the build copies into each book.
FIGURES = Path(__file__).resolve().parents[2] / "_build" / "figures" / Path(__file__).parent.name

Dist = Sequence[float]  # a distribution on a finite set, as a list of probabilities


def logsumexp(values: NDArray[np.float64]) -> float:
    """ln Σ e^v without overflow; -inf for an empty sum."""
    if values.size == 0:
        return -math.inf
    top = float(np.max(values))
    if top == -math.inf:
        return -math.inf
    return top + math.log(float(np.sum(np.exp(values - top))))


# --- Toolkit: the privacy loss of one pair ---------------------------------------------------


def hockey_stick(p: Dist, q: Dist, eps: float) -> float:
    """δ(ε) = E_{e^ε}(P‖Q) = Σ (p - e^ε q)_+, the smallest δ for which the pair is (ε, δ) one way."""
    g = math.exp(eps)
    return sum(max(pi - g * qi, 0.0) for pi, qi in zip(p, q, strict=True))


def kl(p: Dist, q: Dist) -> float:
    """D_KL(P‖Q) = E_P[L], the mean privacy loss."""
    return sum(pi * math.log(pi / qi) for pi, qi in zip(p, q, strict=True) if pi > 0)


def max_loss(p: Dist, q: Dist) -> float:
    """The largest |L| over the outcomes: the smallest ε for which the pair is (ε, 0) both ways."""
    return max(abs(math.log(pi / qi)) for pi, qi in zip(p, q, strict=True) if pi > 0 or qi > 0)


def rr_keep(eps: float) -> float:
    """Probability e^ε / (1 + e^ε) that randomized response reports the true bit."""
    return 1 / (1 + math.exp(-eps))


def rr_pair(eps: float) -> tuple[list[float], list[float]]:
    """Randomized response on D (true bit 0) and on D' (true bit 1), outputs (0, 1)."""
    a = rr_keep(eps)
    return [a, 1 - a], [1 - a, a]


def four_point_pair(eps: float, delta: float) -> tuple[list[float], list[float]]:
    """The (ε, δ) randomized response of Kairouz, Oh & Viswanath, outputs (flag D, 0, 1, flag D')."""
    a = rr_keep(eps)
    return [delta, (1 - delta) * a, (1 - delta) * (1 - a), 0.0], [0.0, (1 - delta) * (1 - a), (1 - delta) * a, delta]


def rr_mean_loss(eps: float) -> float:
    """E_P[L] for randomized response: ε(2a - 1) = ε tanh(ε/2), the largest mean loss of any ε-DP pair."""
    return eps * math.tanh(eps / 2)


def drv_mean_bound(eps: float) -> float:
    """The cruder bound E_P[L] ≤ ε(e^ε - 1) of Dwork, Rothblum & Vadhan."""
    return eps * math.expm1(eps)


def rr_channel(p: Dist, q: Dist, eps: float) -> tuple[list[float], list[float]]:
    """The post-processing K with P = RR_ε(D) K and Q = RR_ε(D') K: K(·|0) = (e^ε p - q)/(e^ε - 1), K(·|1) = (e^ε q - p)/(e^ε - 1)."""
    g = math.exp(eps)
    k0 = [(g * pi - qi) / (g - 1) for pi, qi in zip(p, q, strict=True)]
    k1 = [(g * qi - pi) / (g - 1) for pi, qi in zip(p, q, strict=True)]
    return k0, k1


def apply_channel(source: Dist, rows: Sequence[Dist]) -> list[float]:
    """Push a distribution on the rows' inputs through the channel."""
    return [sum(s * row[y] for s, row in zip(source, rows, strict=True)) for y in range(len(rows[0]))]


def laplace_mean_loss(eps: float) -> float:
    """E_P[L] for Lap(0, 1/ε) against Lap(1, 1/ε): the KL t + e^{-t} - 1 with t = ε."""
    return eps + math.exp(-eps) - 1


# --- Basic and advanced composition --------------------------------------------------------


def basic_epsilon(eps: float, k: int) -> float:
    return k * eps


def advanced_epsilon(eps: float, k: int, delta_prime: float) -> float:
    """Dwork–Rothblum–Vadhan: ε √(2k ln(1/δ')) + kε(e^ε - 1)."""
    return eps * math.sqrt(2 * k * math.log(1 / delta_prime)) + k * drv_mean_bound(eps)


def advanced_epsilon_tanh(eps: float, k: int, delta_prime: float) -> float:
    """The same Azuma argument with the exact worst mean loss: ε √(2k ln(1/δ')) + kε tanh(ε/2)."""
    return eps * math.sqrt(2 * k * math.log(1 / delta_prime)) + k * rr_mean_loss(eps)


def advanced_crossover(eps: float, delta_prime: float) -> int:
    """Smallest k at which advanced composition (DRV form) is below basic composition kε."""
    k = 1
    while advanced_epsilon(eps, k, delta_prime) >= basic_epsilon(eps, k):
        k += 1
    return k


# --- Optimal composition (Kairouz, Oh & Viswanath) -----------------------------------------


def log_binomial_coefficients(k: int) -> NDArray[np.float64]:
    """ln C(k, j) for j = 0..k, built as a cumulative sum of ln((k - j + 1)/j): no factorial overflows."""
    j = np.arange(1, k + 1, dtype=np.float64)
    return np.concatenate(([0.0], np.cumsum(np.log((k - j + 1) / j))))


def rr_product_log_pmfs(eps: float, k: int) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """For k independent copies of RR_ε, with j = the number of outputs that point to D':

    returns (ln P[j], ln Q[j], L_j), with P[j] = C(k, j) a^{k-j} (1-a)^j, Q[j] = P[j] e^{-L_j}, L_j = (k - 2j)ε.
    """
    log_a = -math.log1p(math.exp(-eps))  # ln(e^ε / (1 + e^ε)), stable for large ε
    log_b = -eps + log_a  # ln(1 / (1 + e^ε))
    j = np.arange(k + 1, dtype=np.float64)
    log_p = log_binomial_coefficients(k) + (k - j) * log_a + j * log_b
    losses = (k - 2 * j) * eps
    return log_p, log_p - losses, losses


def kov_delta_pure(eps: float, k: int, eps_prime: float) -> float:
    """δ(ε') of k-fold RR_ε: Σ_j P[j] (1 - e^{ε' - L_j})_+, summed in log space term by term."""
    log_p, _, losses = rr_product_log_pmfs(eps, k)
    keep = losses > eps_prime
    terms = log_p[keep] + np.log(-np.expm1(eps_prime - losses[keep]))
    return math.exp(logsumexp(terms))


def kov_delta(eps: float, delta: float, k: int, eps_prime: float) -> float:
    """δ(ε') of k-fold composition of (ε, δ)-DP mechanisms: 1 - (1 - δ)^k (1 - δ_RR(ε'))."""
    clean = math.exp(k * math.log1p(-delta))  # (1 - δ)^k, the chance that no step shows its flag
    return (1 - clean) + clean * kov_delta_pure(eps, k, eps_prime)


def kov_delta_grid_naive(eps: float, delta: float, k: int, i: int) -> float:
    """The paper's formula at ε' = (k - 2i)ε, evaluated literally in floating point (overflows for large kε)."""
    g = math.exp(eps)
    total = sum(math.comb(k, ell) * (g ** (k - ell) - g ** (k - 2 * i + ell)) for ell in range(i))
    return 1 - (1 - delta) ** k * (1 - total / (1 + g) ** k)


def kov_epsilon(eps: float, delta: float, k: int, delta_target: float) -> float:
    """Smallest ε' ≥ 0 with δ(ε') ≤ δ_target, exactly: on each piece between grid points δ_RR = A - e^{ε'} B."""
    clean = math.exp(k * math.log1p(-delta))
    if delta_target < 1 - clean:
        return math.inf
    target = (delta_target - (1 - clean)) / clean
    log_p, log_q, losses = rr_product_log_pmfs(eps, k)
    # For ε' in (L_i, L_{i-1}], the outputs with loss above ε' are j = 0..i-1. A_i and B_i are their
    # P- and Q-masses, accumulated in log space (e^{L_i} alone can overflow).
    log_a = np.logaddexp.accumulate(log_p)[:-1]  # index i - 1 holds ln A_i, i = 1..k
    log_b = np.logaddexp.accumulate(log_q)[:-1]
    lower = np.maximum(losses[1:], 0.0)  # bottom of piece i, clipped at ε' = 0
    a_sum = np.exp(log_a)
    with np.errstate(divide="ignore", invalid="ignore"):
        gap = np.log(a_sum - target)  # ln(A_i - target); nan or -inf where A_i <= target
    inside = (a_sum > target) & (gap > lower + log_b)  # δ_RR(bottom of piece) > target
    last = int(np.argmax(losses[1:] <= 0))  # the piece that contains ε' = 0
    hits = np.flatnonzero(inside[: last + 1])
    if hits.size == 0:
        return 0.0
    i = int(hits[0])
    return float(gap[i] - log_b[i])


# --- Brute force: enumerate every output sequence ------------------------------------------


def product_pair(p: Dist, q: Dist, k: int) -> tuple[list[float], list[float]]:
    """The k-fold product P^k and Q^k, listed over all |outputs|^k sequences."""
    pk = [math.prod(p[y] for y in seq) for seq in itertools.product(range(len(p)), repeat=k)]
    qk = [math.prod(q[y] for y in seq) for seq in itertools.product(range(len(q)), repeat=k)]
    return pk, qk


def brute_force_delta(p: Dist, q: Dist, k: int, eps_prime: float) -> float:
    pk, qk = product_pair(p, q, k)
    return hockey_stick(pk, qk, eps_prime)


def exact_rr_delta(g: int, k: int, i: int) -> Fraction:
    """δ at ε' = (k - 2i)ε for k-fold RR with e^ε = g an integer, by enumeration in exact rationals."""
    a = Fraction(g, g + 1)
    p, q = [a, 1 - a], [1 - a, a]
    threshold = Fraction(g) ** (k - 2 * i)
    total = Fraction(0)
    for seq in itertools.product((0, 1), repeat=k):
        pk = math.prod((p[y] for y in seq), start=Fraction(1))
        qk = math.prod((q[y] for y in seq), start=Fraction(1))
        total += max(pk - threshold * qk, Fraction(0))
    return total


# --- Gaussian mechanism: per-step (ε, δ) against the exact composed curve ------------------


def gaussian_delta(mu: float, eps: float) -> float:
    """δ(ε) of N(0, σ²) against N(1, σ²) with μ = 1/σ: Φ(μ/2 - ε/μ) - e^ε Φ(-μ/2 - ε/μ)."""
    return normal_cdf(mu / 2 - eps / mu) - math.exp(eps) * normal_cdf(-mu / 2 - eps / mu)


def gaussian_epsilon(mu: float, delta: float) -> float:
    """Smallest ε ≥ 0 with δ(ε) ≤ delta, by bisection (δ(ε) is decreasing)."""
    if gaussian_delta(mu, 0.0) <= delta:
        return 0.0
    lo, hi = 0.0, 1.0
    while gaussian_delta(mu, hi) > delta:
        hi *= 2
    for _ in range(100):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if gaussian_delta(mu, mid) > delta else (lo, mid)
    return hi


def gaussian_composed(sigma: float, k: int, delta: float) -> dict[str, float]:
    """ε after k Gaussian steps (sensitivity 1, noise σ) at total δ, four ways.

    The three generic routes convert each step to one (ε0, δ0) point first and optimize the split of
    δ over a grid; 'exact' composes the curves: k steps are one Gaussian pair with μ = √k/σ.
    """
    mu = 1 / sigma
    splits = [10 ** (-x / 4) for x in range(1, 41)]  # fraction of δ spent per step / on δ'
    basic = basic_epsilon(gaussian_epsilon(mu, delta / k), k)
    advanced = math.inf
    optimal = math.inf
    for s in [*splits, 1.0]:
        d0 = s * delta / k  # per-step δ; the rest goes to δ'
        e0 = gaussian_epsilon(mu, d0)
        if delta - k * d0 > 0:
            advanced = min(advanced, advanced_epsilon_tanh(e0, k, delta - k * d0))
        optimal = min(optimal, kov_epsilon(e0, d0, k, delta))
    return {
        "basic": basic,
        "advanced": advanced,
        "optimal": optimal,
        "exact": gaussian_epsilon(math.sqrt(k) * mu, delta),
    }


# --- Laplace noise in many dimensions -------------------------------------------------------


def per_step_epsilon(total: float, k: int, delta: float, route: str) -> float:
    """The per-step ε0 at which k steps of ε0-DP compose to (total, δ), by bisection.

    route is 'advanced' (the tanh form of advanced composition) or 'optimal' (Kairouz, Oh & Viswanath).
    """
    lo, hi = 0.0, total
    for _ in range(100):
        mid = (lo + hi) / 2
        spent = advanced_epsilon_tanh(mid, k, delta) if route == "advanced" else kov_epsilon(mid, 0.0, k, delta)
        lo, hi = (mid, hi) if spent <= total else (lo, mid)
    return lo


def dense_counts_rms(d: int, eps: float, delta: float) -> dict[str, float]:
    """Root-mean-square error per count when one record moves each of d counts by up to 1.

    Laplace with the whole vector's ℓ1 sensitivity d (pure DP); Laplace per count at the ε0 that advanced or
    optimal composition allows for d steps; the Gaussian mechanism with ℓ2 sensitivity √d, exactly calibrated.
    """
    mu = 1.0
    lo, hi = 1e-6, 50.0
    for _ in range(200):  # largest μ = √d/σ with δ(ε) ≤ δ
        mu = (lo + hi) / 2
        lo, hi = (mu, hi) if gaussian_delta(mu, eps) <= delta else (lo, mu)
    return {
        "pure": math.sqrt(2) * d / eps,
        "advanced": math.sqrt(2) / per_step_epsilon(eps, d, delta, "advanced"),
        "optimal": math.sqrt(2) / per_step_epsilon(eps, d, delta, "optimal"),
        "gaussian": math.sqrt(d) / lo,
    }


# --- Monte Carlo: Laplace steps and an adaptive analyst ------------------------------------


def laplace_losses(rng: np.random.Generator, eps: float, size: tuple[int, ...]) -> NDArray[np.float64]:
    """Privacy loss of Lap(0, 1/ε) against Lap(1, 1/ε), sampled under P: L = ε(|x - 1| - |x|)."""
    x = rng.laplace(0.0, 1 / eps, size=size)
    return eps * (np.abs(x - 1) - np.abs(x))


def rr_losses(rng: np.random.Generator, eps: float, size: tuple[int, ...]) -> NDArray[np.float64]:
    """Privacy loss of RR_ε sampled under P: +ε with probability e^ε/(1 + e^ε), else -ε."""
    return np.where(rng.random(size) < rr_keep(eps), eps, -eps)


def adaptive_losses(rng: np.random.Generator, eps: float, k: int, trials: int) -> NDArray[np.float64]:
    """Total loss of an adaptive analyst who asks a Laplace query after an output that points to D,
    and a randomized-response query otherwise. Each step is ε-DP; the choice depends on past outputs."""
    total = np.zeros(trials)
    last = np.zeros(trials)  # loss of the previous output: a function of that output, so public
    for _ in range(k):
        lap = laplace_losses(rng, eps, (trials,))
        rr = rr_losses(rng, eps, (trials,))
        step = np.where(last > 0, lap, rr)
        total += step
        last = step
    return total


def mc_delta(total_losses: NDArray[np.float64], eps_prime: float) -> float:
    """δ(ε') = E_P[(1 - e^{ε' - L})_+], averaged over samples of the total loss L."""
    return float(np.mean(np.maximum(0.0, -np.expm1(eps_prime - total_losses))))


# --- The chapter's numbers ------------------------------------------------------------------


def print_table(rng: np.random.Generator) -> None:
    eps, k, dp = 0.1, 100, 1e-5
    print(f"[Toolkit: one step, eps = {eps:g}]")
    print(
        f"  RR keep prob={rr_keep(eps):.4f}  mean loss eps*tanh(eps/2)={rr_mean_loss(eps):.6f}  "
        f"DRV bound eps(e^eps-1)={drv_mean_bound(eps):.6f}  eps^2/2={eps**2 / 2:.6f}"
    )
    for e in (1.0, math.log(3)):
        print(
            f"  eps={e:.4f}: tanh form={rr_mean_loss(e):.4f}  DRV={drv_mean_bound(e):.4f}  eps^2/2={e**2 / 2:.4f}  "
            f"Laplace mean loss={laplace_mean_loss(e):.4f}"
        )
    p3, q3 = [0.5, 0.3, 0.2], [0.2, 0.3, 0.5]
    k0, k1 = rr_channel(p3, q3, math.log(3))
    rp, rq = rr_pair(math.log(3))
    rebuilt_p, rebuilt_q = apply_channel(rp, [k0, k1]), apply_channel(rq, [k0, k1])
    print(
        f"  3-output pair: max loss={max_loss(p3, q3):.4f} (ln 3={math.log(3):.4f})  "
        f"K(.|0)={[round(x, 4) for x in k0]}  K(.|1)={[round(x, 4) for x in k1]}  "
        f"rebuilt P={[round(x, 4) for x in rebuilt_p]}  Q={[round(x, 4) for x in rebuilt_q]}"
    )
    print(
        f"  3-output pair: KL={kl(p3, q3):.4f}  RR_ln3 KL={kl(rp, rq):.4f}  "
        f"delta(0)={hockey_stick(p3, q3, 0):.4f}  RR delta(0)={hockey_stick(rp, rq, 0):.4f}"
    )

    print(f"[Privacy loss of k = {k} RR steps, eps = {eps:g}]")
    mean, sd = k * rr_mean_loss(eps), eps * math.sqrt(k * (1 - math.tanh(eps / 2) ** 2))
    print(
        f"  mean={mean:.4f}  sd={sd:.4f}  basic bound k*eps={basic_epsilon(eps, k):.1f}  "
        f"(k*eps - mean)/sd={(basic_epsilon(eps, k) - mean) / sd:.1f}"
    )
    print(f"  P[L = k*eps] = a^k = {rr_keep(eps) ** k:.3e}")
    print(
        f"  J ~ Bin({k}, 1-t): mean={k * (1 - rr_keep(eps)):.1f}  sd={math.sqrt(k * rr_keep(eps) * (1 - rr_keep(eps))):.2f}  "
        f"L > 4 iff J < {int((k - 4 / eps) / 2)}  optimal eps' is {(kov_epsilon(eps, 0.0, k, dp) - mean) / sd:.1f} sd above the mean  "
        f"e^(k*eps)={math.exp(basic_epsilon(eps, k)):.0f}"
    )

    print(f"[Composition of k = {k} eps = {eps:g} mechanisms, delta' = {dp:g}]")
    print(
        f"  basic={basic_epsilon(eps, k):.4f}  advanced DRV={advanced_epsilon(eps, k, dp):.4f}  "
        f"advanced tanh={advanced_epsilon_tanh(eps, k, dp):.4f}  optimal={kov_epsilon(eps, 0.0, k, dp):.4f}"
    )
    print(
        f"  Azuma term eps*sqrt(2k ln(1/delta'))={eps * math.sqrt(2 * k * math.log(1 / dp)):.4f}  "
        f"DRV mean term={k * drv_mean_bound(eps):.4f}  tanh mean term={k * rr_mean_loss(eps):.4f}"
    )
    print(
        f"  exact delta at the advanced eps: {kov_delta(eps, 0.0, k, advanced_epsilon(eps, k, dp)):.3e}  "
        f"at the tanh eps: {kov_delta(eps, 0.0, k, advanced_epsilon_tanh(eps, k, dp)):.3e}"
    )
    print(f"  crossover: advanced (DRV) beats basic from k={advanced_crossover(eps, dp)}")
    opt = kov_epsilon(eps, 0.0, k, dp)
    print(
        f"  optimal below: tanh advanced {1 - opt / advanced_epsilon_tanh(eps, k, dp):.0%}  "
        f"DRV advanced {1 - opt / advanced_epsilon(eps, k, dp):.0%}  basic {1 - opt / basic_epsilon(eps, k):.0%}  "
        f"tanh advanced above optimal by {advanced_epsilon_tanh(eps, k, dp) / opt - 1:.0%}"
    )
    print(
        f"  k=10000 terms: Azuma={eps * math.sqrt(2 * 10_000 * math.log(1 / dp)):.2f}  DRV mean={10_000 * drv_mean_bound(eps):.2f}  "
        f"tanh mean={10_000 * rr_mean_loss(eps):.2f}  sqrt(2 ln(1/delta'))={math.sqrt(2 * math.log(1 / dp)):.4f}  "
        f"2 ln(1/delta')={2 * math.log(1 / dp):.2f}"
    )
    for kk in (10, 1000, 10000):
        print(
            f"  k={kk}: basic={basic_epsilon(eps, kk):.3f}  advanced DRV={advanced_epsilon(eps, kk, dp):.3f}  "
            f"advanced tanh={advanced_epsilon_tanh(eps, kk, dp):.3f}  optimal={kov_epsilon(eps, 0.0, kk, dp):.3f}"
        )

    print("[Optimal composition: checks]")
    for e, d, kk in ((0.5, 0.0, 12), (0.3, 1e-3, 6), (1.0, 0.01, 5)):
        p, q = rr_pair(e) if d == 0 else four_point_pair(e, d)
        worst = 0.0
        for ep in (0.0, 0.37, 1.0, 2.2):
            worst = max(worst, abs(kov_delta(e, d, kk, ep) - brute_force_delta(p, q, kk, ep)))
        grid = max(
            abs(kov_delta(e, d, kk, (kk - 2 * i) * e) - kov_delta_grid_naive(e, d, kk, i)) for i in range(kk // 2 + 1)
        )
        print(
            f"  eps={e:g} delta={d:g} k={kk}: |formula - enumeration| <= {worst:.1e}  "
            f"|log-space - paper's grid formula| <= {grid:.1e}  ({len(p) ** kk} sequences)"
        )
    g, kk = 3, 8
    exact_worst = max(
        abs(float(exact_rr_delta(g, kk, i)) - kov_delta_pure(math.log(g), kk, (kk - 2 * i) * math.log(g)))
        for i in range(kk // 2 + 1)
    )
    print(
        f"  eps=ln 3, k={kk}, exact rationals: max error={exact_worst:.1e}  delta(0)={exact_rr_delta(g, kk, kk // 2)}"
    )
    big_eps, big_k = 1.0, 1000
    try:
        naive = f"{kov_delta_grid_naive(big_eps, 0.0, big_k, 225):.3e}"
    except OverflowError:
        naive = "OverflowError"
    print(
        f"  eps={big_eps:g} k={big_k} at eps'=(k-2*225)eps: naive={naive}  log-space="
        f"{kov_delta_pure(big_eps, big_k, (big_k - 450) * big_eps):.3e}"
    )
    print(
        f"  eps=0.01 k=100000 delta=1e-6: optimal eps={kov_epsilon(0.01, 0.0, 100_000, 1e-6):.4f}  "
        f"advanced tanh={advanced_epsilon_tanh(0.01, 100_000, 1e-6):.4f}"
    )
    print(
        f"  advanced tanh above optimal at eps=0.01 k=100000: "
        f"{advanced_epsilon_tanh(0.01, 100_000, 1e-6) / kov_epsilon(0.01, 0.0, 100_000, 1e-6) - 1:.0%}  "
        f"log10 (1+e)^1000={1000 * math.log10(1 + math.e):.1f}"
    )

    print("[Optimal composition with delta per step]")
    e, d, kk, target = 0.1, 5e-8, 100, 1e-5
    print(
        f"  eps={e:g} delta={d:g} k={kk} target={target:g}: 1-(1-delta)^k={1 - (1 - d) ** kk:.3e}  "
        f"optimal={kov_epsilon(e, d, kk, target):.4f}  advanced (delta'={target - kk * d:.0e})="
        f"{advanced_epsilon(e, kk, target - kk * d):.4f}"
    )

    print(f"[Monte Carlo: k = {k} steps of eps = {eps:g}, delta(eps') at eps' = 2]")
    trials = 100_000
    lap = laplace_losses(rng, eps, (trials, k)).sum(axis=1)
    rr = rr_losses(rng, eps, (trials, k)).sum(axis=1)
    adapt = adaptive_losses(rng, eps, k, trials)
    print(
        f"  RR exact={kov_delta_pure(eps, k, 2.0):.4f}  RR Monte Carlo={mc_delta(rr, 2.0):.4f}  "
        f"Laplace Monte Carlo={mc_delta(lap, 2.0):.4f}  adaptive Monte Carlo={mc_delta(adapt, 2.0):.4f}"
    )
    print(
        f"  mean total loss: RR={k * rr_mean_loss(eps):.4f}  Laplace={k * laplace_mean_loss(eps):.4f} "
        f"(Monte Carlo {float(np.mean(lap)):.4f})  adaptive Monte Carlo={float(np.mean(adapt)):.4f}"
    )

    print("[Gaussian mechanism, sensitivity 1, total delta = 1e-5]")
    for sigma, kk in ((5.0, 100), (20.0, 1000)):
        r = gaussian_composed(sigma, kk, 1e-5)
        print(
            f"  sigma={sigma:g} k={kk}: basic={r['basic']:.3f}  advanced={r['advanced']:.3f}  "
            f"optimal={r['optimal']:.3f}  exact={r['exact']:.3f}"
        )
        print(f"    optimal / exact = {r['optimal'] / r['exact']:.1f}")

    print("[Laplace noise on d dense counts, eps = 1, delta = 1e-5: rms error per count]")
    for dd in (100, 1000, 10000):
        r = dense_counts_rms(dd, 1.0, 1e-5)
        print(
            f"  d={dd}: pure Laplace={r['pure']:.1f}  Laplace + advanced={r['advanced']:.1f}  "
            f"Laplace + optimal={r['optimal']:.1f}  Gaussian={r['gaussian']:.1f}"
        )
    print(
        f"  d=1000 per-count eps0: advanced={per_step_epsilon(1.0, 1000, 1e-5, 'advanced'):.5f}  "
        f"optimal={per_step_epsilon(1.0, 1000, 1e-5, 'optimal'):.5f}"
    )

    print("[Exercises]")
    print(f"  ex1 eps=1: tanh form={rr_mean_loss(1.0):.4f}  DRV={drv_mean_bound(1.0):.4f}  eps^2/2=0.5000")
    print(
        f"  ex2 crossover eps=0.1 delta'=1e-5: formula k > {2 * math.log(1e5) / (1 - math.expm1(0.1)) ** 2:.1f}  "
        f"search k={advanced_crossover(0.1, 1e-5)}"
    )
    rp, rq = rr_pair(math.log(3))
    p2, q2 = product_pair(rp, rq, 2)
    print(
        f"  ex3 k=2 eps=ln3: delta(0)={hockey_stick(p2, q2, 0):.4f}  delta(ln3)={hockey_stick(p2, q2, math.log(3)):.4f}  "
        f"delta(2ln3)={hockey_stick(p2, q2, 2 * math.log(3)):.4f}"
    )


# --- Figures --------------------------------------------------------------------------------


def plot_tradeoff_region(eps: float = math.log(3), delta: float = 0.1) -> Figure:
    """The (ε, δ) boundary as a trade-off curve, attained by the four-point pair, and a Laplace pair inside."""
    alphas = [i / 1000 for i in range(1001)]
    fig, (left, right) = plt.subplots(1, 2, figsize=(11, 4.4))
    for d, style in ((0.0, "-"), (delta, "--")):
        bound = [max(0.0, 1 - d - math.exp(eps) * a, math.exp(-eps) * (1 - d - a)) for a in alphas]
        left.plot(alphas, bound, "k", linestyle=style, label=rf"$(\varepsilon, \delta)$ = (ln 3, {d:g}): RR attains it")
    lap = [laplace_tradeoff(a, eps) for a in alphas]
    left.plot(alphas, lap, "C1", label=r"Laplace, $\varepsilon$ = ln 3: inside the region")
    left.plot(alphas, [1 - a for a in alphas], color="0.7", linestyle=":", label="no information")
    left.set(
        xlabel=r"type I error $\alpha$",
        ylabel=r"type II error $\beta$",
        xlim=(0, 1),
        ylim=(0, 1),
        title="Every (ε, δ)-DP pair lies above the RR curve",
    )
    left.legend(fontsize=8, loc="upper right")

    # Right: the channel that turns RR into the 3-output pair.
    p3, q3 = [0.5, 0.3, 0.2], [0.2, 0.3, 0.5]
    k0, k1 = rr_channel(p3, q3, eps)
    xs = np.arange(3)
    w = 0.2
    right.bar(xs - 1.5 * w, p3, w, color="C0", label="P = M(D)")
    right.bar(xs - 0.5 * w, q3, w, color="C1", label="Q = M(D')")
    right.bar(xs + 0.5 * w, k0, w, color="C0", alpha=0.45, hatch="//", label="K(· | RR says 0)")
    right.bar(xs + 1.5 * w, k1, w, color="C1", alpha=0.45, hatch="//", label="K(· | RR says 1)")
    right.set(
        xticks=list(xs),
        xticklabels=["y = 1", "y = 2", "y = 3"],
        ylabel="probability",
        ylim=(0, 0.8),
        title="A ln 3-DP pair is RR followed by the channel K",
    )
    right.legend(fontsize=8, loc="upper center", ncol=2)
    return fig


def laplace_tradeoff(alpha: float, kappa: float) -> float:
    """Trade-off curve of Lap(0, b) against Lap(b κ, b): the piecewise formula of the hypothesis-testing chapter."""
    if alpha < 0.5 * math.exp(-kappa):
        return 1 - math.exp(kappa) * alpha
    if alpha <= 0.5:
        return math.exp(-kappa) / (4 * alpha)
    return math.exp(-kappa) * (1 - alpha)


def plot_loss_distribution(eps: float = 0.1, k: int = 100, delta_prime: float = 1e-5) -> Figure:
    log_p, _, losses = rr_product_log_pmfs(eps, k)
    fig, ax = plt.subplots(figsize=(9, 4.2))
    ax.bar(
        losses,
        np.exp(log_p),
        width=1.6 * eps,
        color="C0",
        alpha=0.7,
        label=rf"total privacy loss of {k} RR steps, $\varepsilon$ = {eps:g}",
    )
    marks = (
        (k * rr_mean_loss(eps), "C2", ":", rf"mean $k\varepsilon\tanh(\varepsilon/2)$ = {k * rr_mean_loss(eps):.2f}"),
        (kov_epsilon(eps, 0.0, k, delta_prime), "C3", "-", f"optimal ε' = {kov_epsilon(eps, 0.0, k, delta_prime):.2f}"),
        (
            advanced_epsilon(eps, k, delta_prime),
            "C4",
            "--",
            f"advanced ε' = {advanced_epsilon(eps, k, delta_prime):.2f}",
        ),
        (basic_epsilon(eps, k), "k", "-.", rf"basic $k\varepsilon$ = {basic_epsilon(eps, k):g}"),
    )
    for x, color, style, label in marks:
        ax.axvline(x, color=color, linestyle=style, linewidth=1.5, label=label)
    ax.set(
        xlabel=r"total privacy loss $L = \sum_i L_i$",
        ylabel="probability under P",
        xlim=(-4, 10.5),
        title=rf"The loss concentrates near its mean, far below $k\varepsilon$ (δ' = {delta_prime:g})",
    )
    ax.legend(fontsize=8, loc="upper right")
    return fig


def plot_epsilon_vs_k(delta_prime: float = 1e-5) -> Figure:
    ks = sorted({round(10 ** (x / 10)) for x in range(41)})
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    for ax, eps in zip(axes, (0.1, 0.5)):
        ax.plot(ks, [basic_epsilon(eps, k) for k in ks], "k-.", label=r"basic $k\varepsilon$")
        ax.plot(
            ks, [advanced_epsilon(eps, k, delta_prime) for k in ks], "C4--", label="advanced (Dwork–Rothblum–Vadhan)"
        )
        ax.plot(
            ks,
            [advanced_epsilon_tanh(eps, k, delta_prime) for k in ks],
            "C2:",
            label=r"advanced, mean $\varepsilon\tanh(\varepsilon/2)$",
        )
        ax.plot(
            ks,
            [kov_epsilon(eps, 0.0, k, delta_prime) for k in ks],
            "C3-",
            linewidth=2,
            label="optimal (Kairouz–Oh–Viswanath)",
        )
        ax.set(
            xscale="log",
            yscale="log",
            xlabel="number of compositions k",
            ylabel="total ε′",
            title=rf"$\varepsilon$ = {eps:g} per step, total δ = {delta_prime:g}",
        )
        ax.legend(fontsize=8, loc="upper left")
    return fig


def plot_delta_curves(rng: np.random.Generator, eps: float = 0.1, k: int = 100) -> Figure:
    grid = [i / 10 for i in range(106)]
    trials = 40_000
    lap = laplace_losses(rng, eps, (trials, k)).sum(axis=1)
    adapt = adaptive_losses(rng, eps, k, trials)
    fig, ax = plt.subplots(figsize=(8, 4.4))
    ax.plot(grid, [kov_delta_pure(eps, k, x) for x in grid], "C3", linewidth=2, label="optimal: k-fold RR (exact)")
    floor = 100 / trials  # below this, a Monte Carlo average rests on too few samples to plot
    for losses, style, label in (
        (adapt, "C2--", "adaptive mix of RR and Laplace (simulated)"),
        (lap, "C1", "k-fold Laplace (simulated)"),
    ):
        pts = [(x, mc_delta(losses, x)) for x in grid]
        pts = [(x, d) for x, d in pts if d >= floor]
        ax.plot([x for x, _ in pts], [d for _, d in pts], style, label=label)
    for dp, marker in ((1e-2, "o"), (1e-3, "s"), (1e-5, "^"), (1e-7, "D")):
        ax.plot(
            [advanced_epsilon(eps, k, dp)],
            [dp],
            marker,
            color="C4",
            markersize=7,
            label="advanced composition (ε', δ')" if dp == 1e-2 else None,
        )
    ax.plot(
        [basic_epsilon(eps, k)], [1e-8], ">", color="k", markersize=7, label=r"basic: $(k\varepsilon, 0)$ (at the axis)"
    )
    ax.set(
        yscale="log",
        ylim=(1e-8, 1),
        xlim=(0, 10.5),
        xlabel="ε′",
        ylabel="δ(ε′)",
        title=rf"Privacy curves of {k} steps of $\varepsilon$ = {eps:g}",
    )
    ax.legend(fontsize=8, loc="upper right")
    return fig


def plot_gaussian(sigma: float = 20.0, delta: float = 1e-5) -> Figure:
    ks = sorted({round(10 ** (x / 6)) for x in range(19)})
    rows = [gaussian_composed(sigma, k, delta) for k in ks]
    fig, ax = plt.subplots(figsize=(8, 4.4))
    for key, style, label in (
        ("basic", "k-.", "basic, per-step (ε, δ/k)"),
        ("advanced", "C2:", "advanced, best per-step split"),
        ("optimal", "C3-", "optimal for the (ε, δ) class, best split"),
        ("exact", "C0--", "exact Gaussian: one pair with μ = √k / σ"),
    ):
        ax.plot(ks, [r[key] for r in rows], style, linewidth=2 if key in ("optimal", "exact") else 1.5, label=label)
    ax.set(
        xscale="log",
        yscale="log",
        xlabel="number of Gaussian steps k",
        ylabel=r"total $\varepsilon$",
        title=rf"Gaussian mechanism, $\sigma$ = {sigma:g}, total δ = {delta:g}",
    )
    ax.legend(fontsize=8, loc="upper left")
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
        "comp_tradeoff_region.svg": plot_tradeoff_region(),
        "comp_loss_distribution.svg": plot_loss_distribution(),
        "comp_epsilon_vs_k.svg": plot_epsilon_vs_k(),
        "comp_delta_curves.svg": plot_delta_curves(rng),
        "comp_gaussian.svg": plot_gaussian(),
    }
    for name, fig in figures.items():
        fig.tight_layout()
        fig.savefig(out / name, metadata={"Date": None})  # no timestamp, so output is reproducible
        plt.close(fig)
        print(f"saved {out / name}")


if __name__ == "__main__":
    main()
