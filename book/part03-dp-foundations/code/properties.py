"""Check post-processing, group privacy and composition, and draw the figures of 02-properties.md.

    uv run python book/part03-dp-foundations/code/properties.py
    uv run python book/part03-dp-foundations/code/properties.py --seed 1 --out /tmp/figures

Prints the numbers quoted in 02-properties.md (they match exactly with the default seed), then
saves the chapter's figures.

Conventions: a mechanism's output distributions on two datasets are P = M(D) and Q = M(D'), given
as probability lists over a finite set of outputs. The privacy profile is
δ(ε) = max(E_{e^ε}(P‖Q), E_{e^ε}(Q‖P)), so M is (ε, δ)-DP for this pair exactly when δ(ε) ≤ δ.
Logs are natural.
"""

import argparse
import math
from collections.abc import Callable, Sequence
from functools import cache
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
matplotlib.rcParams["svg.hashsalt"] = "properties"  # stable element ids, so reruns don't churn the SVGs
matplotlib.rcParams["svg.fonttype"] = "none"  # keep text as text: smaller, diffable files

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from numpy.typing import NDArray

FIGURES = Path(__file__).resolve().parent.parent / "figures"

Dist = Sequence[float]  # a distribution on a finite set of outputs

EPS_RR = math.log(3)  # randomized response with a fair-coin-then-coin protocol: truth with probability 3/4


# --- Toolkit: privacy profiles of finite distributions ---------------------------------------


def hockey_stick(p: Dist, q: Dist, eps: float) -> float:
    """E_{e^ε}(P‖Q) = Σ (p - e^ε q)_+, the largest P(S) - e^ε Q(S) over events S."""
    g = math.exp(eps)
    return sum(max(pi - g * qi, 0.0) for pi, qi in zip(p, q, strict=True))


def privacy_profile(p: Dist, q: Dist, eps: float) -> float:
    """δ(ε) for the pair, in both directions: the smallest δ with (ε, δ)-DP for D ≃ D'."""
    return max(hockey_stick(p, q, eps), hockey_stick(q, p, eps))


def pure_epsilon(p: Dist, q: Dist) -> float:
    """The worst-case privacy loss max_y |ln p(y)/q(y)|, infinite if one output is impossible under one side."""
    worst = 0.0
    for pi, qi in zip(p, q, strict=True):
        if pi == 0 and qi == 0:
            continue
        if pi == 0 or qi == 0:
            return math.inf
        worst = max(worst, abs(math.log(pi / qi)))
    return worst


def bisect(f: Callable[[float], float], lo: float, hi: float, steps: int = 80) -> float:
    """A root of f in [lo, hi], given that f changes sign there."""
    for _ in range(steps):
        mid = (lo + hi) / 2
        if (f(lo) < 0) == (f(mid) < 0):
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def epsilon_for_delta(profile: Callable[[float], float], delta: float, hi: float) -> float:
    """The smallest ε in [0, hi] with profile(ε) ≤ δ; profile is non-increasing in ε."""
    if profile(0.0) <= delta:
        return 0.0
    return bisect(lambda e: profile(e) - delta, 0.0, hi)


def apply_channel(p: Dist, kernel: Sequence[Sequence[float]]) -> list[float]:
    """Output distribution of a post-processing channel: (PK)(z) = Σ_y p(y) K(z | y)."""
    out = [0.0] * len(kernel[0])
    for py, row in zip(p, kernel, strict=True):
        for z, k in enumerate(row):
            out[z] += py * k
    return out


def product(p: Dist, q: Dist) -> list[float]:
    """The joint distribution of two independent outputs, flattened."""
    return [pi * qi for pi in p for qi in q]


# --- Randomized response and Laplace noise on a count ------------------------------------------


def rr_truth_probability(eps: float) -> float:
    """Randomized response that reports the true bit w.p. e^ε/(1 + e^ε) is exactly ε-DP."""
    return math.exp(eps) / (1 + math.exp(eps))


def rr_output(bit: int, eps: float) -> list[float]:
    """[P(report 0), P(report 1)] for a person whose true bit is `bit`."""
    t = rr_truth_probability(eps)
    return [t, 1 - t] if bit == 0 else [1 - t, t]


def flip_kernel(f: float) -> list[list[float]]:
    """A post-processing that flips a reported bit with probability f, with its own coin."""
    return [[1 - f, f], [f, 1 - f]]


def laplace_cdf(x: float, mu: float, b: float) -> float:
    if x < mu:
        return 0.5 * math.exp((x - mu) / b)
    return 1 - 0.5 * math.exp(-(x - mu) / b)


def laplace_interval(lo: float, hi: float, mu: float, b: float) -> float:
    """P(lo < mu + Z ≤ hi) for Z ~ Lap(0, b), computed from the nearer tail so that far cells keep full precision."""
    if hi <= mu:
        return laplace_cdf(hi, mu, b) - laplace_cdf(lo, mu, b)
    upper_hi = 0.5 * math.exp(-(hi - mu) / b) if not math.isinf(hi) else 0.0  # P(X > hi)
    if lo >= mu:
        return 0.5 * math.exp(-(lo - mu) / b) - upper_hi
    return 1 - laplace_cdf(lo, mu, b) - upper_hi


def laplace_profile(eps: float, t: float) -> float:
    """Exact δ(ε) of Lap(0, b) against Lap(s, b) with t = s/b (Part 1): 1 - e^{-(t - ε)/2} for ε < t."""
    return 1 - math.exp(-(t - eps) / 2) if eps < t else 0.0


def rounded_laplace(mu: float, b: float, lo: int = -60, hi: int = 70) -> list[float]:
    """pmf of round(mu + Z), Z ~ Lap(0, b), on the integers lo..hi; the end cells absorb the tails."""
    return [
        laplace_interval(-math.inf if j == lo else j - 0.5, math.inf if j == hi else j + 0.5, mu, b)
        for j in range(lo, hi + 1)
    ]


def thresholded_laplace(mu: float, b: float, threshold: float) -> list[float]:
    """[P(mu + Z ≤ threshold), P(mu + Z > threshold)]: report only whether the noisy count passes a threshold."""
    below = laplace_cdf(threshold, mu, b)
    return [below, 1 - below]


# --- Group privacy ------------------------------------------------------------------------------


def group_delta(eps: float, delta: float, k: int) -> float:
    """δ_k = δ (1 + e^ε + ... + e^{(k-1)ε}), the δ of group privacy at ε_k = kε."""
    return delta * sum(math.exp(j * eps) for j in range(k))


def group_delta_simple(eps: float, delta: float, k: int) -> float:
    """The simpler, looser form k e^{(k-1)ε} δ."""
    return k * math.exp((k - 1) * eps) * delta


def alarm_mechanism(eps: float, delta: float, size: int, count: int) -> list[float]:
    """[P(alarm), P(quiet)] for a mechanism on counts that makes the group bound tight.

    The alarm probability a_c = δ Σ_{j < size - c} e^{jε} satisfies a_c = e^ε a_{c+1} + δ, so every
    neighbouring pair is (ε, δ)-DP with equality on the event "alarm", and the losses pile up along a path.
    """
    a = delta * sum(math.exp(j * eps) for j in range(max(size - count, 0)))
    return [a, 1 - a]


def normal_cdf(x: float) -> float:
    return 0.5 * math.erfc(-x / math.sqrt(2))


def gaussian_profile(eps: float, mu: float) -> float:
    """Exact δ(ε) of N(0, 1) against N(mu, 1) (Part 1): Φ(mu/2 - ε/mu) - e^ε Φ(-mu/2 - ε/mu)."""
    return normal_cdf(mu / 2 - eps / mu) - math.exp(eps) * normal_cdf(-mu / 2 - eps / mu)


def gaussian_sigma(eps: float, delta: float) -> float:
    """The noise scale σ at which Gaussian noise on a count (shift 1) has δ(ε) = δ exactly."""
    return bisect(lambda s: gaussian_profile(eps, 1 / s) - delta, 0.05, 100.0)


# --- Composition --------------------------------------------------------------------------------


def composed_rr_profile(eps0: float, k: int, eps: float) -> float:
    """Exact δ(ε) of k independent randomized-response answers about the same bit.

    With j answers equal to the true bit, the privacy loss is (2j - k) ε0, and j ~ Bin(k, t) under D.
    """
    p, q = composed_rr_pmfs(eps0, k)
    return float(np.sum(np.maximum(p - math.exp(eps) * q, 0.0)))


@cache
def composed_rr_pmfs(eps0: float, k: int) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """P(j answers agree with bit 1) under D (bit 1) and under D' (bit 0), j = 0..k."""
    t = rr_truth_probability(eps0)
    j = np.arange(k + 1, dtype=np.float64)
    log_comb = np.array([math.lgamma(k + 1) - math.lgamma(i + 1) - math.lgamma(k - i + 1) for i in range(k + 1)])
    p = np.exp(log_comb + j * math.log(t) + (k - j) * math.log(1 - t))
    q = np.exp(log_comb + j * math.log(1 - t) + (k - j) * math.log(t))
    return p, q


def composed_rr_loss_pmf(eps0: float, k: int, bit: int) -> tuple[list[float], list[float]]:
    """The privacy loss L_{D,D'} of k answers, as (values, probabilities) when the true bit is `bit`.

    D has bit 1 and D' has bit 0; under D the answers agree with 1 w.p. t, under D' w.p. 1 - t.
    """
    t = rr_truth_probability(eps0)
    s = t if bit == 1 else 1 - t
    values = [(2 * j - k) * eps0 for j in range(k + 1)]
    probs = [math.comb(k, j) * s**j * (1 - s) ** (k - j) for j in range(k + 1)]
    return values, probs


def composed_rr_epsilon(eps0: float, k: int, delta: float) -> float:
    return epsilon_for_delta(lambda e: composed_rr_profile(eps0, k, e), delta, k * eps0)


def composed_rr_mean_loss(eps0: float, k: int) -> float:
    """E[L] under D = k ε0 (2t - 1) = k ε0 tanh(ε0/2): the KL divergence of the k answers."""
    return k * eps0 * math.tanh(eps0 / 2)


def simulate_rr_losses(rng: np.random.Generator, eps0: float, k: int, runs: int) -> NDArray[np.float64]:
    """Privacy losses of `runs` independent executions of k answers about a person whose bit is 1."""
    agree = rng.binomial(k, rr_truth_probability(eps0), size=runs)
    return ((2 * agree - k) * eps0).astype(np.float64)


def composed_gaussian_epsilon(eps: float, delta: float, k: int) -> float:
    """Exact ε at δ_total = kδ for k Gaussian counts, each calibrated to (ε, δ).

    k independent N(0, σ²) vs N(1, σ²) coordinates have the same likelihood ratio as one pair at shift √k/σ.
    """
    mu = math.sqrt(k) / gaussian_sigma(eps, delta)
    return epsilon_for_delta(lambda e: gaussian_profile(e, mu), k * delta, k * eps)


def adaptive_two_step(eps1: float, eps_if_one: float, eps_if_zero: float) -> list[tuple[int, int, float, float]]:
    """An adaptive composition of two randomized-response answers about the same bit.

    The first answer uses ε1; the second uses eps_if_one if the first answer was 1 and eps_if_zero
    otherwise. Returns (y1, y2, P under D: bit 1, P under D': bit 0) for the four outcomes.
    """
    rows: list[tuple[int, int, float, float]] = []
    for y1 in (1, 0):
        eps2 = eps_if_one if y1 == 1 else eps_if_zero
        for y2 in (1, 0):
            p = rr_output(1, eps1)[y1] * rr_output(1, eps2)[y2]
            q = rr_output(0, eps1)[y1] * rr_output(0, eps2)[y2]
            rows.append((y1, y2, p, q))
    return rows


# --- Parallel composition and noise reuse -------------------------------------------------------


def histogram_epsilon(b: float, moved: bool) -> float:
    """Pure ε of a two-bin histogram with rounded Laplace(b) noise per bin.

    moved=False: one record is added (one bin changes by 1). moved=True: one record is replaced by
    one in the other bin (one bin goes up, the other down).
    """
    p = product(rounded_laplace(0.0, b, -40, 41), rounded_laplace(0.0, b, -40, 41))
    q_second = rounded_laplace(-1.0, b, -40, 41) if moved else rounded_laplace(0.0, b, -40, 41)
    q = product(rounded_laplace(1.0, b, -40, 41), q_second)
    return pure_epsilon(p, q)


def difference_attack_accuracy(rng: np.random.Generator, b: float, shared: bool, runs: int) -> float:
    """Accuracy of guessing whether Alice is in the data from two noisy counts by thresholding their difference.

    c1 counts smokers (Alice included or not), c2 counts smokers aged 40 or over (Alice is 30, so c2 does
    not depend on her). With shared noise, (c1 + Z) - (c2 + Z) = c1 - c2 exactly.
    """
    member: NDArray[np.int64] = rng.integers(0, 2, size=runs)
    c1 = 100.0 + member
    c2 = 60.0
    z1: NDArray[np.float64] = rng.laplace(0.0, b, size=runs)
    z2: NDArray[np.float64] = z1 if shared else rng.laplace(0.0, b, size=runs)
    diff = (c1 + z1) - (c2 + z2)
    correct: NDArray[np.bool_] = (diff > 40.5) == (member == 1)
    return float(np.mean(correct))


# --- The chapter's numbers ------------------------------------------------------------------------


def print_table(rng: np.random.Generator) -> None:
    print("[Post-processing: randomized response, eps = ln 3, then a fresh coin flips the report w.p. f]")
    for f in (0.0, 0.1, 0.25, 0.5):
        p = apply_channel(rr_output(1, EPS_RR), flip_kernel(f))
        q = apply_channel(rr_output(0, EPS_RR), flip_kernel(f))
        print(f"  f={f:g}: P(report 1 | bit 1)={p[1]:.4f}  eps={pure_epsilon(p, q):.4f}")
    print(f"  ln(5/3)={math.log(5 / 3):.4f}")

    b = 1.0
    print(f"[Post-processing: count + Lap(0, {b:g}), D has count 0 and D' count 1]")
    rows: list[tuple[str, list[float], list[float]]] = [
        ("rounded", rounded_laplace(0.0, b), rounded_laplace(1.0, b)),
        ("threshold 0.5", thresholded_laplace(0.0, b, 0.5), thresholded_laplace(1.0, b, 0.5)),
    ]
    print(
        f"  raw: pure eps={1 / b:.4f}  delta(0.5)={laplace_profile(0.5, 1 / b):.4f}  "
        f"delta(0.25)={laplace_profile(0.25, 1 / b):.4f}"
    )
    for name, p, q in rows:
        print(
            f"  {name}: pure eps={pure_epsilon(p, q):.4f}  delta(0.5)={privacy_profile(p, q, 0.5):.4f}  "
            f"delta(0.25)={privacy_profile(p, q, 0.25):.4f}"
        )
    zero_cell = math.log(rounded_laplace(0.0, b)[60] / rounded_laplace(1.0, b)[60])
    print(f"  rounded: privacy loss of the output 0 = {zero_cell:.4f}")
    above = thresholded_laplace(0.0, b, 0.5)[1]
    print(f"  threshold: P(pass | count 0)={above:.4f}  P(pass | count 1)={1 - above:.4f}")

    print("[Group privacy, pure: count + rounded Lap(0, 1), counts differ by k]")
    print(
        "  "
        + "  ".join(
            f"k={k}: eps={pure_epsilon(rounded_laplace(0.0, 1.0), rounded_laplace(float(k), 1.0)):.4f}"
            for k in (1, 2, 3, 5)
        )
    )

    eps, delta, size = 0.5, 1e-5, 10
    sigma = gaussian_sigma(eps, delta)
    print(f"[Group privacy, approximate: each neighbouring pair is ({eps:g}, {delta:g})-DP]")
    worst_pair = max(
        privacy_profile(alarm_mechanism(eps, delta, size, c), alarm_mechanism(eps, delta, size, c + 1), eps)
        for c in range(size)
    )
    print(
        f"  alarm mechanism, size {size}: largest neighbouring delta(0.5)={worst_pair:.3e}  "
        f"P(alarm | count 0)={alarm_mechanism(eps, delta, size, 0)[0]:.3e}"
    )
    print(f"  Gaussian count: sigma={sigma:.4f}  delta(0.5) at shift 1={gaussian_profile(eps, 1 / sigma):.3e}")
    for k in (1, 2, 3, 5, 10):
        chain = privacy_profile(alarm_mechanism(eps, delta, size, 0), alarm_mechanism(eps, delta, size, k), k * eps)
        gauss = gaussian_profile(k * eps, k / sigma)
        print(
            f"  k={k}: eps_k={k * eps:g}  bound={group_delta(eps, delta, k):.3e}  "
            f"simple={group_delta_simple(eps, delta, k):.3e}  alarm exact={chain:.3e}  Gaussian exact={gauss:.3e}"
        )

    k = 10
    gauss10 = gaussian_profile(k * eps, k / sigma)
    print(
        f"  k={k}: simple/bound={group_delta_simple(eps, delta, k) / group_delta(eps, delta, k):.2f}  "
        f"bound/Gaussian={group_delta(eps, delta, k) / gauss10:.2f}  "
        f"k=2: simple/bound={group_delta_simple(eps, delta, 2) / group_delta(eps, delta, 2):.2f}"
    )
    print(f"  eps=1, k=10: delta_k / delta = {group_delta(1.0, 1.0, 10):.1f}")

    print("[Composition: k randomized-response answers about the same bit, eps0 = ln 3]")
    for k in (1, 2, 5, 10, 20, 50, 100):
        values, probs = composed_rr_loss_pmf(EPS_RR, k, 1)
        print(
            f"  k={k}: basic={k * EPS_RR:.3f}  worst loss={max(values):.3f}  "
            f"P(loss=worst)={probs[-1]:.3e}  mean loss={composed_rr_mean_loss(EPS_RR, k):.3f}  "
            f"exact eps(1e-6)={composed_rr_epsilon(EPS_RR, k, 1e-6):.3f}"
        )
    eps0 = 0.1
    print(f"[Composition: k randomized-response answers, eps0 = {eps0:g}]")
    for k in (10, 100, 1000):
        print(
            f"  k={k}: basic={k * eps0:.3f}  mean loss={composed_rr_mean_loss(eps0, k):.3f}  "
            f"exact eps(1e-6)={composed_rr_epsilon(eps0, k, 1e-6):.3f}"
        )

    k, runs = 10, 200_000
    losses = simulate_rr_losses(rng, EPS_RR, k, runs)
    print(f"[Simulation: k={k} answers, eps0 = ln 3, {runs} runs]")
    print(
        f"  mean loss={float(np.mean(losses)):.3f} (exact {composed_rr_mean_loss(EPS_RR, k):.3f})  "
        f"fraction at worst case={float(np.mean(losses >= k * EPS_RR - 1e-9)):.4f} (exact {0.75**k:.4f})"
    )

    print("[Adaptive composition: second answer uses ln 3 after a 1 and 0.5 after a 0]")
    transcripts = adaptive_two_step(EPS_RR, EPS_RR, 0.5)
    for y1, y2, pt, qt in transcripts:
        print(f"  y=({y1},{y2}): P_D={pt:.4f}  P_D'={qt:.4f}  loss={math.log(pt / qt):.4f}")
    print(
        f"  worst |loss|={pure_epsilon([r[2] for r in transcripts], [r[3] for r in transcripts]):.4f}  "
        f"bound eps1 + max eps2={2 * EPS_RR:.4f}"
    )

    print(f"[Composition of Gaussian counts, each ({eps:g}, {delta:g})-DP]")
    for k in (1, 2, 4, 10):
        print(
            f"  k={k}: basic eps={k * eps:g} at delta={k * delta:g}  "
            f"exact eps={composed_gaussian_epsilon(eps, delta, k):.3f}"
        )

    print("[Parallel composition: two-bin histogram, rounded Lap(0, 1) per bin]")
    print(
        f"  add or remove one record: eps={histogram_epsilon(1.0, False):.4f}  "
        f"replace a record by one in the other bin: eps={histogram_epsilon(1.0, True):.4f}"
    )

    runs = 200_000
    print(f"[Noise reuse: two counts with Lap(0, 1) noise, {runs} runs]")
    print(
        f"  shared noise: attack accuracy={difference_attack_accuracy(rng, 1.0, True, runs):.4f}  "
        f"fresh noise: attack accuracy={difference_attack_accuracy(rng, 1.0, False, runs):.4f}  "
        f"best possible with fresh noise={0.5 + 0.5 * laplace_profile(0.0, 1.0):.4f}"
    )


# --- Figures --------------------------------------------------------------------------------------


def plot_postprocessing() -> Figure:
    fig, (left, right) = plt.subplots(1, 2, figsize=(11, 4.2))
    eps_grid = [i / 200 for i in range(241)]
    b = 1.0
    raw = [laplace_profile(e, 1 / b) for e in eps_grid]
    rounded = [privacy_profile(rounded_laplace(0.0, b), rounded_laplace(1.0, b), e) for e in eps_grid]
    thresh = [privacy_profile(thresholded_laplace(0.0, b, 0.5), thresholded_laplace(1.0, b, 0.5), e) for e in eps_grid]
    left.plot(eps_grid, raw, "k", linewidth=2, label="raw noisy count")
    left.plot(eps_grid, rounded, "C0", label="rounded to an integer")
    left.plot(eps_grid, thresh, "C1", label="only: is it above 0.5?")
    left.set(
        xlabel=r"$\varepsilon$",
        ylabel=r"$\delta(\varepsilon)$",
        xlim=(0, 1.2),
        ylim=(0, 0.42),
        title="Count + Lap(0, 1): processing lowers the privacy profile",
    )
    left.legend(fontsize=9)

    fs = [i / 200 for i in range(101)]
    eps_after = [
        pure_epsilon(
            apply_channel(rr_output(1, EPS_RR), flip_kernel(f)), apply_channel(rr_output(0, EPS_RR), flip_kernel(f))
        )
        for f in fs
    ]
    right.plot(fs, eps_after, "C2", linewidth=2)
    right.axhline(EPS_RR, color="k", linestyle="--", linewidth=1, label=r"before processing: $\ln 3$")
    right.plot([0.25], [math.log(5 / 3)], "o", color="C2", label=r"$f = 1/4$: $\ln(5/3)$")
    right.set(
        xlabel="probability f of flipping the report",
        ylabel=r"$\varepsilon$ after processing",
        xlim=(0, 0.5),
        ylim=(0, 1.2),
        title="Randomized response, then a fresh coin flip",
    )
    right.legend(fontsize=9)
    return fig


def plot_group() -> Figure:
    eps, delta, size = 0.5, 1e-5, 12
    sigma = gaussian_sigma(eps, delta)
    ks = list(range(1, size + 1))
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.plot(
        ks, [group_delta_simple(eps, delta, k) for k in ks], "k:", label=r"simple bound $k e^{(k-1)\varepsilon}\delta$"
    )
    ax.plot(ks, [group_delta(eps, delta, k) for k in ks], "k--", label=r"bound $\delta\sum_{j<k} e^{j\varepsilon}$")
    ax.plot(
        ks,
        [
            privacy_profile(alarm_mechanism(eps, delta, size, 0), alarm_mechanism(eps, delta, size, k), k * eps)
            for k in ks
        ],
        "o",
        color="C3",
        label="alarm mechanism (exact)",
    )
    ax.plot(
        ks,
        [gaussian_profile(k * eps, k / sigma) for k in ks],
        "s-",
        color="C0",
        label=f"Gaussian noise on a count, σ = {sigma:.2f} (exact)",
    )
    ax.set(
        yscale="log",
        xlabel="group size k",
        ylabel=r"$\delta_k$ at $\varepsilon_k = k\varepsilon$",
        title=r"Group privacy from a $(0.5, 10^{-5})$ guarantee per record",
    )
    ax.legend(fontsize=9, loc="lower right")
    return fig


def plot_rr_loss(k: int = 10) -> Figure:
    values, under_d = composed_rr_loss_pmf(EPS_RR, k, 1)
    _, under_d2 = composed_rr_loss_pmf(EPS_RR, k, 0)
    fig, ax = plt.subplots(figsize=(8, 4.2))
    width = 0.4
    ax.bar([v - width / 2 for v in values], under_d, width=width, color="C0", label="under M(D): true bit 1")
    ax.bar([v + width / 2 for v in values], under_d2, width=width, color="C1", label="under M(D'): true bit 0")
    ax.axvline(k * EPS_RR, color="k", linestyle="--", linewidth=1)
    ax.text(
        k * EPS_RR + 0.3,
        0.2,
        rf"basic bound $k\varepsilon_0$ = {k * EPS_RR:.2f}"
        + "\n"
        + rf"reached w.p. $(3/4)^{{{k}}}$"
        + f"\n= {0.75**k:.3f}",
        ha="left",
        fontsize=9,
    )
    mean = composed_rr_mean_loss(EPS_RR, k)
    ax.axvline(mean, color="C0", linestyle=":", linewidth=1)
    ax.text(mean + 0.2, 0.33, f"mean under M(D) = {mean:.2f}", ha="left", fontsize=9, color="C0")
    ax.set(
        xlabel=r"privacy loss $L_{D,D'}(y)$ (nats)",
        ylabel="probability",
        ylim=(0, 0.36),
        xlim=(-12.5, 17.5),
        title=rf"Privacy loss of {k} randomized-response answers, $\varepsilon_0 = \ln 3$ each",
    )
    ax.legend(fontsize=9, loc="upper left")
    return fig


def plot_composition() -> Figure:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for ax, eps0, kmax in ((axes[0], EPS_RR, 100), (axes[1], 0.1, 1000)):
        ks = sorted({max(1, round(kmax * i / 50)) for i in range(51)})
        ax.plot(ks, [k * eps0 for k in ks], "k--", label=r"basic $k\varepsilon_0$ = exact worst-case loss")
        for delta, color in ((1e-6, "C0"), (1e-3, "C1")):
            ax.plot(
                ks,
                [composed_rr_epsilon(eps0, k, delta) for k in ks],
                color=color,
                label=rf"exact $\varepsilon$ at $\delta = 10^{{{round(math.log10(delta))}}}$",
            )
        ax.plot(ks, [composed_rr_mean_loss(eps0, k) for k in ks], ":", color="0.4", label="mean privacy loss")
        name = r"\ln 3" if eps0 == EPS_RR else f"{eps0:g}"
        ax.set(
            xlabel="number of answers k",
            ylabel=r"$\varepsilon$ of the k answers",
            xlim=(0, kmax),
            ylim=(0, None),
            title=rf"Randomized response, $\varepsilon_0 = {name}$",
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

    out.mkdir(parents=True, exist_ok=True)
    figures = {
        "dpprop_postprocessing.svg": plot_postprocessing(),
        "dpprop_group.svg": plot_group(),
        "dpprop_rr_loss.svg": plot_rr_loss(),
        "dpprop_composition.svg": plot_composition(),
    }
    for name, fig in figures.items():
        fig.tight_layout()
        fig.savefig(out / name, metadata={"Date": None})  # no timestamp, so output is reproducible
        plt.close(fig)
        print(f"saved {out / name}")


if __name__ == "__main__":
    main()
