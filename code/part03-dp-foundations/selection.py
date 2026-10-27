"""Check the selection mechanisms of the selection chapter and draw its figures.

    uv run python code/part03-dp-foundations/selection.py
    uv run python code/part03-dp-foundations/selection.py --seed 1 --out /tmp/figures

Prints the numbers quoted in 05-selection.md (they match exactly with the default seed),
then saves the chapter's figures.

Conventions: a selection problem has k candidates with scores u(D, r); Δ is the score sensitivity,
the largest change of any one score between neighbouring datasets. Laplace and Gumbel noise are given
by their scale. Logarithms are natural. Probabilities that can be computed exactly (the exponential
mechanism, Laplace report-noisy-max, the sparse vector technique) are computed by a sum or by
numerical integration on a fine grid; Monte Carlo is used only to check them.
"""

import argparse
import math
import sys
from collections.abc import Sequence
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # for code/shared
from shared.figures import configure

configure("selection")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from numpy.typing import NDArray

# Figures are build output: _build/figures/<part>/, which the build copies into each book.
FIGURES = Path(__file__).resolve().parents[2] / "_build" / "figures" / Path(__file__).parent.name

Scores = Sequence[float]
FloatArray = NDArray[np.float64]

EULER_GAMMA = 0.5772156649015329

# The vote example: four options, D has counts (3, 2, 1, 0); D' adds one vote for option B.
VOTES_D: list[float] = [3, 2, 1, 0]
VOTES_D2: list[float] = [3, 3, 1, 0]

# Model selection: 50 models, scored by the number of correct predictions on 1,000 validation examples.
# Model r gets 900 - r right, so neighbouring ranks differ by one example (0.1% accuracy).
MODEL_SCORES: list[float] = [900.0 - r for r in range(50)]

# Five candidates for the Gumbel-max check.
FIVE_SCORES: list[float] = [4, 3, 3, 1, 0]


# --- Toolkit: Laplace and Gumbel noise --------------------------------------------------------


def laplace_log_cdf(x: FloatArray, b: float) -> FloatArray:
    """ln Pr[Z < x] for Z ~ Lap(b), accurate in both tails."""
    return np.where(x < 0, math.log(0.5) + np.minimum(x, 0.0) / b, np.log1p(-0.5 * np.exp(-np.maximum(x, 0.0) / b)))


def laplace_log_sf(x: FloatArray, b: float) -> FloatArray:
    """ln Pr[Z ≥ x] for Z ~ Lap(b); by symmetry it is ln Pr[Z < -x]."""
    return laplace_log_cdf(-x, b)


def laplace_log_pdf(x: FloatArray, b: float) -> FloatArray:
    return -np.abs(x) / b - math.log(2 * b)


def laplace_sf(x: float, b: float) -> float:
    """Pr[Z > x] for Z ~ Lap(b)."""
    return 0.5 * math.exp(-x / b) if x >= 0 else 1 - 0.5 * math.exp(x / b)


def laplace_shift_ratio(t: float, s: float, b: float) -> float:
    """Pr[Z > t - s] / Pr[Z > t]: how much a threshold moved down by s helps. At most e^{s/b} (Fact 3)."""
    return laplace_sf(t - s, b) / laplace_sf(t, b)


# --- The exponential mechanism -----------------------------------------------------------------


def softmax(logits: FloatArray) -> FloatArray:
    z = logits - logits.max()  # subtract the max so exp never overflows
    w = np.exp(z)
    return w / w.sum()


def em_probabilities(scores: Scores, eps: float, sensitivity: float = 1.0, monotone: bool = False) -> FloatArray:
    """Pr[M(D) = r] ∝ exp(ε u(D, r) / (2Δ)); with monotone scores the 2 can be dropped."""
    factor = 1.0 if monotone else 2.0
    return softmax(eps * np.asarray(scores, dtype=np.float64) / (factor * sensitivity))


def max_log_ratio(p: FloatArray, q: FloatArray) -> float:
    """max over outcomes of |ln p/q|: the pure-DP level an output distribution pair attains."""
    return float(np.max(np.abs(np.log(p) - np.log(q))))


def em_gap_tail(scores: Scores, eps: float, c: float, sensitivity: float = 1.0) -> float:
    """Pr[OPT - u(D, M(D)) ≥ c], computed exactly by summing the output probabilities."""
    u = np.asarray(scores, dtype=np.float64)
    p = em_probabilities(scores, eps, sensitivity)
    return float(p[u.max() - u >= c - 1e-12].sum())


def em_gap_bound(k: int, eps: float, c: float, sensitivity: float = 1.0) -> float:
    """The utility theorem: Pr[OPT - u ≥ c] ≤ k exp(-εc/(2Δ)), capped at 1."""
    return min(1.0, k * math.exp(-eps * c / (2 * sensitivity)))


def em_gap_threshold(k: int, eps: float, t: float, sensitivity: float = 1.0) -> float:
    """The gap 2Δ(ln k + t)/ε that the exponential mechanism exceeds with probability at most e^{-t}."""
    return 2 * sensitivity * (math.log(k) + t) / eps


def em_expected_gap(scores: Scores, eps: float, sensitivity: float = 1.0, monotone: bool = False) -> float:
    u = np.asarray(scores, dtype=np.float64)
    p = em_probabilities(scores, eps, sensitivity, monotone)
    return float(np.sum(p * (u.max() - u)))


def em_expected_gap_bound(k: int, eps: float, sensitivity: float = 1.0) -> float:
    """E[OPT - u] ≤ 2Δ(ln k + 1)/ε, from integrating the tail bound."""
    return 2 * sensitivity * (math.log(k) + 1) / eps


def tied_profile(k: int, gap: float, top: float = 900.0) -> list[float]:
    """One best candidate and k - 1 candidates all exactly `gap` below it: the bound's worst case."""
    return [top] + [top - gap] * (k - 1)


# --- Report-noisy-max ---------------------------------------------------------------------------


def report_noisy_max(
    rng: np.random.Generator, scores: Scores, scale: float, size: int, noise: str = "laplace"
) -> NDArray[np.int64]:
    """argmax_r (u_r + Z_r) with i.i.d. noise of the given scale, repeated `size` times."""
    u = np.asarray(scores, dtype=np.float64)
    shape = (size, len(u))
    if noise == "laplace":
        z = rng.laplace(0.0, scale, shape)
    elif noise == "gumbel":
        z = rng.gumbel(0.0, scale, shape)
    elif noise == "exponential":
        z = rng.exponential(scale, shape)
    else:
        raise ValueError(noise)
    return np.argmax(u + z, axis=1).astype(np.int64)


def frequencies(winners: NDArray[np.int64], k: int) -> FloatArray:
    return np.bincount(winners, minlength=k).astype(np.float64) / len(winners)


def laplace_rnm_probabilities(scores: Scores, b: float, points_per_scale: int = 100) -> FloatArray:
    """Pr[candidate i wins] for Laplace report-noisy-max, by numerical integration:

    Pr[i] = ∫ f(z) Π_{j≠i} F(z + u_i - u_j) dz, with f, F the Lap(b) density and CDF.
    """
    u = np.asarray(scores, dtype=np.float64)
    z = np.linspace(-30 * b, 30 * b, 60 * points_per_scale + 1)
    dz = float(z[1] - z[0])
    out = np.empty(len(u))
    for i in range(len(u)):
        others = np.delete(u, i)
        log_f = laplace_log_pdf(z, b) + laplace_log_cdf(z[:, None] + u[i] - others[None, :], b).sum(axis=1)
        out[i] = float(np.exp(log_f).sum() * dz)
    return out


def rnm_gap_bound(k: int, b: float, beta: float) -> float:
    """With probability ≥ 1 - β, the winner's true score is within 2b ln(k/β) of the best (union bound)."""
    return 2 * b * math.log(k / beta)


def rnm_expected_gap(rng: np.random.Generator, scores: Scores, scale: float, size: int, noise: str) -> float:
    u = np.asarray(scores, dtype=np.float64)
    winners = report_noisy_max(rng, scores, scale, size, noise)
    return float(np.mean(u.max() - u[winners]))


def gumbel_max_mean(scores: Scores, beta: float) -> float:
    """E[max_r (u_r + G_r)] for G_r ~ Gumbel(β): the max is Gumbel with location β ln Σ e^{u_r/β}."""
    u = np.asarray(scores, dtype=np.float64) / beta
    lse = float(u.max() + np.log(np.exp(u - u.max()).sum()))
    return beta * (lse + EULER_GAMMA)


def two_candidate_best(d_over_b: float) -> tuple[float, float]:
    """Pr[the better of two candidates wins], scores d apart, for EM (∝ e^{u/b}) and Laplace RNM (scale b)."""
    x = d_over_b
    em = 1 / (1 + math.exp(-x))
    lap = 1 - 0.5 * math.exp(-x) * (1 + x / 2)  # Pr[Z1 - Z2 > -d], Z1 - Z2 has the difference-of-Laplace law
    return em, lap


# --- The sparse vector technique ----------------------------------------------------------------


def svt_grid(b_rho: float, points_per_scale: int = 100, width: float = 30.0) -> tuple[FloatArray, float]:
    r = np.linspace(-width * b_rho, width * b_rho, int(2 * width * points_per_scale) + 1)
    return r, float(r[1] - r[0])


def logsumexp(x: FloatArray, axis: int = 0) -> FloatArray:
    m = np.max(x, axis=axis, keepdims=True)
    return np.squeeze(m, axis=axis) + np.log(np.sum(np.exp(x - m), axis=axis))


def above_threshold_log_probs(f: Scores, threshold: float, b_rho: float, b_nu: float) -> FloatArray:
    """ln Pr[AboveThreshold stops at query k] for k = 0..m-1, and ln Pr[no query is above] last.

    Query i says ⊥ when f_i + ν_i < T + ρ and ⊤ otherwise; the run stops at the first ⊤.
    Exact up to the integration grid over the threshold noise ρ ~ Lap(b_rho); ν_i ~ Lap(b_nu).
    """
    fv = np.asarray(f, dtype=np.float64)
    r, dr = svt_grid(b_rho)
    values, which = np.unique(fv, return_inverse=True)  # evaluate the CDF once per distinct answer
    x = threshold + r[:, None] - values[None, :]  # ν_i must fall below x for a ⊥
    log_below = laplace_log_cdf(x, b_nu)[:, which]
    log_above = laplace_log_sf(x, b_nu)[:, which]
    below_before = np.concatenate([np.zeros((len(r), 1)), np.cumsum(log_below, axis=1)], axis=1)
    terms = np.concatenate([below_before[:, :-1] + log_above, below_before[:, -1:]], axis=1)
    terms = terms + laplace_log_pdf(r, b_rho)[:, None]
    return logsumexp(terms, axis=0) + math.log(dr)


def no_halt_pattern_log_prob(f: Scores, pattern: Sequence[bool], threshold: float, b_rho: float, b_nu: float) -> float:
    """ln Pr[the no-halt variant answers exactly `pattern` (True = ⊤)] on queries f."""
    fv = np.asarray(f, dtype=np.float64)
    above = np.asarray(pattern, dtype=bool)
    r, dr = svt_grid(b_rho)
    values, which = np.unique(fv, return_inverse=True)
    x = threshold + r[:, None] - values[None, :]
    logs = np.where(above[None, :], laplace_log_sf(x, b_nu)[:, which], laplace_log_cdf(x, b_nu)[:, which]).sum(axis=1)
    return float(logsumexp(laplace_log_pdf(r, b_rho) + logs)[()] + math.log(dr))


def svt_counterexample(m: int) -> tuple[list[float], list[float], list[bool]]:
    """2m sensitivity-1 queries: f(D) = (0^m, 1^m), f(D') = (1^m, 0^m), and the output ⊥^m ⊤^m."""
    return [0.0] * m + [1.0] * m, [1.0] * m + [0.0] * m, [False] * m + [True] * m


def above_threshold_max_log_ratio(m: int, eps: float, threshold: float = 0.0) -> float:
    """max over outputs of |ln Pr[out | D] / Pr[out | D']| for AboveThreshold on the counterexample."""
    fd, fd2, _ = svt_counterexample(m)
    a = above_threshold_log_probs(fd, threshold, 2 / eps, 4 / eps)
    b = above_threshold_log_probs(fd2, threshold, 2 / eps, 4 / eps)
    return float(np.max(np.abs(a - b)))


def no_halt_log_ratio(m: int, eps: float, threshold: float = 0.0) -> float:
    """ln Pr[⊥^m ⊤^m | D] / Pr[⊥^m ⊤^m | D'] for the variant that never halts."""
    fd, fd2, pattern = svt_counterexample(m)
    return no_halt_pattern_log_prob(fd, pattern, threshold, 2 / eps, 4 / eps) - no_halt_pattern_log_prob(
        fd2, pattern, threshold, 2 / eps, 4 / eps
    )


def simulate_no_halt(
    rng: np.random.Generator, f: Scores, threshold: float, b_rho: float, b_nu: float, trials: int, chunk: int = 10_000
) -> NDArray[np.bool_]:
    """The ⊤/⊥ answers of the no-halt variant, one row per run (True = ⊤)."""
    fv = np.asarray(f, dtype=np.float64)
    rows: list[NDArray[np.bool_]] = []
    for start in range(0, trials, chunk):
        n = min(chunk, trials - start)
        rho = rng.laplace(0.0, b_rho, (n, 1))
        nu = rng.laplace(0.0, b_nu, (n, len(fv)))
        rows.append(fv[None, :] + nu >= threshold + rho)
    return np.concatenate(rows, axis=0)


def sign_test_rates(
    rng: np.random.Generator, m: int, eps: float, trials: int, threshold: float = 0.0
) -> tuple[float, float]:
    """Pr[more ⊤ in the second half than in the first] for the no-halt variant, under D and under D'.

    Simulated exactly but cheaply: given the threshold noise ρ, the m answers of each half are i.i.d.,
    so the number of ⊤ in a half is Binomial(m, Pr[f + ν ≥ T + ρ]).
    """
    fd, fd2, _ = svt_counterexample(m)
    rates: list[float] = []
    for f in (fd, fd2):
        rho = rng.laplace(0.0, 2 / eps, trials)
        p_first = np.exp(laplace_log_sf(threshold + rho - f[0], 4 / eps))
        p_second = np.exp(laplace_log_sf(threshold + rho - f[-1], 4 / eps))
        first, second = rng.binomial(m, p_first), rng.binomial(m, p_second)
        rates.append(float(np.mean(second > first)))
    return rates[0], rates[1]


def simulate_above_threshold(
    rng: np.random.Generator, f: Scores, threshold: float, b_rho: float, b_nu: float, trials: int, chunk: int = 2_000
) -> NDArray[np.int64]:
    """Index of the first ⊤ in each run (len(f) when every query is ⊥)."""
    fv = np.asarray(f, dtype=np.float64)
    out: list[NDArray[np.int64]] = []
    for start in range(0, trials, chunk):
        n = min(chunk, trials - start)
        rho = rng.laplace(0.0, b_rho, (n, 1))
        nu = rng.laplace(0.0, b_nu, (n, len(fv)))
        above = fv[None, :] + nu >= threshold + rho
        first = np.where(above.any(axis=1), np.argmax(above, axis=1), len(fv))
        out.append(first.astype(np.int64))
    return np.concatenate(out)


def svt_alpha(k: int, beta: float, eps: float, sensitivity: float = 1.0) -> float:
    """Accuracy of AboveThreshold over k queries: α = 8Δ(ln k + ln(2/β))/ε (Dwork & Roth Thm 3.24)."""
    return 8 * sensitivity * (math.log(k) + math.log(2 / beta)) / eps


def svt_noise_quantile(rng: np.random.Generator, k: int, eps: float, beta: float, trials: int) -> float:
    """The (1 - β)-quantile of 2 max(|ρ|, max_i |ν_i|): the smallest α the proof's event needs in practice."""
    vals: list[FloatArray] = []
    for start in range(0, trials, 2_000):
        n = min(2_000, trials - start)
        rho = np.abs(rng.laplace(0.0, 2 / eps, n))
        nu = np.abs(rng.laplace(0.0, 4 / eps, (n, k))).max(axis=1)
        vals.append(2 * np.maximum(rho, nu))
    return float(np.quantile(np.concatenate(vals), 1 - beta))


def monitoring_stream(k: int = 1000, hit: int = 700, low: float = 400.0, high: float = 600.0) -> list[float]:
    """k counting queries, all at `low` except query `hit`, which is at `high`."""
    return [high if i == hit else low for i in range(k)]


def sparse_alpha(k: int, c: int, beta: float, eps: float, sensitivity: float = 1.0) -> float:
    """Margin of Sparse (c restarts of AboveThreshold at ε/c each): 8cΔ(ln k + ln(2c/β))/ε (Dwork & Roth Thm 3.26)."""
    return svt_alpha(k, beta / c, eps / c, sensitivity)


def advanced_tanh(eps0: float, k: int, delta: float) -> float:
    """ε0 √(2k ln(1/δ)) + k ε0 tanh(ε0/2): advanced composition of k steps of ε0-DP, at slack δ."""
    return eps0 * math.sqrt(2 * k * math.log(1 / delta)) + k * eps0 * math.tanh(eps0 / 2)


def sparse_run_epsilon(eps: float, c: int, delta: float) -> float:
    """Largest per-run ε0 whose c-fold advanced composition is at most ε (bisection)."""
    lo, hi = 0.0, eps
    for _ in range(100):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if advanced_tanh(mid, c, delta) <= eps else (lo, mid)
    return lo


def sparse_alpha_approx(k: int, c: int, beta: float, eps: float, delta: float, sensitivity: float = 1.0) -> float:
    """Margin of Sparse when the c runs are composed with advanced composition: (ε, δ)-DP overall."""
    return svt_alpha(k, beta / c, sparse_run_epsilon(eps, c, delta), sensitivity)


# --- Applications: learning by choosing a hypothesis --------------------------------------------

THRESHOLDS = np.linspace(0.0, 1.0, 1001)  # the class C: h_θ(x) = 1[x ≥ θ], θ on a grid of 1001 points
TRUE_THRESHOLD = 0.3
LABEL_NOISE = 0.1


def threshold_sample(rng: np.random.Generator, n: int) -> tuple[FloatArray, NDArray[np.int64]]:
    """x uniform on [0, 1]; y = 1[x ≥ 0.3], flipped with probability 0.1."""
    x = rng.random(n)
    y = (x >= TRUE_THRESHOLD).astype(np.int64) ^ (rng.random(n) < LABEL_NOISE).astype(np.int64)
    return x, y


def threshold_errors(x: FloatArray, y: NDArray[np.int64]) -> FloatArray:
    """Training mistakes of every h_θ: positives below θ plus negatives at or above θ."""
    order = np.argsort(x)
    xs, ys = x[order], y[order]
    below = np.searchsorted(xs, THRESHOLDS, side="left")  # records with x < θ
    pos_cum = np.concatenate([[0], np.cumsum(ys)])
    neg_cum = np.concatenate([[0], np.cumsum(1 - ys)])
    return (pos_cum[below] + (neg_cum[-1] - neg_cum[below])).astype(np.float64)


def population_excess(theta: FloatArray) -> FloatArray:
    """err(h_θ) - err(h_0.3) = (1 - 2η)|θ - 0.3| for x uniform and label noise η."""
    return (1 - 2 * LABEL_NOISE) * np.abs(theta - TRUE_THRESHOLD)


def learner_excess(rng: np.random.Generator, n: int, eps: float | None, trials: int) -> float:
    """Average excess population error of the exponential-mechanism learner (eps None: non-private ERM)."""
    total = 0.0
    excess = population_excess(THRESHOLDS)
    for _ in range(trials):
        x, y = threshold_sample(rng, n)
        mistakes = threshold_errors(x, y)
        if eps is None:
            best = mistakes == mistakes.min()
            total += float(excess[best].mean())
        else:
            total += float(np.sum(em_probabilities(list(-mistakes), eps) * excess))
    return total / trials


def learner_bound(n: int, eps: float, beta: float, size: int = len(THRESHOLDS)) -> tuple[float, float]:
    """The two terms of the excess-error bound: 2√(ln(2|C|/β)/(2n)) and 2 ln(|C|/β)/(εn)."""
    return 2 * math.sqrt(math.log(2 * size / beta) / (2 * n)), 2 * math.log(size / beta) / (eps * n)


def learner_samples(alpha: float, eps: float, beta: float, size: int = len(THRESHOLDS)) -> tuple[int, int]:
    """Records that guarantee excess error α: ⌈8 ln(2|C|/β)/α²⌉ (sampling) and ⌈4 ln(|C|/β)/(αε)⌉ (privacy)."""
    return math.ceil(8 * math.log(2 * size / beta) / alpha**2), math.ceil(4 * math.log(size / beta) / (alpha * eps))


# --- Applications: a private median over an interval -------------------------------------------

NORM_RANGE = 10.0  # the median is searched in [0, B]; B is fixed before looking at the data


def gradient_norms(rng: np.random.Generator, n: int) -> FloatArray:
    """n per-example gradient norms: lognormal with median 1, capped at the range B."""
    return np.minimum(np.exp(rng.normal(0.0, 0.75, n)), NORM_RANGE)


def median_intervals(data: FloatArray, upper: float) -> tuple[FloatArray, FloatArray, FloatArray]:
    """The n + 1 intervals between sorted points of [0, B]: left ends, lengths, and score -|#below - #above|."""
    xs = np.sort(data)
    edges = np.concatenate([[0.0], xs, [upper]])
    n = len(xs)
    j = np.arange(n + 1)
    return edges[:-1], np.diff(edges), -np.abs(2.0 * j - n)


def median_interval_probs(data: FloatArray, eps: float, upper: float = NORM_RANGE) -> FloatArray:
    """Probability that the continuous exponential mechanism lands in each interval: ∝ length × e^{εu/2}."""
    _, lengths, scores = median_intervals(data, upper)
    with np.errstate(divide="ignore"):
        logw = np.log(lengths) + eps * scores / 2
    w = np.exp(logw - logw.max())
    return w / w.sum()


def sample_median(rng: np.random.Generator, data: FloatArray, eps: float, upper: float = NORM_RANGE) -> float:
    """Exact sampling: pick an interval with its probability, then a uniform point inside it."""
    left, lengths, _ = median_intervals(data, upper)
    j = int(rng.choice(len(lengths), p=median_interval_probs(data, eps, upper)))
    return float(left[j] + rng.random() * lengths[j])


def median_rank_tail(data: FloatArray, eps: float, gap: float, upper: float = NORM_RANGE) -> float:
    """Pr[|#below - #above| ≥ gap] for the output, exactly."""
    _, _, scores = median_intervals(data, upper)
    return float(np.sum(median_interval_probs(data, eps, upper)[-scores >= gap]))


def median_rank_bound(data: FloatArray, eps: float, t: float, upper: float = NORM_RANGE) -> float:
    """The utility bound for the continuous version: (2/ε)(ln(B/L*) + t), L* the length of the best interval."""
    _, lengths, scores = median_intervals(data, upper)
    best = float(lengths[scores == scores.max()].sum())
    return 2 / eps * (math.log(upper / best) + t)


def median_expected_abs_error(data: FloatArray, eps: float, upper: float = NORM_RANGE) -> float:
    """E|output - true median|, exactly: each interval contributes the mean distance of a uniform point."""
    left, lengths, _ = median_intervals(data, upper)
    med = float(np.median(data))
    a, b = left, left + lengths
    # mean of |u - med| for u uniform on [a, b]
    inside = (a <= med) & (med <= b)
    safe = np.where(lengths > 0, lengths, 1.0)
    mean_in = ((med - a) ** 2 + (b - med) ** 2) / (2 * safe)
    mean_out = np.abs((a + b) / 2 - med)
    dist = np.where(inside, mean_in, mean_out)
    return float(np.sum(median_interval_probs(data, eps, upper) * dist))


# --- The table ----------------------------------------------------------------------------------


def print_table(rng: np.random.Generator) -> None:
    eps = 1.0
    print("[Toolkit: shifting a Laplace threshold, b = 1, s = 1]")
    for t in (0.0, 0.7, 1.0, 2.0):
        print(
            f"  t={t:g}: Pr[Z > t-1]={laplace_sf(t - 1, 1):.4f}  Pr[Z > t]={laplace_sf(t, 1):.4f}  "
            f"ratio={laplace_shift_ratio(t, 1, 1):.4f}  e^(s/b)={math.e:.4f}"
        )

    print("[Exponential mechanism, votes D = (3, 2, 1, 0), D' = (3, 3, 1, 0), eps = 1, sensitivity 1]")
    for monotone in (False, True):
        p, q = em_probabilities(VOTES_D, eps, monotone=monotone), em_probabilities(VOTES_D2, eps, monotone=monotone)
        name = "monotone e^(eps u)" if monotone else "standard e^(eps u/2)"
        print(f"  {name}: D: " + " ".join(f"{x:.4f}" for x in p) + "  D': " + " ".join(f"{x:.4f}" for x in q))
        print(
            "    ratios D/D': "
            + " ".join(f"{x:.4f}" for x in p / q)
            + f"  max |ln ratio|={max_log_ratio(p, q):.4f}  e^eps={math.exp(eps):.4f}"
        )
    w = np.exp(np.asarray(VOTES_D) / 2)
    w2 = np.exp(np.asarray(VOTES_D2) / 2)
    print(
        f"  normalizers: Z(D)={w.sum():.4f}  Z(D')={w2.sum():.4f}  Z(D')/Z(D)={w2.sum() / w.sum():.4f}  "
        f"e^0.5={math.exp(0.5):.4f}"
    )

    k = len(MODEL_SCORES)
    print(f"[Model selection: {k} models, scores 900 - r correct out of 1000]")
    for e in (0.1, 1.0, 5.0):
        p = em_probabilities(MODEL_SCORES, e)
        print(
            f"  eps={e:g}: Pr[best]={p[0]:.4f}  E[gap]={em_expected_gap(MODEL_SCORES, e):.2f}  "
            f"bound={em_expected_gap_bound(k, e):.2f}  "
            f"monotone E[gap]={em_expected_gap(MODEL_SCORES, e, monotone=True):.2f}"
        )
    c3 = em_gap_threshold(k, eps, 3)
    print(
        f"  eps=1, t=3: gap threshold={c3:.2f}  exact Pr[gap >= that]={em_gap_tail(MODEL_SCORES, eps, c3):.2e}  "
        f"bound e^-3={math.exp(-3):.4f}"
    )
    print(f"  t=3 gap threshold for k=10^6: {em_gap_threshold(10**6, eps, 3):.2f}")
    for gap in (10.0, 20.0):
        prof = tied_profile(k, gap)
        print(
            f"  tied profile, gap {gap:g}, eps=1: Pr[gap >= {gap:g}]={em_gap_tail(prof, eps, gap):.4f}  "
            f"bound={em_gap_bound(k, eps, gap):.4f}"
        )

    print("[Report-noisy-max on the model scores, eps = 1, 200000 runs]")
    m = 200_000
    p_em = em_probabilities(MODEL_SCORES, eps)
    p_lap2 = laplace_rnm_probabilities(MODEL_SCORES, 2 / eps)
    p_lap1 = laplace_rnm_probabilities(MODEL_SCORES, 1 / eps)
    u = np.asarray(MODEL_SCORES)
    gaps = u.max() - u
    print(f"  EM: Pr[best]={p_em[0]:.4f}  E[gap]={float(np.sum(p_em * gaps)):.2f}")
    print(
        f"  Laplace RNM, b=2/eps: Pr[best]={p_lap2[0]:.4f}  E[gap]={float(np.sum(p_lap2 * gaps)):.2f}  "
        f"(simulated E[gap] {rnm_expected_gap(rng, MODEL_SCORES, 2 / eps, m, 'laplace'):.2f})"
    )
    print(f"  Laplace RNM, b=1/eps (monotone): Pr[best]={p_lap1[0]:.4f}  E[gap]={float(np.sum(p_lap1 * gaps)):.2f}")
    print(
        f"  union bound at beta=0.05: b=2: {rnm_gap_bound(k, 2 / eps, 0.05):.2f}  "
        f"b=1: {rnm_gap_bound(k, 1 / eps, 0.05):.2f}"
    )

    print("[Report-noisy-max on the votes, Laplace b = 1/eps (monotone scores)]")
    pl, ql = laplace_rnm_probabilities(VOTES_D, 1 / eps), laplace_rnm_probabilities(VOTES_D2, 1 / eps)
    print(
        "  D: "
        + " ".join(f"{x:.4f}" for x in pl)
        + "  D': "
        + " ".join(f"{x:.4f}" for x in ql)
        + f"  max |ln ratio|={max_log_ratio(pl, ql):.4f}"
    )
    pl2, ql2 = laplace_rnm_probabilities(VOTES_D, 2 / eps), laplace_rnm_probabilities(VOTES_D2, 2 / eps)
    print(f"  b = 2/eps: max |ln ratio|={max_log_ratio(pl2, ql2):.4f}")

    print("[Gumbel-max against the exponential mechanism, scores (4, 3, 3, 1, 0), eps = 1, 1000000 runs]")
    n = 1_000_000
    p5 = em_probabilities(FIVE_SCORES, eps)
    g5 = frequencies(report_noisy_max(rng, FIVE_SCORES, 2 / eps, n, "gumbel"), len(FIVE_SCORES))
    l5 = laplace_rnm_probabilities(FIVE_SCORES, 2 / eps)
    print("  EM exact:        " + " ".join(f"{x:.4f}" for x in p5))
    print(
        "  Gumbel-max freq: "
        + " ".join(f"{x:.4f}" for x in g5)
        + f"  max abs diff={float(np.max(np.abs(g5 - p5))):.4f}"
    )
    print(
        "  Laplace RNM:     "
        + " ".join(f"{x:.4f}" for x in l5)
        + f"  TV to EM={0.5 * float(np.sum(np.abs(l5 - p5))):.4f}"
    )
    em2, lap2 = two_candidate_best(1.0)
    print(f"  two candidates, d/b = 1: EM Pr[best]={em2:.4f}  Laplace Pr[best]={lap2:.4f}")

    print("[AboveThreshold accuracy: k = 1000 queries, beta = 0.05, eps = 1]")
    alpha = svt_alpha(1000, 0.05, eps)
    print(
        f"  alpha={alpha:.2f}  empirical 95% quantile of 2 max(|rho|, max|nu|)="
        f"{svt_noise_quantile(rng, 1000, eps, 0.05, 10_000):.2f}"
    )
    stream = monitoring_stream()
    first = simulate_above_threshold(rng, stream, 500.0, 2 / eps, 4 / eps, 10_000)
    print(
        f"  monitoring stream (query 700 at 600, others at 400, T = 500): "
        f"Pr[output = 700]={float(np.count_nonzero(first == 700)) / len(first):.4f}"
    )
    print(f"  answering all 1000 with Laplace at eps/1000 each: noise scale {1000 / eps:g}")

    print("[AboveThreshold vs the no-halt variant on f(D) = (0^m, 1^m), f(D') = (1^m, 0^m), eps = 1, T = 0]")
    for mm in (1, 5, 10, 20, 40):
        print(
            f"  m={mm}: AboveThreshold max |ln ratio|={above_threshold_max_log_ratio(mm, eps):.4f}  "
            f"no-halt ln ratio of ⊥^m⊤^m={no_halt_log_ratio(mm, eps):.4f}  lower bound m eps/4={mm * eps / 4:.2f}"
        )
    fd20, fd20b, _ = svt_counterexample(20)
    gap20 = np.abs(
        above_threshold_log_probs(fd20, 0.0, 2 / eps, 4 / eps) - above_threshold_log_probs(fd20b, 0.0, 2 / eps, 4 / eps)
    )
    print(f"  m=20: AboveThreshold's worst output is stopping at query {int(np.argmax(gap20)) + 1}")
    for mm in (10, 40, 160):
        pd, pd2 = sign_test_rates(rng, mm, eps, 200_000)
        print(f"  sign test, m={mm}: Pr_D={pd:.4f}  Pr_D'={pd2:.4f}  ln ratio={math.log(pd / pd2):.2f}")
    fd, _, pattern = svt_counterexample(2)
    exact = math.exp(no_halt_pattern_log_prob(fd, pattern, 0.0, 2 / eps, 4 / eps))
    sim = simulate_no_halt(rng, fd, 0.0, 2 / eps, 4 / eps, 400_000)
    hits = float(np.mean(np.all(sim == np.asarray(pattern)[None, :], axis=1)))
    print(f"  check, m=2, Pr_D[⊥⊥⊤⊤]: exact={exact:.4f}  simulated={hits:.4f}")
    a_exact = np.exp(above_threshold_log_probs(fd, 0.0, 2 / eps, 4 / eps))
    a_sim = frequencies(simulate_above_threshold(rng, fd, 0.0, 2 / eps, 4 / eps, 400_000), 5)
    print(
        "  check, m=2, AboveThreshold output law: exact "
        + " ".join(f"{x:.4f}" for x in a_exact)
        + "  simulated "
        + " ".join(f"{x:.4f}" for x in a_sim)
    )

    print("[AboveThreshold tightness, more digits]")
    for mm in (10, 20, 40):
        print(f"  m={mm}: max |ln ratio|={above_threshold_max_log_ratio(mm, eps):.6f}")

    print("[Sparse: c answers above, k = 1000 queries, beta = 0.05, eps = 1]")
    for c in (10, 100):
        print(
            f"  c={c}: pure margin={sparse_alpha(1000, c, 0.05, eps):.0f}  "
            f"advanced (delta=1e-6) per-run eps={sparse_run_epsilon(eps, c, 1e-6):.4f} "
            f"margin={sparse_alpha_approx(1000, c, 0.05, eps, 1e-6):.0f}"
        )

    print("[Learning a threshold with the exponential mechanism: |C| = 1001, label noise 0.1, 200 datasets]")
    for nn in (100, 1000, 10000):
        stat, priv = learner_bound(nn, eps, 0.05)
        print(
            f"  n={nn}: excess error EM eps=1={learner_excess(rng, nn, 1.0, 200):.4f}  "
            f"eps=0.1={learner_excess(rng, nn, 0.1, 200):.4f}  ERM={learner_excess(rng, nn, None, 200):.4f}  "
            f"bound terms: sampling={stat:.4f} privacy(eps=1)={priv:.4f}"
        )
    for a, e in ((0.05, 1.0), (0.05, 0.01)):
        stat_n, priv_n = learner_samples(a, e, 0.05)
        print(f"  alpha={a:g} eps={e:g}: sampling needs n={stat_n:.0f}  privacy needs n={priv_n:.0f}")

    print("[Private median of 1000 gradient norms in [0, 10]]")
    norms = gradient_norms(rng, 1000)
    xs = np.sort(norms)
    print(
        f"  true median={float(np.median(norms)):.4f}  45th-55th percentile=[{xs[449]:.4f}, {xs[549]:.4f}]  "
        f"max={xs[-1]:.3f}"
    )
    for e in (0.1, 1.0):
        print(
            f"  eps={e:g}: Pr[rank gap >= 100]={median_rank_tail(norms, e, 100):.2e}  "
            f"E|output - median|={median_expected_abs_error(norms, e):.4f}  "
            f"bound on rank gap (t=ln 20)={median_rank_bound(norms, e, math.log(20)):.1f}"
        )
    draws = [sample_median(rng, norms, 1.0) for _ in range(20_000)]
    print(
        f"  eps=1, 20000 draws: mean |output - median|={float(np.mean(np.abs(np.asarray(draws) - np.median(norms)))):.4f}"
    )
    print(
        f"  eps=1, range B=1000 instead of 10: E|output - median|={median_expected_abs_error(norms, 1.0, 1000.0):.4f}  "
        f"bound={median_rank_bound(norms, 1.0, math.log(20), 1000.0):.1f}"
    )


# --- Figures ------------------------------------------------------------------------------------


def plot_em_probs() -> Figure:
    fig, (left, right) = plt.subplots(1, 2, figsize=(11, 4))
    ranks = np.arange(len(MODEL_SCORES))
    left.bar(ranks, MODEL_SCORES, color="0.6")
    left.set(
        xlabel="model r (sorted by score)",
        ylabel="u(D, r): correct out of 1000",
        ylim=(840, 905),
        title="Scores: model r gets 900 - r right",
    )
    for e, style in ((0.1, "-"), (0.5, "--"), (1.0, "-."), (5.0, ":")):
        right.plot(
            ranks, em_probabilities(MODEL_SCORES, e), style, marker="o", markersize=3, label=rf"$\varepsilon$ = {e:g}"
        )
    right.axhline(1 / len(MODEL_SCORES), color="0.6", linewidth=0.8)
    right.text(30, 1 / len(MODEL_SCORES) + 0.01, r"uniform ($\varepsilon \to 0$)", fontsize=9, color="0.4")
    right.set(
        xlabel="model r",
        ylabel="Pr[M(D) = r]",
        yscale="linear",
        ylim=(0, 1),
        title=r"Exponential mechanism: Pr $\propto e^{\varepsilon u / 2\Delta}$",
    )
    right.legend(fontsize=9)
    return fig


def plot_em_utility() -> Figure:
    fig, (left, right) = plt.subplots(1, 2, figsize=(11, 4.2))
    k = len(MODEL_SCORES)
    cs = [i / 4 for i in range(4 * 50)]
    for e, color in ((0.5, "C0"), (1.0, "C1")):
        left.plot(
            cs, [em_gap_tail(MODEL_SCORES, e, c) for c in cs], color=color, label=rf"exact, $\varepsilon$ = {e:g}"
        )
        left.plot(
            cs,
            [em_gap_bound(k, e, c) for c in cs],
            color=color,
            linestyle="--",
            label=rf"bound $|\mathcal{{R}}| e^{{-\varepsilon c/2}}$, $\varepsilon$ = {e:g}",
        )
    left.set(
        yscale="log",
        ylim=(1e-8, 2),
        xlim=(0, 50),
        xlabel="gap c (correct predictions)",
        ylabel="Pr[OPT - u(D, M(D)) ≥ c]",
        title="Scores 900 - r: the bound is loose",
    )
    left.legend(fontsize=8, loc="lower left")
    gaps = [i / 4 for i in range(4 * 50 + 1)]
    right.plot(gaps, [em_gap_tail(tied_profile(k, g), 1.0, g) for g in gaps], label="exact Pr[not the best]")
    right.plot(gaps, [em_gap_bound(k, 1.0, g) for g in gaps], "--", label=r"bound $|\mathcal{R}| e^{-\varepsilon g/2}$")
    right.set(
        yscale="log",
        ylim=(1e-8, 2),
        xlim=(0, 50),
        xlabel="gap g of the 49 tied runners-up",
        ylabel="Pr[OPT - u ≥ g]",
        title=r"One winner, 49 tied at OPT - g ($\varepsilon$ = 1): tight",
    )
    right.legend(fontsize=9, loc="lower left")
    return fig


def plot_rnm_threshold(b: float = 1.0, t: float = 0.7, s: float = 1.0) -> Figure:
    z = np.linspace(-5, 5, 1001)
    dens = np.exp(laplace_log_pdf(z, b))
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(z, dens, "k", linewidth=1.5, label=r"density of $Z_i \sim \mathrm{Lap}(b)$")
    ax.fill_between(
        z,
        dens,
        where=z > t - s,
        color="C1",
        alpha=0.25,
        label=f"wins on D': Pr[Z > t(D')] = {laplace_sf(t - s, b):.3f}",
    )
    ax.fill_between(
        z, dens, where=z > t, color="C0", alpha=0.45, label=f"wins on D: Pr[Z > t(D)] = {laplace_sf(t, b):.3f}"
    )
    ax.axvline(t, color="C0", linestyle="--")
    ax.axvline(t - s, color="C1", linestyle="--")
    ax.annotate("", xy=(t - s, 0.12), xytext=(t, 0.12), arrowprops={"arrowstyle": "<->"})
    ax.text(t - s / 2, 0.135, r"shift $\leq s$", ha="center", fontsize=9)
    ax.set(
        xlabel=r"noise $z$ on candidate $i$ (others' noise fixed)",
        ylabel="density",
        ylim=(0, 0.55),
        title=rf"Ratio {laplace_shift_ratio(t, s, b):.3f} ≤ $e^{{s/b}}$ = {math.exp(s / b):.3f}",
    )
    ax.legend(fontsize=9, loc="upper left")
    return fig


def plot_gumbel_em(rng: np.random.Generator, eps: float = 1.0, n: int = 200_000) -> Figure:
    k = len(FIVE_SCORES)
    p = em_probabilities(FIVE_SCORES, eps)
    g = frequencies(report_noisy_max(rng, FIVE_SCORES, 2 / eps, n, "gumbel"), k)
    lap = laplace_rnm_probabilities(FIVE_SCORES, 2 / eps)
    x = np.arange(k)
    fig, ax = plt.subplots(figsize=(8, 4))
    width = 0.27
    ax.bar(x - width, p, width, label="exponential mechanism (exact)")
    ax.bar(x, g, width, label=f"Gumbel noisy max ({n:,} runs)")
    ax.bar(x + width, lap, width, label="Laplace noisy max (exact)")
    ax.set_xticks(x, [f"u = {u:g}" for u in FIVE_SCORES])
    ax.set(
        ylabel="probability of being selected",
        ylim=(0, 0.5),
        title=r"Scale $2\Delta/\varepsilon$, $\varepsilon$ = 1: Gumbel matches EM, Laplace does not quite",
    )
    ax.legend(fontsize=9)
    return fig


def plot_svt_path(rng: np.random.Generator, eps: float = 1.0, threshold: float = 20.0) -> Figure:
    m = 30
    f = np.array([8 + 0.5 * i for i in range(m)])
    rho = float(rng.laplace(0.0, 2 / eps))
    nu = rng.laplace(0.0, 4 / eps, m)
    noisy = f + nu
    above = noisy >= threshold + rho
    stop = int(np.argmax(above)) if bool(above.any()) else m
    idx = np.arange(m)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(idx, f, color="0.5", linewidth=1, label=r"true answers $f_i(D)$")
    ax.axhline(threshold, color="0.5", linestyle=":", label=f"threshold T = {threshold:g}")
    ax.axhline(threshold + rho, color="C3", linestyle="--", label=r"noisy threshold $\tau + \rho$")
    ax.plot(idx[:stop], noisy[:stop], "o", color="C0", label=r"$f_i + \nu_i < \tau + \rho$: answer $\bot$")
    if stop < m:
        ax.plot(
            [stop],
            [noisy[stop]],
            "*",
            color="C1",
            markersize=14,
            label=r"first $f_i + \nu_i \geq \tau + \rho$: answer $\top$, halt",
        )
        ax.plot(idx[stop + 1 :], noisy[stop + 1 :], "o", color="0.85", label="never looked at")
    ax.set(xlabel="query i", ylabel="value", title="AboveThreshold: one noisy threshold, fresh noise per query")
    ax.legend(fontsize=8, loc="upper left")
    return fig


def plot_svt_buggy(rng: np.random.Generator, eps: float = 1.0) -> Figure:
    fig, (left, right) = plt.subplots(1, 2, figsize=(11, 4.2))
    ms = [1, 2, 4, 6, 8, 10, 15, 20, 25, 30, 40]
    left.plot(
        ms, [no_halt_log_ratio(m, eps) for m in ms], "o-", label=r"no halt: $\ln$ ratio of $\bot^m\top^m$ (exact)"
    )
    left.plot(ms, [m * eps / 4 for m in ms], "--", color="C0", label=r"lower bound $m\varepsilon/4$")
    left.plot(
        ms, [above_threshold_max_log_ratio(m, eps) for m in ms], "s-", label="AboveThreshold: worst output (exact)"
    )
    left.axhline(eps, color="C3", linestyle=":", label=r"$\varepsilon$ = 1")
    left.set(
        xlabel="m (the stream has 2m queries)",
        ylabel="privacy loss (nats)",
        title="Exact privacy loss on one neighbouring pair",
    )
    left.legend(fontsize=8, loc="upper left")
    msim = [5, 10, 20, 40, 80, 160, 320]
    rates = [sign_test_rates(rng, m, eps, 100_000) for m in msim]
    right.plot(msim, [r[0] for r in rates], "o-", label=r"Pr[more $\top$ in 2nd half | D]")
    right.plot(msim, [r[1] for r in rates], "s-", label=r"Pr[more $\top$ in 2nd half | D']")
    right.plot(
        msim,
        [min(1.0, math.e * r[1]) for r in rates],
        ":",
        color="C1",
        label=r"$e^{\varepsilon}\,\cdot$ Pr[... | D'], the most ε-DP allows for D",
    )
    right.set(
        xscale="log",
        xlabel="m",
        ylabel="probability (100,000 runs each)",
        ylim=(0, 1.3),
        title="A simple test breaks the no-halt variant",
    )
    right.legend(fontsize=8, loc="upper left")
    return fig


def plot_em_applications(rng: np.random.Generator) -> Figure:
    fig, (left, right) = plt.subplots(1, 2, figsize=(11, 4.2))
    ns = [100, 300, 1000, 3000, 10000]
    for eps, style, label in (
        (0.1, "o-", r"exponential mechanism, $\varepsilon$ = 0.1"),
        (1.0, "s-", r"exponential mechanism, $\varepsilon$ = 1"),
        (None, "^-", "non-private: fewest training mistakes"),
    ):
        vals = [learner_excess(rng, n, eps, 100) for n in ns]
        keep = [i for i, v in enumerate(vals) if v > 0]  # the log axis cannot show an exact 0
        left.plot([ns[i] for i in keep], [vals[i] for i in keep], style, label=label)
    left.set(
        xscale="log",
        yscale="log",
        xlabel="training records n",
        ylabel="excess error over the best threshold",
        title="Choosing one of 1001 threshold classifiers",
    )
    left.legend(fontsize=8)
    norms = gradient_norms(rng, 1000)
    right.hist(norms, bins=np.linspace(0.5, 1.5, 41), density=True, color="0.8", label="gradient norms (histogram)")
    grid = np.linspace(0.5, 1.5, 4001)
    left_ends, lengths, _ = median_intervals(norms, NORM_RANGE)
    for eps, color in ((0.1, "C0"), (1.0, "C1")):
        probs = median_interval_probs(norms, eps)
        with np.errstate(divide="ignore", invalid="ignore"):
            dens_int = np.where(lengths > 0, probs / np.where(lengths > 0, lengths, 1.0), 0.0)
        idx = np.clip(np.searchsorted(left_ends, grid, side="right") - 1, 0, len(lengths) - 1)
        right.plot(
            grid, dens_int[idx], color=color, linewidth=1, label=rf"density of the output, $\varepsilon$ = {eps:g}"
        )
    right.axvline(float(np.median(norms)), color="C3", linestyle=":", label="true median")
    right.set(xlabel="gradient norm", ylabel="density", xlim=(0.5, 1.5), title="Private median over [0, 10], zoomed in")
    right.set_yscale("log")
    right.set_ylim(1e-2, 1e3)
    right.legend(fontsize=8)
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
        "select_em_probs.svg": plot_em_probs(),
        "select_em_utility.svg": plot_em_utility(),
        "select_rnm_threshold.svg": plot_rnm_threshold(),
        "select_gumbel_em.svg": plot_gumbel_em(rng),
        "select_svt_path.svg": plot_svt_path(rng),
        "select_svt_buggy.svg": plot_svt_buggy(rng),
        "select_em_applications.svg": plot_em_applications(rng),
    }
    for name, fig in figures.items():
        fig.tight_layout()
        fig.savefig(out / name, metadata={"Date": None})  # no timestamp, so output is reproducible
        plt.close(fig)
        print(f"saved {out / name}")


if __name__ == "__main__":
    main()
