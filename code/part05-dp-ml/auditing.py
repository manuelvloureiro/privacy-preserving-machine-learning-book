"""Check the numbers in the auditing chapter and draw its figures.

    uv run python code/part05-dp-ml/auditing.py
    uv run python code/part05-dp-ml/auditing.py --seed 1 --out /tmp/figures

Prints the numbers quoted in 10-auditing.md (they match exactly with the default seed), then saves the
chapter's figures.

An audit runs an attack against a training algorithm and turns the attack's error rates into a
lower bound on ε that holds with probability at least 1 - β. Four audits are simulated:

- the white-box audit of DP-SGD's mechanism: a "Dirac" gradient canary of norm C on a coordinate no
  other record touches, observed through every noisy sum. Each step's observation, in units of C, is
  N(0, σ²) without the canary and the mixture (1 - q) N(0, σ²) + q N(1, σ²) with it, so trials are
  cheap to simulate without training a network;
- the one-run audit of Steinke, Nasr & Jagielski (2023): m canaries, each included with probability
  1/2, and a bound on the number of correct guesses;
- the black-box audit of the running example's network (dp_sgd.py): mislabeled canaries inserted
  into the 10,000 training records, scored by their loss on the final model;
- per-step audits of three implementations, one correct and two buggy.

The running example of the part: q = 0.01, σ = 1, C = 0.5, 10,000 steps, ε = 6.71 at δ = 1e-5.
"""

import os

# The models are tiny; multi-threaded BLAS only adds overhead.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import argparse
import math
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from statistics import NormalDist

import matplotlib

matplotlib.use("Agg")
matplotlib.rcParams["svg.hashsalt"] = "auditing"  # stable element ids, so reruns don't churn the SVGs
matplotlib.rcParams["svg.fonttype"] = "none"  # keep text as text: smaller, diffable files

import matplotlib.pyplot as plt
import numpy as np
from dp_sgd import (
    CLIP_RUN,
    DELTA_RUN,
    N_TRAIN,
    Q_RUN,
    SIGMA_RUN,
    STEPS_RUN,
    Dataset,
    DpsgdConfig,
    Params,
    Task,
    accuracy,
    bisect,
    dpsgd_epsilon,
    gaussian_delta,
    logistic_losses,
    make_task,
    train_dpsgd,
)
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle
from numpy.typing import NDArray

# Figures are build output: _build/figures/<part>/, which the build copies into each book.
FIGURES = Path(__file__).resolve().parents[2] / "_build" / "figures" / Path(__file__).parent.name

Array = NDArray[np.float64]
Mask = NDArray[np.bool_]

BETA = 0.05  # every lower bound holds with probability at least 1 - β = 95%
NORMAL = NormalDist()
POOL_TRIALS = 50_000  # white-box trials per hypothesis for the table
FIG_POOL_TRIALS = 30_000  # and for the figures
CANARIES = 1_000  # canaries of the black-box one-run audit
GUESS_GRID: tuple[int, ...] = (20, 50, 100, 200, 500, 1000)  # numbers of guesses r tried (Bonferroni)
Q_MISCOUNTED = 0.005  # the sampling rate an accountant computes when told n = 20,000 instead of 10,000


# --- Toolkit: binomial tails and Clopper–Pearson intervals ------------------------------------------


def _beta_continued_fraction(a: float, b: float, x: float) -> float:
    """The continued fraction of the incomplete beta function (modified Lentz's method)."""
    tiny = 1e-300
    c, d = 1.0, 1.0 - (a + b) * x / (a + 1.0)
    d = 1.0 / (d if abs(d) > tiny else tiny)
    h = d
    for m in range(1, 100_000):
        m2 = 2 * m
        for aa in (m * (b - m) * x / ((a - 1 + m2) * (a + m2)), -(a + m) * (a + b + m) * x / ((a + m2) * (a + 1 + m2))):
            d = 1.0 + aa * d
            d = 1.0 / (d if abs(d) > tiny else tiny)
            c = 1.0 + aa / c
            c = c if abs(c) > tiny else tiny
            h *= d * c
        if abs(d * c - 1.0) < 1e-15:
            break
    return h


def betainc(a: float, b: float, x: float) -> float:
    """The regularized incomplete beta function I_x(a, b)."""
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    log_front = math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b) + a * math.log(x) + b * math.log1p(-x)
    if x < (a + 1.0) / (a + b + 2.0):
        return math.exp(log_front) * _beta_continued_fraction(a, b, x) / a
    return 1.0 - math.exp(log_front) * _beta_continued_fraction(b, a, 1.0 - x) / b


def binom_sf(k: int, n: int, p: float) -> float:
    """P[Bin(n, p) ≥ k] = I_p(k, n - k + 1)."""
    if k <= 0:
        return 1.0
    if k > n:
        return 0.0
    return betainc(k, n - k + 1, p)


def cp_upper(k: int, n: int, a: float) -> float:
    """One-sided Clopper–Pearson upper bound: the rate u with P[Bin(n, u) ≤ k] = a."""
    if k >= n:
        return 1.0
    return bisect(lambda p: 1.0 - binom_sf(k + 1, n, p) <= a, 0.0, 1.0, 60)


def cp_lower(k: int, n: int, a: float) -> float:
    """One-sided Clopper–Pearson lower bound: the rate l with P[Bin(n, l) ≥ k] = a (rounded down)."""
    if k <= 0:
        return 0.0
    lo, hi = 0.0, 1.0
    for _ in range(60):
        mid = (lo + hi) / 2
        if binom_sf(k, n, mid) >= a:
            hi = mid
        else:
            lo = mid
    return lo


# --- From an attack to a lower bound on ε ------------------------------------------------------------


def max_tpr(fpr: float, eps: float, delta: float) -> float:
    """The largest TPR any test can have at this FPR against an (ε, δ)-DP mechanism (both inequalities)."""
    return min(1.0, math.exp(eps) * fpr + delta, 1.0 - math.exp(-eps) * (1.0 - delta - fpr))


def eps_from_rates(fpr: float, tpr: float, delta: float) -> float:
    """The smallest ε compatible with one test's (FPR, TPR) at this δ: TPR ≤ e^ε FPR + δ and its mirror."""
    best = 0.0
    for num, den in ((tpr - delta, fpr), (1.0 - fpr - delta, 1.0 - tpr)):
        if num > 0:
            best = max(best, math.inf if den <= 0 else math.log(num / den))
    return best


def eps_lower_bound(fp: int, n_out: int, tp: int, n_in: int, delta: float, beta: float = BETA) -> float:
    """ε lower bound at confidence 1 - β: Clopper–Pearson at β/2 on each rate, then eps_from_rates."""
    return eps_from_rates(cp_upper(fp, n_out, beta / 2), cp_lower(tp, n_in, beta / 2), delta)


def perfect_attack_eps(n: int, delta: float, beta: float = BETA) -> float:
    """The most an audit with n trials per hypothesis can certify: no false positive, no false negative."""
    return eps_lower_bound(0, n, n, n, delta, beta)


def trials_needed(eps: float, delta: float, beta: float = BETA) -> int:
    """Smallest n per hypothesis with which even a perfect attack certifies ε."""
    hi = 1
    while perfect_attack_eps(hi, delta, beta) < eps:
        hi *= 2
    lo = hi // 2
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if perfect_attack_eps(mid, delta, beta) >= eps:
            hi = mid
        else:
            lo = mid
    return hi


def gdp_mu_from_rates(fpr: float, tpr: float) -> float:
    """The smallest μ for which one test's (FPR, TPR) is allowed under μ-GDP: Φ⁻¹(TPR) − Φ⁻¹(FPR)."""
    if tpr <= 0.0 or fpr >= 1.0:
        return 0.0
    if fpr <= 0.0 or tpr >= 1.0:
        return math.inf
    return max(0.0, NORMAL.inv_cdf(tpr) - NORMAL.inv_cdf(fpr))


def gdp_mu_lower_bound(fp: int, n_out: int, tp: int, n_in: int, beta: float = BETA) -> float:
    return gdp_mu_from_rates(cp_upper(fp, n_out, beta / 2), cp_lower(tp, n_in, beta / 2))


def gdp_eps(mu: float, delta: float) -> float:
    """ε at δ of μ-GDP, solving the Gaussian privacy profile."""
    return bisect(lambda e: gaussian_delta(e, mu) <= delta, 0.0, 200.0, 80)


def gdp_touching_point(mu: float, eps: float) -> tuple[float, float]:
    """(FPR, TPR) of the test where the line TPR = e^ε FPR + δ touches the μ-GDP curve."""
    return NORMAL.cdf(-eps / mu - mu / 2), NORMAL.cdf(-eps / mu + mu / 2)


def clt_mu(q: float, sigma: float, steps: int) -> float:
    """The central-limit μ of DP-SGD: q √(T (e^{1/σ²} − 1))."""
    return q * math.sqrt(steps * math.expm1(1 / sigma**2))


# --- Choosing a threshold on one half of the trials, measuring on the other ---------------------------


@dataclass(frozen=True)
class Counts:
    fp: int
    n_out: int
    tp: int
    n_in: int

    @property
    def fpr(self) -> float:
        return self.fp / self.n_out

    @property
    def tpr(self) -> float:
        return self.tp / self.n_in


def counts_at(scores_in: Array, scores_out: Array, threshold: float) -> Counts:
    """Say "canary in" when the score is at least the threshold."""
    return Counts(
        int(np.sum(scores_out >= threshold)), scores_out.size, int(np.sum(scores_in >= threshold)), scores_in.size
    )


def candidate_thresholds(scores_in: Array, scores_out: Array, count: int = 30) -> list[float]:
    """Thresholds that let through 0, 1, 2, 4, ... false positives, or 0, 1, 2, 4, ... false negatives."""
    out_desc = np.sort(scores_out)[::-1]
    in_asc = np.sort(scores_in)
    ranks = sorted({0} | {int(r) for r in np.geomspace(1, max(1, out_desc.size // 2), count)})
    thresholds = [float(np.nextafter(out_desc[0], np.inf))]
    thresholds += [float((out_desc[r - 1] + out_desc[r]) / 2) for r in ranks if 0 < r < out_desc.size]
    ranks_in = sorted({int(r) for r in np.geomspace(1, max(1, in_asc.size // 2), count)})
    thresholds += [float(in_asc[0])] + [float((in_asc[r - 1] + in_asc[r]) / 2) for r in ranks_in if r < in_asc.size]
    return thresholds


@dataclass(frozen=True)
class Audit:
    threshold: float
    counts: Counts  # on the evaluation half
    eps_lo: float  # rigorous (ε, δ) lower bound at 1 - β
    mu_threshold: float
    mu_counts: Counts
    mu_lo: float  # rigorous lower bound on μ: not μ-GDP for any smaller μ, at 1 - β


def run_audit(scores_in: Array, scores_out: Array, delta: float, beta: float = BETA) -> Audit:
    """Pick the thresholds on the first half of each sample, then bound ε and μ on the second half."""
    h_in, h_out = scores_in.size // 2, scores_out.size // 2
    sel_in, ev_in = scores_in[:h_in], scores_in[h_in:]
    sel_out, ev_out = scores_out[:h_out], scores_out[h_out:]
    candidates = candidate_thresholds(sel_in, sel_out)

    def eps_of(thr: float) -> float:
        c = counts_at(sel_in, sel_out, thr)
        return eps_lower_bound(c.fp, c.n_out, c.tp, c.n_in, delta, beta)

    def mu_of(thr: float) -> float:
        c = counts_at(sel_in, sel_out, thr)
        return gdp_mu_lower_bound(c.fp, c.n_out, c.tp, c.n_in, beta)

    thr_eps = max(candidates, key=eps_of)
    thr_mu = max(candidates, key=mu_of)
    c_eps = counts_at(ev_in, ev_out, thr_eps)
    c_mu = counts_at(ev_in, ev_out, thr_mu)
    return Audit(
        thr_eps,
        c_eps,
        eps_lower_bound(c_eps.fp, c_eps.n_out, c_eps.tp, c_eps.n_in, delta, beta),
        thr_mu,
        c_mu,
        gdp_mu_lower_bound(c_mu.fp, c_mu.n_out, c_mu.tp, c_mu.n_in, beta),
    )


# --- White-box audit of the mechanism: a Dirac canary seen through every noisy sum ----------------------


def lr_statistic_inplace(x: NDArray[np.float32], q: float, sigma: float) -> Array:
    """Σ_t ln(1 − q + q e^{(x_t − 1/2)/σ²}) over each row: the Neyman–Pearson statistic. Overwrites x."""
    x -= 0.5
    x /= sigma**2
    np.exp(x, out=x)
    x *= q
    x += 1.0 - q
    np.log(x, out=x)
    return x.sum(axis=1, dtype=np.float64)


def simulate_lr_scores(
    rng: np.random.Generator,
    n: int,
    inside: bool,
    q: float = Q_RUN,
    sigma: float = SIGMA_RUN,
    steps: int = STEPS_RUN,
    chunk: int = 1000,
) -> Array:
    """n white-box trials of the likelihood-ratio statistic, with or without the canary.

    Each step's projection on the canary direction, in units of C, is σ z_t, plus 1 when the canary
    was sampled. The statistic is a sum over steps, so which steps sampled the canary does not matter:
    the canary is added to the first K steps, K ~ Bin(T, q).
    """
    parts: list[Array] = []
    for start in range(0, n, chunk):
        b = min(chunk, n - start)
        x = rng.standard_normal((b, steps), dtype=np.float32)
        x *= sigma
        if inside:
            k = rng.binomial(steps, q, size=b)
            x += np.arange(steps)[None, :] < k[:, None]
        parts.append(lr_statistic_inplace(x, q, sigma))
    return np.concatenate(parts)


def simulate_sum_scores(
    rng: np.random.Generator,
    n: int,
    inside: bool,
    q: float = Q_RUN,
    noise: float = SIGMA_RUN,
    steps: int = STEPS_RUN,
) -> Array:
    """n white-box trials of the sum of projections Σ_t x_t: K + noise √T z, exactly, K ~ Bin(T, q)."""
    k = rng.binomial(steps, q, size=n).astype(np.float64) if inside else np.zeros(n)
    return k + noise * math.sqrt(steps) * rng.standard_normal(n)


def pointwise_eps(
    scores_in: Array, scores_out: Array, fps: Sequence[int], delta: float
) -> list[tuple[int, float, float, float]]:
    """(false positives, FPR, TPR, ln((TPR − δ)/FPR)) at thresholds letting through `fps` false positives."""
    out_desc = np.sort(scores_out)[::-1]
    rows: list[tuple[int, float, float, float]] = []
    for k in fps:
        thr = float((out_desc[k - 1] + out_desc[k]) / 2)
        c = counts_at(scores_in, scores_out, thr)
        rows.append((k, c.fpr, c.tpr, math.log((c.tpr - delta) / c.fpr) if c.tpr > delta else 0.0))
    return rows


# --- One-run auditing (Steinke, Nasr & Jagielski 2023, Corollary 5.4) -------------------------------------


def one_run_pvalue(m: int, r: int, v: int, eps: float, delta: float) -> float:
    """P-value of v or more correct guesses out of r, among m canaries, if the algorithm is (ε, δ)-DP.

    P[W ≥ v] ≤ f(v) + 2mδ max_i (f(v − i) − f(v))/i with f(v) = P[Bin(r, e^ε/(1 + e^ε)) ≥ v].
    """
    if v <= 0:
        return 1.0
    p = 1.0 / (1.0 + math.exp(-eps))
    j = np.arange(r + 1)
    log_choose = np.array([math.lgamma(r + 1) - math.lgamma(i + 1) - math.lgamma(r - i + 1) for i in range(r + 1)])
    log_p = math.log(p)
    log_1mp = math.log1p(-p) if p < 1 else -math.inf
    pmf = np.exp(log_choose + j * log_p + (r - j) * log_1mp)
    tail = float(pmf[v:].sum())
    below = np.cumsum(pmf[v - 1 :: -1])  # P[v − i ≤ W < v] for i = 1, ..., v
    alpha = float(np.max(below / np.arange(1, v + 1)))
    return min(1.0, tail + 2 * m * delta * alpha)


def one_run_eps(m: int, r: int, v: int, delta: float, beta: float = BETA) -> float:
    """The largest ε rejected at level β: the algorithm is not (ε, δ)-DP for any smaller ε."""
    if one_run_pvalue(m, r, v, 0.0, delta) >= beta:
        return 0.0
    hi = 1.0
    while one_run_pvalue(m, r, v, hi, delta) < beta:
        hi *= 2
        if hi > 64:
            return hi
    lo = 0.0
    for _ in range(40):
        mid = (lo + hi) / 2
        if one_run_pvalue(m, r, v, mid, delta) < beta:
            lo = mid
        else:
            hi = mid
    return lo


def one_run_guesses(scores: Array, included: Mask, r: int) -> int:
    """Guess "in" for the r/2 highest scores and "out" for the r/2 lowest; return the number correct."""
    order = np.argsort(-scores, kind="stable")
    top, bottom = order[: r // 2], order[scores.size - r // 2 :]
    return int(np.sum(included[top]) + np.sum(~included[bottom]))


@dataclass(frozen=True)
class OneRun:
    rows: list[tuple[int, int, float]]  # (guesses r, correct v, ε lower bound with Bonferroni over the grid)

    @property
    def best(self) -> tuple[int, int, float]:
        return max(self.rows, key=lambda row: row[2])


def one_run_audit(scores: Array, included: Mask, delta: float, grid: Sequence[int] = GUESS_GRID) -> OneRun:
    m = scores.size
    beta = BETA / len(grid)  # the auditor tries every r in the grid, so each test runs at β / |grid|
    rows: list[tuple[int, int, float]] = []
    for r in grid:
        if r > m:
            continue
        v = one_run_guesses(scores, included, r)
        rows.append((r, v, one_run_eps(m, r, v, delta, beta)))
    return OneRun(rows)


def idealized_one_run(rng: np.random.Generator, m: int) -> tuple[Array, Mask]:
    """m Dirac canaries on distinct untouched coordinates, each included with probability 1/2, in one run."""
    included = rng.random(m) < 0.5
    scores = np.empty(m)
    scores[included] = simulate_lr_scores(rng, int(included.sum()), True)
    scores[~included] = simulate_lr_scores(rng, int((~included).sum()), False)
    return scores, included


# --- Black-box audit of the running example: mislabeled canaries, one run -----------------------------


def mislabeled_canaries(task: Task, m: int = CANARIES) -> Dataset:
    """m held-out test records with their labels flipped."""
    return Dataset(task.test.x[:m], -task.test.y[:m])


def remaining_test(task: Task, m: int = CANARIES) -> Dataset:
    return Dataset(task.test.x[m:], task.test.y[m:])


@dataclass(frozen=True)
class BlackBoxRun:
    params: Params
    included: Mask
    scores: Array  # minus each canary's loss: high means "trained on"
    test_acc: float


def black_box_run(rng: np.random.Generator, task: Task, canaries: Dataset, cfg: DpsgdConfig) -> BlackBoxRun:
    """Include each canary with probability 1/2, train once, score every canary by its loss."""
    included = rng.random(canaries.y.size) < 0.5
    data = Dataset(
        np.vstack([task.train.x, canaries.x[included]]), np.concatenate([task.train.y, canaries.y[included]])
    )
    params, _ = train_dpsgd(rng, data, cfg)
    scores = -logistic_losses(params, canaries)
    return BlackBoxRun(params, included, scores, accuracy(params, remaining_test(task, canaries.y.size)))


# --- Finding bugs: per-step audits of three implementations -------------------------------------------


@dataclass(frozen=True)
class Implementation:
    name: str
    noise: float  # standard deviation of the noise on the clipped sum, in units of C
    title: str  # for the figure


IMPLEMENTATIONS: tuple[Implementation, ...] = (
    Implementation("correct: noise sigma*C on the sum", SIGMA_RUN, r"correct: noise $\sigma C$"),
    Implementation(
        "variance passed as standard deviation", SIGMA_RUN**2 * CLIP_RUN, r"bug: noise $\sigma^2 C^2 = 0.5\,C$"
    ),
    Implementation(
        "noise scaled for the mean, added to the sum",
        SIGMA_RUN / (Q_RUN * N_TRAIN),
        r"bug: noise $\sigma C / B = 0.01\,C$",
    ),
)


def per_step_observations(rng: np.random.Generator, steps: int, noise: float, inside: bool) -> Array:
    """One run's per-step projections on a Dirac canary inserted at every step (q_c = 1), in units of C."""
    return (1.0 if inside else 0.0) + noise * rng.standard_normal(steps)


def per_step_audit(rng: np.random.Generator, impl: Implementation, steps: int = STEPS_RUN) -> Audit:
    """Two runs, one with the canary at every step: T observations per hypothesis."""
    obs_in = per_step_observations(rng, steps, impl.noise, True)
    obs_out = per_step_observations(rng, steps, impl.noise, False)
    return run_audit(obs_in, obs_out, 0.0)


# --- The chapter's numbers ----------------------------------------------------------------------------


def print_table(rng: np.random.Generator) -> None:
    eps_claim = dpsgd_epsilon(Q_RUN, SIGMA_RUN, STEPS_RUN, DELTA_RUN)
    print("== The claim")
    print(f"accountant: q=0.01, sigma=1, T=10000, delta=1e-5: eps={eps_claim:.2f}")

    print("== Clopper-Pearson")
    print(
        f"k=10, n=10000: 97.5% one-sided bounds [{cp_lower(10, 10_000, 0.025):.5f}, {cp_upper(10, 10_000, 0.025):.5f}]"
    )
    print(f"k=0, n=3000: 95% upper bound {cp_upper(0, 3000, 0.05):.4e}")

    print("== From rates to a bound")
    print(f"point estimate fpr=0.001, tpr=0.5: eps={eps_from_rates(0.001, 0.5, DELTA_RUN):.2f}")
    fpr_hi, tpr_lo = cp_upper(1, 1000, BETA / 2), cp_lower(500, 1000, BETA / 2)
    print(
        f"fp=1/1000, tp=500/1000: fpr_hi={fpr_hi:.5f} tpr_lo={tpr_lo:.4f} "
        f"eps_lo={eps_lower_bound(1, 1000, 500, 1000, DELTA_RUN):.2f}"
    )
    fpr_hi, tpr_lo = cp_upper(10, 10_000, BETA / 2), cp_lower(5000, 10_000, BETA / 2)
    print(
        f"fp=10/10000, tp=5000/10000: fpr_hi={fpr_hi:.5f} tpr_lo={tpr_lo:.4f} "
        f"eps_lo={eps_lower_bound(10, 10_000, 5000, 10_000, DELTA_RUN):.2f}"
    )
    for n in (100, 1000, 10_000, 100_000, 1_000_000):
        print(f"perfect attack, n={n} per hypothesis: eps_lo={perfect_attack_eps(n, DELTA_RUN):.2f}")
    for e in (1.0, 2.0, 4.0, eps_claim, 8.0):
        print(f"trials per hypothesis for a perfect attack to certify eps={e:.2f}: {trials_needed(e, DELTA_RUN)}")

    print("== Where eps lives on the curve")
    mu = clt_mu(Q_RUN, SIGMA_RUN, STEPS_RUN)
    eps_clt = gdp_eps(mu, DELTA_RUN)
    a_star, tpr_star = gdp_touching_point(mu, eps_clt)
    print(f"CLT mu={mu:.4f}  eps at 1e-5={eps_clt:.2f}  touching point FPR={a_star:.1e} TPR={tpr_star:.1e}")
    mu_sum = Q_RUN * STEPS_RUN / math.sqrt(STEPS_RUN * SIGMA_RUN**2 + STEPS_RUN * Q_RUN * (1 - Q_RUN))
    print(f"sum of projections: mu about q*sqrt(T)/sigma={Q_RUN * math.sqrt(STEPS_RUN) / SIGMA_RUN:.2f}")
    print(
        f"sum of projections, with the extra variance of K: {mu_sum:.4f}  eps at 1e-5={gdp_eps(mu_sum, DELTA_RUN):.2f}"
    )
    for fpr in (1e-2, 1e-3, 1e-4):
        tpr = 1 - NORMAL.cdf(NORMAL.inv_cdf(1 - fpr) - mu)
        print(f"GDP mu={mu:.4f} at FPR={fpr:g}: TPR={tpr:.4f}  ln(TPR/FPR)={math.log(tpr / fpr):.2f}")

    print("== White-box audit of the mechanism (likelihood-ratio statistic)")
    lr_in = simulate_lr_scores(rng, POOL_TRIALS, True)
    lr_out = simulate_lr_scores(rng, POOL_TRIALS, False)
    for k, fpr, tpr, e in pointwise_eps(lr_in, lr_out, (5, 10, 50, 100, 500, 5000), DELTA_RUN):
        print(
            f"LR pool: {k} false positives: FPR={fpr:.1e} TPR={tpr:.4f} ln((TPR-delta)/FPR)={e:.2f}  "
            f"local mu={gdp_mu_from_rates(fpr, tpr):.3f}"
        )
    for n in (1000, 3000, 10_000, POOL_TRIALS):
        a = run_audit(lr_in[:n], lr_out[:n], DELTA_RUN)
        c = a.counts
        print(
            f"LR audit, n={n} per hypothesis: eval fp={c.fp}/{c.n_out} tp={c.tp}/{c.n_in}  eps_lo={a.eps_lo:.2f}  "
            f"mu_lo={a.mu_lo:.3f} (fp={a.mu_counts.fp}, tp={a.mu_counts.tp})  GDP eps={gdp_eps(a.mu_lo, DELTA_RUN):.2f}"
            f"  ceiling={perfect_attack_eps(n // 2, DELTA_RUN):.2f}"
        )
    full = run_audit(lr_in, lr_out, DELTA_RUN)
    print("== Sum of projections")
    for n in (1000, 10_000, 100_000, 1_000_000):
        s_in = simulate_sum_scores(rng, n, True)
        s_out = simulate_sum_scores(rng, n, False)
        a = run_audit(s_in, s_out, DELTA_RUN)
        print(f"sum audit, n={n} per hypothesis: eps_lo={a.eps_lo:.2f}  mu_lo={a.mu_lo:.3f}")

    print("== One-run audit: checks against the paper's worked examples")
    print(f"m=r=100, v=75, delta=0: eps_lo={one_run_eps(100, 100, 75, 0.0):.3f}")
    print(f"m=r=100, v=75, delta=1e-4: eps_lo={one_run_eps(100, 100, 75, 1e-4):.3f}")
    print(f"m=1000, r=100, v=75, delta=1e-4: eps_lo={one_run_eps(1000, 100, 75, 1e-4):.3f}")
    for m in (100, 1000, 10_000):
        print(
            f"all {m} guesses right: eps_lo={one_run_eps(m, m, m, 0.0):.2f} (delta=0)  "
            f"{one_run_eps(m, m, m, DELTA_RUN):.2f} (delta=1e-5)"
        )
    print("== One-run audit, white-box Dirac canaries (idealized)")
    for m in (1000, 5000):
        scores, included = idealized_one_run(rng, m)
        res = one_run_audit(scores, included, DELTA_RUN, (*GUESS_GRID, 2000, 5000) if m > 1000 else GUESS_GRID)
        print(f"m={m}: " + "  ".join(f"r={r}: {v} right, eps_lo={e:.2f}" for r, v, e in res.rows))

    print("== One-run audit, black box, the running example's network")
    task = make_task()
    canaries = mislabeled_canaries(task)
    for name, cfg in (
        ("DP-SGD sigma=1", DpsgdConfig()),
        ("clipping only, sigma=0", DpsgdConfig(sigma=0.0)),
        ("non-private", DpsgdConfig(sigma=0.0, clip=math.inf)),
    ):
        run = black_box_run(rng, task, canaries, cfg)
        res = one_run_audit(run.scores, run.included, DELTA_RUN)
        r, v, e = res.best
        print(
            f"{name}: included={int(run.included.sum())}  test acc={run.test_acc:.4f}  mean loss in="
            f"{-float(run.scores[run.included].mean()):.3f} out={-float(run.scores[~run.included].mean()):.3f}  "
            + "  ".join(f"r={rr}: {vv}" for rr, vv, _ in res.rows)
            + f"  best r={r}: {v} right, eps_lo={e:.2f}"
        )

    print("== Finding bugs: per-step audits (canary at every step, two runs of 10,000 steps)")
    half = STEPS_RUN // 2
    print(f"per-step ceiling on mu with {half} trials per hypothesis: {gdp_mu_lower_bound(0, half, half, half):.3f}")
    mu_correct = 0.0
    for impl in IMPLEMENTATIONS:
        a = per_step_audit(rng, impl)
        mu_correct = mu_correct or a.mu_lo
        implied = dpsgd_epsilon(Q_RUN, 1 / a.mu_lo, STEPS_RUN, DELTA_RUN) if a.mu_lo > 0 else 0.0
        print(
            f"{impl.name}: noise={impl.noise:g}C  per-step mu_lo={a.mu_lo:.3f} (claimed {1 / SIGMA_RUN:g})  "
            f"accountant at sigma=1/mu_lo: eps={implied:.2f}"
        )
    eps_mis = dpsgd_epsilon(Q_MISCOUNTED, SIGMA_RUN, STEPS_RUN, DELTA_RUN)
    print(f"miscounted n (accountant uses q={Q_MISCOUNTED}): claimed eps={eps_mis:.2f}")
    print(
        f"  per-step audit of the correct noise, composed at the claimed q={Q_MISCOUNTED}: "
        f"eps={dpsgd_epsilon(Q_MISCOUNTED, 1 / mu_correct, STEPS_RUN, DELTA_RUN):.2f}  (not refuted)"
    )
    print(f"  end-to-end LR audit (n={POOL_TRIALS}): eps_lo={full.eps_lo:.2f}  refuted={full.eps_lo > eps_mis}")
    s_in = simulate_sum_scores(rng, 10_000, True, noise=IMPLEMENTATIONS[2].noise)
    s_out = simulate_sum_scores(rng, 10_000, False, noise=IMPLEMENTATIONS[2].noise)
    a = run_audit(s_in, s_out, DELTA_RUN)
    print(
        f"noise scaled for the mean, end-to-end sum audit (n=10000): fp={a.counts.fp} tp={a.counts.tp}/{a.counts.n_in}  "
        f"eps_lo={a.eps_lo:.2f}  claimed {eps_claim:.2f}"
    )


# --- Figures --------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class FigurePool:
    lr_in: Array
    lr_out: Array


def make_figure_pool(rng: np.random.Generator) -> FigurePool:
    return FigurePool(simulate_lr_scores(rng, FIG_POOL_TRIALS, True), simulate_lr_scores(rng, FIG_POOL_TRIALS, False))


def empirical_roc(scores_in: Array, scores_out: Array) -> tuple[Array, Array]:
    """(FPR, TPR) at every threshold that admits one more false positive."""
    out_desc = np.sort(scores_out)[::-1]
    in_sorted = np.sort(scores_in)
    fpr = np.arange(1, out_desc.size + 1) / out_desc.size
    tpr = 1.0 - np.searchsorted(in_sorted, out_desc, side="left") / in_sorted.size
    return fpr, tpr


def plot_region(rng: np.random.Generator, pool: FigurePool) -> Figure:
    eps_claim = dpsgd_epsilon(Q_RUN, SIGMA_RUN, STEPS_RUN, DELTA_RUN)
    mu = clt_mu(Q_RUN, SIGMA_RUN, STEPS_RUN)
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))
    a = np.geomspace(1e-9, 0.999, 400)
    ax = axes[0]
    ax.loglog(a, [max_tpr(x, eps_claim, DELTA_RUN) for x in a], color="black", label=r"$(6.71, 10^{-5})$-DP limit")
    ax.loglog(
        a,
        [1 - NORMAL.cdf(NORMAL.inv_cdf(1 - x) - mu) for x in a],
        color="tab:gray",
        ls="--",
        label="GDP, CLT $\\mu = 1.31$",
    )
    fpr, tpr = empirical_roc(pool.lr_in, pool.lr_out)
    ax.loglog(fpr, tpr, color="tab:blue", label="audit: likelihood ratio")
    s_in = simulate_sum_scores(rng, FIG_POOL_TRIALS, True)
    s_out = simulate_sum_scores(rng, FIG_POOL_TRIALS, False)
    fpr_s, tpr_s = empirical_roc(s_in, s_out)
    ax.loglog(fpr_s, tpr_s, color="tab:orange", label="audit: sum of projections")
    out_desc = np.sort(pool.lr_out)[::-1]
    for k in (3, 30, 300, 3000):
        thr = float((out_desc[k - 1] + out_desc[k]) / 2)
        c = counts_at(pool.lr_in, pool.lr_out, thr)
        f_lo, f_hi = cp_lower(c.fp, c.n_out, 0.025), cp_upper(c.fp, c.n_out, 0.025)
        t_lo, t_hi = cp_lower(c.tp, c.n_in, 0.025), cp_upper(c.tp, c.n_in, 0.025)
        ax.add_patch(Rectangle((f_lo, t_lo), f_hi - f_lo, t_hi - t_lo, fill=False, ec="tab:blue", lw=1.2))
    ax.set_xlim(1e-7, 1)
    ax.set_ylim(1e-4, 1.05)
    ax.set_xlabel("false positive rate")
    ax.set_ylabel("true positive rate")
    ax.legend(fontsize=8, loc="lower right")
    ax.set_title("Trade-off: 30,000 trials per hypothesis")

    ax = axes[1]
    ax.semilogx(a, [eps_claim] * a.size, color="black", label="accountant, 6.71")
    gdp_pointwise = [math.log(max(1 - NORMAL.cdf(NORMAL.inv_cdf(1 - x) - mu) - DELTA_RUN, 1e-300) / x) for x in a]
    ax.semilogx(a, gdp_pointwise, color="tab:gray", ls="--", label=r"GDP $\mu = 1.31$: $\ln((TPR-\delta)/FPR)$")
    keep = tpr > DELTA_RUN
    ax.semilogx(fpr[keep], np.log((tpr[keep] - DELTA_RUN) / fpr[keep]), color="tab:blue", label="audit, point estimate")
    a_star, _ = gdp_touching_point(mu, gdp_eps(mu, DELTA_RUN))
    ax.axvline(a_star, color="tab:gray", ls=":", label=f"where GDP reaches its $\\varepsilon$: FPR {a_star:.0e}")
    ax.set_xlim(1e-9, 1)
    ax.set_ylim(0, 8)
    ax.set_xlabel("false positive rate of the threshold")
    ax.set_ylabel(r"$\varepsilon$ this threshold certifies")
    ax.legend(fontsize=8, loc="upper right")
    ax.set_title(r"Where $\varepsilon$ at $\delta = 10^{-5}$ lives")
    return fig


def plot_trials(rng: np.random.Generator, pool: FigurePool) -> Figure:
    eps_claim = dpsgd_epsilon(Q_RUN, SIGMA_RUN, STEPS_RUN, DELTA_RUN)
    ns_lr = [300, 1000, 3000, 10_000, FIG_POOL_TRIALS]
    ns_sum = [300, 1000, 3000, 10_000, 30_000, 100_000, 300_000, 1_000_000]
    lr_eps = [run_audit(pool.lr_in[:n], pool.lr_out[:n], DELTA_RUN).eps_lo for n in ns_lr]
    lr_gdp = [gdp_eps(run_audit(pool.lr_in[:n], pool.lr_out[:n], DELTA_RUN).mu_lo, DELTA_RUN) for n in ns_lr]
    sum_eps = [
        run_audit(simulate_sum_scores(rng, n, True), simulate_sum_scores(rng, n, False), DELTA_RUN).eps_lo
        for n in ns_sum
    ]
    grid = np.geomspace(100, 1e6, 60)
    fig, ax = plt.subplots(figsize=(6.8, 4.2))
    ax.semilogx(
        grid,
        [perfect_attack_eps(int(n), DELTA_RUN) for n in grid],
        color="tab:gray",
        ls=":",
        label="perfect attack (ceiling)",
    )
    ax.axhline(eps_claim, color="black", label="accountant, 6.71")
    ax.semilogx(ns_lr, lr_gdp, marker="s", color="tab:green", label="likelihood ratio, Gaussian-shape estimate")
    ax.semilogx(ns_lr, lr_eps, marker="o", color="tab:blue", label="likelihood ratio, rigorous bound")
    ax.semilogx(ns_sum, sum_eps, marker="^", color="tab:orange", label="sum of projections, rigorous bound")
    ax.set_xlabel("trials per hypothesis (training runs = twice this)")
    ax.set_ylabel(r"$\varepsilon$ lower bound at 95%, $\delta = 10^{-5}$")
    ax.set_ylim(0, 10)
    ax.legend(fontsize=8, loc="upper left")
    return fig


def plot_one_run(rng: np.random.Generator) -> Figure:
    task = make_task()
    canaries = mislabeled_canaries(task)
    dp = black_box_run(rng, task, canaries, DpsgdConfig())
    nonpriv = black_box_run(rng, task, canaries, DpsgdConfig(sigma=0.0, clip=math.inf))
    wb_scores, wb_included = idealized_one_run(rng, CANARIES)
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8))
    for ax, run, title in ((axes[0], nonpriv, "non-private"), (axes[1], dp, r"DP-SGD, $\varepsilon = 6.71$")):
        losses = -run.scores
        bins = np.linspace(0, float(np.quantile(losses, 0.995)), 40)
        ax.hist(losses[run.included], bins=bins, alpha=0.6, color="tab:red", label="canary in training set")
        ax.hist(losses[~run.included], bins=bins, alpha=0.6, color="tab:blue", label="canary left out")
        ax.set_xlabel("loss of the canary on the final model")
        ax.set_title(f"Black box, {title}")
        ax.legend(fontsize=8)
    axes[0].set_ylabel("canaries")
    ax = axes[2]
    grid = list(GUESS_GRID)
    for scores, included, label, color in (
        (nonpriv.scores, nonpriv.included, "black box, non-private", "tab:red"),
        (wb_scores, wb_included, "white box, Dirac canaries", "tab:green"),
        (dp.scores, dp.included, "black box, DP-SGD", "tab:blue"),
    ):
        res = one_run_audit(scores, included, DELTA_RUN, grid)
        ax.semilogx([r for r, _, _ in res.rows], [v / r for r, v, _ in res.rows], marker="o", color=color, label=label)
    p_claim = 1 / (1 + math.exp(-dpsgd_epsilon(Q_RUN, SIGMA_RUN, STEPS_RUN, DELTA_RUN)))
    ax.axhline(p_claim, color="black", ls="--", label=r"$e^\varepsilon/(1+e^\varepsilon)$ at $\varepsilon = 6.71$")
    ax.axhline(0.5, color="tab:gray", ls=":", label="chance")
    ax.set_xlabel("number of guesses r (1,000 canaries)")
    ax.set_ylabel("fraction of guesses correct")
    ax.set_ylim(0.4, 1.02)
    ax.legend(fontsize=7, loc="lower left")
    ax.set_title("One run: guesses against canaries")
    return fig


def plot_bugs(rng: np.random.Generator) -> Figure:
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6))
    for ax, impl in zip(axes, IMPLEMENTATIONS, strict=True):
        obs_in = per_step_observations(rng, STEPS_RUN, impl.noise, True)
        obs_out = per_step_observations(rng, STEPS_RUN, impl.noise, False)
        a = run_audit(obs_in, obs_out, 0.0)
        lo, hi = float(min(obs_out.min(), obs_in.min())), float(max(obs_out.max(), obs_in.max()))
        bins = np.linspace(lo, hi, 60)
        ax.hist(obs_out, bins=bins, alpha=0.6, color="tab:blue", label="canary absent")
        ax.hist(obs_in, bins=bins, alpha=0.6, color="tab:red", label="canary present")
        ax.set_title(f"{impl.title}\nper-step $\\mu \\geq$ {a.mu_lo:.2f} (claim: 1)", fontsize=10)
        ax.set_xlabel("projection on the canary, units of C")
    axes[0].set_ylabel("steps")
    axes[0].legend(fontsize=8)
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
    fig_rng = np.random.default_rng(seed + 1)
    pool = make_figure_pool(fig_rng)
    figures = {
        "audit_region.svg": plot_region(fig_rng, pool),
        "audit_trials.svg": plot_trials(fig_rng, pool),
        "audit_one_run.svg": plot_one_run(fig_rng),
        "audit_bugs.svg": plot_bugs(fig_rng),
    }
    for name, fig in figures.items():
        fig.tight_layout()
        fig.savefig(out / name, metadata={"Date": None})  # no timestamp, so output is reproducible
        plt.close(fig)
        print(f"saved {out / name}")


if __name__ == "__main__":
    main()
