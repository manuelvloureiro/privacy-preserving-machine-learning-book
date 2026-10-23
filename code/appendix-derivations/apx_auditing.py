"""Check the derivations of the appendix on auditing, reconstruction and adaptive analysis, and draw its figures.

    uv run python code/appendix-derivations/apx_auditing.py
    uv run python code/appendix-derivations/apx_auditing.py --seed 1 --out /tmp/figures

Prints every number quoted in book-full/appendix-derivations/14-auditing-adaptivity.md (they match exactly
with the default seed), then saves the appendix's figures. The script checks

- the one-run auditing bound: binomial tails, the ε it certifies, the domination lemma on an adaptive
  sequence computed exactly, and simulated audits of randomized response and of Laplace scores;
- the high-probability transfer theorem: the exact posterior of a small coin example, the records Gaussian
  answering needs with high probability, and a simulated adaptive analyst against Gaussian answers;
- the Ladder leaderboard: the counting bound, its best step size, and the boosting attack against an exact
  leaderboard and against the Ladder;
- the reconstruction bound from Rényi divergence of order 2: the Hammersley–Chapman–Robbins inequality
  against the exact error of the Gaussian and Laplace mechanisms.
"""

import argparse
import math
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
matplotlib.rcParams["svg.hashsalt"] = "apx_auditing"  # stable element ids, so reruns don't churn the SVGs
matplotlib.rcParams["svg.fonttype"] = "none"  # keep text as text: smaller, diffable files

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from numpy.typing import NDArray

# Figures are build output: _build/figures/<part>/, which the build copies into each book.
FIGURES = Path(__file__).resolve().parents[2] / "_build" / "figures" / Path(__file__).parent.name

Array = NDArray[np.float64]
BETA = 0.05


def bisect(pred: Callable[[float], bool], lo: float, hi: float, iters: int = 100) -> float:
    """The boundary of a predicate that is False on [lo, x) and True on [x, hi]."""
    for _ in range(iters):
        mid = (lo + hi) / 2
        if pred(mid):
            hi = mid
        else:
            lo = mid
    return (lo + hi) / 2


def golden_min(f: Callable[[float], float], lo: float, hi: float, iters: int = 80) -> tuple[float, float]:
    """Minimize a unimodal function on [lo, hi] by golden-section search; returns (argmin, min)."""
    g = (math.sqrt(5) - 1) / 2
    a, b = lo, hi
    c, d = b - g * (b - a), a + g * (b - a)
    fc, fd = f(c), f(d)
    for _ in range(iters):
        if fc < fd:
            b, d, fd = d, c, fc
            c = b - g * (b - a)
            fc = f(c)
        else:
            a, c, fc = c, d, fd
            d = a + g * (b - a)
            fd = f(d)
    x = (a + b) / 2
    return x, f(x)


# --- One-run auditing ------------------------------------------------------------------------------


def p_eps(eps: float) -> float:
    """The largest chance that one guess is right under ε-DP: e^ε / (1 + e^ε)."""
    return 1 / (1 + math.exp(-eps))


def binom_logpmf(r: int, p: float, v: int) -> float:
    return math.lgamma(r + 1) - math.lgamma(v + 1) - math.lgamma(r - v + 1) + v * math.log(p) + (r - v) * math.log1p(-p)


def binom_pmf(r: int, p: float) -> list[float]:
    return [math.exp(binom_logpmf(r, p, v)) for v in range(r + 1)]


def binom_sf(r: int, p: float, v: int) -> float:
    """Pr[Bin(r, p) >= v]."""
    if v <= 0:
        return 1.0
    return sum(math.exp(binom_logpmf(r, p, j)) for j in range(v, r + 1))


def one_run_eps_lo(r: int, v: int, beta: float = BETA) -> float:
    """The ε lower bound of a pure-DP one-run audit: the ε at which Pr[Bin(r, p_ε) >= v] = β."""
    if binom_sf(r, 0.5, v) >= beta:
        return 0.0
    return bisect(lambda e: binom_sf(r, p_eps(e), v) >= beta, 0.0, 30.0)


def adaptive_sum_pmf(r: int, p_good: float, p_bad: float) -> list[float]:
    """Exact law of X_1 + ... + X_r when Pr[X_j = 1 | past] is p_good after a success and p_bad after a failure.

    The first indicator uses p_good. The state is (count so far, was the last one a success), so a table of
    size 2(r + 1) carries the whole law.
    """
    after_success = [0.0] * (r + 1)
    after_failure = [0.0] * (r + 1)
    after_success[0] = 1.0  # before the first step, behave as after a success
    for _ in range(r):
        new_s = [0.0] * (r + 1)
        new_f = [0.0] * (r + 1)
        for c in range(r):
            for mass, p in ((after_success[c], p_good), (after_failure[c], p_bad)):
                new_s[c + 1] += mass * p
                new_f[c] += mass * (1 - p)
        after_success, after_failure = new_s, new_f
    return [a + b for a, b in zip(after_success, after_failure, strict=True)]


def sf_from_pmf(pmf: list[float], v: int) -> float:
    return sum(pmf[v:])


def rr_one_run(rng: np.random.Generator, m: int, r: int, eps: float, runs: int) -> NDArray[np.int64]:
    """Correct guesses W of a one-run audit of ε-randomized response, over independent runs.

    Each canary is in with probability 1/2. The mechanism reports each inclusion bit, flipped with probability
    1 - p_ε. The auditor scores each canary by its report, breaks ties at random, and guesses "in" for the r/2
    highest scores and "out" for the r/2 lowest.
    """
    p = p_eps(eps)
    s = rng.integers(0, 2, size=(runs, m)) * 2 - 1
    report = np.where(rng.random((runs, m)) < p, s, -s)
    scores = report + rng.random((runs, m)) * 0.5  # random tie-break within each report value
    return guesses_correct(scores, s, r)


def laplace_one_run(rng: np.random.Generator, m: int, rs: list[int], eps: float, runs: int) -> Array:
    """Mean accuracy W / r, for each r, of one-run audits of the scores S_i + Lap(2/ε), which are ε-DP in each bit."""
    s = rng.integers(0, 2, size=(runs, m)) * 2 - 1
    scores = s + rng.laplace(0.0, 2.0 / eps, size=(runs, m))
    return np.array([float(np.mean(guesses_correct(scores, s, r))) / r for r in rs])


def guesses_correct(scores: NDArray[np.float64], s: NDArray[np.int64], r: int) -> NDArray[np.int64]:
    """Number of correct guesses per run: "in" for the r/2 highest scores, "out" for the r/2 lowest."""
    order = np.argsort(scores, axis=1)
    low = np.take_along_axis(s, order[:, : r // 2], axis=1)
    high = np.take_along_axis(s, order[:, scores.shape[1] - r // 2 :], axis=1)
    right_in: NDArray[np.bool_] = high == 1
    right_out: NDArray[np.bool_] = low == -1
    return np.count_nonzero(right_in, axis=1) + np.count_nonzero(right_out, axis=1)


# --- High-probability transfer -------------------------------------------------------------------


def zcdp_to_dp(rho: float, delta: float) -> float:
    """The simple conversion ε = ρ + 2 sqrt(ρ ln(1/δ)) of a ρ-zCDP mechanism to (ε, δ)-DP."""
    return rho + 2 * math.sqrt(rho * math.log(1 / delta))


def coin_posterior_mean(a: float, n: int, p: float, eps: float) -> float:
    """q(Q_π) for n coin records with mean p, after one answer a = K/n + Lap(1/(nε)), with K the number of ones.

    The posterior of the count K given a is proportional to Pr[K] times the Laplace density at a - K/n; the
    posterior mean of a uniformly chosen record is E[K | a] / n.
    """
    b = 1 / (n * eps)
    logw = [binom_logpmf(n, p, c) - abs(a - c / n) / b for c in range(n + 1)]
    top = max(logw)
    w = [math.exp(x - top) for x in logw]
    return sum(c * wc for c, wc in enumerate(w)) / (n * sum(w))


def pure_dp_band(p: float, eps: float) -> tuple[float, float]:
    """Where q(Q_π) must lie under ε-DP: within a factor e^ε of p, and of 1 - p for the complement."""
    lo = max(p * math.exp(-eps), 1 - math.exp(eps) * (1 - p))
    hi = min(p * math.exp(eps), 1 - math.exp(-eps) * (1 - p))
    return lo, hi


@dataclass(frozen=True)
class GaussianBound:
    bound: float
    sigma: float
    delta: float
    eps: float


def expectation_bound_at(n: float, k: int, sigma: float, delta: float) -> GaussianBound:
    """The chapter's expected-error bound σ sqrt(2 ln 2k) + e^ε - 1 + δ for Gaussian answers."""
    eps = zcdp_to_dp(k / (2 * n**2 * sigma**2), delta)
    value = sigma * math.sqrt(2 * math.log(2 * k)) + (math.expm1(eps) if eps < 700 else math.inf) + delta
    return GaussianBound(value, sigma, delta, eps)


def expectation_bound(n: float, k: int) -> GaussianBound:
    """The smallest expected-error bound over σ and δ."""

    def best(log_delta: float) -> float:
        return golden_min(lambda ls: expectation_bound_at(n, k, math.exp(ls), math.exp(log_delta)).bound, -15, 3)[1]

    log_delta, _ = golden_min(best, -60, -0.01)
    log_sigma, _ = golden_min(lambda ls: expectation_bound_at(n, k, math.exp(ls), math.exp(log_delta)).bound, -15, 3)
    return expectation_bound_at(n, k, math.exp(log_sigma), math.exp(log_delta))


@dataclass(frozen=True)
class HighProbBound:
    bound: float  # Pr[max_j |a_j - q_j(D)| > bound] <= beta
    sigma: float
    sample_error: float  # α: the noise is below it except with probability β_s
    c: float
    d: float
    eps: float
    delta: float
    beta_s: float


def high_prob_bound_at(n: float, k: int, beta: float, x: tuple[float, float, float, float]) -> HighProbBound:
    """The page's theorem for Gaussian answers, at log σ, log c, log d and the share f of β given to β_s / c.

    Sample accuracy fails with probability β_s = c f β, so the Markov step costs f β. The privacy slack is
    δ = d (1 - f) β, so the posterior step costs (1 - f) β. The error is α + c + e^ε - 1 + d.
    """
    sigma, c, d = math.exp(x[0]), math.exp(x[1]), math.exp(x[2])
    f = 1 / (1 + math.exp(-x[3]))
    beta_s = min(c * f * beta, 1.0)
    delta = min(d * (1 - f) * beta, 0.5)
    sample_error = sigma * math.sqrt(2 * math.log(2 * k / beta_s))
    eps = zcdp_to_dp(k / (2 * n**2 * sigma**2), delta)
    value = sample_error + c + d + (math.expm1(eps) if eps < 700 else math.inf)
    return HighProbBound(value, sigma, sample_error, c, d, eps, delta, beta_s)


def high_prob_bound(n: float, k: int, beta: float = BETA, sigma: float | None = None) -> HighProbBound:
    """The smallest high-probability bound, by coordinate descent over (log σ, log c, log d, logit f).

    With sigma given, the noise level is held there and only c, d and f are optimized.
    """
    x = [math.log(0.05 if sigma is None else sigma), math.log(0.02), math.log(0.02), 0.0]
    ranges = [(-15.0, 3.0) if sigma is None else (x[0], x[0]), (-12.0, 0.0), (-12.0, 0.0), (-8.0, 8.0)]
    for _ in range(12):
        for i, (lo, hi) in enumerate(ranges):

            def along(t: float, i: int = i) -> float:
                y = list(x)
                y[i] = t
                return high_prob_bound_at(n, k, beta, (y[0], y[1], y[2], y[3])).bound

            x[i] = golden_min(along, lo, hi, 60)[0]
    return high_prob_bound_at(n, k, beta, (x[0], x[1], x[2], x[3]))


def required_n(error: Callable[[float], float], alpha: float, iters: int = 200) -> int:
    """The smallest sample size whose bound is at most alpha, by bisection on log n."""
    log_n = bisect(lambda ln: error(math.exp(ln)) <= alpha, 0.0, 40.0, iters)
    n = math.ceil(math.exp(log_n))
    while n > 1 and error(n - 1) <= alpha:
        n -= 1
    return n


def required_n_rough(error: Callable[[float], float], alpha: float) -> float:
    """The same to three significant digits, for bounds computed by a numerical search."""
    return float(f"{math.exp(bisect(lambda ln: error(math.exp(ln)) <= alpha, 0.0, 40.0, 40)):.3g}")


def boosting_vs_gaussian(rng: np.random.Generator, n: int, k: int, sigma: float, runs: int) -> Array:
    """Largest population error max_j |a_j - 1/2| of Gaussian answers to the boosting analyst, per run.

    The holdout labels are fair coins, so every query "does submission f agree with the label?" has population
    value exactly 1/2. The analyst submits k - 1 random labelings, then the majority vote of those whose noisy
    answer was above 1/2. Each answer is the empirical agreement plus N(0, σ²) noise, clipped to [0, 1].
    """
    out = np.empty(runs)
    for t in range(runs):
        labels = rng.integers(0, 2, size=n, dtype=np.int8)
        subs = rng.integers(0, 2, size=(k - 1, n), dtype=np.int8)
        emp = np.mean(subs == labels, axis=1)
        answers = np.clip(emp + rng.normal(0.0, sigma, size=k - 1), 0.0, 1.0)
        keep = subs[answers > 0.5].astype(np.int64)
        votes: NDArray[np.int64] = np.sum(2 * keep - 1, axis=0)
        final = (votes >= 0).astype(np.int8)
        agree: NDArray[np.bool_] = final == labels
        last = min(max(float(np.mean(agree)) + float(rng.normal(0.0, sigma)), 0.0), 1.0)
        out[t] = max(float(np.max(np.abs(answers - 0.5))), abs(last - 0.5))
    return out


# --- The Ladder ------------------------------------------------------------------------------------


def ladder_log_histories(k: int, steps: int) -> float:
    """ln of the number of possible released sequences: C(k + L, L) for L = 1/η possible drops."""
    return math.lgamma(k + steps + 1) - math.lgamma(k + 1) - math.lgamma(steps + 1)


def ladder_bound(n: int, k: int, steps: int, beta: float = BETA) -> tuple[float, float]:
    """(rounding term η, union-bound term) of the Ladder's leaderboard error bound, with η = 1/steps.

    At most k C(k + L, L) classifiers can be submitted, and Hoeffding with a union bound over them gives
    sqrt(ln(2 k C(k + L, L) / β) / (2n)).
    """
    union = math.sqrt((math.log(2 * k / beta) + ladder_log_histories(k, steps)) / (2 * n))
    return 1 / steps, union


def ladder_closed_form(n: int, k: int, beta: float = BETA) -> tuple[float, float]:
    """(best η, bound) after C(k + L, L) <= (k + 1)^L and sqrt(x + y) <= sqrt(x) + sqrt(y)."""
    a = math.log(k + 1) / (2 * n)
    eta = (a / 4) ** (1 / 3)
    return eta, 3 * eta + math.sqrt(math.log(2 * k / beta) / (2 * n))


def ladder_release(losses: Array, eta: float) -> Array:
    """The Ladder's released values for a sequence of empirical losses: R_0 = 1; R_t = round(loss) when loss < R - η."""
    released = np.empty(len(losses))
    current = 1.0
    for t, loss in enumerate(losses):
        if loss < current - eta:
            current = round(float(loss) / eta) * eta
        released[t] = current
    return released


@dataclass(frozen=True)
class LadderAttack:
    exact_error: float  # mean of |best reported loss - 1/2| for an exact leaderboard
    ladder_error: float  # the same for the Ladder


def ladder_attack(rng: np.random.Generator, n: int, k: int, steps: int, runs: int) -> LadderAttack:
    """The boosting attack on n coin labels, against an exact leaderboard and against the Ladder.

    Every submission's population loss is exactly 1/2. Against the exact leaderboard, the attacker sees each
    submission's empirical loss and keeps those below 1/2. Against the Ladder, it sees only the released value
    and keeps the submissions that moved it. The last submission is the majority vote of the kept ones.
    """
    eta = 1 / steps
    exact, ladder = np.empty(runs), np.empty(runs)
    for t in range(runs):
        labels = rng.integers(0, 2, size=n, dtype=np.int8)
        subs = rng.integers(0, 2, size=(k - 1, n), dtype=np.int8)
        losses = np.mean(subs != labels, axis=1)
        for which in ("exact", "ladder"):
            if which == "exact":
                kept = subs[losses < 0.5]
            else:
                released = ladder_release(losses, eta)
                moved = np.diff(np.concatenate(([1.0], released))) < 0
                kept = subs[moved]
            votes: NDArray[np.int64] = np.sum(2 * kept.astype(np.int64) - 1, axis=0)
            final = (votes >= 0).astype(np.int8)
            wrong: NDArray[np.bool_] = final != labels
            all_losses = np.append(losses, float(np.mean(wrong)))
            if which == "exact":
                exact[t] = abs(float(np.min(all_losses)) - 0.5)
            else:
                ladder[t] = abs(float(ladder_release(all_losses, eta)[-1]) - 0.5)
    return LadderAttack(float(np.mean(exact)), float(np.mean(ladder)))


# --- Reconstruction from Rényi divergence of order 2 -------------------------------------------------


def hcr_bound(eps: float, diameter: float = 1.0) -> float:
    """The mean squared error any unbiased reconstruction must have, at every target: diam² / (4 (e^ε - 1))."""
    return diameter**2 / (4 * math.expm1(eps))


def gaussian_mse(eps: float, sensitivity: float = 1.0) -> float:
    """Exact MSE of the unbiased estimate θ + N(0, σ²) when D_2 = Δ²/σ² equals ε: σ² = Δ²/ε."""
    return sensitivity**2 / eps


def laplace_d2(shift_over_b: float) -> float:
    """Rényi divergence of order 2 between Laplace(0, b) and Laplace(s, b), as a function of s / b."""
    t = shift_over_b
    return math.log(2 / 3 * math.exp(t) + 1 / 3 * math.exp(-2 * t))


def laplace_d2_numeric(shift_over_b: float, grid: int = 400_001) -> float:
    """The same divergence by numerical integration of E_Q[(p/q)²] = ∫ p²/q, as a check of the closed form."""
    t = shift_over_b
    x = np.linspace(-40.0 - t, 40.0 + t, grid)
    log_p = -np.abs(x - t) - math.log(2)
    log_q = -np.abs(x) - math.log(2)
    integrand = np.exp(2 * log_p - log_q)
    return math.log(float(np.sum(integrand) * (x[1] - x[0])))


def laplace_mse(eps: float, sensitivity: float = 1.0) -> float:
    """Exact MSE 2b² of the unbiased estimate θ + Lap(b), with b chosen so that D_2 at shift Δ equals ε."""
    t = bisect(lambda s: laplace_d2(s) >= eps, 0.0, 200.0)
    b = sensitivity / t
    return 2 * b**2


def gaussian_hcr_curve(h: float, sigma: float) -> float:
    """The Hammersley–Chapman–Robbins bound h² / χ² for a Gaussian with step h: χ² = e^{h²/σ²} - 1."""
    return h**2 / math.expm1(h**2 / sigma**2)


# --- Table ---------------------------------------------------------------------------------------------

ONE_RUN_EPS = (0.5, 1.0, math.log(3), 2.0)
RECON_EPS = (0.1, 0.5, 1.0, 2.0, 4.0, 8.0)


def print_table(rng: np.random.Generator) -> None:
    # One-run auditing
    r, v = 100, 75
    for eps in ONE_RUN_EPS:
        p = p_eps(eps)
        print(f"[one-run r={r}] eps={eps:.4f} p_eps={p:.4f} mean={r * p:.2f} Pr[Bin>={v}]={binom_sf(r, p, v):.4f}")
    print(f"[one-run r={r} v={v}] eps_lo={one_run_eps_lo(r, v):.3f}")
    for m in (100, 1000, 10000):
        print(f"[one-run all {m} right] eps_lo={one_run_eps_lo(m, m):.2f}")
    pmf = adaptive_sum_pmf(r, 0.75, 0.6)
    gaps = [sf_from_pmf(pmf, u) - binom_sf(r, 0.75, u) for u in range(r + 1)]
    print(
        f"[domination r={r} p=0.75, 0.6 after a miss] mean={sum(c * q for c, q in enumerate(pmf)):.2f} "
        f"Pr[X>={v}]={sf_from_pmf(pmf, v):.4f} binomial={binom_sf(r, 0.75, v):.4f} "
        f"largest gap={max(gaps):.1e}"
    )
    runs = 20_000
    w = rr_one_run(rng, 1000, r, math.log(3), runs)
    print(
        f"[RR one-run m=1000 r={r} eps=ln3 runs={runs}] mean W={float(np.mean(w)):.2f} "
        f"Pr[W>={v}]={float(np.mean(w >= v)):.4f} binomial={binom_sf(r, 0.75, v):.4f}"
    )
    rs = [20, 100, 500, 1000]
    acc = laplace_one_run(rng, 1000, rs, 1.0, 2000)
    print(
        "[Laplace one-run m=1000 eps=1] accuracy "
        + " ".join(f"r={rr}:{a:.3f}" for rr, a in zip(rs, acc, strict=True))
        + f"  p_eps={p_eps(1.0):.3f}  sign accuracy={1 - 0.5 * math.exp(-0.5):.3f}"
    )

    # High-probability transfer: the coin example
    n_coin, p_coin = 20, 0.3
    for eps in (0.25, 1.0, 4.0):
        lo, hi = pure_dp_band(p_coin, eps)
        grid = [i / 200 for i in range(-40, 241)]
        means = [coin_posterior_mean(a, n_coin, p_coin, eps) for a in grid]
        inside = all(lo - 1e-12 <= mm <= hi + 1e-12 for mm in means)
        print(
            f"[coin n={n_coin} p={p_coin} eps={eps:g}] posterior mean at a=0.6: "
            f"{coin_posterior_mean(0.6, n_coin, p_coin, eps):.4f}  range [{min(means):.4f}, {max(means):.4f}]  "
            f"band [{lo:.4f}, {hi:.4f}] inside={inside}"
        )

    # High-probability transfer: Gaussian answering
    alpha = 0.05
    needed: dict[int, float] = {}
    for k in (1, 100, 1000, 1_000_000):
        n_exp = required_n(lambda nn, k=k: expectation_bound(nn, k).bound, alpha)
        n_hp = required_n_rough(lambda nn, k=k: high_prob_bound(nn, k).bound, alpha)
        print(f"[records for alpha={alpha} k={k}] expectation={n_exp:,} high-prob beta={BETA}: {n_hp:,.0f}")
        needed[k] = n_hp
    hp = high_prob_bound(needed[1000], 1000)
    print(
        f"[high-prob optimum k=1000] sample={hp.sample_error:.4f} c={hp.c:.4f} transfer={math.expm1(hp.eps):.4f} "
        f"d={hp.d:.4f} eps={hp.eps:.4f} delta={hp.delta:.2e} beta_s={hp.beta_s:.2e}"
    )
    n_sim, k_sim, runs_sim = 10_000, 100, 400
    exp_b = expectation_bound(n_sim, k_sim)
    hp_b = high_prob_bound(n_sim, k_sim, sigma=exp_b.sigma)
    errs = boosting_vs_gaussian(rng, n_sim, k_sim, exp_b.sigma, runs_sim)
    print(
        f"[Gaussian answers n={n_sim} k={k_sim} runs={runs_sim}] sigma={exp_b.sigma:.4f} "
        f"mean max error={float(np.mean(errs)):.4f} 95% quantile={float(np.quantile(errs, 0.95)):.4f} "
        f"expectation bound={exp_b.bound:.4f} high-prob bound={hp_b.bound:.4f}"
    )

    # The Ladder
    for n, k in ((1000, 1000), (10_000, 1000)):
        eta, closed = ladder_closed_form(n, k)
        steps = max(1, round(1 / eta))
        rounding, union = ladder_bound(n, k, steps)
        best = min(sum(ladder_bound(n, k, s)) for s in range(1, 400))
        print(
            f"[Ladder n={n} k={k}] closed form eta={eta:.4f} bound={closed:.4f}; "
            f"eta=1/{steps}: {rounding:.4f} + {union:.4f} = {rounding + union:.4f}; best over 1/L={best:.4f}"
        )
    attack = ladder_attack(rng, 1000, 1000, 11, 20)
    print(
        f"[Ladder attack n=1000 k=1000 eta=1/11 runs=20] exact={attack.exact_error:.4f} ladder={attack.ladder_error:.4f}"
    )

    # Reconstruction
    print(f"[Laplace D2 check s/b=1] closed={laplace_d2(1.0):.6f} numeric={laplace_d2_numeric(1.0):.6f}")
    for eps in RECON_EPS:
        b, g, lap = hcr_bound(eps), gaussian_mse(eps), laplace_mse(eps)
        print(
            f"[recon eps={eps:g}] bound={b:.4g} endpoint={4 * b:.4g} Gaussian={g:.4g} Laplace={lap:.4g} "
            f"Gaussian/bound={g / b:.3g}"
        )


# --- Figures ---------------------------------------------------------------------------------------------


def plot_onerun_tails(r: int = 100, v: int = 75) -> Figure:
    fig, ax = plt.subplots(figsize=(7, 4))
    xs = list(range(r + 1))
    for eps, name in zip(ONE_RUN_EPS, ("0.5", "1", "ln 3", "2"), strict=True):
        pmf = binom_pmf(r, p_eps(eps))
        (line,) = ax.plot(xs, pmf, label=f"ε = {name}: Pr[W ≥ {v}] ≤ {binom_sf(r, p_eps(eps), v):.3f}")
        ax.fill_between(xs[v:], pmf[v:], color=line.get_color(), alpha=0.25, step=None)
    ax.axvline(v, color="k", linestyle="--", label=f"observed v = {v}")
    ax.set(
        xlim=(35, 100),
        xlabel="correct guesses out of r = 100",
        ylabel="probability",
        title="The binomial envelope Bin(r, p_ε); shaded: its tail beyond v",
    )
    ax.legend(loc="upper left", fontsize=8)
    return fig


def plot_onerun_sim(rng: np.random.Generator) -> Figure:
    fig, (left, right) = plt.subplots(1, 2, figsize=(11, 4))
    r, eps = 100, math.log(3)
    w = rr_one_run(rng, 1000, r, eps, 20_000)
    counts = np.bincount(w, minlength=r + 1) / len(w)
    xs = np.arange(r + 1)
    left.bar(xs, counts, width=1.0, color="tab:blue", alpha=0.5, label="simulated W, randomized response")
    left.plot(xs, binom_pmf(r, p_eps(eps)), "k", label="Bin(100, 0.75)")
    left.set(
        xlim=(50, 100),
        xlabel="correct guesses W",
        ylabel="probability",
        title="Randomized response: the envelope is exact",
    )
    left.legend(fontsize=8)

    rs = [10, 20, 50, 100, 200, 400, 600, 800, 1000]
    for e, color in ((1.0, "tab:orange"), (2.0, "tab:green")):
        acc = laplace_one_run(rng, 1000, rs, e, 2000)
        right.plot(rs, acc, "o-", color=color, label=f"Laplace scores, ε = {e:g}")
        right.axhline(p_eps(e), color=color, linestyle="--", label=f"p_ε = {p_eps(e):.3f}")
    right.set(
        xscale="log",
        xlabel="guesses r out of m = 1,000 canaries",
        ylabel="average accuracy W / r",
        title="Laplace scores: confident guesses reach p_ε",
    )
    right.legend(fontsize=8)
    return fig


def plot_posterior() -> Figure:
    n, p = 20, 0.3
    grid = [i / 200 for i in range(-20, 221)]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for eps in (0.25, 1.0, 4.0):
        (line,) = ax.plot(grid, [coin_posterior_mean(a, n, p, eps) for a in grid], label=f"q(Q_π), ε = {eps:g}")
        lo, hi = pure_dp_band(p, eps)
        ax.axhspan(lo, hi, color=line.get_color(), alpha=0.08)
        ax.axhline(hi, color=line.get_color(), linestyle=":", linewidth=1)
        ax.axhline(lo, color=line.get_color(), linestyle=":", linewidth=1)
    ax.plot(grid, grid, color="gray", linestyle="--", label="the answer itself")
    ax.axhline(p, color="k", linewidth=1.5, label=f"population value q(D) = {p}")
    ax.set(
        xlim=(-0.1, 1.1),
        ylim=(0, 1),
        xlabel="observed answer a",
        ylabel="posterior mean of a sample record",
        title=f"{n} coin records, one noisy answer",
    )
    ax.legend(fontsize=8, loc="upper left")
    return fig


def plot_transfer_sim(rng: np.random.Generator) -> Figure:
    n, k = 10_000, 100
    exp_b = expectation_bound(n, k)
    hp_b = high_prob_bound(n, k, sigma=exp_b.sigma)
    errs = boosting_vs_gaussian(rng, n, k, exp_b.sigma, 400)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(errs, bins=40, color="tab:blue", alpha=0.6, label="largest population error, 400 runs")
    ax.axvline(float(np.mean(errs)), color="tab:blue", linestyle="-", label=f"its mean {float(np.mean(errs)):.3f}")
    ax.axvline(exp_b.bound, color="tab:orange", linestyle="--", label=f"expectation bound {exp_b.bound:.3f}")
    ax.axvline(hp_b.bound, color="tab:red", linestyle="--", label=f"95% high-probability bound {hp_b.bound:.3f}")
    ax.set(
        xlim=(0, hp_b.bound * 1.1),
        xlabel="largest population error of the k answers",
        ylabel="runs",
        title=f"Gaussian answers (σ = {exp_b.sigma:.3f}) to {k} adaptive queries on {n:,} records",
    )
    ax.legend(fontsize=8)
    return fig


def plot_ladder(n: int = 10_000, k: int = 1000) -> Figure:
    steps = list(range(3, 200))
    etas = [1 / s for s in steps]
    parts = [ladder_bound(n, k, s) for s in steps]
    totals = [a + b for a, b in parts]
    best = int(np.argmin(totals))
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(etas, [a for a, _ in parts], label="rounding term ν")
    ax.plot(etas, [b for _, b in parts], label="union-bound term")
    ax.plot(etas, totals, "k", linewidth=2, label="their sum")
    ax.plot([etas[best]], [totals[best]], "ro", label=f"minimum {totals[best]:.3f} at ν = 1/{steps[best]}")
    ax.set(
        xscale="log",
        ylim=(0, 0.5),
        xlabel="step ν",
        ylabel="bound on the leaderboard error",
        title=f"The Ladder's bound, n = {n:,} records, k = {k:,} submissions",
    )
    ax.legend(fontsize=8)
    return fig


def plot_recon() -> Figure:
    fig, (left, right) = plt.subplots(1, 2, figsize=(11, 4))
    eps_grid = [10 ** (i / 40) for i in range(-80, 37)]
    left.plot(eps_grid, [hcr_bound(e) for e in eps_grid], "k", linewidth=2, label=r"bound $1/(4(e^\varepsilon - 1))$")
    left.plot(eps_grid, [gaussian_mse(e) for e in eps_grid], label="Gaussian mechanism, exact MSE 1/ε")
    left.plot(eps_grid, [laplace_mse(e) for e in eps_grid], label="Laplace mechanism, exact MSE 2b²")
    left.axhline(0.25, color="gray", linestyle=":", label="biased guess 1/2, worst-case MSE 1/4")
    left.set(
        xscale="log",
        yscale="log",
        ylim=(1e-4, 1e3),
        xlabel="RDP of order 2, ε",
        ylabel="mean squared error",
        title="Unbiased reconstruction of x ∈ [0, 1]",
    )
    left.legend(fontsize=8)

    eps = 1.0
    sigma = math.sqrt(1 / eps)
    hs = [i / 200 for i in range(1, 201)]
    right.plot(hs, [gaussian_hcr_curve(h, sigma) for h in hs], label=r"$h^2/\chi^2$ for the Gaussian's own pair")
    right.plot(hs, [h**2 / math.expm1(eps) for h in hs], label=r"$h^2/(e^\varepsilon - 1)$, worst-case $\chi^2$")
    right.axhline(sigma**2, color="gray", linestyle=":", label="exact MSE σ² = 1")
    right.axvline(0.5, color="k", linestyle="--", linewidth=1, label="h = 1/2: the page's choice at x = 1/2")
    right.set(
        xlabel="distance h = |x′ − x|", ylabel="lower bound on the MSE", title="Where the slack comes from, ε = 1"
    )
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
        "apx_audit_onerun_tails.svg": plot_onerun_tails(),
        "apx_audit_onerun_sim.svg": plot_onerun_sim(rng),
        "apx_audit_posterior.svg": plot_posterior(),
        "apx_audit_transfer_sim.svg": plot_transfer_sim(rng),
        "apx_audit_ladder.svg": plot_ladder(),
        "apx_audit_recon.svg": plot_recon(),
    }
    for name, fig in figures.items():
        fig.tight_layout()
        fig.savefig(out / name, metadata={"Date": None})  # no timestamp, so output is reproducible
        plt.close(fig)
        print(f"saved {out / name}")


if __name__ == "__main__":
    main()
