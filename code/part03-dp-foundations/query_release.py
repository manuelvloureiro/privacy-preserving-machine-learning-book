"""Check the query-release algorithms of the query-release chapter and draw its figures.

    uv run python code/part03-dp-foundations/query_release.py
    uv run python code/part03-dp-foundations/query_release.py --seed 1 --out /tmp/figures

Prints the numbers quoted in 06-query-release.md (they match exactly with the default seed),
then saves the chapter's figures.

Conventions: the data universe X has N elements, and a dataset of n records is stored as its
histogram p, the fraction of records equal to each element. A linear query q maps X to [0, 1], and
its answer on p is the inner product <q, p>. Neighbours replace one record, and n is public, so one
answer moves by at most 1/n. The running example has d = 8 yes/no attributes (N = 256) and asks every
cell of every marginal table: 3^8 - 1 = 6560 queries. Logarithms are natural.
"""

import argparse
import itertools
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # for code/shared
from shared.figures import configure
from shared.stats import normal_cdf

configure("query_release")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from numpy.typing import NDArray

# Figures are build output: _build/figures/<part>/, which the build copies into each book.
FIGURES = Path(__file__).resolve().parents[2] / "_build" / "figures" / Path(__file__).parent.name

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]

# The running example: 8 yes/no attributes, records drawn from a mixture of three groups.
D_ATTR = 8
GROUP_WEIGHTS = (0.5, 0.3, 0.2)
GROUP_PROBS = (
    (0.8, 0.7, 0.2, 0.1, 0.6, 0.5, 0.3, 0.9),
    (0.2, 0.3, 0.7, 0.8, 0.4, 0.1, 0.6, 0.5),
    (0.5, 0.9, 0.5, 0.2, 0.1, 0.8, 0.7, 0.3),
)
EPS = 1.0
DELTA = 1e-6
BETA = 0.05


# --- The setting: a universe, a histogram, linear queries ----------------------------------------


def universe(d: int) -> IntArray:
    """All 2^d records as rows of 0/1, element i having bit j equal to (i >> j) & 1."""
    idx = np.arange(2**d, dtype=np.int64)
    return ((idx[:, None] >> np.arange(d, dtype=np.int64)[None, :]) & 1).astype(np.int64)


def sample_histogram(rng: np.random.Generator, n: int, d: int = D_ATTR) -> FloatArray:
    """Histogram (fractions) of n records drawn from the three-group mixture."""
    groups = rng.choice(len(GROUP_WEIGHTS), size=n, p=np.asarray(GROUP_WEIGHTS))
    probs = np.asarray(GROUP_PROBS)[groups, :d]
    bits = (rng.random((n, d)) < probs).astype(np.int64)
    index = bits @ (1 << np.arange(d, dtype=np.int64))
    return np.bincount(index, minlength=2**d).astype(np.float64) / n


def marginal_queries(d: int = D_ATTR) -> tuple[FloatArray, int]:
    """Every cell of every marginal table: one 0/1 query per (set of attributes, values). Also the table count."""
    cells = universe(d)
    rows: list[FloatArray] = []
    tables = 0
    for order in range(1, d + 1):
        for attrs in itertools.combinations(range(d), order):
            tables += 1
            sub = cells[:, list(attrs)]
            for values in itertools.product((0, 1), repeat=order):
                rows.append(np.all(sub == np.asarray(values), axis=1).astype(np.float64))
    return np.asarray(rows), tables


def max_error(queries: FloatArray, p_hat: FloatArray, p: FloatArray) -> float:
    """Largest error over the query set: max_q |<q, p_hat> - <q, p>|."""
    return float(np.max(np.abs(queries @ (p_hat - p))))


# --- Multiplicative weights ----------------------------------------------------------------------


def mw_update(p: FloatArray, loss: FloatArray, gamma: float) -> FloatArray:
    """One multiplicative-weights step: weights times (1 - γ ℓ), renormalized."""
    w = p * (1 - gamma * loss)
    return w / float(w.sum())


def regret_bound(gamma: float, rounds: int, n_experts: int) -> float:
    """γ T + ln N / γ: the regret of multiplicative weights after T rounds with step γ."""
    return gamma * rounds + math.log(n_experts) / gamma


def mistake_bound(n_experts: int, alpha: float) -> float:
    """4 ln N / α²: the largest number of rounds in which the error can exceed α (step γ = α/2)."""
    return 4 * math.log(n_experts) / alpha**2


def nonprivate_release(
    queries: FloatArray, p: FloatArray, alpha: float, gamma: float, max_rounds: int = 100_000
) -> tuple[int, list[float], list[float]]:
    """Non-private MW query release, always feeding the query with the largest error.

    Returns the number of updates until every error is at most α, the largest error before each update,
    and the running total of the per-round losses <ℓ_t, p_t - p> (the regret against p).
    """
    p_t = np.full(p.size, 1.0 / p.size)
    truth = queries @ p
    errors: list[float] = []
    regret: list[float] = []
    total = 0.0
    for _ in range(max_rounds):
        diff = queries @ p_t - truth
        i = int(np.argmax(np.abs(diff)))
        if abs(float(diff[i])) <= alpha:
            return len(errors), errors, regret
        errors.append(abs(float(diff[i])))
        sign = 1.0 if diff[i] > 0 else -1.0
        loss = sign * queries[i]
        total += float(loss @ (p_t - p))
        regret.append(total)
        p_t = mw_update(p_t, loss, gamma)
    return max_rounds, errors, regret


def halving_mistakes(rng: np.random.Generator, n_experts: int, rounds: int) -> int:
    """The halving algorithm against random experts with one perfect expert: its mistakes."""
    alive = np.ones(n_experts, dtype=bool)
    perfect = int(rng.integers(n_experts))
    mistakes = 0
    for _ in range(rounds):
        votes = rng.integers(0, 2, n_experts)
        truth = int(rng.integers(0, 2))
        votes[perfect] = truth
        guess = 1 if 2 * int(votes[alive].sum()) >= int(alive.sum()) else 0
        mistakes += int(guess != truth)
        alive &= votes == truth
    return mistakes


# --- Private multiplicative weights (offline) ----------------------------------------------------


def pure_step_epsilon(eps: float, rounds: int) -> float:
    """ε0 = ε/(2T): each of the 2T steps gets an equal share under basic composition."""
    return eps / (2 * rounds)


def approx_step_epsilon(eps: float, steps: int, delta: float) -> float:
    """ε0 = ε/(2√(2k ln(1/δ))): k steps of ε0-DP are (ε, δ)-DP by advanced composition (for ε ≤ 8 ln(1/δ))."""
    return eps / (2 * math.sqrt(2 * steps * math.log(1 / delta)))


def approx_step_epsilon_exact(eps: float, steps: int, delta: float) -> float:
    """The largest ε0 whose k-fold advanced composition (tanh form) is at most ε, by bisection."""
    lo, hi = 0.0, eps
    for _ in range(100):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if advanced_total(mid, steps, delta) <= eps else (lo, mid)
    return lo


def advanced_total(eps0: float, steps: int, delta: float) -> float:
    """ε0 √(2k ln(1/δ)) + k ε0 tanh(ε0/2): the advanced composition of k steps of ε0-DP."""
    return eps0 * math.sqrt(2 * steps * math.log(1 / delta)) + steps * eps0 * math.tanh(eps0 / 2)


def pmw_error_bound(n_experts: int, n_queries: int, n: int, eps0: float, rounds: int, beta: float) -> float:
    """2√(ln N/T) + (2/(n ε0))(ln|Q| + 2 ln(2T/β)): the error of the averaged histogram after T rounds."""
    return 2 * math.sqrt(math.log(n_experts) / rounds) + 2 / (n * eps0) * (
        math.log(n_queries) + 2 * math.log(2 * rounds / beta)
    )


def best_rounds(n_experts: int, n_queries: int, n: int, eps: float, beta: float, delta: float = 0.0) -> int:
    """The T that minimizes the error bound (pure DP if δ = 0, else advanced composition).

    Searches T ≥ 4 ln N (so that the step γ = √(ln N/T) is at most 1/2) on a grid 1% apart up to 10^7.
    """
    best, best_t = math.inf, 1
    start = math.ceil(4 * math.log(n_experts))
    grid = sorted({math.ceil(start * 1.01**i) for i in range(2000) if start * 1.01**i <= 1e7})
    for t in grid:
        eps0 = pure_step_epsilon(eps, t) if delta == 0 else approx_step_epsilon(eps, 2 * t, delta)
        b = pmw_error_bound(n_experts, n_queries, n, eps0, t, beta)
        if b < best:
            best, best_t = b, t
    return best_t


def pmw(rng: np.random.Generator, queries: FloatArray, p: FloatArray, n: int, eps0: float, rounds: int) -> FloatArray:
    """Offline private multiplicative weights: T rounds of select (EM), measure (Laplace), update (MW).

    Returns the average of the histograms p_1, ..., p_T, which is what the analysis bounds.
    """
    gamma = math.sqrt(math.log(p.size) / rounds)
    p_t = np.full(p.size, 1.0 / p.size)
    total = np.zeros(p.size)
    truth = queries @ p
    for _ in range(rounds):
        total += p_t
        synth = queries @ p_t
        logits = eps0 * n * np.abs(synth - truth) / 2  # exponential mechanism, score sensitivity 1/n
        prob = np.exp(logits - logits.max())
        i = int(rng.choice(len(prob), p=prob / prob.sum()))
        measured = float(truth[i]) + float(rng.laplace(0.0, 1 / (n * eps0)))
        sign = 1.0 if synth[i] > measured else -1.0
        p_t = mw_update(p_t, sign * queries[i], gamma)
    return total / rounds


# --- Baselines: noise on every answer --------------------------------------------------------------


def gaussian_delta(mu: float, eps: float) -> float:
    """δ(ε) of the Gaussian mechanism whose sensitivity is μ standard deviations."""
    return normal_cdf(mu / 2 - eps / mu) - math.exp(eps) * normal_cdf(-mu / 2 - eps / mu)


def gaussian_mu(eps: float, delta: float) -> float:
    """Largest μ (sensitivity over noise standard deviation) that is (ε, δ)-DP, by bisection."""
    lo, hi = 1e-6, 50.0
    for _ in range(200):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if gaussian_delta(mid, eps) <= delta else (lo, mid)
    return lo


def laplace_release(rng: np.random.Generator, truth: FloatArray, l1_sensitivity: float, eps: float) -> FloatArray:
    return truth + rng.laplace(0.0, l1_sensitivity / eps, truth.size)


def gaussian_release(
    rng: np.random.Generator, truth: FloatArray, l2_sensitivity: float, eps: float, delta: float
) -> FloatArray:
    return truth + rng.normal(0.0, l2_sensitivity / gaussian_mu(eps, delta), truth.size)


# --- Sample complexity: records needed for error α on every query ---------------------------------


def normal_upper_quantile(tail: float) -> float:
    """z with Pr[Z > z] = tail for a standard normal Z, by bisection."""
    lo, hi = 0.0, 40.0
    for _ in range(200):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if 1 - normal_cdf(mid) > tail else (lo, mid)
    return hi


def n_laplace(n_queries: int, alpha: float, eps: float, beta: float) -> float:
    """Laplace on every answer, ℓ1 sensitivity |Q|/n: n = |Q| ln(|Q|/β)/(α ε)."""
    return n_queries * math.log(n_queries / beta) / (alpha * eps)


def n_gaussian(n_queries: int, alpha: float, eps: float, delta: float, beta: float) -> float:
    """Gaussian on every answer, ℓ2 sensitivity √|Q|/n: n = √|Q| z/(μ α), z the β/(2|Q|) normal quantile."""
    z = normal_upper_quantile(beta / (2 * n_queries))
    return math.sqrt(n_queries) * z / (gaussian_mu(eps, delta) * alpha)


def pmw_needed(n_experts: int, n_queries: int, alpha: float, eps: float, delta: float, beta: float) -> float:
    """Smallest n (to 1%) at which the offline PMW bound, with its best T, reaches α."""

    def ok(n: float) -> bool:
        t = best_rounds(n_experts, n_queries, int(n), eps, beta, delta)
        eps0 = pure_step_epsilon(eps, t) if delta == 0 else approx_step_epsilon(eps, 2 * t, delta)
        return pmw_error_bound(n_experts, n_queries, int(n), eps0, t, beta) <= alpha

    lo, hi = 1.0, 2.0
    while not ok(hi):
        lo, hi = hi, hi * 2
    while hi / lo > 1.01:
        mid = math.sqrt(lo * hi)
        lo, hi = (lo, mid) if ok(mid) else (mid, hi)
    return hi


def pmw_closed_form(n_experts: int, alpha: float, eps: float, delta: float, log_term: float) -> float:
    """n = 64 √(ln N ln(1/δ)) L/(α² ε): the approximate-DP bound with T at its balance point."""
    return 64 * math.sqrt(math.log(n_experts) * math.log(1 / delta)) * log_term / (alpha**2 * eps)


def smalldb_size(n_queries: int, alpha: float) -> int:
    """m = ⌈2 ln(2|Q|)/α²⌉: a sample of m records has every answer within α/2 with positive probability."""
    return math.ceil(2 * math.log(2 * n_queries) / alpha**2)


def n_smalldb(n_experts: int, n_queries: int, alpha: float, eps: float, beta: float) -> float:
    """SmallDB: n ≥ 4(m ln N + ln(1/β))/(α ε)."""
    m = smalldb_size(n_queries, alpha)
    return 4 * (m * math.log(n_experts) + math.log(1 / beta)) / (alpha * eps)


def online_updates(n_experts: int, alpha: float) -> int:
    """c = ⌈16 ln N/α²⌉: online PMW updates at most this often (loss above α/2, step α/4)."""
    return math.ceil(16 * math.log(n_experts) / alpha**2)


def n_online_pmw(n_experts: int, k: int, alpha: float, eps: float, delta: float, beta: float) -> float:
    """Records online PMW needs: max of the sparse vector margin and the Laplace measurement, each ≤ α/4.

    The c AboveThreshold runs and the c measurements alternate, 2c steps in all, each at
    ε_run = ε/(2√(4c ln(1/δ))), so that advanced composition gives (ε, δ).
    """
    c = online_updates(n_experts, alpha)
    eps_run = approx_step_epsilon(eps, 2 * c, delta)
    n_svt = 32 * (math.log(k) + math.log(4 * c / beta)) / (alpha * eps_run)
    n_lap = 4 * math.log(2 * c / beta) / (alpha * eps_run)
    return max(n_svt, n_lap)


# --- The table ----------------------------------------------------------------------------------


def print_table(rng: np.random.Generator) -> None:
    queries, tables = marginal_queries()
    n_experts = 2**D_ATTR
    nq = queries.shape[0]
    print(f"[Setting: d = {D_ATTR}, N = {n_experts}, |Q| = {nq} marginal cells in {tables} tables]")

    n = 10_000
    p = sample_histogram(rng, n)
    uniform = np.full(n_experts, 1.0 / n_experts)
    print(f"  n={n}: largest error of the uniform histogram={max_error(queries, uniform, p):.3f}")
    print(f"  ln N={math.log(n_experts):.3f}  ln|Q|={math.log(nq):.3f}")

    print("[Baselines at n = 10000, eps = 1 (Gaussian: delta = 1e-6), mean of 20 runs]")
    truth = queries @ p
    l1 = 2 * tables / n
    l2 = math.sqrt(2 * tables) / n
    mu = gaussian_mu(EPS, DELTA)
    lap = [float(np.max(np.abs(laplace_release(rng, truth, l1, EPS) - truth))) for _ in range(20)]
    gau = [float(np.max(np.abs(gaussian_release(rng, truth, l2, EPS, DELTA) - truth))) for _ in range(20)]
    print(f"  l1 sensitivity={l1:.4f}  Laplace scale={l1 / EPS:.4f}  mean largest error={np.mean(lap):.3f}")
    print(f"  l2 sensitivity={l2:.5f}  mu={mu:.4f}  Gaussian sd={l2 / mu:.4f}  mean largest error={np.mean(gau):.3f}")
    print(f"  generic l1 sensitivity |Q|/n={nq / n:.4f}")

    print("[Multiplicative weights: halving, 1024 experts, 200 rounds, 20 runs]")
    halv = [halving_mistakes(rng, 1024, 200) for _ in range(20)]
    print(f"  largest mistakes={max(halv)}  bound log2 N={math.log2(1024):.0f}")

    print("[Non-private MW release on the example, alpha = 0.05, gamma = alpha/2]")
    alpha = 0.05
    rounds, errors, regret = nonprivate_release(queries, p, alpha, alpha / 2)
    print(
        f"  updates until every error <= alpha: {rounds}  mistake bound 4 ln N/alpha^2={mistake_bound(n_experts, alpha):.0f}"
    )
    print(
        f"  first errors: {errors[0]:.3f} {errors[1]:.3f} {errors[2]:.3f}  "
        f"regret after {rounds}={regret[-1]:.3f}  bound gamma T + ln N/gamma={regret_bound(alpha / 2, rounds, n_experts):.3f}"
    )

    print("[Private MW at n = 10000, eps = 1, mean of 20 runs]")
    t_pure = best_rounds(n_experts, nq, n, EPS, BETA)
    for t in (10, t_pure, 100):
        eps0 = pure_step_epsilon(EPS, t)
        runs = [max_error(queries, pmw(rng, queries, p, n, eps0, t), p) for _ in range(20)]
        print(
            f"  pure, T={t}: eps0={eps0:.4f}  mean largest error={np.mean(runs):.3f}  "
            f"bound={pmw_error_bound(n_experts, nq, n, eps0, t, BETA):.3f}"
        )
    print(f"  bound-optimal T, pure: {t_pure}")
    for t in (100, 1000):
        eps0a = approx_step_epsilon(EPS, 2 * t, DELTA)
        eps0x = approx_step_epsilon_exact(EPS, 2 * t, DELTA)
        print(
            f"  T={t}, delta=1e-6: pure eps0={pure_step_epsilon(EPS, t):.5f}  closed-form eps0={eps0a:.5f} "
            f"(spends {advanced_total(eps0a, 2 * t, DELTA):.3f})  largest eps0={eps0x:.5f}"
        )

    print("[Records needed for error alpha = 0.05 on every query, eps = 1, delta = 1e-6, beta = 0.05]")
    big_n = 2**20
    for q_count in (10**3, 10**6, 10**9):
        print(
            f"  |Q|={q_count:.0e}, N=2^20: Laplace={n_laplace(q_count, 0.05, EPS, BETA):.3g}  "
            f"Gaussian={n_gaussian(q_count, 0.05, EPS, DELTA, BETA):.3g}  "
            f"PMW={pmw_needed(big_n, q_count, 0.05, EPS, DELTA, BETA):.3g}  "
            f"SmallDB={n_smalldb(big_n, q_count, 0.05, EPS, BETA):.3g}"
        )

    nq6 = 10**6
    n6 = pmw_needed(big_n, nq6, 0.05, EPS, DELTA, BETA)
    t6 = best_rounds(big_n, nq6, int(n6), EPS, BETA, DELTA)
    log_term = math.log(nq6) + 2 * math.log(2 * t6 / BETA)
    print(
        f"  |Q|=1e6: best T={t6}  L={log_term:.2f}  closed form 64 sqrt(ln N ln(1/delta)) L/(alpha^2 eps)="
        f"{pmw_closed_form(big_n, 0.05, EPS, DELTA, log_term):.3g}"
    )

    print("[SmallDB on the example, alpha = 0.1]")
    m = smalldb_size(nq, 0.1)
    print(
        f"  m={m}  log10 of N^m={m * math.log10(n_experts):.0f}  "
        f"records needed={n_smalldb(n_experts, nq, 0.1, EPS, BETA):.3g}"
    )

    print("[Online PMW, alpha = 0.05, k = 10^6 queries, eps = 1, delta = 1e-6]")
    c = online_updates(big_n, 0.05)
    print(f"  N=2^20: c={c}  records needed={n_online_pmw(big_n, 10**6, 0.05, EPS, DELTA, BETA):.3g}")
    c8 = online_updates(n_experts, 0.05)
    print(f"  N=256: c={c8}")


# --- Figures ------------------------------------------------------------------------------------


def plot_mw(rng: np.random.Generator) -> Figure:
    queries, _ = marginal_queries()
    n_experts = 2**D_ATTR
    p = sample_histogram(rng, 10_000)
    alpha = 0.05
    fig, (left, right) = plt.subplots(1, 2, figsize=(11, 4.2))
    rounds, errors, regret = nonprivate_release(queries, p, alpha, alpha / 2)
    steps = np.arange(1, rounds + 1)
    left.plot(steps, errors, color="C0", label="largest error before the update")
    left.axhline(alpha, color="C3", linestyle=":", label=r"target $\alpha$ = 0.05")
    left.set(
        xlabel="update t",
        ylabel=r"$\max_q |\langle q, p_t\rangle - \langle q, p\rangle|$",
        title="Non-private MW learns the histogram",
    )
    left.legend(fontsize=9)
    right.plot(steps, regret, color="C0", label=r"regret $\sum_t \langle \ell_t, p_t - p\rangle$")
    right.plot(
        steps, alpha * steps, color="C1", linestyle="--", label=r"$\alpha t$: each update loses more than $\alpha$"
    )
    right.plot(
        steps,
        [regret_bound(alpha / 2, int(s), n_experts) for s in steps],
        color="C2",
        linestyle="-.",
        label=r"bound $\gamma t + \ln N/\gamma$",
    )
    right.set(xlabel="update t", ylabel="total", title="Regret against the true histogram")
    right.legend(fontsize=9)
    return fig


def plot_error_vs_n(rng: np.random.Generator, runs: int = 10) -> Figure:
    queries, tables = marginal_queries()
    n_experts = 2**D_ATTR
    nq = queries.shape[0]
    ns = [1_000, 2_000, 5_000, 10_000, 20_000, 50_000, 100_000]
    lap_err: list[float] = []
    gau_err: list[float] = []
    pmw_err: list[float] = []
    uni_err: list[float] = []
    for n in ns:
        p = sample_histogram(rng, n)
        truth = queries @ p
        l1 = 2 * tables / n
        l2 = math.sqrt(2 * tables) / n
        lap_err.append(
            float(np.mean([np.max(np.abs(laplace_release(rng, truth, l1, EPS) - truth)) for _ in range(runs)]))
        )
        gau_err.append(
            float(np.mean([np.max(np.abs(gaussian_release(rng, truth, l2, EPS, DELTA) - truth)) for _ in range(runs)]))
        )
        t = best_rounds(n_experts, nq, n, EPS, BETA)
        eps0 = pure_step_epsilon(EPS, t)
        pmw_err.append(float(np.mean([max_error(queries, pmw(rng, queries, p, n, eps0, t), p) for _ in range(runs)])))
        uni_err.append(max_error(queries, np.full(n_experts, 1.0 / n_experts), p))
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(ns, lap_err, "o-", label=r"Laplace on every answer ($\varepsilon$ = 1)")
    ax.plot(ns, gau_err, "s-", label=r"Gaussian on every answer ($\varepsilon$ = 1, $\delta = 10^{-6}$)")
    ax.plot(ns, pmw_err, "^-", label=r"private MW, pure ($\varepsilon$ = 1)")
    ax.plot(ns, uni_err, ":", color="0.5", label="uniform histogram (no data)")
    ax.set(
        xscale="log",
        yscale="log",
        xlabel="records n",
        ylabel="largest error over 6560 queries",
        title="All marginals of 8 yes/no attributes",
    )
    ax.legend(fontsize=9)
    return fig


def plot_sample_complexity() -> Figure:
    big_n = 2**20
    counts = [10**e for e in range(2, 13)]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(counts, [n_laplace(c, 0.05, EPS, BETA) for c in counts], "o-", label="Laplace on every answer")
    ax.plot(counts, [n_gaussian(c, 0.05, EPS, DELTA, BETA) for c in counts], "s-", label="Gaussian on every answer")
    ax.plot(counts, [pmw_needed(big_n, c, 0.05, EPS, DELTA, BETA) for c in counts], "^-", label="private MW (bound)")
    ax.plot(counts, [n_smalldb(big_n, c, 0.05, EPS, BETA) for c in counts], "d-", label="SmallDB (bound)")
    ax.set(
        xscale="log",
        yscale="log",
        xlabel="number of queries |Q|",
        ylabel="records needed",
        title=r"Error 0.05 on every query, $N = 2^{20}$, $\varepsilon$ = 1, $\delta = 10^{-6}$",
    )
    ax.legend(fontsize=9)
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
        "qr_mw.svg": plot_mw(rng),
        "qr_error_vs_n.svg": plot_error_vs_n(rng),
        "qr_sample_complexity.svg": plot_sample_complexity(),
    }
    for name, fig in figures.items():
        fig.tight_layout()
        fig.savefig(out / name, metadata={"Date": None})  # no timestamp, so output is reproducible
        plt.close(fig)
        print(f"saved {out / name}")


if __name__ == "__main__":
    main()
