"""Check the numbers in the adaptive data analysis chapter and draw its figures.

    uv run python code/part05-dp-ml/adaptive_data_analysis.py
    uv run python code/part05-dp-ml/adaptive_data_analysis.py --seed 1 --out /tmp/figures

Prints the numbers quoted in 11-adaptive-data-analysis.md (they match exactly with the default seed), then
saves the chapter's figures.

Every question an analyst asks is a statistical query: a function q from one record to [0, 1], whose
population value q(D) is its mean over the data distribution and whose empirical value q(S) is its mean over
the n records of the sample S. The script computes

- the error of the point-query attack, in which one adaptively chosen query breaks the empirical answers;
- the leaderboard boosting attack, which overfits a holdout set through exact accuracy scores;
- the expected-error bounds of three ways to answer k queries (fixed queries, sample splitting, and Gaussian
  noise with the transfer theorem) and the sample size each needs;
- the selection bias of a query picked from the empirical values, by an argmax and by the exponential
  mechanism, against the transfer lemma's e^ε - 1;
- and the reusable-holdout experiment: features selected on a training set and confirmed on a holdout set
  that is reused as is, answered with Gaussian noise, or guarded by Thresholdout.
"""

import argparse
import math
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # for code/shared
from shared.figures import configure

configure("adaptive_data_analysis")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from numpy.typing import NDArray

# Figures are build output: _build/figures/<part>/, which the build copies into each book.
FIGURES = Path(__file__).resolve().parents[2] / "_build" / "figures" / Path(__file__).parent.name

Array = NDArray[np.float64]


# --- Toolkit: the maximal inequality and the zCDP conversion -------------------------------------


def max_inequality(variance_proxy: float, k: int) -> float:
    """Bound on E max_j |Z_j| for k centred variables, each sub-Gaussian with this variance proxy.

    E max over the 2k variables ±Z_j is at most sqrt(2 v ln(2k)).
    """
    return math.sqrt(2 * variance_proxy * math.log(2 * k))


def zcdp_to_dp(rho: float, delta: float) -> float:
    """The simple conversion ε = ρ + 2 sqrt(ρ ln(1/δ)) of a ρ-zCDP mechanism to (ε, δ)-DP."""
    return rho + 2 * math.sqrt(rho * math.log(1 / delta))


# --- Reusing a holdout overfits: the point-query attack -------------------------------------------


def point_query_population(n: int, domain: int) -> float:
    """Expected population value of the query "is x in the sample?" for n uniform draws from `domain` values.

    Its empirical value is 1. Its population value is the fraction of the domain the sample covers, whose
    expectation is 1 - (1 - 1/N)^n.
    """
    return 1 - (1 - 1 / domain) ** n


def point_query_attack(rng: np.random.Generator, n: int, domain: int) -> tuple[float, float, float]:
    """One run: (largest error of the N point queries, empirical and population value of the last query)."""
    sample = rng.integers(0, domain, size=n)
    counts = np.bincount(sample, minlength=domain) / n
    point_error = float(np.max(np.abs(counts - 1 / domain)))
    covered = len(np.unique(sample)) / domain
    return point_error, 1.0, covered


# --- Reusing a holdout overfits: the leaderboard boosting attack ----------------------------------


def boosting_approximation(n: int, k: int) -> float:
    """Small-signal approximation of the boosting attack's overfitting: sqrt(k / (2n)) / π."""
    return math.sqrt(k / (2 * n)) / math.pi


def boosting_attack(rng: np.random.Generator, n: int, k: int, trials: int) -> tuple[float, float]:
    """Mean and standard error of the holdout accuracy that the boosting attack's final submission reports.

    The holdout labels are n fair coins. The attacker submits k random labelings, reads each one's exact
    holdout accuracy, and submits the majority vote of those that scored above 1/2 (ties go to 1).
    Its true accuracy is exactly 1/2, because its labels are independent of any fresh label.
    """
    scores = np.empty(trials)
    for t in range(trials):
        labels = rng.integers(0, 2, size=n, dtype=np.int8)
        votes = np.zeros(n, dtype=np.int64)
        for start in range(0, k, 500):  # chunks keep memory small for large k
            m = min(500, k - start)
            subs = rng.integers(0, 2, size=(m, n), dtype=np.int8)
            acc = np.mean(subs == labels, axis=1)
            keep = subs[acc > 0.5].astype(np.int64)
            votes += np.sum(2 * keep - 1, axis=0)
        final = (votes >= 0).astype(np.int8)
        scores[t] = float(np.count_nonzero(final == labels)) / n
    return float(np.mean(scores)), float(np.std(scores, ddof=1) / math.sqrt(trials))


# --- Sample splitting and the rates -------------------------------------------------------------


def nonadaptive_error(n: float, k: int) -> float:
    """Expected largest error of k fixed queries answered empirically: sqrt(ln(2k) / (2n))."""
    return max_inequality(1 / (4 * n), k)


def splitting_error(n: float, k: int) -> float:
    """Expected largest error when query j is answered on its own part of n/k records: sqrt(k ln(2k) / (2n))."""
    return max_inequality(k / (4 * n), k)


def hoeffding_union_n(k: int, alpha: float, beta: float) -> float:
    """Sample size for which k fixed queries are all within alpha with probability 1 - beta."""
    return math.log(2 * k / beta) / (2 * alpha**2)


def _golden_min(f: Callable[[float], float], lo: float, hi: float, iters: int = 80) -> tuple[float, float]:
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


@dataclass(frozen=True)
class GaussianBound:
    bound: float  # E max_j |a_j - q_j(D)| ≤ bound
    sigma: float  # noise standard deviation per answer
    delta: float
    eps: float
    sample_error: float  # σ sqrt(2 ln(2k)), the expected largest noise
    transfer: float  # e^ε - 1 + δ


def gaussian_bound_at(n: float, k: int, sigma: float, delta: float) -> GaussianBound:
    """The transfer theorem's bound for k adaptive queries answered with N(0, σ²) noise on n records."""
    rho = k / (2 * n**2 * sigma**2)  # each answer has replace-sensitivity 1/n, so ρ = (1/n)² / (2σ²) per query
    eps = zcdp_to_dp(rho, delta)
    sample_error = sigma * math.sqrt(2 * math.log(2 * k))
    transfer = math.expm1(eps) + delta if eps < 700 else math.inf
    return GaussianBound(sample_error + transfer, sigma, delta, eps, sample_error, transfer)


def gaussian_error(n: float, k: int) -> GaussianBound:
    """The smallest transfer-theorem bound over the noise level σ and the δ of the conversion."""

    def best_for_delta(log_delta: float) -> float:
        _, v = _golden_min(lambda ls: gaussian_bound_at(n, k, math.exp(ls), math.exp(log_delta)).bound, -15, 3)
        return v

    log_delta, _ = _golden_min(best_for_delta, -60, -0.01)
    log_sigma, _ = _golden_min(lambda ls: gaussian_bound_at(n, k, math.exp(ls), math.exp(log_delta)).bound, -15, 3)
    return gaussian_bound_at(n, k, math.exp(log_sigma), math.exp(log_delta))


def required_n(error: Callable[[float, int], float], k: int, alpha: float) -> int:
    """The smallest sample size whose expected-error bound is at most alpha (bisection on log n)."""
    lo, hi = 0.0, 40.0
    for _ in range(200):
        mid = (lo + hi) / 2
        if error(math.exp(mid), k) <= alpha:
            hi = mid
        else:
            lo = mid
    n = math.ceil(math.exp(hi))
    while n > 1 and error(n - 1, k) <= alpha:
        n -= 1
    return n


def gaussian_error_value(n: float, k: int) -> float:
    return gaussian_error(n, k).bound


# --- DP implies generalization: the selection bias of a chosen query -------------------------------


@dataclass(frozen=True)
class SelectionBias:
    argmax: float  # E[q_W(S)] - q(D) when W picks the largest empirical value
    argmax_se: float
    exponential: dict[float, tuple[float, float]]  # ε -> (bias, standard error)


def selection_bias(rng: np.random.Generator, n: int, k: int, epsilons: list[float], trials: int) -> SelectionBias:
    """Selection bias of k null queries (population value 1/2), each the agreement rate of a random feature.

    Their empirical values are independent Binomial(n, 1/2)/n. The exponential mechanism with score q_j(S),
    which has replace-sensitivity 1/n, picks j with probability proportional to exp(ε n q_j(S) / 2) and is
    ε-DP; its bias given the sample is computed exactly as the probability-weighted mean of the q_j(S).
    """
    values = rng.binomial(n, 0.5, size=(trials, k)) / n
    arg = values.max(axis=1) - 0.5
    exp_bias: dict[float, tuple[float, float]] = {}
    for eps in epsilons:
        logits = eps * n * values / 2
        logits -= logits.max(axis=1, keepdims=True)
        w = np.exp(logits)
        w /= w.sum(axis=1, keepdims=True)
        b = np.sum(w * values, axis=1) - 0.5
        exp_bias[eps] = (float(np.mean(b)), float(np.std(b, ddof=1) / math.sqrt(trials)))
    return SelectionBias(float(np.mean(arg)), float(np.std(arg, ddof=1) / math.sqrt(trials)), exp_bias)


def transfer_lemma_bound(eps: float, delta: float = 0.0) -> float:
    """|E q_W(S) - E q_W(D)| ≤ e^ε - 1 + δ for a query chosen by an (ε, δ)-DP algorithm W."""
    return math.expm1(eps) + delta


# --- The reusable holdout: Thresholdout ------------------------------------------------------------


@dataclass(frozen=True)
class HoldoutData:
    x_train: Array  # ±1 features, n × d
    y_train: Array  # ±1 labels
    x_hold: Array
    y_hold: Array
    p_agree: Array  # P(x_j = y) for each feature: 1/2, or 1/2 + bias for the informative ones


def holdout_data(rng: np.random.Generator, n: int, d: int, informative: int, bias: float) -> HoldoutData:
    """Training and holdout sets of n records each, with d ±1 features and a ±1 label.

    The first `informative` features agree with the label with probability 1/2 + bias; the rest are
    independent of it.
    """
    p = np.full(d, 0.5)
    p[:informative] += bias

    def draw() -> tuple[Array, Array]:
        y = rng.choice(np.array([-1.0, 1.0], dtype=np.float32), size=n)
        agree = rng.random((n, d), dtype=np.float32) < p.astype(np.float32)
        x = np.where(agree, y[:, None], -y[:, None]).astype(np.float32)
        return x.astype(np.float64), y.astype(np.float64)

    x_train, y_train = draw()
    x_hold, y_hold = draw()
    return HoldoutData(x_train, y_train, x_hold, y_hold, p)


def poisson_binomial_pmf(probs: Array) -> Array:
    """P(number of successes = m), m = 0..len(probs), for independent trials with these success probabilities."""
    pmf = np.zeros(len(probs) + 1)
    pmf[0] = 1.0
    for i, p in enumerate(probs):
        pmf[1 : i + 2] = pmf[1 : i + 2] * (1 - p) + pmf[: i + 1] * p
        pmf[0] *= 1 - p
    return pmf


def true_accuracy(signs: Array, p_agree: Array) -> float:
    """Population accuracy of sign(Σ_j s_j x_j), ties broken towards +1, computed exactly.

    Each term s_j x_j y is +1 with probability p_j if s_j = +1 and 1 - p_j if s_j = -1, independently.
    The prediction is right when the sum is positive, and half the time when it is zero.
    """
    probs = np.where(signs > 0, p_agree, 1 - p_agree)
    pmf = poisson_binomial_pmf(probs)
    m = len(probs)
    plus = np.arange(m + 1)
    total = 2 * plus - m  # value of the sum with `plus` terms equal to +1
    return float(np.sum(pmf[total > 0]) + 0.5 * np.sum(pmf[total == 0]))


class Thresholdout:
    """Thresholdout (Dwork et al., Science 2015) on queries with values in [-1, 1].

    For each query it compares the gap between the holdout and training values with a noisy threshold.
    Below it, it answers with the training value; above it, with a Laplace-noised holdout value, and it
    redraws the threshold noise and counts one overfitting detection.
    """

    def __init__(self, rng: np.random.Generator, threshold: float, sigma: float) -> None:
        self.rng = rng
        self.threshold = threshold
        self.sigma = sigma
        self.noisy_threshold = threshold + float(rng.laplace(0, 2 * sigma))
        self.detections = 0

    def answer(self, train_value: float, hold_value: float) -> float:
        nu = float(self.rng.laplace(0, 4 * self.sigma))
        if abs(hold_value - train_value) + nu > self.noisy_threshold:
            self.detections += 1
            self.noisy_threshold = self.threshold + float(self.rng.laplace(0, 2 * self.sigma))
            return hold_value + float(self.rng.laplace(0, self.sigma))
        return train_value


def thresholdout_epsilon(detections: int, n: int, sigma: float) -> float:
    """Pure-DP cost, in the holdout records, of Thresholdout on correlation queries (replace-sensitivity 2/n).

    Each detection costs Δ/(2σ) for the threshold noise, 2Δ/(4σ) for the query noise and Δ/σ for the noisy
    answer: 2Δ/σ in all. The answers below the threshold are free, as in the sparse vector technique.
    """
    sens = 2 / n
    return detections * 2 * sens / sigma


@dataclass(frozen=True)
class ReuseResult:
    sizes: list[int]
    train: Array  # mean training accuracy per size
    reported: Array  # mean accuracy reported by the holdout mechanism per size
    true: Array  # mean exact population accuracy per size
    detections: float  # Thresholdout only: mean number of overfitting detections per trial
    queries: float  # mean number of holdout queries per trial


def holdout_reuse(
    rng: np.random.Generator,
    n: int,
    d: int,
    informative: int,
    bias: float,
    sizes: list[int],
    trials: int,
) -> dict[str, ReuseResult]:
    """The reusable-holdout experiment for three ways to answer the holdout queries.

    The analyst computes the correlation of each feature with the label on the training set, asks the
    holdout for the correlations of the features whose training correlation exceeds 1/sqrt(n), keeps the features where both exceed 1/sqrt(n) in size and agree in
    sign, ranks them by training correlation, and for each size m asks the holdout for the accuracy of
    sign(Σ_j s_j x_j) over the top m. "reuse" answers with the exact holdout value, "gaussian" adds N(0, σ²)
    with σ = 1/sqrt(n), the sampling standard deviation of a correlation, and "thresholdout" uses
    Thresholdout with threshold 4/sqrt(n) and σ = 1/sqrt(n). Accuracies are reported as correlations in
    [-1, 1] internally and converted to [0, 1].
    """
    mechanisms = ("reuse", "gaussian", "thresholdout")
    sums = {m: np.zeros((3, len(sizes))) for m in mechanisms}
    det = {m: 0.0 for m in mechanisms}
    queries = {m: 0.0 for m in mechanisms}
    cut = 1 / math.sqrt(n)
    for _ in range(trials):
        data = holdout_data(rng, n, d, informative, bias)
        corr_train = data.x_train.T @ data.y_train / n
        corr_hold = data.x_hold.T @ data.y_hold / n
        for mech in mechanisms:
            tho = Thresholdout(rng, 4 / math.sqrt(n), 1 / math.sqrt(n))

            def ask(train_value: float, hold_value: float, mech: str = mech, tho: Thresholdout = tho) -> float:
                if mech == "reuse":
                    return hold_value
                if mech == "gaussian":
                    return hold_value + float(rng.normal(0, 1 / math.sqrt(n)))
                return tho.answer(train_value, hold_value)

            candidates = np.flatnonzero(np.abs(corr_train) > cut)  # only these are worth asking the holdout about
            answers = np.array([ask(float(corr_train[j]), float(corr_hold[j])) for j in candidates])
            ok = (np.abs(answers) > cut) & (np.sign(corr_train[candidates]) == np.sign(answers))
            chosen = candidates[ok]
            queries[mech] += len(candidates) + len(sizes)
            chosen = chosen[np.argsort(-np.abs(corr_train[chosen]), kind="stable")]
            for i, m in enumerate(sizes):
                top = chosen[:m]
                signs = np.sign(corr_train[top])
                pred_train = np.where(data.x_train[:, top] @ signs >= 0, 1.0, -1.0)
                pred_hold = np.where(data.x_hold[:, top] @ signs >= 0, 1.0, -1.0)
                acc_train_corr = float(np.mean(pred_train * data.y_train))
                acc_hold_corr = float(np.mean(pred_hold * data.y_hold))
                reported = (1 + ask(acc_train_corr, acc_hold_corr)) / 2
                sums[mech][:, i] += ((1 + acc_train_corr) / 2, reported, true_accuracy(signs, data.p_agree[top]))
            det[mech] += tho.detections
    return {
        m: ReuseResult(
            sizes, sums[m][0] / trials, sums[m][1] / trials, sums[m][2] / trials, det[m] / trials, queries[m] / trials
        )
        for m in mechanisms
    }


# --- The chapter's table ------------------------------------------------------------------------

REUSE_N = 2000
REUSE_D = 5000
REUSE_SIZES = [10, 20, 50, 100, 150, 200]
REUSE_TRIALS = 10
SIGNAL_FEATURES = 20
SIGNAL_BIAS = 1.5 / math.sqrt(REUSE_N)  # agreement 1/2 + bias, a correlation of 3/sqrt(n)


def print_table(rng: np.random.Generator) -> None:
    print("== Fixed queries: Hoeffding and the union bound")
    for k in (10, 1000, 10**6):
        print(f"[Fixed k={k}] n for alpha=0.05 beta=0.05: {math.ceil(hoeffding_union_n(k, 0.05, 0.05))}")
    for n, k in ((1000, 2001), (10_000, 1000)):
        print(f"[Fixed n={n} k={k}] E max error <= {nonadaptive_error(n, k):.4f}")

    print("== Point-query attack")
    for n, domain in ((1000, 2000), (1000, 10_000)):
        pe, emp, pop = point_query_attack(rng, n, domain)
        exp_pop = point_query_population(n, domain)
        print(
            f"[Point n={n} N={domain}] queries={domain + 1}  point-query max error={pe:.4f}  "
            f"last query: empirical={emp:.0f} population={pop:.4f} expected={exp_pop:.4f}  "
            f"error={1 - exp_pop:.4f}  fixed-query bound={nonadaptive_error(n, domain + 1):.4f}"
        )

    print("== Boosting attack on a leaderboard (true accuracy 0.5)")
    for n in (1000, 10_000):
        for k in (10, 100, 1000, 10_000):
            mean, se = boosting_attack(rng, n, k, trials=20)
            print(
                f"[Boost n={n} k={k}] reported={mean:.4f}±{se:.4f}  overfit={mean - 0.5:.4f}  "
                f"approx sqrt(k/2n)/pi={boosting_approximation(n, k):.4f}  fixed-query bound={nonadaptive_error(n, k + 1):.4f}"
            )

    print("== Expected-error bounds and the sample size for alpha = 0.05")
    for k in (1, 10, 100, 1000, 10**4, 10**5, 10**6):
        na = required_n(nonadaptive_error, k, 0.05)
        sp = required_n(splitting_error, k, 0.05)
        ga = required_n(gaussian_error_value, k, 0.05)
        g = gaussian_error(ga, k)
        print(
            f"[Rates k={k}] fixed={na}  splitting={sp}  gaussian={ga}  "
            f"(sigma={g.sigma:.4f} eps={g.eps:.4f} delta={g.delta:.1e} noise={g.sample_error:.4f} "
            f"transfer={g.transfer:.4f})"
        )
    g = gaussian_error(10_000, 1000)
    print(
        f"[Gaussian n=10000 k=1000] bound={g.bound:.4f} sigma={g.sigma:.4f} eps={g.eps:.4f} delta={g.delta:.1e} "
        f"noise={g.sample_error:.4f} transfer={g.transfer:.4f}  splitting={splitting_error(10_000, 1000):.4f}  "
        f"fixed={nonadaptive_error(10_000, 1000):.4f}"
    )

    print("== Selection bias of a chosen query, n = 1000, k = 1000 null queries")
    eps_list = [0.01, 0.1, 1.0]
    sb = selection_bias(rng, 1000, 1000, eps_list, trials=2000)
    print(
        f"[Select argmax] bias={sb.argmax:.4f}±{sb.argmax_se:.4f}  max-inequality bound={nonadaptive_error(1000, 1000):.4f}"
    )
    for eps in eps_list:
        b, se = sb.exponential[eps]
        print(
            f"[Select eps={eps:g}] bias={b:.5f}±{se:.5f}  lemma e^eps-1={transfer_lemma_bound(eps):.5f}  "
            f"sharper (e^eps-1)/2={transfer_lemma_bound(eps) / 2:.5f}"
        )

    print("== Reusable holdout experiment")
    for informative, label in ((0, "no signal"), (SIGNAL_FEATURES, "signal")):
        res = holdout_reuse(rng, REUSE_N, REUSE_D, informative, SIGNAL_BIAS, REUSE_SIZES, REUSE_TRIALS)
        for mech, r in res.items():
            for i, m in enumerate(r.sizes):
                print(
                    f"[Reuse {label} {mech} m={m}] train={r.train[i]:.4f}  reported={r.reported[i]:.4f}  "
                    f"true={r.true[i]:.4f}"
                )
            if mech == "thresholdout":
                eps = r.detections * thresholdout_epsilon(1, REUSE_N, 1 / math.sqrt(REUSE_N))
                print(
                    f"[Reuse {label} thresholdout] queries={r.queries:.0f}  detections={r.detections:.1f}  eps={eps:.1f}"
                )
            if mech == "gaussian":
                rho_g = 2 * r.queries / REUSE_N  # correlation answers: sensitivity 2/n, σ = 1/sqrt(n), so ρ = 2/n each
                print(
                    f"[Reuse {label} gaussian] queries={r.queries:.0f}  rho={rho_g:.2f}  "
                    f"eps(delta=1e-5)={zcdp_to_dp(rho_g, 1e-5):.1f}"
                )
    print(
        f"[Thresholdout] eps per detection at n={REUSE_N}: {thresholdout_epsilon(1, REUSE_N, 1 / math.sqrt(REUSE_N)):.4f}"
    )


# --- Figures ------------------------------------------------------------------------------------


def plot_boosting(rng: np.random.Generator) -> Figure:
    """Overfitting of the boosting attack against k, for two holdout sizes, against the fixed-query bound."""
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    ks = [10, 30, 100, 300, 1000, 3000, 10_000]
    for n, color in ((1000, "tab:blue"), (10_000, "tab:orange")):
        over = [boosting_attack(rng, n, k, trials=10)[0] - 0.5 for k in ks]
        ax.plot(ks, over, "o-", color=color, label=f"boosting attack, n = {n:,}")
        ax.plot(ks, [boosting_approximation(n, k) for k in ks], ":", color=color, label=r"$\sqrt{k/2n}\,/\,\pi$")
        ax.plot(ks, [nonadaptive_error(n, k + 1) for k in ks], "--", color=color, label=f"fixed-query bound, n = {n:,}")
    ax.set(
        xscale="log",
        yscale="log",
        xlabel="number of submissions $k$",
        ylabel="reported accuracy $-$ true accuracy",
        title="Exact leaderboard scores overfit the holdout set",
    )
    ax.legend(fontsize=8)
    return fig


def plot_rates() -> Figure:
    """Sample size needed for expected error 0.05 against the number of queries, for the three methods."""
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    ks = [round(10 ** (e / 4)) for e in range(25)]
    ax.plot(ks, [required_n(nonadaptive_error, k, 0.05) for k in ks], label="fixed queries (not adaptive)")
    ax.plot(ks, [required_n(splitting_error, k, 0.05) for k in ks], label=r"sample splitting, $\propto k$")
    ax.plot(ks, [required_n(gaussian_error_value, k, 0.05) for k in ks], label=r"Gaussian noise, $\propto\sqrt{k}$")
    ax.set(
        xscale="log",
        yscale="log",
        xlabel="number of queries $k$",
        ylabel="records $n$ for expected error 0.05",
        title="Sample size needed to answer $k$ adaptive queries",
    )
    ax.legend(fontsize=8)
    return fig


def plot_selection_bias(rng: np.random.Generator) -> Figure:
    """Selection bias of the exponential mechanism against ε, with the lemma's bound and the argmax."""
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    eps_grid = [float(10 ** (e / 4)) for e in range(-12, 5)]
    sb = selection_bias(rng, 1000, 1000, eps_grid, trials=500)
    ax.plot(eps_grid, [sb.exponential[e][0] for e in eps_grid], "o-", label="exponential mechanism (measured)")
    ax.plot(eps_grid, [transfer_lemma_bound(e) for e in eps_grid], "--", label=r"lemma bound $e^\varepsilon - 1$")
    ax.axhline(sb.argmax, color="black", lw=1, ls=":", label="argmax, no privacy (measured)")
    ax.set(
        xscale="log",
        yscale="log",
        ylim=(1e-4, 1),
        xlabel=r"$\varepsilon$ of the selection",
        ylabel=r"bias: mean of $q_W(D) - q_W(\mathcal{D})$",
        title="Picking the best-looking of 1,000 null queries, $n = 1{,}000$",
    )
    ax.legend(fontsize=8)
    return fig


def plot_reuse(rng: np.random.Generator) -> Figure:
    """The reusable-holdout experiment: reported holdout accuracy against the number of selected features."""
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.6), sharey=True)
    sizes = [10, 20, 50, 100, 150, 200]
    for ax, informative, title in (
        (axes[0], 0, "No signal: every feature is noise"),
        (axes[1], SIGNAL_FEATURES, f"{SIGNAL_FEATURES} informative features"),
    ):
        res = holdout_reuse(rng, REUSE_N, REUSE_D, informative, SIGNAL_BIAS, sizes, trials=REUSE_TRIALS)
        ax.plot(sizes, res["reuse"].train, "s:", color="gray", label="training set")
        ax.plot(sizes, res["reuse"].reported, "o-", color="tab:red", label="holdout, reused as is")
        ax.plot(sizes, res["gaussian"].reported, "^-", color="tab:orange", label="holdout, Gaussian noise")
        ax.plot(sizes, res["thresholdout"].reported, "D-", color="tab:blue", label="holdout, Thresholdout")
        ax.plot(sizes, res["reuse"].true, "k--", label="true accuracy (reused holdout's selection)")
        ax.plot(sizes, res["thresholdout"].true, "b--", lw=1, label="true accuracy (Thresholdout's selection)")
        ax.set(xlabel="number of selected features $m$", title=title)
    axes[0].set_ylabel("accuracy")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, fontsize=8)
    fig.tight_layout(rect=(0, 0.14, 1, 1))  # leave room for the legend below the panels
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
        "ada_boosting.svg": plot_boosting(rng),
        "ada_selection_bias.svg": plot_selection_bias(rng),
        "ada_rates.svg": plot_rates(),
        "ada_reuse.svg": plot_reuse(rng),
    }
    for name, fig in figures.items():
        if name != "ada_reuse.svg":  # its legend sits below the panels, placed by hand
            fig.tight_layout()
        fig.savefig(out / name, metadata={"Date": None})  # no timestamp, so output is reproducible
        plt.close(fig)
        print(f"saved {out / name}")


if __name__ == "__main__":
    main()
