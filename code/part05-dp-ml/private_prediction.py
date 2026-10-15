"""Check the numbers in the private prediction chapter and draw its figures.

    uv run python code/part05-dp-ml/private_prediction.py
    uv run python code/part05-dp-ml/private_prediction.py --seed 1 --out /tmp/figures

Prints the numbers quoted in 05-private-prediction.md (they match exactly with the default seed), then
saves the chapter's figures.

The setting: the part's running example (make_task in dp_sgd.py: 10,000 private training records,
50-dimensional inputs, labels from a fixed random teacher network). The 20,000 test records are split
into a public pool (the first 10,000; its inputs are treated as public and unlabelled, and its labels
are used only to measure label quality) and an evaluation set (the last 10,000), on which every
accuracy is measured.

The private records are split into k disjoint shards by position (a partition fixed in advance, as a
hash of a record identifier would be). Each shard trains a teacher, a regularized logistic regression
(fit_logistic in dp_erm.py). A query is answered by a noisy argmax over the teachers' vote histogram:
Laplace noise (LNMax) or Gaussian noise (GNMax). PATE labels public inputs with GNMax and trains a
student network non-privately on them; its privacy cost depends only on the number of labels.

In-context learning is simulated, since no language model is available: an "in-context learner" is a
prototype classifier, sign(<Σ y_i x_i, x>), built from a handful of exemplars. It is a stand-in for the
mechanics (disjoint exemplar sets, a noisy vote, subsampling) and is far weaker than a language model.
"""

import os

# The models are tiny; multi-threaded BLAS only adds overhead.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import argparse
import math
from collections.abc import Sequence
from dataclasses import dataclass
from functools import cache
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
matplotlib.rcParams["svg.hashsalt"] = "private_prediction"  # stable element ids, so reruns don't churn the SVGs
matplotlib.rcParams["svg.fonttype"] = "none"  # keep text as text: smaller, diffable files

import matplotlib.pyplot as plt
import numpy as np
from dp_erm import LogisticData, fit_logistic, gaussian_delta
from dp_sgd import (
    ORDERS,
    Dataset,
    DpsgdConfig,
    Params,
    Task,
    accuracy,
    bisect,
    epsilon_from_rdp,
    make_task,
    noise_multiplier_for,
    subsampled_gaussian_rdp,
    train_dpsgd,
)
from matplotlib.figure import Figure
from numpy.typing import NDArray

# Figures are build output: _build/figures/<part>/, which the build copies into each book.
FIGURES = Path(__file__).resolve().parents[2] / "_build" / "figures" / Path(__file__).parent.name

Array = NDArray[np.float64]
IntArray = NDArray[np.int64]

DELTA = 1e-5
N_PUBLIC = 10_000  # the first half of the test records: the public, unlabelled pool
TEACHER_LAM = 0.01  # ridge coefficient of each teacher's logistic regression
EPS_GRID: tuple[float, ...] = (1.0, 2.0, 4.0, 8.0)  # budgets for the students
DPSGD_EPS: tuple[float, ...] = (0.5, 1.0, 2.0, 4.0, 8.0)  # budgets for the DP-SGD baseline
TEACHER_GRID: tuple[int, ...] = (25, 50, 100, 250, 500)
QUERY_GRID: tuple[int, ...] = (250, 500, 1000, 2000)
K_STUDENT = 250  # teachers used for the student experiments
STUDENT_STEPS = 2000


# --- The split: private training data, public pool, evaluation set -------------------------------


@dataclass(frozen=True)
class Split:
    private: Dataset  # the 10,000 training records: the only private data
    pool: Dataset  # public inputs; their labels are used only to measure label quality
    evaluation: Dataset


def make_split(task: Task) -> Split:
    pool = Dataset(task.test.x[:N_PUBLIC], task.test.y[:N_PUBLIC])
    evaluation = Dataset(task.test.x[N_PUBLIC:], task.test.y[N_PUBLIC:])
    return Split(task.train, pool, evaluation)


# --- Teachers and their votes -----------------------------------------------------------------


def agreement(a: Array, b: Array) -> float:
    """Fraction of positions where two label vectors agree."""
    same: NDArray[np.bool_] = a == b
    return float(same.mean())


def with_bias(x: Array) -> Array:
    return np.hstack([x, np.ones((x.shape[0], 1))])


def train_teachers(data: Dataset, k: int, lam: float = TEACHER_LAM) -> Array:
    """k logistic regressions on k disjoint shards of consecutive records; returns (k, d + 1) weights."""
    size = data.y.size // k
    return np.array(
        [
            fit_logistic(
                LogisticData(with_bias(data.x[j * size : (j + 1) * size]), data.y[j * size : (j + 1) * size]), lam
            )
            for j in range(k)
        ]
    )


def teacher_accuracies(weights: Array, data: Dataset) -> Array:
    preds = np.where(with_bias(data.x) @ weights.T >= 0, 1.0, -1.0)
    return (preds == data.y[:, None]).mean(axis=0)


def votes_for(weights: Array, x: Array) -> IntArray:
    """Vote histogram for the two labels: column 0 counts votes for -1, column 1 for +1."""
    plus = (with_bias(x) @ weights.T >= 0).sum(axis=1).astype(np.int64)
    return np.stack([weights.shape[0] - plus, plus], axis=1)


def label_index(y: Array) -> IntArray:
    return (y > 0).astype(np.int64)


def correct_gap(votes: IntArray, y: Array) -> Array:
    """Votes for the true label minus votes for the other: positive when the plurality is right."""
    idx = label_index(y)
    rows = np.arange(y.size)
    return (votes[rows, idx] - votes[rows, 1 - idx]).astype(np.float64)


# --- The noisy argmax: LNMax and GNMax ------------------------------------------------------------


def gnmax(rng: np.random.Generator, votes: IntArray, sigma: float) -> IntArray:
    """argmax_i (n_i + N(0, σ²)), one independent draw per class and query."""
    return np.argmax(votes + sigma * rng.standard_normal(votes.shape), axis=1).astype(np.int64)


def lnmax(rng: np.random.Generator, votes: IntArray, b: float) -> IntArray:
    return np.argmax(votes + rng.laplace(0.0, b, votes.shape), axis=1).astype(np.int64)


def gnmax_rdp(alpha: float, sigma: float) -> float:
    """(α, α/σ²)-RDP: the vote histogram has ℓ2 sensitivity √2 (one count down, one up)."""
    return alpha / sigma**2


def gnmax_epsilon(queries: int, sigma: float, delta: float = DELTA) -> float:
    return epsilon_from_rdp([queries * gnmax_rdp(a, sigma) for a in ORDERS], delta)[0]


@cache
def gnmax_rho(eps: float, delta: float = DELTA) -> float:
    """The largest ρ for which the RDP line ρα converts to at most (ε, δ)."""
    return bisect(lambda r: epsilon_from_rdp([r * a for a in ORDERS], delta)[0] > eps, 1e-9, 1e4)


def gnmax_sigma(queries: int, eps: float, delta: float = DELTA) -> float:
    """The smallest σ for which `queries` GNMax answers are (ε, δ)-DP by the RDP accountant: Q/σ² = ρ(ε, δ)."""
    return math.sqrt(queries / gnmax_rho(eps, delta))


def gnmax_gdp_epsilon(queries: int, sigma: float, delta: float = DELTA) -> float:
    """Exact ε for `queries` GNMax answers, from the composed Gaussian: μ = √(2Q)/σ."""
    mu = math.sqrt(2 * queries) / sigma
    return bisect(lambda e: gaussian_delta(e, mu) <= delta, 0.0, 1000.0)


def rr_rdp(alpha: float, eps0: float) -> float:
    """RDP curve of ε0-randomized response, valid for every ε0-DP mechanism (concentrated DP chapter)."""
    p = 1 / (1 + math.exp(-eps0))
    t1 = alpha * math.log(p) + (1 - alpha) * math.log1p(-p)
    t2 = alpha * math.log1p(-p) + (1 - alpha) * math.log(p)
    top = max(t1, t2)
    return (top + math.log(math.exp(t1 - top) + math.exp(t2 - top))) / (alpha - 1)


def lnmax_epsilon(queries: int, b: float, delta: float = DELTA) -> float:
    """LNMax with Lap(b) on each count is (2/b)-DP per query (report-noisy-max, scores that are not monotone)."""
    return epsilon_from_rdp([queries * rr_rdp(a, 2 / b) for a in ORDERS], delta)[0]


def lnmax_scale(queries: int, eps: float, delta: float = DELTA) -> float:
    return bisect(lambda b: lnmax_epsilon(queries, b, delta) <= eps, 1e-2, 1e6)


def normal_cdf(x: Array) -> Array:
    erfc = np.vectorize(math.erfc)
    return 0.5 * erfc(-x / math.sqrt(2))


def answer_accuracy(gaps: Array, sigma: float) -> float:
    """Expected fraction of correct GNMax answers on two classes: Pr[correct] = Φ(gap/(√2 σ)) per query."""
    if sigma == 0:
        return float(np.mean(np.where(gaps > 0, 1.0, np.where(gaps == 0, 0.5, 0.0))))
    values, counts = np.unique(gaps, return_counts=True)
    return float(np.sum(counts * normal_cdf(values / (math.sqrt(2) * sigma))) / gaps.size)


def largest_queries_above(gaps: Array, eps: float, target: float) -> int:
    """The largest number of queries Q for which GNMax answers at (ε, δ) still have expected accuracy ≥ target."""
    if answer_accuracy(gaps, gnmax_sigma(1, eps)) < target:
        return 0
    lo, hi = 1, 2
    while answer_accuracy(gaps, gnmax_sigma(hi, eps)) >= target:
        lo, hi = hi, hi * 2
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if answer_accuracy(gaps, gnmax_sigma(mid, eps)) >= target:
            lo = mid
        else:
            hi = mid
    return lo


# --- Data-dependent analysis --------------------------------------------------------------------


def log_normal_cdf(x: float) -> float:
    """ln Φ(x), with the asymptotic tail for very negative x so that it never underflows."""
    if x > -30:
        return math.log(0.5 * math.erfc(-x / math.sqrt(2)))
    return -x * x / 2 - math.log(-x * math.sqrt(2 * math.pi)) + math.log1p(-1 / (x * x))


def prop7_q(votes: Sequence[int], sigma: float) -> float:
    """Pr[GNMax ≠ plurality] ≤ Σ_{i≠i*} ½ erfc((n_i* - n_i)/(2σ)) (union bound; Papernot et al. 2018, Prop. 7)."""
    top = max(votes)
    star = list(votes).index(top)
    return min(1.0, sum(0.5 * math.erfc((top - n) / (2 * sigma)) for i, n in enumerate(votes) if i != star))


def pure_dd_value(alpha: float, q: float, eps0: float) -> float:
    """(1/(α-1)) ln[(1-q)^α (1 - e^ε0 q)^(1-α) + q e^((α-1)ε0)], for 0 < q < e^-ε0."""
    t1 = alpha * math.log1p(-q) + (1 - alpha) * math.log1p(-math.exp(eps0) * q)
    t2 = math.log(q) + (alpha - 1) * eps0
    top = max(t1, t2)
    return (top + math.log(math.exp(t1 - top) + math.exp(t2 - top))) / (alpha - 1)


def pure_dd_rdp(alpha: float, q: float, eps0: float) -> float:
    """Data-dependent RDP of an ε0-DP mechanism whose likely output fails with probability at most q
    (the Laplace form of Papernot et al. 2017, Theorem 3): the largest value of pure_dd_value over
    failure probabilities in (0, q], capped by the data-independent curve."""
    cap = rr_rdp(alpha, eps0)
    if q <= 0:
        return 0.0
    if q * math.exp(eps0) >= 1:
        return cap
    return min(cap, max(pure_dd_value(alpha, q * j / 200, eps0) for j in range(1, 201)))


def gnmax_dd_rdp(alpha: float, q: float, sigma: float) -> float:
    """Data-dependent RDP of GNMax at order α (Papernot et al. 2018, Theorem 6 with the orders of
    Proposition 10), capped by the data-independent α/σ²."""
    cap = gnmax_rdp(alpha, sigma)
    if q <= 0:
        return 0.0
    if q >= 1:
        return cap
    mu2 = sigma * math.sqrt(math.log(1 / q))
    mu1 = mu2 + 1
    e1, e2 = mu1 / sigma**2, mu2 / sigma**2
    if mu1 < alpha or mu2 <= 1:
        return cap
    if math.log(q) > (mu2 - 1) * e2 - mu2 * math.log(mu1 / (mu1 - 1) * mu2 / (mu2 - 1)):
        return cap
    log_a = math.log1p(-q) - math.log1p(-math.exp((mu2 - 1) / mu2 * (math.log(q) + e2)))
    log_b = e1 - math.log(q) / (mu1 - 1)
    t1 = math.log1p(-q) + (alpha - 1) * log_a
    t2 = math.log(q) + (alpha - 1) * log_b
    top = max(t1, t2)
    return min(cap, (top + math.log(math.exp(t1 - top) + math.exp(t2 - top))) / (alpha - 1))


def _renyi_bernoulli(alpha: float, lp: tuple[float, float], lq: tuple[float, float]) -> float:
    """D_α(P‖Q) for two-point distributions given as (ln P(a), ln P(b))."""
    t1 = alpha * lp[0] + (1 - alpha) * lq[0]
    t2 = alpha * lp[1] + (1 - alpha) * lq[1]
    top = max(t1, t2)
    return (top + math.log(math.exp(t1 - top) + math.exp(t2 - top))) / (alpha - 1)


def exact_binary_rdp(alpha: float, gap: int, sigma: float) -> float:
    """Exact data-dependent RDP of GNMax on two classes: the output is a coin with
    Pr[first class] = Φ(gap/(√2σ)); one teacher moves the gap by ±2. Both directions, worst neighbour."""
    s = math.sqrt(2) * sigma

    def coin(g: int) -> tuple[float, float]:
        return log_normal_cdf(g / s), log_normal_cdf(-g / s)

    here = coin(gap)
    worst = 0.0
    for other in (gap - 2, gap + 2):
        there = coin(other)
        worst = max(worst, _renyi_bernoulli(alpha, here, there), _renyi_bernoulli(alpha, there, here))
    return worst


def gnmax_dd_curve(votes: IntArray, sigma: float) -> list[float]:
    """Total data-dependent RDP curve of GNMax over the queries in `votes` (rows); equal histograms cost the same."""
    rows, counts = np.unique(votes, axis=0, return_counts=True)
    qs = [(prop7_q([int(v) for v in row], sigma), int(c)) for row, c in zip(rows, counts, strict=True)]
    return [sum(c * gnmax_dd_rdp(a, q, sigma) for q, c in qs) for a in ORDERS]


def gnmax_dd_epsilon(votes: IntArray, sigma: float, delta: float = DELTA) -> float:
    return epsilon_from_rdp(gnmax_dd_curve(votes, sigma), delta)[0]


@dataclass(frozen=True)
class Confident:
    answered: NDArray[np.bool_]
    labels: IntArray  # meaningful where answered


def confident_gnmax(rng: np.random.Generator, votes: IntArray, threshold: float, s1: float, s2: float) -> Confident:
    """Confident-GNMax (Papernot et al. 2018, Algorithm 1): answer only if max_i n_i + N(0, σ1²) ≥ T."""
    top: Array = votes.max(axis=1).astype(np.float64)
    noisy_top: Array = top + s1 * rng.standard_normal(top.size)
    answered: NDArray[np.bool_] = np.greater_equal(noisy_top, threshold)
    return Confident(answered, gnmax(rng, votes, s2))


def confident_epsilon(pool: int, answered: int, s1: float, s2: float, delta: float = DELTA) -> float:
    """Data-independent ε: the check (sensitivity 1, ρ = 1/(2σ1²)) on every query, GNMax (ρ = 1/σ2²) on answered ones.
    Valid as a Rényi filter: the per-query spends are lines in α, fixed before each step."""
    rho = pool / (2 * s1**2) + answered / s2**2
    return epsilon_from_rdp([rho * a for a in ORDERS], delta)[0]


def confident_dd_epsilon(pool: int, answered_votes: IntArray, s1: float, s2: float, delta: float = DELTA) -> float:
    check = [pool * a / (2 * s1**2) for a in ORDERS]
    answers = gnmax_dd_curve(answered_votes, s2)
    return epsilon_from_rdp([c + v for c, v in zip(check, answers, strict=True)], delta)[0]


# --- The student --------------------------------------------------------------------------------


def train_student(rng: np.random.Generator, x: Array, labels: Array, steps: int = STUDENT_STEPS) -> Params:
    """The part's two-layer network, trained non-privately on public inputs and their (private) labels."""
    q = min(1.0, 100 / labels.size)
    params, _ = train_dpsgd(rng, Dataset(x, labels), DpsgdConfig(q=q, sigma=0.0, clip=math.inf, lr=0.5, steps=steps))
    return params


def to_labels(idx: IntArray) -> Array:
    return np.where(idx == 1, 1.0, -1.0)


@dataclass(frozen=True)
class StudentResult:
    sigma: float
    label_acc: float
    student_acc: float


def pate_student(
    rng: np.random.Generator, split: Split, pool_votes: IntArray, queries: int, eps: float
) -> StudentResult:
    sigma = gnmax_sigma(queries, eps)
    labels = to_labels(gnmax(rng, pool_votes[:queries], sigma))
    label_acc = agreement(labels, split.pool.y[:queries])
    student = train_student(rng, split.pool.x[:queries], labels)
    return StudentResult(sigma, label_acc, accuracy(student, split.evaluation))


@cache
def dpsgd_sigma(eps: float) -> float:
    return noise_multiplier_for(0.01, 10_000, eps, DELTA)


def dpsgd_accuracy(rng: np.random.Generator, split: Split, eps: float) -> float:
    params, _ = train_dpsgd(rng, split.private, DpsgdConfig(sigma=dpsgd_sigma(eps)))
    return accuracy(params, split.evaluation)


# --- In-context learning, simulated ---------------------------------------------------------------


def prototype(x: Array, y: Array) -> Array:
    """The stand-in in-context learner: the direction Σ y_i x_i of its exemplars."""
    return y @ x


def membership_auc(rng: np.random.Generator, data: Dataset, shots: int, trials: int) -> float:
    """AUC of the attack 'score y·<w, x>' against one s-shot learner: a member exemplar against a non-member."""
    members: list[float] = []
    others: list[float] = []
    n = data.y.size
    for _ in range(trials):
        idx = rng.choice(n, shots + 1, replace=False)
        w = prototype(data.x[idx[:shots]], data.y[idx[:shots]])
        members.append(float(data.y[idx[0]] * (data.x[idx[0]] @ w)))
        others.append(float(data.y[idx[shots]] * (data.x[idx[shots]] @ w)))
    m, o = np.array(members), np.array(others)
    diff: Array = m[:, None] - o[None, :]
    return float(np.mean(np.sign(diff)) / 2 + 0.5)


def icl_rate(learners: int, shots: int, n: int) -> float:
    return learners * shots / n


def icl_epsilon(queries: int, rate: float, sigma: float, delta: float = DELTA) -> float:
    """Poisson-subsampled GNMax: a record joins the query's exemplars with probability `rate`, and then moves one
    vote, a shift of length √2 against noise σ: the subsampled Gaussian with noise multiplier σ/√2."""
    if sigma <= 0:
        return math.inf
    curve = [queries * subsampled_gaussian_rdp(a, rate, sigma / math.sqrt(2)) for a in ORDERS]
    return epsilon_from_rdp(curve, delta)[0]


def icl_sigma(queries: int, rate: float, eps: float, delta: float = DELTA) -> float:
    return bisect(lambda s: icl_epsilon(queries, rate, s, delta) <= eps, 0.05, 200.0, 30)


def icl_answers(
    rng: np.random.Generator, private: Dataset, queries: Dataset, learners: int, rate: float, sigma: float
) -> Array:
    """For each query: Poisson-sample exemplars at `rate`, split them into `learners` disjoint groups by record
    index (fixed in advance), let each group's prototype vote, and release the GNMax label (σ = 0: plain vote)."""
    n = private.y.size
    group = np.arange(n) % learners
    out = np.empty(queries.y.size)
    for t in range(queries.y.size):
        keep = rng.random(n) < rate
        sums = np.zeros((learners, private.x.shape[1]))
        np.add.at(sums, group[keep], private.y[keep, None] * private.x[keep])
        scores: Array = sums @ queries.x[t]
        plus = int(np.count_nonzero(scores > 0))
        noisy = np.array([learners - plus, plus]) + sigma * rng.standard_normal(2)
        out[t] = 1.0 if noisy[1] > noisy[0] else -1.0
    return out


def disjoint_learners(private: Dataset, shots: int) -> Array:
    """All private records split into n/shots disjoint s-shot learners: (n/shots, d) prototypes."""
    m = private.y.size // shots
    return np.array(
        [prototype(private.x[j * shots : (j + 1) * shots], private.y[j * shots : (j + 1) * shots]) for j in range(m)]
    )


def prototype_votes(protos: Array, x: Array) -> IntArray:
    plus = (x @ protos.T > 0).sum(axis=1).astype(np.int64)
    return np.stack([protos.shape[0] - plus, plus], axis=1)


def sign_accuracy(w: Array, data: Dataset) -> float:
    return agreement(np.where(data.x @ w > 0, 1.0, -1.0), data.y)


def single_learner_accuracy(rng: np.random.Generator, split: Split, shots: int, draws: int = 50) -> float:
    n = split.private.y.size
    accs: list[float] = []
    for _ in range(draws):
        idx = rng.choice(n, shots, replace=False)
        accs.append(sign_accuracy(prototype(split.private.x[idx], split.private.y[idx]), split.evaluation))
    return float(np.mean(accs))


# --- The chapter's numbers -----------------------------------------------------------------------


def print_table(rng: np.random.Generator) -> None:
    split = make_split(make_task())
    ev = split.evaluation

    print("== Noise for the noisy argmax (delta=1e-5)")
    for eps in (1.0, 4.0):
        print(f"eps={eps:g}: largest RDP slope rho={gnmax_rho(eps):.4f}")
        for queries in (1, 100, 1000, 10000):
            s = gnmax_sigma(queries, eps)
            b = lnmax_scale(queries, eps)
            print(
                f"eps={eps:g}, Q={queries}: GNMax sigma={s:.3g} (exact GDP eps={gnmax_gdp_epsilon(queries, s):.3f})  "
                f"LNMax b={b:.3g} (sd={math.sqrt(2) * b:.3g})"
            )

    print("== Teachers")
    gaps_by_k: dict[int, Array] = {}
    for k in TEACHER_GRID:
        w = train_teachers(split.private, k)
        t_acc = teacher_accuracies(w, ev)
        votes = votes_for(w, ev.x)
        gaps = correct_gap(votes, ev.y)
        gaps_by_k[k] = gaps
        ens = answer_accuracy(gaps, 0.0)
        frac = float(np.median(np.abs(gaps))) / k
        print(
            f"k={k}: shard={split.private.y.size // k}  teacher acc={float(t_acc.mean()):.4f}  "
            f"plurality acc={ens:.4f}  median |gap|/k={frac:.3f}"
        )

    print("== Answering queries directly (expected accuracy on the evaluation set)")
    for eps in (1.0, 4.0):
        for queries in (100, 1000, 10000):
            s = gnmax_sigma(queries, eps)
            accs = "  ".join(f"k={k}: {answer_accuracy(gaps_by_k[k], s):.4f}" for k in TEACHER_GRID)
            print(f"eps={eps:g}, Q={queries}, sigma={s:.1f}: {accs}")

    print("== DP-SGD baseline on the evaluation set (2 runs)")
    dpsgd: dict[float, float] = {}
    for eps in DPSGD_EPS:
        accs = [dpsgd_accuracy(rng, split, eps) for _ in range(2)]
        dpsgd[eps] = float(np.mean(accs))
        print(f"DP-SGD eps={eps:g} (sigma={dpsgd_sigma(eps):.3g}): accuracy {np.mean(accs):.4f} +- {np.std(accs):.4f}")
    for eps in DPSGD_EPS:
        best_k = max(TEACHER_GRID, key=lambda k: largest_queries_above(gaps_by_k[k], eps, dpsgd[eps]))
        q_max = largest_queries_above(gaps_by_k[best_k], eps, dpsgd[eps])
        print(f"eps={eps:g}: answers beat DP-SGD ({dpsgd[eps]:.4f}) for up to Q={q_max} queries (k={best_k})")

    print("== Vote examples (k=100, eps=4, Q=1000)")
    w100 = train_teachers(split.private, 100)
    pool_votes100 = votes_for(w100, split.pool.x)
    s = gnmax_sigma(1000, 4.0)
    for i in example_queries(pool_votes100):
        v = pool_votes100[i]
        g = int(abs(v[1] - v[0]))
        win = float(normal_cdf(np.array([g / (math.sqrt(2) * s)]))[0])
        print(f"query {i}: votes (-1: {v[0]}, +1: {v[1]})  gap={g}  sigma={s:.1f}  Pr[plurality wins]={win:.4f}")

    print("== PATE student (k=250), labels from GNMax on the public pool")
    w = train_teachers(split.private, K_STUDENT)
    pool_votes = votes_for(w, split.pool.x)
    plural = to_labels(np.argmax(pool_votes, axis=1).astype(np.int64))
    print(f"plurality label accuracy on the pool: {agreement(plural, split.pool.y):.4f}")
    for queries in QUERY_GRID:
        true_acc = accuracy(train_student(rng, split.pool.x[:queries], split.pool.y[:queries]), ev)
        plural_acc = accuracy(train_student(rng, split.pool.x[:queries], plural[:queries]), ev)
        print(f"Q={queries}: student on true labels {true_acc:.4f}  on noiseless plurality labels {plural_acc:.4f}")
    best: dict[float, tuple[int, float]] = {}
    for eps in EPS_GRID:
        for queries in QUERY_GRID:
            r = pate_student(rng, split, pool_votes, queries, eps)
            print(
                f"eps={eps:g}, Q={queries}: sigma={r.sigma:.1f}  label acc {r.label_acc:.4f}  "
                f"student acc {r.student_acc:.4f}"
            )
            if eps not in best or r.student_acc > best[eps][1]:
                best[eps] = (queries, r.student_acc)
    for eps in EPS_GRID:
        print(f"eps={eps:g}: best student {best[eps][1]:.4f} at Q={best[eps][0]}  DP-SGD {dpsgd[eps]:.4f}")

    print("== Data-dependent analysis")
    sigma_dd = 40.0
    for gap in (0, 50, 100, 150, 200):
        qt = 0.5 * math.erfc(gap / (2 * sigma_dd))
        print(
            f"sigma=40, gap={gap}: q~={qt:.3g}  order 8: independent {gnmax_rdp(8, sigma_dd):.3g}  "
            f"Theorem 6 {gnmax_dd_rdp(8, qt, sigma_dd):.3g}  exact two-class {exact_binary_rdp(8, gap, sigma_dd):.3g}"
        )
    eps0 = 0.1
    for q in (0.5, 0.1, 0.01, 0.001):
        print(
            f"LNMax eps0={eps0:g}, q={q:g}: order 8 independent {rr_rdp(8, eps0):.3g}  "
            f"data-dependent {pure_dd_rdp(8, q, eps0):.3g}"
        )
    big = float(np.mean(np.abs(pool_votes[:, 1] - pool_votes[:, 0]) >= 150))
    print(f"k=250: fraction of pool queries with gap >= 150: {big:.4f}")
    for queries in (250, 1000, 4000):
        votes = pool_votes[:queries]
        print(
            f"k=250, sigma=40, Q={queries}: independent eps={gnmax_epsilon(queries, sigma_dd):.2f}  "
            f"data-dependent eps={gnmax_dd_epsilon(votes, sigma_dd):.2f}"
        )
    # Local sensitivity of the data-dependent cost: one teacher switches every vote to the minority.
    votes = pool_votes[:1000]
    w_flip = w.copy()
    w_flip[0] = -w_flip[0]
    flipped = votes_for(w_flip, split.pool.x[:1000])
    print(
        f"Q=1000: data-dependent eps={gnmax_dd_epsilon(votes, sigma_dd):.3f}; "
        f"after one teacher flips its votes: {gnmax_dd_epsilon(flipped, sigma_dd):.3f}"
    )

    print("== Confident-GNMax (k=250, T=200, sigma1=150, sigma2=40)")
    for pool in (1000, 4000):
        conf = confident_gnmax(rng, pool_votes[:pool], 200.0, 150.0, 40.0)
        ans = np.flatnonzero(conf.answered)
        labels = to_labels(conf.labels[ans])
        label_acc = agreement(labels, split.pool.y[ans])
        student = train_student(rng, split.pool.x[ans], labels)
        print(
            f"pool={pool}: answered {ans.size}  label acc {label_acc:.4f}  "
            f"independent eps={confident_epsilon(pool, ans.size, 150.0, 40.0):.2f}  "
            f"data-dependent eps={confident_dd_epsilon(pool, pool_votes[ans], 150.0, 40.0):.2f}  "
            f"student acc {accuracy(student, ev):.4f}"
        )

    print("== In-context learning, simulated")
    for shots in (1, 4, 16):
        print(f"{shots}-shot learner: membership AUC {membership_auc(rng, split.private, shots, 400):.3f}")
    learners, shots = 25, 16
    rate = icl_rate(learners, shots, split.private.y.size)
    print(f"one {shots}-shot learner: accuracy {single_learner_accuracy(rng, split, shots):.4f} (50 draws)")
    queries = Dataset(ev.x[:2000], ev.y[:2000])
    plain = icl_answers(rng, split.private, queries, learners, rate, 0.0)
    print(f"{learners} learners x {shots} shots, rate q={rate:g}, no noise: accuracy {agreement(plain, queries.y):.4f}")
    for eps in (1.0, 4.0):
        for n_q in (100, 1000, 10000):
            s = icl_sigma(n_q, rate, eps)
            noisy = icl_answers(rng, split.private, queries, learners, rate, s)
            print(f"eps={eps:g}, Q={n_q}: sigma={s:.2f}  accuracy {agreement(noisy, queries.y):.4f}")
    protos = disjoint_learners(split.private, shots)
    pv = prototype_votes(protos, split.pool.x)
    print(
        f"{protos.shape[0]} disjoint {shots}-shot learners: plurality accuracy "
        f"{answer_accuracy(correct_gap(prototype_votes(protos, ev.x), ev.y), 0.0):.4f}"
    )
    for n_pub in (16, 64, 256):
        s = gnmax_sigma(n_pub, 1.0)
        labels = to_labels(gnmax(rng, pv[:n_pub], s))
        student = prototype(split.pool.x[:n_pub], labels)
        print(
            f"student prompt with {n_pub} public exemplars, eps=1: sigma={s:.1f}  "
            f"label acc {agreement(labels, split.pool.y[:n_pub]):.4f}  accuracy {sign_accuracy(student, ev):.4f}"
        )


def example_queries(votes: IntArray) -> list[int]:
    """Three pool queries with a large, a middling and a small gap (for k = 100)."""
    gaps = np.abs(votes[:, 1] - votes[:, 0])
    return [int(np.flatnonzero(gaps == g)[0]) for g in (96, 40, 4)]


# --- Figures ---------------------------------------------------------------------------------


def plot_votes() -> Figure:
    split = make_split(make_task())
    votes = votes_for(train_teachers(split.private, 100), split.pool.x)
    sigma = gnmax_sigma(1000, 4.0)
    fig, axes = plt.subplots(1, 4, figsize=(12, 3.4), gridspec_kw={"width_ratios": [1, 1, 1, 1.6]})
    for ax, i in zip(axes[:3], example_queries(votes), strict=True):
        v = votes[i]
        ax.bar([0, 1], v, color=["tab:blue", "tab:orange"], width=0.6)
        ax.errorbar([0, 1], v, yerr=[sigma, sigma], fmt="none", ecolor="black", capsize=6)
        g = abs(int(v[1] - v[0]))
        win = float(normal_cdf(np.array([g / (math.sqrt(2) * sigma)]))[0])
        ax.set_xticks([0, 1], ["label −1", "label +1"])
        ax.set_ylim(-60, 160)
        ax.axhline(0, color="gray", lw=0.5)
        ax.set_title(f"gap {g}: plurality wins\nwith probability {win:.2f}", fontsize=10)
    axes[0].set_ylabel("votes of 100 teachers")
    frac = votes[:, 1] / 100
    axes[3].hist(frac, bins=np.linspace(0, 1, 26), color="tab:gray")
    axes[3].set_xlabel("fraction of teachers voting +1")
    axes[3].set_ylabel("public queries")
    axes[3].set_title("all 10,000 public queries", fontsize=10)
    return fig


def plot_epsilon_queries() -> Figure:
    split = make_split(make_task())
    votes = votes_for(train_teachers(split.private, K_STUDENT), split.pool.x)
    qs = (10, 30, 100, 300, 1000, 3000, 10000)
    sigma = 40.0
    indep = [gnmax_epsilon(q, sigma) for q in qs]
    dep = [gnmax_dd_epsilon(votes[:q], sigma) for q in qs]
    lap = [lnmax_epsilon(q, sigma / math.sqrt(2)) for q in qs]  # the same noise standard deviation
    fig, ax = plt.subplots(figsize=(6.5, 4.0))
    ax.loglog(qs, indep, marker="o", color="black", label=r"GNMax, data-independent, $\sigma = 40$")
    ax.loglog(qs, dep, marker="s", color="tab:red", label=r"GNMax, data-dependent, $\sigma = 40$")
    ax.loglog(qs, lap, marker="^", color="tab:blue", ls="--", label=r"LNMax, same noise standard deviation")
    ax.set_xlabel("number of answered queries $Q$")
    ax.set_ylabel(r"$\varepsilon$ at $\delta = 10^{-5}$")
    ax.set_title("250 teachers, public queries in order")
    ax.legend(fontsize=8)
    return fig


def plot_dd_cost() -> Figure:
    sigma = 40.0
    gaps = list(range(0, 251, 5))
    alpha = 8.0
    fig, ax = plt.subplots(figsize=(6.5, 4.0))
    ax.semilogy(gaps, [gnmax_rdp(alpha, sigma)] * len(gaps), color="black", label=r"data-independent $\alpha/\sigma^2$")
    ax.semilogy(
        gaps,
        [gnmax_dd_rdp(alpha, 0.5 * math.erfc(g / (2 * sigma)), sigma) for g in gaps],
        color="tab:red",
        label="data-dependent bound (Theorem 6)",
    )
    ax.semilogy(
        gaps, [exact_binary_rdp(alpha, g, sigma) for g in gaps], color="tab:blue", ls="--", label="exact, two classes"
    )
    ax.set_xlabel("gap between the two vote counts")
    ax.set_ylabel(r"Rényi DP cost of one answer, order $\alpha = 8$")
    ax.set_title(r"GNMax with $\sigma = 40$")
    ax.legend(fontsize=8)
    return fig


def dpsgd_curve(rng: np.random.Generator, split: Split, eps_list: Sequence[float]) -> dict[float, float]:
    """One DP-SGD run per budget, shared by the figures."""
    return {e: dpsgd_accuracy(rng, split, e) for e in eps_list}


def plot_answers(split: Split, dpsgd: dict[float, float]) -> Figure:
    ev = split.evaluation
    qs = (1, 3, 10, 30, 100, 300, 1000, 3000, 10000)
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), sharey=True)
    gaps = {k: correct_gap(votes_for(train_teachers(split.private, k), ev.x), ev.y) for k in (25, 100, 500)}
    colors = {25: "tab:blue", 100: "black", 500: "tab:red"}
    for ax, eps in zip(axes, (1.0, 4.0), strict=True):
        for k, g in gaps.items():
            accs = [answer_accuracy(g, gnmax_sigma(q, eps)) for q in qs]
            ax.semilogx(qs, accs, marker="o", color=colors[k], label=f"{k} teachers")
        ax.axhline(dpsgd[eps], color="tab:gray", ls="--", label="DP-SGD model, any number of queries")
        ax.set_xlabel("number of queries answered $Q$")
        ax.set_title(rf"$\varepsilon = {eps:g}$, $\delta = 10^{{-5}}$")
    axes[0].set_ylabel("expected accuracy of the answers")
    axes[0].legend(fontsize=8)
    return fig


def plot_student(rng: np.random.Generator, split: Split, dpsgd: dict[float, float]) -> Figure:
    w = train_teachers(split.private, K_STUDENT)
    pool_votes = votes_for(w, split.pool.x)
    eps_list = EPS_GRID
    fig, ax = plt.subplots(figsize=(6.5, 4.0))
    for queries, color in ((500, "tab:blue"), (2000, "tab:red")):
        accs = [pate_student(rng, split, pool_votes, queries, e).student_acc for e in eps_list]
        ax.semilogx(eps_list, accs, marker="o", color=color, label=f"PATE student, {queries} labels")
    ax.semilogx(eps_list, [dpsgd[e] for e in eps_list], marker="s", color="black", label="DP-SGD")
    ev = split.evaluation
    plural = answer_accuracy(correct_gap(votes_for(w, ev.x), ev.y), 0.0)
    ax.axhline(plural, color="tab:gray", ls=":", label="noiseless vote of 250 teachers")
    ax.minorticks_off()
    ax.set_xticks(eps_list, [f"{e:g}" for e in eps_list])
    ax.set_xlabel(r"$\varepsilon$ at $\delta = 10^{-5}$")
    ax.set_ylabel("test accuracy")
    ax.legend(fontsize=8)
    return fig


def plot_icl(rng: np.random.Generator, split: Split) -> Figure:
    learners, shots = 25, 16
    rate = icl_rate(learners, shots, split.private.y.size)
    queries = Dataset(split.evaluation.x[:1000], split.evaluation.y[:1000])
    qs = (10, 100, 1000, 10000, 100000)
    fig, ax = plt.subplots(figsize=(6.5, 4.0))
    for eps, color in ((1.0, "tab:blue"), (4.0, "tab:red")):
        accs = [
            agreement(icl_answers(rng, split.private, queries, learners, rate, icl_sigma(q, rate, eps)), queries.y)
            for q in qs
        ]
        ax.semilogx(qs, accs, marker="o", color=color, label=rf"private vote, $\varepsilon = {eps:g}$")
    plain = agreement(icl_answers(rng, split.private, queries, learners, rate, 0.0), queries.y)
    ax.axhline(plain, color="black", ls="--", label="vote without noise")
    single = single_learner_accuracy(rng, split, shots)
    ax.axhline(single, color="tab:gray", ls=":", label=f"one {shots}-shot learner")
    ax.set_xlabel("number of queries answered")
    ax.set_ylabel("accuracy of the answers")
    ax.set_title(f"{learners} stand-in learners, {shots} exemplars each, per query")
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
    fig_rng = np.random.default_rng(seed + 1)
    split = make_split(make_task())
    dpsgd = dpsgd_curve(fig_rng, split, EPS_GRID)
    figures = {
        "pate_votes.svg": plot_votes(),
        "pate_answers.svg": plot_answers(split, dpsgd),
        "pate_student.svg": plot_student(fig_rng, split, dpsgd),
        "pate_dd_cost.svg": plot_dd_cost(),
        "pate_epsilon_queries.svg": plot_epsilon_queries(),
        "pate_icl.svg": plot_icl(fig_rng, split),
    }
    for name, fig in figures.items():
        fig.tight_layout()
        fig.savefig(out / name, metadata={"Date": None})  # no timestamp, so output is reproducible
        plt.close(fig)
        print(f"saved {out / name}")


if __name__ == "__main__":
    main()
