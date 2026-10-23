"""Check the derivations of the appendix "How much noise to add" and draw its figures.

    uv run python code/appendix-derivations/apx_core_calibration.py
    uv run python code/appendix-derivations/apx_core_calibration.py --seed 1 --out /tmp/figures

Prints every number quoted in book-full/appendix-derivations/01-calibration.md (they match
exactly with the default seed), then saves the appendix's figures.

Conventions: P and Q are the output distributions of a mechanism on neighbouring datasets D and D',
with probabilities or densities p(y) and q(y). The privacy loss is L(y) = ln(p(y) / q(y)), positive
where the output favours D. Laplace noise is given by its scale b, Gaussian noise by its standard
deviation σ. For the Gaussian mechanism μ = Δ₂ / σ is the shift in units of the noise. Logarithms
are natural. Exact quantities come from closed forms, sums or numerical integration; simulation is
used only to check them.
"""

import argparse
import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from itertools import combinations, product
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
matplotlib.rcParams["svg.hashsalt"] = "apx_core_calibration"  # stable element ids, so reruns don't churn the SVGs
matplotlib.rcParams["svg.fonttype"] = "none"  # keep text as text: smaller, diffable files

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from matplotlib.ticker import NullFormatter
from numpy.typing import NDArray

# Figures are build output: _build/figures/<part>/, which the build copies into each book.
FIGURES = Path(__file__).resolve().parents[2] / "_build" / "figures" / Path(__file__).parent.name

FloatArray = NDArray[np.float64]


def normal_cdf(x: float) -> float:
    """Φ(x) = Pr[Z <= x] for Z ~ N(0, 1); erfc keeps full relative precision in the lower tail."""
    return 0.5 * math.erfc(-x / math.sqrt(2))


def normal_pdf(x: float) -> float:
    return math.exp(-x * x / 2) / math.sqrt(2 * math.pi)


def bisect_increasing(g: Callable[[float], float], target: float, lo: float, hi: float, steps: int = 200) -> float:
    """The x in [lo, hi] with g(x) = target, for g increasing; the midpoint is taken in log x."""
    for _ in range(steps):
        mid = math.sqrt(lo * hi)
        if g(mid) < target:
            lo = mid
        else:
            hi = mid
    return math.sqrt(lo * hi)


# --- 1. What DP promises: pointwise ratios and the exact δ ----------------------------------------

# Three outcomes, e^ε = 2: the worked example of the page.
SMALL_P: tuple[float, ...] = (0.5, 0.3, 0.2)
SMALL_Q: tuple[float, ...] = (0.2, 0.3, 0.5)
SMALL_EPS = math.log(2.0)


def largest_ratio(p: Sequence[float], q: Sequence[float]) -> float:
    """max_y p(y) / q(y): the smallest e^ε for which P is pointwise within e^ε of Q."""
    return max(pi / qi for pi, qi in zip(p, q, strict=True))


def excess_mass(p: Sequence[float], q: Sequence[float], eps: float) -> float:
    """Σ_y (p(y) - e^ε q(y))_+, the smallest δ for the ordered pair (P, Q)."""
    return sum(max(pi - math.exp(eps) * qi, 0.0) for pi, qi in zip(p, q, strict=True))


def brute_force_delta(p: Sequence[float], q: Sequence[float], eps: float) -> float:
    """max over every event S of P(S) - e^ε Q(S), by listing all 2^m events."""
    m = len(p)
    best = 0.0
    for size in range(1, m + 1):
        for event in combinations(range(m), size):
            best = max(best, sum(p[i] for i in event) - math.exp(eps) * sum(q[i] for i in event))
    return best


def brute_force_ratio(p: Sequence[float], q: Sequence[float]) -> float:
    """max over every event S with Q(S) > 0 of P(S) / Q(S)."""
    m = len(p)
    best = 0.0
    for size in range(1, m + 1):
        for event in combinations(range(m), size):
            best = max(best, sum(p[i] for i in event) / sum(q[i] for i in event))
    return best


# --- 2. Sensitivity by brute force over small datasets -------------------------------------------

Record = tuple[int, ...]  # one record: (value,) for the scalar queries, or a tuple of yes/no answers
Dataset = tuple[Record, ...]
Query = Callable[[Dataset], list[float]]

VALUES = range(4)  # record values 0..3, so B = 3 and a histogram has k = 4 bins
CLIP = 3.0
ANSWERS = 3  # d = 3 yes/no questions per record, for the vector of yes counts
VALUE_RECORDS: list[Record] = [(v,) for v in VALUES]
ANSWER_RECORDS: list[Record] = list(product((0, 1), repeat=ANSWERS))


def all_datasets(records: Sequence[Record], max_size: int) -> list[Dataset]:
    """Every multiset of 1..max_size records, as sorted tuples."""
    out: set[Dataset] = set()
    for n in range(1, max_size + 1):
        for combo in product(records, repeat=n):
            out.add(tuple(sorted(combo)))
    return sorted(out)


def add_remove_neighbours(d: Dataset, records: Sequence[Record]) -> list[Dataset]:
    """D with one record removed (if something is left) or one record added."""
    out = [tuple(sorted((*d, r))) for r in records]
    if len(d) > 1:
        out += [d[:i] + d[i + 1 :] for i in range(len(d))]
    return out


def replace_neighbours(d: Dataset, records: Sequence[Record]) -> list[Dataset]:
    """D with one record swapped for any other record."""
    return [tuple(sorted((*d[:i], r, *d[i + 1 :]))) for i in range(len(d)) for r in records]


def q_count(d: Dataset) -> list[float]:
    return [float(len(d))]


def q_sum(d: Dataset) -> list[float]:
    return [float(sum(min(float(r[0]), CLIP) for r in d))]


def q_mean(d: Dataset) -> list[float]:
    return [sum(min(float(r[0]), CLIP) for r in d) / len(d)]


def q_histogram(d: Dataset) -> list[float]:
    return [float(sum(1 for r in d if r[0] == v)) for v in VALUES]


def q_yes_counts(d: Dataset) -> list[float]:
    return [float(sum(r[j] for r in d)) for j in range(ANSWERS)]


Neighbours = Callable[[Dataset, Sequence[Record]], list[Dataset]]


def sensitivity(
    query: Query, records: Sequence[Record], max_size: int, neighbours: Neighbours, norm: int, min_size: int = 1
) -> float:
    """max over neighbouring pairs of datasets of min_size..max_size records of the ℓ1 or ℓ2 distance of answers."""
    best = 0.0
    for d in all_datasets(records, max_size):
        if len(d) < min_size:
            continue
        a = query(d)
        for e in neighbours(d, records):
            if len(e) > max_size:
                continue
            diff = [x - y for x, y in zip(a, query(e), strict=True)]
            size = sum(abs(x) for x in diff) if norm == 1 else math.sqrt(sum(x * x for x in diff))
            best = max(best, size)
    return best


# --- 3. Laplace -----------------------------------------------------------------------------------

COUNT_D, COUNT_D2 = 1234.0, 1235.0  # the count on D and on D', which has one more person
EPS = 0.5
DELTA = 1e-5


def laplace_scale(sens1: float, eps: float) -> float:
    return sens1 / eps


def laplace_pdf(y: float, centre: float, b: float) -> float:
    return math.exp(-abs(y - centre) / b) / (2 * b)


def laplace_cdf(y: float, centre: float, b: float) -> float:
    z = (y - centre) / b
    return 0.5 * math.exp(z) if z < 0 else 1 - 0.5 * math.exp(-z)


def laplace_loss(y: float, a: float, a2: float, b: float) -> float:
    """ln p_D(y) / p_D'(y) for Laplace noise around a (on D) and a2 (on D')."""
    return (abs(y - a2) - abs(y - a)) / b


def laplace_vector_loss(y: FloatArray, a: FloatArray, a2: FloatArray, b: float) -> float:
    """Privacy loss of independent Laplace noise on each coordinate: the per-coordinate losses add."""
    return float(np.sum(np.abs(y - a2) - np.abs(y - a)) / b)


def laplace_error_bound(b: float, beta: float) -> float:
    """t with Pr[|Z| > t] = β for Z ~ Lap(b): t = b ln(1/β)."""
    return b * math.log(1 / beta)


def laplace_abs_tail(t: float, b: float) -> float:
    return math.exp(-t / b)


def simulated_laplace_errors(rng: np.random.Generator, b: float, draws: int) -> FloatArray:
    return np.abs(rng.laplace(0.0, b, size=draws))


# --- 4. Randomized response -----------------------------------------------------------------------


def rr_truth_prob(eps: float) -> float:
    """t = e^ε / (1 + e^ε), the probability of reporting the true bit."""
    return math.exp(eps) / (1 + math.exp(eps))


def rr_estimate(mean_report: float, t: float) -> float:
    """The unbiased estimate (ȳ - (1 - t)) / (2t - 1) of the fraction of ones."""
    return (mean_report - (1 - t)) / (2 * t - 1)


def rr_sd(eps: float, n: int) -> float:
    t = rr_truth_prob(eps)
    return math.sqrt(t * (1 - t)) / ((2 * t - 1) * math.sqrt(n))


def simulated_rr_estimates(rng: np.random.Generator, eps: float, n: int, pi: float, surveys: int) -> FloatArray:
    t = rr_truth_prob(eps)
    ones = round(pi * n)
    # People with bit 1 report 1 with probability t; people with bit 0 report 1 with probability 1 - t.
    yes = rng.binomial(ones, t, size=surveys) + rng.binomial(n - ones, 1 - t, size=surveys)
    return (yes / n - (1 - t)) / (2 * t - 1)


# --- 5. Exponential mechanism ---------------------------------------------------------------------

VOTES_D: tuple[float, ...] = (3.0, 2.0, 1.0, 0.0)  # options A, B, C, D
VOTES_D2: tuple[float, ...] = (3.0, 3.0, 1.0, 0.0)  # one more vote for B
EM_EPS = 1.0


def em_weights(scores: Sequence[float], eps: float, sens: float) -> list[float]:
    return [math.exp(eps * u / (2 * sens)) for u in scores]


def em_probs(scores: Sequence[float], eps: float, sens: float) -> list[float]:
    w = em_weights(scores, eps, sens)
    z = sum(w)
    return [x / z for x in w]


def em_gap_tail(scores: Sequence[float], eps: float, sens: float, c: float) -> float:
    """Exact Pr[the chosen score is at least c below the best]."""
    best = max(scores)
    return sum(p for p, u in zip(em_probs(scores, eps, sens), scores, strict=True) if u <= best - c)


def em_gap_bound(k: int, eps: float, sens: float, c: float) -> float:
    """k e^{-εc / (2Δ)}, the bound of the derivation before solving for c."""
    return k * math.exp(-eps * c / (2 * sens))


def em_gap_for(k: int, eps: float, sens: float, t: float) -> float:
    """(2Δ/ε)(ln k + t), the gap exceeded with probability at most e^{-t}."""
    return 2 * sens / eps * (math.log(k) + t)


MODEL_SCORES: tuple[float, ...] = tuple(900.0 - r for r in range(50))


# --- 6. Gaussian, classic calibration ---------------------------------------------------------------


def classic_sigma(sens2: float, eps: float, delta: float) -> float:
    return sens2 * math.sqrt(2 * math.log(1.25 / delta)) / eps


def gauss_loss_tail(eps: float, mu: float) -> float:
    """Pr[L > ε] when L ~ N(μ²/2, μ²): the bad-event probability."""
    return normal_cdf(mu / 2 - eps / mu)


def mills_bound(t: float) -> float:
    """φ(t)/t, an upper bound on Pr[Z > t] for t > 0."""
    return normal_pdf(t) / t


def classic_final_bound(eps: float, delta: float) -> float:
    """δ e^{ε/2} / (1.25 √(2π) t), the bound after expanding t² in the classic proof."""
    c = math.sqrt(2 * math.log(1.25 / delta))
    t = c - eps / (2 * c)
    return delta * math.exp(eps / 2) / (1.25 * math.sqrt(2 * math.pi) * t)


@dataclass(frozen=True)
class ClassicChain:
    c: float
    sigma: float
    t: float
    exact: float
    tail: float
    mills: float
    final: float


def classic_chain(eps: float, delta: float, sens2: float = 1.0) -> ClassicChain:
    c = math.sqrt(2 * math.log(1.25 / delta))
    sigma = classic_sigma(sens2, eps, delta)
    t = c - eps / (2 * c)
    mu = sens2 / sigma
    return ClassicChain(
        c=c,
        sigma=sigma,
        t=t,
        exact=gauss_delta(eps, mu),
        tail=gauss_loss_tail(eps, mu),
        mills=mills_bound(t),
        final=classic_final_bound(eps, delta),
    )


def classic_t_needed() -> float:
    """e^{1/2} / (1.25 √(2π)): t must be at least this for the final bound to be at most δ when ε <= 1."""
    return math.exp(0.5) / (1.25 * math.sqrt(2 * math.pi))


def classic_root_c() -> float:
    """The positive root of c² - k c - 1/2 = 0, where c - 1/(2c) reaches the needed t."""
    k = classic_t_needed()
    return (k + math.sqrt(k * k + 2)) / 2


def classic_largest_delta() -> float:
    """The largest δ for which c - 1/(2c) >= the needed t, with c = √(2 ln(1.25/δ))."""
    k = classic_t_needed()
    c = (k + math.sqrt(k * k + 2)) / 2  # positive root of c² - k c - 1/2 = 0
    return 1.25 * math.exp(-c * c / 2)


# --- 7. Gaussian, exact calibration -----------------------------------------------------------------


def gauss_delta(eps: float, mu: float) -> float:
    """δ(ε) = Φ(μ/2 - ε/μ) - e^ε Φ(-μ/2 - ε/μ), the exact δ of the Gaussian mechanism."""
    return normal_cdf(mu / 2 - eps / mu) - math.exp(eps) * normal_cdf(-mu / 2 - eps / mu)


def gauss_delta_integrated(eps: float, mu: float, step: float = 1e-3, half_width: float = 40.0) -> float:
    """∫ (p(y) - e^ε q(y))_+ dy for p = N(0, 1), q = N(μ, 1), by Simpson's rule."""
    n = int(2 * half_width / step)
    n += n % 2
    y = np.linspace(-half_width, half_width, n + 1)
    p = np.exp(-(y**2) / 2) / math.sqrt(2 * math.pi)
    q = np.exp(-((y - mu) ** 2) / 2) / math.sqrt(2 * math.pi)
    f = np.maximum(p - math.exp(eps) * q, 0.0)
    h = (y[-1] - y[0]) / n
    return float(h / 3 * (f[0] + f[-1] + 4 * f[1:-1:2].sum() + 2 * f[2:-1:2].sum()))


def gauss_delta_monte_carlo(rng: np.random.Generator, eps: float, mu: float, draws: int) -> float:
    """Average of (1 - e^{ε - L})_+ over L ~ N(μ²/2, μ²), the loss under P."""
    loss = rng.normal(mu * mu / 2, mu, size=draws)
    return float(np.mean(np.maximum(1 - np.exp(eps - loss), 0.0)))


def gauss_delta_derivative(eps: float, mu: float) -> float:
    """dδ/dμ = φ(μ/2 - ε/μ)."""
    return normal_pdf(mu / 2 - eps / mu)


def exact_sigma(sens2: float, eps: float, delta: float) -> float:
    """The smallest σ with δ(ε; μ = Δ₂/σ) <= δ, by bisection; δ falls as σ grows."""
    mu = bisect_increasing(lambda m: gauss_delta(eps, m), delta, math.exp(-12), math.exp(8))
    return sens2 / mu


def classic_breaks_at(delta: float) -> float:
    """The ε above which the classic σ gives a true δ larger than the target."""
    return bisect_increasing(lambda e: gauss_delta(e, e / math.sqrt(2 * math.log(1.25 / delta))), delta, 1.0, 50.0)


CALIBRATION_EPS: tuple[float, ...] = (0.1, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0)

# Visible example for the figure of the bad region: μ = 1, ε = 1.
FIG_MU, FIG_EPS = 1.0, 1.0


# --- 8. How ε, δ, σ and the sensitivity trade off ----------------------------------------------------


def normal_upper_tail(x: float) -> float:
    """Pr[Z > x] = ½ erfc(x/√2), computed without subtracting from 1."""
    return 0.5 * math.erfc(x / math.sqrt(2))


def upper_tail_by_subtraction(x: float) -> float:
    """1 - Φ(x), the naive way: loses every digit once Φ(x) rounds to 1."""
    return 1 - normal_cdf(x)


def normal_cdf_by_erf(x: float) -> float:
    """Φ(x) = ½(1 + erf(x/√2)), the textbook form: the same cancellation in the lower tail."""
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def gauss_delta_by_erf(eps: float, mu: float) -> float:
    """The exact δ(ε) with Φ computed from erf, to show the loss of precision for tiny δ."""
    return normal_cdf_by_erf(mu / 2 - eps / mu) - math.exp(eps) * normal_cdf_by_erf(-mu / 2 - eps / mu)


def gauss_delta_bad_set(eps: float, mu: float, width: float = 15.0, steps: int = 200_000) -> float:
    """∫ over the bad set z < μ/2 - ε/μ of φ(z)(1 - e^{ε - L(z)}) dz, by Simpson's rule.

    L(z) = μ²/2 - μz is the privacy loss at noise z (in units of σ), so the integrand is
    φ(z) - e^ε φ(z - μ) written without subtracting two tails; expm1 keeps it accurate near the edge.
    """
    edge = mu / 2 - eps / mu
    z = np.linspace(edge - width, edge, steps + 1)
    f = np.exp(-(z**2) / 2) / math.sqrt(2 * math.pi) * -np.expm1(eps + mu * z - mu * mu / 2)
    h = width / steps
    return float(h / 3 * (f[0] + f[-1] + 4 * f[1:-1:2].sum() + 2 * f[2:-1:2].sum()))


@dataclass(frozen=True)
class DeltaParts:
    """Each intermediate number of the exact δ(ε) for one Δ₂, σ and ε."""

    mu: float
    alpha: float
    beta: float
    phi_alpha: float
    phi_beta: float
    correction: float
    delta: float


def delta_parts(sens2: float, sigma: float, eps: float) -> DeltaParts:
    mu = sens2 / sigma
    alpha, beta = mu / 2 - eps / mu, -mu / 2 - eps / mu
    pa, pb = normal_cdf(alpha), normal_cdf(beta)
    return DeltaParts(mu, alpha, beta, pa, pb, math.exp(eps) * pb, pa - math.exp(eps) * pb)


def gauss_delta_eps_derivative(eps: float, mu: float) -> float:
    """dδ/dε = -e^ε Φ(-μ/2 - ε/μ), negative: the curve δ(ε) falls."""
    return -math.exp(eps) * normal_cdf(-mu / 2 - eps / mu)


def exact_eps(sens2: float, sigma: float, delta: float) -> float:
    """The smallest ε with δ(ε; μ = Δ₂/σ) <= δ, by bisection on ε (δ falls as ε grows)."""
    mu = sens2 / sigma
    return bisect_increasing(lambda e: -gauss_delta(e, mu), -delta, 1e-9, 500.0)


def normal_upper_quantile(delta: float) -> float:
    """z_δ with Pr[Z > z_δ] = δ, by bisection."""
    return bisect_increasing(lambda z: -normal_upper_tail(z), -delta, 1e-3, 40.0)


def tail_sigma(sens2: float, eps: float, delta: float) -> float:
    """The σ that makes the bad-event probability Pr[L > ε] equal δ: Δ₂(z + √(z² + 2ε)) / (2ε)."""
    z = normal_upper_quantile(delta)
    return sens2 * (z + math.sqrt(z * z + 2 * eps)) / (2 * eps)


def tail_eps(sens2: float, sigma: float, delta: float) -> float:
    """The ε at which Pr[L > ε] = δ for a given σ: z μ + μ²/2 with μ = Δ₂/σ."""
    mu = sens2 / sigma
    return normal_upper_quantile(delta) * mu + mu * mu / 2


def composed_sigma(sens2: float, eps: float, delta: float, k: int) -> float:
    """σ for k Gaussian releases that together are (ε, δ)-DP: √k times the σ of one."""
    return math.sqrt(k) * exact_sigma(sens2, eps, delta)


def split_sigma(sens2: float, eps: float, delta: float, k: int) -> float:
    """σ when the budget is split evenly, each release (ε/k, δ/k)-DP, and the k guarantees added up."""
    return exact_sigma(sens2, eps / k, delta / k)


def exposure(delta: float, n: int) -> float:
    """1 - (1 - δ)^n: the chance that publishing each record with probability δ exposes someone."""
    return -math.expm1(n * math.log1p(-delta))


TRADE_SIGMA = 4.0  # the worked example: Δ₂ = 1, σ = 4, ε = 1
CURVE_MUS: tuple[float, ...] = (0.25, 0.5, 1.0, 2.0)
GRID_EPS: tuple[float, ...] = (0.1, 0.5, 1.0, 2.0, 4.0)
GRID_DELTAS: tuple[float, ...] = (1e-3, 1e-5, 1e-7, 1e-9)
PRECISION_DELTAS: tuple[float, ...] = (1e-9, 1e-12, 1e-15, 1e-18)
INVERSE_SIGMAS: tuple[float, ...] = (0.5, 1.0, 2.0, 5.0, 10.0, 20.0)
DATASET_SIZES: tuple[int, ...] = (1_000, 10_000, 100_000, 1_000_000)


# --- Table ---------------------------------------------------------------------------------------


def print_table(rng: np.random.Generator) -> None:
    # 1. Exact δ on three outcomes.
    p, q = SMALL_P, SMALL_Q
    print(
        f"[1 three outcomes] ratios={[round(a / b, 4) for a, b in zip(p, q, strict=True)]}  "
        f"largest={largest_ratio(p, q):.4f}  pure eps=ln {largest_ratio(p, q):.1f}={math.log(largest_ratio(p, q)):.4f}  "
        f"largest event ratio={brute_force_ratio(p, q):.4f}"
    )
    print(
        f"[1 three outcomes, e^eps=2] excess sum={excess_mass(p, q, SMALL_EPS):.4f}  "
        f"brute force over 8 events={brute_force_delta(p, q, SMALL_EPS):.4f}  "
        f"Pr_P[L > eps]={sum(a for a, b in zip(p, q, strict=True) if a > 2 * b):.4f}"
    )

    # 2. Sensitivities.
    print(
        f"[2 sensitivity] all datasets of 1 to 3 records; values in 0..3 (B=3, k=4 bins), or {ANSWERS} yes/no answers"
    )
    rows: list[tuple[str, Query, list[Record]]] = [
        ("count", q_count, VALUE_RECORDS),
        ("sum", q_sum, VALUE_RECORDS),
        ("mean", q_mean, VALUE_RECORDS),
        ("histogram", q_histogram, VALUE_RECORDS),
        ("yes counts", q_yes_counts, ANSWER_RECORDS),
    ]
    relations: tuple[Neighbours, Neighbours] = (add_remove_neighbours, replace_neighbours)
    for name, query, records in rows:
        vals = [sensitivity(query, records, 3, nb, norm) for nb in relations for norm in (1, 2)]
        print(f"[2 {name}] add/remove l1={vals[0]:.4f} l2={vals[1]:.4f}  replace l1={vals[2]:.4f} l2={vals[3]:.4f}")
    fixed = sensitivity(q_mean, VALUE_RECORDS, 3, replace_neighbours, 1, min_size=3)
    print(f"[2 mean, n=3 records] replace l1={fixed:.4f} (B/n={CLIP / 3:.4f})")

    # 3. Laplace.
    b = laplace_scale(1.0, EPS)
    print(f"[3 Laplace count] eps={EPS} b={b:g}  bound e^eps={math.exp(EPS):.4f}")
    for y in (1230.0, 1234.0, 1234.5, 1235.0, 1240.0):
        pd, pd2 = laplace_pdf(y, COUNT_D, b), laplace_pdf(y, COUNT_D2, b)
        print(
            f"[3 Laplace y={y:g}] p_D={pd:.4f} p_D'={pd2:.4f} ratio={pd / pd2:.4f} "
            f"L={laplace_loss(y, COUNT_D, COUNT_D2, b):.4f}"
        )
    grid = np.linspace(1200.0, 1270.0, 70_001)
    losses = [laplace_loss(float(y), COUNT_D, COUNT_D2, b) for y in grid]
    print(f"[3 Laplace grid] loss ranges over [{min(losses):.4f}, {max(losses):.4f}]")
    event_d = laplace_cdf(COUNT_D, COUNT_D, b)
    event_d2 = laplace_cdf(COUNT_D, COUNT_D2, b)
    print(f"[3 Laplace event y<=1234] Pr_D={event_d:.4f} Pr_D'={event_d2:.4f} ratio={event_d / event_d2:.4f}")
    a = np.array([5.0, 2.0, 7.0])
    a2 = np.array([6.0, 3.0, 8.0])  # one person answered yes to all three questions
    bv = laplace_scale(3.0, EPS)
    ys = rng.normal(5.0, 10.0, size=(100_000, 3))
    worst = max(laplace_vector_loss(y, a, a2, bv) for y in ys)
    print(f"[3 Laplace vector d=3] b={bv:g}  largest loss over 100000 outputs={worst:.4f}  (eps={EPS})")
    t95 = laplace_error_bound(b, 0.05)
    errs = simulated_laplace_errors(rng, b, 1_000_000)
    print(
        f"[3 Laplace accuracy b={b:g}] mean |Z|={float(np.mean(errs)):.4f} (exact {b:g})  "
        f"95% bound b ln 20={t95:.4f}  simulated 95th percentile={float(np.quantile(errs, 0.95)):.4f}  "
        f"Pr[|Z| > {t95:.4f}]={float(np.mean(errs > t95)):.4f}"
    )
    for eps in (0.1, 1.0):
        print(f"[3 Laplace 95% error, count] eps={eps:g}: {laplace_error_bound(1 / eps, 0.05):.4f}")

    # 4. Randomized response.
    t = rr_truth_prob(EPS)
    print(f"[4 RR eps={EPS}] t={t:.4f}  1-t={1 - t:.4f}  ratio={t / (1 - t):.4f}  2t-1={2 * t - 1:.4f}")
    n, pi = 1000, 0.3
    mean_y = pi * t + (1 - pi) * (1 - t)
    est = simulated_rr_estimates(rng, EPS, n, pi, 10_000)
    print(
        f"[4 RR n={n} pi={pi}] E[ybar]={mean_y:.4f}  estimate={rr_estimate(mean_y, t):.4f}  "
        f"sd formula={rr_sd(EPS, n):.4f}  simulated mean={float(np.mean(est)):.4f} sd={float(np.std(est)):.4f}"
    )

    # 5. Exponential mechanism.
    wd = em_weights(VOTES_D, EM_EPS, 1.0)
    wd2 = em_weights(VOTES_D2, EM_EPS, 1.0)
    pd_ = em_probs(VOTES_D, EM_EPS, 1.0)
    pd2_ = em_probs(VOTES_D2, EM_EPS, 1.0)
    print(f"[5 EM vote] weights D={[round(w, 4) for w in wd]}  Z(D)={sum(wd):.4f}  Z(D')={sum(wd2):.4f}")
    print(f"[5 EM vote] P_D={[round(x, 4) for x in pd_]}  P_D'={[round(x, 4) for x in pd2_]}")
    ratios = [x / y for x, y in zip(pd_, pd2_, strict=True)]
    numer = [math.exp(EM_EPS * (u - v) / 2) for u, v in zip(VOTES_D, VOTES_D2, strict=True)]
    print(
        f"[5 EM vote] ratios={[round(r, 4) for r in ratios]}  numerators={[round(x, 4) for x in numer]}  "
        f"normalizer ratio={sum(wd2) / sum(wd):.4f}  bound e^(eps/2)={math.exp(EM_EPS / 2):.4f}  "
        f"largest |loss|={max(abs(math.log(r)) for r in ratios):.4f}"
    )
    for c in (2.0, 3.0):
        print(
            f"[5 EM vote gap>={c:g}] exact={em_gap_tail(VOTES_D, EM_EPS, 1.0, c):.4f}  "
            f"bound k e^(-eps c/2)={em_gap_bound(4, EM_EPS, 1.0, c):.4f}"
        )
    k = len(MODEL_SCORES)
    gap = em_gap_for(k, 1.0, 1.0, 3.0)
    print(
        f"[5 EM 50 models eps=1 t=3] gap={gap:.4f}  bound e^-3={math.exp(-3):.4f}  "
        f"exact={em_gap_tail(MODEL_SCORES, 1.0, 1.0, gap):.2e}  "
        f"Pr[best]={em_probs(MODEL_SCORES, 1.0, 1.0)[0]:.4f}"
    )

    # 6. Classic Gaussian.
    for eps in (EPS, 1.0):
        ch = classic_chain(eps, DELTA)
        print(
            f"[6 classic eps={eps:g} delta=1e-5] c={ch.c:.4f} sigma={ch.sigma:.4f} t={ch.t:.4f}  "
            f"exact={ch.exact:.2e}  Pr[L>eps]={ch.tail:.2e}  Mills={ch.mills:.2e}  final={ch.final:.2e}"
        )
    ch = classic_chain(EPS, DELTA)
    print(
        f"[6 classic slack eps={EPS}] target/exact={DELTA / ch.exact:.0f}  target/final={DELTA / ch.final:.1f}  "
        f"tail/exact={ch.tail / ch.exact:.1f}  1.96 sigma={1.96 * ch.sigma:.4f}"
    )
    for eps in (0.05, 1.0):
        ch = classic_chain(eps, DELTA)
        print(f"[6 classic tail/exact eps={eps:g}] {ch.tail / ch.exact:.1f}")
    print(
        f"[6 classic constants] needed t={classic_t_needed():.4f}  root c={classic_root_c():.4f}  holds for delta<={classic_largest_delta():.4f}  "
        f"ln(1.25/1e-5)={math.log(1.25 / DELTA):.4f}"
    )

    # 7. Exact Gaussian.
    mu = FIG_MU
    print(
        f"[7 exact mu={mu:g} eps={FIG_EPS:g}] P(B)={normal_cdf(mu / 2 - FIG_EPS / mu):.4f}  "
        f"e^eps Q(B)={math.exp(FIG_EPS) * normal_cdf(-mu / 2 - FIG_EPS / mu):.4f}  "
        f"delta={gauss_delta(FIG_EPS, mu):.4f}  integrated={gauss_delta_integrated(FIG_EPS, mu):.4f}  "
        f"Monte Carlo={gauss_delta_monte_carlo(rng, FIG_EPS, mu, 1_000_000):.4f}"
    )
    worst_rel = 0.0
    pairs = 0
    for eps in (0.0, 0.5, 1.0, 2.0, 4.0):
        for m in (0.25, 0.5, 1.0, 2.0, 4.0):
            exact = gauss_delta(eps, m)
            if exact > 1e-12:
                pairs += 1
                worst_rel = max(worst_rel, abs(gauss_delta_integrated(eps, m) - exact) / exact)
    print(f"[7 closed form vs Simpson] {pairs} pairs, largest relative difference={worst_rel:.1e}")
    worst_der = 0.0
    for eps in (0.1, 0.5, 1.0, 3.0):
        for m in (0.1, 0.5, 1.0, 2.0):
            h = 1e-6
            numeric = (gauss_delta(eps, m + h) - gauss_delta(eps, m - h)) / (2 * h)
            worst_der = max(worst_der, abs(numeric - gauss_delta_derivative(eps, m)))
    print(f"[7 derivative] d delta/d mu vs phi(mu/2 - eps/mu): largest difference below 1e-9: {worst_der < 1e-9}")
    s_exact = exact_sigma(1.0, EPS, DELTA)
    s_classic = classic_sigma(1.0, EPS, DELTA)
    print(
        f"[7 calibrate eps={EPS} delta=1e-5] exact sigma={s_exact:.4f}  delta there={gauss_delta(EPS, 1 / s_exact):.3e}  "
        f"classic sigma={s_classic:.4f}  ratio={s_classic / s_exact:.3f}  "
        f"true delta at classic={gauss_delta(EPS, 1 / s_classic):.2e}"
    )
    for eps in CALIBRATION_EPS:
        se, sc = exact_sigma(1.0, eps, DELTA), classic_sigma(1.0, eps, DELTA)
        print(
            f"[7 table eps={eps:g}] classic={sc:.4f}  exact={se:.4f}  ratio={sc / se:.3f}  "
            f"true delta at classic={gauss_delta(eps, 1 / sc):.2e}"
        )
    print(f"[7 ratio eps=0.05] classic/exact={classic_sigma(1.0, 0.05, DELTA) / exact_sigma(1.0, 0.05, DELTA):.3f}")
    print(f"[7 classic breaks] true delta exceeds 1e-5 for eps > {classic_breaks_at(DELTA):.2f}")
    print(f"[7 95% error] exact sigma at eps={EPS}: 1.96 sigma={1.96 * s_exact:.4f}")

    # 8. Trade-offs.
    pt = delta_parts(1.0, TRADE_SIGMA, 1.0)
    print(
        f"[8 by hand D2=1 sigma={TRADE_SIGMA:g} eps=1] mu={pt.mu:.4f}  alpha={pt.alpha:.4f}  beta={pt.beta:.4f}  "
        f"Phi(alpha)={pt.phi_alpha:.4e}  Phi(beta)={pt.phi_beta:.4e}  e={math.e:.4f}  "
        f"e Phi(beta)={pt.correction:.4e}  delta={pt.delta:.4e}  first/delta={pt.phi_alpha / pt.delta:.1f}"
    )
    print(
        f"[8 by hand checks] bad-set integral={gauss_delta_bad_set(1.0, pt.mu):.4e}  "
        f"D2=2 sigma=8: delta={delta_parts(2.0, 8.0, 1.0).delta:.4e}  "
        f"D2=0.5 sigma=2: delta={delta_parts(0.5, 2.0, 1.0).delta:.4e}"
    )
    for x in (8.0, 9.0):
        print(
            f"[8 tail x={x:g}] 1 - Phi(x)={upper_tail_by_subtraction(x):.4e}  "
            f"erf form Phi(-x)={normal_cdf_by_erf(-x):.4e}  erfc form={normal_upper_tail(x):.4e}"
        )
    for d in PRECISION_DELTAS:
        s = exact_sigma(1.0, 1.0, d)
        m = 1 / s
        closed = gauss_delta(1.0, m)
        integral = gauss_delta_bad_set(1.0, m)
        print(
            f"[8 precision eps=1 delta={d:.0e}] sigma={s:.4f}  erfc={closed:.4e}  erf={gauss_delta_by_erf(1.0, m):.4e}  "
            f"bad-set integral={integral:.4e}  rel diff={abs(integral - closed) / closed:.0e}  "
            f"first/delta={normal_cdf(m / 2 - 1.0 / m) / closed:.1f}"
        )
    for m in CURVE_MUS:
        print(
            f"[8 curve mu={m:g}] sigma/D2={1 / m:g}  eps at 1e-5={exact_eps(1.0, 1 / m, DELTA):.4f}  "
            f"delta(0)={gauss_delta(0.0, m):.4f}  delta(1)={gauss_delta(1.0, m):.2e}  delta(2)={gauss_delta(2.0, m):.2e}"
        )
    worst = 0.0
    for e in (0.1, 0.5, 1.0, 3.0):
        for m in (0.1, 0.5, 1.0, 2.0):
            h = 1e-6
            numeric = (gauss_delta(e + h, m) - gauss_delta(e - h, m)) / (2 * h)
            worst = max(worst, abs(numeric - gauss_delta_eps_derivative(e, m)))
    print(f"[8 d delta/d eps] matches -e^eps Phi(beta) to below 1e-9: {worst < 1e-9}")
    for e in GRID_EPS:
        print(f"[8 grid eps={e:g}] " + "  ".join(f"{d:.0e}: {exact_sigma(1.0, e, d):.4f}" for d in GRID_DELTAS))
    z = normal_upper_quantile(DELTA)
    print(f"[8 tail rule delta=1e-5] z={z:.4f}  bend at z^2/2={z * z / 2:.2f}")
    for e in (0.1, 0.5, 1.0, 2.0, 4.0, 8.0, 32.0):
        se, st = exact_sigma(1.0, e, DELTA), tail_sigma(1.0, e, DELTA)
        print(
            f"[8 sigma vs eps={e:g}] exact={se:.4f}  eps*sigma={e * se:.4f}  tail rule={st:.4f}  "
            f"tail/exact={st / se:.3f}  z/eps={z / e:.4f}  1/sqrt(2 eps)={1 / math.sqrt(2 * e):.4f}  "
            f"first/delta={normal_cdf(1 / (2 * se) - e * se) / gauss_delta(e, 1 / se):.1f}"
        )
    for e in (0.5, 1.0):
        s5, s6 = exact_sigma(1.0, e, 1e-5), exact_sigma(1.0, e, 1e-6)
        print(f"[8 per decade eps={e:g}] sigma 1e-5={s5:.4f}  1e-6={s6:.4f}  ratio={s6 / s5:.3f}  added={s6 - s5:.4f}")
    z6 = normal_upper_quantile(1e-6)
    print(f"[8 per decade tail rule] z(1e-6)={z6:.4f}  z ratio={z6 / z:.3f}")
    print(
        f"[8 1e-5 to 1e-9 eps=0.5] ratio={exact_sigma(1.0, 0.5, 1e-9) / exact_sigma(1.0, 0.5, 1e-5):.3f}  "
        f"halving eps 0.5 to 0.25: ratio={exact_sigma(1.0, 0.25, DELTA) / exact_sigma(1.0, 0.5, DELTA):.3f}"
    )
    for s in INVERSE_SIGMAS:
        print(f"[8 eps vs sigma={s:g}] exact={exact_eps(1.0, s, DELTA):.4f}  tail rule={tail_eps(1.0, s, DELTA):.4f}")
    one = exact_sigma(1.0, 1.0, DELTA)
    for k in (10, 100):
        c, sp = composed_sigma(1.0, 1.0, DELTA, k), split_sigma(1.0, 1.0, DELTA, k)
        print(
            f"[8 compose k={k} eps=1 delta=1e-5] one={one:.4f}  sqrt(k) sigma={c:.4f}  "
            f"split (eps/k, delta/k)={sp:.4f}  split/composed={sp / c:.2f}  "
            f"check delta(1; mu sqrt k)={gauss_delta(1.0, math.sqrt(k) / c):.3e}"
        )
    print(
        f"[8 delta choice] 1/n^2 vs 1e-5 at n=10000: ratio={exact_sigma(1.0, 1.0, 1e-8) / one:.3f}  "
        f"exposure n=1e6 delta=1e-5={exposure(DELTA, 1_000_000):.5f}  "
        f"eps*sigma at 1 over 0.1={exact_sigma(1.0, 1.0, DELTA) / (0.1 * exact_sigma(1.0, 0.1, DELTA)):.3f}"
    )
    for n in DATASET_SIZES:
        ds = (DELTA, 1 / n, n**-1.1, n**-2.0)
        print(
            f"[8 n={n}] sigma at eps=1: "
            + "  ".join(
                f"{name}={exact_sigma(1.0, 1.0, d):.4f}"
                for name, d in zip(("1e-5", "1/n", "1/n^1.1", "1/n^2"), ds, strict=True)
            )
            + "  exposure: "
            + "  ".join(
                f"{name}={exposure(d, n):.3g}" for name, d in zip(("1e-5", "1/n", "1/n^1.1", "1/n^2"), ds, strict=True)
            )
        )


# --- Figures --------------------------------------------------------------------------------------


def plot_excess() -> Figure:
    """The three-outcome example: p(y) against e^ε q(y), with the excess that makes δ."""
    fig, ax = plt.subplots(figsize=(6, 4))
    xs = np.arange(len(SMALL_P))
    scaled = [math.exp(SMALL_EPS) * x for x in SMALL_Q]
    ax.bar(xs - 0.2, SMALL_P, width=0.4, label="p(y), on D")
    ax.bar(xs + 0.2, scaled, width=0.4, color="tab:orange", alpha=0.6, label=r"$e^\varepsilon q(y)$, on D'")
    for x, top, bottom in zip(xs, SMALL_P, scaled, strict=True):
        if top > bottom:
            ax.bar(x - 0.2, top - bottom, bottom=bottom, width=0.4, color="tab:red", label="excess = δ")
    ax.set(
        xticks=xs,
        xticklabels=["y = 1", "y = 2", "y = 3"],
        ylabel="probability",
        title=r"Only outcomes with $p > e^\varepsilon q$ cost δ ($e^\varepsilon = 2$)",
    )
    ax.legend()
    return fig


def plot_laplace_ratio(b: float = 2.0) -> Figure:
    ys = np.linspace(1222.0, 1247.0, 1001)
    pd = [laplace_pdf(float(y), COUNT_D, b) for y in ys]
    pd2 = [laplace_pdf(float(y), COUNT_D2, b) for y in ys]
    ratio = [x / z for x, z in zip(pd, pd2, strict=True)]
    eps = 1.0 / b
    fig, (top, bottom) = plt.subplots(2, 1, figsize=(6, 6), sharex=True)
    top.plot(ys, pd, label="output density on D (count 1234)")
    top.plot(ys, pd2, label="output density on D' (count 1235)")
    top.set(ylabel="density", title=f"Laplace noise with b = {b:g}: the same shape, shifted by 1")
    top.legend(loc="upper right", fontsize=8)
    bottom.plot(ys, ratio, "k", label="ratio p_D(y) / p_D′(y)")
    bottom.axhline(math.exp(eps), color="tab:red", linestyle="--", label=r"$e^{\pm\varepsilon}$, $\varepsilon = 0.5$")
    bottom.axhline(math.exp(-eps), color="tab:red", linestyle="--")
    bottom.axvspan(COUNT_D, COUNT_D2, color="gray", alpha=0.15)
    bottom.set(
        yscale="log",
        ylim=(0.5, 2.0),
        yticks=[0.6, 1.0, 1.65],
        yticklabels=["0.61", "1", "1.65"],
        xlabel="released value y",
        ylabel="ratio (log scale)",
        title="The ratio never leaves the band",
    )
    bottom.yaxis.set_minor_formatter(NullFormatter())
    bottom.legend(loc="upper right", fontsize=8)
    return fig


def plot_privacy_loss(b: float = 2.0, sigma: float = 2.0) -> Figure:
    ys = np.linspace(1222.0, 1247.0, 1001)
    lap = [laplace_loss(float(y), COUNT_D, COUNT_D2, b) for y in ys]
    shift = COUNT_D2 - COUNT_D
    gauss = [(shift**2 - 2 * shift * (float(y) - COUNT_D)) / (2 * sigma**2) for y in ys]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(ys, lap, linewidth=2, label=f"Laplace, b = {b:g}")
    ax.plot(ys, gauss, linewidth=2, label=f"Gaussian, σ = {sigma:g}")
    ax.axhline(EPS, color="tab:red", linestyle="--", label=r"$\pm\varepsilon$, $\varepsilon = 0.5$")
    ax.axhline(-EPS, color="tab:red", linestyle="--")
    ax.axvspan(COUNT_D, COUNT_D2, color="gray", alpha=0.15)
    ax.set(
        ylim=(-3, 3),
        xlabel="released value y",
        ylabel="privacy loss L(y) = ln p_D(y) / p_D′(y)",
        title="Flat outside the two answers, or a straight line",
    )
    ax.legend(loc="upper right")
    return fig


def plot_exponential() -> Figure:
    pd = em_probs(VOTES_D, EM_EPS, 1.0)
    pd2 = em_probs(VOTES_D2, EM_EPS, 1.0)
    xs = np.arange(4)
    fig, (left, right) = plt.subplots(1, 2, figsize=(10, 4))
    left.bar(xs - 0.2, pd, width=0.4, label="on D, votes (3, 2, 1, 0)")
    left.bar(xs + 0.2, pd2, width=0.4, label="on D', votes (3, 3, 1, 0)")
    left.set(
        xticks=xs,
        xticklabels=list("ABCD"),
        ylabel="probability of being chosen",
        title=r"Exponential mechanism, $\varepsilon = 1$",
    )
    left.legend()
    numer = [math.exp(EM_EPS * (u - v) / 2) for u, v in zip(VOTES_D, VOTES_D2, strict=True)]
    norm = sum(em_weights(VOTES_D2, EM_EPS, 1.0)) / sum(em_weights(VOTES_D, EM_EPS, 1.0))
    right.bar(xs - 0.2, numer, width=0.4, label="own weight factor")
    right.bar(xs + 0.2, [norm] * 4, width=0.4, label="normalizer factor W(D')/W(D)")
    right.plot(xs, [n * norm for n in numer], "ko", label="ratio = product")
    right.axhline(math.exp(EM_EPS / 2), color="tab:red", linestyle=":", label=r"$e^{\varepsilon/2}$")
    right.axhline(math.exp(EM_EPS), color="tab:red", linestyle="--", label=r"$e^{\varepsilon}$")
    right.set(
        xticks=xs,
        xticklabels=list("ABCD"),
        ylim=(0, 3),
        ylabel="factor",
        title="Each factor at most $e^{\\varepsilon/2}$, so the ratio at most $e^\\varepsilon$",
    )
    right.legend(fontsize=8, loc="upper left")
    return fig


def plot_classic_chain() -> Figure:
    epss = np.linspace(0.05, 1.0, 96)
    chains = [classic_chain(float(e), DELTA) for e in epss]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.axhline(DELTA, color="k", linestyle="--", label=r"target $\delta = 10^{-5}$")
    ax.plot(epss, [c.final for c in chains], label=r"after expanding $t^2$ (step 5)")
    ax.plot(epss, [c.mills for c in chains], label=r"Mills bound $\varphi(t)/t$ (step 4)")
    ax.plot(epss, [c.tail for c in chains], label=r"bad event $\Pr[L > \varepsilon]$ (step 3)")
    ax.plot(epss, [c.exact for c in chains], linewidth=2, label=r"exact $\delta(\varepsilon)$")
    ax.set(
        yscale="log",
        ylim=(1e-9, 1e-4),
        xlabel=r"$\varepsilon$",
        ylabel=r"$\delta$ delivered by the classic $\sigma$",
        title="Each step of the classic proof gives away a little",
    )
    ax.legend(fontsize=8, loc="upper right")
    return fig


def plot_bad_region(mu: float = FIG_MU, eps: float = FIG_EPS) -> Figure:
    """Left: output densities in units of σ, with the bad region. Right: the privacy loss distributions."""
    z = np.linspace(-4.0, 5.0, 901)
    p = np.exp(-(z**2) / 2) / math.sqrt(2 * math.pi)
    q = np.exp(-((z - mu) ** 2) / 2) / math.sqrt(2 * math.pi)
    eq = math.exp(eps) * q
    edge = mu / 2 - eps / mu  # L > ε exactly when z < μ/2 - ε/μ
    bad = z < edge
    fig, (left, right) = plt.subplots(1, 2, figsize=(10, 4))
    left.plot(z, p, label="p(y): on D, centred at 0")
    left.plot(z, eq, label=r"$e^\varepsilon q(y)$: on D', centred at μ")
    left.fill_between(z, 0, np.minimum(p, eq), where=bad, color="tab:orange", alpha=0.3, label=r"$e^\varepsilon Q(B)$")
    left.fill_between(
        z, eq, p, where=bad & (p > eq), color="tab:red", alpha=0.6, label=f"δ = {gauss_delta(eps, mu):.4f}"
    )
    left.axvline(edge, color="gray", linestyle=":")
    left.set(xlabel="noise on D, in units of σ", ylabel="density", title=f"Bad region B = {{L > ε}}: y < {edge:g}σ")
    left.legend(fontsize=8)
    ls = np.linspace(-4.0, 4.0, 801)
    pl = np.exp(-((ls - mu * mu / 2) ** 2) / (2 * mu * mu)) / (mu * math.sqrt(2 * math.pi))
    ql = np.exp(-((ls + mu * mu / 2) ** 2) / (2 * mu * mu)) / (mu * math.sqrt(2 * math.pi))
    eql = math.exp(eps) * ql
    right.plot(ls, pl, label="loss on D: N(μ²/2, μ²)")
    right.plot(ls, eql, label=r"$e^\varepsilon$ × loss density on D'")
    right.fill_between(ls, eql, pl, where=ls > eps, color="tab:red", alpha=0.6, label="δ")
    right.axvline(eps, color="gray", linestyle=":", label=r"$L = \varepsilon$")
    right.set(
        xlabel="privacy loss L", ylabel="density", title=f"The same δ as a loss distribution, μ = {mu:g}, ε = {eps:g}"
    )
    right.legend(fontsize=8)
    return fig


def plot_sigma_vs_eps() -> Figure:
    epss = np.logspace(math.log10(0.05), math.log10(20.0), 120)
    exact = [exact_sigma(1.0, float(e), DELTA) for e in epss]
    classic = [classic_sigma(1.0, float(e), DELTA) for e in epss]
    breaks = classic_breaks_at(DELTA)
    fig, (left, right) = plt.subplots(1, 2, figsize=(10, 4))
    left.plot(epss, classic, label="classic formula")
    left.plot(epss, exact, label="exact, by bisection")
    left.axvspan(1.0, 20.0, color="gray", alpha=0.15, label=r"$\varepsilon \geq 1$: classic proof says nothing")
    left.set(
        xscale="log",
        yscale="log",
        xlabel=r"$\varepsilon$",
        ylabel=r"$\sigma / \Delta_2$",
        title=r"Noise for $(\varepsilon, 10^{-5})$-DP",
    )
    left.legend(fontsize=8)
    right.plot(epss, [c / e for c, e in zip(classic, exact, strict=True)], "k")
    right.axhline(1.0, color="tab:red", linestyle="--", label="ratio 1: classic is just enough")
    right.axvspan(1.0, 20.0, color="gray", alpha=0.15)
    right.axvline(breaks, color="tab:red", linestyle=":", label=f"classic too small for ε > {breaks:.2f}")
    right.set(
        xscale="log",
        xlabel=r"$\varepsilon$",
        ylabel="classic σ / exact σ",
        title="How much extra noise the classic formula adds",
    )
    right.legend(fontsize=8)
    return fig


def plot_delta_curves() -> Figure:
    """δ(ε) for several μ = Δ₂/σ: one σ gives a whole falling curve of (ε, δ) pairs."""
    epss = np.linspace(0.0, 12.0, 481)
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    for i, m in enumerate(CURVE_MUS):
        deltas = [gauss_delta(float(e), m) for e in epss]
        (line,) = ax.plot(epss, deltas, label=rf"$\mu$ = {m:g}  ($\sigma$ = {1 / m:g} $\Delta_2$)")
        e5 = exact_eps(1.0, 1 / m, DELTA)
        ax.plot([e5], [DELTA], "o", color=line.get_color())
        offset = (5, 6) if i % 2 == 0 else (5, -13)  # alternate above and below so labels don't collide
        ax.annotate(f"ε = {e5:.2f}", (e5, DELTA), textcoords="offset points", xytext=offset, fontsize=8)
    ax.axhline(DELTA, color="k", linestyle="--", linewidth=1, label=r"$\delta = 10^{-5}$")
    ax.set(
        yscale="log",
        ylim=(1e-12, 1.0),
        xlim=(0.0, 12.0),
        xlabel=r"$\varepsilon$",
        ylabel=r"exact $\delta(\varepsilon)$",
        title=r"One $\sigma$, a whole curve of $(\varepsilon, \delta)$ guarantees",
    )
    ax.legend(fontsize=8, loc="upper right")
    return fig


def plot_tradeoff_map() -> Figure:
    """Left: contours of the exact σ/Δ₂ over (ε, δ). Right: ε σ / Δ₂ against ε, exact and by the tail rule."""
    epss = np.logspace(math.log10(0.05), math.log10(20.0), 45)
    log_deltas = np.linspace(-12.0, -2.0, 41)
    sig = np.array([[exact_sigma(1.0, float(e), 10.0 ** float(ld)) for e in epss] for ld in log_deltas])
    fig, (left, right) = plt.subplots(1, 2, figsize=(11, 4.4))
    levels = [0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0]
    colors = [plt.get_cmap("viridis")(float(x)) for x in np.linspace(0.0, 0.8, len(levels))]
    cs = left.contour(epss, log_deltas, sig, levels=levels, colors=colors)
    left.clabel(cs, fmt="%g", fontsize=8)
    left.axhline(-5.0, color="gray", linestyle=":", linewidth=1)
    left.set(
        xscale="log",
        xlabel=r"$\varepsilon$",
        ylabel=r"$\log_{10}\delta$",
        title=r"Exact noise $\sigma/\Delta_2$: each line is one $\sigma$",
    )
    epsr = np.logspace(math.log10(0.05), math.log10(50.0), 80)
    for d, color in zip((1e-3, 1e-5, 1e-9), ("tab:blue", "tab:orange", "tab:green"), strict=True):
        exact = [float(e) * exact_sigma(1.0, float(e), d) for e in epsr]
        rule = [float(e) * tail_sigma(1.0, float(e), d) for e in epsr]
        right.plot(epsr, exact, color=color, label=rf"exact, $\delta = 10^{{{round(math.log10(d))}}}$")
        right.plot(epsr, rule, color=color, linestyle="--", linewidth=1)
    right.plot([], [], color="gray", linestyle="--", linewidth=1, label="dashed: tail rule, same δ")
    right.set(
        xscale="log",
        ylim=(0.0, 12.0),
        xlabel=r"$\varepsilon$",
        ylabel=r"$\varepsilon\sigma/\Delta_2$",
        title=r"Flat means $\sigma \propto 1/\varepsilon$; it bends up at large $\varepsilon$",
    )
    right.legend(fontsize=8, loc="upper left", ncol=1)
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
        "cal_excess.svg": plot_excess(),
        "cal_laplace_ratio.svg": plot_laplace_ratio(),
        "cal_privacy_loss.svg": plot_privacy_loss(),
        "cal_exponential.svg": plot_exponential(),
        "cal_classic_chain.svg": plot_classic_chain(),
        "cal_bad_region.svg": plot_bad_region(),
        "cal_sigma_vs_eps.svg": plot_sigma_vs_eps(),
        "cal_delta_curves.svg": plot_delta_curves(),
        "cal_tradeoff_map.svg": plot_tradeoff_map(),
    }
    for name, fig in figures.items():
        fig.tight_layout()
        fig.savefig(out / name, metadata={"Date": None})  # no timestamp, so output is reproducible
        plt.close(fig)
        print(f"saved {out / name}")


if __name__ == "__main__":
    main()
