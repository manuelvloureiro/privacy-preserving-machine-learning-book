"""Compare the central, local, shuffle and distributed trust models, and draw the chapter's figures.

    uv run python code/part03-dp-foundations/trust_models.py
    uv run python code/part03-dp-foundations/trust_models.py --seed 1 --out /tmp/figures

Prints the numbers quoted in 10-trust-models.md (they match exactly with the default seed),
then saves the chapter's figures.

Conventions: neighbouring datasets differ by replacing one record, in every model, because a
local randomizer must protect each user's value against every other value. ε0 is the local
budget of one report, (ε, δ) the central guarantee. Frequencies are fractions in [0, 1], and an
estimator's error is measured on fractions, so a count error is n times larger.
"""

import argparse
import math
import sys
from collections.abc import Callable, Sequence
from functools import cache
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # for code/shared
from shared.figures import configure
from shared.stats import normal_cdf

configure("trust_models")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.patches import FancyBboxPatch
from numpy.typing import NDArray

# Figures are build output: _build/figures/<part>/, which the build copies into each book.
FIGURES = Path(__file__).resolve().parents[2] / "_build" / "figures" / Path(__file__).parent.name

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]


# --- Shared helpers ---------------------------------------------------------------------------


def bisect(f: Callable[[float], float], lo: float, hi: float, steps: int = 60) -> float:
    """A root of f on [lo, hi], given that f(lo) and f(hi) have opposite signs."""
    f_lo = f(lo)
    for _ in range(steps):
        mid = 0.5 * (lo + hi)
        if (f(mid) > 0) == (f_lo > 0):
            lo, f_lo = mid, f(mid)
        else:
            hi = mid
    return 0.5 * (lo + hi)


@cache
def log_factorials(n: int) -> FloatArray:
    """ln k! for k = 0..n, from math.lgamma (NumPy has no log-gamma)."""
    return np.array([math.lgamma(k + 1) for k in range(n + 1)], dtype=np.float64)


def binomial_pmf(n: int, p: float) -> FloatArray:
    """Pr[Bin(n, p) = k] for k = 0..n, computed in log space so large n does not underflow."""
    k = np.arange(n + 1)
    lf = log_factorials(n)
    if p in (0.0, 1.0):
        out = np.zeros(n + 1)
        out[0 if p == 0.0 else n] = 1.0
        return out
    log_pmf = lf[n] - lf[k] - lf[n - k] + k * math.log(p) + (n - k) * math.log1p(-p)
    return np.exp(log_pmf)


def convolve(a: FloatArray, b: FloatArray) -> FloatArray:
    """The pmf of the sum of two independent variables, by FFT (np.convolve is quadratic)."""
    size = len(a) + len(b) - 1
    out = np.fft.irfft(np.fft.rfft(a, size) * np.fft.rfft(b, size), size)
    return np.maximum(out, 0.0)  # FFT round-off leaves tiny negative values


def hockey_stick(p: FloatArray, q: FloatArray, eps: float) -> float:
    """E_{e^ε}(P‖Q) = Σ (p − e^ε q)_+ for two pmfs on the same finite set."""
    return float(np.maximum(p - math.exp(eps) * q, 0.0).sum())


def epsilon_for_delta(delta_of_eps: Callable[[float], float], delta: float, hi: float) -> float:
    """The smallest ε in [0, hi] with δ(ε) ≤ delta, for a decreasing curve δ(ε)."""
    if delta_of_eps(0.0) <= delta:
        return 0.0
    if delta_of_eps(hi) > delta:
        return math.inf
    return bisect(lambda e: delta_of_eps(e) - delta, 0.0, hi, steps=50)


# --- Central against local: randomized response for one bit -----------------------------------


def rr_keep(eps: float) -> float:
    """Randomized response reports the true bit with probability e^ε/(1 + e^ε)."""
    return math.exp(eps) / (1 + math.exp(eps))


def rr_estimate(mean_report: float, eps: float) -> float:
    """Unbiased estimate of the fraction of ones from the mean of the reports: (Ȳ − (1−p))/(2p − 1)."""
    p = rr_keep(eps)
    return (mean_report - (1 - p)) / (2 * p - 1)


def rr_variance(n: int, eps: float) -> float:
    """Var of the unbiased estimate of the fraction: e^ε / (n (e^ε − 1)²), whatever the data."""
    return math.exp(eps) / (n * math.expm1(eps) ** 2)


def central_count_sd(n: int, eps: float, sensitivity: float = 1.0) -> float:
    """Standard deviation of the fraction under the central Laplace mechanism, Lap(Δ/ε) on the count."""
    return math.sqrt(2) * sensitivity / (eps * n)


def simulate_rr(rng: np.random.Generator, n: int, ones: int, eps: float, trials: int) -> FloatArray:
    """`trials` estimates of the fraction of ones; n users, `ones` of them hold a 1."""
    p = rr_keep(eps)
    reported_ones = rng.binomial(ones, p, size=trials) + rng.binomial(n - ones, 1 - p, size=trials)
    return (reported_ones / n - (1 - p)) / (2 * p - 1)


def bernoulli_kl(a: float, b: float) -> float:
    """KL(Bern(a)‖Bern(b))."""
    return a * math.log(a / b) + (1 - a) * math.log((1 - a) / (1 - b))


def rr_report_kl(f0: float, f1: float, eps: float) -> float:
    """KL between the distributions of one randomized-response report when the bit is Bern(f0) or Bern(f1)."""
    p = rr_keep(eps)
    return bernoulli_kl(f0 * p + (1 - f0) * (1 - p), f1 * p + (1 - f1) * (1 - p))


def local_kl_bound(tv: float, eps: float) -> float:
    """Any ε-DP local randomizer: KL(M0‖M1) ≤ (e^ε − 1)² TV(P0, P1)²."""
    return math.expm1(eps) ** 2 * tv**2


def local_error_lower_bound(n: int, eps: float, beta: float) -> float:
    """No ε-local protocol is within this of the fraction with probability 1 − β for every population.

    Pinsker on n reports: TV ≤ α (e^ε − 1) √(n/2), and a test built from the estimator needs
    TV ≥ 1 − 2β, so the gap α between two populations must exceed (1 − 2β)√2 / ((e^ε − 1)√n);
    the error is half of it.
    """
    return (1 - 2 * beta) / (math.expm1(eps) * math.sqrt(2 * n))


# --- Local frequency estimation: k-ary randomized response and unary encoding -----------------


def krr_probs(eps: float, k: int) -> tuple[float, float]:
    """k-ary randomized response: report the true value w.p. p, each other value w.p. q."""
    denom = math.exp(eps) + k - 1
    return math.exp(eps) / denom, 1 / denom


def sue_probs(eps: float) -> tuple[float, float]:
    """Symmetric unary encoding (basic RAPPOR): keep each bit w.p. e^{ε/2}/(e^{ε/2} + 1)."""
    p = math.exp(eps / 2) / (math.exp(eps / 2) + 1)
    return p, 1 - p


def oue_probs(eps: float) -> tuple[float, float]:
    """Optimized unary encoding (Wang et al. 2017): a 1 stays 1 w.p. 1/2, a 0 becomes 1 w.p. 1/(e^ε + 1)."""
    return 0.5, 1 / (math.exp(eps) + 1)


def unary_eps(p: float, q: float) -> float:
    """Privacy of a unary encoding: changing the value flips two bits, ratio (p/q)·((1−q)/(1−p))."""
    return math.log(p / q * (1 - q) / (1 - p))


def oracle_variance(p: float, q: float, f: float, n: int) -> float:
    """Var of the unbiased frequency estimate (ĉ/n − q)/(p − q) of a value with true frequency f."""
    return (f * p * (1 - p) + (1 - f) * q * (1 - q)) / (n * (p - q) ** 2)


def oracle_total_mse(p: float, q: float, freqs: Sequence[float], n: int) -> float:
    """Expected squared ℓ2 error of the whole estimated histogram: the sum of the per-value variances."""
    return sum(oracle_variance(p, q, f, n) for f in freqs)


def krr_crossover(eps: float) -> float:
    """k-ary RR beats optimized unary encoding (for rare values) exactly when k < 3e^ε + 2."""
    return 3 * math.exp(eps) + 2


def best_local_probs(eps: float, k: int) -> tuple[float, float]:
    """The better of k-ary RR and optimized unary encoding at budget ε for k values."""
    return krr_probs(eps, k) if k < krr_crossover(eps) else oue_probs(eps)


def population_counts(n: int, k: int) -> IntArray:
    """n users spread over k values with Zipf frequencies ∝ 1/(v + 1); the counts sum to n."""
    weights = 1 / np.arange(1, k + 1)
    counts = np.floor(n * weights / weights.sum()).astype(np.int64)
    counts[0] += n - int(counts.sum())
    return counts


def simulate_krr(rng: np.random.Generator, counts: IntArray, eps: float) -> FloatArray:
    """One run of k-ary RR for every user; returns the unbiased estimated frequencies."""
    k, n = len(counts), int(counts.sum())
    p, q = krr_probs(eps, k)
    histogram = np.zeros(k, dtype=np.int64)
    for v, c in enumerate(counts):
        row = np.full(k, q)
        row[v] = p
        histogram += rng.multinomial(int(c), row)
    return (histogram / n - q) / (p - q)


def simulate_unary(rng: np.random.Generator, counts: IntArray, p: float, q: float) -> FloatArray:
    """One run of a unary encoding: bit v is 1 for Bin(c_v, p) holders and Bin(n − c_v, q) others."""
    n = int(counts.sum())
    ones = rng.binomial(counts, p) + rng.binomial(n - counts, q)
    return (ones / n - q) / (p - q)


def simulate_local(rng: np.random.Generator, counts: IntArray, eps: float) -> FloatArray:
    """The better local frequency oracle at budget ε."""
    k = len(counts)
    if k < krr_crossover(eps):
        return simulate_krr(rng, counts, eps)
    return simulate_unary(rng, counts, *oue_probs(eps))


# --- Large and unknown domains: local hashing and heavy hitters ---------------------------------

HASH_PRIME = 2_147_483_647  # 2^31 − 1, a prime larger than every encoded value


def lh_variance(eps: float, w: float, n: int) -> float:
    """Var of a local-hashing frequency estimate for a rare value (f = 0), with w hash buckets.

    Each user hashes their value with a random hash into w buckets and sends the bucket through
    w-ary randomized response. The variance is (e^ε + w − 1)² / (n (w − 1)(e^ε − 1)²).
    """
    return (math.exp(eps) + w - 1) ** 2 / (n * (w - 1) * math.expm1(eps) ** 2)


def lh_best_width(eps: float) -> int:
    """The integer number of buckets w ≥ 2 with the smallest local-hashing variance; near e^ε + 1."""
    lo = max(2, math.floor(math.exp(eps) + 1))
    return min((lo, lo + 1), key=lambda w: lh_variance(eps, w, 1))


def hash_family(rng: np.random.Generator, size: int) -> tuple[IntArray, IntArray]:
    """`size` random members (a, b) of the family h(x) = ((a x + b) mod P) mod w."""
    a = rng.integers(1, HASH_PRIME, size=size, dtype=np.int64)
    b = rng.integers(0, HASH_PRIME, size=size, dtype=np.int64)
    return a, b


def hash_values(a: IntArray, b: IntArray, x: IntArray, w: int) -> IntArray:
    """h_{a,b}(x) for matching arrays of hash parameters and values; a·x < 2^62 fits in int64."""
    return ((a * x + b) % HASH_PRIME) % w


def simulate_lh(
    rng: np.random.Generator, values: IntArray, candidates: IntArray, eps: float, w: int, family: int
) -> FloatArray:
    """Local hashing with a shared family of `family` hash functions; estimated frequency of each candidate.

    User i draws hash j_i, sends (j_i, w-ary RR of h_{j_i}(x_i)). The server counts, for each
    candidate v, the reports y_i = h_{j_i}(v), and debiases twice: once for the randomized response
    (Fact 1), once for the 1/w chance that another value hashes to v's bucket.
    """
    n = len(values)
    a, b = hash_family(rng, family)
    j = rng.integers(0, family, size=n)
    buckets = hash_values(a[j], b[j], values, w)
    p, q = krr_probs(eps, w)
    keep = rng.random(n) < p
    other = (buckets + rng.integers(1, w, size=n)) % w  # a uniformly random different bucket
    reports = np.where(keep, buckets, other)
    table = np.zeros((family, w), dtype=np.int64)  # table[j, y] = number of users with hash j who sent y
    np.add.at(table, (j, reports), 1)
    estimates = np.empty(len(candidates))
    for c, v in enumerate(candidates):
        support = table[np.arange(family), hash_values(a, b, np.full(family, v), w)].sum() / n
        g = (support - q) / (p - q)  # estimated fraction of users whose own hash puts them in v's bucket
        estimates[c] = (w * g - 1) / (w - 1)  # remove the 1/w collisions of the other values
    return estimates


def false_alarm_z(candidates: int, beta: float) -> float:
    """The threshold, in standard deviations, at which `candidates` pure-noise estimates give β false alarms.

    Each estimate of a value nobody holds is close to normal with mean 0, so the expected number
    above z standard deviations is candidates · (1 − Φ(z)); solve that for β.
    """
    return bisect(lambda z: candidates * (1 - normal_cdf(z)) - beta, 0.0, 40.0)


def word_population(
    rng: np.random.Generator, n: int, words: int, alphabet: int, length: int
) -> tuple[IntArray, IntArray]:
    """n users holding one of `words` random strings with Zipf frequencies ∝ 1/rank; returns (vocabulary, values).

    A string of `length` symbols from an alphabet of size `alphabet` is encoded as a base-`alphabet` integer.
    """
    vocab = rng.choice(alphabet**length, size=words, replace=False).astype(np.int64)
    counts = population_counts(n, words)
    values = np.repeat(vocab, counts)
    rng.shuffle(values)
    return vocab, values


def prefix_heavy_hitters(
    rng: np.random.Generator,
    values: IntArray,
    eps: float,
    alphabet: int,
    length: int,
    beta: float = 0.1,
    family: int = 1024,
) -> tuple[list[tuple[int, float]], list[int]]:
    """Find frequent strings without a dictionary, by growing prefixes one symbol at a time.

    Users are split into `length` groups. Group ℓ reports the prefix of length ℓ of its string by local
    hashing. Each level keeps the candidates whose estimate is above the false-alarm threshold and
    extends each by every symbol. Returns the strings found with their estimated frequencies, and
    the number of candidates tested at each level.
    """
    w = lh_best_width(eps)
    groups = rng.integers(0, length, size=len(values))
    kept = np.zeros(1, dtype=np.int64)  # the empty prefix
    tested: list[int] = []
    found: list[tuple[int, float]] = []
    for level in range(1, length + 1):
        candidates = (kept[:, None] * alphabet + np.arange(alphabet)[None, :]).ravel()
        tested.append(len(candidates))
        members = values[groups == level - 1]
        prefixes = members // alphabet ** (length - level)
        estimates = simulate_lh(rng, prefixes, candidates, eps, w, family)
        sd = math.sqrt(lh_variance(eps, w, len(members)))
        above = estimates > false_alarm_z(len(candidates), beta) * sd
        kept = candidates[above]
        if level == length:
            found = [(int(v), float(e)) for v, e in zip(candidates[above], estimates[above], strict=True)]
    return found, tested


# --- Numbers from one bit: local mean estimation -------------------------------------------------


def one_bit_prob(x: float, m: float, eps: float) -> float:
    """1BitMean (Ding, Kulkarni & Yekhanin 2017): send 1 w.p. 1/(e^ε + 1) + (x/m)(e^ε − 1)/(e^ε + 1)."""
    e = math.exp(eps)
    return 1 / (e + 1) + (x / m) * (e - 1) / (e + 1)


def one_bit_estimate(mean_report: float, m: float, eps: float) -> float:
    """Unbiased mean from the mean of the bits: m ((e^ε + 1) Ȳ − 1)/(e^ε − 1)."""
    e = math.exp(eps)
    return m * ((e + 1) * mean_report - 1) / (e - 1)


def one_bit_sd(values: FloatArray, m: float, eps: float) -> float:
    """Exact sd of the 1BitMean estimate: (m (e^ε + 1)/(e^ε − 1)) √(Σ π_i (1 − π_i)) / n."""
    e = math.exp(eps)
    pi = 1 / (e + 1) + (values / m) * (e - 1) / (e + 1)
    return m * (e + 1) / (e - 1) * math.sqrt(float((pi * (1 - pi)).sum())) / len(values)


def one_bit_sd_bound(n: int, m: float, eps: float) -> float:
    """The worst case over the data, π = 1/2 for everyone: m (e^ε + 1) / (2 (e^ε − 1) √n)."""
    return m * (math.exp(eps) + 1) / (2 * math.expm1(eps) * math.sqrt(n))


def local_laplace_mean_sd(n: int, m: float, eps: float) -> float:
    """Each user sends x + Lap(m/ε); the mean of the reports has sd √2 m / (ε √n)."""
    return math.sqrt(2) * m / (eps * math.sqrt(n))


def one_bit_laplace_crossover() -> float:
    """The ε at which the worst-case 1-bit sd equals the Laplace sd; below it one bit is more accurate."""
    return bisect(lambda e: one_bit_sd_bound(1, 1.0, e) / local_laplace_mean_sd(1, 1.0, e) - 1, 0.1, 10.0)


def usage_minutes(rng: np.random.Generator, n: int, m: float) -> FloatArray:
    """Daily minutes in an app for n users: a log-normal around an hour, clipped to [0, m]."""
    return np.clip(rng.lognormal(math.log(60.0), 0.8, size=n), 0.0, m)


def simulate_one_bit(rng: np.random.Generator, values: FloatArray, m: float, eps: float, trials: int) -> FloatArray:
    """`trials` 1BitMean estimates of the mean of `values`."""
    e = math.exp(eps)
    pi = 1 / (e + 1) + (values / m) * (e - 1) / (e + 1)
    means = np.array([float((rng.random(len(values)) < pi).mean()) for _ in range(trials)])
    return m * ((e + 1) * means - 1) / (e - 1)


# --- Collecting repeatedly: memoization and α-point rounding ------------------------------------


def averaging_log_ratio(yes: int, days: int, eps: float) -> float:
    """ln of Pr[reports | bit 1] / Pr[reports | bit 0] for `yes` ones in `days` fresh randomized responses."""
    return (2 * yes - days) * eps


def majority_error(days: int, eps: float) -> float:
    """Probability that the majority of `days` fresh randomized responses is wrong (ties split evenly)."""
    pmf = binomial_pmf(days, 1 - rr_keep(eps))  # number of flipped reports
    wrong = float(pmf[days // 2 + 1 :].sum())
    if days % 2 == 0:
        wrong += 0.5 * float(pmf[days // 2])
    return wrong


def rappor_eps_inf(hashes: int, f: float) -> float:
    """RAPPOR's bound for all reports of one value: ε∞ = 2h ln((1 − f/2)/(f/2))."""
    return 2 * hashes * math.log((1 - f / 2) / (f / 2))


def rappor_eps_one(hashes: int, f: float, p: float, q: float) -> float:
    """RAPPOR's bound for one report, permanent and instantaneous layers together: ε1 = h ln(q*(1 − p*)/(p*(1 − q*)))."""
    p_star = 0.5 * f * (p + q) + (1 - f) * p
    q_star = 0.5 * f * (p + q) + (1 - f) * q
    return hashes * math.log(q_star * (1 - p_star) / (p_star * (1 - q_star)))


def alpha_point_reports(b0: IntArray, bm: IntArray, alpha: FloatArray, x: FloatArray, m: float) -> IntArray:
    """α-point rounding with memoization: report the memoized bit for m if x + α ≥ m, else the one for 0."""
    return np.where(x + alpha >= m, bm, b0)


def memoized_bits(rng: np.random.Generator, n: int, m: float, eps: float) -> tuple[IntArray, IntArray, FloatArray]:
    """Each user's two memoized 1BitMean bits, for the values 0 and m, and their secret offset α ~ U[0, m)."""
    b0 = (rng.random(n) < one_bit_prob(0.0, m, eps)).astype(np.int64)
    bm = (rng.random(n) < one_bit_prob(m, m, eps)).astype(np.int64)
    return b0, bm, rng.uniform(0.0, m, size=n)


def equal_memoized_bits(eps: float) -> float:
    """Probability that a user's two memoized bits agree, 2e^ε/(e^ε + 1)²: then no report ever depends on the data."""
    e = math.exp(eps)
    return 2 * e / (e + 1) ** 2


def drifting_usage(rng: np.random.Generator, n: int, days: int, spread: float, m: float) -> FloatArray:
    """n users × days: a personal base level from usage_minutes, plus a uniform daily wobble of ± spread."""
    base = usage_minutes(rng, n, m)
    wobble = rng.uniform(-spread, spread, size=(days, n))
    return np.clip(base[None, :] + wobble, 0.0, m)


def changed_fraction(reports: IntArray) -> float:
    """Fraction of users (columns) whose report changes at least once over the days (rows)."""
    return float((reports != reports[0][None, :]).any(axis=0).mean())


def alpha_point_changes(rng: np.random.Generator, n: int, days: int, spread: float, m: float, eps: float) -> float:
    """Fraction of users whose α-point report ever changes when their usage wobbles by ± spread."""
    x = drifting_usage(rng, n, days, spread, m)
    b0, bm, alpha = memoized_bits(rng, n, m, eps)
    return changed_fraction(alpha_point_reports(b0[None, :], bm[None, :], alpha[None, :], x, m))


def expected_range(days: int, spread: float) -> float:
    """E[max − min] of `days` independent uniforms on [−spread, spread]: 2 spread (days − 1)/(days + 1)."""
    return 2 * spread * (days - 1) / (days + 1)


def alpha_point_change_theory(days: int, spread: float, m: float, eps: float) -> float:
    """A report changes only if the two memoized bits differ and the threshold m − α falls inside the wobble."""
    return (1 - equal_memoized_bits(eps)) * expected_range(days, spread) / m


def fresh_changes(rng: np.random.Generator, n: int, days: int, spread: float, m: float, eps: float) -> float:
    """The same for a fresh 1BitMean report every day."""
    x = drifting_usage(rng, n, days, spread, m)
    e = math.exp(eps)
    pi = 1 / (e + 1) + (x / m) * (e - 1) / (e + 1)
    return changed_fraction((rng.random(x.shape) < pi).astype(np.int64))


def output_perturbation_eps(eps: float, gamma: float) -> float:
    """One round of 1BitMean whose bit is then flipped w.p. γ (Ding et al., Theorem 6)."""
    e = math.exp(eps)
    return math.log(((1 - 2 * gamma) * e / (e + 1) + gamma) / ((1 - 2 * gamma) / (e + 1) + gamma))


def shuffle_random_bit_prob(eps0: float) -> float:
    """Binary randomized response at ε0 written as "send a uniform random bit w.p. 2/(e^ε0 + 1), else the truth"."""
    return 2 / (math.exp(eps0) + 1)


# --- Central baselines -------------------------------------------------------------------------


def central_laplace_mse(n: int, k: int, eps: float) -> float:
    """Histogram under replacement has ℓ1 sensitivity 2: Lap(2/ε) per count, variance 8/ε² each."""
    return k * 8 / (eps**2 * n**2)


def simulate_central_laplace(rng: np.random.Generator, counts: IntArray, eps: float) -> FloatArray:
    n = int(counts.sum())
    return (counts + rng.laplace(0.0, 2 / eps, size=len(counts))) / n


# --- The shuffle model -------------------------------------------------------------------------


def fmt_condition(eps0: float, n: int, delta: float) -> bool:
    """Feldman, McMillan & Talwar (2021), Theorem 3.1 needs ε0 ≤ ln(n / (16 ln(2/δ)))."""
    return eps0 <= math.log(n / (16 * math.log(2 / delta)))


def fmt_closed_form(eps0: float, n: int, delta: float) -> float:
    """Theorem 3.1 of Feldman, McMillan & Talwar (2021), equation (1):

    ε ≤ ln(1 + (e^ε0 − 1)/(e^ε0 + 1) · (8 √(e^ε0 ln(4/δ)) / √n + 8 e^ε0 / n)).
    Returns inf when the theorem's condition on ε0 fails.
    """
    if not fmt_condition(eps0, n, delta):
        return math.inf
    e0 = math.exp(eps0)
    inner = 8 * math.sqrt(e0 * math.log(4 / delta)) / math.sqrt(n) + 8 * e0 / n
    return math.log1p((e0 - 1) / (e0 + 1) * inner)


def fmt_max_eps0(eps: float, n: int, delta: float) -> float:
    """The largest local ε0 whose shuffled reports are (ε, δ)-DP by the closed form."""
    cap = math.log(n / (16 * math.log(2 / delta)))
    if fmt_closed_form(cap, n, delta) <= eps:
        return cap
    return bisect(lambda e0: fmt_closed_form(e0, n, delta) - eps, 0.0, cap)


def clone_split(
    r_x: Sequence[float], r_x0: Sequence[float], r_x1: Sequence[float], eps0: float
) -> tuple[list[float], list[float]]:
    """Write R(x) = e^{-ε0}/2 · R(x0) + e^{-ε0}/2 · R(x1) + (1 − e^{-ε0}) · LO(x).

    Returns (the clone part e^{-ε0}/2 (R(x0) + R(x1)), the leftover mass (1 − e^{-ε0}) LO(x)).
    """
    w = math.exp(-eps0) / 2
    clones = [w * (a + b) for a, b in zip(r_x0, r_x1, strict=True)]
    return clones, [r - c for r, c in zip(r_x, clones, strict=True)]


def _window(center: float, half: float, lo: int, hi: int) -> tuple[int, int]:
    return max(lo, int(math.floor(center - half))), min(hi, int(math.ceil(center + half)))


@cache
def fmt_clone_pair(eps0: float, n: int) -> tuple[FloatArray, FloatArray, FloatArray, float]:
    """The pair of Theorem 3.2: C ~ Bin(n−1, e^{-ε0}), A ~ Bin(C, 1/2), Δ ~ Bern(e^ε0/(e^ε0 + 1)).

    P = (A + Δ, C − A + 1 − Δ) and Q = (A + 1 − Δ, C − A + Δ). Both are fixed by (C, first
    coordinate), so they are returned as flat arrays over the pairs (c, k) that carry mass:
    (weight Pr[C = c], P(k | c), Q(k | c), mass dropped by the truncation).
    """
    pd = math.exp(eps0) / (math.exp(eps0) + 1)
    c_pmf = binomial_pmf(n - 1, math.exp(-eps0))
    mean, sd = (n - 1) * math.exp(-eps0), math.sqrt((n - 1) * math.exp(-eps0) * (1 - math.exp(-eps0)))
    c_lo, c_hi = _window(mean, 12 * sd + 5, 0, n - 1)
    lf = log_factorials(n)
    weights: list[FloatArray] = []
    p_rows: list[FloatArray] = []
    q_rows: list[FloatArray] = []
    dropped = float(c_pmf[:c_lo].sum() + c_pmf[c_hi + 1 :].sum())
    for c in range(c_lo, c_hi + 1):
        k_lo, k_hi = _window(c / 2, 7 * math.sqrt(c) + 5, 0, c + 1)
        k = np.arange(k_lo, k_hi + 1)

        # b(j) = Pr[Bin(c, 1/2) = j], zero outside 0..c
        def b(j: IntArray) -> FloatArray:
            inside = (j >= 0) & (j <= c)
            jj = np.clip(j, 0, c)
            return np.where(inside, np.exp(lf[c] - lf[jj] - lf[c - jj] - c * math.log(2)), 0.0)

        b_k, b_km1 = b(k), b(k - 1)
        weights.append(np.full(len(k), c_pmf[c]))
        p_rows.append(pd * b_km1 + (1 - pd) * b_k)
        q_rows.append((1 - pd) * b_km1 + pd * b_k)
    return np.concatenate(weights), np.concatenate(p_rows), np.concatenate(q_rows), dropped


def fmt_numeric_delta(eps: float, eps0: float, n: int) -> float:
    """δ(ε) of the Theorem 3.2 pair (the pair is symmetric, so one direction suffices)."""
    w, p, q, dropped = fmt_clone_pair(eps0, n)
    return float((w * np.maximum(p - math.exp(eps) * q, 0.0)).sum()) + dropped


def fmt_numeric(eps0: float, n: int, delta: float) -> float:
    """The smallest ε that Theorem 3.2 certifies, by computing its hockey-stick divergence exactly."""
    return epsilon_for_delta(lambda e: fmt_numeric_delta(e, eps0, n), delta, eps0)


@cache
def shuffled_rr_pair(eps0: float, n: int, ones: int) -> tuple[FloatArray, FloatArray]:
    """Shuffled binary RR: the analyst sees only the number of 1-reports.

    The other n − 1 users hold `ones` ones; the differing user holds 0 in D and 1 in D'.
    Returns the pmfs of the count of 1-reports under D and D'.
    """
    p = rr_keep(eps0)
    others = convolve(binomial_pmf(ones, p), binomial_pmf(n - 1 - ones, 1 - p))
    shifted = np.concatenate(([0.0], others))
    others = np.concatenate((others, [0.0]))
    return p * others + (1 - p) * shifted, (1 - p) * others + p * shifted


def shuffled_rr_delta(eps: float, eps0: float, n: int, ones: int) -> float:
    d, d_prime = shuffled_rr_pair(eps0, n, ones)
    return max(hockey_stick(d, d_prime, eps), hockey_stick(d_prime, d, eps))


def shuffled_rr_eps(eps0: float, n: int, delta: float, grid: int = 5) -> float:
    """Exact ε of shuffled binary RR on a few dataset pairs, maximized: a lower bound on any valid bound."""
    ones_grid = sorted({round(i * (n - 1) / (grid - 1)) for i in range(grid)})
    return max(epsilon_for_delta(lambda e, m=m: shuffled_rr_delta(e, eps0, n, m), delta, eps0) for m in ones_grid)


# --- Distributed DP: noise split across clients ------------------------------------------------


def gaussian_delta(eps: float, sigma: float, sensitivity: float) -> float:
    """Exact δ(ε) of the Gaussian mechanism (Balle & Wang 2018)."""
    a, b = sensitivity / (2 * sigma), eps * sigma / sensitivity
    return normal_cdf(a - b) - math.exp(eps) * normal_cdf(-a - b)


def gaussian_sigma(eps: float, delta: float, sensitivity: float) -> float:
    """The smallest σ for which the Gaussian mechanism is (ε, δ)-DP."""
    return bisect(lambda s: delta - gaussian_delta(eps, s, sensitivity), 1e-3 * sensitivity, 1e3 * sensitivity)


def gaussian_eps(sigma: float, delta: float, sensitivity: float) -> float:
    """The ε of a Gaussian mechanism with noise σ at a given δ."""
    return epsilon_for_delta(lambda e: gaussian_delta(e, sigma, sensitivity), delta, 200.0)


def distributed_mse(n: int, k: int, eps: float, delta: float) -> float:
    """Distributed noise totalling N(0, σ²) per count, σ calibrated for ℓ2 sensitivity √2."""
    return k * gaussian_sigma(eps, delta, math.sqrt(2)) ** 2 / n**2


def skellam_shares(rng: np.random.Generator, variance: float, clients: int, size: int) -> IntArray:
    """Each client adds Pois(v/(2m)) − Pois(v/(2m)); the m shares sum to a Skellam of variance v."""
    lam = variance / (2 * clients)
    return rng.poisson(lam, size=(clients, size)) - rng.poisson(lam, size=(clients, size))


def secure_sum(rng: np.random.Generator, vectors: IntArray, modulus: int) -> IntArray:
    """Secure aggregation as a black box, here two-server additive sharing mod q.

    Each client splits its vector into two shares that are uniform mod q on their own; each
    server adds the shares it holds; the two totals together reveal only the sum, mod q.
    """
    share_a = rng.integers(0, modulus, size=vectors.shape)
    share_b = (vectors - share_a) % modulus
    total = (share_a.sum(axis=0) + share_b.sum(axis=0)) % modulus
    return np.where(total > modulus // 2, total - modulus, total)  # back to signed integers


def simulate_distributed(rng: np.random.Generator, counts: IntArray, eps: float, delta: float) -> FloatArray:
    """The histogram released by secure aggregation of one-hot vectors plus Skellam shares.

    Secure aggregation releases exactly the sum, and by Poisson additivity the sum of the n
    shares is one Skellam of the total variance, so the sum is drawn directly (the shares and
    the secure sum are checked separately in print_table).
    """
    n, k = int(counts.sum()), len(counts)
    sigma = gaussian_sigma(eps, delta, math.sqrt(2))
    noise = skellam_shares(rng, sigma**2, 1, k)[0]
    return (counts + noise) / n


def collusion_eps(eps: float, delta: float, honest_fraction: float) -> float:
    """ε left when only a fraction of the clients' noise is unknown to the adversary."""
    sigma = gaussian_sigma(eps, delta, math.sqrt(2))
    return gaussian_eps(sigma * math.sqrt(honest_fraction), delta, math.sqrt(2))


# --- Matched-ε comparison -----------------------------------------------------------------------


def model_mse(model: str, n: int, k: int, eps: float, delta: float) -> float:
    """Expected squared ℓ2 error of the estimated histogram under each trust model, at central (ε, δ)."""
    freqs = [float(c) / n for c in population_counts(n, k)]
    if model == "central":
        return central_laplace_mse(n, k, eps)
    if model == "distributed":
        return distributed_mse(n, k, eps, delta)
    if model == "shuffle":
        eps0 = fmt_max_eps0(eps, n, delta)
        return oracle_total_mse(*best_local_probs(eps0, k), freqs, n)
    if model == "local":
        return oracle_total_mse(*best_local_probs(eps, k), freqs, n)
    raise ValueError(model)


def simulate_model(
    rng: np.random.Generator, model: str, n: int, k: int, eps: float, delta: float, trials: int
) -> float:
    """Average squared ℓ2 error of the estimated histogram over `trials` runs."""
    counts = population_counts(n, k)
    truth = counts / n
    errors: list[float] = []
    for _ in range(trials):
        if model == "central":
            est = simulate_central_laplace(rng, counts, eps)
        elif model == "distributed":
            est = simulate_distributed(rng, counts, eps, delta)
        elif model == "shuffle":
            est = simulate_local(rng, counts, fmt_max_eps0(eps, n, delta))
        else:
            est = simulate_local(rng, counts, eps)
        errors.append(float(((est - truth) ** 2).sum()))
    return float(np.mean(errors))


MODELS = ("central", "distributed", "shuffle", "local")


# --- The chapter's numbers ----------------------------------------------------------------------


def print_table(rng: np.random.Generator) -> None:
    eps, n = math.log(3), 10_000
    print(f"[Randomized response, ε = ln 3, n = {n}, 3,000 ones]")
    est = simulate_rr(rng, n, 3_000, eps, 20_000)
    print(
        f"  keep p={rr_keep(eps):.4f}  Var factor e^ε/(e^ε−1)^2={n * rr_variance(n, eps):.4f}  "
        f"sd={math.sqrt(rr_variance(n, eps)):.5f}  (simulated mean {est.mean():.4f}, sd {est.std():.5f})"
    )
    print(
        f"  count sd: local={n * math.sqrt(rr_variance(n, eps)):.1f}  central Laplace={n * central_count_sd(n, eps):.3f}  "
        f"ratio={math.sqrt(rr_variance(n, eps)) / central_count_sd(n, eps):.1f}"
    )

    print("[The √n gap at ε = 1]")
    for m in (1_000, 100_000, 10_000_000):
        local, central = math.sqrt(rr_variance(m, 1.0)), central_count_sd(m, 1.0)
        print(
            f"  n={m}: local sd={local:.2e}  central sd={central:.2e}  ratio={local / central:.0f}  "
            f"lower bound (β=0.1)={local_error_lower_bound(m, 1.0, 0.1):.2e}"
        )
    print(
        f"  one report, Bern(0.5) vs Bern(0.6), ε=1: KL={rr_report_kl(0.5, 0.6, 1.0):.5f}  "
        f"bound (e^ε−1)^2 TV^2={local_kl_bound(0.1, 1.0):.5f}"
    )

    print("[Frequency oracles at ε = 1: n · Var for a rare value (f = 0)]")
    sue = sue_probs(1.0)
    oue = oue_probs(1.0)
    print(f"  unary encodings: symmetric eps={unary_eps(*sue):.4f} optimized eps={unary_eps(*oue):.4f}")
    for k in (2, 8, 32, 128):
        print(
            f"  k={k}: k-RR={oracle_variance(*krr_probs(1.0, k), 0.0, 1):.3f}  "
            f"SUE={oracle_variance(*sue, 0.0, 1):.3f}  OUE={oracle_variance(*oue, 0.0, 1):.3f}"
        )
    print(f"  k-RR wins for k < 3e^ε + 2 = {krr_crossover(1.0):.2f}")

    print("[Clones: 4-ary RR, ε0 = ln 3, x = value 3, x0 = value 1, x1 = value 2]")
    p, q = krr_probs(math.log(3), 4)
    r_x0, r_x1, r_x = [p, q, q, q], [q, p, q, q], [q, q, p, q]
    clones, left = clone_split(r_x, r_x0, r_x1, math.log(3))
    print(
        f"  R(x)={[round(v, 4) for v in r_x]}  clones={[round(v, 4) for v in clones]}  "
        f"leftover={[round(v, 4) for v in left]}  leftover mass={sum(left):.4f}"
    )

    delta, n = 1e-6, 10_000
    print(f"[Shuffling ε0-DP reports, n = {n}, δ = 1e-6; ε0 must be ≤ {math.log(n / (16 * math.log(2 / delta))):.3f}]")
    for eps0 in (0.5, 1.0, 2.0, 3.0):
        print(
            f"  ε0={eps0:g}: closed form={fmt_closed_form(eps0, n, delta):.4f}  "
            f"numeric (Thm 3.2)={fmt_numeric(eps0, n, delta):.4f}  "
            f"shuffled binary RR (exact, lower bound)={shuffled_rr_eps(eps0, n, delta):.4f}"
        )
    for m in (1_000, 10_000, 100_000, 1_000_000):
        print(f"  n={m}: largest ε0 for central ε=1: {fmt_max_eps0(1.0, m, delta):.3f}")
    print(
        f"  8-ary RR at ε0={fmt_max_eps0(1.0, 1_000_000, delta):.3f} keeps the truth w.p. "
        f"{krr_probs(fmt_max_eps0(1.0, 1_000_000, delta), 8)[0]:.3f}"
    )

    print("[Distributed Gaussian noise, (ε, δ) = (1, 1e-6), ℓ2 sensitivity √2, 100 clients]")
    sigma = gaussian_sigma(1.0, delta, math.sqrt(2))
    print(f"  total σ={sigma:.3f}  per-client σ={sigma / 10:.3f}")
    for honest in (0.8, 0.5, 0.1):
        print(f"  honest fraction {honest:g}: ε={collusion_eps(1.0, delta, honest):.3f}")
    shares = skellam_shares(rng, 25.0, 100, 200_000).sum(axis=0)
    print(f"  sum of 100 Skellam shares with total variance 25: mean={shares.mean():.3f}  var={shares.var():.2f}")
    vec = np.array([[3, -2], [5, 7], [0, 1]], dtype=np.int64)
    print(f"  secure sum of {vec.tolist()} mod 2^20: {secure_sum(rng, vec, 2**20).tolist()}")

    k, trials = 8, 200
    print(f"[Matched ε = 1, δ = 1e-6, k = {k} Zipf values: total squared error of the histogram]")
    for m in (1_000, 10_000, 100_000):
        cells = "  ".join(
            f"{model}={model_mse(model, m, k, 1.0, delta):.2e}"
            f" (sim {simulate_model(rng, model, m, k, 1.0, delta, trials):.2e})"
            for model in MODELS
        )
        print(f"  n={m}: {cells}")

    print("[Local hashing: n · Var for a rare value]")
    for e in (1.0, 4.0):
        w = lh_best_width(e)
        print(
            f"  ε={e:g}: e^ε + 1={math.exp(e) + 1:.2f}  best w={w}  local hashing={lh_variance(e, w, 1):.4f}  "
            f"optimized unary={oracle_variance(*oue_probs(e), 0.0, 1):.4f}  w=2: {lh_variance(e, 2, 1):.4f}"
        )

    n, words, alphabet, length, eps = 1_000_000, 200, 26, 6, 4.0
    print(f"[Heavy hitters: n={n}, {words} words of {length} letters, ε={eps:g}, prefixes grown one letter at a time]")
    vocab, values = word_population(rng, n, words, alphabet, length)
    found, tested = prefix_heavy_hitters(rng, values, eps, alphabet, length, family=16_384)
    freqs = dict(zip(vocab.tolist(), (population_counts(n, words) / n).tolist(), strict=True))
    hits = [v for v, _ in found if v in freqs]
    w = lh_best_width(eps)
    group = n // length
    sd = math.sqrt(lh_variance(eps, w, group))
    z_last = false_alarm_z(tested[-1], 0.1)
    z_all = false_alarm_z(alphabet**length, 0.1)
    print(f"  w={w}  users per level≈{group}  sd of one estimate={sd:.2e}  candidates per level={tested}")
    print(f"  found {len(found)} strings: {len(hits)} real words, {len(found) - len(hits)} false")
    missed = max((f for v, f in freqs.items() if v not in hits), default=0.0)
    print(
        f"  smallest frequency found={min(freqs[v] for v in hits):.4f}  largest missed={missed:.4f}  "
        f"last-level threshold={z_last:.2f} sd={z_last * sd:.4f}"
    )
    print(
        f"  testing all {alphabet**length} strings instead: threshold {z_all:.2f} sd={z_all * sd:.4f}; "
        f"at the prefix threshold it would expect {alphabet**length * (1 - normal_cdf(z_last)):.0f} false alarms"
    )

    m, n, eps = 1440.0, 1_000_000, 1.0
    print(f"[Local mean of minutes in [0, {m:g}], n={n}, ε={eps:g}]")
    x = usage_minutes(rng, n, m)
    est = simulate_one_bit(rng, x, m, eps, 200)
    print(
        f"  true mean={x.mean():.2f}  1-bit sd={one_bit_sd(x, m, eps):.3f} (worst case {one_bit_sd_bound(n, m, eps):.3f}; "
        f"simulated mean {est.mean():.2f}, sd {est.std():.3f})  Laplace sd={local_laplace_mean_sd(n, m, eps):.3f}"
    )
    print(
        f"  worst-case 1-bit / Laplace sd={one_bit_sd_bound(n, m, eps) / local_laplace_mean_sd(n, m, eps):.3f}  "
        f"as ε→0: {one_bit_sd_bound(n, m, 1e-6) / local_laplace_mean_sd(n, m, 1e-6):.3f}  "
        f"equal at ε={one_bit_laplace_crossover():.3f}"
    )

    eps = math.log(3)
    print("[Repeated collection]")
    ratio = averaging_log_ratio(75, 100, eps)
    print(f"  fresh RR at ln 3, 75 ones in 100 days: log ratio={ratio:.2f}  ratio={math.exp(ratio):.3g}")
    for days in (1, 5, 25, 101):
        print(f"  days={days}: majority of fresh reports wrong w.p. {majority_error(days, eps):.3g}")
    print(
        f"  RAPPOR homepages (h=2, f=0.75, p=0.5, q=0.75): eps_inf={rappor_eps_inf(2, 0.75):.3f}  "
        f"eps_1={rappor_eps_one(2, 0.75, 0.5, 0.75):.4f}"
    )
    m, days = 1440.0, 30
    print(f"  α-point rounding, ε=1: memoized bits agree w.p. {equal_memoized_bits(1.0):.4f}")
    for spread in (5.0, 15.0, 30.0):
        sim = alpha_point_changes(rng, 100_000, days, spread, m, 1.0)
        fresh = fresh_changes(rng, 100_000, days, spread, m, 1.0)
        print(
            f"  ± {spread:g} min over {days} days: report ever changes α-point={sim:.4f} "
            f"(theory {alpha_point_change_theory(days, spread, m, 1.0):.4f})  fresh={fresh:.4f}"
        )
    b0, bm, alpha = memoized_bits(rng, n, m, 1.0)
    x = usage_minutes(rng, n, m)
    day = alpha_point_reports(b0, bm, alpha, x, m)
    print(
        f"  one day, n={n}: true mean={x.mean():.2f}  α-point estimate={one_bit_estimate(float(day.mean()), m, 1.0):.2f}"
    )
    print(
        f"  output perturbation γ=0.2 on ε=1: one round {output_perturbation_eps(1.0, 0.2):.3f}  "
        f"per round where the memoized bit differs ln(0.8/0.2)={math.log(0.8 / 0.2):.3f}"
    )

    eps0 = fmt_max_eps0(1.0, 1_000_000, 1e-6)
    p_rand = shuffle_random_bit_prob(eps0)
    print(
        f"[Shuffled bits at ε0={eps0:.3f}, n=10^6] random bit w.p. {p_rand:.2e}  expected random senders={1e6 * p_rand:.0f}  "
        f"count sd={1e6 * math.sqrt(rr_variance(1_000_000, eps0)):.1f}  local at ε0=1: {1e6 * math.sqrt(rr_variance(1_000_000, 1.0)):.0f}"
    )


# --- Figures ------------------------------------------------------------------------------------


def _box(ax: Axes, x: float, y: float, w: float, h: float, text: str, color: str) -> None:
    ax.add_patch(
        FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02", facecolor=color, edgecolor="0.3", linewidth=0.8)
    )
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=8)


def _arrow(ax: Axes, x0: float, y0: float, x1: float, y1: float, label: str = "") -> None:
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0), arrowprops={"arrowstyle": "->", "color": "0.3", "lw": 0.9})
    if label:
        ax.text((x0 + x1) / 2, (y0 + y1) / 2 + 0.03, label, ha="center", va="bottom", fontsize=6.5, color="0.25")


def plot_models() -> Figure:
    """Four panels: where the raw data goes and where the noise is added in each trust model."""
    fig, axes = plt.subplots(1, 4, figsize=(13.5, 3.6))
    trusted, untrusted, user = "#cfe3cf", "#f3d0cc", "#dde6f3"
    titles = ["Central", "Local", "Shuffle", "Distributed"]
    for ax, title in zip(axes, titles, strict=True):
        ax.set(xlim=(-0.04, 1.03), ylim=(0.05, 0.95), title=title)
        ax.axis("off")
        for i, y in enumerate((0.75, 0.5, 0.25)):
            _box(ax, 0.0, y - 0.07, 0.15, 0.14, f"user {i + 1}", user)
    a0, a1, a2, a3 = axes
    _box(a0, 0.36, 0.38, 0.26, 0.24, "curator\nadds noise", trusted)
    _box(a0, 0.78, 0.42, 0.21, 0.16, "analyst", untrusted)
    for y in (0.75, 0.5, 0.25):
        _arrow(a0, 0.16, y, 0.35, 0.5, "x" if y == 0.5 else "")
    _arrow(a0, 0.63, 0.5, 0.77, 0.5, "M(D)")
    _box(a1, 0.62, 0.38, 0.36, 0.24, "server\n(untrusted)", untrusted)
    for y in (0.75, 0.5, 0.25):
        _arrow(a1, 0.16, y, 0.61, 0.5, "R(x)" if y == 0.5 else "")
    _box(a2, 0.33, 0.3, 0.22, 0.4, "shuffler\npermutes,\nstrips IDs", trusted)
    _box(a2, 0.74, 0.42, 0.25, 0.16, "analyst", untrusted)
    for y in (0.75, 0.5, 0.25):
        _arrow(a2, 0.16, y, 0.32, y * 0.4 + 0.3, "R(x)" if y == 0.5 else "")
    _arrow(a2, 0.56, 0.5, 0.73, 0.5, "multiset")
    _box(a3, 0.33, 0.3, 0.25, 0.4, "secure\naggregation\n(crypto)", trusted)
    _box(a3, 0.74, 0.42, 0.25, 0.16, "server", untrusted)
    for y in (0.75, 0.5, 0.25):
        _arrow(a3, 0.16, y, 0.32, y * 0.4 + 0.3, "x+share" if y == 0.5 else "")
    _arrow(a3, 0.59, 0.5, 0.73, 0.5, "sum")
    fig.text(
        0.5,
        0.03,
        "green: must be trusted;   red: sees what is released, not trusted;   blue: holds one record",
        ha="center",
        fontsize=9,
        color="0.3",
    )
    return fig


def plot_rr_gap(rng: np.random.Generator, eps: float = 1.0) -> Figure:
    ns = [int(round(10 ** (e / 4))) for e in range(8, 29)]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.loglog(ns, [math.sqrt(rr_variance(m, eps)) for m in ns], label="local: randomized response")
    ax.loglog(
        ns,
        [local_error_lower_bound(m, eps, 0.1) for m in ns],
        "--",
        label=r"local lower bound (any protocol, $\beta = 0.1$)",
    )
    ax.loglog(ns, [central_count_sd(m, eps) for m in ns], label="central: Laplace on the count")
    sim_ns = [100, 1_000, 10_000, 100_000]
    ax.loglog(
        sim_ns,
        [float(simulate_rr(rng, m, m // 3, eps, 4_000).std()) for m in sim_ns],
        "o",
        color="C0",
        label="local, simulated",
    )
    ax.set(
        xlabel="number of users n",
        ylabel="error of the estimated fraction (sd)",
        title=rf"Counting at $\varepsilon = {eps:g}$: local error falls like $1/\sqrt{{n}}$, central like $1/n$",
    )
    ax.legend(fontsize=9)
    return fig


def plot_frequency_oracles() -> Figure:
    ks = list(range(2, 201))
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for ax, eps in zip(axes, (1.0, 3.0), strict=True):
        ax.semilogy(ks, [oracle_variance(*krr_probs(eps, k), 0.0, 1) for k in ks], label="k-ary RR")
        ax.semilogy(ks, [oracle_variance(*sue_probs(eps), 0.0, 1)] * len(ks), label="symmetric unary (RAPPOR)")
        ax.semilogy(ks, [oracle_variance(*oue_probs(eps), 0.0, 1)] * len(ks), label="optimized unary")
        ax.axvline(krr_crossover(eps), color="0.6", linestyle=":", label=r"$k = 3e^\varepsilon + 2$")
        ax.set(
            xlabel="domain size k", ylabel=r"$n\cdot\mathrm{Var}$ of one frequency", title=rf"$\varepsilon = {eps:g}$"
        )
        ax.legend(fontsize=8)
    fig.suptitle("Local frequency oracles: k-ary RR degrades with k, unary encodings do not")
    return fig


def plot_clones() -> Figure:
    eps0 = math.log(3)
    p, q = krr_probs(eps0, 4)
    r_x0, r_x1, r_x = [p, q, q, q], [q, p, q, q], [q, q, p, q]
    clones, left = clone_split(r_x, r_x0, r_x1, eps0)
    w = math.exp(-eps0) / 2
    xs = np.arange(4)
    fig, ax = plt.subplots(figsize=(7, 4.2))
    part0 = [w * a for a in r_x0]
    part1 = [w * b for b in r_x1]
    ax.bar(xs, part0, color="C0", label=r"$\frac{e^{-\varepsilon_0}}{2}R(x_0)$: looks like the target $x_0$")
    ax.bar(xs, part1, bottom=part0, color="C1", label=r"$\frac{e^{-\varepsilon_0}}{2}R(x_1)$: looks like $x_1$")
    ax.bar(xs, left, bottom=clones, color="0.75", label=r"$(1 - e^{-\varepsilon_0})\,\mathrm{LO}(x)$: left over")
    ax.bar(xs, r_x, fill=False, edgecolor="k", linewidth=1.2, label=r"$R(x)$, the whole report distribution")
    ax.set_xticks(xs, ["1", "2", "3", "4"])
    ax.set(
        xlabel="report",
        ylabel="probability",
        ylim=(0, 0.75),
        title=r"4-ary RR, $\varepsilon_0 = \ln 3$: the report of user $x = 3$ split into clones",
    )
    ax.legend(fontsize=8, loc="upper right")
    return fig


def plot_shuffle_bound(delta: float = 1e-6) -> Figure:
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    for i, eps0 in enumerate((1.0, 3.0)):
        n_min = math.ceil(16 * math.log(2 / delta) * math.exp(eps0))
        ns = sorted({int(round(10 ** (e / 8))) for e in range(24, 49)} | {n_min})
        ns = [m for m in ns if m >= n_min]
        ax.loglog(
            ns,
            [fmt_closed_form(eps0, m, delta) for m in ns],
            color=f"C{i}",
            label=rf"closed form, $\varepsilon_0 = {eps0:g}$",
        )
        pts = [m for m in (1_000, 3_000, 10_000, 30_000) if m >= n_min]
        ax.loglog(pts, [fmt_numeric(eps0, m, delta) for m in pts], "o", color=f"C{i}", label="numeric (Theorem 3.2)")
        ax.loglog(
            pts,
            [shuffled_rr_eps(eps0, m, delta) for m in pts],
            "x",
            color=f"C{i}",
            label="exact, shuffled binary RR (lower bound)",
        )
        ax.axhline(eps0, color=f"C{i}", linestyle=":", linewidth=0.8)
    ax.set(
        xlabel="number of users n",
        ylabel=r"central $\varepsilon$ at $\delta = 10^{-6}$",
        ylim=(0.003, 5),
        title=r"Amplification by shuffling: $\varepsilon$ falls like $1/\sqrt{n}$ (dotted: $\varepsilon_0$)",
    )
    ax.legend(fontsize=8, ncol=2, loc="lower left")
    return fig


def plot_matched_error(rng: np.random.Generator, eps: float = 1.0, delta: float = 1e-6, k: int = 8) -> Figure:
    ns = [int(round(10 ** (e / 4))) for e in range(12, 25)]
    sim_ns = [1_000, 10_000, 100_000]
    labels = {
        "central": "central (Laplace)",
        "distributed": "distributed (Skellam + secure sum)",
        "shuffle": r"shuffle (local $\varepsilon_0$ from the closed form)",
        "local": "local (best oracle)",
    }
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    for i, model in enumerate(MODELS):
        ax.loglog(ns, [model_mse(model, m, k, eps, delta) for m in ns], color=f"C{i}", label=labels[model])
        ax.loglog(sim_ns, [simulate_model(rng, model, m, k, eps, delta, 100) for m in sim_ns], "o", color=f"C{i}")
    ax.set(
        xlabel="number of users n",
        ylabel=r"expected squared $\ell_2$ error of the histogram",
        title=rf"Frequency estimation, $k = {k}$, central $(\varepsilon, \delta) = ({eps:g}, 10^{{-6}})$; dots: simulation",
    )
    ax.legend(fontsize=8)
    return fig


def plot_collusion(delta: float = 1e-6) -> Figure:
    fractions = [i / 100 for i in range(5, 101)]
    fig, ax = plt.subplots(figsize=(7, 4.2))
    for eps in (0.5, 1.0, 2.0):
        ax.semilogy(
            fractions, [collusion_eps(eps, delta, f) for f in fractions], label=rf"target $\varepsilon = {eps:g}$"
        )
    ax.set(
        xlabel="fraction of the noise not known to the adversary",
        ylabel=r"actual $\varepsilon$ at $\delta = 10^{-6}$",
        title="Distributed noise: colluding or dropped clients erode the guarantee",
    )
    ax.legend(fontsize=9)
    return fig


def plot_local_hashing() -> Figure:
    widths = list(range(2, 400))
    fig, ax = plt.subplots(figsize=(7, 4.3))
    for i, eps in enumerate((1.0, 4.0)):
        ax.loglog(
            widths,
            [lh_variance(eps, w, 1) for w in widths],
            color=f"C{i}",
            label=rf"local hashing, $\varepsilon = {eps:g}$",
        )
        ax.axhline(
            oracle_variance(*oue_probs(eps), 0.0, 1), color=f"C{i}", linestyle=":", label="optimized unary encoding"
        )
        best = math.exp(eps) + 1
        ax.plot([best], [lh_variance(eps, best, 1)], "o", color=f"C{i}")
    ax.set(
        xlabel="number of hash buckets w",
        ylabel=r"$n\cdot\mathrm{Var}$ of a rare value's frequency",
        title=r"Local hashing matches unary encoding at $w = e^\varepsilon + 1$ (dots)",
    )
    ax.legend(fontsize=8)
    return fig


def plot_heavy_hitters(rng: np.random.Generator) -> Figure:
    n, words, alphabet, length, eps = 1_000_000, 200, 26, 6, 4.0
    vocab, values = word_population(rng, n, words, alphabet, length)
    found, tested = prefix_heavy_hitters(rng, values, eps, alphabet, length, family=16_384)
    true = population_counts(n, words) / n
    est = dict(found)
    ranks = np.arange(1, words + 1)
    hit = np.array([int(v) in est for v in vocab.tolist()])
    sd = math.sqrt(lh_variance(eps, lh_best_width(eps), n // length))
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.loglog(ranks[hit], true[hit], "o", color="C0", label="word found (true frequency)")
    ax.loglog(ranks[~hit], true[~hit], "o", mfc="none", color="0.6", label="word missed")
    ax.loglog(ranks[hit], [est[int(v)] for v in vocab[hit].tolist()], "x", color="C3", label="estimated frequency")
    ax.axhline(false_alarm_z(tested[-1], 0.1) * sd, color="0.3", linestyle="--", label="threshold at the last level")
    ax.set(
        xlabel="rank of the word",
        ylabel="fraction of users",
        title=rf"Heavy hitters without a dictionary: $n = 10^6$, $\varepsilon = {eps:g}$, $26^6$ possible strings",
    )
    ax.legend(fontsize=8)
    return fig


def plot_repeated(rng: np.random.Generator) -> Figure:
    fig, (a0, a1) = plt.subplots(1, 2, figsize=(11.5, 4.3))
    days = list(range(1, 61))
    eps = math.log(3)
    a0.semilogy(days, [majority_error(d, eps) for d in days], label="fresh randomized response every day")
    a0.semilogy(days, [1 - rr_keep(eps)] * len(days), label="memoized: the same report every day")
    a0.set(
        xlabel="days of reports",
        ylabel="probability the majority vote is wrong",
        ylim=(1e-6, 1),
        title=r"Guessing a fixed bit, $\varepsilon_0 = \ln 3$ per report",
    )
    a0.legend(fontsize=8)
    m, horizon = 1440.0, 30
    spreads = [2.5, 5.0, 10.0, 15.0, 20.0, 30.0]
    grid = np.linspace(1, 30, 59)
    a1.plot(
        grid,
        [alpha_point_change_theory(horizon, float(s), m, 1.0) for s in grid],
        color="C0",
        label=r"$\alpha$-point rounding, formula",
    )
    a1.plot(
        spreads,
        [alpha_point_changes(rng, 50_000, horizon, s, m, 1.0) for s in spreads],
        "o",
        color="C0",
        label="simulated",
    )
    a1.plot(
        spreads,
        [fresh_changes(rng, 50_000, horizon, s, m, 1.0) for s in spreads],
        "s",
        color="C1",
        label="fresh 1-bit report every day",
    )
    a1.set(
        xlabel="daily wobble of usage (± minutes)",
        ylabel="fraction of users whose report ever changes",
        yscale="log",
        ylim=(1e-3, 2),
        title=rf"{horizon} days of minutes in an app, $\varepsilon = 1$",
    )
    a1.legend(fontsize=8)
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
        "trust_models.svg": plot_models(),
        "trust_rr_gap.svg": plot_rr_gap(rng),
        "trust_frequency_oracles.svg": plot_frequency_oracles(),
        "trust_clones.svg": plot_clones(),
        "trust_shuffle_bound.svg": plot_shuffle_bound(),
        "trust_matched_error.svg": plot_matched_error(rng),
        "trust_collusion.svg": plot_collusion(),
        "trust_local_hashing.svg": plot_local_hashing(),
        "trust_heavy_hitters.svg": plot_heavy_hitters(rng),
        "trust_repeated.svg": plot_repeated(rng),
    }
    for name, fig in figures.items():
        fig.tight_layout()
        fig.savefig(out / name, metadata={"Date": None})  # no timestamp, so output is reproducible
        plt.close(fig)
        print(f"saved {out / name}")


if __name__ == "__main__":
    main()
