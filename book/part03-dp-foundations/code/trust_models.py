"""Compare the central, local, shuffle and distributed trust models, and draw the chapter's figures.

    uv run python book/part03-dp-foundations/code/trust_models.py
    uv run python book/part03-dp-foundations/code/trust_models.py --seed 1 --out /tmp/figures

Prints the numbers quoted in 10-trust-models.md (they match exactly with the default seed),
then saves the chapter's figures.

Conventions: neighbouring datasets differ by replacing one record, in every model, because a
local randomizer must protect each user's value against every other value. ε0 is the local
budget of one report, (ε, δ) the central guarantee. Frequencies are fractions in [0, 1], and an
estimator's error is measured on fractions, so a count error is n times larger.
"""

import argparse
import math
from collections.abc import Callable, Sequence
from functools import cache
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
matplotlib.rcParams["svg.hashsalt"] = "trust_models"  # stable element ids, so reruns don't churn the SVGs
matplotlib.rcParams["svg.fonttype"] = "none"  # keep text as text: smaller, diffable files

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.patches import FancyBboxPatch
from numpy.typing import NDArray

FIGURES = Path(__file__).resolve().parent.parent / "figures"

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]


# --- Shared helpers ---------------------------------------------------------------------------


def normal_cdf(x: float) -> float:
    """Φ(x); erfc keeps full relative precision in the lower tail."""
    return 0.5 * math.erfc(-x / math.sqrt(2))


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
    }
    for name, fig in figures.items():
        fig.tight_layout()
        fig.savefig(out / name, metadata={"Date": None})  # no timestamp, so output is reproducible
        plt.close(fig)
        print(f"saved {out / name}")


if __name__ == "__main__":
    main()
