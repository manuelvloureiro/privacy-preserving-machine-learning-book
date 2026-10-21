"""Check the lower bounds in the lower-bounds chapter and draw its figures.

    uv run python code/part03-dp-foundations/lower_bounds.py
    uv run python code/part03-dp-foundations/lower_bounds.py --seed 1 --out /tmp/figures

Prints the numbers quoted in 10-lower-bounds.md (they match exactly with the default seed),
then saves the chapter's figures.

Conventions: ε, δ are the privacy parameters, β the failure probability of an accuracy guarantee
and α its error. Histograms use add/remove neighbours (one record more or less, counts move by 1).
The marginals, tracing and reconstruction sections use replace neighbours on datasets of fixed size n,
with records in {-1, +1}^d and released means in [-1, 1]^d.
"""

import argparse
import math
import os
from collections.abc import Callable, Sequence
from pathlib import Path

# Small linear algebra only: BLAS threads cost more than they save, especially on a busy machine.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import matplotlib

matplotlib.use("Agg")
matplotlib.rcParams["svg.hashsalt"] = "lower_bounds"  # stable element ids, so reruns don't churn the SVGs
matplotlib.rcParams["svg.fonttype"] = "none"  # keep text as text: smaller, diffable files

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from numpy.typing import NDArray

# Figures are build output: _build/figures/<part>/, which the build copies into each book.
FIGURES = Path(__file__).resolve().parents[2] / "_build" / "figures" / Path(__file__).parent.name

EPS = 1.0  # the privacy level used in every example
DELTA = 1e-6  # δ for approximate DP
BETA = 0.05  # failure probability of the accuracy guarantees
ALPHA = 0.1  # error of the marginals, as a fraction of the range [-1, 1]

# Tracing experiment: n records, d attributes.
TRACE_N = 50
TRACE_D = 2000
TRACE_TRIALS = 200

# Reconstruction experiment: n secret bits, k = 4n random subset-sum queries.
DN_N = 200
DN_QUERIES = 4 * DN_N
DN_TRIALS = 40


def normal_cdf(x: float) -> float:
    """Φ(x); erfc keeps full relative precision in the lower tail."""
    return 0.5 * math.erfc(-x / math.sqrt(2))


def bisect(f: Callable[[float], float], lo: float, hi: float, steps: int = 200) -> float:
    """A root of f in [lo, hi], assuming f(lo) and f(hi) have opposite signs."""
    f_lo = f(lo)
    for _ in range(steps):
        mid = 0.5 * (lo + hi)
        if (f(mid) > 0) == (f_lo > 0):
            lo, f_lo = mid, f(mid)
        else:
            hi = mid
    return 0.5 * (lo + hi)


def normal_upper_quantile(tail: float) -> float:
    """z with 1 - Φ(z) = tail."""
    return bisect(lambda z: normal_cdf(-z) - tail, 0.0, 40.0)


# --- Toolkit: what DP allows an attacker -----------------------------------------------------


def group_delta(k: int, eps: float, delta: float) -> float:
    """δ of group privacy for k changed records: Σ_{j<k} e^{jε} δ ≤ k e^{(k-1)ε} δ (the sum)."""
    return sum(math.exp(j * eps) for j in range(k)) * delta


def dp_advantage_bound(eps: float, delta: float) -> float:
    """Largest TPR - FPR of any test between M(D) and M(D') under (ε, δ)-DP."""
    return (math.exp(eps) - 1 + 2 * delta) / (math.exp(eps) + 1)


def dp_bit_accuracy_bound(eps: float, delta: float) -> float:
    """Largest probability of guessing a uniform secret bit of one record under (ε, δ)-DP."""
    return (math.exp(eps) + delta) / (math.exp(eps) + 1)


def expectation_shift_bound(mean_other: float, abs_other: float, bound: float, eps: float, delta: float) -> float:
    """Toolkit Fact 2: E h(M(D)) ≤ E h(M(D')) + (e^ε - 1) E|h(M(D'))| + 2δT for |h| ≤ T."""
    return mean_other + (math.exp(eps) - 1) * abs_other + 2 * delta * bound


# --- Packing ---------------------------------------------------------------------------------


def packing_min_k(log_m: float, eps: float, beta: float) -> float:
    """Pure DP: m disjoint good sets at group distance k force k ≥ ln(m(1 - β))/ε."""
    return (log_m + math.log1p(-beta)) / eps


def histogram_error_lower(n_bins: int, eps: float, beta: float) -> float:
    """ℓ∞ error any ε-DP histogram must have with probability β: (ln(N(1 - β))/ε - 1)/2."""
    return max(0.0, (packing_min_k(math.log(n_bins), eps, beta) - 1) / 2)


def laplace_histogram_error(n_bins: int, eps: float, beta: float) -> float:
    """Exact (1 - β)-quantile of max_j |Lap(1/ε)_j| over N bins: -ln(1 - (1 - β)^{1/N})/ε."""
    return -math.log(-math.expm1(math.log1p(-beta) / n_bins)) / eps


def laplace_histogram_error_union(n_bins: int, eps: float, beta: float) -> float:
    """Union bound on the same quantile: ln(N/β)/ε."""
    return math.log(n_bins / beta) / eps


def laplace_histogram_error_sim(rng: np.random.Generator, n_bins: int, eps: float, beta: float, trials: int) -> float:
    noise = rng.laplace(0.0, 1 / eps, size=(trials, n_bins))
    return float(np.quantile(np.abs(noise).max(axis=1), 1 - beta))


def laplace_difference_tail(k: float, eps: float) -> float:
    """Pr[η - η' ≥ k] for independent η, η' ~ Lap(1/ε): e^{-kε}(2 + kε)/4."""
    return math.exp(-k * eps) * (2 + k * eps) / 4


def noisy_max_gap(n_bins: int, eps: float, beta: float) -> float:
    """Count gap k at which report-noisy-max (Laplace 1/ε) fails with probability ≤ β (union bound)."""
    return bisect(lambda k: (n_bins - 1) * laplace_difference_tail(k, eps) - beta, 0.0, 200.0 / eps)


def noisy_max_failure_sim(rng: np.random.Generator, n_bins: int, k: float, eps: float, trials: int) -> float:
    """Simulated probability that report-noisy-max misses the one bin with count k (others 0)."""
    noise = rng.laplace(0.0, 1 / eps, size=(trials, n_bins))
    noise[:, 0] += k
    missed: NDArray[np.bool_] = noise.argmax(axis=1) != 0
    return float(np.mean(missed))


def approx_packing_min_k(log_m: float, eps: float, delta: float, beta: float) -> int:
    """Smallest k with m (e^{-kε}(1 - β) - k e^{-ε} δ) ≤ 1: what packing can force under (ε, δ)-DP."""
    k = 1
    while True:
        floor = math.exp(-k * eps) * (1 - beta) - k * math.exp(-eps) * delta
        if floor <= 0 or log_m + math.log(floor) <= 0:
            return k
        k += 1


def approx_packing_lemma_k(log_m: float, eps: float, delta: float, beta: float) -> int:
    """Smallest k allowed by the (ε, δ) packing lemma: e^{-kε}(1 - β) ≤ 2kδ or m e^{-kε}(1 - β) ≤ 2."""
    k = 1
    while True:
        share = math.exp(-k * eps) * (1 - beta)
        if share <= 2 * k * delta or log_m + math.log(share / 2) <= 0:
            return k
        k += 1


def mean_packing_candidates(r: float, alpha: float) -> int:
    """Means spaced 2R/m ≥ 2α apart in [-R, R]: m = ⌊R/α⌋ disjoint open balls of radius α."""
    return math.floor(r / alpha)


def mean_packing_min_n(r: float, alpha: float, eps: float, beta: float) -> float:
    """Pure DP: n ≥ ln(⌊R/α⌋(1 - β))/ε, since any two datasets of size n are within n replacements."""
    return packing_min_k(math.log(mean_packing_candidates(r, alpha)), eps, beta)


# --- Sample complexity -----------------------------------------------------------------------


def histogram_n_private(n_bins: int, eps: float, alpha: float, beta: float) -> float:
    """Records needed for an ε-DP histogram with ℓ∞ error αn: (ln(N(1 - β)) - ε)/(2αε)."""
    return (math.log(n_bins) + math.log1p(-beta) - eps) / (2 * alpha * eps)


def histogram_n_sampling(n_bins: int, alpha: float, beta: float) -> float:
    """Samples for every empirical frequency to be within α of the truth (Hoeffding + union)."""
    return math.log(2 * n_bins / beta) / (2 * alpha**2)


def marginals_pure_lower(d: int, eps: float, alpha: float, beta: float) -> float:
    """Packing with the 2^d - 1 corner datasets: n ≥ (ln((2^d - 1)(1 - β))/ε - 1)/α."""
    log_m = d * math.log(2) + math.log1p(-(2.0**-d))
    return (packing_min_k(log_m, eps, beta) - 1) / alpha


def marginals_laplace_upper(d: int, eps: float, alpha: float, beta: float) -> float:
    """Laplace with Δ₁ = 2d/n, union bound over d coordinates: n = 2d ln(d/β)/(αε)."""
    return 2 * d * math.log(d / beta) / (alpha * eps)


def gaussian_delta(eps: float, mu: float) -> float:
    """Exact δ(ε) of the Gaussian mechanism with sensitivity/σ = μ (Balle & Wang 2018)."""
    tail = normal_cdf(-mu / 2 - eps / mu)
    weighted = 0.0 if tail == 0 else math.exp(min(eps + math.log(tail), 700.0))  # e^ε Φ(·), overflow-safe
    return normal_cdf(mu / 2 - eps / mu) - weighted


def gaussian_mu(eps: float, delta: float) -> float:
    """The largest sensitivity/σ ratio that is (ε, δ)-DP."""
    return bisect(lambda mu: gaussian_delta(eps, mu) - delta, 1e-6, 100.0)


def gaussian_eps(mu: float, delta: float) -> float:
    """The smallest ε for which the Gaussian mechanism with ratio μ is (ε, δ)-DP."""
    if gaussian_delta(0.0, mu) <= delta:
        return 0.0
    return bisect(lambda e: gaussian_delta(e, mu) - delta, 0.0, mu * mu + 50 * mu + 50)


def marginals_gaussian_upper(d: int, eps: float, delta: float, alpha: float, beta: float) -> float:
    """Gaussian with Δ₂ = 2√d/n and exact calibration, union bound: n = 2√d z/(μ α)."""
    z = normal_upper_quantile(beta / (2 * d))
    return 2 * math.sqrt(d) * z / (gaussian_mu(eps, delta) * alpha)


def fingerprint_lower_constant(alpha: float, beta: float) -> float:
    """n ≥ c d/((e^ε - 1)√d + 4δd) at accuracy α: c = 2/3 - 2α - 4β (no padding)."""
    return 2 / 3 - 2 * alpha - 4 * beta


def fingerprint_best_alpha0(alpha: float, beta: float) -> float:
    """Accuracy α₀ ≥ α of the padded small dataset maximizing α₀(2/3 - 2α₀ - 4β): 1/6 - β."""
    return max(alpha, 1 / 6 - beta)


def marginals_fingerprint_lower(d: int, eps: float, delta: float, alpha: float, beta: float) -> float:
    """Fingerprinting with padding: n ≥ (α₀/α) (2/3 - 2α₀ - 4β) d/((e^ε - 1)√d + 4δd)."""
    a0 = fingerprint_best_alpha0(alpha, beta)
    c = fingerprint_lower_constant(a0, beta)
    return (a0 / alpha) * c * d / ((math.exp(eps) - 1) * math.sqrt(d) + 4 * delta * d)


def crossover_dimension(eps: float, delta: float, alpha: float, beta: float) -> int:
    """Smallest d at which the Gaussian upper bound beats the pure-DP packing lower bound."""

    def gaussian_wins(d: int) -> bool:
        return marginals_gaussian_upper(d, eps, delta, alpha, beta) < marginals_pure_lower(d, eps, alpha, beta)

    lo, hi = 1, 2  # the Gaussian loses at lo and wins at hi; its advantage grows with d
    while not gaussian_wins(hi):
        lo, hi = hi, 2 * hi
    while hi - lo > 1:
        mid = (lo + hi) // 2
        lo, hi = (lo, mid) if gaussian_wins(mid) else (mid, hi)
    return hi


# --- The fingerprinting lemma, computed exactly for small n ---------------------------------


def all_columns(n: int) -> NDArray[np.float64]:
    """Every column x ∈ {-1, +1}^n, one per row (2^n rows)."""
    grid = np.array(np.meshgrid(*([[-1.0, 1.0]] * n), indexing="ij"))
    return grid.reshape(n, -1).T.astype(np.float64)


def fingerprint_terms(
    f: Callable[[NDArray[np.float64]], NDArray[np.float64]], n: int, nodes: int = 64
) -> tuple[float, float, float]:
    """For p ~ U[-1, 1] and x_i i.i.d. ±1 with mean p: (E[f Σ(x_i - p)], 2E[p f], E|f - x̄|).

    Exact up to Gauss–Legendre quadrature in p (degree n + 1 polynomials, so 64 nodes are exact
    for the first two terms).
    """
    xs = all_columns(n)
    fx = f(xs)
    sums = xs.sum(axis=1)
    mean = sums / n
    ps, ws = np.polynomial.legendre.leggauss(nodes)
    ws = ws / 2  # uniform density on [-1, 1]
    corr = two_pf = dev = 0.0
    for p, w in zip(ps, ws, strict=True):
        prob = np.prod((1 + xs * p) / 2, axis=1)
        corr += float(w * np.sum(prob * fx * (sums - n * p)))
        two_pf += float(w * 2 * p * np.sum(prob * fx))
        dev += float(w * np.sum(prob * np.abs(fx - mean)))
    return corr, two_pf, dev


def expected_estimate(
    f: Callable[[NDArray[np.float64]], NDArray[np.float64]], n: int, ps: NDArray[np.float64]
) -> NDArray[np.float64]:
    """g(p) = E[f(x) | p] on a grid of p."""
    xs = all_columns(n)
    fx = f(xs)
    values: list[float] = []
    for p in ps:
        prob: NDArray[np.float64] = np.prod((1 + xs * float(p)) / 2, axis=1)
        values.append(float(np.sum(prob * fx)))
    return np.array(values)


def column_mean(xs: NDArray[np.float64]) -> NDArray[np.float64]:
    return xs.mean(axis=1)


def shrunk_mean(xs: NDArray[np.float64]) -> NDArray[np.float64]:
    return 0.5 * xs.mean(axis=1)


def constant_zero(xs: NDArray[np.float64]) -> NDArray[np.float64]:
    return np.zeros(xs.shape[0])


# --- Tracing attack on noisy means -----------------------------------------------------------


def tracing_scores(
    rng: np.random.Generator, n: int, d: int, trials: int, sigmas: Sequence[float]
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Scores ⟨q̂ - 0, y - p⟩ of members and of fresh non-members, one row per noise level σ.

    Each trial draws attribute means p ~ U[-1, 1]^d, n members and n non-members from the product
    distribution, and one standard normal noise vector η; the release is q̂ = x̄ + σ η.
    """
    members = np.empty((len(sigmas), trials * n))
    others = np.empty((len(sigmas), trials * n))
    for t in range(trials):
        p = rng.uniform(-1.0, 1.0, size=d)
        x = np.where(rng.random((n, d)) < (1 + p) / 2, 1.0, -1.0)
        y = np.where(rng.random((n, d)) < (1 + p) / 2, 1.0, -1.0)
        eta = rng.standard_normal(d)
        mean = x.mean(axis=0)
        for s, sigma in enumerate(sigmas):
            release = mean + sigma * eta
            members[s, t * n : (t + 1) * n] = (x - p) @ release
            others[s, t * n : (t + 1) * n] = (y - p) @ release
    return members, others


def best_advantage(members: NDArray[np.float64], others: NDArray[np.float64]) -> float:
    """max over thresholds of TPR - FPR for the test "score ≥ threshold → member"."""
    scores = np.concatenate([members, others])
    labels = np.concatenate([np.ones(members.size), np.zeros(others.size)])
    order = np.argsort(-scores, kind="stable")
    tpr = np.cumsum(labels[order]) / members.size
    fpr = np.cumsum(1 - labels[order]) / others.size
    return float(np.max(tpr - fpr))


def tpr_at_fpr(members: NDArray[np.float64], others: NDArray[np.float64], fpr: float) -> float:
    threshold = float(np.quantile(others, 1 - fpr))
    return float(np.mean(members > threshold))


def tracing_mean_sd(n: int, d: int, sigma: float) -> tuple[float, float]:
    """Mean gap and non-member standard deviation of the score: 2d/(3n), √(d(2/15 + 8/(15n) + 2σ²/3))."""
    return 2 * d / (3 * n), math.sqrt(d * (2 / 15 + 8 / (15 * n) + 2 * sigma**2 / 3))


def tracing_predicted_advantage(n: int, d: int, sigma: float) -> float:
    """2Φ(m/(2s)) - 1 for two normals with means 0 and m and common standard deviation s."""
    m, s = tracing_mean_sd(n, d, sigma)
    return 2 * normal_cdf(m / (2 * s)) - 1


def gaussian_worst_tv(n: int, d: int, sigma: float) -> float:
    """TV between the releases on neighbouring datasets at worst: 2Φ(Δ₂/(2σ)) - 1, Δ₂ = 2√d/n."""
    if sigma == 0:
        return 1.0
    return 2 * normal_cdf(math.sqrt(d) / (n * sigma)) - 1


def tracing_sigma_for(n: int, d: int, eps: float, delta: float) -> float:
    """σ of the (ε, δ)-DP Gaussian mechanism on the means: Δ₂/μ with Δ₂ = 2√d/n."""
    return 2 * math.sqrt(d) / n / gaussian_mu(eps, delta)


# --- Dinur–Nissim as a lower bound ----------------------------------------------------------


def dn_recovered_fraction(rng: np.random.Generator, n: int, k: int, sigma: float, trials: int) -> float:
    """Least-squares reconstruction from k noisy random subset sums with Gaussian noise σ."""
    correct = 0.0
    for _ in range(trials):
        bits = rng.integers(0, 2, size=n).astype(np.float64)
        queries = rng.integers(0, 2, size=(k, n)).astype(np.float64)
        answers = queries @ bits + sigma * rng.standard_normal(k)
        estimate = np.linalg.solve(queries.T @ queries, queries.T @ answers)  # least squares, normal equations
        right: NDArray[np.bool_] = (estimate > 0.5) == (bits > 0.5)
        correct += float(np.mean(right))
    return correct / trials


def dn_eps_lower(fraction: float, delta: float) -> float:
    """A reconstruction recovering this fraction of uniform bits needs ε ≥ ln((r - δ)/(1 - r))."""
    if fraction >= 1:
        return math.inf
    return math.log(max(fraction - delta, 1e-300) / (1 - fraction))


def dn_eps_upper(k: int, sigma: float, delta: float) -> float:
    """ε of the Gaussian mechanism on k subset sums (Δ₂ ≤ √k) at this δ."""
    return gaussian_eps(math.sqrt(k) / sigma, delta)


# --- The chapter's numbers ------------------------------------------------------------------


def print_table(rng: np.random.Generator) -> None:
    print(f"[Toolkit, eps = {EPS:g}, delta = {DELTA:g}]")
    print(f"  advantage bound={dp_advantage_bound(EPS, 0):.4f}  bit accuracy bound={dp_bit_accuracy_bound(EPS, 0):.4f}")
    print(f"  group delta: k=5: {group_delta(5, EPS, DELTA):.3e}  k=20: {group_delta(20, EPS, DELTA):.3e}")

    print(f"[Packing: mode and histogram, eps = {EPS:g}, beta = {BETA:g}]")
    for n_bins in (10**3, 10**6, 10**9):
        print(
            f"  N={n_bins:.0e}: mode k >= {packing_min_k(math.log(n_bins), EPS, BETA):.2f}  "
            f"noisy-max gap {noisy_max_gap(n_bins, EPS, BETA):.2f}  "
            f"histogram lower {histogram_error_lower(n_bins, EPS, BETA):.2f}  "
            f"Laplace exact {laplace_histogram_error(n_bins, EPS, BETA):.2f}  "
            f"union {laplace_histogram_error_union(n_bins, EPS, BETA):.2f}"
        )
    sim = laplace_histogram_error_sim(rng, 1000, EPS, BETA, 20_000)
    print(f"  N=1e+03 simulated Laplace 95% error {sim:.2f}")
    gap = noisy_max_gap(1000, EPS, BETA)
    print(
        f"  N=1e+03 noisy-max failure at gap {gap:.2f}: simulated {noisy_max_failure_sim(rng, 1000, gap, EPS, 20_000):.4f}"
    )

    print("[Packing under approximate DP: m = 2^d candidates]")
    for d in (10, 100, 1000):
        print(
            f"  d={d}: pure k >= {packing_min_k(d * math.log(2), EPS, BETA):.2f}  "
            f"approx (delta={DELTA:g}) forces k >= {approx_packing_min_k(d * math.log(2), EPS, DELTA, BETA)}"
        )
    print(f"  ln(1/delta)/eps = {math.log(1 / DELTA) / EPS:.2f}")
    for d in (10, 100, 1000):
        print(
            f"  d={d}: (eps, delta) packing lemma forces k >= {approx_packing_lemma_k(d * math.log(2), EPS, DELTA, BETA)}"
        )
    print(f"[Packing for a mean: |mu| <= R, error < alpha={ALPHA:g}, eps = {EPS:g}, beta = {BETA:g}]")
    for r in (10.0, 1e3, 1e6):
        m = mean_packing_candidates(r, ALPHA)
        print(
            f"  R={r:.0e}: m={m}  pure n >= {mean_packing_min_n(r, ALPHA, EPS, BETA):.2f}  "
            f"(eps, delta={DELTA:g}) n >= {approx_packing_lemma_k(math.log(m), EPS, DELTA, BETA)}"
        )

    print(f"[Sample complexity, eps = {EPS:g}, beta = {BETA:g}]")
    n_bins = 10**6
    for alpha in (0.1, 0.01):
        print(
            f"  histogram N=1e6, alpha={alpha:g}: private n >= {histogram_n_private(n_bins, EPS, alpha, BETA):.0f}  "
            f"sampling n = {histogram_n_sampling(n_bins, alpha, BETA):.0f}"
        )
    print(f"  marginals, alpha={ALPHA:g}, delta={DELTA:g}:")
    for d in (100, 1000, 10_000, 100_000):
        print(
            f"    d={d}: pure lower {marginals_pure_lower(d, EPS, ALPHA, BETA):.0f}  "
            f"Laplace upper {marginals_laplace_upper(d, EPS, ALPHA, BETA):.0f}  "
            f"fingerprint lower {marginals_fingerprint_lower(d, EPS, DELTA, ALPHA, BETA):.1f}  "
            f"Gaussian upper {marginals_gaussian_upper(d, EPS, DELTA, ALPHA, BETA):.0f}"
        )
    print(f"  Gaussian upper beats pure lower from d = {crossover_dimension(EPS, DELTA, ALPHA, BETA)}")
    print(f"  Gaussian mu(eps=1, delta=1e-6) = {gaussian_mu(EPS, DELTA):.4f}")
    print(
        f"  fingerprint constant at alpha={ALPHA:g}: {fingerprint_lower_constant(ALPHA, BETA):.4f}  "
        f"best alpha0={fingerprint_best_alpha0(ALPHA, BETA):.4f}  "
        f"padded constant={fingerprint_best_alpha0(ALPHA, BETA) * fingerprint_lower_constant(fingerprint_best_alpha0(ALPHA, BETA), BETA):.4f}"
    )

    print("[Fingerprinting lemma, n = 5, exact]")
    for name, f in (("mean", column_mean), ("half mean", shrunk_mean), ("zero", constant_zero)):
        corr, two_pf, dev = fingerprint_terms(f, 5)
        print(
            f"  {name}: E[f sum(x-p)]={corr:.4f}  2E[pf]={two_pf:.4f}  E|f-mean|={dev:.4f}  "
            f"bound 2/3-2E|f-mean|={2 / 3 - 2 * dev:.4f}"
        )

    n, d = TRACE_N, TRACE_D
    sigma_dp = tracing_sigma_for(n, d, EPS, 1e-5)
    sigmas = [0.0, 1.0, sigma_dp]
    members, others = tracing_scores(rng, n, d, TRACE_TRIALS, sigmas)
    print(f"[Tracing: n = {n}, d = {d}, {TRACE_TRIALS} datasets]")
    m, s = tracing_mean_sd(n, d, 0.0)
    print(
        f"  sigma=0: predicted gap {m:.2f}, sd {s:.2f}; simulated member mean {members[0].mean():.2f}  "
        f"non-member mean {others[0].mean():.2f}  sd {others[0].std():.2f}"
    )
    for i, sigma in enumerate(sigmas):
        print(
            f"  sigma={sigma:.2f}: advantage {best_advantage(members[i], others[i]):.3f} "
            f"(predicted {tracing_predicted_advantage(n, d, sigma):.3f}, worst-case TV {gaussian_worst_tv(n, d, sigma):.3f})  "
            f"TPR at 1% FPR {tpr_at_fpr(members[i], others[i], 0.01):.3f}"
        )
    print(
        f"  Gaussian sigma for (1, 1e-5)-DP: {sigma_dp:.2f}; DP advantage bound {dp_advantage_bound(EPS, 1e-5):.3f}; "
        f"TPR bound at 1% FPR {math.exp(EPS) * 0.01 + 1e-5:.4f}"
    )

    print(f"[Dinur-Nissim: n = {DN_N}, k = {DN_QUERIES} queries, delta = {DELTA:g}]")
    for c in (0.1, 0.25, 0.5, 1.0, 2.0):
        sigma = c * math.sqrt(DN_N)
        r = dn_recovered_fraction(rng, DN_N, DN_QUERIES, sigma, DN_TRIALS)
        print(
            f"  sigma={c:g} sqrt(n): recovered {r:.4f}  eps >= {dn_eps_lower(r, DELTA):.2f}  "
            f"Gaussian eps <= {dn_eps_upper(DN_QUERIES, sigma, DELTA):.2f}"
        )


# --- Figures --------------------------------------------------------------------------------


def plot_packing_budget(eps: float = EPS, beta: float = BETA) -> Figure:
    ks = np.linspace(0, 25, 501)
    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.plot(
        ks,
        (1 - beta) * np.exp(-eps * ks),
        "k",
        linewidth=2,
        label=r"mass DP forces into each good set, $(1-\beta)e^{-k\varepsilon}$",
    )
    for i, m in enumerate((10, 10**3, 10**6, 10**9)):
        k_min = packing_min_k(math.log(m), eps, beta)
        ax.axhline(1 / m, color=f"C{i}", linestyle="--", label=f"mass available per set, 1/m, m = {m:.0e}")
        ax.plot([k_min], [1 / m], "o", color=f"C{i}")
    ax.set(
        yscale="log",
        ylim=(1e-11, 2),
        xlim=(0, 25),
        xlabel="group distance k between $D_0$ and each $D_i$",
        ylabel="probability",
        title=f"Packing: the curve must stay below 1/m (ε = {eps:g}); dots mark the smallest k",
    )
    ax.legend(fontsize=8, loc="lower left")
    return fig


def plot_histogram(rng: np.random.Generator, eps: float = EPS, beta: float = BETA) -> Figure:
    exps = np.linspace(1, 12, 111)
    sizes = [int(round(10**e)) for e in exps]
    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.plot(
        sizes,
        [laplace_histogram_error_union(s, eps, beta) for s in sizes],
        "C1:",
        label=r"Laplace, union bound $\ln(N/\beta)/\varepsilon$",
    )
    ax.plot(sizes, [laplace_histogram_error(s, eps, beta) for s in sizes], "C1", label="Laplace, exact 95% quantile")
    sim_sizes = [10, 100, 1000, 10000]
    ax.plot(
        sim_sizes,
        [laplace_histogram_error_sim(rng, s, eps, beta, 4000) for s in sim_sizes],
        "C1o",
        label="Laplace, simulated",
    )
    ax.plot(
        sizes,
        [histogram_error_lower(s, eps, beta) for s in sizes],
        "C0",
        linewidth=2,
        label=r"packing lower bound $(\ln(N(1-\beta))/\varepsilon - 1)/2$",
    )
    ax.fill_between(sizes, 0, [histogram_error_lower(s, eps, beta) for s in sizes], color="C0", alpha=0.15)
    ax.set(
        xscale="log",
        xlabel="universe size N (number of bins)",
        ylabel=r"$\ell_\infty$ error (counts)",
        ylim=(0, 30),
        title=f"ε-DP histograms, ε = {eps:g}: no mechanism can enter the shaded region",
    )
    ax.legend(fontsize=8, loc="upper left")
    return fig


def plot_fingerprinting(n: int = 5) -> Figure:
    ps = np.linspace(-1, 1, 201)
    fig, (left, right) = plt.subplots(1, 2, figsize=(11, 4.2))
    estimators: list[tuple[str, Callable[[NDArray[np.float64]], NDArray[np.float64]]]] = [
        ("f = mean of the column", column_mean),
        ("f = half the mean", shrunk_mean),
        ("f = 0", constant_zero),
    ]
    for i, (name, f) in enumerate(estimators):
        g = expected_estimate(f, n, ps)
        corr, _, _ = fingerprint_terms(f, n)
        left.plot(ps, g, f"C{i}", label=name)
        slope = np.gradient(g, ps)
        right.plot(ps, (1 - ps**2) * slope, f"C{i}", label=rf"{name}: average {corr:.3f}")
    left.plot(ps, ps, "k:", label="g(p) = p (perfect accuracy)")
    left.set(
        xlabel="attribute mean p",
        ylabel=r"$g(p) = \mathrm{E}[f(x)\,|\,p]$",
        title=f"An accurate estimate must track p (n = {n})",
    )
    left.legend(fontsize=8, loc="upper left")
    right.set(
        xlabel="attribute mean p",
        ylabel="E[f(x) Σ(xᵢ − p) | p] = (1 − p²) · slope of g",
        title="…so it must correlate with the records",
    )
    right.legend(fontsize=8, loc="lower center")
    return fig


def plot_sample_complexity(eps: float = EPS, delta: float = DELTA, alpha: float = ALPHA, beta: float = BETA) -> Figure:
    ds = [int(round(10 ** (e / 10))) for e in range(10, 61)]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(ds, [marginals_laplace_upper(d, eps, alpha, beta) for d in ds], "C1", label="pure DP, Laplace (upper)")
    ax.plot(ds, [marginals_pure_lower(d, eps, alpha, beta) for d in ds], "C1--", label="pure DP, packing (lower)")
    ax.plot(
        ds,
        [marginals_gaussian_upper(d, eps, delta, alpha, beta) for d in ds],
        "C0",
        label="approximate DP, Gaussian (upper)",
    )
    ax.plot(
        ds,
        [marginals_fingerprint_lower(d, eps, delta, alpha, beta) for d in ds],
        "C0--",
        label="approximate DP, fingerprinting (lower)",
    )
    ax.set(
        xscale="log",
        yscale="log",
        xlabel="number of attributes d",
        ylabel="records n needed",
        title=f"d one-way marginals to error {alpha:g}: slope 1 (pure) against ½ (approximate)",
    )
    ax.legend(fontsize=8, loc="upper left")
    return fig


def plot_tracing_scores(rng: np.random.Generator, n: int = TRACE_N, d: int = TRACE_D) -> Figure:
    sigmas = [0.0, tracing_sigma_for(n, d, EPS, 1e-5)]
    members, others = tracing_scores(rng, n, d, 100, sigmas)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, sigma, mem, oth in zip(axes, sigmas, members, others, strict=True):
        bins: NDArray[np.float64] = np.linspace(
            min(float(mem.min()), float(oth.min())), max(float(mem.max()), float(oth.max())), 60
        )
        ax.hist(oth, bins=bins, density=True, alpha=0.5, color="C1", label="non-members")
        ax.hist(mem, bins=bins, density=True, alpha=0.5, color="C0", label="members")
        ax.set(
            xlabel="score: released means · (y − p)",
            ylabel="density",
            title=f"σ = {sigma:.2f}: advantage {best_advantage(mem, oth):.2f}",
        )
        ax.legend(fontsize=8)
    fig.suptitle(f"Tracing scores, n = {n} records, d = {d} attributes; right: Gaussian noise for (1, 1e-5)-DP")
    return fig


def plot_tracing_advantage(rng: np.random.Generator, n: int = TRACE_N, d: int = TRACE_D) -> Figure:
    sigmas = [float(10 ** (e / 4)) for e in range(-12, 7)]
    members, others = tracing_scores(rng, n, d, 60, sigmas)
    grid = [float(10 ** (e / 40)) for e in range(-120, 61)]
    sigma_dp = tracing_sigma_for(n, d, EPS, 1e-5)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(grid, [gaussian_worst_tv(n, d, s) for s in grid], "k--", label="worst-case TV between neighbours")
    ax.plot(
        grid, [tracing_predicted_advantage(n, d, s) for s in grid], "C0", label="tracing attack, normal approximation"
    )
    ax.plot(
        sigmas,
        [best_advantage(members[i], others[i]) for i in range(len(sigmas))],
        "C0o",
        label="tracing attack, simulated",
    )
    ax.axvline(sigma_dp, color="C3", linestyle=":", label=f"σ of the (1, 1e-5)-DP Gaussian mechanism = {sigma_dp:.1f}")
    ax.axvline(math.sqrt(d) / n, color="0.5", linestyle="-.", label="σ = √d / n")
    ax.set(
        xscale="log",
        xlabel="noise standard deviation σ per released mean",
        ylabel="advantage TPR − FPR",
        ylim=(0, 1.02),
        title=f"Tracing advantage against noise, n = {n}, d = {d}",
    )
    ax.legend(fontsize=8, loc="lower left")
    return fig


def plot_dinur_nissim(rng: np.random.Generator, n: int = DN_N, k: int = DN_QUERIES, delta: float = DELTA) -> Figure:
    cs = [float(10 ** (e / 8)) for e in range(-10, 4)]
    fractions = [dn_recovered_fraction(rng, n, k, c * math.sqrt(n), 10) for c in cs]
    fig, (left, right) = plt.subplots(1, 2, figsize=(11, 4.2))
    left.plot(cs, fractions, "C0o-", label="least-squares reconstruction")
    left.axhline(
        dp_bit_accuracy_bound(1.0, delta), color="C3", linestyle=":", label="largest fraction allowed by ε = 1"
    )
    left.axhline(0.5, color="0.5", linewidth=0.8)
    left.set(
        xscale="log",
        ylim=(0.45, 1.02),
        xlabel=r"noise $\sigma/\sqrt{n}$",
        ylabel="fraction of bits recovered",
        title=f"Reconstruction from {k} subset sums of n = {n} bits",
    )
    left.legend(fontsize=8, loc="lower left")
    right.plot(cs, [dn_eps_lower(r, delta) for r in fractions], "C0o-", label="lower bound implied by the attack")
    right.plot(cs, [dn_eps_upper(k, c * math.sqrt(n), delta) for c in cs], "C1", label="Gaussian mechanism (upper)")
    right.set(
        xscale="log",
        yscale="log",
        xlabel=r"noise $\sigma/\sqrt{n}$",
        ylabel=f"ε at δ = {delta:g}",
        title="The true ε of the release lies between the curves",
    )
    right.legend(fontsize=8, loc="upper right")
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
        "lower_packing_budget.svg": plot_packing_budget(),
        "lower_histogram.svg": plot_histogram(rng),
        "lower_sample_complexity.svg": plot_sample_complexity(),
        "lower_fingerprinting.svg": plot_fingerprinting(),
        "lower_tracing_scores.svg": plot_tracing_scores(rng),
        "lower_tracing_advantage.svg": plot_tracing_advantage(rng),
        "lower_dinur_nissim.svg": plot_dinur_nissim(rng),
    }
    for name, fig in figures.items():
        fig.tight_layout()
        fig.savefig(out / name, metadata={"Date": None})  # no timestamp, so output is reproducible
        plt.close(fig)
        print(f"saved {out / name}")


if __name__ == "__main__":
    main()
