"""Check the numbers in the "Beyond global sensitivity" chapter and draw its figures.

    uv run python code/part03-dp-foundations/sensitivity.py
    uv run python code/part03-dp-foundations/sensitivity.py --seed 1 --out /tmp/figures

Prints the numbers quoted in 07-sensitivity.md (they match exactly with the default seed),
then saves the chapter's figures.

Conventions: datasets have a fixed size n (the replace-one-record neighbouring relation), the
median is taken for odd n, values lie in [0, Λ], and a sorted dataset is padded with x_i = 0
for i <= 0 and x_i = Λ for i > n (Nissim, Raskhodnikova & Smith 2007). Indices are 1-based in
the docstrings, as in the text.
"""

import argparse
import itertools
import math
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
matplotlib.rcParams["svg.hashsalt"] = "sensitivity"  # stable element ids, so reruns don't churn the SVGs
matplotlib.rcParams["svg.fonttype"] = "none"  # keep text as text: smaller, diffable files

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from numpy.typing import NDArray

# Figures are build output: _build/figures/<part>/, which the build copies into each book.
FIGURES = Path(__file__).resolve().parents[2] / "_build" / "figures" / Path(__file__).parent.name

Array = NDArray[np.float64]

# The running examples.
AGE_MAX = 100.0  # Λ for the ages data
EPS = 1.0
DELTA = 1e-6
LEAK_X = (10.0, 20.0, 30.0, 31.0, 90.0)  # the leak example; LEAK_X2 replaces 31 by 90
LEAK_X2 = (10.0, 20.0, 30.0, 90.0, 90.0)
LEAK_MAX = 100.0
WALK_X = (1, 3, 4, 6, 9)  # the walkthrough of the A^(k) formula, on the grid {0, ..., 10}
WALK_MAX = 10
WALK_BETA = 0.5


def normal_cdf(x: float) -> float:
    """Φ(x); erfc keeps full relative precision in the lower tail."""
    return 0.5 * math.erfc(-x / math.sqrt(2))


# --- Local sensitivity of the median --------------------------------------------------------


def padded(x: Array | tuple[float, ...], lam: float) -> Array:
    """Sorted data with n + 1 zeros in front and n + 1 copies of Λ behind.

    Entry i + n of the result is x_i in the 1-based, padded notation of the text, for
    every i from -n to 2n + 1.
    """
    xs = np.sort(np.asarray(x, dtype=np.float64))
    n = len(xs)
    return np.concatenate([np.zeros(n + 1), xs, np.full(n + 1, lam)])


def median_local_sensitivity(x: Array | tuple[float, ...], lam: float) -> float:
    """LS(x) = max(x_m - x_{m-1}, x_{m+1} - x_m) with m = (n + 1)/2."""
    p = padded(x, lam)
    n = (len(p) - 2) // 3
    m = (n + 1) // 2
    xm, lo, hi = p[m + n], p[m - 1 + n], p[m + 1 + n]
    return float(max(xm - lo, hi - xm))


def order_stat_a_profile(x: Array | tuple[float, ...], lam: float, r: int) -> Array:
    """A^(k)(x) = max_{0 <= t <= k+1} (x_{r+t} - x_{r+t-k-1}) for k = 0, ..., n.

    The largest local sensitivity of the r-th order statistic over all datasets within
    Hamming distance k of x. r = (n + 1)/2 gives the median (Proposition 3.4 of NRS 2007).
    An (n + 1) × (n + 2) table of gaps over (k, t): the O(n²) computation.
    """
    p = padded(x, lam)
    n = (len(p) - 2) // 3
    k = np.arange(n + 1)[:, None]
    t = np.arange(n + 2)[None, :]
    lo = np.clip(r + t - k - 1 + n, 0, None)  # entries with t > k + 1 are masked below
    gaps = np.where(t <= k + 1, p[r + t + n] - p[lo], -np.inf)
    return np.max(gaps, axis=1)


def median_a_profile(x: Array | tuple[float, ...], lam: float) -> Array:
    n = len(x)
    return order_stat_a_profile(x, lam, (n + 1) // 2)


def smooth_from_profile(a: Array, beta: float) -> tuple[float, int]:
    """S*_β = max_k e^{-kβ} A^(k), and the k that attains it."""
    weighted = np.exp(-beta * np.arange(len(a))) * a
    k = int(np.argmax(weighted))
    return float(weighted[k]), k


def median_smooth_sensitivity(x: Array | tuple[float, ...], lam: float, beta: float) -> float:
    return smooth_from_profile(median_a_profile(x, lam), beta)[0]


def brute_force_median_profile(x: tuple[int, ...], lam: int) -> Array:
    """max LS(y) over all y in {0, ..., Λ}^n with d(x, y) <= k, for k = 0, ..., n, by enumeration."""
    n = len(x)
    grid = np.array(list(itertools.product(range(lam + 1), repeat=n)), dtype=np.int64)
    differs: NDArray[np.bool_] = grid != np.array(x)
    dist = differs.sum(axis=1)
    ys = np.sort(grid, axis=1)
    m = (n + 1) // 2 - 1  # 0-based median index
    ls = np.maximum(ys[:, m] - ys[:, m - 1], ys[:, m + 1] - ys[:, m]).astype(np.float64)
    return np.array([float(np.max(ls[dist <= k])) for k in range(n + 1)])


# --- Why local sensitivity leaks: two Laplace releases of the same median ------------------


def laplace_log_density(y: Array, mu: float, b: float) -> Array:
    return -np.abs(y - mu) / b - math.log(2 * b)


def local_noise_privacy_loss(y: Array, mu: float, b: float, mu2: float, b2: float) -> Array:
    """L(y) = ln p(y)/p'(y) for p = Lap(mu, b) and p' = Lap(mu2, b2)."""
    return laplace_log_density(y, mu, b) - laplace_log_density(y, mu2, b2)


def laplace_hockey_stick(eps: float, mu: float, b: float, mu2: float, b2: float) -> float:
    """δ(ε) = ∫ (p - e^ε p')_+ for p = Lap(mu, b), p' = Lap(mu2, b2), by quadrature on a wide grid."""
    width = 60 * max(b, b2)
    y = np.linspace(min(mu, mu2) - width, max(mu, mu2) + width, 2_000_001)
    p = np.exp(laplace_log_density(y, mu, b))
    q = np.exp(laplace_log_density(y, mu2, b2))
    return float(np.sum(np.maximum(p - math.exp(eps) * q, 0.0)) * (y[1] - y[0]))


# --- Smooth sensitivity: admissible noise ---------------------------------------------------


def cauchy_params(eps: float, gamma: float = 2.0) -> tuple[float, float]:
    """(α, β) = (ε/(2(γ+1)), ε/(2(γ+1))), Lemma 2.7 of NRS 2007, for h(z) ∝ 1/(1 + |z|^γ)."""
    return eps / (2 * (gamma + 1)), eps / (2 * (gamma + 1))


def cauchy_loss_bound(alpha: float, beta: float, gamma: float = 2.0) -> float:
    """The worst-case privacy loss γα + (γ + 1)β derived in the text."""
    return gamma * alpha + (gamma + 1) * beta


def laplace_loss_bound(alpha: float, beta: float, delta: float) -> float:
    """β + e^β α + (e^β - 1) ln(1/δ): the privacy loss bound outside an event of probability δ."""
    return beta + math.exp(beta) * alpha + (math.exp(beta) - 1) * math.log(1 / delta)


def laplace_beta(eps: float, delta: float) -> float:
    """The largest β with laplace_loss_bound(ε/2, β, δ) <= ε, by bisection."""
    lo, hi = 0.0, eps
    for _ in range(100):
        mid = (lo + hi) / 2
        if laplace_loss_bound(eps / 2, mid, delta) <= eps:
            lo = mid
        else:
            hi = mid
    return lo


def nrs_laplace_beta(eps: float, delta: float) -> float:
    """β = ε/(2 ln(2/δ)), Lemma 2.9 of NRS 2007 for d = 1."""
    return eps / (2 * math.log(2 / delta))


@dataclass(frozen=True)
class MedianRelease:
    median: float
    local: float
    smooth_cauchy: float  # S*_β at the Cauchy β
    k_cauchy: int
    cauchy_scale: float  # S*/α; the median of |noise| equals this
    smooth_laplace: float
    k_laplace: int
    laplace_scale: float  # S*/α
    global_scale: float  # Λ/ε for the Laplace mechanism with global sensitivity


def median_release(x: Array, lam: float, eps: float, delta: float) -> MedianRelease:
    a = median_a_profile(x, lam)
    alpha_c, beta_c = cauchy_params(eps)
    s_c, k_c = smooth_from_profile(a, beta_c)
    beta_l = laplace_beta(eps, delta)
    s_l, k_l = smooth_from_profile(a, beta_l)
    return MedianRelease(
        median=float(np.median(x)),
        local=float(a[0]),
        smooth_cauchy=s_c,
        k_cauchy=k_c,
        cauchy_scale=s_c / alpha_c,
        smooth_laplace=s_l,
        k_laplace=k_l,
        laplace_scale=s_l / (eps / 2),
        global_scale=lam / eps,
    )


def ages(rng: np.random.Generator, n: int) -> Array:
    """Synthetic ages: N(38, 14²) clipped to [0, 100]."""
    return np.clip(rng.normal(38.0, 14.0, size=n), 0.0, AGE_MAX)


def empirical_median_abs_error(rng: np.random.Generator, scale: float, cauchy: bool, trials: int) -> float:
    z = rng.standard_cauchy(trials) if cauchy else rng.laplace(0.0, 1.0, trials)
    return float(np.median(np.abs(scale * z)))


# --- Propose-test-release -------------------------------------------------------------------


def distance_to_instability(a: Array, b: float) -> float:
    """min{k : A^(k)(x) > b}: how many records must change before LS can exceed b (∞ if never)."""
    above = np.nonzero(a > b)[0]
    return float(above[0]) if len(above) else math.inf


def ptr_pass_probability(dist: float, eps: float, delta: float) -> float:
    """Pr[dist + Lap(1/ε) > ln(1/δ)/ε]."""
    if math.isinf(dist):
        return 1.0
    gap = eps * dist - math.log(1 / delta)  # in units of the noise scale 1/ε
    return 1 - 0.5 * math.exp(-gap) if gap >= 0 else 0.5 * math.exp(gap)


# --- Inverse sensitivity: the median by the exponential mechanism ---------------------------


def inverse_sensitivity_pieces(x: Array | tuple[float, ...], lam: float) -> tuple[Array, Array]:
    """Lengths of the intervals (x_i, x_{i+1}), i = 0..n, of the padded data, and len(D, t) on each.

    len(D, t), the fewest records to change so that the median becomes t, is i - m + 1 on
    (x_i, x_{i+1}) for i ≥ m and m - i for i < m, with x_0 = 0, x_{n+1} = Λ and m = (n + 1)/2.
    """
    xs = np.sort(np.asarray(x, dtype=np.float64))
    n = len(xs)
    m = (n + 1) // 2
    p = np.concatenate([[0.0], xs, [lam]])
    i = np.arange(n + 1)
    lens = np.where(i >= m, i - m + 1, m - i).astype(np.float64)
    return np.diff(p), lens


def inverse_sensitivity_probabilities(x: Array | tuple[float, ...], lam: float, eps: float) -> Array:
    """Probability of each interval: length × e^{-ε len/2}, normalized (the exponential mechanism, Δ = 1)."""
    lengths, lens = inverse_sensitivity_pieces(x, lam)
    logw = np.log(np.maximum(lengths, 1e-300)) - eps * lens / 2
    w = np.where(lengths > 0, np.exp(logw - logw.max()), 0.0)
    return w / w.sum()


def inverse_sensitivity_median_abs_error(x: Array | tuple[float, ...], lam: float, eps: float) -> float:
    """Exact median of |t - median(D)| for the inverse-sensitivity release, by bisection on its CDF."""
    xs = np.sort(np.asarray(x, dtype=np.float64))
    n = len(xs)
    med = float(xs[(n + 1) // 2 - 1])
    p = np.concatenate([[0.0], xs, [lam]])
    probs = inverse_sensitivity_probabilities(x, lam, eps)
    lengths = np.diff(p)

    def cdf(r: float) -> float:
        lo = np.clip(med - r, p[:-1], p[1:])
        hi = np.clip(med + r, p[:-1], p[1:])
        frac = np.where(lengths > 0, (hi - lo) / np.maximum(lengths, 1e-300), 0.0)
        return float(np.sum(probs * frac))

    a, b = 0.0, lam
    for _ in range(200):
        mid = (a + b) / 2
        if cdf(mid) < 0.5:
            a = mid
        else:
            b = mid
    return (a + b) / 2


def inverse_sensitivity_sample(rng: np.random.Generator, x: Array, lam: float, eps: float, size: int) -> Array:
    p = np.concatenate([[0.0], np.sort(x), [lam]])
    j = rng.choice(len(p) - 1, size=size, p=inverse_sensitivity_probabilities(x, lam, eps))
    return p[j] + rng.uniform(size=size) * (p[j + 1] - p[j])


# --- Stability-based release: the mode ------------------------------------------------------


def mode_stability_distance(gap: int) -> int:
    """max(0, ⌈g/2⌉ - 1) for a lead g of the top count over the runner-up (replace neighbours).

    One replaced record moves the lead by at most 2, so this has sensitivity 1, and it is 0
    whenever some neighbour has a different mode (then g ≤ 2).
    """
    return max(0, math.ceil(gap / 2) - 1)


def mode_gap_for_pass(prob: float, eps: float, delta: float) -> int:
    """Smallest lead g whose stability release answers with probability at least `prob`."""
    g = 0
    while ptr_pass_probability(mode_stability_distance(g), eps, delta) < prob:
        g += 1
    return g


# --- Subsample-and-aggregate: the rate of an exponential distribution -----------------------


def saa_rate(rng: np.random.Generator, data: Array, blocks: int, clip: float, eps: float) -> Array:
    """For each row of data: split into `blocks` blocks, estimate 1/mean on each, clip to [0, clip],
    average, and add Lap(clip/(blocks ε)) (the average of clipped values has sensitivity clip/blocks)."""
    trials, n = data.shape
    per_block = 1.0 / data.reshape(trials, blocks, n // blocks).mean(axis=2)
    agg = np.clip(per_block, 0.0, clip).mean(axis=1)
    return agg + rng.laplace(0.0, clip / (blocks * eps), size=trials)


@dataclass(frozen=True)
class SaaError:
    bias: float
    sampling_sd: float  # standard deviation of the average of the block estimates
    noise_sd: float
    rmse: float


def saa_error(n: int, blocks: int, rate: float, clip: float, eps: float) -> SaaError:
    """Exact moments of the unclipped block estimate 1/x̄ with n_b = n/blocks points:
    E = θ n_b/(n_b - 1), Var = θ² n_b²/((n_b - 1)²(n_b - 2)) (inverse-gamma moments)."""
    nb = n // blocks
    bias = rate / (nb - 1)
    var = rate**2 * nb**2 / ((nb - 1) ** 2 * (nb - 2))
    noise_var = 2 * (clip / (blocks * eps)) ** 2
    return SaaError(bias, math.sqrt(var / blocks), math.sqrt(noise_var), math.sqrt(bias**2 + var / blocks + noise_var))


def saa_rmse_simulated(
    rng: np.random.Generator, n: int, blocks: int, rate: float, clip: float, eps: float, trials: int
) -> float:
    data = rng.exponential(1 / rate, size=(trials, n))
    est = saa_rate(rng, data, blocks, clip, eps)
    return float(np.sqrt(np.mean((est - rate) ** 2)))


SAA_N = 10_000
SAA_RATE = 1.0
SAA_CLIP = 5.0
SAA_BLOCKS = [10, 20, 25, 50, 100, 200, 250, 400, 500, 1000, 2000]

# --- Clipping: a private mean of log-normal incomes -----------------------------------------

INCOME_MU = math.log(40.0)  # median income 40 (thousands)
INCOME_SIGMA = 0.8


@dataclass(frozen=True)
class ClipError:
    bias: float  # E[X] - E[min(X, C)] = E[(X - C)_+]
    sampling_sd: float  # sd of the mean of n clipped values
    noise_sd: float  # sd of Lap(C/(nε))
    rmse: float


def lognormal_mean(mu: float, sigma: float) -> float:
    return math.exp(mu + sigma**2 / 2)


def lognormal_partial_moment(mu: float, sigma: float, c: float, j: int) -> float:
    """E[X^j 1{X <= c}] = e^{jμ + j²σ²/2} Φ((ln c - μ - jσ²)/σ) for X log-normal."""
    return math.exp(j * mu + j * j * sigma**2 / 2) * normal_cdf((math.log(c) - mu - j * sigma**2) / sigma)


def lognormal_tail(mu: float, sigma: float, c: float) -> float:
    """Pr[X > c]."""
    return 1 - normal_cdf((math.log(c) - mu) / sigma)


def clip_error(c: float, n: int, eps: float, mu: float = INCOME_MU, sigma: float = INCOME_SIGMA) -> ClipError:
    """Error of (1/n) Σ min(x_i, C) + Lap(C/(nε)) as an estimate of E[X]."""
    tail = lognormal_tail(mu, sigma, c)
    m1 = lognormal_partial_moment(mu, sigma, c, 1) + c * tail  # E[min(X, C)]
    m2 = lognormal_partial_moment(mu, sigma, c, 2) + c * c * tail  # E[min(X, C)²]
    bias = lognormal_mean(mu, sigma) - m1
    sampling = math.sqrt((m2 - m1 * m1) / n)
    noise = math.sqrt(2) * c / (n * eps)
    return ClipError(bias, sampling, noise, math.sqrt(bias**2 + sampling**2 + noise**2))


def optimal_clip(n: int, eps: float, mu: float = INCOME_MU, sigma: float = INCOME_SIGMA) -> float:
    """The C that minimizes the root-mean-square error, by golden-section search on ln C."""
    lo, hi = math.log(1.0), math.log(1e5)
    g = (math.sqrt(5) - 1) / 2
    for _ in range(200):
        a, b = hi - g * (hi - lo), lo + g * (hi - lo)
        if clip_error(math.exp(a), n, eps, mu, sigma).rmse < clip_error(math.exp(b), n, eps, mu, sigma).rmse:
            hi = b
        else:
            lo = a
    return math.exp((lo + hi) / 2)


def lognormal_quantile_by_bisection(q: float, mu: float = INCOME_MU, sigma: float = INCOME_SIGMA) -> float:
    lo, hi = -10.0, 10.0  # standard normal quantile
    for _ in range(200):
        mid = (lo + hi) / 2
        if normal_cdf(mid) < q:
            lo = mid
        else:
            hi = mid
    return math.exp(mu + sigma * (lo + hi) / 2)


def clipped_mean_rmse_simulated(rng: np.random.Generator, c: float, n: int, eps: float, trials: int) -> float:
    x = rng.lognormal(INCOME_MU, INCOME_SIGMA, size=(trials, n))
    est = np.minimum(x, c).mean(axis=1) + rng.laplace(0.0, c / (n * eps), size=trials)
    return float(np.sqrt(np.mean((est - lognormal_mean(INCOME_MU, INCOME_SIGMA)) ** 2)))


INCOME_N = 1000

# --- The chapter's table --------------------------------------------------------------------


def print_table(rng: np.random.Generator) -> None:
    print("[Local sensitivity leak, Λ=100, ε=1]")
    ls1 = median_local_sensitivity(LEAK_X, LEAK_MAX)
    ls2 = median_local_sensitivity(LEAK_X2, LEAK_MAX)
    med = float(np.median(LEAK_X))
    print(f"  x: median={med:g}  LS={ls1:g};  x': median={float(np.median(LEAK_X2)):g}  LS={ls2:g}")
    zero = np.array([med])
    print(f"  privacy loss at the median: L_x,x'={float(local_noise_privacy_loss(zero, med, ls1, med, ls2)[0]):.3f}")
    threshold = (EPS + math.log(ls2 / ls1)) / (1 / ls1 - 1 / ls2)
    print(
        f"  L_x',x(y) > ε iff |y - median| > {threshold:.2f}; "
        f"Pr under x' = {math.exp(-threshold / ls2):.3f}, under x = {math.exp(-threshold / ls1):.4f}"
    )
    print(
        f"  delta(ε=1): x' vs x = {laplace_hockey_stick(EPS, med, ls2, med, ls1):.3f}  "
        f"x vs x' = {laplace_hockey_stick(EPS, med, ls1, med, ls2):.3f}"
    )

    print(f"[A^(k) walkthrough, x={WALK_X}, Λ={WALK_MAX}, β={WALK_BETA}]")
    a = median_a_profile(tuple(float(v) for v in WALK_X), WALK_MAX)
    brute = brute_force_median_profile(WALK_X, WALK_MAX)
    weighted = np.exp(-WALK_BETA * np.arange(len(a))) * a
    print("  k:        " + "  ".join(f"{k:5d}" for k in range(len(a))))
    print("  formula:  " + "  ".join(f"{v:5.2f}" for v in a))
    print("  brute:    " + "  ".join(f"{v:5.2f}" for v in brute))
    print("  weighted: " + "  ".join(f"{v:5.2f}" for v in weighted))
    s, k = smooth_from_profile(a, WALK_BETA)
    print(f"  S*={s:.3f} at k={k}  max |formula - brute|={float(np.max(np.abs(a - brute))):g}")

    print("[Admissible noise parameters, ε=1, δ=1e-6]")
    alpha_c, beta_c = cauchy_params(EPS)
    print(f"  Cauchy: α={alpha_c:.4f} β={beta_c:.4f}  loss bound={cauchy_loss_bound(alpha_c, beta_c):.4f}")
    beta_l = laplace_beta(EPS, DELTA)
    print(
        f"  Laplace: α=0.5 β={beta_l:.4f} (NRS β={nrs_laplace_beta(EPS, DELTA):.4f}, "
        f"loss bound at NRS β={laplace_loss_bound(0.5, nrs_laplace_beta(EPS, DELTA), DELTA):.4f})"
    )

    n = 1001
    x = ages(rng, n)
    r = median_release(x, AGE_MAX, EPS, DELTA)
    print(f"[Median of n={n} ages, Λ=100, ε=1, δ=1e-6]")
    print(f"  median={r.median:.2f}  LS={r.local:.4f}  global scale Λ/ε={r.global_scale:g}")
    print(f"  Cauchy: S*={r.smooth_cauchy:.4f} at k={r.k_cauchy}  noise scale={r.cauchy_scale:.3f}")
    print(
        f"  Laplace: S*={r.smooth_laplace:.4f} at k={r.k_laplace}  noise scale={r.laplace_scale:.3f}  "
        f"median |error|={math.log(2) * r.laplace_scale:.3f}"
    )
    print(f"  global Laplace median |error|={math.log(2) * r.global_scale:.2f}")
    emp_c = empirical_median_abs_error(rng, r.cauchy_scale, True, 200_000)
    emp_l = empirical_median_abs_error(rng, r.laplace_scale, False, 200_000)
    print(f"  simulated median |error|: Cauchy={emp_c:.3f}  Laplace={emp_l:.3f}")
    c_cauchy = rng.standard_cauchy(200_000)
    print(
        f"  Cauchy: Pr[|error| > 10 x scale]={float(np.mean(np.abs(c_cauchy) > 10)):.4f} "
        f"(exact {1 - 2 / math.pi * math.atan(10):.4f})"
    )

    print("[Propose-test-release, b=1, ε=1, δ=1e-6]")
    profile = median_a_profile(x, AGE_MAX)
    dist = distance_to_instability(profile, 1.0)
    print(f"  threshold ln(1/δ)/ε={math.log(1 / DELTA) / EPS:.2f}")
    print(
        f"  n={n}: distance={dist:g}  pass probability={ptr_pass_probability(dist, EPS, DELTA):.6f}  "
        f"noise scale b/ε=1  median |error|={math.log(2):.3f}"
    )
    small = ages(rng, 101)
    dist_small = distance_to_instability(median_a_profile(small, AGE_MAX), 1.0)
    print(f"  n=101: distance={dist_small:g}  pass probability={ptr_pass_probability(dist_small, EPS, DELTA):.2e}")

    print(f"[Subsample-and-aggregate: rate θ={SAA_RATE:g} from n={SAA_N} exponential draws, clip {SAA_CLIP:g}, ε=1]")
    for blocks in (10, 100, 250, 1000):
        e = saa_error(SAA_N, blocks, SAA_RATE, SAA_CLIP, EPS)
        sim = saa_rmse_simulated(rng, SAA_N, blocks, SAA_RATE, SAA_CLIP, EPS, trials=1000)
        print(
            f"  B={blocks}: bias={e.bias:.4f}  sampling sd={e.sampling_sd:.4f}  noise sd={e.noise_sd:.4f}  "
            f"rmse={e.rmse:.4f}  simulated={sim:.4f}"
        )
    best = min(SAA_BLOCKS, key=lambda b: saa_error(SAA_N, b, SAA_RATE, SAA_CLIP, EPS).rmse)
    print(
        f"  best B among {SAA_BLOCKS}: {best}  rmse={saa_error(SAA_N, best, SAA_RATE, SAA_CLIP, EPS).rmse:.4f}"
        f"  non-private sd θ/sqrt(n)={SAA_RATE / math.sqrt(SAA_N):.4f}"
    )

    print(f"[Clipped mean of log-normal incomes, n={INCOME_N}, ε=1]")
    mean = lognormal_mean(INCOME_MU, INCOME_SIGMA)
    sd = math.sqrt((math.exp(INCOME_SIGMA**2) - 1) * mean**2 / INCOME_N)
    print(f"  true mean={mean:.3f}  non-private sd of the unclipped mean={sd:.3f}")
    c_star = optimal_clip(INCOME_N, EPS)
    q99 = lognormal_quantile_by_bisection(0.99)
    for label, c in (("C*", c_star), ("q50", 40.0), ("q99", q99), ("C=1000", 1000.0)):
        e = clip_error(c, INCOME_N, EPS)
        sim = clipped_mean_rmse_simulated(rng, c, INCOME_N, EPS, trials=5_000)
        print(
            f"  {label}: C={c:.2f}  tail={lognormal_tail(INCOME_MU, INCOME_SIGMA, c):.4f}  bias={e.bias:.3f}  "
            f"sampling sd={e.sampling_sd:.3f}  noise sd={e.noise_sd:.3f}  rmse={e.rmse:.3f}  simulated={sim:.3f}"
        )
    for n_big in (100, 10_000, 100_000):
        c = optimal_clip(n_big, EPS)
        print(f"  n={n_big}: C*={c:.1f}  rmse={clip_error(c, n_big, EPS).rmse:.4f}")

    print(f"[Inverse sensitivity, walkthrough x={WALK_X}, Λ={WALK_MAX}, ε=1]")
    lengths, lens = inverse_sensitivity_pieces(tuple(float(v) for v in WALK_X), WALK_MAX)
    probs = inverse_sensitivity_probabilities(tuple(float(v) for v in WALK_X), WALK_MAX, EPS)
    print("  lengths: " + " ".join(f"{v:g}" for v in lengths))
    print("  len:     " + " ".join(f"{v:g}" for v in lens))
    print("  probs:   " + " ".join(f"{v:.4f}" for v in probs))
    print(f"  Pr[3 <= t <= 6]={float(probs[2] + probs[3]):.4f}")
    print(f"[Inverse sensitivity, median of n={n} ages, ε=1]")
    inv = inverse_sensitivity_median_abs_error(x, AGE_MAX, EPS)
    draws = inverse_sensitivity_sample(rng, x, AGE_MAX, EPS, 200_000)
    sim_inv = float(np.median(np.abs(draws - r.median)))
    print(f"  median |error|={inv:.4f}  simulated={sim_inv:.4f}  Cauchy smooth={r.cauchy_scale:.3f}  LS={r.local:.4f}")
    inv_small = inverse_sensitivity_median_abs_error(small, AGE_MAX, EPS)
    print(f"  n=101: median |error|={inv_small:.3f}")

    print("[Stability-based release of the mode, ε=1, δ=1e-6]")
    for gap in (10, 30, 40, 60):
        dist = mode_stability_distance(gap)
        print(f"  lead={gap}: distance={dist}  pass probability={ptr_pass_probability(dist, EPS, DELTA):.4g}")
    print(f"  lead needed for pass probability 0.95: {mode_gap_for_pass(0.95, EPS, DELTA)}")


# --- Figures --------------------------------------------------------------------------------


def plot_local_leak() -> Figure:
    """The two releases with noise scaled to local sensitivity, and the privacy loss between them."""
    med = float(np.median(LEAK_X))
    b1 = median_local_sensitivity(LEAK_X, LEAK_MAX)
    b2 = median_local_sensitivity(LEAK_X2, LEAK_MAX)
    y = np.linspace(med - 150, med + 150, 1201)
    fig, (left, right) = plt.subplots(1, 2, figsize=(10, 4))
    left.plot(y, np.exp(laplace_log_density(y, med, b1)), label=f"M(D): Lap({b1:g}), LS(D) = {b1:g}")
    left.plot(y, np.exp(laplace_log_density(y, med, b2)), label=f"M(D′): Lap({b2:g}), LS(D′) = {b2:g}")
    left.set(
        yscale="log",
        ylim=(1e-8, 0.1),
        xlabel="released value y",
        ylabel="density",
        title="Same median, different noise scales",
    )
    left.legend(loc="lower center")
    loss = local_noise_privacy_loss(y, med, b2, med, b1)
    right.plot(y, loss, label="privacy loss of D′ against D")
    right.axhline(EPS, color="k", linestyle="--", label=r"$\pm\varepsilon$, $\varepsilon = 1$")
    right.axhline(-EPS, color="k", linestyle="--")
    right.set(
        xlabel="released value y", ylabel="privacy loss (nats)", ylim=(-3, 14), title="The privacy loss is unbounded"
    )
    right.legend(loc="upper center")
    return fig


def plot_smooth_profile(rng: np.random.Generator) -> Figure:
    """A^(k) and e^{-kβ}A^(k) for the walkthrough and for the ages data."""
    fig, (left, right) = plt.subplots(1, 2, figsize=(10, 4))
    a = median_a_profile(tuple(float(v) for v in WALK_X), WALK_MAX)
    ks = np.arange(len(a))
    left.plot(ks, a, "o-", label=r"$A^{(k)}(D)$")
    w = np.exp(-WALK_BETA * ks) * a
    left.plot(ks, w, "s-", label=rf"$e^{{-k\beta}}A^{{(k)}}(D)$, $\beta = {WALK_BETA:g}$")
    s, k = smooth_from_profile(a, WALK_BETA)
    left.plot([k], [s], "k*", markersize=14, label=rf"$S^*_\beta = {s:.2f}$")
    left.set(xlabel="distance k", ylabel="sensitivity", title=f"D = {WALK_X}, Λ = {WALK_MAX}")
    left.legend()

    x = ages(rng, 1001)
    a = median_a_profile(x, AGE_MAX)
    ks = np.arange(len(a))
    right.plot(ks, a, label=r"$A^{(k)}(D)$")
    for name, beta in (("Cauchy", cauchy_params(EPS)[1]), ("Laplace", laplace_beta(EPS, DELTA))):
        w = np.exp(-beta * ks) * a
        s, k = smooth_from_profile(a, beta)
        line = right.plot(ks, w, label=rf"$e^{{-k\beta}}A^{{(k)}}$, $\beta = {beta:.3f}$ ({name})")
        right.plot([k], [s], "*", color=line[0].get_color(), markersize=14)
    right.set(
        xscale="symlog",
        yscale="log",
        xlim=(0, 1001),
        ylim=(1e-3, 200),
        xlabel="distance k",
        ylabel="sensitivity (years)",
        title="Median of 1001 ages, Λ = 100",
    )
    right.legend(loc="upper left")
    return fig


def plot_methods_vs_n(rng: np.random.Generator, datasets: int = 10) -> Figure:
    """Median |error| of each way of releasing the median, and the pass rate of propose-test-release."""
    ns = [11, 21, 51, 101, 201, 501, 1001, 2001]
    rows: list[tuple[float, float, float, float, float]] = []
    inv_rows: list[float] = []
    for n in ns:
        c: list[float] = []
        lap: list[float] = []
        local: list[float] = []
        ptr: list[float] = []
        inv: list[float] = []
        for _ in range(datasets):
            x = ages(rng, n)
            r = median_release(x, AGE_MAX, EPS, DELTA)
            c.append(r.cauchy_scale)
            lap.append(math.log(2) * r.laplace_scale)
            local.append(math.log(2) * r.local / EPS)
            dist = distance_to_instability(median_a_profile(x, AGE_MAX), 1.0)
            ptr.append(ptr_pass_probability(dist, EPS, DELTA))
            inv.append(inverse_sensitivity_median_abs_error(x, AGE_MAX, EPS))
        inv_rows.append(float(np.mean(inv)))
        rows.append(
            (
                float(np.mean(c)),
                float(np.mean(lap)),
                float(np.mean(local)),
                float(np.mean(ptr)),
                math.log(2) * AGE_MAX / EPS,
            )
        )
    fig, (left, right) = plt.subplots(1, 2, figsize=(10, 5))
    left.plot(ns, [r[4] for r in rows], "k--", label="global sensitivity, Laplace")
    left.plot(ns, [r[0] for r in rows], "o-", label=r"smooth, Cauchy ($\delta = 0$)")
    left.plot(ns, [r[1] for r in rows], "s-", label=r"smooth, Laplace ($\delta = 10^{-6}$)")
    left.axhline(math.log(2) * 1.0 / EPS, color="C2", linestyle="-.", label="propose-test-release, b = 1, if it passes")
    left.plot(ns, inv_rows, "^-", color="C4", label=r"inverse sensitivity ($\delta = 0$)")
    left.plot(ns, [r[2] for r in rows], ":", color="gray", label="local sensitivity (not private)")
    left.set(
        xscale="log",
        yscale="log",
        xlabel="dataset size n",
        ylabel="median |noise| (years)",
        title=r"Releasing the median age, $\varepsilon = 1$",
    )
    left.legend(fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=2)
    right.plot(ns, [r[3] for r in rows], "o-", color="C2")
    right.set(
        xscale="log",
        ylim=(-0.05, 1.05),
        xlabel="dataset size n",
        ylabel="pass probability",
        title="Propose-test-release: Pr[not ⊥], b = 1",
    )
    return fig


def plot_saa(rng: np.random.Generator) -> Figure:
    errs = [saa_error(SAA_N, b, SAA_RATE, SAA_CLIP, EPS) for b in SAA_BLOCKS]
    sims = [saa_rmse_simulated(rng, SAA_N, b, SAA_RATE, SAA_CLIP, EPS, trials=200) for b in SAA_BLOCKS]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(SAA_BLOCKS, [e.rmse for e in errs], "k", linewidth=2, label="root-mean-square error (formula)")
    ax.plot(SAA_BLOCKS, sims, "ko", label="simulated")
    ax.plot(SAA_BLOCKS, [e.bias for e in errs], "--", label="bias of small blocks")
    ax.plot(SAA_BLOCKS, [e.noise_sd for e in errs], "--", label="noise sd, √2·R/(Bε)")
    ax.plot(SAA_BLOCKS, [e.sampling_sd for e in errs], ":", label="sampling sd")
    ax.set(
        xscale="log",
        yscale="log",
        xlabel="number of blocks B",
        ylabel="error",
        title=f"Rate of an exponential, n = {SAA_N}, ε = 1",
    )
    ax.legend(fontsize=8)
    return fig


def plot_clip(rng: np.random.Generator) -> Figure:
    cs = [10 ** (1 + i / 40) for i in range(121)]
    errs = [clip_error(c, INCOME_N, EPS) for c in cs]
    fig, (left, right) = plt.subplots(1, 2, figsize=(10, 4))
    left.plot(cs, [e.rmse for e in errs], "k", linewidth=2, label="root-mean-square error")
    left.plot(cs, [e.bias for e in errs], "--", label="bias E[(X − C)₊]")
    left.plot(cs, [e.noise_sd for e in errs], "--", label="noise sd √2·C/(nε)")
    left.plot(cs, [e.sampling_sd for e in errs], ":", label="sampling sd")
    sim_cs = [20.0, 50.0, 100.0, 200.0, 500.0, 1000.0]
    left.plot(
        sim_cs,
        [clipped_mean_rmse_simulated(rng, c, INCOME_N, EPS, trials=5_000) for c in sim_cs],
        "ko",
        label="simulated",
    )
    c_star = optimal_clip(INCOME_N, EPS)
    left.axvline(c_star, color="gray", linestyle=":")
    left.set(
        xscale="log",
        yscale="log",
        ylim=(0.05, 60),
        xlabel="clipping norm C (thousands)",
        ylabel="error",
        title=f"Mean income, n = {INCOME_N}, ε = 1",
    )
    left.legend(fontsize=8)
    ns = [10 ** (2 + i / 8) for i in range(25)]
    stars = [optimal_clip(round(n), EPS) for n in ns]
    right.plot(ns, stars, label="optimal C*")
    right.axhline(lognormal_mean(INCOME_MU, INCOME_SIGMA), color="k", linestyle="--", label="true mean")
    right.axhline(lognormal_quantile_by_bisection(0.99), color="gray", linestyle=":", label="99th percentile")
    right.set(
        xscale="log",
        yscale="log",
        xlabel=r"dataset size n ($\varepsilon = 1$)",
        ylabel="clipping norm (thousands)",
        title="The best clipping norm grows with n",
    )
    right.legend()
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
        "sens_local_leak.svg": plot_local_leak(),
        "sens_smooth_profile.svg": plot_smooth_profile(rng),
        "sens_methods_vs_n.svg": plot_methods_vs_n(rng),
        "sens_saa_blocks.svg": plot_saa(rng),
        "sens_clip_tradeoff.svg": plot_clip(rng),
    }
    for name, fig in figures.items():
        fig.tight_layout()
        fig.savefig(out / name, metadata={"Date": None})  # no timestamp, so output is reproducible
        plt.close(fig)
        print(f"saved {out / name}")


if __name__ == "__main__":
    main()
