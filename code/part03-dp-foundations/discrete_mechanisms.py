"""Check the discrete and floating-point-safe mechanisms of chapter 25 and draw its figures.

    uv run python code/part03-dp-foundations/discrete_mechanisms.py
    uv run python code/part03-dp-foundations/discrete_mechanisms.py --seed 1 --out /tmp/figures

Prints the numbers quoted in 08-discrete-mechanisms.md (they match exactly with the default seed),
then saves the chapter's figures.

Two halves:

- A toy floating-point model of the textbook Laplace sampler, small enough to enumerate every output
  exactly. It shows that the outputs reachable from f(D) = 0 and from f(D') = 1 differ (Mironov 2012),
  and that snapping the output to a coarse grid repairs it.
- The exact samplers of Canonne, Kamath & Steinke (2020), Algorithms 1-3: Bernoulli(exp(-γ)) for rational
  γ, the discrete Laplace and the discrete Gaussian. They use only integer and `fractions.Fraction`
  arithmetic and uniform integers from a `random.Random`. Pass `secrets.SystemRandom()` instead of a
  seeded `random.Random` for a deployment; the seeded one is here so the numbers are reproducible.
"""

import argparse
import math
import random
from collections import Counter
from collections.abc import Callable, Sequence
from fractions import Fraction
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
matplotlib.rcParams["svg.hashsalt"] = "discrete_mechanisms"  # stable element ids, so reruns don't churn the SVGs
matplotlib.rcParams["svg.fonttype"] = "none"  # keep text as text: smaller, diffable files

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from numpy.typing import NDArray

# Figures are build output: _build/figures/<part>/, which the build copies into each book.
FIGURES = Path(__file__).resolve().parents[2] / "_build" / "figures" / Path(__file__).parent.name

Pmf = dict[float, float]  # a distribution on a finite set of floats: value -> probability


def normal_cdf(x: float) -> float:
    """Φ(x); erfc keeps full relative precision in the lower tail."""
    return 0.5 * math.erfc(-x / math.sqrt(2))


# --- Floating-point Laplace: a toy model small enough to enumerate --------------------------
#
# A p-bit float has a p-bit significand (IEEE doubles have p = 53). The toy sampler is the textbook
# one, y = f + S * b * ln(1/U) with S a fair sign, computed in p-bit arithmetic: U is an ideal
# uniform p-bit float, and the logarithm and the sum are each rounded to p bits.


def round_sig(x: NDArray[np.float64], p: int) -> NDArray[np.float64]:
    """Round each entry to the nearest float with a p-bit significand (ties to even)."""
    m, e = np.frexp(x)  # x = m 2^e with 0.5 <= |m| < 1
    return np.ldexp(np.rint(m * 2.0**p), e - p)


def uniform_floats(p: int, emin: int = 40) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Every p-bit float u in [2^-emin, 1) with its probability under an ideal uniform sampler.

    The sampler draws a real U ~ Uniform(0, 1) and rounds it down to a p-bit float, so u has
    probability equal to the gap to the next float, 2^(e - p + 1) for u in [2^e, 2^(e+1)).
    The mass 2^-emin below the smallest u is left out.
    """
    values: list[NDArray[np.float64]] = []
    probs: list[NDArray[np.float64]] = []
    steps = np.arange(2 ** (p - 1), dtype=np.float64)
    for e in range(-emin, 0):
        values.append(2.0**e * (1 + steps / 2 ** (p - 1)))
        probs.append(np.full(steps.size, 2.0 ** (e - p + 1)))
    return np.concatenate(values), np.concatenate(probs)


def float_laplace_pmf(f: float, b: float, p: int, snap: float = 0.0, bound: float = math.inf) -> Pmf:
    """Exact output distribution of the toy p-bit textbook Laplace sampler around f.

    With snap > 0 the output is then rounded to a multiple of `snap` and clamped to [-bound, bound],
    as in the snapping mechanism.
    """
    u, pu = uniform_floats(p)
    magnitude = round_sig(-b * np.log(u), p)  # b ln(1/U), rounded to p bits
    ys = np.concatenate([round_sig(f + magnitude, p), round_sig(f - magnitude, p)])
    ws = np.concatenate([pu, pu]) / 2
    if snap > 0:
        ys = np.clip(snap * np.rint(ys / snap), -bound, bound)
    keys, inverse = np.unique(ys, return_inverse=True)
    mass = np.bincount(inverse, weights=ws)
    return {float(k): float(w) for k, w in zip(keys, mass, strict=True)}


def mass_outside(p_dist: Pmf, q_dist: Pmf) -> float:
    """P(S) for S = outputs P can produce and Q cannot: a lower bound on δ at every ε."""
    return sum(w for y, w in p_dist.items() if y not in q_dist)


def max_log_ratio(p_dist: Pmf, q_dist: Pmf) -> float:
    """max over shared outputs of |ln p(y)/q(y)|; infinite if the supports differ."""
    if p_dist.keys() != q_dist.keys():
        return math.inf
    return max(abs(math.log(w / q_dist[y])) for y, w in p_dist.items())


def support_in(dist: Pmf, lo: float, hi: float) -> list[float]:
    return sorted(y for y in dist if lo <= y <= hi)


# --- Exact randomness: uniform integers, Bernoulli(n/d) --------------------------------------


class UniformSource:
    """Uniform integers from a `random.Random`, counted so the cost of each sampler can be measured.

    `random.Random.randrange(n)` is exactly uniform on {0, ..., n-1} for any n: it draws whole
    random bits and rejects, with no floating point. Use `secrets.SystemRandom()` in a deployment.
    """

    def __init__(self, rng: random.Random) -> None:
        self._rng = rng
        self.draws = 0

    def below(self, n: int) -> int:
        self.draws += 1
        return self._rng.randrange(n)


def bernoulli(p: Fraction, src: UniformSource) -> int:
    """Bernoulli(n/d) exactly: a uniform integer in {0, ..., d-1} is below n with probability n/d."""
    return int(src.below(p.denominator) < p.numerator)


# --- Algorithm 1: Bernoulli(exp(-γ)) for rational γ >= 0 -------------------------------------


def bernoulli_exp_unit(gamma: Fraction, src: UniformSource) -> int:
    """Bernoulli(exp(-γ)) for 0 <= γ <= 1: draw A_k ~ Bernoulli(γ/k) for k = 1, 2, ... until one is 0.

    With K the index of that first 0, Pr[K > k] = γ^k/k!, and Pr[K odd] = e^-γ.
    """
    n, d = gamma.numerator, gamma.denominator
    k = 1
    while src.below(d * k) < n:  # Bernoulli(γ/k) = Bernoulli(n/(dk))
        k += 1
    return k % 2


def bernoulli_exp(gamma: Fraction, src: UniformSource) -> int:
    """Bernoulli(exp(-γ)) for any rational γ >= 0 (Canonne, Kamath & Steinke 2020, Algorithm 1).

    For γ > 1, e^-γ = (e^-1)^floor(γ) e^-(γ - floor(γ)): one Bernoulli(e^-1) per whole unit, stopping
    at the first 0, then one draw for the fractional part.
    """
    if gamma < 0:
        raise ValueError("γ must be non-negative")
    whole = math.floor(gamma)
    for _ in range(whole):
        if bernoulli_exp_unit(Fraction(1), src) == 0:
            return 0
    return bernoulli_exp_unit(gamma - whole, src)


def bernoulli_exp_unit_cost(gamma: float) -> float:
    """Expected uniform draws of bernoulli_exp_unit: E[K] = Σ_k Pr[K > k] = Σ γ^k/k! = e^γ."""
    return math.exp(gamma)


def bernoulli_exp_cost(gamma: float) -> float:
    """Expected uniform draws of bernoulli_exp: the j-th whole unit runs with probability e^-j."""
    whole = math.floor(gamma)
    units = sum(math.exp(-j) for j in range(whole)) * math.e
    return units + math.exp(-whole) * math.exp(gamma - whole)


def first_zero_pmf(gamma: float, k: int) -> float:
    """Pr[K = k] in bernoulli_exp_unit: γ^(k-1)/(k-1)! - γ^k/k!."""
    return gamma ** (k - 1) / math.factorial(k - 1) - gamma**k / math.factorial(k)


def first_zero_counts(gamma: Fraction, n: int, src: UniformSource) -> list[int]:
    """Run bernoulli_exp_unit's loop n times and return each K (the index of the first 0)."""
    out: list[int] = []
    for _ in range(n):
        k = 1
        while bernoulli(gamma / k, src) == 1:
            k += 1
        out.append(k)
    return out


# --- Algorithm 2: the discrete Laplace ---------------------------------------------------------


def discrete_laplace(s: int, t: int, src: UniformSource) -> int:
    """A draw from the discrete Laplace with scale t/s, Pr[Z = z] ∝ exp(-|z| s/t) (Algorithm 2).

    X = U + tV is geometric with Pr[X = x] ∝ e^(-x/t): U in {0..t-1} accepted with probability
    e^(-U/t), V the number of Bernoulli(e^-1) successes before a failure. Y = floor(X/s) is geometric
    with ratio e^(-s/t). A fair sign makes it two-sided; (B = 1, Y = 0) is rejected so 0 is not
    counted twice.
    """
    while True:
        u = src.below(t)
        if bernoulli_exp(Fraction(u, t), src) == 0:
            continue
        v = 0
        while bernoulli_exp(Fraction(1), src) == 1:
            v += 1
        y = (u + t * v) // s
        sign = src.below(2)
        if sign == 1 and y == 0:
            continue
        return (1 - 2 * sign) * y


def discrete_laplace_pmf(z: int, scale: float) -> float:
    """Pr[Z = z] = (1 - r)/(1 + r) r^|z| with r = e^(-1/scale)."""
    r = math.exp(-1 / scale)
    return (1 - r) / (1 + r) * r ** abs(z)


def discrete_laplace_variance(scale: float) -> float:
    """Var Z = 2r/(1 - r)^2 with r = e^(-1/scale); the continuous Lap(scale) has 2 scale^2."""
    r = math.exp(-1 / scale)
    return 2 * r / (1 - r) ** 2


def discrete_laplace_iterations(s: int, t: int) -> float:
    """Expected passes through Algorithm 2's outer loop: 1 / Pr[a pass returns]."""
    accept_u = sum(math.exp(-u / t) for u in range(t)) / t
    y_zero = 1 - math.exp(-s / t)  # Pr[Y = 0] = Pr[X < s]
    return 1 / (accept_u * (1 - y_zero / 2))


# --- Algorithm 3: the discrete Gaussian ---------------------------------------------------------


def floor_sqrt(x: Fraction) -> int:
    """floor(√x) for rational x >= 0: the largest n with n² <= x, and n² <= x iff n² <= floor(x)."""
    return math.isqrt(math.floor(x))


def discrete_gaussian(sigma2: Fraction, src: UniformSource) -> int:
    """A draw from N_Z(0, σ²), Pr[X = x] ∝ exp(-x²/(2σ²)) on the integers (Algorithm 3).

    Propose Y from the discrete Laplace with scale t = floor(σ) + 1 and accept with probability
    exp(-(|Y| - σ²/t)²/(2σ²)). Proposal times acceptance is ∝ exp(-Y²/(2σ²)).
    """
    t = floor_sqrt(sigma2) + 1
    while True:
        y = discrete_laplace(1, t, src)
        gamma = (abs(y) - sigma2 / t) ** 2 / (2 * sigma2)
        if bernoulli_exp(gamma, src) == 1:
            return y


def discrete_gaussian_support(sigma: float) -> range:
    """Integers carrying all but a negligible part (< 1e-30) of N_Z(0, σ²)."""
    width = math.ceil(12 * sigma) + 2
    return range(-width, width + 1)


def discrete_gaussian_pmf(sigma: float, shift: int = 0) -> dict[int, float]:
    """Pr[X = x] for X ~ N_Z(shift, σ²), normalized by summing over the effective support."""
    xs = discrete_gaussian_support(sigma)
    weights = {x + shift: math.exp(-(x * x) / (2 * sigma**2)) for x in xs}
    total = sum(weights.values())
    return {x: w / total for x, w in weights.items()}


def theta_shift(c: float, sigma: float) -> float:
    """Σ_x exp(-(x - c)²/(2σ²)) over the integers: the discrete Gaussian normalizer shifted by c."""
    return sum(math.exp(-((x - c) ** 2) / (2 * sigma**2)) for x in discrete_gaussian_support(sigma))


def discrete_gaussian_variance(sigma: float) -> float:
    """E[X²] for X ~ N_Z(0, σ²), by direct summation."""
    return sum(x * x * w for x, w in discrete_gaussian_pmf(sigma).items())


def discrete_gaussian_variance_deficit(sigma: float, terms: int = 20) -> float:
    """1 - E[X²]/σ² = θ'/((1 + θ)σ²) from Poisson summation, θ = Σ_{k≠0} e^(-2π²σ²k²), θ' = Σ 4π²σ⁴k² e^(...).

    Computed directly, so it stays accurate when it is far below machine precision relative to 1.
    """
    a = 2 * math.pi**2 * sigma**2
    theta = 2 * sum(math.exp(-a * k * k) for k in range(1, terms))
    theta_prime = 2 * sum(4 * math.pi**2 * sigma**4 * k * k * math.exp(-a * k * k) for k in range(1, terms))
    return theta_prime / ((1 + theta) * sigma**2)


def discrete_gaussian_variance_poisson(sigma: float) -> float:
    """E[X²] = σ² - θ'/(1 + θ) from Poisson summation."""
    return sigma**2 * (1 - discrete_gaussian_variance_deficit(sigma))


def discrete_gaussian_renyi(alpha: float, sigma: float, shift: int) -> float:
    """D_α(N_Z(0, σ²) ‖ N_Z(shift, σ²)) by summation over the support."""
    p = discrete_gaussian_pmf(sigma)
    q = discrete_gaussian_pmf(sigma, shift)
    total = sum(w**alpha * q.get(x, 0.0) ** (1 - alpha) for x, w in p.items() if x in q)
    return math.log(total) / (alpha - 1)


def discrete_gaussian_delta(eps: float, sigma: float, sens: int = 1) -> float:
    """δ(ε) of N_Z(0, σ²) against N_Z(sens, σ²): Pr[X > a - Δ] - e^ε Pr[X > a] with a = εσ²/Δ + Δ/2."""
    p = discrete_gaussian_pmf(sigma)
    a = eps * sigma**2 / sens + sens / 2
    upper = sum(w for x, w in p.items() if x > a - sens)
    lower = sum(w for x, w in p.items() if x > a)
    return max(upper - math.exp(eps) * lower, 0.0)


def gaussian_delta(eps: float, sigma: float, sens: float = 1.0) -> float:
    """δ(ε) of the continuous Gaussian mechanism (Balle & Wang 2018)."""
    a, c = sens / (2 * sigma), eps * sigma / sens
    return normal_cdf(a - c) - math.exp(eps) * normal_cdf(-a - c)


def gaussian_acceptance(sigma2: Fraction) -> float:
    """Pr[a proposal is accepted] in Algorithm 3: Σ_y Lap_Z(t)(y) exp(-(|y| - σ²/t)²/(2σ²))."""
    t = floor_sqrt(sigma2) + 1
    s2 = float(sigma2)
    sigma = math.sqrt(s2)
    return sum(
        discrete_laplace_pmf(y, t) * math.exp(-((abs(y) - s2 / t) ** 2) / (2 * s2))
        for y in discrete_gaussian_support(max(sigma, t))
    )


# --- Empirical checks -----------------------------------------------------------------------------


def empirical_pmf(samples: Sequence[int]) -> dict[int, float]:
    counts = Counter(samples)
    return {x: c / len(samples) for x, c in counts.items()}


def tv_to(empirical: dict[int, float], exact: Callable[[int], float], support: range) -> float:
    """TV between an empirical PMF and an exact one, summed over `support` plus any stray values."""
    keys = set(support) | set(empirical)
    return 0.5 * sum(abs(empirical.get(x, 0.0) - exact(x)) for x in keys)


def reference_tv(rng: np.random.Generator, exact: Callable[[int], float], support: range, n: int) -> float:
    """TV between the exact PMF and the empirical PMF of n draws from it: the sampling-noise floor for tv_to."""
    xs = list(support)
    probs = np.array([exact(x) for x in xs])
    draws = rng.choice(len(xs), size=n, p=probs / probs.sum())
    return tv_to(empirical_pmf([xs[int(i)] for i in draws]), exact, support)


def mean_var(samples: Sequence[int]) -> tuple[float, float]:
    arr = np.asarray(samples, dtype=np.float64)
    return float(arr.mean()), float(arr.var())


def seeded_source(rng: np.random.Generator) -> UniformSource:
    """A UniformSource seeded from the NumPy generator, so --seed controls everything."""
    return UniformSource(random.Random(int(rng.integers(2**63))))


# --- Table ------------------------------------------------------------------------------------------


def print_table(rng: np.random.Generator) -> None:
    src = seeded_source(rng)

    print("[Floating-point Laplace, toy p-bit sampler, b = 1, f(D) = 0 vs f(D') = 1]")
    for p in (8, 12, 14):
        m0 = float_laplace_pmf(0.0, 1.0, p)
        m1 = float_laplace_pmf(1.0, 1.0, p)
        print(
            f"  p={p}: outputs of M(D)={len(m0)}  of M(D')={len(m1)}  "
            f"mass of M(D) impossible under M(D')={mass_outside(m0, m1):.4f}  "
            f"reverse={mass_outside(m1, m0):.4f}"
        )
    m0, m1 = float_laplace_pmf(0.0, 1.0, 12), float_laplace_pmf(1.0, 1.0, 12)
    near0 = sum(w for y, w in m0.items() if abs(y) < 0.5)
    near0_out = sum(w for y, w in m0.items() if abs(y) < 0.5 and y not in m1)
    print(f"  p=12, outputs with |y| < 0.5: mass={near0:.4f}  impossible under M(D')={near0_out:.4f}")
    print(
        f"  p=12, outputs in [0.25, 0.2525]: M(D) {len(support_in(m0, 0.25, 0.2525))}  "
        f"M(D') {len(support_in(m1, 0.25, 0.2525))}  shared "
        f"{len(set(support_in(m0, 0.25, 0.2525)) & set(support_in(m1, 0.25, 0.2525)))}"
    )
    for p in (12, 14):
        s0 = float_laplace_pmf(0.0, 1.0, p, snap=1.0, bound=10.0)
        s1 = float_laplace_pmf(1.0, 1.0, p, snap=1.0, bound=10.0)
        print(
            f"  snapped, p={p}, grid 1, clamp 10: outputs {len(s0)}  mass outside={mass_outside(s0, s1):.4f}  "
            f"max |log ratio|={max_log_ratio(s0, s1):.4f}"
        )

    print("[Algorithm 1: Bernoulli(exp(-gamma)), 10000 draws each]")
    n = 10_000
    for gamma in (Fraction(1, 2), Fraction(1), Fraction(5, 2), Fraction(4)):
        src.draws = 0
        hits = sum(bernoulli_exp(gamma, src) for _ in range(n))
        print(
            f"  gamma={gamma}: empirical={hits / n:.4f}  exact={math.exp(-float(gamma)):.4f}  "
            f"draws per sample={src.draws / n:.3f} (expected {bernoulli_exp_cost(float(gamma)):.3f})"
        )
    ks = first_zero_counts(Fraction(7, 10), n, src)
    emp = empirical_pmf(ks)
    print(
        "  gamma=7/10, Pr[K=k]: "
        + "  ".join(f"k={k}: {emp.get(k, 0.0):.4f}/{first_zero_pmf(0.7, k):.4f}" for k in range(1, 6))
    )
    print(f"  gamma=7/10: Pr[K odd]={sum(1 for k in ks if k % 2) / n:.4f}  e^-0.7={math.exp(-0.7):.4f}")

    print("[Algorithm 2: discrete Laplace]")
    for s, t in ((1, 1), (2, 3), (1, 10)):
        scale = t / s
        src.draws = 0
        zs = [discrete_laplace(s, t, src) for _ in range(n)]
        mean, var = mean_var(zs)
        support = range(-60 * t, 60 * t + 1)
        exact_l: Callable[[int], float] = lambda z: discrete_laplace_pmf(z, scale)
        tv = tv_to(empirical_pmf(zs), exact_l, support)
        floor = reference_tv(rng, exact_l, support, n)
        print(
            f"  scale={t}/{s}: eps for sensitivity 1={s / t:.4f}  mean={mean:.4f}  var={var:.4f} "
            f"(exact {discrete_laplace_variance(scale):.4f}, continuous {2 * scale**2:.4f})  "
            f"TV to exact={tv:.4f} (noise floor {floor:.4f})  draws per sample={src.draws / n:.2f}  "
            f"loop passes={discrete_laplace_iterations(s, t):.3f}"
        )
    print(f"  Pr[Z=0] at scale 1: {discrete_laplace_pmf(0, 1.0):.4f}")

    print("[Discrete Gaussian: privacy]")
    for sigma in (0.5, 1.0, 2.0):
        renyis = "  ".join(
            f"D_{alpha:g}={discrete_gaussian_renyi(alpha, sigma, 1):.4f}/{alpha / (2 * sigma**2):.4f}"
            for alpha in (2.0, 2.5, 5.0)
        )
        print(f"  sigma={sigma:g}, shift 1: {renyis}")
    for sigma in (0.5, 1.0):
        shifts = [i / 20 for i in range(21)]
        print(
            f"  sigma={sigma:g}: normalizer at c=0 {theta_shift(0.0, sigma):.6f}  "
            f"at c=0.5 {theta_shift(0.5, sigma):.6f}  max over c at c="
            f"{max(shifts, key=lambda c: theta_shift(c, sigma)):.2f}"
        )
    for sigma in (0.5, 1.0, 2.0):
        print(
            f"  sigma={sigma:g}, eps=1: delta discrete={discrete_gaussian_delta(1.0, sigma):.4e}  "
            f"continuous={gaussian_delta(1.0, sigma):.4e}"
        )

    print("[Discrete Gaussian: variance]")
    for sigma in (0.25, 0.5, 1.0, 2.0):
        exact = discrete_gaussian_variance(sigma)
        print(
            f"  sigma={sigma:g}: Var/sigma^2 summation={exact / sigma**2:.6f}  "
            f"Poisson={discrete_gaussian_variance_poisson(sigma) / sigma**2:.6f}  "
            f"1 - Var/sigma^2={1 - exact / sigma**2:.2e}"
        )

    print("[Algorithm 3: discrete Gaussian sampler]")
    for sigma2 in (Fraction(1, 4), Fraction(1), Fraction(4), Fraction(100)):
        sigma = math.sqrt(float(sigma2))
        src.draws = 0
        xs = [discrete_gaussian(sigma2, src) for _ in range(n)]
        mean, var = mean_var(xs)
        pmf = discrete_gaussian_pmf(sigma)
        exact_g: Callable[[int], float] = lambda x: pmf.get(x, 0.0)
        tv = tv_to(empirical_pmf(xs), exact_g, discrete_gaussian_support(sigma))
        floor = reference_tv(rng, exact_g, discrete_gaussian_support(sigma), n)
        print(
            f"  sigma^2={sigma2}: t={floor_sqrt(sigma2) + 1}  mean={mean:.4f}  var={var:.4f} "
            f"(exact {discrete_gaussian_variance(sigma):.4f})  TV to exact={tv:.4f} (noise floor {floor:.4f})  "
            f"acceptance={gaussian_acceptance(sigma2):.4f}  draws per sample={src.draws / n:.1f}"
        )


# --- Figures ----------------------------------------------------------------------------------------


def plot_float_support() -> Figure:
    p = 12
    m0, m1 = float_laplace_pmf(0.0, 1.0, p), float_laplace_pmf(1.0, 1.0, p)
    fig, (left, right) = plt.subplots(1, 2, figsize=(12, 4))
    lo, hi = 0.25, 0.2525
    for dist, other, y0, color, label in ((m0, m1, 1.0, "C0", "M(D): f = 0"), (m1, m0, 0.0, "C1", "M(D'): f = 1")):
        ys = support_in(dist, lo, hi)
        shared = [y for y in ys if y in other]
        only = [y for y in ys if y not in other]
        left.vlines(only, y0, y0 + 0.8, color=color, linewidth=1.2, label=f"{label}, only here")
        left.vlines(shared, y0, y0 + 0.8, color="k", linewidth=1.2)
    left.plot([], [], "k", label="reachable from both")
    left.set(
        xlim=(lo, hi),
        ylim=(-0.2, 2.6),
        yticks=[0.4, 1.4],
        yticklabels=["M(D')", "M(D)"],
        xlabel="output y",
        title=f"Outputs the {p}-bit sampler can produce in [{lo}, {hi}]",
    )
    left.legend(fontsize=8, loc="upper right", ncol=1)
    left.ticklabel_format(axis="x", useOffset=False)
    ps = list(range(6, 15))
    right.plot(
        ps,
        [mass_outside(float_laplace_pmf(0.0, 1.0, q), float_laplace_pmf(1.0, 1.0, q)) for q in ps],
        "o-",
        label="textbook sampler",
    )
    right.plot(
        ps,
        [
            mass_outside(float_laplace_pmf(0.0, 1.0, q, 1.0, 10.0), float_laplace_pmf(1.0, 1.0, q, 1.0, 10.0))
            for q in ps
        ],
        "s-",
        label="snapped to integers, clamped to ±10",
    )
    right.set(
        xlabel="significand bits p",
        ylabel="Pr[M(D) lands where M(D') cannot]",
        ylim=(-0.03, 1.0),
        title="Mass that identifies f(D) = 0 with certainty",
    )
    right.legend(fontsize=9)
    return fig


def plot_bernoulli_exp(rng: np.random.Generator) -> Figure:
    src = seeded_source(rng)
    fig, (left, right) = plt.subplots(1, 2, figsize=(12, 4))
    gamma = Fraction(7, 10)
    ks = first_zero_counts(gamma, 5000, src)
    emp = empirical_pmf(ks)
    kk = list(range(1, 7))
    colors = ["C0" if k % 2 else "C3" for k in kk]
    left.bar(kk, [first_zero_pmf(0.7, k) for k in kk], color=colors, alpha=0.5, label="exact Pr[K = k]")
    left.plot(kk, [emp.get(k, 0.0) for k in kk], "ko", label="5,000 runs")
    left.bar([0], [0], color="C0", alpha=0.5, label=f"odd K: output 1, total $e^{{-0.7}}$ = {math.exp(-0.7):.3f}")
    left.bar([0], [0], color="C3", alpha=0.5, label="even K: output 0")
    left.set(
        xlabel="K, index of the first Bernoulli(γ/k) that is 0",
        ylabel="probability",
        xticks=kk,
        xlim=(0.4, 6.6),
        title=r"Algorithm 1 for $\gamma$ = 0.7: $\Pr[K > k] = \gamma^k / k!$",
    )
    left.legend(fontsize=8)
    gammas = [Fraction(i, 4) for i in range(17)]
    draws = 2000
    rates = [sum(bernoulli_exp(g, src) for _ in range(draws)) / draws for g in gammas]
    grid = [i / 100 for i in range(401)]
    right.plot(grid, [math.exp(-g) for g in grid], "k", label=r"$e^{-\gamma}$")
    right.plot([float(g) for g in gammas], rates, "o", color="C0", label=f"Algorithm 1, {draws:,} draws per γ")
    right2 = right.twinx()
    right2.plot(grid, [bernoulli_exp_cost(g) for g in grid], "C2--", label="expected uniform draws")
    right2.set_ylabel("expected uniform draws per sample", color="C2")
    right2.set_ylim(0, 4.5)
    right.set(
        xlabel=r"$\gamma$", ylabel="Pr[output 1]", ylim=(0, 1.05), title=r"Exact Bernoulli($e^{-\gamma}$) and its cost"
    )
    handles = right.get_legend_handles_labels()[0] + right2.get_legend_handles_labels()[0]
    labels = right.get_legend_handles_labels()[1] + right2.get_legend_handles_labels()[1]
    right.legend(handles, labels, fontsize=9, loc="center right")
    return fig


def plot_discrete_laplace(rng: np.random.Generator) -> Figure:
    src = seeded_source(rng)
    s, t = 2, 3
    scale = t / s
    zs = [discrete_laplace(s, t, src) for _ in range(10_000)]
    emp = empirical_pmf(zs)
    fig, (left, right) = plt.subplots(1, 2, figsize=(12, 4))
    xs = list(range(-10, 11))
    left.bar(xs, [discrete_laplace_pmf(x, scale) for x in xs], color="C0", alpha=0.5, label="exact PMF")
    left.plot(xs, [emp.get(x, 0.0) for x in xs], "ko", markersize=4, label="10,000 draws of Algorithm 2")
    grid = np.linspace(-10, 10, 401)
    left.plot(grid, np.exp(-np.abs(grid) / scale) / (2 * scale), "C1", label=f"Lap({scale:g}) density")
    left.set(
        yscale="log", ylim=(1e-5, 1), xlabel="z", ylabel="probability", title=f"Discrete Laplace, scale t/s = {t}/{s}"
    )
    left.legend(fontsize=8, loc="lower center")
    ratio = [math.log(discrete_laplace_pmf(x, scale) / discrete_laplace_pmf(x - 1, scale)) for x in xs]
    right.step(xs, ratio, where="mid", color="C0", label=r"$\ln \Pr[Z + 0 = y] / \Pr[Z + 1 = y]$")
    right.axhline(1 / scale, color="k", linestyle=":", linewidth=1)
    right.axhline(-1 / scale, color="k", linestyle=":", linewidth=1)
    right.text(-9.5, 1 / scale + 0.05, rf"$\pm\varepsilon = \pm s/t = \pm{1 / scale:.3f}$", fontsize=9)
    right.set(
        xlabel="output y",
        ylabel="privacy loss (nats)",
        ylim=(-1.2, 1.2),
        title="Counts f(D) = 0 and f(D') = 1: the loss is bounded by s/t",
    )
    right.legend(fontsize=9, loc="lower right")
    return fig


def plot_discrete_gaussian(rng: np.random.Generator) -> Figure:
    src = seeded_source(rng)
    sigma2 = Fraction(4)
    sigma = 2.0
    t = floor_sqrt(sigma2) + 1
    xs_samples = [discrete_gaussian(sigma2, src) for _ in range(10_000)]
    emp = empirical_pmf(xs_samples)
    pmf = discrete_gaussian_pmf(sigma)
    fig, (left, right) = plt.subplots(1, 2, figsize=(12, 4))
    xs = list(range(-10, 11))
    accept = gaussian_acceptance(sigma2)
    proposal = [discrete_laplace_pmf(x, t) for x in xs]
    left.bar(xs, proposal, color="0.8", label=f"proposal: discrete Laplace, t = {t}")
    left.bar(xs, [pmf[x] * accept for x in xs], color="C0", alpha=0.7, label=f"accepted part (acceptance {accept:.3f})")
    left.set(xlabel="y", ylabel="probability", title=r"Algorithm 3 for $\sigma^2 = 4$: propose, then thin")
    left.legend(fontsize=8)
    right.bar(xs, [pmf[x] for x in xs], color="C0", alpha=0.5, label="exact discrete Gaussian, σ² = 4")
    right.plot(xs, [emp.get(x, 0.0) for x in xs], "ko", markersize=4, label="10,000 draws of Algorithm 3")
    grid = np.linspace(-10, 10, 401)
    right.plot(
        grid, np.exp(-(grid**2) / (2 * sigma**2)) / (sigma * math.sqrt(2 * math.pi)), "C1", label="N(0, 4) density"
    )
    right.set(yscale="log", ylim=(1e-6, 0.5), xlabel="x", ylabel="probability", title="Output against the exact PMF")
    right.legend(fontsize=8, loc="lower center")
    return fig


def plot_gaussian_variance() -> Figure:
    sigmas = [0.05 * i for i in range(2, 31)]
    fig, (left, right) = plt.subplots(1, 2, figsize=(12, 4))
    left.plot(
        sigmas,
        [discrete_gaussian_variance(s) / s**2 for s in sigmas],
        "C0",
        linewidth=2,
        label="discrete, by summation",
    )
    left.plot(
        sigmas,
        [discrete_gaussian_variance_poisson(s) / s**2 for s in sigmas],
        "k:",
        label="discrete, Poisson-summation formula",
    )
    left.axhline(1.0, color="C1", linestyle="--", label="continuous: Var = σ²")
    left.set(
        xlabel="σ", ylabel=r"Var / $\sigma^2$", ylim=(0, 1.1), title="Variance of the discrete Gaussian relative to σ²"
    )
    left.legend(fontsize=9, loc="lower right")
    gaps = [discrete_gaussian_variance_deficit(s) for s in sigmas]
    right.semilogy(sigmas, gaps, "C0", linewidth=2, label=r"$1 - \mathrm{Var}/\sigma^2$")
    right.semilogy(
        sigmas,
        [8 * math.pi**2 * s**2 * math.exp(-2 * math.pi**2 * s**2) for s in sigmas],
        "k:",
        label=r"leading term $8\pi^2\sigma^2 e^{-2\pi^2\sigma^2}$",
    )
    right.set(
        xlabel="σ",
        ylabel="relative variance deficit",
        ylim=(1e-18, 2),
        title="The deficit vanishes like a Gaussian in σ",
    )
    right.legend(fontsize=9)
    return fig


def plot_gaussian_delta() -> Figure:
    epss = [i / 50 for i in range(201)]
    fig, ax = plt.subplots(figsize=(7, 4.2))
    for sigma, color in ((0.5, "C0"), (1.0, "C1"), (2.0, "C2")):
        ax.semilogy(
            epss,
            [max(gaussian_delta(e, sigma), 1e-300) for e in epss],
            color=color,
            linestyle="--",
            label=f"continuous, σ = {sigma:g}",
        )
        ax.semilogy(
            epss,
            [max(discrete_gaussian_delta(e, sigma), 1e-300) for e in epss],
            color=color,
            label=f"discrete, σ = {sigma:g}",
        )
    ax.set(
        xlabel="ε",
        ylabel="δ(ε)",
        ylim=(1e-8, 1),
        xlim=(0, 4),
        title="Discrete against continuous Gaussian, sensitivity 1",
    )
    ax.legend(fontsize=8, ncol=2)
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
        "discrete_float_support.svg": plot_float_support(),
        "discrete_bernoulli_exp.svg": plot_bernoulli_exp(rng),
        "discrete_laplace.svg": plot_discrete_laplace(rng),
        "discrete_gaussian_sampler.svg": plot_discrete_gaussian(rng),
        "discrete_gaussian_variance.svg": plot_gaussian_variance(),
        "discrete_gaussian_delta.svg": plot_gaussian_delta(),
    }
    for name, fig in figures.items():
        fig.tight_layout()
        fig.savefig(out / name, metadata={"Date": None})  # no timestamp, so output is reproducible
        plt.close(fig)
        print(f"saved {out / name}")


if __name__ == "__main__":
    main()
