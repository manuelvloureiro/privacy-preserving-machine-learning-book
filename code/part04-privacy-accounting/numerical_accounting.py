"""Check the numerical accountant of the privacy-loss-distribution chapter and draw its figures.

    uv run python code/part04-privacy-accounting/numerical_accounting.py
    uv run python code/part04-privacy-accounting/numerical_accounting.py --seed 1 --out /tmp/figures

Prints the numbers quoted in 06-numerical-accounting.md (they match exactly with the default seed),
then saves the chapter's figures.

Conventions: a pair of output distributions (P, Q) has densities P(y), Q(y) (lowercase q is the
sampling rate here). The privacy loss is L(y) = ln(P(y)/Q(y)) and the privacy loss distribution
(PLD) is the law of L(Y) for Y ~ P. A discretized PLD puts mass on the grid {i h} plus an atom at
+∞; δ(ε) = E[(1 - e^{ε - L})_+] with the atom at +∞ counting fully. The subsampled Gaussian has
sensitivity 1, noise multiplier σ and Poisson sampling rate q. Its two directions are
    remove: P = (1 - q) N(0, σ²) + q N(1, σ²),  Q = N(0, σ²)
    add:    P = N(0, σ²),  Q = (1 - q) N(0, σ²) + q N(1, σ²)
and the add/remove guarantee is the larger δ of the two. Logs are natural.

Grids: the running example (q = 0.01, σ = 1) uses h = 1e-5 and tails truncated at 1e-15 per
side; the whole script runs in half a minute to a few minutes, depending on the machine.
"""

import argparse
import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from functools import cache
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
matplotlib.rcParams["svg.hashsalt"] = "numerical_accounting"  # stable element ids, so reruns don't churn the SVGs
matplotlib.rcParams["svg.fonttype"] = "none"  # keep text as text: smaller, diffable files

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from numpy.typing import NDArray

# Figures are build output: _build/figures/<part>/, which the build copies into each book.
FIGURES = Path(__file__).resolve().parents[2] / "_build" / "figures" / Path(__file__).parent.name

Array = NDArray[np.float64]

# The running example of the part: DP-SGD with Poisson rate q, noise multiplier σ, target δ.
Q_RATE = 0.01
SIGMA = 1.0
DELTA = 1e-5
STEPS = (1, 10, 100, 1000, 10_000)
WIDTH_HS = (1e-3, 1e-4)  # grid widths for the sandwich-width table
KS = (1, 3, 10, 30, 100, 300, 1000, 3000, 10_000)  # the ε-against-k figure; contains STEPS
H_DEFAULT = 1e-5  # loss grid width for the running example
TAIL = 1e-15  # probability mass truncated on each side of a PLD (moved safely, see truncate())
# Losses outside [LOSS_FLOOR, LOSS_CEILING] are cut in truncate() whatever their mass, on the safe side.
# A Gaussian PLD N(μ²/2, μ²) has only about TAIL of mass below μ²/2 - 8μ ≥ -32, so the floor sits far below
# the bulk of every PLD in this chapter. Moving mass from ℓ > LOSS_CEILING to +∞ (or down to the ceiling)
# changes δ(ε) by at most e^{ε - LOSS_CEILING} times that mass: nothing, for any ε below about 170.
LOSS_FLOOR = -100.0
LOSS_CEILING = 200.0

# The RDP orders shared by every chapter of the part.
ORDERS: list[float] = [1 + x / 10 for x in range(1, 100)] + [float(a) for a in range(12, 64)] + [128.0, 256.0, 512.0]


def normal_cdf(x: float) -> float:
    """Φ(x); erfc keeps full relative precision in the lower tail."""
    return 0.5 * math.erfc(-x / math.sqrt(2))


def normal_sf(x: float) -> float:
    """1 - Φ(x), accurate in the upper tail."""
    return 0.5 * math.erfc(x / math.sqrt(2))


def normal_cdf_array(x: Array) -> Array:
    return np.array([normal_cdf(float(v)) for v in x], dtype=np.float64)


def normal_sf_array(x: Array) -> Array:
    return np.array([normal_sf(float(v)) for v in x], dtype=np.float64)


def bisect(f: Callable[[float], float], lo: float, hi: float, steps: int = 100) -> float:
    """A root of an increasing-or-decreasing f on [lo, hi] with f(lo), f(hi) of opposite signs."""
    flo = f(lo)
    for _ in range(steps):
        mid = (lo + hi) / 2
        if (f(mid) > 0) == (flo > 0):
            lo, flo = mid, f(mid)
        else:
            hi = mid
    return (lo + hi) / 2


# --- Section 32.3: the privacy loss distribution and δ(ε) ------------------------------------------


def gaussian_pld_delta(eps: float, mu: float) -> float:
    """δ(ε) of the Gaussian pair with μ = Δ/σ, from L ~ N(μ²/2, μ²): Φ(μ/2 - ε/μ) - e^ε Φ(-μ/2 - ε/μ)."""
    return normal_cdf(mu / 2 - eps / mu) - math.exp(eps) * normal_cdf(-mu / 2 - eps / mu)


def gaussian_pld_delta_quadrature(eps: float, mu: float, n: int = 200_001) -> float:
    """The same δ(ε), as E[(1 - e^{ε - L})_+] integrated numerically over L ~ N(μ²/2, μ²)."""
    mean, sd = mu**2 / 2, mu
    ell = np.linspace(max(eps, mean - 40 * sd), mean + 40 * sd, n)
    density = np.exp(-((ell - mean) ** 2) / (2 * sd**2)) / (sd * math.sqrt(2 * math.pi))
    integrand = density * (1 - np.exp(eps - ell))
    return float(np.sum((integrand[1:] + integrand[:-1]) / 2 * np.diff(ell)))


def gaussian_epsilon(mu: float, delta: float) -> float:
    """Smallest ε ≥ 0 with δ(ε) ≤ delta for the Gaussian pair (δ(ε) decreases in ε)."""
    if gaussian_pld_delta(0.0, mu) <= delta:
        return 0.0
    hi = 1.0
    while gaussian_pld_delta(hi, mu) > delta:
        hi *= 2
    return bisect(lambda e: gaussian_pld_delta(e, mu) - delta, 0.0, hi)


def mc_gaussian_loss_moments(rng: np.random.Generator, mu: float, m: int) -> tuple[float, float, float]:
    """Sample Y ~ N(0, 1), compute L = ln(P/Q) for P = N(0,1), Q = N(μ,1); return mean, variance, E[e^{-L}]."""
    y = rng.standard_normal(m)
    loss = (mu**2 - 2 * mu * y) / 2  # ln φ(y)/φ(y - μ) = (μ² - 2μy)/2
    return float(loss.mean()), float(loss.var()), float(np.exp(-loss).mean())


def mc_gaussian_delta(rng: np.random.Generator, eps: float, mu: float, m: int) -> float:
    """Monte Carlo δ(ε) = E_P[(1 - e^{ε - L})_+] for the Gaussian pair."""
    y = rng.standard_normal(m)
    loss = (mu**2 - 2 * mu * y) / 2
    return float(np.mean(np.maximum(1 - np.exp(eps - loss), 0.0)))


def two_sided_delta(p: Sequence[float], q: Sequence[float], eps: float) -> tuple[float, float]:
    """δ(ε) of a finite pair two ways: Σ (P - e^ε Q)_+ and P[L > ε] - e^ε Q[L > ε] via the PLD."""
    direct = sum(max(pi - math.exp(eps) * qi, 0.0) for pi, qi in zip(p, q, strict=True))
    losses = [math.log(pi / qi) for pi, qi in zip(p, q, strict=True)]
    p_tail = sum(pi for pi, ell in zip(p, losses, strict=True) if ell > eps)
    q_tail = sum(pi * math.exp(-ell) for pi, ell in zip(p, losses, strict=True) if ell > eps)  # Q = e^{-L} P
    return direct, p_tail - math.exp(eps) * q_tail


# --- Discretized PLDs ------------------------------------------------------------------------


@dataclass(frozen=True)
class PLD:
    """A discretized privacy loss distribution: mass[i] sits at loss (offset + i) * h, inf_mass at +∞.

    `pessimistic` records the rounding direction, which decides how tails are truncated.
    """

    offset: int
    h: float
    mass: Array
    inf_mass: float
    pessimistic: bool

    def losses(self) -> Array:
        return (self.offset + np.arange(self.mass.size, dtype=np.float64)) * self.h


def pld_delta(pld: PLD, eps: float) -> float:
    """δ(ε) = Σ_i m_i (1 - e^{ε - ℓ_i})_+ + m_∞."""
    ell = pld.losses()
    above = ell > eps
    return float(np.sum(pld.mass[above] * -np.expm1(eps - ell[above]))) + pld.inf_mass


def pld_epsilon(pld: PLD, delta: float) -> float:
    """Smallest ε ≥ 0 with δ(ε) ≤ delta, solved exactly between grid points.

    For ε in [ℓ_{j-1}, ℓ_j), δ(ε) = m_∞ + A_j - e^ε B_j with A_j = Σ_{i≥j} m_i, B_j = Σ_{i≥j} m_i e^{-ℓ_i}.
    Only bins with ℓ_i > 0 can contribute to δ(ε) at ε ≥ 0, so the sums run over those bins alone. This keeps
    e^{-ℓ_i} ≤ 1: on a grid reaching below ℓ ≈ -709, e^{-ℓ_i} would overflow and make B_j infinite or NaN.
    """
    if pld.inf_mass >= delta:
        return math.inf
    ell = pld.losses()
    first = int(np.searchsorted(ell, 0.0, side="right"))  # ell[first:] > 0
    ell, m = ell[first:], pld.mass[first:]
    if ell.size == 0:
        return 0.0  # δ(0) = m_∞ < delta
    a = np.cumsum(m[::-1])[::-1]  # A_j
    b = np.cumsum((m * np.exp(-ell))[::-1])[::-1]  # B_j
    if pld.inf_mass + float(a[0]) - float(b[0]) <= delta:
        return 0.0  # δ(0) ≤ delta
    # δ(ℓ_t) = m_∞ + A_{t+1} - e^{ℓ_t} B_{t+1} decreases in t. The root lies in (ℓ_{t-1}, ℓ_t] for the first
    # t with δ(ℓ_t) ≤ delta (t = n - 1 if none: δ(ℓ_{n-1}) = m_∞ < delta), where δ(ε) = m_∞ + A_t - e^ε B_t.
    # For t = 0 the interval is (0, ℓ_0]: δ(0) > delta was checked above.
    at_grid = pld.inf_mass + a[1:] - np.exp(ell[:-1]) * b[1:]
    below = np.nonzero(at_grid <= delta)[0]
    t = int(below[0]) if below.size else ell.size - 1
    eps = math.log((pld.inf_mass + float(a[t]) - delta) / float(b[t]))
    return max(eps, 0.0)


def discretize_from_survival(
    sf: Callable[[Array], Array], cdf: Callable[[Array], Array], lo: float, hi: float, h: float, pessimistic: bool
) -> PLD:
    """Discretize a continuous loss with survival function sf and CDF cdf onto the grid {i h}.

    Mass below lo and above hi is at most TAIL. Pessimistic: losses in ((i-1)h, ih] go to ih, mass
    below the grid joins the lowest bin, mass above hi goes to +∞. Optimistic: losses in [ih, (i+1)h)
    go to ih, mass above the grid joins the top bin, mass below lo is dropped.
    """
    if pessimistic:
        i0, i1 = math.ceil(lo / h), math.ceil(hi / h)
        edges = np.arange(i0 - 1, i1 + 1, dtype=np.float64) * h  # bin i covers (edges[i - i0], edges[i - i0 + 1]]
    else:
        i0, i1 = math.floor(lo / h), math.floor(hi / h)
        edges = np.arange(i0, i1 + 2, dtype=np.float64) * h  # bin i covers [edges[i - i0], edges[i - i0 + 1])
    s, c = sf(edges), cdf(edges)
    # Interval masses from whichever of sf and cdf is the smaller, to keep relative precision in both tails.
    mass = np.where(s[:-1] < 0.5, s[:-1] - s[1:], c[1:] - c[:-1])
    mass = np.maximum(mass, 0.0)
    if pessimistic:
        mass[0] += c[0]  # everything below the lowest edge is rounded up into the lowest bin
        return PLD(i0, h, mass, float(s[-1]), True)
    mass[-1] += s[-1]  # everything above the top edge is rounded down into the top bin
    return PLD(i0, h, mass, 0.0, False)


def discretize_points(losses: Sequence[float], probs: Sequence[float], h: float, pessimistic: bool) -> PLD:
    """Round a finitely supported PLD up (pessimistic) or down (optimistic) to the grid {i h}."""
    idx = [math.ceil(ell / h - 1e-12) if pessimistic else math.floor(ell / h + 1e-12) for ell in losses]
    i0 = min(idx)
    mass = np.zeros(max(idx) - i0 + 1)
    for i, p in zip(idx, probs, strict=True):
        mass[i - i0] += p
    return PLD(i0, h, mass, 0.0, pessimistic)


def gaussian_pld(mu: float, h: float, pessimistic: bool) -> PLD:
    """Discretized PLD of the Gaussian pair: L ~ N(μ²/2, μ²)."""
    mean, sd = mu**2 / 2, mu
    z = -bisect(lambda t: math.log(normal_cdf(t)) - math.log(TAIL), -30.0, 0.0)  # Φ(-z) = TAIL
    return discretize_from_survival(
        lambda x: normal_sf_array((x - mean) / sd),
        lambda x: normal_cdf_array((x - mean) / sd),
        mean - z * sd,
        mean + z * sd,
        h,
        pessimistic,
    )


def validity(pld: PLD) -> float:
    """E[e^{-L}] over the finite part: the Q-mass the PLD accounts for. A valid PLD has it ≤ 1."""
    return float(np.sum(pld.mass * np.exp(-pld.losses())))


# --- Sections 32.4 and 32.6: composition by FFT convolution --------------------------------------------


def truncate(pld: PLD, tail: float = TAIL) -> PLD:
    """Drop negligible tails without breaking the rounding direction.

    Pessimistic: the top tail moves to +∞ and the bottom tail up into the lowest kept bin.
    Optimistic: the top tail moves down into the highest kept bin and the bottom tail is dropped.
    Both moves only raise (pessimistic) or only lower (optimistic) δ(ε) for every ε.

    The tails also include everything below LOSS_FLOOR and above LOSS_CEILING, whatever its mass. FFT
    round-off leaves positive noise of about 1e-17 times the largest mass in every bin, and over millions of
    bins it adds up to more than `tail`. For a sharply peaked PLD (q = 0.001, σ = 0.8, say) the noise alone
    then keeps the ends of the grid from ever being cut, and over many squarings the grid grows towards the
    full range k(n - 1): down to losses of several thousand below zero, where e^{-ℓ} overflows, and to
    hundreds of millions of bins.
    Cutting at fixed losses is the same safe move as cutting a tail.
    """
    m = np.maximum(pld.mass, 0.0)  # FFT round-off can leave tiny negative values
    low = np.cumsum(m)
    high = np.cumsum(m[::-1])[::-1]
    start = int(np.searchsorted(low, tail, side="right"))  # m[:start] sums to ≤ tail
    stop = m.size - int(np.searchsorted(np.cumsum(m[::-1]), tail, side="right"))  # m[stop:] sums to ≤ tail
    start = min(max(start, math.ceil(LOSS_FLOOR / pld.h) - pld.offset), m.size - 1)  # and losses below the floor
    stop = max(min(stop, math.floor(LOSS_CEILING / pld.h) - pld.offset + 1), 1)  # and losses above the ceiling
    start, stop = min(start, stop - 1), max(stop, start + 1)
    kept = m[start:stop].copy()
    bottom = float(low[start - 1]) if start > 0 else 0.0
    top = float(high[stop]) if stop < m.size else 0.0
    if pld.pessimistic:
        kept[0] += bottom
        return PLD(pld.offset + start, pld.h, kept, pld.inf_mass + top, True)
    kept[-1] += top
    return PLD(pld.offset + start, pld.h, kept, pld.inf_mass, False)


def fft_size(n: int) -> int:
    """The smallest 2^a 3^b 5^c ≥ n: lengths the FFT handles fast, with less padding than a power of two."""
    best = 1 << (n - 1).bit_length()
    p5 = 1
    while p5 < best:
        p35 = p5
        while p35 < best:
            m = p35
            while m < n:
                m *= 2
            best = min(best, m)
            p35 *= 3
        p5 *= 5
    return best


def convolve(a: PLD, b: PLD) -> PLD:
    """The PLD of the product pair: linear convolution by FFT, zero-padded so nothing wraps around."""
    n = a.mass.size + b.mass.size - 1
    size = fft_size(n)
    mass = np.fft.irfft(np.fft.rfft(a.mass, size) * np.fft.rfft(b.mass, size), size)[:n]
    inf_mass = a.inf_mass + b.inf_mass - a.inf_mass * b.inf_mass
    return truncate(PLD(a.offset + b.offset, a.h, mass, inf_mass, a.pessimistic))


def compose_pld_many(pld: PLD, ks: Sequence[int]) -> dict[int, PLD]:
    """k-fold self-compositions for several k, sharing the squarings PLD^(2^j)."""
    powers = [pld]
    while 2 ** len(powers) <= max(ks):
        powers.append(convolve(powers[-1], powers[-1]))
    out: dict[int, PLD] = {}
    for k in ks:
        result: PLD | None = None
        for j, power in enumerate(powers):
            if k >> j & 1:
                result = power if result is None else convolve(result, power)
        assert result is not None
        out[k] = result
    return out


def compose_pld(pld: PLD, k: int) -> PLD:
    """k-fold self-composition: the k-th power of the transform, by repeated squaring with truncation."""
    return compose_pld_many(pld, [k])[k]


def compose_pld_direct(pld: PLD, k: int) -> PLD:
    """k-fold self-composition as one k-th power of the transform on a window long enough for no wraparound."""
    n = k * (pld.mass.size - 1) + 1
    size = fft_size(n)
    mass = np.fft.irfft(np.fft.rfft(pld.mass, size) ** k, size)[:n]
    return PLD(k * pld.offset, pld.h, np.maximum(mass, 0.0), 1 - (1 - pld.inf_mass) ** k, pld.pessimistic)


def compose_pld_circular(pld: PLD, k: int, size: int, start: int) -> PLD:
    """The k-th power of the transform on a circular window of `size` bins starting at grid index `start`.

    No padding: mass whose total loss falls outside [start h, (start + size) h) wraps around to the other end.
    """
    folded = np.bincount(np.arange(pld.mass.size) % size, weights=pld.mass, minlength=size)  # one step, mod size
    mass = np.fft.irfft(np.fft.rfft(folded) ** k, size)  # index i holds totals ≡ k offset + i (mod size)
    mass = np.roll(mass, (k * pld.offset - start) % size)
    return PLD(start, pld.h, np.maximum(mass, 0.0), 1 - (1 - pld.inf_mass) ** k, pld.pessimistic)


# --- Section 32.5: a tiny grid example (randomized response composed three times) -------------------


def rr_pld(eps0: float) -> tuple[list[float], list[float]]:
    """Randomized response with parameter ε0: loss +ε0 w.p. e^ε0/(1+e^ε0), -ε0 otherwise."""
    keep = math.exp(eps0) / (1 + math.exp(eps0))
    return [-eps0, eps0], [1 - keep, keep]


def rr_exact_delta(eps0: float, k: int, eps: float) -> float:
    """δ(ε) of k-fold randomized response from the binomial law of the number j of +ε0 losses."""
    keep = math.exp(eps0) / (1 + math.exp(eps0))
    total = 0.0
    for j in range(k + 1):
        ell = (2 * j - k) * eps0
        if ell > eps:
            total += math.comb(k, j) * keep**j * (1 - keep) ** (k - j) * -math.expm1(eps - ell)
    return total


# --- Section 32.7: the subsampled Gaussian -----------------------------------------------------------


def subsampled_loss(y: Array, q: float, sigma: float, direction: str) -> Array:
    """L(y) = ±ln((1 - q) + q e^{(2y - 1)/(2σ²)}): + for remove (Y ~ mixture), - for add (Y ~ N(0, σ²))."""
    inner = np.logaddexp(math.log1p(-q), math.log(q) + (2 * y - 1) / (2 * sigma**2))
    return inner if direction == "remove" else -inner


def loss_to_y(ell: Array, q: float, sigma: float) -> Array:
    """Inverse of ℓ = ln((1 - q) + q e^{(2y - 1)/(2σ²)}); -∞ where ℓ ≤ ln(1 - q)."""
    out = np.full(ell.shape, -np.inf)
    ok = ell > math.log1p(-q)
    out[ok] = sigma**2 * np.log((np.expm1(ell[ok]) + q) / q) + 0.5
    return out


def mixture_sf(y: Array, q: float, sigma: float) -> Array:
    return (1 - q) * normal_sf_array(y / sigma) + q * normal_sf_array((y - 1) / sigma)


def mixture_cdf(y: Array, q: float, sigma: float) -> Array:
    return (1 - q) * normal_cdf_array(y / sigma) + q * normal_cdf_array((y - 1) / sigma)


def subsampled_loss_sf_cdf(
    q: float, sigma: float, direction: str
) -> tuple[Callable[[Array], Array], Callable[[Array], Array]]:
    """Survival function and CDF of the loss in one direction, by inverting the monotone map y -> L(y)."""
    if direction == "remove":
        # L increasing in y and Y ~ mixture: P[L > ℓ] = P[Y > y(ℓ)].
        def sf(ell: Array) -> Array:
            return mixture_sf(loss_to_y(ell, q, sigma), q, sigma)

        def cdf(ell: Array) -> Array:
            return mixture_cdf(loss_to_y(ell, q, sigma), q, sigma)

        return sf, cdf

    # add: L(y) = -ℓ'(y) decreasing and Y ~ N(0, σ²): P[L > ℓ] = P[Y < y(-ℓ)].
    def sf_add(ell: Array) -> Array:
        return normal_cdf_array(loss_to_y(-ell, q, sigma) / sigma)

    def cdf_add(ell: Array) -> Array:
        return normal_sf_array(loss_to_y(-ell, q, sigma) / sigma)

    return sf_add, cdf_add


def subsampled_loss_range(q: float, sigma: float, direction: str) -> tuple[float, float]:
    """[lo, hi] outside which the loss has at most TAIL mass on each side."""
    sf, cdf = subsampled_loss_sf_cdf(q, sigma, direction)

    def log_sf(ell: float) -> float:
        return math.log(max(float(sf(np.array([ell]))[0]), 1e-300)) - math.log(TAIL)

    def log_cdf(ell: float) -> float:
        return math.log(max(float(cdf(np.array([ell]))[0]), 1e-300)) - math.log(TAIL)

    if direction == "remove":
        return math.log1p(-q), bisect(log_sf, 0.0, 200.0)
    return bisect(log_cdf, -200.0, 0.0), -math.log1p(-q)


def discretize_subsampled_gaussian_pld(
    q: float, sigma: float, h: float, pessimistic: bool, direction: str = "remove"
) -> PLD:
    """Discretized PLD of the Poisson-subsampled Gaussian in one direction ("remove" or "add")."""
    sf, cdf = subsampled_loss_sf_cdf(q, sigma, direction)
    lo, hi = subsampled_loss_range(q, sigma, direction)
    return discretize_from_survival(sf, cdf, lo, hi, h, pessimistic)


def subsampled_delta_exact(eps: float, q: float, sigma: float, direction: str) -> float:
    """δ(ε) of one subsampled-Gaussian step in closed form: P[L > ε] - e^ε Q[L > ε] through the threshold in y."""
    if direction == "remove":
        t = float(loss_to_y(np.array([eps]), q, sigma)[0])  # L > ε ⇔ y > t
        p_tail = float(mixture_sf(np.array([t]), q, sigma)[0])
        q_tail = normal_sf(t / sigma)
    else:
        t = float(loss_to_y(np.array([-eps]), q, sigma)[0])  # L > ε ⇔ y < t
        p_tail = normal_cdf(t / sigma)
        q_tail = float(mixture_cdf(np.array([t]), q, sigma)[0])
    return p_tail - math.exp(eps) * q_tail


@cache
def subsampled_runs(
    q: float, sigma: float, ks: tuple[int, ...], delta: float, h: float, pessimistic: bool, direction: str
) -> dict[int, tuple[float, int]]:
    """(ε at δ, number of bins of the composed PLD) after each k in ks for one direction.

    Cached: the table and the figures share it, and the composed PLDs themselves are not kept.
    """
    plds = compose_pld_many(discretize_subsampled_gaussian_pld(q, sigma, h, pessimistic, direction), ks)
    return {k: (pld_epsilon(pld, delta), pld.mass.size) for k, pld in plds.items()}


def subsampled_epsilons(
    q: float, sigma: float, ks: tuple[int, ...], delta: float, h: float, pessimistic: bool, direction: str
) -> dict[int, float]:
    """ε at δ after each k in ks for one direction."""
    return {k: eps for k, (eps, _) in subsampled_runs(q, sigma, ks, delta, h, pessimistic, direction).items()}


def subsampled_epsilon_exact_one_step(q: float, sigma: float, delta: float, direction: str) -> float:
    """ε at δ of a single subsampled-Gaussian step, by bisection on the closed-form δ(ε)."""
    if subsampled_delta_exact(0.0, q, sigma, direction) <= delta:
        return 0.0
    hi = 1.0
    while subsampled_delta_exact(hi, q, sigma, direction) > delta:
        hi *= 2
    return bisect(lambda e: subsampled_delta_exact(e, q, sigma, direction) - delta, 0.0, hi)


def pld_epsilon_subsampled_gaussian(
    q: float, sigma: float, steps: int, delta: float, h: float = H_DEFAULT, pessimistic: bool = True
) -> float:
    """ε of `steps` Poisson-subsampled Gaussian steps at δ: the larger of the remove and add directions.

    Returns math.inf when the mass truncated to +∞ alone exceeds δ (the pessimistic bound is then vacuous).
    Raises ValueError on a NaN: Python's max() would silently drop it and report the other direction.
    """
    eps = [subsampled_epsilons(q, sigma, (steps,), delta, h, pessimistic, d)[steps] for d in ("remove", "add")]
    if any(math.isnan(e) for e in eps):
        raise ValueError(f"epsilon is NaN (remove, add) = {eps} for q={q}, sigma={sigma}, steps={steps}, h={h}")
    return max(eps)


# --- Section 32.8: the accountants to compare against ----------------------------------------------


def logsumexp(values: Array) -> float:
    top = float(np.max(values))
    return top + math.log(float(np.sum(np.exp(values - top))))


def subsampled_gaussian_log_a(alpha: float, q: float, sigma: float, integrate: bool = False) -> float:
    """ln A_α, A_α = E_{y~N(0,σ²)}[((1 - q) + q e^{(2y - 1)/(2σ²)})^α].

    Integer α: the binomial expansion Σ_j C(α, j)(1 - q)^{α-j} q^j e^{(j² - j)/(2σ²)}.
    Fractional α: numerical integration on a fine grid in log space.
    """
    if float(alpha).is_integer() and not integrate:
        a = int(alpha)
        j = np.arange(a + 1, dtype=np.float64)
        log_binom = np.array([math.lgamma(a + 1) - math.lgamma(i + 1) - math.lgamma(a - i + 1) for i in range(a + 1)])
        terms = log_binom + (a - j) * math.log1p(-q) + j * math.log(q) + (j * j - j) / (2 * sigma**2)
        return logsumexp(terms)
    y = np.linspace(-30 * sigma, 30 * sigma + alpha, 400_001)
    log_f = -(y**2) / (2 * sigma**2) - math.log(sigma * math.sqrt(2 * math.pi))
    log_f = log_f + alpha * np.logaddexp(math.log1p(-q), math.log(q) + (2 * y - 1) / (2 * sigma**2))
    dy = float(y[1] - y[0])
    return logsumexp(log_f) + math.log(dy)


def rdp_epsilon(q: float, sigma: float, steps: int, delta: float) -> tuple[float, float]:
    """RDP accountant: ε(α) = k ln A_α/(α - 1), converted with ε(α) + ln((α-1)/α) - (ln δ + ln α)/(α - 1).

    Returns (ε, best order α) minimized over ORDERS.
    """
    best, best_alpha = math.inf, 0.0
    for alpha in ORDERS:
        rdp = steps * subsampled_gaussian_log_a(alpha, q, sigma) / (alpha - 1)
        eps = rdp + math.log((alpha - 1) / alpha) - (math.log(delta) + math.log(alpha)) / (alpha - 1)
        if eps < best:
            best, best_alpha = eps, alpha
    return max(best, 0.0), best_alpha


def clt_mu(q: float, sigma: float, steps: int) -> float:
    """GDP central-limit approximation for DP-SGD: μ = q √(k (e^{1/σ²} - 1))."""
    return q * math.sqrt(steps * math.expm1(1 / sigma**2))


def clt_epsilon(q: float, sigma: float, steps: int, delta: float) -> float:
    return gaussian_epsilon(clt_mu(q, sigma, steps), delta)


# --- The chapter's numbers ---------------------------------------------------------------------


def print_table(rng: np.random.Generator) -> None:
    mu = 1.0
    print(f"[Part 1: Gaussian PLD, mu = {mu:g}]")
    mean, var, e_neg = mc_gaussian_loss_moments(rng, mu, 1_000_000)
    print(f"  L ~ N(mu^2/2, mu^2) = N(0.5, 1): Monte Carlo mean={mean:.4f}  variance={var:.4f}  E[e^-L]={e_neg:.4f}")
    for eps in (0.0, 0.5, 1.0, 2.0):
        print(
            f"  eps={eps:g}: closed form delta={gaussian_pld_delta(eps, mu):.6f}  "
            f"quadrature={gaussian_pld_delta_quadrature(eps, mu):.6f}  "
            f"Monte Carlo={mc_gaussian_delta(rng, eps, mu, 1_000_000):.6f}"
        )
    print(f"  epsilon at delta=1e-5: {gaussian_epsilon(mu, DELTA):.4f}")
    p, qd = [0.1, 0.2, 0.3, 0.4], [0.4, 0.3, 0.2, 0.1]
    direct, via_pld = two_sided_delta(p, qd, 0.5)
    print(f"  four-point pair, eps=0.5: sum (P - e^eps Q)_+ = {direct:.4f}  P[L>eps] - e^eps Q[L>eps] = {via_pld:.4f}")

    print("[Part 2: k Gaussians compose to N(k mu^2/2, k mu^2)]")
    for k in (10, 100):
        exact = gaussian_epsilon(math.sqrt(k) * mu, DELTA)
        g_up = compose_pld(gaussian_pld(mu, 1e-3, True), k)
        g_dn = compose_pld(gaussian_pld(mu, 1e-3, False), k)
        print(
            f"  k={k}: exact epsilon={exact:.4f}  grid h=1e-3: pessimistic={pld_epsilon(g_up, DELTA):.4f}  "
            f"optimistic={pld_epsilon(g_dn, DELTA):.4f}"
        )
        ell = g_up.losses()
        mean_up = float(np.sum(ell * g_up.mass))
        var_up = float(np.sum((ell - mean_up) ** 2 * g_up.mass))
        print(
            f"    pessimistic PLD mean={mean_up:.4f}  variance={var_up:.4f}  (exact {k * mu**2 / 2:g}, {k * mu**2:g})"
        )

    print("[Part 3: randomized response eps0 = ln 3, composed 3 times, grid h = 0.5]")
    eps0, h_rr, k_rr = math.log(3), 0.5, 3
    losses, probs = rr_pld(eps0)
    print(f"  one step: losses {losses[0]:.4f}, {losses[1]:.4f} with probabilities {probs[0]:.4f}, {probs[1]:.4f}")
    up = discretize_points(losses, probs, h_rr, True)
    dn = discretize_points(losses, probs, h_rr, False)
    for name, pld in (("rounded up", up), ("rounded down", dn)):
        keep = pld.mass > 0
        print(
            f"  {name}: losses {[f'{x:g}' for x in pld.losses()[keep]]} masses {[f'{x:.4f}' for x in pld.mass[keep]]}  "
            f"E[e^-L]={validity(pld):.4f}"
        )
    up3, dn3 = compose_pld_direct(up, k_rr), compose_pld_direct(dn, k_rr)
    for j in range(k_rr + 1):
        exact_ell = (2 * j - k_rr) * eps0
        pr = math.comb(k_rr, j) * probs[1] ** j * probs[0] ** (k_rr - j)
        print(
            f"  j={j} (+eps0 count): prob={pr:.4f}  exact loss={exact_ell:.4f}  "
            f"pessimistic loss={j * 1.5 + (k_rr - j) * -1.0:g}  optimistic loss={j * 1.0 + (k_rr - j) * -1.5:g}"
        )
    for eps in (0.0, 1.0, 2.0):
        print(
            f"  eps={eps:g}: exact delta={rr_exact_delta(eps0, k_rr, eps):.4f}  "
            f"pessimistic={pld_delta(up3, eps):.4f}  optimistic={pld_delta(dn3, eps):.4f}"
        )
    print(f"  composed pessimistic support {[f'{x:g}' for x in up3.losses()[up3.mass > 1e-12]]}")
    print(f"  composed optimistic support {[f'{x:g}' for x in dn3.losses()[dn3.mass > 1e-12]]}")

    print("[Part 4: FFT composition checks]")
    g1 = gaussian_pld(mu, 1e-3, True)
    sq, direct10 = compose_pld(g1, 10), compose_pld_direct(g1, 10)
    print(
        f"  Gaussian mu=1, h=1e-3, k=10: squaring delta(1)={pld_delta(sq, 1.0):.8f}  "
        f"one-shot k-th power delta(1)={pld_delta(direct10, 1.0):.8f}  exact={gaussian_pld_delta(1.0, math.sqrt(10)):.8f}"
    )
    print(
        f"  bins: one step {g1.mass.size}, after k=10 {sq.mass.size} (ratio {sq.mass.size / g1.mass.size:.2f}, "
        f"sqrt(10)={math.sqrt(10):.2f}), one-shot window {direct10.mass.size}"
    )
    centre = 10 * mu**2 / 2  # mean of the composed loss
    for width in (10.0, 20.0, 40.0):
        size = round(width / 1e-3)
        circ = compose_pld_circular(g1, 10, size, round((centre - width / 2) / 1e-3))
        print(f"  circular window of width {width:g} around the mean (no padding): delta(1)={pld_delta(circ, 1.0):.8f}")
    sd10 = math.sqrt(10) * mu  # the composed loss is N(5, 10)
    print(
        f"  window [0, 10] for N(5, 10): mass below 0={normal_cdf(-centre / sd10):.4f}  "
        f"above 10={normal_sf(centre / sd10):.4f}  share of the mass above 10 that is above 11="
        f"{normal_sf((centre + 1) / sd10) / normal_sf(centre / sd10):.2f}"
    )

    print(f"[Part 5: subsampled Gaussian, q={Q_RATE:g}, sigma={SIGMA:g}]")
    for d in ("remove", "add"):
        lo, hi = subsampled_loss_range(Q_RATE, SIGMA, d)
        one = discretize_subsampled_gaussian_pld(Q_RATE, SIGMA, H_DEFAULT, True, d)
        print(
            f"  {d}: loss range [{lo:.4f}, {hi:.4f}]  bins at h=1e-5: {one.mass.size}  "
            f"E[e^-L]={validity(one):.6f}  inf mass={one.inf_mass:.1e}"
        )
        for eps in (0.0, 0.5, 1.0):
            print(
                f"    eps={eps:g}: exact delta={subsampled_delta_exact(eps, Q_RATE, SIGMA, d):.6e}  "
                f"pessimistic={pld_delta(one, eps):.6e}"
            )

    print(
        f"  one step, exact epsilon at delta=1e-5: remove {subsampled_epsilon_exact_one_step(Q_RATE, SIGMA, DELTA, 'remove'):.4f}  "
        f"add {subsampled_epsilon_exact_one_step(Q_RATE, SIGMA, DELTA, 'add'):.4f}"
    )
    for alpha in (2.0, 10.0):
        print(
            f"  RDP ln A_alpha at alpha={alpha:g}: binomial={subsampled_gaussian_log_a(alpha, Q_RATE, SIGMA):.8f}  "
            f"integration={subsampled_gaussian_log_a(alpha, Q_RATE, SIGMA, integrate=True):.8f}"
        )

    print(f"[Part 6: DP-SGD q={Q_RATE:g}, sigma={SIGMA:g}, delta={DELTA:g}, h={H_DEFAULT:g}]")
    eps_k = {
        (d, pess): subsampled_epsilons(Q_RATE, SIGMA, KS, DELTA, H_DEFAULT, pess, d)
        for d in ("remove", "add")
        for pess in (True, False)
    }
    for k in STEPS:
        r_up, r_dn = eps_k["remove", True][k], eps_k["remove", False][k]
        a_up, a_dn = eps_k["add", True][k], eps_k["add", False][k]
        upper, lower = max(r_up, a_up), max(r_dn, a_dn)
        rdp, alpha = rdp_epsilon(Q_RATE, SIGMA, k, DELTA)
        print(
            f"  k={k}: PLD upper={upper:.4f}  lower={lower:.4f}  width={upper - lower:.4f}  "
            f"(remove {r_up:.4f}/{r_dn:.4f}, add {a_up:.4f}/{a_dn:.4f})  "
            f"RDP={rdp:.4f} (alpha={alpha:g})  CLT={clt_epsilon(Q_RATE, SIGMA, k, DELTA):.4f} "
            f"(mu={clt_mu(Q_RATE, SIGMA, k):.4f})"
        )
        # Differences of the values as rounded in the table, so a reader can check them from the table.
        clt = clt_epsilon(Q_RATE, SIGMA, k, DELTA)
        print(
            f"    RDP minus PLD upper={round(rdp, 4) - round(upper, 4):.4f}  "
            f"PLD lower minus CLT={round(lower, 4) - round(clt, 4):.4f}  width={upper - lower:.1e}"
        )
    bins = {k: subsampled_runs(Q_RATE, SIGMA, KS, DELTA, H_DEFAULT, True, "remove")[k][1] for k in STEPS}
    print(f"  bins of the composed pessimistic remove PLD at h=1e-5: {bins}")

    print("[Part 6: sandwich width against grid size, k=1000]")
    for h in WIDTH_HS:
        up_e = pld_epsilon_subsampled_gaussian(Q_RATE, SIGMA, 1000, DELTA, h, True)
        dn_e = pld_epsilon_subsampled_gaussian(Q_RATE, SIGMA, 1000, DELTA, h, False)
        print(f"  h={h:g}: upper={up_e:.4f}  lower={dn_e:.4f}  width={up_e - dn_e:.4f}  k*h={1000 * h:.4f}")
    up_e = max(eps_k["remove", True][1000], eps_k["add", True][1000])
    dn_e = max(eps_k["remove", False][1000], eps_k["add", False][1000])
    print(f"  h={H_DEFAULT:g}: upper={up_e:.4f}  lower={dn_e:.4f}  width={up_e - dn_e:.4f}  k*h={1000 * H_DEFAULT:.4f}")
    sizes = [
        subsampled_runs(Q_RATE, SIGMA, ks, DELTA, h, True, "remove")[1000][1]
        for h, ks in ((1e-3, (1000,)), (1e-4, (1000,)), (H_DEFAULT, KS))
    ]
    print(
        f"  bins of the pessimistic remove PLD after k=1000 for h=1e-3, 1e-4, 1e-5: {sizes}  "
        f"ratios {sizes[1] / sizes[0]:.2f}, {sizes[2] / sizes[1]:.2f}"
    )


# --- Figures -----------------------------------------------------------------------------------


def plot_pld_weight(mu: float = 1.0, eps: float = 1.0) -> Figure:
    """The Gaussian PLD, the weight (1 - e^{ε-ℓ})_+ and their product, whose area is δ(ε)."""
    ell = np.linspace(-3.5, 4.5, 801)
    density = np.exp(-((ell - mu**2 / 2) ** 2) / (2 * mu**2)) / (mu * math.sqrt(2 * math.pi))
    weight = np.maximum(-np.expm1(eps - ell), 0.0)
    fig, ax = plt.subplots(figsize=(7, 3.8))
    ax.plot(ell, density, color="C0", label=r"PLD density, $L \sim N(\mu^2/2, \mu^2)$")
    ax.plot(ell, weight * 0.4, color="C1", ls="--", label=r"weight $(1 - e^{\varepsilon - \ell})_+$ (scaled by 0.4)")
    ax.fill_between(
        ell,
        density * weight,
        color="C2",
        alpha=0.4,
        label=rf"density $\times$ weight: area $= \delta(\varepsilon) = {gaussian_pld_delta(eps, mu):.4f}$",
    )
    ax.axvline(eps, color="grey", lw=0.8)
    ax.text(eps + 0.05, 0.42, r"$\varepsilon = 1$", color="grey")
    ax.set_xlabel(r"privacy loss $\ell$")
    ax.set_ylabel("density")
    ax.set_ylim(0, 0.47)
    ax.legend(loc="upper left", fontsize=8, frameon=False)
    return fig


def plot_rounding(eps0: float = math.log(3), h: float = 0.5, k: int = 3) -> Figure:
    """Exact, rounded-up and rounded-down PLDs of 3-fold randomized response, and their δ(ε) curves."""
    losses, probs = rr_pld(eps0)
    up = compose_pld_direct(discretize_points(losses, probs, h, True), k)
    dn = compose_pld_direct(discretize_points(losses, probs, h, False), k)
    exact_l = [(2 * j - k) * eps0 for j in range(k + 1)]
    exact_p = [math.comb(k, j) * probs[1] ** j * probs[0] ** (k - j) for j in range(k + 1)]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 3.6))
    ax1.stem(exact_l, exact_p, linefmt="k-", markerfmt="ko", basefmt=" ", label="exact")
    keep_up, keep_dn = up.mass > 1e-12, dn.mass > 1e-12
    ax1.stem(
        up.losses()[keep_up] + 0.04,
        up.mass[keep_up],
        linefmt="C3-",
        markerfmt="C3^",
        basefmt=" ",
        label="rounded up (pessimistic)",
    )
    ax1.stem(
        dn.losses()[keep_dn] - 0.04,
        dn.mass[keep_dn],
        linefmt="C0-",
        markerfmt="C0v",
        basefmt=" ",
        label="rounded down (optimistic)",
    )
    ax1.set_xticks(np.arange(-4.5, 5.0, 1.5))
    ax1.set_xlabel(r"total privacy loss after $k = 3$ steps")
    ax1.set_ylabel("probability")
    ax1.set_ylim(0, 0.55)
    ax1.legend(fontsize=7, frameon=False, loc="upper left")
    eps_grid = np.linspace(0, 4.5, 400)
    ax2.plot(eps_grid, [rr_exact_delta(eps0, k, float(e)) for e in eps_grid], "k", label="exact")
    ax2.plot(eps_grid, [pld_delta(up, float(e)) for e in eps_grid], "C3", label="pessimistic")
    ax2.plot(eps_grid, [pld_delta(dn, float(e)) for e in eps_grid], "C0", label="optimistic")
    ax2.set_xlabel(r"$\varepsilon$")
    ax2.set_ylabel(r"$\delta(\varepsilon)$")
    ax2.legend(fontsize=8, frameon=False)
    return fig


def plot_subsampled_loss(q: float = Q_RATE, sigma: float = SIGMA) -> Figure:
    """The loss maps y -> L(y) in both directions, and the densities of the outputs."""
    y = np.linspace(-4, 7, 600)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 3.6))
    n0 = np.exp(-(y**2) / (2 * sigma**2)) / (sigma * math.sqrt(2 * math.pi))
    n1 = np.exp(-((y - 1) ** 2) / (2 * sigma**2)) / (sigma * math.sqrt(2 * math.pi))
    ax1.plot(y, n0, "C0", label=r"$N(0, \sigma^2)$")
    ax1.plot(y, (1 - q) * n0 + q * n1, "C3--", label=r"mixture $(1-q)N(0,\sigma^2) + qN(1,\sigma^2)$")
    ax1.set_yscale("log")
    ax1.set_ylim(1e-12, 1)
    ax1.set_xlabel(r"output $y$")
    ax1.set_ylabel("density")
    ax1.legend(fontsize=7, frameon=False, loc="lower center")
    ax2.plot(y, subsampled_loss(y, q, sigma, "remove"), "C3", label=r"remove: $\ln((1-q) + q e^{(2y-1)/2\sigma^2})$")
    ax2.plot(y, subsampled_loss(y, q, sigma, "add"), "C0", label="add: minus the same")
    ax2.axhline(0, color="grey", lw=0.6)
    ax2.set_xlabel(r"output $y$")
    ax2.set_ylabel(r"privacy loss $L(y)$")
    ax2.legend(fontsize=7, frameon=False, loc="upper left")
    return fig


def plot_histograms(q: float = Q_RATE, sigma: float = SIGMA, h: float = 1e-4) -> Figure:
    """The pessimistic remove-direction PLD after 1, 10 and 100 steps, re-binned for display."""
    one = discretize_subsampled_gaussian_pld(q, sigma, h, True, "remove")
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.4))
    for ax, k, width in zip(axes, (1, 10, 100), (0.002, 0.005, 0.02), strict=True):
        pld = compose_pld(one, k)
        ell = pld.losses()
        edges = np.arange(math.floor(float(ell[0]) / width), math.ceil(float(ell[-1]) / width) + 1) * width
        hist, _ = np.histogram(ell, bins=edges, weights=pld.mass)
        centres = (edges[:-1] + edges[1:]) / 2
        ax.bar(centres, hist, width=width, color="C3", alpha=0.7)
        ax.set_yscale("log")
        ax.set_ylim(1e-12, 1)
        lim = {1: (-0.02, 0.1), 10: (-0.1, 0.4), 100: (-0.6, 1.2)}[k]
        ax.set_xlim(*lim)
        mean = float(np.sum(ell * pld.mass))
        ax.axvline(mean, color="k", lw=0.8, ls=":")
        ax.set_title(rf"$T = {k}$ (bins of {width:g})", fontsize=10)
        ax.set_xlabel(r"privacy loss $\ell$")
    axes[0].set_ylabel("probability per bin")
    return fig


def plot_epsilon_vs_k(q: float = Q_RATE, sigma: float = SIGMA, delta: float = DELTA) -> Figure:
    """ε at δ against k for the PLD bounds, the RDP accountant and the GDP central-limit approximation."""
    ks = list(KS)
    eps_k = {
        (d, pess): subsampled_epsilons(q, sigma, KS, delta, H_DEFAULT, pess, d)
        for d in ("remove", "add")
        for pess in (True, False)
    }
    upper = [max(eps_k["remove", True][k], eps_k["add", True][k]) for k in ks]
    lower = [max(eps_k["remove", False][k], eps_k["add", False][k]) for k in ks]
    rdp = [rdp_epsilon(q, sigma, k, delta)[0] for k in ks]
    clt = [clt_epsilon(q, sigma, k, delta) for k in ks]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(ks, rdp, "C2s-", ms=4, label="RDP (improved conversion)")
    ax.plot(ks, clt, "C1d-", ms=4, label="GDP central-limit approximation")
    ax.plot(ks, upper, "C3^-", ms=4, label="PLD pessimistic (upper bound)")
    ax.plot(ks, lower, "C0v--", ms=4, label="PLD optimistic (lower bound)")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"number of steps $T$")
    ax.set_ylabel(r"$\varepsilon$ at $\delta = 10^{-5}$")
    ax.set_title(rf"DP-SGD, $q = {q:g}$, $\sigma = {sigma:g}$", fontsize=10)
    ax.legend(fontsize=8, frameon=False)
    return fig


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=FIGURES, help="directory for the figures")
    args = parser.parse_args()
    seed: int = args.seed
    out: Path = args.out

    print_table(np.random.default_rng(seed))

    _ = np.random.default_rng(seed + 1)  # the figures are deterministic; the generator is kept for a uniform interface
    out.mkdir(parents=True, exist_ok=True)
    figures = {
        "numerical_accounting_pld_weight.svg": plot_pld_weight(),
        "numerical_accounting_rounding.svg": plot_rounding(),
        "numerical_accounting_subsampled_loss.svg": plot_subsampled_loss(),
        "numerical_accounting_histograms.svg": plot_histograms(),
        "numerical_accounting_epsilon_vs_k.svg": plot_epsilon_vs_k(),
    }
    for name, fig in figures.items():
        fig.tight_layout()
        fig.savefig(out / name, metadata={"Date": None})  # no timestamp, so output is reproducible
        plt.close(fig)
        print(f"saved {out / name}")


if __name__ == "__main__":
    main()
