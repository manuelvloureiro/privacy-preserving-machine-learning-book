"""Check the derivations of the appendix on composition of trade-off functions and draw its figures.

    uv run python code/appendix-derivations/apx_composition.py
    uv run python code/appendix-derivations/apx_composition.py --seed 1 --out /tmp/figures

Prints every number quoted in book-full/appendix-derivations/11-composition.md (it matches
exactly with the default seed), then saves the appendix's figures.

Conventions follow the f-DP chapter: a test fires ("the data was D'") with probability phi(y), its
type I error is a = E_P[phi] and its type II error is b = 1 - E_Q[phi] (α and β in the appendix,
which has no Rényi order); the privacy loss is
L = ln(p / q). In the dominating-pair section, as in the numerical-accounting chapter, a test fires
for P and E_gamma(P || Q) = sum (P - gamma Q)_+.
"""

import argparse
import math
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from functools import cache
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # for code/shared
from shared.figures import configure

configure("apx_composition")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from numpy.typing import NDArray

# Figures are build output: _build/figures/<part>/, which the build copies into each book.
FIGURES = Path(__file__).resolve().parents[2] / "_build" / "figures" / Path(__file__).parent.name

Array = NDArray[np.float64]
SQRT2 = math.sqrt(2.0)
BERRY_ESSEEN = 0.56  # Shevtsova (2010): the constant for sums of independent, non-identical terms


# --- Normal distribution --------------------------------------------------------------------


def normal_cdf(x: float) -> float:
    return 0.5 * math.erfc(-x / SQRT2)


def normal_sf(x: float) -> float:
    return 0.5 * math.erfc(x / SQRT2)


def normal_cdf_array(x: Array) -> Array:
    return np.fromiter((0.5 * math.erfc(-v / SQRT2) for v in x.tolist()), dtype=np.float64, count=x.size)


def normal_sf_array(x: Array) -> Array:
    return np.fromiter((0.5 * math.erfc(v / SQRT2) for v in x.tolist()), dtype=np.float64, count=x.size)


def normal_ppf(u: float) -> float:
    """Phi^{-1}(u) by bisection on the tail that keeps precision."""
    if u <= 0.0:
        return -math.inf
    if u >= 1.0:
        return math.inf
    lo, hi = -40.0, 40.0
    for _ in range(200):
        mid = (lo + hi) / 2
        if normal_cdf(mid) < u:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def gaussian_tradeoff(mu: float, a: float) -> float:
    """G_mu(a) = Phi(Phi^{-1}(1 - a) - mu), with G(a) = 1 for a <= 0 and 0 for a >= 1."""
    if a <= 0.0:
        return 1.0
    if a >= 1.0:
        return 0.0
    return normal_cdf(-normal_ppf(a) - mu)


def gdp_hockey(mu: float, gamma: float) -> float:
    """E_gamma(P || Q) for P = N(mu, 1), Q = N(0, 1), any gamma > 0 (the GDP profile at eps = ln gamma)."""
    eps = math.log(gamma)
    return normal_cdf(-eps / mu + mu / 2) - gamma * normal_cdf(-eps / mu - mu / 2)


def gdp_delta(mu: float, eps: float) -> float:
    return gdp_hockey(mu, math.exp(eps))


def solve_decreasing(g: Callable[[float], float], target: float, lo: float, hi: float) -> float:
    """The x in [lo, hi] where a non-increasing g crosses target, by bisection."""
    for _ in range(200):
        mid = (lo + hi) / 2
        if g(mid) > target:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def gdp_eps(mu: float, delta: float) -> float:
    return solve_decreasing(lambda e: gdp_delta(mu, e), delta, 0.0, 60.0)


# --- Trade-off curves of finite pairs ----------------------------------------------------------


@dataclass(frozen=True)
class Curve:
    """Vertices (a, b) of the trade-off curve of a finite pair, a non-decreasing from 0 to 1."""

    a: Array
    b: Array

    def at(self, x: float | Array) -> Array:
        return np.asarray(np.interp(x, self.a, self.b), dtype=np.float64)


def tradeoff_curve(p: Array, q: Array) -> Curve:
    """T(P, Q) for finite P, Q: fire on outcomes in decreasing order of q/p (Neyman-Pearson).

    Outcomes with p = 0 cost no type I error, so they are fired on first and set T(0) = 1 - Q[p = 0].
    """
    free = p <= 0
    start = 1.0 - float(q[free].sum())
    pp, qq = p[~free], q[~free]
    order = np.argsort(-(qq / pp), kind="stable")
    a = np.concatenate([[0.0], np.cumsum(pp[order])])
    b = np.concatenate([[start], start - np.cumsum(qq[order])])
    a[-1] = 1.0
    return Curve(a, np.maximum(b, 0.0))


def np_test(p: Array, q: Array, level: float) -> Array:
    """The likelihood-ratio test of size exactly `level` on a finite pair, as a vector phi in [0, 1].

    It fires on outcomes in decreasing order of q/p and randomizes on the one where the budget runs out.
    """
    phi = np.zeros_like(p)
    free = p <= 0
    phi[free] = 1.0
    budget = level
    ratio = np.where(free, np.inf, q / np.where(free, 1.0, p))
    for i in np.argsort(-ratio, kind="stable").tolist():
        if free[i]:
            continue
        if budget <= 0:
            break
        take = min(1.0, budget / float(p[i]))
        phi[i] = take
        budget -= take * float(p[i])
    return phi


def dot(x: Array, y: Array) -> float:
    return float(np.sum(x * y))


def errors(p: Array, q: Array, phi: Array) -> tuple[float, float]:
    """(type I, type II) of the test phi on the pair (P, Q)."""
    return float(np.sum(p * phi)), 1.0 - float(np.sum(q * phi))


# --- 1. The canonical pair ---------------------------------------------------------------------


def eps_delta_tradeoff(eps: float, delta: float, a: float) -> float:
    return max(0.0, 1 - delta - math.exp(eps) * a, math.exp(-eps) * (1 - delta - a))


def canonical_pair(f: Callable[[float], float], cells: int) -> tuple[Array, Array]:
    """The canonical pair of f, coarsened to `cells` equal cells of [0, 1] plus the atom outside.

    P is uniform on [0, 1]; Q has distribution function F(x) = f(1 - x) on [0, 1] and an atom of
    mass 1 - f(0) at a point outside [0, 1] (the last entry, where P has no mass).
    """
    edges = np.linspace(0.0, 1.0, cells + 1)
    cdf = np.array([f(1 - x) for x in edges.tolist()])
    p = np.concatenate([np.full(cells, 1.0 / cells), [0.0]])
    q = np.concatenate([np.diff(cdf), [1 - f(0.0)]])
    return p, q


def canonical_gap(f: Callable[[float], float], cells: int, grid: int = 4001) -> float:
    """Largest |T(P_f, Q_f) - f| of the coarsened canonical pair, on a grid of type I errors."""
    curve = tradeoff_curve(*canonical_pair(f, cells))
    a = np.linspace(0.0, 1.0, grid)
    return float(np.max(np.abs(curve.at(a) - np.array([f(x) for x in a.tolist()]))))


def canonical_density_gaussian(mu: float, x: Array) -> Array:
    """Density of Q_f on (0, 1) for f = G_mu: |G_mu'(1 - x)| = exp(mu Phi^{-1}(x) - mu^2 / 2)."""
    return np.array([math.exp(mu * normal_ppf(v) - mu * mu / 2) for v in x.tolist()])


# --- 2. Composition of f-DP by test replacement --------------------------------------------------


def rr_pair(keep: float) -> tuple[Array, Array]:
    """Randomized response on a bit that is 1 in D and 0 in D'; outputs ordered (1, 0)."""
    return np.array([keep, 1 - keep]), np.array([1 - keep, keep])


@dataclass(frozen=True)
class TwoStep:
    """A two-step adaptive mechanism on one neighbouring pair: P1, Q1 and the rows P2[y1], Q2[y1]."""

    p1: Array
    q1: Array
    p2: Array
    q2: Array

    def joint(self) -> tuple[Array, Array]:
        return (self.p1[:, None] * self.p2).ravel(), (self.q1[:, None] * self.q2).ravel()


def adaptive_rr(keep1: float, keep_after_one: float, keep_after_zero: float) -> TwoStep:
    """The adaptive pair of the Renyi-DP chapter: the second keep-probability depends on y1."""
    p1, q1 = rr_pair(keep1)
    pa, qa = rr_pair(keep_after_one)
    pb, qb = rr_pair(keep_after_zero)
    return TwoStep(p1, q1, np.stack([pa, pb]), np.stack([qa, qb]))


def replace_second(mech: TwoStep, phi: Array, pt: Array, qt: Array) -> Array:
    """Step 2: on each branch y1, swap phi(y1, .) for the NP test on (pt, qt) of the same type I error."""
    rows = [np_test(pt, qt, dot(mech.p2[y1], phi[y1])) for y1 in range(len(mech.p1))]
    return np.stack(rows)


def replace_first(p1: Array, phi: Array, pt: Array, qt: Array) -> Array:
    """Step 4: for each second coordinate z2, swap phi(., z2) for the NP test on (pt, qt) of the same size."""
    cols = [np_test(pt, qt, float(np.sum(p1 * phi[:, z]))) for z in range(phi.shape[1])]
    return np.stack(cols, axis=1)


@dataclass(frozen=True)
class Replacement:
    original: tuple[float, float]
    after_second: tuple[float, float]
    after_both: tuple[float, float]


def replace_test(mech: TwoStep, phi: Array, pair1: tuple[Array, Array], pair2: tuple[Array, Array]) -> Replacement:
    """Errors of a joint test, after step 2 (on P1 x P2~) and after step 4 (on P1~ x P2~)."""
    p, q = mech.joint()
    first = errors(p, q, phi.ravel())
    psi = replace_second(mech, phi, *pair2)
    mid_p = (mech.p1[:, None] * pair2[0][None, :]).ravel()
    mid_q = (mech.q1[:, None] * pair2[1][None, :]).ravel()
    second = errors(mid_p, mid_q, psi.ravel())
    psi2 = replace_first(mech.p1, psi, *pair1)
    end_p = (pair1[0][:, None] * pair2[0][None, :]).ravel()
    end_q = (pair1[1][:, None] * pair2[1][None, :]).ravel()
    return Replacement(first, second, errors(end_p, end_q, psi2.ravel()))


def product_pair(*pairs: tuple[Array, Array]) -> tuple[Array, Array]:
    p, q = np.array([1.0]), np.array([1.0])
    for pi, qi in pairs:
        p, q = np.outer(p, pi).ravel(), np.outer(q, qi).ravel()
    return p, q


def random_test_check(rng: np.random.Generator, mech: TwoStep, keep: float, n: int) -> tuple[int, float, float]:
    """Replace n random tests: (count with changed type I or larger type II, largest |change in a|, mean drop in b)."""
    pair = rr_pair(keep)
    bad, worst_a, drop = 0, 0.0, 0.0
    for _ in range(n):
        phi = rng.uniform(size=(2, 2))
        r = replace_test(mech, phi, pair, pair)
        da = abs(r.after_both[0] - r.original[0])
        worst_a = max(worst_a, da)
        if da > 1e-12 or r.after_both[1] > r.original[1] + 1e-12:
            bad += 1
        drop += r.original[1] - r.after_both[1]
    return bad, worst_a, drop / n


# --- 3. Dominating pairs ----------------------------------------------------------------------


def hockey(p: Array, q: Array, gamma: float) -> float:
    """E_gamma(P || Q) = sum (P - gamma Q)_+."""
    return float(np.sum(np.maximum(p - gamma * q, 0.0)))


def three_step_adaptive(strong: float = 0.75, weak: float = 0.6) -> tuple[Array, Array]:
    """The eight transcripts of the adaptive-composition chapter: strong first, then (strong, weak) or (weak, strong)."""
    p_list: list[float] = []
    q_list: list[float] = []
    for y1 in (1, 0):
        k2, k3 = (strong, weak) if y1 == 1 else (weak, strong)
        for y2 in (1, 0):
            for y3 in (1, 0):
                pp, qq = 1.0, 1.0
                for keep, y in ((strong, y1), (k2, y2), (k3, y3)):
                    pp *= keep if y == 1 else 1 - keep
                    qq *= 1 - keep if y == 1 else keep
                p_list.append(pp)
                q_list.append(qq)
    return np.array(p_list), np.array(q_list)


def region_boundary(p: Array, q: Array) -> Curve:
    """g(c) = max E_P[psi] over tests with E_Q[psi] <= c, as vertices (c, g): fire in decreasing order of p/q."""
    curve = tradeoff_curve(q, p)  # tests of Q against P that fire for P: type I c = E_Q, power E_P = 1 - T
    return Curve(curve.a, 1.0 - curve.b)


def region_check(rng: np.random.Generator, p: Array, q: Array, pt: Array, qt: Array, n: int) -> tuple[int, float]:
    """Random tests on (P, Q): how many lie above the boundary of (P~, Q~), and the smallest margin."""
    g = region_boundary(pt, qt)
    phi = rng.uniform(size=(n, p.size))
    ep, eq = phi @ p, phi @ q
    margin = g.at(eq) - ep
    return int(np.sum(margin < -1e-12)), float(margin.min())


def gaussian_region(mu: float, c: Array) -> Array:
    """Upper boundary of the testing region of N(mu, 1) against N(0, 1): g(c) = 1 - G_mu(c)."""
    return np.array([1 - gaussian_tradeoff(mu, x) for x in c.tolist()])


def gaussian_touch(mu: float, gamma: float) -> tuple[float, float]:
    """Where the line of slope gamma touches g: (E_Q psi, E_P psi) of the test that fires when L > ln gamma."""
    t = math.log(gamma) / mu + mu / 2  # fire when y > t for P = N(mu, 1), Q = N(0, 1)
    return normal_sf(t), normal_sf(t - mu)


# --- 4. The f-DP central limit theorem ---------------------------------------------------------


def rr_functionals(eps: float) -> tuple[float, float, float]:
    """(kl, kappa2, kappa3bar) of f_{eps,0}: the loss is +eps w.p. e^eps/(1+e^eps), else -eps."""
    t = math.exp(eps) / (1 + math.exp(eps))
    kl = eps * math.tanh(eps / 2)
    kappa3bar = t * (eps - kl) ** 3 + (1 - t) * (eps + kl) ** 3
    return kl, eps * eps, kappa3bar


def clt_mu_gamma(eps: float, k: int) -> tuple[float, float]:
    """mu and the Berry-Esseen band gamma of the chapter's eq-gdp-clt-mu, for k copies of f_{eps,0}."""
    kl, kappa2, kappa3bar = rr_functionals(eps)
    var = k * (kappa2 - kl * kl)
    return 2 * k * kl / math.sqrt(var), BERRY_ESSEEN * k * kappa3bar / var**1.5


def log_binom_pmf(k: int, t: float) -> Array:
    j = np.arange(k + 1)
    logc = np.array([math.lgamma(k + 1) - math.lgamma(i + 1) - math.lgamma(k - i + 1) for i in range(k + 1)])
    return logc + j * math.log(t) + (k - j) * math.log1p(-t)


def rr_composed(eps: float, k: int) -> tuple[Array, Array, Array]:
    """Values of the summed loss eps(2B - k) and their probabilities under P and under Q."""
    t = math.exp(eps) / (1 + math.exp(eps))
    loss = eps * (2 * np.arange(k + 1) - k).astype(np.float64)
    return loss, np.exp(log_binom_pmf(k, t)), np.exp(log_binom_pmf(k, 1 - t))


def rr_composed_curve(eps: float, k: int) -> Curve:
    _, p, q = rr_composed(eps, k)
    return tradeoff_curve(p, q)


def band(mu: float, gamma: float, a: Array) -> tuple[Array, Array]:
    lower = np.array([gaussian_tradeoff(mu, x + gamma) - gamma for x in a.tolist()])
    upper = np.array([gaussian_tradeoff(mu, x - gamma) + gamma for x in a.tolist()])
    return lower, upper


def band_slack(eps: float, k: int, points: int = 2001) -> tuple[float, float]:
    """Smallest distance of the exact composed curve to the lower and to the upper band curve on [gamma, 1 - gamma]."""
    mu, gamma = clt_mu_gamma(eps, k)
    a = np.linspace(gamma, 1 - gamma, points)
    f = rr_composed_curve(eps, k).at(a)
    lower, upper = band(mu, gamma, a)
    return float(np.min(f - lower)), float(np.min(upper - f))


def curve_distance(eps: float, k: int, points: int = 2001) -> float:
    """max_a |f_{eps,0}^{(x)k}(a) - G_mu(a)| with the mu of eq-gdp-clt-mu."""
    mu, _ = clt_mu_gamma(eps, k)
    a = np.linspace(0, 1, points)
    g = np.array([gaussian_tradeoff(mu, x) for x in a.tolist()])
    return float(np.max(np.abs(rr_composed_curve(eps, k).at(a) - g)))


def standardized_cdf_gap(values: Array, prob: Array, shift: float = 0.0) -> float:
    """sup_x |Pr[X <= x] - Phi(x + shift)| for a discrete X, checked on both sides of every jump."""
    order = np.argsort(values)
    v, cdf = values[order], np.cumsum(prob[order])
    before = np.concatenate([[0.0], cdf[:-1]])
    phi = normal_cdf_array(v + shift)
    return float(max(np.max(np.abs(cdf - phi)), np.max(np.abs(before - phi))))


def rr_kolmogorov(eps: float, k: int) -> tuple[float, float]:
    """Actual sup-distances of the standardized summed loss from Phi(x) under P and from Phi(x + mu) under Q."""
    loss, p, q = rr_composed(eps, k)
    kl, kappa2, _ = rr_functionals(eps)
    mean, sd = k * kl, math.sqrt(k * (kappa2 - kl * kl))
    mu = 2 * mean / sd
    x = (loss - mean) / sd
    return standardized_cdf_gap(x, p), standardized_cdf_gap(x, q, mu)


def eps_tanh_check(eps_values: Sequence[float]) -> float:
    """Largest |kl(f_{eps,0}) - eps^2/2| / eps^4 over eps: bounded, so sum kl -> mu^2 / 2."""
    return max(abs(rr_functionals(e)[0] - e * e / 2) / e**4 for e in eps_values)


# --- 5. The subsampled Gaussian ---------------------------------------------------------------


def subsampled_loss_cdfs(q: float, sigma: float, ell: Array) -> tuple[Array, Array]:
    """Pr[L <= ell] under P = N(0, s^2) and under Q = (1-q)N(0, s^2) + qN(1, s^2), with L = ln(P(y) / Q(y)).

    L = -ln(1 + qW) with W = exp((y - 1/2) / s^2) - 1 is decreasing in y, so L <= ell exactly when
    y >= y(ell) = 1/2 + s^2 ln(1 + (e^{-ell} - 1) / q); for ell >= -ln(1 - q) that set is everything.
    """
    inside = 1 + np.expm1(-ell) / q
    ok = inside > 0
    y = np.where(ok, 0.5 + sigma**2 * np.log(np.where(ok, inside, 1.0)), -np.inf)
    fin = np.isfinite(y)
    tail_p = np.ones_like(ell)
    tail_q1 = np.ones_like(ell)
    tail_p[fin] = normal_sf_array(y[fin] / sigma)
    tail_q1[fin] = normal_sf_array((y[fin] - 1) / sigma)
    return tail_p, (1 - q) * tail_p + q * tail_q1


def subsampled_moments(q: float, sigma: float, points: int = 400_001) -> dict[str, float]:
    """Exact mean, variance and third absolute central moment of one step's loss under P and Q, by quadrature."""
    y = np.linspace(-14 * sigma, 1 + 14 * sigma, points)
    dens_p = np.exp(-(y**2) / (2 * sigma**2)) / (sigma * math.sqrt(2 * math.pi))
    w = np.expm1((y - 0.5) / sigma**2)
    loss = -np.log1p(q * w)
    dens_q = dens_p * (1 + q * w)
    out: dict[str, float] = {}
    for name, dens in (("P", dens_p), ("Q", dens_q)):
        mean = float(np.trapezoid(dens * loss, y))
        out[f"mean_{name}"] = mean
        out[f"var_{name}"] = float(np.trapezoid(dens * (loss - mean) ** 2, y))
        out[f"abs3_{name}"] = float(np.trapezoid(dens * np.abs(loss - mean) ** 3, y))
    return out


def w_moment_bound(n: int, sigma: float) -> float:
    """E_P|W|^n <= E_P[(1 + W)^n] + 1 = e^{n(n-1)/(2 s^2)} + 1."""
    return math.exp(n * (n - 1) / (2 * sigma**2)) + 1


def mean_remainders(q: float, sigma: float) -> tuple[float, float]:
    """(E_P L - q^2 v/2) / q^3 and (E_Q L + q^2 v/2) / q^3, with v = e^{1/s^2} - 1."""
    m = subsampled_moments(q, sigma)
    v = math.expm1(1 / sigma**2)
    return (m["mean_P"] - q * q * v / 2) / q**3, (m["mean_Q"] + q * q * v / 2) / q**3


def var_remainders(q: float, sigma: float) -> tuple[float, float]:
    m = subsampled_moments(q, sigma)
    v = math.expm1(1 / sigma**2)
    return (m["var_P"] - q * q * v) / q**3, (m["var_Q"] - q * q * v) / q**3


def berry_esseen_subsampled(q: float, sigma: float, k: int) -> tuple[float, float]:
    """The Berry-Esseen bound 0.56 * k E|L - EL|^3 / (k Var L)^{3/2} under P and under Q."""
    m = subsampled_moments(q, sigma)
    return tuple(  # type: ignore[return-value]
        BERRY_ESSEEN * m[f"abs3_{s}"] / (m[f"var_{s}"] ** 1.5 * math.sqrt(k)) for s in ("P", "Q")
    )


def clt_mu_subsampled(q: float, sigma: float, k: int) -> float:
    return q * math.sqrt(k * math.expm1(1 / sigma**2))


@dataclass(frozen=True)
class Lattice:
    """Per-step loss L = ln(P/Q) of the subsampled Gaussian on the grid h*Z, under P and Q, rounded up or down.

    Masses below the lowest grid point `lowest` are reported separately (`tail_p`, `tail_q`).
    """

    h: float
    lowest: int
    p_up: Array
    p_down: Array
    q_up: Array
    q_down: Array
    tail_p: float
    tail_q: float


def subsampled_lattice(q: float, sigma: float, h: float, ell_min: float = -4.0) -> Lattice:
    top = math.ceil(-math.log1p(-q) / h)  # L <= -ln(1 - q) always
    lowest = math.floor(ell_min / h)
    edges = (np.arange(lowest, top + 1) * h).astype(np.float64)
    cdf_p, cdf_q = subsampled_loss_cdfs(q, sigma, edges)
    cell_p, cell_q = np.diff(cdf_p), np.diff(cdf_q)  # mass of (edge_i, edge_{i+1}]
    # rounded up: the cell's mass goes to its right edge; down: to its left edge
    p_up = np.concatenate([[cdf_p[0]], cell_p])  # mass below the lowest edge moved up to it
    q_up = np.concatenate([[cdf_q[0]], cell_q])
    p_down = np.concatenate([cell_p, [0.0]])
    q_down = np.concatenate([cell_q, [0.0]])
    return Lattice(h, lowest, p_up, p_down, q_up, q_down, float(cdf_p[0]), float(cdf_q[0]))


def circular(pmf: Array, lowest: int, size: int) -> Array:
    out = np.zeros(size)
    np.add.at(out, (np.arange(pmf.size) + lowest) % size, pmf)
    return out


def compose_circular(pmf: Array, k: int) -> Array:
    return np.maximum(np.fft.irfft(np.fft.rfft(pmf) ** k, pmf.size), 0.0)


def grid_values(size: int, h: float) -> Array:
    idx = np.arange(size)
    return np.where(idx >= size // 2, idx - size, idx).astype(np.float64) * h


@dataclass(frozen=True)
class Profile:
    """delta(eps) of a lattice loss law, evaluated in O(log n) from suffix sums.

    For eps between two grid losses the set {L > eps} is fixed, so delta(eps) = atom + A - e^eps B,
    with A the mass above eps and B the sum of mass * e^{-L} above eps.
    """

    loss: Array
    above: Array
    weighted: Array
    atom: float

    @classmethod
    def of(cls, loss: Array, mass: Array, atom: float = 0.0) -> "Profile":
        keep = mass > 0
        order = np.argsort(loss[keep])
        lv, mv = loss[keep][order], mass[keep][order]
        above = np.concatenate([np.cumsum(mv[::-1])[::-1], [0.0]])
        weighted = np.concatenate([np.cumsum((mv * np.exp(-lv))[::-1])[::-1], [0.0]])
        return cls(lv, above, weighted, atom)

    def __call__(self, eps: float) -> float:
        j = int(np.searchsorted(self.loss, eps, side="right"))
        return self.atom + max(float(self.above[j] - math.exp(eps) * self.weighted[j]), 0.0)


@dataclass(frozen=True)
class Composed:
    """The k-fold composed loss of the subsampled Gaussian in both directions, rounded both ways."""

    k: int
    loss: Array
    add_up: Array  # law of the summed L under P, each step rounded up (add direction: P = record absent)
    add_down: Array
    rem_up: Array  # law of the summed -L under Q, rounded up (remove direction: the mixture is the null)
    rem_down: Array
    rem_atom: float  # the remove direction's mass at +infinity (losses beyond the grid), kept pessimistic
    q_up: Array  # law of the summed L under Q, each step rounded up
    q_down: Array


def compose_subsampled(lat: Lattice, k: int, size: int) -> Composed:
    loss = grid_values(size, lat.h)
    flip = (-np.arange(size)) % size  # index of -x

    def law(pmf: Array, lowest: int) -> Array:
        return compose_circular(circular(pmf, lowest, size), k)

    q_low = law(lat.q_down, lat.lowest)
    q_high = law(lat.q_up, lat.lowest)
    # Under the remove direction the loss is -L under Q; rounding -L up is rounding L down, and the
    # mass below the grid becomes mass at +infinity, which composes as 1 - (1 - tail)^k.
    return Composed(
        k,
        loss,
        law(lat.p_up, lat.lowest),
        law(lat.p_down, lat.lowest),
        q_low[flip],
        q_high[flip],
        -math.expm1(k * math.log1p(-lat.tail_q)),
        q_high,
        q_low,
    )


def subsampled_profiles(c: Composed) -> dict[str, Profile]:
    return {
        "add_low": Profile.of(c.loss, c.add_down),
        "add_high": Profile.of(c.loss, c.add_up),
        "rem_low": Profile.of(c.loss, c.rem_down),
        "rem_high": Profile.of(c.loss, c.rem_up, c.rem_atom),
    }


def subsampled_eps(c: Composed, delta: float) -> dict[str, float]:
    return {name: solve_decreasing(prof, delta, 0.0, 30.0) for name, prof in subsampled_profiles(c).items()}


def standardized_cdf(loss: Array, mass: Array, mean: float, sd: float, x: Array) -> Array:
    """Pr[(S - mean) / sd <= x] for the lattice law of S."""
    order = np.argsort(loss)
    cdf = np.cumsum(mass[order])
    idx = np.searchsorted(loss[order], mean + sd * x, side="right") - 1
    return np.where(idx >= 0, cdf[np.maximum(idx, 0)], 0.0)


def cdf_distance_bracket(up: Array, down: Array, target: Array) -> tuple[float, float]:
    """Bounds on sup |F - target| when F_up <= F <= F_down pointwise (the exact law lies between its roundings)."""
    low = float(np.max(np.maximum(np.maximum(up - target, target - down), 0.0)))
    high = float(max(np.max(np.abs(up - target)), np.max(np.abs(down - target))))
    return low, high


def subsampled_kolmogorov(c: Composed, q: float, sigma: float) -> tuple[tuple[float, float], tuple[float, float]]:
    """Brackets on sup_x of the distance of the standardized summed loss (exact k-step mean and sd under P)
    from Phi(x) under P, and from Phi(x + mu) under Q with mu = q sqrt(k(e^{1/s^2} - 1)), on a grid of x."""
    m = subsampled_moments(q, sigma)
    mean, sd = c.k * m["mean_P"], math.sqrt(c.k * m["var_P"])
    mu = clt_mu_subsampled(q, sigma, c.k)
    x = np.linspace(-6, 6, 4001)

    def cdf(law: Array) -> Array:
        return standardized_cdf(c.loss, law, mean, sd, x)

    on_p = cdf_distance_bracket(cdf(c.add_up), cdf(c.add_down), normal_cdf_array(x))
    on_q = cdf_distance_bracket(cdf(c.q_up), cdf(c.q_down), normal_cdf_array(x + mu))
    return on_p, on_q


# --- Table ------------------------------------------------------------------------------------

Q_RATE, SIGMA, H, SIZE = 0.01, 1.0, 1e-5, 1 << 22
STEPS = (100, 1000, 10_000)


@cache
def running_example() -> tuple[Composed, ...]:
    """The running DP-SGD example composed k times for each k in STEPS (shared by the table and the figures)."""
    lat = subsampled_lattice(Q_RATE, SIGMA, H)
    return tuple(compose_subsampled(lat, k, SIZE) for k in STEPS)


def print_table(rng: np.random.Generator) -> None:
    # 1. canonical pair
    g1 = canonical_gap(lambda a: gaussian_tradeoff(1.0, a), 1000)
    fed = canonical_gap(lambda a: eps_delta_tradeoff(1.0, 0.1, a), 1000)
    print(f"[canonical] G_1(0.1)={gaussian_tradeoff(1.0, 0.1):.4f}  power={1 - gaussian_tradeoff(1.0, 0.1):.4f}")
    print(f"[canonical] 1000 cells: max gap G_1={g1:.1e}  f_(1,0.1)={fed:.1e}")
    print(f"[canonical] f_(1,0.1) corner a*={(1 - 0.1) / (1 + math.e):.4f}")

    # 2. adaptive composition of two randomized responses
    mech = adaptive_rr(0.75, 0.75, 0.6)
    p, q = mech.joint()
    adaptive = tradeoff_curve(p, q)
    pair = rr_pair(0.75)
    product = tradeoff_curve(*product_pair(pair, pair))
    grid = np.linspace(0, 1, 100_001)
    gap = adaptive.at(grid) - product.at(grid)
    i = int(np.argmax(gap))
    print(f"[adaptive RR] vertices a={np.round(adaptive.a, 4).tolist()} b={np.round(adaptive.b, 4).tolist()}")
    print(f"[adaptive RR] product vertices a={np.round(product.a, 4).tolist()} b={np.round(product.b, 4).tolist()}")
    print(f"[adaptive RR] min gap={gap.min():.1e}  max gap={gap[i]:.4f} at a={grid[i]:.4f}")
    for a in (0.1, 0.25):
        print(f"[adaptive RR] a={a}: adaptive={float(adaptive.at(a)):.4f}  product={float(product.at(a)):.4f}")
    phi = np.array([[0.0, 0.0], [0.0, 1.0]])  # fire on (0, 0); rows are y1 = 1, 0 and columns y2 = 1, 0
    level = dot(mech.p2[1], phi[1])
    power = dot(mech.q2[1], phi[1])
    new_power = float(np.sum(pair[1] * np_test(*pair, level)))
    r = replace_test(mech, phi, pair, pair)
    print(
        f"[replace (0,0)] branch y1=0: level={level:.4f} power {power:.4f} -> {new_power:.4f}  "
        f"joint ({r.original[0]:.4f}, {r.original[1]:.4f}) -> ({r.after_both[0]:.4f}, {r.after_both[1]:.4f})"
    )
    phi = np.array([[1.0, 0.0], [0.0, 0.0]])  # fire on (1, 1)
    r = replace_test(mech, phi, pair, pair)
    print(
        f"[replace (1,1)] original=({r.original[0]:.4f}, {r.original[1]:.4f})  "
        f"after step 2=({r.after_second[0]:.4f}, {r.after_second[1]:.4f})  after step 4=({r.after_both[0]:.4f}, {r.after_both[1]:.4f})"
    )
    bad, worst_a, drop = random_test_check(rng, mech, 0.75, 20_000)
    print(f"[random tests] n=20000  violations={bad}  max |change in a|={worst_a:.1e}  mean drop in b={drop:.4f}")

    # 3. dominating pairs
    pa, qa = three_step_adaptive()
    pp, qp = product_pair(pair, pair, pair)
    gammas = np.concatenate([np.linspace(0.0, 1.0, 101), np.exp(np.linspace(0.0, 4.0, 401))])
    margins = [hockey(pp, qp, g) - hockey(pa, qa, g) for g in gammas.tolist()]
    print(f"[three-step] min E_gamma margin over {gammas.size} gammas in [0, e^4]={min(margins):.1e}")
    for eps in (0.0, 1.0, 2.0, 3.0):
        g = math.exp(eps)
        print(
            f"[three-step] eps={eps:g}: adaptive delta={hockey(pa, qa, g):.4f}  product delta={hockey(pp, qp, g):.4f}"
        )
    for g in (0.5,):
        print(f"[three-step] gamma={g}: adaptive={hockey(pa, qa, g):.4f}  product={hockey(pp, qp, g):.4f}")
    rr_p, rr_q = rr_pair(0.75)
    for g in (0.2, 1.0, 2.0, 4.0):
        print(f"[rr region] gamma={g}: E_gamma={hockey(rr_p, rr_q, g):.4f}")
    count, margin = region_check(rng, pa, qa, pp, qp, 100_000)
    print(f"[region] 100000 random tests above the product's boundary: {count}  smallest margin={margin:.4f}")
    for g in (0.5, 1.0, math.e):
        print(
            f"[gaussian region] gamma={g:.4f}: E_gamma mu=1 {gdp_hockey(1.0, g):.4f}  "
            f"mu=0.6 {gdp_hockey(0.6, g):.4f}  touch={tuple(round(v, 4) for v in gaussian_touch(1.0, g))}"
        )

    # 4. the CLT with its band
    for k in (1, 10, 100, 1000):
        eps = 1 / math.sqrt(k)
        mu, gamma = clt_mu_gamma(eps, k)
        line = f"[clt RR] k={k} eps0={eps:.4f} mu={mu:.4f} gamma={gamma:.4f}"
        if gamma < 0.5:
            lo, hi = band_slack(eps, k)
            ks_p, ks_q = rr_kolmogorov(eps, k)
            line += f"  slack lower={lo:.4f} upper={hi:.4f}  sup|F-Phi| P={ks_p:.4f} Q={ks_q:.4f}"
            line += f"  max|f-G_mu|={curve_distance(eps, k):.4f}"
        print(line)
    kl, kappa2, kappa3bar = rr_functionals(0.1)
    big_k, big_v = 100 * kl, 100 * (kappa2 - kl * kl)
    print(f"[clt RR] k=100: K={big_k:.4f}  V={big_v:.4f}  sum kappa3bar={100 * kappa3bar:.6f}")
    print(f"[clt RR] eps=0.1: kl={kl:.6f}  eps^2/2={0.005:.6f}  kappa2={kappa2:.4f}")
    print(
        f"[clt RR] max |kl - eps^2/2| / eps^4 over eps in (0, 1]: {eps_tanh_check([i / 100 for i in range(1, 101)]):.4f}"
    )

    # 5. the subsampled Gaussian
    v = math.expm1(1 / SIGMA**2)
    m = subsampled_moments(Q_RATE, SIGMA)
    print(
        f"[sgd] v=e^(1/s^2)-1={v:.4f}  leading mean q^2 v/2={Q_RATE**2 * v / 2:.4e}  leading var q^2 v={Q_RATE**2 * v:.4e}"
    )
    print(
        f"[sgd] exact: mean_P={m['mean_P']:.4e} var_P={m['var_P']:.4e} mean_Q={m['mean_Q']:.4e} var_Q={m['var_Q']:.4e}"
    )
    for qq in (0.02, 0.01, 0.005, 0.0025):
        mp, mq = mean_remainders(qq, SIGMA)
        vp, vq = var_remainders(qq, SIGMA)
        print(f"[sgd remainders] q={qq}: mean P {mp:.3f} Q {mq:.3f}  var P {vp:.3f} Q {vq:.3f}  (all / q^3)")
    print(
        f"[sgd bounds] E|W|^3 <= {w_moment_bound(3, SIGMA):.2f}  E W^4 <= {w_moment_bound(4, SIGMA):.2f}  "
        f"mean bound (2/3)E|W|^3={2 / 3 * w_moment_bound(3, SIGMA):.2f}"
    )
    lat = subsampled_lattice(Q_RATE, SIGMA, H)
    print(f"[sgd lattice] h={H:g} cells={lat.p_up.size}  mass below grid P={lat.tail_p:.1e} Q={lat.tail_q:.1e}")
    for c in running_example():
        k = c.k
        mu = clt_mu_subsampled(Q_RATE, SIGMA, k)
        be_p, be_q = berry_esseen_subsampled(Q_RATE, SIGMA, k)
        ks_p, ks_q = subsampled_kolmogorov(c, Q_RATE, SIGMA)
        e = subsampled_eps(c, 1e-5)
        print(
            f"[sgd k={k}] mu={mu:.4f} CLT eps={gdp_eps(mu, 1e-5):.4f}  Berry-Esseen P={be_p:.4f} Q={be_q:.4f}  "
            f"sup|F-Phi| P in [{ks_p[0]:.4f}, {ks_p[1]:.4f}] Q in [{ks_q[0]:.4f}, {ks_q[1]:.4f}]"
        )
        print(
            f"[sgd k={k}] eps at 1e-5: add [{e['add_low']:.4f}, {e['add_high']:.4f}]  "
            f"remove [{e['rem_low']:.4f}, {e['rem_high']:.4f}]"
        )


# --- Figures ----------------------------------------------------------------------------------


def plot_canonical(a: float = 0.1) -> Figure:
    fig, (left, right) = plt.subplots(1, 2, figsize=(10, 4))
    x = np.linspace(0.0005, 0.9995, 1000)
    dens = canonical_density_gaussian(1.0, x)
    left.plot(x, np.ones_like(x), color="C0", label=r"$P_f$: uniform")
    left.plot(x, dens, color="C1", label=r"$Q_f$: density, minus the slope of $f$ at $1 - x$")
    cut = x >= 1 - a
    left.fill_between(x[cut], 0, 1, color="C0", alpha=0.35, label=rf"fire: type I error $\alpha = {a}$")
    left.fill_between(
        x[cut],
        1,
        np.minimum(dens[cut], 4),
        color="C1",
        alpha=0.3,
        label=rf"power $1 - f(\alpha) = {1 - gaussian_tradeoff(1.0, a):.4f}$",
    )
    left.axvline(1 - a, color="gray", linestyle=":")
    left.set(xlim=(0, 1.02), ylim=(0, 4), xlabel="x", ylabel="density", title=r"$f = G_1$: no atom, since $f(0) = 1$")
    left.legend(loc="upper left", fontsize=8)

    eps, delta = 1.0, 0.1
    corner = (1 - delta) / (1 + math.e)
    xs = np.linspace(0, 1, 1001)
    dq = np.where(xs < 1 - corner, math.exp(-eps), math.exp(eps))
    right.plot(xs, np.ones_like(xs), color="C0", label=r"$P_f$: uniform")
    right.plot(xs, dq, color="C1", label=r"$Q_f$ on $[0, 1]$")
    right.bar([1.15], [delta / 0.05], width=0.05, color="C1", alpha=0.6, label=rf"atom of mass $1 - f(0) = {delta}$")
    sel = xs >= 1 - a
    right.fill_between(xs[sel], 0, 1, color="C0", alpha=0.35)
    right.fill_between(xs[sel], 1, dq[sel], color="C1", alpha=0.3)
    right.axvline(1 - a, color="gray", linestyle=":")
    right.set(
        xlim=(0, 1.25),
        ylim=(0, 3.2),
        xlabel="x (the atom sits outside [0, 1])",
        ylabel="density",
        title=r"$f = f_{1, 0.1}$: atom of mass $\delta$",
    )
    right.legend(loc="upper left", fontsize=8)
    return fig


def plot_replacement() -> Figure:
    mech = adaptive_rr(0.75, 0.75, 0.6)
    pair = rr_pair(0.75)
    a = np.linspace(0, 1, 501)
    fig, (left, right) = plt.subplots(1, 2, figsize=(10, 4.4))
    f2 = tradeoff_curve(*pair)
    weak = tradeoff_curve(*rr_pair(0.6))
    left.plot(a, f2.at(a), "k", label=r"$f_2$: branch $y_1 = 1$ (keep 0.75)")
    left.plot(a, weak.at(a), "C2--", label=r"branch $y_1 = 0$ (keep 0.6)")
    phi = np.array([[0.3, 0.6], [1.0, 0.0]])
    psi = replace_second(mech, phi, *pair)
    for y1, color in ((0, "C0"), (1, "C3")):
        p2, q2 = mech.p2[y1], mech.q2[y1]
        before = errors(p2, q2, phi[y1])
        after = errors(*pair, psi[y1])
        left.plot(*before, "o", color=color, label=rf"$\phi(y_1 = {1 - y1}, \cdot)$ and its replacement")
        left.annotate("", xy=after, xytext=before, arrowprops={"arrowstyle": "->", "color": color})
        left.plot(*after, "s", color=color, markerfacecolor="white")
    left.set(
        xlabel=r"type I error $\alpha$",
        ylabel=r"type II error $\beta$",
        title="Step 2: one branch at a time",
        xlim=(0, 1),
        ylim=(0, 1),
    )
    left.legend(fontsize=8, loc="upper right")

    p, q = mech.joint()
    right.plot(a, tradeoff_curve(p, q).at(a), "--", color="0.5", label="adaptive pair (exact curve)")
    right.plot(a, tradeoff_curve(*product_pair(pair, pair)).at(a), "k", label=r"$f_1 \otimes f_2$ (product pair)")
    tests = [
        phi,
        np.array([[0.0, 0.0], [0.0, 1.0]]),
        np.array([[1.0, 0.0], [1.0, 0.0]]),
        np.array([[1.0, 1.0], [0.0, 1.0]]),
    ]
    for j, t in enumerate(tests):
        r = replace_test(mech, t, pair, pair)
        right.plot(*r.original, "o", color=f"C{j}")
        for start, end in ((r.original, r.after_second), (r.after_second, r.after_both)):
            if abs(start[1] - end[1]) > 1e-9:
                right.annotate("", xy=end, xytext=start, arrowprops={"arrowstyle": "->", "color": f"C{j}"})
        if abs(r.after_second[1] - r.after_both[1]) > 1e-9:
            right.plot(*r.after_second, "^", color=f"C{j}", markerfacecolor="white")
        right.plot(*r.after_both, "s", color=f"C{j}", markerfacecolor="white")
    right.plot([], [], "ko", label="a joint test")
    right.plot([], [], "k^", markerfacecolor="white", label="after step 2, when step 4 moves it")
    right.plot([], [], "ks", markerfacecolor="white", label="after steps 2 and 4")
    right.set(
        xlabel=r"type I error $\alpha$",
        ylabel=r"type II error $\beta$",
        title="Same type I error, no larger type II error",
        xlim=(0, 1),
        ylim=(0, 1),
    )
    right.legend(fontsize=8, loc="upper center")
    return fig


def plot_regions() -> Figure:
    fig, ax = plt.subplots(figsize=(6, 5))
    c = np.linspace(0, 1, 801)
    for mu, style, label in (
        (1.0, "k", "dominating pair: N(1, 1) against N(0, 1)"),
        (0.6, "C2", "dominated pair: N(0.6, 1) against N(0, 1)"),
    ):
        g = gaussian_region(mu, c)
        ax.plot(c, g, style, label=label)
        ax.plot(c, 1 - g[::-1], style, linewidth=0.8)
    ax.fill_between(
        c,
        1 - gaussian_region(1.0, c)[::-1],
        gaussian_region(1.0, c),
        color="0.85",
        label="achievable $(\\mathbb{E}_Q\\phi, \\mathbb{E}_P\\phi)$, $\\mu = 1$",
    )
    for gamma, color in ((0.5, "C0"), (1.0, "C1"), (math.e, "C3")):
        e1 = gdp_hockey(1.0, gamma)
        ax.plot(
            c,
            e1 + gamma * c,
            color=color,
            linestyle="--",
            linewidth=1,
            label=rf"slope $\gamma = {gamma:.3g}$, intercept $E_\gamma = {e1:.3f}$",
        )
        ax.plot(*gaussian_touch(1.0, gamma), "o", color=color)
        ax.plot([0], [gdp_hockey(0.6, gamma)], "x", color=color, markersize=9, clip_on=False)
    ax.set(
        xlim=(0, 1),
        ylim=(0, 1.05),
        xlabel=r"$\mathbb{E}_Q\phi$ (fires under $Q$)",
        ylabel=r"$\mathbb{E}_P\phi$ (fires under $P$)",
        title="Testing regions and their supporting lines",
    )
    ax.legend(fontsize=7, loc="lower right")
    return fig


def plot_three_step() -> Figure:
    pa, qa = three_step_adaptive()
    pair = rr_pair(0.75)
    pp, qp = product_pair(pair, pair, pair)
    eps = np.linspace(-2.0, 3.4, 1081)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(
        eps, [hockey(pp, qp, math.exp(e)) for e in eps.tolist()], "k", label=r"product of three $(P_i, Q_i)$, keep 0.75"
    )
    ax.plot(eps, [hockey(pa, qa, math.exp(e)) for e in eps.tolist()], "C2--", label="adaptive three-step run")
    ax.axvline(0.0, color="gray", linestyle=":", linewidth=0.8)
    ax.set(
        yscale="log",
        ylim=(1e-3, 1.5),
        xlabel=r"$\ln\gamma$ (= $\varepsilon$ when $\geq 0$)",
        ylabel=r"$E_\gamma(P\|Q)$",
        title="The product dominates at every $\\gamma \\geq 0$",
    )
    ax.legend(fontsize=8)
    return fig


def plot_clt_band() -> Figure:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.4))
    a = np.linspace(0, 1, 1001)
    for ax, k in zip(axes, (10, 100), strict=True):
        eps = 1 / math.sqrt(k)
        mu, gamma = clt_mu_gamma(eps, k)
        lower, upper = band(mu, gamma, a)
        ax.fill_between(a, np.clip(lower, 0, 1), np.clip(upper, 0, 1), color="0.88", label=rf"band, $w = {gamma:.4f}$")
        ax.plot(a, [gaussian_tradeoff(mu, x) for x in a.tolist()], "C0", label=rf"$G_\mu$, $\mu = {mu:.4f}$")
        ax.plot(a, rr_composed_curve(eps, k).at(a), "k", linewidth=1, label=r"exact $f_{\varepsilon_0, 0}^{\otimes k}$")
        ax.axvline(gamma, color="gray", linestyle=":", linewidth=0.8)
        ax.axvline(1 - gamma, color="gray", linestyle=":", linewidth=0.8)
        ax.set(
            xlim=(0, 1),
            ylim=(0, 1),
            xlabel=r"type I error $\alpha$",
            ylabel=r"type II error $\beta$",
            title=rf"$k = {k}$ randomized responses, $\varepsilon_0 = k^{{-1/2}}$",
        )
        ax.legend(fontsize=8)
    return fig


def plot_clt_cdfs() -> Figure:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    x = np.linspace(-4, 4, 801)
    for ax, k in zip(axes, (10, 100), strict=True):
        eps = 1 / math.sqrt(k)
        loss, p, q = rr_composed(eps, k)
        kl, kappa2, _ = rr_functionals(eps)
        mean, sd = k * kl, math.sqrt(k * (kappa2 - kl * kl))
        mu = 2 * mean / sd
        z = (loss - mean) / sd
        ax.step(z, np.cumsum(p), where="post", color="C0", label=r"under $P$")
        ax.step(z, np.cumsum(q), where="post", color="C3", label=r"under $Q$")
        ax.plot(x, normal_cdf_array(x), "C0--", linewidth=1, label=r"$\Phi(x)$")
        ax.plot(x, normal_cdf_array(x + mu), "C3--", linewidth=1, label=rf"$\Phi(x + \mu)$, $\mu = {mu:.4f}$")
        ax.set(
            xlim=(-4, 4),
            ylim=(0, 1),
            xlabel=r"$x = (L - K)\,/\,V^{1/2}$",
            ylabel="distribution function",
            title=f"Standardized summed loss, k = {k}",
        )
        ax.legend(fontsize=8, loc="lower right")
    return fig


def plot_sgd_cdfs(composed: Sequence[Composed]) -> Figure:
    fig, axes = plt.subplots(1, len(composed), figsize=(12, 4))
    m = subsampled_moments(Q_RATE, SIGMA)
    x = np.linspace(-4, 4, 801)
    for ax, c in zip(axes, composed, strict=True):
        mean, sd = c.k * m["mean_P"], math.sqrt(c.k * m["var_P"])
        mu = clt_mu_subsampled(Q_RATE, SIGMA, c.k)
        on_p = (standardized_cdf(c.loss, c.add_up, mean, sd, x) + standardized_cdf(c.loss, c.add_down, mean, sd, x)) / 2
        on_q = (standardized_cdf(c.loss, c.q_up, mean, sd, x) + standardized_cdf(c.loss, c.q_down, mean, sd, x)) / 2
        ax.plot(x, on_p, "C0", label=r"under $P$")
        ax.plot(x, on_q, "C3", label=r"under $Q$")
        ax.plot(x, normal_cdf_array(x), "k:", linewidth=1, label=r"$\Phi(x)$")
        ax.plot(x, normal_cdf_array(x + mu), "k--", linewidth=1, label=rf"$\Phi(x + \mu)$, $\mu = {mu:.4f}$")
        ax.set(xlim=(-4, 4), ylim=(0, 1), xlabel=r"$x = (S - K_P)\,/\,V_P^{1/2}$", title=f"k = {c.k:,}")
        ax.legend(fontsize=7, loc="lower right")
    axes[0].set_ylabel("distribution function")
    return fig


def plot_sgd_tails(composed: Sequence[Composed]) -> Figure:
    fig, axes = plt.subplots(1, len(composed), figsize=(10, 4))
    for ax, c in zip(axes, composed, strict=True):
        mu = clt_mu_subsampled(Q_RATE, SIGMA, c.k)
        prof = subsampled_profiles(c)
        top = 1.08 * solve_decreasing(prof["rem_high"], 1e-8, 0.0, 40.0)
        eps = np.linspace(0.0, top, 400)
        ax.plot(eps, [prof["rem_high"](e) for e in eps.tolist()], "C3", label="remove direction (numerical)")
        ax.plot(eps, [prof["add_high"](e) for e in eps.tolist()], "C0", label="add direction (numerical)")
        ax.plot(eps, [gdp_delta(mu, e) for e in eps.tolist()], "k--", label=rf"CLT: $G_\mu$, $\mu = {mu:.4f}$")
        ax.axhline(1e-5, color="gray", linestyle=":", label=r"$\delta = 10^{-5}$")
        ax.set(
            yscale="log",
            ylim=(1e-8, 1),
            xlim=(0, top),
            xlabel=r"$\varepsilon$",
            ylabel=r"$\delta(\varepsilon)$",
            title=f"k = {c.k:,}, q = {Q_RATE}, σ = {SIGMA:g}",
        )
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
    composed = running_example()
    figures = {
        "apx_comp_canonical.svg": plot_canonical(),
        "apx_comp_replacement.svg": plot_replacement(),
        "apx_comp_regions.svg": plot_regions(),
        "apx_comp_three_step.svg": plot_three_step(),
        "apx_comp_clt_band.svg": plot_clt_band(),
        "apx_comp_clt_cdfs.svg": plot_clt_cdfs(),
        "apx_comp_sgd_cdfs.svg": plot_sgd_cdfs(composed),
        "apx_comp_sgd_tails.svg": plot_sgd_tails(composed[1:]),
    }
    for name, fig in figures.items():
        fig.tight_layout()
        fig.savefig(out / name, metadata={"Date": None})  # no timestamp, so output is reproducible
        plt.close(fig)
        print(f"saved {out / name}")


if __name__ == "__main__":
    main()
