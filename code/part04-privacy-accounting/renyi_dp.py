"""Check the Rényi differential privacy chapter and draw its figures.

    uv run python code/part04-privacy-accounting/renyi_dp.py
    uv run python code/part04-privacy-accounting/renyi_dp.py --seed 1 --out /tmp/figures

Prints the numbers quoted in 01-renyi-dp.md (they match exactly with the default seed), then saves
the chapter's figures.

Conventions: P = M(D) and Q = M(D') are the output distributions on a neighbouring pair, the order α
is > 1, logs are natural. The Gaussian mechanism adds N(0, σ²Δ²) to a query of ℓ2-sensitivity Δ, so σ
is the noise multiplier; on the worst pair P = N(0, σ²), Q = N(Δ, σ²), with μ = Δ/σ. The Laplace
mechanism adds Lap(0, b) to a real-valued query of sensitivity 1. Randomized response reports one
bit truthfully with probability p > 1/2 (the keep-probability).
"""

import argparse
import math
import sys
from collections.abc import Callable, Sequence
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # for code/shared
from shared.figures import configure
from shared.stats import bisect, logsumexp, normal_cdf

configure("renyi_dp")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from numpy.typing import NDArray

# Figures are build output: _build/figures/<part>/, which the build copies into each book.
FIGURES = Path(__file__).resolve().parents[2] / "_build" / "figures" / Path(__file__).parent.name

# The orders on which every chapter of the part optimizes, so that numbers agree across chapters.
ORDERS: list[float] = [1 + x / 10 for x in range(1, 100)] + [float(x) for x in range(12, 64)] + [128.0, 256.0, 512.0]

Dist = Sequence[float]  # a distribution on a finite set, as a list of probabilities
Curve = Callable[[float], float]  # an RDP curve α ↦ ε(α)

DELTA = 1e-5  # the target δ used throughout the part


def log_normal_cdf(x: float) -> float:
    """ln Φ(x), finite far into the lower tail (asymptotic series once erfc underflows)."""
    if x > -30:
        return math.log(normal_cdf(x))
    return -x * x / 2 - math.log(-x) - 0.5 * math.log(2 * math.pi) + math.log1p(-1 / x**2 + 3 / x**4)


# --- Definition: the RDP curve of a pair ----------------------------------------------------


def renyi_discrete(p: Dist, q: Dist, alpha: float) -> float:
    """D_α(P‖Q) = (1/(α-1)) ln Σ p^α q^{1-α}; α = 1 is KL, α = ∞ the max-divergence."""
    pairs = [(pi, qi) for pi, qi in zip(p, q, strict=True) if pi > 0]
    if alpha == 1:
        return sum(pi * math.log(pi / qi) for pi, qi in pairs)
    if math.isinf(alpha):
        return max(math.log(pi / qi) for pi, qi in pairs)
    return logsumexp([alpha * math.log(pi) + (1 - alpha) * math.log(qi) for pi, qi in pairs]) / (alpha - 1)


def rr_pair(p: float) -> tuple[list[float], list[float]]:
    """Output distributions of randomized response on the true bit 1 (P) and 0 (Q), outputs ordered (1, 0)."""
    return [p, 1 - p], [1 - p, p]


def rdp_curve_pair(p: Dist, q: Dist, alpha: float) -> float:
    """ε(α) for a mechanism with one pair of outputs up to symmetry: the larger of the two directions."""
    return max(renyi_discrete(p, q, alpha), renyi_discrete(q, p, alpha))


# --- RDP curves of the basic mechanisms -----------------------------------------------------


def gaussian_rdp(alpha: float, sigma: float, sensitivity: float = 1.0) -> float:
    """ε(α) = αΔ²/(2σ²) for the Gaussian mechanism (Mironov 2017, Corollary 3).

    `sensitivity` is the distance between the two query answers in units of the sensitivity the noise
    was calibrated to: 1 for neighbours, 2 for inputs two records apart.
    """
    return alpha * sensitivity**2 / (2 * sigma**2)


def laplace_rdp(alpha: float, b: float) -> float:
    """ε(α) of the Laplace mechanism with scale b on a sensitivity-1 real query (Mironov 2017, Corollary 2).

    (1/(α-1)) ln( α/(2α-1) e^{(α-1)/b} + (α-1)/(2α-1) e^{-α/b} ); α = 1 gives KL, α = ∞ gives 1/b.
    """
    t = 1 / b
    if alpha == 1:
        return t + math.exp(-t) - 1
    if math.isinf(alpha):
        return t
    terms = [math.log(alpha / (2 * alpha - 1)) + (alpha - 1) * t, math.log((alpha - 1) / (2 * alpha - 1)) - alpha * t]
    return logsumexp(terms) / (alpha - 1)


def laplace_rdp_shift(alpha: float, b: float, shift: float) -> float:
    """D_α(Lap(0, b)‖Lap(shift, b)): rescaling by 1/shift is a bijection, so it is the sensitivity-1 curve at scale b/shift."""
    return laplace_rdp(alpha, b / shift)


def randomized_response_rdp(alpha: float, p: float) -> float:
    """ε(α) = (1/(α-1)) ln( p^α (1-p)^{1-α} + (1-p)^α p^{1-α} ) for keep-probability p > 1/2."""
    return renyi_discrete(*rr_pair(p), alpha)


def renyi_numeric(
    log_p: Callable[[NDArray[np.float64]], NDArray[np.float64]],
    log_q: Callable[[NDArray[np.float64]], NDArray[np.float64]],
    alpha: float,
    lo: float,
    hi: float,
    n: int,
) -> float:
    """D_α(P‖Q) by the trapezoid rule on n + 1 points of [lo, hi], in log space; no closed form used."""
    x = np.linspace(lo, hi, n + 1)
    log_f = alpha * log_p(x) + (1 - alpha) * log_q(x)
    m = float(np.max(log_f))
    g = np.exp(log_f - m)
    h = (hi - lo) / n
    integral = h * (float(np.sum(g)) - 0.5 * float(g[0] + g[-1]))
    return (m + math.log(integral)) / (alpha - 1)


def laplace_rdp_numeric(alpha: float, b: float) -> float:
    """D_α(Lap(0, b)‖Lap(1, b)) by numerical integration; the grid has nodes at both kinks, 0 and 1."""
    half = 60 * b + 1
    n = int(round(2 * half * 10_000))  # step 1e-4, so 0 and 1 are grid points

    def log_p(x: NDArray[np.float64]) -> NDArray[np.float64]:
        return -np.abs(x) / b - math.log(2 * b)

    def log_q(x: NDArray[np.float64]) -> NDArray[np.float64]:
        return -np.abs(x - 1) / b - math.log(2 * b)

    return renyi_numeric(log_p, log_q, alpha, -half, half, n)


def gaussian_rdp_numeric(alpha: float, sigma: float) -> float:
    """D_α(N(0, σ²)‖N(1, σ²)) by numerical integration."""
    half = 40 * sigma + alpha + 2

    def log_p(x: NDArray[np.float64]) -> NDArray[np.float64]:
        return -(x**2) / (2 * sigma**2) - 0.5 * math.log(2 * math.pi * sigma**2)

    def log_q(x: NDArray[np.float64]) -> NDArray[np.float64]:
        return -((x - 1) ** 2) / (2 * sigma**2) - 0.5 * math.log(2 * math.pi * sigma**2)

    return renyi_numeric(log_p, log_q, alpha, -half, half, 400_000)


# --- Composition ----------------------------------------------------------------------------


def compose_rdp(curves: Sequence[Sequence[float]]) -> list[float]:
    """The RDP curve of an adaptive composition: the pointwise sum of the curves, each given on ORDERS."""
    return [float(sum(values)) for values in zip(*curves, strict=True)]


def curve_on_orders(curve: Curve, orders: Sequence[float] = ORDERS) -> list[float]:
    return [curve(a) for a in orders]


def adaptive_rr_joint(p_first: float, p_after_one: float, p_after_zero: float) -> tuple[list[float], list[float]]:
    """Joint law of (y1, y2) on the true bits 1 (P) and 0 (Q): y1 = RR_{p_first}(x), then y2 = RR_{p(y1)}(x).

    The second keep-probability depends on the first output. Outcomes in the order 11, 10, 01, 00.
    """
    p_out: list[float] = []
    q_out: list[float] = []
    for y1 in (1, 0):
        p2 = p_after_one if y1 == 1 else p_after_zero
        for y2 in (1, 0):
            p_out.append((p_first if y1 == 1 else 1 - p_first) * (p2 if y2 == 1 else 1 - p2))
            q_out.append((1 - p_first if y1 == 1 else p_first) * (1 - p2 if y2 == 1 else p2))
    return p_out, q_out


# --- Post-processing and group privacy ------------------------------------------------------


def weak_triangle_group2(curve: Curve, alpha: float) -> float:
    """Bound on order α for inputs two steps apart (Mironov 2017, Corollary 4):

    D_α(P‖R) ≤ (α - 1/2)/(α - 1) ε(2α) + ε(2α - 1), with ε the curve for neighbours.
    """
    return (alpha - 0.5) / (alpha - 1) * curve(2 * alpha) + curve(2 * alpha - 1)


def mironov_group_bound(curve: Curve, alpha: float, c: int) -> float:
    """Mironov 2017, Proposition 2: inputs 2^c steps apart are (α, 3^c ε(2^c α))-RDP at order α."""
    return 3**c * curve(2**c * alpha)


def iterated_weak_triangle(curve: Curve, c: int) -> Curve:
    """Apply the c = 1 bound c times: a curve for inputs 2^c steps apart."""

    def doubled(inner: Curve) -> Curve:
        def bound(a: float) -> float:
            return weak_triangle_group2(inner, a)

        return bound

    result = curve
    for _ in range(c):
        result = doubled(result)
    return result


def channel_output(source: Dist, rows: Sequence[Dist]) -> list[float]:
    """Push a distribution on inputs through a channel given by rows K(·|x)."""
    return [sum(s * row[j] for s, row in zip(source, rows, strict=True)) for j in range(len(rows[0]))]


# --- The moments accountant -----------------------------------------------------------------


def log_moment_discrete(p: Dist, q: Dist, lam: float) -> float:
    """Abadi et al.'s log-moment ln E_{Y~P}[exp(λ L(Y))] with L = ln(p/q), computed directly."""
    return logsumexp([math.log(pi) + lam * math.log(pi / qi) for pi, qi in zip(p, q, strict=True) if pi > 0])


def log_moment_from_rdp(curve: Curve, lam: float) -> float:
    """The same log-moment read off the RDP curve: λ ε(λ + 1)."""
    return lam * curve(lam + 1)


def log_moment_monte_carlo(rng: np.random.Generator, mu: float, lam: float, m: int) -> float:
    """ln of the sample mean of exp(λL(Y)) for P = N(0, 1), Q = N(μ, 1), with m draws Y ~ P."""
    y = rng.normal(0.0, 1.0, size=m)
    loss = (mu * mu - 2 * mu * y) / 2  # ln(p/q)(y) for P = N(0, 1), Q = N(μ, 1)
    return math.log(float(np.mean(np.exp(lam * loss))))


# --- From RDP to (ε, δ) ---------------------------------------------------------------------


def mironov_epsilon(rdp_eps: float, alpha: float, delta: float) -> float:
    """Mironov 2017, Proposition 3: (α, ε)-RDP gives (ε + ln(1/δ)/(α - 1), δ)-DP."""
    return rdp_eps + math.log(1 / delta) / (alpha - 1)


def rdp_to_dp_mironov(orders: Sequence[float], rdp: Sequence[float], delta: float) -> tuple[float, float]:
    """The best Mironov conversion over a grid of orders: returns (ε, the order that attains it)."""
    best = min((mironov_epsilon(e, a, delta), a) for a, e in zip(orders, rdp, strict=True))
    return best[0], best[1]


def gaussian_mironov_closed_form(sigma: float, delta: float) -> tuple[float, float]:
    """Continuous optimum for the Gaussian: α* = 1 + σ√(2 ln(1/δ)), ε = 1/(2σ²) + √(2 ln(1/δ))/σ."""
    root = math.sqrt(2 * math.log(1 / delta))
    return 1 / (2 * sigma**2) + root / sigma, 1 + sigma * root


def gaussian_delta_exact(eps: float, mu: float) -> float:
    """δ(ε) = Φ(μ/2 - ε/μ) - e^ε Φ(-μ/2 - ε/μ), the exact privacy curve of a Gaussian pair (Balle & Wang 2018)."""
    first = normal_cdf(mu / 2 - eps / mu)
    second = math.exp(min(eps + log_normal_cdf(-mu / 2 - eps / mu), 700.0))
    return max(first - second, 0.0)


def gaussian_eps_exact(mu: float, delta: float) -> float:
    """The smallest ε ≥ 0 with δ(ε) ≤ delta, by bisection (δ(ε) is decreasing in ε)."""
    if gaussian_delta_exact(0.0, mu) <= delta:
        return 0.0
    hi = 1.0
    while gaussian_delta_exact(hi, mu) > delta:
        hi *= 2
    return bisect(lambda e: gaussian_delta_exact(e, mu) - delta, 0.0, hi)


def advanced_composition(eps0: float, k: int, delta_prime: float) -> float:
    """Dwork, Rothblum & Vadhan (Dwork & Roth 2014, Theorem 3.20): ε0 √(2k ln(1/δ')) + k ε0 (e^ε0 - 1)."""
    return eps0 * math.sqrt(2 * k * math.log(1 / delta_prime)) + k * eps0 * math.expm1(eps0)


def advanced_composition_tanh(eps0: float, k: int, delta_prime: float) -> float:
    """The same theorem with the exact worst mean privacy loss of an ε0-DP step (Part 3):
    ε0 √(2k ln(1/δ')) + k ε0 tanh(ε0/2)."""
    return eps0 * math.sqrt(2 * k * math.log(1 / delta_prime)) + k * eps0 * math.tanh(eps0 / 2)


# Fraction s of δ spent on the k per-step δ0 (the rest is δ'): 10^(-1/4), 10^(-2/4), ..., 1e-10, the grid
# Part 3's composition.py uses, so that the advanced-composition value quoted as Part 3's answer is the same.
SPLITS: list[float] = [10 ** (-x / 4) for x in range(1, 41)]


def gaussian_advanced(sigma: float, k: int, delta: float, tanh: bool = True) -> tuple[float, float]:
    """Advanced composition of k Gaussian steps at total δ: each step is (ε0, δ0) from its exact curve,
    with k δ0 = s δ and δ' = (1 - s) δ; returns (ε, the best fraction s on the grid SPLITS).

    tanh=True uses the sharper mean-loss term ε0 tanh(ε0/2), tanh=False the textbook ε0(e^ε0 - 1)."""
    bound = advanced_composition_tanh if tanh else advanced_composition
    best = (math.inf, 0.0)
    for s in SPLITS:
        d0 = s * delta / k
        e0 = gaussian_eps_exact(1 / sigma, d0)
        best = min(best, (bound(e0, k, (1 - s) * delta), s))
    return best


def gaussian_basic(sigma: float, k: int, delta: float) -> float:
    """Basic composition: each step is (ε0, δ/k) from its exact curve, and the ε0 add up."""
    return k * gaussian_eps_exact(1 / sigma, delta / k)


def gaussian_rdp_composed(sigma: float, k: int, delta: float) -> tuple[float, float]:
    """RDP of k Gaussian steps (curves add), converted with Mironov's bound over ORDERS."""
    return rdp_to_dp_mironov(ORDERS, [k * gaussian_rdp(a, sigma) for a in ORDERS], delta)


def gaussian_exact_composed(sigma: float, k: int, delta: float) -> float:
    """k Gaussian steps are one Gaussian pair with μ = √k/σ, so their ε(δ) is exact."""
    return gaussian_eps_exact(math.sqrt(k) / sigma, delta)


# --- The chapter's numbers ------------------------------------------------------------------


def fmt_curve(curve: Curve, alphas: Sequence[float]) -> str:
    return "  ".join(f"eps({a:g})={curve(a):.4f}" for a in alphas)


def print_table(rng: np.random.Generator) -> None:
    print("[Why this matters: 1000 Gaussian steps, sigma = 20, delta = 1e-5]")
    adv, _ = gaussian_advanced(20.0, 1000, DELTA)
    rdp_eps, rdp_order = gaussian_rdp_composed(20.0, 1000, DELTA)
    print(
        f"  advanced={adv:.2f}  RDP+Mironov={rdp_eps:.2f} (alpha={rdp_order:g})  "
        f"exact={gaussian_exact_composed(20.0, 1000, DELTA):.2f}"
    )
    print(f"  textbook advanced (mean term eps0(e^eps0 - 1))={gaussian_advanced(20.0, 1000, DELTA, tanh=False)[0]:.2f}")
    print(f"  orders: {len(ORDERS)} from {ORDERS[0]:g} to {ORDERS[-1]:g}")

    print("[Definition: randomized response, p = 0.75 (eps = ln 3)]")
    p, q = rr_pair(0.75)
    print(f"  P={p}  Q={q}  ln r = {math.log(p[0] / q[0]):.4f}, {math.log(p[1] / q[1]):.4f}")
    alphas: list[float] = [1.0, 1.5, 2.0, 5.0, 10.0, 64.0, math.inf]
    print("  " + "  ".join(f"D_{a:g}={renyi_discrete(p, q, a):.4f}" for a in alphas))
    print("  reverse " + "  ".join(f"D_{a:g}={renyi_discrete(q, p, a):.4f}" for a in alphas))
    print(
        f"  alpha=2 by hand: terms {p[0] ** 2 / q[0]:.4f} + {p[1] ** 2 / q[1]:.4f} = {p[0] ** 2 / q[0] + p[1] ** 2 / q[1]:.4f}"
        f"  ln = {math.log(p[0] ** 2 / q[0] + p[1] ** 2 / q[1]):.4f}"
    )
    print(f"  alpha=1.0001: {renyi_discrete(p, q, 1.0001):.4f}  alpha=1000: {renyi_discrete(p, q, 1000.0):.4f}")

    print("[RDP curves of the basic mechanisms: Gaussian sigma = 1, Laplace b = 1, RR p = 0.75]")
    ks: list[float] = [1.5, 2.0, 10.0, 64.0]
    print("  Gaussian " + fmt_curve(lambda a: gaussian_rdp(a, 1.0), ks))
    print("  Gaussian numeric " + fmt_curve(lambda a: gaussian_rdp_numeric(a, 1.0), [2.0, 10.0]))
    print("  Laplace " + fmt_curve(lambda a: laplace_rdp(a, 1.0), ks) + f"  KL={laplace_rdp(1.0, 1.0):.4f}")
    print("  Laplace numeric " + fmt_curve(lambda a: laplace_rdp_numeric(a, 1.0), ks))
    worst = max(
        abs(laplace_rdp(a, b) - laplace_rdp_numeric(a, b)) for a in (1.5, 2.0, 10.0, 64.0) for b in (0.5, 1.0, 4.0)
    )
    print(f"  Laplace closed form vs numeric, b in (0.5, 1, 4): max abs diff={worst:.1e}")
    shifts = [i / 100 for i in range(1, 101)]
    grows = all(
        laplace_rdp_shift(a, 1.0, s1) < laplace_rdp_shift(a, 1.0, s2)
        for a in (1.5, 2.0, 10.0, 64.0)
        for s1, s2 in zip(shifts, shifts[1:])
    )
    print(f"  Laplace b=1: D_alpha grows with the shift on (0, 1] at orders 1.5, 2, 10, 64: {grows}")
    print("  RR " + fmt_curve(lambda a: randomized_response_rdp(a, 0.75), ks))
    print(
        f"  Laplace b=1 at large order: eps(512)={laplace_rdp(512.0, 1.0):.4f}  "
        f"1/b - ln 2/(alpha-1) = {1 - math.log(2) / 511:.4f}"
    )
    print(f"  RR at large order: eps(512)={randomized_response_rdp(512.0, 0.75):.4f}  ln 3={math.log(3):.4f}")

    print("[Composition: adaptive randomized response]")
    pa, qa = adaptive_rr_joint(0.75, 0.75, 0.6)
    print(f"  P(11,10,01,00)={[round(x, 4) for x in pa]}  Q={[round(x, 4) for x in qa]}")
    terms = [pi**2 / qi for pi, qi in zip(pa, qa, strict=True)]
    print(
        f"  alpha=2 terms p^2/q={[round(t, 4) for t in terms]}  sum={sum(terms):.4f}  D_2={renyi_discrete(pa, qa, 2.0):.4f}"
    )
    e1, e2a, e2b = (randomized_response_rdp(2.0, x) for x in (0.75, 0.75, 0.6))
    print(f"  eps1(2)={e1:.4f}  eps2(2)=max({e2a:.4f}, {e2b:.4f})  bound={e1 + max(e2a, e2b):.4f}")
    pn, qn = adaptive_rr_joint(0.75, 0.75, 0.75)
    print(f"  non-adaptive (both 0.75): D_2={renyi_discrete(pn, qn, 2.0):.4f}  2*eps(2)={2 * e1:.4f}")
    print("[Composition: k Gaussian steps]")
    for k in (10, 100, 1000):
        print(
            f"  k={k}: sigma=20 per step -> eps(alpha) = {k * gaussian_rdp(1.0, 20.0):.4f} alpha  "
            f"= one Gaussian with sigma={20 / math.sqrt(k):.4f}"
        )
    curves = [curve_on_orders(lambda a: gaussian_rdp(a, 20.0)) for _ in range(1000)]
    composed = compose_rdp(curves)
    print(f"  compose_rdp check: eps(2) after 1000 steps={composed[ORDERS.index(2.0)]:.4f}")
    lap_gauss = compose_rdp(
        [curve_on_orders(lambda a: laplace_rdp(a, 1.0)), curve_on_orders(lambda a: gaussian_rdp(a, 1.0))]
    )
    print(
        f"  Laplace b=1 then Gaussian sigma=1: eps(2)={lap_gauss[ORDERS.index(2.0)]:.4f}  "
        f"eps(10)={lap_gauss[ORDERS.index(10.0)]:.4f}"
    )

    print("[Post-processing: randomized response through a binary symmetric channel, flip 0.2]")
    rows = [[0.8, 0.2], [0.2, 0.8]]
    pp, qp = channel_output(p, rows), channel_output(q, rows)
    print(
        f"  outputs P={[round(x, 4) for x in pp]}  D_2 before={renyi_discrete(p, q, 2.0):.4f}  after={renyi_discrete(pp, qp, 2.0):.4f}"
        f"  D_inf before={renyi_discrete(p, q, math.inf):.4f}  after={renyi_discrete(pp, qp, math.inf):.4f}"
    )

    print("[Group privacy: Gaussian sigma = 1, inputs 2^c steps apart]")
    g: Curve = lambda a: gaussian_rdp(a, 1.0)
    for beta in (2.0, 4.0, 10.0):
        print(
            f"  c=1, order {beta:g}: exact={gaussian_rdp(beta, 1.0, 2.0):.4f}  "
            f"weak triangle={weak_triangle_group2(g, beta):.4f}  Prop 2={mironov_group_bound(g, beta, 1):.4f}"
        )
    for c in (1, 2, 3):
        beta = 2.0
        print(
            f"  c={c}, order {beta:g}: exact={gaussian_rdp(beta, 1.0, 2.0**c):.4f}  "
            f"iterated triangle={iterated_weak_triangle(g, c)(beta):.4f}  Prop 2={mironov_group_bound(g, beta, c):.4f}  "
            f"ratio Prop 2/exact={mironov_group_bound(g, beta, c) / gaussian_rdp(beta, 1.0, 2.0**c):.4f}"
        )
    lap: Curve = lambda a: laplace_rdp(a, 1.0)
    print(
        f"  Laplace b=1, c=1, order 2: exact={laplace_rdp(2.0, 0.5):.4f}  weak triangle={weak_triangle_group2(lap, 2.0):.4f}  "
        f"Prop 2={mironov_group_bound(lap, 2.0, 1):.4f}  group eps-DP bound 2/b={2.0:.4f}"
    )

    print("[Moments accountant]")
    for lam in (1.0, 2.0):
        print(
            f"  RR p=0.75, lambda={lam:g}: direct={log_moment_discrete(p, q, lam):.4f}  "
            f"lambda*eps(lambda+1)={log_moment_from_rdp(lambda a: randomized_response_rdp(a, 0.75), lam):.4f}"
        )
    for lam in (1.0, 2.0):
        mc = log_moment_monte_carlo(rng, 1.0, lam, 1_000_000)
        print(
            f"  Gaussian mu=1, lambda={lam:g}: lambda(lambda+1)mu^2/2={lam * (lam + 1) / 2:.4f}  "
            f"lambda*eps(lambda+1)={log_moment_from_rdp(lambda a: gaussian_rdp(a, 1.0), lam):.4f}  Monte Carlo={mc:.4f}"
        )

    print("[From RDP to (eps, delta): one Gaussian, sigma = 1, delta = 1e-5]")
    g_curve = curve_on_orders(g)
    eps_grid, a_grid = rdp_to_dp_mironov(ORDERS, g_curve, DELTA)
    eps_cf, a_cf = gaussian_mironov_closed_form(1.0, DELTA)
    exact1 = gaussian_eps_exact(1.0, DELTA)
    print(f"  ln(1/delta)={math.log(1 / DELTA):.4f}  sqrt(2 ln(1/delta))={math.sqrt(2 * math.log(1 / DELTA)):.4f}")
    print(
        f"  grid: eps={eps_grid:.4f} at alpha={a_grid:g}  closed form: eps={eps_cf:.4f} at alpha*={a_cf:.4f}  "
        f"exact={exact1:.4f}  ratio={eps_grid / exact1:.3f}"
    )
    for a in (2.0, 5.8, 20.0):
        print(
            f"  alpha={a:g}: eps(alpha)={gaussian_rdp(a, 1.0):.4f} + ln(1/delta)/(alpha-1)={math.log(1 / DELTA) / (a - 1):.4f}"
            f" = {mironov_epsilon(gaussian_rdp(a, 1.0), a, DELTA):.4f}"
        )
    print(f"  exact delta at the RDP eps: {gaussian_delta_exact(eps_grid, 1.0):.3e}")
    print("[From RDP to (eps, delta): k Gaussian steps, sigma = 20, delta = 1e-5]")
    for k in (1, 10, 100, 1000):
        adv, split = gaussian_advanced(20.0, k, DELTA)
        r_eps, r_a = gaussian_rdp_composed(20.0, k, DELTA)
        cf, a_star = gaussian_mironov_closed_form(20.0 / math.sqrt(k), DELTA)
        ex = gaussian_exact_composed(20.0, k, DELTA)
        print(
            f"  k={k}: basic={gaussian_basic(20.0, k, DELTA):.3f}  advanced={adv:.3f} (split s={split:.2g})  RDP={r_eps:.3f} (alpha={r_a:g}; closed form {cf:.3f} "
            f"at {a_star:.1f})  exact={ex:.3f}  RDP/exact={r_eps / ex:.3f}  advanced/exact={adv / ex:.2f}"
        )
    print("[Conversion slack against the composed mu = sqrt(k)/sigma, continuous optimum]")
    for mu in (0.01, 0.1, 1.0, 10.0):
        cf_mu, _ = gaussian_mironov_closed_form(1 / mu, DELTA)
        print(
            f"  mu={mu:g}: Mironov={cf_mu:.4g}  exact={gaussian_eps_exact(mu, DELTA):.4g}  "
            f"ratio={cf_mu / gaussian_eps_exact(mu, DELTA):.2f}"
        )
    print("[Check yourself]")
    print(
        f"  composed curves alpha/8 + alpha/2: sigma=sqrt(0.8)={math.sqrt(0.8):.4f}  "
        f"adaptive gap={2 * e1 - renyi_discrete(pa, qa, 2.0):.4f}  "
        f"(10, 0.5)-RDP at delta=1e-5: eps={mironov_epsilon(0.5, 10.0, DELTA):.4f}"
    )
    print(f"  delta promised / exact delta at the RDP eps (sigma=1): {DELTA / gaussian_delta_exact(eps_grid, 1.0):.1f}")
    print("[Full-batch DP-SGD: k = 10000 Gaussian steps, sigma = 1, no subsampling]")
    r_eps, r_a = gaussian_rdp_composed(1.0, 10_000, DELTA)
    cf, a_star = gaussian_mironov_closed_form(1.0 / 100, DELTA)
    print(
        f"  RDP on ORDERS={r_eps:.1f} (alpha={r_a:g})  closed form={cf:.1f} at alpha*={a_star:.3f}  "
        f"exact={gaussian_exact_composed(1.0, 10_000, DELTA):.1f}"
    )


# --- Figures --------------------------------------------------------------------------------


def plot_rr_curve(p: float = 0.75) -> Figure:
    alphas = [1 + 10 ** (x / 40) for x in range(-120, 121)]  # 1.001 .. 101
    pp, qq = rr_pair(p)
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    ax.plot(
        alphas,
        [randomized_response_rdp(a, p) for a in alphas],
        "C0",
        linewidth=2,
        label=r"$\varepsilon(\alpha) = D_\alpha(P\,\|\,Q)$",
    )
    ax.axhline(
        renyi_discrete(pp, qq, 1.0),
        color="C1",
        linestyle="--",
        label=rf"$\alpha \to 1$: KL $= {renyi_discrete(pp, qq, 1.0):.3f}$",
    )
    ax.axhline(
        math.log(p / (1 - p)),
        color="C3",
        linestyle=":",
        label=rf"$\alpha \to \infty$: $\ln 3 = {math.log(p / (1 - p)):.3f}$, the $\varepsilon$ of $\varepsilon$-DP",
    )
    for a in (2.0, 10.0):
        ax.plot([a], [randomized_response_rdp(a, p)], "ko", markersize=4)
        ax.annotate(
            rf"$\varepsilon({a:g}) = {randomized_response_rdp(a, p):.3f}$",
            (a, randomized_response_rdp(a, p)),
            textcoords="offset points",
            xytext=(6, -14),
            fontsize=9,
        )
    ax.set(
        xscale="log",
        xlabel=r"order $\alpha$",
        ylabel=r"$\varepsilon(\alpha)$ (nats)",
        ylim=(0, 1.25),
        xlim=(1, 101),
        title=f"RDP curve of randomized response, p = {p:g}",
    )
    ax.legend(loc="lower right", fontsize=9)
    return fig


def plot_curves() -> Figure:
    alphas = [1 + 10 ** (x / 40) for x in range(-120, 121)]
    fig, ax = plt.subplots(figsize=(7.5, 4.4))
    ax.plot(
        alphas, [gaussian_rdp(a, 1.0) for a in alphas], "C0", linewidth=2, label=r"Gaussian, $\sigma = 1$: $\alpha/2$"
    )
    ax.plot(alphas, [laplace_rdp(a, 1.0) for a in alphas], "C1", linewidth=2, label=r"Laplace, $b = 1$")
    ax.plot(
        alphas,
        [randomized_response_rdp(a, 0.75) for a in alphas],
        "C2",
        linewidth=2,
        label=r"randomized response, $p = 0.75$",
    )
    ax.axhline(1.0, color="C1", linestyle=":", linewidth=1)
    ax.axhline(math.log(3), color="C2", linestyle=":", linewidth=1)
    ax.text(1.05, 1.0 - 0.13, r"$1/b = 1$", color="C1", fontsize=9)
    ax.text(1.05, math.log(3) + 0.05, r"$\ln 3$", color="C2", fontsize=9)
    ax.set(
        xscale="log",
        xlabel=r"order $\alpha$",
        ylabel=r"$\varepsilon(\alpha)$ (nats)",
        ylim=(0, 2.5),
        xlim=(1, 101),
        title="RDP curves: bounded noise levels off, Gaussian noise grows linearly",
    )
    ax.legend(loc="upper left", fontsize=9)
    return fig


def plot_group(sigma: float = 1.0) -> Figure:
    betas = [1.05 + i * 0.05 for i in range(180)]
    valid = [b for b in betas if b >= 2 - 1e-9]  # Proposition 2 needs α = 2β ≥ 4
    g: Curve = lambda a: gaussian_rdp(a, sigma)
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    ax.plot(
        betas,
        [gaussian_rdp(b, sigma, 2.0) for b in betas],
        "k",
        linewidth=2,
        label=r"exact: shift 2, $2\beta/\sigma^2$",
    )
    ax.plot(
        betas,
        [weak_triangle_group2(g, b) for b in betas],
        "C0--",
        label=r"weak triangle: $\frac{\beta - 1/2}{\beta - 1}\varepsilon(2\beta) + \varepsilon(2\beta - 1)$",
    )
    ax.plot(
        valid,
        [mironov_group_bound(g, b, 1) for b in valid],
        "C3:",
        linewidth=2,
        label=r"Proposition 2 (needs $\beta \geq 2$): $3\,\varepsilon(2\beta)$",
    )
    ax.axvline(2.0, color="0.6", linewidth=0.8)
    ax.set(
        xlabel=r"order $\beta$ of the group guarantee",
        ylabel=r"$\varepsilon$ for inputs two steps apart",
        xlim=(1, 10),
        ylim=(0, 32),
        title=rf"Group privacy for 2 records, Gaussian $\sigma$ = {sigma:g}",
    )
    ax.legend(loc="upper left", fontsize=9)
    return fig


def plot_conversion(sigma: float = 20.0, k_max: int = 1000, delta: float = DELTA) -> Figure:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.4))
    sig_eff = sigma / math.sqrt(k_max)
    alphas = [a for a in ORDERS if a <= 40]
    rdp = [gaussian_rdp(a, sig_eff) for a in alphas]
    tail = [math.log(1 / delta) / (a - 1) for a in alphas]
    total = [r + t for r, t in zip(rdp, tail, strict=True)]
    best_eps, best_a = rdp_to_dp_mironov(ORDERS, [gaussian_rdp(a, sig_eff) for a in ORDERS], delta)
    ax1.plot(alphas, rdp, "C0--", label=r"$\varepsilon(\alpha)$, the RDP curve")
    ax1.plot(alphas, tail, "C1:", linewidth=2, label=r"$\ln(1/\delta)/(\alpha - 1)$")
    ax1.plot(alphas, total, "k", linewidth=2, label="their sum: the converted ε")
    ax1.plot([best_a], [best_eps], "ko")
    ax1.annotate(
        rf"best order $\alpha = {best_a:g}$: $\varepsilon = {best_eps:.2f}$",
        (best_a, best_eps),
        textcoords="data",
        xytext=(12, 3),
        fontsize=9,
        arrowprops={"arrowstyle": "-", "color": "0.4"},
    )
    exact = gaussian_exact_composed(sigma, k_max, delta)
    ax1.axhline(exact, color="C2", linewidth=1.5, label=rf"exact $\varepsilon = {exact:.2f}$")
    ax1.set(
        xlabel=r"order $\alpha$",
        ylabel=r"$\varepsilon$",
        ylim=(0, 40),
        xlim=(1, 40),
        title=rf"Converting: k = {k_max}, $\sigma$ = {sigma:g}, δ = {delta:g}",
    )
    ax1.legend(fontsize=8, loc="center right")

    ks = sorted({int(round(10 ** (x / 8))) for x in range(25)})
    ax2.plot(ks, [gaussian_advanced(sigma, k, delta)[0] for k in ks], "C3-.", label="advanced composition, best split")
    ax2.plot(
        ks,
        [gaussian_rdp_composed(sigma, k, delta)[0] for k in ks],
        "C0--",
        linewidth=2,
        label="RDP + Mironov, best order",
    )
    ax2.plot(
        ks,
        [gaussian_exact_composed(sigma, k, delta) for k in ks],
        "C2",
        linewidth=2,
        label="exact: one Gaussian, μ = √k / σ",
    )
    ax2.set(
        xscale="log",
        yscale="log",
        xlabel="number of Gaussian steps k",
        ylabel=r"total $\varepsilon$",
        title=rf"Composing Gaussians, $\sigma$ = {sigma:g}, δ = {delta:g}",
    )
    ax2.legend(fontsize=8, loc="upper left")
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
        "renyi_dp_rr_curve.svg": plot_rr_curve(),
        "renyi_dp_curves.svg": plot_curves(),
        "renyi_dp_group.svg": plot_group(),
        "renyi_dp_conversion.svg": plot_conversion(),
    }
    for name, fig in figures.items():
        fig.tight_layout()
        fig.savefig(out / name, metadata={"Date": None})  # no timestamp, so output is reproducible
        plt.close(fig)
        print(f"saved {out / name}")


if __name__ == "__main__":
    main()
