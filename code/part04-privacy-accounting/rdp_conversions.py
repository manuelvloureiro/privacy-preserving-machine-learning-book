"""Check the conversions in the RDP-conversions chapter and draw its figures.

    uv run python code/part04-privacy-accounting/rdp_conversions.py
    uv run python code/part04-privacy-accounting/rdp_conversions.py --seed 1 --out /tmp/figures

Prints the numbers quoted in 02-rdp-conversions.md (they match exactly with the default seed),
then saves the chapter's figures.

Conventions: P = M(D), Q = M(D') are the output distributions on neighbouring datasets and
L = ln p/q is the privacy loss, so Y ~ P makes L the privacy loss random variable. An RDP curve is
given as two parallel sequences, the orders α > 1 and the values ε(α). In the hypothesis-testing
section, α is always the Rényi order: the type I error of a test (reject "the output came from P"
although it did) is called a, and its type II error is called b.

The Gaussian mechanism with noise multiplier σ composed k times is, for every question asked here,
the pair P = N(0, 1), Q = N(μ, 1) with μ = √k/σ: its RDP curve is ε(α) = αμ²/2.
"""

import argparse
import math
from collections.abc import Callable, Sequence
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
matplotlib.rcParams["svg.hashsalt"] = "rdp_conversions"  # stable element ids, so reruns don't churn the SVGs
matplotlib.rcParams["svg.fonttype"] = "none"  # keep text as text: smaller, diffable files

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from numpy.typing import NDArray

# Figures are build output: _build/figures/<part>/, which the build copies into each book.
FIGURES = Path(__file__).resolve().parents[2] / "_build" / "figures" / Path(__file__).parent.name

# The RDP orders grid shared by the whole part.
ORDERS: list[float] = [1 + x / 10 for x in range(1, 100)] + [float(a) for a in range(12, 64)] + [128.0, 256.0, 512.0]

DELTA = 1e-5  # the part's target δ

Array = NDArray[np.float64]


# --- Normal distribution helpers ------------------------------------------------------------


def normal_cdf(x: float) -> float:
    """Φ(x); erfc keeps full relative precision in the lower tail."""
    return 0.5 * math.erfc(-x / math.sqrt(2))


def log_normal_cdf(x: float) -> float:
    """ln Φ(x), finite even where Φ(x) underflows (x below about -37)."""
    if x > -30:
        return math.log(normal_cdf(x))
    # Asymptotic series Φ(x) = φ(x)/|x| (1 - 1/x² + 3/x⁴ - 15/x⁶ + ...), accurate to 1e-10 here.
    x2 = x * x
    series = 1 - 1 / x2 + 3 / x2**2 - 15 / x2**3
    return -x2 / 2 - math.log(-x * math.sqrt(2 * math.pi)) + math.log(series)


def normal_quantile_upper(p: float) -> float:
    """z with 1 - Φ(z) = p, by bisection on the tail (full precision for tiny p)."""
    return bisect(lambda z: normal_cdf(-z) - p, -40.0, 40.0)


def bisect(f: Callable[[float], float], lo: float, hi: float, steps: int = 200) -> float:
    """A root of a decreasing-or-increasing f on [lo, hi] (f(lo) and f(hi) have opposite signs)."""
    f_lo = f(lo)
    for _ in range(steps):
        mid = (lo + hi) / 2
        if (f(mid) > 0) == (f_lo > 0):
            lo, f_lo = mid, f(mid)
        else:
            hi = mid
        if hi - lo <= 1e-13 * max(1.0, abs(lo)):
            break
    return (lo + hi) / 2


# --- Toolkit: the Gaussian pair, exactly ----------------------------------------------------


def gaussian_rdp(alpha: float, mu: float) -> float:
    """ε(α) = D_α(N(0,1)‖N(μ,1)) = αμ²/2."""
    return alpha * mu * mu / 2


def gaussian_rdp_curve(mu: float, orders: Sequence[float] = ORDERS) -> list[float]:
    return [gaussian_rdp(a, mu) for a in orders]


def gaussian_delta_exact(eps: float, mu: float) -> float:
    """δ(ε) = E_{e^ε}(P‖Q) = Φ(μ/2 - ε/μ) - e^ε Φ(-μ/2 - ε/μ), computed in logs where needed."""
    first = normal_cdf(mu / 2 - eps / mu)
    log_second = eps + log_normal_cdf(-mu / 2 - eps / mu)
    return max(first - math.exp(log_second), 0.0)


def gaussian_tail(eps: float, mu: float) -> float:
    """P[L > ε] for the Gaussian pair: L ~ N(μ²/2, μ²) under P, so this is Φ(μ/2 - ε/μ)."""
    return normal_cdf(mu / 2 - eps / mu)


def gaussian_eps_exact(mu: float, delta: float) -> float:
    """The smallest ε with δ(ε) ≤ δ for the Gaussian pair, by bisection (δ(ε) decreases in ε)."""
    hi = mu * mu / 2 + 40 * mu + 1
    return bisect(lambda e: gaussian_delta_exact(e, mu) - delta, 0.0, hi)


def gaussian_eps_tail(mu: float, delta: float) -> float:
    """The ε at which P[L > ε] = δ: ε = μ²/2 + μ z with 1 - Φ(z) = δ."""
    return mu * mu / 2 + mu * normal_quantile_upper(delta)


def pld_hockey_stick_numeric(eps: float, mu: float, n: int = 400_001) -> float:
    """E_P[(1 - e^{ε - L})_+] with L ~ N(μ²/2, μ²), by a Riemann sum: a check of Fact 1."""
    z = np.linspace(-12.0, 12.0, n)
    loss = mu * mu / 2 + mu * z
    density = np.exp(-z * z / 2) / math.sqrt(2 * math.pi)
    integrand = np.maximum(1 - np.exp(np.minimum(eps - loss, 50.0)), 0.0) * density
    return float(np.sum(integrand) * (z[1] - z[0]))


# --- Section 28.3: Mironov's conversion -----------------------------------------------------------


def mironov_eps(alpha: float, rdp: float, delta: float) -> float:
    """ε' = ε(α) + ln(1/δ)/(α - 1) (Mironov 2017, Prop. 3)."""
    return rdp + math.log(1 / delta) / (alpha - 1)


def mironov_delta(alpha: float, rdp: float, eps: float) -> float:
    """δ = e^{(α - 1)(ε(α) - ε)}, capped at 1."""
    return math.exp(min((alpha - 1) * (rdp - eps), 0.0))


def rdp_to_dp_mironov(orders: Sequence[float], rdp: Sequence[float], delta: float) -> tuple[float, float]:
    """Mironov's conversion minimized over the given orders: returns (ε, best order)."""
    best = min(zip(orders, rdp, strict=True), key=lambda ar: mironov_eps(ar[0], ar[1], delta))
    return mironov_eps(best[0], best[1], delta), best[0]


def rdp_to_delta_mironov(orders: Sequence[float], rdp: Sequence[float], eps: float) -> float:
    return min(mironov_delta(a, r, eps) for a, r in zip(orders, rdp, strict=True))


def mironov_alpha_gaussian(rho: float, delta: float) -> float:
    """The continuous minimizer of αρ + ln(1/δ)/(α - 1): α* = 1 + √(ln(1/δ)/ρ)."""
    return 1 + math.sqrt(math.log(1 / delta) / rho)


def mironov_eps_gaussian(rho: float, delta: float) -> float:
    """Its value: ε* = ρ + 2√(ρ ln(1/δ))."""
    return rho + 2 * math.sqrt(rho * math.log(1 / delta))


# --- Section 28.4: the improved conversion --------------------------------------------------------


def improved_eps(alpha: float, rdp: float, delta: float) -> float:
    """ε' = ε(α) + ln((α - 1)/α) - (ln δ + ln α)/(α - 1)  (Balle et al. 2020, Thm 21; CKS 2020, Prop. 12)."""
    return max(0.0, rdp + math.log((alpha - 1) / alpha) - (math.log(delta) + math.log(alpha)) / (alpha - 1))


def improved_delta(alpha: float, rdp: float, eps: float) -> float:
    """δ = e^{(α - 1)(ε(α) - ε)} / α · (1 - 1/α)^{α - 1}, capped at 1."""
    log_delta = (alpha - 1) * (rdp - eps) - math.log(alpha) + (alpha - 1) * math.log(1 - 1 / alpha)
    return math.exp(min(log_delta, 0.0))


def rdp_to_dp_improved(orders: Sequence[float], rdp: Sequence[float], delta: float) -> tuple[float, float]:
    """The improved conversion minimized over the given orders: returns (ε, best order)."""
    best = min(zip(orders, rdp, strict=True), key=lambda ar: improved_eps(ar[0], ar[1], delta))
    return improved_eps(best[0], best[1], delta), best[0]


def rdp_to_delta_improved(orders: Sequence[float], rdp: Sequence[float], eps: float) -> float:
    return min(improved_delta(a, r, eps) for a, r in zip(orders, rdp, strict=True))


def improvement(alpha: float) -> float:
    """ε_Mironov - ε_improved at the same order: ln α/(α - 1) + ln(α/(α - 1)), independent of ε(α) and δ."""
    return math.log(alpha) / (alpha - 1) + math.log(alpha / (alpha - 1))


def delta_ratio(alpha: float) -> float:
    """δ_improved / δ_Mironov at the same order and ε: (1/α)(1 - 1/α)^{α - 1}."""
    return (1 - 1 / alpha) ** (alpha - 1) / alpha


def hockey_integrand(z: float, eps: float) -> float:
    """The hockey-stick integrand (1 - e^{ε - z})_+ as a function of the privacy loss z."""
    return max(0.0, 1 - math.exp(eps - z))


def mironov_majorant(z: float, alpha: float, eps: float) -> float:
    """1[z > ε] ≤ e^{(α - 1)(z - ε)}: the Markov majorant behind Mironov's conversion."""
    return math.exp((alpha - 1) * (z - eps))


def best_constant(alpha: float, eps: float) -> float:
    """The smallest c with (1 - e^{ε - z})_+ ≤ c e^{(α - 1)z} for all z: e^{-(α-1)ε}(1 - 1/α)^α/(α - 1)."""
    return math.exp(-(alpha - 1) * eps) * (1 - 1 / alpha) ** alpha / (alpha - 1)


def best_constant_numeric(alpha: float, eps: float) -> float:
    """The same constant by brute force: max over a fine grid of z of (1 - e^{ε - z})_+ e^{-(α - 1)z}."""
    z = np.linspace(eps, eps + 20.0, 2_000_001)
    return float(np.max((1 - np.exp(eps - z)) * np.exp(-(alpha - 1) * z)))


def improved_majorant(z: float, alpha: float, eps: float) -> float:
    return best_constant(alpha, eps) * math.exp((alpha - 1) * z)


def improved_alpha_gaussian(rho: float, delta: float) -> float:
    """The continuous minimizer of the improved ε'(α) for ε(α) = αρ: ρ(α - 1)² = ln(1/(αδ)), by fixed point."""
    alpha = mironov_alpha_gaussian(rho, delta)
    for _ in range(200):
        alpha = 1 + math.sqrt(max(math.log(1 / (alpha * delta)), 0.0) / rho)
    return alpha


# --- Section 28.5: RDP as a constraint on tests ---------------------------------------------------


def _log_sigmoid(s: Array) -> Array:
    """ln(1/(1 + e^{-s})), stable for any s."""
    return -np.logaddexp(0.0, -s)


def _min_type2_logit(orders: Array, rdp: Array, log_a: float, log_abar: float, directions: int = 2) -> float:
    """The smallest type II error b consistent with the RDP curve at type I error a, as logit(b).

    a is passed as (ln a, ln(1 - a)) so that a near 0 or near 1 keeps full precision. A test with
    errors (a, b) turns P into Bern(reject: a) and Q into Bern(reject: 1 - b); data processing gives,
    for each order α,
        forward:  a^α (1-b)^{1-α} + (1-a)^α b^{1-α} ≤ e^{(α-1)ε(α)}       (D_α(P‖Q) ≤ ε(α)),
        reverse:  (1-b)^α a^{1-α} + b^α (1-a)^{1-α} ≤ e^{(α-1)ε(α)}       (D_α(Q‖P) ≤ ε(α)).
    Both left sides equal 1 at b = 1 - a and are convex in b, so they decrease on [0, 1 - a]: bisect
    on s = logit(b) ∈ (-∞, ln((1-a)/a)]. directions=1 keeps only the forward constraint.
    """
    finite = np.isfinite(rdp)
    al, budget = orders[finite], (orders[finite] - 1) * rdp[finite]
    hi = log_abar - log_a  # b = 1 - a: always feasible
    lo = -3000.0

    def feasible(s: float) -> bool:
        sa = np.array([s])
        log_b, log_bbar = float(_log_sigmoid(sa)[0]), float(_log_sigmoid(-sa)[0])
        fwd = np.logaddexp(al * log_a + (1 - al) * log_bbar, al * log_abar + (1 - al) * log_b)
        ok = bool(np.all(fwd <= budget + 1e-12))
        if directions == 2 and ok:
            rev = np.logaddexp(al * log_bbar + (1 - al) * log_a, al * log_b + (1 - al) * log_abar)
            ok = bool(np.all(rev <= budget + 1e-12))
        return ok

    if feasible(lo):
        return lo
    for _ in range(80):
        mid = (lo + hi) / 2
        if feasible(mid):
            hi = mid
        else:
            lo = mid
    return hi


def rdp_tradeoff_lower_bound(orders: Sequence[float], rdp: Sequence[float], alpha_err: float) -> float:
    """The smallest type II error any test can have at type I error `alpha_err`, given only the RDP curve.

    Every pair (P, Q) whose Rényi divergences in both directions are at most rdp at the given orders
    has trade-off function T(P, Q) ≥ this bound, and a two-outcome pair attains it.
    """
    if alpha_err <= 0:
        return 1.0 if all(math.isfinite(r) for r in rdp) else 0.0  # a = 0: Q has no mass where P has none
    if alpha_err >= 1:
        return 0.0
    s = _min_type2_logit(np.asarray(orders, float), np.asarray(rdp, float), math.log(alpha_err), math.log1p(-alpha_err))
    return 1 / (1 + math.exp(-s))


def _log_min_type2_at_power(orders: Array, rdp: Array, log_u: float, directions: int = 2) -> float:
    """ln b_min at type I error a = 1 - u, for u = 1 - a given by its logarithm (u may be tiny)."""
    s = _min_type2_logit(orders, rdp, math.log1p(-math.exp(log_u)), log_u, directions)
    return float(_log_sigmoid(np.array([s]))[0])


def _golden_max(f: Callable[[float], float], lo: float, hi: float, steps: int = 60) -> tuple[float, float]:
    """(argmax, max) of a unimodal f on [lo, hi]."""
    g = (math.sqrt(5) - 1) / 2
    x1, x2 = hi - g * (hi - lo), lo + g * (hi - lo)
    f1, f2 = f(x1), f(x2)
    for _ in range(steps):
        if f1 < f2:
            lo, x1, f1 = x1, x2, f2
            x2 = lo + g * (hi - lo)
            f2 = f(x2)
        else:
            hi, x2, f2 = x2, x1, f1
            x1 = hi - g * (hi - lo)
            f1 = f(x1)
    return (x1, f1) if f1 > f2 else (x2, f2)


def _grid_then_golden(f: Callable[[float], float], lo: float, hi: float, n: int = 80) -> tuple[float, float]:
    """Maximize f: a coarse grid locates the peak, golden-section search refines it."""
    xs = [lo + (hi - lo) * i / (n - 1) for i in range(n)]
    values = [f(x) for x in xs]
    i = max(range(n), key=lambda j: values[j])
    return _golden_max(f, xs[max(i - 1, 0)], xs[min(i + 1, n - 1)], 40)


def rdp_to_delta_tradeoff(
    orders: Sequence[float], rdp: Sequence[float], eps: float, directions: int = 2
) -> tuple[float, float, float]:
    """The largest δ(ε) of any pair consistent with the RDP curve: max over tests of 1 - a - e^ε b.

    Returns (δ, u, b) with u = 1 - a at the optimum. The two-outcome pair P = Bern(u), Q = Bern(b)
    (probability of the outcome "accept") has exactly these Rényi constraints, so this δ is attained.
    """
    o, r = np.asarray(orders, float), np.asarray(rdp, float)

    def objective(log_u: float) -> float:
        return math.exp(log_u) - math.exp(eps + _log_min_type2_at_power(o, r, log_u, directions))

    log_u, value = _grid_then_golden(objective, -60.0, -1e-9)
    return max(value, 0.0), math.exp(log_u), math.exp(_log_min_type2_at_power(o, r, log_u, directions))


def rdp_to_dp_tradeoff(orders: Sequence[float], rdp: Sequence[float], delta: float, directions: int = 2) -> float:
    """The smallest ε that every pair consistent with the RDP curve satisfies at δ.

    ε = max over tests with 1 - a > δ of ln((1 - a - δ)/b): a single maximization over u = 1 - a.
    """
    o, r = np.asarray(orders, float), np.asarray(rdp, float)
    log_delta = math.log(delta)

    def objective(t: float) -> float:  # t = ln(u - δ)
        log_u = float(np.logaddexp(t, log_delta))
        return t - _log_min_type2_at_power(o, r, min(log_u, -1e-12), directions)

    _, value = _grid_then_golden(objective, log_delta - 20, math.log1p(-delta) - 1e-9)
    return max(value, 0.0)


def single_order_eps_optimal(alpha: float, rdp: float, delta: float) -> float:
    """The optimal conversion from one Rényi constraint D_α(P‖Q) ≤ ε(α) (forward direction only)."""
    return rdp_to_dp_tradeoff([alpha], [rdp], delta, directions=1)


def gaussian_tradeoff(a: float, mu: float) -> float:
    """G_μ(a) = Φ(Φ^{-1}(1 - a) - μ), the exact trade-off of N(0,1) against N(μ,1)."""
    return normal_cdf(normal_quantile_upper(a) - mu)


# --- Section 28.6: choosing the order -------------------------------------------------------------


def eps_curve(orders: Sequence[float], rdp: Sequence[float], delta: float, method: str) -> list[float]:
    """ε'(α) at every order, for method 'mironov' or 'improved'."""
    convert = mironov_eps if method == "mironov" else improved_eps
    return [convert(a, r, delta) for a, r in zip(orders, rdp, strict=True)]


# --- Section 28.7: the slack no conversion removes ------------------------------------------------


def release_mechanism_rdp(delta: float, alpha: float) -> float:
    """D_α(P‖Q) for the mechanism that publishes the record with probability δ: P has an atom Q lacks."""
    p_atom = delta
    return math.inf if p_atom > 0 and alpha > 1 else 0.0


def randomized_response_rdp(alpha: float, eps: float) -> float:
    """D_α(Bern(e^ε/(1+e^ε)) ‖ Bern(1/(1+e^ε))), the RDP of ε-randomized response."""
    p = math.exp(eps) / (1 + math.exp(eps))
    total = p**alpha * (1 - p) ** (1 - alpha) + (1 - p) ** alpha * p ** (1 - alpha)
    return math.log(total) / (alpha - 1)


def pure_dp_rdp_bound(alpha: float, eps: float) -> float:
    """ε-DP ⇒ (α, min(ε, αε²/2))-RDP (Bun & Steinke 2016, Prop. 3.3, and monotonicity in α)."""
    return min(eps, alpha * eps * eps / 2)


def dense_orders(step: float = 0.002, top: float = 30.0) -> list[float]:
    """Orders from 1.01 to `top` in steps of `step`, every integer up to 512, and the part's grid."""
    n = round((top - 1.01) / step)
    fine = [1.01 + step * i for i in range(n + 1)]
    return sorted(set(fine + [float(a) for a in range(2, 513)] + ORDERS))


def rdp_worst_pair_at_delta(orders: Sequence[float], rdp: Sequence[float], delta: float) -> tuple[float, float]:
    """The two-outcome pair (P(accept), Q(accept)) at the optimum of rdp_to_dp_tradeoff.

    Its ε at δ, ln((P(accept) - δ)/Q(accept)), is the optimal conversion from these RDP values.
    """
    o, r = np.asarray(orders, float), np.asarray(rdp, float)
    log_delta = math.log(delta)

    def objective(t: float) -> float:  # t = ln(u - δ), as in rdp_to_dp_tradeoff
        log_u = float(np.logaddexp(t, log_delta))
        return t - _log_min_type2_at_power(o, r, min(log_u, -1e-12))

    t, _ = _grid_then_golden(objective, log_delta - 20, math.log1p(-delta) - 1e-9)
    log_u = min(float(np.logaddexp(t, log_delta)), -1e-12)
    return math.exp(log_u), math.exp(_log_min_type2_at_power(o, r, log_u))


def gaussian_curve_excess(p: float, q: float, mu: float, alphas: Array) -> Array:
    """D_α(Bern(p)‖Bern(q)) - αμ²/2 at each order: positive where the pair violates the Gaussian's curve."""
    t1 = alphas * math.log1p(-p) + (1 - alphas) * math.log1p(-q)
    t2 = alphas * math.log(p) + (1 - alphas) * math.log(q)
    return np.logaddexp(t1, t2) / (alphas - 1) - alphas * mu * mu / 2


def max_gaussian_curve_excess(p: float, q: float, mu: float) -> tuple[float, float]:
    """(largest excess, its order) of D_α(Bern(p)‖Bern(q)) over the continuous curve αμ²/2, for real α > 1.

    A scan from 1 + 10⁻⁸ to 10⁶ (step 10⁻⁴ on [2, 30]), refined by golden-section search around the
    largest value. Near 1 the excess tends to KL - μ²/2, and as α → ∞ to -∞ when q > 0, so the scan's
    ends cover the limits.
    """
    alphas = np.concatenate(
        [1 + np.logspace(-8, 0, 2000), np.linspace(2.0, 30.0, 280_001), np.logspace(math.log10(30.0), 6.0, 5000)]
    )
    excess = gaussian_curve_excess(p, q, mu, alphas)
    i = int(np.argmax(excess))
    lo, hi = float(alphas[max(i - 1, 0)]), float(alphas[min(i + 1, len(alphas) - 1)])

    def at(a: float) -> float:
        return float(gaussian_curve_excess(p, q, mu, np.array([a]))[0])

    a_best, value = _golden_max(at, lo, hi)
    if float(excess[i]) >= value:
        return float(excess[i]), float(alphas[i])
    return value, a_best


def continuous_worst_pair(mu: float, eps: float, margin: float = 1e-5) -> tuple[float, float]:
    """A two-outcome pair (P(accept), Q(accept)) inside the Gaussian's curve at every real order, with nearly the largest δ(ε).

    Maximizes δ(ε) subject to the curve at dense_orders(), then raises Q(accept) by the relative
    `margin`, which pushes the pair strictly inside the curve between those orders as well.
    """
    orders = dense_orders()
    _, u, b = rdp_to_delta_tradeoff(orders, gaussian_rdp_curve(mu, orders), eps)
    return u, b * (1 + margin)


def continuous_worst_pair_at_delta(mu: float, delta: float, margin: float = 1e-5) -> tuple[float, float]:
    """The same at a fixed δ: a pair inside the curve at every order whose ε at δ is nearly the largest."""
    orders = dense_orders()
    u, b = rdp_worst_pair_at_delta(orders, gaussian_rdp_curve(mu, orders), delta)
    return u, b * (1 + margin)


def pair_eps_at_delta(u: float, b: float, delta: float) -> float:
    """ε at δ of the pair P = Bern(u), Q = Bern(b) with u > e^ε b: ln((u - δ)/b)."""
    return math.log((u - delta) / b)


# --- The chapter's numbers ------------------------------------------------------------------


def print_table(rng: np.random.Generator) -> None:
    del rng  # nothing here is random; the seed is accepted for a uniform interface
    delta = DELTA

    print("[Toolkit: the Gaussian pair, mu = 1]")
    mu = 1.0
    for eps in (1.0, 3.0):
        print(
            f"  eps={eps:g}: delta exact={gaussian_delta_exact(eps, mu):.4e}  "
            f"numeric E_P[(1-e^(eps-L))+]={pld_hockey_stick_numeric(eps, mu):.4e}  "
            f"tail P[L>eps]={gaussian_tail(eps, mu):.4e}"
        )

    print("[Part 1: where Mironov loses, delta = 1e-5]")
    for mu in (1.0, 10.0):
        curve = gaussian_rdp_curve(mu)
        e_exact = gaussian_eps_exact(mu, delta)
        e_tail = gaussian_eps_tail(mu, delta)
        e_mir, a_mir = rdp_to_dp_mironov(ORDERS, curve, delta)
        rho = mu * mu / 2
        print(
            f"  mu={mu:g}: eps exact={e_exact:.3f}  tail step={e_tail:.3f}  Mironov={e_mir:.3f} at alpha={a_mir:g}  "
            f"(continuous: {mironov_eps_gaussian(rho, delta):.3f} at alpha={mironov_alpha_gaussian(rho, delta):.3f})"
        )
        print(
            f"    loss from dropping (1-e^(eps-L)): {e_tail - e_exact:.3f}   loss from Markov: {e_mir - e_tail:.3f}  "
            f"total excess: {e_mir - e_exact:.3f}"
        )
    mu = 1.0
    e_mir, a_mir = rdp_to_dp_mironov(ORDERS, gaussian_rdp_curve(mu), delta)
    d_exact, d_tail = gaussian_delta_exact(e_mir, mu), gaussian_tail(e_mir, mu)
    d_mir = mironov_delta(a_mir, gaussian_rdp(a_mir, mu), e_mir)
    print(
        f"  mu=1 at eps={e_mir:.3f}: delta exact={d_exact:.3e}  tail={d_tail:.3e}  Mironov bound={d_mir:.3e}  "
        f"factors tail/exact={d_tail / d_exact:.1f}  bound/tail={d_mir / d_tail:.1f}"
    )

    print("[Part 2: the improved conversion]")
    for alpha in (2.0, 10.0, 100.0):
        print(
            f"  alpha={alpha:g}: delta ratio improved/Mironov={delta_ratio(alpha):.4f}  "
            f"1/(e alpha)={1 / (math.e * alpha):.4f}  eps saving={improvement(alpha):.4f}"
        )
    alpha, eps = 6.0, 4.0
    print(
        f"  best constant alpha={alpha:g} eps={eps:g}: formula={best_constant(alpha, eps):.6e}  "
        f"grid={best_constant_numeric(alpha, eps):.6e}  at z*={eps - math.log(1 - 1 / alpha):.4f}"
    )
    for mu in (1.0, 10.0):
        curve = gaussian_rdp_curve(mu)
        e_imp, a_imp = rdp_to_dp_improved(ORDERS, curve, delta)
        e_mir, a_mir = rdp_to_dp_mironov(ORDERS, curve, delta)
        e_exact = gaussian_eps_exact(mu, delta)
        print(
            f"  mu={mu:g}: improved={e_imp:.3f} at alpha={a_imp:g}  Mironov={e_mir:.3f} at alpha={a_mir:g}  "
            f"exact={e_exact:.3f}  improved removes {e_mir - e_imp:.3f} of the excess {e_mir - e_exact:.3f}"
        )
    mu = 1.0
    curve = gaussian_rdp_curve(mu)
    e_imp, a_imp = rdp_to_dp_improved(ORDERS, curve, delta)
    print(
        f"  check: delta exact at eps_improved={gaussian_delta_exact(e_imp, mu):.3e}  "
        f"delta improved at eps_improved={rdp_to_delta_improved(ORDERS, curve, e_imp):.3e}"
    )
    for alpha in (2.0, 8.0, 32.0):
        tau = gaussian_rdp(alpha, mu)
        print(
            f"  single order alpha={alpha:g}, eps(alpha)={tau:g}: Mironov={mironov_eps(alpha, tau, delta):.3f}  "
            f"improved={improved_eps(alpha, tau, delta):.3f}  optimal for this order={single_order_eps_optimal(alpha, tau, delta):.3f}"
        )

    print("[Part 3: RDP as a constraint on tests, mu = 1]")
    for a in (0.1, 1e-3, 1e-6):
        lower = rdp_tradeoff_lower_bound(ORDERS, curve, a)
        print(
            f"  a={a:g}: true power 1-G_mu={1 - gaussian_tradeoff(a, mu):.4e}  "
            f"largest power allowed by RDP={1 - lower:.4e}  ratio={(1 - lower) / (1 - gaussian_tradeoff(a, mu)):.2f}"
        )
    for a in (0.1, 1e-3, 1e-6):
        for alpha in (2.0, 8.0):
            lower = rdp_tradeoff_lower_bound([alpha], [gaussian_rdp(alpha, mu)], a)
            print(f"  only order {alpha:g}: largest power at a={a:g}: {1 - lower:.4e}")
    single = [(1 - rdp_tradeoff_lower_bound([o], [gaussian_rdp(o, mu)], 1e-3), o) for o in ORDERS]
    power, order = min(single)
    print(f"  binding single order at a=1e-3: {order:g}, alone allows {power:.4e}")
    e_imp, _ = rdp_to_dp_improved(ORDERS, curve, delta)
    print(
        f"  (eps, delta) line from the improved conversion at a=1e-3: e^eps a + delta={math.exp(e_imp) * 1e-3 + delta:.4f}"
    )

    print("[Part 4: choosing the order]")
    for sigma, k in ((1.0, 1), (1.0, 100), (1.0, 10_000), (10.0, 1), (10.0, 100)):
        mu = math.sqrt(k) / sigma
        rho = mu * mu / 2
        curve = gaussian_rdp_curve(mu)
        e_imp, a_imp = rdp_to_dp_improved(ORDERS, curve, delta)
        a_cont = improved_alpha_gaussian(rho, delta)
        print(
            f"  sigma={sigma:g} k={k}: rho={rho:g}  improved alpha*={a_cont:.3f} (grid {a_imp:g})  "
            f"eps continuous={improved_eps(a_cont, gaussian_rdp(a_cont, mu), delta):.3f}  grid={e_imp:.3f}  "
            f"Mironov alpha*={mironov_alpha_gaussian(rho, delta):.3f}  "
            f"grid cost={e_imp - improved_eps(a_cont, gaussian_rdp(a_cont, mu), delta):.3f}"
        )
    mu = 1.0
    for d in (1e-5, 1e-10):
        a_cont = improved_alpha_gaussian(0.5, d)
        print(f"  mu=1 delta={d:g}: improved alpha*={a_cont:.3f}")

    print("[Part 5: exact vs optimal-from-RDP vs improved vs Mironov, sigma = 10, delta = 1e-5]")
    for k in (1, 10, 100, 1000, 10_000):
        mu = math.sqrt(k) / 10
        curve = gaussian_rdp_curve(mu)
        e_exact = gaussian_eps_exact(mu, delta)
        e_opt = rdp_to_dp_tradeoff(ORDERS, curve, delta)
        e_imp, _ = rdp_to_dp_improved(ORDERS, curve, delta)
        e_mir, _ = rdp_to_dp_mironov(ORDERS, curve, delta)
        print(
            f"  k={k}: mu={mu:.3f}  exact={e_exact:.3f}  optimal={e_opt:.3f}  improved={e_imp:.3f}  "
            f"Mironov={e_mir:.3f}  ratios {e_opt / e_exact:.2f} {e_imp / e_exact:.2f} {e_mir / e_exact:.2f}"
        )
    mu = 1.0
    curve = gaussian_rdp_curve(mu)
    e_exact = gaussian_eps_exact(mu, delta)
    d_opt, u_opt, b_opt = rdp_to_delta_tradeoff(ORDERS, curve, e_exact)
    worst = [gaussian_rdp(a, mu) for a in ORDERS]
    rr_fwd = max(binary_renyi(a, u_opt, b_opt) - w for a, w in zip(ORDERS, worst, strict=True))
    rr_rev = max(binary_renyi(a, b_opt, u_opt) - w for a, w in zip(ORDERS, worst, strict=True))
    print(
        f"  worst two-outcome pair at eps={e_exact:.3f}: P(accept)={u_opt:.4e}  Q(accept)={b_opt:.4e}  "
        f"delta={d_opt:.4e} vs Gaussian {gaussian_delta_exact(e_exact, mu):.4e}"
    )
    print(
        f"    privacy loss: accept L={math.log(u_opt / b_opt):.4f}  reject L={math.log((1 - u_opt) / (1 - b_opt)):.3e}  "
        f"u - e^eps b={u_opt - math.exp(e_exact) * b_opt:.4e}"
    )
    print(f"    max over orders of D_alpha - eps(alpha): forward={rr_fwd:.1e}  reverse={rr_rev:.1e}")
    slack = sorted((w - binary_renyi(a, u_opt, b_opt), a) for a, w in zip(ORDERS, worst, strict=True))
    print(f"    binding order={slack[0][1]:g}  next smallest slack={slack[1][0]:.4f} at order {slack[1][1]:g}")
    print(f"    improved conversion's delta at that eps={rdp_to_delta_improved(ORDERS, curve, e_exact):.4e}")
    print(
        f"  mu=1 to six decimals: optimal from the curve={rdp_to_dp_tradeoff(ORDERS, curve, delta):.6f}  "
        f"improved={rdp_to_dp_improved(ORDERS, curve, delta)[0]:.6f}"
    )
    u_c, b_c = continuous_worst_pair(mu, e_exact)
    print(
        f"  pair inside the curve at every real order, eps={e_exact:.5f}: P(accept)={u_c:.4e}  Q(accept)={b_c:.4e}  "
        f"delta={u_c - math.exp(e_exact) * b_c:.4e}"
    )
    print(f"    privacy loss: accept L={math.log(u_c / b_c):.4f}  reject L={math.log((1 - u_c) / (1 - b_c)):.3e}")
    for name, (p, q) in (
        ("forward", (u_c, b_c)),
        ("reverse", (b_c, u_c)),
        ("forward, printed digits", (float(f"{u_c:.4e}"), float(f"{b_c:.4e}"))),
    ):
        exc, at = max_gaussian_curve_excess(p, q, mu)
        print(f"    max over real alpha>1 of D_alpha - alpha/2, {name}: {exc:.2e} at alpha={at:.4f}")
    p_r, q_r = float(f"{u_c:.4e}"), float(f"{b_c:.4e}")
    print(f"    printed digits: delta at eps={e_exact:.5f}: {p_r - math.exp(e_exact) * q_r:.4e}")
    u_d, b_d = continuous_worst_pair_at_delta(mu, delta)
    a_cont = improved_alpha_gaussian(0.5, delta)
    print(
        f"  continuous curve at delta=1e-5: worst valid pair forces eps={pair_eps_at_delta(u_d, b_d, delta):.6f}  "
        f"(max excess fwd {max_gaussian_curve_excess(u_d, b_d, mu)[0]:.1e}, rev {max_gaussian_curve_excess(b_d, u_d, mu)[0]:.1e});  "
        f"improved at the best real order {a_cont:.4f}: {improved_eps(a_cont, gaussian_rdp(a_cont, mu), delta):.6f}"
    )
    print(f"  release-with-prob-delta mechanism: D_2={release_mechanism_rdp(delta, 2.0)}")
    for alpha in (2.0, 4.0, 10.0):
        print(
            f"  RR eps=0.5, alpha={alpha:g}: D_alpha={randomized_response_rdp(alpha, 0.5):.4f}  "
            f"bound min(eps, alpha eps^2/2)={pure_dp_rdp_bound(alpha, 0.5):.4f}"
        )

    print("[Part 2, over a grid: saving at least Delta(alpha_Mironov)]")
    mu = 1.0
    curve = gaussian_rdp_curve(mu)
    e_imp, _ = rdp_to_dp_improved(ORDERS, curve, delta)
    e_mir, a_mir = rdp_to_dp_mironov(ORDERS, curve, delta)
    print(f"  mu=1: saving={e_mir - e_imp:.3f}  Delta({a_mir:g})={improvement(a_mir):.3f}")


def binary_renyi(alpha: float, p: float, q: float) -> float:
    """D_α(Bern(p)‖Bern(q)), in logs so that tiny p or q keep precision."""
    t1 = alpha * math.log1p(-p) + (1 - alpha) * math.log1p(-q)
    t2 = alpha * math.log(p) + (1 - alpha) * math.log(q)
    return float(np.logaddexp(t1, t2)) / (alpha - 1)


# --- Figures --------------------------------------------------------------------------------


def plot_majorants(mu: float = 1.0, eps: float = 3.0, alpha: float = 6.0) -> Figure:
    """The hockey-stick integrand, the two exponential majorants, and the density of L."""
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    zs = [eps - 3 + 9 * i / 900 for i in range(901)]
    ax.plot(
        zs, [hockey_integrand(z, eps) for z in zs], "k", lw=2, label=r"$(1 - e^{\varepsilon - z})_+$ (exact $\delta$)"
    )
    ax.plot(zs, [1.0 if z > eps else 0.0 for z in zs], color="0.5", ls="--", label=r"$\mathbf{1}[z > \varepsilon]$")
    ax.plot(
        zs,
        [mironov_majorant(z, alpha, eps) for z in zs],
        color="C3",
        label=r"Mironov: $e^{(\alpha - 1)(z - \varepsilon)}$",
    )
    ax.plot(zs, [improved_majorant(z, alpha, eps) for z in zs], color="C0", label=r"improved: $c^* e^{(\alpha - 1) z}$")
    z_star = eps - math.log(1 - 1 / alpha)
    ax.plot([z_star], [hockey_integrand(z_star, eps)], "o", color="C0")
    density = [math.exp(-((z - mu * mu / 2) ** 2) / (2 * mu * mu)) / (mu * math.sqrt(2 * math.pi)) for z in zs]
    ax.fill_between(zs, [2 * d for d in density], color="C2", alpha=0.25, label=r"density of $L$ (scaled), $\mu = 1$")
    ax.axvline(eps, color="0.7", lw=0.8)
    ax.set(
        xlabel=r"privacy loss $z$",
        ylim=(0, 1.9),
        xlim=(zs[0], zs[-1]),
        title=rf"Bounding the hockey-stick integrand by $c\,e^{{(\alpha-1)z}}$ ($\varepsilon = {eps:g}$, $\alpha = {alpha:g}$)",
    )
    ax.legend(fontsize=9, loc="upper right")
    return fig


def plot_tradeoff_region(mu: float = 1.0) -> Figure:
    """Largest power allowed by the RDP curve, by single orders, and the true Gaussian power."""
    curve = gaussian_rdp_curve(mu)
    a_values = [10 ** (-10 + 10 * i / 120) for i in range(121)]
    a_values[-1] = 0.999
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    a_lin = [i / 200 for i in range(1, 200)]
    for ax, xs, log in ((axes[0], a_lin, False), (axes[1], a_values, True)):
        ax.plot(xs, [1 - gaussian_tradeoff(a, mu) for a in xs], "k", lw=2, label=r"true Gaussian, $1 - G_1(a)$")
        ax.plot(
            xs,
            [1 - rdp_tradeoff_lower_bound(ORDERS, curve, a) for a in xs],
            color="C3",
            label="largest power allowed by the RDP curve",
        )
        for alpha, style in ((2.0, ":"), (8.0, "--")):
            ax.plot(
                xs,
                [1 - rdp_tradeoff_lower_bound([alpha], [gaussian_rdp(alpha, mu)], a) for a in xs],
                color="C0",
                ls=style,
                lw=1,
                label=rf"order $\alpha = {alpha:g}$ alone",
            )
        if log:
            e_imp, _ = rdp_to_dp_improved(ORDERS, curve, DELTA)
            ax.plot(
                xs,
                [min(1.0, math.exp(e_imp) * a + DELTA) for a in xs],
                color="0.5",
                lw=1,
                label=rf"$(\varepsilon, \delta) = ({e_imp:.2f}, 10^{{-5}})$" + "\nfrom the improved conversion",
            )
            ax.set(xscale="log", yscale="log", ylim=(1e-6, 1.2), xlim=(1e-10, 1), title="Log axes: the low-$a$ corner")
        else:
            ax.plot([0, 1], [0, 1], color="0.8", lw=0.8)
            ax.set(xlim=(0, 1), ylim=(0, 1.02), title="Linear axes")
        ax.set(xlabel=r"type I error $a$", ylabel=r"power $1 - b$")
    axes[0].legend(fontsize=8, loc="lower right")
    # The other curves are labelled on the left; here only the (ε, δ) line, in the empty lower-right corner.
    grey = axes[1].get_lines()[-1]
    axes[1].legend(handles=[grey], fontsize=8, loc="lower right")
    return fig


def plot_eps_vs_order(mu: float = 1.0) -> Figure:
    curve = gaussian_rdp_curve(mu)
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    for method, color, text_at in (("mironov", "C3", (40.0, 7.5)), ("improved", "C0", (40.0, 2.5))):
        values = eps_curve(ORDERS, curve, DELTA, method)
        ax.plot(ORDERS, values, color=color, label=method.capitalize() if method == "mironov" else "improved")
        i = min(range(len(values)), key=lambda j: values[j])
        ax.plot([ORDERS[i]], [values[i]], "o", color=color)
        ax.annotate(
            rf"best order $\alpha = {ORDERS[i]:g}$, $\varepsilon = {values[i]:.2f}$",
            (ORDERS[i], values[i]),
            xytext=text_at,
            color=color,
            fontsize=9,
            arrowprops={"arrowstyle": "->", "color": color},
        )
    ax.axhline(gaussian_eps_exact(mu, DELTA), color="k", ls="--", lw=1, label=r"exact $\varepsilon$")
    ax.plot(ORDERS, curve, color="0.6", lw=1, label=r"$\varepsilon(\alpha) = \alpha\mu^2/2$")
    ax.set(
        xscale="log",
        xlabel=r"order $\alpha$",
        ylabel=r"converted $\varepsilon$ at $\delta = 10^{-5}$",
        ylim=(0, 15),
        title=r"Converted $\varepsilon$ for each order, Gaussian $\mu = 1$",
    )
    ax.legend(fontsize=9, loc="upper right")
    return fig


def plot_slack(sigma: float = 10.0) -> Figure:
    ks = [round(10 ** (i / 2)) for i in range(9)]
    rows: list[tuple[float, float, float, float]] = []
    for k in ks:
        mu = math.sqrt(k) / sigma
        curve = gaussian_rdp_curve(mu)
        rows.append(
            (
                gaussian_eps_exact(mu, DELTA),
                rdp_to_dp_tradeoff(ORDERS, curve, DELTA),
                rdp_to_dp_improved(ORDERS, curve, DELTA)[0],
                rdp_to_dp_mironov(ORDERS, curve, DELTA)[0],
            )
        )
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    labels = ("exact", "optimal from the RDP curve", "improved", "Mironov")
    colors = ("k", "C2", "C0", "C3")
    for j, (label, color) in enumerate(zip(labels, colors, strict=True)):
        width = 5.0 if j == 1 else 1.5  # the optimal curve is drawn wide: the improved one lies on top of it
        axes[0].plot(ks, [r[j] for r in rows], "o-", ms=3, lw=width, color=color, label=label)
        axes[1].plot(ks, [r[j] / r[0] for r in rows], "o-", ms=3, lw=width, color=color, label=label)
    axes[0].set(
        xscale="log",
        yscale="log",
        xlabel=r"steps $k$",
        ylabel=r"$\varepsilon$ at $\delta = 10^{-5}$",
        title=rf"Composed Gaussian, $\sigma = {sigma:g}$",
    )
    axes[1].set(
        xscale="log",
        xlabel=r"steps $k$",
        ylabel=r"$\varepsilon$ / exact $\varepsilon$",
        ylim=(0.95, 1.5),
        title="Ratio to the exact value",
    )
    axes[0].legend(fontsize=9)
    return fig


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=FIGURES, help="directory for the figures")
    args = parser.parse_args()
    seed: int = args.seed
    out: Path = args.out

    print_table(np.random.default_rng(seed))

    _ = np.random.default_rng(seed + 1)  # separate stream for figures; these figures draw no samples
    out.mkdir(parents=True, exist_ok=True)
    figures = {
        "rdp_conversions_majorants.svg": plot_majorants(),
        "rdp_conversions_tradeoff.svg": plot_tradeoff_region(),
        "rdp_conversions_orders.svg": plot_eps_vs_order(),
        "rdp_conversions_slack.svg": plot_slack(),
    }
    for name, fig in figures.items():
        fig.tight_layout()
        fig.savefig(out / name, metadata={"Date": None})  # no timestamp, so output is reproducible
        plt.close(fig)
        print(f"saved {out / name}")


if __name__ == "__main__":
    main()
