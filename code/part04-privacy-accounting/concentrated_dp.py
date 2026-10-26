"""Check the concentrated differential privacy chapter and draw its figures.

    uv run python code/part04-privacy-accounting/concentrated_dp.py
    uv run python code/part04-privacy-accounting/concentrated_dp.py --seed 1 --out /tmp/figures

Prints the numbers quoted in 03-concentrated-dp.md (they match exactly with the default seed),
then saves the chapter's figures.

Conventions: logs are natural. A mechanism is rho-zCDP when its RDP curve satisfies
eps(alpha) <= rho * alpha for every alpha > 1. The Gaussian mechanism has noise multiplier sigma
(noise std per unit of L2 sensitivity). In the subsampling section q is the Poisson sampling rate,
never a density. Every optimization over alpha uses the part's ORDERS grid unless a function
says it uses the fine continuous grid FINE_ALPHAS.
"""

import argparse
import math
import sys
from collections.abc import Sequence
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # for code/shared
from shared.figures import configure
from shared.stats import logsumexp, normal_cdf

configure("concentrated_dp")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from numpy.typing import NDArray

# Figures are build output: _build/figures/<part>/, which the build copies into each book.
FIGURES = Path(__file__).resolve().parents[2] / "_build" / "figures" / Path(__file__).parent.name

# The part's grid of Rényi orders, shared by every chapter so the numbers agree.
ORDERS: list[float] = [1 + x / 10 for x in range(1, 100)] + [float(a) for a in range(12, 64)] + [128.0, 256.0, 512.0]
# A fine continuous grid of orders, 1 + 10^-4 ... 1 + 10^4, log-spaced.
FINE_ALPHAS: list[float] = [1 + 10 ** (-4 + 8 * i / 20000) for i in range(20001)]

# The part's running example: DP-SGD with Poisson rate q, noise multiplier sigma, k steps.
Q_RUN = 0.01
SIGMA_RUN = 1.0
K_RUN = 10_000
DELTA = 1e-5


# --- Section 29.3: the definition, the Gaussian mechanism, composition ----------------------------


def gaussian_zcdp_rho(sigma: float, sensitivity: float = 1.0) -> float:
    """ρ = Δ²/(2σ²) for a Gaussian mechanism with noise std σ and L2 sensitivity Δ.

    With the part's convention (noise N(0, σ²Δ²), σ the noise multiplier) call it with sensitivity=1.
    """
    return sensitivity**2 / (2 * sigma**2)


def gaussian_rdp(alpha: float, sigma: float, sensitivity: float = 1.0) -> float:
    """ε(α) = αΔ²/(2σ²): the Gaussian mechanism's RDP curve is a line through the origin."""
    return alpha * gaussian_zcdp_rho(sigma, sensitivity)


def compose_zcdp(rhos: Sequence[float]) -> float:
    """Composition of zCDP mechanisms: the ρ's add."""
    return float(sum(rhos))


def gaussian_privacy_loss_samples(rng: np.random.Generator, sigma: float, n: int) -> NDArray[np.float64]:
    """Draws of the privacy loss L(Y) = ln p(Y)/q(Y) for P = N(0, σ²), Q = N(1, σ²), Y ~ P.

    L(y) = ((y - 1)² - y²)/(2σ²) = (1 - 2y)/(2σ²), a normal variable with mean ρ and variance 2ρ.
    """
    y = rng.normal(0.0, sigma, size=n)
    return (1 - 2 * y) / (2 * sigma**2)


def zcdp_log_mgf_bound(alpha: float, rho: float) -> float:
    """ln E[e^{(α-1)L}] ≤ (α-1)αρ, the privacy-loss form of ρ-zCDP."""
    return (alpha - 1) * alpha * rho


def zcdp_tail_bound(lam: float, rho: float) -> float:
    """P[L > ρ + λ] ≤ exp(-λ²/(4ρ)), Chernoff applied to the zCDP moment bound."""
    return math.exp(-(lam**2) / (4 * rho))


def gaussian_loss_tail(lam: float, rho: float) -> float:
    """Exact P[L > ρ + λ] for the Gaussian mechanism, where L ~ N(ρ, 2ρ)."""
    return normal_cdf(-lam / math.sqrt(2 * rho))


# --- Section 29.4: pure DP implies zCDP ------------------------------------------------------------


def pure_to_zcdp(eps: float) -> float:
    """ε-DP implies (ε²/2)-zCDP (Bun & Steinke 2016, Prop. 1.4)."""
    return eps**2 / 2


def pure_to_zcdp_sharp(eps: float) -> float:
    """ε-DP implies (ε tanh(ε/2))-zCDP: every ε-DP pair is a post-processing of ε-randomized response,
    whose curve satisfies D_α ≤ α ε tanh(ε/2). Randomized response attains it as α → 1."""
    return eps * math.tanh(eps / 2)


def binary_renyi(p: Sequence[float], q: Sequence[float], alpha: float) -> float:
    """D_α(P‖Q) for distributions on a finite set, computed in log space (stable for large α)."""
    logs = [alpha * math.log(pi) + (1 - alpha) * math.log(qi) for pi, qi in zip(p, q, strict=True) if pi > 0]
    return logsumexp(logs) / (alpha - 1)


def rr_keep_prob(eps: float) -> float:
    """Randomized response reports the true bit with probability e^ε/(1 + e^ε)."""
    return 1 / (1 + math.exp(-eps))


def rr_rdp(alpha: float, eps: float) -> float:
    """Exact RDP curve of ε-randomized response: D_α(Bern(p)‖Bern(1-p)) with p = e^ε/(1+e^ε)."""
    p = rr_keep_prob(eps)
    return binary_renyi([p, 1 - p], [1 - p, p], alpha)


def rr_kl(eps: float) -> float:
    """KL of ε-randomized response, (2p - 1)ε = ε tanh(ε/2)."""
    return eps * math.tanh(eps / 2)


def hoeffding_rdp_bound(alpha: float, eps: float, mean: float) -> float:
    """The Hoeffding route: D_α ≤ mean + (α-1)ε²/2 for a privacy loss in [-ε, ε] with mean `mean`."""
    return mean + (alpha - 1) * eps**2 / 2


def smallest_zcdp_rho(orders: Sequence[float], rdp: Sequence[float]) -> float:
    """The smallest ρ with ε(α) ≤ ρα at every listed order: max over the orders of ε(α)/α."""
    return max(e / a for a, e in zip(orders, rdp, strict=True))


def br_pair(eps: float, t: float) -> tuple[list[float], list[float]]:
    """A binary ε-bounded-range pair: output probabilities with log ratios t and t - ε (Cesar & Rogers, Def. 2.4)."""
    p0 = (1 - math.exp(t - eps)) / (1 - math.exp(-eps))
    q0 = (math.exp(-t) - math.exp(-eps)) / (1 - math.exp(-eps))
    return [1 - q0, q0], [1 - p0, p0]


def br_worst_ratio(eps: float, alphas: Sequence[float], ts: int = 99) -> float:
    """max over t ∈ (0, ε) and the given α of D_α / (αε²/8), both directions, for the binary ε-BR pairs."""
    worst = 0.0
    for i in range(1, ts + 1):
        p, q = br_pair(eps, eps * i / (ts + 1))
        for a in alphas:
            worst = max(worst, binary_renyi(p, q, a) / (a * eps**2 / 8), binary_renyi(q, p, a) / (a * eps**2 / 8))
    return worst


# --- Section 29.5: from zCDP to (ε, δ) ---------------------------------------------------------------


def zcdp_to_dp_simple(rho: float, delta: float) -> float:
    """ε = ρ + 2√(ρ ln(1/δ)): Mironov's conversion at the best α = 1 + √(ln(1/δ)/ρ)."""
    return rho + 2 * math.sqrt(rho * math.log(1 / delta))


def best_alpha_simple(rho: float, delta: float) -> float:
    return 1 + math.sqrt(math.log(1 / delta) / rho)


def improved_conversion(alpha: float, eps_alpha: float, delta: float) -> float:
    """ε = ε(α) + ln(1/δ)/(α-1) + ln(1 - 1/α) - ln(α)/(α-1) (Canonne, Kamath & Steinke 2020, Prop. 12)."""
    return eps_alpha + math.log(1 / delta) / (alpha - 1) + math.log1p(-1 / alpha) - math.log(alpha) / (alpha - 1)


def rdp_to_dp_improved(orders: Sequence[float], rdp: Sequence[float], delta: float) -> float:
    """Best improved conversion over the listed orders, clipped at 0 (vectorized improved_conversion)."""
    a = np.asarray(orders, dtype=np.float64)
    e = np.asarray(rdp, dtype=np.float64)
    values = e + math.log(1 / delta) / (a - 1) + np.log1p(-1 / a) - np.log(a) / (a - 1)
    return max(0.0, float(np.min(values)))


def rdp_to_dp_mironov(orders: Sequence[float], rdp: Sequence[float], delta: float) -> float:
    """Best of ε(α) + ln(1/δ)/(α-1) over the listed orders."""
    return min(e + math.log(1 / delta) / (a - 1) for a, e in zip(orders, rdp, strict=True))


def zcdp_to_dp_improved(rho: float, delta: float) -> float:
    """Improved conversion of ρ-zCDP, optimized over the fine continuous grid FINE_ALPHAS (not ORDERS)."""
    return rdp_to_dp_improved(FINE_ALPHAS, [rho * a for a in FINE_ALPHAS], delta)


def zcdp_improved_best_alpha(rho: float, delta: float) -> float:
    """The order of FINE_ALPHAS where the improved conversion of ρ-zCDP is smallest."""
    return best_order_improved(FINE_ALPHAS, [rho * a for a in FINE_ALPHAS], delta)


def best_order_improved(orders: Sequence[float], rdp: Sequence[float], delta: float) -> float:
    """The listed order at which the improved conversion is smallest."""
    values = [improved_conversion(a, e, delta) for a, e in zip(orders, rdp, strict=True)]
    return float(orders[int(np.argmin(values))])


def best_order_mironov(orders: Sequence[float], rdp: Sequence[float], delta: float) -> float:
    """The listed order at which ε(α) + ln(1/δ)/(α-1) is smallest."""
    values = [e + math.log(1 / delta) / (a - 1) for a, e in zip(orders, rdp, strict=True)]
    return float(orders[int(np.argmin(values))])


def zcdp_to_dp_improved_orders(rho: float, delta: float, orders: Sequence[float] = ORDERS) -> float:
    """The same conversion restricted to the part's ORDERS grid."""
    return rdp_to_dp_improved(orders, [rho * a for a in orders], delta)


def rho_for_target(eps: float, delta: float) -> float:
    """Largest ρ whose simple conversion gives ε: ρ = (√(ε + ln(1/δ)) - √ln(1/δ))²."""
    log_inv = math.log(1 / delta)
    return (math.sqrt(eps + log_inv) - math.sqrt(log_inv)) ** 2


def gaussian_delta(eps: float, mu: float) -> float:
    """Exact δ(ε) of a Gaussian pair N(0,1) vs N(μ,1): Φ(μ/2 - ε/μ) - e^ε Φ(-μ/2 - ε/μ)."""
    return normal_cdf(mu / 2 - eps / mu) - math.exp(eps) * normal_cdf(-mu / 2 - eps / mu)


def gaussian_eps(mu: float, delta: float) -> float:
    """Exact ε at δ for the Gaussian pair, by bisection (the simple zCDP conversion is a valid upper end)."""
    lo, hi = 0.0, zcdp_to_dp_simple(mu**2 / 2, delta)
    if gaussian_delta(0.0, mu) <= delta:
        return 0.0
    for _ in range(200):
        mid = (lo + hi) / 2
        if gaussian_delta(mid, mu) > delta:
            lo = mid
        else:
            hi = mid
    return hi


def zcdp_gaussian_mu(rho: float) -> float:
    """A ρ-zCDP Gaussian mechanism is μ-GDP with μ = √(2ρ) (shift Δ/σ)."""
    return math.sqrt(2 * rho)


# --- Section 29.6: group privacy ---------------------------------------------------------------------


def group_zcdp_rho(rho: float, k: int) -> float:
    """ρ-zCDP gives (k²ρ)-zCDP for groups of size k (Bun & Steinke 2016, Prop. 1.9)."""
    return k * k * rho


def renyi_triangle_rhs(alpha: float, k: float, d_pr: float, d_rq: float) -> float:
    """Right side of D_α(P‖Q) ≤ kα/(kα-1) · D_{(kα-1)/(k-1)}(P‖R) + D_{kα}(R‖Q), given the two divergences."""
    return k * alpha / (k * alpha - 1) * d_pr + d_rq


def rdp_group_route(alpha: float, eps_alpha: float, c: int) -> tuple[float, float]:
    """Mironov's group privacy for RDP: (α, ε)-RDP gives (α/2^c, 3^c ε)-RDP for groups of 2^c (needs α ≥ 2^{c+1})."""
    return alpha / 2**c, 3**c * eps_alpha


# --- Section 29.7: approximate zCDP ----------------------------------------------------------------


def approx_zcdp_to_dp(rho: float, delta_approx: float, delta: float) -> float:
    """δ_a-approximate ρ-zCDP is (ε, δ_a + δ')-DP with ε from the improved conversion at δ' = δ - δ_a."""
    return zcdp_to_dp_improved(rho, delta - delta_approx)


def compose_failure(deltas: Sequence[float]) -> float:
    """Composed failure probability 1 - Π(1 - δ_i) (Bun & Steinke 2016, Lemma 8.2)."""
    survive = 1.0
    for d in deltas:
        survive *= 1 - d
    return 1 - survive


# --- Section 29.8: subsampling and truncated CDP -----------------------------------------------------


def subsampled_gaussian_rdp_int(alpha: int, q: float, sigma: float) -> float:
    """RDP at integer α of the Poisson-subsampled Gaussian (Mironov, Talwar & Zhang 2019).

    A_α = Σ_j C(α, j) (1-q)^{α-j} q^j exp((j² - j)/(2σ²)), ε(α) = ln A_α/(α - 1), in log space.
    """
    if alpha < 2:
        raise ValueError("integer alpha >= 2 required")
    terms = [
        math.lgamma(alpha + 1)
        - math.lgamma(j + 1)
        - math.lgamma(alpha - j + 1)
        + (alpha - j) * math.log1p(-q)
        + (j * math.log(q) if j > 0 else 0.0)
        + (j * j - j) / (2 * sigma**2)
        for j in range(alpha + 1)
    ]
    return logsumexp(terms) / (alpha - 1)


def integer_orders(orders: Sequence[float]) -> list[int]:
    return [int(a) for a in orders if float(a).is_integer() and a >= 2]


def subsampled_lower_bound(alpha: int, q: float, sigma: float) -> float:
    """ε(α) ≥ α/(2σ²) + α ln q/(α - 1), from keeping only the j = α term of A_α."""
    return alpha / (2 * sigma**2) + alpha * math.log(q) / (alpha - 1)


def tcdp_rho(orders: Sequence[int], rdp: Sequence[float], omega: float) -> float:
    """Smallest ρ with ε(α) ≤ ρα at every listed order α ≤ ω."""
    pairs = [(a, e) for a, e in zip(orders, rdp, strict=True) if a <= omega]
    return smallest_zcdp_rho([float(a) for a, _ in pairs], [e for _, e in pairs])


def tcdp_to_dp(rho: float, omega: float, delta: float) -> float:
    """(ρ, ω)-tCDP to (ε, δ)-DP (Bun, Dwork, Rothblum & Steinke 2018, Lemma 6): α = min(ω, 1 + √(ln(1/δ)/ρ))."""
    log_inv = math.log(1 / delta)
    if log_inv <= (omega - 1) ** 2 * rho:
        return rho + 2 * math.sqrt(rho * log_inv)
    return rho * omega + log_inv / (omega - 1)


def tcdp_subsampling(rho: float, s: float) -> tuple[float, float]:
    """Theorem 11 of Bun et al. 2018: a (ρ, ω')-tCDP mechanism run on a uniformly random s-fraction is
    (13 s² ρ, ln(1/s)/(4ρ))-tCDP, if ρ, s ≤ 0.1, ln(1/s) ≥ 3ρ(2 + log2(1/ρ)) and ω' ≥ ln(1/s)/(2ρ) ≥ 3."""
    if not (0 < rho <= 0.1 and 0 < s <= 0.1):
        raise ValueError("Theorem 11 needs rho, s in (0, 0.1]")
    if math.log(1 / s) < 3 * rho * (2 + math.log2(1 / rho)) or math.log(1 / s) / (2 * rho) < 3:
        raise ValueError("Theorem 11's side conditions fail")
    return 13 * s * s * rho, math.log(1 / s) / (4 * rho)


def running_example_curve(q: float = Q_RUN, sigma: float = SIGMA_RUN) -> tuple[list[int], list[float]]:
    """The subsampled-Gaussian RDP curve of one DP-SGD step on the integer orders of ORDERS."""
    ints = integer_orders(ORDERS)
    return ints, [subsampled_gaussian_rdp_int(a, q, sigma) for a in ints]


# --- Output --------------------------------------------------------------------------------------


def print_table(rng: np.random.Generator) -> None:
    log_inv = math.log(1 / DELTA)

    print("[Part 1: the Gaussian mechanism and its privacy loss]")
    rho1 = gaussian_zcdp_rho(SIGMA_RUN)
    print(f"  sigma=1: rho={rho1:g}  eps(2)={gaussian_rdp(2, 1.0):g}  eps(10)={gaussian_rdp(10, 1.0):g}")
    print(
        f"  sigma=2, sensitivity=1: rho={gaussian_zcdp_rho(2.0):g};  10 steps at sigma=1: rho={compose_zcdp([rho1] * 10):g}"
    )
    losses = gaussian_privacy_loss_samples(rng, SIGMA_RUN, 1_000_000)
    mgf2 = float(np.log(np.mean(np.exp(losses))))
    print(
        f"  privacy loss, Monte Carlo (10^6 draws): mean={float(np.mean(losses)):.4f}  var={float(np.var(losses)):.4f}  "
        f"ln E[e^L]={mgf2:.4f}  (theory: mean=0.5000 var=1.0000 ln E[e^L]={zcdp_log_mgf_bound(2, rho1):.4f})"
    )
    for lam in (1.0, 2.0, 4.0):
        print(
            f"  tail P[L > rho + {lam:g}]: exact={gaussian_loss_tail(lam, rho1):.3e}  "
            f"zCDP bound={zcdp_tail_bound(lam, rho1):.3e}  simulated={float(np.mean(losses > rho1 + lam)):.3e}"
        )
        print(
            f"    exact/bound={gaussian_loss_tail(lam, rho1) / zcdp_tail_bound(lam, rho1):.3f}  "
            f"Mills factor sqrt(rho/pi)/lambda={math.sqrt(rho1 / math.pi) / lam:.3f}"
        )

    print("[Part 2: pure DP implies zCDP; randomized response]")
    for eps in (0.5, 1.0, 2.0, 4.0, 5.0):
        smallest = smallest_zcdp_rho(FINE_ALPHAS[:15001], [rr_rdp(a, eps) for a in FINE_ALPHAS[:15001]])
        print(
            f"  eps={eps:g}: eps^2/2={pure_to_zcdp(eps):.4f}  KL=eps tanh(eps/2)={rr_kl(eps):.4f}  "
            f"smallest rho for RR={smallest:.4f}  sharp rho={pure_to_zcdp_sharp(eps):.4f}"
        )
    print(
        f"  eps=5: the sharp line {pure_to_zcdp_sharp(5.0):.4f} alpha reaches eps=5 at alpha={5 / pure_to_zcdp_sharp(5.0):.4f}"
    )
    eps = 1.0
    print(f"  RR at eps=1, keep prob={rr_keep_prob(eps):.4f}")
    for a in (1.5, 2.0, 5.0, 10.0, 50.0):
        print(
            f"    alpha={a:g}: exact={rr_rdp(a, eps):.4f}  Hoeffding (KL mean)={hoeffding_rdp_bound(a, eps, rr_kl(eps)):.4f}  "
            f"alpha*eps^2/2={a * pure_to_zcdp(eps):.4f}  min(eps, alpha*eps^2/2)={min(eps, a * pure_to_zcdp(eps)):.4f}"
        )
    print(
        f"  ratios to exact: line at alpha=2 {pure_to_zcdp(eps) * 2 / rr_rdp(2.0, eps):.3f}  "
        f"line at alpha=50 {pure_to_zcdp(eps) * 50 / rr_rdp(50.0, eps):.1f}  cap at alpha=50 {eps / rr_rdp(50.0, eps):.4f}"
    )
    print(f"  alpha where alpha*eps^2/2 reaches eps: {2 / eps:g} (eps=1), {2 / 4:g} (eps=4)")
    br_alphas = [1 + 10 ** (-3 + 5 * i / 200) for i in range(201)]
    for eps in (0.5, 1.0, 4.0):
        print(
            f"  bounded range eps={eps:g}: max D_alpha/(alpha eps^2/8) over binary pairs = {br_worst_ratio(eps, br_alphas):.4f}"
        )

    print("[Part 3: zCDP to (eps, delta) at delta=1e-5]")
    print(f"  ln(1/delta)={log_inv:.4f}")
    print(f"  rho=0.5: best alpha (simple)={best_alpha_simple(0.5, DELTA):.3f}")
    for rho in (0.01, 0.1, 0.5, 1.0, 5.0, 50.0):
        mu = zcdp_gaussian_mu(rho)
        print(
            f"  rho={rho:g}: mu={mu:.4f}  simple={zcdp_to_dp_simple(rho, DELTA):.3f}  "
            f"improved fine={zcdp_to_dp_improved(rho, DELTA):.3f}  improved ORDERS={zcdp_to_dp_improved_orders(rho, DELTA):.3f}  "
            f"exact Gaussian={gaussian_eps(mu, DELTA):.3f}"
        )
    print(
        f"  rho=50: best order of the improved conversion on the fine grid={zcdp_improved_best_alpha(50.0, DELTA):.3f}"
    )
    exact_half = gaussian_eps(1.0, DELTA)
    print(
        f"  rho=0.5 ratios to exact: simple={zcdp_to_dp_simple(0.5, DELTA) / exact_half:.3f}  "
        f"improved={zcdp_to_dp_improved(0.5, DELTA) / exact_half:.3f}"
    )
    rho_target = rho_for_target(1.0, DELTA)
    print(f"  rho for eps=1 (simple): {rho_target:.5f}  sigma={1 / math.sqrt(2 * rho_target):.3f}")

    print("[Part 4: group privacy]")
    for k in (1, 2, 4, 8):
        rho_k = group_zcdp_rho(rho1, k)
        print(
            f"  sigma=1, group of {k}: rho={rho_k:g}  eps at 1e-5 (improved)={zcdp_to_dp_improved(rho_k, DELTA):.3f}  "
            f"exact Gaussian with shift {k}={gaussian_eps(float(k), DELTA):.3f}"
        )
    for c in (1, 2, 3):
        a_new, e_new = rdp_group_route(64.0, gaussian_rdp(64.0, 1.0), c)
        print(
            f"  group of {2**c}: zCDP slope {group_zcdp_rho(rho1, 2**c):g}  "
            f"RDP route: order {a_new:g}, eps {e_new:g}, slope {e_new / a_new:g}"
        )
    lhs = gaussian_rdp(3.0, 1.0, 2.0)
    rhs = renyi_triangle_rhs(3.0, 2, gaussian_rdp(5.0, 1.0), gaussian_rdp(6.0, 1.0))
    print(f"  triangle-like inequality, Gaussians 0,1,2, alpha=3, k=2: lhs={lhs:g}  rhs={rhs:g}")
    print(f"  pure eps=1 for group of 8: {8 * 1.0:g}")

    print("[Part 5: approximate zCDP]")
    steps, eps0, delta0 = 10, 0.5, 1e-7
    rho_total = compose_zcdp([pure_to_zcdp(eps0)] * steps + [rho1] * 10)
    delta_a = compose_failure([delta0] * steps)
    print(
        f"  10 x (0.5, 1e-7)-DP + 10 Gaussians sigma=1: rho={rho_total:g}  delta_approx={delta_a:.4e}  "
        f"eps at total delta 1e-5={approx_zcdp_to_dp(rho_total, delta_a, DELTA):.3f}"
    )
    eps_gauss = zcdp_to_dp_improved(compose_zcdp([rho1] * 10), DELTA - steps * delta0)
    print(
        f"  without approximate zCDP: Gaussians alone at delta 9e-6 eps={eps_gauss:.3f}, "
        f"plus basic composition of the stability steps eps={steps * eps0:g}: total {eps_gauss + steps * eps0:.3f}"
    )

    print("[Part 6: subsampled Gaussian, q=0.01, sigma=1]")
    ints, curve = running_example_curve()
    for a in (2, 4, 8, 10, 12, 16, 32, 64, 128, 256, 512):
        e = subsampled_gaussian_rdp_int(a, Q_RUN, SIGMA_RUN)
        print(
            f"  alpha={a}: eps={e:.4e}  eps/alpha={e / a:.4e}  unsubsampled alpha/2={a / 2:g}  "
            f"lower bound={subsampled_lower_bound(a, Q_RUN, SIGMA_RUN):.4f}"
        )
    print(
        f"  lower bound turns positive at alpha = 1 + 2 sigma^2 ln(1/q) = {1 + 2 * SIGMA_RUN**2 * math.log(1 / Q_RUN):.2f}"
    )
    rho_best = smallest_zcdp_rho([float(a) for a in ints], curve)
    print(f"  smallest zCDP rho on integer ORDERS={rho_best:.4f}  (unsubsampled rho={rho1:g})")
    print(f"  q^2 (e^(1/sigma^2) - 1)/2={Q_RUN**2 * (math.e - 1) / 2:.4e}")
    k_rho_grid = K_RUN * rho_best
    print(f"  k=10000 with the integer-ORDERS ratio (not a valid zCDP parameter): rho={k_rho_grid:.1f}")
    k_rho = K_RUN * rho1  # the smallest valid zCDP parameter is the unsubsampled 1/(2 sigma^2)
    print(
        f"  k=10000 via zCDP (rho=1/(2 sigma^2) per step): rho={k_rho:g}  eps simple={zcdp_to_dp_simple(k_rho, DELTA):.1f}  "
        f"eps improved={zcdp_to_dp_improved(k_rho, DELTA):.1f}"
    )
    print(f"  ratio zCDP slope / tCDP slope (omega=8): {rho1 / tcdp_rho(ints, curve, 8.0):.0f}")
    rdp_k = [K_RUN * e for e in curve]
    ints_f = [float(a) for a in ints]
    print(
        f"  k=10000 via RDP on integer ORDERS: Mironov eps={rdp_to_dp_mironov(ints_f, rdp_k, DELTA):.3f}  "
        f"improved eps={rdp_to_dp_improved(ints_f, rdp_k, DELTA):.3f}"
    )
    print(
        f"  best integer order: Mironov {best_order_mironov(ints_f, rdp_k, DELTA):g}  "
        f"improved {best_order_improved(ints_f, rdp_k, DELTA):g}"
    )
    for omega in (2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0, 16.0):
        r = tcdp_rho(ints, curve, omega)
        print(
            f"  tCDP omega={omega:g}: rho per step={r:.4e}  k=10000: rho={K_RUN * r:.4f}  "
            f"eps={tcdp_to_dp(K_RUN * r, omega, DELTA):.3f}"
        )
    r4 = K_RUN * tcdp_rho(ints, curve, 4.0)
    print(
        f"  omega=4: simple formula ignoring omega={zcdp_to_dp_simple(r4, DELTA):.3f}  "
        f"needs alpha={best_alpha_simple(r4, DELTA):.3f}"
    )

    print("[Part 6: tCDP subsampling theorem, sigma=sqrt(5), s=0.01]")
    sigma5 = math.sqrt(5)
    rho5 = gaussian_zcdp_rho(sigma5)
    rho_s, omega_s = tcdp_subsampling(rho5, 0.01)
    curve5 = [subsampled_gaussian_rdp_int(a, 0.01, sigma5) for a in ints]
    print(
        f"  rho={rho5:g}  theorem: rho'={rho_s:.3e}  omega={omega_s:.2f}  "
        f"condition 3rho(2+log2(1/rho))={3 * rho5 * (2 + math.log2(1 / rho5)):.3f} <= ln(1/s)={math.log(100):.3f}"
    )
    print(f"  Poisson curve on integer orders <= omega: rho'={tcdp_rho(ints, curve5, omega_s):.3e}")
    print(
        f"  k=10000: theorem eps={tcdp_to_dp(K_RUN * rho_s, omega_s, DELTA):.3f}  "
        f"RDP improved on integer ORDERS={rdp_to_dp_improved(ints_f, [K_RUN * e for e in curve5], DELTA):.3f}"
    )


# --- Figures -------------------------------------------------------------------------------------


def plot_privacy_loss(rng: np.random.Generator, sigma: float = SIGMA_RUN) -> Figure:
    rho = gaussian_zcdp_rho(sigma)
    losses = gaussian_privacy_loss_samples(rng, sigma, 200_000)
    fig, (left, right) = plt.subplots(1, 2, figsize=(11, 4))
    left.hist(losses, bins=120, density=True, color="C0", alpha=0.5, label="simulated privacy loss L(Y)")
    xs = np.linspace(-5.0, 6.0, 500)
    left.plot(
        xs,
        np.exp(-((xs - rho) ** 2) / (4 * rho)) / math.sqrt(4 * math.pi * rho),
        "k",
        label=rf"normal, mean $\rho$ = {rho:g}, variance $2\rho$",
    )
    left.axvline(rho, color="k", linestyle=":", linewidth=1)
    left.set(
        xlabel="privacy loss L",
        ylabel="density",
        xlim=(-5, 6),
        ylim=(0, 0.56),
        title=rf"Gaussian mechanism, $\sigma$ = {sigma:g}: L is normal",
    )
    left.legend(fontsize=9, loc="upper left")
    lams = [i / 20 for i in range(1, 161)]
    right.plot(lams, [gaussian_loss_tail(lam, rho) for lam in lams], "k", label=r"exact $\Pr[L > \rho + t]$")
    right.plot(
        lams,
        [zcdp_tail_bound(lam, rho) for lam in lams],
        "C1",
        linestyle="--",
        label=r"zCDP bound $e^{-t^2/(4\rho)}$",
    )
    right.set(
        yscale="log",
        xlabel=r"excess $t$ over the mean",
        ylabel="tail probability",
        ylim=(1e-8, 1.5),
        xlim=(0, 6.5),
        title="The tail falls off like a Gaussian's",
    )
    right.legend(fontsize=9, loc="lower left")
    return fig


def plot_rr_line() -> Figure:
    alphas = [1 + 10 ** (-2 + 4.3 * i / 400) for i in range(401)]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, eps in zip(axes, (1.0, 3.0)):
        ax.plot(alphas, [rr_rdp(a, eps) for a in alphas], "k", linewidth=2, label="randomized response, exact")
        ax.plot(
            alphas,
            [a * pure_to_zcdp(eps) for a in alphas],
            "C0",
            linestyle="--",
            label=r"zCDP line $\alpha\varepsilon^2/2$",
        )
        ax.plot(
            alphas,
            [a * pure_to_zcdp_sharp(eps) for a in alphas],
            "C2",
            linestyle="-.",
            label=r"sharp zCDP line $\alpha\varepsilon\tanh(\varepsilon/2)$",
        )
        ax.axhline(eps, color="C3", linestyle=":", label=r"pure DP cap $\varepsilon$")
        ax.plot(
            alphas,
            [rr_kl(eps)] * len(alphas),
            color="0.6",
            linewidth=1,
            label=r"KL $= \varepsilon\tanh(\varepsilon/2)$",
        )
        ax.set(
            xscale="log",
            xlabel=r"order $\alpha$",
            ylabel=r"$\varepsilon(\alpha)$",
            ylim=(0, 3 * eps),
            title=(
                rf"$\varepsilon$ = {eps:g}: "
                + ("the line is close near $\\alpha = 1$" if eps < 2 else "the cap is the better bound")
            ),
        )
        ax.legend(fontsize=8, loc="upper right")
    return fig


def plot_conversion() -> Figure:
    rhos = [10 ** (-3 + 4.7 * i / 120) for i in range(121)]
    fig, (left, right) = plt.subplots(1, 2, figsize=(11, 4))
    simple = [zcdp_to_dp_simple(r, DELTA) for r in rhos]
    improved = [zcdp_to_dp_improved(r, DELTA) for r in rhos]
    exact = [gaussian_eps(zcdp_gaussian_mu(r), DELTA) for r in rhos]
    left.plot(rhos, simple, "C1", linestyle="--", label="simple ρ + 2√(ρ ln(1/δ))")
    left.plot(rhos, improved, "C0", label="improved conversion")
    left.plot(rhos, exact, "k", linewidth=2, label="exact, Gaussian with μ = √(2ρ)")
    left.set(
        xscale="log",
        yscale="log",
        xlabel=r"$\rho$",
        ylabel=r"$\varepsilon$ at $\delta = 10^{-5}$",
        title=r"Converting $\rho$-zCDP to $(\varepsilon, \delta)$",
    )
    left.legend(fontsize=9, loc="upper left")
    right.plot(rhos, [s / e for s, e in zip(simple, exact)], "C1", linestyle="--", label="simple / exact")
    right.plot(rhos, [i / e for i, e in zip(improved, exact)], "C0", label="improved / exact")
    right.axhline(1, color="k", linewidth=1)
    right.set(
        xscale="log",
        xlabel=r"$\rho$",
        ylabel="ratio to the exact Gaussian ε",
        ylim=(0.95, 1.5),
        title="How much the conversion loses",
    )
    right.legend(fontsize=9)
    return fig


def plot_subsampled(q: float = Q_RUN, sigma: float = SIGMA_RUN, omega: float = 8.0) -> Figure:
    alphas = list(range(2, 513))
    curve = [subsampled_gaussian_rdp_int(a, q, sigma) for a in alphas]
    ints, grid_curve = running_example_curve(q, sigma)
    rho_z = gaussian_zcdp_rho(sigma)  # the smallest valid zCDP slope: the ratio ε(α)/α tends to it as α → ∞
    rho_t = tcdp_rho(ints, grid_curve, omega)
    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.plot(alphas, curve, "k", linewidth=2, label=rf"subsampled Gaussian, $q$ = {q:g}, $\sigma$ = {sigma:g}")
    ax.plot(
        alphas,
        [rho_z * a for a in alphas],
        "C3",
        linestyle="--",
        label=rf"best zCDP line $\rho\alpha$, $\rho = 1/(2\sigma^2)$ = {rho_z:g} (= no subsampling)",
    )
    tal = [a for a in alphas if a <= omega]
    ax.plot(
        tal,
        [rho_t * a for a in tal],
        "C0",
        linewidth=3,
        alpha=0.7,
        label=rf"tCDP line on $\alpha \leq \omega$ = {omega:g}, $\rho$ = {rho_t:.2e}",
    )
    ax.axvline(omega, color="C0", linestyle=":", linewidth=1)
    ax.set(
        xscale="log",
        yscale="log",
        xlabel=r"order $\alpha$ (integers)",
        ylabel=r"$\varepsilon(\alpha)$ per step",
        ylim=(1e-4, 600),
        title="One number cannot follow a curve that bends",
    )
    ax.legend(fontsize=9, loc="upper left")
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
        "concentrated_dp_privacy_loss.svg": plot_privacy_loss(rng),
        "concentrated_dp_rr_line.svg": plot_rr_line(),
        "concentrated_dp_conversion.svg": plot_conversion(),
        "concentrated_dp_subsampled.svg": plot_subsampled(),
    }
    for name, fig in figures.items():
        fig.tight_layout()
        fig.savefig(out / name, metadata={"Date": None})  # no timestamp, so output is reproducible
        plt.close(fig)
        print(f"saved {out / name}")


if __name__ == "__main__":
    main()
