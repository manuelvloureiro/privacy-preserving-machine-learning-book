"""Check the numbers in the amplification-by-subsampling chapter and draw its figures.

    uv run python code/part04-privacy-accounting/subsampling.py
    uv run python code/part04-privacy-accounting/subsampling.py --seed 1 --out /tmp/figures

Prints the numbers quoted in 05-subsampling.md (they match exactly with the default seed),
then saves the chapter's figures.

Conventions: q is the sampling rate, never a density. For the subsampled Gaussian, P0 = N(0, σ²)
is the output without the record and P1 = N(1, σ²) the output with it (ℓ2 sensitivity 1), so the
subsampled pair is the mixture Pq = (1 - q)P0 + qP1 against P0 (the "remove" pair of chapter 32).
In code the names mu0/mu1 are kept for P0/P1. Logs are natural.

Public functions used by later chapters: ORDERS, amplify_eps_delta, subsampled_gaussian_rdp,
subsampled_gaussian_rdp_curve, dpsgd_eps_rdp.
"""

import argparse
import math
from collections.abc import Callable, Sequence
from functools import cache
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
matplotlib.rcParams["svg.hashsalt"] = "subsampling"  # stable element ids, so reruns don't churn the SVGs
matplotlib.rcParams["svg.fonttype"] = "none"  # keep text as text: smaller, diffable files

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from numpy.typing import NDArray

# Figures are build output: _build/figures/<part>/, which the build copies into each book.
FIGURES = Path(__file__).resolve().parents[2] / "_build" / "figures" / Path(__file__).parent.name

Array = NDArray[np.float64]

# The RDP orders every chapter of this part optimizes over, so that numbers agree across chapters.
ORDERS: list[float] = [1 + x / 10 for x in range(1, 100)] + [float(a) for a in range(12, 64)] + [128.0, 256.0, 512.0]

# The running example of the part: DP-SGD with Poisson rate q, noise multiplier σ, δ = 1e-5.
Q_RUN, SIGMA_RUN, DELTA_RUN = 0.01, 1.0, 1e-5
STEPS_RUN: tuple[int, ...] = (1, 10, 100, 1000, 10_000)


def normal_cdf(x: float) -> float:
    """Φ(x); erfc keeps full relative precision in the lower tail."""
    return 0.5 * math.erfc(-x / math.sqrt(2))


def log_normal_cdf(x: float) -> float:
    """ln Φ(x), accurate in both tails (log1p near Φ = 1)."""
    if x < 0:
        return math.log(0.5 * math.erfc(-x / math.sqrt(2)))
    return math.log1p(-0.5 * math.erfc(x / math.sqrt(2)))


def logsumexp(values: Array) -> float:
    top = float(np.max(values))
    return top + math.log(float(np.sum(np.exp(values - top))))


def bisect(f: Callable[[float], bool], lo: float, hi: float, steps: int = 100) -> float:
    """Smallest x in [lo, hi] with f(x) true, for f false-then-true (returns the upper end of the bracket)."""
    for _ in range(steps):
        mid = (lo + hi) / 2
        if f(mid):
            hi = mid
        else:
            lo = mid
    return hi


# --- Toolkit: hockey-stick divergence on a finite outcome space --------------------------------


def hockey_stick(p: Sequence[float], r: Sequence[float], gamma: float) -> float:
    """E_γ(P‖R) = Σ (P(y) - γ R(y))_+ for two distributions on the same finite set."""
    return sum(max(pi - gamma * ri, 0.0) for pi, ri in zip(p, r, strict=True))


def mix(a: Sequence[float], b: Sequence[float], w: float) -> list[float]:
    """(1 - w)a + w b."""
    return [(1 - w) * ai + w * bi for ai, bi in zip(a, b, strict=True)]


def pair_delta(a: Sequence[float], b: Sequence[float], eps: float) -> float:
    """The smallest δ with a ≤ e^ε b + δ on every event: E_{e^ε}(a‖b)."""
    return hockey_stick(a, b, math.exp(eps))


def mixture_lemma_ratio(rng: np.random.Generator, eps: float, gamma: float, trials: int, size: int = 4) -> float:
    """Largest observed E_{γ'}(mixture‖mixture') / (γδ) over random triples (ν, μ, μ') on `size` outcomes.

    δ is the smallest value with μ ≤ e^ε μ' + δ, μ ≤ e^ε ν + δ, μ' ≤ e^ε μ + δ and μ' ≤ e^ε ν + δ, so
    the mixture lemma says the ratio never exceeds 1. γ' = 1 + γ(e^ε - 1).
    """
    gamma_prime = 1 + gamma * math.expm1(eps)
    worst = 0.0
    for _ in range(trials):
        nu, mu, mu2 = (list(map(float, rng.dirichlet(np.ones(size) * 0.5))) for _ in range(3))
        delta = max(
            pair_delta(mu, mu2, eps), pair_delta(mu, nu, eps), pair_delta(mu2, mu, eps), pair_delta(mu2, nu, eps)
        )
        if delta < 1e-9:
            continue
        lhs = max(
            hockey_stick(mix(nu, mu, gamma), mix(nu, mu2, gamma), gamma_prime),
            hockey_stick(mix(nu, mu2, gamma), mix(nu, mu, gamma), gamma_prime),
        )
        worst = max(worst, lhs / (gamma * delta))
    return worst


# --- Section 31.3: amplification for (ε, δ)-DP under Poisson sampling ---------------------------------


def amplify_eps_delta(eps: float, delta: float, q: float) -> tuple[float, float]:
    """(ε, δ)-DP base mechanism, Poisson rate q, add/remove: (ln(1 + q(e^ε - 1)), qδ).

    Balle, Barthe & Gaboardi (2018), Theorem 8. The same formula with q = m/n holds for a fixed batch
    of m out of n under replace-one neighbours (their Theorem 9).
    """
    return math.log1p(q * math.expm1(eps)), q * delta


def rr_output(eps: float, flagged: bool) -> list[float]:
    """Randomized response on the bit "the batch contains the flagged record": [P(y = 0), P(y = 1)]."""
    keep = math.exp(eps) / (1 + math.exp(eps))
    return [1 - keep, keep] if flagged else [keep, 1 - keep]


def subsampled_rr_eps(eps: float, q: float) -> float:
    """Exact ε of Poisson-subsampled randomized response: ln of the largest likelihood ratio, both directions."""
    mu0, mu1 = rr_output(eps, False), rr_output(eps, True)
    m = mix(mu0, mu1, q)
    return math.log(max(max(a / b, b / a) for a, b in zip(m, mu0, strict=True)))


# --- Section 31.4: matching the sampling scheme to the neighbouring relation --------------------------


def replace_one_under_poisson(eps: float, delta: float, q: float) -> tuple[float, float]:
    """Poisson sampling but replace-one neighbours: the base pair is two add/remove steps apart.

    Group privacy turns (ε, δ) into (2ε, (1 + e^ε)δ) for the base, then the mixture lemma applies.
    This is an upper bound, not the exact cost (see replace_one_under_poisson_pure).
    """
    return amplify_eps_delta(2 * eps, (1 + math.exp(eps)) * delta, q)


def replace_one_under_poisson_pure(eps: float, q: float) -> float:
    """Exact ε' for an ε-DP (δ = 0) base under Poisson sampling and replace-one neighbours.

    Given the rest of the sample with output R, the outputs are (1-q)R + qP and (1-q)R + qP', with
    P/R and P'/R in [e^{-ε}, e^ε]. The largest ratio is (1 - q + qe^ε)/(1 - q + qe^{-ε}), so
    ε' = ln(1 + q(e^ε - 1)) - ln(1 - q(1 - e^{-ε})). The Laplace mechanism on a sum attains it.
    """
    return math.log1p(q * math.expm1(eps)) - math.log1p(q * math.expm1(-eps))


def gaussian_delta(eps: float, sigma: float) -> float:
    """Exact δ(ε) of the Gaussian mechanism with sensitivity 1 and noise σ (Balle & Wang 2018)."""
    return normal_cdf(1 / (2 * sigma) - eps * sigma) - math.exp(eps) * normal_cdf(-1 / (2 * sigma) - eps * sigma)


def gaussian_eps(delta: float, sigma: float) -> float:
    """Smallest ε with gaussian_delta(ε, σ) ≤ δ."""
    return bisect(lambda e: gaussian_delta(e, sigma) <= delta, 0.0, 200.0, 200)


@cache
def shuffle_max_test(sigma: float, batches: int) -> tuple[Array, Array]:
    """P_S(E_τ), Q_S(E_τ) for the test E_τ = {max_t y_t ≥ τ}, over a grid of thresholds τ.

    Chua et al. (2024): with shuffled fixed-size batches the record sits in one of b batches, uniformly,
    and the other records reveal where; P_S mixes N(2e_t, σ²I) and Q_S mixes N(e_t, σ²I) over t.
    Then P_S(E_τ) = 1 - Φ((τ-2)/σ)Φ(τ/σ)^{b-1} and Q_S(E_τ) = 1 - Φ((τ-1)/σ)Φ(τ/σ)^{b-1}.
    """
    cs = [i * sigma / 100 for i in range(int((2 + 15 * sigma) / (sigma / 100)) + 1)]
    p_e: list[float] = []
    q_e: list[float] = []
    for c in cs:
        rest = (batches - 1) * log_normal_cdf(c / sigma)
        p_e.append(-math.expm1(log_normal_cdf((c - 2) / sigma) + rest))
        q_e.append(-math.expm1(log_normal_cdf((c - 1) / sigma) + rest))
    return np.array(p_e), np.array(q_e)


def shuffle_delta_lower(eps: float, sigma: float, batches: int) -> float:
    """A lower bound on δ(ε) for one epoch of shuffled batches: max over τ of P_S(E_τ) - e^ε Q_S(E_τ)."""
    p_e, q_e = shuffle_max_test(sigma, batches)
    return max(0.0, float(np.max(p_e - math.exp(eps) * q_e)))


def shuffle_eps_lower(delta: float, sigma: float, batches: int) -> float:
    """The smallest ε that the max test does not rule out at this δ: a lower bound on the true ε."""
    return bisect(lambda e: shuffle_delta_lower(e, sigma, batches) <= delta, 0.0, 200.0, 100)


# --- Section 31.5: RDP of the Poisson-subsampled Gaussian ----------------------------------------------


def log_a_binomial(alpha: int, q: float, sigma: float) -> float:
    """ln A_α for integer α: A_α = Σ_j C(α, j)(1-q)^{α-j} q^j exp((j² - j)/(2σ²))  (Mironov et al. 2019, §3.3)."""
    terms = np.array(
        [
            math.lgamma(alpha + 1)
            - math.lgamma(j + 1)
            - math.lgamma(alpha - j + 1)
            + (alpha - j) * math.log1p(-q)
            + j * math.log(q)
            + (j * j - j) / (2 * sigma**2)
            for j in range(alpha + 1)
        ]
    )
    return logsumexp(terms)


def binomial_terms(alpha: int, q: float, sigma: float) -> list[float]:
    """The individual terms C(α, j)(1-q)^{α-j} q^j exp((j² - j)/(2σ²)) of A_α, j = 0..α."""
    return [
        math.comb(alpha, j) * (1 - q) ** (alpha - j) * q**j * math.exp((j * j - j) / (2 * sigma**2))
        for j in range(alpha + 1)
    ]


def _grid(lo: float, hi: float, sigma: float) -> Array:
    return np.linspace(lo, hi, int((hi - lo) / (sigma / 100)) + 1)


def _log_mixture_ratio(z: Array, q: float, sigma: float) -> Array:
    """ln((1 - q) + q μ1(z)/μ0(z)) with μ1/μ0 = exp((2z - 1)/(2σ²)), computed without overflow."""
    return np.logaddexp(math.log1p(-q), math.log(q) + (2 * z - 1) / (2 * sigma**2))


def _log_normal_pdf(z: Array, sigma: float) -> Array:
    return -(z**2) / (2 * sigma**2) - math.log(sigma * math.sqrt(2 * math.pi))


def log_a_numeric(alpha: float, q: float, sigma: float) -> float:
    """ln A_α = ln E_{μ0}[((1-q) + q μ1/μ0)^α] by the trapezoid rule on a fine grid, in log space.

    The integrand peaks near z = 0 and z = α (where the q^α e^{α(2z-1)/2σ²} part tilts μ0), so the grid
    runs from -12σ to α + 12σ with step σ/100.
    """
    z = _grid(-12 * sigma, alpha + 12 * sigma, sigma)
    return logsumexp(_log_normal_pdf(z, sigma) + alpha * _log_mixture_ratio(z, q, sigma)) + math.log(z[1] - z[0])


def log_b_numeric(alpha: float, q: float, sigma: float) -> float:
    """ln B_α with B_α = E_{μ0}[((1-q) + q μ1/μ0)^{-(α-1)}] = exp((α-1) D_α(μ0‖mixture)).

    The integrand is μ0 times a factor between (1-q)^{-(α-1)} and 0, so its mass stays within ±12σ of 0.
    """
    z = _grid(-12 * sigma - 2, 12 * sigma + 2, sigma)
    return logsumexp(_log_normal_pdf(z, sigma) - (alpha - 1) * _log_mixture_ratio(z, q, sigma)) + math.log(z[1] - z[0])


def subsampled_gaussian_rdp_directions(alpha: float, q: float, sigma: float) -> tuple[float, float]:
    """(D_α(mixture‖μ0), D_α(μ0‖mixture)) for mixture = (1-q)N(0,σ²) + qN(1,σ²), μ0 = N(0,σ²)."""
    log_a = log_a_binomial(int(alpha), q, sigma) if float(alpha).is_integer() else log_a_numeric(alpha, q, sigma)
    return log_a / (alpha - 1), log_b_numeric(alpha, q, sigma) / (alpha - 1)


def subsampled_gaussian_rdp(alpha: float, q: float, sigma: float) -> float:
    """RDP ε(α) of the Poisson-subsampled Gaussian (add/remove, sensitivity 1): the larger direction.

    Exact binomial sum for integer α, numerical integration for fractional α.
    """
    if q == 0:
        return 0.0
    if q == 1:
        return alpha / (2 * sigma**2)
    return max(subsampled_gaussian_rdp_directions(alpha, q, sigma))


@cache
def _rdp_curve_cached(q: float, sigma: float, orders: tuple[float, ...]) -> tuple[float, ...]:
    return tuple(subsampled_gaussian_rdp(a, q, sigma) for a in orders)


def subsampled_gaussian_rdp_curve(q: float, sigma: float, orders: Sequence[float] = ORDERS) -> list[float]:
    """ε(α) on every order; cached per (q, σ, orders), since accountants call it repeatedly."""
    return list(_rdp_curve_cached(q, sigma, tuple(orders)))


def mixture_privacy_loss(z: Array, q: float, sigma: float) -> Array:
    """Privacy loss ln(((1-q)μ0 + qμ1)/μ0)(z) of the subsampled Gaussian."""
    return _log_mixture_ratio(z, q, sigma)


# --- Section 31.6: small-q asymptotics and the moments accountant -------------------------------------


def small_q_leading(alpha: float, q: float, sigma: float) -> float:
    """The j = 2 term of the binomial sum: ε(α) ≈ α q² (e^{1/σ²} - 1)/2."""
    return alpha * q**2 * math.expm1(1 / sigma**2) / 2


def abadi_bound(alpha: float, q: float, sigma: float) -> float:
    """Abadi et al. (2016), Lemma 3, leading term: α_M(λ) ≤ q²λ(λ+1)/((1-q)σ²), so ε(α) ≤ q²α/((1-q)σ²)."""
    return q**2 * alpha / ((1 - q) * sigma**2)


def abadi_order_limit(q: float, sigma: float) -> float:
    """Lemma 3 holds for λ ≤ σ² ln(1/(qσ)), that is α = λ + 1 ≤ 1 + σ² ln(1/(qσ))."""
    return 1 + sigma**2 * math.log(1 / (q * sigma))


def first_order_breaking(q: float, sigma: float, factor: float = 2.0) -> float:
    """The first order on the grid where the exact ε(α) exceeds `factor` times the leading term."""
    for a, e in zip(ORDERS, subsampled_gaussian_rdp_curve(q, sigma), strict=True):
        if e > factor * small_q_leading(a, q, sigma):
            return a
    return math.inf


def breaking_order_fine(q: float, sigma: float, lo: float, hi: float, factor: float = 2.0) -> float:
    """The order in [lo, hi] where the exact ε(α) crosses `factor` times the leading term, by bisection.

    Fractional orders use the numerical integral, so this resolves the crossing between grid orders.
    """
    return bisect(lambda a: subsampled_gaussian_rdp(a, q, sigma) > factor * small_q_leading(a, q, sigma), lo, hi, 40)


def breakdown_order(q: float, sigma: float) -> float:
    """α ≈ 2σ² ln(1/q): where the j = α term of the binomial sum overtakes the j = 2 term."""
    return 2 * sigma**2 * math.log(1 / q)


# --- Section 31.7: the running example, DP-SGD -----------------------------------------------------


def mironov_eps(alpha: float, rdp: float, delta: float) -> float:
    """ε = ε(α) + ln(1/δ)/(α - 1) (Mironov 2017, Proposition 3)."""
    return rdp + math.log(1 / delta) / (alpha - 1)


def improved_eps(alpha: float, rdp: float, delta: float) -> float:
    """ε = ε(α) + ln((α - 1)/α) - (ln δ + ln α)/(α - 1) (Balle et al. 2020, Theorem 21)."""
    return max(0.0, rdp + math.log((alpha - 1) / alpha) - (math.log(delta) + math.log(alpha)) / (alpha - 1))


def dpsgd_eps_rdp(
    q: float, sigma: float, steps: int, delta: float, conversion: str = "improved"
) -> tuple[float, float]:
    """(ε, best order) for `steps` Poisson-subsampled Gaussian steps, composed in RDP and converted at δ."""
    convert = {"mironov": mironov_eps, "improved": improved_eps}[conversion]
    curve = subsampled_gaussian_rdp_curve(q, sigma)
    return min((convert(a, steps * e, delta), a) for a, e in zip(ORDERS, curve, strict=True))


def advanced_composition(eps: float, delta: float, k: int, delta_prime: float) -> tuple[float, float]:
    """Dwork, Rothblum & Vadhan: k-fold (ε, δ) is (ε√(2k ln(1/δ')) + kε(e^ε - 1), kδ + δ')."""
    return eps * math.sqrt(2 * k * math.log(1 / delta_prime)) + k * eps * math.expm1(eps), k * delta + delta_prime


def dpsgd_eps_basic(q: float, sigma: float, steps: int, delta: float) -> float:
    """Per step: Gaussian (ε0, δ0) with δ0 = δ/(kq), amplified to (ε1, qδ0); then basic composition kε1."""
    eps1, _ = amplify_eps_delta(gaussian_eps(delta / (steps * q), sigma), 0.0, q)
    return steps * eps1


def advanced_composition_tanh(eps: float, k: int, delta_prime: float) -> float:
    """The sharper advanced composition of Part 3: ε√(2k ln(1/δ')) + kε tanh(ε/2)."""
    return eps * math.sqrt(2 * k * math.log(1 / delta_prime)) + k * eps * math.tanh(eps / 2)


def dpsgd_eps_advanced_tanh(q: float, sigma: float, steps: int, delta: float) -> float:
    """The per-step route with the tanh form of advanced composition, δ split optimized as below."""
    best = math.inf
    for i in range(1, 100):
        frac = i / 100
        eps1, _ = amplify_eps_delta(gaussian_eps((1 - frac) * delta / (steps * q), sigma), 0.0, q)
        best = min(best, advanced_composition_tanh(eps1, steps, frac * delta))
    return best


def rdp_beats_basic_from(q: float, sigma: float, delta: float, max_steps: int = 100) -> int:
    """The smallest k from which the RDP accountant (improved) is below per-step basic composition."""
    for k in range(1, max_steps + 1):
        if dpsgd_eps_rdp(q, sigma, k, delta)[0] < dpsgd_eps_basic(q, sigma, k, delta):
            return k
    return max_steps + 1


def dpsgd_eps_advanced(q: float, sigma: float, steps: int, delta: float) -> tuple[float, float]:
    """Same per-step route, then advanced composition; the split δ = kqδ0 + δ' is optimized over a grid.

    Returns (ε, fraction of δ given to δ').
    """
    best = (math.inf, 0.0)
    for i in range(1, 100):
        frac = i / 100
        eps1, _ = amplify_eps_delta(gaussian_eps((1 - frac) * delta / (steps * q), sigma), 0.0, q)
        eps, _ = advanced_composition(eps1, 0.0, steps, frac * delta)
        best = min(best, (eps, frac))
    return best


# --- Section 31.8: no generic tight amplification in RDP -----------------------------------------------


def zhu_wang_general_bound(alpha: int, q: float, rdp: Callable[[float], float]) -> float:
    """Zhu & Wang (2019), Theorem 5: RDP of any Poisson-subsampled mechanism with RDP curve `rdp`, integer α."""
    total = (1 - q) ** (alpha - 1) * (alpha * q - q + 1)
    total += math.comb(alpha, 2) * q**2 * (1 - q) ** (alpha - 2) * math.exp(rdp(2))
    total += 3 * sum(
        math.comb(alpha, j) * (1 - q) ** (alpha - j) * q**j * math.exp((j - 1) * rdp(j)) for j in range(3, alpha + 1)
    )
    return math.log(total) / (alpha - 1)


# --- The chapter's numbers ------------------------------------------------------------------


def print_table(rng: np.random.Generator) -> None:
    q, sigma, delta = Q_RUN, SIGMA_RUN, DELTA_RUN

    print("[Part 1: (eps, delta) amplification, Poisson, add/remove]")
    for qq in (0.01, 0.1, 0.5):
        row = "  ".join(f"eps={e:g}: {amplify_eps_delta(e, 0.0, qq)[0]:.4f}" for e in (0.1, 1.0, 5.0, 10.0))
        print(f"  q={qq:g}: {row}")
    e, qq = 1.0, 0.1
    mu0, mu1 = rr_output(e, False), rr_output(e, True)
    m = mix(mu0, mu1, qq)
    print(
        f"  randomized response eps={e:g}, q={qq:g}: mu0={[round(v, 4) for v in mu0]}  "
        f"mu1={[round(v, 4) for v in mu1]}  mixture={[round(v, 4) for v in m]}"
    )
    print(
        f"  ratios mixture/mu0={[round(a / b, 4) for a, b in zip(m, mu0)]}  "
        f"mu0/mixture={[round(b / a, 4) for a, b in zip(m, mu0)]}"
    )
    print(
        f"  exact subsampled eps={subsampled_rr_eps(e, qq):.4f}  formula={amplify_eps_delta(e, 0.0, qq)[0]:.4f}  "
        f"q*eps={qq * e:.4f}"
    )
    print(
        f"  mixture lemma, 20000 random triples: largest E/(gamma*delta) = "
        f"{mixture_lemma_ratio(rng, 1.0, 0.1, 20_000):.4f}"
    )

    print("[Part 2: sampling scheme and neighbouring relation]")
    e = 1.0
    print(
        f"  eps={e:g}, q={q:g}: Poisson add/remove {amplify_eps_delta(e, 0.0, q)[0]:.5f}  "
        f"fixed-size replace-one {amplify_eps_delta(e, 0.0, q)[0]:.5f}  "
        f"Poisson replace-one (group privacy) {replace_one_under_poisson(e, 0.0, q)[0]:.5f}  "
        f"ratio {replace_one_under_poisson(e, 0.0, q)[0] / amplify_eps_delta(e, 0.0, q)[0]:.2f}  "
        f"e^eps+1={math.exp(e) + 1:.2f}"
    )
    exact_pure = replace_one_under_poisson_pure(e, q)
    print(
        f"  Poisson replace-one, exact for pure DP: {exact_pure:.5f}  "
        f"ratio to add/remove {exact_pure / amplify_eps_delta(e, 0.0, q)[0]:.2f}"
    )
    batches = round(1 / q)
    eps_p, _ = dpsgd_eps_rdp(q, sigma, batches, delta)
    print(
        f"  one epoch, b={batches} batches, sigma={sigma:g}, delta={delta:g}: Poisson RDP upper eps={eps_p:.2f}  "
        f"shuffled lower eps={shuffle_eps_lower(delta, sigma, batches):.2f}  "
        f"deterministic batches eps={gaussian_eps(delta, sigma):.2f}"
    )
    for s in (0.5, 1.0, 1.5, 2.0, 2.5, 3.0):
        print(
            f"  one epoch, sigma={s:g}: deterministic={gaussian_eps(delta, s):.2f}  "
            f"shuffled lower={shuffle_eps_lower(delta, s, batches):.2f}  "
            f"Poisson upper={dpsgd_eps_rdp(q, s, batches, delta)[0]:.2f}"
        )
    print(
        f"  Chua et al. check, sigma=0.4, b=10000: shuffled delta(4) >= {shuffle_delta_lower(4.0, 0.4, 10_000):.3f}  "
        f"deterministic delta(4) = {gaussian_delta(4.0, 0.4):.3f}"
    )
    for k in (100, 1000, 10_000):
        print(
            f"  k={k}: Poisson sigma={sigma:g} eps={dpsgd_eps_rdp(q, sigma, k, delta)[0]:.2f}  "
            f"fixed-size batches (sigma/2={sigma / 2:g}) eps={dpsgd_eps_rdp(q, sigma / 2, k, delta)[0]:.2f}"
        )
    print(
        f"  Lebeda et al. check, q=0.001, sigma=0.8, k=10000: Poisson RDP eps="
        f"{dpsgd_eps_rdp(0.001, 0.8, 10_000, delta)[0]:.2f}  fixed-size RDP eps="
        f"{dpsgd_eps_rdp(0.001, 0.4, 10_000, delta)[0]:.2f}"
    )

    print(f"[Part 3: RDP of the subsampled Gaussian, q={q:g}, sigma={sigma:g}]")
    a2 = 1 + q**2 * math.expm1(1 / sigma**2)
    print(
        f"  alpha=2 by hand: A_2 = 1 + q^2(e^(1/sigma^2) - 1) = {a2:.8f}  eps(2) = {math.log(a2):.4e}  "
        f"terms={[f'{t:.6f}' for t in binomial_terms(2, q, sigma)]}"
    )
    for a in (2, 3, 10, 32):
        print(
            f"  alpha={a}: binomial={log_a_binomial(a, q, sigma) / (a - 1):.6e}  "
            f"integral={log_a_numeric(float(a), q, sigma) / (a - 1):.6e}"
        )
    for a in (1.5, 2.0, 4.0, 8.0, 16.0, 32.0, 128.0):
        fa, fb = subsampled_gaussian_rdp_directions(a, q, sigma)
        print(f"  alpha={a:g}: D(mixture||mu0)={fa:.4e}  D(mu0||mixture)={fb:.4e}")
    for s in (0.5, 2.0):
        worst = max(fb - fa for fa, fb in (subsampled_gaussian_rdp_directions(a, q, s) for a in ORDERS))
        print(f"  sigma={s:g}: largest D(mu0||mixture) - D(mixture||mu0) over ORDERS = {worst:.2e}")
    for a in (2.0, 8.0, 32.0):
        print(
            f"  alpha={a:g}: subsampled eps={subsampled_gaussian_rdp(a, q, sigma):.4e}  "
            f"unsubsampled alpha/(2 sigma^2)={a / (2 * sigma**2):g}"
        )

    print(f"[Part 4: small-q asymptotics, q={q:g}]")
    for s, lo, hi in ((1.0, 8.0, 10.0), (4.0, 128.0, 256.0)):
        print(
            f"  sigma={s:g}: Abadi order limit alpha <= {abadi_order_limit(q, s):.1f}  "
            f"first order where exact > 2x leading: {first_order_breaking(q, s):g}  "
            f"(fine: {breaking_order_fine(q, s, lo, hi):.1f})  "
            f"2 sigma^2 ln(1/q) = {breakdown_order(q, s):.1f}  sigma^2 ln(1/q) = {breakdown_order(q, s) / 2:.1f}"
        )
        mid = breakdown_order(q, s) / 2
        print(
            f"    at alpha = sigma^2 ln(1/q): exact/leading = "
            f"{subsampled_gaussian_rdp(mid, q, s) / small_q_leading(mid, q, s):.2f}"
        )
        for a in (2, 3, 4, 6, 8, 10, 16, 32):
            ex = subsampled_gaussian_rdp(a, q, s)
            lead = small_q_leading(a, q, s)
            print(
                f"    alpha={a}: exact={ex:.4e}  leading={lead:.4e}  Abadi={abadi_bound(a, q, s):.4e}  "
                f"exact/leading={ex / lead:.2f}"
            )
    terms = binomial_terms(10, q, 1.0)
    print("  alpha=10, sigma=1 binomial terms j=2..10: " + "  ".join(f"{t:.1e}" for t in terms[2:]))
    eps_m, a_m = dpsgd_eps_rdp(0.01, 4.0, 10_000, delta, "mironov")
    eps_i, a_i = dpsgd_eps_rdp(0.01, 4.0, 10_000, delta, "improved")
    print(
        f"  Abadi et al. example q=0.01, sigma=4, k=10000, delta=1e-5: Mironov eps={eps_m:.4f} at alpha={a_m:g}  "
        f"improved eps={eps_i:.4f} at alpha={a_i:g}"
    )

    print(f"[Part 5: DP-SGD q={q:g}, sigma={sigma:g}, delta={delta:g}]")
    for k in STEPS_RUN:
        em, am = dpsgd_eps_rdp(q, sigma, k, delta, "mironov")
        ei, ai = dpsgd_eps_rdp(q, sigma, k, delta, "improved")
        adv, frac = dpsgd_eps_advanced(q, sigma, k, delta)
        print(
            f"  k={k}: RDP+Mironov={em:.2f} (alpha={am:g})  RDP+improved={ei:.2f} (alpha={ai:g})  "
            f"basic={dpsgd_eps_basic(q, sigma, k, delta):.2f}  advanced={adv:.2f} (delta' share {frac:.2f})"
        )
        rdp_ai = subsampled_gaussian_rdp(ai, q, sigma)
        print(
            f"    at the best order {ai:g}: exact/leading={rdp_ai / small_q_leading(ai, q, sigma):.2f}  "
            f"k*eps(alpha)={k * rdp_ai:.3f}  improved saves {100 * (1 - ei / em):.1f}% over Mironov"
        )
    eps0 = gaussian_eps(delta / (10_000 * q), sigma)
    print(
        f"  k=10000 per-step route: delta0={delta / (10_000 * q):.0e}  eps0={eps0:.2f}  "
        f"amplified eps1={amplify_eps_delta(eps0, 0.0, q)[0]:.3f}"
    )
    print(f"  k=10000 advanced composition, tanh form: {dpsgd_eps_advanced_tanh(q, sigma, 10_000, delta):.0f}")
    print(f"  RDP (improved) beats per-step basic composition from k={rdp_beats_basic_from(q, sigma, delta)}")

    print("[Part 6: generic Poisson bound (Zhu & Wang Thm 5) vs exact, Gaussian sigma=1, q=0.01]")
    for a in (2, 4, 8, 16, 32):
        gen = zhu_wang_general_bound(a, q, lambda j: j / (2 * sigma**2))
        ex = subsampled_gaussian_rdp(a, q, sigma)
        print(f"  alpha={a}: general={gen:.4e}  exact={ex:.4e}  ratio={gen / ex:.2f}")


# --- Figures --------------------------------------------------------------------------------


def plot_amplification() -> Figure:
    eps = np.logspace(-2, 1, 200)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for i, qq in enumerate((0.5, 0.1, 0.01)):
        ax.plot(
            eps,
            [amplify_eps_delta(float(e), 0.0, qq)[0] for e in eps],
            color=f"C{i}",
            label=rf"$q$ = {qq:g}: $\ln(1 + q(e^\varepsilon - 1))$",
        )
        ax.plot(eps, qq * eps, color=f"C{i}", linestyle=":", linewidth=1)
    ax.plot(eps, eps, color="0.5", linestyle="--", linewidth=1, label="no sampling: ε′ = ε")
    ax.set(
        xscale="log",
        yscale="log",
        xlabel=r"base mechanism $\varepsilon$",
        ylabel="subsampled ε′",
        ylim=(5e-5, 20),
        title="Amplification: ε′ ≈ qε (dotted) until ε ≈ ln(1/q)",
    )
    ax.legend(fontsize=9, loc="lower right")
    return fig


def plot_mixture(q: float = 0.1, sigma: float = 1.0) -> Figure:
    z = np.linspace(-4, 5, 600)
    pdf0 = np.exp(_log_normal_pdf(z, sigma))
    pdf1 = np.exp(_log_normal_pdf(z - 1, sigma))
    fig, (left, right) = plt.subplots(1, 2, figsize=(11, 4))
    left.plot(z, pdf0, label=r"$P_0$ = N(0, 1): record not used")
    left.plot(z, pdf1, label=r"$P_1$ = N(1, 1): record used")
    left.plot(
        z, (1 - q) * pdf0 + q * pdf1, color="k", linestyle="--", label=rf"mixture $P_q = (1-q)P_0 + qP_1$, $q$ = {q:g}"
    )
    left.set(xlabel="output y", ylabel="density", ylim=(0, 0.55), title="The record is used only with probability q")
    left.legend(fontsize=8, loc="upper left")
    right.plot(z, (2 * z - 1) / (2 * sigma**2), label=r"no sampling: $\ln(P_1/P_0) = (2y-1)/2$")
    right.plot(
        z, mixture_privacy_loss(z, q, sigma), color="k", linestyle="--", label=r"subsampled: $\ln((1-q) + q\,P_1/P_0)$"
    )
    right.plot(
        z,
        (2 * z - 1) / (2 * sigma**2) + math.log(q),
        color="C1",
        linestyle=":",
        label=r"its asymptote $(2y-1)/2 + \ln q$",
    )
    right.axhline(math.log(1 - q), color="0.5", linewidth=0.8, label=r"floor $\ln(1-q)$")
    right.set(
        xlabel="output y",
        ylabel="privacy loss (nats)",
        ylim=(-3, 4),
        title=r"Privacy loss: flat near $\ln(1-q)$, then $\ln(1/q)$ below",
    )
    right.legend(fontsize=8, loc="upper left")
    return fig


def plot_rdp_curve(q: float = Q_RUN, sigma: float = SIGMA_RUN) -> Figure:
    curve = subsampled_gaussian_rdp_curve(q, sigma)
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.plot(ORDERS, [a / (2 * sigma**2) for a in ORDERS], label=r"no sampling: $\alpha/(2\sigma^2)$")
    ax.plot(ORDERS, curve, color="k", label=rf"subsampled, exact ($q$ = {q:g}, $\sigma$ = {sigma:g})")
    ax.plot(
        ORDERS,
        [small_q_leading(a, q, sigma) for a in ORDERS],
        linestyle="--",
        label=r"small-$q$ term $\alpha q^2(e^{1/\sigma^2} - 1)/2$",
    )
    ax.plot(
        ORDERS,
        [abadi_bound(a, q, sigma) for a in ORDERS],
        linestyle=":",
        label=r"Abadi et al. $q^2\alpha/((1-q)\sigma^2)$",
    )
    ax.axvline(
        abadi_order_limit(q, sigma),
        color="0.6",
        linewidth=0.8,
        label=r"Abadi et al.'s range ends: $\alpha = 1 + \sigma^2\ln(1/(q\sigma))$",
    )
    ax.axvline(
        breakdown_order(q, sigma),
        color="0.3",
        linewidth=0.8,
        linestyle="--",
        label=r"expansion breaks: $\alpha \approx 2\sigma^2\ln(1/q)$",
    )
    ax.set(
        xscale="log",
        yscale="log",
        xlabel=r"order $\alpha$",
        ylabel=r"$\varepsilon(\alpha)$",
        ylim=(1e-8, 600),
        title="RDP of the subsampled Gaussian: quadratic in q, then it catches up",
    )
    ax.legend(fontsize=8, loc="lower right", framealpha=0.9)
    return fig


def plot_dpsgd(q: float = Q_RUN, sigma: float = SIGMA_RUN, delta: float = DELTA_RUN) -> Figure:
    ks = sorted({int(round(k)) for k in np.logspace(0, 4, 25)})
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.plot(
        ks, [dpsgd_eps_basic(q, sigma, k, delta) for k in ks], marker=".", label="per step (ε, δ), basic composition"
    )
    ax.plot(
        ks,
        [dpsgd_eps_advanced(q, sigma, k, delta)[0] for k in ks],
        marker=".",
        label="per step (ε, δ), advanced composition",
    )
    ax.plot(
        ks, [dpsgd_eps_rdp(q, sigma, k, delta, "mironov")[0] for k in ks], marker=".", label="RDP, Mironov's conversion"
    )
    ax.plot(
        ks,
        [dpsgd_eps_rdp(q, sigma, k, delta, "improved")[0] for k in ks],
        marker=".",
        color="k",
        label="RDP, improved conversion",
    )
    ax.set(
        xscale="log",
        yscale="log",
        xlabel="steps k",
        ylabel=r"$\varepsilon$ at $\delta = 10^{-5}$",
        title=rf"DP-SGD, $q$ = {q:g}, $\sigma$ = {sigma:g}: accounting route decides $\varepsilon$",
    )
    ax.legend(fontsize=8, loc="upper left")
    return fig


def plot_sampling_schemes(q: float = Q_RUN, delta: float = DELTA_RUN) -> Figure:
    batches = round(1 / q)
    sigmas = [0.5 + i * 0.125 for i in range(21)]
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.plot(sigmas, [gaussian_eps(delta, s) for s in sigmas], label="deterministic batches (exact)")
    ax.plot(
        sigmas,
        [shuffle_eps_lower(delta, s, batches) for s in sigmas],
        marker=".",
        label="shuffled fixed-size batches (lower bound, max test)",
    )
    ax.plot(
        sigmas,
        [dpsgd_eps_rdp(q, s, batches, delta)[0] for s in sigmas],
        color="k",
        label="Poisson sampling (RDP upper bound)",
    )
    ax.set(
        yscale="log",
        xlabel=r"noise multiplier $\sigma$",
        ylabel=r"$\varepsilon$ at $\delta = 10^{-5}$",
        title=rf"One epoch of {batches} batches: the sampler you run sets $\varepsilon$",
    )
    ax.legend(fontsize=8, loc="upper right")
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
        "subsampling_amplification.svg": plot_amplification(),
        "subsampling_mixture.svg": plot_mixture(),
        "subsampling_rdp_curve.svg": plot_rdp_curve(),
        "subsampling_dpsgd.svg": plot_dpsgd(),
        "subsampling_sampling_schemes.svg": plot_sampling_schemes(),
    }
    for name, fig in figures.items():
        fig.tight_layout()
        fig.savefig(out / name, metadata={"Date": None})  # no timestamp, so output is reproducible
        plt.close(fig)
        print(f"saved {out / name}")


if __name__ == "__main__":
    main()
