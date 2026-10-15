"""Check the numbers in the Choosing ε chapter and draw its figures.

    uv run python code/part05-dp-ml/choosing_epsilon.py
    uv run python code/part05-dp-ml/choosing_epsilon.py --seed 1 --out /tmp/figures

Prints the numbers quoted in 09-choosing-epsilon.md (they match exactly with the default seed), then saves
the chapter's figures.

The chapter asks what a budget ε means for an attacker of the part's running example: DP-SGD with Poisson
sampling rate q = 0.01 and T = 10,000 steps, with the noise multiplier σ that the RDP accountant of the
DP-SGD chapter calibrates to each ε at δ = 1e-5. For each budget it computes

- the bound that the single pair (ε, δ) puts on a membership test, TPR ≤ e^ε FPR + δ;
- the bound that the whole RDP curve puts on it, TPR ≤ (FPR e^{R(α)})^{1 - 1/α};
- the exact trade-off curve of the mechanism, from its privacy loss distribution (PLD) composed by FFT on a
  grid, rounded up and rounded down so that the truth lies between the two curves;
- the Gaussian-DP central-limit approximation μ = q √(T (e^{1/σ²} - 1));
- posterior probabilities of membership, reconstruction bounds, the price of a smaller δ;
- and the success of a simple loss-threshold attack on models actually trained at each budget.

The test accuracies of the DP-SGD chapter are quoted from its table (ACCURACY_CH37), not recomputed.
"""

import os

# The models are tiny; multi-threaded BLAS only adds overhead.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import argparse
import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from functools import cache
from pathlib import Path
from statistics import NormalDist

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from dp_sgd import (
    DELTA_RUN,
    ORDERS,
    Q_RUN,
    STEPS_RUN,
    DpsgdConfig,
    accuracy,
    dpsgd_epsilon,
    logistic_losses,
    make_task,
    noise_multiplier_for,
    rdp_curve,
    train_dpsgd,
)
from matplotlib.figure import Figure
from numpy.typing import NDArray

# Set after importing dp_sgd, which sets its own salt at import time.
matplotlib.rcParams["svg.hashsalt"] = "choosing_epsilon"  # stable element ids, so reruns don't churn the SVGs
matplotlib.rcParams["svg.fonttype"] = "none"  # keep text as text: smaller, diffable files

# Figures are build output: _build/figures/<part>/, which the build copies into each book.
FIGURES = Path(__file__).resolve().parents[2] / "_build" / "figures" / Path(__file__).parent.name

Array = NDArray[np.float64]

EPS_GRID: tuple[float, ...] = (0.5, 1.0, 2.0, 4.0, 8.0, 16.0)
FPRS: tuple[float, ...] = (1e-3, 1e-2, 1e-1)
PRIORS: tuple[float, ...] = (0.01, 0.1, 0.5)
N_RECORDS = 10_000
# Loss grid of the PLD: the true composed loss lies within T h of the rounded ones (0.1 and 0.02).
H_COARSE, H_FINE = 1e-5, 2e-6
TAIL = 1e-15  # probability mass truncated on each side of a PLD, always on the safe side
# Test accuracy of the running example at each budget, mean of 3 runs, from the DP-SGD chapter's table.
ACCURACY_CH37: dict[float, float] = {0.5: 0.8209, 1.0: 0.9175, 2.0: 0.9441, 4.0: 0.9548, 8.0: 0.9604}
ACCURACY_NONPRIVATE_CH37 = 0.9610

_STD_NORMAL = NormalDist()


# --- Gaussian tools ---------------------------------------------------------------------------


def normal_cdf(x: float) -> float:
    """Φ(x); erfc keeps full relative precision in the lower tail."""
    return 0.5 * math.erfc(-x / math.sqrt(2))


def normal_sf(x: float) -> float:
    """1 - Φ(x), accurate in the upper tail."""
    return 0.5 * math.erfc(x / math.sqrt(2))


def normal_quantile(p: float) -> float:
    """Φ^{-1}(p)."""
    return _STD_NORMAL.inv_cdf(p)


def _sf_array(x: Array) -> Array:
    return np.array([normal_sf(float(v)) for v in x], dtype=np.float64)


def _cdf_array(x: Array) -> Array:
    return np.array([normal_cdf(float(v)) for v in x], dtype=np.float64)


def bisect_root(f: Callable[[float], float], lo: float, hi: float, steps: int = 100) -> float:
    """A root of a monotone f on [lo, hi] with f(lo) and f(hi) of opposite signs."""
    flo = f(lo)
    for _ in range(steps):
        mid = (lo + hi) / 2
        fmid = f(mid)
        if (fmid > 0) == (flo > 0):
            lo, flo = mid, fmid
        else:
            hi = mid
    return (lo + hi) / 2


# --- Section: what one (ε, δ) pair says about a membership test --------------------------------


def pair_tpr_bound(eps: float, delta: float, fpr: float) -> float:
    """Largest TPR at a given FPR allowed by (ε, δ)-DP: both lines of the testing region, and 1."""
    return min(1.0, math.exp(eps) * fpr + delta, 1 - math.exp(-eps) * (1 - delta - fpr))


def pair_advantage_bound(eps: float, delta: float) -> float:
    """Largest TPR - FPR allowed by (ε, δ)-DP: (e^ε - 1 + 2δ)/(e^ε + 1)."""
    return (math.exp(eps) - 1 + 2 * delta) / (math.exp(eps) + 1)


# --- Section: the whole curve of DP-SGD -------------------------------------------------------


def rdp_total(sigma: float, steps: int = STEPS_RUN, q: float = Q_RUN) -> list[float]:
    """The composed RDP curve T ε_{q,σ}(α) over the part's grid of orders."""
    return [steps * e for e in rdp_curve(q, sigma)]


def rdp_tpr_bound(rdp: Sequence[float], fpr: float) -> float:
    """TPR ≤ min_α (FPR e^{R(α)})^{(α-1)/α}: data processing of the Rényi divergence onto one test."""
    best = min((a - 1) / a * (math.log(fpr) + r) for a, r in zip(ORDERS, rdp, strict=True))
    return min(1.0, math.exp(best))


def clt_mu(sigma: float, steps: int = STEPS_RUN, q: float = Q_RUN) -> float:
    """The Gaussian-DP central-limit approximation μ = q √(T (e^{1/σ²} - 1))."""
    return q * math.sqrt(steps * math.expm1(1 / sigma**2))


def gdp_tpr(mu: float, fpr: float) -> float:
    """Best TPR at a given FPR against μ-GDP: 1 - Φ(Φ^{-1}(1 - FPR) - μ)."""
    if fpr <= 0 or fpr >= 1:
        return min(max(fpr, 0.0), 1.0)
    return normal_sf(normal_quantile(1 - fpr) - mu)


def gdp_delta(eps: float, mu: float) -> float:
    """δ(ε) of μ-GDP."""
    return normal_cdf(mu / 2 - eps / mu) - math.exp(eps) * normal_cdf(-mu / 2 - eps / mu)


def gdp_epsilon(mu: float, delta: float) -> float:
    return bisect_root(lambda e: gdp_delta(e, mu) - delta, 0.0, 100.0)


def gdp_advantage(mu: float) -> float:
    """Largest TPR - FPR against μ-GDP: the total variation distance 2Φ(μ/2) - 1."""
    return 2 * normal_cdf(mu / 2) - 1


# --- The privacy loss distribution of DP-SGD --------------------------------------------------


@dataclass(frozen=True)
class Pld:
    """A discretized privacy loss distribution: mass[i] at loss (offset + i) h, inf_mass at +∞.

    The losses are L = ln(Q(y)/P(y)) distributed under Q, where Q is the output law with the target record
    and P the law without it (the "remove" direction), or the reverse (the "add" direction).
    """

    offset: int
    h: float
    mass: Array
    inf_mass: float
    pessimistic: bool

    def losses(self) -> Array:
        return (self.offset + np.arange(self.mass.size, dtype=np.float64)) * self.h


def _loss_to_y(ell: Array, q: float, sigma: float) -> Array:
    """Inverse of ℓ = ln((1 - q) + q e^{(2y - 1)/(2σ²)}); -∞ where ℓ ≤ ln(1 - q)."""
    out = np.full(ell.shape, -np.inf)
    ok = ell > math.log1p(-q)
    out[ok] = sigma**2 * np.log((np.expm1(ell[ok]) + q) / q) + 0.5
    return out


def _loss_sf_cdf(q: float, sigma: float, direction: str) -> tuple[Callable[[Array], Array], Callable[[Array], Array]]:
    """Survival function and CDF of one step's loss, by inverting the monotone map from output to loss."""
    if direction == "remove":
        # Q = (1 - q) N(0, σ²) + q N(1, σ²), loss increasing in y.
        def sf(ell: Array) -> Array:
            y = _loss_to_y(ell, q, sigma)
            return (1 - q) * _sf_array(y / sigma) + q * _sf_array((y - 1) / sigma)

        def cdf(ell: Array) -> Array:
            y = _loss_to_y(ell, q, sigma)
            return (1 - q) * _cdf_array(y / sigma) + q * _cdf_array((y - 1) / sigma)

        return sf, cdf

    # add: the record is in P, so L = -ln(mixture/normal) under Y ~ N(0, σ²), decreasing in y.
    def sf_add(ell: Array) -> Array:
        return _cdf_array(_loss_to_y(-ell, q, sigma) / sigma)

    def cdf_add(ell: Array) -> Array:
        return _sf_array(_loss_to_y(-ell, q, sigma) / sigma)

    return sf_add, cdf_add


def _loss_range(q: float, sigma: float, direction: str) -> tuple[float, float]:
    sf, cdf = _loss_sf_cdf(q, sigma, direction)

    def log_tail(f: Callable[[Array], Array]) -> Callable[[float], float]:
        return lambda ell: math.log(max(float(f(np.array([ell]))[0]), 1e-300)) - math.log(TAIL)

    if direction == "remove":
        return math.log1p(-q), bisect_root(log_tail(sf), 0.0, 200.0)
    return bisect_root(log_tail(cdf), -200.0, 0.0), -math.log1p(-q)


def discretize_step(q: float, sigma: float, h: float, pessimistic: bool, direction: str) -> Pld:
    """One subsampled-Gaussian step's PLD on the grid {i h}, losses rounded up (pessimistic) or down."""
    sf, cdf = _loss_sf_cdf(q, sigma, direction)
    lo, hi = _loss_range(q, sigma, direction)
    if pessimistic:
        i0, i1 = math.ceil(lo / h), math.ceil(hi / h)
        edges = np.arange(i0 - 1, i1 + 1, dtype=np.float64) * h  # bin i covers (edges[i], edges[i + 1]]
    else:
        i0, i1 = math.floor(lo / h), math.floor(hi / h)
        edges = np.arange(i0, i1 + 2, dtype=np.float64) * h  # bin i covers [edges[i], edges[i + 1])
    s, c = sf(edges), cdf(edges)
    mass = np.maximum(np.where(s[:-1] < 0.5, s[:-1] - s[1:], c[1:] - c[:-1]), 0.0)
    if pessimistic:
        mass[0] += c[0]
        return Pld(i0, h, mass, float(s[-1]), True)
    mass[-1] += s[-1]
    return Pld(i0, h, mass, 0.0, False)


def _truncate(pld: Pld) -> Pld:
    """Cut tails of mass at most TAIL, moving them so that every δ(ε) only rises (pessimistic) or falls."""
    m = np.maximum(pld.mass, 0.0)  # FFT round-off leaves tiny negative values
    low = np.cumsum(m)
    high = np.cumsum(m[::-1])[::-1]
    start = min(int(np.searchsorted(low, TAIL, side="right")), m.size - 1)
    stop = max(m.size - int(np.searchsorted(np.cumsum(m[::-1]), TAIL, side="right")), start + 1)
    kept = m[start:stop].copy()
    bottom = float(low[start - 1]) if start > 0 else 0.0
    top = float(high[stop]) if stop < m.size else 0.0
    if pld.pessimistic:
        kept[0] += bottom
        return Pld(pld.offset + start, pld.h, kept, pld.inf_mass + top, True)
    kept[-1] += top
    return Pld(pld.offset + start, pld.h, kept, pld.inf_mass, False)


def _convolve(a: Pld, b: Pld) -> Pld:
    n = a.mass.size + b.mass.size - 1
    size = 1 << (n - 1).bit_length()
    mass = np.fft.irfft(np.fft.rfft(a.mass, size) * np.fft.rfft(b.mass, size), size)[:n]
    inf_mass = a.inf_mass + b.inf_mass - a.inf_mass * b.inf_mass
    return _truncate(Pld(a.offset + b.offset, a.h, mass, inf_mass, a.pessimistic))


def compose(pld: Pld, k: int) -> Pld:
    """k-fold self-composition: convolution powers by repeated squaring, truncating after each product."""
    result: Pld | None = None
    power = pld
    while k:
        if k & 1:
            result = power if result is None else _convolve(result, power)
        k >>= 1
        if k:
            power = _convolve(power, power)
    assert result is not None
    return result


def pld_delta(pld: Pld, eps: float) -> float:
    """δ(ε) = E_Q[(1 - e^{ε - L})_+] + mass at +∞."""
    ell = pld.losses()
    above = ell > eps
    return float(np.sum(pld.mass[above] * -np.expm1(eps - ell[above]))) + pld.inf_mass


def pld_epsilon(pld: Pld, delta: float) -> float:
    if pld_delta(pld, 0.0) <= delta:
        return 0.0
    return bisect_root(lambda e: pld_delta(pld, e) - delta, 0.0, 200.0, 60)


def pld_roc(pld: Pld) -> tuple[Array, Array]:
    """(FPR, TPR) of the Neyman–Pearson tests "say member when the total loss exceeds t", t on the grid.

    TPR(t) = Q[L > t]; FPR(t) = P[L > t] = E_Q[e^{-L}; L > t] by the change of measure P = e^{-L} Q.
    """
    ell = pld.losses()
    tpr = pld.inf_mass + np.cumsum(pld.mass[::-1])[::-1]
    fpr = np.cumsum((pld.mass * np.exp(-np.minimum(ell, 700.0)))[::-1])[::-1]
    # Thresholds from high to low: FPR and TPR both increase. Start at the test that never says "member".
    fpr_up = np.concatenate([[0.0, 0.0], fpr[::-1], [1.0]])
    tpr_up = np.concatenate([[0.0, pld.inf_mass], tpr[::-1], [1.0]])
    return fpr_up, tpr_up


def pld_tpr(pld: Pld, fpr: float) -> float:
    """Best TPR at the given FPR; randomizing between two thresholds interpolates linearly."""
    f, t = pld_roc(pld)
    return float(np.interp(fpr, f, t))


def pld_advantage(pld: Pld) -> float:
    """Largest TPR - FPR, which is δ(0)."""
    return pld_delta(pld, 0.0)


def pld_loss_tail(pld: Pld, ell: float) -> float:
    """Q[L > ℓ]: the probability, for a member, that the run's total privacy loss exceeds ℓ."""
    return float(np.sum(pld.mass[pld.losses() > ell])) + pld.inf_mass


def pld_loss_quantile(pld: Pld, tail: float) -> float:
    """The smallest grid loss ℓ with Q[L > ℓ] ≤ tail."""
    above = pld.inf_mass + np.cumsum(pld.mass[::-1])[::-1] - pld.mass  # Q[L > ℓ_i]
    return float(pld.losses()[int(np.argmax(above <= tail))])


@cache
def sigma_for(eps: float) -> float:
    """The noise multiplier the RDP accountant requires for ε at δ = 1e-5, q = 0.01, T = 10,000."""
    return noise_multiplier_for(Q_RUN, STEPS_RUN, eps, DELTA_RUN)


def grid_for(sigma: float) -> float:
    """The fine grid where it is cheap (large σ, narrow losses) and the effect of rounding is largest."""
    return H_FINE if sigma >= 2 else H_COARSE


@cache
def dpsgd_pld(sigma: float, pessimistic: bool, direction: str = "remove") -> Pld:
    """The PLD of the whole DP-SGD run, T = 10,000 steps at q = 0.01, on the grid grid_for(σ)."""
    return compose(discretize_step(Q_RUN, sigma, grid_for(sigma), pessimistic, direction), STEPS_RUN)


def pld_epsilon_both(sigma: float, delta: float) -> tuple[float, float]:
    """(upper, lower) bound on the true ε at δ: the larger of the two directions, pessimistic and optimistic."""
    upper = max(pld_epsilon(dpsgd_pld(sigma, True, d), delta) for d in ("remove", "add"))
    lower = pld_epsilon(dpsgd_pld(sigma, False), delta)  # the remove direction alone is already a lower bound
    return upper, lower


# --- Section: the Bayesian reading ----------------------------------------------------------


def posterior(prior: float, loss: float) -> float:
    """Posterior probability of membership when the output's privacy loss is `loss`: odds times e^loss."""
    odds = prior / (1 - prior) * math.exp(loss)
    return odds / (1 + odds)


def flagged_precision(prior: float, tpr: float, fpr: float) -> float:
    """Pr[member | the test says member] = p TPR / (p TPR + (1 - p) FPR)."""
    return prior * tpr / (prior * tpr + (1 - prior) * fpr)


def bad_set_bound(eps: float, delta: float, eps_prime: float) -> float:
    """Probability that an output's loss exceeds ε' > ε under (ε, δ)-DP: δ / (1 - e^{ε - ε'})."""
    return delta / -math.expm1(eps - eps_prime)


# --- Section: reconstruction ----------------------------------------------------------------


def rero_pair_bound(eps: float, delta: float, kappa: float) -> float:
    """Reconstruction success bound from (ε, δ)-DP alone, for baseline success κ: e^ε κ + δ, capped at 1."""
    return min(1.0, math.exp(eps) * kappa + delta)


# --- Section: δ ------------------------------------------------------------------------------


def release_exposure(n: int, delta: float) -> float:
    """Probability that "release each record with probability δ" publishes at least one of n records."""
    return -math.expm1(n * math.log1p(-delta))


# --- Section: deployed values ------------------------------------------------------------------


def zcdp_epsilon(rho: float, delta: float) -> float:
    """ε at δ implied by ρ-zCDP through the simple conversion ρ + 2√(ρ ln(1/δ)) (Bun & Steinke 2016)."""
    return rho + 2 * math.sqrt(rho * math.log(1 / delta))


def rappor_eps_inf(hashes: int, f: float) -> float:
    """RAPPOR's longitudinal bound ε∞ = 2h ln((1 - f/2)/(f/2)) for the memoized permanent response."""
    return 2 * hashes * math.log((1 - f / 2) / (f / 2))


# --- An actual attack on trained models -------------------------------------------------------


def loss_attack_tpr(member_losses: Array, nonmember_losses: Array, fpr: float) -> float:
    """TPR of "member if the loss is below τ", with τ the FPR-quantile of the non-members' losses."""
    tau = float(np.quantile(nonmember_losses, fpr))
    return float(np.mean(member_losses < tau))


@dataclass(frozen=True)
class AttackResult:
    accuracy: float
    tpr: tuple[float, ...]  # at FPRS
    advantage: float


def attack_trained_model(rng: np.random.Generator, sigma: float, nonprivate: bool = False) -> AttackResult:
    """Train the running example once and attack it: training records are members, test records are not."""
    task = make_task()
    cfg = DpsgdConfig(sigma=0.0, clip=math.inf, steps=1000) if nonprivate else DpsgdConfig(sigma=sigma)
    params, _ = train_dpsgd(rng, task.train, cfg)
    members = logistic_losses(params, task.train)
    nonmembers = logistic_losses(params, task.test)
    tprs = tuple(loss_attack_tpr(members, nonmembers, f) for f in FPRS)
    # Advantage of the best threshold on the loss: max over thresholds of TPR - FPR.
    taus = np.quantile(nonmembers, np.linspace(0.001, 0.999, 999))
    adv = max(float(np.mean(members < t)) - float(np.mean(nonmembers < t)) for t in taus)
    return AttackResult(accuracy(params, task.test), tprs, adv)


# --- The chapter's numbers ----------------------------------------------------------------------


def print_table(rng: np.random.Generator) -> None:
    print("== Noise multipliers and accountants (q=0.01, T=10000, delta=1e-5)")
    for e in EPS_GRID:
        s = sigma_for(e)
        upper, lower = pld_epsilon_both(s, DELTA_RUN)
        mu = clt_mu(s)
        print(
            f"eps={e:g}: sigma={s:.3g}  RDP eps={dpsgd_epsilon(Q_RUN, s, STEPS_RUN, DELTA_RUN):.3f}  "
            f"grid h={grid_for(s):g}  PLD eps in [{lower:.3f}, {upper:.3f}]  add direction "
            f"{pld_epsilon(dpsgd_pld(s, True, 'add'), DELTA_RUN):.3f}  CLT mu={mu:.3f}  CLT eps={gdp_epsilon(mu, DELTA_RUN):.3f}"
        )

    print("== Membership: largest TPR at a given FPR")
    for e in EPS_GRID:
        s = sigma_for(e)
        rdp = rdp_total(s)
        mu = clt_mu(s)
        pes, opt = dpsgd_pld(s, True), dpsgd_pld(s, False)
        for f in FPRS:
            print(
                f"eps={e:g} FPR={f:g}: pair bound {pair_tpr_bound(e, DELTA_RUN, f):.4g}  RDP bound "
                f"{rdp_tpr_bound(rdp, f):.4g}  exact in [{pld_tpr(opt, f):.4g}, {pld_tpr(pes, f):.4g}]  "
                f"GDP {gdp_tpr(mu, f):.4g}"
            )
        print(
            f"eps={e:g} advantage: pair bound {pair_advantage_bound(e, DELTA_RUN):.4f}  exact in "
            f"[{pld_advantage(opt):.4f}, {pld_advantage(pes):.4f}]  GDP {gdp_advantage(mu):.4f}"
        )
    print("worked example eps=1, FPR=0.01: e*0.01+1e-5 =", f"{math.e * 0.01 + 1e-5:.5f}")

    print("== Bayes: largest posterior from eps alone")
    for e in EPS_GRID:
        print(f"eps={e:g}: " + "  ".join(f"prior {p:g} -> {posterior(p, e):.4f}" for p in PRIORS))
    print("== Bayes: the member's total privacy loss in the actual run (pessimistic PLD)")
    for e in EPS_GRID:
        pes = dpsgd_pld(sigma_for(e), True)
        median = pld_loss_quantile(pes, 0.5)
        q3 = pld_loss_quantile(pes, 1e-3)
        q5 = pld_loss_quantile(pes, 1e-5)
        print(
            f"eps={e:g}: loss median {median:.3f}  exceeded w.p. 1e-3: {q3:.3f}  w.p. 1e-5: {q5:.3f}  "
            f"P[L > eps]={pld_loss_tail(pes, e):.2e}  posterior at prior 0.01: median {posterior(0.01, median):.4f}  "
            f"1e-3 {posterior(0.01, q3):.4f}  1e-5 {posterior(0.01, q5):.4f}"
        )
    for e in (1.0, 8.0):
        print(
            f"eps={e:g}: outputs with loss above 2 eps have probability at most {bad_set_bound(e, DELTA_RUN, 2 * e):.3g}"
        )
    print("== Bayes: precision of a flagged member (prior 0.01)")
    for e in EPS_GRID:
        pes = dpsgd_pld(sigma_for(e), True)
        parts: list[str] = []
        for f in FPRS:
            pair = flagged_precision(0.01, pair_tpr_bound(e, DELTA_RUN, f), f)
            exact = flagged_precision(0.01, pld_tpr(pes, f), f)
            parts.append(f"FPR {f:g}: pair {pair:.4f} exact {exact:.4f}")
        print(f"eps={e:g}: " + "  ".join(parts))

    print("== Reconstruction: success bound for a uniform prior over m candidates (kappa = 1/m)")
    for e in EPS_GRID:
        pes = dpsgd_pld(sigma_for(e), True)
        parts = [
            f"m={round(1 / k)}: pair {rero_pair_bound(e, DELTA_RUN, k):.4g} curve {pld_tpr(pes, k):.4g}" for k in FPRS
        ]
        print(f"eps={e:g}: " + "  ".join(parts))

    print("== delta")
    for d in (1e-4, 1e-5, 1e-6):
        print(
            f"n=10000, delta={d:g}: n*delta={N_RECORDS * d:g}  P[at least one record released]={release_exposure(N_RECORDS, d):.4g}"
        )
    for e in (1.0, 4.0, 8.0):
        s = sigma_for(e)
        parts = [f"delta={d:g}: {dpsgd_epsilon(Q_RUN, s, STEPS_RUN, d):.3f}" for d in (1e-4, 1e-5, 1e-6, 1e-8, 1e-10)]
        print(f"sigma={s:.3g} (eps={e:g} at 1e-5), RDP eps at " + "  ".join(parts))

    print("== Deployed values: conversions")
    for rho in (2.56, 2.63, 0.81, 0.25, 1.86):
        print(f"rho={rho:g}: eps at delta=1e-10 by rho + 2 sqrt(rho ln(1/delta)) = {zcdp_epsilon(rho, 1e-10):.3f}")
    for h, f in ((2, 0.75), (2, 0.5)):
        print(f"RAPPOR h={h}, f={f:g}: eps_inf={rappor_eps_inf(h, f):.3f}")
    print(
        f"a user-day budget of 1 over a year: {365 * 1.0:g};  Apple QuickType 2 x 8 per day over 90 days: {2 * 8 * 90}"
    )

    print("== A loss-threshold attack on models trained at each budget (one run each)")
    for e in EPS_GRID:
        r = attack_trained_model(rng, sigma_for(e))
        tprs = "  ".join(f"TPR@{f:g}={t:.4f}" for f, t in zip(FPRS, r.tpr, strict=True))
        print(f"eps={e:g}: accuracy {r.accuracy:.4f}  {tprs}  advantage {r.advantage:.4f}")
    r = attack_trained_model(rng, 0.0, nonprivate=True)
    tprs = "  ".join(f"TPR@{f:g}={t:.4f}" for f, t in zip(FPRS, r.tpr, strict=True))
    print(f"non-private: accuracy {r.accuracy:.4f}  {tprs}  advantage {r.advantage:.4f}")

    print("== Accuracy quoted from the DP-SGD chapter")
    for e, a in ACCURACY_CH37.items():
        print(f"eps={e:g}: accuracy {a:.4f}  gap to non-private {ACCURACY_NONPRIVATE_CH37 - a:.4f}")


# --- Figures --------------------------------------------------------------------------------

COLORS: dict[float, str] = {
    0.5: "tab:blue",
    1.0: "tab:cyan",
    2.0: "tab:green",
    4.0: "tab:orange",
    8.0: "tab:red",
    16.0: "tab:purple",
}


def plot_tradeoff() -> Figure:
    fprs = np.logspace(-6, 0, 200)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    for ax, e in zip(axes, (1.0, 8.0), strict=True):
        s = sigma_for(e)
        rdp = rdp_total(s)
        mu = clt_mu(s)
        f_pes, t_pes = pld_roc(dpsgd_pld(s, True))
        ax.loglog(
            fprs,
            [pair_tpr_bound(e, DELTA_RUN, float(f)) for f in fprs],
            color="black",
            ls=":",
            label=r"$(\varepsilon, \delta)$ bound",
        )
        ax.loglog(fprs, [rdp_tpr_bound(rdp, float(f)) for f in fprs], color="tab:green", ls="-.", label="RDP bound")
        ax.loglog(fprs, np.interp(fprs, f_pes, t_pes), color="tab:red", lw=2, label="exact curve (PLD)")
        ax.loglog(
            fprs, [gdp_tpr(mu, float(f)) for f in fprs], color="tab:blue", ls="--", label=f"GDP, $\\mu = {mu:.2f}$"
        )
        ax.loglog(fprs, fprs, color="tab:gray", lw=0.8, label="no information")
        ax.set_xlim(1e-6, 1)
        ax.set_ylim(1e-6, 1.2)
        ax.set_xlabel("false positive rate")
        ax.set_ylabel("largest true positive rate")
        ax.set_title(f"DP-SGD at $\\varepsilon = {e:g}$, $\\delta = 10^{{-5}}$ ($\\sigma = {s:.3g}$)")
        ax.legend(fontsize=8, loc="lower right")
    return fig


def plot_curves_all() -> Figure:
    fprs = np.logspace(-6, 0, 200)
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    for e in EPS_GRID:
        f_pes, t_pes = pld_roc(dpsgd_pld(sigma_for(e), True))
        ax.loglog(fprs, np.interp(fprs, f_pes, t_pes), color=COLORS[e], label=f"$\\varepsilon = {e:g}$")
    ax.loglog(fprs, fprs, color="tab:gray", lw=0.8, ls=":", label="no information")
    ax.set_xlim(1e-6, 1)
    ax.set_ylim(1e-6, 1.2)
    ax.set_xlabel("false positive rate")
    ax.set_ylabel("largest true positive rate")
    ax.set_title("Exact trade-off curves of DP-SGD, $q = 0.01$, $T = 10{,}000$")
    ax.legend(fontsize=8, loc="lower right")
    return fig


def plot_posterior() -> Figure:
    priors = np.logspace(-4, math.log10(0.5), 200)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    for e in EPS_GRID:
        axes[0].loglog(
            priors, [posterior(float(p), e) for p in priors], color=COLORS[e], label=f"$\\varepsilon = {e:g}$"
        )
    axes[0].loglog(priors, priors, color="tab:gray", ls=":", lw=0.8, label="no change")
    axes[0].set_xlabel("prior probability of membership")
    axes[0].set_ylabel("largest posterior, from $\\varepsilon$ alone")
    axes[0].set_ylim(1e-4, 1.5)
    axes[0].legend(fontsize=8, loc="lower right")
    e = 8.0
    pes = dpsgd_pld(sigma_for(e), True)
    axes[1].loglog(
        priors, [posterior(float(p), e) for p in priors], color="black", ls=":", label=r"bound from $\varepsilon = 8$"
    )
    for tail, ls, label in (
        (1e-5, "-", "loss exceeded w.p. $10^{-5}$"),
        (1e-3, "--", "loss exceeded w.p. $10^{-3}$"),
        (0.5, "-.", "median loss"),
    ):
        ell = pld_loss_quantile(pes, tail)
        axes[1].loglog(
            priors, [posterior(float(p), ell) for p in priors], color="tab:red", ls=ls, label=f"{label} ({ell:.2f})"
        )
    axes[1].loglog(priors, priors, color="tab:gray", ls=":", lw=0.8)
    axes[1].set_xlabel("prior probability of membership")
    axes[1].set_ylabel("posterior of a member")
    axes[1].set_ylim(1e-4, 1.5)
    axes[1].set_title(r"DP-SGD at $\varepsilon = 8$: the actual privacy loss")
    axes[1].legend(fontsize=8, loc="lower right")
    return fig


def plot_accuracy_risk() -> Figure:
    eps_list = sorted(ACCURACY_CH37)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.0))
    accs = [ACCURACY_CH37[e] for e in eps_list]
    axes[0].semilogx(eps_list, accs, marker="o", color="black", label="DP-SGD (chapter 37)")
    axes[0].axhline(ACCURACY_NONPRIVATE_CH37, color="tab:gray", ls="--", label="non-private")
    axes[0].set_xticks(eps_list, [f"{e:g}" for e in eps_list])
    axes[0].minorticks_off()
    axes[0].set_xlabel(r"$\varepsilon$ at $\delta = 10^{-5}$")
    axes[0].set_ylabel("test accuracy")
    axes[0].legend(fontsize=8)
    fpr = 1e-2
    exact = [pld_tpr(dpsgd_pld(sigma_for(e), True), fpr) for e in eps_list]
    pair = [pair_tpr_bound(e, DELTA_RUN, fpr) for e in eps_list]
    axes[1].semilogx(exact, accs, marker="o", color="tab:red", label="exact curve")
    axes[1].semilogx(pair, accs, marker="s", color="black", ls=":", label=r"$(\varepsilon, \delta)$ bound")
    for e, x, a in zip(eps_list, exact, accs, strict=True):
        axes[1].annotate(f"{e:g}", (x, a), textcoords="offset points", xytext=(-14, 2), fontsize=8, color="tab:red")
    axes[1].axvline(fpr, color="tab:gray", lw=0.8, ls=":")
    axes[1].set_xlabel("largest TPR of a membership test at FPR 1%")
    axes[1].set_ylabel("test accuracy")
    axes[1].legend(fontsize=8, loc="lower right")
    return fig


def plot_delta_profile() -> Figure:
    deltas = np.logspace(-12, -2, 41)
    fig, ax = plt.subplots(figsize=(6.5, 4.0))
    for e in (1.0, 4.0, 8.0):
        s = sigma_for(e)
        ax.semilogx(
            deltas,
            [dpsgd_epsilon(Q_RUN, s, STEPS_RUN, float(d)) for d in deltas],
            color=COLORS[e],
            label=f"$\\sigma = {s:.3g}$ ($\\varepsilon = {e:g}$ at $10^{{-5}}$)",
        )
    ax.axvline(1 / N_RECORDS, color="tab:gray", ls="--", lw=0.8)
    ax.annotate("$1/n$", (1 / N_RECORDS, 2.5), textcoords="offset points", xytext=(4, 0), fontsize=9)
    ax.set_xlabel(r"$\delta$")
    ax.set_ylabel(r"$\varepsilon$ (RDP accountant)")
    ax.set_title("The privacy profile of three DP-SGD runs")
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
    figures = {
        "epsc_tradeoff.svg": plot_tradeoff(),
        "epsc_curves.svg": plot_curves_all(),
        "epsc_posterior.svg": plot_posterior(),
        "epsc_accuracy_risk.svg": plot_accuracy_risk(),
        "epsc_delta_profile.svg": plot_delta_profile(),
    }
    for name, fig in figures.items():
        fig.tight_layout()
        fig.savefig(out / name, metadata={"Date": None})  # no timestamp, so output is reproducible
        plt.close(fig)
        print(f"saved {out / name}")


if __name__ == "__main__":
    main()
