"""Check the filters, odometers and individual accounting of the adaptive-composition chapter.

    uv run python code/part04-privacy-accounting/adaptive_composition.py
    uv run python code/part04-privacy-accounting/adaptive_composition.py --seed 1 --out /tmp/figures

Prints the numbers quoted in 08-adaptive-composition.md (they match exactly with the default
seed), then saves the chapter's figures.

Conventions: P = M(D) and Q = M(D') are the output distributions on neighbouring datasets, the
privacy loss is L = ln(p/q) evaluated at an output drawn from P, logs are natural. Per-step
privacy parameters are written eps_i (pure DP) or rho_i = eps_i(alpha) (RDP at a fixed order).
"""

import argparse
import math
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

import matplotlib

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # for code/shared
from shared.figures import configure
from shared.stats import normal_cdf

configure("adaptive_composition")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from numpy.typing import NDArray
from subsampling import subsampled_gaussian_rdp_curve

matplotlib.rcParams["svg.hashsalt"] = "adaptive_composition"  # the import above sets its own salt

# Figures are build output: _build/figures/<part>/, which the build copies into each book.
FIGURES = Path(__file__).resolve().parents[2] / "_build" / "figures" / Path(__file__).parent.name

FloatArray = NDArray[np.float64]
BoolArray = NDArray[np.bool_]

# The RDP orders shared by the whole part, so optimized numbers agree across chapters.
ORDERS: list[float] = [1 + x / 10 for x in range(1, 100)] + [float(x) for x in range(12, 64)] + [128.0, 256.0, 512.0]


# --- Toolkit: from a Rényi bound to a tail bound ----------------------------------------------


def rdp_to_dp(rdp: float, alpha: float, delta: float) -> float:
    """Mironov's conversion at one order: (alpha, rdp)-RDP gives (rdp + ln(1/δ)/(α-1), δ)-DP."""
    return rdp + math.log(1 / delta) / (alpha - 1)


def best_dp(curve: Callable[[float], float], delta: float, orders: Sequence[float] = ORDERS) -> tuple[float, float]:
    """min over the order grid of rdp_to_dp; returns (epsilon, best alpha)."""
    return min((rdp_to_dp(curve(a), a, delta), a) for a in orders)


def zcdp_to_dp(rho: float, delta: float) -> float:
    """ρ-zCDP gives (ρ + 2 sqrt(ρ ln(1/δ)), δ)-DP: Mironov's conversion optimized over a continuous α."""
    return rho + 2 * math.sqrt(rho * math.log(1 / delta))


def zcdp_budget_for(eps_g: float, delta: float) -> float:
    """The largest ρ with zcdp_to_dp(ρ, δ) <= eps_g: ρ* = (sqrt(ln(1/δ) + eps_g) - sqrt(ln(1/δ)))²."""
    ell = math.log(1 / delta)
    return (math.sqrt(ell + eps_g) - math.sqrt(ell)) ** 2


# --- Section 34.3: why realized parameters break advanced composition -------------------------------


def rr_keep_probability(eps: float) -> float:
    """Randomized response with parameter eps reports the true bit with probability e^eps/(1+e^eps)."""
    return math.exp(eps) / (1 + math.exp(eps))


def rr_loss_mean(eps: float) -> float:
    """E_P[L] for one randomized-response step: eps(e^eps - 1)/(e^eps + 1) = eps tanh(eps/2)."""
    return eps * math.tanh(eps / 2)


def advanced_composition_epsilon(eps_spent: Sequence[float], delta: float) -> float:
    """Advanced composition (Dwork-Rothblum-Vadhan form) for parameters FIXED in advance:
    sqrt(2 ln(1/δ) Σ eps_i²) + Σ eps_i (e^{eps_i} - 1)."""
    s2 = sum(e * e for e in eps_spent)
    return math.sqrt(2 * math.log(1 / delta) * s2) + sum(e * math.expm1(e) for e in eps_spent)


def plugin_rr_bound(eps: float, t: int, delta: float) -> float:
    """The fixed-t Azuma bound on the loss of t randomized-response steps:
    t·eps·tanh(eps/2) + eps·sqrt(2 t ln(1/δ)). Holds with probability 1 - δ at each FIXED t."""
    return t * rr_loss_mean(eps) + eps * math.sqrt(2 * t * math.log(1 / delta))


def plugin_crossing_probability(eps: float, delta: float, horizons: Sequence[int]) -> dict[int, float]:
    """Exact P_P(∃ t <= n: L_t > plugin_rr_bound(eps, t, δ)) for each n in horizons.

    The loss of t randomized-response steps is eps·j with j = (#truthful - #flipped), a ±1 walk.
    Dynamic programming over j, removing the mass that has crossed, gives the exact probability."""
    n = max(horizons)
    p = rr_keep_probability(eps)
    j = np.arange(-n, n + 1, dtype=np.float64)
    alive = np.zeros(2 * n + 1)
    alive[n] = 1.0
    crossed = 0.0
    out: dict[int, float] = {}
    for t in range(1, n + 1):
        nxt = np.zeros_like(alive)
        nxt[1:] += p * alive[:-1]
        nxt[:-1] += (1 - p) * alive[1:]
        over = eps * j > plugin_rr_bound(eps, t, delta)
        crossed += float(nxt[over].sum())
        nxt[over] = 0.0
        alive = nxt
        if t in horizons:
            out[t] = crossed
    return out


def fixed_time_exceed_probability(eps: float, delta: float, t: int) -> float:
    """Exact P_P(L_t > plugin_rr_bound(eps, t, δ)) at one fixed t (binomial tail)."""
    p = rr_keep_probability(eps)
    bound = plugin_rr_bound(eps, t, delta)
    total = 0.0
    for up in range(t + 1):
        if eps * (2 * up - t) > bound:
            total += math.exp(
                math.lgamma(t + 1)
                - math.lgamma(up + 1)
                - math.lgamma(t - up + 1)
                + up * math.log(p)
                + (t - up) * math.log(1 - p)
            )
    return total


def worst_fixed_time(eps: float, delta: float, horizon: int) -> tuple[float, int]:
    """The largest fixed-t exceedance probability over every t <= horizon, and the t that attains it."""
    return max((fixed_time_exceed_probability(eps, delta, t), t) for t in range(1, horizon + 1))


def stop_when_ahead_paths(
    rng: np.random.Generator, eps: float, delta: float, n: int, paths: int
) -> tuple[FloatArray, BoolArray]:
    """Loss paths L_t/eps of randomized response, frozen once they cross the plug-in bound.
    Returns the paths and which of them crossed."""
    p = rr_keep_probability(eps)
    walk = np.cumsum(np.where(rng.random((paths, n)) < p, 1.0, -1.0), axis=1)
    bound = np.array([plugin_rr_bound(eps, t, delta) / eps for t in range(1, n + 1)])
    crossed = np.zeros(paths, dtype=bool)
    for i in range(paths):
        hit = np.nonzero(walk[i] > bound)[0]
        if hit.size:
            walk[i, hit[0] + 1 :] = walk[i, hit[0]]
            crossed[i] = True
    return walk, crossed


# --- Section 34.4: filters for pure differential privacy -------------------------------------------


def basic_filter_ok(eps_spent: Sequence[float], budget: float) -> bool:
    """Basic-composition filter: continue while Σ eps_i <= budget (valid for pure DP steps)."""
    return sum(eps_spent) <= budget


def _rogers_k(v: float, drift: float, eps_g: float, delta_g: float) -> float:
    x = eps_g**2 / (28.04 * math.log(1 / delta_g))
    return drift + math.sqrt(2 * (v + x) * (1 + 0.5 * math.log(v / x + 1)) * math.log(1 / delta_g))


def rogers_filter_statistic(eps_spent: Sequence[float], eps_g: float, delta_g: float) -> float:
    """K of Rogers, Roth, Ullman & Vadhan (2016), Theorem 5.1, for pure DP steps:
    Σ eps_j(e^{eps_j}-1)/2 + sqrt(2 (V + x)(1 + ½ ln(V/x + 1)) ln(1/δ_g)),
    with V = Σ eps_i² and x = eps_g² / (28.04 ln(1/δ_g))."""
    v = sum(e * e for e in eps_spent)
    drift = sum(e * math.expm1(e) / 2 for e in eps_spent)
    return _rogers_k(v, drift, eps_g, delta_g)


def rogers_filter_ok(eps_spent: Sequence[float], eps_g: float, delta_g: float) -> bool:
    """Rogers et al.'s advanced-composition filter: continue while K <= eps_g (needs δ_g < 1/e)."""
    return rogers_filter_statistic(eps_spent, eps_g, delta_g) <= eps_g


def rogers_filter_budget(eps_spent: Sequence[float], delta_g: float) -> float:
    """The smallest global budget eps_g whose Rogers filter still lets eps_spent through."""
    v = sum(e * e for e in eps_spent)
    drift = sum(e * math.expm1(e) / 2 for e in eps_spent)

    def ok(g: float) -> bool:
        return _rogers_k(v, drift, g, delta_g) <= g

    grid = [10 ** (i / 200) for i in range(-1200, 1000)]
    hi = next(g for g in grid if ok(g))
    lo = hi / 10 ** (1 / 200)
    for _ in range(100):
        mid = (lo + hi) / 2
        if ok(mid):
            hi = mid
        else:
            lo = mid
    return hi


def whitehouse_filter_epsilon(eps_spent: Sequence[float], delta: float) -> float:
    """The (eps, δ)-DP filter of Whitehouse, Ramdas, Rogers & Wu (2023), Theorem 2, for pure DP steps:
    the run may continue while sqrt(2 ln(1/δ) Σ eps_i²) + ½ Σ eps_i² <= eps_g."""
    v = sum(e * e for e in eps_spent)
    return math.sqrt(2 * math.log(1 / delta) * v) + v / 2


def whitehouse_filter_ok(eps_spent: Sequence[float], eps_g: float, delta: float) -> bool:
    return whitehouse_filter_epsilon(eps_spent, delta) <= eps_g


def max_steps(ok: Callable[[Sequence[float]], bool], eps0: float, cap: int = 1_000_000) -> int:
    """How many steps of eps0 a filter lets through (galloping, then bisection on k)."""
    lo, hi = 0, 1
    while hi <= cap and ok([eps0] * hi):
        lo, hi = hi, 2 * hi
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if ok([eps0] * mid):
            lo = mid
        else:
            hi = mid
    return lo


# --- Section 34.5: the Rényi filter -----------------------------------------------------------------


def renyi_filter_ok(rdp_spent: Sequence[float], budget: float) -> bool:
    """Rényi filter at a fixed order α: continue while Σ eps_i(α) <= budget."""
    return sum(rdp_spent) <= budget


def rr_rdp(alpha: float, eps: float) -> float:
    """D_α between the output laws of randomized response on bit 1 and on bit 0."""
    p = rr_keep_probability(eps)
    return math.log(p**alpha * (1 - p) ** (1 - alpha) + (1 - p) ** alpha * p ** (1 - alpha)) / (alpha - 1)


@dataclass
class PathRow:
    y1: int
    y2: int
    p: float
    q: float
    loss: float
    spent: float
    martingale: float


def two_step_rr(alpha: float, eps_first: float, eps_if_one: float, eps_if_zero: float) -> list[PathRow]:
    """Randomized response on one person's bit, twice; the second eps depends on the first output.

    P: the bit is 1 (dataset D); Q: the bit is 0 (dataset D'). For every path, returns the
    probabilities, the loss L_2, the RDP spent along it and N_2 = exp((α-1)(L_2 - spent))."""
    rows: list[PathRow] = []
    for y1 in (1, 0):
        p1 = rr_keep_probability(eps_first)
        eps2 = eps_if_one if y1 == 1 else eps_if_zero
        p2 = rr_keep_probability(eps2)
        for y2 in (1, 0):
            pp = (p1 if y1 else 1 - p1) * (p2 if y2 else 1 - p2)
            qq = (1 - p1 if y1 else p1) * (1 - p2 if y2 else p2)
            loss = math.log(pp / qq)
            spent = rr_rdp(alpha, eps_first) + rr_rdp(alpha, eps2)
            rows.append(PathRow(y1, y2, pp, qq, loss, spent, math.exp((alpha - 1) * (loss - spent))))
    return rows


def joint_renyi(rows: Sequence[PathRow], alpha: float) -> float:
    """D_α(P‖Q) of the whole two-step transcript."""
    return math.log(sum(r.p**alpha * r.q ** (1 - alpha) for r in rows)) / (alpha - 1)


@dataclass
class TranscriptRow:
    ys: tuple[int, ...]
    p: float
    q: float
    loss: float
    spent: float
    martingale: float


def three_step_rr(alpha: float, eps_first: float, strong: float, weak: float) -> list[TranscriptRow]:
    """Randomized response on one person's bit, three times, with the order of the last two steps
    chosen by the first output: strong then weak after y1 = 1, weak then strong after y1 = 0.

    Every path spends the same RDP, eps(eps_first) + eps(strong) + eps(weak), while each of steps 2
    and 3 is strong on some branch. P: the bit is 1; Q: the bit is 0. Returns, for each of the eight
    transcripts, P, Q, the loss L_3, the RDP spent and N_3 = exp((α-1)(L_3 - spent))."""
    rows: list[TranscriptRow] = []
    for y1 in (1, 0):
        schedule = (eps_first, strong, weak) if y1 == 1 else (eps_first, weak, strong)
        for y2 in (1, 0):
            for y3 in (1, 0):
                ys = (y1, y2, y3)
                pp = qq = 1.0
                for y, e in zip(ys, schedule, strict=True):
                    keep = rr_keep_probability(e)
                    pp *= keep if y else 1 - keep
                    qq *= 1 - keep if y else keep
                loss = math.log(pp / qq)
                spent = sum(rr_rdp(alpha, e) for e in schedule)
                rows.append(TranscriptRow(ys, pp, qq, loss, spent, math.exp((alpha - 1) * (loss - spent))))
    return rows


def transcript_renyi(rows: Sequence[TranscriptRow], alpha: float) -> float:
    """D_α(P‖Q) of a transcript given by its rows."""
    return math.log(sum(r.p**alpha * r.q ** (1 - alpha) for r in rows)) / (alpha - 1)


def adaptive_gaussian_filter(
    rng: np.random.Generator, alpha: float, budget: float, paths: int, top_up: bool
) -> tuple[FloatArray, FloatArray, FloatArray]:
    """Adaptive Gaussian steps under a Rényi filter, simulated under P.

    Step t releases N(mu_t, 1) on D and N(0, 1) on D' (sensitivity 1, noise 1), with
    mu_t = 0.3 if the previous output was positive and 0.6 otherwise, so eps_t(α) = α mu_t²/2
    depends on the past. The filter stops a path before a step would exceed the budget; with
    top_up the last step instead uses the mu that spends the remaining budget exactly.
    Returns (loss L_T, RDP spent, number of steps) per path."""
    loss = np.zeros(paths)
    spent = np.zeros(paths)
    steps = np.zeros(paths)
    prev = np.zeros(paths)
    active = np.ones(paths, dtype=bool)
    while active.any():
        mu = np.where(prev > 0, 0.3, 0.6)
        fits = spent + alpha * mu**2 / 2 <= budget + 1e-12
        if top_up:  # a step that would not fit is shrunk to spend exactly what is left
            mu = np.where(fits, mu, np.sqrt(2 * np.maximum(budget - spent, 0.0) / alpha))
            run = active
        else:
            run = active & fits
        y = mu + rng.standard_normal(paths)
        loss = np.where(run, loss + mu * y - mu**2 / 2, loss)
        spent = np.where(run, spent + alpha * mu**2 / 2, spent)
        steps = np.where(run, steps + 1, steps)
        prev = np.where(run, y, prev)
        active = active & fits  # a path whose next step did not fit has now stopped
    return loss, spent, steps


def mc_renyi_from_losses(losses: FloatArray, alpha: float) -> float:
    """D_α(P‖Q) = ln E_P[e^{(α-1)L}]/(α-1), estimated from losses drawn under P."""
    a = (alpha - 1) * losses
    top = float(a.max())
    return (top + math.log(float(np.mean(np.exp(a - top))))) / (alpha - 1)


def supermartingale_paths(
    rng: np.random.Generator, alpha: float, budget: float, paths: int, horizon: int
) -> FloatArray:
    """ln N_t = (α-1)(L_t - Σ eps_j(α)) along the adaptive Gaussian filter, frozen after it stops."""
    out = np.zeros((paths, horizon + 1))
    loss = np.zeros(paths)
    spent = np.zeros(paths)
    prev = np.zeros(paths)
    active = np.ones(paths, dtype=bool)
    for t in range(1, horizon + 1):
        mu = np.where(prev > 0, 0.3, 0.6)
        rho = alpha * mu**2 / 2
        active = active & (spent + rho <= budget + 1e-12)
        y = mu + rng.standard_normal(paths)
        loss = np.where(active, loss + mu * y - mu**2 / 2, loss)
        spent = np.where(active, spent + rho, spent)
        prev = np.where(active, y, prev)
        out[:, t] = (alpha - 1) * (loss - spent)
    return out


# --- Section 34.6: odometers -------------------------------------------------------------------------


def ville_odometer(spent: Callable[[float], float], delta: float, orders: Sequence[float] = ORDERS) -> float:
    """Time-uniform bound from Ville's inequality and a union over the orders:
    min_α [Σ eps_j(α) + ln(|Λ|/δ)/(α-1)]."""
    return min(spent(a) + math.log(len(orders) / delta) / (a - 1) for a in orders)


def lecuyer_odometer(spent: Callable[[float], float], delta: float, orders: Sequence[float] = ORDERS) -> float:
    """Lécuyer (2021) odometer over the Rényi curve: filters with budgets
    eps^f(α) = 2^{f-1} ln(2|Λ|/δ)/(α-1); with f the first filter that covers the spend,
    the bound is eps^f(α) + ln(2|Λ| f²/δ)/(α-1), minimized over α."""
    best = math.inf
    for a in orders:
        base = math.log(2 * len(orders) / delta) / (a - 1)
        f = 1
        while 2 ** (f - 1) * base < spent(a):
            f += 1
        best = min(best, 2 ** (f - 1) * base + math.log(2 * len(orders) * f * f / delta) / (a - 1))
    return best


def fixed_time_conversion(spent: Callable[[float], float], delta: float) -> float:
    """Mironov's conversion at a time fixed in advance (not time-uniform)."""
    return best_dp(spent, delta)[0]


def gaussian_steps_rdp(sigma: float, t: int) -> Callable[[float], float]:
    """RDP curve of t full-batch Gaussian steps, sensitivity 1, noise multiplier sigma."""
    return lambda a: a * t / (2 * sigma**2)


def ville_crossing_mc(rng: np.random.Generator, mu: float, alpha: float, delta: float, n: int, paths: int) -> float:
    """MC estimate of P_P(∃ t <= n: L_t > t·α mu²/2 + ln(1/δ)/(α-1)) for Gaussian steps of shift mu."""
    hits = 0
    chunk = 5_000
    t = np.arange(1, n + 1, dtype=np.float64)
    bound = t * alpha * mu**2 / 2 + math.log(1 / delta) / (alpha - 1)
    for start in range(0, paths, chunk):
        m = min(chunk, paths - start)
        step = mu * (mu + rng.standard_normal((m, n))) - mu**2 / 2
        loss = np.cumsum(step, axis=1)
        hits += int(np.count_nonzero((loss > bound).any(axis=1)))
    return hits / paths


def brownian_line_crossing(a: float, b: float, horizon: float) -> float:
    """P(∃ s <= S: B_s >= a + b s) for standard Brownian motion (a, b > 0)."""
    root = math.sqrt(horizon)
    return normal_cdf((-a - b * horizon) / root) + math.exp(-2 * a * b) * normal_cdf((-a + b * horizon) / root)


# --- Section 34.7: individual accounting in full-batch DP-GD -----------------------------------------


def gaussian_individual_rdp(alpha: float, grad_norms: Sequence[float], sigma: float, clip: float) -> float:
    """Individual RDP of one record over the steps where it took part: the sum of clipped
    gradients gets N(0, σ²C² I), and removing the record moves the sum by its clipped gradient,
    so each step costs α min(‖g_t‖, C)²/(2σ²C²)."""
    return sum(alpha * min(g, clip) ** 2 / (2 * sigma**2 * clip**2) for g in grad_norms)


@dataclass
class Population:
    norms: FloatArray  # (n, k) clipped gradient norms in units of the clip C
    hard: BoolArray  # (n,) True for the minority whose gradients stay large


def synthetic_gradient_norms(
    rng: np.random.Generator, n: int, k: int, frac_hard: float = 0.1, tau: float = 40.0, jitter: float = 0.25
) -> Population:
    """The explicit model of the simulation, with clip C = 1.

    Typical records (1 - frac_hard): ‖g_{i,t}‖ = s_i e^{-t/τ} e^{jitter·Z}, s_i ~ U(0.3, 2): they are
    fitted as training proceeds and their gradients shrink. Hard records (frac_hard):
    ‖g_{i,t}‖ = s_i e^{jitter·Z}, s_i ~ U(0.7, 1.5): they are never fitted. Norms are then clipped at 1."""
    hard = rng.random(n) < frac_hard
    scale = np.where(hard, rng.uniform(0.7, 1.5, n), rng.uniform(0.3, 2.0, n))
    t = np.arange(k, dtype=np.float64)
    decay = np.where(hard[:, None], 1.0, np.exp(-t[None, :] / tau))
    raw = scale[:, None] * decay * np.exp(jitter * rng.standard_normal((n, k)))
    return Population(np.minimum(raw, 1.0), hard)


def individual_filter(norms: FloatArray, alpha: float, sigma: float, budget: float) -> tuple[FloatArray, BoolArray]:
    """Feldman & Zrnic's individual filter at order α (clip C = 1): at step t record i takes part
    only if its spend so far plus this step's α‖g‖²/(2σ²) stays <= budget.
    Returns the cumulative spend (n, k) and the participation mask (n, k)."""
    n, k = norms.shape
    spent = np.zeros(n)
    cum = np.zeros((n, k))
    active = np.zeros((n, k), dtype=bool)
    for t in range(k):
        rho = alpha * norms[:, t] ** 2 / (2 * sigma**2)
        ok = spent + rho <= budget + 1e-12
        spent = np.where(ok, spent + rho, spent)
        cum[:, t] = spent
        active[:, t] = ok
    return cum, active


def individual_epsilons(norms: FloatArray, sigma: float, delta: float, alpha: float) -> tuple[FloatArray, FloatArray]:
    """Per-record epsilon of an unfiltered run: at the fixed order alpha, and optimized over ORDERS.
    The RDP curve of record i is α S_i/(2σ²) with S_i = Σ_t ‖g_{i,t}‖² (clip 1)."""
    s = (norms**2).sum(axis=1) / (2 * sigma**2)  # zCDP-style slope per record
    fixed = alpha * s + math.log(1 / delta) / (alpha - 1)
    best = np.array([min(a * float(si) + math.log(1 / delta) / (a - 1) for a in ORDERS) for si in s])
    return fixed, best


# --- The chapter's numbers ------------------------------------------------------------------

# Section 34.7 setting: n records, noise multiplier sigma, worst-case budget = k0 full-batch steps.
N_RECORDS, SIGMA, K0, K_MAX, DELTA = 10_000, 10.0, 100, 300, 1e-5


def part5_setting() -> tuple[float, float, float]:
    """(alpha*, budget at alpha*, worst-case epsilon) for K0 worst-case steps at DELTA."""
    eps_wc, alpha = best_dp(gaussian_steps_rdp(SIGMA, K0), DELTA)
    return alpha, gaussian_steps_rdp(SIGMA, K0)(alpha), eps_wc


def print_table(rng: np.random.Generator) -> None:
    print("[Part 1: plug-in advanced composition under stop-when-ahead, randomized response eps=0.1]")
    eps = 0.1
    print(f"  per-step mean loss eps*tanh(eps/2)={rr_loss_mean(eps):.6f}")
    horizons = [10, 100, 1000, 10000]
    for delta in (0.05, 0.01, 1e-5):

        def fmt(x: float, delta: float = delta) -> str:
            return f"{x:.4f}" if delta >= 1e-3 else f"{x:.2e}"

        cross = plugin_crossing_probability(eps, delta, horizons)
        fixed, t_worst = worst_fixed_time(eps, delta, 1000)
        print(
            f"  delta={delta:g}: P(cross by n): "
            + "  ".join(f"n={n}: {fmt(cross[n])}" for n in horizons)
            + f"  | max fixed-t P over t<=1000: {fmt(fixed)} at t={t_worst}"
            + f"  | cross by 10^4 / delta={cross[10000] / delta:.2f}"
        )
    print(f"  bound at t=100, delta=0.01: {plugin_rr_bound(eps, 100, 0.01):.3f}")

    print("[Part 2: budgets needed for k steps of eps0, delta=1e-6]")
    delta = 1e-6
    for eps0, k in ((0.01, 100), (0.01, 1000), (0.01, 10000), (0.1, 100), (0.1, 1000)):
        seq = [eps0] * k
        ac = advanced_composition_epsilon(seq, delta)
        ro = rogers_filter_budget(seq, delta)
        wh = whitehouse_filter_epsilon(seq, delta)
        print(
            f"  eps0={eps0:g} k={k}: basic={k * eps0:.3f}  advanced={ac:.3f}  Rogers filter={ro:.3f}  "
            f"Whitehouse filter={wh:.3f}  Rogers/Whitehouse={ro / wh:.2f}"
        )
    eps_g = 1.0
    rho_star = zcdp_budget_for(eps_g, delta)
    print(
        f"  eps_g=1, delta=1e-6: rho*={rho_star:.6f}  zcdp_to_dp(rho*)={zcdp_to_dp(rho_star, delta):.4f}  "
        f"Whitehouse at sum eps^2 = 2 rho*: {whitehouse_filter_epsilon([math.sqrt(2 * rho_star)], delta):.4f}"
    )
    for eps0 in (0.01, 0.001):
        kb = max_steps(lambda s: basic_filter_ok(s, eps_g), eps0)
        ka = max_steps(lambda s: advanced_composition_epsilon(s, delta) <= eps_g, eps0)
        kr = max_steps(lambda s: rogers_filter_ok(s, eps_g, delta), eps0)
        kw = max_steps(lambda s: whitehouse_filter_ok(s, eps_g, delta), eps0)
        print(
            f"  steps allowed at eps_g=1, eps0={eps0:g}: basic={kb}  advanced (fixed k)={ka}  "
            f"Rogers={kr}  Whitehouse={kw}"
        )

    print("[Part 3: two-step randomized response under a Rényi filter, alpha=2]")
    alpha = 2.0
    ea, eb = math.log(3), math.log(1.5)
    print(
        f"  rho(ln 3)={rr_rdp(alpha, ea):.4f}  rho(ln 1.5)={rr_rdp(alpha, eb):.4f}  "
        f"budget B=2 rho(ln 3)={2 * rr_rdp(alpha, ea):.4f}"
    )
    rows = two_step_rr(alpha, ea, ea, eb)
    for r in rows:
        print(
            f"  y=({r.y1},{r.y2}): P={r.p:.4f}  Q={r.q:.4f}  L={r.loss:+.4f}  spent={r.spent:.4f}  "
            f"N2={r.martingale:.4f}  P*N2={r.p * r.martingale:.4f}"
        )
    print(
        f"  E_P[N2]={sum(r.p * r.martingale for r in rows):.4f}  joint D_2={joint_renyi(rows, alpha):.4f}  "
        f"max path spend={max(r.spent for r in rows):.4f}"
    )
    print(
        f"  fixed-curve bound for the two-step run: {rr_rdp(alpha, ea) + max(rr_rdp(alpha, ea), rr_rdp(alpha, eb)):.4f}"
    )
    print("[Part 3: three-step randomized response, branches swap strong (ln 3) and weak (ln 1.5), alpha=2]")
    rows3 = three_step_rr(alpha, ea, ea, eb)
    for r3 in rows3:
        print(
            f"  y=({r3.ys[0]},{r3.ys[1]},{r3.ys[2]}): P={r3.p:.4f}  Q={r3.q:.4f}  L={r3.loss:+.4f}  "
            f"spent={r3.spent:.4f}  N3={r3.martingale:.4f}  P*N3={r3.p * r3.martingale:.4f}"
        )
    fixed_curve = rr_rdp(alpha, ea) + 2 * max(rr_rdp(alpha, ea), rr_rdp(alpha, eb))
    print(
        f"  E_P[N3]={sum(r3.p * r3.martingale for r3 in rows3):.4f}  joint D_2={transcript_renyi(rows3, alpha):.4f}  "
        f"filter budget B={max(r3.spent for r3 in rows3):.4f}  fixed-curve bound={fixed_curve:.4f}"
    )
    log_odds = math.log(rr_keep_probability(ea) / (1 - rr_keep_probability(ea)))
    for y1 in (1, 0):
        cond = sum(r3.p * r3.martingale for r3 in rows3 if r3.ys[0] == y1) / sum(r3.p for r3 in rows3 if r3.ys[0] == y1)
        n1 = math.exp((alpha - 1) * ((log_odds if y1 else -log_odds) - rr_rdp(alpha, ea)))
        print(f"  y1={y1}: L1={(log_odds if y1 else -log_odds):+.4f}  N1={n1:.4f}  E_P[N3 | y1]={cond:.4f}")
    print("[Part 3: adaptive Gaussian steps under a Rényi filter, alpha=2, B=1, 10^6 paths]")
    for top_up in (True, False):
        loss, spent, steps = adaptive_gaussian_filter(rng, 2.0, 1.0, 1_000_000, top_up)
        print(
            f"  top_up={top_up}: MC D_2={mc_renyi_from_losses(loss, 2.0):.4f}  max spend={float(spent.max()):.4f}  "
            f"mean spend={float(spent.mean()):.4f}  steps min={int(steps.min())} max={int(steps.max())}"
        )

    alpha_star, budget, _ = part5_setting()
    per10, per5 = alpha_star / (2 * 10.0**2), alpha_star / (2 * 5.0**2)
    left = budget - 60 * per10
    print(
        f"  noise switch at alpha={alpha_star:g}, B={budget:.3f}: per step sigma=10 {per10:.4f}, sigma=5 {per5:.4f}; "
        f"sigma=10 throughout: {math.floor(budget / per10 + 1e-9)} steps; after 60 steps left {left:.3f}, "
        f"then {math.floor(left / per5 + 1e-9)} steps at sigma=5"
    )

    print("[Part 4: odometers for full-batch Gaussian steps, sigma=10, delta=1e-5]")
    delta = DELTA
    print(
        f"  |orders|={len(ORDERS)}  ln(1/delta)={math.log(1 / delta):.3f}  "
        f"ln(|orders|/delta)={math.log(len(ORDERS) / delta):.3f}"
    )
    for t in (10, 100, 300, 1000):
        curve = gaussian_steps_rdp(SIGMA, t)
        print(
            f"  t={t}: rho={t / (2 * SIGMA**2):.3f}  fixed-time={fixed_time_conversion(curve, delta):.3f}  "
            f"Ville over orders={ville_odometer(curve, delta):.3f}  Lecuyer={lecuyer_odometer(curve, delta):.3f}  "
            f"Ville at alpha=5.8 only={ville_odometer(curve, delta, [5.8]):.3f}  "
            f"Ville/fixed={ville_odometer(curve, delta) / fixed_time_conversion(curve, delta):.2f}"
        )
    q_run, sigma_run, k_run = 0.01, 1.0, 10_000
    per_step = dict(zip(ORDERS, subsampled_gaussian_rdp_curve(q_run, sigma_run), strict=True))

    def run_curve(a: float) -> float:
        return k_run * per_step[a]

    print(
        f"  running example q={q_run:g} sigma={sigma_run:g} k={k_run}: fixed-time={fixed_time_conversion(run_curve, delta):.2f}  "
        f"Ville over orders={ville_odometer(run_curve, delta):.2f}  Lecuyer={lecuyer_odometer(run_curve, delta):.2f}"
    )
    mu, a, d, n = 0.2, 2.0, 0.05, 1000
    est = ville_crossing_mc(rng, mu, a, d, n, 100_000)
    bm = brownian_line_crossing(math.log(1 / d) / (a - 1), (a - 1) / 2, n * mu**2)
    print(
        f"  Ville tightness: Gaussian steps mu={mu:g}, alpha={a:g}, delta={d:g}, n={n}: "
        f"MC P(cross)={est:.4f}  Brownian limit={bm:.4f}"
    )
    t_star = 1.0 / mu**2 * (2 * math.log(1 / d) / ((a - 1) * (a - 1)))
    s = t_star * mu**2
    fixed = normal_cdf(-((a - 1) * s / 2 + math.log(1 / d) / (a - 1)) / math.sqrt(s))
    print(f"  at fixed variance-time s={s:.2f} the same boundary is crossed with probability {fixed:.4f}")

    print("[Part 5: individual accounting, full-batch DP-GD]")
    alpha_star, budget, eps_wc = part5_setting()
    print(f"  n={N_RECORDS} sigma={SIGMA:g} k0={K0} k_max={K_MAX} delta={DELTA:g}")
    print(
        f"  worst case: rho={K0 / (2 * SIGMA**2):.3f}  eps={eps_wc:.3f} at alpha*={alpha_star:g}  "
        f"budget B=eps(alpha*)={budget:.3f}  floor ln(1/delta)/(alpha*-1)={math.log(1 / DELTA) / (alpha_star - 1):.3f}"
    )
    pop = synthetic_gradient_norms(rng, N_RECORDS, K_MAX)
    print(f"  hard records: {int(pop.hard.sum())} ({pop.hard.mean():.1%})")
    fixed, best = individual_epsilons(pop.norms[:, :K0], SIGMA, DELTA, alpha_star)
    for name, m in (("all", np.ones(N_RECORDS, dtype=bool)), ("typical", ~pop.hard), ("hard", pop.hard)):
        q10, q50, q90 = (float(x) for x in np.percentile(fixed[m], [10, 50, 90]))
        print(
            f"  {name}: eps at alpha*: p10={q10:.2f} median={q50:.2f} p90={q90:.2f}  "
            f"median optimized over orders={float(np.median(best[m])):.2f}"
        )
    print(f"  records below half the worst-case eps: {float(np.mean(fixed < eps_wc / 2)):.1%}")
    cum, active = individual_filter(pop.norms, alpha_star, SIGMA, budget)
    for t in (K0, 2 * K0, K_MAX):
        a_all = float(active[:, t - 1].mean())
        a_typ = float(active[~pop.hard, t - 1].mean())
        a_hard = float(active[pop.hard, t - 1].mean())
        print(f"  step {t}: participating all={a_all:.1%}  typical={a_typ:.1%}  hard={a_hard:.1%}")
    first_drop = np.where((~active).any(axis=1), np.argmax(~active, axis=1) + 1, 0)
    hard_drops = first_drop[pop.hard]
    print(
        f"  hard records first excluded: median step {float(np.median(hard_drops[hard_drops > 0])):.0f}  "
        f"earliest {int(hard_drops[hard_drops > 0].min())}"
    )
    ever_typ = float(np.mean(first_drop[~pop.hard] > 0))
    print(
        f"  typical records ever excluded: {ever_typ:.1%}  "
        f"median final spend typical={float(np.median(cum[~pop.hard, -1])):.3f}  of B={budget:.3f}"
    )
    print(
        f"  per-step worst case alpha*/(2 sigma^2)={alpha_star / (2 * SIGMA**2):.4f}  "
        f"check gaussian_individual_rdp(alpha*, [1]*k0)={gaussian_individual_rdp(alpha_star, [1.0] * K0, SIGMA, 1.0):.3f}"
    )


# --- Figures --------------------------------------------------------------------------------


def plot_plugin_failure(rng: np.random.Generator, eps: float = 0.1, delta: float = 0.01) -> Figure:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))
    n = 2000
    paths, crossed = stop_when_ahead_paths(rng, eps, delta, n, 40)
    t = np.arange(1, n + 1)
    bound = [plugin_rr_bound(eps, int(s), delta) / eps for s in t]
    for row, hit in zip(paths, crossed, strict=True):
        ax1.plot(t, row, color="C3" if hit else "0.6", linewidth=0.8, alpha=0.9 if hit else 0.5)
    ax1.plot(t, bound, "k", linewidth=2, label="plug-in bound at each fixed t")
    ax1.set(
        xlabel="step t",
        ylabel=r"loss $L_t/\varepsilon$",
        title=f"Stop when ahead: frozen (red) paths crossed, ε={eps:g}, δ={delta:g}",
    )
    ax1.legend(loc="lower left", fontsize=9)
    horizons = [round(10 ** (i / 4)) for i in range(4, 17)]
    for d, color in ((0.05, "C0"), (0.01, "C1")):
        cross = plugin_crossing_probability(eps, d, horizons)
        ax2.plot(horizons, [cross[h] for h in horizons], "o-", color=color, label=f"P(cross by n), δ={d:g}")
        ax2.axhline(d, color=color, linestyle="--", linewidth=1)
    ax2.set_xscale("log")
    ax2.set(xlabel="horizon n", ylabel="probability", title="Crossing probability passes δ (dashed) and keeps growing")
    ax2.legend(fontsize=9)
    return fig


def plot_filter_budgets(eps0: float = 0.01, delta: float = 1e-6) -> Figure:
    ks = [round(10 ** (i / 8)) for i in range(8, 41)]
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.plot(ks, [k * eps0 for k in ks], "k:", label="basic composition (also a filter)")
    ax.plot(
        ks,
        [advanced_composition_epsilon([eps0] * k, delta) for k in ks],
        "C0--",
        label="advanced composition, k fixed in advance",
    )
    ax.plot(ks, [rogers_filter_budget([eps0] * k, delta) for k in ks], "C3", label="Rogers et al. filter")
    ax.plot(ks, [whitehouse_filter_epsilon([eps0] * k, delta) for k in ks], "C2", label="Whitehouse et al. filter")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set(
        xlabel="steps k of ε₀ = 0.01",
        ylabel=r"global budget $\varepsilon_g$ needed",
        title=f"Budget needed for k steps of ε₀={eps0:g}, $\\delta = 10^{{{round(math.log10(delta))}}}$",
    )
    ax.legend(fontsize=9)
    return fig


def plot_supermartingale(rng: np.random.Generator, alpha: float = 2.0, budget: float = 1.0) -> Figure:
    horizon = 14
    log_n = supermartingale_paths(rng, alpha, budget, 200_000, horizon)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))
    t = np.arange(horizon + 1)
    for row in log_n[:30]:
        ax1.plot(t, row, color="C0", alpha=0.4, linewidth=0.8)
    ax1.axhline(0, color="k", linewidth=1)
    ax1.set(
        xlabel="step t",
        ylabel=r"$\ln N_t = (\alpha-1)(L_t - \sum_{j \leq t} \varepsilon_j(\alpha))$",
        title="Thirty paths of the stopped process",
    )
    means = np.exp(log_n).mean(axis=0)
    ax2.plot(t, means, "o-", color="C0", label=r"Monte Carlo mean of $N_t$")
    ax2.axhline(1, color="k", linestyle="--", label="1 (its starting value)")
    ax2.set(xlabel="step t", ylabel=r"$E_P[N_t]$", ylim=(0.9, 1.1), title="Its mean stays at 1")
    ax2.legend(fontsize=9)
    return fig


def plot_odometers(sigma: float = SIGMA, delta: float = DELTA) -> Figure:
    ts = list(range(5, 1001, 5))
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(
        ts,
        [fixed_time_conversion(gaussian_steps_rdp(sigma, t), delta) for t in ts],
        "k--",
        label="fixed-time conversion (t chosen in advance)",
    )
    ax.plot(
        ts,
        [ville_odometer(gaussian_steps_rdp(sigma, t), delta) for t in ts],
        "C2",
        label="Ville odometer over the order grid",
    )
    ax.plot(
        ts,
        [lecuyer_odometer(gaussian_steps_rdp(sigma, t), delta) for t in ts],
        "C3",
        label="Lécuyer odometer (doubling budgets)",
    )
    ax.plot(
        ts,
        [ville_odometer(gaussian_steps_rdp(sigma, t), delta, [5.8]) for t in ts],
        "C0:",
        label="Ville odometer at α = 5.8 only",
    )
    ax.set(
        xlabel="steps t taken so far",
        ylabel=r"$\varepsilon$ at $\delta = 10^{-5}$",
        ylim=(0, 40),
        title=f"Running privacy bounds for full-batch Gaussian steps, σ={sigma:g}",
    )
    ax.legend(fontsize=9, loc="upper left")
    return fig


def plot_individual_hist(rng: np.random.Generator) -> Figure:
    alpha_star, _, eps_wc = part5_setting()
    pop = synthetic_gradient_norms(rng, N_RECORDS, K_MAX)
    fixed, _ = individual_epsilons(pop.norms[:, :K0], SIGMA, DELTA, alpha_star)
    fig, ax = plt.subplots(figsize=(8, 4.2))
    bins = [2.2 + i * 3.4 / 51 for i in range(52)]
    ax.hist(fixed[~pop.hard], bins=bins, color="C0", alpha=0.8, label="typical records (gradients shrink)")
    ax.hist(fixed[pop.hard], bins=bins, color="C3", alpha=0.8, label="hard records (gradients stay large)")
    ax.axvline(eps_wc, color="k", linestyle="--", label=f"worst case {eps_wc:.2f}")
    ax.axvline(
        math.log(1 / DELTA) / (alpha_star - 1), color="0.5", linestyle=":", label=r"floor $\ln(1/\delta)/(\alpha^*-1)$"
    )
    ax.set(
        xlabel=rf"individual $\varepsilon$ at $\delta = 10^{{-5}}$, order $\alpha^* = {alpha_star:g}$",
        ylabel="records",
        title=f"Individual ε after {K0} full-batch steps, σ={SIGMA:g}",
    )
    ax.set_xlim(2.2, 5.6)
    ax.legend(fontsize=9, loc="upper right", bbox_to_anchor=(0.86, 1.0))
    return fig


def plot_individual_spend(rng: np.random.Generator) -> Figure:
    alpha_star, budget, _ = part5_setting()
    pop = synthetic_gradient_norms(rng, N_RECORDS, K_MAX)
    cum, active = individual_filter(pop.norms, alpha_star, SIGMA, budget)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))
    t = np.arange(1, K_MAX + 1)
    ax1.plot(t, alpha_star * t / (2 * SIGMA**2), "k:", label="worst case, every step at the clip")
    typical = np.nonzero(~pop.hard)[0][:4]
    hard = np.nonzero(pop.hard)[0][:3]
    for i in typical:
        ax1.plot(t, cum[i], color="C0", linewidth=1)
    for i in hard:
        ax1.plot(t, cum[i], color="C3", linewidth=1)
    ax1.plot([], [], color="C0", label="typical records")
    ax1.plot([], [], color="C3", label="hard records")
    ax1.axhline(budget, color="k", linestyle="--", label=f"budget B = {budget:.2f}")
    ax1.set(
        xlabel="step t",
        ylabel=r"spend $\sum_j \varepsilon_{i,j}(\alpha^*)$",
        ylim=(0, budget * 1.25),
        title="Per-record spend under the individual filter",
    )
    ax1.legend(fontsize=8, loc="lower right", bbox_to_anchor=(1.0, 0.42))
    ax2.plot(t, active[~pop.hard].mean(axis=0), color="C0", label="typical")
    ax2.plot(t, active[pop.hard].mean(axis=0), color="C3", label="hard")
    ax2.axvline(K0, color="0.5", linestyle=":", label=f"worst-case run stops at {K0}")
    ax2.set(
        xlabel="step t", ylabel="fraction taking part", ylim=(-0.03, 1.05), title="Who is still in the gradient sum"
    )
    ax2.legend(fontsize=9, loc="center right")
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
        "adaptive_composition_plugin_failure.svg": plot_plugin_failure(rng),
        "adaptive_composition_filter_budgets.svg": plot_filter_budgets(),
        "adaptive_composition_supermartingale.svg": plot_supermartingale(rng),
        "adaptive_composition_odometers.svg": plot_odometers(),
        "adaptive_composition_individual_hist.svg": plot_individual_hist(rng),
        "adaptive_composition_individual_spend.svg": plot_individual_spend(rng),
    }
    for name, fig in figures.items():
        fig.tight_layout()
        fig.savefig(out / name, metadata={"Date": None})
        plt.close(fig)
        print(f"saved {out / name}")


if __name__ == "__main__":
    main()
