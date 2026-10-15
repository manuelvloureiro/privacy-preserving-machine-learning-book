"""Check the numbers in the choosing-an-accountant chapter and draw its figures.

    uv run python book/part04-privacy-accounting/code/choosing_an_accountant.py
    uv run python book/part04-privacy-accounting/code/choosing_an_accountant.py --seed 1 --out /tmp/figures

Prints the numbers quoted in 09-choosing-an-accountant.md (they match exactly with the default seed),
then saves the chapter's figures. Takes two to three minutes, almost all of it in PLD compositions.

Nothing is re-implemented here: every accountant is imported from the chapter that derives it.
  per-step amplification + basic/advanced composition  subsampling.py   (chapter 31)
  RDP with Mironov's or the improved conversion        subsampling.py   (chapters 27, 28, 31)
  RDP with the optimal conversion                      rdp_conversions.py (chapter 28)
  the GDP central-limit approximation                  gaussian_dp.py   (chapter 30)
  pessimistic / optimistic PLD                         numerical_accounting.py (chapter 32)

Conventions: q is the sampling rate, never a density; σ is the noise multiplier; k the number of steps;
neighbours are add/remove unless stated; Poisson sampling unless stated.

Public functions used by the text: accountant_table, sigma_for_target.
"""

import argparse
import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from functools import cache, lru_cache
from pathlib import Path

# The chapter modules configure matplotlib when imported; this module's settings must come last.
import matplotlib
import numpy as np
from gaussian_dp import clt_mu_subsampled_gaussian, gdp_eps, gdp_mu_for_eps
from numerical_accounting import (
    KS,
    PLD,
    compose_pld,
    compose_pld_many,
    discretize_subsampled_gaussian_pld,
    gaussian_pld,
    pld_delta,
    pld_epsilon,
    rr_exact_delta,
)
from rdp_conversions import (
    gaussian_rdp_curve,
    randomized_response_rdp,
    rdp_to_dp_improved,
    rdp_to_dp_mironov,
    rdp_to_dp_tradeoff,
)
from subsampling import (
    ORDERS,
    advanced_composition,
    dpsgd_eps_advanced,
    dpsgd_eps_basic,
    dpsgd_eps_rdp,
    subsampled_gaussian_rdp_curve,
)

matplotlib.use("Agg")
matplotlib.rcParams["svg.hashsalt"] = "choosing_an_accountant"  # stable element ids, so reruns don't churn the SVGs
matplotlib.rcParams["svg.fonttype"] = "none"  # keep text as text: smaller, diffable files

import matplotlib.pyplot as plt
from matplotlib.figure import Figure

FIGURES = Path(__file__).resolve().parent.parent / "figures"

# The running example of the part.
Q_RUN, SIGMA_RUN, DELTA_RUN = 0.01, 1.0, 1e-5
STEPS: tuple[int, ...] = (1, 10, 100, 1000, 10_000)
H = 1e-5  # PLD grid width, as in chapter 32: the sandwich width is k h
TARGETS: tuple[float, ...] = (1.0, 3.0, 8.0)  # target ε for the inverse problem, at k = 10,000
N_RECORDS = 50_000  # dataset size for the reporting example (q n = 500 records per batch on average)
DIRECTIONS = ("remove", "add")

ACCOUNTANTS: tuple[str, ...] = (
    "basic",
    "advanced",
    "rdp-mironov",
    "rdp-improved",
    "rdp-optimal",
    "clt",
    "pld-upper",
    "pld-lower",
)
ACCOUNTANT_NAMES: dict[str, str] = {
    "basic": "per step + basic composition",
    "advanced": "per step + advanced composition",
    "rdp-mironov": "RDP, Mironov conversion",
    "rdp-improved": "RDP, improved conversion",
    "rdp-optimal": "RDP, optimal conversion",
    "clt": "GDP central-limit approximation",
    "pld-upper": "PLD, pessimistic (upper bound)",
    "pld-lower": "PLD, optimistic (lower bound)",
}
BOUNDS = frozenset({"basic", "advanced", "rdp-mironov", "rdp-improved", "rdp-optimal", "pld-upper"})


# --- Section 35.3: every accountant on the running example ------------------------------------------


@lru_cache(maxsize=4)
def composed_plds(
    q: float, sigma: float, ks: tuple[int, ...], h: float, pessimistic: bool
) -> dict[int, tuple[PLD, PLD]]:
    """The (remove, add) PLDs of k Poisson-subsampled Gaussian steps for each k in ks (chapter 32's accountant)."""
    per_direction = {
        d: compose_pld_many(discretize_subsampled_gaussian_pld(q, sigma, h, pessimistic, d), ks) for d in DIRECTIONS
    }
    return {k: (per_direction["remove"][k], per_direction["add"][k]) for k in ks}


def max_direction(eps: Sequence[float]) -> float:
    """The larger ε of the two directions. Raises on NaN, which Python's max would silently hide; inf is allowed."""
    if any(math.isnan(e) for e in eps):
        raise ValueError(f"PLD accountant returned NaN: {list(eps)}")
    return max(eps)


def pld_eps(plds: tuple[PLD, PLD], delta: float) -> float:
    """ε at δ from a pair of composed PLDs: the larger of the two directions."""
    return max_direction([pld_epsilon(p, delta) for p in plds])


def pld_eps_single(q: float, sigma: float, steps: int, delta: float, h: float, pessimistic: bool) -> float:
    """ε at δ for one k, without caching the (large) composed PLDs."""
    return max_direction(
        [
            pld_epsilon(compose_pld(discretize_subsampled_gaussian_pld(q, sigma, h, pessimistic, d), steps), delta)
            for d in DIRECTIONS
        ]
    )


def rdp_optimal_eps(q: float, sigma: float, steps: int, delta: float) -> float:
    """RDP composed over k steps, converted with the optimal conversion of chapter 28 (through tests)."""
    curve = subsampled_gaussian_rdp_curve(q, sigma)
    return rdp_to_dp_tradeoff(ORDERS, [steps * e for e in curve], delta)


def clt_eps(q: float, sigma: float, steps: int, delta: float) -> float:
    """ε at δ of μ-GDP with the central-limit μ = q √(k(e^{1/σ²} - 1)). An approximation, not a bound."""
    return gdp_eps(clt_mu_subsampled_gaussian(q, sigma, steps), delta)


@dataclass(frozen=True)
class AccountantRow:
    """ε at δ after `steps` DP-SGD steps under every accountant of the part."""

    steps: int
    basic: float
    advanced: float
    rdp_mironov: float
    rdp_improved: float
    rdp_optimal: float
    clt: float
    pld_upper: float
    pld_lower: float

    def value(self, accountant: str) -> float:
        return float(getattr(self, accountant.replace("-", "_")))


def accountant_table(q: float, sigma: float, delta: float, ks: Sequence[int], h: float = H) -> list[AccountantRow]:
    """One row per k: every accountant's ε at δ for k steps of DP-SGD (Poisson rate q, noise multiplier σ)."""
    ks_t = tuple(sorted(set(ks)))
    upper, lower = composed_plds(q, sigma, ks_t, h, True), composed_plds(q, sigma, ks_t, h, False)
    rows: list[AccountantRow] = []
    for k in ks:
        rows.append(
            AccountantRow(
                steps=k,
                basic=dpsgd_eps_basic(q, sigma, k, delta),
                advanced=dpsgd_eps_advanced(q, sigma, k, delta)[0],
                rdp_mironov=dpsgd_eps_rdp(q, sigma, k, delta, "mironov")[0],
                rdp_improved=dpsgd_eps_rdp(q, sigma, k, delta, "improved")[0],
                rdp_optimal=rdp_optimal_eps(q, sigma, k, delta),
                clt=clt_eps(q, sigma, k, delta),
                pld_upper=pld_eps(upper[k], delta),
                pld_lower=pld_eps(lower[k], delta),
            )
        )
    return rows


# --- Section 35.4: the inverse problem, σ for a target ε ------------------------------------------


def epsilon(accountant: str, q: float, sigma: float, steps: int, delta: float, h: float = H) -> float:
    """ε at δ of DP-SGD under one named accountant (see ACCOUNTANTS)."""
    match accountant:
        case "basic":
            return dpsgd_eps_basic(q, sigma, steps, delta)
        case "advanced":
            return dpsgd_eps_advanced(q, sigma, steps, delta)[0]
        case "rdp-mironov":
            return dpsgd_eps_rdp(q, sigma, steps, delta, "mironov")[0]
        case "rdp-improved":
            return dpsgd_eps_rdp(q, sigma, steps, delta, "improved")[0]
        case "rdp-optimal":
            return rdp_optimal_eps(q, sigma, steps, delta)
        case "clt":
            return clt_eps(q, sigma, steps, delta)
        case "pld-upper":
            return pld_eps_single(q, sigma, steps, delta, h, True)
        case "pld-lower":
            return pld_eps_single(q, sigma, steps, delta, h, False)
        case _:
            raise ValueError(f"unknown accountant {accountant!r}")


def solve_decreasing(f: Callable[[float], float], target: float, lo: float, hi: float, tol: float = 1e-4) -> float:
    """The x in [lo, hi] with f(x) = target for decreasing f, by the Illinois variant of false position.

    Needs f(lo) > target >= f(hi). Returns the upper end of the final bracket, so f(result) <= target:
    for an accountant that is an upper bound, the returned σ is safe.
    """
    f_lo, f_hi = f(lo) - target, f(hi) - target
    if not (f_lo > 0 >= f_hi):
        raise ValueError(f"target not bracketed: f({lo})-t={f_lo}, f({hi})-t={f_hi}")
    side = 0
    while hi - lo > tol:
        mid = hi - f_hi * (hi - lo) / (f_hi - f_lo)  # where the secant crosses the target
        mid = min(max(mid, lo + tol / 4), hi - tol / 4)  # stay strictly inside the bracket
        f_mid = f(mid) - target
        if f_mid > 0:
            lo, f_lo = mid, f_mid
            if side == -1:
                f_hi /= 2  # Illinois: halve the stale end so the bracket shrinks from both sides
            side = -1
        else:
            hi, f_hi = mid, f_mid
            if side == 1:
                f_lo /= 2
            side = 1
    return hi


def clt_sigma(eps_target: float, q: float, steps: int, delta: float) -> float:
    """Closed form: μ* with gdp_eps(μ*) = target, then q²k(e^{1/σ²} - 1) = μ*², so σ = 1/√ln(1 + μ*²/(q²k))."""
    mu = gdp_mu_for_eps(eps_target, delta)
    return 1 / math.sqrt(math.log1p(mu**2 / (q**2 * steps)))


@cache
def sigma_for_target(eps_target: float, q: float, steps: int, delta: float, accountant: str, h: float = H) -> float:
    """The smallest noise multiplier σ whose ε at δ, under `accountant`, is at most eps_target (to 1e-4)."""
    if accountant == "clt":
        return clt_sigma(eps_target, q, steps, delta)
    if accountant in ("pld-upper", "pld-lower"):
        # PLD needs less noise than RDP (it is tighter), and the CLT value is a good lower guess: bracket
        # the root between them so that only a handful of expensive compositions are needed.
        hi = sigma_for_target(eps_target, q, steps, delta, "rdp-improved")
        lo = 0.9 * min(hi, clt_sigma(eps_target, q, steps, delta))
        return solve_decreasing(lambda s: epsilon(accountant, q, s, steps, delta, h), eps_target, lo, hi)
    lo, hi = 0.2, 1.0
    while epsilon(accountant, q, hi, steps, delta) > eps_target:
        lo, hi = hi, 2 * hi
    return solve_decreasing(lambda s: epsilon(accountant, q, s, steps, delta), eps_target, lo, hi)


# --- Section 35.5: other mechanisms, where the accountants rank differently ----------------------


def gaussian_comparison(mu: float, steps: int, delta: float) -> tuple[float, float, float]:
    """k Gaussian steps (no sampling) with μ each: exact ε (GDP, μ√k), RDP improved, RDP Mironov."""
    curve = [steps * e for e in gaussian_rdp_curve(mu)]
    return (
        gdp_eps(mu * math.sqrt(steps), delta),
        rdp_to_dp_improved(ORDERS, curve, delta)[0],
        rdp_to_dp_mironov(ORDERS, curve, delta)[0],
    )


def eps_from_delta_profile(delta_of_eps: Callable[[float], float], delta: float, hi: float = 50.0) -> float:
    """Smallest ε with δ(ε) <= δ, by bisection on a decreasing privacy profile."""
    lo = 0.0
    for _ in range(100):
        mid = (lo + hi) / 2
        if delta_of_eps(mid) > delta:
            lo = mid
        else:
            hi = mid
    return hi


def randomized_response_comparison(eps0: float, steps: int, delta: float) -> tuple[float, float, float, float]:
    """k-fold ε0-randomized response (pure DP): exact ε (its PLD is finite), RDP improved, zCDP, advanced.

    RDP uses the exact randomized-response curve; zCDP uses the smallest ρ = ε0 tanh(ε0/2) per step
    (chapter 29) and the improved conversion; advanced composition is Dwork–Rothblum–Vadhan with δ' = δ.
    """
    exact = eps_from_delta_profile(lambda e: rr_exact_delta(eps0, steps, e), delta)
    rdp = rdp_to_dp_improved(ORDERS, [steps * randomized_response_rdp(a, eps0) for a in ORDERS], delta)[0]
    rho = steps * eps0 * math.tanh(eps0 / 2)
    zcdp = rdp_to_dp_improved(ORDERS, [rho * a for a in ORDERS], delta)[0]
    adv = advanced_composition(eps0, 0.0, steps, delta)[0]
    return exact, rdp, zcdp, adv


# --- Section 35.8: sanity checks -------------------------------------------------------------------


def full_batch_checks(sigma: float, steps: int, delta: float) -> tuple[float, float, float, float, float]:
    """q = 1: (exact ε, RDP-accountant ε, CLT ε, CLT μ, exact μ) for k full-batch Gaussian steps."""
    exact = gdp_eps(math.sqrt(steps) / sigma, delta)
    rdp = dpsgd_eps_rdp(1.0, sigma, steps, delta, "improved")[0]
    mu_clt = clt_mu_subsampled_gaussian(1.0, sigma, steps)
    return exact, rdp, gdp_eps(mu_clt, delta), mu_clt, math.sqrt(steps) / sigma


def gaussian_pld_bracket(mu: float, steps: int, delta: float, h: float) -> tuple[float, float]:
    """The PLD accountant run on the plain Gaussian pair, which has a closed form to compare with."""
    return (
        pld_epsilon(compose_pld(gaussian_pld(mu, h, True), steps), delta),
        pld_epsilon(compose_pld(gaussian_pld(mu, h, False), steps), delta),
    )


# --- The chapter's numbers ---------------------------------------------------------------------


def print_table(rng: np.random.Generator) -> None:
    _ = rng  # nothing here is random; the generator is kept for a uniform interface
    q, sigma, delta = Q_RUN, SIGMA_RUN, DELTA_RUN

    print(f"[Part 1: DP-SGD q={q:g}, sigma={sigma:g}, delta={delta:g}, PLD grid h={H:g}]")
    rows = {r.steps: r for r in accountant_table(q, sigma, delta, KS)}
    for k in STEPS:
        r = rows[k]
        print(
            f"  k={k}: basic={r.basic:.2f}  advanced={r.advanced:.2f}  RDP Mironov={r.rdp_mironov:.4f}  "
            f"RDP improved={r.rdp_improved:.4f}  RDP optimal={r.rdp_optimal:.4f}  CLT={r.clt:.4f}  "
            f"PLD upper={r.pld_upper:.4f}  PLD lower={r.pld_lower:.4f}"
        )
    for k in STEPS:
        r = rows[k]
        ratios = "  ".join(f"{a}={r.value(a) / r.pld_upper:.2f}" for a in ACCOUNTANTS if a != "pld-upper")
        print(f"  k={k} ratio to PLD upper: {ratios}")
    for k in STEPS:
        r = rows[k]
        print(
            f"  k={k} slack of RDP improved over PLD upper: conversion={r.rdp_improved - r.rdp_optimal:.4f}  "
            f"curve={r.rdp_optimal - r.pld_upper:.4f}  total={r.rdp_improved - r.pld_upper:.4f}  "
            f"({100 * (r.rdp_improved / r.pld_upper - 1):.1f}%)"
        )
    # Proved: Mironov >= improved >= optimal >= true >= PLD lower, and PLD upper >= true. Not proved: optimal >= PLD
    # upper, since the pessimistic PLD may exceed the true ε by up to k h; that one is only observed here.
    print(
        f"  proved order RDP Mironov >= RDP improved >= RDP optimal >= PLD lower and PLD upper >= PLD lower "
        f"at every k in {list(KS)}: "
        f"{all(r.rdp_mironov >= r.rdp_improved >= r.rdp_optimal >= r.pld_lower and r.pld_upper >= r.pld_lower for r in rows.values())}"
    )
    print(
        f"  observed (not implied by theory): RDP optimal >= PLD upper at every k: "
        f"{all(r.rdp_optimal >= r.pld_upper for r in rows.values())}; smallest curve slack "
        f"{min(r.rdp_optimal - r.pld_upper for r in rows.values()):.4f}, largest kh {max(KS) * H:.4f}"
    )
    print(f"  CLT below PLD lower at every k: {all(r.clt < r.pld_lower for r in rows.values())}")

    print(f"[Part 2: noise multiplier for a target eps, k=10000, q={q:g}, delta={delta:g}]")
    sig: dict[tuple[float, str], float] = {}
    for target in TARGETS:
        for a in ("advanced", "rdp-mironov", "rdp-improved", "clt", "pld-upper", "pld-lower"):
            sig[target, a] = sigma_for_target(target, q, 10_000, delta, a)
        line = "  ".join(
            f"{a}={sig[target, a]:.3f}"
            for a in ("advanced", "rdp-mironov", "rdp-improved", "clt", "pld-upper", "pld-lower")
        )
        print(f"  target eps={target:g}: sigma {line}")
        base = sig[target, "pld-upper"]
        print(
            f"    noise relative to PLD upper: advanced={sig[target, 'advanced'] / base:.2f}  "
            f"RDP Mironov={sig[target, 'rdp-mironov'] / base:.3f}  RDP improved={sig[target, 'rdp-improved'] / base:.3f}  "
            f"CLT={sig[target, 'clt'] / base:.3f}  PLD lower={sig[target, 'pld-lower'] / base:.3f}"
        )
        check = epsilon("pld-upper", q, base, 10_000, delta)
        print(
            f"    check: PLD upper eps at sigma={base:.4f} is {check:.4f}; "
            f"RDP improved eps there is {epsilon('rdp-improved', q, base, 10_000, delta):.4f}"
        )
        s_clt = sig[target, "clt"]
        print(
            f"    at the CLT's sigma={s_clt:.4f}: PLD lower eps={epsilon('pld-lower', q, s_clt, 10_000, delta):.4f}  "
            f"PLD upper eps={epsilon('pld-upper', q, s_clt, 10_000, delta):.4f}"
        )

    print("[Part 3: other mechanisms, delta=1e-5]")
    exact, rdp_i, rdp_m = gaussian_comparison(0.1, 100, delta)
    print(
        f"  Gaussian, no sampling, sigma=10, k=100: exact (GDP)={exact:.4f}  RDP improved={rdp_i:.4f} "
        f"({100 * (rdp_i / exact - 1):.1f}% above)  RDP Mironov={rdp_m:.4f} ({100 * (rdp_m / exact - 1):.1f}% above)"
    )
    rr_exact, rr_rdp, rr_zcdp, rr_adv = randomized_response_comparison(0.1, 100, delta)
    print(
        f"  randomized response eps0=0.1, k=100: exact (PLD)={rr_exact:.4f}  RDP improved={rr_rdp:.4f}  "
        f"zCDP improved={rr_zcdp:.4f}  advanced composition={rr_adv:.4f}  basic={100 * 0.1:.4f}"
    )
    r = rows[10_000]
    print(
        f"  subsampled Gaussian k=10000: PLD [{r.pld_lower:.4f}, {r.pld_upper:.4f}]  RDP improved={r.rdp_improved:.4f} "
        f"({100 * (r.rdp_improved / r.pld_upper - 1):.1f}% above the upper bound)  CLT={r.clt:.4f}"
    )

    print("[Part 4: matching the accountant to the code, k=10000]")
    fixed_rdp = dpsgd_eps_rdp(q, sigma / 2, 10_000, delta)[0]
    fixed_pld = pld_eps_single(q, sigma / 2, 10_000, delta, H, True)
    print(
        f"  fixed-size batches accounted honestly (Poisson pair, sigma/2={sigma / 2:g}): RDP={fixed_rdp:.2f}  "
        f"PLD upper={fixed_pld:.2f}  vs Poisson RDP={r.rdp_improved:.2f}  PLD upper={r.pld_upper:.2f}"
    )
    no_amp = gdp_eps(math.sqrt(100) / sigma, delta)
    print(
        f"  shuffle and partition, 100 epochs, no amplification: mu=sqrt(100)/sigma={math.sqrt(100) / sigma:g}  "
        f"eps={no_amp:.4f}"
    )
    # Valid for zero-out neighbours, where each epoch's clipped sum moves by C. Under replace-one it moves by 2C.
    no_amp_replace = gdp_eps(2 * math.sqrt(100) / sigma, delta)
    print(
        f"  replace-one neighbours (shift 2C per epoch): mu=2*sqrt(100)/sigma={2 * math.sqrt(100) / sigma:g}  "
        f"eps={no_amp_replace:.2f}"
    )
    print(
        f"  factors over the Poisson PLD upper bound: fixed-size {fixed_pld / r.pld_upper:.2f}  "
        f"no amplification {no_amp / r.pld_upper:.2f}"
    )
    for trials in (10, 100):
        print(
            f"  {trials} full training runs composed (tuning on private data): RDP improved="
            f"{dpsgd_eps_rdp(q, sigma, trials * 10_000, delta)[0]:.2f}"
        )

    print(f"[Part 5: the privacy statement, n={N_RECORDS}]")
    n = N_RECORDS
    plds = composed_plds(q, sigma, tuple(sorted(KS)), H, True)[10_000]
    plds_lo = composed_plds(q, sigma, tuple(sorted(KS)), H, False)[10_000]
    print(
        f"  expected batch size q*n={q * n:g}  epochs k*q={10_000 * q:g}  1/n={1 / n:.1e}  "
        f"n^-1.1={n**-1.1:.1e}  1/n^2={1 / n**2:.1e}"
    )
    for d in (1 / n, 1e-5, n**-1.1, 1e-6, 1 / n**2):
        up_d, lo_d = round(pld_eps(plds, d), 4), round(pld_eps(plds_lo, d), 4)  # width of the printed bracket
        print(
            f"  delta={d:.1e}: PLD upper={up_d:.4f}  PLD lower={lo_d:.4f}  "
            f"width={up_d - lo_d:.4f}  RDP improved={dpsgd_eps_rdp(q, sigma, 10_000, d)[0]:.4f}"
        )
    curve = subsampled_gaussian_rdp_curve(q, sigma)
    eps_r, alpha_r = dpsgd_eps_rdp(q, sigma, 10_000, delta)
    print(
        f"  RDP at the best order: alpha={alpha_r:g}  eps(alpha)={10_000 * curve[ORDERS.index(alpha_r)]:.4f}  "
        f"(converts to {eps_r:.4f})"
    )

    print("[Part 6: sanity checks]")
    exact, rdp_q1, clt_q1, mu_clt, mu_exact = full_batch_checks(1.0, 10, delta)
    print(
        f"  q=1, sigma=1, k=10: exact={exact:.4f}  RDP accountant={rdp_q1:.4f}  "
        f"CLT={clt_q1:.4f} (mu={mu_clt:.4f} vs exact mu={mu_exact:.4f})"
    )
    up, dn = gaussian_pld_bracket(1.0, 10, delta, 1e-3)
    print(
        f"  PLD accountant on the plain Gaussian, mu=1, k=10, h=1e-3: [{dn:.4f}, {up:.4f}]  "
        f"closed form={gdp_eps(math.sqrt(10), delta):.4f}"
    )
    print(
        f"  delta of the composed pessimistic PLD at its own eps: {max(pld_delta(p, pld_eps(plds, delta)) for p in plds):.3e}"
    )
    eps_k = [rows[k].pld_upper for k in sorted(rows)]
    print(f"  PLD upper increasing in k: {all(a < b for a, b in zip(eps_k, eps_k[1:]))}")


# --- Figures -----------------------------------------------------------------------------------

STYLE: dict[str, tuple[str, str]] = {  # accountant -> (colour, line style)
    "basic": ("0.6", ":"),
    "advanced": ("0.4", "--"),
    "rdp-mironov": ("C4", "-"),
    "rdp-improved": ("C2", "-"),
    "clt": ("C1", "-."),
    "pld-upper": ("C3", "-"),
    "pld-lower": ("C0", "--"),
}


def plot_eps_vs_k(q: float = Q_RUN, sigma: float = SIGMA_RUN, delta: float = DELTA_RUN) -> Figure:
    """Left: ε against k, log-log, for every accountant. Right: each one's ratio to the PLD upper bound."""
    rows = accountant_table(q, sigma, delta, KS)
    ks = [r.steps for r in rows]
    fig, (ax, ax_r) = plt.subplots(1, 2, figsize=(10, 4.2))
    for a in ("basic", "advanced", "rdp-mironov", "rdp-improved", "clt", "pld-upper", "pld-lower"):
        colour, ls = STYLE[a]
        ax.plot(ks, [r.value(a) for r in rows], color=colour, ls=ls, marker=".", label=ACCOUNTANT_NAMES[a])
    ax.set(
        xscale="log",
        yscale="log",
        xlabel=r"number of steps $k$",
        ylabel=r"$\varepsilon$ at $\delta = 10^{-5}$",
        title=rf"DP-SGD, $q = {q:g}$, $\sigma = {sigma:g}$",
    )
    ax.legend(fontsize=7, frameon=False, loc="upper left")
    for a in ("rdp-mironov", "rdp-improved", "clt", "pld-lower"):
        colour, ls = STYLE[a]
        ax_r.plot(
            ks, [r.value(a) / r.pld_upper for r in rows], color=colour, ls=ls, marker=".", label=ACCOUNTANT_NAMES[a]
        )
    ax_r.axhline(1.0, color=STYLE["pld-upper"][0], lw=1)
    ax_r.text(1.2, 1.1, "PLD upper bound", color=STYLE["pld-upper"][0], fontsize=8)
    ax_r.axhspan(0, 1, color="0.92", zorder=0)
    ax_r.text(9000, 0.08, "below the certified upper bound", fontsize=8, color="0.3", ha="right")
    ax_r.set(
        xscale="log",
        ylim=(0, 7.5),
        xlabel=r"number of steps $k$",
        ylabel=r"$\varepsilon$ / PLD upper bound",
        title="Ratio to the certified upper bound",
    )
    ax_r.legend(fontsize=7, frameon=False, loc="upper right")
    return fig


def plot_noise_ratio(q: float = Q_RUN, steps: int = 10_000, delta: float = DELTA_RUN) -> Figure:
    """For each target ε, the σ each accountant asks for, relative to the σ of the PLD upper bound."""
    names = ("rdp-mironov", "rdp-improved", "pld-upper", "clt", "pld-lower")
    fig, ax = plt.subplots(figsize=(7.5, 3.6))
    markers = {1.0: "o", 3.0: "s", 8.0: "^"}
    for t in TARGETS:
        base = sigma_for_target(t, q, steps, delta, "pld-upper")
        xs = [sigma_for_target(t, q, steps, delta, a) / base for a in names]
        ax.plot(
            xs,
            range(len(names)),
            ls="none",
            marker=markers[t],
            ms=7,
            mfc="none",
            label=rf"target $\varepsilon = {t:g}$",
        )
    ax.axvline(1.0, color=STYLE["pld-upper"][0], lw=1)
    ax.axvspan(0.85, 1.0, color="0.92", zorder=0)
    ax.text(0.86, -0.45, "less noise than the\ncertified upper bound needs", fontsize=8, color="0.3", va="bottom")
    ax.set_yticks(range(len(names)), [ACCOUNTANT_NAMES[a] for a in names])
    ax.set(
        xlim=(0.85, 1.3),
        ylim=(-0.6, 4.6),
        xlabel=r"$\sigma$ required / $\sigma$ required by the PLD upper bound",
        title=rf"Noise for a target $\varepsilon$ at $\delta = 10^{{-5}}$, $k = {steps:,}$, $q = {q:g}$",
    )
    ax.legend(fontsize=8, frameon=False, loc="upper right")
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
        "choosing_an_accountant_eps_vs_k.svg": plot_eps_vs_k(),
        "choosing_an_accountant_noise.svg": plot_noise_ratio(),
    }
    for name, fig in figures.items():
        fig.tight_layout()
        fig.savefig(out / name, metadata={"Date": None})  # no timestamp, so output is reproducible
        plt.close(fig)
        print(f"saved {out / name}")


if __name__ == "__main__":
    main()
