"""Check the numbers in the appendix "Accounting for DP-SGD" and draw its figures.

    uv run python code/appendix-derivations/apx_core_dp_sgd.py
    uv run python code/appendix-derivations/apx_core_dp_sgd.py --seed 1 --out /tmp/figures

Prints every number that book-full/appendix-derivations/03-dp-sgd.md quotes (it matches exactly
with the default seed), then saves the page's figures. Self-contained: the accountant is rewritten
here in a few dozen lines, so it can be checked against the chapters' scripts without importing them.
"""

import argparse
import math
from collections.abc import Callable, Sequence
from functools import cache
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
matplotlib.rcParams["svg.hashsalt"] = "apx_core_dp_sgd"  # stable element ids, so reruns don't churn the SVGs
matplotlib.rcParams["svg.fonttype"] = "none"  # keep text as text: smaller, diffable files

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from matplotlib.patches import Circle
from matplotlib.ticker import NullFormatter
from numpy.typing import NDArray

Array = NDArray[np.float64]

# Figures are build output: _build/figures/<part>/, which the build copies into each book.
FIGURES = Path(__file__).resolve().parents[2] / "_build" / "figures" / Path(__file__).parent.name

# The book's running example (Part 4 and the DP-SGD chapter): n records, expected batch B, rate q = B/n,
# noise multiplier σ, 100 epochs of n/B = 100 steps each, δ = 1e-5.
N_RUN, BATCH_RUN = 10_000, 100
Q_RUN = BATCH_RUN / N_RUN
SIGMA_RUN, EPOCHS_RUN, DELTA_RUN = 1.0, 100, 1e-5
STEPS_RUN = EPOCHS_RUN * N_RUN // BATCH_RUN

# The grid of orders the chapters of Part 4 share: fractional orders 1.1 to 10.9, then integers.
ORDERS: tuple[float, ...] = tuple(
    [1 + x / 10 for x in range(1, 100)] + [float(a) for a in range(12, 64)] + [128.0, 256.0, 512.0]
)
# Quoted, not recomputed: the pessimistic privacy loss distribution (PLD) accountant of Part 4's
# "Numerical accounting" chapter (its table tbl-num-dpsgd) for the running example, by epochs.
PLD_UPPER_RUN: dict[int, float] = {1: 0.7185, 10: 1.8332, 100: 6.2377}

INTEGER_ORDERS: tuple[float, ...] = tuple(a for a in ORDERS if a.is_integer())


def logsumexp(x: Array) -> float:
    m = float(np.max(x))
    return m + math.log(float(np.sum(np.exp(x - m))))


def bisect(f: Callable[[float], bool], lo: float, hi: float, iters: int = 60) -> float:
    """The point where the predicate f switches from False (at lo) to True (at hi)."""
    for _ in range(iters):
        mid = (lo + hi) / 2
        if f(mid):
            hi = mid
        else:
            lo = mid
    return hi


# --- Clipping gives sensitivity ------------------------------------------------------------------


def clip(g: Array, c: float) -> Array:
    """g · min(1, C/‖g‖): the same direction, length at most C."""
    norm = float(np.linalg.norm(g))
    return g if norm <= c else g * (c / norm)


def clipped_sum(grads: Sequence[Array], c: float) -> Array:
    return np.sum([clip(g, c) for g in grads], axis=0)


WORKED_D = (np.array([3.0, 4.0]), np.array([0.3, -0.4]))  # two records' gradients, clipping norm C = 1
WORKED_X = np.array([-2.4, 0.7])  # the added record's gradient, length 2.5
WORKED_X_REPLACE = np.array([-3.0, -4.0])  # replaces (3, 4) under the replace relation
WORKED_C = 1.0


def largest_change(rng: np.random.Generator, c: float, trials: int, records: int = 5) -> tuple[float, float]:
    """Largest ‖S(D') - S(D)‖ / C over random 2-D gradient sets, for add/remove and for replace-one."""
    worst_add = worst_replace = 0.0
    for _ in range(trials):
        grads = [g for g in rng.standard_normal((records, 2)) * rng.exponential(3.0, (records, 1))]
        new = rng.standard_normal(2) * rng.exponential(3.0)
        base = clipped_sum(grads, c)
        worst_add = max(worst_add, float(np.linalg.norm(clipped_sum([*grads, new], c) - base)) / c)
        worst_replace = max(worst_replace, float(np.linalg.norm(clipped_sum([new, *grads[1:]], c) - base)) / c)
    return worst_add, worst_replace


# --- One noisy step is a Gaussian mechanism -------------------------------------------------------


def log_ndtr(x: float) -> float:
    """ln Φ(x), accurate far into the lower tail, where Φ itself underflows."""
    if x > -20:
        return math.log(0.5 * math.erfc(-x / math.sqrt(2)))
    # Asymptotic series Φ(x) = φ(x)/|x| · (1 - 1/x² + 3/x⁴ - 15/x⁶ + ...) for x → -∞.
    x2 = x * x
    series = 1 - 1 / x2 + 3 / x2**2 - 15 / x2**3 + 105 / x2**4
    return -x2 / 2 - math.log(-x * math.sqrt(2 * math.pi)) + math.log(series)


def gaussian_log_delta(eps: float, mu: float) -> float:
    """ln δ(ε) for the Gaussian mechanism with ratio μ = Δ/s: δ = Φ(μ/2 - ε/μ) - e^ε Φ(-μ/2 - ε/μ)."""
    log_first = log_ndtr(mu / 2 - eps / mu)
    log_second = eps + log_ndtr(-mu / 2 - eps / mu)
    return log_first + math.log1p(-math.exp(log_second - log_first))


def gaussian_eps(mu: float, delta: float) -> float:
    """The smallest ε with δ(ε) ≤ δ for the Gaussian mechanism with ratio μ."""
    return bisect(lambda e: gaussian_log_delta(e, mu) <= math.log(delta), 0.0, mu * mu + 50 * mu + 50, 200)


def no_sampling_eps(sigma: float, steps: int, delta: float) -> float:
    """T full-batch Gaussian steps compose to one Gaussian mechanism with ratio √T/σ."""
    return gaussian_eps(math.sqrt(steps) / sigma, delta)


# --- Amplification by subsampling ---------------------------------------------------------------


def amplified_eps(eps: float, q: float) -> float:
    """ε' = ln(1 + q(e^ε - 1)) for an ε-DP step run on a Poisson sample with rate q."""
    return math.log1p(q * math.expm1(eps))


def laplace_sampled_log_ratios(eps: float, q: float) -> tuple[float, float]:
    """Largest ln-ratio, both directions, of a Laplace count (scale 1/ε) on a Poisson sample, on a grid.

    Without the record the count is noise around 0; with it, around 1 with probability q.
    """
    y = np.linspace(-20.0, 20.0, 400_001)
    log_p0 = -eps * np.abs(y)
    log_p1 = -eps * np.abs(y - 1)
    log_mix = np.logaddexp(math.log1p(-q) + log_p0, math.log(q) + log_p1)
    return float(np.max(log_mix - log_p0)), float(np.max(log_p0 - log_mix))


# --- RDP of the subsampled Gaussian --------------------------------------------------------------


def log_a_binomial(alpha: int, q: float, sigma: float) -> float:
    """ln A_α = ln Σ_j C(α, j)(1-q)^{α-j} q^j exp((j² - j)/(2σ²)) for integer α, in log space."""
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
    return [
        math.comb(alpha, j) * (1 - q) ** (alpha - j) * q**j * math.exp((j * j - j) / (2 * sigma**2))
        for j in range(alpha + 1)
    ]


def _log_mixture_ratio(y: Array, q: float, sigma: float) -> Array:
    """ln((1 - q) + q P1(y)/P0(y)) with P1/P0 = exp((2y - 1)/(2σ²)), without overflow."""
    return np.logaddexp(math.log1p(-q), math.log(q) + (2 * y - 1) / (2 * sigma**2))


def _log_normal_pdf(y: Array, sigma: float) -> Array:
    return -(y**2) / (2 * sigma**2) - math.log(sigma * math.sqrt(2 * math.pi))


def log_a_numeric(alpha: float, q: float, sigma: float, power: float | None = None) -> float:
    """ln E_{P0}[((1-q) + q P1/P0)^power] by a Riemann sum on a fine grid (power = α by default).

    The integrand has a bump near y = 0 and, for large powers, one near y = α, so the grid runs from
    -12σ - 2 to α + 12σ with step σ/100.
    """
    p = alpha if power is None else power
    y = np.arange(-12 * sigma - 2, alpha + 12 * sigma, sigma / 100)
    return logsumexp(_log_normal_pdf(y, sigma) + p * _log_mixture_ratio(y, q, sigma)) + math.log(sigma / 100)


def rdp_remove(alpha: float, q: float, sigma: float) -> float:
    """ε_{q,σ}(α) = D_α(mixture ‖ P0): exact binomial sum for integer α, numerical average otherwise."""
    if alpha.is_integer():
        return log_a_binomial(int(alpha), q, sigma) / (alpha - 1)
    return log_a_numeric(alpha, q, sigma) / (alpha - 1)


def rdp_add(alpha: float, q: float, sigma: float) -> float:
    """D_α(P0 ‖ mixture) = ln E_{P0}[(mixture/P0)^{-(α-1)}] / (α - 1): the other direction."""
    return log_a_numeric(alpha, q, sigma, power=-(alpha - 1)) / (alpha - 1)


def small_q_term(alpha: float, q: float, sigma: float) -> float:
    """The j = 2 term alone: α q² (e^{1/σ²} - 1) / 2."""
    return alpha * q * q * math.expm1(1 / sigma**2) / 2


def abadi_term(alpha: float, q: float, sigma: float) -> float:
    """The often-quoted q² α / σ² (the leading term of Abadi et al.'s bound, without the 1/(1-q))."""
    return q * q * alpha / sigma**2


@cache
def rdp_curve(q: float, sigma: float, orders: tuple[float, ...] = ORDERS) -> tuple[float, ...]:
    return tuple(rdp_remove(a, q, sigma) for a in orders)


def gaussian_moment_mc(rng: np.random.Generator, j: int, sigma: float, samples: int) -> float:
    """Monte Carlo average of (P1/P0)^j = exp(j(2y - 1)/(2σ²)) for y ~ N(0, σ²)."""
    y = rng.normal(0.0, sigma, samples)
    return float(np.mean(np.exp(j * (2 * y - 1) / (2 * sigma**2))))


# --- From the curve to (ε, δ) --------------------------------------------------------------------


def mironov_eps(alpha: float, rdp: float, delta: float) -> float:
    """ε = ε(α) + ln(1/δ)/(α - 1)."""
    return rdp + math.log(1 / delta) / (alpha - 1)


def improved_eps(alpha: float, rdp: float, delta: float) -> float:
    """ε = ε(α) + ln((α - 1)/α) - (ln δ + ln α)/(α - 1) (Balle et al. 2020)."""
    return max(0.0, rdp + math.log((alpha - 1) / alpha) - (math.log(delta) + math.log(alpha)) / (alpha - 1))


CONVERSIONS: dict[str, Callable[[float, float, float], float]] = {"mironov": mironov_eps, "improved": improved_eps}


def dpsgd_eps(
    q: float, sigma: float, steps: int, delta: float, conversion: str = "mironov", orders: tuple[float, ...] = ORDERS
) -> tuple[float, float]:
    """(ε, best order) for `steps` subsampled Gaussian steps: multiply the curve by T, convert, minimize."""
    convert = CONVERSIONS[conversion]
    curve = rdp_curve(q, sigma, orders)
    return min((convert(a, steps * e, delta), a) for a, e in zip(orders, curve, strict=True))


def sigma_for_eps(target: float, q: float, steps: int, delta: float, conversion: str = "mironov") -> float:
    """The smallest noise multiplier whose ε is at most the target, by bisection (ε falls as σ grows)."""
    return bisect(lambda s: dpsgd_eps(q, s, steps, delta, conversion)[0] <= target, 0.3, 50.0, 40)


# --- The table -------------------------------------------------------------------------------------


def print_table(rng: np.random.Generator) -> None:
    print("== Clipping gives sensitivity ==")
    c = WORKED_C
    for g in WORKED_D + (WORKED_X,):
        cg = clip(g, c)
        print(f"  g=({g[0]:g}, {g[1]:g}) |g|={np.linalg.norm(g):.2f} -> clipped ({cg[0]:.2f}, {cg[1]:.2f})")
    s_d = clipped_sum(list(WORKED_D), c)
    s_add = clipped_sum([*WORKED_D, WORKED_X], c)
    s_rep = clipped_sum([WORKED_X_REPLACE, WORKED_D[1]], c)
    print(
        f"  S(D)=({s_d[0]:.2f}, {s_d[1]:.2f})  S(D+x)=({s_add[0]:.2f}, {s_add[1]:.2f})  change={np.linalg.norm(s_add - s_d):.2f}"
    )
    print(f"  replace (3,4) by (-3,-4): S=({s_rep[0]:.2f}, {s_rep[1]:.2f})  change={np.linalg.norm(s_rep - s_d):.2f}")
    worst_add, worst_rep = largest_change(rng, 1.0, trials=20_000)
    print(f"  random search, 20000 trials: largest change / C  add/remove={worst_add:.4f}  replace={worst_rep:.4f}")

    print("== One noisy step; T steps without sampling ==")
    for steps in (1, 100, STEPS_RUN):
        mu = math.sqrt(steps) / SIGMA_RUN
        print(f"  T={steps}: mu={mu:g}  eps={no_sampling_eps(SIGMA_RUN, steps, DELTA_RUN):.2f}")
    print(f"  full batch, sigma=10.8, T=100: eps={no_sampling_eps(10.8, 100, DELTA_RUN):.3f}")

    print("== Amplification by subsampling ==")
    for eps in (0.1, 1.0, 5.0, 10.0):
        print(f"  q={Q_RUN} eps={eps:g}: eps'={amplified_eps(eps, Q_RUN):.4f}  q*eps={Q_RUN * eps:.4f}")
    print(f"  1 + q(e - 1) = {1 + Q_RUN * math.expm1(1.0):.6f}")
    up, down = laplace_sampled_log_ratios(1.0, Q_RUN)
    print(
        f"  Laplace count, eps=1, q={Q_RUN}: largest log ratio remove={up:.6f} add={down:.6f}  formula={amplified_eps(1.0, Q_RUN):.6f}"
    )
    up, down = laplace_sampled_log_ratios(1.0, 0.1)
    print(
        f"  Laplace count, eps=1, q=0.1: largest log ratio remove={up:.4f} add={down:.4f}  formula={amplified_eps(1.0, 0.1):.4f}"
    )

    print("== RDP of the subsampled Gaussian ==")
    terms = binomial_terms(2, Q_RUN, SIGMA_RUN)
    print("  alpha=2 terms: " + "  ".join(f"{t:.6f}" for t in terms) + f"  A_2={sum(terms):.8f}")
    print(f"  eps(2)={rdp_remove(2.0, Q_RUN, SIGMA_RUN):.4e}")
    mc = gaussian_moment_mc(rng, 2, SIGMA_RUN, 2_000_000)
    print(f"  E[r^2]: formula={math.exp(1 / SIGMA_RUN**2):.4f}  Monte Carlo={mc:.4f}")
    for a in (2.0, 4.0, 8.0, 16.0, 32.0):
        ex = rdp_remove(a, Q_RUN, SIGMA_RUN)
        approx = small_q_term(a, Q_RUN, SIGMA_RUN)
        print(
            f"  alpha={a:g}: exact={ex:.4e} small-q={approx:.4e} ratio={ex / approx:.2f}"
            f" q2a/s2={abadi_term(a, Q_RUN, SIGMA_RUN):.4e} unsampled={a / (2 * SIGMA_RUN**2):g}"
        )
    for a in (2.0, 8.0, 32.0):
        integral = log_a_numeric(a, Q_RUN, SIGMA_RUN) / (a - 1)
        print(
            f"  alpha={a:g}: binomial={rdp_remove(a, Q_RUN, SIGMA_RUN):.6e} integral={integral:.6e}"
            f" add direction={rdp_add(a, Q_RUN, SIGMA_RUN):.4e}"
        )
    for s in (2.0, 4.0):
        print(f"  sigma={s:g}, alpha=8: exact/small-q={rdp_remove(8.0, Q_RUN, s) / small_q_term(8.0, Q_RUN, s):.3f}")

    print("== Accounting: the running example ==")
    print(f"  n={N_RUN} B={BATCH_RUN} q={Q_RUN} sigma={SIGMA_RUN} steps={STEPS_RUN} delta={DELTA_RUN:g}")
    for epochs in (1, 5, 10, 20, 50, 100):
        steps = epochs * N_RUN // BATCH_RUN
        mir, a_mir = dpsgd_eps(Q_RUN, SIGMA_RUN, steps, DELTA_RUN, "mironov")
        imp, a_imp = dpsgd_eps(Q_RUN, SIGMA_RUN, steps, DELTA_RUN, "improved")
        none = no_sampling_eps(SIGMA_RUN, steps, DELTA_RUN)
        print(
            f"  epochs={epochs} T={steps}: no sampling={none:.1f}  Mironov={mir:.2f} (alpha={a_mir:g})"
            f"  improved={imp:.2f} (alpha={a_imp:g})"
        )
    mir_int, a_int = dpsgd_eps(Q_RUN, SIGMA_RUN, STEPS_RUN, DELTA_RUN, "mironov", INTEGER_ORDERS)
    imp_int, a_imp_int = dpsgd_eps(Q_RUN, SIGMA_RUN, STEPS_RUN, DELTA_RUN, "improved", INTEGER_ORDERS)
    print(
        f"  integer orders only: Mironov={mir_int:.4f} (alpha={a_int:g})  improved={imp_int:.4f} (alpha={a_imp_int:g})"
    )
    mir, a_mir = dpsgd_eps(Q_RUN, SIGMA_RUN, STEPS_RUN, DELTA_RUN, "mironov")
    rdp_at = rdp_remove(a_mir, Q_RUN, SIGMA_RUN)
    print(
        f"  at alpha={a_mir:g}: eps(alpha)={rdp_at:.4e}  T*eps(alpha)={STEPS_RUN * rdp_at:.4f}"
        f"  ln(1/delta)/(alpha-1)={math.log(1 / DELTA_RUN) / (a_mir - 1):.4f}  sum={mir:.4f}"
    )
    fixed, _ = dpsgd_eps(Q_RUN, SIGMA_RUN / 2, STEPS_RUN, DELTA_RUN, "improved")
    imp, _ = dpsgd_eps(Q_RUN, SIGMA_RUN, STEPS_RUN, DELTA_RUN, "improved")
    print(f"  fixed-size batches (noise multiplier sigma/2): improved={fixed:.2f} against Poisson {imp:.2f}")
    for delta in (1e-3, 1e-5, 1e-7):
        print(f"  delta={delta:g}: Mironov eps={dpsgd_eps(Q_RUN, SIGMA_RUN, STEPS_RUN, delta, 'mironov')[0]:.2f}")

    print("== Recipe: the noise multiplier for a target eps (T=10000, q=0.01) ==")
    for target in (1.0, 2.0, 4.0, 8.0):
        s_mir = sigma_for_eps(target, Q_RUN, STEPS_RUN, DELTA_RUN, "mironov")
        s_imp = sigma_for_eps(target, Q_RUN, STEPS_RUN, DELTA_RUN, "improved")
        check, _ = dpsgd_eps(Q_RUN, s_mir, STEPS_RUN, DELTA_RUN, "mironov")
        print(f"  target={target:g}: sigma Mironov={s_mir:.3f} (eps={check:.4f})  improved={s_imp:.3f}")


# --- Figures --------------------------------------------------------------------------------------


def plot_clipping(rng: np.random.Generator, c: float = 1.0) -> Figure:
    fig, (left, right) = plt.subplots(1, 2, figsize=(10, 4.6))
    norms = rng.lognormal(0.0, 0.8, 9)
    angles = rng.uniform(0, 2 * math.pi, 9)
    grads = [n * np.array([math.cos(t), math.sin(t)]) for n, t in zip(norms, angles, strict=True)]
    left.add_patch(Circle((0, 0), c, fill=False, linestyle="--", color="k"))
    for i, g in enumerate(grads):
        cg = clip(g, c)
        left.annotate("", xy=(g[0], g[1]), xytext=(0, 0), arrowprops={"arrowstyle": "->", "color": "0.75", "lw": 1.2})
        left.annotate("", xy=(cg[0], cg[1]), xytext=(0, 0), arrowprops={"arrowstyle": "->", "color": "C0", "lw": 1.8})
        if i == 0:
            left.plot([], [], color="0.75", label="per-example gradient")
            left.plot([], [], color="C0", label="after clipping")
    left.plot([], [], "k--", label=f"ball of radius C = {c:g}")
    lim = max(float(np.max(np.abs(np.array(grads)))), 1.2) + 0.3
    left.set(xlim=(-lim, lim), ylim=(-lim, lim), aspect="equal", title="Clipping keeps the direction, caps the length")
    left.legend(loc="lower left", fontsize=8)

    clipped = [clip(g, c) for g in grads[1:]]
    s_d = np.sum(clipped, axis=0)
    added = clip(grads[0], c)
    s_d2 = s_d + added
    right.add_patch(Circle((s_d[0], s_d[1]), c, fill=False, linestyle="--", color="k"))
    right.plot([s_d[0]], [s_d[1]], "o", color="C1", label="clipped sum on D")
    right.plot([s_d2[0]], [s_d2[1]], "s", color="C2", label="clipped sum on D' = D + one record")
    right.annotate(
        "", xy=(s_d2[0], s_d2[1]), xytext=(s_d[0], s_d[1]), arrowprops={"arrowstyle": "->", "color": "C2", "lw": 1.8}
    )
    right.plot([], [], "k--", label="every place D' can land: radius C")
    right.set(
        xlim=(s_d[0] - 2.2 * c, s_d[0] + 2.2 * c),
        ylim=(s_d[1] - 2.2 * c, s_d[1] + 2.2 * c),
        aspect="equal",
        title="One added record moves the sum by at most C",
    )
    right.legend(loc="lower left", fontsize=8)
    return fig


def plot_subsampling_ratio(eps: float = 1.0, q: float = 0.1) -> Figure:
    y = np.linspace(-4.0, 5.0, 901)
    b = 1 / eps
    p0 = np.exp(-np.abs(y) / b) / (2 * b)
    p1 = np.exp(-np.abs(y - 1) / b) / (2 * b)
    mix = (1 - q) * p0 + q * p1
    fig, (left, right) = plt.subplots(1, 2, figsize=(10, 4))
    left.plot(y, p0, label="without the record, $P_0$")
    left.plot(y, p1, label="record used, $P_1$")
    left.plot(y, mix, "k--", label=r"with the record, sampled: $(1-q)P_0 + qP_1$")
    left.set(xlabel="output y", ylabel="density", ylim=(0, 0.8), title=f"Laplace count, ε = {eps:g}, q = {q:g}")
    left.legend(fontsize=8, loc="upper right")
    right.plot(y, p1 / p0, label="$P_1/P_0$ (no sampling)")
    right.plot(y, mix / p0, "k--", label="mixture / $P_0$")
    right.axhline(math.exp(eps), color="C1", linestyle=":", label=rf"$e^\varepsilon$ = {math.exp(eps):.3f}")
    right.axhline(
        1 + q * math.expm1(eps),
        color="k",
        linestyle=":",
        label=rf"$1 + q(e^\varepsilon - 1)$ = {1 + q * math.expm1(eps):.3f}",
    )
    right.set(xlabel="output y", ylabel="likelihood ratio", title="Sampling pulls every ratio toward 1")
    right.legend(fontsize=8)
    return fig


def plot_rdp_curve(q: float = Q_RUN, sigma: float = SIGMA_RUN) -> Figure:
    orders = [float(a) for a in range(2, 65)]
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    ax.plot(orders, [a / (2 * sigma**2) for a in orders], label=r"no sampling: $\alpha/(2\sigma^2)$")
    ax.plot(orders, [rdp_remove(a, q, sigma) for a in orders], "k", lw=2, label=f"subsampled, exact (q = {q:g})")
    ax.plot(
        orders,
        [small_q_term(a, q, sigma) for a in orders],
        "--",
        label=r"small-$q$ term $\alpha q^2(e^{1/\sigma^2}-1)/2$",
    )
    ax.axvline(2 * sigma**2 * math.log(1 / q), color="0.5", linestyle=":", label=r"$\alpha = 2\sigma^2\ln(1/q)$")
    ticks = [2, 4, 8, 16, 32, 64]
    ax.set(
        xscale="log",
        xticks=ticks,
        xticklabels=[str(t) for t in ticks],
        yscale="log",
        xlabel=r"order $\alpha$",
        ylabel=r"RDP $\varepsilon(\alpha)$ of one step",
        title=f"One DP-SGD step, σ = {sigma:g}",
    )
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.legend(fontsize=8, loc="lower right")
    return fig


def plot_eps_epochs(q: float = Q_RUN, sigma: float = SIGMA_RUN, delta: float = DELTA_RUN) -> Figure:
    epochs = [1, 2, 3, 5, 7, 10, 15, 20, 30, 50, 70, 100]
    steps = [e * round(1 / q) for e in epochs]
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    ax.plot(epochs, [no_sampling_eps(sigma, t, delta) for t in steps], "o-", label="every record in every step")
    ax.plot(
        epochs, [dpsgd_eps(q, sigma, t, delta, "mironov")[0] for t in steps], "s-", label="subsampled, RDP, Mironov"
    )
    ax.plot(
        epochs, [dpsgd_eps(q, sigma, t, delta, "improved")[0] for t in steps], "^-", label="subsampled, RDP, improved"
    )
    ax.plot(list(PLD_UPPER_RUN), list(PLD_UPPER_RUN.values()), "kx", ms=9, label="subsampled, PLD upper bound (Part 4)")
    ax.set(
        xscale="log",
        yscale="log",
        xlabel="epochs",
        ylabel=f"ε at δ = {delta:g}",
        title=f"The running example: q = {q:g}, σ = {sigma:g}",
    )
    ax.legend(fontsize=8)
    return fig


def plot_conversion(
    q: float = Q_RUN, sigma: float = SIGMA_RUN, steps: int = STEPS_RUN, delta: float = DELTA_RUN
) -> Figure:
    orders = [a for a in ORDERS if a <= 16]
    curve = [steps * rdp_remove(a, q, sigma) for a in orders]
    toll = [math.log(1 / delta) / (a - 1) for a in orders]
    total = [c + t for c, t in zip(curve, toll, strict=True)]
    best, a_best = dpsgd_eps(q, sigma, steps, delta, "mironov")
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    ax.plot(orders, curve, label=r"$T\varepsilon(\alpha)$: the composed curve, rises")
    ax.plot(orders, toll, label=r"$\ln(1/\delta)/(\alpha-1)$: the conversion toll, falls")
    ax.plot(orders, total, "k", lw=2, label="their sum: ε at this order")
    ax.plot([a_best], [best], "ro", label=f"minimum ε = {best:.2f} at α = {a_best:g}")
    ax.set(ylim=(0, 25), xlabel=r"order $\alpha$", ylabel="ε", title=f"Choosing the order, T = {steps:,} steps")
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

    rng = np.random.default_rng(seed + 1)  # separate stream, so the table stays reproducible
    out.mkdir(parents=True, exist_ok=True)
    figures = {
        "sgd_clipping.svg": plot_clipping(rng),
        "sgd_subsampling_ratio.svg": plot_subsampling_ratio(),
        "sgd_rdp_curve.svg": plot_rdp_curve(),
        "sgd_eps_epochs.svg": plot_eps_epochs(),
        "sgd_conversion.svg": plot_conversion(),
    }
    for name, fig in figures.items():
        fig.tight_layout()
        fig.savefig(out / name, metadata={"Date": None})  # no timestamp, so output is reproducible
        plt.close(fig)
        print(f"saved {out / name}")


if __name__ == "__main__":
    main()
