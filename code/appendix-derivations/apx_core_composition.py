"""Check the numbers in the appendix "Composing private releases" and draw its figures.

    uv run python code/appendix-derivations/apx_core_composition.py
    uv run python code/appendix-derivations/apx_core_composition.py --seed 1 --out /tmp/figures

Prints every number that core-2-composition.md quotes (it matches exactly with the default seed),
then saves the appendix's figures.
"""

import argparse
import math
import sys
from collections.abc import Callable, Sequence
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # for code/shared
from shared.figures import configure
from shared.stats import bisect, normal_cdf

configure("apx_core_composition")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from numpy.typing import NDArray

# Figures are build output: _build/figures/<part>/, which the build copies into each book.
FIGURES = Path(__file__).resolve().parents[2] / "_build" / "figures" / Path(__file__).parent.name

DELTA = 1e-5  # the total δ used throughout the page

# The grid of Rényi orders used by the accounting chapters: 1.1 to 10.9 in steps of 0.1, 12 to 63, 128, 256, 512.
ORDERS: list[float] = [1 + x / 10 for x in range(1, 100)] + [float(a) for a in range(12, 64)] + [128.0, 256.0, 512.0]

# Fraction s of the total δ spent on the k per-step δ0 in advanced composition (the rest is δ'):
# 10^(-1/4), 10^(-2/4), ..., 1e-10, the same grid as the accounting chapters.
SPLITS: list[float] = [10 ** (-x / 4) for x in range(1, 41)]


# --- Small helpers --------------------------------------------------------------------------


def normal_pdf(x: float, mean: float = 0.0, sd: float = 1.0) -> float:
    z = (x - mean) / sd
    return math.exp(-z * z / 2) / (sd * math.sqrt(2 * math.pi))


# --- Basic composition ----------------------------------------------------------------------


def laplace_count_loss(y: NDArray[np.float64], eps: float) -> NDArray[np.float64]:
    """Privacy loss of a count released with Laplace noise of scale 1/ε, centres 0 (on D) and 1 (on D')."""
    return eps * (np.abs(y - 1) - np.abs(y))


def simulate_laplace_total_loss(rng: np.random.Generator, k: int, eps: float, trials: int) -> NDArray[np.float64]:
    """Total privacy loss of k counts, each released with Laplace noise of scale 1/ε, drawn under D."""
    y = rng.laplace(0.0, 1 / eps, size=(trials, k))
    return laplace_count_loss(y, eps).sum(axis=1)


def rr_truth(eps: float) -> float:
    """Randomized response at level ε tells the truth with probability e^ε / (1 + e^ε)."""
    return math.exp(eps) / (1 + math.exp(eps))


def adaptive_rr_table(
    eps1: float, eps_after_1: float, eps_after_0: float
) -> list[tuple[int, int, float, float, float]]:
    """Two randomized-response answers about one bit, the second's level chosen from the first answer.

    Returns rows (y1, y2, probability when the bit is 1, probability when the bit is 0, privacy loss)."""
    rows: list[tuple[int, int, float, float, float]] = []
    for y1 in (1, 0):
        eps2 = eps_after_1 if y1 == 1 else eps_after_0
        for y2 in (1, 0):
            t1, t2 = rr_truth(eps1), rr_truth(eps2)
            p = (t1 if y1 == 1 else 1 - t1) * (t2 if y2 == 1 else 1 - t2)
            q = (1 - t1 if y1 == 1 else t1) * (1 - t2 if y2 == 1 else t2)
            rows.append((y1, y2, p, q, math.log(p / q)))
    return rows


def two_output_pair(eps: float, gap: float) -> tuple[float, float, float]:
    """A pair that is (ε, δ)-DP with small δ although its loss exceeds ε half the time.

    Output A has p = 1/2 and q = e^{-(ε + gap)}/2; output B takes the rest. Returns (δ for P against Q,
    δ for Q against P, Pr_P[L > ε])."""
    pa, qa = 0.5, 0.5 * math.exp(-(eps + gap))
    pb, qb = 1 - pa, 1 - qa
    forward = max(pa - math.exp(eps) * qa, 0.0) + max(pb - math.exp(eps) * qb, 0.0)
    backward = max(qa - math.exp(eps) * pa, 0.0) + max(qb - math.exp(eps) * pb, 0.0)
    tail = (pa if math.log(pa / qa) > eps else 0.0) + (pb if math.log(pb / qb) > eps else 0.0)
    return forward, backward, tail


# --- Advanced composition -------------------------------------------------------------------


def mean_loss_bound(eps: float) -> float:
    """The mean privacy loss of an ε-DP step is at most ε(e^ε - 1)."""
    return eps * math.expm1(eps)


def rr_mean_loss(eps: float) -> float:
    """The mean privacy loss of randomized response at level ε: ε(2t - 1) = ε tanh(ε/2)."""
    return eps * (2 * rr_truth(eps) - 1)


def azuma_term(eps: float, k: int, delta_prime: float) -> float:
    return eps * math.sqrt(2 * k * math.log(1 / delta_prime))


def advanced_composition(eps: float, k: int, delta_prime: float) -> float:
    """ε√(2k ln(1/δ')) + kε(e^ε - 1), the bound derived on the page."""
    return azuma_term(eps, k, delta_prime) + k * mean_loss_bound(eps)


def advanced_composition_tanh(eps: float, k: int, delta_prime: float) -> float:
    """The same with the sharp mean loss ε tanh(ε/2), which the chapter derives."""
    return azuma_term(eps, k, delta_prime) + k * eps * math.tanh(eps / 2)


def log_binom_pmf(k: int, j: int, p: float) -> float:
    return math.lgamma(k + 1) - math.lgamma(j + 1) - math.lgamma(k - j + 1) + j * math.log(p) + (k - j) * math.log1p(-p)


def rr_loss_tail(eps: float, k: int, level: float) -> float:
    """Pr[L > level] under D for k randomized responses at ε: L = (k - 2J)ε with J ~ Bin(k, 1 - t)."""
    lie = 1 - rr_truth(eps)
    return sum(math.exp(log_binom_pmf(k, j, lie)) for j in range(k + 1) if (k - 2 * j) * eps > level)


def rr_exact_delta(eps: float, k: int, eps_prime: float) -> float:
    """Exact δ(ε') of k randomized responses at ε: the average of (1 - e^{ε' - L})_+ under D."""
    lie = 1 - rr_truth(eps)
    total = 0.0
    for j in range(k + 1):
        loss = (k - 2 * j) * eps
        if loss > eps_prime:
            total += math.exp(log_binom_pmf(k, j, lie)) * -math.expm1(eps_prime - loss)
    return total


def rr_exact_eps(eps: float, k: int, delta: float) -> float:
    """Smallest ε' with δ(ε') ≤ delta for k randomized responses: optimal composition of k ε-DP steps (δ = 0)."""
    if rr_exact_delta(eps, k, 0.0) <= delta:
        return 0.0
    return bisect(lambda e: rr_exact_delta(eps, k, e) - delta, 0.0, k * eps)


def rr_walks(rng: np.random.Generator, eps: float, k: int, paths: int) -> NDArray[np.float64]:
    """Running total privacy loss of k randomized responses at ε, drawn under D: steps +ε (truth) or -ε (lie)."""
    steps = np.where(rng.random((paths, k)) < rr_truth(eps), eps, -eps)
    return np.concatenate([np.zeros((paths, 1)), np.cumsum(steps, axis=1)], axis=1)


# --- The Gaussian mechanism: k releases are one release -------------------------------------


def gauss_delta(eps: float, mu: float) -> float:
    """Exact δ(ε) of one Gaussian release whose two centres are μ = Δ/σ noise standard deviations apart:
    Φ(μ/2 - ε/μ) - e^ε Φ(-μ/2 - ε/μ)."""
    first = normal_cdf(mu / 2 - eps / mu)
    tail = normal_cdf(-mu / 2 - eps / mu)
    second = 0.0 if tail == 0.0 else math.exp(min(eps + math.log(tail), 700.0))
    return max(first - second, 0.0)


def gauss_eps(mu: float, delta: float) -> float:
    """Smallest ε ≥ 0 with δ(ε) ≤ delta for one Gaussian release at μ."""
    if gauss_delta(0.0, mu) <= delta:
        return 0.0
    hi = 1.0
    while gauss_delta(hi, mu) > delta:
        hi *= 2
    return bisect(lambda e: gauss_delta(e, mu) - delta, 0.0, hi)


def gauss_sigma(eps: float, delta: float, sensitivity: float = 1.0) -> float:
    """Smallest noise standard deviation σ that makes one Gaussian release (ε, δ)-DP (exact calibration)."""
    log_mu = bisect(lambda lm: gauss_delta(eps, math.exp(lm)) - delta, -12.0, 8.0)
    return sensitivity / math.exp(log_mu)


def gauss_composed_eps(k: int, sigma: float, delta: float, sensitivity: float = 1.0) -> float:
    """Exact ε of k Gaussian releases: one release with μ = √k Δ/σ."""
    return gauss_eps(math.sqrt(k) * sensitivity / sigma, delta)


def simulate_gauss_total_loss(
    rng: np.random.Generator, k: int, sigma: float, trials: int, sensitivity: float = 1.0
) -> NDArray[np.float64]:
    """Total privacy loss of k counts with Gaussian noise σ, drawn under D (centres 0 on D, Δ on D')."""
    z = rng.normal(0.0, sigma, size=(trials, k))
    loss = ((z - sensitivity) ** 2 - z**2) / (2 * sigma**2)
    return loss.sum(axis=1)


def delta_from_losses(losses: NDArray[np.float64], eps: float) -> float:
    """Monte Carlo estimate of δ(ε) = E[(1 - e^{ε - L})_+] from samples of the privacy loss."""
    return float(np.mean(np.maximum(-np.expm1(eps - losses), 0.0)))


# --- Rényi DP of the Gaussian ---------------------------------------------------------------


def gauss_rdp(alpha: float, sigma: float, sensitivity: float = 1.0) -> float:
    """D_α between N(0, σ²) and N(Δ, σ²): αΔ² / (2σ²)."""
    return alpha * sensitivity**2 / (2 * sigma**2)


def gauss_rdp_numeric(alpha: float, sigma: float, sensitivity: float = 1.0) -> float:
    """The same divergence, from the integral of p^α q^{1-α} on a fine grid (no closed form used)."""
    lo, hi = -40 * sigma - abs(alpha) * sensitivity, 40 * sigma + abs(alpha) * sensitivity
    y = np.linspace(lo, hi, 400_001)
    log_p = -(y**2) / (2 * sigma**2)
    log_q = -((y - sensitivity) ** 2) / (2 * sigma**2)
    integrand = np.exp(alpha * log_p + (1 - alpha) * log_q) / (sigma * math.sqrt(2 * math.pi))
    integral = float(np.sum(integrand) * (y[1] - y[0]))
    return math.log(integral) / (alpha - 1)


def rdp_to_dp(curve: Callable[[float], float], delta: float, orders: Sequence[float] = ORDERS) -> tuple[float, float]:
    """ε = min over the orders of ε_RDP(α) + ln(1/δ)/(α - 1); returns (ε, best order)."""
    return min((curve(a) + math.log(1 / delta) / (a - 1), a) for a in orders)


def gauss_rdp_composed_eps(k: int, sigma: float, delta: float) -> tuple[float, float]:
    """RDP accounting for k Gaussian releases: the curves add, then convert once."""
    return rdp_to_dp(lambda a: k * gauss_rdp(a, sigma), delta)


def gauss_rdp_closed_form(k: int, sigma: float, delta: float) -> float:
    """Continuous optimum over α of kα/(2σ²) + ln(1/δ)/(α - 1): k/(2σ²) + √(2k ln(1/δ))/σ."""
    return k / (2 * sigma**2) + math.sqrt(2 * k * math.log(1 / delta)) / sigma


# --- Comparing the accountants on one mechanism ---------------------------------------------


def gauss_basic_eps(k: int, sigma: float, delta: float) -> float:
    """Basic composition: each release is (ε0, δ/k) on its exact curve, and the ε0 add up."""
    return k * gauss_eps(1 / sigma, delta / k)


def gauss_advanced_eps(k: int, sigma: float, delta: float, tanh: bool = False) -> tuple[float, float]:
    """Advanced composition: each release is (ε0, sδ/k) on its exact curve and δ' = (1 - s)δ;
    returns (ε, the best fraction s on SPLITS). tanh=False is the e^ε - 1 form derived on the page."""
    bound = advanced_composition_tanh if tanh else advanced_composition
    best = (math.inf, 0.0)
    for s in SPLITS:
        eps0 = gauss_eps(1 / sigma, s * delta / k)
        best = min(best, (bound(eps0, k, (1 - s) * delta), s))
    return best


# --- Printed numbers ------------------------------------------------------------------------


def print_table(rng: np.random.Generator) -> None:
    print("[Basic: 10 Laplace counts at eps = 0.1 each]")
    k, eps = 10, 0.1
    losses = simulate_laplace_total_loss(rng, k, eps, trials=1_000_000)
    at_max = float(np.mean(losses >= k * eps - 1e-9))
    print(
        f"  total eps={k * eps:.3f}  simulated max loss={losses.max():.4f}  "
        f"Pr[loss = k eps] simulated={at_max:.6f} exact={0.5**k:.6f}  "
        f"mean={losses.mean():.4f} (exact {k * (eps + math.exp(-eps) - 1):.4f})"
    )

    print("[Basic: adaptive randomized response, ln 3 then ln 3 after a 1 or ln 2 after a 0]")
    for y1, y2, p, q, loss in adaptive_rr_table(math.log(3), math.log(3), math.log(2)):
        print(f"  ({y1},{y2}): bit 1={p:.4f}  bit 0={q:.4f}  loss={loss:+.4f}")

    fwd, bwd, tail = two_output_pair(1.0, 0.1)
    print(
        f"[Basic: tail condition is only sufficient] delta forward={fwd:.4f}  backward={bwd:.4f}  Pr[L > 1]={tail:.2f}"
    )

    print("[Advanced: mean loss of one step]")
    for e in (0.1, 1.0):
        print(f"  eps={e:g}: bound eps(e^eps-1)={mean_loss_bound(e):.4f}  randomized response={rr_mean_loss(e):.4f}")

    print("[Advanced: 100 counts at eps = 0.1, delta' = 1e-5]")
    k, eps = 100, 0.1
    adv = advanced_composition(eps, k, DELTA)
    print(
        f"  basic={k * eps:.4f}  azuma={azuma_term(eps, k, DELTA):.4f}  mean term={k * mean_loss_bound(eps):.4f}  "
        f"advanced={adv:.4f}  tanh form={advanced_composition_tanh(eps, k, DELTA):.4f}  "
        f"optimal={rr_exact_eps(eps, k, DELTA):.4f}"
    )
    print(
        f"  randomized response: mean loss={k * rr_mean_loss(eps):.4f}  "
        f"sd={eps * math.sqrt(k * (1 - math.tanh(eps / 2) ** 2)):.4f}  "
        f"Pr[L > {adv:.4f}]={rr_loss_tail(eps, k, adv):.3e}  exact delta={rr_exact_delta(eps, k, adv):.3e}"
    )
    print(
        f"  crossover: advanced beats basic once k > 2 ln(1/delta')/(2 - e^eps)^2 = "
        f"{2 * math.log(1 / DELTA) / (2 - math.exp(eps)) ** 2:.2f}"
    )
    walks = rr_walks(rng, eps, 1000, paths=100_000)
    for kk in (100, 1000):
        final = walks[:, kk]
        print(
            f"  simulated walk k={kk}: mean={final.mean():.4f} (exact {kk * rr_mean_loss(eps):.4f})  "
            f"sd={final.std():.4f} (exact {eps * math.sqrt(kk * (1 - math.tanh(eps / 2) ** 2)):.4f})  "
            f"max={final.max():.1f} (basic {kk * eps:g}, advanced {advanced_composition(eps, kk, DELTA):.4f})"
        )

    print("[Gaussian: k releases are one release with sqrt(k) times the shift]")
    trials = 400_000
    for kk, sigma in ((16, 4.0),):
        mu = 1 / sigma
        sim = simulate_gauss_total_loss(rng, kk, sigma, trials)
        print(
            f"  k={kk} sigma={sigma:g}: loss mean={sim.mean():.4f} (k mu^2/2={kk * mu**2 / 2:.4f})  "
            f"var={sim.var():.4f} (k mu^2={kk * mu**2:.4f})  "
            f"delta(1) simulated={delta_from_losses(sim, 1.0):.4f} formula={gauss_delta(1.0, math.sqrt(kk) * mu):.4f}"
        )
    s1 = gauss_sigma(1.0, DELTA)
    print(f"  one release at (1, 1e-5): sigma_1={s1:.4f}")
    for kk in (10, 100, 1000):
        sk = math.sqrt(kk) * s1
        split = gauss_sigma(1.0 / kk, DELTA / kk)
        print(
            f"  k={kk}: sigma_k=sqrt(k) sigma_1={sk:.4f}  exact eps check={gauss_composed_eps(kk, sk, DELTA):.4f}  "
            f"basic split ({1 / kk:g}, {DELTA / kk:.0e}) each: sigma={split:.4f}  ratio={split / sk:.2f}"
        )
    sig = 7.0318
    print(f"  10 counts each (0.5, 1e-5), sigma={sig}: exact eps at 1e-5={gauss_composed_eps(10, sig, DELTA):.4f}")

    print("[RDP of the Gaussian]")
    for alpha, sigma in ((2.0, 1.0), (10.0, 1.0), (2.0, 4.0)):
        print(
            f"  alpha={alpha:g} sigma={sigma:g}: formula={gauss_rdp(alpha, sigma):.4f}  "
            f"integral={gauss_rdp_numeric(alpha, sigma):.4f}  moment e^((a-1)D)={math.exp((alpha - 1) * gauss_rdp(alpha, sigma)):.4g}"
        )
    for kk, sigma in ((1, 1.0), (1000, 20.0)):
        e_grid, a_grid = gauss_rdp_composed_eps(kk, sigma, DELTA)
        print(
            f"  k={kk} sigma={sigma:g}: RDP eps={e_grid:.4f} at alpha={a_grid:g}  "
            f"closed form={gauss_rdp_closed_form(kk, sigma, DELTA):.4f}  exact={gauss_composed_eps(kk, sigma, DELTA):.4f}"
        )

    print("[Compare: Gaussian releases with sigma = 20, sensitivity 1, total delta = 1e-5]")
    for kk in (1, 10, 100, 1000, 10_000):
        adv_eps, s = gauss_advanced_eps(kk, 20.0, DELTA)
        rdp_eps, a = gauss_rdp_composed_eps(kk, 20.0, DELTA)
        ex = gauss_composed_eps(kk, 20.0, DELTA)
        print(
            f"  k={kk}: basic={gauss_basic_eps(kk, 20.0, DELTA):.3f}  advanced={adv_eps:.3f} (s={s:.2g})  "
            f"RDP={rdp_eps:.3f} (alpha={a:g})  exact={ex:.3f}  RDP/exact={rdp_eps / ex:.2f}"
        )
    print(f"  k=1000: advanced, tanh form={gauss_advanced_eps(1000, 20.0, DELTA, tanh=True)[0]:.3f}")


# --- Figures --------------------------------------------------------------------------------


def plot_product(eps: float = 0.5) -> Figure:
    """Basic composition as a sum of bounded losses: one Laplace count, and the total of two."""
    y = np.linspace(-6.0, 7.0, 521)
    fig, (left, right) = plt.subplots(1, 2, figsize=(10, 4))
    left.plot(y, laplace_count_loss(y, eps), "C0", linewidth=2, label=r"$L(y)$, one count")
    left.axhline(eps, color="gray", linestyle=":", label=r"$\pm\varepsilon$")
    left.axhline(-eps, color="gray", linestyle=":")
    left.axvline(0, color="k", linewidth=0.5)
    left.axvline(1, color="k", linewidth=0.5)
    left.set(
        xlabel="output y (true counts 0 on D, 1 on D')",
        ylabel="privacy loss",
        ylim=(-1.2, 1.2),
        title=f"One Laplace count, ε = {eps:g}: |L| ≤ ε",
    )
    left.legend(loc="upper right")

    y1, y2 = np.meshgrid(y, y)
    total = laplace_count_loss(y1, eps) + laplace_count_loss(y2, eps)
    mesh = right.pcolormesh(y1, y2, total, cmap="RdBu_r", vmin=-2 * eps, vmax=2 * eps, shading="auto")
    for x, yy, text in (
        (-3.0, -3.0, f"+2ε = {2 * eps:g}"),
        (4.0, 4.0, f"−2ε = {-2 * eps:g}"),
        (-3.0, 4.0, "0"),
        (4.0, -3.0, "0"),
    ):
        right.text(x, yy, text, ha="center", va="center", fontsize=11, color="k")
    fig.colorbar(mesh, ax=right, label=r"$L_1(y_1) + L_2(y_2)$")
    right.set(
        xlabel=r"first output $y_1$",
        ylabel=r"second output $y_2$",
        title=f"Two counts: the losses add, |L| ≤ 2ε = {2 * eps:g}",
    )
    return fig


def plot_walk(rng: np.random.Generator, eps: float = 0.1, k: int = 1000, paths: int = 25) -> Figure:
    """The total privacy loss of k randomized responses as a random walk, against the composition bounds."""
    walks = rr_walks(rng, eps, k, paths)
    steps = np.arange(k + 1)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for w in walks:
        ax.plot(steps, w, color="C0", alpha=0.35, linewidth=0.8)
    ax.plot(steps, eps * steps, "k--", label=r"basic composition $i\varepsilon$")
    adv = [math.nan] + [advanced_composition(eps, i, DELTA) for i in range(1, k + 1)]
    ax.plot(steps, adv, "C3", linewidth=2, label="advanced composition bound, δ' = 1e-5")
    mean = rr_mean_loss(eps) * steps
    sd = eps * np.sqrt(steps * (1 - math.tanh(eps / 2) ** 2))
    ax.plot(steps, mean, "C2", linewidth=2, label=r"mean $i\varepsilon\tanh(\varepsilon/2)$")
    ax.fill_between(
        steps, mean - 2 * sd, mean + 2 * sd, color="C2", alpha=0.15, label="mean ± 2 sd (width grows like √i)"
    )
    ax.set(
        xlim=(0, k),
        ylim=(-12, 40),
        xlabel="number of releases i",
        ylabel="total privacy loss",
        title=f"Randomized response at ε = {eps:g}: {paths} runs of the loss, δ' = 1e-5",
    )
    ax.legend(loc="upper left", fontsize=8)
    return fig


def plot_gauss_sum(rng: np.random.Generator, sigma: float = 4.0, ks: Sequence[int] = (1, 4, 16)) -> Figure:
    """Simulated total loss of k Gaussian releases against N(kμ²/2, kμ²), the loss of one release at √k μ."""
    mu = 1 / sigma
    fig, ax = plt.subplots(figsize=(7, 4.5))
    grid = np.linspace(-4.0, 6.0, 801)
    for i, kk in enumerate(ks):
        sim = simulate_gauss_total_loss(rng, kk, sigma, trials=200_000)
        m, sd = kk * mu**2 / 2, math.sqrt(kk) * mu
        ax.hist(sim, bins=120, range=(-4.0, 6.0), density=True, color=f"C{i}", alpha=0.3)
        ax.plot(
            grid,
            [normal_pdf(float(g), m, sd) for g in grid],
            color=f"C{i}",
            linewidth=2,
            label=f"k = {kk}: one release with μ = √{kk}/{sigma:g} = {math.sqrt(kk) * mu:g}",
        )
    ax.axvline(1.0, color="gray", linestyle=":", label="ε = 1")
    ax.set(
        xlabel="total privacy loss",
        ylabel="density",
        title=f"k Gaussian counts with σ = {sigma:g}: simulated (bars) and N(kμ²/2, kμ²) (lines)",
    )
    ax.legend(fontsize=8)
    return fig


def plot_rdp_square(alpha: float = 2.0, sigma: float = 1.0, sensitivity: float = 1.0) -> Figure:
    """The integrand p^α q^{1-α} is a normal bump centred at (1 - α)Δ with area e^{α(α-1)Δ²/(2σ²)}."""
    y = np.linspace(-5.0, 5.0, 1001)
    p = np.exp(-(y**2) / (2 * sigma**2)) / (sigma * math.sqrt(2 * math.pi))
    q = np.exp(-((y - sensitivity) ** 2) / (2 * sigma**2)) / (sigma * math.sqrt(2 * math.pi))
    integrand = p**alpha * q ** (1 - alpha)
    area = math.exp(alpha * (alpha - 1) * sensitivity**2 / (2 * sigma**2))
    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.plot(y, p, "C0", linewidth=2, label="p: N(0, 1), output on D")
    ax.plot(y, q, "C1", linewidth=2, label="q: N(1, 1), output on D'")
    ax.fill_between(y, integrand, color="C2", alpha=0.3)
    ax.plot(
        y, integrand, "C2", linewidth=2, label=rf"$p^{{{alpha:g}}} q^{{{1 - alpha:g}}}$: area $e^{{1}}$ = {area:.4f}"
    )
    centre = (1 - alpha) * sensitivity
    ax.axvline(centre, color="C2", linestyle=":", label=rf"centre $(1 - \alpha)\Delta$ = {centre:g}")
    ax.set(
        xlabel="output y",
        ylabel="density",
        title=rf"Completing the square at $\alpha$ = {alpha:g}, $\Delta$ = {sensitivity:g}, $\sigma$ = {sigma:g}",
    )
    ax.legend(fontsize=8, loc="upper right")
    return fig


def plot_compare(sigma: float = 20.0, delta: float = DELTA) -> Figure:
    """Total ε against the number of Gaussian releases, under four accountants."""
    ks = sorted({round(10 ** (x / 8)) for x in range(33)})
    basic = [gauss_basic_eps(k, sigma, delta) for k in ks]
    adv = [gauss_advanced_eps(k, sigma, delta)[0] for k in ks]
    rdp = [gauss_rdp_composed_eps(k, sigma, delta)[0] for k in ks]
    exact = [gauss_composed_eps(k, sigma, delta) for k in ks]
    fig, (left, right) = plt.subplots(1, 2, figsize=(11, 4.5))
    left.plot(ks, basic, "k--", label="basic: (ε0, δ/k) each, ε0 added")
    left.plot(ks, adv, "C3-.", label="advanced: best split of δ")
    left.plot(ks, rdp, "C1", label="RDP: curves added, converted once")
    left.plot(ks, exact, "C0", linewidth=2.5, label="exact: one Gaussian with √k times the shift")
    left.set(
        xscale="log",
        yscale="log",
        xlabel="number of releases k",
        ylabel="total ε at δ = 1e-5",
        title=f"Gaussian releases, Δ = 1, σ = {sigma:g}",
    )
    left.legend(fontsize=8)
    right.plot(ks, [b / e for b, e in zip(basic, exact, strict=True)], "k--", label="basic")
    right.plot(ks, [a / e for a, e in zip(adv, exact, strict=True)], "C3-.", label="advanced")
    right.plot(ks, [r / e for r, e in zip(rdp, exact, strict=True)], "C1", label="RDP")
    right.axhline(1.0, color="C0", linewidth=2.5, label="exact")
    right.set(
        xscale="log",
        yscale="log",
        xlabel="number of releases k",
        ylabel="ε divided by the exact ε",
        title="How far each accountant overstates ε",
    )
    right.legend(fontsize=8)
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
        "ccomp_product.svg": plot_product(),
        "ccomp_walk.svg": plot_walk(rng),
        "ccomp_gauss_sum.svg": plot_gauss_sum(rng),
        "ccomp_rdp_square.svg": plot_rdp_square(),
        "ccomp_compare.svg": plot_compare(),
    }
    for name, fig in figures.items():
        fig.tight_layout()
        fig.savefig(out / name, metadata={"Date": None})  # no timestamp, so output is reproducible
        plt.close(fig)
        print(f"saved {out / name}")


if __name__ == "__main__":
    main()
