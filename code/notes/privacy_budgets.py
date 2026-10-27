"""Compute the numbers and draw the figures of the study note notes/privacy-budgets/index.md.

    uv run python code/notes/privacy_budgets.py [--out DIR]

The note budgets k counting queries (sensitivity 1, add/remove one record) so that their total guarantee
reaches the reference (ε, δ) = (1, 10⁻⁶), under each of four definitions: pure DP, approximate DP,
concentrated DP (Dwork and Rothblum's) and zero-concentrated DP. For each way of accounting, the script finds
the noise standard deviation each query needs, prints the table the note quotes, and draws two figures into
_build/figures/notes/privacy-budgets/ (or --out): the privacy loss of the Laplace and Gaussian mechanisms,
and the noise each accounting needs as the number of queries grows. Nothing here is random.
"""

import argparse
import math
from collections.abc import Callable
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

matplotlib.rcParams["svg.hashsalt"] = "privacy_budgets"  # stable element ids, so reruns don't churn the SVGs
matplotlib.rcParams["svg.fonttype"] = "none"  # keep text as text: smaller, diffable files

NOTE = "privacy-budgets"
FIGURES = Path(__file__).resolve().parents[2] / "_build" / "figures" / "notes" / NOTE

EPS = 1.0  # the reference ε
DELTA = 1e-6  # the reference δ
KS = (1, 10, 100, 1000)  # the numbers of queries in the note's table
K = 100  # the running example's number of queries


# --- Conversions to the reference --------------------------------------------------------------------------------


def log_inv(delta: float) -> float:
    """ln(1/δ), the quantity every conversion to (ε, δ)-DP goes through."""
    return math.log(1.0 / delta)


def zcdp_epsilon(rho: float, delta: float) -> float:
    """The ε that ρ-zCDP gives at δ: ρ + 2√(ρ ln(1/δ))."""
    return rho + 2.0 * math.sqrt(rho * log_inv(delta))


def zcdp_rho_for(eps: float, delta: float) -> float:
    """The largest ρ whose conversion gives at most ε at δ: (√(ln(1/δ) + ε) - √(ln(1/δ)))²."""
    return (math.sqrt(log_inv(delta) + eps) - math.sqrt(log_inv(delta))) ** 2


def cdp_epsilon(mu: float, tau: float, delta: float) -> float:
    """The ε that (μ, τ)-CDP gives at δ: μ + τ√(2 ln(1/δ))."""
    return mu + tau * math.sqrt(2.0 * log_inv(delta))


def std_normal_cdf(x: float) -> float:
    """Φ(x), through the complementary error function so that far tails keep their precision."""
    return 0.5 * math.erfc(-x / math.sqrt(2.0))


def gaussian_delta(eps: float, s: float) -> float:
    """Exact δ(ε) of one Gaussian mechanism whose sensitivity is s standard deviations: Φ(s/2 - ε/s) - e^ε Φ(-s/2 - ε/s)."""
    return std_normal_cdf(s / 2.0 - eps / s) - math.exp(eps) * std_normal_cdf(-s / 2.0 - eps / s)


def gaussian_s_for(eps: float, delta: float) -> float:
    """The largest ratio s = Δ/σ at which the Gaussian mechanism is (ε, δ)-DP exactly (δ(ε) increases with s)."""
    return solve_increasing(lambda s: gaussian_delta(eps, s), delta)


def gaussian_epsilon(s: float, delta: float) -> float:
    """The smallest ε at which a Gaussian mechanism with ratio s is (ε, δ)-DP exactly (δ(ε) decreases with ε)."""
    return solve_increasing(lambda e: -gaussian_delta(e, s), -delta, lo=0.0, hi=50.0)


def advanced_epsilon(eps0: float, k: int, delta_prime: float) -> float:
    """Advanced composition of k (ε₀, δ₀)-DP mechanisms, as Dwork and Roth state it: ε₀√(2k ln(1/δ')) + kε₀(e^ε₀ - 1)."""
    return eps0 * math.sqrt(2.0 * k * log_inv(delta_prime)) + k * eps0 * math.expm1(eps0)


def solve_increasing(fn: Callable[[float], float], target: float, lo: float = 1e-12, hi: float = 200.0) -> float:
    """The x in (lo, hi) at which an increasing function fn reaches target, by bisection."""
    for _ in range(100):
        mid = (lo + hi) / 2.0
        if fn(mid) < target:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2.0


# --- Noise per query for each accounting ---------------------------------------------------------------------------


def laplace_sd(eps0: float) -> float:
    """Standard deviation of the Laplace noise that makes a count ε₀-DP: the scale is 1/ε₀, the sd √2/ε₀."""
    return math.sqrt(2.0) / eps0


def pure_basic(k: int) -> tuple[float, float]:
    """Laplace noise, pure DP, basic composition: ε₀ = ε/k. Returns (ε₀, noise sd)."""
    eps0 = EPS / k
    return eps0, laplace_sd(eps0)


def laplace_zcdp(k: int) -> tuple[float, float]:
    """Laplace noise accounted in zCDP: each query is (ε₀²/2)-zCDP, and kε₀²/2 must equal the reference's ρ."""
    eps0 = math.sqrt(2.0 * zcdp_rho_for(EPS, DELTA) / k)
    return eps0, laplace_sd(eps0)


def laplace_cdp(k: int) -> tuple[float, float]:
    """Laplace noise accounted in CDP: each query is (ε₀ tanh(ε₀/2), ε₀)-CDP; the means add, the τ² add."""
    eps0 = solve_increasing(lambda e: cdp_epsilon(k * e * math.tanh(e / 2.0), math.sqrt(k) * e, DELTA), EPS, hi=50.0)
    return eps0, laplace_sd(eps0)


def gauss_basic(k: int) -> tuple[float, float]:
    """Gaussian noise, approximate DP, basic composition: each query (ε/k, δ/k), calibrated exactly. Returns (s, σ)."""
    s = gaussian_s_for(EPS / k, DELTA / k)
    return s, 1.0 / s


def gauss_advanced(k: int, grid: int = 199) -> tuple[float, float, float, float]:
    """Gaussian noise, approximate DP, advanced composition. A share of δ goes to δ', the rest to kδ₀; the share is
    chosen on a grid to need the least noise. Returns (share, ε₀, δ₀, σ)."""
    best = (0.0, 0.0, 0.0, math.inf)
    for share in np.linspace(0.005, 0.995, grid):
        delta_prime = float(share) * DELTA
        delta0 = (DELTA - delta_prime) / k
        eps0 = solve_increasing(lambda e, dp=delta_prime: advanced_epsilon(e, k, dp), EPS, hi=50.0)
        sigma = 1.0 / gaussian_s_for(eps0, delta0)
        if sigma < best[3]:
            best = (float(share), eps0, delta0, sigma)
    return best


def gauss_concentrated(k: int) -> float:
    """Gaussian noise accounted in zCDP (or CDP, which gives the same conversion): kΔ²/(2σ²) = ρ. Returns σ."""
    return math.sqrt(k / (2.0 * zcdp_rho_for(EPS, DELTA)))


def gauss_exact(k: int) -> float:
    """Gaussian noise accounted exactly: k counts with noise σ are one Gaussian mechanism with s = √k/σ. Returns σ."""
    return math.sqrt(k) / gaussian_s_for(EPS, DELTA)


def normalized_rows(k: int) -> list[tuple[str, float, float, float, float]]:
    """For each accounting of k counts that meets the reference: the guarantee of one count in the reference's unit,
    (ε₀, δ₀) with δ₀ = δ/k unless the count needs no δ, and the total that basic composition of those per-count
    guarantees gives, (kε₀, kδ₀). The definition's own composition gives the reference itself, (1, 10⁻⁶)."""
    delta0 = DELTA / k
    rho0 = zcdp_rho_for(EPS, DELTA) / k
    rows: list[tuple[str, float, float]] = [
        ("Laplace, pure DP", pure_basic(k)[0], 0.0),
        ("Laplace, zCDP", laplace_zcdp(k)[0], 0.0),
        ("Gaussian, basic", EPS / k, delta0),
        ("Gaussian, advanced", gauss_advanced(k)[1], gauss_advanced(k)[2]),
        ("Gaussian, CDP or zCDP", zcdp_epsilon(rho0, delta0), delta0),
        ("Gaussian, exact", gaussian_epsilon(1.0 / gauss_exact(k), delta0), delta0),
    ]
    return [(name, e0, d0, k * e0, k * d0) for name, e0, d0 in rows]


def first_k_below(fn: Callable[[int], float], other: Callable[[int], float], k_max: int = 1000) -> int:
    """The smallest number of queries at which accounting fn needs less noise per query than accounting other,
    by bisection: once fn needs less noise, it keeps needing less (the curves cross once, as the figure shows)."""
    lo, hi = 0, k_max  # fn is not below other at lo (or lo = 0), and is below at hi
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if fn(mid) < other(mid):
            hi = mid
        else:
            lo = mid
    return hi


# --- Figures --------------------------------------------------------------------------------------------------------


def figure_losses(out: Path) -> None:
    """Left: the privacy loss ln(p/q) against the output, for Laplace (b = 1) and Gaussian (σ = 1) noise on a count
    whose two true answers are 0 and 1. Right: the distribution of the Gaussian's loss, with the tail above ε = 1."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8.6, 3.6))
    y = np.linspace(-3.0, 4.0, 400)
    lap = np.abs(y) - np.abs(y - 1.0)  # ln p/q with p centred at 1, q at 0, b = 1
    gauss = y - 0.5  # ((y)² - (y - 1)²)/2
    ax1.plot(y, lap, color="#1f77b4", label="Laplace, b = 1")
    ax1.plot(y, gauss, color="#d62728", label="Gaussian, σ = 1")
    for level in (-1.0, 1.0):
        ax1.axhline(level, color="grey", linestyle=":", linewidth=1)
    ax1.set_xlabel("released value y (true answers 1 and 0)")
    ax1.set_ylabel("privacy loss L(y)")
    ax1.set_title("The loss at each output", fontsize=10)
    ax1.legend(loc="upper left", fontsize=8)
    ax1.grid(True, alpha=0.3)

    loss = np.linspace(-3.5, 4.5, 400)
    density = np.exp(-((loss - 0.5) ** 2) / 2.0) / math.sqrt(2.0 * math.pi)
    ax2.plot(loss, density, color="#d62728")
    tail = loss >= 1.0
    ax2.fill_between(loss[tail], density[tail], color="#d62728", alpha=0.25, label="Pr[L > 1] = 0.31")
    ax2.axvline(0.5, color="black", linestyle="--", linewidth=1, label="mean ρ = 0.5")
    ax2.set_xlabel("privacy loss L")
    ax2.set_ylabel("density")
    ax2.set_title("Gaussian, σ = 1: the loss is N(0.5, 1)", fontsize=10)
    ax2.legend(loc="upper right", fontsize=8)
    ax2.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "losses.svg", metadata={"Date": None})
    plt.close(fig)


def figure_noise(out: Path) -> None:
    """The noise standard deviation each query needs for the total to meet (1, 10⁻⁶), against the number of queries."""
    ks = np.unique(np.round(np.logspace(0, 3, 40)).astype(int))
    curves: list[tuple[str, Callable[[int], float], str, str]] = [
        ("Laplace, pure DP, basic composition", lambda k: pure_basic(k)[1], "#1f77b4", "-"),
        ("Laplace, accounted in zCDP", lambda k: laplace_zcdp(k)[1], "#1f77b4", "--"),
        ("Gaussian, basic composition", lambda k: gauss_basic(k)[1], "#ff7f0e", "-"),
        ("Gaussian, advanced composition", lambda k: gauss_advanced(k, grid=49)[3], "#2ca02c", "-"),
        ("Gaussian, CDP or zCDP", gauss_concentrated, "#d62728", "-"),
        ("Gaussian, exact", gauss_exact, "black", ":"),
    ]
    fig, ax = plt.subplots(figsize=(7.0, 4.4))
    for label, fn, colour, style in curves:
        ax.plot(ks, [fn(int(k)) for k in ks], color=colour, linestyle=style, label=label)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("number of counting queries k")
    ax.set_ylabel("noise standard deviation per query")
    ax.set_title("Noise per query for a total of (ε, δ) = (1, 10⁻⁶)", fontsize=10)
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(True, which="major", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "noise_per_query.svg", metadata={"Date": None})
    plt.close(fig)


# --- The table ------------------------------------------------------------------------------------------------------


def print_table() -> None:
    """Print every number the note quotes, in the order of its sections."""
    rho = zcdp_rho_for(EPS, DELTA)
    print(f"reference: eps={EPS:g} delta={DELTA:g} ln(1/delta)={log_inv(DELTA):.4f}")
    print(f"zCDP budget: rho={rho:.5f}  check eps={zcdp_epsilon(rho, DELTA):.6f}")
    s = math.sqrt(2.0 * rho)
    print(
        f"CDP budget (Gaussian shape): mu={s * s / 2:.5f} tau={s:.5f}  check eps={cdp_epsilon(s * s / 2, s, DELTA):.6f}"
    )
    print(
        f"Gaussian sigma=1 loss tail: P(L>1)={1 - std_normal_cdf(0.5):.4f}  exact delta(1)={gaussian_delta(1.0, 1.0):.4f}"
    )

    print(f"\nk={K}")
    e, sd = pure_basic(K)
    print(f"  pure basic: eps0={e:.4f} b={1 / e:.2f} sd={sd:.2f}")
    e, sd = laplace_zcdp(K)
    print(f"  Laplace zCDP: eps0={e:.5f} rho0={e * e / 2:.3e} b={1 / e:.2f} sd={sd:.2f}")
    e, sd = laplace_cdp(K)
    print(f"  Laplace CDP: eps0={e:.5f} b={1 / e:.2f} sd={sd:.2f}")
    _, sigma = gauss_basic(K)
    print(f"  Gaussian basic: eps0={EPS / K:g} delta0={DELTA / K:.0e} sigma={sigma:.2f}")
    share, e, d0, sigma = gauss_advanced(K)
    print(
        f"  Gaussian advanced: share={share:.3f} delta'={share * DELTA:.2e} eps0={e:.5f} delta0={d0:.2e} sigma={sigma:.2f}"
    )
    sigma = gauss_concentrated(K)
    print(f"  Gaussian zCDP: rho0={rho / K:.3e} sigma={sigma:.2f}")
    print(f"  Gaussian CDP: mu0={1 / (2 * sigma**2):.3e} tau0={1 / sigma:.5f} sigma={sigma:.2f}")
    sigma_exact = gauss_exact(K)
    print(f"  Gaussian exact: s={gaussian_s_for(EPS, DELTA):.5f} sigma={sigma_exact:.2f}")
    print(f"  zCDP over exact: {gauss_concentrated(K) / sigma_exact:.3f}")

    print("\ncrossovers: the smallest k at which the first needs less noise than the second")
    basic_sd: Callable[[int], float] = lambda k: pure_basic(k)[1]
    print(f"  Gaussian exact < Laplace basic: k={first_k_below(gauss_exact, basic_sd)}")
    print(f"  Gaussian zCDP < Laplace basic: k={first_k_below(gauss_concentrated, basic_sd)}")
    print(f"  Laplace zCDP < Laplace basic: k={first_k_below(lambda k: laplace_zcdp(k)[1], basic_sd)}")
    adv: Callable[[int], float] = lambda k: gauss_advanced(k, grid=49)[3]
    print(f"  Gaussian advanced < Gaussian basic: k={first_k_below(adv, lambda k: gauss_basic(k)[1])}")
    print(f"  Gaussian advanced < Laplace basic: k={first_k_below(adv, basic_sd)}")

    print(f"\nnormalized per count (k={K}) and basic composition of the per-count guarantees")
    for name, e0, d0, et, dt in normalized_rows(K):
        print(f"  {name}: eps0={e0:.5f} delta0={d0:.2e} -> total eps={et:.3f} delta={dt:.2e}")

    print("\nnoise sd per query for (1, 1e-6)")
    print("  k     pure-basic  lap-zCDP  gauss-basic  gauss-adv  gauss-zCDP  gauss-exact")
    for k in KS:
        print(
            f"  {k:<5d} {pure_basic(k)[1]:10.2f} {laplace_zcdp(k)[1]:9.2f} {gauss_basic(k)[1]:12.2f}"
            f" {gauss_advanced(k)[3]:10.2f} {gauss_concentrated(k):11.2f} {gauss_exact(k):12.2f}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, default=FIGURES)
    args = parser.parse_args()
    out: Path = args.out
    out.mkdir(parents=True, exist_ok=True)
    print_table()
    figure_losses(out)
    figure_noise(out)


if __name__ == "__main__":
    main()
