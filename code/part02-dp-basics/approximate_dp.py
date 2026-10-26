"""Check the numbers in the course's approximate-DP chapter and draw its figure.

    uv run python code/part02-dp-basics/approximate_dp.py
    uv run python code/part02-dp-basics/approximate_dp.py --seed 1 --out /tmp/figures

Prints the numbers quoted in book-course/part02-dp-basics/03-approximate-dp.md, then saves the
chapter's figure. The Gaussian mechanism adds N(0, σ²) noise to each coordinate, with
σ = Δ₂ √(2 ln(1/δ) + ε)/ε. Its privacy loss is N(η, 2η) with η = ‖f(X) - f(X')‖₂²/(2σ²), and the
smallest δ it achieves at ε is Φ(s/2 - ε/s) - e^ε Φ(-s/2 - ε/s), where s = ‖f(X) - f(X')‖₂/σ.
"""

import argparse
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # for code/shared
from shared.figures import configure

configure("approximate_dp")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

# Figures are build output: _build/figures/<part>/, which the build copies into each book.
FIGURES = Path(__file__).resolve().parents[2] / "_build" / "figures" / Path(__file__).parent.name


# --- Gaussian tails ---------------------------------------------------------------------------


def normal_cdf(x: float) -> float:
    """Φ(x), the standard normal CDF."""
    return 0.5 * math.erfc(-x / math.sqrt(2.0))


def normal_tail(x: float) -> float:
    """Pr[Z > x] for a standard normal Z."""
    return 0.5 * math.erfc(x / math.sqrt(2.0))


# --- The Gaussian mechanism ------------------------------------------------------------------


def gaussian_sigma(eps: float, delta: float, sensitivity: float = 1.0) -> float:
    """σ = Δ₂ √(2 ln(1/δ) + ε)/ε, which the chapter proves (ε, δ)-DP for every ε > 0."""
    return sensitivity * math.sqrt(2.0 * math.log(1.0 / delta) + eps) / eps


def loss_tail(eps: float, s: float) -> float:
    """Pr[L > ε] for L ~ N(η, 2η), η = s²/2: the sufficient condition's δ."""
    eta = s * s / 2.0
    return normal_tail((eps - eta) / s)


def exact_delta(eps: float, s: float) -> float:
    """Smallest δ with (ε, δ)-DP, Φ(s/2 - ε/s) - e^ε Φ(-s/2 - ε/s), where s = Δ₂/σ."""
    return normal_cdf(s / 2.0 - eps / s) - math.exp(eps) * normal_cdf(-s / 2.0 - eps / s)


def smallest_sigma(eps: float, delta: float) -> float:
    """The smallest σ/Δ₂ whose exact δ at ε is at most `delta`, by bisection (δ falls as σ grows)."""
    lo, hi = 1e-3, 1e3
    for _ in range(200):
        mid = math.sqrt(lo * hi)
        if exact_delta(eps, 1.0 / mid) > delta:
            lo = mid
        else:
            hi = mid
    return hi


def simulate_gaussian_loss(rng: np.random.Generator, v: np.ndarray, sigma: float, reps: int) -> np.ndarray:
    """Draws of ln(p(y)/q(y)) for y = f(X) + noise, where f(X) - f(X') = v."""
    noise = rng.normal(0.0, sigma, size=(reps, v.size))
    # ln p - ln q = (‖y - f(X') ‖² - ‖y - f(X)‖²)/(2σ²) with y - f(X) = noise, y - f(X') = noise + v.
    return (np.sum((noise + v) ** 2, axis=1) - np.sum(noise**2, axis=1)) / (2.0 * sigma**2)


# --- Laplace and Gaussian for the mean of bit vectors ---------------------------------------


def laplace_rms(d: int, n: int, eps: float) -> float:
    """√E‖Y‖₂² for Y with d coordinates Lap(d/(εn)): √(2d) d/(εn)."""
    b = d / (eps * n)
    return math.sqrt(2.0 * d) * b


def gaussian_rms(d: int, n: int, eps: float, delta: float) -> float:
    """√E‖Y‖₂² for Y with d coordinates N(0, σ²), σ calibrated to Δ₂ = √d/n: √d σ."""
    return math.sqrt(d) * gaussian_sigma(eps, delta, math.sqrt(d) / n)


def simulate_rms(rng: np.random.Generator, d: int, n: int, eps: float, delta: float, reps: int) -> tuple[float, float]:
    """Root mean squared ℓ2 error of the two mechanisms, estimated from `reps` runs each."""
    lap = rng.laplace(0.0, d / (eps * n), size=(reps, d))
    gau = rng.normal(0.0, gaussian_sigma(eps, delta, math.sqrt(d) / n), size=(reps, d))
    return float(np.sqrt(np.mean(np.sum(lap**2, axis=1)))), float(np.sqrt(np.mean(np.sum(gau**2, axis=1))))


# --- Table and figure ------------------------------------------------------------------------


def floor_to(x: float, digits: int) -> float:
    """Round x down to `digits` decimals, so that a quoted lower bound stays a lower bound."""
    return math.floor(x * 10**digits) / 10**digits


def ceil_to(x: float, digits: int) -> float:
    """Round x up to `digits` decimals, so that a quoted upper bound stays an upper bound."""
    return math.ceil(x * 10**digits) / 10**digits


def print_classic_constant() -> None:
    """The two cases of the proof that σ = Δ₂ √(2 ln(1.25/δ))/ε gives (ε, δ)-DP for ε ≤ 1."""
    t = math.sqrt(2.0 * math.log(1.25 / 0.74))  # t - 1/(2t) increases in t, so these are the worst cases
    t_low, u_low = floor_to(t, 4), floor_to(t - 1.0 / (2.0 * t), 4)
    factor = ceil_to(math.exp(0.5) / (1.25 * u_low * math.sqrt(2.0 * math.pi)), 3)
    print(f"classic, delta<=0.74: t>={t_low:.4f}  u>={u_low:.4f}  factor={factor:.3f}")
    t = math.sqrt(2.0 * math.log(1.25))
    t_min, u_min = floor_to(t, 4), floor_to(t - 1.0 / (2.0 * t), 4)
    tail = ceil_to(0.5 - u_min / math.sqrt(2.0 * math.pi), 3)
    print(f"classic, delta>0.74: t>{t_min:.4f}  u>{u_min:.4f}  tail<={tail:.3f}")
    worst = max(
        exact_delta(e / 100, (e / 100) / math.sqrt(2.0 * math.log(1.25 / 10 ** (-j / 20)))) / 10 ** (-j / 20)
        for e in range(1, 101)
        for j in range(1, 400)
    )
    print(f"classic, largest exact delta / delta over a grid of eps<=1: {worst:.3f}")


def print_table(rng: np.random.Generator) -> None:
    print("== Name and shame")
    for n in (1_000, 1_000_000):
        print(f"n={n}: release prob at delta=1/n: {1 - (1 - 1 / n) ** n:.4f}")
    print("== Laplace privacy loss, eps=1")
    print(
        f"atom at +eps={0.5:.4f}  atom at -eps={0.5 * math.exp(-1):.4f}  continuous part={0.5 * (1 - math.exp(-1)):.4f}"
    )
    print("== Gaussian mechanism")
    eps, delta = 1.0, 1e-5
    sigma = gaussian_sigma(eps, delta)
    print(f"eps={eps:g} delta={delta:g}: sigma/Delta={sigma:.4f}")
    s = 1.0 / sigma
    print(f"Pr[L > eps]={loss_tail(eps, s):.3e}  exact delta={exact_delta(eps, s):.3e}")
    for e in (0.5, 1.0, 2.0):
        print(f"same sigma, eps={e:g}: exact delta={exact_delta(e, s):.3e}")
    print(f"sigma=Delta: Pr[L > 1]={loss_tail(1.0, 1.0):.4f}  exact delta(1)={exact_delta(1.0, 1.0):.4f}")
    print(f"smallest sigma/Delta for (1, 1e-5): {smallest_sigma(eps, delta):.4f}")
    print(f"sigma for (1, 1e-5) by 2 ln(1.25/delta): {math.sqrt(2 * math.log(1.25 / delta)) / eps:.4f}")
    print_classic_constant()
    v = np.zeros(3)
    v[0], v[1] = 0.6, 0.8  # ‖v‖₂ = 1
    draws = simulate_gaussian_loss(rng, v, 2.0, 200_000)
    print(
        f"loss simulation, sigma=2, |v|=1: mean={draws.mean():.4f} (eta={1 / 8:.4f})  var={draws.var():.4f} (2 eta={1 / 4:.4f})"
    )
    print("== Mean of bit vectors")
    d, n = 100, 10_000
    lap, gau = laplace_rms(d, n, eps), gaussian_rms(d, n, eps, 1e-6)
    print(f"d={d} n={n} eps=1 delta=1e-6: laplace rms={lap:.4f}  gaussian rms={gau:.4f}  ratio={lap / gau:.2f}")
    sim_lap, sim_gau = simulate_rms(rng, d, n, eps, 1e-6, 20_000)
    print(f"simulated: laplace rms={sim_lap:.4f}  gaussian rms={sim_gau:.4f}")


def plot_losses() -> Figure:
    fig, (left, right) = plt.subplots(1, 2, figsize=(9.0, 3.8))
    eps = 1.0
    # Laplace mechanism with |f(X) - f(X')| = Δ: atoms at ±ε and a density on (-ε, ε).
    grid = np.linspace(-eps, eps, 200)
    left.plot(grid, 0.25 * np.exp((grid - eps) / 2.0), color="C0", label="density on $(-\\varepsilon, \\varepsilon)$")
    for x, mass in ((eps, 0.5), (-eps, 0.5 * math.exp(-eps))):
        left.vlines(x, 0.0, mass, color="C1", linewidth=2)
        left.plot([x], [mass], "o", color="C1")
        left.annotate(f"mass {mass:.3f}", (x, mass), textcoords="offset points", xytext=(0, 6), ha="center", fontsize=8)
    left.set_xlim(-2.0, 2.0)
    left.set_ylim(0.0, 0.62)
    left.set_title("Laplace mechanism, $\\varepsilon = 1$", fontsize=10)
    left.set_xlabel("privacy loss $L$")
    left.legend(fontsize=8, loc="upper left")
    # Gaussian mechanism with σ = Δ₂ and ‖f(X) - f(X')‖₂ = Δ₂: the loss is N(1/2, 1), unbounded.
    s_ratio = 1.0
    eta = s_ratio * s_ratio / 2.0
    xs = np.linspace(-3.0, 4.0, 400)
    dens = np.exp(-((xs - eta) ** 2) / (4.0 * eta)) / math.sqrt(4.0 * math.pi * eta)
    right.plot(xs, dens, color="C0", label="density of $L$")
    tail = xs >= eps
    right.fill_between(
        xs[tail], dens[tail], color="C3", alpha=0.4, label=f"$\\Pr[L > \\varepsilon] = {loss_tail(eps, s_ratio):.3f}$"
    )
    right.axvline(eps, color="0.4", linestyle="--", linewidth=1)
    right.set_xlim(-3.0, 4.0)
    right.set_title("Gaussian mechanism, $\\sigma = \\Delta_2 f$", fontsize=10)
    right.set_xlabel("privacy loss $L$")
    right.legend(fontsize=8, loc="upper left")
    for ax in (left, right):
        ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=FIGURES)
    args = parser.parse_args()
    print_table(np.random.default_rng(args.seed))
    args.out.mkdir(parents=True, exist_ok=True)
    fig = plot_losses()
    fig.savefig(args.out / "adp-loss.svg", metadata={"Date": None})
    plt.close(fig)


if __name__ == "__main__":
    main()
