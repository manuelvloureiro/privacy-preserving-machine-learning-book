"""Check the numbers in the course's private deep learning chapter and draw its figure.

    uv run python code/part04-learning-statistics/private_deep_learning.py
    uv run python code/part04-learning-statistics/private_deep_learning.py --out /tmp/figures

Prints the privacy parameter ε of DP-SGD (Poisson sampling rate q, noise multiplier σ, target δ)
by two accountings of the same algorithm: amplification plus advanced composition, and the
moments accountant, which composes the log-moments of the privacy loss and converts once with
Markov's inequality. Also prints the number of teachers the private mode needs.

The figure plots ε against the number of epochs for both accountings. Nothing here is random,
so --seed is accepted for a uniform interface and ignored.
"""

import argparse
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
matplotlib.rcParams["svg.hashsalt"] = "private_deep_learning"  # stable element ids across reruns
matplotlib.rcParams["svg.fonttype"] = "none"  # keep text as text

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

# Figures are build output: _build/figures/<part>/, which the build copies into each book.
FIGURES = Path(__file__).resolve().parents[2] / "_build" / "figures" / Path(__file__).parent.name

Q, SIGMA, DELTA = 0.01, 4.0, 1e-5
MAX_ORDER = 256  # moments λ = 1, ..., MAX_ORDER


# --- Gaussian mechanism and its moments ------------------------------------------------------


def gaussian_epsilon(rho: float, delta: float) -> float:
    """ε = ρ + 2√(ρ ln(1/δ)) for privacy-loss moments λ(λ+1)ρ, optimized over real λ > 0."""
    return rho + 2.0 * math.sqrt(rho * math.log(1.0 / delta))


def subsampled_log_moment(q: float, sigma: float, order: int) -> float:
    """ln A_α for integer α = order + 1, A_α = Σ_k C(α,k)(1-q)^(α-k) q^k exp((k² - k)/(2σ²))."""
    alpha = order + 1
    terms = [
        math.lgamma(alpha + 1)
        - math.lgamma(k + 1)
        - math.lgamma(alpha - k + 1)
        + (alpha - k) * math.log1p(-q)
        + (k * math.log(q) if k > 0 else 0.0)
        + (k * k - k) / (2.0 * sigma**2)
        for k in range(alpha + 1)
    ]
    top = max(terms)
    return top + math.log(sum(math.exp(t - top) for t in terms))


def accountant_epsilon(q: float, sigma: float, steps: int, delta: float) -> float:
    """min over λ of (T ln A_(λ+1) + ln(1/δ))/λ: compose the moments, convert once."""
    return min(
        (steps * subsampled_log_moment(q, sigma, lam) + math.log(1.0 / delta)) / lam for lam in range(1, MAX_ORDER + 1)
    )


def composition_epsilon(q: float, sigma: float, steps: int, delta: float) -> float:
    """Per step (ε0, δ0) Gaussian, amplified by Poisson sampling, then advanced composition.

    δ is split as δ/2 for the composition slack and δ/2 over the T steps: each amplified step has
    failure q δ0 = δ/(2T). Advanced composition is ε1 √(2T ln(1/δ')) + T ε1 tanh(ε1/2).
    """
    delta0 = delta / (2.0 * steps * q)
    eps0 = gaussian_epsilon(1.0 / (2.0 * sigma**2), delta0)
    eps1 = math.log1p(q * math.expm1(eps0))
    return eps1 * math.sqrt(2.0 * steps * math.log(2.0 / delta)) + steps * eps1 * math.tanh(eps1 / 2.0)


def teachers_needed(eps: float, delta: float, beta: float) -> tuple[float, int]:
    """Threshold t = 2 + (2/ε) ln(1/(2δ)) and the teachers m ≥ max(4t + (8/ε) ln(1/β), 32 ln(2/β))."""
    t = 2.0 + (2.0 / eps) * math.log(1.0 / (2.0 * delta))
    m = max(4.0 * t + (8.0 / eps) * math.log(1.0 / beta), 32.0 * math.log(2.0 / beta))
    return t, math.ceil(m)


# --- Table and figure ------------------------------------------------------------------------


def print_table() -> None:
    print(f"== DP-SGD accounting, q={Q:g} sigma={SIGMA:g} delta={DELTA:g}")
    for epochs in (1, 10, 100, 400):
        steps = round(epochs / Q)
        comp = composition_epsilon(Q, SIGMA, steps, DELTA)
        acc = accountant_epsilon(Q, SIGMA, steps, DELTA)
        print(f"epochs={epochs} T={steps}: composition eps={comp:.2f}  accountant eps={acc:.2f}")
    print("== Private mode")
    t, m = teachers_needed(1.0, 1e-5, 0.01)
    print(f"eps=1 delta=1e-05 beta=0.01: threshold={t:.2f}  teachers={m}")


def plot_accounting() -> Figure:
    epochs = np.unique(np.round(np.logspace(0, math.log10(400), 30)).astype(int))
    comp = [composition_epsilon(Q, SIGMA, round(float(e) / Q), DELTA) for e in epochs]
    acc = [accountant_epsilon(Q, SIGMA, round(float(e) / Q), DELTA) for e in epochs]
    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    ax.plot(epochs, comp, "--", color="C3", label="amplification + advanced composition")
    ax.plot(epochs, acc, "-", color="C0", label="moments accountant")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"epochs $Tq$")
    ax.set_ylabel(r"$\varepsilon$ at $\delta = 10^{-5}$")
    ax.legend(fontsize=9)
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    return fig


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=FIGURES)
    args = parser.parse_args()
    print_table()
    args.out.mkdir(parents=True, exist_ok=True)
    fig = plot_accounting()
    fig.savefig(args.out / "dpdl-accounting.svg", metadata={"Date": None})
    plt.close(fig)


if __name__ == "__main__":
    main()
