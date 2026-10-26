"""Draw the figure of the course's private ERM chapter and print the numbers behind it.

    uv run python code/part04-learning-statistics/private_erm.py
    uv run python code/part04-learning-statistics/private_erm.py --seed 1 --out /tmp/figures

Fits a polynomial of degree n - 1 to n points by least squares, with and without the ridge
penalty (λ/2)‖θ‖², on a dataset and on a neighbour that moves one point. Without the penalty the
fit interpolates and swings far from the moved point; with it, the two fits stay close, which is
the sensitivity bound that output perturbation relies on. The noise scale of private SGD is
printed too.
"""

import argparse
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # for code/shared
from shared.figures import configure

configure("private_erm")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from numpy.typing import NDArray

# Figures are build output: _build/figures/<part>/, which the build copies into each book.
FIGURES = Path(__file__).resolve().parents[2] / "_build" / "figures" / Path(__file__).parent.name

N_POINTS = 10
NOISE = 0.15
LAMBDA = 1e-2
MOVE = 0.8  # how far the neighbour moves the label of one point


# --- Regularized polynomial regression -------------------------------------------------------


def features(x: NDArray[np.float64], degree: int) -> NDArray[np.float64]:
    """The monomials 1, x, ..., x^degree of each point, one row per point."""
    return np.vander(x, degree + 1, increasing=True)


def fit(x: NDArray[np.float64], y: NDArray[np.float64], degree: int, lam: float) -> NDArray[np.float64]:
    """argmin_θ Σ (⟨φ(x_i), θ⟩ - y_i)² + (λ/2)‖θ‖², the least-norm least-squares fit when λ = 0."""
    phi = features(x, degree)
    if lam == 0.0:
        theta, *_ = np.linalg.lstsq(phi, y, rcond=None)
        return theta
    # Setting the gradient 2Φᵀ(Φθ - y) + λθ to zero gives (2ΦᵀΦ + λI)θ = 2Φᵀy.
    return np.linalg.solve(2.0 * phi.T @ phi + lam * np.eye(degree + 1), 2.0 * phi.T @ y)


def dataset(rng: np.random.Generator) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """n points on a sine curve with Gaussian noise, features evenly spread over [0, 1]."""
    x = (np.arange(N_POINTS) + rng.uniform(0.2, 0.8, size=N_POINTS)) / N_POINTS
    y = np.sin(2.0 * np.pi * x) + NOISE * rng.standard_normal(N_POINTS)
    return x, y


def neighbour(y: NDArray[np.float64]) -> NDArray[np.float64]:
    """The same dataset with the label of the middle point raised by MOVE."""
    moved = y.copy()
    moved[N_POINTS // 2] += MOVE
    return moved


def curve_gap(x: NDArray[np.float64], y: NDArray[np.float64], lam: float) -> tuple[float, float]:
    """‖θ - θ'‖ and the largest gap between the two fitted curves on [0, 1]."""
    degree = N_POINTS - 1
    theta = fit(x, y, degree, lam)
    theta_n = fit(x, neighbour(y), degree, lam)
    grid = np.linspace(0.0, 1.0, 401)
    phi = features(grid, degree)
    return float(np.linalg.norm(theta - theta_n)), float(np.max(np.abs(phi @ (theta - theta_n))))


def sgd_noise_over_n(n: int, eps: float, delta: float) -> float:
    """σ/(nL) = 16 √(ln(3.3n/δ) ln(2/δ))/ε, the per-step noise of private SGD relative to nL."""
    return 16.0 * math.sqrt(math.log(3.3 * n / delta) * math.log(2.0 / delta)) / eps


def sgd_risk_per_record(n: int, d: int, eps: float, delta: float) -> float:
    """The excess-risk bound of private SGD divided by n‖C‖L: (1 + √d σ/(nL))/n."""
    return (1.0 + math.sqrt(d) * sgd_noise_over_n(n, eps, delta)) / n


# --- Table and figure ------------------------------------------------------------------------


def print_table(rng: np.random.Generator) -> None:
    print("== Output perturbation: one moved point")
    x, y = dataset(rng)
    for lam in (0.0, LAMBDA):
        dtheta, dcurve = curve_gap(x, y, lam)
        print(f"lambda={lam:g}: |theta - theta'|={dtheta:.3g}  max curve gap={dcurve:.3g}")
    print("== Private SGD noise")
    n, eps, delta = 10_000, 1.0, 1e-6
    print(f"n={n} eps={eps:g} delta={delta:g}: sigma/(nL)={sgd_noise_over_n(n, eps, delta):.1f}")
    d = 100
    sqrt_d_noise = math.sqrt(d) * sgd_noise_over_n(n, eps, delta)
    print(f"d={d}: sqrt(d) sigma/(nL)={sqrt_d_noise:.0f}  bound per record={sgd_risk_per_record(n, d, eps, delta):.2f}")


def plot_fits(rng: np.random.Generator) -> Figure:
    x, y = dataset(rng)
    moved = neighbour(y)
    degree = N_POINTS - 1
    grid = np.linspace(0.0, 1.0, 401)
    phi = features(grid, degree)
    fig, axes = plt.subplots(1, 2, figsize=(8.5, 3.8), sharey=True)
    for ax, lam, title in zip(axes, (0.0, LAMBDA), ("no regularization", rf"ridge, $\lambda = {LAMBDA:g}$")):
        ax.plot(grid, phi @ fit(x, y, degree, lam), color="C0", label="fit on the dataset")
        ax.plot(grid, phi @ fit(x, moved, degree, lam), "--", color="C3", label="fit on the neighbour")
        ax.plot(x, y, "o", color="C0", markersize=4)
        ax.plot(x[N_POINTS // 2], moved[N_POINTS // 2], "s", color="C3", markersize=5)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel(r"feature $x$")
        ax.set_ylim(-2.5, 2.5)
        ax.grid(True, alpha=0.3)
    axes[0].set_ylabel(r"label $y$")
    axes[0].legend(fontsize=8, loc="lower left")
    fig.tight_layout()
    return fig


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=FIGURES)
    args = parser.parse_args()
    print_table(np.random.default_rng(args.seed))
    args.out.mkdir(parents=True, exist_ok=True)
    fig = plot_fits(np.random.default_rng(args.seed + 1))
    fig.savefig(args.out / "erm-regularization.svg", metadata={"Date": None})
    plt.close(fig)


if __name__ == "__main__":
    main()
