"""Check the worked solutions to the exercises of the Laplace and Gaussian mechanisms chapter.

    uv run python code/part03-dp-foundations/solutions_laplace_gaussian.py

Prints every number quoted in book-exercises/part03-dp-foundations/03-laplace-gaussian.md (seed 0 for the simulations).
Makes no figures; --out is accepted and ignored.

Conventions as in laplace_gaussian.py: μ = Δ₂/σ for the Gaussian mechanism, t = Δ₁/b for the
Laplace mechanism, and δ(ε) is the hockey-stick divergence E_{e^ε}.
"""

import argparse
import math
from pathlib import Path

import numpy as np
from laplace_gaussian import (
    analytic_sigma,
    classic_sigma,
    gaussian_delta,
    gaussian_epsilon,
    laplace_delta,
    laplace_loss_density,
    laplace_loss_masses,
    laplace_mean_loss,
    laplace_scale,
    normal_cdf,
    normal_pdf,
)

# Figures are build output: _build/figures/<part>/, which the build copies into each book.
FIGURES = Path(__file__).resolve().parents[2] / "_build" / "figures" / Path(__file__).parent.name

# --- Exercise 1: a private mean from a noisy sum and a noisy count --------------------------


def ratio_mean_rms(n: int, mean: float, eps: float) -> float:
    """First-order root-mean-square error of (S + Z_s)/(n + Z_c), with Z_s, Z_c ~ Lap(2/ε) (ε/2 each).

    (S + Z_s)/(n + Z_c) - S/n ≈ (Z_s - mean Z_c)/n, whose variance is 2b²(1 + mean²)/n² with b = 2/ε.
    """
    b = laplace_scale(1.0, eps / 2)
    return math.sqrt(2 * b * b * (1 + mean * mean)) / n


def simulated_ratio_mean_rms(rng: np.random.Generator, n: int, mean: float, eps: float, trials: int) -> float:
    b = laplace_scale(1.0, eps / 2)
    s = n * mean
    est = (s + rng.laplace(0.0, b, trials)) / (n + rng.laplace(0.0, b, trials))
    return float(np.sqrt(np.mean((est - mean) ** 2)))


def replace_mean_rms(n: int, eps: float) -> float:
    """Replace neighbours, n public: the mean has sensitivity 1/n, Laplace error √2/(nε)."""
    return math.sqrt(2) * laplace_scale(1.0 / n, eps)


# --- Exercise 3: the slope of δ(ε) ----------------------------------------------------------


def gaussian_delta_slope(eps: float, mu: float) -> float:
    """dδ/dε = -e^ε Φ(-μ/2 - ε/μ): the φ terms cancel."""
    return -math.exp(eps) * normal_cdf(-mu / 2 - eps / mu)


def density_identity_gap(eps: float, mu: float) -> float:
    """φ(μ/2 - ε/μ) - e^ε φ(-μ/2 - ε/μ), which is zero."""
    return normal_pdf(mu / 2 - eps / mu) - math.exp(eps) * normal_pdf(-mu / 2 - eps / mu)


# --- Exercise 5: composing Gaussian mechanisms exactly --------------------------------------


def composed_mu(k: int, sigma: float) -> float:
    """k Gaussian releases, each with Δ₂ = 1 and noise σ, form one Gaussian mechanism with Δ₂ = √k."""
    return math.sqrt(k) / sigma


def basic_composition_epsilon(k: int, sigma: float, delta: float) -> float:
    """Split δ evenly over k releases, convert each exactly, and add the ε's."""
    return k * gaussian_epsilon(delta / k, 1 / sigma)


# --- Exercise 6: a private mean of vectors --------------------------------------------------


def vector_mean_l2_rms(d: int, n: int, eps: float, delta: float) -> tuple[float, float]:
    """Root of E‖noise‖₂² for a mean of n vectors in the unit ℓ2 ball, replace neighbours.

    Δ₂ = 2/n and Δ₁ = 2√d/n. Laplace: √(2d) b with b = Δ₁/ε. Gaussian: √d σ with σ = analytic σ for Δ₂.
    """
    sens2, sens1 = 2 / n, 2 * math.sqrt(d) / n
    lap = math.sqrt(2 * d) * laplace_scale(sens1, eps)
    gau = math.sqrt(d) * analytic_sigma(eps, delta, sens2)
    return lap, gau


# --- Exercise 7: the privacy loss distribution of the Laplace mechanism ---------------------


def integrated_laplace_loss(eps: float, steps: int = 100_000) -> tuple[float, float]:
    """Total mass and mean of the Laplace loss distribution, with the continuous part by the midpoint rule."""
    top, bottom, _ = laplace_loss_masses(eps)
    h = 2 * eps / steps
    mids = [-eps + (i + 0.5) * h for i in range(steps)]
    mass = sum(laplace_loss_density(v, eps) for v in mids) * h
    first = sum(v * laplace_loss_density(v, eps) for v in mids) * h
    return top + bottom + mass, eps * top - eps * bottom + first


def print_table(rng: np.random.Generator) -> None:
    print("[Exercise 1: noisy sum / noisy count]")
    n, mean, eps = 1000, 0.3, 1.0
    print(
        f"  n={n} mean={mean:g} eps={eps:g}: first-order rms={ratio_mean_rms(n, mean, eps):.5f}  "
        f"simulated={simulated_ratio_mean_rms(rng, n, mean, eps, 1_000_000):.5f}  "
        f"replace-neighbour Laplace rms={replace_mean_rms(n, eps):.5f}"
    )

    print("[Exercise 2: the Laplace scale cannot be smaller]")
    print(f"  b=0.9, eps=1: Pr ratio on y <= 0 is e^(1/0.9)={math.exp(1 / 0.9):.4f} > e={math.e:.4f}")
    print(f"  b=1, Delta_1=1: delta(0.5)={laplace_delta(0.5, 1.0):.4f}  delta(0)={laplace_delta(0.0, 1.0):.4f}")

    print("[Exercise 3: slope of delta(eps)]")
    mu, e, h = 1.0, 1.0, 1e-6
    fd = (gaussian_delta(e + h, mu) - gaussian_delta(e - h, mu)) / (2 * h)
    print(
        f"  mu=1 eps=1: density identity gap={density_identity_gap(e, mu):.1e}  "
        f"slope formula={gaussian_delta_slope(e, mu):.4f}  finite difference={fd:.4f}"
    )
    print(f"  mu=1: delta(0)={gaussian_delta(0.0, mu):.4f} = 2 Phi(1/2) - 1 = {2 * normal_cdf(0.5) - 1:.4f}")

    print("[Exercise 4: calibrate by bisection]")
    lo, hi = 1e-3, 100.0
    rows: list[str] = []
    for step in range(1, 61):
        mid = math.sqrt(lo * hi)
        if gaussian_delta(0.5, mid) > 1e-6:
            hi = mid
        else:
            lo = mid
        if step in (1, 2, 3, 10, 20, 60):
            rows.append(f"step {step}: mu in [{lo:.6f}, {hi:.6f}]")
    print("  " + "  ".join(rows))
    print(
        f"  eps=0.5 delta=1e-6: analytic sigma={analytic_sigma(0.5, 1e-6):.4f}  classic sigma={classic_sigma(0.5, 1e-6):.4f}  "
        f"ratio={classic_sigma(0.5, 1e-6) / analytic_sigma(0.5, 1e-6):.3f}"
    )

    print("[Exercise 5: composing Gaussian mechanisms]")
    k, sigma, delta = 16, 4.0, 1e-5
    mu = composed_mu(k, sigma)
    print(
        f"  k={k} sigma={sigma:g}: mu={mu:.4f}  exact eps at delta=1e-5: {gaussian_epsilon(delta, mu):.4f}  "
        f"each alone: eps={gaussian_epsilon(delta / k, 1 / sigma):.4f} at delta/k  "
        f"basic composition: eps={basic_composition_epsilon(k, sigma, delta):.4f}"
    )

    print("[Exercise 6: mean of vectors in the unit ball]")
    lap, gau = vector_mean_l2_rms(100, 10_000, 1.0, 1e-5)
    print(
        f"  d=100 n=10000 eps=1 delta=1e-5: Laplace l2 rms={lap:.5f}  Gaussian l2 rms={gau:.5f}  ratio={lap / gau:.2f}"
    )

    print("[Exercise 7: the Laplace privacy loss distribution]")
    for e in (0.5, 1.0):
        top, bottom, middle = laplace_loss_masses(e)
        total, mean = integrated_laplace_loss(e)
        print(
            f"  eps={e:g}: atoms {top:.4f} at +eps and {bottom:.4f} at -eps  continuous {middle:.4f}  "
            f"total mass={total:.4f}  mean loss={mean:.4f}  formula eps + e^-eps - 1={laplace_mean_loss(e):.4f}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=FIGURES, help="accepted for a uniform interface; no figures")
    args = parser.parse_args()
    seed: int = args.seed
    print_table(np.random.default_rng(seed))


if __name__ == "__main__":
    main()
