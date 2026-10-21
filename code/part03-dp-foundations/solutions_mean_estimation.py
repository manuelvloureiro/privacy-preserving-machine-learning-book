"""Compute the numbers quoted in book-exercises/part03-dp-foundations/11-mean-estimation.md.

    uv run python code/part03-dp-foundations/solutions_mean_estimation.py

Each exercise of the "Private mean estimation" chapter has a block in print_table. The
functions reuse the chapter's script, code/mean_estimation.py.
"""

import argparse
import math
from pathlib import Path

import numpy as np
from mean_estimation import (
    BETA,
    DELTA,
    EPS,
    PARETO_SHAPE,
    N,
    bounded_n_for_rmse,
    bounded_rmse,
    coarse_bins,
    coarse_failure_bound,
    gaussian_mu,
    gaussian_spread,
    heavy_error,
    heavy_tau_bound,
    norm_spread,
    pareto_mean,
    pareto_scale,
)

# Figures are build output: _build/figures/<part>/, which the build copies into each book.
FIGURES = Path(__file__).resolve().parents[2] / "_build" / "figures" / Path(__file__).parent.name


# --- Exercise 2: the smallest budget for the coarse step ---------------------------------------


def coarse_eps_needed(n_bins: int, n: int, beta: float) -> float:
    """ε₁ with m e^{-ε₁ n/4} = β: ε₁ = 4 ln(m/β)/n."""
    return 4 * math.log(n_bins / beta) / n


def fine_rmse(n: int, eps2: float, beta: float) -> float:
    b = 2 * gaussian_spread(n, beta)
    return math.sqrt(1 / n + 2 * (3 * b / (n * eps2)) ** 2)


# --- Exercise 3: the Pareto tail -----------------------------------------------------------------


def pareto_excess(t: float, shape: float = PARETO_SHAPE) -> float:
    """E[(X - t)_+] = x_m^a t^{1-a}/(a - 1) for t ≥ x_m."""
    xm = pareto_scale(shape)
    return xm**shape * t ** (1 - shape) / (shape - 1)


# --- Exercise 4: when does a round shrink the ball? ----------------------------------------------


def one_round_radius(d: int, n: int, r: float, eps_t: float, delta_t: float, steps: int, beta: float) -> float:
    tau = norm_spread(d, 2 * n, beta)
    u = math.sqrt(2 * math.log(4 * steps / beta))
    sigma = 2 * (r + tau) / (n * gaussian_mu(eps_t, delta_t))
    return (math.sqrt(d) + u) / math.sqrt(n) + sigma * (math.sqrt(d) + u)


def n_to_halve(d: int, r: float, eps_t: float, delta_t: float, steps: int, beta: float) -> int:
    """Smallest n whose round brings the radius from r to at most r/2 (the radius falls with n)."""
    lo, hi = 1, 10**9
    while lo < hi:
        mid = (lo + hi) // 2
        if one_round_radius(d, mid, r, eps_t, delta_t, steps, beta) <= r / 2:
            hi = mid
        else:
            lo = mid + 1
    return lo


# --- Exercise 5: fingerprinting in ℓ₂ ------------------------------------------------------------


def fingerprint_n_l2(d: int, alpha_l2: float, eps: float, delta: float) -> float:
    """eq-lower-fp with per-coordinate error α' = α/√d (ℓ₁ ≤ √d ℓ₂), without padding."""
    a1 = alpha_l2 / math.sqrt(d)
    return (2 / 3 - 2 * a1) * d / ((math.exp(eps) - 1) * math.sqrt(d) + 4 * delta * d)


def fingerprint_n_l2_padded(d: int, alpha_l2: float, eps: float, delta: float) -> float:
    """Padded to per-coordinate error α₀ = 1/6: n ≥ (α₀/α')(2/3 - 2α₀) d/R = (√d/α) d/(18 R)."""
    a1 = alpha_l2 / math.sqrt(d)
    a0 = 1 / 6
    return (a0 / a1) * (2 / 3 - 2 * a0) * d / ((math.exp(eps) - 1) * math.sqrt(d) + 4 * delta * d)


# --- The solutions' numbers ----------------------------------------------------------------------


def print_table(rng: np.random.Generator) -> None:
    print("[Exercise 1: a survey at eps=0.1]")
    for alpha in (0.05, 0.01):
        root = bounded_n_for_rmse(alpha, 0.5, 1.0, 0.1)
        need = math.ceil(root)
        print(
            f"  alpha={alpha:g}: root={root:.1f}  n needed={need}  rmse at n={bounded_rmse(need, 0.5, 1.0, 0.1):.4f}  "
            f"simple bound={0.25 / alpha**2 + math.sqrt(2) / (alpha * 0.1):.0f}  non-private={0.25 / alpha**2:.0f}"
        )

    print(f"[Exercise 2: the smallest coarse budget, n={N}, R=1e6, beta={BETA:g}]")
    m = len(coarse_bins(1e6, N, BETA)) - 1
    e1 = coarse_eps_needed(m, N, BETA)
    print(f"  m={m}  eps_1={e1:.4f}  failure bound at eps_1={coarse_failure_bound(m, N, e1):.3f}")
    print(
        f"  fine rmse with eps_2={EPS - e1:.4f}: {fine_rmse(N, EPS - e1, BETA):.4f}  (with eps_2=0.75: {fine_rmse(N, 0.75, BETA):.4f})"
    )

    print("[Exercise 3: the Pareto tail]")
    mu = pareto_mean(PARETO_SHAPE)
    for n in (1000, 100_000):
        t = heavy_tau_bound(n, EPS)
        exact = heavy_error(t, n, EPS).bias
        print(
            f"  n={n}: tau_b={t:.2f}  excess formula={pareto_excess(mu + t):.6f}  exact bias={exact:.6f}  "
            f"bound 1/tau={1 / t:.4f}"
        )

    print("[Exercise 4: one round halves the ball, R=1000, eps_t=0.125, delta_t=delta/3]")
    for d in (10, 100, 1000):
        need = n_to_halve(d, 1000.0, 0.125, DELTA / 3, 3, BETA)
        print(f"  d={d}: n={need}  n/sqrt(d)={need / math.sqrt(d):.0f}")
    print(f"  4/mu(eps_t, delta_t)={4 / gaussian_mu(0.125, DELTA / 3):.0f}")

    print(f"[Exercise 5: fingerprinting in l2, eps={EPS:g}, delta={DELTA:g}]")
    for d in (100, 1000, 10_000):
        print(
            f"  d={d}: alpha=1: unpadded n >= {fingerprint_n_l2(d, 1.0, EPS, DELTA):.1f}  "
            f"padded n >= {fingerprint_n_l2_padded(d, 1.0, EPS, DELTA):.1f}  d/(18(e^eps - 1))={d / (18 * (math.e - 1)):.1f}"
        )
    _ = rng


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=FIGURES, help="accepted for a uniform interface; no figures")
    args = parser.parse_args()
    seed: int = args.seed
    print_table(np.random.default_rng(seed))


if __name__ == "__main__":
    main()
