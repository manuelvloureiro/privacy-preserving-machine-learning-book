"""Check the worked solutions to the exercises of the numerical-accounting chapter.

    uv run python code/part04-privacy-accounting/solutions_numerical_accounting.py

Prints every number quoted in book-exercises/part04-privacy-accounting/06-numerical-accounting.md (seed 0; nothing here is random,
the seed is accepted for a uniform interface). Makes no figures; --out is accepted and ignored.

Conventions as in numerical_accounting.py: densities P(y), Q(y); privacy loss L = ln(P/Q) with Y ~ P;
q is the sampling rate.
"""

import argparse
import math
from pathlib import Path

import numpy as np
from numerical_accounting import (
    DELTA,
    PLD,
    Q_RATE,
    SIGMA,
    compose_pld,
    compose_pld_circular,
    compose_pld_direct,
    convolve,
    discretize_points,
    discretize_subsampled_gaussian_pld,
    gaussian_epsilon,
    gaussian_pld,
    gaussian_pld_delta,
    normal_cdf,
    pld_delta,
    pld_epsilon,
    rr_pld,
    subsampled_delta_exact,
    validity,
)

# Figures are build output: _build/figures/<part>/, which the build copies into each book.
FIGURES = Path(__file__).resolve().parents[2] / "_build" / "figures" / Path(__file__).parent.name

# --- Exercise 1: the PLD of the swapped pair ---------------------------------------------------


def swap_pld(pld: PLD) -> PLD:
    """PLD of (Q, P) from the PLD of (P, Q): mass e^{-ℓ} m at loss -ℓ; Q-mass not covered goes to +∞."""
    tilted = pld.mass * np.exp(-pld.losses())
    deficit = max(1.0 - float(np.sum(tilted)), 0.0)
    return PLD(-(pld.offset + pld.mass.size - 1), pld.h, tilted[::-1].copy(), deficit, pld.pessimistic)


# --- Exercise 2: δ from tail probabilities -----------------------------------------------------


def gaussian_tail_bounds(eps: float, mu: float) -> tuple[float, float, float]:
    """Upper bound P[L > ε] and lower bound max_{ε'>ε} (1 - e^{ε-ε'}) P[L > ε'] on δ(ε); also the best ε'."""
    upper = normal_cdf(mu / 2 - eps / mu)  # P[L > ε], L ~ N(μ²/2, μ²)
    grid = np.linspace(eps, eps + 5 * mu, 50_001)[1:]
    lower = np.array([-math.expm1(eps - e) * normal_cdf(mu / 2 - float(e) / mu) for e in grid])
    best = int(np.argmax(lower))
    return upper, float(lower[best]), float(grid[best])


# --- Exercise 4: wraparound ------------------------------------------------------------------


def outside_window_mass(k: int, mu: float, width: float) -> float:
    """Mass of N(kμ²/2, kμ²) more than width/2 from its mean."""
    return 2 * normal_cdf(-width / 2 / (math.sqrt(k) * mu))


def chebyshev_bound(k: int, mu: float, width: float) -> float:
    """Chebyshev: P[|S - mean| ≥ width/2] ≤ variance / (width/2)²."""
    return k * mu**2 / (width / 2) ** 2


def print_table(rng: np.random.Generator) -> None:
    del rng  # nothing is random

    print("[Exercise 1: the swapped pair]")
    mu = 1.0
    g = gaussian_pld(mu, 1e-3, True)
    print(
        f"  Gaussian mu=1: E[e^-L] = {validity(g):.6f}; swapped PLD delta(1)={pld_delta(swap_pld(g), 1.0):.6f}  "
        f"original delta(1)={pld_delta(g, 1.0):.6f}  exact={gaussian_pld_delta(1.0, mu):.6f}"
    )
    rem = discretize_subsampled_gaussian_pld(Q_RATE, SIGMA, 1e-5, True, "remove")
    add_from_rem = swap_pld(rem)
    add_direct = discretize_subsampled_gaussian_pld(Q_RATE, SIGMA, 1e-5, True, "add")
    print(
        f"  subsampled Gaussian: remove E[e^-L]={validity(rem):.6f}  deficit moved to +inf={add_from_rem.inf_mass:.1e}"
    )
    for eps in (0.0, 0.005):
        print(
            f"  add direction eps={eps:g}: from swapped remove PLD={pld_delta(add_from_rem, eps):.6e}  "
            f"direct={pld_delta(add_direct, eps):.6e}  exact={subsampled_delta_exact(eps, Q_RATE, SIGMA, 'add'):.6e}"
        )

    print("[Exercise 2: delta from tails, Gaussian mu=1]")
    for eps in (1.0, 3.0):
        upper, lower, at = gaussian_tail_bounds(eps, mu)
        print(
            f"  eps={eps:g}: P[L>eps]={upper:.4f}  delta={gaussian_pld_delta(eps, mu):.4f}  "
            f"lower bound={lower:.4f} at eps'={at:.3f}"
        )

    print("[Exercise 3: the sandwich width]")
    losses, probs = rr_pld(math.log(3))
    for h in (0.5, 0.1, math.log(3) / 2):
        up = discretize_points(losses, probs, h, True)
        dn = discretize_points(losses, probs, h, False)
        up3, dn3 = compose_pld_direct(up, 3), compose_pld_direct(dn, 3)
        print(
            f"  RR ln 3, h={h:.4f}: E[e^-L] up={validity(up):.4f} down={validity(dn):.4f}  "
            f"eps at delta=0.3: up={pld_epsilon(up3, 0.3):.4f} down={pld_epsilon(dn3, 0.3):.4f}"
        )
    for h in (1e-3, 1e-4):
        for k in (100, 1000):
            up_k = compose_pld(discretize_subsampled_gaussian_pld(Q_RATE, SIGMA, h, True, "remove"), k)
            dn_k = compose_pld(discretize_subsampled_gaussian_pld(Q_RATE, SIGMA, h, False, "remove"), k)
            width = pld_epsilon(up_k, DELTA) - pld_epsilon(dn_k, DELTA)
            print(f"  remove direction h={h:g} k={k}: width={width:.4f}  k*h={k * h:.4f}")

    print("[Exercise 4: wraparound, Gaussian mu=1, k=10, h=1e-3]")
    k, h = 10, 1e-3
    g1 = gaussian_pld(mu, h, True)
    padded = pld_delta(compose_pld_direct(g1, k), 1.0)
    for width in (10.0, 20.0, 30.0, 40.0):
        circ = compose_pld_circular(g1, k, round(width / h), round((k * mu**2 / 2 - width / 2) / h))
        err = abs(pld_delta(circ, 1.0) - padded)
        print(
            f"  width {width:g}: |delta error|={err:.2e}  mass outside={outside_window_mass(k, mu, width):.2e}  "
            f"Chebyshev={chebyshev_bound(k, mu, width):.4f}"
        )

    print("[Exercise 5: 1000 DP-SGD steps then one Gaussian release with sigma=10]")
    h = 1e-5
    for pess in (True, False):
        g_final = gaussian_pld(0.1, h, pess)
        eps_dirs: list[float] = []
        for d in ("remove", "add"):
            sgd = compose_pld(discretize_subsampled_gaussian_pld(Q_RATE, SIGMA, h, pess, d), 1000)
            eps_dirs.append(pld_epsilon(convolve(sgd, g_final), DELTA))
        name = "pessimistic" if pess else "optimistic"
        print(f"  {name}: remove {eps_dirs[0]:.4f}  add {eps_dirs[1]:.4f}  max {max(eps_dirs):.4f}")
    print(f"  the sigma=10 Gaussian release alone: epsilon at delta=1e-5 = {gaussian_epsilon(0.1, DELTA):.4f}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=FIGURES, help="ignored; the solutions make no figures")
    args = parser.parse_args()
    seed: int = args.seed
    print_table(np.random.default_rng(seed))


if __name__ == "__main__":
    main()
