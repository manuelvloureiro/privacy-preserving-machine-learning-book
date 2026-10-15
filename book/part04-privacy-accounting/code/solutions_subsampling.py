"""Check the worked solutions to the exercises of the amplification-by-subsampling chapter.

    uv run python book/part04-privacy-accounting/code/solutions_subsampling.py

Prints every number quoted in solutions/05-subsampling.md (seed 0; nothing here is random, the
seed is accepted for a uniform interface). Makes no figures; --out is accepted and ignored.

Conventions as in subsampling.py: q is the sampling rate, P0 = N(0, σ²), P1 = N(1, σ²) (mu0, mu1 in code).
"""

import argparse
import math
from pathlib import Path

import numpy as np
from subsampling import (
    binomial_terms,
    bisect,
    dpsgd_eps_rdp,
    gaussian_delta,
    log_a_binomial,
    replace_one_under_poisson,
    replace_one_under_poisson_pure,
)

FIGURES = Path(__file__).resolve().parent.parent / "figures"

# --- Exercise 1: subsampling twice -----------------------------------------------------------


def amplify_eps(eps: float, q: float) -> float:
    return math.log1p(q * math.expm1(eps))


# --- Exercise 2: the binomial sum at α = 3 --------------------------------------------------


def a3_by_hand(q: float, sigma: float) -> float:
    """A_3 = (1-q)³ + 3q(1-q)² + 3q²(1-q)e^{1/σ²} + q³e^{3/σ²}."""
    return (
        (1 - q) ** 3
        + 3 * q * (1 - q) ** 2
        + 3 * q**2 * (1 - q) * math.exp(1 / sigma**2)
        + q**3 * math.exp(3 / sigma**2)
    )


# --- Exercise 3: replace-one under Poisson sampling, pure DP --------------------------------


def laplace_event_ratio(eps: float, q: float) -> float:
    """The ratio of the two mixtures' probabilities of {Y > 1} for the Laplace attack.

    Laplace noise of scale 1/ε on a sum; the other records are 0, x = 1, x' = -1, so R = Lap(0),
    P = Lap(1), P' = Lap(-1), with P(Y > 1) = 1/2, R(Y > 1) = e^{-ε}/2, P'(Y > 1) = e^{-2ε}/2.
    """
    p, r, p2 = 0.5, 0.5 * math.exp(-eps), 0.5 * math.exp(-2 * eps)
    return ((1 - q) * r + q * p) / ((1 - q) * r + q * p2)


# --- Exercise 4: inverting the accountant ---------------------------------------------------


def sigma_for_target(q: float, steps: int, delta: float, target: float) -> float:
    """Smallest σ (to 1e-4) whose RDP accountant (improved conversion) reports ε ≤ target."""
    return bisect(lambda s: dpsgd_eps_rdp(q, round(s, 4), steps, delta)[0] <= target, 0.5, 10.0, 20)


# --- Exercise 6: the hockey-stick identity for the first direction --------------------------


def mixture_hockey_sticks(eps: float, q: float, sigma: float) -> tuple[float, float]:
    """E_{γ'}(mixture‖μ0) and E_{γ'}(μ0‖mixture) by the trapezoid rule, γ' = 1 + q(e^ε - 1)."""
    gamma = 1 + q * math.expm1(eps)
    z = np.linspace(-15 * sigma, 15 * sigma + 1, 600_001)
    mu0 = np.exp(-(z**2) / (2 * sigma**2)) / (sigma * math.sqrt(2 * math.pi))
    mu1 = np.exp(-((z - 1) ** 2) / (2 * sigma**2)) / (sigma * math.sqrt(2 * math.pi))
    m = (1 - q) * mu0 + q * mu1
    dz = float(z[1] - z[0])
    return float(np.sum(np.maximum(m - gamma * mu0, 0)) * dz), float(np.sum(np.maximum(mu0 - gamma * m, 0)) * dz)


def print_table() -> None:
    print("[Exercise 1: subsampling twice]")
    e, q1, q2 = 1.0, 0.1, 0.1
    twice = amplify_eps(amplify_eps(e, q1), q2)
    print(
        f"  eps={e:g}: after q1={q1:g} {amplify_eps(e, q1):.4f}; then q2={q2:g} {twice:.5f}; "
        f"directly with q1*q2 {amplify_eps(e, q1 * q2):.5f}"
    )

    print("[Exercise 2: alpha = 3 by hand, q=0.01, sigma=1]")
    q, s = 0.01, 1.0
    terms = binomial_terms(3, q, s)
    print("  terms j=0..3: " + "  ".join(f"{t:.8f}" for t in terms))
    a3 = a3_by_hand(q, s)
    print(f"  A_3={a3:.8f}  eps(3)=ln(A_3)/2={math.log(a3) / 2:.4e}  from the script {log_a_binomial(3, q, s) / 2:.4e}")
    cost2 = terms[2] - 3 * q**2 * (1 - q)
    cost3 = terms[3] - q**3
    print(f"  privacy cost of j=2: {cost2:.8f}  of j=3: {cost3:.8f}  share of j=2: {cost2 / (cost2 + cost3):.2f}")
    print(
        f"  without the e-factors the terms sum to {(1 - q) ** 3 + 3 * q * (1 - q) ** 2 + 3 * q**2 * (1 - q) + q**3:.8f}"
    )

    print("[Exercise 3: replace-one under Poisson sampling, pure DP, q=0.01]")
    for eps in (0.1, 1.0, 5.0):
        add_remove = amplify_eps(eps, q)
        exact = replace_one_under_poisson_pure(eps, q)
        group = replace_one_under_poisson(eps, 0.0, q)[0]
        print(
            f"  eps={eps:g}: add/remove={add_remove:.5f}  exact replace-one={exact:.5f}  group bound={group:.5f}  "
            f"exact/add-remove={exact / add_remove:.2f}  group/add-remove={group / add_remove:.2f}  "
            f"Laplace event ln-ratio={math.log(laplace_event_ratio(eps, q)):.5f}"
        )

    print("[Exercise 4: noise for a target eps, q=0.01, k=10000, delta=1e-5]")
    print(
        f"  bracket: eps at sigma=0.5 is {dpsgd_eps_rdp(0.01, 0.5, 10_000, 1e-5)[0]:.2f}, "
        f"at sigma=10 is {dpsgd_eps_rdp(0.01, 10.0, 10_000, 1e-5)[0]:.3f}"
    )
    for target in (1.0, 2.0, 8.0):
        sig = sigma_for_target(0.01, 10_000, 1e-5, target)
        eps, order = dpsgd_eps_rdp(0.01, round(sig, 4), 10_000, 1e-5)
        print(f"  target eps={target:g}: sigma={sig:.3f}  (check: eps={eps:.3f} at alpha={order:g})")

    print("[Exercise 5: fixed-size batches, q=0.01, k=10000, delta=1e-5]")
    for sig in (1.0, 2.0):
        print(
            f"  sigma={sig:g}: Poisson eps={dpsgd_eps_rdp(0.01, sig, 10_000, 1e-5)[0]:.2f}  "
            f"fixed-size eps={dpsgd_eps_rdp(0.01, sig / 2, 10_000, 1e-5)[0]:.2f}"
        )

    print("[Exercise 6: E_gamma' of the subsampled Gaussian, q=0.01, sigma=1]")
    for eps in (1.0, 2.0, 3.0):
        first, second = mixture_hockey_sticks(eps, q, s)
        print(
            f"  eps={eps:g}: eps'={amplify_eps(eps, q):.5f}  q*delta(eps)={q * gaussian_delta(eps, s):.5e}  "
            f"E(mixture||mu0)={first:.5e}  E(mu0||mixture)={second:.5e}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=FIGURES, help="ignored; the solutions draw no figures")
    parser.parse_args()
    print_table()


if __name__ == "__main__":
    main()
