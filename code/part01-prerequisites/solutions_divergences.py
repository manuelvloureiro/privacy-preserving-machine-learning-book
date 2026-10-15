"""Check the worked solutions to the exercises of the divergences chapter.

    uv run python code/part01-prerequisites/solutions_divergences.py

Prints every number quoted in book-exercises/part01-prerequisites/03-divergences.md (seed 0; nothing here is random, the
seed is accepted for a uniform interface). Makes no figures; --out is accepted and ignored.

Conventions as in divergences.py: P is the first argument, Q the second, logs are natural.
"""

import argparse
import math
from collections.abc import Sequence
from pathlib import Path

import numpy as np
from divergences import (
    FAIR,
    LOADED,
    Dist,
    binary_kl,
    hellinger_sq,
    hockey_stick,
    kl,
    push_forward,
    renyi,
    tv,
)
from numpy.typing import NDArray

# Figures are build output: _build/figures/<part>/, which the build copies into each book.
FIGURES = Path(__file__).resolve().parents[2] / "_build" / "figures" / Path(__file__).parent.name

# --- Exercise 1: chain rule for KL ----------------------------------------------------------


def conditional(p: Dist, labels: Sequence[int], label: int) -> list[float]:
    """The distribution of the outcome given that its label is `label` (outcomes in that group only)."""
    group = [pi for pi, y in zip(p, labels, strict=True) if y == label]
    total = sum(group)
    return [pi / total for pi in group]


def kl_chain_rule(p: Dist, q: Dist, labels: Sequence[int]) -> tuple[float, list[float], float]:
    """Split KL(P‖Q) over the outcome Y into KL of the label X plus the expected conditional KL.

    Returns (KL of the labels, [KL of Y given each label], E_{x~P_X}[KL of Y given x]).
    """
    p_x = push_forward(p, labels)
    q_x = push_forward(q, labels)
    conditional_kls = [kl(conditional(p, labels, x), conditional(q, labels, x)) for x in range(len(p_x))]
    expected = sum(px * c for px, c in zip(p_x, conditional_kls, strict=True))
    return kl(p_x, q_x), conditional_kls, expected


# --- Exercise 2: Gaussians with different variances -----------------------------------------


def gaussian_kl_general(mu0: float, s0: float, mu1: float, s1: float) -> float:
    """KL(N(μ0, σ0²)‖N(μ1, σ1²)) = ln(σ1/σ0) + (σ0² + (μ0 - μ1)²)/(2σ1²) - 1/2."""
    return math.log(s1 / s0) + (s0**2 + (mu0 - mu1) ** 2) / (2 * s1**2) - 0.5


def mixed_variance(alpha: float, s0: float, s1: float) -> float:
    """σ_α² = α σ1² + (1 - α) σ0²; the Rényi integral is finite iff it is positive."""
    return alpha * s1**2 + (1 - alpha) * s0**2


def gaussian_renyi_general(alpha: float, mu0: float, s0: float, mu1: float, s1: float) -> float:
    """D_α(N(μ0, σ0²)‖N(μ1, σ1²)) = ln(σ1/σ0) + ln(σ1²/σ_α²)/(2(α-1)) + α(μ0-μ1)²/(2σ_α²), or ∞."""
    if alpha == 1:
        return gaussian_kl_general(mu0, s0, mu1, s1)
    var = mixed_variance(alpha, s0, s1)
    if var <= 0:
        return math.inf
    return math.log(s1 / s0) + math.log(s1**2 / var) / (2 * (alpha - 1)) + alpha * (mu0 - mu1) ** 2 / (2 * var)


def renyi_order_limit(s0: float, s1: float) -> float:
    """The largest α with finite D_α: σ0²/(σ0² - σ1²) when σ0 > σ1, else ∞."""
    return s0**2 / (s0**2 - s1**2) if s0 > s1 else math.inf


def gaussian_log_pdf(x: NDArray[np.float64], mu: float, sigma: float) -> NDArray[np.float64]:
    return -((x - mu) ** 2) / (2 * sigma**2) - math.log(sigma * math.sqrt(2 * math.pi))


def quadrature_grid(mu0: float, s0: float, mu1: float, s1: float) -> NDArray[np.float64]:
    half_width = 40 * max(s0, s1) + abs(mu0 - mu1)
    return np.linspace(min(mu0, mu1) - half_width, max(mu0, mu1) + half_width, 400_001)


def quadrature_kl(mu0: float, s0: float, mu1: float, s1: float) -> float:
    """KL by the trapezoid rule on a wide grid: ∫ p ln(p/q)."""
    x = quadrature_grid(mu0, s0, mu1, s1)
    log_p, log_q = gaussian_log_pdf(x, mu0, s0), gaussian_log_pdf(x, mu1, s1)
    return float(np.trapezoid(np.exp(log_p) * (log_p - log_q), x))


def quadrature_renyi(alpha: float, mu0: float, s0: float, mu1: float, s1: float) -> float:
    """D_α by the trapezoid rule: ln ∫ p^α q^{1-α} / (α - 1), for an order where the integral converges."""
    x = quadrature_grid(mu0, s0, mu1, s1)
    log_integrand = alpha * gaussian_log_pdf(x, mu0, s0) + (1 - alpha) * gaussian_log_pdf(x, mu1, s1)
    return math.log(float(np.trapezoid(np.exp(log_integrand), x))) / (alpha - 1)


# --- Exercise 3: Bhattacharyya and Hellinger ------------------------------------------------


def renyi_half_from_hellinger(h2: float) -> float:
    """D_{1/2} = -2 ln(1 - H²/2), since Σ√(pq) = 1 - H²/2."""
    return -2 * math.log(1 - h2 / 2)


def gaussian_renyi_half(mu: float) -> float:
    """D_{1/2} for N(0, σ²) against N(s, σ²) with μ = s/σ: -2 ln e^{-μ²/8} = μ²/4."""
    return -2 * math.log(math.exp(-mu * mu / 8))


# --- Exercise 4: rejection sampling with a low envelope ------------------------------------


def rejection_output(p: Dist, q: Dist, gamma: float) -> list[float]:
    """Distribution of accepted proposals: m ∝ min(p, γq)."""
    covered = [min(pi, gamma * qi) for pi, qi in zip(p, q, strict=True)]
    total = sum(covered)
    return [c / total for c in covered]


# --- Exercise 6: Pinsker's constant is sharp ------------------------------------------------


def pinsker_ratio(delta: float) -> float:
    """kl(½ + δ, ½) / (2δ²), which tends to 1 as δ → 0."""
    return binary_kl(0.5 + delta, 0.5) / (2 * delta * delta)


# --- Exercises 7 and 8: n flips of two coins ------------------------------------------------


def binomial_pmf(n: int, a: float) -> list[float]:
    return [math.comb(n, k) * a**k * (1 - a) ** (n - k) for k in range(n + 1)]


def coins_tv(n: int, a: float, b: float) -> float:
    """TV between n flips of a coin with heads probability a and n flips of one with b.

    The number of heads is a sufficient statistic, so its TV equals the TV of the whole sequence.
    """
    return tv(binomial_pmf(n, a), binomial_pmf(n, b))


def pinsker_flips(level: float, kl_one: float) -> int:
    """Smallest n with sqrt(n·KL/2) ≥ level: fewer flips cannot reach TV = level."""
    return math.ceil(2 * level**2 / kl_one)


def bretagnolle_huber_flips(level: float, kl_one: float) -> int:
    """Smallest n with sqrt(1 - e^{-n·KL}) ≥ level."""
    return math.ceil(math.log(1 / (1 - level**2)) / kl_one)


def first_flips_reaching(level: float, a: float, b: float, n_max: int = 1000) -> int:
    """Smallest n with coins_tv(n) ≥ level (TV is non-decreasing in n, Exercise 8)."""
    for n in range(1, n_max + 1):
        if coins_tv(n, a, b) >= level:
            return n
    raise ValueError("level not reached")


# --- The numbers on the solutions page ------------------------------------------------------


def print_table(rng: np.random.Generator) -> None:
    del rng  # every number here is exact or a deterministic quadrature

    print("[Exercise 1: chain rule, X = parity of the roll, Y = the roll]")
    parity = [1, 0, 1, 0, 1, 0]  # 0 = even, 1 = odd
    kl_x, cond, expected = kl_chain_rule(LOADED, FAIR, parity)
    print(f"  KL(P_X‖Q_X)={kl_x:.4f}  KL(P_Y|even‖Q_Y|even)={cond[0]:.4f}  KL(P_Y|odd‖Q_Y|odd)={cond[1]:.4f}")
    print(f"  expected conditional KL={expected:.4f}  sum={kl_x + expected:.4f}  direct KL={kl(LOADED, FAIR):.4f}")

    print("[Exercise 2: Gaussians with different variances]")
    print(
        f"  KL(N(0,1)‖N(0,4)): formula={gaussian_kl_general(0, 1, 0, 2):.4f}  "
        f"quadrature={quadrature_kl(0, 1, 0, 2):.4f}"
    )
    print(f"  KL(N(0,4)‖N(0,1)): formula={gaussian_kl_general(0, 2, 0, 1):.4f}")
    print(
        f"  D_2(N(0,1)‖N(0,4)): formula={gaussian_renyi_general(2, 0, 1, 0, 2):.4f}  "
        f"quadrature={quadrature_renyi(2, 0, 1, 0, 2):.4f}"
    )
    print(
        f"  D_2(N(1,1)‖N(0,4)): formula={gaussian_renyi_general(2, 1, 1, 0, 2):.4f}  "
        f"quadrature={quadrature_renyi(2, 1, 1, 0, 2):.4f}"
    )
    print(
        f"  D_alpha(N(0,4)‖N(0,1)) finite only for alpha < {renyi_order_limit(2, 1):.4f}; "
        f"D_1.2={gaussian_renyi_general(1.2, 0, 2, 0, 1):.4f} "
        f"(quadrature {quadrature_renyi(1.2, 0, 2, 0, 1):.4f})  "
        f"D_2={gaussian_renyi_general(2, 0, 2, 0, 1)}"
    )

    print("[Exercise 3: Bhattacharyya and Hellinger]")
    h2 = hellinger_sq(LOADED, FAIR)
    print(
        f"  die: H^2={h2:.4f}  Bhattacharyya coefficient={1 - h2 / 2:.4f}  "
        f"-2 ln(1 - H^2/2)={renyi_half_from_hellinger(h2):.4f}  D_0.5 directly={renyi(LOADED, FAIR, 0.5):.4f}"
    )
    print(
        f"  Gaussian pair: D_0.5 at mu=1 is {gaussian_renyi_half(1):.4f}; "
        f"at mu=1/sqrt(2) is {gaussian_renyi_half(1 / math.sqrt(2)):.4f}"
    )

    print("[Exercise 4: rejection sampling, die, envelope gamma * fair]")
    for gamma in (1.0, 1.5, 2.0, 2.5, 3.0):
        m = rejection_output(LOADED, FAIR, gamma)
        print(
            f"  gamma={gamma:g}: m(6)={m[5]:.4f}  TV(P,M)={tv(LOADED, m):.4f}  "
            f"E_gamma={hockey_stick(LOADED, FAIR, gamma):.4f}"
        )

    print("[Exercise 5: E_gamma after reporting the parity of the roll]")
    p_y, q_y = push_forward(LOADED, parity), push_forward(FAIR, parity)
    for gamma in (1.0, 1.2, 2.0):
        print(
            f"  gamma={gamma:g}: raw E_gamma={hockey_stick(LOADED, FAIR, gamma):.4f}  "
            f"parity E_gamma={hockey_stick(p_y, q_y, gamma):.4f}"
        )

    print("[Exercise 6: kl(1/2 + delta, 1/2) / (2 delta^2)]")
    for delta in (0.2, 0.1, 0.01):
        print(f"  delta={delta:g}: ratio={pinsker_ratio(delta):.4f}  1 + 2delta^2/3={1 + 2 * delta**2 / 3:.4f}")

    print("[Exercise 7: flips needed to tell a fair coin from one with heads probability 0.6]")
    coin = binary_kl(0.6, 0.5)
    level = 0.8
    print(
        f"  kl(0.6, 0.5)={coin:.5f}  Pinsker: n >= {pinsker_flips(level, coin)}  "
        f"Bretagnolle-Huber: n >= {bretagnolle_huber_flips(level, coin)}"
    )
    print(
        f"  exact TV at n=64: {coins_tv(64, 0.6, 0.5):.4f}  "
        f"exact TV first reaches 0.8 at n={first_flips_reaching(level, 0.6, 0.5)}"
    )

    print("[Exercise 8: TV between n flips of the two coins]")
    print("  " + "  ".join(f"n={n}: {coins_tv(n, 0.6, 0.5):.4f}" for n in (1, 2, 3, 4, 16, 64, 256)))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=FIGURES, help="accepted for a uniform interface; no figures")
    args = parser.parse_args()
    seed: int = args.seed
    print_table(np.random.default_rng(seed))


if __name__ == "__main__":
    main()
