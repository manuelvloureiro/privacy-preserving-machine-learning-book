"""Check the worked solutions to the exercises of the advanced-and-optimal-composition chapter.

    uv run python code/part03-dp-foundations/solutions_composition.py

Prints every number quoted in book-exercises/part03-dp-foundations/04-composition.md (nothing here is random; the seed is
accepted for a uniform interface). Makes no figures; --out is accepted and ignored.

Conventions as in composition.py: P is the output distribution on D, Q the one on D', the privacy
loss is L = ln(p/q), and randomized response at level ε is truthful with probability e^ε/(1 + e^ε).
"""

import argparse
import math
from fractions import Fraction
from pathlib import Path

import numpy as np
from composition import (
    advanced_crossover,
    advanced_epsilon,
    advanced_epsilon_tanh,
    basic_epsilon,
    drv_mean_bound,
    gaussian_delta,
    gaussian_epsilon,
    hockey_stick,
    kov_delta,
    kov_epsilon,
    max_loss,
    product_pair,
    rr_channel,
    rr_keep,
    rr_mean_loss,
    rr_pair,
)

# Figures are build output: _build/figures/<part>/, which the build copies into each book.
FIGURES = Path(__file__).resolve().parents[2] / "_build" / "figures" / Path(__file__).parent.name

# --- Exercise 1: the sharp mean loss --------------------------------------------------------


def mean_bound_ratio(eps: float) -> float:
    """tanh(ε/2) / (e^ε - 1) = 1 / (e^ε + 1): the sharp mean over the Dwork–Rothblum–Vadhan one."""
    return rr_mean_loss(eps) / drv_mean_bound(eps)


# --- Exercise 2: when advanced composition pays --------------------------------------------


def crossover_threshold(eps: float, delta_prime: float) -> float:
    """Advanced (DRV form) < kε iff k > 2 ln(1/δ') / (2 - e^ε)², for ε < ln 2."""
    return 2 * math.log(1 / delta_prime) / (2 - math.exp(eps)) ** 2


def crossover_threshold_tanh(eps: float, delta_prime: float) -> float:
    """The same for the tanh form: k > 2 ln(1/δ') / (1 - tanh(ε/2))²."""
    return 2 * math.log(1 / delta_prime) / (1 - math.tanh(eps / 2)) ** 2


def first_k_tanh_beats_basic(eps: float, delta_prime: float) -> int:
    k = 1
    while advanced_epsilon_tanh(eps, k, delta_prime) >= basic_epsilon(eps, k):
        k += 1
    return k


# --- Exercise 3: two rounds by hand ---------------------------------------------------------


def two_round_delta_exact(eps_prime_power: int) -> Fraction:
    """δ(ε') for two rounds of randomized response at ln 3, at ε' = m ln 3 (m = 0, 1, 2): 9/16 - 3^m/16."""
    return max(Fraction(9, 16) - Fraction(3**eps_prime_power, 16), Fraction(0))


# --- Exercise 7: two different steps --------------------------------------------------------


def hetero_outputs(g1: int, g2: int) -> list[tuple[Fraction, Fraction]]:
    """(P, Q) probabilities of the four outputs of RR at ln g1 then RR at ln g2, in the order
    (true, true), (true, flipped), (flipped, true), (flipped, flipped)."""
    a1, a2 = Fraction(g1, g1 + 1), Fraction(g2, g2 + 1)
    p = [a1 * a2, a1 * (1 - a2), (1 - a1) * a2, (1 - a1) * (1 - a2)]
    q = [(1 - a1) * (1 - a2), (1 - a1) * a2, a1 * (1 - a2), a1 * a2]
    return list(zip(p, q, strict=True))


def hetero_delta(g1: int, g2: int, gamma: Fraction) -> Fraction:
    """δ at e^{ε'} = gamma, exactly: Σ (p - γ q)_+."""
    return sum((max(p - gamma * q, Fraction(0)) for p, q in hetero_outputs(g1, g2)), start=Fraction(0))


# --- The page's numbers ---------------------------------------------------------------------


def print_table(rng: np.random.Generator) -> None:
    del rng  # nothing random here
    print("[Exercise 1]")
    for eps in (0.01, 0.1, 0.5, 1.0, 2.0):
        print(
            f"  eps={eps:g}: tanh form={rr_mean_loss(eps):.6f}  eps^2/2={eps**2 / 2:.6f}  DRV={drv_mean_bound(eps):.6f}  "
            f"ratio tanh/DRV={mean_bound_ratio(eps):.4f}"
        )

    print("[Exercise 2]")
    print(
        f"  eps=0.1 delta'=1e-5: threshold={crossover_threshold(0.1, 1e-5):.1f}  first k={advanced_crossover(0.1, 1e-5)}  "
        f"advanced at k=28: {advanced_epsilon(0.1, 28, 1e-5):.4f}  at k=29: {advanced_epsilon(0.1, 29, 1e-5):.4f}"
    )
    print(
        f"  tanh form: threshold={crossover_threshold_tanh(0.1, 1e-5):.1f}  first k={first_k_tanh_beats_basic(0.1, 1e-5)}"
    )
    print(f"  eps=0.5: threshold={crossover_threshold(0.5, 1e-5):.1f}  first k={advanced_crossover(0.5, 1e-5)}")

    print("[Exercise 3]")
    rp, rq = rr_pair(math.log(3))
    p2, q2 = product_pair(rp, rq, 2)
    for m in (0, 1, 2):
        print(
            f"  eps'={m} ln 3: exact={two_round_delta_exact(m)}  enumeration={hockey_stick(p2, q2, m * math.log(3)):.4f}  "
            f"formula={kov_delta(math.log(3), 0.0, 2, m * math.log(3)):.4f}"
        )

    print("[Exercise 4]")
    e, d, k, target = 0.1, 5e-8, 100, 1e-5
    print(
        f"  1-(1-delta)^k={1 - (1 - d) ** k:.6e}  k*delta={k * d:.6e}  optimal={kov_epsilon(e, d, k, target):.4f}  "
        f"advanced={advanced_epsilon(e, k, target - k * d):.4f}  advanced tanh={advanced_epsilon_tanh(e, k, target - k * d):.4f}"
    )
    print(f"  target 4e-6 < 1-(1-delta)^k: optimal eps={kov_epsilon(e, d, k, 4e-6)}")

    print("[Exercise 5]")
    p3, q3 = [0.5, 0.3, 0.2], [0.2, 0.3, 0.5]
    eps_min = max_loss(p3, q3)
    k0, k1 = rr_channel(p3, q3, eps_min)
    t = rr_keep(eps_min)
    print(
        f"  smallest eps=ln 2.5={eps_min:.4f}  t={t:.4f}  K(.|0)={[round(x, 4) for x in k0]}  K(.|1)={[round(x, 4) for x in k1]}  "
        f"rebuilt P={[round(t * a + (1 - t) * b, 4) for a, b in zip(k0, k1)]}"
    )
    k0s, _ = rr_channel(p3, q3, 0.8)
    print(f"  at eps=0.8: K(.|0)={[round(x, 4) for x in k0s]}")

    print("[Exercise 6]")
    sigma, kk, delta = 20.0, 1000, 1e-5
    mu_k = math.sqrt(kk) / sigma
    d0 = 5e-9
    e0 = gaussian_epsilon(1 / sigma, d0)
    print(
        f"  mu_k=sqrt(k)/sigma={mu_k:.4f}  exact eps={gaussian_epsilon(mu_k, delta):.3f}  "
        f"check delta={gaussian_delta(mu_k, gaussian_epsilon(mu_k, delta)):.2e}"
    )
    print(
        f"  true mean loss k/(2 sigma^2)={kk / (2 * sigma**2):.2f}  per-step eps0 at delta0={d0:g}: {e0:.4f}  "
        f"k*eps0*tanh(eps0/2)={kk * rr_mean_loss(e0):.2f}  eps0*sqrt(2k ln(1/delta'))={e0 * math.sqrt(2 * kk * math.log(1 / (delta - kk * d0))):.2f}"
    )

    print("[Exercise 7]")
    for (p, q), name in zip(
        hetero_outputs(3, 2), ("(true, true)", "(true, flipped)", "(flipped, true)", "(flipped, flipped)")
    ):
        print(f"  {name}: P={p}  Q={q}  e^L={p / q}")
    for gamma, name in ((Fraction(1), "0"), (Fraction(3, 2), "ln 3/2"), (Fraction(2), "ln 2"), (Fraction(6), "ln 6")):
        print(f"  delta({name})={hetero_delta(3, 2, gamma)}")
    print(
        f"  homogeneous at eps=ln 3, k=2: delta(0)={kov_delta(math.log(3), 0.0, 2, 0.0):.4f}; "
        f"at eps=ln 2: {kov_delta(math.log(2), 0.0, 2, 0.0):.4f}"
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
