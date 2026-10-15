"""Check the worked solutions to the exercises of the threat-models chapter.

    uv run python book/part02-attacks/code/solutions_threat_models.py

Prints every number quoted in solutions/01-threat-models.md (seed 0). Makes no figures; --out is
accepted and ignored.

Conventions as in threat_models.py: b = 1 means "the target is in the data" (Q), b = 0 means it is
not (P); TPR = Pr[say 1 | b = 1], FPR = Pr[say 1 | b = 0].
"""

import argparse
import math
from pathlib import Path

import numpy as np
from threat_models import (
    Mask,
    average_case_tv,
    average_case_tv_normal,
    bayes_attribute_accuracy,
    binomial_weights,
    bisect,
    centroid_games,
    count_reconstruction_accuracy,
    fraction,
    normal_cdf,
    precision,
    worst_case_tv,
)

FIGURES = Path(__file__).resolve().parent.parent / "figures"

# --- Exercise 1: winning with a prior ------------------------------------------------------


def prior_win_bound(prior: float, tv: float) -> float:
    """max(π, 1 − π) + min(π, 1 − π)·TV: no attack wins the game with prior π more often."""
    return max(prior, 1 - prior) + min(prior, 1 - prior) * tv


# --- Exercise 2: the exact count and Stirling ----------------------------------------------


def exact_count_tv_by_sum(m: int) -> float:
    """½ Σ_k |Pr[K = k] − Pr[K + 1 = k]| for K ~ Binomial(m, 1/2), summed directly."""
    w = list(binomial_weights(m))
    p = w + [0.0]
    q = [0.0] + w
    return 0.5 * sum(abs(a - b) for a, b in zip(p, q, strict=True))


def stirling_central_weight(m: int) -> float:
    """√(2/(π m)), the Stirling approximation of the central binomial weight."""
    return math.sqrt(2 / (math.pi * m))


# --- Exercise 3: how big a crowd? ----------------------------------------------------------


def crowd_needed(target_tv: float, sigma: float) -> tuple[float, int]:
    """(m from the normal approximation, smallest m whose exact average-case TV is ≤ target_tv)."""
    # 2Φ(1/(2s)) − 1 = target  ⇔  s = 1/(2 Φ⁻¹((1 + target)/2)),  and s² = m/4 + σ².
    x = bisect(lambda u: normal_cdf(u) - (1 + target_tv) / 2, 0.0, 10.0)
    s = 1 / (2 * x)
    m_normal = 4 * (s**2 - sigma**2)
    m = 1
    while average_case_tv(m, sigma) > target_tv:
        m += 1
    return m_normal, m


# --- Exercise 5: releasing one record from a small domain ----------------------------------


def release_rates_discrete(n: int, values: int) -> tuple[float, float]:
    """(TPR, FPR) of "member iff the released record equals z" with records uniform on `values` values."""
    fpr = 1 / values
    tpr = 1 / n + (1 - 1 / n) / values
    return tpr, fpr


def simulate_release_discrete(rng: np.random.Generator, n: int, values: int, games: int) -> tuple[float, float]:
    released_in_is_target: Mask = rng.integers(0, n, games) == 0
    other_in = rng.integers(0, values, games)
    z_in = rng.integers(0, values, games)
    hit_in: Mask = released_in_is_target | (other_in == z_in)
    hit_out: Mask = rng.integers(0, values, games) == rng.integers(0, values, games)
    return fraction(hit_in), fraction(hit_out)


# --- Exercise 6: the inner-product attack --------------------------------------------------


def inner_product_advantage_normal(n: int, d: int) -> float:
    """2Φ(√(d/n)/2) − 1: the normal approximation of the inner-product attack's best advantage."""
    return 2 * normal_cdf(math.sqrt(d / n) / 2) - 1


def inner_product_moments(n: int, d: int) -> tuple[float, float, float]:
    """(mean in, variance in, variance out) of ⟨z, θ⟩; the mean out is 0."""
    return d / n, (2 * d + d * (n - 1)) / n**2, d / n


def simulated_inner_product_advantage(rng: np.random.Generator, n: int, d: int, games: int) -> float:
    z_in, th_in, z_out, th_out = centroid_games(rng, n, d, games)
    s_in = np.sum(z_in * th_in, axis=1)
    s_out = np.sum(z_out * th_out, axis=1)
    scores = np.concatenate([s_in, s_out])
    is_in = np.concatenate([np.ones(games), np.zeros(games)])
    order = np.argsort(-scores, kind="stable")
    gap = np.cumsum(is_in[order]) / games - np.cumsum(1 - is_in[order]) / games
    return float(np.max(gap))


# --- Exercise 7: mixing hides --------------------------------------------------------------


def two_world_mixture_tv() -> tuple[float, float]:
    """One unknown other bit, exact count: (TV of the mixtures, average TV when the bit is known)."""
    return average_case_tv(1, 0.0), 1.0


def print_solutions(rng: np.random.Generator) -> None:
    sigma = 1.0
    tv = worst_case_tv(sigma)
    print("[Exercise 1]")
    for prior in (0.5, 0.1):
        print(
            f"  prior {prior:g}: bound={prior_win_bound(prior, tv):.4f}  "
            f"best={bayes_attribute_accuracy(prior, 1 / sigma):.4f}"
        )

    print("[Exercise 2]")
    for m in (10, 100):
        print(
            f"  m={m}: TV by sum={exact_count_tv_by_sum(m):.4f}  central weight={average_case_tv(m, 0):.4f}  "
            f"Stirling={stirling_central_weight(m):.4f}"
        )

    print("[Exercise 3]")
    m_normal, m_exact = crowd_needed(0.05, sigma)
    x = bisect(lambda u: normal_cdf(u) - 0.525, 0.0, 10.0)
    print(f"  Phi^-1(0.525)={x:.5f}  s={1 / (2 * x):.3f}  s^2={1 / (2 * x) ** 2:.2f}")
    print(
        f"  TV <= 0.05 at sigma=1: normal approximation m={m_normal:.1f}  exact smallest m={m_exact}  "
        f"TV there={average_case_tv(m_exact, sigma):.4f}  normal={average_case_tv_normal(m_exact, sigma):.4f}"
    )

    print("[Exercise 4]")
    for m in (1, 2, 3, 10):
        print(f"  m={m}: fraction recovered={count_reconstruction_accuracy(m):.4f}")

    print("[Exercise 5]")
    n, values = 1000, 1000
    tpr, fpr = release_rates_discrete(n, values)
    s_tpr, s_fpr = simulate_release_discrete(rng, n, values, 2_000_000)
    print(
        f"  n={n}, {values} values: TPR={tpr:.6f}  FPR={fpr:.6f}  advantage={tpr - fpr:.6f}  "
        f"precision at prior 1/2={precision(tpr, fpr, 0.5):.4f}"
    )
    print(f"  simulated: TPR={s_tpr:.6f}  FPR={s_fpr:.6f}")

    print("[Exercise 6]")
    n, d = 100, 50
    mean_in, var_in, var_out = inner_product_moments(n, d)
    print(f"  n={n}, d={d}: mean in={mean_in:.4f}  sd in={math.sqrt(var_in):.4f}  sd out={math.sqrt(var_out):.4f}")
    print(
        f"  normal approximation={inner_product_advantage_normal(n, d):.4f}  "
        f"simulated={simulated_inner_product_advantage(rng, n, d, 100_000):.4f}"
    )

    print("[Exercise 7]")
    mixed, known = two_world_mixture_tv()
    print(f"  one unknown bit: TV of mixtures={mixed:.4f}  rest known={known:.4f}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=FIGURES, help="ignored: this script makes no figures")
    args = parser.parse_args()
    seed: int = args.seed
    print_solutions(np.random.default_rng(seed))


if __name__ == "__main__":
    main()
