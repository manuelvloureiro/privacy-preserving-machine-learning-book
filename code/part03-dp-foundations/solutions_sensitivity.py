"""Compute the numbers quoted in book-exercises/part03-dp-foundations/06-sensitivity.md.

    uv run python code/part03-dp-foundations/solutions_sensitivity.py

Each exercise of the "Beyond global sensitivity" chapter has a block in print_table. The
functions reuse the chapter's script, code/sensitivity.py.
"""

import argparse
import math
from pathlib import Path

import numpy as np
from sensitivity import (
    AGE_MAX,
    DELTA,
    EPS,
    SAA_BLOCKS,
    SAA_CLIP,
    SAA_N,
    SAA_RATE,
    Array,
    ages,
    cauchy_params,
    distance_to_instability,
    median_a_profile,
    median_smooth_sensitivity,
    order_stat_a_profile,
    ptr_pass_probability,
    saa_error,
    smooth_from_profile,
)

# Figures are build output: _build/figures/<part>/, which the build copies into each book.
FIGURES = Path(__file__).resolve().parents[2] / "_build" / "figures" / Path(__file__).parent.name

# --- Exercise 1: local sensitivity of the mean ----------------------------------------------


def mean_local_sensitivity(x: Array, lam: float) -> float:
    """max over one replacement of |change in the mean| = max_i max(x_i, Λ - x_i)/n."""
    return float(np.max(np.maximum(x, lam - x))) / len(x)


# --- Exercise 2: the sliding constant of h(z) ∝ 1/(1 + |z|^γ) -------------------------------


def sliding_constant(gamma: float) -> float:
    """sup_u γ u^{γ-1}/(1 + u^γ) = (γ - 1)^{(γ-1)/γ}, attained at u^γ = γ - 1."""
    return (gamma - 1) ** ((gamma - 1) / gamma)


def sliding_constant_numeric(gamma: float) -> float:
    u = np.linspace(0.0, 50.0, 500_001)
    return float(np.max(gamma * u ** (gamma - 1) / (1 + u**gamma)))


# --- Exercise 4: propose-test-release --------------------------------------------------------


def ptr_distance_needed(eps: float, delta: float, target: float) -> float:
    """Smallest real distance d with Pr[d + Lap(1/ε) > ln(1/δ)/ε] >= target (for target > 1/2)."""
    return (math.log(1 / delta) + math.log(1 / (2 * (1 - target)))) / eps


# --- Exercise 5: subsample-and-aggregate block count ----------------------------------------


def saa_blocks_approx(n: int, rate: float, clip: float, eps: float) -> float:
    """Minimizer of (θB/n)² + 2(R/(Bε))²: B* = (2R²n²/(θ²ε²))^{1/4}."""
    return (2 * clip**2 * n**2 / (rate**2 * eps**2)) ** 0.25


# --- Exercise 6: clipping exponential data ---------------------------------------------------


def exp_clip_objective(c: float, n: int, eps: float) -> float:
    """Squared bias e^{-2C} plus noise variance 2C²/(nε)², for X ~ Exp(1)."""
    return math.exp(-2 * c) + 2 * c * c / (n * eps) ** 2


def exp_clip_optimum(n: int, eps: float) -> float:
    """Root of e^{-2C} = 2C/(nε)², by bisection (the left side falls, the right side rises)."""
    lo, hi = 0.0, 50.0
    for _ in range(200):
        mid = (lo + hi) / 2
        if math.exp(-2 * mid) > 2 * mid / (n * eps) ** 2:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


# --- The solutions' numbers ------------------------------------------------------------------


def print_table(rng: np.random.Generator) -> None:
    x = ages(rng, 1001)  # the same draw as the chapter's ages data (first use of seed 0)
    _, beta_c = cauchy_params(EPS)

    print("[Exercise 1: local sensitivity of the mean of the ages]")
    ls = mean_local_sensitivity(x, AGE_MAX)
    print(
        f"  min age={float(np.min(x)):.2f}  max age={float(np.max(x)):.2f}  LS={ls:.5f}  "
        f"GS=Λ/n={AGE_MAX / len(x):.5f}  lower bound Λ/(2n)={AGE_MAX / (2 * len(x)):.5f}"
    )

    print("[Exercise 2: sliding constant]")
    for gamma in (2.0, 3.0, 4.0):
        print(
            f"  gamma={gamma:g}: formula={sliding_constant(gamma):.4f}  numeric={sliding_constant_numeric(gamma):.4f}"
        )
    s = median_smooth_sensitivity(x, AGE_MAX, beta_c)
    print(f"  ages: S*={s:.4f}  NRS scale 6S*/ε={6 * s / EPS:.3f}  sharp scale 2S*/ε={2 * s / EPS:.3f}")

    print("[Exercise 3: smooth sensitivity of the 90th percentile]")
    n = len(x)
    r = 901
    xs = np.sort(x)
    a90 = order_stat_a_profile(x, AGE_MAX, r)
    s90, k90 = smooth_from_profile(a90, beta_c)
    s50, k50 = smooth_from_profile(median_a_profile(x, AGE_MAX), beta_c)
    print(
        f"  n={n} r={r}: x_r={float(xs[r - 1]):.2f}  LS={float(a90[0]):.4f}  S*={s90:.4f} at k={k90}  "
        f"noise scale 6S*/ε={6 * s90 / EPS:.3f}"
    )
    print(f"  median: S*={s50:.4f} at k={k50}")

    print("[Exercise 4: propose-test-release threshold, ε=1, δ=1e-6]")
    d = ptr_distance_needed(EPS, DELTA, 0.99)
    need = math.ceil(d)
    print(
        f"  distance needed for pass probability 0.99: {d:.2f} -> {need}  "
        f"pass probability at {need}: {ptr_pass_probability(need, EPS, DELTA):.4f}"
    )
    a = median_a_profile(x, AGE_MAX)
    b_min = float(a[need - 1])
    print(
        f"  smallest b with distance >= {need}: A^({need - 1})={b_min:.4f}  "
        f"distance at b={distance_to_instability(a, b_min):g}  at b=1: {distance_to_instability(a, 1.0):g}"
    )

    print("[Exercise 5: number of blocks for subsample-and-aggregate]")
    b_approx = saa_blocks_approx(SAA_N, SAA_RATE, SAA_CLIP, EPS)
    divisors = [b for b in range(3, SAA_N // 3 + 1) if SAA_N % b == 0]
    best = min(divisors, key=lambda b: saa_error(SAA_N, b, SAA_RATE, SAA_CLIP, EPS).rmse)
    print(
        f"  approximate B*={b_approx:.1f}  exact best divisor={best}  "
        f"rmse={saa_error(SAA_N, best, SAA_RATE, SAA_CLIP, EPS).rmse:.4f}  "
        f"grid of the section {SAA_BLOCKS[0]}..{SAA_BLOCKS[-1]}"
    )
    print(
        "  "
        + "  ".join(f"B={b}: rmse={saa_error(SAA_N, b, SAA_RATE, SAA_CLIP, EPS).rmse:.4f}" for b in (200, 250, 400))
    )

    print("[Exercise 6: clipping Exp(1) data, n=1000, ε=1]")
    c = exp_clip_optimum(1000, EPS)
    print(
        f"  C*={c:.3f}  bias={math.exp(-c):.4f}  noise sd={math.sqrt(2) * c / 1000:.4f}  "
        f"rmse (no sampling term)={math.sqrt(exp_clip_objective(c, 1000, EPS)):.4f}"
    )
    for n_big in (100, 10_000):
        print(f"  n={n_big}: C*={exp_clip_optimum(n_big, EPS):.3f}")

    print("[Exercise 7: the degenerate pair, Λ=100, β=1/6]")
    lam = 100.0
    x0 = (0.0, 0.0, 0.0, 0.0, lam)
    x1 = (0.0, 0.0, 0.0, lam, lam)
    for name, data in (("x", x0), ("x'", x1)):
        prof = median_a_profile(data, lam)
        sv, kv = smooth_from_profile(prof, beta_c)
        print(f"  {name}: A=" + ",".join(f"{v:g}" for v in prof) + f"  S*={sv:.3f} at k={kv}")
    s0 = median_smooth_sensitivity(x0, lam, beta_c)
    s1 = median_smooth_sensitivity(x1, lam, beta_c)
    print(f"  ratio S*(x')/S*(x)={s1 / s0:.4f}  e^β={math.exp(beta_c):.4f}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=FIGURES, help="accepted for a uniform interface; no figures")
    args = parser.parse_args()
    seed: int = args.seed
    print_table(np.random.default_rng(seed))


if __name__ == "__main__":
    main()
