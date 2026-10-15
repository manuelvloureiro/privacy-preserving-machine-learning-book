"""Check the worked solutions to the exercises of the user-level differential privacy chapter.

    uv run python code/part05-dp-ml/solutions_user_level.py

Prints every number quoted in book-exercises/part05-dp-ml/07-user-level.md (seed 0). Makes no figures;
--out is accepted and ignored.
"""

import argparse
import math
from pathlib import Path

import numpy as np
from dp_erm import gaussian_sigma
from dp_sgd import noise_multiplier_for, rdp_curve
from user_level import (
    ORDERS,
    eps_from_curve_log,
    log_binom_pmf,
    make_task,
    make_users,
    pair_moment_alpha2,
    unit_rdp_integer,
)

# Figures are build output: _build/figures/<part>/, which the build copies into each book.
FIGURES = Path(__file__).resolve().parents[2] / "_build" / "figures" / Path(__file__).parent.name

# --- Exercise 1: group privacy of an (ε, δ) guarantee --------------------------------------------------


def group_delta(eps: float, delta: float, k: int) -> float:
    """δ_k = δ (e^{kε} - 1)/(e^ε - 1), the δ of group privacy for k records."""
    return delta * math.expm1(k * eps) / math.expm1(eps)


# --- Exercise 2: when group privacy gives nothing ------------------------------------------------------


def log10_user_delta(log_delta_rec: float, q: float, sigma: float, steps: int, k: int) -> tuple[float, float]:
    """(log10 of the user's δ, record ε) when the record-level guarantee is taken at δ_rec = e^{log_delta_rec}."""
    rdp = [steps * e for e in rdp_curve(q, sigma)]
    e = eps_from_curve_log(ORDERS, rdp, log_delta_rec)[0]
    log_ratio = k * e + math.log1p(-math.exp(-k * e)) - e - math.log1p(-math.exp(-e))
    return (log_delta_rec + log_ratio) / math.log(10), e


# --- Exercise 3: the second moment and the rate-kq approximation ----------------------------------------


def rate_kq_alpha2(q: float, sigma: float, k: int) -> float:
    """ε(2) of one record sampled at rate kq: ln(1 + (kq)² (e^{1/σ²} - 1))."""
    return math.log1p((k * q) ** 2 * math.expm1(1 / sigma**2))


def pair_moment_alpha2_check(q: float, sigma: float, k: int) -> float:
    """The same A_2 by enumerating the binomial distribution once more (sanity check of the formula)."""
    p = np.exp(log_binom_pmf(k, q))
    total = 0.0
    for a in range(k + 1):
        for b in range(k + 1):
            total += float(p[a] * p[b]) * math.exp(a * b / sigma**2)
    return math.log(total)


# --- Exercise 5: averaging or summing before clipping --------------------------------------------------


def influence(count: int, norm: float, clip: float) -> tuple[float, float]:
    """Length of one user's clipped contribution when its `count` records all have gradient g with ‖g‖ = norm:
    (average then clip, sum then clip)."""
    return min(norm, clip), min(count * norm, clip)


# --- Exercise 6: capping contributions to a count --------------------------------------------------------


def cap_count_errors(counts: list[int], noise_unit: float) -> list[tuple[int, int, float, float]]:
    """For each cap k: (k, records kept, bias n_k - n, root mean squared error of the noisy capped count)."""
    n = sum(counts)
    rows: list[tuple[int, int, float, float]] = []
    for k in range(1, max(counts) + 1):
        kept = sum(min(c, k) for c in counts)
        bias = float(kept - n)
        rows.append((k, kept, bias, math.sqrt(bias**2 + (k * noise_unit) ** 2)))
    return rows


def print_table(rng: np.random.Generator) -> None:
    del rng  # nothing random here
    print("== Exercise 1")
    for k in (2, 5, 10):
        d = group_delta(1.0, 1e-6, k)
        print(f"eps=1, delta=1e-6, k={k}: eps_k={k}  delta_k={d:.3e}")

    print("== Exercise 2")
    for log10_rec in (-5, -10, -20, -50, -100):
        parts: list[str] = []
        e = 0.0
        for k in (2, 5, 10):
            user, e = log10_user_delta(log10_rec * math.log(10), 0.01, 1.0, 10_000, k)
            parts.append(f"k={k}: user delta 10^{user:.1f}")
        print(f"record delta 10^{log10_rec}: record eps {e:.2f}  " + "  ".join(parts))

    print("== Exercise 3")
    for k in (2, 5, 10):
        exact = pair_moment_alpha2(0.01, 1.0, k)
        print(
            f"k={k}, q=0.01, sigma=1: eps(2) exact {exact:.6e}  enumeration {pair_moment_alpha2_check(0.01, 1.0, k):.6e}  "
            f"rate kq {rate_kq_alpha2(0.01, 1.0, k):.6e}  ratio {exact / rate_kq_alpha2(0.01, 1.0, k):.4f}"
        )

    print("== Exercise 4")
    s1 = noise_multiplier_for(0.1, 1_000, 4.0, 1e-5)
    print(f"user-level DP-SGD, q=0.1, T=1000, eps=4: sigma={s1:.3g}")
    for k in (2, 5):
        print(
            f"k={k}: per-record clipping, summed: sigma={k * s1:.3g}  "
            f"check at alpha=3: {unit_rdp_integer(3, 0.1, k * s1, k):.6e} vs {unit_rdp_integer(3, 0.1, s1, 1):.6e}"
        )

    print("== Exercise 5")
    for count in (1, 10, 348):
        avg, tot = influence(count, 0.01, 0.5)
        print(f"user with {count} records, |g|=0.01, C=0.5: average-then-clip {avg:.3g}  sum-then-clip {tot:.3g}")

    print("== Exercise 6")
    users = make_users(make_task().train)
    counts = [int(c) for c in users.counts]
    unit = gaussian_sigma(1.0, 1.0, 1e-5)
    print(f"Gaussian noise for one unit of sensitivity at (1, 1e-5): {unit:.3f}")
    rows = cap_count_errors(counts, unit)
    for k, kept, bias, rmse in rows:
        if k in (1, 2, 5, 10, 20, 50, 100, 200, 348):
            print(f"cap k={k}: kept={kept}  bias={bias:.0f}  noise sd={k * unit:.1f}  rmse={rmse:.1f}")
    best = min(rows, key=lambda r: r[3])
    print(f"best cap k={best[0]}: kept={best[1]}  rmse={best[3]:.1f}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=FIGURES, help="ignored: this script makes no figures")
    args = parser.parse_args()
    print_table(np.random.default_rng(args.seed))


if __name__ == "__main__":
    main()
