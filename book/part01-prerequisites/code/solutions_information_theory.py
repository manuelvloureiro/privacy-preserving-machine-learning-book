"""Check the worked solutions to the exercises of the information-theory chapter.

    uv run python book/part01-prerequisites/code/solutions_information_theory.py

Prints every number quoted in solutions/05-information-theory.md. Everything is an exact
computation, so the seed is unused and no figures are drawn; --seed and --out are accepted
for the same interface as the chapter's script. All logarithms are base 2 (bits).
"""

import argparse
import math
from pathlib import Path

import numpy as np
from information_theory import (
    GUESS_CHANNEL,
    GUESS_PRIOR,
    averaged_min_entropy,
    binary_entropy,
    blahut_arimoto,
    bsc,
    conditional_entropy,
    conditional_min_entropy,
    fano_bound,
    fano_lhs,
    fano_sample_bound,
    joint_from,
    map_error,
    min_capacity,
    min_entropy,
    mutual_information,
    reveal_if_zero_channel,
    symmetric_channel,
    symmetric_channel_capacity,
    uniform,
    z_channel,
)

FIGURES = Path(__file__).resolve().parent.parent / "figures"


# --- Exercise 1: entropy of a geometric variable ---------------------------------------------


def geometric_pmf(k_max: int) -> list[float]:
    """Pr[X = k] = 2^-k for k = 1..k_max (the tail beyond k_max is dropped)."""
    return [2.0**-k for k in range(1, k_max + 1)]


def truncated_entropy(p: list[float]) -> float:
    """Σ p(k) log2(1/p(k)) over the listed values only (no renormalization)."""
    return sum(-pk * math.log2(pk) for pk in p if pk > 0)


# --- Exercise 2: information from two observations ------------------------------------------


def xor_joint() -> list[list[float]]:
    """Rows X ∈ {0, 1}; columns (Y1, Y2) ∈ {00, 01, 10, 11}; X = Y1 xor Y2 for fair independent bits."""
    return [[0.25 if (y1 ^ y2) == x else 0.0 for y1 in (0, 1) for y2 in (0, 1)] for x in (0, 1)]


def two_bsc_joint(p: float) -> list[list[float]]:
    """X a fair bit, Y1 and Y2 two independent BSC(p) copies of it. Columns (Y1, Y2) as above."""
    c = bsc(p)
    return [[0.5 * c[x][y1] * c[x][y2] for y1 in (0, 1) for y2 in (0, 1)] for x in (0, 1)]


def marginalize_columns(joint: list[list[float]], keep: int) -> list[list[float]]:
    """From a joint over X and (Y1, Y2) (columns 2*y1 + y2), keep only Y1 (keep=0) or Y2 (keep=1)."""
    return [
        [sum(row[2 * y1 + y2] for y1 in (0, 1) for y2 in (0, 1) if (y1, y2)[keep] == v) for v in (0, 1)]
        for row in joint
    ]


# --- Exercise 3: min-capacity against capacity for the BSC ----------------------------------


def bsc_min_capacity(p: float) -> float:
    """log2(2 max(p, 1 - p))."""
    return math.log2(2 * max(p, 1 - p))


def worst_bsc_gap(points: int) -> float:
    """min over a grid of p in [0, 1] of log2(2 max(p, 1-p)) - (1 - h(p)); must be >= 0."""
    return min(bsc_min_capacity(i / points) - (1 - binary_entropy(i / points)) for i in range(points + 1))


# --- Exercise 4: the Z-channel --------------------------------------------------------------


def z_information(q: float) -> float:
    """Closed form I(X; Y) = h(q/2) - q for the Z-channel with Pr[X = 1] = q."""
    return binary_entropy(q / 2) - q


def z_information_gap(points: int) -> float:
    """Largest difference between the closed form and the joint-table formula over a grid of q."""
    return max(
        abs(z_information(i / points) - mutual_information(joint_from([1 - i / points, i / points], z_channel(0.5))))
        for i in range(points + 1)
    )


def grid_argmax(points: int) -> float:
    """The q on a grid of step 1/points that maximizes the Z-channel information."""
    return max((i / points for i in range(points + 1)), key=z_information)


# --- Exercise 5: capacities of the M-ary symmetric channel ----------------------------------


def symmetric_min_capacity(m: int, eps: float) -> float:
    """log2(M (1 - eps)), valid when 1 - eps >= eps / (M - 1)."""
    return math.log2(m * (1 - eps))


# --- Exercise 6: Smith versus the average of logs -------------------------------------------


def half_reveal_smith(n: int) -> float:
    """Smith's H_∞(X | Y) for reveal-if-zero(1) on a uniform n-bit secret: n - log2(2^(n-1) + 1)."""
    return n - math.log2(2 ** (n - 1) + 1)


def half_reveal_average(n: int) -> float:
    """Σ_y p(y) H_∞(X | Y = y) for reveal-if-zero(1): half the time 0, half the time n - 1."""
    return (n - 1) / 2


# --- Exercise 7: Fano is tight on the symmetric channel -------------------------------------


def symmetric_fano_gap(m: int, eps: float) -> float:
    """|H(X | Y) - (h(ε) + ε log2(M - 1))| for uniform input; zero when Fano is tight at P_e = ε."""
    jf = joint_from(uniform(m), symmetric_channel(m, eps))
    return abs(conditional_entropy(jf) - fano_lhs(eps, m))


# --- Exercise 8: a sample-size lower bound --------------------------------------------------


def fano_samples_needed(kl_bits: float, m: int, target: float) -> float:
    """Smallest n with 1 - (n κ + 1) / log2 M <= target: n >= ((1 - target) log2 M - 1) / κ."""
    return ((1 - target) * math.log2(m) - 1) / kl_bits


# --- The solutions' table -------------------------------------------------------------------


def print_table(rng: np.random.Generator) -> None:  # every number here is exact, so rng is unused
    del rng
    print("== Exercise 1: geometric ==")
    p = geometric_pmf(200)
    print(f"[geometric, k<=200] H={truncated_entropy(p):.4f}  H_min={min_entropy(p):.4f}")
    for k_max in (5, 10, 20):
        print(f"[geometric, k<={k_max}] partial sum of k 2^-k={truncated_entropy(geometric_pmf(k_max)):.4f}")

    print("== Exercise 2: two observations ==")
    jx = xor_joint()
    print(
        f"[xor] I(X;Y1)={mutual_information(marginalize_columns(jx, 0)):.4f}  "
        f"I(X;Y2)={mutual_information(marginalize_columns(jx, 1)):.4f}  I(X;Y1,Y2)={mutual_information(jx):.4f}"
    )
    jb = two_bsc_joint(0.1)
    print(
        f"[two BSC(0.1) copies] I(X;Y1)={mutual_information(marginalize_columns(jb, 0)):.4f}  "
        f"I(X;Y2)={mutual_information(marginalize_columns(jb, 1)):.4f}  I(X;Y1,Y2)={mutual_information(jb):.4f}  "
        f"sum={mutual_information(marginalize_columns(jb, 0)) + mutual_information(marginalize_columns(jb, 1)):.4f}"
    )

    print("== Exercise 3: BSC min-capacity vs capacity ==")
    for q in (0.11, 0.25):
        print(f"[BSC p={q}] 1-h(p)={1 - binary_entropy(q):.4f}  log2(2max)={bsc_min_capacity(q):.4f}")
    print(f"[BSC grid of 1001 p] min gap={worst_bsc_gap(1000):.4f}")

    print("== Exercise 4: Z-channel ==")
    print(f"[Z] closed form vs joint table, max gap={z_information_gap(1000):.1e}")
    print(
        f"[Z] grid argmax q={grid_argmax(1000):.4f}  I(0.4)={z_information(0.4):.4f}  "
        f"log2(5/4)={math.log2(5 / 4):.4f}  I(0.5)={z_information(0.5):.4f}"
    )
    ba, r = blahut_arimoto(z_channel(0.5))
    print(f"[Z] Blahut-Arimoto={ba:.4f}  optimal Pr[X=1]={r[1]:.4f}")

    print("== Exercise 5: M-ary symmetric channel ==")
    ch = symmetric_channel(8, 0.3)
    ba, r = blahut_arimoto(ch)
    print(
        f"[M=8 eps=0.3] h(eps)={binary_entropy(0.3):.4f}  eps*log2(M-1)={0.3 * math.log2(7):.4f}  "
        f"capacity={symmetric_channel_capacity(8, 0.3):.4f}  Blahut-Arimoto={ba:.4f}  "
        f"max |r - 1/M|={max(abs(v - 1 / 8) for v in r):.1e}"
    )
    print(
        f"[M=8 eps=0.3] eps/(M-1)={0.3 / 7:.4f}  min-capacity={symmetric_min_capacity(8, 0.3):.4f}  "
        f"from column maxima={min_capacity(ch):.4f}"
    )

    print("== Exercise 6: Smith vs average of logs ==")
    jg = joint_from(GUESS_PRIOR, GUESS_CHANNEL)
    print(f"[walk-through] Smith={conditional_min_entropy(jg):.4f}  average={averaged_min_entropy(jg):.4f}")
    jr = joint_from(uniform(2**6), reveal_if_zero_channel(6, 1))
    print(
        f"[reveal-if-zero(1), n=6, explicit] Smith={conditional_min_entropy(jr):.4f}  "
        f"average={averaged_min_entropy(jr):.4f}  closed forms: {half_reveal_smith(6):.4f}, "
        f"{half_reveal_average(6):.4f}"
    )
    print(f"[reveal-if-zero(1), n=64] Smith={half_reveal_smith(64):.4f}  average={half_reveal_average(64):.4f}")

    print("== Exercise 7: Fano on the symmetric channel ==")
    for eps in (0.1, 0.2, 0.5, 0.875):
        jf = joint_from(uniform(8), symmetric_channel(8, eps))
        hc = conditional_entropy(jf)
        print(
            f"[M=8 eps={eps}] MAP error={map_error(jf):.4f}  H(X|Y)={hc:.4f}  "
            f"h(eps)+eps*log2(7)={fano_lhs(eps, 8):.4f}  Fano bound={fano_bound(hc, 8):.4f}  "
            f"gap={symmetric_fano_gap(8, eps):.1e}"
        )

    print("== Exercise 8: sample size ==")
    print(f"[M=1024 kappa=0.01 target 0.1] n >= {fano_samples_needed(0.01, 1024, 0.1):.1f}")
    for n in (799, 800):
        print(f"[M=1024 kappa=0.01 n={n}] P_e >= {fano_sample_bound(n, 0.01, 1024):.4f}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=FIGURES, help="unused: these solutions draw no figures")
    args = parser.parse_args()
    seed: int = args.seed
    print_table(np.random.default_rng(seed))


if __name__ == "__main__":
    main()
