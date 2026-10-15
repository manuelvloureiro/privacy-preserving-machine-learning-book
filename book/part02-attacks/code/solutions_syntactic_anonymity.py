"""Check the worked solutions to the exercises of the syntactic anonymity chapter.

    uv run python book/part02-attacks/code/solutions_syntactic_anonymity.py

Prints every number quoted in solutions/08-syntactic-anonymity.md (seed 0). Makes no figures;
--out is accepted and ignored.
"""

import argparse
import math
from pathlib import Path

import numpy as np
from syntactic_anonymity import (
    POPULATION,
    draw_table,
    k_anonymous,
    load_for_violation,
    mondrian,
    ordered_emd,
)

FIGURES = Path(__file__).resolve().parent.parent / "figures"

# --- Exercise 2: entropy l-diversity and the most common value -------------------------------


def top_heavy_entropy(q: float, m: int) -> float:
    """Entropy in bits of (q, (1 - q)/(m - 1), ..., (1 - q)/(m - 1))."""
    rest = (1 - q) / (m - 1)
    return -q * math.log2(q) - (m - 1) * rest * math.log2(rest)


def largest_top_share(l: int, m: int) -> float:
    """The largest q for which the top-heavy distribution is still entropy l-diverse."""
    lo, hi = 1 / m, 1 - 1e-12  # the entropy falls from log2 m to 0 as q goes from 1/m to 1
    for _ in range(200):
        mid = (lo + hi) / 2
        if top_heavy_entropy(mid, m) >= math.log2(l):
            lo = mid
        else:
            hi = mid
    return lo


# --- Exercise 4: intersecting two l-diverse classes ------------------------------------------


def pinned_l_diverse(m: int, l: int) -> float:
    """Each class holds the true value and l - 1 others chosen uniformly from the remaining m - 1:
    the chance that the two sets of others are disjoint."""
    return math.comb(m - l, l - 1) / math.comb(m - 1, l - 1)


def pinned_l_diverse_simulated(rng: np.random.Generator, m: int, l: int, trials: int) -> float:
    hits = 0
    for _ in range(trials):
        a = set(rng.choice(m - 1, size=l - 1, replace=False).tolist())
        b = set(rng.choice(m - 1, size=l - 1, replace=False).tolist())
        hits += not (a & b)
    return hits / trials


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=FIGURES, help="ignored: this script makes no figures")
    args = parser.parse_args()
    rng = np.random.default_rng(args.seed)

    print("== Exercise 1: bits that 10-anonymity allows for a million people")
    n = 1_000_000
    lam = load_for_violation(10, 0.01)
    print(
        f"lambda={lam:.2f}  log2 n={math.log2(n):.2f}  bits allowed={math.log2(n / lam):.2f}  "
        f"below log2 n by {math.log2(lam):.2f}"
    )
    for k in (2, 10, 100):
        lam_k = load_for_violation(k, 0.01)
        print(
            f"k={k}: lambda={lam_k:.2f}  lambda/k={lam_k / k:.2f}  (k + 2.33 sqrt(k))/k={(k + 2.33 * math.sqrt(k)) / k:.2f}"
        )

    print("\n== Exercise 2: entropy 3-diversity over 8 values")
    q = largest_top_share(3, 8)
    print(f"largest top share q={q:.3f}  check entropy={top_heavy_entropy(q, 8):.4f}  log2 3={math.log2(3):.4f}")
    print(
        f"recursive (c, 3) with this share needs c > 7q/(6(1 - q))={7 * q / (6 * (1 - q)):.2f}; "
        f"posterior bound c/(1+c) at that c={7 * q / (6 * (1 - q)) / (1 + 7 * q / (6 * (1 - q))):.3f}"
    )
    print(f"top share allowed by recursive (2, 3): q < 12/19={12 / 19:.3f}")

    print("\n== Exercise 3: ordered earth mover's distance (salaries 3k..11k)")
    table = [1 / 9] * 9
    low = [1 / 3, 1 / 3, 1 / 3, 0, 0, 0, 0, 0, 0]
    spread = [0, 0, 0, 1 / 3, 0, 1 / 3, 0, 0, 1 / 3]
    print(f"class {{3k,4k,5k}}: EMD={ordered_emd(low, table):.3f}")
    print(f"class {{6k,8k,11k}}: EMD={ordered_emd(spread, table):.3f}")

    print("\n== Exercise 4: two l-diverse classes, m values")
    for m, l in ((8, 3), (8, 2), (20, 3)):
        sim = pinned_l_diverse_simulated(rng, m, l, 100_000)
        print(f"m={m} l={l}: pinned={pinned_l_diverse(m, l):.4f}  simulated={sim:.4f}")

    print("\n== Exercise 5: Mondrian class sizes")
    hospital = draw_table(rng, POPULATION)
    for k in (5, 10):
        sizes = np.bincount(mondrian(hospital, k_anonymous(k)))
        within = float(np.mean(sizes <= 2 * k - 1))
        print(
            f"k={k}: sizes {int(sizes.min())}..{int(sizes.max())}  classes within [k, 2k-1]={within:.3f}  "
            f"linkage={len(sizes) / POPULATION:.4f}  bounds 1/(2k-1)={1 / (2 * k - 1):.4f}  1/k={1 / k:.4f}"
        )


if __name__ == "__main__":
    main()
