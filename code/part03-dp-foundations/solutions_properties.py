"""Check the worked solutions to the exercises of the properties chapter.

    uv run python code/part03-dp-foundations/solutions_properties.py

Prints every number quoted in book-exercises/part03-dp-foundations/02-properties.md (seed 0; nothing here is random, the
seed is accepted for a uniform interface). Makes no figures; --out is accepted and ignored.

Conventions as in properties.py: P = M(D), Q = M(D') as probability lists, logs are natural.
"""

import argparse
import math
from pathlib import Path

import numpy as np
from properties import (
    EPS_RR,
    alarm_mechanism,
    composed_rr_epsilon,
    composed_rr_mean_loss,
    composed_rr_pmfs,
    group_delta,
    group_delta_simple,
    histogram_epsilon,
    hockey_stick,
    privacy_profile,
    product,
    pure_epsilon,
    rounded_laplace,
    rr_output,
    rr_truth_probability,
)

# Figures are build output: _build/figures/<part>/, which the build copies into each book.
FIGURES = Path(__file__).resolve().parents[2] / "_build" / "figures" / Path(__file__).parent.name

# --- Exercise 1: majority of three answers -----------------------------------------------------


def majority_of_three(bit: int, eps0: float) -> list[float]:
    """[P(majority 0), P(majority 1)] of three independent randomized-response answers."""
    s = rr_output(bit, eps0)[1]  # probability that one answer is 1
    one = s**3 + 3 * s**2 * (1 - s)
    return [1 - one, one]


# --- Exercise 3: records to users to releases --------------------------------------------------


def user_level(eps: float, delta: float, records: int) -> tuple[float, float]:
    """Group privacy for a user with `records` records: (kε, δ Σ_{j<k} e^{jε})."""
    return records * eps, group_delta(eps, delta, records)


# --- Exercise 5: shared noise ------------------------------------------------------------------


def point_mass(index: int, size: int) -> list[float]:
    return [1.0 if i == index else 0.0 for i in range(size)]


# --- Exercise 6: Hoeffding preview of advanced composition -------------------------------------


def hoeffding_epsilon(eps0: float, k: int, delta: float) -> float:
    """ε = E[L] + ε0 √(2k ln(1/δ)), from Hoeffding on a sum of k terms in [-ε0, ε0]."""
    return composed_rr_mean_loss(eps0, k) + eps0 * math.sqrt(2 * k * math.log(1 / delta))


def composed_rr_loss_tail(eps0: float, k: int, eps: float) -> float:
    """Pr_D[L > ε] for k randomized-response answers: L = (2j - k) ε0 with j ~ Bin(k, t)."""
    p, _ = composed_rr_pmfs(eps0, k)
    losses = (2 * np.arange(k + 1) - k) * eps0
    return float(np.sum(p[losses > eps]))


# --- Exercise 7: ask until yes -----------------------------------------------------------------


def ask_until_yes(bit: int, eps0: float, cap: int) -> list[float]:
    """Distribution of the number of answers asked: stop at the first 1, at most `cap` answers.

    Entries 0..cap-1 are "stopped after i + 1 answers"; the last entry is "cap answers and no 1".
    """
    s = rr_output(bit, eps0)[1]
    out = [(1 - s) ** i * s for i in range(cap)]
    out.append((1 - s) ** cap)
    return out


# --- The page's numbers ----------------------------------------------------------------------


def print_table(rng: np.random.Generator) -> None:
    del rng  # nothing random here

    print("[Exercise 1: majority of three answers, eps0 = ln 3]")
    p, q = majority_of_three(1, EPS_RR), majority_of_three(0, EPS_RR)
    print(
        f"  P(majority 1 | bit 1)={p[1]:.5f} (54/64={54 / 64:.5f})  P(majority 1 | bit 0)={q[1]:.5f} "
        f"(10/64={10 / 64:.5f})  ratio={p[1] / q[1]:.4f}"
    )
    print(f"  exact eps={pure_epsilon(p, q):.4f}  ln 5.4={math.log(5.4):.4f}  composition bound={3 * EPS_RR:.4f}")

    print("[Exercise 2: geometric sum against the simple form]")
    eps, delta, k = 0.5, 1e-5, 10
    print(
        f"  eps={eps:g} delta={delta:g} k={k}: sum={group_delta(eps, delta, k):.3e}  "
        f"simple={group_delta_simple(eps, delta, k):.3e}"
    )
    for e in (1.0, 0.1, 0.01):
        print(f"  eps={e:g}, k={k}: simple/sum={group_delta_simple(e, delta, k) / group_delta(e, delta, k):.4f}")
    a0 = alarm_mechanism(eps, delta, k, 0)[0]
    worst = max(
        privacy_profile(alarm_mechanism(eps, delta, k, c), alarm_mechanism(eps, delta, k, c + 1), eps)
        for c in range(k + 2)
    )
    print(
        f"  alarm mechanism, size {k}: a_0={a0:.3e}  1 - e^-eps={1 - math.exp(-eps):.4f}  "
        f"largest neighbouring delta(eps)={worst:.3e}"
    )

    print("[Exercise 3: user with 5 records, (0.2, 1e-7) per record, then 10 runs]")
    eps_u, delta_u = user_level(0.2, 1e-7, 5)
    print(f"  user level: eps={eps_u:.4f}  delta={delta_u:.3e}")
    print(f"  10 runs: eps={10 * eps_u:.4f}  delta={10 * delta_u:.3e}")
    eps_r, delta_r = user_level(10 * 0.2, 10 * 1e-7, 5)
    print(
        f"  compose first, then group: eps={eps_r:.4f}  delta={delta_r:.3e}  "
        f"ratio to group first={delta_r / (10 * delta_u):.0f}"
    )

    print("[Exercise 4: histogram with rounded Lap(0, 1) per bin]")
    print(
        f"  add/remove: eps={histogram_epsilon(1.0, False):.4f}  replace across bins: "
        f"eps={histogram_epsilon(1.0, True):.4f}  replace within a slot: eps={histogram_epsilon(1.0, False):.4f}"
    )
    print(
        f"  b=2: add/remove eps={histogram_epsilon(2.0, False):.4f}  replace across bins "
        f"eps={histogram_epsilon(2.0, True):.4f}"
    )

    print("[Exercise 5: shared noise]")
    size = 3  # the difference c1 - c2 takes the value c (index 1) under D and c - 1 (index 0) under D'
    pd, pd2 = point_mass(1, size), point_mass(0, size)
    print("  " + "  ".join(f"E_(e^{e:g})={hockey_stick(pd, pd2, e):.4f}" for e in (0.0, 1.0, 5.0, 20.0)))
    fresh_d = product(rounded_laplace(1.0, 1.0, -40, 41), rounded_laplace(0.0, 1.0, -40, 41))
    fresh_d2 = product(rounded_laplace(0.0, 1.0, -40, 41), rounded_laplace(0.0, 1.0, -40, 41))
    print(f"  fresh noise: pure eps of the pair={pure_epsilon(fresh_d, fresh_d2):.4f}")

    print("[Exercise 6: Hoeffding bound for k randomized-response answers]")
    eps0, k, delta = 0.1, 1000, 1e-6
    bound = hoeffding_epsilon(eps0, k, delta)
    print(
        f"  eps0={eps0:g} k={k} delta={delta:g}: mean={composed_rr_mean_loss(eps0, k):.3f}  "
        f"spread term={eps0 * math.sqrt(2 * k * math.log(1 / delta)):.3f}  Hoeffding eps={bound:.2f}  "
        f"exact eps={composed_rr_epsilon(eps0, k, delta):.3f}  basic={k * eps0:.0f}"
    )
    print(f"  exact P_D[L > {bound:.2f}]={composed_rr_loss_tail(eps0, k, bound):.3e}")
    print(
        f"  truth probability at eps0={eps0:g}: {rr_truth_probability(eps0):.4f}  "
        f"2t - 1={2 * rr_truth_probability(eps0) - 1:.5f}  tanh(eps0/2)={math.tanh(eps0 / 2):.5f}"
    )

    print("[Exercise 7: ask until yes, at most 3 answers, eps0 = ln 3]")
    p, q = ask_until_yes(1, EPS_RR, 3), ask_until_yes(0, EPS_RR, 3)
    print("  bit 1: " + "  ".join(f"{x:.6f}" for x in p))
    print("  bit 0: " + "  ".join(f"{x:.6f}" for x in q))
    print("  ratios: " + "  ".join(f"{a / b:.4f}" for a, b in zip(p, q, strict=True)))
    print(f"  exact eps={pure_epsilon(p, q):.4f}  3 ln 3={3 * EPS_RR:.4f}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=FIGURES, help="accepted for a uniform interface; no figures")
    args = parser.parse_args()
    seed: int = args.seed
    print_table(np.random.default_rng(seed))


if __name__ == "__main__":
    main()
