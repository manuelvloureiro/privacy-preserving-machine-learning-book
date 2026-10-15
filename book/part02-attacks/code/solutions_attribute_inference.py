"""Check the worked solutions to the exercises of the attribute inference chapter.

    uv run python book/part02-attacks/code/solutions_attribute_inference.py

Prints every number quoted in solutions/03-attribute-inference.md (seed 0 for the one
simulation). Makes no figures; --out is accepted and ignored.

The smoking table is J[s, x, y] with s = 1 smoker, x = 1 old, y = 1 disease, as in
attribute_inference.py.
"""

import argparse
import math
from pathlib import Path

import numpy as np
from attribute_inference import (
    P_DISEASE,
    P_OLD_GIVEN_S,
    Array,
    leakage_bits,
    prior_vulnerability,
    smoking_joint,
    vulnerability_given_x,
    vulnerability_given_xy,
)

FIGURES = Path(__file__).resolve().parent.parent / "figures"


def normal_cdf(x: float) -> float:
    """Φ(x); erfc keeps full relative precision in the lower tail."""
    return 0.5 * math.erfc(-x / math.sqrt(2))


def p_disease(s: int, x: int, y: int) -> float:
    """Pr[y | x, s] in the smoking table."""
    p1 = P_DISEASE[s][x]
    return p1 if y == 1 else 1 - p1


def p_s_given_x(joint: Array) -> Array:
    """Pr[s | x] as a 2×2 array indexed [s, x]."""
    sx = joint.sum(axis=2)
    return sx / sx.sum(axis=0)


# --- Exercise 2: the marginal prior instead of Pr[s | x] ------------------------------------


def marginal_prior_success(joint: Array) -> tuple[float, list[str]]:
    """Success of argmax_s π(s)·Pr[y | x, s], ties broken toward the more common value of s.

    Returns the success probability and a description of each (x, y) cell's decision.
    """
    prior = joint.sum(axis=(1, 2))
    success = 0.0
    notes: list[str] = []
    for x in (0, 1):
        for y in (0, 1):
            scores = [float(prior[s]) * p_disease(s, x, y) for s in (0, 1)]
            if math.isclose(scores[0], scores[1]):
                guess = int(np.argmax(prior))
                tag = "tie"
            else:
                guess = int(np.argmax(scores))
                tag = "ok"
            success += float(joint[guess, x, y])
            notes.append(f"x={x},y={y}: scores {scores[0]:.3f}/{scores[1]:.3f} -> s={guess} ({tag})")
    return success, notes


# --- Exercise 3: the reduction at low false positive rate -----------------------------------


def tpr_at_fpr(tpr: float, fpr: float, target_fpr: float) -> float:
    """Best TPR at a lower FPR from a test whose only output is one bit: fire with prob target/fpr."""
    return tpr * target_fpr / fpr


# --- Exercise 4: label-only inversion -------------------------------------------------------


def label_only_success(joint: Array) -> tuple[float, list[str]]:
    """Inversion against a classifier that predicts disease iff Pr[y = 1 | x, s] > 1/2.

    The attacker sees y, queries the label at both values of s, scores Pr[s | x]·1[pred(x, s) = y]
    and breaks ties with Pr[s | x].
    """
    psx = p_s_given_x(joint)
    success = 0.0
    notes: list[str] = []
    for x in (0, 1):
        for y in (0, 1):
            like = [1.0 if (P_DISEASE[s][x] > 0.5) == (y == 1) else 0.0 for s in (0, 1)]
            scores = [float(psx[s, x]) * like[s] for s in (0, 1)]
            guess = int(np.argmax(psx[:, x])) if scores[0] == scores[1] else int(np.argmax(scores))
            success += float(joint[guess, x, y])
            notes.append(f"x={x},y={y}: labels match {like} -> s={guess}")
    return success, notes


# --- Exercise 5: two independent Gaussian clues ---------------------------------------------


def two_clue_success(a: float, b: float) -> float:
    """Bayes success Φ(√(a² + b²)) for s uniform, clues N(±a, 1) and N(±b, 1)."""
    return normal_cdf(math.hypot(a, b))


def two_clue_simulation(rng: np.random.Generator, a: float, b: float, m: int) -> float:
    sign = np.where(rng.random(m) < 0.5, 1.0, -1.0)
    u = sign * a + rng.standard_normal(m)
    v = sign * b + rng.standard_normal(m)
    llr = 2 * a * u + 2 * b * v  # ln of the likelihood ratio for s = +1 against s = -1
    right: np.ndarray[tuple[int], np.dtype[np.bool_]] = np.equal(np.sign(llr), sign)
    return float(np.mean(right))


# --- Exercise 6: a model that memorizes its training records --------------------------------


def lookup_member_success(joint: Array) -> tuple[float, list[str]]:
    """Members against a lookup model: confidence 1 on the stored (x, s, y), the population elsewhere.

    The attacker guesses the true s iff Pr[s | x]·1 > Pr[s' | x]·Pr[y | x, s'].
    """
    psx = p_s_given_x(joint)
    success = 0.0
    notes: list[str] = []
    for s in (0, 1):
        for x in (0, 1):
            for y in (0, 1):
                other = 1 - s
                true_score = float(psx[s, x])
                other_score = float(psx[other, x]) * p_disease(other, x, y)
                right = true_score > other_score
                if right:
                    success += float(joint[s, x, y])
                notes.append(
                    f"s={s},x={x},y={y} (mass {joint[s, x, y]:.3f}): "
                    f"{true_score:.4f} vs {other_score:.4f} -> {'right' if right else 'wrong'}"
                )
    return success, notes


def print_table(rng: np.random.Generator) -> None:
    base = smoking_joint()

    print("[Exercise 1: smoking table with prior 0.5]")
    half = smoking_joint(prior=0.5)
    v0, vx, vxy = prior_vulnerability(half), vulnerability_given_x(half), vulnerability_given_xy(half)
    cells = "  ".join(f"x={x},y={y}:{half[1, x, y] / half[:, x, y].sum():.4f}" for x in (0, 1) for y in (0, 1))
    print(f"  Pr[smoker | x, y]: {cells}")
    print(
        f"  V(S)={v0:.4f}  V(S|X)={vx:.4f}  V(S|X,Y)={vxy:.4f}  "
        f"leakage from x={leakage_bits(vx, v0):.4f} bits  from (x, y)={leakage_bits(vxy, v0):.4f} bits"
    )
    print(f"  Pr[old | s] = {P_OLD_GIVEN_S}")

    print("[Exercise 2: marginal prior pi(s) instead of Pr[s | x]]")
    psx = p_s_given_x(base)
    print(f"  Pr[smoker | young]={psx[1, 0]:.4f}  Pr[smoker | old]={psx[1, 1]:.4f}")
    success, notes = marginal_prior_success(base)
    for note in notes:
        print(f"  {note}")
    print(f"  success={success:.4f}  imputation V(S|X,Y)={vulnerability_given_xy(base):.4f}")

    print("[Exercise 3: the attribute attack as a membership test at low FPR]")
    tpr, fpr = 0.7832, 0.7262
    for target in (0.1, 1e-3):
        print(f"  FPR={target:g}: TPR={tpr_at_fpr(tpr, fpr, target):.5f}  (ratio TPR/FPR={tpr / fpr:.4f})")

    print("[Exercise 4: label-only inversion on the smoking table]")
    success, notes = label_only_success(base)
    for note in notes:
        print(f"  {note}")
    print(
        f"  success={success:.4f}  imputation={vulnerability_given_xy(base):.4f}  "
        f"loss={vulnerability_given_xy(base) - success:.4f}"
    )

    print("[Exercise 5: two independent Gaussian clues]")
    a, b = 0.5, 1.0
    print(
        f"  a={a:g}: Phi(a)={normal_cdf(a):.4f}  b={b:g}: Phi(b)={normal_cdf(b):.4f}  "
        f"both: Phi(sqrt(a^2+b^2))=Phi({math.hypot(a, b):.4f})={two_clue_success(a, b):.4f}  "
        f"(simulated {two_clue_simulation(rng, a, b, 1_000_000):.4f})"
    )

    print("[Exercise 6: a model that memorizes]")
    success, notes = lookup_member_success(base)
    for note in notes:
        print(f"  {note}")
    imp = vulnerability_given_xy(base)
    print(f"  members={success:.4f}  non-members={imp:.4f}  advantage={success - imp:.4f}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=FIGURES, help="accepted for a uniform interface; no figures")
    args = parser.parse_args()
    seed: int = args.seed
    print_table(np.random.default_rng(seed))


if __name__ == "__main__":
    main()
