"""Check the worked solutions to the exercises of the property-inference chapter.

    uv run python book/part02-attacks/code/solutions_property_inference.py

Prints every number quoted in solutions/04-property-inference.md (seed 0). Makes no figures;
--out is accepted and ignored.

Conventions as in property_inference.py: hypothesis 0 is t0, hypothesis 1 is t1, advantage is
TPR - FPR and accuracy is (1 + advantage) / 2 with a fair bit.
"""

import argparse
import math
from pathlib import Path

import numpy as np
from property_inference import (
    MLP,
    SHADOWS,
    TARGETS,
    Array,
    binomial_pmf,
    centred_pair,
    fit_lda,
    kappa,
    logistic_models,
    membership_advantage_count,
    mlp_logits_shared,
    normal_cdf,
    normal_quantile,
    permute_units,
    predicted_delta,
    property_advantage_count,
    set_features,
    tv_pmf,
)

FIGURES = Path(__file__).resolve().parent.parent / "figures"

# --- Exercise 1: shifting a unimodal pmf by one ---------------------------------------------


def shift_tv(pmf: Array) -> float:
    """TV(X, X + 1) for X with the given pmf on 0..len - 1."""
    return tv_pmf(np.concatenate([pmf, [0.0]]), np.concatenate([[0.0], pmf]))


# --- Exercise 2: how much data the property attack needs ------------------------------------


def count_accuracy(n: int, t0: float, t1: float) -> float:
    return (1 + property_advantage_count(n, t0, t1)) / 2


def n_needed_normal(accuracy: float, t0: float, t1: float) -> float:
    """Solve Φ(√n |t1 - t0| / (2σ̄)) = accuracy for n, with σ̄² = t̄(1 - t̄)."""
    t_bar = (t0 + t1) / 2
    root_n = 2 * math.sqrt(t_bar * (1 - t_bar)) * normal_quantile(accuracy) / abs(t1 - t0)
    return root_n**2


def n_needed_exact(accuracy: float, t0: float, t1: float) -> int:
    return next(n for n in range(1, 100_000) if count_accuracy(n, t0, t1) >= accuracy)


# --- Exercise 3: effective number of leaked records ------------------------------------------


def records_matching(accuracy: float, t0: float, t1: float) -> int:
    """The fewest directly observed records whose best test reaches `accuracy`."""
    return n_needed_exact(accuracy, t0, t1)


# --- Exercise 5: several released models -----------------------------------------------------


def multi_model_accuracy(rng: np.random.Generator, n: int, t0: float, t1: float, k: int) -> float:
    """Sum the linear meta-classifier's scores over k target models, each on its own dataset."""
    meta = fit_lda(logistic_models(rng, SHADOWS, n, t0), logistic_models(rng, SHADOWS, n, t1))
    scores: list[Array] = []
    for t in (t0, t1):
        g = logistic_models(rng, TARGETS * k, n, t)
        scores.append(((g - meta.mid) @ meta.w).reshape(TARGETS, k).sum(axis=1))
    return float(((scores[0] <= 0).mean() + (scores[1] > 0).mean()) / 2)


# --- Exercise 6: noise sized for one record --------------------------------------------------


def noisy_count_advantages(n: int, t0: float, t1: float, sigma: float) -> tuple[float, float]:
    """Normal approximations for S + N(0, σ²): (property advantage, membership advantage at t̄).

    S has variance ≈ n t̄(1 - t̄) under both hypotheses; adding the target shifts S by 1 with
    probability 1 - t̄ (the mixture factor of the chapter).
    """
    t_bar = (t0 + t1) / 2
    sd = math.sqrt(n * t_bar * (1 - t_bar) + sigma**2)
    prop = 2 * normal_cdf(n * abs(t1 - t0) / (2 * sd)) - 1
    member = (1 - t_bar) * (2 * normal_cdf(1 / (2 * sd)) - 1)
    return prop, member


def sigma_to_hide(n: int, t0: float, t1: float, advantage: float) -> float:
    """The noise σ at which the property advantage on S + N(0, σ²) falls to `advantage`."""
    t_bar = (t0 + t1) / 2
    sd = n * abs(t1 - t0) / (2 * normal_quantile((1 + advantage) / 2))
    return math.sqrt(sd**2 - n * t_bar * (1 - t_bar))


def print_table(rng: np.random.Generator) -> None:
    print("== Exercise 1 ==")
    small = binomial_pmf(4, 0.5)
    print("Bin(4, 1/2) pmf x 16: " + " ".join(f"{p * 16:.0f}" for p in small))
    print(f"TV(X, X+1)={shift_tv(small):.4f}  max pmf={small.max():.4f}")
    big = binomial_pmf(99, 0.5)
    print(
        f"Bin(99, 1/2): TV(X, X+1)={shift_tv(big):.4f}  max pmf={big.max():.4f}  "
        f"half of it={0.5 * big.max():.4f}  membership game={membership_advantage_count(100, 0.5):.4f}"
    )
    two_point = np.array([0.5, 0.0, 0.5])
    print(f"X uniform on {{0, 2}}: TV(X, X+1)={shift_tv(two_point):.4f}  max pmf={two_point.max():.4f}")

    print("\n== Exercise 2 ==")
    z = normal_quantile(0.95)
    approx = n_needed_normal(0.95, 0.45, 0.55)
    exact = n_needed_exact(0.95, 0.45, 0.55)
    print(
        f"Phi^-1(0.95)={z:.4f}  normal approximation n={approx:.1f}  exact smallest n={exact}  "
        f"exact accuracy there={count_accuracy(exact, 0.45, 0.55):.4f}  "
        f"at n={exact - 1}: {count_accuracy(exact - 1, 0.45, 0.55):.4f}"
    )

    print("\n== Exercise 3 ==")
    k = kappa()
    for n in (200, 1000):
        t0, t1 = centred_pair(0.1)
        acc = normal_cdf(predicted_delta(n, t0, t1) / 2)
        print(
            f"n={n}, |t1 - t0|=0.1: model accuracy (Gaussian prediction)={acc:.4f}  "
            f"records matching it={records_matching(acc, t0, t1)}  n (kappa/2)^2={n * (k / 2) ** 2:.1f}"
        )

    print("\n== Exercise 4 ==")
    for h in (4, 8, 100):
        count = math.factorial(h) * 2**h
        print(f"h={h}: h! 2^h = {count:.4e}" if h == 100 else f"h={h}: h! 2^h = {count}")
    net = MLP(
        rng.standard_normal((2, 4, 2)), rng.standard_normal((2, 4)), rng.standard_normal((2, 4)), rng.standard_normal(2)
    )
    perm = np.array([2, 0, 3, 1])
    signs = np.array([1.0, -1.0, -1.0, 1.0])
    moved = permute_units(net, perm, signs)
    x = rng.standard_normal((500, 2))
    same_f = float(np.abs(mlp_logits_shared(net, x) - mlp_logits_shared(moved, x)).max()) < 1e-12
    same_set = float(np.abs(set_features(net) - set_features(moved)).max()) < 1e-12
    raw_moved = float(np.abs(net.w - moved.w).max()) > 0.1
    print(f"same function: {same_f}  same second-moment features: {same_set}  raw parameters differ: {raw_moved}")

    print("\n== Exercise 5 ==")
    t0, t1 = centred_pair(0.2)
    delta = predicted_delta(200, t0, t1)
    for k_models in (1, 4):
        print(
            f"k={k_models}: Delta={math.sqrt(k_models) * delta:.3f}  "
            f"predicted accuracy={normal_cdf(math.sqrt(k_models) * delta / 2):.3f}  "
            f"simulated={multi_model_accuracy(rng, 200, t0, t1, k_models):.3f}"
        )
    print(f"one model on 800 records: Delta={predicted_delta(800, t0, t1):.3f}")

    print("\n== Exercise 6 ==")
    n, t0, t1 = 10_000, 0.45, 0.55
    for sigma in (0.0, 10.0):
        prop, member = noisy_count_advantages(n, t0, t1, sigma)
        print(f"sigma={sigma:g}: property advantage={prop:.4f}  membership advantage={member:.4f}")
    hide = sigma_to_hide(n, t0, t1, 0.1)
    prop, member = noisy_count_advantages(n, t0, t1, hide)
    print(
        f"Phi^-1(0.55)={normal_quantile(0.55):.4f}  sigma for property advantage 0.1: {hide:.0f}  "
        f"(property {prop:.4f}, membership {member:.6f})"
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
