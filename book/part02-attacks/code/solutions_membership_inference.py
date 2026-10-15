"""Check the worked solutions to the exercises of the membership inference chapter.

    uv run python book/part02-attacks/code/solutions_membership_inference.py

Prints every number quoted in solutions/02-membership-inference.md (seed 0). Makes no figures;
--out is accepted and ignored.

Conventions as in membership_inference.py: φ = y <w, x> is the logit of the model's confidence in
the true label, TPR and FPR are rates over (target model, pool point) pairs.
"""

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")  # set before NumPy loads, as in membership_inference.py
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import argparse
import math
from pathlib import Path

import numpy as np
from membership_inference import (
    ALPHA,
    N_SHADOWS,
    N_TARGETS,
    TWO_TYPES,
    Array,
    Experiment,
    Population,
    exposed_few_rates,
    fit_gaussians,
    gap_report,
    gaussian_tpr,
    likelihood_ratio_rates,
    lira_online,
    loss_attack_scores,
    normal_cdf,
    normal_isf,
    roc_curve,
    run_experiment,
    sample_population,
    separation,
    shadow_attack_model,
    tpr_at_fpr,
    train_models,
    weak_all_rates,
)

FIGURES = Path(__file__).resolve().parent.parent / "figures"


# --- Exercise 1: accuracy with an unbalanced membership prior ----------------------------------


def correct_attack_accuracy(prior: float, train_error: float, test_error: float) -> float:
    """Accuracy of 'member iff correctly classified' when a fraction `prior` of queries are members."""
    return prior * (1 - train_error) + (1 - prior) * test_error


# --- Exercise 2: the Gaussian log-likelihood ratio ----------------------------------------------


def gaussian_log_lr(z: float, s: float) -> float:
    """log Λ for N(s, 1) against N(0, 1) at the standardized score z: s z - s²/2."""
    return s * z - s * s / 2


def gaussian_log_lr_direct(z: float, s: float) -> float:
    """The same, from the two log-densities."""
    return -0.5 * (z - s) ** 2 + 0.5 * z * z


# --- Exercise 4: how many non-members to certify a low FPR --------------------------------------


def zero_count_negatives(alpha: float, level: float = 0.95) -> float:
    """n with (1 - alpha)^n = 1 - level: zero false positives in n trials rule out FPR >= alpha."""
    return math.log(1 - level) / math.log1p(-alpha)


# --- Exercise 5: membership in a Gaussian mean ---------------------------------------------------


def mean_estimation_tpr(rng: np.random.Generator, d: int, n: int, trials: int, alpha: float) -> tuple[float, float]:
    """Simulate the test 'T = <z, θ̂> / (||z|| / √n) >= Φ⁻¹(1 - α)' for θ̂ the mean of n points of N(0, I_d).

    Under OUT, T ~ N(0, 1) exactly (θ̂ is independent of z). Under IN, θ̂ = (z + sum of n - 1 others)/n.
    The sum of n - 1 standard normal vectors is √(n - 1) g with g ~ N(0, I_d). Returns (TPR, FPR).
    """
    t = normal_isf(alpha)
    z = rng.standard_normal((trials, d))
    norm_z = np.linalg.norm(z, axis=1)
    g_in = rng.standard_normal((trials, d))
    theta_in = (z + math.sqrt(n - 1) * g_in) / n
    g_out = rng.standard_normal((trials, d))
    theta_out = math.sqrt(n) * g_out / n
    scale = norm_z / math.sqrt(n)
    t_in = np.sum(z * theta_in, axis=1) / scale
    t_out = np.sum(z * theta_out, axis=1) / scale
    return float(np.mean(t_in >= t)), float(np.mean(t_out >= t))


def mean_estimation_tpr_formula(d: int, n: int, alpha: float) -> float:
    """With ||z||² ≈ d, T ~ N(√(d/n), (n - 1)/n) under IN; ignore the (n - 1)/n and use the Gaussian curve."""
    return gaussian_tpr(alpha, math.sqrt(d / n))


# --- Exercise 6: remove the outliers -------------------------------------------------------------


def restrict(pop: Population, keep: np.ndarray[tuple[int], np.dtype[np.bool_]]) -> Population:
    return Population(pop.x[keep], pop.y[keep], pop.outlier[keep])


def inlier_tpr(exp: Experiment, alpha: float) -> float:
    """LiRA online (per-example variance) TPR at FPR alpha, over the inlier pairs only."""
    scores: Array = lira_online(exp.targets.phi, fit_gaussians(exp.shadows))
    keep = ~exp.pop.outlier
    return tpr_at_fpr(roc_curve(scores[:, keep], exp.targets.member[:, keep]), alpha)


def inlier_separation(exp: Experiment) -> float:
    """Mean per-example separation (μ_in - μ_out)/σ over the inliers, from the shadow models."""
    return float(np.mean(separation(fit_gaussians(exp.shadows))[~exp.pop.outlier]))


def onion(rng: np.random.Generator, alpha: float) -> tuple[Experiment, Experiment]:
    """The experiment with the outliers in the pool, and again with them removed and every model retrained."""
    pop = sample_population(rng)
    with_outliers = Experiment(pop, train_models(rng, pop, N_TARGETS), train_models(rng, pop, N_SHADOWS))
    clean = restrict(pop, ~pop.outlier)
    without = Experiment(clean, train_models(rng, clean, N_TARGETS), train_models(rng, clean, N_SHADOWS))
    return with_outliers, without


def print_table(rng: np.random.Generator) -> None:
    exp = run_experiment(rng)  # the same draws as the chapter's table
    g = gap_report(rng, exp.targets, 2.0)

    print("[Exercise 1: accuracy of 'correct means member' with membership prior pi]")
    print(f"  train error={g.train_error:.4f}  test error={g.test_error:.4f}")
    for prior in (0.5, 0.1):
        acc = correct_attack_accuracy(prior, g.train_error, g.test_error)
        print(f"  pi={prior:g}: attack accuracy={acc:.4f}  always 'non-member'={1 - prior:.4f}")

    print("[Exercise 2: log-likelihood ratio of N(s,1) against N(0,1)]")
    for z, s in ((3.0902, 0.5), (3.0902, 4.0), (1.0, 4.0)):
        print(
            f"  z={z:g} s={s:g}: s*z - s^2/2={gaussian_log_lr(z, s):.4f}  from densities={gaussian_log_lr_direct(z, s):.4f}"
        )
    c, fprs, tpr = likelihood_ratio_rates(TWO_TYPES, ALPHA)
    cuts = [(c + (e.mu_in - e.mu_out) ** 2 / 2) / (e.mu_in - e.mu_out) for e in TWO_TYPES]
    print(
        f"  two types: log-threshold c={c:.3f}  z-cut typical={cuts[0]:.3f}  z-cut atypical={cuts[1]:.3f}  "
        f"FPR typical={fprs[0]:.2e}  FPR atypical={fprs[1]:.4f}  pooled FPR="
        f"{sum(e.weight * f for e, f in zip(TWO_TYPES, fprs, strict=True)):.4f}  TPR={tpr:.4f}"
    )

    print("[Exercise 3: same AUC, different exposure]")
    for q in (0.05, 0.01):
        a_auc, a_adv, a_tpr = exposed_few_rates(q, ALPHA)
        mu, b_adv, b_tpr = weak_all_rates(a_auc, ALPHA)
        print(
            f"  q={q:g}: AUC={a_auc:.4f}  A: advantage={a_adv:.4f} TPR={a_tpr:.5f}  "
            f"B: mu={mu:.4f} AUC check={normal_cdf(mu / math.sqrt(2)):.4f} advantage={b_adv:.4f} TPR={b_tpr:.5f}"
        )

    print("[Exercise 4: non-members needed]")
    for a in (1e-3, 1e-4, 1e-5):
        n = zero_count_negatives(a)
        print(f"  alpha={a:g}: zero false positives needs n >= {n:.1f}, so {math.ceil(n)}")
    n_neg = int((~exp.targets.member).sum())
    print(f"  non-member pairs in the experiment={n_neg}  expected false positives at 1e-3={n_neg * 1e-3:.0f}")

    print("[Exercise 5: membership in a Gaussian mean, alpha=1e-3]")
    for d, n in ((1000, 100), (100, 100), (250, 25)):
        tpr_sim, fpr_sim = mean_estimation_tpr(rng, d, n, 20_000, ALPHA)
        print(
            f"  d={d} n={n}: separation={math.sqrt(d / n):.4f}  formula TPR={mean_estimation_tpr_formula(d, n, ALPHA):.4f}  "
            f"simulated TPR={tpr_sim:.4f}  FPR={fpr_sim:.4f}"
        )

    print("[Exercise 6: remove the outliers and retrain]")
    before, after = onion(rng, ALPHA)
    print(
        f"  pool after removal={len(after.pop.y)}  inlier member pairs={int(before.targets.member[:, ~before.pop.outlier].sum())}"
        f" and {int(after.targets.member.sum())}"
    )
    print(
        f"  inlier TPR at FPR {ALPHA:g}: with outliers={inlier_tpr(before, ALPHA):.4f}  without={inlier_tpr(after, ALPHA):.4f}"
    )
    print(
        f"  inlier mean separation: with outliers={inlier_separation(before):.3f}  without={inlier_separation(after):.3f}"
    )

    print("[Exercise 7: the shadow attack model is a per-class threshold]")
    neg, pos = shadow_attack_model(exp.shadows, exp.pop.y)
    print(
        f"  class -1: a={float(neg[0]):.3f} b={float(neg[1]):.3f}  class +1: a={float(pos[0]):.3f} b={float(pos[1]):.3f}"
    )
    loss_auc = roc_curve(loss_attack_scores(exp.targets.phi), exp.targets.member)
    for c_label, coef in (("-1", neg), ("+1", pos)):
        phi_half = -float(coef[0]) / float(coef[1])
        print(f"  class {c_label}: attack says 'member' with probability > 1/2 when phi > {phi_half:.3f}")
    print(f"  loss attack TPR at FPR 1%={tpr_at_fpr(loss_auc, 1e-2):.4f}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=FIGURES, help="accepted for a uniform interface; no figures")
    args = parser.parse_args()
    seed: int = args.seed
    print_table(np.random.default_rng(seed))


if __name__ == "__main__":
    main()
