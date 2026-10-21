"""Membership inference on your own classifier: the free check, then LiRA with shadow models.

    uv run python code/part02-attacks/practitioner_lira.py
    uv run python code/part02-attacks/practitioner_lira.py --models 16    # fewer shadow models, faster

The workflow of the practitioner edition's risk-assessment chapter (book-practitioner/part02-attacks/
10-risk-assessment.md, steps 2 and 3), and the `lira-scores` excerpt of 02-membership-inference.md.

Data: the handwritten digits bundled with scikit-learn, `sklearn.datasets.load_digits` (1,797 images of 8 x 8
pixels, 10 classes; the test set of the UCI "Optical Recognition of Handwritten Digits" dataset, Alpaydin & Kaynak,
1998; licence CC BY 4.0, shipped with scikit-learn under its BSD licence). No download.

Model: StandardScaler then LogisticRegression(C=10, max_iter=2000). K = 64 models (16 with --models 16) are trained
on Bernoulli-1/2 subsets of the 1,797 images: each image is in each model's training set with probability 1/2,
independently. Each model in turn is the target, and the other K - 1 are its shadow models. So every
(target model, image) pair is one trial of the membership game, about 57,000 member and 57,000 non-member pairs.

It prints, with seed 0, every number the chapter quotes:
- the free check: train and test accuracy of the first target and the average gap over all K targets, and the
  loss attack (one threshold on the confidence, no shadow models);
- AUC and TPR at FPR 1e-3 and 1e-2 for the loss attack, offline LiRA, online LiRA and online LiRA with a global
  variance, read from sklearn.metrics.roc_curve as the best TPR among thresholds whose FPR is at most the target;
- at each target FPR, the true and false positive counts behind each TPR, with exact (Clopper-Pearson) 95%
  intervals from scipy.stats.binomtest, the point estimate ln(TPR/FPR) of epsilon (not a certified
  bound: the pairs share models, so they are not independent trials), and whether the
  TPR's lower bound exceeds 10 x the target FPR (the blocking threshold of step 1);
- the ten most exposed images: those whose IN and OUT confidences are furthest apart, with how often online LiRA
  flags them when they are members.

The thresholds are chosen on the same trials that measure the rates, as in the literature; the rates at FPR 1e-3
rest on about 57 false positives. No figure: the practitioner scripts draw none.
"""

import os

# The models are small; multi-threaded BLAS only adds overhead and run-to-run noise in the last digits.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import argparse
import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.stats import binomtest, norm
from sklearn.datasets import load_digits
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import auc, roc_curve
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

Array = NDArray[np.float64]
IntArray = NDArray[np.int64]
Mask = NDArray[np.bool_]

N_MODELS = 64
TARGET_FPRS = (1e-3, 1e-2)
BLOCK_FACTOR = 10  # step 1: block when the TPR's lower bound exceeds 10 x the FPR
N_EXPOSED = 10


# ---------------------------------------------------------------------------
# Training the models
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Models:
    member: Mask  # (K, n): image i is in model k's training set
    phi: Array  # (K, n): logit of model k's confidence on image i's true label
    confidence: Array  # (K, n): model k's probability of image i's true label
    train_acc: Array  # (K,)
    test_acc: Array  # (K,): accuracy on the images model k did not train on


def train_models(x: Array, y: IntArray, n_models: int, rng: np.random.Generator) -> Models:
    n = len(y)
    phi = np.empty((n_models, n))
    confidence = np.empty((n_models, n))
    train_acc = np.empty(n_models)
    test_acc = np.empty(n_models)
    # excerpt: lira-shadows
    # Each image is in each model's training set with probability 1/2.
    member = rng.random((n_models, n)) < 0.5
    for k in range(n_models):
        model = make_pipeline(StandardScaler(), LogisticRegression(C=10, max_iter=2000))
        model.fit(x[member[k]], y[member[k]])
        # The attacker sees the probabilities the model returns, here for every image.
        p = model.predict_proba(x)[np.arange(n), y]  # confidence on the true label
        p = np.clip(p, 1e-15, 1 - 1e-15)  # keep the logit finite
        phi[k] = np.log(p) - np.log(1 - p)  # its logit, close to Gaussian over retrainings
        confidence[k] = p
        # end excerpt
        train_acc[k] = model.score(x[member[k]], y[member[k]])
        test_acc[k] = model.score(x[~member[k]], y[~member[k]])
    return Models(member, phi, confidence, train_acc, test_acc)


# ---------------------------------------------------------------------------
# The attacks: each gives one score per (target model, image); higher means "member"
# ---------------------------------------------------------------------------


def in_out_stats(phi: Array, member: Mask) -> tuple[Array, Array, Array, Array]:
    """Per-image mean and standard deviation of φ over the models that trained on it (IN) and the others (OUT)."""
    phi_in = np.where(member, phi, np.nan)
    phi_out = np.where(~member, phi, np.nan)
    mu_in, mu_out = np.nanmean(phi_in, axis=0), np.nanmean(phi_out, axis=0)
    sd_in, sd_out = np.nanstd(phi_in, axis=0, ddof=1), np.nanstd(phi_out, axis=0, ddof=1)
    return mu_in, mu_out, sd_in, sd_out


def lira_target(phi: Array, shadow_phi: Array, shadow_member: Mask, global_variance: bool) -> tuple[Array, Array]:
    """Online and offline LiRA scores of every image on one target model, from its shadow models."""
    # excerpt: lira-scores
    # Fit a Gaussian to each image's φ over the shadows that trained on it (IN) and those that did not (OUT).
    phi_in = np.where(shadow_member, shadow_phi, np.nan)  # NaN where the shadow did not train on the image
    phi_out = np.where(~shadow_member, shadow_phi, np.nan)
    mu_in, sd_in = np.nanmean(phi_in, axis=0), np.nanstd(phi_in, axis=0, ddof=1)
    mu_out, sd_out = np.nanmean(phi_out, axis=0), np.nanstd(phi_out, axis=0, ddof=1)
    if global_variance:  # one spread shared by all images: steadier with few shadow models
        sd_in = np.full_like(sd_in, np.sqrt(np.nanmean(sd_in**2)))
        sd_out = np.full_like(sd_out, np.sqrt(np.nanmean(sd_out**2)))
    # Online: how much more likely the target's φ is under the image's IN fit than under its OUT fit.
    online = norm.logpdf(phi, loc=mu_in, scale=sd_in) - norm.logpdf(phi, loc=mu_out, scale=sd_out)
    # Offline (OUT shadows only): how many OUT standard deviations φ lies above the OUT mean.
    offline = (phi - mu_out) / sd_out
    # end excerpt
    return online, offline


def lira_scores(models: Models, global_variance: bool = False) -> tuple[Array, Array]:
    """Online and offline LiRA scores, (K, n): model k is the target, the other K - 1 its shadow models."""
    n_models = len(models.phi)
    online = np.empty_like(models.phi)
    offline = np.empty_like(models.phi)
    for k in range(n_models):
        shadows = np.delete(np.arange(n_models), k)
        online[k], offline[k] = lira_target(models.phi[k], models.phi[shadows], models.member[shadows], global_variance)
    return online, offline


# ---------------------------------------------------------------------------
# Reading an attack: TPR at a low FPR, with exact intervals
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Reading:
    target_fpr: float
    tpr: float
    fpr: float
    true_pos: int
    n_members: int
    false_pos: int
    n_nonmembers: int
    tpr_ci: tuple[float, float]
    fpr_ci: tuple[float, float]
    threshold: float


def clopper_pearson(k: int, n: int) -> tuple[float, float]:
    ci = binomtest(k, n).proportion_ci(confidence_level=0.95, method="exact")
    return ci.low, ci.high


def read_at_fpr(scores: Array, member: Mask, target_fpr: float) -> Reading:
    # excerpt: lira-report
    # Every (target model, image) pair is one trial; members are the positives.
    fpr, tpr, thresholds = roc_curve(member.ravel(), scores.ravel())
    # The best TPR among thresholds whose FPR does not exceed the target.
    i = int(np.flatnonzero(fpr <= target_fpr)[-1])
    n_members, n_nonmembers = int(member.sum()), int((~member).sum())
    true_pos = round(tpr[i] * n_members)
    false_pos = round(fpr[i] * n_nonmembers)
    # Exact 95% intervals (Clopper-Pearson) for both rates.
    tpr_ci = clopper_pearson(true_pos, n_members)
    fpr_ci = clopper_pearson(false_pos, n_nonmembers)
    # end excerpt
    return Reading(
        target_fpr,
        float(tpr[i]),
        float(fpr[i]),
        true_pos,
        n_members,
        false_pos,
        n_nonmembers,
        tpr_ci,
        fpr_ci,
        float(thresholds[i]),
    )


def roc_auc(scores: Array, member: Mask) -> float:
    fpr, tpr, _ = roc_curve(member.ravel(), scores.ravel())
    return auc(fpr, tpr)


def epsilon_point(r: Reading) -> float:
    """ln(TPR/FPR): a point estimate of ε, not a certified bound, since the pairs share models and are not independent."""
    return math.log(r.tpr / r.fpr) if r.fpr > 0 and r.tpr > 0 else math.nan


# ---------------------------------------------------------------------------
# Who is exposed
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Exposed:
    index: int
    label: int
    separation: float  # (μ_in - μ_out) / σ over all K models
    conf_in: float  # mean confidence on the true label over models that trained on it
    conf_out: float  # ... and over models that did not
    flagged: int  # target models where it was a member and online LiRA said "member" at FPR 1e-3
    member_of: int  # target models where it was a member


def most_exposed(models: Models, y: IntArray, online: Array, threshold: float, top: int) -> list[Exposed]:
    mu_in, mu_out, sd_in, sd_out = in_out_stats(models.phi, models.member)
    sep = (mu_in - mu_out) / np.sqrt((sd_in**2 + sd_out**2) / 2)
    conf_in = np.nanmean(np.where(models.member, models.confidence, np.nan), axis=0)
    conf_out = np.nanmean(np.where(~models.member, models.confidence, np.nan), axis=0)
    flagged = ((online >= threshold) & models.member).sum(axis=0)
    order = np.argsort(-sep, kind="stable")[:top]
    return [
        Exposed(
            int(i),
            int(y[i]),
            float(sep[i]),
            float(conf_in[i]),
            float(conf_out[i]),
            int(flagged[i]),
            int(models.member[:, i].sum()),
        )
        for i in order
    ]


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


def report(n_models: int, seed: int) -> None:
    x, y = load_digits(return_X_y=True)
    rng = np.random.default_rng(seed)
    models = train_models(x, y, n_models, rng)
    member = models.member
    print(f"[Data] digits: {len(y):,} images, 64 pixels, 10 classes; {n_models} models on Bernoulli-1/2 subsets")
    print(f"  member pairs={int(member.sum()):,}  non-member pairs={int((~member).sum()):,}")

    # excerpt: gap-and-loss
    # The free check: how much better the target does on its training images than on the others.
    print(f"target 0: train accuracy={models.train_acc[0]:.4f}  test accuracy={models.test_acc[0]:.4f}")
    gap = models.train_acc - models.test_acc
    print(f"train-test gap, mean over {n_models} targets={gap.mean():.4f}")
    # The loss attack: say "member" when the loss on the true label is low, for every image alike.
    loss = np.log(models.confidence)  # minus the cross-entropy loss
    for target_fpr in (1e-3, 1e-2):
        r = read_at_fpr(loss, member, target_fpr)
        print(f"loss attack at FPR {target_fpr:g}: TPR={r.tpr:.4f}")
    # end excerpt

    online, offline = lira_scores(models)
    online_global, _ = lira_scores(models, global_variance=True)
    # With few shadow models an image can lack two IN or two OUT shadows, and its fit is undefined: drop the pair.
    keep = np.isfinite(online) & np.isfinite(offline) & np.isfinite(online_global)
    attacks = {
        "loss attack": loss[keep],
        "LiRA offline": offline[keep],
        "LiRA online": online[keep],
        "LiRA online, global variance": online_global[keep],
    }
    member = member[keep]
    print(f"\n[Attacks on {n_models} target models, each with {n_models - 1} shadow models]")
    print(f"  pairs without a fit, dropped: {int((~keep).sum())}")
    for name, scores in attacks.items():
        tprs = "  ".join(f"TPR@{a:g}={read_at_fpr(scores, member, a).tpr:.4f}" for a in TARGET_FPRS)
        print(f"  {name}: AUC={roc_auc(scores, member):.3f}  {tprs}")

    for a in TARGET_FPRS:
        rule = f"; step 1 blocks if the TPR's lower end > {BLOCK_FACTOR * a:g}" if a == TARGET_FPRS[0] else ""
        print(
            f"\n[At FPR {a:g}: counts, exact 95% intervals, ln(TPR/FPR) as a point estimate, not a certified bound{rule}]"
        )
        for name, scores in attacks.items():
            r = read_at_fpr(scores, member, a)
            verdict = ""
            if a == TARGET_FPRS[0]:
                verdict = "  block" if r.tpr_ci[0] > BLOCK_FACTOR * a else "  pass"
            print(
                f"  {name}: TP={r.true_pos}/{r.n_members}  TPR={r.tpr:.4f} [{r.tpr_ci[0]:.4f}, {r.tpr_ci[1]:.4f}]  "
                f"FP={r.false_pos}/{r.n_nonmembers}  FPR={r.fpr:.5f} [{r.fpr_ci[0]:.5f}, {r.fpr_ci[1]:.5f}]  "
                f"ln(TPR/FPR)={epsilon_point(r):.2f}{verdict}"
            )

    threshold = read_at_fpr(attacks["LiRA online"], member, TARGET_FPRS[0]).threshold
    print(f"\n[The {N_EXPOSED} most exposed images: largest (mu_in - mu_out)/sigma over all {n_models} models]")
    for e in most_exposed(models, y, online, threshold, N_EXPOSED):
        print(
            f"  image {e.index}: label {e.label}  separation={e.separation:.2f}  "
            f"confidence IN={e.conf_in:.3f} OUT={e.conf_out:.3f}  "
            f"flagged by online LiRA at FPR 0.001 in {e.flagged} of {e.member_of} models"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--models", type=int, default=N_MODELS, help="number of models (targets and shadows)")
    args = parser.parse_args()
    seed: int = args.seed
    n_models: int = args.models
    report(n_models, seed)


if __name__ == "__main__":
    main()
