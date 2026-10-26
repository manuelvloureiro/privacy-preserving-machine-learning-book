"""Run the membership inference attacks of the membership inference chapter and draw its figures.

    uv run python code/part02-attacks/membership_inference.py
    uv run python code/part02-attacks/membership_inference.py --seed 1 --out /tmp/figures

Prints the numbers quoted in 02-membership-inference.md (they match exactly with the default
seed), then saves the chapter's figures.

The running example: a pool of 1,000 labelled points in R^100, 5% of them outliers (large norm,
random label). Each model is L2-regularized logistic regression trained on a random half of the
pool. Target models are attacked; shadow models, trained the same way, are what the attacker
uses to calibrate. The attack statistic is the logit of the model's confidence in the true label,
which for logistic regression is the margin phi = y <w, x>.
"""

import os

# The models are tiny; multi-threaded BLAS only adds overhead (and is 40x slower on some machines).
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import argparse
import math
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # for code/shared
from shared.figures import configure
from shared.stats import normal_cdf, normal_sf, sigmoid

configure("membership_inference")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from matplotlib.ticker import NullLocator
from numpy.typing import NDArray

# Figures are build output: _build/figures/<part>/, which the build copies into each book.
FIGURES = Path(__file__).resolve().parents[2] / "_build" / "figures" / Path(__file__).parent.name

Array = NDArray[np.float64]
Mask = NDArray[np.bool_]

D = 100  # feature dimension
N_POOL = 1000  # each model trains on a random half, n = 500
OUTLIER_FRAC = 0.05  # share of the pool that is atypical
OUTLIER_SCALE = 5.0  # outliers have 5 times the typical feature norm and a random label
SIGNAL = 3.0  # P(y = 1 | x) = sigmoid(SIGNAL * sqrt(D) * x_1) for typical points
LAM = 1e-3  # L2 regularization of the trained models
N_TARGETS = 128  # models that are attacked
N_SHADOWS = 64  # models the attacker trains
ALPHA = 1e-3  # the false positive rate at which attacks are compared


# --- Normal distribution (for the Gaussian fits and the closed-form examples) -----------------


def normal_isf(q: float) -> float:
    """Φ⁻¹(1 - q), the x with Pr[Z > x] = q, by bisection on the decreasing tail."""
    if not 0.0 < q < 1.0:
        raise ValueError("q must lie in (0, 1)")
    lo, hi = -40.0, 40.0
    for _ in range(200):
        mid = (lo + hi) / 2
        if mid in (lo, hi):
            break
        if normal_sf(mid) > q:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def bisect(f: Callable[[float], float], lo: float, hi: float, steps: int = 200) -> float:
    """Root of an increasing function f on [lo, hi]."""
    for _ in range(steps):
        mid = (lo + hi) / 2
        if f(mid) < 0:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


# --- The population and the models ------------------------------------------------------------


@dataclass(frozen=True)
class Population:
    x: Array  # (N, D) features
    y: Array  # (N,) labels ±1
    outlier: Mask  # (N,) True for the atypical points


def sample_population(
    rng: np.random.Generator,
    n: int = N_POOL,
    d: int = D,
    frac: float = OUTLIER_FRAC,
    scale: float = OUTLIER_SCALE,
    signal: float = SIGNAL,
) -> Population:
    """Typical points: x ~ N(0, I/d), y = ±1 with P(y = 1 | x) = sigmoid(signal √d x_1).

    Outliers: the same x scaled by `scale` and a label drawn by a fair coin, so the label carries
    no information the rest of the data could predict: the model can only get it right by fitting it.
    """
    x = rng.standard_normal((n, d)) / math.sqrt(d)
    outlier = np.zeros(n, dtype=np.bool_)
    outlier[: round(frac * n)] = True
    x[outlier] *= scale
    p = 0.5 * (1 + np.tanh(0.5 * signal * math.sqrt(d) * x[:, 0]))
    y = np.where(rng.random(n) < p, 1.0, -1.0)
    y[outlier] = np.where(rng.random(int(outlier.sum())) < 0.5, 1.0, -1.0)
    return Population(x, y, outlier)


def logistic_loss(phi: Array) -> Array:
    """Logistic loss ln(1 + e^{-φ}) as a function of the margin φ = y <w, x>."""
    return np.logaddexp(0.0, -phi)


def regularized_objective(w: Array, x: Array, y: Array, lam: float) -> float:
    return float(np.mean(logistic_loss(y * (x @ w)))) + 0.5 * lam * float(w @ w)


def fit_logistic(x: Array, y: Array, lam: float, tol: float = 1e-10) -> Array:
    """argmin_w (1/n) Σ ln(1 + exp(-y_i <w, x_i>)) + (λ/2)||w||², by damped Newton."""
    n, d = x.shape
    w: Array = np.zeros(d)
    eye: Array = np.eye(d)
    for _ in range(100):
        s = sigmoid(-y * (x @ w))
        grad: Array = -(x.T @ (y * s)) / n + lam * w
        if float(np.max(np.abs(grad))) < tol:
            break
        hess: Array = (x.T * (s * (1 - s))) @ x / n + lam * eye
        step: Array = np.linalg.solve(hess, grad)
        decrease, t = float(grad @ step), 1.0
        if decrease > 1e-12:
            f0 = regularized_objective(w, x, y, lam)
            while regularized_objective(w - t * step, x, y, lam) > f0 - 0.25 * t * decrease and t > 1e-10:
                t /= 2
        w = w - t * step
    return w


@dataclass(frozen=True)
class Models:
    phi: Array  # (k, N) margin of every pool point under every model: the logit of its confidence
    member: Mask  # (k, N) True where the point was in that model's training set


def train_models(rng: np.random.Generator, pop: Population, k: int, lam: float = LAM) -> Models:
    """k models in complementary pairs: one trains on a random half of the pool, the next on the
    other half. So every point is a member of exactly half the models."""
    n = len(pop.y)
    phi = np.zeros((k, n))
    member = np.zeros((k, n), dtype=np.bool_)
    for j in range(k):
        if j % 2 == 0:
            member[j, rng.permutation(n)[: n // 2]] = True
        else:
            member[j] = ~member[j - 1]
        w = fit_logistic(pop.x[member[j]], pop.y[member[j]], lam)
        phi[j] = pop.y * (pop.x @ w)
    return Models(phi, member)


@dataclass(frozen=True)
class Experiment:
    pop: Population
    targets: Models
    shadows: Models


def run_experiment(
    rng: np.random.Generator, lam: float = LAM, n_targets: int = N_TARGETS, n_shadows: int = N_SHADOWS
) -> Experiment:
    pop = sample_population(rng)
    return Experiment(pop, train_models(rng, pop, n_targets, lam), train_models(rng, pop, n_shadows, lam))


# --- Evaluating an attack ---------------------------------------------------------------------


@dataclass(frozen=True)
class Roc:
    fpr: Array
    tpr: Array
    n_pos: int
    n_neg: int


def roc_curve(scores: Array, labels: Mask) -> Roc:
    """ROC of the tests 'say member when score >= t' over every distinct t (ties kept together)."""
    s = scores.ravel()
    lab = labels.ravel()
    order = np.argsort(-s, kind="stable")
    s, lab = s[order], lab[order]
    tp = np.cumsum(lab)
    fp = np.cumsum(~lab)
    last = np.r_[s[1:] != s[:-1], True]  # keep the last index of each group of tied scores
    n_pos, n_neg = int(lab.sum()), int((~lab).sum())
    fpr = np.r_[0.0, fp[last] / n_neg]
    tpr = np.r_[0.0, tp[last] / n_pos]
    return Roc(fpr, tpr, n_pos, n_neg)


def tpr_at_fpr(roc: Roc, alpha: float) -> float:
    """Best TPR among thresholds whose FPR is at most alpha."""
    return float(np.max(roc.tpr[roc.fpr <= alpha + 1e-15]))


def auc(roc: Roc) -> float:
    return float(np.sum(np.diff(roc.fpr) * (roc.tpr[1:] + roc.tpr[:-1]) / 2))


def best_advantage(roc: Roc) -> float:
    """max_t TPR - FPR: the best membership advantage of a threshold on this score."""
    return float(np.max(roc.tpr - roc.fpr))


def rates_at_threshold(scores: Array, labels: Mask, t: float) -> tuple[float, float]:
    """(TPR, FPR) of 'say member when score >= t'."""
    said = scores >= t
    return float(np.mean(said[labels])), float(np.mean(said[~labels]))


def threshold_for_fpr(scores: Array, labels: Mask, alpha: float) -> float:
    """The smallest threshold whose FPR on these (scores, labels) is at most alpha."""
    neg = np.sort(scores[~labels])[::-1]
    k = math.floor(alpha * len(neg))  # at most k negatives may reach the threshold
    return float(np.nextafter(neg[k], np.inf))


def binomial_cdf(k: int, n: int, p: float) -> float:
    """Pr[Bin(n, p) <= k], for 0 < p < 1, summing the k + 1 terms in log space."""
    if k < 0:
        return 0.0
    if k >= n:
        return 1.0
    j = np.arange(k + 1, dtype=np.float64)
    log_binom = np.array([math.lgamma(n + 1) - math.lgamma(i + 1) - math.lgamma(n - i + 1) for i in range(k + 1)])
    log_terms = log_binom + j * math.log(p) + (n - j) * math.log1p(-p)
    return float(np.sum(np.exp(log_terms)))


def clopper_pearson(k: int, n: int, level: float = 0.95) -> tuple[float, float]:
    """Exact two-sided interval for a binomial proportion (see the hypothesis testing chapter)."""
    a = (1 - level) / 2
    lower = 0.0 if k == 0 else bisect(lambda p: (1 - binomial_cdf(k - 1, n, p)) - a, 1e-12, 1 - 1e-12, 60)
    upper = 1.0 if k == n else bisect(lambda p: a - binomial_cdf(k, n, p), 1e-12, 1 - 1e-12, 60)
    return lower, upper


# --- The loss attack and the generalization gap -----------------------------------------------


def loss_attack_scores(phi: Array) -> Array:
    """Yeom et al.: a low loss means 'member'. The score is minus the loss."""
    return -logistic_loss(phi)


@dataclass(frozen=True)
class GapReport:
    train_error: float
    test_error: float
    correct_attack_tpr: float  # Pr[correctly classified | member]
    correct_attack_fpr: float  # Pr[correctly classified | non-member]
    train_loss: float
    test_loss: float
    clipped_train_loss: float  # E[min(loss, B) | member]
    clipped_test_loss: float
    bounded_attack_advantage: float  # simulated advantage of 'member with probability 1 - min(loss, B)/B'
    yeom_threshold: float  # the mean training loss, Yeom's threshold
    yeom_tpr: float
    yeom_fpr: float


def gap_report(rng: np.random.Generator, models: Models, bound: float) -> GapReport:
    phi, member = models.phi, models.member
    correct = phi > 0
    loss = logistic_loss(phi)
    clipped = np.minimum(loss, bound)
    says_member = rng.random(phi.shape) < 1 - clipped / bound
    tau = float(np.mean(loss[member]))
    yeom_tpr, yeom_fpr = rates_at_threshold(-loss, member, -tau)
    return GapReport(
        train_error=float(np.mean(~correct[member])),
        test_error=float(np.mean(~correct[~member])),
        correct_attack_tpr=float(np.mean(correct[member])),
        correct_attack_fpr=float(np.mean(correct[~member])),
        train_loss=float(np.mean(loss[member])),
        test_loss=float(np.mean(loss[~member])),
        clipped_train_loss=float(np.mean(clipped[member])),
        clipped_test_loss=float(np.mean(clipped[~member])),
        bounded_attack_advantage=float(np.mean(says_member[member]) - np.mean(says_member[~member])),
        yeom_threshold=tau,
        yeom_tpr=yeom_tpr,
        yeom_fpr=yeom_fpr,
    )


# --- Shadow models: a learned attack model ----------------------------------------------------


def shadow_attack_model(shadows: Models, y: Array) -> tuple[Array, Array]:
    """Shokri et al.: one attack model per class, trained on the shadow models' outputs.

    The attack model is a logistic regression of 'member' on (1, φ) for the points of each class.
    Returns the coefficient pairs (a_c, b_c) for c = -1 and c = +1.
    """
    coefs: list[Array] = []
    for c in (-1.0, 1.0):
        cols = y == c
        phi = shadows.phi[:, cols].ravel()
        lab = np.where(shadows.member[:, cols].ravel(), 1.0, -1.0)
        feats = np.column_stack([np.ones_like(phi), phi])
        coefs.append(fit_logistic(feats, lab, 1e-8))
    return coefs[0], coefs[1]


def shadow_attack_scores(phi: Array, y: Array, coefs: tuple[Array, Array]) -> Array:
    """Attack-model probability of 'member' for every (model, point)."""
    neg, pos = coefs
    a = np.where(y > 0, float(pos[0]), float(neg[0]))
    b = np.where(y > 0, float(pos[1]), float(neg[1]))
    return sigmoid(a + b * phi)


# --- LiRA: a per-example likelihood-ratio test ------------------------------------------------


@dataclass(frozen=True)
class GaussianFits:
    mu_in: Array  # (N,) mean of φ over shadow models that trained on the point
    mu_out: Array  # (N,) mean of φ over shadow models that did not
    sd_in: Array
    sd_out: Array


def fit_gaussians(shadows: Models, global_variance: bool = False) -> GaussianFits:
    """Per-example IN and OUT Gaussians for φ, from the shadow models.

    With global_variance, every example gets the same σ_in (and σ_out): the average of the
    per-example variances. That is what Carlini et al. recommend with few shadow models.
    """
    phi, m = shadows.phi, shadows.member
    c_in = m.sum(axis=0)
    c_out = (~m).sum(axis=0)
    mu_in = np.where(m, phi, 0.0).sum(axis=0) / c_in
    mu_out = np.where(~m, phi, 0.0).sum(axis=0) / c_out
    var_in = np.where(m, (phi - mu_in) ** 2, 0.0).sum(axis=0) / np.maximum(c_in - 1, 1)
    var_out = np.where(~m, (phi - mu_out) ** 2, 0.0).sum(axis=0) / np.maximum(c_out - 1, 1)
    if global_variance:
        var_in = np.full_like(var_in, np.mean(var_in))
        var_out = np.full_like(var_out, np.mean(var_out))
    return GaussianFits(mu_in, mu_out, np.sqrt(var_in), np.sqrt(var_out))


def log_normal_pdf(x: Array, mu: Array, sd: Array) -> Array:
    return -0.5 * ((x - mu) / sd) ** 2 - np.log(sd) - 0.5 * math.log(2 * math.pi)


def lira_online(phi: Array, fits: GaussianFits) -> Array:
    """log Λ = ln N(φ; μ_in, σ_in²) - ln N(φ; μ_out, σ_out²), per (model, point)."""
    return log_normal_pdf(phi, fits.mu_in, fits.sd_in) - log_normal_pdf(phi, fits.mu_out, fits.sd_out)


def lira_offline(phi: Array, fits: GaussianFits) -> Array:
    """(φ - μ_out)/σ_out: how far above its OUT distribution the observed φ lies (a z-score)."""
    return (phi - fits.mu_out) / fits.sd_out


def separation(fits: GaussianFits) -> Array:
    """Per-example (μ_in - μ_out)/σ, with σ² the average of the IN and OUT variances."""
    return (fits.mu_in - fits.mu_out) / np.sqrt((fits.sd_in**2 + fits.sd_out**2) / 2)


def subset(models: Models, k: int) -> Models:
    return Models(models.phi[:k], models.member[:k])


def attack_scores(exp: Experiment) -> dict[str, Array]:
    """Every attack's score for every (target model, pool point)."""
    phi, y = exp.targets.phi, exp.pop.y
    per_example = fit_gaussians(exp.shadows)
    shared = fit_gaussians(exp.shadows, global_variance=True)
    return {
        "loss": loss_attack_scores(phi),
        "shadow": shadow_attack_scores(phi, y, shadow_attack_model(exp.shadows, y)),
        "offline": lira_offline(phi, per_example),
        "online": lira_online(phi, per_example),
        "online_global": lira_online(phi, shared),
    }


ATTACK_NAMES = {
    "loss": "loss threshold",
    "shadow": "shadow attack model",
    "offline": "LiRA offline",
    "online": "LiRA online",
    "online_global": "LiRA online, global variance",
}


# --- Closed-form examples -----------------------------------------------------------------------


def gaussian_tpr(alpha: float, sep: float) -> float:
    """TPR at FPR alpha of the best test between N(0, 1) and N(sep, 1): 1 - Φ(Φ⁻¹(1 - α) - sep)."""
    return normal_sf(normal_isf(alpha) - sep)


@dataclass(frozen=True)
class ExampleType:
    weight: float  # share of the pool
    mu_out: float  # mean of φ when the point is not in the training set
    mu_in: float  # mean of φ when it is (both with standard deviation 1)


TWO_TYPES = (ExampleType(0.95, 3.0, 3.5), ExampleType(0.05, -2.0, 2.0))


def global_threshold_rates(types: tuple[ExampleType, ...], alpha: float) -> tuple[float, float]:
    """(threshold, TPR) of 'say member when φ >= t' with t set so the pooled FPR is alpha."""

    def fpr_minus(t: float) -> float:
        return alpha - sum(e.weight * normal_sf(t - e.mu_out) for e in types)

    t = bisect(fpr_minus, -20, 20)
    return t, sum(e.weight * normal_sf(t - e.mu_in) for e in types)


def per_example_z_tpr(types: tuple[ExampleType, ...], alpha: float) -> float:
    """TPR of 'say member when (φ - μ_out) >= Φ⁻¹(1 - α)': the same FPR alpha on every example."""
    z = normal_isf(alpha)
    return sum(e.weight * normal_sf(z - (e.mu_in - e.mu_out)) for e in types)


def likelihood_ratio_rates(types: tuple[ExampleType, ...], alpha: float) -> tuple[float, list[float], float]:
    """Neyman–Pearson for the pooled game: one threshold c on log Λ for every example.

    With unit variances, log Λ = s (φ - μ_out) - s²/2 with s = μ_in - μ_out, so log Λ >= c
    means φ - μ_out >= (c + s²/2)/s. Returns (c, per-type FPR, pooled TPR).
    """

    def cut(e: ExampleType, c: float) -> float:
        s = e.mu_in - e.mu_out
        return (c + s * s / 2) / s

    def fpr_minus(c: float) -> float:
        return alpha - sum(e.weight * normal_sf(cut(e, c)) for e in types)

    c = bisect(fpr_minus, -50, 50)
    fprs = [normal_sf(cut(e, c)) for e in types]
    tpr = sum(e.weight * normal_sf(cut(e, c) - (e.mu_in - e.mu_out)) for e in types)
    return c, fprs, tpr


def exposed_few_rates(q: float, alpha: float) -> tuple[float, float, float]:
    """Attack A: a fraction q of members is recognized with certainty, the rest scores like a
    non-member. Returns (AUC, best advantage, TPR at FPR alpha)."""
    return 0.5 + q / 2, q, q + (1 - q) * alpha


def weak_all_rates(auc_target: float, alpha: float) -> tuple[float, float, float]:
    """Attack B: members score N(μ, 1), non-members N(0, 1), with μ set so the AUC is auc_target.

    AUC = Φ(μ/√2), so μ = √2 Φ⁻¹(AUC). Returns (μ, best advantage = 2Φ(μ/2) - 1, TPR at alpha).
    """
    mu = math.sqrt(2) * -normal_isf(auc_target)
    return mu, 2 * normal_cdf(mu / 2) - 1, gaussian_tpr(alpha, mu)


# --- Outliers -----------------------------------------------------------------------------------


@dataclass(frozen=True)
class OutlierReport:
    sep_inlier: float
    sep_outlier: float
    mu_out_inlier: float
    mu_out_outlier: float
    loss_influence_inlier: float  # mean over points of E[loss | out] - E[loss | in]
    loss_influence_outlier: float
    tpr_inlier: float  # online LiRA at the pooled FPR-alpha threshold, among members of each group
    tpr_outlier: float
    fpr_inlier: float
    fpr_outlier: float
    share_of_hits: float  # fraction of the true positives that are outliers
    predicted_tpr_inlier: float  # mean of gaussian_tpr(alpha, sep_i) over each group
    predicted_tpr_outlier: float


def outlier_report(exp: Experiment, scores: Array, alpha: float) -> OutlierReport:
    fits = fit_gaussians(exp.shadows)
    sep = separation(fits)
    out = exp.pop.outlier
    loss, m = logistic_loss(exp.shadows.phi), exp.shadows.member
    influence = (np.where(~m, loss, 0.0).sum(0) / (~m).sum(0)) - (np.where(m, loss, 0.0).sum(0) / m.sum(0))
    labels = exp.targets.member
    t = threshold_for_fpr(scores, labels, alpha)
    hit = scores >= t
    is_out = np.broadcast_to(out, labels.shape)
    predicted = np.array([gaussian_tpr(alpha, float(s)) for s in sep])
    return OutlierReport(
        sep_inlier=float(np.mean(sep[~out])),
        sep_outlier=float(np.mean(sep[out])),
        mu_out_inlier=float(np.mean(fits.mu_out[~out])),
        mu_out_outlier=float(np.mean(fits.mu_out[out])),
        loss_influence_inlier=float(np.mean(influence[~out])),
        loss_influence_outlier=float(np.mean(influence[out])),
        tpr_inlier=float(np.mean(hit[labels & ~is_out])),
        tpr_outlier=float(np.mean(hit[labels & is_out])),
        fpr_inlier=float(np.mean(hit[~labels & ~is_out])),
        fpr_outlier=float(np.mean(hit[~labels & is_out])),
        share_of_hits=float(np.mean(is_out[hit & labels])),
        predicted_tpr_inlier=float(np.mean(predicted[~out])),
        predicted_tpr_outlier=float(np.mean(predicted[out])),
    )


# --- Number of shadow models -------------------------------------------------------------------

SHADOW_COUNTS = (4, 8, 16, 32, 64)


def tpr_vs_shadow_count(exp: Experiment, alpha: float) -> dict[str, list[float]]:
    """TPR at FPR alpha of each LiRA variant when the attacker has only the first k shadows."""
    res: dict[str, list[float]] = {"online": [], "online_global": [], "offline": [], "offline_global": []}
    for k in SHADOW_COUNTS:
        sh = subset(exp.shadows, k)
        per, shared = fit_gaussians(sh), fit_gaussians(sh, global_variance=True)
        phi, lab = exp.targets.phi, exp.targets.member
        res["online"].append(tpr_at_fpr(roc_curve(lira_online(phi, per), lab), alpha))
        res["online_global"].append(tpr_at_fpr(roc_curve(lira_online(phi, shared), lab), alpha))
        res["offline"].append(tpr_at_fpr(roc_curve(lira_offline(phi, per), lab), alpha))
        res["offline_global"].append(tpr_at_fpr(roc_curve(lira_offline(phi, shared), lab), alpha))
    return res


# --- Table ----------------------------------------------------------------------------------------


def print_table(rng: np.random.Generator) -> None:
    exp = run_experiment(rng)
    n_neg = int((~exp.targets.member).sum())
    print(
        f"[Setup] pool={N_POOL}  d={D}  outliers={int(exp.pop.outlier.sum())}  lambda={LAM:g}  "
        f"target models={N_TARGETS}  shadow models={N_SHADOWS}  member pairs={int(exp.targets.member.sum())}  "
        f"non-member pairs={n_neg}"
    )

    print("[Loss attack and the generalization gap]")
    bound = 2.0
    g = gap_report(rng, exp.targets, bound)
    print(f"  train error={g.train_error:.4f}  test error={g.test_error:.4f}  gap={g.test_error - g.train_error:.4f}")
    print(
        f"  'correct means member': TPR={g.correct_attack_tpr:.4f}  FPR={g.correct_attack_fpr:.4f}  "
        f"advantage={g.correct_attack_tpr - g.correct_attack_fpr:.4f}"
    )
    print(f"  train loss={g.train_loss:.4f}  test loss={g.test_loss:.4f}  gap={g.test_loss - g.train_loss:.4f}")
    print(
        f"  loss clipped at B={bound:g}: train={g.clipped_train_loss:.4f}  test={g.clipped_test_loss:.4f}  "
        f"gap/B={(g.clipped_test_loss - g.clipped_train_loss) / bound:.4f}  "
        f"simulated randomized attack advantage={g.bounded_attack_advantage:.4f}"
    )
    print(
        f"  Yeom threshold tau=mean train loss={g.yeom_threshold:.4f}: TPR={g.yeom_tpr:.4f}  "
        f"FPR={g.yeom_fpr:.4f}  advantage={g.yeom_tpr - g.yeom_fpr:.4f}"
    )

    scores = attack_scores(exp)
    labels = exp.targets.member
    rocs = {k: roc_curve(v, labels) for k, v in scores.items()}
    print(f"[Attacks on {N_TARGETS} target models, FPR alpha={ALPHA:g}]")
    for k, r in rocs.items():
        print(
            f"  {ATTACK_NAMES[k]}: AUC={auc(r):.3f}  best advantage={best_advantage(r):.3f}  "
            f"TPR@0.1%={tpr_at_fpr(r, ALPHA):.4f}  TPR@1%={tpr_at_fpr(r, 1e-2):.4f}"
        )
    r = rocs["online"]
    tpr = tpr_at_fpr(r, ALPHA)
    k_tp = round(tpr * r.n_pos)
    lo, hi = clopper_pearson(k_tp, r.n_pos)
    k_fp = math.floor(ALPHA * r.n_neg)
    lo_f, hi_f = clopper_pearson(k_fp, r.n_neg)
    print(
        f"  LiRA online at FPR {ALPHA:g}: {k_tp} true positives of {r.n_pos}, 95% CI [{lo:.4f}, {hi:.4f}]; "
        f"{k_fp} false positives of {r.n_neg}, 95% CI [{lo_f:.5f}, {hi_f:.5f}]"
    )
    ratio = tpr / tpr_at_fpr(rocs["loss"], ALPHA)
    print(f"  LiRA online / loss attack TPR at FPR {ALPHA:g}: {ratio:.1f}x")

    # The shadow attack's threshold, calibrated on the shadow models only
    coefs = shadow_attack_model(exp.shadows, exp.pop.y)
    t = threshold_for_fpr(shadow_attack_scores(exp.shadows.phi, exp.pop.y, coefs), exp.shadows.member, 1e-2)
    s_tpr, s_fpr = rates_at_threshold(scores["shadow"], labels, t)
    print(f"  shadow attack, threshold set on shadows for FPR 1%: on targets TPR={s_tpr:.4f}  FPR={s_fpr:.4f}")
    print(f"  shadow attack slopes: class -1 b={coefs[0][1]:.3f}  class +1 b={coefs[1][1]:.3f}")

    print("[Two example types: 95% typical (phi_out ~ N(3,1), phi_in ~ N(3.5,1)), 5% atypical (N(-2,1), N(2,1))]")
    t_glob, tpr_glob = global_threshold_rates(TWO_TYPES, ALPHA)
    c, fprs, tpr_lr = likelihood_ratio_rates(TWO_TYPES, ALPHA)
    print(f"  global threshold on phi: t={t_glob:.3f}  TPR={tpr_glob:.4f}")
    print(f"  per-example z-score, same FPR on each type: TPR={per_example_z_tpr(TWO_TYPES, ALPHA):.4f}")
    print(
        f"  likelihood ratio (Neyman-Pearson): log-threshold={c:.3f}  FPR typical={fprs[0]:.2e}  "
        f"FPR atypical={fprs[1]:.4f}  TPR={tpr_lr:.4f}"
    )
    for e in TWO_TYPES:
        print(f"  type weight {e.weight:g}: TPR at FPR {ALPHA:g} alone={gaussian_tpr(ALPHA, e.mu_in - e.mu_out):.4f}")

    print("[Same AUC, different exposure: q=0.05]")
    q = 0.05
    a_auc, a_adv, a_tpr = exposed_few_rates(q, ALPHA)
    mu, b_adv, b_tpr = weak_all_rates(a_auc, ALPHA)
    print(f"  A (5% exposed): AUC={a_auc:.4f}  best advantage={a_adv:.4f}  TPR@0.1%={a_tpr:.5f}")
    print(
        f"  B (weak on all): mu={mu:.4f}  AUC={normal_cdf(mu / math.sqrt(2)):.4f}  best advantage={b_adv:.4f}  "
        f"TPR@0.1%={b_tpr:.5f}"
    )

    print("[Outliers]")
    o = outlier_report(exp, scores["online"], ALPHA)
    print(f"  separation: inliers={o.sep_inlier:.3f}  outliers={o.sep_outlier:.3f}")
    print(f"  mean phi when out: inliers={o.mu_out_inlier:.3f}  outliers={o.mu_out_outlier:.3f}")
    print(f"  loss self-influence: inliers={o.loss_influence_inlier:.4f}  outliers={o.loss_influence_outlier:.4f}")
    frac = float(np.mean(exp.pop.outlier))
    mean_influence = (1 - frac) * o.loss_influence_inlier + frac * o.loss_influence_outlier
    print(
        f"  mean loss self-influence={mean_influence:.4f}  outliers' share of it="
        f"{frac * o.loss_influence_outlier / mean_influence:.3f}"
    )
    print(
        f"  LiRA online at pooled FPR {ALPHA:g}: TPR inliers={o.tpr_inlier:.4f}  outliers={o.tpr_outlier:.4f}; "
        f"FPR inliers={o.fpr_inlier:.5f}  outliers={o.fpr_outlier:.5f}"
    )
    print(f"  share of true positives that are outliers={o.share_of_hits:.3f}")
    print(
        f"  Gaussian-predicted TPR at FPR {ALPHA:g}: inliers={o.predicted_tpr_inlier:.4f}  "
        f"outliers={o.predicted_tpr_outlier:.4f}"
    )

    print("[Number of shadow models]")
    counts = tpr_vs_shadow_count(exp, ALPHA)
    for name, vals in counts.items():
        print(f"  {name}: " + "  ".join(f"{k}={v:.4f}" for k, v in zip(SHADOW_COUNTS, vals, strict=True)))


# --- Figures -----------------------------------------------------------------------------------

COLORS = {"loss": "C0", "shadow": "C1", "offline": "C2", "online": "C3", "online_global": "C4"}


def plot_loss_distributions(exp: Experiment) -> Figure:
    """Histograms of φ for members and non-members, with Yeom's threshold (loss <= τ means φ >= φ_τ)."""
    phi, m = exp.targets.phi, exp.targets.member
    tau = float(np.mean(logistic_loss(phi[m])))
    phi_tau = -math.log(math.expm1(tau))  # ln(1 + e^{-φ}) = τ  ⟺  φ = -ln(e^τ - 1)
    bins = [float(b) for b in np.linspace(-6, 8, 85)]
    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.hist(phi[m], bins=bins, density=True, histtype="step", linewidth=1.6, label="members (in the training set)")
    ax.hist(phi[~m], bins=bins, density=True, histtype="step", linewidth=1.6, label="non-members")
    ax.axvline(
        phi_tau,
        color="k",
        linestyle="--",
        linewidth=1,
        label=rf"Yeom threshold: loss $\leq$ mean training loss {tau:.2f}",
    )
    ax.set(
        xlabel=r"$\phi = y\langle \theta, x\rangle$, the logit of the target model's confidence in the true label",
        ylabel="density",
        xlim=(-6, 8),
        ylim=(0, 0.62),
        title="Members score higher on average, but the two distributions overlap almost entirely",
    )
    ax.title.set_fontsize(10)
    ax.legend(fontsize=9, loc="upper right")
    return fig


def plot_lira_example(exp: Experiment) -> Figure:
    """IN and OUT distributions of φ over the shadow models for a typical point and an outlier."""
    fits = fit_gaussians(exp.shadows)
    sep = separation(fits)
    out = exp.pop.outlier
    inl = np.flatnonzero(~out)
    typical = int(inl[np.argmin(np.abs(sep[inl] - np.median(sep[inl])))])
    outl = np.flatnonzero(out)
    atypical = int(outl[np.argmin(np.abs(sep[outl] - np.median(sep[outl])))])
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, i, name in ((axes[0], typical, "a typical point"), (axes[1], atypical, "an outlier")):
        m = exp.shadows.member[:, i]
        v = exp.shadows.phi[:, i]
        lo = float(min(v.min(), fits.mu_out[i] - 3 * fits.sd_out[i]))
        hi = float(max(v.max(), fits.mu_in[i] + 3 * fits.sd_in[i]))
        bins = [float(b) for b in np.linspace(lo, hi, 25)]
        grid = np.linspace(lo, hi, 300)
        ax.hist(v[~m], bins=bins, density=True, alpha=0.35, color="C0", label="OUT models")
        ax.hist(v[m], bins=bins, density=True, alpha=0.35, color="C3", label="IN models")
        for mu, sd, col in ((fits.mu_out[i], fits.sd_out[i], "C0"), (fits.mu_in[i], fits.sd_in[i], "C3")):
            ax.plot(grid, np.exp(log_normal_pdf(grid, np.full_like(grid, mu), np.full_like(grid, sd))), color=col)
        ax.set(
            xlabel=r"$\phi$ = logit of the confidence in the true label",
            ylabel="density",
            title=f"{name}: separation {sep[i]:.2f}",
        )
        ax.legend(fontsize=9)
    return fig


def plot_same_auc(alpha: float, q: float = 0.05) -> Figure:
    a_auc, _, _ = exposed_few_rates(q, alpha)
    mu, _, _ = weak_all_rates(a_auc, alpha)
    fprs = np.logspace(-5, 0, 300)[:-1]  # FPR 1 is the trivial endpoint
    tpr_a = q + (1 - q) * fprs
    tpr_b = np.array([gaussian_tpr(float(f), mu) for f in fprs])
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.2))
    for ax, log in ((ax1, False), (ax2, True)):
        ax.plot(fprs, tpr_a, color="C3", label=f"A: {q:.0%} of members fully exposed")
        ax.plot(fprs, tpr_b, color="C0", label=rf"B: every member shifted by $\mu$ = {mu:.3f}")
        ax.plot(fprs, fprs, ":", color="0.5", label="random guessing")
        ax.set(xlabel="false positive rate", ylabel="true positive rate")
        if log:
            ax.set(xscale="log", yscale="log", xlim=(1e-5, 1), ylim=(1e-5, 1), title="log axes")
            ax.axvline(alpha, color="k", linewidth=0.8, linestyle="--")
        else:
            ax.set(xlim=(0, 1), ylim=(0, 1), title=f"linear axes: both AUCs are {a_auc:.3f}")
        ax.legend(fontsize=8, loc="lower right")
    return fig


def plot_roc(exp: Experiment, alpha: float) -> Figure:
    scores = attack_scores(exp)
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    for k, s in scores.items():
        r = roc_curve(s, exp.targets.member)
        ax.plot(
            np.maximum(r.fpr, 1e-6),
            np.maximum(r.tpr, 1e-6),
            color=COLORS[k],
            label=f"{ATTACK_NAMES[k]} (AUC {auc(r):.2f})",
        )
    ax.plot([1e-5, 1], [1e-5, 1], ":", color="0.5", label="random guessing")
    ax.axvline(alpha, color="k", linewidth=0.8, linestyle="--")
    ax.set(
        xscale="log",
        yscale="log",
        xlim=(1e-5, 1),
        ylim=(1e-5, 1),
        xlabel="false positive rate",
        ylabel="true positive rate",
        title="ROC curves of the attacks, log axes",
    )
    ax.legend(fontsize=8, loc="lower right")
    return fig


def plot_outliers(exp: Experiment) -> Figure:
    fits = fit_gaussians(exp.shadows)
    sep = separation(fits)
    out = exp.pop.outlier
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.scatter(fits.mu_out[~out], sep[~out], s=8, alpha=0.5, color="C0", label="typical points")
    ax.scatter(fits.mu_out[out], sep[out], s=16, alpha=0.8, color="C3", marker="^", label="outliers")
    ax.axvline(0, color="0.6", linewidth=0.8)
    ax.set(
        xlabel=r"$\mu_{\mathrm{out}}$: mean $\phi$ of models that did not train on the point",
        ylabel=r"separation $(\mu_{\mathrm{in}} - \mu_{\mathrm{out}})/\sigma$",
        title="Points the model cannot predict without seeing them are the most exposed",
    )
    ax.title.set_fontsize(10)
    ax.legend(fontsize=9)
    return fig


def plot_shadow_count(exp: Experiment, alpha: float) -> Figure:
    res = tpr_vs_shadow_count(exp, alpha)
    labels = {
        "online": "online, per-example variance",
        "online_global": "online, global variance",
        "offline": "offline, per-example variance",
        "offline_global": "offline, global variance",
    }
    styles = {"online": "o-", "online_global": "o--", "offline": "s-", "offline_global": "s--"}
    colors = {"online": "C3", "online_global": "C4", "offline": "C2", "offline_global": "C8"}
    fig, ax = plt.subplots(figsize=(7, 4.2))
    for k, vals in res.items():
        ax.plot(SHADOW_COUNTS, vals, styles[k], color=colors[k], label=labels[k])
    loss_tpr = tpr_at_fpr(roc_curve(loss_attack_scores(exp.targets.phi), exp.targets.member), alpha)
    ax.axhline(loss_tpr, color=COLORS["loss"], linestyle=":", label="loss threshold (no shadows)")
    ax.set(
        xscale="log",
        xticks=list(SHADOW_COUNTS),
        xticklabels=[str(k) for k in SHADOW_COUNTS],
        xlabel="number of shadow models",
        ylabel=f"TPR at FPR {alpha:g}",
        title="LiRA needs enough shadow models to estimate per-example variances",
    )
    ax.xaxis.set_minor_locator(NullLocator())
    ax.title.set_fontsize(10)
    ax.legend(fontsize=8)
    return fig


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=FIGURES, help="directory for the figures")
    args = parser.parse_args()
    seed: int = args.seed
    out: Path = args.out

    print_table(np.random.default_rng(seed))

    rng = np.random.default_rng(seed + 1)  # separate stream, so the table stays reproducible
    exp = run_experiment(rng)
    out.mkdir(parents=True, exist_ok=True)
    figures = {
        "mia_loss_distributions.svg": plot_loss_distributions(exp),
        "mia_lira_example.svg": plot_lira_example(exp),
        "mia_same_auc.svg": plot_same_auc(ALPHA),
        "mia_roc.svg": plot_roc(exp, ALPHA),
        "mia_outliers.svg": plot_outliers(exp),
        "mia_shadow_count.svg": plot_shadow_count(exp, ALPHA),
    }
    for name, fig in figures.items():
        fig.tight_layout()
        fig.savefig(out / name, metadata={"Date": None})  # no timestamp, so output is reproducible
        plt.close(fig)
        print(f"saved {out / name}")


if __name__ == "__main__":
    main()
