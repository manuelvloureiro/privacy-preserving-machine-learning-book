"""Check the property-inference attacks of the property-inference chapter and draw its figures.

    uv run python book/part02-attacks/code/property_inference.py
    uv run python book/part02-attacks/code/property_inference.py --seed 1 --out /tmp/figures

Prints the numbers quoted in 04-property-inference.md (they match exactly with the default seed),
then saves the chapter's figures.

The game: a dataset of n records is drawn from a distribution in which a fraction t of the records
belong to a subgroup (the property). The challenger picks t0 (b = 0) or t1 (b = 1), trains a model,
and hands it over. The adversary guesses b. Hypothesis 0 plays the role of P in the hypothesis-testing
chapter and hypothesis 1 the role of Q, so advantage = TPR - FPR and accuracy = (1 + advantage) / 2.

Three toys:
  * a released count of subgroup members (exact binomial calculations);
  * logistic regression, where the subgroup has a different label rule, attacked by a meta-classifier
    trained on the parameters of shadow models;
  * a one-hidden-layer tanh network, where the meta-classifier must cope with the symmetries of the
    hidden units.
"""

import argparse
import math
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
matplotlib.rcParams["svg.hashsalt"] = "property_inference"  # stable element ids, so reruns don't churn the SVGs
matplotlib.rcParams["svg.fonttype"] = "none"  # keep text as text: smaller, diffable files

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from numpy.typing import NDArray

FIGURES = Path(__file__).resolve().parent.parent / "figures"

Array = NDArray[np.float64]
FeatureMap = Callable[["MLP"], Array]

# The logistic-regression toy: two standard normal features; the majority group labels with
# w_A, the subgroup (the property) with w_B. The group itself is never a feature.
W_MAJORITY: tuple[float, float, float] = (2.0, 1.0, 0.0)  # (w1, w2, bias)
W_SUBGROUP: tuple[float, float, float] = (2.0, -1.0, 0.0)
SHADOWS = 150  # shadow models per hypothesis
TARGETS = 150  # target models per hypothesis, used only to measure accuracy


def normal_cdf(x: float) -> float:
    """Φ(x); erfc keeps full relative precision in the lower tail."""
    return 0.5 * math.erfc(-x / math.sqrt(2))


def normal_quantile(p: float) -> float:
    """Φ⁻¹(p) by bisection."""
    lo, hi = -40.0, 40.0
    for _ in range(200):
        mid = (lo + hi) / 2
        if normal_cdf(mid) < p:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def sigmoid(z: Array) -> Array:
    return 0.5 * (1.0 + np.tanh(0.5 * z))  # equal to 1/(1 + e^{-z}), without overflow


# --- A released count: membership against property ----------------------------------------


def binomial_pmf(n: int, t: float) -> Array:
    """P(Bin(n, t) = k) for k = 0..n, computed in log space."""
    k = np.arange(n + 1)
    log_choose = np.array([math.lgamma(n + 1) - math.lgamma(i + 1) - math.lgamma(n - i + 1) for i in range(n + 1)])
    with np.errstate(divide="ignore"):
        log_p = log_choose + k * math.log(t) + (n - k) * math.log1p(-t)
    return np.exp(log_p)


def tv_pmf(p: Array, q: Array) -> float:
    return float(0.5 * np.abs(p - q).sum())


def property_advantage_count(n: int, t0: float, t1: float) -> float:
    """Best advantage from the count S ~ Bin(n, t): TV(Bin(n, t0), Bin(n, t1))."""
    return tv_pmf(binomial_pmf(n, t0), binomial_pmf(n, t1))


def property_advantage_normal(n: int, t0: float, t1: float) -> float:
    """Normal approximation 2Φ(√n |t1 - t0| / (2σ̄)) - 1 with σ̄² = t̄(1 - t̄), t̄ the midpoint."""
    t_bar = (t0 + t1) / 2
    return 2 * normal_cdf(math.sqrt(n) * abs(t1 - t0) / (2 * math.sqrt(t_bar * (1 - t_bar)))) - 1


def membership_advantage_count(n: int, t: float) -> float:
    """Best advantage from S for "is the target (a subgroup member) in D?".

    D is n - 1 population records plus either the target (S = X + 1) or a fresh record
    (S = X + B, B ~ Bernoulli(t)), with X ~ Bin(n - 1, t). The answer is (1 - t) max_k P(X = k).
    """
    pmf_x = binomial_pmf(n - 1, t)
    with_target = np.concatenate([[0.0], pmf_x])  # law of X + 1 on 0..n
    without = t * with_target + (1 - t) * np.concatenate([pmf_x, [0.0]])  # law of X + B
    return tv_pmf(with_target, without)


def membership_advantage_formula(n: int, t: float) -> float:
    return (1 - t) * float(binomial_pmf(n - 1, t).max())


def membership_advantage_normal(n: int, t: float) -> float:
    """(1 - t) / √(2π(n - 1)t(1 - t)): the mode of a binomial, from its normal approximation."""
    return (1 - t) / math.sqrt(2 * math.pi * (n - 1) * t * (1 - t))


def simulate_count_game(rng: np.random.Generator, n: int, t0: float, t1: float, games: int) -> float:
    """Empirical advantage of the likelihood-ratio test on S in the property game."""
    b = rng.integers(0, 2, size=games)
    s = rng.binomial(n, np.where(b == 1, t1, t0))
    p0, p1 = binomial_pmf(n, t0), binomial_pmf(n, t1)
    guess = (p1[s] > p0[s]).astype(np.int64)
    tpr = float(guess[b == 1].mean())
    fpr = float(guess[b == 0].mean())
    return tpr - fpr


# --- Shadow models and the meta-classifier (logistic regression) ---------------------------


def sample_datasets(rng: np.random.Generator, models: int, n: int, t: float) -> tuple[Array, Array]:
    """`models` datasets of n records each; a fraction t (in expectation) from the subgroup.

    Returns X of shape (models, n, 3), the two features and a constant 1, and labels y (models, n).
    """
    x = rng.standard_normal((models, n, 2))
    group = rng.random((models, n)) < t
    w_a = np.array(W_MAJORITY)
    w_b = np.array(W_SUBGROUP)
    design = np.concatenate([x, np.ones((models, n, 1))], axis=2)
    logits = np.where(group, design @ w_b, design @ w_a)
    y = (rng.random((models, n)) < sigmoid(logits)).astype(np.float64)
    return design, y


def train_logistic(design: Array, y: Array, l2: float = 1e-3, steps: int = 8) -> Array:
    """Many logistic regressions at once by Newton's method; returns parameters (models, 3).

    Minimizes the mean log-loss plus (l2/2)|θ|², so every problem is strongly convex.
    """
    models, n, d = design.shape
    cols = [design[:, :, i] for i in range(d)]  # elementwise products beat tiny batched matmuls
    theta = np.zeros((models, d))
    eye = np.eye(d)
    for _ in range(steps):
        p = sigmoid(sum((theta[:, i, None] * cols[i] for i in range(d)), np.zeros((models, n))))
        resid, weight = p - y, p * (1 - p)
        grad = np.stack([(resid * col).mean(axis=1) for col in cols], axis=1) + l2 * theta
        hess = np.empty((models, d, d))
        for i in range(d):
            for j in range(i, d):
                hess[:, i, j] = hess[:, j, i] = (weight * cols[i] * cols[j]).mean(axis=1)
        theta = theta - np.linalg.solve(hess + l2 * eye, grad[..., None])[..., 0]
    return theta


def logistic_models(rng: np.random.Generator, models: int, n: int, t: float) -> Array:
    design, y = sample_datasets(rng, models, n, t)
    return train_logistic(design, y)


@dataclass(frozen=True)
class Meta:
    """A linear meta-classifier: guess b = 1 when w·(f - mid) > 0."""

    w: Array
    mid: Array


def fit_lda(f0: Array, f1: Array, ridge: float = 1e-6) -> Meta:
    """Linear discriminant: the Neyman–Pearson test for two Gaussians with a shared covariance.

    Features are not standardized; the ridge is relative to the average variance.
    """
    m0, m1 = f0.mean(axis=0), f1.mean(axis=0)
    centred = np.concatenate([f0 - m0, f1 - m1])
    cov = centred.T @ centred / (len(centred) - 2)
    cov += ridge * float(np.trace(cov)) / len(cov) * np.eye(len(cov))
    return Meta(w=np.linalg.solve(cov, m1 - m0), mid=(m0 + m1) / 2)


def meta_accuracy(meta: Meta, g0: Array, g1: Array) -> float:
    """Accuracy on target models, half from each hypothesis."""
    correct0 = ((g0 - meta.mid) @ meta.w <= 0).mean()
    correct1 = ((g1 - meta.mid) @ meta.w > 0).mean()
    return float((correct0 + correct1) / 2)


def mahalanobis(f0: Array, f1: Array) -> float:
    """Δ = √((m1 - m0)ᵀ Σ⁻¹ (m1 - m0)) with the pooled covariance."""
    meta = fit_lda(f0, f1, ridge=0.0)
    m0, m1 = f0.mean(axis=0), f1.mean(axis=0)
    return math.sqrt(max(float((m1 - m0) @ meta.w), 0.0))


def gaussian_accuracy(delta: float) -> float:
    """Best accuracy between N(m0, Σ) and N(m1, Σ) at Mahalanobis distance Δ: Φ(Δ/2)."""
    return normal_cdf(delta / 2)


# The population view: every trained model scatters around the population fit θ(t), with
# covariance Σ₁(t)/n. Both are expectations over x ~ N(0, I₂), computed by Gauss–Hermite quadrature.

GH_POINTS = 40


def quadrature_grid() -> tuple[Array, Array]:
    """Nodes (k, 3), rows (x1, x2, 1), and weights (k,) for E[g(x)] with x ~ N(0, I₂)."""
    nodes, weights = np.polynomial.hermite_e.hermegauss(GH_POINTS)  # weight function e^{-x²/2}
    weights = weights / weights.sum()
    x1, x2 = np.meshgrid(nodes, nodes, indexing="ij")
    w = np.outer(weights, weights).ravel()
    return np.stack([x1.ravel(), x2.ravel(), np.ones(x1.size)], axis=1), w


def mean_label(points: Array, t: float) -> Array:
    """E[y | x] = (1 - t) σ(w_A·x) + t σ(w_B·x): the mixture's label probability."""
    return (1 - t) * sigmoid(points @ np.array(W_MAJORITY)) + t * sigmoid(points @ np.array(W_SUBGROUP))


def population_fit(t: float, steps: int = 30) -> Array:
    """θ(t): the minimizer of the expected log-loss (Newton's method on the quadrature grid)."""
    points, weights = quadrature_grid()
    m = mean_label(points, t)
    theta = np.zeros(3)
    for _ in range(steps):
        p = sigmoid(points @ theta)
        grad = points.T @ (weights * (p - m))
        hess = points.T @ (points * (weights * p * (1 - p))[:, None])
        theta = theta - np.linalg.solve(hess, grad)
    return theta


def per_record_covariance(t: float) -> Array:
    """Σ₁(t) = H⁻¹GH⁻¹, so that θ̂ ≈ N(θ(t), Σ₁/n): the sandwich covariance of an M-estimator.

    H = E[p(1 - p) x xᵀ] is the Hessian of the expected loss, G = E[(y - p)² x xᵀ] the covariance of
    one record's gradient, with E[(y - p)² | x] = m(1 - m) + (m - p)² for m = E[y | x].
    """
    points, weights = quadrature_grid()
    m = mean_label(points, t)
    p = sigmoid(points @ population_fit(t))
    hess = points.T @ (points * (weights * p * (1 - p))[:, None])
    grad_cov = points.T @ (points * (weights * (m * (1 - m) + (m - p) ** 2))[:, None])
    h_inv = np.linalg.inv(hess)
    return h_inv @ grad_cov @ h_inv


def predicted_delta(n: int, t0: float, t1: float) -> float:
    """Mahalanobis distance between N(θ(t0), Σ̄/n) and N(θ(t1), Σ̄/n), Σ̄ the average of Σ₁(t0), Σ₁(t1)."""
    gap = population_fit(t1) - population_fit(t0)
    cov = (per_record_covariance(t0) + per_record_covariance(t1)) / 2
    return math.sqrt(n * float(gap @ np.linalg.solve(cov, gap)))


def kappa(t: float = 0.5, h: float = 1e-4) -> float:
    """κ = |dθ/dt| in the metric Σ₁⁻¹, so that Δ ≈ κ √n |t1 - t0| for nearby t0, t1."""
    slope = (population_fit(t + h) - population_fit(t - h)) / (2 * h)
    return math.sqrt(float(slope @ np.linalg.solve(per_record_covariance(t), slope)))


@dataclass(frozen=True)
class MetaResult:
    accuracy: float  # of the meta-classifier on fresh target models
    predicted: float  # Φ(Δ/2) with Δ from the population view


def logistic_attack(rng: np.random.Generator, n: int, t0: float, t1: float) -> MetaResult:
    """Train shadow models on both hypotheses, fit the meta-classifier, test on target models."""
    s0 = logistic_models(rng, SHADOWS, n, t0)
    s1 = logistic_models(rng, SHADOWS, n, t1)
    g0 = logistic_models(rng, TARGETS, n, t0)
    g1 = logistic_models(rng, TARGETS, n, t1)
    meta = fit_lda(s0, s1)
    return MetaResult(meta_accuracy(meta, g0, g1), gaussian_accuracy(predicted_delta(n, t0, t1)))


def centred_pair(delta_t: float) -> tuple[float, float]:
    return 0.5 - delta_t / 2, 0.5 + delta_t / 2


# --- Hidden-unit symmetries (one-hidden-layer tanh network) --------------------------------


@dataclass(frozen=True)
class MLP:
    """A batch of networks f(x) = σ(vᵀ tanh(Wx + b) + c). Shapes: W (m, h, 2), b, v (m, h), c (m,)."""

    w: Array
    b: Array
    v: Array
    c: Array


def mlp_logits(net: MLP, x: Array) -> Array:
    """Logits of each network on its own inputs x (m, n, 2); returns (m, n)."""
    hidden = np.tanh(x @ net.w.transpose(0, 2, 1) + net.b[:, None, :])
    return (hidden @ net.v[:, :, None])[:, :, 0] + net.c[:, None]


def mlp_logits_shared(net: MLP, x: Array) -> Array:
    """Logits of every network on the same inputs x (k, 2); returns (m, k)."""
    hidden = np.tanh(x[None, :, :] @ net.w.transpose(0, 2, 1) + net.b[:, None, :])
    return (hidden @ net.v[:, :, None])[:, :, 0] + net.c[:, None]


def train_mlp(rng: np.random.Generator, x: Array, y: Array, hidden: int, steps: int = 300, lr: float = 2.0) -> MLP:
    """Full-batch gradient descent on the mean log-loss, from a random initialization.

    Runs in float32 with the record axis last, which is several times faster than float64.
    """
    m, n, d = x.shape
    f32 = np.float32
    w = rng.standard_normal((m, hidden, d)).astype(f32)
    b = (0.5 * rng.standard_normal((m, hidden))).astype(f32)
    v = (rng.standard_normal((m, hidden)) / math.sqrt(hidden)).astype(f32)
    c = np.zeros(m, dtype=f32)
    x1, x2 = x[:, None, :, 0].astype(f32), x[:, None, :, 1].astype(f32)  # (m, 1, n)
    y32 = y.astype(f32)
    for _ in range(steps):
        act = np.tanh(w[:, :, 0, None] * x1 + w[:, :, 1, None] * x2 + b[:, :, None])  # (m, h, n)
        logit = (v[:, None, :] @ act)[:, 0, :] + c[:, None]
        err = (f32(0.5) * (1 + np.tanh(f32(0.5) * logit)) - y32) / f32(n)  # d(mean loss)/d logit
        grad_v = (act @ err[:, :, None])[:, :, 0]
        grad_c = err.sum(axis=1)
        back = err[:, None, :] * v[:, :, None] * (1 - act * act)  # d loss / d pre-activation
        grad_w = np.stack([(back * x1).sum(axis=2), (back * x2).sum(axis=2)], axis=2)
        grad_b = back.sum(axis=2)
        step = f32(lr)
        w, b, v, c = w - step * grad_w, b - step * grad_b, v - step * grad_v, c - step * grad_c
    return MLP(w.astype(np.float64), b.astype(np.float64), v.astype(np.float64), c.astype(np.float64))


def mlp_datasets(rng: np.random.Generator, models: int, n: int, t: float) -> tuple[Array, Array]:
    """Same records as the logistic toy, without the constant column."""
    design, y = sample_datasets(rng, models, n, t)
    return design[:, :, :2], y


def permute_units(net: MLP, perm: NDArray[np.int64], signs: Array) -> MLP:
    """Reorder the hidden units by `perm` and flip unit j when signs[j] = -1 (tanh is odd)."""
    return MLP(
        net.w[:, perm, :] * signs[None, :, None],
        net.b[:, perm] * signs[None, :],
        net.v[:, perm] * signs[None, :],
        net.c,
    )


def units(net: MLP) -> Array:
    """Each hidden unit as a vector u_j = (W_j1, W_j2, b_j, v_j); shape (m, h, 4)."""
    return np.concatenate([net.w, net.b[..., None], net.v[..., None]], axis=2)


def raw_features(net: MLP) -> Array:
    """The flattened parameter vector, in whatever order training left the units."""
    m = len(net.c)
    return np.concatenate([units(net).reshape(m, -1), net.c[:, None]], axis=1)


def canonical_features(net: MLP) -> Array:
    """Flip each unit so that v_j > 0, then sort the units by v_j: one representative per orbit."""
    u = units(net)
    u = u * np.sign(u[:, :, 3:4])
    order = np.argsort(u[:, :, 3], axis=1)
    u = np.take_along_axis(u, order[:, :, None], axis=1)
    return np.concatenate([u.reshape(len(u), -1), net.c[:, None]], axis=1)


def set_features(net: MLP) -> Array:
    """Σ_j u_j u_jᵀ (upper triangle) and c: invariant to reordering and to sign flips."""
    u = units(net)
    second = np.einsum("mhi,mhj->mij", u, u)
    rows, cols = np.triu_indices(u.shape[2])
    return np.concatenate([second[:, rows, cols], net.c[:, None]], axis=1)


PROBES = np.array([[math.cos(a), math.sin(a)] for a in np.linspace(0, 2 * math.pi, 16, endpoint=False)]) * 1.5


def probe_features(net: MLP) -> Array:
    """Black-box features: the logits on 16 fixed probe inputs on a circle of radius 1.5."""
    return mlp_logits_shared(net, PROBES)


REPRESENTATIONS: dict[str, FeatureMap] = {
    "raw parameters": raw_features,
    "canonical order": canonical_features,
    "unit second moments": set_features,
    "probe outputs": probe_features,
}


def mlp_models(rng: np.random.Generator, models: int, n: int, t: float, hidden: int) -> MLP:
    x, y = mlp_datasets(rng, models, n, t)
    return train_mlp(rng, x, y, hidden)


def subset(net: MLP, rows: slice) -> MLP:
    return MLP(net.w[rows], net.b[rows], net.v[rows], net.c[rows])


def standardized_gaps(f0: Array, f1: Array) -> Array:
    """|mean difference| / pooled standard deviation, feature by feature."""
    pooled = np.sqrt((f0.var(axis=0, ddof=1) + f1.var(axis=0, ddof=1)) / 2)
    return np.abs(f1.mean(axis=0) - f0.mean(axis=0)) / pooled


@dataclass(frozen=True)
class RepresentationResult:
    accuracy: float  # of the linear meta-classifier on the target networks
    largest_gap: float  # the largest standardized mean gap over features, on the shadow networks


def mlp_attack(
    rng: np.random.Generator, n: int, t0: float, t1: float, hidden: int, models: int = SHADOWS
) -> dict[str, RepresentationResult]:
    """Meta-classifier accuracy for each representation: `models` shadow and target networks per side."""
    nets0 = mlp_models(rng, 2 * models, n, t0, hidden)
    nets1 = mlp_models(rng, 2 * models, n, t1, hidden)
    shadow, target = slice(0, models), slice(models, 2 * models)
    result: dict[str, RepresentationResult] = {}
    for name, feature in REPRESENTATIONS.items():
        s0, s1 = feature(subset(nets0, shadow)), feature(subset(nets1, shadow))
        meta = fit_lda(s0, s1, ridge=1e-3)
        accuracy = meta_accuracy(meta, feature(subset(nets0, target)), feature(subset(nets1, target)))
        result[name] = RepresentationResult(accuracy, float(standardized_gaps(s0, s1).max()))
    return result


# --- Table ---------------------------------------------------------------------------------

COUNT_NS = (100, 1000, 10_000)
SWEEP_NS = (50, 100, 200, 500, 1000)
SWEEP_DELTAS = (0.05, 0.1, 0.2, 0.4)
MLP_N = 200
MLP_HIDDEN = 4
MLP_T = (0.2, 0.8)


def print_table(rng: np.random.Generator) -> None:
    t0, t1 = 0.45, 0.55
    print("== A released count: property against membership (t0=0.45, t1=0.55; membership at t=0.5) ==")
    for n in COUNT_NS:
        print(
            f"n={n}: property TV={property_advantage_count(n, t0, t1):.4f}  "
            f"normal approx={property_advantage_normal(n, t0, t1):.4f}  "
            f"membership={membership_advantage_count(n, 0.5):.4f}  "
            f"(1-t) max pmf={membership_advantage_formula(n, 0.5):.4f}  "
            f"normal approx={membership_advantage_normal(n, 0.5):.4f}"
        )
    print(
        f"simulated property game, n=100, 200000 games: advantage={simulate_count_game(rng, 100, t0, t1, 200_000):.4f}"
    )

    print("\n== Logistic regression: the population fit theta(t) and the per-record covariance ==")
    for t in (0.0, 0.25, 0.5, 0.75, 1.0):
        theta = population_fit(t)
        print(f"t={t:.2f}: w1={theta[0]:.3f}  w2={theta[1]:.3f}  bias={abs(theta[2]):.3f}")
    cov = per_record_covariance(0.5)
    print(f"Sigma_1(0.5) diagonal: {cov[0, 0]:.2f}  {cov[1, 1]:.2f}  {cov[2, 2]:.2f}")
    slope = (population_fit(0.5 + 1e-4) - population_fit(0.5 - 1e-4)) / 2e-4
    k = kappa()
    print(
        f"d theta/dt at 0.5: ({slope[0]:.3f}, {slope[1]:.3f}, {abs(slope[2]):.3f})  kappa={k:.3f}  "
        f"released-count kappa=2  effective fraction (kappa/2)^2={(k / 2) ** 2:.3f}"
    )
    empirical = logistic_models(rng, 2000, 1000, 0.5)
    emp_cov = np.cov(empirical.T) * 1000
    print(
        f"2000 models at n=1000, t=0.5: mean w1={empirical[:, 0].mean():.3f}  "
        f"n * var: {emp_cov[0, 0]:.2f}  {emp_cov[1, 1]:.2f}  {emp_cov[2, 2]:.2f}"
    )

    print("\n== Logistic regression: meta-classifier accuracy (prediction Phi(Delta/2)); t centred on 0.5 ==")
    for n in SWEEP_NS:
        cells: list[str] = []
        for d in SWEEP_DELTAS:
            res = logistic_attack(rng, n, *centred_pair(d))
            cells.append(f"d={d:g}: {res.accuracy:.3f} ({res.predicted:.3f})")
        print(f"n={n}: " + "  ".join(cells))

    print(f"\n== Tanh network, h={MLP_HIDDEN}, n={MLP_N}, t0={MLP_T[0]}, t1={MLP_T[1]}: accuracy by representation ==")
    for name, res in mlp_attack(rng, MLP_N, MLP_T[0], MLP_T[1], MLP_HIDDEN).items():
        print(f"{name}: accuracy={res.accuracy:.3f}  largest standardized mean gap={res.largest_gap:.2f}")
    net = mlp_models(rng, 3, 50, 0.5, MLP_HIDDEN)
    perm = rng.permutation(MLP_HIDDEN)
    signs = np.where(rng.random(MLP_HIDDEN) < 0.5, -1.0, 1.0)
    x = rng.standard_normal((1000, 2))
    diff = float(np.abs(mlp_logits_shared(net, x) - mlp_logits_shared(permute_units(net, perm, signs), x)).max())
    print(f"permuted and sign-flipped network: same logits on 1000 inputs (to 1e-12): {diff < 1e-12}")
    print(f"equivalent parameter vectors per network: h! 2^h = {math.factorial(MLP_HIDDEN) * 2**MLP_HIDDEN}")


# --- Figures -------------------------------------------------------------------------------


def plot_count_leak() -> Figure:
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(11, 4.2))
    n = 1000
    k = np.arange(n + 1) / n
    for t, colour in ((0.45, "C0"), (0.55, "C1")):
        ax0.plot(k, binomial_pmf(n, t), color=colour, label=f"fraction under t = {t}")
    shifted = np.concatenate([[0.0], binomial_pmf(n - 1, 0.5)])
    ax0.plot(k, binomial_pmf(n, 0.5), color="0.3", linewidth=3, alpha=0.5, label="t = 0.5, without the target")
    ax0.plot(k, shifted, color="C3", linestyle=":", label="t = 0.5, target added")
    ax0.set(
        xlim=(0.38, 0.62),
        ylim=(0, 0.037),
        xlabel="released fraction S/n",
        ylabel="probability",
        title=f"n = {n}: the property moves the count by 100,\none record moves it by 1",
    )
    ax0.legend(fontsize=8, loc="upper left", ncols=2)
    ns = np.unique(np.round(np.logspace(1, 4.3, 60)).astype(int))
    ax1.plot(ns, [property_advantage_count(int(m), 0.45, 0.55) for m in ns], label="property: t = 0.45 vs 0.55")
    ax1.plot(ns, [property_advantage_count(int(m), 0.48, 0.52) for m in ns], label="property: t = 0.48 vs 0.52")
    ax1.plot(
        ns,
        [membership_advantage_count(int(m), 0.5) for m in ns],
        color="0.3",
        label="membership of one record (t = 0.5)",
    )
    ax1.set(
        xscale="log",
        yscale="log",
        ylim=(1e-3, 1.3),
        xlabel="dataset size n",
        ylabel="best advantage",
        title="More data: the property leaks more,\neach record leaks less",
    )
    ax1.legend(fontsize=8, loc="lower left")
    return fig


def plot_parameter_cloud(rng: np.random.Generator) -> Figure:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.3), sharex=True, sharey=True)
    t0, t1 = centred_pair(0.2)
    for ax, n in zip(axes, (100, 1000), strict=True):
        s0 = logistic_models(rng, 300, n, t0)
        s1 = logistic_models(rng, 300, n, t1)
        ax.scatter(s0[:, 0], s0[:, 1], s=6, alpha=0.5, color="C0", label=f"trained with t = {t0:.1f}")
        ax.scatter(s1[:, 0], s1[:, 1], s=6, alpha=0.5, color="C1", label=f"trained with t = {t1:.1f}")
        meta = fit_lda(s0[:, :2], s1[:, :2])
        xs = np.linspace(0.8, 3.6, 2)
        # boundary: w·(θ - mid) = 0, solved for w2
        ys = meta.mid[1] - meta.w[0] / meta.w[1] * (xs - meta.mid[0])
        ax.plot(xs, ys, "k--", linewidth=1, label="linear meta-classifier")
        ax.set(
            xlabel=r"$w_1$",
            title=f"n = {n}: accuracy {meta_accuracy(meta, s0[:, :2], s1[:, :2]):.2f} (on these models)",
        )
    axes[0].set(ylabel=r"$w_2$", xlim=(0.8, 3.6), ylim=(-1.2, 1.2))
    axes[0].legend(fontsize=8, loc="lower right")
    return fig


def plot_meta_accuracy(rng: np.random.Generator) -> Figure:
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(11, 4.2))
    ns = SWEEP_NS
    for d in SWEEP_DELTAS:
        acc = [logistic_attack(rng, n, *centred_pair(d)).accuracy for n in ns]
        ax0.plot(ns, acc, "o-", markersize=4, label=f"|t1 - t0| = {d:g}")
        ax1.plot([math.sqrt(n) * d for n in ns], acc, "o", markersize=4, label=f"|t1 - t0| = {d:g}")
    # Gaussian law, slope from the population parameters: Δ ≈ √n |Δt| κ
    k = kappa()
    xs = np.linspace(0, 13, 200)
    ax1.plot(
        xs, [gaussian_accuracy(k * x) for x in xs], "k--", linewidth=1, label=r"$\Phi(\kappa\sqrt{n}\,|t_1 - t_0|/2)$"
    )
    ax0.set(
        xscale="log",
        xlabel="dataset size n",
        ylabel="meta-classifier accuracy",
        ylim=(0.45, 1.02),
        title="Accuracy grows with n and with the gap",
    )
    ax1.set(
        xlabel=r"$\sqrt{n}\,|t_1 - t_0|$",
        ylim=(0.45, 1.02),
        title=r"Against $\sqrt{n}\,|t_1 - t_0|$ the curves collapse",
    )
    ax0.legend(fontsize=8, loc="lower right")
    ax1.legend(fontsize=8, loc="lower right")
    return fig


def plot_symmetries(rng: np.random.Generator, models: int = 100) -> Figure:
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(11, 4.3), gridspec_kw={"width_ratios": [1.3, 1]})
    t0, t1 = MLP_T
    nets0 = mlp_models(rng, 2 * models, MLP_N, t0, MLP_HIDDEN)
    nets1 = mlp_models(rng, 2 * models, MLP_N, t1, MLP_HIDDEN)
    shadow, target = slice(0, models), slice(models, 2 * models)
    names = list(REPRESENTATIONS)
    accuracies: list[float] = []
    for i, name in enumerate(names):
        feature = REPRESENTATIONS[name]
        s0, s1 = feature(subset(nets0, shadow)), feature(subset(nets1, shadow))
        gaps = standardized_gaps(s0, s1)
        ax0.scatter(np.full(len(gaps), i) + np.linspace(-0.25, 0.25, len(gaps)), gaps, s=12, color=f"C{i}")
        meta = fit_lda(s0, s1, ridge=1e-3)
        accuracies.append(meta_accuracy(meta, feature(subset(nets0, target)), feature(subset(nets1, target))))
    labels = [name.replace(" ", "\n", 1) for name in names]
    ax0.set_xticks(range(len(names)), labels)
    ax0.set(ylabel="|mean gap| / standard deviation", title=f"Each feature's shift between t = {t0} and t = {t1}")
    ax1.bar(range(len(names)), accuracies, color=[f"C{i}" for i in range(len(names))])
    ax1.axhline(0.5, color="0.4", linestyle="--", linewidth=1, label="chance")
    ax1.set_xticks(range(len(names)), labels)
    ax1.set(ylim=(0, 1.05), ylabel="linear meta-classifier accuracy", title="What a linear meta-classifier can use")
    ax1.legend(fontsize=8, loc="upper left")
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
    out.mkdir(parents=True, exist_ok=True)
    figures = {
        "prop_count_leak.svg": plot_count_leak(),
        "prop_parameter_cloud.svg": plot_parameter_cloud(rng),
        "prop_meta_accuracy.svg": plot_meta_accuracy(rng),
        "prop_symmetries.svg": plot_symmetries(rng),
    }
    for name, fig in figures.items():
        fig.tight_layout()
        fig.savefig(out / name, metadata={"Date": None})  # no timestamp, so output is reproducible
        plt.close(fig)
        print(f"saved {out / name}")


if __name__ == "__main__":
    main()
