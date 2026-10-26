"""Check the derivations of the appendix on optimization and estimation, and draw its figures.

    uv run python code/appendix-derivations/apx_optimization.py
    uv run python code/appendix-derivations/apx_optimization.py --seed 1 --out /tmp/figures

Prints every number quoted in book-full/appendix-derivations/07-optimization-estimation.md (they
match exactly with the default seed), then saves the appendix's figures. The six sections are:
convexity and linear convergence of gradient descent, co-coercivity, the lower bound for convex
Lipschitz optimization, the 1-nearest-neighbour limit, M-estimators and the influence of one
record, and the Bayes-optimal membership attack under a Gibbs posterior.
"""

import os

# The models are tiny; multi-threaded BLAS only adds overhead (and is 40x slower on some machines).
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import argparse
import itertools
import math
import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # for code/shared
from shared.figures import configure
from shared.stats import sigmoid

configure("apx_optimization")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from numpy.typing import NDArray

# Figures are build output: _build/figures/<part>/, which the build copies into each book.
FIGURES = Path(__file__).resolve().parents[2] / "_build" / "figures" / Path(__file__).parent.name

Array = NDArray[np.float64]


def sigmoid_scalar(s: float) -> float:
    return 0.5 * (1.0 + math.tanh(0.5 * s))


# --- 1. Convexity and linear convergence of gradient descent ----------------------------------
# The data are those of the optimization chapter: x uniform on the unit sphere in R^5, labels from
# a logistic model. The risk is the average logistic loss plus (λ/2)‖w‖².

W_TRUE = np.array([2.0, -2.0, 1.0, 0.0, 0.0])
GD_N = 1000
GD_LAM = 0.01


@dataclass(frozen=True)
class Data:
    x: Array  # (n, d), every row has unit norm
    y: Array  # (n,), labels ±1


def make_logistic_data(rng: np.random.Generator, n: int, w_true: Array = W_TRUE) -> Data:
    """x uniform on the unit sphere; y = +1 with probability σ(<w_true, x>)."""
    z = rng.standard_normal((n, w_true.size))
    x = z / np.linalg.norm(z, axis=1, keepdims=True)
    y = np.where(rng.random(n) < sigmoid(x @ w_true), 1.0, -1.0)
    return Data(x, y)


def ridge_risk(data: Data, w: Array, lam: float) -> float:
    """F(w) = (1/n) Σ ln(1 + e^{-y<w,x>}) + (λ/2)‖w‖²."""
    return float(np.mean(np.logaddexp(0.0, -data.y * (data.x @ w)))) + 0.5 * lam * float(w @ w)


def ridge_grad(data: Data, w: Array, lam: float) -> Array:
    margins = data.y * (data.x @ w)
    return data.x.T @ (-data.y * sigmoid(-margins)) / data.y.size + lam * w


def ridge_hessian(data: Data, w: Array, lam: float) -> Array:
    """(1/n) Σ σ(m)(1 - σ(m)) x xᵀ + λ I."""
    s = sigmoid(data.y * (data.x @ w))
    return (data.x.T * (s * (1 - s))) @ data.x / data.y.size + lam * np.eye(data.x.shape[1])


def ridge_minimizer(data: Data, lam: float, tol: float = 1e-13) -> Array:
    """Newton's method with backtracking."""
    w = np.zeros(data.x.shape[1])
    for _ in range(100):
        g = ridge_grad(data, w, lam)
        if float(np.linalg.norm(g)) < tol:
            break
        step = np.linalg.solve(ridge_hessian(data, w, lam), g)
        f0, t = ridge_risk(data, w, lam), 1.0
        while ridge_risk(data, w - t * step, lam) > f0 - 0.25 * t * float(g @ step) and t > 1e-10:
            t /= 2
        w = w - t * step
    return w


def ridge_smoothness(lam: float) -> float:
    """β = 1/4 + λ: the logistic curvature is at most 1/4 along x, ‖x‖ = 1, and the ridge adds λ."""
    return 0.25 + lam


def strong_convexity_ratio(rng: np.random.Generator, data: Data, lam: float, pairs: int, radius: float) -> float:
    """Smallest (F(w') - F(w) - <∇F(w), w' - w>) / (½‖w' - w‖²) over random pairs in a ball: at least λ."""
    smallest = math.inf
    for _ in range(pairs):
        w, w2 = rng.uniform(-1, 1, (2, data.x.shape[1])) * radius / math.sqrt(data.x.shape[1])
        gap = ridge_risk(data, w2, lam) - ridge_risk(data, w, lam) - float(ridge_grad(data, w, lam) @ (w2 - w))
        smallest = min(smallest, gap / (0.5 * float((w2 - w) @ (w2 - w))))
    return smallest


@dataclass(frozen=True)
class GdRun:
    gaps: Array  # F(w_t) - F*, t = 0..steps
    dists: Array  # ‖w_t - w*‖


def gd_run(data: Data, lam: float, steps: int) -> GdRun:
    """Gradient descent with step 1/β from w_0 = 0."""
    w_star = ridge_minimizer(data, lam)
    f_star = ridge_risk(data, w_star, lam)
    eta = 1 / ridge_smoothness(lam)
    w = np.zeros(data.x.shape[1])
    gaps: list[float] = []
    dists: list[float] = []
    for _ in range(steps + 1):
        gaps.append(ridge_risk(data, w, lam) - f_star)
        dists.append(float(np.linalg.norm(w - w_star)))
        w = w - eta * ridge_grad(data, w, lam)
    return GdRun(np.array(gaps), np.array(dists))


def pl_bound(gap0: float, kappa: float, t: int) -> float:
    """F(w_t) - F* ≤ (1 - 1/κ)^t (F(w_0) - F*), from the PL inequality and the per-step decrease."""
    return (1 - 1 / kappa) ** t * gap0


def distance_gap_bound(beta: float, dist0: float, kappa: float, t: int) -> float:
    """F(w_t) - F* ≤ (β/2)‖w_t - w*‖² ≤ (β/2)(1 - 1/κ)^{2t}‖w_0 - w*‖²."""
    return 0.5 * beta * (1 - 1 / kappa) ** (2 * t) * dist0**2


def logistic_ridge_1d(w: Array, lam: float) -> Array:
    """The one-dimensional function of the convexity figure: ln(1 + e^{-w}) + (λ/2)w²."""
    return np.logaddexp(0.0, -w) + 0.5 * lam * w * w


def logistic_ridge_1d_slope(w: float, lam: float) -> float:
    return -sigmoid_scalar(-w) + lam * w


# --- 2. Co-coercivity ---------------------------------------------------------------------------

RHO_LOGISTIC = 0.25  # the 1-D logistic loss ln(1 + e^{-w}) has f'' = σ(1 - σ) ≤ 1/4


def logistic_slope_1d(w: Array) -> Array:
    """f'(w) for f(w) = ln(1 + e^{-w})."""
    return -sigmoid(-w)


def cocoercive_sides(u: Array, v: Array, rho: float) -> tuple[Array, Array]:
    """(f'(u) - f'(v))(u - v) and (f'(u) - f'(v))²/ρ for the 1-D logistic loss."""
    diff = logistic_slope_1d(u) - logistic_slope_1d(v)
    return diff * (u - v), diff * diff / rho


def step_ratio(u: Array, v: Array, eta: float) -> Array:
    """|ψ(u) - ψ(v)| / |u - v| for the gradient step ψ(w) = w - η f'(w) on the logistic loss."""
    pu = u - eta * logistic_slope_1d(u)
    pv = v - eta * logistic_slope_1d(v)
    return np.abs(pu - pv) / np.abs(u - v)


def cocoercive_ratio_risk(rng: np.random.Generator, data: Data, pairs: int, radius: float) -> float:
    """Smallest <Δ, u - v> / (‖Δ‖²/ρ) for the d-dimensional logistic risk (ρ = 1/4): at least 1."""
    smallest = math.inf
    for _ in range(pairs):
        u, v = rng.uniform(-1, 1, (2, data.x.shape[1])) * radius
        delta = ridge_grad(data, u, 0.0) - ridge_grad(data, v, 0.0)
        smallest = min(smallest, float(delta @ (u - v)) / (float(delta @ delta) / RHO_LOGISTIC))
    return smallest


# --- 3. The lower bound for convex Lipschitz optimization -----------------------------------------


def hard_constants(radius: float, lipschitz: float, k: int) -> tuple[float, float]:
    """γ = L√k/(1 + √k), α = L/(R(1 + √k)): Lipschitz constant L on the ball, minimizer at distance R."""
    root = math.sqrt(k)
    return lipschitz * root / (1 + root), lipschitz / (radius * (1 + root))


def hard_value(x: Array, gamma: float, alpha: float, k: int) -> float:
    """f(x) = γ max_{i ≤ k} x_i + (α/2)‖x‖²."""
    return gamma * float(np.max(x[:k])) + 0.5 * alpha * float(x @ x)


def hard_minimum(gamma: float, alpha: float, k: int) -> float:
    """f* = -γ²/(2αk), reached at x_i = -γ/(αk) for i ≤ k."""
    return -(gamma**2) / (2 * alpha * k)


def lower_bound(radius: float, lipschitz: float, steps: int) -> float:
    """RL/(2(1 + √(T + 1))): no span-restricted method beats it after T gradient queries."""
    return radius * lipschitz / (2 * (1 + math.sqrt(steps + 1)))


def upper_bound(radius: float, lipschitz: float, steps: int) -> float:
    """RL/√T: the SGD guarantee of the optimization chapter."""
    return radius * lipschitz / math.sqrt(steps)


@dataclass(frozen=True)
class HardRun:
    averaged_gap: float  # f(w̄_T) - f*
    smallest_value: float  # min_t f(w_t), never below 0


def subgradient_on_hard(radius: float, lipschitz: float, steps: int) -> HardRun:
    """Projected subgradient method with step R/(L√T) against the resisting oracle, from 0.

    The oracle returns γ e_i + αx with i the first index of the largest of x_1..x_k, k = T + 1.
    """
    k = steps + 1
    gamma, alpha = hard_constants(radius, lipschitz, k)
    eta = radius / (lipschitz * math.sqrt(steps))
    x = np.zeros(k)
    total = np.zeros(k)
    smallest = math.inf
    for _ in range(steps):
        total += x
        smallest = min(smallest, hard_value(x, gamma, alpha, k))
        g = alpha * x
        g[int(np.argmax(x))] += gamma
        x = x - eta * g
        norm = float(np.linalg.norm(x))
        if norm > radius:
            x *= radius / norm
    avg = total / steps
    return HardRun(hard_value(avg, gamma, alpha, k) - hard_minimum(gamma, alpha, k), smallest)


# --- 4. The 1-nearest-neighbour limit -------------------------------------------------------------
# The running example of the stability chapter: x uniform on the unit sphere in R^10, and
# Pr[y = 1 | x] = σ(5 x_1).

NN_DIM = 10
NN_SIGNAL = 5.0


def sphere_angle_grid(d: int = NN_DIM, m: int = 20_001) -> tuple[Array, Array]:
    """Angles φ in [0, π] and weights for E[g(x_1)] = Σ g(cos φ) weight, x uniform on the sphere.

    The angle between x and a fixed unit vector has density proportional to sin^{d-2} φ.
    """
    phi = np.linspace(0.0, math.pi, m)
    dens = np.sin(phi) ** (d - 2)
    weight = dens / dens.sum()
    return phi, weight


def bayes_and_limit(d: int = NN_DIM, signal: float = NN_SIGNAL) -> tuple[float, float]:
    """Bayes error E[min(p, 1 - p)] and the 1-NN limit E[2p(1 - p)], p = σ(signal · x_1)."""
    phi, weight = sphere_angle_grid(d)
    p = sigmoid(signal * np.cos(phi))
    return float(np.minimum(p, 1 - p) @ weight), float((2 * p * (1 - p)) @ weight)


def cap_probability(r: float, d: int = NN_DIM, m: int = 4001) -> float:
    """Pr[‖x - x0‖ ≤ r] for x uniform on the unit sphere in R^d and a fixed unit x0.

    Distance r is the angle a = 2 arcsin(r/2), and the angle has density sin^{d-2} φ / ∫_0^π sin^{d-2}.
    """
    a = 2 * math.asin(min(r / 2, 1.0))
    phi = np.linspace(0.0, a, m)
    vals = np.sin(phi) ** (d - 2)
    part = float(np.sum((vals[1:] + vals[:-1]) / 2) * (a / (m - 1)))
    whole = math.sqrt(math.pi) * math.exp(math.lgamma((d - 1) / 2) - math.lgamma(d / 2))
    return part / whole


def median_nn_distance(n: int, d: int = NN_DIM) -> float:
    """The r with Pr[no training point within r] = (1 - cap(r))^n = 1/2, by bisection."""
    target = -math.expm1(-math.log(2) / n)  # the cap probability that solves the equation
    lo, hi = 0.0, 2.0
    for _ in range(80):
        mid = (lo + hi) / 2
        if cap_probability(mid, d) < target:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def sample_sphere(rng: np.random.Generator, n: int, d: int = NN_DIM) -> tuple[Array, Array, Array]:
    """Points uniform on the sphere, labels with Pr[y = 1 | x] = σ(5 x_1), and p(x)."""
    x = rng.standard_normal((n, d))
    x /= np.linalg.norm(x, axis=1, keepdims=True)
    p = sigmoid(NN_SIGNAL * x[:, 0])
    y = np.where(rng.random(n) < p, 1.0, -1.0)
    return x, y, p


@dataclass(frozen=True)
class NnResult:
    error: float  # measured 0-1 test error of 1-NN
    plug_in: float  # average of p(x)(1 - p(x')) + (1 - p(x))p(x'), x' the nearest neighbour
    median_distance: float  # median distance from a test point to its nearest neighbour


def one_nn(rng: np.random.Generator, n: int, n_test: int, block: int = 500) -> NnResult:
    x, y, p = sample_sphere(rng, n)
    xt, yt, pt = sample_sphere(rng, n_test)
    nearest = np.empty(n_test, dtype=np.int64)
    for start in range(0, n_test, block):
        nearest[start : start + block] = np.argmax(xt[start : start + block] @ x.T, axis=1)
    dist = np.sqrt(np.maximum(2 - 2 * np.einsum("ij,ij->i", xt, x[nearest]), 0.0))
    q = p[nearest]
    return NnResult(
        np.count_nonzero(y[nearest] != yt) / n_test,
        float(np.mean(pt * (1 - q) + (1 - pt) * q)),
        float(np.median(dist)),
    )


# --- 5. M-estimators: fitted parameters are approximately Gaussian --------------------------------
# The logistic-regression toy of the property-inference chapter: x ~ N(0, I_2), majority labels
# with σ(2x_1 + x_2), the subgroup (share t) with σ(2x_1 - x_2); the model has an intercept and a
# ridge penalty 10^-3. Parameters are θ = (w_1, w_2, intercept).

W_MAJORITY = np.array([2.0, 1.0, 0.0])
W_SUBGROUP = np.array([2.0, -1.0, 0.0])
RIDGE = 1e-3
GH_POINTS = 40
HESSIAN_LIPSCHITZ = 1 / (6 * math.sqrt(3))  # max |d/dm σ(m)(1 - σ(m))|


def sigmoid_curvature_slope_max(m: int = 200_001) -> float:
    """Largest |d/dm [σ(m)(1 - σ(m))]| = |σ(1 - σ)(1 - 2σ)| on a fine grid."""
    s = sigmoid(np.linspace(-10, 10, m))
    return float(np.max(np.abs(s * (1 - s) * (1 - 2 * s))))


def sample_datasets(rng: np.random.Generator, models: int, n: int, t: float) -> tuple[Array, Array]:
    """`models` datasets of n records: design (models, n, 3) with rows (x_1, x_2, 1), labels 0/1."""
    x = rng.standard_normal((models, n, 2))
    group = rng.random((models, n)) < t
    design = np.concatenate([x, np.ones((models, n, 1))], axis=2)
    logits = np.where(group, design @ W_SUBGROUP, design @ W_MAJORITY)
    y = (rng.random((models, n)) < sigmoid(logits)).astype(np.float64)
    return design, y


def train_logistic(design: Array, y: Array, l2: float = RIDGE, steps: int = 15) -> Array:
    """Many logistic regressions at once by Newton's method; returns parameters (models, 3).

    Minimizes the mean log-loss plus (l2/2)‖θ‖², so every problem is strongly convex.
    """
    models, n, d = design.shape
    cols = [design[:, :, i] for i in range(d)]
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


def quadrature_grid() -> tuple[Array, Array]:
    """Nodes (k, 3), rows (x_1, x_2, 1), and weights (k,) for E[g(x)] with x ~ N(0, I_2)."""
    nodes, weights = np.polynomial.hermite_e.hermegauss(GH_POINTS)
    weights = weights / weights.sum()
    x1, x2 = np.meshgrid(nodes, nodes, indexing="ij")
    w = np.outer(weights, weights).ravel()
    return np.stack([x1.ravel(), x2.ravel(), np.ones(x1.size)], axis=1), w


def mean_label(points: Array, t: float) -> Array:
    """E[y | x] = (1 - t) σ(w_A·x) + t σ(w_B·x)."""
    return (1 - t) * sigmoid(points @ W_MAJORITY) + t * sigmoid(points @ W_SUBGROUP)


@dataclass(frozen=True)
class Sandwich:
    theta: Array  # population fit θ*
    h: Array  # H = E ∇²ℓ + λI
    g: Array  # G = covariance of the per-record gradient at θ*
    sigma: Array  # Σ₁ = H⁻¹ G H⁻¹


def sandwich(t: float, lam: float = 0.0, steps: int = 40) -> Sandwich:
    """Population fit of E[log-loss] + (λ/2)‖θ‖² and its sandwich covariance, by quadrature."""
    points, weights = quadrature_grid()
    m = mean_label(points, t)
    theta = np.zeros(3)
    for _ in range(steps):
        p = sigmoid(points @ theta)
        grad = points.T @ (weights * (p - m)) + lam * theta
        hess = points.T @ (points * (weights * p * (1 - p))[:, None]) + lam * np.eye(3)
        theta = theta - np.linalg.solve(hess, grad)
    p = sigmoid(points @ theta)
    h = points.T @ (points * (weights * p * (1 - p))[:, None]) + lam * np.eye(3)
    # E[(y - p)² | x] = m(1 - m) + (m - p)²; subtracting the squared mean λ²θθᵀ gives the covariance.
    g = points.T @ (points * (weights * (m * (1 - m) + (m - p) ** 2))[:, None]) - lam**2 * np.outer(theta, theta)
    h_inv = np.linalg.inv(h)
    return Sandwich(theta, h, g, h_inv @ g @ h_inv)


@dataclass(frozen=True)
class Refits:
    theta: Array  # (models, 3) fitted parameters
    max_grad: float  # largest gradient norm left by Newton's method


def refits(rng: np.random.Generator, models: int, n: int, t: float) -> Refits:
    design, y = sample_datasets(rng, models, n, t)
    theta = train_logistic(design, y)
    p = sigmoid(np.einsum("mnd,md->mn", design, theta))
    grad: Array = np.einsum("mn,mnd->md", p - y, design) / n + RIDGE * theta
    return Refits(theta, float(np.max(np.linalg.norm(grad, axis=1))))


# The influence of one record, on the running example of the membership-inference chapter:
# 950 typical points x ~ N(0, I/100), y = ±1 with Pr[y = 1 | x] = σ(30 x_1); 50 outliers with
# 5 times longer x and a fair-coin label; λ = 10^-3; a model trains on n = 500 points.

MIA_D = 100
MIA_POOL = 1000
MIA_LAM = 1e-3


@dataclass(frozen=True)
class Pool:
    x: Array
    y: Array
    outlier: NDArray[np.bool_]


def sample_pool(rng: np.random.Generator) -> Pool:
    x = rng.standard_normal((MIA_POOL, MIA_D)) / math.sqrt(MIA_D)
    outlier = np.zeros(MIA_POOL, dtype=np.bool_)
    outlier[: MIA_POOL // 20] = True
    x[outlier] *= 5.0
    p = sigmoid(3.0 * math.sqrt(MIA_D) * x[:, 0])
    y = np.where(rng.random(MIA_POOL) < p, 1.0, -1.0)
    y[outlier] = np.where(rng.random(int(outlier.sum())) < 0.5, 1.0, -1.0)
    return Pool(x, y, outlier)


def fit_margin_model(x: Array, y: Array, lam: float, tol: float = 1e-11) -> Array:
    """argmin_w (1/n) Σ ln(1 + e^{-y<w,x>}) + (λ/2)‖w‖², by damped Newton."""
    data = Data(x, y)
    w = np.zeros(x.shape[1])
    for _ in range(100):
        g = ridge_grad(data, w, lam)
        if float(np.max(np.abs(g))) < tol:
            break
        step = np.linalg.solve(ridge_hessian(data, w, lam), g)
        f0, t = ridge_risk(data, w, lam), 1.0
        while ridge_risk(data, w - t * step, lam) > f0 - 0.25 * t * float(g @ step) and t > 1e-10:
            t /= 2
        w = w - t * step
    return w


@dataclass(frozen=True)
class Influence:
    measured: Array  # φ_z(θ_S) - φ_z(θ_{S - z}) for each member z
    predicted: Array  # (1/n) σ(-φ_z) x_zᵀ H⁻¹ x_z, everything at the fit without z
    outlier: NDArray[np.bool_]


def influence_check(rng: np.random.Generator, members: int) -> Influence:
    """Leave each of `members` training points out, refit, and compare with the first-order formula.

    Removing z's term from the sum (keeping the 1/n weights) changes the objective by exactly
    (1/n)ℓ(θ, z), which is the setting of the derivation.
    """
    pool = sample_pool(rng)
    train = rng.permutation(MIA_POOL)[: MIA_POOL // 2]
    x, y = pool.x[train], pool.y[train]
    n = len(train)
    w_full = fit_margin_model(x, y, MIA_LAM)
    chosen = rng.permutation(n)[:members]
    measured: list[float] = []
    predicted: list[float] = []
    for i in chosen:
        keep = np.arange(n) != i
        # The fit without z's term: (1/n) Σ_{j≠i} ℓ_j + (λ/2)‖w‖², i.e. n - 1 records with ridge λ n/(n - 1).
        w_out = fit_margin_model(x[keep], y[keep], MIA_LAM * n / (n - 1))
        phi_out = float(y[i] * (x[i] @ w_out))
        # Hessian of the full objective (z's term included) at the fit without z.
        s_out = sigmoid_scalar(phi_out)
        h = ridge_hessian(Data(x[keep], y[keep]), w_out, MIA_LAM * n / (n - 1)) * (n - 1) / n
        h += s_out * (1 - s_out) * np.outer(x[i], x[i]) / n
        lever = float(x[i] @ np.linalg.solve(h, x[i]))
        measured.append(float(y[i] * (x[i] @ w_full)) - phi_out)
        predicted.append(sigmoid_scalar(-phi_out) * lever / n)
    return Influence(np.array(measured), np.array(predicted), pool.outlier[train][chosen])


# --- 6. Bayes-optimal membership inference under a Gibbs posterior --------------------------------


def log_sigmoid(s: Array) -> Array:
    return -np.logaddexp(0.0, -s)


@dataclass(frozen=True)
class GibbsCheck:
    identity_error: float  # max |exact posterior - E[σ(...)] formula|
    approx_error: float  # max |exact posterior - σ(-(ℓ - τ̄)/T + log odds)|


def gibbs_check(rng: np.random.Generator, n: int = 10, temp: float = 1.0, prior: float = 0.5) -> GibbsCheck:
    """Exact Bayes posterior of m_1 on a grid of θ, against Sablayrolles et al.'s formula.

    Candidates z_1..z_n ~ N(0, 1), membership m_i ~ Bernoulli(prior), loss ½(θ - z)², and the
    posterior p(θ | m) ∝ exp(-(1/T) Σ m_i ℓ(θ, z_i)) on a grid (a flat prior on the grid).
    """
    z = rng.standard_normal(n)
    grid = np.linspace(-4.0, 4.0, 801)
    loss: Array = 0.5 * (grid[None, :] - z[:, None]) ** 2  # (n, grid)
    masks: Array = np.array(list(itertools.product((0.0, 1.0), repeat=n - 1)))  # memberships of z_2..z_n
    energy_rest = masks @ loss[1:] / temp  # (2^{n-1}, grid)
    log_prior_rest = masks.sum(axis=1) * math.log(prior) + (n - 1 - masks.sum(axis=1)) * math.log(1 - prior)

    def log_normalize(energy: Array) -> Array:
        top = np.max(-energy, axis=1, keepdims=True)
        return -energy - (top + np.log(np.sum(np.exp(-energy - top), axis=1, keepdims=True)))

    log_post_out = log_normalize(energy_rest)  # log p(θ | m_1 = 0, m_rest)
    log_post_in = log_normalize(energy_rest + loss[0][None, :] / temp)  # log p(θ | m_1 = 1, m_rest)
    joint_in = log_prior_rest[:, None] + math.log(prior) + log_post_in
    joint_out = log_prior_rest[:, None] + math.log(1 - prior) + log_post_out
    both: Array = np.logaddexp(joint_in, joint_out)
    log_total = np.log(np.sum(np.exp(both - both.max(axis=0)), axis=0)) + both.max(axis=0)
    exact = np.exp(np.log(np.sum(np.exp(joint_in - both.max(axis=0)), axis=0)) + both.max(axis=0) - log_total)

    # τ(z_1, m_rest) = -T ln E_{θ' ~ p(· | m_1 = 0, m_rest)} e^{-ℓ(θ', z_1)/T}
    tilted: Array = np.exp(log_post_out - loss[0][None, :] / temp)
    tau: Array = -temp * np.log(tilted.sum(axis=1))
    log_odds = math.log(prior / (1 - prior))
    inner = sigmoid(-(loss[0][None, :] - tau[:, None]) / temp + log_odds)  # (2^{n-1}, grid)
    weight_rest = np.exp(both - log_total[None, :])  # Pr[m_rest | θ]
    formula = np.sum(weight_rest * inner, axis=0)

    prior_rest = np.exp(log_prior_rest)
    tau_bar = -temp * math.log(float(prior_rest @ np.exp(-tau / temp)))
    approx = sigmoid(-(loss[0] - tau_bar) / temp + log_odds)
    # Compare where θ is plausible: grid points carrying 99.9% of the marginal mass of θ.
    marginal = np.exp(log_total - log_total.max())
    order = np.argsort(marginal)[::-1]
    keep = order[: int(np.searchsorted(np.cumsum(marginal[order]) / marginal.sum(), 0.999)) + 1]
    return GibbsCheck(
        float(np.max(np.abs(exact - formula))),
        float(np.max(np.abs(exact[keep] - approx[keep]))),
    )


# The centroid detector of the threat-models chapter: θ is the mean of n records x_i ~ N(0, I_d).

CENTROID_N = 100
CENTROID_D = 50


@dataclass(frozen=True)
class CentroidGames:
    theta: Array  # (games, d)
    z: Array  # (games, d)
    member: NDArray[np.bool_]


def centroid_games(rng: np.random.Generator, games: int, n: int = CENTROID_N, d: int = CENTROID_D) -> CentroidGames:
    """`games` member games and `games` non-member games."""
    z = rng.standard_normal((2 * games, d))
    rest = rng.standard_normal((games, d)) * math.sqrt(n - 1)
    theta_in = (z[:games] + rest) / n
    theta_out = rng.standard_normal((games, d)) / math.sqrt(n)
    member = np.zeros(2 * games, dtype=np.bool_)
    member[:games] = True
    return CentroidGames(np.concatenate([theta_in, theta_out]), z, member)


def centroid_log_lr_direct(theta: Array, z: Array, n: int = CENTROID_N) -> Array:
    """ln N(θ; z/n, (n-1)/n² I) - ln N(θ; 0, I/n), straight from the two densities."""
    d = theta.shape[1]
    var_in, var_out = (n - 1) / n**2, 1 / n
    log_in = -np.sum((theta - z / n) ** 2, axis=1) / (2 * var_in) - 0.5 * d * math.log(var_in)
    log_out = -np.sum(theta**2, axis=1) / (2 * var_out) - 0.5 * d * math.log(var_out)
    return log_in - log_out


def centroid_loss(theta: Array, z: Array) -> Array:
    """ℓ(θ, z) = ½‖θ - z‖²."""
    return 0.5 * np.sum((theta - z) ** 2, axis=1)


def centroid_tau(z: Array, n: int = CENTROID_N) -> Array:
    """τ(z) = (n - 1)‖z‖²/(2n), the per-target calibration."""
    return (n - 1) * np.sum(z * z, axis=1) / (2 * n)


def centroid_log_lr_formula(theta: Array, z: Array, n: int = CENTROID_N) -> Array:
    """-(n/(n-1)) [ℓ(θ, z) - τ(z)] + (d/2) ln(n/(n-1))."""
    d = theta.shape[1]
    return -(n / (n - 1)) * (centroid_loss(theta, z) - centroid_tau(z, n)) + 0.5 * d * math.log(n / (n - 1))


def best_advantage(scores: Array, member: NDArray[np.bool_]) -> float:
    """max over thresholds of TPR - FPR for 'member when the score is above the threshold'."""
    order = np.argsort(-scores, kind="stable")
    hits = member[order]
    tpr = np.cumsum(hits) / hits.sum()
    fpr = np.cumsum(~hits) / (~hits).sum()
    return float(np.max(tpr - fpr))


# --- The appendix's table ---------------------------------------------------------------------


def print_table(rng: np.random.Generator) -> None:
    # 1. Convexity and linear convergence.
    data = make_logistic_data(rng, GD_N)
    beta = ridge_smoothness(GD_LAM)
    kappa = beta / GD_LAM
    ratio = strong_convexity_ratio(rng, data, GD_LAM, pairs=2000, radius=5.0)
    print(f"[Strong convexity, λ={GD_LAM}] smallest tangent gap / (½‖w'-w‖²) over 2000 pairs = {ratio:.4f}")
    run = gd_run(data, GD_LAM, steps=300)
    w_star = ridge_minimizer(data, GD_LAM)
    eig = np.linalg.eigvalsh(ridge_hessian(data, w_star, GD_LAM))
    print(
        f"[GD, λ={GD_LAM}] β={beta:.2f} κ={kappa:.1f} 1-1/κ={1 - 1 / kappa:.4f} "
        f"Hessian eigenvalues at w*: {eig[0]:.4f}..{eig[-1]:.4f}"
    )
    early = slice(0, 41)  # before the gap reaches machine precision
    gaps, dists = run.gaps[early], run.dists[early]
    fun_ratio = float(np.max(gaps[1:] / gaps[:-1]))
    dist_ratio = float(np.max(dists[1:] / dists[:-1]))
    observed = float((gaps[40] / gaps[20]) ** (1 / 20))
    local = (1 - float(eig[0]) / beta) ** 2
    print(
        f"[GD, steps 1..40] largest per-step ratio: gap {fun_ratio:.4f}, distance {dist_ratio:.4f}; "
        f"typical gap ratio (steps 20..40) {observed:.3f}; (1 - λ_min(w*)/β)² = {local:.3f}"
    )
    gap0, dist0 = float(run.gaps[0]), float(run.dists[0])
    print(f"[GD] F(w_0)-F*={gap0:.4f} ‖w_0-w*‖={dist0:.4f}")
    for t in (25, 50):
        print(
            f"[GD t={t}] gap={run.gaps[t]:.2e}  PL bound={pl_bound(gap0, kappa, t):.2e}  "
            f"distance bound={distance_gap_bound(beta, dist0, kappa, t):.2e}"
        )

    # 2. Co-coercivity.
    u0, v0 = np.array([0.0]), np.array([2.0])
    lhs0, rhs0 = cocoercive_sides(u0, v0, RHO_LOGISTIC)
    print(
        f"[Co-coercivity u=0 v=2] f'(0)={float(logistic_slope_1d(u0)[0]):.4f} "
        f"f'(2)={float(logistic_slope_1d(v0)[0]):.4f} lhs={float(lhs0[0]):.4f} rhs={float(rhs0[0]):.4f}"
    )
    u, v = rng.normal(0.0, 4.0, (2, 500))
    lhs, rhs = cocoercive_sides(u, v, RHO_LOGISTIC)
    print(f"[Co-coercivity, 500 pairs] smallest lhs/rhs={float(np.min(lhs / rhs)):.4f}")
    print(
        f"[Co-coercivity, logistic risk d=5] smallest ratio over 2000 pairs={cocoercive_ratio_risk(rng, data, 2000, 3.0):.4f}"
    )
    for eta in (4.0, 8.0, 10.0):
        print(
            f"[Gradient step η={eta:g}] largest distance ratio over the pairs={float(np.max(step_ratio(u, v, eta))):.4f}"
        )

    # 3. Lower bound.
    for t in (10, 100, 1000, 10_000):
        hard = subgradient_on_hard(1.0, 1.0, t)
        print(
            f"[Lower bound T={t}] lower={lower_bound(1.0, 1.0, t):.4f}  "
            f"subgradient method={hard.averaged_gap:.4f}  upper={upper_bound(1.0, 1.0, t):.4f}  "
            f"min f(w_t)={hard.smallest_value:.4f}"
        )
    print(f"[Lower bound T=10000] method / lower bound = {hard.averaged_gap / lower_bound(1.0, 1.0, 10_000):.2f}")
    gamma, alpha = hard_constants(1.0, 1.0, 3)
    print(f"[Hard function k=3, R=L=1] γ={gamma:.4f} α={alpha:.4f} f*={hard_minimum(gamma, alpha, 3):.4f}")

    # 4. 1-NN.
    bayes, limit = bayes_and_limit()
    print(f"[1-NN] Bayes error={bayes:.3f}  limit E[2p(1-p)]={limit:.3f}")
    for n in (100, 4000, 10**6, 10**9):
        print(f"[1-NN n={n}] exact median nearest-neighbour distance={median_nn_distance(n):.3f}")
    for n in (100, 4000):
        res = one_nn(rng, n, n_test=4000)
        print(
            f"[1-NN n={n}, 4000 test points] error={res.error:.3f}  plug-in={res.plug_in:.3f}  "
            f"median distance={res.median_distance:.3f}"
        )

    # 5. M-estimators.
    print(
        f"[Hessian Lipschitz] max |σ(1-σ)(1-2σ)|={sigmoid_curvature_slope_max():.5f}  1/(6√3)={HESSIAN_LIPSCHITZ:.5f}"
    )
    for lam in (0.0, RIDGE):
        s = sandwich(0.5, lam)
        print(
            f"[Sandwich t=0.5 λ={lam:g}] θ*=({s.theta[0]:.3f}, {s.theta[1]:.3f}, {s.theta[2]:.3f})  "
            f"Σ₁ diag=({s.sigma[0, 0]:.2f}, {s.sigma[1, 1]:.2f}, {s.sigma[2, 2]:.2f})  "
            f"largest off-diagonal={float(np.max(np.abs(s.sigma - np.diag(np.diag(s.sigma))))):.1e}"
        )
    s0 = sandwich(0.0)
    print(
        f"[Well specified t=0] θ*=({s0.theta[0]:.3f}, {s0.theta[1]:.3f}, {s0.theta[2]:.3f})  "
        f"max |G - H| below 1e-12: {float(np.max(np.abs(s0.g - s0.h))) < 1e-12}"
    )
    mis = sandwich(0.5)
    s_g, s_h = np.diag(mis.g), np.diag(mis.h)
    print(
        f"[Misspecified t=0.5] G diag=({s_g[0]:.3f}, {s_g[1]:.3f}, {s_g[2]:.3f})  H diag=({s_h[0]:.3f}, {s_h[1]:.3f}, {s_h[2]:.3f})"
    )
    half = sandwich(0.5, RIDGE)
    for n in (50, 200, 1000):
        fits = refits(rng, 2000, n, 0.5)
        nvar = n * np.var(fits.theta, axis=0)
        scaled = math.sqrt(n) * fits.theta[:, 1] / math.sqrt(float(half.sigma[1, 1]))
        inside = float(np.mean(np.abs(scaled) <= 1.96))
        print(
            f"[Refits n={n}, 2000 models] n·Var=({nvar[0]:.2f}, {nvar[1]:.2f}, {nvar[2]:.2f})  "
            f"mean w1={float(np.mean(fits.theta[:, 0])):.3f}  within ±1.96 sd: {inside:.3f}  max grad={fits.max_grad:.0e}"
        )
    inf = influence_check(rng, members=200)
    for name, mask in (("typical", ~inf.outlier), ("outliers", inf.outlier)):
        m, p = inf.measured[mask], inf.predicted[mask]
        print(
            f"[Influence, {name}: {int(mask.sum())} points] mean measured={float(np.mean(m)):.4f}  "
            f"mean predicted={float(np.mean(p)):.4f}  median measured/predicted={float(np.median(m / p)):.3f}"
        )

    # 6. Gibbs posterior and the centroid.
    for n in (10,):
        g = gibbs_check(rng, n=n)
        print(
            f"[Gibbs n={n}] identity error below 1e-12: {g.identity_error < 1e-12}  "
            f"error of the τ approximation={g.approx_error:.3f}"
        )
    games = centroid_games(rng, 100_000)
    direct = centroid_log_lr_direct(games.theta, games.z)
    formula = centroid_log_lr_formula(games.theta, games.z)
    calibrated = np.sum(games.z**2, axis=1) - np.sum((games.z - games.theta) ** 2, axis=1)
    dropped = np.sum(games.z**2, axis=1) / (2 * (CENTROID_N - 1))
    print(f"[Centroid] max |direct - formula| log-LR below 1e-10: {float(np.max(np.abs(direct - formula))) < 1e-10}")
    print(
        f"[Centroid, 100000 games per world] advantage: LR={best_advantage(direct, games.member):.4f}  "
        f"calibrated={best_advantage(calibrated, games.member):.4f}  "
        f"raw loss={best_advantage(-centroid_loss(games.theta, games.z), games.member):.4f}"
    )
    print(
        f"[Centroid] sd of log-LR={float(np.std(direct)):.3f}  sd of dropped term ‖z‖²/(2(n-1))={float(np.std(dropped)):.4f}"
    )


# --- Figures --------------------------------------------------------------------------------


def plot_convexity(lam: float = 0.2) -> Figure:
    ws = np.linspace(-4.0, 4.0, 400)
    fs = logistic_ridge_1d(ws, lam)
    fig, (left, right) = plt.subplots(1, 2, figsize=(10, 4))

    a, b, mix = -3.0, 3.0, 0.3
    z = mix * b + (1 - mix) * a
    fa, fb = float(logistic_ridge_1d(np.array([a]), lam)[0]), float(logistic_ridge_1d(np.array([b]), lam)[0])
    fz = float(logistic_ridge_1d(np.array([z]), lam)[0])
    slope = logistic_ridge_1d_slope(z, lam)
    tangent = fz + slope * (ws - z)
    left.plot(ws, fs, "k", linewidth=2, label="f")
    left.plot([a, b], [fa, fb], color="C0", label="chord")
    left.plot(ws, tangent, color="C1", linestyle="--", label="tangent at z")
    for p, fp in ((a, fa), (b, fb)):
        left.plot([p, p], [fz + slope * (p - z), fp], color="C1", linewidth=4, alpha=0.5)
    chord_z = mix * fb + (1 - mix) * fa
    left.plot([z, z], [fz, chord_z], color="C0", linewidth=4, alpha=0.5)
    left.plot([a, z, b], [fa, fz, fb], "ko", markersize=4)
    for p, text in ((a, "θ"), (z, "z"), (b, "θ′")):
        left.annotate(text, (p, -0.35), ha="center")
    left.set(
        ylim=(-0.5, 5.0),
        xlabel="θ",
        ylabel="value",
        title="Gaps at θ and θ′ (orange) average to the gap at z (blue)",
    )
    left.legend(loc="upper center")

    w0 = -1.0
    f0 = float(logistic_ridge_1d(np.array([w0]), lam)[0])
    s0 = logistic_ridge_1d_slope(w0, lam)
    right.plot(ws, fs, "k", linewidth=2, label="f")
    right.plot(ws, f0 + s0 * (ws - w0), color="C1", linestyle="--", label="tangent at θ")
    bowl = f0 + s0 * (ws - w0) + 0.5 * lam * (ws - w0) ** 2
    right.plot(ws, bowl, color="C2", label="tangent + (λ/2)(θ′ − θ)²")
    right.fill_between(ws, bowl, fs, color="C2", alpha=0.15)
    right.plot([w0], [f0], "ko", markersize=4)
    right.set(
        ylim=(-0.5, 5.0),
        xlabel="θ′",
        ylabel="value",
        title=f"Curvature at least λ = {lam}: f stays above the bowl",
    )
    right.legend(loc="upper center")
    return fig


def plot_gd_linear(rng: np.random.Generator, steps: int = 80) -> Figure:
    data = make_logistic_data(rng, GD_N)
    beta = ridge_smoothness(GD_LAM)
    kappa = beta / GD_LAM
    run = gd_run(data, GD_LAM, steps)
    ts = list(range(steps + 1))
    gap0, dist0 = float(run.gaps[0]), float(run.dists[0])
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(ts, run.gaps, "k", linewidth=2, label=r"gradient descent, $\mathcal{L}(\theta_t) - \mathcal{L}^*$")
    ax.plot(
        ts,
        [pl_bound(gap0, kappa, t) for t in ts],
        label=r"Polyak–Łojasiewicz bound $(1 - 1/\kappa)^t (\mathcal{L}(\theta_0) - \mathcal{L}^*)$",
    )
    ax.plot(
        ts,
        [distance_gap_bound(beta, dist0, kappa, t) for t in ts],
        label=r"distance bound $\frac{\beta}{2}(1 - 1/\kappa)^{2t}\|\theta_0 - \theta^*\|^2$",
    )
    w_star = ridge_minimizer(data, GD_LAM)
    h_min = float(np.linalg.eigvalsh(ridge_hessian(data, w_star, GD_LAM))[0])
    ax.plot(
        ts,
        [gap0 * (1 - h_min / beta) ** (2 * t) for t in ts],
        color="gray",
        linestyle=":",
        label="same rate with the smallest curvature at θ*",
    )
    ax.set(
        yscale="log",
        ylim=(1e-14, 10),
        xlabel="step t",
        ylabel=r"$\mathcal{L}(\theta_t) - \mathcal{L}^*$",
        title=rf"Regularized logistic regression, $\lambda = {GD_LAM}$, $\kappa = {kappa:.0f}$",
    )
    ax.legend(loc="lower left")
    return fig


def plot_cocoercive(rng: np.random.Generator) -> Figure:
    u, v = rng.normal(0.0, 4.0, (2, 500))
    lhs, rhs = cocoercive_sides(u, v, RHO_LOGISTIC)
    fig, (left, right) = plt.subplots(1, 2, figsize=(10, 4))
    left.scatter(rhs, lhs, s=8, alpha=0.6)
    top = float(max(lhs.max(), rhs.max()))
    left.plot([0, top], [0, top], "k--", label="y = x")
    left.set(
        xlabel="(f′(u) − f′(v))² / β",
        ylabel="(f′(u) − f′(v)) (u − v)",
        title="Logistic loss, β = 1/4: every pair lies above y = x",
    )
    left.legend(loc="upper left")

    etas = np.linspace(0.05, 12.0, 240)
    worst = [float(np.max(step_ratio(u, v, float(e)))) for e in etas]
    right.plot(etas, worst, "k", linewidth=2, label="logistic loss, worst of 500 pairs")
    right.plot(etas, np.abs(1 - etas * RHO_LOGISTIC), color="C1", label="quadratic (β/2)θ²: |1 − ηβ|")
    right.axhline(1.0, color="gray", linestyle=":")
    right.axvline(2 / RHO_LOGISTIC, color="C3", linestyle="--", label="η = 2/β = 8")
    right.set(
        ylim=(0, 2.2),
        xlabel="learning rate η",
        ylabel="|ψ(u) − ψ(v)| / |u − v|",
        title="One gradient step: non-expansive up to 2/β",
    )
    right.legend(loc="upper left")
    return fig


def plot_lower_bound() -> Figure:
    ts = [10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10_000]
    measured = [subgradient_on_hard(1.0, 1.0, t).averaged_gap for t in ts]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(ts, [upper_bound(1.0, 1.0, t) for t in ts], label=r"SGD guarantee $RL/\sqrt{T}$")
    ax.plot(ts, measured, "ko", label="subgradient method on the hard function")
    ax.plot(ts, [lower_bound(1.0, 1.0, t) for t in ts], label=r"lower bound $RL/(2(1 + \sqrt{T + 1}))$")
    ax.set(
        xscale="log",
        yscale="log",
        xlabel="gradient queries T",
        ylabel=r"error $f(\bar\theta_T) - f^*$",
        title="R = L = 1: the method is pinned between the bounds",
    )
    ax.legend()
    return fig


def plot_nn(rng: np.random.Generator) -> Figure:
    ns = [10, 30, 100, 300, 1000, 3000, 10_000, 30_000]
    results = [one_nn(rng, n, n_test=20_000) for n in ns]
    _, limit = bayes_and_limit()
    fine = [round(10 ** (k / 4)) for k in range(4, 37)]
    fig, (left, right) = plt.subplots(1, 2, figsize=(10, 4))
    left.plot(fine, [median_nn_distance(n) for n in fine], label="exact median")
    left.plot(ns, [r.median_distance for r in results], "ko", label="simulated median")
    left.set(
        xscale="log",
        yscale="log",
        xlabel="training points n",
        ylabel="distance to the nearest neighbour",
        title=r"It shrinks to 0, but only like $n^{-1/9}$",
    )
    left.legend()
    right.plot(ns, [r.error for r in results], "ko-", label="1-NN test error")
    right.plot(ns, [r.plug_in for r in results], "s--", color="C0", label="average of p(1 − p′) + (1 − p)p′")
    right.axhline(limit, color="C3", linestyle=":", label=rf"limit $\mathbb{{E}}[2p(1 - p)] = {limit:.3f}$")
    right.set(xscale="log", xlabel="training points n", ylabel="0-1 error", title="The error falls towards the limit")
    right.legend()
    return fig


def normal_pdf(x: Array, var: float) -> Array:
    return np.exp(-x * x / (2 * var)) / math.sqrt(2 * math.pi * var)


def plot_mest_hist(rng: np.random.Generator) -> Figure:
    half = sandwich(0.5, RIDGE)
    var = float(half.sigma[1, 1])
    grid = np.linspace(-10, 10, 400)
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8), sharey=True)
    for ax, n in zip(axes, (50, 200, 1000), strict=True):
        fits = refits(rng, 2000, n, 0.5)
        scaled = math.sqrt(n) * (fits.theta[:, 1] - float(half.theta[1]))
        ax.hist(scaled, bins=np.linspace(-10, 10, 41).tolist(), density=True, alpha=0.6, label="2,000 refits")
        ax.plot(grid, normal_pdf(grid, var), "k", label=f"normal, variance {var:.2f}")
        ax.set(xlabel="√n × (fitted θ₂ − population θ₂)", title=f"n = {n}")
    axes[0].set(ylabel="density")
    axes[0].legend(loc="upper left")
    return fig


def plot_mest_variance(rng: np.random.Generator) -> Figure:
    half = sandwich(0.5, RIDGE)
    ns = [25, 50, 100, 200, 500, 1000, 2000]
    nvars = [n * np.var(refits(rng, 2000, n, 0.5).theta, axis=0) for n in ns]
    names = (r"$\theta_1$", r"$\theta_2$", "intercept")
    inf = influence_check(rng, members=120)
    fig, (left, right) = plt.subplots(1, 2, figsize=(10, 4))
    for k in range(3):
        left.plot(ns, [float(v[k]) for v in nvars], "o-", color=f"C{k}", label=names[k])
        left.axhline(float(half.sigma[k, k]), color=f"C{k}", linestyle="--")
    left.set(
        xscale="log",
        ylim=(0, 18),
        xlabel="training records n",
        ylabel="n × variance over 2,000 refits",
        title=r"Dashed: the sandwich $H^{-1}GH^{-1}$",
    )
    left.legend()
    typ = ~inf.outlier
    right.scatter(inf.predicted[typ], inf.measured[typ], s=10, label="typical points")
    right.scatter(inf.predicted[inf.outlier], inf.measured[inf.outlier], s=24, marker="^", label="outliers")
    top = float(max(inf.predicted.max(), inf.measured.max()))
    right.plot([0.03, top], [0.03, top], "k--", label="y = x")
    right.set(
        xscale="log",
        yscale="log",
        xlabel="predicted (1/n) σ(−φ) xᵀH⁻¹x",
        ylabel="measured change of the margin φ",
        title="Training on z raises its margin as predicted",
    )
    right.legend()
    return fig


def plot_gibbs(rng: np.random.Generator, games: int = 1500) -> Figure:
    g = centroid_games(rng, games)
    log_lr = centroid_log_lr_direct(g.theta, g.z)
    loss = centroid_loss(g.theta, g.z)
    excess = loss - centroid_tau(g.z)
    fig, (left, right) = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    for ax, xs, label in ((left, loss, r"loss $\ell(\theta; z)$"), (right, excess, r"$\ell(\theta; z) - \tau(z)$")):
        ax.scatter(xs[~g.member], log_lr[~g.member], s=6, alpha=0.5, label="non-members")
        ax.scatter(xs[g.member], log_lr[g.member], s=6, alpha=0.5, label="members")
        ax.set(xlabel=label)
    left.set(ylabel="exact log likelihood ratio", title="Raw loss: a cloud")
    right.set(title=r"Calibrated loss: a line of slope $-n/(n-1)$")
    left.legend()
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
        "apx_opt_convexity.svg": plot_convexity(),
        "apx_opt_gd_linear.svg": plot_gd_linear(rng),
        "apx_opt_cocoercive.svg": plot_cocoercive(rng),
        "apx_opt_lower_bound.svg": plot_lower_bound(),
        "apx_opt_nn.svg": plot_nn(rng),
        "apx_opt_mest_hist.svg": plot_mest_hist(rng),
        "apx_opt_mest_variance.svg": plot_mest_variance(rng),
        "apx_opt_gibbs.svg": plot_gibbs(rng),
    }
    for name, fig in figures.items():
        fig.tight_layout()
        fig.savefig(out / name, metadata={"Date": None})  # no timestamp, so output is reproducible
        plt.close(fig)
        print(f"saved {out / name}")


if __name__ == "__main__":
    main()
