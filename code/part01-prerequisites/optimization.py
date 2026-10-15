"""Check the convergence and sensitivity bounds in the optimization chapter and draw its figures.

    uv run python code/part01-prerequisites/optimization.py
    uv run python code/part01-prerequisites/optimization.py --seed 1 --out /tmp/figures

Prints the numbers quoted in 06-optimization.md (they match exactly with the default seed),
then saves the chapter's figures.
"""

import argparse
import math
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
matplotlib.rcParams["svg.hashsalt"] = "optimization"  # stable element ids, so reruns don't churn the SVGs
matplotlib.rcParams["svg.fonttype"] = "none"  # keep text as text: smaller, diffable files

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from numpy.typing import NDArray

# Figures are build output: _build/figures/<part>/, which the build copies into each book.
FIGURES = Path(__file__).resolve().parents[2] / "_build" / "figures" / Path(__file__).parent.name

Array = NDArray[np.float64]

W_TRUE = np.array([2.0, -2.0, 1.0, 0.0, 0.0])  # the model that generates the labels
RADIUS = 5.0  # the constraint set W is the ball of this radius
SIGMOID_SMOOTHNESS = 1 / (6 * math.sqrt(3))  # max |σ''|, the smoothness of the sigmoid loss


def sigmoid(s: Array) -> Array:
    """σ(s) = 1/(1 + e^{-s}), written with tanh so it never overflows."""
    return 0.5 * (1.0 + np.tanh(0.5 * s))


# --- Data: logistic-regression examples with ‖x‖ = 1 --------------------------------------------


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


def replace_one(data: Data, i: int, x_new: Array, y_new: float) -> Data:
    """The neighbouring dataset: example i replaced by (x_new, y_new)."""
    x, y = data.x.copy(), data.y.copy()
    x[i], y[i] = x_new, y_new
    return Data(x, y)


# --- Logistic loss: value, gradients, Hessian --------------------------------------------------


def logistic_risks(data: Data, w: Array, lam: float = 0.0) -> Array:
    """Regularized empirical risk F(w) + λ/2 ‖w‖² for each row of w (shape (k, d))."""
    margins = (data.x @ w.T) * data.y[:, None]  # (n, k)
    return np.mean(np.logaddexp(0.0, -margins), axis=0) + 0.5 * lam * np.sum(w * w, axis=1)


def logistic_risk(data: Data, w: Array, lam: float = 0.0) -> float:
    return float(logistic_risks(data, w[None, :], lam)[0])


def logistic_example_grads(x: Array, y: Array, w: Array) -> Array:
    """Per-example gradients -y σ(-y<w,x>) x, one row of (x, y, w) at a time."""
    margins = y * np.einsum("ij,ij->i", x, w)
    return (-y * sigmoid(-margins))[:, None] * x


def logistic_grad(data: Data, w: Array, lam: float = 0.0) -> Array:
    margins = data.y * (data.x @ w)
    return data.x.T @ (-data.y * sigmoid(-margins)) / data.y.size + lam * w


def logistic_hessian(data: Data, w: Array, lam: float = 0.0) -> Array:
    """(1/n) Σ σ(m)(1-σ(m)) x xᵀ + λ I."""
    s = sigmoid(data.y * (data.x @ w))
    curv = s * (1 - s)
    return (data.x.T * curv) @ data.x / data.y.size + lam * np.eye(data.x.shape[1])


def logistic_point(w: Array, x: Array, y: float) -> tuple[float, float, float]:
    """Loss, gradient norm and largest Hessian eigenvalue of one example (for the hand example)."""
    m = y * float(w @ x)
    s = 1 / (1 + math.exp(-m))
    norm_x = float(np.linalg.norm(x))
    return math.log1p(math.exp(-m)), (1 - s) * norm_x, s * (1 - s) * norm_x**2


def erm_minimizer(data: Data, lam: float, w0: Array | None = None, tol: float = 1e-12) -> Array:
    """Newton's method with backtracking for the (regularized) logistic risk."""
    w = np.zeros(data.x.shape[1]) if w0 is None else w0.copy()
    for _ in range(100):
        g = logistic_grad(data, w, lam)
        if float(np.linalg.norm(g)) < tol:
            break
        step = np.linalg.solve(logistic_hessian(data, w, lam), g)
        f0, t = logistic_risk(data, w, lam), 1.0
        while logistic_risk(data, w - t * step, lam) > f0 - 0.25 * t * float(g @ step) and t > 1e-10:
            t /= 2
        w = w - t * step
    return w


# --- Lipschitz against smooth: three one-dimensional losses ------------------------------------


def abs_slope(w: float) -> float:
    """Derivative of |w| (taking 0 at the kink)."""
    return math.copysign(1.0, w) if w != 0 else 0.0


def quadratic_slope(w: float) -> float:
    """Derivative of w²/2."""
    return w


def logistic_slope(w: float) -> float:
    """Derivative of ln(1 + e^{-w})."""
    return -1 / (1 + math.exp(w))


def logistic_curvature(w: float) -> float:
    """Second derivative of ln(1 + e^{-w}): σ(w)(1 - σ(w))."""
    s = 1 / (1 + math.exp(-w))
    return s * (1 - s)


def sigmoid_loss_curvature(w: float) -> float:
    """Second derivative of the sigmoid loss σ(-w): σ(1-σ)(1-2σ) evaluated at w (up to sign)."""
    s = 1 / (1 + math.exp(-w))
    return s * (1 - s) * (1 - 2 * s)


# --- Gradient descent and SGD on a two-dimensional least-squares problem -----------------------


@dataclass(frozen=True)
class LeastSquares:
    x: Array  # (n, 2)
    y: Array  # (n,)

    @property
    def hessian(self) -> Array:
        return self.x.T @ self.x / self.y.size

    @property
    def minimizer(self) -> Array:
        return np.linalg.solve(self.x.T @ self.x, self.x.T @ self.y)


def make_least_squares(rng: np.random.Generator, n: int = 200) -> LeastSquares:
    """x ~ N(0, diag(1, 0.1)), so the risk is an elongated bowl; y = <(1, 1), x> + noise."""
    x = rng.standard_normal((n, 2)) * np.array([1.0, math.sqrt(0.1)])
    y = x @ np.array([1.0, 1.0]) + 0.5 * rng.standard_normal(n)
    return LeastSquares(x, y)


def least_squares_risk(ls: LeastSquares, w: Array) -> float:
    r = ls.x @ w - ls.y
    return float(r @ r) / (2 * ls.y.size)


def gd_path(ls: LeastSquares, w0: Array, eta: float, steps: int) -> Array:
    """Iterates of full gradient descent, shape (steps + 1, 2)."""
    h, b = ls.hessian, ls.x.T @ ls.y / ls.y.size
    path = [w0]
    for _ in range(steps):
        path.append(path[-1] - eta * (h @ path[-1] - b))
    return np.array(path)


def sgd_path(rng: np.random.Generator, ls: LeastSquares, w0: Array, eta: float, steps: int) -> Array:
    """Iterates of single-example SGD with a constant step, shape (steps + 1, 2)."""
    path = [w0]
    for i in rng.integers(ls.y.size, size=steps):
        w = path[-1]
        path.append(w - eta * (float(ls.x[i] @ w) - float(ls.y[i])) * ls.x[i])
    return np.array(path)


def gd_steps_to_tolerance(ls: LeastSquares, w0: Array, tol: float) -> int:
    """GD steps with η = 1/β until ‖w_t - w*‖ ≤ tol ‖w_0 - w*‖."""
    beta = float(np.linalg.eigvalsh(ls.hessian)[-1])
    w_star, w, t = ls.minimizer, w0, 0
    start = float(np.linalg.norm(w0 - w_star))
    while float(np.linalg.norm(w - w_star)) > tol * start:
        w = gd_path(ls, w, 1 / beta, 1)[-1]
        t += 1
    return t


def gd_steps_predicted(kappa: float, tol: float) -> int:
    """Smallest t with (1 - 1/κ)^t ≤ tol."""
    return math.ceil(math.log(tol) / math.log(1 - 1 / kappa))


# --- Minibatch gradient variance ---------------------------------------------------------------


def minibatch_variance_formula(sigma1_sq: float, n: int, b: int, replace: bool) -> float:
    """E‖g_B - ∇F‖²: σ₁²/B with replacement, times (n-B)/(n-1) without."""
    return sigma1_sq / b if replace else sigma1_sq / b * (n - b) / (n - 1)


def minibatch_variance_empirical(rng: np.random.Generator, grads: Array, b: int, trials: int, replace: bool) -> float:
    """Monte Carlo E‖g_B - ∇F‖² for minibatches of b rows of the per-example gradients."""
    n = grads.shape[0]
    if replace:
        idx = rng.integers(n, size=(trials, b))
    else:
        idx = np.argsort(rng.random((trials, n)), axis=1)[:, :b]
    dev = grads[idx].mean(axis=1) - grads.mean(axis=0)
    return float(np.mean(np.sum(dev * dev, axis=1)))


def per_example_variance(grads: Array) -> float:
    """σ₁² = (1/n) Σ ‖∇ℓ_i - ∇F‖²."""
    dev = grads - grads.mean(axis=0)
    return float(np.mean(np.sum(dev * dev, axis=1)))


# --- Projected SGD on logistic regression: convex and strongly convex --------------------------


def project_ball(w: Array, radius: float) -> Array:
    """Project each row of w onto the ball of the given radius."""
    norms = np.sqrt(np.einsum("ij,ij->i", w, w))[:, None]
    return w * np.minimum(1.0, radius / np.maximum(norms, 1e-300))


def sgd_average(
    rng: np.random.Generator, data: Data, etas: Array, steps: int, radius: float, noise: float = 0.0
) -> Array:
    """Projected single-example SGD with a constant step per row; returns (1/T) Σ_{t≤T} w_t.

    Each row of the batch is an independent run with step etas[row]. `noise` adds
    N(0, noise²/d I) to every stochastic gradient, so E‖ξ‖² = noise².
    Random draws are made in chunks of 1,000 steps, which is much faster than one call per step.
    """
    k, (n, d) = etas.size, data.x.shape
    w, total = np.zeros((k, d)), np.zeros((k, d))
    step = etas[:, None]
    for start in range(0, steps, 1000):
        m = min(1000, steps - start)
        idx = rng.integers(n, size=(m, k))
        xi = rng.standard_normal((m, k, d)) * (noise / math.sqrt(d)) if noise > 0 else None
        for j in range(m):
            total += w
            g = logistic_example_grads(data.x[idx[j]], data.y[idx[j]], w)
            if xi is not None:
                g += xi[j]
            w = project_ball(w - step * g, radius)
    return total / steps


def convex_sgd_bound(radius: float, lipschitz: float, noise: float, eta: float, steps: int) -> float:
    """R²/(2ηT) + η(L² + σ²)/2."""
    return radius**2 / (2 * eta * steps) + eta * (lipschitz**2 + noise**2) / 2


def convex_sgd_step(radius: float, lipschitz: float, noise: float, steps: int) -> float:
    """The step R/(√(L² + σ²) √T) that minimizes the bound."""
    return radius / (math.sqrt(lipschitz**2 + noise**2) * math.sqrt(steps))


NOISE_LEVELS = (0.0, 1.0, 3.0)  # σ = √E‖ξ‖² of the added gradient noise


def step_size_grid() -> Array:
    """Step sizes 10^-3, 10^-2.75, ..., 10^1."""
    return np.array([10 ** (-3 + i / 4) for i in range(17)])


def convex_sgd_gap(
    rng: np.random.Generator, data: Data, f_star: float, eta: float, steps: int, runs: int, noise: float = 0.0
) -> float:
    """Average over runs of F(w̄_T) - F*."""
    w_bar = sgd_average(rng, data, np.full(runs, eta), steps, RADIUS, noise)
    return float(np.mean(logistic_risks(data, w_bar))) - f_star


def step_size_sweep(
    rng: np.random.Generator, data: Data, f_star: float, etas: Array, steps: int, runs: int, noise: float
) -> Array:
    """F(w̄_T) - F* averaged over runs, for each step size (all run in one batch)."""
    w_bar = sgd_average(rng, data, np.repeat(etas, runs), steps, RADIUS, noise)
    return (logistic_risks(data, w_bar) - f_star).reshape(etas.size, runs).mean(axis=1)


@dataclass(frozen=True)
class StronglyConvexRun:
    checkpoints: list[int]
    uniform: list[float]  # η_t = 1/(λt), uniform average
    weighted: list[float]  # η_t = 2/(λ(t+1)), average weighted by t
    last: list[float]  # last iterate of the second schedule


def strongly_convex_sgd(
    rng: np.random.Generator, data: Data, lam: float, j_star: float, checkpoints: list[int], runs: int
) -> StronglyConvexRun:
    """Projected SGD on F + λ/2‖w‖² with the two step schedules, gaps J(·) - J* at checkpoints."""
    n, d = data.x.shape
    wu, wl = np.zeros((runs, d)), np.zeros((runs, d))
    su, sw = np.zeros((runs, d)), np.zeros((runs, d))
    out_u: list[float] = []
    out_w: list[float] = []
    out_l: list[float] = []
    for t in range(1, max(checkpoints) + 1):
        su += wu
        sw += t * wl
        if t in checkpoints:
            out_u.append(float(np.mean(logistic_risks(data, su / t, lam))) - j_star)
            out_w.append(float(np.mean(logistic_risks(data, sw / (t * (t + 1) / 2), lam))) - j_star)
            out_l.append(float(np.mean(logistic_risks(data, wl, lam))) - j_star)
        idx = rng.integers(n, size=runs)
        gu = logistic_example_grads(data.x[idx], data.y[idx], wu) + lam * wu
        idx = rng.integers(n, size=runs)
        gl = logistic_example_grads(data.x[idx], data.y[idx], wl) + lam * wl
        wu = project_ball(wu - gu / (lam * t), RADIUS)
        wl = project_ball(wl - 2 * gl / (lam * (t + 1)), RADIUS)
    return StronglyConvexRun(checkpoints, out_u, out_w, out_l)


def strongly_convex_bounds(g: float, lam: float, steps: int) -> tuple[float, float]:
    """G²(1 + ln T)/(2λT) for the uniform average and 2G²/(λ(T+1)) for the weighted one."""
    return g**2 * (1 + math.log(steps)) / (2 * lam * steps), 2 * g**2 / (lam * (steps + 1))


# --- Non-convex: SGD on the sigmoid loss --------------------------------------------------------


def sigmoid_loss_grads(x: Array, y: Array, w: Array) -> Array:
    """Per-example gradients of σ(-y<w,x>): -y σ'(-m) x."""
    s = sigmoid(-y * np.sum(x * w, axis=1))
    return (-y * s * (1 - s))[:, None] * x


def sigmoid_loss_full_grad_sq(data: Data, w: Array) -> Array:
    """‖∇F(w)‖² for each row of w, F the average sigmoid loss."""
    s = sigmoid(-(data.x @ w.T) * data.y[:, None])  # (n, k)
    grad = -(data.x.T @ (data.y[:, None] * s * (1 - s))) / data.y.size  # (d, k)
    return np.sum(grad * grad, axis=0)


def nonconvex_step(beta: float, delta: float, sigma_sq: float, steps: int) -> float:
    """η = min(1/β, √(2Δ/(βσ²T)))."""
    return min(1 / beta, math.sqrt(2 * delta / (beta * sigma_sq * steps)))


def nonconvex_bound(beta: float, delta: float, sigma_sq: float, eta: float, steps: int) -> float:
    """(1/T) Σ E‖∇F(w_t)‖² ≤ 2Δ/(ηT) + βησ², valid for η ≤ 1/β."""
    return 2 * delta / (eta * steps) + beta * eta * sigma_sq


def nonconvex_sgd(rng: np.random.Generator, data: Data, eta: float, steps: int, runs: int) -> float:
    """Average over t ≤ T and over runs of ‖∇F(w_t)‖² for SGD on the sigmoid loss from w = 0."""
    n, d = data.x.shape
    w, total = np.zeros((runs, d)), 0.0
    for _ in range(steps):
        total += float(np.mean(sigmoid_loss_full_grad_sq(data, w)))
        idx = rng.integers(n, size=runs)
        w = w - eta * sigmoid_loss_grads(data.x[idx], data.y[idx], w)
    return total / steps


# --- Sensitivity of regularized ERM ------------------------------------------------------------


def erm_sensitivity_bound(lipschitz: float, lam: float, n: int) -> float:
    return 2 * lipschitz / (lam * n)


def max_minimizer_shift(rng: np.random.Generator, n: int, lam: float, trials: int) -> float:
    """Largest ‖w*(D) - w*(D')‖ over `trials` random single-example replacements."""
    data = make_logistic_data(rng, n)
    w = erm_minimizer(data, lam)
    worst = 0.0
    for _ in range(trials):
        new = make_logistic_data(rng, 1)
        neighbour = replace_one(data, int(rng.integers(n)), new.x[0], float(new.y[0]))
        worst = max(worst, float(np.linalg.norm(erm_minimizer(neighbour, lam, w) - w)))
    return worst


def data_curvature(data: Data, w: Array) -> float:
    """Smallest eigenvalue of the unregularized logistic Hessian at w."""
    return float(np.linalg.eigvalsh(logistic_hessian(data, w))[0])


# --- Non-convex: a double well whose minimizer jumps -------------------------------------------


def double_well_risk(w: Array, zbar: float, c: float = 1.0) -> Array:
    """(1/n) Σ [(w² - 1)²/4 - c z_i w] = (w² - 1)²/4 - c z̄ w."""
    return (w * w - 1) ** 2 / 4 - c * zbar * w


def double_well_argmin(zbar: float, c: float = 1.0) -> float:
    """Global minimizer on [-2, 2]: grid search, then Newton on w³ - w - c z̄ = 0."""
    grid = np.linspace(-2.0, 2.0, 40001)
    w = float(grid[np.argmin(double_well_risk(grid, zbar, c))])
    for _ in range(20):
        w -= (w**3 - w - c * zbar) / (3 * w * w - 1)
    return w


def double_well_zbar(n: int, positives: int) -> float:
    """Mean of n labels z_i ∈ {-1, +1} with the given number of +1s."""
    return (2 * positives - n) / n


# --- The chapter's table -----------------------------------------------------------------------


def print_table(rng: np.random.Generator) -> None:
    # Hand example: one logistic example with x = (0.6, 0.8), y = +1.
    x0 = np.array([0.6, 0.8])
    for w in (np.zeros(2), np.array([3.0, 4.0])):
        loss, gnorm, curv = logistic_point(w, x0, 1.0)
        print(f"[logistic point w=({w[0]:g},{w[1]:g})] loss={loss:.4f}  |grad|={gnorm:.4f}  curvature={curv:.4f}")

    ls = make_least_squares(rng)
    eig = np.linalg.eigvalsh(ls.hessian)
    kappa = float(eig[-1] / eig[0])
    w0 = np.array([-1.0, 3.0])
    print(
        f"[least squares] lambda_min={eig[0]:.3f}  beta={eig[-1]:.3f}  kappa={kappa:.2f}  "
        f"GD steps to 1e-6: observed={gd_steps_to_tolerance(ls, w0, 1e-6)}  "
        f"predicted={gd_steps_predicted(kappa, 1e-6)}"
    )

    data = make_logistic_data(rng, 1000)
    w_star = erm_minimizer(data, 0.0)
    f_star = logistic_risk(data, w_star)
    print(
        f"[logistic data n=1000 d=5] |w*|={np.linalg.norm(w_star):.3f}  F*={f_star:.4f}  "
        f"data curvature={data_curvature(data, w_star):.4f}"
    )

    grads = logistic_example_grads(data.x, data.y, np.tile(w_star, (1000, 1)))
    s1 = per_example_variance(grads)
    print(f"[minibatch at w*] sigma1^2={s1:.4f}")
    for b in (1, 10, 100, 500):
        print(
            f"[minibatch B={b}] with repl: {minibatch_variance_empirical(rng, grads, b, 4000, True):.2e} "
            f"vs {minibatch_variance_formula(s1, 1000, b, True):.2e}   without: "
            f"{minibatch_variance_empirical(rng, grads, b, 4000, False):.2e} "
            f"vs {minibatch_variance_formula(s1, 1000, b, False):.2e}"
        )

    for steps in (100, 1000, 10000):
        eta = convex_sgd_step(RADIUS, 1.0, 0.0, steps)
        gap = convex_sgd_gap(rng, data, f_star, eta, steps, runs=20)
        print(
            f"[convex SGD T={steps}] eta={eta:.4f}  gap={gap:.2e}  bound={convex_sgd_bound(RADIUS, 1.0, 0.0, eta, steps):.2e}"
        )

    lam = 0.1
    w_lam = erm_minimizer(data, lam)
    j_star = logistic_risk(data, w_lam, lam)
    g = 1.0 + lam * RADIUS
    sc = strongly_convex_sgd(rng, data, lam, j_star, [10000], runs=20)
    bu, bw = strongly_convex_bounds(g, lam, 10000)
    print(
        f"[strongly convex lam={lam} T=10000] |w*|={np.linalg.norm(w_lam):.3f}  G={g:.1f}  "
        f"uniform gap={sc.uniform[0]:.2e} bound={bu:.2e}  weighted gap={sc.weighted[0]:.2e} bound={bw:.2e}  "
        f"last={sc.last[0]:.2e}"
    )

    steps = 10000
    etas = step_size_grid()
    for noise in NOISE_LEVELS:
        eta = convex_sgd_step(RADIUS, 1.0, noise, steps)
        gaps = step_size_sweep(rng, data, f_star, np.append(etas, eta), steps, runs=10, noise=noise)
        best = int(np.argmin(gaps[:-1]))
        print(
            f"[noise sigma={noise:g} T={steps}] eta*={eta:.4f}  bound={convex_sgd_bound(RADIUS, 1.0, noise, eta, steps):.3f}  "
            f"gap at eta*={gaps[-1]:.2e}  best grid eta={etas[best]:.3f} gap={gaps[best]:.2e}"
        )

    beta, delta, sigma_sq = SIGMOID_SMOOTHNESS, 0.5, 1 / 16
    print(
        f"[sigmoid loss] beta={beta:.4f}  |grad F(0)|^2={float(sigmoid_loss_full_grad_sq(data, np.zeros((1, 5)))[0]):.2e}"
    )
    for steps in (100, 1000, 10000):
        eta = nonconvex_step(beta, delta, sigma_sq, steps)
        avg = nonconvex_sgd(rng, data, eta, steps, runs=5)
        print(
            f"[non-convex T={steps}] eta={eta:.3f}  mean |grad|^2={avg:.2e}  "
            f"bound={nonconvex_bound(beta, delta, sigma_sq, eta, steps):.2e}"
        )

    for n in (100, 1000, 10000):
        for lam in (0.01, 0.1, 1.0):
            shift = max_minimizer_shift(rng, n, lam, trials=20)
            bound = erm_sensitivity_bound(1.0, lam, n)
            print(f"[ERM n={n} lam={lam:g}] max shift={shift:.2e}  bound={bound:.2e}  ratio={shift / bound:.3f}")

    for n in (11, 1001):
        zp, zm = double_well_zbar(n, (n + 1) // 2), double_well_zbar(n, (n - 1) // 2)
        a, b = double_well_argmin(zp), double_well_argmin(zm)
        print(f"[double well n={n}] argmin before={a:.4f}  after={b:.4f}  shift={abs(a - b):.4f}")


# --- Figures -----------------------------------------------------------------------------------


def plot_lipschitz_smooth() -> Figure:
    ws = [-4 + i / 100 for i in range(801)]
    fig, (left, right) = plt.subplots(1, 2, figsize=(10, 4))
    left.plot(ws, [abs_slope(w) if abs(w) > 1e-9 else math.nan for w in ws], label=r"$|w|$: bounded slope that jumps")
    left.plot(ws, [quadratic_slope(w) for w in ws], label=r"$w^2/2$: smooth slope, unbounded")
    left.plot(ws, [logistic_slope(w) for w in ws], label=r"$\ln(1+e^{-w})$: bounded and smooth")
    for level in (-1.0, 1.0):
        left.axhline(level, color="gray", linestyle=":", linewidth=1)
    left.set(ylim=(-2.5, 2.5), xlabel="w", ylabel=r"slope $f'(w)$", title="Lipschitz bounds the slope")
    left.legend(loc="upper left", fontsize=8)
    right.plot(ws, [logistic_curvature(w) for w in ws], label=r"logistic: $\sigma(w)(1-\sigma(w))$")
    right.plot(ws, [abs(sigmoid_loss_curvature(w)) for w in ws], label=r"sigmoid loss: $|\sigma''(w)|$")
    right.axhline(0.25, color="C0", linestyle="--", linewidth=1, label=r"$\beta = 1/4$")
    right.axhline(SIGMOID_SMOOTHNESS, color="C1", linestyle="--", linewidth=1, label=r"$\beta = 1/(6\sqrt{3})$")
    right.set(ylim=(0, 0.3), xlabel="w", ylabel=r"curvature $|f''(w)|$", title="Smooth bounds the change of slope")
    right.legend(loc="upper right", fontsize=8)
    return fig


def plot_gd_sgd_paths(rng: np.random.Generator) -> Figure:
    ls = make_least_squares(rng)
    w0 = np.array([-1.0, 3.0])
    beta = float(np.linalg.eigvalsh(ls.hessian)[-1])
    gd = gd_path(ls, w0, 1 / beta, 60)
    sgd = sgd_path(rng, ls, w0, 0.1, 600)
    w_star = ls.minimizer
    g1, g2 = np.meshgrid(np.linspace(-1.5, 3.0, 181), np.linspace(-1.0, 3.5, 181))
    risk = np.array([[least_squares_risk(ls, np.array([a, b])) for a, b in zip(r1, r2)] for r1, r2 in zip(g1, g2)])
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.contour(
        g1, g2, risk, levels=np.quantile(risk, [0.005, 0.02, 0.05, 0.1, 0.2, 0.35, 0.5, 0.7]), colors="lightgray"
    )
    ax.plot(sgd[:, 0], sgd[:, 1], "-", color="C1", linewidth=0.7, alpha=0.8, label="SGD, one example per step, η = 0.1")
    ax.plot(gd[:, 0], gd[:, 1], "o-", color="C0", markersize=3, label="gradient descent, η = 1/β")
    ax.plot([w_star[0]], [w_star[1]], "k*", markersize=12, label="minimizer w*")
    ax.set(xlabel=r"$w_1$", ylabel=r"$w_2$", aspect="equal", title="Least squares with condition number ≈ 10")
    ax.legend(loc="upper right", fontsize=8)
    return fig


def plot_minibatch_variance(rng: np.random.Generator) -> Figure:
    data = make_logistic_data(rng, 1000)
    w_star = erm_minimizer(data, 0.0)
    grads = logistic_example_grads(data.x, data.y, np.tile(w_star, (1000, 1)))
    s1 = per_example_variance(grads)
    bs = [1, 2, 5, 10, 20, 50, 100, 200, 500, 900]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(bs, [minibatch_variance_formula(s1, 1000, b, True) for b in bs], color="C0", label=r"$\sigma_1^2/B$")
    ax.plot(
        bs,
        [minibatch_variance_formula(s1, 1000, b, False) for b in bs],
        color="C1",
        label=r"$\frac{\sigma_1^2}{B}\cdot\frac{n-B}{n-1}$",
    )
    ax.plot(
        bs,
        [minibatch_variance_empirical(rng, grads, b, 2000, True) for b in bs],
        "o",
        color="C0",
        label="sampled with replacement",
    )
    ax.plot(
        bs,
        [minibatch_variance_empirical(rng, grads, b, 2000, False) for b in bs],
        "s",
        color="C1",
        label="sampled without replacement",
    )
    ax.set(
        xscale="log",
        yscale="log",
        xlabel="batch size B",
        ylabel=r"$\mathbb{E}\|g_B - \nabla F\|^2$",
        title="Minibatch gradient noise, n = 1000 logistic examples",
    )
    ax.legend(fontsize=8)
    return fig


def plot_sgd_convex(rng: np.random.Generator) -> Figure:
    data = make_logistic_data(rng, 1000)
    f_star = logistic_risk(data, erm_minimizer(data, 0.0))
    ts = [10, 30, 100, 300, 1000, 3000, 10000]
    gaps = [convex_sgd_gap(rng, data, f_star, convex_sgd_step(RADIUS, 1.0, 0.0, t), t, 10) for t in ts]
    bounds = [convex_sgd_bound(RADIUS, 1.0, 0.0, convex_sgd_step(RADIUS, 1.0, 0.0, t), t) for t in ts]

    lam = 0.1
    j_star = logistic_risk(data, erm_minimizer(data, lam), lam)
    g = 1.0 + lam * RADIUS
    sc = strongly_convex_sgd(rng, data, lam, j_star, ts, runs=10)
    sc_bounds = [strongly_convex_bounds(g, lam, t) for t in ts]

    fig, (left, right) = plt.subplots(1, 2, figsize=(10, 4))
    left.plot(ts, bounds, "--", color="C0", label=r"bound $RL/\sqrt{T}$")
    left.plot(ts, gaps, "o-", color="C0", label="observed, averaged iterate")
    left.set(
        xscale="log",
        yscale="log",
        xlabel="steps T",
        ylabel=r"$F(\bar w_T) - F^*$",
        title="Convex, 1-Lipschitz (logistic, R = 5)",
    )
    left.legend(fontsize=8)
    right.plot(ts, [b[0] for b in sc_bounds], "--", color="C0", label=r"bound $G^2(1+\ln T)/(2\lambda T)$")
    right.plot(ts, sc.uniform, "o-", color="C0", label=r"uniform average, $\eta_t = 1/(\lambda t)$")
    right.plot(ts, [b[1] for b in sc_bounds], "--", color="C1", label=r"bound $2G^2/(\lambda(T+1))$")
    right.plot(ts, sc.weighted, "s-", color="C1", label=r"weighted average, $\eta_t = 2/(\lambda(t+1))$")
    right.plot(ts, sc.last, "^:", color="C2", label="last iterate (same run)")
    right.set(
        xscale="log",
        yscale="log",
        xlabel="steps T",
        ylabel=r"$J(\bar w_T) - J^*$",
        title=f"λ-strongly convex (logistic + ridge, λ = {lam:g})",
    )
    right.legend(fontsize=7)
    return fig


def plot_noise_step_size(rng: np.random.Generator, steps: int = 10000) -> Figure:
    data = make_logistic_data(rng, 1000)
    f_star = logistic_risk(data, erm_minimizer(data, 0.0))
    etas = step_size_grid()
    fig, ax = plt.subplots(figsize=(6, 5))
    for color, noise in zip(("C0", "C1", "C2"), NOISE_LEVELS):
        gaps = step_size_sweep(rng, data, f_star, etas, steps, runs=5, noise=noise)
        ax.plot(etas, gaps, "o-", color=color, markersize=4, label=f"observed, σ = {noise:g}")
        ax.plot(
            etas,
            [convex_sgd_bound(RADIUS, 1.0, noise, float(e), steps) for e in etas],
            "--",
            color=color,
            linewidth=1,
            label=f"bound, σ = {noise:g}",
        )
        eta = convex_sgd_step(RADIUS, 1.0, noise, steps)
        ax.axvline(eta, color=color, linestyle=":", linewidth=1)
    ax.set(
        xscale="log",
        yscale="log",
        ylim=(1e-4, 30),
        xlabel="step size η (dotted: η* from the bound)",
        ylabel=r"$F(\bar w_T) - F^*$",
        title=f"Projected SGD on logistic regression, T = {steps}",
    )
    ax.legend(fontsize=8, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.15))
    return fig


def plot_nonconvex(rng: np.random.Generator) -> Figure:
    data = make_logistic_data(rng, 1000)
    beta, delta, sigma_sq = SIGMOID_SMOOTHNESS, 0.5, 1 / 16
    ts = [10, 30, 100, 300, 1000, 3000, 10000]
    etas = [nonconvex_step(beta, delta, sigma_sq, t) for t in ts]
    observed = [nonconvex_sgd(rng, data, eta, t, runs=5) for eta, t in zip(etas, ts)]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(
        ts,
        [nonconvex_bound(beta, delta, sigma_sq, eta, t) for eta, t in zip(etas, ts)],
        "--",
        label=r"bound $2\Delta/(\eta T) + \beta\eta\sigma^2$",
    )
    ax.plot(ts, observed, "o-", label=r"observed $\frac{1}{T}\sum_t \|\nabla F(w_t)\|^2$")
    ax.set(
        xscale="log",
        yscale="log",
        xlabel="steps T",
        ylabel="mean squared gradient norm",
        title="SGD on the non-convex sigmoid loss",
    )
    ax.legend()
    return fig


def plot_erm_sensitivity(rng: np.random.Generator) -> Figure:
    ns = [100, 300, 1000, 3000, 10000]
    fig, (left, right) = plt.subplots(1, 2, figsize=(10, 4))
    for color, lam in zip(("C0", "C1", "C2"), (0.01, 0.1, 1.0)):
        left.plot(ns, [erm_sensitivity_bound(1.0, lam, n) for n in ns], "--", color=color, linewidth=1)
        left.plot(ns, [max_minimizer_shift(rng, n, lam, 10) for n in ns], "o-", color=color, label=f"λ = {lam:g}")
    left.set(
        xscale="log",
        yscale="log",
        xlabel="n",
        ylabel=r"$\|w^*(D) - w^*(D')\|$",
        title="Observed (solid) and 2L/(λn) (dashed)",
    )
    left.legend()
    lams = [10 ** (-3 + i / 2) for i in range(8)]
    right.plot(lams, [erm_sensitivity_bound(1.0, lam, 1000) for lam in lams], "--", label="bound 2L/(λn)")
    right.plot(lams, [max_minimizer_shift(rng, 1000, lam, 10) for lam in lams], "o-", label="observed (max of 10)")
    right.set(
        xscale="log",
        yscale="log",
        xlabel="λ",
        ylabel=r"$\|w^*(D) - w^*(D')\|$",
        title="n = 1000: the data adds curvature",
    )
    right.legend()
    return fig


def plot_nonconvex_jump(n: int = 11) -> Figure:
    ws = np.linspace(-1.8, 1.8, 721)
    fig, ax = plt.subplots(figsize=(6, 4))
    for color, pos, name in (("C0", (n + 1) // 2, "D"), ("C1", (n - 1) // 2, "D′")):
        zbar = double_well_zbar(n, pos)
        ax.plot(ws, double_well_risk(ws, zbar), color=color, label=f"{name}: {pos} of {n} labels are +1")
        w = double_well_argmin(zbar)
        ax.plot([w], [float(double_well_risk(np.array([w]), zbar)[0])], "o", color=color, markersize=8)
    ax.set(
        ylim=(-0.15, 0.6), xlabel="w", ylabel="empirical risk", title="Changing one example moves the minimizer across"
    )
    ax.legend(loc="upper center")
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
        "optimization_lipschitz_smooth.svg": plot_lipschitz_smooth(),
        "optimization_gd_sgd_paths.svg": plot_gd_sgd_paths(rng),
        "optimization_minibatch_variance.svg": plot_minibatch_variance(rng),
        "optimization_sgd_convex.svg": plot_sgd_convex(rng),
        "optimization_noise_step_size.svg": plot_noise_step_size(rng),
        "optimization_nonconvex_sgd.svg": plot_nonconvex(rng),
        "optimization_erm_sensitivity.svg": plot_erm_sensitivity(rng),
        "optimization_nonconvex_jump.svg": plot_nonconvex_jump(),
    }
    for name, fig in figures.items():
        fig.tight_layout()
        fig.savefig(out / name, metadata={"Date": None})  # no timestamp, so output is reproducible
        plt.close(fig)
        print(f"saved {out / name}")


if __name__ == "__main__":
    main()
