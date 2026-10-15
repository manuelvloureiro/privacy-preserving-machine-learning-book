"""Check the numbers in the private empirical risk minimization chapter and draw its figures.

    uv run python code/part05-dp-ml/dp_erm.py
    uv run python code/part05-dp-ml/dp_erm.py --seed 1 --out /tmp/figures

Prints the numbers quoted in 01-dp-erm.md (they match exactly with the default seed), then saves
the chapter's figures.

The running example: regularized logistic regression, J(w; D) = (1/n) Σ ln(1 + e^{-y<w, x>}) + (λ/2)‖w‖²,
on n examples with x uniform on the unit sphere of R^d (so the loss is 1-Lipschitz in w) and labels
drawn from a logistic model. Neighbouring datasets differ by replacing one example. Three ways of
making the trained weights private are compared: noise on the output, noise in the objective and
noise on every gradient.

Public functions used by later chapters: sigmoid, LogisticData, make_logistic_data, logistic_objective,
logistic_grad, fit_logistic, excess_risk, normal_cdf, gaussian_delta, gaussian_mu, gaussian_sigma.
"""

import os

# The models are tiny; multi-threaded BLAS only adds overhead.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import argparse
import math
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
matplotlib.rcParams["svg.hashsalt"] = "dp_erm"  # stable element ids, so reruns don't churn the SVGs
matplotlib.rcParams["svg.fonttype"] = "none"  # keep text as text: smaller, diffable files

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from numpy.typing import NDArray

# Figures are build output: _build/figures/<part>/, which the build copies into each book.
FIGURES = Path(__file__).resolve().parents[2] / "_build" / "figures" / Path(__file__).parent.name

Array = NDArray[np.float64]

# The running example of the chapter.
D_RUN = 10  # dimension
LAM_RUN = 0.01  # ridge coefficient λ
EPS_RUN, DELTA_RUN = 1.0, 1e-5
NS_RUN: tuple[int, ...] = (1_000, 3_000, 10_000, 30_000, 100_000)
TRIALS_RUN = 20
C_LOGISTIC = 0.25  # the logistic loss's second derivative in the margin is at most 1/4
W_TRUE: Array = np.full(D_RUN, 5.0 / math.sqrt(D_RUN))  # ‖w_true‖ = 5


# --- Gaussian tools -------------------------------------------------------------------------


def normal_cdf(x: float) -> float:
    """Φ(x); erfc keeps full relative precision in the lower tail."""
    return 0.5 * math.erfc(-x / math.sqrt(2))


def bisect(f: Callable[[float], bool], lo: float, hi: float, steps: int = 200) -> float:
    """Smallest x in [lo, hi] with f(x) true, for f false-then-true (returns the upper end of the bracket)."""
    for _ in range(steps):
        mid = (lo + hi) / 2
        if f(mid):
            hi = mid
        else:
            lo = mid
    return hi


def gaussian_delta(eps: float, mu: float) -> float:
    """Exact δ(ε) of a Gaussian mechanism whose sensitivity is μ noise standard deviations (Balle & Wang 2018)."""
    if mu <= 0:
        return 0.0
    return normal_cdf(mu / 2 - eps / mu) - math.exp(eps) * normal_cdf(-mu / 2 - eps / mu)


def gaussian_mu(eps: float, delta: float) -> float:
    """The largest ratio μ = Δ/σ for which the Gaussian mechanism is (ε, δ)-DP."""
    return bisect(lambda m: gaussian_delta(eps, m) > delta, 1e-9, 100.0) * (1 - 1e-12)


def gaussian_sigma(sensitivity: float, eps: float, delta: float) -> float:
    """Smallest noise standard deviation that makes the Gaussian mechanism (ε, δ)-DP (exact calibration)."""
    return sensitivity / gaussian_mu(eps, delta)


def classic_sigma(sensitivity: float, eps: float, delta: float) -> float:
    """The classic calibration Δ√(2 ln(1.25/δ))/ε, valid for ε < 1."""
    return sensitivity * math.sqrt(2 * math.log(1.25 / delta)) / eps


# --- Data and regularized logistic regression ---------------------------------------------------


def sigmoid(s: Array) -> Array:
    """σ(s) = 1/(1 + e^{-s}), written with tanh so it never overflows."""
    return 0.5 * (1.0 + np.tanh(0.5 * s))


@dataclass(frozen=True)
class LogisticData:
    x: Array  # (n, d), every row has unit norm
    y: Array  # (n,), labels ±1


def make_logistic_data(rng: np.random.Generator, n: int, w_true: Array = W_TRUE) -> LogisticData:
    """x uniform on the unit sphere; y = +1 with probability σ(<w_true, x>)."""
    z = rng.standard_normal((n, w_true.size))
    x = z / np.linalg.norm(z, axis=1, keepdims=True)
    y = np.where(rng.random(n) < sigmoid(x @ w_true), 1.0, -1.0)
    return LogisticData(x, y)


def logistic_objective(w: Array, data: LogisticData, lam: float, b: Array | None = None) -> float:
    """J(w) = mean logistic loss + (λ/2)‖w‖², plus <b, w>/n when a linear perturbation b is given."""
    margins = data.y * (data.x @ w)
    value = float(np.mean(np.logaddexp(0.0, -margins))) + 0.5 * lam * float(w @ w)
    if b is not None:
        value += float(b @ w) / data.y.size
    return value


def example_grads(w: Array, data: LogisticData) -> Array:
    """Per-example gradients of the logistic loss, -y σ(-y<w, x>) x; each has norm at most ‖x‖ = 1."""
    margins = data.y * (data.x @ w)
    return (-data.y * sigmoid(-margins))[:, None] * data.x


def logistic_grad(w: Array, data: LogisticData, lam: float, b: Array | None = None) -> Array:
    g = example_grads(w, data).mean(axis=0) + lam * w
    if b is not None:
        g = g + b / data.y.size
    return g


def logistic_hessian(w: Array, data: LogisticData, lam: float) -> Array:
    u = sigmoid(data.y * (data.x @ w))
    curv = u * (1 - u)
    return (data.x * curv[:, None]).T @ data.x / data.y.size + lam * np.eye(w.size)


def fit_logistic(data: LogisticData, lam: float, b: Array | None = None, tol: float = 1e-12) -> Array:
    """The exact minimizer of the (perturbed) regularized objective, by Newton's method with backtracking."""
    w = np.zeros(data.x.shape[1])
    for _ in range(100):
        g = logistic_grad(w, data, lam, b)
        if float(np.linalg.norm(g)) < tol:
            break
        step = np.linalg.solve(logistic_hessian(w, data, lam), g)
        t, f0 = 1.0, logistic_objective(w, data, lam, b)
        while logistic_objective(w - t * step, data, lam, b) > f0 - 0.25 * t * float(g @ step) and t > 1e-10:
            t /= 2
        w = w - t * step
    return w


def excess_risk(w: Array, w_star: Array, data: LogisticData, lam: float) -> float:
    """J(w) - J(w*): how much worse the private weights fit the training objective."""
    return logistic_objective(w, data, lam) - logistic_objective(w_star, data, lam)


def erm_sensitivity(n: int, lam: float, lipschitz: float = 1.0) -> float:
    """ℓ2 sensitivity of the regularized minimizer under replace-one: 2L/(λn) (the optimization chapter)."""
    return 2 * lipschitz / (lam * n)


# --- Noise with density proportional to exp(-‖b‖/s) ---------------------------------------------


def norm_noise(rng: np.random.Generator, d: int, scale: float) -> Array:
    """A draw from the density ∝ exp(-‖b‖/scale) on R^d: uniform direction, Gamma(d, scale) length."""
    z = rng.standard_normal(d)
    return z / np.linalg.norm(z) * rng.gamma(d, scale)


def norm_noise_second_moment(d: int, scale: float) -> float:
    """E‖b‖² for the density ∝ exp(-‖b‖/scale): the Gamma(d, scale) second moment d(d+1)scale²."""
    return d * (d + 1) * scale**2


# --- Section: output perturbation ---------------------------------------------------------------


def output_perturbation_gauss(
    rng: np.random.Generator, w_star: Array, n: int, lam: float, eps: float, delta: float
) -> Array:
    sigma = gaussian_sigma(erm_sensitivity(n, lam), eps, delta)
    return w_star + sigma * rng.standard_normal(w_star.size)


def output_perturbation_pure(rng: np.random.Generator, w_star: Array, n: int, lam: float, eps: float) -> Array:
    return w_star + norm_noise(rng, w_star.size, erm_sensitivity(n, lam) / eps)


def approximate_minimizer_sensitivity(n: int, lam: float, grad_tol: float, lipschitz: float = 1.0) -> float:
    """Sensitivity of any point whose gradient norm is at most grad_tol: 2L/(λn) + 2 grad_tol/λ."""
    return erm_sensitivity(n, lam, lipschitz) + 2 * grad_tol / lam


# --- Section: objective perturbation -------------------------------------------------------------


@dataclass(frozen=True)
class ObjectivePlan:
    eps_noise: float  # ε' spent on the noise vector
    extra_lam: float  # extra ridge added when ε' would otherwise be too small


def objective_perturbation_plan(n: int, lam: float, eps: float, c: float = C_LOGISTIC) -> ObjectivePlan:
    """Chaudhuri, Monteleoni & Sarwate (2011), Algorithm 2: split ε between the noise and the Jacobian."""
    eps_noise = eps - 2 * math.log1p(c / (n * lam))
    if eps_noise > 0:
        return ObjectivePlan(eps_noise, 0.0)
    return ObjectivePlan(eps / 2, c / (n * math.expm1(eps / 4)) - lam)


def objective_perturbation(
    rng: np.random.Generator, data: LogisticData, lam: float, eps: float, c: float = C_LOGISTIC
) -> Array:
    plan = objective_perturbation_plan(data.y.size, lam, eps, c)
    b = norm_noise(rng, data.x.shape[1], 2.0 / plan.eps_noise)
    return fit_logistic(data, lam + plan.extra_lam, b)


# One-dimensional check of the change-of-variables argument: the density of the output under D and D'.


@dataclass(frozen=True)
class Density1D:
    w: Array
    p: Array  # density of the output under D
    p_prime: Array  # density under D'
    noise_part: Array  # ε'(|b'(w)| - |b(w)|)/2, the log ratio of the noise densities
    jacobian_part: Array  # ln |b'(w)|/|b'_{D'}(w)|, the log ratio of the Jacobians


def _b_and_slope(x: Array, y: Array, lam: float, grid: Array) -> tuple[Array, Array]:
    """b(w) = -n J'(w) on the grid, and its derivative db/dw = -n J''(w)."""
    s = sigmoid(-(y[None, :] * x[None, :] * grid[:, None]))
    b = (y[None, :] * x[None, :] * s).sum(axis=1) - x.size * lam * grid
    db = -(x[None, :] ** 2 * s * (1 - s)).sum(axis=1) - x.size * lam
    return b, db


def objective_density_1d(
    xs: Array, ys: Array, x_new: float, y_new: float, lam: float, eps_noise: float, grid: Array
) -> Density1D:
    """Exact output densities of 1-D objective perturbation for D and D' (last example replaced).

    The minimizer w of J(w) + bw/n satisfies b = -n J'(w), so w ↦ b is a bijection and the density of
    w is ν(b(w)) |b'(w)| with ν(b) = (ε'/4) e^{-ε'|b|/2}, the one-dimensional density ∝ exp(-ε'|b|/2).
    """
    x2, y2 = xs.copy(), ys.copy()
    x2[-1], y2[-1] = x_new, y_new
    b1, db1 = _b_and_slope(xs, ys, lam, grid)
    b2, db2 = _b_and_slope(x2, y2, lam, grid)
    p1 = eps_noise / 4 * np.exp(-eps_noise * np.abs(b1) / 2) * np.abs(db1)
    p2 = eps_noise / 4 * np.exp(-eps_noise * np.abs(b2) / 2) * np.abs(db2)
    return Density1D(grid, p1, p2, eps_noise * (np.abs(b2) - np.abs(b1)) / 2, np.log(np.abs(db1)) - np.log(np.abs(db2)))


@dataclass(frozen=True)
class Check1D:
    total: float  # largest |ln p/p'|
    noise: float  # largest |noise part|
    jacobian: float  # largest |Jacobian part|
    noise_budget: float  # ε'
    jacobian_budget: float  # 2 ln(1 + c/(nλ))


def worst_case_1d(rng: np.random.Generator, n: int, lam: float, eps: float, trials: int, grid: Array) -> Check1D:
    """Largest log ratios over random 1-D datasets and replacements, against the two budgets."""
    plan = objective_perturbation_plan(n, lam, eps)
    lam_total = lam + plan.extra_lam
    total = noise = jac = 0.0
    for _ in range(trials):
        xs = rng.uniform(-1.0, 1.0, n)
        ys = rng.choice([-1.0, 1.0], n)
        x_new, y_new = float(rng.uniform(-1.0, 1.0)), float(rng.choice([-1.0, 1.0]))
        dens = objective_density_1d(xs, ys, x_new, y_new, lam_total, plan.eps_noise, grid)
        total = max(total, float(np.max(np.abs(np.log(dens.p) - np.log(dens.p_prime)))))
        noise = max(noise, float(np.max(np.abs(dens.noise_part))))
        jac = max(jac, float(np.max(np.abs(dens.jacobian_part))))
    return Check1D(total, noise, jac, plan.eps_noise, 2 * math.log1p(C_LOGISTIC / (n * lam_total)))


# --- Section: gradient perturbation (noisy gradient descent) ------------------------------------


def noisy_gd_sigma(n: int, steps: int, eps: float, delta: float, lipschitz: float = 1.0) -> float:
    """Per-step noise on the mean gradient so that `steps` full-batch steps are (ε, δ)-DP.

    Each step is a Gaussian mechanism with sensitivity 2L/n (replace-one) and noise σ, so its ratio is
    μ_step = 2L/(nσ). Adaptive steps compose like one Gaussian with μ = √steps · μ_step (Gaussian DP).
    """
    mu_total = gaussian_mu(eps, delta)
    return 2 * lipschitz / n * math.sqrt(steps) / mu_total


def noisy_gd(
    rng: np.random.Generator, data: LogisticData, lam: float, eps: float, delta: float, steps: int, lr: float
) -> Array:
    """Full-batch gradient descent with Gaussian noise added to every gradient; returns the last iterate."""
    sigma = noisy_gd_sigma(data.y.size, steps, eps, delta)
    w = np.zeros(data.x.shape[1])
    for _ in range(steps):
        w = w - lr * (logistic_grad(w, data, lam) + sigma * rng.standard_normal(w.size))
    return w


def noisy_gd_estimate(steps: int, n: int, d: int, lam: float, eps: float, delta: float) -> float:
    """A public estimate of noisy GD's excess risk with step 1/β, as a function of the step count T.

    Optimization: J is λ-strongly convex and β-smooth, so T noiseless steps shrink J - J* by (1 - λ/β)^T
    from at most J(0) = ln 2. Noise: at stationarity each step's noise costs d σ²/(4β) in expectation,
    and σ² grows linearly in T because the budget is split over T steps.
    """
    beta = C_LOGISTIC + lam
    sigma = noisy_gd_sigma(n, steps, eps, delta)
    return (1 - lam / beta) ** steps * math.log(2) + d * sigma**2 / (4 * beta)


def noisy_gd_steps_rule(n: int, d: int, lam: float, eps: float, delta: float) -> int:
    """The step count that minimizes the public estimate: chosen without looking at the data."""
    return min(range(1, 5001), key=lambda t: noisy_gd_estimate(t, n, d, lam, eps, delta))


def noisy_gd_lr(lam: float) -> float:
    """Step size 1/β with β = 1/4 + λ, a public bound on the curvature: the logistic curvature is ≤ 1/4."""
    return 1.0 / (C_LOGISTIC + lam)


# --- Comparison across n ----------------------------------------------------------------------


@dataclass(frozen=True)
class Comparison:
    n: int
    output_gauss: float
    output_pure: float
    objective: float
    noisy_gd: float
    nonprivate_test_acc: float
    test_acc: tuple[float, float, float, float]


def accuracy(w: Array, data: LogisticData) -> float:
    correct: NDArray[np.bool_] = np.sign(data.x @ w) == data.y
    return float(correct.mean())


def compare(rng: np.random.Generator, n: int, lam: float, eps: float, delta: float, trials: int) -> Comparison:
    test = make_logistic_data(rng, 20_000)
    risks = np.zeros((4, trials))
    accs = np.zeros((4, trials))
    base_acc = np.zeros(trials)
    steps = noisy_gd_steps_rule(n, D_RUN, lam, eps, delta)
    for t in range(trials):
        data = make_logistic_data(rng, n)
        w_star = fit_logistic(data, lam)
        base_acc[t] = accuracy(w_star, test)
        ws = (
            output_perturbation_gauss(rng, w_star, n, lam, eps, delta),
            output_perturbation_pure(rng, w_star, n, lam, eps),
            objective_perturbation(rng, data, lam, eps),
            noisy_gd(rng, data, lam, eps, delta, steps, noisy_gd_lr(lam)),
        )
        for k, w in enumerate(ws):
            risks[k, t] = excess_risk(w, w_star, data, lam)
            accs[k, t] = accuracy(w, test)
    r = risks.mean(axis=1)
    a = accs.mean(axis=1)
    return Comparison(
        n,
        float(r[0]),
        float(r[1]),
        float(r[2]),
        float(r[3]),
        float(base_acc.mean()),
        (float(a[0]), float(a[1]), float(a[2]), float(a[3])),
    )


def loglog_slope(ns: tuple[int, ...], values: list[float]) -> float:
    return float(np.polyfit(np.log(np.array(ns, dtype=float)), np.log(np.array(values)), 1)[0])


# --- Predicted excess risks (second-order approximations) --------------------------------------------


def predicted_output_gauss(hess_trace: float, n: int, lam: float, eps: float, delta: float) -> float:
    """E[J(w* + Z) - J(w*)] ≈ ½ σ² tr(H) for Z ~ N(0, σ²I)."""
    sigma = gaussian_sigma(erm_sensitivity(n, lam), eps, delta)
    return 0.5 * sigma**2 * hess_trace


def predicted_output_pure(hess_trace: float, d: int, n: int, lam: float, eps: float) -> float:
    """½ E‖b‖² tr(H)/d for b with a uniform direction: the norm noise's second moment spread over d axes."""
    return 0.5 * norm_noise_second_moment(d, erm_sensitivity(n, lam) / eps) * hess_trace / d


def predicted_objective(hess_inv_trace: float, d: int, n: int, lam: float, eps: float) -> float:
    """The minimizer moves by -H⁻¹b/n, which costs ½ (b/n)ᵀH⁻¹(b/n) ≈ E‖b‖² tr(H⁻¹)/(2 d n²)."""
    plan = objective_perturbation_plan(n, lam, eps)
    return 0.5 * norm_noise_second_moment(d, 2.0 / plan.eps_noise) * hess_inv_trace / (d * n**2)


def hessian_traces(rng: np.random.Generator, n: int, lam: float) -> tuple[float, float, float, float]:
    """tr(H), tr(H⁻¹), and the smallest and largest eigenvalue of H at w*, on one dataset of size n."""
    data = make_logistic_data(rng, n)
    h = logistic_hessian(fit_logistic(data, lam), data, lam)
    eig = np.linalg.eigvalsh(h)
    return float(eig.sum()), float((1 / eig).sum()), float(eig.min()), float(eig.max())


# --- Section: the noisy-GD step count -----------------------------------------------------------


def noisy_gd_vs_steps(
    rng: np.random.Generator, n: int, lam: float, eps: float, delta: float, steps_list: tuple[int, ...], trials: int
) -> list[float]:
    out: list[float] = []
    datasets = [make_logistic_data(rng, n) for _ in range(trials)]
    stars = [fit_logistic(d, lam) for d in datasets]
    for steps in steps_list:
        vals = [
            excess_risk(noisy_gd(rng, d, lam, eps, delta, steps, noisy_gd_lr(lam)), s, d, lam)
            for d, s in zip(datasets, stars, strict=True)
        ]
        out.append(float(np.mean(vals)))
    return out


def noiseless_gd_gap(data: LogisticData, lam: float, steps: int) -> float:
    """Excess risk of plain gradient descent with the same step size after `steps` steps (optimization error)."""
    w_star = fit_logistic(data, lam)
    w = np.zeros(data.x.shape[1])
    lr = noisy_gd_lr(lam)
    for _ in range(steps):
        w = w - lr * logistic_grad(w, data, lam)
    return excess_risk(w, w_star, data, lam)


# --- The chapter's numbers ----------------------------------------------------------------------


def print_table(rng: np.random.Generator) -> None:
    print("== Toolkit: Gaussian calibration")
    mu = gaussian_mu(EPS_RUN, DELTA_RUN)
    print(
        f"eps=1, delta=1e-5: mu={mu:.4f}  exact sigma per unit sensitivity={1 / mu:.4f}  "
        f"classic={classic_sigma(1.0, EPS_RUN, DELTA_RUN):.4f}"
    )

    print("== Output perturbation")
    ex = output_1d_example()
    print(
        f"1-D example, n=20, lambda=0.5: w*(D)={ex.w:.4f}  w*(D')={ex.w_prime:.4f}  "
        f"gap={abs(ex.w - ex.w_prime):.4f}  bound={ex.sensitivity:.4f}  sigma={ex.sigma:.4f}"
    )
    for n in (1_000, 10_000, 100_000):
        s = erm_sensitivity(n, LAM_RUN)
        print(
            f"n={n}: sensitivity 2/(lam n)={s:.4g}  Gaussian sigma={gaussian_sigma(s, EPS_RUN, DELTA_RUN):.4g}  "
            f"norm-noise mean length d*s/eps={D_RUN * s / EPS_RUN:.4g}"
        )
    data = make_logistic_data(rng, 10_000)
    w_star = fit_logistic(data, LAM_RUN)
    print(f"n=10000: |w*|={float(np.linalg.norm(w_star)):.3f}")
    for tol in (1e-6, 1e-4, 1e-3):
        print(
            f"grad tol {tol:g}: sensitivity of an approximate minimizer at n=10000 = "
            f"{approximate_minimizer_sensitivity(10_000, LAM_RUN, tol):.4g}  "
            f"(exact minimizer {erm_sensitivity(10_000, LAM_RUN):.4g})"
        )
    tr_h, tr_hinv, eig_min, eig_max = hessian_traces(rng, 10_000, LAM_RUN)
    print(
        f"Hessian at w*, n=10000: eigenvalues in [{eig_min:.4f}, {eig_max:.4f}]  tr(H)={tr_h:.4f}  "
        f"tr(H^-1)={tr_hinv:.1f}"
    )

    print("== Objective perturbation")
    for n in (100, 1_000, 10_000, 100_000):
        plan = objective_perturbation_plan(n, LAM_RUN, EPS_RUN)
        print(f"n={n}: eps'={plan.eps_noise:.4f}  extra lambda={plan.extra_lam:.4g}")
    grid = np.linspace(-30.0, 30.0, 6001)
    chk = worst_case_1d(rng, 5, 0.1, EPS_RUN, 500, grid)
    print(
        f"1-D check, n=5, lambda=0.1, 500 datasets: largest |ln p/p'|={chk.total:.4f} (eps=1)  "
        f"noise part {chk.noise:.4f} (budget eps'={chk.noise_budget:.4f})  "
        f"Jacobian part {chk.jacobian:.4f} (budget {chk.jacobian_budget:.4f})"
    )

    print("== Noisy gradient descent")
    for n in NS_RUN:
        steps = noisy_gd_steps_rule(n, D_RUN, LAM_RUN, EPS_RUN, DELTA_RUN)
        print(
            f"n={n}: steps T={steps}  per-step sigma on the mean gradient="
            f"{noisy_gd_sigma(n, steps, EPS_RUN, DELTA_RUN):.4g}  "
            f"estimate={noisy_gd_estimate(steps, n, D_RUN, LAM_RUN, EPS_RUN, DELTA_RUN):.3e}"
        )
    data = make_logistic_data(rng, 10_000)
    for steps in (20, 100, 400):
        print(f"noiseless GD, n=10000, {steps} steps: excess risk={noiseless_gd_gap(data, LAM_RUN, steps):.3e}")

    print("== Comparison (mean over trials)")
    results: list[Comparison] = []
    for n in NS_RUN:
        c = compare(rng, n, LAM_RUN, EPS_RUN, DELTA_RUN, TRIALS_RUN)
        results.append(c)
        print(
            f"n={n}: output-Gauss={c.output_gauss:.3e}  output-pure={c.output_pure:.3e}  "
            f"objective={c.objective:.3e}  noisy-GD={c.noisy_gd:.3e}"
        )
        print(
            f"n={n}: test accuracy nonprivate={c.nonprivate_test_acc:.3f}  output-Gauss={c.test_acc[0]:.3f}  "
            f"output-pure={c.test_acc[1]:.3f}  objective={c.test_acc[2]:.3f}  noisy-GD={c.test_acc[3]:.3f}"
        )
    for name, values in (
        ("output-Gauss", [c.output_gauss for c in results]),
        ("output-pure", [c.output_pure for c in results]),
        ("objective", [c.objective for c in results]),
        ("noisy-GD", [c.noisy_gd for c in results]),
    ):
        print(f"log-log slope of excess risk against n, {name}: {loglog_slope(NS_RUN, values):.2f}")

    print("== Predicted against measured, n=10000")
    mid = next(c for c in results if c.n == 10_000)
    print(
        f"output-Gauss: predicted {predicted_output_gauss(tr_h, 10_000, LAM_RUN, EPS_RUN, DELTA_RUN):.3e}  "
        f"measured {mid.output_gauss:.3e}"
    )
    print(
        f"output-pure: predicted {predicted_output_pure(tr_h, D_RUN, 10_000, LAM_RUN, EPS_RUN):.3e}  "
        f"measured {mid.output_pure:.3e}"
    )
    print(
        f"objective: predicted {predicted_objective(tr_hinv, D_RUN, 10_000, LAM_RUN, EPS_RUN):.3e}  "
        f"measured {mid.objective:.3e}"
    )

    print("== Noisy GD against the step count, n=10000")
    steps_list = (5, 10, 20, 50, 100, 200, 500, 1000)
    vals = noisy_gd_vs_steps(rng, 10_000, LAM_RUN, EPS_RUN, DELTA_RUN, steps_list, 10)
    print("  ".join(f"T={s}: {v:.3e}" for s, v in zip(steps_list, vals, strict=True)))


# --- Figures --------------------------------------------------------------------------------


@dataclass(frozen=True)
class Output1D:
    data: LogisticData
    data_prime: LogisticData
    lam: float
    w: float
    w_prime: float
    sensitivity: float
    sigma: float


def output_1d_example() -> Output1D:
    """A one-dimensional regularized logistic regression on 20 examples, and its neighbour."""
    rng = np.random.default_rng(3)
    n, lam = 20, 0.5
    xs = rng.uniform(0.2, 1.0, n)
    ys = np.where(rng.random(n) < 0.7, 1.0, -1.0)
    x2, y2 = xs.copy(), ys.copy()
    xs[-1] = 1.0
    x2[-1], y2[-1] = 1.0, -ys[-1]
    data, data2 = LogisticData(xs[:, None], ys), LogisticData(x2[:, None], y2)
    sens = erm_sensitivity(n, lam)
    return Output1D(
        data,
        data2,
        lam,
        float(fit_logistic(data, lam)[0]),
        float(fit_logistic(data2, lam)[0]),
        sens,
        gaussian_sigma(sens, 1.0, 1e-5),
    )


def plot_output_1d() -> Figure:
    """Two neighbouring regularized objectives in one dimension, their minimizers and the Gaussian noise around them."""
    ex = output_1d_example()
    data, data2, lam, w1, w2, sens, sigma = ex.data, ex.data_prime, ex.lam, ex.w, ex.w_prime, ex.sensitivity, ex.sigma
    grid = np.linspace(-1.0, 2.0, 400)
    j1 = [logistic_objective(np.array([g]), data, lam) for g in grid]
    j2 = [logistic_objective(np.array([g]), data2, lam) for g in grid]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    ax = axes[0]
    ax.plot(grid, j1, color="black", label="J(w; D)")
    ax.plot(grid, j2, color="tab:red", ls="--", label="J(w; D')")
    ax.axvline(w1, color="black", lw=0.8)
    ax.axvline(w2, color="tab:red", lw=0.8, ls="--")
    ax.set_xlabel(r"$w$")
    ax.set_title(f"Minimizers {abs(w1 - w2):.3f} apart; bound $2L/(\\lambda n)$ = {sens:.3f}")
    ax.legend()
    ax = axes[1]
    g2 = np.linspace(min(w1, w2) - 4 * sigma, max(w1, w2) + 4 * sigma, 400)
    for w, color, ls, lab in ((w1, "black", "-", "w*(D) + Z"), (w2, "tab:red", "--", "w*(D') + Z")):
        ax.plot(
            g2, np.exp(-0.5 * ((g2 - w) / sigma) ** 2) / (sigma * math.sqrt(2 * math.pi)), color=color, ls=ls, label=lab
        )
    ax.set_xlabel("released weight")
    ax.set_title(r"Gaussian noise for $\varepsilon = 1$, $\delta = 10^{-5}$")
    ax.legend()
    return fig


def plot_objective_density() -> Figure:
    """The density of the objective-perturbation output under D and D', and the log ratio inside ±ε."""
    rng = np.random.default_rng(4)
    n, lam, eps = 5, 0.1, 1.0
    plan = objective_perturbation_plan(n, lam, eps)
    xs = rng.uniform(0.3, 1.0, n)
    ys = np.where(rng.random(n) < 0.6, 1.0, -1.0)
    xs[-1], ys[-1] = 1.0, 1.0
    grid = np.linspace(-12.0, 12.0, 2001)
    dens = objective_density_1d(xs, ys, -1.0, 1.0, lam + plan.extra_lam, plan.eps_noise, grid)
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    ax = axes[0]
    ax.plot(grid, dens.p, color="black", label="output density, D")
    ax.plot(grid, dens.p_prime, color="tab:red", ls="--", label="output density, D'")
    ax.set_xlabel("released weight")
    ax.legend()
    ax.set_title("Objective perturbation, one dimension, $n = 5$")
    ax = axes[1]
    ax.plot(grid, np.log(dens.p / dens.p_prime), color="black", label="ln of the density ratio, D against D'")
    ax.axhline(eps, color="tab:gray", ls=":", label=r"$\pm\varepsilon$")
    ax.axhline(-eps, color="tab:gray", ls=":")
    ax.set_xlabel("released weight")
    ax.set_ylim(-1.3 * eps, 1.3 * eps)
    ax.legend()
    ax.set_title("Privacy loss stays within $\\pm\\varepsilon$")
    return fig


def plot_comparison(rng: np.random.Generator) -> Figure:
    """Excess empirical risk against n for the four private methods, log-log."""
    ns = (1_000, 3_000, 10_000, 30_000, 100_000)
    results = [compare(rng, n, LAM_RUN, EPS_RUN, DELTA_RUN, 10) for n in ns]
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    styles = (
        ("output perturbation, Gaussian $(\\varepsilon,\\delta)$", "tab:blue", "o", [c.output_gauss for c in results]),
        ("output perturbation, pure $\\varepsilon$", "tab:cyan", "s", [c.output_pure for c in results]),
        ("objective perturbation, pure $\\varepsilon$", "tab:red", "^", [c.objective for c in results]),
        ("noisy gradient descent $(\\varepsilon,\\delta)$", "black", "D", [c.noisy_gd for c in results]),
    )
    for label, color, marker, vals in styles:
        ax.loglog(ns, vals, marker=marker, color=color, label=label)
    ref = np.array(ns, dtype=float)
    ax.loglog(ref, results[0].output_gauss * (ref / ref[0]) ** -2, color="tab:gray", ls=":", label=r"slope $-2$")
    ax.set_xlabel("training examples $n$")
    ax.set_ylabel("excess risk J(released) - J(w*)")
    ax.set_title(r"Logistic regression, $d = 10$, $\lambda = 0.01$, $\varepsilon = 1$")
    ax.legend(fontsize=8)
    return fig


def plot_noisy_gd_steps(rng: np.random.Generator) -> Figure:
    """Excess risk of noisy GD against the number of steps: optimization error falls, noise rises."""
    steps_list = (2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000)
    n = 10_000
    vals = noisy_gd_vs_steps(rng, n, LAM_RUN, EPS_RUN, DELTA_RUN, steps_list, 5)
    data = make_logistic_data(rng, n)
    opt = [noiseless_gd_gap(data, LAM_RUN, s) for s in steps_list]
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    ax.loglog(steps_list, vals, marker="o", color="black", label="noisy GD, $(1, 10^{-5})$-DP")
    ax.loglog(steps_list, opt, marker="s", color="tab:gray", ls="--", label="the same steps without noise")
    ax.set_ylim(1e-6, 0.2)
    ax.set_xlabel("number of steps $T$")
    ax.set_ylabel(r"excess risk $J(w_T) - J(w^*)$")
    ax.set_title("$n = 10{,}000$: more steps, more noise per step")
    ax.legend()
    return fig


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=FIGURES, help="directory for the figures")
    args = parser.parse_args()
    seed: int = args.seed
    out: Path = args.out

    print_table(np.random.default_rng(seed))

    out.mkdir(parents=True, exist_ok=True)
    fig_rng = np.random.default_rng(seed + 1)
    figures = {
        "erm_output_1d.svg": plot_output_1d(),
        "erm_objective_density.svg": plot_objective_density(),
        "erm_comparison.svg": plot_comparison(fig_rng),
        "erm_noisy_gd_steps.svg": plot_noisy_gd_steps(fig_rng),
    }
    for name, fig in figures.items():
        fig.tight_layout()
        fig.savefig(out / name, metadata={"Date": None})  # no timestamp, so output is reproducible
        plt.close(fig)
        print(f"saved {out / name}")


if __name__ == "__main__":
    main()
