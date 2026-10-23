"""Check the derivations of the appendix on private learning and tuning, and draw its figures.

    uv run python code/appendix-derivations/apx_learning.py
    uv run python code/appendix-derivations/apx_learning.py --seed 1 --out /tmp/figures

Prints every number quoted in book-full/appendix-derivations/13-private-learning.md (seed 0),
then saves the page's figures.
"""

import argparse
import itertools
import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
matplotlib.rcParams["svg.hashsalt"] = "apx_learning"  # stable element ids, so reruns don't churn the SVGs
matplotlib.rcParams["svg.fonttype"] = "none"  # keep text as text: smaller, diffable files

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from numpy.typing import NDArray

# Figures are build output: _build/figures/<part>/, which the build copies into each book.
FIGURES = Path(__file__).resolve().parents[2] / "_build" / "figures" / Path(__file__).parent.name

Array = NDArray[np.float64]


# --- Shared helpers ---------------------------------------------------------------------------


def normal_cdf(x: float) -> float:
    """Φ(x); erfc keeps full relative precision in the lower tail."""
    return 0.5 * math.erfc(-x / math.sqrt(2))


def bisect(f: Callable[[float], bool], lo: float, hi: float, steps: int = 200) -> float:
    """Smallest x in [lo, hi] with f(x) true, for f false-then-true."""
    for _ in range(steps):
        mid = (lo + hi) / 2
        if f(mid):
            hi = mid
        else:
            lo = mid
    return hi


def logsumexp(v: Sequence[float]) -> float:
    top = max(v)
    return top + math.log(sum(math.exp(x - top) for x in v))


def renyi(p: Sequence[float], q: Sequence[float], alpha: float) -> float:
    """D_α(p ‖ q) for discrete distributions, in log space so large orders do not overflow."""
    terms = [alpha * math.log(a) + (1 - alpha) * math.log(b) for a, b in zip(p, q, strict=True) if a > 0]
    return logsumexp(terms) / (alpha - 1)


# --- Section: optimal rates of private convex ERM -----------------------------------------------

D_RUN = 10  # the running example of the DP-ERM chapter: dimension,
LAM_RUN = 0.01  # ridge λ,
EPS_RUN, DELTA_RUN = 1.0, 1e-5  # budget,
C_LOGISTIC = 0.25  # and the logistic loss's largest curvature in the margin
W_TRUE: Array = np.full(D_RUN, 5.0 / math.sqrt(D_RUN))  # ‖w_true‖ = 5


def gaussian_delta(eps: float, mu: float) -> float:
    """Exact δ(ε) of a Gaussian mechanism with sensitivity-to-noise ratio μ (Balle & Wang 2018)."""
    return normal_cdf(mu / 2 - eps / mu) - math.exp(eps) * normal_cdf(-mu / 2 - eps / mu)


def gaussian_mu(eps: float, delta: float) -> float:
    """The largest ratio μ = Δ/σ for which the Gaussian mechanism is (ε, δ)-DP."""
    return bisect(lambda m: gaussian_delta(eps, m) > delta, 1e-9, 100.0) * (1 - 1e-12)


def sgd_weighted_bound(steps: int, g2: float, lam: float) -> float:
    """Part 1: projected SGD with steps 2/(λ(t+1)) and the t-weighted average, error ≤ 2G²/(λ(T+1))."""
    return 2 * g2 / (lam * (steps + 1))


def sgd_uniform_bound(steps: int, g2: float, lam: float) -> float:
    """Part 1: steps 1/(λt) and the uniform average, error ≤ G²(1 + ln T)/(2λT)."""
    return g2 * (1 + math.log(steps)) / (2 * lam * steps)


def strong_noise_sigma(n: int, steps: int, mu: float, lip: float = 1.0) -> float:
    """Per-step noise on the mean gradient: T steps of sensitivity 2L/n compose to ratio μ."""
    return 2 * lip * math.sqrt(steps) / (n * mu)


def strong_steps_rule(n: int, d: int, mu: float) -> int:
    """T = ⌈n²μ²/d⌉, the point where the optimization term falls below the noise term."""
    return math.ceil(n * n * mu * mu / d)


def strong_bound_at(n: int, d: int, lam: float, mu: float, steps: int, lip: float = 1.0) -> float:
    """2G²/(λ(T+1)) with G² = (L + λR)² + dσ² and R = L/λ, so L + λR = 2L."""
    g2 = (2 * lip) ** 2 + d * strong_noise_sigma(n, steps, mu, lip) ** 2
    return sgd_weighted_bound(steps, g2, lam)


def strong_upper(n: int, d: int, lam: float, mu: float, lip: float = 1.0) -> float:
    """The closed form 16 L² d/(λ n² μ²), valid once T ≥ n²μ²/d."""
    return 16 * lip**2 * d / (lam * n * n * mu * mu)


def fingerprint_k(d: int, eps: float, delta: float) -> float:
    """K = √d/((e^ε - 1) + 4δ√d): the fingerprinting bound says n ≥ (2/3 - 2α')K."""
    return math.sqrt(d) / (math.expm1(eps) + 4 * delta * math.sqrt(d))


def strong_lower(n: int, d: int, lam: float, eps: float, delta: float, lip: float = 1.0) -> float:
    """max over m of (L²/(2λ)) [(m/n)(1/3 - m/(2K))]²: m real records padded with n - m zeros."""
    k = fingerprint_k(d, eps, delta)
    best = max((m / n) * (1 / 3 - m / (2 * k)) for m in range(1, min(n, math.ceil(k)) + 1))
    return lip**2 * max(best, 0.0) ** 2 / (2 * lam)


def convex_lower(n: int, d: int, radius: float, eps: float, delta: float, lip: float = 1.0) -> float:
    """max over m of (m L R/(16 n)) (2/3 - m/K)², the ridge chosen as λ = mL/(2nR)."""
    k = fingerprint_k(d, eps, delta)
    best = max(m * max(2 / 3 - m / k, 0.0) ** 2 for m in range(1, min(n, math.ceil(k)) + 1))
    return lip * radius * best / (16 * n)


def sigmoid(s: Array) -> Array:
    return 0.5 * (1.0 + np.tanh(0.5 * s))


@dataclass(frozen=True)
class LogisticData:
    x: Array  # (n, d), unit rows
    y: Array  # (n,), labels ±1


def make_logistic_data(rng: np.random.Generator, n: int) -> LogisticData:
    """The running example: x uniform on the unit sphere, y = +1 with probability σ(<w_true, x>)."""
    z = rng.standard_normal((n, D_RUN))
    x = z / np.linalg.norm(z, axis=1, keepdims=True)
    y = np.where(rng.random(n) < sigmoid(x @ W_TRUE), 1.0, -1.0)
    return LogisticData(x, y)


def objective(w: Array, data: LogisticData, lam: float, b: Array | None = None) -> float:
    """J(w) = mean logistic loss + (λ/2)‖w‖², plus <b, w>/n for objective perturbation."""
    value = float(np.mean(np.logaddexp(0.0, -data.y * (data.x @ w)))) + 0.5 * lam * float(w @ w)
    if b is not None:
        value += float(b @ w) / data.y.size
    return value


def gradient(w: Array, data: LogisticData, lam: float, b: Array | None = None) -> Array:
    margins = data.y * (data.x @ w)
    g: Array = (-(data.y * sigmoid(-margins)) @ data.x) / data.y.size + lam * w
    if b is not None:
        g = g + b / data.y.size
    return g


def hessian(w: Array, data: LogisticData, lam: float) -> Array:
    u = sigmoid(data.y * (data.x @ w))
    return (data.x * (u * (1 - u))[:, None]).T @ data.x / data.y.size + lam * np.eye(w.size)


def fit(data: LogisticData, lam: float, b: Array | None = None) -> Array:
    """The exact minimizer by Newton's method (the objective is smooth and strongly convex)."""
    w = np.zeros(data.x.shape[1])
    for _ in range(60):
        g = gradient(w, data, lam, b)
        if float(np.linalg.norm(g)) < 1e-13:
            break
        w = w - np.linalg.solve(hessian(w, data, lam), g)
    return w


def noisy_gd_weighted(rng: np.random.Generator, data: LogisticData, lam: float, mu: float, steps: int) -> Array:
    """Noisy projected GD with steps 2/(λ(t+1)) on the ball of radius L/λ, t-weighted average of the iterates."""
    n, d = data.x.shape
    radius = 1.0 / lam
    sigma = strong_noise_sigma(n, steps, mu)
    noise: Array = sigma * rng.standard_normal((steps, d))
    w: Array = np.zeros(d)
    avg: Array = np.zeros(d)
    for t in range(1, steps + 1):
        w = w - 2 / (lam * (t + 1)) * (gradient(w, data, lam) + noise[t - 1])
        length = float(np.linalg.norm(w))
        if length > radius:
            w = w * (radius / length)
        avg += t * w
    return avg * 2 / (steps * (steps + 1))


def measured_noisy_gd(rng: np.random.Generator, n: int, mu: float, trials: int) -> float:
    total = 0.0
    for _ in range(trials):
        data = make_logistic_data(rng, n)
        w_star = fit(data, LAM_RUN)
        w = noisy_gd_weighted(rng, data, LAM_RUN, mu, strong_steps_rule(n, D_RUN, mu))
        total += objective(w, data, LAM_RUN) - objective(w_star, data, LAM_RUN)
    return total / trials


def gamma_norm_noise(rng: np.random.Generator, d: int, scale: float) -> Array:
    """Density ∝ exp(-‖b‖/scale): uniform direction, Gamma(d, scale) length."""
    z = rng.standard_normal(d)
    return z / np.linalg.norm(z) * rng.gamma(d, scale)


@dataclass(frozen=True)
class Perturbation:
    lower: float  # rigorous lower bound
    approx: float  # the chapter's second-order value
    measured: float  # mean over releases
    upper: float  # rigorous upper bound


def output_perturbation_check(rng: np.random.Generator, data: LogisticData, mu: float, trials: int) -> Perturbation:
    """Output perturbation with Gaussian noise: (λ/2)E‖Z‖² ≤ E[excess] ≤ (β/2)E‖Z‖², E‖Z‖² = dσ²."""
    n, d = data.x.shape
    w_star = fit(data, LAM_RUN)
    sigma = 2 / (LAM_RUN * n * mu)
    beta = C_LOGISTIC + LAM_RUN
    j_star = objective(w_star, data, LAM_RUN)
    draws: Array = sigma * rng.standard_normal((trials, d))
    vals = [objective(w_star + z, data, LAM_RUN) - j_star for z in draws]
    tr_h = float(np.trace(hessian(w_star, data, LAM_RUN)))
    m2 = d * sigma**2
    return Perturbation(LAM_RUN * m2 / 2, sigma**2 * tr_h / 2, float(np.mean(vals)), beta * m2 / 2)


def objective_perturbation_check(rng: np.random.Generator, data: LogisticData, eps: float, trials: int) -> Perturbation:
    """Objective perturbation: λE‖b‖²/(2β²n²) ≤ E[excess] ≤ E‖b‖²/(2λn²), E‖b‖² = d(d+1)(2/ε')²."""
    n, d = data.x.shape
    eps_noise = eps - 2 * math.log1p(C_LOGISTIC / (n * LAM_RUN))
    scale = 2 / eps_noise
    w_star = fit(data, LAM_RUN)
    j_star = objective(w_star, data, LAM_RUN)
    vals: list[float] = []
    for _ in range(trials):
        b = gamma_norm_noise(rng, d, scale)
        vals.append(objective(fit(data, LAM_RUN, b), data, LAM_RUN) - j_star)
    m2 = d * (d + 1) * scale**2
    beta = C_LOGISTIC + LAM_RUN
    tr_hinv = float(np.trace(np.linalg.inv(hessian(w_star, data, LAM_RUN))))
    return Perturbation(
        LAM_RUN * m2 / (2 * beta**2 * n * n),
        m2 * tr_hinv / (2 * d * n * n),
        float(np.mean(vals)),
        m2 / (2 * LAM_RUN * n * n),
    )


def objective_inequalities_hold(rng: np.random.Generator, data: LogisticData, trials: int) -> bool:
    """Per draw: ‖b‖/(nβ) ≤ ‖w̃ - w*‖ ≤ ‖b‖/(nλ) and excess ≤ ‖b‖²/(2λn²)."""
    n = data.y.size
    beta = C_LOGISTIC + LAM_RUN
    w_star = fit(data, LAM_RUN)
    j_star = objective(w_star, data, LAM_RUN)
    ok = True
    for _ in range(trials):
        b = gamma_norm_noise(rng, D_RUN, 2.0)
        w = fit(data, LAM_RUN, b)
        dist, nb = float(np.linalg.norm(w - w_star)), float(np.linalg.norm(b))
        excess = objective(w, data, LAM_RUN) - j_star
        ok = ok and nb / (n * beta) <= dist * (1 + 1e-9) and dist <= nb / (n * LAM_RUN) * (1 + 1e-9)
        ok = ok and excess <= nb**2 / (2 * LAM_RUN * n * n) * (1 + 1e-9)
    return ok


# --- Section: PATE's Gaussian data-dependent bound --------------------------------------------

SIGMA_PATE = 40.0  # the chapter's example: GNMax noise and order
ALPHA_PATE = 8.0


@dataclass(frozen=True)
class PateOrders:
    mu1: float
    mu2: float
    eps1: float
    eps2: float


def pate_orders(qt: float, sigma: float) -> PateOrders:
    """Proposition 10 of Papernot et al. (2018): μ2 = σ√ln(1/q̃), μ1 = μ2 + 1, ε_i = μ_i/σ²."""
    mu2 = sigma * math.sqrt(math.log(1 / qt))
    return PateOrders(mu2 + 1, mu2, (mu2 + 1) / sigma**2, mu2 / sigma**2)


def pate_p_bound(q: float, o: PateOrders) -> float:
    """Order-μ2 RDP on the event "not i*": p ≤ (q e^{ε2})^{(μ2-1)/μ2}."""
    return math.exp((o.mu2 - 1) / o.mu2 * (math.log(q) + o.eps2))


def pate_log_terms(alpha: float, q: float, o: PateOrders) -> tuple[float, float]:
    """ln of (1 - q)A^{α-1} and q B^{α-1}, the likely output's term and the other outputs' term."""
    log_a = math.log1p(-q) - math.log1p(-pate_p_bound(q, o))
    log_b = o.eps1 - math.log(q) / (o.mu1 - 1)
    return math.log1p(-q) + (alpha - 1) * log_a, math.log(q) + (alpha - 1) * log_b


def pate_value(alpha: float, q: float, o: PateOrders) -> float:
    """(1/(α-1)) ln[(1 - q)A(q)^{α-1} + q B(q)^{α-1}] with the orders held fixed."""
    t1, t2 = pate_log_terms(alpha, q, o)
    return logsumexp([t1, t2]) / (alpha - 1)


def pate_condition(qt: float, sigma: float) -> bool:
    """The condition of Theorem 6: q̃ ≤ e^{(μ2-1)ε2}/(μ1/(μ1-1) · μ2/(μ2-1))^{μ2}."""
    o = pate_orders(qt, sigma)
    if o.mu2 <= 1:
        return False
    return math.log(qt) <= (o.mu2 - 1) * o.eps2 - o.mu2 * math.log(o.mu1 / (o.mu1 - 1) * o.mu2 / (o.mu2 - 1))


def pate_valid(alpha: float, qt: float, sigma: float) -> bool:
    """The bound's own requirements: μ2 > 1, α ≤ μ1 and q̃ e^{ε2} < 1."""
    o = pate_orders(qt, sigma)
    return o.mu2 > 1 and alpha <= o.mu1 and qt * math.exp(o.eps2) < 1


def pate_grid(qt: float, points: int = 400) -> list[float]:
    """Failure probabilities q in (0, q̃], geometric near 0 so small q are resolved."""
    return [qt * 10 ** (-12 * (1 - j / points)) for j in range(points + 1)]


def pate_rigorous(alpha: float, qt: float, sigma: float) -> float:
    """The bound with only q ≤ q̃ known: the largest value over q in (0, q̃], orders set from q̃."""
    o = pate_orders(qt, sigma)
    return max(pate_value(alpha, q, o) for q in pate_grid(qt))


def pate_increasing(alpha: float, qt: float, sigma: float) -> bool:
    o = pate_orders(qt, sigma)
    vals = [pate_value(alpha, q, o) for q in pate_grid(qt)]
    return all(b >= a - 1e-15 for a, b in itertools.pairwise(vals))


def two_label_q(gap: float, sigma: float) -> float:
    """GNMax on two labels: the difference of the noises is N(0, 2σ²), so P(not i*) = ½erfc(g/(2σ))."""
    return 0.5 * math.erfc(gap / (2 * sigma))


def two_label_exact(alpha: float, gap: float, sigma: float) -> float:
    """D_α(M(D)‖M(D')) on two labels, worst over the neighbours with gap g - 2 and g + 2."""
    p = two_label_q(gap, sigma)
    out = 0.0
    for g2 in (gap - 2, gap + 2):
        pp = two_label_q(g2, sigma)
        out = max(out, renyi([1 - p, p], [1 - pp, pp], alpha))
    return out


def gnmax_probs(votes: Sequence[float], sigma: float) -> list[float]:
    """P(label i wins) for GNMax: ∫ φ(t) Π_{j≠i} Φ((n_i - n_j)/σ + t) dt, by the trapezoid rule."""
    t = np.linspace(-10.0, 10.0, 8001)
    phi = np.exp(-(t**2) / 2) / math.sqrt(2 * math.pi)
    erfc = np.vectorize(math.erfc)
    out: list[float] = []
    for i, ni in enumerate(votes):
        integrand = phi.copy()
        for j, nj in enumerate(votes):
            if j != i:
                integrand = integrand * 0.5 * erfc(-((ni - nj) / sigma + t) / math.sqrt(2))
        out.append(float(np.trapezoid(integrand, t)))
    return out


def three_label_check(votes: tuple[int, int, int], alpha: float, sigma: float) -> tuple[float, float, float]:
    """(exact worst D_α over the six one-teacher moves, q̃ of eq-pate-qtilde, Theorem 6 bound at q̃)."""
    p = gnmax_probs([float(v) for v in votes], sigma)
    worst = 0.0
    for a, b in itertools.permutations(range(3), 2):
        moved = [float(v) for v in votes]
        moved[a] -= 1
        moved[b] += 1
        worst = max(worst, renyi(p, gnmax_probs(moved, sigma), alpha))
    top = max(range(3), key=lambda i: votes[i])
    qt = sum(two_label_q(votes[top] - votes[i], sigma) for i in range(3) if i != top)
    return worst, qt, pate_rigorous(alpha, qt, sigma)


def pure_value(alpha: float, q: float, eps0: float) -> float:
    """The bracket of the pure-DP form, (1-q)^α(1 - e^{ε0}q)^{1-α} + q e^{(α-1)ε0}."""
    return (1 - q) ** alpha * (1 - math.exp(eps0) * q) ** (1 - alpha) + q * math.exp((alpha - 1) * eps0)


def pure_convex_and_increasing(alpha: float, eps0: float, points: int = 2000) -> bool:
    """Second differences ≥ 0 and the slope at 0, e^{(α-1)ε0} + (α-1)e^{ε0} - α, ≥ 0."""
    qmax = math.exp(-eps0) * (1 - 1e-6)
    qs = [qmax * j / points for j in range(points + 1)]
    v = [pure_value(alpha, q, eps0) for q in qs]
    second = all(v[i - 1] - 2 * v[i] + v[i + 1] >= -1e-12 * abs(v[i]) for i in range(1, points))
    slope0 = math.exp((alpha - 1) * eps0) + (alpha - 1) * math.exp(eps0) - alpha
    return second and slope0 >= 0 and all(b >= a for a, b in itertools.pairwise(v))


# --- Section: the truncated negative binomial number of runs --------------------------------


def tnb_mean(eta: float, gamma: float) -> float:
    if eta == 0:
        return (1 / gamma - 1) / math.log(1 / gamma)
    return eta * (1 - gamma) / (gamma * (1 - gamma**eta))


def tnb_gamma(eta: float, mean: float) -> float:
    """The γ for which the truncated negative binomial has the given mean (the mean falls as γ grows)."""
    return bisect(lambda g: tnb_mean(eta, g) <= mean, 1e-15, 1 - 1e-12)


def tnb_pgf(eta: float, gamma: float) -> Callable[[float], float]:
    if eta == 0:
        return lambda x: math.log(1 - (1 - gamma) * x) / math.log(gamma)
    return lambda x: ((1 - (1 - gamma) * x) ** (-eta) - 1) / (gamma ** (-eta) - 1)


def tnb_fprime(eta: float, gamma: float, x: float) -> float:
    """f'(x) = E[K] γ^{η+1} a(x)^{-(η+1)} with a(x) = 1 - (1 - γ)x."""
    return tnb_mean(eta, gamma) * gamma ** (eta + 1) * (1 - (1 - gamma) * x) ** (-(eta + 1))


def best_of_pmf(pmf: Sequence[float], pgf: Callable[[float], float]) -> list[float]:
    """P[best = y_i] = f(P[run no better than y_i]) - f(P[run worse]), outcomes listed best first."""
    return [pgf(sum(pmf[i:])) - pgf(sum(pmf[i + 1 :])) for i in range(len(pmf))]


def rr_pair(eps: float) -> tuple[list[float], list[float]]:
    """Randomized response, best outcome first: (1, e^ε)/(1 + e^ε) on D and (e^ε, 1)/(1 + e^ε) on D'."""
    z = 1 + math.exp(eps)
    return [1 / z, math.exp(eps) / z], [math.exp(eps) / z, 1 / z]


def rr_rdp(lam: float, eps: float) -> float:
    p, pp = rr_pair(eps)
    return renyi(p, pp, lam)


@dataclass(frozen=True)
class TnbTerms:
    one_run: float  # ε(λ)
    tail: float  # (1 + η)(1 - 1/λ̂)ε(λ̂)
    gamma: float  # (1 + η)ln(1/γ)/λ̂
    mean: float  # ln E[K]/(λ - 1)

    @property
    def total(self) -> float:
        return self.one_run + self.tail + self.gamma + self.mean


def tnb_terms(eps_of: Callable[[float], float], eta: float, gamma: float, lam: float, lam_hat: float) -> TnbTerms:
    return TnbTerms(
        eps_of(lam),
        (1 + eta) * (1 - 1 / lam_hat) * eps_of(lam_hat),
        (1 + eta) * math.log(1 / gamma) / lam_hat,
        math.log(tnb_mean(eta, gamma)) / (lam - 1),
    )


LAM_HATS: list[float] = [1 + x / 20 for x in range(1, 2000)]


def tnb_best(eps_of: Callable[[float], float], eta: float, gamma: float, lam: float) -> tuple[float, float]:
    """(best λ̂ on a grid, bound there)."""
    best = min(LAM_HATS, key=lambda h: tnb_terms(eps_of, eta, gamma, lam, h).total)
    return best, tnb_terms(eps_of, eta, gamma, lam, best).total


def tnb_pure_limit(eps: float, eta: float, gamma: float, lam: float) -> float:
    """Theorem 2 as λ̂ → ∞ for an ε-DP run: ε(λ) + (1 + η)ε + ln E[K]/(λ - 1)."""
    return rr_rdp(lam, eps) + (1 + eta) * eps + math.log(tnb_mean(eta, gamma)) / (lam - 1)


def tnb_exact_rr(eps: float, eta: float, gamma: float, lam: float) -> float:
    """Exact D_λ of the best-of release on randomized response, the worse of the two directions."""
    p, pp = rr_pair(eps)
    a, aa = best_of_pmf(p, tnb_pgf(eta, gamma)), best_of_pmf(pp, tnb_pgf(eta, gamma))
    return max(renyi(a, aa, lam), renyi(aa, a, lam))


def binary_rdp_ok(r: float, rp: float, lam_hat: float, eps_hat: float) -> bool:
    """Whether rejection probabilities (r, r') obey order-λ̂ RDP on the binary test, D_λ̂(r'‖r) ≤ ε̂."""
    s = rp**lam_hat * r ** (1 - lam_hat) + (1 - rp) ** lam_hat * (1 - r) ** (1 - lam_hat)
    return math.log(s) / (lam_hat - 1) <= eps_hat


def tnb_ratio(r: float, rp: float, gamma: float, lam: float) -> float:
    """γ a'^{λ-1}/a^λ with a = γ + (1-γ)r, the quantity raised to 1 + η."""
    a, ap = gamma + (1 - gamma) * r, gamma + (1 - gamma) * rp
    return gamma * ap ** (lam - 1) / a**lam


def tnb_chain_holds(rng: np.random.Generator, trials: int) -> bool:
    """Random (γ, λ, λ̂, ε̂, r, r') obeying the binary constraint: every step of the chain holds."""
    ok = True
    for _ in range(trials):
        gamma = 10 ** rng.uniform(-6, -0.5)
        lam, lam_hat, eps_hat = 1 + 20 * rng.random(), 1 + 30 * rng.random(), 3 * rng.random()
        r = 10 ** rng.uniform(-8, 0)
        s = 1 - 1 / lam_hat
        rp_max = min(1.0, math.exp(s * eps_hat) * r**s)
        rp = rp_max * rng.random()
        if not binary_rdp_ok(r, rp, lam_hat, eps_hat):
            continue
        a, ap = gamma + (1 - gamma) * r, gamma + (1 - gamma) * rp
        ok = ok and ap <= math.exp(s * eps_hat) * a**s * (1 + 1e-12)
        bound = math.exp((lam - 1) * s * eps_hat) * gamma ** (-(lam - 1) / lam_hat)
        ok = ok and tnb_ratio(r, rp, gamma, lam) <= bound * (1 + 1e-9)
    return ok


def zcdp_closed_form(rho: float, eta: float, gamma: float) -> float:
    """Corollary 4: 2√(ρ ln E[K]) + 2(1+η)√(ρ ln(1/γ)) - ηρ, for orders up to 1 + √(ln E[K]/ρ)."""
    return (
        2 * math.sqrt(rho * math.log(tnb_mean(eta, gamma)))
        + 2 * (1 + eta) * math.sqrt(rho * math.log(1 / gamma))
        - eta * rho
    )


def zcdp_numeric(rho: float, eta: float, gamma: float) -> float:
    """Theorem 2 with ε(λ) = ρλ at λ* = 1 + √(ln E[K]/ρ), minimized over λ̂ ≥ 1 on a fine grid."""
    lam = 1 + math.sqrt(math.log(tnb_mean(eta, gamma)) / rho)
    hats = [1 + j / 1000 for j in range(200_000)]
    return min(tnb_terms(lambda a: rho * a, eta, gamma, lam, h).total for h in hats)


# --- Section: a user with k records in example-level DP-SGD -----------------------------------


def binom_pmf(k: int, q: float) -> Array:
    return np.array([math.comb(k, m) * q**m * (1 - q) ** (k - m) for m in range(k + 1)])


def pair_moment(alpha: int, k: int, q: float, sigma: float) -> float:
    """A_α = E[exp(Σ_{i<j} M_i M_j/σ²)], M_i iid Binomial(k, q): the aligned user (eq-ulvl-pair)."""
    b = binom_pmf(k, q)
    total = 0.0
    for ms in itertools.product(range(k + 1), repeat=alpha):
        weight = math.prod(float(b[m]) for m in ms)
        cross = sum(ms[i] * ms[j] for i in range(alpha) for j in range(i + 1, alpha))
        total += weight * math.exp(cross / sigma**2)
    return total


def subset_moment(alpha: int, gram: Array, q: float, sigma: float) -> float:
    """A_α = E[exp(Σ_{i<j} <v_{S_i}, v_{S_j}>/σ²)] over independent Poisson subsets S_1..S_α of the user's
    records, for gradients with Gram matrix `gram` (in units of C)."""
    k = gram.shape[0]
    masks = np.array(list(itertools.product((0.0, 1.0), repeat=k)))  # every subset as a 0/1 row
    probs = np.prod(np.where(masks > 0, q, 1 - q), axis=1)
    inner = masks @ gram @ masks.T  # <v_S, v_S'> for every pair of subsets
    if alpha == 2:
        return float(probs @ np.exp(inner / sigma**2) @ probs)
    if alpha == 3:
        e = np.exp(inner / sigma**2)
        return float(np.einsum("a,b,c,ab,ac,bc->", probs, probs, probs, e, e, e))
    raise ValueError("alpha must be 2 or 3")


def equal_angle_gram(k: int, theta: float) -> Array:
    """k unit vectors with every pairwise angle θ (possible while cos θ ≥ -1/(k-1))."""
    return np.full((k, k), math.cos(theta)) + (1 - math.cos(theta)) * np.eye(k)


def mgf_identity_check(grads: Array, q: float, sigma: float, alpha: int = 2) -> tuple[float, float]:
    """(∫ P0 (P/P0)^α by 2-D quadrature, the formula E[exp(Σ_{i<j}<v_i, v_j>/σ²)]) for 2-D gradients."""
    h = 0.04
    grid = np.arange(-12.0, 12.0 + h, h)
    yy1, yy2 = np.meshgrid(grid, grid)
    y = np.stack([yy1.ravel(), yy2.ravel()], axis=1)
    k = grads.shape[0]
    masks = np.array(list(itertools.product((0.0, 1.0), repeat=k)))
    probs = np.prod(np.where(masks > 0, q, 1 - q), axis=1)
    shifts = masks @ grads  # v_S for every subset
    ratio = np.zeros(y.shape[0])
    for p, v in zip(probs, shifts, strict=True):
        ratio += p * np.exp((2 * (y @ v) - float(v @ v)) / (2 * sigma**2))
    p0 = np.exp(-(y**2).sum(axis=1) / (2 * sigma**2)) / (2 * math.pi * sigma**2)
    quad = float((p0 * ratio**alpha).sum() * h * h)
    return quad, subset_moment(alpha, grads @ grads.T, q, sigma)


def random_configs_below_aligned(rng: np.random.Generator, k: int, q: float, sigma: float, trials: int) -> bool:
    """Random gradients of length ≤ 1 in R^3 never beat the aligned configuration, at α = 2 and 3."""
    aligned = {a: pair_moment(a, k, q, sigma) for a in (2, 3)}
    ok = True
    for _ in range(trials):
        g = rng.standard_normal((k, 3))
        g = g / np.linalg.norm(g, axis=1, keepdims=True) * rng.random((k, 1))
        for a in (2, 3):
            ok = ok and subset_moment(a, g @ g.T, q, sigma) <= aligned[a] * (1 + 1e-12)
    return ok


# --- Section: a group's loss change in one DP-SGD step ----------------------------------------

TOY_GRADS: Array = np.array([[0.2, 0.0], [0.0, 4.0]])  # the chapter's toy: majority and minority gradients
TOY_SHARES: Array = np.array([0.95, 0.05])
TOY_CURV = 1.0  # each group's loss is (h/2)‖w - w_g*‖², with gradient TOY_GRADS[g] at the current w
TOY_LR = 0.5
TOY_SIGMA = 1.36  # the running example's noise multiplier at ε = 4
TOY_BATCH = 100  # the toy's 100 records, all in every step


@dataclass(frozen=True)
class StepTerms:
    descent: float  # -η<∇L_g, ∇L>, the step without clipping or noise
    clipping: float  # -η<∇L_g, m - ∇L>, what clipping changes
    curvature: float  # (η²/2) mᵀH_g m
    noise: float  # (η²/2) τ² tr H_g

    @property
    def total(self) -> float:
        return self.descent + self.clipping + self.curvature + self.noise


def clip(v: Array, c: float) -> Array:
    return v * min(1.0, c / float(np.linalg.norm(v)))


def toy_mean_update(c: float) -> Array:
    """m = Σ_h s_h m_h, every record of a group having its group's gradient."""
    return sum((s * clip(g, c) for s, g in zip(TOY_SHARES, TOY_GRADS, strict=True)), np.zeros(2))


def toy_step_terms(group: int, c: float) -> StepTerms:
    grad_g = TOY_GRADS[group]
    full = TOY_SHARES @ TOY_GRADS
    m = toy_mean_update(c)
    tau = TOY_SIGMA * c / TOY_BATCH
    return StepTerms(
        -TOY_LR * float(grad_g @ full),
        -TOY_LR * float(grad_g @ (m - full)) + 0.0,
        TOY_LR**2 / 2 * TOY_CURV * float(m @ m),
        TOY_LR**2 / 2 * tau**2 * TOY_CURV * 2,
    )


def toy_step_monte_carlo(
    rng: np.random.Generator, group: int, c: float, sigma: float, draws: int
) -> tuple[float, float]:
    """(simulated E[L_g(w - η(m + τξ))] - L_g(w), the formula), with noise multiplier `sigma`."""
    grad_g = TOY_GRADS[group]
    full = TOY_SHARES @ TOY_GRADS
    m = toy_mean_update(c)
    tau = sigma * c / TOY_BATCH
    u = m + tau * rng.standard_normal((draws, 2))
    x = grad_g / TOY_CURV  # w - w_g*: the loss is (h/2)‖x‖²
    after = 0.5 * TOY_CURV * ((x - TOY_LR * u) ** 2).sum(axis=1)
    sim = float(after.mean()) - 0.5 * TOY_CURV * float(x @ x)
    formula = (
        -TOY_LR * float(grad_g @ full)
        - TOY_LR * float(grad_g @ (m - full))
        + TOY_LR**2 / 2 * TOY_CURV * (float(m @ m) + 2 * tau**2)
    )
    return sim, formula


def floor_balance(share: float, lr: float, tau: float, dim: int, curv: float) -> tuple[float, float]:
    """(the floor of eq-fair-floor, the expected one-step change of L_g when E L_g sits at that floor)."""
    floor = dim * lr * tau**2 / (2 * share * (2 - lr * share * curv))
    a = lr * share * curv
    # E[ΔL_g] = -ηλ s λ‖x‖²(1 - ηsλ/2)... written with E L_g = (λ/2)E‖x‖²: -2a(1 - a/2) E L_g + η²τ²kλ/2.
    change = -2 * a * (1 - a / 2) * floor + lr**2 * tau**2 * dim * curv / 2
    return floor, change


# --- The numbers quoted in the text --------------------------------------------------------------


def print_table(rng: np.random.Generator) -> None:
    mu = gaussian_mu(EPS_RUN, DELTA_RUN)
    print(f"[ERM] mu(1, 1e-5) = {mu:.4f}")
    for n in (1_000, 10_000):
        t = strong_steps_rule(n, D_RUN, mu)
        sig = strong_noise_sigma(n, t, mu)
        print(
            f"[ERM strong n={n}] T={t}  sigma={sig:.4f}  bound at T={strong_bound_at(n, D_RUN, LAM_RUN, mu, t):.4e}  "
            f"closed form={strong_upper(n, D_RUN, LAM_RUN, mu):.4e}  uniform-average bound="
            f"{sgd_uniform_bound(t, 4 + D_RUN * sig**2, LAM_RUN):.4e}  "
            f"lower={strong_lower(n, D_RUN, LAM_RUN, EPS_RUN, DELTA_RUN):.4e}"
        )
    for n in (300, 1_000, 3_000):
        meas = measured_noisy_gd(rng, n, mu, trials=2)
        print(
            f"[ERM measured n={n}] T={strong_steps_rule(n, D_RUN, mu)}  measured={meas:.4e}  "
            f"bound={strong_upper(n, D_RUN, LAM_RUN, mu):.4e}  ratio={strong_upper(n, D_RUN, LAM_RUN, mu) / meas:.1f}"
        )
    print(
        f"[ERM K] d=10: K={fingerprint_k(10, EPS_RUN, DELTA_RUN):.4f}  d=1000: K={fingerprint_k(1000, EPS_RUN, DELTA_RUN):.4f}"
    )
    for d in (10, 1_000, 10_000):
        n = 10_000
        lo = strong_lower(n, d, LAM_RUN, EPS_RUN, DELTA_RUN)
        up = strong_upper(n, d, LAM_RUN, mu)
        print(
            f"[ERM gap d={d} n={n}] lower={lo:.4e}  upper={up:.4e}  ratio={up / lo:.3g}  "
            f"lower*lam*n^2/d={lo * LAM_RUN * n * n / d:.3e}"
        )
    for d in (1_000, 10_000):
        n = 10_000
        lo = convex_lower(n, d, 5.0, EPS_RUN, DELTA_RUN)
        up = 4 * 5.0 * math.sqrt(d * math.log(1.25 / DELTA_RUN)) / (n * EPS_RUN)
        print(
            f"[ERM convex d={d} n={n} R=5] lower={lo:.4e}  upper (eq-dperm-convex-rate)={up:.4e}  ratio={up / lo:.3g}"
        )

    data = make_logistic_data(rng, 10_000)
    out = output_perturbation_check(rng, data, mu, trials=20)
    print(
        f"[ERM output n=10000] lower={out.lower:.4e}  approx={out.approx:.4e}  measured={out.measured:.4e}  "
        f"upper={out.upper:.4e}"
    )
    obj = objective_perturbation_check(rng, data, EPS_RUN, trials=20)
    print(
        f"[ERM objective n=10000] lower={obj.lower:.4e}  approx={obj.approx:.4e}  measured={obj.measured:.4e}  "
        f"upper={obj.upper:.4e}"
    )
    print(f"[ERM bound/measured] output={out.upper / out.measured:.1f}  objective={obj.upper / obj.measured:.1f}")
    print(
        f"[ERM objective per-draw inequalities] hold={objective_inequalities_hold(rng, make_logistic_data(rng, 1_000), 50)}"
    )

    qt = two_label_q(150, SIGMA_PATE)
    o = pate_orders(qt, SIGMA_PATE)
    t1, t2 = pate_log_terms(ALPHA_PATE, qt, o)
    print(
        f"[PATE gap=150 sigma=40 alpha=8] q~={qt:.4e}  mu2={o.mu2:.3f}  mu1={o.mu1:.3f}  eps1={o.eps1:.5f}  "
        f"eps2={o.eps2:.5f}  p-bound={pate_p_bound(qt, o):.4e}  term A={math.exp(t1):.6f}  term B={math.exp(t2):.4e}  "
        f"bound={pate_value(ALPHA_PATE, qt, o):.4e}  rigorous={pate_rigorous(ALPHA_PATE, qt, SIGMA_PATE):.4e}  "
        f"exact={two_label_exact(ALPHA_PATE, 150, SIGMA_PATE):.4e}  condition={pate_condition(qt, SIGMA_PATE)}"
    )
    print(
        f"[PATE mu2 choice] p-bound/q~ = {pate_p_bound(qt, o) / qt:.4f}  "
        f"exp(2 sqrt(ln 1/q~)/sigma - 1/sigma^2) = {math.exp(2 * math.sqrt(math.log(1 / qt)) / SIGMA_PATE - 1 / SIGMA_PATE**2):.4f}"
    )
    gaps = range(0, 301, 5)
    cond_gaps = [
        g
        for g in gaps
        if pate_valid(ALPHA_PATE, two_label_q(g, SIGMA_PATE), SIGMA_PATE)
        and pate_condition(two_label_q(g, SIGMA_PATE), SIGMA_PATE)
    ]
    inc = all(pate_increasing(ALPHA_PATE, two_label_q(g, SIGMA_PATE), SIGMA_PATE) for g in cond_gaps)
    below = all(
        two_label_exact(ALPHA_PATE, g, SIGMA_PATE) <= pate_rigorous(ALPHA_PATE, two_label_q(g, SIGMA_PATE), SIGMA_PATE)
        for g in cond_gaps
    )
    print(
        f"[PATE sweep] condition holds from gap {min(cond_gaps)}; increasing on (0, q~] at all {len(cond_gaps)} "
        f"such gaps: {inc}; exact two-label below bound: {below}"
    )
    q_limit = bisect(lambda q: not pate_condition(q, SIGMA_PATE), 1e-300, 0.5)
    below_limit = all(pate_condition(q_limit * 10 ** (-j / 10), SIGMA_PATE) for j in range(1, 200))
    print(f"[PATE condition holds for every q~ below the limit] {below_limit}")
    cross = min(
        g
        for g in range(301)
        if pate_valid(ALPHA_PATE, two_label_q(g, SIGMA_PATE), SIGMA_PATE)
        and pate_condition(two_label_q(g, SIGMA_PATE), SIGMA_PATE)
        and pate_rigorous(ALPHA_PATE, two_label_q(g, SIGMA_PATE), SIGMA_PATE) < ALPHA_PATE / SIGMA_PATE**2
    )
    print(
        f"[PATE smallest gap where the bound beats alpha/sigma^2] {cross}  ratio at 150="
        f"{pate_value(ALPHA_PATE, qt, o) / two_label_exact(ALPHA_PATE, 150, SIGMA_PATE):.2f}"
    )
    p3, pp3 = [0.99, 0.008, 0.002], [0.985, 0.01, 0.005]
    parts = [a * a / b for a, b in zip(p3, pp3, strict=True)]
    print(
        f"[PATE three-answer example, order 2] terms={parts[0]:.5f} {parts[1]:.4f} {parts[2]:.4f}  sum={sum(parts):.5f}  "
        f"average ratio off i*={(parts[1] + parts[2]) / 0.01:.2f}"
    )
    print(f"[PATE condition limit sigma=40] largest q~ = {q_limit:.4e}")
    worst, qt3, bound3 = three_label_check((200, 40, 10), ALPHA_PATE, SIGMA_PATE)
    print(
        f"[PATE three labels (200, 40, 10)] exact worst D={worst:.4e}  q~={qt3:.4e}  bound={bound3:.4e}  "
        f"data-independent={ALPHA_PATE / SIGMA_PATE**2:.4e}"
    )
    pure_ok = all(pure_convex_and_increasing(a, e) for a in (1.5, 2, 8, 32) for e in (0.1, 0.5, 1, 2))
    print(f"[PATE pure form] convex and increasing in q on every tested (alpha, eps0): {pure_ok}")

    gamma = tnb_gamma(0, 10)
    hat, bound = tnb_best(lambda a: rr_rdp(a, 0.5), 0.0, gamma, 5.0)
    terms = tnb_terms(lambda a: rr_rdp(a, 0.5), 0.0, gamma, 5.0, hat)
    print(
        f"[TNB RR eps=0.5 log mean 10 lam=5] gamma={gamma:.5f}  best lam_hat={hat:.2f}  one run={terms.one_run:.4f}  "
        f"tail={terms.tail:.4f}  gamma term={terms.gamma:.4f}  mean term={terms.mean:.4f}  bound={bound:.4f}  "
        f"exact={tnb_exact_rr(0.5, 0.0, gamma, 5.0):.4f}  limit lam_hat->inf={tnb_pure_limit(0.5, 0.0, gamma, 5.0):.4f}"
    )
    for lam in (2.0, 20.0):
        print(
            f"[TNB RR order {lam:g}] grid bound={tnb_best(lambda a: rr_rdp(a, 0.5), 0.0, gamma, lam)[1]:.4f}  "
            f"limit={tnb_pure_limit(0.5, 0.0, gamma, lam):.4f}"
        )
    print(f"[TNB chain] every step holds on random cases: {tnb_chain_holds(rng, 20_000)}")
    rho, eta = 0.1, 1.0
    g1 = tnb_gamma(eta, 10)
    print(
        f"[TNB zCDP rho={rho} eta={eta} mean 10] gamma={g1:.5f}  closed form={zcdp_closed_form(rho, eta, g1):.4f}  "
        f"numeric={zcdp_numeric(rho, eta, g1):.4f}  lam*={1 + math.sqrt(math.log(10) / rho):.4f}  "
        f"lam_hat={math.sqrt(math.log(1 / g1) / rho):.4f}"
    )

    k, q, sigma = 5, 0.05, 1.0
    a2 = pair_moment(2, k, q, sigma)
    print(
        f"[User k=5 q=0.05 sigma=1] A2 aligned={a2:.6f}  eps(2)={math.log(a2):.6f}  "
        f"A2 at 90 deg={subset_moment(2, equal_angle_gram(k, math.pi / 2), q, sigma):.6f}  "
        f"A2 at 60 deg={subset_moment(2, equal_angle_gram(k, math.pi / 3), q, sigma):.6f}"
    )
    grads = np.array([[1.0, 0.0], [0.5, math.sqrt(3) / 2], [-math.sqrt(3) / 2, 0.5]])
    quad, formula = mgf_identity_check(grads, 0.3, 1.0)
    print(f"[User MGF identity, 3 gradients at 0/60/150 deg, q=0.3] quadrature={quad:.6f}  formula={formula:.6f}")
    print(f"[User random configurations below aligned] {random_configs_below_aligned(rng, 4, 0.1, 1.0, 200)}")
    b = binom_pmf(k, q)
    contrib = {
        (m1, m2): float(b[m1] * b[m2]) * math.exp(m1 * m2 / sigma**2) for m1 in range(k + 1) for m2 in range(k + 1)
    }
    zero = sum(v for (m1, m2), v in contrib.items() if m1 * m2 == 0)
    print(
        f"[User contributions] M1*M2=0: {zero:.6f}  (1,1): {contrib[(1, 1)]:.6f}  (1,2)+(2,1): "
        f"{contrib[(1, 2)] + contrib[(2, 1)]:.6f}  (2,2): {contrib[(2, 2)]:.6f}  (5,5): {contrib[(5, 5)]:.3e}"
    )

    for g in (0, 1):
        st = toy_step_terms(g, 0.5)
        print(
            f"[Fair toy C=0.5 group={'majority' if g == 0 else 'minority'}] descent={st.descent:.5f}  "
            f"clipping={st.clipping:.5f}  curvature={st.curvature:.5f}  noise={st.noise:.3e}  total={st.total:.5f}"
        )
    sim, formula = toy_step_monte_carlo(rng, 1, 0.5, 100.0, 2_000_000)
    print(f"[Fair exactness, minority, C=0.5, sigma=100] simulated={sim:.5f}  formula={formula:.5f}")
    floor, change = floor_balance(0.05, 0.5, 0.0068, 51, 1.0)
    print(f"[Fair floor s=0.05] floor={floor:.4e}  expected change at the floor={change:.1e}")


# --- Figures --------------------------------------------------------------------------------


def plot_erm_rates(rng: np.random.Generator) -> Figure:
    mu = gaussian_mu(EPS_RUN, DELTA_RUN)
    ns = [int(10 ** (2.5 + j / 10)) for j in range(26)]
    meas_ns = [300, 1_000, 3_000]
    meas = [measured_noisy_gd(rng, n, mu, trials=2) for n in meas_ns]
    fig, (left, right) = plt.subplots(1, 2, figsize=(10, 4))
    left.loglog(ns, [strong_upper(n, D_RUN, LAM_RUN, mu) for n in ns], label=r"upper bound $16L^2d/(\lambda n^2\mu^2)$")
    left.loglog(meas_ns, meas, "ko", label="noisy projected GD, measured")
    left.loglog(
        ns,
        [strong_lower(n, D_RUN, LAM_RUN, EPS_RUN, DELTA_RUN) for n in ns],
        "--",
        label="lower bound, any (1, 10⁻⁵)-DP algorithm",
    )
    left.set(xlabel="records n", ylabel="excess risk", title=f"Running example, d = {D_RUN}, λ = {LAM_RUN}")
    left.legend(fontsize=8)
    ds = [int(10 ** (1 + j / 8)) for j in range(33)]
    n = 10_000
    right.loglog(ds, [strong_upper(n, d, LAM_RUN, mu) for d in ds], label="upper bound")
    right.loglog(ds, [strong_lower(n, d, LAM_RUN, EPS_RUN, DELTA_RUN) for d in ds], "--", label="lower bound")
    right.set(xlabel="dimension d", ylabel="excess risk", title=f"n = {n:,}, λ = {LAM_RUN}: both grow like d")
    right.legend(fontsize=8)
    return fig


def plot_erm_reduction(rng: np.random.Generator) -> Figure:
    """d = 2, L = λ = 1: m = 4 real records at corners (±1, ±1)/√2, n - m = 4 zero records."""
    corners = np.array([[1, 1], [1, -1], [-1, 1], [1, 1]]) / math.sqrt(2)
    n, m = 8, 4
    real_mean = corners.mean(axis=0)
    w_star = real_mean * m / n  # x̄ = (m/n) x̄_real, and w* = x̄/λ
    w_priv = w_star + 0.12 * rng.standard_normal(2)
    estimate = w_priv * n / m
    fig, ax = plt.subplots(figsize=(6, 5.4))
    grid = np.linspace(-1.1, 1.1, 300)
    gx, gy = np.meshgrid(grid, grid)
    excess = 0.5 * ((gx - w_star[0]) ** 2 + (gy - w_star[1]) ** 2)
    cs = ax.contour(gx, gy, excess, levels=[0.01, 0.04, 0.09, 0.16, 0.25], colors="0.7", linewidths=0.8)
    ax.clabel(cs, fontsize=7, fmt="%.2f")
    for xy, label in (((1, 1), "2 records"), ((1, -1), "1 record"), ((-1, 1), "1 record")):
        ax.plot(*(np.array(xy) / math.sqrt(2)), "s", color="tab:blue", markersize=8 if label == "1 record" else 11)
        ax.annotate(
            label, (xy[0] / math.sqrt(2), xy[1] / math.sqrt(2)), textcoords="offset points", xytext=(6, 6), fontsize=8
        )
    ax.plot(-1 / math.sqrt(2), -1 / math.sqrt(2), "s", color="0.85", markersize=8)
    ax.annotate(
        "no record", (-1 / math.sqrt(2), -1 / math.sqrt(2)), textcoords="offset points", xytext=(6, 6), fontsize=8
    )
    ax.plot(0, 0, "o", color="tab:gray", markersize=9)
    ax.annotate("4 zero records", (0, 0), textcoords="offset points", xytext=(-80, -14), fontsize=8)
    ax.plot(*real_mean, "*", color="tab:blue", markersize=13, label="mean of the real records")
    ax.plot(*w_star, "*", color="black", markersize=13, label="minimizer θ* = mean of all 8 records")
    ax.plot(*w_priv, "o", color="tab:red", label="private output")
    ax.plot(*estimate, "D", color="tab:red", mfc="none", label="mean estimate: private output times n/m")
    ax.annotate(
        "",
        (float(estimate[0]), float(estimate[1])),
        (float(w_priv[0]), float(w_priv[1])),
        arrowprops={"arrowstyle": "->", "color": "tab:red"},
    )
    ax.set(
        xlim=(-1.1, 1.1),
        ylim=(-1.1, 1.1),
        aspect="equal",
        xlabel="coordinate 1",
        ylabel="coordinate 2",
        title="Grey circles: excess risk, set by the distance to θ*",
    )
    ax.legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=2)
    return fig


def plot_erm_sandwich(rng: np.random.Generator) -> Figure:
    data = make_logistic_data(rng, 10_000)
    w_star = fit(data, LAM_RUN)
    u = rng.standard_normal(D_RUN)
    u /= np.linalg.norm(u)
    h = float(u @ hessian(w_star, data, LAM_RUN) @ u)
    beta = C_LOGISTIC + LAM_RUN
    ts = np.linspace(-4, 4, 161)
    j_star = objective(w_star, data, LAM_RUN)
    vals = [objective(w_star + t * u, data, LAM_RUN) - j_star for t in ts]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(ts, [beta / 2 * t * t for t in ts], "--", label=r"$\frac{\beta_s}{2}t^2$, smoothness (upper)")
    ax.plot(ts, vals, "k", linewidth=2, label=r"$\mathcal{L}(\theta^* + tu) - \mathcal{L}(\theta^*)$")
    ax.plot(ts, [h / 2 * t * t for t in ts], ":", label=r"$\frac{h_u}{2}t^2$, second-order approximation")
    ax.plot(ts, [LAM_RUN / 2 * t * t for t in ts], "--", label=r"$\frac{\lambda}{2}t^2$, strong convexity (lower)")
    ax.set(
        ylim=(0, 0.6),
        xlabel="distance t from the minimizer along a random direction u",
        ylabel="excess risk",
        title="Running example, n = 10,000",
    )
    ax.legend(fontsize=8)
    return fig


def plot_pate_terms() -> Figure:
    gaps = list(range(0, 301, 2))
    data_ind = ALPHA_PATE / SIGMA_PATE**2
    part_a: list[float] = []
    part_b: list[float] = []
    total: list[float] = []
    exact: list[float] = []
    for g in gaps:
        qt = two_label_q(g, SIGMA_PATE)
        exact.append(two_label_exact(ALPHA_PATE, g, SIGMA_PATE))
        if not (pate_valid(ALPHA_PATE, qt, SIGMA_PATE) and pate_condition(qt, SIGMA_PATE)):
            part_a.append(math.nan)
            part_b.append(math.nan)
            total.append(math.nan)
            continue
        o = pate_orders(qt, SIGMA_PATE)
        t1, t2 = pate_log_terms(ALPHA_PATE, qt, o)
        part_a.append((math.exp(t1) - (1 - qt)) / (ALPHA_PATE - 1))
        part_b.append((math.exp(t2) - qt) / (ALPHA_PATE - 1))
        total.append(pate_value(ALPHA_PATE, qt, o))
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    ax.semilogy(gaps, [data_ind] * len(gaps), color="black", label=r"data-independent $\alpha/\sigma^2$")
    ax.semilogy(gaps, total, color="tab:red", linewidth=2, label="Theorem 6 bound")
    ax.semilogy(gaps, part_a, color="tab:orange", ls="--", label="likely output's term, minus its value 1 - β at A = 1")
    ax.semilogy(gaps, part_b, color="tab:purple", ls="--", label="other outputs' term, minus its value β at B = 1")
    ax.semilogy(gaps, exact, color="tab:blue", ls=":", label="exact, two labels")
    ax.set(
        ylim=(1e-9, 2e-2),
        xlabel="gap g between the two vote counts",
        ylabel="RDP cost at order 8",
        title=f"GNMax, σ = {SIGMA_PATE:g}, α = {ALPHA_PATE:g}",
    )
    ax.legend(fontsize=7.5, loc="lower left")
    return fig


def plot_pate_q() -> Figure:
    fig, (left, right) = plt.subplots(1, 2, figsize=(10, 4))
    for g, color in ((150, "tab:red"), (200, "tab:blue")):
        qt = two_label_q(g, SIGMA_PATE)
        o = pate_orders(qt, SIGMA_PATE)
        qs = [qt * 10 ** (-3 + 3 * j / 200) for j in range(201)] + [qt * (1 + 0.5 * j / 50) for j in range(1, 51)]
        vals = [pate_value(ALPHA_PATE, q, o) if q * math.exp(o.eps2) < 1 else math.nan for q in qs]
        left.loglog(qs, vals, color=color, label=f"orders set for gap {g}")
        left.axvline(qt, color=color, ls=":", linewidth=1)
        left.plot([qt], [pate_value(ALPHA_PATE, qt, o)], "o", color=color)
    left.set(
        xlabel="true failure probability β",
        ylabel="bound with orders fixed",
        title="Increasing in β up to the bound β̃ (dotted)",
    )
    left.legend(fontsize=8)
    q_limit = bisect(lambda q: not pate_condition(q, SIGMA_PATE), 1e-300, 0.5)
    qts = [10 ** (-12 + 11.5 * j / 300) for j in range(301)]
    right.loglog(qts, [ALPHA_PATE / SIGMA_PATE**2] * len(qts), "k", label=r"$\alpha/\sigma^2$")
    right.loglog(
        qts,
        [pate_rigorous(ALPHA_PATE, q, SIGMA_PATE) if pate_valid(ALPHA_PATE, q, SIGMA_PATE) else math.nan for q in qts],
        color="tab:red",
        label="largest value over β ≤ β̃",
    )
    right.axvline(q_limit, color="gray", ls="--", label="largest β̃ the condition allows")
    right.set(
        ylim=(1e-12, 1e-1),
        xlabel="bound β̃ on the failure probability",
        ylabel="RDP cost at order 8",
        title=f"σ = {SIGMA_PATE:g}",
    )
    right.legend(fontsize=8)
    return fig


def plot_tuning_terms() -> Figure:
    gamma = tnb_gamma(0, 10)
    hats = [10 ** (0.01 + 4 * j / 400) for j in range(401)]

    def eps_of(a: float) -> float:
        return rr_rdp(a, 0.5)

    terms = [tnb_terms(eps_of, 0.0, gamma, 5.0, h) for h in hats]
    limit = tnb_pure_limit(0.5, 0.0, gamma, 5.0)
    end = LAM_HATS[-1]
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    ax.semilogx(hats, [t.one_run for t in terms], label="one run, ε(α)")
    ax.semilogx(hats, [t.tail for t in terms], label="(1 + ν)(1 - 1/α̂) ε(α̂)")
    ax.semilogx(hats, [t.gamma for t in terms], label="(1 + ν) ln(1/ω)/α̂")
    ax.semilogx(hats, [t.mean for t in terms], label="ln E[K]/(α - 1)")
    ax.semilogx(hats, [t.total for t in terms], "k", linewidth=2, label="sum, the bound")
    ax.axhline(limit, color="gray", ls="--", linewidth=1, label=f"limit as α̂ → ∞: {limit:.4f}")
    at_end = tnb_terms(eps_of, 0.0, gamma, 5.0, end).total
    ax.plot([end], [at_end], "ko")
    ax.annotate(f"α̂ = {end:.2f}: {at_end:.4f}", (end, at_end), textcoords="offset points", xytext=(-30, 12))
    ax.set(
        ylim=(0, 4),
        xlabel="second order α̂ (log scale)",
        ylabel="Rényi bound at α = 5",
        title="Randomized response, ε = 0.5, logarithmic K with mean 10",
    )
    ax.legend(fontsize=8, loc="upper right")
    return fig


def plot_tuning_square() -> Figure:
    lam, lam_hat, eps_hat, gamma = 5.0, 4.0, 0.5, 0.05
    grid = np.linspace(1e-4, 1 - 1e-4, 400)
    rr, rrp = np.meshgrid(grid, grid)
    s = rrp**lam_hat * rr ** (1 - lam_hat) + (1 - rrp) ** lam_hat * (1 - rr) ** (1 - lam_hat)
    allowed = np.log(s) / (lam_hat - 1) <= eps_hat
    a, ap = gamma + (1 - gamma) * rr, gamma + (1 - gamma) * rrp
    ratio = gamma * ap ** (lam - 1) / a**lam
    fig, ax = plt.subplots(figsize=(5.8, 5.4))
    ax.contourf(rr, rrp, allowed.astype(float), levels=[0.5, 1.5], colors=["#cfe3f5"])
    cs = ax.contour(rr, rrp, np.log10(ratio), levels=[-3, -2, -1, 0, 1], colors="0.4", linewidths=0.8)

    def power_label(v: float) -> str:
        return f"$10^{{{v:g}}}$"

    ax.clabel(cs, fontsize=7, fmt=power_label)
    sexp = 1 - 1 / lam_hat
    ax.plot(
        grid,
        np.minimum(1, math.exp(sexp * eps_hat) * grid**sexp),
        "tab:red",
        label="r' = exp((1 - 1/α̂) ε̂) r^(1 - 1/α̂), the bound kept",
    )
    bound = math.exp((lam - 1) * sexp * eps_hat) * gamma ** (-(lam - 1) / lam_hat)
    masked = np.where(allowed, ratio, -np.inf)
    i, j = np.unravel_index(int(np.argmax(masked)), masked.shape)
    ax.plot(rr[i, j], rrp[i, j], "ko", label=f"largest value in the shaded region: {float(ratio[i, j]):.2f}")
    ax.plot([], [], " ", label=f"the derivation's bound: {bound:.2f}")
    ax.set(
        xlabel="rejection probability r on D",
        ylabel="rejection probability r' on D'",
        aspect="equal",
        title=f"α = {lam:g}, α̂ = {lam_hat:g}, ε̂ = {eps_hat}, ω = {gamma}",
    )
    ax.legend(fontsize=7.5, loc="lower right")
    return fig


def plot_user_angle() -> Figure:
    k, q, sigma = 5, 0.05, 1.0
    theta_max = math.acos(-1 / (k - 1))
    thetas = [theta_max * j / 200 for j in range(201)]
    fig, ax = plt.subplots(figsize=(6, 4))
    for alpha, color in ((2, "tab:blue"), (3, "tab:red")):
        aligned = math.log(pair_moment(alpha, k, q, sigma)) / (alpha - 1)
        vals = [math.log(subset_moment(alpha, equal_angle_gram(k, t), q, sigma)) / (alpha - 1) for t in thetas]
        ax.plot(
            [math.degrees(t) for t in thetas],
            [v / aligned for v in vals],
            color=color,
            label=f"order {alpha} (aligned value {aligned:.3g})",
        )
    ax.axhline(1.0, color="gray", ls=":", linewidth=1)
    ax.set(
        xlabel="angle between every pair of the user's gradients (degrees)",
        ylim=(0, 1.08),
        ylabel="divergence / its aligned value",
        title="k = 5 records, q = 0.05, σ = 1",
    )
    ax.legend()
    return fig


def plot_user_pairs() -> Figure:
    k, q, sigma = 5, 0.05, 1.0
    b = binom_pmf(k, q)
    fig, (left, right) = plt.subplots(1, 2, figsize=(10, 4))
    labels: list[str] = []
    plain: list[float] = []
    weighted: list[float] = []
    for m1 in range(4):
        for m2 in range(4):
            labels.append(f"{m1},{m2}")
            plain.append(float(b[m1] * b[m2]))
            weighted.append(float(b[m1] * b[m2]) * math.exp(m1 * m2 / sigma**2))
    x = np.arange(len(labels))
    left.bar(x - 0.2, plain, width=0.4, label=r"$\Pr[N_1 = m_1, N_2 = m_2]$")
    left.bar(x + 0.2, weighted, width=0.4, label=r"times $e^{m_1m_2/\sigma^2}$, its share of $A_2$")
    left.set(
        yscale="log", xticks=x, xlabel="(m₁, m₂)", ylabel="probability or contribution", title="k = 5, q = 0.05, σ = 1"
    )
    left.set_xticklabels(labels, fontsize=7, rotation=90)
    left.legend(fontsize=8)
    ks = list(range(1, 21))
    right.semilogy(ks, [math.log(pair_moment(2, kk, q, sigma)) for kk in ks], "o-", label="exact, order 2")
    right.semilogy(
        ks, [(kk * q) ** 2 * math.expm1(1 / sigma**2) for kk in ks], "--", label=r"$(kq)^2(e^{1/\sigma^2}-1)$"
    )
    right.set(
        xticks=[1, 5, 10, 15, 20],
        xlabel="records k of the user",
        ylabel="RDP at order 2",
        title="q = 0.05, σ = 1: close to rate kq while kq is small",
    )
    right.legend(fontsize=8)
    return fig


def plot_fair_terms() -> Figure:
    cs = [10 ** (-1.5 + 2.2 * j / 200) for j in range(201)]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.6))
    for group, ax in enumerate(axes):
        terms = [toy_step_terms(group, c) for c in cs]
        ax.loglog(cs, [-t.descent for t in terms], color="black", label="decrease without clipping or noise")
        ax.loglog(
            cs,
            [t.clipping if t.clipping > 0 else math.nan for t in terms],
            color="tab:red",
            label="increase from clipping",
        )
        ax.loglog(cs, [t.curvature for t in terms], color="tab:green", label="increase from the mean step's curvature")
        ax.loglog(cs, [t.noise for t in terms], color="tab:blue", label="increase from noise")
        grad_g: Array = TOY_GRADS[group]
        ax.axvline(float(np.linalg.norm(grad_g)), color="gray", ls=":", linewidth=1)
        name = "majority (95 records, gradient length 0.2)" if group == 0 else "minority (5 records, gradient length 4)"
        ax.set(xlabel="clipping norm C", ylabel="size of the term", title=name)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.02), ncol=2, fontsize=8)
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
        "learn_erm_rates.svg": plot_erm_rates(rng),
        "learn_erm_reduction.svg": plot_erm_reduction(rng),
        "learn_erm_sandwich.svg": plot_erm_sandwich(rng),
        "learn_pate_terms.svg": plot_pate_terms(),
        "learn_pate_q.svg": plot_pate_q(),
        "learn_tuning_terms.svg": plot_tuning_terms(),
        "learn_tuning_square.svg": plot_tuning_square(),
        "learn_user_angle.svg": plot_user_angle(),
        "learn_user_pairs.svg": plot_user_pairs(),
        "learn_fair_terms.svg": plot_fair_terms(),
    }
    for name, fig in figures.items():
        fig.tight_layout()
        fig.savefig(out / name, metadata={"Date": None}, bbox_inches="tight")  # no timestamp, so output is reproducible
        plt.close(fig)
        print(f"saved {out / name}")


if __name__ == "__main__":
    main()
