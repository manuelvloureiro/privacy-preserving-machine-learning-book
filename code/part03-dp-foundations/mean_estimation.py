"""Check the numbers in the "Private mean estimation" chapter and draw its figures.

    uv run python code/part03-dp-foundations/mean_estimation.py
    uv run python code/part03-dp-foundations/mean_estimation.py --seed 1 --out /tmp/figures

Prints the numbers quoted in 11-mean-estimation.md (they match exactly with the default seed),
then saves the chapter's figures.

Conventions: datasets have a fixed size n and neighbours differ in one replaced record. The data
are drawn independently from a distribution whose mean μ is the target. Accuracy is measured by
the root-mean-square error (RMSE) of the released estimate around μ, and every clipping range is
chosen so that, with probability at least 1 - β, it contains every data point.
"""

import argparse
import math
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
matplotlib.rcParams["svg.hashsalt"] = "mean_estimation"  # stable element ids, so reruns don't churn the SVGs
matplotlib.rcParams["svg.fonttype"] = "none"  # keep text as text: smaller, diffable files

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from numpy.typing import NDArray

# Figures are build output: _build/figures/<part>/, which the build copies into each book.
FIGURES = Path(__file__).resolve().parents[2] / "_build" / "figures" / Path(__file__).parent.name

Array = NDArray[np.float64]

EPS = 1.0
DELTA = 1e-6
BETA = 0.05  # failure probability of every "all the data lie in this range" statement
N = 1000  # records in the univariate examples
RANGES = (10.0, 1e3, 1e6)  # the bounds R on |μ| in the examples
COARSE_SHARE = 0.25  # share of ε spent on the coarse step, or on all the shrinking rounds together
PARETO_SHAPE = 3.0  # heavy-tailed example: a Pareto law with shape 3 and variance 1
MULTI_N = 100_000
MULTI_DIMS = (10, 100, 1000)
MULTI_RANGE = 1000.0


def normal_cdf(x: float) -> float:
    """Φ(x); erfc keeps full relative precision in the lower tail."""
    return 0.5 * math.erfc(-x / math.sqrt(2))


def normal_pdf(x: float) -> float:
    return math.exp(-x * x / 2) / math.sqrt(2 * math.pi)


# --- Toolkit -----------------------------------------------------------------------------------


def gaussian_spread(n: int, beta: float) -> float:
    """τ = sqrt(2 ln(2n/β)): n draws of N(μ, 1) all lie within μ ± τ with probability ≥ 1 - β.

    One draw leaves μ ± t with probability at most 2e^{-t²/2}; a union bound over n draws.
    """
    return math.sqrt(2 * math.log(2 * n / beta))


def gaussian_max_deviation_sim(rng: np.random.Generator, n: int, trials: int) -> float:
    """Fraction of datasets of n standard normal draws whose largest |x_i| exceeds τ(n, β)."""
    tau = gaussian_spread(n, BETA)
    worst = np.abs(rng.standard_normal((trials, n))).max(axis=1)
    return float(np.mean(worst > tau))


def laplace_tail_bound(scale: float, beta: float) -> float:
    """|Lap(scale)| ≤ scale·ln(1/β) with probability 1 - β."""
    return scale * math.log(1 / beta)


# --- Bounded data ------------------------------------------------------------------------------


def bounded_rmse(n: int, sd: float, width: float, eps: float) -> float:
    """RMSE of the mean of data in an interval of the given width, plus Lap(width/(nε)) noise."""
    return math.sqrt(sd**2 / n + 2 * (width / (n * eps)) ** 2)


def bounded_n_for_rmse(alpha: float, sd: float, width: float, eps: float) -> float:
    """Smallest n with sd²/n + 2 width²/(nε)² ≤ α², from the quadratic in 1/n."""
    a = sd**2
    c = 2 * (width / eps) ** 2
    return (a + math.sqrt(a * a + 4 * c * alpha**2)) / (2 * alpha**2)


# --- Clipped Gaussian moments ------------------------------------------------------------------


def clipped_normal_moments(mu: float, lo: float, hi: float) -> tuple[float, float]:
    """Mean and variance of min(max(X, lo), hi) for X ~ N(μ, 1)."""
    a, b = lo - mu, hi - mu
    pa, pb = normal_cdf(a), normal_cdf(b)
    fa, fb = normal_pdf(a), normal_pdf(b)
    m1 = a * pa + b * (1 - pb) + (fa - fb)
    m2 = a * a * pa + b * b * (1 - pb) + (pb - pa) + (a * fa - b * fb)
    return mu + m1, m2 - m1 * m1


@dataclass(frozen=True)
class Error:
    bias: float
    sampling_sd: float
    noise_sd: float

    @property
    def rmse(self) -> float:
        return math.sqrt(self.bias**2 + self.sampling_sd**2 + self.noise_sd**2)


def clipped_laplace_error(mu: float, lo: float, hi: float, n: int, eps: float) -> Error:
    """Exact error terms of clip to [lo, hi], average, add Lap((hi - lo)/(nε)), for N(μ, 1) data."""
    mean, var = clipped_normal_moments(mu, lo, hi)
    return Error(mean - mu, math.sqrt(var / n), math.sqrt(2) * (hi - lo) / (n * eps))


def clipped_laplace_release(rng: np.random.Generator, x: Array, lo: float, hi: float, eps: float) -> float:
    n = len(x)
    return float(np.clip(x, lo, hi).mean() + rng.laplace(0.0, (hi - lo) / (n * eps)))


# --- The naive estimator -----------------------------------------------------------------------


def naive_interval(r: float, n: int, beta: float) -> tuple[float, float]:
    tau = gaussian_spread(n, beta)
    return -r - tau, r + tau


def naive_error(mu: float, r: float, n: int, eps: float) -> Error:
    lo, hi = naive_interval(r, n, BETA)
    return clipped_laplace_error(mu, lo, hi, n, eps)


# --- Coarse then fine --------------------------------------------------------------------------


def coarse_bins(r: float, n: int, beta: float) -> Array:
    """Edges of bins of width b = 2τ covering [-R - b, R + b]."""
    b = 2 * gaussian_spread(n, beta)
    m = math.ceil(2 * (r + b) / b)
    return -r - b + b * np.arange(m + 1, dtype=np.float64)


def exp_mech_bin_probabilities(counts: Array, eps: float) -> Array:
    """Exponential mechanism with score = count (sensitivity 1 under replacement): weights e^{ε c/2}."""
    z = eps * counts / 2
    w = np.exp(z - z.max())
    return w / w.sum()


def coarse_failure_bound(n_bins: int, n: int, eps: float) -> float:
    """Pr[an empty bin is chosen] ≤ m e^{-ε (n/2)/2}: every empty bin has weight 1, the top one ≥ e^{εn/4}."""
    return n_bins * math.exp(-eps * n / 4)


def coarse_n_needed(n_bins: int, eps: float, beta: float) -> float:
    """n with m e^{-εn/4} = β: n = 4 ln(m/β)/ε."""
    return 4 * math.log(n_bins / beta) / eps


def coarse_choose(rng: np.random.Generator, x: Array, edges: Array, eps: float) -> int:
    counts = np.histogram(x, bins=edges)[0].astype(np.float64)
    p = exp_mech_bin_probabilities(counts, eps)
    return int(rng.choice(len(p), p=p))


def fine_window(edges: Array, j: int) -> tuple[float, float]:
    """The chosen bin and its two neighbours: it holds every point once the bin holds one."""
    b = float(edges[1] - edges[0])
    return float(edges[j]) - b, float(edges[j + 1]) + b


def coarse_fine_error(mu: float, n: int, eps: float, share: float) -> Error:
    """Error terms of the fine step, on the event that the coarse step chose a non-empty bin.

    The window has width 3b = 6τ wherever it lands, so its noise does not depend on R; the
    bias is computed for the window centred on μ (it is negligible for every placement).
    """
    b = 2 * gaussian_spread(n, BETA)
    return clipped_laplace_error(mu, mu - 1.5 * b, mu + 1.5 * b, n, (1 - share) * eps)


def coarse_fine_release(rng: np.random.Generator, x: Array, r: float, eps: float, share: float) -> float:
    edges = coarse_bins(r, len(x), BETA)
    j = coarse_choose(rng, x, edges, share * eps)
    lo, hi = fine_window(edges, j)
    return clipped_laplace_release(rng, x, lo, hi, (1 - share) * eps)


# --- Heavy tails: a Pareto law with variance 1 -------------------------------------------------


def pareto_scale(shape: float) -> float:
    """x_m with variance x_m² a/((a - 1)²(a - 2)) = 1."""
    return math.sqrt((shape - 1) ** 2 * (shape - 2) / shape)


def pareto_mean(shape: float) -> float:
    return shape * pareto_scale(shape) / (shape - 1)


def _pareto_tail_integral(xm: float, a: float, lo: float, hi: float, power: int) -> float:
    """∫_lo^hi p·t^{p-1} Pr[X > t] dt for p = power, Pr[X > t] = (xm/t)^a above xm, 1 below."""
    total = 0.0
    flat_hi = min(hi, xm)
    if flat_hi > lo:
        total += flat_hi**power - lo**power
    start = max(lo, xm)
    if hi > start:
        e = power - a  # ∫ p t^{p-1} xm^a t^{-a} dt = p xm^a t^e/e
        total += power * xm**a * (hi**e - start**e) / e
    return total


def pareto_clipped_moments(shape: float, lo: float, hi: float) -> tuple[float, float]:
    """Mean and variance of min(max(X, lo), hi), X Pareto(x_m, shape), for 0 ≤ lo < hi.

    E[g(clip X)] = g(lo) + ∫_lo^hi g'(t) Pr[X > t] dt, with g(t) = t and g(t) = t².
    """
    xm = pareto_scale(shape)
    m1 = lo + _pareto_tail_integral(xm, shape, lo, hi, 1)
    m2 = lo**2 + _pareto_tail_integral(xm, shape, lo, hi, 2)
    return m1, m2 - m1 * m1


def heavy_error(tau: float, n: int, eps: float, shape: float = PARETO_SHAPE) -> Error:
    """Clip to μ ± τ around the true mean μ, average, add Lap(2τ/(nε))."""
    mu = pareto_mean(shape)
    lo = max(mu - tau, 0.0)
    m, v = pareto_clipped_moments(shape, lo, mu + tau)
    return Error(m - mu, math.sqrt(v / n), math.sqrt(2) * 2 * tau / (n * eps))


def heavy_tau_no_clipping(n: int, beta: float) -> float:
    """Chebyshev and a union bound: all n points within σ·sqrt(n/β) of μ with probability 1 - β."""
    return math.sqrt(n / beta)


def heavy_tau_bound(n: int, eps: float) -> float:
    """Minimizer of σ²/τ + 2√2 τ/(nε) with σ = 1: τ = sqrt(nε/(2√2))."""
    return math.sqrt(n * eps / (2 * math.sqrt(2)))


def heavy_bound_value(n: int, eps: float) -> float:
    """1/√n + min_τ (1/τ + 2√2 τ/(nε)) = 1/√n + 2 sqrt(2√2/(nε))."""
    return 1 / math.sqrt(n) + 2 * math.sqrt(2 * math.sqrt(2) / (n * eps))


def heavy_tau_optimal(n: int, eps: float) -> float:
    """Exact RMSE minimizer for the Pareto example, by golden-section search on log τ."""
    lo, hi = math.log(0.1), math.log(10 * math.sqrt(n))
    g = (math.sqrt(5) - 1) / 2
    for _ in range(200):
        a = hi - g * (hi - lo)
        b = lo + g * (hi - lo)
        if heavy_error(math.exp(a), n, eps).rmse < heavy_error(math.exp(b), n, eps).rmse:
            hi = b
        else:
            lo = a
    return math.exp((lo + hi) / 2)


def heavy_simulated_rmse(rng: np.random.Generator, tau: float, n: int, eps: float, trials: int) -> float:
    xm = pareto_scale(PARETO_SHAPE)
    mu = pareto_mean(PARETO_SHAPE)
    x = xm * rng.uniform(size=(trials, n)) ** (-1 / PARETO_SHAPE)
    est = np.clip(x, max(mu - tau, 0.0), mu + tau).mean(axis=1) + rng.laplace(0.0, 2 * tau / (n * eps), size=trials)
    return float(np.sqrt(np.mean((est - mu) ** 2)))


# --- Many dimensions ---------------------------------------------------------------------------


def gaussian_delta(eps: float, mu: float) -> float:
    """Exact δ(ε) of the Gaussian mechanism whose sensitivity is μ standard deviations (Balle & Wang 2018)."""
    tail = normal_cdf(-mu / 2 - eps / mu)
    weighted = 0.0 if tail == 0 else math.exp(min(eps + math.log(tail), 700.0))  # e^ε Φ(·), overflow-safe
    return normal_cdf(mu / 2 - eps / mu) - weighted


def gaussian_mu(eps: float, delta: float) -> float:
    """The largest sensitivity/σ ratio that is (ε, δ)-DP, by bisection (δ(ε) grows with μ)."""
    lo, hi = 1e-6, 100.0
    for _ in range(200):
        mid = (lo + hi) / 2
        if gaussian_delta(eps, mid) > delta:
            hi = mid
        else:
            lo = mid
    return lo


def gaussian_sigma(l2_sensitivity: float, eps: float, delta: float) -> float:
    """Noise standard deviation of the exactly calibrated Gaussian mechanism: Δ₂/μ(ε, δ)."""
    return l2_sensitivity / gaussian_mu(eps, delta)


def norm_spread(d: int, count: int, beta: float) -> float:
    """√d + sqrt(2 ln(count/β)): `count` draws of N(0, I_d) all have norm below it, w.p. ≥ 1 - β.

    One draw has Pr[‖Z‖ > √d + t] ≤ e^{-t²/2} (Gaussian concentration of the norm); a union bound.
    """
    return math.sqrt(d) + math.sqrt(2 * math.log(count / beta))


@dataclass(frozen=True)
class Plan:
    """The radii of the confidence balls and the noise of each step, fixed before seeing the data."""

    radii: list[float]  # r_0 = R, then one radius per shrinking round
    sigmas: list[float]  # noise sd of each round, then of the final step
    spread: float  # τ_d: every point lies within τ_d of μ

    @property
    def rounds(self) -> int:
        return len(self.radii) - 1


def multi_plan(d: int, n: int, r: float, eps: float, delta: float, share: float, rounds: int, beta: float) -> Plan:
    """Rounds spend share·ε/T and δ/(T+1) each; the final step spends (1 - share)ε and δ/(T+1).

    The next radius bounds the sampling error and the noise of the round, each with probability
    1 - β/(4(T+1)): r' = (√d + u)/√n + σ_t(√d + u) with u = sqrt(2 ln(4(T+1)/β)). The data lie
    within τ_d = √d + sqrt(2 ln(2n/β)) of μ with probability 1 - β/2.
    """
    steps = rounds + 1
    tau = norm_spread(d, 2 * n, beta)
    u = math.sqrt(2 * math.log(4 * steps / beta))
    radii = [r]
    sigmas: list[float] = []
    for _ in range(rounds):
        sigma = gaussian_sigma(2 * (radii[-1] + tau) / n, share * eps / rounds, delta / steps)
        sigmas.append(sigma)
        radii.append((math.sqrt(d) + u) / math.sqrt(n) + sigma * (math.sqrt(d) + u))
    final_eps = (1 - share) * eps if rounds else eps
    sigmas.append(gaussian_sigma(2 * (radii[-1] + tau) / n, final_eps, delta / steps))
    return Plan(radii, sigmas, tau)


@dataclass(frozen=True)
class MultiError:
    sampling: float  # sqrt(E‖X̄ - μ‖²) = sqrt(d/n)
    noise: float  # sqrt(E‖noise‖²) = σ√d

    @property
    def rmse(self) -> float:
        return math.sqrt(self.sampling**2 + self.noise**2)


def plan_error(plan: Plan, d: int, n: int) -> MultiError:
    return MultiError(math.sqrt(d / n), plan.sigmas[-1] * math.sqrt(d))


def multi_best_plan(d: int, n: int, r: float, eps: float, delta: float, share: float, beta: float) -> Plan:
    """The number of rounds (0 = the naive estimator) with the smallest final noise."""
    best = multi_plan(d, n, r, eps, delta, share, 0, beta)
    for rounds in range(1, 30):
        plan = multi_plan(d, n, r, eps, delta, share, rounds, beta)
        if plan.sigmas[-1] < best.sigmas[-1] - 1e-15:
            best = plan
    return best


def project_to_ball(x: Array, centre: Array, radius: float) -> Array:
    """Move every row of x that lies outside the ball onto its surface (ℓ₂ clipping)."""
    diff = x - centre
    norms = np.linalg.norm(diff, axis=1, keepdims=True)
    scale = np.minimum(1.0, radius / np.maximum(norms, 1e-300))
    return centre + diff * scale


def multi_release(rng: np.random.Generator, x: Array, plan: Plan, mu: Array) -> tuple[Array, bool]:
    """Run the plan on the data; also report whether every ball contained the true mean."""
    d = x.shape[1]
    centre: Array = np.zeros(d)
    covered = float(np.linalg.norm(mu - centre)) <= plan.radii[0]
    for t in range(plan.rounds):
        clipped = project_to_ball(x, centre, plan.radii[t] + plan.spread)
        centre = clipped.mean(axis=0) + rng.normal(0.0, plan.sigmas[t], size=d)
        covered = covered and float(np.linalg.norm(mu - centre)) <= plan.radii[t + 1]
    clipped = project_to_ball(x, centre, plan.radii[-1] + plan.spread)
    return clipped.mean(axis=0) + rng.normal(0.0, plan.sigmas[-1], size=d), covered


# --- The printed table -------------------------------------------------------------------------


def coarse_crossover(n: int, eps: float, share: float) -> float:
    """The R above which coarse-then-fine beats the naive estimator, by bisection on log R."""
    lo, hi = 0.0, 8.0
    for _ in range(100):
        mid = (lo + hi) / 2
        r = 10**mid
        if naive_error(r / 2, r, n, eps).rmse < coarse_fine_error(r / 2, n, eps, share).rmse:
            lo = mid
        else:
            hi = mid
    return 10**lo


def print_table(rng: np.random.Generator) -> None:
    tau = gaussian_spread(N, BETA)
    print(f"[Toolkit, n={N}, beta={BETA:g}]")
    print(
        f"  tau=sqrt(2 ln(2n/beta))={tau:.3f}  simulated Pr[max|x_i - mu| > tau]="
        f"{gaussian_max_deviation_sim(rng, N, 20_000):.4f}"
    )

    print("[Bounded data: Bernoulli, eps=1]")
    for n in (100, 1000, 10_000):
        print(f"  n={n}: rmse={bounded_rmse(n, 0.5, 1.0, EPS):.4f}  non-private={0.5 / math.sqrt(n):.4f}")
    for alpha in (0.05, 0.01):
        need = math.ceil(bounded_n_for_rmse(alpha, 0.5, 1.0, EPS))
        print(
            f"  alpha={alpha:g}: n needed={need}  non-private n={0.25 / alpha**2:.0f}  extra={need - 0.25 / alpha**2:.0f}"
        )

    print(f"[Gaussian N(mu,1), n={N}, eps={EPS:g}, |mu| <= R, mu = R/2, coarse share {COARSE_SHARE:g}]")
    share = COARSE_SHARE
    b = 2 * tau
    print(f"  bin width b=2tau={b:.3f}  fine window 3b={3 * b:.2f}  sampling sd={1 / math.sqrt(N):.4f}")
    for r in RANGES:
        mu = r / 2
        naive = naive_error(mu, r, N, EPS)
        m = len(coarse_bins(r, N, BETA)) - 1
        cf = coarse_fine_error(mu, N, EPS, share)
        print(
            f"  R={r:.0e}: naive noise sd={naive.noise_sd:.4f} rmse={naive.rmse:.4f} | bins m={m}  "
            f"coarse failure <= {coarse_failure_bound(m, N, share * EPS):.1e}  "
            f"n needed for beta: {math.ceil(coarse_n_needed(m, share * EPS, BETA))} | fine noise sd={cf.noise_sd:.4f} "
            f"rmse={cf.rmse:.4f}"
        )
    print(f"  coarse-then-fine beats naive from R = {coarse_crossover(N, EPS, share):.2f}")
    for r in RANGES:
        mu = r / 2
        naive_est: list[float] = []
        cf_est: list[float] = []
        for _ in range(4000):
            x = mu + rng.standard_normal(N)
            lo, hi = naive_interval(r, N, BETA)
            naive_est.append(clipped_laplace_release(rng, x, lo, hi, EPS))
            cf_est.append(coarse_fine_release(rng, x, r, EPS, share))
        naive_rmse = float(np.sqrt(np.mean((np.asarray(naive_est) - mu) ** 2)))
        cf_rmse = float(np.sqrt(np.mean((np.asarray(cf_est) - mu) ** 2)))
        print(f"  simulated R={r:.0e}: naive={naive_rmse:.4f}  coarse-fine={cf_rmse:.4f}")

    print(f"[Heavy tails: Pareto shape {PARETO_SHAPE:g}, variance 1, eps={EPS:g}]")
    print(f"  x_m={pareto_scale(PARETO_SHAPE):.4f}  mean={pareto_mean(PARETO_SHAPE):.4f}")
    for n in (1000, 100_000):
        rows = (
            ("no clipping", heavy_tau_no_clipping(n, BETA)),
            ("bound", heavy_tau_bound(n, EPS)),
            ("optimal", heavy_tau_optimal(n, EPS)),
        )
        for label, t in rows:
            e = heavy_error(t, n, EPS)
            sim = heavy_simulated_rmse(rng, t, n, EPS, 4000 if n == 1000 else 400)
            print(
                f"  n={n} {label}: tau={t:.2f}  bias={e.bias:.4f}  sampling sd={e.sampling_sd:.4f}  "
                f"noise sd={e.noise_sd:.4f}  rmse={e.rmse:.4f}  simulated={sim:.4f}"
            )
        print(
            f"  n={n}: worst-case bound on rmse={heavy_bound_value(n, EPS):.4f}  non-private sd={1 / math.sqrt(n):.4f}"
        )

    print(
        f"[Many dimensions: N(mu, I_d), ||mu|| <= R={MULTI_RANGE:g}, n={MULTI_N}, eps={EPS:g}, delta={DELTA:g}, "
        f"beta={BETA:g}]"
    )
    print(f"  mu(eps=1, delta=1e-6)={gaussian_mu(EPS, DELTA):.4f}")
    for d in MULTI_DIMS:
        naive = plan_error(multi_plan(d, MULTI_N, MULTI_RANGE, EPS, DELTA, share, 0, BETA), d, MULTI_N)
        plan = multi_best_plan(d, MULTI_N, MULTI_RANGE, EPS, DELTA, share, BETA)
        shrunk = plan_error(plan, d, MULTI_N)
        print(
            f"  d={d}: non-private={naive.sampling:.4f}  naive noise={naive.noise:.4f} rmse={naive.rmse:.4f}  "
            f"shrinking rounds={plan.rounds} radii=" + ",".join(f"{v:.3g}" for v in plan.radii) + "  "
            f"noise={shrunk.noise:.4f} rmse={shrunk.rmse:.4f}  noise*n*eps/d={shrunk.noise * MULTI_N * EPS / d:.2f}"
        )
    d = 100
    plan = multi_best_plan(d, MULTI_N, MULTI_RANGE, EPS, DELTA, share, BETA)
    mu = np.full(d, MULTI_RANGE / (2 * math.sqrt(d)))  # ‖μ‖ = R/2
    errs: list[float] = []
    covered = 0
    trials = 40
    for _ in range(trials):
        x = mu + rng.standard_normal((MULTI_N, d))
        est, ok = multi_release(rng, x, plan, mu)
        errs.append(float(np.sum((est - mu) ** 2)))
        covered += ok
    print(
        f"  simulated d={d}: rmse={math.sqrt(float(np.mean(errs))):.4f} "
        f"(formula {plan_error(plan, d, MULTI_N).rmse:.4f})  every ball held mu in {covered}/{trials}"
    )

    print(f"[Coarse step on one dataset: R=100, mu=37, n={N}, eps_1={COARSE_SHARE * EPS:g}]")
    x1 = 37.0 + rng.standard_normal(N)
    edges = coarse_bins(100.0, N, BETA)
    counts = np.histogram(x1, bins=edges)[0].astype(np.float64)
    p = exp_mech_bin_probabilities(counts, COARSE_SHARE * EPS)
    top = int(np.argmax(counts))
    empty = counts == 0
    print(
        f"  bins={len(counts)}  non-empty counts=" + ",".join(f"{int(c)}" for c in counts[~empty]) + "  "
        f"Pr[top bin]={float(p[top]):.12f}  Pr[one empty bin]={float(p[empty].max()):.1e}"
    )


# --- Figures -----------------------------------------------------------------------------------


def plot_range(rng: np.random.Generator) -> Figure:
    """RMSE against the bound R for the naive and the coarse-then-fine estimators."""
    rs = [10 ** (i / 4) for i in range(-4, 29)]
    share = COARSE_SHARE
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    ax.plot(rs, [naive_error(r / 2, r, N, EPS).rmse for r in rs], label="clip to [−R − τ, R + τ] (naive)")
    ax.plot(rs, [coarse_fine_error(r / 2, N, EPS, share).rmse for r in rs], label="coarse histogram, then fine")
    ax.axhline(1 / math.sqrt(N), color="k", linestyle=":", label="non-private, 1/√n")
    for r in (1.0, 10.0, 1e3, 1e6):
        mu = r / 2
        cf_est = [coarse_fine_release(rng, mu + rng.standard_normal(N), r, EPS, share) for _ in range(300)]
        ax.plot([r], [float(np.sqrt(np.mean((np.asarray(cf_est) - mu) ** 2)))], "o", color="C1")
    ax.set(
        xscale="log",
        yscale="log",
        xlabel="bound R on |μ| (in standard deviations)",
        ylabel="root-mean-square error",
        title=f"Gaussian mean, n = {N}, ε = 1 (dots: simulation)",
    )
    ax.legend(fontsize=8)
    return fig


def plot_coarse(rng: np.random.Generator) -> Figure:
    """The coarse histogram and the probability that the exponential mechanism picks each bin."""
    r = 100.0
    mu = 37.0
    x = mu + rng.standard_normal(N)
    edges = coarse_bins(r, N, BETA)
    counts = np.histogram(x, bins=edges)[0].astype(np.float64)
    p = exp_mech_bin_probabilities(counts, COARSE_SHARE * EPS)
    centres = (edges[:-1] + edges[1:]) / 2
    width = float(edges[1] - edges[0])
    fig, (top, bottom) = plt.subplots(2, 1, figsize=(7, 5), sharex=True)
    top.bar(centres, counts, width=width * 0.95, color="C0")
    top.axvline(mu, color="k", linestyle=":", label="true mean μ = 37")
    top.set(ylabel="records in the bin", title=f"Coarse step: {len(counts)} bins of width b = {width:.1f}, R = {r:g}")
    top.legend(fontsize=8)
    bottom.semilogy(centres, p, "o", color="C3", markersize=4)
    bottom.set(
        ylim=(1e-60, 10),
        xlabel="value",
        ylabel="probability of choosing it",
        title=r"Exponential mechanism, weights $e^{\varepsilon_1 c/2}$, $\varepsilon_1 = 0.25$",
    )
    return fig


def plot_heavy(rng: np.random.Generator) -> Figure:
    """Error terms against the clip half-width τ for the Pareto example."""
    n = 1000
    taus = [10 ** (-0.5 + i / 40) for i in range(121)]
    errs = [heavy_error(t, n, EPS) for t in taus]
    fig, ax = plt.subplots(figsize=(8.5, 4.2))
    ax.plot(taus, [e.rmse for e in errs], "k", linewidth=2, label="root-mean-square error")
    ax.plot(taus, [abs(e.bias) for e in errs], "--", label="|bias| (exact)")
    ax.plot(taus, [1 / t for t in taus], ":", label="bias bound σ²/τ")
    ax.plot(taus, [e.noise_sd for e in errs], "--", label="noise sd 2√2·τ/(nε)")
    ax.plot(taus, [e.sampling_sd for e in errs], ":", label="sampling sd")
    sim_taus = [1.0, 3.0, 10.0, 30.0, 100.0]
    ax.plot(sim_taus, [heavy_simulated_rmse(rng, t, n, EPS, 2000) for t in sim_taus], "ko", label="simulated")
    ax.axvline(heavy_tau_no_clipping(n, BETA), color="gray", linestyle="-.")
    ax.axvline(heavy_tau_bound(n, EPS), color="gray", linestyle="--")
    ax.set(
        xscale="log",
        yscale="log",
        ylim=(1e-3, 3),
        xlabel="clip half-width τ (standard deviations)",
        ylabel="error",
        title="Pareto data with variance 1, n = 1000, ε = 1",
    )
    ax.legend(fontsize=7, loc="upper left", bbox_to_anchor=(1.01, 1.0), borderaxespad=0.0)
    return fig


def plot_dimension() -> Figure:
    """ℓ₂ error against d: sampling, the naive Gaussian mechanism, and shrinking balls."""
    dims = [round(10 ** (i / 4)) for i in range(4, 17)]
    plans = [multi_best_plan(d, MULTI_N, MULTI_RANGE, EPS, DELTA, COARSE_SHARE, BETA) for d in dims]
    naive = [
        plan_error(multi_plan(d, MULTI_N, MULTI_RANGE, EPS, DELTA, COARSE_SHARE, 0, BETA), d, MULTI_N).rmse
        for d in dims
    ]
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    ax.plot(dims, naive, "o-", label=rf"clip to the ball of radius $R + \tau_d$, $R = {MULTI_RANGE:g}$ (naive)")
    ax.plot(dims, [plan_error(p, d, MULTI_N).rmse for p, d in zip(plans, dims)], "s-", label="shrinking balls")
    ax.plot(dims, [math.sqrt(d / MULTI_N) for d in dims], "k:", label="non-private, sqrt(d/n)")
    ax.set(
        xscale="log",
        yscale="log",
        xlabel="dimension d",
        ylabel="ℓ₂ root-mean-square error",
        title=f"Gaussian mean in d dimensions, n = {MULTI_N:,}, (1, 10⁻⁶)-DP",
    )
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
    out.mkdir(parents=True, exist_ok=True)
    figures = {
        "mean_range.svg": plot_range(rng),
        "mean_coarse.svg": plot_coarse(rng),
        "mean_heavy.svg": plot_heavy(rng),
        "mean_dimension.svg": plot_dimension(),
    }
    for name, fig in figures.items():
        fig.tight_layout()
        fig.savefig(out / name, metadata={"Date": None})  # no timestamp, so output is reproducible
        plt.close(fig)
        print(f"saved {out / name}")


if __name__ == "__main__":
    main()
