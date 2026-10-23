"""Check the results in the hypothesis-testing chapter and draw its figures.

    uv run python code/part01-prerequisites/hypothesis_testing.py
    uv run python code/part01-prerequisites/hypothesis_testing.py --seed 1 --out /tmp/figures

Prints every number quoted in 04-hypothesis-testing.md (they match exactly with the
default seed), then saves the chapter's figures.
"""

import argparse
import math
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
matplotlib.rcParams["svg.hashsalt"] = "hypothesis_testing"  # stable element ids, so reruns don't churn the SVGs
matplotlib.rcParams["svg.fonttype"] = "none"  # keep text as text: smaller, diffable files

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from numpy.typing import NDArray

# Figures are build output: _build/figures/<part>/, which the build copies into each book.
FIGURES = Path(__file__).resolve().parents[2] / "_build" / "figures" / Path(__file__).parent.name

# --- Normal distribution: CDF, tail and its inverse (no SciPy) ------------------------------


def normal_cdf(x: float) -> float:
    """Φ(x) = Pr[Z <= x] for Z ~ N(0, 1)."""
    return 0.5 * math.erfc(-x / math.sqrt(2))


def normal_sf(x: float) -> float:
    """1 - Φ(x) = Pr[Z > x]; erfc keeps full relative precision in the upper tail."""
    return 0.5 * math.erfc(x / math.sqrt(2))


def normal_pdf(x: float) -> float:
    return math.exp(-x * x / 2) / math.sqrt(2 * math.pi)


def normal_isf(q: float) -> float:
    """Φ⁻¹(1 - q): the x with Pr[Z > x] = q, by bisection on the decreasing tail.

    Solving for the tail directly, instead of computing Φ⁻¹ at 1 - q, keeps precision
    when q is tiny (1 - q would round to 1).
    """
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


def normal_ppf(p: float) -> float:
    """Φ⁻¹(p), using the symmetry Φ⁻¹(p) = -Φ⁻¹(1 - p)."""
    return -normal_isf(p)


# --- Laplace distribution -------------------------------------------------------------------


def laplace_cdf(x: float, mu: float = 0.0, b: float = 1.0) -> float:
    z = (x - mu) / b
    return 0.5 * math.exp(z) if z < 0 else 1 - 0.5 * math.exp(-z)


def laplace_pdf(x: float, mu: float = 0.0, b: float = 1.0) -> float:
    return math.exp(-abs(x - mu) / b) / (2 * b)


# --- A discrete example: suspicious-word counts in ham (P) and spam (Q) ----------------------

OUTCOMES = (0, 1, 2, 3)
HAM = (0.4, 0.3, 0.2, 0.1)  # P: the null hypothesis
SPAM = (0.1, 0.2, 0.3, 0.4)  # Q: the alternative


@dataclass(frozen=True)
class Errors:
    alpha: float  # type I error: E_P[φ]
    beta: float  # type II error: 1 - E_Q[φ]

    @property
    def power(self) -> float:
        return 1 - self.beta


def discrete_errors(phi: tuple[float, ...], p: tuple[float, ...], q: tuple[float, ...]) -> Errors:
    """Errors of a randomized test φ(x) = Pr[reject H0 | x] on a finite space."""
    alpha = sum(f * px for f, px in zip(phi, p, strict=True))
    power = sum(f * qx for f, qx in zip(phi, q, strict=True))
    return Errors(alpha, 1 - power)


def likelihood_ratios(p: tuple[float, ...], q: tuple[float, ...]) -> list[float]:
    return [qx / px for px, qx in zip(p, q, strict=True)]


@dataclass(frozen=True)
class NeymanPearsonTest:
    eta: float  # threshold on the likelihood ratio
    gamma: float  # rejection probability on the tie set {q = η p}
    phi: tuple[float, ...]


def neyman_pearson_test(alpha: float, p: tuple[float, ...], q: tuple[float, ...]) -> NeymanPearsonTest:
    """The likelihood-ratio test of size exactly alpha (all p > 0 assumed).

    Walk down the distinct likelihood ratios from the largest. Reject every outcome
    above the threshold, and randomize on the outcomes at it to use up the rest of α.
    """
    lr = likelihood_ratios(p, q)
    used = 0.0
    for eta in sorted(set(lr), reverse=True):
        tie_mass = sum(px for px, r in zip(p, lr, strict=True) if r == eta)
        if used + tie_mass >= alpha - 1e-15:
            gamma = (alpha - used) / tie_mass
            phi = tuple(1.0 if r > eta else gamma if r == eta else 0.0 for r in lr)
            return NeymanPearsonTest(eta, gamma, phi)
        used += tie_mass
    raise ValueError("alpha must lie in [0, 1]")


def random_tests(rng: np.random.Generator, n: int, size: int) -> NDArray[np.float64]:
    """n random tests on a space of `size` outcomes: each φ(x) uniform on [0, 1]."""
    return rng.random((n, size))


def best_random_power(rng: np.random.Generator, alpha: float, trials: int) -> tuple[float, int]:
    """Largest power among random tests whose size is at most alpha, and how many qualified."""
    phis = random_tests(rng, trials, len(OUTCOMES))
    sizes = phis @ np.array(HAM)
    powers = phis @ np.array(SPAM)
    ok = sizes <= alpha
    return float(np.max(powers[ok])), int(np.sum(ok))


def total_variation(p: tuple[float, ...], q: tuple[float, ...]) -> float:
    return 0.5 * sum(abs(px - qx) for px, qx in zip(p, q, strict=True))


# --- Gaussian shift: N(0, 1) against N(μ, 1) -------------------------------------------------


def gaussian_tradeoff(alpha: float, mu: float) -> float:
    """T(α) = Φ(Φ⁻¹(1 - α) - μ): the smallest type II error at type I error α."""
    if alpha <= 0.0:
        return 1.0
    if alpha >= 1.0:
        return 0.0
    return normal_cdf(normal_isf(alpha) - mu)


def gaussian_tv(mu: float) -> float:
    """TV(N(0,1), N(μ,1)) = 2Φ(μ/2) - 1, attained by the threshold at μ/2."""
    return 2 * normal_cdf(abs(mu) / 2) - 1


def gaussian_mu_for_tv(tv: float) -> float:
    """The shift μ whose Gaussian pair has the given total variation."""
    return 2 * normal_isf((1 - tv) / 2)


def empirical_threshold_errors(
    rng: np.random.Generator, sample_p: str, sample_q: str, shift: float, t: float, n: int
) -> Errors:
    """Simulated errors of 'reject when x > t' for P = base(0, 1) and Q = base(shift, 1)."""
    x_p = rng.normal(0, 1, n) if sample_p == "normal" else rng.laplace(0, 1, n)
    x_q = shift + (rng.normal(0, 1, n) if sample_q == "normal" else rng.laplace(0, 1, n))
    return Errors(float(np.mean(x_p > t)), float(np.mean(x_q <= t)))


def samples_for_power(alpha: float, power: float, effect: float) -> int:
    """Smallest n with power >= target, for the mean of n draws: μ = effect·√n."""
    return math.ceil(((normal_isf(alpha) + normal_isf(1 - power)) / effect) ** 2)


def stein_rate(alpha: float, mu: float, n: int) -> float:
    """-ln(β_n)/n for the optimal size-α test on n draws: β_n = Φ(Φ⁻¹(1-α) - μ√n)."""
    return -math.log(gaussian_tradeoff(alpha, mu * math.sqrt(n))) / n


# --- Laplace shift: Lap(0, b) against Lap(s, b) ----------------------------------------------


def laplace_log_lr(x: float, s: float, b: float = 1.0) -> float:
    """ln q(x)/p(x) for P = Lap(0, b), Q = Lap(s, b): clipped to [-s/b, s/b]."""
    return (abs(x) - abs(x - s)) / b


def laplace_tradeoff(alpha: float, kappa: float) -> float:
    """T(α) for Lap(0, b) against Lap(s, b), with kappa = s/b: three pieces."""
    if alpha <= 0.0:
        return 1.0
    if alpha >= 1.0:
        return 0.0
    if alpha < math.exp(-kappa) / 2:
        return 1 - math.exp(kappa) * alpha
    if alpha <= 0.5:
        return math.exp(-kappa) / (4 * alpha)
    return math.exp(-kappa) * (1 - alpha)


def laplace_threshold(alpha: float) -> float:
    """t with Pr[Lap(0, 1) > t] = α."""
    return -math.log(2 * alpha) if alpha <= 0.5 else math.log(2 * (1 - alpha))


def laplace_tv(kappa: float) -> float:
    return 1 - math.exp(-kappa / 2)


# --- Advantage and total variation ------------------------------------------------------------


def best_advantage_on_grid(tradeoff: list[float], alphas: list[float]) -> tuple[float, float]:
    """max over a grid of 1 - α - T(α), and the α where it is attained."""
    gaps = [1 - a - t for a, t in zip(alphas, tradeoff, strict=True)]
    i = max(range(len(gaps)), key=lambda j: gaps[j])
    return gaps[i], alphas[i]


# --- ROC, AUC and low false-positive rates -----------------------------------------------------


def binormal_tpr(fpr: float, m: float, sigma: float) -> float:
    """TPR of 'score > t' when negatives score N(0, 1) and positives N(m, σ²)."""
    return normal_sf((normal_isf(fpr) - m) / sigma)


def binormal_auc(m: float, sigma: float) -> float:
    """AUC = Pr[S_pos > S_neg], and S_pos - S_neg ~ N(m, 1 + σ²)."""
    return normal_cdf(m / math.sqrt(1 + sigma * sigma))


def mixture_tpr(fpr: float, w: float, far: float) -> float:
    """Positives are N(0, 1) except a fraction w that scores N(far, 1)."""
    return (1 - w) * fpr + w * normal_sf(normal_isf(fpr) - far)


def mixture_auc(w: float, far: float) -> float:
    return (1 - w) * 0.5 + w * binormal_auc(far, 1.0)


def empirical_auc(rng: np.random.Generator, m: float, sigma: float, n: int) -> float:
    """Fraction of independent (positive, negative) pairs ranked correctly."""
    return float(np.mean(rng.normal(m, sigma, n) > rng.normal(0, 1, n)))


# --- Bayesian view ---------------------------------------------------------------------------


def posterior(prior: float, likelihood_ratio: float) -> float:
    """Posterior Pr[H1 | evidence] from posterior odds = prior odds × likelihood ratio."""
    odds = prior / (1 - prior) * likelihood_ratio
    return odds / (1 + odds)


def bayes_threshold_gaussian(prior: float, mu: float) -> float:
    """Say H1 iff LR(x) > (1-π)/π; for the Gaussian pair that is x > μ/2 + ln((1-π)/π)/μ."""
    return mu / 2 + math.log((1 - prior) / prior) / mu


# --- Error rates from finite trials ------------------------------------------------------------


def binomial_cdf(k: int, n: int, p: float) -> float:
    """Pr[Bin(n, p) <= k], for 0 < p < 1, summing the k + 1 terms in log space."""
    if k < 0:
        return 0.0
    if k >= n:
        return 1.0
    log_n = math.lgamma(n + 1)
    log_p, log_q = math.log(p), math.log1p(-p)
    return math.fsum(
        math.exp(log_n - math.lgamma(j + 1) - math.lgamma(n - j + 1) + j * log_p + (n - j) * log_q)
        for j in range(k + 1)
    )


def _clopper_pearson_end(k: int, n: int, tail: float, upper: bool) -> float:
    """Bisection for one end of the Clopper–Pearson interval.

    Upper end: the u with Pr[Bin(n, u) <= k] = tail (this probability decreases in u).
    Lower end: the l with Pr[Bin(n, l) >= k] = tail (this probability increases in l).
    """
    lo, hi = 0.0, 1.0
    for _ in range(100):
        mid = (lo + hi) / 2
        if upper:
            go_right = binomial_cdf(k, n, mid) > tail
        else:
            go_right = 1 - binomial_cdf(k - 1, n, mid) < tail
        if go_right:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def clopper_pearson(k: int, n: int, level: float = 0.95) -> tuple[float, float]:
    """Exact two-sided interval for a binomial proportion from k successes in n trials."""
    a = (1 - level) / 2
    lower = 0.0 if k == 0 else _clopper_pearson_end(k, n, a, upper=False)
    upper = 1.0 if k == n else _clopper_pearson_end(k, n, a, upper=True)
    return lower, upper


def zero_count_upper(n: int, level: float = 0.95) -> float:
    """One-sided upper bound when 0 events are seen in n trials: solve (1 - u)^n = 1 - level."""
    return 1 - (1 - level) ** (1 / n)


def hoeffding_halfwidth(n: int, level: float = 0.95) -> float:
    """Two-sided Hoeffding half-width for a mean of n variables in [0, 1]."""
    return math.sqrt(math.log(2 / (1 - level)) / (2 * n))


def wald_interval(k: int, n: int, level: float = 0.95) -> tuple[float, float]:
    """Normal-approximation interval p̂ ± z sqrt(p̂(1-p̂)/n)."""
    p = k / n
    half = normal_isf((1 - level) / 2) * math.sqrt(p * (1 - p) / n)
    return p - half, p + half


def clopper_pearson_coverage(rng: np.random.Generator, p: float, n: int, reps: int) -> float:
    """Fraction of simulated experiments whose interval contains the true rate."""
    counts = rng.binomial(n, p, size=reps)
    cache: dict[int, tuple[float, float]] = {}
    hits = 0
    for k in counts.tolist():
        if k not in cache:
            cache[k] = clopper_pearson(k, n)
        lo, hi = cache[k]
        hits += lo <= p <= hi
    return hits / reps


# --- The chapter's table -----------------------------------------------------------------------


def print_table(rng: np.random.Generator) -> None:
    print("== Discrete example: ham P vs spam Q ==")
    lr = likelihood_ratios(HAM, SPAM)
    print("likelihood ratios q/p: " + "  ".join(f"{x}:{r:.3f}" for x, r in zip(OUTCOMES, lr, strict=True)))
    for name, phi in (
        ("reject none", (0.0, 0.0, 0.0, 0.0)),
        ("reject {3}", (0.0, 0.0, 0.0, 1.0)),
        ("reject {2,3}", (0.0, 0.0, 1.0, 1.0)),
        ("reject {1,2,3}", (0.0, 1.0, 1.0, 1.0)),
        ("reject all", (1.0, 1.0, 1.0, 1.0)),
        ("reject {0}", (1.0, 0.0, 0.0, 0.0)),
        ("reject {0,3}", (1.0, 0.0, 0.0, 1.0)),
    ):
        e = discrete_errors(phi, HAM, SPAM)
        print(f"[{name:15s}] alpha={e.alpha:.2f}  beta={e.beta:.2f}  power={e.power:.2f}")
    np_test = neyman_pearson_test(0.25, HAM, SPAM)
    e = discrete_errors(np_test.phi, HAM, SPAM)
    print(
        f"[NP alpha=0.25] eta={np_test.eta:.2f}  gamma={np_test.gamma:.2f}  "
        f"alpha={e.alpha:.3f}  power={e.power:.3f}  beta={e.beta:.3f}"
    )
    best, count = best_random_power(rng, 0.25, trials=200_000)
    print(f"[random tests, size <= 0.25] qualifying={count}  best power={best:.3f}")
    print(f"[TV ham/spam] {total_variation(HAM, SPAM):.2f}")

    print("== Inverse normal ==")
    print(
        f"Phi^-1(0.95)={normal_isf(0.05):.4f}  Phi^-1(0.999)={normal_isf(1e-3):.4f}  "
        f"Phi^-1(0.975)={normal_isf(0.025):.4f}  Phi^-1(0.99)={normal_isf(0.01):.4f}  "
        f"Phi^-1(0.9)={normal_isf(0.1):.4f}"
    )

    print("== Gaussian trade-off T(alpha) = Phi(Phi^-1(1-alpha) - mu) ==")
    for mu in (1.0, 2.0):
        for alpha in (0.05, 1e-3):
            t = gaussian_tradeoff(alpha, mu)
            print(f"[mu={mu:g} alpha={alpha:g}] threshold={normal_isf(alpha):.4f}  beta={t:.4f}  power={1 - t:.4f}")
    sim = empirical_threshold_errors(rng, "normal", "normal", 2.0, normal_isf(0.05), n=1_000_000)
    print(f"[simulated mu=2 alpha=0.05] alpha={sim.alpha:.4f}  beta={sim.beta:.4f}")
    print(
        f"[symmetry mu=1] T(0.05)={gaussian_tradeoff(0.05, 1.0):.4f}  "
        f"T(T(0.05))={gaussian_tradeoff(gaussian_tradeoff(0.05, 1.0), 1.0):.4f}"
    )
    print(f"[samples for power 0.9 at alpha=0.01, effect 0.25] n={samples_for_power(0.01, 0.9, 0.25)}")

    print("== Laplace trade-off, kappa = s/b = 1 ==")
    kappa = 1.0
    print(
        f"[breakpoints] e^-kappa/2={math.exp(-kappa) / 2:.4f}  slope ends={-math.exp(kappa):.4f}, {-math.exp(-kappa):.4f}"
    )
    for alpha in (1e-3, 0.05, 0.25, 0.75):
        t = laplace_tradeoff(alpha, kappa)
        print(f"[Laplace kappa=1 alpha={alpha:g}] beta={t:.4f}  power={1 - t:.4f}")
    sim = empirical_threshold_errors(rng, "laplace", "laplace", 1.0, laplace_threshold(0.25), n=1_000_000)
    print(f"[simulated Laplace alpha=0.25] alpha={sim.alpha:.4f}  beta={sim.beta:.4f}")
    tv_l = laplace_tv(kappa)
    mu_match = gaussian_mu_for_tv(tv_l)
    print(f"[TV Laplace kappa=1] {tv_l:.4f}  matching Gaussian mu={mu_match:.4f}")
    for alpha in (1e-3, 1e-6):
        print(
            f"[same TV, alpha={alpha:g}] Laplace power={1 - laplace_tradeoff(alpha, kappa):.3e}  "
            f"Gaussian power={1 - gaussian_tradeoff(alpha, mu_match):.3e}"
        )

    print("== Best advantage = TV ==")
    alphas = [i / 100_000 for i in range(100_001)]
    gap, at = best_advantage_on_grid([gaussian_tradeoff(a, 1.0) for a in alphas], alphas)
    print(
        f"[Gaussian mu=1] TV formula={gaussian_tv(1.0):.4f}  grid max gap={gap:.4f} at alpha={at:.4f}  "
        f"(1 - Phi(1/2)={normal_sf(0.5):.4f})"
    )
    gap, at = best_advantage_on_grid([laplace_tradeoff(a, 1.0) for a in alphas], alphas)
    print(f"[Laplace kappa=1] TV formula={laplace_tv(1.0):.4f}  grid max gap={gap:.4f} at alpha={at:.4f}")

    print("== AUC versus low FPR ==")
    m_a, s_b = 1.0, 2.0
    auc = binormal_auc(m_a, 1.0)
    m_b = m_a * math.sqrt((1 + s_b**2) / 2)
    print(
        f"[A: N(0,1) vs N(1,1)] AUC={auc:.4f}  TPR@1e-3={binormal_tpr(1e-3, m_a, 1.0):.4f}  "
        f"TPR@0.1={binormal_tpr(0.1, m_a, 1.0):.4f}  empirical AUC={empirical_auc(rng, m_a, 1.0, 1_000_000):.4f}"
    )
    print(
        f"[B: N(0,1) vs N({m_b:.4f},4)] AUC={binormal_auc(m_b, s_b):.4f}  TPR@1e-3={binormal_tpr(1e-3, m_b, s_b):.4f}  "
        f"TPR@0.1={binormal_tpr(0.1, m_b, s_b):.4f}  empirical AUC={empirical_auc(rng, m_b, s_b, 1_000_000):.4f}"
    )
    w, far = 0.05, 6.0
    mix_auc = mixture_auc(w, far)
    mu_same = math.sqrt(2) * normal_ppf(mix_auc)
    print(
        f"[mixture w={w} far={far:g}] AUC={mix_auc:.4f}  TPR@1e-3={mixture_tpr(1e-3, w, far):.4f}  "
        f"equal-AUC Gaussian mu={mu_same:.4f} TPR@1e-3={binormal_tpr(1e-3, mu_same, 1.0):.4f}"
    )

    print("== Bayesian view ==")
    prior, mu = 0.01, 2.0
    for alpha in (0.05, 1e-3):
        power = 1 - gaussian_tradeoff(alpha, mu)
        print(
            f"[prevalence {prior} mu={mu:g} alpha={alpha:g}] LR of a positive={power / alpha:.1f}  "
            f"posterior={posterior(prior, power / alpha):.3f}"
        )
    t = bayes_threshold_gaussian(prior, mu)
    print(f"[Bayes rule prior={prior}] threshold={t:.4f}  alpha={normal_sf(t):.2e}  power={normal_sf(t - mu):.4f}")
    print(f"[equal priors mu=1] Bayes error={0.5 * (1 - gaussian_tv(1.0)):.4f}")

    print("== Error rates from finite trials ==")
    n = 10_000
    lo, hi = clopper_pearson(10, n)
    print(f"[CP k=10 n={n}] [{lo:.5f}, {hi:.5f}]")
    lo, hi = clopper_pearson(100, 100_000)
    print(f"[CP k=100 n=100000] [{lo:.5f}, {hi:.5f}]  relative [{lo / 1e-3:.2f}, {hi / 1e-3:.2f}]")
    wlo, whi = wald_interval(10, n)
    print(f"[Wald k=10 n={n}] [{wlo:.5f}, {whi:.5f}]  Wald k=0: {wald_interval(0, n)}")
    print(f"[Hoeffding n={n}] half-width={hoeffding_halfwidth(n):.4f}")
    print(
        f"[zero count n=3000] upper={zero_count_upper(3000):.3e}  3/n={3 / 3000:.3e}  "
        f"n needed for 1e-3: {math.ceil(math.log(20) / -math.log1p(-1e-3))}"
    )
    cov = clopper_pearson_coverage(rng, 1e-3, n, reps=2_000)
    print(f"[CP coverage p=1e-3 n={n} reps=2000] {cov:.4f}")
    t = normal_isf(1e-3)
    x_neg = rng.normal(0, 1, n)
    x_pos = rng.normal(2, 1, n)
    k_fp, k_tp = int(np.sum(x_neg > t)), int(np.sum(x_pos > t))
    lo_f, hi_f = clopper_pearson(k_fp, n)
    lo_t, hi_t = clopper_pearson(k_tp, n)
    print(
        f"[one run mu=2 n={n}] FP={k_fp}  FPR CI=[{lo_f:.5f}, {hi_f:.5f}]  TP={k_tp}  TPR={k_tp / n:.4f}  "
        f"TPR CI=[{lo_t:.4f}, {hi_t:.4f}]"
    )

    print("== Stein exponent, mu=1, alpha=0.05 (KL = mu^2/2 = 0.5) ==")
    print("  ".join(f"n={k}: {stein_rate(0.05, 1.0, k):.4f}" for k in (10, 100, 1000)))
    print(f"[uniform shift s=0.2] T(0)={1 - 0.2:.1f}  TV={0.2:.1f}")


# --- Figures -----------------------------------------------------------------------------------


def plot_errors(mu: float = 2.0, alpha: float = 0.05) -> Figure:
    t = normal_isf(alpha)
    xs = [-4 + i * 0.01 for i in range(1001)]
    p = [normal_pdf(x) for x in xs]
    q = [normal_pdf(x - mu) for x in xs]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(xs, p, label="P = N(0, 1) (null)")
    ax.plot(xs, q, label=f"Q = N({mu:g}, 1) (alternative)")
    right = [x >= t for x in xs]
    left = [x <= t for x in xs]
    ax.fill_between(xs, 0, p, where=right, alpha=0.35, color="C0", label=rf"type I error $a$ = {alpha:g}")
    ax.fill_between(
        xs,
        0,
        q,
        where=left,
        alpha=0.35,
        color="C1",
        label=rf"type II error $b$ = {gaussian_tradeoff(alpha, mu):.3f}",
    )
    ax.axvline(t, color="k", linestyle="--", label=rf"threshold $\tau$ = {t:.3f}")
    ax.set(xlabel="observation x", ylabel="density", ylim=(0, 0.5), title=r"Reject the null when $x > \tau$")
    ax.legend(loc="upper left", fontsize=8)
    return fig


def plot_discrete_tradeoff(rng: np.random.Generator) -> Figure:
    alphas = [i / 1000 for i in range(1001)]
    curve = [discrete_errors(neyman_pearson_test(a, HAM, SPAM).phi, HAM, SPAM).beta for a in alphas]
    phis = random_tests(rng, 400, len(OUTCOMES))
    rand = [discrete_errors(tuple(float(v) for v in row), HAM, SPAM) for row in phis]
    vertices = [
        discrete_errors(phi, HAM, SPAM)
        for phi in (
            (0.0, 0.0, 0.0, 0.0),
            (0.0, 0.0, 0.0, 1.0),
            (0.0, 0.0, 1.0, 1.0),
            (0.0, 1.0, 1.0, 1.0),
            (1.0, 1.0, 1.0, 1.0),
        )
    ]
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter([e.alpha for e in rand], [e.beta for e in rand], s=6, color="0.7", label="400 random tests")
    ax.plot(alphas, curve, "C0", linewidth=2, label="likelihood-ratio tests (Neyman–Pearson)")
    ax.plot(
        [e.alpha for e in vertices], [e.beta for e in vertices], "ko", label="deterministic LR tests (reject count ≥ k)"
    )
    np_e = discrete_errors(neyman_pearson_test(0.25, HAM, SPAM).phi, HAM, SPAM)
    ax.plot([np_e.alpha], [np_e.beta], "C3s", label=rf"randomized, $a$ = 0.25, $b$ = {np_e.beta:.3f}")
    ax.plot([0, 1], [1, 0], color="gray", linestyle=":", label=r"ignore the data: $b = 1 - a$")
    ax.set(
        xlabel=r"type I error $a$",
        ylabel=r"type II error $b$",
        xlim=(0, 1),
        ylim=(0, 1),
        title="Ham vs spam word counts: no test beats the LR curve",
    )
    ax.legend(loc="upper right", fontsize=8)
    return fig


def plot_gaussian_tradeoff(rng: np.random.Generator) -> Figure:
    alphas = [i / 1000 for i in range(1001)]
    fig, ax = plt.subplots(figsize=(6, 5))
    for mu in (0.0, 0.5, 1.0, 2.0, 3.0):
        ax.plot(alphas, [gaussian_tradeoff(a, mu) for a in alphas], label=rf"$\mu$ = {mu:g}")
    thresholds = [-1.0 + 0.5 * i for i in range(9)]
    sims = [empirical_threshold_errors(rng, "normal", "normal", 1.0, t, n=20_000) for t in thresholds]
    ax.plot([e.alpha for e in sims], [e.beta for e in sims], "kx", label=r"simulated thresholds, $\mu$ = 1")
    ax.set(
        xlabel=r"type I error $a$",
        ylabel=r"type II error $T(a)$",
        xlim=(0, 1),
        ylim=(0, 1),
        title=r"$T(a) = \Phi(\Phi^{-1}(1-a) - \mu)$",
    )
    ax.legend(loc="upper right")
    return fig


def plot_laplace_tradeoff(kappa: float = 1.0) -> Figure:
    xs = [-3 + i * 0.01 for i in range(701)]
    mu_match = gaussian_mu_for_tv(laplace_tv(kappa))
    fig, (left, right) = plt.subplots(1, 2, figsize=(11, 4.5))
    left.plot(xs, [laplace_log_lr(x, kappa) for x in xs], label=r"Laplace, $\kappa = m_1/s = 1$")
    left.plot(xs, [mu_match * x - mu_match**2 / 2 for x in xs], label=rf"Gaussian, $\mu$ = {mu_match:.2f}")
    left.axhline(kappa, color="gray", linestyle=":")
    left.axhline(-kappa, color="gray", linestyle=":")
    left.set(
        xlabel="observation x",
        ylabel=r"log likelihood ratio $\ln(q(x)/p(x))$",
        ylim=(-4, 4),
        title="Laplace LR is clipped; Gaussian LR is not",
    )
    left.legend(loc="upper left")
    alphas = [10 ** (-6 + 6 * i / 600) for i in range(601)]
    right.plot(alphas, [1 - laplace_tradeoff(a, kappa) for a in alphas], label=r"Laplace, $\kappa = m_1/s = 1$")
    right.plot(alphas, [1 - gaussian_tradeoff(a, mu_match) for a in alphas], label=rf"Gaussian, $\mu$ = {mu_match:.2f}")
    right.plot(
        alphas,
        [min(1.0, math.exp(kappa) * a) for a in alphas],
        color="gray",
        linestyle=":",
        label=r"$e^{\kappa}a$",
    )
    right.plot(alphas, alphas, color="k", linestyle=":", linewidth=0.8, label=r"chance: power = $a$")
    right.set(
        xscale="log",
        yscale="log",
        xlim=(1e-6, 1),
        ylim=(1e-6, 1.5),
        xlabel=r"type I error $a$ (log scale)",
        ylabel=r"power $1 - T(a)$ (log scale)",
        title="Same total variation, different power at small a",
    )
    right.legend(loc="lower right", fontsize=8)
    return fig


def plot_tv_advantage(mu: float = 1.0) -> Figure:
    alphas = [i / 1000 for i in range(1001)]
    curve = [gaussian_tradeoff(a, mu) for a in alphas]
    a_star = normal_sf(mu / 2)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(alphas, curve, label=rf"$T(a)$, Gaussian $\mu$ = {mu:g}")
    ax.plot([0, 1], [1, 0], color="gray", linestyle=":", label=r"$1 - a$")
    ax.plot(
        [a_star, a_star],
        [gaussian_tradeoff(a_star, mu), 1 - a_star],
        "C3",
        linewidth=2,
        label=rf"largest gap = TV = {gaussian_tv(mu):.4f}",
    )
    ax.plot(
        alphas,
        [1 - a - c for a, c in zip(alphas, curve, strict=True)],
        "C2--",
        label=r"advantage $1 - a - T(a)$",
    )
    ax.set(
        xlabel=r"type I error $a$",
        ylabel="error / advantage",
        xlim=(0, 1),
        ylim=(0, 1),
        title="Best advantage equals total variation",
    )
    ax.legend(loc="upper right", fontsize=9)
    return fig


def plot_auc_low_fpr() -> Figure:
    m_b = math.sqrt(5 / 2)
    fprs = [10 ** (-5 + 5 * i / 500) for i in range(500)]
    lin = [i / 500 for i in range(1, 500)]
    fig, (left, right) = plt.subplots(1, 2, figsize=(11, 4.5))
    for fp, ax, label_scale in ((lin, left, "linear"), (fprs, right, "log")):
        ax.plot(
            fp,
            [binormal_tpr(f, 1.0, 1.0) for f in fp],
            label=f"A: positives N(1, 1), AUC = {binormal_auc(1.0, 1.0):.3f}",
        )
        ax.plot(
            fp,
            [binormal_tpr(f, m_b, 2.0) for f in fp],
            label=rf"B: positives N({m_b:.2f}, $2^2$), AUC = {binormal_auc(m_b, 2.0):.3f}",
        )
        ax.plot(fp, fp, color="k", linestyle=":", linewidth=0.8, label="chance")
        if label_scale == "log":
            ax.axvline(1e-3, color="gray", linestyle="--", label="FPR = $10^{-3}$")
            ax.set(xscale="log", yscale="log", ylim=(1e-5, 1.2), xlim=(1e-5, 1))
        else:
            ax.set(xlim=(0, 1), ylim=(0, 1))
        ax.set(xlabel=f"false-positive rate ({label_scale} scale)", ylabel=f"true-positive rate ({label_scale} scale)")
        ax.legend(loc="lower right", fontsize=8)
    left.set_title("Linear axes: equal areas under the curves")
    ratio = binormal_tpr(1e-3, m_b, 2.0) / binormal_tpr(1e-3, 1.0, 1.0)
    right.set_title(f"Log axes: {ratio:.0f}× apart at FPR = $10^{{-3}}$")
    return fig


def plot_posterior() -> Figure:
    priors = [10 ** (-4 + 3.7 * i / 400) for i in range(401)]
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    for alpha in (0.05, 1e-3):
        power = 1 - gaussian_tradeoff(alpha, 2.0)
        ax.plot(priors, [posterior(p, power / alpha) for p in priors], label=rf"FPR = {alpha:g}, TPR = {power:.3f}")
    ax.axvline(0.01, color="gray", linestyle="--", label="prevalence 1%")
    ax.set(
        xscale="log",
        ylim=(0, 1),
        xlabel=r"prior $\Pr[H_1]$ (log scale)",
        ylabel=r"$\Pr[H_1 \mid \mathrm{positive}]$",
        title=r"Posterior after a positive, $\mu$ = 2 test",
    )
    ax.legend(loc="upper left")
    return fig


def plot_rate_ci(p: float = 1e-3) -> Figure:
    counts = sorted({round(10 ** (3 * i / 40)) for i in range(41)})
    ns = [round(k / p) for k in counts]
    bounds = [clopper_pearson(k, n) for k, n in zip(counts, ns, strict=True)]
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    ax.fill_between(
        ns, [b[0] for b in bounds], [b[1] for b in bounds], alpha=0.35, label="Clopper–Pearson 95% interval"
    )
    ax.plot(ns, [p + hoeffding_halfwidth(n) for n in ns], "C1--", label="Hoeffding upper end")
    ax.axhline(p, color="k", linestyle=":", label=f"observed rate {p:g}")
    ax.set(
        xscale="log",
        yscale="log",
        ylim=(1e-5, 0.1),
        xlabel="number of negatives n (log scale)",
        ylabel="false-positive rate (log scale)",
        title=rf"Intervals when exactly {p:g}$\cdot n$ false positives are seen",
    )
    ax.legend(loc="upper right")
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
        "hypothesis_testing_errors.svg": plot_errors(),
        "hypothesis_testing_discrete_tradeoff.svg": plot_discrete_tradeoff(rng),
        "hypothesis_testing_gaussian_tradeoff.svg": plot_gaussian_tradeoff(rng),
        "hypothesis_testing_laplace_tradeoff.svg": plot_laplace_tradeoff(),
        "hypothesis_testing_tv_advantage.svg": plot_tv_advantage(),
        "hypothesis_testing_auc_low_fpr.svg": plot_auc_low_fpr(),
        "hypothesis_testing_posterior.svg": plot_posterior(),
        "hypothesis_testing_rate_ci.svg": plot_rate_ci(),
    }
    for name, fig in figures.items():
        fig.tight_layout()
        fig.savefig(out / name, metadata={"Date": None})  # no timestamp, so output is reproducible
        plt.close(fig)
        print(f"saved {out / name}")


if __name__ == "__main__":
    main()
