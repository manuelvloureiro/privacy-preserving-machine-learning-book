"""Check the core appendix on what ε means and on converting between definitions, and draw its figures.

    uv run python code/appendix-derivations/apx_core_conversions.py
    uv run python code/appendix-derivations/apx_core_conversions.py --seed 1 --out /tmp/figures

Prints every number quoted in book-full/appendix-derivations/04-conversions.md (seed 0),
then saves the page's figures.
"""

import argparse
import itertools
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
matplotlib.rcParams["svg.hashsalt"] = "apx_core_conversions"  # stable element ids, so reruns don't churn the SVGs
matplotlib.rcParams["svg.fonttype"] = "none"  # keep text as text: smaller, diffable files

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

# Figures are build output: _build/figures/<part>/, which the build copies into each book.
FIGURES = Path(__file__).resolve().parents[2] / "_build" / "figures" / Path(__file__).parent.name

DELTA = 1e-5  # the δ used throughout the page


# --- The standard normal ----------------------------------------------------------------------


def norm_cdf(x: float) -> float:
    """Φ(x) = Pr[Z <= x]; erfc keeps full relative precision in the left tail."""
    return 0.5 * math.erfc(-x / math.sqrt(2))


def log_norm_cdf(x: float) -> float:
    """ln Φ(x), accurate far into the left tail (asymptotic series below x = -30)."""
    if x > -30:
        return math.log(norm_cdf(x))
    return -x * x / 2 - math.log(-x) - 0.5 * math.log(2 * math.pi) + math.log1p(-1 / (x * x))


def norm_ppf(p: float) -> float:
    """Φ⁻¹(p) by bisection on Φ, which is increasing: 200 halvings of [-40, 40]."""
    lo, hi = -40.0, 40.0
    for _ in range(200):
        mid = (lo + hi) / 2
        if norm_cdf(mid) < p:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


# --- Section 1: ε as a limit on every test ---------------------------------------------------


def min_fnr(fpr: float, eps: float, delta: float = 0.0) -> float:
    """Smallest false negative rate the two inequalities allow at this false positive rate."""
    return max(0.0, 1 - delta - math.exp(eps) * fpr, math.exp(-eps) * (1 - delta - fpr))


def rr_corner(eps: float) -> float:
    """Randomized response's test 'say D' on report 0' has FPR = FNR = 1/(1 + e^ε)."""
    return 1 / (1 + math.exp(eps))


def rr_vector_probs(bits: tuple[int, ...], eps: float) -> dict[tuple[int, ...], float]:
    """Probability of each report vector when every person runs randomized response at ε."""
    keep = math.exp(eps) / (1 + math.exp(eps))
    probs: dict[tuple[int, ...], float] = {}
    for reports in itertools.product((0, 1), repeat=len(bits)):
        probs[reports] = math.prod(keep if r == b else 1 - keep for r, b in zip(reports, bits, strict=True))
    return probs


def worst_test_slack(eps: float) -> tuple[int, float, int]:
    """Enumerate every test on three people's reports.

    Returns how many tests there are, the smallest slack, and how many tests have slack 0.

    The slack of a test is the smallest of FPR + e^ε FNR - 1 and e^ε FPR + FNR - 1, which the
    testing inequalities say is at least 0 (δ = 0).
    """
    p = rr_vector_probs((1, 0, 1), eps)  # D
    q = rr_vector_probs((0, 0, 1), eps)  # D', the first bit changed
    outputs = list(p)
    worst = math.inf
    count = 0
    tight = 0
    for mask in range(2 ** len(outputs)):
        region = [y for i, y in enumerate(outputs) if mask >> i & 1]  # the test says D' here
        fpr = sum(p[y] for y in region)
        fnr = 1 - sum(q[y] for y in region)
        slack = min(fpr + math.exp(eps) * fnr - 1, math.exp(eps) * fpr + fnr - 1)
        worst = min(worst, slack)
        count += 1
        tight += abs(slack) < 1e-12
    return count, worst, tight


# --- Section 2: ε as a limit on belief ---------------------------------------------------------


def max_posterior(prior: float, eps: float) -> float:
    """Largest posterior probability of D after one output of an ε-DP mechanism."""
    return math.exp(eps) * prior / (math.exp(eps) * prior + 1 - prior)


def min_posterior(prior: float, eps: float) -> float:
    return max_posterior(prior, -eps)


def simulated_rr_posterior(rng: np.random.Generator, prior: float, eps: float, n: int) -> float:
    """Draw the secret bit with Pr[bit = 1] = prior, run randomized response, and return the
    fraction of '1' reports that came from a true 1."""
    keep = math.exp(eps) / (1 + math.exp(eps))
    bit = rng.random(n) < prior
    report = np.where(rng.random(n) < keep, bit, ~bit)
    return float(bit[report].mean())


def rr_bayes_table(prior: float, eps: float) -> tuple[float, float, float, float]:
    """Randomized response, report '1', secret bit 1 under D and 0 under D'.

    Returns the two joint weights prior x likelihood, their sum, and the posterior of D.
    """
    keep = math.exp(eps) / (1 + math.exp(eps))
    joint_d = prior * keep
    joint_d2 = (1 - prior) * (1 - keep)
    total = joint_d + joint_d2
    return joint_d, joint_d2, total, joint_d / total


# --- Section 3: a small loss exceeds ε only on a rare set -----------------------------------


def three_output_check(eps: float) -> tuple[float, float, float]:
    """Three outputs with p = (0.6, 0.39, 0.01) and q = (0.6, 0.3999, 0.0001).

    Returns the P-probability of the outputs with loss above ε, the largest P(S) - e^ε Q(S) over
    all 8 events S, and the largest privacy loss.
    """
    p = (0.6, 0.39, 0.01)
    q = (0.6, 0.3999, 0.0001)
    losses = [math.log(a / b) for a, b in zip(p, q, strict=True)]
    bad = sum(a for a, loss in zip(p, losses, strict=True) if loss > eps)
    worst = max(
        sum(p[i] for i in range(3) if mask >> i & 1) - math.exp(eps) * sum(q[i] for i in range(3) if mask >> i & 1)
        for mask in range(8)
    )
    return bad, worst, max(losses)


# --- The Gaussian pair N(0, 1) against N(μ, 1) ------------------------------------------------


def gauss_delta(eps: float, mu: float) -> float:
    """Exact δ(ε) of the pair: Φ(μ/2 - ε/μ) - e^ε Φ(-μ/2 - ε/μ), second term in logs."""
    first = norm_cdf(mu / 2 - eps / mu)
    second = math.exp(eps + log_norm_cdf(-mu / 2 - eps / mu))
    return first - second


def gauss_eps(delta: float, mu: float) -> float:
    """Smallest ε with gauss_delta(ε, μ) <= δ, by bisection (δ(ε) falls as ε grows)."""
    lo, hi = 0.0, 1.0
    while gauss_delta(hi, mu) > delta:
        hi *= 2
    for _ in range(200):
        mid = (lo + hi) / 2
        if gauss_delta(mid, mu) > delta:
            lo = mid
        else:
            hi = mid
    return hi


def loss_tail(eps: float, mu: float) -> float:
    """Pr_P[L > ε]: the privacy loss μ²/2 - μy is N(μ²/2, μ²) under P."""
    return norm_cdf(mu / 2 - eps / mu)


def tradeoff_g(a: float, mu: float) -> float:
    """G_μ(a) = Φ(Φ⁻¹(1 - a) - μ), the smallest FNR at FPR a."""
    if a <= 0:
        return 1.0
    if a >= 1:
        return 0.0
    return norm_cdf(norm_ppf(1 - a) - mu)


def touching_point(eps: float, mu: float) -> tuple[float, float, float]:
    """Threshold t = ε/μ + μ/2 where G_μ has slope -e^ε, and the FPR and power there."""
    t = eps / mu + mu / 2
    return t, norm_cdf(-t), norm_cdf(mu - t)


def delta_by_search(eps: float, mu: float, n: int = 20_001) -> float:
    """max over a of 1 - G_μ(a) - e^ε a, on a grid of thresholds (each threshold is one test)."""
    ts = np.linspace(-10.0, 15.0, n)
    best = -math.inf
    for t in ts:
        a = norm_cdf(-float(t))
        power = norm_cdf(mu - float(t))
        best = max(best, power - math.exp(eps) * a)
    return best


def simulate_threshold_test(rng: np.random.Generator, mu: float, a: float, n: int) -> tuple[float, float]:
    """Run the size-a threshold test on n draws from N(0, 1) and n from N(μ, 1)."""
    t = norm_ppf(1 - a)
    fpr = float((rng.standard_normal(n) > t).mean())
    fnr = float((rng.standard_normal(n) + mu <= t).mean())
    return fpr, fnr


# --- Section 3: RDP to (ε, δ) -----------------------------------------------------------------


def gauss_rdp(alpha: float, mu: float) -> float:
    """ε(α) = αμ²/2, the Rényi divergence of the pair."""
    return alpha * mu * mu / 2


def mironov(alpha: float, eps_alpha: float, delta: float) -> float:
    """ε = ε(α) + ln(1/δ)/(α - 1)."""
    return eps_alpha + math.log(1 / delta) / (alpha - 1)


def improved(alpha: float, eps_alpha: float, delta: float) -> float:
    """ε = ε(α) + ln((α - 1)/α) - (ln δ + ln α)/(α - 1)."""
    return eps_alpha + math.log((alpha - 1) / alpha) - (math.log(delta) + math.log(alpha)) / (alpha - 1)


def best_order_closed_form(mu: float, delta: float) -> tuple[float, float]:
    """Minimizer of αμ²/2 + ln(1/δ)/(α - 1): α* = 1 + √(2 ln(1/δ))/μ, and the value there."""
    alpha = 1 + math.sqrt(2 * math.log(1 / delta)) / mu
    return alpha, mironov(alpha, gauss_rdp(alpha, mu), delta)


def fine_orders() -> list[float]:
    """20,001 log-spaced orders from 1 + 10^-4 to 1 + 10^4."""
    return [1 + 10 ** (-4 + 8 * i / 20_000) for i in range(20_001)]


def best_improved(mu: float, delta: float) -> tuple[float, float]:
    """Smallest improved ε over the fine grid of orders, and the order that gives it."""
    return min((improved(a, gauss_rdp(a, mu), delta), a) for a in fine_orders())


def mironov_fine(mu: float, delta: float) -> float:
    return min(mironov(a, gauss_rdp(a, mu), delta) for a in fine_orders())


def mgf_check(rng: np.random.Generator, alpha: float, mu: float, n: int) -> float:
    """Monte Carlo ln E_P[e^{(α-1)L}] / (α - 1), which should be αμ²/2."""
    y = rng.standard_normal(n)
    loss = mu * mu / 2 - mu * y
    return float(np.log(np.mean(np.exp((alpha - 1) * loss)))) / (alpha - 1)


# --- Section 4: zCDP --------------------------------------------------------------------------


def zcdp_eps(rho: float, delta: float) -> float:
    """ε = ρ + 2√(ρ ln(1/δ))."""
    return rho + 2 * math.sqrt(rho * math.log(1 / delta))


def zcdp_rho_for(eps: float, delta: float) -> float:
    """Largest ρ whose simple conversion stays within ε: (√(ε + ln(1/δ)) - √(ln(1/δ)))²."""
    ell = math.log(1 / delta)
    return (math.sqrt(eps + ell) - math.sqrt(ell)) ** 2


# --- The page's table -------------------------------------------------------------------------


def print_table(rng: np.random.Generator) -> None:
    # Section 1
    eps, fpr = 1.0, 0.05
    print(
        f"[testing eps=1 FPR=0.05] line1 FNR>={math.exp(-eps) * (1 - fpr):.4f}  "
        f"line2 FNR>={1 - math.exp(eps) * fpr:.4f}  bound={min_fnr(fpr, eps):.4f}  "
        f"with delta=1e-5: {min_fnr(fpr, eps, DELTA):.6f}"
    )
    print(f"[testing eps=1] FPR=0.01: FNR>={min_fnr(0.01, eps):.4f}  FPR=0.25: FNR>={min_fnr(0.25, eps):.4f}")
    print(
        "[RR corner] "
        + "  ".join(f"eps={e:g}: {rr_corner(e):.4f}" for e in (0.5, 1.0, 2.0))
        + f"  check FPR+e*FNR at eps=1: {rr_corner(1) * (1 + math.e):.4f}"
    )
    count, slack, tight = worst_test_slack(math.log(3))
    print(
        f"[three people, eps=ln 3] tests={count}  smallest slack={abs(slack) if abs(slack) < 1e-12 else slack:.6f}  tight={tight}"
    )

    # Section 2
    prior = 0.01
    odds = prior / (1 - prior)
    post_odds = odds * math.e
    print(
        f"[posterior prior=0.01 eps=1] prior odds={odds:.6f}  posterior odds={post_odds:.6f}  "
        f"posterior={post_odds / (1 + post_odds):.4f}  min={min_posterior(prior, 1):.4f}"
    )
    print(f"[posterior prior=0.5 eps=1] max={max_posterior(0.5, 1):.4f}  min={min_posterior(0.5, 1):.4f}")
    for e in (0.5, 2.0, 4.0):
        print(
            f"[posterior eps={e:g}] prior 0.01 -> {max_posterior(0.01, e):.4f}  prior 0.5 -> {max_posterior(0.5, e):.4f}"
        )
    jd, jd2, tot, post = rr_bayes_table(prior, 1.0)
    print(f"[Bayes table RR eps=1 prior=0.01] D: {jd:.6f}  D': {jd2:.6f}  sum: {tot:.6f}  posterior: {post:.4f}")
    sim = simulated_rr_posterior(rng, prior, 1.0, 2_000_000)
    print(f"[RR posterior simulation, 2e6 draws] {sim:.4f}  exact {max_posterior(prior, 1):.4f}")
    eps_exact = gauss_eps(DELTA, 1.0)
    tail = loss_tail(eps_exact, 1.0)
    print(f"[Gaussian mu=1 at eps={eps_exact:.4f}] Pr[L > eps]={tail:.3e}  ratio to delta={tail / DELTA:.2f}")
    bad = DELTA / (1 - math.exp(-1))
    print(f"[bad set eps'=eps+1] bound={bad:.3e}  exact Pr[L > eps+1]={loss_tail(eps_exact + 1, 1.0):.3e}")

    # Section 3
    mu = 1.0
    alpha_star, eps_m = best_order_closed_form(mu, DELTA)
    print(f"[RDP mu=1] ln(1/delta)={math.log(1 / DELTA):.4f}  alpha*={alpha_star:.4f}  Mironov eps={eps_m:.4f}")
    print(f"[RDP mu=1] Mironov on fine grid={mironov_fine(mu, DELTA):.4f}")
    eps_i, alpha_i = best_improved(mu, DELTA)
    print(
        f"[RDP mu=1] improved eps={eps_i:.4f} at alpha={alpha_i:.3f}  exact eps={eps_exact:.4f}  "
        f"saving={eps_m - eps_i:.4f}  Mironov excess={eps_m - eps_exact:.4f}  improved excess={eps_i - eps_exact:.4f}"
    )
    print(
        f"[RDP mu=1] at Mironov eps: Pr[L > eps]={loss_tail(eps_m, mu):.3e}  exact delta={gauss_delta(eps_m, mu):.3e}  "
        f"at improved eps: exact delta={gauss_delta(eps_i, mu):.3e}  paid/got={DELTA / gauss_delta(eps_m, mu):.1f}"
    )
    for a in (2.0, 5.8, 32.0):
        print(
            f"[RDP mu=1 alpha={a:g}] eps(alpha)={gauss_rdp(a, mu):.4f}  Mironov={mironov(a, gauss_rdp(a, mu), DELTA):.4f}"
        )
    bad, worst, top = three_output_check(0.1)
    print(
        f"[three outputs eps=0.1] P[L > eps]={bad:.4f}  largest P(S) - e^eps Q(S)={worst:.5f}  largest loss={top:.3f}"
    )
    mc = mgf_check(rng, 3.0, mu, 2_000_000)
    print(f"[RDP MGF check alpha=3 mu=1, 2e6 draws] {mc:.4f}  formula {gauss_rdp(3.0, mu):.4f}")

    # Section 4
    for rho in (0.125, 0.2, 0.5, 5.0):
        mu_r = math.sqrt(2 * rho)
        simple, better, exact_r = zcdp_eps(rho, DELTA), best_improved(mu_r, DELTA)[0], gauss_eps(DELTA, mu_r)
        print(
            f"[zCDP rho={rho:g}] mu={mu_r:.4f}  alpha*={1 + math.sqrt(math.log(1 / DELTA) / rho):.4f}  "
            f"simple={simple:.4f}  improved={better:.4f}  exact={exact_r:.4f}  "
            f"simple/exact={simple / exact_r:.3f}  gap recovered={(simple - better) / (simple - exact_r):.2f}"
        )
    rho_mix = 10 / (2 * 25) + 0.5**2 / 2
    print(f"[zCDP mix] rho={rho_mix:.4f}  simple eps={zcdp_eps(rho_mix, DELTA):.4f}")
    rho_t = zcdp_rho_for(1.0, DELTA)
    print(
        f"[zCDP calibrate eps=1] rho={rho_t:.5f}  sigma={1 / math.sqrt(2 * rho_t):.3f}  back={zcdp_eps(rho_t, DELTA):.4f}"
    )

    # Section 5
    print(
        "[GDP G_1] "
        + "  ".join(f"a={a:g}: {tradeoff_g(a, 1.0):.4f}" for a in (0.01, 0.05, 0.1))
        + f"  threshold at 0.05: {norm_ppf(0.95):.4f}"
    )
    t, a_star, power = touching_point(1.0, 1.0)
    print(
        f"[GDP mu=1 eps=1] t={t:.4f}  a*={a_star:.6f}  power={power:.6f}  "
        f"delta={power - math.e * a_star:.6f}  formula={gauss_delta(1.0, 1.0):.6f}  search={delta_by_search(1.0, 1.0):.6f}"
    )
    print("[GDP delta(1)] " + "  ".join(f"mu={m:g}: {gauss_delta(1.0, m):.4f}" for m in (0.5, 1.0, 2.0)))
    print("[GDP eps at 1e-5] " + "  ".join(f"mu={m:g}: {gauss_eps(DELTA, m):.4f}" for m in (0.5, 1.0, 2.0)))
    print(f"[GDP compose] 4 x 0.5 -> {math.sqrt(4 * 0.25):.4f}  100 x 0.1 -> {math.sqrt(100 * 0.01):.4f}")
    fpr_s, fnr_s = simulate_threshold_test(rng, 1.0, 0.05, 1_000_000)
    print(
        f"[GDP simulation mu=1 a=0.05, 1e6 draws] FPR={fpr_s:.4f}  FNR={fnr_s:.4f}  exact {tradeoff_g(0.05, 1.0):.4f}"
    )
    print(f"[ppf check] Phi(ppf(0.999))={norm_cdf(norm_ppf(0.999)):.6f}  ppf(0.975)={norm_ppf(0.975):.4f}")


# --- Figures ----------------------------------------------------------------------------------


def plot_testing_region() -> Figure:
    fig, ax = plt.subplots(figsize=(6, 5))
    a = np.linspace(0.0, 1.0, 801)
    for eps, color in ((0.5, "C0"), (1.0, "C1"), (2.0, "C2")):
        f = np.array([min_fnr(float(x), eps) for x in a])
        ax.fill_between(a, 0, f, color=color, alpha=0.12)
        ax.plot(a, f, color=color, label=rf"$\varepsilon = {eps:g}$: smallest FNR any test can reach")
        c = rr_corner(eps)
        ax.plot([c], [c], "o", color=color, markersize=7)
    ax.plot([], [], "ko", label="randomized response at the same $\\varepsilon$")
    ax.plot([0, 1], [1, 0], "k:", linewidth=1, label="guessing (ignore the output)")
    ax.plot([0.05], [min_fnr(0.05, 1.0)], "kx", markersize=9, mew=2)
    ax.annotate(
        "FPR 5%, ε = 1:\nFNR at least 86%",
        xy=(0.05, min_fnr(0.05, 1.0)),
        xytext=(0.42, 0.6),
        arrowprops={"arrowstyle": "->"},
    )
    ax.text(0.04, 0.04, "shaded: no test\ncan be here", fontsize=9)
    ax.set(
        xlim=(0, 1),
        ylim=(0, 1),
        aspect="equal",
        xlabel="false positive rate (says D' when the data were D)",
        ylabel="false negative rate (says D when the data were D')",
        title="ε-DP forbids every test below the lines",
    )
    ax.legend(loc="upper right", fontsize=8)
    return fig


def plot_posterior() -> Figure:
    fig, (left, right) = plt.subplots(1, 2, figsize=(11, 4.6))
    eps_list = (0.5, 1.0, 2.0, 4.0)
    lo = np.linspace(-6, 6, 241)
    for eps, color in zip(eps_list, ("C0", "C1", "C2", "C3"), strict=True):
        left.fill_between(lo, lo - eps, lo + eps, color=color, alpha=0.08)
        left.plot(lo, lo + eps, color=color, label=rf"$\varepsilon = {eps:g}$")
        left.plot(lo, lo - eps, color=color, linestyle="--")
    left.plot(lo, lo, "k:", linewidth=1)
    left.set(
        xlim=(-6, 6),
        ylim=(-6, 6),
        aspect="equal",
        xlabel="log prior odds of D",
        ylabel="log posterior odds of D",
        title="In log odds: a band of half-width ε",
    )
    left.legend(loc="upper left", fontsize=8)
    priors = np.linspace(0.0005, 0.9995, 800)
    for eps, color in zip(eps_list, ("C0", "C1", "C2", "C3"), strict=True):
        right.plot(
            priors, [max_posterior(float(p), eps) for p in priors], color=color, label=rf"$\varepsilon = {eps:g}$"
        )
        right.plot(priors, [min_posterior(float(p), eps) for p in priors], color=color, linestyle="--")
    right.plot([0, 1], [0, 1], "k:", linewidth=1, label="no change")
    right.plot([0.01, 0.5], [max_posterior(0.01, 1), max_posterior(0.5, 1)], "ko")
    right.annotate(
        "prior 1% → at most 2.7%",
        xy=(0.01, max_posterior(0.01, 1)),
        xytext=(0.18, 0.06),
        arrowprops={"arrowstyle": "->"},
    )
    right.annotate(
        "prior 50% → at most 73%", xy=(0.5, max_posterior(0.5, 1)), xytext=(0.55, 0.45), arrowprops={"arrowstyle": "->"}
    )
    right.set(
        xlim=(0, 1),
        ylim=(0, 1),
        aspect="equal",
        xlabel="prior probability of D",
        ylabel="posterior probability of D",
        title="In probabilities: the band bends",
    )
    right.legend(loc="lower right", fontsize=8)
    return fig


def plot_rdp_order(mu: float = 1.0) -> Figure:
    alphas = np.linspace(1.3, 20.0, 600)
    ell = math.log(1 / DELTA)
    term1 = [gauss_rdp(float(a), mu) for a in alphas]
    term2 = [ell / (float(a) - 1) for a in alphas]
    mir = [mironov(float(a), gauss_rdp(float(a), mu), DELTA) for a in alphas]
    imp = [improved(float(a), gauss_rdp(float(a), mu), DELTA) for a in alphas]
    alpha_star, eps_m = best_order_closed_form(mu, DELTA)
    eps_i, alpha_i = best_improved(mu, DELTA)
    exact = gauss_eps(DELTA, mu)
    fig, ax = plt.subplots(figsize=(7, 4.6))
    ax.plot(alphas, term1, color="C0", linestyle="--", label=r"$\varepsilon(\alpha) = \alpha\mu^2/2$ (grows)")
    ax.plot(alphas, term2, color="C2", linestyle="--", label=r"$\ln(1/\delta)/(\alpha - 1)$ (shrinks)")
    ax.plot(alphas, mir, color="C3", linewidth=2, label="their sum: Markov conversion")
    ax.plot(alphas, imp, color="C4", linewidth=2, label="improved conversion")
    ax.axhline(exact, color="k", linestyle=":", label=f"exact ε of the Gaussian = {exact:.3f}")
    ax.plot([alpha_star], [eps_m], "o", color="C3")
    ax.annotate(
        rf"$\alpha^* = {alpha_star:.2f}$, $\varepsilon = {eps_m:.3f}$",
        xy=(alpha_star, eps_m),
        xytext=(8, 8.5),
        arrowprops={"arrowstyle": "->"},
    )
    ax.plot([alpha_i], [eps_i], "o", color="C4")
    ax.annotate(rf"$\varepsilon = {eps_i:.3f}$", xy=(alpha_i, eps_i), xytext=(12, 2.4), arrowprops={"arrowstyle": "->"})
    ax.set(
        xlim=(1, 20),
        ylim=(0, 14),
        xlabel=r"Rényi order $\alpha$",
        ylabel=r"$\varepsilon$ at $\delta = 10^{-5}$",
        title=r"Gaussian with $\mu = \Delta/\sigma = 1$: every order gives a valid $\varepsilon$",
    )
    ax.legend(loc="upper right", fontsize=8)
    return fig


def plot_gdp(mus: tuple[float, ...] = (0.5, 1.0, 2.0)) -> Figure:
    fig, (left, right) = plt.subplots(1, 2, figsize=(11, 4.8))
    a = np.linspace(0.0, 1.0, 801)
    eps = 1.0
    for mu, color in zip(mus, ("C0", "C1", "C2"), strict=True):
        d = gauss_delta(eps, mu)
        left.plot(
            a, [tradeoff_g(float(x), mu) for x in a], color=color, label=rf"$G_{{{mu:g}}}$, $\delta(1) = {d:.3f}$"
        )
        left.plot(a, np.maximum(0, 1 - d - math.e * a), color=color, linestyle="--", linewidth=1)
        _, a_t, power = touching_point(eps, mu)
        left.plot([a_t], [1 - power], "o", color=color)
    left.plot([], [], "k--", linewidth=1, label=r"line of slope $-e^{\varepsilon}$, $\varepsilon = 1$, that touches it")
    left.plot([0, 1], [1, 0], "k:", linewidth=1)
    left.set(
        xlim=(0, 1),
        ylim=(0, 1.02),
        aspect="equal",
        xlabel="false positive rate $a$",
        ylabel="smallest false negative rate",
        title=r"Each $(\varepsilon, \delta)$ is one line under $G_\mu$",
    )
    left.legend(loc="upper right", fontsize=8)
    es = np.linspace(0.0, 12.0, 481)
    for mu, color in zip(mus, ("C0", "C1", "C2"), strict=True):
        right.plot(es, [max(gauss_delta(float(e), mu), 1e-300) for e in es], color=color, label=rf"$\mu = {mu:g}$")
        e5 = gauss_eps(DELTA, mu)
        right.plot([e5], [DELTA], "o", color=color)
        right.annotate(f"{e5:.2f}", xy=(e5, DELTA), xytext=(e5 + 0.2, DELTA * 4), color=color, fontsize=8)
    right.axhline(DELTA, color="k", linestyle=":", linewidth=1, label=r"$\delta = 10^{-5}$")
    right.set(
        yscale="log",
        ylim=(1e-10, 1),
        xlim=(0, 12),
        xlabel=r"$\varepsilon$",
        ylabel=r"exact $\delta(\varepsilon)$",
        title=r"The whole $(\varepsilon, \delta)$ curve of $\mu$-GDP",
    )
    right.legend(loc="upper right", fontsize=8)
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
    figures = {
        "conv_testing_region.svg": plot_testing_region(),
        "conv_posterior.svg": plot_posterior(),
        "conv_rdp_order.svg": plot_rdp_order(),
        "conv_gdp.svg": plot_gdp(),
    }
    for name, fig in figures.items():
        fig.tight_layout()
        fig.savefig(out / name, metadata={"Date": None})  # no timestamp, so output is reproducible
        plt.close(fig)
        print(f"saved {out / name}")


if __name__ == "__main__":
    main()
