"""Check the numbers in the definition-of-differential-privacy chapter and draw its figures.

    uv run python code/part03-dp-foundations/definition.py
    uv run python code/part03-dp-foundations/definition.py --seed 1 --out /tmp/figures

Prints the numbers quoted in 01-definition.md (they match exactly with the default seed),
then saves the chapter's figures.

Conventions: M(D) and M(D') are the output distributions on two neighbouring datasets, with
densities or probability mass functions p and q. The privacy loss is L(y) = ln p(y)/q(y).
A mechanism is (ε, δ)-DP when P(S) ≤ e^ε Q(S) + δ for every event S and both orders of every
neighbouring pair. The three running mechanisms:

- randomized response: report a bit truthfully with probability e^ε0/(1 + e^ε0), else flip it;
- a count plus Laplace noise of scale b: neighbouring outputs are Lap(c, b) and Lap(c + 1, b);
- a count plus Gaussian noise of standard deviation σ: N(c, σ²) and N(c + 1, σ²), with μ = 1/σ.
"""

import argparse
import itertools
import math
from collections.abc import Callable, Sequence
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
matplotlib.rcParams["svg.hashsalt"] = "definition"  # stable element ids, so reruns don't churn the SVGs
matplotlib.rcParams["svg.fonttype"] = "none"  # keep text as text: smaller, diffable files

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from numpy.typing import NDArray

# Figures are build output: _build/figures/<part>/, which the build copies into each book.
FIGURES = Path(__file__).resolve().parents[2] / "_build" / "figures" / Path(__file__).parent.name

Dist = Sequence[float]  # a distribution on a finite set, as a list of probabilities

LN3 = math.log(3.0)


def normal_cdf(x: float) -> float:
    """Φ(x); erfc keeps full relative precision in the lower tail."""
    return 0.5 * math.erfc(-x / math.sqrt(2))


def normal_ppf(u: float) -> float:
    """Φ^{-1}(u) by bisection on Φ."""
    return bisect(lambda x: normal_cdf(x) - u, -40.0, 40.0)


def bisect(f: Callable[[float], float], lo: float, hi: float, steps: int = 200) -> float:
    """A root of an increasing f on [lo, hi]."""
    for _ in range(steps):
        mid = 0.5 * (lo + hi)
        if f(mid) < 0:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


# --- Toolkit: events, ratios and the hockey-stick divergence on finite outcome sets -------------


def hockey_stick(p: Dist, q: Dist, gamma: float) -> float:
    """E_γ(P‖Q) = Σ (p - γq)_+ = max over events S of P(S) - γQ(S)."""
    return sum(max(pi - gamma * qi, 0.0) for pi, qi in zip(p, q, strict=True))


def max_over_events(p: Dist, q: Dist, gamma: float) -> float:
    """max_S [P(S) - γQ(S)] by brute force over all 2^k events (small k only)."""
    best = 0.0
    k = len(p)
    for mask in range(1 << k):
        gap = sum(p[i] - gamma * q[i] for i in range(k) if mask >> i & 1)
        best = max(best, gap)
    return best


def max_event_ratio(p: Dist, q: Dist) -> float:
    """max over events S with Q(S) > 0 of P(S)/Q(S), by brute force over all events."""
    best = 0.0
    k = len(p)
    for mask in range(1, 1 << k):
        ps = sum(p[i] for i in range(k) if mask >> i & 1)
        qs = sum(q[i] for i in range(k) if mask >> i & 1)
        if qs > 0:
            best = max(best, ps / qs)
    return best


def max_divergence(p: Dist, q: Dist) -> float:
    """D_∞(P‖Q) = ln max p/q over the support of P."""
    return math.log(max(pi / qi if qi > 0 else math.inf for pi, qi in zip(p, q, strict=True) if pi > 0))


def smallest_epsilon(p: Dist, q: Dist) -> float:
    """The smallest ε with (P, Q) ε-indistinguishable in both directions."""
    return max(max_divergence(p, q), max_divergence(q, p))


def smallest_delta(p: Dist, q: Dist, eps: float) -> float:
    """The smallest δ with P(S) ≤ e^ε Q(S) + δ and Q(S) ≤ e^ε P(S) + δ for every S."""
    g = math.exp(eps)
    return max(hockey_stick(p, q, g), hockey_stick(q, p, g))


def pointwise_delta(p: Dist, q: Dist, eps: float) -> float:
    """The smallest δ with p(y) ≤ e^ε q(y) + δ for every single outcome y (a weaker, wrong definition)."""
    g = math.exp(eps)
    return max(max(pi - g * qi, 0.0) for pi, qi in zip(p, q, strict=True))


def product(dists: Sequence[Dist]) -> list[float]:
    """The joint distribution of independent coordinates, outcomes in lexicographic order."""
    return [math.prod(c) for c in itertools.product(*dists)]


# --- Neighbouring datasets ------------------------------------------------------------------------


def laplace_log_ratio(y: NDArray[np.float64], shift: float, b: float) -> NDArray[np.float64]:
    """ln Lap(y; 0, b)/Lap(y; shift, b) = (|y - shift| - |y|)/b."""
    return (np.abs(y - shift) - np.abs(y)) / b


def histogram_max_loss(shifts: Sequence[float], b: float) -> float:
    """Largest privacy loss of a histogram released with independent Lap(b) noise per bin.

    The neighbouring histograms differ by `shifts` (one entry per bin). The log ratio of the joint
    density is the sum of the per-bin log ratios, so its maximum is the sum of the per-bin maxima,
    found here on a grid.
    """
    grid = np.linspace(-20.0, 20.0, 40_001)
    return float(sum(np.max(laplace_log_ratio(grid, s, b)) for s in shifts))


# --- Randomized response --------------------------------------------------------------------------


def rr_truth_prob(eps: float) -> float:
    """Probability that randomized response at level ε reports the true bit: e^ε/(1 + e^ε)."""
    return math.exp(eps) / (1.0 + math.exp(eps))


def rr_output(bit: int, eps: float) -> list[float]:
    """[Pr(report 0), Pr(report 1)] for a true bit."""
    t = rr_truth_prob(eps)
    return [t, 1.0 - t] if bit == 0 else [1.0 - t, t]


def coin_rr_output(bit: int) -> list[float]:
    """Warner's two-coin protocol: heads, tell the truth; tails, toss again and say yes on heads."""
    p_yes = 0.5 * bit + 0.5 * 0.5
    return [1.0 - p_yes, p_yes]


def rr_dataset_output(bits: Sequence[int], eps: float) -> list[float]:
    """Distribution of the vector of reports when every person runs randomized response independently."""
    return product([rr_output(b, eps) for b in bits])


def rr_estimate(reports: NDArray[np.int64], eps: float) -> float:
    """Unbiased estimate of the fraction of ones from randomized-response reports."""
    t = rr_truth_prob(eps)
    return (float(np.mean(reports)) - (1.0 - t)) / (2.0 * t - 1.0)


def rr_estimator_sd(n: int, eps: float) -> float:
    """Standard deviation of the randomized-response estimate on a fixed dataset: sqrt(t(1 - t)/n)/(2t - 1).

    Every report is a Bernoulli variable with variance t(1 - t), whatever the true bit, so the noise
    does not depend on the data. It equals e^{ε/2}/((e^ε - 1)√n).
    """
    t = rr_truth_prob(eps)
    return math.sqrt(t * (1.0 - t) / n) / (2.0 * t - 1.0)


def simulate_rr_estimates(rng: np.random.Generator, pi: float, n: int, eps: float, reps: int) -> NDArray[np.float64]:
    """`reps` independent surveys of n people, a fraction pi of whom hold a one; one estimate per survey."""
    ones = round(pi * n)
    truth = np.zeros(n, dtype=np.int64)
    truth[:ones] = 1
    t = rr_truth_prob(eps)
    keep = rng.random((reps, n)) < t
    reports = np.where(keep, truth, 1 - truth)
    return np.array([rr_estimate(row, eps) for row in reports])


# --- Divergence forms: δ(ε) for the three running mechanisms -------------------------------------


def rr_delta(eps: float, eps0: float) -> float:
    """Smallest δ for randomized response at level ε0 to be (ε, δ)-DP: (t - e^ε(1 - t))_+."""
    return smallest_delta(rr_output(0, eps0), rr_output(1, eps0), eps)


def laplace_delta(eps: float, b: float) -> float:
    """Smallest δ for a count plus Lap(b) noise to be (ε, δ)-DP: 1 - e^{-(1/b - ε)/2} below ε = 1/b."""
    t = 1.0 / b
    return 0.0 if eps >= t else 1.0 - math.exp(-(t - eps) / 2.0)


def gaussian_delta(eps: float, mu: float) -> float:
    """Smallest δ for a count plus N(0, 1/μ²) noise: Φ(μ/2 - ε/μ) - e^ε Φ(-μ/2 - ε/μ)."""
    return normal_cdf(mu / 2 - eps / mu) - math.exp(eps) * normal_cdf(-mu / 2 - eps / mu)


def gaussian_loss_tail(c: float, mu: float) -> float:
    """Pr_{y~M(D)}[L(y) > c] for the Gaussian count; L ~ N(μ²/2, μ²) under M(D)."""
    return normal_cdf(mu / 2 - c / mu)


def mc_gaussian_delta(rng: np.random.Generator, eps: float, mu: float, m: int) -> float:
    """Monte Carlo δ(ε) = E_{y~M(D)}[(1 - e^{ε - L(y)})_+] for the Gaussian count."""
    loss = rng.normal(mu * mu / 2, mu, size=m)
    return float(np.mean(np.maximum(1.0 - np.exp(eps - loss), 0.0)))


def privacy_loss_bad_set_bound(eps: float, delta: float, eps_prime: float) -> float:
    """Pr_{M(D)}[L > ε'] ≤ δ/(1 - e^{ε - ε'}) for any (ε, δ)-DP mechanism and ε' > ε."""
    return delta / (1.0 - math.exp(eps - eps_prime))


# --- Hypothesis testing ---------------------------------------------------------------------------


def dp_tradeoff(alpha: float, eps: float, delta: float) -> float:
    """f_{ε,δ}(α) = max(0, 1 - δ - e^ε α, e^{-ε}(1 - δ - α)): the least FNR any test can have at FPR α."""
    return max(0.0, 1.0 - delta - math.exp(eps) * alpha, math.exp(-eps) * (1.0 - delta - alpha))


def gaussian_tradeoff(alpha: float, mu: float) -> float:
    """T(N(0,1), N(μ,1))(α) = Φ(Φ^{-1}(1 - α) - μ)."""
    return normal_cdf(normal_ppf(1.0 - alpha) - mu)


def max_tpr(fpr: float, eps: float, delta: float) -> float:
    """The largest true-positive rate an (ε, δ)-DP mechanism allows at a given false-positive rate."""
    return 1.0 - dp_tradeoff(fpr, eps, delta)


# --- Membership advantage -------------------------------------------------------------------------


def advantage_bound(eps: float, delta: float = 0.0) -> float:
    """max TPR - FPR under (ε, δ)-DP: (e^ε - 1 + 2δ)/(e^ε + 1)."""
    return (math.exp(eps) - 1.0 + 2.0 * delta) / (math.exp(eps) + 1.0)


def yeom_bound(eps: float) -> float:
    """The looser bound e^ε - 1 on the membership advantage of an ε-DP mechanism (Yeom et al. 2018)."""
    return math.exp(eps) - 1.0


def laplace_tv(eps: float) -> float:
    """TV between Lap(0, 1/ε) and Lap(1, 1/ε): 1 - e^{-ε/2}, the best advantage against the Laplace count."""
    return 1.0 - math.exp(-eps / 2.0)


def rr_advantage(eps: float) -> float:
    """TV between the report distributions of randomized response: 2t - 1."""
    return 2.0 * rr_truth_prob(eps) - 1.0


def game_rr(rng: np.random.Generator, eps: float, trials: int) -> float:
    """Empirical advantage of the best adversary in the membership game against randomized response.

    The challenger flips b; the target's bit is b; the adversary sees the report and guesses b = report.
    Advantage = 2 Pr[correct] - 1.
    """
    b = rng.integers(0, 2, size=trials)
    keep = rng.random(trials) < rr_truth_prob(eps)
    report: NDArray[np.int64] = np.where(keep, b, 1 - b)
    correct: NDArray[np.bool_] = np.equal(report, b)
    return 2.0 * float(np.mean(correct)) - 1.0


def game_laplace(rng: np.random.Generator, eps: float, trials: int, count: int = 100) -> float:
    """Membership game against a count plus Lap(1/ε) noise: in (b = 1) adds one to the count.

    The adversary uses the likelihood-ratio test, which here is "guess in when the output exceeds count + 1/2".
    """
    b = rng.integers(0, 2, size=trials)
    y = count + b + rng.laplace(0.0, 1.0 / eps, size=trials)
    guess: NDArray[np.int64] = (y > count + 0.5).astype(np.int64)
    correct: NDArray[np.bool_] = np.equal(guess, b)
    return 2.0 * float(np.mean(correct)) - 1.0


# --- Bayesian view --------------------------------------------------------------------------------


def posterior_bound(prior: float, eps: float) -> float:
    """Largest posterior on D after any output of an ε-DP mechanism: e^ε π / (e^ε π + 1 - π)."""
    return math.exp(eps) * prior / (math.exp(eps) * prior + 1.0 - prior)


# --- Why δ must be much smaller than 1/n ----------------------------------------------------------


def release_records_output(n: int, delta: float) -> list[float]:
    """Output distribution of "publish each of n records independently with probability δ".

    Outcomes are the subsets of the n records, indexed by bit masks.
    """
    return product([[1.0 - delta, delta]] * n)


def release_records_neighbour(n: int, delta: float) -> tuple[list[float], list[float]]:
    """Output distributions on D (n records) and D' = D plus one record, on the subsets of D'.

    The last coordinate is the extra record, which D can never publish.
    """
    on_d = product([[1.0 - delta, delta]] * n + [[1.0, 0.0]])
    on_d_prime = release_records_output(n + 1, delta)
    return on_d, on_d_prime


def prob_someone_exposed(n: int, delta: float) -> float:
    """Pr[at least one of n records is published] = 1 - (1 - δ)^n."""
    return -math.expm1(n * math.log1p(-delta))


def simulate_exposed(rng: np.random.Generator, n: int, delta: float, trials: int) -> tuple[float, float]:
    """Mean number of published records and the fraction of runs that publish at least one."""
    counts = rng.binomial(n, delta, size=trials)
    return float(np.mean(counts)), float(np.mean(counts > 0))


# --- What differential privacy does not protect -------------------------------------------------


def group_max_loss(k: int, eps: float) -> float:
    """Largest privacy loss of a Lap(1/ε) count when k records enter or leave together: kε."""
    return float(np.max(laplace_log_ratio(np.linspace(-50.0, 50.0, 100_001), float(k), 1.0 / eps)))


def print_table(rng: np.random.Generator) -> None:
    print("== Neighbouring datasets")
    eps = 1.0
    print(
        f"histogram, Lap(1/eps) per bin, eps={eps:g}: add/remove max loss={histogram_max_loss([1.0], 1 / eps):.4f}  "
        f"replace max loss={histogram_max_loss([1.0, -1.0], 1 / eps):.4f}"
    )

    print("== Pure and approximate DP")
    m = 1000
    p = [1.0 / m] * m + [0.0] * m
    q = [0.0] * m + [1.0 / m] * m
    print(
        f"disjoint uniforms, m={m}: pointwise delta={pointwise_delta(p, q, 1.0):.4f}  "
        f"event delta={smallest_delta(p, q, 1.0):.4f}"
    )
    a = rr_dataset_output([1, 0, 1], LN3)
    b = rr_dataset_output([0, 0, 1], LN3)
    print(
        f"randomized response, 3 people, neighbours: max event ratio={max_event_ratio(a, b):.4f}  "
        f"max pointwise ratio={math.exp(max_divergence(a, b)):.4f}  "
        f"events checked={1 << len(a)}"
    )
    print(
        f"max_S P(S) - 2Q(S): brute force={max_over_events(a, b, 2.0):.4f}  hockey-stick={hockey_stick(a, b, 2.0):.4f}"
    )

    print("== Randomized response")
    print(
        f"two coins: Pr(yes | yes)={coin_rr_output(1)[1]:.4f}  Pr(yes | no)={coin_rr_output(0)[1]:.4f}  "
        f"eps=ln 3={LN3:.4f}"
    )
    for e in (0.1, 0.5, 1.0, LN3, 2.0, 5.0):
        print(f"eps={e:.4f}: truth prob={rr_truth_prob(e):.4f}")
    pi, n, reps = 0.3, 1000, 10_000
    est = simulate_rr_estimates(rng, pi, n, LN3, reps)
    print(
        f"survey pi={pi:g} n={n} eps=ln3: report rate={pi * 0.75 + (1 - pi) * 0.25:.4f}  "
        f"sd formula={rr_estimator_sd(n, LN3):.4f}  simulated mean={float(np.mean(est)):.4f} sd={float(np.std(est)):.4f}  "
        f"survey sampling sd={math.sqrt(pi * (1 - pi) / n):.4f}"
    )
    for e in (0.1, 1.0, LN3):
        print(f"eps={e:.4f}: estimator sd x sqrt(n)={rr_estimator_sd(1, e):.4f}")
    print(f"eps=0.5: estimator sd at n={n}={rr_estimator_sd(n, 0.5):.4f}")

    print("== Divergence forms")
    print(
        f"randomized response eps0=ln3: delta(0.5)={rr_delta(0.5, LN3):.4f}  delta(1)={rr_delta(1.0, LN3):.4f}  "
        f"delta(ln3)={rr_delta(LN3, LN3):.4f}"
    )
    print(f"Laplace count b=1: delta(0.5)={laplace_delta(0.5, 1.0):.4f}  delta(1)={laplace_delta(1.0, 1.0):.4f}")
    mu = 0.5
    for e in (0.5, 1.0, 2.0, 3.0):
        print(
            f"Gaussian count mu={mu:g}: eps={e:g}  delta={gaussian_delta(e, mu):.3e}  "
            f"Pr[L > eps]={gaussian_loss_tail(e, mu):.3e}  "
            f"ratio={gaussian_loss_tail(e, mu) / gaussian_delta(e, mu):.2f}"
        )
    print(f"Gaussian count mu={mu:g}: Monte Carlo delta(1)={mc_gaussian_delta(rng, 1.0, mu, 1_000_000):.3e}")

    print("== Hypothesis testing")
    for e, d in ((1.0, 1e-5), (LN3, 0.0), (0.1, 0.0)):
        print(
            f"eps={e:.4f} delta={d:g}: max TPR at FPR 1e-3={max_tpr(1e-3, e, d):.6f}  "
            f"at FPR 0.25={max_tpr(0.25, e, d):.4f}"
        )
    t = rr_truth_prob(LN3)
    print(f"randomized response test: FPR={1 - t:.4f} FNR={1 - t:.4f}  FPR + 3 FNR={(1 - t) + 3 * (1 - t):.4f}")
    dg = gaussian_delta(1.0, mu)
    print(
        f"Gaussian mu={mu:g}: TPR at FPR 1e-3={1 - gaussian_tradeoff(1e-3, mu):.4f}  "
        f"bound with eps=1, delta={dg:.3e}: {max_tpr(1e-3, 1.0, dg):.4f}"
    )

    print("== Membership advantage")
    for e in (0.1, 0.5, 1.0, LN3, 2.0, 5.0):
        print(
            f"eps={e:.4f}: bound={advantage_bound(e):.4f}  e^eps-1={yeom_bound(e):.4f}  "
            f"randomized response={rr_advantage(e):.4f}  Laplace count={laplace_tv(e):.4f}"
        )
    print(
        f"eps=1 delta=1e-5: bound={advantage_bound(1.0, 1e-5):.6f}  success prob={(math.e + 1e-5) / (math.e + 1):.6f}"
    )
    trials = 200_000
    for e in (0.5, 1.0, 2.0):
        print(
            f"game eps={e:g}, {trials} trials: randomized response {game_rr(rng, e, trials):.4f} "
            f"(exact {rr_advantage(e):.4f})  Laplace count {game_laplace(rng, e, trials):.4f} "
            f"(exact {laplace_tv(e):.4f})"
        )

    print("== Bayesian view")
    for e in (0.1, 1.0, LN3, 3.0):
        print(f"eps={e:.4f}: " + "  ".join(f"prior {pr:g} -> {posterior_bound(pr, e):.4f}" for pr in (0.01, 0.1, 0.5)))
    dg = gaussian_delta(1.0, mu)
    print(
        f"Gaussian mu={mu:g}, eps=1: delta={dg:.4f}  Pr[L > 2]={gaussian_loss_tail(2.0, mu):.3e}  "
        f"bound delta/(1-e^-1)={privacy_loss_bad_set_bound(1.0, dg, 2.0):.4f}"
    )

    print("== delta and n")
    d_small = 0.1
    on_d, on_d_prime = release_records_neighbour(3, d_small)
    print(
        f"release each record w.p. {d_small:g}, n=3: eps=0 delta={smallest_delta(on_d, on_d_prime, 0.0):.4f}  "
        f"brute force={max(max_over_events(on_d, on_d_prime, 1.0), max_over_events(on_d_prime, on_d, 1.0)):.4f}"
    )
    n = 10_000
    for d in (1.0 / n, 1e-6, 1e-8):
        print(f"n={n} delta={d:g}: expected published={n * d:g}  Pr[at least one]={prob_someone_exposed(n, d):.4e}")
    mean_pub, frac = simulate_exposed(rng, n, 1.0 / n, 100_000)
    print(f"simulated delta=1/n, 100000 runs: mean published={mean_pub:.4f}  fraction with one or more={frac:.4f}")

    print("== Limits")
    k, e = 5, 1.0
    print(
        f"group of k={k}, eps={e:g}: max loss={group_max_loss(k, e):.4f}  advantage bound={advantage_bound(k * e):.4f}  "
        f"Laplace exact={laplace_tv(k * e):.4f}"
    )


# --- Figures --------------------------------------------------------------------------------------


def plot_ratio_band(eps: float = 1.0) -> Figure:
    """The definition as a picture: p must stay inside [e^-ε q, e^ε q]; a Laplace count does, a Gaussian count doesn't."""
    y = np.linspace(-6.0, 7.0, 1301)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    lap_p = np.exp(-np.abs(y) * eps) * eps / 2
    lap_q = np.exp(-np.abs(y - 1) * eps) * eps / 2
    sigma = 2.0
    gau_p = np.exp(-(y**2) / (2 * sigma**2)) / (sigma * math.sqrt(2 * math.pi))
    gau_q = np.exp(-((y - 1) ** 2) / (2 * sigma**2)) / (sigma * math.sqrt(2 * math.pi))
    panels = (
        (axes[0], lap_p, lap_q, rf"count + Laplace noise, $b = 1/\varepsilon$, $\varepsilon = {eps:g}$"),
        (axes[1], gau_p, gau_q, rf"count + Gaussian noise, $\sigma = {sigma:g}$"),
    )
    for ax, p, q, title in panels:
        ax.fill_between(
            y,
            math.exp(-eps) * q,
            math.exp(eps) * q,
            color="C1",
            alpha=0.18,
            label=r"band $[e^{-\varepsilon} q,\ e^{\varepsilon} q]$",
        )
        ax.plot(y, q, color="C1", label=r"$q$: output density on D′")
        ax.plot(y, p, color="C0", label=r"$p$: output density on D")
        out = p > math.exp(eps) * q
        ax.fill_between(
            y, math.exp(eps) * q, p, where=out, color="C3", alpha=0.6, label=r"mass of $p$ above $e^{\varepsilon} q$"
        )
        ax.set(xlabel="released value $y$ (true counts 0 and 1)", ylabel="density", title=title)
        ax.set_yscale("log")
        ax.set_ylim(1e-4, 1.5)
        ax.legend(fontsize=8, loc="lower center")
    return fig


def plot_rr(rng: np.random.Generator, pi: float = 0.3, n: int = 1000, reps: int = 4000) -> Figure:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    ax = axes[0]
    width = 0.35
    for bit, color in ((0, "C1"), (1, "C0")):
        probs = rr_output(bit, LN3)
        ax.bar(
            [0 + (bit - 0.5) * width, 1 + (bit - 0.5) * width],
            probs,
            width,
            color=color,
            label=f"true answer {'yes' if bit else 'no'}",
        )
    ax.set(
        xticks=[0, 1],
        xticklabels=["report no", "report yes"],
        ylabel="probability",
        ylim=(0, 1),
        title=r"Randomized response, $\varepsilon = \ln 3$: every ratio is 3 or 1/3",
    )
    ax.legend(fontsize=9)
    ax = axes[1]
    for e, color in ((LN3, "C0"), (0.5, "C2")):
        est = simulate_rr_estimates(rng, pi, n, e, reps)
        ax.hist(
            est,
            bins=50,
            density=True,
            alpha=0.45,
            color=color,
            label=rf"$\varepsilon = {e:.2f}$, sd {float(np.std(est)):.3f}",
        )
        sd = rr_estimator_sd(n, e)
        xs = np.linspace(pi - 4 * sd, pi + 4 * sd, 300)
        ax.plot(xs, np.exp(-((xs - pi) ** 2) / (2 * sd * sd)) / (sd * math.sqrt(2 * math.pi)), color=color)
    ax.axvline(pi, color="k", linestyle=":", linewidth=1)
    ax.set(
        xlabel="estimated fraction of yes", ylabel="density", title=f"Estimates from {n} people (true fraction {pi:g})"
    )
    ax.legend(fontsize=9)
    return fig


def plot_delta_curves(mu: float = 0.5) -> Figure:
    eps = np.linspace(0.0, 3.0, 301)
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.plot(eps, [rr_delta(float(e), LN3) for e in eps], label=r"randomized response, $\varepsilon_0 = \ln 3$")
    ax.plot(eps, [laplace_delta(float(e), 1.0) for e in eps], label="count + Laplace noise, $b = 1$")
    ax.plot(eps, [gaussian_delta(float(e), mu) for e in eps], label=rf"count + Gaussian noise, $\sigma = {1 / mu:g}$")
    ax.plot(
        eps,
        [gaussian_loss_tail(float(e), mu) for e in eps],
        "C2--",
        label=r"Gaussian: tail $\Pr[L > \varepsilon]$ (sufficient, loose)",
    )
    ax.set_yscale("log")
    ax.set(
        xlabel=r"$\varepsilon$",
        ylabel=r"smallest $\delta$",
        ylim=(1e-4, 1.0),
        title=r"Every mechanism has a curve $\delta(\varepsilon)$, not one pair",
    )
    ax.legend(fontsize=9, loc="lower left")
    return fig


def plot_tradeoff_region(mu: float = 0.5) -> Figure:
    alphas = np.linspace(0.0, 1.0, 1001)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    ax = axes[0]
    ax.plot(alphas, 1 - alphas, "k:", linewidth=1, label="coin flip")
    for e, d, style in ((LN3, 0.0, "C0-"), (1.0, 0.0, "C1-"), (1.0, 0.05, "C1--")):
        ax.plot(
            alphas,
            [dp_tradeoff(float(a), e, d) for a in alphas],
            style,
            label=rf"$f_{{\varepsilon,\delta}}$, $\varepsilon = {e:.2f}$, $\delta = {d:g}$",
        )
    ax.fill_between(alphas, 0, [dp_tradeoff(float(a), LN3, 0.0) for a in alphas], color="C0", alpha=0.12)
    t = rr_truth_prob(LN3)
    ax.plot([1 - t], [1 - t], "ko", label="randomized response test")
    ax.set(
        xlabel="false positive rate",
        ylabel="false negative rate",
        xlim=(0, 1),
        ylim=(0, 1),
        aspect="equal",
        title=r"Forbidden region (shaded) for $\ln 3$-DP",
    )
    ax.legend(fontsize=8)
    ax = axes[1]
    fprs = np.logspace(-5, 0, 400)
    dg = gaussian_delta(1.0, mu)
    ax.plot(
        fprs, [max_tpr(float(a), 1.0, dg) for a in fprs], "C1-", label=rf"bound, $\varepsilon = 1$, $\delta = {dg:.4f}$"
    )
    ax.plot(fprs, [max_tpr(float(a), 1.0, 0.0) for a in fprs], "C1--", label=r"bound, $\varepsilon = 1$, $\delta = 0$")
    ax.plot(
        fprs,
        [1 - gaussian_tradeoff(float(a), mu) for a in fprs],
        "C2-",
        label=rf"Gaussian count $\sigma = {1 / mu:g}$, exact",
    )
    ax.plot(fprs, fprs, "k:", linewidth=1, label="coin flip")
    ax.set(
        xscale="log",
        yscale="log",
        xlabel="false positive rate",
        ylabel="true positive rate",
        xlim=(1e-5, 1),
        ylim=(1e-5, 1.1),
        title="The same bound as an ROC curve, log axes",
    )
    ax.legend(fontsize=8, loc="lower right")
    return fig


def plot_advantage(rng: np.random.Generator, trials: int = 100_000) -> Figure:
    eps = np.linspace(0.0, 4.0, 401)
    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    ax.plot(
        eps,
        [advantage_bound(float(e)) for e in eps],
        "k-",
        label=r"bound $(e^\varepsilon - 1)/(e^\varepsilon + 1)$ = randomized response",
    )
    ax.plot(eps, [min(yeom_bound(float(e)), 1.2) for e in eps], "k:", label=r"looser bound $e^\varepsilon - 1$")
    ax.plot(eps, [laplace_tv(float(e)) for e in eps], "C1-", label="count + Laplace noise, exact")
    pts = [0.25, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0]
    ax.plot(pts, [game_rr(rng, e, trials) for e in pts], "ko", markersize=5, label="game: randomized response")
    ax.plot(
        pts, [game_laplace(rng, e, trials) for e in pts], "s", color="C1", markersize=5, label="game: Laplace count"
    )
    ax.set(
        xlabel=r"$\varepsilon$",
        ylabel="membership advantage TPR − FPR",
        ylim=(0, 1.05),
        title="No adversary beats the bound; randomized response meets it",
    )
    ax.legend(fontsize=8, loc="lower right")
    return fig


def plot_posterior() -> Figure:
    priors = np.logspace(-3, 0, 400)[:-1]
    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    ax.plot(priors, priors, "k:", linewidth=1, label="no change")
    for e, name, color in ((0.1, "0.1", "C0"), (1.0, "1", "C1"), (LN3, r"\ln 3", "C2"), (3.0, "3", "C3")):
        ax.plot(
            priors,
            [posterior_bound(float(pr), e) for pr in priors],
            color=color,
            label=rf"$\varepsilon = {name}$: largest (solid), smallest (dashed)",
        )
        ax.plot(
            priors, [1 - posterior_bound(1 - float(pr), e) for pr in priors], color=color, linestyle="--", linewidth=0.9
        )
    ax.set(
        xscale="log",
        xlabel=r"prior $\Pr[D]$",
        ylabel=r"posterior $\Pr[D \mid y]$",
        ylim=(0, 1),
        title=r"$\varepsilon$-DP: posterior odds within $e^{\pm\varepsilon}$ of prior odds",
    )
    ax.legend(fontsize=8, loc="upper left")
    return fig


def plot_release(n: int = 10_000) -> Figure:
    nd = np.logspace(-3, 1.5, 300)
    fig, ax = plt.subplots(figsize=(7.5, 4.4))
    ax.plot(
        nd,
        [prob_someone_exposed(n, float(x) / n) for x in nd],
        "C3-",
        label=r"$\Pr[\text{someone's record is published}] = 1 - (1 - \delta)^n$",
    )
    ax.plot(nd, np.minimum(nd, 1.0), "k:", linewidth=1, label=r"$n\delta$ (expected number published)")
    for x, lab in ((1.0, r"$\delta = 1/n$"), (1e-2, r"$\delta = 1/n^{1.5}$")):
        ax.axvline(x, color="0.6", linewidth=0.8)
        ax.text(x * 1.1, 0.03, lab, fontsize=9)
    ax.set(
        xscale="log",
        yscale="log",
        xlabel=r"$n\delta$",
        ylabel="probability",
        ylim=(1e-3, 1.5),
        title=f"Release each of n = {n:,} records with probability δ: (0, δ)-DP",
    )
    ax.legend(fontsize=9, loc="lower right")
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
        "dpdef_ratio_band.svg": plot_ratio_band(),
        "dpdef_rr.svg": plot_rr(rng),
        "dpdef_delta_curves.svg": plot_delta_curves(),
        "dpdef_tradeoff_region.svg": plot_tradeoff_region(),
        "dpdef_advantage.svg": plot_advantage(rng),
        "dpdef_posterior.svg": plot_posterior(),
        "dpdef_release.svg": plot_release(),
    }
    for name, fig in figures.items():
        fig.tight_layout()
        fig.savefig(out / name, metadata={"Date": None})  # no timestamp, so output is reproducible
        plt.close(fig)
        print(f"saved {out / name}")


if __name__ == "__main__":
    main()
