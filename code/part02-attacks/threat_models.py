"""Check the numbers in the threat-models chapter and draw its figures.

    uv run python code/part02-attacks/threat_models.py
    uv run python code/part02-attacks/threat_models.py --seed 1 --out /tmp/figures

Prints the numbers quoted in 01-threat-models.md (they match exactly with the default seed),
then saves the chapter's figures.

Two toy mechanisms run through the chapter.

- The centroid detector: n records x_i ~ N(0, I_d); the model is their mean θ = (1/n) Σ x_i, and the
  anomaly score of a point x is ‖x − θ‖². The membership game asks whether a target z was one of
  the n records. Three access levels: white-box (θ), black-box (the score of z), label-only (whether
  z is flagged as an anomaly, score > τ).
- The noisy count: records are bits, and the mechanism releases Σ_{x ∈ D} x + σZ with Z ~ N(0, 1).
  The target has bit 1. In the worst-case game the adversary knows the other m records; in the
  average-case game they are m fresh Bernoulli(1/2) bits it does not see.

Conventions: b = 1 means "the target is in the data" (the alternative, Q); b = 0 means it is not
(the null, P). An attack outputs 1 for "member"; TPR = Pr[1 | b = 1], FPR = Pr[1 | b = 0].
"""

import argparse
import math
from collections.abc import Callable
from functools import cache
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
matplotlib.rcParams["svg.hashsalt"] = "threat_models"  # stable element ids, so reruns don't churn the SVGs
matplotlib.rcParams["svg.fonttype"] = "none"  # keep text as text: smaller, diffable files

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from numpy.typing import NDArray

# Figures are build output: _build/figures/<part>/, which the build copies into each book.
FIGURES = Path(__file__).resolve().parents[2] / "_build" / "figures" / Path(__file__).parent.name

Array = NDArray[np.float64]
Mask = NDArray[np.bool_]

# The running parameters.
N_RECORDS = 100  # records behind the centroid detector
DIM = 50  # dimension of each record
SIGMA = 1.0  # noise of the noisy count
CROWD = 100  # other records in the noisy count's average-case game
GAMES = 200_000  # games per world in the noisy-count simulations


# --- Toolkit: advantage, winning probability, precision ------------------------------------


def normal_cdf(x: float) -> float:
    """Φ(x); erfc keeps full relative precision in the lower tail."""
    return 0.5 * math.erfc(-x / math.sqrt(2))


def normal_sf(x: float) -> float:
    """1 − Φ(x), the upper tail, without cancellation."""
    return 0.5 * math.erfc(x / math.sqrt(2))


def normal_pdf(x: float, mu: float = 0.0, sigma: float = 1.0) -> float:
    return math.exp(-((x - mu) ** 2) / (2 * sigma**2)) / (sigma * math.sqrt(2 * math.pi))


def bisect(f: Callable[[float], float], lo: float, hi: float, steps: int = 100) -> float:
    """A root of f in [lo, hi], where f changes sign."""
    f_lo = f(lo)
    for _ in range(steps):
        mid = 0.5 * (lo + hi)
        if (f(mid) > 0) == (f_lo > 0):
            lo, f_lo = mid, f(mid)
        else:
            hi = mid
    return 0.5 * (lo + hi)


def normal_isf(alpha: float) -> float:
    """The x with 1 − Φ(x) = α, by bisection on the tail."""
    return bisect(lambda x: normal_sf(x) - alpha, -40.0, 40.0)


def fraction(mask: Mask) -> float:
    """The fraction of True entries."""
    return float(np.count_nonzero(mask)) / mask.size


def win_probability(tpr: float, fpr: float, prior: float = 0.5) -> float:
    """Pr[guess = b] when b = 1 with probability `prior`: prior·TPR + (1 − prior)(1 − FPR)."""
    return prior * tpr + (1 - prior) * (1 - fpr)


def precision(tpr: float, fpr: float, prior: float) -> float:
    """Pr[b = 1 | attack says 1], by Bayes' rule."""
    return prior * tpr / (prior * tpr + (1 - prior) * fpr)


def empirical_rates(scores_in: Array, scores_out: Array, threshold: float) -> tuple[float, float]:
    """(TPR, FPR) of the attack "say member when score > threshold"."""
    return float(np.mean(scores_in > threshold)), float(np.mean(scores_out > threshold))


def best_threshold_advantage(scores_in: Array, scores_out: Array) -> float:
    """max over thresholds t of TPR − FPR for "say member when score > t", on the given samples."""
    scores = np.concatenate([scores_in, scores_out])
    is_in = np.concatenate([np.ones(scores_in.size), np.zeros(scores_out.size)])
    order = np.argsort(-scores, kind="stable")
    tpr = np.cumsum(is_in[order]) / scores_in.size
    fpr = np.cumsum(1 - is_in[order]) / scores_out.size
    return float(max(0.0, float(np.max(tpr - fpr))))


def empirical_roc(scores_in: Array, scores_out: Array, fprs: Array) -> Array:
    """TPR of the score-threshold attack at each requested FPR (threshold from the out-scores)."""
    thresholds = np.quantile(scores_out, 1 - fprs)
    return np.array([float(np.mean(scores_in > float(t))) for t in thresholds])


# --- Threat models: the centroid detector at three access levels ----------------------------


def centroid_games(rng: np.random.Generator, n: int, d: int, games: int) -> tuple[Array, Array, Array, Array]:
    """Play the membership game on the centroid detector `games` times in each world.

    Returns (z_in, theta_in, z_out, theta_out): targets and centroids, one row per game. In world
    b = 1 the target is one of the n records, so θ = (z + S)/n with S the sum of the n − 1 others,
    S ~ N(0, (n − 1) I). In world b = 0, θ is the mean of n records that do not include z.
    """
    z_in = rng.standard_normal((games, d))
    theta_in = (z_in + math.sqrt(n - 1) * rng.standard_normal((games, d))) / n
    z_out = rng.standard_normal((games, d))
    theta_out = math.sqrt(n) * rng.standard_normal((games, d)) / n
    return z_in, theta_in, z_out, theta_out


def white_box_score(z: Array, theta: Array, n: int) -> Array:
    """The log-likelihood ratio of θ given z, member against non-member, up to a constant.

    Member: θ ~ N(z/n, (n − 1)/n² I). Non-member: θ ~ N(0, I/n).
    """
    return -(n**2) * np.sum((theta - z / n) ** 2, axis=1) / (2 * (n - 1)) + n * np.sum(theta**2, axis=1) / 2


def anomaly_score(z: Array, theta: Array) -> Array:
    """What the black-box interface returns: the squared distance ‖z − θ‖²."""
    return np.sum((z - theta) ** 2, axis=1)


def loss_attack_score(z: Array, theta: Array) -> Array:
    """Raw loss attack: a low anomaly score means member."""
    return -anomaly_score(z, theta)


def calibrated_loss_score(z: Array, theta: Array) -> Array:
    """Black-box, calibrated: compare the score with the target's own ‖z‖², which the adversary knows."""
    return np.sum(z**2, axis=1) - anomaly_score(z, theta)


def label_only_rates(z_in: Array, theta_in: Array, z_out: Array, theta_out: Array, tau: float) -> tuple[float, float]:
    """(TPR, FPR) of "say member when z is not flagged", where the flag is ‖z − θ‖² > τ."""
    tpr = float(np.mean(anomaly_score(z_in, theta_in) <= tau))
    fpr = float(np.mean(anomaly_score(z_out, theta_out) <= tau))
    return tpr, fpr


def access_advantages(rng: np.random.Generator, n: int, d: int, games: int) -> dict[str, float]:
    """Advantage of the best threshold on each view's score, and of the label-only attack (τ = d)."""
    z_in, th_in, z_out, th_out = centroid_games(rng, n, d, games)
    tpr, fpr = label_only_rates(z_in, th_in, z_out, th_out, tau=float(d))
    return {
        "white-box": best_threshold_advantage(white_box_score(z_in, th_in, n), white_box_score(z_out, th_out, n)),
        "black-box, calibrated": best_threshold_advantage(
            calibrated_loss_score(z_in, th_in), calibrated_loss_score(z_out, th_out)
        ),
        "black-box, raw loss": best_threshold_advantage(
            loss_attack_score(z_in, th_in), loss_attack_score(z_out, th_out)
        ),
        "label-only": tpr - fpr,
    }


# --- The noisy count: worst case and average case -------------------------------------------


@cache
def binomial_weights(m: int) -> tuple[float, ...]:
    """Pr[K = k] for K ~ Binomial(m, 1/2), exactly."""
    log_half = m * math.log(0.5)
    return tuple(
        math.exp(math.lgamma(m + 1) - math.lgamma(k + 1) - math.lgamma(m - k + 1) + log_half) for k in range(m + 1)
    )


def worst_case_tv(sigma: float) -> float:
    """TV(N(c, σ²), N(c + 1, σ²)) = 2Φ(1/(2σ)) − 1: the adversary knows the other records."""
    return 2 * normal_cdf(1 / (2 * sigma)) - 1


def mixture_sf(t: float, m: int, sigma: float, shift: int) -> float:
    """Pr[K + shift + σZ > t] with K ~ Binomial(m, 1/2): the count's tail in the average-case game."""
    return sum(w * normal_sf((t - k - shift) / sigma) for k, w in enumerate(binomial_weights(m)) if w > 1e-17)


def average_case_tv(m: int, sigma: float) -> float:
    """TV between the count without and with the target when the m other bits are unknown.

    The likelihood ratio is increasing (a binomial convolved with a normal is log-concave), so the best
    test is a threshold, and by symmetry the densities cross at m/2 + 1/2. With σ = 0 the count is
    exact and the TV is the largest binomial weight.
    """
    if sigma == 0:
        return max(binomial_weights(m))
    t = m / 2 + 0.5
    return mixture_sf(t, m, sigma, 1) - mixture_sf(t, m, sigma, 0)


def average_case_tv_normal(m: int, sigma: float) -> float:
    """The other bits act as extra noise of variance m/4: TV ≈ 2Φ(1/(2√(m/4 + σ²))) − 1."""
    return worst_case_tv(math.sqrt(m / 4 + sigma**2))


def worst_case_tpr(fpr: float, sigma: float) -> float:
    """The Gaussian trade-off curve with shift 1/σ: TPR = 1 − Φ(Φ⁻¹(1 − FPR) − 1/σ)."""
    return normal_sf(normal_isf(fpr) - 1 / sigma)


def average_case_tpr(fpr: float, m: int, sigma: float) -> float:
    """TPR of the threshold test with the given FPR in the average-case game."""
    t = bisect(lambda x: mixture_sf(x, m, sigma, 0) - fpr, -20 * sigma - 1, m + 20 * sigma + 1)
    return mixture_sf(t, m, sigma, 1)


def count_games(rng: np.random.Generator, m: int, sigma: float, games: int, know_rest: bool) -> tuple[Array, Array]:
    """Outputs of the noisy count in world b = 1 (target in) and b = 0 (target out).

    With know_rest the other m bits are fixed at m/2 ones and known to the adversary; otherwise they
    are fresh Bernoulli(1/2) bits in every game.
    """
    if know_rest:
        rest_in = np.full(games, m // 2, dtype=np.float64)
        rest_out = np.full(games, m // 2, dtype=np.float64)
    else:
        rest_in = rng.binomial(m, 0.5, games).astype(np.float64)
        rest_out = rng.binomial(m, 0.5, games).astype(np.float64)
    return rest_in + 1 + sigma * rng.standard_normal(games), rest_out + sigma * rng.standard_normal(games)


def random_target_advantage(rng: np.random.Generator, m: int, sigma: float, games: int) -> float:
    """Average-case game where the target's bit is itself drawn from the population.

    The adversary sees z. If z = 0 the output does not depend on b, and it says "non-member";
    if z = 1 it thresholds the count at m/2 + 1/2.
    """
    out_in, out_out = count_games(rng, m, sigma, games, know_rest=False)
    z_in = rng.integers(0, 2, games)
    z_out = rng.integers(0, 2, games)
    out_in = out_in - (1 - z_in)  # a target with bit 0 adds nothing to the count
    say_in: Mask = (z_in == 1) & (out_in > m / 2 + 0.5)
    say_out: Mask = (z_out == 1) & (out_out > m / 2 + 0.5)
    return fraction(say_in) - fraction(say_out)


# --- Attribute inference with a prior ------------------------------------------------------


def gaussian_hockey_stick(gamma: float, mu: float) -> float:
    """E_γ(N(μ, 1)‖N(0, 1)) = Φ(μ/2 − ln γ/μ) − γ Φ(−μ/2 − ln γ/μ)."""
    a = math.log(gamma) / mu
    return normal_cdf(mu / 2 - a) - gamma * normal_cdf(-mu / 2 - a)


def bayes_attribute_accuracy(prior: float, mu: float) -> float:
    """Best accuracy guessing a bit with Pr[1] = prior from N(0,1) (bit 0) vs N(μ,1) (bit 1).

    max over tests of (1 − prior)(1 − FPR) + prior·TPR = (1 − prior) + prior·E_γ(Q‖P), γ = (1 − prior)/prior.
    """
    return (1 - prior) + prior * gaussian_hockey_stick((1 - prior) / prior, mu)


def simulated_attribute_accuracy(rng: np.random.Generator, prior: float, mu: float, games: int) -> float:
    """Play the attribute game: the Bayes rule says 1 when the likelihood ratio exceeds (1 − prior)/prior."""
    bits: Mask = rng.random(games) < prior
    y = mu * bits + rng.standard_normal(games)
    threshold = math.log((1 - prior) / prior) / mu + mu / 2  # μy − μ²/2 > ln γ
    guess: Mask = y > threshold
    correct: Mask = guess == bits
    return fraction(correct)


# --- Reconstruction from one exact count ---------------------------------------------------


def count_reconstruction_accuracy(m: int) -> float:
    """Expected fraction of m uniform bits recovered from their exact sum: E[max(K, m − K)]/m."""
    return sum(w * max(k, m - k) for k, w in enumerate(binomial_weights(m))) / m


def simulated_reconstruction_accuracy(rng: np.random.Generator, m: int, games: int) -> float:
    bits = rng.integers(0, 2, (games, m))
    count = bits.sum(axis=1)
    guess = (2 * count > m).astype(np.int64)  # majority; ties guess 0, which is as good as 1
    correct: Mask = bits == guess[:, None]
    return fraction(correct)


# --- Releasing one random record -----------------------------------------------------------


def release_one_record_rates(rng: np.random.Generator, n: int, games: int) -> tuple[float, float]:
    """(TPR, FPR) of "member iff the released record equals z" when the mechanism publishes one record.

    Records are continuous, so distinct with probability 1. In world b = 1 the target is record 0 and
    is released when the uniform index is 0; in world b = 0 the released record is an independent draw.
    """
    released_target: Mask = rng.integers(0, n, games) == 0
    tpr = fraction(released_target)
    released_out = rng.standard_normal(games)
    z_out = rng.standard_normal(games)
    collided: Mask = released_out == z_out
    fpr = fraction(collided)
    return tpr, fpr


# --- The chapter's numbers ------------------------------------------------------------------


def print_table(rng: np.random.Generator) -> None:
    n, d = N_RECORDS, DIM
    print(f"[Centroid detector: n = {n} records in d = {d} dimensions, 100,000 games per world]")
    adv = access_advantages(rng, n, d, 100_000)
    print("  " + "  ".join(f"{k}={v:.4f}" for k, v in adv.items()))

    sigma, m, games = SIGMA, CROWD, GAMES
    print(f"[Noisy count, sigma = {sigma:g}, target bit 1]")
    tv_w = worst_case_tv(sigma)
    out_in, out_out = count_games(rng, m, sigma, games, know_rest=True)
    tpr, fpr = empirical_rates(out_in, out_out, m // 2 + 0.5)
    print(
        f"  worst case (rest known): TV={tv_w:.4f}  simulated TPR={tpr:.4f} FPR={fpr:.4f} "
        f"advantage={tpr - fpr:.4f}  win probability={win_probability(tpr, fpr):.4f} "
        f"(1/2 + TV/2 = {0.5 + 0.5 * tv_w:.4f})"
    )
    t3 = m // 2 + sigma * normal_isf(1e-3)
    tpr3, fpr3 = empirical_rates(out_in, out_out, t3)
    exact3 = worst_case_tpr(1e-3, sigma)
    print(f"  worst case at FPR 1e-3: TPR={exact3:.4f} (simulated TPR={tpr3:.4f} at FPR={fpr3:.5f})")
    for prior in (0.5, 0.01):
        print(f"  precision at FPR 1e-3, prior {prior:g}: {precision(exact3, 1e-3, prior):.3f}")
    tv_a = average_case_tv(m, sigma)
    out_in, out_out = count_games(rng, m, sigma, games, know_rest=False)
    tpr, fpr = empirical_rates(out_in, out_out, m / 2 + 0.5)
    print(
        f"  average case (m = {m} unknown bits): TV={tv_a:.4f}  normal approximation="
        f"{average_case_tv_normal(m, sigma):.4f}  simulated advantage={tpr - fpr:.4f}"
    )
    t3 = float(np.quantile(out_out, 1 - 1e-3))
    print(
        f"  average case at FPR 1e-3: TPR={average_case_tpr(1e-3, m, sigma):.4f} "
        f"(simulated {float(np.mean(out_in > t3)):.4f})"
    )
    print(
        f"  random target bit: advantage TV/2={tv_a / 2:.4f}  "
        f"simulated={random_target_advantage(rng, m, sigma, games):.4f}"
    )
    print(
        f"  exact count (sigma = 0): worst-case TV=1  average-case TV={average_case_tv(m, 0):.4f}  "
        f"sqrt(2/(pi m))={math.sqrt(2 / (math.pi * m)):.4f}"
    )

    print("[Attribute inference from the worst-case noisy count, mu = 1/sigma]")
    mu = 1 / sigma
    for prior in (0.5, 0.1):
        print(
            f"  prior {prior:g}: baseline={max(prior, 1 - prior):.4f}  "
            f"best accuracy={bayes_attribute_accuracy(prior, mu):.4f}  "
            f"E_gamma={gaussian_hockey_stick((1 - prior) / prior, mu):.4f}  "
            f"simulated={simulated_attribute_accuracy(rng, prior, mu, games):.4f}"
        )

    print("[Reconstruction of m uniform bits from their exact count]")
    for m_bits in (10, 100, 1000):
        print(
            f"  m={m_bits}: fraction recovered={count_reconstruction_accuracy(m_bits):.4f}  "
            f"1/2 + 1/sqrt(2 pi m)={0.5 + 1 / math.sqrt(2 * math.pi * m_bits):.4f}"
        )
    print(f"  m=10 simulated: {simulated_reconstruction_accuracy(rng, 10, 100_000):.4f}")

    n_release = 1000
    tpr, fpr = release_one_record_rates(rng, n_release, 200_000)
    print(
        f"[Release one random record of n = {n_release}]  TPR={tpr:.5f}  FPR={fpr:.5f}  "
        f"advantage={tpr - fpr:.5f}  (1/n = {1 / n_release:.5f})"
    )

    print(f"[Centroid detector across dimensions, n = {n}, 100,000 games per world]")
    for dim in (1, 500):
        print(f"  d={dim}: white-box={access_advantages(rng, n, dim, 100_000)['white-box']:.4f}")


# --- Figures --------------------------------------------------------------------------------


def plot_game() -> Figure:
    """A diagram of the membership game: challenger, adversary, and what passes between them."""
    fig, ax = plt.subplots(figsize=(10, 4.2))
    ax.set(xlim=(0, 10), ylim=(0, 4.4))
    ax.axis("off")
    boxes = (
        (
            0.2,
            0.9,
            3.6,
            3.3,
            "Challenger",
            "1. draw a dataset $D$ from the population\n"
            "2. train a model $\\theta$ on $D$\n"
            "3. flip a fair bit $b$\n"
            "     $b = 1$: target $z$ drawn from $D$\n"
            "     $b = 0$: fresh $z$ from the population",
            "C0",
        ),
        (
            6.2,
            0.9,
            3.6,
            3.3,
            "Adversary",
            "knows: the training algorithm,\nthe population, its side\ninformation\n\n4. outputs a guess b′",
            "C1",
        ),
    )
    for x, y, w, h, title, body, color in boxes:
        ax.add_patch(
            FancyBboxPatch(
                (x, y), w, h, boxstyle="round,pad=0.05", facecolor=color, alpha=0.12, edgecolor=color, linewidth=1.5
            )
        )
        ax.text(x + w / 2, y + h - 0.3, title, ha="center", va="center", fontsize=13, weight="bold", color=color)
        ax.text(x + 0.2, y + h - 0.7, body, ha="left", va="top", fontsize=10)
    ax.add_patch(FancyArrowPatch((3.9, 3.1), (6.1, 3.1), arrowstyle="-|>", mutation_scale=18, color="k"))
    ax.text(
        5.0,
        3.55,
        "target $z$ and a view of $\\theta$:\nparameters, scores or labels",
        ha="center",
        va="center",
        fontsize=9.5,
    )
    ax.add_patch(FancyArrowPatch((6.1, 1.6), (3.9, 1.6), arrowstyle="-|>", mutation_scale=18, color="k"))
    ax.text(5.0, 1.9, "guess b′", ha="center", va="center", fontsize=10)
    ax.text(
        5.0,
        0.35,
        "advantage = Pr[b′ = 1 | b = 1] − Pr[b′ = 1 | b = 0] = TPR − FPR = 2 Pr[b′ = b] − 1",
        ha="center",
        va="center",
        fontsize=11,
    )
    return fig


def plot_access(rng: np.random.Generator, n: int = N_RECORDS, d: int = DIM) -> Figure:
    """Left: the white-box score in both worlds. Right: advantage against dimension for each view."""
    fig, (left, right) = plt.subplots(1, 2, figsize=(11.5, 4.2))
    z_in, th_in, z_out, th_out = centroid_games(rng, n, d, 20_000)
    s_in = calibrated_loss_score(z_in, th_in)
    s_out = calibrated_loss_score(z_out, th_out)
    bins = np.linspace(float(np.quantile(s_out, 0.001)), float(np.quantile(s_in, 0.999)), 80)
    left.hist(s_out, bins=bins, density=True, alpha=0.5, color="C0", label="non-member ($b = 0$)")
    left.hist(s_in, bins=bins, density=True, alpha=0.5, color="C1", label="member ($b = 1$)")
    left.set(
        xlabel=r"calibrated score $\|z\|^2 - \|z - \theta\|^2$",
        ylabel="density",
        title=f"Black-box scores, n = {n}, d = {d}",
    )
    left.legend(fontsize=9)

    dims = [1, 2, 5, 10, 20, 50, 100, 200, 500]
    curves: dict[str, list[float]] = {}
    for dim in dims:
        for name, value in access_advantages(rng, n, dim, 20_000).items():
            curves.setdefault(name, []).append(value)
    styles = {
        "white-box": ("k", "o", "-"),
        "black-box, calibrated": ("C1", "s", "-"),
        "black-box, raw loss": ("C2", "^", "--"),
        "label-only": ("C3", "v", ":"),
    }
    for name, values in curves.items():
        color, marker, line = styles[name]
        right.plot(dims, values, color=color, marker=marker, linestyle=line, label=name)
    right.set(
        xscale="log",
        xlabel="dimension d (log scale)",
        ylabel="best advantage TPR − FPR",
        ylim=(0, 1),
        title=f"More access, more advantage (n = {n})",
    )
    right.legend(fontsize=9, loc="upper left")
    return fig


def plot_tradeoff(rng: np.random.Generator, m: int = CROWD, sigma: float = SIGMA) -> Figure:
    """Left: the count's distribution in both worlds, rest known vs unknown. Right: ROC on log axes."""
    fig, (left, right) = plt.subplots(1, 2, figsize=(11.5, 4.2))
    c = m // 2
    ys = np.linspace(c - 12, c + 13, 1001)
    p_w = [normal_pdf(float(y), c, sigma) for y in ys]
    q_w = [normal_pdf(float(y), c + 1, sigma) for y in ys]
    weights = binomial_weights(m)
    p_a = [sum(w * normal_pdf(float(y), k, sigma) for k, w in enumerate(weights)) for y in ys]
    q_a = [sum(w * normal_pdf(float(y), k + 1, sigma) for k, w in enumerate(weights)) for y in ys]
    left.plot(ys, p_w, "C0", label="rest known, target out")
    left.plot(ys, q_w, "C1", label="rest known, target in")
    left.plot(ys, p_a, "C0", linestyle="--", label="rest unknown, target out")
    left.plot(ys, q_a, "C1", linestyle="--", label="rest unknown, target in")
    left.set(
        xlabel=r"released count $\sum_{x \in D} x + \sigma Z$",
        ylabel="density",
        ylim=(0, 0.5),
        title=f"Noisy count, σ = {sigma:g}, m = {m} other records",
    )
    left.legend(fontsize=8, loc="upper left")

    fprs = np.logspace(-5, -0.001, 60)  # FPR = 1 exactly has no finite threshold
    worst = [worst_case_tpr(float(f), sigma) for f in fprs]
    average = [average_case_tpr(float(f), m, sigma) for f in fprs]
    right.plot(fprs, worst, "k", label=f"rest known: TV = {worst_case_tv(sigma):.3f}")
    right.plot(fprs, average, "C2", label=f"rest unknown: TV = {average_case_tv(m, sigma):.3f}")
    right.plot(fprs, fprs, color="0.6", linestyle=":", label="guessing: TPR = FPR")
    sim_fprs = np.logspace(-3, -0.3, 8)
    for know, color in ((True, "k"), (False, "C2")):
        s_in, s_out = count_games(rng, m, sigma, 100_000, know_rest=know)
        right.plot(
            sim_fprs, empirical_roc(s_in, s_out, sim_fprs), color=color, marker="o", linestyle="none", markersize=4
        )
    right.set(
        xscale="log",
        yscale="log",
        xlim=(1e-5, 1),
        ylim=(1e-5, 1.2),
        xlabel="FPR (log scale)",
        ylabel="TPR (log scale)",
        title="Membership ROC: curves exact, dots simulated",
    )
    right.legend(fontsize=9, loc="lower right")
    return fig


def plot_average_worst() -> Figure:
    """TV against the number of unknown other records, for several noise levels."""
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ms = [round(10 ** (k / 10)) for k in range(31)]
    ms = sorted(set(ms))
    for sigma, color in ((0.5, "C0"), (1.0, "C1"), (2.0, "C2")):
        ax.plot(ms, [average_case_tv(m, sigma) for m in ms], color=color, label=f"rest unknown, σ = {sigma:g}")
        ax.axhline(worst_case_tv(sigma), color=color, linestyle="--", linewidth=1, label=f"rest known, σ = {sigma:g}")
    ax.plot(ms, [average_case_tv(m, 0) for m in ms], "k.", markersize=4, label="exact count (σ = 0), rest unknown")
    ax.set(
        xscale="log",
        xlabel="number m of other records the adversary does not know (log scale)",
        ylabel="best membership advantage (TV)",
        ylim=(0, 1.08),
        title="The crowd hides the target only from an adversary who does not know it",
    )
    ax.legend(fontsize=8, loc="upper right", ncol=2)
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
        "threat_game.svg": plot_game(),
        "threat_access.svg": plot_access(rng),
        "threat_tradeoff.svg": plot_tradeoff(rng),
        "threat_average_worst.svg": plot_average_worst(),
    }
    for name, fig in figures.items():
        fig.tight_layout()
        fig.savefig(out / name, metadata={"Date": None})  # no timestamp, so output is reproducible
        plt.close(fig)
        print(f"saved {out / name}")


if __name__ == "__main__":
    main()
