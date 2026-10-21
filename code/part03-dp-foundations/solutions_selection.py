"""Check the worked solutions to the exercises of the selection chapter.

    uv run python code/part03-dp-foundations/solutions_selection.py

Prints every number quoted in book-exercises/part03-dp-foundations/05-selection.md (seed 0). Makes no figures; --out is
accepted and ignored.

Conventions as in selection.py: k candidates with scores u(D, r) of sensitivity Δ, Laplace and
Gumbel noise given by their scale, natural logarithms.
"""

import argparse
import math
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from selection import (
    EULER_GAMMA,
    FIVE_SCORES,
    MODEL_SCORES,
    em_gap_threshold,
    em_probabilities,
    gumbel_max_mean,
    laplace_rnm_probabilities,
    laplace_sf,
    learner_samples,
    non_monotone_ratio,
    report_noisy_max,
    sparse_alpha,
    sparse_alpha_approx,
    svt_alpha,
    two_candidate_best,
)

# Figures are build output: _build/figures/<part>/, which the build copies into each book.
FIGURES = Path(__file__).resolve().parents[2] / "_build" / "figures" / Path(__file__).parent.name

# --- Exercise 1: the factor 2 is needed ---------------------------------------------------------


def non_monotone_ratio_closed_form(k: int, eps: float) -> float:
    """ln[e^ε (1 + (k-1) e^ε) / (e^ε + k - 1)], the loss of output 1 with weights e^{εu}."""
    e = math.exp(eps)
    return math.log(e * (1 + (k - 1) * e) / (e + k - 1))


# --- Exercise 2: a private median ---------------------------------------------------------------


def median_dataset(n: int = 1001) -> list[int]:
    """n values in {0, ..., 100}, skewed toward 0: x_i = floor(100 (i/(n-1))²). The median is 25."""
    return [math.floor(100 * (i / (n - 1)) ** 2) for i in range(n)]


def median_scores(data: list[int], candidates: range) -> list[float]:
    """u(D, r) = -|#{x < r} - #{x > r}|."""
    x = np.asarray(data)
    return [-float(abs(int(np.sum(x < r)) - int(np.sum(x > r)))) for r in candidates]


def median_sensitivity(data: list[int], candidates: range) -> float:
    """Largest change of any score when one record (any value in the domain) is added or one is removed."""
    base = np.asarray(median_scores(data, candidates))
    worst = 0.0
    for v in candidates:  # add a record with value v
        worst = max(worst, float(np.max(np.abs(np.asarray(median_scores(data + [v], candidates)) - base))))
    for v in sorted(set(data)):  # remove one record with value v
        smaller = list(data)
        smaller.remove(v)
        worst = max(worst, float(np.max(np.abs(np.asarray(median_scores(smaller, candidates)) - base))))
    return worst


# --- Exercise 3: the noisy maximum value --------------------------------------------------------


def simulated_gumbel_max_mean(rng: np.random.Generator, scores: list[float], beta: float, n: int) -> float:
    u = np.asarray(scores, dtype=np.float64)
    return float(np.mean(np.max(u + rng.gumbel(0.0, beta, (n, len(u))), axis=1)))


# --- Exercise 4: splitting the budget of AboveThreshold -----------------------------------------


def svt_alpha_split(eps1: float, eps2: float, k: int, beta: float, sensitivity: float = 1.0) -> float:
    """α = max(2Δ ln(2/β)/ε1, 4Δ ln(2k/β)/ε2) for ρ ~ Lap(Δ/ε1), ν_i ~ Lap(2Δ/ε2)."""
    return max(2 * sensitivity * math.log(2 / beta) / eps1, 4 * sensitivity * math.log(2 * k / beta) / eps2)


def svt_best_split(eps: float, k: int, beta: float, sensitivity: float = 1.0) -> tuple[float, float]:
    """The split that equalizes the two terms: α* = (2 ln(2/β) + 4 ln(2k/β))Δ/ε and ε1 = 2Δ ln(2/β)/α*."""
    a = 2 * sensitivity * math.log(2 / beta)
    c = 4 * sensitivity * math.log(2 * k / beta)
    alpha = (a + c) / eps
    return alpha, a / alpha


# --- Exercise 5: no query noise -----------------------------------------------------------------


def no_noise_bot_top(rng: np.random.Generator, f: tuple[float, float], b_rho: float, n: int) -> float:
    """Pr[⊥⊤] for two queries compared with T + ρ (T = 0) and no query noise, by simulation."""
    rho = rng.laplace(0.0, b_rho, n)
    first_below: NDArray[np.bool_] = f[0] < rho
    second_above: NDArray[np.bool_] = f[1] >= rho
    return float(np.count_nonzero(first_below & second_above)) / n


# --- Exercise 6: two candidates -----------------------------------------------------------------


def simulated_two_candidate(rng: np.random.Generator, d_over_b: float, n: int) -> tuple[float, float]:
    em = report_noisy_max(rng, [d_over_b, 0.0], 1.0, n, "gumbel")
    lap = report_noisy_max(rng, [d_over_b, 0.0], 1.0, n, "laplace")
    return float(np.count_nonzero(em == 0)) / n, float(np.count_nonzero(lap == 0)) / n


def print_table(rng: np.random.Generator) -> None:
    print("[Exercise 1: weights e^(eps u) on non-monotone scores, eps = 1]")
    for k in (2, 3, 10, 1000):
        print(
            f"  k={k}: loss={non_monotone_ratio(k, 1.0):.4f}  closed form={non_monotone_ratio_closed_form(k, 1.0):.4f}"
        )
    print(f"  limit 2 eps = {2.0:.4f}")

    print("[Exercise 2: private median, n = 1001 values in {0..100}, eps = 1]")
    data = median_dataset()
    cands = range(101)
    scores = median_scores(data, cands)
    best = max(scores)
    print(
        f"  true median={sorted(data)[len(data) // 2]}  OPT={best:g}  argmax={scores.index(best)}  "
        f"sensitivity (brute force)={median_sensitivity(data, cands):g}"
    )
    bound = em_gap_threshold(101, 1.0, math.log(20))
    p = em_probabilities(scores, 1.0)
    gaps = best - np.asarray(scores)
    print(
        f"  gap bound={bound:.2f}  exact Pr[gap > bound]={float(p[gaps > bound].sum()):.2e}  "
        f"Pr[output = 25]={float(p[25]):.4f}  E|output - 25|={float(np.sum(p * np.abs(np.arange(101) - 25))):.4f}"
    )
    x = np.asarray(data)
    for r in (24, 25, 26):
        print(
            f"  r={r}: below={int(np.sum(x < r))}  above={int(np.sum(x > r))}  u={scores[r]:g}  gap={best - scores[r]:g}"
        )
    print(f"  records equal to 25: {np.count_nonzero(x == 25)}")
    near = [r for r in cands if gaps[r] <= bound]
    print(f"  candidates within the bound: {near[0]}..{near[-1]}")

    print("[Exercise 3: max of u_r + Gumbel(2), scores (4, 3, 3, 1, 0)]")
    beta = 2.0
    lse = beta * math.log(sum(math.exp(u / beta) for u in FIVE_SCORES))
    print(
        f"  location beta ln sum e^(u/beta)={lse:.4f}  mean={gumbel_max_mean(FIVE_SCORES, beta):.4f}  "
        f"(location + beta gamma = {lse + beta * EULER_GAMMA:.4f})  "
        f"simulated mean={simulated_gumbel_max_mean(rng, FIVE_SCORES, beta, 1_000_000):.4f}"
    )
    print(f"  max u + beta ln k = {max(FIVE_SCORES) + beta * math.log(len(FIVE_SCORES)):.4f}")

    print("[Exercise 4: AboveThreshold budget split, k = 1000, beta = 0.05, eps = 1]")
    alpha_eq = svt_alpha_split(0.5, 0.5, 1000, 0.05)
    alpha_best, eps1 = svt_best_split(1.0, 1000, 0.05)
    print(
        f"  equal split alpha={alpha_eq:.2f} (Dwork-Roth formula {svt_alpha(1000, 0.05, 1.0):.2f})  "
        f"terms: threshold {2 * math.log(40) / 0.5:.2f}, queries {4 * math.log(40000) / 0.5:.2f}"
    )
    print(
        f"  a=2 ln(2/beta)={2 * math.log(40):.2f}  c=4 ln(2k/beta)={4 * math.log(40000):.2f}  "
        f"improvement={1 - alpha_best / alpha_eq:.3f}"
    )
    print(
        f"  best split: alpha={alpha_best:.2f}  eps1={eps1:.3f}  eps2={1 - eps1:.3f}  "
        f"check={svt_alpha_split(eps1, 1 - eps1, 1000, 0.05):.2f}"
    )

    print("[Exercise 5: no query noise, T = 0, rho ~ Lap(2)]")
    exact = laplace_sf(0.0, 2.0) - laplace_sf(1.0, 2.0)
    print(f"  Pr[bot top | D]: exact={exact:.4f}  simulated={no_noise_bot_top(rng, (0.0, 1.0), 2.0, 1_000_000):.4f}")
    print(f"  Pr[bot top | D']: exact=0  simulated={no_noise_bot_top(rng, (1.0, 0.0), 2.0, 1_000_000):.4f}")

    print("[Exercise 6: two candidates, scores d apart, scale b]")
    for x in (0.5, 1.0, 2.0, 4.0):
        em, lap = two_candidate_best(x)
        print(f"  d/b={x:g}: EM={em:.4f}  Laplace={lap:.4f}")
    sim_em, sim_lap = simulated_two_candidate(rng, 1.0, 1_000_000)
    print(f"  d/b=1 simulated: Gumbel={sim_em:.4f}  Laplace={sim_lap:.4f}")
    u = np.asarray(MODEL_SCORES)
    p_em, p_lap = em_probabilities(MODEL_SCORES, 1.0), laplace_rnm_probabilities(MODEL_SCORES, 2.0)
    print(
        f"  50 models, eps=1, scale 2: EM Pr[best]={float(p_em[0]):.4f} E[gap]={float(np.sum(p_em * (u.max() - u))):.2f}  "
        f"Laplace Pr[best]={float(p_lap[0]):.4f} E[gap]={float(np.sum(p_lap * (u.max() - u))):.2f}"
    )

    print("[Exercise 7: the Laplace mechanism as an exponential mechanism, eps = 1, Delta = 1]")
    grid = np.linspace(-60.0, 61.0, 1_210_001)
    step = float(grid[1] - grid[0])
    for scale, name in ((2.0, "generic e^(eps u/2)"), (1.0, "without the 2")):
        dens_d = np.exp(-np.abs(grid - 0.0) / scale)
        dens_d2 = np.exp(-np.abs(grid - 1.0) / scale)
        z_d, z_d2 = float(dens_d.sum() * step), float(dens_d2.sum() * step)
        ratio = float(np.max(np.abs(np.log(dens_d / z_d) - np.log(dens_d2 / z_d2))))
        print(
            f"  {name}: Z(D)={z_d:.4f}  Z(D')={z_d2:.4f}  largest |ln density ratio|={ratio:.4f}  "
            f"rms error={math.sqrt(2) * scale:.4f}"
        )

    print("[Exercise 8: records for the threshold learner, beta = 0.05]")
    for size, a, e in ((1001, 0.05, 1.0), (10**6, 0.05, 1.0), (1001, 0.05, 0.01)):
        stat, priv = learner_samples(a, e, 0.05, size)
        print(f"  |C|={size} alpha={a:g} eps={e:g}: sampling={stat:.0f}  privacy={priv:.0f}")
    size = 1001
    crit = 0.05 * math.log(size / 0.05) / (2 * math.log(2 * size / 0.05))
    print(f"  privacy dominates below eps={crit:.4f} (alpha=0.05, |C|=1001)")

    print("[Exercise 9: Sparse with basic and advanced composition, k = 1000, beta = 0.05, eps = 1, delta = 1e-6]")
    cross = next(
        c for c in range(1, 1000) if sparse_alpha_approx(1000, c, 0.05, 1.0, 1e-6) < sparse_alpha(1000, c, 0.05, 1.0)
    )
    print(f"  advanced beats basic from c={cross}")
    for c in (cross - 1, cross, 1000):
        print(
            f"  c={c}: basic={sparse_alpha(1000, c, 0.05, 1.0):.0f}  advanced={sparse_alpha_approx(1000, c, 0.05, 1.0, 1e-6):.0f}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=FIGURES, help="accepted for a uniform interface; no figures")
    args = parser.parse_args()
    seed: int = args.seed
    print_table(np.random.default_rng(seed))


if __name__ == "__main__":
    main()
