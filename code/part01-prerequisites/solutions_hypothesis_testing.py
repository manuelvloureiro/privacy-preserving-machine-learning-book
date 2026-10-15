"""Check the numbers in the worked solutions to the hypothesis-testing exercises.

    uv run python code/part01-prerequisites/solutions_hypothesis_testing.py
    uv run python code/part01-prerequisites/solutions_hypothesis_testing.py --seed 1

Prints every number quoted in book-exercises/part01-prerequisites/04-hypothesis-testing.md (they match exactly with
the default seed). No figures: --out is accepted for a uniform interface and ignored.
"""

import argparse
import itertools
import math
from pathlib import Path

import numpy as np
from hypothesis_testing import (
    HAM,
    SPAM,
    binormal_auc,
    binormal_tpr,
    clopper_pearson,
    discrete_errors,
    gaussian_tradeoff,
    gaussian_tv,
    laplace_tv,
    mixture_auc,
    mixture_tpr,
    normal_cdf,
    normal_isf,
    normal_ppf,
    samples_for_power,
    stein_rate,
)
from numpy.typing import NDArray

# Figures are build output: _build/figures/<part>/, which the build copies into each book.
FIGURES = Path(__file__).resolve().parents[2] / "_build" / "figures" / Path(__file__).parent.name

# --- Exercise 1: symmetry of the Gaussian curve ------------------------------------------------


def max_symmetry_defect(mu: float, alphas: list[float]) -> float:
    """max over a grid of |T(T(α)) - α| for the Gaussian curve at shift μ."""
    return max(abs(gaussian_tradeoff(gaussian_tradeoff(a, mu), mu) - a) for a in alphas)


# --- Exercise 2: swapping the hypotheses ------------------------------------------------------


def lr_vertices(p: tuple[float, ...], q: tuple[float, ...]) -> list[tuple[float, float]]:
    """Corners (α, β) of T(P, Q) on a finite space with p > 0: add outcomes by decreasing q/p."""
    order = sorted(range(len(p)), key=lambda i: q[i] / p[i], reverse=True)
    corners = [(0.0, 1.0)]
    alpha, power = 0.0, 0.0
    for i in order:
        alpha += p[i]
        power += q[i]
        corners.append((alpha, 1 - power))
    return corners


# --- Exercise 3: Laplace total variation ------------------------------------------------------


def laplace_l1_pieces(kappa: float) -> tuple[float, float, float, float]:
    """∫|q - p| for Lap(0, 1) vs Lap(κ, 1) over x <= 0, (0, κ/2), (κ/2, κ), x >= κ (closed form)."""
    tail = 0.5 * (1 - math.exp(-kappa))
    middle = 0.5 * (1 - math.exp(-kappa / 2)) ** 2
    return tail, middle, middle, tail


def laplace_tv_numeric(kappa: float, points: int = 2_000_001) -> float:
    """½∫|q - p| by the trapezoid rule on [-40, 40 + κ]."""
    x = np.linspace(-40.0, 40.0 + kappa, points)
    p = 0.5 * np.exp(-np.abs(x))
    q = 0.5 * np.exp(-np.abs(x - kappa))
    return float(0.5 * np.trapezoid(np.abs(q - p), x))


# --- Exercise 4: sample size ----------------------------------------------------------------


def samples_for_power_exact(alpha: float, power: float, effect: float) -> float:
    """The continuous bound ((Φ⁻¹(1-α) + Φ⁻¹(power)) / effect)², before rounding up."""
    return ((normal_isf(alpha) + normal_isf(1 - power)) / effect) ** 2


def mean_test_power(alpha: float, effect: float, n: int) -> float:
    """Power of the optimal size-α test on the mean of n readings: 1 - T(α) at μ = effect·√n."""
    return 1 - gaussian_tradeoff(alpha, effect * math.sqrt(n))


# --- Exercise 5: a detector that AUC undersells ------------------------------------------------


def equal_auc_shift(auc: float) -> float:
    """The shift μ with Φ(μ/√2) = AUC: the N(0,1)-vs-N(μ,1) detector of the same AUC."""
    return math.sqrt(2) * normal_ppf(auc)


def empirical_mixture_auc(rng: np.random.Generator, w: float, far: float, n: int) -> float:
    """Fraction of n independent (positive, negative) pairs ranked correctly."""
    positives = rng.normal(0, 1, n) + far * (rng.random(n) < w)
    negatives = rng.normal(0, 1, n)
    return float(np.mean(positives > negatives))


# --- Exercise 6: a curve that starts below 1 ---------------------------------------------------


def uniform_tradeoff(alpha: float, s: float) -> float:
    """T(α) = max(0, 1 - s - α) for U[0, 1] against U[s, 1 + s]."""
    return max(0.0, 1 - s - alpha)


def uniform_tv_numeric(s: float, points: int = 1_000_001) -> float:
    """½∫|q - p| for U[0, 1] against U[s, 1 + s], on a midpoint grid over [0, 1 + s]."""
    width = (1 + s) / points
    x = (np.arange(points) + 0.5) * width
    p = ((x >= 0) & (x <= 1)).astype(float)
    q = ((x >= s) & (x <= 1 + s)).astype(float)
    return float(0.5 * np.sum(np.abs(q - p)) * width)


def simulate_uniform_lr_test(rng: np.random.Generator, alpha: float, s: float, n: int) -> tuple[float, float]:
    """Simulated (size, power) of the LR test: reject x > 1, and x in [s, 1] w.p. α/(1 - s)."""
    gamma = alpha / (1 - s)

    def reject(x: NDArray[np.float64]) -> NDArray[np.bool_]:
        coin: NDArray[np.bool_] = rng.random(len(x)) < gamma
        return (x > 1) | ((x >= s) & (x <= 1) & coin)

    x_p = rng.random(n)
    x_q = s + rng.random(n)
    return float(np.mean(reject(x_p))), float(np.mean(reject(x_q)))


# --- Exercise 7: Bayes error and total variation ----------------------------------------------


def min_bayes_error_deterministic(p: tuple[float, ...], q: tuple[float, ...]) -> tuple[float, tuple[int, ...]]:
    """Smallest ½(α + β) over all 2^k deterministic tests, and the rejection region attaining it."""
    best, region = 2.0, ()
    for phi in itertools.product((0.0, 1.0), repeat=len(p)):
        e = discrete_errors(phi, p, q)
        err = 0.5 * (e.alpha + e.beta)
        if err < best - 1e-12:
            best, region = err, tuple(i for i, f in enumerate(phi) if f == 1.0)
    return best, region


def gaussian_bayes_error_grid(mu: float, points: int = 100_001) -> float:
    """min over a grid of α of ½(α + T(α)) for the Gaussian pair."""
    return min(0.5 * (a + gaussian_tradeoff(a, mu)) for a in np.linspace(0.0, 1.0, points).tolist())


# --- Exercise 8: errors decay at the KL rate ---------------------------------------------------


def stein_rate_mills(alpha: float, mu: float, n: int) -> float:
    """Lower bound on -ln(β_n)/n from the Mills ratio Φ(-x) <= φ(x)/x, x = μ√n - Φ⁻¹(1-α) > 0."""
    x = mu * math.sqrt(n) - normal_isf(alpha)
    return (x * x / 2 + math.log(x) + 0.5 * math.log(2 * math.pi)) / n


def stein_rate_upper(alpha: float, mu: float, n: int) -> float:
    """Upper bound on -ln(β_n)/n from Φ(-x) >= φ(x + 1), x = μ√n - Φ⁻¹(1-α) >= 0."""
    x = mu * math.sqrt(n) - normal_isf(alpha)
    return ((x + 1) ** 2 / 2 + 0.5 * math.log(2 * math.pi)) / n


# --- Exercise 9: certifying a low rate ---------------------------------------------------------


def negatives_to_certify(rate: float, tail: float) -> tuple[float, int]:
    """Zero events in n trials certify p < rate when (1 - rate)^n <= tail: the bound and its ceiling."""
    bound = math.log(1 / tail) / -math.log1p(-rate)
    return bound, math.ceil(bound)


# --- The solutions' table ----------------------------------------------------------------------


def print_table(rng: np.random.Generator) -> None:
    print("== Exercise 1: symmetry of the Gaussian curve ==")
    t1 = gaussian_tradeoff(0.05, 1.0)
    print(
        f"[mu=1] T(0.05)={t1:.4f}  T(T(0.05))={gaussian_tradeoff(t1, 1.0):.4f}  T(0.7405)={gaussian_tradeoff(0.7405, 1.0):.4f}"
    )
    grid = np.linspace(0.001, 0.999, 999).tolist()
    worst = max(max_symmetry_defect(mu, grid) for mu in (0.5, 1.0, 2.0, 3.0))
    print(f"max |T(T(a)) - a| over 999 alphas, mu in {{0.5, 1, 2, 3}}: below 1e-9: {worst < 1e-9}")

    print("== Exercise 2: swapping the hypotheses ==")
    p, q = (0.5, 0.5), (0.2, 0.8)
    print("T(P,Q) corners: " + "  ".join(f"({a:.1f}, {b:.1f})" for a, b in lr_vertices(p, q)))
    print("T(Q,P) corners: " + "  ".join(f"({a:.1f}, {b:.1f})" for a, b in lr_vertices(q, p)))

    print("== Exercise 3: Laplace total variation ==")
    pieces = laplace_l1_pieces(1.0)
    print("[kappa=1] integral pieces: " + "  ".join(f"{v:.4f}" for v in pieces) + f"  sum={sum(pieces):.4f}")
    print(
        f"[kappa=1] TV formula={laplace_tv(1.0):.4f}  half sum={sum(pieces) / 2:.4f}  numeric={laplace_tv_numeric(1.0):.4f}"
    )

    print("== Exercise 4: sample size ==")
    for effect in (0.25, 0.125):
        bound = samples_for_power_exact(0.01, 0.9, effect)
        n = samples_for_power(0.01, 0.9, effect)
        print(
            f"[effect {effect}] bound={bound:.1f}  n={n}  power(n-1)={mean_test_power(0.01, effect, n - 1):.4f}"
            f"  power(n)={mean_test_power(0.01, effect, n):.4f}"
        )
    ratio = samples_for_power_exact(0.01, 0.9, 0.125) / samples_for_power_exact(0.01, 0.9, 0.25)
    print(f"bound ratio={ratio:.4f}")

    print("== Exercise 5: a detector that AUC undersells ==")
    w, far = 0.05, 6.0
    auc = mixture_auc(w, far)
    t = normal_isf(1e-3)
    print(
        f"Phi(6/sqrt2)={binormal_auc(far, 1.0):.6f}  AUC={auc:.4f}  simulated AUC={empirical_mixture_auc(rng, w, far, 1_000_000):.4f}"
    )
    print(
        f"threshold={t:.4f}  bulk part={(1 - w) * 1e-3:.5f}  far part={w * normal_cdf(far - t):.5f}"
        f"  TPR@1e-3={mixture_tpr(1e-3, w, far):.4f}"
    )
    mu = equal_auc_shift(auc)
    shift_tpr = binormal_tpr(1e-3, mu, 1.0)
    print(f"equal-AUC shift mu={mu:.4f}  TPR@1e-3={shift_tpr:.4f}  ratio={mixture_tpr(1e-3, w, far) / shift_tpr:.0f}")

    print("== Exercise 6: a curve that starts below 1 ==")
    s = 0.2
    print(
        f"[s=0.2] T(0)={uniform_tradeoff(0.0, s):.1f}  T(0.3)={uniform_tradeoff(0.3, s):.1f}  T(0.9)={uniform_tradeoff(0.9, s):.1f}"
    )
    print(f"[s=0.2] TV numeric={uniform_tv_numeric(s):.4f}")
    size, power = simulate_uniform_lr_test(rng, 0.3, s, 1_000_000)
    print(f"[s=0.2 LR test alpha=0.3] simulated size={size:.4f}  power={power:.4f}")

    print("== Exercise 7: Bayes error and total variation ==")
    err, region = min_bayes_error_deterministic(HAM, SPAM)
    print(f"[spam] min error over 16 deterministic tests={err:.2f}  region={set(region)}")
    tv = gaussian_tv(1.0)
    print(f"[Gaussian mu=1] TV={tv:.4f}  (1-TV)/2={(1 - tv) / 2:.4f}  grid min={gaussian_bayes_error_grid(1.0):.4f}")

    print("== Exercise 8: errors decay at the KL rate ==")
    for n in (10, 100, 1000):
        print(
            f"[n={n}] beta={gaussian_tradeoff(0.05, math.sqrt(n)):.3e}  rate={stein_rate(0.05, 1.0, n):.4f}"
            f"  Mills lower={stein_rate_mills(0.05, 1.0, n):.4f}  upper={stein_rate_upper(0.05, 1.0, n):.4f}"
        )
    for n in (10_000, 100_000, 1_000_000):
        print(f"[n={n}] Mills lower={stein_rate_mills(0.05, 1.0, n):.4f}  upper={stein_rate_upper(0.05, 1.0, n):.4f}")

    print("== Exercise 9: certifying a low rate ==")
    bound, n = negatives_to_certify(1e-3, 0.05)
    print(
        f"[one-sided 95%] bound={bound:.1f}  n={n}  0.999^{n - 1}={(1 - 1e-3) ** (n - 1):.5f}  0.999^{n}={(1 - 1e-3) ** n:.5f}"
    )
    print(
        f"[CP 90% two-sided upper, k=0] n={n - 1}: {clopper_pearson(0, n - 1, 0.90)[1]:.4e}  n={n}: {clopper_pearson(0, n, 0.90)[1]:.4e}"
    )
    bound2, n2 = negatives_to_certify(1e-3, 0.025)
    print(f"[two-sided 95%] bound={bound2:.1f}  n={n2}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=FIGURES, help="accepted for a uniform interface; no figures")
    args = parser.parse_args()
    seed: int = args.seed
    print_table(np.random.default_rng(seed))


if __name__ == "__main__":
    main()
