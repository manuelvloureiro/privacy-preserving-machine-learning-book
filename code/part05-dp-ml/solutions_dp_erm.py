"""Check the worked solutions to the exercises of the private empirical risk minimization chapter.

    uv run python code/part05-dp-ml/solutions_dp_erm.py

Prints every number quoted in book-exercises/part05-dp-ml/01-dp-erm.md (seed 0). Makes no figures;
--out is accepted and ignored.
"""

import argparse
import math
from pathlib import Path

import numpy as np
from dp_erm import (
    C_LOGISTIC,
    bisect,
    classic_sigma,
    convex_bound,
    convex_limit,
    convex_steps_rule,
    gaussian_mu,
    gaussian_sigma,
    noisy_gd_sigma,
    objective_perturbation_plan,
)
from numpy.typing import NDArray

# Figures are build output: _build/figures/<part>/, which the build copies into each book.
FIGURES = Path(__file__).resolve().parents[2] / "_build" / "figures" / Path(__file__).parent.name

Array = NDArray[np.float64]

# --- Exercise 1: exact against classic calibration ----------------------------------------------


def calibration_ratio(eps: float, delta: float) -> float:
    return classic_sigma(1.0, eps, delta) / gaussian_sigma(1.0, eps, delta)


# --- Exercise 2: ridge regression on a ball --------------------------------------------------------


def ridge_lipschitz(radius: float, x_norm: float = 1.0, y_max: float = 1.0) -> float:
    """Gradient of ½(<w, x> - y)² is (<w, x> - y)x, of length at most (R‖x‖ + |y|)‖x‖ on ‖w‖ ≤ R."""
    return (radius * x_norm + y_max) * x_norm


# --- Exercise 3: objective perturbation's budget ---------------------------------------------------


def smallest_n_positive(eps: float, lam: float, c: float = C_LOGISTIC) -> int:
    """Smallest n with 2 ln(1 + c/(nλ)) < ε, so that no extra ridge is needed."""
    return math.floor(c / (lam * math.expm1(eps / 2))) + 1


# --- Exercise 4: pure against Gaussian noise in d dimensions --------------------------------------


def crossover_dimension(eps: float, delta: float) -> int:
    """Smallest d with d(d+1)/ε² > d σ², σ the Gaussian noise per unit sensitivity: d + 1 > ε² σ²."""
    sigma = gaussian_sigma(1.0, eps, delta)
    return math.floor(eps**2 * sigma**2 - 1) + 1


# --- Exercise 5: noisy GD with weaker composition theorems ----------------------------------------


def advanced_eps(eps0: float, steps: int, delta_prime: float) -> float:
    """Advanced composition of `steps` (ε0, δ0)-DP steps: ε0√(2T ln(1/δ')) + Tε0(e^{ε0} - 1)."""
    return eps0 * math.sqrt(2 * steps * math.log(1 / delta_prime)) + steps * eps0 * math.expm1(eps0)


def sigma_per_step(n: int, steps: int, eps: float, delta: float, method: str) -> float:
    """Noise on the mean gradient (sensitivity 2/n per step) under three accounting methods."""
    sens = 2.0 / n
    if method == "exact":
        return sens * math.sqrt(steps) / gaussian_mu(eps, delta)
    if method == "basic":
        return gaussian_sigma(sens, eps / steps, delta / steps)
    # advanced: half of δ to the theorem's δ', half split over the steps
    eps0 = bisect(lambda e: advanced_eps(e, steps, delta / 2) > eps, 0.0, eps)
    return gaussian_sigma(sens, eps0, delta / (2 * steps))


# --- Exercise 6: a sharper Jacobian bound ----------------------------------------------------------


def jacobian_log_ratio(rng: np.random.Generator, d: int, n: int, lam: float, c: float) -> tuple[float, float]:
    """|ln det(A')/det(A)| for a random A = Σ c_i x_i x_iᵀ + nλI and a replaced record; and ln(1 + c/(nλ))."""
    xs = rng.standard_normal((n, d))
    xs /= np.linalg.norm(xs, axis=1, keepdims=True)
    xs *= rng.uniform(0.0, 1.0, (n, 1))
    curv = rng.uniform(0.0, c, n)
    a = (xs * curv[:, None]).T @ xs + n * lam * np.eye(d)
    x_new = rng.standard_normal(d)
    x_new /= np.linalg.norm(x_new)
    a2 = a - curv[-1] * np.outer(xs[-1], xs[-1]) + c * np.outer(x_new, x_new)
    _, logdet1 = np.linalg.slogdet(a)
    _, logdet2 = np.linalg.slogdet(a2)
    return abs(float(logdet2 - logdet1)), math.log1p(c / (n * lam))


def worst_jacobian_share(rng: np.random.Generator, trials: int, d: int, n: int, lam: float) -> float:
    """Largest |ln det ratio| / ln(1 + c/(nλ)) over random instances; the exercise shows it is at most 1."""
    worst = 0.0
    for _ in range(trials):
        val, bound = jacobian_log_ratio(rng, d, n, lam, C_LOGISTIC)
        worst = max(worst, val / bound)
    return worst


def worst_jacobian_share_extreme(d: int, n: int, lam: float) -> float:
    """The extreme case: other records contribute no curvature, the added record has curvature c along
    an axis and the removed one curvature 0. Then det ratio = 1 + c/(nλ) exactly."""
    a = n * lam * np.eye(d)
    x_new = np.zeros(d)
    x_new[0] = 1.0
    a2 = a + C_LOGISTIC * np.outer(x_new, x_new)
    _, l1 = np.linalg.slogdet(a)
    _, l2 = np.linalg.slogdet(a2)
    return float(l2 - l1) / math.log1p(C_LOGISTIC / (n * lam))


# --- Exercise 7: how tight must an approximate minimizer be ---------------------------------------


def gradient_tolerance(n: int, lipschitz: float, slack: float) -> float:
    """Largest g with 2g/λ ≤ slack · 2L/(λn): g = slack L/n."""
    return slack * lipschitz / n


# --- Exercise 8: privacy noise as gradient variance ----------------------------------------------


def nonprivate_bound(radius: float, steps: int, lipschitz: float = 1.0) -> float:
    """RL/√T: projected gradient descent's guarantee when the gradients carry no noise."""
    return radius * lipschitz / math.sqrt(steps)


# --- Exercise 9: how many steps without a ridge ---------------------------------------------------


def steps_within(n: int, d: int, eps: float, delta: float, slack: float) -> int:
    """Smallest T with RL√(1/T + a) ≤ (1 + slack) RL√a, a = 4d/(n²μ²): T ≥ 1/(((1 + slack)² - 1) a)."""
    a = 4 * d / (n**2 * gaussian_mu(eps, delta) ** 2)
    return math.ceil(1 / (((1 + slack) ** 2 - 1) * a))


def print_table(rng: np.random.Generator) -> None:
    print("== Exercise 1")
    for eps in (0.25, 0.5, 1.0):
        print(
            f"eps={eps}: classic sigma={classic_sigma(1.0, eps, 1e-5):.4f}  exact sigma="
            f"{gaussian_sigma(1.0, eps, 1e-5):.4f}  ratio={calibration_ratio(eps, 1e-5):.3f}"
        )
    print("== Exercise 2")
    lip = ridge_lipschitz(2.0)
    print(
        f"R=2: L={lip:.1f}  sensitivity at lambda=0.1, n=1000: {2 * lip / (0.1 * 1000):.3f}  "
        f"Gaussian sigma (eps=1, delta=1e-5)={gaussian_sigma(2 * lip / (0.1 * 1000), 1.0, 1e-5):.4f}"
    )
    print("== Exercise 3")
    print(f"eps=0.1, lambda=0.01: smallest n with eps'>0 is {smallest_n_positive(0.1, 0.01)}")
    for n in (487, 488):
        plan = objective_perturbation_plan(n, 0.01, 0.1)
        print(f"n={n}: eps'={plan.eps_noise:.5f}  extra lambda={plan.extra_lam:.4f}")
    plan = objective_perturbation_plan(100, 0.01, 0.1)
    print(
        f"n=100: eps'={plan.eps_noise:.4f}  extra lambda={plan.extra_lam:.4f}  total lambda="
        f"{0.01 + plan.extra_lam:.4f}  Jacobian share 2ln(1+c/(n(lam+extra)))="
        f"{2 * math.log1p(C_LOGISTIC / (100 * (0.01 + plan.extra_lam))):.4f}"
    )
    print("== Exercise 4")
    s = gaussian_sigma(1.0, 1.0, 1e-5)
    print(f"eps=1, delta=1e-5: sigma^2={s**2:.3f}  crossover d={crossover_dimension(1.0, 1e-5)}")
    for d in (12, 13, 100):
        print(f"d={d}: pure d(d+1)={d * (d + 1)}  Gaussian d sigma^2={d * s**2:.1f}")
    s2 = gaussian_sigma(1.0, 0.1, 1e-5)
    print(f"eps=0.1: eps^2 sigma^2={0.01 * s2**2:.3f}  crossover d={crossover_dimension(0.1, 1e-5)}")
    print("== Exercise 5")
    for steps in (10, 100, 1000):
        ex = sigma_per_step(10_000, steps, 1.0, 1e-5, "exact")
        ba = sigma_per_step(10_000, steps, 1.0, 1e-5, "basic")
        ad = sigma_per_step(10_000, steps, 1.0, 1e-5, "advanced")
        print(f"T={steps}: exact sigma={ex:.5f}  basic={ba:.5f} ({ba / ex:.2f}x)  advanced={ad:.5f} ({ad / ex:.2f}x)")
    print("== Exercise 6")
    share = worst_jacobian_share(rng, 2000, 5, 20, 0.05)
    print(f"d=5, n=20, lambda=0.05, 2000 random instances: largest |ln det ratio|/ln(1+c/(n lam)) = {share:.4f}")
    print(f"extreme instance: ratio = {worst_jacobian_share_extreme(5, 20, 0.05):.4f}")
    a = C_LOGISTIC / (5 * 0.1)
    print(
        f"n=5, lambda=0.1: ln(1+a)={math.log1p(a):.4f}  2ln(1+a)={2 * math.log1p(a):.4f}  "
        f"eps' with the sharper bound={1 - math.log1p(a):.4f}"
    )
    print("== Exercise 7")
    print(f"n=10000, L=1, 10% slack: g={gradient_tolerance(10_000, 1.0, 0.1):.0e}")
    print("== Exercise 8")
    n, d, radius = 1_000, 10, 5.0
    steps = convex_steps_rule(n, d, 1.0, 1e-5)
    sigma = noisy_gd_sigma(n, steps, 1.0, 1e-5)
    print(
        f"n={n}, d={d}, R=5: T={steps}  sigma={sigma:.4f}  d sigma^2={d * sigma**2:.4f}  "
        f"private bound={convex_bound(n, d, steps, radius, 1.0, 1e-5):.4f}  "
        f"noiseless bound={nonprivate_bound(radius, steps):.4f}  "
        f"ratio={convex_bound(n, d, steps, radius, 1.0, 1e-5) / nonprivate_bound(radius, steps):.4f}"
    )
    print("== Exercise 9")
    for n in (1_000, 10_000):
        t = steps_within(n, 10, 1.0, 1e-5, 0.1)
        print(
            f"n={n}, d=10: T within 10% of the limit={t}  bound={convex_bound(n, 10, t, 5.0, 1.0, 1e-5):.4f}  "
            f"limit={convex_limit(n, 10, 5.0, 1.0, 1e-5):.4f}  gradient evaluations n*T={n * t:.3g}  "
            f"one record per step, n^2 steps={n**2:.3g}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=FIGURES, help="ignored: this script makes no figures")
    args = parser.parse_args()
    print_table(np.random.default_rng(args.seed))


if __name__ == "__main__":
    main()
