"""Check the numbers in the worked solutions to the optimization exercises.

    uv run python code/part01-prerequisites/solutions_optimization.py
    uv run python code/part01-prerequisites/solutions_optimization.py --seed 1

Prints every number quoted in book-exercises/part01-prerequisites/06-optimization.md (they match exactly with the default
seed). The solutions draw no figures; --out is accepted for symmetry with the chapter scripts.
"""

import argparse
import math
from dataclasses import dataclass
from fractions import Fraction
from itertools import combinations, permutations
from pathlib import Path

import numpy as np
from optimization import (
    SIGMOID_SMOOTHNESS,
    Array,
    Data,
    erm_minimizer,
    erm_sensitivity_bound,
    logistic_example_grads,
    logistic_grad,
    logistic_hessian,
    logistic_risk,
    make_logistic_data,
    replace_one,
)

# Figures are build output: _build/figures/<part>/, which the build copies into each book.
FIGURES = Path(__file__).resolve().parents[2] / "_build" / "figures" / Path(__file__).parent.name


# --- Exercise 1: constants of regularized logistic regression ---------------------------------


def regularized_logistic_constants(lam: float, radius: float) -> tuple[float, float, float]:
    """(strong convexity λ, smoothness 1/4 + λ, Lipschitz constant 1 + λR on the ball)."""
    return lam, 0.25 + lam, 1.0 + lam * radius


def worst_gradient_on_ball(lam: float, radius: float) -> float:
    """‖∇(ℓ + λ/2‖w‖²)‖ at w = -R x with ‖x‖ = 1, y = +1: σ(R) + λR, close to 1 + λR."""
    return 1 / (1 + math.exp(-radius)) + lam * radius


# --- Exercise 2: gradient descent on a smooth convex function ---------------------------------


def gd_logistic_gaps(data: Data, beta: float, horizons: list[int]) -> list[float]:
    """F(w_{T+1}) - F* for GD with η = 1/β from w_1 = 0, for each T in `horizons`."""
    f_star = logistic_risk(data, erm_minimizer(data, 0.0))
    w, gaps = np.zeros(data.x.shape[1]), list[float]()
    for t in range(1, max(horizons) + 1):
        w = w - logistic_grad(data, w) / beta
        if t in horizons:
            gaps.append(logistic_risk(data, w) - f_star)
    return gaps


def gd_smooth_bound(beta: float, dist_sq: float, steps: int) -> float:
    """β‖w_1 - w*‖²/(2T)."""
    return beta * dist_sq / (2 * steps)


# --- Exercise 3: SGD on a noisy one-dimensional quadratic ------------------------------------


def noisy_quadratic_exact(lam: float, sigma_sq: float, steps: int) -> float:
    """E F(w_{T+1}) = σ²/(2λT)."""
    return sigma_sq / (2 * lam * steps)


def noisy_quadratic_sgd(
    rng: np.random.Generator, lam: float, sigma: float, steps: int, runs: int, w1: float
) -> tuple[float, float]:
    """Mean and standard error over runs of F(w_{T+1}) = λ/2 w², SGD with η_t = 1/(λt)."""
    w = np.full(runs, w1)
    for t in range(1, steps + 1):
        g = lam * w + sigma * rng.standard_normal(runs)
        w = w - g / (lam * t)
    f = 0.5 * lam * w * w
    return float(np.mean(f)), float(np.std(f) / math.sqrt(runs))


# --- Exercise 4: smoothness of the sigmoid loss ----------------------------------------------


def sigmoid_second_derivative(s: Array) -> Array:
    """σ'' = σ(1 - σ)(1 - 2σ)."""
    u = 0.5 * (1.0 + np.tanh(0.5 * s))
    return u * (1 - u) * (1 - 2 * u)


def sigmoid_curvature_max(half_width: float = 10.0, points: int = 2_000_001) -> tuple[float, float]:
    """Grid maximum of |σ''| on [-a, a] and the |s| where it is attained."""
    s = np.linspace(-half_width, half_width, points)
    v = np.abs(sigmoid_second_derivative(s))
    i = int(np.argmax(v))
    return float(v[i]), abs(float(s[i]))


# --- Exercise 5: sampling without replacement, by enumeration --------------------------------


def pair_covariance(u: list[Fraction]) -> Fraction:
    """E<u_I, u_J> over all ordered pairs of distinct indices."""
    pairs = list(permutations(range(len(u)), 2))
    return sum((u[i] * u[j] for i, j in pairs), Fraction(0)) / len(pairs)


def batch_variance_enumerated(u: list[Fraction], b: int) -> Fraction:
    """E[ū²] over all batches of b distinct indices (u has mean zero)."""
    batches = list(combinations(range(len(u)), b))
    return sum((Fraction(sum((u[i] for i in s), Fraction(0)), b) ** 2 for s in batches), Fraction(0)) / len(batches)


def batch_variance_formula(u: list[Fraction], b: int) -> Fraction:
    """σ₁²/B · (n - B)/(n - 1)."""
    n = len(u)
    sigma1_sq = sum((v * v for v in u), Fraction(0)) / n
    return sigma1_sq / b * Fraction(n - b, n - 1)


# --- Exercise 6: critical batch size ----------------------------------------------------------


def critical_batch(beta: float, delta: float, sigma_sq: float, budget: int) -> float:
    """B_crit = σ √(N/(2βΔ))."""
    return math.sqrt(sigma_sq) * math.sqrt(budget / (2 * beta * delta))


def batch_bound(beta: float, delta: float, sigma_sq: float, budget: int, b: int) -> tuple[float, float]:
    """(best η ≤ 1/β, bound) for 2Δ/(ηT) + βησ²/B with T = N/B."""
    steps = budget / b
    eta = min(1 / beta, math.sqrt(2 * delta * b / (beta * sigma_sq * steps)))
    return eta, 2 * delta / (eta * steps) + beta * eta * sigma_sq / b


# --- Exercise 7: a sharper sensitivity bound --------------------------------------------------


@dataclass(frozen=True)
class Replacement:
    shift: float  # ‖w*(D) - w*(D')‖
    gamma: float  # ‖∇ℓ(w', z') - ∇ℓ(w', z_i)‖
    mu: float  # smallest Hessian eigenvalue of J(·; D) on the segment from w to w'


def segment_strong_convexity(data: Data, lam: float, w: Array, w_new: Array, points: int = 11) -> float:
    """min over s ∈ [0, 1] (on a grid) of λ_min(∇²J(w + s(w' - w)))."""
    return min(
        float(np.linalg.eigvalsh(logistic_hessian(data, w + s * (w_new - w), lam))[0])
        for s in np.linspace(0.0, 1.0, points)
    )


def sensitivity_replacements(rng: np.random.Generator, n: int, lam: float, trials: int) -> list[Replacement]:
    """Replace one random example `trials` times; record shift, Γ and μ for each."""
    data = make_logistic_data(rng, n)
    w = erm_minimizer(data, lam)
    out: list[Replacement] = []
    for _ in range(trials):
        new = make_logistic_data(rng, 1)
        i = int(rng.integers(n))
        w_new = erm_minimizer(replace_one(data, i, new.x[0], float(new.y[0])), lam, w)
        xs = np.stack([new.x[0], data.x[i]])
        ys = np.array([new.y[0], data.y[i]])
        g = logistic_example_grads(xs, ys, np.tile(w_new, (2, 1)))
        diff: Array = g[0] - g[1]
        out.append(
            Replacement(
                shift=float(np.linalg.norm(w_new - w)),
                gamma=float(np.linalg.norm(diff)),
                mu=segment_strong_convexity(data, lam, w, w_new),
            )
        )
    return out


def predicted_ratio(lam: float, data_curvature: float, gamma: float, lipschitz: float = 1.0) -> float:
    """(Γ/(μn)) / (2L/(λn)) = λΓ/(2Lμ), with μ = λ + data curvature."""
    return lam * gamma / (2 * lipschitz * (lam + data_curvature))


# --- Exercise 8: the regularized median -------------------------------------------------------


def median_objective(w: float, z: list[float], lam: float) -> float:
    return sum(abs(w - v) for v in z) / len(z) + 0.5 * lam * w * w


def median_argmin(z: list[float], lam: float, lo: float = -2.0, hi: float = 2.0) -> float:
    """Ternary search: the objective is strictly convex."""
    for _ in range(200):
        a, b = lo + (hi - lo) / 3, hi - (hi - lo) / 3
        if median_objective(a, z, lam) < median_objective(b, z, lam):
            hi = b
        else:
            lo = a
    return 0.5 * (lo + hi)


def median_move(n: int, lam: float) -> float:
    """Movement of the regularized median when one of k + 1 zeros (k ones) becomes a one."""
    k = (n - 1) // 2
    before = [0.0] * (k + 1) + [1.0] * k
    after = [0.0] * k + [1.0] * (k + 1)
    return abs(median_argmin(after, lam) - median_argmin(before, lam))


def median_move_formula(n: int, lam: float) -> float:
    """min(1, 1/(λn))."""
    return min(1.0, 1 / (lam * n))


# --- The solutions' table ---------------------------------------------------------------------


def print_table(rng: np.random.Generator) -> None:
    lam, radius = 0.1, 5.0
    sc, sm, lip = regularized_logistic_constants(lam, radius)
    print(
        f"[ex1] strong convexity={sc:g}  smoothness={sm:g}  Lipschitz={lip:g}  "
        f"gradient at w=-Rx: {worst_gradient_on_ball(lam, radius):.4f}"
    )

    data = make_logistic_data(rng, 1000)
    w_star = erm_minimizer(data, 0.0)
    dist_sq = float(w_star @ w_star)
    horizons = [1, 3, 10, 30]
    print(f"[ex2] |w*|^2={dist_sq:.3f}  beta=0.25")
    for t, gap in zip(horizons, gd_logistic_gaps(data, 0.25, horizons)):
        print(f"[ex2 T={t}] gap={gap:.2e}  bound={gd_smooth_bound(0.25, dist_sq, t):.2e}")

    for t in (100, 1000):
        mean, se = noisy_quadratic_sgd(rng, 1.0, 1.0, t, runs=100_000, w1=5.0)
        print(f"[ex3 T={t}] exact={noisy_quadratic_exact(1.0, 1.0, t):.4f}  simulated={mean:.5f} +- {se:.1e}")

    vmax, smax = sigmoid_curvature_max()
    print(
        f"[ex4] 1/(6 sqrt 3)={SIGMOID_SMOOTHNESS:.4f}  grid max={vmax:.4f} at |s|={smax:.4f}  "
        f"ln(2+sqrt 3)={math.log(2 + math.sqrt(3)):.4f}"
    )

    u = [Fraction(2), Fraction(1), Fraction(-3)]
    print(f"[ex5] covariance={pair_covariance(u)}")
    for b in (1, 2, 3):
        print(f"[ex5 B={b}] enumerated={batch_variance_enumerated(u, b)}  formula={batch_variance_formula(u, b)}")

    beta, delta, sigma_sq, budget = SIGMOID_SMOOTHNESS, 0.5, 1 / 16, 10_000
    print(f"[ex6] critical batch={critical_batch(beta, delta, sigma_sq, budget):.1f}")
    for b in (1, 10, 80, 200, 1000, 10_000):
        eta, bound = batch_bound(beta, delta, sigma_sq, budget, b)
        print(f"[ex6 B={b}] T={budget // b}  eta={eta:.3f}  bound={bound:.2e}")

    lam, n = 0.01, 1000
    print(f"[ex7] predicted ratio={predicted_ratio(lam, 0.0233, 1.0):.3f}")
    reps = sensitivity_replacements(rng, n, lam, trials=20)
    worst = max(reps, key=lambda r: r.shift)
    bound = erm_sensitivity_bound(1.0, lam, n)
    tight = [r.shift / (r.gamma / (r.mu * n)) for r in reps]
    print(
        f"[ex7 worst] shift={worst.shift:.2e}  ratio={worst.shift / bound:.3f}  Gamma={worst.gamma:.3f}  "
        f"mu={worst.mu:.4f}  sharp bound={worst.gamma / (worst.mu * n):.2e}"
    )
    print(
        f"[ex7 all] shift / sharp bound between {min(tight):.2f} and {max(tight):.2f}  "
        f"Gamma between {min(r.gamma for r in reps):.3f} and {max(r.gamma for r in reps):.3f}  "
        f"mu between {min(r.mu for r in reps):.4f} and {max(r.mu for r in reps):.4f}"
    )

    for n, lam in ((101, 0.1), (1001, 0.1), (11, 0.1), (11, 0.05)):
        print(
            f"[ex8 n={n} lam={lam:g}] move={median_move(n, lam):.4f}  formula={median_move_formula(n, lam):.4f}  "
            f"bound={erm_sensitivity_bound(1.0, lam, n):.4f}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=FIGURES, help="unused: the solutions draw no figures")
    args = parser.parse_args()
    seed: int = args.seed
    print_table(np.random.default_rng(seed))


if __name__ == "__main__":
    main()
