"""Check the worked solutions to the exercises of the DP-FTRL and matrix factorization chapter.

    uv run python code/part05-dp-ml/solutions_matrix_mechanisms.py

Prints every number quoted in book-exercises/part05-dp-ml/04-matrix-mechanisms.md (seed 0). Makes no
figures; --out is accepted and ignored.
"""

import argparse
import math
from pathlib import Path

import numpy as np
from dp_sgd import dpsgd_epsilon
from matrix_mechanisms import (
    dpsgd_sigma,
    final_prefix_noise,
    inverse_series,
    optimal_factorization,
    prefix_matrix,
    prefix_singular_values,
    profile,
    sqrt_coeffs,
    toeplitz_lower,
    toeplitz_sensitivity,
    trace_norm_bound,
)

# Figures are build output: _build/figures/<part>/, which the build copies into each book.
FIGURES = Path(__file__).resolve().parents[2] / "_build" / "figures" / Path(__file__).parent.name


# --- Exercise 1: when does the tree beat independent noise? ---------------------------------------


def tree_crossover(limit: int = 4096) -> int:
    """Smallest T at which the tree's root-mean-square error is below independent noise's."""
    for t in range(2, limit):
        if profile("tree", t).rmse < profile("identity", t).rmse:
            return t
    return -1


# --- Exercise 3: the optimum for T = 2 ------------------------------------------------------------


def two_step_objective(rho: float) -> float:
    """‖L‖_F² for X = [[1, ρ], [ρ, 1]] and A the 2×2 prefix matrix: (3 - 2ρ)/(1 - ρ²)."""
    return (3 - 2 * rho) / (1 - rho * rho)


# --- Exercise 5: two participations under the square root ---------------------------------------------


def two_participation_parts(t: int, sep: int) -> tuple[float, float, float]:
    """(‖r_0‖², ‖r_sep‖², 2 r_0·r_sep) for the square-root encoder of size T: columns 0 and sep."""
    r = sqrt_coeffs(t)
    a = float(np.sum(r[:t] ** 2))
    b = float(np.sum(r[: t - sep] ** 2))
    cross = 2 * float(np.dot(r[sep:t], r[: t - sep]))
    return a, b, cross


def print_table(rng: np.random.Generator) -> None:
    del rng  # nothing random here
    print("== Exercise 1")
    for t in (8, 16):
        print(f"T={t}: tree rmse={profile('tree', t).rmse:.3f}  identity rmse={profile('identity', t).rmse:.3f}")
    print(f"smallest T with tree below identity: {tree_crossover()}")
    t = tree_crossover()
    print(f"T={t}: tree rmse={profile('tree', t).rmse:.3f}  identity rmse={profile('identity', t).rmse:.3f}")

    print("== Exercise 2")
    r4 = toeplitz_lower(sqrt_coeffs(4), 4)
    print(f"max |R R - A| at T=4: {float(np.max(np.abs(r4 @ r4 - prefix_matrix(4)))):.1e}")
    sens2 = float(np.sum(sqrt_coeffs(4) ** 2))
    print(f"T=4: sum r_k^2={sens2:.4f}  sens={math.sqrt(sens2):.4f}")
    h = inverse_series(sqrt_coeffs(4), 4)
    print("h_0..h_3: " + ", ".join(f"{x:.4f}" for x in h))
    print(
        f"variance of the step-4 noise / s^2 = {float(np.sum(h**2)):.4f}  sd ratio = {math.sqrt(float(np.sum(h**2))):.4f}"
    )
    print(
        f"T=4: square-root max error={profile('sqrt', 4).max_error:.4f}  identity max error={profile('identity', 4).max_error:.4f}"
    )

    print("== Exercise 3")
    rho = (3 - math.sqrt(5)) / 2
    print(f"rho*={rho:.4f}  objective={two_step_objective(rho):.4f}  (3+sqrt5)/2={(3 + math.sqrt(5)) / 2:.4f}")
    o = optimal_factorization(prefix_matrix(2))
    x = o.c.T @ o.c
    print(f"fixed point: X_12={float(x[0, 1]):.4f}  ||L||_F^2={float(np.sum(o.b**2)):.4f}")
    s2 = prefix_singular_values(2)
    print(
        f"singular values of A (T=2): {float(s2[0]):.4f}, {float(s2[1]):.4f}  trace-norm bound ||A||_*^2/T={float(s2.sum()) ** 2 / 2:.4f}"
    )
    rs = toeplitz_lower(sqrt_coeffs(2), 2)
    print(
        f"square root: sens^2 * ||L||_F^2 = {float(np.sum(sqrt_coeffs(2) ** 2)) * float(np.sum(rs**2)):.4f}  identity: {3.0:.4f}"
    )

    print("== Exercise 4")
    for t in (100, 1000, 10_000):
        bound = trace_norm_bound(t)
        sq = profile("sqrt", t).rmse
        print(f"T={t}: lower bound={bound:.3f}  square root={sq:.3f}  ratio={sq / bound:.3f}")
    print(f"ln(10)/pi={math.log(10) / math.pi:.3f}")

    print("== Exercise 5")
    a, b, cross = two_participation_parts(200, 100)
    print(
        f"T=200, columns 0 and 100: ||r_0||^2={a:.4f}  ||r_100||^2={b:.4f}  cross={cross:.4f}  total={a + b + cross:.4f}"
    )
    print(f"square-root sensitivity={toeplitz_sensitivity(sqrt_coeffs(200), 200, 100, 2):.4f}")
    print(
        f"banded sensitivity={toeplitz_sensitivity(sqrt_coeffs(200)[:100].copy(), 200, 100, 2):.4f}  "
        f"sqrt(2)*||r||={math.sqrt(2 * float(np.sum(sqrt_coeffs(100) ** 2))):.4f}"
    )

    print("== Exercise 6")
    for name in ("dpsgd", "identity", "sqrt"):
        print(f"eps=2, E=1: final prefix noise {name}={final_prefix_noise(name, 1, 2.0):.2f}")
    target = final_prefix_noise("sqrt", 1, 2.0) / math.sqrt(100)
    print(f"DP-SGD sigma matching the square root: {target:.3f}  eps={dpsgd_epsilon(0.01, target, 100, 1e-5):.2f}")
    print(f"DP-SGD sigma at eps=2, T=100: {dpsgd_sigma(100, 2.0):.3g}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=FIGURES, help="ignored: this script makes no figures")
    args = parser.parse_args()
    print_table(np.random.default_rng(args.seed))


if __name__ == "__main__":
    main()
