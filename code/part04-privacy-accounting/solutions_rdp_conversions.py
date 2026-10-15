"""Check the worked solutions to the exercises of the RDP-conversions chapter.

    uv run python code/part04-privacy-accounting/solutions_rdp_conversions.py

Prints every number quoted in book-exercises/part04-privacy-accounting/02-rdp-conversions.md (seed 0; nothing here is random, the
seed is accepted for a uniform interface). Makes no figures; --out is accepted and ignored.

Conventions as in rdp_conversions.py: P = M(D), Q = M(D'), L = ln p/q, α is the Rényi order.
"""

import argparse
import math
from pathlib import Path

import numpy as np
from rdp_conversions import (
    DELTA,
    ORDERS,
    best_constant,
    best_constant_numeric,
    delta_ratio,
    gaussian_rdp,
    gaussian_rdp_curve,
    improved_alpha_gaussian,
    improved_eps,
    mironov_alpha_gaussian,
    mironov_eps,
    mironov_eps_gaussian,
    rdp_to_dp_improved,
    rdp_to_dp_mironov,
    rdp_tradeoff_lower_bound,
    single_order_eps_optimal,
)

# Figures are build output: _build/figures/<part>/, which the build copies into each book.
FIGURES = Path(__file__).resolve().parents[2] / "_build" / "figures" / Path(__file__).parent.name


# --- Exercise 1: the optimal constant -------------------------------------------------------


def alpha_times_ratio(alpha: float) -> float:
    """α · δ_improved/δ_Mironov = (1 - 1/α)^{α - 1}, which tends to 1/e."""
    return alpha * delta_ratio(alpha)


# --- Exercise 2: optimizing the order for a linear curve ------------------------------------


def improved_stationarity_gap(alpha: float, rho: float, delta: float) -> float:
    """ρ(α - 1)² - ln(1/(αδ)): zero at the optimal order of the improved conversion."""
    return rho * (alpha - 1) ** 2 - math.log(1 / (alpha * delta))


# --- Exercise 3: one order's boundary at small type I error --------------------------------


def power_slope_crude(alpha: float, rdp: float, a: float) -> float:
    """e^{(α-1)ε(α)/α} a^{(α-1)/α}: the largest power, dropping the second term of the reverse constraint."""
    return math.exp((alpha - 1) * rdp / alpha) * a ** ((alpha - 1) / alpha)


def power_slope_refined(alpha: float, rdp: float, a: float) -> float:
    """(e^{(α-1)ε(α)} - 1)^{1/α} a^{(α-1)/α}: the same with the second term replaced by its value 1 at b = 1."""
    return math.expm1((alpha - 1) * rdp) ** (1 / alpha) * a ** ((alpha - 1) / alpha)


# --- Exercise 4: composing Laplace in RDP ---------------------------------------------------


def laplace_rdp(alpha: float, eps0: float) -> float:
    """RDP of the Laplace mechanism with scale Δ/ε0 (Mironov 2017); checked by laplace_rdp_numeric."""
    total = alpha / (2 * alpha - 1) * math.exp((alpha - 1) * eps0) + (alpha - 1) / (2 * alpha - 1) * math.exp(
        -alpha * eps0
    )
    return math.log(total) / (alpha - 1)


def laplace_rdp_numeric(alpha: float, eps0: float, n: int = 2_000_001) -> float:
    """The same D_α(Lap(0, 1/ε0) ‖ Lap(1, 1/ε0)) by a Riemann sum of p^α q^{1-α}."""
    b = 1 / eps0
    x = np.linspace(-60 * b, 60 * b, n)
    log_p = -np.abs(x) / b - math.log(2 * b)
    log_q = -np.abs(x - 1) / b - math.log(2 * b)
    integral = float(np.sum(np.exp(alpha * log_p + (1 - alpha) * log_q)) * (x[1] - x[0]))
    return math.log(integral) / (alpha - 1)


def advanced_composition(eps0: float, k: int, delta: float) -> float:
    """ε0 √(2k ln(1/δ)) + k ε0 (e^{ε0} - 1): advanced composition of k ε0-DP mechanisms."""
    return eps0 * math.sqrt(2 * k * math.log(1 / delta)) + k * eps0 * (math.exp(eps0) - 1)


# --- The page's numbers ---------------------------------------------------------------------


def print_table(rng: np.random.Generator) -> None:
    del rng
    delta = DELTA

    print("[Exercise 1: the optimal constant]")
    for alpha, eps in ((2.0, 1.0), (6.0, 4.0)):
        print(
            f"  alpha={alpha:g} eps={eps:g}: c* formula={best_constant(alpha, eps):.6e}  grid={best_constant_numeric(alpha, eps):.6e}  "
            f"z*={eps - math.log(1 - 1 / alpha):.4f}"
        )
    for alpha in (2.0, 10.0, 100.0, 1000.0):
        print(f"  alpha={alpha:g}: ratio={delta_ratio(alpha):.6f}  alpha*ratio={alpha_times_ratio(alpha):.4f}")
    print(f"  1/e={1 / math.e:.4f}")

    print("[Exercise 2: the best order for eps(alpha) = rho alpha]")
    for rho in (0.5, 0.005):
        for d in (1e-5, 1e-10):
            a_m = mironov_alpha_gaussian(rho, d)
            a_i = improved_alpha_gaussian(rho, d)
            print(
                f"  rho={rho:g} delta={d:g}: Mironov alpha*={a_m:.3f} eps*={mironov_eps_gaussian(rho, d):.3f} "
                f"(direct {mironov_eps(a_m, rho * a_m, d):.3f})  improved alpha*={a_i:.3f} "
                f"eps*={improved_eps(a_i, rho * a_i, d):.3f}  gap={improved_stationarity_gap(a_i, rho, d):.1e}"
            )

    print("[Exercise 3: one order's boundary at small a, Gaussian mu = 1]")
    for alpha in (2.0, 8.0):
        tau = gaussian_rdp(alpha, 1.0)
        for a in (1e-3, 1e-6):
            exact = 1 - rdp_tradeoff_lower_bound([alpha], [tau], a)
            crude, refined = power_slope_crude(alpha, tau, a), power_slope_refined(alpha, tau, a)
            print(
                f"  alpha={alpha:g} a={a:g}: exact={exact:.4e}  crude={crude:.4e}  refined={refined:.4e}  "
                f"crude/exact={crude / exact:.4f}  refined/exact={refined / exact:.4f}"
            )
        print(
            f"  alpha={alpha:g}: (1 - e^(-(alpha-1)eps(alpha)))^(1/alpha)={(-math.expm1(-(alpha - 1) * tau)) ** (1 / alpha):.4f}"
        )

    print("[Exercise 4: k = 100 Laplace mechanisms with eps0 = 0.1, delta = 1e-5]")
    eps0, k = 0.1, 100
    for alpha in (2.0, 20.0):
        print(
            f"  alpha={alpha:g}: Laplace RDP formula={laplace_rdp(alpha, eps0):.6f}  numeric={laplace_rdp_numeric(alpha, eps0):.6f}  "
            f"Gaussian-like alpha eps0^2/2={alpha * eps0 * eps0 / 2:.6f}"
        )
    curve = [k * laplace_rdp(a, eps0) for a in ORDERS]
    e_imp, a_imp = rdp_to_dp_improved(ORDERS, curve, delta)
    e_mir, a_mir = rdp_to_dp_mironov(ORDERS, curve, delta)
    bound_curve = [k * min(eps0, a * eps0 * eps0 / 2) for a in ORDERS]
    e_bound, _ = rdp_to_dp_improved(ORDERS, bound_curve, delta)
    print(
        f"  basic={k * eps0:.3f}  advanced={advanced_composition(eps0, k, delta):.3f}  "
        f"RDP improved={e_imp:.3f} (alpha {a_imp:g})  RDP Mironov={e_mir:.3f} (alpha {a_mir:g})  "
        f"via min(eps0, alpha eps0^2/2)={e_bound:.3f}  gain of the exact curve={e_bound - e_imp:.3f}"
    )

    print("[Exercise 5: grid resolution]")
    for sigma, k in ((1.0, 10_000), (10.0, 1), (100.0, 1), (300.0, 1)):
        mu = math.sqrt(k) / sigma
        rho = mu * mu / 2
        a_cont = improved_alpha_gaussian(rho, delta)
        e_cont = improved_eps(a_cont, gaussian_rdp(a_cont, mu), delta)
        e_grid, a_grid = rdp_to_dp_improved(ORDERS, gaussian_rdp_curve(mu), delta)
        print(
            f"  sigma={sigma:g} k={k}: rho={rho:.4g}  continuous alpha*={a_cont:.3f} eps={e_cont:.4f}  "
            f"grid alpha={a_grid:g} eps={e_grid:.4f}  excess={100 * (e_grid / e_cont - 1):.2f}%"
        )

    print("[Exercise 6: when the improved conversion is optimal for one order, mu = 1]")
    for alpha in (1.5, 2.0, 4.0, 8.0):
        tau = gaussian_rdp(alpha, 1.0)
        e_imp = improved_eps(alpha, tau, delta)
        print(
            f"  alpha={alpha:g}: improved={e_imp:.4f}  optimal={single_order_eps_optimal(alpha, tau, delta):.4f}  "
            f"c*={best_constant(alpha, e_imp):.3e}  c*/delta={best_constant(alpha, e_imp) / delta:.3f}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=FIGURES, help="ignored: this script makes no figures")
    args = parser.parse_args()
    seed: int = args.seed
    print_table(np.random.default_rng(seed))


if __name__ == "__main__":
    main()
