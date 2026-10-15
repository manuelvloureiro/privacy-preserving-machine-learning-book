"""Check the worked solutions to the exercises of the Rényi differential privacy chapter.

    uv run python code/part04-privacy-accounting/solutions_renyi_dp.py

Prints every number quoted in book-exercises/part04-privacy-accounting/01-renyi-dp.md (seed 0; nothing here is random, the seed is
accepted for a uniform interface). Makes no figures; --out is accepted and ignored.

Conventions as in renyi_dp.py: P = M(D), Q = M(D'), orders α > 1, natural logarithms.
"""

import argparse
import math
from pathlib import Path

import numpy as np
from renyi_dp import (
    DELTA,
    ORDERS,
    advanced_composition,
    advanced_composition_tanh,
    gaussian_eps_exact,
    gaussian_mironov_closed_form,
    gaussian_rdp,
    laplace_rdp,
    randomized_response_rdp,
    rdp_to_dp_mironov,
)

# Figures are build output: _build/figures/<part>/, which the build copies into each book.
FIGURES = Path(__file__).resolve().parents[2] / "_build" / "figures" / Path(__file__).parent.name

# --- Exercise 1: randomized response at order 2 ---------------------------------------------


def rr_chi2(p: float) -> float:
    """χ²(P‖Q) = Σ (p - q)²/q for randomized response: (2p - 1)²/(p(1 - p))."""
    return (2 * p - 1) ** 2 / (1 - p) + (2 * p - 1) ** 2 / p


def rr_order2_closed_form(p: float) -> float:
    return math.log(1 + (2 * p - 1) ** 2 / (p * (1 - p)))


# --- Exercise 2: composing Laplace mechanisms three ways ------------------------------------


def laplace_composed_rdp(b: float, k: int, delta: float) -> tuple[float, float]:
    return rdp_to_dp_mironov(ORDERS, [k * laplace_rdp(a, b) for a in ORDERS], delta)


# --- Exercise 3: the moments accountant's tail bound ----------------------------------------


def moments_tail_delta(log_moment: list[float], lambdas: list[float], eps: float) -> tuple[float, float]:
    """Abadi et al.: δ = min_λ exp(α_M(λ) - λε); returns (δ, the best λ)."""
    best = min((lm - lam * eps, lam) for lm, lam in zip(log_moment, lambdas, strict=True))
    return math.exp(best[0]), best[1]


# --- Exercise 4: the weak triangle inequality on three Gaussians ----------------------------


def gaussian_pair_renyi(alpha: float, shift: float) -> float:
    """D_α(N(a, 1)‖N(a + shift, 1)) = α shift²/2."""
    return alpha * shift**2 / 2


def weak_triangle_rhs(alpha: float, p: float, d_pr: float, d_rq: float) -> float:
    """(α - 1/u)/(α - 1) · d_pr + d_rq, where d_pr = D_{uα}(P‖R) and d_rq = D_{v(α - 1/u)}(R‖Q), with u = p."""
    return (alpha - 1 / p) / (alpha - 1) * d_pr + d_rq


# --- The page's numbers ---------------------------------------------------------------------


def print_table() -> None:
    print("[Exercise 1: randomized response at order 2]")
    for p in (0.75, 0.6):
        print(
            f"  p={p:g}: chi2={rr_chi2(p):.4f}  ln(1 + chi2)={math.log(1 + rr_chi2(p)):.4f}  "
            f"closed form={rr_order2_closed_form(p):.4f}  eps(2)={randomized_response_rdp(2.0, p):.4f}"
        )
    p = 0.55
    print(
        f"  p={p:g}: eps(2)={randomized_response_rdp(2.0, p):.5f}  16 gamma^2 with gamma = p - 1/2: {16 * (p - 0.5) ** 2:.5f}  "
        f"KL={randomized_response_rdp(1.0, p):.5f}  8 gamma^2={8 * (p - 0.5) ** 2:.5f}"
    )

    print("[Exercise 2: 100 Laplace mechanisms, b = 10, delta = 1e-5]")
    b, k = 10.0, 100
    eps0 = 1 / b
    rdp_eps, rdp_a = laplace_composed_rdp(b, k, DELTA)
    print(
        f"  basic={k * eps0:.4f}  advanced (textbook)={advanced_composition(eps0, k, DELTA):.4f}  "
        f"advanced (tanh)={advanced_composition_tanh(eps0, k, DELTA):.4f}  RDP={rdp_eps:.4f} at alpha={rdp_a:g}"
    )
    print(
        f"  single step: eps(alpha={rdp_a:g})={laplace_rdp(rdp_a, b):.5f}  eps0^2/2={eps0**2 / 2:.5f}  "
        f"k eps(alpha)={k * laplace_rdp(rdp_a, b):.4f}  ln(1/delta)/(alpha-1)={math.log(1 / DELTA) / (rdp_a - 1):.4f}"
    )
    for kk in (10, 1000):
        r, a = laplace_composed_rdp(b, kk, DELTA)
        print(
            f"  k={kk}: basic={kk * eps0:.4f}  advanced (tanh)={advanced_composition_tanh(eps0, kk, DELTA):.4f}  "
            f"RDP={r:.4f} at alpha={a:g}"
        )

    print("[Exercise 3: the moments accountant's tail bound, Gaussian sigma = 1]")
    lambdas = [a - 1 for a in ORDERS]
    log_moments = [lam * gaussian_rdp(lam + 1, 1.0) for lam in lambdas]
    eps_target, best_a = rdp_to_dp_mironov(ORDERS, [gaussian_rdp(a, 1.0) for a in ORDERS], DELTA)
    delta_back, lam_best = moments_tail_delta(log_moments, lambdas, eps_target)
    print(
        f"  Mironov eps at delta=1e-5: {eps_target:.4f} (alpha={best_a:g})  tail bound at that eps: "
        f"delta={delta_back:.3e} at lambda={lam_best:.1f}"
    )
    for eps in (3.0, 4.0):
        d, lam = moments_tail_delta(log_moments, lambdas, eps)
        print(
            f"  eps={eps:g}: delta={d:.3e} at lambda={lam:.1f}  (continuous: lambda*=eps-1/2={eps - 0.5:.1f}, "
            f"delta=exp(-(eps-1/2)^2/2)={math.exp(-((eps - 0.5) ** 2) / 2):.3e})"
        )

    print("[Exercise 4: weak triangle inequality, P = N(0,1), R = N(0.5,1), Q = N(1.5,1)]")
    alpha = 3.0
    lhs = gaussian_pair_renyi(alpha, 1.5)
    for p in (2.0, 1.5, 3.0):
        q = p / (p - 1)
        d_pr = gaussian_pair_renyi(p * alpha, 0.5)
        d_rq = gaussian_pair_renyi(q * (alpha - 1 / p), 1.0)
        print(
            f"  alpha={alpha:g}, u={p:g}, v={q:g}: D_alpha(P‖Q)={lhs:.4f}  orders {p * alpha:g} and "
            f"{q * (alpha - 1 / p):g}: rhs={weak_triangle_rhs(alpha, p, d_pr, d_rq):.4f}"
        )

    u_star = 1 + 1.0 * (alpha - 1) / (alpha * 0.5)
    v_star = u_star / (u_star - 1)
    rhs_star = weak_triangle_rhs(
        alpha, u_star, gaussian_pair_renyi(u_star * alpha, 0.5), gaussian_pair_renyi(v_star * (alpha - 1 / u_star), 1.0)
    )
    print(
        f"  best u = 1 + b(alpha-1)/(alpha a) = {u_star:.4f}  v={v_star:.4f}  orders {u_star * alpha:g} and "
        f"{v_star * (alpha - 1 / u_star):g}: rhs={rhs_star:.4f}"
    )
    print("[Exercise 5: the edge of the order grid]")
    root = math.sqrt(2 * math.log(1 / DELTA))
    print(f"  alpha* exceeds {ORDERS[-1]:g} once sigma > {(ORDERS[-1] - 1) / root:.1f}")
    for sigma in (100.0, 200.0):
        grid, a = rdp_to_dp_mironov(ORDERS, [gaussian_rdp(x, sigma) for x in ORDERS], DELTA)
        cf, a_star = gaussian_mironov_closed_form(sigma, DELTA)
        exact = gaussian_eps_exact(1 / sigma, DELTA)
        print(
            f"  sigma={sigma:g}: grid eps={grid:.5f} at alpha={a:g}  continuous eps={cf:.5f} at alpha*={a_star:.1f}  "
            f"grid/continuous={grid / cf:.3f}  exact={exact:.5f}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=FIGURES, help="ignored: this script makes no figures")
    parser.parse_args()
    np.random.default_rng(0)  # nothing is random; kept for the uniform interface
    print_table()


if __name__ == "__main__":
    main()
