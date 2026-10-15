"""Check the worked solutions to the exercises of the concentrated differential privacy chapter.

    uv run python book/part04-privacy-accounting/code/solutions_concentrated_dp.py

Prints every number quoted in solutions/03-concentrated-dp.md (seed 0; nothing here is random,
the seed is accepted for a uniform interface). Makes no figures; --out is accepted and ignored.
"""

import argparse
import math
from pathlib import Path

import numpy as np
from concentrated_dp import (
    DELTA,
    FINE_ALPHAS,
    K_RUN,
    Q_RUN,
    SIGMA_RUN,
    approx_zcdp_to_dp,
    best_alpha_simple,
    binary_renyi,
    compose_failure,
    compose_zcdp,
    gaussian_eps,
    gaussian_zcdp_rho,
    logsumexp,
    pure_to_zcdp,
    renyi_triangle_rhs,
    rho_for_target,
    rr_kl,
    running_example_curve,
    smallest_zcdp_rho,
    subsampled_gaussian_rdp_int,
    subsampled_lower_bound,
    tcdp_rho,
    tcdp_to_dp,
    zcdp_gaussian_mu,
    zcdp_to_dp_improved,
    zcdp_to_dp_simple,
)

FIGURES = Path(__file__).resolve().parent.parent / "figures"

# --- Exercise 1: Laplace in a zCDP budget -----------------------------------------------------


def laplace_rdp(alpha: float, t: float) -> float:
    """D_α(Lap(0, 1)‖Lap(t, 1)) = ln(α/(2α-1) e^{(α-1)t} + (α-1)/(2α-1) e^{-αt})/(α-1), in log space."""
    return logsumexp(
        [
            math.log(alpha / (2 * alpha - 1)) + (alpha - 1) * t,
            math.log((alpha - 1) / (2 * alpha - 1)) - alpha * t,
        ]
    ) / (alpha - 1)


def laplace_kl(t: float) -> float:
    return t + math.exp(-t) - 1


# --- Exercise 2: calibrating sigma for k steps --------------------------------------------------


def sigma_for_target(eps: float, delta: float, k: int) -> float:
    """Noise multiplier that makes k Gaussian steps (ε, δ)-DP by the simple conversion: σ = √k/√(2ρ)."""
    return math.sqrt(k) / math.sqrt(2 * rho_for_target(eps, delta))


# --- Exercise 5: the subsampled Gaussian at order 2 -------------------------------------------------


def subsampled_order2(q: float, sigma: float) -> float:
    """ε(2) = ln(1 + q²(e^{1/σ²} - 1)) for the Poisson-subsampled Gaussian."""
    return math.log1p(q * q * math.expm1(1 / sigma**2))


def print_table(rng: np.random.Generator) -> None:
    del rng  # nothing random

    print("[Exercise 1: Laplace at eps=1]")
    t = 1.0
    grid = FINE_ALPHAS[:15001]
    smallest = smallest_zcdp_rho(grid, [laplace_rdp(a, t) for a in grid])
    print(
        f"  smallest rho={smallest:.4f}  KL=t+e^-t-1={laplace_kl(t):.4f}  eps^2/2={pure_to_zcdp(t):.4f}  "
        f"randomized response={rr_kl(t):.4f}"
    )
    print(
        f"  overcharge of eps^2/2 for Laplace={pure_to_zcdp(t) / smallest:.2f}  "
        f"overcharge of the sharp eps tanh(eps/2)={rr_kl(t) / smallest:.2f}"
    )
    for a in (1.001, 2.0, 10.0, 100.0):
        print(f"  alpha={a:g}: D_alpha={laplace_rdp(a, t):.4f}  D_alpha/alpha={laplace_rdp(a, t) / a:.4f}")

    print("[Exercise 2: calibrating sigma]")
    rho = rho_for_target(1.0, DELTA)
    sigma100 = sigma_for_target(1.0, DELTA, 100)
    print(
        f"  ln(1/delta)={math.log(1 / DELTA):.4f}  rho={rho:.5f}  per step={rho / 100:.4e}  sigma(k=100)={sigma100:.2f}"
    )
    total = compose_zcdp([gaussian_zcdp_rho(sigma100)] * 100)
    print(
        f"  check: 100 steps at that sigma: rho={total:.5f}  simple eps={zcdp_to_dp_simple(total, DELTA):.4f}  "
        f"improved eps={zcdp_to_dp_improved(total, DELTA):.4f}  exact Gaussian eps={gaussian_eps(zcdp_gaussian_mu(total), DELTA):.4f}"
    )

    print("[Exercise 3: triangle-like inequality, k=2, alpha=2]")
    p, q = [0.8, 0.2], [0.2, 0.8]
    for r_one in (0.4, 0.5):
        r = [1 - r_one, r_one]
        lhs = binary_renyi(p, q, 2.0)
        d_pr, d_rq = binary_renyi(p, r, 3.0), binary_renyi(r, q, 4.0)
        print(
            f"  R=Bern({r_one:g}): D_2(P||Q)={lhs:.4f}  D_3(P||R)={d_pr:.4f}  D_4(R||Q)={d_rq:.4f}  "
            f"rhs=(4/3)D_3+D_4={renyi_triangle_rhs(2.0, 2, d_pr, d_rq):.4f}"
        )
    logit = [math.log(x / (1 - x)) for x in (0.2, 0.4, 0.5, 0.8)]
    print("  log-odds of 0.2, 0.4, 0.5, 0.8: " + "  ".join(f"{v:.4f}" for v in logit))

    print("[Exercise 4: a mixed pipeline]")
    steps, eps0, delta0 = 20, 0.25, 1e-8
    rho_g = gaussian_zcdp_rho(5.0)
    rho_total = compose_zcdp([pure_to_zcdp(eps0)] * steps + [rho_g] * 50)
    delta_a = compose_failure([delta0] * steps)
    print(
        f"  per stability step rho={pure_to_zcdp(eps0):.5f}  per Gaussian rho={rho_g:.3f}  total rho={rho_total:.4f}  "
        f"delta_a={delta_a:.4e}"
    )
    print(f"  eps at total delta 1e-5 (improved)={approx_zcdp_to_dp(rho_total, delta_a, DELTA):.3f}")
    eps_g = zcdp_to_dp_improved(compose_zcdp([rho_g] * 50), DELTA - steps * delta0)
    print(
        f"  without approximate zCDP: Gaussians alone eps={eps_g:.3f}  + basic {steps * eps0:g} = {eps_g + steps * eps0:.3f}"
    )

    print("[Exercise 5: subsampled Gaussian at order 2]")
    print(
        f"  closed form={subsampled_order2(Q_RUN, SIGMA_RUN):.4e}  binomial sum={subsampled_gaussian_rdp_int(2, Q_RUN, SIGMA_RUN):.4e}"
    )
    threshold = 1 + 2 * SIGMA_RUN**2 * math.log(1 / Q_RUN)
    print(
        f"  threshold 1 + 2 sigma^2 ln(1/q)={threshold:.2f}  lower bound at 10: {subsampled_lower_bound(10, Q_RUN, SIGMA_RUN):.4f}"
        f"  at 11: {subsampled_lower_bound(11, Q_RUN, SIGMA_RUN):.4f}"
    )

    print("[Exercise 6: tCDP conversion, running example]")
    ints, curve = running_example_curve()
    log_inv = math.log(1 / DELTA)
    for omega in (4.0, 8.0):
        rho_k = K_RUN * tcdp_rho(ints, curve, omega)
        branch = "first" if log_inv <= (omega - 1) ** 2 * rho_k else "second"
        print(
            f"  omega={omega:g}: rho={rho_k:.4f}  (omega-1)^2 rho={(omega - 1) ** 2 * rho_k:.3f}  "
            f"alpha*={best_alpha_simple(rho_k, DELTA):.3f}  branch={branch}  eps={tcdp_to_dp(rho_k, omega, DELTA):.3f}"
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
