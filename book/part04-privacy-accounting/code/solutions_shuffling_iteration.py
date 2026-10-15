"""Check the worked solutions to the exercises of the shuffling and iteration chapter.

    uv run python book/part04-privacy-accounting/code/solutions_shuffling_iteration.py

Prints every number quoted in solutions/07-shuffling-iteration.md (seed 0; nothing here is random,
the seed is accepted for a uniform interface). Makes no figures; --out is accepted and ignored.

Conventions as in shuffling_iteration.py: replace-one neighbours, θ = e^ε0/(1 + e^ε0), and the
one-dimensional iteration w_k = c w_{k-1} + b_k + N(0, σ²).
"""

import argparse
import math
from collections.abc import Sequence
from pathlib import Path

import numpy as np
from shuffling_iteration import (
    clone_split,
    gaussian_rdp,
    gdp_epsilon,
    hidden_state_limit_mu,
    noisy_gd_last_iterate_mu,
    plateau_rdp_bound,
    rr_truth_prob,
    shuffled_rr_delta,
    shuffled_rr_epsilon,
    shuffled_rr_log_pmfs,
)

FIGURES = Path(__file__).resolve().parent.parent / "figures"

# --- Exercise 2: how many users? -------------------------------------------------------------------


def mu_for_epsilon(eps: float, delta: float) -> float:
    """The μ whose GDP curve has δ(eps) = delta, by bisection (ε at fixed δ grows with μ)."""
    lo, hi = 1e-6, 10.0
    for _ in range(100):
        mid = (lo + hi) / 2
        if gdp_epsilon(mid, delta) > eps:
            hi = mid
        else:
            lo = mid
    return lo


def clt_users(eps0: float, eps: float, delta: float) -> int:
    """Users needed by the normal approximation: 2 sinh(ε0/2)/√n = μ*, rounded up."""
    return math.ceil((2 * math.sinh(eps0 / 2) / mu_for_epsilon(eps, delta)) ** 2)


def exact_users(eps0: float, eps: float, delta: float, lo: int, hi: int) -> int:
    """Smallest n in [lo, hi] whose exact shuffled ε is at most eps (bisection; ε falls with n)."""
    while lo < hi:
        mid = (lo + hi) // 2
        if shuffled_rr_epsilon(eps0, mid, delta) <= eps:
            hi = mid
        else:
            lo = mid + 1
    return lo


# --- Exercise 3: clones of a three-valued report ---------------------------------------------------


def krr(eps0: float, k: int, value: int) -> list[float]:
    """k-ary randomized response: the true value w.p. e^ε0/(e^ε0 + k - 1), each other value w.p. 1/(e^ε0 + k - 1)."""
    z = math.exp(eps0) + k - 1
    return [math.exp(eps0) / z if y == value else 1 / z for y in range(k)]


def largest_clone_probability(r_x: Sequence[float], r_x0: Sequence[float], r_x1: Sequence[float]) -> float:
    """The largest p with R(x) ≥ (p/2)(R(x0) + R(x1)) pointwise: min_y 2 R(x)(y) / (R(x0)(y) + R(x1)(y))."""
    return min(2 * a / (b + c) for a, b, c in zip(r_x, r_x0, r_x1, strict=True))


# --- Exercise 4: shift schedules ------------------------------------------------------------------


def schedule_cost(alpha: float, sigma: float, payments: Sequence[float]) -> float:
    return sum(gaussian_rdp(alpha, a, sigma) for a in payments)


def remaining_shifts(shifts: Sequence[float], payments: Sequence[float]) -> list[float]:
    """z_t = Σ_{i≤t} (s_i - a_i); a schedule is feasible when every z_t ≥ 0 and the last is 0."""
    out: list[float] = []
    z = 0.0
    for s, a in zip(shifts, payments, strict=True):
        z += s - a
        out.append(z)
    return out


# --- Exercise 5: the hidden-state limit -----------------------------------------------------------


def steps_to_fraction(contraction: float, fraction: float) -> int:
    """Smallest T with μ_T ≥ fraction × lim μ (full batch, s/σ = 1)."""
    limit = hidden_state_limit_mu(contraction, 1.0, 1.0)
    t = 1
    while noisy_gd_last_iterate_mu(contraction, t, 1.0, 1.0, 1) < fraction * limit:
        t += 1
    return t


def print_table(rng: np.random.Generator) -> None:
    del rng  # every number here is exact

    print("[Exercise 1: two users, eps0 = ln 3]")
    eps0 = math.log(3)
    log_p, log_q = shuffled_rr_log_pmfs(eps0, 2)
    p, q = np.exp(log_p), np.exp(log_q)
    print(
        "  P x 16 = " + "  ".join(f"{16 * v:.0f}" for v in p) + "   Q x 16 = " + "  ".join(f"{16 * v:.0f}" for v in q)
    )
    print(f"  e^0.5={math.exp(0.5):.4f}")
    forward = float(np.maximum(p - math.exp(0.5) * q, 0).sum())
    backward = float(np.maximum(q - math.exp(0.5) * p, 0).sum())
    print(f"  E(P||Q)={forward:.4f}  E(Q||P)={backward:.4f}  delta(0.5)={shuffled_rr_delta(0.5, eps0, 2):.4f}")
    for n in (2, 10, 100):
        lp, lq = shuffled_rr_log_pmfs(eps0, n)
        print(f"  n={n}: ln P(0)/Q(0)={float(lp[0] - lq[0]):.4f}  P(0)={math.exp(float(lp[0])):.3e}")

    print("[Exercise 2: users needed for eps = 0.1 at delta = 1e-6, eps0 = 2]")
    print(f"  theta={rr_truth_prob(2.0):.4f}")
    mu_star = mu_for_epsilon(0.1, 1e-6)
    n_clt = clt_users(2.0, 0.1, 1e-6)
    print(
        f"  mu*={mu_star:.6f}  2 sinh(1)={2 * math.sinh(1):.4f}  (2 sinh(1)/mu*)^2={(2 * math.sinh(1) / mu_star) ** 2:.1f}  "
        f"normal approximation n={n_clt}"
    )
    print(f"  exact eps at that n={shuffled_rr_epsilon(2.0, n_clt, 1e-6):.4f}")
    n_exact = exact_users(2.0, 0.1, 1e-6, n_clt // 2, 2 * n_clt)
    print(
        f"  smallest n with exact eps <= 0.1: {n_exact}  (exact eps there {shuffled_rr_epsilon(2.0, n_exact, 1e-6):.4f})  "
        f"ratio {n_exact / n_clt:.3f}"
    )

    print("[Exercise 3: 3-ary RR, eps0 = ln 4, x = value 3, x0 = value 1, x1 = value 2]")
    e3 = math.log(4)
    r1, r2, r3 = krr(e3, 3, 0), krr(e3, 3, 1), krr(e3, 3, 2)
    clones, left = clone_split(r3, r1, r2, e3)
    print(f"  R(3) x 48 = {[round(48 * v, 4) for v in r3]}  clone part x 48 = {[round(48 * v, 4) for v in clones]}")
    print(f"  LO(3) x 12 = {[round(12 * v, 4) for v in left]}")
    print(
        f"  largest clone probability: x=3: {largest_clone_probability(r3, r1, r2):.4f}  "
        f"x=1: {largest_clone_probability(r1, r1, r2):.4f}  e^-eps0={math.exp(-e3):.4f}"
    )
    theta = rr_truth_prob(math.log(3))
    print(
        f"  binary RR, eps0 = ln 3: largest clone probability "
        f"{largest_clone_probability([theta, 1 - theta], [theta, 1 - theta], [1 - theta, theta]):.4f}"
    )

    print("[Exercise 4: shift schedules, sigma = 1, alpha = 2]")
    print(
        f"  one use, z = 1: (0.8, 0.2) costs {schedule_cost(2.0, 1.0, [0.8, 0.2]):.4f}  "
        f"(0.5, 0.5) costs {schedule_cost(2.0, 1.0, [0.5, 0.5]):.4f}"
    )
    shifts = [1.0, 0.0, 0.0, 1.0]
    even = [0.5, 0.5, 0.5, 0.5]
    best = [1 / 3, 1 / 3, 1 / 3, 1.0]
    print(f"  two uses at steps 1 and 4: even schedule z = {[round(z, 4) for z in remaining_shifts(shifts, even)]}")
    print(
        f"  best feasible (1/3, 1/3, 1/3, 1): z = {[round(z, 4) for z in remaining_shifts(shifts, best)]}  "
        f"cost={schedule_cost(2.0, 1.0, best):.4f}"
    )
    print(f"  exact for translations: means differ by 2, variance 4: D_2={gaussian_rdp(2.0, 2.0, 2.0):.4f}")

    print("[Exercise 5: steps to reach 99% of the limit, s/sigma = 1]")
    for c in (0.9, 0.99):
        print(
            f"  c={c:g}: limit={hidden_state_limit_mu(c, 1.0, 1.0):.4f}  steps to 99%={steps_to_fraction(c, 0.99)}  "
            f"times (1 - c)={steps_to_fraction(c, 0.99) * (1 - c):.2f}"
        )

    print(f"  ln 100={math.log(100):.2f}")

    print("[Exercise 6: plateau, d_K = 1, s = 0.01, sigma = 0.1, alpha = 2]")
    print(
        f"  m*=d_K/s={1 / 0.01:.0f}  plateau 2 alpha d_K s / sigma^2={2 * 2 * 1 * 0.01 / 0.1**2:.4f}  "
        f"starts at T=4d_K/s={4 / 0.01:.0f}"
    )
    print(
        f"  bound at T=399: {plateau_rdp_bound(2.0, 399, 0.01, 0.1, 1.0):.4f}  "
        f"at T=400: {plateau_rdp_bound(2.0, 400, 0.01, 0.1, 1.0):.4f}  "
        f"at T=10000: {plateau_rdp_bound(2.0, 10_000, 0.01, 0.1, 1.0):.4f}"
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
