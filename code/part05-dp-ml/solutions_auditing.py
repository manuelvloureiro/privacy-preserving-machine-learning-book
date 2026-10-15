"""Check the worked solutions to the exercises of the auditing chapter.

    uv run python code/part05-dp-ml/solutions_auditing.py

Prints every number quoted in book-exercises/part05-dp-ml/10-auditing.md (seed 0). Makes no figures;
--out is accepted and ignored.
"""

import argparse
import math
from pathlib import Path

import numpy as np
from auditing import (
    BETA,
    clt_mu,
    eps_from_rates,
    gdp_eps,
    gdp_touching_point,
    one_run_eps,
    one_run_pvalue,
    perfect_attack_eps,
    trials_needed,
)
from dp_sgd import DELTA_RUN, Q_RUN, STEPS_RUN, dpsgd_epsilon

# Figures are build output: _build/figures/<part>/, which the build copies into each book.
FIGURES = Path(__file__).resolve().parents[2] / "_build" / "figures" / Path(__file__).parent.name

# --- Exercise 1: both inequalities ---------------------------------------------------------------------


def both_directions(fpr: float, tpr: float, delta: float) -> tuple[float, float]:
    """(ln((TPR − δ)/FPR), ln((1 − FPR − δ)/(1 − TPR))): the two bounds one test gives."""
    return math.log((tpr - delta) / fpr), math.log((1 - fpr - delta) / (1 - tpr))


# --- Exercise 2: the ceiling of an audit -----------------------------------------------------------------


def ceiling_approx(n: int, beta: float = BETA) -> float:
    """ln(n / ln(2/β)): the rule-of-three approximation of the perfect attack's bound."""
    return math.log(n / math.log(2 / beta))


# --- Exercise 4: what the sum of projections loses -------------------------------------------------------


def efficiency(sigma: float) -> float:
    """μ of the likelihood ratio over μ of the sum of projections: σ √(e^{1/σ²} − 1)."""
    return sigma * math.sqrt(math.expm1(1 / sigma**2))


# --- Exercise 5: how many canaries one run needs -----------------------------------------------------------


def canaries_needed_pure(eps: float, beta: float = BETA) -> int:
    """Smallest r for which r correct guesses out of r reject ε-DP: p^r < β with p = e^ε/(1 + e^ε)."""
    p = 1 / (1 + math.exp(-eps))
    return math.floor(math.log(beta) / math.log(p)) + 1


def canaries_needed(eps: float, delta: float, beta: float = BETA, limit: int = 1 << 20) -> int:
    """Smallest m = r for which m correct guesses out of m reject (ε, δ)-DP; 0 if none up to the limit."""
    m = 1
    while m <= limit and one_run_eps(m, m, m, delta, beta) < eps:
        m *= 2
    if m > limit:
        return 0
    lo, hi = m // 2, m
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if one_run_eps(mid, mid, mid, delta, beta) >= eps:
            hi = mid
        else:
            lo = mid
    return hi


# --- Exercise 6: randomized response meets the bound --------------------------------------------------------


def randomized_response_correct(rng: np.random.Generator, r: int, eps: float, trials: int) -> float:
    """Average number of correct guesses when each guess is ε-randomized response on the true bit."""
    p = 1 / (1 + math.exp(-eps))
    return float(rng.binomial(r, p, size=trials).mean())


def print_table(rng: np.random.Generator) -> None:
    print("== Exercise 1")
    for fpr, tpr in ((0.05, 0.95), (0.001, 0.05), (0.5, 0.999)):
        a, b = both_directions(fpr, tpr, DELTA_RUN)
        print(
            f"FPR={fpr:g}, TPR={tpr:g}: direct={a:.2f}  mirror={b:.2f}  bound={eps_from_rates(fpr, tpr, DELTA_RUN):.2f}"
        )

    print("== Exercise 2")
    for n in (100, 1000, 10_000, 100_000):
        print(f"n={n}: exact={perfect_attack_eps(n, DELTA_RUN):.2f}  approx ln(n/ln 40)={ceiling_approx(n):.2f}")
    eps_claim = dpsgd_epsilon(Q_RUN, 1.0, STEPS_RUN, DELTA_RUN)
    print(
        f"eps={eps_claim:.2f}: exact trials={trials_needed(eps_claim, DELTA_RUN)}  "
        f"approx e^eps ln 40={math.exp(eps_claim) * math.log(2 / BETA):.0f}"
    )

    print("== Exercise 3")
    for mu in (clt_mu(Q_RUN, 1.0, STEPS_RUN), 0.5, 3.0):
        e = gdp_eps(mu, DELTA_RUN)
        a, t = gdp_touching_point(mu, e)
        print(
            f"mu={mu:.4f}: eps={e:.2f}  touching FPR={a:.1e}  TPR={t:.1e}  OUT trials for 10 false positives={10 / a:.1e}"
        )

    print("== Exercise 4")
    for sigma in (0.5, 1.0, 2.0, 4.0):
        print(f"sigma={sigma:g}: mu_LR/mu_sum={efficiency(sigma):.3f}  step factor={efficiency(sigma) ** 2:.2f}")

    print("== Exercise 5")
    print(f"pure DP, eps={eps_claim:.2f}: canaries needed={canaries_needed_pure(eps_claim)}")
    for delta in (1e-6, DELTA_RUN):
        print(f"delta={delta:g}: canaries needed={canaries_needed(eps_claim, delta)}")
    for m in (2000, 2500, 5000):
        print(f"m=r=v={m}, delta=1e-5: eps_lo={one_run_eps(m, m, m, DELTA_RUN):.2f}")

    print("== Exercise 6")
    print(f"p-value of 75/100 at eps=ln 3, delta=0: {one_run_pvalue(100, 100, 75, math.log(3), 0.0):.3f}")
    print(
        f"randomized response, eps=ln 3: mean correct out of 100 = {randomized_response_correct(rng, 100, math.log(3), 100_000):.2f}"
    )
    print(f"p-value of 85/100 at eps=ln 3, delta=0: {one_run_pvalue(100, 100, 85, math.log(3), 0.0):.4f}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=FIGURES, help="ignored: the solutions draw no figures")
    args = parser.parse_args()
    print_table(np.random.default_rng(args.seed))


if __name__ == "__main__":
    main()
