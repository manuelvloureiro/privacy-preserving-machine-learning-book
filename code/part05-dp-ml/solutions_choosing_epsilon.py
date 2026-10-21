"""Check the worked solutions to the exercises of the Choosing ε chapter.

    uv run python code/part05-dp-ml/solutions_choosing_epsilon.py

Prints every number quoted in book-exercises/part05-dp-ml/09-choosing-epsilon.md (seed 0). Makes no
figures; --out is accepted and ignored.
"""

import argparse
import math
from pathlib import Path

import numpy as np
from dp_sgd import DELTA_RUN, Q_RUN, STEPS_RUN, dpsgd_epsilon

# Figures are build output: _build/figures/<part>/, which the build copies into each book.
FIGURES = Path(__file__).resolve().parents[2] / "_build" / "figures" / Path(__file__).parent.name

SIGMA_EPS2 = 2.28  # the noise multiplier of the running example's ε = 2 run (DP-SGD chapter)


def normal_cdf(x: float) -> float:
    return 0.5 * math.erfc(-x / math.sqrt(2))


# --- Exercise 1: what an audit can prove ---------------------------------------------------------


def audit_epsilon(tpr: float, fpr: float, delta: float) -> float:
    """The smallest ε consistent with a measured (FPR, TPR): solve TPR ≤ e^ε FPR + δ for ε."""
    return math.log((tpr - delta) / fpr)


# --- Exercise 2: the advantage of μ-GDP ----------------------------------------------------------


def gdp_advantage(mu: float) -> float:
    return 2 * normal_cdf(mu / 2) - 1


def rr_epsilon_for_advantage(adv: float) -> float:
    """The ε of randomized response, whose advantage is tanh(ε/2), with the given advantage."""
    return math.log((1 + adv) / (1 - adv))


# --- Exercise 3: when can a flag be trusted -------------------------------------------------------


def fpr_for_even_odds(prior: float, eps: float, delta: float) -> float:
    """Largest FPR at which the pair allows precision ≥ 1/2: p(e^ε α + δ) ≥ (1 - p) α. inf if every FPR."""
    slack = (1 - prior) - prior * math.exp(eps)
    return math.inf if slack <= 0 else prior * delta / slack


# --- Exercise 4: a family -----------------------------------------------------------------------


def group_privacy(eps: float, delta: float, k: int) -> tuple[float, float]:
    """(kε, k e^{(k-1)ε} δ) for a group of k records."""
    return k * eps, k * math.exp((k - 1) * eps) * delta


def posterior(prior: float, loss: float) -> float:
    odds = prior / (1 - prior) * math.exp(loss)
    return odds / (1 + odds)


# --- Exercise 5: two ways to add up the Census budget --------------------------------------------


def zcdp_epsilon(rho: float, delta: float) -> float:
    return rho + 2 * math.sqrt(rho * math.log(1 / delta))


# --- Exercise 6: retraining every month -----------------------------------------------------------


def retrain_epsilon(months: int, sigma: float, delta: float) -> float:
    """RDP composition of `months` identical DP-SGD runs on the same data: one run of months × T steps."""
    return dpsgd_epsilon(Q_RUN, sigma, months * STEPS_RUN, delta)


# --- Exercise 7: from actions to users ------------------------------------------------------------


def zcdp_epsilon_tight(rho: float, delta: float) -> float:
    """ε = ρ + √(4ρ ln(√(πρ)/δ)), the conversion of the Facebook URLs release (Bun & Steinke 2016)."""
    return rho + math.sqrt(4 * rho * math.log(math.sqrt(math.pi * rho) / delta))


def print_table(rng: np.random.Generator) -> None:
    del rng  # nothing here is random
    print("== Exercise 1")
    for tpr, fpr in ((0.05, 1e-3), (0.0589, 1e-3), (0.2092, 1e-2), (0.5901, 0.1)):
        print(f"TPR={tpr:g} at FPR={fpr:g}: eps >= {audit_epsilon(tpr, fpr, DELTA_RUN):.3f}")
    print("== Exercise 2")
    for mu in (0.847, 1.511):
        adv = gdp_advantage(mu)
        print(
            f"mu={mu:g}: advantage {adv:.4f}  randomized-response eps with that advantage {rr_epsilon_for_advantage(adv):.3f}"
        )
    print("== Exercise 3")
    for eps in (1.0, 4.0, 8.0):
        print(f"prior 0.01, eps={eps:g}: precision >= 1/2 needs FPR <= {fpr_for_even_odds(0.01, eps, DELTA_RUN):.3g}")
    print("== Exercise 4")
    e3, d3 = group_privacy(1.0, DELTA_RUN, 3)
    print(f"k=3, (1, 1e-5): group ({e3:g}, {d3:.3g})  posterior from prior 0.01: {posterior(0.01, e3):.4f}")
    e3, d3 = group_privacy(4.0, DELTA_RUN, 3)
    print(f"k=3, (4, 1e-5): group ({e3:g}, {d3:.3g})")
    print("== Exercise 5")
    persons, units = zcdp_epsilon(2.56, 1e-10), zcdp_epsilon(0.07, 1e-10)
    print(
        f"persons {persons:.3f} + housing {units:.3f} = {persons + units:.3f};  "
        f"total rho 2.63 at once: {zcdp_epsilon(2.63, 1e-10):.3f}"
    )
    print("== Exercise 6")
    for months in (1, 3, 12):
        print(
            f"{months} runs at sigma=2.28: RDP eps={retrain_epsilon(months, SIGMA_EPS2, DELTA_RUN):.3f}  basic {2 * months:g}"
        )
    print("== Exercise 7")
    sigma = 22.0
    for k in (1, 5, 10):
        rho = k / (2 * sigma**2)
        print(
            f"likes, sigma=22, {k} URLs: rho={rho:.5f}  eps={zcdp_epsilon_tight(rho, DELTA_RUN):.3f}  "
            f"simple eps={zcdp_epsilon(rho, DELTA_RUN):.3f}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=FIGURES, help="ignored: this script draws no figures")
    args = parser.parse_args()
    print_table(np.random.default_rng(args.seed))


if __name__ == "__main__":
    main()
