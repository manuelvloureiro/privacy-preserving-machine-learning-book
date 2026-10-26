"""Check the numbers in the course's local differential privacy chapter and draw its figure.

    uv run python code/part05-deployments/local_dp.py
    uv run python code/part05-deployments/local_dp.py --seed 1 --out /tmp/figures

Prints the numbers quoted in book-course/part05-deployments/01-local-dp.md, then saves the
chapter's figure. Central model: the mean of n bits plus Laplace noise of scale 1/(εn). Local
model: randomized response with truth probability e^ε/(1 + e^ε), debiased by the analyst. Then
the parameters of the deployed protocols: RAPPOR's two layers of randomized response, the
fragment search space of a word-discovery protocol, and the one-bit mean protocol.
"""

import argparse
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # for code/shared
from shared.figures import configure
from shared.stats import binomial_pmf

configure("local_dp")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from numpy.typing import NDArray

# Figures are build output: _build/figures/<part>/, which the build copies into each book.
FIGURES = Path(__file__).resolve().parents[2] / "_build" / "figures" / Path(__file__).parent.name


# --- Counting bits: central and local ----------------------------------------------------------


def central_error_bound(eps: float, n: int, beta: float) -> float:
    """ln(1/β)/(εn): the Laplace mechanism's error on the mean, exceeded with probability β."""
    return math.log(1.0 / beta) / (eps * n)


def local_error_bound(eps: float, n: int, beta: float) -> float:
    """coth(ε/2) √(ln(2/β)/(2n)): randomized response's error, exceeded with probability at most β."""
    return math.sqrt(math.log(2.0 / beta) / (2.0 * n)) / math.tanh(eps / 2.0)


def central_sample_size(eps: float, alpha: float, beta: float) -> int:
    """Smallest n with ln(1/β)/(εn) ≤ α."""
    return math.ceil(math.log(1.0 / beta) / (eps * alpha))


def local_sample_size(eps: float, alpha: float, beta: float) -> int:
    """Smallest n with coth(ε/2) √(ln(2/β)/(2n)) ≤ α."""
    return math.ceil(math.log(2.0 / beta) / (2.0 * alpha**2 * math.tanh(eps / 2.0) ** 2))


def lower_bound_sample_size(eps: float, alpha: float) -> float:
    """ln 2/(4(e^ε - 1)²α²): fewer people cannot tell Bern(1/2 - α) from Bern(1/2 + α) locally."""
    return math.log(2.0) / (4.0 * math.expm1(eps) ** 2 * alpha**2)


def simulate_central(rng: np.random.Generator, eps: float, n: int, reps: int) -> NDArray[np.float64]:
    """|μ̂ - μ| for the Laplace mechanism: the error does not depend on the data."""
    return np.abs(rng.laplace(0.0, 1.0 / (eps * n), size=reps))


def simulate_local(rng: np.random.Generator, eps: float, n: int, mu: float, reps: int) -> NDArray[np.float64]:
    """|μ̂ - μ| for debiased randomized response, on a dataset with round(μn) ones."""
    ones = round(mu * n)
    truth = math.exp(eps) / (1.0 + math.exp(eps))
    reported = rng.binomial(ones, truth, size=reps) + rng.binomial(n - ones, 1.0 - truth, size=reps)
    estimate = ((math.exp(eps) + 1.0) * reported / n - 1.0) / math.expm1(eps)
    return np.abs(estimate - ones / n)


# --- Repeated randomized response --------------------------------------------------------------


# --- RAPPOR ------------------------------------------------------------------------------------


def rappor_permanent_eps(f: float, h: int) -> float:
    """ε∞ = 2h ln((1 - f/2)/(f/2)) for the memoized Bloom filter."""
    return 2.0 * h * math.log((1.0 - f / 2.0) / (f / 2.0))


def rappor_report_probs(f: float, p: float, q: float) -> tuple[float, float]:
    """(q*, p*): the chance that a reported bit is 1 when the Bloom filter bit is 1, resp. 0."""
    q_star = (1.0 - f / 2.0) * q + (f / 2.0) * p
    p_star = (f / 2.0) * q + (1.0 - f / 2.0) * p
    return q_star, p_star


def rappor_report_eps(f: float, p: float, q: float, h: int) -> float:
    """ε₁ = h ln(q*(1 - p*)/(p*(1 - q*))) for one report."""
    q_star, p_star = rappor_report_probs(f, p, q)
    return h * math.log(q_star * (1.0 - p_star) / (p_star * (1.0 - q_star)))


# --- Numbers in one bit ------------------------------------------------------------------------


def laplace_local_sd(eps: float, m: float, n: int) -> float:
    """Standard deviation √2 m/(ε√n) of the mean of n local Laplace reports."""
    return math.sqrt(2.0) * m / (eps * math.sqrt(n))


def one_bit_sd_bound(eps: float, m: float, n: int) -> float:
    """m coth(ε/2)/(2√n), the largest standard deviation of the one-bit estimator."""
    return m / (2.0 * math.tanh(eps / 2.0) * math.sqrt(n))


def memo_bits_differ(eps: float) -> float:
    """Pr[A(0) ≠ A(m)] = (1 + e^{2ε})/(1 + e^ε)²."""
    return (1.0 + math.exp(2.0 * eps)) / (1.0 + math.exp(eps)) ** 2


# --- Table and figure ------------------------------------------------------------------------


def print_table(rng: np.random.Generator) -> None:
    print("== Counting bits")
    eps, alpha, beta = 1.0, 0.01, 0.05
    print(f"eps={eps:g} alpha={alpha:g} beta={beta:g}: central n={central_sample_size(eps, alpha, beta)}")
    print(f"eps={eps:g} alpha={alpha:g} beta={beta:g}: local n={local_sample_size(eps, alpha, beta)}")
    print(f"ratio local/central={local_sample_size(eps, alpha, beta) / central_sample_size(eps, alpha, beta):.0f}")
    for e in (0.1, 1.0):
        print(f"eps={e:g} alpha={alpha:g}: lower bound n={lower_bound_sample_size(e, alpha):.0f}")
    n = 10_000
    print(
        f"n={n} eps={eps:g} beta={beta:g}: central bound={central_error_bound(eps, n, beta):.5f}  "
        f"local bound={local_error_bound(eps, n, beta):.4f}"
    )
    sim_c = float(np.quantile(simulate_central(rng, eps, n, 2000), 0.95))
    sim_l = float(np.quantile(simulate_local(rng, eps, n, 0.3, 2000), 0.95))
    print(f"n={n}: simulated 95th percentile central={sim_c:.5f}  local={sim_l:.4f}")

    print("== Averaging attack")
    days, yes, truth = 50, 40, 0.8
    p_no = binomial_pmf(yes, days, 1.0 - truth)
    p_yes = binomial_pmf(yes, days, truth)
    print(f"eps=ln 4={math.log(4):.4f}: {yes} yes of {days} given no={p_no:.2e}  given yes={p_yes:.2e}")
    print(f"likelihood ratio=4^{2 * yes - days}={p_yes / p_no:.3e}  composition eps={days * math.log(4):.1f}")

    print("== RAPPOR")
    f, p, q, h = 0.5, 0.5, 0.75, 2
    q_star, p_star = rappor_report_probs(f, p, q)
    print(f"f={f:g} h={h}: eps_inf={rappor_permanent_eps(f, h):.4f}")
    print(f"p={p:g} q={q:g}: q*={q_star:.4f}  p*={p_star:.4f}  eps_1={rappor_report_eps(f, p, q, h):.4f}")

    print("== Word discovery")
    print(f"26^8={26**8:,}  26^2*256={26**2 * 256:,}")

    print("== One bit")
    m, n = 1.0, 10_000
    print(
        f"eps={eps:g} m={m:g} n={n}: Laplace sd={laplace_local_sd(eps, m, n):.5f}  "
        f"one-bit sd bound={one_bit_sd_bound(eps, m, n):.5f}  "
        f"ratio={one_bit_sd_bound(eps, m, n) / laplace_local_sd(eps, m, n):.3f}"
    )
    print(f"eps={eps:g}: Pr[A(0) != A(m)]={memo_bits_differ(eps):.4f}")


def plot_gap(rng: np.random.Generator) -> Figure:
    eps, beta = 1.0, 0.05
    ns = np.unique(np.logspace(2, 6, 13).astype(int))
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    sim_c = [float(np.quantile(simulate_central(rng, eps, int(n), 2000), 0.95)) for n in ns]
    sim_l = [float(np.quantile(simulate_local(rng, eps, int(n), 0.3, 2000), 0.95)) for n in ns]
    ax.plot(ns, sim_l, "-o", color="C0", markersize=3, label="local, simulated")
    ax.plot(ns, [local_error_bound(eps, int(n), beta) for n in ns], "--", color="C0", label="local, bound")
    ax.plot(ns, sim_c, "-o", color="C1", markersize=3, label="central, simulated")
    ax.plot(ns, [central_error_bound(eps, int(n), beta) for n in ns], "--", color="C1", label="central, bound")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"number of people $n$")
    ax.set_ylabel("absolute error of the mean (95th percentile)")
    ax.legend(fontsize=8)
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    return fig


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=FIGURES)
    args = parser.parse_args()
    print_table(np.random.default_rng(args.seed))
    args.out.mkdir(parents=True, exist_ok=True)
    fig = plot_gap(np.random.default_rng(args.seed + 1))
    fig.savefig(args.out / "ldp-gap.svg", metadata={"Date": None})
    plt.close(fig)


if __name__ == "__main__":
    main()
