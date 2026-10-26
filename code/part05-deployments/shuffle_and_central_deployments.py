"""Check the numbers in the course's shuffle-model and central-deployments chapter and draw its figure.

    uv run python code/part05-deployments/shuffle_and_central_deployments.py
    uv run python code/part05-deployments/shuffle_and_central_deployments.py --seed 1 --out /tmp/figures

Prints the numbers quoted in book-course/part05-deployments/02-shuffle-and-central-deployments.md,
then saves the chapter's figure. Shuffled randomized response: each person keeps their bit with
probability 1 - q and otherwise sends a fair coin, with q = 16 ln(4/δ)/(τ²(n - 1)) and
τ = tanh(ε/2); the shuffler releases only the number of ones. Then the composition of bounded-range
mechanisms and a budget under zero-concentrated differential privacy.
"""

import argparse
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # for code/shared
from shared.figures import configure

configure("shuffle_and_central_deployments")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from numpy.typing import NDArray

# Figures are build output: _build/figures/<part>/, which the build copies into each book.
FIGURES = Path(__file__).resolve().parents[2] / "_build" / "figures" / Path(__file__).parent.name


# --- Shuffled randomized response --------------------------------------------------------------


def shuffle_q(eps: float, delta: float, n: int) -> float:
    """q = 16 ln(4/δ)/(τ²(n - 1)), the chance that a person sends a fair coin instead of their bit."""
    tau = math.tanh(eps / 2.0)
    return 16.0 * math.log(4.0 / delta) / (tau**2 * (n - 1))


def min_people(eps: float, delta: float) -> int:
    """Smallest n with q ≤ 1, that is, n - 1 ≥ 16 ln(4/δ)/τ²."""
    tau = math.tanh(eps / 2.0)
    return 1 + math.ceil(16.0 * math.log(4.0 / delta) / tau**2)


def local_eps_of_message(q: float) -> float:
    """ε₀ = ln((1 - q/2)/(q/2)): the privacy of one message before shuffling."""
    return math.log((1.0 - q / 2.0) / (q / 2.0))


def shuffle_sd_bound(eps: float, delta: float, n: int) -> float:
    """√(nq/2)/(1 - q), the largest standard deviation of the shuffled count estimate."""
    q = shuffle_q(eps, delta, n)
    return math.sqrt(n * q / 2.0) / (1.0 - q)


def local_count_sd(eps: float, n: int) -> float:
    """√n e^{ε/2}/(e^ε - 1), the standard deviation of the randomized-response count."""
    return math.sqrt(n) * math.exp(eps / 2.0) / math.expm1(eps)


def central_count_sd(eps: float) -> float:
    """√2/ε, the standard deviation of Laplace noise of scale 1/ε."""
    return math.sqrt(2.0) / eps


def simulate_shuffle(
    rng: np.random.Generator, eps: float, delta: float, n: int, ones: int, reps: int
) -> NDArray[np.float64]:
    """|estimated count - true count| for shuffled randomized response."""
    q = shuffle_q(eps, delta, n)
    reported = rng.binomial(ones, 1.0 - q / 2.0, size=reps) + rng.binomial(n - ones, q / 2.0, size=reps)
    estimate = (reported - n * q / 2.0) / (1.0 - q)
    return np.abs(estimate - ones)


def simulate_local(rng: np.random.Generator, eps: float, n: int, ones: int, reps: int) -> NDArray[np.float64]:
    """|estimated count - true count| for local randomized response."""
    truth = math.exp(eps) / (1.0 + math.exp(eps))
    reported = rng.binomial(ones, truth, size=reps) + rng.binomial(n - ones, 1.0 - truth, size=reps)
    estimate = ((math.exp(eps) + 1.0) * reported - n) / math.expm1(eps)
    return np.abs(estimate - ones)


def simulate_central(rng: np.random.Generator, eps: float, reps: int) -> NDArray[np.float64]:
    return np.abs(rng.laplace(0.0, 1.0 / eps, size=reps))


# --- Composition -------------------------------------------------------------------------------


def br_composition(eps: float, k: int, delta: float) -> float:
    """k ε²/8 + ε √(k ln(1/δ)/2), for k ε-bounded-range mechanisms."""
    return k * eps**2 / 8.0 + eps * math.sqrt(k * math.log(1.0 / delta) / 2.0)


def advanced_composition(eps: float, k: int, delta: float) -> float:
    """ε √(2k ln(1/δ)) + k ε tanh(ε/2), for k ε-DP mechanisms."""
    return eps * math.sqrt(2.0 * k * math.log(1.0 / delta)) + k * eps * math.tanh(eps / 2.0)


# --- Zero-concentrated differential privacy ----------------------------------------------------


def zcdp_to_dp(rho: float, delta: float) -> float:
    """ρ + 2√(ρ ln(1/δ))."""
    return rho + 2.0 * math.sqrt(rho * math.log(1.0 / delta))


def rho_for(eps: float, delta: float) -> float:
    """The ρ with ρ + 2√(ρ ln(1/δ)) = ε."""
    log_term = math.log(1.0 / delta)
    root = math.sqrt(log_term + eps) - math.sqrt(log_term)
    return root**2


# --- Table and figure ------------------------------------------------------------------------


def print_table(rng: np.random.Generator) -> None:
    print("== Shuffled randomized response")
    eps, delta, n = 1.0, 1e-6, 1_000_000
    q = shuffle_q(eps, delta, n)
    print(f"eps={eps:g} delta={delta:g}: tau={math.tanh(eps / 2):.4f}  min n={min_people(eps, delta)}")
    print(f"n={n}: q={q:.6f}  expected coins nq={n * q:.0f}  local eps of one message={local_eps_of_message(q):.2f}")
    print(f"clones range: ln(n/(16 ln(2/delta)))={math.log(n / (16 * math.log(2 / delta))):.2f}")
    print(
        f"n={n}: sd bounds shuffle={shuffle_sd_bound(eps, delta, n):.1f}  "
        f"local={local_count_sd(eps, n):.1f}  central={central_count_sd(eps):.3f}"
    )
    ones = n // 2
    sh = float(np.quantile(simulate_shuffle(rng, eps, delta, n, ones, 2000), 0.95))
    lo = float(np.quantile(simulate_local(rng, eps, n, ones, 2000), 0.95))
    ce = float(np.quantile(simulate_central(rng, eps, 2000), 0.95))
    print(f"n={n}: simulated 95th percentile shuffle={sh:.1f}  local={lo:.0f}  central={ce:.2f}")

    print("== Bounded range")
    eps_br, k, d = 0.1, 100, 1e-6
    print(
        f"eps={eps_br:g} k={k} delta={d:g}: basic={k * eps_br:.2f}  "
        f"advanced={advanced_composition(eps_br, k, d):.2f}  bounded range={br_composition(eps_br, k, d):.2f}"
    )

    print("== zCDP budget")
    target, d2, actions = 0.5, 1e-6, 20
    rho_user = rho_for(target, d2)
    rho_action = rho_user / actions
    print(f"target eps={target:g} delta={d2:g}: rho user={rho_user:.5f}  round trip={zcdp_to_dp(rho_user, d2):.4f}")
    # Replacing one action changes two cells by one each: Euclidean sensitivity √2.
    sigma = math.sqrt(2.0) / math.sqrt(2.0 * rho_action)
    print(f"{actions} actions: rho action={rho_action:.6f}  sigma={sigma:.1f}")
    print(f"one action alone: eps={zcdp_to_dp(rho_action, d2):.3f}")


def plot_counts(rng: np.random.Generator) -> Figure:
    eps, delta = 1.0, 1e-6
    ns = np.unique(np.logspace(4, 8, 13).astype(int))
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    local = [float(np.quantile(simulate_local(rng, eps, int(n), int(n) // 2, 2000), 0.95)) for n in ns]
    shuffle = [float(np.quantile(simulate_shuffle(rng, eps, delta, int(n), int(n) // 2, 2000), 0.95)) for n in ns]
    central = [float(np.quantile(simulate_central(rng, eps, 2000), 0.95)) for _ in ns]
    ax.plot(ns, local, "-o", color="C0", markersize=3, label="local randomized response")
    ax.plot(ns, shuffle, "-o", color="C2", markersize=3, label=r"shuffled randomized response, $\delta = 10^{-6}$")
    ax.plot(ns, central, "-o", color="C1", markersize=3, label="central Laplace mechanism")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"number of people $n$")
    ax.set_ylabel("absolute error of the count (95th percentile)")
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
    fig = plot_counts(np.random.default_rng(args.seed + 1))
    fig.savefig(args.out / "shuf-counts.svg", metadata={"Date": None})
    plt.close(fig)


if __name__ == "__main__":
    main()
