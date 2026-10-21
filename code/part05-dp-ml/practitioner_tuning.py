"""Tune DP-SGD's learning rate on Adult and count what the tuning costs: step 9 of 13-train-dp-sgd.md.

    uv run python code/part05-dp-ml/practitioner_tuning.py

Prints, with seed 0, every number the chapter quotes: the noise multiplier each tuning strategy needs for a total
epsilon of 4 at delta = 1e-5 (one run, a grid of six learning rates counted naively and honestly, and a random
search whose number of runs is drawn from a Poisson distribution with mean 4); the true epsilon of the naive
grid; then the random search itself (the number of runs drawn, each run's learning rate and validation accuracy,
the chosen learning rate and its test accuracy); and, for comparison, a non-private sweep over the same learning
rates (practitioner_dp_sgd.py chooses its own rates on the public stand-in rows instead, so its tuning costs no
privacy). The run takes about half a minute on a CPU.

The accounting of tuning uses dp_accounting's RDP accountant, the one that supports RepeatAndSelectDpEvent
(Papernot & Steinke, 2022: a random number of runs, the best one released); all four strategies are calibrated
with it so they compare like with like. Training runs use Opacus with the PRV accountant, as in
practitioner_dp_sgd.py, whose data, model and training loop this script imports.

Dataset: Adult ("Census Income"), UCI Machine Learning Repository dataset 2 (Becker & Kohavi, 1996), licence
CC BY 4.0, loaded by practitioner_datasets.adult(); split as in practitioner_dp_sgd.py (seed 0): 30,000 private
training rows, 2,000 validation rows standing in for public data (the runs are scored on them), and 16,842 test
rows. Scoring on rows outside the private training set keeps each run's score inside its DP guarantee.
"""

import argparse
import math
import warnings
from collections.abc import Callable

import dp_accounting
import numpy as np
from practitioner_dp_sgd import (
    BATCH,
    EPOCHS,
    N_TRAIN,
    accuracy,
    delta_for,
    load_splits,
    train_non_private,
    train_private,
)
from torch import nn

LEARNING_RATES: tuple[float, ...] = (0.05, 0.1, 0.2, 0.5, 1.0, 2.0)
MEAN_RUNS = 4.0
TOTAL_EPSILON = 4.0


# ---------------------------------------------------------------------------------------------------------------
# What each strategy costs, as a dp_accounting event of the noise multiplier sigma of each run

SAMPLE_RATE = BATCH / N_TRAIN
STEPS = round(EPOCHS / SAMPLE_RATE)


def one_run(sigma: float) -> dp_accounting.DpEvent:
    step = dp_accounting.PoissonSampledDpEvent(SAMPLE_RATE, dp_accounting.GaussianDpEvent(sigma))
    return dp_accounting.SelfComposedDpEvent(step, STEPS)


def grid_composed(sigma: float) -> dp_accounting.DpEvent:
    return dp_accounting.SelfComposedDpEvent(one_run(sigma), len(LEARNING_RATES))


# excerpt: tuning-cost
def random_search(sigma: float) -> dp_accounting.DpEvent:
    """A Poisson number of runs with mean 4 (shape=inf selects the Poisson case); only the best is released."""
    return dp_accounting.dp_event.RepeatAndSelectDpEvent(one_run(sigma), mean=MEAN_RUNS, shape=math.inf)


def sigma_for(strategy: Callable[[float], dp_accounting.DpEvent], epsilon: float, delta: float) -> float:
    """The smallest noise multiplier per run that keeps the whole strategy within (epsilon, delta).

    The search starts at 0.1 and tries 1 first.
    """
    return dp_accounting.calibrate_dp_mechanism(
        dp_accounting.rdp.RdpAccountant, strategy, epsilon, delta, dp_accounting.LowerEndpointAndGuess(0.1, 1.0)
    )


# end excerpt


def rdp_epsilon(event: dp_accounting.DpEvent, delta: float) -> float:
    return dp_accounting.rdp.RdpAccountant().compose(event).get_epsilon(delta)


# ---------------------------------------------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    seed: int = args.seed
    warnings.filterwarnings("ignore", category=UserWarning)  # Opacus's secure-RNG and hook reminders

    train, validation, test = load_splits(seed)
    delta = delta_for(N_TRAIN)
    print(f"n={N_TRAIN:,}  delta={delta:.0e}  q={SAMPLE_RATE:.4f}  steps={STEPS}")
    print(f"learning rates: {', '.join(f'{lr:g}' for lr in LEARNING_RATES)}")

    print(f"== Noise multiplier per run for a total epsilon of {TOTAL_EPSILON:g} (RDP accountant)")
    sigma_one = sigma_for(one_run, TOTAL_EPSILON, delta)
    sigma_grid = sigma_for(grid_composed, TOTAL_EPSILON, delta)
    sigma = sigma_for(random_search, TOTAL_EPSILON, delta)
    print(f"one run, no tuning: sigma={sigma_one:.3f}")
    print(f"grid of {len(LEARNING_RATES)}, all runs counted: sigma={sigma_grid:.3f}")
    print(f"random search, Poisson mean {MEAN_RUNS:g} runs: sigma={sigma:.3f}")
    naive = rdp_epsilon(grid_composed(sigma_one), delta)
    print(f"naive grid of {len(LEARNING_RATES)} at sigma={sigma_one:.3f}: reports epsilon={TOTAL_EPSILON:.2f}")
    print(f"naive grid, true epsilon of all {len(LEARNING_RATES)} runs = {naive:.2f}")

    print("== Random search")
    rng = np.random.default_rng(seed)
    # excerpt: tuning-search
    runs = int(rng.poisson(MEAN_RUNS))  # drawn, not chosen: the guarantee counts on the randomness
    print(f"runs drawn: {runs}")
    results: list[tuple[float, float, nn.Module]] = []
    for i in range(runs):
        lr = float(rng.choice(LEARNING_RATES))
        run = train_private(train, sigma=sigma, delta=delta, epochs=EPOCHS, lr=lr, seed=seed + i)
        score = accuracy(run.model, validation)  # scored on rows outside the private training set
        print(f"run {i + 1}: lr={lr:g}  validation accuracy {score:.3f}")
        results.append((score, lr, run.model))
    # end excerpt
    if not results:
        print("no run drawn: nothing is released, and the guarantee still holds")
    else:
        _, best_lr, best_model = max(results, key=lambda result: result[0])
        print(f"chosen lr={best_lr:g}: test accuracy {accuracy(best_model, test):.3f}")
        print(f"total epsilon of the search (RDP) = {rdp_epsilon(random_search(sigma), delta):.2f}")

    print("== Non-private sweep on the private rows, for comparison only (validation accuracy)")
    scores: dict[float, float] = {}
    for lr in LEARNING_RATES:
        scores[lr] = accuracy(train_non_private(train, epochs=EPOCHS, lr=lr, seed=seed), validation)
        print(f"non-private lr={lr:g}: validation accuracy {scores[lr]:.3f}")
    print(f"best non-private lr on the private rows (not used): {max(scores, key=lambda lr: scores[lr]):g}")


if __name__ == "__main__":
    main()
