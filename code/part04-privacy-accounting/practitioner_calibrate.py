"""Calibrate and account a DP-SGD run before training it: the practitioner edition's workflow chapter
book-practitioner/part04-privacy-accounting/10-calibrate-dp-sgd.md.

    uv run python code/part04-privacy-accounting/practitioner_calibrate.py
    uv run python code/part04-privacy-accounting/practitioner_calibrate.py --clip 0.5 --workers 4
    uv run python code/part04-privacy-accounting/practitioner_calibrate.py --n 60000 --batch 256 --epochs 10 \
        --epsilon 3 --dataset MNIST --unit image

The options --n, --batch, --epochs, --epsilon, --delta (default: the 1/n^1.1 rule), --dataset and --unit choose the
run of steps 2 and 4-7; the defaults are Part 5's run and reproduce the output the page quotes. The planning table of
step 3 is fixed.

No model is trained here; Part 5's DP-SGD workflow trains it. This script does the accounting around that run, in
the page's seven steps, and prints every number the page quotes on a labelled line:

1. The privacy unit, n and delta: delta = 10^floor(log10(n^-1.1)), the book's one rule.
2. The sampler: Poisson sampling with rate q = B/n, and T = int(E/q) steps, the count Opacus's
   `get_noise_multiplier` uses. Also what Opacus's `DPDataLoader.from_data_loader` does instead: q = 1/ceil(n/B) and
   ceil(n/B) steps an epoch. The chosen run (steps 4-7) uses the loader's convention, the steps training actually
   runs, and checks that it equals get_noise_multiplier's count int(E/q) at that q.
3. The sigma planning table: for each (n, B, E, target epsilon), the noise multiplier from Opacus
   (`get_noise_multiplier(..., accountant="prv")`) and from dp_accounting (`calibrate_dp_mechanism` with a PLD
   accountant), and sigma/B x 1000, the noise in the averaged gradient per unit of clipping norm.
4. At the chosen cell, the run Part 5 trains (Adult, B = 1,000, E = 10, epsilon = 4, which gives sigma = 0.973), its
   epsilon by five accountants: Opacus RDP, PRV and GDP, dp_accounting RDP and PLD, with a pass or fail line, and the
   PLD bracket: the optimistic (lower) estimate next to the pessimistic (upper) one, the accountant's error bar.
5. What if the loader is not Poisson: fixed-size batches under add/remove (sigma/2), fixed-size batches under
   replace-one (sampling without replacement at sigma/2, dp_accounting RDP), and shuffling under zero-out with no
   amplification.
6. The accountant during training: one step per logical batch, epsilon each epoch, an early stop, the epsilon to
   report (the planned maximum), and a save, load and resume of the accountant's state_dict.
7. The filled privacy statement.

The whole run is the default and takes about 45 s on an idle 12-core CPU (4 min of CPU time): the planning
table's 24 cells (48 calibrations, about 1-8 s each) run in parallel worker processes, one thread each. The test
(tests/test_practitioner_calibrate.py) runs it and checks every number the page quotes. There is no reduced mode:
the page quotes this run.

Accuracy settings. Opacus's calibration stops when epsilon is within 0.01 below the target (`epsilon_tolerance`);
its PRV accountant uses its default `eps_error=0.01`. dp_accounting's calibration stops when sigma is within 0.01
(`tol`), and the PLD accountant discretizes privacy loss on a grid of 1e-3 (`value_discretization_interval`) in the
table, which keeps a cell near 1 s, and on the default 1e-4 for the single-run epsilons of steps 4, 5 and 7. A grid of
1e-2 is too coarse here: it overstates epsilon enough to double sigma in some cells.

Datasets:

- Adult (UCI Machine Learning Repository, dataset 2, "Census Income"; Becker & Kohavi, 1996; licence CC BY 4.0),
  loaded with `practitioner_datasets.adult()`: 48,842 rows, one person each. The run trained on it is Part 5's
  DP-SGD workflow, whose training split has n = 30,000 people; this script uses that n, so its delta and its chosen
  cell match Part 5's run exactly. Only the count is used.
- MNIST (LeCun, Cortes & Burges; the torchvision copy; licence CC BY-SA 3.0): n = 60,000 training images. Size only,
  a constant here: nothing is downloaded.
- Breast Cancer Wisconsin (Diagnostic) (UCI dataset 17; bundled with scikit-learn as load_breast_cancer; licence
  CC BY 4.0): n = 569. Size only, a constant here.

Library versions: opacus 1.6.0, dp-accounting 0.6.0 (the practitioner dependency group).
"""

import os

# The planning table runs one calibration per process; multi-threaded BLAS in twelve processes only contends.
for _variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_variable, "1")

import argparse
import math
import sys
import warnings
from collections.abc import Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from importlib.metadata import version
from multiprocessing import get_context
from pathlib import Path

import opacus
from dp_accounting import (
    DpEvent,
    GaussianDpEvent,
    LowerEndpointAndGuess,
    NeighboringRelation,
    PoissonSampledDpEvent,
    SampledWithoutReplacementDpEvent,
    SelfComposedDpEvent,
    calibrate_dp_mechanism,
)
from dp_accounting.pld import PLDAccountant, privacy_loss_distribution
from dp_accounting.rdp import RdpAccountant
from opacus.accountants import GaussianAccountant, PRVAccountant, RDPAccountant
from opacus.accountants.utils import get_noise_multiplier

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import practitioner_datasets

# Opacus's RDP accountant warns when the best Rényi order is the largest one it tries; the epsilon it returns is
# still a valid upper bound, and the page compares it with the other accountants.
warnings.filterwarnings("ignore", message="Optimal order is the largest alpha")
# Opacus's GDP accountant warns that it can underestimate epsilon; step 4 shows that it does.
warnings.filterwarnings("ignore", message="GDP accounting is experimental")

# The run Part 5 trains (code/part05-dp-ml/practitioner_dp_sgd.py): Adult's training split, Poisson batches of 1,000
# (q = 1/30), 10 epochs (300 steps), target epsilon 4; it trains at the sigma get_noise_multiplier returns, unrounded.
N_ADULT = 30_000  # Part 5's DP-SGD workflow trains on 30,000 of Adult's 48,842 rows
N_MNIST = 60_000  # MNIST's training set (size only)
N_BREAST_CANCER = 569  # scikit-learn's breast cancer data (size only)
BATCH, EPOCHS, TARGET_EPSILON = 1_000, 10, 4.0
PLD_GRID = 1e-3  # value_discretization_interval of the PLD accountant in the planning table


# --- Step 1: the privacy unit, n and delta ---------------------------------------------------------------------


# excerpt: calibrate-delta
def delta_for(n: int) -> float:
    """The book's default delta: at most 1/n^1.1, rounded down to a power of ten."""
    return 10.0 ** math.floor(math.log10(n**-1.1))


# end excerpt


# --- Step 2: the sampler ----------------------------------------------------------------------------------------


def poisson_plan(n: int, batch: int, epochs: int) -> tuple[float, int]:
    """Sampling rate q = B/n and the step count T = int(E/q) that Opacus's get_noise_multiplier uses."""
    q = batch / n
    return q, int(epochs / q)


def opacus_loader_plan(n: int, batch: int, epochs: int) -> tuple[float, int, int]:
    """What DPDataLoader.from_data_loader does: q = 1/len(loader), int(1/q) steps an epoch, and the total it runs."""
    batches = math.ceil(n / batch)  # len() of a DataLoader with drop_last=False
    q = 1 / batches
    per_epoch = int(1 / q)
    return q, per_epoch, per_epoch * epochs


def opacus_planned_steps(q: float, epochs: int) -> int:
    """The step count get_noise_multiplier (and so make_private_with_epsilon) calibrates sigma for."""
    return int(epochs / q)


# --- Step 3: the sigma planning table ---------------------------------------------------------------------------


# excerpt: calibrate-sigma
def sigma_opacus(epsilon: float, delta: float, q: float, epochs: int) -> float:
    """Opacus: the sigma whose PRV epsilon after `epochs` is at most the target (to 0.01 in epsilon)."""
    return get_noise_multiplier(
        target_epsilon=epsilon, target_delta=delta, sample_rate=q, epochs=epochs, accountant="prv"
    )


def sigma_dp_accounting(epsilon: float, delta: float, q: float, steps: int) -> float:
    """dp_accounting: the same search with a PLD accountant (to 0.01 in sigma)."""

    def training_run(sigma: float) -> DpEvent:
        return SelfComposedDpEvent(PoissonSampledDpEvent(q, GaussianDpEvent(sigma)), steps)

    def accountant() -> PLDAccountant:
        return PLDAccountant(value_discretization_interval=PLD_GRID)

    return calibrate_dp_mechanism(accountant, training_run, epsilon, delta, LowerEndpointAndGuess(0.3, 1.0), tol=0.01)


# end excerpt


@dataclass(frozen=True)
class Cell:
    dataset: str
    n: int
    batch: int
    epochs: int
    epsilon: float


@dataclass(frozen=True)
class Row:
    cell: Cell
    delta: float
    q: float
    steps: int
    sigma_prv: float
    sigma_pld: float

    @property
    def noise_per_example(self) -> float:
        """sigma/B x 1000: the noise in the averaged gradient, per unit of clipping norm, times 1000."""
        return self.sigma_prv / self.cell.batch * 1000

    @property
    def gap(self) -> float:
        """Relative difference between the two accountants' sigma."""
        return abs(self.sigma_prv - self.sigma_pld) / self.sigma_pld


def plan_cell(cell: Cell) -> Row:
    delta = delta_for(cell.n)
    q, steps = poisson_plan(cell.n, cell.batch, cell.epochs)
    return Row(
        cell,
        delta,
        q,
        steps,
        sigma_opacus(cell.epsilon, delta, q, cell.epochs),
        sigma_dp_accounting(cell.epsilon, delta, q, steps),
    )


def planning_cells() -> list[Cell]:
    """Adult: B in {100, 300, 1000} x E in {10, 20} x epsilon in {1, 4, 8}; MNIST and breast cancer at one (B, E)."""
    cells = [
        Cell("Adult", N_ADULT, batch, epochs, epsilon)
        for batch in (100, 300, 1000)
        for epochs in (10, 20)
        for epsilon in (1.0, 4.0, 8.0)
    ]
    cells += [Cell("MNIST", N_MNIST, 256, 20, epsilon) for epsilon in (1.0, 4.0, 8.0)]
    cells += [Cell("breast cancer", N_BREAST_CANCER, 64, 20, epsilon) for epsilon in (1.0, 4.0, 8.0)]
    return cells


def planning_table(cells: Sequence[Cell], workers: int) -> list[Row]:
    """Every cell, calibrated in parallel processes (each cell is independent and deterministic)."""
    with ProcessPoolExecutor(max_workers=workers, mp_context=get_context("spawn")) as pool:
        return list(pool.map(plan_cell, cells))


# --- Step 4: check epsilon with five accountants ----------------------------------------------------------------


def round_up(x: float, decimals: int = 2) -> float:
    """Round up, so that a reported epsilon is never below the accountant's."""
    scale = 10**decimals
    return math.ceil(x * scale - 1e-9) / scale


# excerpt: calibrate-crosscheck
def crosscheck_epsilon(sigma: float, q: float, steps: int, delta: float) -> dict[str, float]:
    """Epsilon of one planned run by five accountants (GDP is the trap: it is not an upper bound)."""
    epsilons: dict[str, float] = {}
    for name, opacus_accountant in [("Opacus RDP", RDPAccountant()), ("Opacus PRV", PRVAccountant())]:
        opacus_accountant.history = [(sigma, q, steps)]  # what `steps` calls of .step() would record
        epsilons[name] = opacus_accountant.get_epsilon(delta)
    run = SelfComposedDpEvent(PoissonSampledDpEvent(q, GaussianDpEvent(sigma)), steps)
    epsilons["dp_accounting RDP"] = RdpAccountant().compose(run).get_epsilon(delta)
    epsilons["dp_accounting PLD"] = PLDAccountant().compose(run).get_epsilon(delta)
    gdp = GaussianAccountant()
    gdp.history = [(sigma, q, steps)]
    epsilons["Opacus GDP"] = gdp.get_epsilon(delta)
    return epsilons


# end excerpt


def crosscheck_passes(epsilons: dict[str, float], target: float) -> tuple[bool, bool, bool]:
    """PLD at most the target; RDP at least PLD; Opacus PRV within 3% of dp_accounting PLD."""
    pld = epsilons["dp_accounting PLD"]
    return (
        pld <= target,
        epsilons["dp_accounting RDP"] >= pld,
        abs(epsilons["Opacus PRV"] - pld) / pld < 0.03,
    )


def pld_bracket(sigma: float, q: float, steps: int, delta: float, grid: float = 1e-4) -> tuple[float, float]:
    """The PLD's lower (optimistic) and upper (pessimistic, as PLDAccountant computes it) epsilon on the same grid."""
    optimistic = privacy_loss_distribution.from_gaussian_mechanism(
        sigma, sampling_prob=q, pessimistic_estimate=False, value_discretization_interval=grid, use_connect_dots=False
    )
    lower = optimistic.self_compose(steps).get_epsilon_for_delta(delta)
    run = SelfComposedDpEvent(PoissonSampledDpEvent(q, GaussianDpEvent(sigma)), steps)
    upper = PLDAccountant(value_discretization_interval=grid).compose(run).get_epsilon(delta)
    return lower, upper


# --- Step 5: what if the loader is not Poisson ------------------------------------------------------------------


# excerpt: calibrate-samplers
def sampler_epsilons(sigma: float, n: int, batch: int, epochs: int, delta: float) -> dict[str, float]:
    """Epsilon of the same sigma, n, B and E under each way of forming batches (steps as the loaders run them)."""
    q, _, steps = opacus_loader_plan(n, batch, epochs)  # the steps the loaders run
    poisson = SelfComposedDpEvent(PoissonSampledDpEvent(q, GaussianDpEvent(sigma)), steps)
    # Fixed-size batches, add/remove: one record changes a batch's sum by up to 2C, so the noise counts as sigma/2.
    fixed = SelfComposedDpEvent(PoissonSampledDpEvent(q, GaussianDpEvent(sigma / 2)), steps)
    # Fixed-size batches, replace-one: swapping a record moves the sum by up to 2C, so sigma/2 again; only RDP
    # accounts for sampling without replacement.
    without = SelfComposedDpEvent(SampledWithoutReplacementDpEvent(n, batch, GaussianDpEvent(sigma / 2)), steps)
    replace_one = RdpAccountant(neighboring_relation=NeighboringRelation.REPLACE_ONE)
    # Shuffle and partition, zero-out relation: no amplification applies, so one Gaussian release per epoch.
    shuffled = SelfComposedDpEvent(GaussianDpEvent(sigma), epochs)
    return {
        "Poisson (PLD)": PLDAccountant().compose(poisson).get_epsilon(delta),
        "fixed size, add/remove, sigma/2 (PLD)": PLDAccountant().compose(fixed).get_epsilon(delta),
        "fixed size, replace-one (RDP)": replace_one.compose(without).get_epsilon(delta),
        "shuffle, zero-out, no amplification (PLD)": PLDAccountant().compose(shuffled).get_epsilon(delta),
    }


# end excerpt


def pld_supports_without_replacement(n: int, batch: int, sigma: float) -> tuple[bool, bool]:
    """Whether dp_accounting's PLD accountant can account for sampling without replacement (it cannot)."""
    event = SampledWithoutReplacementDpEvent(n, batch, GaussianDpEvent(sigma))
    return (
        PLDAccountant().supports(event),
        PLDAccountant(NeighboringRelation.REPLACE_ONE).supports(event),
    )


# --- Step 6: the accountant during training ---------------------------------------------------------------------


# excerpt: calibrate-train
def train_epochs(
    accountant: PRVAccountant, sigma: float, q: float, steps_per_epoch: int, epochs: int, delta: float
) -> list[float]:
    """Step the accountant once per logical batch for `epochs` epochs; epsilon at the end of each epoch."""
    epsilons: list[float] = []
    for _ in range(epochs):
        for _ in range(steps_per_epoch):
            accountant.step(noise_multiplier=sigma, sample_rate=q)  # what the optimizer's step does
        epsilons.append(float(accountant.get_epsilon(delta)))
    return epsilons


# end excerpt


@dataclass(frozen=True)
class Resumed:
    stopped: list[float]  # epsilon after each epoch, up to the early stop
    planned_maximum: float  # epsilon of the whole planned run: what to report
    resumed: list[float]  # epsilon after each epoch, continued from a saved state
    uninterrupted: list[float]  # epsilon after each epoch of a run with no stop
    steps_recorded: int


def stop_and_resume(sigma: float, q: float, steps_per_epoch: int, epochs: int, stop: int, delta: float) -> Resumed:
    accountant = PRVAccountant()
    stopped = train_epochs(accountant, sigma, q, steps_per_epoch, stop, delta)
    planned = PRVAccountant()
    planned.history = [(sigma, q, steps_per_epoch * epochs)]
    # excerpt: calibrate-resume
    state = accountant.state_dict()  # saved with the model; PrivacyEngine.save_checkpoint stores it too
    # ... the job stops, and later restarts ...
    restarted = PRVAccountant()
    restarted.load_state_dict(state)  # without this, epsilon would restart from zero
    resumed = train_epochs(restarted, sigma, q, steps_per_epoch, epochs - stop, delta)
    steps_recorded = sum(steps for _, _, steps in restarted.history)
    # end excerpt
    uninterrupted = train_epochs(PRVAccountant(), sigma, q, steps_per_epoch, epochs, delta)
    return Resumed(stopped, float(planned.get_epsilon(delta)), resumed, uninterrupted, steps_recorded)


# --- Step 7: the privacy statement ------------------------------------------------------------------------------


# excerpt: calibrate-statement
def privacy_statement(
    unit: str, dataset: str, n: int, batch: int, epochs: int, sigma: float, clip: float, delta: float, calibration: str
) -> str:
    """The filled template of the reporting page: every input a reviewer needs to recompute epsilon."""
    q, _, steps = opacus_loader_plan(n, batch, epochs)  # the steps the Poisson loader runs
    run = SelfComposedDpEvent(PoissonSampledDpEvent(q, GaussianDpEvent(sigma)), steps)
    epsilon = round_up(PLDAccountant().compose(run).get_epsilon(delta))
    epsilon_tiny_delta = round_up(PLDAccountant().compose(run).get_epsilon(1 / n**2))
    return f"""\
Unit: one {unit} (a row of {dataset}); neighbouring datasets differ by adding or removing one {unit}
Data: n = {n:,}; Poisson sampling with q = {q:.6f} (B = {batch}); T = {steps:,} steps ({epochs} epochs)
Mechanism: DP-SGD, clipping norm C = {clip}, noise multiplier sigma = {sigma:.3f}
Accountant: PLD, dp-accounting {version("dp-accounting")}; {calibration}
Guarantee: (epsilon = {epsilon:.2f}, delta = {delta:.0e}); epsilon = {epsilon_tiny_delta:.2f} at delta = 1/n^2 = {1 / n**2:.1e}
Not covered: hyperparameter tuning runs (see Part 5), and any model released before this one"""


# end excerpt


# --- The run ----------------------------------------------------------------------------------------------------


def print_table(rows: Sequence[Row]) -> None:
    print(
        f"{'dataset':<14}{'n':>7}{'delta':>7}{'B':>6}{'E':>4}{'eps':>5}{'T':>7}{'sigma PRV':>11}{'sigma PLD':>11}"
        f"{'sigma/B x1000':>15}"
    )
    for r in rows:
        c = r.cell
        print(
            f"{c.dataset:<14}{c.n:>7}{r.delta:>7.0e}{c.batch:>6}{c.epochs:>4}{c.epsilon:>5g}{r.steps:>7}"
            f"{r.sigma_prv:>11.2f}{r.sigma_pld:>11.2f}{r.noise_per_example:>15.2f}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seed", type=int, default=0, help="unused: accounting is deterministic")
    parser.add_argument("--n", type=int, default=N_ADULT, help="records (privacy units) in the training set")
    parser.add_argument("--batch", type=int, default=BATCH, help="expected (logical) batch size B")
    parser.add_argument("--epochs", type=int, default=EPOCHS, help="planned maximum number of epochs E")
    parser.add_argument("--epsilon", type=float, default=TARGET_EPSILON, help="target epsilon")
    parser.add_argument("--delta", type=float, default=None, help="default: 10^floor(log10(n^-1.1))")
    parser.add_argument("--dataset", default="Adult", help="dataset name, for the printed lines")
    parser.add_argument("--unit", default="person", help="what one row protects, for the statement")
    parser.add_argument("--clip", type=float, default=1.0, help="clipping norm C, for the statement only")
    parser.add_argument(
        "--workers",
        type=int,
        default=int(os.environ.get("PRIVACY_WORKERS") or min(12, os.cpu_count() or 1)),
        help="processes for the table",
    )
    args = parser.parse_args()
    n: int = args.n
    batch: int = args.batch
    epochs: int = args.epochs
    target: float = args.epsilon
    dataset: str = args.dataset
    unit: str = args.unit
    clip: float = args.clip
    workers: int = args.workers
    delta: float = delta_for(n) if args.delta is None else args.delta

    print("== Step 1: unit, n and delta")
    if n == N_ADULT and dataset == "Adult":
        adult = practitioner_datasets.adult()
        print(f"Adult rows: {adult.height:,}; Part 5's training split: n={n:,}")
    else:
        print(f"{dataset}: n={n:,}")
    print(f"delta_for({n:,}) = {delta_for(n):.0e}")
    if args.delta is not None:
        print(f"delta used (--delta): {delta:.0e}")
    for other in [m for m in (N_MNIST, N_BREAST_CANCER) if m != n]:
        print(f"delta_for({other:,}) = {delta_for(other):.0e}")

    print("\n== Step 2: sampler")
    q_plan, steps_plan = poisson_plan(n, batch, epochs)
    print(f"Poisson: B={batch}, E={epochs}: q={q_plan:.6f}, T={steps_plan:,}")
    q, steps_per_epoch, steps = opacus_loader_plan(n, batch, epochs)
    print(
        f"Opacus DPDataLoader.from_data_loader: q=1/{math.ceil(n / batch)}={q:.6f}, "
        f"{steps_per_epoch} steps an epoch, {steps:,} in {epochs} epochs"
    )
    calibrated_steps = opacus_planned_steps(q, epochs)
    print(
        f"chosen run uses the loader's q and steps; get_noise_multiplier at that q counts int(E/q)="
        f"{calibrated_steps:,}: equal {calibrated_steps == steps}"
    )

    print("\n== Step 3: sigma planning table (Opacus PRV and dp_accounting PLD)")
    rows = planning_table(planning_cells(), workers)
    print_table(rows)
    worst = max(rows, key=lambda r: r.gap)
    c = worst.cell
    print(
        f"largest gap between PRV and PLD sigma: {worst.gap:.1%} ({c.dataset}, B={c.batch}, E={c.epochs}, "
        f"eps={c.epsilon:g})"
    )

    print("\n== Step 4: chosen cell, epsilon by five accountants")
    sigma = sigma_opacus(target, delta, q, epochs)  # what make_private_with_epsilon computes; Part 5 trains at it
    print(f"chosen cell: {dataset}, B={batch}, E={epochs}, eps={target:g}: sigma={sigma:.6f} (Opacus PRV, as Part 5)")
    epsilons = crosscheck_epsilon(sigma, q, steps, delta)
    for name, value in epsilons.items():
        print(f"{name:<18} eps={value:.2f}")
    below, rdp_above, close = crosscheck_passes(epsilons, target)
    print(f"PLD <= target: {below}; RDP >= PLD: {rdp_above}; |PRV - PLD|/PLD < 3%: {close}")
    print(f"crosscheck: {'pass' if below and rdp_above and close else 'FAIL'}")
    lower, upper = pld_bracket(sigma, q, steps, delta)
    print(
        f"PLD bracket (grid 1e-4): lower {lower:.4f} (optimistic), upper {upper:.4f} (pessimistic); width {upper - lower:.4f}"
    )

    print("\n== Step 5: other samplers, same sigma, n, B, E")
    for name, value in sampler_epsilons(sigma, n, batch, epochs, delta).items():
        print(f"{name:<42} eps={value:.2f}")
    add_remove, replace_one = pld_supports_without_replacement(n, batch, sigma)
    print(f"PLDAccountant supports sampling without replacement: add/remove {add_remove}, replace-one {replace_one}")

    print("\n== Step 6: accountant during training (Opacus PRV, one step per logical batch)")
    stop = max(1, 3 * epochs // 5)  # stop early at 60% of the planned epochs
    print(f"steps per epoch: {steps_per_epoch}; planned: {epochs} epochs = {steps:,} steps")
    result = stop_and_resume(sigma, q, steps_per_epoch, epochs, stop, delta)
    for epoch in sorted({e for e in (1, 2, 4, stop) if e <= stop}):
        print(f"epoch {epoch:>2}: eps={result.stopped[epoch - 1]:.2f}")
    monotone = all(a <= b for a, b in zip(result.uninterrupted, result.uninterrupted[1:]))
    print(f"epsilon never decreases over {epochs} epochs: {monotone}")
    print(
        f"stopped early at epoch {stop}: eps={result.stopped[-1]:.2f}; "
        f"report the planned maximum: eps={result.planned_maximum:.2f}"
    )
    print(
        f"after resuming at epoch {stop}: eps at epoch {epochs}={result.resumed[-1]:.2f}; "
        f"uninterrupted run: eps={result.uninterrupted[-1]:.2f}; equal: {result.resumed[-1] == result.uninterrupted[-1]}"
    )
    print(
        f"steps recorded: {result.steps_recorded:,}; planned (the loader's {epochs} x {steps_per_epoch}): "
        f"{steps:,}; equal: {result.steps_recorded == steps}"
    )

    print("\n== Step 7: privacy statement")
    calibration = f"calibrated with Opacus {opacus.__version__} (PRV)"
    print(privacy_statement(unit, dataset, n, batch, epochs, sigma, clip, delta, calibration))


if __name__ == "__main__":
    main()
