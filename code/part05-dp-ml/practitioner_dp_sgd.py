"""Train a model on Adult with DP-SGD in Opacus, end to end: the workflow of the practitioner chapter 13-train-dp-sgd.md.

    uv run python code/part05-dp-ml/practitioner_dp_sgd.py
    uv run python code/part05-dp-ml/practitioner_dp_sgd.py --record /tmp/run.json
    uv run python code/part05-dp-ml/practitioner_dp_sgd.py --epsilon 2 --epochs 20   # another budget or length

--epsilon (default 4) sets the target ε of the main run (the released model, steps 6-8 and 12) and --epochs
(default 10) the number of epochs of every run; the defaults give the output the chapter quotes.

Prints, with seed 0, every number the chapter quotes, one labelled line each: n and delta (step 1), the number
of features (step 2), the non-private and clip-only baselines (step 3), the validator's verdict (step 4), the
per-example gradient norms at initialisation (step 5), the noise multiplier for epsilon = 4 (step 6), the
training run and its epsilon (step 7), the epsilon of the same run under two other accountants (step 8), test
accuracy at epsilon = 0.25, 1, 2, 4, 8 (step 10) and accuracy by sex with exact 95% intervals (step 10). It then
writes the run's record as JSON (step 12) for practitioner_release.py. Step 9, tuning the learning rate, is
practitioner_tuning.py. The default run takes about a minute on a CPU, and the test runs it in full.

Dataset: Adult ("Census Income"), UCI Machine Learning Repository dataset 2 (Becker & Kohavi, 1996), licence
CC BY 4.0, https://archive.ics.uci.edu/static/public/2/adult.zip, loaded by practitioner_datasets.adult(): the
48,842 rows of adult.data and adult.test, the same rows as OpenML data_id 1590. One row is one person.

The split (seed 0): 30,000 rows are the private training set, 2,000 are a validation set that stands in for
public data (it sets the clipping norm here and selects the learning rate in practitioner_tuning.py; it is
never trained on), and the other 16,842 rows are the test set.

Preprocessing uses only public facts, never statistics of the training rows: the category lists are copied
from the dataset's documentation (adult.names), and each numeric column is scaled by public bounds. A missing
value ('?' in the source) becomes its own category, '?', so no row is dropped. fnlwgt (a survey weight, not a
property of the person) and education (the same information as education-num) are not used.

The model is a multilayer perceptron, 91 inputs -> 64 ReLU units -> 2 logits (6,018 weights), trained with
SGD with momentum 0.9, logical batch 1,000 (sampling rate q = 1/30, physical batches of 500) and 10 epochs
(300 steps), clipping norm C = 1. The learning rate is chosen before any private row is used: six non-private
runs (learning rates 0.05 to 2) train on 1,500 of the validation rows for the same 300 steps and are scored on the
other 500; the best is the non-private baseline's learning rate, and DP-SGD uses ten times it, the rule of thumb
for clipped gradients (02-dp-sgd.md). practitioner_tuning.py shows how to tune on private data with the cost
counted. Every run starts from the same initial weights and the same random state, torch.manual_seed(seed), with one CPU thread, so the
printed numbers repeat exactly from run to run.
"""

import argparse
import importlib.metadata
import json
import math
import sys
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path

import dp_accounting
import numpy as np
import polars as pl
import torch
from numpy.typing import NDArray
from opacus import PrivacyEngine
from opacus.accountants.utils import get_noise_multiplier
from opacus.data_loader import DPDataLoader
from opacus.grad_sample import GradSampleModule
from opacus.utils.batch_memory_manager import BatchMemoryManager
from opacus.validators import ModuleValidator
from scipy.stats import binomtest
from sklearn.preprocessing import OneHotEncoder
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import practitioner_datasets

ROOT = Path(__file__).resolve().parents[2]
RECORD = ROOT / "_build" / "practitioner" / "dp_sgd_run.json"

Array = NDArray[np.float64]

N_TRAIN, N_VALIDATION = 30_000, 2_000
BATCH, PHYSICAL_BATCH = 1_000, 500
EPOCHS = 10
CLIP = 1.0
SWEEP_RATES: tuple[float, ...] = (0.05, 0.1, 0.2, 0.5, 1.0, 2.0)  # tried on the public stand-in rows only
N_SWEEP_FIT = 1_500  # validation rows the sweep trains on; it scores on the other 500
DP_RATE_FACTOR = 10  # DP-SGD's learning rate is ten times the non-private one: clipped gradients are shorter
TARGET_EPSILON = 4.0
BUDGETS: tuple[float, ...] = (0.25, 1.0, 2.0, 4.0, 8.0)
HIDDEN = 64

torch.set_num_threads(1)  # the model is tiny; one thread is as fast and keeps float sums in a fixed order


# ---------------------------------------------------------------------------------------------------------------
# Step 1: unit, n and delta


# excerpt: dpsgd-delta
def delta_for(n: int) -> float:
    """The book's default: delta <= 1/n^1.1, rounded down to a power of ten."""
    return 10.0 ** math.floor(math.log10(n**-1.1))


# end excerpt


# ---------------------------------------------------------------------------------------------------------------
# Step 2: preprocessing from public facts

# Category lists from the dataset's documentation (adult.names), not from the data. '?' marks a missing value,
# which only workclass, occupation and native-country have.
CATEGORIES: dict[str, list[str]] = {
    "workclass": [
        "Private", "Self-emp-not-inc", "Self-emp-inc", "Federal-gov", "Local-gov", "State-gov", "Without-pay",
        "Never-worked", "?",
    ],
    "marital-status": [
        "Married-civ-spouse", "Divorced", "Never-married", "Separated", "Widowed", "Married-spouse-absent",
        "Married-AF-spouse",
    ],
    "occupation": [
        "Tech-support", "Craft-repair", "Other-service", "Sales", "Exec-managerial", "Prof-specialty",
        "Handlers-cleaners", "Machine-op-inspct", "Adm-clerical", "Farming-fishing", "Transport-moving",
        "Priv-house-serv", "Protective-serv", "Armed-Forces", "?",
    ],
    "relationship": ["Wife", "Own-child", "Husband", "Not-in-family", "Other-relative", "Unmarried"],
    "race": ["White", "Asian-Pac-Islander", "Amer-Indian-Eskimo", "Other", "Black"],
    "sex": ["Female", "Male"],
    "native-country": [
        "United-States", "Cambodia", "England", "Puerto-Rico", "Canada", "Germany", "Outlying-US(Guam-USVI-etc)",
        "India", "Japan", "Greece", "South", "China", "Cuba", "Iran", "Honduras", "Philippines", "Italy", "Poland",
        "Jamaica", "Vietnam", "Mexico", "Portugal", "Ireland", "France", "Dominican-Republic", "Laos", "Ecuador",
        "Taiwan", "Haiti", "Columbia", "Hungary", "Guatemala", "Nicaragua", "Scotland", "Thailand", "Yugoslavia",
        "El-Salvador", "Trinadad&Tobago", "Peru", "Hong", "Holand-Netherlands", "?",
    ],
}  # fmt: skip

# Public bounds: the survey's age range (17 to 90, its top code), the 16 education levels, a week's hours
# (1 to 99), and the survey's top code for capital gains (99,999), used for losses too, on a log scale.
NUMERIC_BOUNDS: dict[str, tuple[float, float]] = {
    "age": (17.0, 90.0),
    "education-num": (1.0, 16.0),
    "hours-per-week": (1.0, 99.0),
    "capital-gain": (0.0, math.log1p(99_999)),
    "capital-loss": (0.0, math.log1p(99_999)),
}


@dataclass(frozen=True)
class Split:
    x: torch.Tensor  # float32 features
    y: torch.Tensor  # int64 labels, 1 for income > $50K
    sex: list[str]


# excerpt: dpsgd-preprocess
def features(rows: pl.DataFrame) -> Array:
    """One-hot categories from the documented lists, numbers scaled by public bounds."""
    rows = rows.with_columns(pl.col("capital-gain", "capital-loss").log1p())
    numeric = [
        np.clip((rows[name].to_numpy() - low) / (high - low), 0.0, 1.0) for name, (low, high) in NUMERIC_BOUNDS.items()
    ]
    encoder = OneHotEncoder(categories=list(CATEGORIES.values()), sparse_output=False)
    text = rows.select(pl.col(list(CATEGORIES)).fill_null("?")).to_numpy()
    encoder.fit(text[:1])  # the categories are given, so fitting learns nothing from the rows
    return np.hstack([np.column_stack(numeric), encoder.transform(text)])


# end excerpt


def to_split(rows: pl.DataFrame) -> Split:
    x = torch.tensor(features(rows), dtype=torch.float32)
    y = torch.tensor((rows["income"] == ">50K").to_numpy(), dtype=torch.int64)
    return Split(x, y, rows["sex"].to_list())


def load_splits(seed: int) -> tuple[Split, Split, Split]:
    """Training (30,000), validation (2,000) and test (the other 16,842) rows of Adult, shuffled with the seed."""
    adult = practitioner_datasets.adult()
    order = np.random.default_rng(seed).permutation(adult.height)
    train = adult[order[:N_TRAIN]]
    validation = adult[order[N_TRAIN : N_TRAIN + N_VALIDATION]]
    test = adult[order[N_TRAIN + N_VALIDATION :]]
    return to_split(train), to_split(validation), to_split(test)


# ---------------------------------------------------------------------------------------------------------------
# The model, evaluation and the non-private loop


def make_model(n_features: int, seed: int) -> nn.Module:
    torch.manual_seed(seed)
    return nn.Sequential(nn.Linear(n_features, HIDDEN), nn.ReLU(), nn.Linear(HIDDEN, 2))


def predictions(model: nn.Module, split: Split) -> torch.Tensor:
    model.eval()
    with torch.no_grad():
        logits: torch.Tensor = model(split.x)
    return logits.argmax(dim=1)


def accuracy(model: nn.Module, split: Split) -> float:
    return float((predictions(model, split) == split.y).double().mean())


class WholeBatches(TensorDataset):
    """A TensorDataset that hands a batch over in one indexing operation instead of row by row.

    For a model this small, fetching 1,000 rows one at a time costs ten times more than the training step.
    """

    def __getitems__(self, indices: list[int]) -> list[torch.Tensor]:
        rows = torch.tensor(indices, dtype=torch.int64)
        return [tensor[rows] for tensor in self.tensors]


def as_batch(batch: list[torch.Tensor]) -> list[torch.Tensor]:
    return batch


def loader(split: Split) -> DataLoader[tuple[torch.Tensor, ...]]:
    return DataLoader(WholeBatches(split.x, split.y), batch_size=BATCH, shuffle=True, collate_fn=as_batch)


def train_non_private(train: Split, *, epochs: int, lr: float, seed: int) -> nn.Module:
    """The same model, batch, epochs and optimizer, with no clipping and no noise."""
    model = make_model(train.x.shape[1], seed)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    criterion = nn.CrossEntropyLoss()
    for _ in range(epochs):
        model.train()
        for x, y in loader(train):
            optimizer.zero_grad()
            criterion(model(x), y).backward()
            optimizer.step()
    return model


def public_sweep(validation: Split, *, steps: int, seed: int) -> dict[float, float]:
    """Non-private runs on the public stand-in rows only: fit on 1,500, score on the other 500, same steps.

    No private row is touched, so choosing the learning rate here costs no privacy.
    """
    fit = Split(validation.x[:N_SWEEP_FIT], validation.y[:N_SWEEP_FIT], validation.sex[:N_SWEEP_FIT])
    score = Split(validation.x[N_SWEEP_FIT:], validation.y[N_SWEEP_FIT:], validation.sex[N_SWEEP_FIT:])
    epochs = steps // math.ceil(N_SWEEP_FIT / BATCH)  # as many optimizer steps as the private run
    return {lr: accuracy(train_non_private(fit, epochs=epochs, lr=lr, seed=seed), score) for lr in SWEEP_RATES}


# ---------------------------------------------------------------------------------------------------------------
# Step 5: per-example gradient norms on the validation (proxy) data


def stacked_norms(module: nn.Module) -> torch.Tensor:
    """The norm of each row's whole gradient, from the per-example gradients Opacus stores in p.grad_sample."""
    squares = torch.zeros(())
    for p in module.parameters():
        sample: torch.Tensor = vars(p)["grad_sample"]  # an attribute Opacus sets, unknown to torch's types
        squares = squares + sample.flatten(start_dim=1).pow(2).sum(dim=1)
    return squares.sqrt()


def per_example_grad_norms(model: nn.Module, split: Split) -> Array:
    """Each validation row's gradient norm at the model's current weights."""
    # excerpt: dpsgd-grad-norms
    wrapped = GradSampleModule(model)  # adds hooks that keep one gradient per row
    wrapped.train()
    logits: torch.Tensor = wrapped(split.x)
    nn.CrossEntropyLoss()(logits, split.y).backward()  # fills p.grad_sample for every weight p
    norms = stacked_norms(wrapped)
    # end excerpt
    return norms.numpy().astype(np.float64)


# ---------------------------------------------------------------------------------------------------------------
# Steps 6 and 7: calibrate and train with Opacus


def calibrate(epsilon: float, delta: float, sample_rate: float, epochs: int) -> float:
    # excerpt: dpsgd-calibrate
    sigma = get_noise_multiplier(
        target_epsilon=epsilon,
        target_delta=delta,
        sample_rate=sample_rate,
        epochs=epochs,
        accountant="prv",
    )
    # end excerpt
    return sigma


@dataclass(frozen=True)
class PrivateRun:
    model: nn.Module
    sigma: float
    sample_rate: float
    steps: int
    epsilon: float  # from the PRV accountant; inf for the clip-only run
    sampler: str  # read from the loader Opacus returned: "poisson" for its DPDataLoader
    secure_mode: bool  # read from the engine: False here, so that the noise is seeded and the numbers repeat


def train_private(
    train: Split,
    *,
    sigma: float,
    delta: float,
    epochs: int,
    lr: float,
    seed: int,
    clip: float = CLIP,
    model: nn.Module | None = None,
) -> PrivateRun:
    """DP-SGD with Opacus: Poisson sampling, per-example clipping at clip, noise sigma * clip, virtual batches.

    A model passed in (pretrained, or with frozen weights) is trained instead of a fresh one; Opacus trains only
    the weights that require gradients.
    """
    if model is None:
        model = make_model(train.x.shape[1], seed)
    else:
        torch.manual_seed(seed)
    data_loader = loader(train)
    # excerpt: dpsgd-setup
    engine = PrivacyEngine(accountant="prv")
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    model, optimizer, data_loader = engine.make_private(
        module=model,
        optimizer=optimizer,
        data_loader=data_loader,
        noise_multiplier=sigma,
        max_grad_norm=clip,
    )  # Poisson sampling is the default: each row joins each batch with probability q
    # end excerpt
    criterion = nn.CrossEntropyLoss()
    for _ in range(epochs):
        model.train()
        # excerpt: dpsgd-train-loop
        with BatchMemoryManager(
            data_loader=data_loader, max_physical_batch_size=PHYSICAL_BATCH, optimizer=optimizer
        ) as batches:
            for x, y in batches:  # Opacus splits each logical batch; it steps once per logical batch
                optimizer.zero_grad()
                loss = criterion(model(x), y)
                loss.backward()
                optimizer.step()
        # end excerpt
    steps = sum(count for _, _, count in engine.accountant.history)
    epsilon = engine.get_epsilon(delta) if sigma > 0 else math.inf
    sampler = "poisson" if isinstance(data_loader, DPDataLoader) else "not poisson: " + type(data_loader).__name__
    return PrivateRun(model, sigma, 1 / len(data_loader), steps, epsilon, sampler, engine.secure_mode)


# ---------------------------------------------------------------------------------------------------------------
# Step 8: cross-check epsilon with dp_accounting


def epsilon_pld_rdp(sigma: float, sample_rate: float, steps: int, delta: float) -> tuple[float, float]:
    # excerpt: dpsgd-cross-check
    step = dp_accounting.PoissonSampledDpEvent(sample_rate, dp_accounting.GaussianDpEvent(sigma))
    training = dp_accounting.SelfComposedDpEvent(step, steps)
    epsilon_pld = dp_accounting.pld.PLDAccountant().compose(training).get_epsilon(delta)
    epsilon_rdp = dp_accounting.rdp.RdpAccountant().compose(training).get_epsilon(delta)
    # end excerpt
    return epsilon_pld, epsilon_rdp


# ---------------------------------------------------------------------------------------------------------------
# Step 10: accuracy by group, with exact (Clopper-Pearson) intervals


@dataclass(frozen=True)
class GroupAccuracy:
    group: str
    correct: int
    total: int
    low: float
    high: float

    @property
    def accuracy(self) -> float:
        return self.correct / self.total


def by_group(model: nn.Module, split: Split) -> list[GroupAccuracy]:
    right = (predictions(model, split) == split.y).numpy()
    sex = np.array(split.sex)
    rows: list[GroupAccuracy] = []
    for group in CATEGORIES["sex"]:
        # excerpt: dpsgd-per-group
        in_group = sex == group
        correct, total = int(right[in_group].sum()), int(in_group.sum())
        interval = binomtest(correct, total).proportion_ci(confidence_level=0.95, method="exact")
        # end excerpt
        rows.append(GroupAccuracy(group, correct, total, interval.low, interval.high))
    return rows


# ---------------------------------------------------------------------------------------------------------------
# Step 12: the run's record, for practitioner_release.py


@dataclass(frozen=True)
class Record:
    unit: str
    n: int
    delta: float
    sigma: float
    sample_rate: float
    steps: int
    epochs: int
    max_grad_norm: float
    accountant: str
    sampler: str
    secure_mode: bool
    seed: int
    library_versions: dict[str, str]
    epsilon_prv: float
    epsilon_pld: float
    epsilon_rdp: float
    tuning: dict[str, str | bool]


def library_versions() -> dict[str, str]:
    return {name: importlib.metadata.version(name) for name in ("opacus", "torch", "dp-accounting", "numpy")}


def write_record(record: Record, path: Path) -> None:
    # excerpt: dpsgd-record
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(asdict(record), indent=2) + "\n")
    print(f"record written to {path}")
    # end excerpt


# ---------------------------------------------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--record", type=Path, default=RECORD, help="where to write the run's JSON record")
    parser.add_argument("--epsilon", type=float, default=TARGET_EPSILON, help="target epsilon of the main run")
    parser.add_argument("--epochs", type=int, default=EPOCHS, help="epochs of every training run")
    args = parser.parse_args()
    seed: int = args.seed
    record_path: Path = args.record
    target_epsilon: float = args.epsilon
    epochs: int = args.epochs
    # Opacus's reminders (secure RNG off for experiments, hooks firing on inputs without gradients) say nothing
    # about this run; keep stderr for real problems.
    warnings.filterwarnings("ignore", category=UserWarning)

    train, validation, test = load_splits(seed)
    n = len(train.y)
    delta = delta_for(n)
    # Opacus samples at q = 1/len(loader) = 1/ceil(n/B) and takes len(loader) steps per epoch; plan with the same
    # values, so that the cross-check in step 8 compares like with like (q = B/n only when B divides n).
    sample_rate = 1 / math.ceil(n / BATCH)
    steps = epochs * math.ceil(n / BATCH)

    print("== Step 1: unit, n and delta")
    print(f"unit: one person (one row); n={n:,}  validation={len(validation.y):,}  test={len(test.y):,}")
    print(f"delta = 10^floor(log10(n^-1.1)) = {delta:.0e}")

    print("== Step 2: preprocessing from public facts")
    print(f"features={train.x.shape[1]}  missing values kept as category '?'")
    model = make_model(train.x.shape[1], seed)
    print(f"weights={sum(p.numel() for p in model.parameters()):,}")

    print("== Step 3: baselines")
    sweep = public_sweep(validation, steps=steps, seed=seed)
    print("learning-rate sweep on the public stand-in rows (fit 1,500, score 500):")
    print("  " + "  ".join(f"lr={lr:g}: {acc:.3f}" for lr, acc in sweep.items()))
    non_private_rate = max(sweep, key=lambda lr: sweep[lr])
    learning_rate = DP_RATE_FACTOR * non_private_rate
    print(f"non-private learning rate {non_private_rate:g}; DP-SGD learning rate {learning_rate:g} (x{DP_RATE_FACTOR})")
    non_private = train_non_private(train, epochs=epochs, lr=non_private_rate, seed=seed)
    print(f"non-private: test accuracy {accuracy(non_private, test):.3f}")
    clip_only = train_private(train, sigma=0.0, delta=delta, epochs=epochs, lr=learning_rate, seed=seed)
    print(f"clip only (C={CLIP:g}, no noise): test accuracy {accuracy(clip_only.model, test):.3f}")
    majority = float((test.y == 0).double().mean())
    print(f"always '<=50K': test accuracy {majority:.3f}")

    print("== Step 4: check the model")
    # excerpt: dpsgd-validate
    errors = ModuleValidator.validate(model, strict=False)
    print(f"validator errors: {len(errors)}")
    # end excerpt

    print("== Step 5: clipping norm from per-example gradient norms on the validation rows")
    norms = per_example_grad_norms(make_model(train.x.shape[1], seed), validation)
    print(f"at initialisation: median norm {np.median(norms):.3f}  90th percentile {np.quantile(norms, 0.9):.3f}")
    print(f"fraction clipped at C={CLIP:g}: {np.mean(norms > CLIP):.3f}")
    print(
        f"batch B={BATCH:,} (physical {PHYSICAL_BATCH})  q=1/ceil(n/B)={sample_rate:.4f}  epochs={epochs}  steps={steps}"
    )
    print(f"noise std on the averaged gradient sigma*C/B at sigma=1: {CLIP / BATCH:.4f}")

    print(f"== Step 6: calibrate sigma for epsilon={target_epsilon:g}, delta={delta:.0e}")
    sigma = calibrate(target_epsilon, delta, sample_rate, epochs)
    print(f"sigma={sigma:.3f}")

    print("== Step 7: train with DP-SGD")
    run = train_private(train, sigma=sigma, delta=delta, epochs=epochs, lr=learning_rate, seed=seed)
    print(f"steps={run.steps}  q={run.sample_rate:.4f}")
    print(f"epsilon (Opacus PRV) = {run.epsilon:.2f}")
    print(f"test accuracy at epsilon={target_epsilon:g}: {accuracy(run.model, test):.3f}")

    print("== Step 8: cross-check epsilon")
    # The cross-check gets the planned sigma, q and steps, not the values Opacus recorded, so that a run that did
    # not do what the plan says shows up as a disagreement with Opacus's epsilon. It cannot catch a loader that does
    # not sample by Poisson (poisson_sampling=False): Opacus's accountant records the same history and
    # engine.get_epsilon returns the same epsilon. The record's sampler, read from the loader, shows that bug.
    epsilon_pld, epsilon_rdp = epsilon_pld_rdp(sigma, sample_rate, steps, delta)
    print(f"epsilon PRV (Opacus) = {run.epsilon:.2f}")
    print(f"epsilon PLD (dp_accounting) = {epsilon_pld:.2f}")
    print(f"epsilon RDP (dp_accounting) = {epsilon_rdp:.2f}")

    print("== Step 10: utility against epsilon")
    print(f"non-private: test accuracy {accuracy(non_private, test):.3f}")
    print(f"clip only: test accuracy {accuracy(clip_only.model, test):.3f}")
    for epsilon in BUDGETS:
        if epsilon == target_epsilon:
            budget_run = run
        else:
            budget_sigma = calibrate(epsilon, delta, sample_rate, epochs)
            budget_run = train_private(
                train, sigma=budget_sigma, delta=delta, epochs=epochs, lr=learning_rate, seed=seed
            )
        print(
            f"epsilon={epsilon:g} (sigma={budget_run.sigma:.3f}): test accuracy {accuracy(budget_run.model, test):.3f}"
        )

    print(f"== Step 10: accuracy by sex, exact 95% intervals (non-private and epsilon={target_epsilon:g})")
    for name, groups in (("non-private", by_group(non_private, test)), ("DP", by_group(run.model, test))):
        for g in groups:
            print(f"{name} {g.group}: {g.accuracy:.3f} [{g.low:.3f}, {g.high:.3f}]  ({g.correct:,}/{g.total:,})")
    for plain, private in zip(by_group(non_private, test), by_group(run.model, test), strict=True):
        print(f"cost of privacy, {plain.group}: {plain.accuracy - private.accuracy:+.3f}")

    print("== Step 12: record the run")
    record = Record(
        unit="person (one row of Adult)",
        n=n,
        delta=delta,
        sigma=run.sigma,
        sample_rate=run.sample_rate,
        steps=run.steps,
        epochs=epochs,
        max_grad_norm=CLIP,
        accountant="prv",
        sampler=run.sampler,
        secure_mode=run.secure_mode,
        seed=seed,
        library_versions=library_versions(),
        epsilon_prv=run.epsilon,
        epsilon_pld=epsilon_pld,
        epsilon_rdp=epsilon_rdp,
        tuning={
            "choices": f"learning rate {learning_rate:g} = {DP_RATE_FACTOR}x the non-private {non_private_rate:g}, "
            f"the best of {len(SWEEP_RATES)} non-private runs trained and scored only on the {N_VALIDATION:,} "
            f"public stand-in rows; batch {BATCH:,}, {epochs} epochs, momentum 0.9, C={CLIP:g} fixed in advance",
            "private_runs": f"one run at epsilon={target_epsilon:g}, fixed in advance, is released; the runs at "
            "the other budgets were trained only to draw the utility curve, and nothing (not epsilon, not a "
            "hyperparameter) was chosen from them or from the baselines, none of which is released",
            "counted_in_epsilon": True,
            "not_counted": "",
        },
    )
    write_record(record, record_path)


if __name__ == "__main__":
    main()
