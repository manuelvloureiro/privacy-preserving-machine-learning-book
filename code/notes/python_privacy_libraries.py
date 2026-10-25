"""The computed content of the study note notes/python-privacy-libraries/: the same jobs run in each installed library.

    uv run python code/notes/python_privacy_libraries.py [--seeds 20] [--out DIR]

The note compares the Python packages for privacy-preserving computation. Four of them are installed here, pinned in
uv.lock: OpenDP 0.16.0 with Polars, diffprivlib 0.6.6, Opacus 1.6.0 on CPU PyTorch, and dp-accounting 0.6.0. This
script gives them the same three jobs on scikit-learn's bundled copy of the Breast Cancer Wisconsin (Diagnostic)
dataset (UCI dataset 17; Wolberg, Street and Mangasarian, 1995; licence CC BY 4.0): 569 records, one per patient,
30 measurements of a tumour each, labelled malignant or benign. Every bound the script uses comes from the public
dataset description (the Min and Max columns of its summary table), never from the private values.

1. Release a count and a mean. Each library gets the records, the public bounds of the "mean radius" column and a
   budget of epsilon = 1, split evenly over the two queries. The script prints the noise scale each library chose,
   checks it against the textbook formula, measures the error of 2,000 releases, and shows what happens when a
   third query asks for budget that is no longer there. It also shows OpenDP's public-length trap: declaring the
   number of records public while keeping one contribution per person leaves no neighbouring dataset, and the
   mean is released almost without noise.
2. Train a classifier. Opacus trains a logistic regression with DP-SGD, and diffprivlib trains its naive Bayes and
   random forest, each at several epsilons, against the same models trained without differential privacy (DP).
   diffprivlib's own logistic regression cannot run with the pinned scikit-learn (see `diffprivlib_logistic_regression`).
3. Account one DP-SGD run. The same noise multiplier, sampling rate and step count go to every accountant the
   libraries offer: Opacus's RDP, PRV and GDP accountants, and dp-accounting's RDP and PLD accountants, with the
   PLD's lower estimate as the error bar.

Output: two SVG figures (training.svg, accounting.svg) and three Markdown tables (_release.md, _training.md,
_accounting.md) that the note includes, written to _build/figures/notes/python-privacy-libraries/ (or --out), and
every number the note quotes, printed.
"""

import argparse
import logging
import math
import os
import re
import warnings
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from importlib.metadata import version
from pathlib import Path
from typing import Any, Protocol

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import opendp.prelude as dp
import polars as pl
import torch
from diffprivlib import tools as dp_tools
from diffprivlib.accountant import BudgetAccountant
from diffprivlib.models import GaussianNB as DPGaussianNB
from diffprivlib.models import RandomForestClassifier as DPRandomForest
from diffprivlib.utils import BudgetError
from dp_accounting import GaussianDpEvent, PoissonSampledDpEvent, SelfComposedDpEvent
from dp_accounting.pld import PLDAccountant, privacy_loss_distribution
from dp_accounting.rdp import RdpAccountant
from matplotlib.axes import Axes
from opacus import PrivacyEngine
from opacus.accountants import GaussianAccountant, PRVAccountant, RDPAccountant
from opacus.accountants.utils import get_noise_multiplier
from opendp.context import Query
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from torch import nn

dp.enable_features("contrib")
torch.set_num_threads(1)  # the model is tiny; one thread is as fast and keeps float sums in a fixed order
# Opacus warns that its noise is seeded (wanted here, for reproducible numbers) and about a backward hook on a
# model whose inputs need no gradient; its RDP accountant warns when the best order is the largest it tries, and
# its GDP accountant warns that it can underestimate epsilon, which step 3 shows. None affects the comparison.
warnings.filterwarnings("ignore", message="Secure RNG turned off")
warnings.filterwarnings("ignore", message="Full backward hook")
warnings.filterwarnings("ignore", message="Optimal order is the")
warnings.filterwarnings("ignore", message="GDP accounting")
# dp-accounting's RDP accountant logs, through absl, each fractional order whose series does not converge and is
# dropped; the remaining orders give the same epsilon as Opacus's accountant with the same orders.
logging.getLogger("absl").setLevel(logging.ERROR)

matplotlib.rcParams["svg.hashsalt"] = "python_privacy_libraries"  # stable element ids, so reruns don't churn the SVGs
matplotlib.rcParams["svg.fonttype"] = "none"  # keep text as text: smaller, diffable files

NOTE = "python-privacy-libraries"
FIGURES = Path(__file__).resolve().parents[2] / "_build" / "figures" / "notes" / NOTE

Array = npt.NDArray[np.float64]
Labels = npt.NDArray[np.int64]

COLUMN = "mean radius"  # the column the statistics are about
RELEASE_EPSILON = 1.0  # the budget for the count and the mean together
DRAWS = 2_000  # releases per implementation in the empirical check

TEST_SHARE = 0.2
SPLIT_SEED = 0
BATCH = 32
EPOCHS = 20
CLIP = 1.0
LR = 0.5
MOMENTUM = 0.9
EPSILONS = [0.1, 0.3, 1.0, 3.0, 10.0]
ACCOUNTING_SIGMA = 1.0  # the noise multiplier of the run that step 3 accounts
PLD_GRID = 1e-4  # the PLD accountants' discretization in the table (the figure uses a coarser 1e-3, for speed)
MAX_EPOCHS = 40  # the accounting figure's range

COLOURS = {
    "DP-SGD logistic regression (Opacus)": "tab:blue",
    "Naive Bayes (diffprivlib)": "tab:orange",
    "Random forest (diffprivlib)": "tab:green",
}


# --- The data and its public facts ------------------------------------------------------------------------------


def public_bounds(description: str) -> dict[str, tuple[float, float]]:
    """The Min and Max of each of the 30 features, read from the dataset's published summary table, keyed by the
    column names scikit-learn uses ("radius (mean)" in the table is "mean radius" in the data)."""
    rows = re.findall(r"^([a-z ]+?) \((mean|standard error|worst)\):\s+(\S+)\s+(\S+)\s*$", description, re.MULTILINE)
    prefix = {"mean": "mean", "standard error": "", "worst": "worst"}
    bounds: dict[str, tuple[float, float]] = {}
    for name, kind, low, high in rows:
        key = f"{prefix[kind]} {name}".strip() if kind != "standard error" else f"{name} error"
        bounds[key] = (float(low), float(high))
    return bounds


@dataclass(frozen=True)
class Data:
    """The dataset as the jobs see it: the raw column for the statistics, and the scaled split for the models."""

    column: Array  # the private values of COLUMN
    bounds: tuple[float, float]  # the public bounds of COLUMN
    x_train: Array  # features scaled into [0, 1] by the public bounds
    y_train: Labels
    x_test: Array
    y_test: Labels
    outside: int  # raw values outside the published bounds across all features, which clipping moved


def load() -> Data:
    """Load the dataset, scale every feature into [0, 1] with the public bounds, and split it 80/20."""
    bunch = load_breast_cancer()
    names = [str(n) for n in bunch.feature_names]
    bounds = public_bounds(bunch.DESCR)
    low = np.array([bounds[n][0] for n in names])
    high = np.array([bounds[n][1] for n in names])
    x = bunch.data
    outside = int(((x < low) | (x > high)).sum())
    scaled = np.clip((x - low) / (high - low), 0.0, 1.0)
    x_train, x_test, y_train, y_test = train_test_split(
        scaled, bunch.target, test_size=TEST_SHARE, random_state=SPLIT_SEED, stratify=bunch.target
    )
    column = x[:, names.index(COLUMN)].astype(np.float64)
    return Data(column, bounds[COLUMN], x_train, y_train, x_test, y_test, outside)


def delta_for(n: int) -> float:
    """The book's rule for delta: 1/n^1.1, rounded down to a power of ten."""
    return 10.0 ** math.floor(math.log10(n**-1.1))


# --- Job 1: release a count and a mean ------------------------------------------------------------------------------


@dataclass(frozen=True)
class Release:
    count: float
    mean: float
    count_scale: float  # the Laplace scale the implementation used, or implied, for the count
    mean_scale: float


def laplace_scales(n: int, bounds: tuple[float, float], eps: float) -> tuple[float, float]:
    """The textbook Laplace scales for a count (sensitivity 1) and a mean of n values clamped to `bounds`
    (sensitivity (high - low)/n), each at privacy loss `eps`."""
    low, high = bounds
    return 1.0 / eps, (high - low) / (n * eps)


# excerpt: ppl-by-hand
def by_hand(values: Array, bounds: tuple[float, float], eps: float, rng: np.random.Generator) -> Release:
    """Textbook Laplace mechanism in NumPy: half the budget to the count, half to the clamped mean."""
    n = len(values)
    count_scale, mean_scale = laplace_scales(n, bounds, eps / 2)
    count = n + rng.laplace(scale=count_scale)
    mean = float(np.clip(values, *bounds).mean()) + rng.laplace(scale=mean_scale)
    return Release(count, mean, count_scale, mean_scale)


# end excerpt


def vector_query(context: dp.Context) -> Query:
    """The query builder of a Context over plain (non-Polars) data."""
    query = context.query()
    assert isinstance(query, Query)
    return query


# excerpt: ppl-opendp
def with_opendp(values: Array, bounds: tuple[float, float], eps: float) -> Release:
    """OpenDP's Context API: the budget is declared once, and each query takes an even share of it."""
    context = dp.Context.compositor(
        data=[float(v) for v in values],
        privacy_unit=dp.unit_of(contributions=1),  # neighbours add or remove one person's record
        privacy_loss=dp.loss_of(epsilon=eps),
        split_evenly_over=2,
        domain=dp.vector_domain(dp.atom_domain(T=float, nan=False)),
    )
    count = vector_query(context).count().laplace()  # OpenDP picks the scale that spends exactly eps/2
    mean = vector_query(context).clamp(bounds).resize(size=len(values), constant=bounds[0]).mean().laplace()
    return Release(count.release(), mean.release(), count.param(), mean.param())


# end excerpt


# excerpt: ppl-diffprivlib
def with_diffprivlib(values: Array, bounds: tuple[float, float], eps: float, seed: int) -> Release:
    """diffprivlib's NumPy-like tools: each call takes its own epsilon, and a BudgetAccountant adds them up."""
    accountant = BudgetAccountant(epsilon=eps)  # refuses a query once the total would pass eps
    rng = np.random.RandomState(seed)
    count = dp_tools.count_nonzero(values > -np.inf, epsilon=eps / 2, random_state=rng, accountant=accountant)
    mean = dp_tools.mean(values, epsilon=eps / 2, bounds=bounds, random_state=rng, accountant=accountant)
    count_scale, mean_scale = laplace_scales(len(values), bounds, eps / 2)  # what its mechanisms use
    return Release(float(count), float(mean), count_scale, mean_scale)


# end excerpt


def opendp_mechanisms(
    bounds: tuple[float, float], n: int, scales: tuple[float, float]
) -> tuple[dp.Measurement, dp.Measurement]:
    """The count and mean mechanisms of `with_opendp` as plain OpenDP measurements with the given scales, so that
    the empirical check can call them thousands of times without a Context, which would stop at its budget."""
    domain = dp.vector_domain(dp.atom_domain(T=float, nan=False))
    metric = dp.symmetric_distance()
    count = dp.t.make_count(domain, metric)
    clamp = dp.t.make_clamp(domain, metric, bounds)
    resize = dp.t.make_resize(clamp.output_domain, clamp.output_metric, n, bounds[0])
    mean = dp.t.make_mean(resize.output_domain, resize.output_metric)
    return (
        count >> dp.m.make_laplace(count.output_domain, count.output_metric, scales[0]),
        clamp >> (resize >> (mean >> dp.m.make_laplace(mean.output_domain, mean.output_metric, scales[1]))),
    )


def laplace_half_width(scale: float, alpha: float = 0.05) -> float:
    """Half the width of a (1 - alpha) interval for Laplace noise of this scale: scale * ln(1/alpha)."""
    return scale * math.log(1.0 / alpha)


class Releaser(Protocol):
    def __call__(self, seed: int, /) -> Release: ...


def empirical_errors(release: Releaser, truth: tuple[float, float], draws: int) -> tuple[float, float, float, float]:
    """Over `draws` releases: the mean absolute error and the 95th percentile of the absolute error, for the count and
    for the mean. The textbook values for Laplace noise of scale b are b and b ln 20."""
    count_errors = np.empty(draws)
    mean_errors = np.empty(draws)
    for seed in range(draws):
        r = release(seed)
        count_errors[seed] = abs(r.count - truth[0])
        mean_errors[seed] = abs(r.mean - truth[1])
    return (
        float(count_errors.mean()),
        float(np.quantile(count_errors, 0.95)),
        float(mean_errors.mean()),
        float(np.quantile(mean_errors, 0.95)),
    )


def opendp_refusal(values: Array, bounds: tuple[float, float], eps: float) -> str:
    """What OpenDP does when a third query asks for budget the Context no longer has."""
    context = dp.Context.compositor(
        data=[float(v) for v in values],
        privacy_unit=dp.unit_of(contributions=1),
        privacy_loss=dp.loss_of(epsilon=eps),
        split_evenly_over=2,
        domain=dp.vector_domain(dp.atom_domain(T=float, nan=False)),
    )
    vector_query(context).count().laplace().release()
    vector_query(context).clamp(bounds).sum().laplace().release()
    try:
        vector_query(context).count().laplace().release()
    except Exception as error:  # noqa: BLE001  (the note reports the exception's name)
        return f"{type(error).__name__}: {str(error).strip().splitlines()[0]}"
    return "released"


def diffprivlib_refusal(values: Array, bounds: tuple[float, float], eps: float) -> str:
    """What diffprivlib does when a third query would take the BudgetAccountant past its epsilon."""
    accountant = BudgetAccountant(epsilon=eps)
    rng = np.random.RandomState(0)
    dp_tools.count_nonzero(values > -np.inf, epsilon=eps / 2, random_state=rng, accountant=accountant)
    dp_tools.mean(values, epsilon=eps / 2, bounds=bounds, random_state=rng, accountant=accountant)
    try:
        dp_tools.count_nonzero(values > -np.inf, epsilon=eps / 2, random_state=rng, accountant=accountant)
    except BudgetError as error:
        return f"BudgetError: {error}"
    return "released"


# excerpt: ppl-opendp-length
def opendp_public_length(values: Array, bounds: tuple[float, float], eps: float, contributions: int) -> float:
    """The noise scale OpenDP's Polars plan gives the sum inside a mean when the number of records is declared
    public (`invariant="lengths"`), for a privacy unit of `contributions` rows."""
    frame = pl.DataFrame({COLUMN: values}).lazy()
    context = dp.Context.compositor(
        data=frame,
        privacy_unit=dp.unit_of(contributions=contributions),
        privacy_loss=dp.loss_of(epsilon=eps),
        split_evenly_over=1,
        margins=[dp.polars.Margin(by=[], max_length=len(values), invariant="lengths")],
    )
    column: Any = pl.col(COLUMN)  # typed Any because OpenDP adds the `.dp` namespace to Polars expressions at run time
    query = context.query().select(column.dp.mean(bounds))
    plan = query.summarize()  # one row per noisy quantity: the sum and the length
    return float(plan.filter(pl.col("aggregate") == "Sum")["scale"].item())


# end excerpt


def release_table(releases: dict[str, Release]) -> str:
    """The Markdown table of job 1: what each implementation needed and the scale it used."""
    told = {
        "By hand (NumPy)": "the bounds and $n$, and the formula for each scale",
        "OpenDP": "the bounds, $n$ for the mean, the unit (one record) and the total budget",
        "diffprivlib": "the bounds and an $\\varepsilon$ per call; $n$ it reads from the array",
    }
    lines = [
        "| Implementation | Count scale | Mean scale | 95% half-width of the mean | What it had to be told |",
        "|:-------|------:|------:|---------:|:------------------------------|",
    ]
    for name, r in releases.items():
        lines.append(
            f"| {name} | {r.count_scale:.4f} | {r.mean_scale:.4f} | {laplace_half_width(r.mean_scale):.3f} | {told[name]} |"
        )
    lines.append("")
    lines.append(
        f": The count and the mean of the {COLUMN} column at $\\varepsilon = {RELEASE_EPSILON:g}$, half to each "
        f"query, in three implementations. The half-width is for Laplace noise of the mean's scale. "
        "{#tbl-release}"
    )
    return "\n".join(lines) + "\n"


# --- Job 2: train a classifier ----------------------------------------------------------------------------------


@dataclass(frozen=True)
class Run:
    technique: str
    epsilon: float | None  # None: trained without DP
    seed: int
    accuracy: float
    reported_epsilon: float | None  # what the library's accountant reports, when it has one


def sampling_rate(n: int, batch: int) -> float:
    """Opacus's sampling rate for an expected batch of `batch` records: 1/ceil(n/B)."""
    return 1.0 / math.ceil(n / batch)


def noise_multiplier(eps: float, n: int, delta: float) -> float:
    """The noise multiplier that makes DP-SGD with these settings (eps, delta)-DP, by Opacus's PRV accountant."""
    return float(
        get_noise_multiplier(
            target_epsilon=eps, target_delta=delta, sample_rate=sampling_rate(n, BATCH), epochs=EPOCHS, accountant="prv"
        )
    )


# excerpt: ppl-opacus
def train_dp_sgd(data: Data, sigma: float | None, seed: int, delta: float) -> tuple[float, float | None]:
    """A logistic regression trained by SGD in PyTorch; with a noise multiplier, Opacus makes it DP-SGD."""
    torch.manual_seed(seed)
    x = torch.tensor(data.x_train, dtype=torch.float32)
    y = torch.tensor(data.y_train)
    loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x, y), batch_size=BATCH, shuffle=True)
    model: nn.Module = nn.Linear(x.shape[1], 2)
    optimizer: torch.optim.Optimizer = torch.optim.SGD(model.parameters(), lr=LR, momentum=MOMENTUM)
    engine: PrivacyEngine | None = None
    if sigma is not None:
        engine = PrivacyEngine(accountant="prv")
        model, optimizer, loader = engine.make_private(  # Poisson sampling, per-example clipping, Gaussian noise
            module=model, optimizer=optimizer, data_loader=loader, noise_multiplier=sigma, max_grad_norm=CLIP
        )
    for _ in range(EPOCHS):
        for xb, yb in loader:
            optimizer.zero_grad()
            nn.functional.cross_entropy(model(xb), yb).backward()
            optimizer.step()
    with torch.no_grad():
        predicted = model(torch.tensor(data.x_test, dtype=torch.float32)).argmax(dim=1).numpy()
    return float((predicted == data.y_test).mean()), engine.get_epsilon(delta) if engine else None


# end excerpt


# excerpt: ppl-diffprivlib-models
def train_naive_bayes(data: Data, eps: float | None, seed: int) -> float:
    """scikit-learn's Gaussian naive Bayes, or diffprivlib's, which perturbs each class's means and variances."""
    d = data.x_train.shape[1]
    if eps is None:
        return GaussianNB().fit(data.x_train, data.y_train).score(data.x_test, data.y_test)
    model = DPGaussianNB(epsilon=eps, bounds=(np.zeros(d), np.ones(d)), random_state=seed)  # public bounds
    return model.fit(data.x_train, data.y_train).score(data.x_test, data.y_test)


def train_random_forest(data: Data, eps: float | None, seed: int) -> float:
    """scikit-learn's random forest, or diffprivlib's, whose trees are built blind and whose leaves vote with noise."""
    d = data.x_train.shape[1]
    if eps is None:
        return RandomForestClassifier(random_state=seed).fit(data.x_train, data.y_train).score(data.x_test, data.y_test)
    model = DPRandomForest(epsilon=eps, bounds=(np.zeros(d), np.ones(d)), classes=[0, 1], random_state=seed)
    return model.fit(data.x_train, data.y_train).score(data.x_test, data.y_test)


# end excerpt


def diffprivlib_logistic_regression() -> str:
    """Why the note does not train diffprivlib's logistic regression: its constructor passes `multi_class` to
    scikit-learn, which removed that argument in 1.8, so with the pinned scikit-learn it fails before fitting."""
    from diffprivlib.models import LogisticRegression as DPLogisticRegression

    try:
        DPLogisticRegression(epsilon=1.0, data_norm=1.0)
    except TypeError as error:
        return f"TypeError: {error}"
    return "constructed"


_data: Data | None = None


def _init(data: Data) -> None:
    global _data
    _data = data


def evaluate(job: tuple[str, float | None, int, float]) -> Run:
    """One training run: a technique, an epsilon (None for no DP), a seed, and delta for DP-SGD."""
    technique, eps, seed, delta = job
    assert _data is not None
    reported: float | None = None
    if technique.startswith("DP-SGD"):
        sigma = None if eps is None else noise_multiplier(eps, len(_data.y_train), delta)
        accuracy, reported = train_dp_sgd(_data, sigma, seed, delta)
    elif technique.startswith("Naive Bayes"):
        accuracy = train_naive_bayes(_data, eps, seed)
    else:
        accuracy = train_random_forest(_data, eps, seed)
    return Run(technique, eps, seed, accuracy, reported)


@dataclass(frozen=True)
class Summary:
    technique: str
    epsilon: float | None
    mean: float
    low: float  # 10th percentile over seeds
    high: float  # 90th percentile


def summarize(runs: list[Run]) -> list[Summary]:
    out: list[Summary] = []
    for technique in COLOURS:
        for eps in [None, *EPSILONS]:
            acc = np.array([r.accuracy for r in runs if r.technique == technique and r.epsilon == eps])
            out.append(
                Summary(technique, eps, float(acc.mean()), float(np.quantile(acc, 0.1)), float(np.quantile(acc, 0.9)))
            )
    return out


def training_table(summaries: list[Summary]) -> str:
    """The Markdown table of job 2: mean test accuracy by technique and epsilon."""
    header = "| Technique | " + " | ".join(f"$\\varepsilon = {e:g}$" for e in EPSILONS) + " | No DP |"
    lines = [header, "|:---|" + "---:|" * (len(EPSILONS) + 1)]
    for technique in COLOURS:
        by_eps = {s.epsilon: s for s in summaries if s.technique == technique}
        cells = [f"{by_eps[e].mean:.3f}" for e in EPSILONS] + [f"{by_eps[None].mean:.3f}"]
        lines.append(f"| {technique} | " + " | ".join(cells) + " |")
    lines.append("")
    lines.append(
        ": Mean test accuracy over the seeds, by technique and privacy level; DP-SGD at $\\delta = 10^{-3}$, "
        "the diffprivlib models with pure $\\varepsilon$-DP. {#tbl-training}"
    )
    return "\n".join(lines) + "\n"


def figure_training(summaries: list[Summary], out: Path) -> None:
    """Test accuracy against epsilon, one line per technique with its 10th-90th percentile band, and each
    technique's accuracy without DP as a dashed line of the same colour."""
    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    ax_t: Axes = ax
    for technique, colour in COLOURS.items():
        private = [s for s in summaries if s.technique == technique and s.epsilon is not None]
        xs = [s.epsilon for s in private if s.epsilon is not None]
        ax_t.plot(xs, [s.mean for s in private], marker="o", color=colour, label=technique)
        ax_t.fill_between(xs, [s.low for s in private], [s.high for s in private], color=colour, alpha=0.15)
        baseline = next(s for s in summaries if s.technique == technique and s.epsilon is None)
        ax_t.axhline(baseline.mean, color=colour, linestyle="--", linewidth=1)
    ax_t.set_xscale("log")
    ax_t.set_xlabel(r"privacy level $\varepsilon$ (DP-SGD at $\delta = 10^{-3}$)")
    ax_t.set_ylabel("test accuracy")
    ax_t.set_ylim(0.45, 1.0)
    ax_t.grid(alpha=0.3)
    ax_t.legend(loc="lower right", fontsize=9)
    fig.tight_layout()
    fig.savefig(out / "training.svg", metadata={"Date": None})
    plt.close(fig)


# --- Job 3: account one DP-SGD run -------------------------------------------------------------------------------


# excerpt: ppl-accountants
def accountants(sigma: float, q: float, steps: int, delta: float, grid: float = PLD_GRID) -> dict[str, float]:
    """Epsilon for one DP-SGD run (noise multiplier sigma, sampling rate q, `steps` steps) by every accountant the
    two libraries offer, plus the PLD's lower estimate, which no valid upper bound may fall below."""
    out: dict[str, float] = {}
    for name, accountant in [("Opacus RDP", RDPAccountant()), ("Opacus PRV", PRVAccountant())]:
        accountant.history = [(sigma, q, steps)]  # what `steps` calls of .step() would have recorded
        out[name] = accountant.get_epsilon(delta)
    gdp = GaussianAccountant()
    gdp.history = [(sigma, q, steps)]
    out["Opacus GDP"] = gdp.get_epsilon(delta)
    run = SelfComposedDpEvent(PoissonSampledDpEvent(q, GaussianDpEvent(sigma)), steps)
    out["dp-accounting RDP"] = RdpAccountant().compose(run).get_epsilon(delta)
    out["dp-accounting PLD"] = PLDAccountant(value_discretization_interval=grid).compose(run).get_epsilon(delta)
    optimistic = privacy_loss_distribution.from_gaussian_mechanism(
        sigma, sampling_prob=q, pessimistic_estimate=False, value_discretization_interval=grid, use_connect_dots=False
    )
    out["PLD lower estimate"] = optimistic.self_compose(steps).get_epsilon_for_delta(delta)
    return out


# end excerpt


def accounting_table(epsilons: dict[str, float], sigma: float, q: float, steps: int, delta: float) -> str:
    kind = {
        "Opacus RDP": "upper bound",
        "Opacus PRV": "upper bound, numerical",
        "Opacus GDP": "approximation, not a bound",
        "dp-accounting RDP": "upper bound",
        "dp-accounting PLD": "upper bound, numerical",
        "PLD lower estimate": "lower bound",
    }
    lines = ["| Accountant | $\\varepsilon$ | What the number is |", "|:---|---:|:---|"]
    for name, eps in epsilons.items():
        lines.append(f"| {name} | {eps:.3f} | {kind[name]} |")
    lines.append("")
    lines.append(
        f": One DP-SGD run, noise multiplier {sigma:g}, sampling rate {q:.4f}, {steps} steps, under every accountant "
        f"the libraries offer, at $\\delta = 10^{{{round(math.log10(delta))}}}$. {{#tbl-accounting}}"
    )
    return "\n".join(lines) + "\n"


def figure_accounting(sigma: float, q: float, delta: float, out: Path) -> None:
    """Epsilon against epochs for each accountant, on the run of job 3."""
    steps_per_epoch = round(1 / q)
    epochs = list(range(2, MAX_EPOCHS + 1, 2))
    curves: dict[str, list[float]] = {}
    for e in epochs:
        for name, eps in accountants(sigma, q, e * steps_per_epoch, delta, grid=1e-3).items():
            if name != "PLD lower estimate":  # on this coarser grid it sits well below the table's value
                curves.setdefault(name, []).append(eps)
    styles = {
        "Opacus RDP": ("tab:red", "-"),
        "dp-accounting RDP": ("tab:red", ":"),
        "Opacus PRV": ("tab:blue", "-"),
        "dp-accounting PLD": ("tab:blue", ":"),
        "Opacus GDP": ("tab:purple", "-."),
    }
    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    ax_t: Axes = ax
    for name, values in curves.items():
        colour, style = styles[name]
        ax_t.plot(epochs, values, color=colour, linestyle=style, label=name)
    ax_t.set_xlabel(f"epochs (noise multiplier {sigma:g}, sampling rate {q:.4f})")
    ax_t.set_ylabel(r"$\varepsilon$ at $\delta = 10^{-3}$")
    ax_t.grid(alpha=0.3)
    ax_t.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(out / "accounting.svg", metadata={"Date": None})
    plt.close(fig)


# --- Main ----------------------------------------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seeds", type=int, default=20, help="training runs per technique and epsilon")
    parser.add_argument("--draws", type=int, default=DRAWS, help="releases per implementation in the empirical check")
    parser.add_argument("--out", type=Path, default=FIGURES)
    args = parser.parse_args()
    out: Path = args.out
    out.mkdir(parents=True, exist_ok=True)
    seeds: int = args.seeds
    draws: int = args.draws

    print(
        f"versions: opendp {version('opendp')}  diffprivlib {version('diffprivlib')}  opacus {version('opacus')}  "
        f"dp-accounting {version('dp-accounting')}  scikit-learn {version('scikit-learn')}  torch {version('torch')}"
    )
    data = load()
    n = len(data.column)
    low, high = data.bounds
    truth = (float(n), float(np.clip(data.column, low, high).mean()))
    print(f"records {n}  train {len(data.y_train)}  test {len(data.y_test)}  features {data.x_train.shape[1]}")
    majority = max(float(data.y_test.mean()), 1 - float(data.y_test.mean()))
    print(f"majority class share of the test set: {majority:.3f} (the accuracy of always predicting it)")
    print(f"{COLUMN}: public bounds [{low}, {high}]  true mean {truth[1]:.4f}")
    print(f"raw values outside the published bounds, over all features: {data.outside}")

    print("== Job 1: release a count and a mean")
    releases = {
        "By hand (NumPy)": by_hand(data.column, data.bounds, RELEASE_EPSILON, np.random.default_rng(0)),
        "OpenDP": with_opendp(data.column, data.bounds, RELEASE_EPSILON),
        "diffprivlib": with_diffprivlib(data.column, data.bounds, RELEASE_EPSILON, 0),
    }
    count_scale, mean_scale = laplace_scales(n, data.bounds, RELEASE_EPSILON / 2)
    print(
        f"textbook scales: count {count_scale:.4f}  mean {mean_scale:.6f}  mean 95% half-width {laplace_half_width(mean_scale):.4f}"
    )
    for name, r in releases.items():
        print(
            f"{name:16s} count scale {r.count_scale:.4f}  mean scale {r.mean_scale:.6f}  one release: count {r.count:.1f}  mean {r.mean:.4f}"
        )
    print(
        f"OpenDP 95% half-width of the count: {dp.laplacian_scale_to_accuracy(releases['OpenDP'].count_scale, 0.05):.4f}"
    )
    print(f"== empirical errors over {draws} releases (mean |error|, 95th percentile), textbook b and b ln 20")
    print(
        f"textbook: count {count_scale:.3f} {laplace_half_width(count_scale):.3f}  mean {mean_scale:.4f} {laplace_half_width(mean_scale):.4f}"
    )
    opendp_count, opendp_mean = opendp_mechanisms(
        data.bounds, n, (releases["OpenDP"].count_scale, releases["OpenDP"].mean_scale)
    )
    values = [float(v) for v in data.column]
    releasers: dict[str, Releaser] = {
        "By hand (NumPy)": lambda seed: by_hand(data.column, data.bounds, RELEASE_EPSILON, np.random.default_rng(seed)),
        "OpenDP": lambda _seed: Release(opendp_count(values), opendp_mean(values), 0.0, 0.0),  # cannot be seeded
        "diffprivlib": lambda seed: with_diffprivlib(data.column, data.bounds, RELEASE_EPSILON, seed),
    }
    for name, release in releasers.items():
        c_mean, c_q95, m_mean, m_q95 = empirical_errors(release, truth, draws)
        print(f"{name:16s} count {c_mean:.3f} {c_q95:.3f}  mean {m_mean:.4f} {m_q95:.4f}")
    print("== a third query after the budget is spent")
    print(f"OpenDP: {opendp_refusal(data.column, data.bounds, RELEASE_EPSILON)}")
    print(f"diffprivlib: {diffprivlib_refusal(data.column, data.bounds, RELEASE_EPSILON)}")
    print("== OpenDP with the number of records declared public: scale of the sum inside the mean")
    for contributions in (1, 2):
        scale = opendp_public_length(data.column, data.bounds, RELEASE_EPSILON, contributions)
        print(
            f"contributions={contributions}: sum scale {scale:.4g}  (as a mean: {scale / n:.4g}; textbook (high-low)/(n eps) = {(high - low) / (n * RELEASE_EPSILON):.4g})"
        )
    (out / "_release.md").write_text(release_table(releases))

    print("== Job 2: train a classifier")
    n_train = len(data.y_train)
    delta = delta_for(n_train)
    q = sampling_rate(n_train, BATCH)
    steps = int(EPOCHS / q)
    print(f"delta {delta:g} (1/n^1.1 = {n_train**-1.1:.2e})  sampling rate {q:.4f}  steps {steps}")
    for eps in EPSILONS:
        print(f"epsilon {eps:g}: noise multiplier {noise_multiplier(eps, n_train, delta):.3f}")
    print(
        f"diffprivlib LogisticRegression with scikit-learn {version('scikit-learn')}: {diffprivlib_logistic_regression()}"
    )
    jobs: list[tuple[str, float | None, int, float]] = [
        (technique, eps, seed, delta) for technique in COLOURS for eps in [None, *EPSILONS] for seed in range(seeds)
    ]
    with ProcessPoolExecutor(max_workers=os.cpu_count(), initializer=_init, initargs=(data,)) as pool:
        runs = list(pool.map(evaluate, jobs, chunksize=4))
    summaries = summarize(runs)
    for s in summaries:
        eps_label = "no DP" if s.epsilon is None else f"eps {s.epsilon:g}"
        print(f"{s.technique:40s} {eps_label:8s} accuracy {s.mean:.3f} [{s.low:.3f}, {s.high:.3f}]")
    reported = [r.reported_epsilon for r in runs if r.technique.startswith("DP-SGD") and r.epsilon == 1.0]
    print(
        f"Opacus reports for the eps=1 runs: {min(x for x in reported if x is not None):.3f} to {max(x for x in reported if x is not None):.3f}"
    )
    (out / "_training.md").write_text(training_table(summaries))
    figure_training(summaries, out)

    print("== Job 3: account one DP-SGD run")
    epsilons = accountants(ACCOUNTING_SIGMA, q, steps, delta)
    for name, eps in epsilons.items():
        print(f"{name:20s} {eps:.4f}")
    (out / "_accounting.md").write_text(accounting_table(epsilons, ACCOUNTING_SIGMA, q, steps, delta))
    figure_accounting(ACCOUNTING_SIGMA, q, delta, out)


if __name__ == "__main__":
    main()
