"""The worked example of the study note notes/privacy-utility-tradeoff/: a privacy-utility frontier on Adult.

    uv run python code/notes/privacy_utility_tradeoff.py [--seeds 20] [--out DIR]

The use case. A bank wants to send a premium-card offer to the customers who earn more than $50,000 a year, and it
will only train a targeting model on its customer records with differential privacy (DP). Adult stands in for those
records: 48,842 people from the 1994 US Current Population Survey, one row each, labelled by whether their income
exceeds $50,000 (UCI Machine Learning Repository, dataset 2; Becker & Kohavi, 1996; licence CC BY 4.0), loaded by
code/practitioner_datasets.py. The privacy unit is the row, that is, one customer. The training file (32,561 rows)
plays the bank's customer records and the test file (16,281 rows) the customers who receive the next campaign.

What the script computes, for each technique and each epsilon, over `--seeds` random seeds:

- technical metrics on the campaign customers: AUC, and accuracy at the 0.5 threshold;
- business metrics for a campaign that sends as many offers as there are eligible customers in expectation: the
  share of eligible customers reached, the number reached, and the number of offers wasted on ineligible customers;
- the privacy statement of each epsilon: the highest belief that an attacker who starts at even odds can reach about
  one customer's presence, and the bound on a membership attack's true-positive rate at a 1% false-positive rate.

The two techniques are DP-SGD (a logistic regression trained with Opacus: Poisson sampling, per-example clipping,
Gaussian noise calibrated by the PRV accountant to the target epsilon at delta = 1e-5) and diffprivlib's GaussianNB
(pure epsilon-DP, which is also (epsilon, 1e-5)-DP), each against the same model trained without DP. Both protect the
same unit, one customer, so on the axis (epsilon at delta = 1e-5) they are directly comparable.

Nothing is tuned on the private data. The preprocessing uses only the survey's codebook (adult.names): the category
lists and public bounds on each number. The models use their library defaults. The offer capacity uses the share of
high earners, which the Census Bureau publishes; the script reads it from the training file only to state it.

Output: two SVG figures and three Markdown snippets (_frontier.md, _menu.md, _decision.md) that the note includes,
written to _build/figures/notes/privacy-utility-tradeoff/ (or --out), and the full results printed as a table.

Library versions: Opacus with CPU PyTorch, diffprivlib 0.6.6 with scikit-learn 1.8.0 (pinned in uv.lock).
"""

import argparse
import math
import os
import sys
import warnings
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import polars as pl
import torch
from diffprivlib.models import GaussianNB as DPGaussianNB
from matplotlib.axes import Axes
from opacus import PrivacyEngine
from opacus.accountants.utils import get_noise_multiplier
from sklearn.metrics import roc_auc_score
from sklearn.naive_bayes import GaussianNB
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import practitioner_datasets

torch.set_num_threads(1)  # the model is tiny; one thread is as fast and keeps float sums in a fixed order
# Opacus warns that its noise is seeded (wanted here, for reproducible numbers) and about a backward hook on a
# model whose inputs need no gradient; neither affects the guarantee.
warnings.filterwarnings("ignore", message="Secure RNG turned off")
warnings.filterwarnings("ignore", message="Full backward hook")
warnings.filterwarnings("ignore", message="Optimal order is the")

matplotlib.rcParams["svg.hashsalt"] = "privacy_utility_tradeoff"  # stable element ids, so reruns don't churn the SVGs
matplotlib.rcParams["svg.fonttype"] = "none"  # keep text as text: smaller, diffable files

NOTE = "privacy-utility-tradeoff"
FIGURES = Path(__file__).resolve().parents[2] / "_build" / "figures" / "notes" / NOTE

Array = npt.NDArray[np.float64]

TRAIN_ROWS = 32_561  # adult.data; the remaining 16,281 rows are adult.test
EPSILONS = [0.01, 0.02, 0.03, 0.05, 0.1, 0.2, 0.3, 1.0, 3.0, 10.0]
SIZES = [2_000, 8_000, TRAIN_ROWS]  # customer records the bank trains on, for the data-size sweep (DP-SGD only)
MENU_EPSILONS = [0.1, 0.3, 1.0, 3.0]
DELTA = 1e-5  # 1/n^1.1 for n = 32,561 customers is 1.1e-5, rounded down to a power of ten
# DP-SGD settings, fixed before training (not tuned on the private data): 33 Poisson batches per epoch.
BATCH, EPOCHS, CLIP, LR, MOMENTUM = 1_000, 10, 1.0, 0.5, 0.9
CEILING_BELIEF = 0.75  # the use case's policy: an attacker at even odds must stay at or below 75%
FPR = 0.01  # the false-positive rate at which membership attacks are judged

# The codebook (adult.names): category lists and public bounds, never read from the data.
CATEGORIES: dict[str, list[str]] = {
    "workclass": [
        "Private", "Self-emp-not-inc", "Self-emp-inc", "Federal-gov", "Local-gov", "State-gov", "Without-pay",
        "Never-worked",
    ],
    "marital-status": [
        "Married-civ-spouse", "Divorced", "Never-married", "Separated", "Widowed", "Married-spouse-absent",
        "Married-AF-spouse",
    ],
    "occupation": [
        "Tech-support", "Craft-repair", "Other-service", "Sales", "Exec-managerial", "Prof-specialty",
        "Handlers-cleaners", "Machine-op-inspct", "Adm-clerical", "Farming-fishing", "Transport-moving",
        "Priv-house-serv", "Protective-serv", "Armed-Forces",
    ],
    "relationship": ["Wife", "Own-child", "Husband", "Not-in-family", "Other-relative", "Unmarried"],
    "race": ["White", "Asian-Pac-Islander", "Amer-Indian-Eskimo", "Other", "Black"],
    "sex": ["Female", "Male"],
}  # fmt: skip
# Linear scaling to [0, 1] within the survey's ranges (age 17-90, education 1-16, hours 1-99).
LINEAR_BOUNDS: dict[str, tuple[float, float]] = {"age": (17, 90), "education-num": (1, 16), "hours-per-week": (1, 99)}
# Capital gains and losses are skewed: log-scale them against the survey's top code, 99,999.
LOG_COLUMNS = ["capital-gain", "capital-loss"]
TOP_CODE = 99_999

# Each row has at most one 1 per categorical column and numbers in [0, 1], so its Euclidean norm is at most this.
DATA_NORM = math.sqrt(len(LINEAR_BOUNDS) + len(LOG_COLUMNS) + len(CATEGORIES))


# --- Data ------------------------------------------------------------------------------------------------------


# excerpt: put-features
def features(df: pl.DataFrame) -> Array:
    """Map each row to [0, 1]^d using only the codebook; a missing category is a row of zeros."""
    cols: list[Array] = []
    for name, (lo, hi) in LINEAR_BOUNDS.items():
        x = df[name].cast(pl.Float64).to_numpy()
        cols.append(np.clip((x - lo) / (hi - lo), 0.0, 1.0))
    for name in LOG_COLUMNS:
        x = df[name].cast(pl.Float64).to_numpy()
        cols.append(np.clip(np.log1p(np.clip(x, 0, TOP_CODE)) / math.log1p(TOP_CODE), 0.0, 1.0))
    for name, values in CATEGORIES.items():
        column = df[name].to_list()
        for value in values:
            cols.append(np.array([1.0 if v == value else 0.0 for v in column]))
    return np.column_stack(cols)


# end excerpt


def labels(df: pl.DataFrame) -> npt.NDArray[np.int64]:
    return (df["income"] == ">50K").cast(pl.Int64).to_numpy()


# --- Metrics ---------------------------------------------------------------------------------------------------


# excerpt: put-reached
def reached(scores: Array, y: npt.NDArray[np.int64], offers: int) -> int:
    """Eligible customers among the `offers` highest-scored; ties are broken by position, deterministically."""
    top = np.argsort(-scores, kind="stable")[:offers]
    return int(y[top].sum())


# end excerpt


# excerpt: put-belief
def belief_ceiling(eps: float, prior: float = 0.5) -> float:
    """Highest posterior that epsilon-DP allows an attacker who starts at `prior` that one customer is in the data."""
    odds = math.exp(eps) * prior / (1 - prior)
    return odds / (1 + odds)


def epsilon_for_belief(belief: float, prior: float = 0.5) -> float:
    """The largest epsilon whose belief ceiling, from `prior`, is `belief`."""
    return math.log((belief / (1 - belief)) / (prior / (1 - prior)))


# end excerpt


def tpr_bound(eps: float, fpr: float = FPR) -> float:
    """Highest true-positive rate of any membership test at false-positive rate `fpr` against (eps, DELTA)-DP."""
    return min(1.0, math.exp(eps) * fpr + DELTA)


# excerpt: put-frontier
def non_dominated(points: list[tuple[float, float]]) -> list[int]:
    """Indices of the (privacy loss, utility) points that no other point dominates."""
    order = sorted(range(len(points)), key=lambda i: (points[i][0], -points[i][1]))
    frontier: list[int] = []
    best = float("-inf")
    for i in order:
        if points[i][1] > best:
            frontier.append(i)
            best = points[i][1]
    return frontier


# end excerpt


# --- Models ----------------------------------------------------------------------------------------------------


class Classifier(Protocol):
    def predict_proba(self, X: Array) -> Array: ...


class Linear:
    """A logistic regression trained by (DP-)SGD, wrapped to score customers like a scikit-learn model."""

    def __init__(self, layer: nn.Linear) -> None:
        self.layer = layer

    def predict_proba(self, X: Array) -> Array:
        with torch.no_grad():
            p = torch.sigmoid(self.layer(torch.as_tensor(X, dtype=torch.float32))).squeeze(1).double().numpy()
        return np.column_stack([1 - p, p])


# excerpt: put-calibrate
def noise_multiplier(eps: float, n: int) -> float:
    """The noise multiplier that makes DP-SGD with these settings (eps, DELTA)-DP, by Opacus's PRV accountant."""
    return float(
        get_noise_multiplier(
            target_epsilon=eps, target_delta=DELTA, sample_rate=BATCH / n, epochs=EPOCHS, accountant="prv"
        )
    )


# end excerpt


def train_sgd(X: Array, y: npt.NDArray[np.int64], seed: int, sigma: float | None) -> tuple[Linear, float]:
    """Logistic regression by SGD; sigma=None trains without clipping or noise. Returns the model and its epsilon."""
    torch.manual_seed(seed)
    layer = nn.Linear(X.shape[1], 1)
    data = torch.utils.data.TensorDataset(
        torch.as_tensor(X, dtype=torch.float32), torch.as_tensor(y, dtype=torch.float32)
    )
    loader: torch.utils.data.DataLoader[tuple[torch.Tensor, ...]] = torch.utils.data.DataLoader(
        data, batch_size=BATCH, shuffle=True
    )
    # excerpt: put-private
    optimizer = torch.optim.SGD(layer.parameters(), lr=LR, momentum=MOMENTUM)
    engine: PrivacyEngine | None = None
    model: nn.Module = layer
    if sigma is not None:
        engine = PrivacyEngine(accountant="prv")
        model, optimizer, loader = engine.make_private(  # Poisson sampling, per-example clipping, Gaussian noise
            module=layer, optimizer=optimizer, data_loader=loader, noise_multiplier=sigma, max_grad_norm=CLIP
        )
    # end excerpt
    criterion = nn.BCEWithLogitsLoss()
    for _ in range(EPOCHS):
        for xb, yb in loader:
            if len(xb) == 0:  # Poisson sampling can draw an empty batch
                continue
            optimizer.zero_grad()
            criterion(model(xb).squeeze(1), yb).backward()
            optimizer.step()
    eps = engine.get_epsilon(DELTA) if engine is not None else math.inf
    return Linear(layer), float(eps)


def fit(technique: str, eps: float | None, seed: int, X: Array, y: npt.NDArray[np.int64]) -> Classifier:
    """Train one model; eps=None trains the non-private counterpart with the same settings."""
    if technique == "DP-SGD":
        model, spent = train_sgd(X, y, seed, None if eps is None else noise_multiplier(eps, len(y)))
        assert eps is None or spent <= eps * 1.02, f"spent {spent} for target {eps}"
        return model
    d = X.shape[1]
    if eps is None:
        return GaussianNB().fit(X, y)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return DPGaussianNB(epsilon=eps, bounds=(np.zeros(d), np.ones(d)), random_state=seed).fit(X, y)


@dataclass(frozen=True)
class Run:
    technique: str
    eps: float | None
    n: int
    seed: int
    auc: float
    accuracy: float
    reached: int


Data = tuple[Array, npt.NDArray[np.int64], Array, npt.NDArray[np.int64]]
_data: Data | None = None  # set once in each worker process


def _init(data: Data) -> None:
    global _data
    _data = data


def subsample(n: int) -> npt.NDArray[np.int64]:
    """The first n of a fixed random order of the training rows, so smaller samples are nested in larger ones."""
    return np.random.default_rng(0).permutation(TRAIN_ROWS)[:n]


def evaluate(job: tuple[str, float | None, int, int]) -> Run:
    """Train one model on n customers and run the campaign with it."""
    technique, eps, n, seed = job
    assert _data is not None
    X_train, y_train, X_test, y_test = _data
    rows = subsample(n)
    offers = int(y_test.sum())  # as many offers as eligible customers, see the module docstring
    scores = fit(technique, eps, seed, X_train[rows], y_train[rows]).predict_proba(X_test)[:, 1]
    scores = np.nan_to_num(scores, nan=0.5)
    auc = float(roc_auc_score(y_test, scores)) if float(scores.max() - scores.min()) > 0 else 0.5
    accuracy = np.count_nonzero((scores >= 0.5) == (y_test == 1)) / len(y_test)
    return Run(technique, eps, n, seed, auc, accuracy, reached(scores, y_test, offers))


@dataclass(frozen=True)
class Summary:
    technique: str
    eps: float | None
    n: int
    auc: float
    auc_lo: float
    auc_hi: float
    accuracy: float
    reached: float
    reached_lo: float
    reached_hi: float


def summarise(runs: list[Run]) -> Summary:
    """Mean and 10th-90th percentile band over seeds."""
    auc = np.array([r.auc for r in runs])
    hit = np.array([r.reached for r in runs], dtype=float)
    return Summary(
        runs[0].technique,
        runs[0].eps,
        runs[0].n,
        float(auc.mean()),
        float(np.percentile(auc, 10)),
        float(np.percentile(auc, 90)),
        float(np.mean([r.accuracy for r in runs])),
        float(hit.mean()),
        float(np.percentile(hit, 10)),
        float(np.percentile(hit, 90)),
    )


# --- Figures ---------------------------------------------------------------------------------------------------

COLOURS = {"DP-SGD": "#1f5f9e", "naive Bayes": "#c0622b"}


def belief_axis(ax: Axes) -> None:
    """A top axis that labels epsilon with the attacker's belief ceiling from even odds."""
    top = ax.twiny()
    top.set_xscale("log")
    top.set_xlim(ax.get_xlim())
    ticks = [0.1, 0.3, 1.0, 3.0, 10.0]
    top.set_xticks(ticks)
    top.set_xticklabels([f"{belief_ceiling(e):.1%}" if belief_ceiling(e) < 0.999 else ">99.9%" for e in ticks])
    top.minorticks_off()
    top.set_xlabel("highest belief an attacker at 50% can reach about one customer")


def figure_technical(summaries: list[Summary], out: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    for technique, colour in COLOURS.items():
        private = [s for s in summaries if s.technique == technique and s.eps is not None]
        public = next(s for s in summaries if s.technique == technique and s.eps is None)
        eps = [s.eps for s in private if s.eps is not None]
        ax.plot(eps, [s.auc for s in private], "o-", color=colour, label=f"{technique}, DP")
        ax.fill_between(eps, [s.auc_lo for s in private], [s.auc_hi for s in private], color=colour, alpha=0.15)
        ax.axhline(public.auc, color=colour, ls="--", lw=1, label=f"{technique}, no DP")
    ax.axhline(0.5, color="grey", ls=":", lw=1, label="random scores")
    ax.set_xscale("log")
    ax.set_xlim(0.008, 12)
    ax.set_xlabel(r"privacy parameter $\varepsilon$ (smaller is more private)")
    ax.set_ylabel("AUC on the campaign customers")
    ax.legend(loc="lower right", fontsize=8)
    belief_axis(ax)
    fig.tight_layout()
    fig.savefig(out / "technical.svg", metadata={"Date": None})  # no timestamp, so output is reproducible
    plt.close(fig)


def figure_business(summaries: list[Summary], eligible: int, baseline: float, ceiling: float, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    for technique, colour in COLOURS.items():
        private = [s for s in summaries if s.technique == technique and s.eps is not None]
        eps = [s.eps for s in private if s.eps is not None]
        ax.plot(eps, [s.reached for s in private], "o-", color=colour, label=f"{technique}, DP")
        ax.fill_between(eps, [s.reached_lo for s in private], [s.reached_hi for s in private], color=colour, alpha=0.15)
    best_public = max(s.reached for s in summaries if s.eps is None)
    ax.axhline(best_public, color="black", ls="--", lw=1, label="best model without DP")
    ax.axhline(baseline, color="grey", ls=":", lw=1.2, label="no model (offers sent at random)")
    ax.axvline(ceiling, color="#7a3e9d", lw=1, label=f"policy ceiling, $\\varepsilon$ = {ceiling:.2f}")
    ax.axvspan(ceiling, 12, color="#7a3e9d", alpha=0.06)
    ax.set_xscale("log")
    ax.set_xlim(0.008, 12)
    ax.set_ylim(baseline * 0.9, eligible * 0.8)
    ax.set_xlabel(r"privacy parameter $\varepsilon$ (smaller is more private)")
    ax.set_ylabel(f"eligible customers reached by {eligible:,} offers")
    ax.legend(loc="lower right", fontsize=8)
    belief_axis(ax)
    fig.tight_layout()
    fig.savefig(out / "business.svg", metadata={"Date": None})
    plt.close(fig)


def figure_sizes(summaries: list[Summary], eligible: int, baseline: float, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    shades = {SIZES[0]: "#9ecae1", SIZES[1]: "#4292c6", SIZES[2]: "#08306b"}
    for n, colour in shades.items():
        private = sorted((s for s in summaries if s.n == n and s.eps is not None), key=lambda s: s.eps or 0.0)
        public = next(s for s in summaries if s.n == n and s.eps is None)
        eps = [s.eps for s in private if s.eps is not None]
        ax.plot(eps, [s.reached for s in private], "o-", color=colour, label=f"{n:,} customers, DP")
        ax.axhline(public.reached, color=colour, ls="--", lw=1)
    ax.axhline(baseline, color="grey", ls=":", lw=1.2, label="no model (offers sent at random)")
    ax.set_xscale("log")
    ax.set_xlim(0.008, 12)
    ax.set_ylim(baseline * 0.9, eligible * 0.8)
    ax.set_xlabel(r"privacy parameter $\varepsilon$ (smaller is more private)")
    ax.set_ylabel(f"eligible customers reached by {eligible:,} offers")
    ax.legend(loc="lower right", fontsize=8, title="DP-SGD trained on", title_fontsize=8)
    belief_axis(ax)
    fig.tight_layout()
    fig.savefig(out / "sizes.svg", metadata={"Date": None})
    plt.close(fig)


# --- Snippets for the note -------------------------------------------------------------------------------------


def best_at(summaries: list[Summary], eps: float | None) -> Summary:
    return max((s for s in summaries if s.eps == eps), key=lambda s: s.reached)


def frontier_table(summaries: list[Summary], eligible: int) -> str:
    """Every DP point, marked if it is on the frontier (no other point is as private and reaches more)."""
    private = [s for s in summaries if s.eps is not None]
    on = set(non_dominated([(s.eps or 0.0, s.reached) for s in private]))
    rows = [
        "| Technique | $\\varepsilon$ | Attacker's belief ceiling | AUC | Eligible reached | On the frontier |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for i, s in sorted(enumerate(private), key=lambda p: (p[1].eps or 0.0, p[1].technique)):
        eps = s.eps or 0.0
        rows.append(
            f"| {s.technique} | {eps:g} | {belief_ceiling(eps):.1%} | {s.auc:.3f} | "
            f"{s.reached:,.0f} of {eligible:,} ({s.reached / eligible:.0%}) | {'yes' if i in on else ''} |"
        )
    return "\n".join(rows) + "\n\n: Every private configuration, averaged over the seeds. {#tbl-frontier}\n"


def menu_table(summaries: list[Summary], eligible: int, baseline: float) -> str:
    public = best_at(summaries, None)
    rows = [
        (
            "| Option | Attacker's belief ceiling | Membership TPR bound at 1% FPR | Eligible reached | "
            "Offers wasted | Missed vs. no DP | Gained vs. no model |"
        ),
        "| --- | --- | --- | --- | --- | --- | --- |",
        (
            f"| No model (random offers) | — | — | {baseline:,.0f} | {eligible - baseline:,.0f} | "
            f"{public.reached - baseline:,.0f} | 0 |"
        ),
    ]
    for eps in MENU_EPSILONS:
        s = best_at(summaries, eps)
        rows.append(
            f"| $\\varepsilon$ = {eps:g} ({s.technique}) | {belief_ceiling(eps):.1%} | {tpr_bound(eps):.1%} | "
            f"{s.reached:,.0f} | {eligible - s.reached:,.0f} | {public.reached - s.reached:,.0f} | "
            f"{s.reached - baseline:,.0f} |"
        )
    rows.append(
        f"| No DP (the same logistic regression, no clipping or noise) | no limit | no limit | {public.reached:,.0f} | "
        f"{eligible - public.reached:,.0f} | 0 | {public.reached - baseline:,.0f} |"
    )
    caption = (
        f": The options menu for one campaign of {eligible:,} offers, best technique at each option. {{#tbl-menu}}"
    )
    return "\n".join(rows) + "\n\n" + caption + "\n"


def size_table(summaries: list[Summary], ceiling: float) -> str:
    """For each training-set size: the model without DP, and the cost of privacy at a few epsilons."""
    shown = [0.1, 0.3, max(e for e in EPSILONS if e <= ceiling)]
    head = " | ".join(f"Missed at $\\varepsilon$ = {e:g}" for e in shown)
    rows = [f"| Customers trained on | Reached without DP | {head} |", "| --- | --- |" + " --- |" * len(shown)]
    for n in SIZES:
        public = next(s for s in summaries if s.n == n and s.eps is None)
        cells = " | ".join(
            f"{public.reached - next(s for s in summaries if s.n == n and s.eps == e).reached:,.0f}" for e in shown
        )
        rows.append(f"| {n:,} | {public.reached:,.0f} | {cells} |")
    caption = ": Eligible customers missed because of privacy, by the number of customers the model is trained on. {#tbl-sizes}"
    return "\n".join(rows) + "\n\n" + caption + "\n"


def decision_text(summaries: list[Summary], eligible: int, baseline: float, ceiling: float) -> str:
    allowed = [e for e in EPSILONS if e <= ceiling]
    eps = max(allowed)
    s = best_at(summaries, eps)
    public = best_at(summaries, None)
    missed = public.reached - s.reached
    gained = s.reached - baseline
    loosest = best_at(summaries, max(EPSILONS))
    return (
        f"The largest value on the grid below the ceiling is $\\varepsilon = {eps:g}$, and at that value the "
        f"{s.technique} model is on the frontier. Across the seeds it reaches between {s.reached_lo:,.0f} and "
        f"{s.reached_hi:,.0f} of the {eligible:,} eligible customers, {s.reached:,.0f} on average. The model without "
        f"DP reaches {public.reached:,.0f}, and random offers reach {baseline:,.0f}. So privacy costs about "
        f"{missed:,.0f} eligible customers per campaign, while the private model still reaches about {gained:,.0f} "
        f"more than sending the offers at random. In money, with $V$ the value to the bank of reaching one eligible "
        f"customer, the cost of privacy is about ${missed:,.0f}\\,V$ per campaign and the value of the private model "
        f"over no model is about ${gained:,.0f}\\,V$. Even at $\\varepsilon = {max(EPSILONS):g}$, where the noise is "
        f"small, the private model reaches {loosest.reached:,.0f}. So almost all of the cost at the ceiling comes "
        f"from clipping each customer's influence, which DP-SGD does at every $\\varepsilon$, and almost none from "
        f"the noise that the stricter policy adds.\n"
    )


# --- Main ------------------------------------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seeds", type=int, default=20)
    parser.add_argument("--size-seeds", type=int, default=10, help="seeds per point of the data-size sweep")
    parser.add_argument("--out", type=Path, default=FIGURES)
    args = parser.parse_args()
    out: Path = args.out
    seeds: int = args.seeds
    size_seeds: int = args.size_seeds
    out.mkdir(parents=True, exist_ok=True)

    # excerpt: put-campaign
    df = practitioner_datasets.adult()
    train, test = df.head(TRAIN_ROWS), df.slice(TRAIN_ROWS)
    data = (features(train), labels(train), features(test), labels(test))
    eligible = int(data[3].sum())
    # Random offers reach an eligible customer at the base rate, eligible / campaign customers.
    baseline = eligible * eligible / len(data[3])
    # end excerpt
    ceiling = epsilon_for_belief(CEILING_BELIEF)
    print(f"rows: train {len(train):,}  campaign {len(test):,}  features {data[0].shape[1]}  data norm {DATA_NORM:.3f}")
    print(f"high earners: train {data[1].mean():.4f}  campaign {data[3].mean():.4f}")
    print(f"offers = eligible customers in the campaign: {eligible:,}  random offers reach {baseline:,.1f}")
    print(f"policy: belief ceiling {CEILING_BELIEF:.0%} from even odds -> epsilon <= {ceiling:.4f}")

    # excerpt: put-sweep
    jobs: list[tuple[str, float | None, int, int]] = []
    for technique in COLOURS:
        jobs += [(technique, eps, TRAIN_ROWS, seed) for eps in [None, *EPSILONS] for seed in range(seeds)]
    for n in SIZES[:-1]:
        jobs += [("DP-SGD", eps, n, seed) for eps in [None, *EPSILONS] for seed in range(size_seeds)]
    with ProcessPoolExecutor(max_workers=os.cpu_count(), initializer=_init, initargs=(data,)) as pool:
        runs = list(pool.map(evaluate, jobs, chunksize=4))
    # end excerpt
    groups: dict[tuple[str, float | None, int], list[Run]] = {}
    for r in runs:
        groups.setdefault((r.technique, r.eps, r.n), []).append(r)
    every = [summarise(g) for g in groups.values()]
    summaries = [s for s in every if s.n == TRAIN_ROWS]
    by_size = [s for s in every if s.technique == "DP-SGD"]

    print("\ntechnique             n       eps     belief  TPR@1%   AUC (p10-p90)          acc    reached (p10-p90)")
    for s in sorted(every, key=lambda s: (s.technique, s.n, -1.0 if s.eps is None else s.eps)):
        e = s.eps
        privacy = (
            f"{e:<6g}  {belief_ceiling(e):6.1%}  {tpr_bound(e):6.1%}" if e is not None else "none    -       -     "
        )
        print(
            f"{s.technique:<20}  {s.n:<6}  {privacy}  {s.auc:.3f} ({s.auc_lo:.3f}-{s.auc_hi:.3f})  {s.accuracy:.3f}  "
            f"{s.reached:7.1f} ({s.reached_lo:.0f}-{s.reached_hi:.0f})"
        )

    figure_technical(summaries, out)
    figure_business(summaries, eligible, baseline, ceiling, out)
    figure_sizes(by_size, eligible, baseline, out)
    (out / "_frontier.md").write_text(frontier_table(summaries, eligible))
    (out / "_menu.md").write_text(menu_table(summaries, eligible, baseline))
    (out / "_sizes.md").write_text(size_table(by_size, ceiling))
    (out / "_decision.md").write_text(decision_text(summaries, eligible, baseline, ceiling))
    print("\n" + (out / "_decision.md").read_text())


if __name__ == "__main__":
    main()
