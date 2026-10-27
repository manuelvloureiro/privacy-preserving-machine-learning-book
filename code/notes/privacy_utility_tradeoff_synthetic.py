"""The worked example of the study note notes/privacy-utility-tradeoff-synthetic/: a privacy-utility frontier for
differentially private synthetic data on Adult.

    uv run python code/notes/privacy_utility_tradeoff_synthetic.py [--seeds 5] [--out DIR]

The use case. A bank wants an outside agency to plan and run a credit-card campaign, and the agency asks for the
customer table itself: it wants to count customers by segment and to train its own targeting model. The bank will
not hand over its records, so it releases a synthetic table generated with differential privacy (DP) instead. Adult
stands in for the records: 48,842 people from the 1994 US Current Population Survey, one row each, labelled by
whether their income exceeds $50,000 (UCI Machine Learning Repository, dataset 2; Becker & Kohavi, 1996; licence
CC BY 4.0), loaded by code/practitioner_datasets.py. The privacy unit is the row, one customer. The training file
(32,561 rows) plays the bank's customer records, from which the synthetic table is generated, and the test file
(16,281 rows) the customers who receive the next campaign.

Every attribute is discrete, so a table is a set of counts. The codebook (adult.names) gives the categories, and
the numbers are put into public bands (age, hours, capital gains and losses). The survey weight (fnlwgt) and the
duplicate of education (education-num) are dropped, which leaves 13 attributes, income included.

What the script computes, for each generator and each epsilon, over `--seeds` random seeds:

- the technical metric: the average error of the synthetic table on every two-way table (pair of attributes),
  measured as the total variation distance between the synthetic and the real proportions;
- two business metrics: the campaign plan, which counts eligible customers (income above $50,000) in each segment
  of age band and occupation, and the number of eligible customers the plan puts in the wrong segment; and the
  campaign itself, a logistic regression trained on the synthetic table and scored on the campaign customers, which
  sends as many offers as there are eligible customers and counts how many of them it reaches;
- the privacy statement of each epsilon: the highest belief that an attacker who starts at even odds can reach about
  one customer's presence, and the bound on a membership attack's true-positive rate at a 1% false-positive rate.

The three generators all measure tables of counts with Gaussian noise, fit a graphical model to the noisy counts
with Private-PGM (the mbi library) and sample rows from it. They differ in which tables they measure. Independent
measures each attribute's one-way table and nothing else. MST (McKenna, Miklau & Sheldon, 2021) measures the
one-way tables, then chooses a tree of two-way tables with the exponential mechanism and measures those. AIM
(McKenna, Mullins, Sheldon & Miklau, 2022) spends the budget in rounds, each time choosing the table its current
model gets most wrong among the two-way tables and measuring it, with the noise scale adapting as it goes. The
noise is accounted in zero-concentrated DP and converted to (epsilon, 1e-5)-DP, so the generators share an axis
with the companion note's DP-SGD model, which the script also trains on the same attributes for comparison.

Nothing is tuned on the private data. The bands and category lists come from the codebook, the generators use the
settings of their reference implementations (github.com/ryan112358/private-pgm) and the classifier its library
defaults. The offer capacity uses the share of high earners, which the Census Bureau publishes; the script reads it
from the campaign file only to state it.

Output: four SVG figures and four Markdown snippets (_frontier.md, _menu.md, _sizes.md, _decision.md) that the
note includes, written to _build/figures/notes/privacy-utility-tradeoff-synthetic/ (or --out), and the results
printed as a table. Each run (one generator, epsilon, size and seed) is saved as JSON under runs/ in the output
folder, keyed by a hash of the experiment's code, so that a rerun after a change to the figures or the text only
generates what the cache lacks.

Library versions: mbi 1.1.0 (Private-PGM on jax, CPU), scikit-learn 1.8.0, Opacus 1.6.0 (pinned in uv.lock).
"""

import argparse
import hashlib
import inspect
import itertools
import json
import math
import multiprocessing
import os
import sys
import warnings
from collections.abc import Callable, Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path

# One thread per worker process (jax, and the BLAS behind numpy and scikit-learn), 64-bit floats, and no jax
# compilation cache: mbi compiles many small programs, and the cache would only slow a sweep down.
for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(variable, "1")
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("XLA_FLAGS", "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1")
os.environ.setdefault("JAX_ENABLE_X64", "True")
os.environ.setdefault("JAX_ENABLE_COMPILATION_CACHE", "False")
warnings.filterwarnings("ignore", message="JAX is running in float32")
warnings.filterwarnings("ignore", message="JAX persistent compilation cache")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import numpy.typing as npt
import polars as pl
from matplotlib.axes import Axes
from mbi import CliqueVector, Dataset, Domain, LinearMeasurement, MarkovRandomField, estimation, junction_tree
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import practitioner_datasets

matplotlib.rcParams["svg.hashsalt"] = "privacy_utility_tradeoff_synthetic"  # stable element ids across reruns
matplotlib.rcParams["svg.fonttype"] = "none"  # keep text as text: smaller, diffable files

NOTE = "privacy-utility-tradeoff-synthetic"
FIGURES = Path(__file__).resolve().parents[2] / "_build" / "figures" / "notes" / NOTE

Array = npt.NDArray[np.float64]
Ints = npt.NDArray[np.int64]
Clique = tuple[str, ...]
Model = MarkovRandomField  # what Private-PGM's estimator returns

TRAIN_ROWS = 32_561  # adult.data; the remaining 16,281 rows are adult.test
EPSILONS = [0.03, 0.05, 0.1, 0.2, 0.3, 0.5, 1.0, 2.0, 3.0, 10.0]
SIZES = [2_000, 8_000, TRAIN_ROWS]  # customer records the bank generates from, for the data-size sweep (AIM only)
MENU_EPSILONS = [0.1, 0.3, 1.0, 3.0]
DELTA = 1e-5  # as in the companion note: below 1/n for n = 32,561 customers
CEILING_BELIEF = 0.75  # the policy: an attacker at even odds must stay at or below 75%
FPR = 0.01  # the false-positive rate at which membership attacks are judged
LABEL = "income"
ELIGIBLE = ">50K"
SEGMENTS: Clique = ("age", "occupation")  # the campaign plan counts eligible customers by age band and occupation

# The codebook (adult.names): category lists and public bands, never read from the data. A column with blanks in
# the survey gets a "missing" category.
CATEGORIES: dict[str, list[str]] = {
    "workclass": [
        "Private", "Self-emp-not-inc", "Self-emp-inc", "Federal-gov", "Local-gov", "State-gov", "Without-pay",
        "Never-worked", "missing",
    ],
    "education": [
        "Bachelors", "Some-college", "11th", "HS-grad", "Prof-school", "Assoc-acdm", "Assoc-voc", "9th", "7th-8th",
        "12th", "Masters", "1st-4th", "10th", "Doctorate", "5th-6th", "Preschool",
    ],
    "marital-status": [
        "Married-civ-spouse", "Divorced", "Never-married", "Separated", "Widowed", "Married-spouse-absent",
        "Married-AF-spouse",
    ],
    "occupation": [
        "Tech-support", "Craft-repair", "Other-service", "Sales", "Exec-managerial", "Prof-specialty",
        "Handlers-cleaners", "Machine-op-inspct", "Adm-clerical", "Farming-fishing", "Transport-moving",
        "Priv-house-serv", "Protective-serv", "Armed-Forces", "missing",
    ],
    "relationship": ["Wife", "Own-child", "Husband", "Not-in-family", "Other-relative", "Unmarried"],
    "race": ["White", "Asian-Pac-Islander", "Amer-Indian-Eskimo", "Other", "Black"],
    "sex": ["Female", "Male"],
    "native-country": [
        "United-States", "Cambodia", "England", "Puerto-Rico", "Canada", "Germany", "Outlying-US(Guam-USVI-etc)",
        "India", "Japan", "Greece", "South", "China", "Cuba", "Iran", "Honduras", "Philippines", "Italy", "Poland",
        "Jamaica", "Vietnam", "Mexico", "Portugal", "Ireland", "France", "Dominican-Republic", "Laos", "Ecuador",
        "Taiwan", "Haiti", "Columbia", "Hungary", "Guatemala", "Nicaragua", "Scotland", "Thailand", "Yugoslavia",
        "El-Salvador", "Trinadad&Tobago", "Peru", "Hong", "Holand-Netherlands", "missing",
    ],
    LABEL: ["<=50K", ELIGIBLE],
}  # fmt: skip
# Bands for the numbers, as lists of upper limits; the last band is open. Ages run from 17 to 90 in the survey,
# hours from 1 to 99, and capital gains and losses are top-coded at 99,999.
BANDS: dict[str, list[int]] = {
    "age": [24, 34, 44, 54, 64],
    "hours-per-week": [19, 34, 40, 50],
    "capital-gain": [0, 4_999, 9_999],
    "capital-loss": [0, 1_499, 1_999],
}
ATTRIBUTES = [*BANDS, *CATEGORIES]  # the 13 columns of the released table, income last


# --- Data ------------------------------------------------------------------------------------------------------


def band_labels(limits: list[int]) -> list[str]:
    lows = [0, *(x + 1 for x in limits)]
    return [f"{lo}-{hi}" for lo, hi in zip(lows, limits, strict=False)] + [f"{lows[-1]}+"]


def domain() -> Domain:
    """The released table's attributes and how many values each has, from the codebook alone."""
    sizes = {name: len(limits) + 1 for name, limits in BANDS.items()}
    sizes |= {name: len(values) for name, values in CATEGORIES.items()}
    return Domain.fromdict(sizes)


# excerpt: syn-encode
def encode(df: pl.DataFrame) -> Dataset:
    """Map each row to one integer per attribute: its band, or its category's position in the codebook's list."""
    columns: dict[str, Ints] = {}
    for name, limits in BANDS.items():
        x = df[name].cast(pl.Int64).to_numpy()
        columns[name] = np.searchsorted(np.array(limits), x, side="left").astype(np.int64)
    for name, values in CATEGORIES.items():
        index = {v: i for i, v in enumerate(values)}
        missing = index.get("missing", -1)
        columns[name] = np.array([index[v] if v is not None else missing for v in df[name].to_list()], dtype=np.int64)
        assert columns[name].min() >= 0, f"a value of {name} is not in the codebook"
    return Dataset(columns, domain())


# end excerpt


def one_hot(data: Dataset) -> Array:
    """The attributes other than income as 0-or-1 columns, the classifier's input."""
    cols = [np.eye(data.domain[a])[np.asarray(data.to_dict()[a])] for a in ATTRIBUTES if a != LABEL]
    return np.column_stack(cols).astype(np.float64)


def labels(data: Dataset) -> Ints:
    return np.asarray(data.to_dict()[LABEL], dtype=np.int64)


def subsample(data: Dataset, n: int) -> Dataset:
    """The first n of a fixed random order of the rows, so smaller samples are nested in larger ones."""
    rows = np.random.default_rng(0).permutation(data.records)[:n]
    return Dataset({a: np.asarray(v)[rows] for a, v in data.to_dict().items()}, data.domain)


# --- Privacy accounting --------------------------------------------------------------------------------------------


# excerpt: syn-belief
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


# excerpt: syn-zcdp
def zcdp_delta(rho: float, eps: float) -> float:
    """The smallest delta for which rho-zCDP implies (eps, delta)-DP (Canonne, Kamath & Steinke, 2020, Prop. 7)."""
    lo, hi = 1.01, (eps + 1) / (2 * rho) + 2  # the best order alpha lies in this range
    alpha = (lo + hi) / 2
    for _ in range(100):
        alpha = (lo + hi) / 2
        if (2 * alpha - 1) * rho - eps + math.log1p(-1 / alpha) < 0:
            lo = alpha
        else:
            hi = alpha
    return min(1.0, math.exp((alpha - 1) * (alpha * rho - eps) + alpha * math.log1p(-1 / alpha)) / (alpha - 1))


def rho_for_epsilon(eps: float, delta: float = DELTA) -> float:
    """The largest zCDP budget rho that (eps, delta)-DP allows, found by bisection."""
    lo, hi = 0.0, eps + 1
    for _ in range(100):
        rho = (lo + hi) / 2
        lo, hi = (rho, hi) if zcdp_delta(rho, eps) <= delta else (lo, rho)
    return lo


# end excerpt


# --- The generators ------------------------------------------------------------------------------------------


# excerpt: syn-measure
def measure(data: Dataset, clique: Clique, sigma: float, rng: np.random.Generator) -> LinearMeasurement:
    """The table of counts over `clique`, with Gaussian noise of scale sigma on every cell: rho = 1 / (2 sigma^2)."""
    counts = np.asarray(data.project(clique).datavector(), dtype=np.float64)
    return LinearMeasurement(counts + rng.normal(0.0, sigma, counts.size), clique, stddev=sigma)


def exponential_mechanism(scores: dict[Clique, float], eps: float, rng: np.random.Generator) -> Clique:
    """One candidate, drawn with probability proportional to exp(eps * score / 2); each score has sensitivity 1."""
    cliques = list(scores)
    logits = np.array([eps * scores[c] / 2 for c in cliques])
    weights = np.exp(logits - logits.max())
    return cliques[rng.choice(len(cliques), p=weights / weights.sum())]


# end excerpt


def fit(dom: Domain, measurements: list[LinearMeasurement], iters: int, warm: CliqueVector | None = None) -> Model:
    """Private-PGM: the graphical model whose marginals best match the noisy measurements."""
    return estimation.mirror_descent(dom, measurements, iters=iters, potentials=warm)


def pairs(dom: Domain) -> list[Clique]:
    return [(a, b) for a, b in itertools.combinations(list(dom), 2)]


def independent(data: Dataset, rho: float, rng: np.random.Generator) -> Dataset:
    """Every one-way table, measured with the budget split equally, and nothing about how attributes relate."""
    sigma = math.sqrt(len(data.domain) / (2 * rho))
    measurements = [measure(data, (a,), sigma, rng) for a in data.domain]
    return fit(data.domain, measurements, iters=1000).synthetic_data()


# excerpt: syn-mst
def mst(data: Dataset, rho: float, rng: np.random.Generator) -> Dataset:
    """MST: one-way tables, then a spanning tree of the two-way tables that a one-way model gets most wrong."""
    d = len(data.domain)  # a third of the budget each: the one-way tables, the choice of tree, the tree's tables
    one_way = [measure(data, (a,), math.sqrt(3 * d / (2 * rho)), rng) for a in data.domain]
    model = fit(data.domain, one_way, iters=2500)
    scores = {
        pair: float(np.abs(data.project(pair).datavector() - model.project(pair).datavector()).sum())
        for pair in pairs(data.domain)
    }
    tree: list[Clique] = []
    component = {a: a for a in data.domain}  # which tree component each attribute belongs to
    eps_round = math.sqrt(8 * (rho / 3) / (d - 1))
    for _ in range(d - 1):
        joining = {p: s for p, s in scores.items() if component[p[0]] != component[p[1]]}
        a, b = exponential_mechanism(joining, eps_round, rng)
        tree.append((a, b))
        component = {x: component[a] if c == component[b] else c for x, c in component.items()}
    two_way = [measure(data, edge, math.sqrt(3 * (d - 1) / (2 * rho)), rng) for edge in tree]
    return fit(data.domain, one_way + two_way, iters=5000).synthetic_data()


# end excerpt


def aim(data: Dataset, rho: float, rng: np.random.Generator, max_model_size: float = 80.0) -> Dataset:
    """AIM: an adaptive and iterative mechanism, as in its reference implementation (rounds = 16 attributes)."""
    dom = data.domain
    candidates: dict[Clique, float] = {}  # every one- and two-way table, weighted by the two-way tables it serves
    for clique in [(a,) for a in dom] + pairs(dom):
        candidates[clique] = float(sum(len(set(clique) & set(p)) for p in pairs(dom)))
    answers = {c: np.asarray(data.project(c).datavector(), dtype=np.float64) for c in candidates}
    rounds = 16 * len(dom)
    sigma = math.sqrt(rounds / (2 * 0.9 * rho))  # nine tenths of the budget for measuring, one tenth for choosing
    eps = math.sqrt(8 * 0.1 * rho / rounds)
    # excerpt: syn-aim
    measurements = [measure(data, (a,), sigma, rng) for a in dom]
    spent = len(dom) * 0.5 / sigma**2
    model = fit(dom, measurements, iters=1000)
    last = False
    while not last:
        if rho - spent < 2 * (0.5 / sigma**2 + eps**2 / 8):  # the next round would overspend: make it the last
            sigma, eps, last = math.sqrt(1 / (2 * 0.9 * (rho - spent))), math.sqrt(8 * 0.1 * (rho - spent)), True
        spent += eps**2 / 8 + 0.5 / sigma**2
        allowed = allowed_cliques(candidates, model, max_model_size * spent / rho)
        tables = model_tables(model, allowed)  # the model's answer to every candidate, without the library's compiler
        scores = {c: candidates[c] * (l1(answers[c], tables[c]) - bias(sigma, dom, c)) for c in allowed}
        clique = exponential_mechanism(scores, eps, rng)
        measurements.append(measure(data, clique, sigma, rng))
        before = error(answers[clique], model, clique)
        measured = list(dict.fromkeys(m.clique for m in measurements))  # a table can be measured twice
        model = fit(dom, measurements, iters=1000, warm=model.potentials.expand(measured))
        if before - error(answers[clique], model, clique) <= bias(sigma, dom, clique):  # too noisy to help
            sigma, eps = sigma / 2, eps * 2
    # end excerpt
    return fit(dom, measurements, iters=1000, warm=model.potentials).synthetic_data()


def error(answer: Array, model: Model, clique: Clique) -> float:
    """The L1 distance between the real table of counts and the model's, for a table the model holds."""
    return l1(answer, np.asarray(model.project(clique).datavector(), dtype=np.float64))


def l1(answer: Array, table: Array) -> float:
    return float(np.abs(answer - table).sum())


def bias(sigma: float, dom: Domain, clique: Clique) -> float:
    """What the L1 error of a table measured with noise sigma would be if the model were exact."""
    return math.sqrt(2 / math.pi) * sigma * dom.size(clique)


def allowed_cliques(candidates: dict[Clique, float], model: Model, limit: float) -> list[Clique]:
    """The candidates whose addition keeps the model within `limit` megabytes, plus those already in it."""
    cliques = list(model.cliques)
    free = {sub for c in cliques for r in range(1, len(c) + 1) for sub in itertools.combinations(sorted(c), r)}
    return [
        c
        for c in candidates
        if tuple(sorted(c)) in free or junction_tree.hypothetical_model_size(model.domain, [*cliques, c]) <= limit
    ]


# --- The model's tables, in numpy -----------------------------------------------------------------------------
# AIM scores about ninety candidate tables per round, and the library compiles a program for each table and each
# model structure, which takes longer than the round's fit. The model is small (its tables hold a few thousand
# cells), so the same answers come faster from message passing in numpy over the model's junction tree.


@dataclass(frozen=True)
class Table:
    """A table of numbers over some attributes, one axis per attribute."""

    attrs: Clique
    values: Array

    def aligned(self, attrs: Clique) -> Array:
        """The values laid out over `attrs`, a superset of this table's, with a size-1 axis for each absent one."""
        order = [self.attrs.index(a) for a in attrs if a in self.attrs]
        shape = [self.values.shape[self.attrs.index(a)] if a in self.attrs else 1 for a in attrs]
        return np.transpose(self.values, order).reshape(shape)

    def combine(self, other: "Table", op: Callable[[Array, Array], Array]) -> "Table":
        attrs = self.attrs + tuple(a for a in other.attrs if a not in self.attrs)
        return Table(attrs, op(self.aligned(attrs), other.aligned(attrs)))

    def marginal(self, keep: Sequence[str], log: bool = False) -> "Table":
        """Sum out the attributes not in `keep` (in log space, if `log`)."""
        axes = tuple(i for i, a in enumerate(self.attrs) if a not in keep)
        if log:
            top = np.max(self.values, axis=axes, keepdims=True)
            values = np.log(np.exp(self.values - top).sum(axis=axes)) + top.reshape(
                [n for n in top.shape if n > 1] or []
            )
        else:
            values = self.values.sum(axis=axes)
        return Table(tuple(a for a in self.attrs if a in keep), np.asarray(values, dtype=np.float64))


def divide(x: Array, y: Array) -> Array:
    return np.where(y != 0, x / np.where(y != 0, y, 1.0), 0.0)


def clique_tables(model: Model) -> tuple["nx.Graph[Clique]", dict[Clique, Table]]:
    """The model's junction tree and its table of counts over each maximal clique, by HUGIN message passing."""
    dom = model.domain
    tree, _ = junction_tree.make_junction_tree(dom, list(model.cliques))
    maximal = junction_tree.maximal_cliques(tree)
    beliefs = {c: Table(c, np.zeros([dom[a] for a in c])) for c in maximal}
    for clique in model.cliques:  # each potential goes into one maximal clique that contains it
        home = next(c for c in maximal if set(clique) <= set(c))
        factor = model.potentials[clique]
        beliefs[home] = beliefs[home].combine(Table(tuple(factor.domain), np.asarray(factor.values)), np.add)
    messages: dict[tuple[Clique, Clique], Table] = {}
    for i, j in junction_tree.message_passing_order(tree):
        tau = beliefs[i].combine(messages[(j, i)], np.subtract) if (j, i) in messages else beliefs[i]
        messages[(i, j)] = tau.marginal([a for a in i if a in j], log=True)
        beliefs[j] = beliefs[j].combine(messages[(i, j)], np.add)
    for c, b in beliefs.items():
        beliefs[c] = Table(b.attrs, np.exp(b.values - b.marginal([], log=True).values) * float(model.total))
    return tree, beliefs


def model_tables(model: Model, cliques: Sequence[Clique]) -> dict[Clique, Array]:
    """The model's table of counts for each clique, flattened in the clique's attribute order."""
    tree, beliefs = clique_tables(model)
    out: dict[Clique, Array] = {}
    for clique in cliques:
        homes = [c for c in beliefs if set(clique) <= set(c)]
        if homes:
            out[clique] = beliefs[homes[0]].marginal(clique).aligned(clique).flatten()
            continue
        a, b = clique  # two attributes in different cliques: pass the table of `a` along the path between them
        starts, ends = [c for c in beliefs if a in c], [c for c in beliefs if b in c]
        paths = [nx.shortest_path(tree, s, e) for s in starts for e in ends if nx.has_path(tree, s, e)]
        if not paths:  # the model relates nothing between the two attributes: they are independent
            one_way = beliefs[starts[0]].marginal([a]).combine(beliefs[ends[0]].marginal([b]), np.multiply)
            out[clique] = one_way.aligned(clique).flatten() / float(model.total)
            continue
        path = min(paths, key=len)
        table = beliefs[path[0]].marginal([x for x in path[0] if x == a or x in path[1]])
        for before, here, after in zip(path, path[1:], [*path[2:], (b,)]):
            conditional = beliefs[here].combine(beliefs[here].marginal([x for x in here if x in before]), divide)
            table = table.combine(conditional, np.multiply).marginal(
                [x for x in table.attrs + here if x == a or x in after]
            )
        out[clique] = table.aligned(clique).flatten()
    return out


GENERATORS = {"Independent": independent, "MST": mst, "AIM": aim}


# --- Metrics ---------------------------------------------------------------------------------------------------


def reached(scores: Array, y: Ints, offers: int) -> int:
    """Eligible customers among the `offers` highest-scored; ties are broken by position, deterministically."""
    top = np.argsort(-scores, kind="stable")[:offers]
    return int(y[top].sum())


# excerpt: syn-metrics
def workload_error(real: Dataset, synth: Dataset) -> float:
    """Average over the two-way tables of the total variation distance between real and synthetic proportions."""
    errors: list[float] = []
    for pair in pairs(real.domain):
        x = np.asarray(real.project(pair).datavector(), dtype=np.float64)
        y = np.asarray(synth.project(pair).datavector(), dtype=np.float64)
        errors.append(0.5 * float(np.abs(x / x.sum() - y / y.sum()).sum()))
    return float(np.mean(errors))


def misplaced(real: Dataset, synth: Dataset) -> float:
    """Eligible customers the synthetic plan puts in the wrong segment: half the L1 distance between the plans."""
    return 0.5 * float(np.abs(plan(real) - plan(synth)).sum())


# end excerpt


def plan(data: Dataset) -> Array:
    """Eligible customers in each segment: the three-way table of the segment attributes and income."""
    return np.asarray(data.project((*SEGMENTS, LABEL)).datavector(flatten=False), dtype=np.float64)[..., 1]


def campaign(train: Dataset, campaign_x: Array, campaign_y: Ints, seed: int) -> tuple[float, int]:
    """Train the agency's logistic regression on `train` and run the campaign: AUC and eligible customers reached."""
    from sklearn.metrics import roc_auc_score

    y = labels(train)
    if y.min() == y.max():  # a synthetic table with one class only: every score is the same
        scores = np.full(len(campaign_y), 0.5)
    else:
        model = LogisticRegression(max_iter=1000, random_state=seed).fit(one_hot(train), y)
        scores = model.predict_proba(campaign_x)[:, 1]
    auc = float(roc_auc_score(campaign_y, scores)) if scores.max() > scores.min() else 0.5
    return auc, reached(scores, campaign_y, int(campaign_y.sum()))


def dp_sgd(train: Dataset, campaign_x: Array, campaign_y: Ints, eps: float | None, seed: int) -> tuple[float, int]:
    """The companion note's route: a logistic regression trained with DP-SGD on the same attributes."""
    import privacy_utility_tradeoff as companion
    from sklearn.metrics import roc_auc_score

    sigma = None if eps is None else companion.noise_multiplier(eps, train.records)
    model, _ = companion.train_sgd(one_hot(train), labels(train), seed, sigma)
    scores = model.predict_proba(campaign_x)[:, 1]
    return float(roc_auc_score(campaign_y, scores)), reached(scores, campaign_y, int(campaign_y.sum()))


# --- The experiment ---------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Run:
    technique: str
    eps: float | None
    n: int
    seed: int
    workload: float  # average two-way error (nan for the model routes)
    misplaced: float  # eligible customers in the wrong segment of the plan (nan for the model routes)
    auc: float
    reached: int


Data = tuple[Dataset, Array, Ints]
_data: Data | None = None  # set once in each worker process


# Processes for the sweep. Each loads JAX and PyTorch, so the default stays at four; set PRIVACY_WORKERS to change it.
WORKERS = int(os.environ.get("PRIVACY_WORKERS") or min(4, os.cpu_count() or 1))


def _init(data: Data) -> None:
    global _data
    _data = data


def evaluate(job: tuple[str, float | None, int, int, Path | None]) -> Run:
    """Generate one synthetic table (or train one model) from n customers and measure it, through the cache."""
    technique, eps, n, seed, cache = job
    path = cache / f"{technique}_{eps}_{n}_{seed}.json" if cache is not None else None
    if path is not None and path.exists():
        return Run(**json.loads(path.read_text()))
    run = compute(technique, eps, n, seed)
    if path is not None:
        path.write_text(json.dumps(run.__dict__))
    return run


def compute(technique: str, eps: float | None, n: int, seed: int) -> Run:
    assert _data is not None
    records, campaign_x, campaign_y = _data
    train = subsample(records, n)
    if technique == "DP-SGD":
        auc, hit = dp_sgd(train, campaign_x, campaign_y, eps, seed)
        return Run(technique, eps, n, seed, math.nan, math.nan, auc, hit)
    if eps is None:  # the real table, released as it is
        auc, hit = campaign(train, campaign_x, campaign_y, seed)
        return Run("Real table", None, n, seed, 0.0, 0.0, auc, hit)
    rng = np.random.default_rng([seed, int(eps * 1000), n])
    np.random.seed(seed)  # mbi samples the synthetic rows from numpy's global generator
    synth = GENERATORS[technique](train, rho_for_epsilon(eps), rng)
    auc, hit = campaign(synth, campaign_x, campaign_y, seed)
    return Run(technique, eps, n, seed, workload_error(train, synth), misplaced(train, synth), auc, hit)


@dataclass(frozen=True)
class Summary:
    technique: str
    eps: float | None
    n: int
    workload: float
    misplaced: float
    misplaced_lo: float
    misplaced_hi: float
    auc: float
    reached: float
    reached_lo: float
    reached_hi: float


def summarise(runs: list[Run]) -> Summary:
    """Mean and 10th-90th percentile band over seeds."""
    hit = np.array([r.reached for r in runs], dtype=float)
    wrong = np.array([r.misplaced for r in runs], dtype=float)
    return Summary(
        runs[0].technique,
        runs[0].eps,
        runs[0].n,
        float(np.mean([r.workload for r in runs])),
        float(wrong.mean()),
        float(np.percentile(wrong, 10)),
        float(np.percentile(wrong, 90)),
        float(np.mean([r.auc for r in runs])),
        float(hit.mean()),
        float(np.percentile(hit, 10)),
        float(np.percentile(hit, 90)),
    )


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


# --- Figures ---------------------------------------------------------------------------------------------------

COLOURS = {"Independent": "#8c8c8c", "MST": "#c0622b", "AIM": "#1f5f9e"}
MODEL_COLOUR = "#4a9a4a"


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


def private_points(summaries: list[Summary], technique: str, n: int = TRAIN_ROWS) -> list[Summary]:
    return sorted((s for s in summaries if s.technique == technique and s.eps is not None and s.n == n), key=eps_of)


def eps_of(s: Summary) -> float:
    return s.eps if s.eps is not None else math.inf


def finish(ax: Axes, xlabel: bool = True) -> None:
    ax.set_xscale("log")
    ax.set_xlim(0.02, 12)
    if xlabel:
        ax.set_xlabel(r"privacy parameter $\varepsilon$ (smaller is more private)")
    belief_axis(ax)


def figure_technical(summaries: list[Summary], out: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    for technique, colour in COLOURS.items():
        private = private_points(summaries, technique)
        ax.plot([eps_of(s) for s in private], [s.workload for s in private], "o-", color=colour, label=technique)
    ax.set_yscale("log")
    ax.set_ylabel("average error on the two-way tables (TVD)")
    ax.legend(loc="upper right", fontsize=8)
    finish(ax)
    fig.tight_layout()
    fig.savefig(out / "technical.svg", metadata={"Date": None})  # no timestamp, so output is reproducible
    plt.close(fig)


def figure_plan(summaries: list[Summary], eligible_records: int, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    for technique, colour in COLOURS.items():
        private = private_points(summaries, technique)
        eps = [eps_of(s) for s in private]
        ax.plot(eps, [s.misplaced for s in private], "o-", color=colour, label=technique)
        ax.fill_between(
            eps, [s.misplaced_lo for s in private], [s.misplaced_hi for s in private], color=colour, alpha=0.15
        )
    ax.set_yscale("log")
    ax.set_ylabel(f"eligible customers in the wrong segment, of {eligible_records:,}")
    ax.legend(loc="upper right", fontsize=8)
    finish(ax)
    fig.tight_layout()
    fig.savefig(out / "plan.svg", metadata={"Date": None})
    plt.close(fig)


def figure_business(summaries: list[Summary], eligible: int, baseline: float, ceiling: float, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    for technique, colour in COLOURS.items():
        private = private_points(summaries, technique)
        eps = [eps_of(s) for s in private]
        ax.plot(eps, [s.reached for s in private], "o-", color=colour, label=f"{technique} synthetic table")
        ax.fill_between(eps, [s.reached_lo for s in private], [s.reached_hi for s in private], color=colour, alpha=0.15)
    model = private_points(summaries, "DP-SGD")
    ax.plot([eps_of(s) for s in model], [s.reached for s in model], "s--", color=MODEL_COLOUR, label="DP-SGD model")
    real = next(s for s in summaries if s.technique == "Real table" and s.n == TRAIN_ROWS)
    ax.axhline(real.reached, color="black", ls="--", lw=1, label="the real table")
    ax.axhline(baseline, color="grey", ls=":", lw=1.2, label="no data (offers sent at random)")
    ax.axvline(ceiling, color="#7a3e9d", lw=1, label=f"policy ceiling, $\\varepsilon$ = {ceiling:.2f}")
    ax.axvspan(ceiling, 12, color="#7a3e9d", alpha=0.06)
    ax.set_ylim(baseline * 0.9, eligible * 0.8)
    ax.set_ylabel(f"eligible customers reached by {eligible:,} offers")
    ax.legend(loc="lower right", fontsize=8)
    finish(ax)
    fig.tight_layout()
    fig.savefig(out / "business.svg", metadata={"Date": None})
    plt.close(fig)


def figure_sizes(summaries: list[Summary], eligible: int, baseline: float, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    shades = {SIZES[0]: "#9ecae1", SIZES[1]: "#4292c6", SIZES[2]: "#08306b"}
    for n, colour in shades.items():
        private = private_points(summaries, "AIM", n)
        real = next(s for s in summaries if s.technique == "Real table" and s.n == n)
        ax.plot(
            [eps_of(s) for s in private], [s.reached for s in private], "o-", color=colour, label=f"{n:,} customers"
        )
        ax.axhline(real.reached, color=colour, ls="--", lw=1)
    ax.axhline(baseline, color="grey", ls=":", lw=1.2, label="no data (offers sent at random)")
    ax.set_ylim(baseline * 0.9, eligible * 0.8)
    ax.set_ylabel(f"eligible customers reached by {eligible:,} offers")
    ax.legend(loc="lower right", fontsize=8, title="AIM table generated from", title_fontsize=8)
    finish(ax)
    fig.tight_layout()
    fig.savefig(out / "sizes.svg", metadata={"Date": None})
    plt.close(fig)


# --- Snippets for the note -------------------------------------------------------------------------------------


def epsilons_run(summaries: list[Summary]) -> list[float]:
    """The values of epsilon the sweep ran, in order (a short run passes fewer than EPSILONS)."""
    return sorted({s.eps for s in summaries if s.eps is not None})


def best_at(summaries: list[Summary], eps: float | None, n: int = TRAIN_ROWS) -> Summary:
    """The synthetic generator (not the model route) that reaches most customers at this epsilon."""
    return max((s for s in summaries if s.eps == eps and s.n == n and s.technique in COLOURS), key=lambda s: s.reached)


def frontier_table(summaries: list[Summary], eligible: int, eligible_records: int) -> str:
    """Every synthetic configuration, marked if it is on the frontier (no other is as private and reaches more)."""
    private = [s for s in summaries if s.eps is not None and s.n == TRAIN_ROWS and s.technique in COLOURS]
    on = set(non_dominated([(eps_of(s), s.reached) for s in private]))
    rows = [
        (
            "| Generator | $\\varepsilon$ | Attacker's belief ceiling | Two-way error | Eligible misplaced in the plan | "
            "Eligible reached | On the frontier |"
        ),
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for i, s in sorted(enumerate(private), key=lambda p: (eps_of(p[1]), list(COLOURS).index(p[1].technique))):
        rows.append(
            f"| {s.technique} | {eps_of(s):g} | {belief_ceiling(eps_of(s)):.1%} | {s.workload:.3f} | "
            f"{s.misplaced:,.0f} of {eligible_records:,} | {s.reached:,.0f} of {eligible:,} ({s.reached / eligible:.0%}) | "
            f"{'yes' if i in on else ''} |"
        )
    return "\n".join(rows) + "\n\n: Every synthetic configuration, averaged over the seeds. {#tbl-frontier}\n"


def menu_table(summaries: list[Summary], eligible: int, eligible_records: int, baseline: float) -> str:
    real = next(s for s in summaries if s.technique == "Real table" and s.n == TRAIN_ROWS)
    rows = [
        (
            "| Option | Attacker's belief ceiling | Membership TPR bound at 1% FPR | Eligible misplaced in the plan | "
            "Eligible reached | Missed vs. the real table | Gained vs. no data |"
        ),
        "| --- | --- | --- | --- | --- | --- | --- |",
        f"| No data (random offers) | — | — | — | {baseline:,.0f} | {real.reached - baseline:,.0f} | 0 |",
    ]
    for eps in [e for e in MENU_EPSILONS if e in epsilons_run(summaries)]:  # a short run sweeps fewer values
        s = best_at(summaries, eps)
        m = next(x for x in summaries if x.technique == "DP-SGD" and x.eps == eps and x.n == TRAIN_ROWS)
        rows.append(
            f"| $\\varepsilon$ = {eps:g}, {s.technique} table | {belief_ceiling(eps):.1%} | {tpr_bound(eps):.1%} | "
            f"{s.misplaced:,.0f} of {eligible_records:,} | {s.reached:,.0f} | {real.reached - s.reached:,.0f} | "
            f"{s.reached - baseline:,.0f} |"
        )
        rows.append(
            f"| $\\varepsilon$ = {eps:g}, DP-SGD model | {belief_ceiling(eps):.1%} | {tpr_bound(eps):.1%} | "
            f"no plan | {m.reached:,.0f} | {real.reached - m.reached:,.0f} | {m.reached - baseline:,.0f} |"
        )
    rows.append(
        f"| The real table, no DP | no limit | no limit | 0 | {real.reached:,.0f} | 0 | {real.reached - baseline:,.0f} |"
    )
    caption = (
        f": The options menu for one campaign of {eligible:,} offers: the best generator at each option, "
        "and the companion note's model route at the same privacy level. {#tbl-menu}"
    )
    return "\n".join(rows) + "\n\n" + caption + "\n"


def size_table(summaries: list[Summary], ceiling: float) -> str:
    """For each number of customer records: the real table, and the cost of privacy at a few epsilons."""
    shown = [e for e in (0.1, 0.3, max(e for e in EPSILONS if e <= ceiling)) if e in epsilons_run(summaries)]
    head = " | ".join(f"Missed at $\\varepsilon$ = {e:g}" for e in shown)
    rows = [
        f"| Customers in the records | Reached with the real table | {head} |",
        "| --- | --- |" + " --- |" * len(shown),
    ]
    for n in SIZES:
        real = next(s for s in summaries if s.technique == "Real table" and s.n == n)
        cells = " | ".join(
            f"{real.reached - next(s for s in summaries if s.technique == 'AIM' and s.n == n and s.eps == e).reached:,.0f}"
            for e in shown
        )
        rows.append(f"| {n:,} | {real.reached:,.0f} | {cells} |")
    caption = (
        ": Eligible customers missed because of privacy with an AIM table, by the number of customer records "
        "it is generated from. {#tbl-sizes}"
    )
    return "\n".join(rows) + "\n\n" + caption + "\n"


def decision_text(
    summaries: list[Summary], eligible: int, eligible_records: int, baseline: float, ceiling: float
) -> str:
    eps = max(e for e in EPSILONS if e <= ceiling)
    s = best_at(summaries, eps)
    real = next(x for x in summaries if x.technique == "Real table" and x.n == TRAIN_ROWS)
    model = next(x for x in summaries if x.technique == "DP-SGD" and x.eps == eps and x.n == TRAIN_ROWS)
    missed = real.reached - s.reached
    gained = s.reached - baseline
    return (
        f"The largest value on the grid below the ceiling is $\\varepsilon = {eps:g}$, and at that value the "
        f"{s.technique} table is on the frontier. Across the seeds its campaign reaches between {s.reached_lo:,.0f} "
        f"and {s.reached_hi:,.0f} of the {eligible:,} eligible customers, {s.reached:,.0f} on average, and its plan "
        f"puts {s.misplaced:,.0f} of the {eligible_records:,} eligible customers in the records into the wrong "
        f"segment, {s.misplaced / eligible_records:.1%} of them. The real table reaches {real.reached:,.0f} and random "
        f"offers reach {baseline:,.0f}. So privacy costs about {missed:,.0f} eligible customers per campaign, while "
        f"the synthetic table is still worth about {gained:,.0f} customers more than no data. In money, with $V$ the "
        f"value to the bank of reaching one eligible customer, the cost of privacy is about ${missed:,.0f}\\,V$ per "
        f"campaign and the value of the table over no data about ${gained:,.0f}\\,V$. The model route of the "
        f"companion note, a DP-SGD model at the same $\\varepsilon$, reaches {model.reached:,.0f}, so the table costs "
        f"about {model.reached - s.reached:,.0f} more customers per campaign than the model, and in return it answers "
        f"the planning question and any other the agency asks.\n"
    )


# --- Main ------------------------------------------------------------------------------------------------------


def experiment_hash() -> str:
    """A hash of the code and constants that decide a run's result, which names the cache of finished runs."""
    parts = [
        inspect.getsource(f)
        for f in (
            encode, one_hot, labels, subsample, zcdp_delta, rho_for_epsilon, measure, exponential_mechanism, fit,
            independent, mst, aim, error, l1, bias, allowed_cliques, Table, divide, clique_tables, model_tables,
            reached, workload_error, misplaced, plan, campaign, dp_sgd, compute,
        )
    ]  # fmt: skip
    parts += [repr((TRAIN_ROWS, DELTA, SEGMENTS, LABEL, ELIGIBLE, CATEGORIES, BANDS, ATTRIBUTES))]
    return hashlib.sha256("\n".join(parts).encode()).hexdigest()[:12]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--size-seeds", type=int, default=2, help="seeds per point of the data-size sweep")
    parser.add_argument("--epsilons", type=float, nargs="+", default=EPSILONS)
    parser.add_argument("--out", type=Path, default=FIGURES)
    args = parser.parse_args()
    out: Path = args.out
    seeds: int = args.seeds
    size_seeds: int = args.size_seeds
    epsilons: list[float] = args.epsilons
    EPSILONS[:] = epsilons
    cache = out / "runs" / experiment_hash()
    cache.mkdir(parents=True, exist_ok=True)

    # excerpt: syn-campaign
    df = practitioner_datasets.adult()
    records, customers = encode(df.head(TRAIN_ROWS)), encode(df.slice(TRAIN_ROWS))
    data: Data = (records, one_hot(customers), labels(customers))
    eligible = int(data[2].sum())  # offers = eligible customers in the campaign, as in the companion note
    baseline = eligible * eligible / customers.records  # random offers reach them at the base rate
    eligible_records = int(labels(records).sum())  # the eligible customers the plan distributes over segments
    # end excerpt
    ceiling = epsilon_for_belief(CEILING_BELIEF)
    dom = records.domain
    print(f"rows: records {records.records:,}  campaign {customers.records:,}  attributes {len(dom)}")
    print(f"domain: {dict(zip(list(dom), dom.shape))}  two-way tables {len(pairs(dom))}")
    print(f"plan: {len(SEGMENTS)} attributes, {dom.size(SEGMENTS)} segments, {eligible_records:,} eligible customers")
    print(f"offers = eligible customers in the campaign: {eligible:,}  random offers reach {baseline:,.1f}")
    print(f"policy: belief ceiling {CEILING_BELIEF:.0%} from even odds -> epsilon <= {ceiling:.4f}")
    print("zCDP budgets: " + ", ".join(f"eps {e:g} -> rho {rho_for_epsilon(e):.4g}" for e in epsilons))

    # excerpt: syn-sweep
    jobs: list[tuple[str, float | None, int, int, Path | None]] = [("Real table", None, TRAIN_ROWS, 0, cache)]
    for technique in [*GENERATORS, "DP-SGD"]:
        jobs += [(technique, eps, TRAIN_ROWS, seed, cache) for eps in epsilons for seed in range(seeds)]
    for n in SIZES[:-1]:
        jobs += [("Real table", None, n, 0, cache)]
        jobs += [("AIM", eps, n, seed, cache) for eps in epsilons for seed in range(size_seeds)]
    jobs.sort(key=lambda j: (j[0] != "AIM", -(j[1] or 0.0)))  # the slow jobs first, so the cores stay busy
    context = multiprocessing.get_context("spawn")  # jax must start fresh in each worker
    # A worker's memory grows with every program jax compiles, so each job gets a fresh process.
    pool = ProcessPoolExecutor(WORKERS, mp_context=context, initializer=_init, initargs=(data,), max_tasks_per_child=1)
    with pool:
        runs = list(pool.map(evaluate, jobs))
    # end excerpt
    groups: dict[tuple[str, float | None, int], list[Run]] = {}
    for r in runs:
        groups.setdefault((r.technique, r.eps, r.n), []).append(r)
    summaries = [summarise(g) for g in groups.values()]

    print(
        "\ntechnique      n       eps     belief  TPR@1%   2-way err  misplaced (p10-p90)     AUC    reached (p10-p90)"
    )
    order = ["Real table", *GENERATORS, "DP-SGD"]
    for s in sorted(summaries, key=lambda s: (order.index(s.technique), s.n, eps_of(s))):
        e = s.eps
        privacy = (
            f"{e:<6g}  {belief_ceiling(e):6.1%}  {tpr_bound(e):6.1%}" if e is not None else "none    -       -     "
        )
        wrong = (
            f"{s.misplaced:7.0f} ({s.misplaced_lo:.0f}-{s.misplaced_hi:.0f})" if not math.isnan(s.misplaced) else "-"
        )
        print(
            f"{s.technique:<13}  {s.n:<6}  {privacy}  {s.workload:9.4f}  {wrong:<22}  {s.auc:.3f}  "
            f"{s.reached:7.1f} ({s.reached_lo:.0f}-{s.reached_hi:.0f})"
        )

    figure_technical(summaries, out)
    figure_plan(summaries, eligible_records, out)
    figure_business(summaries, eligible, baseline, ceiling, out)
    figure_sizes(summaries, eligible, baseline, out)
    (out / "_frontier.md").write_text(frontier_table(summaries, eligible, eligible_records))
    (out / "_menu.md").write_text(menu_table(summaries, eligible, eligible_records, baseline))
    (out / "_sizes.md").write_text(size_table(summaries, ceiling))
    (out / "_decision.md").write_text(decision_text(summaries, eligible, eligible_records, baseline, ceiling))
    print("\n" + (out / "_decision.md").read_text())


if __name__ == "__main__":
    main()
