"""Evaluating and tuning synthetic data: the chapter "Evaluating and tuning synthetic data" of the OpenDP book (Part 5).

    uv run python code/part05-synthetic-data/opendp_evaluating_synthetic_data.py

Goes beyond the documentation: it measures how good a synthetic table is. One AIM release at rho = 0.1, with the
context parameters, columns, keys, cuts, null filling and rounds of the previous chapter, is compared with the real
data three ways. First the workload error (McKenna et al. 2022): the total variation distance and the largest cell
error of every one-way and every measured marginal of the fitted model, against the bound the table's own std() gives.
Then a query nobody measured: the four-way table, which AIM never measures because its candidates have at most three
columns, and the share of people working in each cell of SEX, AGE and HWUSUAL read off it. Then a downstream task in
the manner of Tao et al. (2021): a logistic regression that predicts ILOSTAT == 1 from SEX, the AGE band and the
HWUSUAL band, trained on 200,000 synthetic rows drawn with method="sample" and on 200,000 real rows, and scored on
200,000 held-out real rows (train on synthetic, test on real). The same evaluation is then run at rho = 0.02 and
rho = 0.5 and for MST at rho = 0.1, each in a context of its own. Last, the number of synthetic rows and the two
methods of synthesize() are compared: sampling adds noise of its own, and in mbi 1.1.0 rounding couples columns that
are drawn from the same parent columns. It prints every number the page quotes, on labelled lines, in the order of
the page's sections, and the run time at the end.

The script imports the previous chapter's script, code/part05-synthetic-data/opendp_aim_and_mst.py, for its public
functions and constants (fill_nulls, timed_release, synthesize, binned, marginal, tv_bound, cells, measured,
gaussian_scale, epsilon_of_rho, KEYS, CUTS, FILL, COLUMNS, CONTRIBUTIONS, ROUNDS, RHO_QUERY, DELTA_QUERY) rather than
duplicating them; only what is new to this chapter is defined here. Its own make_context and table_query take the
budget as an argument, and its own distance compares counts of any scale as distributions, since the fitted model,
the synthetic frames and the real table have different totals.

Dataset: the France LFS sample of the OpenDP tutorials (3,811,932 rows, 1,682,306 people), loaded by
code/practitioner_datasets.py. The data are public, so the real marginals and the held-out rows are available for the
comparison, which a real release could not make on its private data.

What is deterministic and what is not. OpenDP draws its noise from a cryptographic generator that cannot be seeded,
so every release, every fitted model, every synthetic frame and every distance changes from run to run; they are
printed under "one run" labels and the test checks them against bounds. The budgets, scales, thresholds, conversions
to epsilon, the facts about the data and the real-data classifier are deterministic: the seed drives the Polars
samples of real rows, NumPy's global generator (which synthesize() uses) and the classifier. The expected sampling
noise is computed from the released table's own shares, so it is random too.

Versions: opendp[mbi,polars] 0.16.0 with polars 1.36.1, mbi 1.1.0 (the Private-PGM version OpenDP pins) and
scikit-learn 1.8.0. JAX runs on the CPU and its import-time advice is silenced, as in the previous chapter.
"""

import argparse
import importlib.metadata
import itertools
import math
import sys
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import opendp.prelude as dp
import polars as pl
from numpy.typing import NDArray
from opendp.extras.mbi import Algorithm, ContingencyTable
from opendp.extras.polars.contingency_table import ContingencyTableQuery
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import opendp_aim_and_mst as synth  # the previous chapter's functions and constants
import practitioner_datasets

dp.enable_features("contrib")

CONTRIBUTIONS = synth.CONTRIBUTIONS  # 36: a person answers at most once per quarter over nine years
DELTA = synth.DELTA_QUERY  # 1e-7, the slack that pays for releasing ILOSTAT's keys
RHO = synth.RHO_QUERY  # 0.1, the budget of the main release, as in the previous chapter
BUDGETS = (0.02, 0.1, 0.5)  # the budgets compared, each in a context of its own
ROUNDS = synth.ROUNDS  # 10 planned rounds of AIM, so that a release takes seconds
COLUMNS = synth.COLUMNS
ROWS = 200_000  # synthetic rows drawn for the classifier and for the comparison of methods
HOLDOUT = 200_000  # real rows held out to score the classifiers, and as many real rows to train the baseline
FEATURES = ("SEX", "AGE", "HWUSUAL")  # the classifier's inputs, as keys and bins
TARGET = "ILOSTAT"
WORKING = 1  # the ILOSTAT code the classifier predicts
KEYS_ILOSTAT: list[int | None] = [1, 2, 3, 9, None]  # the keys every release here gives ILOSTAT; checked per release
MIN_CELL = 1_000  # real rows a cell needs before its conditional share is compared


# ---------------------------------------------------------------------------------------------------------------
# One context per release
# ---------------------------------------------------------------------------------------------------------------


# excerpt: eval-context
def make_context(frame: pl.LazyFrame, rho: float) -> dp.Context:
    """A context for one release at budget rho, with the previous chapter's unit and slack."""
    return dp.Context.compositor(
        data=frame,
        privacy_unit=dp.unit_of(contributions=CONTRIBUTIONS),  # neighbours add or remove one person's 36 rows
        privacy_loss=dp.loss_of(rho=rho, delta=DELTA),  # zCDP with Gaussian noise; delta pays for ILOSTAT's keys
    )


def table_query(context: dp.Context, rho: float, algorithm: Algorithm) -> ContingencyTableQuery:
    """The previous chapter's query with its nulls filled, at the budget rho, with the algorithm plugged in."""
    return (
        context.query(rho=rho, delta=DELTA)
        .with_columns(synth.fill_nulls())  # every null lands in a public key before anything is counted
        .select(*COLUMNS)
        .contingency_table(keys=synth.KEYS, cuts=synth.CUTS, algorithm=algorithm)
    )


# end excerpt


# ---------------------------------------------------------------------------------------------------------------
# The distance between two marginals
# ---------------------------------------------------------------------------------------------------------------


def distance(real: pl.DataFrame, other: pl.DataFrame, by: Sequence[str]) -> tuple[float, int]:
    """The total variation distance between two marginals read as distributions, and the largest cell error in rows
    at the real table's size, so that counts of different totals compare."""
    joined = real.join(other, on=list(by), how="full", coalesce=True, nulls_equal=True, suffix="_other")
    real_counts = joined["len"].fill_null(0).cast(pl.Float64).to_numpy()
    other_counts = joined["len_other"].fill_null(0).cast(pl.Float64).to_numpy()
    p = real_counts / real_counts.sum()
    q = other_counts / other_counts.sum()
    return 0.5 * float(np.abs(p - q).sum()), round(float(np.abs(p - q).max()) * float(real_counts.sum()))


@dataclass
class MarginalError:
    clique: tuple[str, ...]
    tv: float
    worst: int  # the largest cell error in rows at the real table's size
    std: float
    bound: float

    @property
    def label(self) -> str:
        return self.clique[0] if len(self.clique) == 1 else "(" + ", ".join(self.clique) + ")"


# excerpt: eval-errors
def marginal_errors(
    table: ContingencyTable, real: pl.DataFrame, cliques: Sequence[tuple[str, ...]]
) -> list[MarginalError]:
    """Each clique's distance between the fitted model's marginal and the real one, with the table's own standard
    deviation for it and the bound that three standard deviations in every cell would give."""
    errors: list[MarginalError] = []
    for clique in cliques:
        fitted = table.project_melted(clique)  # the model's counts: no rows are drawn
        tv, worst = distance(synth.marginal(real, clique), fitted, clique)
        std = table.std(clique)
        bound = synth.tv_bound(synth.cells(table, clique), std, real.height)
        errors.append(MarginalError(clique, tv, worst, std, bound))
    return errors


# end excerpt


def working_share(counts: pl.DataFrame) -> pl.DataFrame:
    """The share of rows with ILOSTAT == 1 in every combination of SEX, AGE band and HWUSUAL band, from the counts
    of the four-way table."""
    working = pl.when(pl.col(TARGET) == WORKING).then(pl.col("len")).otherwise(0).cast(pl.Float64)
    return counts.group_by(list(FEATURES)).agg(
        (working.sum() / pl.col("len").cast(pl.Float64).sum()).alias("share"), pl.col("len").sum()
    )


def share_error(real: pl.DataFrame, fitted: pl.DataFrame) -> tuple[float, dict[str, Any], float, float]:
    """The largest error of the conditional share, in percentage points, over the cells with at least MIN_CELL real
    rows, with the cell and its two shares."""
    joined = (
        working_share(real)
        .filter(pl.col("len") >= MIN_CELL)
        .join(working_share(fitted), on=list(FEATURES), how="left", nulls_equal=True, suffix="_model")
        .with_columns(pl.col("share_model").fill_null(0.0).fill_nan(0.0))  # a cell the model leaves empty
        .with_columns((100 * (pl.col("share") - pl.col("share_model")).abs()).alias("error"))
        .sort("error", descending=True)
    )
    row = joined.row(0, named=True)
    cell = {column: row[column] for column in FEATURES}
    return float(row["error"]), cell, float(row["share"]), float(row["share_model"])


def sampling_tv(shares: NDArray[np.float64], rows: int) -> float:
    """The expected total variation distance that drawing `rows` independent rows adds to a marginal with these
    shares: sqrt(1 / (2 pi rows)) times the sum over cells of sqrt(p (1 - p))."""
    return math.sqrt(1 / (2 * math.pi * rows)) * float(np.sqrt(shares * (1 - shares)).sum())


# ---------------------------------------------------------------------------------------------------------------
# Train on synthetic, test on real
# ---------------------------------------------------------------------------------------------------------------


def design(frame: pl.DataFrame, keys: dict[str, Any]) -> NDArray[np.float64]:
    """One indicator column per key of SEX, AGE and HWUSUAL, in the table's key order; the binned frames carry
    bin labels, so every synthetic row and every real row maps to the same columns."""
    exprs: list[pl.Expr] = []
    for column in FEATURES:
        for key in keys[column].to_list():
            indicator = pl.col(column).is_null() if key is None else pl.col(column) == key
            exprs.append(indicator.fill_null(False).cast(pl.Float64).alias(f"{column}={key}"))
    return frame.select(exprs).to_numpy()


def labels(frame: pl.DataFrame) -> NDArray[np.int64]:
    return frame.select((pl.col(TARGET) == WORKING).fill_null(False).cast(pl.Int64)).to_numpy().ravel()


# excerpt: eval-classifier
def classifier(train: pl.DataFrame, keys: dict[str, Any], seed: int) -> LogisticRegression:
    """A logistic regression that predicts ILOSTAT == 1 from SEX, the AGE band and the HWUSUAL band, fitted on
    `train`, which is either synthetic rows or real rows."""
    return LogisticRegression(max_iter=1000, random_state=seed).fit(design(train, keys), labels(train))


def accuracy(model: LogisticRegression, test: pl.DataFrame, keys: dict[str, Any]) -> float:
    """The share of the held-out real rows the model labels correctly."""
    return float((model.predict(design(test, keys)) == labels(test)).mean())


# end excerpt


def probability_gap(
    model: LogisticRegression, baseline: LogisticRegression, test: pl.DataFrame, keys: dict[str, Any]
) -> float:
    """The mean absolute difference, over the held-out rows, between the two models' predicted probabilities of
    working: a finer comparison than accuracy, which only sees which side of one half a probability falls."""
    features = design(test, keys)
    return float(np.abs(model.predict_proba(features)[:, 1] - baseline.predict_proba(features)[:, 1]).mean())


# ---------------------------------------------------------------------------------------------------------------
# One evaluation per release
# ---------------------------------------------------------------------------------------------------------------


@dataclass
class Evaluation:
    name: str
    seconds_release: float
    oneway: list[MarginalError]
    measured: list[MarginalError]
    four_tv: float
    four_worst: int
    share: tuple[float, dict[str, Any], float, float]
    accuracy_synthetic: float
    accuracy_real: float
    probability_gap: float

    @property
    def worst_measured(self) -> MarginalError:
        return max(self.measured, key=lambda e: e.tv)

    @property
    def best_measured(self) -> MarginalError:
        return min(self.measured, key=lambda e: e.tv)

    @property
    def oneway_mean(self) -> float:
        return sum(e.tv for e in self.oneway) / len(self.oneway)

    @property
    def gap(self) -> float:
        return self.accuracy_real - self.accuracy_synthetic


def evaluate(
    name: str,
    table: ContingencyTable,
    seconds_release: float,
    real: pl.DataFrame,
    test: pl.DataFrame,
    baseline: LogisticRegression,
    seed: int,
) -> Evaluation:
    """The three checks of one release: its marginals, the four-way table nobody measured, and the classifier."""
    if table.keys["ILOSTAT"].to_list() != KEYS_ILOSTAT:
        raise ValueError(f"{name}: the released keys of ILOSTAT differ from the real rows' binning")
    cliques = sorted({clique for clique, _ in synth.measured(table)}, key=lambda c: (len(c), c))
    real_four = synth.marginal(real, COLUMNS)
    fitted_four = table.project_melted(COLUMNS)
    four_tv, four_worst = distance(real_four, fitted_four, COLUMNS)
    # excerpt: eval-synthesize
    rows = synth.synthesize(table, seed, rows=ROWS, method="sample")  # independent draws from the model
    rows = synth.binned(rows.lazy(), table.keys, fill=False).collect()  # the numbers back in their bins
    model = classifier(rows, table.keys, seed)  # train on synthetic
    # end excerpt
    return Evaluation(
        name=name,
        seconds_release=seconds_release,
        oneway=marginal_errors(table, real, [(column,) for column in COLUMNS]),
        measured=marginal_errors(table, real, cliques),
        four_tv=four_tv,
        four_worst=four_worst,
        share=share_error(real_four, fitted_four),
        accuracy_synthetic=accuracy(model, test, table.keys),
        accuracy_real=accuracy(baseline, test, table.keys),
        probability_gap=probability_gap(model, baseline, test, table.keys),
    )


def error_lines(kind: str, name: str, errors: Sequence[MarginalError]) -> list[str]:
    return [
        f"{kind} {e.label} ({name}): tv={e.tv:.5f} largest cell error={e.worst:,} rows  std={e.std:.1f} bound={e.bound:.5f}"
        for e in errors
    ]


def share_line(e: Evaluation) -> str:
    error, cell, real_share, model_share = e.share
    return (
        f"share working by SEX, AGE, HWUSUAL ({e.name}, one run): largest error={error:.2f} points at {cell}"
        f"  real={100 * real_share:.2f}%  model={100 * model_share:.2f}%"
    )


def budget_line(name: str, rho: float, query: ContingencyTableQuery, mst: bool) -> str:
    """The deterministic numbers of a release: the one-way scale and threshold, the scale of a round's marginal (AIM)
    or of a pair (MST), and the epsilon the budget converts to."""
    line = f"{name}: oneway scale={query.oneway_scale:.4f} threshold={query.oneway_threshold}"
    if mst:
        rho_pair = rho / 2 * dp.mbi.MST().measure_split / (len(COLUMNS) - 1)
        line += f"  scale of a pair={synth.gaussian_scale(CONTRIBUTIONS, rho_pair):.4f}"
    else:
        rho_round = rho / 2 / ROUNDS * dp.mbi.AIM().measure_split
        line += f"  scale of a round's marginal={synth.gaussian_scale(CONTRIBUTIONS, rho_round):.4f}"
    return line + f"  epsilon at delta={DELTA:g}: {synth.epsilon_of_rho(rho, DELTA):.4f}"


def drawn(table: ContingencyTable, seed: int, method: str) -> pl.DataFrame:
    """ROWS synthetic rows drawn by `method`, with the numbers back in their bins."""
    return synth.binned(
        synth.synthesize(table, seed, rows=ROWS, method=method).lazy(), table.keys, fill=False
    ).collect()


def method_lines(name: str, table: ContingencyTable, real: pl.DataFrame, seed: int) -> list[str]:
    """The fitted model against rows drawn from it by the two methods: the one-way distances summed over the four
    columns, the four-way distance, and the pair whose distance rounding moves furthest from the model's."""
    frames = {"model": None, "round": drawn(table, seed, "round"), "sample": drawn(table, seed, "sample")}

    def tv(by: tuple[str, ...], frame: pl.DataFrame | None) -> float:
        other = table.project_melted(by) if frame is None else synth.marginal(frame, by)
        return distance(synth.marginal(real, by), other, by)[0]

    lines: list[str] = []
    oneway = {key: sum(tv((column,), frame) for column in COLUMNS) for key, frame in frames.items()}
    four = {key: tv(COLUMNS, frame) for key, frame in frames.items()}
    pairs = list(itertools.combinations(COLUMNS, 2))
    gaps = {pair: tv(pair, frames["round"]) - tv(pair, None) for pair in pairs}
    pair = max(pairs, key=lambda p: gaps[p])
    worst = {key: tv(pair, frame) for key, frame in frames.items()}
    for label, values in (("one-way tv summed", oneway), ("four-way tv", four), (f"pair {pair} tv", worst)):
        lines.append(
            f"{name} {label} (one run): model={values['model']:.5f}  round={values['round']:.5f}"
            f"  sample={values['sample']:.5f}"
        )
    return lines


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seed", type=int, default=0, help="seed of the real-row samples, synthesize() and the fit")
    args = parser.parse_args()
    seed: int = args.seed
    start = time.perf_counter()

    print(f"opendp version: {dp.__version__}  mbi version: {importlib.metadata.version('mbi')}")
    frame = pl.scan_parquet(practitioner_datasets.france_lfs())

    # Section 1: the release under evaluation.
    print(f"privacy unit: {dp.unit_of(contributions=CONTRIBUTIONS)!r}")
    print(f"privacy loss: {dp.loss_of(rho=RHO, delta=DELTA)!r}")
    # excerpt: eval-release
    aim_query = table_query(make_context(frame, RHO), RHO, dp.mbi.AIM(rounds=ROUNDS))
    table_aim, seconds_aim = synth.timed_release(aim_query)
    real = synth.binned(frame, table_aim.keys, fill=True).collect()  # the real rows in the table's keys and bins
    # end excerpt
    print(budget_line("aim rho=0.1", RHO, aim_query, mst=False))
    print(f"rows: {real.height:,}  keys ILOSTAT: {table_aim.keys['ILOSTAT'].to_list()}")
    print(f"aim rho=0.1 release (one run): {seconds_aim:.1f} s")
    for clique, std in synth.measured(table_aim):
        print(f"aim rho=0.1 measured: {clique} std={std:.2f}")

    shuffled = real.sample(n=2 * HOLDOUT, shuffle=True, seed=seed)  # seeded, so the same rows on every run
    train_real, test_real = shuffled.head(HOLDOUT), shuffled.tail(HOLDOUT)
    baseline = classifier(train_real, table_aim.keys, seed)  # train on real, the yardstick
    accuracy_real = accuracy(baseline, test_real, table_aim.keys)
    main_eval = evaluate("aim rho=0.1", table_aim, seconds_aim, real, test_real, baseline, seed)

    # Section 2: the workload error.
    for line in error_lines("one-way", main_eval.name, main_eval.oneway):
        print(line)
    for line in error_lines("measured", main_eval.name, main_eval.measured):
        print(line)
    worst = main_eval.worst_measured
    print(
        f"worst measured (aim rho=0.1, one run): {worst.label} tv={worst.tv:.5f}  bound/tv={worst.bound / worst.tv:.1f}"
    )

    # Section 3: a query nobody measured: the four-way table, and a conditional share read off it.
    four_cells = synth.cells(table_aim, COLUMNS)
    print(f"four-way cells: {four_cells}  measured by AIM: {COLUMNS in {c for c, _ in synth.measured(table_aim)}}")
    # excerpt: eval-unmeasured
    try:
        four_std = f"{table_aim.std(COLUMNS):.1f}"
    except ValueError as err:
        four_std = str(err)  # no measured marginal covers all four columns, so the table has no error bar for it
    # end excerpt
    print(f"std of the four-way table: {four_std}")
    best = main_eval.best_measured
    print(
        f"four-way (aim rho=0.1, one run): tv={main_eval.four_tv:.5f} largest cell error={main_eval.four_worst:,} rows"
        f"  best measured {best.label} tv={best.tv:.5f}  worst measured tv={worst.tv:.5f}"
    )
    true_share = working_share(synth.marginal(real, COLUMNS)).filter(pl.col("len") >= MIN_CELL)
    print(f"cells of the share with at least {MIN_CELL:,} real rows: {true_share.height}")
    print(share_line(main_eval))

    # Section 4: train on synthetic, test on real.
    print(f"held-out real rows: {HOLDOUT:,}  real training rows: {HOLDOUT:,}  synthetic training rows: {ROWS:,}")
    print(f"features: {design(test_real.head(1), table_aim.keys).shape[1]} indicators")
    print(f"share working in the held-out rows: {labels(test_real).mean():.4f}")
    print(f"train on real, test on real: accuracy={accuracy_real:.4f}")
    standard_error = math.sqrt(accuracy_real * (1 - accuracy_real) / HOLDOUT)
    print(f"standard error of an accuracy on {HOLDOUT:,} rows: {standard_error:.4f}")
    print(
        f"train on synthetic (aim rho=0.1, one run), test on real: accuracy={main_eval.accuracy_synthetic:.4f}"
        f"  gap={main_eval.gap:+.4f}  mean probability difference={main_eval.probability_gap:.4f}"
    )

    # Section 5: the same evaluation at two more budgets and for MST, each in a context of its own.
    evaluations = {main_eval.name: main_eval}
    tables = {main_eval.name: table_aim}
    for rho in BUDGETS:
        if rho == RHO:
            continue
        name = f"aim rho={rho:g}"
        query = table_query(make_context(frame, rho), rho, dp.mbi.AIM(rounds=ROUNDS))
        print(budget_line(name, rho, query, mst=False))
        table, seconds = synth.timed_release(query)
        evaluations[name] = evaluate(name, table, seconds, real, test_real, baseline, seed)
        tables[name] = table
    # excerpt: eval-mst
    mst_query = table_query(make_context(frame, RHO), RHO, dp.mbi.MST())
    table_mst, seconds_mst = synth.timed_release(mst_query)
    mst_eval = evaluate("mst rho=0.1", table_mst, seconds_mst, real, test_real, baseline, seed)
    # end excerpt
    print(budget_line("mst rho=0.1", RHO, mst_query, mst=True))
    evaluations[mst_eval.name] = mst_eval
    tables[mst_eval.name] = table_mst
    for clique, std in synth.measured(table_mst):
        print(f"mst rho=0.1 measured: {clique} std={std:.2f}")
    print(share_line(mst_eval))
    for name in ("aim rho=0.02", "aim rho=0.1", "aim rho=0.5", "mst rho=0.1"):
        e = evaluations[name]
        print(
            f"tuning ({name}, one run): release {e.seconds_release:.1f} s  cliques={len(e.measured)}"
            f"  one-way mean tv={e.oneway_mean:.5f}  worst measured tv={e.worst_measured.tv:.5f}"
            f"  four-way tv={e.four_tv:.5f}  share error={e.share[0]:.2f} points"
            f"  accuracy={e.accuracy_synthetic:.4f}  gap={e.gap:+.4f}  probability difference={e.probability_gap:.4f}"
        )
    print(f"budget the four contexts add up to if all were released from the same data: rho={sum(BUDGETS) + RHO:g}")

    # Section 6: rows and the sampling method.
    for column in COLUMNS:
        fitted = table_aim.project_melted((column,))["len"].to_numpy()  # the release's own counts, not the data
        shares = fitted / fitted.sum()
        print(
            f"sampling noise on one-way {column}: expected tv at {ROWS:,} rows={sampling_tv(shares, ROWS):.5f}"
            f"  at {real.height:,} rows={sampling_tv(shares, real.height):.5f}"
        )
    for name in ("aim rho=0.1", "mst rho=0.1"):
        for line in method_lines(name, tables[name], real, seed):
            print(line)

    # Section 7: what to publish.
    epsilon = synth.epsilon_of_rho(RHO, DELTA)
    # excerpt: eval-statement
    print(
        f"This synthetic dataset of {ROWS:,} rows was generated by AIM (OpenDP {dp.__version__}, mbi "
        f"{importlib.metadata.version('mbi')}, rounds={ROUNDS}, queries up to 3 columns) from one contingency table "
        f"over {', '.join(COLUMNS)} released under (rho = {RHO:g}, delta = {DELTA:g})-zero-concentrated differential "
        f"privacy, which is (epsilon = {epsilon:.2f}, delta = {2 * DELTA:g})-differential privacy, for adding or "
        f"removing all of one person's survey responses, at most {CONTRIBUTIONS} of them, under the central model. "
        f"The keys of SEX and the cuts of AGE and HWUSUAL were fixed in advance; nulls were recoded to "
        f"{synth.FILL}; the keys of ILOSTAT were released with a threshold of {aim_query.oneway_threshold}. "
        f"Drawing the rows by independent sampling is post-processing and spent nothing."
    )
    # end excerpt
    print(f"total run time (one run): {time.perf_counter() - start:.0f} s")


if __name__ == "__main__":
    main()
