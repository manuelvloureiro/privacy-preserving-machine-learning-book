"""Preparing microdata with OpenDP: the chapter "Preparing microdata" of the OpenDP book (Part 2).

    uv run python code/part02-tabular-data/opendp_preparing_microdata.py

Follows the documentation's Preparing Microdata tutorial, its Data Types page and the Expressions pages of the Polars
user guide (boolean, columns, manipulation, operators, string, temporal): a context over the France Labour Force
Survey with a privacy unit of 36 rows per person and a budget of (epsilon = 1, delta = 1e-7) split over four
queries, as in the documentation; the row-by-row rule for with_columns, filter and group_by; casting, filling nulls,
clipping and binning; boolean predicates and arithmetic between columns; a date built from YEAR and QUARTER and parsed
with a mandatory format; and a worked preparation that releases the mean difference between actual and usual hours
among people who work. It prints every number and every error message the page quotes, on labelled lines, in the
order of the page's sections.

Dataset: the France LFS sample of the OpenDP tutorials (3,811,932 quarterly responses from 1,682,306 people), read
through code/practitioner_datasets.py, which caches it in data/ as Parquet; the documentation reads the same file
through dp.examples.get_france_lfs_path(). Two toy frames built from literal values stand in for the LFS where the
documentation uses one (replace_strict without a default).

What is deterministic and what is not. OpenDP draws its noise from a cryptographic generator that cannot be seeded,
so the released values change on every run; they are printed under "released (one run)" and the test checks them
against the true values within three times their 95% accuracy (four times for the tables, which have several
values each). The noise scales, the accuracies, the thresholds, the error messages and the true values are
deterministic. The --seed flag is kept for the shape every chapter script shares; nothing here draws from it.

Library versions: opendp[polars] 0.16.0 with polars 1.36.1.
"""

import argparse
import itertools
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

import opendp.prelude as dp
import polars as pl
from opendp.extras.polars import LazyFrameQuery

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import practitioner_datasets

dp.enable_features("contrib")

EPSILON = 1.0
DELTA = 1e-7
CONTRIBUTIONS = 36  # a person answers at most once per quarter over 9 years
QUERIES = 4
MAX_LENGTH = 150_000 * 36  # public: the sample never holds more than 150,000 people per quarter
BREAKS = [0, 20, 40, 60, 80, 98]  # hours usually worked; 99 means "not applicable"
HOUR_BOUNDS = (0, 100)  # the documentation's bounds for a sum of hours
LOST_BOUNDS = (-40, 40)  # public choice: hours lost or gained in a week, clipped to half the range
QUARTER_END_MONTHS = [3, 6, 9, 12]
RETIREMENT_AGE = 64


# ---------------------------------------------------------------------------------------------------------------
# The context, as in the documentation
# ---------------------------------------------------------------------------------------------------------------


# excerpt: prep-context
def make_context(frame: pl.LazyFrame) -> dp.Context:
    """The documentation's context: 36 rows per person, (epsilon = 1, delta = 1e-7) split evenly over 4 queries."""
    return dp.Context.compositor(
        data=frame,
        privacy_unit=dp.unit_of(contributions=CONTRIBUTIONS),
        privacy_loss=dp.loss_of(epsilon=EPSILON, delta=DELTA),
        split_evenly_over=QUERIES,
        margins=[dp.polars.Margin(max_length=MAX_LENGTH)],  # public: a bound on the number of rows
    )


# end excerpt


def lazy_query(context: dp.Context) -> LazyFrameQuery:
    """A fresh query on the context, typed as the Polars query that a LazyFrame context hands out."""
    query = context.query()
    assert isinstance(query, LazyFrameQuery)
    return query


def reason(call: Callable[[], Any]) -> str:
    """The message of the OpenDPException that a call raises, or "accepted" if it raises none."""
    try:
        call()
    except dp.OpenDPException as err:
        return (err.message or "").splitlines()[0]
    return "accepted"


# ---------------------------------------------------------------------------------------------------------------
# Columns, filters and groups
# ---------------------------------------------------------------------------------------------------------------


def cut_labels(breaks: list[int], left_closed: bool) -> list[str]:
    """The bin labels that `.cut` produces, from the documentation, so that the keys can be public."""
    edges = ["-inf", *(str(b) for b in breaks), "inf"]
    left, right = ("[", ")") if left_closed else ("(", "]")
    return [f"{left}{lo}, {hi}{right}" for lo, hi in itertools.pairwise(edges)]


# excerpt: prep-cut
def binned_counts(context: dp.Context) -> LazyFrameQuery:
    """How many responses fall in each band of usual hours: a row-by-row `cut`, then a count per bin."""
    labels = pl.Series("HWUSUAL", cut_labels(BREAKS, left_closed=True), dtype=pl.Categorical)
    return (
        lazy_query(context)
        .with_columns(pl.col("HWUSUAL").cut(breaks=BREAKS, left_closed=True))  # shadows HWUSUAL with its bin
        .group_by("HWUSUAL")
        .agg(dp.len(signed=True))
        .with_keys(pl.LazyFrame([labels]))  # the bin labels are public, so no threshold is needed
    )


# end excerpt


def true_binned_counts(frame: pl.LazyFrame) -> dict[str, int]:
    bins = frame.with_columns(pl.col("HWUSUAL").cut(breaks=BREAKS, left_closed=True).cast(str))
    counts = bins.group_by("HWUSUAL").agg(pl.len()).collect()
    return {("null" if k is None else str(k)): int(v) for k, v in counts.iter_rows()}


def sample_and_aggregate(context: dp.Context) -> pl.DataFrame:
    """The documentation's stable group_by/agg: the minimum usual hours in each of 1,000 groups, then a DP mean."""
    usual: Any = pl.col("HWUSUAL")  # typed Any: OpenDP adds the `.dp` namespace to expressions at run time
    return (
        lazy_query(context)
        .filter(pl.col("HWUSUAL") != 99)
        .group_by(pl.col("PIDENT") % 1000)
        .agg(pl.col("HWUSUAL").min())
        .select(usual.cast(int).dp.mean((0, 30)))
        .summarize()
    )


def make_context_one_group(frame: pl.LazyFrame) -> dp.Context:
    """The same context, told that a person's rows all fall in one of the 1,000 groups."""
    return dp.Context.compositor(
        data=frame,
        privacy_unit=dp.unit_of(
            contributions=[
                dp.polars.Bound(per_group=CONTRIBUTIONS),
                dp.polars.Bound(by=[pl.col("PIDENT") % 1000], num_groups=1),
            ]
        ),
        privacy_loss=dp.loss_of(epsilon=EPSILON, delta=DELTA),
        split_evenly_over=QUERIES,
        margins=[dp.polars.Margin(max_length=MAX_LENGTH)],
    )


# ---------------------------------------------------------------------------------------------------------------
# Casting, filling and clipping
# ---------------------------------------------------------------------------------------------------------------


# excerpt: prep-cast
def hours_sum(context: dp.Context, as_integers: bool) -> pl.DataFrame:
    """The noise scale of a sum of usual hours, on the float column or after casting it to integers."""
    usual: Any = pl.col("HWUSUAL")  # typed Any: OpenDP adds the `.dp` namespace to expressions at run time
    column = usual.cast(int) if as_integers else usual
    return lazy_query(context).select(column.dp.sum(HOUR_BOUNDS)).summarize(alpha=0.05)


# end excerpt


# excerpt: prep-fill
def prepared_sum(context: dp.Context, steps: pl.Expr) -> pl.DataFrame:
    """A sum with noise on a column prepared in with_columns: OpenDP checks the steps left it non-null and bounded."""
    total: Any = pl.col("HWUSUAL").sum()
    return lazy_query(context).with_columns(steps).select(total.dp.noise()).summarize()


# end excerpt


def replace_strict_demo() -> tuple[str, str, str]:
    """replace_strict on a toy column: what happens to a value the mapping leaves out, with and without a default."""
    context = dp.Context.compositor(
        data=pl.LazyFrame({"code": [1, 2, 3] * 100}),
        privacy_unit=dp.unit_of(contributions=1),
        privacy_loss=dp.loss_of(epsilon=1.0),
        split_evenly_over=2,
        margins=[dp.polars.Margin(max_length=1000)],
    )
    partial = pl.col("code").replace_strict({1: "one", 2: "two"})
    without = lazy_query(context).group_by(partial).agg(dp.len(signed=True))
    without = without.with_keys(pl.LazyFrame({"code": ["one", "two", None]}))
    summarized = f"scale={without.summarize()['scale'].item():.1f}"
    released = reason(lambda: without.release().collect())
    complete = pl.col("code").replace_strict({1: "one", 2: "two"}, default="other")
    with_default = lazy_query(context).group_by(complete).agg(dp.len(signed=True))
    keys = with_default.with_keys(pl.LazyFrame({"code": ["one", "two", "other"]})).release().collect()
    return summarized, released, " ".join(sorted(keys["code"].to_list()))


# ---------------------------------------------------------------------------------------------------------------
# Predicates, arithmetic, strings and dates
# ---------------------------------------------------------------------------------------------------------------


# excerpt: prep-bool
def over_retirement_age(context: dp.Context) -> LazyFrameQuery:
    """How many responses come from people over 64: a boolean expression as the grouping column."""
    return (
        lazy_query(context)
        .group_by(pl.col("AGE") > RETIREMENT_AGE)
        .agg(dp.len(signed=True))
        .with_keys(pl.LazyFrame({"AGE": [True, False]}))  # a null age is in neither group and is dropped
    )


# end excerpt


# excerpt: prep-date
def quarter_end() -> pl.Expr:
    """The last month of the quarter as a date, built from YEAR and QUARTER and parsed with a fixed format."""
    year_month = pl.col("YEAR") * 100 + pl.col("QUARTER") * 3  # 2005 and 1 become 200503
    return year_month.cast(str).str.to_date(format="%Y%m")


def responses_per_month(context: dp.Context) -> LazyFrameQuery:
    """How many responses carry each quarter-end month, read off the parsed date."""
    return (
        lazy_query(context)
        .with_columns(DATE=quarter_end())
        .group_by(pl.col("DATE").dt.month().alias("MONTH"))
        .agg(dp.len(signed=True))
        .with_keys(pl.LazyFrame({"MONTH": pl.Series(QUARTER_END_MONTHS, dtype=pl.Int8)}))
    )


# end excerpt


def true_counts(frame: pl.LazyFrame, key: pl.Expr, name: str) -> dict[str, int]:
    counts = frame.group_by(key.alias(name)).agg(pl.len()).collect()
    return {("null" if k is None else str(k).lower()): int(v) for k, v in counts.iter_rows()}


# ---------------------------------------------------------------------------------------------------------------
# The worked preparation: hours lost to absence
# ---------------------------------------------------------------------------------------------------------------


# excerpt: prep-worked
def prepare[Frame: (LazyFrameQuery, pl.LazyFrame)](frame: Frame) -> Frame:
    """The rows of people who work, with their usual hours known, and the hours they lost that week, clipped."""
    low, high = LOST_BOUNDS
    return (
        frame.filter(pl.col("ILOSTAT") == 1)  # working for pay or profit
        .with_columns(pl.col("HWUSUAL").cast(int).fill_null(0), pl.col("HWACTUAL").cast(int))
        .filter((pl.col("HWUSUAL") > 0) & (pl.col("HWUSUAL") < 99))  # 0 is "varies", 99 "not applicable"
        .with_columns(pl.col("HWACTUAL").fill_null(pl.col("HWUSUAL")))  # unknown actual hours: assume the usual
        .with_columns(LOST=(pl.col("HWACTUAL") - pl.col("HWUSUAL")).clip(low, high))
    )


# end excerpt


def mean_error_bar(sum_accuracy: float, length_accuracy: float, true_mean: float, true_length: int) -> float:
    """How far the released mean can sit from the true one when the sum and the length are within their bars."""
    return (sum_accuracy + abs(true_mean) * length_accuracy) / (true_length - length_accuracy)


# ---------------------------------------------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------------------------------------------


def summary_lines(summary: pl.DataFrame, label: str) -> list[str]:
    out: list[str] = []
    for row in summary.iter_rows(named=True):
        accuracy = row.get("accuracy")
        acc = "" if accuracy is None else f" 95% accuracy={accuracy:,.1f}"
        threshold = row.get("threshold")
        thr = "" if threshold is None else f" threshold={threshold:,}"
        out.append(
            f"{label}: {row['column']} {row['aggregate']}, {row['distribution']}, scale={row['scale']:,.2f}{acc}{thr}"
        )
    return out


def accuracy_of(summary: pl.DataFrame, aggregate: str) -> float:
    return float(summary.filter(pl.col("aggregate") == aggregate)["accuracy"].item())


def counts_line(counts: dict[str, int], order: list[str]) -> str:
    return "  ".join(f"{key}={counts.get(key, 0):,}" for key in order)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seed", type=int, default=0, help="unused: nothing here samples (OpenDP's noise has no seed)")
    parser.parse_args()

    frame = pl.scan_parquet(practitioner_datasets.france_lfs())
    rows, people = frame.select(pl.len(), pl.col("PIDENT").n_unique()).collect().row(0)
    print(f"opendp version: {dp.__version__}")
    print(f"rows: {rows:,}  people: {people:,}")

    # Section 1: prepare inside the query. The context reads the schema; everything after it is accounted for.
    context = make_context(frame)
    print(f"privacy unit: {dp.unit_of(contributions=CONTRIBUTIONS)!r}")
    print(f"privacy loss: {dp.loss_of(epsilon=EPSILON, delta=DELTA)!r}")
    epsilon_q, delta_q = (context.d_mids or [])[0]
    print(f"budget per query: epsilon={epsilon_q:.4g} delta={delta_q:.3g}")
    schema = frame.collect_schema()
    print("schema: " + "  ".join(f"{name}={dtype}" for name, dtype in schema.items()))

    # Section 2: columns, filters and groups. A cut in with_columns, counted per bin with public keys; without the
    # keys the same query needs a threshold; a filter before a sum; the documentation's stable group_by/agg.
    bins = binned_counts(context)
    bin_summary = bins.summarize(alpha=0.05)
    for line in summary_lines(bin_summary, "binned counts"):
        print(line)
    no_keys = lazy_query(context).with_columns(pl.col("HWUSUAL").cut(breaks=BREAKS, left_closed=True))
    for line in summary_lines(no_keys.group_by("HWUSUAL").agg(dp.len(signed=True)).summarize(), "without keys"):
        print(line)
    bin_order = ["null", *cut_labels(BREAKS, left_closed=True)]
    true_bins = true_binned_counts(frame)
    print("true bins: " + counts_line(true_bins, bin_order))
    usual: Any = pl.col("HWUSUAL")
    filtered = lazy_query(context).with_columns(usual.cast(int)).filter(pl.col("HWUSUAL") != 99)
    for line in summary_lines(filtered.select(usual.dp.sum((0, 80))).summarize(alpha=0.05), "filtered sum"):
        print(line)
    for line in summary_lines(sample_and_aggregate(context), "sample and aggregate"):
        print(line)
    for line in summary_lines(sample_and_aggregate(make_context_one_group(frame)), "one group per person"):
        print(line)

    # Section 4: casting, filling and clipping. The float and the integer sum; the order of fill and clip.
    for line in summary_lines(hours_sum(context, as_integers=False), "float sum"):
        print(line)
    for line in summary_lines(hours_sum(context, as_integers=True), "integer sum"):
        print(line)
    for line in summary_lines(prepared_sum(context, usual.cast(int).fill_null(0).clip(0, 100)), "prepared sum"):
        print(line)
    print("cast without fill_null: " + reason(lambda: prepared_sum(context, usual.cast(int).clip(0, 100))))
    print("clip before fill_null: " + reason(lambda: prepared_sum(context, usual.cast(int).clip(0, 100).fill_null(0))))
    print("fill_null before fill_nan: " + reason(lambda: prepared_sum(context, usual.fill_null(0.0).clip(0.0, 100.0))))
    both = usual.fill_null(0.0).fill_nan(0.0).clip(0.0, 100.0)
    for line in summary_lines(prepared_sum(context, both), "float prepared sum"):
        print(line)
    actual: Any = pl.col("HWACTUAL")
    column_fill = (
        lazy_query(context).with_columns(actual.cast(int).fill_null(0)).with_columns(usual.cast(int).fill_null(actual))
    )
    for line in summary_lines(column_fill.select(usual.dp.sum(HOUR_BOUNDS)).summarize(), "filled from a column"):
        print(line)
    nullable_fill = lazy_query(context).with_columns(usual.cast(int).fill_null(actual.cast(int)))
    print(
        "filled from a nullable column: " + reason(lambda: nullable_fill.select(usual.dp.sum(HOUR_BOUNDS)).summarize())
    )
    print(
        "drop_nulls in with_columns: "
        + reason(lambda: lazy_query(context).with_columns(usual.drop_nulls()).select(dp.len(signed=True)).summarize())
    )
    summarized, released, keys = replace_strict_demo()
    print(f"replace_strict without default: summarize {summarized}; release: {released}")
    print(f"replace_strict with default: released keys {keys}")

    # Section 5: predicates and arithmetic. A boolean grouping column; what is_in and abs do in 0.16.0.
    over = over_retirement_age(context)
    for line in summary_lines(over.summarize(alpha=0.05), "over 64"):
        print(line)
    true_over = true_counts(frame, pl.col("AGE") > RETIREMENT_AGE, "AGE")
    print("true over 64: " + counts_line(true_over, ["true", "false", "null"]))
    print(
        "is_in: "
        + reason(
            lambda: lazy_query(context).filter(pl.col("ILOSTAT").is_in([1, 2])).select(dp.len(signed=True)).summarize()
        )
    )
    either = (pl.col("ILOSTAT") == 1) | (pl.col("ILOSTAT") == 2)
    for line in summary_lines(
        lazy_query(context).filter(either).select(dp.len(signed=True)).summarize(alpha=0.05), "employed"
    ):
        print(line)
    difference: Any = pl.col("HWACTUAL") - pl.col("HWUSUAL")
    print("abs: " + reason(lambda: lazy_query(context).select(difference.abs().dp.sum((0, 80))).summarize()))

    # Section 6: strings and dates. The format is mandatory; nanoseconds and concat_str are refused.
    year: Any = pl.col("YEAR").cast(str)
    print(
        "to_date without format: "
        + reason(lambda: lazy_query(context).group_by(year.str.to_date()).agg(dp.len(signed=True)).summarize())
    )
    nanoseconds = year.str.to_datetime(format="%Y", time_unit="ns")
    print(
        "nanosecond datetime: "
        + reason(lambda: lazy_query(context).group_by(nanoseconds).agg(dp.len(signed=True)).summarize())
    )
    joined = pl.concat_str("YEAR", pl.col("QUARTER") * 3, pl.lit("01"), separator="-")
    print("concat_str: " + reason(lambda: lazy_query(context).group_by(joined).agg(dp.len(signed=True)).summarize()))
    months = responses_per_month(context)
    for line in summary_lines(months.summarize(alpha=0.05), "per month"):
        print(line)
    first_dates = frame.select(pl.col("YEAR"), pl.col("QUARTER"), DATE=quarter_end()).unique().sort("DATE").head(2)
    print("first dates: " + "  ".join(f"{y} Q{q} -> {d}" for y, q, d in first_dates.collect().iter_rows()))
    true_months = true_counts(frame.with_columns(DATE=quarter_end()), pl.col("DATE").dt.month(), "MONTH")
    print("true per month: " + counts_line(true_months, [str(m) for m in QUARTER_END_MONTHS]))

    # Section 7: the worked preparation, summarized, checked against the truth, then released.
    lost: Any = pl.col("LOST")
    hours_lost = prepare(lazy_query(context)).select(lost.dp.mean(LOST_BOUNDS))
    lost_summary = hours_lost.summarize(alpha=0.05)
    for line in summary_lines(lost_summary, "hours lost"):
        print(line)
    wide = prepare(lazy_query(context)).select(lost.dp.mean((-80, 80))).summarize(alpha=0.05)
    for line in summary_lines(wide.filter(pl.col("aggregate") == "Sum"), "hours lost, bounds (-80, 80)"):
        print(line)
    raw = pl.col("HWACTUAL") - pl.col("HWUSUAL")
    truth = prepare(frame).select(
        pl.len().alias("n"),
        pl.col("LOST").sum().alias("sum"),
        pl.col("LOST").mean().alias("mean"),
        raw.mean().alias("raw_mean"),
        ((raw < LOST_BOUNDS[0]) | (raw > LOST_BOUNDS[1])).sum().alias("clipped"),
    )
    n, total, mean, raw_mean, clipped = truth.collect().row(0)
    print(f"true prepared rows: {n:,}  true sum of hours lost: {total:,}  true mean: {mean:.4f}")
    print(f"rows the clip changes: {clipped:,} ({100 * clipped / n:.3f}%)  mean without the clip: {raw_mean:.4f}")
    bar = mean_error_bar(accuracy_of(lost_summary, "Sum"), accuracy_of(lost_summary, "Length"), mean, n)
    print(f"mean error bar at 95%: {bar:.4f}")

    print("released (one run):")
    bins_hat = bins.release().collect().sort("HWUSUAL")
    print("  bins: " + "  ".join(f"{k}={v:,}" for k, v in bins_hat.iter_rows()))
    over_hat = over_retirement_age(context).release().collect().sort("AGE", descending=True)
    print("  over 64: " + "  ".join(f"{str(k).lower()}={v:,}" for k, v in over_hat.iter_rows()))
    months_hat = months.release().collect().sort("MONTH")
    print("  per month: " + "  ".join(f"{k}={v:,}" for k, v in months_hat.iter_rows()))
    mean_hat = float(hours_lost.release().collect().item())
    print(f"  hours lost: mean {mean_hat:.4f}  (true {mean:.4f}, within {bar:.4f} with probability 0.95)")
    print(f"budget spent: {len(context.current_privacy_loss())} of {QUERIES} queries")


if __name__ == "__main__":
    main()
