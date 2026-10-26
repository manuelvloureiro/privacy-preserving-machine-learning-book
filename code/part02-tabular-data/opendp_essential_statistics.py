"""Essential statistics with OpenDP: the chapter "Essential statistics" of the OpenDP book (Part 2).

    uv run python code/part02-tabular-data/opendp_essential_statistics.py

Follows the documentation's "Essential Statistics" tutorial and the "Aggregation" page of the Polars user guide: on
the France Labour Force Survey sample, under a privacy unit of 36 rows per person and a budget of epsilon = 1 split
evenly over five queries, it releases the number of responses, the total of usual weekly hours (`HWUSUAL`), a sum
and a count in one query for a mean, a bounded-DP mean with the length invariant, the median and the three quartiles
of the hours. It prints every number the page quotes, on labelled lines, in the order of the page's sections.

Dataset: the France LFS sample from the OpenDP tutorials (Eurostat public microdata, prepared by the OpenDP team),
loaded by code/practitioner_datasets.py and scanned lazily as Parquet: 3,811,932 quarterly responses from 1,682,306
people. `HWUSUAL` holds hours usually worked per week, with 99 for "not applicable" and nulls for no answer.

What is deterministic and what is not. OpenDP draws its noise from a cryptographic generator that cannot be seeded,
so the released values change on every run; they are printed under "released (one run)" and the test checks them
against the true values within three times their 95% accuracy, and the median and quartiles against the true ones
exactly (the score of the true quartile is tens of thousands below every other candidate's, against a noise scale
of at most 3,240, so any other candidate has a probability below 1e-15). Everything else is deterministic: the
unit, the budget and its split, the noise scales, the accuracies, the refusals and the true values. The --seed
argument exists for symmetry with the other scripts; nothing here samples.

The chapter's five releases come from one context. The comparisons the page makes (the count variants, the float
sum, the sum without `fill_null`, the quantile without a margin, a quartile alone, the sum of the unfiltered column)
each take a share of a budget, so they run on fresh contexts with the same settings, as the documentation's own
examples do.

Library versions: opendp[polars] 0.16.0 with polars 1.36.1.
"""

import argparse
import math
import re
import sys
import warnings
from collections.abc import Callable
from pathlib import Path
from typing import Any, Literal

import opendp.prelude as dp
import polars as pl
from opendp.extras.polars import LazyFrameQuery
from opendp.mod import OpenDPException

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import practitioner_datasets

dp.enable_features("contrib")

EPSILON = 1.0
CONTRIBUTIONS = 36  # one row per quarter over 9 years
QUERIES = 5
MAX_LENGTH = 150_000 * 36  # public: at most 150,000 responses per quarter, 36 quarters
NOT_APPLICABLE = 99.0  # the code for "not applicable" in HWUSUAL
HOURS_BOUNDS = (0, 80)  # public: a working week, in hours
IMPUTED_HOURS = 35  # public: the legal working week in France, for a missing answer
CANDIDATES = list(range(20, 60))  # the values the median and the quartiles are chosen from
QUARTILES = [0.25, 0.5, 0.75]
HUGE_LENGTH = 1_000_000_000  # a careless bound on the number of rows, to show what it does to a float sum


# ---------------------------------------------------------------------------------------------------------------
# The context
# ---------------------------------------------------------------------------------------------------------------


# excerpt: stats-context
def make_context(frame: pl.LazyFrame, invariant: Literal["keys", "lengths"] | None = None) -> dp.Context:
    """The chapter's context: 36 rows per person, epsilon = 1 split evenly over 5 queries, and one public margin."""
    return dp.Context.compositor(
        data=frame,
        privacy_unit=dp.unit_of(contributions=CONTRIBUTIONS),  # neighbours add or remove one person's 36 rows
        privacy_loss=dp.loss_of(epsilon=EPSILON),  # pure differential privacy
        split_evenly_over=QUERIES,  # each query gets epsilon / 5
        margins=[dp.polars.Margin(max_length=MAX_LENGTH, invariant=invariant)],  # a public bound on the rows
    )


# end excerpt


def context_with(frame: pl.LazyFrame, margins: list[dp.polars.Margin] | None) -> dp.Context:
    """The same context with other margins, or none, for the comparisons."""
    return dp.Context.compositor(
        data=frame,
        privacy_unit=dp.unit_of(contributions=CONTRIBUTIONS),
        privacy_loss=dp.loss_of(epsilon=EPSILON),
        split_evenly_over=QUERIES,
        margins=margins,
    )


def hours_worked() -> Any:
    """`HWUSUAL` as an integer column, the expression the sum, mean and quantiles aggregate.

    Typed Any because OpenDP adds the `.dp` namespace (`.dp.sum`, `.dp.median`) to Polars expressions at run time,
    which a type checker cannot see.
    """
    return pl.col("HWUSUAL").cast(int)


def applicable(context: dp.Context) -> LazyFrameQuery:
    """A query on the responses whose usual hours apply (the code 99 means "not applicable")."""
    return context.query().filter(pl.col("HWUSUAL") != NOT_APPLICABLE)


# ---------------------------------------------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------------------------------------------


def summary_lines(summary: pl.DataFrame, label: str) -> list[str]:
    """One labelled line per statistic of a `summarize` table."""
    out: list[str] = []
    for row in summary.iter_rows(named=True):
        accuracy = row.get("accuracy")
        acc = "none" if accuracy is None or math.isnan(accuracy) else f"{accuracy:,.1f}"
        out.append(
            f"{label} {row['column']}: {row['aggregate']}, {row['distribution']}, "
            f"scale={row['scale']:,.2f} 95% accuracy={acc}"
        )
    return out


def scale_of(summary: pl.DataFrame, column: str, aggregate: str | None = None) -> float:
    return float(statistic(summary, column, aggregate)["scale"].item())


def accuracy_of(summary: pl.DataFrame, column: str, aggregate: str | None = None) -> float:
    return float(statistic(summary, column, aggregate)["accuracy"].item())


def statistic(summary: pl.DataFrame, column: str, aggregate: str | None) -> pl.DataFrame:
    """The row of a `summarize` table for one column, and one aggregate when the column has several."""
    rows = summary.filter(pl.col("column") == column)
    return rows if aggregate is None else rows.filter(pl.col("aggregate") == aggregate)


def refusal(build: Callable[[], object]) -> str:
    """The library's reason for refusing a query, or "accepted" when it does not refuse."""
    try:
        build()
    except OpenDPException as err:
        text = err.message or ""
        m = re.search(r'MakeTransformation\("([^"]*)"\)', text)
        return m.group(1) if m else text.strip()
    return "accepted"


def quantile_scores(values: pl.Series, alpha: float) -> list[int]:
    """OpenDP's score of each candidate: |(1 - alpha) #(x < c) - alpha #(x > c)|, counted in alpha's denominator.

    The quartiles have denominator 4 and the median 2, which is why their scores, and the sensitivities and
    scales that follow from them, differ by a factor of three at the same share of the budget.
    """
    den = 4 if alpha in (0.25, 0.75) else 2
    num = round(alpha * den)
    return [abs((den - num) * int((values < c).sum()) - num * int((values > c).sum())) for c in CANDIDATES]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--seed", type=int, default=0, help="unused: nothing here samples, and OpenDP's noise has no seed"
    )
    parser.parse_args()

    frame = pl.scan_parquet(practitioner_datasets.france_lfs())
    low, high = HOURS_BOUNDS
    share = EPSILON / QUERIES
    hours_int = hours_worked()

    # Section 1: the context. Five releases come from it; every comparison below uses a fresh one.
    print(f"opendp version: {dp.__version__}")
    print(f"privacy unit: {dp.unit_of(contributions=CONTRIBUTIONS)!r}")
    print(f"privacy loss: {dp.loss_of(epsilon=EPSILON)!r}")
    context = make_context(frame)
    print(f"budget per query: {[round(d, 4) for d in (context.d_mids or [])]}")
    print(f"margin: max_length={MAX_LENGTH:,}")

    # Section 2: counting. The frame length, read through summarize at two levels, then the count variants.
    # excerpt: stats-count
    count_query = context.query().select(dp.len(signed=True))
    count_summary = count_query.summarize(alpha=0.05)
    dp_count = count_query.release().collect().item()
    # end excerpt
    print("count summary (alpha=0.05):")
    print(count_summary)
    count_scale = scale_of(count_summary, "len")
    count_accuracy = accuracy_of(count_summary, "len")
    count_accuracy_99 = accuracy_of(count_query.summarize(alpha=0.01), "len")
    print(f"count: scale={count_scale:.1f} 95% accuracy={count_accuracy:.2f} 99% accuracy={count_accuracy_99:.2f}")
    print(f"sensitivity {CONTRIBUTIONS} over epsilon {share:.1f}: scale {CONTRIBUTIONS / share:.1f}")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        make_context(frame).query().select(dp.len()).summarize()
    print(f"without signed=True: {type(caught[0].message).__name__}: {str(caught[0].message).strip()}")
    variants = make_context(frame)
    hwusual: Any = pl.col("HWUSUAL")  # Any: the .dp namespace is added at run time
    for name, expr in (
        ("len", hwusual.dp.len(signed=True)),
        ("count", hwusual.dp.count(signed=True)),
        ("null_count", hwusual.dp.null_count(signed=True)),
        ("n_unique", hwusual.dp.n_unique(signed=True)),
    ):
        print(f"HWUSUAL {name} alone: scale={scale_of(variants.query().select(expr).summarize(), 'HWUSUAL'):.1f}")
    # excerpt: stats-variants
    five_counts = variants.query().select(
        dp.len(signed=True),  # rows of the frame
        hwusual.dp.len(signed=True).alias("expr_len"),  # rows of the column, nulls included
        hwusual.dp.count(signed=True).alias("count"),  # rows of the column that are not null
        hwusual.dp.null_count(signed=True).alias("null_count"),  # rows of the column that are null
        hwusual.dp.n_unique(signed=True).alias("n_unique"),  # distinct values, null among them
    )
    five_summary = five_counts.summarize(alpha=0.05)
    # end excerpt
    scales = sorted({round(float(s), 1) for s in five_summary["scale"]})
    print(f"five counts in one query: scale={scales[0]:.1f} each, 95% accuracy={accuracy_of(five_summary, 'len'):.1f}")

    # Section 4: the sum. The margin's max_length, the integer cast and the imputation.
    # excerpt: stats-sum
    hours: Any = pl.col("HWUSUAL").cast(int)  # Any: the .dp namespace is added at run time
    sum_query = (
        context.query()
        .filter(pl.col("HWUSUAL") != 99.0)  # 99 means "not applicable"
        .select(hours.fill_null(35).dp.sum(bounds=(0, 80)))  # fill, clip, sum, add noise
    )
    sum_summary = sum_query.summarize(alpha=0.05)
    dp_sum = sum_query.release().collect().item()
    # end excerpt
    for line in summary_lines(sum_summary, "sum"):
        print(line)
    sum_accuracy = accuracy_of(sum_summary, "HWUSUAL")
    print(
        f"sensitivity {CONTRIBUTIONS} x {high} = {CONTRIBUTIONS * high:,} over epsilon {share:.1f}: "
        f"scale {CONTRIBUTIONS * high / share:,.0f}"
    )
    no_margin = context_with(frame, None)

    def sum_without_margin() -> pl.DataFrame:
        return applicable(no_margin).select(hours_int.dp.sum(bounds=HOURS_BOUNDS)).summarize()

    print(f"without a margin: {refusal(sum_without_margin)}")
    float_hours: Any = hwusual.fill_null(float(IMPUTED_HOURS))
    comparisons = make_context(frame)
    float_summary = applicable(comparisons).select(float_hours.dp.sum(bounds=(0.0, 80.0))).summarize(alpha=0.05)
    for line in summary_lines(float_summary, f"float sum, max_length {MAX_LENGTH:,}:"):
        print(line)
    huge = context_with(frame, [dp.polars.Margin(max_length=HUGE_LENGTH)])
    for label, expr in (
        ("float sum", float_hours.dp.sum(bounds=(0.0, 80.0))),
        ("int sum", hours_int.dp.sum(bounds=HOURS_BOUNDS)),
    ):
        summary = applicable(huge).select(expr).summarize(alpha=0.05)
        for line in summary_lines(summary, f"{label}, max_length {HUGE_LENGTH:,}:"):
            print(line)
    no_fill = applicable(comparisons).select(hours_int.dp.sum(bounds=HOURS_BOUNDS)).summarize(alpha=0.05)
    midpoint = (low + high) // 2
    print(
        f"sum without fill_null: scale={scale_of(no_fill, 'HWUSUAL'):,.1f} (accepted; a null becomes the midpoint {midpoint})"
    )

    def sum_by_hand() -> pl.DataFrame:
        return applicable(comparisons).select(hours_int.clip(low, high).sum().dp.noise()).summarize()

    print(f"clip, sum and laplace by hand, without fill_null: {refusal(sum_by_hand)}")
    # The whole column, nulls included, to confirm the imputation on a release: the two candidate truths are
    # 20,819 x 40 hours apart, far more than the error bar.
    unfiltered_query = make_context(frame).query().select(hours_int.dp.sum(bounds=HOURS_BOUNDS))
    dp_unfiltered = unfiltered_query.release().collect().item()

    # Section 5: the mean, as a sum and a count in one query, then with the length invariant.
    # excerpt: stats-mean-joint
    joint_query = (
        context.query()
        .filter(pl.col("HWUSUAL") != 99.0)
        .select(hours.dp.sum(bounds=(0, 80)), dp.len(signed=True))  # a null becomes the midpoint, 40
    )
    joint_summary = joint_query.summarize(alpha=0.05)
    joint = joint_query.release().collect().with_columns(mean=pl.col("HWUSUAL") / pl.col("len"))
    # end excerpt
    for line in summary_lines(joint_summary, "joint"):
        print(line)
    mean_unbounded = applicable(comparisons).select(hours_int.dp.mean(bounds=HOURS_BOUNDS)).summarize(alpha=0.05)
    for line in summary_lines(mean_unbounded, "dp.mean on the add/remove context:"):
        print(line)
    # excerpt: stats-mean-bounded
    applicable_frame = frame.filter(pl.col("HWUSUAL") != 99.0)  # filtered outside the context: see the text
    bounded = make_context(applicable_frame, invariant="lengths")  # the number of rows is public
    mean_query = bounded.query().select(hours.dp.mean(bounds=(0, 80)))
    mean_summary = mean_query.summarize(alpha=0.05)
    dp_mean = mean_query.release().collect().item()
    # end excerpt
    for line in summary_lines(mean_summary, "bounded mean"):
        print(line)
    # A filter inside the bounded context makes the number of rows unknown again, so the invariant is lost.
    filtered_inside = bounded.query().filter(pl.col("SEX") == 1).select(hours_int.dp.mean(bounds=HOURS_BOUNDS))
    for line in summary_lines(
        filtered_inside.summarize(alpha=0.05), "dp.mean after a filter inside the bounded context:"
    ):
        print(line)

    # Section 6: the median and the quartiles, chosen among candidates.
    # excerpt: stats-median
    candidates = list(range(20, 60))  # whole hours from 20 to 59
    median_query = context.query().filter(pl.col("HWUSUAL") != 99.0).select(hours.dp.median(candidates))
    median_summary = median_query.summarize(alpha=0.05)
    dp_median = median_query.release().collect().item()
    # end excerpt
    for line in summary_lines(median_summary, "median"):
        print(line)
    print(
        f"score sensitivity {CONTRIBUTIONS}: scale 2 x {CONTRIBUTIONS} / {share:.1f} = {2 * CONTRIBUTIONS / share:.0f}"
    )

    def median_without_margin() -> pl.DataFrame:
        return applicable(no_margin).select(hours_int.dp.median(candidates)).summarize()

    print(f"median without a margin: {refusal(median_without_margin)}")
    # excerpt: stats-quartiles
    quartile_query = (
        context.query()
        .filter(pl.col("HWUSUAL") != 99.0)
        .select(hours.dp.quantile(a, candidates).alias(f"{a}-Quantile") for a in [0.25, 0.5, 0.75])
    )
    quartile_summary = quartile_query.summarize()
    dp_quartiles = quartile_query.release().collect().row(0)
    # end excerpt
    for line in summary_lines(quartile_summary, "quartiles"):
        print(line)
    alone = applicable(comparisons).select(hours_int.dp.quantile(0.25, candidates)).summarize()
    print(f"0.25-quantile alone: scale={scale_of(alone, 'HWUSUAL'):,.1f}")

    # Section 7: the dev-time check against the true values (this public dataset only).
    true_count = frame.select(pl.len()).collect().item()
    column = frame.select(pl.col("HWUSUAL")).collect()["HWUSUAL"]
    applicable_hours = column.filter(column != NOT_APPLICABLE)
    true_applicable = applicable_hours.len()
    true_sum = int(applicable_hours.cast(pl.Int64).fill_null(IMPUTED_HOURS).clip(low, high).sum())
    true_mean = true_sum / true_applicable
    mean_accuracy = accuracy_of(mean_summary, "HWUSUAL", "Sum") / true_applicable
    values = applicable_hours.cast(pl.Int64)
    true_quartiles = [int(values.quantile(a, interpolation="lower") or 0) for a in QUARTILES]
    true_unfiltered = int(column.cast(pl.Int64).fill_null(midpoint).clip(low, high).sum())
    true_unfiltered_dropped = int(column.cast(pl.Int64).drop_nulls().clip(low, high).sum())
    print(f"true responses: {true_count:,}")
    print(
        f"true responses with HWUSUAL not 99: {true_applicable:,} "
        f"(nulls among them: {applicable_hours.null_count()}; in the whole column: {column.null_count():,})"
    )
    print(
        f"true sum of hours after the filter, clipped to [{low}, {high}]: {true_sum:,} "
        f"(values above {high}: {int((applicable_hours > high).sum())})"
    )
    print(
        f"true sum of the whole column, clipped: {true_unfiltered:,} with nulls as {midpoint}, {true_unfiltered_dropped:,} with nulls dropped"
    )
    print(f"true mean: {true_mean:.2f}  (bounded mean 95% accuracy: {mean_accuracy:.4f})")
    print(f"true quartiles: {true_quartiles[0]}, {true_quartiles[1]}, {true_quartiles[2]}")
    gaps: list[str] = []
    for alpha, true_q in zip(QUARTILES, true_quartiles, strict=True):
        scores = quantile_scores(values, alpha)
        best = scores[CANDIDATES.index(true_q)]
        gaps.append(f"{alpha}: {min(s for c, s in zip(CANDIDATES, scores, strict=True) if c != true_q) - best:,}")
    print(f"score gap from the true quartile to the nearest other candidate: {'  '.join(gaps)}")

    # The releases of one run: random every time.
    print("released (one run):")
    print(f"  responses {dp_count:,}  (true {true_count:,}, within {count_accuracy:.1f} with probability 0.95)")
    print(f"  sum of hours {dp_sum:,}  (true {true_sum:,}, within {sum_accuracy:,.1f})")
    joint_sum, joint_len, joint_mean = joint.row(0)
    print(f"  joint: sum {joint_sum:,}  count {joint_len:,}  mean {joint_mean:.2f}  (true {true_mean:.2f})")
    print(f"  bounded mean {dp_mean:.4f}  (true {true_mean:.4f}, within {mean_accuracy:.4f})")
    print(
        f"  median {dp_median}  quartiles {dp_quartiles[0]}, {dp_quartiles[1]}, {dp_quartiles[2]}  "
        f"(true {true_quartiles[1]}; {true_quartiles[0]}, {true_quartiles[1]}, {true_quartiles[2]})"
    )
    print(f"  sum of the whole column {dp_unfiltered:,}  (true with nulls as {midpoint}: {true_unfiltered:,})")


if __name__ == "__main__":
    main()
