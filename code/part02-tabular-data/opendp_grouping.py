"""Grouped counts with OpenDP: the chapter "Grouping" of the OpenDP book (Part 2).

    uv run python code/part02-tabular-data/opendp_grouping.py

Follows the documentation's Grouping tutorial, in its order: why grouping keys are dangerous; stable keys, where
OpenDP releases only the groups whose noisy count clears a threshold that costs delta; explicit keys, where a public
key list is joined in and every key is released, zeros included; invariant keys and invariant lengths, margins that
release the keys or the group sizes in the clear; what max_groups and max_length change; and how the noise per group
compares with the ungrouped count. It prints every number the page quotes, on labelled lines, in the order of the
page's sections, and a dev-time check against the true group counts, which this public data allows.

Data: the France Labour Force Survey sample of the OpenDP tutorials, loaded through code/practitioner_datasets.py
(the documentation reads the same file with dp.examples.get_france_lfs_path()). 3,811,932 quarterly responses from
1,682,306 people; a person appears at most once per quarter over 9 years, so at most 36 times, which is the privacy
unit (dp.unit_of(contributions=36)). The budgets are the documentation's: epsilon = 1/4 with delta = 1e-7 for the
stable-keys count, epsilon = 1/4 for the counts with explicit or invariant keys, and epsilon = 1 for the mean hours.

What is deterministic and what is not. OpenDP draws its noise from a cryptographic generator that cannot be seeded,
so the released counts and means change on every run; they are printed under "released (one run)" and the test
checks them against the true values within a few times their 95% accuracy. Everything else is the same on every run:
the scales, accuracies and thresholds, the error messages, the true counts and the tail probabilities of the dev-time
check. The --seed flag is accepted like in every script of the book, but nothing in this chapter is sampled.

Library versions: opendp[polars] 0.16.0 with polars 1.36.1.
"""

import argparse
import math
import re
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import opendp.prelude as dp
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import practitioner_datasets

dp.enable_features("contrib")

CONTRIBUTIONS = 36  # a person answers at most once per quarter, 9 years x 4 quarters
EPSILON = 0.25  # the documentation's budget for one grouped count
DELTA = 1e-7  # pays for the threshold of the stable keys
MEAN_EPSILON = 1.0  # the documentation's budget for the mean hours by sex
HOURS_BOUNDS = (0, 98)  # the codebook: 0 to 98 are hours, 99 is "not applicable"
MAX_LENGTH = 150_000 * CONTRIBUTIONS  # public: average quarterly sample size times the number of quarters
YEARS = list(range(2005, 2014))  # the survey years, public
STATUSES = [1, 2, 3, 4, 5, 9]  # every ILOSTAT code in the codebook; 4 and 5 never occur in this sample
GROUPS_IN_TRUTH = 16  # the number of AGE x ILOSTAT groups, a dev-time count on this public data


def load() -> pl.LazyFrame:
    """The LFS sample as a LazyFrame; the Parquet file is cached in data/."""
    return pl.scan_parquet(practitioner_datasets.france_lfs())


def message(err: Exception) -> str:
    """The text of an OpenDP error: `MakeMeasurement("...")` without the wrapper and its line break.

    A key set in the message, `{col("AGE"), col("ILOSTAT")}`, is printed with its columns sorted, because the library
    lists them in an order that changes from run to run.
    """
    text = str(err).strip()
    start, end = text.find('("'), text.rfind('")')
    text = text[start + 2 : end] if 0 <= start < end else text
    return re.sub(r"\{(col\(.*?\))\}", lambda m: "{" + ", ".join(sorted(m.group(1).split(", "))) + "}", text)


def summary_lines(summary: pl.DataFrame, label: str) -> list[str]:
    """One printed line per row of a `summarize` table."""
    out: list[str] = []
    for row in summary.iter_rows(named=True):
        accuracy = row.get("accuracy")
        acc = "none" if accuracy is None or math.isnan(accuracy) else f"{accuracy:,.1f}"
        threshold = row.get("threshold")
        extra = "" if threshold is None else f" threshold={threshold}"
        out.append(
            f"{label} {row['column']}: {row['aggregate']}, {row['distribution']}, "
            f"scale={row['scale']:,.2f} 95% accuracy={acc}{extra}"
        )
    return out


# ---------------------------------------------------------------------------------------------------------------
# Stable keys: the threshold and the delta it costs
# ---------------------------------------------------------------------------------------------------------------


def clearing_probability(threshold: int, scale: float, rows: int, groups: int) -> tuple[float, float]:
    """How likely a group of `rows` rows is to clear the threshold, and how likely any of `groups` such groups is.

    OpenDP keeps a group when its count plus discrete Laplace noise of the given scale exceeds the threshold. The
    tail of the discrete Laplace distribution is exp(-k / scale) / (1 + exp(-1 / scale)) at and above k.
    """
    single = math.exp(-(threshold + 1 - rows) / scale) / (1 + math.exp(-1 / scale))
    return single, 1 - (1 - single) ** groups


def public_keys() -> pl.LazyFrame:
    """Every survey year crossed with every ILOSTAT code of the codebook: a key list fixed without the data."""
    return pl.LazyFrame({"YEAR": [y for y in YEARS for _ in STATUSES], "ILOSTAT": STATUSES * len(YEARS)})


def hours() -> Any:
    """`HWUSUAL` as an integer, as a Polars expression.

    Typed Any because OpenDP adds the `.dp` namespace (`.dp.mean`, `.dp.sum`) to Polars expressions at run time,
    which a type checker cannot see.
    """
    return pl.col("HWUSUAL").cast(pl.Int64)


def true_counts(frame: pl.LazyFrame, by: Sequence[str]) -> pl.DataFrame:
    """The true count per group, which only this public data allows."""
    return frame.group_by(by).agg(pl.len().alias("true")).collect()


def extreme(series: pl.Series, largest: bool = False) -> int:
    """The smallest (or largest) value of an integer series, as an int."""
    value = series.max() if largest else series.min()
    assert isinstance(value, int), value
    return value


def largest_error(released: pl.DataFrame, truth: pl.DataFrame, by: Sequence[str], column: str = "len") -> int:
    """The largest absolute difference between a released count and its true count; a key without rows counts 0."""
    compared = released.join(truth, on=list(by), how="left").fill_null(0)
    return extreme((compared[column] - compared["true"]).abs(), largest=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seed", type=int, default=0, help="accepted for uniformity; nothing here is sampled")
    parser.parse_args()

    frame = load()
    unit = dp.unit_of(contributions=CONTRIBUTIONS)

    # Section 1: why grouping keys are dangerous. The true table, and what the library says without a delta.
    rows = frame.select(pl.len()).collect().item()
    people = frame.select(pl.col("PIDENT").n_unique()).collect().item()
    print(f"rows: {rows:,}  people: {people:,}")
    truth = true_counts(frame, ["AGE", "ILOSTAT"]).sort("true")
    print(f"groups by AGE and ILOSTAT: {truth.height}")
    smallest = "; ".join(
        f"AGE={'null' if r['AGE'] is None else int(r['AGE'])} ILOSTAT={r['ILOSTAT']} rows={r['true']:,}"
        for r in truth.head(3).iter_rows(named=True)
    )
    print(f"smallest groups: {smallest}")
    top = truth.row(-1, named=True)
    print(f"largest group: AGE={int(top['AGE'])} ILOSTAT={top['ILOSTAT']} rows={top['true']:,}")
    pure = dp.Context.compositor(
        data=frame, privacy_unit=unit, privacy_loss=dp.loss_of(epsilon=EPSILON), split_evenly_over=1
    )
    try:
        pure.query().group_by("AGE", "ILOSTAT").agg(dp.len(signed=True)).summarize()
        refusal = "released"
    except dp.OpenDPException as err:
        refusal = message(err)
    print(f"without delta: {refusal}")

    # Section 2: stable keys. The threshold costs delta, so the budget has one.
    # excerpt: group-stable
    context = dp.Context.compositor(
        data=frame,
        privacy_unit=dp.unit_of(contributions=CONTRIBUTIONS),  # a person answers in at most 36 quarters
        privacy_loss=dp.loss_of(epsilon=EPSILON, delta=DELTA),  # the delta pays for the threshold
        split_evenly_over=1,
    )
    stable = context.query().group_by("AGE", "ILOSTAT").agg(dp.len(signed=True))
    stable_summary = stable.summarize(alpha=0.05)
    stable_hat = stable.release().collect()
    # end excerpt
    print(f"stable keys: privacy loss {dp.loss_of(epsilon=EPSILON, delta=DELTA)!r}")
    for line in summary_lines(stable_summary, "stable keys"):
        print(line)
    row = stable_summary.row(0, named=True)
    scale, accuracy, threshold = float(row["scale"]), float(row["accuracy"]), int(row["threshold"])
    single, joint = clearing_probability(threshold, scale, CONTRIBUTIONS, CONTRIBUTIONS)
    print(
        f"one person's {CONTRIBUTIONS} rows clear {threshold} with probability {single:.2e}; "
        f"over {CONTRIBUTIONS} groups {joint:.2e} <= delta {DELTA:.0e}"
    )
    _, joint_below = clearing_probability(threshold - 1, scale, CONTRIBUTIONS, CONTRIBUTIONS)
    print(f"at threshold {threshold - 1}: over {CONTRIBUTIONS} groups {joint_below:.2e} > delta")
    above = truth.filter(pl.col("true") > threshold).height
    print(f"groups with true count above the threshold: {above} of {truth.height}")
    for label, loss in (
        ("delta=1e-05", dp.loss_of(epsilon=EPSILON, delta=1e-5)),
        ("epsilon=1", dp.loss_of(epsilon=1.0, delta=DELTA)),
    ):
        variant = dp.Context.compositor(data=frame, privacy_unit=unit, privacy_loss=loss, split_evenly_over=1)
        summary = variant.query().group_by("AGE", "ILOSTAT").agg(dp.len(signed=True)).summarize()
        print(f"threshold with {label}: scale={summary['scale'].item():,.2f} threshold={summary['threshold'].item()}")

    # Section 3: explicit keys. A public key list; every key is released, with or without rows.
    keys = public_keys().collect()
    year_truth = true_counts(frame, ["YEAR", "ILOSTAT"])
    with_rows = keys.join(year_truth, on=["YEAR", "ILOSTAT"], how="inner").height
    print(f"explicit keys: {keys.height} keys, {with_rows} with rows in the data, {keys.height - with_rows} without")
    context = dp.Context.compositor(
        data=frame, privacy_unit=unit, privacy_loss=dp.loss_of(epsilon=EPSILON), split_evenly_over=1
    )
    # excerpt: group-explicit
    explicit = (
        context.query()
        .group_by("YEAR", "ILOSTAT")
        .agg(dp.len(signed=True))
        .with_keys(public_keys())  # the years crossed with the codebook's codes, fixed without the data
    )
    explicit_summary = explicit.summarize(alpha=0.05)
    explicit_hat = explicit.release().collect()
    # end excerpt
    for line in summary_lines(explicit_summary, "explicit keys"):
        print(line)
    # The same query written as a join, and a join the library refuses; a context of its own, since the one above
    # had one share and the release spent it.
    context = dp.Context.compositor(
        data=frame, privacy_unit=unit, privacy_loss=dp.loss_of(epsilon=EPSILON), split_evenly_over=1
    )
    joined = (
        context.query()
        .group_by("YEAR", "ILOSTAT")
        .agg(dp.len(signed=True))
        .join(public_keys(), how="right", on=["YEAR", "ILOSTAT"])
    )
    for line in summary_lines(joined.summarize(alpha=0.05), "right join"):
        print(line)
    try:
        context.query().group_by("YEAR", "ILOSTAT").agg(dp.len(signed=True)).join(
            public_keys(), how="inner", on=["YEAR", "ILOSTAT"]
        ).summarize()
        refusal = "released"
    except dp.OpenDPException as err:
        refusal = message(err)
    print(f"inner join: {refusal}")

    # Section 4: invariant keys. The margin says the quarters are public, so no threshold and no delta.
    # excerpt: group-invariant-keys
    context = dp.Context.compositor(
        data=frame,
        privacy_unit=dp.unit_of(contributions=CONTRIBUTIONS),
        privacy_loss=dp.loss_of(epsilon=EPSILON),
        split_evenly_over=1,
        margins=[dp.polars.Margin(by=["YEAR", "QUARTER"], invariant="keys")],  # the quarters are public
    )
    quarterly = context.query().group_by("YEAR", "QUARTER").agg(dp.len(signed=True))
    quarterly_summary = quarterly.summarize(alpha=0.05)
    quarterly_hat = quarterly.release().collect()
    # end excerpt
    for line in summary_lines(quarterly_summary, "invariant keys"):
        print(line)
    quarter_truth = true_counts(frame, ["YEAR", "QUARTER"]).sort("true")
    low, high = quarter_truth.row(0, named=True), quarter_truth.row(-1, named=True)
    print(
        f"true quarterly counts: {quarter_truth.height} quarters, smallest {low['true']:,} ({low['YEAR']} Q{low['QUARTER']}), "
        f"largest {high['true']:,} ({high['YEAR']} Q{high['QUARTER']})"
    )
    print(
        f"error bar as a share of the smallest quarter: {accuracy / low['true']:.2%}, of the largest: {accuracy / high['true']:.2%}"
    )

    # Section 5: invariant lengths. The group sizes are public; the filter must happen before the context.
    working = frame.filter(pl.col("HWUSUAL") < 99)  # outside the context: a filter inside would void the margin
    hours_truth = (
        working.group_by("SEX").agg(pl.len().alias("true"), hours().mean().alias("mean")).sort("SEX").collect()
    )
    print(f"rows with HWUSUAL < 99: {hours_truth['true'].sum():,}")
    # excerpt: group-invariant-lengths
    context = dp.Context.compositor(
        data=working,
        privacy_unit=dp.unit_of(contributions=CONTRIBUTIONS),
        privacy_loss=dp.loss_of(epsilon=MEAN_EPSILON),
        split_evenly_over=1,
        margins=[
            dp.polars.Margin(
                by=["SEX"],
                invariant="lengths",  # the number of responses per sex is public, keys included
                max_length=MAX_LENGTH,  # needed by any sum: a public bound on the rows in a group
                max_groups=3,  # male, female or null
            )
        ],
    )
    mean_hours = context.query().group_by("SEX").agg(hours().dp.mean(HOURS_BOUNDS))
    mean_summary = mean_hours.summarize(alpha=0.05)
    mean_hat = mean_hours.release().collect()
    # end excerpt
    for line in summary_lines(mean_summary, "invariant lengths"):
        print(line)
    sum_accuracy = float(mean_summary.filter(pl.col("aggregate") == "Sum")["accuracy"].item())
    print("true lengths: " + "  ".join(f"SEX={r['SEX']} {r['true']:,}" for r in hours_truth.iter_rows(named=True)))
    print("true mean hours: " + "  ".join(f"SEX={r['SEX']} {r['mean']:.3f}" for r in hours_truth.iter_rows(named=True)))
    mean_accuracy = {int(r["SEX"]): sum_accuracy / r["true"] for r in hours_truth.iter_rows(named=True)}
    print("mean 95% accuracy: " + "  ".join(f"SEX={sex} {acc:.4f}" for sex, acc in mean_accuracy.items()))
    context = dp.Context.compositor(
        data=frame,
        privacy_unit=unit,
        privacy_loss=dp.loss_of(epsilon=MEAN_EPSILON),
        split_evenly_over=1,
        margins=[dp.polars.Margin(by=["SEX"], invariant="lengths", max_length=MAX_LENGTH, max_groups=3)],
    )
    try:
        context.query().filter(pl.col("HWUSUAL") < 99).group_by("SEX").agg(hours().dp.mean(HOURS_BOUNDS)).summarize()
        refusal = "released"
    except dp.OpenDPException as err:
        refusal = message(err)
    print(f"filter inside the query: {refusal}")

    # Section 6: margins as public descriptors. max_groups lowers the threshold; max_length is needed by a sum.
    # excerpt: group-max-groups
    context = dp.Context.compositor(
        data=frame,
        privacy_unit=dp.unit_of(contributions=CONTRIBUTIONS),
        privacy_loss=dp.loss_of(epsilon=EPSILON, delta=DELTA),
        split_evenly_over=1,
        margins=[dp.polars.Margin(by=["AGE", "ILOSTAT"], max_groups=GROUPS_IN_TRUTH)],  # at most 16 groups
    )
    bounded_summary = context.query().group_by("AGE", "ILOSTAT").agg(dp.len(signed=True)).summarize()
    # end excerpt
    print(f"max_groups={GROUPS_IN_TRUTH}: threshold={bounded_summary['threshold'].item()}")
    context = dp.Context.compositor(
        data=working,
        privacy_unit=unit,
        privacy_loss=dp.loss_of(epsilon=MEAN_EPSILON),
        split_evenly_over=1,
        margins=[dp.polars.Margin(by=["SEX"], invariant="keys")],
    )
    try:
        context.query().group_by("SEX").agg(hours().dp.sum(HOURS_BOUNDS)).summarize()
        refusal = "released"
    except dp.OpenDPException as err:
        refusal = message(err)
    print(f"sum per SEX without max_length: {refusal}")

    # Section 7: the noise per group against the ungrouped count. Same scale, different relative error.
    context = dp.Context.compositor(
        data=frame, privacy_unit=unit, privacy_loss=dp.loss_of(epsilon=EPSILON), split_evenly_over=1
    )
    for line in summary_lines(context.query().select(dp.len(signed=True)).summarize(alpha=0.05), "ungrouped count"):
        print(line)
    released_truth = truth.filter(pl.col("true") > threshold)
    smallest_released = extreme(released_truth["true"])
    largest_dropped = extreme(truth.filter(pl.col("true") <= threshold)["true"], largest=True)
    print(
        f"error bar as a share of the true count: all rows {accuracy / rows:.3%}, "
        f"largest group {accuracy / top['true']:.3%}, smallest group above the threshold {accuracy / smallest_released:.1%}"
    )

    # Section 8: the dev-time check and the release of one run.
    print(f"smallest true count above the threshold: {smallest_released:,}  largest below it: {largest_dropped:,}")
    print(f"test bound for a table: 4 x accuracy = {4 * accuracy:,.1f}")
    compared = stable_hat.join(truth, on=["AGE", "ILOSTAT"], how="left")
    print("released (one run):")
    print(
        f"  stable groups released: {stable_hat.height}  smallest released count: {extreme(stable_hat['len']):,}  "
        f"smallest true count among them: {extreme(compared['true']):,}"
    )
    for age, status in ((47.0, 1), (65.0, 2)):
        r = compared.filter((pl.col("AGE") == age) & (pl.col("ILOSTAT") == status)).row(0, named=True)
        print(f"  stable AGE={int(age)} ILOSTAT={status}: {r['len']:,}  (true {r['true']:,})")
    print(f"  stable largest |released - true|: {largest_error(stable_hat, truth, ['AGE', 'ILOSTAT'])}")
    r = explicit_hat.filter((pl.col("YEAR") == 2005) & (pl.col("ILOSTAT") == 4)).row(0, named=True)
    print(f"  explicit YEAR=2005 ILOSTAT=4: {r['len']}  (true 0)")
    r = (
        explicit_hat.join(year_truth, on=["YEAR", "ILOSTAT"])
        .filter((pl.col("YEAR") == 2013) & (pl.col("ILOSTAT") == 1))
        .row(0, named=True)
    )
    print(f"  explicit YEAR=2013 ILOSTAT=1: {r['len']:,}  (true {r['true']:,})")
    print(
        f"  explicit keys released: {explicit_hat.height}  largest |released - true|: {largest_error(explicit_hat, year_truth, ['YEAR', 'ILOSTAT'])}"
    )
    quarters = quarterly_hat.join(quarter_truth, on=["YEAR", "QUARTER"])
    inside = quarters.filter((pl.col("len") - pl.col("true")).abs() <= accuracy).height
    r = quarters.filter((pl.col("YEAR") == 2005) & (pl.col("QUARTER") == 1)).row(0, named=True)
    print(f"  quarter 2005 Q1: {r['len']:,}  (true {r['true']:,})")
    print(
        f"  quarters released: {quarterly_hat.height}  within their 95% accuracy: {inside} of {quarters.height}  "
        f"largest |released - true|: {largest_error(quarterly_hat, quarter_truth, ['YEAR', 'QUARTER'])}"
    )
    means = mean_hat.join(hours_truth, on="SEX").sort("SEX")
    print(
        "  mean hours: "
        + "  ".join(f"SEX={r['SEX']} {r['HWUSUAL']:.3f} (true {r['mean']:.3f})" for r in means.iter_rows(named=True))
    )


if __name__ == "__main__":
    main()
