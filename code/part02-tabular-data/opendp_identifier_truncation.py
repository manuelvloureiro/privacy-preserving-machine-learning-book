"""Identifier truncation and bounds with OpenDP: the chapter "Identifier truncation and bounds" of the OpenDP book (Part 2).

    uv run python code/part02-tabular-data/opendp_identifier_truncation.py

Follows the documentation's pages "Identifier Truncation" (getting started) and "Identifier Truncation and Bounds"
(Polars user guide): a privacy unit with an identifier instead of a count of contributions, a mean that truncates
each identifier's rows with `truncate_per_group` (and its `keep` options), a table per quarter that also bounds the
groups an identifier touches with `truncate_num_groups`, the mean of one row per identifier, and bounds declared in
the privacy unit with `dp.polars.Bound`. It also tests, end to end, what OpenDP 0.16.0 does with
`truncate_num_groups`: `keep="sample"` passes `summarize` and fails at `collect` after the budget is spent, and
`keep="first"` keeps one group fewer than asked. It prints every number the page quotes, on labelled lines, in the
order of the page's sections.

Dataset: the France Labour Force Survey sample of the OpenDP tutorials (3,811,932 quarterly responses, 1,682,306
identifiers), loaded by code/practitioner_datasets.py, which caches it in data/ as Parquet; the documentation reads
the same file through dp.examples.get_france_lfs_path(). The data are public, so the true values after each
truncation are computed and compared with the releases.

What is random and what is not. OpenDP draws its noise from a cryptographic generator that cannot be seeded, so the
released values change on every run; they are printed under "released (one run)" and the test checks them against
the true values within three times their 95% accuracy (four times for the 36 quarters, since there are many of
them). `truncate_per_group` with the default `keep="sample"` also keeps a random row, which the dev-time truth
approximates with the first row. The noise scales, the accuracies, the thresholds, the error messages, the query
plans and the facts about the data are deterministic; `--seed` seeds only the shuffle of the dev-time truncation.

Library versions: opendp[polars] 0.16.0 with polars 1.36.1.
"""

import argparse
import re
import sys
from pathlib import Path
from typing import Any

import opendp.prelude as dp
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import practitioner_datasets

dp.enable_features("contrib")

EPSILON = 1.0
DELTA = 1e-8
QUERIES = 4
MAX_LENGTH = 150_000 * 36  # public: the documentation's bound on the number of rows
HOURS = (0, 80)  # public: hours usually worked per week, once 99 ("not applicable") is filtered out
ROWS_PER_IDENTIFIER = 10  # the documentation's truncation: at most 10 rows per identifier
GROUPS_PER_IDENTIFIER = 10  # and at most 10 quarters per identifier
QUARTERLY = [pl.col("QUARTER"), pl.col("YEAR")]  # the grouping columns of the table per quarter


def hours() -> Any:
    """`HWUSUAL` cast to an integer, as the documentation's queries aggregate it.

    Typed Any because OpenDP adds the `.dp` namespace (`.dp.mean`, `.dp.sum`) to Polars expressions at run time,
    which a type checker cannot see.
    """
    return pl.col("HWUSUAL").cast(int)


# ---------------------------------------------------------------------------------------------------------------
# Contexts
# ---------------------------------------------------------------------------------------------------------------


# excerpt: trunc-context
def make_context(frame: pl.LazyFrame, privacy_unit: tuple[dp.Metric, Any]) -> dp.Context:
    """The documentation's context: approximate differential privacy, split evenly over four queries."""
    return dp.Context.compositor(
        data=frame,
        privacy_unit=privacy_unit,  # dp.unit_of(contributions=1, identifier="PIDENT") below
        privacy_loss=dp.loss_of(epsilon=EPSILON, delta=DELTA),
        split_evenly_over=QUERIES,
        margins=[dp.polars.Margin(max_length=MAX_LENGTH)],
    )


# end excerpt


def identifier_unit() -> tuple[dp.Metric, Any]:
    """One identifier per person: neighbours add or remove every row that shares one value of `PIDENT`."""
    return dp.unit_of(contributions=1, identifier="PIDENT")


# ---------------------------------------------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------------------------------------------


def summary_lines(summary: pl.DataFrame, label: str) -> list[str]:
    """One line per statistic of a `summarize(alpha=0.05)` table: distribution, scale, accuracy and threshold."""
    out: list[str] = []
    for row in summary.iter_rows(named=True):
        threshold = row.get("threshold")
        extra = f" threshold={threshold:,}" if threshold is not None else ""
        out.append(
            f"{label}: {row['column']} {row['aggregate']}, {row['distribution']}, "
            f"scale={row['scale']:,.1f} 95% accuracy={row['accuracy']:,.1f}{extra}"
        )
    return out


def scales(summary: pl.DataFrame) -> str:
    return ", ".join(f"{s:,.1f}" for s in summary["scale"].to_list())


def accuracy_of(summary: pl.DataFrame, aggregate: str) -> float:
    return float(summary.filter(pl.col("aggregate") == aggregate)["accuracy"].item())


def reason(err: Exception) -> str:
    """The message inside OpenDP's wrapper, `Variant("message")`, with its escaped quotes undone."""
    lines = re.split(r"\n|\\n", str(err))  # a Polars error inside the wrapper has literal "\n" in it
    line = next(line.strip() for line in lines if line.strip())
    line = re.sub(r'^(\w+\("?)+', "", line)  # FailedMap(" or FailedFunction("Duplicate(ErrString("
    return re.sub(r'("\))+$', "", line).replace('\\"', '"')


def plan_line(query: Any, word: str) -> str:
    """The first line of the query's Polars plan that mentions `word`: the filter a truncation builds."""
    return next(line.strip() for line in query.explain().splitlines() if word in line)


def mean_bar(summary: pl.DataFrame, size: int, mean: float) -> float:
    """An error bar for a released mean, from the bars of the sum and the length it divides.

    If the noisy sum is within a of the true sum and the noisy length within c of the true length n, then the
    ratio is within (a + |mean| c) / (n - c) of the true mean. Both bars hold with probability at least 0.9.
    """
    a, c = accuracy_of(summary, "Sum"), accuracy_of(summary, "Length")
    return (a + abs(mean) * c) / (size - c)


def quarter(row: dict[str, Any]) -> str:
    return f"{row['YEAR']}Q{row['QUARTER']}"


# ---------------------------------------------------------------------------------------------------------------
# The true values, for the dev-time check (this public dataset only)
# ---------------------------------------------------------------------------------------------------------------


def identifier_facts(frame: pl.LazyFrame) -> list[str]:
    """What one identifier owns in this file: rows, quarters, and rows within one quarter."""
    per_id = frame.group_by("PIDENT").agg(pl.len().alias("rows"), pl.struct(QUARTERLY).n_unique().alias("quarters"))
    per_id_quarter = frame.group_by("PIDENT", "YEAR", "QUARTER").agg(pl.len().alias("rows"))
    ids, idq = per_id.collect(), per_id_quarter.collect()
    rows = ids["rows"]
    p50, p90, p99 = (int(rows.quantile(q, interpolation="lower") or 0) for q in (0.5, 0.9, 0.99))
    return [
        f"identifiers: {ids.height:,}  rows: {rows.sum():,}",
        f"rows per identifier: p50={p50} p90={p90} p99={p99} max={rows.max()}",
        f"identifiers with more than {ROWS_PER_IDENTIFIER} rows: {(rows > ROWS_PER_IDENTIFIER).sum():,}",
        f"quarters per identifier: max {ids['quarters'].max()}  rows per identifier and quarter: max {idq['rows'].max()}",
    ]


def working(frame: pl.LazyFrame) -> pl.DataFrame:
    """The rows the queries keep before truncating: `HWUSUAL` is not 99 (and not null)."""
    return frame.filter(pl.col("HWUSUAL") != 99).collect()


def truncated_rows(rows: pl.DataFrame, k: int, seed: int) -> pl.DataFrame:
    """At most k random rows per identifier: `truncate_per_group(k)` with a seed, so that the truth is fixed."""
    return rows.filter(pl.int_range(pl.len()).shuffle(seed=seed).over("PIDENT") < k)


def one_row_per_quarter(rows: pl.DataFrame) -> pl.DataFrame:
    """The first row of each identifier in each quarter: `truncate_per_group(1, by=QUARTERLY, keep="first")`."""
    return rows.filter(pl.int_range(pl.len()).over("PIDENT", *QUARTERLY) < 1)


def groups_kept(rows: pl.DataFrame, k: int) -> pl.DataFrame:
    """The documentation's equivalent of `truncate_num_groups(k, by=QUARTERLY, keep="first")`, run on the data."""
    return rows.filter(pl.struct(QUARTERLY).rank("dense").over("PIDENT") < k)


def per_quarter(rows: pl.DataFrame) -> pl.DataFrame:
    """Rows and hours per quarter, the true values of the grouped queries."""
    low, high = HOURS
    agg = rows.group_by(QUARTERLY).agg(pl.len().alias("len"), hours().clip(low, high).sum().alias("HWUSUAL"))
    return agg.sort(["YEAR", "QUARTER"])


def grouped_query(context: dp.Context, keep: Any, k: int = GROUPS_PER_IDENTIFIER) -> Any:
    """The documentation's table per quarter: one row per identifier and quarter, at most k quarters each."""
    return (
        context.query()
        .filter(pl.col("HWUSUAL") != 99)
        .truncate_per_group(1, by=QUARTERLY)
        .truncate_num_groups(k, by=QUARTERLY, keep=keep)
        .group_by(QUARTERLY)
        .agg(dp.len(signed=True), hours().dp.sum(HOURS))
    )


def within(released: pl.DataFrame, truth: pl.DataFrame, column: str, bar: float) -> int:
    """How many quarters' released values lie within `bar` of the truth."""
    joined = released.join(truth, on=["YEAR", "QUARTER"], suffix="_true")
    return joined.filter((pl.col(column) - pl.col(f"{column}_true")).abs() <= bar).height


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seed", type=int, default=0, help="seed of the dev-time truncation (OpenDP's noise has none)")
    args = parser.parse_args()
    seed: int = args.seed

    frame = pl.scan_parquet(practitioner_datasets.france_lfs())

    # Section 1: a privacy unit with an identifier. The bound on a person's rows leaves the unit; the query must
    # supply it, and a query that does not is refused.
    print(f"opendp version: {dp.__version__}")
    unit = identifier_unit()
    print(f"privacy unit: {unit!r}")
    for line in identifier_facts(frame):
        print(line)
    context = make_context(frame, unit)
    epsilon_q, delta_q = (context.d_mids or [])[0]
    print(f"budget per query: epsilon={epsilon_q:.4g} delta={delta_q:.3g}")
    # excerpt: trunc-untruncated
    try:
        context.query().filter(pl.col("HWUSUAL") != 99).select(hours().dp.mean(HOURS)).summarize()
        untruncated = "accepted"
    except dp.OpenDPException as err:
        untruncated = reason(err)
    # end excerpt
    print(f"mean without truncation: {untruncated}")

    # Section 2: truncating each identifier's rows. The query's share is split between the sum and the length of
    # the mean, so each gets epsilon / 8, and 10 rows of at most 80 hours give the sum a scale of 6,400.
    # excerpt: trunc-per-group
    mean_query = (
        context.query()
        .filter(pl.col("HWUSUAL") != 99)  # 99 means "not applicable"
        .truncate_per_group(ROWS_PER_IDENTIFIER)  # at most 10 rows per identifier, chosen at random
        .select(hours().dp.mean(HOURS))
    )
    mean_summary = mean_query.summarize(alpha=0.05)
    dp_mean = mean_query.release().collect().item()
    # end excerpt
    for line in summary_lines(mean_summary, f"mean, {ROWS_PER_IDENTIFIER} rows per identifier"):
        print(line)
    print(f"truncate_per_group({ROWS_PER_IDENTIFIER}) plan: {plan_line(mean_query, 'int_range')}")
    keeps: dict[str, Any] = {"first": "first", "last": "last", "SortBy(ILOSTAT)": dp.polars.SortBy(pl.col("ILOSTAT"))}
    for name, keep in keeps.items():
        kept_query = (
            context.query()
            .filter(pl.col("HWUSUAL") != 99)
            .truncate_per_group(ROWS_PER_IDENTIFIER, keep=keep)
            .select(hours().dp.mean(HOURS))
        )
        print(f"keep={name}: scales {scales(kept_query.summarize())}")
    rows = working(frame)
    per_id = rows.group_by("PIDENT").agg(pl.len().cast(pl.Int64).alias("rows"))["rows"]
    dropped = (per_id - ROWS_PER_IDENTIFIER).clip(lower_bound=0).sum()
    print(f"rows with HWUSUAL not 99: {rows.height:,} by {per_id.len():,} identifiers")
    print(f"ILOSTAT values among them: {sorted(rows['ILOSTAT'].unique().to_list())}")
    print(
        f"identifiers with more than {ROWS_PER_IDENTIFIER} such rows: {(per_id > ROWS_PER_IDENTIFIER).sum()}  rows dropped: {dropped}"
    )
    truncated = truncated_rows(rows, ROWS_PER_IDENTIFIER, seed)
    true_mean = float(truncated.select(pl.col("HWUSUAL").mean()).item())
    mean_error = mean_bar(mean_summary, truncated.height, true_mean)
    print(f"mean error bar at the true size: {mean_error:.4f}")
    plain = make_context(frame, dp.unit_of(contributions=36))
    plain_query = plain.query().filter(pl.col("HWUSUAL") != 99).select(hours().dp.mean(HOURS))
    print(f"unit of 36 contributions, no identifier: scales {scales(plain_query.summarize())}")

    # Section 3: truncating the groups an identifier touches. Without it the grouped query is refused; with it,
    # keep="sample" fails at collect in 0.16.0 and keep="first" keeps k - 1 groups.
    try:
        (
            context.query()
            .filter(pl.col("HWUSUAL") != 99)
            .truncate_per_group(1, by=QUARTERLY)
            .group_by(QUARTERLY)
            .agg(dp.len(signed=True), hours().dp.sum(HOURS))
            .summarize()
        )
        ungrouped = "accepted"
    except dp.OpenDPException as err:
        ungrouped = reason(err).split(" To bound")[0]  # the suggested `by` order changes between runs
    print(f"per quarter without truncate_num_groups: {ungrouped}")
    # excerpt: trunc-num-groups
    quarter_query = (
        context.query()
        .filter(pl.col("HWUSUAL") != 99)
        .truncate_per_group(1, by=QUARTERLY)  # one row per identifier and quarter
        .truncate_num_groups(GROUPS_PER_IDENTIFIER, by=QUARTERLY, keep="first")  # at most 10 quarters each
        .group_by(QUARTERLY)
        .agg(dp.len(signed=True), hours().dp.sum(HOURS))
    )
    quarter_summary = quarter_query.summarize(alpha=0.05)
    quarters_hat = quarter_query.release().collect()
    # end excerpt
    for line in summary_lines(quarter_summary, f"per quarter, {GROUPS_PER_IDENTIFIER} quarters per identifier"):
        print(line)
    print(f"truncate_num_groups({GROUPS_PER_IDENTIFIER}, keep=first) plan: {plan_line(quarter_query, 'rank')}")
    one_each = one_row_per_quarter(rows)
    quarters_per_id = one_each.group_by("PIDENT").agg(pl.len().alias("quarters"))["quarters"]
    print(
        f"identifier-quarters after the filter: {one_each.height:,}  identifiers with 4 quarters: {(quarters_per_id == 4).sum():,}"
    )
    for k in (4, 5):
        kept = groups_kept(one_each, k)
        most = kept.group_by("PIDENT").agg(pl.len().alias("quarters"))["quarters"].max()
        summary = grouped_query(context, "first", k).summarize(alpha=0.05)
        threshold = int(summary["threshold"].drop_nulls().item())
        print(
            f"truncate_num_groups({k}, keep=first): keeps at most {most} quarters, {kept.height:,} identifier-quarters;"
            f" scale {summary['scale'][0]:,.1f} threshold {threshold:,}"
        )
    try:
        (
            context.query()
            .truncate_per_group(1, by=QUARTERLY)
            .filter(pl.col("HWUSUAL") != 99)
            .truncate_num_groups(GROUPS_PER_IDENTIFIER, by=QUARTERLY, keep="first")
            .group_by(QUARTERLY)
            .agg(dp.len(signed=True))
            .summarize()
        )
        separated = "accepted"
    except dp.OpenDPException as err:
        separated = reason(err)
    print(f"a filter between the two truncations: {separated}")

    # Section 4: one row per identifier. Grouping by the identifier first bounds every person to one row, so the
    # scales fall tenfold without any truncation parameter.
    # excerpt: trunc-per-person
    person_query = (
        context.query()
        .filter(pl.col("HWUSUAL") != 99)
        .group_by("PIDENT")  # one row per identifier: the truncation
        .agg(pl.col("HWUSUAL").mean())  # any expression may summarize the person's rows
        .select(hours().dp.mean(HOURS))
    )
    person_summary = person_query.summarize(alpha=0.05)
    dp_person_mean = person_query.release().collect().item()
    # end excerpt
    for line in summary_lines(person_summary, "mean of one row per identifier"):
        print(line)
    person_means = rows.group_by("PIDENT").agg(pl.col("HWUSUAL").mean())["HWUSUAL"]
    true_person_mean = float(person_means.cast(pl.Int64).to_frame().select(pl.col("HWUSUAL").mean()).item())
    print(
        f"true mean of one row per identifier: {true_person_mean:.4f}"
        f"  (before the cast to int: {person_means.mean():.4f}; all rows: {rows['HWUSUAL'].mean():.4f})"
    )
    person_error = mean_bar(person_summary, person_means.len(), true_person_mean)
    print(f"per-identifier mean error bar at the true size: {person_error:.4f}")

    # The fourth share: keep="sample" passes summarize, release() spends the share, collect fails.
    # excerpt: trunc-sample
    sample_query = grouped_query(context, keep="sample")  # the same table, with the default keep
    sample_summary = sample_query.summarize(alpha=0.05)
    released = sample_query.release()  # spends the query's share of the budget
    try:
        released.collect()
        sample_outcome = "collected"
    except dp.OpenDPException as err:
        sample_outcome = reason(err)
    # end excerpt
    print(f"keep=sample: summarize scales {scales(sample_summary)}; collect: {sample_outcome}")
    print(f"shares spent after the failed collect: {len(context.current_privacy_loss())} of {QUERIES}")

    # Section 5: bounds in the privacy unit. The documentation's two bounds, then num_groups as a promise.
    # excerpt: trunc-bounds
    two_sources = dp.unit_of(
        contributions=[
            dp.polars.Bound(per_group=2),  # a person may appear under two identifiers
            dp.polars.Bound(by=QUARTERLY, per_group=1),  # but under one identifier in any quarter
        ],
        identifier="PIDENT",
    )
    bounded = make_context(frame, two_sources)
    bounded_query = grouped_query(bounded, keep="first", k=5)  # each identifier may affect up to 5 quarters
    bounded_summary = bounded_query.summarize(alpha=0.05)
    bounded_hat = bounded_query.release().collect()
    # end excerpt
    for line in summary_lines(bounded_summary, "two identifiers, one per quarter, 5 quarters"):
        print(line)
    one_source = dp.unit_of(contributions=[dp.polars.Bound(per_group=1)], identifier="PIDENT")
    single_summary = grouped_query(make_context(frame, one_source), keep="first", k=5).summarize(alpha=0.05)
    print(f"one identifier as a bound, 5 quarters: scales {scales(single_summary)}")
    kept_word = dp.unit_of(
        contributions=[dp.polars.Bound(per_group=1), dp.polars.Bound(by=QUARTERLY, per_group=1, num_groups=10)],
        identifier="PIDENT",
    )
    kept_word_query = (
        make_context(frame, kept_word)
        .query()
        .filter(pl.col("HWUSUAL") != 99)
        .truncate_per_group(1, by=QUARTERLY)
        .group_by(QUARTERLY)
        .agg(dp.len(signed=True), hours().dp.sum(HOURS))
    )
    for line in summary_lines(kept_word_query.summarize(alpha=0.05), "num_groups=10 promised in the unit"):
        print(line)
    # excerpt: trunc-promise
    promised = dp.unit_of(
        contributions=[
            dp.polars.Bound(per_group=1),
            dp.polars.Bound(by=QUARTERLY, per_group=1, num_groups=3),  # a promise: at most 3 quarters each
        ],
        identifier="PIDENT",
    )
    promise_query = (
        make_context(frame, promised)
        .query()
        .filter(pl.col("HWUSUAL") != 99)
        .truncate_per_group(1, by=QUARTERLY)  # no truncate_num_groups: the unit already bounds the quarters
        .group_by(QUARTERLY)
        .agg(dp.len(signed=True), hours().dp.sum(HOURS))
    )
    promise_summary = promise_query.summarize(alpha=0.05)
    promise_hat = promise_query.release().collect()
    # end excerpt
    for line in summary_lines(promise_summary, "num_groups=3 promised in the unit"):
        print(line)
    print(f"identifiers that break the promise of 3 quarters: {(quarters_per_id > 3).sum():,}")

    # Section 6: the dev-time check against the true values, and the releases of one run.
    truth = per_quarter(one_each)
    print(f"true mean, {ROWS_PER_IDENTIFIER} rows per identifier: {true_mean:.4f}  ({truncated.height:,} rows)")
    print(
        f"true per quarter (first row per identifier and quarter): {truth.height} quarters, rows from {truth['len'].min():,} to {truth['len'].max():,}"
    )
    for row in truth.iter_rows(named=True):
        print(f"true {quarter(row)}: rows {row['len']:,}  hours {row['HWUSUAL']:,}")
    len_bar, sum_bar = accuracy_of(quarter_summary, "Frame Length"), accuracy_of(quarter_summary, "Sum")
    promise_bar = accuracy_of(promise_summary, "Frame Length")

    print("released (one run):")
    print(
        f"  mean, {ROWS_PER_IDENTIFIER} rows per identifier: {dp_mean:.4f}  (true {true_mean:.4f}, within {mean_error:.4f} with probability at least 0.9)"
    )
    print(
        f"  per quarter: {quarters_hat.height} quarters released; rows within their 95% accuracy: {within(quarters_hat, truth, 'len', len_bar)} of {truth.height}; hours: {within(quarters_hat, truth, 'HWUSUAL', sum_bar)} of {truth.height}"
    )
    for row in quarters_hat.sort(["YEAR", "QUARTER"]).iter_rows(named=True):
        print(f"  {quarter(row)}: rows {row['len']:,}  hours {row['HWUSUAL']:,}")
    print(
        f"  mean of one row per identifier: {dp_person_mean:.4f}  (true {true_person_mean:.4f}, within {person_error:.4f} with probability at least 0.9)"
    )
    print(
        f"  two bounds, 5 quarters: {bounded_hat.height} quarters released; rows within four times their 95% accuracy: {within(bounded_hat, truth, 'len', 4 * len_bar)} of {truth.height}"
    )
    print(
        f"  num_groups=3 promised: {promise_hat.height} quarters released; rows within four times their 95% accuracy: {within(promise_hat, truth, 'len', 4 * promise_bar)} of {truth.height}"
    )


if __name__ == "__main__":
    main()
