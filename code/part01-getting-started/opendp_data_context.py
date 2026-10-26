"""The data and the context: the chapter "The data and the context" of the OpenDP book (Part 1).

    uv run python code/part01-getting-started/opendp_data_context.py

Follows the documentation's "Working with Tabular Data" introduction (the dataset, loading data, mediating access
with a Context), "Polars vs. OpenDP" and "Context" in the API user guide, and the count of "Essential Statistics":
it loads the France Labour Force Survey sample lazily with Polars, puts it behind a context with a privacy unit
of 36 rows and a budget of epsilon = 1 split over five queries, shows how a LazyFrameQuery differs from a
LazyFrame, releases the count of responses, and previews the margin that the next chapter's sums need. It prints
every number the page quotes, on labelled lines, in the order of the page's sections.

Dataset: the France Labour Force Survey (LFS) sample of the OpenDP tutorials (Eurostat public microdata for
France, 2005 to 2013, with a synthetic person identifier), https://github.com/opendp/dp-test-datasets, loaded by
code/practitioner_datasets.py, which caches it in data/ as Parquet: 3,811,932 quarterly responses from 1,682,306
identifiers. It is public microdata already protected by statistical disclosure control, so the true count is
printed and compared with the release.

What is deterministic and what is not. OpenDP draws its noise from a cryptographic generator that cannot be
seeded, so the released count changes on every run; it is printed under "released (one run)" and the test checks
it against the true count within three times its 95% accuracy. Everything else is deterministic and quotable: the
sizes, the unit, the budget and its splits, the mock query's schema and refusals, the scale and accuracy of the
count, the margin and the true count. The --seed option is accepted for uniformity with the other scripts; nothing
here draws from a seeded generator.

Library versions: opendp[polars] 0.16.0 with polars 1.36.1.
"""

import argparse
import sys
import warnings
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import opendp.prelude as dp
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import practitioner_datasets

dp.enable_features("contrib")

EPSILON = 1.0
CONTRIBUTIONS = 36  # one response per quarter over the survey's 9 years: 9 x 4 rows at most per person
QUERIES = 5
WEIGHTS = [2, 1, 1, 1]  # the first query matters twice as much as each of the other three
QUARTER_RESPONSES = 150_000  # public: 50,000 households per quarter, under three people each


# ---------------------------------------------------------------------------------------------------------------
# The data, read at development time: this sample is public, so looking at it leaks nothing
# ---------------------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Sizes:
    rows: int
    people: int
    first_year: int
    last_year: int
    quarters: int
    max_responses: int
    max_quarters: int


def sizes(frame: pl.LazyFrame) -> Sizes:
    """The sizes the page quotes: rows, people, years, and how often one identifier appears."""
    totals = frame.select(
        pl.len().alias("rows"),
        pl.col("PIDENT").n_unique().alias("people"),
        pl.col("YEAR").min().alias("first_year"),
        pl.col("YEAR").max().alias("last_year"),
        pl.struct("YEAR", "QUARTER").n_unique().alias("quarters"),
    )
    per_person = frame.group_by("PIDENT").agg(
        pl.len().alias("responses"), pl.struct("YEAR", "QUARTER").n_unique().alias("quarters")
    )
    most = per_person.select(pl.col("responses").max(), pl.col("quarters").max())
    t = totals.collect().row(0, named=True)
    max_responses, max_quarters = most.collect().row(0)
    return Sizes(
        rows=int(t["rows"]),
        people=int(t["people"]),
        first_year=int(t["first_year"]),
        last_year=int(t["last_year"]),
        quarters=int(t["quarters"]),
        max_responses=int(max_responses),
        max_quarters=int(max_quarters),
    )


# ---------------------------------------------------------------------------------------------------------------
# The context
# ---------------------------------------------------------------------------------------------------------------


# excerpt: ctx-context
def make_context(frame: pl.LazyFrame, margins: Sequence[dp.polars.Margin] | None = None) -> dp.Context:
    """A context on the LazyFrame: a privacy unit of 36 rows, a budget of epsilon = 1 split evenly over 5 queries."""
    return dp.Context.compositor(
        data=frame,
        privacy_unit=dp.unit_of(contributions=CONTRIBUTIONS),  # neighbours add or remove one person's 36 rows
        privacy_loss=dp.loss_of(epsilon=EPSILON),  # pure differential privacy
        split_evenly_over=QUERIES,  # each query gets epsilon / 5
        margins=margins,  # public facts about the frame; the count needs none
    )


# end excerpt


def refusal(build: Any) -> str:
    """What the library says when a query cannot be built: the message, without its surrounding whitespace."""
    try:
        build()
    except (ValueError, dp.OpenDPException) as err:
        return str(err).strip()
    return "built without complaint"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seed", type=int, default=0, help="unused: nothing here is seeded (OpenDP's noise has none)")
    parser.parse_args()

    # Section 1: the dataset. The sample is public, so its sizes are read directly, before any context exists.
    print(f"opendp version: {dp.__version__}")
    # excerpt: ctx-load
    frame = pl.scan_parquet(practitioner_datasets.france_lfs())  # a LazyFrame: the file is read only when a query runs
    print("schema: " + ", ".join(f"{name} {dtype}" for name, dtype in frame.collect_schema().items()))
    # end excerpt
    n = sizes(frame)
    print(f"rows: {n.rows:,}  people: {n.people:,}")
    print(f"years: {n.first_year} to {n.last_year}  quarters: {n.quarters}")
    print(f"responses per person: max {n.max_responses}  distinct quarters per person: max {n.max_quarters}")

    # Section 3: the context. The weighted and the bounded contexts exist only to print their budget and their
    # margin; every release below comes from `context`. The documentation recommends one context per analysis.
    # excerpt: ctx-weights
    weighted = dp.Context.compositor(
        data=frame,
        privacy_unit=dp.unit_of(contributions=CONTRIBUTIONS),
        privacy_loss=dp.loss_of(epsilon=EPSILON),
        split_by_weights=WEIGHTS,  # the first query gets twice the share of each of the other three
    )
    # end excerpt
    # excerpt: ctx-margin
    margin = dp.polars.Margin(
        # at most 150,000 responses per quarter (50,000 households of under three people, both public figures)
        # over the 36 quarters of the survey: a bound on the length of the frame that costs no privacy
        max_length=QUARTER_RESPONSES * CONTRIBUTIONS,
    )
    bounded = make_context(frame, margins=[margin])
    # end excerpt
    # excerpt: ctx-delete
    context = make_context(frame)
    del frame  # from here on, every computation on the data goes through a context
    # end excerpt
    print(f"privacy unit: {dp.unit_of(contributions=CONTRIBUTIONS)!r}")
    print(f"privacy loss: {dp.loss_of(epsilon=EPSILON)!r}")
    print(f"budget split evenly over {QUERIES}: {[round(d, 4) for d in (context.d_mids or [])]}")
    print(f"budget split by weights {WEIGHTS}: {[round(d, 4) for d in (weighted.d_mids or [])]}")

    # Section 4: a LazyFrameQuery is a mock of the frame. It knows the schema, refuses to run before a release,
    # and accepts only the expressions OpenDP has a proof for.
    mock = context.query()
    print(f"query type: {type(mock).__name__}")
    print(f"query schema: {mock.collect_schema()}")
    print(f"collect before release: {refusal(lambda: mock.collect())}")
    print(f"plain Polars mean: {refusal(lambda: context.query().select(pl.col('AGE').mean()).summarize())}")

    # Section 5: the first release, the count of responses. The query takes a share of the budget when it is
    # built and spends it when it is released; summarize reads the scale and the error bar without spending.
    # excerpt: ctx-count
    count_query = context.query().select(dp.len(signed=True))  # signed: the noise may take the count below zero
    count_summary = count_query.summarize(alpha=0.05)  # the scale and the 95% error bar, before any release
    released = count_query.release()  # spends the share; a OnceFrame that collects once
    dp_count = released.collect().item()
    # end excerpt
    print(count_summary)
    count_accuracy = float(count_summary["accuracy"].item())
    wider = float(count_query.summarize(alpha=0.01)["accuracy"].item())
    print(f"count at alpha=0.01: accuracy={wider:.4f}")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        context.query().select(dp.len()).summarize()
    future = [str(w.message).strip() for w in caught if issubclass(w.category, FutureWarning)]
    print(f"dp.len() without signed: FutureWarning: {future[0] if future else 'none'}")

    # Section 6: the margin, previewed. A sum needs a public bound on the length of the frame; a count does not.
    print(f"margin: {margin!r}")
    hours: Any = pl.col("HWUSUAL").fill_null(35.0)  # Any: OpenDP adds the `.dp` namespace at run time
    print(f"sum without a margin: {refusal(lambda: context.query().select(hours.dp.sum((0.0, 80.0))).summarize())}")
    sum_scale = float(bounded.query().select(hours.dp.sum((0.0, 80.0))).summarize()["scale"].item())
    plain_scale = CONTRIBUTIONS * 80 / (EPSILON / QUERIES)  # sensitivity over epsilon, before the float sum's rounding
    print(f"sum with the margin: scale={sum_scale:,.1f}  (36 x 80 over epsilon / 5 = {plain_scale:,.0f})")

    # Section 7: the released count, read against its error bar and the true count of this public sample.
    print(f"true count: {n.rows:,}")
    print("released (one run):")
    print(f"  count {dp_count:,}  (true {n.rows:,}, within {count_accuracy:.1f} with probability 0.95)")
    print(f"  budget spent: {[round(d, 4) for d in context.current_privacy_loss()]}")
    print(f"  second collect: {refusal(lambda: released.collect())}")


if __name__ == "__main__":
    main()
