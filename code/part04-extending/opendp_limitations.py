"""Limitations and troubleshooting: the chapter "Limitations and troubleshooting" of the OpenDP book (Part 4).

    uv run python code/part04-extending/opendp_limitations.py

Follows the documentation's "Limitations" page and the feature listing of the API user guide, the "Troubleshooting"
page of Getting Started, and the warnings about data loading, CSV schema inference and float sums in "Working with
Tabular Data" and "Essential Statistics": it shows the finite-precision term in a float sum's sensitivity, switches
each feature flag off and on and records what the library says, infers a CSV schema from three small files, reproduces
every error message the page quotes, reads an OpenDPException with and without the Rust stack trace, and releases
one count of the France Labour Force Survey sample so that a OnceFrame can be collected twice. It prints every
message and number the page quotes, on labelled lines, in the order of the page's sections.

Dataset: the France Labour Force Survey (LFS) sample of the OpenDP tutorials (Eurostat public microdata for
France, 2005 to 2013, with a synthetic person identifier), https://github.com/opendp/dp-test-datasets, loaded by
code/practitioner_datasets.py, which caches it in data/ as Parquet: 3,811,932 quarterly responses from 1,682,306
identifiers. It is public microdata already protected by statistical disclosure control, so the true count is
printed and compared with the release. The framework examples use literal values and tiny lists.

What is deterministic and what is not. OpenDP draws its noise from a cryptographic generator that cannot be
seeded, so the released count changes on every run; it is printed under "released (one run)" and the test checks
it against the true count within three times its 95% accuracy. Everything else is deterministic and quotable: the
sensitivities, the error messages, the warning, the inferred schemas, the scales and the accuracy. The --seed
option is accepted for uniformity with the other scripts; nothing here draws from a seeded generator.

Library versions: opendp[polars] 0.16.0 with polars 1.36.1.
"""

import argparse
import io
import sys
import warnings
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import opendp.prelude as dp
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import practitioner_datasets

# Not enabled at import time, unlike the other scripts: the feature-flag section switches `contrib` on itself.

EPSILON = 1.0
DELTA = 1e-7
CONTRIBUTIONS = 36  # one response per quarter over the survey's 9 years: 9 x 4 rows at most per person
HOURS_BOUNDS = (0.0, 80.0)  # public: hours usually worked, clipped as the documentation does
IMPUTED_HOURS = 35.0  # public: the average working week in France, which the documentation fills nulls with
MAX_LENGTH = 150_000 * 36  # public: 50,000 households a quarter, under three people each, over 36 quarters
TRUE_ROWS = 3_811_932


def refusal(build: Callable[[], object]) -> str:
    """What the library says when a call is refused: the exception's type and its message, or 'no error'."""
    try:
        build()
    except (ValueError, TypeError, dp.OpenDPException, pl.exceptions.ComputeError) as err:
        return f"{type(err).__name__}: {str(err).strip()}"
    return "no error"


def first_line(text: str) -> str:
    """The first line of a message that runs over several."""
    return text.splitlines()[0]


# ---------------------------------------------------------------------------------------------------------------
# What OpenDP does not promise: the finite-precision term in a float sum's sensitivity
# ---------------------------------------------------------------------------------------------------------------


# excerpt: limits-numerics
def sum_sensitivities(d_in: int) -> tuple[float, int]:
    """The sensitivity of a float sum and of an integer sum of values in [0, 80], at `d_in` added or removed rows."""
    float_sum = (dp.vector_domain(dp.atom_domain(bounds=HOURS_BOUNDS)), dp.symmetric_distance()) >> dp.t.then_sum()
    int_sum = (dp.vector_domain(dp.atom_domain(bounds=(0, 80))), dp.symmetric_distance()) >> dp.t.then_sum()
    return float_sum.map(d_in), int_sum.map(d_in)


# end excerpt


# ---------------------------------------------------------------------------------------------------------------
# The feature flags: a plugin that needs honest-but-curious
# ---------------------------------------------------------------------------------------------------------------


def identity(value: float) -> float:
    """The function of a plugin measurement that releases its input unchanged."""
    return value


def identity_map(d_in: float) -> float:
    """A privacy map that claims the privacy loss equals the distance between inputs."""
    return d_in


# excerpt: limits-plugin
def plugin_measurement() -> dp.Measurement:
    """A plugin measurement whose privacy map is only a claim: the library cannot check `identity_map`."""
    return dp.m.make_user_measurement(
        dp.atom_domain(T=float, nan=False),
        dp.absolute_distance(T=float),
        dp.max_divergence(),
        function=identity,
        privacy_map=identity_map,
    )


# end excerpt


# ---------------------------------------------------------------------------------------------------------------
# The error messages, on the LFS behind a context
# ---------------------------------------------------------------------------------------------------------------


# excerpt: limits-context
def make_context(
    frame: pl.LazyFrame, delta: float | None = None, margins: Sequence[dp.polars.Margin] | None = None
) -> dp.Context:
    """The LFS behind a unit of 36 rows and a budget of epsilon = 1, with or without a delta, for one query."""
    return dp.Context.compositor(
        data=frame,
        privacy_unit=dp.unit_of(contributions=CONTRIBUTIONS),
        privacy_loss=dp.loss_of(epsilon=EPSILON, delta=delta) if delta else dp.loss_of(epsilon=EPSILON),
        split_evenly_over=1,
        margins=margins,
    )


# end excerpt


def hours_sum(query: Any) -> Any:
    """The documentation's float sum of usual hours: nulls filled with the public average, values clipped."""
    hours: Any = pl.col("HWUSUAL").fill_null(IMPUTED_HOURS)  # typed Any: the .dp namespace is added at run time
    return query.select(hours.dp.sum(HOURS_BOUNDS))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seed", type=int, default=0, help="accepted for uniformity; nothing here is seeded")
    parser.parse_args()
    warnings.filterwarnings("ignore", category=FutureWarning)  # the signed-count default, not this chapter's point

    # Section 1: what OpenDP does not promise. The float sum's sensitivity carries a term for rounding, so the
    # library does not assume idealized numerics there; the integer sum has none.
    print(f"opendp version: {dp.__version__}")
    dp.enable_features("contrib")
    float_sensitivity, int_sensitivity = sum_sensitivities(CONTRIBUTIONS)
    print(f"float sum of 36 rows in [0, 80]: sensitivity {float_sensitivity:.10f}")
    print(f"int sum of 36 rows in [0, 80]: sensitivity {int_sensitivity}")

    # Section 2: the feature flags, one by one. Each is switched off, tried, and switched back on.
    space = dp.atom_domain(T=float, nan=False), dp.absolute_distance(T=float)
    # excerpt: limits-contrib
    dp.disable_features("contrib")
    without_contrib = refusal(lambda: space >> dp.m.then_laplace(scale=1.0))
    dp.enable_features("contrib")
    # end excerpt
    print(f"laplace without contrib: {without_contrib}")
    print(f"plugin without honest-but-curious: {refusal(plugin_measurement)}")
    dp.enable_features("honest-but-curious")
    print(f"plugin with honest-but-curious: map(1.0)={plugin_measurement().map(1.0)}")
    dp.disable_features("honest-but-curious")
    dp.disable_features("idealized-numerics")
    float_laplace = refusal(lambda: space >> dp.m.then_laplace(scale=1.0))
    print(f"float laplace without idealized-numerics: {float_laplace}")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        dp.enable_features("floating-point")
    print(f'enable_features("floating-point"): {caught[0].category.__name__}: {caught[0].message}')
    alias_on = refusal(lambda: dp.assert_features("idealized-numerics"))
    print(f'assert_features("idealized-numerics") with the alias on: {alias_on}')
    dp.disable_features("idealized-numerics")
    alias_off = refusal(lambda: dp.assert_features("idealized-numerics"))
    print(f'assert_features("idealized-numerics") after disabling it: {alias_off}')
    both = refusal(lambda: dp.assert_features("contrib", "honest-but-curious"))
    print(f'assert_features("contrib", "honest-but-curious"): {both}')

    # Section 3: what lies outside the guarantee. Schema inference reads the values, so a private value can
    # change a column's type or make the load fail.
    # excerpt: limits-csv
    for raw in (b"x\n1\n2\n", b"x\n1\n2.5\n", b"x\n1\nabc\n"):
        print(f"csv {raw!r}: column x inferred as {pl.read_csv(io.BytesIO(raw)).schema['x']}")
    late_text = b"x\n" + b"1\n" * 200 + b"abc\n"  # a text value after 200 numbers
    failed = refusal(lambda: pl.read_csv(io.BytesIO(late_text), infer_schema_length=100))
    print(f"csv with a text value on row 201, inferred from 100 rows: {first_line(failed)}")
    as_strings = pl.read_csv(io.BytesIO(late_text), infer_schema=False).schema["x"]
    print(f"the same file with infer_schema=False: column x read as {as_strings}")
    # end excerpt

    # Section 4: the error messages, each reproduced. First the framework ones, on tiny lists.
    unit, loss = dp.unit_of(contributions=1), dp.loss_of(epsilon=EPSILON)
    ints = dp.Context.compositor(data=[1, 2, 3], privacy_unit=unit, privacy_loss=loss, split_evenly_over=1)
    float_bounds = refusal(lambda: ints.query().clamp((0.0, 10.0)).sum().laplace().param())
    print(f"float bounds on integer data: {float_bounds}")
    print(f"integer bounds on integer data: scale={ints.query().clamp((0, 10)).sum().laplace().param():.1f}")
    approximate = dp.c.make_approximate(space >> dp.m.then_laplace(scale=1.0))
    scalar_check = refusal(lambda: approximate.check(1.0, 1.0))
    print(f"check(1.0, 1.0) on an approximate measurement: {scalar_check}")
    print(f"check(1.0, (1.0, 0.0)) on the same measurement: {approximate.check(1.0, (1.0, 0.0))}")
    floats = dp.Context.compositor(data=[1.0, 2.0, 3.0], privacy_unit=unit, privacy_loss=loss, split_evenly_over=1)
    nan_clamp = refusal(lambda: floats.query().clamp(HOURS_BOUNDS).sum().laplace().param())
    print(f"clamp on floats that may be nan: {nan_clamp}")
    imputed_scale = floats.query().impute_constant(0.0).clamp(HOURS_BOUNDS).sum().laplace().param()
    print(f"impute, then clamp: scale={imputed_scale:.1f}")
    # excerpt: limits-chain
    vectors = dp.vector_domain(dp.atom_domain(T=float, nan=False)), dp.symmetric_distance()
    clamp = vectors >> dp.t.then_clamp((0.0, 10.0))
    wide_sum = dp.t.make_sum(dp.vector_domain(dp.atom_domain(bounds=(0.0, 100.0))), dp.symmetric_distance())
    mismatch = refusal(lambda: clamp >> wide_sum)  # the clamp's output has bounds [0, 10], the sum wants [0, 100]
    chained = clamp >> dp.t.then_sum()  # a partial constructor takes the clamp's output space, so they agree
    # end excerpt
    print(f"clamp to [0, 10] then a sum built for [0, 100]: {mismatch}")
    print(f"clamp to [0, 10] then then_sum(): sensitivity {chained.map(1):.1f}")

    # Then the Polars ones, on the LFS.
    frame = pl.scan_parquet(practitioner_datasets.france_lfs())
    context = make_context(frame)
    print(f"float sum without a margin: {refusal(lambda: hours_sum(context.query()).summarize())}")
    bounded = make_context(frame, margins=[dp.polars.Margin(max_length=MAX_LENGTH)])
    bounded_scale = float(hours_sum(bounded.query()).summarize()["scale"][0])
    print(f"float sum with max_length={MAX_LENGTH:,}: scale={bounded_scale:,.2f}")
    # excerpt: limits-once
    count_query = context.query().select(dp.len(signed=True))
    before_release = refusal(lambda: count_query.collect())  # the plan has no data; only a released OnceFrame does
    once = count_query.release()
    dp_count = int(once.collect()["len"][0])
    twice = refusal(lambda: once.collect())  # a second collection would be a second release
    exhausted = refusal(lambda: context.query().select(dp.len(signed=True)).release())
    # end excerpt
    summary = count_query.summarize(alpha=0.05)
    count_scale, count_accuracy = float(summary["scale"][0]), float(summary["accuracy"][0])
    print(f"count: scale={count_scale:.1f} 95% accuracy={count_accuracy:.2f}")
    print(f"collect before release: {before_release}")
    print(f"collect a OnceFrame twice: {twice}")
    print(f"query after the budget is spent: {exhausted}")
    approx = make_context(frame, delta=DELTA)
    keys = {"SEX": [1, 2], "ILOSTAT": [1, 2, 3, 4, 5, 9]}
    table_error = refusal(lambda: approx.query().select("SEX", "ILOSTAT").contingency_table(keys=keys))
    print(f"contingency table with keys for every column under (epsilon, delta): {table_error}")
    pure_table = make_context(frame).query().select("SEX", "ILOSTAT").contingency_table(keys=keys)
    print(f"the same table under pure epsilon: {type(pure_table).__name__}")

    # Section 5: reading an OpenDPException, with and without the Rust frames.
    # excerpt: limits-exception
    try:
        _ = clamp >> wide_sum
    except dp.OpenDPException as err:
        print(f"variant: {err.variant}")
        print(f"message: {first_line(err.message or '')}")
        print(f"rust frames recorded: {err.raw_traceback is not None}")
        dp.enable_features("rust-stack-trace")
        print(str(err).strip())
        dp.disable_features("rust-stack-trace")
    # end excerpt

    # The one release of the run, read against its error bar.
    print("released (one run):")
    print(f"  count {dp_count:,}  (true {TRUE_ROWS:,}, within {count_accuracy:.0f} with probability 0.95)")


if __name__ == "__main__":
    main()
