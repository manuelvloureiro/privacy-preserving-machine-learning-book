"""Utilities and the Context API inside: the chapter "Utilities and the Context API inside" of the OpenDP book (Part 3).

    uv run python code/part03-framework/opendp_utilities.py

Follows the documentation's Typing, Parameter Search, Serialization, Context and API Comparison pages, and the
"Inferred type is XXX, expected YYY" entry of its Troubleshooting page: type descriptors and the type error, the
three binary searches and the same search written by hand, a measurement's round trip through JSON and a Polars
plan's round trip through bytes, then the Context API rebuilt from the framework: the adaptive composition behind a
context, the partial chain behind `.laplace()` without a scale, the search that fixes it, the queryable that
releases it, a compositor nested in a query, and the casts between privacy measures. It prints every number the
page quotes, on labelled lines, in the order of the page's sections.

The plain-data queries run on the 100 integers of the first chapter, drawn uniformly from 0 to 100 with NumPy's
generator at the given seed (seed 0 on the page); the Polars plan counts the rows of the France Labour Force Survey
sample (3,811,932 rows from 1,682,306 people, loaded by code/practitioner_datasets.py). OpenDP draws its noise from
a cryptographic generator that cannot be seeded, so the released counts change on every run; they are printed
under "released (one run)" and the test checks them against the true values within three times their 95% accuracy.
Everything else is deterministic: type descriptors, error messages, the scales the searches find, the privacy maps
and the budget splits.

Library versions: opendp[polars] 0.16.0 with polars 1.36.1.
"""

import argparse
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import opendp.prelude as dp
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import practitioner_datasets

dp.enable_features("contrib")

EPSILON = 1.0
QUERIES = 3  # the first chapter's split: each query gets epsilon / 3
DELTA = 1e-7  # the slack of the approximate budget, as in the first chapter's loss_of example
RHO = 0.1  # the zero-concentrated budget of the same example
N = 100
CONTRIBUTIONS = 36  # a person answers at most once per quarter over 9 years
MAX_LENGTH = 150_000 * 36  # a public bound on the number of rows of the survey


def toy_data(seed: int) -> list[float]:
    """The first chapter's data: 100 integers from 0 to 100, as floats."""
    rng = np.random.default_rng(seed)
    return [float(x) for x in rng.integers(0, 101, N)]


def first_line(err: BaseException) -> str:
    """The first line of an OpenDP error, without the link to the documentation that some of them carry."""
    return str(err).strip().splitlines()[0].split(". See ")[0]


def plain_query(context: dp.Context) -> dp.Query:
    """`context.query()` on plain data is a Query; on a LazyFrame it is a LazyFrameQuery."""
    query = context.query()
    if not isinstance(query, dp.Query):
        raise TypeError("the data of this context are not a plain list")
    return query


def make_context(data: list[float]) -> dp.Context:
    """The first chapter's context: one row per person, epsilon = 1 split evenly over 3 queries."""
    return dp.Context.compositor(
        data=data,
        privacy_unit=dp.unit_of(contributions=1),
        privacy_loss=dp.loss_of(epsilon=EPSILON),
        split_evenly_over=QUERIES,
    )


# ---------------------------------------------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------------------------------------------

# excerpt: utils-space
SPACE = dp.vector_domain(dp.atom_domain(T=float)), dp.symmetric_distance()  # the space of a list of floats


def count_laplace(scale: float) -> dp.Measurement:
    """Count the rows, then add discrete Laplace noise of the given scale: the first chapter's count query."""
    return SPACE >> dp.t.then_count() >> dp.m.then_laplace(scale=scale)


# end excerpt


def type_error(action: Callable[[], object]) -> str:
    """The first line of the TypeError that `action` raises, or "no error"."""
    try:
        action()
    except TypeError as err:
        return first_line(err)
    return "no error"


# ---------------------------------------------------------------------------------------------------------------
# Parameter search
# ---------------------------------------------------------------------------------------------------------------

FLOAT_SPACE = dp.atom_domain(T=float, nan=False), dp.absolute_distance(T=float)


def make_laplace(scale: float) -> dp.Measurement:
    """Laplace noise of the given scale on one float."""
    return FLOAT_SPACE >> dp.m.then_laplace(scale=scale)


def make_gaussian(scale: float) -> dp.Measurement:
    """Gaussian noise of the given scale on one float, measured in zero-concentrated differential privacy."""
    return FLOAT_SPACE >> dp.m.then_gaussian(scale=scale)


# excerpt: utils-by-hand
def search_by_hand(passes: Callable[[float], bool], low: float, high: float, tolerance: float) -> tuple[float, int]:
    """Bisection on a predicate that fails at `low` and passes at `high`: the smallest passing value, and the steps."""
    steps = 0
    while high - low > tolerance:
        middle = (low + high) / 2
        if passes(middle):
            high = middle  # the boundary is at or below the middle
        else:
            low = middle  # the boundary is above it
        steps += 1
    return high, steps


# end excerpt


def search_error(action: Callable[[], object]) -> str:
    """The message of the ValueError that a failed search raises."""
    try:
        action()
    except ValueError as err:
        return first_line(err)
    return "no error"


# ---------------------------------------------------------------------------------------------------------------
# Serialization
# ---------------------------------------------------------------------------------------------------------------


# excerpt: utils-lfs-context
def lfs_context(frame: pl.LazyFrame) -> dp.Context:
    """The survey behind a privacy unit of one person (at most 36 rows) and a budget of epsilon = 1 for one query."""
    return dp.Context.compositor(
        data=frame,
        privacy_unit=dp.unit_of(contributions=CONTRIBUTIONS),
        privacy_loss=dp.loss_of(epsilon=EPSILON),
        split_evenly_over=1,
        margins=[dp.polars.Margin(max_length=MAX_LENGTH)],
    )


# end excerpt


def pair(d: Any) -> str:
    """An (epsilon, delta) pair, formatted."""
    return f"({d[0]:.4f}, {d[1]:.1e})"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seed", type=int, default=0, help="seed of the toy data (OpenDP's noise has none)")
    args = parser.parse_args()
    seed: int = args.seed
    data = toy_data(seed)
    share = EPSILON / QUERIES

    # Section 1: types. A descriptor is a string, a Python type or an opendp.typing object; nothing is cast.
    print(f"opendp version: {dp.__version__}")
    print(f"int means {dp.RuntimeType.parse(int)}, float means {dp.RuntimeType.parse(float)}")
    print(f"count output: {count_laplace(3.0).input_space[0]} -> {dp.t.make_count(*SPACE).output_domain}")
    # excerpt: utils-type-error
    int_space = dp.atom_domain(T=int), dp.absolute_distance(T=int)
    laplace_int = int_space >> dp.m.then_laplace(scale=1.0)  # Laplace noise on a 32-bit integer
    try:
        laplace_int.map(1.5)  # a float sensitivity, on a metric whose distances are integers
        message = "no error"
    except TypeError as err:
        message = first_line(err)
    # end excerpt
    print(f"map(1.5) on an integer metric: TypeError: {message}")
    print(f"map(2) on an integer metric: {laplace_int.map(2)}")
    print(f"float d_in on the count: TypeError: {type_error(lambda: count_laplace(3.0).map(1.0))}")
    print(f"integer d_in on the count: {count_laplace(3.0).map(1):.4f}")
    approx_int = dp.c.make_approximate(laplace_int)
    print(f"scalar epsilon on an approximate measure: TypeError: {type_error(lambda: approx_int.check(1, 1.0))}")
    print(f"pair on an approximate measure: {approx_int.check(1, (1.0, 0.0))}")

    # Section 2: parameter search. The predicate is check, monotone in the parameter, so bisection finds the edge.
    # excerpt: utils-search
    scale = dp.binary_search_param(make_laplace, d_in=1.0, d_out=share)  # the smallest scale within the share
    chain = dp.binary_search_chain(make_laplace, d_in=1.0, d_out=share)  # the measurement at that scale
    largest_d_in = dp.binary_search(lambda d_in: make_laplace(1.0).check(d_in, EPSILON))  # the other way round
    # end excerpt
    print(
        f"search: scale={scale!r} chain.map(1.0)={chain.map(1.0):.4f} chain.check(1.0, share)={chain.check(1.0, share)}"
    )
    print(f"search: largest d_in at scale 1 within epsilon 1: {largest_d_in}")
    print(
        f"search: gaussian scale for rho=1 at d_in=1: {dp.binary_search_param(make_gaussian, d_in=1.0, d_out=1.0):.4f}"
    )
    by_hand_scale, steps = search_by_hand(lambda s: make_laplace(s).check(1.0, share), 0.0, 100.0, 1e-6)
    print(f"search by hand: scale={by_hand_scale:.6f} after {steps} steps")
    print(f"search with bounds (0, 100): {dp.binary_search_param(make_laplace, 1.0, share, bounds=(0.0, 100.0))!r}")
    narrow = search_error(lambda: dp.binary_search_param(make_laplace, 1.0, share, bounds=(0.0, 1.0)))
    print(f"search with bounds (0, 1): ValueError: {narrow}")
    constant = search_error(lambda: dp.binary_search(lambda x: True))
    print(f"search on a constant predicate: ValueError: {constant}")

    # Section 3: serialization. A measurement is a recipe of constructors; JSON stores the recipe.
    # excerpt: utils-serialize
    measurement = count_laplace(3.0)
    text = dp.serialize(measurement)  # JSON that names each constructor and its arguments
    restored = dp.deserialize(text)
    same_space = restored.input_space == measurement.input_space
    # end excerpt
    print(f"serialized: {len(text)} characters, starting {text[:33]}")
    print(f"restored: map(1)={restored.map(1):.4f} same input space: {same_space}")
    resolved = plain_query(make_context(data)).count().laplace().resolve()
    print(f"param before a round trip: {getattr(resolved, 'param', None)!r}")
    print(f"param after a round trip: {getattr(dp.deserialize(dp.serialize(resolved)), 'param', None)!r}")
    try:
        dp.serialize(make_context(data))
        context_message = "serialized"
    except Exception as err:  # noqa: BLE001  (the encoder raises a bare Exception)
        context_message = str(err).split(": It may")[0]
    print(f"serialize a context: {context_message}")
    frame = pl.scan_parquet(practitioner_datasets.france_lfs())
    true_rows = frame.select(pl.len()).collect().item()
    # excerpt: utils-plan
    client = lfs_context(pl.LazyFrame(schema=frame.collect_schema()))  # the schema, but no rows
    plan = client.query().select(dp.len(signed=True)).polars_plan.serialize()  # bytes
    server = lfs_context(frame)  # the real data
    query = server.deserialize_polars_plan(plan)
    summary = query.summarize(alpha=0.05)
    # end excerpt
    print(f"plan: {len(plan)} bytes")
    print(f"plan on the server: scale={summary['scale'][0]:.1f} 95% accuracy={summary['accuracy'][0]:.4f}")
    dp_rows = query.release().collect().item()

    # Section 4: the compositor inside a context.
    context = make_context(data)
    accountant = context.accountant
    print(f"context accountant: {accountant.input_domain} {accountant.input_metric} {accountant.output_measure}")
    print(f"context d_in: {context.d_in}  d_mids: {[round(d, 4) for d in (context.d_mids or [])]}")
    print(f"context accountant map(1): {accountant.map(1):.4f}")
    # excerpt: utils-accountant
    by_hand_accountant = dp.c.make_adaptive_composition(
        input_domain=SPACE[0],
        input_metric=SPACE[1],
        output_measure=dp.max_divergence(),
        d_in=1,  # the privacy unit's distance: one row added or removed
        d_mids=[share] * QUERIES,  # the shares, which add up to the budget
    )
    queryable = by_hand_accountant(data)  # answers measurement queries on the data, and keeps count
    # end excerpt
    print(f"by hand accountant map(1): {by_hand_accountant.map(1):.4f}  queryable: {queryable}")
    weighted = dp.Context.compositor(
        data=data,
        privacy_unit=dp.unit_of(contributions=1),
        privacy_loss=dp.loss_of(epsilon=EPSILON),
        split_by_weights=[1, 2, 1],
    )
    print(f"d_mids by weights [1, 2, 1]: {[round(d, 4) for d in (weighted.d_mids or [])]}")

    # Section 5: a query inside. The chain grows from a space to a transformation to a partial chain.
    # excerpt: utils-query
    query = plain_query(context)  # the chain is the data's space: a domain and a metric
    counted = query.count()  # the chain is a Transformation
    noisy = counted.laplace()  # no scale: the chain is a PartialChain, waiting for one number
    is_partial = isinstance(noisy._chain, dp.PartialChain)
    # end excerpt
    print(
        f"query chain: {type(query._chain).__name__} -> {type(counted._chain).__name__} -> {type(noisy._chain).__name__}"
    )
    print(f"partial chain: {is_partial}")
    print(f"resolve a transformation: ValueError: {search_error(counted.resolve)}")
    print(f"param of .laplace(): {noisy.param()!r}")
    print(f"param of .laplace(scale=3.0): {counted.laplace(scale=3.0).param()!r}")
    print(f"two missing parameters: ValueError: {search_error(lambda: noisy.gaussian())}")
    # excerpt: utils-resolve
    by_hand_param = dp.binary_search_param(count_laplace, d_in=1, d_out=share)  # what resolve runs on the chain
    by_hand_measurement = count_laplace(by_hand_param)
    dp_count_by_hand = queryable(by_hand_measurement)  # what release does: the queryable runs the measurement
    # end excerpt
    print(f"resolve: context param={noisy.param()!r} by hand={by_hand_param!r} equal={by_hand_param == noisy.param()}")
    count_accuracy = dp.discrete_laplacian_scale_to_accuracy(scale=by_hand_param, alpha=0.05)
    dp_count = context(noisy.resolve())
    print(f"d_mids after context(measurement): {[round(d, 4) for d in (context.d_mids or [])]}")

    # Section 6: a compositor inside a query, and the measure casts.
    # excerpt: utils-nested
    inner = plain_query(context).compositor(split_evenly_over=2).release()  # a Context inside one share
    inner_scale = plain_query(inner).count().laplace().param()
    # end excerpt
    print(f"inner context: d_in={inner.d_in} d_out={inner.d_out:.4f} d_mids={[round(d, 4) for d in inner.d_mids]}")
    print(f"inner count scale: {inner_scale:.4f}")
    print(f"outer d_mids after the inner compositor: {[round(d, 4) for d in (context.d_mids or [])]}")
    # excerpt: utils-cast
    approx = dp.Context.compositor(
        data=data,
        privacy_unit=dp.unit_of(contributions=1),
        privacy_loss=dp.loss_of(epsilon=EPSILON, delta=DELTA),  # approximate differential privacy
        split_evenly_over=QUERIES,
    )
    laplace_cast = plain_query(approx).count().laplace().resolve()  # pure DP, cast to approximate DP
    gaussian_cast = plain_query(approx).count().gaussian().resolve()  # zCDP, converted and its delta fixed
    # end excerpt
    print(f"approx share: {pair(approx.d_mids[0]) if approx.d_mids else None}")
    print(f"laplace under (epsilon, delta): {laplace_cast.output_measure} map(1)={pair(laplace_cast.map(1))}")
    print(f"gaussian under (epsilon, delta): {gaussian_cast.output_measure} map(1)={pair(gaussian_cast.map(1))}")
    print(f"param after a cast: {plain_query(approx).count().laplace().param()!r}")
    concentrated = dp.Context.compositor(
        data=data,
        privacy_unit=dp.unit_of(contributions=1),
        privacy_loss=dp.loss_of(rho=RHO),
        split_evenly_over=QUERIES,
    )
    laplace_rho = plain_query(concentrated).count().laplace().resolve()
    print(f"laplace under rho: {laplace_rho.output_measure} map(1)={laplace_rho.map(1):.4f}")
    print(f"gaussian under rho: scale={plain_query(concentrated).count().gaussian().param():.4f}")
    print(
        f"gaussian under epsilon: ValueError: {search_error(plain_query(make_context(data)).count().gaussian().resolve)}"
    )

    print("released (one run):")
    print(f"  survey rows {dp_rows}  (true {true_rows}, within {summary['accuracy'][0]:.1f} with probability 0.95)")
    print(f"  count by hand {dp_count_by_hand}  (true {N}, within {count_accuracy:.1f} with probability 0.95)")
    print(f"  count through the context {dp_count}  (true {N})")


if __name__ == "__main__":
    main()
