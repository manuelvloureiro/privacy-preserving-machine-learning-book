"""Combinators in OpenDP: the chapter "Combinators" of the OpenDP book (Part 3).

    uv run python code/part03-framework/opendp_combinators.py

Follows the documentation's Combinators pages: chaining (make_chain_tt, make_chain_mt, make_chain_pm and the >>
shortcut, post-processing with a Python callable, the domain and metric checks that stop a mismatched chain),
non-adaptive composition (make_composition, in pure DP and in zCDP), adaptive composition (make_adaptive_composition
and its queryable), fully adaptive composition and odometers (make_fully_adaptive_composition), privacy filters
(make_privacy_filter), compositor chaining and nesting, measure casting (make_pureDP_to_zCDP, make_zCDP_to_approxDP,
make_fix_delta, make_approximate, make_fixed_approxDP_to_approxDP) and amplification by subsampling
(make_population_amplification). It prints every number the page quotes, on labelled lines, in the order of the
page's sections.

The toy data are 20 integers from 0 to 10 drawn by NumPy's generator at the given seed (seed 0 on the page); the
measurements on them are a Laplace count and a Laplace sum of the clamped values, as in the documentation. One
example runs on the France Labour Force Survey sample (Eurostat, loaded by code/practitioner_datasets.py): the sum of
HWUSUAL under a context split over one query and over three, where composition changes the noise scale. OpenDP draws
its noise from a cryptographic generator that cannot be seeded, so the released values change on every run; they are
printed under "released (one run)" and the test checks them against the true values within three times their 95%
accuracy. The privacy maps, the profiles, the scales, the accuracies and the error messages are deterministic.

Library versions: opendp[polars] 0.16.0 with polars 1.36.1.
"""

import argparse
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import opendp.prelude as dp
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import practitioner_datasets

dp.enable_features("contrib")

N = 20
BOUNDS = (0, 10)  # public: the toy values are integers from 0 to 10
COUNT_SCALE = 1.0
SUM_SCALE = 5.0
CONTRIBUTIONS = 36  # a person answers at most once per quarter over 9 years
MAX_LENGTH = 150_000 * 36  # a public bound on the number of rows of the LFS sample
HOURS_BOUNDS = (0.0, 80.0)  # public: hours usually worked per week, with 99 meaning "not applicable"
ALPHA = 0.05  # the error bars are 95% ones


def toy_data(seed: int) -> list[int]:
    """20 integers from 0 to 10, the small vector every toy measurement sums or counts."""
    rng = np.random.default_rng(seed)
    return [int(x) for x in rng.integers(0, 11, N)]


def message(err: Exception) -> str:
    """The message inside an OpenDP error, without the variant and the traceback."""
    inner = getattr(err, "message", None)
    return str(inner) if inner is not None else str(err)


# ---------------------------------------------------------------------------------------------------------------
# Chaining
# ---------------------------------------------------------------------------------------------------------------

INT_SPACE = dp.vector_domain(dp.atom_domain(T=int)), dp.symmetric_distance()


# excerpt: comb-chain-long
def chained_by_hand() -> tuple[dp.Transformation, dp.Measurement]:
    """The clamped sum and its Laplace release, built with the long-form chainers."""
    clamp = INT_SPACE >> dp.t.then_clamp(BOUNDS)
    total = dp.t.make_sum(clamp.output_domain, clamp.output_metric)
    clamped_sum = dp.c.make_chain_tt(total, clamp)  # a transformation: clamp, then sum
    laplace = dp.m.make_laplace(clamped_sum.output_domain, clamped_sum.output_metric, scale=SUM_SCALE)
    noisy_sum = dp.c.make_chain_mt(laplace, clamped_sum)  # a measurement: the chain, then noise
    return clamped_sum, noisy_sum


# end excerpt


def count_measurement() -> dp.Measurement:
    """The Laplace count of the toy vector, at scale COUNT_SCALE."""
    return INT_SPACE >> dp.t.then_count() >> dp.m.then_laplace(scale=COUNT_SCALE)


def sum_measurement() -> dp.Measurement:
    """The Laplace sum of the clamped toy vector, at scale SUM_SCALE: the same chain as chained_by_hand with >>."""
    return INT_SPACE >> dp.t.then_clamp(BOUNDS) >> dp.t.then_sum() >> dp.m.then_laplace(scale=SUM_SCALE)


def mismatched_chains() -> tuple[str, str]:
    """The messages of a chain whose domains do not match and of one whose metrics do not match."""
    float_space = dp.vector_domain(dp.atom_domain(T=float, nan=False)), dp.symmetric_distance()
    float_sum = float_space >> dp.t.then_clamp((0.0, 10.0)) >> dp.t.then_sum()
    int_laplace = dp.m.make_laplace(dp.atom_domain(T=int), dp.absolute_distance(T=int), scale=SUM_SCALE)
    # excerpt: comb-mismatch
    try:
        chain = float_sum >> int_laplace  # a float sum feeding a Laplace mechanism on integers
        domain_error = f"chained: {chain}"
    except dp.OpenDPException as err:
        domain_error = str(err.message)
    # end excerpt
    count = INT_SPACE >> dp.t.then_count()
    float_distance_laplace = dp.m.make_laplace(dp.atom_domain(T=int), dp.absolute_distance(T=float), scale=1.0)
    try:
        chain = count >> float_distance_laplace  # the same domain, but the metrics' distance types differ
        metric_error = f"chained: {chain}"
    except dp.OpenDPException as err:
        metric_error = str(err.message)
    return domain_error, metric_error


# ---------------------------------------------------------------------------------------------------------------
# Composition
# ---------------------------------------------------------------------------------------------------------------


def gaussian_count() -> dp.Measurement:
    """The Gaussian count, whose privacy is measured in zCDP."""
    return INT_SPACE >> dp.t.then_count() >> dp.m.then_gaussian(scale=COUNT_SCALE)


def gaussian_sum() -> dp.Measurement:
    """The Gaussian sum of the clamped toy vector, in zCDP."""
    return INT_SPACE >> dp.t.then_clamp(BOUNDS) >> dp.t.then_sum() >> dp.m.then_gaussian(scale=SUM_SCALE)


# excerpt: comb-adaptive
def adaptive_compositor(d_mids: list[float]) -> dp.Measurement:
    """An adaptive compositor over the toy space: one share of pure-DP budget per query, fixed in advance."""
    return dp.c.make_adaptive_composition(
        input_domain=INT_SPACE[0],
        input_metric=INT_SPACE[1],
        output_measure=dp.max_divergence(),
        d_in=1,  # neighbours add or remove one record
        d_mids=d_mids,  # the epsilon each query may spend, in the order the queries arrive
    )


# end excerpt


def lfs_context(queries: int) -> dp.Context:
    """A context over the LFS sample with epsilon = 1 split evenly over the given number of queries."""
    frame = pl.scan_parquet(practitioner_datasets.france_lfs())
    return dp.Context.compositor(
        data=frame,
        privacy_unit=dp.unit_of(contributions=CONTRIBUTIONS),
        privacy_loss=dp.loss_of(epsilon=1.0),
        split_evenly_over=queries,
        margins=[dp.polars.Margin(max_length=MAX_LENGTH)],
    )


def hours_sum_expression() -> Any:
    """The DP sum of HWUSUAL within the public bounds.

    Typed Any because OpenDP adds the `.dp` namespace to Polars expressions at run time, which a type checker
    cannot see.
    """
    hours: Any = pl.col("HWUSUAL")
    return hours.dp.sum(HOURS_BOUNDS)


def hours_sum_query(context: dp.Context) -> Any:
    """The query for the sum of hours usually worked, with the "not applicable" code 99 filtered out."""
    return context.query().filter(pl.col("HWUSUAL") != 99.0).select(hours_sum_expression())


def true_hours_sum() -> float:
    """The sum of HWUSUAL clipped to the public bounds, on the rows the query keeps: the dev-time check."""
    frame = pl.scan_parquet(practitioner_datasets.france_lfs())
    low, high = HOURS_BOUNDS
    total = frame.filter(pl.col("HWUSUAL") != 99.0).select(pl.col("HWUSUAL").clip(low, high).sum()).collect()
    return float(total.item())


# ---------------------------------------------------------------------------------------------------------------
# Measure casting
# ---------------------------------------------------------------------------------------------------------------

FLOAT_SPACE = dp.atom_domain(T=float, nan=False), dp.absolute_distance(T=float)


def simple_epsilon(rho: float, delta: float) -> float:
    """The simple conversion of rho-zCDP to (epsilon, delta): epsilon = rho + 2 sqrt(rho ln(1/delta))."""
    return rho + 2.0 * math.sqrt(rho * math.log(1.0 / delta))


def improved_epsilon(rho: float, delta: float) -> float:
    """The improved conversion (Canonne, Kamath and Steinke 2020), minimized over the order by a grid search."""

    def at(alpha: float) -> float:
        return (
            rho * alpha
            + math.log(1.0 / delta) / (alpha - 1.0)
            + math.log1p(-1.0 / alpha)
            - math.log(alpha) / (alpha - 1.0)
        )

    alphas = [1.0 + i / 10_000 for i in range(1, 2_000_000)]
    return min(at(alpha) for alpha in alphas)


# ---------------------------------------------------------------------------------------------------------------
# Amplification
# ---------------------------------------------------------------------------------------------------------------

SAMPLE = 10
POPULATION = 100


# excerpt: comb-amplify
def sample_mean() -> dp.Measurement:
    """The Laplace mean of a sample of exactly SAMPLE values in [0, 10]: the domain carries the size."""
    space = dp.vector_domain(dp.atom_domain(bounds=(0.0, 10.0), nan=False), size=SAMPLE), dp.symmetric_distance()
    return space >> dp.t.then_mean() >> dp.m.then_laplace(scale=0.5)


def amplified_mean(population_size: int) -> dp.Measurement:
    """The same mean, with its guarantee read for the population the sample was drawn from."""
    dp.enable_features("honest-but-curious")  # the library cannot check that the data are such a sample
    return dp.c.make_population_amplification(sample_mean(), population_size=population_size)


# end excerpt


def amplification_bound(epsilon: float, sample: int, population: int) -> float:
    """The amplified epsilon for a simple random sample: ln(1 + (n/N)(e^epsilon - 1))."""
    return math.log1p(sample / population * math.expm1(epsilon))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seed", type=int, default=0, help="seed of the toy data (OpenDP's noise has none)")
    args = parser.parse_args()
    seed: int = args.seed

    print(f"opendp version: {dp.__version__}")
    data = toy_data(seed)
    true_count = len(data)
    true_sum = sum(min(max(x, BOUNDS[0]), BOUNDS[1]) for x in data)
    print(f"toy data: {data}")
    print(f"true count: {true_count}  true clamped sum: {true_sum}")

    # Section 1: chaining. The stability of the transformation times nothing: the privacy map of the measurement
    # is the sensitivity the chain delivers, divided by the scale.
    clamped_sum, noisy_sum = chained_by_hand()
    print(f"clamp >> sum: map(1)={clamped_sum.map(1)}")
    print(f"clamp >> sum >> laplace: map(1)={noisy_sum.map(1):.1f} check(1, 2.0)={noisy_sum.check(1, 2.0)}")
    shortcut = sum_measurement()
    print(f"with >>: map(1)={shortcut.map(1):.1f}")
    dp.enable_features("honest-but-curious")  # a Python post-processor cannot be checked by the library

    # excerpt: comb-postprocess
    def divide_by_size(total: int) -> float:
        return total / N  # the released sum over the public size: a mean, at no further cost

    mean_of_sum = shortcut >> divide_by_size
    # end excerpt
    print(f"post-processed: map(1)={mean_of_sum.map(1):.1f}")
    dp.disable_features("honest-but-curious")
    domain_error, metric_error = mismatched_chains()
    print(f"domain mismatch: {domain_error.strip()}")
    print(f"metric mismatch: {metric_error.strip()}")
    sum_accuracy = dp.discrete_laplacian_scale_to_accuracy(scale=SUM_SCALE, alpha=ALPHA)
    count_accuracy = dp.discrete_laplacian_scale_to_accuracy(scale=COUNT_SCALE, alpha=ALPHA)
    print(f"sum: scale={SUM_SCALE:g} 95% accuracy={sum_accuracy:.4f}")
    print(f"count: scale={COUNT_SCALE:g} 95% accuracy={count_accuracy:.4f}")

    # Section 2: non-adaptive composition. The map of the composition is the sum of the maps.
    count = count_measurement()
    # excerpt: comb-composition
    both = dp.c.make_composition([shortcut, count])  # one measurement that releases the sum and the count
    print(f"composition: map(1)={both.map(1):.1f}  parts: {shortcut.map(1):.1f} + {count.map(1):.1f}")
    # end excerpt
    print(f"composition output measure: {both.output_measure}")
    gaussian_both = dp.c.make_composition([gaussian_sum(), gaussian_count()])
    print(
        f"zCDP composition: map(1)={gaussian_both.map(1):.2f}  "
        f"parts: {gaussian_sum().map(1):.2f} + {gaussian_count().map(1):.2f}"
    )
    try:
        dp.c.make_composition([shortcut, gaussian_count()])
        mixed = "composed"
    except dp.OpenDPException as err:
        mixed = message(err)
    print(f"mixed measures: {mixed}")
    released_sum, released_count = both(data)

    # Section 3: adaptive composition. The queryable answers one measurement at a time, each within its share.
    compositor = adaptive_compositor(d_mids=[2.0, 1.0])
    print(f"adaptive: map(1)={compositor.map(1):.1f}")
    # excerpt: comb-queryable
    queryable = compositor(data)  # the data go in once; what comes back answers measurements
    adaptive_sum = queryable(shortcut)  # spends the first share, 2.0
    try:
        queryable(shortcut)  # asks 2.0 of the second share, 1.0
        over_share = "answered"
    except dp.OpenDPException as err:
        over_share = str(err.message)
    adaptive_count = queryable(count)  # spends the second share, 1.0
    # end excerpt
    print(f"second query over its share: {over_share}")
    try:
        queryable(count)
        third = "answered"
    except dp.OpenDPException as err:
        third = message(err)
    print(f"third query: {third}")
    # The context of Part 1 wraps this compositor. On the LFS, the number of shares sets the scale of a real sum.
    scales: dict[int, float] = {}
    accuracies: dict[int, float] = {}
    hours_released: dict[int, float] = {}
    for queries in (1, 3):
        context = lfs_context(queries)
        d_mids = [round(d, 4) for d in (context.d_mids or [])]
        print(
            f"LFS context over {queries}: accountant map(36)={context.accountant.map(CONTRIBUTIONS):.1f} d_mids={d_mids}"
        )
        query = hours_sum_query(context)
        summary = query.summarize(alpha=ALPHA)
        scales[queries] = float(summary["scale"][0])
        accuracies[queries] = float(summary["accuracy"][0])
        print(f"LFS HWUSUAL sum over {queries}: scale={scales[queries]:,.2f} 95% accuracy={accuracies[queries]:,.1f}")
        hours_released[queries] = float(query.release().collect().item())
    hours_true = true_hours_sum()
    print(f"true HWUSUAL sum: {hours_true:,.0f}")

    # Section 4: fully adaptive composition. The odometer's queryable reports the loss so far.
    # excerpt: comb-odometer
    odometer = dp.c.make_fully_adaptive_composition(
        input_domain=INT_SPACE[0], input_metric=INT_SPACE[1], output_measure=dp.max_divergence()
    )
    odometer_queryable = odometer(data)
    loss_before = odometer_queryable.privacy_loss(1)  # nothing spent yet
    odometer_sum = odometer_queryable(shortcut)
    loss_after_sum = odometer_queryable.privacy_loss(1)
    odometer_count = odometer_queryable(count)
    loss_after_count = odometer_queryable.privacy_loss(1)
    # end excerpt
    odometer_second_count = odometer_queryable(count)
    print(
        f"odometer loss: before={loss_before:.1f} after sum={loss_after_sum:.1f} after count={loss_after_count:.1f} "
        f"after another count={odometer_queryable.privacy_loss(1):.1f}"
    )
    print(f"odometer loss at d_in=2 after three queries: {odometer_queryable.privacy_loss(2):.1f}")

    # Section 5: privacy filters. A cap turns the odometer into a measurement.
    # excerpt: comb-filter
    privacy_filter = dp.c.make_privacy_filter(odometer, d_in=1, d_out=2.0)
    filter_queryable = privacy_filter(data)
    filter_count_1 = filter_queryable(count)
    filter_count_2 = filter_queryable(count)
    try:
        filter_queryable(count)  # would take the loss to 3.0
        refusal = "answered"
    except dp.OpenDPException as err:
        refusal = str(err.message)
    # end excerpt
    print(f"filter: map(1)={privacy_filter.map(1):.1f}")
    print(f"filter third query: {refusal}")
    try:
        filter_queryable.privacy_loss(1)
        after = "reported"
    except dp.OpenDPException as err:
        after = message(err)
    print(f"filter loss after refusal: {after}")
    unsplit = dp.Context.compositor(
        data=data, privacy_unit=dp.unit_of(contributions=1), privacy_loss=dp.loss_of(epsilon=2.0)
    )
    unsplit_count = unsplit.query().count().laplace(scale=COUNT_SCALE).release()
    print(
        f"context without a split: queryable={type(unsplit.queryable).__name__} "
        f"spent={unsplit.current_privacy_loss():.1f} remaining={unsplit.remaining_privacy_loss():.1f}"
    )

    # Section 6: compositor chaining and nesting.
    clamped_sum_only = INT_SPACE >> dp.t.then_clamp(BOUNDS) >> dp.t.then_sum()
    # excerpt: comb-chain-compositor
    sum_compositor = clamped_sum_only >> dp.c.then_adaptive_composition(
        output_measure=dp.max_divergence(),
        d_in=clamped_sum_only.map(1),  # the sum moves by at most 10 when one record changes
        d_mids=[2.0, 1.0],
    )
    sum_space = clamped_sum_only.output_domain, clamped_sum_only.output_metric
    laplace_on_sum = sum_space >> dp.m.then_laplace(scale=SUM_SCALE)  # a query on the sum, not on the vector
    # end excerpt
    print(f"transformation >> compositor: d_in={clamped_sum_only.map(1)} map(1)={sum_compositor.map(1):.1f}")
    print(f"laplace on the sum: map(10)={laplace_on_sum.map(10):.1f}")
    nested_sum = sum_compositor(data)(laplace_on_sum)
    # excerpt: comb-nest
    outer = adaptive_compositor(d_mids=[3.0, 1.0])
    inner = adaptive_compositor(d_mids=[2.0, 1.0])  # a measurement costing 3.0: it fits the outer's first share
    outer_queryable = outer(data)
    inner_queryable = outer_queryable(inner)  # the answer to a compositor query is a queryable
    inner_sum = inner_queryable(shortcut)
    outer_count = outer_queryable(count)  # the outer's second share, while the inner queryable stays open
    inner_count = inner_queryable(count)  # pure DP composes concurrently, so the inner one still answers
    # end excerpt
    print(f"nested: outer map(1)={outer.map(1):.1f} inner map(1)={inner.map(1):.1f}")

    # Section 7: measure casting.
    laplace_one = FLOAT_SPACE >> dp.m.then_laplace(scale=1.0)
    as_zcdp = dp.c.make_pureDP_to_zCDP(laplace_one)
    print(f"pure DP to zCDP: epsilon={laplace_one.map(1.0):.1f} rho={as_zcdp.map(1.0):.2f}")
    rho = 0.1
    delta = 1e-7
    # excerpt: comb-profile
    gaussian = FLOAT_SPACE >> dp.m.then_gaussian(scale=math.sqrt(1 / (2 * rho)))  # rho = 1 / (2 scale^2)
    profile_measurement = dp.c.make_zCDP_to_approxDP(gaussian)
    profile = profile_measurement.map(1.0)  # a curve, not a number
    epsilon_at_delta = profile.epsilon(delta=delta)
    delta_at_one = profile.delta(epsilon=1.0)
    fixed = dp.c.make_fix_delta(profile_measurement, delta=delta)  # the curve at one delta: an (epsilon, delta) pair
    # end excerpt
    print(f"gaussian: rho={gaussian.map(1.0):.4f}")
    print(
        f"profile at rho={rho:g}: epsilon(delta={delta:g})={epsilon_at_delta:.4f} delta(epsilon=1)={delta_at_one:.6f}"
    )
    print(
        f"simple conversion: {simple_epsilon(rho, delta):.4f}  improved conversion: {improved_epsilon(rho, delta):.4f}"
    )
    fixed_epsilon, fixed_delta = fixed.map(1.0)
    print(f"fix_delta: map(1.0)=({fixed_epsilon:.4f}, {fixed_delta:g})  measure={fixed.output_measure}")
    approximate = dp.c.make_approximate(laplace_one)
    approx_epsilon, approx_delta = approximate.map(1.0)
    print(f"approximate: map(1.0)=({approx_epsilon:.1f}, {approx_delta:.1f})  measure={approximate.output_measure}")
    flat = dp.c.make_fixed_approxDP_to_approxDP(approximate).map(1.0)
    print(
        f"fixed to profile: epsilon(delta={delta:g})={flat.epsilon(delta=delta):.1f} epsilon(delta=0)={flat.epsilon(delta=0.0):.1f}"
    )
    gaussian_count_fixed = dp.c.make_fix_delta(dp.c.make_zCDP_to_approxDP(gaussian_count()), delta=delta)
    mixed_composition = dp.c.make_composition([dp.c.make_approximate(count), gaussian_count_fixed])
    mixed_epsilon, mixed_delta = mixed_composition.map(1)
    print(f"laplace count + gaussian count: map(1)=({mixed_epsilon:.4f}, {mixed_delta:g})")

    # Section 8: amplification by subsampling.
    mean = sample_mean()
    print(f"sample mean: map(2)={mean.map(2):.4f}")
    for population in (POPULATION, 10 * POPULATION):
        amplified = amplified_mean(population)
        bound = amplification_bound(mean.map(2), SAMPLE, population)
        print(f"amplified, population {population}: map(2)={amplified.map(2):.4f}  formula={bound:.4f}")
    try:
        dp.c.make_population_amplification(shortcut, population_size=POPULATION)
        unsized = "amplified"
    except dp.OpenDPException as err:
        unsized = message(err)
    print(f"amplifying an unsized domain: {unsized}")
    dp.disable_features("honest-but-curious")

    print("released (one run):")
    print(f"  composition: sum {released_sum}  count {released_count}  (true {true_sum} and {true_count})")
    print(f"  adaptive: sum {adaptive_sum}  count {adaptive_count}")
    print(f"  odometer: sum {odometer_sum}  counts {odometer_count} {odometer_second_count}")
    print(f"  filter: counts {filter_count_1} {filter_count_2}")
    print(f"  context without a split: count {unsplit_count}")
    print(
        f"  nested: sum on the sum {nested_sum}  inner sum {inner_sum}  inner count {inner_count}  outer count {outer_count}"
    )
    for queries in (1, 3):
        print(
            f"  LFS HWUSUAL sum over {queries}: {hours_released[queries]:,.0f}  "
            f"(true {hours_true:,.0f}, within {accuracies[queries]:,.0f} with probability 0.95)"
        )


if __name__ == "__main__":
    main()
