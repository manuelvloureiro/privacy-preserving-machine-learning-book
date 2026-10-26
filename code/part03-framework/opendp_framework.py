"""The framework behind OpenDP: the chapter "A framework to understand differential privacy" of the OpenDP book (Part 3).

    uv run python code/part03-framework/opendp_framework.py

Follows the documentation's Framework API pages (A Framework to Understand DP, Differential Privacy with OpenDP, Core
Structures, Supporting Elements) and its API Comparison: the five elements of a Laplace measurement and what its
privacy map and `check` say; a chain of a clamp, a sum and Laplace noise on a vector of integers, with the stability
of the sum and the privacy loss of the whole chain read from their maps; and the typical workflow of the book's first
chapter redone with the framework API next to the Context API, which gives the same scales. It prints every number the
page quotes, on labelled lines, in the order of the page's sections.

The data. The vector of the sum example is the `HWUSUAL` column of the France Labour Force Survey sample (hours usually
worked per week), loaded by code/practitioner_datasets.py, with the nulls and the code 99 ("not applicable") dropped
and the remaining values collected to a list of integers between 0 and 80. The workflow runs on the first chapter's
toy data: 100 integers from 0 to 100 drawn by NumPy's generator at the given seed (seed 0 on the page). OpenDP draws
its noise from a cryptographic generator that cannot be seeded, so the released values change on every run; they are
printed under "released (one run)" and the test checks them against the true values within three times their 95%
accuracy. The domains, metrics, measures, maps, checks, scales and accuracies are deterministic.

Library versions: opendp[polars] 0.16.0 with polars 1.36.1.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import opendp.prelude as dp
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import practitioner_datasets

dp.enable_features("contrib")

LAPLACE_SCALE = 5.0  # the documentation's running example: Laplace noise of scale 5 on one float
HOURS_BOUNDS = (0, 80)  # public: hours usually worked per week, as the documentation clamps HWUSUAL
CONTRIBUTIONS = 36  # public: a person answers at most once a quarter over 9 years, so at most 36 rows
EPSILON = 1.0
QUERIES = 3
BOUNDS = (0.0, 100.0)  # public: the toy scores run from 0 to 100
IMPUTED = 50.0  # public: the midpoint of the bounds stands in for a missing score
N = 100


def toy_data(seed: int) -> list[float]:
    """The first chapter's data: 100 integers from 0 to 100, as floats."""
    rng = np.random.default_rng(seed)
    return [float(x) for x in rng.integers(0, 101, N)]


def usual_hours() -> list[int]:
    """`HWUSUAL` as a list of integers: the hours reported by the respondents the question applied to.

    The column holds 20,819 nulls and 2,294,921 rows coded 99 for "not applicable"; both are dropped, and the rest are
    whole numbers of hours from 0 to 80. The filter is a condition on each row alone, so it keeps each person's rows
    at most 36.
    """
    frame = pl.scan_parquet(practitioner_datasets.france_lfs())
    hours = frame.filter(pl.col("HWUSUAL").is_not_null() & (pl.col("HWUSUAL") < 99)).select("HWUSUAL").collect()
    return [int(h) for h in hours["HWUSUAL"].to_list()]


# ---------------------------------------------------------------------------------------------------------------
# The running example: the five elements of a Laplace measurement
# ---------------------------------------------------------------------------------------------------------------


# excerpt: frame-laplace
def laplace_measurement(scale: float) -> dp.Measurement:
    """Laplace noise of the given scale on one finite float, compared by absolute distance, under pure DP."""
    return dp.m.make_laplace(
        dp.atom_domain(T=float, nan=False),  # input domain: any float that is not NaN
        dp.absolute_distance(T=float),  # input metric: |a - b|, the units of the sensitivity
        scale=scale,
    )


# end excerpt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seed", type=int, default=0, help="seed of the toy data (OpenDP's noise has none)")
    args = parser.parse_args()
    seed: int = args.seed

    # Section 1: transformations and measurements. The five elements of the Laplace measurement, read back.
    print(f"opendp version: {dp.__version__}")
    laplace = laplace_measurement(LAPLACE_SCALE)
    print(f"laplace scale 5: type={type(laplace).__name__}")
    print(f"laplace scale 5: input domain={laplace.input_domain!r}")
    print(f"laplace scale 5: input metric={laplace.input_metric!r}")
    print(f"laplace scale 5: output measure={laplace.output_measure!r}")
    print(f"laplace scale 5: function={type(laplace.function).__name__}")

    # Section 5: the maps. The privacy map turns a sensitivity into a bound on epsilon; check compares it with a budget.
    # excerpt: frame-map
    epsilon_at_1 = laplace.map(d_in=1.0)  # the sensitivity over the scale: 1 / 5
    epsilon_at_10 = laplace.map(d_in=10.0)
    fits = laplace.check(d_in=1.0, d_out=0.2)  # is map(1.0) <= 0.2?
    too_tight = laplace.check(d_in=1.0, d_out=0.19)
    # end excerpt
    print(f"laplace scale 5: map(1.0)={epsilon_at_1} map(10.0)={epsilon_at_10}")
    print(f"laplace scale 5: check(1.0, 0.2)={fits} check(1.0, 0.19)={too_tight}")
    noisy_zero = laplace(0.0)
    laplace_accuracy = dp.laplacian_scale_to_accuracy(scale=LAPLACE_SCALE, alpha=0.05)
    print(f"laplace scale 5: 95% accuracy={laplace_accuracy:.4f}")

    # Section 6: a chain on the hours usually worked. Clamp and sum are transformations; the stability of the sum is
    # the sensitivity the Laplace measurement needs, and the chain's privacy map composes the two maps.
    hours = usual_hours()
    true_sum = sum(hours)
    print(f"hours: n={len(hours):,} min={min(hours)} max={max(hours)} true sum={true_sum:,}")
    # excerpt: frame-chain
    clamp = dp.t.make_clamp(dp.vector_domain(dp.atom_domain(T=int)), dp.symmetric_distance(), bounds=HOURS_BOUNDS)
    bounded_sum = dp.t.make_sum(clamp.output_domain, clamp.output_metric)  # the clamp's output space is its input space
    laplace_int = dp.m.make_laplace(bounded_sum.output_domain, bounded_sum.output_metric, scale=LAPLACE_SCALE)
    noisy_sum = clamp >> bounded_sum >> laplace_int  # the chain: its input space is the clamp's, its measure Laplace's
    sum_sensitivity_1 = bounded_sum.map(d_in=1)  # one row of at most 80 hours
    sum_sensitivity_36 = bounded_sum.map(d_in=CONTRIBUTIONS)  # 36 rows
    epsilon_at_scale_5 = noisy_sum.map(d_in=CONTRIBUTIONS)
    # end excerpt
    print(f"clamp: input domain={clamp.input_domain!r}")
    print(f"clamp: output domain={clamp.output_domain!r}  map(36)={clamp.map(CONTRIBUTIONS)}")
    print(f"sum: output domain={bounded_sum.output_domain!r}  output metric={bounded_sum.output_metric!r}")
    print(f"sum: map(1)={sum_sensitivity_1} map(36)={sum_sensitivity_36}")
    print(f"chain scale 5: input domain={noisy_sum.input_domain!r}")
    print(f"chain scale 5: input metric={noisy_sum.input_metric!r}  output measure={noisy_sum.output_measure!r}")
    print(f"chain scale 5: map(36)={epsilon_at_scale_5}")
    # excerpt: frame-chain-scale
    sum_scale = dp.binary_search_param(
        lambda scale: clamp >> bounded_sum >> dp.m.then_laplace(scale), d_in=CONTRIBUTIONS, d_out=EPSILON
    )
    noisy_sum = clamp >> bounded_sum >> dp.m.then_laplace(sum_scale)
    sum_accuracy = dp.discrete_laplacian_scale_to_accuracy(scale=sum_scale, alpha=0.05)
    released_sum = noisy_sum(hours)
    # end excerpt
    print(f"chain for epsilon 1 at d_in 36: scale={sum_scale:.1f} map(36)={noisy_sum.map(CONTRIBUTIONS)}")
    print(
        f"chain for epsilon 1 at d_in 36: 95% accuracy={sum_accuracy:,.1f} ({sum_accuracy / true_sum:.5%} of the sum)"
    )

    # Section 7: the typical workflow, framework API next to Context API. Steps 1 to 3 are the same decisions as in
    # the first chapter; step 4 builds the compositor by hand; step 5 builds each query by hand.
    data = toy_data(seed)
    true_count = len(data)
    true_mean = sum(data) / len(data)
    print(f"true count: {true_count}  true mean: {true_mean:.2f}")
    # excerpt: frame-unit-loss
    d_in = 1  # neighbouring datasets differ by at most one row...
    input_metric = dp.symmetric_distance()  # ...counted as additions and removals
    input_domain = dp.vector_domain(dp.atom_domain(T=float))  # a vector of floats, NaN allowed
    d_out = EPSILON  # the distributions of two releases differ by at most epsilon...
    privacy_measure = dp.max_divergence()  # ...measured by the max divergence, which is pure DP
    # end excerpt
    print(f"unit: d_in={d_in} input metric={input_metric!r}  (Context API: {dp.unit_of(contributions=1)!r})")
    print(f"loss: d_out={d_out} output measure={privacy_measure!r}  (Context API: {dp.loss_of(epsilon=EPSILON)!r})")
    print(f"input domain: {input_domain!r}")
    # excerpt: frame-mediate
    compositor = dp.c.make_adaptive_composition(
        input_domain=input_domain,
        input_metric=input_metric,
        output_measure=privacy_measure,
        d_in=d_in,
        d_mids=[d_out / QUERIES] * QUERIES,  # the shares of the budget, one per query
    )
    queryable = compositor(data)  # a measurement; invoking it on the data gives the queryable that answers queries
    # end excerpt
    print(f"compositor: type={type(compositor).__name__} map(1)={compositor.map(d_in)}")
    print(f"queryable: type={type(queryable).__name__}")
    context = dp.Context.compositor(
        data=data,
        privacy_unit=dp.unit_of(contributions=1),
        privacy_loss=dp.loss_of(epsilon=EPSILON),
        split_evenly_over=QUERIES,
    )
    print(
        f"context accountant: type={type(context.accountant).__name__} d_mids={[round(d, 4) for d in context.d_mids or []]}"
    )

    # Step 5, the count. An explicit scale of 3 fails the check by one rounding step; the search finds one that passes.
    # excerpt: frame-count
    count = dp.t.make_count(input_domain, input_metric)
    count_sensitivity = count.map(d_in)  # adding or removing one row changes a count by 1
    count_scale = dp.binary_search_param(
        lambda scale: count >> dp.m.then_laplace(scale), d_in=d_in, d_out=d_out / QUERIES
    )
    count_measurement = count >> dp.m.then_laplace(count_scale)
    dp_count = queryable(count_measurement)
    # end excerpt
    explicit = count >> dp.m.then_laplace(3.0)
    print(f"count: sensitivity={count_sensitivity}  output metric={count.output_metric!r}")
    print(f"count at scale 3.0: map(1)={explicit.map(d_in)!r} check(1, 1/3)={explicit.check(d_in, d_out / QUERIES)}")
    print(f"count by search: scale={count_scale!r} check(1, 1/3)={count_measurement.check(d_in, d_out / QUERIES)}")
    count_accuracy = dp.discrete_laplacian_scale_to_accuracy(scale=count_scale, alpha=0.05)
    context_count_scale = context.query().count().laplace().param()
    print(
        f"count scale: framework={count_scale:.4f} context={context_count_scale:.4f} equal={count_scale == context_count_scale}"
    )
    print(f"count: 95% accuracy={count_accuracy:.4f}")

    # Step 5, the mean. Its scale depends on the released count, so the deterministic comparison is at the true size.
    # excerpt: frame-mean
    mean = (
        dp.t.make_impute_constant(input_domain, input_metric, IMPUTED)  # NaN becomes the public midpoint
        >> dp.t.then_clamp(BOUNDS)
        >> dp.t.then_resize(size=dp_count, constant=IMPUTED)  # the released count fixes the size
        >> dp.t.then_mean()
    )
    mean_scale = dp.binary_search_param(
        lambda scale: mean >> dp.m.then_laplace(scale), d_in=d_in, d_out=d_out / QUERIES
    )
    dp_mean = queryable(mean >> dp.m.then_laplace(mean_scale))
    # end excerpt
    print(f"mean: output domain={mean.output_domain!r}  output metric={mean.output_metric!r}")
    mean_at_n = (
        dp.t.make_impute_constant(input_domain, input_metric, IMPUTED)
        >> dp.t.then_clamp(BOUNDS)
        >> dp.t.then_resize(size=N, constant=IMPUTED)
        >> dp.t.then_mean()
    )
    mean_scale_at_n = dp.binary_search_param(
        lambda scale: mean_at_n >> dp.m.then_laplace(scale), d_in=d_in, d_out=d_out / QUERIES
    )
    context_mean_scale_at_n = (
        context.query().impute_constant(IMPUTED).clamp(BOUNDS).resize(size=N, constant=IMPUTED).mean().laplace().param()
    )
    mean_accuracy = dp.laplacian_scale_to_accuracy(scale=mean_scale_at_n, alpha=0.05)
    print(f"mean at size {N}: sensitivity={mean_at_n.map(d_in):.4f}")
    print(
        f"mean scale at size {N}: framework={mean_scale_at_n:.4f} context={context_mean_scale_at_n:.4f} "
        f"equal={mean_scale_at_n == context_mean_scale_at_n}"
    )
    print(f"mean at size {N}: 95% accuracy={mean_accuracy:.4f}")
    context_mean_scale = (
        context.query()
        .impute_constant(IMPUTED)
        .clamp(BOUNDS)
        .resize(size=dp_count, constant=IMPUTED)
        .mean()
        .laplace()
        .param()
    )
    print(f"mean scale at the released count: equal={mean_scale == context_mean_scale}")

    # The third query spends the last share; the queryable refuses a fourth.
    dp_third = queryable(count_measurement)
    # excerpt: frame-refused
    try:
        queryable(count_measurement)
        refusal = "released"
    except dp.OpenDPException as err:
        refusal = f"{err.variant}: {err.message}"
    # end excerpt
    print(f"fourth query: {refusal}")

    print("released (one run):")
    print(f"  laplace scale 5 on 0.0: {noisy_zero:.2f}  (within {laplace_accuracy:.1f} with probability 0.95)")
    print(f"  hours sum {released_sum:,}  (true {true_sum:,}, within {sum_accuracy:,.0f} with probability 0.95)")
    print(f"  count {dp_count}  (true {true_count}, within {count_accuracy:.1f} with probability 0.95)")
    print(f"  mean scale {mean_scale:.4f}  mean {dp_mean:.2f}  (true {true_mean:.2f})")
    print(f"  third query, a second count {dp_third}")


if __name__ == "__main__":
    main()
