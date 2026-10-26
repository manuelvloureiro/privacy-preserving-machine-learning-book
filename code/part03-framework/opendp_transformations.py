"""Transformations in OpenDP's framework API: the chapter "Transformations" of the OpenDP book (Part 3).

    uv run python code/part03-framework/opendp_transformations.py

Follows the documentation's Transformations user guide (the listing of constructors, and the pages on resize, the sum,
the mean and quantiles): preprocessing steps (clamp, resize) and aggregators (sum, mean, variance, quantile scores)
built with `make_` and `then_` constructors and chained with `>>`, with the stability of each read from `map`. It
prints every number the page quotes, on labelled lines, in the order of the page's sections.

Dataset: the France Labour Force Survey sample (3,811,932 quarterly responses), loaded by code/practitioner_datasets.py
as Parquet. The vector is the `HWUSUAL` column (hours usually worked) without its nulls and without the code 99 ("not
applicable"), cast to int: 1,496,192 values. The privacy unit is one person, who answers at most 36 times, so every
stability map is read at an input distance of 36, as the documentation's `dp.unit_of(contributions=36)`. The bounds
0 to 98 hours come from the survey's codebook, not from the data.

What is deterministic and what is not. OpenDP draws its noise from a cryptographic generator that cannot be seeded,
so the released sum, mean and median change on every run, and so does the float sum whose input is truncated to
2^20 rows by a random sample; they are printed under "released (one run)" and the test checks them against the true
values within three times their 95% accuracy (the median against the gap between the two best scores). The
sensitivities, scales, accuracies, scores and true values are deterministic. The `--seed` flag is kept for the
uniform interface of the book's scripts: this script draws no toy data of its own.

Library versions: opendp[polars] 0.16.0 with polars 1.36.1.
"""

import argparse
import statistics
import sys
from pathlib import Path

import opendp.prelude as dp
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import practitioner_datasets

dp.enable_features("contrib")

CONTRIBUTIONS = 36  # a person answers at most once per quarter over 9 years
BOUNDS = (0, 98)  # public: the codebook's range of hours usually worked; 99 is "not applicable"
FLOAT_BOUNDS = (0.0, 98.0)
EPSILON = 0.5  # the budget each of the three measurements below is calibrated to on its own
CANDIDATES = list(range(BOUNDS[0], BOUNDS[1] + 1))  # every whole number of hours is a candidate median
SIZE_LIMIT = 4_000_000  # a public upper bound on the number of rows, above the survey's 3.8 million


# ---------------------------------------------------------------------------------------------------------------
# The data
# ---------------------------------------------------------------------------------------------------------------


# excerpt: trans-hours
def hours_usually_worked(frame: pl.LazyFrame) -> list[int]:
    """The `HWUSUAL` column without nulls and without the code 99, as a list of ints."""
    kept = frame.filter(pl.col("HWUSUAL").is_not_null() & (pl.col("HWUSUAL") != 99))
    return kept.select(pl.col("HWUSUAL").cast(pl.Int64)).collect()["HWUSUAL"].to_list()


# end excerpt


# ---------------------------------------------------------------------------------------------------------------
# Sums
# ---------------------------------------------------------------------------------------------------------------


# excerpt: trans-int-sum
def int_sum(size: int | None = None) -> dp.Transformation:
    """Clamp to the public bounds, then sum; with a size the domain says how many rows there are."""
    space = dp.vector_domain(dp.atom_domain(T=int), size=size), dp.symmetric_distance()
    return space >> dp.t.then_clamp(BOUNDS) >> dp.t.then_sum()


# end excerpt


# excerpt: trans-float-sum
def float_sum(size_limit: int | None = None) -> dp.Transformation:
    """The float sum: `then_sum` truncates to 2^20 rows; a size limit of your own lifts that."""
    space = dp.vector_domain(dp.atom_domain(T=float, nan=False)), dp.symmetric_distance()
    clamped = space >> dp.t.then_clamp(FLOAT_BOUNDS)
    if size_limit is None:
        return clamped >> dp.t.then_sum()
    return clamped >> dp.t.make_bounded_float_checked_sum(size_limit=size_limit, bounds=FLOAT_BOUNDS)


# end excerpt


# ---------------------------------------------------------------------------------------------------------------
# The mean and the variance
# ---------------------------------------------------------------------------------------------------------------


# excerpt: trans-mean
def mean_chain(size: int, constant: float) -> dp.Transformation:
    """Clamp, resize to a known size (padding with the constant), then the mean over that size."""
    space = dp.vector_domain(dp.atom_domain(T=float, nan=False)), dp.symmetric_distance()
    return space >> dp.t.then_clamp(FLOAT_BOUNDS) >> dp.t.then_resize(size=size, constant=constant) >> dp.t.then_mean()


# end excerpt


def variance_chain(size: int, constant: float) -> dp.Transformation:
    """The same preprocessing as the mean, then the sample variance."""
    space = dp.vector_domain(dp.atom_domain(T=float, nan=False)), dp.symmetric_distance()
    clamped = space >> dp.t.then_clamp(FLOAT_BOUNDS)
    return clamped >> dp.t.then_resize(size=size, constant=constant) >> dp.t.then_variance()


# ---------------------------------------------------------------------------------------------------------------
# The median through quantile scores
# ---------------------------------------------------------------------------------------------------------------


# excerpt: trans-median
def median_scores() -> dp.Transformation:
    """One score per candidate: how far it is from being the median; the lowest score is the best candidate."""
    space = dp.vector_domain(dp.atom_domain(T=int)), dp.symmetric_distance()
    return space >> dp.t.then_quantile_score_candidates(candidates=CANDIDATES, alpha=0.5)


def median_measurement(scale: float) -> dp.Measurement:
    """Noisy max over the scores releases an index; `negate` makes the lowest score the one most likely to win."""
    return median_scores() >> dp.m.then_noisy_max(dp.max_divergence(), scale=scale, negate=True)


# end excerpt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seed", type=int, default=0, help="unused: no toy data here, and OpenDP's noise has no seed")
    parser.parse_args()

    frame = pl.scan_parquet(practitioner_datasets.france_lfs())
    hours = hours_usually_worked(frame)
    n = len(hours)
    true_sum = sum(hours)
    true_mean = true_sum / n
    true_median = statistics.median(hours)
    print(f"opendp version: {dp.__version__}")
    print(f"hours vector: {n:,} values")
    print(f"true sum: {true_sum:,}  true mean: {true_mean:.4f}  true median: {true_median:g}")

    # Section: the sum. Integers first, exact; then floats, whose sensitivity carries a rounding term and whose
    # unsized sum truncates the data to 2^20 rows.
    int_chain = int_sum()
    print(f"int sum: value {int_chain(hours):,}  map(1)={int_chain.map(1)}  map(36)={int_chain.map(CONTRIBUTIONS)}")
    sized_chain = int_sum(size=n)
    print(f"sized int sum: map(36)={sized_chain.map(CONTRIBUTIONS)}  map(35)={sized_chain.map(35)}")
    fhours = [float(h) for h in hours]
    truncated = float_sum()
    print(f"float sum, size limit 2^20={2**20:,}: map(36)={truncated.map(CONTRIBUTIONS):.10f}")
    dp_truncated = truncated(fhours)
    lifted = float_sum(size_limit=SIZE_LIMIT)
    print(
        f"float sum, size limit {SIZE_LIMIT:,}: value {lifted(fhours):,.0f}  map(36)={lifted.map(CONTRIBUTIONS):.10f}"
    )
    sum_scale = int_chain.map(CONTRIBUTIONS) / EPSILON
    sum_accuracy = dp.discrete_laplacian_scale_to_accuracy(scale=sum_scale, alpha=0.05)
    sum_measurement = int_chain >> dp.m.then_laplace(scale=sum_scale)
    print(
        f"sum release: scale={sum_scale:g}  map(36)={sum_measurement.map(CONTRIBUTIONS):.4f}  95% accuracy={sum_accuracy:,.1f}"
    )
    dp_sum = sum_measurement(hours)

    # Section: the mean and the variance. The mean needs a size in its input domain; resize puts it there at the
    # price of doubling the input distance.
    try:
        _ = (dp.vector_domain(dp.atom_domain(T=float, nan=False)), dp.symmetric_distance()) >> dp.t.then_mean()
        size_error = "built"
    except dp.OpenDPException as err:
        size_error = err.message or ""
    print(f"mean without a size: {size_error}")
    mean = mean_chain(size=n, constant=true_median)
    print(f"mean chain: value {mean(fhours):.4f}  map(36)={mean.map(CONTRIBUTIONS):.7f}")
    mean_scale = mean.map(CONTRIBUTIONS) / EPSILON
    mean_accuracy = dp.laplacian_scale_to_accuracy(scale=mean_scale, alpha=0.05)
    mean_measurement = mean >> dp.m.then_laplace(scale=mean_scale)
    print(
        f"mean release: scale={mean_scale:.7f}  map(36)={mean_measurement.map(CONTRIBUTIONS):.4f}  95% accuracy={mean_accuracy:.4f}"
    )
    dp_mean = mean_measurement(fhours)
    variance = variance_chain(size=n, constant=true_median)
    print(f"variance chain: value {variance(fhours):.4f}  map(36)={variance.map(CONTRIBUTIONS):.4f}")

    # Section: quantiles. The scores are deterministic; the selection among them is the next chapter's noisy max.
    scores_chain = median_scores()
    scores: list[int] = scores_chain(hours)
    ranked = sorted(range(len(CANDIDATES)), key=lambda i: scores[i])
    best, runner_up = ranked[0], ranked[1]
    print(
        f"median scores: lowest {scores[best]:,} at {CANDIDATES[best]} hours, "
        f"next {scores[runner_up]:,} at {CANDIDATES[runner_up]}  map(36)={scores_chain.map(CONTRIBUTIONS)}"
    )
    median_scale = 2 * scores_chain.map(CONTRIBUTIONS) / EPSILON
    median_meas = median_measurement(scale=median_scale)
    gap = scores[runner_up] - scores[best]
    print(
        f"median release: scale={median_scale:g}  map(36)={median_meas.map(CONTRIBUTIONS):.4f}  score gap={gap:,}  gap/scale={gap / median_scale:.0f}"
    )
    released_index: int = median_meas(hours)
    dp_median = CANDIDATES[released_index]  # naming the candidate behind the released index is post-processing

    # Section: reading a chain. `check` along the chain, and the two refusals a mismatched chain produces.
    print(
        f"int sum chain: check(36, 3528)={int_chain.check(CONTRIBUTIONS, 3528)}  check(36, 3527)={int_chain.check(CONTRIBUTIONS, 3527)}"
    )
    clamped = (dp.vector_domain(dp.atom_domain(T=int)), dp.symmetric_distance()) >> dp.t.then_clamp(BOUNDS)
    # excerpt: trans-mismatch
    try:
        _ = clamped >> dp.t.make_sum(dp.vector_domain(dp.atom_domain(bounds=(0, 100))), dp.symmetric_distance())
        mismatch = "built"
    except dp.OpenDPException as err:
        mismatch = f"{err.variant}: {err.message}"
    # end excerpt
    print(f"mismatched chain: {mismatch.rstrip()}")

    print("released (one run):")
    print(f"  sum {dp_sum:,}  (true {true_sum:,}, within {sum_accuracy:,.0f} with probability 0.95)")
    print(f"  float sum truncated to 2^20 rows {dp_truncated:,.0f}  (true {true_sum:,}; about {2**20 / n:.0%} of it)")
    print(f"  mean {dp_mean:.4f}  (true {true_mean:.4f}, within {mean_accuracy:.4f} with probability 0.95)")
    print(f"  median {dp_median}  (true {true_median:g})")


if __name__ == "__main__":
    main()
