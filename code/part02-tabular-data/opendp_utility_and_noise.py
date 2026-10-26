"""Assessing utility and choosing the noise with OpenDP: the chapter "Assessing utility and choosing the noise" of
the OpenDP book (Part 2).

    uv run python code/part02-tabular-data/opendp_utility_and_noise.py

Follows the documentation's "Assessing Utility" page and the Polars user guide's "Gaussian Noise" page: the four
conversions between a noise scale and a 95% error bar, the scale a target error bar needs and what it costs in
budget, the same count released under pure DP (Laplace noise) and under zero-concentrated DP (Gaussian noise), the
grouped count by year and quarter with and without the bound of one record per group, the conversion of rho back to
(epsilon, delta), and a small experiment on the bias of clamping noisy counts at zero. It prints every number the
page quotes, on labelled lines, in the order of the page's sections.

Dataset: the France Labour Force Survey sample of the OpenDP tutorials (3,811,932 quarterly responses from 1,682,306
people; columns PIDENT, YEAR, QUARTER, SEX, AGE, HWUSUAL, HWACTUAL, ILOSTAT), loaded by code/practitioner_datasets.py
as Parquet and scanned lazily. The documentation reads the same file through dp.examples.get_france_lfs_path(). The
data are public, so the true counts are printed and compared with the releases.

OpenDP draws its noise from a cryptographic generator that cannot be seeded, so the released values and the means of
the clamping experiment change on every run; they are printed under "released (one run)" and "clamping experiment
(one run)", and the test checks them against the true values within three times their 95% accuracy (four standard
errors for the experiment's means). The scales, accuracies, thresholds, budgets and conversions are deterministic.
The --seed flag is accepted for uniformity with the other scripts; nothing here samples.

Library versions: opendp[polars] 0.16.0 with polars 1.36.1.
"""

import argparse
import math
import sys
import warnings
from pathlib import Path
from typing import Any

import opendp.prelude as dp
import polars as pl
from opendp.mod import Measure

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import practitioner_datasets

dp.enable_features("contrib")

EPSILON = 1.0  # the pure-DP budget of the documentation's tutorial
RHO = 0.1  # the zCDP budget of the documentation's Gaussian page
DELTA = 1e-7  # pays for releasing the group keys with a threshold
QUERIES = 5  # the budget is split evenly over five queries, as in the documentation
CONTRIBUTIONS = 36  # a person answers at most once per quarter over 9 years
ALPHA = 0.05  # the error bars hold with probability 0.95
TARGET = 100.0  # the error bar of the worked example
TOY_SCALE = 30.0  # the Gaussian scale of the clamping experiment
REPEATS = 1_000  # releases in the clamping experiment


# ---------------------------------------------------------------------------------------------------------------
# Contexts: the same data and privacy unit under three budgets
# ---------------------------------------------------------------------------------------------------------------


# excerpt: util-contexts
def laplace_context(frame: pl.LazyFrame, epsilon: float) -> dp.Context:
    """Pure differential privacy: the context adds Laplace noise."""
    return dp.Context.compositor(
        data=frame,
        privacy_unit=dp.unit_of(contributions=CONTRIBUTIONS),
        privacy_loss=dp.loss_of(epsilon=epsilon),
        split_evenly_over=QUERIES,
    )


def gaussian_context(frame: pl.LazyFrame, rho: float, delta: float | None = None) -> dp.Context:
    """Zero-concentrated differential privacy: the context adds Gaussian noise."""
    return dp.Context.compositor(
        data=frame,
        privacy_unit=dp.unit_of(contributions=CONTRIBUTIONS),
        privacy_loss=dp.loss_of(rho=rho, delta=delta),
        split_evenly_over=QUERIES,
    )


# end excerpt


# excerpt: util-bounded-context
def bounded_context(frame: pl.LazyFrame, privacy_loss: tuple[Measure, Any]) -> dp.Context:
    """A budget of either kind, with the privacy unit also bounded to one record per year and quarter."""
    return dp.Context.compositor(
        data=frame,
        privacy_unit=dp.unit_of(
            contributions=[
                dp.polars.Bound(by=[], per_group=CONTRIBUTIONS),  # at most 36 records overall
                dp.polars.Bound(by=["YEAR", "QUARTER"], per_group=1),  # and one in each year and quarter
            ]
        ),
        privacy_loss=privacy_loss,
        split_evenly_over=QUERIES,
    )


# end excerpt


# ---------------------------------------------------------------------------------------------------------------
# Reading a query's summary, converting rho, and the clamping experiment
# ---------------------------------------------------------------------------------------------------------------


def summary_line(label: str, summary: pl.DataFrame) -> str:
    """One line per `summarize` row: the distribution, the scale, the 95% accuracy and the threshold if any."""
    row = summary.row(0, named=True)
    threshold = f" threshold={row['threshold']}" if "threshold" in row else ""
    return f"{label}: {row['distribution']} scale={row['scale']:g} 95% accuracy={row['accuracy']:g}{threshold}"


def accuracy_of(summary: pl.DataFrame) -> float:
    return float(summary["accuracy"].item())


# excerpt: util-epsilon
def epsilon_of_rho(rho: float, delta: float) -> float:
    """The epsilon that OpenDP's conversion gives for rho-zCDP at delta, read off a Gaussian measurement."""
    space = dp.atom_domain(T=float, nan=False), dp.absolute_distance(T=float)
    gaussian = dp.m.make_gaussian(*space, scale=1 / math.sqrt(2 * rho))  # sensitivity 1 at this scale is rho-zCDP
    approx = dp.c.make_fix_delta(dp.c.make_zCDP_to_approxDP(gaussian), delta)
    epsilon, _ = approx.map(1.0)
    return float(epsilon)


# end excerpt


def closed_form_epsilon(rho: float, delta: float) -> float:
    """The textbook conversion rho + 2 sqrt(rho ln(1/delta)), looser than OpenDP's."""
    return rho + 2 * math.sqrt(rho * math.log(1 / delta))


# excerpt: util-clamping
def mean_noisy_count_of_nobody(signed: bool, scale: float, repeats: int) -> float:
    """The mean of `repeats` noisy counts of an empty frame, at a fixed Gaussian scale, signed or clamped at zero."""
    empty = pl.LazyFrame(schema={"A": pl.Int32})
    domain = dp.with_margin(
        dp.lazyframe_domain([dp.series_domain("A", dp.atom_domain(T="i32"))]),
        dp.polars.Margin(by=[], max_length=1000),
    )
    count = dp.m.make_private_lazyframe(
        domain, dp.symmetric_distance(), dp.zero_concentrated_divergence(), empty.select(dp.len(signed=signed)), scale
    )
    return sum(int(count(empty).collect().item()) for _ in range(repeats)) / repeats


# end excerpt


def true_counts(frame: pl.LazyFrame) -> tuple[int, pl.DataFrame]:
    """The true number of responses, and the true number per year and quarter."""
    total = int(frame.select(pl.len()).collect().item())
    groups = frame.group_by("YEAR", "QUARTER").agg(pl.len().alias("true")).sort("YEAR", "QUARTER").collect()
    return total, groups


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seed", type=int, default=0, help="accepted for uniformity; nothing here samples")
    parser.parse_args()

    frame = pl.scan_parquet(practitioner_datasets.france_lfs())
    total, groups = true_counts(frame)
    print(f"opendp version: {dp.__version__}")
    print(f"true count: {total:,}")

    # Section 1: what an accuracy statement means. The documentation's example, and the round trip back.
    scale_for_2 = float(dp.accuracy_to_gaussian_scale(accuracy=2.0, alpha=ALPHA))
    print(f"gaussian scale for accuracy 2 at alpha {ALPHA}: {scale_for_2:.4f}")
    print(f"accuracy of that scale: {float(dp.gaussian_scale_to_accuracy(scale=scale_for_2, alpha=ALPHA)):.4f}")

    # Section 2: the scale a target error bar needs, on the count of the survey.
    # excerpt: util-target
    laplace = laplace_context(frame, EPSILON)
    count_laplace = laplace.query().select(dp.len(signed=True))
    laplace_summary = count_laplace.summarize(alpha=ALPHA)
    scale_needed = float(dp.accuracy_to_discrete_laplacian_scale(accuracy=TARGET, alpha=ALPHA))
    epsilon_needed = CONTRIBUTIONS / scale_needed  # the Laplace scale is the sensitivity over epsilon
    # end excerpt
    share: float = (laplace.d_mids or [0.0])[0]
    print(summary_line("laplace count", laplace_summary))
    print(f"budget per query: epsilon={share:.4f}")
    print(f"error bar relative to the true count: {accuracy_of(laplace_summary) / total:.3%}")
    print(f"continuous laplace accuracy at scale 180: {float(dp.laplacian_scale_to_accuracy(180.0, ALPHA)):.4f}")
    print(f"discrete laplace scale for accuracy {TARGET:g}: {scale_needed:.4f}")
    print(
        f"epsilon that scale needs at sensitivity {CONTRIBUTIONS}: {epsilon_needed:.4f}  ({epsilon_needed / share:.2f} times the share)"
    )
    doubled = laplace_context(frame, 2 * EPSILON).query().select(dp.len(signed=True)).summarize(alpha=ALPHA)
    print(summary_line(f"laplace count at epsilon {2 * EPSILON:g}", doubled))

    # Section 3: Gaussian noise under zCDP. Only the budget changes; the context switches the distribution.
    # excerpt: util-gaussian
    gaussian = gaussian_context(frame, RHO)
    count_gaussian = gaussian.query().select(dp.len(signed=True))
    gaussian_summary = count_gaussian.summarize(alpha=ALPHA)
    # end excerpt
    rho_share: float = (gaussian.d_mids or [0.0])[0]
    print(f"privacy loss: {dp.loss_of(rho=RHO)!r}")
    print(summary_line("gaussian count", gaussian_summary))
    print(f"budget per query: rho={rho_share:.4f}  sqrt(2 rho)={math.sqrt(2 * rho_share):.4f}")
    print(f"continuous gaussian accuracy at scale 180: {float(dp.gaussian_scale_to_accuracy(180.0, ALPHA)):.4f}")
    quadrupled = gaussian_context(frame, 4 * RHO).query().select(dp.len(signed=True)).summarize(alpha=ALPHA)
    print(summary_line(f"gaussian count at rho {4 * RHO:g}", quadrupled))

    # Section 4: L1 against L2 sensitivity. The grouped count, with and without the bound of one record per group.
    print(
        f"L1 sensitivity of the grouped count: {CONTRIBUTIONS}  L2 when the 36 records fall in 36 groups: {math.sqrt(CONTRIBUTIONS):g}"
    )
    print(f"privacy loss with a threshold: {dp.loss_of(rho=RHO, delta=DELTA)!r}")
    unbounded = gaussian_context(frame, RHO, DELTA).query().group_by("YEAR", "QUARTER").agg(dp.len(signed=True))
    print(summary_line("grouped count, 36 contributions", unbounded.summarize(alpha=ALPHA)))
    # excerpt: util-grouped
    bounded = bounded_context(frame, dp.loss_of(rho=RHO, delta=DELTA))
    count_by_quarter = bounded.query().group_by("YEAR", "QUARTER").agg(dp.len(signed=True))
    grouped_summary = count_by_quarter.summarize(alpha=ALPHA)
    # end excerpt
    print(summary_line("grouped count, 1 per year and quarter", grouped_summary))
    laplace_bounded = bounded_context(frame, dp.loss_of(epsilon=EPSILON, delta=DELTA))
    laplace_grouped = laplace_bounded.query().group_by("YEAR", "QUARTER").agg(dp.len(signed=True))
    print(summary_line("grouped count, 1 per year and quarter, pure DP", laplace_grouped.summarize(alpha=ALPHA)))
    print(f"true groups: {groups.height}  smallest {groups['true'].min():,}  largest {groups['true'].max():,}")

    # Section 5: the guarantee in epsilon.
    print(f"rho={RHO:g} at delta={DELTA:g}: epsilon={epsilon_of_rho(RHO, DELTA):.4f}")
    print(f"closed form rho + 2 sqrt(rho ln(1/delta)): {closed_form_epsilon(RHO, DELTA):.4f}")
    print(f"rho={rho_share:g} (one query) at delta={DELTA:g}: epsilon={epsilon_of_rho(rho_share, DELTA):.4f}")
    whole: Any = dp.c.make_fix_delta(dp.c.make_zCDP_to_approxDP(gaussian.accountant), DELTA).map(gaussian.d_in)
    print(f"the whole context converted: epsilon={float(whole[0]):.4f} delta={float(whole[1]):g}")

    # Section 6: signed and unsigned counts. The means are random, so they are reported as one run.
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        dp.len()
    print(f"dp.len() without signed: warns={bool(caught)} clamped={'clamped at zero' in str(caught[0].message)}")
    print(f"expected clamped mean at scale {TOY_SCALE:g}: scale/sqrt(2 pi)={TOY_SCALE / math.sqrt(2 * math.pi):.4f}")
    print(f"standard error of the signed mean: scale/sqrt({REPEATS})={TOY_SCALE / math.sqrt(REPEATS):.4f}")
    clamped = mean_noisy_count_of_nobody(signed=False, scale=TOY_SCALE, repeats=REPEATS)
    signed = mean_noisy_count_of_nobody(signed=True, scale=TOY_SCALE, repeats=REPEATS)
    print(f"clamping experiment (one run): {REPEATS} noisy counts of nobody at gaussian scale {TOY_SCALE:g}")
    print(f"  unsigned mean {clamped:.2f}  signed mean {signed:.2f}  (true count 0)")

    # Section 7: the releases of one run, read against their error bars.
    laplace_accuracy = accuracy_of(laplace_summary)
    gaussian_accuracy = accuracy_of(gaussian_summary)
    grouped_accuracy = accuracy_of(grouped_summary)
    released_laplace = int(count_laplace.release().collect().item())
    released_gaussian = int(count_gaussian.release().collect().item())
    released_groups = count_by_quarter.release().collect().join(groups, on=["YEAR", "QUARTER"]).sort("YEAR", "QUARTER")
    print("released (one run):")
    print(
        f"  laplace count {released_laplace:,}  (true {total:,}, within {laplace_accuracy:.1f} with probability 0.95)"
    )
    print(
        f"  gaussian count {released_gaussian:,}  (true {total:,}, within {gaussian_accuracy:.1f} with probability 0.95)"
    )
    for row in released_groups.head(4).iter_rows(named=True):
        print(
            f"  {row['YEAR']} Q{row['QUARTER']} count {row['len']:,}  (true {row['true']:,}, within {grouped_accuracy:g})"
        )
    inside = released_groups.filter((pl.col("len") - pl.col("true")).abs() <= grouped_accuracy).height
    print(
        f"  groups released {released_groups.height} of {groups.height}; within {grouped_accuracy:g} of the truth: {inside}"
    )


if __name__ == "__main__":
    main()
