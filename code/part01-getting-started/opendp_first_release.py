"""A first release with OpenDP: the chapter "A first release" of the OpenDP book (Part 1).

    uv run python code/part01-getting-started/opendp_first_release.py

Follows the documentation's Quickstart and Typical Workflow pages: one Laplace mechanism built with the framework API,
then the five steps of an analysis with the Context API on a list of 100 numbers, releasing a count and a mean under a
budget of epsilon = 1 split over three queries, and a fourth query that the context refuses. It prints every number the
page quotes, on labelled lines, in the order of the page's sections.

The data are 100 integers drawn uniformly from 0 to 100 with NumPy's generator at the given seed (seed 0 on the page),
standing in for a column of ages or scores; nothing about them is sensitive, so the true count and mean are printed and
compared with the releases. OpenDP draws its noise from a cryptographic generator that cannot be seeded, so the released
values change on every run; they are printed under "released (one run)" and the test checks them against the true values
within three times their 95% accuracy. The noise scales, the accuracies and the privacy maps are deterministic.

Library versions: opendp 0.16.0.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import opendp.prelude as dp

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

dp.enable_features("contrib")

EPSILON = 1.0
QUERIES = 3
BOUNDS = (0.0, 100.0)  # public: the values are scores from 0 to 100
IMPUTED = 50.0  # public: the midpoint of the bounds stands in for a missing value
N = 100


def toy_data(seed: int) -> list[float]:
    """100 integers from 0 to 100, as floats, which is what the chapter's queries take."""
    rng = np.random.default_rng(seed)
    return [float(x) for x in rng.integers(0, 101, N)]


# ---------------------------------------------------------------------------------------------------------------
# One noisy number, with the framework API
# ---------------------------------------------------------------------------------------------------------------


# excerpt: first-laplace
def laplace_mechanism(scale: float) -> dp.Measurement:
    """Laplace noise of the given scale on one float, as a measurement on the space of floats."""
    space = dp.atom_domain(T=float, nan=False), dp.absolute_distance(T=float)
    return space >> dp.m.then_laplace(scale=scale)


# end excerpt


# ---------------------------------------------------------------------------------------------------------------
# The five steps, with the Context API
# ---------------------------------------------------------------------------------------------------------------


# excerpt: first-context
def make_context(data: list[float]) -> dp.Context:
    """The context: the data behind a privacy unit of one row, a budget of epsilon = 1, split evenly over 3 queries."""
    return dp.Context.compositor(
        data=data,
        privacy_unit=dp.unit_of(contributions=1),  # neighbours add or remove one person's row
        privacy_loss=dp.loss_of(epsilon=EPSILON),  # pure differential privacy
        split_evenly_over=QUERIES,  # each query gets epsilon / 3
    )


# end excerpt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seed", type=int, default=0, help="seed of the toy data (OpenDP's noise has none)")
    args = parser.parse_args()
    seed: int = args.seed

    # Section 2: one noisy number. The privacy map turns a distance between inputs into a bound on epsilon.
    print(f"opendp version: {dp.__version__}")
    laplace = laplace_mechanism(scale=1.0)
    print(f"laplace scale 1: map(1.0)={laplace.map(1.0):.1f} map(2.0)={laplace.map(2.0):.1f}")
    print(f"laplace scale 1: check(1.0, 1.0)={laplace.check(1.0, 1.0)} check(1.0, 0.5)={laplace.check(1.0, 0.5)}")
    laplace_wide = laplace_mechanism(scale=2.0)
    print(f"laplace scale 2: map(1.0)={laplace_wide.map(1.0):.1f}")

    # Section 3: the five steps. Steps 1 to 3 are decisions; step 4 builds the context; step 5 queries it.
    data = toy_data(seed)
    true_count = len(data)
    true_mean = sum(data) / len(data)
    print(f"true count: {true_count}  true mean: {true_mean:.2f}")
    print(f"privacy unit: {dp.unit_of(contributions=1)!r}")
    print(f"privacy loss: {dp.loss_of(epsilon=EPSILON)!r}")
    print(f"bounds: {BOUNDS}  imputed value: {IMPUTED:g}")
    context = make_context(data)
    print(f"budget per query: {[round(d, 4) for d in (context.d_mids or [])]}")

    # Section 4: a count and a mean. A query takes the next share of the budget when it is built and spends it
    # when it is released, so the count is built, inspected and released before the mean is built.
    # excerpt: first-count
    count_query = context.query().count().laplace()
    count_scale = count_query.param()  # the scale the budget allows: sensitivity 1 over epsilon / 3
    count_accuracy = dp.discrete_laplacian_scale_to_accuracy(scale=count_scale, alpha=0.05)
    dp_count = count_query.release()
    # end excerpt
    print(f"count: scale={count_scale:.4f} 95% accuracy={count_accuracy:.4f}")
    # excerpt: first-mean
    mean_query = (
        context.query()
        .impute_constant(IMPUTED)  # a missing value becomes the public midpoint
        .clamp(BOUNDS)  # every value is clipped into the public bounds
        .resize(size=dp_count, constant=IMPUTED)  # the mean needs a known size: the released count
        .mean()
        .laplace()
    )
    mean_scale = mean_query.param()
    dp_mean = mean_query.release()
    # end excerpt
    # Dev time only: the scale the mean would get at the true size, which is deterministic and quotable.
    dev_query = make_context(data).query().impute_constant(IMPUTED).clamp(BOUNDS).resize(size=N, constant=IMPUTED)
    dev_scale = dev_query.mean().laplace().param()
    print(f"mean at size {N}: scale={dev_scale:.4f} 95% accuracy={dp.laplacian_scale_to_accuracy(dev_scale, 0.05):.4f}")

    # Section 5: what the context refuses. The third query spends the last share; a fourth is refused.
    print(f"budget spent after two queries: {[round(d, 4) for d in context.current_privacy_loss()]}")
    third = context.query().count().laplace()
    dp_third = third.release()
    # excerpt: first-refused
    try:
        context.query().count().laplace().release()
        refusal = "released"
    except ValueError as err:
        refusal = str(err)
    # end excerpt
    print(f"fourth query: {refusal}")

    print("released (one run):")
    print(f"  count {dp_count}  (true {true_count}, within {count_accuracy:.1f} with probability 0.95)")
    print(f"  mean scale {mean_scale:.4f}  mean {dp_mean:.2f}  (true {true_mean:.2f})")
    print(f"  third query, a second count {dp_third}")


if __name__ == "__main__":
    main()
