"""Plugins: the chapter "Plugins" of the OpenDP book (Part 4).

    uv run python code/part04-extending/opendp_plugins.py

Follows the documentation's Plugins pages: a measurement plugin (the constant measurement of the docs and a Laplace
mechanism written in Python), a transformation plugin (the docs' repeat), a constructor registered for the Context
API, the Theil-Sen regression built from plugins, and the private choice of grouping columns with the exponential
mechanism. It prints every number the page quotes, on labelled lines, in the order of the page's sections.

Data. The regression runs on the documentation's toy: 100 points with x drawn from a standard normal and
y = 2x + 1 plus normal noise of standard deviation 0.5, from NumPy's legacy generator at the given seed (seed 0 on
the page), which also drives the random pairing inside the Theil-Sen transformation. The choice of grouping columns
runs on the France Labour Force Survey sample (3,811,932 rows, 1,682,306 people, loaded by
code/practitioner_datasets.py); the privacy unit is a person, who appears in at most 36 quarters.

What is random: OpenDP draws its noise from a cryptographic generator that cannot be seeded, so every released value
changes from run to run; they are printed under "released (one run)" and the test checks them against bounds the
page states: a Laplace release within three times its 95% accuracy, a selected candidate among those whose score is
within the printed bound of the best. Everything else (maps, checks, error messages, scores, the candidates within
the bound) is deterministic.

Library versions: opendp[polars] 0.16.0 with polars 1.36.1 and numpy.
"""

import argparse
import math
import sys
from itertools import chain, combinations
from pathlib import Path
from typing import Any

import numpy as np
import opendp.prelude as dp
import polars as pl
from numpy.typing import NDArray

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import practitioner_datasets

dp.enable_features("contrib", "honest-but-curious")

EPSILON = 1.0
CONTRIBUTIONS = 36  # a person appears at most once per quarter over 9 years
BETA = 1e-4  # the failure probability of the test's bounds on a selection
N_POINTS = 100  # the documentation's toy regression
X_BOUNDS = (-3.0, 3.0)
Y_BOUNDS = (-10.0, 10.0)
CANDIDATE_COLUMNS = ["SEX", "ILOSTAT", "YEAR", "QUARTER", "AGE", "PIDENT"]
MIN_GROUP = 100  # a group with fewer rows is not worth releasing: the public threshold of the score


def first_line(err: BaseException) -> str:
    """The first non-empty line of an OpenDP error, without the link to the discussion."""
    text = next(line.strip() for line in str(err).splitlines() if line.strip())
    return text.split(". See ")[0]


# ---------------------------------------------------------------------------------------------------------------
# A measurement plugin
# ---------------------------------------------------------------------------------------------------------------


# excerpt: plug-constant
def make_base_constant(constant: str) -> dp.Measurement:
    """The documentation's example: a measurement that ignores its input and returns a constant."""

    def function(_arg: int) -> str:
        return constant  # the output does not depend on the data

    def privacy_map(_d_in: int) -> float:
        return 0.0  # so two neighbouring inputs give the same distribution: a privacy loss of zero

    return dp.m.make_user_measurement(
        input_domain=dp.atom_domain(T=int),
        input_metric=dp.absolute_distance(T=int),
        output_measure=dp.max_divergence(),
        function=function,
        privacy_map=privacy_map,
        TO=str,
    )


# end excerpt


# excerpt: plug-laplace
def make_numpy_laplace(input_domain: dp.Domain, input_metric: dp.Metric, scale: float) -> dp.Measurement:
    """Laplace noise drawn by NumPy instead of OpenDP: the same map as the built-in, but on your word."""
    rng = np.random.default_rng()

    def function(arg: float) -> float:
        return float(arg + rng.laplace(scale=scale))

    def privacy_map(d_in: float) -> float:
        return d_in / scale  # the Laplace mechanism's loss: the sensitivity over the scale

    return dp.m.make_user_measurement(
        input_domain, input_metric, dp.max_divergence(), function=function, privacy_map=privacy_map, TO=float
    )


# end excerpt


def make_halved_laplace(scale: float) -> dp.Measurement:
    """The same mechanism with a map that claims half the loss, which the library accepts as readily."""
    rng = np.random.default_rng()

    def function(arg: float) -> float:
        return float(arg + rng.laplace(scale=scale))

    def privacy_map(d_in: float) -> float:
        return d_in / (2 * scale)  # wrong: the real loss is twice this

    return dp.m.make_user_measurement(
        dp.atom_domain(T=float, nan=False),
        dp.absolute_distance(T=float),
        dp.max_divergence(),
        function=function,
        privacy_map=privacy_map,
        TO=float,
    )


# ---------------------------------------------------------------------------------------------------------------
# A transformation plugin
# ---------------------------------------------------------------------------------------------------------------


# excerpt: plug-repeat
def make_repeat(multiplicity: int) -> dp.Transformation:
    """The documentation's example: a transformation that repeats each record `multiplicity` times."""

    def function(arg: list[int]) -> list[int]:
        return arg * multiplicity

    def stability_map(d_in: int) -> int:
        return d_in * multiplicity  # one added record of the input is `multiplicity` added records of the output

    return dp.t.make_user_transformation(
        input_domain=dp.vector_domain(dp.atom_domain(T=int)),
        input_metric=dp.symmetric_distance(),
        output_domain=dp.vector_domain(dp.atom_domain(T=int)),
        output_metric=dp.symmetric_distance(),
        function=function,
        stability_map=stability_map,
    )


# end excerpt


def repeated_sum() -> dp.Transformation:
    """The documentation's chain up to the sum: strings cast to integers, repeated twice, clamped to 1..2, summed."""
    space = dp.vector_domain(dp.atom_domain(T=str)), dp.symmetric_distance()
    return space >> dp.t.then_cast_default(TOA=int) >> make_repeat(2) >> dp.t.then_clamp((1, 2)) >> dp.t.then_sum()


def mismatched_chain() -> str:
    """The error when the repeat follows a cast to floats: its input domain is a vector of integers."""
    space = dp.vector_domain(dp.atom_domain(T=str)), dp.symmetric_distance()
    try:
        chained = space >> dp.t.then_cast_default(TOA=float) >> make_repeat(2)
    except dp.OpenDPException as err:
        return first_line(err)
    return f"chained: {chained}"


# ---------------------------------------------------------------------------------------------------------------
# Registering a constructor for the Context API
# ---------------------------------------------------------------------------------------------------------------


# excerpt: plug-register
def make_anything_constant(input_domain: dp.Domain, input_metric: dp.Metric, constant: str) -> dp.Measurement:
    """The documentation's constructor for the Context API: it accepts whatever space the previous step outputs."""

    def function(_arg: Any) -> str:
        return constant

    def privacy_map(_d_in: Any) -> float:
        return 0.0

    return dp.m.make_user_measurement(
        input_domain, input_metric, dp.max_divergence(), function=function, privacy_map=privacy_map, TO=str
    )


dp.register(make_anything_constant)  # from now on, context.query().anything_constant(...)
# end excerpt


def make_int_constant(constant: str) -> dp.Measurement:
    """The documentation's second constructor: no partial application, so its input space is fixed to integers."""

    def function(_arg: int) -> str:
        return constant

    def privacy_map(_d_in: int) -> float:
        return 0.0

    return dp.m.make_user_measurement(
        dp.atom_domain(T=int),
        dp.absolute_distance(T=int),
        dp.max_divergence(),
        function=function,
        privacy_map=privacy_map,
        TO=str,
    )


def select_cols() -> dp.Measurement:
    """A constructor whose name breaks the naming rule."""
    return make_int_constant("denied")


# ---------------------------------------------------------------------------------------------------------------
# The Theil-Sen regression
# ---------------------------------------------------------------------------------------------------------------


def toy_regression(seed: int) -> NDArray[np.float64]:
    """The documentation's toy: 100 points with y = 2x + 1 plus normal noise, as a two-column array."""
    np.random.seed(seed)
    x = np.random.normal(size=N_POINTS, loc=0, scale=1.0)
    y = 2 * x + 1 + np.random.normal(size=N_POINTS, loc=0, scale=0.5)
    return np.stack([x, y], axis=1)


# excerpt: plug-pairwise
def pairwise_predict(data: NDArray[np.float64], x_cuts: NDArray[np.float64]) -> NDArray[np.float64]:
    """Pair the rows at random, fit a line through each pair, and predict y at the two cut points."""
    data = np.array(data, copy=True)[: len(data) // 2 * 2]
    np.random.shuffle(data)
    p1, p2 = np.array_split(data, 2)
    dx, dy = (p2 - p1).T
    x_bar, y_bar = (p1 + p2).T / 2
    points = dy / dx * (x_cuts[None].T - x_bar) + y_bar
    return points.T[dx != 0]


# end excerpt


# excerpt: plug-pairwise-transformation
def make_pairwise_predict(x_cuts: NDArray[np.float64], runs: int = 1) -> dp.Transformation:
    """A transformation from the (x, y) rows to the predictions: one row per pair and run."""

    def function(arg: NDArray[np.float64]) -> NDArray[np.float64]:
        return np.vstack([pairwise_predict(arg, x_cuts) for _ in range(runs)])

    def stability_map(d_in: int) -> int:
        return d_in * runs  # each record sits in one pair, so it changes one prediction per run

    return dp.t.make_user_transformation(
        input_domain=dp.numpy.array2_domain(num_columns=2, T=float),
        input_metric=dp.symmetric_distance(),
        output_domain=dp.numpy.array2_domain(num_columns=2, T=float),
        output_metric=dp.symmetric_distance(),
        function=function,
        stability_map=stability_map,
    )


# end excerpt


# excerpt: plug-select-column
def make_select_column(j: int) -> dp.Transformation:
    """Column j of the predictions as a vector; one row in is one row out, so the stability is the input distance."""

    def function(arg: NDArray[np.float64]) -> NDArray[np.float64]:
        return arg[:, j]

    def stability_map(d_in: int) -> int:
        return d_in

    return dp.t.make_user_transformation(
        input_domain=dp.numpy.array2_domain(num_columns=2, T=float),
        input_metric=dp.symmetric_distance(),
        output_domain=dp.vector_domain(dp.atom_domain(T=float)),
        output_metric=dp.symmetric_distance(),
        function=function,
        stability_map=stability_map,
    )


# end excerpt


# excerpt: plug-medians
def make_private_medians(output_measure: dp.Measure, y_bounds: tuple[float, float], scale: float) -> dp.Measurement:
    """The private median of the predictions at each cut, by the library's quantile, composed."""
    median = dp.m.then_private_quantile(output_measure, candidates=np.linspace(*y_bounds, 100), alpha=0.5, scale=scale)
    return dp.c.make_composition(
        [
            make_select_column(0) >> dp.t.then_drop_null() >> median,
            make_select_column(1) >> dp.t.then_drop_null() >> median,
        ]
    )


# end excerpt


# excerpt: plug-theil-sen
def make_private_theil_sen(
    output_measure: dp.Measure, x_bounds: tuple[float, float], y_bounds: tuple[float, float], scale: float
) -> dp.Measurement:
    """Pair and predict, release the two medians, then solve for the slope and the intercept (post-processing)."""
    x_cuts = x_bounds[0] + (x_bounds[1] - x_bounds[0]) * np.array([0.25, 0.75])
    p_inv = np.linalg.inv(np.vstack([x_cuts, np.ones_like(x_cuts)]).T)  # the line through two points

    def to_coefficients(medians: list[float]) -> tuple[float, float]:
        slope, intercept = p_inv @ np.array(medians)
        return float(slope), float(intercept)

    return make_pairwise_predict(x_cuts) >> make_private_medians(output_measure, y_bounds, scale) >> to_coefficients


# end excerpt


def theil_sen_cuts() -> NDArray[np.float64]:
    return X_BOUNDS[0] + (X_BOUNDS[1] - X_BOUNDS[0]) * np.array([0.25, 0.75])


def candidates_within(
    values: list[float], candidates: NDArray[np.float64], bound: float
) -> tuple[float, float, int, float, float]:
    """The best candidate and its score, and how many candidates score within `bound` of it, from which to which."""
    space = dp.vector_domain(dp.atom_domain(T=float, nan=False)), dp.symmetric_distance()
    scorer = space >> dp.t.then_quantile_score_candidates(candidates=candidates, alpha=0.5)
    scores = np.array(scorer(values), dtype=float)
    near = candidates[scores <= scores.min() + bound]
    return float(candidates[scores.argmin()]), float(scores.min()), len(near), float(near.min()), float(near.max())


# ---------------------------------------------------------------------------------------------------------------
# Choosing grouping columns privately
# ---------------------------------------------------------------------------------------------------------------


def powerset(columns: list[str]) -> list[tuple[str, ...]]:
    """Every non-empty set of the columns, as the documentation builds the candidates."""
    return list(chain.from_iterable(combinations(columns, r) for r in range(1, len(columns) + 1)))


# excerpt: plug-score
def score(frame: pl.DataFrame, columns: tuple[str, ...], min_rows: int) -> float:
    """How many groups of these columns have at least `min_rows` rows."""
    return float(frame.group_by(list(columns)).len().filter(pl.col("len") >= min_rows).height)


# end excerpt


# excerpt: plug-score-transformation
def make_grouping_cols_score(
    input_domain: dp.Domain, input_metric: dp.Metric, candidates: list[tuple[str, ...]], min_rows: int
) -> dp.Transformation:
    """A transformation from the frame to one score per candidate set of grouping columns."""
    dp.assert_features("contrib")

    def function(frame: pl.DataFrame) -> list[float]:
        return [score(frame, columns, min_rows) for columns in candidates]

    def stability_map(d_in: int) -> float:
        return float(d_in)  # one row changes one group's size, so it moves each score by at most 1

    return dp.t.make_user_transformation(
        input_domain=input_domain,
        input_metric=input_metric,
        output_domain=dp.vector_domain(dp.atom_domain(T=float, nan=False)),
        output_metric=dp.linf_distance(T=float),  # not monotonic: a person's rows may be added and removed
        function=function,
        stability_map=stability_map,
    )


# end excerpt


# excerpt: plug-select
def make_select_grouping_cols(
    input_domain: dp.Domain, input_metric: dp.Metric, candidates: list[tuple[str, ...]], min_rows: int, scale: float
) -> dp.Measurement:
    """Score the candidates, pick one by the exponential mechanism, and name its columns."""

    def name(index: int) -> tuple[str, ...]:
        return candidates[index]  # post-processing: the index is already private

    scores = make_grouping_cols_score(input_domain, input_metric, candidates, min_rows)
    return scores >> dp.m.then_noisy_max(dp.max_divergence(), scale) >> name


dp.register(make_select_grouping_cols)


def frame_domain() -> dp.Domain:
    """A domain for a Polars DataFrame: the library has none, so membership is a Python check."""

    def member(value: object) -> bool:
        return isinstance(value, pl.DataFrame)

    return dp.user_domain("PolarsDataFrameDomain", member=member)


# end excerpt


def monotonic_map(candidates: list[tuple[str, ...]], scale: float) -> float:
    """The map the documentation's monotonic metric would give, for comparison."""

    def function(frame: pl.DataFrame) -> list[float]:
        return [score(frame, columns, MIN_GROUP) for columns in candidates]

    def stability_map(d_in: int) -> float:
        return float(d_in)

    scores = dp.t.make_user_transformation(
        input_domain=frame_domain(),
        input_metric=dp.symmetric_distance(),
        output_domain=dp.vector_domain(dp.atom_domain(T=float, nan=False)),
        output_metric=dp.linf_distance(T=float, monotonic=True),
        function=function,
        stability_map=stability_map,
    )
    return float((scores >> dp.m.then_noisy_max(dp.max_divergence(), scale)).map(CONTRIBUTIONS))


def columns_of(candidate: tuple[str, ...]) -> str:
    return ", ".join(candidate)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seed", type=int, default=0, help="seed of the toy regression (OpenDP's noise has none)")
    args = parser.parse_args()
    seed: int = args.seed
    released: dict[str, Any] = {}

    # Section 1: trust. The constructors need the flag; without it they refuse.
    print(f"opendp version: {dp.__version__}")
    dp.disable_features("honest-but-curious")
    try:
        make_base_constant("denied")
        refusal = "built"
    except dp.OpenDPException as err:
        refusal = first_line(err)
    print(f"without the flag: {refusal}")
    dp.enable_features("honest-but-curious")

    # Section 2: a measurement plugin. The docs' constant, then a Laplace mechanism in Python.
    constant = make_base_constant("denied")
    print(f"constant: input={constant.input_domain} metric={constant.input_metric} measure={constant.output_measure}")
    space = dp.vector_domain(dp.atom_domain(bounds=(0, 10))), dp.symmetric_distance()
    sum_then_constant = space >> dp.t.then_sum() >> make_base_constant("denied")
    print(f"sum >> constant: map(1)={sum_then_constant.map(1)} release={sum_then_constant([2, 3, 4])}")
    float_space = dp.atom_domain(T=float, nan=False), dp.absolute_distance(T=float)
    numpy_laplace = make_numpy_laplace(*float_space, scale=2.0)
    built_in = float_space >> dp.m.then_laplace(scale=2.0)
    print(f"python laplace scale 2: map(1.0)={numpy_laplace.map(1.0)} built-in map(1.0)={built_in.map(1.0)}")
    halved = make_halved_laplace(scale=2.0)
    print(f"halved map, same noise: map(1.0)={halved.map(1.0)} check(1.0, 0.25)={halved.check(1.0, 0.25)}")
    laplace_accuracy = dp.laplacian_scale_to_accuracy(scale=2.0, alpha=0.05)
    print(f"laplace scale 2: 95% accuracy={laplace_accuracy:.4f}")
    released["python laplace of 10.0"] = numpy_laplace(10.0)

    # Section 3: a transformation plugin. The docs' repeat, chained with built-in steps.
    repeat = make_repeat(2)
    print(f"repeat 2: [1, 2] -> {repeat([1, 2])} map(1)={repeat.map(1)} map(3)={repeat.map(3)}")
    summed = repeated_sum()
    print(
        f"cast >> repeat 2 >> clamp (1, 2) >> sum: map(1)={summed.map(1)} on ['0', '1', '2', '3'] = {summed(['0', '1', '2', '3'])}"
    )
    noisy = summed >> dp.m.then_laplace(scale=1.0)
    print(f"... >> laplace scale 1: map(1)={noisy.map(1)}")
    print(f"domain mismatch: {mismatched_chain()}")
    released["repeated sum"] = noisy(["0", "1", "2", "3"])

    # Section 5: registering a constructor. The partial-application rule and the naming rule.
    context = dp.Context.compositor(
        data=[1, 2, 3],
        privacy_unit=dp.unit_of(contributions=CONTRIBUTIONS),
        privacy_loss=dp.loss_of(epsilon=EPSILON),
        split_evenly_over=2,
    )
    query = context.query().anything_constant("denied")
    print(f"anything_constant: map(36)={query.resolve().map(CONTRIBUTIONS)} release={query.release()}")
    print(f"budget spent by the plugin query: {[round(d, 4) for d in context.current_privacy_loss()]}")
    dp.register(make_int_constant)
    print(
        f"clamp >> sum >> int_constant: release={context.query().clamp((0, 5)).sum().int_constant('denied').release()}"
    )
    try:
        dp.register(make_anything_constant)
        again = "registered"
    except ValueError as err:
        again = str(err)
    print(f"register twice: {again}")
    try:
        dp.register(select_cols)
        badly_named = "registered"
    except ValueError as err:
        badly_named = str(err)
    print(f"register without make_: {badly_named}")

    # Section 6: the Theil-Sen regression on the documentation's toy data.
    x_cuts = theil_sen_cuts()
    candidates = np.linspace(*Y_BOUNDS, 100)
    print(
        f"theil-sen: x cuts {x_cuts[0]} and {x_cuts[1]}, {len(candidates)} candidates from {Y_BOUNDS[0]} to {Y_BOUNDS[1]} step {candidates[1] - candidates[0]:.4f}"
    )
    theil_sen = make_private_theil_sen(dp.max_divergence(), X_BOUNDS, Y_BOUNDS, scale=1.0)
    pairwise = make_pairwise_predict(x_cuts)
    print(
        f"pairwise predict: map(1)={pairwise.map(1)} medians: map(1)={make_private_medians(dp.max_divergence(), Y_BOUNDS, 1.0).map(1)} theil-sen: map(1)={theil_sen.map(1)} map(2)={theil_sen.map(2)}"
    )
    data = toy_regression(seed)
    ls_slope, ls_intercept = np.polyfit(data[:, 0], data[:, 1], 1)
    print(f"toy data: n={N_POINTS} least-squares slope={ls_slope:.4f} intercept={ls_intercept:.4f}")
    np.random.seed(seed)
    predictions = pairwise_predict(data, x_cuts)
    true_medians = np.median(predictions, axis=0)
    print(f"pairs: {len(predictions)} medians at the cuts {true_medians[0]:.4f} and {true_medians[1]:.4f}")
    selection_bound = 2.0 * math.log((len(candidates) - 1) / (2 * BETA))  # noisy max at scale 2 over 99 others
    print(f"selection bound: {selection_bound:.1f} score units at noise scale 2.0")
    ranges: list[tuple[float, float]] = []
    for j, cut in enumerate(x_cuts):
        best, best_score, n_near, low, high = candidates_within(list(predictions[:, j]), candidates, selection_bound)
        ranges.append((low, high))
        print(
            f"cut {cut}: best candidate {best:.4f} score {best_score:.0f}; within the bound: {low:.4f} to {high:.4f} ({n_near} candidates)"
        )
    np.random.seed(seed)
    slope, intercept = theil_sen(data)
    released["theil-sen slope, intercept"] = (slope, intercept)
    released["theil-sen medians at the cuts"] = (slope * x_cuts[0] + intercept, slope * x_cuts[1] + intercept)

    # Section 7: choosing grouping columns on the LFS.
    frame = pl.scan_parquet(practitioner_datasets.france_lfs()).collect()
    sets = powerset(CANDIDATE_COLUMNS)
    print(f"LFS: {frame.height:,} rows; candidate columns {', '.join(CANDIDATE_COLUMNS)}; {len(sets)} candidate sets")
    distinct = " ".join(f"{c} {frame[c].n_unique():,}" for c in CANDIDATE_COLUMNS)
    print(f"distinct values: {distinct}")
    scores = [score(frame, columns, MIN_GROUP) for columns in sets]
    order = sorted(range(len(sets)), key=lambda i: -scores[i])
    print(f"score: groups of at least {MIN_GROUP} rows")
    for rank, i in enumerate(order[:3], start=1):
        print(
            f"rank {rank}: {columns_of(sets[i])} score {scores[i]:.0f} of {frame.group_by(list(sets[i])).len().height:,} groups"
        )
    with_pident = [scores[i] for i, columns in enumerate(sets) if "PIDENT" in columns]
    print(f"sets with PIDENT: {len(with_pident)}, largest score {max(with_pident):.0f}")
    scale = 2 * CONTRIBUTIONS / EPSILON
    select = make_select_grouping_cols(frame_domain(), dp.symmetric_distance(), sets, MIN_GROUP, scale)
    print(
        f"select scale {scale:.0f}: map(36)={select.map(CONTRIBUTIONS)} monotonic would be {monotonic_map(sets, scale)}"
    )
    lfs_context = dp.Context.compositor(
        data=frame,
        privacy_unit=dp.unit_of(contributions=CONTRIBUTIONS),
        privacy_loss=dp.loss_of(epsilon=EPSILON),
        split_evenly_over=1,
        domain=frame_domain(),
    )
    lfs_query = lfs_context.query().select_grouping_cols(sets, MIN_GROUP)
    print(f"context query: param scale={lfs_query.param()}")
    grouping_bound = scale * math.log((len(sets) - 1) / (2 * BETA))
    near = [i for i in order if scores[i] >= scores[order[0]] - grouping_bound]
    print(f"within {grouping_bound:.0f} of the best: {len(near)} candidates")
    for i in near:
        print(f"  candidate: {columns_of(sets[i])} score {scores[i]:.0f}")
    released["grouping columns"] = columns_of(lfs_query.release())

    print("released (one run):")
    print(
        f"  python laplace of 10.0: {released['python laplace of 10.0']:.4f}  (within {laplace_accuracy:.1f} with probability 0.95)"
    )
    print(f"  repeated sum of ['0', '1', '2', '3'] with laplace scale 1: {released['repeated sum']}  (true 12)")
    ts_slope, ts_intercept = released["theil-sen slope, intercept"]
    m0, m1 = released["theil-sen medians at the cuts"]
    print(
        f"  theil-sen: slope {ts_slope:.4f} intercept {ts_intercept:.4f}  (medians at the cuts {m0:.4f} and {m1:.4f})"
    )
    print(f"  grouping columns: {released['grouping columns']}")


if __name__ == "__main__":
    main()
