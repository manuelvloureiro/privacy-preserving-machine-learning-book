"""Statistical modelling with OpenDP: the chapter "Statistical modelling" of the OpenDP book (Part 4).

    uv run python code/part04-extending/opendp_statistical_modelling.py

Follows the documentation's Statistical Modeling pages: the differentially private linear regression
(`dp.sklearn.linear_model.LinearRegression`, a Theil-Sen estimator built from pairwise slopes and two private
medians) on the documentation's generated points and on the Labour Force Survey, the differentially private PCA
(`dp.sklearn.decomposition.PCA`, after Amin et al., 2019) on the documentation's generated table, the NumPy domain
and clamp underneath both, and the feature flags each estimator needs. It prints every number the page quotes, on
labelled lines, in the order of the page's sections.

Data. The regression's toy is the documentation's: 100 points with x standard normal and y = 2x + 1 plus normal noise
of standard deviation 0.5, drawn by NumPy's generator at the given seed. The survey fit uses the France LFS sample
(code/practitioner_datasets.py, cached in data/): the people working in the fourth quarter of 2013 whose usual and
actual weekly hours are recorded, one row per person, so the privacy unit of the fit is one person. The PCA's table is
the documentation's: 10,000 rows of 4 columns from a multivariate normal with a random covariance, centred, drawn by
the same seeded generator. Nothing in the toys is sensitive and the survey is public, so the exact least-squares fits,
median lines and decompositions are printed and compared with the releases.

What is random. The seed fixes the toys and the pairing of rows inside the Theil-Sen estimator, which uses NumPy's
global generator. OpenDP's private medians and the PCA's noise come from a cryptographic generator that cannot be
seeded, so the released fits change from run to run; they are printed under "(one run)" and the test checks them
within the bounds the page states. Privacy maps, bounds, cut points, grid steps and the exact fits are deterministic.

Library versions: opendp 0.16.0, scikit-learn 1.8.0, numpy 2.5.4.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import opendp.prelude as dp
import polars as pl
from numpy.typing import NDArray

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import practitioner_datasets

dp.enable_features("contrib", "honest-but-curious", "idealized-numerics")

TOY_N = 100
TOY_X_BOUNDS = (-3.0, 3.0)  # the documentation's bounds for its generated points
TOY_Y_BOUNDS = (-10.0, 10.0)
HOURS_BOUNDS = (0.0, 98.0)  # public: the survey codes weekly hours from 0 to 98; 99 means "not applicable"
REGRESSION_SCALE = 2.0  # two private medians at scale 2 cost epsilon = 1 in all
PCA_ROWS = 10_000
PCA_COLUMNS = 4
PCA_EPSILON = 1.0
ROW_NORM = 1.0  # the documentation's row norm, which clamps almost every generated row
LOOSE_ROW_NORM = 40.0  # a row norm above every generated row, which clamps nothing

Array = NDArray[np.float64]


# ---------------------------------------------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------------------------------------------


def toy_points(seed: int) -> Array:
    """The documentation's regression toy: 100 rows of (x, y) with y = 2x + 1 plus noise of standard deviation 0.5."""
    rng = np.random.default_rng(seed)
    x = rng.normal(size=TOY_N)
    y = 2.0 * x + 1.0 + rng.normal(scale=0.5, size=TOY_N)
    return np.stack([x, y], axis=1)


def survey_hours() -> Array:
    """Usual and actual weekly hours of the people working in 2013 Q4, one row per person, hours coded 0 to 98."""
    frame = pl.scan_parquet(practitioner_datasets.france_lfs())
    rows = (
        frame.filter(pl.col("YEAR") == 2013, pl.col("QUARTER") == 4, pl.col("ILOSTAT") == 1)
        .filter(pl.col("HWUSUAL").is_between(0, 98), pl.col("HWACTUAL").is_between(0, 98))  # drops nulls and 99
        .unique(subset="PIDENT", keep="first", maintain_order=True)  # one row per person: the privacy unit
        .select("HWUSUAL", "HWACTUAL")
        .collect()
    )
    return rows.to_numpy().astype(np.float64)


def pca_table(seed: int) -> Array:
    """The documentation's PCA toy: 10,000 centred rows of 4 columns from a normal with a random covariance."""
    rng = np.random.default_rng(seed)
    a = rng.uniform(0, PCA_COLUMNS, size=(PCA_COLUMNS, PCA_COLUMNS))
    table = rng.multivariate_normal(np.zeros(PCA_COLUMNS), a.T @ a, size=PCA_ROWS)
    return table - table.mean(axis=0)


# ---------------------------------------------------------------------------------------------------------------
# Linear regression
# ---------------------------------------------------------------------------------------------------------------


# excerpt: model-regression
def make_regression(
    x_bounds: tuple[float, float], y_bounds: tuple[float, float], scale: float, runs: int = 1
) -> dp.sklearn.linear_model.LinearRegression:
    """The private Theil-Sen regression under pure differential privacy, with public bounds on x and y."""
    return dp.sklearn.linear_model.LinearRegression(
        dp.max_divergence(),  # the output measure: the privacy loss is read in epsilon
        x_bounds=[x_bounds],  # fixes the two cut points, a quarter and three quarters of the way across
        y_bounds=y_bounds,  # the range over which the candidate medians are spread
        scale=scale,  # the noise scale of each of the two private medians
        runs=runs,  # how many random pairings to pool; each multiplies the privacy loss
    )


# end excerpt


def least_squares(points: Array) -> tuple[float, float]:
    """The exact least-squares slope and intercept of y on x."""
    slope, intercept = np.polyfit(points[:, 0], points[:, 1], 1)
    return float(slope), float(intercept)


def cut_points(x_bounds: tuple[float, float]) -> Array:
    """Where the estimator predicts y: a quarter and three quarters of the way across the x bounds."""
    return x_bounds[0] + (x_bounds[1] - x_bounds[0]) * np.array([0.25, 0.75])


def median_line(points: Array, x_bounds: tuple[float, float], seed: int) -> tuple[float, float, Array, Array]:
    """The line the estimator targets, without noise: the medians of the pairwise predictions at the two cut points.

    Pairs the rows after a shuffle with the given seed, as the library does, fits the line through each pair, reads
    it at the two cut points, takes the exact median at each, and solves the two medians for a slope and intercept.
    Returns the slope, the intercept, the two medians and the readings (one row per pair, one column per cut point).
    """
    rng = np.random.default_rng(seed)
    data = points[: len(points) // 2 * 2].copy()
    rng.shuffle(data)
    first, second = np.array_split(data, 2)
    dx, dy = (second - first).T
    x_mid, y_mid = (first + second).T / 2
    cuts = cut_points(x_bounds)
    with np.errstate(divide="ignore", invalid="ignore"):
        predictions = (dy / dx * (cuts[None].T - x_mid) + y_mid).T[dx != 0]  # a pair with equal x has no slope
    medians = np.median(predictions, axis=0)
    slope, intercept = np.linalg.solve(np.vstack([cuts, np.ones_like(cuts)]).T, medians)
    return float(slope), float(intercept), medians, predictions


def fit_regression(model: dp.sklearn.linear_model.LinearRegression, points: Array, seed: int) -> tuple[float, float]:
    """Release a slope and an intercept; the pairing is seeded, the two private medians are not."""
    np.random.seed(seed)  # the estimator pairs rows with NumPy's global generator
    with np.errstate(divide="ignore", invalid="ignore"):  # pairs with equal x are dropped, but NumPy warns first
        fit = model.fit(points[:, :1], points[:, 1])
    return float(fit.coef_[0]), float(fit.intercept_)


# ---------------------------------------------------------------------------------------------------------------
# Principal component analysis
# ---------------------------------------------------------------------------------------------------------------


# excerpt: model-pca
def make_pca(epsilon: float, row_norm: float, n_components: int | None = None) -> dp.sklearn.decomposition.PCA:
    """The private PCA of the generated table, with the two public facts the algorithm needs."""
    return dp.sklearn.decomposition.PCA(
        epsilon=epsilon,
        row_norm=row_norm,  # public: every row is scaled into a ball of this radius before anything else
        n_samples=PCA_ROWS,  # public: the exact number of rows, which fixes the domain
        n_features=PCA_COLUMNS,
        n_components=n_components,  # None releases every component
    )


# end excerpt


def clamp_rows(table: Array, norm: float) -> Array:
    """Scale every row whose length exceeds the norm down onto the ball of that radius, as the library does."""
    lengths = np.linalg.norm(table, axis=1, keepdims=True)
    return table / np.maximum(lengths / norm, 1.0)


def exact_pca(table: Array) -> tuple[Array, Array]:
    """NumPy's singular values and components of the centred table."""
    _, singular_values, components = np.linalg.svd(table - table.mean(axis=0), full_matrices=False)
    return singular_values, components


def cosines(components: Array, exact: Array) -> Array:
    """The absolute cosine between each released component and the exact one; 1 means the same direction."""
    return np.abs(np.einsum("ij,ij->i", components, exact[: len(components)]))


def fmt(values: Array, digits: int = 2) -> str:
    return "[" + ", ".join(f"{v:.{digits}f}" for v in values) + "]"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seed", type=int, default=0, help="seed of the toys and of the pairing (OpenDP has none)")
    args = parser.parse_args()
    seed: int = args.seed
    print(f"opendp version: {dp.__version__}")

    # Section 2: linear regression. The estimator, its privacy map and the toy fit.
    toy = toy_points(seed)
    model = make_regression(TOY_X_BOUNDS, TOY_Y_BOUNDS, scale=1.0)
    measurement = model.measurement
    print(f"regression space: {measurement.input_domain} {measurement.input_metric} {measurement.output_measure}")
    for scale, runs in ((1.0, 1), (2.0, 1), (2.0, 3)):
        epsilon = make_regression(TOY_X_BOUNDS, TOY_Y_BOUNDS, scale=scale, runs=runs).measurement.map(1)
        print(f"regression map: scale={scale:g} runs={runs} -> epsilon={epsilon:.1f}")
    step = np.diff(np.linspace(*TOY_Y_BOUNDS, 100))[0]
    print(f"toy: n={TOY_N} cut points={fmt(cut_points(TOY_X_BOUNDS), 1)} candidate step={step:.4f}")
    slope, intercept = least_squares(toy)
    print(f"toy least squares: slope={slope:.4f} intercept={intercept:.4f}")
    slope, intercept, _, readings = median_line(toy, TOY_X_BOUNDS, seed)
    print(f"toy median line, no noise: slope={slope:.4f} intercept={intercept:.4f} readings={len(readings)}")
    toy_slope, toy_intercept = fit_regression(model, toy, seed)

    # Section 2, the survey: actual against usual hours for the people working in 2013 Q4.
    hours = survey_hours()
    print(f"survey: working people in 2013 Q4 with hours recorded: {len(hours):,} (one row each)")
    print(f"survey: of them with 0 actual hours that week: {int((hours[:, 1] == 0).sum()):,}")
    step = np.diff(np.linspace(*HOURS_BOUNDS, 100))[0]
    print(f"survey: bounds={HOURS_BOUNDS} cut points={fmt(cut_points(HOURS_BOUNDS), 1)} candidate step={step:.4f}")
    slope, intercept = least_squares(hours)
    print(f"survey least squares: slope={slope:.4f} intercept={intercept:.4f}")
    slope, intercept, medians, readings = median_line(hours, HOURS_BOUNDS, seed)
    print(f"survey median line, no noise: slope={slope:.4f} intercept={intercept:.4f} medians={fmt(medians, 2)}")
    near = [int((np.abs(readings[:, j] - medians[j]) <= step).sum()) for j in range(2)]
    print(
        f"survey readings: {len(readings):,} pairs; within one candidate step of the medians: {near[0]:,} and {near[1]:,}"
    )
    # excerpt: model-fit
    model = make_regression(HOURS_BOUNDS, HOURS_BOUNDS, scale=REGRESSION_SCALE)
    epsilon = model.measurement.map(1)  # the privacy loss for adding or removing one person's row
    survey_slope, survey_intercept = fit_regression(model, hours, seed)
    # end excerpt
    print(f"survey private fit: scale={REGRESSION_SCALE:g} epsilon={epsilon:.1f}")

    # Section 3: PCA. The table, the exact decompositions, the model, its maps, and the fits.
    table = pca_table(seed)
    lengths = np.linalg.norm(table, axis=1)
    print(
        f"pca table: {PCA_ROWS:,} rows x {PCA_COLUMNS} columns, row length mean={lengths.mean():.2f} max={lengths.max():.2f}"
    )
    print(
        f"pca table: rows longer than {ROW_NORM:g}: {(lengths > ROW_NORM).mean():.1%}  longer than {LOOSE_ROW_NORM:g}: {(lengths > LOOSE_ROW_NORM).mean():.1%}"
    )
    raw_values, _ = exact_pca(table)
    print(f"pca exact, raw rows: singular values={fmt(raw_values, 1)}")
    exact_values, exact_components = exact_pca(clamp_rows(table, ROW_NORM))
    print(f"pca exact, rows clamped to {ROW_NORM:g}: singular values={fmt(exact_values, 2)}")
    print(f"pca exact, rows clamped to {ROW_NORM:g}: first component={fmt(exact_components[0], 3)}")
    # excerpt: model-pca-fit
    pca = make_pca(PCA_EPSILON, ROW_NORM)
    measurement = pca.measurement()  # the measurement behind fit(), for its map and for compositors
    pca.fit(table)  # the private release: the mean, the singular values and the components
    # end excerpt
    print(f"pca model: {pca}")
    print(f"pca space: {measurement.input_domain} {measurement.input_metric} {measurement.output_measure}")
    print(
        f"pca map: map(2)={measurement.map(2):.1f} map(1)={measurement.map(1):.1f} check(2, 1.0)={measurement.check(2, 1.0)} check(2, 2.0)={measurement.check(2, 2.0)}"
    )
    rows = dp.numpy.array2_domain(num_columns=PCA_COLUMNS, size=PCA_ROWS, T=float)
    plain = dp.sklearn.decomposition.make_private_pca(rows, dp.symmetric_distance(), PCA_EPSILON, norm=ROW_NORM)
    print(f"make_private_pca unit_epsilon={PCA_EPSILON:g}: map(2)={plain.map(2):.1f} map(1)={plain.map(1):.1f}")
    two = make_pca(PCA_EPSILON, ROW_NORM, n_components=2)
    two_measurement = two.measurement()
    print(f"pca n_components=2: map(2)={two_measurement.map(2):.1f}")
    both = dp.c.make_composition([make_pca(PCA_EPSILON, ROW_NORM).measurement(), make_pca(0.5, ROW_NORM).measurement()])
    print(f"pca composition of epsilon 1 and 0.5: map(2)={both.map(2):.1f}")
    loose = make_pca(PCA_EPSILON, LOOSE_ROW_NORM)
    loose.fit(table)
    returned = two_measurement(table)  # fits `two` and returns it

    # Section 4: the NumPy domain and clamp underneath, and a PCA on a domain that is already bounded.
    bounded = dp.numpy.array2_domain(norm=ROW_NORM, p=2, origin=0.0, num_columns=PCA_COLUMNS, size=PCA_ROWS, T=float)
    print(f"bounded domain: {bounded}")
    split = dp.sklearn.decomposition.PCAEpsilons(eigvals=0.5, eigvecs=[1 / 6] * 3, mean=None)
    on_bounded = dp.sklearn.decomposition.make_private_pca(bounded, dp.symmetric_distance(), split)
    print(
        f"pca on the bounded domain: map(1)={on_bounded.map(1):.2f} map(2)={on_bounded.map(2):.2f} map(4)={on_bounded.map(4):.2f}"
    )

    # Section 5: what a reviewer should ask. The feature flags, and the size the library does not check.
    dp.disable_features("honest-but-curious")
    try:
        make_regression(TOY_X_BOUNDS, TOY_Y_BOUNDS, scale=1.0)
        flags = "constructed"
    except dp.OpenDPException as err:
        flags = str(err).strip().split(". See")[0]
    print(f"regression without honest-but-curious: {flags}")
    try:
        make_pca(PCA_EPSILON, ROW_NORM).fit(table)
        flags = "fitted"
    except dp.OpenDPException as err:
        flags = str(err).strip().split(". See")[0]
    print(f"pca without honest-but-curious: {flags}")
    dp.enable_features("honest-but-curious")
    dp.disable_features("idealized-numerics")
    try:
        make_pca(PCA_EPSILON, ROW_NORM).fit(table)
        flags = "fitted"
    except dp.OpenDPException as err:
        flags = str(err).strip().split(". See")[0]
    print(f"pca without idealized-numerics: {flags}")
    try:
        make_regression(TOY_X_BOUNDS, TOY_Y_BOUNDS, scale=1.0)
        flags = "constructed"
    except dp.OpenDPException as err:
        flags = str(err).strip().split(". See")[0]
    print(f"regression without idealized-numerics: {flags}")
    dp.enable_features("idealized-numerics")
    wrong_size = dp.sklearn.decomposition.PCA(
        epsilon=PCA_EPSILON, row_norm=ROW_NORM, n_samples=PCA_ROWS - 1, n_features=PCA_COLUMNS
    )
    try:
        wrong_size.fit(table)
        checked = "fitted without complaint"
    except Exception as err:  # noqa: BLE001
        checked = f"refused: {type(err).__name__}"
    print(f"pca with n_samples one short of the table: {checked}")

    print("released (one run):")
    print(f"  toy private fit: slope={toy_slope:.4f} intercept={toy_intercept:.4f}")
    print(f"  survey private fit: slope={survey_slope:.4f} intercept={survey_intercept:.4f}")
    print(f"  survey private fit at 35 usual hours: {survey_slope * 35 + survey_intercept:.2f} actual")
    print(f"  pca row_norm={ROW_NORM:g}: singular values={fmt(pca.singular_values_, 2)}")
    print(
        f"  pca row_norm={ROW_NORM:g}: |cos| with exact components={fmt(cosines(pca.components_, exact_components), 3)}"
    )
    print(
        f"  pca row_norm={ROW_NORM:g}: mean={fmt(pca.mean_, 3)}  explained variance ratio={fmt(pca.explained_variance_ratio_, 3)}"
    )
    loose_values, loose_components = exact_pca(clamp_rows(table, LOOSE_ROW_NORM))
    print(
        f"  pca row_norm={LOOSE_ROW_NORM:g}: singular values={fmt(loose.singular_values_, 1)} (exact {fmt(loose_values, 1)})"
    )
    print(
        f"  pca row_norm={LOOSE_ROW_NORM:g}: |cos| with exact components={fmt(cosines(loose.components_, loose_components), 3)}"
    )
    print(
        f"  pca n_components=2: returned the model: {returned is two}  singular values={fmt(two.singular_values_, 2)}"
    )
    print(f"  pca n_components=2: |cos| with exact components={fmt(cosines(two.components_, exact_components), 3)}")


if __name__ == "__main__":
    main()
