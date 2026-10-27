# Statistical modelling

Every release so far has been a table of statistics, such as a count, a sum, a mean or a quantile per group. Often the question is about a relation instead, such as how many hours people actually work for each hour they usually work, or which directions of a table carry its variance. OpenDP ships two models for such questions, a differentially private linear regression and a differentially private principal component analysis (PCA), both with interfaces like those of scikit-learn. The one idea of this chapter is that a fitted model is a release like any other. Its cost is the privacy map of the measurement behind `fit`, not the argument named `epsilon`.

You will fit the regression on the documentation's generated points and on the Labour Force Survey (LFS), read its privacy loss, and see why its sensitivity is bounded. You will fit the PCA on the documentation's generated table and compare it with NumPy's exact decomposition. Then you will meet the NumPy domain under both estimators, and the questions a reviewer should ask about a private model.

The script [opendp_statistical_modelling.py](../../code/part04-extending/opendp_statistical_modelling.py) runs every step on OpenDP 0.16.0 with scikit-learn 1.8.0. A seeded NumPy generator makes the toy data and fixes how the regression pairs its rows. So the exact fits, the bounds, the cut points and every privacy map are the same on every run. The released fits change, because OpenDP's generator cannot be seeded, and they appear in one block marked as one run.

```bash
uv run python code/part04-extending/opendp_statistical_modelling.py
```

## A model or a table? {#sec-model-choice}

The documentation's page on statistical modelling makes one point before any code. When you know which relation you need, a model made for it spends the whole budget on the few numbers that answer the question. A slope costs two private medians, as the next section shows, while a table fine enough to read the slope from would cost a noisy count in every cell. In the same way, the PCA releases a few directions, not a full covariance matrix.

The flexible alternative is synthetic data, which answers any later question and can go to people who need microdata. The price is accuracy, because a copy that serves every question answers each one less accurately than a direct release under the same budget ([AIM and MST](../part05-synthetic-data/17-aim-and-mst.md)). So release a table when the questions are counts, a model when the question is a relation, and synthetic data when the questions are not yet known.

The two models are narrow on purpose. The regression takes one explanatory column and one response, and the PCA takes a numeric table whose rows have a bounded length. For many explanatory variables, use the methods of [the full edition's chapter on private empirical risk minimization](../../book-full/part05-dp-ml/01-dp-erm.md), which OpenDP does not ship.

## Linear regression {#sec-model-regression}

A least-squares line is a poor thing to release directly. Its slope is a ratio of sums whose sensitivity is unbounded unless you clip both columns, and clipping changes the line. OpenDP ships a **Theil-Sen regression**, which uses medians instead. It pairs the rows at random, fits the line through each pair, and reads each line at two fixed cut points on the x axis. At each cut point it releases the median of those readings with a private quantile. Then it solves the two medians for a slope and an intercept, which is post-processing and costs nothing.

The sensitivity is bounded because of how the private median chooses. It scores every candidate on a fixed grid by how many readings lie on either side of it. Then the exponential mechanism picks a candidate with a probability that favours high scores. One person's row sits in one pair, so it moves one reading and changes the score of any candidate by at most one. How far the reading moved does not matter, so the estimator needs no clipping and survives a person who reports a thousand hours. [The full edition's chapter on selection](../../book-full/part03-dp-foundations/05-selection.md) derives the exponential mechanism and the private quantile, and [its chapter on sensitivity](../../book-full/part03-dp-foundations/07-sensitivity.md) explains why a count of readings is the quantity to bound.

```{.python excerpt="model-regression"}
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
```

The first argument is the output measure, and `dp.max_divergence()` asks for pure differential privacy (DP), whose parameter is the privacy loss $\varepsilon$. The bounds are public and never clip the data. The x bounds only place the two cut points, at the fractions `fraction_bounds` of the way across, a quarter and three quarters by default. The y bounds only fix the grid of `candidates_count` evenly spaced candidates for each median, a hundred by default. The scale is the noise scale of each private median, and `runs` repeats the random pairing and pools the readings. [Plugins](13-plugins.md) builds the same estimator from plugins, as the documentation does.

```text
regression space: NPArray2Domain(num_columns=2, nan=True, T=f64) SymmetricDistance() MaxDivergence
regression map: scale=1 runs=1 -> epsilon=2.0
regression map: scale=2 runs=1 -> epsilon=1.0
regression map: scale=2 runs=3 -> epsilon=3.0
```

The `measurement` attribute of the estimator reads like any other measurement. Its input is a two-column NumPy array under the symmetric distance, so its map at distance 1 is the privacy loss for adding or removing one row. Two medians at scale 1 cost $\varepsilon = 2$, and doubling the scale halves that. Three runs at scale 2 cost $\varepsilon = 3$, because a row then sits in three pairs and moves three readings. More runs make the median better defined, but at a fixed budget they force a larger scale, so they rarely pay on a large dataset.

The documentation's toy data is a hundred points with a true slope of 2 and an intercept of 1, here from a seeded generator. The script prints the exact values to read a fit against.

```text
toy: n=100 cut points=[-1.5, 1.5] candidate step=0.2020
toy least squares: slope=2.0275 intercept=0.9725
toy median line, no noise: slope=2.0379 intercept=1.0786 readings=50
```

The candidate step is the y range divided into 99 intervals, about 0.2. The cut points are 3 apart, so one grid step at one median changes the slope by about 0.07, before any noise. The median line is what the estimator would return with exact medians and the seeded pairing, from the 50 readings of 100 points. It is close to least squares because the noise in the toy data is symmetric.

### Actual hours against usual hours on the survey

The survey question is how many hours people actually worked in the reference week for each hour they usually work. The script takes the people at work in the fourth quarter of 2013 with both hours recorded. The survey codes hours from 0 to 98, so the bounds are public, and it drops the code 99, "not applicable". It also keeps the first row of each person, so that the privacy unit of the fit is one person ([Identifier truncation](../part02-tabular-data/05-identifier-truncation.md)). These steps run in Polars before the array reaches the measurement, so they are your claim and not the library's.

```text
survey: working people in 2013 Q4 with hours recorded: 36,914 (one row each)
survey: of them with 0 actual hours that week: 4,109
survey: bounds=(0.0, 98.0) cut points=[24.5, 73.5] candidate step=0.9899
survey least squares: slope=0.8582 intercept=-0.1935
survey median line, no noise: slope=1.0000 intercept=0.0000 medians=[24.50, 73.50]
survey readings: 16,463 pairs; within one candidate step of the medians: 5,606 and 5,252
```

The two exact fits disagree, and that is the lesson. Of the 36,914 people, 4,109 worked no hours that week, through leave or illness, and they pull least squares down to 0.86 actual hours per usual hour. The median line ignores them, because most pairs of people lie on the line where actual equals usual. So its medians are the cut points themselves, and its slope is exactly 1. A private Theil-Sen fit therefore estimates the typical relation, not the average one, and the median line is the exact value to compare it with. The readings are fewer than half the people, because a pair with equal usual hours has no slope, and thousands of them lie within one grid step of each median.

```{.python excerpt="model-fit"}
model = make_regression(HOURS_BOUNDS, HOURS_BOUNDS, scale=REGRESSION_SCALE)
epsilon = model.measurement.map(1)  # the privacy loss for adding or removing one person's row
survey_slope, survey_intercept = fit_regression(model, hours, seed)
```

```text
survey private fit: scale=2 epsilon=1.0
```

The fit at scale 2 costs $\varepsilon = 1$ for one person. Like scikit-learn, `fit` takes the explanatory column as a two-dimensional array and the response as a vector. It returns a fitted scikit-learn `LinearRegression`, so `coef_`, `intercept_` and `predict` work as usual.

## Principal component analysis {#sec-model-pca}

A PCA finds the directions along which a table varies most. OpenDP's version releases three things under one budget. They are the mean of the rows, the singular values, which say how much variance each direction carries, and the components, which are the directions.

The algorithm, from Amin and others (2019), scales every row into a ball of radius `row_norm` and centres the rows on a private mean. Then it forms the matrix of sums of squares and cross-products. It releases the eigenvalues of that matrix with Laplace noise, because replacing one row moves their sum by at most twice the squared row norm. Then it draws the eigenvectors one at a time with the exponential mechanism on the sphere, which favours directions of high variance, and removes each drawn direction from the matrix.

So two facts must be public. The row norm is the sensitivity of the whole algorithm, and every map below is in units of it. The number of rows is part of the domain, which gives the mean a known divisor. So neighbouring tables have the same size with one row replaced, which is distance 2 under the symmetric distance. The library declares both facts and does not check them, as the reviewer's section shows. The budget is split in three equal parts for the mean, the eigenvalues and the eigenvectors, and the third for the eigenvectors is split again over the draws.

The table is the documentation's, ten thousand centred rows of four columns from a normal distribution with a random covariance. A seeded generator makes the exact decomposition reproducible.

```text
pca table: 10,000 rows x 4 columns, row length mean=8.53 max=36.56
pca table: rows longer than 1: 99.9%  longer than 40: 0.0%
pca exact, raw rows: singular values=[901.9, 278.6, 222.2, 156.9]
pca exact, rows clamped to 1: singular values=[80.48, 40.67, 34.29, 26.29]
pca exact, rows clamped to 1: first component=[-0.614, -0.534, -0.526, -0.247]
```

The documentation fits with `row_norm=1.0`, and the first two lines show what that means here. Almost every row is longer than 1, so the clamp scales almost every row onto the unit sphere. The algorithm can then only estimate the singular values of the clamped table, about 80 instead of about 900 for the first. The directions survive the clamp, because scaling a row does not turn it, so the release is compared with the first component of the clamped table. The script also fits with a row norm of 40, which clamps nothing, to show the other side of the trade.

```{.python excerpt="model-pca"}
def make_pca(epsilon: float, row_norm: float, n_components: int | None = None) -> dp.sklearn.decomposition.PCA:
    """The private PCA of the generated table, with the two public facts the algorithm needs."""
    return dp.sklearn.decomposition.PCA(
        epsilon=epsilon,
        row_norm=row_norm,  # public: every row is scaled into a ball of this radius before anything else
        n_samples=PCA_ROWS,  # public: the exact number of rows, which fixes the domain
        n_features=PCA_COLUMNS,
        n_components=n_components,  # None releases every component
    )
```

```{.python excerpt="model-pca-fit"}
pca = make_pca(PCA_EPSILON, ROW_NORM)
measurement = pca.measurement()  # the measurement behind fit(), for its map and for compositors
pca.fit(table)  # the private release: the mean, the singular values and the components
```

```text
pca model: PCA(epsilon=1.0, n_components=4, n_features=4, n_samples=10000, row_norm=1.0)
pca space: NPArray2Domain(size=10000, num_columns=4, nan=True, T=f64) SymmetricDistance() MaxDivergence
pca map: map(2)=2.0 map(1)=2.0 check(2, 1.0)=False check(2, 2.0)=True
make_private_pca unit_epsilon=1: map(2)=1.0 map(1)=1.0
pca n_components=2: map(2)=2.0
pca composition of epsilon 1 and 0.5: map(2)=3.0
```

Read the third line twice. The model was built with `epsilon=1.0`, but the map of its measurement at distance 2, one replaced row, is 2.0. And `check` confirms that this fit does not stay within $\varepsilon = 1$.

The reason is in the constructor. The `PCA` class calls the framework's `make_private_pca` with twice its `epsilon`, divided by `n_changes`, the number of rows one person may change. The fourth line shows that `make_private_pca` with a unit budget of 1 has a map of 1.0 at distance 2. So in version 0.16.0 a `PCA(epsilon=1.0)` fit costs $\varepsilon = 2$ per changed row by its own map, and the map is what you should quote. The map is flat below distance 2, because the algorithm is an adaptive composition declared at that distance ([Combinators](../part03-framework/11-combinators.md)).

The fifth line shows that two components leave the total unchanged, and only the split inside changes. The library draws one eigenvector fewer than there are columns, because the others fix the last direction. So the third for the eigenvectors is split over three draws for four components and over two draws for two, and each draw gets more. The `measurement()` method is there for the compositors. The last line composes a fit at `epsilon=1.0` with one at `epsilon=0.5`, and the maps add to 3.0 at distance 2.

## The NumPy domain underneath {#sec-model-numpy}

Both estimators use `dp.numpy.array2_domain`, the domain of two-dimensional NumPy arrays, described by a number of columns, an optional number of rows and the element type. It can also have a norm, its order `p` and an origin, which say that every row lies within that norm of the origin. A domain with a norm is a promise about the data, and `make_np_clamp` is the transformation that makes the promise true.

```python
import numpy as np
import opendp.prelude as dp

dp.enable_features("contrib", "honest-but-curious")
rows = dp.numpy.array2_domain(num_columns=2, T=float)
clamp = dp.numpy.make_np_clamp(rows, dp.symmetric_distance(), norm=1.0, p=2, origin=np.array([1.0, 2.0]))
print(clamp(np.array([[1.0, 2.0], [4.0, 6.0], [1.5, 2.0]])).tolist())  # [[1.0, 2.0], [1.6, 2.8], [1.5, 2.0]]
print(clamp.map(1))  # 1
print(clamp.output_domain)  # NPArray2Domain(origin=[1. 2.], norm=1.0, p=2, num_columns=2, nan=False, T=f64)
```

The clamp leaves the first and third rows alone, because they lie within distance 1 of the origin, and pulls the second row from distance 5 back to the sphere. Its stability map is the identity, because clamping each row on its own neither adds nor removes rows ([Transformations](../part03-framework/09-transformations.md)). Its output domain records the norm, the order, the origin, and that no value is missing.

That output domain is what `make_private_pca` prefers to receive. Such a domain states that the rows are already bounded around a known centre. So the algorithm skips the private mean and its third of the budget.

```text
bounded domain: NPArray2Domain(origin=[0. 0. 0. 0.], norm=1.0, p=2, size=10000, num_columns=4, nan=True, T=f64)
pca on the bounded domain: map(1)=0.50 map(2)=1.00 map(4)=2.00
```

The script passes an explicit split as a `PCAEpsilons`, with half for the eigenvalues, a sixth for each of three eigenvectors and nothing for the mean. The map is then linear in the distance, not flat, because no compositor is declared in between. Use this route when the centre is public, for example for scores on a known scale, and the class when it is not.

## What a reviewer should ask {#sec-model-review}

The first question is the privacy unit. Both estimators count rows, so the claim "one person" holds only if each person has one row, which on the survey was your own Polars step. The second is the bounds. The bounds of the regression and the row norm and row count of the PCA must come from the code book or the domain. A bound read from the data leaks it. The third is which features are on, because a feature is a statement about what you accept on trust.

```text
regression without honest-but-curious: Attempted to use function that requires "contrib", "honest-but-curious", but not enabled
pca without honest-but-curious: Attempted to use function that requires "contrib", "honest-but-curious", but not enabled
pca without idealized-numerics: Attempted to use function that requires "contrib", "idealized-numerics", but not enabled
regression without idealized-numerics: constructed
pca with n_samples one short of the table: fitted without complaint
```

Both estimators need `honest-but-curious`, because they use user-defined transformations whose functions and stability maps the library takes as written. For example, the pairing step of the regression claims that one row moves `runs` readings, and the library records that claim without proof. The PCA also needs `idealized-numerics`, which concedes that its proofs hold for real arithmetic and not for floating point. Every guarantee of the PCA rests on that concession, and the library's own source notes that it knows no floating-point-safe sampler for the eigenvector draw. The regression constructs without the flag, because its private medians run in the Rust core and only the pairing is Python.

The last line is the trap for a reviewer. The number of rows is part of the domain of the PCA, but the library does not check that the array has that many rows. So a wrong `n_samples` is a wrong guarantee that nothing reports. The clamp enforces the row norm, and the bounds of the regression need no enforcement because its sensitivity does not depend on them. [Limitations and troubleshooting](15-limitations-and-troubleshooting.md) collects the other things the library does not check.

## The releases of one run {#sec-model-run}

```text
released (one run):
  toy private fit: slope=1.9529 intercept=1.0101
  survey private fit: slope=1.0101 intercept=-0.9899
  survey private fit at 35 usual hours: 34.36 actual
  pca row_norm=1: singular values=[80.48, 40.66, 34.28, 26.37]
  pca row_norm=1: |cos| with exact components=[1.000, 0.998, 0.996, 0.996]
  pca row_norm=1: mean=[0.001, 0.002, 0.001, 0.001]  explained variance ratio=[0.648, 0.165, 0.117, 0.070]
  pca row_norm=40: singular values=[906.8, 268.6, 212.6, 153.2] (exact [901.9, 278.6, 222.2, 156.9])
  pca row_norm=40: |cos| with exact components=[0.996, 0.659, 0.426, 0.086]
  pca n_components=2: returned the model: True  singular values=[80.41, 40.64]
  pca n_components=2: |cos| with exact components=[0.999, 0.999]
```

The toy fit is about one grid step off the least-squares line at one cut point, which is what a private median on 50 readings at $\varepsilon = 2$ gives. The test allows the slope within 0.3 of the least-squares slope and the intercept within 0.5, about three grid steps at either cut point. A candidate three steps from the best scores lower by every reading in between, and the exponential mechanism makes such a candidate rare at this budget.

The survey fit is the same on every run. The medians sit exactly at the cut points, and the nearest candidates on the grid give a slope of 1.0101 and an intercept of about $-1$. Thousands of readings lie within one grid step of each median, so the next candidate scores thousands worse than the best, and noise at scale 2 never overturns that gap. The error of the fit is therefore the grid, not the noise, and a finer `candidates_count` removes it at no cost in privacy. The test allows the slope within 0.1 of the median line and the intercept within 4 hours, two grid steps at the lower cut point. The fit predicts about 34.4 actual hours for someone who usually works 35.

At row norm 1 the PCA recovers the singular values of the clamped table to within a tenth, and all four directions to a cosine above 0.99. Its mean is within a few thousandths of the true zero. The test allows each singular value within 1 and each cosine above 0.9. Both bounds leave a wide margin, because the Laplace noise on the eigenvalues is a few units against eigenvalues in the thousands, and the four directions are well separated.

At row norm 40 the singular values are those of the raw table, within a few percent, but only the first direction is right. The concentration of the eigenvector sampler falls with the square of the row norm, so the last three directions are close to random. The test checks only the first singular value, within 30, and the first cosine, above 0.9. The two-component fit returns the model itself, and its two directions are as good as the first two of the four-component fit.

## Summary

| Call or concept | What it does | Rule of thumb |
| --- | --- | --- |
| `dp.sklearn.linear_model.LinearRegression(measure, x_bounds, y_bounds, scale, runs)` | A Theil-Sen regression from two private medians | Bounds fix the cut points and the grid; they never clip the data |
| `model.measurement.map(1)` | The privacy loss of the fit per added or removed row | Two medians at scale $s$ cost $2\,\mathrm{runs}/s$; quote this, not the arguments |
| `candidates_count` | The grid the medians are chosen from | On a large dataset the grid, not the noise, is the error; refine it for free |
| The median line against least squares | What the estimator targets | Compare a Theil-Sen fit with the median line; outliers pull least squares only |
| `dp.sklearn.decomposition.PCA(epsilon, row_norm, n_samples, n_features, n_components)` | Releases the mean, the singular values and the components | `row_norm` and `n_samples` are public facts the library does not check |
| `pca.measurement()` | The measurement behind `fit`, for maps and compositors | In 0.16.0 its map at distance 2 is twice `epsilon`; read the map |
| `n_components` | How many directions to draw | Fewer draws give each more budget at the same total |
| `row_norm` | The sensitivity scale of the whole algorithm | Tight: magnitudes are the clamped table's, directions good. Loose: magnitudes right, directions poor |
| `dp.numpy.array2_domain(norm, p, origin)` and `make_np_clamp` | The bounded domain and the transformation that enforces it | A known origin skips the private mean and its third of the budget |
| `honest-but-curious`, `idealized-numerics` | Trust the plugins' maps; accept real-arithmetic proofs | Both estimators need the first; the whole PCA rests on the second |

: The calls and concepts of private modelling, what each does, and the rule of thumb that goes with it. {#tbl-model-summary}

## References

- OpenDP documentation: "Statistical Modeling", "Linear Regression" with the API reference of `opendp.extras.sklearn.linear_model`, "Differentially Private PCA", and "Theil-Sen Regression" in the plugins user guide.
- Theil, "A Rank-Invariant Method of Linear and Polynomial Regression Analysis" (1950), and Sen, "Estimates of the Regression Coefficient Based on Kendall's Tau" (1968): the median-of-slopes estimator.
- Dwork and Lei, "Differential Privacy and Robust Statistics" (2009): why robust estimators such as medians are the natural private ones.
- Amin, Dick, Kulesza, Munoz and Vassilvitskii, "Differentially Private Covariance Estimation" (2019): the eigenvalue and eigenvector mechanisms behind the PCA.
- The full edition's chapters on [selection](../../book-full/part03-dp-foundations/05-selection.md), [sensitivity](../../book-full/part03-dp-foundations/07-sensitivity.md) and [private empirical risk minimization](../../book-full/part05-dp-ml/01-dp-erm.md) derive the results this chapter states.
