# Plugins

Suppose the mechanism you need is not in the library, such as a new estimator from a paper or a score that only makes sense for your data. The framework of the previous part was built for this case. A transformation is any function with a stability map, and a measurement is any function with a privacy map. So you can write both in Python and hand them to the library as **plugins**. From then on they chain, compose and run through a context like the built-in ones.

The one idea of this chapter is that a plugin is exactly as private as its map, and the library cannot check the map. Every built-in constructor comes with a written proof of its map, and for a plugin that proof is yours. So each example below shows its calls and then says in a sentence why its map is right, or where the documentation leaves that open. You will write a measurement, a transformation, and your own domains, metrics and measures, then register a constructor for the Context API. The chapter ends with the documentation's two larger examples, a Theil-Sen regression and a private choice of grouping columns.

The script [opendp_plugins.py](../../code/part04-extending/opendp_plugins.py) runs every step on OpenDP 0.16.0. Its regression runs on the documentation's toy data, a hundred points on a line with normal noise from a seeded generator. Its choice of grouping columns runs on the Labour Force Survey (LFS) sample. Every map, check, error message, score and candidate range on this page is the same on every run. The released values are not, because OpenDP's noise cannot be seeded, so they appear in one block marked as one run at the end.

```bash
uv run python code/part04-extending/opendp_plugins.py
```

## Trust and the honest-but-curious flag {#sec-plug-trust}

The documentation gives two reasons for plugins. Differential privacy is a wide and growing field, so the library cannot ship every mechanism. And an idea is easier to try in Python than in Rust, so a plugin is how a contribution usually starts. Both reasons put a Python function at the heart of a guarantee, and that changes who is responsible for it.

The library can inspect its own constructors, but it cannot read a Python function. So it cannot tell whether your privacy map is right, whether your function has side effects, or whether it raises only when the data make it. The constructors of this chapter therefore need a second feature flag next to `contrib`, named `honest-but-curious`, and the name is the trust model it describes. Whoever writes the plugin is honest, in that the map they write is one they have proved, and the library guards only against curiosity from everyone downstream. Without the flag a plugin constructor refuses to build.

```text
without the flag: Attempted to use function that requires "contrib", "honest-but-curious", but not enabled
```

Each constructor's docstring spells out the promise. Take any two inputs in the domain that are at most `d_in` apart under the input metric. Then the two output distributions must be at most `privacy_map(d_in)` apart under the output measure, the function must have no side effects, and the map must be pure. A wrong map breaks the guarantee silently, because everything downstream believes it. The context charges what the map reports, the composition adds up the maps, and the release goes out. The snippet is the extreme case, a measurement that returns its input and claims a privacy loss of zero.

```python
import opendp.prelude as dp

dp.enable_features("contrib", "honest-but-curious")
space = dp.vector_domain(dp.atom_domain(T=int)), dp.symmetric_distance()
leak = dp.m.make_user_measurement(*space, dp.max_divergence(), function=lambda x: x, privacy_map=lambda d: 0.0, TO=list[int])
print(leak.map(1))  # 0.0
print(leak([1, 2, 3]))  # [1, 2, 3]
```

The library accepts it, and a context would release the raw data at no charge. Nothing in the rest of this chapter protects you from this. The flag records that the claim rests on your word, and the sentence that justifies each map below is the habit that makes the word worth something.

## A measurement plugin {#sec-plug-measurement}

The documentation's first example is deliberately useless, a measurement that ignores its input and returns a constant. It is useful here because its map is the easiest one to justify. The constructor `make_user_measurement` takes the five things that define a measurement in [A framework to understand differential privacy](../part03-framework/08-a-framework-for-privacy.md). They are an input domain, an input metric, an output measure, the function and the privacy map.

```{.python excerpt="plug-constant"}
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
```

The input domain says the function takes one integer, and the input metric compares two inputs by their absolute difference. The output measure is the max divergence, the measure of pure differential privacy, so the map's value is an $\varepsilon$. The map returns zero at every distance, and that is correct because the output does not depend on the input, so two neighbouring inputs give the same output distribution. The last argument, `TO`, names the type the function returns. The script chains the result after a sum on a vector of bounded integers.

```text
constant: input=AtomDomain(T=i32) metric=AbsoluteDistance(i32) measure=MaxDivergence
sum >> constant: map(1)=0.0 release=denied
```

The chain works because the sum's output space, an integer under the absolute distance, is exactly the plugin's input space. The library checks that match, because domains and metrics are its own objects, but it does not check the map.

A plugin with a real map shows what the justification looks like. The next constructor adds Laplace noise drawn by NumPy rather than by OpenDP. It takes its input space as its first two arguments, which is the convention the Context API will need later.

```{.python excerpt="plug-laplace"}
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
```

The map is the sensitivity over the scale. Laplace noise of a given scale on a quantity that one person moves by at most the sensitivity costs exactly that ratio in $\varepsilon$. [The full edition's chapter on the Laplace and Gaussian mechanisms](../../book-full/part03-dp-foundations/03-laplace-gaussian.md) derives the bound. The built-in `then_laplace` at the same scale reports the same map, which is the check to run on any plugin with a built-in cousin.

```text
python laplace scale 2: map(1.0)=0.5 built-in map(1.0)=0.5
halved map, same noise: map(1.0)=0.25 check(1.0, 0.25)=True
laplace scale 2: 95% accuracy=5.9915
```

The second line is the same noise with a map that claims half the loss. The library accepts it, and `check` confirms a fit within a quarter of a unit of $\varepsilon$, which is false. A context would hand such a plugin a quarter and spend half. The third line is the error bar at scale 2, which the released value at the end of the page is read against. In production, prefer the Rust core, because NumPy's float sampling leaks a little through rounding and OpenDP's own samplers avoid that.

## A transformation plugin {#sec-plug-transformation}

A transformation plugin takes one more argument than a measurement, an output domain. Its map is a stability map, which bounds how far apart two outputs can be when the inputs are a given distance apart, as [Transformations](../part03-framework/09-transformations.md) introduced. The documentation's example repeats each record a fixed number of times.

```{.python excerpt="plug-repeat"}
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
```

Both metrics are the symmetric distance, the number of records added or removed between two datasets. The map multiplies that distance by the multiplicity. Every input record becomes `multiplicity` output records, so adding or removing one input record adds or removes that many output records. The output domain is a second claim you make. Every output must belong to it, or the function must fail in a way that does not depend on the data. Here the output is a vector of integers whenever the input is.

The documentation chains the plugin between built-in steps, a cast from strings and a clamp before a sum, and ends with Laplace noise.

```text
repeat 2: [1, 2] -> [1, 2, 1, 2] map(1)=2 map(3)=6
cast >> repeat 2 >> clamp (1, 2) >> sum: map(1)=4 on ['0', '1', '2', '3'] = 12
... >> laplace scale 1: map(1)=4.0
domain mismatch: DomainMismatch("Intermediate domains don't match
```

The stability of a chain is the product of the stabilities of its links, as [Combinators](../part03-framework/11-combinators.md) explains. The cast and the clamp have stability 1, and the repeat has 2. The clamp bounds every value by 2, so one added input record, repeated twice, moves the sum by at most 4. Laplace noise of scale 1 then costs that much $\varepsilon$. The last line is what happens when the cast produces floats instead. The plugin's input domain is a vector of integers, and the library refuses the chain, because the spaces are its objects and it can compare them.

It cannot compare the function with the output domain, and that is the second way a transformation plugin goes wrong. The snippet declares that its output lies between 0 and 1 and then returns its input unchanged.

```python
import opendp.prelude as dp

dp.enable_features("contrib", "honest-but-curious")
ints = dp.vector_domain(dp.atom_domain(T=int))
bounded = dp.vector_domain(dp.atom_domain(bounds=(0, 1)))
claim = dp.t.make_user_transformation(ints, dp.symmetric_distance(), bounded, dp.symmetric_distance(), function=lambda x: x, stability_map=lambda d: d)
print(claim([5, 6]))  # [5, 6]
print((claim >> dp.t.then_sum()).map(1))  # 1
```

The sum after it trusts the bounds and reports a sensitivity of 1, while the real sum moves by the whole value of a record. A wrong output domain is a wrong map one step later, so a transformation plugin needs two justifications, one for each.

## Your own domains, metrics and measures {#sec-plug-spaces}

Sometimes a plugin works on data the library has no domain for, such as a data frame or a graph, or compares neighbours by a metric it does not define. Three constructors let you name your own, and each carries its own promise.

A **user domain** is built from an identifier and a membership function. Two user domains are equal when their identifiers are equal, which is how the library decides whether two steps chain. So the identifier must name one set and nothing else, or two different domains would chain and break the stability of the chain. The membership function must return true only for members. A **user distance** is a metric that is nothing but a name, and giving it is a promise that the distance you have in mind is a pseudo-metric. It is zero from a point to itself, never negative and symmetric, and it obeys the triangle inequality. The snippet declares a domain of even integers and an edit distance on strings, and uses the distance in a transformation that measures a string's length.

```python
import opendp.prelude as dp

dp.enable_features("contrib", "honest-but-curious")
even = dp.user_domain("EvenInts", member=lambda x: x % 2 == 0)
print(even.member(4), even.member(3))  # True False
edits = dp.user_distance("EditDistance")
length = dp.t.make_user_transformation(dp.atom_domain(T=str), edits, dp.atom_domain(T=int), dp.absolute_distance(T=int), function=len, stability_map=lambda d: d)
print(length("hello"), length.map(1))  # 5 1
print((length >> dp.m.then_laplace(scale=1.0)).map(1))  # 1.0
```

The stability map of the length is the identity, because one edit changes the length by at most one. The output metric is the absolute distance on integers, which is what the Laplace measurement wants. So the chain closes, and a scale of 1 costs an $\varepsilon$ of 1 per edit.

A **user divergence** is a privacy measure of your own. The one property a measure must have is that post-processing cannot increase it, since the promise that a release stays private under further computation rests on it. [The full edition's chapter on properties](../../book-full/part03-dp-foundations/02-properties.md) proves that for the max divergence. Two more constructors complete the set. The function `new_privacy_profile` turns a curve from $\varepsilon$ to the slack $\delta$ into a privacy profile the library can invert. And `new_function` wraps a Python post-processor, which [Combinators](../part03-framework/11-combinators.md) did behind the scenes for a bare function after `>>`. The post-processor must not read global state or raise on the data, because either would open a channel from the data to the caller that no map accounts for.

```python
import numpy as np
import opendp.prelude as dp

dp.enable_features("contrib", "honest-but-curious")
space = dp.atom_domain(T=float, nan=False), dp.absolute_distance(T=float)
renyi = dp.user_divergence("RenyiDivergence(alpha=2)")
rng = np.random.default_rng()
gaussian = dp.m.make_user_measurement(*space, renyi, function=lambda x: float(x + rng.normal()), privacy_map=lambda d: d**2, TO=float)
print(gaussian.map(1.0), gaussian.output_measure)  # 1.0 RenyiDivergence(alpha=2)
try:
    dp.c.make_composition([gaussian, gaussian])
except dp.OpenDPException as err:
    print(str(err).split(". See")[0].strip())  # Type("No match for concrete type ExtrinsicDivergence
profile = dp.new_privacy_profile(lambda eps: 1.0 if eps < 0.5 else 1e-8)
print(profile.delta(epsilon=0.499), profile.epsilon(delta=1e-8))  # 1.0 0.5
double = dp.new_function(lambda x: 2 * x, TO=float)
laplace = space >> dp.m.then_laplace(scale=1.0)
print((laplace >> double).map(1.0))  # 1.0
```

The measurement adds standard normal noise and reports its loss under the Rényi divergence of order 2. Its map is the squared sensitivity, because the Rényi divergence of order 2 between two unit Gaussians is the square of the distance between their centres. [The full edition's chapter on Rényi differential privacy](../../book-full/part04-privacy-accounting/01-renyi-dp.md) derives that. The map works, but the composition does not, because the combinators add up only the measures the library defines. So a plugin measure gets `map` and `check` and nothing more. The profile answers both ways, the slack at a given $\varepsilon$ and the smallest $\varepsilon$ at a given slack, and its curve must be decreasing. The post-processor leaves the Laplace measurement's map unchanged, which is the point of post-processing.

## Registering a constructor for the Context API {#sec-plug-register}

A plugin built with the framework API is a loose measurement, and the lesson of [A first release](../part01-getting-started/01-first-release.md) was to put every query through a context. The function `dp.register` adds a constructor to the Context API, as a method of `context.query()`. Two rules govern it. The constructor's name must start with `make_`, and the method is the name without that prefix, unless you pass a `name`. And if its first two parameters are named `input_domain` and `input_metric`, the context fills them in. They come from the previous step's output space, or from its own input space for a first step. The documentation calls that partial application, and it is the convention that gives every built-in `make_` constructor its `then_` form.

```{.python excerpt="plug-register"}
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
```

The script builds the documentation's context over three integers, with a privacy unit of 36 contributions and a budget of $\varepsilon = 1$ split over two queries. Then it calls the plugin through it.

```text
anything_constant: map(36)=0.0 release=denied
budget spent by the plugin query: [0.5]
clamp >> sum >> int_constant: release=denied
register twice: 'anything_constant' is already registered in the Context API. Please choose a different name.
register without make_: constructor.__name__ must start with 'make_', found select_cols
```

On the second line the plugin's map is zero, but the context charged the query its full share of the budget. A context splits the budget before it sees a query and only checks that each query fits within its share, as [Utilities and the Context API inside](../part03-framework/12-utilities-and-the-context-api.md) shows. A plugin that costs less than its share does not give the difference back. The third line uses the documentation's other style, `make_int_constant`, without the two leading parameters. It registers too, but its input space is fixed to integers under the absolute distance, so it works after a clamp and a sum but not elsewhere. The documentation recommends the first style, with the caution that a constructor accepting any space should check that the space suits its function. The last two lines are the errors `register` raises, for a name already taken and for a name without the prefix.

## A Theil-Sen regression {#sec-plug-theil-sen}

The documentation's first larger example fits a line through noisy points. Least squares is a poor private estimator, because one far-away point can move the fitted line by any amount. The **Theil-Sen** estimator takes the median of the slopes through pairs of points instead, and a median is a quantile, which OpenDP already releases privately. The documentation's variant pairs the points at random and fits a line through each pair. It records where that line crosses two fixed vertical cuts, at a quarter and three quarters of the range of $x$. The medians of the crossings at each cut are two points on the fitted line, which give the slope and the intercept. The library ships a vetted version of this estimator, which [Statistical modelling](14-statistical-modelling.md) uses, so this one is for learning how plugins combine.

The first plugin is the pairing, a transformation from a two-column array to a two-column array of predictions.

```{.python excerpt="plug-pairwise"}
def pairwise_predict(data: NDArray[np.float64], x_cuts: NDArray[np.float64]) -> NDArray[np.float64]:
    """Pair the rows at random, fit a line through each pair, and predict y at the two cut points."""
    data = np.array(data, copy=True)[: len(data) // 2 * 2]
    np.random.shuffle(data)
    p1, p2 = np.array_split(data, 2)
    dx, dy = (p2 - p1).T
    x_bar, y_bar = (p1 + p2).T / 2
    points = dy / dx * (x_cuts[None].T - x_bar) + y_bar
    return points.T[dx != 0]
```

The transformation wraps that function.

```{.python excerpt="plug-pairwise-transformation"}
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
```

The domains come from the library's NumPy extension, a two-dimensional array with two columns. The documentation states the map, the input distance times the number of runs, without an argument, so here is the one it relies on. Each record sits in exactly one pair, so adding or removing a record changes at most one prediction per run. Two things make that claim weaker than a built-in map. The pairing is random, so the claim is really about a coupling of two runs rather than about a function. And the function drops the last record when the count is odd, so removing one record can also change another's pair. This is the kind of gap the flag asks you to notice.

The second plugin selects one column of the predictions, and the library's quantile does the rest.

```{.python excerpt="plug-select-column"}
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
```

The library's quantile then runs once for each cut.

```{.python excerpt="plug-medians"}
def make_private_medians(output_measure: dp.Measure, y_bounds: tuple[float, float], scale: float) -> dp.Measurement:
    """The private median of the predictions at each cut, by the library's quantile, composed."""
    median = dp.m.then_private_quantile(output_measure, candidates=np.linspace(*y_bounds, 100), alpha=0.5, scale=scale)
    return dp.c.make_composition(
        [
            make_select_column(0) >> dp.t.then_drop_null() >> median,
            make_select_column(1) >> dp.t.then_drop_null() >> median,
        ]
    )
```

Selecting a column keeps one row per row, so its stability is the input distance. The private quantile scores a hundred candidate values between the public bounds of $y$ and picks one by noisy max, as [Measurements](../part03-framework/10-measurements.md) showed. Its map is the input distance over the scale. The composition adds the two maps, by the basic composition rule of [the full edition's chapter on composition](../../book-full/part03-dp-foundations/04-composition.md). The third piece solves for the line, which is post-processing of two released numbers and costs nothing.

```{.python excerpt="plug-theil-sen"}
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
```

The script runs it with the documentation's bounds and a scale of 1, on the toy points. Their true line has slope 2 and intercept 1.

```text
theil-sen: x cuts -1.5 and 1.5, 100 candidates from -10.0 to 10.0 step 0.2020
pairwise predict: map(1)=1 medians: map(1)=2.0 theil-sen: map(1)=2.0 map(2)=4.0
toy data: n=100 least-squares slope=2.0573 intercept=1.0376
pairs: 50 medians at the cuts -2.2424 and 4.2857
selection bound: 26.2 score units at noise scale 2.0
cut -1.5: best candidate -2.3232 score 2; within the bound: -2.9293 to -1.5152 (8 candidates)
cut 1.5: best candidate 4.3434 score 8; within the bound: 3.3333 to 6.3636 (16 candidates)
```

The pairing has stability 1 and the two medians cost 2 together, so the whole estimator costs an $\varepsilon$ of 2 per record. The least-squares fit of the toy is close to the truth, and the exact medians of the fifty pairwise predictions are the numbers the private medians aim at.

The last three lines are the bound the released medians are read against. The private quantile adds exponential noise of scale 2 to each candidate's score and picks the best. The difference of two such draws is a Laplace draw of the same scale. So the chance that a worse candidate wins falls exponentially in its gap to the best, measured in scales. A union bound over the other 99 candidates gives the printed bound, which a losing candidate beats at most once in ten thousand runs. The candidates within the bound span about one and a half units of $y$ at the lower cut and about three at the upper cut. The predictions there are more spread out, so neighbouring candidates score alike.

## Choosing grouping columns privately {#sec-plug-grouping}

The documentation's second larger example solves a problem you meet before any grouped release. You want to group the rows by some columns before releasing counts, but you may not look at the data to choose them. Some columns are too uniform, with most rows sharing a value, and some are too diverse, with most rows having their own. Grouping by too many columns leaves most groups too small to survive the thresholds of [Grouping](../part02-tabular-data/04-grouping.md). The example scores each candidate set by how many of its groups reach a public minimum of rows. It then picks a set with the **exponential mechanism**, which chooses a candidate with probability growing exponentially in its score. [The full edition's chapter on selection](../../book-full/part03-dp-foundations/05-selection.md) derives it.

The score is a transformation plugin, and it needs a user domain because the library has no domain for a whole data frame. The documentation scores a pandas frame, and the same score runs here on a Polars frame.

```{.python excerpt="plug-score"}
def score(frame: pl.DataFrame, columns: tuple[str, ...], min_rows: int) -> float:
    """How many groups of these columns have at least `min_rows` rows."""
    return float(frame.group_by(list(columns)).len().filter(pl.col("len") >= min_rows).height)
```

The transformation applies it to every candidate set.

```{.python excerpt="plug-score-transformation"}
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
```

The output metric is the L-infinity distance, the largest change in any one score, because noisy max compares scores rather than summing them. The map is the identity on the input distance, and the reason is that one row belongs to one group of each candidate. Adding or removing it changes that group's size by one, so it moves each score by at most one, and `d_in` rows move each score by at most `d_in`.

The documentation declares the metric monotonic, which halves the cost of noisy max. That is right when neighbours differ by one row added or one removed, since every score then moves the same way. Here the privacy unit is a person with up to 36 rows, and a neighbouring dataset can add some of them and remove others. The scores can then move both ways, so the script keeps the metric plain and pays the factor of two. [The full edition's chapter on sensitivity](../../book-full/part03-dp-foundations/07-sensitivity.md) is where these bounds come from.

The measurement chains the scores into noisy max and names the chosen candidate, and it is registered so that a context can run it.

```{.python excerpt="plug-select"}
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
```

On the survey the candidates are every non-empty subset of six columns. Five are `SEX`, `ILOSTAT`, `YEAR`, `QUARTER` and `AGE`, and the sixth is the person identifier `PIDENT`. This sample codes age in seven bands. A group counts when it has at least 100 rows, a public choice standing in for the smallest group worth a noisy count. The context is built over the collected frame with the user domain passed as `domain`, a privacy unit of 36 contributions and a budget of $\varepsilon = 1$. The registered method is called with the scale left out, so that the context searches for it against the plugin's map.

```text
LFS: 3,811,932 rows; candidate columns SEX, ILOSTAT, YEAR, QUARTER, AGE, PIDENT; 63 candidate sets
distinct values: SEX 2 ILOSTAT 4 YEAR 9 QUARTER 4 AGE 7 PIDENT 1,682,306
score: groups of at least 100 rows
rank 1: SEX, ILOSTAT, YEAR, QUARTER, AGE score 996 of 1,105 groups
rank 2: ILOSTAT, YEAR, QUARTER, AGE score 504 of 570 groups
rank 3: SEX, YEAR, QUARTER, AGE score 432 of 470 groups
sets with PIDENT: 32, largest score 0
select scale 72: map(36)=1.0 monotonic would be 0.5
context query: param scale=72.0
within 910 of the best: 10 candidates
```

The scores are computed at development time, which is allowed because the survey is public, and they show the shape the documentation describes. The five columns without the identifier score best, with most of their groups above the minimum, and the next set scores about half of that. Every set that includes `PIDENT` scores zero, because no person has a hundred rows. The map of the selection is twice the sensitivity over the scale, so a scale of 72 costs an $\varepsilon$ of 1, and the context's search lands on that scale.

The bound follows the same reasoning as for the medians, with exponential noise at scale 72 and a union bound over 62 other candidates. A set that scores more than 910 below the best therefore wins at most once in ten thousand runs, and ten sets lie within that bound. The best beats the runner-up by 492, under seven noise scales, so it wins most runs, but any of the ten can appear. That spread is the price of the person-level unit on data with few, coarse columns. A person can move each score by 36, while the scores themselves are only in the hundreds. Raising the minimum group size lowers every score and brings the candidates closer together, until the choice is nearly random, and spending more budget is the only other lever.

```text
released (one run):
  python laplace of 10.0: 9.6979  (within 6.0 with probability 0.95)
  repeated sum of ['0', '1', '2', '3'] with laplace scale 1: 13  (true 12)
  theil-sen: slope 2.1549 intercept 1.1111  (medians at the cuts -2.1212 and 4.3434)
  grouping columns: SEX, ILOSTAT, YEAR, QUARTER, AGE
```

The Python Laplace landed within its error bar, and the repeated sum within one of the truth. Both private medians fell inside their candidate ranges, so the slope and the intercept land near 2 and 1. The chosen grouping columns are the five that score best, and your run can differ in each of these.

## Summary

| Call or concept | What it does | Rule of thumb |
| --- | --- | --- |
| `dp.enable_features("honest-but-curious")` | Unlocks the plugin constructors | The library cannot check a Python map; the proof is yours |
| `make_user_measurement(...)` | A measurement from your function and privacy map | Compare the map with a built-in cousin where one exists |
| `make_user_transformation(...)` | A transformation from your function and stability map | Justify the map and the output domain |
| `user_domain`, `user_distance`, `user_divergence` | Your own domain, metric and measure | Identifiers name one thing; a user measure cannot be composed |
| `new_privacy_profile`, `new_function` | A profile from a curve; a post-processor | No global state, no data-dependent exceptions |
| `dp.register(make_thing)` | `context.query().thing(...)` | Take `input_domain` and `input_metric` first |
| A plugin in a context | Charged its share of the budget, not its map | A cheap plugin does not give its share back |
| Theil-Sen from plugins | Pair, predict at two cuts, two private medians, solve | Use the vetted estimator for real work |
| Grouping columns by noisy max | Score each set by its groups above a minimum, pick one | Monotonic only for one-row units |

: The plugin constructors, what each does, and the rule of thumb that goes with it. {#tbl-plug-summary}

## References

- OpenDP documentation: "Plugins", "Measurement example", "Transformation example", "Context API Plugins", "Thiel-Sen Regression" and "Privately Selecting Grouping Columns" in the API user guide.
- Theil (1950) and Sen, "Estimates of the regression coefficient based on Kendall's tau" (1968): the estimator behind the regression example.
- McSherry and Talwar, "Mechanism Design via Differential Privacy" (2007): the exponential mechanism behind noisy max.
- Gaboardi, Hay and Vadhan, "A Programming Framework for OpenDP" (2020): the maps a plugin must satisfy.
- The full edition's chapters on [the Laplace and Gaussian mechanisms](../../book-full/part03-dp-foundations/03-laplace-gaussian.md), [composition](../../book-full/part03-dp-foundations/04-composition.md), [selection](../../book-full/part03-dp-foundations/05-selection.md) and [sensitivity](../../book-full/part03-dp-foundations/07-sensitivity.md) derive the results this chapter states.
