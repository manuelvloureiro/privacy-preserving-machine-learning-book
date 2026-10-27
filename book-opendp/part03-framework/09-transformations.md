# Transformations

Every query of the previous parts started with the data and ended with noise, with a chain of steps between them that cast, imputed, clamped, resized and aggregated. In OpenDP each step is a **transformation**, a function from data to data. Its stability map bounds how much its output can change when its input changes ([A framework to understand differential privacy](08-a-framework-for-privacy.md)). This chapter builds the transformations, from preprocessing to the sum, the mean and the quantile scores, and shows how stability passes along a chain.

A chain works only when the output space of each step is exactly the input space of the next. Then the chain's stability map applies the maps of its steps in turn. At the end of the chapter, you can build a sum, a mean or a median's scores from the framework alone. You can also read the sensitivity for your privacy unit, and recognize the error that OpenDP raises when two spaces do not match.

The script [opendp_transformations.py](../../code/part03-framework/opendp_transformations.py) runs every step on OpenDP 0.16.0 with the Labour Force Survey (LFS) sample. Its vector is the `HWUSUAL` column, the hours a respondent usually works, as integers, without its nulls and the code 99 for "not applicable". The sensitivities, scales, accuracies, scores and true values are the same on every run. The noisy releases change, because OpenDP's noise cannot be seeded, and so does a float sum that keeps a random sample of its input. They are in one block marked as one run.

```bash
uv run python code/part03-framework/opendp_transformations.py
```

## Constructors and chains {#sec-trans-chains}

Every constructor has two forms. The `make_` form takes the input space first, as a domain and a metric, and then its own arguments, as in `make_clamp(input_domain, input_metric, bounds)`. The `then_` form leaves the space out. It returns a partial constructor, and `>>` fills in the space from the object to its left, which is a space written as a tuple or a transformation. The two forms build the same object.

```python
import opendp.prelude as dp

dp.enable_features("contrib")
space = dp.vector_domain(dp.atom_domain(T=int)), dp.symmetric_distance()
clamp = dp.t.make_clamp(*space, bounds=(0, 10))
print(clamp([-5, 3, 50]))  # [0, 3, 10]
print(clamp.output_space)  # (VectorDomain(AtomDomain(bounds=[0, 10], T=i32)), SymmetricDistance())
total = dp.t.make_sum(*clamp.output_space)
chain = space >> dp.t.then_clamp(bounds=(0, 10)) >> dp.t.then_sum()
print(chain([-5, 3, 50]), (clamp >> total)([-5, 3, 50]))  # 13 13
print(chain.output_space)  # (AtomDomain(T=i32), AbsoluteDistance(i32))
print(clamp.map(1), total.map(1), chain.map(1))  # 1 10 10
```

The clamp changes the domain and not the metric. Its output is vectors of integers between 0 and 10, still under the symmetric distance, the number of rows to add or remove to turn one vector into the other. The sum takes that output space as its input space, as `make_sum(*clamp.output_space)` says, and outputs one integer under the absolute distance. So read `output_space` after each step, because the next constructor must accept exactly that space (@sec-trans-reading).

The number that `map` returns is the **stability**, the largest change in the output for inputs at most `d_in` apart. For the clamp it is 1, because one added row in gives one added row out. For the sum it is 10, the larger bound, and for the chain it is the sum's map applied to the clamp's. When a chain ends with an aggregator, its stability is the sensitivity of the query, the number a measurement needs to calibrate its noise. [The full edition's chapter on sensitivity](../../book-full/part03-dp-foundations/07-sensitivity.md) derives it for sums, means and histograms.

## Preprocessing {#sec-trans-preprocessing}

Preprocessing turns the data into the domain that an aggregator accepts: the sum needs bounded values, and the mean needs bounded values and a known size. The documentation lists the preprocessing constructors in the usual order of a chain, which is cast, impute, clamp and resize, and the subsections keep that order. All of them except the resize work row by row, so one added row in is one added row out, and the stability is 1.

### Casting and imputing

Data from a file are strings, and `make_cast` turns a vector of strings into a vector of another type. A string that does not parse becomes a missing value, so the output is a vector of options, which hold a value or nothing. No aggregator accepts options. If it skipped the missing values, it would understate the sensitivity, and if it kept them, it would show whether a value was missing. So remove them first. The step `make_impute_constant` puts in a value you choose, and `make_drop_null` drops the rows. The step `make_cast_default` casts and imputes the type's default, 0 for numbers.

```python
import opendp.prelude as dp

dp.enable_features("contrib")
strings = dp.vector_domain(dp.atom_domain(T=str)), dp.symmetric_distance()
cast = strings >> dp.t.then_cast(TOA=int)
print(cast.output_domain)  # VectorDomain(OptionDomain(AtomDomain(T=i32)))
imputed = cast >> dp.t.then_impute_constant(0)
print(imputed(["1", "2", "x"]))  # [1, 2, 0]
print(imputed.output_domain)  # VectorDomain(AtomDomain(T=i32))
by_default = strings >> dp.t.then_cast_default(TOA=int)
print(by_default(["1", "2", "x"]))  # [1, 2, 0]
floats = dp.vector_domain(dp.atom_domain(T=float)), dp.symmetric_distance()
print(floats[0])  # VectorDomain(AtomDomain(nan=true, T=f64))
uniform = floats >> dp.t.then_impute_uniform_float(bounds=(0.0, 10.0))
print(uniform.output_domain)  # VectorDomain(AtomDomain(T=f64))
print(cast.map(1), imputed.map(1), uniform.map(1))  # 1 1 1
```

Floats have their own missing value, NaN, which the plain `atom_domain(T=float)` allows, as the printed `nan=true` shows. The constructor `make_impute_constant` accepts that domain too. And `make_impute_uniform_float` replaces each NaN by a uniform draw from the bounds you give, so that imputed values do not collect at one point. Both output the domain without NaN that the aggregators need, with a stability of 1. [Preparing microdata](../part02-tabular-data/06-preparing-microdata.md) does the same on a whole table through Polars.

### Clamping and resizing

Clamping moves every value into public bounds: a value below the lower bound becomes the lower bound, and a value above the upper becomes the upper. So it costs no accuracy on data that are already in the bounds, and it makes the sum's sensitivity finite. Take the bounds from public knowledge, such as a codebook. Do not take them from the data, because a bound read from the data leaks it.

Resizing has a stability of 2. It pads a vector shorter than the target size with a constant, and cuts a longer one down by a random sample. The mean and the variance need it, because they must know the number of rows they divide by, and that number is private. So the resize writes a size into the domain, which OpenDP then treats as public: a guess, a released count or public knowledge. For the stability, add one row to an input shorter than the target. The output gains that row and loses one padded row, so two rows change for one.

```python
import opendp.prelude as dp

dp.enable_features("contrib")
floats = dp.vector_domain(dp.atom_domain(T=float, nan=False)), dp.symmetric_distance()
clamp = floats >> dp.t.then_clamp(bounds=(0.0, 10.0))
print(clamp([-5.0, 3.0, 50.0]), clamp.map(1))  # [0.0, 3.0, 10.0] 1
resize = clamp >> dp.t.then_resize(size=5, constant=5.0)
print(resize([1.0, 2.0, 3.0]))  # [1.0, 2.0, 3.0, 5.0, 5.0]
print(resize.output_domain)  # VectorDomain(AtomDomain(bounds=[0.0, 10.0], T=f64), size=5)
print(resize.map(1))  # 2
```

A wrong target biases the result, as the documentation's resize page shows. A target below the true size throws rows away, which widens the spread of the mean, and a target above it pulls the mean toward the constant. No error bar shows either error, because the size is treated as true, and the resize cannot warn you without leaking the true size. So spend a little budget on a count first, which lands close to the truth because its sensitivity is 1.

### Categories and bins

Three constructors map between categorical values and indices. The first, `make_find`, replaces each value by its position in a list of categories. A value outside the list becomes a missing index, so an imputation follows and gives it the next index. The step `make_find_bin` turns a number into the index of its bin, from the bin edges, and `make_index` turns an index back into a label.

```python
import opendp.prelude as dp

dp.enable_features("contrib")
strings = dp.vector_domain(dp.atom_domain(T=str)), dp.symmetric_distance()
find = strings >> dp.t.then_find(categories=["A", "B", "C"]) >> dp.t.then_impute_constant(3)
print(find(["A", "B", "C", "A", "D"]))  # [0, 1, 2, 0, 3]
floats = dp.vector_domain(dp.atom_domain(T=float, nan=False)), dp.symmetric_distance()
bins = floats >> dp.t.then_find_bin(edges=[1.0, 2.0, 10.0])
print(bins([0.0, 1.0, 3.0, 15.0]))  # [0, 1, 2, 3]
label = bins >> dp.t.then_index(categories=["low", "mid", "high"], null="other")
print(label([0.0, 1.0, 3.0, 15.0]))  # ['low', 'mid', 'high', 'other']
counts = strings >> dp.t.then_count_by_categories(categories=["A", "B", "C"])
print(counts(["A", "B", "C", "A", "D"]))  # [2, 1, 1, 1]
print(counts.output_metric, counts.map(1))  # L1Distance(i32) 1
```

The aggregator for categories is `make_count_by_categories`, which counts each category and adds a count of all other values unless you set `null_category=False`. Its output is a vector of counts, so its output metric is a vector norm. The default $L_1$ distance goes with Laplace noise, and the $L_2$ distance, through the `MO` argument, goes with Gaussian noise. One added row changes one count by one, so the stability is 1 in either norm.

## The sum {#sec-trans-sum}

The sum is the base of the other aggregators, and its documentation page shows where sensitivity comes from. Start with integers in public bounds and no known size.

```python
import opendp.prelude as dp

dp.enable_features("contrib")
bounded = dp.vector_domain(dp.atom_domain(bounds=(0, 10))), dp.symmetric_distance()
total = bounded >> dp.t.then_sum()
print(total([1, 2, 4]), total.map(1))  # 7 10
sized = dp.vector_domain(dp.atom_domain(bounds=(-10, 10)), size=3), dp.symmetric_distance()
sized_total = sized >> dp.t.then_sum()
print([sized_total.map(d) for d in range(5)])  # [0, 0, 20, 20, 40]
```

One added row changes the sum by at most the larger bound in absolute value, here 10, which is `map(1)`. When the size is public, you write it into the domain, and the bound is tighter. Two datasets of the same size differ at least by one replaced row, which is one removal and one addition, a symmetric distance of 2. A replacement changes the sum by at most the width of the bounds, which is `map(2)`. Odd distances round down, so `map(1)` is 0 and the last line is a staircase.

On the survey, the script builds the vector once with Polars, and the chain clamps the hours into the codebook's bounds, 0 to 98, and sums them.

```{.python excerpt="trans-hours"}
def hours_usually_worked(frame: pl.LazyFrame) -> list[int]:
    """The `HWUSUAL` column without nulls and without the code 99, as a list of ints."""
    kept = frame.filter(pl.col("HWUSUAL").is_not_null() & (pl.col("HWUSUAL") != 99))
    return kept.select(pl.col("HWUSUAL").cast(pl.Int64)).collect()["HWUSUAL"].to_list()
```

```{.python excerpt="trans-int-sum"}
def int_sum(size: int | None = None) -> dp.Transformation:
    """Clamp to the public bounds, then sum; with a size the domain says how many rows there are."""
    space = dp.vector_domain(dp.atom_domain(T=int), size=size), dp.symmetric_distance()
    return space >> dp.t.then_clamp(BOUNDS) >> dp.t.then_sum()
```

```text
hours vector: 1,496,192 values
true sum: 56,315,594  true mean: 37.6393  true median: 37
int sum: value 56,315,594  map(1)=98  map(36)=3528
sized int sum: map(36)=1764  map(35)=1666
```

Every map on this page is read at an input distance of 36, the most rows one person has ([the privacy unit](../part01-getting-started/02-the-data-and-the-context.md#sec-ctx-context)). Without a size, the sensitivity is 36 rows of 98 hours, which is 3,528 hours. With the size, it is half that, 18 replacements, and a distance of 35 rounds down to 17. But the Polars filter runs outside OpenDP, so the guarantee does not cover it. [Preparing microdata](../part02-tabular-data/06-preparing-microdata.md) runs it inside the guarantee, and [Essential statistics](../part02-tabular-data/03-essential-statistics.md) releases this sum through the Context API.

Floats are different, for two reasons from the documentation's sum page. The first is rounding: each partial sum of floats rounds to the nearest float. So the sums of two datasets at distance 2 can differ by a little more than the width of the bounds, and the stability map adds a term for that.

```python
import opendp.prelude as dp

dp.enable_features("contrib")
metric = dp.symmetric_distance()
sized = dp.vector_domain(dp.atom_domain(bounds=(-10.0, 10.0)), size=1000), metric
print((sized >> dp.t.then_sum()).map(2))  # 20.00000000004426
unsized = dp.vector_domain(dp.atom_domain(bounds=(-10.0, 10.0))), metric
print((unsized >> dp.t.then_sum()).map(1))  # 20.00000009313226
negative = dp.vector_domain(dp.atom_domain(bounds=(-10.0, 0.0))), metric
print((negative >> dp.t.then_sum()).map(1))  # 10.00000009313226
print(dp.t.make_bounded_float_checked_sum(size_limit=100, bounds=(-10.0, 0.0)).map(1))  # 10.00000000000295
```

With a known size of a thousand rows, the term is in the eleventh decimal place. The second reason applies when the size is unknown. Then the rounding error has no bound, so the float sum first keeps at most $2^{20}$ rows, a little over a million, by a random sample. A neighbour with one more row may then drop a different row, so the sensitivity becomes the largest of the magnitudes of the bounds and their width. That doubles it for bounds symmetric about zero (the second line) but not for bounds of one sign (the third). With `make_bounded_float_checked_sum` you set the size limit, and a limit of 100 shrinks the rounding term to the twelfth decimal place.

This limit matters on the survey, because the hours vector has 1,496,192 values, more than $2^{20}$. So a plain `then_sum` on them as floats sums a random million of them. The result is about 70% of the true sum, as the one-run block at the end shows, and nothing warns you. The guarantee holds for the truncated sum, but the number answers a different question. So set a size limit from public knowledge above any size the data can have, here four million rows. Or put the exact size in the domain when it is public.

```{.python excerpt="trans-float-sum"}
def float_sum(size_limit: int | None = None) -> dp.Transformation:
    """The float sum: `then_sum` truncates to 2^20 rows; a size limit of your own lifts that."""
    space = dp.vector_domain(dp.atom_domain(T=float, nan=False)), dp.symmetric_distance()
    clamped = space >> dp.t.then_clamp(FLOAT_BOUNDS)
    if size_limit is None:
        return clamped >> dp.t.then_sum()
    return clamped >> dp.t.make_bounded_float_checked_sum(size_limit=size_limit, bounds=FLOAT_BOUNDS)
```

```text
float sum, size limit 2^20=1,048,576: map(36)=3528.0000009127
float sum, size limit 4,000,000: value 56,315,594  map(36)=3528.0000038179
```

Both sensitivities are 3,528 hours plus a rounding term, and the larger limit gives a larger term, in the sixth decimal place instead of the seventh. The term also depends on the summation algorithm, which the specialized constructors take as the `S` argument.

```python
import opendp.prelude as dp

dp.enable_features("contrib")
pairwise = dp.t.make_sized_bounded_float_checked_sum(1_000, (0.0, 10.0), S="Pairwise<f64>")
sequential = dp.t.make_sized_bounded_float_checked_sum(1_000, (0.0, 10.0), S="Sequential<f64>")
print(f"{pairwise.map(0):.2e} {sequential.map(0):.2e}")  # 4.43e-11 4.44e-09
pairwise = dp.t.make_sized_bounded_float_checked_sum(1_000_000, (0.0, 10.0), S="Pairwise<f64>")
sequential = dp.t.make_sized_bounded_float_checked_sum(1_000_000, (0.0, 10.0), S="Sequential<f64>")
print(f"{pairwise.map(0):.2e} {sequential.map(0):.2e}")  # 8.85e-08 4.44e-03
```

The map at distance 0 isolates the rounding term, because the same rows in a different order can sum to different floats. The default pairwise sum adds the values as a balanced tree, so its term grows a little faster than the number of rows. The sequential sum adds them one after another, and its term grows with the square of the number of rows. Use the sequential analysis for a sum computed elsewhere, because database engines add floats in sequence. Integer sums have no rounding term, which is one reason to cast hours to integers.

To release the integer sum, add a Laplace measurement with a scale equal to the sensitivity over $\varepsilon$ ([the full edition's chapter on the Laplace and Gaussian mechanisms](../../book-full/part03-dp-foundations/03-laplace-gaussian.md)). Each of the three measurements on this page uses $\varepsilon = 0.5$ on its own, and [Combinators](11-combinators.md) composes them. Here the scale is 7,056 hours, and the error bar is 0.04% of the total.

```text
sum release: scale=7056  map(36)=0.5000  95% accuracy=21,138.4
```

## The mean and the variance {#sec-trans-mean}

The mean is a sum divided by a size, and `make_mean` divides by the size in its input domain, not by the length of the data. So the domain must have a size, or the constructor refuses.

```text
mean without a size: dataset size must be known. Either specify size in the input domain or use make_resize
```

```python
import opendp.prelude as dp

dp.enable_features("contrib")
space = dp.vector_domain(dp.atom_domain(bounds=(0.0, 10.0)), size=10), dp.symmetric_distance()
mean = space >> dp.t.then_mean()
print(mean([5.0] * 10), mean.map(2))  # 5.0 1.0000000000000169
print(mean([5.0]))  # 0.5
print(mean.input_domain.member([5.0]))  # False
```

With ten values between 0 and 10, one replaced value moves the mean by at most the width over the size. So `map(2)` is 1, plus a rounding term. The second print shows the documentation's warning: the mean of a one-element list is 0.5, because the division uses the declared size and nothing checks the data. Each transformation assumes that its input is in its input domain, and `member` checks that.

On the survey the size is not public, so the chain resizes first, with the true median of 37 hours as the padding constant. A real release would resize to a released count, as in [A first release](../part01-getting-started/01-first-release.md), or to a public total. The script resizes to the true size, which it can do because the data are public, so nothing is padded.

```{.python excerpt="trans-mean"}
def mean_chain(size: int, constant: float) -> dp.Transformation:
    """Clamp, resize to a known size (padding with the constant), then the mean over that size."""
    space = dp.vector_domain(dp.atom_domain(T=float, nan=False)), dp.symmetric_distance()
    return space >> dp.t.then_clamp(FLOAT_BOUNDS) >> dp.t.then_resize(size=size, constant=constant) >> dp.t.then_mean()
```

```text
mean chain: value 37.6393  map(36)=0.0023580
mean release: scale=0.0047160  map(36)=0.5000  95% accuracy=0.0141
```

Compare this map with the sized sum's. That map was 18 replacements of 98 hours, and divided by the size it is 0.0012. The mean's map is twice that, because the resize in front of it is 2-stable and turns the input distance of 36 into 72. That is the cost of an unknown size, paid once per chain. At $\varepsilon = 0.5$, the Laplace scale is 0.0047 hours and the error bar 0.014 hours.

The documentation's mean page also shows the plug-in approach, which releases a sum and a count and then divides. The division is post-processing, so it costs no more budget ([the full edition's chapter on properties](../../book-full/part03-dp-foundations/02-properties.md)). But at the same budget the resized mean is tighter, because its numerator and denominator use the same number of rows. The variance uses the same preprocessing. On the resized chain, `then_variance` gives 137.63 hours squared, and its stability at distance 36 is 0.2311.

```text
variance chain: value 137.6290  map(36)=0.2311
```

## Quantile scores {#sec-trans-quantiles}

Noise alone cannot release a quantile, and the documentation's quantile page shows why. Take a dataset with half its values at the lower bound and half at the upper. One added value moves its median from the middle of the range to the upper bound, so the sensitivity is half the range. Laplace noise of that size hides the median. So OpenDP scores every candidate by how far it is from the quantile and lets a selection mechanism pick one. This is the exponential mechanism of [the full edition's chapter on selection](../../book-full/part03-dp-foundations/05-selection.md), and `make_quantile_score_candidates` computes the scores.

```python
import opendp.prelude as dp

dp.enable_features("contrib")
space = dp.vector_domain(dp.atom_domain(T=int)), dp.symmetric_distance()
median = space >> dp.t.then_quantile_score_candidates(candidates=[0, 25, 50, 75, 100], alpha=0.5)
print(median([10, 20, 30, 40, 50, 60, 70, 80, 90]))  # [9, 5, 0, 5, 9]
print(median.output_metric, median.map(1))  # LInfDistance(T=u64) 1
quartile = space >> dp.t.then_quantile_score_candidates(candidates=[0, 25, 50, 75, 100], alpha=0.25)
print(quartile([10, 20, 30, 40, 50, 60, 70, 80, 90]), quartile.map(1))  # [9, 1, 8, 19, 27] 3
```

A candidate's score for the median is the absolute difference between the number of values below it and the number above it. In the snippet, the middle candidate has four values on each side and scores zero, while the lowest has all nine above it. So the lowest score is the best, and the selection picks the minimum. For another quantile, the two counts get weights. The quarter quantile weights the values below by three, so the score is zero where a quarter of the data are below, and its stability is 3. The output metric is the $L_\infty$ distance, the largest change of any score, because one row moves every score at once.

On the survey, the candidates are the whole numbers of hours from 0 to 98.

```{.python excerpt="trans-median"}
def median_scores() -> dp.Transformation:
    """One score per candidate: how far it is from being the median; the lowest score is the best candidate."""
    space = dp.vector_domain(dp.atom_domain(T=int)), dp.symmetric_distance()
    return space >> dp.t.then_quantile_score_candidates(candidates=CANDIDATES, alpha=0.5)


def median_measurement(scale: float) -> dp.Measurement:
    """Noisy max over the scores releases an index; `negate` makes the lowest score the one most likely to win."""
    return median_scores() >> dp.m.then_noisy_max(dp.max_divergence(), scale=scale, negate=True)
```

```text
median scores: lowest 15,190 at 37 hours, next 89,573 at 36  map(36)=36
median release: scale=144  map(36)=0.5000  score gap=74,383  gap/scale=517
```

The lowest score is at 37 hours, the true median, and the next candidate scores worse by 74,383. To release it, the noisy max of [Measurements](10-measurements.md) adds noise to every score and reports the index of the best one. Its privacy map under the max divergence is twice the stability over the scale. So the scores' stability of 36 at a scale of 144 gives $\varepsilon = 0.5$. The gap between the two best scores is over 500 times the scale, so the released median is 37 on almost every run. The `make_private_quantile` of the next chapter chains the same steps but doubles its scale for the median, so its scale 72 is this chain's 144.

## Reading a chain {#sec-trans-reading}

Two methods read the promise of a chain. The `map` method gives the stability, or the privacy loss when a measurement ends the chain. And the `check` method asks whether the promise at an input distance is within a bound. The sum chain's map at 36 is 3,528, so `check(36, 3528)` is true and `check(36, 3527)` is false. Neither method touches the data, and the context runs `check` in a loop to find a scale ([Utilities and the Context API inside](12-utilities-and-the-context-api.md)).

```text
int sum chain: check(36, 3528)=True  check(36, 3527)=False
```

You must also read refusals. The `>>` operator checks, when it builds the chain, that the output space of each step is the input space of the next. Below, a sum built on vectors bounded by 0 and 100 follows a clamp to 0 and 98.

```{.python excerpt="trans-mismatch"}
try:
    _ = clamped >> dp.t.make_sum(dp.vector_domain(dp.atom_domain(bounds=(0, 100))), dp.symmetric_distance())
    mismatch = "built"
except dp.OpenDPException as err:
    mismatch = f"{err.variant}: {err.message}"
```

```text
mismatched chain: DomainMismatch: Intermediate domains don't match. See https://github.com/opendp/opendp/discussions/297
    output_domain: VectorDomain(AtomDomain(bounds=[0, 98], T=i32))
    input_domain:  VectorDomain(AtomDomain(bounds=[0, 100], T=i32))
```

The message names both spaces. To fix the chain, build the sum from the clamp's output space, as `make_sum(*clamped.output_space)`. The `then_` form does that for you. A metric mismatch is similar. An ordered sum needs the insert-delete distance, so it refuses to follow a clamp under the symmetric distance, and `make_ordered_random` between them converts the metric. The third refusal is a domain that lacks what the constructor needs, such as a size for the mean or bounds for the sum.

```python
import opendp.prelude as dp

dp.enable_features("contrib")
space = dp.vector_domain(dp.atom_domain(T=int)), dp.symmetric_distance()
clamp = space >> dp.t.then_clamp(bounds=(0, 10))
try:
    clamp >> dp.t.make_bounded_int_ordered_sum(bounds=(0, 10))
except dp.OpenDPException as err:
    print(err.variant)  # MetricMismatch
try:
    space >> dp.t.then_sum()
except dp.OpenDPException as err:
    print(err.message)  # `input_domain` must be bounded. Use `make_clamp` to bound data.
```

Every refusal happens when the chain is built, before it touches any data. Here are the releases of one run, to read against their error bars.

```text
released (one run):
  sum 56,307,689  (true 56,315,594, within 21,138 with probability 0.95)
  float sum truncated to 2^20 rows 39,467,741  (true 56,315,594; about 70% of it)
  mean 37.6367  (true 37.6393, within 0.0141 with probability 0.95)
  median 37  (true 37)
```

The sum and the mean landed within their bars, as they do about 19 times in 20. The truncated float sum is not noise but a wrong computation, 30% short, and no error bar covers it. The median is exact, as the gap between its scores predicted.

## Summary

| Call or concept | What it does | Rule of thumb |
| --- | --- | --- |
| `make_x(domain, metric, ...)`, `space >> then_x(...)` | Two forms of every constructor | Read `output_space` after each step |
| `transformation.map(d_in)` | The stability at input distance `d_in` | Read it at the privacy unit's distance, 36 here |
| `make_cast`, `make_cast_default` | Parse strings into a type; failures become options or the default | Impute or drop options before any aggregator |
| `make_impute_constant`, `make_impute_uniform_float` | Replace missing values by a constant or a uniform draw from bounds | Stability 1; the constant or bounds are public |
| `make_clamp(bounds)` | Pin values into public bounds | Bounds come from a codebook, never from the data |
| `make_resize(size, constant)` | Pad or sample to a known size, written into the domain | Stability 2; a wrong size biases silently |
| `make_find`, `make_find_bin`, `make_index`, `make_count_by_categories` | Values to indices, numbers to bins, indices to labels, counts per category | Counts carry an $L_1$ or $L_2$ metric |
| `make_sum` | Sum of bounded values; with a size, per replacement | Unsized float sums truncate to $2^{20}$ rows: set `size_limit` |
| `make_mean`, `make_variance` | Divide by the size in the domain | A resize before them doubles the input distance |
| `make_quantile_score_candidates` | A score per candidate, lowest best | Feed the scores to the next chapter's noisy max |

: The transformations of this chapter, what each does, and the rule of thumb that goes with it. {#tbl-trans-summary}

## References

- OpenDP documentation: "Transformations", "Preprocessing: Resize", "Aggregation: Sum", "Aggregation: Mean" and "Aggregation: Quantile" in the API user guide.
- Casacuberta, Shoemate, Vadhan and Wagaman, "Widespread Underestimation of Sensitivity in Differentially Private Libraries and How to Fix It" (2022): the rounding term and the truncation of the float sum.
- Smith, "Privacy-Preserving Statistical Estimation with Optimal Convergence Rates" (2011): the quantile scores and their sensitivity.
- McKenna and Sheldon, "Permute-and-Flip: A New Mechanism for Differentially Private Selection" (2020): the selection behind the noisy max.
- The full edition's chapters on [sensitivity](../../book-full/part03-dp-foundations/07-sensitivity.md), [the Laplace and Gaussian mechanisms](../../book-full/part03-dp-foundations/03-laplace-gaussian.md), [post-processing and composition](../../book-full/part03-dp-foundations/02-properties.md) and [selection](../../book-full/part03-dp-foundations/05-selection.md) derive the results this chapter states.
