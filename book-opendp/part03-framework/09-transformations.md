# Transformations

Every query of the previous parts began with the data and ended with noise, and in between stood a chain of steps that cast, imputed, clamped, resized and aggregated. In OpenDP each of those steps is a **transformation**, a function from data to data that comes with a promise about how much its output can change when its input changes. [A framework to understand differential privacy](08-a-framework-for-privacy.md) introduced that promise as the stability map. This chapter builds the transformations themselves, from the preprocessing steps to the sum, the mean and the scores behind a quantile, and shows how the promise travels down a chain.

The idea that holds the chapter together is that a chain is only as good as the match between its links. Each transformation has an input space and an output space, and the output space of one must be exactly the input space of the next. That match is what makes the stability of a chain the product of the stabilities of its parts. It is also why so many constructors exist, because each one takes the data one provable step closer to the shape an aggregator needs. By the end you will be able to build a sum, a mean or a median's scores from the framework alone and read off the sensitivity for your privacy unit. You will also recognize the error OpenDP raises when two spaces do not match.

The script [opendp_transformations.py](../../code/part03-framework/opendp_transformations.py) runs every step on OpenDP 0.16.0 with the Labour Force Survey (LFS) sample. Its vector is the `HWUSUAL` column, the hours a respondent usually works, without its nulls and without the code 99 for "not applicable", cast to integers. The sensitivities, scales, accuracies, scores and true values it prints are the same on every run. The released values are not, because OpenDP's noise cannot be seeded, and neither is a float sum that truncates its input by a random sample. Those appear in one block marked as one run. The short snippets use toy vectors instead and print only what is deterministic.

```bash
uv run python code/part03-framework/opendp_transformations.py
```

## Constructors and chains {#sec-trans-chains}

Every constructor comes in two forms. The `make_` form takes the input space first, as a domain and a metric, and then the constructor's own arguments, so `make_clamp(input_domain, input_metric, bounds)` is the full call. The `then_` form leaves the space out. It returns a partial constructor, and the `>>` operator fills in the space from whatever stands to its left, either a space written as a tuple or a transformation whose output space is used. The two forms build the same object, and the snippet shows both.

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

Clamping changes the domain and not the metric. The input domain was the vectors of integers, and the output domain is the vectors of integers between 0 and 10. The metric is still the symmetric distance, which counts the rows you must add or remove to turn one vector into the other. The sum then takes that output space as its input space, which `make_sum(*clamp.output_space)` says literally, and its own output is a single integer under the absolute distance. Reading `output_space` after each step is the habit this chapter teaches, because the next constructor must accept exactly that space. When it does not, the chain refuses to build, as @sec-trans-reading shows.

The number that `map` returns is the transformation's **stability**, the largest change in the output for inputs at most `d_in` apart, each change measured in its own metric. For the clamp it is 1, because one added row of the input is one added row of the output. For the sum it is 10, because one added row can add at most the larger bound to the total. For the chain it is the product of the two. When the last step of a chain is an aggregator, its stability is what differential privacy calls the sensitivity of the query, which [the full edition's chapter on sensitivity](../../book-full/part03-dp-foundations/07-sensitivity.md) derives for sums, means and histograms. That number is what a measurement needs to calibrate its noise.

## Preprocessing {#sec-trans-preprocessing}

Preprocessing turns the data into the domain an aggregator accepts, and every aggregator states that domain. The sum wants bounded values, for example, while the mean wants bounded values and a known size. The documentation lists the preprocessing constructors in the order you typically chain them, which is casting, imputing, clamping and resizing, and the subsections below keep that order. All but one of them work row by row, so one added row of the input is one added row of the output and the stability is 1. The exception is the resize.

### Casting and imputing

Data read from a file arrive as strings, and `make_cast` turns a vector of strings into a vector of another type. A string that does not parse becomes a missing value, so the output domain is a vector of options, a type that holds either a value or nothing. No aggregator accepts options, for a reason the documentation states plainly. An aggregator that skipped the missing values would understate the sensitivity of the people with values, and one that propagated them would reveal whether anything was missing. So the missing values have to go first. `make_impute_constant` replaces each with a value of your choice, and `make_drop_null` removes the rows. `make_cast_default` folds the cast and the imputation into one step, with the type's default standing in, which is 0 for numbers.

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

Floats carry their own missing value, the not-a-number value, and the domain says whether it is allowed. The plain `atom_domain(T=float)` allows it, which the `nan=true` in the printed domain records. `make_impute_constant` accepts that domain too, and `make_impute_uniform_float` replaces each missing float by a uniform draw from the bounds you pass, so that imputed values do not pile up at one point. Both output the domain without `nan`, which is the one the aggregators want. The stability of every step here is 1, as the last line shows. [Preparing microdata](../part02-tabular-data/06-preparing-microdata.md) does the same preparation on a whole table through Polars, inside the guarantee.

### Clamping and resizing

Clamping pins every value into public bounds. A value below the lower bound becomes the lower bound, and a value above the upper becomes the upper. It changes nothing in between, so it costs no accuracy on data that already lie within the bounds, and it is what makes the sum's sensitivity finite. Without a bound, one person could move a sum by any amount, and no finite noise could hide them. The bounds must come from public knowledge, such as a codebook, never from looking at the data, because a bound read off the data leaks it.

Resizing is the step with a stability of 2. It takes a target size and makes every vector that size: a shorter vector is padded with a constant, and a longer one is cut down by a random sample. It exists because some aggregators, the mean and the variance among them, must know how many rows they divide by, and the number of rows is itself private. The resize writes a size into the domain, which from then on OpenDP treats as public, and that size can be a guess, a released count or public knowledge. The stability follows from one case. Add one row to an input that was shorter than the target. The output gains that row and loses one padded row, so two rows changed for one.

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

The documentation's resize page shows by simulation what a wrong target does. A target below the true size throws rows away, which widens the spread of the mean that follows. A target above it pads with the constant, which pulls that mean toward it. Neither error appears in an error bar, because the size is treated as true. Nor can the resize warn you, since an error that depended on the true size would leak it. So spend a little budget on a count first, as the previous chapters did, because a count has a sensitivity of 1 and lands close to the truth.

### Categories and bins

Categorical data go through three constructors that map between values and indices. `make_find` replaces each value by its position in a category list, and a value outside the list becomes a missing index, so an imputation follows and gives it the next index. `make_find_bin` turns a number into the index of the bin it falls in, given the bin edges, and `make_index` goes the other way, from an index back to a label.

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

The aggregator that goes with them is `make_count_by_categories`. It counts how often each category occurs and appends a count of everything else, unless you turn that off with `null_category=False`. Its output is a vector of counts, so its output metric is a vector norm. The default is the $L_1$ distance, which chains with Laplace noise, and the `MO` argument switches it to the $L_2$ distance, which chains with Gaussian noise. Adding one row changes one count by one, so the stability is 1 in either norm.

## The sum {#sec-trans-sum}

The sum is the aggregator everything else is built on, and its documentation page is a study in where sensitivity comes from. Start with integers in public bounds and no known size.

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

Adding one row changes the sum by at most the larger of the two bounds in absolute value, here 10, and that is what `map(1)` reports. When the dataset's size is public, you write it into the domain and the sum uses a tighter bound. Two datasets of the same size cannot differ by one row. The nearest neighbours differ by one row replaced, which is one removal and one addition, a symmetric distance of 2. A replacement changes the sum by at most the width of the bounds, and that is `map(2)`. Odd distances round down, which gives the staircase in the last line. A distance of 1 maps to 0 because no two datasets of the same size are one row apart.

On the survey the same two calls read as follows. The vector is built once with Polars, and the chain clamps the hours into the codebook's bounds, 0 to 98, and sums them.

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

The privacy unit of this survey is one person, who appears at most 36 times, so every map on this page is read at an input distance of 36. Without a size the sensitivity is that many added rows of 98 hours each, which is 3,528 hours. With the size in the domain it is 18 replacements, which is half as much, and an odd distance of 35 rounds down to 17 replacements. The Polars filter that builds the vector runs outside OpenDP, as data preparation this chapter does not protect. [Preparing microdata](../part02-tabular-data/06-preparing-microdata.md) runs the same filter inside the guarantee, and [Essential statistics](../part02-tabular-data/03-essential-statistics.md) releases this sum through the Context API.

Floats behave differently, for two reasons that the documentation's sum page sets out. The first is rounding. Adding floats rounds each partial sum to the nearest representable number. So two datasets at distance 2 can have sums that differ by a little more than the width of the bounds, and the stability map adds a term for it.

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

With a known size of a thousand rows the term is tiny, in the eleventh decimal place. The second reason appears when the size is unknown. Then the rounding error has no bound, so the float sum first truncates its input to at most $2^{20}$ rows, a little over a million, by a random sample. Truncation changes the sensitivity analysis, because a neighbouring dataset with one more row may have to drop a random row to stay within the limit. The sensitivity therefore becomes the largest of the magnitudes of the bounds and their width. That doubles it for bounds symmetric about zero, as the second line shows, and changes nothing for bounds of the same sign, as the third line shows. The specialized constructor `make_bounded_float_checked_sum` lets you set the size limit yourself, and a limit of 100 shrinks the rounding term from the eighth decimal place to the twelfth.

The notebook's examples are too small to show why the limit matters. The hours vector has 1,496,192 values, more than $2^{20}$. A plain `then_sum` on it as floats keeps a random million of them and sums those. The result is about 70% of the true sum, as the one-run block at the end of the chapter shows. Nothing warns you. The privacy guarantee holds, because the sensitivity is right for the truncated computation, but the number answers a different question from the one you asked. The fix is a size limit above any size the data could have, from public knowledge, here four million rows, or the exact size in the domain when it is public.

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

Both sensitivities are 3,528 hours plus a rounding term, and the larger limit buys a larger term, in the sixth decimal place rather than the seventh. The term also depends on the summation algorithm, which the specialized constructors take as the `S` argument.

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

The map at distance 0 isolates the rounding term, since two datasets with the same rows in a different order can already sum to different floats. The default is the pairwise sum, which adds the values as a balanced tree, so its term grows a little faster than the number of rows. The sequential sum adds the values one after another, and its term grows with the square of the number of rows. At a million rows the sequential term is 0.0044 against the pairwise 0.0000001. The reason to ask for the sequential analysis is a sum computed elsewhere, because database engines add floats in sequence, and a sensitivity claimed for their result must match how they add. Integer sums have no rounding term, which is one reason to cast hours to integers.

To release the integer sum, chain a Laplace measurement onto the chain. Its scale is the sensitivity over $\varepsilon$, which [the full edition's chapter on the Laplace and Gaussian mechanisms](../../book-full/part03-dp-foundations/03-laplace-gaussian.md) derives. Each of the three measurements on this page is calibrated to $\varepsilon = 0.5$ on its own, and composing them is the job of [Combinators](11-combinators.md). At that budget the scale is 7,056 hours. The error bar is about 21,000 hours on a total of 56 million, which is 0.04%.

```text
sum release: scale=7056  map(36)=0.5000  95% accuracy=21,138.4
```

## The mean and the variance {#sec-trans-mean}

The mean is a sum divided by a size, and `make_mean` divides by the size in its input domain, not by the length of the data it is given. So the domain must carry a size, and without one the constructor refuses.

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

With ten values between 0 and 10, replacing one value moves the mean by at most the width over the size, which is 1. That is `map(2)`, up to a rounding term. The second print is the warning the documentation attaches to the mean. The mean of a one-element list came out as 0.5, because the division is by the size the domain declares, and nothing checks the data. A transformation assumes its input is a member of its input domain, and `member` checks that.

On the survey the size is not public, so the chain resizes first. The constant pads a vector that comes out short, and here it is the true median of 37 hours. A real release would resize to a released count, as [A first release](../part01-getting-started/01-first-release.md) did, or to a public total. The script resizes to the true size, which is allowed here because the data are public, so nothing is padded.

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

Read the map against the sized sum. The sized sum's map was 18 replacements of 98 hours, and dividing that by the size gives 0.0012. The mean's map is twice that, because the resize in front of it is 2-stable. The input distance of 36 becomes an output distance of 72, which the mean reads as twice as many replacements. That is the price of not knowing the size, and it is paid once per chain. At $\varepsilon = 0.5$ the Laplace scale is 0.0047 hours, and the error bar is 0.014 hours.

The documentation's mean page also shows the alternative, which it calls the plug-in approach: release a sum and a count separately and divide. The division is post-processing, so it costs nothing beyond the two releases, by the property that [the full edition's chapter on properties](../../book-full/part03-dp-foundations/02-properties.md) proves. The resize page compares the two by simulation and finds the resized mean tighter at the same budget, because its numerator and denominator agree on how many rows there are. The variance takes the same preprocessing and the same kind of map. On the resized chain `then_variance` gives 137.63 hours squared, with a stability of 0.2311 at distance 36, and a Laplace measurement would release it exactly as the mean.

```text
variance chain: value 137.6290  map(36)=0.2311
```

## Quantile scores {#sec-trans-quantiles}

A quantile is not a sum, and noise alone cannot release it. The documentation's quantile page opens with the reason. Take a dataset in which half the values sit at the lower bound and half at the upper. Its median jumps from the middle of the range to the upper bound when one value is added, so the median's sensitivity is half the range, and Laplace noise at that scale would drown it. Instead OpenDP scores every candidate quantile by how far it is from the truth and lets a selection mechanism pick one. That is the exponential mechanism of [the full edition's chapter on selection](../../book-full/part03-dp-foundations/05-selection.md), and `make_quantile_score_candidates` computes the scores.

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

The score of a candidate for the median is the difference between the number of values below it and the number above it, in absolute value. In the snippet the candidate in the middle has four values on either side and scores zero, while the lowest candidate has all nine above it. The lowest score marks the best candidate, which is why the selection in the next chapter is told to pick the minimum. For another quantile the two counts get weights. The quarter quantile weights the values below by three and the values above by one, so that the score is zero where a quarter of the data lie below. Its stability rises to 3, because one added row can move a weighted count by that much. The output metric is the $L_\infty$ distance, since one row moves every candidate's score at once, and that is the metric the noisy max measurement expects.

On the survey the candidates are every whole number of hours from 0 to 98.

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

The lowest score is at 37 hours, the true median. The next best candidate is one hour less and scores worse by 74,383. The scores' stability at distance 36 is 36. The measurement that turns scores into a quantile is the noisy max of [Measurements](10-measurements.md). It adds noise to every score and reports the index of the best one. Its privacy map under the max divergence is twice the stability over the scale, so a scale of 144 gives $\varepsilon = 0.5$. The released index names a candidate, which is post-processing. The gap between the two best scores is over five hundred times the scale, so the released median is 37 on essentially every run. The bundled `make_private_quantile` of the next chapter chains these same steps. Its scale is doubled before it reaches the noisy max for the median, so a private quantile at scale 72 is this chain at scale 144.

## Reading a chain {#sec-trans-reading}

Two methods read a chain's promise. `map` gives the stability, or the privacy loss once a measurement ends the chain, and `check` asks whether the promise at a given input distance stays within a bound. The sum chain's map at 36 is 3,528, so `check(36, 3528)` is true and `check(36, 3527)` is false. Both cost nothing and touch no data. The context of the first chapter runs `check` in a loop to find the largest scale that fits a budget, and [Utilities and the Context API inside](12-utilities-and-the-context-api.md) shows that search.

```text
int sum chain: check(36, 3528)=True  check(36, 3527)=False
```

The other thing to read is a refusal. The rule every chain obeys is that the output space of one step is the input space of the next, and `>>` checks it when the chain is built. The sum below is built on the vectors bounded by 0 and 100, then chained after a clamp to 0 and 98.

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

The message names both spaces. The fix is to build the sum from the clamp's output space, as `make_sum(*clamped.output_space)`, or to use the `then_` form, which does so for you. A metric mismatch reads the same way: an ordered sum needs the insert-delete distance, so it refuses to follow a clamp under the symmetric distance, and `make_ordered_random` between the two converts the metric. The third kind of refusal is a constructor whose input domain lacks what it needs, as the mean showed for a missing size and the sum shows for missing bounds.

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

Every refusal happens at construction, before any data are touched. Finally, the releases of one run, read against their error bars.

```text
released (one run):
  sum 56,307,689  (true 56,315,594, within 21,138 with probability 0.95)
  float sum truncated to 2^20 rows 39,467,741  (true 56,315,594; about 70% of it)
  mean 37.6367  (true 37.6393, within 0.0141 with probability 0.95)
  median 37  (true 37)
```

The sum and the mean landed within their bars, as they should about 19 times in 20. The truncated float sum is not a noisy release but a wrong computation, short by 30% with no noise added, and no error bar covers it. The median is exact, as the gap between its scores said it would be.

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
