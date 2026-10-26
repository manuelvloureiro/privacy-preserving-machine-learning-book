# A framework to understand differential privacy

Every query you released in the first two parts went through the Context API. It chose a noise scale, spent a share of the budget and refused the query that went over. This chapter opens that box. Underneath, OpenDP describes a private computation with two kinds of object. A transformation maps data to data, and it promises how far apart its outputs can be when its inputs are close. A measurement maps data to a random release, and it promises how far apart the distributions of its outputs can be. Each promise is a map from an input distance to an output distance. Differential privacy (DP) in OpenDP is the bookkeeping of such maps.

The chapter follows the documentation's Framework API pages. You will meet the two objects and their fields, then the supporting elements one at a time, with code for each. Domains say what the data may be. Metrics say how two datasets or two numbers are compared, and measures say how two distributions of releases are compared. The maps tie them together, and chaining a transformation with a measurement carries the guarantee from the data to the release. The chapter ends by redoing the first chapter's workflow with the framework API next to the Context API, which gives the same scales. By the end you will be able to read the guarantee of any OpenDP object off its fields and its map.

The script [opendp_framework.py](../../code/part03-framework/opendp_framework.py) runs every step on OpenDP 0.16.0. The vector of the sum example is the `HWUSUAL` column of the Labour Force Survey (LFS) sample, collected to a list of integers. The workflow runs on the first chapter's hundred toy scores. The domains, metrics, measures, maps, scales and error bars the script prints are the same on every run. The released values are not, because OpenDP's noise cannot be seeded, so the page shows them in one block marked as one run.

```bash
uv run python code/part03-framework/opendp_framework.py
```

## Two kinds of object {#sec-frame-objects}

The first chapter built a Laplace measurement from a space and a constructor without saying what came back. The documentation's running example is the same mechanism with a scale of 5. It is built with `make_laplace`, which takes the domain and the metric as its first two arguments instead of through `>>`.

```{.python excerpt="frame-laplace"}
def laplace_measurement(scale: float) -> dp.Measurement:
    """Laplace noise of the given scale on one finite float, compared by absolute distance, under pure DP."""
    return dp.m.make_laplace(
        dp.atom_domain(T=float, nan=False),  # input domain: any float that is not NaN
        dp.absolute_distance(T=float),  # input metric: |a - b|, the units of the sensitivity
        scale=scale,
    )
```

What comes back is a `Measurement`, and the script reads its fields.

```text
laplace scale 5: type=Measurement
laplace scale 5: input domain=AtomDomain(T=f64)
laplace scale 5: input metric=AbsoluteDistance(f64)
laplace scale 5: output measure=MaxDivergence
laplace scale 5: function=Function
```

A transformation and a measurement share four fields. The `input_domain` is the set of values the function accepts, and the `output_domain` is the set it produces. The `function` is the computation itself, and the `input_metric` is the rule for the distance between two inputs. The two objects differ in their last two fields, because a **transformation** is deterministic while a **measurement** is randomized. A transformation's outputs are data again, so it has an `output_metric` and a **stability map**, which bounds the distance between two outputs given a bound on the distance between the inputs. A measurement's output is a draw from a distribution, so it has an `output_measure` and a **privacy map**, which bounds the distance between two output distributions in the same way.

Read the Laplace measurement's fields in those terms. It accepts one finite float and compares two inputs by their absolute difference. It measures the gap between two output distributions by the max divergence, which is the quantity that $\varepsilon$ bounds. The documentation says that a transformation is (`d_in`, `d_out`)-stable when inputs at most `d_in` apart give outputs at most `d_out` apart. A measurement is (`d_in`, `d_out`)-differentially private when inputs at most `d_in` apart give output distributions at most `d_out` apart. Both sentences have the same shape, and that is the point of the framework. Every element of either object is one ingredient of that sentence, and the rest of the chapter takes them one at a time.

## Functions {#sec-frame-functions}

The function is the part you would have written anyway. A transformation or a measurement is called like a function, and `invoke` does the same thing under a name. The clamp below is built on the space of vectors of finite floats under the symmetric distance. Its bounds are 0 to 5, and it clips every element into them.

```python
import opendp.prelude as dp

dp.enable_features("contrib")
space = dp.vector_domain(dp.atom_domain(T=float, nan=False)), dp.symmetric_distance()
clamp = space >> dp.t.then_clamp(bounds=(0.0, 5.0))
print(type(clamp).__name__)  # Transformation
print(clamp([1.3, 7.8, -2.5, 7.0]))  # [1.3, 5.0, 0.0, 5.0]
print(clamp.invoke([10.0]))  # [5.0]
print(type(clamp.function).__name__)  # Function
```

The `function` field holds the computation as an object of its own, which the plugin chapter of the next part uses to wrap a Python function. Calling a transformation gives you exactly what the function computes, with no noise, so the result is not private. That is by design. A transformation prepares or aggregates the data inside the guarantee, and only a measurement may release anything. So a transformation has to be chained with a measurement before its output leaves the box. If you ever find yourself printing the output of a transformation, you have stepped outside the guarantee.

## Domains {#sec-frame-domains}

A domain is a set of values, and the library uses it in two ways. The proof behind a map holds only for inputs in the domain, so the domain states the assumptions the map relies on. And when two objects are chained, the library checks that the first one's output domain equals the second one's input domain. That way every intermediate value is one the next function was proved on. The most common domain is `atom_domain`, which describes one value of a type `T`, and `vector_domain` lifts it to vectors of such values.

```python
import opendp.prelude as dp

floats = dp.atom_domain(T=float, nan=False)
print(floats)  # AtomDomain(T=f64)
print(floats.member(1.0), floats.member(float("nan")))  # True False
print(dp.atom_domain(T=float))  # AtomDomain(nan=true, T=f64)
small = dp.atom_domain(bounds=(-2, 2))
print(small)  # AtomDomain(bounds=[-2, 2], T=i32)
print(small.member(-2), small.member(3))  # True False
pairs = dp.vector_domain(dp.atom_domain(T=bool), size=2)
print(pairs)  # VectorDomain(AtomDomain(T=bool), size=2)
print(pairs.member([True, True]), pairs.member([True, True, True]))  # True False
```

The `member` method tests whether a value belongs to the domain, which is how you check an assumption by hand. Three arguments shape an atom domain. The type `T` is written in Python terms, where `float` means a 64-bit float and `int` a 32-bit integer. That is why the printed forms say `f64` and `i32`. The `nan` flag says whether a float may be NaN. The installed version prints the flag only when it is true, so `atom_domain(T=float)` allows NaN and `atom_domain(T=float, nan=False)` does not. A measurement that adds noise needs the second, because the absolute distance between two values is not defined when one of them is NaN.

The `bounds` restrict the values to an interval. A domain with bounds is what the sum transformation demands, because the sum of unbounded values has no finite sensitivity. A vector domain adds a `size`, which fixes the length of the vector. The mean transformation needs it, for the same reason that the first chapter's mean needed a resize.

## Metrics {#sec-frame-metrics}

A metric is a rule for the distance between two members of a domain, and the framework never evaluates it. It only uses the metric to give `d_in` its meaning. Under the **symmetric distance**, two datasets are compared by the number of additions and removals that turn one into the other, counted with multiplicity. The function below computes it for two small vectors from the documentation's notebook, where the second lacks the first person's record.

```python
from collections import Counter

import opendp.prelude as dp


def symmetric_distance(u: list[int], v: list[int]) -> int:
    """The number of additions and removals that turn the multiset u into v."""
    cu, cv = Counter(u), Counter(v)
    return sum(((cu - cv) + (cv - cu)).values())


print(symmetric_distance([12, 10, 8, 7], [10, 8, 7]))  # 1
print(symmetric_distance([12, 10, 8, 7], [10, 10, 8, 7]))  # 2
print(dp.symmetric_distance(), dp.absolute_distance(T=float))  # SymmetricDistance() AbsoluteDistance(f64)
print(dp.unit_of(contributions=1, ordered=True))  # (InsertDeleteDistance(), 1)
print(dp.unit_of(changes=1))  # (ChangeOneDistance(), 1)
```

Removing one record gives a distance of 1. Changing a 12 into a 10 gives 2, because a change is one removal and one addition. So under the symmetric distance, `d_in` is the largest number of records one person can add or remove. That is the privacy unit of the first chapter in the framework's words. Two datasets are **neighbouring**, written $D \sim D'$, when they are at most `d_in` apart under the input metric, and here that relation is add/remove. The symmetric distance is the metric behind `dp.unit_of(contributions=k)`. A person who answers the LFS at most 36 times is at most 36 away from a dataset without them.

The **absolute distance** compares two numbers by the absolute value of their difference. It is the metric of an aggregate, so under it `d_in` is a **sensitivity**, the most an aggregate can change when one person's records change. You rarely supply a sensitivity yourself. It appears midway through a chain, as the output distance of an aggregation and the input distance of the noise. The full edition's [chapter on sensitivity](../../book-full/part03-dp-foundations/07-sensitivity.md) works out the sensitivities of the common aggregates.

The other dataset metrics are variations on the symmetric distance. The insert-delete distance counts additions and removals at specific positions, so it also protects the order of the records. The change-one distance counts replaced records in datasets of a fixed size, which is the replace-one relation that other books call bounded DP. The Hamming distance counts the positions at which two vectors of the same length differ, so it is the ordered version of change-one.

## Measures {#sec-frame-measures}

A measure is a rule for the distance between two probability distributions. It is what makes a measurement's `d_out` a privacy guarantee. The **max divergence** is the largest value, over every set of outputs, of the log of the ratio of the two probabilities of that set. Bounding it by $\varepsilon$ is pure differential privacy, which the full edition defines in [its chapter on the definition](../../book-full/part03-dp-foundations/01-definition.md). The library offers four measures, and `d_out` has a different form under each.

```python
import opendp.prelude as dp

dp.enable_features("contrib")
print(dp.max_divergence(), dp.zero_concentrated_divergence())  # MaxDivergence ZeroConcentratedDivergence
print(dp.smoothed_max_divergence(), dp.fixed_smoothed_max_divergence())  # SmoothedMaxDivergence Approximate(MaxDivergence)
space = dp.atom_domain(T=float, nan=False), dp.absolute_distance(T=float)
gaussian = space >> dp.m.then_gaussian(scale=1.0)
print(gaussian.output_measure, gaussian.map(1.0))  # ZeroConcentratedDivergence 0.5
profile = dp.c.make_zCDP_to_approxDP(gaussian).map(1.0)
print(f"{profile.epsilon(delta=1e-7):.4f}")  # 5.6709
print(dp.c.make_fix_delta(dp.c.make_zCDP_to_approxDP(gaussian), delta=1e-7).map(1.0))  # (5.6708588354687715, 1e-07)
```

Under the max divergence, `d_out` is a single number, $\varepsilon$. Under the **zero-concentrated divergence**, `d_out` is the parameter $\rho$ of zero-concentrated differential privacy (zCDP). That measure bounds a family of Rényi divergences at once, and it is the native measure of the Gaussian mechanism. A Gaussian of scale 1 at sensitivity 1 costs $\rho = 0.5$, which is the sensitivity squared over twice the scale squared.

The two smoothed measures carry approximate differential privacy, whose guarantee holds up to a slack $\delta$. Under the smoothed max divergence, `d_out` is a whole curve, a **privacy profile** that gives the smallest $\varepsilon$ for any $\delta$ you ask for, and the snippet asks it for $\delta = 10^{-7}$. Under the fixed smoothed max divergence, `d_out` is instead one pair of $\varepsilon$ and $\delta$. The installed version prints that measure as `Approximate(MaxDivergence)`, and the first chapter's `dp.loss_of(epsilon=1.0, delta=1e-7)` is a budget under it. The combinator `make_fix_delta` turns the curve into the pair by choosing the $\delta$. The conversions between the measures are the subject of [Combinators](11-combinators.md). Here the point is only that a measurement's `output_measure` tells you how to read its `d_out`.

## Stability maps and privacy maps {#sec-frame-maps}

A map is the function that turns an input distance into an output distance, and it is where the guarantee lives. For a measurement, `map(d_in)` returns the smallest `d_out` the library can prove. The measurement is then (`d_in`, `d_out`)-differentially private. For a transformation, the same call returns the smallest `d_out` for which it is (`d_in`, `d_out`)-stable. The `check` method compares the map's value with a `d_out` you propose and answers whether the guarantee fits inside it. The Laplace measurement of the first section is the running example, and the script reads its map at two sensitivities.

```{.python excerpt="frame-map"}
epsilon_at_1 = laplace.map(d_in=1.0)  # the sensitivity over the scale: 1 / 5
epsilon_at_10 = laplace.map(d_in=10.0)
fits = laplace.check(d_in=1.0, d_out=0.2)  # is map(1.0) <= 0.2?
too_tight = laplace.check(d_in=1.0, d_out=0.19)
```

```text
laplace scale 5: map(1.0)=0.2 map(10.0)=2.0
laplace scale 5: check(1.0, 0.2)=True check(1.0, 0.19)=False
```

The Laplace mechanism's privacy map is the sensitivity divided by the scale, because the scale that gives $\varepsilon$ is the sensitivity over $\varepsilon$. So a sensitivity of 1 at scale 5 costs $\varepsilon = 0.2$. A sensitivity ten times larger costs ten times more. The check at $\varepsilon = 0.2$ passes and the check at 0.19 fails, which is all `check` ever does. It is also all the Context API does when it searches for a scale. It tries scales until `check` passes at the query's share of the budget. The full edition derives the map in [its chapter on the Laplace and Gaussian mechanisms](../../book-full/part03-dp-foundations/03-laplace-gaussian.md).

The value a map returns is an upper bound that the library can prove, not always the smallest true one, and the sum shows why that matters.

```python
import opendp.prelude as dp

dp.enable_features("contrib")
float_sum = dp.t.make_sum(dp.vector_domain(dp.atom_domain(bounds=(0.0, 5.0))), dp.symmetric_distance())
int_sum = dp.t.make_sum(dp.vector_domain(dp.atom_domain(bounds=(0, 5))), dp.symmetric_distance())
print(float_sum([1.3, 3.8, 0.0, 5.0]))  # 10.1
print(float_sum.map(d_in=2))  # 10.00000004656613
print(int_sum.map(d_in=2))  # 10
print(int_sum.check(d_in=2, d_out=10), float_sum.check(d_in=2, d_out=10.0))  # True False
```

A person who contributes two records of at most 5 can move the sum by 10, and the integer sum's stability map says exactly that. The float sum's map says a little more than 10. Adding floats rounds, and a proof that covers the rounding must give the rounding its share. That extra is tiny, but `check` is a comparison. So the float sum fails at a `d_out` of exactly 10 while the integer sum passes. The lesson is to let the maps produce the distances rather than typing in the number you know from theory. The workflow section shows the same lesson once more, for a scale.

## Chaining a transformation and a measurement {#sec-frame-chain}

A chain is where the framework pays off. Suppose a transformation's output domain and output metric equal a measurement's input domain and input metric. Then the `>>` operator joins them into a new measurement. Its function applies the transformation and then the measurement. Its input domain and metric are the transformation's, and its output measure is the measurement's. Its privacy map is the measurement's map applied to the transformation's map. The script builds the documentation's chain of a clamp, a sum and Laplace noise on the LFS. The vector is the `HWUSUAL` column, the hours a respondent usually works per week. The nulls and the code 99 for "not applicable" are dropped, which leaves 1,496,192 whole numbers of hours from 0 to 80.

```{.python excerpt="frame-chain"}
clamp = dp.t.make_clamp(dp.vector_domain(dp.atom_domain(T=int)), dp.symmetric_distance(), bounds=HOURS_BOUNDS)
bounded_sum = dp.t.make_sum(clamp.output_domain, clamp.output_metric)  # the clamp's output space is its input space
laplace_int = dp.m.make_laplace(bounded_sum.output_domain, bounded_sum.output_metric, scale=LAPLACE_SCALE)
noisy_sum = clamp >> bounded_sum >> laplace_int  # the chain: its input space is the clamp's, its measure Laplace's
sum_sensitivity_1 = bounded_sum.map(d_in=1)  # one row of at most 80 hours
sum_sensitivity_36 = bounded_sum.map(d_in=CONTRIBUTIONS)  # 36 rows
epsilon_at_scale_5 = noisy_sum.map(d_in=CONTRIBUTIONS)
```

Each constructor takes the previous object's output space as its input space, so the domains match by construction. The clamp turns a vector of any integers into a vector of integers in the public bounds 0 to 80, and the sum accepts exactly that domain. The script prints the fields along the chain and the maps.

```text
clamp: input domain=VectorDomain(AtomDomain(T=i32))
clamp: output domain=VectorDomain(AtomDomain(bounds=[0, 80], T=i32))  map(36)=36
sum: output domain=AtomDomain(T=i32)  output metric=AbsoluteDistance(i32)
sum: map(1)=80 map(36)=2880
chain scale 5: input domain=VectorDomain(AtomDomain(T=i32))
chain scale 5: input metric=SymmetricDistance()  output measure=MaxDivergence
chain scale 5: map(36)=576.0
```

Read the distances from left to right. The clamp is 1-stable, since clipping values changes no record's count, so 36 records in give 36 records out. The sum's stability map turns a dataset distance into a sensitivity. One record of at most 80 hours moves the sum by at most 80. A person with 36 records moves it by 36 times as much, which is the 2880 printed. The chain's privacy map is the Laplace map applied to that sensitivity, and at a scale of 5 it says $\varepsilon = 576$. That is no privacy at all. The scale was chosen for one float in the documentation's example, not for a sum over 36 records of 80 hours. The map tells you so before anything is released.

```{.python excerpt="frame-chain-scale"}
sum_scale = dp.binary_search_param(
    lambda scale: clamp >> bounded_sum >> dp.m.then_laplace(scale), d_in=CONTRIBUTIONS, d_out=EPSILON
)
noisy_sum = clamp >> bounded_sum >> dp.m.then_laplace(sum_scale)
sum_accuracy = dp.discrete_laplacian_scale_to_accuracy(scale=sum_scale, alpha=0.05)
released_sum = noisy_sum(hours)
```

```text
chain for epsilon 1 at d_in 36: scale=2880.0 map(36)=1.0
chain for epsilon 1 at d_in 36: 95% accuracy=8,628.2 (0.01532% of the sum)
```

The right scale comes from the map, not from arithmetic. The utility `binary_search_param` takes a function from a scale to a chain. It searches for the largest scale whose `check` passes at the given distances, and it returns 2880, the sensitivity over $\varepsilon = 1$. Here `then_laplace` replaces `make_laplace`, since a `then_` constructor takes its input space from the object before it in the chain. The sum is an integer, so the noise is discrete Laplace and the error bar comes from the discrete accuracy function. An error of about 8,600 hours on a sum of 56 million hours is tiny. Sums over large datasets are the easy case in differential privacy. The release itself is below, in the block of one run. [Transformations](09-transformations.md) works out the stability of each transformation the library offers, and [Measurements](10-measurements.md) does the same for the mechanisms.

## The typical workflow in both APIs {#sec-frame-workflow}

The Context API is a layer over the framework, so the first chapter's five steps can be written in either. The script does both on the same hundred toy scores as [A first release](../part01-getting-started/01-first-release.md), drawn by the same seeded generator. So the scales can be compared with the ones you saw there. The first two steps are the privacy unit and the budget. The framework states the first as a distance and a metric, and the second as a distance and a measure.

```{.python excerpt="frame-unit-loss"}
d_in = 1  # neighbouring datasets differ by at most one row...
input_metric = dp.symmetric_distance()  # ...counted as additions and removals
input_domain = dp.vector_domain(dp.atom_domain(T=float))  # a vector of floats, NaN allowed
d_out = EPSILON  # the distributions of two releases differ by at most epsilon...
privacy_measure = dp.max_divergence()  # ...measured by the max divergence, which is pure DP
```

```text
unit: d_in=1 input metric=SymmetricDistance()  (Context API: (SymmetricDistance(), 1))
loss: d_out=1.0 output measure=MaxDivergence  (Context API: (MaxDivergence, 1.0))
input domain: VectorDomain(AtomDomain(nan=true, T=f64))
```

The pairs that `dp.unit_of` and `dp.loss_of` returned in the first chapter are these same objects. The framework also asks for the input domain up front, which the context inferred from the list you handed it. The domain allows NaN because a list of floats may contain one. The third step, the public bounds and the imputed value, is identical in both. The fourth step is to mediate access, and here the framework shows what a context is made of.

```{.python excerpt="frame-mediate"}
compositor = dp.c.make_adaptive_composition(
    input_domain=input_domain,
    input_metric=input_metric,
    output_measure=privacy_measure,
    d_in=d_in,
    d_mids=[d_out / QUERIES] * QUERIES,  # the shares of the budget, one per query
)
queryable = compositor(data)  # a measurement; invoking it on the data gives the queryable that answers queries
```

```text
compositor: type=Measurement map(1)=1.0
queryable: type=Queryable
context accountant: type=Measurement d_mids=[0.3333, 0.3333, 0.3333]
```

An **adaptive composition** is itself a measurement. Its privacy map at `d_in` is the sum of the shares, because the budgets of sequential releases add up. That is the composition property that [the full edition's chapter on properties](../../book-full/part03-dp-foundations/02-properties.md) proves. Invoking the composition on the data does not release anything. It returns a **queryable**, an object that holds the data and answers measurements submitted to it, one share at a time. The composition is adaptive because each measurement may depend on the answers before it. The context's `accountant` field is this same object, and its `d_mids` are the same three shares. The fifth step submits the queries, and each is built by hand.

```{.python excerpt="frame-count"}
count = dp.t.make_count(input_domain, input_metric)
count_sensitivity = count.map(d_in)  # adding or removing one row changes a count by 1
count_scale = dp.binary_search_param(
    lambda scale: count >> dp.m.then_laplace(scale), d_in=d_in, d_out=d_out / QUERIES
)
count_measurement = count >> dp.m.then_laplace(count_scale)
dp_count = queryable(count_measurement)
```

```text
count: sensitivity=1  output metric=AbsoluteDistance(i32)
count at scale 3.0: map(1)=0.33333333333333337 check(1, 1/3)=False
count by search: scale=3.0000000000000004 check(1, 1/3)=True
count scale: framework=3.0000 context=3.0000 equal=True
count: 95% accuracy=9.4457
```

The count is a transformation whose stability map gives the sensitivity 1, and the Laplace measurement after it needs a scale. You know from theory that the scale is the sensitivity over a third of $\varepsilon$, which is 3. Typing in 3.0 fails the check, because one over three in floating point lands one rounding step above the share. The binary search returns a scale one rounding step above 3, which passes. It is bit for bit the scale that `context.query().count().laplace().param()` returned in the first chapter, because the context runs the same search. Submitting the measurement to the queryable spends the first share and returns the noisy count. The mean follows the first chapter's recipe, with the released count fixing the size.

```{.python excerpt="frame-mean"}
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
```

```text
mean: output domain=AtomDomain(T=f64)  output metric=AbsoluteDistance(f64)
mean at size 100: sensitivity=1.0000
mean scale at size 100: framework=3.0000 context=3.0000 equal=True
mean at size 100: 95% accuracy=8.9872
mean scale at the released count: equal=True
```

Four transformations chain into one, from a vector of floats with NaN allowed to a single float. The stability of the whole is the maps of the parts composed. At the true size of 100, one row moves the mean by at most 1, so the scale is again 3. The context gives the same scale, both at the true size and at the released count. The scale of the mean changes with the released count, so the page quotes the one at the true size, which is a development-time number on a public toy. A third query takes the last share, and a fourth is refused.

```{.python excerpt="frame-refused"}
try:
    queryable(count_measurement)
    refusal = "released"
except dp.OpenDPException as err:
    refusal = f"{err.variant}: {err.message}"
```

```text
fourth query: FailedFunction: out of queries
```

The queryable raises the library's own exception, with the variant `FailedFunction` and the message "out of queries". The context wrapped the same refusal in a `ValueError` about an exhausted allowance. The queryable also refuses a single query whose map exceeds its share, before it touches the data. What the framework version makes plain is that a context is three objects you can build yourself, namely a space, an adaptive composition over that space, and a queryable. So [Utilities and the Context API inside](12-utilities-and-the-context-api.md) follows a query through the context to show where each is used. Finally the releases of one run, read against their error bars.

```text
released (one run):
  laplace scale 5 on 0.0: 12.84  (within 15.0 with probability 0.95)
  hours sum 56,316,146  (true 56,315,594, within 8,628 with probability 0.95)
  count 99  (true 100, within 9.4 with probability 0.95)
  mean scale 3.0303  mean 50.43  (true 50.93)
  third query, a second count 104
```

The Laplace draw at scale 5 landed within its bar of about 15, and the sum of hours within its bar of about 8,600. The count and the mean landed within theirs. The comparison with the truth is possible only because the LFS sample and the toy scores are public. On real data you would read the error bars, which cost nothing, and release once.

## Summary

| Call or concept | What it does | Rule of thumb |
| --- | --- | --- |
| `Transformation` | A deterministic map from data to data, with an `output_metric` and a stability map | Its output is not private; chain it with a measurement before anything leaves |
| `Measurement` | A randomized map from data to a release, with an `output_measure` and a privacy map | Its `map(d_in)` is the guarantee; read it before releasing |
| `trans(x)`, `trans.invoke(x)` | Evaluate the `function` field | Printing a transformation's output steps outside the guarantee |
| `atom_domain(T, bounds, nan)` | One value of type `T`, optionally bounded, with or without NaN | Noise needs `nan=False`; a sum needs `bounds`; the printed form shows `nan` only when true |
| `vector_domain(domain, size)` | Vectors of a domain, optionally of fixed length | A mean needs the `size`; `member` tests a value by hand |
| `symmetric_distance()` | Additions and removals, with multiplicity | `d_in` is the records one person can add or remove |
| `absolute_distance(T)` | The difference of two numbers | `d_in` is a sensitivity; it appears midway through a chain |
| `insert_delete`, `change_one`, `hamming` | Ordered add/remove, replace-one, ordered replace-one | Pick the one that matches how your data can change |
| `max_divergence()` | Pure differential privacy; `d_out` is $\varepsilon$ | The measure behind `loss_of(epsilon=...)` |
| `zero_concentrated_divergence()` | zCDP; `d_out` is $\rho$ | The Gaussian mechanism's native measure |
| `smoothed_max_divergence()`, fixed | Approximate DP; `d_out` is a privacy profile, or one $(\varepsilon, \delta)$ pair | `make_fix_delta` turns the curve into the pair |
| `obj.map(d_in)`, `obj.check(d_in, d_out)` | The smallest provable `d_out`; whether a proposed one fits | Let the map produce the distance; a hand-typed 3.0 fails by one rounding step |
| `trans >> meas` | A measurement whose map composes the two maps | Domains and metrics must match; `then_` constructors match them for you |
| `binary_search_param` | The largest scale whose chain passes `check` | What the Context API runs for `param()` |
| `make_adaptive_composition` | A measurement whose invocation returns a queryable | A context is this, over a space, plus the queryable |

: The objects and elements of the framework, what each does, and the rule of thumb that goes with it. {#tbl-frame-summary}

## References

- OpenDP documentation: "A Framework to Understand DP", "Differential Privacy with OpenDP", "Core Structures" and "Supporting Elements" in the Framework API user guide, and "API Comparison".
- Gaboardi, Hay and Vadhan, "A Programming Framework for OpenDP" (2020): the framework of domains, metrics, measures, stability maps and privacy maps that the library implements.
- Dwork, McSherry, Nissim and Smith, "Calibrating Noise to Sensitivity in Private Data Analysis" (2006): sensitivity and the Laplace mechanism.
- The full edition's chapters on [the definition](../../book-full/part03-dp-foundations/01-definition.md), [post-processing and composition](../../book-full/part03-dp-foundations/02-properties.md), [the Laplace and Gaussian mechanisms](../../book-full/part03-dp-foundations/03-laplace-gaussian.md) and [sensitivity](../../book-full/part03-dp-foundations/07-sensitivity.md) derive the results this chapter states.
