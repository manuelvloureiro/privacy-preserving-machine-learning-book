# A framework to understand differential privacy

Every query of the first two parts used the Context API, which chose a noise scale, spent a share of the budget and refused a query above it. Under that API, OpenDP describes a private computation with two kinds of object. A transformation maps data to data, and it promises how far apart its outputs can be when its inputs are close. A measurement maps data to a random release, and it promises how far apart the distributions of its releases can be. Each promise is a map from an input distance to an output distance, and differential privacy (DP) in OpenDP is the bookkeeping of such maps. At the end of the chapter, you can read the guarantee of any OpenDP object from its fields and its map.

The script [opendp_framework.py](../../code/part03-framework/opendp_framework.py) runs every step on OpenDP 0.16.0. Its sum example uses the `HWUSUAL` column of the Labour Force Survey (LFS) sample as a list of integers, and its workflow uses the first chapter's hundred toy scores. The domains, metrics, measures, maps, scales and error bars are the same on every run. The released values change, because OpenDP's noise cannot be seeded, so the page shows them in one block marked as one run.

```bash
uv run python code/part03-framework/opendp_framework.py
```

## Two kinds of object {#sec-frame-objects}

The first chapter built a Laplace measurement without saying what came back. The documentation's example is the same mechanism with a scale of 5. It uses `make_laplace`, which takes the domain and the metric as its first two arguments instead of through `>>`.

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

A transformation and a measurement share four fields. The `function` is the computation, it accepts the `input_domain` and produces the `output_domain`, and the `input_metric` gives the distance between two inputs. The last two fields differ, because a **transformation** is deterministic while a **measurement** is random. A transformation's output is data, so it has an `output_metric` and a **stability map**, which bounds the distance between two outputs from a bound on the distance between the inputs. A measurement's output is a random draw, so it has an `output_measure` and a **privacy map**, which bounds the distance between two output distributions.

So the Laplace measurement accepts one finite float, compares two inputs by their absolute difference, and compares two output distributions by the max divergence, which $\varepsilon$ bounds. The documentation states the two promises in one shape. A transformation is (`d_in`, `d_out`)-stable when inputs at most `d_in` apart give outputs at most `d_out` apart. A measurement is (`d_in`, `d_out`)-differentially private when the same inputs give output distributions at most `d_out` apart. The sections below take the parts of these sentences in turn.

## Functions {#sec-frame-functions}

The function is the part you would write anyway. You call a transformation or a measurement like a function, and `invoke` does the same. The clamp below works on vectors of finite floats under the symmetric distance, and it clips every element into the bounds 0 to 5.

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

The `function` field holds the computation as an object, which the plugin chapter of the next part uses to wrap a Python function. A transformation adds no noise, so its result is not private. This is by design: a transformation prepares the data inside the guarantee, and only a measurement may release anything. If you print the output of a transformation, you are outside the guarantee.

## Domains {#sec-frame-domains}

A domain is a set of values, and the library uses it in two ways. The proof behind a map holds only for inputs in the domain, so the domain states what the map assumes. And when you chain two objects, the library checks that the output domain of the first equals the input domain of the second. The most common domain is `atom_domain`, one value of a type `T`, and `vector_domain` lifts it to vectors.

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

The `member` method tests whether a value is in the domain, so you can check an assumption by hand. The type `T` is written in Python terms, so `float` prints as `f64` and `int` as `i32`. The `nan` flag says whether a float may be NaN, and the installed version prints it only when it is true. So `atom_domain(T=float)` allows NaN and `atom_domain(T=float, nan=False)` does not. A measurement that adds noise needs the second, because the absolute distance to a NaN is not defined.

The `bounds` restrict the values to an interval, and the sum needs them, because a sum of unbounded values has no finite sensitivity. A vector domain can also have a `size`, which the mean needs, as the first chapter's resize showed.

## Metrics {#sec-frame-metrics}

A metric is a rule for the distance between two members of a domain. The framework never computes it, and only uses it to give `d_in` its meaning. Under the **symmetric distance**, the distance between two datasets is the number of additions and removals that turn one into the other, counted with multiplicity. The function below computes it for two small vectors from the documentation's notebook, where the second lacks the first person's record.

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

If you remove one record, the distance is 1. If you change a 12 into a 10, the distance is 2, because a change is one removal and one addition. So under the symmetric distance, `d_in` is the largest number of records that one person can add or remove, which is the privacy unit. Two datasets are **neighbouring**, written $D \sim D'$, when they are at most `d_in` apart under the input metric, here add/remove. This is the metric behind `dp.unit_of(contributions=k)`, so on the LFS, where a person has at most 36 rows ([the privacy unit](../part01-getting-started/02-the-data-and-the-context.md#sec-ctx-context)), `d_in` is 36.

The **absolute distance** is the absolute value of the difference of two numbers. Under it, `d_in` is a **sensitivity**, the most an aggregate can change when one person's records change. So you rarely give a sensitivity yourself, because it appears in the middle of a chain, as the output distance of an aggregate. The full edition's [chapter on sensitivity](../../book-full/part03-dp-foundations/07-sensitivity.md) works out the common ones.

The other dataset metrics change the symmetric distance in one way each. The insert-delete distance also protects the order of the records. The change-one distance counts replaced records in datasets of a fixed size, the replace-one relation of bounded DP. The Hamming distance is the ordered version of change-one.

## Measures {#sec-frame-measures}

A measure is a rule for the distance between two probability distributions, and it is what makes a measurement's `d_out` a privacy guarantee. The **max divergence** is the largest log ratio of the two probabilities of a set of outputs, over every such set. A bound of $\varepsilon$ on it is pure differential privacy, which the full edition defines in [its chapter on the definition](../../book-full/part03-dp-foundations/01-definition.md). The library has four measures, and `d_out` has a different form under each.

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

Under the max divergence, `d_out` is one number, $\varepsilon$. Under the **zero-concentrated divergence**, `d_out` is the parameter $\rho$ of zero-concentrated differential privacy (zCDP), which bounds a family of Rényi divergences at once and is the native measure of the Gaussian mechanism. A Gaussian of scale 1 at sensitivity 1 costs $\rho = 0.5$, the sensitivity squared over twice the scale squared.

The two smoothed measures carry approximate DP, whose guarantee holds up to a slack $\delta$. Under the smoothed max divergence, `d_out` is a **privacy profile**, a curve that gives the smallest $\varepsilon$ for any $\delta$, here $10^{-7}$. Under the fixed smoothed max divergence, printed `Approximate(MaxDivergence)`, `d_out` is one pair of $\varepsilon$ and $\delta$, like the first chapter's `dp.loss_of(epsilon=1.0, delta=1e-7)`. The combinator `make_fix_delta` turns the curve into the pair. [Combinators](11-combinators.md) covers the conversions. Here, remember that a measurement's `output_measure` tells you how to read its `d_out`.

## Stability maps and privacy maps {#sec-frame-maps}

A map turns an input distance into an output distance, and it holds the guarantee. For a measurement, the call `map(d_in)` returns the smallest `d_out` that the library can prove, so the measurement is (`d_in`, `d_out`)-differentially private. For a transformation, it returns the smallest `d_out` for which it is (`d_in`, `d_out`)-stable. The `check` method says whether the guarantee fits inside a `d_out` you propose. The script reads the map of the Laplace measurement at two sensitivities.

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

The Laplace privacy map is the sensitivity divided by the scale. So a sensitivity of 1 at scale 5 costs $\varepsilon = 0.2$, and ten times the sensitivity costs ten times more. So the check at 0.2 passes and the check at 0.19 fails. The Context API uses only this comparison when it finds a scale: it tries scales until `check` passes at the query's share of the budget. The full edition derives the map in [its chapter on the Laplace and Gaussian mechanisms](../../book-full/part03-dp-foundations/03-laplace-gaussian.md).

A map returns an upper bound that the library can prove, which is not always the smallest true one, and the sum shows why this matters.

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

A person who adds two records of at most 5 can move the sum by 10, and the integer sum's map says exactly 10. The float sum's map says a little more, because adding floats rounds, and the proof adds a margin for the rounding. The margin is tiny, but `check` is a strict comparison, so the float sum fails at exactly 10. So let the maps give you the distances, and do not type in the number you know from theory.

## Chaining a transformation and a measurement {#sec-frame-chain}

When a transformation's output domain and metric equal a measurement's input domain and metric, the `>>` operator joins them into a new measurement. The new measurement applies the transformation and then the measurement. Its privacy map applies the measurement's map to the output of the transformation's map. The script builds the documentation's chain of a clamp, a sum and Laplace noise on `HWUSUAL`, the hours a respondent usually works per week. Without the nulls and the code 99 for "not applicable", that is 1,496,192 whole numbers of hours from 0 to 80.

```{.python excerpt="frame-chain"}
clamp = dp.t.make_clamp(dp.vector_domain(dp.atom_domain(T=int)), dp.symmetric_distance(), bounds=HOURS_BOUNDS)
bounded_sum = dp.t.make_sum(clamp.output_domain, clamp.output_metric)  # the clamp's output space is its input space
laplace_int = dp.m.make_laplace(bounded_sum.output_domain, bounded_sum.output_metric, scale=LAPLACE_SCALE)
noisy_sum = clamp >> bounded_sum >> laplace_int  # the chain: its input space is the clamp's, its measure Laplace's
sum_sensitivity_1 = bounded_sum.map(d_in=1)  # one row of at most 80 hours
sum_sensitivity_36 = bounded_sum.map(d_in=CONTRIBUTIONS)  # 36 rows
epsilon_at_scale_5 = noisy_sum.map(d_in=CONTRIBUTIONS)
```

Each constructor takes the output space of the object before it as its input space, so the domains match by construction: the clamp outputs integers in 0 to 80, which the sum accepts. The script prints the fields and the maps along the chain.

```text
clamp: input domain=VectorDomain(AtomDomain(T=i32))
clamp: output domain=VectorDomain(AtomDomain(bounds=[0, 80], T=i32))  map(36)=36
sum: output domain=AtomDomain(T=i32)  output metric=AbsoluteDistance(i32)
sum: map(1)=80 map(36)=2880
chain scale 5: input domain=VectorDomain(AtomDomain(T=i32))
chain scale 5: input metric=SymmetricDistance()  output measure=MaxDivergence
chain scale 5: map(36)=576.0
```

Read the distances from left to right. The clamp is 1-stable, because clipping adds or removes no record, so 36 records in give 36 out. The sum's map turns that into a sensitivity: one record moves the sum by at most 80 hours, so 36 records move it by at most 2880. The Laplace map at a scale of 5 then gives $\varepsilon = 576$, which is no privacy at all. That scale was chosen for one float, and the map shows the problem before anything is released.

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

The utility `binary_search_param` gets the right scale from the map. It takes a function from a scale to a chain and finds the smallest scale whose `check` passes, here 2880, the sensitivity over $\varepsilon = 1$. The chain uses `then_laplace`, because a `then_` constructor takes its input space from the object before it. The sum is an integer, so the noise is discrete Laplace, with the discrete accuracy function. An error of about 8,600 hours on 56 million is tiny, because a sum over a large dataset is the easy case. [Transformations](09-transformations.md) and [Measurements](10-measurements.md) give the maps of the other pieces.

## The typical workflow in both APIs {#sec-frame-workflow}

The Context API is a layer over the framework, so you can write the first chapter's five steps in either API. The script does both on the hundred toy scores of [A first release](../part01-getting-started/01-first-release.md), from the same seed, so the scales are comparable. The framework states the privacy unit as a distance and a metric, and the budget as a distance and a measure.

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

These are the same pairs that `dp.unit_of` and `dp.loss_of` returned in the first chapter. The framework also asks for the input domain, which the context found from your list, and it allows NaN because a list of floats can hold one. The public bounds and the imputed value are the same in both APIs. The fourth step mediates access, and it shows what a context is made of.

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

An **adaptive composition** is itself a measurement, whose privacy map is the sum of the shares, because the budgets of sequential releases add up ([the full edition's chapter on properties](../../book-full/part03-dp-foundations/02-properties.md)). When you invoke it on the data, it releases nothing. It returns a **queryable**, which holds the data and answers the measurements you submit, one share each. It is adaptive because each measurement may depend on the earlier answers. A context's `accountant` field is this same object, with the same shares. The fifth step submits the queries, each built by hand.

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

The count's stability map gives the sensitivity 1, so from theory the Laplace scale is 1 over a third of $\varepsilon$, which is 3. But a typed 3.0 fails the check, because one over three in floating point is one rounding step above the share. The search returns a scale one rounding step above 3, which passes. It is bit for bit the scale that `context.query().count().laplace().param()` gave in the first chapter, because the context runs the same search. The queryable spends the first share on the count. The mean uses the released count as its size, as in the first chapter.

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

The four transformations chain into one, and its stability is the maps of the parts composed. At the true size of 100, one row moves the mean by at most 1, so the scale is again 3, the same as the context's. The page quotes the scale at the true size, because the scale at the released count changes from run to run. A third query takes the last share, and the queryable refuses a fourth.

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

The queryable raises the library's own exception, which the context wrapped in a `ValueError`. It also refuses a query whose map is more than its share, before it touches the data. So a context is three objects you can build yourself: a space, an adaptive composition over it, and a queryable. [Utilities and the Context API inside](12-utilities-and-the-context-api.md) shows where a query uses each. Here are the releases of one run.

```text
released (one run):
  laplace scale 5 on 0.0: 12.84  (within 15.0 with probability 0.95)
  hours sum 56,316,146  (true 56,315,594, within 8,628 with probability 0.95)
  count 99  (true 100, within 9.4 with probability 0.95)
  mean scale 3.0303  mean 50.43  (true 50.93)
  third query, a second count 104
```

Each release landed within its error bar. You can compare them with the truth only because the LFS sample and the toy scores are public. On real data, read the error bars, which cost nothing, and release once.

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
| `binary_search_param` | The smallest scale whose chain passes `check` | What the Context API runs for `param()` |
| `make_adaptive_composition` | A measurement whose invocation returns a queryable | A context is this, over a space, plus the queryable |

: The objects and elements of the framework, what each does, and the rule of thumb that goes with it. {#tbl-frame-summary}

## References

- OpenDP documentation: "A Framework to Understand DP", "Differential Privacy with OpenDP", "Core Structures" and "Supporting Elements" in the Framework API user guide, and "API Comparison".
- Gaboardi, Hay and Vadhan, "A Programming Framework for OpenDP" (2020): the framework of domains, metrics, measures, stability maps and privacy maps that the library implements.
- Dwork, McSherry, Nissim and Smith, "Calibrating Noise to Sensitivity in Private Data Analysis" (2006): sensitivity and the Laplace mechanism.
- The full edition's chapters on [the definition](../../book-full/part03-dp-foundations/01-definition.md), [post-processing and composition](../../book-full/part03-dp-foundations/02-properties.md), [the Laplace and Gaussian mechanisms](../../book-full/part03-dp-foundations/03-laplace-gaussian.md) and [sensitivity](../../book-full/part03-dp-foundations/07-sensitivity.md) derive the results this chapter states.
