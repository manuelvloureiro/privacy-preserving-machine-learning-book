# A framework to understand differential privacy

The first two parts used the Context API to choose noise scales, spend budget and reject queries that exceed it. The framework beneath the context has two main objects. A transformation maps data to data and bounds how input distance affects output distance. A measurement maps data to a random release and bounds how input distance affects the output distributions. Each bound is a map from input distance to output distance; OpenDP uses these maps to track differential privacy (DP). This chapter shows how an object's fields and map state its guarantee.

The script [opendp_framework.py](../../code/part03-framework/opendp_framework.py) runs each example on OpenDP 0.16.0. Its sum example uses integer values from the Labour Force Survey (LFS) column `HWUSUAL`; its workflow uses the hundred toy scores from the first chapter. Domains, metrics, measures, maps, scales and error bars are deterministic. Released values vary because OpenDP's noise cannot be seeded, so the page labels them as one run.

```bash
uv run python code/part03-framework/opendp_framework.py
```

## Two kinds of object {#sec-frame-objects}

The first chapter built a Laplace measurement with scale 1. The documentation repeats the construction at scale 5 using `make_laplace`, which takes the domain and metric as its first two arguments rather than attaching them with `>>`.

```{.python excerpt="frame-laplace"}
def laplace_measurement(scale: float) -> dp.Measurement:
    """Laplace noise of the given scale on one finite float, compared by absolute distance, under pure DP."""
    return dp.m.make_laplace(
        dp.atom_domain(T=float, nan=False),  # input domain: any float that is not NaN
        dp.absolute_distance(T=float),  # input metric: |a - b|, the units of the sensitivity
        scale=scale,
    )
```

The constructor returns a `Measurement`; the script prints its fields.

```text
laplace scale 5: type=Measurement
laplace scale 5: input domain=AtomDomain(T=f64)
laplace scale 5: input metric=AbsoluteDistance(f64)
laplace scale 5: output measure=MaxDivergence
laplace scale 5: function=Function
```

Transformations and measurements share four fields. `function` holds the computation, which maps `input_domain` to `output_domain`; `input_metric` measures input distance. Their output fields differ because a **transformation** is deterministic and a **measurement** is random. A transformation returns data, so `output_metric` and the **stability map** describe its output distance as a function of input distance. A measurement returns a random draw, so `output_measure` and the **privacy map** describe the distance between output distributions.

The Laplace measurement accepts a finite float, measures input differences by absolute distance and compares output distributions by max divergence, bounded by $\varepsilon$. Both object guarantees have the same form: a transformation is (`d_in`, `d_out`)-stable if inputs at most `d_in` apart produce outputs at most `d_out` apart; a measurement is (`d_in`, `d_out`)-differentially private if such inputs produce output distributions at most `d_out` apart.

## Functions {#sec-frame-functions}

Call a transformation or measurement like a function; `invoke` does the same. The clamp below clips each finite float in a vector to the interval [0, 5], under symmetric distance.

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

The `function` field stores the computation as an object; the next part's plugin chapter uses it to wrap a Python function. A transformation adds no noise and does not make its output private. It prepares data within the guarantee; only a measurement can release data. Printing a transformation's output is outside that guarantee.

## Domains {#sec-frame-domains}

A domain is a set of values. A map's proof applies only to inputs in that domain, so the domain states its assumptions. When chaining objects, OpenDP checks that the first object's output domain matches the next object's input domain. The common `atom_domain` contains one value of type `T`; `vector_domain` lifts it to vectors.

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

`member` tests whether a value belongs to a domain. Type `T` uses Python names, so `float` prints as `f64` and `int` as `i32`. The `nan` flag controls whether floats may be NaN; version 0.16.0 prints it only when true. Thus `atom_domain(T=float)` allows NaN, while `atom_domain(T=float, nan=False)` rejects it. A noise measurement needs the latter because absolute distance from NaN is undefined.

`bounds` restrict values to an interval. A sum needs bounds to have finite sensitivity. A vector domain can also specify `size`, which a mean needs, as the first chapter's resize demonstrated.

## Metrics {#sec-frame-metrics}

A metric defines distance between members of a domain. The framework uses it to interpret `d_in`; it does not compute the distance. Under **symmetric distance**, two datasets are separated by the number of additions and removals needed to transform one into the other, counting duplicates. The function below computes this distance for two small vectors from the documentation's notebook.

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

Removing one record gives distance 1. Replacing 12 with 10 gives distance 2: one removal and one addition. Under symmetric distance, `d_in` is the maximum records one person can add or remove, which defines the privacy unit. Datasets $D$ and $D'$ are **neighbours** when their input-metric distance is at most `d_in`. This is the metric behind `dp.unit_of(contributions=k)`. In the LFS, one person contributes at most 36 rows ([the privacy unit](../part01-getting-started/02-the-data-and-the-context.md#sec-ctx-context)), so `d_in=36`.

**Absolute distance** is the absolute difference between two numbers. Under this metric, `d_in` is a **sensitivity**: the maximum change in an aggregate when one person's records change. Sensitivity usually appears in the middle of a chain as an aggregate's output distance, rather than as an input you specify. The full edition's [chapter on sensitivity](../../book-full/part03-dp-foundations/07-sensitivity.md) derives common bounds.

Other dataset metrics modify symmetric distance. Insert-delete distance also protects record order. Change-one distance counts replacements in fixed-size datasets, the replace-one relation of bounded DP. Hamming distance is the ordered version of change-one.

## Measures {#sec-frame-measures}

A measure defines distance between probability distributions and gives a measurement's `d_out` its privacy meaning. **Max divergence** is the largest log ratio between the probabilities of an output set under the two distributions. A bound of $\varepsilon$ gives pure differential privacy, defined in [the full edition's chapter on the definition](../../book-full/part03-dp-foundations/01-definition.md). OpenDP provides four measures, each with a different form for `d_out`.

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

Under max divergence, `d_out` is $\varepsilon$. Under **zero-concentrated divergence**, it is $\rho$, the zCDP parameter that bounds a family of Rényi divergences and is the Gaussian mechanism's native measure. At sensitivity 1 and scale 1, a Gaussian mechanism costs $\rho=0.5$, equal to sensitivity squared divided by twice the scale squared.

The two smoothed measures express approximate DP, whose guarantee allows slack $\delta$. Under smoothed max divergence, `d_out` is a **privacy profile**, a curve giving the smallest $\varepsilon$ for each $\delta$; the example evaluates it at $10^{-7}$. Fixed smoothed max divergence, printed as `Approximate(MaxDivergence)`, gives one $(\varepsilon,\delta)$ pair, as in the first chapter's `dp.loss_of(epsilon=1.0, delta=1e-7)`. `make_fix_delta` converts a profile to this pair. [Combinators](11-combinators.md) explains the conversions. Use `output_measure` to interpret a measurement's `d_out`.

## Stability maps and privacy maps {#sec-frame-maps}

A map states the guarantee by converting input distance to output distance. For a measurement, `map(d_in)` returns the smallest provable `d_out`, so the measurement is (`d_in`, `d_out`)-differentially private. For a transformation, it returns the smallest `d_out` that makes it (`d_in`, `d_out`)-stable. `check` tests whether a proposed `d_out` meets the guarantee. The script evaluates a Laplace map at two sensitivities.

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

The Laplace privacy map divides sensitivity by scale. Sensitivity 1 at scale 5 costs $\varepsilon=0.2$; ten times the sensitivity costs ten times more. Thus `check` passes at 0.2 and fails at 0.19. The Context API searches for a scale by testing whether `check` passes at the query's budget share. The full edition derives the map in [its chapter on the Laplace and Gaussian mechanisms](../../book-full/part03-dp-foundations/03-laplace-gaussian.md).

A map returns a provable upper bound, which may exceed the true value. The sum below illustrates this difference.

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

Two records bounded by 5 can change the sum by 10, and the integer sum's map returns exactly 10. Floating-point addition rounds, so its proof includes a small extra margin. Because `check` uses strict comparison, the float sum fails at exactly 10. Use the map's result rather than entering the theoretical value yourself.

## Chaining a transformation and a measurement {#sec-frame-chain}

When a transformation's output domain and metric match a measurement's input domain and metric, `>>` chains them into a measurement. It applies the transformation first, then the measurement; the combined privacy map composes their maps. The script chains clamping, summation and Laplace noise for `HWUSUAL`, usual weekly hours. After removing nulls and code 99 ("not applicable"), the data contain 1,496,192 integer values from 0 to 80.

```{.python excerpt="frame-chain"}
clamp = dp.t.make_clamp(dp.vector_domain(dp.atom_domain(T=int)), dp.symmetric_distance(), bounds=HOURS_BOUNDS)
bounded_sum = dp.t.make_sum(clamp.output_domain, clamp.output_metric)  # the clamp's output space is its input space
laplace_int = dp.m.make_laplace(bounded_sum.output_domain, bounded_sum.output_metric, scale=LAPLACE_SCALE)
noisy_sum = clamp >> bounded_sum >> laplace_int  # the chain: its input space is the clamp's, its measure Laplace's
sum_sensitivity_1 = bounded_sum.map(d_in=1)  # one row of at most 80 hours
sum_sensitivity_36 = bounded_sum.map(d_in=CONTRIBUTIONS)  # 36 rows
epsilon_at_scale_5 = noisy_sum.map(d_in=CONTRIBUTIONS)
```

Each constructor uses the preceding object's output space as its input, so domains match by construction. The clamp outputs integers from 0 to 80, which the sum accepts. The script prints the fields and maps along the chain.

```text
clamp: input domain=VectorDomain(AtomDomain(T=i32))
clamp: output domain=VectorDomain(AtomDomain(bounds=[0, 80], T=i32))  map(36)=36
sum: output domain=AtomDomain(T=i32)  output metric=AbsoluteDistance(i32)
sum: map(1)=80 map(36)=2880
chain scale 5: input domain=VectorDomain(AtomDomain(T=i32))
chain scale 5: input metric=SymmetricDistance()  output measure=MaxDivergence
chain scale 5: map(36)=576.0
```

Read the maps from left to right. Clamping is 1-stable because it changes values without adding or removing records; 36 input records produce 36 output records. The sum changes by at most 80 hours per record, so 36 records give sensitivity 2,880. Laplace noise at scale 5 then gives $\varepsilon=576$, an inadequate guarantee. The map exposes this problem before release; scale 5 was chosen for one float.

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

`binary_search_param` finds the smallest scale whose chained measurement passes `check`; here it returns 2,880, sensitivity divided by $\varepsilon=1$. The chain uses `then_laplace`, whose input space comes from the preceding object. The integer sum receives discrete Laplace noise and uses discrete accuracy. An error of about 8,600 hours is small relative to 56 million. [Transformations](09-transformations.md) and [Measurements](10-measurements.md) describe the other maps.

## The typical workflow in both APIs {#sec-frame-workflow}

The Context API wraps the framework. The script implements the first chapter's five steps in both APIs using the same seed and hundred toy scores from [A first release](../part01-getting-started/01-first-release.md), allowing a direct scale comparison. The framework represents the privacy unit as an input distance and metric, and the budget as an output distance and measure.

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

These pairs match the first chapter's `dp.unit_of` and `dp.loss_of` outputs. The framework also requires an input domain. The context inferred a vector of floats from the list, which allows NaN. Both APIs use the same public bounds and imputed value. The fourth step mediates access and exposes the context's components.

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

An **adaptive composition** is a measurement whose privacy map sums the shares, because sequential privacy budgets add ([the full edition's chapter on properties](../../book-full/part03-dp-foundations/02-properties.md)). Invoking it on data releases nothing; it returns a **queryable** that holds the data and answers submitted measurements, one share each. The composition is adaptive because each measurement can depend on earlier answers. A context's `accountant` field stores the same composition and shares. The fifth step submits manually built queries.

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

The count's stability map gives sensitivity 1, so theory predicts scale $1/(\varepsilon/3)=3$. However, `check` rejects typed scale 3.0 because floating-point $1/3$ is one rounding step above the budget share. The search returns the next representable scale above 3, which passes. It matches `context.query().count().laplace().param()` bit for bit because the context uses the same search. The queryable spends the first share on the count; the mean uses that released count as its size.

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

The four transformations form one chain whose stability composes their maps. At the true size of 100, one row changes the mean by at most 1, giving scale 3, as in the context. The page reports this scale because the scale at the noisy released count varies by run. A third query uses the final share; the queryable rejects a fourth.

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

The queryable raises OpenDP's exception; the context wraps the same refusal in a `ValueError`. It also rejects a query whose map exceeds its share before accessing data. A context therefore consists of three objects: a space, an adaptive composition over it and a queryable. [Utilities and the Context API inside](12-utilities-and-the-context-api.md) shows how a query uses them. The next block reports one run.

```text
released (one run):
  laplace scale 5 on 0.0: 12.84  (within 15.0 with probability 0.95)
  hours sum 56,316,146  (true 56,315,594, within 8,628 with probability 0.95)
  count 99  (true 100, within 9.4 with probability 0.95)
  mean scale 3.0303  mean 50.43  (true 50.93)
  third query, a second count 104
```

Each release falls within its error bar. The LFS sample and toy scores are public, so their outputs can be compared with the truth. For private data, inspect error bars, which incur no cost, and release once.

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
