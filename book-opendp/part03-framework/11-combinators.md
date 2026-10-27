# Combinators

A real release is a clamp, then a sum, then noise, then a second query that depends on the first. It is reported as one guarantee of differential privacy (DP). The **combinators** join such pieces into bigger ones. Each takes objects of the framework and returns a new one, with a privacy map that gives its cost. So you never add budgets by hand, and the combinator refuses to join pieces whose promises do not fit. At the end of the chapter, you can read the privacy map of any chain and say why a context refuses a query. You can also convert a budget between privacy measures.

The script [opendp_combinators.py](../../code/part03-framework/opendp_combinators.py) runs every step on OpenDP 0.16.0. As in the documentation, its toy data are twenty seeded integers from 0 to 10, with a Laplace count and a Laplace sum of the clamped values. One example runs on the Labour Force Survey (LFS) sample. The maps, profiles, scales and error messages are the same on every run, but the releases change, because OpenDP's noise cannot be seeded. So they are in one block marked as one run.

```bash
uv run python code/part03-framework/opendp_combinators.py
```

## Chaining {#sec-comb-chaining}

The simplest way to combine two pieces is to run one after the other. The constructor `make_chain_tt` joins two transformations, `make_chain_mt` puts a measurement after a transformation, and `make_chain_pm` attaches a post-processing function to a measurement. There is no chainer for two measurements, because a computation after a measurement is post-processing, which needs no accounting. The script first builds the clamped sum and its Laplace release with these long forms.

```{.python excerpt="comb-chain-long"}
def chained_by_hand() -> tuple[dp.Transformation, dp.Measurement]:
    """The clamped sum and its Laplace release, built with the long-form chainers."""
    clamp = INT_SPACE >> dp.t.then_clamp(BOUNDS)
    total = dp.t.make_sum(clamp.output_domain, clamp.output_metric)
    clamped_sum = dp.c.make_chain_tt(total, clamp)  # a transformation: clamp, then sum
    laplace = dp.m.make_laplace(clamped_sum.output_domain, clamped_sum.output_metric, scale=SUM_SCALE)
    noisy_sum = dp.c.make_chain_mt(laplace, clamped_sum)  # a measurement: the chain, then noise
    return clamped_sum, noisy_sum
```

Each call takes the outer piece first and the inner piece second, as in function composition. The chained transformation has a stability map ([Transformations](09-transformations.md)). One added or removed record changes the clamped sum by at most the width of the bounds, so the map at distance 1 is 10. The chained measurement has a privacy map ([Measurements](10-measurements.md)), which is that stability over the noise scale. So Laplace noise of scale 5 on the sum costs a privacy loss of $\varepsilon = 2$.

```text
clamp >> sum: map(1)=10
clamp >> sum >> laplace: map(1)=2.0 check(1, 2.0)=True
with >>: map(1)=2.0
```

The `>>` operator chooses the right chainer for you, and with a `then_` constructor on its right you never repeat the input space. Here is the same measurement in one line.

```python
import opendp.prelude as dp

dp.enable_features("contrib")
space = dp.vector_domain(dp.atom_domain(T=int)), dp.symmetric_distance()
noisy_sum = space >> dp.t.then_clamp((0, 10)) >> dp.t.then_sum() >> dp.m.then_laplace(scale=5.0)
print(noisy_sum.map(1))  # 2.0
print(noisy_sum.input_domain)  # VectorDomain(AtomDomain(T=i32))
print(noisy_sum.output_measure)  # MaxDivergence
```

**Post-processing** is any computation on a release that does not touch the data again. It costs nothing, because what an adversary could learn from the processed value, they could learn from the release itself ([the full edition's chapter on properties](../../book-full/part03-dp-foundations/02-properties.md)). In OpenDP you attach a Python function to a measurement with `>>`, and the result has the same privacy map.

```{.python excerpt="comb-postprocess"}
def divide_by_size(total: int) -> float:
    return total / N  # the released sum over the public size: a mean, at no further cost

mean_of_sum = shortcut >> divide_by_size
```

```text
post-processed: map(1)=2.0
```

In version 0.16.0 this needs a second feature flag, `honest-but-curious`, because the library cannot inspect a Python function. The script switches it on just before the chain and off just after. A post-processor must not read global state, such as the clock, or raise an exception that depends on the data. Either opens a channel that the privacy map does not count.

A chain forms only when the pieces fit: the output domain and metric of the inner piece must equal the input domain and metric of the outer one. The script gives a sum of floats to a Laplace mechanism built for integers, and the library refuses.

```{.python excerpt="comb-mismatch"}
try:
    chain = float_sum >> int_laplace  # a float sum feeding a Laplace mechanism on integers
    domain_error = f"chained: {chain}"
except dp.OpenDPException as err:
    domain_error = str(err.message)
```

```text
domain mismatch: Intermediate domains don't match. See https://github.com/opendp/opendp/discussions/297
    output_domain: AtomDomain(T=f64)
    input_domain:  AtomDomain(T=i32)
metric mismatch: Intermediate metrics don't match. See https://github.com/opendp/opendp/discussions/297
    output_metric: AbsoluteDistance(i32)
    input_metric:  AbsoluteDistance(f64)
```

The second message comes from a count, whose output is an integer distance, before a Laplace mechanism that measures distances in floats. So read the two lines of the error and insert the cast, clamp or impute step that makes them equal.

Here are the error bars of the sum and the count that the rest of the chapter composes ([Assessing utility and choosing the noise](../part02-tabular-data/07-utility-and-noise.md) explains them).

```text
sum: scale=5 95% accuracy=15.4537
count: scale=1 95% accuracy=3.3756
```

## Non-adaptive composition {#sec-comb-composition}

When you run the sum and the count on the same data, you spend both budgets. **Basic composition** says that the total is the sum of the parts. So an $\varepsilon_1$-DP release and an $\varepsilon_2$-DP release on the same data are together $(\varepsilon_1 + \varepsilon_2)$-DP ([the full edition's chapter on properties](../../book-full/part03-dp-foundations/02-properties.md)). The combinator `make_composition` builds that rule into one measurement, which runs every part and returns the releases as a list.

```{.python excerpt="comb-composition"}
both = dp.c.make_composition([shortcut, count])  # one measurement that releases the sum and the count
print(f"composition: map(1)={both.map(1):.1f}  parts: {shortcut.map(1):.1f} + {count.map(1):.1f}")
```

```text
composition: map(1)=3.0  parts: 2.0 + 1.0
composition output measure: MaxDivergence
```

The parts must agree on their input domain, input metric and output measure. A composition is a statement about one dataset, one notion of neighbours and one measure of privacy. It is non-adaptive, because every part is fixed before the data are seen.

The same combinator composes measurements under other measures, with each measure's own rule. **Zero-concentrated differential privacy (zCDP)** measures privacy by one parameter $\rho$, which bounds the Rényi divergence between the output distributions on neighbours at every order. Under zCDP the parameters add like $\varepsilon$ ([the full edition's chapter on concentrated DP](../../book-full/part04-privacy-accounting/03-concentrated-dp.md)). The Gaussian sum at scale 5 has $\rho = 2$, the Gaussian count at scale 1 has $\rho = 0.5$, and the composition reports their sum.

```text
zCDP composition: map(1)=2.50  parts: 2.00 + 0.50
mixed measures: All output measures must be the same
```

The second line shows that you cannot mix measures, because the maps of a Laplace count and a Gaussian count return numbers that mean different things. @sec-comb-casting shows how to bring them to one measure first.

## Adaptive composition {#sec-comb-adaptive}

Often you choose the second query only after you see the first answer. Composition still holds when each query depends on the earlier answers, if the budget of each is fixed in advance ([the full edition's chapter on composition](../../book-full/part03-dp-foundations/04-composition.md)). The combinator for this is `make_adaptive_composition`, which needs more arguments, because it cannot read the supporting elements from parts it has not seen.

```{.python excerpt="comb-adaptive"}
def adaptive_compositor(d_mids: list[float]) -> dp.Measurement:
    """An adaptive compositor over the toy space: one share of pure-DP budget per query, fixed in advance."""
    return dp.c.make_adaptive_composition(
        input_domain=INT_SPACE[0],
        input_metric=INT_SPACE[1],
        output_measure=dp.max_divergence(),
        d_in=1,  # neighbours add or remove one record
        d_mids=d_mids,  # the epsilon each query may spend, in the order the queries arrive
    )
```

The argument `d_in` is the distance between neighbouring datasets, here 1 for one added or removed record. The argument `d_mids` lists the privacy loss that each query may spend, in order, and the compositor's privacy map is their sum, here 3.

```text
adaptive: map(1)=3.0
```

When you invoke the compositor on the data, it releases nothing. It returns a **queryable**, a small state machine that holds the data and the remaining shares. It accepts a measurement as a query, checks that the measurement's map at `d_in` fits within the next share, and answers with the release. The script submits the sum, then the sum again, then the count.

```{.python excerpt="comb-queryable"}
queryable = compositor(data)  # the data go in once; what comes back answers measurements
adaptive_sum = queryable(shortcut)  # spends the first share, 2.0
try:
    queryable(shortcut)  # asks 2.0 of the second share, 1.0
    over_share = "answered"
except dp.OpenDPException as err:
    over_share = str(err.message)
adaptive_count = queryable(count)  # spends the second share, 1.0
```

```text
second query over its share: insufficient budget for query: 2.0 > 1.0
third query: out of queries
```

The shares are spent in order, so the queryable refuses a query larger than the next share, without spending it, and after the second share it refuses every query. A context wraps this object. With `split_evenly_over=3` in [A first release](../part01-getting-started/01-first-release.md), the library built an adaptive compositor with three equal shares, the context's `accountant`. Each `release()` submits to the queryable, and the context's refusal is the "out of queries" above.

On real data, the number of shares sets the noise of every query. The script builds two contexts over the LFS sample, one with the whole budget of $\varepsilon = 1$ on one query and one split evenly over three. Both ask for the sum of `HWUSUAL`, the hours usually worked per week, without the code 99 for "not applicable" and bounded to 0 to 80 hours. One person has at most 36 rows ([the privacy unit](../part01-getting-started/02-the-data-and-the-context.md#sec-ctx-context)), so the sum moves by up to 36 times the upper bound.

```text
LFS context over 1: accountant map(36)=1.0 d_mids=[1.0]
LFS HWUSUAL sum over 1: scale=2,881.04 95% accuracy=8,630.8
LFS context over 3: accountant map(36)=1.0 d_mids=[0.3333, 0.3333, 0.3333]
LFS HWUSUAL sum over 3: scale=8,643.11 95% accuracy=25,892.4
true HWUSUAL sum: 56,315,594
```

Both accountants map the privacy unit's distance to 1, because the shares add up to the declared budget. But the scale is the sensitivity over the share, so three shares triple the scale and the error bar. An error bar of 26 thousand is small on 56 million hours, but not on a small group. The scale is slightly above the sensitivity over the share because of the float sum's rounding term ([Transformations](09-transformations.md)).

## Odometers {#sec-comb-odometers}

Adaptive composition fixes the budget of each query before the first answer, which is sometimes too early. **Fully adaptive composition** lets you choose the privacy loss of each query as you go. An **odometer** tracks the running total of the loss spent, valid at whatever point you stop. [The full edition's chapter on adaptive composition](../../book-full/part04-privacy-accounting/08-adaptive-composition.md) shows why the ordinary theorems fail with adaptively chosen parameters, and which totals are valid. For pure DP the sum of the losses spent is valid, and OpenDP's odometer reports it.

```{.python excerpt="comb-odometer"}
odometer = dp.c.make_fully_adaptive_composition(
    input_domain=INT_SPACE[0], input_metric=INT_SPACE[1], output_measure=dp.max_divergence()
)
odometer_queryable = odometer(data)
loss_before = odometer_queryable.privacy_loss(1)  # nothing spent yet
odometer_sum = odometer_queryable(shortcut)
loss_after_sum = odometer_queryable.privacy_loss(1)
odometer_count = odometer_queryable(count)
loss_after_count = odometer_queryable.privacy_loss(1)
```

The constructor takes the three supporting elements of the adaptive compositor, but no `d_in` and no `d_mids`. It returns an `Odometer`, not a `Measurement`, with no privacy map, because it promises nothing about the total. On the data it gives an **odometer queryable**, which answers measurements and reports `privacy_loss(d_in)`, the total spent so far at that input distance.

```text
odometer loss: before=0.0 after sum=2.0 after count=3.0 after another count=4.0
odometer loss at d_in=2 after three queries: 8.0
```

The total starts at zero, grows by the map of each query, and never refuses anything. At distance 2, each Laplace query costs twice as much, so the total doubles. So the odometer says how much you spent, and the decision to spend more is yours. That suits open exploration, but nothing stops the total from passing what you may claim, and the next combinator adds that stop.

## Privacy filters {#sec-comb-filters}

A **privacy filter** is an odometer with a cap: it refuses any query that would take the running total past a bound fixed in advance. Since the total can never pass the bound, the filter is a measurement with an ordinary privacy map. For pure DP, a cap on the sum of the realized losses is a valid guarantee ([the full edition's chapter on adaptive composition](../../book-full/part04-privacy-accounting/08-adaptive-composition.md)). The combinator `make_privacy_filter` takes the odometer, the input distance and the cap.

```{.python excerpt="comb-filter"}
privacy_filter = dp.c.make_privacy_filter(odometer, d_in=1, d_out=2.0)
filter_queryable = privacy_filter(data)
filter_count_1 = filter_queryable(count)
filter_count_2 = filter_queryable(count)
try:
    filter_queryable(count)  # would take the loss to 3.0
    refusal = "answered"
except dp.OpenDPException as err:
    refusal = str(err.message)
```

```text
filter: map(1)=2.0
filter third query: filter is now exhausted: pending privacy loss (3.0) would exceed privacy budget (2.0)
filter loss after refusal: filter is exhausted: no more queries can be answered
```

The filter's map at distance 1 is the cap, whatever queries arrive later, so the filter is a measurement. Two counts at a cost of 1 each fit under a cap of 2. The filter refuses the third with a message that names the pending total and the budget. After one refusal, the installed version treats the filter as finished, and even `privacy_loss` raises an error. So read the running total before the query that can exhaust it.

A context wraps a filter when you give it a budget but no split. The script builds a context on the toy data with `dp.loss_of(epsilon=2.0)` and no split. Its queryable is an odometer queryable behind a filter, and `current_privacy_loss` reads the odometer. The count query needs an explicit scale, because there is no share to fit.

```text
context without a split: queryable=OdometerQueryable spent=1.0 remaining=1.0
```

## Chaining and nesting compositors {#sec-comb-nesting}

Compositors and filters are measurements, so you can chain and compose them too. In the first common pattern, a transformation before a compositor does the preprocessing once for all queries. The script chains the clamped sum before an adaptive compositor, so that the queries are measurements on the sum, one integer.

```{.python excerpt="comb-chain-compositor"}
sum_compositor = clamped_sum_only >> dp.c.then_adaptive_composition(
    output_measure=dp.max_divergence(),
    d_in=clamped_sum_only.map(1),  # the sum moves by at most 10 when one record changes
    d_mids=[2.0, 1.0],
)
sum_space = clamped_sum_only.output_domain, clamped_sum_only.output_metric
laplace_on_sum = sum_space >> dp.m.then_laplace(scale=SUM_SCALE)  # a query on the sum, not on the vector
```

```text
transformation >> compositor: d_in=10 map(1)=3.0
laplace on the sum: map(10)=2.0
```

Set `d_in` with care. The compositor comes after the sum, so its input distance is the sum's stability map at the privacy unit's distance, here 10. The `then_` form takes the domain and metric from the transformation but cannot know the distance, so compute it with `map`. A Laplace mechanism of scale 5 on the sum then costs 2, which fits the first share.

In the second pattern, a compositor is inside a compositor. The inner compositor is a measurement whose map is the sum of its shares. So you can submit it to an outer one, which answers with the inner queryable.

```{.python excerpt="comb-nest"}
outer = adaptive_compositor(d_mids=[3.0, 1.0])
inner = adaptive_compositor(d_mids=[2.0, 1.0])  # a measurement costing 3.0: it fits the outer's first share
outer_queryable = outer(data)
inner_queryable = outer_queryable(inner)  # the answer to a compositor query is a queryable
inner_sum = inner_queryable(shortcut)
outer_count = outer_queryable(count)  # the outer's second share, while the inner queryable stays open
inner_count = inner_queryable(count)  # pure DP composes concurrently, so the inner one still answers
```

```text
nested: outer map(1)=4.0 inner map(1)=3.0
```

The outer compositor charges the inner map, 3, whatever the inner one answers later. The script asks the inner queryable for the sum, the outer one for the count, and the inner one again for the count. This works because pure DP supports concurrent composition, where several queryables from one compositor stay open at once. For approximate zCDP and approximate Rényi differential privacy (RDP), post-processing is not proved to hold while an analyst can still interact. So there the library freezes an earlier queryable when a new query arrives. The documentation nests a zCDP compositor inside an approximate-DP one, after it casts the inner measure.

## Measure casting {#sec-comb-casting}

A composition needs all of its parts to have the same output measure. But a Laplace release is measured in pure DP, a Gaussian release in zCDP, and a report usually needs a privacy loss $\varepsilon$ with a slack $\delta$, the $(\varepsilon, \delta)$ of approximate DP. The casting combinators move a measurement from one measure to another, each by a conversion that the full edition derives.

The first conversion goes from pure DP to zCDP. An $\varepsilon$-DP mechanism is $\rho$-zCDP with $\rho = \varepsilon^2/2$ ([the full edition's chapter on concentrated DP](../../book-full/part04-privacy-accounting/03-concentrated-dp.md)), and `make_pureDP_to_zCDP` applies that to the map. The combinator `make_approximate` turns a pure-DP measurement into an approximate-DP one with a slack of zero, so that it can compose with measurements that have a real $\delta$.

```python
import opendp.prelude as dp

dp.enable_features("contrib")
space = dp.atom_domain(T=float, nan=False), dp.absolute_distance(T=float)
laplace = space >> dp.m.then_laplace(scale=1.0)
print(dp.c.make_pureDP_to_zCDP(laplace).map(1.0))  # 0.5
print(dp.c.make_approximate(laplace).map(1.0))  # (1.0, 0.0)
```

You will use the conversion from zCDP to $(\varepsilon, \delta)$ most, because that is how a Gaussian release is reported. A $\rho$-zCDP mechanism satisfies approximate DP at every slack at once, with a loss that grows as the slack shrinks. So the conversion gives the **privacy profile**, a curve that gives the smallest privacy loss for each slack. The combinator `make_zCDP_to_approxDP` returns a measurement whose map is that curve, with the methods `epsilon(delta)` and `delta(epsilon)`. The script reads it at a slack of one in ten million, for a Gaussian with $\rho = 0.1$, which is the scale $1/\sqrt{2\rho}$ at unit sensitivity.

```{.python excerpt="comb-profile"}
gaussian = FLOAT_SPACE >> dp.m.then_gaussian(scale=math.sqrt(1 / (2 * rho)))  # rho = 1 / (2 scale^2)
profile_measurement = dp.c.make_zCDP_to_approxDP(gaussian)
profile = profile_measurement.map(1.0)  # a curve, not a number
epsilon_at_delta = profile.epsilon(delta=delta)
delta_at_one = profile.delta(epsilon=1.0)
fixed = dp.c.make_fix_delta(profile_measurement, delta=delta)  # the curve at one delta: an (epsilon, delta) pair
```

```text
gaussian: rho=0.1000
profile at rho=0.1: epsilon(delta=1e-07)=2.3484 delta(epsilon=1)=0.008933
simple conversion: 2.6391  improved conversion: 2.3484
fix_delta: map(1.0)=(2.3484, 1e-07)  measure=Approximate(MaxDivergence)
```

So a budget of $\rho = 0.1$ is $(2.35, 10^{-7})$-DP. The third line compares that with the two conversions of the full edition. The simple one is
$$\varepsilon = \rho + 2\sqrt{\rho \ln(1/\delta)},$$
which gives 2.64 here. The improved conversion of Canonne, Kamath and Steinke minimizes a tighter expression over the Rényi order, and the script's numerical minimum agrees with OpenDP to four decimals. So OpenDP uses the improved conversion, which matters when you compare its numbers with a paper that quotes the simple one. [The full edition's chapter on RDP conversions](../../book-full/part04-privacy-accounting/02-rdp-conversions.md) shows where the simple formula loses.

A curve cannot compose with a pair, so `make_fix_delta` evaluates the curve at one $\delta$ and returns a measurement whose map is the pair. Its output measure is the approximate max divergence, as from `make_approximate`, so the two can compose. The last casting combinator, `make_fixed_approxDP_to_approxDP`, turns a fixed pair back into a curve. For a pure-DP release cast with `make_approximate`, that curve is flat, because $\varepsilon$-DP at zero slack holds at every slack.

```text
approximate: map(1.0)=(1.0, 0.0)  measure=Approximate(MaxDivergence)
fixed to profile: epsilon(delta=1e-07)=1.0 epsilon(delta=0)=1.0
laplace count + gaussian count: map(1)=(6.6709, 1e-07)
```

The last line shows the result. The Laplace count, which @sec-comb-composition could not compose with the Gaussian count, is cast with `make_approximate`. The Gaussian count is cast to a curve and fixed at the same slack, and then `make_composition` accepts the two. The total is large, because a Gaussian count with $\rho = 0.5$ is expensive at so small a slack: casting makes a composition legal, not cheap. When the Gaussian parts dominate, account the whole release in zCDP and convert once at the end, as [the practitioner edition's release workflow](../../book-practitioner/part03-dp-foundations/12-release-statistics.md) does.

## Amplification by subsampling {#sec-comb-amplification}

The last combinator changes what a privacy map promises without changing the computation. Suppose that your dataset is a simple random sample of $n$ people from a population of $N$, and nobody knows who is in the sample. Then a person may not be in the sample, which makes the release more private with respect to the population. [The full edition's chapter on subsampling](../../book-full/part04-privacy-accounting/05-subsampling.md) derives this amplification and shows why the sampling scheme must match the neighbouring relation. Under pure DP the amplified loss is
$$\varepsilon' = \ln\!\Big(1 + \frac{n}{N}\,(e^{\varepsilon} - 1)\Big).$$
The script prints it next to the library's map. The combinator `make_population_amplification` needs a measurement whose input domain has the sample size, because the ratio of the two sizes is part of the guarantee.

```{.python excerpt="comb-amplify"}
def sample_mean() -> dp.Measurement:
    """The Laplace mean of a sample of exactly SAMPLE values in [0, 10]: the domain carries the size."""
    space = dp.vector_domain(dp.atom_domain(bounds=(0.0, 10.0), nan=False), size=SAMPLE), dp.symmetric_distance()
    return space >> dp.t.then_mean() >> dp.m.then_laplace(scale=0.5)


def amplified_mean(population_size: int) -> dp.Measurement:
    """The same mean, with its guarantee read for the population the sample was drawn from."""
    dp.enable_features("honest-but-curious")  # the library cannot check that the data are such a sample
    return dp.c.make_population_amplification(sample_mean(), population_size=population_size)
```

```text
sample mean: map(2)=2.0000
amplified, population 100: map(2)=0.4940  formula=0.4940
amplified, population 1000: map(2)=0.0619  formula=0.0619
amplifying an unsized domain: elements of the vector domain have unknown size
```

The maps are read at distance 2, because two datasets of the same size differ by a swap of one record. A swap moves the mean of ten values in a range of 10 by 1, so Laplace noise of half that costs $\varepsilon = 2$. For a population of a hundred, the release costs about 0.49, and for a thousand, about 0.06, as the formula also gives. The library refuses a vector domain without a size.

The combinator needs the `honest-but-curious` flag, for the same reason as the Python post-processor: the library cannot verify the claim. The guarantee holds only if the data are a uniform sample of a population of the stated size. It also needs that the readers of the release do not know who is in the sample. So use the map without amplification when the sample depends on the people in it, or when the sampled identities are public.

## Summary

| Call or concept | What it does | Rule of thumb |
| --- | --- | --- |
| `make_chain_tt`, `make_chain_mt`, `make_chain_pm` | Join two transformations, a transformation and a measurement, or a measurement and a post-processor | Write `>>`; it picks the right one |
| `measurement >> python_function` | Post-processing, at no cost | Needs `honest-but-curious` in 0.16.0; no global state, no data-dependent exceptions |
| `DomainMismatch`, `MetricMismatch` | A chain whose intermediate domain or metric differs | Read the two printed lines and insert the step that makes them equal |
| `make_composition([m1, m2])` | One measurement that releases every part; its map is the sum of the parts' maps | Parts share domain, metric and measure; $\rho$ adds like $\varepsilon$ |
| `make_adaptive_composition(..., d_in, d_mids)` | A queryable that answers measurements one at a time, each within the next share | What a context with a split wraps; a query over its share is refused |
| `make_fully_adaptive_composition` | An odometer whose queryable reports `privacy_loss(d_in)` so far | Never refuses; for exploration, not for a release |
| `make_privacy_filter(odometer, d_in, d_out)` | An odometer with a cap, which makes it a measurement | What a context without a split wraps; after one refusal it answers nothing |
| `transformation >> then_adaptive_composition(...)` | Preprocessing shared by every query of a compositor | Set `d_in` to the transformation's map at the privacy unit's distance |
| A compositor submitted to a compositor | A sub-budget, answered by a queryable | Costs the inner compositor's map; pure DP lets both queryables stay open |
| `make_pureDP_to_zCDP`, `make_approximate` | $\varepsilon$ to $\rho = \varepsilon^2/2$; $\varepsilon$ to $(\varepsilon, 0)$ | Cast to make a composition legal, then read the total once |
| `make_zCDP_to_approxDP` | $\rho$ to a privacy profile with `epsilon(delta)` and `delta(epsilon)` | OpenDP uses the improved conversion; $\rho = 0.1$ gives 2.35 at $\delta = 10^{-7}$ |
| `make_fix_delta`, `make_fixed_approxDP_to_approxDP` | A profile at one $\delta$, and a pair back to a flat profile | Fix $\delta$ last, after composing in zCDP |
| `make_population_amplification` | A smaller map for a secret simple random sample of a larger population | Sized input domain, `honest-but-curious`, and a sampling you can vouch for |

: The combinators of this chapter, what each does, and the rule of thumb that goes with it. {#tbl-comb-summary}

## References

- OpenDP documentation: "Combinators", "Chaining", "(Non-Adaptive) Composition", "Adaptive Composition", "Fully Adaptive Composition and Odometers", "Privacy Filters", "Compositor Chaining and Nesting", "Measure Casting" and "Amplification" in the API user guide.
- Rogers, Roth, Ullman and Vadhan, "Privacy Odometers and Filters: Pay-as-you-Go Composition" (2016): the odometer and the filter behind `make_fully_adaptive_composition` and `make_privacy_filter`.
- Whitehouse, Ramdas, Rogers and Wu, "Fully Adaptive Composition in Differential Privacy" (2023): the fully adaptive composition theorem the odometer implements.
- Haney, Shoemate, Tian, Vadhan, Vyrros, Xu and Zhang, "Concurrent Composition for Interactive Differential Privacy with Adaptive Privacy-Loss Parameters" (2023): why several queryables may stay open at once.
- Bun and Steinke, "Concentrated Differential Privacy: Simplifications, Extensions, and Lower Bounds" (2016): zCDP, its composition and the cast from pure DP.
- Canonne, Kamath and Steinke, "The Discrete Gaussian for Differential Privacy" (2020): the improved conversion from zCDP to $(\varepsilon, \delta)$ that the privacy profile implements.
- The full edition's chapters on [post-processing and basic composition](../../book-full/part03-dp-foundations/02-properties.md), [composition](../../book-full/part03-dp-foundations/04-composition.md), [concentrated DP](../../book-full/part04-privacy-accounting/03-concentrated-dp.md), [RDP conversions](../../book-full/part04-privacy-accounting/02-rdp-conversions.md), [subsampling](../../book-full/part04-privacy-accounting/05-subsampling.md) and [adaptive composition](../../book-full/part04-privacy-accounting/08-adaptive-composition.md) derive the results this chapter states.
