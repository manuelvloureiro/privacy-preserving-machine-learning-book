# Combinators

The previous two chapters built transformations and measurements one at a time. A real release is never one of them. It is a clamp, then a sum, then noise, then a second query that depends on the first. At the end it is reported as one guarantee of differential privacy (DP). The pieces that join measurements and transformations into bigger ones are the **combinators**. Each takes one or more objects of the framework and returns a new one, with a privacy map that says what the new object costs. You never add budgets by hand, because the combinator's map does it, and it refuses to join pieces whose promises do not fit.

The chapter follows the documentation's Combinators pages in order. Chaining joins a transformation to what comes after it. Composition releases several measurements on the same data, first all at once, then one at a time, then without deciding the budget of each in advance. Measure casting converts a guarantee between privacy measures, so that a Gaussian release can be reported with the privacy loss $\varepsilon$ and the slack $\delta$ of approximate DP. Amplification by subsampling reads a smaller guarantee for a sample of a larger population. By the end you will be able to read the privacy map of any chain, say why a context refuses a query, and convert a budget between measures.

The script [opendp_combinators.py](../../code/part03-framework/opendp_combinators.py) runs every step on OpenDP 0.16.0. Its toy data are twenty integers from 0 to 10 drawn by a seeded generator. Its toy measurements are a Laplace count and a Laplace sum of the clamped values, as in the documentation. One example runs on the Labour Force Survey (LFS) sample, where the number of queries in a composition changes the noise scale of a real sum. The maps, profiles, scales and error messages on this page are the same on every run. The released values are not, because OpenDP's noise cannot be seeded, so they appear in one block marked as one run.

```bash
uv run python code/part03-framework/opendp_combinators.py
```

## Chaining {#sec-comb-chaining}

The simplest way to combine two pieces is to run one after the other, and three constructors do that. `make_chain_tt` joins two transformations. `make_chain_mt` puts a measurement after a transformation, and `make_chain_pm` attaches a post-processing function to a measurement. There is no chainer for two measurements, because anything computed after a measurement is post-processing and needs no further accounting. The script builds the clamped sum and its Laplace release with the long forms first.

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

The outer piece comes first in each call and the inner piece second, as in function composition. Each constructor takes its input space from the output space of the piece before it. The chained transformation has a stability map, which [Transformations](09-transformations.md) introduced as the bound on how far the output moves when the input moves by a given distance. Adding or removing one record changes the clamped sum by at most the width of the bounds, so the map at distance 1 is 10. The chained measurement has a privacy map, which [Measurements](10-measurements.md) introduced, and it is the stability of the inner part divided by the noise scale. A sum that moves by 10 under Laplace noise of half that scale costs $\varepsilon = 2$.

```text
clamp >> sum: map(1)=10
clamp >> sum >> laplace: map(1)=2.0 check(1, 2.0)=True
with >>: map(1)=2.0
```

The `>>` operator chooses the right chainer for you, and it takes a `then_` constructor on its right so that the input space never has to be repeated. Below is the same measurement in one line.

```python
import opendp.prelude as dp

dp.enable_features("contrib")
space = dp.vector_domain(dp.atom_domain(T=int)), dp.symmetric_distance()
noisy_sum = space >> dp.t.then_clamp((0, 10)) >> dp.t.then_sum() >> dp.m.then_laplace(scale=5.0)
print(noisy_sum.map(1))  # 2.0
print(noisy_sum.input_domain)  # VectorDomain(AtomDomain(T=i32))
print(noisy_sum.output_measure)  # MaxDivergence
```

**Post-processing** is any computation on a release that does not touch the data again. It costs nothing, because whatever an adversary could learn from the processed value they could have learned from the release itself. [The full edition's chapter on properties](../../book-full/part03-dp-foundations/02-properties.md) proves that. In OpenDP you attach a Python function to a measurement with `>>`, and the result is a measurement with the same privacy map.

```{.python excerpt="comb-postprocess"}
def divide_by_size(total: int) -> float:
    return total / N  # the released sum over the public size: a mean, at no further cost

mean_of_sum = shortcut >> divide_by_size
```

```text
post-processed: map(1)=2.0
```

In version 0.16.0 this needs a second feature flag, `honest-but-curious`, which the script switches on just before the chain and off just after. The reason is that the library cannot inspect a Python function. A post-processor must not read global state, such as the clock, and must not raise an exception whose presence depends on the data. Either would open a channel from the data to the caller that the privacy map does not account for.

A chain only forms when the pieces fit. The output domain of the inner piece must equal the input domain of the outer one, and the same holds for the metrics. The script tries to feed a sum of floats into a Laplace mechanism built for integers, and the library refuses.

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

The second message comes from a count, whose output is an integer at an integer absolute distance, chained to a Laplace mechanism whose metric measures distances in floats. With a mismatched domain, the outer piece's proof would cover inputs the inner piece never promised to produce. When you meet one of these errors, read the two lines it prints and insert the cast, clamp or impute step that makes them equal.

The sum and count that the rest of the chapter composes have the error bars below, which [Assessing utility and choosing the noise](../part02-tabular-data/07-utility-and-noise.md) explains how to read.

```text
sum: scale=5 95% accuracy=15.4537
count: scale=1 95% accuracy=3.3756
```

## Non-adaptive composition {#sec-comb-composition}

Suppose you want both the sum and the count of the toy vector. Running the two measurements on the same data spends both of their budgets, and the question is what the total is. **Basic composition** says that it is the sum of the parts. A release that is $\varepsilon_1$-DP followed by one that is $\varepsilon_2$-DP on the same data is $(\varepsilon_1 + \varepsilon_2)$-DP, which [the full edition's chapter on properties](../../book-full/part03-dp-foundations/02-properties.md) proves. The combinator `make_composition` builds that rule into one measurement, which runs every part on the data and returns their releases as a list.

```{.python excerpt="comb-composition"}
both = dp.c.make_composition([shortcut, count])  # one measurement that releases the sum and the count
print(f"composition: map(1)={both.map(1):.1f}  parts: {shortcut.map(1):.1f} + {count.map(1):.1f}")
```

```text
composition: map(1)=3.0  parts: 2.0 + 1.0
composition output measure: MaxDivergence
```

The parts must agree on their input domain, their input metric and their output measure. That is because a composition is a statement about one dataset under one notion of neighbours and one way of measuring privacy. The sum and the count share all three. The composition is called non-adaptive because every part is fixed before any data are seen, so no part can depend on another's answer.

The same combinator composes measurements in other measures, and the rule it applies is the measure's own. **Zero-concentrated differential privacy (zCDP)** measures privacy by a single parameter $\rho$, which bounds the Rényi divergence between the output distributions on neighbours at every order. Under zCDP the parameters add just as $\varepsilon$ does, which [the full edition's chapter on concentrated DP](../../book-full/part04-privacy-accounting/03-concentrated-dp.md) derives. The Gaussian sum at scale 5 has $\rho = 2$, the Gaussian count at scale 1 has $\rho = 0.5$, and the composition reports their sum.

```text
zCDP composition: map(1)=2.50  parts: 2.00 + 0.50
mixed measures: All output measures must be the same
```

The second line shows the check that stops you from mixing measures, because the maps of a Laplace count and of a Gaussian count return numbers that mean different things. @sec-comb-casting shows the combinators that bring them to a common measure first.

## Adaptive composition {#sec-comb-adaptive}

Often you do not know the second query until you have seen the first answer. You release a count, and it tells you whether a mean is worth asking for. Composition still holds when each query is chosen after the earlier answers, as long as the budget of each is fixed in advance. [The full edition's chapter on composition](../../book-full/part03-dp-foundations/04-composition.md) proves this adaptive form. The combinator for it is `make_adaptive_composition`. Its constructor is more verbose than the non-adaptive one, because it cannot read the supporting elements off parts it has not seen yet.

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

The argument `d_in` is the distance between neighbouring datasets, here 1 for adding or removing one record. The argument `d_mids` lists the privacy loss each query may spend, in order, and the privacy map of the compositor is their sum, here 3.

```text
adaptive: map(1)=3.0
```

Invoking the compositor on the data does not release anything. It returns a **queryable**, which is a small state machine. It holds the data and the shares that remain, and it accepts a measurement as a query. It checks that the measurement's map at `d_in` fits within the next share, and it answers with the measurement's release. The script submits the sum, then tries the sum again, then submits the count.

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

The shares are consumed in order, so a query that asks for more than the next share is refused, and the share is not spent. After the count takes the second share, a third query of any size is refused as well. This is the object a context wraps. When [A first release](../part01-getting-started/01-first-release.md) built a context with `split_evenly_over=3`, the library built an adaptive compositor with three equal shares and kept its queryable. The context's `accountant` attribute is that compositor, and its `d_mids` are the shares. A `release()` is a submission to the queryable, and the refusal you saw there is the "out of queries" above, translated.

On real data the number of shares decides the noise of every query. The script builds two contexts over the LFS sample, one with the whole budget of $\varepsilon = 1$ on a single query and one split evenly over three. Both ask for the sum of `HWUSUAL`, the hours usually worked per week, with the code 99 for "not applicable" filtered out. The values are bounded to 0 to 80 hours. A person appears in at most 36 quarters, so the privacy unit is 36 contributions. The sum therefore moves by up to 36 times the upper bound when one person is added or removed.

```text
LFS context over 1: accountant map(36)=1.0 d_mids=[1.0]
LFS HWUSUAL sum over 1: scale=2,881.04 95% accuracy=8,630.8
LFS context over 3: accountant map(36)=1.0 d_mids=[0.3333, 0.3333, 0.3333]
LFS HWUSUAL sum over 3: scale=8,643.11 95% accuracy=25,892.4
true HWUSUAL sum: 56,315,594
```

Both accountants map the privacy unit's distance to 1, because the shares always add up to the budget you declared. What changes is the scale the context finds for the query, which is the sensitivity over the share. Three shares triple the scale and the error bar. On a sum of about 56 million hours an error bar of 26 thousand is still small. On a small group it would not be. The scale sits slightly above the sensitivity over the share, because the sum of floats carries a rounding term that [Transformations](09-transformations.md) explains.

## Odometers {#sec-comb-odometers}

Adaptive composition fixes each query's budget before the first answer. Sometimes even that is too early, because the answer to the first query decides how much the second one deserves. **Fully adaptive composition** lets the privacy loss of each query be chosen as you go. The object that tracks it is an **odometer**, a running total of the loss spent so far that stays valid at whatever point you choose to stop. [The full edition's chapter on adaptive composition](../../book-full/part04-privacy-accounting/08-adaptive-composition.md) shows why plugging adaptively chosen parameters into the ordinary theorems fails, and which running totals are valid. For pure DP the sum of the losses spent is one, and that is what OpenDP's odometer reports.

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

The constructor takes the same three supporting elements as the adaptive compositor but no `d_in` and no `d_mids`, because nothing about the budget is decided yet. What it returns is an `Odometer`, not a `Measurement`, and an odometer has no privacy map, because it makes no promise about the total. Invoking it on the data gives an **odometer queryable**. It answers measurements like the compositor's queryable did, and it also reports `privacy_loss(d_in)`, the total spent so far at the given input distance.

```text
odometer loss: before=0.0 after sum=2.0 after count=3.0 after another count=4.0
odometer loss at d_in=2 after three queries: 8.0
```

The total starts at zero, grows by the map of each query as it is answered, and never refuses anything. The last line shows why the argument matters. At distance 2, each Laplace query costs twice as much, so the total doubles. The odometer answers "how much have I spent?" and leaves "may I spend more?" to you. That suits an open-ended exploration, but nothing stops the total from growing past what you may claim. The next combinator adds the stop.

## Privacy filters {#sec-comb-filters}

A **privacy filter** is an odometer with a cap. It refuses any query that would take the running total past a bound fixed in advance. The total can then never exceed the bound, so the filter is a measurement with an ordinary privacy map. [The full edition's chapter on adaptive composition](../../book-full/part04-privacy-accounting/08-adaptive-composition.md) proves that for pure DP the cap on the sum of the realized losses is a valid guarantee. The combinator `make_privacy_filter` takes the odometer, the input distance and the cap.

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

The filter's map at distance 1 is the cap, whatever queries later arrive, which is what makes it a measurement. Two counts at a cost of 1 each fit under a cap of 2, and the third is refused with a message that names the pending total and the budget. The installed version then treats the filter as finished. After one refusal, even `privacy_loss` raises, so read the running total before the query that might exhaust it, not after. Unlike the adaptive compositor, a filter needs no list of shares.

The filter is also what a context wraps when you give it a budget but no split. The script builds a context on the toy data with `dp.loss_of(epsilon=2.0)` and neither `split_evenly_over` nor `split_by_weights`. Its queryable is then an odometer queryable behind a filter, and `current_privacy_loss` reads the odometer. The script gives its count query an explicit scale, since there is no share for the context to search within.

```text
context without a split: queryable=OdometerQueryable spent=1.0 remaining=1.0
```

## Chaining and nesting compositors {#sec-comb-nesting}

Compositors and filters are measurements, so the chainers and the compositors apply to them too. Two patterns come up in practice. The first is a transformation before a compositor, which saves repeating the same preprocessing in every query. The script chains the clamped sum in front of an adaptive compositor, so that the compositor's queries are measurements on the sum, a single integer, rather than on the vector.

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

The one thing to get right is `d_in`. The compositor sits after the sum, so its input distance is the sum's output distance, which is the stability map of the sum at the privacy unit's distance, here 10. The `then_` form takes the domain and metric from the transformation, but it cannot guess the distance, so compute it with `map` rather than typing it. A query to this compositor is checked at that distance, and a Laplace mechanism of scale 5 on the sum then costs 2, which fits the first share.

The second pattern is a compositor inside a compositor. The inner compositor is a measurement whose map is the sum of its shares, so it can be submitted as a query to an outer one. The answer is the inner queryable.

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

The outer compositor charges 3 for the inner one, whatever the inner one later answers, because 3 is the inner map. The script interleaves the queries. It asks the inner queryable for the sum, the outer one for the count, and the inner one again for the count. That works because pure DP supports concurrent composition, where several queryables spawned from one compositor stay open at the same time. For approximate zCDP and approximate Rényi differential privacy (RDP) the library instead freezes an earlier queryable when a new query arrives, because post-processing has not been proved to hold there while an analyst can still interact with it. The documentation uses nesting to run a zCDP compositor inside an approximate-DP one, after casting the inner compositor's measure with the combinators of the next section.

## Measure casting {#sec-comb-casting}

Every measurement carries an output measure, and a composition needs all of its parts to carry the same one. A Laplace release is measured in pure DP, a Gaussian release in zCDP, and a report is usually wanted as $(\varepsilon, \delta)$. The casting combinators move a measurement from one measure to another, each by a conversion the full edition derives, so that the pieces can be composed.

The first conversion goes from pure DP to zCDP. An $\varepsilon$-DP mechanism is $\rho$-zCDP with $\rho = \varepsilon^2/2$, which [the full edition's chapter on concentrated DP](../../book-full/part04-privacy-accounting/03-concentrated-dp.md) proves, and `make_pureDP_to_zCDP` applies it to the map. `make_approximate` turns a pure-DP measurement into an approximate-DP one with a slack of zero. It can then be composed with measurements that have a real $\delta$.

```python
import opendp.prelude as dp

dp.enable_features("contrib")
space = dp.atom_domain(T=float, nan=False), dp.absolute_distance(T=float)
laplace = space >> dp.m.then_laplace(scale=1.0)
print(dp.c.make_pureDP_to_zCDP(laplace).map(1.0))  # 0.5
print(dp.c.make_approximate(laplace).map(1.0))  # (1.0, 0.0)
```

The conversion you will use most goes from zCDP to $(\varepsilon, \delta)$, because that is how a Gaussian release is reported. A $\rho$-zCDP mechanism satisfies approximate DP at every slack at once, with a privacy loss that grows as the slack shrinks. So the honest output of the conversion is not a pair but a curve, the **privacy profile**, which gives the smallest privacy loss for each slack and the reverse. `make_zCDP_to_approxDP` returns a measurement whose map is that curve, with the methods `epsilon(delta)` and `delta(epsilon)`. The script builds a Gaussian mechanism with $\rho = 0.1$, which the scale $1/\sqrt{2\rho}$ gives at unit sensitivity. It then reads the curve at a slack of one in ten million.

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

A budget of $\rho = 0.1$ is $(2.35, 10^{-7})$-DP. The third line compares that with the two conversions the full edition states. The simple one is
$$\varepsilon = \rho + 2\sqrt{\rho \ln(1/\delta)},$$
which gives 2.64 here. The improved conversion of Canonne, Kamath and Steinke minimizes a tighter expression over the Rényi order, and the script's numerical minimum agrees with OpenDP to four decimals. So the library implements the improved conversion, which matters when you check its numbers against a paper that quotes the simple one. [The full edition's chapter on RDP conversions](../../book-full/part04-privacy-accounting/02-rdp-conversions.md) explains where the simple formula loses.

A curve cannot be composed with a pair, so `make_fix_delta` evaluates the curve at one $\delta$ of your choice and returns a measurement whose map is the pair. Its output measure is then the approximate max divergence, the same as `make_approximate` produces, and the two can meet in a composition. The last casting combinator, `make_fixed_approxDP_to_approxDP`, goes the other way and turns a fixed pair back into a curve. For a pure-DP release cast with `make_approximate`, that curve is flat, since the mechanism is $\varepsilon$-DP at zero slack and therefore at every larger slack too.

```text
approximate: map(1.0)=(1.0, 0.0)  measure=Approximate(MaxDivergence)
fixed to profile: epsilon(delta=1e-07)=1.0 epsilon(delta=0)=1.0
laplace count + gaussian count: map(1)=(6.6709, 1e-07)
```

The last line is the payoff. The Laplace count, which @sec-comb-composition could not compose with the Gaussian count, is cast with `make_approximate`. The Gaussian count is cast to a curve and fixed at the same slack, and `make_composition` accepts the two. The total privacy loss is large because a Gaussian count with $\rho = 0.5$ is expensive at so small a slack. Casting makes a composition legal, not cheap. When the Gaussian parts dominate, account the whole release in zCDP and convert once at the end, as [the practitioner edition's release workflow](../../book-practitioner/part03-dp-foundations/12-release-statistics.md) does.

## Amplification by subsampling {#sec-comb-amplification}

The last combinator changes what a privacy map promises without changing the computation. Suppose your dataset is a simple random sample from a larger population, and nobody knows who was sampled. Then the release is more private with respect to the population than to the sample, because a person who may not be in the sample is harder to detect. [The full edition's chapter on subsampling](../../book-full/part04-privacy-accounting/05-subsampling.md) derives this amplification and shows why the sampling scheme must match the neighbouring relation. Write $n$ for the size of the sample and $N$ for the size of the population. Under pure DP the amplified loss is
$$\varepsilon' = \ln\!\Big(1 + \frac{n}{N}\,(e^{\varepsilon} - 1)\Big).$$
The script prints it beside the library's map. `make_population_amplification` needs a measurement whose input domain carries the sample size, because the ratio of the two sizes is part of the guarantee.

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

The maps are read at distance 2, because two datasets of the same fixed size differ by a swap of one record, which the symmetric distance counts as one removal and one addition. The mean of ten values in a range of 10 moves by 1 under a swap, and Laplace noise of half that costs $\varepsilon = 2$. Read for a population of a hundred, the same release costs about 0.49, and for a population of a thousand about 0.06. The formula matches the library's map, and the gain grows with the population. A measurement on a vector domain without a size is refused, as the last line shows.

The combinator needs the `honest-but-curious` flag, for the same reason as the Python post-processor. The guarantee is only true if the data really are a uniform sample of a population of the stated size, and if the sample's membership is unknown to whoever reads the release. The library cannot verify either from the data, so the flag records that the claim rests on your word. Use the unamplified map when the sample was chosen in a way that depends on the people in it, or when the sampled identities are public.

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
