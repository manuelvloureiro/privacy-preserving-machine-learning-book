# Utilities and the Context API inside

So far, each query ended with `.laplace()` and let the context choose a scale. The context uses framework utilities that are also available directly: type descriptors specify value types, binary search converts budgets to scales, and serializers convert measurements to text. The Context API wraps these utilities and the combinators in [the previous chapter](11-combinators.md). Understanding this layer helps predict query behaviour and reproduce queries manually when the API rejects them.

The chapter dissects the count query from [the first chapter](../part01-getting-started/01-first-release.md), which returns the same scale of 3. It also shows how to interpret type errors, run the context's parameter search, serialize a measurement and identify the framework object created by each Context API call.

The script [opendp_utilities.py](../../code/part03-framework/opendp_utilities.py) runs on OpenDP 0.16.0 and Polars 1.36.1. Plain-data queries use the first chapter's hundred seeded integers; the Polars query counts rows in the France Labour Force Survey (LFS) sample. Type descriptors, errors, scales and privacy maps are deterministic. Released counts vary because OpenDP's noise cannot be seeded, so the page labels them as one run.

```bash
uv run python code/part03-framework/opendp_utilities.py
```

## Types {#sec-utils-types}

OpenDP's Rust core supports signed and unsigned integers from 8 to 128 bits and two float widths. It preserves this precision because proof constants depend on the type; a 32-bit integer sum can overflow where a 64-bit sum cannot. Every domain, metric and measure therefore carries a **type descriptor**, such as the `T=` arguments in [the framework chapter](08-a-framework-for-privacy.md). OpenDP does not cast between types automatically.

Type descriptors have three equivalent forms. Canonical strings include `"i32"` and `"Vec<f64>"`. Python types such as `int`, `float`, `str` and `bool` are aliases; `list[int]` aliases a vector. Objects from `opendp.typing`, including `dp.i32`, `dp.f64` and `dp.Vec[dp.i32]`, provide the same types with editor completion.

```python
import opendp.prelude as dp

print(dp.atom_domain(T=float))  # AtomDomain(nan=true, T=f64)
print(dp.atom_domain((0, 1), T=dp.i32))  # AtomDomain(bounds=[0, 1], T=i32)
print(dp.RuntimeType.parse(list[int]))  # Vec<i32>
print(dp.Vec[dp.i32] == dp.RuntimeType.parse(list[int]))  # True
```

Python aliases have fixed widths: `int` means 32-bit and `float` means 64-bit. `dp.RuntimeType.infer` uses the same mapping. NumPy's default integer is 64-bit, so an array value has type `i64` and does not match a domain built with `T=int`. If your integers are 64-bit, use `set_default_int_type` to change the alias.

```python
import numpy as np
import opendp.prelude as dp

print(dp.RuntimeType.parse(int))  # i32
print(dp.RuntimeType.infer(np.int64(1)))  # i64
dp.set_default_int_type(dp.i64)
print(dp.RuntimeType.parse(int))  # i64
print(dp.vector_domain(dp.atom_domain(T=int)))  # VectorDomain(AtomDomain(T=i64))
```

When the types do not meet, the error reads `inferred type is XXX, expected YYY`, which the documentation's troubleshooting page lists. The script builds a Laplace mechanism on an integer domain with an integer metric, and then asks for its privacy map at a sensitivity of 1.5.

```{.python excerpt="utils-type-error"}
int_space = dp.atom_domain(T=int), dp.absolute_distance(T=int)
laplace_int = int_space >> dp.m.then_laplace(scale=1.0)  # Laplace noise on a 32-bit integer
try:
    laplace_int.map(1.5)  # a float sensitivity, on a metric whose distances are integers
    message = "no error"
except TypeError as err:
    message = first_line(err)
```

```text
map(1.5) on an integer metric: TypeError: inferred type is f64, expected i32
map(2) on an integer metric: 2.0
float d_in on the count: TypeError: inferred type is f64, expected u32
integer d_in on the count: 0.3333
```

The error reports the inferred and expected types; here, pass integer 2. The next lines show a common mismatch: symmetric distance uses unsigned integer distances, so `map(1.0)` fails and `map(1)` succeeds. Another mismatch occurs when a scalar $\varepsilon$ is passed to a measure that expects an $(\varepsilon,\delta)$ pair. Version 0.16.0 reports this differently from the documentation, as shown below.

```text
scalar epsilon on an approximate measure: TypeError: Cannot coerce a non-tuple type to a tuple
pair on an approximate measure: True
```

[The chapter on limitations and troubleshooting](../part04-extending/15-limitations-and-troubleshooting.md) covers other errors. Read the expected type and adjust the argument, bounds or data. Do not change the domain to fit the argument; OpenDP chose that type for the proof.

## Parameter search {#sec-utils-search}

A measurement has three parameter types: input distance `d_in` (the privacy-unit bound), output distance `d_out` (the budget) and constructor arguments such as noise scale. With these fixed, `check(d_in, d_out)` tests whether the measurement meets the budget, as explained in [the measurements chapter](10-measurements.md). The result is monotone: any larger scale or budget also passes. Binary search locates the boundary and solves for one parameter from the others.

OpenDP provides three searches. `dp.binary_search` takes a predicate and returns the value nearest the boundary on the passing side. `dp.binary_search_param` takes a function that builds a chain from one parameter and uses `check` with `d_in` and `d_out`. `dp.binary_search_chain` performs the same search and returns the chain at the selected value.

```python
import opendp.prelude as dp

print(dp.binary_search(lambda x: x >= 5.0))  # 5.0
print(dp.binary_search(lambda x: x > 5, T=int))  # 6
print(dp.binary_search(lambda x: x <= 5, bounds=(0, None)))  # 5
```

The three lines show the arguments of a search. Without bounds, the search first finds bounds by steps outwards from zero. The type `T` says whether the parameter is a float or an integer, and bounds with `None` on one side give a one-sided search. The script then runs the search of a context on the first chapter's Laplace mechanism, with a share of $\varepsilon/3$.

```{.python excerpt="utils-search"}
scale = dp.binary_search_param(make_laplace, d_in=1.0, d_out=share)  # the smallest scale within the share
chain = dp.binary_search_chain(make_laplace, d_in=1.0, d_out=share)  # the measurement at that scale
largest_d_in = dp.binary_search(lambda d_in: make_laplace(1.0).check(d_in, EPSILON))  # the other way round
```

```text
search: scale=3.0000000000000004 chain.map(1.0)=0.3333 chain.check(1.0, share)=True
search: largest d_in at scale 1 within epsilon 1: 1.0
search: gaussian scale for rho=1 at d_in=1: 0.7071
```

The scale is the smallest one whose map at a sensitivity of 1 is within a third of $\varepsilon = 1$. It is a little above 3, because the search stops on the passing side of the boundary, and it is the scale that `param()` reported in the first chapter. The third call asks for the largest sensitivity that a scale of 1 hides within $\varepsilon = 1$, which is 1. The Gaussian mechanism is measured in zero-concentrated differential privacy (zCDP), with a loss $\rho$ of half the squared sensitivity over the squared scale. So the search for $\rho = 1$ gives a scale of 0.7071 ([the full edition's chapter on the two mechanisms](../../book-full/part03-dp-foundations/03-laplace-gaussian.md)).

To show what the search does, the script writes it by hand. The predicate is `check`, the interval starts at 0 to 100, and each step keeps the half that contains the boundary.

```{.python excerpt="utils-by-hand"}
def search_by_hand(passes: Callable[[float], bool], low: float, high: float, tolerance: float) -> tuple[float, int]:
    """Bisection on a predicate that fails at `low` and passes at `high`: the smallest passing value, and the steps."""
    steps = 0
    while high - low > tolerance:
        middle = (low + high) / 2
        if passes(middle):
            high = middle  # the boundary is at or below the middle
        else:
            low = middle  # the boundary is above it
        steps += 1
    return high, steps
```

```text
search by hand: scale=3.000000 after 27 steps
search with bounds (0, 100): 3.0000000000000004
```

After twenty-seven halvings, the interval of 100 is below a millionth, and its passing end is the scale. The library's search on the same bounds lands on the float just above 3, so it bisects further than that tolerance. A search can fail in two ways, and the script shows both messages.

```text
search with bounds (0, 1): ValueError: the decision boundary of the predicate is outside the bounds
search on a constant predicate: ValueError: unable to infer bounds
```

Bounds that do not contain the boundary give the first error, so make them wider. A predicate that never changes gives the second, here one that always passes. With a real chain, it means that no value of the parameter changes the answer, so look at the chain before the bounds.

## Serialization {#sec-utils-serialize}

A measurement is a recipe that names its constructors and their arguments. The function `dp.serialize` writes that recipe as a JSON string, and `dp.deserialize` turns it back into a measurement. So an analyst can build and inspect a query on a machine without the data, and send the text to the machine that holds them. The script serializes the first chapter's count measurement.

```{.python excerpt="utils-serialize"}
measurement = count_laplace(3.0)
text = dp.serialize(measurement)  # JSON that names each constructor and its arguments
restored = dp.deserialize(text)
same_space = restored.input_space == measurement.input_space
```

```text
serialized: 2152 characters, starting {"__function__": "make_chain_mt",
restored: map(1)=0.3333 same input space: True
```

The text starts with the outermost constructor, the chain of a transformation and a measurement, and nests the rest inside it. The restored measurement has the same input space and privacy map, so a reviewer can check the guarantee before anyone touches the data. The documentation does not promise a stable format between versions, and it asks you not to write it by hand.

Two things do not survive a round trip. The first is the `param` attribute that a context attaches to a measurement after a search ([a later section](#sec-utils-query)). It is a Python attribute and not part of the recipe. The second is anything with Python state that the constructors did not set, such as a context, a query or a plugin built from a Python function.

```text
param before a round trip: 3.0000000000000004
param after a round trip: None
serialize a context: OpenDP JSON Encoder does not handle instances of <class 'opendp.context.Context'>
```

A Polars query has its own route, because Polars can serialize the plan of a `LazyFrame` to bytes. The method `Context.deserialize_polars_plan` wraps those bytes in a query on a context of its own. So a client can write the plan against the survey's schema, and the server can run it on the rows.

```{.python excerpt="utils-plan"}
client = lfs_context(pl.LazyFrame(schema=frame.collect_schema()))  # the schema, but no rows
plan = client.query().select(dp.len(signed=True)).polars_plan.serialize()  # bytes
server = lfs_context(frame)  # the real data
query = server.deserialize_polars_plan(plan)
summary = query.summarize(alpha=0.05)
```

```text
plan: 1792 bytes
plan on the server: scale=36.0 95% accuracy=108.3429
```

The client's context holds an empty frame with the survey's schema, and the server's holds the file. Both must state the budget and the privacy unit of 36 rows per person ([the chapter on the data and the context](../part01-getting-started/02-the-data-and-the-context.md#sec-ctx-context)). The plan carries the query but not the guarantee. The scale of 36 is the unit's distance over the budget, and the error bar is about 108 rows on a count of millions.

## The compositor behind a context {#sec-utils-compositor}

The rest of the chapter rebuilds the Context API from the combinators of [the previous chapter](11-combinators.md). The constructor `dp.Context.compositor` builds an **adaptive composition** over the data's space and calls it on the data. That composition takes a list of budgets, the `d_mids`, and accepts one measurement per share, each of which may depend on the earlier answers. Together they stay within the sum ([the full edition's chapter on composition](../../book-full/part03-dp-foundations/04-composition.md)). The script reads the context's accountant and builds the same one by hand.

```{.python excerpt="utils-accountant"}
by_hand_accountant = dp.c.make_adaptive_composition(
    input_domain=SPACE[0],
    input_metric=SPACE[1],
    output_measure=dp.max_divergence(),
    d_in=1,  # the privacy unit's distance: one row added or removed
    d_mids=[share] * QUERIES,  # the shares, which add up to the budget
)
queryable = by_hand_accountant(data)  # answers measurement queries on the data, and keeps count
```

```text
context accountant: VectorDomain(AtomDomain(nan=true, T=f64)) SymmetricDistance() MaxDivergence
context d_in: 1  d_mids: [0.3333, 0.3333, 0.3333]
context accountant map(1): 1.0000
by hand accountant map(1): 1.0000  queryable: Queryable(Q=AnyMeasurement)
d_mids by weights [1, 2, 1]: [0.25, 0.5, 0.25]
```

The context found the domain from the data, a list of floats, and the metric and the measure come from `unit_of` and `loss_of`. Both compositions map the unit's distance to the whole budget, 1. On the data, the accountant returns a **queryable**, which answers one measurement at a time and refuses when the shares run out. The context keeps that queryable, and the data are only inside it, which is why the first chapter told you to drop your own reference.

The shares come from `split_evenly_over` or `split_by_weights`. The context gives the weights to `binary_search_param` and scales them until the composition's map at `d_in` meets the budget. For pure differential privacy, that is the expected proportional split: weights of 1, 2 and 1 give a quarter, a half and a quarter of $\varepsilon$.

## The query behind a release {#sec-utils-query}

The call `context.query()` returns a **Query**, which holds a chain and the share it must fit, and on a LazyFrame it is the `LazyFrameQuery` of the Polars chapters. The chain starts as the data's space, and each method you call, such as `count` or `laplace`, chains on the constructor of that name. So `query.count()` is `space >> dp.t.then_count()`. The script shows the type of the chain at each step.

```{.python excerpt="utils-query"}
query = plain_query(context)  # the chain is the data's space: a domain and a metric
counted = query.count()  # the chain is a Transformation
noisy = counted.laplace()  # no scale: the chain is a PartialChain, waiting for one number
is_partial = isinstance(noisy._chain, dp.PartialChain)
```

```text
query chain: tuple -> Transformation -> PartialChain
partial chain: True
resolve a transformation: ValueError: Query is not yet a measurement or odometer.
```

The important step is `laplace()` without a scale. The query finds one argument missing from the constructor's signature, so it wraps the constructor in a **partial chain**, a function from the missing number to the finished chain. Only one number may be missing at a time, so a second noise step without a scale fails. A query that is still a transformation cannot resolve into a measurement, as the third line shows.

The method `resolve` finds the number. It calls the partial chain's `fix`, which runs `binary_search` with the predicate "the chain at this value checks within the query's share". That is the search of @sec-utils-search, and `fix` stores the value on the measurement as `param`. So `param()` reports the scale of the search, or `None` when there was no search.

```text
param of .laplace(): 3.0000000000000004
param of .laplace(scale=3.0): None
two missing parameters: ValueError: gaussian is missing 1 parameter(s).
```

The script then runs the same search by hand on the same chain, and releases through the queryable built by hand.

```{.python excerpt="utils-resolve"}
by_hand_param = dp.binary_search_param(count_laplace, d_in=1, d_out=share)  # what resolve runs on the chain
by_hand_measurement = count_laplace(by_hand_param)
dp_count_by_hand = queryable(by_hand_measurement)  # what release does: the queryable runs the measurement
```

```text
resolve: context param=3.0000000000000004 by hand=3.0000000000000004 equal=True
d_mids after context(measurement): [0.3333, 0.3333]
```

The two scales are equal, because they come from the same search on the same predicate. The method `release` gives the resolved measurement to the queryable and records the share as spent, so the first share leaves `d_mids`. The version by hand does the same with `queryable(measurement)`, but nothing records the spend. That record is what the context adds.

## Nested compositors and other measures {#sec-utils-measures}

Two features of the Context API are easier to understand when you can see the chain. The first is `Query.compositor`, which nests an adaptive composition inside one share. The query's share becomes the inner budget, split again, and the release of the query returns a new `Context` and not a number.

```{.python excerpt="utils-nested"}
inner = plain_query(context).compositor(split_evenly_over=2).release()  # a Context inside one share
inner_scale = plain_query(inner).count().laplace().param()
```

```text
inner context: d_in=1 d_out=0.3333 d_mids=[0.1667, 0.1667]
inner count scale: 6.0000
outer d_mids after the inner compositor: [0.3333]
```

The inner context has the outer share, a third of $\varepsilon$, as its whole budget and splits it in two, so a count inside it gets a scale of 6. The outer context counts the nested compositor as one query, whatever happens inside it, because the composition's guarantee covers everything its queryable answers.

The second feature is a budget in one measure for a mechanism proved in another. The Laplace mechanism's measure is the max divergence of pure differential privacy, and the Gaussian mechanism's is zCDP. When the context's measure is different, `resolve` casts the measurement with the converters of [Combinators](11-combinators.md) before it checks the budget.

```{.python excerpt="utils-cast"}
approx = dp.Context.compositor(
    data=data,
    privacy_unit=dp.unit_of(contributions=1),
    privacy_loss=dp.loss_of(epsilon=EPSILON, delta=DELTA),  # approximate differential privacy
    split_evenly_over=QUERIES,
)
laplace_cast = plain_query(approx).count().laplace().resolve()  # pure DP, cast to approximate DP
gaussian_cast = plain_query(approx).count().gaussian().resolve()  # zCDP, converted and its delta fixed
```

```text
approx share: (0.3333, 3.3e-08)
laplace under (epsilon, delta): Approximate(MaxDivergence) map(1)=(0.3333, 0.0e+00)
gaussian under (epsilon, delta): Approximate(MaxDivergence) map(1)=(0.3333, 3.3e-08)
param after a cast: None
```

Under an $(\varepsilon, \delta)$ budget, `make_approximate` wraps the Laplace mechanism and reports $(\varepsilon, 0)$, so the search fits only $\varepsilon$. The Gaussian mechanism goes through `make_zCDP_to_approxDP` and `make_fix_delta`, which spend the share's slack to convert $\rho$ into $\varepsilon$. But the cast wraps the old measurement in a new one, so the `param` attribute is on the inner measurement, and `param()` reads `None`. To see the scale, run `binary_search_param` on the chain yourself.

```text
laplace under rho: ZeroConcentratedDivergence map(1)=0.0333
gaussian under rho: scale=3.8730
gaussian under epsilon: ValueError: Unable to cast measure from ZeroConcentratedDivergence to MaxDivergence
```

Under a $\rho$ budget, the Laplace mechanism is cast with `make_pureDP_to_zCDP`, and the Gaussian mechanism needs no cast, so its `param()` reports a scale of 3.873. A Gaussian mechanism under a pure $\varepsilon$ budget cannot be cast, because zCDP never implies pure differential privacy, and the context says so. [The full edition's chapter on concentrated differential privacy](../../book-full/part04-privacy-accounting/03-concentrated-dp.md) derives the conversions behind each cast.

Here are the releases of one run, to read against their error bars.

```text
released (one run):
  survey rows 3811875  (true 3811932, within 108.3 with probability 0.95)
  count by hand 94  (true 100, within 9.4 with probability 0.95)
  count through the context 101  (true 100)
```

All three are within their bars. The two counts of the hundred integers came from the same measurement, through the queryable built by hand and through the context. So they differ only by their noise.

## Summary

| Call or concept | What it does | Rule of thumb |
| --- | --- | --- |
| `"i32"`, `int`, `dp.i32` | Three spellings of one type descriptor | `int` is `i32` and `float` is `f64`; NumPy's integers are `i64` |
| `dp.set_default_int_type(dp.i64)` | Changes what `int` means for the rest of the program | The clean fix when all your integers have 64 bits |
| `inferred type is XXX, expected YYY` | Your argument's type does not match the metric's or the domain's | Change the argument to the expected type, not the domain to the argument |
| `dp.binary_search(predicate, bounds, T)` | The value nearest the boundary of a monotone predicate | Pass bounds when the chain is only valid in a range, and `T=int` for an integer |
| `dp.binary_search_param(make_chain, d_in, d_out)` | The smallest scale whose chain checks within the budget | What a context runs for `.laplace()` without a scale |
| `dp.binary_search_chain(...)` | The chain built at the value the search found | It carries no `param`; use `binary_search_param` for the number |
| `dp.serialize`, `dp.deserialize` | A measurement as JSON, and back | The recipe survives; `param`, contexts and Python plugins do not |
| `Context.deserialize_polars_plan(bytes)` | A Polars plan from a client, run on the server's context | The plan carries the query; each context states the guarantee |
| `dp.Context.compositor(...)` | An adaptive composition over the data's space, called on the data | Its queryable holds the data; `d_mids` are the shares |
| `context.query()` | A `Query` whose chain grows with each constructor | A missing number makes a partial chain; one at a time |
| `query.param()` | The scale that `resolve` found | `None` when the scale was given or the measure was cast |
| `Query.compositor(...)` | A context nested inside one share | The outer context counts it as one query |
| A Gaussian query under a pure budget | `ValueError: Unable to cast measure ...` | zCDP never implies pure differential privacy; state the budget with a $\delta$ or as $\rho$ |

: The utilities and the pieces of the Context API, what each does, and the rule of thumb that goes with it. {#tbl-utils-summary}

## References

- OpenDP documentation: "Typing", "Parameter Search" and "Serialization" under "Utilities" in the API user guide; "Context" and "API Comparison" in the same guide; and the entry `Inferred type is XXX, expected YYY` of "Troubleshooting" in Getting Started.
- Gaboardi, Hay and Vadhan, "A Programming Framework for OpenDP" (2020): the domains, metrics, measures and maps that the type descriptors and the searches work on.
- Dwork, Rothblum and Vadhan, "Boosting and Differential Privacy" (2010): composition of adaptively chosen mechanisms, which the compositor behind a context implements.
- Bun and Steinke, "Concentrated Differential Privacy: Simplifications, Extensions, and Lower Bounds" (2016): zCDP and its conversions to approximate differential privacy, which the measure casts use.
- The full edition's chapters on [the Laplace and Gaussian mechanisms](../../book-full/part03-dp-foundations/03-laplace-gaussian.md), [composition](../../book-full/part03-dp-foundations/04-composition.md) and [concentrated differential privacy](../../book-full/part04-privacy-accounting/03-concentrated-dp.md) derive the results this chapter states.
