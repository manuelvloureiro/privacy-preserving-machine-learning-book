# A first release

Consider the scores of a hundred people. The goal is to publish their count and average while limiting what the release reveals about any individual's score. This chapter uses OpenDP to make these releases and introduces its central guarantee. A **differentially private** release is a randomized computation whose noise is calibrated to how much one person can change the answer. OpenDP calibrates the noise, rejects computations it cannot calibrate and tracks the privacy budget.

Following the documentation's quickstart, the chapter first builds a Laplace mechanism by hand and reads its guarantee. It then uses an OpenDP context to release a count and a mean.

The script [opendp_first_release.py](../../code/part01-getting-started/opendp_first_release.py) runs every step on OpenDP 0.16.0, using a seeded generator to produce a hundred integers. Noise scales, error bars and privacy maps are reproducible. Released values vary because OpenDP uses an unseeded cryptographic generator, so this page labels them as one run.

```bash
uv run python code/part01-getting-started/opendp_first_release.py
```

## Install the library {#sec-first-install}

OpenDP is a Python package with a Rust core. Install the Polars extra to read tables:

```bash
pip install 'opendp[polars]'
```

The repository pins `opendp[polars]==0.16.0` and Polars 1.36.1; all numbers in this book use those versions. Next, import the prelude, which gathers the library's modules under one name, and enable the `contrib` feature.

```python
import opendp.prelude as dp

dp.enable_features("contrib")
print(dp.__version__)  # 0.16.0
```

The feature flag marks components whose written proofs have not completed review. OpenDP runs a `contrib` component only when you opt in. In version 0.16.0, almost every component used in this book is still `contrib`, so every script enables it.

## One noisy number {#sec-first-number}

The simplest private computation adds noise to one number. OpenDP builds it from a **space**, a pair of a domain and a metric. A constructor then turns the space into a **measurement**, which is a randomized function with a privacy guarantee.

```python
import opendp.prelude as dp

dp.enable_features("contrib")
space = dp.atom_domain(T=float, nan=False), dp.absolute_distance(T=float)
laplace = space >> dp.m.then_laplace(scale=1.0)
print(laplace.map(1.0))  # 1.0
print(laplace.map(2.0))  # 2.0
print(laplace.check(1.0, 0.5))  # False
```

The domain restricts inputs to finite floats. The metric measures their absolute difference. The constructor `then_laplace` adds Laplace noise with scale 1, and `>>` attaches it to the space.

The **privacy map** takes a bound on the distance between inputs, called the **sensitivity** of the released quantity. It returns the largest privacy loss $\varepsilon$ for inputs within that distance. For Laplace noise, the map divides sensitivity by the noise scale. At scale 1, the cost equals the sensitivity. The `check` method tests whether the cost at a given sensitivity is at most a given $\varepsilon$. Doubling the scale halves the cost.

```text
laplace scale 1: map(1.0)=1.0 map(2.0)=2.0
laplace scale 1: check(1.0, 1.0)=True check(1.0, 0.5)=False
laplace scale 2: map(1.0)=0.5
```

The value $\varepsilon$ bounds the factor by which a change to one person's data can alter the probability of any output: at most $e^\varepsilon$. [The full edition's chapter on the Laplace and Gaussian mechanisms](../../book-full/part03-dp-foundations/03-laplace-gaussian.md) derives this guarantee. It depends on both the noise and a bound on one person's effect. This bound is the argument to `map`; the rest of the chapter shows how OpenDP finds it.

## The five steps of an analysis {#sec-first-steps}

The documentation's typical workflow has five steps: specify the privacy unit, budget and public information; provide the data through a context; then submit queries through it.

First identify the **privacy unit**, the entity the guarantee protects. The documentation calls it the unit of privacy. Here each person contributes one row, so neighbouring datasets differ by one row. OpenDP represents this relation as symmetric distance 1.

```python
import opendp.prelude as dp

dp.enable_features("contrib")
print(dp.unit_of(contributions=1))  # (SymmetricDistance(), 1)
print(dp.loss_of(epsilon=1.0))  # (MaxDivergence, 1.0)
print(dp.loss_of(epsilon=1.0, delta=1e-7))  # (Approximate(MaxDivergence), (1.0, 1e-07))
print(dp.loss_of(rho=0.1))  # (ZeroConcentratedDivergence, 0.1)
```

Next set the **privacy budget**, which the documentation calls the privacy loss. It pairs a measure, which compares the distributions of two releases, with a bound on that comparison. Pure differential privacy uses max divergence bounded by $\varepsilon$, so `loss_of(epsilon=1.0)` returns the printed pair. The other two lines show budgets covered later: $\delta$ gives approximate differential privacy, and $\rho$ gives zero-concentrated differential privacy, introduced in [Assessing utility and choosing the noise](../part02-tabular-data/07-utility-and-noise.md). A common starting budget is $\varepsilon = 1$; the OpenDP deployments registry lists budgets from real releases.

Then collect **public information**: facts about the data that the guarantee need not protect. Here the score bounds are 0 and 100, and 50 is the value for a missing score. Take bounds from the domain or another public source, never from private data; deriving a bound from those data would leak information.

The fourth step is to **mediate access** through a **context**. It holds the data, privacy unit and budget, and defines how to split the budget. Every later computation on the data goes through it.

```{.python excerpt="first-context"}
def make_context(data: list[float]) -> dp.Context:
    """The context: the data behind a privacy unit of one row, a budget of epsilon = 1, split evenly over 3 queries."""
    return dp.Context.compositor(
        data=data,
        privacy_unit=dp.unit_of(contributions=1),  # neighbours add or remove one person's row
        privacy_loss=dp.loss_of(epsilon=EPSILON),  # pure differential privacy
        split_evenly_over=QUERIES,  # each query gets epsilon / 3
    )
```

The context is a **compositor**: it adds the costs of multiple releases. Three queries that each use $\varepsilon/3$ cost $\varepsilon$ in total, by the basic composition rule proved in [the full edition's chapter on properties](../../book-full/part03-dp-foundations/02-properties.md). The script prints the split.

```text
privacy unit: (SymmetricDistance(), 1)
privacy loss: (MaxDivergence, 1.0)
bounds: (0.0, 100.0)  imputed value: 50
budget per query: [0.3333, 0.3333, 0.3333]
```

Finally, submit queries through the context. Delete other references to the data after creating it. The context's guarantee does not cover computations outside it.

## A count and a mean {#sec-first-count}

Build a query by chaining methods on `context.query()`. Each method is a framework constructor without an explicit space; the context supplies that space. The count query counts rows and adds Laplace noise.

```{.python excerpt="first-count"}
count_query = context.query().count().laplace()
count_scale = count_query.param()  # the scale the budget allows: sensitivity 1 over epsilon / 3
count_accuracy = dp.discrete_laplacian_scale_to_accuracy(scale=count_scale, alpha=0.05)
dp_count = count_query.release()
```

`laplace()` does not specify a scale. The context chooses the smallest scale whose privacy map at the unit's distance fits the query's budget share; `param` reports that scale. For a count, one row has sensitivity 1. The share is $\varepsilon/3$, so the scale is 3. Because the count is an integer, OpenDP adds discrete Laplace noise. The discrete accuracy function converts the scale to an error bar: with probability 0.95, the released count is within about 9.4 of the true count.

```text
count: scale=3.0000 95% accuracy=9.4457
```

Check both values before release. They depend on the budget and sensitivity, not on the data, and therefore incur no privacy cost.

The mean needs more preparation. Its sensitivity is bounded only after you address missing values, bound each value and fix the list size.

```{.python excerpt="first-mean"}
mean_query = (
    context.query()
    .impute_constant(IMPUTED)  # a missing value becomes the public midpoint
    .clamp(BOUNDS)  # every value is clipped into the public bounds
    .resize(size=dp_count, constant=IMPUTED)  # the mean needs a known size: the released count
    .mean()
    .laplace()
)
mean_scale = mean_query.param()
dp_mean = mean_query.release()
```

Imputation replaces missing values, for which the mean is undefined. Clamping bounds every value; without bounds, one person's contribution could change the mean without limit. Resizing fixes the row count, which is the mean's denominator and is needed to bound sensitivity. Here the size is the previously released count. Reusing that public value costs nothing by the post-processing property proved in [the full edition](../../book-full/part03-dp-foundations/02-properties.md). If the noisy count is too small, resizing pads with the imputed value; if it is too large, resizing drops rows. The resulting mean is therefore computed on a modified list. Its error bar does not include this change.

The mean's scale changes from run to run because it uses the released count. Since this toy dataset is public, the script also reports the scale and error bar at its true size of 100.

```text
mean at size 100: scale=3.0000 95% accuracy=8.9872
```

An error bar of about 9 points is wide on a range of 100. The query receives only $\varepsilon/3$, and one person's row can move the mean by a fixed fraction of the full range. [Essential statistics](../part02-tabular-data/03-essential-statistics.md) shows how releasing a sum and count on a larger dataset improves accuracy.

## What the context refuses {#sec-first-refused}

The count and mean consume two of the three budget shares. The script reads the remaining shares from the context.

```text
budget spent after two queries: [0.3333, 0.3333]
```

The third query uses the last share. A fourth query exceeds the budget, so the context raises an error and releases nothing.

```{.python excerpt="first-refused"}
try:
    context.query().count().laplace().release()
    refusal = "released"
except ValueError as err:
    refusal = str(err)
```

```text
fourth query: Privacy allowance has been exhausted
```

The context enforces the declared budget. Without it, the framework does not prevent repeated calls to a measurement or track their cumulative cost.

Here are the releases of one run.

```text
released (one run):
  count 105  (true 100, within 9.4 with probability 0.95)
  mean scale 2.8571  mean 53.32  (true 50.93)
  third query, a second count 108
```

The count and mean fall within their error bars in this run. The count's 95% error bar covers the true count in about 19 runs out of 20, not every run. The mean's scale is below 3 because the released count exceeds 100. You can compare with the true values only because this dataset is a public toy. For real data, test on public or historical data during development and release the private data once.

## Summary

| Call or concept | What it does | Rule of thumb |
| --- | --- | --- |
| `dp.enable_features("contrib")` | Switches on the components whose proofs are not yet vetted | Put it at the top of every script; leave it out to see what is vetted |
| `space >> dp.m.then_laplace(scale)` | A Laplace measurement on a domain and a metric | The scale is the sensitivity over $\varepsilon$ |
| `measurement.map(d_in)` | The largest $\varepsilon$ for inputs at most `d_in` apart | Read it before releasing; it costs nothing |
| `measurement.check(d_in, d_out)` | Whether the map at `d_in` stays within `d_out` | What the context runs for you when it searches for a scale |
| `dp.unit_of(contributions=1)` | The privacy unit: neighbours add or remove one row | Choose it before anything else; one row per person is the default case |
| `dp.loss_of(epsilon=1.0)` | The privacy budget under pure differential privacy | $\varepsilon = 1$ is a common start; the registry lists real values |
| `dp.Context.compositor(...)` | Holds the data, the unit, the budget and its split | One context per release; delete your own reference to the data |
| `context.query().count().laplace()` | A query whose scale the context fills in from the budget | Read `param()` and the accuracy before `release()` |
| `impute_constant`, `clamp`, `resize` | Make the mean's sensitivity finite | Bounds and imputed values come from public information |
| A fourth query on a budget split in three | `ValueError: Privacy allowance has been exhausted` | The refusal is the point of the context |

: The calls of a first release, what each does, and the rule of thumb that goes with it. {#tbl-first-summary}

## References

- OpenDP documentation: "Quickstart", "Typical Workflow", and "Context" in the API user guide.
- Dwork, McSherry, Nissim and Smith, "Calibrating Noise to Sensitivity in Private Data Analysis" (2006): the Laplace mechanism and the definition it satisfies.
- Gaboardi, Hay and Vadhan, "A Programming Framework for OpenDP" (2020): the framework of domains, metrics, measures and maps behind every call.
- The full edition's chapters on [the definition](../../book-full/part03-dp-foundations/01-definition.md), [post-processing and composition](../../book-full/part03-dp-foundations/02-properties.md) and [the Laplace and Gaussian mechanisms](../../book-full/part03-dp-foundations/03-laplace-gaussian.md) derive the results this chapter states.
