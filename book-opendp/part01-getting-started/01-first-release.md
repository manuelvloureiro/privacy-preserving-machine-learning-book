# A first release

Suppose you hold the scores of a hundred people. You want to publish their number and their average, so that nobody can work out any one person's score. This chapter does that with OpenDP, and it introduces the idea that the whole library is built on. A **differentially private** release is a randomized computation whose noise is calibrated to how much one person can change the answer. OpenDP does the calibration, refuses a computation that it cannot calibrate, and keeps count of what you spend.

The chapter follows the documentation's quickstart and its typical workflow. You build a Laplace mechanism by hand and read its guarantee. Then the Context API does the work. You state the privacy unit and the budget, give the data once, and release a count and a mean.

The script [opendp_first_release.py](../../code/part01-getting-started/opendp_first_release.py) runs every step on OpenDP 0.16.0, with a hundred integers from a seeded generator in place of the scores. The noise scales, the error bars and the privacy maps that it prints are the same on every run. The released values are not, because OpenDP's noise comes from a cryptographic generator without a seed. So this page shows them in one block, marked as one run.

```bash
uv run python code/part01-getting-started/opendp_first_release.py
```

## Install the library {#sec-first-install}

OpenDP is a Python package with a Rust core, and it reads tables through Polars, so install it with the Polars extra.

```bash
pip install 'opendp[polars]'
```

Every number in this book comes from `opendp[polars]==0.16.0` with Polars 1.36.1, which the repository pins. Then import the prelude, which gathers the library's modules under one name, and switch on the `contrib` feature.

```python
import opendp.prelude as dp

dp.enable_features("contrib")
print(dp.__version__)  # 0.16.0
```

The feature flag is about trust. Each component of OpenDP must come with a written proof that it does what it claims, and a review of that proof. Until the review is complete, the component is marked `contrib`, and the library runs it only if you opt in. In version 0.16.0, almost everything in this book is still `contrib`, so the line goes at the top of every script.

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

The domain says what the input can be, here any finite float. The metric says how to compare two inputs, here by their absolute difference. The constructor `then_laplace` adds a draw from the Laplace distribution with scale 1 to its input, and the `>>` operator attaches it to the space.

The guarantee is in the **privacy map**. Its argument is a bound on how far apart two inputs can be, which differential privacy calls the **sensitivity** of the released quantity. Its value is the largest privacy loss $\varepsilon$ that the mechanism can have on two such inputs. For Laplace noise, the map is the sensitivity divided by the scale, so at scale 1 the cost equals the sensitivity. The `check` method asks whether the map at a given sensitivity stays within a given $\varepsilon$. If you double the scale, the cost halves, as the script shows.

```text
laplace scale 1: map(1.0)=1.0 map(2.0)=2.0
laplace scale 1: check(1.0, 1.0)=True check(1.0, 0.5)=False
laplace scale 2: map(1.0)=0.5
```

The value $\varepsilon$ promises that a change to one person's data changes the probability of any output by a factor of at most $e^\varepsilon$. [The full edition's chapter on the Laplace and Gaussian mechanisms](../../book-full/part03-dp-foundations/03-laplace-gaussian.md) derives this guarantee. Note where it comes from: not from the noise alone, but from the noise together with a bound on what one person can change. You gave that bound as the argument of `map`, and the rest of the chapter shows how the library finds it for you.

## The five steps of an analysis {#sec-first-steps}

Every analysis in this book follows the five steps of the documentation's typical workflow. The first three are decisions that you make before you touch the data. The fourth gives the data to the library, and the fifth asks it questions.

The first step is to identify the **privacy unit**, which is what the guarantee protects. The documentation calls it the unit of privacy. Here each person contributes one row, so the unit is one row. Two datasets are **neighbours** when one is the other with one more or one less row. OpenDP writes this as a distance of 1 under the symmetric distance.

```python
import opendp.prelude as dp

dp.enable_features("contrib")
print(dp.unit_of(contributions=1))  # (SymmetricDistance(), 1)
print(dp.loss_of(epsilon=1.0))  # (MaxDivergence, 1.0)
print(dp.loss_of(epsilon=1.0, delta=1e-7))  # (Approximate(MaxDivergence), (1.0, 1e-07))
print(dp.loss_of(rho=0.1))  # (ZeroConcentratedDivergence, 0.1)
```

The second step is to set the **privacy budget**, which the documentation calls the privacy loss. It is a measure, which says how to compare the distributions of two releases, together with a bound on that comparison. Pure differential privacy compares them by the max divergence and bounds it by $\varepsilon$, so `loss_of(epsilon=1.0)` gives the printed pair. The other two lines show budgets of later chapters. A slack $\delta$ gives approximate differential privacy. A budget $\rho$ gives zero-concentrated differential privacy, which [Assessing utility and choosing the noise](../part02-tabular-data/07-utility-and-noise.md) introduces. A budget of $\varepsilon = 1$ is a common start, and the OpenDP deployments registry lists the values of real releases.

The third step is to collect **public information**: facts about the data that you do not protect, because they are known anyway. Here they are the bounds on a score, 0 to 100, and the midpoint 50 as the value for a missing score. Take bounds from the domain or from another public source, never from the private data, because a bound read off the data leaks it.

The fourth step is to **mediate access**. The script builds a **context**, which holds the data, the privacy unit, the budget and the rule that splits the budget. Every later computation on the data goes through the context.

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

The context is a **compositor**, because the costs of several releases add up. Three queries at a third of $\varepsilon$ each cost $\varepsilon$ in total, by the basic composition rule that [the full edition's chapter on properties](../../book-full/part03-dp-foundations/02-properties.md) proves. The script prints the split.

```text
privacy unit: (SymmetricDistance(), 1)
privacy loss: (MaxDivergence, 1.0)
bounds: (0.0, 100.0)  imputed value: 50
budget per query: [0.3333, 0.3333, 0.3333]
```

The fifth step is to submit queries, which the next section does. Before that, adopt one habit from the documentation: when the context exists, delete your own reference to the data. Anything that you compute outside the context is outside the guarantee.

## A count and a mean {#sec-first-count}

You build a query by chaining methods on `context.query()`. Each method is a constructor of the framework without its space, because the context knows the space. The count query counts the rows and adds Laplace noise.

```{.python excerpt="first-count"}
count_query = context.query().count().laplace()
count_scale = count_query.param()  # the scale the budget allows: sensitivity 1 over epsilon / 3
count_accuracy = dp.discrete_laplacian_scale_to_accuracy(scale=count_scale, alpha=0.05)
dp_count = count_query.release()
```

Nothing in `laplace()` gives a scale. The context finds the smallest scale whose privacy map, at the unit's distance, fits within the query's share of the budget, and `param` reports it. For the count, the unit's distance of one row is a sensitivity of 1, and the share is $\varepsilon/3$, so the scale is 3. The count is an integer, so the noise is discrete Laplace, and the discrete accuracy function turns the scale into an error bar. With probability 0.95, the released count is within about 9.4 of the true count.

```text
count: scale=3.0000 95% accuracy=9.4457
```

Read those two numbers before you release anything. They depend on the budget and the sensitivity, not on the data, so they cost nothing.

The mean needs more preparation, because the mean of a list is not a well-behaved function of the list until you fix three things.

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

Imputation removes missing values, because the mean of a list with a hole in it has no value. Clamping bounds every value, because without bounds one person could move the mean by any amount, and no finite noise would hide them. Resizing fixes the number of rows, because the mean divides by it, and the sensitivity of a mean has a bound only when that divisor is known. The size is the count released just before. A released value is public, so using it again costs nothing, by the post-processing property that [the full edition](../../book-full/part03-dp-foundations/02-properties.md) proves. If the noisy count is too small, the resize pads the list with the imputed value, and if it is too large, the resize drops rows. So the released mean is the mean of a slightly changed list, and the error bar below does not cover that change.

So the scale of the mean changes from run to run. The data are a public toy, so the script also computes the scale and its error bar at the true size of 100.

```text
mean at size 100: scale=3.0000 95% accuracy=8.9872
```

An error bar of about 9 points on a range of 100 is wide. The cause is the small budget: a third of $\varepsilon = 1$, on rows whose mean one person can move by a fixed fraction of the range. [Essential statistics](../part02-tabular-data/03-essential-statistics.md) shows how a larger dataset, with a sum released together with a count, does better.

## What the context refuses {#sec-first-refused}

After the count and the mean, two of the three shares are spent, and the script reads them back from the context.

```text
budget spent after two queries: [0.3333, 0.3333]
```

A third query takes the last share. A fourth has no share to take, so the context raises an error and releases nothing.

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

This refusal is the reason to use a context. With the framework alone, nothing stops you from calling a measurement twice and spending twice. With a context, the budget is a hard limit, and your guarantee is what you declared at the start.

Here are the releases of one run.

```text
released (one run):
  count 105  (true 100, within 9.4 with probability 0.95)
  mean scale 2.8571  mean 53.32  (true 50.93)
  third query, a second count 108
```

The count is within its error bar, as it must be about 19 times in 20. The mean's scale is below 3 because the released count is above 100, and the mean is also within its bar. You can do this check only because the data are a toy. On real data, make these decisions on public or historical data during development, and release the real data once.

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
