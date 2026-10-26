# A first release

Suppose you hold a list of a hundred people's scores and want to publish how many there are and their average, without letting anyone work out any one person's score. This chapter does exactly that with OpenDP, and on the way it introduces the one idea the whole library is built on. A **differentially private** release is a randomized computation whose noise is calibrated to how much one person can change the answer. OpenDP's job is to do the calibration, to refuse a computation it cannot calibrate, and to keep count of what has been spent.

The chapter follows the documentation's quickstart and its typical workflow. You will install the library, build a Laplace mechanism by hand and read its guarantee, then let the Context API do the work: state the privacy unit and the budget, hand over the data once, and release a count and a mean. By the end you will be able to explain what each line of a first OpenDP program promises.

The script [opendp_first_release.py](../../code/part01-getting-started/opendp_first_release.py) runs every step on OpenDP 0.16.0, with a hundred integers drawn by a seeded generator standing in for the scores. The noise scales, the error bars and the privacy maps it prints are the same on every run. The released values are not, because OpenDP draws its noise from a cryptographic generator that cannot be seeded, so this page shows them in one block marked as one run.

```bash
uv run python code/part01-getting-started/opendp_first_release.py
```

## Install the library {#sec-first-install}

OpenDP is a Python package with a Rust core, and it reads tables through Polars, so install it with the Polars extra.

```bash
pip install 'opendp[polars]'
```

The repository pins `opendp[polars]==0.16.0` with Polars 1.36.1, which is what every number in this book comes from. Then import the prelude, which gathers the library's modules under one name, and switch on the `contrib` feature.

```python
import opendp.prelude as dp

dp.enable_features("contrib")
print(dp.__version__)  # 0.16.0
```

The feature flag is a statement about trust. Every component of OpenDP is meant to come with a written proof that it does what it claims, and a review of that proof. Components whose proof has not finished that process are marked `contrib`, and the library will not run them unless you opt in. Almost everything in this book, from the Context API to the Polars queries, is still `contrib` in version 0.16.0, so the line goes at the top of every script. A reviewer who needs only vetted components can leave it out and see what stops working.

## One noisy number {#sec-first-number}

The simplest private computation adds noise to one number. OpenDP builds it from a **space**, which is a pair of a domain and a metric, and a constructor that turns the space into a **measurement**, which is a randomized function with a privacy guarantee.

```python
import opendp.prelude as dp

dp.enable_features("contrib")
space = dp.atom_domain(T=float, nan=False), dp.absolute_distance(T=float)
laplace = space >> dp.m.then_laplace(scale=1.0)
print(laplace.map(1.0))  # 1.0
print(laplace.map(2.0))  # 2.0
print(laplace.check(1.0, 0.5))  # False
```

The domain says what the input may be, here any finite float. The metric says how two inputs are compared, here by their absolute difference. The constructor `then_laplace` adds a draw from the Laplace distribution with scale 1 to its input, and the `>>` operator attaches it to the space. Calling `laplace(123.0)` gives a noisy value near 123, different each time.

The guarantee lives in the **privacy map**. Its argument is a bound on how far apart two inputs can be, which differential privacy calls the **sensitivity** of the quantity you are releasing. Its value is the largest privacy loss $\varepsilon$ the mechanism can incur on two such inputs. For Laplace noise the map is the sensitivity divided by the scale, so a sensitivity of 1 at scale 1 costs $\varepsilon = 1$ and a sensitivity of 2 costs $\varepsilon = 2$. The `check` method asks whether the map at a given sensitivity stays within a given $\varepsilon$, and it answers that sensitivity 1 at scale 1 does not fit within $\varepsilon = 0.5$. Doubling the scale halves the cost, which the script confirms.

```text
laplace scale 1: map(1.0)=1.0 map(2.0)=2.0
laplace scale 1: check(1.0, 1.0)=True check(1.0, 0.5)=False
laplace scale 2: map(1.0)=0.5
```

What $\varepsilon$ promises is that changing one person's data changes the probability of any output by a factor of at most $e^\varepsilon$. The full edition derives the Laplace mechanism's guarantee, and why the scale must be the sensitivity over $\varepsilon$, in [its chapter on the Laplace and Gaussian mechanisms](../../book-full/part03-dp-foundations/03-laplace-gaussian.md). For now the point is where the guarantee comes from: not from the noise alone, but from the noise together with a bound on what one person can change. You supplied that bound as the argument of `map`. The rest of the chapter shows how the library works it out for you.

## The five steps of an analysis {#sec-first-steps}

The documentation's typical workflow has five steps, and every analysis in this book follows them. The first three are decisions you make before touching the data. The fourth hands the data to the library, and the fifth asks it questions.

The first step is to identify the **privacy unit**, which is what the guarantee protects. The documentation calls it the unit of privacy. Here each person contributes one row, so the unit is one row, and two datasets are **neighbours** when one is the other with a row added or removed. OpenDP writes that as a metric, the symmetric distance, and a distance of 1 under it.

```python
import opendp.prelude as dp

dp.enable_features("contrib")
print(dp.unit_of(contributions=1))  # (SymmetricDistance(), 1)
print(dp.loss_of(epsilon=1.0))  # (MaxDivergence, 1.0)
print(dp.loss_of(epsilon=1.0, delta=1e-7))  # (Approximate(MaxDivergence), (1.0, 1e-07))
print(dp.loss_of(rho=0.1))  # (ZeroConcentratedDivergence, 0.1)
```

The second step is to set the **privacy budget**, which the documentation calls the privacy loss. It is a measure, which says how the distributions of two releases are compared, and a bound on that comparison. Pure differential privacy compares them by the max divergence and bounds it by $\varepsilon$, so `loss_of(epsilon=1.0)` is the pair you see printed. The other two lines show the budgets later chapters use. Adding a slack $\delta$ gives approximate differential privacy, and a budget $\rho$ gives zero-concentrated differential privacy, which [Assessing utility and choosing the noise](../part02-tabular-data/07-utility-and-noise.md) introduces. A budget of $\varepsilon = 1$ is a common starting point, and the documentation points to the OpenDP deployments registry for the values real releases have used.

The third step is to collect **public information**: facts about the data that you choose not to protect because they are known anyway. Here they are the bounds on a score, 0 to 100, and a value to stand in for a missing score, the midpoint 50. Bounds must come from the domain or from another public source, never from looking at the private data, because a bound read off the data leaks it.

The fourth step is to **mediate access**. The script builds a **context**, which holds the data, the privacy unit, the budget and the rule for splitting it, and from then on every computation on the data goes through it.

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

The context is a **compositor**, because the budget of several releases adds up. Three queries at a third of $\varepsilon$ each cost $\varepsilon$ in all, which is the basic composition rule that [the full edition's chapter on properties](../../book-full/part03-dp-foundations/02-properties.md) proves. The script prints the split.

```text
privacy unit: (SymmetricDistance(), 1)
privacy loss: (MaxDivergence, 1.0)
bounds: (0.0, 100.0)  imputed value: 50
budget per query: [0.3333, 0.3333, 0.3333]
```

The fifth step is to submit queries, which the next section does. One habit from the documentation is worth adopting now: once the context exists, stop touching the data directly, and delete your own reference to it. Anything you compute outside the context is outside the guarantee.

## A count and a mean {#sec-first-count}

A query is built by chaining methods on `context.query()`, each a constructor of the framework with its space left out, because the context knows the space. The count query counts the rows and adds Laplace noise.

```{.python excerpt="first-count"}
count_query = context.query().count().laplace()
count_scale = count_query.param()  # the scale the budget allows: sensitivity 1 over epsilon / 3
count_accuracy = dp.discrete_laplacian_scale_to_accuracy(scale=count_scale, alpha=0.05)
dp_count = count_query.release()
```

Nothing in `laplace()` says what scale to use. The context fills it in, by searching for the smallest scale whose privacy map at the unit's distance still fits within the query's share of the budget, and `param` reports what it found. A larger scale would also fit, but it would add more noise than the budget requires. Adding or removing one row changes a count by 1, so the sensitivity is 1, and the share is $\varepsilon/3$, so the scale is 3. The accuracy function turns that scale into an error bar: with probability 0.95, the released count is within about 9.4 of the true count. The count is an integer, so the noise is the discrete Laplace distribution and the accuracy function is the discrete one.

```text
count: scale=3.0000 95% accuracy=9.4457
```

Read those two numbers before you release anything. They depend on the budget and the sensitivity, not on the data, so you can decide whether the error bar is acceptable without spending anything.

The mean needs more preparation, because the mean of a list is not a well-behaved function of the list until three things are fixed.

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

Imputation removes missing values, since the mean of a list with a hole in it is undefined. Clamping bounds every value, since without bounds one person could move the mean by any amount and no finite noise would hide them. Resizing fixes the number of rows, since the mean divides by it, and the sensitivity of a mean is only bounded when that divisor is known. The size passed in is the count just released, which is allowed because a released value is public and using it again costs nothing. That is the post-processing property, proved in [the full edition](../../book-full/part03-dp-foundations/02-properties.md). If the noisy count is too small, the resize pads the list with the imputed value, and if it is too large, it drops rows, so the released mean is the mean of a slightly altered list. The error bar below does not cover that alteration.

Because the scale of the mean depends on the released count, it changes from run to run. At the true size of 100 the script computes the scale and its error bar at development time, which is fine here because the data are a public toy.

```text
mean at size 100: scale=3.0000 95% accuracy=8.9872
```

A scale of 3 on a mean of values between 0 and 100 gives an error bar of about 9 points, which is wide. The reason is the budget. A third of $\varepsilon = 1$ on a hundred rows is little, and the Laplace mechanism's error does not shrink with the number of rows when the sensitivity is a fixed fraction of the range. [Essential statistics](../part02-tabular-data/03-essential-statistics.md) shows how a larger dataset and a sum released alongside a count do better.

## What the context refuses {#sec-first-refused}

The context keeps count. After the count and the mean, two of the three shares are spent, and the script reads them back.

```text
budget spent after two queries: [0.3333, 0.3333]
```

A third query takes the last share. A fourth has nothing left to take, and the context raises an error instead of releasing anything.

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

This refusal is the reason to use a context rather than loose mechanisms. With the framework alone, nothing stops you from calling a measurement twice and spending twice, and the guarantee you can claim is whatever you tracked by hand. With a context, the budget is a hard limit, and the claim is what you declared at the start.

Finally the releases of one run, read against the error bars.

```text
released (one run):
  count 105  (true 100, within 9.4 with probability 0.95)
  mean scale 2.8571  mean 53.32  (true 50.93)
  third query, a second count 108
```

The count landed within its error bar, as it should about 19 times in 20. The mean's scale came out below 3 because the released count was above 100, and the mean itself landed within its bar too. Comparing a release with the truth is possible here only because the data are a toy. On real data you cannot run that check, so you make these decisions on public or historical data while you develop, and release the real data once.

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
