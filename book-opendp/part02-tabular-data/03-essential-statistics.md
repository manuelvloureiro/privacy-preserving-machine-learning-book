# Essential statistics

With a context on a table, you can ask the everyday questions. How many responses are there, how many hours do people work in total and on average, and what is a typical week? This chapter answers them on the Labour Force Survey (LFS) sample with OpenDP's Polars queries. Each answer is a statistic with noise. The library finds the noise scale from a bound on what one person can change and from a share of the budget. You read that scale and its error bar before you release anything.

The chapter follows the documentation's tutorial on essential statistics and its page on aggregation. You count rows, decide where bounds come from, sum a column, release a mean in two ways, and choose a median and quartiles from candidates. [The practitioner edition's workflow on releasing statistics](../../book-practitioner/part03-dp-foundations/12-release-statistics.md) does the same work on a retail dataset.

The script [opendp_essential_statistics.py](../../code/part02-tabular-data/opendp_essential_statistics.py) runs every query on OpenDP 0.16.0. It reads the sample lazily through `practitioner_datasets.france_lfs()`, as the documentation reads the same file through `dp.examples.get_france_lfs_path()`. The scales, error bars, refusals and true values that it prints are the same on every run. The released values are not, because OpenDP's noise cannot be seeded, so this page shows them in one block marked as one run.

```bash
uv run python code/part02-tabular-data/opendp_essential_statistics.py
```

## The context for this chapter {#sec-stats-context}

The context is the one that [the previous chapter](../part01-getting-started/02-the-data-and-the-context.md#sec-ctx-context) builds, with a privacy unit of 36 rows, one per quarter. Its budget is a privacy parameter $\varepsilon$ of 1 under pure differential privacy (DP), split evenly over five queries. It also has that chapter's margin, a public bound on the number of rows. The new argument `invariant` is for the section on the mean.

```{.python excerpt="stats-context"}
def make_context(frame: pl.LazyFrame, invariant: Literal["keys", "lengths"] | None = None) -> dp.Context:
    """The chapter's context: 36 rows per person, epsilon = 1 split evenly over 5 queries, and one public margin."""
    return dp.Context.compositor(
        data=frame,
        privacy_unit=dp.unit_of(contributions=CONTRIBUTIONS),  # neighbours add or remove one person's 36 rows
        privacy_loss=dp.loss_of(epsilon=EPSILON),  # pure differential privacy
        split_evenly_over=QUERIES,  # each query gets epsilon / 5
        margins=[dp.polars.Margin(max_length=MAX_LENGTH, invariant=invariant)],  # a public bound on the rows
    )
```

The chapter's five releases come from this one context. A query takes its share of the budget when you build it, so the comparisons on this page, which only read scales, use new contexts with the same settings. A real analysis builds one context and plans its shares before the first query.

## Counting responses {#sec-stats-count}

The first release on the survey is its number of rows. The query selects `dp.len`, OpenDP's private version of Polars' `len`, and the context fills in the noise.

```{.python excerpt="stats-count"}
count_query = context.query().select(dp.len(signed=True))
count_summary = count_query.summarize(alpha=0.05)
dp_count = count_query.release().collect().item()
```

Before `release`, call `summarize`. It describes the measurement that the context built, without running it, one row per statistic.

```text
shape: (1, 5)
┌────────┬──────────────┬─────────────────┬───────┬────────────┐
│ column ┆ aggregate    ┆ distribution    ┆ scale ┆ accuracy   │
│ ---    ┆ ---          ┆ ---             ┆ ---   ┆ ---        │
│ str    ┆ str          ┆ str             ┆ f64   ┆ f64        │
╞════════╪══════════════╪═════════════════╪═══════╪════════════╡
│ len    ┆ Frame Length ┆ Integer Laplace ┆ 180.0 ┆ 539.731115 │
└────────┴──────────────┴─────────────────┴───────┴────────────┘
```

The aggregate is the length of the frame, and the distribution is integer Laplace, because a count is an integer. The scale comes from the privacy unit. One person more or less changes up to 36 rows, so the **sensitivity** of the count, the most that one person can change it, is 36. The Laplace scale is the sensitivity over the query's share of $\varepsilon$, so 36 over 0.2. [The full edition's chapter on the Laplace and Gaussian mechanisms](../../book-full/part03-dp-foundations/03-laplace-gaussian.md) derives why that scale gives $\varepsilon$-DP. The accuracy is the error bar at the level that you passed as `alpha`: with probability 0.95, the released count is within 539.73 of the true one.

The error bar depends on `alpha`, not on the data, so both numbers below cost nothing to read. If you ask for more confidence, the bar widens, and at `alpha=0.01` the same scale gives 829.43.

```text
count: scale=180.0 95% accuracy=539.73 99% accuracy=829.43
sensitivity 36 over epsilon 0.2: scale 180.0
```

The bar covers the noise and only the noise. It does not cover the sampling error between this sample and the population of France, or the recoding and suppression that Eurostat applied before publication. And it counts responses, not people. The argument `signed=True` keeps the released count a signed integer, so a small count can come out negative when the noise is larger. Without the argument, the library clamps a negative count to zero, which biases small counts upwards, and it warns that the default will change.

```text
without signed=True: FutureWarning: The default value is currently signed=False, and counts are clamped at zero, consistent with past releases, but the default will be changing to signed=True.
```

A column has its own counts, which the documentation's aggregation page lists. On a column, `.dp.len` counts the rows with nulls, `.dp.count` the rows that are not null, and `.dp.null_count` the rows that are. So the column counts answers to the hours question, and its null count measures non-response. Then `.dp.n_unique` counts distinct values, with null as one value. One person's rows change each count by at most 36, so each has the scale of the frame length when it is alone in a query. The excerpt asks for all five in one query.

```{.python excerpt="stats-variants"}
five_counts = variants.query().select(
    dp.len(signed=True),  # rows of the frame
    hwusual.dp.len(signed=True).alias("expr_len"),  # rows of the column, nulls included
    hwusual.dp.count(signed=True).alias("count"),  # rows of the column that are not null
    hwusual.dp.null_count(signed=True).alias("null_count"),  # rows of the column that are null
    hwusual.dp.n_unique(signed=True).alias("n_unique"),  # distinct values, null among them
)
five_summary = five_counts.summarize(alpha=0.05)
```

```text
HWUSUAL len alone: scale=180.0
HWUSUAL count alone: scale=180.0
HWUSUAL null_count alone: scale=180.0
HWUSUAL n_unique alone: scale=180.0
five counts in one query: scale=900.0 each, 95% accuracy=2696.7
```

Five statistics in one query share its budget equally, so each gets a fifth of it and five times the scale. Every query on this page divides its share evenly among its statistics, unless you set their scales yourself. [Grouping](04-grouping.md) gets the null and the non-null count together for one count's noise, by grouping on whether the value is null.

## Where bounds come from {#sec-stats-bounds}

Every statistic after the count needs the range of its values. A sum or a mean needs a lower and an upper bound, and a median needs a list of candidates. The bounds set the sensitivity, because one person's rows can move a sum by at most their number times the width of the bounds. That is the clipping idea, which [the full edition's chapter on sensitivity](../../book-full/part03-dp-foundations/07-sensitivity.md) derives. Bounds that are too wide add more noise than the data need. Bounds that are too narrow clip real values and bias the result, and the error bar does not show that bias.

Take bounds from domain knowledge, from a comparable public dataset, or from a share of the budget spent on a coarse quantile query. Never take them from the private data. A bound read off the data, such as its maximum, depends on one person's value and leaks it, outside the guarantee. This page uses domain knowledge: a working week of 0 to 80 hours, with the whole hours from 20 to 59 as candidates.

## The sum of hours {#sec-stats-sum}

The column `HWUSUAL` holds the hours that a person usually works in a week, with 99 for "not applicable". So the query first filters out those rows, and OpenDP accounts for the filter as part of the query. Then `.dp.sum` clips every value into the bounds, sums, and adds noise.

```{.python excerpt="stats-sum"}
hours: Any = pl.col("HWUSUAL").cast(int)  # Any: the .dp namespace is added at run time
sum_query = (
    context.query()
    .filter(pl.col("HWUSUAL") != 99.0)  # 99 means "not applicable"
    .select(hours.fill_null(35).dp.sum(bounds=(0, 80)))  # fill, clip, sum, add noise
)
sum_summary = sum_query.summarize(alpha=0.05)
dp_sum = sum_query.release().collect().item()
```

```text
sum HWUSUAL: Sum, Integer Laplace, scale=14,400.00 95% accuracy=43,139.0
sensitivity 36 x 80 = 2,880 over epsilon 0.2: scale 14,400
```

One person owns at most 36 rows of at most 80 hours each, so the sensitivity is their product. An error bar of 43,139 hours is small against a total near 56 million. The noise does not grow with the number of rows, so the relative error decreases as the table grows.

The sum needs the margin, and it says so when the margin is missing.

```text
without a margin: must specify 'max_length' in a margin with by=[]
```

As [the previous chapter](../part01-getting-started/02-the-data-and-the-context.md#sec-ctx-margin) explains, the library must know the largest number of rows to account for overflow and rounding in a sum. The `by=[]` in the message names the margin of the whole table, and [Grouping](04-grouping.md) adds margins per group. The refusal does not depend on the data, as a snippet on three rows shows.

```python
import opendp.prelude as dp
import polars as pl

dp.enable_features("contrib")
context = dp.Context.compositor(
    data=pl.LazyFrame({"hours": [35.0, 39.0, 40.0]}),
    privacy_unit=dp.unit_of(contributions=1),
    privacy_loss=dp.loss_of(epsilon=1.0),
    split_evenly_over=1,
)
try:
    context.query().select(pl.col("hours").dp.sum(bounds=(0.0, 80.0))).summarize()
except dp.OpenDPException as err:
    print(err.message)  # must specify 'max_length' in a margin with by=[]
```

The integer cast is the second adjustment. The column holds floats, and a float sum pays for the rounding of up to `max_length` additions. At this chapter's bound the price is small, but a careless bound of a billion rows raises the float scale to 192,035.68. The cast removes the rounding and its dependence on `max_length`, so cast whenever the values are whole numbers.

```text
float sum, max_length 5,400,000: HWUSUAL: Sum, Float Laplace, scale=14,405.18 95% accuracy=43,154.1
float sum, max_length 1,000,000,000: HWUSUAL: Sum, Float Laplace, scale=192,035.68 95% accuracy=575,287.5
int sum, max_length 1,000,000,000: HWUSUAL: Sum, Integer Laplace, scale=14,400.00 95% accuracy=43,139.0
```

The third adjustment is `fill_null`, because a sum has no value when a row is null. If you write the clip and the sum by hand, the library refuses a nullable column. The shortcut `.dp.sum` instead fills a null with the midpoint of the bounds, 40 here. With `fill_null(35)` you choose the legal working week in France, a public fact. On this data the choice has no effect: Polars drops a row whose test `!= 99` is null, so the filter drops every null. To confirm the midpoint rule, the script also sums the whole column, nulls included.

```text
sum without fill_null: scale=14,400.0 (accepted; a null becomes the midpoint 40)
clip, sum and laplace by hand, without fill_null: input data (HWUSUAL) might contain nulls. Preprocess your data with `.fill_null`.
```

The error bar is about the *clipped* sum. A value above 80 becomes 80 before the sum, and no error bar reports that. Here no applicable value is above the bound, as the check at the end shows.

## The mean {#sec-stats-mean}

A mean is a sum divided by a count, and under the add/remove relation both are private. The documentation recommends that you release the two in one query and divide afterwards. The division of two released numbers is post-processing, which costs nothing, as [the full edition's chapter on properties](../../book-full/part03-dp-foundations/02-properties.md) proves.

```{.python excerpt="stats-mean-joint"}
joint_query = (
    context.query()
    .filter(pl.col("HWUSUAL") != 99.0)
    .select(hours.dp.sum(bounds=(0, 80)), dp.len(signed=True))  # a null becomes the midpoint, 40
)
joint_summary = joint_query.summarize(alpha=0.05)
joint = joint_query.release().collect().with_columns(mean=pl.col("HWUSUAL") / pl.col("len"))
```

```text
joint HWUSUAL: Sum, Integer Laplace, scale=28,800.00 95% accuracy=86,277.6
joint len: Frame Length, Integer Laplace, scale=360.00 95% accuracy=1,079.0
```

The two statistics split the query's share evenly, so each has twice the scale that it had alone. But the pair costs only one share, and one released count can also divide the sums of other columns. On this context, `.dp.mean` releases the same noisy sum and noisy length, as the script shows.

```text
dp.mean on the add/remove context: HWUSUAL: Sum, Integer Laplace, scale=28,800.00 95% accuracy=86,277.6
dp.mean on the add/remove context: HWUSUAL: Length, Integer Laplace, scale=360.00 95% accuracy=1,079.0
```

The alternative is to declare the number of rows public, which the documentation calls bounded DP. The neighbouring relation becomes replace-one: two neighbours have the same number of rows and differ in one person's values. In a margin, `invariant="lengths"` makes the number of rows public, and `.dp.mean` then adds noise to the sum only.

```{.python excerpt="stats-mean-bounded"}
applicable_frame = frame.filter(pl.col("HWUSUAL") != 99.0)  # filtered outside the context: see the text
bounded = make_context(applicable_frame, invariant="lengths")  # the number of rows is public
mean_query = bounded.query().select(hours.dp.mean(bounds=(0, 80)))
mean_summary = mean_query.summarize(alpha=0.05)
dp_mean = mean_query.release().collect().item()
```

```text
bounded mean HWUSUAL: Sum, Integer Laplace, scale=7,200.00 95% accuracy=21,569.8
bounded mean HWUSUAL: Length, Integer Laplace, scale=0.00 95% accuracy=none
```

The sum's scale is half of the scale with a whole share under add/remove. When the length is fixed, a changed row counts as one removal and one addition, so a distance of 36 means at most 18 changed rows. The length has a scale of zero, because it is released exactly. That is the weaker guarantee. The release still hides what a person answered, but not whether the person is in the table.

The excerpt shows a second cost: the filter on the code 99 runs outside the context. Inside the context, a filter makes the number of rows unknown again. Then the invariant is lost, and `.dp.mean` falls back to a noisy sum and a noisy length, as the script shows. A filter outside the context is sound only if the number of applicable responses is truly public. If it is not, use the joint query.

```text
dp.mean after a filter inside the bounded context: HWUSUAL: Sum, Integer Laplace, scale=28,800.00 95% accuracy=86,277.6
dp.mean after a filter inside the bounded context: HWUSUAL: Length, Integer Laplace, scale=360.00 95% accuracy=1,079.0
```

## The median and the quartiles {#sec-stats-quantiles}

You cannot release a median by adding noise to it. When few values lie near the middle, one person's rows can move the median by any amount, so no finite scale covers every dataset. So OpenDP chooses the median from **candidates** that you supply. Each candidate gets a score for how unbalanced the counts below and above it are. OpenDP adds exponential noise to every score and releases the candidate with the smallest noisy score. That is the exponential mechanism, which [the full edition's chapter on selection](../../book-full/part03-dp-foundations/05-selection.md) derives. The candidates here are the whole hours from 20 to 59.

```{.python excerpt="stats-median"}
candidates = list(range(20, 60))  # whole hours from 20 to 59
median_query = context.query().filter(pl.col("HWUSUAL") != 99.0).select(hours.dp.median(candidates))
median_summary = median_query.summarize(alpha=0.05)
dp_median = median_query.release().collect().item()
```

```text
median HWUSUAL: 0.5-Quantile, ExponentialMin, scale=360.00 95% accuracy=none
score sensitivity 36: scale 2 x 36 / 0.2 = 360
median without a margin: Must know max_length
```

The aggregate reads 0.5-quantile because `.dp.median` calls `.dp.quantile` with `alpha=0.5`. The scale is twice the score's sensitivity over the query's share, a factor that comes from the analysis of the exponential mechanism. The accuracy is empty because the noise goes into the scores, not into the answer, so there is no error bar in hours. Instead, a candidate whose score is far above the best one is very unlikely to win, as the check at the end shows. The quantile also needs the margin, because the scores are integers whose size depends on the number of rows.

A quantile other than the median takes `alpha`, the share of values below it. The excerpt asks for the three quartiles in one query.

```{.python excerpt="stats-quartiles"}
quartile_query = (
    context.query()
    .filter(pl.col("HWUSUAL") != 99.0)
    .select(hours.dp.quantile(a, candidates).alias(f"{a}-Quantile") for a in [0.25, 0.5, 0.75])
)
quartile_summary = quartile_query.summarize()
dp_quartiles = quartile_query.release().collect().row(0)
```

```text
quartiles 0.25-Quantile: 0.25-Quantile, ExponentialMin, scale=3,240.00 95% accuracy=none
quartiles 0.5-Quantile: 0.5-Quantile, ExponentialMin, scale=1,080.00 95% accuracy=none
quartiles 0.75-Quantile: 0.75-Quantile, ExponentialMin, scale=3,240.00 95% accuracy=none
0.25-quantile alone: scale=1,080.0
```

Three quantiles in one query split its share three ways, so the median's scale triples. The quartiles' scales are three times larger again, but the median did not get more of the budget. A quantile's score is an integer scaled by the denominator of `alpha`, four for a quartile and two for the median. So one person's rows move a quartile's score three times as far. The last line confirms the even split: the first quartile alone gets a third of its scale in the trio.

## A check against the truth {#sec-stats-check}

The sample is public, so the script computes the true values and reads one run's releases against them. On private data, do this check on public or historical data during development, and release the real data once.

```text
true responses: 3,811,932
true responses with HWUSUAL not 99: 1,496,192 (nulls among them: 0; in the whole column: 20,819)
true sum of hours after the filter, clipped to [0, 80]: 56,315,594 (values above 80: 0)
true sum of the whole column, clipped: 240,742,034 with nulls as 40, 239,909,274 with nulls dropped
true mean: 37.64  (bounded mean 95% accuracy: 0.0144)
true quartiles: 35, 37, 40
score gap from the true quartile to the nearest other candidate: 0.25: 219,506  0.5: 74,383  0.75: 116,811
```

The second and third lines settle two open questions: the filter leaves no nulls, and no applicable value is above 80. The bounded mean's error bar is the sum's error bar divided by the exact length. The last line takes the place of an error bar for the quantiles. Each nearest other candidate scores at least 74,383 above the true quartile, against a noise scale of at most 3,240. So a wrong candidate is far too unlikely to see, and the test requires the exact quartiles.

```text
released (one run):
  responses 3,811,961  (true 3,811,932, within 539.7 with probability 0.95)
  sum of hours 56,308,788  (true 56,315,594, within 43,139.0)
  joint: sum 56,335,491  count 1,495,681  mean 37.67  (true 37.64)
  bounded mean 37.6411  (true 37.6393, within 0.0144)
  median 37  quartiles 35, 37, 40  (true 37; 35, 37, 40)
  sum of the whole column 240,726,542  (true with nulls as 40: 240,742,034)
```

The count and the sum are inside their error bars, as they must be about 19 times in 20. A count error of a few hundred is small on 3.8 million responses. The joint mean is off in the second decimal, because it divides two noisy numbers. The bounded mean is inside a bar of about a hundredth of an hour, which is what a public length buys. The median and the quartiles are exact. The sum of the whole column is near the total that counts each null as 40 hours, far from the total without them.

## Summary

| Call or concept | What it does | Rule of thumb |
| --- | --- | --- |
| `dp.unit_of(contributions=36)` | The privacy unit: neighbours differ by one person's rows, up to 36 | Count the most rows one person can own, not the typical number |
| `dp.polars.Margin(max_length=...)` | A public bound on the number of rows | Sums, means and quantiles refuse to run without it; take it from public facts |
| `dp.len(signed=True)` | The number of rows, with Laplace noise of scale sensitivity over $\varepsilon$ | Without `signed=True` a negative count is clamped to zero |
| `.dp.len`, `.dp.count`, `.dp.null_count`, `.dp.n_unique` | A column's rows, non-nulls, nulls and distinct values | Each is as sensitive as the frame length; several in one query share its $\varepsilon$ |
| `query.summarize(alpha=0.05)` | The column, aggregate, distribution, scale and error bar of each statistic | Read it before `release`; a smaller `alpha` widens the bar |
| `.dp.sum(bounds=(0, 80))` | Fills nulls with the midpoint, clips, sums, adds noise | Cast to an integer; the error bar excludes the bias of clipping |
| Sum and `dp.len` in one query | A mean under add/remove, by dividing afterwards | Each statistic gets twice the scale; the division is free post-processing |
| `invariant="lengths"` and `.dp.mean` | A bounded-DP mean: noise on the sum, the exact length | Weaker guarantee; filter outside the context only if the row count is truly public |
| `.dp.median(candidates)`, `.dp.quantile(alpha, candidates)` | The candidate nearest the quantile, by the exponential mechanism | No error bar; candidates from domain knowledge; several quantiles split the share evenly |

: The calls of this chapter, what each does, and the rule of thumb that goes with it. {#tbl-stats-summary}

## References

- OpenDP documentation: "Essential Statistics" in the tutorial on tabular data, and "Aggregation" in the Polars user guide.
- Dwork, McSherry, Nissim and Smith, "Calibrating Noise to Sensitivity in Private Data Analysis" (2006): the Laplace mechanism behind the counts and sums.
- McSherry and Talwar, "Mechanism Design via Differential Privacy" (2007): the exponential mechanism behind the median and the quantiles.
- The full edition's chapters on [the Laplace and Gaussian mechanisms](../../book-full/part03-dp-foundations/03-laplace-gaussian.md), [post-processing and composition](../../book-full/part03-dp-foundations/02-properties.md), [selection](../../book-full/part03-dp-foundations/05-selection.md) and [sensitivity](../../book-full/part03-dp-foundations/07-sensitivity.md) derive the results this chapter states.
