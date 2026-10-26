# Essential statistics

Once a context mediates access to a table, the everyday questions follow. They ask how many responses there are, how many hours people work in total and on average, and what a typical week looks like. This chapter answers them on the Labour Force Survey (LFS) sample with OpenDP's Polars queries, and one idea runs through every answer. A statistic is released by adding noise, and the library works out the scale of that noise from two things you supply. One is a bound on what one person can change, and the other is a share of the budget. You read that scale and its error bar before you release anything, and only then decide whether the release is worth its share.

The chapter follows the documentation's tutorial on essential statistics and its page on aggregation. You will count rows, read a `summarize` table and decide where bounds come from. Then you will sum a column, release a mean in two ways, and choose a median and quartiles among candidates. By the end you will be able to say, for each of these statistics, what the library asked you for and what its error bar covers. [The practitioner edition's workflow on releasing statistics](../../book-practitioner/part03-dp-foundations/12-release-statistics.md) does the same job on a retail dataset in which a customer owns a varying number of rows. Here the privacy unit is the simplest one the survey allows.

The script [opendp_essential_statistics.py](../../code/part02-tabular-data/opendp_essential_statistics.py) runs every query on OpenDP 0.16.0. It reads the sample lazily through `practitioner_datasets.france_lfs()`, as the documentation reads the same file through `dp.examples.get_france_lfs_path()`. The noise scales, the error bars, the refusals and the true values it prints are the same on every run. The released values are not, because OpenDP's noise cannot be seeded, so this page shows them in one block marked as one run.

```bash
uv run python code/part02-tabular-data/opendp_essential_statistics.py
```

## The context for this chapter {#sec-stats-context}

[The previous chapter](../part01-getting-started/02-the-data-and-the-context.md) chose the privacy unit for this survey. A person answers at most once per quarter over nine years, so one person owns at most 36 rows, and `dp.unit_of(contributions=36)` says so. Two neighbouring datasets are therefore two tables that differ by one person's rows, up to 36 of them, added or removed. The budget is a privacy loss $\varepsilon$ of 1 under pure differential privacy (DP), split evenly over five queries, so each query gets $\varepsilon = 0.2$.

The context adds one thing the first chapter's did not: a **margin**, which is a set of facts about the table that you declare public. Here the only fact is `max_length`, an upper bound on the number of rows. The survey reaches about 50,000 households a quarter, with fewer than three people in each, so 150,000 responses a quarter is a safe public bound. Over 36 quarters that gives 5,400,000 rows. The sum, the mean and the quantiles below refuse to run without this bound, for a reason the section on the sum explains.

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

```text
privacy unit: (SymmetricDistance(), 36)
privacy loss: (MaxDivergence, 1.0)
budget per query: [0.2, 0.2, 0.2, 0.2, 0.2]
margin: max_length=5,400,000
```

The chapter's five releases come from this one context: a count, a sum, a sum with a count, a median and three quartiles. A query takes the next share of the budget when it is built. So the comparisons on this page, which read scales without releasing anything, run on fresh contexts with the same settings. The documentation's tutorial does the same, though a real analysis builds one context and plans its shares before the first query.

## Counting responses {#sec-stats-count}

The simplest statistic is the number of rows. The query selects `dp.len`, OpenDP's private version of Polars' `len`, and the context fills in the noise.

```{.python excerpt="stats-count"}
count_query = context.query().select(dp.len(signed=True))
count_summary = count_query.summarize(alpha=0.05)
dp_count = count_query.release().collect().item()
```

The method to call before `release` is `summarize`. It describes the measurement the context built without running it, one row per statistic.

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

The column is `len`, the aggregate is the length of the frame, and the distribution is the integer Laplace distribution, because a count is an integer. The scale is 180, and the accuracy is the error bar at the level you passed as `alpha`. With probability 0.95, the released count is within 539.73 of the true one. The scale comes from the privacy unit. Removing a person removes up to 36 rows, so the **sensitivity** of the count, the most one person can change it, is 36. The Laplace scale is the sensitivity over the query's $\varepsilon$, and [the full edition's chapter on the Laplace and Gaussian mechanisms](../../book-full/part03-dp-foundations/03-laplace-gaussian.md) derives why that scale gives $\varepsilon$-DP.

The error bar depends on `alpha`, not on the data. Asking for more confidence widens it, so at `alpha=0.01` the same scale gives an error bar of 829.43. Both numbers cost nothing to read.

```text
count: scale=180.0 95% accuracy=539.73 99% accuracy=829.43
sensitivity 36 over epsilon 0.2: scale 180.0
```

The argument `signed=True` keeps the released count a signed integer, so a small count can come out negative when the noise outweighs it. Without the argument, the library clamps a negative count to zero and warns that this default will change.

```text
without signed=True: FutureWarning: The default value is currently signed=False, and counts are clamped at zero, consistent with past releases, but the default will be changing to signed=True.
```

A column has its own counts, and the documentation's aggregation page lists them. On a column, `.dp.len` counts the rows with nulls included, which equals the frame length unless an earlier expression changed the number of rows. The expression `.dp.count` counts the rows that are not null and `.dp.null_count` the rows that are, so the two add up to the length. Finally `.dp.n_unique` counts distinct values, with null as one of them. Each is as sensitive as the frame length, because one person's rows can change any of them by at most 36. So each gets the same scale of 180 when it is the only statistic in a query.

Which count you want depends on the question. The frame length counts responses, the non-null count counts those that answered the hours question, and the null count measures non-response. The distinct count tells you how many different answers appear, which is how you would notice that a column is coded rather than free. The excerpt asks for all five in one query.

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

Five statistics in one query share that query's $\varepsilon = 0.2$ equally, so each gets a fifth of it and five times the scale. This is the rule for every query on this page. A query's share is divided evenly among its statistics unless you set their scales yourself. Putting several counts in one query is therefore cheap in budget but not in noise. [Grouping](04-grouping.md) shows a better way to get the null and the non-null count together, by grouping on whether the value is null, which costs one count's noise for both.

## Where bounds come from {#sec-stats-bounds}

Every statistic after the count needs to know the range of its values. A sum and a mean need a lower and an upper bound, and a median or a quantile needs a list of candidates. The bounds set the sensitivity, because one person's rows can move a sum by at most their number times the width of the bounds. That is the clipping idea, and [the full edition's chapter on sensitivity](../../book-full/part03-dp-foundations/07-sensitivity.md) derives it. Bounds that are too wide buy more noise than the data need. Bounds that are too narrow clip real values and bias the result, and the error bar says nothing about that bias.

Three sources of bounds are legitimate. The first is domain knowledge, such as a working week of at most about 80 hours. The second is a comparable public dataset, from which you read the range. The third is the budget itself. You spend a share on a quantile query with widely spaced candidates to estimate the tails, and the rest on the statistic you wanted.

The one source that is never legitimate is the private data. A bound read off the data, such as its maximum, depends on one person's value and leaks it, and nothing in the guarantee accounts for that. This page takes its bounds from domain knowledge, with hours between 0 and 80. The candidates for the median are then the whole hours from 20 to 59.

## The sum of hours {#sec-stats-sum}

The column `HWUSUAL` holds the hours a person usually works in a week. The column codes "not applicable" as 99, so the query first filters those rows out, and OpenDP accounts for the filter as part of the query. Then `.dp.sum` clips every value into the bounds, sums, and adds noise.

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

One person owns at most 36 rows of at most 80 hours each. Adding or removing those rows changes the sum by at most 2,880, which is the sensitivity, and the scale is that sensitivity over the query's $\varepsilon$. The error bar of 43,139 hours looks large on its own, but it is small against a total near 56 million. The noise does not grow with the number of rows, so the relative error shrinks as the table grows.

The sum needs the margin, and it says so when the margin is missing.

```text
without a margin: must specify 'max_length' in a margin with by=[]
```

Why does a sum need a bound on the number of rows when the count did not? Because a sum is computed in finite arithmetic. Integers can overflow, and every addition of floats rounds, so the result of summing a column depends slightly on how many values were summed. The library accounts for that in the sensitivity, and it can only do so if it knows the largest number of rows a table can have. The `by=[]` in the message names the margin of the whole table, since no grouping is in play; [Grouping](04-grouping.md) adds margins per group. The refusal does not depend on the data, as a snippet on three rows shows.

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

The integer cast is the second adjustment. The column is stored as floats, and a float sum pays for the rounding of every addition, up to `max_length` of them. At this chapter's bound the price is small, a scale of 14,405.18 instead of 14,400. A careless bound of a billion rows raises the float scale to 192,035.68, while the integer scale does not move. Casting to an integer removes the rounding and with it the dependence on `max_length`, so cast whenever the values are whole numbers.

```text
float sum, max_length 5,400,000: HWUSUAL: Sum, Float Laplace, scale=14,405.18 95% accuracy=43,154.1
float sum, max_length 1,000,000,000: HWUSUAL: Sum, Float Laplace, scale=192,035.68 95% accuracy=575,287.5
int sum, max_length 1,000,000,000: HWUSUAL: Sum, Integer Laplace, scale=14,400.00 95% accuracy=43,139.0
```

The third adjustment is `fill_null`, and it exists because a sum has no defined value when a row is null. The library therefore insists that the column it sums has no nulls, and if you write the clip and the sum by hand, it refuses a nullable column. The shortcut `.dp.sum` instead fills a null with the midpoint of the bounds, 40 here, before clipping. Writing `fill_null(35)` makes the choice yours, and 35 is the legal working week in France, a public fact. On this data the choice turns out not to matter, because the filter has already dropped every null. A null is not equal to 99, and Polars drops a row whose test is null, so the check at the end of the chapter finds none left. To confirm the midpoint rule on a release, the script also sums the whole column, where the nulls do remain.

```text
sum without fill_null: scale=14,400.0 (accepted; a null becomes the midpoint 40)
clip, sum and laplace by hand, without fill_null: input data (HWUSUAL) might contain nulls. Preprocess your data with `.fill_null`.
```

Finally, consider what the error bar covers. The released sum is within 43,139 of the *clipped* sum with probability 0.95. A value above 80 would be pulled down to 80 before summing, and no error bar reports that. Here no applicable value exceeds the bound, as the check at the end shows, so the clipped sum is the true sum.

## The mean {#sec-stats-mean}

A mean is a sum divided by a count, and under the add/remove relation both are private quantities. The documentation's recommendation is to release the two in one query and divide afterwards. Dividing two released numbers is post-processing, which costs nothing, as [the full edition's chapter on properties](../../book-full/part03-dp-foundations/02-properties.md) proves.

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

The query's share is split evenly between its two statistics, so each has twice the scale it had alone: 28,800 for the sum and 360 for the count. The pair costs the same share as either did on its own, which is why the joint query is the recommended one. The sum here has no `fill_null`, so a null would become the midpoint, as the comment says. One released count can also serve several means, by dividing the sums of other columns by it. Calling `.dp.mean` on this context changes nothing: it releases the same noisy sum and noisy length, as the script's comparison shows.

```text
dp.mean on the add/remove context: HWUSUAL: Sum, Integer Laplace, scale=28,800.00 95% accuracy=86,277.6
dp.mean on the add/remove context: HWUSUAL: Length, Integer Laplace, scale=360.00 95% accuracy=1,079.0
```

The alternative is to declare the number of rows public. The documentation calls this bounded DP. In the full edition's terms the neighbouring relation becomes replace-one, where two neighbours have the same number of rows and differ in one person's values. In a margin, `invariant="lengths"` declares the number of rows an invariant, and `.dp.mean` then adds noise to the sum alone and divides by the exact length.

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

The sum's scale is 7,200, half of what the sum had with a whole share under add/remove. When the length is fixed, changing one row counts as one removal and one addition. A distance of 36 under the privacy unit's metric therefore means at most 18 replaced rows, so the sensitivity halves. The length row shows a scale of zero and no accuracy, because the length is released exactly. That is the weaker guarantee you accept. The release still hides what a person answered, but not how many rows the table has, and so not whether a person is in it.

There is a second cost, visible in the excerpt. The filter on the code 99 runs outside the context, on the LazyFrame itself. Inside the context, a filter makes the number of remaining rows unknown again, so the invariant is lost. Then `.dp.mean` falls back to a noisy sum and a noisy length, which the script confirms. Filtering outside the context is only sound if the number of applicable responses is truly public information, as the documentation says. If it is not, use the joint query.

```text
dp.mean after a filter inside the bounded context: HWUSUAL: Sum, Integer Laplace, scale=28,800.00 95% accuracy=86,277.6
dp.mean after a filter inside the bounded context: HWUSUAL: Length, Integer Laplace, scale=360.00 95% accuracy=1,079.0
```

## The median and the quartiles {#sec-stats-quantiles}

A median cannot be released by adding noise to it. When few values lie near the middle, one person's rows can move the median by an arbitrary amount, so no finite scale covers every dataset. OpenDP instead chooses the median among **candidates** you supply. It scores each candidate by how unbalanced the counts below and above it are, adds exponentially distributed noise to every score, and releases the candidate with the smallest noisy score. That is the exponential mechanism, which [the full edition's chapter on selection](../../book-full/part03-dp-foundations/05-selection.md) derives. The candidates here are the whole hours from 20 to 59.

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

The aggregate reads 0.5-quantile because `.dp.median` calls `.dp.quantile` with `alpha=0.5`. The distribution is named for the mechanism, exponential noise and a minimum. Its scale is twice the sensitivity of the score over the query's $\varepsilon$, with the factor of two that the exponential mechanism's analysis carries. The accuracy is empty, and it has to be. The noise went into the scores, not into the answer, and the answer is whichever candidate won, so there is no error bar in hours. A candidate whose score is far above the best one is very unlikely to win, and the check at the end of the chapter puts numbers on that. The quantile also needs the margin, as the refusal shows, because the scores are computed in integers whose size depends on the number of rows.

A quantile other than the median takes `alpha`, the share of values that lie below it. The excerpt asks for the three quartiles in one query.

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

Three quantiles in one query split its share three ways, so the median's scale triples from 360 to 1,080. The quartiles' scales are three times larger again, which looks as if the median got more of the budget, and the documentation says it did not. The reason is internal to the score. A quantile's score is counted in integers scaled by the denominator of `alpha`, which is four for a quartile and two for the median. So a quartile's score moves three times as far when one person's rows change, and its noise scales with it. The last line shows the even split directly. The first quartile on its own gets a scale of 1,080, a third of what it has inside the trio, so each of the three had the same $\varepsilon$.

## A check against the truth {#sec-stats-check}

The sample is public, so the script computes the true values at development time and reads one run's releases against them. On private data you cannot do this, which is why you do it on public or historical data while you develop, and release the real data once.

```text
true responses: 3,811,932
true responses with HWUSUAL not 99: 1,496,192 (nulls among them: 0; in the whole column: 20,819)
true sum of hours after the filter, clipped to [0, 80]: 56,315,594 (values above 80: 0)
true sum of the whole column, clipped: 240,742,034 with nulls as 40, 239,909,274 with nulls dropped
true mean: 37.64  (bounded mean 95% accuracy: 0.0144)
true quartiles: 35, 37, 40
score gap from the true quartile to the nearest other candidate: 0.25: 219,506  0.5: 74,383  0.75: 116,811
```

Two of these lines settle questions left open above. The filter leaves no nulls among the applicable responses, so the imputed value never entered the sum, and no applicable value exceeds 80, so clipping changed nothing. The bounded mean's error bar is the sum's error bar divided by the exact length, which gives 0.0144 hours. The last line is the quantiles' substitute for an error bar. The nearest competing candidate scores at least 74,383 above the true quartile, against a noise scale of at most 3,240. A wrong candidate is therefore far too unlikely to see in any number of runs. The test therefore requires the exact quartiles.

```text
released (one run):
  responses 3,811,961  (true 3,811,932, within 539.7 with probability 0.95)
  sum of hours 56,308,788  (true 56,315,594, within 43,139.0)
  joint: sum 56,335,491  count 1,495,681  mean 37.67  (true 37.64)
  bounded mean 37.6411  (true 37.6393, within 0.0144)
  median 37  quartiles 35, 37, 40  (true 37; 35, 37, 40)
  sum of the whole column 240,726,542  (true with nulls as 40: 240,742,034)
```

The count and the sum landed inside their error bars, as they should about 19 times in 20. The joint mean is off in the second decimal, because it divides two noisy numbers, each with twice the scale of a lone release. The bounded mean is far closer, inside a bar of about a hundredth of an hour, which is what declaring the length public buys. The median and the quartiles came out exactly as the score gaps predicted. The sum of the whole column landed inside its bar of the total that counts each null as 40 hours, and more than 800,000 hours from the total that drops them, which confirms what `.dp.sum` does with a null.

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
