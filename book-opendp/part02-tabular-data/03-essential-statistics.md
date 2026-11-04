# Essential statistics

This chapter answers four questions on the Labour Force Survey (LFS) sample: how many responses are there, how many hours do people work, and what are the mean and typical work week? OpenDP's Polars queries release each statistic with noise. The library sets the noise scale from a bound on one person's contribution and the query's budget share. Read the scale and error bar before release.

The chapter follows the documentation's tutorial on essential statistics and its aggregation guide. It counts rows, chooses bounds, sums a column, releases a mean in two ways and selects a median and quartiles from candidates. [The practitioner edition's workflow on releasing statistics](../../book-practitioner/part03-dp-foundations/12-release-statistics.md) applies these methods to retail data.

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

The five releases use one context. Building a query reserves its budget share, so scale comparisons use separate contexts with the same settings. In a real analysis, plan the shares before building queries in one context.

## Counting responses {#sec-stats-count}

The first release counts the survey rows. The query selects `dp.len`, OpenDP's private version of Polars' `len`; the context calibrates the noise.

```{.python excerpt="stats-count"}
count_query = context.query().select(dp.len(signed=True))
count_summary = count_query.summarize(alpha=0.05)
dp_count = count_query.release().collect().item()
```

Call `summarize` before `release` to inspect the measurement without running it. The summary has one row per statistic.

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

The aggregate is frame length, and its distribution is integer Laplace because the count is an integer. One person's addition or removal changes up to 36 rows, so count **sensitivity** is 36. The Laplace scale is sensitivity divided by the query's $\varepsilon$ share: $36/0.2$. [The full edition's chapter on the Laplace and Gaussian mechanisms](../../book-full/part03-dp-foundations/03-laplace-gaussian.md) derives this calibration. The accuracy is the error bar for the requested `alpha`; at 95% confidence, the released count is within 539.73 of the true count.

The error bar depends on `alpha`, not the data, so reading it has no privacy cost. Greater confidence requires a wider bar: at `alpha=0.01`, this scale gives 829.43.

```text
count: scale=180.0 95% accuracy=539.73 99% accuracy=829.43
sensitivity 36 over epsilon 0.2: scale 180.0
```

The bar covers noise only. It excludes sampling error between this sample and France's population, as well as Eurostat's recoding and suppression. It also counts responses, not people. `signed=True` allows a negative released count when noise exceeds a small true count. Without this argument, OpenDP clamps negative counts to zero, which biases small counts upward; the library warns that this default will change.

```text
without signed=True: FutureWarning: The default value is currently signed=False, and counts are clamped at zero, consistent with past releases, but the default will be changing to signed=True.
```

Columns support additional counts described in the documentation's aggregation guide. `.dp.len` counts rows including nulls, `.dp.count` counts non-null rows, and `.dp.null_count` counts nulls. These give the number of hours responses and the non-response count. `.dp.n_unique` counts distinct values, treating null as one value. Each count has sensitivity at most 36, so its scale matches frame length when queried alone. The excerpt requests all five counts together.

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

Five statistics in one query split its budget equally, giving each a fifth of the share and five times the scale. The queries on this page use equal splits unless their scales are set explicitly. [Grouping](04-grouping.md) releases null and non-null counts with one count's noise by grouping on null status.

## Where bounds come from {#sec-stats-bounds}

Statistics other than counts need value bounds or candidates. Sums and means need lower and upper bounds; medians need candidate values. The bounds determine sensitivity: one person's rows can change a sum by at most their count times the interval width. [The full edition's chapter on sensitivity](../../book-full/part03-dp-foundations/07-sensitivity.md) derives sensitivity under clipping. Wide bounds add unnecessary noise. Narrow bounds clip valid values and bias the result; the error bar does not include that bias.

Choose bounds from domain knowledge, a comparable public dataset or a budgeted coarse-quantile query. Never estimate them from private data: a bound such as the observed maximum depends on individual values and leaks information outside the guarantee. This chapter uses domain knowledge: 0 to 80 hours for a work week and whole-hour candidates from 20 to 59.

## The sum of hours {#sec-stats-sum}

`HWUSUAL` records usual weekly hours; 99 means "not applicable." The query filters out those rows, and OpenDP accounts for the filter. `.dp.sum` then clips values to the bounds, sums them and adds noise.

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

One person contributes at most 36 rows of at most 80 hours each, so sensitivity is their product. The 43,139-hour error bar is small relative to the total of about 56 million. Noise does not grow with the row count, so relative error decreases as the table grows.

The sum needs the margin, and it says so when the margin is missing.

```text
without a margin: must specify 'max_length' in a margin with by=[]
```

[The previous chapter](../part01-getting-started/02-the-data-and-the-context.md#sec-ctx-margin) explains that a sum needs a row-count bound to account for overflow and rounding. The message's `by=[]` identifies the whole-table margin; [Grouping](04-grouping.md) adds margins for groups. The refusal is data-independent, as the three-row snippet demonstrates.

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

The second adjustment casts the float column to integers. A float sum accounts for rounding across up to `max_length` additions. The cost is small at this chapter's bound, but a bound of one billion rows raises the float scale to 192,035.68. Integer casting avoids that rounding cost and its dependence on `max_length`; use it when the values are whole numbers.

```text
float sum, max_length 5,400,000: HWUSUAL: Sum, Float Laplace, scale=14,405.18 95% accuracy=43,154.1
float sum, max_length 1,000,000,000: HWUSUAL: Sum, Float Laplace, scale=192,035.68 95% accuracy=575,287.5
int sum, max_length 1,000,000,000: HWUSUAL: Sum, Integer Laplace, scale=14,400.00 95% accuracy=43,139.0
```

The third adjustment is `fill_null`. A sum cannot process a null row, so OpenDP rejects a nullable column when you write the clipping and sum manually. `.dp.sum` instead fills nulls with the bounds' midpoint, 40 here. `fill_null(35)` uses a legal French work week, a public fact. This choice does not affect this dataset: Polars treats a null comparison with `!= 99` as null, so the filter drops every null row. The script also sums the whole column to confirm the midpoint rule.

```text
sum without fill_null: scale=14,400.0 (accepted; a null becomes the midpoint 40)
clip, sum and laplace by hand, without fill_null: input data (HWUSUAL) might contain nulls. Preprocess your data with `.fill_null`.
```

The error bar applies to the *clipped* sum. Values above 80 become 80, and the error bar does not capture this change. The final check shows that no applicable value exceeds the bound in this dataset.

## The mean {#sec-stats-mean}

A mean is a sum divided by a count. Under the add/remove relation, both are private, so the documentation recommends releasing them in one query and dividing afterwards. This division is post-processing and incurs no additional cost, as proved in [the full edition's chapter on properties](../../book-full/part03-dp-foundations/02-properties.md).

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

The query splits its share between the two statistics, doubling each scale compared with a standalone query. Together they cost one share, and the released count can divide sums from other columns. In this context, `.dp.mean` releases the same noisy sum and length.

```text
dp.mean on the add/remove context: HWUSUAL: Sum, Integer Laplace, scale=28,800.00 95% accuracy=86,277.6
dp.mean on the add/remove context: HWUSUAL: Length, Integer Laplace, scale=360.00 95% accuracy=1,079.0
```

Alternatively, declare the row count public, which the documentation calls bounded DP. Neighbours then have equal row counts and differ in one person's values (the replace-one relation). Setting `invariant="lengths"` in a margin makes the length public, so `.dp.mean` adds noise only to the sum.

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

The sum's scale is half the add/remove scale for a full budget share. With fixed length, changing one row is one removal and one addition, so distance 36 allows at most 18 changed rows. The length has scale zero because it is released exactly. This weaker guarantee hides a person's answers but not their presence in the table.

The excerpt also filters out code 99 outside the context. Filtering inside the context makes the row count unknown, so the invariant no longer holds and `.dp.mean` uses a noisy sum and length. Filter outside the context only when the number of applicable responses is public. Otherwise, use the joint query.

```text
dp.mean after a filter inside the bounded context: HWUSUAL: Sum, Integer Laplace, scale=28,800.00 95% accuracy=86,277.6
dp.mean after a filter inside the bounded context: HWUSUAL: Length, Integer Laplace, scale=360.00 95% accuracy=1,079.0
```

## The median and the quartiles {#sec-stats-quantiles}

Direct noise on a median has no finite sensitivity bound: if few values lie near the middle, one person's rows can shift it arbitrarily. OpenDP instead chooses among supplied **candidates**. It scores each candidate by the imbalance between counts below and above it, adds exponential noise to the scores and releases the candidate with the smallest noisy score. [The full edition's chapter on selection](../../book-full/part03-dp-foundations/05-selection.md) derives this exponential mechanism. Here the candidates are whole hours from 20 to 59.

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

The aggregate is labelled 0.5-quantile because `.dp.median` calls `.dp.quantile` with `alpha=0.5`. Its scale is twice the score sensitivity divided by the query's budget share, as required by the exponential mechanism. Accuracy is empty because noise perturbs scores, not the answer, so there is no error bar in hours. A candidate with a much worse score is unlikely to win; the final check quantifies this gap. The quantile also needs a margin because its integer scores depend on the row count.

For other quantiles, `alpha` is the fraction of values below the target. The excerpt requests all three quartiles in one query.

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

Three quantiles split the query's share equally, tripling the median's scale. The quartile scales are three times larger again, without increasing the median's budget share. A quantile score is an integer multiplied by the denominator of `alpha`: four for a quartile and two for the median. One person's rows therefore move a quartile score three times farther. The final line shows that the first quartile's standalone scale is one-third its scale in the trio.

## A check against the truth {#sec-stats-check}

Because the sample is public, the script computes true values and compares them with one run's releases. For private data, perform this check during development on public or historical data, then release the private data once.

```text
true responses: 3,811,932
true responses with HWUSUAL not 99: 1,496,192 (nulls among them: 0; in the whole column: 20,819)
true sum of hours after the filter, clipped to [0, 80]: 56,315,594 (values above 80: 0)
true sum of the whole column, clipped: 240,742,034 with nulls as 40, 239,909,274 with nulls dropped
true mean: 37.64  (bounded mean 95% accuracy: 0.0144)
true quartiles: 35, 37, 40
score gap from the true quartile to the nearest other candidate: 0.25: 219,506  0.5: 74,383  0.75: 116,811
```

The second and third lines show that the filter leaves no nulls and no applicable value exceeds 80. The bounded mean's error bar is the sum's error bar divided by the exact length. For quantiles, the final line replaces an error bar: the nearest alternative candidate scores at least 74,383 above the true quartile, while the maximum noise scale is 3,240. A wrong candidate is therefore unlikely, and the test requires the exact quartiles.

```text
released (one run):
  responses 3,811,961  (true 3,811,932, within 539.7 with probability 0.95)
  sum of hours 56,308,788  (true 56,315,594, within 43,139.0)
  joint: sum 56,335,491  count 1,495,681  mean 37.67  (true 37.64)
  bounded mean 37.6411  (true 37.6393, within 0.0144)
  median 37  quartiles 35, 37, 40  (true 37; 35, 37, 40)
  sum of the whole column 240,726,542  (true with nulls as 40: 240,742,034)
```

In this run, the count and sum fall within their 95% error bars; each does so in about 19 runs out of 20. A count error of a few hundred is small relative to 3.8 million responses. The joint mean differs in the second decimal because it divides two noisy values. The bounded mean falls within an error bar of about a hundredth of an hour, the benefit of a public length. The median and quartiles match exactly. The whole-column sum is close to the total that imputes 40 hours for each null, not the total that drops nulls.

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
