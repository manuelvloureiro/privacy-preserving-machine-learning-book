# Identifier truncation and bounds

A person in the survey owns several rows, one per quarter answered. The chapters so far used the fixed unit of [the data and the context](../part01-getting-started/02-the-data-and-the-context.md#sec-ctx-context), up to 36 rows per person, so every noise scale grew 36-fold. But hardly anyone appears that often. This chapter uses the column that says which rows belong together, `PIDENT`, and lets each query decide how many of a person's rows it keeps. The guarantee still protects a person's whole record, but the noise fits what the query lets one person contribute.

The chapter follows the documentation's page on identifier truncation and its user guide on truncation and bounds. At the end, you know how to bound a person's rows and groups in a query, and which bounds the library enforces and which it takes on trust.

The script [opendp_identifier_truncation.py](../../code/part02-tabular-data/opendp_identifier_truncation.py) runs every query on the Labour Force Survey (LFS) sample with OpenDP 0.16.0. The scales, error bars, thresholds, error messages and facts about the data that it prints are the same on every run. The released values are not, because OpenDP's noise cannot be seeded, so the page shows them in one block marked as one run.

```bash
uv run python code/part02-tabular-data/opendp_identifier_truncation.py
```

## A privacy unit with an identifier {#sec-trunc-unit}

Until now the privacy unit was a number of rows. With an identifier, it is a value of a column. Two datasets are neighbours, written $D \sim D'$, when one is the other with every row of one `PIDENT` value added or removed. That is **user-level privacy**, which hides a person's whole record and not only one row of it. [The full edition's chapter on user-level privacy](../../book-full/part05-dp-ml/07-user-level.md) defines and analyses it. OpenDP writes the unit as a metric on identifiers and a distance of one.

```python
import opendp.prelude as dp

dp.enable_features("contrib")
print(dp.unit_of(contributions=1, identifier="PIDENT"))  # (SymmetricIdDistance { identifier: col("PIDENT") }, 1)
```

The argument `contributions=1` now counts identifiers, not rows: a person appears under at most one value of `PIDENT`, a public fact about the survey. The unit does not say how many rows a person owns. The old 36 had to hold for the whole dataset before any query ran. Now each query states its own bound as a truncation that the library applies, so the bound holds because the query makes it hold.

The script first looks at what an identifier owns in this file. This look is harmless only because the data are public. On real data, reason from the design of the survey, because a bound read off the data leaks it.

```text
identifiers: 1,682,306  rows: 3,811,932
rows per identifier: p50=2 p90=4 p99=7 max=15
identifiers with more than 10 rows: 195
quarters per identifier: max 4  rows per identifier and quarter: max 8
```

So the worst case of 36 rows is far from this file. An identifier spans at most four quarters, but it can own up to eight rows in one quarter.

The context is the one that the documentation builds, with the public bound on the number of rows of [the data and the context](../part01-getting-started/02-the-data-and-the-context.md#sec-ctx-margin). Its budget, split evenly over four queries, is a privacy budget $\varepsilon$ of 1 and a slack $\delta$ of $10^{-8}$. The slack is the small probability with which the guarantee can fail. The table per quarter needs the slack, because the library selects its quarters from the data.

```{.python excerpt="trunc-context"}
def make_context(frame: pl.LazyFrame, privacy_unit: tuple[dp.Metric, Any]) -> dp.Context:
    """The documentation's context: approximate differential privacy, split evenly over four queries."""
    return dp.Context.compositor(
        data=frame,
        privacy_unit=privacy_unit,  # dp.unit_of(contributions=1, identifier="PIDENT") below
        privacy_loss=dp.loss_of(epsilon=EPSILON, delta=DELTA),
        split_evenly_over=QUERIES,
        margins=[dp.polars.Margin(max_length=MAX_LENGTH)],
    )
```

```text
budget per query: epsilon=0.25 delta=2.5e-09
```

The library refuses a query on this context that does not truncate, because it cannot bound how far one identifier's rows move a statistic. In version 0.16.0, the error message is not helpful.

```{.python excerpt="trunc-untruncated"}
try:
    context.query().filter(pl.col("HWUSUAL") != 99).select(hours().dp.mean(HOURS)).summarize()
    untruncated = "accepted"
except dp.OpenDPException as err:
    untruncated = reason(err)
```

```text
mean without truncation: A step in your query is not recognized at this time: "DF ["PIDENT", "YEAR", "QUARTER", "SEX", ...]; PROJECT */8 COLUMNS". ""If you would like to see this supported, please file an issue.
```

The message blames the scan of the data, not the missing truncation. If you see it on a context with an identifier, first make sure that every statistic comes after a truncation.

## Truncating each person's rows {#sec-trunc-rows}

The documentation's first query is the mean of `HWUSUAL`, the hours usually worked, without the code 99 for "not applicable". The mean needs public bounds on the hours, 0 to 80, and a bound on the rows per person. `truncate_per_group(10)` gives that bound: it keeps at most ten random rows per identifier before the aggregation.

```{.python excerpt="trunc-per-group"}
mean_query = (
    context.query()
    .filter(pl.col("HWUSUAL") != 99)  # 99 means "not applicable"
    .truncate_per_group(ROWS_PER_IDENTIFIER)  # at most 10 rows per identifier, chosen at random
    .select(hours().dp.mean(HOURS))
)
mean_summary = mean_query.summarize(alpha=0.05)
dp_mean = mean_query.release().collect().item()
```

```text
mean, 10 rows per identifier: HWUSUAL Sum, Integer Laplace, scale=6,400.0 95% accuracy=19,173.2
mean, 10 rows per identifier: HWUSUAL Length, Integer Laplace, scale=80.0 95% accuracy=240.2
```

A mean is released as a sum and a count, as in [Essential statistics](03-essential-statistics.md). The two split the query's share, so each gets $\varepsilon/8$. One person now moves the count by at most ten rows, so the count's sensitivity is 10, and its scale is 10 over $\varepsilon/8$. Each row adds at most 80 hours to the sum, so the sum's sensitivity and scale are 80 times larger. Truncation and clipping are the two ways to make a sensitivity finite, and [the full edition's chapter on sensitivity](../../book-full/part03-dp-foundations/07-sensitivity.md) derives what each does to the answer.

Under the unit of 36 contributions, the same mean gets scales of 23,040 and 288. The price of the truncation is bias, because the mean leaves out the rows after the tenth. On this file the price is nothing: the truncation drops 20 rows of about 1.5 million.

```text
rows with HWUSUAL not 99: 1,496,192 by 809,710 identifiers
identifiers with more than 10 such rows: 18  rows dropped: 20
unit of 36 contributions, no identifier: scales 23,040.0, 288.0
```

The `keep` argument says which rows remain. The default, `"sample"`, keeps a random ten: its plan numbers each identifier's rows in a shuffled order and keeps the numbers below ten. With `"first"` or `"last"` the numbering follows the file's order, which is cheaper but biased when the order means something. A `SortBy` keeps the rows that sort first by an expression, such as `ILOSTAT`, the employment status. No choice changes the scales, because the bound of ten rows holds whichever rows remain.

```text
truncate_per_group(10) plan: FILTER [(0.int_range([len().cast(Int64)]).shuffle().over([col("PIDENT")])) < (10)]
keep=first: scales 6,400.0, 80.0
keep=last: scales 6,400.0, 80.0
keep=SortBy(ILOSTAT): scales 6,400.0, 80.0
ILOSTAT values among them: [1]
```

The last line shows that every row after the filter has `ILOSTAT` equal to 1, so here the sort changes nothing.

## Truncating the groups a person touches {#sec-trunc-groups}

The second query is a table with one row per quarter, grouped by `QUARTER` and `YEAR`, with the number of responses and the sum of their hours. Now the noise depends on two bounds: the rows that a person adds to one group, and the number of groups that a person touches. The counts change by at most the first bound in each of at most the second number of groups. So the total sensitivity is the product of the two. The full edition derives this in [its section on contribution bounding](../../book-full/part05-dp-ml/07-user-level.md#sec-ulvl-bounding).

So the query truncates twice. `truncate_per_group(1, by=quarterly)` keeps one row per identifier and quarter, and `truncate_num_groups(10, by=quarterly)` keeps at most ten quarters per identifier. Without the second truncation the library refuses the query, and this time the message says what is missing. (The script cuts the message's suggestion, whose column order changes from run to run.)

```text
per quarter without truncate_num_groups: num_groups (None), total contributions (None), and per_group (Some(1)) are not sufficiently well-defined. This is likely due to a missing truncation earlier in the data pipeline.
```

The query below is the documentation's, except for `keep="first"` in the group truncation, for a reason given below.

```{.python excerpt="trunc-num-groups"}
quarter_query = (
    context.query()
    .filter(pl.col("HWUSUAL") != 99)
    .truncate_per_group(1, by=QUARTERLY)  # one row per identifier and quarter
    .truncate_num_groups(GROUPS_PER_IDENTIFIER, by=QUARTERLY, keep="first")  # at most 10 quarters each
    .group_by(QUARTERLY)
    .agg(dp.len(signed=True), hours().dp.sum(HOURS))
)
quarter_summary = quarter_query.summarize(alpha=0.05)
quarters_hat = quarter_query.release().collect()
```

```text
per quarter, 10 quarters per identifier: len Frame Length, Integer Laplace, scale=80.0 95% accuracy=240.2 threshold=1,714
per quarter, 10 quarters per identifier: HWUSUAL Sum, Integer Laplace, scale=6,400.0 95% accuracy=19,173.2
```

The scales are the ones of the mean, because one row in each of ten quarters is again ten rows per person. The query gives no public keys, so the library releases a quarter only if its noisy count is above the threshold, as in [Grouping](04-grouping.md). Every quarter has at least 27,228 responses, so all 36 appear.

Version 0.16.0 has two problems with `truncate_num_groups`. With the default, `keep="sample"`, the query passes `summarize` and `release` spends its share, but `collect` fails inside Polars.

```{.python excerpt="trunc-sample"}
sample_query = grouped_query(context, keep="sample")  # the same table, with the default keep
sample_summary = sample_query.summarize(alpha=0.05)
released = sample_query.release()  # spends the query's share of the budget
try:
    released.collect()
    sample_outcome = "collected"
except dp.OpenDPException as err:
    sample_outcome = reason(err)
```

```text
keep=sample: summarize scales 80.0, 6,400.0; collect: multiple fields with name 'QUARTER' found
shares spent after the failed collect: 4 of 4
```

The share is gone, because the budget is spent when the measurement is built. But Polars checks a plan without the data, so `explain()` before `release` shows the same error at no cost, as on this toy frame.

```python
import opendp.prelude as dp
import polars as pl

dp.enable_features("contrib")
toy = pl.LazyFrame({"PIDENT": [7, 7, 7], "YEAR": [2005] * 3, "QUARTER": [1, 2, 3], "HWUSUAL": [35.0, 36.0, 37.0]})
context = dp.Context.compositor(
    data=toy,
    privacy_unit=dp.unit_of(contributions=1, identifier="PIDENT"),
    privacy_loss=dp.loss_of(epsilon=1.0, delta=1e-8),
    split_evenly_over=1,
    margins=[dp.polars.Margin(max_length=100)],
)
quarterly = [pl.col("QUARTER"), pl.col("YEAR")]
query = context.query().truncate_per_group(1, by=quarterly).truncate_num_groups(10, by=quarterly)
query = query.group_by(quarterly).agg(dp.len(signed=True))
print(f"{query.summarize()['scale'].item():.1f}")  # 10.0
try:
    query.explain()
except pl.exceptions.DuplicateError as err:
    print(str(err).splitlines()[0])  # multiple fields with name 'QUARTER' found
```

The second problem is with `keep="first"` and `keep="last"`, which run. Their plan keeps the ranks of an identifier's quarters below the number that you ask for. But Polars counts ranks from one, so the truncation keeps one group fewer than asked, as the two numberings show.

```python
import polars as pl

toy = pl.DataFrame({"PIDENT": [7, 7, 7], "YEAR": [2005] * 3, "QUARTER": [1, 2, 3]})
quarterly = [pl.col("QUARTER"), pl.col("YEAR")]
numbered = toy.with_columns(
    index=pl.int_range(pl.len()).over("PIDENT"),  # what truncate_per_group compares with k
    rank=pl.struct(quarterly).rank("dense").over("PIDENT"),  # what truncate_num_groups compares with k
)
print(numbered["index"].to_list(), numbered["rank"].to_list())  # [0, 1, 2] [1, 2, 3]
```

No identifier spans more than four quarters, so the script measures the effect at a smaller `k`. A request for four keeps at most three quarters, so the 13,948 identifiers with four quarters lose one. A request for five keeps all four, at the scale and threshold of five.

```text
truncate_num_groups(10, keep=first) plan: FILTER [(col("QUARTER").as_struct([col("YEAR")]).rank().over([col("PIDENT")])) < (10)]
identifier-quarters after the filter: 1,241,896  identifiers with 4 quarters: 13,948
truncate_num_groups(4, keep=first): keeps at most 3 quarters, 1,227,948 identifier-quarters; scale 32.0 threshold 657
truncate_num_groups(5, keep=first): keeps at most 4 quarters, 1,241,896 identifier-quarters; scale 40.0 threshold 830
```

So in version 0.16.0, ask for one group more than you want to keep, and accept the noise of that larger number. The guarantee is safe either way, because the library calibrates to the number asked for. The estimate is not, because the dropped group is always the last in sort order. [The practitioner edition's release workflow](../../book-practitioner/part03-dp-foundations/12-release-statistics.md) avoids the call for these reasons.

Put the truncations of a query next to each other, with the group truncation last before the aggregation. With a filter between them, the library calls the truncation's own expression unrecognized.

```text
a filter between the two truncations: Expr is not recognized at this time: dyn int: 0.int_range([len()]).shuffle().over([col("PIDENT"), col("QUARTER"), col("YEAR")]). If you would like to see this supported, please file an issue.
```

## One row per person {#sec-trunc-person}

For a mean, there is a better method than ten rows per person. Group by the identifier first, aggregate each person's rows to one row with any Polars expression, and take the private mean of those rows. After the group-by, no identifier owns more than one row.

```{.python excerpt="trunc-per-person"}
person_query = (
    context.query()
    .filter(pl.col("HWUSUAL") != 99)
    .group_by("PIDENT")  # one row per identifier: the truncation
    .agg(pl.col("HWUSUAL").mean())  # any expression may summarize the person's rows
    .select(hours().dp.mean(HOURS))
)
person_summary = person_query.summarize(alpha=0.05)
dp_person_mean = person_query.release().collect().item()
```

```text
mean of one row per identifier: HWUSUAL Sum, Integer Laplace, scale=640.0 95% accuracy=1,917.8
mean of one row per identifier: HWUSUAL Length, Integer Laplace, scale=8.0 95% accuracy=24.5
```

The count's sensitivity is now 1 and the sum's is 80, so both scales are a tenth of the scales with ten rows. And no row is dropped.

But the question changes. The first query averaged over responses, so a person with four responses counted four times. This one averages each person's own average, so every person counts once, which gives 37.24 hours against 37.64. The cast to an integer then truncates each person's mean towards zero, which lowers the true value to 37.10. With an error bar of about 0.0035 hours, the choice of what to average matters far more than the noise.

```text
true mean of one row per identifier: 37.0970  (before the cast to int: 37.2359; all rows: 37.6393)
per-identifier mean error bar at the true size: 0.0035
```

## Bounds in the privacy unit {#sec-trunc-bounds}

The unit can also hold finer knowledge, as a list of `dp.polars.Bound` objects. Each names its grouping, a bound on identifiers per group and a bound on the groups that one person touches. The documentation's example is data merged from two sources that assigned identifiers separately. A person can then appear under two identifiers, which would double the noise, but never under two in the same quarter.

```{.python excerpt="trunc-bounds"}
two_sources = dp.unit_of(
    contributions=[
        dp.polars.Bound(per_group=2),  # a person may appear under two identifiers
        dp.polars.Bound(by=QUARTERLY, per_group=1),  # but under one identifier in any quarter
    ],
    identifier="PIDENT",
)
bounded = make_context(frame, two_sources)
bounded_query = grouped_query(bounded, keep="first", k=5)  # each identifier may affect up to 5 quarters
bounded_summary = bounded_query.summarize(alpha=0.05)
bounded_hat = bounded_query.release().collect()
```

```text
two identifiers, one per quarter, 5 quarters: len Frame Length, Integer Laplace, scale=80.0 95% accuracy=240.2 threshold=1,714
two identifiers, one per quarter, 5 quarters: HWUSUAL Sum, Integer Laplace, scale=6,400.0 95% accuracy=19,173.2
one identifier as a bound, 5 quarters: scales 40.0, 3,200.0
```

The query keeps one row per identifier and quarter, and five quarters per identifier. Two identifiers times five quarters is ten rows per person, so the scale is 80 again. The second bound tells the library that the two identifiers never meet in one quarter, so one person adds at most one row to any quarter. It does not lower the count's scale, because the ten rows can still fall in ten different quarters. With a single identifier as the bound, the same query gets half the scale.

A bound in the unit can also have `num_groups`, the number of groups that a person touches. Then the query needs no `truncate_num_groups`, and a promise of ten quarters gives the same scale as the truncation at ten. But the library enforces a truncation, while it takes a bound on trust. The script makes a promise that this file breaks, that no identifier spans more than three quarters, and the release goes through with no complaint.

```{.python excerpt="trunc-promise"}
promised = dp.unit_of(
    contributions=[
        dp.polars.Bound(per_group=1),
        dp.polars.Bound(by=QUARTERLY, per_group=1, num_groups=3),  # a promise: at most 3 quarters each
    ],
    identifier="PIDENT",
)
promise_query = (
    make_context(frame, promised)
    .query()
    .filter(pl.col("HWUSUAL") != 99)
    .truncate_per_group(1, by=QUARTERLY)  # no truncate_num_groups: the unit already bounds the quarters
    .group_by(QUARTERLY)
    .agg(dp.len(signed=True), hours().dp.sum(HOURS))
)
promise_summary = promise_query.summarize(alpha=0.05)
promise_hat = promise_query.release().collect()
```

```text
num_groups=10 promised in the unit: len Frame Length, Integer Laplace, scale=80.0 95% accuracy=240.2 threshold=1,714
num_groups=3 promised in the unit: len Frame Length, Integer Laplace, scale=24.0 95% accuracy=72.4 threshold=486
num_groups=3 promised in the unit: HWUSUAL Sum, Integer Laplace, scale=1,920.0 95% accuracy=5,752.3
identifiers that break the promise of 3 quarters: 13,948
```

The noise is calibrated for three quarters, but 13,948 identifiers have four. For those people the privacy loss is above the declared $\varepsilon$, and nothing in the output says so. So truncate in the query, where the library enforces the bound. Declare a bound in the unit only when no truncation can express it, as with the two identifiers above, which no filter can recognize as one person. The same advice applies to values: clip them in the query, and do not declare their range in the input domain. [Assessing utility and choosing the noise](07-utility-and-noise.md) returns to bounds per group with Gaussian noise, whose scale depends on them differently.

## The check against the truth {#sec-trunc-check}

The data are public, so the script computes what each query would return without noise, with its own seeded random ten rows for the first mean. The quarters' true values use the first row of each identifier in each quarter, while the release keeps a random row. So the check is exact for the counts and approximate for the hours.

```text
true mean, 10 rows per identifier: 37.6393  (1,496,172 rows)
true per quarter (first row per identifier and quarter): 36 quarters, rows from 27,228 to 41,868
true 2005Q1: rows 29,561  hours 1,110,610
true 2005Q2: rows 29,406  hours 1,106,170
true 2005Q3: rows 28,364  hours 1,070,826
true 2005Q4: rows 28,681  hours 1,080,009
```

Here are the releases of one run. Each mean's error bar comes from the bars of its sum and its count, and holds whenever both hold.

```text
released (one run):
  mean, 10 rows per identifier: 37.6366  (true 37.6393, within 0.0189 with probability at least 0.9)
  per quarter: 36 quarters released; rows within their 95% accuracy: 36 of 36; hours: 35 of 36
  2005Q1: rows 29,644  hours 1,105,518
  2005Q2: rows 29,460  hours 1,098,965
  2005Q3: rows 28,244  hours 1,063,846
  2005Q4: rows 28,645  hours 1,087,028
  mean of one row per identifier: 37.0970  (true 37.0970, within 0.0035 with probability at least 0.9)
  two bounds, 5 quarters: 36 quarters released; rows within four times their 95% accuracy: 36 of 36
  num_groups=3 promised: 36 quarters released; rows within four times their 95% accuracy: 36 of 36
```

The means are within their bars, and so are the quarters' counts and all but one of their sums. A bar of 95% predicts about that many misses for 36 values. The test checks the two means within three times their bars and the 36 quarters within four times theirs, so a correct run almost never fails it.

## Summary

| Call or concept | What it does | Rule of thumb |
| --- | --- | --- |
| `dp.unit_of(contributions=1, identifier="PIDENT")` | The privacy unit is every row of one identifier | `contributions` now counts identifiers; the rows per person are bounded by the query |
| A query with no truncation | Refused, with a message that blames the data scan | On a context with an identifier, every statistic sits behind a truncation |
| `truncate_per_group(k, keep=...)` | Keeps at most `k` rows per identifier, and per group with `by` | `"sample"` is unbiased; `"first"`, `"last"` and `SortBy` are cheaper or deliberate, and biased when order matters |
| `truncate_num_groups(k, by=...)` | Keeps at most `k` groups per identifier | In 0.16.0, `"sample"` fails at `collect` after the budget is spent, and `"first"` keeps `k - 1` groups; call `explain()` before `release()` |
| Truncations in a query | Must sit next to each other, the group truncation last | A filter between them makes the query unrecognized |
| `group_by("PIDENT").agg(...)` before a statistic | One row per person, so the sensitivity is that of one row | The best choice for a mean, if the mean over people is the one you want |
| `dp.polars.Bound(by, per_group, num_groups)` in `unit_of` | Finer bounds on identifiers per group and groups per person | A promise the library trusts; a person who breaks it loses more than $\varepsilon$ |
| Truncate or declare | The query enforces a truncation; the unit declares a bound | Truncate, and declare only what no truncation can express |

: The calls of identifier truncation and bounds, what each does, and the rule of thumb that goes with it. {#tbl-trunc-summary}

## References

- OpenDP documentation: "Identifier Truncation" in the tabular-data tutorial, and "Identifier Truncation and Bounds" in the Polars user guide.
- Dwork, McSherry, Nissim and Smith, "Calibrating Noise to Sensitivity in Private Data Analysis" (2006): sensitivity and the Laplace mechanism.
- Amin, Kulesza, Muñoz Medina and Vassilvitskii, "Bounding User Contributions: A Bias-Variance Trade-off in Differential Privacy" (2019): why truncating a person's rows trades bias for variance.
- Wilson, Zhang, Lam, Desfontaines, Simmons-Marengo and Gipson, "Differentially Private SQL with Bounded User Contribution" (2020): bounding rows per group and groups per user in grouped queries.
- The full edition's chapters on [sensitivity](../../book-full/part03-dp-foundations/07-sensitivity.md) and [user-level privacy](../../book-full/part05-dp-ml/07-user-level.md) derive the results this chapter states.
