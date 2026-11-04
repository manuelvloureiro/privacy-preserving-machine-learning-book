# Identifier truncation and bounds

A survey respondent may own several rows, one per quarter. Earlier chapters used the fixed unit from [The data and the context](../part01-getting-started/02-the-data-and-the-context.md#sec-ctx-context), allowing up to 36 rows per person and increasing every noise scale by that factor. This chapter uses `PIDENT` to identify a person's rows and bounds their contribution separately for each query. The guarantee still protects the whole record, while the noise reflects the query's contribution bound.

The chapter follows the documentation's identifier-truncation page and user guide on truncation and bounds. It shows how to bound each person's rows and groups under differential privacy (DP), and distinguishes bounds enforced by the library from bounds it trusts.

The script [opendp_identifier_truncation.py](../../code/part02-tabular-data/opendp_identifier_truncation.py) runs every query on the Labour Force Survey (LFS) sample with OpenDP 0.16.0. The scales, error bars, thresholds, error messages and facts about the data that it prints are the same on every run. The released values are not, because OpenDP's noise cannot be seeded, so the page shows them in one block marked as one run.

```bash
uv run python code/part02-tabular-data/opendp_identifier_truncation.py
```

## A privacy unit with an identifier {#sec-trunc-unit}

Previously, the privacy unit was a number of rows. With an identifier, it is a column value. Datasets $D$ and $D'$ are neighbours when one is obtained from the other by adding or removing every row with one `PIDENT` value. This is **user-level privacy**: it protects a person's whole record, not just one row. [The full edition's chapter on user-level privacy](../../book-full/part05-dp-ml/07-user-level.md) defines and analyses it. OpenDP represents this unit with a metric on identifiers and distance one.

```python
import opendp.prelude as dp

dp.enable_features("contrib")
print(dp.unit_of(contributions=1, identifier="PIDENT"))  # (SymmetricIdDistance { identifier: col("PIDENT") }, 1)
```

Here, `contributions=1` counts identifiers rather than rows. In this survey, each person has at most one `PIDENT` value. The unit does not bound a person's row count. The earlier bound of 36 had to hold across the entire dataset before querying; now each query specifies a bound that the library enforces through truncation.

The script first counts rows per identifier. This inspection is safe only because the data are public. For private data, derive the bound from the survey design; a bound calculated from the data leaks information.

```text
identifiers: 1,682,306  rows: 3,811,932
rows per identifier: p50=2 p90=4 p99=7 max=15
identifiers with more than 10 rows: 195
quarters per identifier: max 4  rows per identifier and quarter: max 8
```

The 36-row worst case is far above the observed maximum. An identifier spans at most four quarters but can own up to eight rows in one quarter.

The context follows the documentation and includes the public row-count bound from [The data and the context](../part01-getting-started/02-the-data-and-the-context.md#sec-ctx-margin). It splits an approximate-DP budget of $\varepsilon=1$ and $\delta=10^{-8}$ evenly across four queries. The quarterly table needs $\delta$ because the library selects its keys from the data.

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

OpenDP rejects queries on this context unless they truncate, because it cannot otherwise bound one identifier's effect on a statistic. In version 0.16.0, the error message does not identify the missing truncation.

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

The message blames the data scan rather than the missing truncation. On an identifier-based context, check that every statistic follows a truncation.

## Truncating each person's rows {#sec-trunc-rows}

The documentation first computes the mean of `HWUSUAL` (usual hours), excluding code 99 ("not applicable"). The mean needs public bounds of 0 to 80 hours and a bound on rows per person. `truncate_per_group(10)` keeps at most ten random rows per identifier before aggregation.

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

A mean uses a noisy sum and count, as in [Essential statistics](03-essential-statistics.md). They split the query's budget share, so each gets $\varepsilon/8$. Truncation limits count sensitivity to 10, giving scale $10/(\varepsilon/8)$. Each row contributes at most 80 hours, so sum sensitivity and scale are 80 times larger. [The full edition's chapter on sensitivity](../../book-full/part03-dp-foundations/07-sensitivity.md) derives how truncation and clipping affect sensitivity and the answer.

With a 36-row unit, the same mean has scales 23,040 and 288. Truncation can bias the mean by dropping rows after the tenth. Here it drops only 20 of about 1.5 million rows.

```text
rows with HWUSUAL not 99: 1,496,192 by 809,710 identifiers
identifiers with more than 10 such rows: 18  rows dropped: 20
unit of 36 contributions, no identifier: scales 23,040.0, 288.0
```

The `keep` argument selects which rows remain. The default, `"sample"`, retains ten random rows by shuffling each identifier's rows and keeping the first ten positions. `"first"` and `"last"` follow file order; they are cheaper but can be biased when that order matters. `SortBy` retains rows ordered by an expression such as employment status, `ILOSTAT`. The scales are unchanged because each method retains at most ten rows.

```text
truncate_per_group(10) plan: FILTER [(0.int_range([len().cast(Int64)]).shuffle().over([col("PIDENT")])) < (10)]
keep=first: scales 6,400.0, 80.0
keep=last: scales 6,400.0, 80.0
keep=SortBy(ILOSTAT): scales 6,400.0, 80.0
ILOSTAT values among them: [1]
```

Every retained row has `ILOSTAT=1`, so sorting makes no difference in this dataset.

## Truncating the groups a person touches {#sec-trunc-groups}

The second query groups responses by `QUARTER` and `YEAR`, releasing each quarter's count and sum of hours. Noise depends on two bounds: rows one person can add to a group and the number of groups they can affect. Sensitivity is the product of these bounds. [The full edition's section on contribution bounding](../../book-full/part05-dp-ml/07-user-level.md#sec-ulvl-bounding) derives this result.

The query applies two truncations. `truncate_per_group(1, by=quarterly)` keeps one row per identifier and quarter; `truncate_num_groups(10, by=quarterly)` keeps at most ten quarters per identifier. Without the second truncation, OpenDP rejects the query and identifies the missing bound. The script omits the message's suggestion because its column order changes between runs.

```text
per quarter without truncate_num_groups: num_groups (None), total contributions (None), and per_group (Some(1)) are not sufficiently well-defined. This is likely due to a missing truncation earlier in the data pipeline.
```

The query follows the documentation but sets `keep="first"` for group truncation; the reason appears below.

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

These scales match the mean's: one row in each of ten quarters gives ten rows per person. Because the query has no public key list, OpenDP releases only quarters whose noisy counts exceed the threshold, as described in [Grouping](04-grouping.md). Each quarter has at least 27,228 responses, so all 36 appear.

OpenDP 0.16.0 has two problems with `truncate_num_groups`. With the default `keep="sample"`, `summarize` succeeds and `release` spends the budget share, but Polars fails during `collect`.

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

The share is spent when the measurement is built. Polars can check the plan without reading data, so `explain()` detects the same error before `release`, at no cost. The toy frame demonstrates this check.

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

`keep="first"` and `keep="last"` run, but their plans retain ranks below the requested group count. Polars ranks from one, so the truncation keeps one fewer group than requested, as the two numberings show.

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

No identifier spans more than four quarters, so the script tests smaller values of `k`. Requesting four keeps at most three quarters and removes one quarter for each of the 13,948 identifiers with four. Requesting five keeps all four, with the scale and threshold for five groups.

```text
truncate_num_groups(10, keep=first) plan: FILTER [(col("QUARTER").as_struct([col("YEAR")]).rank().over([col("PIDENT")])) < (10)]
identifier-quarters after the filter: 1,241,896  identifiers with 4 quarters: 13,948
truncate_num_groups(4, keep=first): keeps at most 3 quarters, 1,227,948 identifier-quarters; scale 32.0 threshold 657
truncate_num_groups(5, keep=first): keeps at most 4 quarters, 1,241,896 identifier-quarters; scale 40.0 threshold 830
```

In version 0.16.0, request one more group than you intend to retain and accept the corresponding larger noise. The guarantee remains valid because OpenDP calibrates to the requested count. The estimate can be biased because the dropped group is always last in sort order. The [practitioner edition's release workflow](../../book-practitioner/part03-dp-foundations/12-release-statistics.md) avoids this call.

Place truncations next to each other, with group truncation immediately before aggregation. A filter between them makes the truncation expression unrecognized.

```text
a filter between the two truncations: Expr is not recognized at this time: dyn int: 0.int_range([len()]).shuffle().over([col("PIDENT"), col("QUARTER"), col("YEAR")]). If you would like to see this supported, please file an issue.
```

## One row per person {#sec-trunc-person}

For a mean, grouping by identifier can be preferable to retaining ten rows per person. Aggregate each person's rows to one row with a Polars expression, then take the private mean. After grouping, each identifier contributes one row.

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

Count sensitivity is now 1 and sum sensitivity is 80, so both scales are one-tenth of those with ten rows. No rows are dropped.

This changes the estimand. The first query averages responses, so a person with four responses contributes four times. This query averages person-level means, giving each person equal weight: 37.24 hours rather than 37.64. Casting to integers truncates each person's mean toward zero and lowers the true value to 37.10. With an error bar of about 0.0035 hours, this choice matters far more than the noise.

```text
true mean of one row per identifier: 37.0970  (before the cast to int: 37.2359; all rows: 37.6393)
per-identifier mean error bar at the true size: 0.0035
```

## Bounds in the privacy unit {#sec-trunc-bounds}

The unit can encode more detailed knowledge in a list of `dp.polars.Bound` objects. Each object specifies a grouping, the maximum identifiers per group and the maximum groups one person can affect. The documentation considers data merged from two sources that assigned identifiers separately. A person may then have two identifiers, increasing noise, but never two within the same quarter.

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

The query keeps one row per identifier and quarter and up to five quarters per identifier. Two identifiers across five quarters give ten rows per person, so the scale is again 80. The second bound says the identifiers never occur in the same quarter; one person therefore adds at most one row per quarter. This does not lower count sensitivity because the ten rows may occupy ten different quarters. A single-identifier bound halves the scale.

`num_groups` can also bound the number of groups one person affects. The query then needs no `truncate_num_groups`, and a promise of ten quarters gives the same scale as truncating at ten. Unlike truncation, however, OpenDP does not enforce this bound. The script declares that no identifier spans more than three quarters, although the data contain 13,948 identifiers spanning four; the release proceeds without warning.

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

The noise assumes three quarters, but 13,948 identifiers span four. Their privacy loss exceeds the declared $\varepsilon$, with no warning in the output. Prefer truncation in the query, where OpenDP enforces the bound. Declare a unit bound only when truncation cannot express it, as when two identifiers represent one person and a filter cannot link them. Likewise, clip values in the query instead of declaring their range in the input domain. [Assessing utility and choosing the noise](07-utility-and-noise.md) revisits per-group bounds under Gaussian noise, whose scale depends on them differently.

## The check against the truth {#sec-trunc-check}

Because the data are public, the script computes each query's value without noise. The first mean uses its own seeded sample of ten rows. For quarterly values, the check uses the first row per identifier and quarter, while the release selects a random row. The count check is exact; the hours check is approximate.

```text
true mean, 10 rows per identifier: 37.6393  (1,496,172 rows)
true per quarter (first row per identifier and quarter): 36 quarters, rows from 27,228 to 41,868
true 2005Q1: rows 29,561  hours 1,110,610
true 2005Q2: rows 29,406  hours 1,106,170
true 2005Q3: rows 28,364  hours 1,070,826
true 2005Q4: rows 28,681  hours 1,080,009
```

The following values come from one run. Each mean's error bar combines the sum and count error bars and holds whenever both component bars hold.

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

Both means fall within their error bars. So do all quarterly counts and 35 of the 36 sums, consistent with 95% coverage across 36 values. The test allows three error bars for the means and four for the quarterly values, so a correct run rarely fails.

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
