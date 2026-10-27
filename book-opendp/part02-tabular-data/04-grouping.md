# Grouping

Most private releases are tables with one row per group, such as counts by age and employment status or the mean hours worked by sex. Grouping adds a danger that a single count does not have. The keys of the table are data too, because a group that one person fills alone shows that person just by its presence in the table. This chapter gives the three ways in which OpenDP releases a grouped table, and what each one costs.

First the library selects the keys, and only the groups whose noisy count passes a threshold appear. Then a public key list gives the keys, and then margins that declare the keys or the group sizes public. At the end, you can choose between the three and say what each gives up.

The script [opendp_grouping.py](../../code/part02-tabular-data/opendp_grouping.py) runs every query of this chapter on OpenDP 0.16.0 against the full survey sample. The scales, error bars, thresholds, error messages and true group counts that it prints are the same on every run. The released counts and means are not, because OpenDP's noise cannot be seeded, so the page shows them in one block marked as one run.

```bash
uv run python code/part02-tabular-data/opendp_grouping.py
```

## Why grouping keys are dangerous {#sec-group-danger}

The data are the France Labour Force Survey (LFS) sample of [The data and the context](../part01-getting-started/02-the-data-and-the-context.md#sec-ctx-data). The privacy unit is the one of [that chapter's context](../part01-getting-started/02-the-data-and-the-context.md#sec-ctx-context): up to 36 rows per person, one per quarter.

A grouping of the responses by `AGE` and by `ILOSTAT`, the labour status code, gives 16 groups in truth. The smallest, the responses with no recorded age, holds 255 rows. A noisy count with an error bar of a few hundred hides how many rows it has, but not that it exists. In a finer grouping, say by age, sex and town, one person can fill a group alone. Then the noise hides the count, but the row's presence says that someone with those keys answered the survey. So the keys of a grouped table need protection as much as the counts do.

OpenDP enforces this. On a context with a pure budget, `summarize` refuses a count by `AGE` and `ILOSTAT` before it computes anything.

```text
without delta: The key-set of {col("AGE"), col("ILOSTAT")} is private and cannot be released without a filter or join. Please pass a filtering threshold into make_private_lazyframe or conduct a join against a public key-set.
```

The message names the two ways out. A filter keeps only the groups whose noisy count passes a threshold, and it needs a $\delta$ in the budget. A join uses a public key list. The snippet shows the summary of each on a toy frame.

```python
import opendp.prelude as dp
import polars as pl

dp.enable_features("contrib")
data = pl.LazyFrame({"region": ["A", "A", "B"], "hours": [35, 40, 20]})
unit = dp.unit_of(contributions=1)
pure = dp.Context.compositor(data=data, privacy_unit=unit, privacy_loss=dp.loss_of(epsilon=1.0), split_evenly_over=1)
counts = pure.query().group_by("region").agg(dp.len(signed=True))
try:
    counts.summarize()
except dp.OpenDPException as err:
    print("cannot be released without a filter or join" in str(err))  # True
loss = dp.loss_of(epsilon=1.0, delta=1e-6)
approx = dp.Context.compositor(data=data, privacy_unit=unit, privacy_loss=loss, split_evenly_over=1)
summary = approx.query().group_by("region").agg(dp.len(signed=True)).summarize(alpha=0.05)
print(summary.columns)  # ['column', 'aggregate', 'distribution', 'scale', 'accuracy', 'threshold']
print(int(summary["threshold"].item()))  # 14
keyed = counts.with_keys(pl.LazyFrame({"region": ["A", "B", "C"]}))
print(keyed.summarize().columns)  # ['column', 'aggregate', 'distribution', 'scale']
```

## Stable keys {#sec-group-stable}

The first way is to let the library decide which groups to show. The documentation calls the keys that it keeps **stable keys**, because a group is stable when it stays in the data after the removal of any one person. The release keeps a group only when its noisy count is above a threshold. The library sets the threshold so that a group that one person fills alone appears with probability at most $\delta$. The chapter uses the documentation's budget, `dp.loss_of(epsilon=0.25, delta=1e-7)`. This is **partition selection**, and [the full edition's chapter on sensitivity](../../book-full/part03-dp-foundations/07-sensitivity.md) derives why a count with a threshold satisfies approximate differential privacy.

```{.python excerpt="group-stable"}
context = dp.Context.compositor(
    data=frame,
    privacy_unit=dp.unit_of(contributions=CONTRIBUTIONS),  # a person answers in at most 36 quarters
    privacy_loss=dp.loss_of(epsilon=EPSILON, delta=DELTA),  # the delta pays for the threshold
    split_evenly_over=1,
)
stable = context.query().group_by("AGE", "ILOSTAT").agg(dp.len(signed=True))
stable_summary = stable.summarize(alpha=0.05)
stable_hat = stable.release().collect()
```

```text
stable keys: privacy loss (Approximate(MaxDivergence), (0.25, 1e-07))
stable keys len: Frame Length, Integer Laplace, scale=144.00 95% accuracy=431.9 threshold=2773
```

Read the summary before you release. One person's 36 rows can all be in one group, so the count of any group has sensitivity 36, as the ungrouped count had. That sensitivity over $\varepsilon = 0.25$ gives the scale of 144. With probability 0.95, a released count is within 431.9 of the truth. The new column is the threshold, 2773: the release drops a group whose noisy count is not above it.

The library calibrates the threshold to the worst case for one person. Their 36 rows could all go into one otherwise empty group, with a true count of 36. Discrete Laplace noise of scale 144 lifts that group above 2773 with probability $2.77 \times 10^{-9}$. But the same person's rows could instead spread over 36 groups of one row each, and any of them might pass. So the library adds the chances over 36 groups, which gives $9.98 \times 10^{-8}$, just within $\delta = 10^{-7}$. One step lower, the total is above $\delta$, so 2773 is the smallest threshold that keeps the promise.

```text
one person's 36 rows clear 2773 with probability 2.77e-09; over 36 groups 9.98e-08 <= delta 1e-07
at threshold 2772: over 36 groups 1.01e-07 > delta
```

The threshold falls with a looser $\delta$ and with a smaller scale, as the script shows. A larger privacy unit raises both scale and threshold, because a person who owns more rows can make a larger group alone.

```text
threshold with delta=1e-05: scale=144.00 threshold=2110
threshold with epsilon=1: scale=36.00 threshold=720
```

On this public data, 14 of the 16 groups have a true count above 2773, so the release must have 14 rows. A dropped group does not show itself, and the number of rows is itself noisy information. If you see fewer groups than you expected, do not look at the data to find which are missing, because that look is outside the guarantee.

## Explicit keys {#sec-group-explicit}

If you know the keys in advance, you can give them to the library and keep a pure budget. **Explicit keys** are a key list that you supply, and the release has every key in it, with or without rows in the data. The list must be fixed without a look at the data, because a key list read off the data is a release in itself. Here it is every survey year crossed with every status code of the codebook, including codes that never occur in this sample.

```{.python excerpt="group-explicit"}
explicit = (
    context.query()
    .group_by("YEAR", "ILOSTAT")
    .agg(dp.len(signed=True))
    .with_keys(public_keys())  # the years crossed with the codebook's codes, fixed without the data
)
explicit_summary = explicit.summarize(alpha=0.05)
explicit_hat = explicit.release().collect()
```

```text
explicit keys: 54 keys, 36 with rows in the data, 18 without
explicit keys len: Frame Length, Integer Laplace, scale=144.00 95% accuracy=431.9
```

The scale and the accuracy do not change, and the threshold column is gone, because no group is dropped. The 18 keys without rows come back as signed noise around zero, so a reader cannot tell an empty group from a small one. A released key set is public, so the documentation reuses the keys of a stable release as the explicit keys of a later query.

`with_keys` stands for a join: the library rewrites the query as a right join of the aggregation onto the key list. If you write that join yourself, you get the same summary.

```text
right join len: Frame Length, Integer Laplace, scale=144.00 95% accuracy=431.9
```

Only a left or a right join against a key list releases a key set. An inner join keeps only the keys that have rows, which shows which keys the data holds, so the library refuses it.

```text
inner join: only left or right joins can be used to release key-sets
```

## Invariant keys {#sec-group-invariant}

The third way is to declare the keys of a grouping public through a **margin**, the set of public facts about a grouping that [the data and the context](../part01-getting-started/02-the-data-and-the-context.md#sec-ctx-margin) introduced. The margin `dp.polars.Margin(by=["YEAR", "QUARTER"], invariant="keys")` says that the set of quarters in the data is public. The library then releases every group's key in the clear, with no threshold and no $\delta$.

```{.python excerpt="group-invariant-keys"}
context = dp.Context.compositor(
    data=frame,
    privacy_unit=dp.unit_of(contributions=CONTRIBUTIONS),
    privacy_loss=dp.loss_of(epsilon=EPSILON),
    split_evenly_over=1,
    margins=[dp.polars.Margin(by=["YEAR", "QUARTER"], invariant="keys")],  # the quarters are public
)
quarterly = context.query().group_by("YEAR", "QUARTER").agg(dp.len(signed=True))
quarterly_summary = quarterly.summarize(alpha=0.05)
quarterly_hat = quarterly.release().collect()
```

```text
invariant keys len: Frame Length, Integer Laplace, scale=144.00 95% accuracy=431.9
true quarterly counts: 36 quarters, smallest 82,504 (2006 Q3), largest 131,656 (2011 Q4)
error bar as a share of the smallest quarter: 0.52%, of the largest: 0.33%
```

An **invariant** is a fact about the data that the guarantee no longer covers, so the guarantee is weaker. With invariant keys, a quarter with a single respondent would appear in the table and point at that person. Here nothing is lost, because the quarters are the survey's design and were known before anyone answered. Elsewhere, be reluctant to use invariants, as the documentation advises. The error bar is the same as before.

### Invariant lengths {#sec-group-lengths}

A margin can go one step further and declare the group sizes public, with `invariant="lengths"`. Public sizes imply public keys, so this gives up more. The documentation's example is the mean of `HWUSUAL`, the hours usually worked, by sex. The code 99 means not applicable and would bias the mean, so those rows must go first. But a filter inside the query changes the group sizes and voids the margin, and then the library treats the keys as private again.

```text
filter inside the query: The key-set of {col("SEX")} is private and cannot be released without a filter or join. Please pass a filtering threshold into make_private_lazyframe or conduct a join against a public key-set.
```

So filter the LazyFrame before you build the context, and the margin then describes the filtered data, 1,496,192 rows.

```{.python excerpt="group-invariant-lengths"}
context = dp.Context.compositor(
    data=working,
    privacy_unit=dp.unit_of(contributions=CONTRIBUTIONS),
    privacy_loss=dp.loss_of(epsilon=MEAN_EPSILON),
    split_evenly_over=1,
    margins=[
        dp.polars.Margin(
            by=["SEX"],
            invariant="lengths",  # the number of responses per sex is public, keys included
            max_length=MAX_LENGTH,  # needed by any sum: a public bound on the rows in a group
            max_groups=3,  # male, female or null
        )
    ],
)
mean_hours = context.query().group_by("SEX").agg(hours().dp.mean(HOURS_BOUNDS))
mean_summary = mean_hours.summarize(alpha=0.05)
mean_hat = mean_hours.release().collect()
```

```text
invariant lengths HWUSUAL: Sum, Integer Laplace, scale=1,764.00 95% accuracy=5,285.0
invariant lengths HWUSUAL: Length, Integer Laplace, scale=0.00 95% accuracy=none
true lengths: SEX=1 779,714  SEX=2 716,478
true mean hours: SEX=1 40.778  SEX=2 34.224
mean 95% accuracy: SEX=1 0.0068  SEX=2 0.0074
```

The library releases the mean as a sum divided by a length, with a summary row for each. The length has scale 0, because it is public and exact. The sum has scale 1764 at $\varepsilon = 1$, which is its sensitivity. The group sizes cannot change, so one person's rows can only be swapped for other rows. That is at most 18 swaps, each of which moves the sum by at most 98 hours. The documentation's text gives 7056 for this scale, but its own table and the installed version give 1764.

The sum's error bar of 5285 hours is divided by a public length of several hundred thousand. So on the mean, the error bar is below a hundredth of an hour for either sex. In version 0.16.0 the released means are floats, while the documentation shows them rounded to whole hours.

## Margins as public descriptors {#sec-group-margins}

A margin describes one grouping, named by its `by` columns, and a context takes a list of margins. So one context can protect most groupings fully and relax a few. Besides `invariant`, a margin has two bounds. The first, `max_groups`, is a public upper bound on the number of groups. It lowers the threshold of a release with stable keys. A person's rows can then make at most that many groups, so the library spreads $\delta$ over fewer of them. With a bound of 16, the threshold falls from 2773 to 2656.

```{.python excerpt="group-max-groups"}
context = dp.Context.compositor(
    data=frame,
    privacy_unit=dp.unit_of(contributions=CONTRIBUTIONS),
    privacy_loss=dp.loss_of(epsilon=EPSILON, delta=DELTA),
    split_evenly_over=1,
    margins=[dp.polars.Margin(by=["AGE", "ILOSTAT"], max_groups=GROUPS_IN_TRUTH)],  # at most 16 groups
)
bounded_summary = context.query().group_by("AGE", "ILOSTAT").agg(dp.len(signed=True)).summarize()
```

```text
max_groups=16: threshold=2656
```

The second, `max_length`, is a public upper bound on the number of rows in any one group. A sum needs it, because the guarantee covers overflow, and a sum of unbounded length could overflow. A count needs neither bound. A sum of hours per sex, with a margin that has no `max_length`, is refused.

```text
sum per SEX without max_length: must specify 'max_length' in a margin with by=[col("SEX")]
```

The guarantee does not check either bound. The bound of 16 above comes from the public data, which is acceptable for a tutorial but wrong for a release. Set both bounds conservatively from the design of the survey. A bound that the data violate voids the guarantee without an error.

## The noise per group and the ungrouped count {#sec-group-noise}

Every count in this chapter has the same scale, 144, from a group of the 16 to the ungrouped count of all rows.

```text
ungrouped count len: Frame Length, Integer Laplace, scale=144.00 95% accuracy=431.9
error bar as a share of the true count: all rows 0.011%, largest group 0.063%, smallest group above the threshold 3.3%
```

All 36 rows of a person can be in one group, so a group's count is as sensitive as the total. What changes is the relative error, which is 3.3% for the smallest group above the threshold. So finer groupings cost utility twice: more groups fall below the threshold, and the others have a larger relative error. Group as finely as the question needs and no finer, and ask for several statistics in one query. [Assessing utility and choosing the noise](07-utility-and-noise.md) returns to this grouping with Gaussian noise, whose scale depends on how a person's rows spread over the groups. [Identifier truncation and bounds](05-identifier-truncation.md) shows how to bound that spread.

## Checking the release against the truth {#sec-group-check}

The LFS sample is public, so the script compares each release with the true table, a check to run on public or historical data during development. For the stable keys, the smallest group above the threshold has 13,157 rows, and the largest below it has 847. So no group is near the threshold, and every run must return 14 groups. A group within one or two error bars of the threshold would appear on some runs only.

```text
smallest true count above the threshold: 13,157  largest below it: 847
test bound for a table: 4 x accuracy = 1,727.5
released (one run):
  stable groups released: 14  smallest released count: 13,321  smallest true count among them: 13,157
  stable AGE=47 ILOSTAT=1: 652,892  (true 652,801)
  stable AGE=65 ILOSTAT=2: 13,321  (true 13,157)
  stable largest |released - true|: 349
  explicit YEAR=2005 ILOSTAT=4: 312  (true 0)
  explicit YEAR=2013 ILOSTAT=1: 188,396  (true 188,349)
  explicit keys released: 54  largest |released - true|: 530
  quarter 2005 Q1: 87,307  (true 87,372)
  quarters released: 36  within their 95% accuracy: 36 of 36  largest |released - true|: 307
  mean hours: SEX=1 40.786 (true 40.778)  SEX=2 34.229 (true 34.224)
```

Read the run against the error bar of 431.9. The stable release has its 14 groups, all within the bar. The explicit release has all 54 keys, including the conscripts of 2005, whose true count is zero and whose released count is pure noise. Its largest error is a little over the bar. That is expected in a table of 54 counts, because each count misses a 95% bar about once in twenty. All 36 quarters are within their bars, and the means are within a hundredth of an hour of the truth.

The test checks the deterministic lines exactly, and every named count and mean within three error bars. For the largest error of a whole table it allows four bars, because a table has up to 54 counts. [Release statistics with a library](../../book-practitioner/part03-dp-foundations/12-release-statistics.md), in the practitioner edition, does the same kind of check on a shop's customers.

## Summary

| Call or concept | What it does | Rule of thumb |
| --- | --- | --- |
| `group_by(...).agg(dp.len(signed=True))` without keys or margins | Refused: the key set is private | Choose stable keys, explicit keys or a margin before you query |
| Stable keys, `dp.loss_of(epsilon, delta)` | Releases only groups whose noisy count exceeds a threshold | The threshold costs $\delta$; read it in `summarize` |
| The `threshold` column | The smallest value such that a one-person group clears it with probability at most $\delta$, over every group that person could create | Grows with the scale, with the privacy unit and with a smaller $\delta$ |
| `with_keys(keys)` | Joins a public key list; every key is released, zeros as noise | The list is fixed without the data; a released key set counts as public |
| `join(keys, how="right")` | The join that `with_keys` stands for | Only left or right joins release a key set |
| `Margin(by, invariant="keys")` | The keys of that grouping are released in the clear | Presence and absence of a key are unprotected; use for a design the public knows |
| `Margin(by, invariant="lengths")` | The group sizes are public, keys included | Gives up more than keys; filter before the context, not inside the query |
| `max_groups`, `max_length` | Public bounds on the number of groups and the rows per group | `max_groups` lowers the threshold; a sum needs `max_length`; set both from the design |
| One scale for every group | A person's rows can all sit in one group | Finer groups keep the noise and lose relative accuracy |

: The calls and concepts of a grouped release, what each does, and the rule of thumb that goes with it. {#tbl-group-summary}

## References

- OpenDP documentation: "Grouping" in the tabular-data tutorial, and "Aggregation" and "Polars" in the API user guide.
- Korolova, Kenthapadi, Mishra and Ntoulas, "Releasing Search Queries and Clicks Privately" (2009): thresholded noisy counts as a way to release keys privately.
- Desfontaines, Voss, Gipson and Mandayam, "Differentially Private Partition Selection" (2022): the threshold, its $\delta$, and why one person's groups add up.
- The full edition's chapters on [sensitivity](../../book-full/part03-dp-foundations/07-sensitivity.md), whose stability-based release is the thresholded count, and on [post-processing and composition](../../book-full/part03-dp-foundations/02-properties.md), which is why a released key set can be reused for free.
