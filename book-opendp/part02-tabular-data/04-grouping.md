# Grouping

Most private releases are tables with one row per group, such as counts by age and employment status or the mean hours worked by sex. Grouping adds a danger that a single count does not have. The keys of the table are data too, because a group that one person fills alone announces that person just by appearing in the table. This chapter is about the three ways OpenDP lets you release a grouped table, and what each one costs.

You will release counts by age and labour status with keys the library selects, so that only the groups whose noisy count clears a threshold appear. You will read that threshold and the $\delta$ it costs. Then you will release the same kind of table with a public key list, and with margins that declare the keys or the group sizes public. By the end you will be able to choose between the three, say what each gives up, and read a grouped release against its error bars.

The script [opendp_grouping.py](../../code/part02-tabular-data/opendp_grouping.py) runs every query of this chapter on OpenDP 0.16.0 against the full survey sample. The scales, error bars, thresholds and error messages it prints are the same on every run. So are the true group counts, which this public dataset lets the script compute. The released counts and means are not, because OpenDP's noise cannot be seeded, so the page shows them in one block marked as one run.

```bash
uv run python code/part02-tabular-data/opendp_grouping.py
```

## Why grouping keys are dangerous {#sec-group-danger}

The dataset is the France Labour Force Survey (LFS) sample of the OpenDP tutorials, which [Essential statistics](03-essential-statistics.md) introduced. It holds 3,811,932 quarterly responses from 1,682,306 people. A person answers at most once per quarter over nine years, so at most 36 times, and the privacy unit of the whole chapter is `dp.unit_of(contributions=36)`. Two datasets are neighbours when one is the other with all of one person's rows added or removed.

Grouping the responses by `AGE` and by `ILOSTAT`, the labour status code, gives 16 groups in truth. The largest, the children too young to work, holds 688,977 rows. The smallest, the responses with no recorded age, holds 255. A noisy count of that small group would come with an error bar of a few hundred, which hides how many rows it has but not that it exists. Now imagine a finer grouping, say by age, sex and town, in which one person fills a group alone. The noise hides their count, but the row's presence in the table says that someone with those keys answered the survey. So the keys of a grouped table need protection as much as the counts do.

OpenDP enforces this. Ask a context with a pure budget for a count by `AGE` and `ILOSTAT`, and `summarize` refuses before anything is computed.

```text
without delta: The key-set of {col("AGE"), col("ILOSTAT")} is private and cannot be released without a filter or join. Please pass a filtering threshold into make_private_lazyframe or conduct a join against a public key-set.
```

The message names the two ways out, and the next two sections take them in turn. A filter keeps only the groups whose noisy count clears a threshold, which needs a $\delta$ in the budget. A join against a public key list releases only keys you already knew. The snippet below shows both on a toy frame, so you can see the shape of each summary without the dataset.

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

The first way is to let the library decide which groups to show. The documentation calls the keys it keeps **stable keys**, because a group is stable when it would still be there after any one person was removed from the data. The release keeps a group only when its noisy count exceeds a threshold. The threshold is set so that a group one person fills alone is released with probability at most $\delta$. That slack is why the budget needs a $\delta$, and the chapter uses the documentation's values, `dp.loss_of(epsilon=0.25, delta=1e-7)`. This is **partition selection**, and [the full edition's chapter on sensitivity](../../book-full/part03-dp-foundations/07-sensitivity.md) derives why a thresholded count satisfies approximate differential privacy and where its $\delta$ comes from.

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

Read the summary before releasing. The scale is 144, for the same reason as in the ungrouped count of the previous chapter. One person's 36 rows can all sit in one group, so the count of any group has sensitivity 36. That sensitivity over $\varepsilon = 0.25$ is the scale. The accuracy says that with probability 0.95 a released count is within 431.9 of the truth. The new column is the threshold, 2773, and a group whose noisy count does not exceed it is dropped from the release.

The threshold is calibrated to the worst case for one person. Their 36 rows could all land in one otherwise empty group, whose true count would then be 36. That group is released only if the noise lifts it above the threshold, which for discrete Laplace noise of scale 144 has probability $2.77 \times 10^{-9}$. But the same person's rows could instead spread over 36 groups with one row each, and any of them might slip through. So the library adds up the chances over 36 groups. That gives $9.98 \times 10^{-8}$, just within $\delta = 10^{-7}$. One step lower, the total would exceed $\delta$. The script computes both tails, so the threshold you see is the smallest one that keeps the promise.

```text
one person's 36 rows clear 2773 with probability 2.77e-09; over 36 groups 9.98e-08 <= delta 1e-07
at threshold 2772: over 36 groups 1.01e-07 > delta
```

The threshold moves with the scale and with $\delta$. A looser slack of $\delta = 10^{-5}$ lowers it to 2110 at the same scale. A budget of $\varepsilon = 1$ shrinks the scale to 36. The threshold falls to 720 with it. A larger privacy unit raises both, because a person who owns more rows can build a bigger group alone.

```text
threshold with delta=1e-05: scale=144.00 threshold=2110
threshold with epsilon=1: scale=36.00 threshold=720
```

On this public data the script can also count what the threshold will do. Of the 16 groups, 14 have a true count above 2773, and the two below it are the groups with 255 and with 847 rows. So the release should come back with 14 rows, and it did in the run shown at the end of the chapter. A dropped group does not announce itself. The release has no row saying that something was removed, and the number of rows you get back is itself noisy information. If you see fewer groups than you expected, do not look at the data to find out which are missing, because that look is outside the guarantee.

## Explicit keys {#sec-group-explicit}

If you know the keys in advance, you can give them to the library and keep a pure budget. **Explicit keys** are a key list that you supply, and every key in it is released, whether or not the data holds rows for it. The list must be public, which means fixed without looking at the data, because a key list read off the data is a release in itself. Here the list is every survey year crossed with every status code of the codebook, 54 keys in all. It includes the conscripts and the "other" category, which never occur in this sample.

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

The scale and the accuracy are unchanged, and the threshold column is gone, since no group is dropped. Of the 54 keys, 36 have rows in the data. The other 18 come back as noise around zero, which can be negative because the counts are signed. A reader of the table cannot tell an empty group from a small one, and that is the point. The documentation reuses the keys of a stable release as the explicit keys of a later query, which is allowed because a released key set is public.

`with_keys` stands for a join. The library rewrites the query as a right join of the aggregation onto the key list, and you can write that join yourself with the same summary.

```text
right join len: Frame Length, Integer Laplace, scale=144.00 95% accuracy=431.9
```

Only a left or a right join against a key list releases a key set. An inner join would keep just the keys that have rows, which reveals which keys the data holds, so the library refuses it.

```text
inner join: only left or right joins can be used to release key-sets
```

## Invariant keys {#sec-group-invariant}

The third way is to declare the keys of a grouping public through a **margin**, which is OpenDP's description of what is known about a grouped table, as in the margins of a printed table. The margin `dp.polars.Margin(by=["YEAR", "QUARTER"], invariant="keys")` says that the set of quarters in the data is public. The library then releases every group's key in the clear, with no threshold and no $\delta$.

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

An **invariant** is a fact about the data that the guarantee no longer covers, and the documentation calls the resulting guarantee weaker for that reason. With invariant keys, the presence and the absence of each key are released exactly, so a quarter with a single respondent would appear in the table and point at them. Here nothing is lost, because the quarters are the survey's design and were known before anyone answered. The documentation adds that invariants have been used in high-profile releases, by the United States Census Bureau among others, and that you should still be reluctant to use them. The error bar is the same as before, but the quarters are smaller than the age groups, so it is a larger share of each count, about half a percent of the smallest quarter.

### Invariant lengths {#sec-group-lengths}

A margin can go one step further and declare the group sizes public, with `invariant="lengths"`. That implies the keys are public too, so it gives up more. The documentation's example is the mean of `HWUSUAL`, the hours usually worked, by sex. The code 99 means not applicable and would bias the mean, so those rows must go first. But a filter inside the query changes the group sizes and voids the margin, and the library then treats the keys as private again.

```text
filter inside the query: The key-set of {col("SEX")} is private and cannot be released without a filter or join. Please pass a filtering threshold into make_private_lazyframe or conduct a join against a public key-set.
```

The way around it is to filter the LazyFrame before the context is built, so that the margin describes the filtered data. That leaves 1,496,192 rows.

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

The mean is released as a sum divided by a length, and the summary has a row for each. The length has scale 0, because it is public and used exactly. The sum has scale 1764 at $\varepsilon = 1$, which is its sensitivity. Since the group sizes cannot change, one person's rows can only be swapped for other rows. That is at most 18 swaps, and each swap moves the sum by at most 98 hours. The documentation's text quotes 7056 for this scale, but its own table and the installed version give 1764.

An error bar of 5285 hours on a sum looks large, but the sum is divided by a public length of several hundred thousand. On the mean the error bar is below a hundredth of an hour for either sex, as the script computes from the true lengths. The released means are floats in version 0.16.0, where the documentation shows them rounded to whole hours.

## Margins as public descriptors {#sec-group-margins}

A margin describes one grouping, named by its `by` columns, and a context takes a list of margins. So one context can protect most groupings fully while relaxing a few, which is how a real analysis uses them. Besides `invariant`, a margin carries two bounds. The first, `max_groups`, is a public upper bound on the number of groups the grouping can have. It lowers the threshold of a stable-keys release, because a person's rows can then create at most that many groups and the $\delta$ is spread over fewer of them. With a bound of 16 the threshold falls from 2773 to 2656.

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

The second, `max_length`, is a public upper bound on the number of rows in any one group of the grouping. A sum needs it, because the library's guarantee covers overflow and a sum of unbounded length could overflow. A count needs neither bound. Ask for a sum of hours per sex with a margin that lacks it, and the query is refused.

```text
sum per SEX without max_length: must specify 'max_length' in a margin with by=[col("SEX")]
```

Both bounds are promises about the data that the guarantee does not check. The bound of 16 above was read off the public data at development time, which is fine for a tutorial and wrong for a release. Set them from the design of the survey, never from the private data, and set them conservatively, because a bound that the data violates voids the guarantee without any error.

## The noise per group and the ungrouped count {#sec-group-noise}

Every count in this chapter had the same scale, 144, whether it was one group among 16, one quarter among 36 or the ungrouped count of all rows.

```text
ungrouped count len: Frame Length, Integer Laplace, scale=144.00 95% accuracy=431.9
error bar as a share of the true count: all rows 0.011%, largest group 0.063%, smallest group above the threshold 3.3%
```

The reason is the privacy unit. All 36 rows of a person can sit in one group, so a group's count is as sensitive as the total, and grouping does not spread the noise thinner. What changes is the relative error. The same error bar of 431.9 is a hundredth of a percent of all rows. It is 3.3% of the smallest group that clears the threshold. Finer groupings therefore cost utility twice, since more groups fall below the threshold and the survivors carry a larger relative error.

The documentation's advice follows from this. Group as finely as the question needs and no finer, and ask for several statistics in one query rather than releasing the same keys again and again. [Assessing utility and choosing the noise](07-utility-and-noise.md) returns to this grouping with Gaussian noise, whose scale depends on how a person's rows spread over the groups. [Identifier truncation and bounds](05-identifier-truncation.md) shows how to bound that spread.

## Checking the release against the truth {#sec-group-check}

Because the LFS sample is public, the script compares each release with the true table, which a real release never can. This is the check to run on public or historical data while you develop. For the stable keys it answers two questions, how many groups should survive and how close to the threshold the dropped ones were. The smallest group above the threshold has 13,157 rows, and the largest below it has 847. So no group sits within reach of the threshold, and the release should return 14 groups on every run. A group within an error bar or two of the threshold would appear on some runs and not on others, and the script would print it here.

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

Read the run against the error bar of 431.9. The stable release has its 14 groups, and every one of them landed within the bar. The explicit release has all 54 keys, including the conscripts of 2005, whose true count is zero and whose released count is a noise draw. Its largest error is a little over the bar, which is what a table of 54 counts should show, since each count misses a 95% bar about once in twenty. All 36 quarters landed within theirs. The means are within a hundredth of an hour of the truth, as their error bars said.

The test checks the deterministic lines exactly. For the stable release it checks three things. It must have 14 groups, each must exceed the threshold, and each must have a true count above it. Every named count must be within three error bars of its truth. For a whole table the test allows the largest error four bars, the 1727.5 the script prints, because a table has up to 54 counts and the chance that any of them misses four bars is small enough. Each mean must be within three of its own bar. The practitioner edition's workflow [Release statistics with a library](../../book-practitioner/part03-dp-foundations/12-release-statistics.md) runs the same kind of check on a shop's customers. There the keys are products and the privacy unit is a customer with many lines.

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
