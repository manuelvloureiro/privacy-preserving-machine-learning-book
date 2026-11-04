# Grouping

Many private releases are grouped tables, such as counts by age and employment status or mean hours by sex. Their keys can reveal information: a group containing only one person's data identifies that person by its presence. This chapter compares three ways OpenDP releases grouped tables and the privacy cost of each.

The first method selects keys and retains groups whose noisy counts exceed a threshold. The second uses a public key list. The third declares keys or group sizes public through margins. The choice determines what information the release protects.

The script [opendp_grouping.py](../../code/part02-tabular-data/opendp_grouping.py) runs every query of this chapter on OpenDP 0.16.0 against the full survey sample. The scales, error bars, thresholds, error messages and true group counts that it prints are the same on every run. The released counts and means are not, because OpenDP's noise cannot be seeded, so the page shows them in one block marked as one run.

```bash
uv run python code/part02-tabular-data/opendp_grouping.py
```

## Why grouping keys are dangerous {#sec-group-danger}

The data are the France Labour Force Survey (LFS) sample described in [The data and the context](../part01-getting-started/02-the-data-and-the-context.md#sec-ctx-data). The privacy unit matches [that chapter's context](../part01-getting-started/02-the-data-and-the-context.md#sec-ctx-context): up to 36 rows per person, one per quarter.

Grouping responses by `AGE` and `ILOSTAT` (labour status) produces 16 groups. The smallest group, with no recorded age, has 255 rows. Noise with an error bar of a few hundred can hide its count but not its existence. In a finer grouping by age, sex and town, one person could fill a group alone. Noise would hide the count, but the row would reveal that someone with those keys answered the survey. Group keys therefore need protection as well as counts.

OpenDP enforces this. On a context with a pure budget, `summarize` refuses a count by `AGE` and `ILOSTAT` before it computes anything.

```text
without delta: The key-set of {col("AGE"), col("ILOSTAT")} is private and cannot be released without a filter or join. Please pass a filtering threshold into make_private_lazyframe or conduct a join against a public key-set.
```

The message gives two options. A filter retains groups whose noisy counts pass a threshold and requires a $\delta$ budget. A join uses a public key list. The snippet summarizes both methods on a toy frame.

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

The first method lets OpenDP select the groups to release. The documentation calls retained groups **stable keys**: a group is stable if it remains after removing any one person. OpenDP releases a group only when its noisy count exceeds a threshold. It sets the threshold so a group filled by one person appears with probability at most $\delta$. This chapter uses the documentation's budget, `dp.loss_of(epsilon=0.25, delta=1e-7)`. This method, **partition selection**, satisfies approximate differential privacy; [the full edition's chapter on sensitivity](../../book-full/part03-dp-foundations/07-sensitivity.md) derives the guarantee.

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

The summary gives sensitivity 36 because one person's rows can all belong to one group. Dividing by $\varepsilon=0.25$ gives scale 144. With probability 0.95, each released count is within 431.9 of its true value. The threshold is 2773; groups with noisy counts at or below it are dropped.

The threshold accounts for the worst case across one person's rows. All 36 rows could enter one empty group, giving it a true count of 36. Discrete Laplace noise with scale 144 raises that count above 2773 with probability $2.77 \times 10^{-9}$. Alternatively, the rows could enter 36 separate one-row groups. OpenDP adds the 36 probabilities, giving $9.98 \times 10^{-8}$, just below $\delta=10^{-7}$. A threshold of 2772 exceeds $\delta$, so 2773 is the smallest threshold that meets the guarantee.

```text
one person's 36 rows clear 2773 with probability 2.77e-09; over 36 groups 9.98e-08 <= delta 1e-07
at threshold 2772: over 36 groups 1.01e-07 > delta
```

The threshold decreases when $\delta$ is larger or the noise scale is smaller. A larger privacy unit raises both scale and threshold because one person can contribute more rows to a group.

```text
threshold with delta=1e-05: scale=144.00 threshold=2110
threshold with epsilon=1: scale=36.00 threshold=720
```

In this public dataset, 14 of 16 groups have true counts above 2773, so the release contains 14 rows. The number of released groups is itself private information. Do not inspect the data to identify missing groups; that inspection is outside the guarantee.

## Explicit keys {#sec-group-explicit}

If the keys are known in advance, provide them to the library and use a pure privacy budget. **Explicit keys** are a supplied key list; the release includes every listed key, whether or not it occurs in the data. Construct the list without inspecting private data, because keys selected from the data are a release. Here the list contains every survey year crossed with every status code, including codes absent from the sample.

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

The scale and accuracy are unchanged, and there is no threshold because no group is dropped. The 18 empty keys receive signed noise around zero, so readers cannot distinguish empty groups from small ones. A released key set is public, so the documentation reuses the keys from a stable release in a later query.

`with_keys` performs a right join from the aggregation to the key list. Writing the join directly gives the same summary.

```text
right join len: Frame Length, Integer Laplace, scale=144.00 95% accuracy=431.9
```

Only left or right joins against a key list can release a key set. An inner join retains only keys present in the data and therefore reveals which keys occur; OpenDP rejects it.

```text
inner join: only left or right joins can be used to release key-sets
```

## Invariant keys {#sec-group-invariant}

The third method declares grouping keys public through a **margin**, introduced in [The data and the context](../part01-getting-started/02-the-data-and-the-context.md#sec-ctx-margin). The margin `dp.polars.Margin(by=["YEAR", "QUARTER"], invariant="keys")` declares the set of quarters public. OpenDP then releases every group key in the clear, without a threshold or $\delta$.

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

An **invariant** is a public fact excluded from the guarantee, which makes the guarantee weaker. Public keys could reveal a quarter containing a single respondent. Here the quarter set comes from the survey design and was known before responses were collected. In other settings, use invariants cautiously, as the documentation advises. The error bar is unchanged.

### Invariant lengths {#sec-group-lengths}

A margin can also declare group sizes public with `invariant="lengths"`. Public sizes imply public keys, so this reveals more. The documentation's example computes mean `HWUSUAL` (usual hours) by sex. Code 99 means not applicable and would bias the mean, so remove those rows first. A filter inside the query changes group sizes and invalidates the margin, causing OpenDP to treat the keys as private again.

```text
filter inside the query: The key-set of {col("SEX")} is private and cannot be released without a filter or join. Please pass a filtering threshold into make_private_lazyframe or conduct a join against a public key-set.
```

Filter the `LazyFrame` before building the context; the margin then describes the filtered data, which has 1,496,192 rows.

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

OpenDP releases the mean as a sum divided by a length and reports each separately. The public length has scale 0. At $\varepsilon=1$, the sum's scale is 1764, its sensitivity. Because group sizes are fixed, one person's rows can only replace other rows: at most 18 swaps, each changing the sum by up to 98 hours. The documentation's prose gives scale 7056, but its table and OpenDP 0.16.0 give 1764.

Dividing the sum's 5285-hour error bar by a public group length of several hundred thousand gives a mean error below 0.01 hours for either sex. In version 0.16.0, released means are floats; the documentation rounds them to whole hours.

## Margins as public descriptors {#sec-group-margins}

A margin describes one grouping through its `by` columns, and a context accepts multiple margins. One context can therefore protect most groupings fully and relax selected ones. In addition to `invariant`, a margin has two bounds. `max_groups` publicly bounds the number of groups and lowers the threshold for stable keys: one person's rows can affect at most that many groups, so OpenDP divides $\delta$ among fewer groups. With a bound of 16, the threshold falls from 2773 to 2656.

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

`max_length` publicly bounds the rows in each group. A sum needs this bound to account for overflow; an unbounded sum could overflow. Counts need neither bound. OpenDP rejects a sum of hours by sex if the margin lacks `max_length`.

```text
sum per SEX without max_length: must specify 'max_length' in a margin with by=[col("SEX")]
```

OpenDP does not verify either bound. The bound of 16 comes from the public sample, which is suitable for a tutorial but not a private release. Set both bounds conservatively from the survey design. If the data violate a bound, the guarantee fails without an error.

## The noise per group and the ungrouped count {#sec-group-noise}

Every count in this chapter has scale 144, whether it counts one of the 16 groups or all rows.

```text
ungrouped count len: Frame Length, Integer Laplace, scale=144.00 95% accuracy=431.9
error bar as a share of the true count: all rows 0.011%, largest group 0.063%, smallest group above the threshold 3.3%
```

One person can contribute all 36 rows to a group, so its count is as sensitive as the total. Relative error differs: it is 3.3% for the smallest group above the threshold. Finer groupings reduce utility twice: more groups fall below the threshold, and those that remain have larger relative errors. Use only the grouping detail the question requires, and request several statistics in one query. [Assessing utility and choosing the noise](07-utility-and-noise.md) compares Gaussian noise, whose scale depends on how one person's rows spread across groups. [Identifier truncation and bounds](05-identifier-truncation.md) shows how to bound that spread.

## Checking the release against the truth {#sec-group-check}

The LFS sample is public, so the script compares each release with the true table. Run this check on public or historical data during development. For stable keys, the smallest group above the threshold has 13,157 rows; the largest below it has 847. No group is near the threshold, so every run returns 14 groups. A group within one or two error bars of the threshold would appear only on some runs.

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

Compare the run with the 431.9 error bar. The stable release contains 14 groups, all within the bar. The explicit release contains all 54 keys, including the 2005 conscript group, whose true count is zero and whose released count is pure noise. Its largest error slightly exceeds the bar, as expected for 54 counts when each misses a 95% bar about once in twenty. All 36 quarters fall within their bars, and the means are within 0.01 hours of the true values.

The test checks deterministic outputs exactly and each named count and mean within three error bars. It allows four bars for the largest table error because a table has up to 54 counts. The practitioner edition's [Release statistics with a library](../../book-practitioner/part03-dp-foundations/12-release-statistics.md) performs a similar check on retail customers.

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
