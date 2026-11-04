# Assessing utility and choosing the noise

Earlier chapters reported each query's noise scale and accuracy before release. This chapter explains how to interpret and adjust them. **Accuracy** is an error bar's half-width: with the stated probability, the released value is within that distance of the value to which noise was added. OpenDP converts between scale and accuracy, so you can price a target error bar before spending the budget.

The chapter also compares noise mechanisms. A pure differential privacy (DP) budget makes the context add Laplace noise; a zero-concentrated differential privacy (zCDP) budget makes it add Gaussian noise. An argument to `dp.loss_of` selects the guarantee and mechanism. The choice matters most when one person contributes to many groups: Gaussian noise uses sensitivity proportional to the square root of the group count rather than the count itself. On the survey, this reduces quarterly-count noise six-fold under zCDP.

The script [opendp_utility_and_noise.py](../../code/part02-tabular-data/opendp_utility_and_noise.py) runs every step on OpenDP 0.16.0 against the Labour Force Survey (LFS) sample of the earlier chapters. The conversions, scales, accuracies, thresholds and budgets that it prints are the same on every run. The released counts and the means of a small clamping experiment are not, because OpenDP's noise cannot be seeded, so this page shows them in blocks marked as one run.

```bash
uv run python code/part02-tabular-data/opendp_utility_and_noise.py
```

## What an accuracy statement means {#sec-util-accuracy}

At a given scale, the released value lies within the error bar of the mechanism's input with probability $1-\alpha$, or confidence $(1-\alpha)\times100\%$. This book uses $\alpha=0.05$, so about one in twenty releases falls outside the bar. Smaller $\alpha$ gives a wider bar, as shown at $\alpha=0.01$ in [Essential statistics](03-essential-statistics.md).

The `opendp.accuracy` module, exposed as `dp` in the prelude, provides eight conversion functions. Four convert continuous Laplace or Gaussian scales and accuracies; four functions prefixed `discrete_` do the same for integer noise. The documentation's example finds the Gaussian scale needed to stay within 2 of the input with 95% confidence.

```python
import opendp.prelude as dp

print(dp.accuracy_to_gaussian_scale(accuracy=2.0, alpha=0.05))  # 1.020426913849308
print(dp.gaussian_scale_to_accuracy(scale=180.0, alpha=0.05))  # 352.7935172172097
print(dp.discrete_gaussian_scale_to_accuracy(scale=180.0, alpha=0.05))  # 354.0
print(dp.laplacian_scale_to_accuracy(scale=180.0, alpha=0.05))  # 539.2318092397184
print(dp.discrete_laplacian_scale_to_accuracy(scale=180.0, alpha=0.05))  # 539.7311147961669
```

The continuous conversions have closed forms. Laplace accuracy is scale times $\ln(1/\alpha)$; Gaussian accuracy is scale times the two-sided normal quantile, about 1.96 at the usual confidence level. Discrete conversions sum integer-noise probabilities, so discrete Gaussian accuracy is an integer. Use discrete conversions for counts, which receive integer noise, and continuous conversions for sums and means of floats. At scale 180 the difference is small; `summarize` selects the appropriate conversion.

The error bar covers noise only. It measures the distance from the release to the mechanism's input: the statistic after preparation. Clipping, imputation, row truncation and resizing can change that input. The bar does not cover the resulting bias relative to the intended quantity; repeated releases do not remove bias. The practitioner edition's [release workflow](../../book-practitioner/part03-dp-foundations/12-release-statistics.md) measures bias from truncation and clipping, which retain about half the revenue.

## A target error bar on the survey {#sec-util-target}

Suppose the survey response count must fall within 100 of its true value with 95% confidence. The count in [Essential statistics](03-essential-statistics.md#sec-stats-count) used $\varepsilon=1$ split among five queries. The excerpt summarizes that query and calculates the discrete Laplace scale required for the target accuracy.

```{.python excerpt="util-target"}
laplace = laplace_context(frame, EPSILON)
count_laplace = laplace.query().select(dp.len(signed=True))
laplace_summary = count_laplace.summarize(alpha=ALPHA)
scale_needed = float(dp.accuracy_to_discrete_laplacian_scale(accuracy=TARGET, alpha=ALPHA))
epsilon_needed = CONTRIBUTIONS / scale_needed  # the Laplace scale is the sensitivity over epsilon
```

```text
laplace count: Integer Laplace scale=180 95% accuracy=539.731
budget per query: epsilon=0.2000
error bar relative to the true count: 0.014%
discrete laplace scale for accuracy 100: 33.2152
epsilon that scale needs at sensitivity 36: 1.0838  (5.42 times the share)
```

The current scale, 180, is sensitivity 36 divided by the query share 0.2; see [the privacy unit](../part01-getting-started/02-the-data-and-the-context.md#sec-ctx-context). Accuracy 100 requires scale 33.2, which would consume more than the full budget on this query. The current error bar is 0.014% of the true count, already small for a headline figure.

The conversion also gives the trade-off: Laplace scale is sensitivity divided by $\varepsilon$, and accuracy is proportional to scale. Halving the error bar therefore doubles the budget. At $\varepsilon=2$, this query's scale is halved.

```text
laplace count at epsilon 2: Integer Laplace scale=90 95% accuracy=270.115
```

The discrete conversion makes the bar slightly different from half because integer noise is rounded. The rule of thumb still holds. [The full edition's chapter on the Laplace mechanism](../../book-full/part03-dp-foundations/03-laplace-gaussian.md) derives the scale and tail bound.

## Laplace under pure DP, Gaussian under zCDP {#sec-util-gaussian}

The budget type determines the default noise. `dp.loss_of(epsilon=...)` selects pure DP and Laplace noise; `dp.loss_of(rho=...)` selects zCDP and Gaussian noise. Other query details stay the same unless a query specifies a distribution, so the two context functions differ only in their budget declarations.

```{.python excerpt="util-contexts"}
def laplace_context(frame: pl.LazyFrame, epsilon: float) -> dp.Context:
    """Pure differential privacy: the context adds Laplace noise."""
    return dp.Context.compositor(
        data=frame,
        privacy_unit=dp.unit_of(contributions=CONTRIBUTIONS),
        privacy_loss=dp.loss_of(epsilon=epsilon),
        split_evenly_over=QUERIES,
    )


def gaussian_context(frame: pl.LazyFrame, rho: float, delta: float | None = None) -> dp.Context:
    """Zero-concentrated differential privacy: the context adds Gaussian noise."""
    return dp.Context.compositor(
        data=frame,
        privacy_unit=dp.unit_of(contributions=CONTRIBUTIONS),
        privacy_loss=dp.loss_of(rho=rho, delta=delta),
        split_evenly_over=QUERIES,
    )
```

Privacy loss is the log ratio of an output's probabilities with and without a person's data. Pure DP bounds it by $\varepsilon$ in the worst case. zCDP bounds its moment-generating behaviour: a mechanism is $\rho$-zCDP when privacy loss has mean at most $\rho$ and tails no heavier than a normal variable with variance $2\rho$. The Gaussian mechanism satisfies this bound exactly, and $\rho$ budgets add across queries like $\varepsilon$ budgets. [The full edition's chapter on concentrated DP](../../book-full/part04-privacy-accounting/03-concentrated-dp.md) derives the guarantee. The documentation's Gaussian example repeats the count with $\rho=0.1$ split across five queries.

```{.python excerpt="util-gaussian"}
gaussian = gaussian_context(frame, RHO)
count_gaussian = gaussian.query().select(dp.len(signed=True))
gaussian_summary = count_gaussian.summarize(alpha=ALPHA)
```

```text
privacy loss: (ZeroConcentratedDivergence, 0.1)
gaussian count: Integer Gaussian scale=180 95% accuracy=354
budget per query: rho=0.0200  sqrt(2 rho)=0.2000
continuous gaussian accuracy at scale 180: 352.7935
```

The scale is again 180. Let $\sigma$ denote the Gaussian standard deviation (OpenDP's scale) and $\Delta$ the sensitivity. The mechanism satisfies $\Delta^2/(2\sigma^2)$-zCDP, so the context sets $\sigma=\Delta/\sqrt{2\rho}$. At $\rho=0.02$, the denominator is 0.2, matching the Laplace budget share. Gaussian noise has a narrower bar at equal scale because its tails decay faster. Equal scales do not imply equal privacy: @sec-util-epsilon converts this $\rho$ share to $\varepsilon\approx1$, five times the Laplace share.

The Gaussian trade-off differs: scale is sensitivity divided by $\sqrt{2\rho}$, so halving the error bar requires four times the $\rho$ budget. The script confirms this at $\rho=0.4$.

```text
gaussian count at rho 0.4: Integer Gaussian scale=90 95% accuracy=177
```

Gaussian noise under zCDP offers no gain for a single count. Its advantage appears in queries that release many values at once.

## One person in many groups: L1 against L2 {#sec-util-sensitivity}

A grouped count returns one number per group. Noise is calibrated to the distance one person can move the full vector. Laplace noise uses the $L_1$ norm, the sum of absolute changes; Gaussian noise uses the $L_2$ norm, the square root of the sum of squared changes. Opposite corners of a unit square have distances 2 in $L_1$ and $\sqrt2$ in $L_2$. In $d$ dimensions, these distances are $d$ and $\sqrt d$.

Each survey respondent answers at most once per quarter, so their 36 records occupy 36 year-quarter groups, changing each count by one. The count vectors have $L_1$ distance 36 and $L_2$ distance $\sqrt{36}=6$. Without this information, OpenDP must assume all 36 records fall in one group, making both distances 36. A `dp.polars.Bound`, described in [Identifier truncation and bounds](05-identifier-truncation.md), supplies the per-group bound.

```{.python excerpt="util-bounded-context"}
def bounded_context(frame: pl.LazyFrame, privacy_loss: tuple[Measure, Any]) -> dp.Context:
    """A budget of either kind, with the privacy unit also bounded to one record per year and quarter."""
    return dp.Context.compositor(
        data=frame,
        privacy_unit=dp.unit_of(
            contributions=[
                dp.polars.Bound(by=[], per_group=CONTRIBUTIONS),  # at most 36 records overall
                dp.polars.Bound(by=["YEAR", "QUARTER"], per_group=1),  # and one in each year and quarter
            ]
        ),
        privacy_loss=privacy_loss,
        split_evenly_over=QUERIES,
    )
```

The query has no public keys, so its budget includes $\delta$ and the context thresholds group counts as described in [Grouping](04-grouping.md). The script summarizes quarterly counts under three settings: zCDP with only the 36-contribution bound, zCDP with a per-group bound, and pure DP with that bound.

```{.python excerpt="util-grouped"}
bounded = bounded_context(frame, dp.loss_of(rho=RHO, delta=DELTA))
count_by_quarter = bounded.query().group_by("YEAR", "QUARTER").agg(dp.len(signed=True))
grouped_summary = count_by_quarter.summarize(alpha=ALPHA)
```

```text
privacy loss with a threshold: (Approximate(ZeroConcentratedDivergence), (0.1, 1e-07))
grouped count, 36 contributions: Integer Gaussian scale=180 95% accuracy=354 threshold=1133
grouped count, 1 per year and quarter: Integer Gaussian scale=30 95% accuracy=60 threshold=184
grouped count, 1 per year and quarter, pure DP: Integer Laplace scale=180 95% accuracy=539.731 threshold=3712
true groups: 36  smallest 82,504  largest 131,656
```

The per-group bound reduces the Gaussian scale from 180 to 30, a factor of six, and lowers the threshold. Under pure DP it changes nothing because the $L_1$ distance remains 36. Both thresholds are below the smallest quarterly count, so all groups are released; the Gaussian error bar is 60. [The full edition's chapter on the two mechanisms](../../book-full/part03-dp-foundations/03-laplace-gaussian.md) derives Gaussian calibration from $L_2$ sensitivity.

When one person's records are spread across many groups, $L_2$ sensitivity can be much smaller than $L_1$ sensitivity, making Gaussian noise under zCDP preferable. This gain requires a per-group bound. If public information cannot justify it, enforce the bound through truncation.

## Stating the guarantee in epsilon {#sec-util-epsilon}

A $\rho$ budget is convenient for accounting but less familiar to release readers. OpenDP converts it to $(\varepsilon,\delta)$ with two combinators described in [Combinators](../part03-framework/11-combinators.md). `make_zCDP_to_approxDP` produces an $\varepsilon$-versus-$\delta$ privacy profile; `make_fix_delta` selects a $\delta$ from that profile. The script applies them to a Gaussian measurement calibrated so sensitivity 1 costs exactly $\rho$.

```{.python excerpt="util-epsilon"}
def epsilon_of_rho(rho: float, delta: float) -> float:
    """The epsilon that OpenDP's conversion gives for rho-zCDP at delta, read off a Gaussian measurement."""
    space = dp.atom_domain(T=float, nan=False), dp.absolute_distance(T=float)
    gaussian = dp.m.make_gaussian(*space, scale=1 / math.sqrt(2 * rho))  # sensitivity 1 at this scale is rho-zCDP
    approx = dp.c.make_fix_delta(dp.c.make_zCDP_to_approxDP(gaussian), delta)
    epsilon, _ = approx.map(1.0)
    return float(epsilon)
```

```text
rho=0.1 at delta=1e-07: epsilon=2.3484
closed form rho + 2 sqrt(rho ln(1/delta)): 2.6391
rho=0.02 (one query) at delta=1e-07: epsilon=0.9951
the whole context converted: epsilon=2.3484 delta=1e-07
```

At $\delta=10^{-7}$, the full $\rho=0.1$ budget converts to $\varepsilon=2.35$, and one query's share converts to about $\varepsilon=1$. The textbook conversion is
$$\varepsilon = \rho + 2\sqrt{\rho \ln(1/\delta)},$$
which gives 2.64 here. OpenDP's tighter conversion over Rényi-divergence orders is derived in [the full edition's chapter on conversions](../../book-full/part04-privacy-accounting/02-rdp-conversions.md). To convert a full context, apply the combinators to its `accountant` attribute, the measurement that composes all queries. It also gives 2.35, the value to report with the release.

You choose $\delta$. Keep it far below the reciprocal of the number of people; otherwise, the guarantee permits a mechanism that publishes one person's record. For 1.7 million people, $10^{-7}$ is defensible. Conversion $\delta$ is separate from the grouping-threshold $\delta$, so a release using both must budget both.

The Gaussian count summary reports accuracy 354; the continuous conversion at the same scale gives 352.79. The context adds discrete Gaussian noise to integer counts, so it reports discrete accuracy, as [Measurements](../part03-framework/10-measurements.md) explains. Use the discrete conversion to calculate count accuracy from scale.

## Signed and unsigned counts {#sec-util-signed}

Every count here uses `signed=True` to avoid upward bias. Noise can make a small count negative; OpenDP previously clamped such results to zero. In version 0.16.0, `signed=True` casts the count to a signed integer before adding noise, so negative results remain. `signed=False` or an omitted argument clamps at zero; OpenDP warns that the default will change.

Clamping can reduce error for one small cell, but it biases sums of counts upward. A clamped count can exceed its true value but cannot fall below zero, so totals over many small cells accumulate positive bias. The OpenDP cookbook demonstrates this on 400 mostly empty cells: clamped totals are too high in every repetition, while signed totals are correct on average.

Clamp signed counts after release for display if needed; this is free post-processing. Release signed counts whenever others may sum, average or model them. The script illustrates the bias by releasing an empty frame 1,000 times with fixed Gaussian scale 30. It uses the framework API without a context and is intended only for toy data.

```{.python excerpt="util-clamping"}
def mean_noisy_count_of_nobody(signed: bool, scale: float, repeats: int) -> float:
    """The mean of `repeats` noisy counts of an empty frame, at a fixed Gaussian scale, signed or clamped at zero."""
    empty = pl.LazyFrame(schema={"A": pl.Int32})
    domain = dp.with_margin(
        dp.lazyframe_domain([dp.series_domain("A", dp.atom_domain(T="i32"))]),
        dp.polars.Margin(by=[], max_length=1000),
    )
    count = dp.m.make_private_lazyframe(
        domain, dp.symmetric_distance(), dp.zero_concentrated_divergence(), empty.select(dp.len(signed=signed)), scale
    )
    return sum(int(count(empty).collect().item()) for _ in range(repeats)) / repeats
```

```text
dp.len() without signed: warns=True clamped=True
expected clamped mean at scale 30: scale/sqrt(2 pi)=11.9683
standard error of the signed mean: scale/sqrt(1000)=0.9487
clamping experiment (one run): 1000 noisy counts of nobody at gaussian scale 30
  unsigned mean 11.48  signed mean 1.02  (true count 0)
```

The true count is zero. The signed mean is near zero and within the standard error of 1,000 draws. The unsigned mean is near 12, the expected positive half of Gaussian noise at scale 30. Repeated releases do not remove this bias; an unsigned table of empty cells carries the same upward bias in every cell.

## Which noise to choose {#sec-util-choose}

Choose the mechanism based on the output and the guarantee you must report, as shown in @tbl-util-choice.

| Situation | Noise and budget | Why |
| --- | --- | --- |
| One count, sum or mean, and a guarantee in pure $\varepsilon$ | Laplace under `dp.loss_of(epsilon=...)` | The guarantee has no $\delta$ and is the easiest to state |
| A grouped table where a person touches many groups | Gaussian under `dp.loss_of(rho=..., delta=...)` with a `Bound` per group | The $L_2$ sensitivity grows with the root of the number of groups |
| Many queries composed into one release | Gaussian under zCDP | Budgets in $\rho$ add up tightly and convert once at the end |
| A release that others will analyse assuming normal errors | Gaussian | The noise is normal, so standard intervals apply |
| A guarantee that must be reported as $(\varepsilon, \delta)$ | Either; convert $\rho$ with `make_zCDP_to_approxDP` and `make_fix_delta` | OpenDP's conversion is tighter than the closed form |
| Counts that anyone might add up | Either, with `signed=True` | Clamping at zero biases totals upward |

: Which noise to choose, by what is released. {#tbl-util-choice}

Gaussian noise cannot satisfy pure DP, so policies requiring $\delta=0$ rule it out. The $L_2$ sensitivity gain also requires a known per-group contribution bound; that bound is part of the mechanism choice.

## The releases of one run {#sec-util-run}

The script releases counts under both budgets and a quarterly table under bounded zCDP. Its Laplace count repeats the [Essential statistics](03-essential-statistics.md#sec-stats-check) result with the same error bar.

```text
released (one run):
  laplace count 3,811,867  (true 3,811,932, within 539.7 with probability 0.95)
  gaussian count 3,811,994  (true 3,811,932, within 354.0 with probability 0.95)
  2005 Q1 count 87,442  (true 87,372, within 60)
  2005 Q2 count 86,457  (true 86,469, within 60)
  2005 Q3 count 83,536  (true 83,518, within 60)
  2005 Q4 count 84,746  (true 84,739, within 60)
  groups released 36 of 36; within 60 of the truth: 32
```

Both overall counts fall within their error bars. Of the 36 quarterly counts, 32 are within their bars; 2005 Q1 is one of the misses. At $\alpha=0.05$, about one cell in twenty falls outside its bar, and the missed cell is not known in advance. For private data, choose from scales and accuracies, which cost nothing to inspect, then release once.

## Summary

| Call or concept | What it does | Rule of thumb |
| --- | --- | --- |
| `summarize(alpha=0.05)` | The scale and the 95% accuracy of each statistic, before release | Read it before `release()`; it costs nothing |
| `laplacian_scale_to_accuracy`, `gaussian_scale_to_accuracy` | A continuous scale to a 95% error bar | For sums and means of floats |
| `discrete_laplacian_scale_to_accuracy`, `discrete_gaussian_scale_to_accuracy` | The same for integer noise | For counts, to match `summarize` |
| `accuracy_to_laplacian_scale` and the three others | A target error bar to the scale it needs | Divide the sensitivity by the scale to see the $\varepsilon$ it costs |
| The error bar | Covers the noise only | Clipping, imputation, truncation and resizing bias the input, and the bar does not see it |
| `dp.loss_of(epsilon=...)` | Pure DP; the context adds Laplace noise | Halving the bar doubles $\varepsilon$ |
| `dp.loss_of(rho=...)` | zCDP; the context adds Gaussian noise | Halving the bar quadruples $\rho$ |
| `dp.polars.Bound(by=[...], per_group=1)` | One record per person per group | Cuts the $L_2$ sensitivity to the root of the number of groups; does nothing for Laplace |
| `make_zCDP_to_approxDP` then `make_fix_delta` | $\rho$ to $(\varepsilon, \delta)$, tighter than the closed form | Apply to `context.accountant` for the whole release |
| `signed=True` | Keeps negative noisy counts | Always, if the counts may be summed; clamp for display afterwards |

: The calls and concepts of this chapter, what each does, and the rule of thumb that goes with it. {#tbl-util-summary}

## References

- OpenDP documentation: "Assessing Utility" in Getting Started, "Gaussian Noise" in the Polars user guide, and "Aggregation" in the Polars expressions guide for `signed`. The OpenDP Cookbook's "Signed Counts: Why Clipping at Zero Biases Totals" is the experiment the signed-counts section summarizes.
- Bun and Steinke, "Concentrated Differential Privacy: Simplifications, Extensions, and Lower Bounds" (2016): zCDP, the Gaussian mechanism under it, and the conversion to $(\varepsilon, \delta)$.
- Canonne, Kamath and Steinke, "The Discrete Gaussian for Differential Privacy" (2020): the integer Gaussian that the context adds to counts and its tail bounds.
- Kamath, Mouzakis, Regehr, Singhal and Steinke, "A Bias-Variance-Privacy Trilemma for Statistical Estimation" (2023): why the bias of clipping cannot be removed in general, cited by the documentation's note on accuracy.
- The full edition's chapters on [the Laplace and Gaussian mechanisms](../../book-full/part03-dp-foundations/03-laplace-gaussian.md), [concentrated DP](../../book-full/part04-privacy-accounting/03-concentrated-dp.md) and [conversions to (ε, δ)](../../book-full/part04-privacy-accounting/02-rdp-conversions.md) derive the results this chapter states.
