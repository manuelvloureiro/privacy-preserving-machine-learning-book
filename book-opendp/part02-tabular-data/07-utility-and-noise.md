# Assessing utility and choosing the noise

Every query in the earlier chapters printed a scale and an accuracy before its release. This chapter is about how to read those two numbers and change them. An **accuracy** is the half-width of an error bar: with a stated probability, the released value is within that distance of the value that the noise was added to. OpenDP converts a noise scale to an accuracy and back, so you can find what a target error bar costs before you spend anything.

The second half of the chapter is about which noise to add. Under pure differential privacy (DP) the context adds Laplace noise, and under zero-concentrated differential privacy (zCDP) it adds Gaussian noise. One argument of `dp.loss_of` selects between them. The choice matters most when one person's records fall in many groups. Then Gaussian noise is calibrated to a sensitivity that grows with the square root of the number of groups, not with the number itself. On the survey, this cuts the noise on a quarterly count six-fold, but only under zCDP.

The script [opendp_utility_and_noise.py](../../code/part02-tabular-data/opendp_utility_and_noise.py) runs every step on OpenDP 0.16.0 against the Labour Force Survey (LFS) sample of the earlier chapters. The conversions, scales, accuracies, thresholds and budgets that it prints are the same on every run. The released counts and the means of a small clamping experiment are not, because OpenDP's noise cannot be seeded, so this page shows them in blocks marked as one run.

```bash
uv run python code/part02-tabular-data/opendp_utility_and_noise.py
```

## What an accuracy statement means {#sec-util-accuracy}

The documentation gives a template to read the accuracy functions. At a given scale, the released value is within the accuracy of the true value at a significance level $\alpha$, that is, with confidence $(1 - \alpha) \times 100\%$. Every accuracy in this book has $\alpha = 0.05$, so about one release in twenty is outside its error bar. A smaller $\alpha$ gives a wider bar, as [Essential statistics](03-essential-statistics.md) showed at $\alpha = 0.01$.

The module `opendp.accuracy` has eight conversion functions, which the prelude gives under `dp`. Four convert a Laplace or Gaussian scale to an accuracy and back, and the other four, with `discrete_` in their names, do the same for integer noise. The documentation's example asks for the Gaussian scale at which a release is within 2 of its input with 95% confidence.

```python
import opendp.prelude as dp

print(dp.accuracy_to_gaussian_scale(accuracy=2.0, alpha=0.05))  # 1.020426913849308
print(dp.gaussian_scale_to_accuracy(scale=180.0, alpha=0.05))  # 352.7935172172097
print(dp.discrete_gaussian_scale_to_accuracy(scale=180.0, alpha=0.05))  # 354.0
print(dp.laplacian_scale_to_accuracy(scale=180.0, alpha=0.05))  # 539.2318092397184
print(dp.discrete_laplacian_scale_to_accuracy(scale=180.0, alpha=0.05))  # 539.7311147961669
```

The continuous functions are closed forms. For Laplace noise, the accuracy is the scale times $\ln(1/\alpha)$. For Gaussian noise, it is the scale times the two-sided normal quantile, about 1.96 at the usual level. The discrete functions sum the probabilities of the integer noise, so a discrete Gaussian accuracy is a whole number. Use the discrete ones for counts, because OpenDP adds integer noise to an integer, and the continuous ones for sums and means of floats. At a scale of 180 the difference is small, and `summarize` always uses the correct one.

The error bar is about the noise and nothing else, and this is the most frequent mistake in how people read one. The accuracy functions say how far the release can be from the input to the noise mechanism: the statistic of the data after every preparation step. A clip into public bounds, an imputed constant, a truncation of a person's rows and a resize all change that input. The error bar says nothing about the gap between the input and the quantity that you want. Those gaps are biases, which no number of repeated releases removes. The practitioner edition's [release workflow](../../book-practitioner/part03-dp-foundations/12-release-statistics.md) measures one, where truncation and clipping keep about half of the revenue.

## A target error bar on the survey {#sec-util-target}

Suppose you want the count of survey responses within 100 of the truth with 95% confidence. Start from the count that [Essential statistics](03-essential-statistics.md#sec-stats-count) released, with $\varepsilon = 1$ split evenly over five queries. The excerpt summarizes it again, and then asks the discrete Laplace conversion for the scale that the target needs.

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

The scale of 180 is the sensitivity over the share of 0.2, with a sensitivity of 36 from [the privacy unit](../part01-getting-started/02-the-data-and-the-context.md#sec-ctx-context). The target needs a scale of 33.2, which costs more than the whole budget on this one query. So a bar of 100 is not available at this budget. But a bar of 0.014% of the true count is already small for a headline figure.

The conversion also gives the exchange rate. The Laplace scale is the sensitivity over $\varepsilon$, and the accuracy is proportional to the scale, so half the error bar costs twice the budget. At $\varepsilon = 2$ the same query gets half the scale.

```text
laplace count at epsilon 2: Integer Laplace scale=90 95% accuracy=270.115
```

The bar is not exactly half, because the discrete conversion rounds the integer noise, but the rule of thumb holds. [The full edition's chapter on the Laplace mechanism](../../book-full/part03-dp-foundations/03-laplace-gaussian.md) derives the scale and the tail bound behind the conversion.

## Laplace under pure DP, Gaussian under zCDP {#sec-util-gaussian}

OpenDP chooses the noise from the kind of budget. With `dp.loss_of(epsilon=...)`, the guarantee is pure DP and the context adds Laplace noise. With `dp.loss_of(rho=...)`, the guarantee is zCDP and the context adds Gaussian noise. Nothing else in a query changes, unless the query names a distribution. So the script's two context functions differ only in the budget line.

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

The privacy loss is the log of how much more likely an output is with a person in the data than without. Pure DP bounds it by $\varepsilon$ in the worst case. Zero-concentrated DP bounds it on average. A mechanism is $\rho$-zCDP when its privacy loss has mean at most $\rho$ and tails no heavier than those of a normal variable with variance $2\rho$. The Gaussian mechanism fits this definition exactly, and budgets in $\rho$ add up across queries as budgets in $\varepsilon$ do. [The full edition's chapter on concentrated DP](../../book-full/part04-privacy-accounting/03-concentrated-dp.md) derives all of this. The documentation's Gaussian page repeats the count under $\rho = 0.1$, split over the same five queries.

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

The scale is again 180, on purpose. Write $\sigma$ for the standard deviation of the Gaussian noise, which OpenDP calls its scale, and $\Delta$ for the sensitivity. The mechanism is $\Delta^2 / (2\sigma^2)$-zCDP, so the context sets the scale to the sensitivity over $\sqrt{2\rho}$. With a share of $\rho = 0.02$, that root is 0.2, the Laplace share. At equal scale the Gaussian bar is narrower, because Gaussian tails fall faster. But equal scale is not equal privacy: @sec-util-epsilon converts this share of $\rho$ to an $\varepsilon$ of about 1, five times the Laplace share.

The exchange rate also differs. The Gaussian scale is the sensitivity over $\sqrt{2\rho}$, so half the error bar costs four times $\rho$, as the script confirms at $\rho = 0.4$.

```text
gaussian count at rho 0.4: Integer Gaussian scale=90 95% accuracy=177
```

So on a single count, Gaussian noise under zCDP gives no gain. Its advantage is on a query that answers many things at once.

## One person in many groups: L1 against L2 {#sec-util-sensitivity}

A grouped count returns a vector, one number per group, and the noise is calibrated to how far one person can move the whole vector. Laplace noise measures that distance in the $L_1$ norm, the sum of the changes across groups. Gaussian noise measures it in the $L_2$ norm, the root of the sum of their squares. Between opposite corners of a unit square, the $L_1$ distance is 2 and the $L_2$ distance is $\sqrt 2$. In $d$ dimensions they become $d$ and $\sqrt d$.

On the survey, a person answers at most once per quarter, so their 36 records fall in 36 different groups of year and quarter, each changed by one. The $L_1$ distance between the two count vectors is 36, but the $L_2$ distance is $\sqrt{36} = 6$. If you do not tell the library, it must assume that all 36 records are in one group, and then both distances are 36. The `dp.polars.Bound` of [Identifier truncation and bounds](05-identifier-truncation.md) tells it.

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

The budget now has a $\delta$, because the query has no public keys. So the context releases only the groups whose noisy count passes a threshold, as in [Grouping](04-grouping.md). The script summarizes the quarterly count three ways: under zCDP with 36 contributions only, with the bound per group, and with that bound under pure DP.

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

The bound per group cuts the Gaussian scale from 180 to 30, the factor of six above, and the threshold falls with it. Under pure DP the bound changes nothing, because the $L_1$ distance is 36 either way. The thresholds are far below the smallest quarterly count, so every group is released, with an error bar of 60. [The full edition's chapter on the two mechanisms](../../book-full/part03-dp-foundations/03-laplace-gaussian.md) proves why Gaussian noise scales with the $L_2$ sensitivity.

In general, when a person's records spread thinly over many groups, the $L_2$ sensitivity is far below the $L_1$ one. Then Gaussian noise under zCDP is the correct tool. But the gain needs a bound per group. If public knowledge cannot justify the bound, enforce it with a truncation first.

## Stating the guarantee in epsilon {#sec-util-epsilon}

A budget in $\rho$ is convenient for accounting, but most readers of a release do not know it. OpenDP converts it to the usual pair $(\varepsilon, \delta)$ with two combinators, which [Combinators](../part03-framework/11-combinators.md) covers. The first, `make_zCDP_to_approxDP`, turns a zCDP guarantee into a curve of $\varepsilon$ against $\delta$. The second, `make_fix_delta`, picks a $\delta$ on that curve. The script applies them to a Gaussian measurement whose scale makes a sensitivity of 1 cost exactly $\rho$.

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

At $\delta = 10^{-7}$, the whole budget of $\rho = 0.1$ is worth $\varepsilon = 2.35$, and one query's share is worth about $\varepsilon = 1$. The textbook conversion is
$$\varepsilon = \rho + 2\sqrt{\rho \ln(1/\delta)},$$
which gives 2.64 here. OpenDP's smaller number comes from a tighter conversion over the orders of the Rényi divergence, which [the full edition's chapter on conversions](../../book-full/part04-privacy-accounting/02-rdp-conversions.md) derives. You can also convert a whole context: its `accountant` attribute is the measurement that composes every query. The two combinators on it give the same 2.35, the number to write in the release's statement.

You choose this $\delta$. Keep it far below one over the number of people, because a $\delta$ of that size allows a mechanism that publishes one person's record. With 1.7 million people, $10^{-7}$ is defensible. This $\delta$ is a conversion cost, separate from the $\delta$ of the threshold in the grouped query, so a release that needs both must budget both.

The Gaussian count's summary said 354, while the continuous conversion at the same scale gives 352.79. The context adds a discrete Gaussian to an integer count, so it reports the discrete accuracy, as [Measurements](../part03-framework/10-measurements.md) explains. So to recompute an accuracy from a scale, use the discrete function for counts.

## Signed and unsigned counts {#sec-util-signed}

Every count on this page has `signed=True`, because of bias. A noisy count can be negative when the true count is small, and OpenDP used to raise such a value to zero. In version 0.16.0, `signed=True` casts the count to a signed integer before the noise, so a negative result stays. With `signed=False` or no argument, the result is clamped at zero, and without the argument the library warns that the default will change.

For a single small cell, clamping even reduces the error. The problem is a sum of counts. A clamped count can be too high but never too low, so each small cell is biased upwards. A total over hundreds of cells adds up those biases. The OpenDP cookbook shows this on a table of 400 mostly empty cells. The clamped total is too high on every repetition, while the signed total is correct on average.

You can clamp a signed release afterwards for display, because that is free post-processing. So release signed counts whenever anyone might sum, average or model them. The script shows the bias on a small scale. It counts an empty frame a thousand times at a fixed Gaussian scale of 30, through the framework API without a context, which suits toy data only.

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

The true count is zero. The signed mean is near zero, within the standard error of a thousand draws. The unsigned mean is near 12, the mean of the positive half of a Gaussian of scale 30, and no number of repetitions brings it down. A table of empty cells released unsigned has that 12 in every cell.

## Which noise to choose {#sec-util-choose}

The choice depends on what you release and to whom you must explain it, as @tbl-util-choice shows.

| Situation | Noise and budget | Why |
| --- | --- | --- |
| One count, sum or mean, and a guarantee in pure $\varepsilon$ | Laplace under `dp.loss_of(epsilon=...)` | The guarantee has no $\delta$ and is the easiest to state |
| A grouped table where a person touches many groups | Gaussian under `dp.loss_of(rho=..., delta=...)` with a `Bound` per group | The $L_2$ sensitivity grows with the root of the number of groups |
| Many queries composed into one release | Gaussian under zCDP | Budgets in $\rho$ add up tightly and convert once at the end |
| A release that others will analyse assuming normal errors | Gaussian | The noise is normal, so standard intervals apply |
| A guarantee that must be reported as $(\varepsilon, \delta)$ | Either; convert $\rho$ with `make_zCDP_to_approxDP` and `make_fix_delta` | OpenDP's conversion is tighter than the closed form |
| Counts that anyone might add up | Either, with `signed=True` | Clamping at zero biases totals upward |

: Which noise to choose, by what is released. {#tbl-util-choice}

Two cautions go with the table. Gaussian noise cannot satisfy pure DP, so a policy that demands $\delta = 0$ rules it out. And the gain from the $L_2$ sensitivity exists only when the library knows how a person's records spread across groups. So the bound per group is part of the choice.

## The releases of one run {#sec-util-run}

The script ends with the count under both budgets and the quarterly table under the bounded zCDP context. The Laplace count repeats the count of [Essential statistics](03-essential-statistics.md#sec-stats-check), with the same error bar.

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

Both counts are inside their bars. Of the 36 quarterly counts, 32 are within their bar, and the first quarter of 2005 is one of the misses. That is what an error bar at $\alpha = 0.05$ means across a table: about one cell in twenty is outside it, and you cannot tell which. On your own data, make these choices from the scales and accuracies, which cost nothing to read, and release once.

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
