# Assessing utility and choosing the noise

Every query in the earlier chapters printed a scale and an accuracy before it was released. This chapter is about reading those two numbers and changing them. An **accuracy** is the half-width of an error bar: with a stated probability, the released value lies within that distance of the value the noise was added to. OpenDP converts a noise scale to an accuracy and back, for Laplace and for Gaussian noise. So you can work out what a target error bar costs before you spend anything.

The second half of the chapter is about which noise to add. Under pure differential privacy (DP) the context adds Laplace noise, and under zero-concentrated differential privacy (zCDP) it adds Gaussian noise. Switching between them is one argument of `dp.loss_of`. The choice matters most when one person's records fall in many groups. Gaussian noise is then calibrated to a sensitivity that grows with the square root of the number of groups, rather than with the number itself. On the survey, telling the library that a person answers at most once per quarter cuts the noise on a quarterly count six-fold, but only under zCDP. By the end you will be able to say what an error bar covers and what it leaves out. You will also convert a budget in $\rho$ back to an $\varepsilon$, and choose the noise for a table.

The script [opendp_utility_and_noise.py](../../code/part02-tabular-data/opendp_utility_and_noise.py) runs every step on OpenDP 0.16.0 against the Labour Force Survey (LFS) sample of the earlier chapters. The conversions, scales, accuracies, thresholds and budgets it prints are the same on every run. The released counts and the means of a small clamping experiment are not, because OpenDP's noise cannot be seeded, so this page shows them in blocks marked as one run.

```bash
uv run python code/part02-tabular-data/opendp_utility_and_noise.py
```

## What an accuracy statement means {#sec-util-accuracy}

The documentation gives a template for reading the accuracy functions, and it is worth learning by heart. When the noise has a given scale, the released value differs from the true value by no more than the accuracy at a significance level $\alpha$, or with confidence $(1 - \alpha) \times 100\%$. Every accuracy on a page of this book has $\alpha = 0.05$, so the error bar holds with probability 0.95, and about one release in twenty lands outside it. A smaller $\alpha$ buys a wider bar, which [Essential statistics](03-essential-statistics.md) showed by summarizing the same count at $\alpha = 0.01$.

The module `opendp.accuracy` has eight conversion functions, and the prelude exposes them under `dp`. Four convert a scale to an accuracy and back for continuous noise, `laplacian_scale_to_accuracy`, `accuracy_to_laplacian_scale`, `gaussian_scale_to_accuracy` and `accuracy_to_gaussian_scale`. The other four do the same for the discrete distributions, and their names carry `discrete_`. The documentation's example asks for the Gaussian scale at which a release differs from its input by at most 2 with 95% confidence.

```python
import opendp.prelude as dp

print(dp.accuracy_to_gaussian_scale(accuracy=2.0, alpha=0.05))  # 1.020426913849308
print(dp.gaussian_scale_to_accuracy(scale=180.0, alpha=0.05))  # 352.7935172172097
print(dp.discrete_gaussian_scale_to_accuracy(scale=180.0, alpha=0.05))  # 354.0
print(dp.laplacian_scale_to_accuracy(scale=180.0, alpha=0.05))  # 539.2318092397184
print(dp.discrete_laplacian_scale_to_accuracy(scale=180.0, alpha=0.05))  # 539.7311147961669
```

The continuous functions are closed forms. For Laplace noise the accuracy is the scale times $\ln(1/\alpha)$. For Gaussian noise it is the scale times the two-sided normal quantile, which is about 1.96 at the usual level. The discrete functions sum the probabilities of the integer noise instead, which is why a discrete Gaussian accuracy is a whole number. Use the discrete ones for counts, because OpenDP adds integer noise to an integer, and the continuous ones for sums and means of floats. The difference is small at a scale of 180, as the snippet shows. At a scale near 1 the discrete bars are noticeably wider, but `summarize` always picks the right one for you.

The error bar is a statement about the noise and nothing else. The documentation says so in a note, and the point deserves a paragraph here, because it is the mistake a reader of an error bar makes most often. The accuracy functions describe how far the release can fall from the input to the noise mechanism. That input is the count, sum or mean of the data after every preparation step has run. Clipping values into public bounds, imputing a constant for missing values, truncating a person's rows and resizing the frame all change that input. The error bar says nothing about the gap between the input and the quantity you had in mind.

Clipping a sum pulls it toward the bounds, and truncation drops rows. Both are biases that no number of repeated releases averages away. The practitioner edition's [release workflow](../../book-practitioner/part03-dp-foundations/12-release-statistics.md) measures such a gap on another dataset, where truncation and clipping keep about half of the revenue. Here the quantity is a count of rows with no preparation, so the bar is the whole story.

## A target error bar on the survey {#sec-util-target}

Suppose you want the count of survey responses within 100 of the truth with 95% confidence. The count under the tutorial's pure-DP budget of $\varepsilon = 1$, split evenly over five queries, gives you a place to start. The excerpt summarizes it and then asks the discrete Laplace conversion for the scale the target needs.

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

The scale of 180 comes from the privacy unit and the budget. A person contributes at most 36 rows, so adding or removing a person changes the count by at most 36. That is the sensitivity, and the Laplace scale is the sensitivity over the query's share of $\varepsilon$, which is 0.2. The error bar of about 540 is 0.014% of the true count, which is small for a headline figure. The target of 100 needs a scale of 33.2. At sensitivity 36, that scale costs $\varepsilon = 1.08$ on this one query alone, which is more than the whole budget. The honest conclusion is that a bar of 100 is not available at this budget, not that the budget should grow to meet it.

The conversion also tells you the exchange rate. Because the Laplace scale is the sensitivity over $\varepsilon$, and the accuracy is proportional to the scale, halving the error bar means doubling the budget. The script checks this by rebuilding the context at $\varepsilon = 2$, where the same query gets a scale of 90 and about half the error bar.

```text
laplace count at epsilon 2: Integer Laplace scale=90 95% accuracy=270.115
```

The bar is not exactly half of 539.7, because the discrete conversion adds a constant for the rounding of integer noise, but the rule of thumb holds. [The full edition's chapter on the Laplace mechanism](../../book-full/part03-dp-foundations/03-laplace-gaussian.md) derives both the scale and the tail bound behind the conversion.

## Laplace under pure DP, Gaussian under zCDP {#sec-util-gaussian}

OpenDP chooses the noise distribution from the kind of budget you give it. If the budget is `dp.loss_of(epsilon=...)`, the guarantee is pure DP and the context adds Laplace noise. If it is `dp.loss_of(rho=...)`, the guarantee is zCDP and the context adds Gaussian noise. Nothing else in a query changes, as long as the query does not name a distribution itself. So the two context constructors of the script differ only in the budget line.

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

What does $\rho$ mean? Pure DP bounds the privacy loss, the log of how much more likely an output is with a person in the data than without, by $\varepsilon$ in the worst case. Zero-concentrated DP bounds it on average instead. A mechanism is $\rho$-zCDP when its privacy loss has mean at most $\rho$ and tails no heavier than those of a normal variable with variance $2\rho$. The Gaussian mechanism fits this definition exactly, which is why the two go together, and budgets in $\rho$ add up across queries just as budgets in $\varepsilon$ do. [The full edition's chapter on concentrated DP](../../book-full/part04-privacy-accounting/03-concentrated-dp.md) defines it through Rényi divergences and derives all of this. The documentation's Gaussian page repeats the count under $\rho = 0.1$, split over the same five queries.

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

The scale is again 180, and that is no accident. Write $\sigma$ for the standard deviation of the Gaussian noise, which is what OpenDP calls its scale, and $\Delta$ for the sensitivity of the quantity it is added to. Then the mechanism is $\Delta^2 / (2\sigma^2)$-zCDP, so the context solves for the scale and finds the sensitivity over $\sqrt{2\rho}$. With a share of $\rho = 0.02$, that root is 0.2, which is exactly the Laplace share, and the documentation chose $\rho = 0.1$ so that its two pages compare at equal scale. At equal scale the Gaussian bar is narrower, 354 against 540, because Gaussian tails fall faster than Laplace tails. Equal scale is not equal privacy, though. @sec-util-epsilon converts this share of $\rho$ to an $\varepsilon$ of about 1 at a small $\delta$. That is five times the Laplace share, so the two pages compare different guarantees.

The exchange rate also differs. Because the Gaussian scale is the sensitivity over $\sqrt{2\rho}$, halving the error bar means quadrupling $\rho$, not doubling it. The script confirms this at $\rho = 0.4$.

```text
gaussian count at rho 0.4: Integer Gaussian scale=90 95% accuracy=177
```

So on a single count, Gaussian noise under zCDP is no bargain. Its advantage appears when a query answers many things at once, which is the subject of the next section.

## One person in many groups: L1 against L2 {#sec-util-sensitivity}

A grouped count returns a vector, one number per group, and the noise is calibrated to how far one person can move the whole vector. Laplace noise measures that distance in the $L_1$ norm, which adds up the changes across groups. Gaussian noise measures it in the $L_2$ norm, which adds up their squares and takes the root. The documentation puts it in one picture. Walking between two corners of a unit square takes 2 along the streets and $\sqrt 2$ as the crow flies. In $d$ dimensions the gap grows to $d$ against $\sqrt d$.

The survey shows the gap at its most useful. A person answers at most once per quarter, so their 36 records fall in 36 different year-and-quarter groups, each changed by one. The $L_1$ distance between the two count vectors is 36, but the $L_2$ distance is $\sqrt{36} = 6$. Without being told that, the library must assume the worst case, in which all 36 records land in one group, and then both distances are 36. [Identifier truncation and bounds](05-identifier-truncation.md) introduced `dp.polars.Bound` for exactly this statement, and the excerpt uses it.

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

The budget now carries a $\delta$, because the query groups by keys the library has not been given. As [Grouping](04-grouping.md) explained, the context then releases only the groups whose noisy count clears a threshold. The small probability $\delta$ pays for the chance that a group of one person slips through. The script summarizes the quarterly count three ways. The first is under zCDP with the plain bound of 36 contributions, the second adds the per-group bound, and the third uses the per-group bound under pure DP.

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

The per-group bound cuts the Gaussian scale from 180 to 30, the factor of six that the arithmetic predicted. The threshold falls with it, from 1,133 to 184. Under pure DP the same bound changes nothing, because the $L_1$ distance is 36 either way, and the Laplace scale stays at 180. The thresholds are far below the smallest quarterly count of 82,504, so every group is released and the error bar of 60 is what the table buys. [The full edition's chapter on the two mechanisms](../../book-full/part03-dp-foundations/03-laplace-gaussian.md) proves why Gaussian noise scales with the $L_2$ sensitivity and compares the two in many dimensions.

The lesson generalizes beyond this survey. Whenever a person's records are spread thinly over many groups, say one purchase per day, the $L_2$ sensitivity is far below the $L_1$ one. Then Gaussian noise under zCDP is the right tool. The gain depends on telling the library about the spread with a bound per group. A bound you cannot justify from public knowledge must be enforced by truncation first, as the identifier truncation chapter showed.

## Stating the guarantee in epsilon {#sec-util-epsilon}

A budget in $\rho$ is convenient for accounting but unfamiliar to most readers of a release. OpenDP converts it to the usual pair $(\varepsilon, \delta)$ with two combinators, which [Combinators](../part03-framework/11-combinators.md) covers in general. The first, `make_zCDP_to_approxDP`, turns a measurement's zCDP guarantee into a curve of $\varepsilon$ against $\delta$. The second, `make_fix_delta`, picks a $\delta$ on that curve. The script applies them to a Gaussian measurement built by hand, whose scale is set so that a sensitivity of 1 costs exactly $\rho$.

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

Take $\delta = 10^{-7}$. The whole budget of $\rho = 0.1$ is then worth $\varepsilon = 2.35$. One query's share of $\rho = 0.02$ is worth about $\varepsilon = 1$ at the same $\delta$. The textbook conversion is
$$\varepsilon = \rho + 2\sqrt{\rho \ln(1/\delta)},$$
which gives 2.64 here, and OpenDP's smaller number is not a mistake. The library uses a tighter conversion that optimizes over the orders of the Rényi divergence, which [the full edition's chapter on conversions](../../book-full/part04-privacy-accounting/02-rdp-conversions.md) derives. You can also convert a whole context rather than one measurement. Its `accountant` attribute is the measurement that composes every query. The same two combinators applied to it give the same 2.35, which is the number to write in the release's statement.

The $\delta$ here is your choice. It should be far below one over the number of people, because a $\delta$ of that size would allow a mechanism that publishes one person's record outright. With 1.7 million people, $10^{-7}$ is a defensible value. This $\delta$ is a conversion cost, separate from the $\delta$ that paid for the threshold in the grouped query. A release that needs both should budget them both.

A last detail is which accuracy the zCDP context reports. The Gaussian count's summary said 354, and the continuous conversion at the same scale says 352.79. The context adds a discrete Gaussian to an integer count, so it reports the discrete accuracy, which [Measurements](../part03-framework/10-measurements.md) explains when it builds the integer mechanisms. The difference is a rounding. But if you recompute an accuracy by hand from a scale, use the discrete function for counts to match what `summarize` prints.

## Signed and unsigned counts {#sec-util-signed}

Every count on this page passed `signed=True`, and the reason is bias. A noisy count can come out negative when the true count is small, and OpenDP's older behaviour was to raise such a value to zero. Version 0.16.0 adds the `signed` argument. Pass `signed=True`, and the count is cast to a signed integer before the noise, so a negative result is kept. Pass `signed=False`, and the result is clamped at zero. Leaving the argument out still clamps and prints a warning, because the default is about to change.

Clamping looks harmless for one count, and for a single small cell it even reduces the error. The trouble starts when counts are added. A clamped count can be too high but never too low, so each small cell is biased upward, and a total over hundreds of cells piles those biases up. The OpenDP cookbook demonstrates this on a table of 400 cells, most of them nearly empty. There the clamped total overshoots on every repetition, while the signed total is right on average.

A signed release can be clamped afterwards for display, because clamping a released value is post-processing and costs nothing. The rule is to release signed counts whenever anyone might sum, average or model them, and to tidy up the presentation last. The script makes the point in miniature. It counts an empty frame a thousand times at a fixed Gaussian scale of 30 through the framework API, which bypasses a context and so suits toy data only.

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

The true count is zero. The signed mean sits near zero, within the standard error of 0.95 that a thousand draws at scale 30 leave. The unsigned mean sits near 12, which is the mean of the positive half of a Gaussian of scale 30, and no number of repetitions brings it down. A table of empty and near-empty cells released unsigned carries that 12 in every cell.

## Which noise to choose {#sec-util-choose}

The choice comes down to what you release and to whom you must explain it. @tbl-util-choice gathers the cases of this chapter.

| Situation | Noise and budget | Why |
| --- | --- | --- |
| One count, sum or mean, and a guarantee in pure $\varepsilon$ | Laplace under `dp.loss_of(epsilon=...)` | The guarantee has no $\delta$ and is the easiest to state |
| A grouped table where a person touches many groups | Gaussian under `dp.loss_of(rho=..., delta=...)` with a `Bound` per group | The $L_2$ sensitivity grows with the root of the number of groups |
| Many queries composed into one release | Gaussian under zCDP | Budgets in $\rho$ add up tightly and convert once at the end |
| A release that others will analyse assuming normal errors | Gaussian | The noise is normal, so standard intervals apply |
| A guarantee that must be reported as $(\varepsilon, \delta)$ | Either; convert $\rho$ with `make_zCDP_to_approxDP` and `make_fix_delta` | OpenDP's conversion is tighter than the closed form |
| Counts that anyone might add up | Either, with `signed=True` | Clamping at zero biases totals upward |

: Which noise to choose, by what is released. {#tbl-util-choice}

Two cautions go with the table. Gaussian noise cannot satisfy pure DP, so a policy that demands $\delta = 0$ rules it out whatever the sensitivity. And the gain from the $L_2$ sensitivity exists only when the library knows how a person's records spread across groups. So the per-group bound is part of the choice, not an optional extra.

## The releases of one run {#sec-util-run}

The script ends by releasing the count under both budgets and the quarterly table under the bounded zCDP context, and reads each against its error bar.

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

Both counts landed well inside their bars, as they should about nineteen times in twenty. Of the 36 quarterly counts, 32 landed within their bar. The first quarter of 2005 is one of the misses. That is what an error bar at $\alpha = 0.05$ means across a table: about one cell in twenty falls outside it, and you cannot tell which. Comparing with the truth is possible here only because the survey is public. On your own data you make these choices on the scales and accuracies, which cost nothing to read, and release once.

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
