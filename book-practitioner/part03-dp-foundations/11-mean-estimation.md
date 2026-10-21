# Private mean estimation

## Why this matters

The mean is the statistic that private machine learning computes most often. Every step of private gradient descent clips per-example gradients, averages them and adds noise, and even a survey share is a mean of 0/1 answers. But what an analyst wants is usually the mean of the **population** the records came from. So the planning questions are how many records a private estimate needs, and how to choose a clipping range without looking at the data.

Differential privacy (DP) must hold for every dataset, including datasets your assumptions say cannot happen ([The definition of differential privacy](01-definition.md)). So an assumption about the data, such as "the records are normal", may make a mechanism more accurate, but it must never enter the privacy argument. Every mechanism below clips the data to a range, which keeps it private whatever the data are. Its noise is proportional to the width of that range, and the recipe of the chapter follows from it. Spend a small part of the budget finding where the data are, then clip there and spend the rest on the mean.

## Bounded data: privacy adds a lower-order term {#sec-mean-bounded}

Throughout the chapter, datasets have a fixed size $n$, and neighbours differ in one replaced record. An estimate is judged by its **root-mean-square error** (RMSE), the typical distance between the estimate and the population mean.

Start with records in a known interval, such as yes/no answers coded 0 and 1. Write $w$ for the width of the interval and $\sigma^2$ for the variance of a record. Release the average plus Laplace noise with scale $w/(n\varepsilon)$, which is $\varepsilon$-DP because one record moves the average by at most $w/n$ ([The Laplace and Gaussian mechanisms](03-laplace-gaussian.md)). The RMSE then combines the sampling error and the noise:
$$\mathrm{RMSE} = \sqrt{\frac{\sigma^2}{n} + \frac{2w^2}{n^2\varepsilon^2}}.$$
To plan a study, set the RMSE equal to a target error $\alpha$ and solve for $n$. The snippet in the Code section does this exactly, and a simple upper bound is
$$n \le \frac{\sigma^2}{\alpha^2} + \frac{\sqrt2\,w}{\alpha\varepsilon}.$$

So sampling needs about $\sigma^2/\alpha^2$ records whatever you do, and privacy adds at most about $w/(\alpha\varepsilon)$ more. For a yes/no survey in the worst case for sampling, a population that is half yes, an RMSE of 0.05 at $\varepsilon = 1$ needs 108 people instead of 100.

But the privacy term grows like $1/\varepsilon$, so it dominates at strong privacy and a coarse target. It also grows with the width $w$, the most one record can move, which is why the rest of the chapter is about making the range narrow.

Report the error in its three parts, which are sampling, the bias that clipping causes and the noise. With a public range that holds every record the bias is zero. But the later sections clip into the data, and a confidence interval that ignores that bias is too narrow. The formulas also treat $n$ as public, which holds when neighbours replace a record. If the dataset size is itself private, release a noisy sum and a noisy count and divide, with the budget split between them.

Libraries follow this pattern. In diffprivlib, `tools.mean(x, epsilon=..., bounds=(lower, upper))` clips to the bounds and adds noise, and `tools.var` and `tools.quantile` do the same for variances and quantiles. If you leave out `bounds`, it computes them from the data and raises a `PrivacyLeakWarning`, because a range read off the observed minimum and maximum leaks them. Tumult Analytics' `QueryBuilder` takes the bounds as arguments of `average(column, low, high)`, and OpenDP likewise makes you declare the clamping bounds and the dataset size before it builds a mean.

## The cost of an unknown range {#sec-mean-coarse}

Now let the records be normal with variance 1, with a mean known only to lie between $-R$ and $R$. Normal records rarely stray far from their mean. For 1,000 records, all of them lie within a distance $\tau$ of 4.604, except with a small failure probability $\beta$ of 0.05.

The **naive estimator** clips every record to the widest place the data could be, $R + \tau$ on either side of zero, and adds Laplace noise for that width. It is private, but its error grows in step with $R$. With 1,000 records at $\varepsilon = 1$ and a range of a million, its RMSE of 2828.44 is useless. That is the price of a clumsy mechanism, because [Lower bounds](10-lower-bounds.md) shows that privacy forces a cost of only about $\ln R/\varepsilon$ records for the range.

**Coarse then fine** (Karwa & Vadhan 2018) comes close to that floor. In the coarse step, cut the range into bins of width $2\tau$, so that the records fill at most two bins. Pick one bin with the [exponential mechanism](05-selection.md), scoring each bin by the number of records in it, and spend part of the budget, $\varepsilon_1$, on it. The fine step then clips every record to the chosen bin and its two neighbours, and it releases their average with Laplace noise for that window.

The fine step spends the rest of the budget, so the two steps together are $\varepsilon$-DP by composition ([Post-processing, group privacy and composition](02-properties.md)). Whether the result is accurate depends on the coarse step. Write $m$ for the number of bins. Then the coarse step picks a full bin, except with probability $\beta$, once
$$n \ge \frac{4\ln(m/\beta)}{\varepsilon_1}.$$
After that, the fine step's error does not depend on $R$ at all. With a quarter of the budget on the coarse step, the full edition's 1,000 records give an RMSE of 0.0609 whether $R$ is 10 or a million.

So choose by what you know. When the range is already narrow, the naive estimator wins, and with 1,000 records at $\varepsilon = 1$ the crossover is at $R = 13.81$. Beyond it, use coarse then fine. A private quantile, such as diffprivlib's `tools.quantile`, can also do the locating. If you can accept approximate DP, with its small slack $\delta$, you can drop $R$ altogether. A stability-based histogram does the coarse step with about $\ln(1/\delta)/\varepsilon$ records whatever the range ([Beyond global sensitivity](07-sensitivity.md)).

Two things go wrong. A budget split, bin width or $\beta$ chosen after looking at the data leaks it. And a coarse step with too few records can pick an empty bin, so the fine step clips everything to the wrong place.

## Many dimensions: clip the norm {#sec-mean-multi}

Vectors in $d$ dimensions, such as embeddings or gradients, follow the same recipe with a different measure of size. Clip each record's $\ell_2$ norm, its length, to a radius $C$, so that one replaced record moves the average by at most $2C/n$. Then add Gaussian noise calibrated to that sensitivity ([The Laplace and Gaussian mechanisms](03-laplace-gaussian.md)). This is exactly the step of private gradient descent, which clips per-example gradients to a fixed norm ([private training](../part05-dp-ml/02-dp-sgd.md)). With the radius close to the spread of the data, an error $\alpha$ needs about
$$n \approx \frac{d}{\alpha^2} + \frac{d\sqrt{\ln(1/\delta)}}{\alpha\varepsilon}$$
records, and the [fingerprinting lower bound](10-lower-bounds.md) shows that the privacy term cannot be avoided.

When you do not know where the mean is, the range problem returns, and a grid of bins would cost records in proportion to $d$. The **shrinking balls** of CoinPress (Biswas, Dong, Kamath & Ullman 2020) avoid the grid. Over a few rounds, they clip to a ball around the current guess, release a noisy average and shrink the ball, for about $\sqrt d\ln R/\varepsilon$ extra records. On 100,000 records in a thousand dimensions, the full edition's script finds an RMSE of 0.1736 against 0.1000 without privacy, because the privacy noise grows like $d$ while sampling grows like $\sqrt d$. Fix the number of rounds and the budget split before seeing the data, because the rounds compose and too many of them eat the budget.

## Heavy tails: clip into the data {#sec-mean-heavy}

Incomes, session lengths and per-example gradient norms have heavy tails. Suppose all you know is that the variance is at most $\sigma^2$. Then a clipping range wide enough to touch no record is enormous. The way out is to clip into the data on purpose and accept a bias.

Clip every record to within a half-width $\tau$ of a centre near the mean, found by a coarse step. With the centre at the mean, the bias is at most $\sigma^2/\tau$, and the noise grows in proportion to $\tau$. The bound on the total error is smallest at
$$\tau_b = \sigma\sqrt{\frac{n\varepsilon}{2\sqrt2}},$$
which uses only $n$, $\varepsilon$ and the variance bound. At that width, an error $\alpha$ needs about
$$n \approx \frac{\sigma^2}{\alpha^2} + \frac{8\sqrt2\,\sigma^2}{\alpha^2\varepsilon}$$
records. So with heavy tails privacy is no longer a lower-order cost, because its term has the same power of $\alpha$ as sampling. No $\varepsilon$-DP mechanism does better for every such distribution. A bound on a higher moment lowers the cost.

@tbl-mean-heavy shows the trade-off between bias and noise on Pareto records, a heavy-tailed distribution scaled to variance 1. The worst-case width $\tau_b = 18.80$ cuts the error from 0.4012 to 0.0607, and the best width for this distribution does better still.

| Clipping half-width $\tau$ | Bias | Sampling sd | Noise sd | RMSE |
| --- | --- | --- | --- | --- |
| 141.42 (no record clipped) | 0.0000 | 0.0313 | 0.4000 | 0.4012 |
| 18.80 (worst-case best, $\tau_b$) | −0.0018 | 0.0293 | 0.0532 | 0.0607 |
| 5.36 (best for this distribution) | −0.0153 | 0.0249 | 0.0152 | 0.0329 |

: Clipped private means of Pareto records with variance 1, 1,000 records, $\varepsilon = 1$, for three clipping half-widths. {#tbl-mean-heavy}

So treat $\tau_b$ as a safe default and not as the best width. A narrower range usually does better, but it must still be chosen without looking at the records, from domain knowledge or with a private quantile ([Beyond global sensitivity](07-sensitivity.md)). Setting the width to the observed maximum brings back the leak of a data-dependent noise scale.

## Code

The snippet computes the records a bounded mean needs and the default clipping width for heavy tails.

```python
from math import ceil, log, sqrt


def records_for_bounded_mean(alpha: float, sd: float, width: float, eps: float) -> int:
    """Smallest n with sd^2/n + 2 (width/(n eps))^2 <= alpha^2."""
    a, c = sd**2, 2 * (width / eps) ** 2
    return ceil((a + sqrt(a * a + 4 * c * alpha**2)) / (2 * alpha**2))


def heavy_tail_width(n: int, eps: float, sd: float) -> float:
    """The clipping half-width that balances the bias bound sd^2/tau against the noise."""
    return sd * sqrt(n * eps / (2 * sqrt(2)))


print(records_for_bounded_mean(0.05, 0.5, 1.0, 1.0))  # 108
print(f"{heavy_tail_width(1000, 1.0, 1.0):.2f}")  # 18.80
```

## Cheat sheet

| What you know | Mechanism | Records for error $\alpha$ (up to log factors) |
| --- | --- | --- |
| Bounded, width $w$, variance $\sigma^2$ | Mean plus $\mathrm{Lap}(w/(n\varepsilon))$ | $\sigma^2/\alpha^2 + \sqrt2\,w/(\alpha\varepsilon)$ |
| Normal, mean within $R$ of zero | Coarse histogram, then clip to three bins | $1/\alpha^2 + 1/(\alpha\varepsilon) + \ln R/\varepsilon$ |
| Same, approximate DP | Stability-based coarse step | $1/\alpha^2 + 1/(\alpha\varepsilon) + \ln(1/\delta)/\varepsilon$, no $R$ |
| $d$ dimensions, mean length at most $R$ | Clip the norm, Gaussian noise; shrinking balls to find the mean | $d/\alpha^2 + d\sqrt{\ln(1/\delta)}/(\alpha\varepsilon) + \sqrt d\ln R/\varepsilon$ |
| Variance at most $\sigma^2$ only | Clip to within $\tau_b$ of the centre | $\sigma^2/\alpha^2 + 8\sqrt2\,\sigma^2/(\alpha^2\varepsilon)$ |

: Private mean estimation at a glance, with $\tau_b = \sigma\sqrt{n\varepsilon/(2\sqrt2)}$. {#tbl-mean-cheat}

**Rules of thumb**

- Let assumptions about the data affect only accuracy. The clipping range keeps the release private, for every dataset.
- Never clip to a range set by your ignorance, or to the observed minimum and maximum. Spend a quarter of the budget or less finding the data first.
- Fix every parameter, from the budget split to the bin width, before you see the data.
- With heavy tails, clip into the data, with a width chosen from $n$, $\varepsilon$ and a variance bound, and put the clipping bias into every reported interval.
- If $n$ is not public, divide a noisy sum by a noisy count.

## Further reading

- Karwa & Vadhan, "Finite Sample Differentially Private Confidence Intervals" (ITCS 2018): coarse then fine for normal means and variances, under pure and approximate DP.
- Biswas, Dong, Kamath & Ullman, "CoinPress: Practical Private Mean and Covariance Estimation" (NeurIPS 2020): shrinking balls in practice, with code.

**Derivations.** Every result here is derived step by step in [the full edition's chapter](../../book-full/part03-dp-foundations/11-mean-estimation.md).
