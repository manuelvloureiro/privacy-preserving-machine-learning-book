# Private mean estimation

## Why this matters

The mean is the statistic that private machine learning computes most often. Each step of private stochastic gradient descent averages per-example gradients, each clipped to a maximum length first, and even a survey share is a mean of 0/1 answers. The planning questions are how many records a private estimate needs, and how to choose a clipping range without looking at the data.

Differential privacy (DP) must hold for every dataset, including datasets your assumptions say cannot happen ([The definition of differential privacy](01-definition.md)). So an assumption about the data, such as "the records are normal", may make a mechanism more accurate, but it must never enter the privacy argument. Every mechanism below clips each record to a range, which caps one record's influence whatever the data are, and its noise grows with the width of that range. A narrow range is cheap, but it must not be read off the data. So spend a small part of the budget finding where the data are, then clip there and spend the rest on the mean.

## Bounded data: privacy adds a lower-order term {#sec-mean-bounded}

Throughout the chapter, datasets have a fixed size $n$, and neighbours differ in one replaced record. An estimate is judged by its **root-mean-square error** (RMSE), the typical distance between the estimate and the mean of the population the records came from.

Start with records in a known interval, such as yes/no answers coded 0 and 1. Write $w$ for the width of the interval and $\sigma^2$ for the variance of a record. Release the average plus Laplace noise with scale $w/(n\varepsilon)$, which is $\varepsilon$-DP because one record moves the average by at most $w/n$ ([The Laplace and Gaussian mechanisms](03-laplace-gaussian.md)). Laplace noise has variance twice its scale squared, and it is independent of the sampling error, so the two variances add:
$$\mathrm{RMSE} = \sqrt{\frac{\sigma^2}{n} + \frac{2w^2}{n^2\varepsilon^2}}.$$
To plan a study, set the RMSE equal to a target error $\alpha$ and solve for $n$. The snippet in the Code section does this exactly, and the number of records needed is at most
$$\frac{\sigma^2}{\alpha^2} + \frac{\sqrt2\,w}{\alpha\varepsilon}.$$

So sampling needs about $\sigma^2/\alpha^2$ records whatever you do, and privacy adds at most about $w/(\alpha\varepsilon)$ more. For a yes/no survey of a population that is half yes, the worst case for sampling, an RMSE of 0.05 needs 100 people. At $\varepsilon = 1$, privacy raises that to 108. Its term shrinks only like $1/\alpha$, against $1/\alpha^2$ for sampling, so it matters most at a coarse target and a small $\varepsilon$. It also grows with the width $w$, so the rest of the chapter is about narrowing the range.

Libraries make you state the range. In diffprivlib, `tools.mean(x, epsilon=..., bounds=(lower, upper))` clips to the bounds and adds noise. Without `bounds`, it computes them from the data and raises a `PrivacyLeakWarning`, because a range read off the observed minimum and maximum leaks them. OpenDP likewise needs clamping bounds, its word for clipping bounds, before it builds a mean, and [Release statistics with a library](12-release-statistics.md) does the whole job with it.

Bounds that are too tight fail in another way, because they clip real records. The script [practitioner_adult_statistics.py](../../code/part03-dp-foundations/practitioner_adult_statistics.py) takes the hours worked per week by the 48,842 people of the Adult census data. It releases the mean at $\varepsilon = 1$ twice, once with loose bounds and once with tight ones.

```{.python excerpt="adult-mean-bounds"}
hours = adult["hours-per-week"].to_numpy()
loose = tools.mean(hours, epsilon=1.0, bounds=(1, 99), random_state=seed)
tight = tools.mean(hours, epsilon=1.0, bounds=(20, 60), random_state=seed)
clipped = np.clip(hours, 20, 60).mean()  # what the tight bounds can at best recover
print(f"true mean hours: {hours.mean():.3f}")
print(f"bounds (1, 99): {loose:.3f}  bounds (20, 60): {tight:.3f}")
print(f"bias of (20, 60): {clipped - hours.mean():+.3f}")
```

Its seed-0 run, `uv run python code/part03-dp-foundations/practitioner_adult_statistics.py`, prints these lines, the last two from code next to the excerpt.

```text
true mean hours: 40.422
bounds (1, 99): 40.421  bounds (20, 60): 40.361
bias of (20, 60): -0.061
people outside (20, 60): 8.7%
noise scale (1, 99): 0.0020  (20, 60): 0.0008
```

The tight bounds shrink the noise scale from 0.0020 to 0.0008 hours. But they clip the 8.7% of people outside them, which moves the mean by 0.061 hours, far more than the noise they save. No budget removes that bias, so report the error in three parts, which are sampling, clipping bias and noise. Choose the bounds from the schema or public data, and check the share they clip on data you may look at.

The formulas also treat $n$ as public, which holds when neighbours replace a record, since they then have the same size. If the size is itself private, divide a noisy sum by a noisy count, with the budget split between the two.

## The cost of an unknown range {#sec-mean-coarse}

Now let the records be normal with variance 1, and suppose you know only that their mean lies within some distance $R$ of zero. Normal records rarely stray far from their mean, so write $\tau$ for a distance from the mean that holds them all, except with a small failure probability $\beta$. For 1,000 records at $\beta = 0.05$, it is 4.604.

The **naive estimator** clips every record to the widest place the data could be, $R + \tau$ on either side of zero, and adds Laplace noise for that width. It is private, but its error grows in step with $R$. With 1,000 records at $\varepsilon = 1$, its RMSE reaches 2828.44 when $R$ is a million. Yet [Lower bounds](10-lower-bounds.md) shows that an unknown range need cost only about $\ln R/\varepsilon$ extra records.

**Coarse then fine** (Karwa & Vadhan 2018) comes close to that floor in two steps. The coarse step cuts the range into bins of width $2\tau$ and picks the fullest bin with the [exponential mechanism](05-selection.md), spending a part $\varepsilon_1$ of the budget. It succeeds with a number of records that grows only with the logarithm of the number of bins. The fine step clips every record to the chosen bin and its two neighbours, because the data can straddle two bins. It releases their average with Laplace noise for that window, spending the rest of the budget. By [composition](02-properties.md), the two steps together are $\varepsilon$-DP.

With a quarter of the budget on the coarse step, the full edition's 1,000 records give an RMSE of 0.0609 for every $R$ from 10 to a million. But the fine step gets only three quarters of the budget and a window three bins wide, so for small ranges the naive estimator wins, here up to $R = 13.81$. Beyond it, use coarse then fine, or locate the data with a private quantile such as diffprivlib's `tools.quantile`. Under approximate DP, with its small slack $\delta$, a stability-based histogram does the coarse step with about $\ln(1/\delta)/\varepsilon$ records whatever the number of bins, so $R$ drops out ([Beyond global sensitivity](07-sensitivity.md)). Still, a budget split, bin width or $\beta$ chosen after looking at the data leaks it. And a coarse step with too few records can pick an empty bin and clip everything to the wrong place.

## Many dimensions: clip the norm {#sec-mean-multi}

Vectors in $d$ dimensions, such as embeddings or gradients, follow the same recipe with a different measure of size. Clip each record's $\ell_2$ norm, its length, to a radius $C$, so that one replaced record moves the average by at most $2C/n$. Then add Gaussian noise calibrated to that sensitivity ([The Laplace and Gaussian mechanisms](03-laplace-gaussian.md)). This is exactly the step of private gradient descent, and [private training](../part05-dp-ml/02-dp-sgd.md) gives the rules for choosing the clipping norm there. With unit variance per coordinate, records lie about $\sqrt d$ from their mean, so the radius must grow like $\sqrt d$. Noise at that scale in every coordinate then grows like $d$ in total, against $\sqrt d$ for the sampling error. So in high dimension privacy is the larger cost, and the [fingerprinting lower bound](10-lower-bounds.md) shows that no mechanism avoids it.

When you do not know where the mean is, the range problem returns, and a grid of bins would pay for the range once per dimension. The **shrinking balls** of CoinPress (Biswas, Dong, Kamath & Ullman 2020) avoid the grid. Over a few rounds, they clip to a ball around the current guess, release a noisy average and shrink the ball. The full edition runs it on 100,000 records in a thousand dimensions at $\varepsilon = 1$, with the default $\delta = 10^{-6}$. The whole vector's RMSE is 0.1736, against 0.1000 without privacy. Fix the number of rounds and the budget split in advance, because the rounds compose and too many of them eat the budget.

## Heavy tails: clip into the data {#sec-mean-heavy}

Incomes, session lengths and per-example gradient norms have heavy tails. When all you know is that the variance is at most $\sigma^2$, a range wide enough to touch no record is enormous. So clip into the data on purpose, and accept a bias.

Reuse $\tau$ for the clipping half-width, and clip every record to within $\tau$ of a centre near the mean, found by a coarse step. Only the rare records beyond $\tau$ move, so with the centre at the mean the bias is at most $\sigma^2/\tau$, while the noise grows in proportion to $\tau$. The bound on the total error is smallest at
$$\tau_b = \sigma\sqrt{\frac{n\varepsilon}{2\sqrt2}},$$
which uses only $n$, $\varepsilon$ and the variance bound. There, an error $\alpha$ needs about $\sigma^2/\alpha^2 + 8\sqrt2\,\sigma^2/(\alpha^2\varepsilon)$ records. So privacy is no longer a lower-order cost, since both terms grow like $1/\alpha^2$. No $\varepsilon$-DP mechanism does better for every distribution with that variance, though a bound on a higher moment lowers the cost.

@tbl-mean-heavy shows the trade-off between bias and noise on Pareto records, a heavy-tailed distribution scaled to variance 1. Its first width clips no record except with probability 0.05. The worst-case width $\tau_b$ cuts the error to 0.0607, and the best width for this distribution does better still.

| Clipping half-width $\tau$ | Bias | Sampling sd | Noise sd | RMSE |
| --- | --- | --- | --- | --- |
| 141.42 (no record clipped) | 0.0000 | 0.0313 | 0.4000 | 0.4012 |
| 18.80 (worst-case best, $\tau_b$) | −0.0018 | 0.0293 | 0.0532 | 0.0607 |
| 5.36 (best for this distribution) | −0.0153 | 0.0249 | 0.0152 | 0.0329 |

: Clipped private means of Pareto records with variance 1, 1,000 records, $\varepsilon = 1$, for three clipping half-widths. {#tbl-mean-heavy}

So treat $\tau_b$ as a safe default, not as the best width. A narrower range usually does better, but choose it without looking at the records, from domain knowledge or with a private quantile ([Beyond global sensitivity](07-sensitivity.md)).

## Code

The snippet computes the records a bounded mean needs and the default clipping width for heavy tails. Setting the RMSE formula equal to $\alpha$ and multiplying by $n^2$ gives a quadratic in $n$, whose positive root the first function rounds up.

```python
from math import ceil, sqrt


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

- Never clip to the observed minimum and maximum. Take bounds from the schema or public data, or spend a quarter of the budget or less, a simple default, finding the data.
- Bounds that clip real records bias the mean whatever $\varepsilon$ is, so put the clipping bias into every reported interval.

## References

- Karwa & Vadhan, "Finite Sample Differentially Private Confidence Intervals" (ITCS 2018): coarse then fine for normal means and variances, under pure and approximate DP.
- Biswas, Dong, Kamath & Ullman, "CoinPress: Practical Private Mean and Covariance Estimation" (NeurIPS 2020): shrinking balls in practice, with code.

**Derivations.** Every result here is derived step by step in [the full edition's chapter](../../book-full/part03-dp-foundations/11-mean-estimation.md).
