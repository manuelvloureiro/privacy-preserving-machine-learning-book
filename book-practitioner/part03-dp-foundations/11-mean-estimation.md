# Private mean estimation

## Why this matters

Private machine learning computes the mean more often than any other statistic. Each step of private gradient descent averages clipped per-example gradients, and a survey share is a mean of 0/1 answers. Before you release a mean, you must know how many records it needs, and how to choose a clipping range without a look at the data.

Differential privacy (DP) must hold for every dataset, also for datasets that your assumptions exclude ([The definition of differential privacy](01-definition.md)). So an assumption such as "the records are normal" can make a mechanism more accurate, but never enters the privacy argument. For this reason, every mechanism below clips each record to a range. The noise grows with the width of that range, and you must not read a narrow range off the data. So spend a small part of the budget to find where the data are, then clip there.

## Bounded data: privacy adds a lower-order term {#sec-mean-bounded}

In this chapter, datasets have a fixed size $n$, and neighbours differ in one replaced record. The measure of an estimate is its **root-mean-square error** (RMSE), the typical distance from the estimate to the mean of the population.

Start with records in a known interval of width $w$, such as yes/no answers coded 0 and 1, and write $\sigma^2$ for the variance of a record. Release the average plus Laplace noise with scale $w/(n\varepsilon)$, which is $\varepsilon$-DP because one record moves the average by at most $w/n$ ([The Laplace and Gaussian mechanisms](03-laplace-gaussian.md)). Laplace noise has variance twice its scale squared, and it is independent of the sampling error, so the two variances add:
$$\mathrm{RMSE} = \sqrt{\frac{\sigma^2}{n} + \frac{2w^2}{n^2\varepsilon^2}}.$$
To plan a study, set the RMSE equal to a target error $\alpha$ and solve for $n$, as the snippet in the Code section does. The number of records that you need is at most
$$\frac{\sigma^2}{\alpha^2} + \frac{\sqrt2\,w}{\alpha\varepsilon}.$$

So sampling alone needs about $\sigma^2/\alpha^2$ records, and privacy adds at most about $w/(\alpha\varepsilon)$ more. Take a yes/no survey of a population that is half yes, the worst case for sampling. An RMSE of 0.05 then needs 100 people, and privacy at $\varepsilon = 1$ raises that to 108. The privacy term grows only like $1/\alpha$, against $1/\alpha^2$ for sampling, so it matters most at a coarse target and a small $\varepsilon$. It also grows with the width $w$, and the rest of the chapter makes the range narrower.

Libraries ask you for the range. In diffprivlib, `tools.mean(x, epsilon=..., bounds=(lower, upper))` clips to the bounds and adds noise. Without `bounds`, it takes the observed minimum and maximum, which leaks them, and raises a `PrivacyLeakWarning`.

Bounds that are too tight cause a different problem, because they clip real records. The script [practitioner_adult_statistics.py](../../code/part03-dp-foundations/practitioner_adult_statistics.py) releases the mean weekly hours of work of the 48,842 people in the Adult census data at $\varepsilon = 1$, with loose and with tight bounds.

```{.python excerpt="adult-mean-bounds"}
hours = adult["hours-per-week"].to_numpy()
loose = tools.mean(hours, epsilon=1.0, bounds=(1, 99), random_state=seed)
tight = tools.mean(hours, epsilon=1.0, bounds=(20, 60), random_state=seed)
clipped = np.clip(hours, 20, 60).mean()  # what the tight bounds can at best recover
print(f"true mean hours: {hours.mean():.3f}")
print(f"bounds (1, 99): {loose:.3f}  bounds (20, 60): {tight:.3f}")
print(f"bias of (20, 60): {clipped - hours.mean():+.3f}")
```

Its seed-0 run, `uv run python code/part03-dp-foundations/practitioner_adult_statistics.py`, prints these lines. The last two come from code next to the excerpt.

```text
true mean hours: 40.422
bounds (1, 99): 40.421  bounds (20, 60): 40.361
bias of (20, 60): -0.061
people outside (20, 60): 8.7%
noise scale (1, 99): 0.0020  (20, 60): 0.0008
```

The tight bounds decrease the noise scale from 0.0020 to 0.0008 hours. But they clip the 8.7% of people outside them, and this moves the mean by 0.061 hours, much more than the noise they save. No budget removes that bias, so report the error in three parts, which are sampling, clipping bias and noise. Choose the bounds from the schema or public data, and measure the share that they clip on data that you may look at.

The formulas also treat $n$ as public, which is true only because neighbours replace a record. If the size is private, divide a noisy sum by a noisy count, and split the budget between the two.

## The cost of an unknown range {#sec-mean-coarse}

Now let the records be normal with variance 1, with a mean that you know only to be within a distance $R$ of zero. Write $\tau$ for a distance from the mean that holds all the records, except with a small failure probability $\beta$. For 1,000 records at $\beta = 0.05$, $\tau$ is 4.604.

The **naive estimator** clips every record to the widest range where the data can be, $R + \tau$ on each side of zero. It then adds Laplace noise for that width, so its error grows in proportion to $R$. With 1,000 records at $\varepsilon = 1$, its RMSE is 2828.44 when $R$ is a million. But [Lower bounds](10-lower-bounds.md) shows that an unknown range needs only about $\ln R/\varepsilon$ more records.

**Coarse then fine** (Karwa & Vadhan 2018) comes close to that floor in two steps. The coarse step cuts the range into bins of width $2\tau$. It then uses a part $\varepsilon_1$ of the budget to pick the fullest bin with the [exponential mechanism](05-selection.md). The records it needs grow only with the logarithm of the number of bins. The data can be across two bins, so the fine step clips every record to the chosen bin and its two neighbours. It releases the average with Laplace noise for that window and the rest of the budget, so by [composition](02-properties.md) the two steps are $\varepsilon$-DP.

With a quarter of the budget on the coarse step, the same 1,000 records give an RMSE of 0.0609 for every $R$ from 10 to a million. But the fine step has only three quarters of the budget and a window three bins wide. So for small ranges the naive estimator is better, here up to $R = 13.81$. For larger ranges, use coarse then fine, or find the data with a private quantile such as diffprivlib's `tools.quantile`. Under approximate DP, with its slack $\delta$, a stability-based histogram does the coarse step with about $\ln(1/\delta)/\varepsilon$ records, so $R$ goes away ([Beyond global sensitivity](07-sensitivity.md)).

Two things go wrong. A budget split, bin width or $\beta$ that you choose after a look at the data leaks it. And a coarse step with too few records can pick an empty bin and clip all records to the wrong place.

## Many dimensions: clip the norm {#sec-mean-multi}

Vectors in $d$ dimensions, such as embeddings or gradients, use the same recipe with a different measure of size. Here the size is the $\ell_2$ norm, or length. Clip each record to a radius $C$, so that one replaced record moves the average by at most $2C/n$. Then add Gaussian noise for that sensitivity ([The Laplace and Gaussian mechanisms](03-laplace-gaussian.md)). This is the step of private gradient descent, and [private training](../part05-dp-ml/02-dp-sgd.md) tells how to choose its clipping norm.

The radius sets the cost. With unit variance per coordinate, records are about $\sqrt d$ from their mean, so the radius must grow like $\sqrt d$. The noise in all coordinates then grows like $d$ in total, against $\sqrt d$ for the sampling error. So in high dimension privacy is the larger cost, and the [fingerprinting lower bound](10-lower-bounds.md) shows that no mechanism avoids it.

If you do not know where the mean is, a grid of bins pays for the range once per dimension. The **shrinking balls** of CoinPress (Biswas, Dong, Kamath & Ullman 2020) need no grid. Each round clips to a ball around a noisy guess of the mean, then makes the ball smaller. Fix the number of rounds and their budget split before the run.

## Heavy tails: clip into the data {#sec-mean-heavy}

Incomes, session lengths and per-example gradient norms have heavy tails. If you know only that the variance is at most $\sigma^2$, a range that clips no record is very wide. So clip into the data on purpose, and accept a bias.

Now let $\tau$ be a clipping half-width. A coarse step finds a centre near the mean, and you clip every record to within $\tau$ of it. Only the rare records beyond $\tau$ move, so with the centre at the mean the bias is at most $\sigma^2/\tau$. But the noise grows in proportion to $\tau$. The bound on the total error is smallest at
$$\tau_b = \sigma\sqrt{\frac{n\varepsilon}{2\sqrt2}},$$
which uses only $n$, $\varepsilon$ and the variance bound. At this width, an error $\alpha$ needs about $\sigma^2/\alpha^2 + 8\sqrt2\,\sigma^2/(\alpha^2\varepsilon)$ records. Both terms grow like $1/\alpha^2$, so privacy now costs as much as sampling.

@tbl-mean-heavy shows the trade-off on Pareto records, a heavy-tailed distribution scaled to variance 1. The first width clips a record only with probability 0.05, and noise is most of its error. The worst-case width $\tau_b$ decreases the error to 0.0607, and the best width for this distribution is better again.

| Clipping half-width $\tau$ | Bias | Sampling sd | Noise sd | RMSE |
| --- | --- | --- | --- | --- |
| 141.42 (no record clipped) | 0.0000 | 0.0313 | 0.4000 | 0.4012 |
| 18.80 (worst-case best, $\tau_b$) | −0.0018 | 0.0293 | 0.0532 | 0.0607 |
| 5.36 (best for this distribution) | −0.0153 | 0.0249 | 0.0152 | 0.0329 |

: Clipped private means of Pareto records with variance 1, 1,000 records, $\varepsilon = 1$, for three clipping half-widths. {#tbl-mean-heavy}

So use $\tau_b$ as a safe default, not as the best width. A narrower range is usually better, but choose it from domain knowledge or with a private quantile, never from a look at the records ([Beyond global sensitivity](07-sensitivity.md)).

## Code

The snippet computes the records that a bounded mean needs, checks its noise with OpenDP, and computes the default clipping width for heavy tails. The RMSE equal to $\alpha$, times $n^2$, is a quadratic in $n$, and the first function rounds its positive root up.

```python
from math import ceil, sqrt

import opendp.prelude as dp


def records_for_bounded_mean(alpha: float, sd: float, width: float, eps: float) -> int:
    """Smallest n with sd^2/n + 2 (width/(n eps))^2 <= alpha^2."""
    a, c = sd**2, 2 * (width / eps) ** 2
    return ceil((a + sqrt(a * a + 4 * c * alpha**2)) / (2 * alpha**2))


def heavy_tail_width(n: int, eps: float, sd: float) -> float:
    """The clipping half-width that balances the bias bound sd^2/tau against the noise."""
    return sd * sqrt(n * eps / (2 * sqrt(2)))


n = records_for_bounded_mean(0.05, 0.5, 1.0, 1.0)
print(n)  # 108
print(f"{dp.laplacian_scale_to_accuracy(1.0 / (n * 1.0), alpha=0.05):.4f}")  # 0.0277
print(f"{heavy_tail_width(1000, 1.0, 1.0):.2f}")  # 18.80
```

OpenDP gives accuracy as a half-width that holds the noise with a chosen probability. So `laplacian_scale_to_accuracy` changes the noise scale $w/(n\varepsilon)$ of a range of width $w$ at the planned 108 records into the half-width at 95%.

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
