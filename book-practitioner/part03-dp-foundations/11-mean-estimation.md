# Private mean estimation

## Why this matters

Private machine learning uses means extensively: each private gradient-descent step averages clipped per-example gradients, and a survey share is the mean of binary answers. Before releasing a mean, determine the required sample size and choose a clipping range without inspecting private data.

Differential privacy (DP) must hold for every dataset, including datasets excluded by modelling assumptions ([The definition of differential privacy](01-definition.md)). An assumption such as normally distributed records may improve accuracy, but does not enter the privacy proof. The mechanisms below therefore clip each record to a range. Noise increases with range width, but the range cannot be estimated directly from private data. Spend part of the budget to estimate the range privately, then clip to it.

## Bounded data: privacy adds a lower-order term {#sec-mean-bounded}

This chapter assumes fixed-size datasets of size $n$, with neighbours differing in one replaced record. It measures estimation error by **root-mean-square error** (RMSE), the typical distance between the estimate and the population mean.

Suppose records lie in a known interval of width $w$, such as binary answers coded 0 and 1, and let $\sigma^2$ be their variance. Release the sample mean plus Laplace noise with scale $w/(n\varepsilon)$. Since one record changes the mean by at most $w/n$, this gives $\varepsilon$-DP ([The Laplace and Gaussian mechanisms](03-laplace-gaussian.md)). Laplace noise has variance twice the squared scale and is independent of sampling error, so the variances add:
$$\mathrm{RMSE} = \sqrt{\frac{\sigma^2}{n} + \frac{2w^2}{n^2\varepsilon^2}}.$$
To plan a study, set RMSE to target error $\alpha$ and solve for $n$, as in the Code section. The required record count is at most
$$\frac{\sigma^2}{\alpha^2} + \frac{\sqrt2\,w}{\alpha\varepsilon}.$$

Sampling alone requires about $\sigma^2/\alpha^2$ records; privacy adds at most about $w/(\alpha\varepsilon)$. For a binary survey with half the population answering yes, an RMSE of 0.05 requires 100 people, increasing to 108 at $\varepsilon = 1$. The privacy term grows as $1/\alpha$, compared with $1/\alpha^2$ for sampling, so it matters most for coarse targets and small $\varepsilon$. It also increases with interval width $w$; subsequent sections reduce this width.

Libraries require a range. In diffprivlib, `tools.mean(x, epsilon=..., bounds=(lower, upper))` clips values to the bounds and adds noise. Without `bounds`, it uses the observed minimum and maximum, which leaks them, and raises `PrivacyLeakWarning`.

Bounds that are too narrow clip valid records. The script [practitioner_adult_statistics.py](../../code/part03-dp-foundations/practitioner_adult_statistics.py) estimates mean weekly work hours in Adult, a census dataset with 48,842 people, at $\varepsilon = 1$ using both loose and tight bounds.

```{.python excerpt="adult-mean-bounds"}
hours = adult["hours-per-week"].to_numpy()
loose = tools.mean(hours, epsilon=1.0, bounds=(1, 99), random_state=seed)
tight = tools.mean(hours, epsilon=1.0, bounds=(20, 60), random_state=seed)
clipped = np.clip(hours, 20, 60).mean()  # what the tight bounds can at best recover
print(f"true mean hours: {hours.mean():.3f}")
print(f"bounds (1, 99): {loose:.3f}  bounds (20, 60): {tight:.3f}")
print(f"bias of (20, 60): {clipped - hours.mean():+.3f}")
```

The seed-0 run, `uv run python code/part03-dp-foundations/practitioner_adult_statistics.py`, prints the following; the final two values come from code adjacent to the excerpt.

```text
true mean hours: 40.422
bounds (1, 99): 40.421  bounds (20, 60): 40.361
bias of (20, 60): -0.061
people outside (20, 60): 8.7%
noise scale (1, 99): 0.0020  (20, 60): 0.0008
```

Tight bounds reduce the noise scale from 0.0020 to 0.0008 hours but clip 8.7% of people, shifting the mean by 0.061 hours—more than the noise reduction. Privacy budget cannot remove this bias. Report sampling error, clipping bias and noise separately. Choose bounds from the schema or public data, and measure the clipped share only on data available for inspection.

These formulas assume public $n$, which holds for replace-one-record neighbours. If dataset size is private, release a noisy sum and noisy count, split the budget between them, then divide.

## The cost of an unknown range {#sec-mean-coarse}

Now suppose records are normal with variance 1 and their mean is known only to lie within distance $R$ of zero. Let $\tau$ be a radius that contains all records except with failure probability $\beta$. For 1,000 records and $\beta = 0.05$, $\tau$ is 4.604.

The **naive estimator** clips records to the full possible range, from $-(R+\tau)$ to $R+\tau$, then adds Laplace noise calibrated to that width. Its error therefore grows linearly with $R$. With 1,000 records at $\varepsilon = 1$, RMSE is 2828.44 when $R$ is one million. [Lower bounds](10-lower-bounds.md) shows that an unknown range should require only about $\ln R/\varepsilon$ additional records.

**Coarse then fine** (Karwa & Vadhan 2018) approaches this bound in two steps. The coarse step partitions the range into bins of width $2\tau$ and uses budget $\varepsilon_1$ with the [exponential mechanism](05-selection.md) to select the fullest bin. Its record requirement grows logarithmically with the bin count. Because the data may span two bins, the fine step clips records to the selected bin and its two neighbours, then releases the mean with Laplace noise using the remaining budget. By [composition](02-properties.md), the combined procedure is $\varepsilon$-DP.

Allocating one quarter of the budget to the coarse step gives RMSE 0.0609 for the same 1,000 records across $R$ from 10 to one million. The fine step uses the remaining three quarters and a three-bin window, so the naive estimator performs better for small ranges, up to $R = 13.81$. For larger ranges, use coarse then fine or estimate the range with a private quantile such as diffprivlib's `tools.quantile`. Under approximate DP with slack $\delta$, a stability-based histogram performs the coarse step with about $\ln(1/\delta)/\varepsilon$ records, eliminating dependence on $R$ ([Beyond global sensitivity](07-sensitivity.md)).

Two failures are possible. Choosing the budget split, bin width or $\beta$ after inspecting private data leaks information. With too few records, the coarse step may select an empty bin and clip every record incorrectly.

## Many dimensions: clip the norm {#sec-mean-multi}

For $d$-dimensional vectors, such as embeddings or gradients, measure size with the $\ell_2$ norm. Clip each record to radius $C$, so one replacement changes the mean by at most $2C/n$. Add Gaussian noise calibrated to this sensitivity ([The Laplace and Gaussian mechanisms](03-laplace-gaussian.md)). Private gradient descent uses this procedure; [private training](../part05-dp-ml/02-dp-sgd.md) explains how to select its clipping norm.

The clipping radius determines the cost. With unit variance per coordinate, records lie about $\sqrt d$ from their mean, so the radius must scale as $\sqrt d$. Total noise across coordinates then grows as $d$, compared with sampling error of order $\sqrt d$. Privacy therefore dominates in high dimensions, as the [fingerprinting lower bound](10-lower-bounds.md) confirms.

If the mean's location is unknown, a grid of bins incurs a range cost in every dimension. CoinPress's **shrinking balls** method (Biswas, Dong, Kamath & Ullman 2020) avoids a grid. Each round clips to a ball centred on a noisy mean estimate and reduces the radius. Fix the number of rounds and budget allocation before using private data.

## Heavy tails: clip into the data {#sec-mean-heavy}

Incomes, session lengths and per-example gradient norms can have heavy-tailed distributions. If only the variance bound $\sigma^2$ is known, a range that clips no records may be too wide for useful privacy. Clipping can reduce noise, but introduces bias.

Let $\tau$ be the clipping half-width. A coarse step estimates a centre near the mean, then each record is clipped to within $\tau$ of it. If the centre equals the mean, the bias is at most $\sigma^2/\tau$, since only records beyond $\tau$ change. Noise grows in proportion to $\tau$. The resulting error bound is minimized at
$$\tau_b = \sigma\sqrt{\frac{n\varepsilon}{2\sqrt2}},$$
This width depends only on $n$, $\varepsilon$ and the variance bound. At this width, error $\alpha$ requires about $\sigma^2/\alpha^2 + 8\sqrt2\,\sigma^2/(\alpha^2\varepsilon)$ records. Both terms scale as $1/\alpha^2$, so privacy can cost as much as sampling.

@tbl-mean-heavy shows this trade-off for Pareto records scaled to variance 1. At the first width, 0.05 of records are clipped and noise dominates the error. The worst-case width $\tau_b$ reduces error to 0.0607; a width tuned to this distribution reduces it further.

| Clipping half-width $\tau$ | Bias | Sampling sd | Noise sd | RMSE |
| --- | --- | --- | --- | --- |
| 141.42 (no record clipped) | 0.0000 | 0.0313 | 0.4000 | 0.4012 |
| 18.80 (worst-case best, $\tau_b$) | −0.0018 | 0.0293 | 0.0532 | 0.0607 |
| 5.36 (best for this distribution) | −0.0153 | 0.0249 | 0.0152 | 0.0329 |

: Clipped private means of Pareto records with variance 1, 1,000 records, $\varepsilon = 1$, for three clipping half-widths. {#tbl-mean-heavy}

Use $\tau_b$ as a conservative default rather than an optimal width. A narrower range may improve accuracy, but choose it from domain knowledge or a private quantile, not by inspecting the records ([Beyond global sensitivity](07-sensitivity.md)).

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
