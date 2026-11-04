# The Laplace and Gaussian mechanisms

## Why this matters

Suppose a hospital wants to publish the number of patients with diabetes under differential privacy (DP). Most private releases use one of the two mechanisms in this chapter. Private training also uses them to add noise to gradients. Each mechanism computes a query $f$ on dataset $D$, adds fresh noise $Z$ and publishes
$$M(D) = f(D) + Z.$$
Three choices determine the mechanism. Calibrate noise to the query's **sensitivity**, the largest change one record can cause. Choose Laplace noise for pure DP or Gaussian noise with slack $\delta$. Then apply a calibration formula to obtain the noise required for target $\varepsilon$. The formulas follow from privacy loss in [the definition of differential privacy](01-definition.md#sec-dpdef-divergences).

## Sensitivity {#sec-lg-sensitivity}

Sensitivity measures the largest change one record can cause. For a vector query with $d$ outputs, such as a histogram, the change needs a norm. Laplace noise uses the $\ell_1$ norm, the sum of absolute changes; Gaussian noise uses the $\ell_2$ norm, the square root of the sum of squared changes. For neighbouring datasets $D \sim D'$, the query's **$\ell_p$ sensitivity** is the largest change between their answers:
$$\Delta_pf = \max_{D \sim D'}\big\|f(D) - f(D')\big\|_p, \qquad p \in \{1, 2\}.$$
The two are equal when one record moves a single coordinate. When it moves all $d$ of them, $\Delta_1f$ can be $\sqrt d$ times $\Delta_2f$.

The sensitivity also depends on the neighbouring relation. Under add/remove, one dataset has one record more than the other. Under replace, one record changes into another, so the size $n$ is public. With $B$ for the largest value of a record after clipping, @tbl-lg-sensitivity gives the standard queries.

| Query | Add/remove $\Delta_1$ | Add/remove $\Delta_2$ | Replace $\Delta_1$ | Replace $\Delta_2$ |
| --- | --- | --- | --- | --- |
| Count $\lvert D\rvert$ | 1 | 1 | 0 | 0 |
| Sum, values in $[0, B]$ | $B$ | $B$ | $B$ | $B$ |
| Mean, values in $[0, B]$, $n$ public | none (@sec-lg-recipe) | none | $B/n$ | $B/n$ |
| Histogram with $k$ bins | 1 | 1 | 2 | $\sqrt2$ |
| $d$ threshold counts $\#\{x \ge j\}$, $j = 1..d$ | $d$ | $\sqrt d$ | $d$ | $\sqrt d$ |

: Sensitivities of the standard queries under the two neighbouring relations. {#tbl-lg-sensitivity}

A replaced record leaves one histogram bin and goes into another, so it changes two bins. The threshold counts, the numbers of records at or above each of $d$ values, are a **dense** query, in which one record can move every coordinate.

Calculate sensitivity from the domain before examining the data. A data-dependent bound, such as the observed maximum, can leak information ([Beyond global sensitivity](07-sensitivity.md)). Unbounded data have infinite sensitivity, so clip each value to a fixed range. If ages are clipped at 100, the mean age of 1,000 patients has sensitivity $B/n = 0.1$ under replace.

## The Laplace mechanism {#sec-lg-laplace}

The Laplace mechanism adds independent noise from the **Laplace distribution**, a two-sided exponential distribution centred at 0, to each coordinate. Write $\mathrm{Lap}(b)$ for Laplace noise with scale $b$. For an $\varepsilon$-DP guarantee, set the scale to sensitivity divided by $\varepsilon$:
$$b = \frac{\Delta_1f}{\varepsilon}.$$ {#eq-lg-laplace}

The mechanism's error is its noise. Each coordinate's mean absolute error is $b$, and its probability of exceeding distance $t$ is $e^{-t/b}$. To keep the failure probability across all $d$ coordinates at most $\beta$, allocate $\beta/d$ to each coordinate. The simultaneous error bound is
$$\Pr\Big[\max_i|Z_i| > b\ln\frac d\beta\Big] \le \beta.$$
Publish this width with the release. In reverse, the width gives an error budget, because for a single number you can fix a target width $t$ and solve for the privacy level,
$$\varepsilon = \frac{\Delta_1f\,\ln(1/\beta)}{t}.$$
A count required to be within 10 of its true value with probability 0.95 needs $\varepsilon = \ln(20)/10 = 0.2996$. For $d$ outputs, use failure probability $\beta/d$ for each and add their privacy levels by [composition](02-properties.md#sec-dpprop-comp).

At the hospital, 1,234 of the 10,000 patients have diabetes. A count has sensitivity 1 under add/remove, so at $\varepsilon = 0.1$ the noise is $\mathrm{Lap}(10)$. Its 95% error bar is $10\ln 20 = 29.96$, or 2.43% of the count.

The noise scale does not depend on dataset size, so relative error for a count of size $n$ is about $1/(n\varepsilon)$. Sensitivity must include every output a record can change. If one record affects all 10 released counts, calibrating each count separately makes the release $10\varepsilon$-DP, not $\varepsilon$-DP.

## The Gaussian mechanism {#sec-lg-gauss}

The Gaussian mechanism adds independent normal noise with mean 0 and standard deviation $\sigma$ to each coordinate. Its privacy loss is unbounded in the tail, so its guarantee requires $\delta$. Gaussian noise uses $\ell_2$ sensitivity, which may be much smaller than $\ell_1$ sensitivity. Let $\mu = \Delta_2f/\sigma$ be the neighbour-output distance in noise standard deviations, and let $\Phi$ be the standard normal cumulative distribution function (CDF). The mechanism is $(\varepsilon, \delta)$-DP exactly when $\delta$ is at least
$$\delta_\mu(\varepsilon) = \Phi\Big(\frac\mu2 - \frac\varepsilon\mu\Big) - e^\varepsilon\,\Phi\Big(-\frac\mu2 - \frac\varepsilon\mu\Big).$$ {#eq-lg-analytic}

Calibrate the mechanism using this formula. As $\mu$ increases, exact $\delta$ increases from 0 to 1, so a unique value $\mu^*$ gives the target. The minimum noise scale is
$$\sigma^* = \frac{\Delta_2f}{\mu^*}, \qquad \text{where } \delta_{\mu^*}(\varepsilon) = \delta,$$
and libraries solve for $\mu^*$ (@sec-lg-code). For 10,000 patients, the [book's default slack](01-definition.md#sec-dpdef-small-delta) is $\delta = 10^{-5}$. At $\varepsilon = 1$, this **analytic calibration** then gives $\sigma^* = 3.7306\,\Delta_2f$.

Many papers and libraries still use the classic calibration of Dwork and Roth (2014), proved only for $\varepsilon$ and $\delta$ between 0 and 1:
$$\sigma = \frac{\Delta_2f\sqrt{2\ln(1.25/\delta)}}{\varepsilon}.$$ {#eq-lg-classic}
At $\varepsilon = 1$ and $\delta = 10^{-5}$, it gives $4.8448\,\Delta_2f$, 30% more noise than the analytic $3.7306\,\Delta_2f$. Above $\varepsilon = 8.42$, it gives insufficient noise for its claimed guarantee. Use analytic calibration instead.

A Gaussian mechanism has a privacy curve, with a value of $\delta$ for each $\varepsilon$. Accountants in [Part 4](../part04-privacy-accounting/00-introduction.md) compose these curves. Report $\sigma/\Delta_2f$ and the selected $(\varepsilon,\delta)$ for each Gaussian release.

At the hospital, Gaussian noise for the diabetes count at $\varepsilon = 0.1$ and $\delta = 10^{-5}$ gives a 95% error bar of 60.27, which is 1.96 standard deviations. Laplace noise gave 29.96 with no $\delta$, so for a single count Laplace noise is better in error and in guarantee.

## Laplace or Gaussian? {#sec-lg-choose}

Choose a mechanism based on how one record's change is distributed across coordinates. Compare the root-mean-square error (RMSE) per released value at the same $\varepsilon$. Laplace noise with scale $b = \Delta_1f/\varepsilon$ has standard deviation $\sqrt2\,b$. Let $\sigma_1$ be the analytic Gaussian noise scale at unit $\ell_2$ sensitivity. The per-coordinate errors are
$$\text{Laplace: } \frac{\sqrt2\,\Delta_1f}{\varepsilon}, \qquad \text{Gaussian: } \Delta_2f\,\sigma_1.$$

In a sparse query, such as a histogram under add/remove, one record moves one coordinate, so the two sensitivities are equal. At $\varepsilon = 1$ and the default $\delta$, $\sigma_1 = 3.7306$. So Laplace noise is better at every $d$, because $\sqrt2 = 1.4142$ is smaller. In a dense query, such as the threshold counts, one record moves every coordinate. Then the Laplace error grows like $d$ and the Gaussian error only like $\sqrt d$. So Laplace noise is more accurate only while
$$d < \frac{(\varepsilon\sigma_1)^2}2,$$
so with $\sigma_1 = 3.7306$, Gaussian noise is better for 7 or more threshold counts.

Gaussian noise is also better for many releases, because several Gaussian releases compose exactly into one Gaussian mechanism. Private training has dense queries and many releases, because one record's gradient moves every coordinate of a model at every step. So it clips the $\ell_2$ norm of each gradient and calibrates Gaussian noise to that clip.

## Calibrating a release {#sec-lg-recipe}

Every noisy release has five steps.

1. Fix the neighbouring relation and the unit of privacy, usually one person under add/remove.
2. Clip every value to a range fixed in advance. Then take the sensitivity from @tbl-lg-sensitivity, or calculate it from the worst record.
3. Use Laplace noise for one number or a sparse vector, and Gaussian noise for a dense vector.
4. Set the scale with @eq-lg-laplace or @eq-lg-analytic. Charge it to your budget ([Post-processing, group privacy and composition](02-properties.md)).
5. Draw the noise with a library that experts have checked. Publish the error bar and the noise parameters with the result.

A mean under add/remove has no simple sensitivity because its denominator changes. Release a noisy sum and a noisy count, splitting the budget between them, then divide; this division is post-processing. To obtain a 95% interval for the mean, use 97.5% intervals for both quantities. The mean is at least the sum's lower bound divided by the count's upper bound, and at most the sum's upper bound divided by the count's lower bound when both lower bounds are positive.

OpenDP's `.dp.mean` uses the same split, and `summarize(alpha=0.025)` reports both 97.5% intervals. If the clipping range is unknown, [Private mean estimation](11-mean-estimation.md) describes how to estimate it privately. Use `numpy.random.laplace` only for prototypes: floating-point samplers can leak through low-order output bits ([Discrete and floating-point-safe mechanisms](08-discrete-mechanisms.md)). For a complete release with clipping, grouping and one budget, see [Release statistics with a library](12-release-statistics.md).

## Code {#sec-lg-code}

The snippet checks the hospital's values and analytic $\sigma^*$ at $\varepsilon = 1$. In OpenDP, a mechanism's `map` converts sensitivity to privacy cost. Accuracy helpers convert between noise scale and a 95% error bound. For a count with sensitivity 1, the reciprocal of the Laplace scale is the privacy level. IBM's diffprivlib calibrates the analytic Gaussian mechanism.


```python
import math

import opendp.prelude as dp
from diffprivlib.mechanisms import GaussianAnalytic

dp.enable_features("contrib")
laplace = dp.m.make_laplace(dp.atom_domain(T=float, nan=False), dp.absolute_distance(T=float), scale=10.0)
print(laplace.map(1.0))  # 0.1
print(f"{dp.laplacian_scale_to_accuracy(10.0, alpha=0.05):.2f}")  # 29.96
print(f"{1 / dp.accuracy_to_laplacian_scale(10.0, alpha=0.05):.4f}")  # 0.2996
gauss = GaussianAnalytic(epsilon=1.0, delta=1e-5, sensitivity=1.0)
print(f"{math.sqrt(gauss.variance(0.0)):.4f}")  # 3.7306

```

Apply the mechanism to the value to add noise, as in `laplace(1234.0)`.

## Cheat sheet

| Result | Formula | Notes |
| --- | --- | --- |
| Laplace mechanism | $\mathrm{Lap}(\Delta_1f/\varepsilon)$ per coordinate | Pure DP; error does not depend on $n$ |
| Laplace error bar | $b\ln(d/\beta)$ for all $d$ coordinates, scale $b$ | Solve it for $\varepsilon$ to budget error |
| Exact Gaussian $\delta$ | $\Phi(\frac\mu2 - \frac\varepsilon\mu) - e^\varepsilon\Phi(-\frac\mu2 - \frac\varepsilon\mu)$, $\mu = \Delta_2f/\sigma$ | Solve for $\sigma$ with `GaussianAnalytic` |
| Classic Gaussian | $\sigma = \Delta_2f\sqrt{2\ln(1.25/\delta)}/\varepsilon$ | Too much noise below 1, unsafe above 8.42 |
| Dense query | Laplace wins only for $d < (\varepsilon\sigma_1)^2/2$ | Otherwise use Gaussian noise |

: The results of this chapter. {#tbl-lg-cheat}

**Rules of thumb**


- Calibrate Laplace noise to $\Delta_1f$ and Gaussian noise to $\Delta_2f$, never the other way round.


- When $n$ is not public, never use $B/n$ as the sensitivity of a mean.

## References

- Dwork & Roth, *The Algorithmic Foundations of Differential Privacy* (2014), §3.3 and Appendix A: the Laplace mechanism and the classic Gaussian calibration.
- Balle & Wang, "Improving the Gaussian Mechanism for Differential Privacy" (ICML 2018): the exact privacy curve and the bisection calibration.

**Derivations.** Every result here is derived step by step in [the full edition's chapter](../../book-full/part03-dp-foundations/03-laplace-gaussian.md).
