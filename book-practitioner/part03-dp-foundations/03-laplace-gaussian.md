# The Laplace and Gaussian mechanisms

## Why this matters

Suppose you run a hospital database and want to publish how many patients have diabetes with a differential privacy (DP) guarantee. Almost every such release is built with one of the two mechanisms of this chapter, and so are the noisy gradients of private model training.

Both compute the exact answer of a query $f$ on the dataset $D$. They then add fresh random noise, written $Z$, so that the mechanism publishes
$$M(D) = f(D) + Z.$$
Three decisions fix the whole mechanism. The **sensitivity**, how far one record can move the answer, is what the noise must hide. The noise distribution sets the kind of guarantee, since Laplace noise gives pure DP while Gaussian noise needs a small slack $\delta$. And a calibration formula sets the amount of noise for a target $\varepsilon$, through the privacy loss of [the definition of differential privacy](01-definition.md#sec-dpdef-divergences).

## Sensitivity {#sec-lg-sensitivity}

The noise must cover the largest change that one record can make to the answer. A query often returns a vector of $d$ numbers, such as the bins of a histogram, so you need a size for a change to a vector. Laplace noise uses the $\ell_1$ norm, the sum of the absolute changes. Gaussian noise uses the $\ell_2$ norm, the square root of the sum of the squared changes. Writing $D \sim D'$ for a pair of neighbours, the **$\ell_p$ sensitivity** of the query is the largest change between their answers,
$$\Delta_pf = \max_{D \sim D'}\big\|f(D) - f(D')\big\|_p, \qquad p \in \{1, 2\}.$$
The two are equal when one record moves a single coordinate. When it moves all $d$ of them, the $\ell_1$ sensitivity can be up to $\sqrt d$ times the $\ell_2$ one.

The sensitivity also depends on the neighbouring relation. Under add/remove one dataset has one record more than the other, and under replace one record is swapped for another, so the size $n$ is public. Write $B$ for the largest value a record may take after clipping, which the next paragraphs explain. @tbl-lg-sensitivity then gives the standard queries.

| Query | Add/remove $\Delta_1$ | Add/remove $\Delta_2$ | Replace $\Delta_1$ | Replace $\Delta_2$ |
| --- | --- | --- | --- | --- |
| Count $\lvert D\rvert$ | 1 | 1 | 0 | 0 |
| Sum, values in $[0, B]$ | $B$ | $B$ | $B$ | $B$ |
| Mean, values in $[0, B]$, $n$ public | none (@sec-lg-recipe) | none | $B/n$ | $B/n$ |
| Histogram with $k$ bins | 1 | 1 | 2 | $\sqrt2$ |
| $d$ threshold counts $\#\{x \ge j\}$, $j = 1..d$ | $d$ | $\sqrt d$ | $d$ | $\sqrt d$ |

: Sensitivities of the standard queries under the two neighbouring relations. {#tbl-lg-sensitivity}

A replaced record leaves one histogram bin and joins another, which is why the relation must be named. The threshold counts, the numbers of records at or above each of $d$ values, are a **dense** query, in which one record can move every coordinate.

Compute the sensitivity from the domain, before you look at the data. A sensitivity read off the data, for example from the largest value present, makes the noise itself leak that value ([Beyond global sensitivity](07-sensitivity.md)). For unbounded data the sensitivity is infinite, so clip every value to a fixed range first. For example, take the mean age of 1,000 patients, with ages clipped to at most 100. Under replace its sensitivity is 0.1, since no patient can move the mean by more than a tenth of a year.

## The Laplace mechanism {#sec-lg-laplace}

The Laplace mechanism (Dwork et al., 2006) adds independent noise from the **Laplace distribution** to each coordinate, a two-sided exponential peaked at 0. Its scale, written $b$, sets its typical size, and $\mathrm{Lap}(b)$ denotes it. For an $\varepsilon$-DP guarantee, the mechanism sets the scale to the sensitivity divided by $\varepsilon$,
$$b = \frac{\Delta_1f}{\varepsilon}.$$ {#eq-lg-laplace}

The error of the mechanism is the noise itself. Each coordinate is off by $b$ on average, and lands more than a distance $t$ from the truth with probability $e^{-t/b}$. To bound all $d$ coordinates at once, except with a small failure probability $\beta$, give each coordinate a share $\beta/d$ of it, so that the shares add up to the whole. The width that results is
$$\Pr\Big[\max_i|Z_i| > b\ln\frac d\beta\Big] \le \beta.$$
Publish this width with the release. You can also read it backwards as an error budget. For a single number, where $d = 1$, fix a target width $t$ and solve for the privacy level, which gives
$$\varepsilon = \frac{\Delta_1f\,\ln(1/\beta)}{t}.$$
For example, a count that must be within 10 of the truth, with probability 0.95, needs an $\varepsilon$ of 0.2996. For $d$ numbers released together, solve each one with the failure probability $\beta/d$. Their $\varepsilon$ then add up by [composition](02-properties.md#sec-dpprop-comp).

At the hospital, 1,234 of the 10,000 patients have diabetes. A count has sensitivity 1 under add/remove, so at $\varepsilon = 0.1$ the noise is $\mathrm{Lap}(10)$. Its 95% error bar is $10\ln 20 = 29.96$, a relative error of 0.0243.

The scale does not contain the size of the data, so the relative error of a count of size $n$ is about $1/(n\varepsilon)$. But the sensitivity must include every number one record can move. If one record can change all of 10 released counts, noise calibrated for one count makes the release $10\varepsilon$-DP, not $\varepsilon$-DP.

## The Gaussian mechanism {#sec-lg-gauss}

The Gaussian mechanism adds independent normal noise with mean 0 to each coordinate, with a standard deviation written $\sigma$. Its privacy loss is unbounded in the far tail, so no $\varepsilon$ covers every output, and it always carries a $\delta$. In exchange, it is calibrated to the $\ell_2$ sensitivity, which can be much smaller than the $\ell_1$ one. Only the ratio of the sensitivity to the noise matters, so write $\mu = \Delta_2f/\sigma$ for the number of noise standard deviations between the answers on two neighbours. Write $\Phi$ for the cumulative distribution function (CDF) of the standard normal distribution. The mechanism is then $(\varepsilon, \delta)$-DP exactly when $\delta$ is at least
$$\delta_\mu(\varepsilon) = \Phi\Big(\frac\mu2 - \frac\varepsilon\mu\Big) - e^\varepsilon\,\Phi\Big(-\frac\mu2 - \frac\varepsilon\mu\Big).$$ {#eq-lg-analytic}

The formula is how you calibrate. As $\mu$ grows, $\delta_\mu(\varepsilon)$ rises steadily from 0 to 1, so exactly one value $\mu^*$ hits your target $\delta$. The smallest noise that works is then
$$\sigma^* = \frac{\Delta_2f}{\mu^*}, \qquad \text{where } \delta_{\mu^*}(\varepsilon) = \delta,$$
and libraries solve for $\mu^*$ for you (@sec-lg-code). Take $\varepsilon = 1$ and $\delta = 10^{-5}$, the book's default for the hospital's 10,000 patients ([The definition of differential privacy](01-definition.md#sec-dpdef-small-delta)). Then this **analytic calibration** gives $\sigma^* = 3.7306\,\Delta_2f$.

Many papers and libraries still use the classic calibration of Dwork and Roth (2014). For $\varepsilon$ and $\delta$ both between 0 and 1, it sets
$$\sigma = \frac{\Delta_2f\sqrt{2\ln(1.25/\delta)}}{\varepsilon}.$$ {#eq-lg-classic}
@tbl-lg-calibration compares the two at $\delta = 10^{-5}$.

| $\varepsilon$ | Classic $\sigma/\Delta_2f$ | Analytic $\sigma^*/\Delta_2f$ | Ratio |
| --- | --- | --- | --- |
| 0.1 | 48.4481 | 30.7496 | 1.576 |
| 1 | 4.8448 | 3.7306 | 1.299 |
| 8 | 0.6056 | 0.6002 | 1.009 |

: Classic against analytic calibration of the Gaussian mechanism at $\delta = 10^{-5}$. {#tbl-lg-calibration}

At $\varepsilon = 1$ the classic formula adds 30% more noise than needed, and used above 1 it is wrong, not just loose. At $\delta = 10^{-5}$ the two calibrations cross at $\varepsilon = 8.42$, and beyond it the classic $\sigma$ is too small to give the claimed guarantee.

A Gaussian mechanism satisfies a whole curve of guarantees, one $\delta$ for each $\varepsilon$, and the accountants of [Part 4](../part04-privacy-accounting/00-introduction.md) compose whole curves. So report $\sigma/\Delta_2f$ with every Gaussian release, as well as the pair you chose.

Return to the diabetes count, at the Laplace record's $\varepsilon = 0.1$ and with $\delta = 10^{-5}$. Its 95% error bar is 1.96 standard deviations, or 60.27. Laplace noise gave 29.96 with no $\delta$ at all, so for a single count it wins on both error and guarantee.

## Laplace or Gaussian? {#sec-lg-choose}

The choice depends on how one record spreads its change across the coordinates, as the root-mean-square error (RMSE) of each released number shows. Write $\sigma_1$ for the analytic noise level at unit $\ell_2$ sensitivity. A $\mathrm{Lap}(b)$ variable has standard deviation $\sqrt2\,b$, so at the same $\varepsilon$ the errors per coordinate are
$$\text{Laplace: } \frac{\sqrt2\,\Delta_1f}{\varepsilon}, \qquad \text{Gaussian: } \Delta_2f\,\sigma_1.$$

In a sparse query, such as a histogram under add/remove, one record moves one coordinate. The two sensitivities are then equal, so at $\varepsilon = 1$ and the default $\delta$ Laplace noise wins at every $d$, since $\sqrt2 = 1.4142$ is below 3.7306. In a dense query, by contrast, one record moves every coordinate. The Laplace error then grows like $d$ while the Gaussian error grows only like $\sqrt d$, so Laplace noise is more accurate only while
$$d < \frac{(\varepsilon\sigma_1)^2}2.$$
At $\varepsilon = 1$ and the default $\delta$, the unit noise level is 3.7306, so for threshold counts Gaussian noise wins from 7 counts on.

Gaussian noise also wins when releases are composed many times, because several Gaussian releases compose exactly into one Gaussian mechanism, whose $\ell_2$ sensitivity is the square root of the sum of the squared sensitivities. Private training needs both, since one record's gradient moves every coordinate of a model at every step. So it clips each gradient's $\ell_2$ norm and calibrates Gaussian noise to that clip.

## Calibrating a release {#sec-lg-recipe}

Every noisy release takes five steps.

1. Fix the neighbouring relation and the unit of privacy, usually one person under add/remove.
2. Clip every value to a range fixed in advance, and read the sensitivity off @tbl-lg-sensitivity or work it out from the worst record.
3. Choose Laplace noise for one number or a sparse vector, and Gaussian noise for a dense vector.
4. Set the scale with @eq-lg-laplace or @eq-lg-analytic, and charge it to your budget ([Post-processing, group privacy and composition](02-properties.md)).
5. Draw the noise with a vetted library, and publish the error bar and the noise parameters with the result.

A mean under add/remove has no simple sensitivity, because its denominator moves too. Release instead a noisy sum and a noisy count, each with half of the budget, and divide one by the other. Dividing is post-processing, so it costs nothing more. The mean's error bar comes from theirs. Take each bar at 97.5%, so that both hold together 95% of the time. The mean then lies above the sum's lower end divided by the count's upper end. It lies below the sum's upper end divided by the count's lower end, provided both lower ends are positive. OpenDP's `.dp.mean` makes the same split into a sum and a count, and `summarize(alpha=0.025)` reports the two bars at 97.5%. When you do not know the range to clip to, [Private mean estimation](11-mean-estimation.md) shows how to find it privately.

Plain `numpy.random.laplace` is for prototypes only, because textbook floating-point samplers can leak through the low-order bits of their output ([Discrete and floating-point-safe mechanisms](08-discrete-mechanisms.md)). For a whole release with clipping, grouping and one budget, follow [Release statistics with a library](12-release-statistics.md).

## Code {#sec-lg-code}

The libraries do the calibration for you. In OpenDP, a mechanism's `map` takes the sensitivity and returns the $\varepsilon$ it costs. The accuracy helpers turn a scale into a 95% error bar or back, and since a count has sensitivity 1, one over the scale is the $\varepsilon$ of the error budget. IBM's diffprivlib calibrates the analytic Gaussian from $\varepsilon$, $\delta$ and the sensitivity. The snippet checks the hospital's numbers and a row of @tbl-lg-calibration.

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

To add the noise, call the mechanism on the value, as in `laplace(1234.0)`.

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
