# The Laplace and Gaussian mechanisms

## Why this matters

Suppose a hospital wants to publish how many of its patients have diabetes, with a differential privacy (DP) guarantee. Almost every such release, and every noisy gradient of private training, is built with one of the two mechanisms of this chapter. Both compute the exact answer of a query $f$ on the dataset $D$, then add fresh random noise, written $Z$, and publish
$$M(D) = f(D) + Z.$$
Three decisions fix such a mechanism. The **sensitivity**, how far one record can move the answer, is what the noise must hide. The noise distribution sets the kind of guarantee: Laplace noise gives pure DP, while Gaussian noise needs a small slack $\delta$. And a calibration formula sets the amount of noise for a target $\varepsilon$, through the privacy loss of [the definition of differential privacy](01-definition.md#sec-dpdef-divergences).

## Sensitivity {#sec-lg-sensitivity}

The noise must cover the largest change one record can make to the answer. A query often returns a vector of $d$ numbers, such as the bins of a histogram, so the change needs a size. Laplace noise uses the $\ell_1$ norm, the sum of the absolute changes, and Gaussian noise the $\ell_2$ norm, the square root of the sum of the squared changes. Writing $D \sim D'$ for a pair of neighbours, the **$\ell_p$ sensitivity** of the query is the largest change between their answers,
$$\Delta_pf = \max_{D \sim D'}\big\|f(D) - f(D')\big\|_p, \qquad p \in \{1, 2\}.$$
The two are equal when one record moves a single coordinate, but when it moves all $d$ of them, $\Delta_1f$ can be $\sqrt d$ times $\Delta_2f$.

The sensitivity also depends on the neighbouring relation. Under add/remove one dataset has one record more than the other, while under replace one record is swapped for another, so the size $n$ is public. Write $B$ for the largest value a record may take after clipping. Then @tbl-lg-sensitivity gives the standard queries.

| Query | Add/remove $\Delta_1$ | Add/remove $\Delta_2$ | Replace $\Delta_1$ | Replace $\Delta_2$ |
| --- | --- | --- | --- | --- |
| Count $\lvert D\rvert$ | 1 | 1 | 0 | 0 |
| Sum, values in $[0, B]$ | $B$ | $B$ | $B$ | $B$ |
| Mean, values in $[0, B]$, $n$ public | none (@sec-lg-recipe) | none | $B/n$ | $B/n$ |
| Histogram with $k$ bins | 1 | 1 | 2 | $\sqrt2$ |
| $d$ threshold counts $\#\{x \ge j\}$, $j = 1..d$ | $d$ | $\sqrt d$ | $d$ | $\sqrt d$ |

: Sensitivities of the standard queries under the two neighbouring relations. {#tbl-lg-sensitivity}

A replaced record leaves one histogram bin and joins another, so it changes two bins. The threshold counts, the numbers of records at or above each of $d$ values, are a **dense** query, in which one record can move every coordinate.

Compute the sensitivity from the domain, before you look at the data. A sensitivity read off the data, such as the largest value present, makes the noise leak that value ([Beyond global sensitivity](07-sensitivity.md)). Unbounded data have infinite sensitivity, so clip every value to a fixed range first. With ages clipped to at most 100, the mean age of 1,000 patients has sensitivity $B/n = 0.1$ under replace.

## The Laplace mechanism {#sec-lg-laplace}

The Laplace mechanism adds independent noise from the **Laplace distribution**, a two-sided exponential peaked at 0, to each coordinate. Its scale $b$ sets its typical size, and $\mathrm{Lap}(b)$ denotes it. For an $\varepsilon$-DP guarantee, the scale is the sensitivity divided by $\varepsilon$,
$$b = \frac{\Delta_1f}{\varepsilon}.$$ {#eq-lg-laplace}

The error of the mechanism is the noise itself. Each coordinate of the noise is off by $b$ on average, and exceeds a distance $t$ with probability $e^{-t/b}$. Allow a small failure probability $\beta$ for all $d$ coordinates together, and give each coordinate a share $\beta/d$ of it. The width that holds for all of them is then
$$\Pr\Big[\max_i|Z_i| > b\ln\frac d\beta\Big] \le \beta.$$
Publish this width with the release. Read backwards, it is an error budget: for a single number, fix a target width $t$ and solve for the privacy level,
$$\varepsilon = \frac{\Delta_1f\,\ln(1/\beta)}{t}.$$
So a count that must be within 10 of the truth with probability 0.95 needs $\varepsilon = \ln(20)/10 = 0.2996$. For $d$ numbers, solve each with the failure probability $\beta/d$, and add up their $\varepsilon$ by [composition](02-properties.md#sec-dpprop-comp).

At the hospital, 1,234 of the 10,000 patients have diabetes. A count has sensitivity 1 under add/remove, so at $\varepsilon = 0.1$ the noise is $\mathrm{Lap}(10)$. Its 95% error bar is $10\ln 20 = 29.96$, or 2.43% of the count.

The scale does not depend on the size of the data, so the relative error of a count of size $n$ is about $1/(n\varepsilon)$. But the sensitivity must include every number one record can move. If one record can change all of 10 released counts, noise calibrated for one count makes the release $10\varepsilon$-DP, not $\varepsilon$-DP.

## The Gaussian mechanism {#sec-lg-gauss}

The Gaussian mechanism adds independent normal noise with mean 0 and standard deviation $\sigma$ to each coordinate. Its privacy loss is unbounded in the far tail, so it always carries a $\delta$. In exchange, it is calibrated to the $\ell_2$ sensitivity, which can be much smaller than the $\ell_1$ one. Only the ratio of the sensitivity to the noise matters, so write $\mu = \Delta_2f/\sigma$ for the number of noise standard deviations between the answers on two neighbours. Write $\Phi$ for the cumulative distribution function (CDF) of the standard normal distribution. The mechanism is then $(\varepsilon, \delta)$-DP exactly when $\delta$ is at least
$$\delta_\mu(\varepsilon) = \Phi\Big(\frac\mu2 - \frac\varepsilon\mu\Big) - e^\varepsilon\,\Phi\Big(-\frac\mu2 - \frac\varepsilon\mu\Big).$$ {#eq-lg-analytic}

This formula is how you calibrate. As $\mu$ grows, the exact $\delta$ rises steadily from 0 to 1, so exactly one value $\mu^*$ hits your target. The smallest noise that works is then
$$\sigma^* = \frac{\Delta_2f}{\mu^*}, \qquad \text{where } \delta_{\mu^*}(\varepsilon) = \delta,$$
and libraries solve for $\mu^*$ for you (@sec-lg-code). For 10,000 patients the book's default slack is $\delta = 10^{-5}$ ([The definition of differential privacy](01-definition.md#sec-dpdef-small-delta)). At $\varepsilon = 1$ this **analytic calibration** then gives $\sigma^* = 3.7306\,\Delta_2f$.

Many papers and libraries still use the classic calibration of Dwork and Roth (2014), proved only for $\varepsilon$ and $\delta$ both between 0 and 1,
$$\sigma = \frac{\Delta_2f\sqrt{2\ln(1.25/\delta)}}{\varepsilon}.$$ {#eq-lg-classic}
At $\varepsilon = 1$ and $\delta = 10^{-5}$ it gives $4.8448\,\Delta_2f$, 30% more noise than the analytic $3.7306\,\Delta_2f$. Above $\varepsilon = 1$ it has no proof behind it, and beyond $\varepsilon = 8.42$ it gives too little noise for the claimed guarantee. So use the analytic calibration.

A Gaussian mechanism satisfies a whole curve of guarantees, one $\delta$ for each $\varepsilon$, and the accountants of [Part 4](../part04-privacy-accounting/00-introduction.md) compose whole curves. So report $\sigma/\Delta_2f$ with every Gaussian release, as well as the pair you chose.

Back at the hospital, Gaussian noise for the diabetes count at $\varepsilon = 0.1$ and $\delta = 10^{-5}$ gives a 95% error bar of 60.27, which is 1.96 standard deviations. Laplace noise gave 29.96 with no $\delta$ at all, so for a single count it wins on both error and guarantee.

## Laplace or Gaussian? {#sec-lg-choose}

The choice depends on how one record spreads its change across the coordinates. To see why, compare the root-mean-square error (RMSE) of each released number at the same $\varepsilon$. Laplace noise of scale $b = \Delta_1f/\varepsilon$ has standard deviation $\sqrt2\,b$. Write $\sigma_1$ for the analytic Gaussian noise level at unit $\ell_2$ sensitivity. The errors per coordinate are then
$$\text{Laplace: } \frac{\sqrt2\,\Delta_1f}{\varepsilon}, \qquad \text{Gaussian: } \Delta_2f\,\sigma_1.$$

In a sparse query, such as a histogram under add/remove, one record moves one coordinate, so the two sensitivities are equal. At $\varepsilon = 1$ and the default $\delta$, $\sigma_1 = 3.7306$, so Laplace noise wins at every $d$, since $\sqrt2 = 1.4142$ is smaller. In a dense query, such as the threshold counts, one record moves every coordinate, so the Laplace error grows like $d$ and the Gaussian error only like $\sqrt d$. Laplace noise is then more accurate only while
$$d < \frac{(\varepsilon\sigma_1)^2}2,$$
so with $\sigma_1 = 3.7306$, Gaussian noise wins for threshold counts from 7 counts on.

Gaussian noise also wins under heavy composition, because several Gaussian releases compose exactly into one Gaussian mechanism, with the root of the summed squared sensitivities. Private training has both, since one record's gradient moves every coordinate of a model at every step. So it clips each gradient's $\ell_2$ norm and calibrates Gaussian noise to that clip.

## Calibrating a release {#sec-lg-recipe}

Every noisy release takes five steps.

1. Fix the neighbouring relation and the unit of privacy, usually one person under add/remove.
2. Clip every value to a range fixed in advance, and read the sensitivity off @tbl-lg-sensitivity or work it out from the worst record.
3. Choose Laplace noise for one number or a sparse vector, and Gaussian noise for a dense vector.
4. Set the scale with @eq-lg-laplace or @eq-lg-analytic, and charge it to your budget ([Post-processing, group privacy and composition](02-properties.md)).
5. Draw the noise with a vetted library, and publish the error bar and the noise parameters with the result.

A mean under add/remove has no simple sensitivity, because its denominator moves too. So release a noisy sum and a noisy count, each with half of the budget, and divide one by the other, which as post-processing costs nothing more. For the mean's error bar, take each one's bar at 97.5%, so that both hold together 95% of the time. The mean then lies above the sum's lower end divided by the count's upper end. It lies below the sum's upper end divided by the count's lower end, provided both lower ends are positive. OpenDP's `.dp.mean` makes the same split, and `summarize(alpha=0.025)` reports the two bars at 97.5%. When you do not know the range to clip to, [Private mean estimation](11-mean-estimation.md) finds it privately.

Plain `numpy.random.laplace` is for prototypes only, because textbook floating-point samplers can leak through the low-order bits of their output ([Discrete and floating-point-safe mechanisms](08-discrete-mechanisms.md)). For a whole release with clipping, grouping and one budget, follow [Release statistics with a library](12-release-statistics.md).

## Code {#sec-lg-code}

The snippet checks the hospital's numbers and the analytic $\sigma^*$ at $\varepsilon = 1$. In OpenDP, a mechanism's `map` takes the sensitivity and returns the $\varepsilon$ it costs, and the accuracy helpers turn a scale into a 95% error bar or back. For a count, with sensitivity 1, one over that scale is the $\varepsilon$ of the error budget. IBM's diffprivlib calibrates the analytic Gaussian.

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
