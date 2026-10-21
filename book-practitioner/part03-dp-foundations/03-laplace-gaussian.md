# The Laplace and Gaussian mechanisms

## Why this matters

Suppose you run a hospital database and want to publish how many patients have diabetes with a differential privacy (DP) guarantee. The two mechanisms of this chapter are how almost every such release is built, up to the noisy gradients of private model training.

Both compute the exact answer of a query $f$ on the dataset $D$. They then add fresh random noise, written $Z$, so that the mechanism publishes
$$M(D) = f(D) + Z.$$
Three decisions fix the whole mechanism. The **sensitivity**, how far one record can move the answer, sets the scale of the noise. The noise distribution sets the kind of guarantee, since Laplace noise gives pure DP while Gaussian noise needs a small slack $\delta$. And a calibration formula sets the exact amount, through the privacy loss of [The definition of differential privacy](01-definition.md#sec-dpdef-divergences).

## Sensitivity {#sec-lg-sensitivity}

The noise must cover the largest change that one record can make to the answer. A query often returns a vector of $d$ numbers, such as the bins of a histogram, so you need a size for a change to a vector. Laplace noise uses the $\ell_1$ norm, the sum of the absolute changes. Gaussian noise uses the $\ell_2$ norm, the square root of the sum of the squared changes. Writing $D \simeq D'$ for a pair of neighbours, the **$\ell_p$ sensitivity** of the query is the largest change between their answers,
$$\Delta_pf = \max_{D \simeq D'}\big\|f(D) - f(D')\big\|_p, \qquad p \in \{1, 2\}.$$
The two are never far apart, since
$$\Delta_2f \le \Delta_1f \le \sqrt d\,\Delta_2f.$$ {#eq-lg-norms}
They are equal when one record moves a single coordinate.

The sensitivity also depends on the neighbouring relation. Under add/remove one dataset has one record more than the other, and under replace one record is swapped for another, so the size $n$ is public. Write $B$ for the largest value a record may take after clipping. @tbl-lg-sensitivity then gives the standard queries.

| Query | Add/remove $\Delta_1$ | Add/remove $\Delta_2$ | Replace $\Delta_1$ | Replace $\Delta_2$ |
| --- | --- | --- | --- | --- |
| Count $\lvert D\rvert$ | 1 | 1 | 0 | 0 |
| Sum, values in $[0, B]$ | $B$ | $B$ | $B$ | $B$ |
| Mean, values in $[0, B]$, $n$ public | — | — | $B/n$ | $B/n$ |
| Histogram with $k$ bins | 1 | 1 | 2 | $\sqrt2$ |
| $d$ threshold counts $\#\{x \ge j\}$, $j = 1..d$ | $d$ | $\sqrt d$ | $d$ | $\sqrt d$ |

: Sensitivities of the standard queries under the two neighbouring relations. {#tbl-lg-sensitivity}

A replaced record leaves one histogram bin and joins another, which is why the relation must be named. The threshold counts are a **dense** query, in which one record can move every coordinate.

Compute the sensitivity from the domain, before you look at the data. A sensitivity read off the data, for example from the largest value present, makes the noise itself leak that value ([Beyond global sensitivity](07-sensitivity.md)). For unbounded data the sensitivity is infinite, so clip every value to a fixed range first. For example, take the mean age of 1,000 patients, with ages clipped to at most 100. Under replace its sensitivity is 0.1, since no patient can move the mean by more than a tenth of a year.

## The Laplace mechanism {#sec-lg-laplace}

The Laplace mechanism (Dwork et al., 2006) adds independent noise from the **Laplace distribution** to each coordinate, a two-sided exponential peaked at 0. Its scale, written $b$, sets its typical size. To be $\varepsilon$-DP at a privacy level $\varepsilon$, the mechanism sets the scale to
$$b = \frac{\Delta_1f}{\varepsilon}.$$ {#eq-lg-laplace}

The error of the mechanism is the noise itself. Each coordinate is off by $b$ on average, and lands more than a distance $t$ from the truth with probability $e^{-t/b}$. To bound all $d$ coordinates at once, except with a small failure probability $\beta$, give each coordinate a share $\beta/d$ of it. The width that results is
$$\Pr\Big[\max_i|Z_i| > b\ln\frac d\beta\Big] \le \beta.$$
Publish this width with the release, and read it backwards as an error budget. Fix a target width $t$ for one number and solve for the privacy level, which gives
$$\varepsilon = \frac{\Delta_1f\,\ln(1/\beta)}{t}.$$
For example, a count that must be within 10 of the truth, with probability 0.95, needs an $\varepsilon$ of 0.2996.

At the hospital, 1,234 of the 10,000 patients have diabetes. A count has sensitivity 1 under add/remove, so at $\varepsilon = 0.1$ the noise is $\mathrm{Lap}(10)$. Its 95% error bar is 29.96, a relative error of 0.0243.

The scale does not contain the size of the data, so the relative error of a count of size $n$ is about $1/(n\varepsilon)$. But the sensitivity must include every number one record can move. If one record can change all of 10 released counts, noise calibrated for one count gives only $10\varepsilon$.

## The Gaussian mechanism {#sec-lg-gauss}

The Gaussian mechanism adds independent normal noise with mean 0 to each coordinate, with a standard deviation written $\sigma$. Its privacy loss is unbounded in the far tail, so it always carries a $\delta$. In exchange, it is calibrated to the $\ell_2$ sensitivity, which can be much smaller than the $\ell_1$ one. Only the ratio of the sensitivity to the noise matters, so write $\mu = \Delta_2f/\sigma$ for the number of noise standard deviations between the answers on two neighbours. Write $\Phi$ for the cumulative distribution function (CDF) of the standard normal distribution. The mechanism is then $(\varepsilon, \delta)$-DP exactly when $\delta$ is at least
$$\delta_\mu(\varepsilon) = \Phi\Big(\frac\mu2 - \frac\varepsilon\mu\Big) - e^\varepsilon\,\Phi\Big(-\frac\mu2 - \frac\varepsilon\mu\Big).$$ {#eq-lg-analytic}

The formula is how you calibrate. As $\mu$ grows, $\delta_\mu(\varepsilon)$ rises steadily from 0 to 1, so exactly one value $\mu^*$ hits your target $\delta$. The smallest noise that works is then
$$\sigma^* = \frac{\Delta_2f}{\mu^*}, \qquad \text{where } \delta_{\mu^*}(\varepsilon) = \delta,$$
and bisection finds $\mu^*$ in a few lines (@sec-lg-code). At the common setting of $\varepsilon = 1$ and $\delta = 10^{-5}$, this **analytic calibration** gives $\sigma^* = 3.7306\,\Delta_2f$.

Many papers and libraries still use the classic calibration of Dwork and Roth (2014). For $\varepsilon$ and $\delta$ both between 0 and 1, it sets
$$\sigma = \frac{\Delta_2f\sqrt{2\ln(1.25/\delta)}}{\varepsilon}.$$ {#eq-lg-classic}
@tbl-lg-calibration compares the two at $\delta = 10^{-5}$.

| $\varepsilon$ | Classic $\sigma/\Delta_2f$ | Analytic $\sigma^*/\Delta_2f$ | Ratio |
| --- | --- | --- | --- |
| 0.1 | 48.4481 | 30.7496 | 1.576 |
| 1 | 4.8448 | 3.7306 | 1.299 |
| 8 | 0.6056 | 0.6002 | 1.009 |

: Classic against analytic calibration of the Gaussian mechanism at $\delta = 10^{-5}$. {#tbl-lg-calibration}

For small $\varepsilon$ the classic formula adds 30–70% more noise than needed. For large $\varepsilon$ it is wrong, not just loose. At $\delta = 10^{-5}$ the two calibrations cross at $\varepsilon = 8.42$, and beyond it the classic $\sigma$ is too small to give the claimed guarantee.

A Gaussian mechanism satisfies a whole curve of guarantees, one $\delta$ for each $\varepsilon$, and the accountants of [Part 4](../part04-privacy-accounting/00-introduction.md) compose whole curves. So report $\sigma/\Delta_2f$ with every Gaussian release, as well as the pair you chose. Accountants that work in zero-concentrated DP take the equivalent $\rho = \Delta_2f^2/(2\sigma^2)$ ([Concentrated differential privacy](../part04-privacy-accounting/03-concentrated-dp.md)).

For the diabetes count at the same $\varepsilon$ and $\delta = 10^{-5}$, the 95% error bar is 1.96 standard deviations, or 60.27. Laplace noise gave 29.96 with no $\delta$ at all, so for a single count it wins twice.

## Laplace or Gaussian? {#sec-lg-choose}

The choice depends on how one record spreads its change across the coordinates, as the root-mean-square error (RMSE) of each released number shows. Write $\sigma_1$ for the analytic noise level at unit $\ell_2$ sensitivity. At the same $\varepsilon$, the errors per coordinate are then
$$\text{Laplace: } \frac{\sqrt2\,\Delta_1f}{\varepsilon}, \qquad \text{Gaussian: } \Delta_2f\,\sigma_1.$$

In a sparse query, such as a histogram under add/remove, one record moves one coordinate. The two sensitivities are then equal, so at the common setting Laplace noise wins at every $d$. In a dense query, by contrast, one record moves every coordinate. The Laplace error then grows like $d$ while the Gaussian error grows only like $\sqrt d$, so Laplace noise is more accurate only while
$$d < \frac{(\varepsilon\sigma_1)^2}2.$$
At the common setting, the unit noise level is 3.7306, so for threshold counts Gaussian noise wins from 7 counts on.

Gaussian noise also wins when releases are composed many times. Several Gaussian releases compose exactly into one Gaussian mechanism, whose $\ell_2$ sensitivity is the square root of the sum of the squared sensitivities. Private training needs both advantages. One example's gradient moves every coordinate of a model, so private training clips each gradient's $\ell_2$ norm and calibrates Gaussian noise to that clip.

## Calibrating a release {#sec-lg-recipe}

Every noisy release takes five steps.

1. Fix the neighbouring relation and the unit of privacy, usually one person under add/remove.
2. Clip every value to a range fixed in advance, and read the sensitivity off @tbl-lg-sensitivity or work it out from the worst record.
3. Choose Laplace noise for one number or a sparse vector, and Gaussian noise for a dense vector.
4. Set the scale with @eq-lg-laplace or @eq-lg-analytic, and charge it to your budget ([Post-processing, group privacy and composition](02-properties.md)).
5. Draw the noise with a vetted library, and publish the error bar and the noise parameters with the result.

A mean under add/remove has no simple sensitivity, because its denominator moves too. Release instead a noisy sum and a noisy count, each with half of the budget, and divide one by the other. Dividing is post-processing, so it costs nothing more. When you do not know the range to clip to, [Private mean estimation](11-mean-estimation.md) shows how to find it privately.

In OpenDP, `dp.t.make_clamp` and `dp.t.make_sum` build the clipped query, and `dp.m.make_laplace` or `dp.m.make_gaussian` adds noise of a given scale. Its `dp.binary_search_param` then finds the scale for a target $\varepsilon$. IBM's diffprivlib has `mechanisms.Laplace` and `mechanisms.GaussianAnalytic`, whose `.randomise(x)` adds the noise, and `tools.mean(x, epsilon=..., bounds=...)` for a clipped mean. Plain `numpy.random.laplace` is for prototypes only, because textbook floating-point samplers can leak through the low-order bits of their output ([Discrete and floating-point-safe mechanisms](08-discrete-mechanisms.md)).

## Code {#sec-lg-code}

The analytic calibration needs only the normal CDF, which `math.erfc` provides. The snippet also computes Laplace error bars.

```python
from math import erfc, exp, log, sqrt


def normal_cdf(x: float) -> float:
    return 0.5 * erfc(-x / sqrt(2))


def gaussian_delta(eps: float, mu: float) -> float:
    """The exact delta of the Gaussian mechanism at eps, where mu = sensitivity / sigma."""
    return normal_cdf(mu / 2 - eps / mu) - exp(eps) * normal_cdf(-mu / 2 - eps / mu)


def analytic_sigma(eps: float, delta: float, sensitivity: float = 1.0) -> float:
    """The smallest sigma for which the Gaussian mechanism is (eps, delta)-DP."""
    lo, hi = 1e-6, 100.0  # gaussian_delta grows with mu: bisect for gaussian_delta(eps, mu) = delta
    for _ in range(60):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if gaussian_delta(eps, mid) < delta else (lo, mid)
    return sensitivity / lo


def laplace_width(scale: float, beta: float, d: int = 1) -> float:
    """All d coordinates of Lap(scale) noise stay within this width, except with probability beta."""
    return scale * log(d / beta)


print(f"{analytic_sigma(1.0, 1e-5):.4f}")  # 3.7306
print(f"{analytic_sigma(0.1, 1e-5):.4f}")  # 30.7496
print(f"{laplace_width(1 / 0.1, 0.05):.2f}")  # 29.96
print(f"{log(1 / 0.05) / 10:.4f}")  # 0.2996
```

## Cheat sheet

| Result | Formula | Notes |
| --- | --- | --- |
| Laplace mechanism | $\mathrm{Lap}(\Delta_1f/\varepsilon)$ per coordinate | Pure DP; error does not depend on $n$ |
| Laplace error bar | $b\ln(d/\beta)$ for all $d$ coordinates | Solve it for $\varepsilon$ to budget error |
| Exact Gaussian $\delta$ | $\Phi(\frac\mu2 - \frac\varepsilon\mu) - e^\varepsilon\Phi(-\frac\mu2 - \frac\varepsilon\mu)$, $\mu = \Delta_2f/\sigma$ | Solve for $\sigma$ by bisection |
| Classic Gaussian | $\sigma = \Delta_2f\sqrt{2\ln(1.25/\delta)}/\varepsilon$ | Too much noise below 1, unsafe above 8.42 |
| Dense query | Laplace wins only for $d < (\varepsilon\sigma_1)^2/2$ | Otherwise use Gaussian noise |

: The results of this chapter. {#tbl-lg-cheat}

**Rules of thumb**

- Compute the sensitivity from the domain, not the data.
- Calibrate Laplace noise to $\Delta_1f$ and Gaussian noise to $\Delta_2f$, never the other way round.
- Use the analytic Gaussian calibration, never the classic formula.
- Report $\sigma/\Delta_2f$ for a Gaussian release, and an error bar for every release.
- When $n$ is not public, never use $B/n$ as the sensitivity of a mean.

## Further reading

- Dwork & Roth, *The Algorithmic Foundations of Differential Privacy* (2014), §3.3 and Appendix A: the Laplace mechanism and the classic Gaussian calibration.
- Balle & Wang, "Improving the Gaussian Mechanism for Differential Privacy" (ICML 2018): the exact privacy curve and the bisection calibration.

**Derivations.** Every result here is derived step by step in [the full edition's chapter](../../book-full/part03-dp-foundations/03-laplace-gaussian.md).
