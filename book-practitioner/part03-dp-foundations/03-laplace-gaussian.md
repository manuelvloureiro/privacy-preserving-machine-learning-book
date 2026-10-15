# The Laplace and Gaussian mechanisms

## Why this matters

Suppose you run a hospital database and want to publish how many patients have diabetes, the average age of a cohort, or a histogram of diagnoses. Each release should come with a differential privacy (DP) guarantee. [The definition of differential privacy](01-definition.md) says what such a release must satisfy, namely that changing one person's record barely changes the distribution of what you publish. The definition does not say how to build a release that meets it, and the two mechanisms of this chapter are the answer used almost everywhere.

Both mechanisms work the same way. They compute the exact answer of a query $f$ on a dataset $D$, written $f(D)$. Then they blur it with random noise, written $Z$ and drawn fresh for every release, so that the mechanism $M$ publishes
$$M(D) = f(D) + Z.$$
An observer then cannot tell from the output whether a given person was in the data, provided the noise is wide compared with that person's effect on the answer.

Three decisions then fix the whole mechanism. The first is how far one record can move the answer, which is called the sensitivity and sets the scale of the noise. The second is which distribution the noise comes from, and here the two mechanisms differ. Laplace noise gives the strongest form of the guarantee, with no slack at all, while Gaussian noise allows a small slack. In exchange, Gaussian noise scales better when you release many numbers at once. The third decision is exactly how much noise to add, which a one-line formula answers for Laplace noise. Gaussian noise, by contrast, is often calibrated with a classic formula that adds more noise than needed, while the exact amount comes from solving one equation numerically. Every later mechanism that adds noise, up to the noisy gradients of private model training, is calibrated with the tools of this chapter.

## What added noise does to the privacy loss {#sec-lg-loss}

The guarantee compares two **neighbouring** datasets, which differ in one person's record. It asks that no set of outputs become much more likely when that one record changes. The allowed factor is written $e^\varepsilon$, for a privacy parameter $\varepsilon$, and a small additive slack, written $\delta$, is allowed on top of it.

To state it precisely, write $D'$ for a neighbour of $D$. A mechanism is $(\varepsilon, \delta)$-DP if, for every pair of neighbours, every set $S$ of outputs satisfies
$$\Pr[M(D) \in S] \le e^\varepsilon\Pr[M(D') \in S] + \delta.$$
Here the probabilities are over the noise, with the data fixed. When $\delta = 0$, the guarantee is called **pure DP**.

The measure behind the definition is the **privacy loss** of an output $y$. Write $p_{M(D)}$ for the density of the published output when the data are $D$. The privacy loss of $y$ is the logarithm of how much more likely it is under $D$ than under its neighbour,
$$L(y) = \ln\frac{p_{M(D)}(y)}{p_{M(D')}(y)}.$$
A loss of 0 means that the output is equally likely under both datasets and says nothing about which one was used. A loss of $\varepsilon$ means that the output is $e^\varepsilon$ times more likely under $D$.

For additive noise the privacy loss has a simple shape. On the two neighbours the mechanism publishes $f(D) + Z$ and $f(D') + Z$, which is the same noise placed around two different centres. So the output distribution on $D'$ is the one on $D$ slid over by the **shift** between the two answers, written $v = f(D') - f(D)$. The true answer itself drops out. For example, take a count that is 1,234 on one dataset and 1,235 on its neighbour. Subtracting 1,234 from every published value is something anyone can do, and it turns the two releases into $Z$ and $1 + Z$. Since the subtraction can also be undone, the original release hides the difference exactly as well as this simpler one. Privacy is therefore a property of the noise and of the shifts that neighbouring datasets can produce, and of nothing else.

The privacy loss is then the log-ratio of the noise density and a copy of itself moved by the shift. How it behaves depends on how fast the log of the density falls off. The log of a Laplace density falls off like $-|z|$, a straight line on each side. Sliding the density over changes its log by at most the shift divided by the noise scale $b$, so the loss is bounded at every output. The log of a normal density falls off like $-z^2$, a parabola. The difference of two shifted parabolas is a straight line, so the loss grows without limit, but it is large only far in the tail, where outputs are rare. @tbl-lg-shifted-pairs shows both for a count, where the shift is 1 and the two centres are 0 and 1.

| Noise | Privacy loss $L(y)$ for centres 0 and 1 | Largest value | Guarantee |
| --- | --- | --- | --- |
| $\mathrm{Lap}(b)$ | $(\lvert y - 1\rvert - \lvert y\rvert)/b$ | $1/b$, for every $y \le 0$ | $\varepsilon$-DP with $\varepsilon = 1/b$ |
| $\mathcal N(0, \sigma^2)$ | $(1 - 2y)/(2\sigma^2)$ | unbounded as $y \to -\infty$ | $(\varepsilon, \delta)$-DP, with $\delta$ set by the tail where $L > \varepsilon$ |

: The privacy loss of a count released with Laplace noise of scale $b$ and with Gaussian noise of standard deviation $\sigma$. {#tbl-lg-shifted-pairs}

A bounded loss gives pure DP. No set of outputs can favour $D$ by more than a factor $e^\varepsilon$ when no single output does. The reason is that a set's probability is the sum of its outputs' probabilities, and each term on $D$ is at most $e^\varepsilon$ times the matching term on $D'$. The bound must hold at every output and for every pair of neighbours. If it fails on outputs that carry some probability, those outputs together form a set that breaks pure DP, and only a $\delta$ can cover them. The Laplace mechanism is proved private exactly this way.

An unbounded loss needs $\delta$. Call the outputs whose loss exceeds $\varepsilon$ the **bad event** $B$. Write $P(B)$ for its probability on $D$ and $Q(B)$ for its probability on the neighbour. As [the definition](01-definition.md) explained, each output may be up to $e^\varepsilon$ times more likely on $D$ for free, and $\delta$ pays only for the excess. Only the bad outputs have an excess, and each one's excess is its probability on $D$ minus $e^\varepsilon$ times its probability on the neighbour. Adding these excesses over the bad event gives the smallest $\delta$ that works for this pair. This total excess is called the **hockey-stick divergence** between the two output distributions, and it satisfies
$$P(B) - e^\varepsilon Q(B) \le P(B) = \Pr[L > \varepsilon].$$ {#eq-lg-bad-event}

The inequality drops the subtracted term, which can never be negative. So the probability that the loss exceeds $\varepsilon$ is always a valid $\delta$, but usually a loose one. The classic Gaussian calibration of @sec-lg-gauss-classic rests on this looser bound, and the exact one of @sec-lg-gauss rests on the difference.

The same exact $\delta$ is also an average over the outputs of $M(D)$. An output with loss $L$ is $e^L$ times more likely on $D$ than on the neighbour. Its excess is therefore the fraction $1 - e^{\varepsilon - L}$ of its probability, or nothing when that fraction is negative. Averaging this payment needs only the distribution of the privacy loss, which is how [Numerical accounting](../part04-privacy-accounting/06-numerical-accounting.md) computes $\delta$. Finally, the smallest $\delta$ that works for every neighbouring pair depends on the $\varepsilon$ you ask about. As a function of $\varepsilon$, it is called the **privacy curve** of the mechanism.

## Sensitivity {#sec-lg-sensitivity}

To hide one record, the noise has to cover the largest change that one record can make to the answer. That change is the **sensitivity** of the query. It depends on the query, on how you measure a change, and on the neighbouring relation.

A query often returns a vector of $d$ numbers, such as the bins of a histogram. You then need a way to measure the size of a change to a vector, and two are used. The $\ell_1$ norm adds up the absolute changes over all coordinates, and Laplace noise is calibrated to it. The $\ell_2$ norm is the straight-line length of the change, the square root of the sum of squared changes, and Gaussian noise is calibrated to it.

The **$\ell_p$ sensitivity** of the query is the largest change between the answers on any two neighbours, measured in one of these norms. Writing $D \simeq D'$ for a pair of neighbours, it is
$$\Delta_pf = \max_{D \simeq D'}\big\|f(D) - f(D')\big\|_p, \qquad p \in \{1, 2\}.$$

The two norms are never far apart, and how far depends on how one record spreads its change. The $\ell_2$ norm is never larger than the $\ell_1$ norm, because squaring the sum of absolute changes gives every squared change plus cross terms that cannot be negative. In the other direction, the Cauchy–Schwarz inequality shows that the $\ell_1$ norm is at most $\sqrt d$ times the $\ell_2$ norm. Taking the worst case over neighbours gives
$$\Delta_2f \le \Delta_1f \le \sqrt d\,\Delta_2f.$$ {#eq-lg-norms}
The two sensitivities are equal when one record moves a single coordinate. The right end is reached when one record moves all $d$ coordinates by the same amount. A change of 1 in each coordinate, for example, adds up to $d$ but has a straight-line length of only $\sqrt d$.

The neighbouring relation says what "differ in one record" means. Under **add/remove**, one dataset has one record more than the other. The guarantee then hides whether a person is present at all, and with it the size of the dataset. Under **replace**, one record is swapped for another, so the size $n$ is fixed and public. @tbl-lg-sensitivity gives the sensitivities of the standard queries, for records with values in a range from 0 to $B$. The histogram there has $k$ bins, and the threshold counts are explained below the table.

| Query | Add/remove $\Delta_1$ | Add/remove $\Delta_2$ | Replace $\Delta_1$ | Replace $\Delta_2$ |
| --- | --- | --- | --- | --- |
| Count $\lvert D\rvert$ | 1 | 1 | 0 | 0 |
| Sum, records in $[0, B]$ | $B$ | $B$ | $B$ | $B$ |
| Mean, records in $[0, B]$, $n$ public | — | — | $B/n$ | $B/n$ |
| Histogram with $k$ bins | 1 | 1 | 2 | $\sqrt2$ |
| $d$ threshold counts $\#\{x \ge j\}$, $j = 1..d$ | $d$ | $\sqrt d$ | $d$ | $\sqrt d$ |

: Sensitivities of the standard queries under the two neighbouring relations. {#tbl-lg-sensitivity}

Each entry comes from asking what the single worst record can do. Adding or removing a record changes a count by exactly 1. Replacing a record leaves the count unchanged, which is why the count of all records needs no noise under replace. A record with a value up to $B$ moves a sum by at most $B$, whether it is added, removed or swapped for another record in the range. Under replace, the mean is the sum divided by the public size $n$, so its sensitivity is $B/n$. Under add/remove the denominator moves too, so there is no such simple bound. You can instead release a noisy sum and a noisy count, each with half of the privacy budget, and divide one by the other. Dividing is post-processing, so it costs nothing more.

A histogram shows the two relations differing. Under add/remove, the new record raises one bin by 1, so both norms of the change are 1. Under replace, the swapped record leaves one bin and joins another, so one bin falls by 1 and another rises by 1. That change has $\ell_1$ norm 2 and $\ell_2$ norm $\sqrt2$. The last row is a **dense** query, one in which a single record can move every coordinate. A threshold count is the number of records at or above a threshold $j$. So a record at the top of the range raises every one of the $d$ counts by 1.

Sensitivity is a worst case over every dataset the domain allows, not a property of the data you hold. So you can compute it before looking at the data, and you should. If you computed it from your data, for example from the largest value present, the noise itself would leak information about that value ([Beyond global sensitivity](06-sensitivity.md)). The same worst case makes the sensitivity infinite for unbounded data, because one record in a sum of arbitrary numbers can be as large as it likes. Clipping every value to a fixed range forces a finite $B$.

Since the noise is proportional to the sensitivity, a guarantee stated without its neighbouring relation is ambiguous by up to a factor of 2 in $\varepsilon$. The histogram's $\ell_1$ sensitivity shows that factor.

As a worked example, take the **cohort mean**. A study releases the mean age of $n = 1000$ patients, with ages clipped to the range from 0 to 100, under replace neighbours. One patient can move the sum by at most 100, so the sensitivity of the mean is $100/1000 = 0.1$. However old a patient is, they move the mean by at most a tenth of a year.

## The Laplace mechanism {#sec-lg-laplace}

The Laplace mechanism is the first mechanism to try (Dwork et al. 2006). It draws its noise from the **Laplace distribution**, a two-sided exponential peaked at 0, whose scale, written $b$, sets its typical size. The Laplace distribution with scale $b$, written $\mathrm{Lap}(b)$, has the density
$$\frac1{2b}e^{-|z|/b}.$$
The mechanism adds independent $\mathrm{Lap}(b)$ noise to each of the $d$ coordinates of the answer. It chooses the scale so that the largest shift one record can cause, divided by the scale, is exactly $\varepsilon$. That scale is
$$b = \frac{\Delta_1f}{\varepsilon},$$ {#eq-lg-laplace}
and with it the mechanism is $\varepsilon$-DP.

The reason is the straight-line log of the Laplace density. Take one coordinate, and compare the density at an output $y$ with its copy moved by a shift $s$. Their log-ratio is the difference of the distances from $y$ to the two centres, divided by $b$. By the triangle inequality, that difference is at most the distance between the centres, $|s|$, whatever $y$ is. With independent noise the log-ratios of the coordinates add, so the total loss is at most the $\ell_1$ norm of the shift divided by $b$. That is at most the sensitivity divided by the scale, which is $\varepsilon$, and a bounded loss gives pure DP. The argument uses independence across coordinates, which is why the mechanism draws a fresh value for each one.

No smaller scale works, and the worst case is not rare. Take a count with true answers 0 and 1, and look at the outputs at or below 0, to the left of both centres. Because both log-densities fall in parallel straight lines out there, moving further left multiplies both densities by the same factor. So their ratio stays fixed at $e^{1/b}$, and the loss equals the bound exactly (@fig-lg-privacy-loss, left). Under $M(D)$ these outputs carry probability $\frac12$, by the symmetry of the noise. If $b$ were smaller than $1/\varepsilon$, this event would be more than $e^\varepsilon$ times as likely on one dataset as on the other, and the definition would fail.

![Left: a count released with $\mathrm{Lap}(1)$ noise on neighbouring datasets with true counts 0 and 1; the privacy loss is $+\varepsilon$ for every $y \le 0$ and $-\varepsilon$ for every $y \ge 1$, never more. Right: the same count with $\mathcal N(0, 2^2)$ noise; the loss is a straight line and exceeds $\varepsilon = 1$ on the shaded tail, of probability 0.0401, while the exact $\delta(1)$ is only 0.0068.](figures/lapgauss_privacy_loss.svg){#fig-lg-privacy-loss}

The error of the mechanism is the noise itself, so its accuracy follows from the Laplace distribution ([Concentration inequalities](../part01-prerequisites/02-concentration.md)). Each coordinate's noise $Z_i$ is off by $b$ on average, and its variance is $2b^2$. Its tail has an exact formula, since the chance that it lands more than a distance $t$ from zero is $e^{-t/b}$. Suppose you want to bound all $d$ coordinates at once, and allow the bound to fail with a small probability $\beta$. Give each coordinate a share $\beta/d$ of that failure probability. The union bound then says that at least one coordinate fails with probability at most $d$ times $\beta/d$, which is $\beta$. Setting the tail equal to $\beta/d$ gives the width, so that
$$\Pr\Big[\max_i|Z_i| > b\ln\frac d\beta\Big] \le \beta.$$

Here is the mechanism on the hospital's data, where 1,234 of the 10,000 patients have diabetes. Under add/remove a count has sensitivity 1, so a release at $\varepsilon = 0.1$ needs a scale ten times the sensitivity. The noise is therefore $\mathrm{Lap}(10)$, and its expected absolute error is the scale itself. Its 95% guarantee comes from the accuracy bound with a single coordinate and $\beta = 0.05$. So with probability 0.95, the error is below $10\ln20 = 29.96$. Against the true count of 1,234, that is a relative error of 0.0243.

Now take a histogram of diagnoses with 20 bins. Under add/remove its sensitivity is 1, the same as a single count's, because each patient falls in one bin. So at $\varepsilon = 1$ each bin gets $\mathrm{Lap}(1)$ noise. Sharing a failure probability of 0.05 over the bins gives the width $\ln(20/0.05) = 5.99$. So every bin is within that width of its true count with probability 0.95. The bound is nearly exact, because the union bound adds the failure probabilities as if failures never happened together, and for independent coordinates they rarely do.

The cohort mean of @sec-lg-sensitivity has sensitivity 0.1, so at $\varepsilon = 1$ its scale equals its sensitivity. With probability 0.95, the released mean is then within $0.1\ln20 = 0.2996$ years of the truth.

Notice what the scale does not contain, namely the size of the data. One record moves a count by 1 whether the database holds a hundred records or a hundred million, so the noise is the same. The relative error of a count of size $n$ is therefore about $1/(n\varepsilon)$. Large populations are cheap to protect, and small subgroups are expensive.

The flip side is that the sensitivity must include every number one record can move. Suppose you release 10 counts, each with $\mathrm{Lap}(1/\varepsilon)$ noise, and one record can change all of them. The vector then has an $\ell_1$ sensitivity of 10, so the release is only $10\varepsilon$-DP. That matches basic composition in [Post-processing, group privacy and composition](02-properties.md), which adds up the $\varepsilon$ of each release.

A Laplace mechanism also satisfies weaker guarantees at smaller values of $\varepsilon$. Take a smaller value, written $\varepsilon'$. The mechanism is not $\varepsilon'$-DP, but it is $(\varepsilon', \delta)$-DP with
$$\delta = 1 - e^{-(\varepsilon - \varepsilon')/2}.$$
This is the hockey-stick divergence of two Laplace densities whose centres are the full sensitivity apart, which is $\varepsilon$ times the scale. Between the two centres the loss falls in a straight line, from $\varepsilon$ down to $-\varepsilon$. So some of those outputs have a loss above $\varepsilon'$, and $\delta$ pays for them. The required $\delta$ falls to zero as $\varepsilon'$ rises to $\varepsilon$, and @fig-lg-delta-curves draws it as the dashed curve.

Other noise shapes would work too, because the argument used only the bounded slope of the log-density. Some do slightly better when $\varepsilon$ is large, such as the staircase mechanism of Geng and Viswanath, whose density falls in steps. But the Laplace mechanism is within a small factor of them and is the standard choice.

## The Gaussian mechanism {#sec-lg-gauss}

Gaussian noise is the other natural choice. It has lighter tails than Laplace noise and, above all, it is calibrated to the $\ell_2$ sensitivity, which can be much smaller than the $\ell_1$ sensitivity. The **Gaussian mechanism** adds independent normal noise to each coordinate, with mean 0 and a standard deviation written $\sigma$. Because its privacy loss is unbounded, it cannot give pure DP, and its guarantee always carries a $\delta$.

The privacy loss of Gaussian noise depends only on the length of the shift. The reason is that the normal distribution with independent, equal-variance coordinates looks the same in every direction. So you can rotate the axes until the shift points along one of them, and the other coordinates then have the same distribution under both datasets and carry no evidence.

Write $\mu$ for the length of the shift divided by $\sigma$, the number of noise standard deviations between the two answers. The privacy loss is a straight-line function of the output, as @tbl-lg-shifted-pairs showed, and a straight-line function of a normal variable is again normal. So when the output comes from $M(D)$, the privacy loss is normal, with a mean and a variance set by $\mu$ alone,
$$L \sim \mathcal N\Big(\frac{\mu^2}2, \mu^2\Big).$$
Its mean, $\mu^2/2$, is positive, because outputs from $D$ tend to favour $D$. Its variance is $\mu^2$, and the loss itself has no upper limit.

Since the loss is normal, the exact $\delta$ of @eq-lg-bad-event has a closed form, which Balle and Wang (2018) turned into a calibration. Both terms are normal tail probabilities. Write $\Phi$ for the standard normal cumulative distribution function (CDF), so that $\Phi(t)$ is the probability that a standard normal variable is below $t$. To standardize the loss, subtract its mean and divide by its standard deviation $\mu$. The loss then exceeds $\varepsilon$ exactly when a standard normal variable exceeds $\frac\varepsilon\mu - \frac\mu2$. So on $D$ the bad event has probability $\Phi(\frac\mu2 - \frac\varepsilon\mu)$, by the symmetry of the normal curve. On $D'$ the outputs sit one shift further along, which lowers every loss by $\mu^2$, so the loss is centred at $-\mu^2/2$ instead. The same standardization then gives the bad event the probability $\Phi(-\frac\mu2 - \frac\varepsilon\mu)$ on $D'$.

The largest shift is the $\ell_2$ sensitivity, written $\Delta_2f$ as before, so take $\mu = \Delta_2f/\sigma$. The Gaussian mechanism is then $(\varepsilon, \delta)$-DP if and only if $\delta \ge \delta_\mu(\varepsilon)$, where
$$\delta_\mu(\varepsilon) = \Phi\Big(\frac\mu2 - \frac\varepsilon\mu\Big) - e^\varepsilon\,\Phi\Big(-\frac\mu2 - \frac\varepsilon\mu\Big).$$ {#eq-lg-analytic}

The formula is enough for every pair of neighbours because smaller shifts never cost more. The reason is that a pair with a smaller shift can be produced from a pair with a larger one. You scale the output down to shrink the shift, then add fresh noise to restore the variance. Processing an output in this way can never make two datasets easier to tell apart. The formula is also necessary, provided some pair of neighbours really is $\Delta_2f$ apart, which the definition of sensitivity as a maximum ensures.

The formula has the properties you would expect, and they make it easy to use. It decreases as $\varepsilon$ grows, so a larger $\varepsilon$ always comes with a smaller $\delta$. At $\varepsilon = 0$ it equals the **total variation (TV) distance** between the two output distributions, which is the largest difference in probability they give to any set of outputs. And it increases with $\mu$, so less noise always means a larger $\delta$.

That last property is what makes calibration possible. As $\mu$ runs from 0 to infinity, $\delta_\mu(\varepsilon)$ rises steadily from 0 to 1. So for any target $\delta$ between 0 and 1, exactly one value $\mu^*$ hits it. The **analytic calibration** is the noise level
$$\sigma^* = \frac{\Delta_2f}{\mu^*}, \qquad \text{where } \delta_{\mu^*}(\varepsilon) = \delta,$$
and it is the smallest $\sigma$ that makes the Gaussian mechanism $(\varepsilon, \delta)$-DP. You find $\mu^*$ by bisection, which keeps halving an interval that contains the single crossing point (@sec-lg-code). Take $\varepsilon = 1$ and $\delta = 10^{-5}$, a common setting that the rest of this chapter returns to. There, bisection gives $\sigma^* = 3.7306\,\Delta_2f$.

Take $\mu = 1$ and $\varepsilon = 1$ as an example. The bad event then has probability 0.3085 on $D$. Subtracting $e$ times its probability on the neighbour leaves the exact $\delta$ of 0.1269.

One Gaussian mechanism gives a whole curve of guarantees, not one. It satisfies every point of its privacy curve at once, so you can trade a larger $\varepsilon$ for a smaller $\delta$ (@fig-lg-delta-curves). @tbl-lg-curve gives points of the curve when the noise equals the sensitivity, so that $\mu = 1$. The curve starts at the TV distance and falls steeply as $\varepsilon$ grows. It reaches $10^{-5}$ at $\varepsilon = 4.3772$. So if someone asks for this mechanism's guarantee at that $\delta$, the answer is $\varepsilon = 4.3772$.

| $\varepsilon$ | 0 | 1 | 2 | 4 | 4.3772 |
| --- | --- | --- | --- | --- | --- |
| $\delta_1(\varepsilon)$ | 0.3829 | 0.1269 | 0.0209 | $4.71\times10^{-5}$ | $10^{-5}$ |

: The privacy curve of the Gaussian mechanism at $\sigma = \Delta_2f$. At $\varepsilon = 0$ it is the TV distance. {#tbl-lg-curve}

The privacy accountants of [Part 4](../part04-privacy-accounting/00-introduction.md), which track the total privacy cost of many releases, compose the whole curve, not one point on it. They need the noise level to rebuild the curve, so when you report a Gaussian release, report $\sigma/\Delta_2f$ (or $\mu$) as well as the pair you chose.

![The privacy curve $\delta(\varepsilon)$ of the Gaussian mechanism for four noise levels $\sigma/\Delta_2f$, on a log scale, and of the Laplace mechanism with $b = \Delta_1f$ (dashed), which drops to zero at $\varepsilon = 1$. Crosses are numerical checks. The dotted line marks $\delta = 10^{-5}$.](figures/lapgauss_delta_curves.svg){#fig-lg-delta-curves}

Only the ratio $\mu = \Delta_2f/\sigma$ enters the formula, so doubling the sensitivity and doubling the noise give the same curve. The same ratio is the parameter of the Gaussian trade-off curve in [Hypothesis testing](../part01-prerequisites/04-hypothesis-testing.md), the best any test can do at telling two normal distributions $\mu$ standard deviations apart.

Back to the diabetes count, now with Gaussian noise at $\varepsilon = 0.1$ and $\delta = 10^{-5}$. The sensitivity is 1, so the analytic calibration gives $\sigma^* = 30.75$. A normal variable stays within 1.96 standard deviations of its mean with probability 0.95. So the 95% error is $1.96\sigma^* = 60.27$. Laplace noise gave only 29.96, with $\delta = 0$. So for a single count the Laplace mechanism wins twice, with a smaller error and a stronger guarantee. The Gaussian mechanism earns its place only when many numbers are released together (@sec-lg-choose).

## The classic Gaussian calibration {#sec-lg-gauss-classic}

Many papers and libraries still use the calibration from the textbook of Dwork and Roth (2014). For $\varepsilon$ and $\delta$ both between 0 and 1, it sets
$$\sigma = \frac{\Delta_2f\sqrt{2\ln(1.25/\delta)}}{\varepsilon}.$$ {#eq-lg-classic}
The noise grows with the $\ell_2$ sensitivity and shrinks as $\varepsilon$ grows. As $\delta$ shrinks it grows only slowly, like the square root of a logarithm.

The proof behind the formula explains both its constants and its looseness. It bounds $\delta$ by the probability of the bad event, the right side of @eq-lg-bad-event, and so drops the subtracted term. It then bounds that normal tail probability by a simpler expression, the **Mills-ratio bound**. That bound says a standard normal variable lands beyond a positive point $t$ with probability at most the normal density at $t$ divided by $t$. Finally the proof requires this simpler expression to be no larger than the target $\delta$, and solves for $\sigma$.

That step leaves a factor $e^{\varepsilon/2}$ and a few other constants behind, and the number 1.25 in the formula is chosen to absorb them. The condition $\varepsilon < 1$ is what keeps $e^{\varepsilon/2}$ small enough for that. The proof also needs $\delta$ to be at most 0.7448, because a larger $\delta$ puts the start of the tail too close to zero for the last step to work. Every $\delta$ used in practice meets that condition.

Each of these approximations gives away some privacy you could have kept. At the common setting of @sec-lg-gauss, the classic formula gives $\sigma = 4.8448\,\Delta_2f$. The analytic calibration needs only $3.7306\,\Delta_2f$, and the true $\delta$ of the classic $\sigma$ is about 240 times smaller than you asked for. Dropping the subtracted term accounts for a factor of about 26, and the constants for another factor of about 9, while the Mills-ratio step itself costs almost nothing.

@tbl-lg-calibration compares the two calibrations at $\delta = 10^{-5}$. For values of $\varepsilon$ up to 1, the classic formula uses more noise than necessary, and the excess grows as $\varepsilon$ shrinks. The variance grows with the square of $\sigma$, so at $\varepsilon = 1$ the classic formula uses 69% more variance than needed. The last column shows the true $\delta$ that the classic $\sigma$ delivers, from @eq-lg-analytic.

| $\varepsilon$ | Classic $\sigma/\Delta_2f$ | Analytic $\sigma^*/\Delta_2f$ | Ratio | True $\delta$ at the classic $\sigma$ |
| --- | --- | --- | --- | --- |
| 0.05 | 96.8961 | 57.7707 | 1.677 | $1.29\times10^{-9}$ |
| 0.1 | 48.4481 | 30.7496 | 1.576 | $2.64\times10^{-9}$ |
| 0.25 | 19.3792 | 13.2855 | 1.459 | $7.10\times10^{-9}$ |
| 0.5 | 9.6896 | 7.0318 | 1.378 | $1.61\times10^{-8}$ |
| 1 | 4.8448 | 3.7306 | 1.299 | $4.11\times10^{-8}$ |
| 2 | 2.4224 | 1.9938 | 1.215 | $1.34\times10^{-7}$ |
| 4 | 1.2112 | 1.0812 | 1.120 | $6.84\times10^{-7}$ |
| 8 | 0.6056 | 0.6002 | 1.009 | $7.97\times10^{-6}$ |

: Classic against analytic calibration of the Gaussian mechanism at $\delta = 10^{-5}$. The last column is @eq-lg-analytic evaluated at the classic $\sigma$. {#tbl-lg-calibration}

For large $\varepsilon$ the classic formula is wrong, not just loose. The classic $\sigma$ shrinks like $1/\varepsilon$, while the exact one shrinks more slowly, so the two cross. At $\delta = 10^{-5}$ they cross at $\varepsilon = 8.42$, and beyond that point the classic $\sigma$ is below the necessary noise (@fig-lg-calibration). Between 1 and 8.42 the formula happens to stay safe, but nothing in its proof says so. Outside the range the classic proof covers, you need the exact curve.

![Left: the noise needed for $(\varepsilon, 10^{-5})$-DP by the classic formula and by the analytic calibration. Right: the true $\delta$ that the classic $\sigma$ delivers, from @eq-lg-analytic. It is orders of magnitude below the target for small $\varepsilon$ and crosses the target at $\varepsilon = 8.42$, in the shaded region where the classic theorem makes no claim.](figures/lapgauss_calibration.svg){#fig-lg-calibration}

## Laplace or Gaussian? {#sec-lg-choose}

Real releases are vectors, such as a histogram over many bins, all the marginals of a table, or a gradient with millions of coordinates. The Laplace mechanism scales with $\Delta_1f$ and the Gaussian mechanism with $\Delta_2f$, which by @eq-lg-norms can differ by a factor of up to $\sqrt d$. Which noise is better depends on how one record spreads its change across the coordinates.

To compare the two, look at the typical error in each released number, the root-mean-square error per coordinate, at the same $\varepsilon$. For Laplace noise it is the standard deviation of $\mathrm{Lap}(\Delta_1f/\varepsilon)$, which is $\sqrt2$ times the scale. For Gaussian noise it is $\sigma^*$. Because the exact calibration depends only on the ratio $\mu$, you can solve it once for unit sensitivity and scale the result. Write $\sigma_1$ for the analytic noise level when the $\ell_2$ sensitivity is 1, at your chosen privacy parameters. The errors per coordinate are then
$$\text{Laplace: } \frac{\sqrt2\,\Delta_1f}{\varepsilon}, \qquad \text{Gaussian: } \Delta_2f\,\sigma_1.$$

Two kinds of query cover most cases. In a **sparse query**, one record moves one coordinate, as in a histogram under add/remove, and the two sensitivities are equal. Dividing the Laplace error by the Gaussian one then leaves $\sqrt2/(\varepsilon\sigma_1)$, which does not depend on $d$. So Laplace noise wins whenever $\sqrt2 < \varepsilon\sigma_1$. In a dense query, one record moves every coordinate by 1, as in the threshold counts, so that $\Delta_1f = d$ and $\Delta_2f = \sqrt d$. The Laplace error then grows like $d$ and the Gaussian error only like $\sqrt d$, so their ratio grows like $\sqrt d$. Setting that ratio below 1 and squaring shows that Laplace noise is more accurate only while
$$d < \frac{(\varepsilon\sigma_1)^2}2.$$

At the common setting from @sec-lg-gauss, the unit noise level is $\sigma_1 = 3.7306$. So for a single count, Laplace noise has a root-mean-square error of 1.41 against 3.73 for Gaussian noise, so Laplace wins, and with a stronger guarantee besides. For threshold counts the crossover is at $d = 6.96$, so Gaussian noise wins from 7 counts on (@fig-lg-high-dim). An empirical distribution of ages read off at many thresholds is such a dense query, since one person's age moves every count with a threshold at or below it. The advantage of Gaussian noise grows with the number of thresholds (@tbl-lg-high-dim).

| $d$ | Laplace | Gaussian | Laplace / Gaussian |
| --- | --- | --- | --- |
| 1 | 1.41 | 3.73 | 0.38 |
| 10 | 14.14 | 11.80 | 1.20 |
| 100 | 141.42 | 37.31 | 3.79 |
| 1000 | 1414.21 | 117.97 | 11.99 |

: Root-mean-square error per count for $d$ threshold counts at $\varepsilon = 1$, $\delta = 10^{-5}$. {#tbl-lg-high-dim}

![Root-mean-square error per count at $\varepsilon = 1$ and $\delta = 10^{-5}$, as the number of released counts $d$ grows; dots are simulations. Left: a histogram, where Laplace noise is smaller at every $d$. Right: threshold counts, where the Laplace error grows like $d$ and the Gaussian like $\sqrt d$, crossing near $d = 7$.](figures/lapgauss_high_dim.svg){#fig-lg-high-dim}

This is why private training of neural networks uses Gaussian noise. One example's gradient moves every coordinate of a model with millions of parameters, which is the densest query there is. Private training clips each gradient to a bound on its $\ell_2$ norm, and the Gaussian mechanism is calibrated to exactly that norm.

The comparison is at equal $\varepsilon$ but not at equal guarantees, because the Gaussian release also carries a $\delta$. But Gaussian noise has a second advantage that the comparison does not show. Several Gaussian releases with the same noise level compose exactly into one Gaussian mechanism, because stacking independent Gaussian answers gives one Gaussian answer. The combined mechanism's $\ell_2$ sensitivity is the square root of the sum of the squared sensitivities of the parts.

## Code {#sec-lg-code}

The analytic calibration needs only the normal CDF, which the standard library provides through `math.erfc`. This is the whole of it.

```python
from math import erfc, exp, sqrt


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


print(f"{analytic_sigma(1.0, 1e-5):.4f}")  # 3.7306
print(f"{analytic_sigma(0.1, 1e-5):.4f}")  # 30.7496
print(f"{gaussian_delta(1.0, 1.0):.4f}")  # 0.1269
```

The first printed line is the unit noise level $\sigma_1$ at the common setting. The second is the noise for the diabetes count at $\varepsilon = 0.1$, as in the analytic column of @tbl-lg-calibration. The third is the exact $\delta$ at $\mu = 1$ and $\varepsilon = 1$. For the Laplace mechanism there is nothing to solve, since the scale is $\Delta_1f/\varepsilon$.

Drawing the noise is a separate problem. Textbook floating-point Laplace samplers can break the guarantee through the low-order bits of their output ([Discrete and floating-point-safe mechanisms](07-discrete-mechanisms.md)). So in production, take the noise from a vetted DP library, and use this calculation to check the scale it reports. The full edition's script [laplace_gaussian.py](../../code/part03-dp-foundations/laplace_gaussian.py) computes every number in this chapter and checks each mechanism by simulation. Run it from the repository root.

```bash
uv run python code/part03-dp-foundations/laplace_gaussian.py
```

## Cheat sheet

| Result | Formula | Notes |
| --- | --- | --- |
| Sensitivity | $\Delta_pf = \max_{D \simeq D'}\lVert f(D) - f(D')\rVert_p$ | $\Delta_2 \le \Delta_1 \le \sqrt d\,\Delta_2$; depends on the neighbouring relation |
| Count | $1$ | Add/remove; 0 under replace |
| Sum of records in $[0, B]$ | $B$ | Either relation |
| Mean of records in $[0, B]$, public $n$ | $B/n$ | Replace |
| Laplace mechanism | $\mathrm{Lap}(\Delta_1f/\varepsilon)$ per coordinate is $\varepsilon$-DP | Tight: the loss equals $\varepsilon$ on outputs carrying half the probability |
| Laplace accuracy | $\Pr[\max_i\lvert Z_i\rvert > b\ln(d/\beta)] \le \beta$; mean absolute error $b$ | Error does not depend on $n$ |
| Exact Gaussian privacy curve, $\mu = \Delta_2f/\sigma$ | $\delta_\mu(\varepsilon) = \Phi(\frac\mu2 - \frac\varepsilon\mu) - e^\varepsilon\Phi(-\frac\mu2 - \frac\varepsilon\mu)$ | If and only if; equals the TV distance at $\varepsilon = 0$ |
| Analytic calibration | $\sigma^* = \Delta_2f/\mu^*$ with $\delta_{\mu^*}(\varepsilon) = \delta$ | Solve by bisection; one root |
| Classic calibration | $\sigma = \Delta_2f\sqrt{2\ln(1.25/\delta)}/\varepsilon$ | Only for $\varepsilon < 1$; 1.3–1.7 times too much noise there |
| Error per coordinate | Laplace $\sqrt2\,\Delta_1f/\varepsilon$; Gaussian $\Delta_2f\,\sigma_1$ | Dense queries: Laplace wins only for $d < (\varepsilon\sigma_1)^2/2$ |

: The results of this chapter. {#tbl-lg-cheat}

**Rules of thumb**

- Compute the sensitivity from the domain, not the data, and state the neighbouring relation with every $\varepsilon$.
- For one number or a sparse vector, use Laplace noise and get pure DP. For a dense vector, use Gaussian noise calibrated to $\Delta_2f$.
- Solve @eq-lg-analytic rather than use the classic Gaussian formula, which adds 30–70% more noise than needed.
- Report $\sigma/\Delta_2f$ for a Gaussian release, not just one $(\varepsilon, \delta)$ pair.
- Sample the noise with a vetted DP library, never with a plain floating-point sampler.

## Further reading

- Dwork & Roth, *The Algorithmic Foundations of Differential Privacy* (2014), §3.3 and Appendix A: the Laplace mechanism, its accuracy, and the classic Gaussian calibration.
- Balle & Wang, "Improving the Gaussian Mechanism for Differential Privacy: Analytical Calibration and Optimal Denoising" (ICML 2018): the exact privacy curve, the bisection calibration, and why the classic formula fails for large $\varepsilon$.
- Near & Abuah, *Programming Differential Privacy*, the chapters on sensitivity and on the Laplace and Gaussian mechanisms: the same material with code, and the practical pitfalls of computing sensitivity.
- Geng & Viswanath, "The Optimal Noise-Adding Mechanism in Differential Privacy" (IEEE Transactions on Information Theory, 2016): the staircase mechanism, and how close the Laplace mechanism comes to it.

**Derivations.** Every result in this chapter is derived step by step in [the full edition's chapter](../../book-full/part03-dp-foundations/03-laplace-gaussian.md), except the staircase mechanism, which it cites, and the exact composition of Gaussian releases, which it states and its exercises derive, and the floating-point warning, which [Discrete and floating-point-safe mechanisms](07-discrete-mechanisms.md) explains.
