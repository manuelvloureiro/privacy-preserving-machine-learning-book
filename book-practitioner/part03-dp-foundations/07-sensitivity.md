# Beyond global sensitivity

## Why this matters

The [Laplace and Gaussian mechanisms](03-laplace-gaussian.md) set their noise by the **global sensitivity**. That is the most one record can move the answer, over every pair of neighbouring datasets that could exist. It is 1 for a count, but for many statistics it is useless. The median of ages between 0 and 100 has global sensitivity 100. So at a privacy level of $\varepsilon = 1$, the Laplace mechanism misses the median by 69.31 years in a typical release. The mean of incomes has no bound, and for a trained model nobody knows one. Yet on a sample of 1,001 ages, one person moves the median by only 0.1386 years.

This chapter shows how to add noise on the scale of the data you hold and still satisfy differential privacy (DP). Every technique respects one rule: **a noise scale computed from the data is itself a statistic of the data, so it must be private too.** Throughout, datasets have a fixed size $n$, and two datasets are neighbours when one replaces a single record of the other.

## Local sensitivity, and why it leaks {#sec-sens-local}

The **local sensitivity** of a query $f$ at your dataset $D$ is the most that one record can move the answer, starting from $D$ itself:
$$\mathrm{LS}_f(D) = \max_{D' \text{ a neighbour of } D}|f(D) - f(D')|.$$
For the median of an odd number of values, it is the larger of the two gaps between the middle value and its sorted neighbours.

The tempting release adds Laplace noise of scale $\mathrm{LS}_f(D)/\varepsilon$, and it is not private. For example, the five ages $(0, 0, 0, 0, 100)$ have median 0 and no gap around it, so they get no noise and the output is exactly 0. Their neighbour $(0, 0, 0, 100, 100)$ has the same median but a gap of 100 above it. Its output is spread out and never exactly 0, so an output of 0 names the first dataset with certainty. An observer reads the data from the width of the noise, not from its centre.

The same leak hides in any parameter set from the private data, such as a clip bound set to the observed maximum. The diffprivlib functions `diffprivlib.tools.mean` and `diffprivlib.tools.median` take a `bounds` argument, and if you leave it out they compute the bounds from the data and raise a `PrivacyLeakWarning`. Treat that warning as an error.

## Clipping {#sec-sens-clip}

Clipping forces a bound on unbounded data. Replace each non-negative value $x$ by $\min(x, C)$ for a clip bound $C$ that you choose. One record then moves the sum by at most $C$ and the mean by at most $C/n$, so the clipped mean is $\varepsilon$-DP:
$$M(D) = \frac1n\sum_{i=1}^n\min(x_i, C) + \mathrm{Lap}\Big(\frac{C}{n\varepsilon}\Big).$$
Per-example gradient clipping in private training is the same move applied to Euclidean length, set in Opacus by the `max_grad_norm` argument of `PrivacyEngine.make_private` ([Private stochastic gradient descent](../part05-dp-ml/02-dp-sgd.md)).

The price is a bias, because clipping removes the part of each value above $C$. Write $X$ for a value drawn from the data's distribution, and $(x - C)_+$ for $\max(x - C, 0)$. For $n$ independent values, the mean squared error is
$$\operatorname{err}(C)^2 = \underbrace{\mathbb E[(X - C)_+]^2}_{\text{bias}^2} + \underbrace{\frac{\operatorname{Var}(\min(X, C))}{n}}_{\text{sampling}} + \underbrace{\frac{2C^2}{n^2\varepsilon^2}}_{\text{noise}}.$$ {#eq-sens-clip-error}
Raising $C$ shrinks the bias and grows the noise.

Take log-normal incomes in thousands with median 40, and a thousand people at $\varepsilon = 1$. Clipping at the median gives a root-mean-square error (RMSE) of 23.417, because half the incomes are cut back. The optimum is 331.34, which cuts off only 0.4% of the population. Its RMSE is 1.666, while the non-private mean has 1.649.

Choosing $C$ is the step that goes wrong, so never set it to the observed maximum or an empirical percentile. Use this order instead:

1. Take $C$ from domain knowledge, such as a legal maximum or a sensor range.
2. Otherwise, fix it from an error model like @eq-sens-clip-error, with a distribution assumed before you see any records.
3. Otherwise, spend a small slice of the budget on a private high quantile by inverse sensitivity (@sec-sens-inverse), for example with `diffprivlib.tools.quantile`, and clip at that value.
4. Charge that slice to the budget by [composition](02-properties.md). Forgetting it is a common bug.

The libraries make the bounds explicit. OpenDP clamps with `dp.t.make_clamp` before `dp.t.make_sum`, diffprivlib takes `bounds` in `diffprivlib.tools.mean`, and Tumult Analytics takes them in `QueryBuilder(...).average(column, low, high)`.

Clipping only from above always underestimates the mean, and a two-sided clip cancels its bias only when the range is centred on the data, so estimate the centre privately first ([Private mean estimation](11-mean-estimation.md)). Under add/remove neighbours the size $n$ is not public, so release a noisy clipped sum and a noisy count, and divide them.

## Subsample-and-aggregate {#sec-sens-saa}

When $f$ is a model fit, a ratio or a pipeline you cannot analyse, **subsample-and-aggregate** treats it as a black box. Split the $n$ records into $B$ disjoint blocks, with a partition fixed in advance, for example by a hash of a record identifier. Run $f$ on each block, and clip each answer to the range from 0 to a bound $R$ fixed in advance. Then release the average of the clipped answers $z_j$ plus Laplace noise:
$$M(D) = \frac1B\sum_{j=1}^B \operatorname{clip}_{[0,R]}(z_j) + \mathrm{Lap}\Big(\frac{R}{B\varepsilon}\Big).$$
One record lives in one block, so it moves the average by at most $R/B$, and the release is $\varepsilon$-DP for any $f$.

The number of blocks trades bias against noise. Write $z$ for the output of $f$ on one block and $\theta$ for the target. When clipping is inactive and the blocks are independent, the mean squared error is
$$\underbrace{\big(\mathbb E[z] - \theta\big)^2}_{\text{bias of } f \text{ on } n/B \text{ records}} + \underbrace{\frac{\operatorname{Var}(z)}{B}}_{\text{sampling}} + \underbrace{2\Big(\frac{R}{B\varepsilon}\Big)^2}_{\text{noise}}.$$ {#eq-sens-saa-error}
More blocks dilute the noise, but they make each block smaller, which raises the bias.

For example, estimate the rate of 10,000 exponential waiting times by one over the block mean, with $R = 5$. With 10 blocks the noise dominates and the RMSE is 0.7072, while with 1,000 blocks the bias dominates and it is 0.1120. The best grid point lies between them, at 250 blocks, where the RMSE is 0.0396. A non-private estimate has 0.0100, so privacy costs a factor of four, for an estimator whose sensitivity you never had to bound. Choose $B$ from this error model before looking at the data.

## Medians and quantiles: inverse sensitivity {#sec-sens-inverse}

For a median or a quantile, start with inverse sensitivity (Asi & Duchi 2020). For each candidate output $t$, it counts how many records you would have to change before the median equals $t$, and calls that count $\mathrm{len}(D, t)$. It then picks $t$ with a probability that falls by a factor $e^{-\varepsilon/2}$ for each change needed:
$$p_D(t) \propto e^{-\varepsilon\,\mathrm{len}(D, t)/2}.$$
This is the [exponential mechanism](05-selection.md) with the count as a penalty. One record changes every count by at most 1, so the release is pure $\varepsilon$-DP.

For the median, the count is easy to compute. Sort the values, and pad them with 0 below and with the top of the range $\Lambda$ above. The two gaps next to the median need one change, the next two out need two, and so on. So you pick a gap with probability proportional to its length times $e^{-\varepsilon\,\mathrm{len}/2}$, and then a point uniformly inside it. The function `diffprivlib.tools.quantile` works this way, given `quant`, `epsilon` and `bounds`.

On the 1,001 ages at $\varepsilon = 1$, the typical error is 0.0783 years. That is below even the local sensitivity, and it beats every other method of this chapter. It struggles on small or degenerate data, though. Smaller datasets have wider gaps, so on 101 ages the error grows to 0.696 years.

## Propose-test-release {#sec-sens-ptr}

Propose-test-release (Dwork and Lei 2009) needs only a way to measure how stable a statistic is. You propose a bound $b$ on the local sensitivity before looking at the data. The **distance to instability** is the number of records you must change before the local sensitivity can exceed $b$, and one record moves it by at most 1. So the mechanism adds $\mathrm{Lap}(1/\varepsilon)$ noise to the distance and refuses, which is written $\bot$, if the result is at or below $\ln(1/\delta)/\varepsilon$, where $\delta$ is a small slack. Otherwise it releases $f(D) + \mathrm{Lap}(b/\varepsilon)$, and the whole is $(2\varepsilon, \delta/2)$-DP.

The threshold is demanding, because at $\varepsilon = 1$ and $\delta = 10^{-6}$ it is 13.82 changes. For example, propose $b = 1$ year for the 1,001 ages. Their distance is 21, so the test passes with probability 0.999621, and the release misses by $\ln 2 = 0.693$ years in a typical release, at a cost of $2\varepsilon$.

## Stability-based release {#sec-sens-stable}

One record usually cannot change a discrete answer, such as the most common diagnosis, so the **stability-based release** publishes it exactly after a private check. For the most common value, write $g$ for its lead over the runner-up. Its stability score is
$$s = \max\big(0,\ \lceil g/2\rceil - 1\big),$$
where $\lceil g/2\rceil$ rounds $g/2$ up, and the score is 0 whenever one change could flip the answer. The release adds $\mathrm{Lap}(1/\varepsilon)$ to the score and refuses if the result is at or below $\ln(1/\delta)/\varepsilon$, but otherwise it outputs the exact mode, and the whole is $(\varepsilon, \delta/2)$-DP.

At the same settings as above, the smallest lead that answers 95% of the time is 35 records, whatever the number of categories. In a close race the release refuses, so plan for refusals downstream.

## Smooth sensitivity {#sec-sens-smooth}

Smooth sensitivity (Nissim, Raskhodnikova and Smith 2007) keeps "answer plus noise", but it scales the noise to a bound that changes only slowly between neighbours. For pure DP that noise must have heavy tails, as in the Cauchy release $f(D) + \frac{6}{\varepsilon}\,S^*_{\varepsilon/6}(D)\,Z$, where $S^*$ is the smooth sensitivity and $Z$ a Cauchy draw. It pays off only on dense data, with $n$ in the hundreds or more, and on the 1,001 ages it is about twelve times less accurate than inverse sensitivity. Never pair it with Laplace noise and claim $\delta = 0$, because the Laplace version needs a slack and a much smaller smoothing parameter.

## Cheat sheet

| Releasing | Technique | Release | Privacy |
| --- | --- | --- | --- |
| A median or quantile | Inverse sensitivity | Pick $t$ with probability $\propto e^{-\varepsilon\,\mathrm{len}(D, t)/2}$ | $\varepsilon$ |
| Dense data with a smooth-sensitivity formula | Smooth sensitivity, Cauchy | $f(D) + \frac{6}{\varepsilon}S^*_{\varepsilon/6}(D)\,Z$ | $\varepsilon$ |
| A stable statistic with no formula | Propose-test-release | $\bot$ if $\operatorname{dist}_b + \mathrm{Lap}(\frac1\varepsilon) \le \frac{\ln(1/\delta)}{\varepsilon}$, else $f(D) + \mathrm{Lap}(\frac b\varepsilon)$ | $(2\varepsilon, \delta/2)$ |
| A discrete answer with a clear winner | Stability-based release | The same test on a stability score, then $f(D)$ exactly | $(\varepsilon, \delta/2)$ |
| A black-box estimator or model | Subsample-and-aggregate | Mean of $B$ clipped block outputs $+ \mathrm{Lap}(\frac{R}{B\varepsilon})$ | $\varepsilon$ |
| A mean or sum of unbounded values | Clipping | $\frac1n\sum\min(x_i, C) + \mathrm{Lap}(\frac{C}{n\varepsilon})$, bias $\mathbb E[(X - C)_+]$ | $\varepsilon$ |

: Which technique to use for which release, with neighbours that replace one record. {#tbl-sens-cheat}

**Rules of thumb**

- Noise scaled to the local sensitivity is not private, and neither is a clip bound or range read off the data.
- For medians and quantiles, start with inverse sensitivity.
- A discrete answer can be released exactly once it leads by about $2\ln(1/\delta)/\varepsilon$ records.
- Fix clip bounds, block counts and proposals before looking at the data, never from its minimum or maximum, and charge any budget spent choosing them.
- Smooth sensitivity with Laplace noise needs a $\delta$; never ship it as pure DP.

## Further reading

- Nissim, Raskhodnikova & Smith, "Smooth Sensitivity and Sampling in Private Data Analysis" (STOC 2007): smooth sensitivity, the median formula and sample-and-aggregate.
- Asi & Duchi, "Instance-optimality in Differential Privacy via Approximate Inverse Sensitivity Mechanisms" (NeurIPS 2020): inverse sensitivity, and why it is close to the best possible on every dataset.

**Derivations.** Every result here is derived step by step in [the full edition's chapter](../../book-full/part03-dp-foundations/07-sensitivity.md).
