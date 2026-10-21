# Private mean estimation

## Why this matters

The mean is the statistic that private machine learning computes most often. Every step of private gradient descent clips per-example gradients, averages them and adds noise, and a survey that reports the share of people with some property releases a mean of 0/1 answers. Earlier chapters released the mean of the dataset in hand. An analyst usually wants something else: the mean of the **population** the data were drawn from. This chapter asks: **how many records does a private estimate of a population mean need, and what does each thing you do not know about the data cost?**

Two guarantees with different shapes meet here. Privacy must hold for *every* dataset, including ones that your assumptions say cannot happen, because a guarantee that fails when an assumption fails protects nobody. Accuracy, on the other hand, is a statement about data drawn from a distribution, such as "the records are normal with variance 1". So an assumption about the data may make a mechanism more accurate, but it may never enter the privacy proof. Every mechanism below respects that split: it clips the data to a range, which makes it private whatever the data are, and the assumption only says how often the clipping changes nothing.

The organizing idea of the chapter is this:

> **Spend a small part of the budget to learn where the data are, then clip there and spend the rest on the mean.**

The noise of a clipped mean is proportional to the width of the clipping range. If you clip to a range set by your prior ignorance, the noise pays for that ignorance in full. If you first find the data privately, which costs a number of records that grows only like the logarithm of the range, the final step can clip to a narrow window and add little noise. The same idea, with more work, handles many dimensions and heavy tails.

@tbl-mean-uses lists the results and where each is used.

| Result | What it gives you | Where it is used |
| --- | --- | --- |
| Bounded data | Privacy adds about $1/(\alpha\varepsilon)$ records to the $1/\alpha^2$ that sampling needs | Proportions, surveys, clipped sums |
| The naive estimator | Clipping to a known range costs records in proportion to the range | The baseline everything below improves on |
| Coarse then fine | The range costs only about $\ln R/\varepsilon$ records | Means with unknown location; choosing clip bounds privately |
| Shrinking balls | In $d$ dimensions, about $d/(\alpha\varepsilon)$ records for privacy, plus $\sqrt d\ln R/\varepsilon$ for the range | Private averages of vectors, such as gradients |
| Heavy tails | Clipping into the data, with a bias, costs about $1/(\alpha^2\varepsilon)$ records | Incomes, session lengths, gradient norms |

: The results of this chapter, what each gives you, and where it is used. Here $\alpha$ is the target error, $\varepsilon$ the privacy parameter, $R$ a bound on how far the mean can be from 0 and $d$ the dimension. {#tbl-mean-uses}

**Conventions.** Datasets have a fixed size $n$, and two datasets are **neighbours** when they differ in one replaced record, as in [Beyond global sensitivity](07-sensitivity.md). The records, written $X_1, \dots, X_n$, are drawn independently from a distribution whose mean $\mu$ is the target of the estimate. An estimate $\hat\mu$ is judged by its **root-mean-square error** (RMSE), the square root of the average of $(\hat\mu - \mu)^2$ over the data and the mechanism's noise.

Recall what differential privacy (DP) asks of a randomized mechanism $M$. Take any two neighbouring datasets $D$ and $D'$ and any set $S$ of outputs. With a privacy parameter $\varepsilon$ and a small slack $\delta$, the mechanism is $(\varepsilon, \delta)$-DP if
$$\Pr[M(D) \in S] \le e^\varepsilon\Pr[M(D') \in S] + \delta,$$
and it is **pure** $\varepsilon$-DP when $\delta = 0$ ([The definition of differential privacy](01-definition.md)).

## Toolkit: five facts every derivation uses

### Fact 1 — The error splits into sampling, bias and noise

Every estimator in this chapter has the same form: clip each record with some function $c$, average, and add independent noise $\eta$ with mean 0. Its error splits into three parts, each with its own cause. The average of the clipped records varies from sample to sample, clipping shifts its expected value away from $\mu$, and the noise adds its own spread. The mean squared error is the sum of the three squares:
$$\mathbb E\big[(\hat\mu - \mu)^2\big] = \underbrace{\frac{\operatorname{Var}(c(X))}{n}}_{\text{sampling}} + \underbrace{\big(\mathbb E[c(X)] - \mu\big)^2}_{\text{bias}^2} + \underbrace{\operatorname{Var}(\eta)}_{\text{noise}}.$$ {#eq-mean-split}
To see it, write $\hat\mu - \mu$ as the sum of three pieces: the average of $c(X_i) - \mathbb E[c(X)]$, the constant $\mathbb E[c(X)] - \mu$, and $\eta$. The first and the last have mean 0 and are independent of each other, so when you square the sum and average, every cross term contains a mean-zero factor and vanishes. The average of $n$ independent terms has variance $1/n$ times that of one term, which gives the sampling part. This is the decomposition of the clipping section of [Beyond global sensitivity](07-sensitivity.md#sec-sens-clip).

### Fact 2 — A clipped mean with Laplace noise

If every clipped record lies in an interval of width $w$, then replacing one record moves the sum by at most $w$ and the mean by at most $w/n$. That is the sensitivity of the mean, so by the [Laplace mechanism](03-laplace-gaussian.md) adding noise $\mathrm{Lap}(w/(n\varepsilon))$ makes it $\varepsilon$-DP. A Laplace variable with scale $s$ has variance $2s^2$, so the noise term of @eq-mean-split is
$$\operatorname{Var}(\eta) = \frac{2w^2}{n^2\varepsilon^2}.$$ {#eq-mean-laplace-noise}
The privacy holds for every dataset, because the clipping, not an assumption, keeps the records in the interval. An interval chosen from the data must be chosen privately, as Fact 5 allows.

### Fact 3 — Normal draws stay close to their mean

A standard normal variable $Z$ exceeds a threshold $t$ in absolute value with probability at most $2e^{-t^2/2}$. This is the Chernoff bound on each of its two tails. [Concentration inequalities](../part01-prerequisites/02-concentration.md) proves a bound half as large, but the factor 2 changes $\tau$ below only slightly, so the chapter keeps the simpler form. Now take $n$ draws from a normal distribution with mean $\mu$ and variance 1, and allow a failure probability $\beta$. The union bound adds the $n$ tail probabilities, so all $n$ draws lie within a distance $\tau$ of $\mu$ with probability at least $1 - \beta$ once $2ne^{-\tau^2/2} = \beta$, that is, for
$$\tau = \sqrt{2\ln(2n/\beta)}.$$ {#eq-mean-spread}
For $n = 1000$ and $\beta = 0.05$, $\tau = 4.604$. The bound is loose: in 20,000 simulated datasets the largest distance exceeded $\tau$ in a fraction 0.0039 of them, not 0.05. In $d$ dimensions the analogue is about the length of the vector $X - \mu$, which concentrates near $\sqrt d$ rather than near 0. The section on many dimensions uses it.

### Fact 4 — The exponential mechanism rarely picks a bad candidate

The [exponential mechanism](05-selection.md) picks one of $m$ candidates with probability proportional to $e^{\varepsilon u/2}$, where $u$ is a score that one record changes by at most 1. It is $\varepsilon$-DP. Suppose the best score is $u^*$. A candidate whose score is at most $u^* - g$ has a weight at most $e^{\varepsilon(u^* - g)/2}$, while the total weight is at least that of the best candidate, $e^{\varepsilon u^*/2}$. Adding over at most $m$ such candidates,
$$\Pr[\text{the chosen score is at most } u^* - g] \le m\,e^{-\varepsilon g/2}.$$ {#eq-mean-em-utility}

### Fact 5 — Budgets add, even when a step uses an earlier output

If one step is $(\varepsilon_1, \delta_1)$-DP and a second step, which may use the first step's output, is $(\varepsilon_2, \delta_2)$-DP for each value of that output, then together they are $(\varepsilon_1 + \varepsilon_2, \delta_1 + \delta_2)$-DP. This is basic composition, adaptive case ([Post-processing, group privacy and composition](02-properties.md)). It is what allows a first step to choose the clipping range of a second.

## Bounded data: privacy adds a lower-order term {#sec-mean-bounded}

### Context

Start where nothing is unknown except the mean. The records lie in a known interval, such as yes/no answers coded 0 and 1. How many more records does a private estimate need than a non-private one?

### Statement

Let the records lie in an interval of width $w$ and have variance $\sigma^2$. Release their average plus $\mathrm{Lap}(w/(n\varepsilon))$. This is $\varepsilon$-DP, and its RMSE is
$$\mathrm{RMSE} = \sqrt{\frac{\sigma^2}{n} + \frac{2w^2}{n^2\varepsilon^2}}.$$ {#eq-mean-bounded}
The RMSE is at most $\alpha$ once
$$n \ge \frac{\sigma^2 + \sqrt{\sigma^4 + 8w^2\alpha^2/\varepsilon^2}}{2\alpha^2}, \qquad\text{which is at most}\qquad \frac{\sigma^2}{\alpha^2} + \frac{\sqrt2\,w}{\alpha\varepsilon}.$$ {#eq-mean-bounded-n}

### Derivation

1. **Privacy.** No clipping is needed, since the records already lie in the interval, so Fact 2 applies with width $w$.
2. **Error.** In @eq-mean-split the bias is 0, the sampling term is $\sigma^2/n$, and the noise term is $2w^2/(n\varepsilon)^2$ by @eq-mean-laplace-noise. Their sum is the square of @eq-mean-bounded.
3. **Solve for $n$.** Setting the square of @eq-mean-bounded equal to $\alpha^2$ and multiplying by $n^2$ gives the quadratic $\alpha^2n^2 - \sigma^2n - 2w^2/\varepsilon^2 = 0$. Its positive root is the first form of @eq-mean-bounded-n.
4. **The simpler bound.** The square root of a sum is at most the sum of the square roots, so $\sqrt{\sigma^4 + 8w^2\alpha^2/\varepsilon^2} \le \sigma^2 + 2\sqrt2\,w\alpha/\varepsilon$. Inserting this and dividing by $2\alpha^2$ gives the second form.

### Interpretation

The second form of @eq-mean-bounded-n reads as two costs added together. Sampling needs about $\sigma^2/\alpha^2$ records whatever you do. Privacy adds about $w/(\alpha\varepsilon)$ more. The ratio of the privacy term to the sampling term is $\sqrt2\,w\alpha/(\varepsilon\sigma^2)$, so for a small target error $\alpha$ privacy costs a lower-order number of records. That is the best case, and it is what the rest of the chapter tries to recover when the data are not bounded.

### Example: a survey proportion

Take yes/no answers, so $w = 1$, and the worst case for sampling, a population that is half yes, so $\sigma = 0.5$. At $\varepsilon = 1$, the private RMSE is 0.0520 for 100 people against 0.0500 without privacy, 0.0159 against 0.0158 for 1,000, and 0.0050 for both at 10,000. To reach $\alpha = 0.05$ you need 108 people instead of 100, and to reach $\alpha = 0.01$ you need 2,508 instead of 2,500. Privacy costs eight people at a sample size where sampling already needs 2,500.

### Check yourself

- If you halve $\varepsilon$, which term of @eq-mean-bounded-n changes, and by how much? *(Only the privacy term, which doubles. The sampling term does not involve $\varepsilon$.)*
- Why does the privacy term grow with the width $w$ but not with the variance $\sigma^2$? *(The noise is set by the worst-case change one record can make, which is the width, whatever the data look like. The variance only describes typical data, which enter the sampling term.)*

## Gaussian data with a known range: the naive estimator {#sec-mean-naive}

### Context

Now let the records be unbounded, but drawn from a normal distribution with unknown mean $\mu$ and variance 1, and suppose all you know about the mean is that $|\mu| \le R$. The bound $R$ measures your ignorance. A height in centimetres might have $R$ of a few hundred, while a feature of a model you have never looked at might need $R = 10^6$. The obvious estimator clips to the range where the data could possibly be.

### Statement

With $\tau$ from @eq-mean-spread, clip every record to $[-R - \tau, R + \tau]$, average, and add $\mathrm{Lap}\big(2(R + \tau)/(n\varepsilon)\big)$. This is $\varepsilon$-DP. With probability at least $1 - \beta$ the clipping changes no record. On that event the estimate is the plain average plus noise, and ignoring the rare event where it fails, the error has no bias and
$$\mathrm{RMSE} = \sqrt{\frac1n + \frac{8(R + \tau)^2}{n^2\varepsilon^2}}.$$ {#eq-mean-naive}
The noise grows linearly in $R$, so reaching an error $\alpha$ takes about $1/\alpha^2 + R/(\alpha\varepsilon)$ records.

### Derivation

1. **Privacy.** The clipping interval has width $w = 2(R + \tau)$, so Fact 2 gives $\varepsilon$-DP for every dataset.
2. **Clipping changes nothing, usually.** By Fact 3, with probability at least $1 - \beta$ every record lies within $\tau$ of $\mu$. Since $|\mu| \le R$, every record then lies in $[-R - \tau, R + \tau]$, where clipping leaves it alone.
3. **Error.** On that event the clipped average is the plain average, so @eq-mean-split has no bias, a sampling term $1/n$, and the noise term @eq-mean-laplace-noise with $w = 2(R + \tau)$. Their sum is @eq-mean-naive. The script also computes the exact error, without conditioning on the event, from the mean and variance of a clipped normal variable, and the two agree to the digits quoted.

### Interpretation

The estimator pays for $R$ in full. Doubling your uncertainty about where the data are doubles the noise, even though the data themselves have not changed. The [lower-bounds chapter](10-lower-bounds.md) shows that this is not the price of privacy. Its packing argument gives a lower bound of only $\ln(R/\alpha)/\varepsilon$ records, so a cleverer mechanism should pay logarithmically, not linearly, for the range.

### Example: three ranges

Take $n = 1000$ records at $\varepsilon = 1$, with $\beta = 0.05$ so that $\tau = 4.604$, and put the true mean at $\mu = R/2$. For $R = 10$, the noise standard deviation is 0.0413 and the RMSE is 0.0520, against 0.0316 without privacy. The error grows with the range, to 2.8416 for $R = 10^3$ and to 2828.44 for $R = 10^6$. That last estimate is useless, although the 1,000 records pin the mean down to within about 0.03.

### Check yourself

- Why widen the interval by $\tau$ instead of clipping to $[-R, R]$? *(A mean near $R$ has half its records above $R$. Clipping them would bias the estimate toward 0. The margin $\tau$ keeps every record unclipped with probability $1 - \beta$.)*
- Does the privacy of the estimator depend on the data being normal? *(No. The clipping bounds the sensitivity for every dataset. Normality is used only to show that the clipping usually changes nothing.)*

## Coarse then fine {#sec-mean-coarse}

### Context

The naive estimator wastes its noise on the empty parts of $[-R, R]$. Karwa & Vadhan (2018) split the work in two. A **coarse step** spends a small part of the budget to find, privately, a short interval that holds the data. A **fine step** then clips to that interval and spends the rest of the budget on the mean. The coarse step is a selection problem among about $R$ candidate intervals, and selection costs only the logarithm of the number of candidates.

### Statement

Cut $[-R - b, R + b]$ into $m = \lceil 2(R + b)/b\rceil$ bins of width $b = 2\tau$, where $\lceil\cdot\rceil$ rounds up. Split the budget as $\varepsilon = \varepsilon_1 + \varepsilon_2$.

1. **Coarse step.** Choose a bin $J$ with the exponential mechanism at $\varepsilon_1$, with the number of records in the bin as its score.
2. **Fine step.** Clip every record to the **window** made of bin $J$ and its two neighbours, an interval of width $3b$. Average, and add $\mathrm{Lap}(3b/(n\varepsilon_2))$.

The mechanism is $\varepsilon$-DP. With probability at least $1 - \beta - m e^{-\varepsilon_1 n/4}$, no record is clipped. On that event the estimate is the plain average plus noise, and ignoring the rare event where it fails,
$$\mathrm{RMSE} = \sqrt{\frac1n + \frac{18b^2}{n^2\varepsilon_2^2}},$$ {#eq-mean-coarse}
which does not depend on $R$. The coarse step fails with probability at most $\beta$ once
$$n \ge \frac{4\ln(m/\beta)}{\varepsilon_1}.$$ {#eq-mean-coarse-n}
In total, an error $\alpha$ needs about $1/\alpha^2 + 1/(\alpha\varepsilon) + \ln R/\varepsilon$ records, up to factors of $\sqrt{\ln n}$ hidden in $b$.

### Derivation

1. **Privacy.** Replacing one record takes it out of one bin and puts it into another, so every count changes by at most 1. The coarse step is then the exponential mechanism with a score of sensitivity 1, which is $\varepsilon_1$-DP. For each value of $J$, the fine step is a clipped mean over a window of width $3b$, which is $\varepsilon_2$-DP by Fact 2. By Fact 5 the two steps together are $\varepsilon$-DP. The window depends on the data only through $J$, which was chosen privately, so no leak of the kind seen in [Beyond global sensitivity](07-sensitivity.md) arises.
2. **The data fill at most two bins.** By Fact 3, with probability at least $1 - \beta$ every record lies within $\tau = b/2$ of $\mu$. Then the records span an interval of length at most $b$, which meets at most two neighbouring bins. So the fullest bin holds at least $n/2$ records, and every other bin except one is empty.
3. **The coarse step picks a non-empty bin.** An empty bin has score 0, at least $n/2$ below the best. Fact 4 with $g = n/2$ bounds the chance of picking one by $m e^{-\varepsilon_1 n/4}$. Setting this equal to $\beta$ and solving for $n$ gives @eq-mean-coarse-n.
4. **The window holds every record.** A non-empty bin $J$ contains a record within $b/2$ of $\mu$, so $\mu$ lies within $b/2$ of the bin. Every record lies within $b/2$ of $\mu$, hence within $b$ of the bin, which is inside the window.
5. **Error.** On the events of steps 2 and 3, the fine step clips nothing, so @eq-mean-split has no bias, a sampling term $1/n$, and the noise term @eq-mean-laplace-noise with $w = 3b$ and $\varepsilon_2$. That is @eq-mean-coarse.

### Interpretation

The range $R$ now enters only through the number of bins $m$, and only inside a logarithm in @eq-mean-coarse-n. The packing bound @eq-lower-mean-packing of [the lower-bounds chapter](10-lower-bounds.md) says a pure-DP mechanism must pay that logarithm, so the coarse step is optimal up to constants. The fine step pays for a window of width $3b = 6\sqrt{2\ln(2n/\beta)}$, a few standard deviations, which is where the $\ln n$ factors come from.

The split has a cost when you already know a lot. The fine step gets only $\varepsilon_2$ and clips to a window wider than the spread of the data, so for small $R$ the naive estimator wins. With $n = 1000$, $\varepsilon = 1$ and a quarter of the budget for the coarse step, the crossover is at $R = 13.81$ (@fig-mean-range). The quarter is a simple default rather than the best split, and [Exercise 2](../../book-exercises/part03-dp-foundations/11-mean-estimation.md#sec-sol-mean-2) sizes the coarse budget from @eq-mean-coarse-n instead.

Under approximate DP the range can go altogether. The [stability-based release](07-sensitivity.md#sec-sens-stable) reports the fullest bin exactly, with high probability, once it leads the others by about $2\ln(1/\delta)/\varepsilon$ records, whatever the number of bins. Its test is noisy, so a smaller lead gives a refusal rather than a wrong bin. Karwa & Vadhan use a stability-based histogram of this kind, so their $(\varepsilon, \delta)$-DP coarse step needs a number of records of order $\ln(1/\delta)/\varepsilon$ and works even with no bound $R$ at all. That matches the packing bound under approximate DP, @eq-lower-packing-approx.

The same trick handles an unknown variance. Differences of pairs of records, $X_2 - X_1$ and so on, have mean 0 and a spread set by the variance alone. Put the sizes $|X_2 - X_1|, |X_4 - X_3|, \dots$ into bins whose edges double, from $[1, 2)$ to $[2, 4)$ and so on, scaled to the possible range of the standard deviation $\sigma$. Most sizes fall within a factor of a few of $\sigma$, so the fullest bins locate $\sigma$ within a constant factor. With $\sigma$ known to lie between $\sigma_{\min}$ and $\sigma_{\max}$ there are about $\log_2(\sigma_{\max}/\sigma_{\min})$ bins, and choosing a full one with the exponential mechanism costs, as in Fact 4, records of order the logarithm of that number divided by $\varepsilon$ (Karwa & Vadhan 2018 give the details).

### Example: three ranges again

Take $n = 1000$, $\varepsilon = 1$ and $\beta = 0.05$ again, with $\varepsilon_1 = 0.25$ and $\varepsilon_2 = 0.75$. The bins have width $b = 9.207$ and the window $27.62$. @tbl-mean-coarse compares the two estimators. The fine step's noise has standard deviation 0.0521 and the RMSE is 0.0609 for every $R$, while the naive estimator's error grows with $R$. The number of records the coarse step needs, by @eq-mean-coarse-n, grows from 74 to 245 as $R$ grows by a factor of $10^5$. The bound on its failure probability is below $10^{-21}$ in all three cases, because 1,000 records is far more than it needs.

| Range $R$ | Bins $m$ | Coarse step needs $n \ge$ | Naive RMSE | Coarse then fine: RMSE | Simulated, naive | Simulated, coarse then fine |
| --- | --- | --- | --- | --- | --- | --- |
| 10 | 5 | 74 | 0.0520 | 0.0609 | 0.0517 | 0.0588 |
| $10^3$ | 220 | 135 | 2.8416 | 0.0609 | 2.7615 | 0.0606 |
| $10^6$ | 217,223 | 245 | 2828.44 | 0.0609 | 2803.08 | 0.0596 |

: The naive and coarse-then-fine estimators of a normal mean, $n = 1000$, $\varepsilon = 1$, $\beta = 0.05$, a quarter of the budget for the coarse step: the formulas @eq-mean-naive and @eq-mean-coarse, and simulations with 4,000 repetitions, whose own error is about 2%. {#tbl-mean-coarse}

@fig-mean-coarse shows the coarse step on one dataset with $R = 100$ and $\mu = 37$, which has 24 bins. On the script's own draw of such a dataset, 876 and 124 records fall into the two non-empty bins. The exponential mechanism picks the fuller one with a probability that prints as 1 to twelve digits, and any given empty bin with probability about $2.8\times10^{-48}$.

![Top: the coarse histogram of 1,000 normal records with mean 37, in 24 bins of width 9.2 covering a range with $R = 100$ (a fresh draw). Bottom: the probability that the exponential mechanism with $\varepsilon_1 = 0.25$ picks each bin, on a log scale.](figures/mean_coarse.svg){#fig-mean-coarse}

![Root-mean-square error of the naive estimator and of coarse then fine, against the bound $R$ on the mean, for $n = 1000$ normal records and $\varepsilon = 1$. The dots are simulations of coarse then fine; the dotted line is the non-private error $1/\sqrt n$.](figures/mean_range.svg){#fig-mean-range}

### Check yourself

- Why is the bin width $2\tau$ and not, say, $\tau/10$? *(The proof needs the data to fill at most two bins, so that the fullest bin holds at least half the records. Narrower bins split the data over many bins, and the lead of the best bin over an empty one shrinks.)*
- The window has width $3b$. Why not clip to bin $J$ alone? *(The data can straddle two bins, and $J$ may be the less full one. Adding a neighbour on each side guarantees that every record is inside, by step 4.)*

## Many dimensions: shrinking balls {#sec-mean-multi}

### Context

Now each record is a vector in $d$ dimensions, drawn from a normal distribution with mean $\mu$ and identity covariance, and all you know is that the length of $\mu$, $\|\mu\|$, is at most $R$. Error is measured by the length of $\hat\mu - \mu$. A histogram over cells of a grid would need about $(R/b)^d$ cells, and the logarithm of that is $d\ln(R/b)$, a cost linear in $d$. Biswas, Dong, Kamath & Ullman (2020) shrink a ball instead. Keep a ball that contains $\mu$, clip the data to a slightly larger ball, take a noisy mean, and use it to draw a smaller ball. A few rounds take the ball from radius $R$ down to the spread of the data.

### Statement

The noise is that of the [Gaussian mechanism](03-laplace-gaussian.md), calibrated exactly. Write $\Delta_2$ for the $\ell_2$ sensitivity of a query, the most one record can move its answer in length. Gaussian noise with standard deviation $\sigma$ makes the query $(\varepsilon, \delta)$-DP when $\Delta_2/\sigma$ is at most a number $\mu(\varepsilon, \delta)$, which that chapter computes from the exact privacy curve. (This $\mu(\varepsilon, \delta)$ is a ratio, not the mean.) Two lengths set the scale. The first bounds how far records lie from the mean,
$$\tau_d = \sqrt d + \sqrt{2\ln(2n/\beta)},$$
and the second, $u$, is a margin for the $T + 1$ steps of the mechanism, $T$ rounds and a final step:
$$u = \sqrt{2\ln(4(T + 1)/\beta)}.$$

Start from the ball of radius $r_0 = R$ around 0. In round $t = 0, \dots, T - 1$, with centre $c_t$ and radius $r_t$:

1. Move every record that lies outside the ball of radius $r_t + \tau_d$ around $c_t$ onto its surface.
2. Release the average plus Gaussian noise with standard deviation $\sigma_t = 2(r_t + \tau_d)/(n\,\mu(\varepsilon_t, \delta_t))$, and call the result $c_{t+1}$.
3. Set the next radius to
$$r_{t+1} = \frac{\sqrt d + u}{\sqrt n} + \sigma_t\big(\sqrt d + u\big).$$ {#eq-mean-shrink}

A final step does the same once more with the rest of the budget and releases the noisy average. If the rounds share $\varepsilon_t$ and $\delta_t$ so that all the steps add up to $\varepsilon$ and $\delta$, the mechanism is $(\varepsilon, \delta)$-DP. With probability at least $1 - \beta$, every ball contains $\mu$ and no record is ever moved. On that event the final release is the plain average plus noise with standard deviation $\sigma_T$, and ignoring the rare event where it fails,
$$\mathrm{RMSE} = \sqrt{\frac dn + d\,\sigma_T^2}.$$ {#eq-mean-multi}
Reaching an error $\alpha$ takes about $d/\alpha^2 + d\sqrt{\ln(1/\delta)}/(\alpha\varepsilon) + \sqrt d\ln R/\varepsilon$ records, up to logarithmic factors.

### Derivation

1. **Records stay within $\tau_d$ of the mean.** For a standard normal vector $Z$ in $d$ dimensions, the probability that its length exceeds $\sqrt d + t$ is at most $e^{-t^2/2}$. This holds because the length is a function that moves by at most $|a - b|$ when $Z$ moves by a vector of length $|a - b|$, and such functions of normal vectors concentrate around their average, which here is at most $\sqrt d$ (Boucheron, Lugosi & Massart 2013, ch. 5). A union bound over the $n$ records, each allowed $\beta/(2n)$, gives $\tau_d$.
2. **Privacy of one step.** After step 1 every record lies in a ball of radius $r_t + \tau_d$, whose diameter is $2(r_t + \tau_d)$. Replacing one record therefore moves the sum by a vector of length at most $2(r_t + \tau_d)$, and the average by that divided by $n$, which is the $\ell_2$ sensitivity. The noise $\sigma_t$ is calibrated to it, so the step is $(\varepsilon_t, \delta_t)$-DP for each value of the centre, which was chosen privately. Fact 5 adds up the steps.
3. **The balls keep the mean.** Suppose $\mu$ lies in the ball of round $t$ and the event of step 1 holds. Then every record lies within $\tau_d$ of $\mu$, hence inside the clipping ball, and nothing moves. So $c_{t+1} - \mu$ is the sampling error $\bar X - \mu$ plus the noise. The sampling error is a normal vector with variance $1/n$ in each coordinate, so its length is at most $(\sqrt d + u)/\sqrt n$, and the noise's length is at most $\sigma_t(\sqrt d + u)$, each with probability at least $1 - \beta/(4(T + 1))$ by the bound of step 1. So $\mu$ lies in the next ball, of radius @eq-mean-shrink. A union bound over all the steps keeps the total failure probability below $\beta$.
4. **Final error.** On that event, the final release is $\bar X$ plus independent noise. The expected squared length of $\bar X - \mu$ is $d/n$, the sum of $d$ variances of $1/n$, and that of the noise is $d\sigma_T^2$. The cross term has mean 0, as in Fact 1, which gives @eq-mean-multi.
5. **The rates.** In a round, the radius drops from $r_t$ to about $2(r_t + \tau_d)\sqrt d/(n\,\mu(\varepsilon_t, \delta_t))$. When $n$ is larger than a constant times $\sqrt d/\mu(\varepsilon_t, \delta_t)$, which is of order $\sqrt{d\ln(1/\delta)}/\varepsilon_t$, that is a reduction by a constant factor, and about $\ln R$ rounds bring it down to the order of $\tau_d \approx \sqrt d$. In the final step the sensitivity is then about $2\sqrt d/n$, so $d\sigma_T^2$ is about $4d^2/(n\,\mu)^2$, and an error $\alpha$ needs $n$ of order $d/(\alpha\,\mu)$, that is, $d\sqrt{\ln(1/\delta)}/(\alpha\varepsilon)$.

### Interpretation

Without privacy the error is $\sqrt{d/n}$. The privacy noise is about $d/(n\varepsilon)$ up to logarithmic factors, so it grows like $d$ instead of $\sqrt d$, and it is larger than the sampling error $\sqrt{d/n}$ unless $n$ is above about $d/\varepsilon^2$. That is not a weakness of the method. The [fingerprinting lower bound](10-lower-bounds.md#sec-lower-fingerprinting) says that releasing $d$ means with an error $\alpha'$ in each coordinate needs about $\sqrt d/(\alpha'\varepsilon)$ records. An $\ell_2$ error $\alpha$ spread over $d$ coordinates is an error of about $\alpha/\sqrt d$ in each, so substituting $\alpha' = \alpha/\sqrt d$ gives $d/(\alpha\varepsilon)$. [Exercise 5](../../book-exercises/part03-dp-foundations/11-mean-estimation.md#sec-sol-mean-5) does the substitution with the constants, and Kamath, Li, Singhal & Ullman (2019) prove the matching lower bound directly for normal data.

The range costs $\sqrt d\ln R/\varepsilon$, which is less than the $d\ln R/\varepsilon$ of a grid histogram, because each round makes progress in all $d$ directions at once. The rounds compose here by Fact 5, which adds the budgets. The tighter accountants of Part 4 make several rounds cheaper, and the original method uses one of them.

This is the shape of every private average of vectors, including the average of clipped gradients in private training. There the clipping radius is a fixed number rather than a shrinking ball, and choosing it is the clipping trade-off of [Beyond global sensitivity](07-sensitivity.md#sec-sens-clip) in many dimensions.

### Example: a thousand dimensions

Take $n = 100{,}000$ records with $\|\mu\| \le 1000$, at $\varepsilon = 1$ and $\delta = 10^{-6}$, with $\beta = 0.05$ and a quarter of the budget for the rounds. Here $\mu(1, 10^{-6}) = 0.2367$. @tbl-mean-multi compares the naive estimator, which clips to the ball of radius $R + \tau_d$, with shrinking balls. The script tries every number of rounds up to 29 and keeps the best, which is two for each $d$. For $d = 100$ the radius falls from 1000 to 8.54 after one round and to 0.243 after the second. The ratio of the privacy noise to $d/(n\varepsilon)$ falls from 32.2 at $d = 10$ to 14.2 at $d = 1000$, as the terms in $\sqrt d$ give way to the term in $d$. A simulation at $d = 100$ with 40 datasets gives an RMSE of 0.0363 against the formula's 0.0366, and every ball contained $\mu$ in all 40.

| $d$ | Non-private $\sqrt{d/n}$ | Naive RMSE | Rounds | Radii | Shrinking balls: RMSE |
| --- | --- | --- | --- | --- | --- |
| 10 | 0.0100 | 0.2697 | 2 | 1000, 4.12, 0.0725 | 0.0105 |
| 100 | 0.0316 | 0.8586 | 2 | 1000, 8.54, 0.243 | 0.0366 |
| 1000 | 0.1000 | 2.7730 | 2 | 1000, 22.9, 1.43 | 0.1736 |

: The mean of $n = 100{,}000$ normal vectors in $d$ dimensions with $\|\mu\| \le 1000$, at $(1, 10^{-6})$-DP, by the naive estimator and by shrinking balls: the $\ell_2$ RMSE from @eq-mean-multi. {#tbl-mean-multi}

![The $\ell_2$ root-mean-square error of the mean of 100,000 normal vectors against the dimension $d$, at $(1, 10^{-6})$-DP with $\|\mu\| \le 1000$: clipping to the ball of radius $R + \tau_d$, shrinking balls, and the non-private error $\sqrt{d/n}$.](figures/mean_dimension.svg){#fig-mean-dimension}

### Check yourself

- Why is the noise in the final step set by $\tau_d \approx \sqrt d$ and not by the final radius, which can be much smaller? *(The records themselves are spread about $\sqrt d$ around $\mu$, so the clipping ball must have radius at least $\tau_d$ to leave them alone. The sensitivity is the diameter of that ball divided by $n$.)*
- What would a shrinking round do if $n$ were smaller than about $\sqrt d/\mu(\varepsilon_t, \delta_t)$? *(The new radius from @eq-mean-shrink would be larger than the old one, so the ball would not shrink. The method needs enough records per round, which is where the $\sqrt d\ln R/\varepsilon$ term comes from, and [Exercise 4](../../book-exercises/part03-dp-foundations/11-mean-estimation.md#sec-sol-mean-4) computes it.)*

## Heavy tails: clip into the data {#sec-mean-heavy}

### Context

Normal data rarely stray far, so every estimator so far could clip without touching a record. Many real quantities have heavy tails: incomes, session lengths, the norms of per-example gradients. Suppose all you know is that the variance is at most $\sigma^2$. Chebyshev's inequality then only says that a record is more than $\sigma\sqrt{n/\beta}$ from the mean with probability at most $\beta/n$, so a clipping range wide enough to touch no record has half-width $\sigma\sqrt{n/\beta}$, and its noise is enormous. The way out is to clip into the data on purpose and accept a bias.

### Statement

Suppose a coarse step has placed a centre $c$ near the mean; to keep the formulas clean, take $c = \mu$. Clip every record to $[\mu - \tau, \mu + \tau]$, average, and add $\mathrm{Lap}(2\tau/(n\varepsilon))$. This is $\varepsilon$-DP for every $\tau$, and for any distribution with variance at most $\sigma^2$,
$$\mathrm{RMSE} \le \frac{\sigma}{\sqrt n} + \frac{\sigma^2}{\tau} + \frac{2\sqrt2\,\tau}{n\varepsilon}.$$ {#eq-mean-heavy}
The right side is smallest at
$$\tau_b = \sigma\sqrt{\frac{n\varepsilon}{2\sqrt2}}, \qquad\text{where it equals}\qquad \frac{\sigma}{\sqrt n} + 2\sigma\sqrt{\frac{2\sqrt2}{n\varepsilon}}.$$ {#eq-mean-heavy-opt}
So an error $\alpha$ needs about $\sigma^2/\alpha^2 + 8\sqrt2\,\sigma^2/(\alpha^2\varepsilon)$ records, of order $1/(\alpha^2\varepsilon)$. Without clipping into the data, at $\tau = \sigma\sqrt{n/\beta}$, it would need of order $1/(\beta\alpha^2\varepsilon^2)$.

### Derivation

1. **Bias.** Clipping moves a record $X$ toward $\mu$ by $(|X - \mu| - \tau)_+$, the amount by which its distance from $\mu$ exceeds $\tau$, or 0. So the bias is at most the average of that amount. The amount is positive only where $|X - \mu| > \tau$, and there it is at most $|X - \mu| \le (X - \mu)^2/\tau$, because dividing $|X - \mu|$ by $\tau$ gives a number above 1. Averaging, the bias is at most $\mathbb E[(X - \mu)^2]/\tau \le \sigma^2/\tau$.
2. **Sampling.** A variance is half the average squared difference between two independent copies of a variable. Clipping to an interval never moves two values further apart, so the clipped record varies at most as much as the record itself, and its variance is at most $\sigma^2$. The sampling term of @eq-mean-split is at most $\sigma^2/n$.
3. **Noise.** The clipping interval has width $2\tau$, so by @eq-mean-laplace-noise the noise standard deviation is $\sqrt2\cdot2\tau/(n\varepsilon)$.
4. **Add.** By @eq-mean-split, the RMSE is the length of a vector with these three components, and a length is at most the sum of its components. That gives @eq-mean-heavy.
5. **Minimize.** The two terms that depend on $\tau$ are $\sigma^2/\tau$, falling, and $2\sqrt2\,\tau/(n\varepsilon)$, rising. The derivative of their sum, $-\sigma^2/\tau^2 + 2\sqrt2/(n\varepsilon)$, is zero at $\tau_b$, and there the two terms are equal, each $\sigma\sqrt{2\sqrt2/(n\varepsilon)}$. That gives @eq-mean-heavy-opt. Requiring the second term of @eq-mean-heavy-opt to be at most $\alpha$ and squaring gives the $8\sqrt2\,\sigma^2/(\alpha^2\varepsilon)$ records.

### Interpretation

With heavy tails, privacy stops being a lower-order cost. For bounded data the privacy term in @eq-mean-bounded-n had a single $1/\alpha$. Here it has $1/\alpha^2$, the same power as sampling, divided by $\varepsilon$. Kamath, Singhal & Ullman (2020) show that this is unavoidable: no $\varepsilon$-DP mechanism does better for every distribution with bounded variance. They also show what more knowledge buys. If the $k$-th moment $\mathbb E|X - \mu|^k$ is bounded for some $k > 2$, the cost falls to about $1/(\varepsilon\,\alpha^{k/(k-1)})$ records, and as $k$ grows it approaches the $1/(\alpha\varepsilon)$ of bounded data.

The bound @eq-mean-heavy is a worst case, and it is tight up to a constant factor. A sharper version of step 1 gives $\sigma^2/(4\tau)$, because $(|y| - \tau)_+ \le y^2/(4\tau)$ for every $y$, and a distribution that puts a little mass at distance $2\tau$ on one side of the mean attains that. So the worst case hides its variance in a few far-out records on one side. For an ordinary heavy-tailed distribution the bias falls faster than $\sigma^2/\tau$, the best $\tau$ is smaller than $\tau_b$, and the error is lower. But $\tau_b$ uses only $n$, $\varepsilon$ and the variance bound, so it can be set without looking at the data. Setting $\tau$ from the data, such as to the observed maximum, would bring back the leak of [Beyond global sensitivity](07-sensitivity.md#sec-sens-local).

### Example: Pareto records

Take a Pareto distribution with shape 3, the law of $x_m U^{-1/3}$ for a uniform $U$ on $(0, 1)$, scaled so that its variance is 1. Its smallest value is $x_m = 1.1547$ and its mean is 1.7321, and its third moment is infinite. With $n = 1000$ and $\varepsilon = 1$, @tbl-mean-heavy compares three choices of $\tau$. Clipping wide enough to touch nothing costs an RMSE of 0.4012. The worst-case choice $\tau_b = 18.80$ gives 0.0607, and the best $\tau$ for this particular distribution, 5.36, gives 0.0329, close to the non-private 0.0316. The bound @eq-mean-heavy-opt promises 0.1380, which is loose for this distribution, as the interpretation predicted ([Exercise 3](../../book-exercises/part03-dp-foundations/11-mean-estimation.md#sec-sol-mean-3) computes the exact bias). At $n = 100{,}000$ the pattern repeats, with RMSE 0.0401, 0.0062 and 0.0032 for the three choices.

| $\tau$ | Bias | Sampling sd | Noise sd | RMSE | Simulated |
| --- | --- | --- | --- | --- | --- |
| 141.42 (no record clipped) | 0.0000 | 0.0313 | 0.4000 | 0.4012 | 0.3997 |
| 18.80 (worst-case best, $\tau_b$) | −0.0018 | 0.0293 | 0.0532 | 0.0607 | 0.0591 |
| 5.36 (best for this distribution) | −0.0153 | 0.0249 | 0.0152 | 0.0329 | 0.0325 |

: Clipped private means of Pareto records with variance 1, $n = 1000$, $\varepsilon = 1$: the exact terms of @eq-mean-split for three clipping half-widths $\tau$ around the mean, and simulations with 4,000 repetitions. {#tbl-mean-heavy}

![Error of the clipped private mean of Pareto records with variance 1, $n = 1000$, $\varepsilon = 1$, against the clipping half-width $\tau$: the RMSE, the exact bias and its bound $\sigma^2/\tau$, the noise and the sampling standard deviations, and simulated points. The dashed vertical line is $\tau_b$ and the dash-dotted one the width that clips no record.](figures/mean_heavy.svg){#fig-mean-heavy}

### Check yourself

- Why does the bias of clipping at $\tau$ fall faster than $\sigma^2/\tau$ for the Pareto records? *(For this law the average excess beyond a point $t$, $\mathbb E[(X - t)_+]$, equals $x_m^3/(2t^2)$, which falls like $t^{-2}$. The bound is close to tight only when the variance sits in a few far points on one side of the mean.)*
- If the variance bound $\sigma^2$ is four times too large, how does $\tau_b$ change, and is the release still private? *(It doubles, since $\tau_b$ is proportional to $\sigma$. It stays private, because privacy holds for every $\tau$. Only the accuracy suffers, from the extra noise.)*

## Code: check every number by computation

The script [code/mean_estimation.py](../../code/part03-dp-foundations/mean_estimation.py) computes every number in this chapter and draws its figures. Each estimator and each error formula is a small typed function. The exact errors use the mean and variance of a clipped normal or Pareto variable, and the simulations run the mechanisms on fresh data. From the repository root:

```bash
uv run python code/part03-dp-foundations/mean_estimation.py
```

@tbl-mean-results shows the output with seed 0.

| Experiment | Result |
| --- | --- |
| Spread of 1,000 normal records, $\beta = 0.05$ | $\tau = 4.604$; exceeded in a fraction 0.0039 of 20,000 simulated datasets |
| Survey proportion, $\varepsilon = 1$ | 108 records for RMSE 0.05 (100 without privacy); 2,508 for 0.01 (2,500) |
| Naive estimator, $n = 1000$ | RMSE 0.0520, 2.8416 and 2828.44 for $R = 10$, $10^3$, $10^6$ |
| Coarse then fine, $n = 1000$ | RMSE 0.0609 for every $R$ (simulated 0.0588, 0.0606, 0.0596); coarse step needs 74, 135, 245 records; beats naive from $R = 13.81$ |
| Shrinking balls, $n = 100{,}000$, $R = 1000$ | RMSE 0.0105, 0.0366, 0.1736 for $d = 10$, 100, 1000, against 0.0100, 0.0316, 0.1000 without privacy; simulated 0.0363 at $d = 100$ |
| Pareto records, $n = 1000$ | RMSE 0.4012 without clipping into the data, 0.0607 at $\tau_b$, 0.0329 at the best $\tau$; bound 0.1380 |

: Results of the chapter's script with seed 0. {#tbl-mean-results}

[tests/test_mean_estimation.py](../../tests/test_mean_estimation.py) checks that the output still matches this table.

**What to take from it.** For bounded data, privacy costs a handful of records on top of what sampling needs. The range is the expensive unknown for the naive estimator, whose error grows by a factor of $10^5$ across the three ranges, while coarse then fine stays at 0.0609 throughout and needs only 245 records for its coarse step even at $R = 10^6$. In many dimensions the privacy noise grows like $d$, against $\sqrt d$ for sampling, so at $d = 1000$ it is already the larger part of the error. Heavy tails change the rate itself, and clipping into the data on purpose is what keeps the error near the non-private one.

## References

- Karwa & Vadhan, "Finite Sample Differentially Private Confidence Intervals" (ITCS 2018): coarse then fine for normal means and variances, under pure and approximate DP
- Biswas, Dong, Kamath & Ullman, "CoinPress: Practical Private Mean and Covariance Estimation" (NeurIPS 2020): shrinking confidence balls for means and covariances, with code
- Kamath, Li, Singhal & Ullman, "Privately Learning High-Dimensional Distributions" (COLT 2019): the $d/(\alpha\varepsilon)$ cost for normal means in $d$ dimensions, with a matching lower bound
- Kamath, Singhal & Ullman, "Private Mean Estimation of Heavy-Tailed Distributions" (COLT 2020): the $1/(\alpha^2\varepsilon)$ rate for bounded variance and its generalization to $k$-th moments, with lower bounds
- Kamath & Ullman, "A Primer on Private Statistics" (arXiv 2020): the univariate and multivariate cases and the lower bounds, at textbook depth
- Kamath, *CS 860: Algorithms for Private Data Analysis* (University of Waterloo, lectures on private mean estimation): the ladder of rates from bounded to heavy-tailed data that organizes this chapter
- Bun & Steinke, "Average-Case Averages: Private Algorithms for Smooth Sensitivity and Mean Estimation" (NeurIPS 2019): mean estimation by smooth sensitivity of a trimmed mean
- Boucheron, Lugosi & Massart, *Concentration Inequalities* (Oxford, 2013), ch. 5: concentration of the length of a normal vector

## Cheat sheet

@tbl-mean-cheat collects the results. Records are independent, neighbours differ in one replaced record, and $\tau = \sqrt{2\ln(2n/\beta)}$.

| Setting | Mechanism | Records for error $\alpha$ (up to log factors) |
| --- | --- | --- |
| Bounded, width $w$, variance $\sigma^2$ | Mean $+ \mathrm{Lap}(w/(n\varepsilon))$ | $\sigma^2/\alpha^2 + \sqrt2\,w/(\alpha\varepsilon)$ |
| Normal, $\lvert\mu\rvert \le R$, naive | Clip to $[-R - \tau, R + \tau]$, Laplace | $1/\alpha^2 + R/(\alpha\varepsilon)$ |
| Normal, $\lvert\mu\rvert \le R$, coarse then fine | Exponential mechanism on bins of width $2\tau$, then clip to 3 bins | $1/\alpha^2 + 1/(\alpha\varepsilon) + \ln R/\varepsilon$; coarse step $4\ln(m/\beta)/\varepsilon_1$ |
| Same, approximate DP | Stability-based histogram for the coarse step | $1/\alpha^2 + 1/(\alpha\varepsilon) + \ln(1/\delta)/\varepsilon$, no $R$ |
| $d$ dimensions, $\lVert\mu\rVert \le R$ | Shrinking balls, Gaussian noise, radius @eq-mean-shrink | $d/\alpha^2 + d\sqrt{\ln(1/\delta)}/(\alpha\varepsilon) + \sqrt d\ln R/\varepsilon$ |
| Variance $\le \sigma^2$ only | Clip to $\mu \pm \tau_b$, $\tau_b = \sigma\sqrt{n\varepsilon/(2\sqrt2)}$ | $\sigma^2/\alpha^2 + 8\sqrt2\,\sigma^2/(\alpha^2\varepsilon)$ |
| $k$-th moment bounded | Clip into the data | $1/\alpha^2 + 1/(\varepsilon\,\alpha^{k/(k-1)})$ |

: Summary of the results in this chapter. {#tbl-mean-cheat}

**Rules of thumb**

- Privacy must hold for every dataset; let assumptions about the data affect only accuracy, through how often clipping changes nothing.
- Never clip to a range set by your ignorance. Spend a quarter of the budget or less finding the data first, which costs about $\ln R/\varepsilon$ records.
- In $d$ dimensions, budget about $d\sqrt{\ln(1/\delta)}/(\alpha\varepsilon)$ records for privacy on top of $d/\alpha^2$ for sampling.
- With heavy tails, clip into the data. Choose the width from $n$, $\varepsilon$ and a variance bound, never from the data.

**Practice.** Exercises for this chapter, each with a worked solution, are in [the exercise book](../../book-exercises/part03-dp-foundations/11-mean-estimation.md).
