# Exercises: Private prediction

Exercises for [the chapter on private prediction](../../book-full/part05-dp-ml/05-private-prediction.md), each with a worked solution. Try each exercise before opening its solution. The script `code/part05-dp-ml/solutions_private_prediction.py` computes every number quoted here; run it with `uv run python code/part05-dp-ml/solutions_private_prediction.py`.

The solutions use the chapter's notation. To give differential privacy (DP) to predictions, $k$ teachers are trained on disjoint shards of the private data and each query is answered by a noisy vote: LNMax adds $\mathrm{Lap}(b)$ noise to every vote count, GNMax adds $\mathcal N(0, \sigma^2)$, and only the label with the largest noisy count is released. The gap $g$ is the difference between the top two counts, $Q$ the number of answered queries, and $\delta = 10^{-5}$. Costs are measured in Rényi differential privacy (RDP) of order $\alpha$; one GNMax answer is $(\alpha, \alpha/\sigma^2)$-RDP. Private aggregation of teacher ensembles (PATE) trains a student on public inputs labelled this way.

## The failure probability of a Laplace vote {#sec-sol-pate-1}

> **Exercise 1.** **The failure probability of a Laplace vote.** Show that if $Z_1$ and $Z_2$ are independent $\mathrm{Lap}(b)$, then for $t \ge 0$, $\Pr[Z_1 - Z_2 > t] = \frac14(2 + t/b)e^{-t/b}$. Use it to compute the probability that LNMax with $b = 5$ returns the runner-up on a two-label query with gap 10, and compare with GNMax at the same noise standard deviation, using @eq-pate-two-class. Repeat for a gap of 30. *Check: 0.1353 (simulated 0.1356) against 0.1587; at gap 30, $4.96 \times 10^{-3}$ against $1.35 \times 10^{-3}$.* *(Convolution of densities)*

::: {.callout-tip collapse="true" title="Solution"}
1. **Condition on $Z_2$.** $\Pr[Z_1 - Z_2 > t] = \mathbb E\big[\Pr[Z_1 > t + Z_2 \mid Z_2]\big]$, and for a Laplace variable $\Pr[Z_1 > s] = \frac12e^{-s/b}$ when $s \ge 0$ and $1 - \frac12e^{s/b}$ when $s < 0$.
2. **Split the integral at $Z_2 = -t$.** For $z \ge -t$, $t + z \ge 0$ and the inner probability is $\frac12e^{-(t + z)/b}$. Integrating against the density $\frac1{2b}e^{-|z|/b}$ over $z \in [-t, 0]$ gives $\frac14e^{-t/b}\cdot\frac tb$, and over $z \ge 0$ gives $\frac14e^{-t/b}\cdot\frac12$.
3. **The rest.** For $z < -t$, the inner probability is $1 - \frac12e^{(t + z)/b}$; the density there integrates to $\frac12e^{-t/b}$, and the second part contributes $-\frac14e^{t/b}\int_{-\infty}^{-t}\frac1be^{2z/b}dz = -\frac18e^{-t/b}$. Together, $\frac12e^{-t/b} - \frac18e^{-t/b} = \frac38e^{-t/b}$.
4. **Add.** $\frac14\frac tbe^{-t/b} + \frac18e^{-t/b} + \frac38e^{-t/b} = \frac14(2 + t/b)e^{-t/b}$. At $t = 0$ this is $\frac12$, as symmetry requires.
5. **Numbers.** With $t = 10$ and $b = 5$: $\frac14 \times 4 \times e^{-2} = e^{-2} = 0.1353$, and a simulation with a million draws gives 0.1356. The Laplace standard deviation is $\sqrt2b = 7.07$; GNMax with $\sigma = 7.07$ fails with probability $\Phi(-10/(\sqrt2 \times 7.07)) = \Phi(-1) = 0.1587$. At a gap of 30, Laplace gives $\frac14 \times 8 \times e^{-6} = 4.96 \times 10^{-3}$, Gaussian $\Phi(-3) = 1.35 \times 10^{-3}$.

*At small gaps the two are close, the Laplace slightly better; at large gaps the Laplace tail, which falls like $e^{-t/b}$ rather than $e^{-t^2/(4\sigma^2)}$, is much heavier. This is the Laplace failure bound that Papernot et al. (2017) use in their data-dependent analysis, and one reason GNMax replaced it.*
:::

## The gap a reliable answer needs {#sec-sol-pate-2}

> **Exercise 2.** **The gap a reliable answer needs.** GNMax answers 1,000 queries at $\varepsilon = 4$. Using @eq-pate-sigma and @eq-pate-two-class, find the gap at which a two-label answer returns the plurality with probability 0.9 and 0.99, and the fraction of 250 teachers that must agree for that gap. Compare with @tbl-pate-teachers. *Check: $\sigma = 51.8$; gaps 93.8 and 170.3; fractions 0.688 and 0.841.* *(Inverting a normal probability)*

::: {.callout-tip collapse="true" title="Solution"}
1. **The noise.** By @eq-pate-sigma, $\sigma = \sqrt{1{,}000/0.3731} = 51.8$.
2. **Invert @eq-pate-two-class.** $\Phi(g/(\sqrt2\sigma)) = p$ gives $g = \sqrt2\sigma\,\Phi^{-1}(p)$. With $\Phi^{-1}(0.9) = 1.2816$ and $\Phi^{-1}(0.99) = 2.3263$: $g = 93.8$ and $g = 170.3$.
3. **Agreement.** If $a$ of the 250 teachers agree, the gap is $250a - 250(1 - a) = 250(2a - 1)$, so $a = (250 + g)/500$: 0.688 and 0.841.
4. **Compare.** With 250 teachers, the median gap on the running example is 0.408 of 250, about 102, so the median query is answered correctly with probability a little above 0.9; a substantial share of queries has much smaller gaps.

*Reliable answers need strong agreement: more than 84% of the teachers on one side for 99% reliability at this noise. Teachers that each learn the task well are worth more than many teachers that each learn it a little.*
:::

## Where Gaussian noise overtakes Laplace noise {#sec-sol-pate-3}

> **Exercise 3.** **Where Gaussian noise overtakes Laplace noise.** For a budget of $\varepsilon = 1$ spread over $Q$ answers, compare the noise standard deviation LNMax needs ($\sqrt2b$, each answer charged as a $(2/b)$-DP step through the randomized-response curve) with GNMax's $\sigma$. Find the smallest $Q$ at which Gaussian noise is smaller, and repeat for $\varepsilon = 4$. Explain why Laplace wins for a single answer. *Check: from 5 answers at $\varepsilon = 1$, from 6 at $\varepsilon = 4$.* *(Composition of pure and Gaussian mechanisms)*

::: {.callout-tip collapse="true" title="Solution"}
1. **One answer.** A single LNMax answer is $(2/b)$-DP, so $\varepsilon = 1$ needs $b$ close to 2; the accountant, which goes through the randomized-response curve and @eq-pate-convert, gives $b = 2.02$, a standard deviation of 2.85. A single GNMax answer needs $\sigma = 5.72$ (@tbl-pate-noise). A Gaussian has no pure-DP guarantee: for one release, its $\varepsilon$ at $\delta = 10^{-5}$ is set by how far into its tail the privacy loss must be controlled, which takes a lot of noise.
2. **Many answers.** The pure-DP curve of each Laplace answer is bounded by $\varepsilon_0$ and, at small orders, behaves like $\alpha\varepsilon_0^2/2$; summed over $Q$ answers, the conversion trades between these. The Gaussian curve is the line $\alpha/\sigma^2$, which composes exactly. As $Q$ grows, the Laplace cost per unit of noise variance is worse than the Gaussian's, and the ratio of standard deviations tends to about 2 (@tbl-pate-noise).
3. **The crossover.** Scanning $Q = 1, 2, \dots$ with the accountant, Gaussian noise is smaller from $Q = 5$ at $\varepsilon = 1$ and from $Q = 6$ at $\varepsilon = 4$.

*Use Laplace for a handful of answers and Gaussian beyond. The same pattern holds for any pure-DP versus Gaussian comparison: pure DP is cheap for few releases, Gaussian noise for many.*
:::

## Small failure probabilities are nearly free {#sec-sol-pate-4}

> **Exercise 4.** **Small failure probabilities are nearly free.** Show that the right side of @eq-pate-dd-pure has derivative $\big(e^{(\alpha - 1)\varepsilon_0} + (\alpha - 1)e^{\varepsilon_0} - \alpha\big)/(\alpha - 1)$ at $\beta = 0$, and is therefore about $\beta$ times this constant for small $\beta$. Evaluate it for $\varepsilon_0 = 0.1$ at $\alpha = 8$ and $\alpha = 32$, and compare $0.001$ times the slope with the bound at $\beta = 0.001$. *Check: slopes 0.2500 and 0.7890; $2.50 \times 10^{-4}$ against $2.50 \times 10^{-4}$, and $7.89 \times 10^{-4}$ against $7.80 \times 10^{-4}$.* *(First-order expansion)*

::: {.callout-tip collapse="true" title="Solution"}
1. **The bracket.** Let $x = e^{\varepsilon_0}$ and $f(\beta) = (1 - \beta)^\alpha(1 - x\beta)^{1 - \alpha} + \beta x^{\alpha - 1}$, so the bound is $\frac{1}{\alpha - 1}\ln f(\beta)$ and $f(0) = 1$.
2. **Differentiate at 0.** The first term's derivative at $\beta = 0$ is $-\alpha + (\alpha - 1)x$ (product rule, with $(1 - x\beta)^{1 - \alpha}$ contributing $(\alpha - 1)x$); the second term's is $x^{\alpha - 1}$. So $f'(0) = x^{\alpha - 1} + (\alpha - 1)x - \alpha$.
3. **The logarithm.** $\ln f(\beta) \approx f'(0)\beta$ near $\beta = 0$, since $f(0) = 1$. Dividing by $\alpha - 1$ gives the slope. It is non-negative: $x \ge 1$ makes $x^{\alpha - 1} \ge 1$ and $(\alpha - 1)x \ge \alpha - 1$.
4. **Numbers.** At $\alpha = 8$: $(e^{0.7} + 7e^{0.1} - 8)/7 = (2.0138 + 7.7362 - 8)/7 = 0.2500$. At $\alpha = 32$: $(e^{3.1} + 31e^{0.1} - 32)/31 = 0.7890$. At $\beta = 0.001$ the bound is $2.50 \times 10^{-4}$ and $7.80 \times 10^{-4}$, against the linear approximations $2.50 \times 10^{-4}$ and $7.89 \times 10^{-4}$: the approximation is exact to three digits at the lower order and 1% high at the higher, where the curvature of $f$ matters sooner.

*An answer that the noise almost never overturns costs almost nothing, in proportion to the chance of overturning it. That is the whole mechanism behind data-dependent accounting; the data-independent cost at $\alpha = 8$ is 0.0367, about 150 times the cost at $\beta = 0.001$.*
:::

## The budget of Confident-GNMax {#sec-sol-pate-5}

> **Exercise 5.** **The budget of Confident-GNMax.** In @tbl-pate-confident, 459 of 1,000 queries pass the check with $\sigma_1 = 150$ and are answered with $\sigma_2 = 40$. Split the RDP slope of @eq-pate-confident into the check and the answers, convert the total to $\varepsilon$, and compare with plain GNMax answering all 1,000 queries with $\sigma = 40$. For which $\sigma_1$ would the check cost as much as the answers? *Check: slopes 0.0222 and 0.2869; $\varepsilon = 3.59$ against 5.38; $\sigma_1 = 41.7$.* *(Accounting for a two-step mechanism)*

::: {.callout-tip collapse="true" title="Solution"}
1. **The check.** Each of the 1,000 checks is a Gaussian mechanism on $\max_iN_i$, sensitivity 1, noise 150: slope $1/(2 \times 150^2)$, and 1,000 of them $1{,}000/45{,}000 = 0.0222$.
2. **The answers.** 459 GNMax answers with $\sigma_2 = 40$: $459/1{,}600 = 0.2869$.
3. **Convert.** The total line has slope 0.3091, which converts to $\varepsilon = 3.59$ at $\delta = 10^{-5}$ by @eq-pate-convert.
4. **Plain GNMax.** 1,000 answers with $\sigma = 40$ have slope $1{,}000/1{,}600 = 0.625$ and $\varepsilon = 5.38$.
5. **Equal costs.** $1{,}000/(2\sigma_1^2) = 0.2869$ gives $\sigma_1 = \sqrt{1{,}000/0.5738} = 41.7$.

*Skipping 541 queries saves almost half of the slope, and the check costs less than a tenth of what it saves. The check can afford much more noise than the answers because it only has to tell strong consensus from weak, not pick a label. The accounting is valid because the budget is fixed in advance: the number of answered queries is random, and only a filter makes a random number of RDP costs compose as if it were fixed.*
:::

## One teacher can move every query's cost {#sec-sol-pate-6}

> **Exercise 6.** **One teacher can move every query's cost.** With $\sigma = 40$ at order 8, compute the data-dependent cost of @eq-pate-dd-gauss for a two-label query with gap $g$ and with gap $g - 2$, the gap after one teacher switches sides, for $g = 150$, 200 and 250. Why does this make the total data-dependent cost of $Q$ queries hard to release with noise calibrated to global sensitivity? *Check: ratios 1.096, 1.130 and 1.165; at $g = 150$, 0.00123 against 0.00135.* *(Local sensitivity of a sum)*

::: {.callout-tip collapse="true" title="Solution"}
1. **One query.** With $\tilde\beta = \frac12\operatorname{erfc}(g/80)$, @eq-pate-dd-gauss gives 0.00123 at $g = 150$ and 0.00135 at $g = 148$, a ratio of 1.096. At $g = 200$ the ratio is 1.130, and at $g = 250$, 1.165: the relative change grows with the gap, because the cost falls faster than exponentially in the gap.
2. **All queries.** One record changes one teacher, and that teacher may switch sides on every query, so every term of the sum can move at once. The local sensitivity of the total, how much it changes for this dataset's neighbours, is a sum of $Q$ such changes, each small in absolute terms.
3. **Global sensitivity.** Over all datasets, the worst case includes histograms near a tie, where the answer is decided by one teacher and the cost jumps between nearly 0 and the full $\alpha/\sigma^2$. The global sensitivity of the total is then close to the data-independent total itself, and noise of that size would make the released $\varepsilon$ useless.
4. **The fix.** Smooth sensitivity ([Beyond global sensitivity](../../book-full/part03-dp-foundations/07-sensitivity.md#sec-sens-smooth)) scales the noise to how much the total can change near the actual dataset, discounted with distance, which on a dataset with strong consensus is small. This is what Papernot et al. (2018) do; in the chapter's example, switching all of one teacher's votes on 1,000 queries moves the data-dependent $\varepsilon$ only from 4.375 to 4.389.

*A data-dependent cost is a statistic of the private data like any other. It is cheap to release only because its local behaviour is tame, and that has to be proved and paid for, not assumed.*
:::

## Subsampling in private in-context learning {#sec-sol-pate-7}

> **Exercise 7.** **Subsampling in private in-context learning.** In @tbl-pate-icl, 1,000 queries at $\varepsilon = 1$ use $\sigma = 7.39$ with Poisson sampling rate 0.04. Compute $\varepsilon$ for the same $\sigma$ and $Q$ without subsampling, and the $\sigma$ that 1,000 answers at $\varepsilon = 1$ need without subsampling. How many queries does each record take part in, on average? *Check: $\varepsilon = 1.00$ against 45.8; $\sigma = 180.9$; 40 queries.* *(Amplification by subsampling)*

::: {.callout-tip collapse="true" title="Solution"}
1. **With subsampling.** By @eq-pate-icl the curve is 1,000 times the subsampled-Gaussian curve with rate 0.04 and noise multiplier $7.39/\sqrt2$, which converts to $\varepsilon = 1.00$.
2. **Without.** Every record is in every query's exemplars, each answer is $(\alpha, \alpha/7.39^2)$-RDP, and 1,000 of them convert to $\varepsilon = 45.8$.
3. **The noise without subsampling.** By @eq-pate-sigma, $\sigma = \sqrt{1{,}000/0.0306} = 180.9$, about 24 times more.
4. **Participation.** A record is drawn with probability 0.04 per query, so it takes part in $1{,}000 \times 0.04 = 40$ queries on average.

*Subsampling helps for two reasons: each record takes part in only 40 of the 1,000 answers, and when it does not, its absence is hidden among the random draws. But the noise must still be compared with the vote gaps of an ensemble of only 25 voters, which is why the stand-in's accuracy falls anyway. With a large language model, whose votes agree far more, the same noise costs much less accuracy.*
:::

## Distance to instability and the threshold {#sec-sol-pate-8}

> **Exercise 8.** **Distance to instability and the threshold.** Compute the distance to instability of @eq-pate-dist for the two-label votes 0–0, 5–4, 6–4, 7–4, 55–45 and 135–115, and say for each why one more changed teacher would or would not be safe. Then find the threshold $\tau$ of @eq-pate-stable-threshold at $\varepsilon_0 = 1$ and $\delta_{\text{bad}} = 5 \times 10^{-6}$ for pools of 1,000 and 10,000 queries, and the smallest gap whose distance reaches it. Why does a tenfold larger pool raise $\tau$ by about $4\ln 10$? *Check: distances 0, 0, 0, 1, 4, 9; $\tau = 74.8$ and 84.0; gaps 151 and 171; $4\ln 10 = 9.2$.* *(Distances to a set; union bounds)*

::: {.callout-tip collapse="true" title="Solution"}
1. **The distances.** The gaps are 0, 1, 2, 3, 10 and 20, so $\lceil g/2\rceil - 1$ gives 0, 0, 0, 1, 4 and 9 (0 for a tie). With 5–4, one teacher moving from the leader to the other label makes it 4–5, and the leader changes. With 6–4, one move makes it 5–5, a tie that the argmax may break either way, so the distance is still 0. With 7–4, one move gives 6–5 and the leader stays strictly on top, but a second move gives 5–6. In general each move costs the leader 2 votes of gap.
2. **The thresholds.** Bisection on $Q_{\mathrm{pool}}\cdot\Pr[\nu - \xi > \tau] \le 5 \times 10^{-6}$, with $\nu \sim \mathrm{Lap}(4)$ and $\xi \sim \mathrm{Lap}(2)$, gives $\tau = 74.8$ for $Q_{\mathrm{pool}} = 1{,}000$ and $\tau = 84.0$ for $Q_{\mathrm{pool}} = 10{,}000$.
3. **The gaps.** The distance must be at least $\tau$, so at least $\lceil \tau\rceil$, which needs $\lceil g/2\rceil \ge \lceil \tau\rceil + 1$, that is $g \ge 2\lceil \tau\rceil + 1$: gaps of 151 and 171.
4. **The pool size.** For large $\tau$ the tail is dominated by its slower term, about $\frac23e^{-\tau\varepsilon_0/4}$. Multiplying the pool by 10 requires the tail to fall by 10, so $\tau$ grows by $4\ln 10/\varepsilon_0 = 9.2$, which matches $84.0 - 74.8$.

*The threshold protects every query of the pool at once, so it grows with the logarithm of the pool, and it is measured in teachers. An ensemble whose typical gap is smaller than about $2\tau$ cannot use stable aggregation, however good its teachers are.*
:::

## What stable aggregation needs {#sec-sol-pate-9}

> **Exercise 9.** **What stable aggregation needs.** You want 1,000 exact labels from stable aggregation with a cutoff of $c = 20$ unstable answers, a pool of 10,000 queries and $\delta_{\text{bad}} = \delta' = 5 \times 10^{-6}$. Compute the budget and the smallest passing gap at $\varepsilon_0 = 0.5$ and $\varepsilon_0 = 1$. Suppose a share $f$ of the queries is unstable, in the sense that the test says so. How small must $f$ be for 20 unstable answers to leave 1,000 labelled queries on average? What does that say about the number of teachers? *Check: $\varepsilon = 9.93$ with gap 339, and 20.00 with gap 171; $f \le 0.0196$.* *(Sparse vector; planning a budget)*

::: {.callout-tip collapse="true" title="Solution"}
1. **Budgets.** Twenty runs of $\varepsilon_0$-DP AboveThreshold compose by @eq-pate-stable-eps to $\varepsilon = 9.93$ at $\varepsilon_0 = 0.5$ and 20.00 at $\varepsilon_0 = 1$. At this scale the composition is almost linear, about $20\varepsilon_0$.
2. **Gaps.** The thresholds are $\tau = 168.1$ and 84.0, so by Exercise 8 the plurality must lead by at least 339 and 171 votes.
3. **The share of unstable queries.** Before each unstable answer the procedure labels on average $(1 - f)/f$ stable queries, so $c$ unstable answers leave about $c(1 - f)/f$ labels. Setting this to at least 1,000 gives $f \le c/(c + 1{,}000) = 20/1{,}020 = 0.0196$.
4. **Teachers.** Fewer than 2% of the queries may fall short of a lead of 171 votes at $\varepsilon = 20$, or of 339 votes at $\varepsilon = 9.93$. Since the gap cannot exceed the number of teachers, that takes several hundred teachers, nearly unanimous on 98% of the queries. The running example's 500 teachers clear a distance of 84 on only 0.3923 of the pool.

*Stable aggregation buys a budget fixed in advance and exact labels, and it pays with a demand on consensus that few tasks meet. Measure the share of queries that clear the threshold on public or held-out data before choosing it.*
:::
