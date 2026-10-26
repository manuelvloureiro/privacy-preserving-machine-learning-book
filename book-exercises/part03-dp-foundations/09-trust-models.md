# Exercises: Local, shuffle and distributed trust models

Exercises for [Local, shuffle and distributed trust models](../../book-full/part03-dp-foundations/09-trust-models.md), each with a worked solution. Try each exercise before opening its solution. The script `code/part03-dp-foundations/solutions_trust_models.py` computes every number quoted here; run it with `uv run python code/part03-dp-foundations/solutions_trust_models.py`.

The solutions use the chapter's notation: for differential privacy (DP), $\varepsilon_0$ is the local level of one report and $(\varepsilon, \delta)$ the central guarantee, neighbours differ by replacing one record, and logarithms are natural.

## The forced-response variant {#sec-sol-trust-1}

> **Exercise 1.** **The forced-response variant.** Each person answers truthfully with probability $s$ and otherwise reports a fair coin. Show that this is randomized response with truth probability $t = (1 + s)/2$, and find its $\varepsilon_0$. *Check: $s = \frac12$ gives $\varepsilon_0 = \ln 3 = 1.0986$.* *(Local randomizer)*

::: {.callout-tip collapse="true" title="Solution"}
1. **The report equals the true bit in two ways.** Either the person answers truthfully (probability $s$), or they toss the coin and it happens to show their bit (probability $(1 - s)\cdot\frac12$). So $\Pr[\text{report} = x] = s + \frac{1-s}{2} = \frac{1+s}{2} = t$, for both values of $x$.
2. **The report is the flipped bit otherwise,** with probability $1 - t = \frac{1-s}{2}$. A mechanism on one bit is determined by these two numbers, so this is randomized response with truth probability $t$.
3. **Its level.** As in the table of output probabilities for randomized response, the largest ratio between the two rows is $t/(1 - t)$, so
$$\varepsilon_0 = \ln\frac{t}{1-t} = \ln\frac{1+s}{1-s}.$$
4. **Check.** $s = \frac12$ gives $t = 0.75$ and $\varepsilon_0 = \ln 3 = 1.0986$, the two coins of the forced-response design (Boruch, 1971), the form Dwork & Roth present. $s = 0.25$ gives $t = 0.625$ and $\varepsilon_0 = 0.5108$; $s = 0.1$ gives $t = 0.55$ and $\varepsilon_0 = 0.2007$.

*For small $s$, $\varepsilon_0 = \ln\frac{1+s}{1-s} \approx 2s$: the truthful fraction is half the privacy level.*
:::

## Why $p = \frac12$ {#sec-sol-trust-2}

> **Exercise 2.** **Why $p = \frac12$.** Among unary encodings with $\frac{p(1-q)}{q(1-p)} = e^{\varepsilon_0}$, minimize the rare-value variance $q(1-q)/(p-q)^2$ over $p$. Show the minimum is at $p = \frac12$. *Check: at $\varepsilon_0 = 1$ the minimum is 3.683, and the symmetric choice gives 3.918.* *(Debiasing, optimization)*

::: {.callout-tip collapse="true" title="Solution"}
Write $a = e^{\varepsilon_0}$, and work with odds: $u = p/(1-p)$ and $v = q/(1-q)$. The constraint says exactly $u = av$.

1. **Rewrite both pieces in odds.** $p = u/(1+u)$ and $q = v/(1+v)$, so
$$p - q = \frac{u - v}{(1+u)(1+v)}, \qquad q(1-q) = \frac{v}{(1+v)^2}.$$
2. **The variance in one variable.** Dividing, the factors $(1+v)^2$ cancel:
$$\frac{q(1-q)}{(p-q)^2} = \frac{v(1+u)^2}{(u-v)^2} = \frac{v(1 + av)^2}{(a-1)^2v^2} = \frac{(1 + av)^2}{(a-1)^2\,v}.$$
3. **Minimize over $v > 0$.** The derivative of $(1 + av)^2/v$ is $\big(2av(1 + av) - (1 + av)^2\big)/v^2 = (1 + av)(av - 1)/v^2$. It is negative for $av < 1$ and positive for $av > 1$, so the minimum is at $av = 1$, that is $u = 1$, that is $p = \frac12$. Then $q = v/(1+v) = 1/(a + 1)$: optimized unary encoding.
4. **The minimum value.** At $av = 1$, $(1 + 1)^2/((a-1)^2/a) = 4a/(a-1)^2$, the optimized rare-value variance of @prp-trust-schemes.
5. **Check at $\varepsilon_0 = 1$.** $4e/(e-1)^2 = 3.6827$. A grid search over $p \in (0.25, 0.75]$ finds the minimum at $p = 0.5000$ with the same value. @tbl-sol-trust-unary shows the curve around it; it is symmetric in $p \leftrightarrow 1 - p$ because $(1 + av)^2/v$ takes the same value at $v$ and at $1/(a^2v)$. The symmetric choice $p = e^{1/2}/(e^{1/2} + 1) = 0.6225$ gives 3.9177.

| $p$ | 0.3 | 0.4 | 0.5 | 0.6 | 0.7 |
| --- | --- | --- | --- | --- | --- |
| $q$ | 0.1362 | 0.1970 | 0.2689 | 0.3556 | 0.4619 |
| $n\cdot\operatorname{Var}$ | 4.3842 | 3.8361 | 3.6827 | 3.8361 | 4.3842 |

: Unary encodings at $\varepsilon_0 = 1$: the probability $q$ forced by the privacy constraint and the rare-value variance, for several truth probabilities $p$. {#tbl-sol-trust-unary}

*Since almost all bits are 0, what matters is how rarely a 0 becomes a 1; the optimum gives up accuracy on the single 1 to keep $q$ small.*
:::

## How many people? {#sec-sol-trust-3}

> **Exercise 3.** **How many people?** At $\varepsilon = 1$, how many people does randomized response need to estimate a fraction with standard deviation 0.001, and how many does the central Laplace mechanism need? *Check: 920,674 local and 1,415 central.* *(Variance formulas)*

::: {.callout-tip collapse="true" title="Solution"}
1. **Local.** By @prp-trust-rr the standard deviation is $\sqrt{e^\varepsilon/(n(e^\varepsilon - 1)^2)}$. Setting it to at most $0.001$ and solving for $n$: $n \ge \frac{e^\varepsilon}{(e^\varepsilon - 1)^2}\cdot10^6$. At $\varepsilon = 1$ the factor is $e/(e-1)^2 = 0.9207$, so $n \ge 920{,}674$ (rounding up).
2. **Central.** Under replace, the count of ones has sensitivity 1, and $\mathrm{Lap}(1/\varepsilon)$ on the count gives a fraction with standard deviation $\sqrt2/(n\varepsilon)$. Setting it to at most $0.001$: $n \ge \sqrt2\times1000 = 1414.2$, so $n \ge 1{,}415$.
3. **Compare.** The ratio is about 650. Halving the target standard deviation multiplies the local requirement by 4 but the central one only by 2: the $\sqrt n$ gap seen from the side of sample sizes.

*Local DP turns an error target $a$ into a sample size of order $1/(\varepsilon^2a^2)$; central DP into one of order $1/(\varepsilon a)$.*
:::

## Inverting the shuffle bound {#sec-sol-trust-4}

> **Exercise 4.** **Inverting the shuffle bound.** For $n = 10^6$ and $\delta = 10^{-8}$, find the largest $\varepsilon_0$ for which @eq-trust-fmt gives $\varepsilon \le 0.5$, and check the theorem's condition. *Check: $\varepsilon_0 = 5.809$; the condition allows up to 8.093.* *(Bisection)*

::: {.callout-tip collapse="true" title="Solution"}
1. **The bound increases with $\varepsilon_0$.** Both $\frac{e^{\varepsilon_0} - 1}{e^{\varepsilon_0} + 1}$ and the bracket increase with $\varepsilon_0$, and $\ln(1 + u)$ is increasing. So the set of allowed $\varepsilon_0$ is an interval $[0, \varepsilon_0^*]$, and $\varepsilon_0^*$ solves "bound $= 0.5$".
2. **The condition.** $\ln\big(n/(16\ln(2/\delta))\big) = \ln\big(10^6/(16\times19.11)\big) = 8.093$. The theorem applies only up to there.
3. **Bisect** on $[0, 8.093]$: at the upper end the bound exceeds 0.5, at 0 it is 0, so a root lies in between. Bisection gives $\varepsilon_0^* = 5.809$, where the bound equals 0.5000. This is inside the condition.
4. **Check the pieces.** At $\varepsilon_0 = 5.809$: $e^{\varepsilon_0} = 333.3$, $\frac{e^{\varepsilon_0} - 1}{e^{\varepsilon_0} + 1} = \tanh(\varepsilon_0/2) = 0.9940$, $\ln(4/\delta) = 19.81$, the square-root term $8\sqrt{333.3\times19.81}/\sqrt{10^6} = 0.6500$ and the $1/n$ term $8\times333.3/10^6 = 0.0027$. Their sum times $0.9940$ is $u = 0.6487$, and $\ln(1 + u) = 0.5000$. Using $u$ itself, the small-$\varepsilon$ approximation, would overstate $\varepsilon$ by 30% at this size.

*A million users let each report be almost unrandomized ($e^{\varepsilon_0} \approx 333$) while the shuffled collection is $(0.5, 10^{-8})$-DP.*
:::

## Sybils in the shuffle {#sec-sol-trust-5}

> **Exercise 5.** **Sybils in the shuffle.** An adversary controls $m$ of the $n$ senders. Show that the honest people's guarantee is @eq-trust-fmt with $n - m$ in place of $n$. With $n = 10^4$, $\varepsilon_0 = 2$ and $\delta = 10^{-6}$, compare $m = 0$ and $m = 5{,}000$. *Check: 0.5009 and 0.6535.* *(Post-processing)*

::: {.callout-tip collapse="true" title="Solution"}
1. **Split the multiset.** The shuffled output is the multiset union of the $n - m$ honest reports and the $m$ adversarial ones. The adversary chose its own reports, so it knows them.
2. **The full output is a post-processing of the honest part.** Given the honest multiset $H$, the adversary's view is $H \cup A$, where $A$ is a fixed (or independently random) multiset that does not depend on the honest records. A function of $H$ and independent randomness is post-processing ([Post-processing, group privacy and composition](../../book-full/part03-dp-foundations/02-properties.md)).
3. **The honest part is a shuffle of $n - m$ reports.** Its neighbouring datasets replace one honest record, and @thm-trust-fmt applies with $n - m$ people, under its condition with $n - m$.
4. **So the guarantee for an honest person** is @eq-trust-fmt evaluated at $n - m$. It cannot be better: the adversary can always remove its own reports, so they hide nobody.
5. **Numbers.** With $m = 0$: $n = 10{,}000$, $\varepsilon = 0.5009$ (condition: $\varepsilon_0 \le 3.763$). With $m = 5{,}000$: $5{,}000$ honest people, $\varepsilon = 0.6535$ (condition: $\varepsilon_0 \le 3.070$). With $m = 9{,}000$ only 1,000 honest people remain, the condition allows only $\varepsilon_0 \le 1.460$, and the theorem says nothing at $\varepsilon_0 = 2$.

*Halving the honest crowd multiplies $\varepsilon$ by about $\sqrt2$ (here 1.30, slightly less because of the logarithm), and too few honest users void the guarantee entirely.*
:::

## Splitting Laplace noise {#sec-sol-trust-6}

> **Exercise 6.** **Splitting Laplace noise.** Show that if $G_i, G'_i$ are independent $\mathrm{Gamma}(1/n, b)$ (shape $1/n$, scale $b$), then $\sum_{i=1}^n(G_i - G'_i) \sim \mathrm{Lap}(b)$. *Hint: compare moment generating functions; $\mathrm{Lap}(b)$ has $1/(1 - \lambda^2b^2)$.* Check by simulation for $n = 100$, $b = 2$. *Check: variance $2b^2 = 8$.* *(Infinite divisibility)*

::: {.callout-tip collapse="true" title="Solution"}
1. **Gamma moment generating function.** For $G \sim \mathrm{Gamma}(k, b)$ and $\lambda < 1/b$, $\mathbb E[e^{\lambda G}] = (1 - b\lambda)^{-k}$. For $-G'$, replace $\lambda$ by $-\lambda$: $(1 + b\lambda)^{-k}$.
2. **One client's share.** $G_i$ and $G'_i$ are independent, so for $|\lambda| < 1/b$,
$$\mathbb E[e^{\lambda(G_i - G'_i)}] = (1 - b\lambda)^{-1/n}(1 + b\lambda)^{-1/n} = (1 - b^2\lambda^2)^{-1/n}.$$
3. **The sum.** The $n$ shares are independent, so the moment generating functions multiply: $\big((1 - b^2\lambda^2)^{-1/n}\big)^n = 1/(1 - b^2\lambda^2)$, the Laplace one.
4. **Same moment generating function on an interval around 0, same distribution.** This is the uniqueness theorem for moment generating functions (for example Billingsley, *Probability and Measure*, section 30).
5. **Simulation.** With $n = 100$ and $b = 2$, 200,000 sums give mean $-0.0077$, variance 8.025 (exact $2b^2 = 8$), and $\Pr[|X| \ge 4] = 0.1361$ against the exact $e^{-2} = 0.1353$.

*Laplace noise can be distributed exactly, so a distributed protocol can give pure $\varepsilon$-DP; but Gamma shares are real-valued, and secure aggregation over the integers needs a discrete analogue (differences of Pólya variables split the discrete Laplace the same way).*
:::

## Planning for dropouts {#sec-sol-trust-7}

> **Exercise 7.** **Planning for dropouts.** 100 clients calibrate distributed Gaussian noise for $(1, 10^{-6})$ with $\Delta_2 = \sqrt2$. If 30% drop out after the noise was planned, what $\varepsilon$ results? How much noise per client keeps $\varepsilon = 1$ if up to 30% may drop out? *Check: $\varepsilon = 1.212$; per-client $\sigma = 0.714$ instead of 0.597.* *(Gaussian calibration)*

::: {.callout-tip collapse="true" title="Solution"}
1. **Planned noise.** As in @exm-trust-clients, the exact Gaussian calibration at $(1, 10^{-6})$ and $\Delta_2 = \sqrt2$ gives total $\sigma = 5.975$, so each of the 100 clients adds $\mathcal N(0, \sigma^2/100)$, standard deviation 0.597.
2. **Thirty clients drop out.** The sum now has 70 shares, total variance $0.7\sigma^2$, and by @prp-trust-distributed the Gaussian mechanism with noise $\sigma\sqrt{0.7}$ is $(1.212, 10^{-6})$-DP: about 21% more $\varepsilon$ than promised. The guarantee falls to $\varepsilon = 2$ only when the honest fraction drops to 0.279.
3. **Over-provision.** To keep total variance $\sigma^2$ with only 70 shares, each client adds $\sigma^2/70$: standard deviation $5.975/\sqrt{70} = 0.714$.
4. **The cost.** If nobody drops out, the sum carries $100/70 = 1.429$ times the variance needed, so the error is $\sqrt{1.429} = 1.20$ times the central one.

*Over-provisioning trades a fixed accuracy loss for a guarantee that survives the worst case you planned for.*
:::

## Averaging fresh reports {#sec-sol-trust-8}

> **Exercise 8.** **Averaging fresh reports.** A person reports the same bit every day by randomized response at $\varepsilon_0 = \ln 3$, with fresh randomness each day. An observer starts with even odds. What is the observer's probability that the bit is 1 after 7 ones in 10 days, and after 5 ones in 10 days? After how many days is the majority of the reports wrong with probability below 1%? What changes if the report is memoized? *Check: 0.9878 and 0.5000; 19 days; memoized, the majority stays wrong with probability 0.25.* *(likelihood ratios, binomial tails)*

::: {.callout-tip collapse="true" title="Solution"}
1. **The likelihood ratio.** By @prp-trust-fresh, a sequence with $k$ ones in $T$ days has likelihood ratio $e^{\varepsilon_0(2k - T)} = 3^{2k - T}$ between bit 1 and bit 0.
2. **The posterior.** Bayes' rule multiplies the prior odds by the likelihood ratio. With even prior odds and $k = 7$, $T = 10$, the odds are $3^4 = 81$, so the probability is $81/82 = 0.9878$. With $k = 5$ the ratio is $3^0 = 1$, and the probability stays at 0.5000: a balanced sequence says nothing.
3. **The majority.** The majority is wrong when more than half of the $T$ reports are flipped, and a tie is broken by a fair coin. The number of flipped reports is binomial with $T$ trials and probability $\frac14$. Computing the tail day by day, it first falls below 0.1 after 7 days, where it is 0.07056. It falls below 0.01 after 19 days, at 0.008903, and below 0.001 after 33 days, at 0.000951.
4. **Memoized.** With a memoized report, every day repeats the first report. The majority is then just that report, which is wrong with probability $1 - t = 0.25$ however many days pass.

*Fresh noise on a fixed value is averaged away at an exponential rate, which is why basic composition is tight here and why deployments memoize.*
:::

## RAPPOR's one-report level {#sec-sol-trust-9}

> **Exercise 9.** **RAPPOR's one-report level.** RAPPOR passes each bit $x$ of a Bloom filter, set by a number $H$ of hash functions, through two layers. The permanent layer keeps $x$ with probability $1 - \phi$, for a parameter $\phi$, and otherwise replaces it by a fair coin. The instantaneous layer then sends 1 with probability $q$ if the permanent bit is 1, and with probability $p$ if it is 0. Show that a sent bit is 1 with probability $p^* = \frac\phi2q + (1 - \frac\phi2)p$ when $x = 0$ and $q^* = (1 - \frac\phi2)q + \frac\phi2p$ when $x = 1$, and that one report is $\varepsilon_1$-DP with
> $$\varepsilon_1 = H\ln\frac{q^*(1 - p^*)}{p^*(1 - q^*)}.$$
> Evaluate it for the homepage collection, with $H = 2$, $\phi = 0.75$, $p = 0.5$ and $q = 0.75$. Then evaluate it for the process-name collection, which differs only in $\phi = 0.5$, and compare both with $\varepsilon_\infty$. *Check: 0.5343 and 1.0743, against $\varepsilon_\infty$ of 2.0433 and 4.3944.* *(composing two randomizers in series)*

::: {.callout-tip collapse="true" title="Solution"}
1. **Through both layers.** If $x = 0$, the permanent bit is 1 with probability $\frac\phi2$ (the coin came up 1) and 0 otherwise. The sent bit is then 1 with probability $q$ or $p$, so by the law of total probability it is 1 with probability $\frac\phi2q + (1 - \frac\phi2)p = p^*$. If $x = 1$, the permanent bit is 1 with probability $1 - \frac\phi2$, which gives $q^*$ in the same way.
2. **One bit's ratios.** Since $q > p$, $q^* > p^*$. A sent 1 has ratio $q^*/p^*$ between $x = 1$ and $x = 0$, and a sent 0 has ratio $(1 - p^*)/(1 - q^*)$ the other way round.
3. **The whole report.** Two values set at most $H$ bits each, so in the worst case $H$ bits go from 1 to 0 and $H$ from 0 to 1, and all other bits cancel. As in unary encoding, the worst outcome sends 1 on the bits that only the first value sets and 0 on the bits that only the second value sets, which multiplies $H$ factors $q^*/p^*$ and $H$ factors $(1 - p^*)/(1 - q^*)$. Its logarithm is $\varepsilon_1$.
4. **The homepage numbers.** With $\phi = 0.75$, $p^* = 0.375 \times 0.75 + 0.625 \times 0.5 = 0.59375$ and $q^* = 0.65625$. Then $\varepsilon_1 = 2\ln\frac{0.65625 \times 0.40625}{0.59375 \times 0.34375} = 0.5343$, the value the RAPPOR paper reports. With $\phi = 0.5$, $p^* = 0.5625$, $q^* = 0.6875$ and $\varepsilon_1 = 1.0743$, again the paper's value.
5. **Against $\varepsilon_\infty$.** @eq-trust-rappor gives $\varepsilon_\infty = 2.0433$ and $4.3944$. One report is about four times better protected than all reports of the value together, because the instantaneous layer adds noise that averaging over many reports removes, until only the permanent layer is left.

*Two randomizers in series are one randomizer whose probabilities come from the law of total probability, so its level is computed once, from the composed probabilities.*
:::

## How many buckets? {#sec-sol-trust-10}

> **Exercise 10.** **How many buckets?** Local hashing has the rare-value variance of @eq-trust-lh. Find the best integer number of buckets at $\varepsilon_0 = 1$ and $\varepsilon_0 = 2$, compare its variance with optimized unary encoding and with $k$-ary randomized response over $k = 1{,}000$ values, and say how many bits the bucket takes. *Check: $w = 4$ gives 3.6917 against 3.6827; $w = 8$ gives 0.7246 against 0.7241; $k$-ary randomized response gives 338.9 and 24.6.* *(optimizing a variance over an integer)*

::: {.callout-tip collapse="true" title="Solution"}
1. **The continuous optimum.** Writing $u = w - 1$, the variance is proportional to $e^{2\varepsilon_0}/u + 2e^{\varepsilon_0} + u$. This is convex in $u$, and by @prp-trust-lh it is smallest at $u = e^{\varepsilon_0}$, so the best real $w$ is $e^{\varepsilon_0} + 1$.
2. **Round to an integer.** A convex function is smallest at one of the two integers around its minimum. At $\varepsilon_0 = 1$, $e + 1 = 3.718$, and the variance is 3.7701 at $w = 3$, 3.6917 at $w = 4$ and 3.8218 at $w = 5$. So $w = 4$, two bits. At $\varepsilon_0 = 2$, $e^2 + 1 = 8.389$, and the variance is 0.7246 at $w = 8$ against 0.7252 at $w = 9$. So $w = 8$, three bits.
3. **Against the other oracles.** Optimized unary encoding gives 3.6827 and 0.7241, a fraction of a percent better, but with a report of one bit per value. By contrast, $k$-ary randomized response over 1,000 values gives 338.9 and 24.6, because it spreads its signal over the whole domain.

*Local hashing loses almost nothing to the best oracle, as long as the number of buckets follows $\varepsilon_0$ rather than the size of the domain.*
:::

## A year of $\alpha$-point rounding {#sec-sol-trust-11}

> **Exercise 11.** **A year of $\alpha$-point rounding.** Under $\alpha$-point rounding with $m = 1{,}440$ minutes and $\varepsilon_0 = 1$, each person's daily minutes wobble uniformly by up to $\pm s$ around their own level, independently each day. What fraction of people change their report at least once in 365 days, for $s = 15$ and $s = 60$? With output perturbation at a flip probability $\kappa$ of 0.1, 0.2 or 0.3, what does one round cost, and what does each day on which the kept bit changes cost? *Check: 0.0126 and 0.0503; one round 0.776, 0.569 and 0.374; per change 2.197, 1.386 and 0.847.* *(order statistics, the formula for output perturbation)*

::: {.callout-tip collapse="true" title="Solution"}
1. **When a report can change.** By item 3 of @prp-trust-alpha, a person's report changes only if the two kept bits differ and the threshold $m - \alpha$ falls between the smallest and largest value they take. The bits differ with probability $1 - 2e/(e + 1)^2 = 0.6068$. The threshold is uniform on $(0, m]$ and independent of the values, so given the band it falls inside with probability equal to the band's width divided by $m$.
2. **The expected width.** For $T$ independent uniforms on $[-s, s]$, the expected largest minus the expected smallest is $2s(T - 1)/(T + 1)$, because the order statistics of uniforms split the interval into $T + 1$ gaps of equal expected size. With $T = 365$ this is $2s \times 364/366$.
3. **The fractions.** For $s = 15$, $0.6068 \times 30 \times 364/366 / 1{,}440 = 0.0126$. For $s = 60$, four times the width gives 0.0503. Over a year, about one person in 80 with a steady habit ever changes their report.
4. **Output perturbation.** @eq-trust-op at $\varepsilon_0 = 1$ gives 0.776, 0.569 and 0.374 for $\kappa = 0.1$, 0.2 and 0.3. Each day on which the kept bit changes contributes at most $\ln((1 - \kappa)/\kappa)$, which is 2.197, 1.386 and 0.847.

*A larger $\kappa$ protects both a single round and the days of change, at the price of a noisier bit and so a less accurate mean.*
:::
