# Exercises: Local, shuffle and distributed trust models

Exercises for [Local, shuffle and distributed trust models](../../book-full/part03-dp-foundations/08-trust-models.md), each with a worked solution. Try each exercise before opening its solution. The script `code/part03-dp-foundations/solutions_trust_models.py` computes every number quoted here; run it with `uv run python code/part03-dp-foundations/solutions_trust_models.py`.

The solutions use the chapter's notation: for differential privacy (DP), $\varepsilon_0$ is the local level of one report and $(\varepsilon, \delta)$ the central guarantee, neighbours differ by replacing one record, and logarithms are natural.

## Warner's forced-response variant {#sec-sol-trust-1}

> **Exercise 1.** **Warner's forced-response variant.** Each person answers truthfully with probability $s$ and otherwise reports a fair coin. Show that this is randomized response with keep probability $t = (1 + s)/2$, and find its $\varepsilon_0$. *Check: $s = \frac12$ gives $\varepsilon_0 = \ln 3 = 1.0986$.* *(Local randomizer)*

::: {.callout-tip collapse="true" title="Solution"}
1. **The report equals the true bit in two ways.** Either the person answers truthfully (probability $s$), or they toss the coin and it happens to show their bit (probability $(1 - s)\cdot\frac12$). So $\Pr[\text{report} = x] = s + \frac{1-s}{2} = \frac{1+s}{2} = t$, for both values of $x$.
2. **The report is the flipped bit otherwise,** with probability $1 - t = \frac{1-s}{2}$. A mechanism on one bit is determined by these two numbers, so this is randomized response with keep probability $t$.
3. **Its level.** As in the table of output probabilities for randomized response, the largest ratio between the two rows is $t/(1 - t)$, so
$$\varepsilon_0 = \ln\frac{t}{1-t} = \ln\frac{1+s}{1-s}.$$
4. **Check.** $s = \frac12$ gives $t = 0.75$ and $\varepsilon_0 = \ln 3 = 1.0986$, Warner's two coins. $s = 0.25$ gives $t = 0.625$ and $\varepsilon_0 = 0.5108$; $s = 0.1$ gives $t = 0.55$ and $\varepsilon_0 = 0.2007$.

*For small $s$, $\varepsilon_0 = \ln\frac{1+s}{1-s} \approx 2s$: the truthful fraction is half the privacy level.*
:::

## Why $p = \frac12$ {#sec-sol-trust-2}

> **Exercise 2.** **Why $p = \frac12$.** Among unary encodings with $\frac{p(1-q)}{q(1-p)} = e^{\varepsilon_0}$, minimize the rare-value variance $q(1-q)/(p-q)^2$ over $p$. Show the minimum is at $p = \frac12$. *Check: at $\varepsilon_0 = 1$ the minimum is 3.683, and the symmetric choice gives 3.918.* *(Debiasing, optimization)*

::: {.callout-tip collapse="true" title="Solution"}
Write $a = e^{\varepsilon_0}$, and work with odds: $P = p/(1-p)$ and $Q = q/(1-q)$. The constraint says exactly $P = aQ$.

1. **Rewrite both pieces in odds.** $p = P/(1+P)$ and $q = Q/(1+Q)$, so
$$p - q = \frac{P - Q}{(1+P)(1+Q)}, \qquad q(1-q) = \frac{Q}{(1+Q)^2}.$$
2. **The variance in one variable.** Dividing, the factors $(1+Q)^2$ cancel:
$$\frac{q(1-q)}{(p-q)^2} = \frac{Q(1+P)^2}{(P-Q)^2} = \frac{Q(1 + aQ)^2}{(a-1)^2Q^2} = \frac{(1 + aQ)^2}{(a-1)^2\,Q}.$$
3. **Minimize over $Q > 0$.** The derivative of $(1 + aQ)^2/Q$ is $\big(2aQ(1 + aQ) - (1 + aQ)^2\big)/Q^2 = (1 + aQ)(aQ - 1)/Q^2$. It is negative for $aQ < 1$ and positive for $aQ > 1$, so the minimum is at $aQ = 1$, that is $P = 1$, that is $p = \frac12$. Then $q = Q/(1+Q) = 1/(a + 1)$: optimized unary encoding.
4. **The minimum value.** At $aQ = 1$, $(1 + 1)^2/((a-1)^2/a) = 4a/(a-1)^2$, the formula of the chapter.
5. **Check at $\varepsilon_0 = 1$.** $4e/(e-1)^2 = 3.6827$. A grid search over $p \in (0.25, 0.75]$ finds the minimum at $p = 0.5000$ with the same value. @tbl-sol-trust-unary shows the curve around it; it is symmetric in $p \leftrightarrow 1 - p$ because $(1 + aQ)^2/Q$ takes the same value at $Q$ and at $1/(a^2Q)$. The symmetric choice $p = e^{1/2}/(e^{1/2} + 1) = 0.6225$ gives 3.9177.

| $p$ | 0.3 | 0.4 | 0.5 | 0.6 | 0.7 |
| --- | --- | --- | --- | --- | --- |
| $q$ | 0.1362 | 0.1970 | 0.2689 | 0.3556 | 0.4619 |
| $n\cdot\operatorname{Var}$ | 4.3842 | 3.8361 | 3.6827 | 3.8361 | 4.3842 |

: Unary encodings at $\varepsilon_0 = 1$: the probability $q$ forced by the privacy constraint and the rare-value variance, for several keep probabilities $p$. {#tbl-sol-trust-unary}

*Since almost all bits are 0, what matters is how rarely a 0 becomes a 1; the optimum gives up accuracy on the single 1 to keep $q$ small.*
:::

## How many people? {#sec-sol-trust-3}

> **Exercise 3.** **How many people?** At $\varepsilon = 1$, how many people does randomized response need to estimate a fraction with standard deviation 0.001, and how many does the central Laplace mechanism need? *Check: 920,674 local and 1,415 central.* *(Variance formulas)*

::: {.callout-tip collapse="true" title="Solution"}
1. **Local.** The standard deviation is $\sqrt{e^\varepsilon/(n(e^\varepsilon - 1)^2)}$. Setting it to at most $0.001$ and solving for $n$: $n \ge \frac{e^\varepsilon}{(e^\varepsilon - 1)^2}\cdot10^6$. At $\varepsilon = 1$ the factor is $e/(e-1)^2 = 0.9207$, so $n \ge 920{,}674$ (rounding up).
2. **Central.** Under replace, the count of ones has sensitivity 1, and $\mathrm{Lap}(1/\varepsilon)$ on the count gives a fraction with standard deviation $\sqrt2/(n\varepsilon)$. Setting it to at most $0.001$: $n \ge \sqrt2\times1000 = 1414.2$, so $n \ge 1{,}415$.
3. **Compare.** The ratio is about 650. Halving the target standard deviation multiplies the local requirement by 4 but the central one only by 2: the $\sqrt n$ gap seen from the side of sample sizes.

*Local DP turns an error target $\sigma$ into a sample size of order $1/(\varepsilon^2\sigma^2)$; central DP into one of order $1/(\varepsilon\sigma)$.*
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
3. **The honest part is a shuffle of $n - m$ reports.** Its neighbouring datasets replace one honest record, and Theorem 3.1 applies with $n - m$ people, under its condition with $n - m$.
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
1. **Planned noise.** The exact Gaussian calibration at $(1, 10^{-6})$ and $\Delta_2 = \sqrt2$ gives total $\sigma = 5.975$, so each of the 100 clients adds $\mathcal N(0, \sigma^2/100)$, standard deviation 0.597.
2. **Thirty clients drop out.** The sum now has 70 shares, total variance $0.7\sigma^2$, and the Gaussian mechanism with noise $\sigma\sqrt{0.7}$ is $(1.212, 10^{-6})$-DP: about 21% more $\varepsilon$ than promised. The guarantee falls to $\varepsilon = 2$ only when the honest fraction drops to 0.279.
3. **Over-provision.** To keep total variance $\sigma^2$ with only 70 shares, each client adds $\sigma^2/70$: standard deviation $5.975/\sqrt{70} = 0.714$.
4. **The cost.** If nobody drops out, the sum carries $100/70 = 1.429$ times the variance needed, so the error is $\sqrt{1.429} = 1.20$ times the central one.

*Over-provisioning trades a fixed accuracy loss for a guarantee that survives the worst case you planned for.*
:::
