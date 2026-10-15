# Exercises: Beyond global sensitivity

Exercises for [Beyond global sensitivity](../../book-full/part03-dp-foundations/06-sensitivity.md), each with a worked solution. Try each exercise before opening its solution. The script `code/part03-dp-foundations/solutions_sensitivity.py` computes every number quoted here; run it with `uv run python code/part03-dp-foundations/solutions_sensitivity.py`.

The solutions use the chapter's notation. Datasets have a fixed size $n$, neighbours differ in one replaced record, $d$ is the Hamming distance, and $\varepsilon$, $\delta$ are the parameters of differential privacy (DP). Sorted values are $x_1 \le \dots \le x_n$ in $[0, \Lambda]$, padded with $x_i = 0$ for $i \le 0$ and $x_i = \Lambda$ for $i > n$. $A^{(k)}(D)$ is the largest local sensitivity within distance $k$ of $D$, and $S^*_\beta(D) = \max_k e^{-k\beta}A^{(k)}(D)$. "The ages" are the chapter's 1,001 synthetic ages (seed 0), with $\Lambda = 100$ and $\varepsilon = 1$.

## Local sensitivity of the mean {#sec-sol-sensitivity-1}

> **Exercise 1.** **Local sensitivity of the mean.** For values in $[0, \Lambda]$ under the replace relation, show $\mathrm{LS}_{\mathrm{mean}}(D) = \max_i\max(x_i, \Lambda - x_i)/n$, and deduce $\Lambda/(2n) \le \mathrm{LS}_{\mathrm{mean}}(D) \le \Lambda/n$. Why can smooth sensitivity never gain more than a factor of 2 over global sensitivity for the mean? *Check: the 1,001 ages of the chapter include an age clipped to 0, so $\mathrm{LS} = \Delta = 0.0999$.* *(Local sensitivity)*

::: {.callout-tip collapse="true" title="Solution"}
1. **One replacement.** Replacing $x_i$ by $x'$ changes the sum by $x' - x_i$, and the mean by $(x' - x_i)/n$.
2. **Worst replacement of $x_i$.** Over $x' \in [0, \Lambda]$, $|x' - x_i|$ is largest at an endpoint: $x' = 0$ gives $x_i$, and $x' = \Lambda$ gives $\Lambda - x_i$. So the largest change is $\max(x_i, \Lambda - x_i)/n$.
3. **Worst record.** Maximize over $i$: $\mathrm{LS}_{\mathrm{mean}}(D) = \max_i\max(x_i, \Lambda - x_i)/n$.
4. **Bounds.** For any $a \in [0, \Lambda]$, $\max(a, \Lambda - a) \ge \Lambda/2$ (the two numbers add up to $\Lambda$, so the larger is at least half), and $\max(a, \Lambda - a) \le \Lambda$. So $\Lambda/(2n) \le \mathrm{LS} \le \Lambda/n = \Delta$, the global sensitivity.
5. **At most a factor of 2.** Every smooth upper bound satisfies $S^*_\beta(D) \ge \mathrm{LS}(D) \ge \Lambda/(2n) = \Delta/2$. Noise scaled by $S^*$ is therefore at least half the global noise, before even paying the $1/\alpha$ factor of the admissible noise.
6. **The ages.** The smallest age is 0 (six ages were clipped to 0), so $\max(x_i, \Lambda - x_i) = 100$ for that record and $\mathrm{LS} = 100/1001 = 0.0999 = \Delta$.

*Local sensitivity helps when the statistic is insensitive on typical data but not in the worst case. For the mean, every dataset is within a factor of 2 of the worst case, so clipping, not smoothing, is the tool.*
:::

## A sharper Cauchy {#sec-sol-sensitivity-2}

> **Exercise 2.** **A sharper Cauchy.** Show that $\sup_{u \ge 0}\gamma u^{\gamma-1}/(1 + u^\gamma) = (\gamma - 1)^{(\gamma-1)/\gamma}$, so for the Cauchy the sliding constant is 1 rather than $\gamma = 2$. Redo the pure-DP condition with this constant and find the largest $\alpha$ for $\beta = \varepsilon/6$. *Check: $\alpha = \varepsilon/2$, and the noise scale for the median age drops from 0.979 to 0.326.* *(Admissible noise)*

::: {.callout-tip collapse="true" title="Solution"}
1. **Find the maximum.** Let $g(u) = \gamma u^{\gamma-1}/(1 + u^\gamma)$. It is 0 at $u = 0$ (for $\gamma > 1$) and tends to 0 as $u \to \infty$, so its maximum is at a critical point. By the quotient rule, $g'(u) = 0$ when
$$(\gamma - 1)u^{\gamma-2}(1 + u^\gamma) = \gamma u^{\gamma-1}\cdot u^{\gamma-1}\quad\Longleftrightarrow\quad(\gamma - 1)(1 + u^\gamma) = \gamma u^\gamma\quad\Longleftrightarrow\quad u^\gamma = \gamma - 1.$$
2. **Evaluate.** There $1 + u^\gamma = \gamma$, so $g = u^{\gamma-1} = (\gamma - 1)^{(\gamma-1)/\gamma}$. A numerical maximum on a fine grid agrees (@tbl-sol-sensitivity-sliding).

| $\gamma$ | $(\gamma - 1)^{(\gamma-1)/\gamma}$ | Numerical maximum | Bound used in the chapter |
| --- | --- | --- | --- |
| 2 | 1.0000 | 1.0000 | 2 |
| 3 | 1.5874 | 1.5874 | 3 |
| 4 | 2.2795 | 2.2795 | 4 |

: The sliding constant $\sup_u \gamma u^{\gamma-1}/(1 + u^\gamma)$ for three tail exponents, against the bound $\gamma$. {#tbl-sol-sensitivity-sliding}

3. **Redo the privacy loss.** In step 4 of the chapter's derivation, the mean value theorem now gives a sliding term of at most $c_\gamma\alpha$ with $c_\gamma = (\gamma - 1)^{(\gamma-1)/\gamma}$. The scale term ($\beta$) and the dilation term ($\gamma\beta$) are unchanged. So the condition becomes $c_\gamma\alpha + (\gamma + 1)\beta \le \varepsilon$.
4. **The Cauchy.** With $\gamma = 2$, $c_2 = 1$ and $\beta = \varepsilon/6$: $\alpha \le \varepsilon - 3\varepsilon/6 = \varepsilon/2$.
5. **The median age.** $S^*_{1/6} = 0.1632$ does not change, since $\beta$ did not. The noise scale $S^*/\alpha$ goes from $6S^*/\varepsilon = 0.979$ to $2S^*/\varepsilon = 0.326$, three times smaller.

*Constants in admissibility lemmas are usually stated for convenience, not optimality. A direct bound on the privacy loss of your one noise distribution can recover a constant factor in the error.*
:::

## Other quantiles {#sec-sol-sensitivity-3}

> **Exercise 3.** **Other quantiles.** Generalize @eq-sens-median-ak to the $r$-th order statistic and compute the smooth sensitivity of the 90th percentile ($r = 901$) of the 1,001 ages with $\beta = 1/6$. Why is it larger than the median's although its local sensitivity is smaller? *Check: $\mathrm{LS} = 0.0114$, $S^* = 0.2666$ at $k = 7$.* *(Smooth sensitivity)*

::: {.callout-tip collapse="true" title="Solution"}
1. **Local sensitivity.** Replacing one record moves the $r$-th order statistic to anywhere in $[x_{r-1}, x_{r+1}]$ (Fact 4 of the chapter with $j = r$, and the same two replacements, by 0 and by $\Lambda$, attain both ends). So $\mathrm{LS}_r(D) = \max(x_r - x_{r-1}, x_{r+1} - x_r)$.
2. **Distance $k$.** The derivation of the median formula used the rank $m$ only through "exactly $m$ values are at or below the order statistic". Replacing $m$ by $r$ everywhere gives
$$A^{(k)}_r(D) = \max_{0 \le t \le k+1}\big(x_{r+t} - x_{r+t-k-1}\big),$$
with the same padding: the windows now slide around rank $r$. The script's function `order_stat_a_profile` takes $r$ as a parameter, and the median is the case $r = (n + 1)/2$.
3. **Compute.** For the ages, $x_{901} = 55.21$, $\mathrm{LS} = 0.0114$, and $S^*_{1/6} = 0.2666$ at $k = 7$. The median has $\mathrm{LS} = 0.1386$ but $S^*_{1/6} = 0.1632$, also at $k = 7$. With Cauchy noise the 90th percentile gets scale $6S^*/\varepsilon = 1.599$.
4. **Why the order flips.** The local sensitivity is one gap, and a single gap is noisy: the 90th percentile happened to sit between two close values. The smooth sensitivity is governed by $A^{(7)}$, the span of 8 consecutive values. Near the 90th percentile the data are sparser than near the median (the normal density is lower about 1.2 standard deviations above the mean), so 8 consecutive values span more.

*Smooth sensitivity reflects the density of the data around the statistic, not one lucky gap. Quantiles in the tails cost more noise than the median.*
:::

## Sizing the proposal {#sec-sol-sensitivity-4}

> **Exercise 4.** **Sizing the proposal.** With $\varepsilon = 1$ and $\delta = 10^{-6}$, how large must $\operatorname{dist}_b(D)$ be for propose-test-release to answer with probability at least 0.99? For the ages, what is the smallest $b$ that achieves it? *Check: distance 18 (17.73 rounded up), pass probability 0.9924; $b = A^{(17)} = 0.7892$.* *(Propose-test-release)*

::: {.callout-tip collapse="true" title="Solution"}
1. **The pass probability.** With threshold $T = \ln(1/\delta)/\varepsilon = 13.82$ and distance $d \ge T$, the test passes when $\mathrm{Lap}(1/\varepsilon) > T - d$, which has probability $1 - \frac12e^{-\varepsilon(d - T)}$.
2. **Solve for 0.99.** $\frac12e^{-\varepsilon(d - T)} \le 0.01$ means $d \ge T + \ln(50)/\varepsilon = 13.82 + 3.91 = 17.73$.
3. **Round up.** The distance counts records, so it is an integer: $d \ge 18$. At $d = 18$ the pass probability is $1 - \frac12e^{-(18 - 13.82)} = 0.9924$.
4. **Translate into $b$.** $\operatorname{dist}_b(D) = \min\{k : A^{(k)}(D) > b\}$, and $A^{(k)}$ is non-decreasing in $k$. So $\operatorname{dist}_b(D) \ge 18$ exactly when $A^{(17)}(D) \le b$.
5. **The ages.** $A^{(17)} = 0.7892$, so the smallest proposal that passes with probability 0.99 is $b = 0.7892$ years; the script confirms $\operatorname{dist}_{0.7892} = 18$. The chapter's $b = 1$ has distance 21, with room to spare.

*Propose-test-release needs stability against about $\ln(1/\delta)/\varepsilon$ changes plus a few more for reliability. The proposal has to cover the local sensitivity at that distance, not at distance 1.*
:::

## The number of blocks {#sec-sol-sensitivity-5}

> **Exercise 5.** **The number of blocks.** Approximate @eq-sens-saa-error by $(\theta B/n)^2 + \theta^2/n + 2(R/(B\varepsilon))^2$ and find the minimizing $B$ in closed form. Compare with the exact optimum over the divisors of $n = 10{,}000$. *Check: $B^* = (2R^2n^2/(\theta\varepsilon)^2)^{1/4} = 265.9$; the best divisor is 250.* *(Bias–variance trade-off)*

::: {.callout-tip collapse="true" title="Solution"}
1. **Where the approximation comes from.** The bias $\theta/(n_b - 1)$ with $n_b = n/B$ is about $\theta B/n$ when blocks are large, and the sampling variance $\operatorname{Var}(z)/B$ is about $\theta^2/n$, which does not depend on $B$.
2. **Differentiate** $g(B) = \theta^2B^2/n^2 + 2R^2/(B^2\varepsilon^2)$: $g'(B) = 2\theta^2B/n^2 - 4R^2/(B^3\varepsilon^2)$.
3. **Set to zero:** $B^4 = 2R^2n^2/(\theta^2\varepsilon^2)$, so $B^* = \big(2R^2n^2/(\theta\varepsilon)^2\big)^{1/4}$. The second derivative is positive, so this is a minimum.
4. **Numbers.** With $R = 5$, $n = 10^4$, $\theta = \varepsilon = 1$: $B^* = (2 \times 25 \times 10^8)^{1/4} = 265.9$.
5. **Exact optimum.** $B$ must divide $n$ for equal blocks. Evaluating the exact error @eq-sens-saa-error at every divisor, the best is $B = 250$ (blocks of 40), with error $0.0396$. The neighbouring divisors are worse: 200 gives $0.0421$ and 400 gives $0.0465$.

*At the optimum, bias and noise are of the same size, and $B^*$ grows like $\sqrt{n}$: more data buys more, smaller blocks.*
:::

## Clipping exponential data {#sec-sol-sensitivity-6}

> **Exercise 6.** **Clipping exponential data.** For $X \sim \mathrm{Exponential}(1)$, show $\mathbb E[(X - C)_+] = e^{-C}$ and, ignoring the sampling term, that the best clip bound solves $e^{-2C} = 2C/(n\varepsilon)^2$. *Check: $C^* = 5.692$ for $n = 1000$, $\varepsilon = 1$; $3.616$ for $n = 100$ and $7.834$ for $n = 10{,}000$.* *(Clipping)*

::: {.callout-tip collapse="true" title="Solution"}
1. **The bias.** $\mathbb E[(X - C)_+] = \int_C^\infty(x - C)e^{-x}\,dx$. Substitute $u = x - C$: $e^{-C}\int_0^\infty ue^{-u}\,du = e^{-C}$. (This is memorylessness: given $X > C$, the excess is again $\mathrm{Exponential}(1)$, with mean 1, and $\Pr[X > C] = e^{-C}$.)
2. **The objective without the sampling term:** $e^{-2C} + 2C^2/(n\varepsilon)^2$.
3. **Differentiate and set to zero:** $-2e^{-2C} + 4C/(n\varepsilon)^2 = 0$, that is $e^{-2C} = 2C/(n\varepsilon)^2$. The left side falls and the right side rises in $C$, so there is exactly one root, which bisection finds.
4. **Numbers.** For $n = 1000$, $\varepsilon = 1$: $C^* = 5.692$, where the bias is $e^{-5.692} = 0.0034$, the noise standard deviation $\sqrt2\,C^*/1000 = 0.0080$, and their combined root-mean-square error $0.0087$. For $n = 100$, $C^* = 3.616$; for $n = 10{,}000$, $C^* = 7.834$.
5. **Growth with $n$.** Taking logarithms, $2C^* = 2\ln(n\varepsilon) - \ln(2C^*)$, so $C^* \approx \ln(n\varepsilon)$ minus a slowly growing correction: each factor of 10 in $n$ adds about $\ln 10 = 2.3$ minus a little, as the three values show.

*For a light (exponential) tail, the clip bound grows only logarithmically with $n\varepsilon$. For heavier tails, such as the chapter's log-normal incomes, it grows much faster.*
:::

## Smoothness at work {#sec-sol-sensitivity-7}

> **Exercise 7.** **Smoothness at work.** For the degenerate pair $D = (0, 0, 0, 0, 100)$ and $D' = (0, 0, 0, 100, 100)$ with $\beta = 1/6$, compute $A^{(k)}$ for both and the smooth sensitivities. Check that their ratio is at most $e^\beta$, and explain why it is exactly $e^\beta$ here. *Check: $S^*(D) = 84.648$ at $k = 1$, $S^*(D') = 100$, ratio $1.1814 = e^{1/6}$.* *(Smooth sensitivity)*

::: {.callout-tip collapse="true" title="Solution"}
1. **$D$, distance 0.** $n = 5$, $m = 3$, so $\mathrm{LS}(D) = \max(x_3 - x_2, x_4 - x_3) = \max(0, 0) = 0$.
2. **$D$, distance $k \ge 1$.** With $k = 1$ the windows are $x_3 - x_1 = 0$, $x_4 - x_2 = 0$ and $x_5 - x_3 = 100$, so $A^{(1)} = 100$: replace $x_4 = 0$ by 100 and the median sits at the bottom of a gap of 100. Since $A^{(k)} \le \Lambda = 100$, $A^{(k)} = 100$ for all $k \ge 1$ (@tbl-sol-sensitivity-degenerate).
3. **$D'$.** Already $\mathrm{LS}(D') = x_4 - x_3 = 100$, so $A^{(k)} = 100$ for every $k$.

| $k$ | 0 | 1 | 2 | 3 | 4 | 5 |
| --- | --- | --- | --- | --- | --- | --- |
| $A^{(k)}(D)$ | 0 | 100 | 100 | 100 | 100 | 100 |
| $A^{(k)}(D')$ | 100 | 100 | 100 | 100 | 100 | 100 |

: $A^{(k)}$ for the degenerate pair $D = (0, 0, 0, 0, 100)$ and $D' = (0, 0, 0, 100, 100)$ with $\Lambda = 100$. {#tbl-sol-sensitivity-degenerate}

4. **Smooth sensitivities.** For $D'$ the maximum of $e^{-k/6}\times100$ is at $k = 0$: $S^*(D') = 100$. For $D$ the $k = 0$ term is 0, and the largest of the rest is at $k = 1$: $S^*(D) = 100e^{-1/6} = 84.648$.
5. **The ratio** is $100/84.648 = 1.1814 = e^{1/6}$, exactly the largest ratio smoothness allows.
6. **Why it is tight.** $D$'s largest discounted term is $D'$ itself, one step away. So $S^*(D)$ is $S^*(D')$ discounted by exactly one factor $e^{-\beta}$, which is step 2 of the chapter's smoothness proof holding with equality.

*Compare with local sensitivity, which jumps from 0 to 100 on this pair and makes the release on $D$ deterministic. Smooth sensitivity sees $D'$ coming and already adds noise of scale 84.6/α on $D$.*
:::
