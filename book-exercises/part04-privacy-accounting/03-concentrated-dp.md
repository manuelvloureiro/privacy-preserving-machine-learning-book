# Exercises: Concentrated differential privacy

Exercises for [Concentrated differential privacy](../../book-full/part04-privacy-accounting/03-concentrated-dp.md), each with a worked solution. Try each exercise before opening its solution. The script `code/part04-privacy-accounting/solutions_concentrated_dp.py` computes every number quoted here; run it with `uv run python code/part04-privacy-accounting/solutions_concentrated_dp.py`.

The solutions use the chapter's notation. A mechanism is $\rho$-zero-concentrated differentially private ($\rho$-zCDP) when its Rényi differential privacy (RDP) curve satisfies $\varepsilon(\alpha) \le \rho\alpha$ for all $\alpha > 1$; $(\varepsilon, \delta)$-differential privacy (DP) is the guarantee you report; $\ell = \ln(1/\delta)$, and $\delta = 10^{-5}$ throughout, so $\ell = 11.5129$. Divergences are Kullback–Leibler (KL) or Rényi.

## Laplace in a zCDP budget {#sec-sol-cdp-1}

> **Exercise 1.** **Laplace in a zCDP budget.** The Laplace mechanism with scale $b$ on a sensitivity-1 query is $\varepsilon$-DP with $\varepsilon = 1/b$. Using its exact RDP curve (@eq-div-laplace-renyi with $t = \varepsilon$), find numerically the smallest $\rho$ for $\varepsilon = 1$, and compare it with $\varepsilon^2/2$ and with randomized response at the same $\varepsilon$. *Check: the smallest $\rho$ is the KL divergence $t + e^{-t} - 1 = 0.3679$, reached as $\alpha \to 1$.* *(Pure DP to zCDP, tightness)*

::: {.callout-tip collapse="true" title="Solution"}
1. **The curve.** The outputs on neighbours are $\mathrm{Lap}(0, b)$ and $\mathrm{Lap}(1, b)$, a shift of $t = 1/b = \varepsilon$ scale units. By @eq-div-laplace-renyi,
$$\varepsilon(\alpha) = \frac{1}{\alpha - 1}\ln\left(\frac{\alpha}{2\alpha - 1}e^{(\alpha - 1)t} + \frac{\alpha - 1}{2\alpha - 1}e^{-\alpha t}\right).$$
The script evaluates it in log space, so large orders do not overflow.
2. **The smallest $\rho$ is the largest ratio.** $\varepsilon(\alpha) \le \rho\alpha$ for all $\alpha$ exactly when $\rho \ge \sup_{\alpha > 1}\varepsilon(\alpha)/\alpha$. @tbl-sol-cdp-laplace evaluates the ratio at $t = 1$.

| $\alpha$ | $\varepsilon(\alpha)$ | $\varepsilon(\alpha)/\alpha$ |
| --- | --- | --- |
| 1.001 | 0.3682 | 0.3678 |
| 2 | 0.6191 | 0.3096 |
| 10 | 0.9287 | 0.0929 |
| 100 | 0.9930 | 0.0099 |

: The RDP curve of the Laplace mechanism with $t = \varepsilon = 1$ and its slope from the origin. {#tbl-sol-cdp-laplace}

3. **The ratio is largest as $\alpha \to 1$.** There $\varepsilon(\alpha) \to D_{\mathrm{KL}} = t + e^{-t} - 1$ (the KL divergence of the Laplace pair, from [Divergences](../../book-full/part01-prerequisites/03-divergences.md)), and $\alpha \to 1$. On the fine grid of orders the maximum of $\varepsilon(\alpha)/\alpha$ is $0.3679$, equal to the KL divergence $0.3679$.
4. **Compare.** The general bound is $\varepsilon^2/2 = 0.5000$; randomized response at $\varepsilon = 1$ needs $\varepsilon\tanh(\varepsilon/2) = 0.4621$; the Laplace mechanism only $0.3679$.

*The general $\varepsilon^2/2$ of @thm-cdp-pure is a convenient upper bound, not the cost of any mechanism at fixed $\varepsilon$. Every $\varepsilon$-DP mechanism is a post-processing of randomized response (@thm-comp-worst), so none needs more than randomized response's $\varepsilon\tanh(\varepsilon/2) = 0.4621$ at $\varepsilon = 1$ (@thm-cdp-pure-sharp). The Laplace mechanism, whose privacy loss reaches $\pm\varepsilon$ only in its tails, needs $0.3679$. Charging it $\varepsilon^2/2$ is safe but overcharges by a factor $0.5000/0.3679 = 1.36$; the sharp $\varepsilon\tanh(\varepsilon/2)$ still overcharges by $0.4621/0.3679 = 1.26$.*
:::

## Calibrating σ for many steps {#sec-sol-cdp-2}

> **Exercise 2.** **Calibrating $\sigma$ for many steps.** Derive @eq-cdp-inverse. Then find the noise multiplier that makes $k = 100$ Gaussian steps $(1, 10^{-5})$-DP by the simple conversion, and the $\varepsilon$ that the improved conversion and the exact Gaussian profile give for that $\sigma$. *Check: $\sigma = 49.01$.* *(Optimizing over $\alpha$, inverting a conversion)*

::: {.callout-tip collapse="true" title="Solution"}
1. **Inversion.** The simple conversion @eq-cdp-simple is $\varepsilon = \rho + 2\sqrt{\rho\ell}$, increasing in $\rho$. Put $x = \sqrt\rho \ge 0$: $x^2 + 2\sqrt\ell\,x = \varepsilon$. Add $\ell$ to both sides: $(x + \sqrt\ell)^2 = \varepsilon + \ell$. So $x = \sqrt{\varepsilon + \ell} - \sqrt\ell$ (the other root is negative), and every $\rho \le x^2$ meets the target. That is @eq-cdp-inverse.
2. **The total budget.** With $\varepsilon = 1$ and $\ell = 11.5129$, $\rho = 0.02082$.
3. **Split it over the steps.** Composition adds $\rho$, so each of the 100 steps may spend $\rho/100 = 2.0820 \times 10^{-4}$. A Gaussian step with noise multiplier $\sigma$ spends $1/(2\sigma^2)$, so $\sigma = 1/\sqrt{2\rho/100} = \sqrt{100}/\sqrt{2\rho} = 49.01$.
4. **Check the other conversions.** 100 steps at $\sigma = 49.01$ give back $\rho = 0.02082$ and $\varepsilon = 1.0000$ by the simple formula. The improved conversion gives $0.8118$. The composition of 100 Gaussian steps is itself a Gaussian pair with $\mu = \sqrt{2\rho}$, whose exact $\varepsilon$ is $0.7416$.

*Calibrating with the simple formula buys a guarantee that is really about $\varepsilon = 0.74$. Calibrating with the exact Gaussian profile instead would let you use less noise for the same reported $\varepsilon$.*
:::

## Hölder for two steps {#sec-sol-cdp-3}

> **Exercise 3.** **Hölder for two steps.** Write out @eq-cdp-triangle for $m = 2$ and $\alpha = 2$, and check it on $P = \mathrm{Bern}(0.2)$, $Q = \mathrm{Bern}(0.8)$ with $R = \mathrm{Bern}(0.4)$ and with $R = \mathrm{Bern}(0.5)$. When does it hold with equality? *Check: the left side is $1.1787$; the right side is $1.1872$ for $R = \mathrm{Bern}(0.4)$ and $1.1787$ for $R = \mathrm{Bern}(0.5)$.* *(Hölder's inequality, group privacy)*

::: {.callout-tip collapse="true" title="Solution"}
1. **The orders.** With group size $m = 2$ and $\alpha = 2$: the first order is $\frac{m\alpha - 1}{m - 1} = 3$, the coefficient $\frac{m\alpha}{m\alpha - 1} = \frac43$, and the second order $m\alpha = 4$:
$$D_2(P\|Q) \le \tfrac43D_3(P\|R) + D_4(R\|Q).$$
The Hölder exponents are $u = \frac{m\alpha - 1}{\alpha(m - 1)} = \frac32$ and $v = \frac{m\alpha - 1}{\alpha - 1} = 3$, and indeed $\frac23 + \frac13 = 1$.
2. **Evaluate.** Each divergence is a two-term sum, $D_\alpha(P\|Q) = \frac{1}{\alpha-1}\ln\sum_y P(y)^\alpha Q(y)^{1-\alpha}$. @tbl-sol-cdp-holder lists the values.

| $R$ | $D_2(P\Vert Q)$ | $D_3(P\Vert R)$ | $D_4(R\Vert Q)$ | $\frac43D_3 + D_4$ |
| --- | --- | --- | --- | --- |
| $\mathrm{Bern}(0.4)$ | 1.1787 | 0.1934 | 0.9294 | 1.1872 |
| $\mathrm{Bern}(0.5)$ | 1.1787 | 0.3662 | 0.6904 | 1.1787 |

: The triangle-like inequality for $m = 2$, $\alpha = 2$, $P = \mathrm{Bern}(0.2)$, $Q = \mathrm{Bern}(0.8)$. {#tbl-sol-cdp-holder}

3. **When equality holds.** Hölder's inequality $\mathbb E[XY] \le (\mathbb E X^u)^{1/u}(\mathbb E Y^v)^{1/v}$ is an equality when $X^u$ and $Y^v$ are proportional. Here $X = (P/R)^\alpha$ and $Y = (R/Q)^{\alpha - 1}$, ratios of the densities at an output drawn from $R$, so $X^u = (P/R)^3$ and $Y^v = (R/Q)^3$: equality needs $\ln(P/R) - \ln(R/Q)$ to be the same for every outcome. For Bernoulli distributions that means the log-odds of $R$ sit exactly halfway between those of $P$ and $Q$. The log-odds of $0.2$, $0.4$, $0.5$, $0.8$ are $-1.3863$, $-0.4055$, $0.0000$, $1.3863$, so $\mathrm{Bern}(0.5)$ is the midpoint and $\mathrm{Bern}(0.4)$ is not.
4. **The link to the Gaussian.** A chain of Gaussians $\mathcal N(0, \sigma^2)$, $\mathcal N(1, \sigma^2)$, $\mathcal N(2, \sigma^2)$ has the same structure: both log-ratios are linear in $y$ with the same slope, so they differ by a constant. That is why group privacy $m^2\rho$ is exact for the Gaussian mechanism.

*The inequality loses nothing when the middle dataset is "halfway" in the sense of the privacy loss. For the Gaussian mechanism every chain of neighbours is like that.*
:::

## A mixed pipeline {#sec-sol-cdp-4}

> **Exercise 4.** **A mixed pipeline.** A pipeline runs 20 propose-test-release steps, each $(0.25, 10^{-8})$-DP, and 50 Gaussian steps with $\sigma = 5$. Find its $\rho$, its $\delta_a$, and $\varepsilon$ at total $\delta = 10^{-5}$ with the improved conversion. *Check: $\rho = 1.625$.* *(Approximate zCDP)*

::: {.callout-tip collapse="true" title="Solution"}
1. **Each stability step.** An $(\varepsilon, \delta)$-DP step is $\delta$-approximately $(\varepsilon^2/2)$-zCDP (@prp-cdp-approx, part 1), so each costs $\rho = 0.25^2/2 = 0.03125$ and fails with probability $10^{-8}$.
2. **Each Gaussian step** costs $1/(2 \cdot 25) = 0.020$ and never fails.
3. **Compose.** $\rho = 20 \times 0.03125 + 50 \times 0.020 = 1.6250$ and $\delta_a = 1 - (1 - 10^{-8})^{20} = 2.0000 \times 10^{-7}$.
4. **Convert.** Spend the rest of the $\delta$ on the conversion: $\delta' = 10^{-5} - \delta_a$. The improved conversion at $\rho = 1.625$ and this $\delta'$ gives $\varepsilon = 9.463$, at total $\delta = \delta_a + \delta' = 10^{-5}$ (@prp-cdp-approx, part 3, which allows $\delta_a + (1 - \delta_a)\delta' \le 10^{-5}$).
5. **Compare.** Treating the two kinds of steps separately, the Gaussians alone ($\rho = 1$) convert to $\varepsilon = 7.084$ at $\delta = 10^{-5} - 2 \times 10^{-7} = 9.8 \times 10^{-6}$, and basic composition of the stability steps adds $20 \times 0.25 = 5$, for $12.084$.

*Twenty steps of $\varepsilon = 0.25$ cost 5 under basic composition but only $\rho = 0.625$ in the zCDP ledger, because they compose like Gaussian steps. That is the point of approximate zCDP.*
:::

## The subsampled Gaussian at order 2 {#sec-sol-cdp-5}

> **Exercise 5.** **The subsampled Gaussian at order 2.** From @eq-cdp-sg, show $\varepsilon(2) = \ln\big(1 + q^2(e^{1/\sigma^2} - 1)\big)$, and evaluate it for $q = 0.01$, $\sigma = 1$. Then show that the lower bound @eq-cdp-sg-lower is positive exactly when $\alpha > 1 + 2\sigma^2\ln(1/q)$. *Check: $\varepsilon(2) = 1.7181 \times 10^{-4}$.* *(Binomial expansion)*

::: {.callout-tip collapse="true" title="Solution"}
1. **The three terms of $A_2$.** In @eq-cdp-sg with $\alpha = 2$, $j = 0$ gives $(1 - q)^2$, $j = 1$ gives $2q(1 - q)e^0$, and $j = 2$ gives $q^2e^{(4 - 2)/(2\sigma^2)} = q^2e^{1/\sigma^2}$.
2. **Add.** $(1 - q)^2 + 2q(1 - q) = (1 - q)(1 + q) = 1 - q^2$, so $A_2 = 1 - q^2 + q^2e^{1/\sigma^2} = 1 + q^2(e^{1/\sigma^2} - 1)$.
3. **Divide** $\ln A_2$ by $\alpha - 1 = 1$. For $q = 0.01$, $\sigma = 1$: the closed form and the binomial sum both give $1.7181 \times 10^{-4}$. Without subsampling, $\varepsilon(2) = 1$, so subsampling at 1% buys a factor of about 5,800 at this order: $1/q^2 = 10^4$, divided by $e^{1/\sigma^2} - 1 = e - 1 = 1.72$.
4. **Where the lower bound turns positive.** $\rho_0\alpha + \frac{\alpha\ln q}{\alpha - 1} > 0$ with $\rho_0 = 1/(2\sigma^2)$ and $\alpha > 1$ means $\frac{\alpha - 1}{2\sigma^2} > \ln(1/q)$, that is $\alpha > 1 + 2\sigma^2\ln(1/q)$.
5. **Check.** For $q = 0.01$, $\sigma = 1$, the threshold is $10.21$. The bound is $-0.1169$ at $\alpha = 10$ and $0.4343$ at $\alpha = 11$.

*The threshold $1 + 2\sigma^2\ln(1/q)$ is where the curve bends in @fig-cdp-subsampled: below it the rare event "the record is in every sample" is too rare to matter, above it that event dominates.*
:::

## The tCDP conversion {#sec-sol-cdp-6}

> **Exercise 6.** **The tCDP conversion.** Apply the conversion @eq-cdp-tcdp-conv of truncated concentrated DP (tCDP) to $10{,}000$ steps of the running example with $\omega = 4$ and $\omega = 8$, saying which branch applies and where the unconstrained best order lies. *Check: $7.469$ and $8.289$.* *(Constrained optimization over $\alpha$)*

::: {.callout-tip collapse="true" title="Solution"}
1. **The rule.** By @prp-cdp-tcdp, the first branch of @eq-cdp-tcdp-conv applies when the unconstrained best order $\alpha^* = 1 + \sqrt{\ell/\rho}$ is at most $\omega$, that is when $\ell \le (\omega - 1)^2\rho$. Otherwise the best admissible order is $\omega$ itself.
2. **$\omega = 4$.** Per step $\rho = 9.0789 \times 10^{-5}$ (chapter, @tbl-cdp-tcdp), so $10{,}000$ steps give $\rho = 0.9079$. Then $(\omega - 1)^2\rho = 8.171 < \ell = 11.5129$: second branch, $\varepsilon = 7.469$. The unconstrained minimizer $\alpha^* = 4.561$ lies beyond $\omega$.
3. **$\omega = 8$.** $\rho = 1.1171$ and $(\omega - 1)^2\rho = 54.736 \ge \ell$: first branch, with $\alpha^* = 4.210 \le 8$, and $\varepsilon = 8.289$.

*A larger $\omega$ makes more orders available but forces a larger slope, because the line has to clear the curve on more orders. Here the slope wins, and the smaller truncation point gives the better $\varepsilon$.*
:::

## A line for a 5-DP mechanism {#sec-sol-cdp-7}

> **Exercise 7.** **A line for a 5-DP mechanism.** A mechanism is $5$-DP. Find the $\rho$ that @thm-cdp-pure and @thm-cdp-pure-sharp charge for it, find the order at which each line $\rho\alpha$ reaches the pure-DP cap $\varepsilon = 5$ of @eq-cdp-pure, and say how an accountant should describe the mechanism. *Check: $\rho = 12.5$ and $4.9331$; the sharp line reaches the cap at $\alpha = 1.0136$.* *(Pure DP to zCDP, the cap on the RDP curve)*

::: {.callout-tip collapse="true" title="Solution"}
1. **The general constant.** @thm-cdp-pure charges $\varepsilon^2/2 = 25/2 = 12.5$. The line $12.5\alpha$ is above 5 at every order $\alpha > 1$, because $12.5 > 5$ already at $\alpha = 1$. So it is worse than the cap everywhere.
2. **The sharp constant.** @thm-cdp-pure-sharp charges $\varepsilon\tanh(\varepsilon/2) = 5\tanh(2.5) = 4.9331$. This line starts just below the cap and reaches it where $4.9331\alpha = 5$, at $\alpha = 5/4.9331 = 1.0136$.
3. **What the accountant should keep.** Above $\alpha = 1.0136$ the cap $\varepsilon = 5$ is below both lines, and the exact curve of randomized response, which bounds every $5$-DP mechanism (@thm-cdp-pure-sharp), is lower still. Converting either line back gives a worse guarantee than the one you started with. The simple conversion @eq-cdp-simple returns $\rho$ plus a positive term, and $\rho = 4.9331$ is already within $0.07$ of 5, so the result exceeds 5 for every $\delta < 1$.

*For a large $\varepsilon$, as in local DP, describe the mechanism by its RDP curve or by $\varepsilon$ itself, never by a single $\rho$. A line through the origin cannot follow a curve that flattens almost at once.*
:::
