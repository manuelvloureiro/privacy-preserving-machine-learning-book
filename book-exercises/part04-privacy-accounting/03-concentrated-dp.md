# Exercises: Concentrated differential privacy

Exercises for [Concentrated differential privacy](../../book-full/part04-privacy-accounting/03-concentrated-dp.md), each with a worked solution. Try each exercise before opening its solution. The script `code/part04-privacy-accounting/solutions_concentrated_dp.py` computes every number quoted here; run it with `uv run python code/part04-privacy-accounting/solutions_concentrated_dp.py`.

The solutions use the chapter's notation. A mechanism is $\rho$-zero-concentrated differentially private ($\rho$-zCDP) when its Rényi differential privacy (RDP) curve satisfies $\varepsilon(\alpha) \le \rho\alpha$ for all $\alpha > 1$; $(\varepsilon, \delta)$-differential privacy (DP) is the guarantee you report; $\ell = \ln(1/\delta)$, and $\delta = 10^{-5}$ throughout, so $\ell = 11.5129$. Divergences are Kullback–Leibler (KL) or Rényi.

## Laplace in a zCDP budget {#sec-sol-cdp-1}

> **Exercise 1.** **Laplace in a zCDP budget.** The Laplace mechanism with scale $b$ on a sensitivity-1 query is $\varepsilon$-DP with $\varepsilon = 1/b$. Using its exact RDP curve (@eq-laplace-renyi with $t = \varepsilon$), find numerically the smallest $\rho$ for $\varepsilon = 1$, and compare it with $\varepsilon^2/2$ and with randomized response at the same $\varepsilon$. *Check: the smallest $\rho$ is the KL divergence $t + e^{-t} - 1 = 0.3679$, reached as $\alpha \to 1$.* *(Pure DP to zCDP, tightness)*

::: {.callout-tip collapse="true" title="Solution"}
1. **The curve.** The outputs on neighbours are $\mathrm{Lap}(0, b)$ and $\mathrm{Lap}(1, b)$, a shift of $t = 1/b = \varepsilon$ scale units. By @eq-laplace-renyi,
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

*The general $\varepsilon^2/2$ is a convenient upper bound, not the cost of any mechanism at fixed $\varepsilon$. Every $\varepsilon$-DP mechanism is a post-processing of randomized response ([Composition](../../book-full/part03-dp-foundations/04-composition.md)), so none needs more than randomized response's $\varepsilon\tanh(\varepsilon/2) = 0.4621$ at $\varepsilon = 1$ (@eq-cdp-pure-sharp). The Laplace mechanism, whose privacy loss reaches $\pm\varepsilon$ only in its tails, needs $0.3679$. Charging it $\varepsilon^2/2$ is safe but overcharges by a factor $0.5000/0.3679 = 1.36$; the sharp $\varepsilon\tanh(\varepsilon/2)$ still overcharges by $0.4621/0.3679 = 1.26$.*
:::

## Calibrating σ for many steps {#sec-sol-cdp-2}

> **Exercise 2.** **Calibrating $\sigma$ for many steps.** Derive @eq-cdp-inverse. Then find the noise multiplier that makes $k = 100$ Gaussian steps $(1, 10^{-5})$-DP by the simple conversion, and the $\varepsilon$ that the improved conversion and the exact Gaussian profile give for that $\sigma$. *Check: $\sigma = 49.01$.* *(Optimizing over α, inverting a conversion)*

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
4. **The link to the Gaussian.** A chain of Gaussians $\mathcal N(0, \sigma^2)$, $\mathcal N(1, \sigma^2)$, $\mathcal N(2, \sigma^2)$ has the same structure: both log-ratios are linear in $y$ with the same slope, so they differ by a constant. That is why group privacy $k^2\rho$ is exact for the Gaussian mechanism.

*The inequality loses nothing when the middle dataset is "halfway" in the sense of the privacy loss. For the Gaussian mechanism every chain of neighbours is like that.*
:::

## A mixed pipeline {#sec-sol-cdp-4}

> **Exercise 4.** **A mixed pipeline.** A pipeline runs 20 propose-test-release steps, each $(0.25, 10^{-8})$-DP, and 50 Gaussian steps with $\sigma = 5$. Find its $\rho$, its $\delta_a$, and $\varepsilon$ at total $\delta = 10^{-5}$ with the improved conversion. *Check: $\rho = 1.625$.* *(Approximate zCDP)*

::: {.callout-tip collapse="true" title="Solution"}
1. **Each stability step.** An $(\varepsilon, \delta)$-DP step is $\delta$-approximately $(\varepsilon^2/2)$-zCDP, so each costs $\rho = 0.25^2/2 = 0.03125$ and fails with probability $10^{-8}$.
2. **Each Gaussian step** costs $1/(2 \cdot 25) = 0.020$ and never fails.
3. **Compose.** $\rho = 20 \times 0.03125 + 50 \times 0.020 = 1.6250$ and $\delta_a = 1 - (1 - 10^{-8})^{20} = 2.0000 \times 10^{-7}$.
4. **Convert.** Spend the rest of the $\delta$ on the conversion: $\delta' = 10^{-5} - \delta_a$. The improved conversion at $\rho = 1.625$ and this $\delta'$ gives $\varepsilon = 9.463$, at total $\delta = \delta_a + \delta' = 10^{-5}$ (property 4 of the approximate zCDP section, which allows $\delta_a + (1 - \delta_a)\delta' \le 10^{-5}$).
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

> **Exercise 6.** **The tCDP conversion.** Derive @eq-cdp-tcdp-conv and apply it to $10{,}000$ steps of the running example with $\omega = 4$ and $\omega = 8$, saying which branch applies. *Check: $7.469$ and $8.289$.* *(Constrained optimization over α)*

::: {.callout-tip collapse="true" title="Solution"}
1. **Admissible orders.** $(\rho, \omega)$-truncated concentrated DP (tCDP) bounds $D_\alpha$ by $\rho\alpha$ for $\alpha \in (1, \omega)$, and at $\alpha = \omega$ too, because $D_\alpha$ is non-decreasing and lower semicontinuous in $\alpha$, so $D_\omega \le \lim_{\alpha \uparrow \omega}D_\alpha \le \rho\omega$. Mironov's conversion at any such order gives $g(\alpha) = \rho\alpha + \ell/(\alpha - 1)$.
2. **The unconstrained minimum.** $g$ is convex with minimizer $\alpha^* = 1 + \sqrt{\ell/\rho}$ and minimum $\rho + 2\sqrt{\rho\ell}$ (the chapter's conversion section).
3. **First branch.** If $\alpha^* \le \omega$, which is $\ell \le (\omega - 1)^2\rho$, the minimizer is admissible.
4. **Second branch.** Otherwise $g' < 0$ on all of $(1, \omega]$ (a convex function decreases before its minimizer), so the best admissible order is $\omega$: $\varepsilon = \rho\omega + \ell/(\omega - 1)$.
5. **$\omega = 4$.** Per step $\rho = 9.0789 \times 10^{-5}$ (chapter, @tbl-cdp-tcdp), so $10{,}000$ steps give $\rho = 0.9079$. Then $(\omega - 1)^2\rho = 8.171 < \ell = 11.5129$: second branch, $\varepsilon = 7.469$. The unconstrained minimizer $\alpha^* = 4.561$ lies beyond $\omega$.
6. **$\omega = 8$.** $\rho = 1.1171$ and $(\omega - 1)^2\rho = 54.736 \ge \ell$: first branch, with $\alpha^* = 4.210 \le 8$, and $\varepsilon = 8.289$.

*A larger $\omega$ makes more orders available but forces a larger slope, because the line has to clear the curve on more orders. Here the slope wins, and the smaller truncation point gives the better $\varepsilon$.*
:::
