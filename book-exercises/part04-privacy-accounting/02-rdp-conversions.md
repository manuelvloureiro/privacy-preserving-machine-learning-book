# Exercises: From Rényi DP to (ε, δ) and to tests

Exercises for [the chapter on converting Rényi differential privacy](../../book-full/part04-privacy-accounting/02-rdp-conversions.md), each with a worked solution. Try each exercise before opening its solution. The script `code/part04-privacy-accounting/solutions_rdp_conversions.py` computes every number quoted here; run it with `uv run python code/part04-privacy-accounting/solutions_rdp_conversions.py`.

The solutions use the chapter's notation. Rényi differential privacy (RDP) is converted to $(\varepsilon, \delta)$-differential privacy (DP). $P = M(D)$ and $Q = M(D')$, $L = \ln(p/q)$ is the privacy loss, and under $P$ it is the privacy loss random variable (PRV). $\alpha$ is always the Rényi order, $\varepsilon(\alpha)$ the RDP curve, and $\delta = 10^{-5}$ unless stated otherwise.

## The optimal constant, in general {#sec-sol-conv-1}

> **Exercise 1.** **The optimal constant, in general.** Show that $\sup_z (1 - e^{\varepsilon - z})_+ e^{-(\alpha - 1)z}$ is attained at $z^* = \varepsilon - \ln(1 - 1/\alpha)$, and that the ratio of the improved to Mironov's $\delta$ is $\frac1\alpha(1 - \frac1\alpha)^{\alpha - 1}$, with $\alpha$ times this ratio tending to $1/e$. *Check: at $\alpha = 2$, $\varepsilon = 1$, $c^* = 9.196986\times10^{-2}$; $\alpha$ times the ratio is 0.5000, 0.3874, 0.3697, 0.3681 at $\alpha = 2, 10, 100, 1000$.* *(Calculus, majorizing an integrand)*

::: {.callout-tip collapse="true" title="Solution"}
1. **Only $z > \varepsilon$ matters.** For $z \le \varepsilon$ the positive part is 0. For $z > \varepsilon$ the function is $f(z) = e^{-(\alpha - 1)z} - e^{\varepsilon - \alpha z}$.
2. **Stationary point.** $f'(z) = e^{-\alpha z}\big(\alpha e^\varepsilon - (\alpha - 1)e^z\big)$. The bracket is positive for small $z$ and negative for large $z$, and vanishes at $e^{z} = \frac{\alpha}{\alpha - 1}e^\varepsilon$, so $f$ rises and then falls, with its maximum at $z^* = \varepsilon + \ln\frac{\alpha}{\alpha - 1} = \varepsilon - \ln(1 - 1/\alpha)$.
3. **Value.** $f(z^*) = e^{-\alpha z^*}(e^{z^*} - e^\varepsilon) = e^{-\alpha z^*}\frac{e^\varepsilon}{\alpha - 1}$, and $e^{-\alpha z^*} = e^{-\alpha\varepsilon}(1 - 1/\alpha)^\alpha$, so $c^* = \frac{e^{-(\alpha - 1)\varepsilon}}{\alpha - 1}(1 - 1/\alpha)^\alpha$.
4. **Check.** At $\alpha = 2$ and $\varepsilon = 1$: $z^* = 1 + \ln 2 = 1.6931$ and $c^* = e^{-1}\cdot\frac14 = 9.196986\times10^{-2}$. The maximum of $f$ over a grid of two million points gives the same seven digits.
5. **The ratio.** Mironov's bound uses $c = e^{-(\alpha - 1)\varepsilon}$; both bounds multiply $c$ by the same $e^{(\alpha - 1)\varepsilon(\alpha)}$. So the ratio is $c^*/c = \frac{(1 - 1/\alpha)^\alpha}{\alpha - 1} = \frac1\alpha(1 - \frac1\alpha)^{\alpha - 1}$, using $\frac{1 - 1/\alpha}{\alpha - 1} = \frac1\alpha$.
6. **The limit.** $\alpha$ times the ratio is $(1 - 1/\alpha)^{\alpha - 1} = \exp\big((\alpha - 1)\ln(1 - 1/\alpha)\big)$. With $\ln(1 - x) = -x - x^2/2 - \dots$ and $x = 1/\alpha$, the exponent is $-(\alpha - 1)/\alpha - O(1/\alpha) \to -1$. The script gives 0.5000, 0.3874, 0.3697 and 0.3681 at $\alpha = 2, 10, 100, 1000$, approaching $1/e = 0.3679$.

*So at large orders the improved conversion divides Mironov's $\delta$ by about $e\alpha$: the gain is largest in $\delta$ exactly where it is smallest in $\varepsilon$, because $\ln(e\alpha)$ is divided by $\alpha - 1$.*
:::

## Best orders for a linear curve {#sec-sol-conv-2}

> **Exercise 2.** **Best orders for a linear curve.** For $\varepsilon(\alpha) = \rho\alpha$, derive Mironov's best order and value and @eq-conv-alpha-star, and evaluate both at $\rho = 0.5$ and $\rho = 0.005$ for $\delta = 10^{-5}$ and $10^{-10}$. *Check: at $\rho = 0.5$, $\delta = 10^{-10}$, Mironov 7.286 at order 7.786 and improved 6.839 at order 7.483.* *(Optimization over α)*

::: {.callout-tip collapse="true" title="Solution"}
1. **Mironov.** $g(\alpha) = \rho\alpha + \frac{\ln(1/\delta)}{\alpha - 1}$ has $g'(\alpha) = \rho - \frac{\ln(1/\delta)}{(\alpha - 1)^2}$ and $g'' > 0$. The zero is $\alpha^* = 1 + \sqrt{\ln(1/\delta)/\rho}$. At it, $\rho\alpha^* = \rho + \sqrt{\rho\ln(1/\delta)}$ and $\frac{\ln(1/\delta)}{\alpha^* - 1} = \sqrt{\rho\ln(1/\delta)}$, so $\varepsilon^* = \rho + 2\sqrt{\rho\ln(1/\delta)}$.
2. **Improved.** $h(\alpha) = \rho\alpha + \ln(1 - 1/\alpha) + \frac{\ln(1/\delta) - \ln\alpha}{\alpha - 1}$. The derivative of the middle term is $\frac{1}{\alpha(\alpha - 1)}$; the derivative of the last is $-\frac{1}{\alpha(\alpha - 1)} - \frac{\ln(1/\delta) - \ln\alpha}{(\alpha - 1)^2}$. They cancel in part, leaving $h'(\alpha) = \rho - \frac{\ln(1/(\alpha\delta))}{(\alpha - 1)^2}$, whose zero is @eq-conv-alpha-star.
3. **Solve it.** Rewrite as $\alpha = 1 + \sqrt{\ln(1/(\alpha\delta))/\rho}$ and iterate from Mironov's $\alpha^*$. The right side changes slowly with $\alpha$ (through $\ln\alpha$), so the iteration converges; the script's residual $\rho(\alpha - 1)^2 - \ln(1/(\alpha\delta))$ is at most $8.9\times10^{-16}$.
4. **Values.** @tbl-sol-conv-orders. At $\rho = 0.5$ and $\delta = 10^{-10}$, Mironov gives 7.286 at order 7.786 and the improved conversion 6.839 at order 7.483. Evaluating $g$ directly at Mironov's $\alpha^*$ reproduces each $\varepsilon^*$.

| $\rho$ | $\delta$ | Mironov order | Mironov $\varepsilon$ | Improved order | Improved $\varepsilon$ |
| --- | --- | --- | --- | --- | --- |
| 0.5 | $10^{-5}$ | 5.799 | 5.299 | 5.432 | 4.728 |
| 0.5 | $10^{-10}$ | 7.786 | 7.286 | 7.483 | 6.839 |
| 0.005 | $10^{-5}$ | 48.985 | 0.485 | 40.525 | 0.375 |
| 0.005 | $10^{-10}$ | 68.861 | 0.684 | 62.467 | 0.604 |

: Best continuous orders for a linear RDP curve. {#tbl-sol-conv-orders}

*The improved conversion always prefers a smaller order than Mironov's, because the $-\ln\alpha/(\alpha - 1)$ term rewards orders near the left of the U.*
:::

## One order's boundary at small $a$ {#sec-sol-conv-3}

> **Exercise 3.** **One order's boundary at small $a$.** For a single order $\alpha$, show from the second inequality of @eq-conv-region that every test has power $1 - b \le e^{(\alpha - 1)\varepsilon(\alpha)/\alpha}a^{(\alpha - 1)/\alpha}$, and that keeping the second term at its value 1 for $b = 1$ gives the approximation $1 - b \approx (e^{(\alpha - 1)\varepsilon(\alpha)} - 1)^{1/\alpha}a^{(\alpha - 1)/\alpha}$. For the Gaussian at $\mu = 1$, compare both with the exact single-order boundary at $a = 10^{-6}$ for orders 2 and 8, and say when the slope picture of @fig-conv-tradeoff is accurate. *Check: at order 2, $1.6487\times10^{-3}$, $1.3108\times10^{-3}$ and exact $1.3118\times10^{-3}$; at order 8 all three are $1.8622\times10^{-4}$.* *(Test region, asymptotics)*

::: {.callout-tip collapse="true" title="Solution"}
1. **A bound that always holds.** The second inequality of @eq-conv-region is $(1 - b)^\alpha a^{1 - \alpha} + b^\alpha(1 - a)^{1 - \alpha} \le e^{(\alpha - 1)\varepsilon(\alpha)}$. The second term is non-negative, so $(1 - b)^\alpha \le e^{(\alpha - 1)\varepsilon(\alpha)}a^{\alpha - 1}$. Taking the $\alpha$-th root gives $1 - b \le e^{(\alpha - 1)\varepsilon(\alpha)/\alpha}a^{(\alpha - 1)/\alpha}$ for every $a$: on log axes a line of slope $(\alpha - 1)/\alpha$.
2. **The approximation.** At small $a$ the allowed power is small, so $b$ is near 1 and the second term $b^\alpha(1 - a)^{1 - \alpha}$ is near 1. Moving it to the right side as 1 gives $(1 - b)^\alpha a^{1 - \alpha} \lesssim e^{(\alpha - 1)\varepsilon(\alpha)} - 1$, which is the approximation. It is not a bound: $b^\alpha \approx 1 - \alpha(1 - b)$ is a little below 1, so the exact boundary allows slightly more power.
3. **The ratio of the two.** The approximation divided by the bound is $(1 - e^{-(\alpha - 1)\varepsilon(\alpha)})^{1/\alpha}$, the same at every $a$. With $\varepsilon(\alpha) = \alpha/2$ it is 0.7951 at order 2, where $(\alpha - 1)\varepsilon(\alpha) = 1$, and 1.0000 at order 8, where $(\alpha - 1)\varepsilon(\alpha) = 28$ and $e^{-28}$ is negligible.
4. **Against the exact boundary.** @tbl-sol-conv-slope compares both with the exact single-order boundary, computed by bisection with both inequalities of @eq-conv-region. At order 2 and $a = 10^{-6}$ the bound gives $1.6487\times10^{-3}$, the approximation $1.3108\times10^{-3}$ and the exact boundary $1.3118\times10^{-3}$. At order 8 all three are $1.8622\times10^{-4}$.

| Order | $a$ | Exact | Bound (step 1) | Approximation (step 2) | Bound / exact | Approximation / exact |
| --- | --- | --- | --- | --- | --- | --- |
| 2 | $10^{-3}$ | $4.2431\times10^{-2}$ | $5.2137\times10^{-2}$ | $4.1452\times10^{-2}$ | 1.2287 | 0.9769 |
| 2 | $10^{-6}$ | $1.3118\times10^{-3}$ | $1.6487\times10^{-3}$ | $1.3108\times10^{-3}$ | 1.2568 | 0.9992 |
| 8 | $10^{-3}$ | $7.8529\times10^{-2}$ | $7.8529\times10^{-2}$ | $7.8529\times10^{-2}$ | 1.0000 | 1.0000 |
| 8 | $10^{-6}$ | $1.8622\times10^{-4}$ | $1.8622\times10^{-4}$ | $1.8622\times10^{-4}$ | 1.0000 | 1.0000 |

: Largest power allowed by one order of the Gaussian's RDP curve at $\mu = 1$: exact, the bound of step 1 and the approximation of step 2. {#tbl-sol-conv-slope}

5. **When the slope picture is accurate.** The slope $(\alpha - 1)/\alpha$ is right for every order: the approximation is the same line shifted down by the constant factor of step 3. The intercept $e^{(\alpha - 1)\varepsilon(\alpha)/\alpha}$ is right only when the budget $(\alpha - 1)\varepsilon(\alpha)$ is large, as at order 8; at order 2 it overstates the power by a factor of about 1.26. The approximation is close once the power is small (0.9992 at $a = 10^{-6}$), and less so at $a = 10^{-3}$, where order 2 still allows power 0.04.

*The orders that guard small $a$ are the large ones, and for them the simple line is exact to all printed digits; the small orders matter in the bulk, where neither line is the whole story and the script bisects.*
:::

## Composing Laplace mechanisms in RDP {#sec-sol-conv-4}

> **Exercise 4.** **Composing Laplace mechanisms in RDP.** Extend Exercise 2 of [Rényi DP](../../book-full/part04-privacy-accounting/01-renyi-dp.md) with the improved conversion. One hundred Laplace mechanisms each with $\varepsilon_0 = 0.1$: compare basic composition, advanced composition in its textbook form ([composition](../../book-full/part03-dp-foundations/04-composition.md)), and the improved conversion of the summed curve, 100 times the Laplace RDP curve $\varepsilon(\alpha) = \frac{1}{\alpha - 1}\ln\big[\frac{\alpha}{2\alpha - 1}e^{(\alpha - 1)\varepsilon_0} + \frac{\alpha - 1}{2\alpha - 1}e^{-\alpha\varepsilon_0}\big]$ (Mironov 2017). *Check: 10.000, 5.850 and 4.533 at $\delta = 10^{-5}$.* *(Composition in RDP, conversion)*

::: {.callout-tip collapse="true" title="Solution"}
1. **Check the curve.** A Riemann sum of $p^\alpha q^{1 - \alpha}$ for $\mathrm{Lap}(0, 1/\varepsilon_0)$ against $\mathrm{Lap}(1, 1/\varepsilon_0)$ gives 0.009644 at $\alpha = 2$ and 0.065854 at $\alpha = 20$, equal to the formula. For comparison, $\alpha\varepsilon_0^2/2$ is 0.010000 and 0.100000: the Laplace curve is below the generic $\varepsilon$-DP bound of the chapter's claim 3.
2. **Basic composition.** $k\varepsilon_0 = 100 \times 0.1 = 10.000$, with $\delta = 0$.
3. **Advanced composition, textbook form.** $\varepsilon_0\sqrt{2k\ln(1/\delta)} + k\varepsilon_0(e^{\varepsilon_0} - 1) = 5.850$ at $\delta = 10^{-5}$.
4. **RDP.** Composition adds curves, so the total is $100\,\varepsilon(\alpha)$. The improved conversion over the grid gives 4.533 at order 5.8; Mironov's gives 5.071 at order 6.3.
5. **Using only the ε-DP bound.** Replacing the Laplace curve by $\min(\varepsilon_0, \alpha\varepsilon_0^2/2)$ gives 4.729. That is the Gaussian number of the chapter: $100 \times \alpha\varepsilon_0^2/2 = \alpha/2$, the RDP curve of a Gaussian with $\mu = 1$.

*RDP beats advanced composition here even with the crude conversion, and the exact Laplace curve gains a further 0.196 over the generic bound.*
:::

## How fine must the grid be? {#sec-sol-conv-5}

> **Exercise 5.** **How fine must the grid be?** For $\varepsilon(\alpha) = \rho\alpha$, compute the relative excess of the grid's improved $\varepsilon$ over the continuous optimum at $\rho = 5000$, $0.005$, $5\times10^{-5}$ and $5.556\times10^{-6}$. *Check: 2.48%, 0.01%, 4.76% and 22.86%.* *(Choosing the order)*

::: {.callout-tip collapse="true" title="Solution"}
1. **Continuous optimum.** Solve @eq-conv-alpha-star by the iteration of Exercise 2.
2. **Grid optimum.** Evaluate @eq-conv-improved at every order of the grid and take the smallest.
3. **Compare.** @tbl-sol-conv-grid lists the four cases, written as a Gaussian with $\sigma$ and $k$.

| $\sigma$ | $k$ | $\rho$ | Best order | $\varepsilon$ | Grid order | Grid $\varepsilon$ | Excess |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 10,000 | 5000 | 1.048 | 5475.7911 | 1.1 | 5611.7783 | 2.48% |
| 10 | 1 | 0.005 | 40.525 | 0.3753 | 41 | 0.3753 | 0.01% |
| 100 | 1 | $5\times10^{-5}$ | 338.312 | 0.0308 | 256 | 0.0323 | 4.76% |
| 300 | 1 | $5.556\times10^{-6}$ | 919.694 | 0.0091 | 512 | 0.0112 | 22.86% |

: The grid's improved ε against the continuous optimum. {#tbl-sol-conv-grid}

4. **Read it.** Where the grid is dense, its integer and tenth-step orders are fine enough (0.01%). Where it is sparse (between 63 and 512) or ends, the loss grows quickly: the curve $\varepsilon'(\alpha)$ is steep on the side of the optimum where the grid stops, at $\rho = 5\times10^{-5}$ the best order 338.312 falls between the grid's 256 and 512, and at $\rho = 5.556\times10^{-6}$ the grid's largest order 512 is far below the best order 919.694.

*If an accountant's best order is an endpoint of its grid, extend the grid; a fixed grid is only safe for the range of $\rho$ it was designed for.*
:::

## When is the improved conversion optimal for one order? {#sec-sol-conv-6}

> **Exercise 6.** **When is the improved conversion optimal for one order?** Show that for any pair, the improved bound exceeds the true $\delta$ by at least $c^*\,\mathbb E_P[e^{(\alpha - 1)L}\mathbf 1[L \le \varepsilon]]$, and use the worst two-outcome pair to explain why the improved conversion is optimal at large orders and not at small ones. *Check: at $\mu = 1$, improved and optimal are 11.1266 and 10.6680 at order 2, and 5.2141 and 5.2141 at order 8.* *(Optimal conversion, data processing)*

::: {.callout-tip collapse="true" title="Solution"}
1. **The gap as an expectation.** The improved bound is $c^*e^{(\alpha - 1)\varepsilon(\alpha)} \ge c^*\,\mathbb E_P[e^{(\alpha - 1)L}]$ by the RDP curve (Fact 2 of the chapter). The true $\delta$ is $\mathbb E_P[(1 - e^{\varepsilon - L})_+]$ (Fact 1). So the gap is at least $\mathbb E_P\big[c^*e^{(\alpha - 1)L} - (1 - e^{\varepsilon - L})_+\big]$.
2. **Split by $L$.** Where $L > \varepsilon$ the integrand is $\ge 0$, because $c^*e^{(\alpha - 1)z}$ lies above $(1 - e^{\varepsilon - z})_+$ everywhere. Where $L \le \varepsilon$ the second term is 0. Dropping the first part leaves the claimed lower bound $c^*\,\mathbb E_P[e^{(\alpha - 1)L}\mathbf 1[L \le \varepsilon]]$.
3. **The worst pair cannot avoid it.** The optimal conversion for one order is attained by a two-outcome pair (the chapter's test region). One outcome carries the large loss, near $z^*$; the other must carry almost all the mass, at a loss near 0, because $P$ and $Q$ both sum to 1. That outcome contributes about $c^* \times 1$ to the gap. So the improved $\delta$ exceeds the optimal $\delta$ by about $c^*$.
4. **Compare $c^*$ with δ.** At the improved $\varepsilon$, $c^*/\delta$ is 0.687 at order 1.5, 0.368 at order 2, 0.002 at order 4 and essentially 0 at order 8 ($c^* = 6.914\times10^{-18}$). Since $c^* \propto e^{-(\alpha - 1)\varepsilon}/(\alpha - 1)$, it is negligible against $\delta$ once $(\alpha - 1)\varepsilon$ clearly exceeds $\ln(1/\delta)$.
5. **In ε.** @tbl-sol-conv-single gives the two conversions for the Gaussian's single values $\varepsilon(\alpha) = \alpha/2$. The optimal one uses the forward constraint $D_\alpha(P\|Q) \le \varepsilon(\alpha)$ alone, as the improved conversion does. At order 2 the improved conversion gives 11.1266 against the optimal 10.6680; at order 8 both give 5.2141.

| Order | Improved | Optimal for this order | $c^*/\delta$ |
| --- | --- | --- | --- |
| 1.5 | 21.8663 | 19.5415 | 0.687 |
| 2 | 11.1266 | 10.6680 | 0.368 |
| 4 | 5.0879 | 5.0870 | 0.002 |
| 8 | 5.2141 | 5.2141 | 0.000 |

: The improved conversion against the optimal conversion from one order, Gaussian at μ = 1. {#tbl-sol-conv-single}

*For the plain Gaussian, and for long runs of any mechanism, the best order is large enough that $c^*$ is negligible, so there the improved conversion is optimal and searching the whole test region of the chapter gains nothing over it. Short runs are different: after one step of a subsampled Gaussian the curve is tiny, the conversion's fixed toll $\ln(1/\delta)/(\alpha - 1)$ is most of the answer, and the conversion slack is 0.4656 of the 0.7561 excess over the numerical accountant (the slack table in [choosing an accountant](../../book-full/part04-privacy-accounting/09-choosing-an-accountant.md)).*
:::
