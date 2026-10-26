# Exercises: The Laplace and Gaussian mechanisms

Exercises for [the Laplace and Gaussian mechanisms](../../book-full/part03-dp-foundations/03-laplace-gaussian.md), each with a worked solution. Try each exercise before opening its solution. The script `code/part03-dp-foundations/solutions_laplace_gaussian.py` computes every number quoted here; run it with `uv run python code/part03-dp-foundations/solutions_laplace_gaussian.py`.

The solutions use the chapter's notation. Guarantees are differential privacy (DP) guarantees; $\Delta_1$ and $\Delta_2$ are the $\ell_1$ and $\ell_2$ sensitivities; $\mu = \Delta_2/\sigma$ for the Gaussian mechanism; $\Phi$ and $\varphi$ are the standard normal distribution function and density; and $\delta_\mu(\varepsilon) = \Phi(\frac\mu2 - \frac\varepsilon\mu) - e^\varepsilon\Phi(-\frac\mu2 - \frac\varepsilon\mu)$ is the exact privacy profile of the Gaussian mechanism (@eq-lg-analytic).

## A private mean under add/remove {#sec-sol-lg-1}

> **Exercise 1.** **A private mean under add/remove.** Records lie in $[0, 1]$ and neighbours differ by adding or removing a record, so $n$ is private. Release the sum $s$ and the count $n$, each with the Laplace mechanism at $\varepsilon/2$, and estimate the mean as $(s + Z_s)/(n + Z_c)$. Show that the release is $\varepsilon$-DP and that the root-mean-square error is approximately $\frac{2\sqrt2}{n\varepsilon}\sqrt{1 + m^2}$ for true mean $m$. *Check: at $n = 1000$, $m = 0.3$, $\varepsilon = 1$ this is 0.00295 (simulated 0.00296), against 0.00141 for the replace-neighbour mean with $n$ public.* *(Sensitivity, basic composition, first-order error)*

::: {.callout-tip collapse="true" title="Solution"}
**Privacy.**

1. **Sensitivities.** Adding or removing a record $x \in [0, 1]$ changes the count by 1 and the sum by $x \le 1$. So each query has $\Delta_1 = 1$.
2. **Each release.** $s + Z_s$ with $Z_s \sim \mathrm{Lap}(2/\varepsilon)$ is the Laplace mechanism with scale $\Delta_1/(\varepsilon/2)$, so it is $\frac\varepsilon2$-DP. The same holds for $n + Z_c$.
3. **Together.** By basic composition ([Post-processing, group privacy and composition](../../book-full/part03-dp-foundations/02-properties.md)) the pair is $\varepsilon$-DP. Equivalently, the vector $(s, n)$ has $\Delta_1 = 1 + 1 = 2$, and independent $\mathrm{Lap}(2/\varepsilon)$ noise on both coordinates is exactly the Laplace mechanism for it.
4. **The estimate.** Dividing the two noisy numbers is post-processing, so it costs nothing more.

**Error.** The noise facts are those of @thm-lg-laplace: each $\mathrm{Lap}(b)$ coordinate has variance $2b^2$.

1. **Write the error.** With $s = nm$,
$$\frac{nm + Z_s}{n + Z_c} - m = \frac{Z_s - mZ_c}{n + Z_c}.$$
2. **First order.** The noise has scale $b = 2/\varepsilon$, tiny next to $n$, so $n + Z_c \approx n$ and the error is $\approx (Z_s - mZ_c)/n$.
3. **Variance.** $Z_s$ and $Z_c$ are independent with variance $2b^2$ each, so the variance is $2b^2(1 + m^2)/n^2$. Take the square root and substitute $b = 2/\varepsilon$: $\frac{\sqrt2\cdot2}{n\varepsilon}\sqrt{1 + m^2} = \frac{2\sqrt2}{n\varepsilon}\sqrt{1 + m^2}$.
4. **Check.** At $n = 1000$, $m = 0.3$, $\varepsilon = 1$: $\frac{2\sqrt2}{1000}\sqrt{1.09} = 0.00295$. A million simulated releases give 0.00296.
5. **Compare.** Under replace neighbours with $n$ public, the mean has $\Delta_1 = 1/n$ and needs only $\mathrm{Lap}(1/(n\varepsilon))$ noise, with root-mean-square error $\sqrt2/(n\varepsilon) = 0.00141$. Hiding $n$ and splitting the budget costs a factor of about 2.

**A caveat.** Strictly, the ratio has no finite variance: $n + Z_c$ has a positive density at 0, so the error can be arbitrarily large with probability of order $e^{-n\varepsilon/2}$, which is $e^{-500}$ here and never shows up in a simulation. In practice you replace the denominator by $\max(n + Z_c, 1)$, which is post-processing and removes the problem.

*Under add/remove the size of the dataset is itself private, and releasing it costs part of the budget.*
:::

## The Laplace scale cannot be smaller {#sec-sol-lg-2}

> **Exercise 2.** **The Laplace scale cannot be smaller.** For a count, show that $\mathrm{Lap}(b)$ noise with $b < 1/\varepsilon$ is not $\varepsilon$-DP, by exhibiting an event. Then show that $\mathrm{Lap}(1/\varepsilon)$ noise is $(\varepsilon', \delta)$-DP for $\varepsilon' < \varepsilon$ with $\delta = 1 - e^{-(\varepsilon - \varepsilon')/2}$. *Check: $b = 0.9$, $\varepsilon = 1$ gives the ratio $e^{1/0.9} = 3.0377 > e$; with $b = 1$, $\delta(0.5) = 0.2212$.* *(Tightness, hockey-stick of a Laplace pair)*

::: {.callout-tip collapse="true" title="Solution"}
**The violating event.** The tightness step of @thm-lg-laplace already exhibits it: for counts $f(D) = 0$ and $f(D') = 1$ and the event $S = \{y \le 0\}$, the two probabilities are $\frac12$ and $\frac12e^{-1/b}$, so their ratio is $e^{1/b}$, which exceeds $e^\varepsilon$ whenever $b < 1/\varepsilon$. At $b = 0.9$ and $\varepsilon = 1$ the ratio is $e^{1/0.9} = 3.0377 > e = 2.7183$.

**The $\delta$ curve.** Work in units of $b = 1/\varepsilon$, so the shift is $t = 1/b = \varepsilon$. Let $P = \mathrm{Lap}(0, 1)$ and $Q = \mathrm{Lap}(t, 1)$.

1. **The privacy loss.** $L(y) = |y - t| - |y|$, which is $t$ for $y \le 0$, $t - 2y$ for $0 \le y \le t$, and $-t$ for $y \ge t$.
2. **The bad event.** For $0 \le \varepsilon' < t$, $L(y) > \varepsilon'$ exactly when $y < c$ with $c = (t - \varepsilon')/2 \in (0, t/2]$.
3. **Its two probabilities.** $\Pr_P[y < c] = 1 - \frac12e^{-c}$ and $\Pr_Q[y < c] = \frac12e^{-(t - c)}$.
4. **The hockey-stick** (@lem-lg-bad-event): $\delta = \Pr_P[B] - e^{\varepsilon'}\Pr_Q[B] = 1 - \frac12e^{-c} - \frac12e^{\varepsilon' - t + c}$. Since $\varepsilon' - t = -2c$, the last exponent is $-c$, and $\delta = 1 - e^{-c} = 1 - e^{-(\varepsilon - \varepsilon')/2}$.
5. **Check.** $b = 1$ ($\varepsilon = 1$) at $\varepsilon' = 0.5$: $1 - e^{-0.25} = 0.2212$. At $\varepsilon' = 0$ the formula gives $1 - e^{-1/2} = 0.3935$, the total variation distance of the pair, and at $\varepsilon' = \varepsilon$ it gives 0.

*The Laplace mechanism is exactly $\varepsilon$-DP and no better: below $\varepsilon$ you must pay a $\delta$ that is not small.*
:::

## The slope of the privacy profile {#sec-sol-lg-3}

> **Exercise 3.** **The slope of the privacy profile.** Show that $\varphi(\frac\mu2 - \frac\varepsilon\mu) = e^\varepsilon\varphi(-\frac\mu2 - \frac\varepsilon\mu)$, and deduce $\frac{\mathrm d}{\mathrm d\varepsilon}\delta_\mu(\varepsilon) = -e^\varepsilon\Phi(-\frac\mu2 - \frac\varepsilon\mu)$. Conclude that $\delta_\mu$ is strictly decreasing in $\varepsilon$, and show in the same way that $\frac{\mathrm d}{\mathrm d\mu}\delta_\mu(\varepsilon) = \varphi(\frac\mu2 - \frac\varepsilon\mu) > 0$. *Check: at $\mu = 1$, $\varepsilon = 1$ the slope is $-0.1816$, matching a finite difference.* *(Differentiating @eq-lg-analytic)*

::: {.callout-tip collapse="true" title="Solution"}
Write $a = \frac\mu2 - \frac\varepsilon\mu$ and $c = -\frac\mu2 - \frac\varepsilon\mu$, so $\delta_\mu(\varepsilon) = \Phi(a) - e^\varepsilon\Phi(c)$.

1. **The density identity.** $a^2 - c^2 = (a - c)(a + c) = \mu\cdot\big(-\frac{2\varepsilon}\mu\big) = -2\varepsilon$. So $\varphi(a)/\varphi(c) = e^{-(a^2 - c^2)/2} = e^\varepsilon$. This is the statement that the likelihood ratio at the boundary of the bad event is exactly $e^\varepsilon$.
2. **Differentiate in $\varepsilon$.** $\frac{\mathrm da}{\mathrm d\varepsilon} = \frac{\mathrm dc}{\mathrm d\varepsilon} = -\frac1\mu$. By the product and chain rules,
$$\frac{\mathrm d\delta_\mu}{\mathrm d\varepsilon} = -\frac{\varphi(a)}\mu - e^\varepsilon\Phi(c) + \frac{e^\varepsilon\varphi(c)}\mu.$$
3. **Cancel.** By step 1 the first and last terms cancel, leaving $-e^\varepsilon\Phi(c) < 0$. So $\delta_\mu$ is strictly decreasing in $\varepsilon$.
4. **Differentiate in $\mu$.** $\frac{\mathrm da}{\mathrm d\mu} = \frac12 + \frac\varepsilon{\mu^2}$ and $\frac{\mathrm dc}{\mathrm d\mu} = -\frac12 + \frac\varepsilon{\mu^2}$, so $\frac{\mathrm d\delta_\mu}{\mathrm d\mu} = \varphi(a)\big(\frac12 + \frac\varepsilon{\mu^2}\big) - e^\varepsilon\varphi(c)\big(-\frac12 + \frac\varepsilon{\mu^2}\big)$. Replace $e^\varepsilon\varphi(c)$ by $\varphi(a)$ (step 1): the $\varepsilon/\mu^2$ terms cancel and the result is $\varphi(a) > 0$.
5. **Check.** At $\mu = 1$, $\varepsilon = 1$: $c = -1.5$ and the slope is $-e\,\Phi(-1.5) = -0.1816$. A central finite difference with step $10^{-6}$ gives $-0.1816$ too, and the script evaluates the identity of step 1 to $0.0$ in floating point. At $\varepsilon = 0$, $\delta_1(0) = \Phi(\frac12) - \Phi(-\frac12) = 2\Phi(\frac12) - 1 = 0.3829$, the total variation distance.

*The slope $-e^\varepsilon\Phi(c)$ is the correction term of the bad-event bound, $e^\varepsilon \Pr_Q[B]$, so the two terms of the exact $\delta$ are tied together by one derivative.*
:::

## Calibrate by hand {#sec-sol-lg-4}

> **Exercise 4.** **Calibrate by hand.** Run bisection on $\ln\mu$ over $[10^{-3}, 100]$ for $\varepsilon = 0.5$, $\delta = 10^{-6}$, and compare the result with the classic formula. *Check: $\mu^* = 0.124106$, so $\sigma^* = 8.0576\,\Delta_2f$, against the classic 10.5976 (a ratio of 1.315).* *(Analytic calibration)*

::: {.callout-tip collapse="true" title="Solution"}
1. **Why bisection works.** By Exercise 3, $\delta_\mu(0.5)$ is strictly increasing in $\mu$, so $\delta_\mu(0.5) = 10^{-6}$ has exactly one root, and at any trial $\mu$ the sign of $\delta_\mu(0.5) - 10^{-6}$ says on which side of it you are.
2. **The steps.** Take the midpoint in $\ln\mu$, $\sqrt{\mu_{\mathrm{lo}}\mu_{\mathrm{hi}}}$. If $\delta_\mu(0.5) > 10^{-6}$ the noise is too small, so the root is below and $\mu_{\mathrm{hi}}$ moves down; otherwise $\mu_{\mathrm{lo}}$ moves up. The first midpoint is $\sqrt{10^{-3}\times100} = 0.316228$, where $\delta$ is too large. @tbl-sol-lg-bisection shows the interval after selected steps.

| Step | $\mu_{\mathrm{lo}}$ | $\mu_{\mathrm{hi}}$ |
| --- | --- | --- |
| 1 | 0.001000 | 0.316228 |
| 2 | 0.017783 | 0.316228 |
| 3 | 0.074989 | 0.316228 |
| 10 | 0.122983 | 0.124373 |
| 20 | 0.124106 | 0.124107 |
| 60 | 0.124106 | 0.124106 |

: Bisection on $\ln\mu$ for the analytic Gaussian calibration at $\varepsilon = 0.5$, $\delta = 10^{-6}$. {#tbl-sol-lg-bisection}

3. **The noise.** $\sigma^* = \Delta_2f/0.124106 = 8.0576\,\Delta_2f$.
4. **The classic formula.** $\sigma = \Delta_2f\sqrt{2\ln(1.25\times10^6)}/0.5 = 10.5976\,\Delta_2f$, a factor of 1.315 more, or 73% more variance.

*After $j$ steps the ratio $\mu_{\mathrm{hi}}/\mu_{\mathrm{lo}}$ is $(10^5)^{1/2^j}$, so 20 steps already give six digits.*
:::

## Composing Gaussian mechanisms exactly {#sec-sol-lg-5}

> **Exercise 5.** **Composing Gaussian mechanisms exactly.** You answer $k$ queries, each with $\Delta_2 = 1$, by adding independent $\mathcal N(0, \sigma^2)$ noise to each. Show that the $k$ answers together are one Gaussian mechanism with $\ell_2$ sensitivity $\sqrt k$, so their exact privacy profile is $\delta_{\sqrt k/\sigma}(\varepsilon)$. Compare with basic composition. *Check: $k = 16$, $\sigma = 4$ gives $\mu = 1$ and $\varepsilon = 4.3772$ at $\delta = 10^{-5}$; basic composition with $\delta/16$ per query gives $\varepsilon = 16 \times 1.0864 = 17.3818$.* *(@lem-lg-shift, sensitivity of a concatenation)*

::: {.callout-tip collapse="true" title="Solution"}
1. **Stack the queries.** Let $F(D) = (f_1(D), \dots, f_k(D)) \in \mathbb R^k$ (each $f_i$ real-valued, or stack the coordinates if they are vectors). The release is $F(D) + Z$ with $Z \sim \mathcal N(0, \sigma^2I_k)$, because the noises are independent with the same variance.
2. **Its sensitivity.** $\|F(D) - F(D')\|_2^2 = \sum_i|f_i(D) - f_i(D')|^2 \le k$, so $\Delta_2F \le \sqrt k$.
3. **Its privacy profile.** By @thm-lg-analytic, the release is $(\varepsilon, \delta_\mu(\varepsilon))$-DP with $\mu = \sqrt k/\sigma$ for every $\varepsilon$, and exactly that when some pair reaches $\Delta_2F = \sqrt k$.
4. **Check.** $k = 16$, $\sigma = 4$: $\mu = 4/4 = 1$. Solving $\delta_1(\varepsilon) = 10^{-5}$ gives $\varepsilon = 4.3772$, the value read off @fig-lg-delta-curves for $\sigma/\Delta_2 = 1$.
5. **Basic composition.** Each query alone has $\mu = 1/4$. Spend $\delta/16 = 6.25\times10^{-7}$ on each: solving $\delta_{1/4}(\varepsilon) = 6.25\times10^{-7}$ gives $\varepsilon = 1.0864$ per query, and basic composition adds them to $\varepsilon = 17.3818$ at total $\delta = 10^{-5}$. That is four times the exact value.

**Why.** Basic composition adds $\varepsilon$'s, so it grows like $k$. The exact curve depends on $\mu = \sqrt k/\sigma$, which grows like $\sqrt k$: sixteen queries cost as much as one query with a quarter of the noise. The same argument holds for adaptively chosen queries, with more care; [Advanced and optimal composition](../../book-full/part03-dp-foundations/04-composition.md) and [Part 4](../../book-full/part04-privacy-accounting/00-introduction.md) develop it.

*Gaussian noise composes into Gaussian noise, so for Gaussian mechanisms the exact composed guarantee is a one-line formula.*
:::

## A private mean of vectors {#sec-sol-lg-6}

> **Exercise 6.** **A private mean of vectors.** Each of $n$ records is a vector in $\mathbb R^d$ with $\|x\|_2 \le 1$, and neighbours replace one record. Find $\Delta_2$ and an upper bound on $\Delta_1$ for the mean, and compare the expected squared $\ell_2$ error of the Laplace and Gaussian mechanisms. *Check: at $d = 100$, $n = 10{,}000$, $\varepsilon = 1$, $\delta = 10^{-5}$ the root-mean-square $\ell_2$ errors are 0.02828 (Laplace) and 0.00746 (Gaussian).* *(@eq-lg-norms, dense queries)*

::: {.callout-tip collapse="true" title="Solution"}
1. **$\ell_2$ sensitivity.** Replacing $x$ by $x'$ moves the mean by $(x' - x)/n$, and $\|x' - x\|_2 \le \|x\|_2 + \|x'\|_2 \le 2$. So $\Delta_2 = 2/n$, reached by $x' = -x$.
2. **$\ell_1$ sensitivity.** By @eq-lg-norms, $\Delta_1 \le \sqrt d\,\Delta_2 = 2\sqrt d/n$, reached when $x$ and $x'$ are opposite vectors with all coordinates $\pm1/\sqrt d$.
3. **Laplace error.** Each of the $d$ coordinates has variance $2b^2$ with $b = \Delta_1/\varepsilon$, so $\mathbb E[\|Z\|_2^2] = 2db^2$ and the root-mean-square $\ell_2$ error is $\sqrt{2d}\cdot\frac{2\sqrt d}{n\varepsilon} = \frac{2\sqrt2\,d}{n\varepsilon}$.
4. **Gaussian error.** $\mathbb E[\|Z\|_2^2] = d\sigma^2$ with $\sigma = \Delta_2\sigma_1(\varepsilon, \delta)$, so the root-mean-square error is $\sqrt d\cdot\frac2n\sigma_1(\varepsilon, \delta)$.
5. **Check.** At $d = 100$, $n = 10{,}000$, $\varepsilon = 1$: Laplace gives $2\sqrt2\times100/10^4 = 0.02828$. With $\sigma_1(1, 10^{-5}) = 3.7306$, Gaussian gives $10\times2\times10^{-4}\times3.7306 = 0.00746$. The ratio is 3.79, and it grows like $\sqrt d$.

*Bounding a vector's $\ell_2$ norm makes its $\ell_1$ sensitivity grow like $\sqrt d$, so for norm-bounded data, such as clipped gradients, Gaussian noise is the right choice.*
:::

## The privacy loss distribution of the Laplace mechanism {#sec-sol-lg-7}

> **Exercise 7.** **The privacy loss distribution of the Laplace mechanism.** A count is released with $\mathrm{Lap}(1/\varepsilon)$ noise, and its true answers are 0 on $D$ and 1 on $D'$. Find the distribution of the privacy loss $L$ of an output drawn from $M(D)$: its two atoms, and the density of its continuous part. Check that the total probability is 1, and that the mean loss is $\varepsilon + e^{-\varepsilon} - 1$. *Check: at $\varepsilon = 1$ the atoms carry 0.5000 and 0.1839, and the mean loss is 0.3679.* *(@fig-lg-loss-distributions)*

::: {.callout-tip collapse="true" title="Solution"}
1. **The loss as a function of the output.** With $b = 1/\varepsilon$, the loss is $L(y) = \varepsilon(|y - 1| - |y|)$. It equals $\varepsilon$ for $y \le 0$ and $-\varepsilon$ for $y \ge 1$. In between, for $0 < y < 1$, it is $\varepsilon(1 - 2y)$, a straight line from $\varepsilon$ down to $-\varepsilon$.
2. **The atoms.** Under $M(D)$ the output is $\mathrm{Lap}(0, 1/\varepsilon)$. So $\Pr[y \le 0] = \frac12$ by symmetry, and $\Pr[y \ge 1] = \frac12e^{-\varepsilon}$, the Laplace tail beyond one unit, which is $\varepsilon$ scales. These are the atoms at $+\varepsilon$ and $-\varepsilon$.
3. **The continuous part.** On $0 < y < 1$ the density of $y$ is $\frac\varepsilon2e^{-\varepsilon y}$. Change variables to $l = \varepsilon(1 - 2y)$, so $y = (1 - l/\varepsilon)/2$ and $|\mathrm dy/\mathrm dl| = 1/(2\varepsilon)$. The density of $L$ on $(-\varepsilon, \varepsilon)$ is $\frac\varepsilon2e^{-(\varepsilon - l)/2}\cdot\frac1{2\varepsilon} = \frac14e^{(l - \varepsilon)/2}$.
4. **Total probability.** The continuous part integrates to $\frac14 \cdot 2\big(1 - e^{-\varepsilon}\big) = \frac12(1 - e^{-\varepsilon})$. Adding the atoms gives $\frac12 + \frac12e^{-\varepsilon} + \frac12 - \frac12e^{-\varepsilon} = 1$. The script's midpoint sum gives 1.0000 at $\varepsilon = 0.5$ and at $\varepsilon = 1$.
5. **The mean.** The mean of the privacy loss under $M(D)$ is the Kullback–Leibler divergence of the pair, which for two Laplace densities with centres $\varepsilon$ scales apart is $\varepsilon + e^{-\varepsilon} - 1$. The script computes the mean from the atoms and the density: 0.3679 at $\varepsilon = 1$ and 0.1065 at $\varepsilon = 0.5$, equal to the formula.

*The loss never leaves $[-\varepsilon, \varepsilon]$, which is pure DP, and half the time it sits at the worst value $\varepsilon$. A Gaussian loss is centred at a much smaller value but has no end points.*
:::
