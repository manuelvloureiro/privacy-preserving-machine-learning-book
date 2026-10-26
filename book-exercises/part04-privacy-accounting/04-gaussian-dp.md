# Exercises: f-DP and Gaussian differential privacy

Exercises for the chapter on differential privacy (DP) as a trade-off curve, [f-DP and Gaussian differential privacy](../../book-full/part04-privacy-accounting/04-gaussian-dp.md), each with a worked solution. Try each exercise before opening its solution. The script `code/part04-privacy-accounting/solutions_gaussian_dp.py` computes every number quoted here; run it with `uv run python code/part04-privacy-accounting/solutions_gaussian_dp.py`.

The solutions use the chapter's notation. $\mu$-Gaussian differential privacy (GDP) means the trade-off curve $G_\mu(\alpha) = \Phi(\Phi^{-1}(1 - \alpha) - \mu)$. A test has type I error $\alpha$ and type II error $\beta$; the chapter writes them $\mathrm{FPR}$ and $\mathrm{FNR}$, because it needs $\alpha$ for the order of Rényi divergences. $P = M(D)$ and $Q = M(D')$, the privacy loss is $L = \ln(P/Q)$, and Dong, Roth & Su (2022) is cited as Dong et al. Three more acronyms appear: the total variation (TV) distance, the central limit theorem (CLT) for composition, and differentially private stochastic gradient descent (SGD), written DP-SGD.

## The Laplace mechanism is GDP {#sec-sol-gdp-1}

> **Exercise 1.** **The Laplace mechanism is GDP.** Find the smallest $\mu$ such that the Laplace mechanism with $\varepsilon = 1$ is $\mu$-GDP, that is $G_\mu \le T(\mathrm{Lap}(0, 1), \mathrm{Lap}(1, 1))$ everywhere. Check numerically that the binding point is on the diagonal, derive the closed form this gives, and prove that it suffices. What $\varepsilon$ at $\delta = 10^{-5}$ does this GDP summary report, and what is the truth? *Check: $\mu = 1.0301$ (the same as Part 1's Gaussian with equal TV); the GDP summary gives $\varepsilon = 4.5301$, while the mechanism is $(1, 0)$-DP.* *(Ordering trade-off curves, symmetry)*

::: {.callout-tip collapse="true" title="Solution"}
Write $T_\varepsilon = T(\mathrm{Lap}(0, 1/\varepsilon), \mathrm{Lap}(1, 1/\varepsilon))$, Part 1's Laplace curve with $\kappa = \varepsilon$.

1. **Both curves are symmetric.** $G_\mu$ is symmetric (Part 1, Exercise 1), and $T_\varepsilon$ is too: its first and last pieces, $1 - e^\varepsilon \alpha$ and $e^{-\varepsilon}(1 - \alpha)$, are reflections of each other, and the middle piece $\beta = e^{-\varepsilon}/(4\alpha)$ is symmetric in $\alpha$ and $\beta$.
2. **A necessary condition on the diagonal.** On the diagonal $\alpha = \beta$, $G_\mu(\alpha) = \alpha$ means $\Phi^{-1}(1 - \alpha) - \mu = \Phi^{-1}(\alpha) = -\Phi^{-1}(1 - \alpha)$, so $\alpha = \Phi(-\mu/2)$. The Laplace curve meets the diagonal on its middle piece: $\alpha^2 = e^{-\varepsilon}/4$, so $\alpha = \frac12e^{-\varepsilon/2}$. For $G_\mu \le T_\varepsilon$ the Gaussian must cross the diagonal no further out, $\Phi(-\mu/2) \le \frac12 e^{-\varepsilon/2}$, that is
$$\mu \ge \mu^* = 2\,\Phi^{-1}\big(1 - \tfrac12e^{-\varepsilon/2}\big).$$
3. **It is also sufficient.** Both curves are symmetric, so by the duality of the chapter (step 3 there) each is the upper envelope of its lines $f_{\varepsilon',\delta(\varepsilon')}$, $\varepsilon' \ge 0$. Since $f_{\varepsilon',\delta}$ falls as $\delta$ grows, $G_\mu \le T_\varepsilon$ holds as soon as $\delta_G(\varepsilon') \ge \delta_{\mathrm{Lap}}(\varepsilon')$ for every $\varepsilon' \ge 0$, where $\delta_G$ is @eq-gdp-delta at $\mu = \mu^*$ and $\delta_{\mathrm{Lap}}(\varepsilon') = 1 - e^{(\varepsilon' - \varepsilon)/2}$ for $\varepsilon' \le \varepsilon$ and 0 beyond (the Laplace profile of Part 3, [the Laplace and Gaussian mechanisms](../../book-full/part03-dp-foundations/03-laplace-gaussian.md), Exercise 2). Let $D(\varepsilon') = \delta_G(\varepsilon') - \delta_{\mathrm{Lap}}(\varepsilon')$.
    - *At $\varepsilon' = 0$:* $\delta_G(0) = 2\Phi(\mu^*/2) - 1 = 1 - 2\Phi(-\mu^*/2) = 1 - e^{-\varepsilon/2} = \delta_{\mathrm{Lap}}(0)$ by the choice of $\mu^*$, so $D(0) = 0$.
    - *The slope.* Part 3 (Exercise 3 of [the Laplace and Gaussian mechanisms](../../book-full/part03-dp-foundations/03-laplace-gaussian.md)) showed $\delta_G'(\varepsilon') = -e^{\varepsilon'}\Phi(v)$ with $v = -\varepsilon'/\mu^* - \mu^*/2$, and $\delta_{\mathrm{Lap}}'(\varepsilon') = -\frac12e^{(\varepsilon' - \varepsilon)/2}$ on $[0, \varepsilon]$. So $D' \ge 0$ exactly when $h(\varepsilon') = e^{\varepsilon'/2}\Phi(v) \le \frac12e^{-\varepsilon/2} = \Phi(-\mu^*/2) = h(0)$.
    - *$h$ is non-increasing.* $h'(\varepsilon') = e^{\varepsilon'/2}\big[\frac12\Phi(v) - \varphi(v)/\mu^*\big]$. Since $v \le -\mu^*/2 < 0$, the Mills-ratio bound $\Phi(v) < \varphi(v)/|v|$ ([Concentration inequalities](../../book-full/part01-prerequisites/02-concentration.md)) gives $\Phi(v)/\varphi(v) < 1/|v| \le 2/\mu^*$, so $h' < 0$.
    - *Conclusion.* $h(\varepsilon') \le h(0)$, so $D' \ge 0$ and $D \ge 0$ on $[0, \varepsilon]$. Beyond $\varepsilon$, $\delta_{\mathrm{Lap}} = 0 \le \delta_G$. Hence $G_{\mu^*} \le T_\varepsilon$, and with step 2, $\mu^*$ is the smallest $\mu$ for every $\varepsilon > 0$.

    The script confirms it numerically. It finds the smallest $\mu$ with $G_\mu \le T_\varepsilon$ on a grid of 1,999 values of $\alpha$ by bisection. For $\varepsilon = 1$ it gives $\mu = 1.0301$, equal to the closed form 1.0301, and the two curves come closest (they touch) at $\alpha = 0.3035$, the grid point next to the diagonal point $\frac12e^{-1/2} = 0.3033$. For $\varepsilon = 0.5$: $\mu = 0.5618$ from both, touching at $\alpha = 0.3895$, next to $0.3894$.
4. **The same as matching TV.** The diagonal point of a symmetric curve is where the best advantage $1 - \alpha - \beta = 1 - 2\alpha$ is attained (Part 1). So $\mu^*$ is the Gaussian shift with the same total variation as the Laplace pair, $2\Phi(\mu/2) - 1 = 1 - e^{-\varepsilon/2}$, which Part 1 found to be 1.0301 at $\kappa = 1$.
5. **What the summary reports.** By @eq-gdp-delta, 1.0301-GDP gives $\varepsilon = 4.5301$ at $\delta = 10^{-5}$ (and 0.5618-GDP gives 2.2706). The Laplace mechanism is $(1, 0)$-DP.

*The GDP summary of a Laplace mechanism is tight at one point, the diagonal, and very loose in the tails: the Laplace privacy loss is bounded by $\varepsilon$, while every $G_\mu$ has an unbounded one.*
:::

## Group privacy {#sec-sol-gdp-2}

> **Exercise 2.** **Group privacy.** Let $h(\alpha) = 1 - f(\alpha)$ be the largest power at type I error $\alpha$ that a curve $f$ allows. Show that if $T(P, Q) \ge f$ and $T(Q, R) \ge f$, then $T(P, R) \ge 1 - h(h(\alpha))$, so an $f$-DP mechanism is $\big(1 - h^{\circ k}\big)$-DP for groups of $k$ records (Dong et al., Theorem 2.14). For $f = G_\mu$, show that $1 - h^{\circ k} = G_{k\mu}$. What $\varepsilon$ at $\delta = 10^{-5}$ does a 0.5-GDP mechanism give a group of 2 and a group of 4, and what does Part 3's group property (@eq-dpprop-group) give when applied to the single point $(1.9931, 10^{-5})$ instead? *Check: groups of 2 and 4 are 1-GDP and 2-GDP, so $\varepsilon = 4.3772$ and $9.9973$. From the point, a group of 4 gets $(7.9724, 4.57\times10^{-3})$, while 2-GDP has $\delta(7.9724) = 5.20\times10^{-4}$.* *(Trade-off curves as power bounds, GDP)*

::: {.callout-tip collapse="true" title="Solution"}
1. **A curve bounds power.** $T(P, Q) \ge f$ says that every test $\phi$ with $\mathbb E_P\phi \le \alpha$ has type II error at least $f(\alpha)$, that is, power $\mathbb E_Q\phi \le 1 - f(\alpha) = h(\alpha)$. Since $f$ is non-increasing, $h$ is non-decreasing.
2. **Chain the bounds.** Take a test $\phi$ of $P$ against $R$ with $\mathbb E_P\phi \le \alpha$. Read as a test of $P$ against $Q$, it has power $\mathbb E_Q\phi \le h(\alpha)$. Read as a test of $Q$ against $R$, its type I error is $\mathbb E_Q\phi \le h(\alpha)$, so its power is $\mathbb E_R\phi \le h(\mathbb E_Q\phi) \le h(h(\alpha))$, using that $h$ is non-decreasing. So its type II error against $R$ is at least $1 - h(h(\alpha))$, and $T(P, R) \ge 1 - h\circ h$.
3. **Groups.** Datasets that differ in $k$ records are joined by a path $D = D_0 \sim D_1 \sim \dots \sim D_k$. Applying step 2 along the path $k - 1$ times gives $T(M(D_0), M(D_k)) \ge 1 - h^{\circ k}$.
4. **The Gaussian case.** For $f = G_\mu$, $h(\alpha) = 1 - \Phi(\Phi^{-1}(1 - \alpha) - \mu) = \Phi(\mu - \Phi^{-1}(1 - \alpha)) = \Phi(\Phi^{-1}(\alpha) + \mu)$, using $\Phi^{-1}(1 - \alpha) = -\Phi^{-1}(\alpha)$. Each application of $h$ adds $\mu$ inside $\Phi$, so $h^{\circ k}(\alpha) = \Phi(\Phi^{-1}(\alpha) + k\mu)$ and $1 - h^{\circ k}(\alpha) = \Phi(\Phi^{-1}(1 - \alpha) - k\mu) = G_{k\mu}(\alpha)$. The script checks $1 - h(h(\alpha)) = G_{2\mu}(\alpha)$ at $\mu = 0.5$ on a grid of 1,999 values of $\alpha$: the largest difference is $2.2\times10^{-16}$. For the Gaussian mechanism the bound is tight, because a group of $k$ shifts the mean by up to $k$ sensitivities.
5. **The numbers.** A 0.5-GDP mechanism is 1-GDP for groups of 2 and 2-GDP for groups of 4, so by @eq-gdp-delta it gives $\varepsilon = 4.3772$ and $9.9973$ at $\delta = 10^{-5}$.
6. **Going through one point instead.** 0.5-GDP gives $\varepsilon = 1.9931$ at $\delta = 10^{-5}$. Part 3's group property turns $(\varepsilon, \delta)$ into $\big(k\varepsilon, \delta\frac{e^{k\varepsilon} - 1}{e^\varepsilon - 1}\big)$:

| Group size $k$ | $k\varepsilon$ | $\delta_k$ from the point | GDP $\delta(k\varepsilon)$ of $G_{k\mu}$ | $\varepsilon$ at $\delta = 10^{-5}$ from GDP |
| --- | --- | --- | --- | --- |
| 2 | 3.9862 | $8.34\times10^{-5}$ | $4.98\times10^{-5}$ | 4.3772 |
| 4 | 7.9724 | $4.57\times10^{-3}$ | $5.20\times10^{-4}$ | 9.9973 |

: Group privacy of a 0.5-GDP mechanism: through one $(\varepsilon, \delta)$ point and through the curve. {#tbl-sol-gdp-group}

The point route pays twice (@tbl-sol-gdp-group): $\delta_k$ is 1.7 times the curve's $\delta$ at the same $\varepsilon$ for a pair, and 8.8 times for a group of 4.

*Group privacy is one more place where keeping the curve beats keeping a point: for GDP the curve of a group is again Gaussian, with $\mu$ growing linearly in the group size.*
:::

## A tensor product by hand {#sec-sol-gdp-3}

> **Exercise 3.** **A tensor product by hand.** Show that $f_{\varepsilon,0}\otimes f_{\varepsilon,0} = \max\big(f_{2\varepsilon,0}, f_{0,\tanh(\varepsilon/2)}\big)$, and read off its privacy profile $\delta(\varepsilon')$ for $0 \le \varepsilon' \le 2\varepsilon$ from the corners. *Check: at $\varepsilon = 1$, $\delta(0) = 0.4621$, $\delta(1) = 0.3378$, $\delta(2) = 0$.* *(Tensor product, duality)*

::: {.callout-tip collapse="true" title="Solution"}
Let $\pi = e^\varepsilon/(1 + e^\varepsilon)$, so randomized response reports the true bit with probability $\pi$, and note $2\pi - 1 = \tanh(\varepsilon/2)$ and $\pi/(1 - \pi) = e^\varepsilon$.

1. **The product pair.** As in @tbl-gdp-rr2, the output pairs 11, 01 or 10, and 00 have likelihood ratios $Q/P = e^{2\varepsilon}$, $1$ and $e^{-2\varepsilon}$, and $P$-probabilities $(1 - \pi)^2$, $2\pi(1 - \pi)$ and $\pi^2$.
2. **The corners.** Adding outcomes in decreasing order of $Q/P$ gives the corners $(0, 1)$, $\big((1 - \pi)^2, 1 - \pi^2\big)$, $\big(1 - \pi^2, (1 - \pi)^2\big)$ and $(1, 0)$. At $\varepsilon = 1$ these are $(0.0723, 0.4656)$ and $(0.4656, 0.0723)$.
3. **First segment.** It has slope $-e^{2\varepsilon}$ and starts at $(0, 1)$: $\beta = 1 - e^{2\varepsilon}\alpha$. It ends at the second corner because $e^{2\varepsilon}(1 - \pi)^2 = \pi^2$.
4. **Middle segment.** Slope $-1$ through $\big((1 - \pi)^2, 1 - \pi^2\big)$: $\beta = 1 - \pi^2 + (1 - \pi)^2 - \alpha = 1 - (2\pi - 1) - \alpha = 1 - \tanh(\varepsilon/2) - \alpha$.
5. **Last segment.** Slope $-e^{-2\varepsilon}$ through $(1, 0)$: $\beta = e^{-2\varepsilon}(1 - \alpha)$.
6. **A convex polygon is the maximum of its lines.** So the curve is $\max\big(0, 1 - e^{2\varepsilon}\alpha, e^{-2\varepsilon}(1 - \alpha), 1 - \tanh(\varepsilon/2) - \alpha\big)$. The first three terms make up $f_{2\varepsilon,0}$ and the last (with 0) is $f_{0,\tanh(\varepsilon/2)}$, as claimed. On a grid of 1,999 values of $\alpha$, the two sides differ by at most $2.2\times10^{-16}$ at $\varepsilon = 1$, where $\tanh(1/2) = 0.4621$.

**The privacy profile.** By the duality, $\delta(\varepsilon') = \max_{\alpha}(1 - f(\alpha) - e^{\varepsilon'}\alpha)$, and for a polygon the maximum is at a corner. The two interior corners, and the midpoint $(1 - \pi, 1 - \pi)$ of the middle segment, give $1 - \beta - e^{\varepsilon'}\alpha$ equal to $\pi^2 - (1 - \pi)^2e^{\varepsilon'}$, $\big(1 - (1 - \pi)^2\big) - (1 - \pi^2)e^{\varepsilon'}$ and $\pi - (1 - \pi)e^{\varepsilon'}$. For $\varepsilon' \ge 0$ the first is the largest: it beats the second by $(e^{\varepsilon'} - 1)(1 - \pi^2 - (1 - \pi)^2) = (e^{\varepsilon'} - 1)\,2\pi(1 - \pi) \ge 0$, and the midpoint value is the average of the two. So
$$\delta(\varepsilon') = \pi^2 - (1 - \pi)^2e^{\varepsilon'}, \qquad 0 \le \varepsilon' \le 2\varepsilon,$$
which reaches 0 at $\varepsilon' = 2\varepsilon$. At $\varepsilon = 1$ ($\pi^2 = 0.5344$, $(1 - \pi)^2 = 0.0723$):

| $\varepsilon'$ | 0 | 0.5 | 1 | 1.5 | 2 |
| --- | --- | --- | --- | --- | --- |
| $\delta(\varepsilon')$ from the corners | 0.4621 | 0.4152 | 0.3378 | 0.2103 | 0.0000 |
| $\delta(\varepsilon')$ from the exact binomial loss | 0.4621 | 0.4152 | 0.3378 | 0.2103 | 0.0000 |

: The privacy profile of two runs of randomized response with $\varepsilon = 1$. {#tbl-sol-gdp-rr2}

At $\varepsilon' = 0$ all three points give $2\pi - 1 = \tanh(1/2)$, the total variation of the product pair. @tbl-sol-gdp-rr2 agrees with the hockey-stick formula applied to the exact loss.

*Two composed $(\varepsilon, 0)$ mechanisms are $(2\varepsilon, 0)$-DP and $(0, \tanh(\varepsilon/2))$-DP at once, and only the pair of statements, not either one, is the whole guarantee.*
:::

## Inverting the duality {#sec-sol-gdp-4}

> **Exercise 4.** **Inverting the duality.** Find the largest $\mu$ such that $\mu$-GDP implies $(1, 10^{-5})$-DP. How many full-batch Gaussian steps with $\sigma = 20$ fit in that budget? How many DP-SGD steps with $q = 0.01$, $\sigma = 1$ fit in budgets $\varepsilon = 1, 2, 4$ according to the CLT? *Check: $\mu^* = 0.2681$; 28 full-batch steps; 418, 1463 and 4978 DP-SGD steps.* *(GDP duality, composition)*

::: {.callout-tip collapse="true" title="Solution"}
1. **$\delta(\varepsilon)$ grows with $\mu$.** A larger $\mu$ gives a lower curve, so every supporting line moves down and $\delta(\varepsilon) = 1 + G_\mu^*(-e^\varepsilon)$ goes up. So the set of $\mu$ that satisfy $\delta_\mu(1) \le 10^{-5}$ is an interval $[0, \mu^*]$, found by bisection on @eq-gdp-delta: $\mu^* = 0.2681$, where $\delta_{\mu^*}(1) = 1.00\times10^{-5}$.
2. **Full batch.** $T$ Gaussian steps with $\sigma = 20$ compose to $\sqrt T/20$-GDP (@eq-gdp-compose). The budget allows $\sqrt T/20 \le 0.2681$, so $T \le (20 \times 0.2681)^2$, and the largest integer is $T = 28$.
3. **DP-SGD by the CLT.** With $\mu = q\sqrt{T(e^{1/\sigma^2} - 1)}$ the budget allows $T \le (\mu^*/q)^2/(e^{1/\sigma^2} - 1)$. With $q = 0.01$ and $\sigma = 1$:

| Budget $\varepsilon$ at $\delta = 10^{-5}$ | $\mu^*$ | Largest $T$ | CLT $\varepsilon$ at that $T$ |
| --- | --- | --- | --- |
| 1 | 0.2681 | 418 | 0.9998 |
| 2 | 0.5016 | 1463 | 1.9992 |
| 4 | 0.9249 | 4978 | 3.9996 |

: How many DP-SGD steps fit in a budget, according to the CLT. {#tbl-sol-gdp-steps}

$\mu^*$ grows more slowly than $\varepsilon$, and $T$ grows like $\mu^{*2}$, so doubling the budget from 2 to 4 multiplies the number of steps by 3.40 (@tbl-sol-gdp-steps). Subsampling is what makes these numbers large: full batch with $\sigma = 1$ would allow only $T \le \mu^{*2}$, less than one step for $\varepsilon = 1$.

*Remember that these step counts inherit the CLT's optimism: the chapter showed that for this example the CLT under-reports $\varepsilon$, so a certified accountant would allow fewer steps.*
:::

## Berry–Esseen for randomized response {#sec-sol-gdp-5}

> **Exercise 5.** **Berry–Esseen for randomized response.** Compute $\mathrm{kl}$, $\kappa_2$ and $\bar\kappa_3$ for $f_{\varepsilon_0,0}$, and show that @eq-gdp-clt-mu gives $\mu = 2\sqrt k\sinh(\varepsilon_0/2)$ for $k$ copies. *Check: $\mathrm{kl} = \varepsilon_0\tanh(\varepsilon_0/2) = 0.004996$ at $\varepsilon_0 = 0.1$; at $k = 100$, $\mu = 1.0004$ and $\xi = 0.0562$.* *(CLT functionals)*

::: {.callout-tip collapse="true" title="Solution"}
Let $t = \tanh(\varepsilon_0/2) = 2\pi - 1$ with $\pi = e^{\varepsilon_0}/(1 + e^{\varepsilon_0})$.

1. **The privacy loss under $P$.** Randomized response outputs the likely value with probability $\pi$, where $L = \ln(\pi/(1 - \pi)) = \varepsilon_0$, and the other value with probability $1 - \pi$, where $L = -\varepsilon_0$.
2. **The functionals are moments of $L$** (the chapter's derivation, step 1):
    - $\mathrm{kl} = \mathbb E_P L = \varepsilon_0\pi - \varepsilon_0(1 - \pi) = \varepsilon_0 t$;
    - $\kappa_2 = \mathbb E_P L^2 = \varepsilon_0^2$;
    - $\bar\kappa_3 = \mathbb E_P|L - \varepsilon_0t|^3 = \pi\,\varepsilon_0^3(1 - t)^3 + (1 - \pi)\,\varepsilon_0^3(1 + t)^3$. With $\pi = (1 + t)/2$ this is $\frac{\varepsilon_0^3(1 - t^2)}{2}\big((1 - t)^2 + (1 + t)^2\big) = \varepsilon_0^3(1 - t^2)(1 + t^2)$.
3. **The variance.** $\kappa_2 - \mathrm{kl}^2 = \varepsilon_0^2(1 - t^2) = \varepsilon_0^2/\cosh^2(\varepsilon_0/2)$, using $1 - \tanh^2 = 1/\cosh^2$.
4. **$\mu$ for $k$ copies.** @eq-gdp-clt-mu gives
$$\mu = \frac{2k\varepsilon_0t}{\sqrt{k\,\varepsilon_0^2/\cosh^2(\varepsilon_0/2)}} = 2\sqrt k\,\tanh(\varepsilon_0/2)\cosh(\varepsilon_0/2) = 2\sqrt k\,\sinh(\varepsilon_0/2).$$
5. **$\xi$ for $k$ copies.** $\xi = \frac{0.56\,k\varepsilon_0^3(1 - t^2)(1 + t^2)}{\big(k\varepsilon_0^2(1 - t^2)\big)^{3/2}} = \frac{0.56\,(1 + t^2)\cosh(\varepsilon_0/2)}{\sqrt k}$.
6. **Check.** At $\varepsilon_0 = 0.1$: $\mathrm{kl} = 0.1\tanh(0.05) = 0.004996$. At $k = 100$: $\mu = 20\sinh(0.05) = 1.0004$ and $\xi = 0.0562$, from the closed form and from the general formula alike.

| $k$ | $\varepsilon_0$ | $\mathrm{kl}$ | $\mu = 2\sqrt k\sinh(\varepsilon_0/2)$ | $\xi$ |
| --- | --- | --- | --- | --- |
| 10 | 0.3162 | 0.049587 | 1.0042 | 0.1837 |
| 100 | 0.1000 | 0.004996 | 1.0004 | 0.0562 |
| 400 | 0.0500 | 0.001250 | 1.0001 | 0.0280 |

: The CLT parameters for $k$-fold randomized response with $k\varepsilon_0^2 = 1$. {#tbl-sol-gdp-be}

Since $\sinh x \ge x$, $\mu \ge \sqrt k\,\varepsilon_0$: at finite $k$ the CLT centre sits slightly above the limit $\sqrt{\sum\varepsilon_i^2}$. The band $\xi$ shrinks only like $0.56/\sqrt k$ (@tbl-sol-gdp-be): quadrupling $k$ from 100 to 400 halves it.

*The Berry–Esseen band is honest but wide: at $k = 100$ it allows the true curve anywhere within 0.0562 of $G_\mu$, while the actual error in $\delta(1)$ was 0.0014 (0.1271 from the CLT against the exact 0.1257).*
:::

## A sanity limit for the subsampled CLT {#sec-sol-gdp-6}

> **Exercise 6.** **A sanity limit for the subsampled CLT.** Put $q = 1$ in @eq-gdp-sgd-mu and compare with the exact $\sqrt T/\sigma$. Which way is it off, and why does the derivation not apply? *Check: the ratio, the same for every $T$, is 1.3108 for $\sigma = 1$, 1.0659 for $\sigma = 2$ and 1.0101 for $\sigma = 5$.* *(Subsampled CLT, expansions)*

::: {.callout-tip collapse="true" title="Solution"}
1. **The formula at $q = 1$.** $\mu_{\mathrm{CLT}} = \sqrt{T(e^{1/\sigma^2} - 1)}$, while full-batch training is exactly $\sqrt T/\sigma$-GDP. Their ratio is $\sigma\sqrt{e^{1/\sigma^2} - 1}$, independent of $T$.
2. **Which way.** $e^x - 1 \ge x$ for all $x$, so $\sigma^2(e^{1/\sigma^2} - 1) \ge 1$: the formula over-states $\mu$ at $q = 1$. As $\sigma$ grows, $e^{1/\sigma^2} - 1 \approx 1/\sigma^2$ and the ratio tends to 1.

| $\sigma$ | CLT $\mu$ ($T = 100$) | Exact $\mu = \sqrt T/\sigma$ | Ratio |
| --- | --- | --- | --- |
| 1 | 13.1083 | 10.0000 | 1.3108 |
| 2 | 5.3294 | 5.0000 | 1.0659 |
| 5 | 2.0202 | 2.0000 | 1.0101 |

: The subsampled CLT formula at $q = 1$ against the exact full-batch $\mu$. {#tbl-sol-gdp-q1}

3. **Why the derivation fails.** Step 3 of the derivation expanded $\ln(1 + qW)$ to second order, which needs $qW$ small. At $q = 1$, $\ln(1 + W) = (y - \frac12)/\sigma^2$ exactly: the loss is Gaussian with variance $1/\sigma^2$, not $e^{1/\sigma^2} - 1$. The expansion replaced the variance of $\ln(1 + W)$ by the variance of $W$, which is larger, and the gap grows as $\sigma$ shrinks (@tbl-sol-gdp-q1).
4. **The regime.** The CLT formula is meant for $q \to 0$ with $q\sqrt T$ fixed. There $qW$ is small with high probability and the expansion is accurate, although for small $\sigma$ the heavy right tail of $W$ is exactly where it is least accurate.

*Formulas derived for a limit regime should be checked at the edges of that regime; here the edge $q = 1$ is known exactly and shows the formula's bias.*
:::
