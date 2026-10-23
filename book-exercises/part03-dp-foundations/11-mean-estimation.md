# Exercises: Private mean estimation

Exercises for [Private mean estimation](../../book-full/part03-dp-foundations/11-mean-estimation.md), each with a worked solution. Try each exercise before opening its solution. The script `code/part03-dp-foundations/solutions_mean_estimation.py` computes every number quoted here; run it with `uv run python code/part03-dp-foundations/solutions_mean_estimation.py`.

The solutions use the chapter's notation. Datasets have a fixed size $n$ and neighbours differ in one replaced record. The privacy parameters of differential privacy (DP) are $\varepsilon$ and $\delta$, the failure probability of a "the data lie in this range" statement is $\beta = 0.05$, and errors are root-mean-square errors (RMSE) around the population mean $\mu$. For normal data with variance 1, $\tau = \sqrt{2\ln(2n/\beta)}$ bounds how far the $n$ records lie from $\mu$, and the coarse step's bins have width $b = 2\tau$. For heavy-tailed data with variance at most $\sigma^2$, $\tau_b = \sigma\sqrt{n\varepsilon/(2\sqrt2)}$ is the clipping half-width of @eq-mean-heavy-opt.

## A survey at stronger privacy {#sec-sol-mean-1}

> **Exercise 1.** **A survey at stronger privacy.** Redo the chapter's survey example at $\varepsilon = 0.1$: yes/no answers with $w = 1$ and $\sigma = 0.5$. How many people does an RMSE of 0.05 need, and of 0.01? Compare each with the simple bound of @eq-mean-bounded-n and with the non-private 100 and 2,500. *Check: 338 and 3,138 people; the simple bound gives 383 and 3,914.* *(Sample complexity of a bounded mean)*

::: {.callout-tip collapse="true" title="Solution"}
1. **The quadratic.** By @eq-mean-bounded the squared error is $\sigma^2/n + 2w^2/(n\varepsilon)^2$. Setting it equal to $\alpha^2$ and multiplying by $n^2$ gives $\alpha^2n^2 - \sigma^2n - 2w^2/\varepsilon^2 = 0$, whose positive root is the first form of @eq-mean-bounded-n.
2. **Insert the numbers.** With $\sigma^2 = 0.25$, $w = 1$ and $\varepsilon = 0.1$, the constant term is $2/0.01 = 200$. For $\alpha = 0.05$ the root is $(0.25 + \sqrt{0.0625 + 4\times200\times0.0025})/(2\times0.0025) = 337.2$, so 338 people are needed, and the script checks that their RMSE is 0.0499. For $\alpha = 0.01$ the root is 3,137.5, so 3,138 people are needed, with RMSE 0.0100.
3. **The simple bound.** $\sigma^2/\alpha^2 + \sqrt2\,w/(\alpha\varepsilon)$ gives $100 + 283 = 383$ and $2500 + 1414 = 3914$. It overstates the need because the square root of a sum is smaller than the sum of the square roots.
4. **Compare.** At $\varepsilon = 1$ privacy cost 8 extra people at both targets. At $\varepsilon = 0.1$ it costs 238 and 638. The privacy term scales like $1/\varepsilon$, so it is no longer small next to sampling at the coarser target.

*By the simple bound, the privacy term is smaller than the sampling term only while $\alpha$ is below $\sigma^2\varepsilon/(\sqrt2\,w)$. At strong privacy and a coarse target, the privacy term dominates.*
:::

## The smallest budget for the coarse step {#sec-sol-mean-2}

> **Exercise 2.** **The smallest budget for the coarse step.** With $n = 1000$ normal records and $R = 10^6$, the coarse step has $m = 217{,}223$ bins. Find the smallest $\varepsilon_1$ for which the bound $m e^{-\varepsilon_1 n/4}$ on its failure probability is at most $\beta$, and the RMSE of the fine step when it gets the rest of $\varepsilon = 1$. *Check: $\varepsilon_1 = 0.0611$; the fine RMSE drops from 0.0609 (with $\varepsilon_2 = 0.75$) to 0.0523.* *(Splitting a privacy budget)*

::: {.callout-tip collapse="true" title="Solution"}
1. **Solve the failure bound for $\varepsilon_1$.** Setting $m e^{-\varepsilon_1 n/4} = \beta$ and taking logarithms gives $\varepsilon_1 = 4\ln(m/\beta)/n$, the same equation as @eq-mean-coarse-n solved for $\varepsilon_1$ instead of $n$.
2. **Insert the numbers.** $\ln(217{,}223/0.05) = 15.28$, so $\varepsilon_1 = 4\times15.28/1000 = 0.0611$, and the script confirms that the bound equals 0.050 there.
3. **The fine step.** It gets $\varepsilon_2 = 1 - 0.0611 = 0.9389$. By @eq-mean-coarse its RMSE is $\sqrt{1/n + 18b^2/(n\varepsilon_2)^2}$ with $b = 9.207$, which is 0.0523 against 0.0609 with $\varepsilon_2 = 0.75$.

*The coarse step needs very little budget once $n$ is comfortably larger than $4\ln(m/\beta)$, about 61 here. Sizing it from @eq-mean-coarse-n, rather than with a fixed quarter of the budget, lowers the error by 14%. The split must still be fixed before looking at the data.*
:::

## The Pareto tail {#sec-sol-mean-3}

> **Exercise 3.** **The Pareto tail.** For a Pareto variable with smallest value $x_m$ and shape $a > 1$, show that $\mathbb E[(X - t)_+] = x_m^a t^{1-a}/(a - 1)$ for $t \ge x_m$. For the chapter's records (shape 3, variance 1), evaluate it at $t = \mu + \tau_b$ for $n = 1000$ and $n = 100{,}000$, compare it with the exact bias of the clipped mean, and with the bound $\sigma^2/\tau_b$. *Check: 0.001826 against the bound 0.0532 at $n = 1000$; 0.000021 against 0.0053 at $n = 100{,}000$.* *(Clipping bias)*

::: {.callout-tip collapse="true" title="Solution"}
1. **Write the excess as an integral of tails.** For a variable $Y \ge 0$, $\mathbb E[Y] = \int_0^\infty\Pr[Y > s]\,\mathrm ds$. Applied to $Y = (X - t)_+$, which exceeds $s$ exactly when $X$ exceeds $t + s$, this gives $\mathbb E[(X - t)_+] = \int_t^\infty\Pr[X > v]\,\mathrm dv$.
2. **Insert the Pareto tail.** $\Pr[X > v] = (x_m/v)^a$ for $v \ge x_m$, so the integral is $x_m^a\int_t^\infty v^{-a}\,\mathrm dv = x_m^a t^{1-a}/(a - 1)$, finite because $a > 1$.
3. **Only the top is clipped.** With $\mu = 1.7321$ and $\tau_b = 18.80$ at $n = 1000$, the lower end $\mu - \tau_b$ is below the smallest value $x_m = 1.1547$, so clipping only pulls values down from $\mu + \tau_b$. The bias is minus the excess at $t = \mu + \tau_b$.
4. **The numbers.** The formula gives 0.001826 at $n = 1000$ and 0.000021 at $n = 100{,}000$ ($\tau_b = 188.03$), equal to the exact biases the script computes, with the sign reversed. The bound $\sigma^2/\tau_b$ is 0.0532 and 0.0053.

*For shape 3 the excess falls like $t^{-2}$, while the bound falls like $t^{-1}$, so the bound is 29 times too large at $n = 1000$ and about 250 times at $n = 100{,}000$. That is why the best clipping width for these records is far below $\tau_b$.*
:::

## When does a round shrink the ball? {#sec-sol-mean-4}

> **Exercise 4.** **When does a round shrink the ball?** Take one round of shrinking balls with $\|\mu\| \le R = 1000$, budget $\varepsilon_t = 0.125$ and $\delta_t = 10^{-6}/3$, the margin $u = \sqrt{2\ln(4(T + 1)/\beta)}$ for $T + 1 = 3$ steps, and $\beta = 0.05$. The noise uses the chapter's exact Gaussian ratio $\mu(\varepsilon_t, \delta_t)$, a number and not the mean. Find the smallest $n$ for which @eq-mean-shrink at least halves the radius, for $d = 10$, 100 and 1000, and explain how it grows with $d$. *Check: 821, 1,699 and 4,556 records.* *(Shrinking confidence balls)*

::: {.callout-tip collapse="true" title="Solution"}
1. **The new radius.** By @eq-mean-shrink, $r_1 = (\sqrt d + u)/\sqrt n + \sigma_0(\sqrt d + u)$, with $\sigma_0 = 2(R + \tau_d)/(n\,\mu(\varepsilon_t, \delta_t))$ and $\tau_d = \sqrt d + \sqrt{2\ln(2n/\beta)}$.
2. **It falls with $n$.** The first term falls like $1/\sqrt n$, and $\sigma_0$ falls like $1/n$ apart from the slowly growing $\tau_d$. So the condition $r_1 \le R/2$ holds for all $n$ above a threshold, which the script finds by bisection.
3. **The numbers.** The thresholds are 821, 1,699 and 4,556 records.
4. **The growth.** Near the threshold the noise term dominates, and with $R$ much larger than $\tau_d$ it is about $2R\sqrt d/(n\,\mu)$. Halving $R$ therefore needs $n$ of about $4\sqrt d/\mu(\varepsilon_t, \delta_t)$, which grows like $\sqrt d$. Here $4/\mu(\varepsilon_t, \delta_t) = 126$, and the ratio $n/\sqrt d$ is 260, 170 and 144, falling toward 126 as the margin $u$ matters less next to $\sqrt d$.

*Each round needs about $\sqrt d/\varepsilon_t$ records to shrink the ball, and the rounds reuse the same records. About $\ln R$ rounds are needed, so each gets $\varepsilon_t$ of order $\varepsilon/\ln R$, and that is where the $\sqrt d\ln R/\varepsilon$ term of the chapter comes from.*
:::

## The fingerprinting bound in length {#sec-sol-mean-5}

> **Exercise 5.** **The fingerprinting bound in length.** The fingerprinting bound @eq-lower-fp is for an average error $\alpha'$ per coordinate on $\pm1$ data. A mechanism whose error has expected length at most $\alpha$ has $\alpha' \le \alpha/\sqrt d$. Apply the bound directly, and then after padding to a per-coordinate error $\alpha_0 = 1/6$, which needs $\alpha' \le \alpha_0$, as in [Exercise 4 of the lower-bounds chapter](10-lower-bounds.md#sec-sol-lower-4), and show that the padded bound is about $d/(18\alpha(e^\varepsilon - 1))$. *Check: at $\varepsilon = 1$, $\delta = 10^{-6}$, $\alpha = 1$ and $d = 1000$, the direct bound is 11.1 records and the padded one 32.3.* *(Lower bounds, reductions)*

::: {.callout-tip collapse="true" title="Solution"}
1. **From length to coordinates.** For a vector $v$ in $d$ dimensions, the sum of the absolute values of its coordinates is at most $\sqrt d$ times its length, by the Cauchy–Schwarz inequality applied to $v$ and the all-ones vector. Dividing by $d$, the average error per coordinate is at most $\alpha/\sqrt d$.
2. **Direct.** @eq-lower-fp with $\alpha' = \alpha/\sqrt d$ gives $n \ge (2/3 - 2\alpha/\sqrt d)d/((e^\varepsilon - 1)\sqrt d + 4\delta d)$, about $0.39\sqrt d$ at $\varepsilon = 1$: 11.1 records for $d = 1000$. It grows only like $\sqrt d$, because its numerator $2/3 - 2\alpha'$ is at most $2/3$, so a smaller per-coordinate error earns nothing in it.
3. **Padded.** Padding to $\alpha_0$ multiplies the bound by $\alpha_0/\alpha'$ and uses $\alpha_0$ in place of $\alpha'$, so $n \ge (\alpha_0/\alpha')(2/3 - 2\alpha_0)d/((e^\varepsilon - 1)\sqrt d + 4\delta d)$. With $\alpha_0 = 1/6$ the product $\alpha_0(2/3 - 2\alpha_0) = 1/18$, and with $\alpha' = \alpha/\sqrt d$ and $\delta$ negligible the bound is $(\sqrt d/\alpha)\cdot d/(18(e^\varepsilon - 1)\sqrt d) = d/(18\alpha(e^\varepsilon - 1))$.
4. **The numbers.** At $d = 100$, 1000 and 10,000 the padded bound is 3.2, 32.3 and 323.2 records, against 2.7, 11.1 and 37.6 directly.

*The padded bound grows like $d/(\alpha\varepsilon)$, the privacy term of the shrinking-balls mechanism. Its constant, $1/18$, is far from the mechanism's, but the rate in $d$ is the point.*
:::
