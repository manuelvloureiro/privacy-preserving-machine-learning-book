# Exercises: Rényi differential privacy

Exercises for [Rényi differential privacy](../../book-full/part04-privacy-accounting/01-renyi-dp.md), each with a worked solution. Try each exercise before opening its solution. The script `code/part04-privacy-accounting/solutions_renyi_dp.py` computes every number quoted here; run it with `uv run python code/part04-privacy-accounting/solutions_renyi_dp.py`.

The solutions use the chapter's notation. Rényi differential privacy (RDP) is tracked by the curve $\varepsilon(\alpha)$ of a mechanism $M$; $P = M(D)$ and $Q = M(D')$ are its outputs on neighbours, $r = p/q$ the likelihood ratio and $L = \ln r$ the privacy loss. Differential privacy (DP) without qualifier means $(\varepsilon, \delta)$-DP. The Kullback–Leibler (KL) divergence is the limit of $D_\alpha$ as $\alpha \to 1$. Logarithms are natural, and $\delta = 10^{-5}$ throughout.

## Randomized response at order 2 {#sec-sol-rdp-1}

> **Exercise 1.** **Randomized response at order 2.** Show that $\varepsilon(2) = \ln\big(1 + \chi^2(P\|Q)\big)$ for randomized response, with $\chi^2(P\|Q) = \frac{(2p - 1)^2}{p(1 - p)}$. *Check: 0.8473 for $p = 0.75$ and 0.1542 for $p = 0.6$.* For $p = \frac12 + \gamma$ with small $\gamma$, show $\varepsilon(2) \approx 16\gamma^2$ and KL $\approx 8\gamma^2$. *(Special orders of the Rényi divergence)*

::: {.callout-tip collapse="true" title="Solution"}
1. **Order 2 is the second moment of the likelihood ratio.** By Fact 1 of the chapter at $\alpha = 2$, $e^{D_2(P\|Q)} = \mathbb E_Q[r^2]$. Write $r^2 = 1 + 2(r - 1) + (r - 1)^2$ and use $\mathbb E_Q[r] = 1$: $\mathbb E_Q[r^2] = 1 + \mathbb E_Q[(r - 1)^2] = 1 + \chi^2(P\|Q)$, with $\chi^2 = \sum_y (p(y) - q(y))^2/q(y)$.
2. **The two outputs of randomized response.** From @tbl-rdp-rr, $p(y) - q(y) = \pm(2p - 1)$ on the two outputs, where $p$ is the keep-probability, and $q(y)$ is $1 - p$ on one output and $p$ on the other. So
$$\chi^2 = \frac{(2p-1)^2}{1 - p} + \frac{(2p-1)^2}{p} = \frac{(2p - 1)^2}{p(1-p)}.$$
3. **Check.** For $p = 0.75$: $\chi^2 = 0.25/0.1875 = 1.3333$, and $\ln 2.3333 = 0.8473$, the value of @tbl-rdp-rr-orders. For $p = 0.6$: $\chi^2 = 0.04/0.24 = 0.1667$, and $\ln 1.1667 = 0.1542$.
4. **Small $\gamma$, order 2.** With $p = \frac12 + \gamma$, $2p - 1 = 2\gamma$ and $p(1 - p) = \frac14 - \gamma^2$. So $\chi^2 = \frac{4\gamma^2}{1/4 - \gamma^2} = 16\gamma^2 + O(\gamma^4)$, and $\ln(1 + x) = x + O(x^2)$ gives $\varepsilon(2) = 16\gamma^2 + O(\gamma^4)$.
5. **Small $\gamma$, KL.** The KL divergence is $\mathbb E_P[L] = p\ln\frac{p}{1-p} + (1 - p)\ln\frac{1-p}{p} = (2p - 1)\ln\frac{p}{1-p}$. Here $\ln\frac{1/2 + \gamma}{1/2 - \gamma} = \ln(1 + 2\gamma) - \ln(1 - 2\gamma) = 4\gamma + O(\gamma^3)$. So KL $= 2\gamma \cdot 4\gamma + O(\gamma^4) = 8\gamma^2 + O(\gamma^4)$.
6. **Numerically**, at $p = 0.55$ ($\gamma = 0.05$): $\varepsilon(2) = 0.03961$ against $16\gamma^2 = 0.04000$, and KL $= 0.02007$ against $8\gamma^2 = 0.02000$.

*For a weak mechanism the curve is about $\alpha$ times the KL divergence at moderate orders: $\varepsilon(2) \approx 2\,$KL. A curve that is linear in $\alpha$ is what [Concentrated DP](../../book-full/part04-privacy-accounting/03-concentrated-dp.md) builds on.*
:::

## Composing Laplace mechanisms three ways {#sec-sol-rdp-2}

> **Exercise 2.** **Composing Laplace mechanisms three ways.** Run $k = 100$ Laplace mechanisms with $b = 10$ (each $0.1$-DP) at $\delta = 10^{-5}$. Compute $\varepsilon$ by basic composition, by advanced composition with $\delta' = \delta$, and by RDP with Mironov's conversion over `ORDERS`. *Check: 10, 5.2981 (with the $\tanh$ term; 5.8502 with the textbook term) and 5.0705 at $\alpha = 6.3$.* Explain why RDP wins although each Laplace step is pure DP. *(Composition, conversion)*

::: {.callout-tip collapse="true" title="Solution"}
1. **Basic composition** adds the $\varepsilon_0 = 1/b = 0.1$ of the steps: $100 \times 0.1 = 10$, with $\delta = 0$.
2. **Advanced composition** with $\varepsilon_0 = 0.1$, $\delta_0 = 0$ and $\delta' = 10^{-5}$ gives $\varepsilon_0\sqrt{2k\ln(1/\delta')} + k\varepsilon_0\tanh(\varepsilon_0/2) = 5.2981$. With the textbook mean term $k\varepsilon_0(e^{\varepsilon_0} - 1)$ it is 5.8502.
3. **RDP.** Each step has the Laplace curve @eq-rdp-laplace with $b = 10$. Composition multiplies it by 100 (@eq-rdp-compose). Mironov's conversion (@eq-rdp-mironov), minimized over `ORDERS`, picks $\alpha = 6.3$. There a single step has $\varepsilon(6.3) = 0.02898$, so the composed curve is $2.8983$, and the tail term is $\ln(1/\delta)/5.3 = 2.1723$. The total is 5.0705.
4. **Why RDP wins.** Pure DP charges every step its worst case, $\varepsilon_0 = 0.1$, as if the privacy loss were always at its maximum. At order 6.3 the Laplace curve is 0.02898, less than a third of $\varepsilon_0$: the loss of a Laplace step is usually far below its maximum, and the moderate orders see that. Composition adds these small numbers, and the conversion is paid once.
5. **It does not always win.** For $k = 10$ steps, basic composition gives 1.0000, while RDP gives 1.0090 and advanced composition 1.5674. The RDP order there is 512, the top of the grid, so a longer grid would do a little better (Exercise 5). For $k = 1000$, basic composition gives 100.0000, advanced composition 20.1701 and RDP 19.6905. RDP's advantage over advanced composition is small for pure DP mechanisms; its large gains come from mechanisms like the Gaussian whose $(\varepsilon_0, \delta_0)$ points hide a whole curve (@tbl-rdp-compare). An accountant can always report the smallest of the valid bounds.

*Composing in RDP and converting once is never much worse than the classical theorems for pure DP steps, and it handles Gaussian and Laplace steps in the same ledger.*
:::

## The moments accountant's tail bound {#sec-sol-rdp-3}

> **Exercise 3.** **The moments accountant's tail bound.** Show that Abadi et al.'s bound $\delta = \min_\lambda\exp(K(\lambda) - \lambda\varepsilon)$ is @eq-rdp-mironov solved for $\delta$. For the Gaussian with $\sigma = 1$, find the optimal $\lambda$ for a given $\varepsilon$ in closed form. *Check: at $\varepsilon = 3$, $\lambda^* = 2.5$ and $\delta = 4.394 \times 10^{-2}$.* *(Change of measure, Chernoff)*

::: {.callout-tip collapse="true" title="Solution"}
1. **Rewrite the moment in RDP terms.** By @eq-rdp-moments, $K(\lambda) = \lambda\,\varepsilon(\lambda + 1)$. With $\alpha = \lambda + 1$, the exponent is $K(\lambda) - \lambda\varepsilon = (\alpha - 1)\big(\varepsilon(\alpha) - \varepsilon\big)$.
2. **Solve Mironov's conversion for $\delta$.** At order $\alpha$, @eq-rdp-mironov says $\varepsilon = \varepsilon(\alpha) + \ln(1/\delta)/(\alpha - 1)$. Solving, $\ln\delta = (\alpha - 1)\big(\varepsilon(\alpha) - \varepsilon\big)$, which is the exponent of step 1. So at each order the two bounds are the same pair $(\varepsilon, \delta)$, read in opposite directions.
3. **The two optimizations agree.** Abadi et al. fix $\varepsilon$ and choose $\lambda$ to make $\delta$ smallest; Mironov fixes $\delta$ and chooses $\alpha$ to make $\varepsilon$ smallest. Both search the same family of valid pairs. The script checks a round trip: Mironov's best $\varepsilon$ for $\sigma = 1$ is 5.2985 (at $\alpha = 5.8$), and the tail bound at that $\varepsilon$ returns $\delta = 1.000 \times 10^{-5}$ at $\lambda = 4.8$, the same order.
4. **The Gaussian in closed form.** With $\sigma = 1$, $K(\lambda) = \lambda(\lambda + 1)/2$ (the chapter's Gaussian example with $\mu = 1$). The exponent $\lambda(\lambda + 1)/2 - \lambda\varepsilon$ has derivative $\lambda + \frac12 - \varepsilon$, which vanishes at $\lambda^* = \varepsilon - \frac12$.
5. **Substitute:** $\lambda^*(\lambda^* + 1)/2 - \lambda^*\varepsilon = (\varepsilon - \tfrac12)\big[\tfrac12(\varepsilon + \tfrac12) - \varepsilon\big] = -\tfrac12(\varepsilon - \tfrac12)^2$. So $\delta = \exp\big(-(\varepsilon - \tfrac12)^2/2\big)$.
6. **Check.** At $\varepsilon = 3$: $\lambda^* = 2.5$ and $\delta = e^{-3.125} = 4.394 \times 10^{-2}$; the grid search finds the same $\lambda$, because $\alpha = 3.5$ is on the grid. At $\varepsilon = 4$: $\lambda^* = 3.5$ and $\delta = 2.187 \times 10^{-3}$.

*The bound $\delta = e^{-(\varepsilon - 1/2)^2/2}$ is a Gaussian tail in $\varepsilon$: the Chernoff bound for a normal privacy loss with mean and variance 1/2 and 1.*
:::

## The weak triangle inequality on three Gaussians {#sec-sol-rdp-4}

> **Exercise 4.** **The weak triangle inequality on three Gaussians.** Take $P = \mathcal N(0, 1)$, $R = \mathcal N(0.5, 1)$, $Q = \mathcal N(1.5, 1)$ and $\alpha = 3$. Evaluate both sides of @eq-rdp-weak-triangle for $u = 2$, $u = 1.5$ and $u = 3$. *Check: left side 3.3750; right sides 3.4375, 4.1562 and 3.5000.* Then minimize the right side over $u$ for the shifts $s_1$ from $P$ to $R$ and $s_2$ from $R$ to $Q$. *Check: the best exponent is $u^* = 1 + \frac{s_2(\alpha - 1)}{\alpha s_1} = 2.3333$, where the right side equals the left side, 3.3750.* Explain the equality through the equality case of Hölder's inequality. *(Hölder's inequality)*

::: {.callout-tip collapse="true" title="Solution"}
Unit-variance normals with shift $s$ have $D_\alpha = \alpha s^2/2$ (@eq-gauss-renyi). Here $s_1 = 0.5$, $s_2 = 1$ and the total shift is $s_1 + s_2 = 1.5$.

1. **Left side.** $D_3(P\|Q) = 3 \times 1.5^2/2 = 3.3750$.
2. **$u = 2$, $v = 2$.** The orders are $u\alpha = 6$ and $v(\alpha - 1) + 1 = 5$. The right side is $\frac{3 - 1/2}{2}\cdot\frac{6 \times 0.25}{2} + \frac{5 \times 1}{2} = 1.25 \times 0.75 + 2.5 = 3.4375$.
3. **$u = 1.5$, $v = 3$.** Orders 4.5 and 7: $\frac{3 - 2/3}{2}\cdot\frac{4.5 \times 0.25}{2} + \frac{7}{2} = 4.1562$.
4. **$u = 3$, $v = 1.5$.** Orders 9 and 4: $\frac{3 - 1/3}{2}\cdot\frac{9 \times 0.25}{2} + \frac{4}{2} = 3.5000$.
5. **The right side as a function of $u$.** With $v = u/(u - 1)$,
$$\mathrm{RHS}(u) = \frac{\alpha u - 1}{u(\alpha - 1)}\cdot\frac{u\alpha s_1^2}{2} + \Big(\frac{u}{u-1}(\alpha - 1) + 1\Big)\frac{s_2^2}{2} = \frac{\alpha(\alpha u - 1)s_1^2}{2(\alpha - 1)} + \Big(1 + \frac{(\alpha - 1)u}{u - 1}\Big)\frac{s_2^2}{2}.$$
6. **Minimize.** $\frac{\mathrm d}{\mathrm d u}\frac{u}{u - 1} = -\frac{1}{(u-1)^2}$, so the derivative is $\frac{\alpha^2s_1^2}{2(\alpha - 1)} - \frac{(\alpha - 1)s_2^2}{2(u - 1)^2}$. It vanishes at $(u - 1)^2 = \frac{(\alpha - 1)^2s_2^2}{\alpha^2s_1^2}$, that is, $u^* = 1 + \frac{s_2(\alpha - 1)}{\alpha s_1} = 1 + \frac{2}{1.5} = 2.3333$, with $v^* = 1.75$. The orders become 7 and 4.5, and the right side is 3.3750, equal to the left side.
7. **Why equality is possible.** Hölder's inequality $\mathbb E_R[XY] \le (\mathbb E_RX^u)^{1/u}(\mathbb E_RY^v)^{1/v}$ is an equality exactly when $X^u$ is a constant multiple of $Y^v$. Here $X = (p/r)^\alpha$ and $Y = (r/q)^{\alpha - 1}$. For normals with equal variance, $\ln(p/r)$ and $\ln(r/q)$ are both linear in $y$, with slopes proportional to $-s_1$ and $-s_2$. So $X^u = cY^v$ holds when the slopes match, $u\alpha s_1 = v(\alpha - 1)s_2$. With $v = u/(u - 1)$ this gives $(u - 1)\alpha s_1 = (\alpha - 1)s_2$, the same $u^*$ as step 6.

*The weak triangle inequality loses nothing for Gaussians when $u$ is tuned to the two shifts. The fixed choice $u = 2$ is what group privacy uses, and its loss is small (3.4375 against 3.3750 here).*
:::

## The edge of the order grid {#sec-sol-rdp-5}

> **Exercise 5.** **The edge of the order grid.** For a single Gaussian step at $\delta = 10^{-5}$, find the noise multiplier above which the continuous optimum $\alpha^*$ of @eq-rdp-gauss-opt exceeds the largest order 512. *Check: $\sigma > 106.5$.* At $\sigma = 200$, how much does the grid lose against the continuous optimum? *Check: 0.02893 against 0.02401, a factor 1.205.* *(Calculus, the order grid)*

::: {.callout-tip collapse="true" title="Solution"}
1. **The threshold.** $\alpha^* = 1 + \sigma\sqrt{2\ln(1/\delta)}$ exceeds 512 when $\sigma > 511/\sqrt{2\ln(1/\delta)} = 511/4.7985 = 106.5$.
2. **Just below the edge.** At $\sigma = 100$, $\alpha^* = 480.9$. The grid has no order between 256 and 512, so it uses 512 and gets 0.04813 against the continuous 0.04804, a loss of 0.2%.
3. **Beyond the edge.** At $\sigma = 200$, $\alpha^* = 960.7$, but the largest order is 512. The converted $\varepsilon(\alpha) + \ln(1/\delta)/(\alpha - 1)$ is convex in $\alpha$ and still decreasing at 512, so the grid's best is the last order: 0.02893, against 0.02401 at $\alpha^*$, a factor 1.205.
4. **Remedy.** Add larger orders (1024, 2048, ...) when the noise is large, or optimize $\alpha$ continuously when the curve has a closed form. Both converted values are still well above the exact $\varepsilon = 0.01251$ of this mechanism; that remaining slack is the conversion's, not the grid's.

*A fixed grid is a convenience for adding curves of different mechanisms. Check that the chosen order is not at the edge of the grid; if it is, the grid is too short.*
:::
