# Exercises: Divergences

Exercises for [Divergences](../../book-full/part01-prerequisites/03-divergences.md), each with a worked solution. Try each exercise before opening its solution. The script `code/part01-prerequisites/solutions_divergences.py` computes every number quoted here; run it with `uv run python code/part01-prerequisites/solutions_divergences.py`.

The solutions use the chapter's notation and its three acronyms: the Kullback–Leibler (KL) divergence, the total variation (TV) distance and the data processing inequality (DPI). $r = p/q$ is the likelihood ratio and $L = \ln r$ the log-likelihood ratio, logarithms are natural, and on the die $P$ is the loaded die and $Q$ the fair one (@tbl-div-die).

## Chain rule for KL {#sec-sol-divergences-1}

> **Exercise 1.** **Chain rule for KL.** For joint distributions $P_{XY}$ and $Q_{XY}$, show $D_{\mathrm{KL}}(P_{XY}\|Q_{XY}) = D_{\mathrm{KL}}(P_X\|Q_X) + \mathbb E_{x\sim P_X}\big[D_{\mathrm{KL}}(P_{Y\mid x}\|Q_{Y\mid x})\big]$, and recover additivity as the special case of independent coordinates. *Hint: factor $p(x, y) = p(x)p(y\mid x)$ inside the logarithm.* *(KL, additivity)*

::: {.callout-tip collapse="true" title="Solution"}
1. **Factor both joints.** $p(x, y) = p(x)\,p(y\mid x)$ and $q(x, y) = q(x)\,q(y\mid x)$, where $p(x)$, $q(x)$ are the marginals of $X$.
2. **The log-likelihood ratio splits into a sum** (the logarithm of a product):
$$\ln\frac{p(x, y)}{q(x, y)} = \ln\frac{p(x)}{q(x)} + \ln\frac{p(y\mid x)}{q(y\mid x)}.$$
3. **Average the first term under $P_{XY}$.** It depends only on $x$, so its average is $\sum_xp(x)\ln\frac{p(x)}{q(x)} = D_{\mathrm{KL}}(P_X\|Q_X)$.
4. **Average the second term in two stages:** first over $y$ given $x$, then over $x$. For fixed $x$, $\sum_yp(y\mid x)\ln\frac{p(y\mid x)}{q(y\mid x)} = D_{\mathrm{KL}}(P_{Y\mid x}\|Q_{Y\mid x})$. Averaging over $x \sim P_X$ gives the expected conditional KL.
5. **Add the two averages** (linearity of expectation). This is the chain rule.

**Additivity.** If $X$ and $Y$ are independent under both distributions, then $p(y\mid x) = p_2(y)$ and $q(y\mid x) = q_2(y)$ for every $x$. The conditional KL in step 4 is then $D_{\mathrm{KL}}(P_2\|Q_2)$ for every $x$, and its average over $x$ is the same number. The chain rule becomes @eq-kl-additive.

**Concretely, on the die.** Let $Y$ be the roll and $X$ its parity. Then $(X, Y)$ carries the same information as $Y$, so the left side is $D_{\mathrm{KL}}(P\|Q) = 0.2939$. The parity part is $0.0823$ (from @tbl-div-dpi-die). Given "even", the loaded die gives rolls 2, 4, 6 with probabilities $0.1/0.7, 0.1/0.7, 0.5/0.7 = \frac17, \frac17, \frac57$, and the fair die gives $\frac13$ each. Given "odd", both are uniform on 1, 3, 5.

| Parity $x$ | $p(x)$ | $P_{Y\mid x}$ | $Q_{Y\mid x}$ | $D_{\mathrm{KL}}(P_{Y\mid x}\Vert Q_{Y\mid x})$ |
| --- | --- | --- | --- | --- |
| even | 0.7 | $(\frac17, \frac17, \frac57)$ | $(\frac13, \frac13, \frac13)$ | $\frac27\ln\frac37 + \frac57\ln\frac{15}7 = 0.3023$ |
| odd | 0.3 | $(\frac13, \frac13, \frac13)$ | $(\frac13, \frac13, \frac13)$ | 0 |

: The chain rule on the die, with $X$ the parity and $Y$ the roll. {#tbl-sol-divergences-chain}

The expected conditional KL is $0.7 \times 0.3023 = 0.2116$, and $0.0823 + 0.2116 = 0.2939$, the KL of the raw roll.

*The chain rule says exactly how much the data-processing inequality loses: the KL dropped by keeping only $X$ is the expected KL of what you threw away.*
:::

## Gaussians with different variances {#sec-sol-divergences-2}

> **Exercise 2.** **Gaussians with different variances.** Show $D_{\mathrm{KL}}(\mathcal N(\mu_0, \sigma_0^2)\|\mathcal N(\mu_1, \sigma_1^2)) = \ln\frac{\sigma_1}{\sigma_0} + \frac{\sigma_0^2 + (\mu_0 - \mu_1)^2}{2\sigma_1^2} - \frac12$. *Check: $D_{\mathrm{KL}}(\mathcal N(0, 1)\|\mathcal N(0, 4)) = \ln2 + \frac18 - \frac12 = 0.3181$.* For the Rényi divergence, repeat route 1 of the Gaussian section and find the condition on $\alpha$ under which the integral is finite. *Hint: the coefficient of $x^2$ must stay negative: $\alpha/\sigma_0^2 + (1-\alpha)/\sigma_1^2 > 0$.* *(Completing the square)*

::: {.callout-tip collapse="true" title="Solution"}
Write $P = \mathcal N(\mu_0, \sigma_0^2)$, $Q = \mathcal N(\mu_1, \sigma_1^2)$ and $\Delta = \mu_0 - \mu_1$.

**KL.**

1. **The log-likelihood ratio.** The normal density is $\frac{1}{\sigma\sqrt{2\pi}}e^{-(x - \mu)^2/(2\sigma^2)}$, so
$$L(x) = \ln\frac{\sigma_1}{\sigma_0} - \frac{(x - \mu_0)^2}{2\sigma_0^2} + \frac{(x - \mu_1)^2}{2\sigma_1^2}.$$
Unlike the equal-variance case, $L$ is quadratic in $x$.
2. **Two second moments under $P$.** $\mathbb E_P(X - \mu_0)^2 = \sigma_0^2$. For the other, write $X - \mu_1 = (X - \mu_0) + \Delta$; the cross term averages to 0, so $\mathbb E_P(X - \mu_1)^2 = \sigma_0^2 + \Delta^2$.
3. **Average $L$ under $P$:** $D_{\mathrm{KL}} = \ln\frac{\sigma_1}{\sigma_0} - \frac12 + \frac{\sigma_0^2 + \Delta^2}{2\sigma_1^2}$, the claimed formula.
4. **Check.** For $\mathcal N(0, 1)\|\mathcal N(0, 4)$: $\sigma_0 = 1$, $\sigma_1 = 2$, $\Delta = 0$, so $D_{\mathrm{KL}} = \ln2 + \frac18 - \frac12 = 0.3181$. A numerical integral of $p\ln(p/q)$ gives 0.3181 as well. In the other direction, $D_{\mathrm{KL}}(\mathcal N(0, 4)\|\mathcal N(0, 1)) = -\ln2 + 2 - \frac12 = 0.8069$: KL is not symmetric here. With $\sigma_0 = \sigma_1$ the formula reduces to $\Delta^2/(2\sigma^2)$, as in @eq-gauss-renyi.

**Rényi, by route 1.**

1. **Constants.** $p^\alpha q^{1-\alpha}$ has the constant factor $\frac{1}{\sqrt{2\pi}}\sigma_0^{-\alpha}\sigma_1^{-(1-\alpha)}$ times $\exp\big(-\frac12B(x)\big)$, with
$$B(x) = w_0(x - \mu_0)^2 + w_1(x - \mu_1)^2, \qquad w_0 = \frac{\alpha}{\sigma_0^2},\quad w_1 = \frac{1 - \alpha}{\sigma_1^2}.$$
2. **Finiteness.** The coefficient of $x^2$ in $-\frac12B$ is $-\frac12(w_0 + w_1)$. If $w_0 + w_1 \le 0$, the integrand does not decay in at least one direction (it grows like $e^{cx^2}$, or, when the coefficient is exactly 0, like $e^{cx}$ or a constant), and the integral is infinite. So the integral is finite exactly when
$$w_0 + w_1 = \frac{\alpha}{\sigma_0^2} + \frac{1 - \alpha}{\sigma_1^2} > 0 \iff \sigma_\alpha^2 := \alpha\sigma_1^2 + (1 - \alpha)\sigma_0^2 > 0$$
(multiply by $\sigma_0^2\sigma_1^2 > 0$). Note $w_0 + w_1 = \sigma_\alpha^2/(\sigma_0^2\sigma_1^2)$.
3. **Complete the square.** For weights with $w_0 + w_1 > 0$, $w_0(x - \mu_0)^2 + w_1(x - \mu_1)^2 = (w_0 + w_1)(x - m)^2 + \frac{w_0w_1}{w_0 + w_1}\Delta^2$ with $m = \frac{w_0\mu_0 + w_1\mu_1}{w_0 + w_1}$. (Expand both sides: the $x^2$ and $x$ terms match by the choice of $m$, and the constant is what is left.) Here $\frac{w_0w_1}{w_0 + w_1} = \frac{\alpha(1-\alpha)}{\sigma_0^2\sigma_1^2}\cdot\frac{\sigma_0^2\sigma_1^2}{\sigma_\alpha^2} = \frac{\alpha(1 - \alpha)}{\sigma_\alpha^2}$.
4. **Integrate.** $\int e^{-(w_0 + w_1)(x - m)^2/2}dx = \sqrt{2\pi/(w_0 + w_1)} = \sqrt{2\pi}\,\sigma_0\sigma_1/\sigma_\alpha$. So
$$\int p^\alpha q^{1-\alpha}\,dx = \frac{\sigma_0^{1-\alpha}\sigma_1^{\alpha}}{\sigma_\alpha}\exp\left(-\frac{\alpha(1-\alpha)\Delta^2}{2\sigma_\alpha^2}\right).$$
5. **Take the logarithm and divide by $\alpha - 1$.** The exponent gives $\frac{\alpha\Delta^2}{2\sigma_\alpha^2}$, as in the equal-variance case. The prefactor gives $\frac{(1-\alpha)\ln\sigma_0 + \alpha\ln\sigma_1 - \ln\sigma_\alpha}{\alpha - 1} = \ln\frac{\sigma_1}{\sigma_0} + \frac{\ln\sigma_1 - \ln\sigma_\alpha}{\alpha - 1}$. Together,
$$D_\alpha(P\|Q) = \ln\frac{\sigma_1}{\sigma_0} + \frac{1}{2(\alpha - 1)}\ln\frac{\sigma_1^2}{\sigma_\alpha^2} + \frac{\alpha\Delta^2}{2\sigma_\alpha^2} \quad\text{if } \sigma_\alpha^2 > 0, \qquad D_\alpha = \infty \text{ otherwise}.$$

**When is it finite?** For $0 < \alpha < 1$, $\sigma_\alpha^2$ is a weighted average of two positive numbers, so always positive. For $\alpha > 1$, $\sigma_\alpha^2 = \sigma_0^2 + \alpha(\sigma_1^2 - \sigma_0^2)$. If $\sigma_1 \ge \sigma_0$ (the second distribution is at least as wide), it is positive for every $\alpha$. If $\sigma_1 < \sigma_0$, it is positive only for
$$\alpha < \frac{\sigma_0^2}{\sigma_0^2 - \sigma_1^2}.$$
The reason: $r^{\alpha-1}$ grows like $e^{cx^2}$ in the tails when $Q$ is the narrower one, and for large $\alpha$ that beats the decay of $p$.

**Checks.** As $\alpha \to 1$, $\sigma_\alpha^2 \to \sigma_1^2$ and $\frac{1}{\alpha - 1}\ln\frac{\sigma_1^2}{\sigma_\alpha^2} \to -\frac{\sigma_1^2 - \sigma_0^2}{\sigma_1^2}$ (the derivative of $-\ln\sigma_\alpha^2$ at $\alpha = 1$), so $D_\alpha \to \ln\frac{\sigma_1}{\sigma_0} + \frac{\sigma_0^2}{2\sigma_1^2} - \frac12 + \frac{\Delta^2}{2\sigma_1^2}$, the KL above. With $\sigma_0 = \sigma_1$, $\sigma_\alpha = \sigma$ and only $\alpha\Delta^2/(2\sigma^2)$ remains, which is @eq-gauss-renyi. @tbl-sol-divergences-gauss compares the formula with a numerical integral.

| Pair $P\Vert Q$ | Order | Formula | Numerical integral |
| --- | --- | --- | --- |
| $\mathcal N(0, 1)\Vert\mathcal N(0, 4)$ | $\alpha = 2$, $\sigma_2^2 = 7$ | $\ln2 + \frac12\ln\frac47 = 0.4133$ | 0.4133 |
| $\mathcal N(1, 1)\Vert\mathcal N(0, 4)$ | $\alpha = 2$, $\sigma_2^2 = 7$ | $0.4133 + \frac{2}{14} = 0.5562$ | 0.5562 |
| $\mathcal N(0, 4)\Vert\mathcal N(0, 1)$ | $\alpha = 1.2$, $\sigma_{1.2}^2 = 0.4$ | $-\ln2 + \frac{1}{0.4}\ln\frac{1}{0.4} = 1.5976$ | 1.5976 |
| $\mathcal N(0, 4)\Vert\mathcal N(0, 1)$ | $\alpha = 2$, $\sigma_2^2 = -2$ | $\infty$ | — |

: Rényi divergences between Gaussians with different variances, from the closed form and by numerical integration. {#tbl-sol-divergences-gauss}

For $\mathcal N(0, 4)\|\mathcal N(0, 1)$ the limit is $\alpha < \frac{4}{4 - 1} = 1.3333$: every order $\alpha \ge \frac43$ is infinite, although the KL (order 1) is only 0.8069.

*A wider $P$ against a narrower $Q$ makes the likelihood ratio grow like $e^{cx^2}$, and then only low Rényi orders are finite.*
:::

## Bhattacharyya and Hellinger {#sec-sol-divergences-3}

> **Exercise 3.** **Bhattacharyya and Hellinger.** Show $D_{1/2}(P\|Q) = -2\ln\big(1 - \frac12H^2\big)$ with $H^2 = \sum(\sqrt p - \sqrt q)^2$, and check it on the die using @tbl-div-f and @tbl-div-die-renyi. For the Gaussian pair, show $D_{1/2} = \mu^2/4$. *(Rényi, f-divergences)*

::: {.callout-tip collapse="true" title="Solution"}
1. **$D_{1/2}$ in terms of the Bhattacharyya coefficient.** Put $\alpha = \frac12$ in the definition: $D_{1/2} = \frac{1}{-1/2}\ln\sum_xp^{1/2}q^{1/2} = -2\ln\mathrm{BC}$, with $\mathrm{BC} = \sum_x\sqrt{p(x)q(x)}$.
2. **Expand the Hellinger sum.** $(\sqrt p - \sqrt q)^2 = p - 2\sqrt{pq} + q$. Sum over $x$: $p$ and $q$ each sum to 1, so $H^2 = 2 - 2\,\mathrm{BC}$.
3. **Solve for $\mathrm{BC}$:** $\mathrm{BC} = 1 - \frac12H^2$. Substitute into step 1: $D_{1/2} = -2\ln\big(1 - \frac12H^2\big)$.

**On the die.** @tbl-div-f gives $H^2 = 0.1317$. Then $\mathrm{BC} = 1 - 0.06585 = 0.9342$ and $-2\ln0.9342 = 0.1362$, the $\alpha = 0.5$ entry of @tbl-div-die-renyi in both directions.

**For the Gaussian pair.** $P = \mathcal N(0, \sigma^2)$, $Q = \mathcal N(s, \sigma^2)$, $\mu = s/\sigma$.

1. Step 4 of route 1 in the Gaussian section, at $\alpha = \frac12$, gives $\mathrm{BC} = \int\sqrt{pq}\,dx = \exp\big(-\frac{\frac12\cdot\frac12 s^2}{2\sigma^2}\big) = e^{-\mu^2/8}$.
2. So $D_{1/2} = -2\ln e^{-\mu^2/8} = \mu^2/4$. This is @eq-gauss-renyi at $\alpha = \frac12$.
3. As a by-product, $H^2 = 2\big(1 - e^{-\mu^2/8}\big)$.

At $\mu = 1/\sqrt2$ (the Gaussian pair with the variance of the Laplace pair at $t = 1$) this gives $D_{1/2} = 0.1250$, the entry in @tbl-div-laplace-gauss.

*Rényi divergence of order $\frac12$ and squared Hellinger distance carry the same information: each is an increasing function of the other.*
:::

## Rejection sampling {#sec-sol-divergences-4}

> **Exercise 4.** **Rejection sampling.** In the rejection-sampling example, show in general that $\mathrm{TV}(P, M) \le E_\gamma(P\|Q)$. *Hint: write $\min(p, \gamma q) = p - e$ with $e = (p - \gamma q)_+$, so $m = (p - e)/(1 - E_\gamma)$, and use $e \le p$.* *(Hockey-stick)*

::: {.callout-tip collapse="true" title="Solution"}
Write $E = E_\gamma(P\|Q)$ and assume $E < 1$, so that some proposals are accepted.

1. **The uncovered part.** $\min(p, \gamma q) = p - (p - \gamma q)_+ = p - e$. (Where $p \le \gamma q$, $e = 0$ and the min is $p$; where $p > \gamma q$, the min is $\gamma q = p - e$.)
2. **Normalize.** $\sum_xe(x) = E$ by the definition of $E_\gamma$, so $\sum_x\min(p, \gamma q) = 1 - E$ and $m = (p - e)/(1 - E)$.
3. **The difference $p - m$.** $p - m = \frac{p(1 - E) - p + e}{1 - E} = \frac{e - pE}{1 - E}$.
4. **Use $e \le p$.** Since $e \le p$ and $E \ge 0$, $e - pE \le e - eE = e(1 - E)$. The right side is non-negative, so $(e - pE)_+ \le e(1 - E)$.
5. **Sum, using the form $\mathrm{TV} = \sum(p - m)_+$ of @eq-tv:**
$$\mathrm{TV}(P, M) = \frac{1}{1 - E}\sum_x(e - pE)_+ \le \frac{1}{1 - E}\sum_xe(1 - E) = E.$$

**On the die.** @tbl-sol-divergences-reject runs the envelope $\gamma\cdot$fair over the range $1 \le \gamma \le 3$. The bound holds everywhere, and both sides reach 0 at $\gamma = 3 = \max r$.

| $\gamma$ | 1 | 1.5 | 2 | 2.5 | 3 |
| --- | --- | --- | --- | --- | --- |
| $m(6)$ | 0.2500 | 0.3333 | 0.4000 | 0.4545 | 0.5000 |
| $\mathrm{TV}(P, M)$ | 0.2500 | 0.1667 | 0.1000 | 0.0455 | 0.0000 |
| $E_\gamma(P\Vert Q)$ | 0.3333 | 0.2500 | 0.1667 | 0.0833 | 0.0000 |

: Rejection sampling from the loaded die with a fair-die proposal and envelope $\gamma q$: the output's distance from the target, against the hockey-stick divergence. {#tbl-sol-divergences-reject}

The bound is loose by the factor $1 - E$ lost in step 4: the deficit $e$ is not all removed from $m$, because renormalizing gives part of it back.

*A low envelope costs at most $E_\gamma$ in total variation, so $E_\gamma$ tells you how low an envelope you can afford.*
:::

## DPI for $E_\gamma$ directly {#sec-sol-divergences-5}

> **Exercise 5.** **DPI for $E_\gamma$ directly.** Without the f-divergence machinery, show $E_\gamma(PK\|QK) \le E_\gamma(P\|Q)$. *Hint: $PK(B) - \gamma QK(B) = \sum_xK(B\mid x)\big(p(x) - \gamma q(x)\big)$ with $K(B\mid x) \in [0, 1]$; use the randomized version of Fact 3.* *(Fact 3, DPI)*

::: {.callout-tip collapse="true" title="Solution"}
This is the data-processing inequality of @sec-div-dpi, for $E_\gamma$ alone.

1. **Output probabilities of an event.** For an output event $B$, $PK(B) = \sum_{y\in B}\sum_xp(x)K(y\mid x) = \sum_xp(x)K(B\mid x)$, where $K(B\mid x) = \sum_{y\in B}K(y\mid x)$ is the probability that input $x$ lands in $B$. The same holds for $Q$. Subtract:
$$PK(B) - \gamma QK(B) = \sum_xK(B\mid x)\big(p(x) - \gamma q(x)\big).$$
2. **This is a randomized test on the input.** $\phi(x) = K(B\mid x)$ takes values in $[0, 1]$. By the second part of Fact 3, $\sum_x\phi(x)(p(x) - \gamma q(x)) \le \sum_x(p(x) - \gamma q(x))_+ = E_\gamma(P\|Q)$.
3. **Maximize over $B$.** Step 2 holds for every $B$. By Fact 3 on the output space, $E_\gamma(PK\|QK) = \max_B[PK(B) - \gamma QK(B)]$, so it is at most $E_\gamma(P\|Q)$.

**On the die, reporting the parity** ($PK = (0.7, 0.3)$, $QK = (0.5, 0.5)$ for even, odd):

- $\gamma = 1$: $E_1 = (0.7 - 0.5)_+ = 0.2000$ after processing, against TV $= 0.3333$ before.
- $\gamma = 1.2$: $(0.7 - 0.6)_+ = 0.1000$, against $0.5 - 1.2/6 = 0.3000$.
- $\gamma = 2$: $(0.7 - 1)_+ = 0$, against $0.1667$. The parity bit has likelihood ratios 1.4 and 0.6, so its $E_\gamma$ is 0 for every $\gamma \ge 1.4$.

*Any processing only lowers the whole $E_\gamma$ curve, so every statement made through the $E_\gamma$ curves survives processing.*
:::

## Pinsker's constant is sharp {#sec-sol-divergences-6}

> **Exercise 6.** **Pinsker's constant is sharp.** For $P = \mathrm{Bernoulli}(\frac12 + \delta)$ and $Q = \mathrm{Bernoulli}(\frac12)$, show $\mathrm{kl}(\frac12 + \delta, \frac12)/(2\delta^2) \to 1$ as $\delta \to 0$. *Hint: expand $\ln(1 \pm 2\delta)$ to second order.* *(Pinsker)*

::: {.callout-tip collapse="true" title="Solution"}
1. **Write out kl.** $\mathrm{kl}(\frac12 + \delta, \frac12) = (\frac12 + \delta)\ln\frac{1/2 + \delta}{1/2} + (\frac12 - \delta)\ln\frac{1/2 - \delta}{1/2} = (\frac12 + \delta)\ln(1 + 2\delta) + (\frac12 - \delta)\ln(1 - 2\delta)$.
2. **Expand the logarithms.** $\ln(1 + u) = u - \frac{u^2}2 + O(u^3)$, so $\ln(1 \pm 2\delta) = \pm2\delta - 2\delta^2 + O(\delta^3)$.
3. **Multiply out**, keeping terms up to $\delta^2$:
$$(\tfrac12 + \delta)(2\delta - 2\delta^2) = \delta - \delta^2 + 2\delta^2 + O(\delta^3), \qquad (\tfrac12 - \delta)(-2\delta - 2\delta^2) = -\delta - \delta^2 + 2\delta^2 + O(\delta^3).$$
4. **Add.** The $\delta$ terms cancel and the $\delta^2$ terms give $2\delta^2$: $\mathrm{kl}(\frac12 + \delta, \frac12) = 2\delta^2 + O(\delta^3)$. Divide by $2\delta^2$: the ratio is $1 + O(\delta) \to 1$.

**Why this makes the constant sharp.** Here $\mathrm{TV}(P, Q) = \delta$ (one outcome gains $\delta$, the other loses it). Pinsker says $\mathrm{KL} \ge 2\,\mathrm{TV}^2 = 2\delta^2$. If some constant $c > 2$ worked, $\mathrm{KL} \ge c\,\mathrm{TV}^2$, the ratio $\mathrm{kl}/(2\delta^2)$ would stay at least $c/2 > 1$ for all $\delta$, which contradicts the limit 1.

**How fast.** kl is even in $\delta$ (swapping heads and tails maps $\delta$ to $-\delta$), so the $O(\delta^3)$ term is in fact zero; carrying the expansion one order further gives the ratio $1 + \frac23\delta^2 + O(\delta^4)$. @tbl-sol-divergences-pinsker shows it.

| $\delta$ | 0.2 | 0.1 | 0.01 |
| --- | --- | --- | --- |
| $\mathrm{kl}(\frac12 + \delta, \frac12)/(2\delta^2)$ | 1.0285 | 1.0068 | 1.0001 |
| $1 + \frac23\delta^2$ | 1.0267 | 1.0067 | 1.0001 |

: The ratio of the binary KL to Pinsker's lower bound $2\delta^2$, for a coin with bias $\frac12 + \delta$ against a fair coin. {#tbl-sol-divergences-pinsker}

*Near two fair coins, Pinsker's inequality is an equality to first order, so its constant cannot be improved.*
:::

## Samples needed for two coins {#sec-sol-divergences-7}

> **Exercise 7.** **Samples needed for two coins.** You flip a coin that is either fair or has heads probability 0.6. Use additivity and Pinsker to find how many flips $n$ are necessary before the TV between $n$ flips of the two coins can reach 0.8. *Check: $\mathrm{kl}(0.6, 0.5) = 0.02014$, so $n \ge 2\times0.8^2/0.02014$, that is $n \ge 64$.* Is this a lower or an upper bound on the flips you need? *(Additivity, Pinsker)*

::: {.callout-tip collapse="true" title="Solution"}
1. **One flip.** $\mathrm{kl}(0.6, 0.5) = 0.6\ln\frac{0.6}{0.5} + 0.4\ln\frac{0.4}{0.5} = 0.6\ln1.2 + 0.4\ln0.8 = 0.02014$.
2. **$n$ flips, by additivity** (@eq-kl-additive): $\mathrm{KL}_n = 0.02014\,n$.
3. **Pinsker** (@eq-pinsker): $\mathrm{TV}_n \le \sqrt{\mathrm{KL}_n/2} = \sqrt{0.02014\,n/2}$.
4. **Require TV to reach 0.8.** That needs $\sqrt{0.02014\,n/2} \ge 0.8$, that is $n \ge 2\times0.8^2/0.02014 = 63.6$. Since $n$ is an integer, $n \ge 64$. The check value is right.

**Lower or upper bound?** A **lower bound**. Pinsker caps the TV, so with fewer than 64 flips *no* test can reach TV 0.8. It does not say 64 flips are enough. The exact TV (computed on the number of heads, which is a sufficient statistic, so by the equality case of the DPI it has the same TV as the whole sequence) is only 0.5805 at $n = 64$, and first reaches 0.8 at $n = 162$. Pinsker is off by a factor of about 2.5 here, because it is tight only near TV $= 0$. Bretagnolle–Huber (@eq-bh) gives the weaker requirement $n \ge \ln\frac{1}{1 - 0.64}/0.02014$, that is $n \ge 51$: the KL that Pinsker needs for TV $= 0.8$ is $2\times0.8^2 = 1.28$, below the crossover 1.59 of @sec-div-pinsker, so Pinsker is the stronger of the two here.

*Pinsker turns "KL is small" into "no test can succeed yet", which is always a statement about necessary, not sufficient, sample sizes.*
:::

## Monotonicity of TV over samples {#sec-sol-divergences-8}

> **Exercise 8.** **Monotonicity of TV over samples.** Show that $\mathrm{TV}(P^n, Q^n)$ is non-decreasing in $n$. *Hint: dropping the last sample is a channel.* *(DPI)*

::: {.callout-tip collapse="true" title="Solution"}
1. **The channel.** Let $K$ map $(x_1, \dots, x_{n+1})$ to $(x_1, \dots, x_n)$ with probability 1: $K(y\mid x) = \mathbf 1[y = (x_1, \dots, x_n)]$. It is a deterministic function, so a channel.
2. **It maps $P^{n+1}$ to $P^n$.** $P^{n+1}K(x_1, \dots, x_n) = \sum_{x_{n+1}}p(x_1)\cdots p(x_n)\,p(x_{n+1}) = p(x_1)\cdots p(x_n)$, since $p$ sums to 1. The same for $Q$.
3. **Apply the DPI.** TV is the f-divergence with $f(t) = \frac12\lvert t - 1\rvert$ (@tbl-div-f), so @sec-div-dpi gives $\mathrm{TV}(P^n, Q^n) = \mathrm{TV}(P^{n+1}K, Q^{n+1}K) \le \mathrm{TV}(P^{n+1}, Q^{n+1})$.

You can also see it directly from the max form of @eq-tv: any event about the first $n$ samples is also an event about the first $n + 1$ (it ignores the last one), so the maximum over the larger family of events is at least as large.

**Concretely, with the coins of Exercise 7.** One flip: TV $= 0.6 - 0.5 = 0.1$. Two flips:

| Outcome | $HH$ | $HT$ | $TH$ | $TT$ | Sum |
| --- | --- | --- | --- | --- | --- |
| $p$ (heads 0.6) | 0.36 | 0.24 | 0.24 | 0.16 | 1 |
| $q$ (fair) | 0.25 | 0.25 | 0.25 | 0.25 | 1 |
| $(p - q)_+$ | 0.11 | 0 | 0 | 0 | 0.11 |

: Total variation between two flips of a coin with heads probability 0.6 and two flips of a fair coin. {#tbl-sol-divergences-two-flips}

So TV rises from 0.1000 to 0.1100. Further values are 0.1480 at $n = 3$, 0.1627 at $n = 4$, 0.3143 at $n = 16$, 0.5805 at $n = 64$ and 0.8930 at $n = 256$. The growth is slow and uneven (only 0.01 from one flip to two), but it never goes down.

*More independent samples can never make two distributions harder to tell apart, because you can always ignore the extra samples.*
:::
