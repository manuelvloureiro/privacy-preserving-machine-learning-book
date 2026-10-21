# Exercises: Advanced and optimal composition

Exercises for [Advanced and optimal composition](../../book-full/part03-dp-foundations/04-composition.md), each with a worked solution. Try each exercise before opening its solution. The script `code/part03-dp-foundations/solutions_composition.py` computes every number quoted here; run it with `uv run python code/part03-dp-foundations/solutions_composition.py`.

The solutions use the chapter's notation and its abbreviation for differential privacy (DP). $P = M(D)$ and $Q = M(D')$, $L = \ln(p/q)$ is the privacy loss with $Y \sim P$, $\delta(\varepsilon) = \mathbb E[(1 - e^{\varepsilon - L})_+]$, and randomized response at level $\varepsilon$ is truthful with probability $t = e^\varepsilon/(1 + e^\varepsilon)$.

## The sharp mean loss {#sec-sol-composition-1}

> **Exercise 1.** **The sharp mean loss.** Show that $\varepsilon\tanh(\varepsilon/2) \le \varepsilon^2/2$ and $\varepsilon\tanh(\varepsilon/2) \le \varepsilon(e^\varepsilon - 1)$ for every $\varepsilon \ge 0$, and that the ratio of the two mean bounds tends to $\frac12$ as $\varepsilon \to 0$. *Check: at $\varepsilon = 1$ the three values are 0.4621, 0.5 and 1.7183.* *(Fact 3, Taylor expansion)*

::: {.callout-tip collapse="true" title="Solution"}
1. **$\tanh u \le u$ for $u \ge 0$.** Both sides are 0 at $u = 0$, and the derivative of $\tanh$ is $1 - \tanh^2 u \le 1$, the derivative of $u$. So $\tanh u$ never catches up. With $u = \varepsilon/2$: $\varepsilon\tanh(\varepsilon/2) \le \varepsilon\cdot\varepsilon/2 = \varepsilon^2/2$.
2. **Against $\varepsilon(e^\varepsilon - 1)$.** Write $\tanh(\varepsilon/2) = \frac{e^{\varepsilon/2} - e^{-\varepsilon/2}}{e^{\varepsilon/2} + e^{-\varepsilon/2}} = \frac{e^\varepsilon - 1}{e^\varepsilon + 1}$ (multiply numerator and denominator by $e^{\varepsilon/2}$). The denominator $e^\varepsilon + 1$ is at least 2, so $\tanh(\varepsilon/2) \le \frac12(e^\varepsilon - 1) \le e^\varepsilon - 1$.
3. **The ratio.** By step 2, $\frac{\varepsilon\tanh(\varepsilon/2)}{\varepsilon(e^\varepsilon - 1)} = \frac{1}{e^\varepsilon + 1}$, which tends to $\frac12$ as $\varepsilon \to 0$ and decreases as $\varepsilon$ grows.
4. **Check.** @tbl-sol-composition-means lists the three values.

| $\varepsilon$ | $\varepsilon\tanh(\varepsilon/2)$ | $\varepsilon^2/2$ | $\varepsilon(e^\varepsilon - 1)$ | Ratio $1/(e^\varepsilon + 1)$ |
| --- | --- | --- | --- | --- |
| 0.01 | 0.000050 | 0.000050 | 0.000101 | 0.4975 |
| 0.1 | 0.004996 | 0.005000 | 0.010517 | 0.4750 |
| 0.5 | 0.122459 | 0.125000 | 0.324361 | 0.3775 |
| 1 | 0.462117 | 0.500000 | 1.718282 | 0.2689 |
| 2 | 1.523188 | 2.000000 | 12.778112 | 0.1192 |

: Three bounds on the mean privacy loss of an $\varepsilon$-DP step. {#tbl-sol-composition-means}

*For small $\varepsilon$ the sharp bound and $\varepsilon^2/2$ agree, and the Dwork–Rothblum–Vadhan bound is twice as large. For large $\varepsilon$ the gap explodes, which is why the $\tanh$ form of advanced composition matters most when $k\varepsilon^2$ is large.*
:::

## When does advanced composition pay? {#sec-sol-composition-2}

> **Exercise 2.** **When does advanced composition pay?** Show that @eq-comp-advanced is below $k\varepsilon$ exactly when $k > 2\ln(1/\delta')/(2 - e^\varepsilon)^2$, for $\varepsilon < \ln 2$. *Check: for $\varepsilon = 0.1$ and $\delta' = 10^{-5}$ the threshold is 28.8, so advanced composition wins from $k = 29$.* *(Advanced composition)*

::: {.callout-tip collapse="true" title="Solution"}
1. **Write the inequality.** $\varepsilon\sqrt{2k\ln(1/\delta')} + k\varepsilon(e^\varepsilon - 1) < k\varepsilon$.
2. **Divide by $\varepsilon > 0$ and collect the $k$ terms:** $\sqrt{2k\ln(1/\delta')} < k(2 - e^\varepsilon)$.
3. **Sign.** For $\varepsilon < \ln 2$, $2 - e^\varepsilon > 0$, so both sides are positive and squaring keeps the inequality: $2k\ln(1/\delta') < k^2(2 - e^\varepsilon)^2$.
4. **Divide by $k(2 - e^\varepsilon)^2 > 0$:** $k > 2\ln(1/\delta')/(2 - e^\varepsilon)^2$. (For $\varepsilon \ge \ln 2$ the right side of step 2 is not positive, and advanced composition never wins.)
5. **Check.** $2\ln 10^5 = 23.03$ and $(2 - e^{0.1})^2 = 0.8948^2 = 0.8007$, so the threshold is 28.8. The script confirms: advanced composition gives 2.8336 at $k = 28$ (above $2.8$) and 2.8891 at $k = 29$ (below $2.9$).

With the $\tanh$ mean term, the same steps give $k > 2\ln(1/\delta')/(1 - \tanh(\varepsilon/2))^2$, which is 25.5 at $\varepsilon = 0.1$, so that form wins from $k = 26$. At $\varepsilon = 0.5$ the original form needs $k > 186.6$.

*Advanced composition is a large-$k$ tool: below about $2\ln(1/\delta')$ steps, the $\delta'$ it spends buys nothing.*
:::

## Two rounds by hand {#sec-sol-composition-3}

> **Exercise 3.** **Two rounds by hand.** From @tbl-comp-two-rr, compute $\delta(\varepsilon')$ for two rounds of randomized response at $\varepsilon = \ln 3$, for all $\varepsilon' \in [0, 2\ln 3]$. *Check: $\delta(0) = \frac12$, $\delta(\ln 3) = \frac38$, $\delta(2\ln 3) = 0$.* Compare with what basic composition says. *(Fact 1, optimal composition)*

::: {.callout-tip collapse="true" title="Solution"}
1. **Which outputs count.** By Fact 1, $\delta(\varepsilon') = \sum_y p(y)(1 - e^{\varepsilon' - L(y)})_+$, and only outputs with $L(y) > \varepsilon'$ contribute. For $\varepsilon' \ge 0$, the outputs with $L = 0$ and $L = -2\ln 3$ never count. Only $(0, 0)$, with $p = \frac9{16}$ and $L = 2\ln 3$, counts, and only while $\varepsilon' < 2\ln 3$.
2. **The curve.** $\delta(\varepsilon') = \frac9{16}\big(1 - e^{\varepsilon'}/9\big) = \frac{9 - e^{\varepsilon'}}{16}$ for $0 \le \varepsilon' \le 2\ln 3$, and 0 beyond.
3. **Check.** $\varepsilon' = 0$: $\frac{8}{16} = \frac12$. $\varepsilon' = \ln 3$: $\frac{6}{16} = \frac38$. $\varepsilon' = 2\ln 3$: $0$. Enumeration of the four outputs and @eq-comp-kov give the same three values.
4. **Compared with basic composition.** Basic composition says $(2\ln 3, 0)$, the end point of this curve, and is exact there. For two rounds there is nothing to gain at $\delta = 0$. What the curve adds is every other point: for example, the pair is $(\ln 3, \frac38)$-DP. A $\delta$ of $\frac38$ is useless for privacy, which is the point: with $k = 2$ the loss has not had the chance to concentrate.

*Composition gains come from many steps. With two steps the worst output has probability $\frac9{16}$, and you cannot trade it for a small $\delta$.*
:::

## The per-step δ {#sec-sol-composition-4}

> **Exercise 4.** **The per-step $\delta$.** Prove the identity $1 - (1 - \delta)^k \le k\delta$ used for advanced composition with $\delta > 0$, and show that @eq-comp-kov is non-trivial only for a target $\delta_{\text{total}} > 1 - (1 - \delta)^k$. *Check: with $\varepsilon = 0.1$, $\delta = 5\times10^{-8}$, $k = 100$ and $\delta_{\text{total}} = 10^{-5}$, the optimal $\varepsilon'$ is 4.4469 and advanced composition gives 5.9926.* *(Optimal composition)*

::: {.callout-tip collapse="true" title="Solution"}
1. **The inequality is a union bound.** $1 - (1 - \delta)^k$ is the probability that at least one of $k$ independent events of probability $\delta$ happens: some step shows its flag. The union bound gives at most $k\delta$. (Or by induction: $(1 - \delta)^{k} \ge (1 - \delta)^{k-1} - \delta \ge 1 - (k - 1)\delta - \delta$.)
2. **The floor of @eq-comp-kov.** $\delta_{\mathrm{RR}}(\varepsilon') \ge 0$, so the right side of @eq-comp-kov is at least $1 - (1 - \delta)^k$ for every $\varepsilon'$. This floor is attained by the composed four-point pair: the flags have infinite loss. So no finite $\varepsilon'$ achieves a smaller total $\delta$, and the script returns $\varepsilon' = \infty$ for a target of $4\times10^{-6}$.
3. **Above the floor.** For a target $\delta_{\text{total}} > 1 - (1 - \delta)^k$, solve $\delta_{\mathrm{RR}}(\varepsilon') = \frac{\delta_{\text{total}} - (1 - (1 - \delta)^k)}{(1 - \delta)^k}$, a pure randomized-response problem.
4. **Check.** $1 - (1 - 5\times10^{-8})^{100} = 4.999988\times10^{-6}$, just below $k\delta = 5\times10^{-6}$. That leaves about $5\times10^{-6}$ for the randomized-response part, and the optimal $\varepsilon'$ is 4.4469. Advanced composition with $\delta' = 10^{-5} - 100 \times 5\times10^{-8} = 5\times10^{-6}$ gives 5.9926, and its $\tanh$ form 5.4404.

*The per-step $\delta$ is spent outright, step by step; only what is left can be traded against $\varepsilon'$.*
:::

## Which pairs factor through randomized response? {#sec-sol-composition-5}

> **Exercise 5.** **Which pairs factor through randomized response?** For $P = (0.5, 0.3, 0.2)$ and $Q = (0.2, 0.3, 0.5)$, find the smallest $\varepsilon$ for which @eq-comp-channel is a valid channel, and write the channel. *Hint: a row of $K$ goes negative exactly where the pair violates $\varepsilon$-DP.* *Check: $\varepsilon = \ln 2.5 = 0.9163$; the channel rows are then $(0.7, 0.3, 0)$ and $(0, 0.3, 0.7)$.* *(Randomized response is the worst)*

::: {.callout-tip collapse="true" title="Solution"}
1. **Positivity.** $K(y \mid 0) \ge 0$ iff $e^\varepsilon p(y) \ge q(y)$, and $K(y \mid 1) \ge 0$ iff $e^\varepsilon q(y) \ge p(y)$. Together: $e^\varepsilon \ge \max_y\max(p/q, q/p)$, that is $\varepsilon \ge \max_y|L(y)|$.
2. **The ratios.** $p/q = 2.5, 1, 0.4$ on the three outputs. The largest $|\ln(p/q)|$ is $\ln 2.5 = 0.9163$. So the smallest valid $\varepsilon$ is $\ln 2.5$, the pair's pure-DP level.
3. **The channel.** With $e^\varepsilon = 2.5$, $K(\cdot \mid 0) = (2.5P - Q)/1.5 = (1.05, 0.45, 0)/1.5 = (0.7, 0.3, 0)$ and $K(\cdot \mid 1) = (2.5Q - P)/1.5 = (0, 0.3, 0.7)$.
4. **Check.** $t = 2.5/3.5 = 0.7143$, and $t(0.7, 0.3, 0) + (1 - t)(0, 0.3, 0.7) = (0.5, 0.3, 0.2) = P$. Below $\ln 2.5$ the construction fails: at $\varepsilon = 0.8$, $K(\cdot \mid 0) = (0.7448, 0.3, -0.0448)$ has a negative entry, at the third output, where $q/p = 2.5 > e^{0.8}$.

*At the tight level each row of the channel puts zero mass on one output: randomized response's "report 0" can never produce the output that most favours $D'$.*
:::

## Composing Gaussians exactly {#sec-sol-composition-6}

> **Exercise 6.** **Composing Gaussians exactly.** Show that $k$ Gaussian mechanisms with noise $\sigma$ on sensitivity-1 queries, composed non-adaptively, have the same $\delta(\varepsilon)$ as one Gaussian pair with shift $\sqrt k/\sigma$. *Hint: the privacy loss of each step is normal; use Part 1's Gaussian pair.* Then explain the gap in @fig-comp-gaussian by comparing the mean loss $k/(2\sigma^2)$ with $k\varepsilon_0\tanh(\varepsilon_0/2)$ for the per-step $\varepsilon_0$ at $\delta_0 = 5\times10^{-9}$. *Check: for $\sigma = 20$ and $k = 1{,}000$, the exact $\varepsilon$ at $\delta = 10^{-5}$ is 7.511, the true mean loss is 1.25, and $k\varepsilon_0\tanh(\varepsilon_0/2) = 29.93$.* *(Optimal composition, Gaussian pair)*

::: {.callout-tip collapse="true" title="Solution"}
1. **One step.** The worst pair of a sensitivity-1 Gaussian mechanism is $\mathcal N(0, \sigma^2)$ against $\mathcal N(1, \sigma^2)$. With $\mu = 1/\sigma$, Part 1 ([Divergences](../../book-full/part01-prerequisites/03-divergences.md), the Gaussian pair) shows that its privacy loss under $P$ is normal with mean $\mu^2/2$ and variance $\mu^2$.
2. **$k$ steps.** By Fact 2 of the chapter, the total loss is the sum of $k$ independent such losses: normal with mean $k\mu^2/2$ and variance $k\mu^2$.
3. **One pair with shift $\mu_k = \sqrt k\mu$** has loss with mean $\mu_k^2/2 = k\mu^2/2$ and variance $\mu_k^2 = k\mu^2$: the same distribution.
4. **Same loss, same curve.** By Fact 1, $\delta(\varepsilon)$ depends on the pair only through the distribution of $L$ under $P$. So the two curves agree. (The same holds in the other direction by symmetry.)
5. **Check.** $\mu_k = \sqrt{1000}/20 = 1.5811$, and solving the Gaussian $\delta(\varepsilon) = 10^{-5}$ by bisection gives $\varepsilon = 7.511$.
6. **The gap.** The true mean loss of the run is $k\mu^2/2 = 1000/800 = 1.25$. The per-step point at $\delta_0 = 5\times10^{-9}$ is $\varepsilon_0 = 0.2453$, and optimal composition must assume the worst mechanism with that $\varepsilon_0$, randomized response, whose mean loss per step is $\varepsilon_0\tanh(\varepsilon_0/2)$. Over $k$ steps that is $29.93$, 24 times the truth. The fluctuation term $\varepsilon_0\sqrt{2k\ln(1/\delta')}$ with the remaining $\delta' = 5\times10^{-6}$ is 38.32 on top.
7. **Why.** A Gaussian step's loss is rarely as large as $\varepsilon_0$: $\varepsilon_0$ was chosen so that it is exceeded only with probability about $\delta_0$. Randomized response with the same $\varepsilon_0$ has loss $\pm\varepsilon_0$ on *every* output. Replacing each Gaussian step by its worst-case $(\varepsilon_0, \delta_0)$ stand-in inflates both the mean and the spread of the total.

*The $(\varepsilon, \delta)$ summary of one step is a worst case over a whole class, and composing worst cases compounds the loss. Composing the actual loss distributions avoids it.*
:::

## Two different steps {#sec-sol-composition-7}

> **Exercise 7.** **Two different steps.** Compose randomized response at $\varepsilon_1 = \ln 3$ with randomized response at $\varepsilon_2 = \ln 2$. List the four outputs with their probabilities and losses, and compute $\delta(\varepsilon')$ at $\varepsilon' = 0$, $\ln\frac32$ and $\ln 2$. *Check: $\frac12$, $\frac38$ and $\frac13$.* *(Fact 1, Fact 2)*

::: {.callout-tip collapse="true" title="Solution"}
1. **The two steps.** The first is truthful with probability $\frac34$, the second with $\frac23$.
2. **The four outputs** (@tbl-sol-composition-hetero). On $D$ the probabilities multiply. On $D'$ "truthful" and "flipped" swap roles. By Fact 2 the loss adds: $\pm\ln 3 \pm \ln 2$.

| Output | $p$ | $q$ | $e^L = p/q$ |
| --- | --- | --- | --- |
| both truthful | 1/2 | 1/12 | 6 |
| first truthful, second flipped | 1/4 | 1/6 | 3/2 |
| first flipped, second truthful | 1/6 | 1/4 | 2/3 |
| both flipped | 1/12 | 1/2 | 1/6 |

: Randomized response at $\ln 3$ followed by randomized response at $\ln 2$. {#tbl-sol-composition-hetero}

3. **$\delta(0)$.** The outputs with $e^L > 1$ count: $\frac12(1 - \frac16) + \frac14(1 - \frac23) = \frac5{12} + \frac1{12} = \frac12$.
4. **$\delta(\ln\frac32)$.** Only $e^L = 6$ counts: $\frac12(1 - \frac{3/2}{6}) = \frac12\cdot\frac34 = \frac38$.
5. **$\delta(\ln 2)$.** Again only $e^L = 6$: $\frac12(1 - \frac26) = \frac13$. And $\delta(\ln 6) = 0$, the basic composition point $(\ln 3 + \ln 2, 0)$.

*With different $\varepsilon_i$ the loss takes $2^k$ distinct values in general, not $k + 1$. For two steps you can list them; for many, the exact computation is #P-complete, as Murtagh & Vadhan showed.*
:::

## Laplace noise in many dimensions {#sec-sol-composition-8}

> **Exercise 8.** **Laplace noise in many dimensions.** One record can move each of $d$ counts by up to 1. Release each count with its own Laplace noise of scale $1/\varepsilon_0$ and compose the $d$ steps with the $\tanh$ form of advanced composition (@eq-comp-advanced-tanh) at $\delta = 10^{-5}$. Show that $\varepsilon_0 = \varepsilon/(2\sqrt{2d\ln(1/\delta)})$ keeps the total at most $\varepsilon$ when $\varepsilon \le 8\ln(1/\delta)$, so the error per count grows like $\sqrt d$. At $\varepsilon = 1$, compare this closed form with the largest $\varepsilon_0$ the formula allows. *Check: at $d = 1000$ the closed form gives $\varepsilon_0 = 0.00330$, which spends 0.5054 of the budget, and a root-mean-square error of 429.2; the largest $\varepsilon_0$ is 0.00645, with error 219.2.* *(Advanced composition, budgeting)*

::: {.callout-tip collapse="true" title="Solution"}
1. **The first term.** With $\varepsilon_0 = \varepsilon/(2\sqrt{2d\ln(1/\delta)})$, the term $\varepsilon_0\sqrt{2d\ln(1/\delta)}$ is exactly $\varepsilon/2$.
2. **The drift term.** Since $\tanh y \le y$, the term $d\varepsilon_0\tanh(\varepsilon_0/2)$ is at most $d\varepsilon_0^2/2 = d\varepsilon^2/(16d\ln(1/\delta)) = \varepsilon^2/(16\ln(1/\delta))$. This is at most $\varepsilon/2$ when $\varepsilon \le 8\ln(1/\delta)$, which at $\delta = 10^{-5}$ allows any $\varepsilon$ up to 92.
3. **The error.** A $\mathrm{Lap}(1/\varepsilon_0)$ count has root-mean-square error $\sqrt2/\varepsilon_0 = 2\sqrt2\sqrt{2d\ln(1/\delta)}/\varepsilon$, which grows like $\sqrt d$.
4. **Numbers.** At $\varepsilon = 1$ and $d = 1000$, $\varepsilon_0 = 1/(2\sqrt{2000\ln 10^5}) = 0.00330$, and advanced composition charges $0.5054$ for the $d$ steps: the closed form wastes about half the budget, because its drift term is tiny. Its error is 429.2. The largest $\varepsilon_0$ that keeps the total at 1, found by bisection, is 0.00645, with error 219.2, the value of @tbl-comp-dense. At $d = 100$ and $10{,}000$ the closed form gives 135.7 and 1,357.2, against 69.3 and 693.0.

*A closed-form budget is easy to state and safe, but it can leave half the budget unused. When the per-step mechanism is fixed, solve for the largest step budget, or better, compute the exact composition.*
:::
