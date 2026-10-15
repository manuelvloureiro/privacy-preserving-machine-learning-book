# Exercises: Amplification by shuffling and by iteration

Exercises for [Amplification by shuffling and by iteration](../../book-full/part04-privacy-accounting/07-shuffling-iteration.md), each with a worked solution. Try each exercise before opening its solution. The script `code/part04-privacy-accounting/solutions_shuffling_iteration.py` computes every number quoted here; run it with `uv run python code/part04-privacy-accounting/solutions_shuffling_iteration.py`.

The solutions use the chapter's notation for differential privacy (DP) and noisy gradient descent (GD). Neighbouring datasets differ in one record's value (replace-one). Binary randomized response reports the true bit with probability $\theta = e^{\varepsilon_0}/(1 + e^{\varepsilon_0})$. $P$ and $Q$ are the adversary's views on $D$ and $D'$, and $\delta(\varepsilon)$ is the larger of the hockey-stick divergences $E_{e^\varepsilon}(P\|Q)$ and $E_{e^\varepsilon}(Q\|P)$. Gaussian differential privacy (GDP) with parameter $\mu$ is the privacy of a pair of unit-variance Gaussians a distance $\mu$ apart, and Rényi differential privacy (RDP) is measured by the Rényi divergence $D_\alpha$. The one-dimensional iteration is $w_k = c\,w_{k-1} + b_k + \mathcal N(0, \sigma^2)$, with the two runs' offsets differing by $s$ at the steps that use the record. As in the chapter, $\sigma$ is a noise standard deviation in parameter units, $L_f$ a Lipschitz constant and $d_K$ the diameter of the parameter set.

## Two users {#sec-sol-shuf-1}

> **Exercise 1.** **Two users.** Two users run binary randomized response with $\varepsilon_0 = \ln 3$ behind a shuffler; the other user holds 0. Write down the distributions of the count on $D$ and $D'$, and compute $\delta(0.5)$. Then show that for every $n$ the pure-DP level of shuffled randomized response is still $\varepsilon_0$. *Check: $\delta(0.5) = 0.2534$.* *(Exact hockey-stick computation)*

::: {.callout-tip collapse="true" title="Solution"}
Here $\theta = \frac34$.

1. **The other user's count.** The other user holds 0 and reports 1 with probability $1 - \theta = \frac14$: $b = (\frac34, \frac14)$ on counts $(0, 1)$.
2. **The pair.** By @eq-shuf-rr-pair of the chapter, $P(y) = \frac34b(y) + \frac14b(y - 1)$ and $Q(y) = \frac14b(y) + \frac34b(y - 1)$. @tbl-sol-shuf-two lists them.

| Count $y$ | 0 | 1 | 2 |
| --- | --- | --- | --- |
| $P(y)$, you hold 0 | $\frac34\cdot\frac34 = \frac9{16}$ | $\frac34\cdot\frac14 + \frac14\cdot\frac34 = \frac6{16}$ | $\frac14\cdot\frac14 = \frac1{16}$ |
| $Q(y)$, you hold 1 | $\frac14\cdot\frac34 = \frac3{16}$ | $\frac14\cdot\frac14 + \frac34\cdot\frac34 = \frac{10}{16}$ | $\frac34\cdot\frac14 = \frac3{16}$ |
| $P(y)/Q(y)$ | 3 | 0.6 | $\frac13$ |

: Shuffled binary randomized response with two users at $\varepsilon_0 = \ln 3$; the other user holds 0. {#tbl-sol-shuf-two}

3. **Forward direction at $\varepsilon = 0.5$.** $e^{0.5} = 1.6487$. Only $y = 0$ has a ratio above it, so $E_{e^{0.5}}(P\|Q) = \frac{9 - 3e^{0.5}}{16} = 0.2534$.
4. **Backward direction.** $Q/P$ is $\frac13$, $\frac53$ and 3; the last two exceed $e^{0.5}$, so $E_{e^{0.5}}(Q\|P) = \frac{(10 - 6e^{0.5}) + (3 - e^{0.5})}{16} = 0.0912$.
5. **So $\delta(0.5) = 0.2534$**, the larger of the two. A single unshuffled report has $\delta(0.5) = 0.3378$ (chapter, @tbl-shuf-n3).

**Pure DP is not amplified.** For any $n$ with the other users all holding 0, the count $y = 0$ has $P(0) = \theta\,b(0)$ and $Q(0) = (1 - \theta)\,b(0)$, with the same $b(0) = \theta^{n-1} > 0$. So $P(0)/Q(0) = \theta/(1 - \theta) = e^{\varepsilon_0}$ and $\delta(\varepsilon) > 0$ for every $\varepsilon < \varepsilon_0$. The script confirms $\ln(P(0)/Q(0)) = 1.0986 = \ln 3$ at $n = 2, 10, 100$, while $P(0)$ falls from $5.625\times10^{-1}$ to $3.207\times10^{-13}$. Since some pair of datasets has this property, the shuffled mechanism is $\varepsilon$-DP (with $\delta = 0$) for no $\varepsilon < \varepsilon_0$. It is $\varepsilon_0$-DP: the bag is a post-processing of the $n$ independent reports (shuffle them), and changing your data changes only your own report, which is $\varepsilon_0$-DP while the others' reports have the same distribution on $D$ and $D'$. So the pure-DP level is exactly $\varepsilon_0$.

*Shuffling buys a small $\delta$ at a smaller $\varepsilon$; the pure-DP level stays at $\varepsilon_0$, carried by an outcome whose probability vanishes as $n$ grows.*
:::

## How many users? {#sec-sol-shuf-2}

> **Exercise 2.** **How many users?** Using the normal approximation @eq-shuf-rr-clt and the GDP conversion of Fact 3, estimate how many users $n$ make shuffled randomized response with $\varepsilon_0 = 2$ reach $\varepsilon = 0.1$ at $\delta = 10^{-6}$, then check your $n$ with the exact computation. *Hint: find the $\mu$ whose $\varepsilon$ at $\delta = 10^{-6}$ is 0.1, then solve $2\sinh(1)/\sqrt n = \mu$.* *Check: the approximation gives $n = 7282$; the exact computation needs $n = 7781$.* *(Central limit approximation)*

::: {.callout-tip collapse="true" title="Solution"}
1. **Target $\mu$.** The $\varepsilon$ of $\mu$-GDP at fixed $\delta$ increases with $\mu$, so bisection on $\delta(\varepsilon) = \Phi(-\frac\varepsilon\mu + \frac\mu2) - e^\varepsilon\Phi(-\frac\varepsilon\mu - \frac\mu2)$ finds the $\mu$ with $\delta(0.1) = 10^{-6}$: $\mu^* = 0.027545$.
2. **Solve for $n$.** $2\sinh(1) = 2.3504$, so $n = (2.3504/\mu^*)^2 = 7281.3$, rounded up: $n = 7282$.
3. **Check exactly.** At $n = 7282$ the exact $\varepsilon$ of shuffled randomized response is 0.1037, slightly above the target. Bisection on $n$ with the exact computation gives the smallest $n$ with $\varepsilon \le 0.1$: $n = 7781$, where $\varepsilon = 0.1000$.

The approximation undershoots: the exact computation needs 1.069 times as many users. The count of 1-reports is a sum of bits with $\theta = 0.8808$, so it is skewed, and at $\delta = 10^{-6}$ the privacy loss is decided far in its tails, where a normal approximation with the right mean and variance is least accurate. At $\varepsilon_0 = 2$ there are many 1-reports, so the error is small; at $\varepsilon_0 = 4$ and small $n$ it is large (chapter, @tbl-shuf-compare).

*Use the normal approximation to size a deployment, then confirm with the exact computation, which is cheap.*
:::

## Clones of a three-valued report {#sec-sol-shuf-3}

> **Exercise 3.** **Clones of a three-valued report.** For 3-ary randomized response with $\varepsilon_0 = \ln 4$ (report the true value with probability $\frac23$, each other value with probability $\frac16$), decompose the report of a user holding value 3 as in @eq-shuf-clone-split, with $x^0 = 1$ and $x^1 = 2$. Then find the largest clone probability $p$ for which @eq-shuf-clone-split, with $e^{-\varepsilon_0}$ replaced by $p$, holds for every value $x$, and compare it with $e^{-\varepsilon_0}$. *Check: $\mathrm{LO}(3) = (\frac1{12}, \frac1{12}, \frac56)$, and $p = \frac25$ against $e^{-\varepsilon_0} = \frac14$.* *(Clone decomposition)*

::: {.callout-tip collapse="true" title="Solution"}
1. **The three report distributions,** in 48ths on outputs $(1, 2, 3)$: $R(1) = (32, 8, 8)$, $R(2) = (8, 32, 8)$, $R(3) = (8, 8, 32)$. The largest likelihood ratio is $\frac{32}{8} = 4 = e^{\varepsilon_0}$, so the randomizer is $\ln 4$-DP.
2. **The clone part** is $\frac{e^{-\varepsilon_0}}2\big(R(1) + R(2)\big) = \frac18(40, 40, 16)/48 = (5, 5, 2)/48$.
3. **The leftover mass** is $R(3)$ minus the clone part: $(3, 3, 30)/48$, of total $\frac{36}{48} = \frac34 = 1 - e^{-\varepsilon_0}$. Dividing by $\frac34$ gives $\mathrm{LO}(3) = (\frac1{12}, \frac1{12}, \frac56)$. All entries are non-negative, as step 1 of the clone derivation guarantees.
4. **The largest clone probability.** The decomposition with probability $p$ needs $R(x)(y) \ge \frac p2\big(R(x^0)(y) + R(x^1)(y)\big)$ for every output $y$, that is $p \le \min_y\frac{2R(x)(y)}{R(x^0)(y) + R(x^1)(y)}$. For $x = 3$ the binding outputs are 1 and 2: $\frac{2\cdot8}{32 + 8} = \frac25$. For $x = 1$ (a user with your $D$ value) the binding output is 2: $\frac{2\cdot8}{8 + 32} = \frac25$; $x = 2$ is symmetric. So $p = \frac25 = 0.4000$ works for every user, against $e^{-\varepsilon_0} = 0.2500$.
5. **Why the general argument uses less.** Step 1 of the clone derivation uses only $R(x) \ge e^{-\varepsilon_0}R(x^0)$ and $R(x) \ge e^{-\varepsilon_0}R(x^1)$ separately. Averaging two separate lower bounds wastes slack whenever they are not both tight at the same output. For $k$-ary randomized response they are never both tight, since $R(x^0)$ and $R(x^1)$ peak at different outputs. Binary randomized response at $\varepsilon_0 = \ln 3$ likewise allows $p = 0.5000 = 2/(1 + e^{\varepsilon_0})$, above $e^{-\varepsilon_0} = \frac13$.

More clones mean a smaller $\varepsilon$. Section 4 of Feldman, McMillan & Talwar (2021) gives an analysis for specific randomizers that exploits this.

*The clone probability $e^{-\varepsilon_0}$ is what every $\varepsilon_0$-DP randomizer guarantees; a specific randomizer may hide you among more clones.*
:::

## The best shift schedule {#sec-sol-shuf-4}

> **Exercise 4.** **The best shift schedule.** Show that among payments $a_1, \dots, a_m \ge 0$ with $\sum_ka_k = z$, the even schedule minimizes $\sum_k\alpha a_k^2/(2\sigma^2)$, and derive Theorem 23 from it. *Check: with $z = 1$, $\sigma = 1$, $\alpha = 2$, the schedule $(0.8, 0.2)$ costs 0.6800 against 0.5000 for $(0.5, 0.5)$.* Then let one record be used at steps 1 and 4 of a four-step run of translations (two passes), with shift 1 each time. Show that the even schedule is not allowed and find the best one. *Check: $(\frac13, \frac13, \frac13, 1)$, cost 1.3333.* *(Cauchy–Schwarz, shift reduction)*

::: {.callout-tip collapse="true" title="Solution"}
**One use.**

1. **Cauchy–Schwarz.** $z = \sum_k1\cdot a_k \le \sqrt m\sqrt{\sum_ka_k^2}$, so $\sum_ka_k^2 \ge z^2/m$, with equality when all $a_k$ are equal.
2. **The cost.** $\frac{\alpha}{2\sigma^2}\sum_ka_k^2 \ge \frac{\alpha z^2}{2m\sigma^2}$, attained by $a_k = z/m$. With $z = 1$, $\sigma = 1$, $\alpha = 2$, $m = 2$: $(0.5, 0.5)$ costs 0.5000 and $(0.8, 0.2)$ costs $0.64 + 0.04 = 0.6800$.
3. **Theorem 23.** For the record at step $t$ of $n$, the two runs differ by $s \le 2\eta L_f$ at step $t$ only, and the noise in parameter space has standard deviation $\eta\sigma$. Before step $t$ nothing needs paying ($a_k = 0$). From step $t$ on there are $m = n - t + 1$ noisy steps, and the even schedule keeps every remaining shift $z_k$ non-negative and ends at 0. The cost is $\frac{\alpha(2\eta L_f)^2}{2(n - t + 1)\eta^2\sigma^2} = \frac{2\alpha L_f^2}{\sigma^2(n - t + 1)}$.

**Two uses.** Now $s_1 = s_4 = 1$, $s_2 = s_3 = 0$, and Theorem 22 needs $z_t = \sum_{i\le t}(s_i - a_i) \ge 0$ for every $t$ and $z_4 = 0$.

1. **The even schedule fails.** Paying $\frac24 = 0.5$ at every step gives $z = (0.5, 0, -0.5, 0)$. At step 3 more has been paid than has arrived, which the shifted divergence does not allow: you cannot pay off a shift before the record creates it.
2. **The constraint.** Since $z_3 = 1 - (a_1 + a_2 + a_3) \ge 0$, the first three payments total $S \le 1$, and $a_4 = 2 - S$.
3. **Optimize.** For a given $S$, the even split of the first three is best (one use), so the cost is $\frac{\alpha}{2\sigma^2}\big(\frac{S^2}3 + (2 - S)^2\big)$. Its derivative in $S$ is $\frac{2S}3 - 2(2 - S)$, which is negative for all $S < 1.5$. So the best feasible choice is $S = 1$: payments $(\frac13, \frac13, \frac13, 1)$, remaining shifts $(0.6667, 0.3333, 0, 0)$, and cost $\frac13 + 1 = 1.3333$ at $\alpha = 2$, $\sigma = 1$.

For translations the exact value is smaller: the final means differ by 2 and the variance is 4, so $D_2 = 1.0000$. The schedule bound cannot use the noise of steps 1 to 3 against the second shift, because that shift does not exist yet when the noise is added.

*A record used again late in training gets little amplification: the last use dominates, as in a single pass.*
:::

## The hidden-state limit {#sec-sol-shuf-5}

> **Exercise 5.** **The hidden-state limit.** Derive the limit in @eq-shuf-last-iterate and find the number of steps after which full-batch noisy GD with $c = 0.9$ and $s/\sigma = 1$ is within 1% of its limit. Show that the number of steps needed grows like $1/(1 - c)$. *Check: the limit is 4.3589, and $\mu_T$ is within 1% of it from $T = 44$.* *(Geometric series)*

::: {.callout-tip collapse="true" title="Solution"}
1. **The two sums.** With $t = 1$, the mean difference is $s\sum_{k=1}^Tc^{T-k} = s\frac{1 - c^T}{1 - c}$, and the variance is $\sigma^2\sum_{j=0}^{T-1}c^{2j} = \sigma^2\frac{1 - c^{2T}}{1 - c^2}$ (geometric series).
2. **The ratio.** $\mu_T = \frac s\sigma\cdot\frac{1 - c^T}{1 - c}\sqrt{\frac{1 - c^2}{1 - c^{2T}}} = \frac s\sigma\sqrt{\frac{1 + c}{1 - c}}\cdot\frac{1 - c^T}{\sqrt{1 - c^{2T}}}$, using $1 - c^2 = (1 - c)(1 + c)$.
3. **The limit.** For $0 \le c < 1$, $c^T \to 0$, so $\mu_T \to \frac s\sigma\sqrt{\frac{1 + c}{1 - c}}$. At $c = 0.9$: $\sqrt{19} = 4.3589$.
4. **The approach.** The correction factor is $g(T) = \frac{1 - c^T}{\sqrt{1 - c^{2T}}} = \sqrt{\frac{1 - c^T}{1 + c^T}}$ (divide $1 - c^{2T} = (1 - c^T)(1 + c^T)$). For small $c^T$, $g \approx 1 - c^T$. So $g \ge 0.99$ needs roughly $c^T \le 0.01$, that is $T \ge \ln 100/\ln(1/c) \approx 4.61/(1 - c)$, using $\ln(1/c) \approx 1 - c$ for $c$ near 1.
5. **Numerically.** The script finds the first $T$ with $\mu_T \ge 0.99\lim\mu$: 44 steps for $c = 0.9$ and 458 for $c = 0.99$ (limit 14.1067). Times $1 - c$ these are 4.40 and 4.58, close to $\ln 100 = 4.61$.

*The last iterate forgets at the rate the optimizer contracts. After a few multiples of $1/(1 - c)$ steps, more training costs no more privacy.*
:::

## Where the plateau starts {#sec-sol-shuf-6}

> **Exercise 6.** **Where the plateau starts.** In @eq-shuf-plateau, show that the minimum over $m$ is at $m = d_K/s$ when $d_K/s$ is an integer, that the plateau is $2\alpha d_Ks/\sigma^2$, and that the bound leaves composition at $T = 4d_K/s$. Evaluate both at $d_K = 1$, $s = 0.01$, $\sigma = 0.1$ and $\alpha = 2$. *Check: plateau 4.0000, starting at $T = 400$.* *(Optimizing a schedule)*

::: {.callout-tip collapse="true" title="Solution"}
1. **Expand.** $h(m) = \frac{(d_K + ms)^2}m = \frac{d_K^2}m + 2d_Ks + ms^2$.
2. **Minimize.** $h'(m) = -\frac{d_K^2}{m^2} + s^2 = 0$ at $m = d_K/s$, and $h'' = 2d_K^2/m^3 > 0$, so it is a minimum over real $m > 0$: $h(d_K/s) = d_Ks + 2d_Ks + d_Ks = 4d_Ks$. When $d_K/s$ is an integer this $m$ is allowed. (Otherwise take the better of the two integer neighbours, which costs less than an extra $s^2$; here $d_K/s = 100$ is an integer.)
3. **The plateau.** $\frac{\alpha}{2\sigma^2}\cdot 4d_Ks = \frac{2\alpha d_Ks}{\sigma^2}$, independent of $T$, available once $T \ge d_K/s$.
4. **Where it beats composition.** The composition term is $\frac{\alpha Ts^2}{2\sigma^2}$. It equals the plateau when $Ts^2 = 4d_Ks$, at $T = 4d_K/s$. For $d_K/s \le T < 4d_K/s$, the minimum over $m \le T$ is still $4d_Ks$, but composition is smaller, so the bound follows composition until $T = 4d_K/s$.
5. **Numbers.** $m^* = 100$, the plateau is $2\cdot2\cdot1\cdot0.01/0.1^2 = 4.0000$, and it starts at $T = 400$. The script gives the bound as 3.9900 at $T = 399$ (composition), 4.0000 at $T = 400$, and 4.0000 at $T = 10{,}000$.

*Only the last $d_K/s$ steps are paid for: the steps before are summarized by a starting point somewhere in the domain.*
:::
