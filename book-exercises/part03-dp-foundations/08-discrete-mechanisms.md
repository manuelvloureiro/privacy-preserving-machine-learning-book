# Exercises: Discrete and floating-point-safe mechanisms

Exercises for [Discrete and floating-point-safe mechanisms](../../book-full/part03-dp-foundations/08-discrete-mechanisms.md), each with a worked solution. Try each exercise before opening its solution. The script `code/part03-dp-foundations/solutions_discrete_mechanisms.py` computes every number quoted here; run it with `uv run python code/part03-dp-foundations/solutions_discrete_mechanisms.py`.

The solutions use the chapter's notation. $\mathrm{Lap}_{\mathbb Z}(b)$ is the discrete Laplace with probability mass function (PMF) $\frac{1 - r}{1 + r}r^{|z|}$ and $r = e^{-1/b}$; $\mathcal N_{\mathbb Z}(0, \sigma^2)$ is the discrete Gaussian; Algorithms 1–3 are the samplers of Canonne, Kamath and Steinke (2020). Exercise 7 is about differential privacy (DP), and exercise 6 measures the distance between two PMFs in total variation (TV).

## The cost of Algorithm 1 {#sec-sol-discrete-1}

> **Exercise 1.** **The cost of Algorithm 1.** Show that the expected number of uniform draws for $\mathrm{Bern}(e^{-\lambda})$ is $e\sum_{j=0}^{n-1}e^{-j} + e^{-n}e^{\lambda - n}$ with $n = \lfloor\lambda\rfloor$, and that it is below $\frac{e}{1 - e^{-1}} + 1$ for every $\lambda$. *Check: 3.9414 for $\lambda = 5/2$.* *(Algorithm 1, @eq-bern-exp)*

::: {.callout-tip collapse="true" title="Solution"}
1. **One unit coin.** For $\lambda' \le 1$, the loop draws one uniform integer per coin $A_1, A_2, \dots, A_K$, so it costs $K$ draws, and $\mathbb E[K] = e^{\lambda'}$ by @eq-bern-exp. A coin with $\lambda' = 1$ costs $e$ on average.
2. **Which coins run.** Write $\lambda = n + \phi$ with $n = \lfloor\lambda\rfloor$ and $0 \le \phi < 1$. The $(j + 1)$-th of the $n$ coins with probability $e^{-1}$ runs only if the first $j$ all came up 1, which has probability $e^{-j}$. The final coin, with probability $e^{-\phi}$, runs only if all $n$ came up 1, with probability $e^{-n}$.
3. **Add the expected costs.** A coin that runs costs on average what step 1 says, so by linearity of expectation
$$\mathbb E[\text{draws}] = \sum_{j=0}^{n-1}e^{-j}\cdot e + e^{-n}\cdot e^{\phi},$$
which is the claimed formula since $\phi = \lambda - n$.
4. **The bound.** The first sum is below the full geometric series $e\sum_{j \ge 0}e^{-j} = \frac{e}{1 - e^{-1}} = 4.3003$. For $n \ge 1$ the last term is $e^{\phi - n} \le e^{1 - 1} = 1$. For $n = 0$ the whole cost is $e^\lambda < e$. Either way the total is below $4.3003 + 1 = 5.3003$.
5. **Check.** For $\lambda = 5/2$: $n = 2$, $\phi = \frac12$, so the cost is $e(1 + e^{-1}) + e^{-2}e^{1/2} = 3.7183 + 0.2231 = 3.9414$. Over 20,000 coins the script counts 3.9233 draws per coin. At $\lambda = \frac12$ the formula gives $e^{1/2} = 1.6487$ (simulated 1.6597) and at $\lambda = 1$ it gives $e + e^{-1} = 3.0862$ (simulated 3.0854).

*The cost jumps at each integer: just below 1 it is $e = 2.7183$, at 1 it is 3.0862, because the split always runs a final coin, even for $\phi = 0$, where that coin costs exactly one draw.*
:::

## Discrete Laplace as a difference of geometrics {#sec-sol-discrete-2}

> **Exercise 2.** **Discrete Laplace as a difference of geometrics.** Let $G_1, G_2$ be independent with $\Pr[G = g] = (1 - r)r^g$ for $g \ge 0$. Show that $G_1 - G_2 \sim \mathrm{Lap}_{\mathbb Z}(b)$ with $r = e^{-1/b}$, the integer version of "a difference of exponentials is Laplace" in [Probability distributions](../../book-full/part01-prerequisites/01-distributions.md). *Check: $\Pr[G_1 - G_2 = 0] = 0.4621$ for $r = e^{-1}$.* *(Fact 3, convolution)*

::: {.callout-tip collapse="true" title="Solution"}
1. **Condition on $G_2$.** For $z \ge 0$, $G_1 - G_2 = z$ means $G_1 = g + z$ when $G_2 = g$, for some $g \ge 0$. The events for different $g$ are disjoint, so
$$\Pr[G_1 - G_2 = z] = \sum_{g \ge 0}(1 - r)r^{g + z}\cdot(1 - r)r^g = (1 - r)^2r^z\sum_{g \ge 0}r^{2g}.$$
2. **Sum the geometric series.** $\sum_gr^{2g} = \frac{1}{1 - r^2} = \frac{1}{(1 - r)(1 + r)}$, so $\Pr[G_1 - G_2 = z] = \frac{1 - r}{1 + r}r^z$.
3. **Negative $z$.** $G_1 - G_2$ and $G_2 - G_1$ have the same distribution (swap the names), so $\Pr[G_1 - G_2 = -z] = \Pr[G_1 - G_2 = z]$, and the PMF is $\frac{1 - r}{1 + r}r^{|z|}$ for all $z$. That is $\mathrm{Lap}_{\mathbb Z}(b)$.
4. **Check.** At $r = e^{-1}$, $\Pr[G_1 - G_2 = 0] = \frac{1 - e^{-1}}{1 + e^{-1}} = 0.4621$. The script sums the convolution numerically and finds it within $4.3\times10^{-19}$ of the PMF for every $|z| \le 10$.

*Algorithm 2 builds one geometric and a fair sign instead of two geometrics, because the sign trick needs only one coin; both give the same distribution.*
:::

## Variance of the discrete Laplace {#sec-sol-discrete-3}

> **Exercise 3.** **Variance of the discrete Laplace.** Show $\frac{2r}{(1 - r)^2} = \frac{1}{2\sinh^2(1/(2b))}$, deduce that it is below $2b^2$, and show that $2b^2 - \operatorname{Var}(Z) \to \frac16$ as $b \to \infty$. *Check: 1.8413 at $b = 1$ and 199.8334 at $b = 10$.* *Hint: $\sinh x \ge x$ and $\sinh^2x = x^2 + x^4/3 + O(x^6)$.* *(Discrete Laplace)*

::: {.callout-tip collapse="true" title="Solution"}
Let $x = \frac{1}{2b}$, so $r = e^{-2x}$.

1. **Rewrite.** Divide the numerator and the denominator by $r$: $\frac{2r}{(1 - r)^2} = \frac{2}{(r^{-1/2} - r^{1/2})^2} = \frac{2}{(e^x - e^{-x})^2} = \frac{2}{4\sinh^2x} = \frac{1}{2\sinh^2x}$.
2. **The bound.** $\sinh x = x + x^3/6 + \dots \ge x$ for $x \ge 0$, so $\frac{1}{2\sinh^2x} \le \frac{1}{2x^2} = 2b^2$, with strict inequality for every finite $b$.
3. **The gap.** $\sinh^2x = x^2(1 + x^2/3 + O(x^4))$, so $\frac{1}{2\sinh^2x} = \frac{1}{2x^2}\big(1 - \frac{x^2}{3} + O(x^4)\big) = 2b^2 - \frac16 + O(1/b^2)$.
4. **Check** (@tbl-sol-discrete-var).

| $b$ | $\frac{2r}{(1 - r)^2}$ | $\frac{1}{2\sinh^2(1/(2b))}$ | $2b^2$ | Gap |
| --- | --- | --- | --- | --- |
| 0.5 | 0.3620 | 0.3620 | 0.5000 | 0.1380 |
| 1 | 1.8413 | 1.8413 | 2.0000 | 0.1587 |
| 10 | 199.8334 | 199.8334 | 200.0000 | 0.1666 |
| 100 | 19999.8333 | 19999.8333 | 20000.0000 | 0.1667 |

: The variance of $\mathrm{Lap}_{\mathbb Z}(b)$ in both forms, against the continuous $2b^2$. {#tbl-sol-discrete-var}

*The discrete Laplace has the same privacy as $\mathrm{Lap}(b)$ and a variance smaller by almost exactly $\frac16$: a fixed saving that matters only at small scales.*
:::

## Accuracy of the geometric mechanism {#sec-sol-discrete-4}

> **Exercise 4.** **Accuracy of the geometric mechanism.** Show $\Pr[|Z| \ge k] = \frac{2r^k}{1 + r}$ for $k \ge 1$. For a count released at $\varepsilon = 1$, find the smallest $k$ with $\Pr[|Z| > k] \le 0.05$, and compare with the continuous Laplace at $\varepsilon = 0.1$. *Check: $\Pr[|Z| \ge 3] = 0.0728$ at $\varepsilon = 1$, and $k = 3$.* *(Discrete Laplace tails)*

::: {.callout-tip collapse="true" title="Solution"}
1. **One tail.** $\Pr[Z \ge k] = \frac{1 - r}{1 + r}\sum_{z \ge k}r^z = \frac{1 - r}{1 + r}\cdot\frac{r^k}{1 - r} = \frac{r^k}{1 + r}$.
2. **Both tails.** By symmetry, $\Pr[|Z| \ge k] = \frac{2r^k}{1 + r}$ for $k \ge 1$. (For $k = 0$ it is 1, not $\frac{2}{1 + r}$, because 0 is counted once.)
3. **At $\varepsilon = 1$** ($b = 1$, $r = e^{-1}$): $\Pr[|Z| \ge 3] = 2e^{-3}/(1 + e^{-1}) = 0.0728$ and $\Pr[|Z| \ge 4] = 0.0268$. $\Pr[|Z| > k] = \Pr[|Z| \ge k + 1]$, so the smallest $k$ with $\Pr[|Z| > k] \le 0.05$ is $k = 3$: the released count is within 3 of the truth with probability at least 95%. For the continuous $\mathrm{Lap}(1)$, $\Pr[|X| \ge 3] = e^{-3} = 0.0498$.
4. **At $\varepsilon = 0.1$** ($b = 10$): the script finds the smallest radius is 30. The continuous Laplace has $\Pr[|X| \ge k] = e^{-k/b} \le 0.05$ for $k \ge b\ln20 = 29.96$. At large scales the two agree.

*At $\varepsilon = 1$ the comparison with the continuous Laplace is between an integer radius and a real one: the discrete noise can't be 2.5, so its tail at 3 includes the mass the continuous one puts on $[2.5, 3)$.*
:::

## The acceptance rate of Algorithm 3 {#sec-sol-discrete-5}

> **Exercise 5.** **The acceptance rate of Algorithm 3.** Derive @eq-dgauss-accept and show that $A \to \sqrt{\pi/2}\,e^{-1/2}$ as $\sigma \to \infty$. *Check: $A = 0.6629$ at $\sigma^2 = 4$ and 0.7601 at $\sigma^2 = 10^4$.* *Hint: $\sigma/t \to 1$, and $\sum_ye^{-y^2/(2\sigma^2)}/(\sqrt{2\pi}\sigma) \to 1$ by @eq-poisson-theta.* *(Rejection sampling, Poisson summation)*

::: {.callout-tip collapse="true" title="Solution"}
1. **Average the acceptance over the proposal.** $A = \sum_y\Pr[Y = y]\,e^{-(|y| - \sigma^2/t)^2/(2\sigma^2)}$ with $\Pr[Y = y] = \frac{1 - r}{1 + r}e^{-|y|/t}$.
2. **Combine the exponents,** as in step 1 of the sampler's derivation: $-\frac{|y|}{t} - \frac{(|y| - \sigma^2/t)^2}{2\sigma^2} = -\frac{y^2}{2\sigma^2} - \frac{\sigma^2}{2t^2}$. So
$$A = \frac{1 - r}{1 + r}\,e^{-\sigma^2/(2t^2)}\sum_ye^{-y^2/(2\sigma^2)},$$
which is @eq-dgauss-accept.
3. **Each factor as $\sigma \to \infty$.** $t = \lfloor\sigma\rfloor + 1$ satisfies $\sigma < t \le \sigma + 1$, so $\sigma/t \to 1$ and $e^{-\sigma^2/(2t^2)} \to e^{-1/2}$. Next, $\frac{1 - r}{1 + r} = \tanh\frac{1}{2t}$, which is $\frac{1}{2t}(1 + O(1/t^2))$. Finally @eq-poisson-theta at $c = 0$ gives $\sum_ye^{-y^2/(2\sigma^2)} = \sqrt{2\pi}\sigma(1 + 2e^{-2\pi^2\sigma^2} + \dots)$, so the sum is $\sqrt{2\pi}\sigma$ up to an error that vanishes like $e^{-2\pi^2\sigma^2}$.
4. **Multiply.** $A \approx \frac{1}{2t}\cdot e^{-1/2}\cdot\sqrt{2\pi}\sigma \to \frac{\sqrt{2\pi}}{2}e^{-1/2} = \sqrt{\pi/2}\,e^{-1/2} = 0.7602$.
5. **Check** (@tbl-sol-discrete-accept).

| $\sigma^2$ | $t$ | $A$, closed form | $A$, summed over the proposal |
| --- | --- | --- | --- |
| 1/4 | 1 | 0.5185 | 0.5185 |
| 1 | 2 | 0.5418 | 0.5418 |
| 4 | 3 | 0.6629 | 0.6629 |
| 100 | 11 | 0.7532 | 0.7532 |
| 10,000 | 101 | 0.7601 | 0.7601 |

: The acceptance probability of Algorithm 3. It rises towards the limit 0.7602 as $\sigma$ grows. {#tbl-sol-discrete-accept}

*At $\sigma = 0.5$ and $\sigma = 1$, $t$ is twice $\sigma$, which is why the rate is lowest there; for large $\sigma$, $t$ is within 1 of $\sigma$ and the rate settles near its limit. A draw never needs more than about two proposals on average.*
:::

## Sums of discrete Gaussians {#sec-sol-discrete-6}

> **Exercise 6.** **Sums of discrete Gaussians.** Compute the PMF of the sum of two independent $\mathcal N_{\mathbb Z}(0, \sigma^2)$ variables and compare it with $\mathcal N_{\mathbb Z}(0, 2\sigma^2)$ for $\sigma^2 = \frac14$, 1 and 4. Show that the sum's variance is $2\operatorname{Var}(X)$, and explain why it differs from the variance of $\mathcal N_{\mathbb Z}(0, 2\sigma^2)$ at small $\sigma$. *Check: at $\sigma^2 = \frac14$, $\Pr[\text{sum} = 0] = 0.6414$ against 0.5641.* *(Discrete Gaussian, @eq-dgauss-var)*

::: {.callout-tip collapse="true" title="Solution"}
1. **The convolution.** $\Pr[X_1 + X_2 = z] = \frac{1}{Z^2}\sum_xe^{-x^2/(2\sigma^2)}e^{-(z - x)^2/(2\sigma^2)}$, with $Z$ the discrete Gaussian normalizer.
2. **Complete the square.** $x^2 + (z - x)^2 = 2(x - z/2)^2 + z^2/2$. So
$$\Pr[X_1 + X_2 = z] = \frac{e^{-z^2/(4\sigma^2)}}{Z^2}\sum_xe^{-(x - z/2)^2/\sigma^2}.$$
The factor $e^{-z^2/(4\sigma^2)}$ is the $\mathcal N_{\mathbb Z}(0, 2\sigma^2)$ shape. The sum over $x$ is the lattice sum of @eq-poisson-theta with variance $\sigma^2/2$ and shift $c = z/2$: an integer for even $z$, a half-integer for odd $z$.
3. **Why it is not a discrete Gaussian.** By the lemma, the lattice sum is larger at integer shifts. So even $z$ get a little more weight than $\mathcal N_{\mathbb Z}(0, 2\sigma^2)$ gives them, and odd $z$ a little less. By @eq-poisson-theta, the ratio of the two values is $\frac{1 + 2e^{-\pi^2\sigma^2} + \cdots}{1 - 2e^{-\pi^2\sigma^2} + \cdots}$, because $2\pi^2(\sigma^2/2)k^2 = \pi^2\sigma^2k^2$ and $\cos(2\pi k\cdot\frac12) = (-1)^k$.
4. **Variance.** Variances of independent variables add, so $\operatorname{Var}(X_1 + X_2) = 2\operatorname{Var}(X)$, which is $2\sigma^2$ minus twice the deficit of @eq-dgauss-var. $\mathcal N_{\mathbb Z}(0, 2\sigma^2)$ has its own, much smaller deficit, because the deficit decays like $e^{-2\pi^2\sigma^2}$ and $\sigma^2$ has doubled.
5. **Check** (@tbl-sol-discrete-sum).

| $\sigma^2$ each | $\Pr[\text{sum} = 0]$ | $\mathcal N_{\mathbb Z}(0, 2\sigma^2)$ at 0 | Variance of the sum | Variance of $\mathcal N_{\mathbb Z}(0, 2\sigma^2)$ | TV |
| --- | --- | --- | --- | --- | --- |
| 1/4 | 0.6414 | 0.5641 | 0.4300 | 0.4990 | $8.01\times10^{-2}$ |
| 1 | 0.2821 | 0.2821 | 2.0000 | 2.0000 | $5.17\times10^{-5}$ |
| 4 | 0.1410 | 0.1410 | 8.0000 | 8.0000 | $1.19\times10^{-16}$ |

: The sum of two independent discrete Gaussians against a single discrete Gaussian with twice the parameter. {#tbl-sol-discrete-sum}

At $\sigma^2 = \frac14$, $\operatorname{Var}(X) = 0.2150$ (from the chapter's script), so the sum has variance 0.4300, while $\mathcal N_{\mathbb Z}(0, \frac12)$ has 0.4990. At $\sigma = 1$ the TV distance is $5.17\times10^{-5}$, of the order of $e^{-\pi^2} = 5.2\times10^{-5}$, as step 3 predicts.

*When clients each add discrete Gaussian noise before secure aggregation, the total is not exactly a discrete Gaussian. For $\sigma \ge 1$ per client the difference is far below anything that matters, and Kairouz, Liu and Steinke (2021) bound it.*
:::

## The support argument as a test {#sec-sol-discrete-7}

> **Exercise 7.** **The support argument as a test.** In the 12-bit toy, the test "say $D$ when $y \notin \mathcal Y_{D'}$, and $D'$ otherwise" never says $D$ when the data are $D'$, so its false negative rate is 0. Using the hypothesis-testing form of $(\varepsilon, \delta)$-DP, show that the toy sampler is not $(\varepsilon, \delta)$-DP for any $\varepsilon$ when $\delta < 0.4791$, and find the bound the reverse test gives. Then show that after snapping to $\Lambda = 1$ and clamping to $[-10, 10]$, every integer in $[-10, 10]$ is reachable from both inputs. *Check: the reverse test gives $\delta \ge 0.2033$.* *(Support argument, snapping)*

::: {.callout-tip collapse="true" title="Solution"}
1. **The test.** Say "$D$" when $y \in S = \mathcal Y_D \setminus \mathcal Y_{D'}$, and "$D'$" otherwise. With the convention of [The definition of differential privacy](../../book-full/part03-dp-foundations/01-definition.md), the false positive rate (FPR) is the chance of saying "$D'$" when the data are $D$, and the false negative rate (FNR) the chance of saying "$D$" when they are $D'$. Here $\mathrm{FNR} = \Pr[M(D') \in S] = 0$, because $M(D')$ never lands in $S$, and $\mathrm{FPR} = 1 - \Pr[M(D) \in S] = 1 - 0.4791$ (@tbl-discrete-float).
2. **The constraint.** $(\varepsilon, \delta)$-DP makes every test satisfy $\mathrm{FPR} + e^\varepsilon\,\mathrm{FNR} \ge 1 - \delta$; this is the hypothesis-testing form in [The definition of differential privacy](../../book-full/part03-dp-foundations/01-definition.md), and also the definition applied to the event $S$ on which the test says "$D$". With $\mathrm{FNR} = 0$ the factor $e^\varepsilon$ multiplies nothing, and the inequality reads $1 - 0.4791 \ge 1 - \delta$, that is, $0.4791 \le \delta$. So for $\delta < 0.4791$ no $\varepsilon$ works.
3. **The reverse test.** Swap the roles: say "$D'$" when $y \in S' = \mathcal Y_{D'} \setminus \mathcal Y_D$, and "$D$" otherwise. Its false positive rate is $\Pr[M(D) \in S'] = 0$ and its true positive rate is $\Pr[M(D') \in S'] = 0.2033$. DP must hold in both directions, and the other inequality of the hypothesis-testing form, $\mathrm{TPR} \le e^\varepsilon\cdot\mathrm{FPR} + \delta$, then gives $\delta \ge 0.2033$. The direction from $D$ gives the stronger bound.
4. **Snapping.** With $b = 1$, an output $k \in \{-9, \dots, 9\}$ arises from input $f \in \{0, 1\}$ when the rounded noise lands in an interval of width 1 around $k - f$, which lies in $[-10.5, 9.5]$. The noise magnitudes $\mathrm{round}(\ln(1/u))$ run from about $2^{-12}$ up to $40\ln2 \approx 27.7$, and consecutive ones differ by at most about $2^{-11}$ of their size (consecutive $u$ differ by a relative step of at most $2^{-11}$, and $\ln$ turns a relative step into an additive one). So every interval of width 1 below 27.7 contains noise values, with either sign. The clamp sends everything beyond $\pm9.5$ to $\pm10$, which both inputs reach. So all 21 integers in $[-10, 10]$ are reachable from both inputs, and the script confirms both supports have exactly 21 points with $\Pr[M(D) \in S] = 0$.

*The support argument gives a lower bound on $\delta$ from one single event, without any statistics. Auditing tools use exactly such events, found by search, to refute claimed guarantees.*
:::
