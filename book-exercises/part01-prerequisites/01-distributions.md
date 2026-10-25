# Exercises: Probability distributions

Exercises for [Probability distributions](../../book-full/part01-prerequisites/01-distributions.md), each with a worked solution. Try each exercise before opening its solution. The numbers come from [code/solutions_distributions.py](../../code/part01-prerequisites/solutions_distributions.py); run it from the repository root with `uv run python code/part01-prerequisites/solutions_distributions.py`.

The solutions work with the probability mass function (PMF) and the moment generating function (MGF) of each family, exactly as the chapter defines them.

## Binomial variance from the PMF {#sec-sol-dist-1}

> **Exercise 1.** **Binomial variance from the PMF.** Compute $\mathbb E[S(S-1)]$ for $S \sim \mathrm{Bin}(n, p)$ directly from the PMF, and recover $\operatorname{Var}S = np(1-p)$. Compare the effort with the decomposition argument. *(Binomial)*

::: {.callout-tip collapse="true" title="Solution"}
Write $q = 1 - p$.

1. **Write the expectation as a sum over the PMF.** The terms $k = 0$ and $k = 1$ vanish because $k(k-1) = 0$:
$$\mathbb E[S(S-1)] = \sum_{k=2}^n k(k-1)\binom nk p^kq^{n-k}.$$

2. **Absorb $k(k-1)$ into the binomial coefficient.** For $k \ge 2$,
$$k(k-1)\binom nk = \frac{n!}{(k-2)!\,(n-k)!} = n(n-1)\binom{n-2}{k-2}.$$

3. **Pull out $n(n-1)p^2$ and re-index with $j = k - 2$:**
$$\mathbb E[S(S-1)] = n(n-1)p^2\sum_{j=0}^{n-2}\binom{n-2}{j}p^jq^{(n-2)-j}.$$

4. **The remaining sum is a whole PMF.** It is the sum of the $\mathrm{Bin}(n-2, p)$ PMF over all its values, which is 1 (equivalently, $(p + q)^{n-2} = 1$ by the binomial theorem). So
$$\mathbb E[S(S-1)] = n(n-1)p^2.$$

5. **Recover the variance.** $\mathbb E S^2 = \mathbb E[S(S-1)] + \mathbb E S$, and the same trick with one factor $k$ gives $\mathbb E S = np$. So
$$\operatorname{Var}S = n(n-1)p^2 + np - n^2p^2 = np - np^2 = np(1-p).$$

**A small case.** For $n = 3$, @tbl-sol-distributions-binomial-n3 lists every term of the sum in step 1.

| $k$ | $k(k-1)$ | $\Pr[S = k]$ | $k(k-1)\Pr[S = k]$ |
| --- | --- | --- | --- |
| 0 | 0 | $q^3$ | 0 |
| 1 | 0 | $3pq^2$ | 0 |
| 2 | 2 | $3p^2q$ | $6p^2q$ |
| 3 | 6 | $p^3$ | $6p^3$ |

: The terms of $\mathbb E[S(S-1)]$ for $S \sim \mathrm{Bin}(3, p)$, with $q = 1 - p$. {#tbl-sol-distributions-binomial-n3}

The total is $6p^2q + 6p^3 = 6p^2(q + p) = 6p^2 = n(n-1)p^2$, as step 4 says. Summed numerically over the PMF, the script gets $\mathbb E[S(S-1)] = 34.2000$ and $\operatorname{Var}S = 4.2000$ for $\mathrm{Bin}(20, 0.3)$, and $2497.5000$ and $47.5000$ for $\mathrm{Bin}(1000, 0.05)$, matching $n(n-1)p^2$ and $np(1-p)$.

**The effort.** The PMF route needs the factorial moment $\mathbb E[S(S-1)]$ (not $\mathbb E S^2$, which does not telescope), an identity for the coefficient, a re-index, and the fact that a PMF sums to 1. The decomposition $S = X_1 + \dots + X_n$ needs one line: each Bernoulli has variance $p(1-p)$, and independent variances add.

*Factorial moments are the moments that a PMF with factorials sums easily, but decomposing into independent pieces is shorter.*
:::

## Poisson thinning {#sec-sol-dist-2}

> **Exercise 2.** **Poisson thinning.** $N \sim \mathrm{Poisson}(\mu)$ events occur, and each is kept independently with probability $q$. Show the number kept is $\mathrm{Poisson}(q\mu)$. *Hint: condition on $N$ and use the MGF.* *(Poisson)*

::: {.callout-tip collapse="true" title="Solution"}
Let $K$ be the number kept.

1. **Given $N$, the count kept is binomial.** If $N = n$, then $K$ counts the successes in $n$ independent $\mathrm{Bern}(q)$ trials, so $K \sim \mathrm{Bin}(n, q)$. Its MGF is the binomial MGF from the chapter:
$$\mathbb E[e^{\lambda K} \mid N = n] = \big(1 + q(e^\lambda - 1)\big)^n.$$

2. **Average over $N$.** The MGF of $K$ is the average of these conditional MGFs, weighted by $\Pr[N = n]$:
$$M_K(\lambda) = \sum_{n\ge0}\Pr[N = n]\,\big(1 + q(e^\lambda - 1)\big)^n = \mathbb E\big[c^N\big], \qquad c = 1 + q(e^\lambda - 1).$$

3. **Recognize the Poisson MGF.** Write $c^N = e^{N\ln c}$, so $\mathbb E[c^N] = M_N(\ln c) = \exp\big(\mu(e^{\ln c} - 1)\big) = \exp\big(\mu(c - 1)\big)$. Substituting $c - 1 = q(e^\lambda - 1)$,
$$M_K(\lambda) = \exp\big(q\mu(e^\lambda - 1)\big).$$

4. **Identify the distribution.** This is the $\mathrm{Poisson}(q\mu)$ MGF, for every $\lambda$. Two distributions with the same MGF around 0 are the same (@sec-dist-mgf), so $K \sim \mathrm{Poisson}(q\mu)$.

**Step 2 on a small case.** Take the outcome $N = 2$. The two events are kept or dropped independently, so @tbl-sol-distributions-thinning lists the four possibilities.

| Event 1 | Event 2 | Probability | $K$ | $e^{\lambda K}$ |
| --- | --- | --- | --- | --- |
| dropped | dropped | $(1 - q)^2$ | 0 | 1 |
| kept | dropped | $q(1 - q)$ | 1 | $e^\lambda$ |
| dropped | kept | $(1 - q)q$ | 1 | $e^\lambda$ |
| kept | kept | $q^2$ | 2 | $e^{2\lambda}$ |

: The four outcomes of keeping or dropping two events, each with probability $q$ of being kept. {#tbl-sol-distributions-thinning}

The conditional average is $(1-q)^2 + 2q(1-q)e^\lambda + q^2e^{2\lambda} = \big((1-q) + qe^\lambda\big)^2 = \big(1 + q(e^\lambda - 1)\big)^2$, which is step 1 with $n = 2$. Step 2 does this for every $n$ and weights the results by $\Pr[N = n]$.

As a check, the script draws $N \sim \mathrm{Poisson}(4)$ a million times and keeps each event with probability $q = 0.3$. The kept count has mean 1.198 and variance 1.200, against $q\mu = 1.200$ for both, as a Poisson needs.

*Randomly discarding events from a Poisson count leaves a Poisson count, with the mean scaled by the sampling rate $q$.*
:::

## Quality of the Poisson approximation {#sec-sol-dist-3}

> **Exercise 3.** **Quality of the Poisson approximation.** For $\mathrm{Bin}(n, 5/n)$, compute the total variation distance $\frac12\sum_k|\Pr[S = k] - \Pr[N = k]|$ to $\mathrm{Poisson}(5)$ for $n = 10, 100, 1000$ using the functions in the script. *Check: it shrinks roughly like $1/n$.* *(Poisson limit)*

::: {.callout-tip collapse="true" title="Solution"}
1. **Split the sum at $n$.** The binomial puts no mass above $n$, so for $k > n$ each term is $\Pr[N = k]$, and those terms add up to $\Pr[N > n]$:
$$\frac12\sum_k\big|\Pr[S = k] - \Pr[N = k]\big| = \frac12\left(\sum_{k=0}^n\big|\Pr[S = k] - \Pr[N = k]\big| + 1 - \sum_{k=0}^n\Pr[N = k]\right).$$

2. **Evaluate with the chapter's functions.** `binomial_poisson_tv` in the solutions script computes this with `binomial_pmf` and `poisson_pmf` from `distributions.py`. @tbl-sol-distributions-tv gives the results.

| $n$ | Total variation distance | $n \times$ distance |
| --- | --- | --- |
| 10 | 0.171835 | 1.7184 |
| 100 | 0.012598 | 1.2598 |
| 1,000 | 0.001228 | 1.2276 |

: Total variation distance between $\mathrm{Bin}(n, 5/n)$ and $\mathrm{Poisson}(5)$. {#tbl-sol-distributions-tv}

3. **Read off the rate.** Each tenfold increase in $n$ cuts the distance by about ten, and $n$ times the distance settles near 1.23. So the distance is roughly $1.23/n$, which confirms the check.

**Why $1/n$.** The two distributions have the same mean $\mu = 5$, but the binomial variance is $np(1 - p) = \mu - \mu^2/n$, smaller than the Poisson's $\mu$ by $\mu^2/n = 25/n$. The binomial is narrower by an amount of order $1/n$, which is the gap @fig-poisson-limit shows at $n = 10$.

(The chapter's `binomial_pmf` evaluates $\binom nk$ as an exact integer and then converts it to a float. That conversion overflows once $n$ is a little above 1,000, so going further, to $n = 10{,}000$, would need the PMF in log space, as `poisson_pmf` already does.)

*The Poisson approximation to $\mathrm{Bin}(n, \mu/n)$ is off by an amount of order $1/n$, because the two variances differ by $\mu^2/n$.*
:::

## Median from absolute loss {#sec-sol-dist-4}

> **Exercise 4.** **Median from absolute loss.** Show that $\sum_i|y_i - m|$ is minimized at any median of $y_1, \dots, y_n$. *Hint: look at the slope in $m$.* *(Laplace)*

::: {.callout-tip collapse="true" title="Solution"}
Let $f(m) = \sum_i|y_i - m|$. A **median** is any $m$ with at most half the points strictly below it and at most half strictly above it: $\#\{i : y_i < m\} \le n/2$ and $\#\{i : y_i > m\} \le n/2$.

1. **Slope of one term.** As a function of $m$, $|y_i - m|$ has slope $-1$ for $m < y_i$ and $+1$ for $m > y_i$.

2. **Slope of the sum.** At any $m$ that is not one of the $y_i$, add the slopes:
$$f'(m) = \#\{i : y_i < m\} - \#\{i : y_i > m\}.$$

3. **The slope never decreases.** As $m$ moves right past a data point, that point moves from the "above" count to the "below" count, so the slope rises by 2 (more if several points coincide). So $f$ is continuous and piecewise linear, with slopes that only increase: it is convex.

4. **Where the minimum is.** A convex piecewise-linear function is minimal exactly at the points where the slope just to the left is $\le 0$ and the slope just to the right is $\ge 0$. At $m$, those one-sided slopes are
$$f'(m^-) = \#\{y_i < m\} - \#\{y_i \ge m\}, \qquad f'(m^+) = \#\{y_i \le m\} - \#\{y_i > m\}.$$

5. **Translate the two conditions.** Since $\#\{y_i \ge m\} = n - \#\{y_i < m\}$, the condition $f'(m^-) \le 0$ says $\#\{y_i < m\} \le n/2$. Since $\#\{y_i \le m\} = n - \#\{y_i > m\}$, the condition $f'(m^+) \ge 0$ says $\#\{y_i > m\} \le n/2$. Together they are the definition of a median. So $f$ is minimized exactly at the medians.

**A small case.** Take the data $1, 2, 7$, and then add a fourth point, 10. @tbl-sol-distributions-median-slopes gives the slope of $f$ on each interval.

| Interval of $m$ | Slope for $1, 2, 7$ | Slope for $1, 2, 7, 10$ |
| --- | --- | --- |
| $m < 1$ | $-3$ | $-4$ |
| $1 < m < 2$ | $-1$ | $-2$ |
| $2 < m < 7$ | $+1$ | $0$ |
| $7 < m < 10$ | $+3$ | $+2$ |
| $m > 10$ | $+3$ | $+4$ |

: Slope of $f(m) = \sum_i|y_i - m|$ between the data points, for three and for four data points. {#tbl-sol-distributions-median-slopes}

With three points the slope changes sign at $m = 2$, the middle value, where $f(2) = 1 + 0 + 5 = 6$. With four points the slope is 0 on $[2, 7]$, so every $m$ there is a minimizer with $f(m) = (m - 1) + (m - 2) + (7 - m) + (10 - m) = 14$, and every such $m$ is a median. Now replace 10 by 1,000. No slope in the first three rows changes sign, so the minimizers are still $[2, 7]$, while the mean jumps from 5 to 252.5. This is the robustness the chapter's absolute-error regression example describes.

*Minimizing absolute error balances the count of points on each side, not their distances, so a median is the minimizer.*
:::

## Laplace kurtosis {#sec-sol-dist-5}

> **Exercise 5.** **Laplace kurtosis.** Show that $\mathbb E X^4/(\mathbb E X^2)^2$ is 6 for the Laplace and 3 for the normal. What does the larger value say about the shape? *(Laplace, normal)*

::: {.callout-tip collapse="true" title="Solution"}
The ratio is the same for every location-0 member of each family, since scaling $X$ by $c$ multiplies both numerator and denominator by $c^4$. Take location 0.

**Laplace.** Let $X \sim \mathrm{Lap}(b)$.

1. **Use $|X| \sim \mathrm{Exp}(b)$.** Even powers do not see the sign, so $\mathbb E X^4 = \mathbb E|X|^4$ and $\mathbb E X^2 = \mathbb E|X|^2$.

2. **Exponential moments.** The chapter's Gamma integral (or its check-yourself question) gives $\mathbb E Y^k = k!\,b^k$ for $Y \sim \mathrm{Exp}(b)$. So $\mathbb E X^4 = 4!\,b^4 = 24b^4$ and $\mathbb E X^2 = 2b^2$.

3. **Divide.** $\mathbb E X^4/(\mathbb E X^2)^2 = 24b^4/(4b^4) = 6$.

**Normal.** Let $X \sim \mathcal N(0, \sigma^2)$.

1. **Expand the MGF.** With $\mu = 0$, $M_X(\lambda) = e^{\sigma^2\lambda^2/2} = 1 + \frac{\sigma^2\lambda^2}{2} + \frac{\sigma^4\lambda^4}{8} + \dots$, from $e^u = 1 + u + u^2/2 + \dots$ with $u = \sigma^2\lambda^2/2$.

2. **Read off the moments.** The coefficient of $\lambda^k$ in $M_X$ is $\mathbb E X^k/k!$, because $M_X^{(k)}(0) = \mathbb E X^k$. So $\mathbb E X^2 = 2!\cdot\frac{\sigma^2}{2} = \sigma^2$ and $\mathbb E X^4 = 4!\cdot\frac{\sigma^4}{8} = 3\sigma^4$.

3. **Divide.** $\mathbb E X^4/(\mathbb E X^2)^2 = 3\sigma^4/\sigma^4 = 3$.

**What the larger value says.** The fourth power weighs values far from 0 much more heavily than the square does. At equal variance, a ratio of 6 against 3 means the Laplace puts more of its mass far out. To keep the same variance it must then also put more mass near 0, and less in the shoulders in between. That is the sharp peak and fat tails of @fig-laplace-vs-normal. The chapter's check-yourself numbers show the tail side: $\Pr[|X| \ge 3b] = e^{-3} \approx 0.050$ for the Laplace against about 0.034 for the normal with the same variance.

*A larger fourth-moment ratio at the same variance means heavier tails and a sharper peak.*
:::

## Maximum of normals {#sec-sol-dist-6}

> **Exercise 6.** **Maximum of normals.** For $Z_1, \dots, Z_k \sim \mathcal N(0, 1)$ (not necessarily independent), show $\mathbb E\max_i Z_i \le \sqrt{2\ln k}$. *Hint: $e^{\lambda\mathbb E\max} \le \mathbb E e^{\lambda\max} \le \sum_i\mathbb E e^{\lambda Z_i} = ke^{\lambda^2/2}$, then optimize $\lambda$.* *(Normal MGF)*

::: {.callout-tip collapse="true" title="Solution"}
Take $k \ge 2$ and any $\lambda > 0$. Write $M = \max_i Z_i$.

1. **Jensen.** $u \mapsto e^{\lambda u}$ is convex, so $e^{\lambda\mathbb E M} \le \mathbb E e^{\lambda M}$. (This is the same inequality the chapter uses for the norm, with a convex function instead of a concave one.)

2. **Replace the max by a sum.** Since $\lambda > 0$, $e^{\lambda M} = \max_i e^{\lambda Z_i}$. Every term is positive, so the maximum is at most the sum: $e^{\lambda M} \le \sum_i e^{\lambda Z_i}$.

3. **Take expectations.** Linearity, which needs no independence, gives $\mathbb E e^{\lambda M} \le \sum_i\mathbb E e^{\lambda Z_i}$. Each $Z_i$ is $\mathcal N(0, 1)$, with MGF $e^{\lambda^2/2}$, so the sum is $ke^{\lambda^2/2}$.

4. **Take logs and divide by $\lambda$.** From steps 1 to 3, $\lambda\mathbb E M \le \ln k + \lambda^2/2$, so
$$\mathbb E M \le \frac{\ln k}{\lambda} + \frac\lambda2 \qquad\text{for every } \lambda > 0.$$

5. **Optimize $\lambda$.** The derivative of the right side is $-\ln k/\lambda^2 + 1/2$, which is zero at $\lambda = \sqrt{2\ln k}$ (positive because $k \ge 2$). There the two terms are equal, each $\sqrt{2\ln k}/2$, so
$$\mathbb E\max_i Z_i \le \sqrt{2\ln k}.$$

For $k = 1$ the claim reads $\mathbb E Z_1 \le 0$, which holds with equality.

**How good is it?** Only step 3 used the distribution, and only one $Z_i$ at a time, so the bound holds whatever the dependence. When the $Z_i$ are all the same variable, $\mathbb E M = \mathbb E Z_1 = 0$, and the bound is far off. When they are independent, @tbl-sol-distributions-max-normals compares it with simulation.

| $k$ | Simulated $\mathbb E\max_i Z_i$ | Bound $\sqrt{2\ln k}$ | Ratio |
| --- | --- | --- | --- |
| 2 | 0.557 | 1.177 | 0.47 |
| 10 | 1.547 | 2.146 | 0.72 |
| 100 | 2.505 | 3.035 | 0.83 |
| 1,000 | 3.242 | 3.717 | 0.87 |

: Mean of the maximum of $k$ independent $\mathcal N(0, 1)$ variables over 10,000 draws (seed 0), against the bound. {#tbl-sol-distributions-max-normals}

The $k = 2$ row has an exact answer. $\max(Z_1, Z_2) = \frac{Z_1 + Z_2}{2} + \frac{|Z_1 - Z_2|}{2}$, and $Z_1 - Z_2 \sim \mathcal N(0, 2)$, so $\mathbb E|Z_1 - Z_2| = \sqrt2\cdot\sqrt{2/\pi}$ and $\mathbb E\max = 1/\sqrt\pi = 0.564$, consistent with the simulated 0.557. The bound is loose by a factor of about 2 at $k = 2$, but the ratio climbs toward 1 as $k$ grows: for independent normals, $\sqrt{2\ln k}$ is the right growth rate.

*The maximum of $k$ standard normals grows only like $\sqrt{2\ln k}$, whatever their dependence, because the normal MGF is $e^{\lambda^2/2}$.*
:::

## Random vectors are nearly orthogonal {#sec-sol-dist-7}

> **Exercise 7.** **Random vectors are nearly orthogonal.** For independent $X, Y \sim \mathcal N(0, I_d)$, show $\langle X, Y\rangle/(\|X\|\|Y\|)$ is of order $1/\sqrt d$. Check by simulation for $d = 10, 1000$. *(Rotation invariance, norm concentration)*

::: {.callout-tip collapse="true" title="Solution"}
Let $\rho = \langle X, Y\rangle/(\|X\|\|Y\|)$, the cosine of the angle between $X$ and $Y$.

1. **Fix $Y$.** Condition on $Y$ and let $u = Y/\|Y\|$, a fixed unit vector. Since $X$ is independent of $Y$, $X$ is still $\mathcal N(0, I_d)$, and
$$\rho = \frac{\langle u, X\rangle}{\|X\|}.$$

2. **Numerator: rotation invariance.** $\langle u, X\rangle \sim \mathcal N(0, 1)$ for any unit $u$. Its distribution does not depend on $u$, so it does not depend on $Y$ either.

3. **Denominator: norm concentration.** $\|X\| = \sqrt d\,\big(1 \pm O(1/\sqrt d)\big)$, by the chapter's norm result.

4. **Combine.** $\rho \approx \langle u, X\rangle/\sqrt d$: a standard normal divided by $\sqrt d$. So $\rho$ is typically of size $1/\sqrt d$.

**The exact standard deviation.** Steps 3 and 4 are approximate, but the second moment is exact.

5. **Turn $u$ into the first coordinate.** The $\mathcal N(0, I_d)$ density $(2\pi)^{-d/2}e^{-\|x\|^2/2}$ depends on $x$ only through $\|x\|$. A rotation $R$ that maps $u$ to $e_1 = (1, 0, \dots, 0)$ preserves norms and inner products, so $RX$ has the same distribution as $X$, and $\rho = \langle Ru, RX\rangle/\|RX\| = \langle e_1, RX\rangle/\|RX\|$. So $\rho$ has the same distribution as $X_1/\|X\|$. In $d = 2$, for example, $R$ turns the plane until $u$ points along the first axis, and the circular contours of the density don't move.

6. **Symmetry among coordinates.** The coordinates of $X$ are independent and identically distributed, so $\mathbb E[X_i^2/\|X\|^2]$ is the same for every $i$. These $d$ numbers add up to $\mathbb E[\sum_iX_i^2/\|X\|^2] = 1$, so each is $1/d$.

7. **Mean and standard deviation.** $X$ and $-X$ have the same distribution, so $\mathbb E \rho = 0$, and $\operatorname{Var}(\rho) = \mathbb E \rho^2 = 1/d$. The standard deviation of $\rho$ is exactly $1/\sqrt d$.

**Simulation.** @tbl-sol-distributions-cosines reports 10,000 pairs for each $d$.

| $d$ | Mean of $\rho$ | Std of $\rho$ | $1/\sqrt d$ | $\sqrt d \times$ std | Largest $\lvert \rho\rvert$ |
| --- | --- | --- | --- | --- | --- |
| 10 | 0.0021 | 0.3178 | 0.3162 | 1.005 | 0.908 |
| 1,000 | −0.0003 | 0.0315 | 0.0316 | 0.997 | 0.121 |

: Cosine $\rho$ between independent $X, Y \sim \mathcal N(0, I_d)$, over 10,000 pairs (seed 0). {#tbl-sol-distributions-cosines}

The standard deviation matches $1/\sqrt d$ to within 1%. At $d = 10$ some pairs are nearly parallel (largest $|\rho|$ 0.908). At $d = 1{,}000$, none of the 10,000 pairs has $|\rho|$ above 0.121, an angle within 7.0° of a right angle.

*In high dimension, two independent random directions are almost orthogonal: their cosine has standard deviation exactly $1/\sqrt d$.*
:::
