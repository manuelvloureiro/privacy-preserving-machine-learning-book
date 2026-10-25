# Exercises: Hypothesis testing

Exercises for [Hypothesis testing](../../book-full/part01-prerequisites/04-hypothesis-testing.md), each with a worked solution. Try each exercise before opening its solution. The notation is the chapter's: $T = T(P, Q)$ is the trade-off function, $\Phi$ the standard normal cumulative distribution function (CDF), and the summaries are the total variation (TV), the true positive rate (TPR), the false positive rate (FPR), the receiver operating characteristic (ROC) curve and the area under the curve (AUC). Every number that is not a one-line hand calculation comes from [code/solutions_hypothesis_testing.py](../../code/part01-prerequisites/solutions_hypothesis_testing.py), which you run with `uv run python code/part01-prerequisites/solutions_hypothesis_testing.py`.

## Symmetry of the Gaussian curve {#sec-sol-ht-1}

> **Exercise 1.** Show that $T(T(a)) = a$ for @eq-ht-gaussian-tradeoff. *Hint: use $1 - \Phi(x) = \Phi(-x)$.* *Check: at $\mu = 1$, $T(0.05) = 0.7405$ and $T(0.7405) = 0.0500$.* *(Gaussian trade-off)*

::: {.callout-tip collapse="true" title="Solution"}
Fix $a \in (0, 1)$. At the ends, $T(0) = 1$ and $T(1) = 0$, so $T(T(0)) = 0$ and $T(T(1)) = 1$ directly.

1. **Name the threshold.** Let $z = \Phi^{-1}(1 - a)$. By @eq-ht-gaussian-tradeoff, $b = T(a) = \Phi(z - \mu)$.
2. **Compute $1 - b$.** By the hint (Fact 4 of the chapter), $1 - b = 1 - \Phi(z - \mu) = \Phi(\mu - z)$.
3. **Invert.** $\Phi$ is strictly increasing, so step 2 gives $\Phi^{-1}(1 - b) = \mu - z$.
4. **Apply $T$ again.** By @eq-ht-gaussian-tradeoff at $b$, $T(b) = \Phi\big(\Phi^{-1}(1 - b) - \mu\big) = \Phi(\mu - z - \mu) = \Phi(-z)$.
5. **Use the hint once more.** $\Phi(-z) = 1 - \Phi(z) = 1 - (1 - a) = a$.

**The check, step by step.** At $\mu = 1$ and $a = 0.05$: $z = 1.6449$, so $b = \Phi(0.6449) = 0.7405$. Then $\Phi^{-1}(1 - 0.7405) = \Phi^{-1}(0.2595) = -0.6449 = \mu - z$, and $T(0.7405) = \Phi(-0.6449 - 1) = \Phi(-1.6449) = 0.0500$. The script also evaluates $|T(T(a)) - a|$ on 999 values of $a$ for $\mu = 0.5, 1, 2, 3$: every defect is below $10^{-9}$, which is rounding error.

*Geometrically, the point $(a, b)$ is on the curve exactly when $(b, a)$ is: the Gaussian curve is symmetric about the diagonal $a = b$. With Exercise 2, this says that swapping which normal is the null hypothesis does not make the problem easier or harder.*
:::

## Swapping the hypotheses {#sec-sol-ht-2}

> **Exercise 2.** Show that if $T(P, Q)$ is strictly decreasing on $[0, 1]$, then $T(Q, P)$ is its inverse function. *Hint: a test $\phi$ for $P$ against $Q$ with errors $(a, b)$ gives the test $1 - \phi$ for $Q$ against $P$ with errors $(b, a)$.* *(Trade-off functions)*

::: {.callout-tip collapse="true" title="Solution"}
Write $f = T(P, Q)$ and $g = T(Q, P)$.

1. **The hint.** In the problem "$Q$ against $P$", the null hypothesis is $Q$ and the alternative is $P$. By Fact 1, the test $\phi' = 1 - \phi$ has type I error $\mathbb E_Q[\phi'] = 1 - \mathbb E_Q[\phi] = b_\phi$ and type II error $1 - \mathbb E_P[\phi'] = \mathbb E_P[\phi] = a_\phi$. The map $\phi \mapsto 1 - \phi$ is a bijection between tests, so the achievable pairs of the swapped problem are exactly the pairs $(b, a)$ with $(a, b)$ achievable in the original.
2. **What $f$ looks like.** $f$ is continuous (property 4 of @sec-ht-tradeoff), strictly decreasing by assumption, and $f(1) = 0$ (property 2). So $f$ maps $[0, 1]$ one-to-one onto $[0, f(0)]$, and for each $b \in [0, f(0)]$ there is exactly one $a = f^{-1}(b)$ with $f(a) = b$.
3. **$g(b) \le a$.** By the Neyman–Pearson lemma, the infimum in @eq-ht-tradeoff is attained: some test has errors $(a_\phi, b_\phi) = (a, f(a)) = (a, b)$ for $P$ against $Q$ (for $a = 0$, the test $\mathbf 1[p = 0]$). By step 1, $1 - \phi$ has type I error $b$ and type II error $a$ for $Q$ against $P$. So $g(b) \le a$.
4. **$g(b) \ge a$.** Take any test of the swapped problem with type I error at most $b$. By step 1 it is $1 - \phi$ for a test $\phi$ of the original problem with $b_\phi \le b$, and its type II error is $a_\phi$. If $a_\phi < a$, then $f(a_\phi) > f(a) = b$ because $f$ is strictly decreasing, while $f(a_\phi) \le b_\phi \le b$ by the definition of $f$ as an infimum. That is a contradiction, so $a_\phi \ge a$, and taking the infimum, $g(b) \ge a$.
5. **Conclusion.** $g(b) = f^{-1}(b)$ for every $b \in [0, f(0)]$.

**A boundary case.** If $f(0) < 1$ ($Q$ puts mass where $P$ has none), $f^{-1}$ is defined only on $[0, f(0)]$. For $b > f(0)$, the size-0 test of the original problem, flipped, has type I error $f(0) \le b$ and type II error 0, so $g(b) = 0$ there. "$g$ is the inverse of $f$" then means: $g = f^{-1}$ on $[0, f(0)]$ and $g = 0$ beyond. When $f(0) = 1$, as for every pair in the chapter, $g = f^{-1}$ on all of $[0, 1]$.

**Concrete.** Take one bit with $P = (0.5, 0.5)$ and $Q = (0.2, 0.8)$ on $\{0, 1\}$, which is not symmetric like the spam example. For $P$ against $Q$ the likelihood ratios are $q/p = 0.4$ at 0 and $1.6$ at 1, so the likelihood ratio (LR) tests reject nothing, then $\{1\}$, then everything. For $Q$ against $P$ the ratios are $p/q = 2.5$ at 0 and $0.625$ at 1, so the LR tests reject nothing, then $\{0\}$, then everything. The corners of the two curves are in @tbl-sol-hypothesis-testing-swap. Each curve is piecewise linear between its corners (Fact 3), and the corners of one are those of the other with the coordinates swapped, so each curve is the reflection of the other across the diagonal.

| Problem | Corners $(a, b)$ |
| --- | --- |
| $T(P, Q)$ | $(0, 1)$, $(0.5, 0.2)$, $(1, 0)$ |
| $T(Q, P)$ | $(0, 1)$, $(0.2, 0.5)$, $(1, 0)$ |

: Corners of the trade-off curves for $P = (0.5, 0.5)$ against $Q = (0.2, 0.8)$ and with the roles swapped. {#tbl-sol-hypothesis-testing-swap}

*Swapping null and alternative reflects the trade-off curve across the diagonal. Exercise 1 is the special case where the reflection gives back the same curve.*
:::

## Laplace total variation {#sec-sol-ht-3}

> **Exercise 3.** Compute $\frac12\int|q - p|$ directly for $\mathrm{Lap}(0, s)$ and $\mathrm{Lap}(m_1, s)$, with scale $s$ and shift $m_1$ as in the chapter, and confirm $1 - e^{-m_1/(2s)}$. *Check: 0.3935 at $m_1 = s$.* *(Total variation)*

::: {.callout-tip collapse="true" title="Solution"}
1. **Rescale.** Substitute $x = su$. The densities are $p(x) = \frac1{2s}e^{-|x|/s}$ and $q(x) = \frac1{2s}e^{-|x - m_1|/s}$, and $\mathrm dx = s\,\mathrm du$, so $\int|q - p|\,\mathrm dx = \int\big|\tfrac12e^{-|u - \kappa|} - \tfrac12e^{-|u|}\big|\,\mathrm du$ with $\kappa = m_1/s$. From here on $s = 1$ and the shift is $\kappa$.
2. **Where $q > p$.** $q(u) > p(u)$ exactly when $|u - \kappa| < |u|$, that is, when $u$ is closer to $\kappa$ than to 0: $u > \kappa/2$. So split the line at $0$, $\kappa/2$ and $\kappa$, where the absolute values change form.
3. **Left tail, $u \le 0$.** Here $p = \frac12e^u$ and $q = \frac12e^{u - \kappa}$, so $p - q = \frac12e^u(1 - e^{-\kappa})$ and $\int_{-\infty}^0(p - q)\,\mathrm du = \frac12(1 - e^{-\kappa})$.
4. **Right tail, $u \ge \kappa$.** Here $p = \frac12e^{-u}$ and $q = \frac12e^{-(u - \kappa)}$, so $q - p = \frac12e^{-u}(e^\kappa - 1)$ and $\int_\kappa^\infty(q - p)\,\mathrm du = \frac12e^{-\kappa}(e^\kappa - 1) = \frac12(1 - e^{-\kappa})$.
5. **Middle, $0 < u < \kappa$.** Here $p = \frac12e^{-u}$ and $q = \frac12e^{u - \kappa}$. On $(0, \kappa/2)$, $p > q$ and
$$\int_0^{\kappa/2}\tfrac12\big(e^{-u} - e^{u - \kappa}\big)\,\mathrm du = \tfrac12\big[(1 - e^{-\kappa/2}) - (e^{-\kappa/2} - e^{-\kappa})\big] = \tfrac12\big(1 - e^{-\kappa/2}\big)^2.$$
The substitution $u \mapsto \kappa - u$ swaps $p$ and $q$ and maps $(\kappa/2, \kappa)$ onto $(0, \kappa/2)$, so the integral over $(\kappa/2, \kappa)$ is the same.
6. **Add.** With $c = e^{-\kappa/2}$, so that $e^{-\kappa} = c^2$,
$$\int|q - p| = (1 - c^2) + (1 - c)^2 = 2 - 2c, \qquad \tfrac12\int|q - p| = 1 - e^{-\kappa/2} = 1 - e^{-m_1/(2s)}.$$

**The check.** At $m_1 = s$, $\kappa = 1$. @tbl-sol-hypothesis-testing-laplace-tv lists the four pieces. Their sum is 0.7869, half of it is 0.3935, and $1 - e^{-1/2} = 0.3935$. A trapezoid-rule integral of $\frac12|q - p|$ on 2,000,001 points also gives 0.3935.

| Piece | $u \le 0$ | $0 < u < \frac12$ | $\frac12 < u < 1$ | $u \ge 1$ | Total |
| --- | --- | --- | --- | --- | --- |
| $\int|q - p|$ | 0.3161 | 0.0774 | 0.0774 | 0.3161 | 0.7869 |

: The four pieces of $\int|q - p|$ for $\mathrm{Lap}(0, 1)$ against $\mathrm{Lap}(1, 1)$. {#tbl-sol-hypothesis-testing-laplace-tv}

*Most of the $L^1$ distance sits in the two tails, where the likelihood ratio is flat at $e^{\pm\kappa}$. The result matches the chapter's route through the trade-off curve, which maximized $1 - a - T(a)$ on the middle piece of @eq-ht-laplace-tradeoff.*
:::

## Sample size {#sec-sol-ht-4}

> **Exercise 4.** How many readings does the drift monitor need for power 0.9 at $a = 0.01$ when $h/\sigma = 0.25$? Rederive the formula, then halve $h$: by what factor does $n$ grow? *Check: $n = 209$; halving $h$ multiplies $n$ by 4.* *(Gaussian trade-off, many observations)*

::: {.callout-tip collapse="true" title="Solution"}
1. **Reduce to one Gaussian pair.** The mean of $n$ readings is $\mathcal N(0, \sigma^2/n)$ under "no drift" and $\mathcal N(h, \sigma^2/n)$ under "drift". The chapter showed it carries all the evidence, so the best test has the trade-off @eq-ht-gaussian-tradeoff with $\mu = h\sqrt n/\sigma$.
2. **Write the power target.** Power $1 - b'$ at level $a$ means $T(a) \le b'$, that is, $\Phi\big(\Phi^{-1}(1 - a) - \mu\big) \le b'$.
3. **Invert $\Phi$.** $\Phi$ is increasing, so this is $\Phi^{-1}(1 - a) - \mu \le \Phi^{-1}(b')$. By Fact 4, $\Phi^{-1}(b') = -\Phi^{-1}(1 - b')$, so the condition is
$$\mu \ge \Phi^{-1}(1 - a) + \Phi^{-1}(1 - b').$$
4. **Solve for $n$.** Substitute $\mu = h\sqrt n/\sigma$ and square both sides (both are positive):
$$n \ge \left(\frac{\Phi^{-1}(1 - a) + \Phi^{-1}(1 - b')}{h/\sigma}\right)^2.$$
5. **Plug in.** $a = 0.01$ and $b' = 0.1$ give $\Phi^{-1}(0.99) = 2.3263$ and $\Phi^{-1}(0.9) = 1.2816$. With $h/\sigma = 0.25$, $n \ge (3.6079/0.25)^2 = 208.3$, so $n = 209$. As a check, the power at $n = 208$ is 0.8996, just short of 0.9, and at $n = 209$ it is 0.9011.
6. **Halve $h$.** The bound in step 4 is proportional to $(\sigma/h)^2$, so halving $h$ multiplies it by exactly 4: $208.3 \to 833.1$, and $n = 834$. The rounded counts grow by $834/209 = 3.99$ because each is rounded up separately.

*Distinguishability grows like $\sqrt n$, so the number of readings grows like $1/h^2$. Detecting half the drift costs four times the data.*
:::

## A detector that AUC undersells {#sec-sol-ht-5}

> **Exercise 5.** Negatives score $\mathcal N(0, 1)$. Positives score $\mathcal N(0, 1)$ too, except a fraction $w = 0.05$ that scores $\mathcal N(6, 1)$. Show that $\mathrm{AUC} = (1 - w)\frac12 + w\,\Phi(6/\sqrt2)$ and find the TPR at FPR $10^{-3}$. Compare with a Gaussian-shift detector of the same AUC. *Check: AUC 0.5250 and TPR 0.0509; the equal-AUC shift is $\mu = 0.0887$, with TPR 0.0013.* *(AUC, ROC at low FPR)*

::: {.callout-tip collapse="true" title="Solution"}
**The AUC.**

1. **Ranking probability.** By @sec-ht-auc, $\mathrm{AUC} = \Pr[S_+ > S_-]$ for independent $S_+$ and $S_-$.
2. **Condition on the component.** Let $J$ be the component of the positive: "bulk" with probability $1 - w$, "far" with probability $w$. By the law of total probability, $\Pr[S_+ > S_-] = (1 - w)\Pr[S_+ > S_- \mid \text{bulk}] + w\Pr[S_+ > S_- \mid \text{far}]$.
3. **Bulk.** Given bulk, $S_+$ and $S_-$ are independent draws from the same continuous distribution. By symmetry $\Pr[S_+ > S_-] = \Pr[S_- > S_+]$, and ties have probability 0, so each is $\frac12$.
4. **Far.** Given far, $S_+ - S_- \sim \mathcal N(6, 2)$, the binormal case of @sec-ht-auc. So $\Pr[S_+ > S_-] = \Phi(6/\sqrt2) = 0.999989$.
5. **Add.** $\mathrm{AUC} = 0.95 \times 0.5 + 0.05 \times 0.999989 = 0.5250$. A simulation with 1,000,000 pairs gives 0.5259, within two standard errors (about 0.0005 each).

**The TPR at FPR $10^{-3}$.**

1. **Threshold.** Negatives are $\mathcal N(0, 1)$, so the threshold with FPR $10^{-3}$ is $\tau = \Phi^{-1}(0.999) = 3.0902$ (Fact 4).
2. **Split the TPR by component.** A bulk positive is distributed like a negative, so it passes with probability $10^{-3}$. A far positive passes with probability $\Pr[\mathcal N(6, 1) > \tau] = \Phi(6 - \tau)$. So
$$\mathrm{TPR} = (1 - w)\cdot10^{-3} + w\,\Phi(6 - 3.0902) = 0.00095 + 0.04991 = 0.0509.$$

**The equal-AUC shift detector.** A detector whose positives score $\mathcal N(\mu, 1)$ has $\mathrm{AUC} = \Phi(\mu/\sqrt2)$ (binormal model with $\sigma = 1$). Setting this to 0.5250 gives $\mu = \sqrt2\,\Phi^{-1}(0.5250) = 0.0887$. Its TPR at the same threshold is $\Pr[\mathcal N(\mu, 1) > 3.0902] = \Phi(\mu - 3.0902) = 0.0013$.

**Comparison.** The two detectors have the same AUC, 0.5250, barely above a coin flip. At FPR $10^{-3}$ the mixture detector catches 0.0509 of the positives and the shift detector 0.0013: 38 times fewer. The mixture's AUC is low because 95% of its positives are indistinguishable from negatives. Those pairs contribute $\frac12$ each and swamp the average. A strict threshold ignores them and catches nearly every far positive.

The threshold test is in fact the best possible test here. The likelihood ratio of the positive score against the negative one is $(1 - w) + w\,e^{6s - 18}$, which is increasing in $s$, so by the monotone-ratio rule of @sec-ht-np thresholding $s$ is an LR test.

*AUC averages over all FPRs. A detector that is excellent on a small subpopulation and blind elsewhere gets an AUC near 0.5 and can still dominate where false alarms must be rare.*
:::

## A curve that starts below 1 {#sec-sol-ht-6}

> **Exercise 6.** For $P = U[0, 1]$ and $Q = U[s, 1 + s]$ with $0 < s < 1$, show that $T(a) = \max(0, 1 - s - a)$. Which outcomes have $\Lambda = \infty$, $\Lambda = 1$ and $\Lambda = 0$? *Check: at $s = 0.2$, $T(0) = 0.8$ and $\mathrm{TV} = 0.2$.* *(Neyman–Pearson with $p = 0$, continuity at 0)*

::: {.callout-tip collapse="true" title="Solution"}
$p = 1$ on $[0, 1]$ and $q = 1$ on $[s, 1 + s]$, so the line splits into the three regions of @tbl-sol-hypothesis-testing-uniform.

| Region | $p$ | $q$ | $\Lambda = q/p$ | $P$-mass | $Q$-mass |
| --- | --- | --- | --- | --- | --- |
| $[0, s)$ | 1 | 0 | 0 | $s$ | 0 |
| $[s, 1]$ | 1 | 1 | 1 | $1 - s$ | $1 - s$ |
| $(1, 1 + s]$ | 0 | 1 | $\infty$ | 0 | $s$ |

: The three regions of the uniform pair. Outside $[0, 1 + s]$ both densities are 0, which has probability 0 under both hypotheses. {#tbl-sol-hypothesis-testing-uniform}

1. **The likelihood ratio.** $\Lambda = \infty$ on $(1, 1 + s]$, where only $Q$ can land; $\Lambda = 1$ on the overlap $[s, 1]$; and $\Lambda = 0$ on $[0, s)$, where only $P$ can land.
2. **$a = 0$.** A size-0 test must have $\phi = 0$ on almost all of $[0, 1]$, so it can only reject on $(1, 1 + s]$. Rejecting all of it gives power $s$. So $T(0) = 1 - s$, which is the chapter's $T(0) = 1 - \Pr_Q[p = 0]$.
3. **$0 < a \le 1 - s$.** Take the LR test @eq-ht-lr-test with $\tau = 1$: reject where $q > p$, which is $(1, 1 + s]$; on the tie set $q = p$, which is $[s, 1]$, reject with the constant probability $\gamma = a/(1 - s) \in (0, 1]$; keep $[0, s)$. Its size is $\gamma(1 - s) = a$ and its power is $s + \gamma(1 - s) = s + a$. By the Neyman–Pearson lemma it is the most powerful test of level $a$, so $T(a) = 1 - s - a$.
4. **$a > 1 - s$.** The test "reject $x \ge s$" has size $1 - s < a$ and power 1, so $T(a) = 0$.
5. **Combine.** Steps 2 to 4 give $T(a) = 1 - s - a$ for $a \le 1 - s$ and 0 beyond, which is $\max(0, 1 - s - a)$.

**The check.** At $s = 0.2$, $T(0) = 0.8$. By @eq-ht-tv, $\mathrm{TV} = \frac12\int|q - p| = \frac12(s + s) = s = 0.2$: the two regions where exactly one density is 1. A midpoint-rule integral gives 0.2000. A simulation of the randomized test of step 3 at $a = 0.3$, with 1,000,000 draws per hypothesis, gives size 0.3004 and power 0.5001, against the exact 0.3 and $1 - T(0.3) = 0.5$.

**Continuity at 0.** $T$ starts at $1 - s < 1$, but it is continuous there: $T(a) \to 1 - s = T(0)$ as $a \to 0$. This is property 4 of @sec-ht-tradeoff, and the $\mathbb E_P[\Lambda\,\mathbf 1[\Lambda > K]]$ term in its proof is 0 here because $\Lambda \le 1$ wherever $p > 0$. The curve is also a single segment of slope $-1$, so the advantage $1 - a - T(a) = s$ at every $a \le 1 - s$: every point of the segment attains the TV.

*Mass that only $Q$ can produce is free power: a test catches it with no false alarms, so the curve starts at $1 - \Pr_Q[p = 0]$ instead of 1.*
:::

## Bayes error and total variation {#sec-sol-ht-7}

> **Exercise 7.** Prove that with equal priors the smallest achievable error probability is $\frac12(1 - \mathrm{TV}(P, Q))$. *Check: 0.3085 for the Gaussian pair at $\mu = 1$.* *(Bayesian view, best advantage)*

::: {.callout-tip collapse="true" title="Solution"}
1. **Error of a test.** With prior $\pi = \frac12$, a test $\phi$ errs with probability $\frac12a_\phi + \frac12b_\phi$ (step 2 of the Bayesian derivation in @sec-ht-bayes).
2. **Rewrite with the advantage.** $\frac12(a_\phi + b_\phi) = \frac12\big(1 - (1 - a_\phi - b_\phi)\big)$. The error is a decreasing function of the advantage $1 - a_\phi - b_\phi$, so the test with the smallest error is the one with the largest advantage.
3. **Use the best advantage.** By @eq-ht-tv the largest advantage over all tests, randomized or not, is $\mathrm{TV}(P, Q)$, attained by rejecting where $q > p$. So the smallest error is $\frac12(1 - \mathrm{TV}(P, Q))$, attained by the same test, which is the Bayes rule $\Lambda(x) > (1 - \pi)/\pi = 1$.

**Concrete: the spam example.** $\mathrm{TV} = 0.40$, so the bound is $\frac12(1 - 0.40) = 0.30$. Of the 16 deterministic tests on $\{0, 1, 2, 3\}$, the script finds the smallest error 0.30, at the rejection region $\{2, 3\}$: $\frac12(0.3 + 0.3) = 0.30$ from @tbl-ht-spam-tests.

**The check.** For the Gaussian pair at $\mu = 1$, $\mathrm{TV} = 2\Phi(\frac12) - 1 = 0.3829$, so the smallest error is $\frac12(1 - 0.3829) = 1 - \Phi(\frac12) = 0.3085$. Minimizing $\frac12(a + T(a))$ over a grid of 100,001 values of $a$ gives the same 0.3085. It is also the optimal $a$ of @fig-ht-tv-advantage: at $\tau = 1$ the threshold is $\mu/2$, so $a = b = 1 - \Phi(\frac12)$, and their average is the same number.

*With equal priors, the best accuracy is $\frac12 + \frac12\mathrm{TV}$, which is @eq-tv-test. TV is exactly what a single balanced decision can extract.*
:::

## Errors decay at the KL rate {#sec-sol-ht-8}

> **Exercise 8.** For $n$ draws from $\mathcal N(0, 1)$ against $\mathcal N(1, 1)$ and a fixed $a = 0.05$, the optimal test has $b_n = \Phi(\Phi^{-1}(0.95) - \sqrt n)$. Compute $-\ln(b_n)/n$ and compare its limit with $D_{\mathrm{KL}}(P\|Q) = \mu^2/2$ from [Divergences](../../book-full/part01-prerequisites/03-divergences.md). This is the Chernoff–Stein lemma. *Check: 0.2740, 0.3796 and 0.4537 at $n = 10$, 100 and 1,000, approaching 0.5 slowly.* *(Many observations, Gaussian tails)*

::: {.callout-tip collapse="true" title="Solution"}
Here the Kullback–Leibler (KL) divergence is $\mu^2/2 = 0.5$ by @eq-gauss-renyi.

1. **Where $b_n$ comes from.** By the averaging rule of @sec-ht-gaussian, $n$ draws with $h = \sigma = 1$ give @eq-ht-gaussian-tradeoff with $\mu = \sqrt n$. At $a = 0.05$, with $z = \Phi^{-1}(0.95) = 1.6449$, that is $b_n = \Phi(z - \sqrt n) = \Phi(-x)$ with $x = \sqrt n - z$, which is positive for $n \ge 3$.
2. **Compute.** @tbl-sol-hypothesis-testing-stein gives the exact rate. It matches the check: 0.2740, 0.3796, 0.4537.
3. **A lower bound on the rate.** The Mills-ratio bound of [Concentration inequalities](../../book-full/part01-prerequisites/02-concentration.md) gives $\Phi(-x) = \Pr[Z \ge x] \le \varphi(x)/x$. Take $-\ln$:
$$-\ln b_n \ge \frac{x^2}2 + \ln x + \tfrac12\ln(2\pi).$$
4. **An upper bound on the rate.** $\varphi$ is decreasing on $[x, x + 1]$, so $\Phi(-x) \ge \int_x^{x + 1}\varphi(z)\,\mathrm dz \ge \varphi(x + 1)$. Take $-\ln$:
$$-\ln b_n \le \frac{(x + 1)^2}2 + \tfrac12\ln(2\pi).$$
5. **Divide by $n$ and let $n \to \infty$.** With $x = \sqrt n - z$, $\frac{x^2}{2n} = \frac12 - \frac z{\sqrt n} + \frac{z^2}{2n} \to \frac12$, and $\frac{(x + 1)^2}{2n} \to \frac12$ the same way. The $\ln x$ and constant terms divided by $n$ tend to 0. Both bounds tend to $\frac12$, so
$$\lim_{n \to \infty}\frac{-\ln b_n}{n} = \frac12 = D_{\mathrm{KL}}(P\|Q).$$
6. **Why so slowly.** The leading correction in step 5 is $-z/\sqrt n$, which shrinks only like $1/\sqrt n$. At $n = 1{,}000$ it is $-1.6449/31.62 = -0.052$, and most of it remains in the gap $0.5 - 0.4537 = 0.046$. A fixed $a$ costs a threshold $z$ that does not shrink with $n$, and that offset is what slows the approach.

| $n$ | $b_n$ | $-\ln(b_n)/n$ | Mills lower bound | Upper bound |
| --- | --- | --- | --- | --- |
| 10 | $6.458\times10^{-2}$ | 0.2740 | 0.2487 | 0.4088 |
| 100 | $3.268\times10^{-17}$ | 0.3796 | 0.3795 | 0.4468 |
| 1,000 | $9.520\times10^{-198}$ | 0.4537 | 0.4537 | 0.4807 |
| 10,000 | below double precision | | 0.4842 | 0.4937 |
| 100,000 | below double precision | | 0.4949 | 0.4980 |
| 1,000,000 | below double precision | | 0.4984 | 0.4994 |

: The exponent $-\ln(b_n)/n$ of the optimal size-0.05 test for $\mathcal N(0, 1)$ against $\mathcal N(1, 1)$, with the bounds of steps 3 and 4. Beyond $n = 1{,}000$ the value of $b_n$ underflows, but the bounds still pin the exponent down. {#tbl-sol-hypothesis-testing-stein}

The Mills lower bound is within $10^{-4}$ of the exact rate at $n = 100$ and agrees to four digits at $n = 1{,}000$, because the Mills ratio is asymptotically exact. Both bounds close in on 0.5.

*At a fixed level, the type II error of the best test decays like $e^{-n\,D_{\mathrm{KL}}(P\|Q)}$: KL divergence is the exponent of distinguishability. The convergence is slow, so read it as the rate for large $n$, not as a formula for $b_n$ at moderate $n$.*
:::

## Certifying a low rate {#sec-sol-ht-9}

> **Exercise 9.** How many negatives with zero false positives do you need to claim FPR $< 10^{-3}$ at 95% confidence? *Check: 2,995.* *(Clopper–Pearson, rule of three)*

::: {.callout-tip collapse="true" title="Solution"}
1. **The one-sided bound.** "FPR below $u$ at 95% confidence" is a one-sided Clopper–Pearson upper bound. Rerun the upper-end derivation of @sec-ht-estimation with $\beta/2$ replaced by $0.05$: if $u$ solves $\Pr[\mathrm{Bin}(n, u) \le k] = 0.05$, then $\Pr[p \ge u(K)] \le 0.05$.
2. **Zero counts.** At $k = 0$, $\Pr[\mathrm{Bin}(n, u) \le 0] = (1 - u)^n$, so $u$ solves $(1 - u)^n = 0.05$.
3. **The requirement.** You need $u \le 10^{-3}$. $(1 - u)^n$ is decreasing in $u$, so this holds exactly when $(1 - 10^{-3})^n \le 0.05$.
4. **Solve for $n$.** Take logs and divide by $\ln(1 - 10^{-3}) < 0$, which flips the inequality:
$$n \ge \frac{\ln 20}{-\ln(1 - 10^{-3})} = 2{,}994.2,$$
so $n = 2{,}995$.
5. **Check both sides.** $0.999^{2994} = 0.05001 > 0.05$ but $0.999^{2995} = 0.04996 \le 0.05$. The chapter's two-sided Clopper–Pearson routine at 90% has the same upper end as the one-sided 95% bound. At $k = 0$ it gives $1.0001\times10^{-3}$ for $n = 2{,}994$ and $9.9974\times10^{-4}$ for $n = 2{,}995$.

**As a hypothesis test.** Step 3 is a test of "FPR $\ge 10^{-3}$" against "FPR $< 10^{-3}$" that rejects the first when it sees no false positive. If the true FPR is at least $10^{-3}$, it sees none with probability at most $0.999^n$, its type I error. That is at most 0.05 from $n = 2{,}995$ on. The rule of three, $3/10^{-3} = 3{,}000$, is within five of it. If you report the two-sided 95% interval instead, its upper end uses $0.025$, and you need $\ln 40/(-\ln 0.999) = 3{,}687.0$, so 3,688 negatives.

*Certifying a rate of $10^{-3}$ needs about $3/10^{-3}$ clean trials, and that only bounds the rate. Measuring it to within about 20% needs on the order of $100/10^{-3} = 10^5$, over thirty times more.*
:::
