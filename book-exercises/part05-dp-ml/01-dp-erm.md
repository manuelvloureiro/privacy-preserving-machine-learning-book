# Exercises: Private empirical risk minimization

Exercises for [Private empirical risk minimization](../../book-full/part05-dp-ml/01-dp-erm.md), each with a worked solution. Try each exercise before opening its solution. The script `code/part05-dp-ml/solutions_dp_erm.py` computes every number quoted here; run it with `uv run python code/part05-dp-ml/solutions_dp_erm.py`.

The solutions use the chapter's notation. Empirical risk minimization (ERM) picks the weights $w^*$ that minimize $J(w; D) = \frac1n\sum_i\ell(w, z_i) + \frac\lambda2\|w\|^2$. Differential privacy (DP) is required for replace-one neighbours. $L$ bounds the length of each record's loss gradient, and $\mu(\varepsilon, \delta)$ is the largest ratio of sensitivity to noise standard deviation for which the Gaussian mechanism is $(\varepsilon, \delta)$-DP.

## Exact against classic calibration {#sec-sol-dperm-1}

> **Exercise 1.** **Exact against classic calibration.** Output perturbation adds Gaussian noise to the minimizer. Compare the noise the classic formula $\sigma = \Delta\sqrt{2\ln(1.25/\delta)}/\varepsilon$ prescribes with the exact calibration of @eq-dperm-gauss-delta at $\delta = 10^{-5}$ and $\varepsilon = 0.25$, 0.5 and 1. By what factor does the excess risk of output perturbation grow if you use the classic formula? *Check: the noise ratio is 1.299 at $\varepsilon = 1$.* *(Gaussian calibration)*

::: {.callout-tip collapse="true" title="Solution"}
1. **Classic.** $\sqrt{2\ln(1.25/10^{-5})} = \sqrt{2 \times 11.736} = 4.8448$, so $\sigma/\Delta = 4.8448/\varepsilon$: 19.3792, 9.6896 and 4.8448 for the three budgets.
2. **Exact.** Solve @eq-dperm-gauss-delta for the largest $\mu$ with $\delta(\varepsilon) \le 10^{-5}$ by bisection, then $\sigma/\Delta = 1/\mu$. The script gives 13.2855, 7.0318 and 3.7306.
3. **Ratios.** $19.3792/13.2855 = 1.459$, $9.6896/7.0318 = 1.378$ and $4.8448/3.7306 = 1.299$. The classic formula is looser at small $\varepsilon$, where its $1/\varepsilon$ scaling overshoots the exact curve.
4. **Excess risk.** By @eq-dperm-output-risk the excess risk is proportional to $\sigma^2$, so it grows by the square of the ratio: $1.299^2 \approx 1.69$ at $\varepsilon = 1$ and $1.459^2 \approx 2.13$ at $\varepsilon = 0.25$.

*The classic formula is valid, but it throws away 30% to 46% of the noise budget, and twice that in excess risk. Since the exact calibration is one bisection, there is no reason not to use it.*
:::

## Ridge regression on a ball {#sec-sol-dperm-2}

> **Exercise 2.** **Ridge regression on a ball.** The squared loss $\ell(w, (x, y)) = \frac12(\langle w, x\rangle - y)^2$ has no global gradient bound. Suppose $\|x\| \le 1$, $|y| \le 1$, and the minimizer is computed over the ball $\|w\| \le R$. Find a Lipschitz constant $L$ on the ball and the output-perturbation noise for $R = 2$, $\lambda = 0.1$, $n = 1{,}000$, $\varepsilon = 1$, $\delta = 10^{-5}$. Why does restricting to the ball not break Fact 2? *Check: $L = 3$, sensitivity 0.06, $\sigma = 0.2238$.* *(Lipschitz constants on a bounded domain)*

::: {.callout-tip collapse="true" title="Solution"}
1. **The gradient.** $\nabla_w\ell = (\langle w, x\rangle - y)x$. Its length is $|\langle w, x\rangle - y|\,\|x\|$.
2. **Bound it on the ball.** By Cauchy–Schwarz $|\langle w, x\rangle| \le \|w\|\|x\| \le R$, so $|\langle w, x\rangle - y| \le R + 1$, and the length is at most $(R + 1) \cdot 1 = 3$ for $R = 2$.
3. **Sensitivity.** Fact 2 gives $2L/(\lambda n) = 6/100 = 0.06$, and the exact calibration gives $\sigma = 3.7306 \times 0.06 = 0.2238$.
4. **Why the ball is fine.** Fact 2's argument works for the minimizer over any convex set: strong convexity still converts a bound on the gradient difference into a bound on the distance between the constrained minimizers, using the first-order optimality condition $\langle\nabla J(w^*), v - w^*\rangle \ge 0$ for all $v$ in the set in place of $\nabla J(w^*) = 0$. The set must be fixed in advance, not chosen from the data.

*Losses without a global gradient bound become Lipschitz on a bounded domain, and the domain's radius enters the noise. Choosing $R$ too large costs noise, too small biases the solution: the same trade-off as clipping.*
:::

## Objective perturbation's budget {#sec-sol-dperm-3}

> **Exercise 3.** **Objective perturbation's budget.** For logistic regression ($c = \frac14$) with $\lambda = 0.01$ and $\varepsilon = 0.1$, find the smallest $n$ for which @eq-dperm-eps-prime gives $\varepsilon' > 0$ without extra ridge. Compute $\varepsilon'$ at that $n$ and at $n - 1$, where the extra ridge is used. What is wrong with the rule, and how would you fix it? *Check: $n = 488$; $\varepsilon' = 0.00008$ at $n = 488$ but $0.05000$ at $n = 487$.* *(Splitting a privacy budget)*

::: {.callout-tip collapse="true" title="Solution"}
1. **The condition.** $\varepsilon' > 0$ means $2\ln(1 + c/(n\lambda)) < \varepsilon$, that is $c/(n\lambda) < e^{\varepsilon/2} - 1$, or $n > c/(\lambda(e^{\varepsilon/2} - 1))$.
2. **Numbers.** $e^{0.05} - 1 = 0.051271$, so $n > 0.25/(0.01 \times 0.051271) = 487.6$, and the smallest integer is 488.
3. **At $n = 488$.** The Jacobian takes $2\ln(1 + 25/488) = 0.09992$ of the budget, so the noise gets $\varepsilon' = 0.00008$: the noise vector has mean length $d \cdot 2/\varepsilon' = 25{,}000\,d$, and the output is useless.
4. **At $n = 487$.** Now $\varepsilon' \le 0$, so the rule adds ridge until the Jacobian takes exactly $\varepsilon/2$, and the noise gets $\varepsilon' = 0.05$, with an extra ridge of 0.0103.
5. **The fix.** The rule's switch is in the wrong place. Add extra ridge whenever $\varepsilon'$ would fall below $\varepsilon/2$, not only when it would be negative. Then the noise always gets at least half the budget, and one more record never makes the output worse.

*A budget split that hands the remainder to the noise can leave it almost nothing. Whenever a guarantee is a sum of two terms, check what happens at the boundary where one of them eats the budget.*
:::

## Pure against Gaussian noise in $d$ dimensions {#sec-sol-dperm-4}

> **Exercise 4.** **Pure against Gaussian noise in $d$ dimensions.** Output perturbation can use Gaussian noise for $(\varepsilon, \delta)$-DP or noise with density $\propto e^{-\|b\|\varepsilon/\Delta}$ for $\varepsilon$-DP. Using Fact 3, find the dimension $d$ above which the Gaussian has smaller mean squared length, at $\varepsilon = 1$, $\delta = 10^{-5}$. Repeat at $\varepsilon = 0.1$. *Check: $d = 13$ at $\varepsilon = 1$; $d = 9$ at $\varepsilon = 0.1$.* *(Moments of the gamma distribution)*

::: {.callout-tip collapse="true" title="Solution"}
1. **The two second moments.** With $s = \Delta/\varepsilon$, Fact 3 gives $\mathbb E\|b\|^2 = d(d+1)\Delta^2/\varepsilon^2$. The Gaussian gives $d\sigma^2$, with $\sigma = \Delta/\mu(\varepsilon, \delta)$.
2. **Compare.** The Gaussian is smaller when $d\sigma^2 < d(d + 1)\Delta^2/\varepsilon^2$, that is $d + 1 > \varepsilon^2\sigma^2/\Delta^2$.
3. **At $\varepsilon = 1$.** $\sigma^2/\Delta^2 = 3.7306^2 = 13.918$, so $d + 1 > 13.918$, $d \ge 13$. At $d = 12$ the pure noise has 156 against 167.0; at $d = 13$, 182 against 180.9. At $d = 100$ it is 10,100 against 1,391.8.
4. **At $\varepsilon = 0.1$.** $\varepsilon^2\sigma^2/\Delta^2 = 9.455$, so $d \ge 9$.

*Pure DP's noise grows like $d^2$ in squared length, the Gaussian's like $d$. In more than about ten dimensions the small $\delta$ is worth paying.*
:::

## Noisy gradient descent with weaker composition {#sec-sol-dperm-5}

> **Exercise 5.** **Noisy gradient descent with weaker composition.** For noisy gradient descent at $n = 10{,}000$ with $(\varepsilon, \delta) = (1, 10^{-5})$, compute the per-step noise on the mean gradient for $T = 10$, 100 and 1,000 steps under three accountings: exact composition of Gaussians (Fact 5); basic composition, giving each step $(\varepsilon/T, \delta/T)$; and advanced composition, $\varepsilon = \varepsilon_0\sqrt{2T\ln(1/\delta')} + T\varepsilon_0(e^{\varepsilon_0} - 1)$ with $\delta' = \delta/2$ and each step $\delta/(2T)$. *Check: at $T = 100$, exact 0.00746, basic 0.07240, advanced 0.04032.* *(Composition theorems)*

::: {.callout-tip collapse="true" title="Solution"}
1. **Exact.** Each step has sensitivity $2/n = 2 \times 10^{-4}$, and @eq-dperm-noisy-gd gives $\sigma = 2 \times 10^{-4}\sqrt T/0.2681$: 0.00236, 0.00746 and 0.02359.
2. **Basic.** Each step must be $(1/T, 10^{-5}/T)$-DP on its own, and the exact single-step calibration gives 0.00726, 0.07240 and 0.72383: 3.08, 9.70 and 30.68 times the exact noise. The ratio grows like $\sqrt T$, because basic composition adds the $\varepsilon$s, while the exact composition adds their squares.
3. **Advanced.** Solve for the per-step $\varepsilon_0$ by bisection and calibrate each step for $(\varepsilon_0, \delta/(2T))$: 0.01194, 0.04032 and 0.13534, that is 5.06, 5.40 and 5.74 times the exact noise. It has the right $\sqrt T$ growth but a worse constant, and at $T = 10$ it is worse than basic composition.
4. **In excess risk.** The noise floor of @eq-dperm-gd-risk is proportional to $\sigma^2$, so at $T = 100$ advanced composition multiplies it by $5.40^2 \approx 29$ and basic composition by $9.70^2 \approx 94$.

*The accountant is part of the algorithm. The same noisy gradient descent, accounted for with a loose theorem, needs several times the noise for the same guarantee; [Privacy accounting](../../book-full/part04-privacy-accounting/00-introduction.md) is about closing exactly this gap.*
:::

## A sharper Jacobian bound {#sec-sol-dperm-6}

> **Exercise 6.** **A sharper Jacobian bound.** In step 5 of the objective-perturbation derivation, $M = \alpha\tilde x'\tilde x'^\top - \beta\tilde x\tilde x^\top$ with $\alpha, \beta \in [0, c]$. Show that $M$ has at most one positive and at most one negative eigenvalue, and conclude that $\det(I + M)$ lies in $[1/(1 + a), 1 + a]$ with $a = c/(n\lambda)$ once both directions are considered. So the Jacobian costs only $\ln(1 + a)$, half of what @eq-dperm-eps-prime charges. Check the bound on random instances. *Check: over 2,000 random instances ($d = 5$, $n = 20$, $\lambda = 0.05$) the largest $|\ln\det|$ is 0.9759 times $\ln(1 + a)$; an extreme instance attains 1.0000.* *(Eigenvalue interlacing)*

::: {.callout-tip collapse="true" title="Solution"}
1. **At most one positive eigenvalue.** $M \preceq \alpha\tilde x'\tilde x'^\top$, because the difference $\beta\tilde x\tilde x^\top$ is positive semi-definite. By the Courant–Fischer characterization, the second-largest eigenvalue of $M$ is at most that of $\alpha\tilde x'\tilde x'^\top$, which is 0. So only the largest eigenvalue $\lambda_1$ can be positive, and it is at most $\alpha\|\tilde x'\|^2 \le c/(n\lambda) = a$.
2. **At most one negative eigenvalue.** The same argument applied to $-M \preceq \beta\tilde x\tilde x^\top$ gives one negative eigenvalue $\lambda_2 \ge -a$ at most.
3. **The determinant.** $\det(I + M) = (1 + \lambda_1)(1 + \lambda_2)$ with $\lambda_1 \in [0, a]$ and $\lambda_2 \in [-a, 0]$, so it lies in $[1 - a, 1 + a]$. Exchanging $D$ and $D'$ bounds the reciprocal the same way, so $\det(I + M) \ge 1/(1 + a)$, which is larger than $1 - a$. Hence $|\ln\det(I + M)| \le \ln(1 + a)$.
4. **Check.** The script draws 2,000 random instances with curvatures in $[0, c]$ and features of length at most 1. The largest $|\ln\det|$ is 0.9759 times $\ln(1 + a)$, and an instance where the other records add no curvature and the new record adds $c$ along an axis attains it exactly, 1.0000.
5. **What it buys.** In the chapter's one-dimensional example ($n = 5$, $\lambda = 0.1$), $\ln(1 + a) = \ln 1.5 = 0.4055$ instead of 0.8109, so the noise could get $\varepsilon' = 1 - 0.4055 = 0.5945$ instead of 0.1891, three times as much, and the noise length would shrink threefold.

*Bounds that charge each of two terms its worst case separately are safe but can be loose when the two worst cases point in opposite directions. Here the record that leaves and the record that arrives push the determinant in opposite directions, and only one can win.*
:::

## How tight must an approximate minimizer be? {#sec-sol-dperm-7}

> **Exercise 7.** **How tight must an approximate minimizer be?** Using @eq-dperm-approx, find the largest gradient tolerance $g$ for which an approximate minimizer's sensitivity is within 10% of the exact $2L/(\lambda n)$. Evaluate it for $n = 10{,}000$, $L = 1$. Does it depend on $\lambda$? *Check: $g = 10^{-5}$.* *(Strong convexity: small gradient, close to the minimizer)*

::: {.callout-tip collapse="true" title="Solution"}
1. **The condition.** The extra term $2g/\lambda$ must be at most $0.1 \times 2L/(\lambda n)$.
2. **Solve.** Multiply both sides by $\lambda/2$: $g \le 0.1L/n$. The $\lambda$ cancels.
3. **Numbers.** $g = 0.1/10{,}000 = 10^{-5}$.
4. **Interpretation.** The tolerance shrinks like $1/n$: the more data, the more precisely you must solve. For Newton's method this is cheap, a few extra steps; for stochastic gradient descent (SGD) it can be the dominant cost.

*An approximate solver changes the sensitivity, not just the accuracy. Stop on a gradient check at a tolerance tied to $L/n$, and the privacy proof survives.*
:::
