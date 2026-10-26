# Exercises: Optimization

Exercises for [Optimization](../../book-full/part01-prerequisites/06-optimization.md), each with a worked solution. Try each exercise before opening its solution. The solutions use gradient descent (GD), stochastic gradient descent (SGD) and regularized empirical risk minimization (ERM) exactly as the chapter defines them. The numbers come from the script `code/part01-prerequisites/solutions_optimization.py`; run it with `uv run python code/part01-prerequisites/solutions_optimization.py`.

## Constants of regularized logistic regression {#sec-sol-opt-1}

> **Exercise 1.** Show that $\ell(\theta) + \frac\lambda2\|\theta\|^2$, with $\ell$ the logistic loss and $\|x\| \le 1$, is $\lambda$-strongly convex and $(\frac14 + \lambda)$-smooth on $\mathbb R^d$, and $(1 + \lambda R)$-Lipschitz on the ball of radius $R$. *Check: for $\lambda = 0.1$ and $R = 5$, the constants are 0.1, 0.35 and 1.5.* *(Hessian bounds)*

::: {.callout-tip collapse="true" title="Solution"}
Write $f(\theta) = \ell(\theta) + \frac\lambda2\|\theta\|^2$ with $m = y\langle \theta, x\rangle$.

1. **Gradient and Hessian.** @exm-opt-logistic gives $\nabla\ell(\theta) = -\operatorname{sig}(-m)\,y\,x$ and $\nabla^2\ell(\theta) = \operatorname{sig}(m)(1 - \operatorname{sig}(m))\,xx^\top$, where $\operatorname{sig}(m) = 1/(1 + e^{-m})$ is the logistic function. The regularizer adds $\lambda \theta$ to the gradient and $\lambda I$ to the Hessian:
$$\nabla f(\theta) = -\operatorname{sig}(-m)\,y\,x + \lambda \theta, \qquad \nabla^2 f(\theta) = \operatorname{sig}(m)(1 - \operatorname{sig}(m))\,xx^\top + \lambda I.$$
2. **Eigenvalues of the Hessian.** $xx^\top$ has eigenvalue $\|x\|^2$ along $x$ and 0 in every direction orthogonal to $x$. Adding $\lambda I$ adds $\lambda$ to each. So the eigenvalues are $\lambda + \operatorname{sig}(m)(1 - \operatorname{sig}(m))\|x\|^2$ (once) and $\lambda$ (the rest).
3. **Strong convexity.** Since $\operatorname{sig}(m)(1 - \operatorname{sig}(m)) \ge 0$, every eigenvalue is at least $\lambda$, so $\nabla^2 f \succeq \lambda I$. The same conclusion follows without the Hessian from part (b) of @cor-opt-strong: $\ell$ is convex, and adding $\frac\lambda2\|\theta\|^2$ makes it $\lambda$-strongly convex.
4. **Smoothness.** $u(1 - u) \le \frac14$ for $u \in [0, 1]$ and $\|x\|^2 \le 1$, so every eigenvalue lies in $[\lambda, \frac14 + \lambda] \subset [-(\frac14 + \lambda), \frac14 + \lambda]$. By part (c) of @prp-opt-constants (a bounded Hessian gives smoothness), $f$ is $(\frac14 + \lambda)$-smooth on all of $\mathbb R^d$.
5. **Gradient on the ball.** By the triangle inequality, $\|\nabla f(\theta)\| \le \operatorname{sig}(-m)\|x\| + \lambda\|\theta\| \le 1 + \lambda R$ when $\|\theta\| \le R$, since $\operatorname{sig} < 1$, $|y| = 1$ and $\|x\| \le 1$.
6. **Lipschitz on the ball.** Step 1 of the proof of @prp-opt-constants integrates the gradient along the segment from $\theta$ to $\theta'$. The ball is convex, so for $\theta, \theta'$ in the ball the whole segment stays in it, where step 5 bounds the gradient by $1 + \lambda R$. So $|f(\theta) - f(\theta')| \le (1 + \lambda R)\|\theta - \theta'\|$.

For $\lambda = 0.1$ and $R = 5$: strong convexity $0.1$, smoothness $\frac14 + 0.1 = 0.35$, and Lipschitz constant $1 + 0.1 \times 5 = 1.5$, as the check says.

The Lipschitz constant is nearly attained. Take $\|x\| = 1$, $y = +1$ and $\theta = -Rx$, on the boundary of the ball. Then $m = -R$ and $\nabla f(\theta) = -\operatorname{sig}(R)x - \lambda Rx$, of norm $\operatorname{sig}(R) + \lambda R = 1.4933$ for $\lambda = 0.1$, $R = 5$.

*The regularizer shifts every curvature up by $\lambda$, which helps strong convexity and costs smoothness, and its gradient $\lambda \theta$ is what limits the Lipschitz constant to a bounded set.*
:::

## GD on a smooth convex function {#sec-sol-opt-2}

> **Exercise 2.** Show that GD with $\eta = 1/\beta$ on a convex $\beta$-smooth $\mathcal L$ satisfies $\mathcal L(\theta_{T+1}) - \mathcal L^* \le \beta\|\theta_1 - \theta^*\|^2/(2T)$. *Hint: combine the descent lemma with @lem-opt-tangent to get $\mathcal L(\theta_{t+1}) - \mathcal L^* \le \frac\beta2(\|\theta_t - \theta^*\|^2 - \|\theta_{t+1} - \theta^*\|^2)$, telescope, and use that $\mathcal L(\theta_t)$ decreases.* *(Descent lemma, telescoping)*

::: {.callout-tip collapse="true" title="Solution"}
Write $g_t = \nabla \mathcal L(\theta_t)$ and $a_t = \|\theta_t - \theta^*\|^2$, so $\theta_{t+1} = \theta_t - g_t/\beta$.

1. **Descent lemma.** By @prp-opt-gd-descent, $\mathcal L(\theta_{t+1}) \le \mathcal L(\theta_t) - \frac{1}{2\beta}\|g_t\|^2$.
2. **Convexity** (@lem-opt-tangent, @eq-opt-first-order with $\theta = \theta_t$, $\theta' = \theta^*$): $\mathcal L(\theta_t) \le \mathcal L^* + \langle g_t, \theta_t - \theta^*\rangle$. Substitute into step 1:
$$\mathcal L(\theta_{t+1}) - \mathcal L^* \le \langle g_t, \theta_t - \theta^*\rangle - \frac{1}{2\beta}\|g_t\|^2.$$
3. **Recognize a difference of squares.** Expand the square
$$a_{t+1} = \Big\|\theta_t - \theta^* - \frac{g_t}{\beta}\Big\|^2 = a_t - \frac2\beta\langle g_t, \theta_t - \theta^*\rangle + \frac{1}{\beta^2}\|g_t\|^2.$$
Multiply by $\beta/2$ and rearrange: $\frac\beta2(a_t - a_{t+1}) = \langle g_t, \theta_t - \theta^*\rangle - \frac{1}{2\beta}\|g_t\|^2$, which is the right side of step 2. So
$$\mathcal L(\theta_{t+1}) - \mathcal L^* \le \frac\beta2(a_t - a_{t+1}),$$
the inequality in the hint.
4. **Telescope** over $t = 1, \dots, T$: $\sum_{t=1}^T(\mathcal L(\theta_{t+1}) - \mathcal L^*) \le \frac\beta2(a_1 - a_{T+1}) \le \frac\beta2 a_1$.
5. **Monotonicity.** By step 1, $\mathcal L(\theta_{t+1}) \le \mathcal L(\theta_t)$, so $\mathcal L(\theta_{T+1})$ is the smallest of the $T$ terms on the left of step 4. Hence $T(\mathcal L(\theta_{T+1}) - \mathcal L^*) \le \frac\beta2\|\theta_1 - \theta^*\|^2$. Divide by $T$.

Unlike SGD, GD needs no averaging: the last iterate is the best one, because every step decreases $\mathcal L$.

How tight is it? The script runs GD with $\eta = 1/\beta = 4$ on unregularized logistic regression (a fresh dataset with $n = 1{,}000$, $d = 5$, drawn the same way as in the chapter; $\beta = \frac14$ is valid because each record's loss is $\frac14$-smooth). Here $\|\theta_1 - \theta^*\|^2 = \|\theta^*\|^2 = 10.849$.

| Steps $T$ | Observed $\mathcal L(\theta_{T+1}) - \mathcal L^*$ | Bound $\beta\Vert \theta_1 - \theta^*\Vert^2/(2T)$ |
| --- | --- | --- |
| 1 | $1.21\times10^{-1}$ | 1.36 |
| 3 | $6.02\times10^{-2}$ | 0.452 |
| 10 | $9.24\times10^{-3}$ | 0.136 |
| 30 | $1.67\times10^{-4}$ | 0.0452 |

: GD with $\eta = 1/\beta$ on logistic regression against the smooth convex bound. {#tbl-sol-opt-gd}

The bound in @tbl-sol-opt-gd holds and is loose by a factor of 11 at $T = 1$ and 270 at $T = 30$. As in the chapter's SGD record, the risk has curvature near $\theta^*$, so GD converges linearly there, which a bound that assumes only convexity cannot see.

*Smoothness alone gives GD a $1/T$ rate on convex functions, against $1/\sqrt T$ for SGD with noisy gradients.*
:::

## The $1/(\lambda T)$ rate is exact for a noisy quadratic {#sec-sol-opt-3}

> **Exercise 3.** Let $\mathcal L(\theta) = \frac\lambda2 \theta^2$ on $\mathbb R$ and $g_t = \lambda \theta_t + \xi_t$ with independent $\xi_t$, $\mathbb E\xi_t = 0$, $\mathbb E\xi_t^2 = \sigma^2$. With $\eta_t = 1/(\lambda t)$, show that $t\,\theta_{t+1} = (t - 1)\theta_t - \xi_t/\lambda$, hence $\theta_{T+1} = -\frac{1}{\lambda T}\sum_{t\le T}\xi_t$ and $\mathbb E \mathcal L(\theta_{T+1}) = \sigma^2/(2\lambda T)$. *Check: $\lambda = 1$, $\sigma^2 = 1$, $T = 100$ gives 0.005.* *(Strongly convex SGD)*

::: {.callout-tip collapse="true" title="Solution"}
1. **One step.** $\theta_{t+1} = \theta_t - \frac{1}{\lambda t}(\lambda \theta_t + \xi_t) = \big(1 - \frac1t\big)\theta_t - \frac{\xi_t}{\lambda t}$. Multiply by $t$: $t\,\theta_{t+1} = (t - 1)\theta_t - \xi_t/\lambda$.
2. **Telescope.** Put $v_t = (t - 1)\theta_t$. Step 1 says $v_{t+1} = v_t - \xi_t/\lambda$, and $v_1 = 0 \cdot \theta_1 = 0$. So $v_{T+1} = -\frac1\lambda\sum_{t \le T}\xi_t$, that is $T \theta_{T+1} = -\frac1\lambda\sum_{t\le T}\xi_t$. For $T = 2$: $\theta_2 = -\xi_1/\lambda$ and $2\theta_3 = \theta_2 - \xi_2/\lambda = -(\xi_1 + \xi_2)/\lambda$. The start $\theta_1$ is forgotten after the first step, whose learning rate $\eta_1 = 1/\lambda$ jumps straight to the minimizer of the noiseless part.
3. **Second moment.** The $\xi_t$ are independent with mean zero, so the cross terms of $(\sum_t\xi_t)^2$ have mean zero and $\mathbb E(\sum_t\xi_t)^2 = T\sigma^2$. Hence $\mathbb E \theta_{T+1}^2 = \frac{T\sigma^2}{\lambda^2T^2} = \frac{\sigma^2}{\lambda^2T}$.
4. **Risk.** $\mathbb E \mathcal L(\theta_{T+1}) = \frac\lambda2\mathbb E \theta_{T+1}^2 = \frac{\sigma^2}{2\lambda T}$. Here $\mathcal L^* = 0$, so this is the suboptimality.

For $\lambda = 1$, $\sigma^2 = 1$, $T = 100$: $1/200 = 0.005$. The script runs the recursion from $\theta_1 = 5$ with Gaussian noise, 100,000 runs:

| $T$ | Exact $\sigma^2/(2\lambda T)$ | Simulated $\mathcal L(\theta_{T+1})$ |
| --- | --- | --- |
| 100 | 0.0050 | $0.00499 \pm 2.2\times10^{-5}$ |
| 1,000 | 0.0005 | $0.00050 \pm 2.2\times10^{-6}$ |

: SGD with $\eta_t = 1/(\lambda t)$ on $\mathcal L(\theta) = \frac12\theta^2$ with unit-variance gradient noise; $\pm$ is one standard error. {#tbl-sol-opt-quadratic}

@tbl-sol-opt-quadratic matches the formula within the Monte Carlo error. The SGD iterate is exactly the minimizer of the noiseless part, $0$, plus the average of $T$ noise terms scaled by $1/\lambda$, and an average of $T$ independent terms has variance $\sigma^2/T$. No algorithm that only sees these noisy gradients can average the noise away faster.

*The $1/(\lambda T)$ dependence in @eq-opt-sgd-strong is not slack: on this problem, SGD's error is exactly of that order.*
:::

## Smoothness of the sigmoid loss {#sec-sol-opt-4}

> **Exercise 4.** Let $\operatorname{sig}(s) = 1/(1 + e^{-s})$ be the logistic function. Show that $|\operatorname{sig}''(s)| \le 1/(6\sqrt3)$ for all $s$, with equality where $\operatorname{sig}(s) = \frac12 \pm \frac{\sqrt3}{6}$. *Check: $1/(6\sqrt3) \approx 0.0962$.* *(Calculus of the logistic function)*

::: {.callout-tip collapse="true" title="Solution"}
1. **First derivative.** $\operatorname{sig}(s) = (1 + e^{-s})^{-1}$, so $\operatorname{sig}'(s) = \frac{e^{-s}}{(1 + e^{-s})^2} = \operatorname{sig}(s)\cdot\frac{e^{-s}}{1 + e^{-s}} = \operatorname{sig}(s)(1 - \operatorname{sig}(s))$.
2. **Second derivative.** Differentiate $\operatorname{sig}(1 - \operatorname{sig})$ by the product rule: $\operatorname{sig}'' = \operatorname{sig}'(1 - \operatorname{sig}) - \operatorname{sig}\operatorname{sig}' = \operatorname{sig}'(1 - 2\operatorname{sig}) = \operatorname{sig}(1 - \operatorname{sig})(1 - 2\operatorname{sig})$.
3. **Change variable.** As $s$ runs over $\mathbb R$, $u = \operatorname{sig}(s)$ runs over $(0, 1)$. So $\max_s|\operatorname{sig}''(s)| = \sup_{u \in (0,1)}|p(u)|$ with $p(u) = u(1 - u)(1 - 2u) = u - 3u^2 + 2u^3$.
4. **Critical points.** $p'(u) = 1 - 6u + 6u^2 = 0$ at $u = \frac{6 \pm \sqrt{36 - 24}}{12} = \frac12 \pm \frac{\sqrt3}{6}$.
5. **Values.** At $u = \frac12 - \frac{\sqrt3}{6}$: $u(1 - u) = \frac14 - \frac{3}{36} = \frac16$ and $1 - 2u = \frac{\sqrt3}{3} = \frac{1}{\sqrt3}$, so $p = \frac{1}{6\sqrt3}$. At $u = \frac12 + \frac{\sqrt3}{6}$, $u(1 - u)$ is the same and $1 - 2u = -\frac{1}{\sqrt3}$, so $p = -\frac{1}{6\sqrt3}$.
6. **Endpoints.** $p(u) \to 0$ as $u \to 0$ or $u \to 1$. A continuous function on $(0, 1)$ that vanishes at both ends attains its largest $|p|$ at a critical point, so $|\operatorname{sig}''| \le \frac{1}{6\sqrt3} = 0.0962$, with equality exactly at $\operatorname{sig}(s) = \frac12 \pm \frac{\sqrt3}{6}$.

In terms of $s$: $\operatorname{sig}(s) = u$ means $s = \ln\frac{u}{1 - u}$, and $\frac{1/2 + \sqrt3/6}{1/2 - \sqrt3/6} = \frac{3 + \sqrt3}{3 - \sqrt3} = 2 + \sqrt3$. So the extremes are at $s = \pm\ln(2 + \sqrt3) = \pm1.3170$. A grid search over $[-10, 10]$ in the script finds the maximum $0.0962$ at $|s| = 1.3170$. Since the sigmoid-loss Hessian is $\operatorname{sig}''(-m)\,xx^\top$ with $\|x\| \le 1$ (@exm-opt-sigmoid), this is its smoothness constant $\beta$.

*The sigmoid loss is about 2.6 times smoother than the logistic loss ($0.0962$ against $\frac14$), but non-convex: $\operatorname{sig}''$ takes both signs.*
:::

## Covariance under sampling without replacement {#sec-sol-opt-5}

> **Exercise 5.** Step 3 of the proof of @prp-opt-minibatch shows that two distinct uniform draws from deviations $u_i$ that sum to zero have $\mathbb E\langle u_{I}, u_{J}\rangle = -\sigma_1^2/(n - 1)$. Verify @eq-opt-minibatch directly at $B = n$. Then check the covariance and the variance at $B = 2$ by enumeration on the toy data of @sec-opt-sgd-convex: the records $z = (1, 2, 6)$ with the loss $\frac12(\theta - z)^2$, at $\theta = 4$. *Check: $7/6$.* *(Minibatch variance)*

::: {.callout-tip collapse="true" title="Solution"}
**At $B = n$.** The formula gives $\frac{\sigma_1^2}{n}\cdot\frac{n - n}{n - 1} = 0$. Directly: a batch of $n$ distinct indices is the whole population, so $g_B = \frac1n\sum_i\nabla\ell(\theta; z_i) = \nabla \mathcal L(\theta)$ exactly, and the variance is 0. The two agree.

**Enumeration on the toy data.** The per-record gradients $\theta - z_i$ at $\theta = 4$ are $3, 2, -2$, and the true gradient is their mean, 1. So the deviations are $u = (2, 1, -3)$, which sum to zero, and $\sigma_1^2 = \frac{4 + 1 + 9}{3} = \frac{14}{3}$.

- The covariance. The distinct pairs have products $u_1u_2 = 2$, $u_1u_3 = -6$, $u_2u_3 = -3$, each appearing twice among the 6 ordered pairs, so the mean is $\frac{2(2 - 6 - 3)}{6} = -\frac73$. The formula gives $-\frac{14/3}{3 - 1} = -\frac73$.
- The batch variance at $B = 2$. Expand $\bar u^2 = \frac14(u_I^2 + u_J^2 + 2u_Iu_J)$ and take expectations: $\frac14\big(2\cdot\frac{14}{3} - 2\cdot\frac73\big) = \frac14\cdot\frac{14}{3} = \frac76$. Listing the three batches $\{1, 2\}$, $\{1, 3\}$ and $\{2, 3\}$, the mean deviations are $1.5$, $-0.5$ and $-1$, with squares 2.25, 0.25 and 1, which average to $\frac13(2.25 + 0.25 + 1) = \frac76$ too. So does @eq-opt-minibatch: $\frac{14/3}{2}\cdot\frac{1}{2} = \frac76$, half of the $\frac73$ that drawing with replacement would give.

The script enumerates every batch in exact fractions:

| $B$ | Enumerated $\mathbb E\bar u^2$ | @eq-opt-minibatch, without replacement |
| --- | --- | --- |
| 1 | 14/3 | 14/3 |
| 2 | 7/6 | 7/6 |
| 3 | 0 | 0 |

: Exact minibatch variance on the toy data of @sec-opt-sgd-convex, by enumeration and by formula. {#tbl-sol-opt-enum}

*The negative covariance is what the factor $(n - B)/(n - 1)$ in @eq-opt-minibatch accounts for: once one record is in the batch, the rest are tilted the other way.*
:::

## Critical batch size {#sec-sol-opt-6}

> **Exercise 6.** Fix a budget of $N = TB$ gradient evaluations and put $\sigma^2/B$ in place of $\sigma^2$ in @eq-opt-nonconvex. Show that the optimized bound equals $2\sqrt{2\beta\Gamma\sigma^2/N}$, independent of $B$, as long as $B \le \sigma\sqrt{N/(2\beta\Gamma)}$, and grows with $B$ beyond that. *Check: for the sigmoid-loss record ($\beta = 0.0962$, $\Gamma = 0.5$, $\sigma^2 = 1/16$) and $N = 10^4$, the critical batch size is about 81.* *(Non-convex SGD)*

::: {.callout-tip collapse="true" title="Solution"}
1. **The bound in terms of $B$.** With $T = N/B$ steps and variance $\sigma^2/B$, @eq-opt-nonconvex reads
$$\Phi(\eta) = \frac{2\Gamma B}{\eta N} + \frac{\beta\eta\sigma^2}{B}, \qquad 0 < \eta \le \frac1\beta.$$
2. **Unconstrained minimum.** $\Phi = a/\eta + b\eta$ with $a = 2\Gamma B/N$ and $b = \beta\sigma^2/B$. Its derivative $-a/\eta^2 + b$ vanishes at $\eta_B = \sqrt{a/b} = B\sqrt{2\Gamma/(\beta\sigma^2N)}$, where both terms equal $\sqrt{ab}$, so $\Phi(\eta_B) = 2\sqrt{ab} = 2\sqrt{2\beta\Gamma\sigma^2/N}$. The $B$ in $a$ and the $1/B$ in $b$ cancel: the value does not depend on $B$.
3. **When the constraint is inactive.** $\eta_B \le 1/\beta$ means $B\sqrt{2\Gamma/(\beta\sigma^2N)} \le 1/\beta$, that is $B \le \frac1\beta\sqrt{\frac{\beta\sigma^2N}{2\Gamma}} = \sigma\sqrt{\frac{N}{2\beta\Gamma}} = B_{\text{crit}}$. Up to $B_{\text{crit}}$ the optimized bound is $2\sqrt{2\beta\Gamma\sigma^2/N}$.
4. **Beyond it.** For $B > B_{\text{crit}}$, $\eta_B > 1/\beta$, and $\Phi$ decreases on $(0, \eta_B)$, so the best allowed step is $\eta = 1/\beta$:
$$\Phi(1/\beta) = \frac{2\beta\Gamma B}{N} + \frac{\sigma^2}{B}.$$
Its derivative in $B$ is $\frac{2\beta\Gamma}{N} - \frac{\sigma^2}{B^2}$, which is positive exactly when $B^2 > \frac{\sigma^2N}{2\beta\Gamma} = B_{\text{crit}}^2$. So the bound grows with $B$. At $B = B_{\text{crit}}$ both terms equal $\sigma\sqrt{2\beta\Gamma/N}$, so the two pieces join continuously; for $B \gg B_{\text{crit}}$ the bound is about $2\beta\Gamma B/N = 2\beta\Gamma/T$, the noiseless rate with only $T = N/B$ steps.
5. **The numbers.** With $\beta = 1/(6\sqrt3)$, $\Gamma = 0.5$, $\sigma = \frac14$, $N = 10^4$: $B_{\text{crit}} = \frac14\sqrt{10^4/0.0962} = 80.6$, about 81 as the check says.

| Batch $B$ | Steps $T = N/B$ | Best step $\eta$ | Optimized bound |
| --- | --- | --- | --- |
| 1 | 10,000 | 0.129 | $1.55\times10^{-3}$ |
| 10 | 1,000 | 1.289 | $1.55\times10^{-3}$ |
| 80 | 125 | 10.316 | $1.55\times10^{-3}$ |
| 200 | 50 | 10.392 | $2.24\times10^{-3}$ |
| 1,000 | 10 | 10.392 | $9.69\times10^{-3}$ |
| 10,000 | 1 | 10.392 | $9.62\times10^{-2}$ |

: The non-convex bound for the sigmoid-loss constants at a fixed budget of $N = 10^4$ gradient evaluations, optimized over $\eta \le 1/\beta = 10.392$. {#tbl-sol-opt-batch}

In @tbl-sol-opt-batch, the bound is flat up to $B = 80$, where the best step reaches the cap $1/\beta$, and then grows almost linearly in $B$. At $B = 1$ it equals the chapter's bound at $T = 10{,}000$ in @tbl-opt-nonconvex. Below $B_{\text{crit}}$ a larger batch reduces the noise by exactly as much as it reduces the number of steps; above it, the learning rate can no longer grow to use the lower noise, so the extra evaluations are wasted.

*Up to the critical batch size, batches trade steps for noise one for one, so they cost nothing per gradient evaluation and pay off through parallelism; beyond it they waste computation.*
:::

## A sharper sensitivity bound {#sec-sol-opt-7}

> **Exercise 7.** Redo the proof of @thm-opt-erm-sensitivity assuming $\mathcal L_\lambda(\cdot; D)$ is $\lambda'$-strongly convex and the two replaced gradients differ by at most $K$ at $\theta'$. Show that the movement is at most $K/(\lambda' n)$. Using $\lambda' \approx \lambda + 0.0233$ and $K \approx 1$, predict the ratio observed/bound at $\lambda = 0.01$ and compare with @tbl-opt-erm. *(Strong monotonicity)*

::: {.callout-tip collapse="true" title="Solution"}
Use the notation of the proof of @thm-opt-erm-sensitivity: $\mathcal L_\lambda(\cdot; D)$ and $\mathcal L_\lambda(\cdot; D') = \mathcal L_\lambda(\cdot; D) + \psi$ with $\psi(v) = \frac1n(\ell(v; z') - \ell(v; z_i))$, $\theta = \theta^*(D)$, $\theta' = \theta^*(D')$.

1. **The gradient difference at $\theta'$.** $\nabla \psi(\theta') = \frac1n\big(\nabla\ell(\theta'; z') - \nabla\ell(\theta'; z_i)\big)$, so $\|\nabla \psi(\theta')\| \le K/n$ by assumption. (The chapter used the triangle inequality to get $2L/n$; here $K$ replaces $2L$.)
2. **Optimality,** as in step 2 of that proof: $\nabla \mathcal L_\lambda(\theta; D) = 0$ and $\nabla \mathcal L_\lambda(\theta'; D) = -\nabla \psi(\theta')$.
3. **Strong monotonicity** of the $\lambda'$-strongly convex $\mathcal L_\lambda(\cdot; D)$ (@eq-opt-strong-monotone with $\lambda'$ in place of $\lambda$): $\lambda'\|\theta' - \theta\|^2 \le \langle\nabla \mathcal L_\lambda(\theta'; D) - \nabla \mathcal L_\lambda(\theta; D), \theta' - \theta\rangle = \langle -\nabla \psi(\theta'), \theta' - \theta\rangle$.
4. **Cauchy–Schwarz** and step 1: $\lambda'\|\theta' - \theta\|^2 \le \frac K n\|\theta' - \theta\|$.
5. **Divide** by $\lambda'\|\theta' - \theta\|$: $\|\theta' - \theta\| \le \frac{K}{\lambda' n}$. With $\lambda' = \lambda$ and $K = 2L$ this is @eq-opt-erm-sensitivity.

**Prediction.** Dividing the sharper bound by the chapter's, the $n$ cancels:
$$\frac{K/(\lambda' n)}{2L/(\lambda n)} = \frac{\lambda K}{2L\lambda'} = \frac{0.01 \times 1}{2 \times (0.01 + 0.0233)} = 0.150.$$
If the sharper bound were nearly attained, the observed/bound ratio at $\lambda = 0.01$ would be about 0.150. @tbl-opt-erm reports 0.10 to 0.13: the right size, and a little lower.

**Why a little lower.** The script repeats the experiment at $n = 1{,}000$, $\lambda = 0.01$ (20 replacements on a fresh dataset) and measures, for each replacement, $K$ at $\theta'$ and $\lambda'$ as the smallest Hessian eigenvalue of $\mathcal L_\lambda(\cdot; D)$ along the segment from $\theta$ to $\theta'$ (all the proof needs).

- The largest movement is $2.38\times10^{-2}$, ratio 0.119 to $2L/(\lambda n)$, inside the chapter's range.
- For that replacement, $K = 1.133$ and $\lambda' = 0.0404$, so the sharper bound is $2.81\times10^{-2}$.
- Over all 20 replacements, $\lambda'$ lies between 0.0402 and 0.0404, larger than $0.0333$: at the regularized minimizer $\|\theta\|$ is smaller than at the unregularized one, and the logistic curvature $\operatorname{sig}(m)(1 - \operatorname{sig}(m))$ of @exm-opt-logistic is larger at smaller margins. $K$ ranges from 0.103 to 1.133, depending on how different the swapped records are.
- The movement is between 0.74 and 0.88 of $K/(\lambda' n)$ for every replacement. The rest of the slack is Cauchy–Schwarz in step 4: $\nabla \psi(\theta')$ is not exactly parallel to $\theta' - \theta$.

*Most of the gap between @eq-opt-erm-sensitivity and reality is the curvature of the data, which the bound ignores; with the true $\lambda'$ and $K$ the bound is within a factor of 1.4 of the observed movement.*
:::

## The regularized median {#sec-sol-opt-8}

> **Exercise 8.** For $n = 2k + 1$ values, $k + 1$ zeros and $k$ ones, minimize $\frac1n\sum_i|\theta - z_i| + \frac\lambda2 \theta^2$ before and after replacing one 0 by a 1. *Hint: the subgradient at 0 before the change is $[-1, 1/n]$, which contains 0; after it, the derivative on $(0, 1)$ is $-1/n + \lambda \theta$.* *Check: the move is $1/(\lambda n)$, half the bound; for $n = 101$ and $\lambda = 0.1$, 0.099 against 0.198.* *(Subgradients, sensitivity)*

::: {.callout-tip collapse="true" title="Solution"}
The objective is $\lambda$-strongly convex (part (b) of @cor-opt-strong, which holds with subgradients too), so its minimizer is unique, and $\theta$ is the minimizer exactly when 0 lies in the subdifferential at $\theta$. The subdifferential of $|\theta - z|$ is $\{-1\}$ for $\theta < z$, $\{1\}$ for $\theta > z$, and $[-1, 1]$ at $\theta = z$.

**Before** ($k + 1$ zeros, $k$ ones). At $\theta = 0$:

1. The $k + 1$ terms $|\theta|$ contribute $\frac{k + 1}{n}[-1, 1]$.
2. The $k$ terms $|\theta - 1|$ have derivative $-1$ at 0 and contribute $-\frac kn$.
3. The regularizer has derivative $\lambda \cdot 0 = 0$.
4. Total: $\big[\frac{-(k + 1) - k}{n}, \frac{(k + 1) - k}{n}\big] = [-1, \frac1n]$, using $n = 2k + 1$. It contains 0, so $\theta^*(D) = 0$.

**After** ($k$ zeros, $k + 1$ ones).

1. **On $(0, 1)$** the objective is differentiable. The zeros contribute $+\frac kn$, the ones $-\frac{k + 1}{n}$, the regularizer $\lambda \theta$: the derivative is $-\frac1n + \lambda \theta$. It vanishes at $\theta = \frac{1}{\lambda n}$, which lies in $(0, 1)$ when $\lambda n > 1$.
2. **0 is no longer optimal.** At 0 the subdifferential is $\frac kn[-1, 1] - \frac{k + 1}{n} = [-1, -\frac1n]$, which excludes 0.
3. **The cap.** If $\lambda n \le 1$, the derivative on $(0, 1)$ stays negative, and at $\theta = 1$ the subdifferential is $\frac kn + \frac{k + 1}{n}[-1, 1] + \lambda = [\lambda - \frac1n, 1 + \lambda]$, which contains 0. Then $\theta^*(D') = 1$, the unregularized median.

So the minimizer moves by $\min\big(1, \frac{1}{\lambda n}\big)$. The chapter's bound with $L = 1$ is $\frac{2}{\lambda n}$: when $\lambda n > 1$ the move is exactly half of it. For $n = 101$ and $\lambda = 0.1$, the move is $\frac{1}{10.1} = 0.0990$ against $0.1980$. The script minimizes both objectives numerically:

| $n$ | $\lambda$ | Numerical move | $\min(1, 1/(\lambda n))$ | Bound $2/(\lambda n)$ |
| --- | --- | --- | --- | --- |
| 101 | 0.1 | 0.0990 | 0.0990 | 0.1980 |
| 1,001 | 0.1 | 0.0100 | 0.0100 | 0.0200 |
| 11 | 0.1 | 0.9091 | 0.9091 | 1.8182 |
| 11 | 0.05 | 1.0000 | 1.0000 | 3.6364 |

: Movement of the regularized median when one of $k + 1$ zeros becomes a one, against @eq-opt-erm-sensitivity. {#tbl-sol-opt-median}

**Why half.** In step 2 of the proof of @thm-opt-erm-sensitivity, the gradient of $\mathcal L_\lambda(\cdot; D)$ at $\theta'$ is $\frac{k + 1}{n} - \frac kn + \lambda \theta' = \frac2n$ — the full $2L/n$. But strong monotonicity, on which part (a) of @cor-opt-strong rests, compares it with the subgradient 0 at $\theta = 0$, and the subgradient of $\mathcal L_\lambda(\cdot; D)$ jumps from 0 to $\frac1n$ at the kink: just to the right of 0 it is already $\frac1n + \lambda \theta$. Half of the $\frac2n$ is absorbed by that jump, and only the other half must be supplied by the regularizer's slope $\lambda \theta'$. The bound assumes all of it is.

*Strong convexity turns a jump of 1 into a move of $1/(\lambda n)$, and the kink of the absolute loss at the old minimizer acts as extra curvature that the bound does not count.*
:::
