# Optimization and estimation

This appendix proves seven results that the chapters on optimization, generalization and attacks use without deriving them in full. The first four are about gradient methods on convex functions: three ways to recognize convexity and the constant factor by which gradient descent (GD) closes the gap on a strongly convex function, a gradient step that never pushes two points apart, the best time to stop stochastic gradient descent (SGD), and a lower bound that no gradient method can beat on functions whose slope is bounded (Lipschitz functions). The other three treat a trained model as a random object, random because its training set is. They find the limit of the error of the 1-nearest-neighbour rule, show that the parameters fitted by logistic regression are approximately Gaussian, with a formula for how much one record moves them, and show when the best membership attack looks at the parameters only through the target's loss. @tbl-apx-opt-map lists where each result is stated or used.

| Derivation | Where the book states or uses it |
| --- | --- |
| Three tests for convexity; linear convergence of GD | @sec-opt-definitions and the GD section of [Optimization](../part01-prerequisites/06-optimization.md) |
| Co-coercivity of the gradient | The proof of @lem-stability-nonexpansive; @sec-shuf-iteration |
| Early stopping balances optimization error against stability | The SGD section of [Algorithmic stability and generalization](../part01-prerequisites/07-stability-generalization.md) |
| Lower bound $RL/\sqrt T$ for convex Lipschitz optimization | The section on SGD for convex Lipschitz objectives in [Optimization](../part01-prerequisites/06-optimization.md) |
| Limit of the 1-nearest-neighbour error | @prp-stability-nn |
| Fitted parameters are approximately Gaussian; influence of one record | @sec-prop-meta, @sec-mia-lira, @sec-mia-outliers, @sec-attr-inversion |
| The best membership attack depends on the loss | @sec-threat-models |

: The derivations of this appendix and the chapter sections that state or use them. {#tbl-apx-opt-map}

## Convexity three ways, and the linear rate of gradient descent {#sec-apx-opt-convexity}

### What the chapter uses

Write $f$ for a differentiable function of the parameters $\theta \in \mathbb R^d$, and $\lambda \ge 0$ for a level of curvature. [Optimization](../part01-prerequisites/06-optimization.md) uses four facts. It proves that chords give tangents (@lem-opt-tangent), and the fourth fact only for a quadratic. First, tangents give chords: if $f$ lies above all its tangents (@eq-opt-first-order), then for every mixing weight $\omega \in [0, 1]$,
$$f(\omega \theta' + (1-\omega)\theta) \le \omega f(\theta') + (1-\omega)f(\theta).$$
Second, if the Hessian $\nabla^2 f$, the matrix of second derivatives, has every eigenvalue at least $\lambda$ everywhere, written $\nabla^2 f \succeq \lambda I$, then $f$ lies above each tangent by a bowl of curvature $\lambda$:
$$f(\theta') \ge f(\theta) + \langle\nabla f(\theta), \theta' - \theta\rangle + \frac\lambda2\|\theta' - \theta\|^2.$$ {#eq-apx-opt-strong}
This is what **$\lambda$-strongly convex** means, and with $\lambda = 0$ it is convexity. For $\lambda = 0$ the test also works backwards: a convex twice-differentiable $f$ has $\nabla^2 f \succeq 0$. Third, a strongly monotone gradient (@eq-opt-strong-monotone) gives @eq-apx-opt-strong without second derivatives.

Fourth, take the empirical risk $\mathcal L(\theta)$ on a fixed dataset, $\lambda$-strongly convex with minimum value $\mathcal L^*$ and $\beta$-smooth, so that no curvature exceeds $\beta$, and write $\kappa = \beta/\lambda$ for its condition number. Then GD with the learning rate $\eta = 1/\beta$ shrinks the gap to the minimum by the factor $1 - 1/\kappa$ at every step, where $\theta_t$ is the point after $t$ steps:
$$\mathcal L(\theta_t) - \mathcal L^* \le \left(1 - \frac1\kappa\right)^t\big(\mathcal L(\theta_0) - \mathcal L^*\big).$$ {#eq-apx-opt-gd-values}
And when $\lambda I \preceq \nabla^2 \mathcal L \preceq \beta I$, the chapter's distance bound @eq-opt-gd-linear holds as stated for the quadratic. The chapter cites Nesterov's *Lectures on Convex Optimization*, §2.1, for the second and third facts, and its interpretation claims the fourth for every strongly convex smooth function.

### The idea

To get a chord from tangents, draw the tangent at the mixed point $z = \omega \theta' + (1-\omega)\theta$. Take $f(\theta) = \theta^2$, $\theta = 0$, $\theta' = 2$ and $\omega = \frac12$, so $z = 1$. The tangent at 1 is $1 + 2(\theta - 1)$, which gives $-1$ and $3$ at the two ends, below the function values 0 and 4. The tangent is a straight line, so its two end values average to its value at $z$, which is $1 = f(z)$. The function values average to 2, which is at least 1, so the chord lies above the graph at $z$.

### Derivation

1. **Tangents give chords.** Let $z = \omega \theta' + (1-\omega)\theta$. The tangent inequality at $z$, applied towards $\theta$ and towards $\theta'$, gives $f(\theta) \ge f(z) + \langle\nabla f(z), \theta - z\rangle$ and $f(\theta') \ge f(z) + \langle\nabla f(z), \theta' - z\rangle$. Multiply the first by $1 - \omega$ and the second by $\omega$, and add. The inner products combine into $\langle\nabla f(z), (1-\omega)\theta + \omega \theta' - z\rangle$, which is zero because the vector on the right is $z - z$. What is left is $(1-\omega)f(\theta) + \omega f(\theta') \ge f(z)$, the chord inequality. @fig-apx-opt-convexity (left) draws the two gaps and their average.
2. **A curvature bound gives the bowl.** Fix $\theta$ and $\theta'$, and let $u = \theta' - \theta$ be the step between them. Walking a fraction $s \in [0, 1]$ of that step gives the one-variable function $g(s) = f(\theta + su)$, the values of $f$ along the segment. By the chain rule, $g'(s) = \langle\nabla f(\theta + su), u\rangle$ and $g''(s) = u^\top\nabla^2 f(\theta + su)\,u$. A symmetric matrix $A$ satisfies $A \succeq \lambda I$ exactly when $u^\top Au \ge \lambda\|u\|^2$ for every $u$, so the curvature bound says $g''(s) \ge \lambda\|u\|^2$. Now integrate twice. By the fundamental theorem of calculus, the slope at $s$ is the slope at 0 plus the integral of $g''$ from 0 to $s$, so $g'(s) \ge g'(0) + \lambda s\|u\|^2$. Integrating again over $s \in [0, 1]$ gives $g(1) \ge g(0) + g'(0) + \frac\lambda2\|u\|^2$, which is @eq-apx-opt-strong written in terms of $f$.
3. **And back: convexity gives a Hessian with no negative eigenvalue.** Step 5 of the proof of @prp-opt-constants, with $\lambda = 0$, shows that a convex $f$ has a monotone gradient, $\langle\nabla f(\theta + su) - \nabla f(\theta), su\rangle \ge 0$. In terms of $g$ this says $s\,(g'(s) - g'(0)) \ge 0$. Divide by $s^2$ and let $s \to 0$. The ratio $(g'(s) - g'(0))/s$ tends to $g''(0) = u^\top\nabla^2 f(\theta)u$, so that is at least 0 for every direction $u$.
4. **Strong monotonicity gives the bowl.** Keep the same $g$. Strong monotonicity @eq-opt-strong-monotone, applied to the points $\theta + su$ and $\theta$, says $\langle\nabla f(\theta + su) - \nabla f(\theta), su\rangle \ge \lambda s^2\|u\|^2$. Dividing by $s > 0$ gives $g'(s) - g'(0) \ge \lambda s\|u\|^2$, the inequality of step 2 without second derivatives. Integrating it over $s \in [0, 1]$ again gives @eq-apx-opt-strong.
5. **A small gradient means a small gap.** Apply @eq-apx-opt-strong at a point $\theta$ towards every other point $\theta'$. The right side is a parabola in $\theta'$, and completing the square shows that its lowest value is $\mathcal L(\theta) - \|\nabla \mathcal L(\theta)\|^2/(2\lambda)$, reached at $\theta' = \theta - \nabla \mathcal L(\theta)/\lambda$. Every value of $\mathcal L$ is above the right side, and so above that lowest value. In particular $\mathcal L^* \ge \mathcal L(\theta) - \|\nabla \mathcal L(\theta)\|^2/(2\lambda)$, which rearranges to the **Polyak–Łojasiewicz inequality**:
$$\mathcal L(\theta) - \mathcal L^* \le \frac{\|\nabla \mathcal L(\theta)\|^2}{2\lambda}.$$ {#eq-apx-opt-pl}
6. **Each step removes a fixed fraction of the gap.** The chapter derives from the descent lemma @eq-opt-descent-lemma that one GD step with the learning rate $\eta = 1/\beta$ gives $\mathcal L(\theta_{t+1}) \le \mathcal L(\theta_t) - \|\nabla \mathcal L(\theta_t)\|^2/(2\beta)$. Subtract $\mathcal L^*$ from both sides. By @eq-apx-opt-pl, $\|\nabla \mathcal L(\theta_t)\|^2 \ge 2\lambda(\mathcal L(\theta_t) - \mathcal L^*)$, so the subtracted term is at least $\frac\lambda\beta(\mathcal L(\theta_t) - \mathcal L^*)$. Hence $\mathcal L(\theta_{t+1}) - \mathcal L^* \le (1 - 1/\kappa)(\mathcal L(\theta_t) - \mathcal L^*)$, and repeating this $t$ times gives @eq-apx-opt-gd-values.
7. **The distance bound, when second derivatives exist.** Write $\theta^*$ for the minimizer. It has zero gradient, so the GD step can be written as $\theta_{t+1} - \theta^* = (\theta_t - \theta^*) - \frac1\beta(\nabla \mathcal L(\theta_t) - \nabla \mathcal L(\theta^*))$. Step 4 of the proof of @prp-opt-constants writes the gradient difference as $\bar H_t(\theta_t - \theta^*)$, where $\bar H_t = \int_0^1\nabla^2 \mathcal L(\theta^* + s(\theta_t - \theta^*))\,ds$ is the average Hessian along the segment. For a unit vector $u$, $u^\top\bar H_tu$ is an average of numbers $u^\top\nabla^2 \mathcal L\,u$ that all lie in $[\lambda, \beta]$, so it lies there too. The smallest and largest values of $u^\top Au$ over unit vectors are the extreme eigenvalues of a symmetric $A$, so every eigenvalue of $\bar H_t$ is in $[\lambda, \beta]$. So the GD step is $\theta_{t+1} - \theta^* = (I - \bar H_t/\beta)(\theta_t - \theta^*)$, a matrix with eigenvalues between 0 and $1 - 1/\kappa$ applied to the error. This is the quadratic's step 1 with the fixed $H$ replaced by $\bar H_t$, and its steps 2 and 3 then go through unchanged and give @eq-opt-gd-linear.

Step 7 also gives a faster rate for the gap. The gradient vanishes at $\theta^*$, so the descent lemma at $\theta^*$ gives $\mathcal L(\theta_t) - \mathcal L^* \le \frac\beta2\|\theta_t - \theta^*\|^2$, and with @eq-opt-gd-linear the gap falls like $(1 - 1/\kappa)^{2t}$. That exponent is twice the one in @eq-apx-opt-gd-values, which is the price of an argument that uses only first derivatives.

![Left: the chord of $f(\theta) = \ln(1 + e^{-\theta}) + 0.1\theta^2$ between $\theta = -3$ and $\theta' = 3$, the tangent at the mixed point $z$, and the two gaps between $f$ and the tangent at the ends (orange), whose weighted average is the gap between the chord and $f(z)$ (blue). Right: the same function with curvature at least $\lambda = 0.2$, which stays above its tangent plus the bowl $\frac\lambda2(\theta' - \theta)^2$.](figures/apx_opt_convexity.svg){#fig-apx-opt-convexity}

### Checking it

The script uses the logistic data of the optimization chapter, $n = 1{,}000$ records with $x$ uniform on the unit sphere in $\mathbb R^5$, plus a ridge penalty $\frac\lambda2\|\theta\|^2$ with $\lambda = 0.01$. Each $x$ has length 1, so the logistic curvature is at most $\frac14$, and with the ridge $\beta = 0.26$ and $\kappa = 26$. Over 2,000 random pairs of points in the ball of radius 5, the smallest ratio of the tangent gap to $\frac12\|\theta' - \theta\|^2$ is 0.0372, above $\lambda = 0.01$ as @eq-apx-opt-strong requires. GD from $\theta_0 = 0$ satisfies both bounds with room to spare: after 25 steps the gap is $1.22\times10^{-5}$, against $5.30\times10^{-2}$ from @eq-apx-opt-gd-values.

The bounds are loose here because they know only the curvature $\lambda = 0.01$ of the penalty, while the data add curvature of their own. At the minimizer, the smallest eigenvalue of $\nabla^2 \mathcal L$ is 0.0404. The rate of step 7 with that curvature in place of $\lambda$ is $(1 - 0.0404/0.26)^2 = 0.713$ per step, and the measured rate between steps 20 and 40 is 0.712. So the form of the rate is exact, and only the curvature it is given is pessimistic, as @fig-apx-opt-gd-linear shows.

![The gap $\mathcal L(\theta_t) - \mathcal L^*$ of GD on ridge-regularized logistic regression ($\lambda = 0.01$, $\kappa = 26$), with the Polyak–Łojasiewicz bound @eq-apx-opt-gd-values, the bound from the distance @eq-opt-gd-linear, and the same rate computed with the smallest curvature at the minimizer, which tracks the run.](figures/apx_opt_gd_linear.svg){#fig-apx-opt-gd-linear}

### What it does not cover

The distance bound @eq-opt-gd-linear was proved here only for twice-differentiable $\mathcal L$. It holds without second derivatives as well, by a longer proof that applies the co-coercivity of the next section to $\mathcal L - \frac\lambda2\|\theta\|^2$, which is convex and $(\beta - \lambda)$-smooth (Nesterov 2018, Theorem 2.1.12). The gap bound @eq-apx-opt-gd-values needs only the descent lemma and @eq-apx-opt-pl, so it holds for any function that satisfies the Polyak–Łojasiewicz inequality, even one that is not convex (Karimi, Nutini & Schmidt 2016).

## The gradient of a convex smooth function is co-coercive {#sec-apx-opt-cocoercive}

### What the chapter uses

Step 1 of the proof of @lem-stability-nonexpansive states, citing Nesterov's Theorem 2.1.5, that the gradient of a convex $\beta$-smooth function $f$ is **co-coercive**: for all $u$ and $v$,
$$\langle\nabla f(u) - \nabla f(v),\, u - v\rangle \ge \frac1\beta\|\nabla f(u) - \nabla f(v)\|^2.$$ {#eq-apx-opt-cocoercive}
In words, two gradients that differ by a lot must come from points that are far apart along the direction of that difference. The lemma derives from it that a gradient step $\psi(\theta) = \theta - \eta\nabla f(\theta)$ with $0 \le \eta \le 2/\beta$ is non-expansive, $\|\psi(u) - \psi(v)\| \le \|u - v\|$, which drives the stability bound @eq-stability-hrs for SGD and the contraction of @sec-shuf-iteration.

### The idea

Take the logistic loss $f(\theta) = \ln(1 + e^{-\theta})$, whose slope is $f'(\theta) = -\sigma(-\theta)$, where $\sigma(s) = 1/(1 + e^{-s})$ is the logistic function, and whose curvature is at most $\beta = \frac14$. At $u = 0$ and $v = 2$ the slopes are $-0.5000$ and $-0.1192$. The left side of @eq-apx-opt-cocoercive is $(-0.5000 + 0.1192)(0 - 2) = 0.7616$, and the right side is $0.3808^2/0.25 = 0.5800$, smaller as it should be. For the quadratic $f(\theta) = \frac\beta2 \theta^2$ the slope difference is $\beta(u - v)$, so both sides equal $\beta(u - v)^2$, and the inequality is tight.

### Derivation

The proof subtracts from $f$ its tangent at $v$, which makes $v$ the lowest point (step 1). A gradient step from $u$ cannot go below that lowest point (step 2), which gives a tangent inequality with an extra term (step 3). Writing it from both points and adding gives @eq-apx-opt-cocoercive (step 4).

1. **Shift the minimum to $v$.** Define $\varphi(\theta) = f(\theta) - \langle\nabla f(v), \theta\rangle$. Subtracting a linear function changes neither the curvature nor the differences between gradients, so $\varphi$ is convex and $\beta$-smooth. Its gradient is $\nabla\varphi(\theta) = \nabla f(\theta) - \nabla f(v)$, which is zero at $\theta = v$. So the tangent inequality @eq-opt-first-order for $\varphi$ at $v$ reads $\varphi(\theta) \ge \varphi(v)$ for every $\theta$, and $v$ minimizes $\varphi$.
2. **A gradient step from $u$ cannot go below the minimum.** Apply the descent lemma @eq-opt-descent-lemma to $\varphi$ at $u$, with the step $\theta' = u - \nabla\varphi(u)/\beta$. As in the chapter's GD step, this gives $\varphi(\theta') \le \varphi(u) - \|\nabla\varphi(u)\|^2/(2\beta)$. Since $\varphi(v) \le \varphi(\theta')$ by step 1, we get $\varphi(v) \le \varphi(u) - \|\nabla\varphi(u)\|^2/(2\beta)$.
3. **Return to $f$.** Write $\Delta = \nabla f(u) - \nabla f(v)$, which is $\nabla\varphi(u)$. The difference $\varphi(u) - \varphi(v)$ is $f(u) - f(v) - \langle\nabla f(v), u - v\rangle$, so step 2 becomes a tangent inequality with an extra term:
$$f(u) \ge f(v) + \langle\nabla f(v), u - v\rangle + \frac{1}{2\beta}\|\Delta\|^2.$$
4. **Swap and add.** The same inequality with $u$ and $v$ exchanged is $f(v) \ge f(u) + \langle\nabla f(u), v - u\rangle + \frac{1}{2\beta}\|\Delta\|^2$, because exchanging the points only flips the sign of $\Delta$. Adding the two, the function values cancel and the inner products combine into $-\langle\Delta, u - v\rangle$, which leaves $0 \ge -\langle\Delta, u - v\rangle + \frac1\beta\|\Delta\|^2$. That is @eq-apx-opt-cocoercive.
5. **The step is non-expansive.** This is the chapter's own argument, repeated so that the page stands alone. Expand $\psi(u) - \psi(v) = (u - v) - \eta\Delta$ as a square and replace $\langle\Delta, u - v\rangle$ by the smaller $\|\Delta\|^2/\beta$:
$$\|\psi(u) - \psi(v)\|^2 = \|u - v\|^2 - 2\eta\langle\Delta, u - v\rangle + \eta^2\|\Delta\|^2 \le \|u - v\|^2 - \eta\Big(\frac2\beta - \eta\Big)\|\Delta\|^2.$$
For $0 \le \eta \le 2/\beta$ the subtracted term is not negative, so the distance does not grow.

### Checking it

The script evaluates both sides of @eq-apx-opt-cocoercive for the logistic loss on 500 pairs $(u, v)$ drawn from a normal distribution with standard deviation 4. The smallest ratio of left to right side is 1.0006, so the inequality holds and is nearly tight, for pairs near 0, where the curvature is close to $\frac14$ and the loss looks like the quadratic of the example. Over the same pairs, the largest ratio $|\psi(u) - \psi(v)|/|u - v|$ is 0.9988 at $\eta = 8 = 2/\beta$, and 1.4985 at $\eta = 10$, where the step pushes some pairs apart. @fig-apx-opt-cocoercive shows both checks.

![Left: for 500 random pairs, $(f'(u) - f'(v))(u - v)$ against $(f'(u) - f'(v))^2/\beta$ for the logistic loss, $\beta = \frac14$. Every point lies above the line $y = x$. Right: the largest ratio of distances after and before one gradient step, over the same pairs, against the learning rate $\eta$, with the quadratic's factor $|1 - \eta\beta|$. Both cross 1 at $\eta = 2/\beta = 8$.](figures/apx_opt_cocoercive.svg){#fig-apx-opt-cocoercive}

### What it does not cover

Both convexity and smoothness are needed. Without convexity, step 1 fails, because a zero gradient no longer marks a minimum, and a gradient step can expand distances: on the concave function $-\frac\beta2\theta^2$ it multiplies every distance by $1 + \eta\beta$. Without smoothness there is no descent lemma, and the gradient of a function with a kink, such as $|\theta|$, jumps by 2 between points arbitrarily close together, so no constant $\beta$ can work.

## When to stop SGD: balancing optimization against stability {#sec-apx-opt-early-stop}

### What the chapter uses

@thm-stability-sgd in [Algorithmic stability and generalization](../part01-prerequisites/07-stability-generalization.md) shows that SGD with a constant step $\eta$ for $T$ steps, on $n$ training records and a loss whose slope is at most $L$, has uniform stability
$$\gamma \le \frac{2L^2\eta T}{n}.$$
So training longer lowers the training loss but raises $\gamma$. For a loss that is convex, smooth and $L$-Lipschitz, a constant step small enough for that theorem, and SGD started at $\theta_0 = 0$, the expected excess test loss of the averaged weights $\bar\theta$ over the best weights $\theta^*$ is at most
$$\frac{\|\theta^*\|^2}{2\eta T} + \frac{2L^2\eta T}{n} + \frac{\eta L^2}{2}.$$ {#eq-apx-opt-early-stop}
The sum is smallest when $\eta T = \|\theta^*\|\sqrt n/(2L)$, and there the excess test loss is at most $2L\|\theta^*\|/\sqrt n$, plus $\eta L^2/2$. The chapter makes the case for stopping early and points here for this balance.

### The idea

The trained weights test worse than the best weights for two reasons. SGD has not finished minimizing the training loss, which is the optimization error, and the training loss of the trained weights is an optimistic estimate of their test loss, which is the generalization gap. The first falls like $1/(\eta T)$ and the second grows like $\eta T$. A sum $a/u + bu$ of a falling and a rising term is smallest where the two terms are equal.

### Derivation

The plan is to split the excess test loss into three pieces, bound each piece by a result the chapters prove, and then choose the training time that minimizes the sum. Write $\mathcal L_{\mathcal D}(\theta)$ for the test loss of weights $\theta$, the average loss on a new record, and $\mathcal L(\theta; D)$ for the training loss on the training set $D$. Let $\theta^*$ be weights that minimize the test loss. They are fixed in advance, so they do not depend on $D$.

1. **Split the excess loss.** Add and subtract the training loss of $\bar\theta$ and of $\theta^*$:
$$\mathcal L_{\mathcal D}(\bar\theta) - \mathcal L_{\mathcal D}(\theta^*) = \big[\mathcal L_{\mathcal D}(\bar\theta) - \mathcal L(\bar\theta; D)\big] + \big[\mathcal L(\bar\theta; D) - \mathcal L(\theta^*; D)\big] + \big[\mathcal L(\theta^*; D) - \mathcal L_{\mathcal D}(\theta^*)\big].$$
The first bracket is the generalization gap of $\bar\theta$, the second is its optimization error on the training set, and the third compares a fixed model's training loss with its test loss.

2. **The third piece has mean zero.** The weights $\theta^*$ do not depend on $D$, so by @prp-stability-fixed-model their training loss is an unbiased estimate of their test loss, and the third bracket averages to zero over $D$.

3. **The averaged weights are as stable as the last ones.** Let $\theta_0, \theta_1, \dots$ be the iterates of SGD on $D$, and $\theta_t'$ those of the coupled run on the neighbouring set $D^{(i)}$, as in the proof of @thm-stability-sgd. That proof shows that the distance $r_t = \|\theta_t - \theta_t'\|$ has $\mathbb E\,r_t \le 2L\eta t/n$, which is at most $2L\eta T/n$ for every $t \le T$. The averaged weights are $\bar\theta = \frac1T\sum_{t=0}^{T-1}\theta_t$. The distance between two averages is at most the average of the distances, by the triangle inequality, so $\mathbb E\|\bar\theta - \bar\theta'\|$ is also at most $2L\eta T/n$. The loss is $L$-Lipschitz, so the averaged weights are uniformly stable with $\gamma \le 2L^2\eta T/n$, as in step 5 of that proof.

4. **The first piece.** By @eq-stability-expected, a $\gamma$-uniformly stable algorithm has expected gap at most $\gamma$. With step 3, the first bracket averages to at most $2L^2\eta T/n$.

5. **The second piece.** @thm-opt-sgd-convex bounds the optimization error of SGD with constant step $\eta$ on a convex objective whose random gradients are unbiased with mean squared length at most $G^2$, from a start at distance at most $R$ from $\theta^*$. Here the objective is the training loss $\mathcal L(\cdot; D)$ on all of $\mathbb R^d$, so there is no projection. A step that picks a record uniformly at random has a gradient whose average is the gradient of the training loss, so it is unbiased, and an $L$-Lipschitz loss gives $G = L$. Starting at $\theta_0 = 0$ makes $R = \|\theta^*\|$. The theorem compares with a minimizer of the objective, but its proof never uses that $\theta^*$ minimizes anything, because the convexity step holds for any fixed point. So, averaged over the steps' random choices,
$$\mathbb E\big[\mathcal L(\bar\theta; D) - \mathcal L(\theta^*; D)\big] \le \frac{\|\theta^*\|^2}{2\eta T} + \frac{\eta L^2}{2}.$$

6. **Add the three pieces.** Taking expectations in step 1 and using steps 2, 4 and 5 gives @eq-apx-opt-early-stop.

7. **Choose the training time.** For a fixed step $\eta$, training time enters @eq-apx-opt-early-stop only through $u = \eta T$, the total distance the learning rates allow. The part that depends on it is
$$h(u) = \frac{\|\theta^*\|^2}{2u} + \frac{2L^2u}{n}.$$
Its derivative $-\frac{\|\theta^*\|^2}{2u^2} + \frac{2L^2}{n}$ is negative for small $u$ and positive for large $u$, and it is zero at
$$u^* = \frac{\|\theta^*\|\sqrt n}{2L}.$$

8. **The value at the balance.** Substituting $u^*$, each of the two terms of $h$ equals $L\|\theta^*\|/\sqrt n$, so the excess test loss is at most $2L\|\theta^*\|/\sqrt n$, plus the term $\eta L^2/2$. Past $u^*$ the rising term of $h$ grows faster than the falling term shrinks, so longer training buys optimization accuracy that the stability term gives back.

### Checking it

The chapter's SGD experiment, @exm-stability-sgd, uses the logistic loss on $n = 100$ records with features on the unit sphere, so $L = 1$, and step $\eta = 0.25$. Its labels are $+1$ with probability $\sigma(5x_1)$, where $\sigma$ is the logistic function and $x_1$ the first feature. The logistic loss is the negative log-likelihood of a model that says $+1$ with probability $\sigma(\langle\theta, x\rangle)$, and at every $x$ its expected value is smallest when the model's probability equals the true one. So $\theta^* = (5, 0, \dots, 0)$, with $\|\theta^*\| = 5$. Step 7 then gives $u^* = 5\sqrt{100}/2 = 25$, which at $\eta = 0.25$ is $T = 100$ steps, one pass over the 100 records, and the bound there is $2 \cdot 5/10 = 1$, plus $\eta L^2/2 = 0.125$. @fig-apx-opt-early-stop draws the two terms and their sum against $u$.

In the experiment, though, the test loss keeps falling after one epoch, from 0.596 to 0.557 after five, and rises only after that (@tbl-stability-sgd). So the bound stops too early, because its stability term is too pessimistic: after one epoch it allows a gap of 0.5, while the measured gap is 0.036 and grows far more slowly than linearly. The experiment measures the last iterate rather than the average, so the comparison checks the shape of the balance, not its constant.

![The bound @eq-apx-opt-early-stop against $u = \eta T$, the learning rate times the number of steps, for $n = 100$, $\|\theta^*\| = 5$, $L = 1$ and $\eta = 0.25$, on log axes: the falling optimization term $\|\theta^*\|^2/(2u)$, the rising stability term $2L^2u/n$ (both dashed), and their sum plus $\eta L^2/2$ (solid), smallest at $u^* = 25$.](figures/apx_opt_early_stop.svg){#fig-apx-opt-early-stop}

### What it does not cover

The bound needs a convex loss, so it does not cover neural networks. Hardt, Recht & Singer (2016) also bound the stability of SGD on smooth non-convex losses, with steps that shrink like $1/t$, and their stability term then grows like a power of $T$ below 1. But the optimization error of a non-convex loss has no bound of the kind step 5 uses, so there is nothing to balance it against. With a step that shrinks over time, the sum of the learning rates replaces $\eta T$ in the stability term, and the balance is found the same way.

## No gradient method beats $RL/\sqrt T$ on every convex Lipschitz function {#sec-apx-opt-lower}

### What the chapter uses

The section on SGD for convex Lipschitz objectives in [Optimization](../part01-prerequisites/06-optimization.md) proves an error of at most $RL/\sqrt T$ after $T$ steps (@eq-opt-sgd-convex), where $R$ is the distance from the start to the minimizer and $L$ is the Lipschitz constant, the largest size of a gradient. It then says, citing Bubeck (2015, §3.5), that no method that only queries gradients can do better in order. This section proves that for the methods the book uses. A method is **span-restricted** if it starts at $\theta_1 = 0$ and every point it queries or outputs is a combination of the gradients it has received so far. GD and SGD, with a constant or a changing learning rate, with projection onto a ball centred at 0, which only rescales a vector, and with averaged iterates, are all span-restricted. In dimension $d \ge T + 1$ there is a convex function that is $L$-Lipschitz on the ball $\{\|\theta\| \le R\}$, with its minimizer at distance $R$ from the start, on which every span-restricted method, after $T$ gradient queries, outputs a point $\theta$ with
$$f(\theta) - f^* \ge \frac{RL}{2(1 + \sqrt{T + 1})}.$$ {#eq-apx-opt-lower}
For large $T$ the right side is close to half of $RL/\sqrt T$, so the SGD guarantee is optimal up to a factor of about 2.

### The idea

The hard function hides its minimum in many coordinates and reveals them one gradient at a time. It uses one more coordinate than the method has queries, $k = T + 1$:
$$f(\theta) = \gamma\max_{i \le k} \theta_i + \frac\alpha2\|\theta\|^2,$$
with positive constants $\gamma$ and $\alpha$ chosen in step 2. Take $T = 2$, so $k = 3$, with $R = L = 1$. The constants are then $\gamma = 0.6340$ and $\alpha = 0.3660$, and the minimum is $f^* = -0.1830$, reached with equal negative values in all three coordinates. The first query is at $\theta_1 = 0$, where all three coordinates tie at 0, so the gradient points along the first coordinate, and the method's next point has the form $(a, 0, 0)$. Whatever $a$ is, the largest coordinate is first reached at the first or the second position, so the second gradient involves only the first two coordinates. Any output then has the form $(a', b', 0)$. Its third coordinate makes the largest coordinate at least 0, so $f \ge 0$, and the error is at least $0.1830$, which is $1/(2(1 + \sqrt3))$, the right side of @eq-apx-opt-lower.

### Derivation

1. **The minimum.** The largest of $k$ numbers is at least their average, so $f(\theta) \ge \sum_{i \le k}\big(\frac\gamma k\theta_i + \frac\alpha2\theta_i^2\big) + \frac\alpha2\sum_{i > k}\theta_i^2$. Each term of the first sum is a parabola in $\theta_i$ whose lowest value is $-\gamma^2/(2\alpha k^2)$, at $\theta_i = -\gamma/(\alpha k)$. The terms of the second sum are at least 0. Adding up, $f(\theta) \ge -\gamma^2/(2\alpha k)$. The point with $\theta_i = -\gamma/(\alpha k)$ for $i \le k$ and 0 elsewhere reaches this value, because its first $k$ coordinates are equal and so their largest equals their average. So
$$f^* = -\frac{\gamma^2}{2\alpha k}, \qquad \|\theta^*\| = \frac{\gamma}{\alpha\sqrt k}.$$
2. **The constants.** The max term changes by at most $\gamma$ times the distance moved, since no coordinate changes by more than the distance. On the ball of radius $R$ the quadratic term changes by at most $\alpha R$ times the distance, since $\frac\alpha2(\|\theta\|^2 - \|\theta'\|^2) = \frac\alpha2\langle \theta + \theta', \theta - \theta'\rangle$ and $\|\theta + \theta'\| \le 2R$. So $f$ is $(\gamma + \alpha R)$-Lipschitz on the ball. Solving the two equations $\gamma + \alpha R = L$ and $\|\theta^*\| = R$ gives
$$\gamma = \frac{L\sqrt k}{1 + \sqrt k}, \qquad \alpha = \frac{L}{R(1 + \sqrt k)}, \qquad -f^* = \frac{RL}{2(1 + \sqrt k)}.$$
3. **The gradients the method sees.** At a point $\theta$, let $i$ be the first index at which $\theta_1, \dots, \theta_k$ reach their largest value, and return $g = \gamma e_i + \alpha \theta$, where $e_i$ is the $i$-th unit vector. This is a legitimate gradient for a function with a kink, a subgradient as in the chapter, because its tangent stays below the graph. The reason is that $\max_j \theta'_j \ge \theta'_i = \max_j \theta_j + \langle e_i, \theta' - \theta\rangle$, while the quadratic term lies above its own tangent. Adding the two gives $f(\theta') \ge f(\theta) + \langle g, \theta' - \theta\rangle$.
4. **One new coordinate per query.** Write $V_s$ for the vectors whose coordinates after the $s$-th are zero. The claim is that the $s$-th query point lies in $V_{s-1}$ and its gradient in $V_s$. The first query is at $0 \in V_0$. Suppose the $s$-th query point $\theta_s$ is in $V_{s-1}$, with $s \le k$. Its coordinates $s, \dots, k$ are zero, so its largest coordinate among the first $k$ is at least 0. If that largest value is 0, coordinate $s$ reaches it, so the first index that does is at most $s$. If it is positive, it is reached among the first $s - 1$ coordinates. Either way $e_i \in V_s$, and $\alpha \theta_s \in V_{s-1}$, so $g_s \in V_s$. A span-restricted method then chooses $\theta_{s+1}$ in $V_s$.
5. **The error after $T$ queries.** Any output lies in the span of $g_1, \dots, g_T$, which is inside $V_T$. Its coordinate $k = T + 1$ is zero, so its largest coordinate is at least 0 and $f(\theta) \ge \frac\alpha2\|\theta\|^2 \ge 0$. The error is therefore at least $-f^*$, which by step 2 with $k = T + 1$ is @eq-apx-opt-lower.

### Checking it

The script runs the projected subgradient method of the chapter, with the learning rate $\eta = R/(L\sqrt T)$ and the averaged iterate, against these gradients, with $R = L = 1$. @tbl-apx-opt-lower shows that its error always lies between the two bounds.

| $T$ | Lower bound @eq-apx-opt-lower | Subgradient method | SGD guarantee $RL/\sqrt T$ |
| --- | --- | --- | --- |
| 10 | 0.1158 | 0.1285 | 0.3162 |
| 100 | 0.0452 | 0.0519 | 0.1000 |
| 1,000 | 0.0153 | 0.0178 | 0.0316 |
| 10,000 | 0.0050 | 0.0058 | 0.0100 |

: Error of the averaged iterate on the hard function, $R = L = 1$, against the lower and upper bounds. {#tbl-apx-opt-lower}

No iterate ever has a negative value of $f$, as step 5 predicts. The method sits about 17% above the lower bound at $T = 10{,}000$, so the lower bound is close to exact for this method, and the upper bound is about twice the lower one. @fig-apx-opt-lower shows the three on a log scale.

![The lower bound $RL/(2(1 + \sqrt{T + 1}))$, the SGD guarantee $RL/\sqrt T$, and the error of the projected subgradient method on the hard function, against the number of gradient queries $T$, with $R = L = 1$.](figures/apx_opt_lower_bound.svg){#fig-apx-opt-lower}

### What it does not cover

The proof needs the method to stay in the span of its gradients. Nemirovski & Yudin (1983) prove the same order of bound for every method that uses only gradients, by hiding the coordinates in a random rotation that the method cannot know in advance, and their longer argument stays cited. The dimension must also be at least $T + 1$, because in low dimension methods that cut away part of the space at each query, such as the ellipsoid method, have an error that shrinks like $e^{-cT/d}$ for a constant $c$ (Bubeck 2015, ch. 2). Finally, the bound is for functions that are not smooth. On a smooth function GD with exact gradients does better than $1/\sqrt T$, and the $1/\sqrt T$ of SGD then comes from its gradient noise.

## The 1-nearest-neighbour error tends to $\mathbb E[2p(1-p)]$ {#sec-apx-opt-nn}

### What the chapter uses

The nearest neighbour (NN) rule with one neighbour, 1-NN, predicts the label of the closest training point. Let the training pairs $(x_i, y_i)$ and the test pair $(x, y)$ be independent draws from the same distribution, and write $p(x) = \Pr[y = 1 \mid x]$. Assume that $x$ lies in a closed and bounded set $K$, that $p$ is continuous on $K$, and that ties between equally near neighbours are broken by a rule that looks only at the points, not at their labels. Then the test error $R_n$ of 1-NN, averaged over the training set and the test pair, satisfies
$$\lim_{n\to\infty}R_n = \mathbb E\big[2p(x)(1 - p(x))\big].$$ {#eq-apx-opt-nn}
@prp-stability-nn states this, citing Cover & Hart (1967), and its step 3 says only that the nearest neighbour lies arbitrarily close to the test point for large $n$.

### The idea

Given where all the points are, the neighbour's label is $+1$ with probability $p(x')$, where $x'$ is the neighbour, and the test label is $+1$ with probability $p(x)$, independently. They disagree with probability
$$e(x, x') = p(x)(1 - p(x')) + (1 - p(x))p(x'),$$
which approaches $2p(x)(1 - p(x))$ as $x'$ approaches $x$. @tbl-apx-opt-nn-coins shows the two coins at a test point with $p(x) = 0.8$. Even a neighbour at the same $p$ disagrees with probability 0.32, which is $2 \times 0.8 \times 0.2$, because both labels are noisy.

| Neighbour's $p(x')$ | Test $+1$, neighbour $-1$ | Test $-1$, neighbour $+1$ | Disagree |
| --- | --- | --- | --- |
| 0.7 | $0.8 \times 0.3 = 0.24$ | $0.2 \times 0.7 = 0.14$ | 0.38 |
| 0.8 | $0.8 \times 0.2 = 0.16$ | $0.2 \times 0.8 = 0.16$ | 0.32 |

: The chance that the 1-NN prediction is wrong at a test point with $p(x) = 0.8$, for two neighbours. {#tbl-apx-opt-nn-coins}

### Derivation

1. **The error given the points.** Fix all the feature vectors, of the training set and the test point. The nearest neighbour $x'_n$ is then fixed, since it depends only on the points. Given the points, the labels are independent, each $+1$ with probability $p$ at its own point. The neighbour was chosen without looking at labels, so its label is still $+1$ with probability $p(x'_n)$. The prediction is wrong when the two labels differ, which has probability $e(x, x'_n)$. Averaging over the points gives $R_n = \mathbb E[e(x, x'_n)]$.
2. **The nearest neighbour gets close.** Fix the test point $x$ and a radius $r > 0$, and write $P_r(x)$ for the probability that one training point lands within distance $r$ of $x$. The training points are independent, so the probability that none of them does is
$$\Pr\big[\|x'_n - x\| > r \,\big|\, x\big] = (1 - P_r(x))^n,$$
which tends to 0 whenever $P_r(x) > 0$.
3. **Almost every test point has $P_r(x) > 0$.** Suppose some ball around $x$ has probability zero. Then a smaller ball inside it, whose centre has fractions as coordinates and whose radius is a fraction, also has probability zero and still contains $x$. Such a ball exists because fractions come arbitrarily close to any number: take a centre within $r/3$ of $x$ and a radius between $r/3$ and $2r/3$, where $r$ is the radius of the original ball. There are only countably many such balls, so by the union bound the union of those with probability zero has probability zero. So the test points with a zero-probability ball around them have probability zero together, and for all other test points, step 2 applies for every $r$. The averaged probability $\Pr[\|x'_n - x\| > r] = \mathbb E[(1 - P_r(x))^n]$ then tends to 0, by the **bounded convergence theorem**: if quantities between 0 and 1 tend to 0 at almost every point, so does their average.
4. **The error converges.** Expanding $e$ gives $e(x, x') - e(x, x) = (p(x') - p(x))(1 - 2p(x))$, so $|e(x, x') - e(x, x)| \le |p(x') - p(x)|$. A continuous function on a closed and bounded set is uniformly continuous. So for every tolerance $\xi > 0$ there is an $r > 0$ with $|p(x') - p(x)| \le \xi$ whenever $\|x' - x\| \le r$. Split the average according to whether the neighbour is within $r$. Inside, the difference is at most $\xi$, and outside it is at most 1, so
$$\big|R_n - \mathbb E[2p(1 - p)]\big| \le \mathbb E\big|e(x, x'_n) - e(x, x)\big| \le \xi + \Pr\big[\|x'_n - x\| > r\big].$$
By step 3 the last term tends to 0. So the limit of the left side is at most $\xi$, for every $\xi > 0$, which proves @eq-apx-opt-nn.

### Checking it

In the chapter's running example, $x$ is uniform on the unit sphere in $\mathbb R^{10}$ and $p(x) = \sigma(5x_1)$. By numerical integration over the first coordinate, the Bayes error, the error $\mathbb E[\min(p, 1 - p)]$ of the best possible classifier, is 0.250, and the limit $\mathbb E[2p(1 - p)]$ is 0.338. The sphere looks the same from every point, so every test point sees the same $P_r$, and the median distance to the nearest neighbour solves $(1 - P_r)^n = \frac12$ exactly. @tbl-apx-opt-nn-dist shows that it shrinks very slowly.

| Training points $n$ | 100 | 4,000 | $10^6$ | $10^9$ |
| --- | --- | --- | --- | --- |
| Median distance to the nearest neighbour | 0.758 | 0.489 | 0.261 | 0.121 |

: Exact median distance from a test point to its nearest neighbour, for points uniform on the unit sphere in $\mathbb R^{10}$. {#tbl-apx-opt-nn-dist}

The sphere in $\mathbb R^{10}$ is a nine-dimensional surface, so $P_r$ grows like $r^9$ and the distance shrinks only like $n^{-1/9}$, which is why the error approaches its limit slowly. With 4,000 test points, the measured 1-NN error is 0.398 at $n = 100$ and 0.356 at $n = 4{,}000$, and the average of $e(x, x'_n)$, which step 1 says has the same mean, is 0.392 and 0.358. Both are still above 0.338, because at $n = 4{,}000$ the typical neighbour is still about 0.49 away, where $p$ differs noticeably. @fig-apx-opt-nn shows both parts of the argument.

![Left: the median distance from a test point to its nearest neighbour against $n$, exact (line) and simulated (dots), for points uniform on the unit sphere in $\mathbb R^{10}$. Right: the measured 1-NN test error and the average of $p(1 - p') + (1 - p)p'$ over the same 20,000 test points, with the limit $\mathbb E[2p(1 - p)] = 0.338$.](figures/apx_opt_nn.svg){#fig-apx-opt-nn}

### What it does not cover

The limit holds without continuity of $p$, by a longer argument that approximates $p$ by continuous functions (Devroye, Györfi & Lugosi 1996, ch. 5). The proof says nothing about the speed of convergence, and the example shows that none can be given without more assumptions, because the speed depends on the dimension of the data. Finally, $k$-NN with $k$ growing and $k/n$ shrinking converges to the Bayes error itself, because averaging many neighbours' labels estimates $p(x)$ rather than drawing one more coin (Stone 1977).

## Fitted parameters are approximately Gaussian, and one record moves them by its gradient {#sec-apx-opt-mest}

### What the chapter uses

This section proves, for logistic regression, two facts that four chapters use. A record is $z = (x, y)$, with a feature vector $x$, which may include a constant 1 for an intercept, and a label $y = \pm1$. (The property-inference chapter writes the whole record as $x$.) The loss is $\ell(\theta; z) = \ln(1 + e^{-y\langle\theta, x\rangle})$. The fit $\hat\theta$ minimizes the training objective on the dataset $D = (z_1, \dots, z_n)$, and the population fit $\theta^*$ minimizes its average over the data distribution $\mathcal D$, with a ridge weight $\lambda \ge 0$:
$$\mathcal L(\theta; D) = \frac1n\sum_{i=1}^n\ell(\theta; z_i) + \frac\lambda2\|\theta\|^2, \qquad \mathcal L_{\mathcal D}(\theta) = \mathbb E_{z \sim \mathcal D}\,\ell(\theta; z) + \frac\lambda2\|\theta\|^2.$$
(The property-inference chapter writes $\theta$ for the fit and $\theta(t)$ for the population fit.) The curvature $H$ at the population fit and the covariance $G$ of one record's gradient there are
$$H = \nabla^2 \mathcal L_{\mathcal D}(\theta^*), \qquad G = \mathbb E\big[g\,g^\top\big], \quad g = \nabla\ell(\theta^*; z) + \lambda\theta^*.$$
Assume that the records are independent draws from one distribution, that $\mathbb E\|x\|^6$ is finite, and that $H$ has no zero eigenvalue, and write $h > 0$ for its smallest eigenvalue. Then $\hat\theta$ is within a distance of order $1/\sqrt n$ of $\theta^*$, and $\sqrt n(\hat\theta - \theta^*)$ **converges in distribution** to a Gaussian with the **sandwich covariance** $H^{-1}GH^{-1}$, which means that for large $n$ the probability that it lands in any box or half-space approaches the probability for the Gaussian:
$$\sqrt n\,(\hat\theta - \theta^*) \to \mathcal N\big(0,\ H^{-1}GH^{-1}\big).$$ {#eq-apx-opt-mest}
Second, fix a dataset $D$ that contains a record $z = (x, y)$, take $\lambda > 0$, and write $\phi_z = y\langle\theta, x\rangle$ for the margin of $z$, as in the membership-inference chapter. Compare the fit $\hat\theta_D$ with the fit $\hat\theta_{-z}$ of the same objective with $z$'s term removed, the other terms keeping their weight $1/n$. Training on $z$ raises its margin by
$$\phi_z(\hat\theta_D) - \phi_z(\hat\theta_{-z}) = \frac1n\,\sigma(-\phi_z)\,x^\top H^{-1}x + O(1/n^2),$$ {#eq-apx-opt-influence}
where $\phi_z$ on the right is the margin under $\hat\theta_{-z}$, $\sigma(s) = 1/(1 + e^{-s})$ is the logistic function, and $H$ is the Hessian of the full training objective at $\hat\theta_{-z}$. @sec-prop-meta sketches the first fact, citing van der Vaart (1998, ch. 5), and uses it for @eq-prop-delta and @eq-prop-kappa. @sec-mia-lira uses it for the Gaussian shape of $\phi_z$, @sec-mia-outliers explains the self-influence $\mu_{\mathrm{in}, z} - \mu_{\mathrm{out}, z}$ of @eq-mia-per-example in words, and @sec-attr-inversion asserts that the model approaches the population's probabilities.

### The idea

The simplest case shows why the covariance is a sandwich. For the loss $\frac12(\theta - a)^2$ of a number $a$, the fit is the sample mean. The curvature is $H = 1$ and the gradient $\theta^* - a$ has variance $G = \operatorname{Var}(a)$, so the sandwich gives the familiar $\operatorname{Var}(a)/n$. Doubling the loss to $(\theta - a)^2$ doubles the curvature and quadruples the gradient variance. The two copies of $H^{-1}$ undo the quadrupling, so the fit's variance does not change, as it should not, since the fit is still the mean.

### Derivation

The proof first shows that the fit is close to $\theta^*$, because near $\theta^*$ the objective curves up and its gradient is small (steps 1 to 4). Then the condition "gradient zero at $\hat\theta$" becomes an exact linear equation for the error (step 5), whose matrix approaches $H$ (step 6) and whose right side is approximately Gaussian by the central limit theorem (step 7). Steps 8 to 10 combine these and apply them.

1. **The curvature of the logistic loss changes slowly.** The Hessian of one record's loss is $s(\langle\theta, x\rangle)\,xx^\top$, with $s(m) = \sigma(m)(1 - \sigma(m))$. Its derivative is $s'(m) = \sigma(1 - \sigma)(1 - 2\sigma)$, whose largest size is $c = 1/(6\sqrt3) \approx 0.0962$. To see this, write $\sigma = \frac12 + v$, so that $|s'| = 2|v|(\frac14 - v^2)$, which is largest at $v = 1/(2\sqrt3)$. The matrix $xx^\top$ stretches no vector by more than $\|x\|^2$, and $|\langle\theta - \theta', x\rangle| \le \|\theta - \theta'\|\,\|x\|$, so the Hessian moves by at most $c\|x\|^3\|\theta - \theta'\|$. Averaging over the records, with $\hat M_3 = \frac1n\sum_i\|x_i\|^3$,
$$\big\|\nabla^2\mathcal L(\theta; D) - \nabla^2\mathcal L(\theta'; D)\big\| \le c\,\hat M_3\,\|\theta - \theta'\|.$$
2. **Three averages settle down.** Each of $\nabla^2\mathcal L(\theta^*; D)$, $\hat M_3$ and $\nabla\mathcal L(\theta^*; D)$ is an average of $n$ independent terms. Their means are $H$, the third moment $M_3 = \mathbb E\|x\|^3$, and 0. The last mean is 0 because $\theta^*$ minimizes $\mathcal L_{\mathcal D}$, so $\mathbb E g = \nabla \mathcal L_{\mathcal D}(\theta^*) = 0$. Their variances are finite, because the sixth moment is. By Chebyshev's inequality, an average of $n$ terms with variance $v$ strays from its mean by more than $a$ with probability at most $v/(na^2)$. (For a matrix, apply it to the sum of the squared entries, which is at least the square of the largest factor by which the matrix stretches a vector.) So each average is near its mean with probability tending to 1. Choose the radius $r = h/(8cM_3)$, whose role step 3 shows, and call $E_n$ the event that all three are near their means in the following sense:
$$\big\|\nabla^2\mathcal L(\theta^*; D) - H\big\| \le \frac h4, \qquad \hat M_3 \le 2M_3, \qquad \big\|\nabla\mathcal L(\theta^*; D)\big\| < \frac{hr}4.$$
Then $E_n$ holds with probability tending to 1. For the gradient the mean square is exact: the cross terms of independent mean-zero vectors average to zero, so
$$\mathbb E\big\|\nabla\mathcal L(\theta^*; D)\big\|^2 = \frac{\operatorname{tr}G}{n}.$$
3. **The objective curves up near $\theta^*$.** On $E_n$, take any $\theta$ within distance $r$ of $\theta^*$. By step 1 its Hessian differs from $\nabla^2\mathcal L(\theta^*; D)$ by at most $c \cdot 2M_3 \cdot r = h/4$, and that one differs from $H$ by at most $h/4$. Since $H \succeq hI$, every eigenvalue of $\nabla^2\mathcal L(\theta; D)$ is at least $h - h/4 - h/4 = h/2$. So, by step 2 of @sec-apx-opt-convexity applied to segments inside the ball, $\mathcal L(\cdot; D)$ is $(h/2)$-strongly convex on the ball of radius $r$ around $\theta^*$.
4. **The fit is inside the ball, and close.** On the sphere $\|\theta - \theta^*\| = r$, strong convexity and Cauchy–Schwarz give $\mathcal L(\theta; D) \ge \mathcal L(\theta^*; D) - \|\nabla\mathcal L(\theta^*; D)\|r + \frac h4r^2$. By the third condition of $E_n$, this is larger than $\mathcal L(\theta^*; D)$. A convex function that is higher on a whole sphere than at its centre cannot be lower anywhere outside: if some outside point $\theta$ had $\mathcal L(\theta; D) \le \mathcal L(\theta^*; D)$, the segment from $\theta^*$ to $\theta$ would cross the sphere, and by the chord inequality the value at the crossing would be at most $\mathcal L(\theta^*; D)$. So the minimizer $\hat\theta$ is inside the ball. There, part (a) of @cor-opt-strong, with strong convexity $h/2$ and $\nabla\mathcal L(\hat\theta; D) = 0$, gives
$$\|\hat\theta - \theta^*\| \le \frac{2}{h}\,\|\nabla\mathcal L(\theta^*; D)\|,$$ {#eq-apx-opt-consistency}
whose square has mean at most $4\operatorname{tr}G/(h^2n)$ by step 2. So the fit is within a distance of order $1/\sqrt n$ of the population fit, and in particular it converges to it.
5. **An exact linear equation for the error.** The fit has zero gradient. By the fundamental theorem of calculus along the segment from $\theta^*$ to $\hat\theta$, as in step 4 of the proof of @prp-opt-constants, $0 = \nabla\mathcal L(\hat\theta; D) = \nabla\mathcal L(\theta^*; D) + \bar H_n(\hat\theta - \theta^*)$. Here $\bar H_n$ is the average of $\nabla^2\mathcal L(\cdot; D)$ along the segment, whose eigenvalues are at least $h/2$ by step 3, so it can be inverted:
$$\hat\theta - \theta^* = -\bar H_n^{-1}\,\nabla\mathcal L(\theta^*; D).$$ {#eq-apx-opt-linear}
This is the chapter's step 2 with "$\approx$" replaced by "$=$" and $H$ replaced by $\bar H_n$.
6. **The average curvature approaches $H$.** By step 1, $\bar H_n$ differs from $\nabla^2\mathcal L(\theta^*; D)$ by at most $c\hat M_3\|\hat\theta - \theta^*\|/2$, which tends to 0 by step 4. By step 2, $\nabla^2\mathcal L(\theta^*; D)$ tends to $H$. The inverses follow, because $\bar H_n^{-1} - H^{-1} = \bar H_n^{-1}(H - \bar H_n)H^{-1}$ has size at most $\frac2h \cdot \frac1h\|H - \bar H_n\|$.
7. **The average gradient is approximately Gaussian.** The scaled gradient $\sqrt n\,\nabla\mathcal L(\theta^*; D) = \frac1{\sqrt n}\sum_ig_i$ is a scaled sum of independent vectors with mean 0 and covariance $G$. Take any fixed direction $a$. The numbers $\langle a, g_i\rangle$ are independent with mean 0 and variance $a^\top Ga$, so the central limit theorem in one dimension (@sec-apx-prob-clt) makes their scaled sum approximately $\mathcal N(0, a^\top Ga)$, which is the distribution of $\langle a, Y\rangle$ for $Y \sim \mathcal N(0, G)$. The **Cramér–Wold device** says that the distributions of all the one-dimensional views $\langle a, \cdot\rangle$ determine the distribution of a vector, also in the limit (van der Vaart 1998, §2.2). So $\sqrt n\,\nabla\mathcal L(\theta^*; D)$ approaches $\mathcal N(0, G)$.
8. **Combine.** Multiply @eq-apx-opt-linear by $\sqrt n$ and split it in two:
$$\sqrt n(\hat\theta - \theta^*) = -H^{-1}\sqrt n\,\nabla\mathcal L(\theta^*; D) + (H^{-1} - \bar H_n^{-1})\sqrt n\,\nabla\mathcal L(\theta^*; D).$$
By step 7 the first term approaches $-H^{-1}Y$. A linear map $A$ turns a Gaussian with covariance $G$ into one with covariance $AGA^\top$, which here is $H^{-1}GH^{-1}$ because $H$ is symmetric. In the second term, the matrix tends to 0 by step 6, while the vector has mean square $\operatorname{tr}G$ for every $n$, so their product tends to 0 in probability, by Chebyshev's inequality. Adding a term that tends to 0 in probability does not change the limiting distribution (Slutsky's lemma, van der Vaart 1998, Lemma 2.8). This proves @eq-apx-opt-mest. The event $E_n$ fails with probability tending to 0, so it changes no limit.
9. **When the model is right, $G = H$.** Suppose $\lambda = 0$ and the labels really follow the model, $\Pr[y = 1 \mid x] = \sigma(\langle\theta^*, x\rangle) = p$. One record's gradient is $-\sigma(-y\langle\theta^*, x\rangle)\,y\,x$, so $gg^\top$ is $(1 - p)^2xx^\top$ when $y = 1$, which has probability $p$, and $p^2xx^\top$ when $y = -1$. On average, given $x$, that is $p(1 - p)^2 + (1 - p)p^2 = p(1 - p)$ times $xx^\top$, which is exactly the Hessian's weight $s$. So $G = H$ and the covariance is $H^{-1}$, the inverse of the Fisher information. The same case answers @sec-attr-inversion. The expected log-loss at each $x$ is smallest when the predicted probability equals the true one, so the population fit of a correct model is the true parameter, and by step 4 the fitted model converges to the population's probabilities.
10. **The influence of one record.** Now fix the dataset $D$ and the record $z = (x, y)$ in it, with $\lambda > 0$. The objective without $z$'s term is $\mathcal L_{-z}$, and with it $\mathcal L(\cdot; D) = \mathcal L_{-z} + \frac1n\ell(\cdot; z)$. At $\hat\theta_{-z}$ the gradient of $\mathcal L_{-z}$ is zero, so the gradient of $\mathcal L(\cdot; D)$ there is just $\frac1n\nabla\ell(\hat\theta_{-z}; z)$. Step 5, applied to $\mathcal L(\cdot; D)$ around $\hat\theta_{-z}$ instead of around $\theta^*$, gives exactly $\hat\theta_D - \hat\theta_{-z} = -\frac1n\bar H^{-1}\nabla\ell(\hat\theta_{-z}; z)$, with $\bar H$ the average Hessian of $\mathcal L(\cdot; D)$ along the segment. This move is small: $\mathcal L(\cdot; D)$ is $\lambda$-strongly convex and the logistic gradient has length at most $\|x\|$, so consequence (a) gives $\|\hat\theta_D - \hat\theta_{-z}\| \le \|x\|/(\lambda n)$. So replacing $\bar H$ by the Hessian $H$ of $\mathcal L(\cdot; D)$ at $\hat\theta_{-z}$ changes the matrix by $O(1/n)$, by step 1, and the move by $O(1/n^2)$. The logistic gradient is $\nabla\ell = -\sigma(-\phi_z)\,y\,x$, so the move is $\frac1n\sigma(-\phi_z)\,y\,H^{-1}x$. The margin $\phi_z = y\langle \theta, x\rangle$ changes by $y$ times the inner product of the move with $x$, which is $\frac1n\sigma(-\phi_z)\,y^2\,x^\top H^{-1}x$. Since $y^2 = 1$, that is @eq-apx-opt-influence.

Step 10 turns the heuristic of @sec-mia-outliers into a formula. The factor $\sigma(-\phi_z)$ is close to 1 when the model trained without $z$ gets its label wrong, and close to 0 when it already predicts it confidently. The factor $x^\top H^{-1}x$ counts the length of $x$ twice, once for how hard $z$ pulls the weights and once for how much a move of the weights changes $z$'s own margin, and divides by the curvature in that direction. In the chapter, the models without $z$ train on the same number of records, with another record in $z$'s place, which pulls the fit by its own gradient. Averaged over which record it is, that pull is the average gradient of the loss, which is close to zero near the population fit, where only the small ridge term $\lambda \theta$ balances it. So, to first order, the self-influence $\mu_{\mathrm{in}, z} - \mu_{\mathrm{out}, z}$ is the average of @eq-apx-opt-influence over training sets.

### Checking it

The script checks the Gaussian picture on the logistic-regression toy of @sec-prop-meta. The features are $x \sim \mathcal N(0, I_2)$ with an intercept. Records from the majority get the label $+1$ with probability $\sigma(2x_1 + x_2)$, records from a hidden subgroup with probability $\sigma(2x_1 - x_2)$, and the subgroup makes up a share $t = \frac12$ of the population. The chapter trains its models with a ridge weight $10^{-3}$, and including it in $\mathcal L_{\mathcal D}$, as the theorem allows, gives by quadrature the population fit $(1.683, 0, 0)$ for $(\theta_1, \theta_2, \text{intercept})$ and the sandwich diagonal $(11.83, 6.76, 5.89)$. @tbl-apx-opt-mest compares $n$ times the variance over 2,000 refits with that sandwich. It also shows the share of refits whose $\theta_2$ lands within 1.96 predicted standard deviations of its centre, which is 0.950 for an exact Gaussian.

| $n$ | $n\operatorname{Var}(\theta_1)$ | $n\operatorname{Var}(\theta_2)$ | $n\operatorname{Var}(\text{intercept})$ | $\theta_2$ within $\pm1.96$ sd |
| --- | --- | --- | --- | --- |
| 50 | 22.06 | 9.62 | 7.94 | 0.902 |
| 200 | 13.61 | 7.48 | 6.16 | 0.935 |
| 1,000 | 11.72 | 6.95 | 5.92 | 0.950 |
| sandwich | 11.83 | 6.76 | 5.89 | 0.950 |

: Scaled variances of logistic regressions refitted on 2,000 training sets at subgroup share $t = \frac12$, against the sandwich covariance, which includes the ridge weight $10^{-3}$. {#tbl-apx-opt-mest}

At $n = 1{,}000$ every variance is within 3% of the sandwich, about the sampling error of a variance estimated from 2,000 refits. At $n = 50$ the variances are larger and the tails heavier, because the linearization of step 5 is not yet accurate. The ridge weight also accounts for most of the gap of about 5% in $\theta_1$ that @sec-prop-meta finds against the sandwich without it, whose value is 12.59. @fig-apx-opt-mest-hist shows the histograms becoming Gaussian, and @fig-apx-opt-mest-variance (left) shows the scaled variances settling on the sandwich.

![Histograms of $\sqrt n(\hat\theta_2 - \theta_2^*)$ over 2,000 logistic regressions refitted at subgroup share $t = \frac12$, for $n = 50$, 200 and 1,000, with the normal density whose variance is the sandwich value.](figures/apx_opt_mest_hist.svg){#fig-apx-opt-mest-hist}

Step 9 checks out too, with $\lambda = 0$ as it requires. With the majority's rule alone, $t = 0$, the model is correct, its population fit is the true weights $(2.000, 1.000, 0.000)$, and $G$ and $H$ agree within $10^{-12}$. At $t = \frac12$ no single logistic rule describes the mixture, and the diagonals differ: $(0.076, 0.191, 0.166)$ for $G$ against $(0.078, 0.166, 0.166)$ for $H$.

The influence formula is checked on the running example of the membership-inference chapter: $d = 100$ features, 950 typical points and 50 outliers with five times longer $x$ and a random label, and $\lambda = 10^{-3}$. One model is trained on $n = 500$ points, and for 200 of them the script removes the point's term, refits, and compares the change in its margin with @eq-apx-opt-influence. For the 188 typical points, the mean change is 0.2287 against a predicted 0.2291. For the 12 outliers, it is 3.92 against a predicted 8.21. An outlier moves its own margin by several units, far outside the range where the linear expansion is accurate, because the $O(1/n^2)$ term of step 10 has a constant that grows with $\|x\|$ and with $1/\lambda$. So the formula ranks the records correctly but overstates the largest influences, as @fig-apx-opt-mest-variance (right) shows.

![Left: $n$ times the variance of each fitted parameter over 2,000 refits, against $n$, with the sandwich values (dashed). Right: for points of the membership-inference running example, the measured change of the margin $\phi_z$ when the point's term is added, against the first-order prediction @eq-apx-opt-influence. Typical points lie on the diagonal; outliers fall below it.](figures/apx_opt_mest_variance.svg){#fig-apx-opt-mest-variance}

### What it does not cover

The result is a limit, so it says nothing about how large $n$ must be, and a finite-sample version (a Berry–Esseen bound) needs further moment assumptions. The proof also needs $H$ to be invertible and a minimizer to exist. Both can fail with $\lambda = 0$: when a line separates the classes perfectly, the loss keeps falling as the weights grow, so no minimizer exists. When the number of features grows with $n$, so that $d/n$ does not go to zero, the sandwich is wrong: the fitted weights are pushed away from zero, and their variance is larger than predicted (Sur & Candès 2019). This matters for the membership example, where $d/n = 0.2$. The general theory of M-estimators, the fits that minimize an average loss, for other losses and without convexity, is in van der Vaart (1998, ch. 5). For deep networks, which have many minima, the Gaussian shape of $\phi_z$ that @sec-mia-lira relies on is an empirical finding of Carlini et al., not a consequence of this theorem.

## The best membership attack looks at the parameters through the loss {#sec-apx-opt-gibbs}

### What the chapter uses

There are $n$ candidate records $z_1, \dots, z_n$. Each is a member of the training set independently with probability $\pi_1$, and $m_i \in \{0, 1\}$ marks whether $z_i$ is one. Training draws the parameters $\theta$ from a **Gibbs posterior** at a temperature $\nu > 0$, a density that falls exponentially in the total training loss:
$$p(\theta \mid m) = \frac{1}{Z(m)}\exp\Big(-\frac1\nu\sum_{i=1}^n m_i\,\ell(\theta; z_i)\Big),$$
where $Z(m)$ makes the density integrate to 1. Write $m_{-1}$ for the memberships of the other candidates, and $\sigma$ for the logistic function. An attacker who knows the candidates and sees $\theta$ finds that the target $z_1$ is a member with probability exactly
$$\Pr[m_1 = 1 \mid \theta] = \mathbb E\Big[\sigma\Big(-\frac{\ell(\theta; z_1) - \tau(z_1, m_{-1})}{\nu} + \ln\frac{\pi_1}{1 - \pi_1}\Big)\,\Big|\,\theta\Big],$$ {#eq-apx-opt-gibbs}
with the calibration
$$\tau(z_1, m_{-1}) = -\nu\ln\mathbb E_{\theta' \sim p(\cdot \mid m_1 = 0,\ m_{-1})}\big[e^{-\ell(\theta'; z_1)/\nu}\big].$$
Inside the logistic function, $\theta$ appears only through the loss. It also appears in the outer average over the unknown $m_{-1}$, so replacing $\tau$ by one number $\tau(z_1)$ per target, which gives an attack that thresholds $\ell(\theta; z_1) - \tau(z_1)$, is Sablayrolles et al.'s (2019) approximation, not a theorem. For the **centroid detector** of @sec-threat-models no approximation is needed. There the model is the mean $\theta = \frac1n\sum_ix_i$ of $n$ records $x_i \sim \mathcal N(0, I_d)$, the loss is $\ell(\theta; z) = \frac12\|\theta - z\|^2$, and the log likelihood ratio between the member world, density $q(\theta \mid z)$, and the non-member world, density $p(\theta \mid z)$, is exactly
$$\ln\frac{q(\theta \mid z)}{p(\theta \mid z)} = -\frac{n}{n - 1}\big(\ell(\theta; z) - \tau(z)\big) + \frac d2\ln\frac{n}{n - 1}, \qquad \tau(z) = \frac{(n - 1)\|z\|^2}{2n}.$$ {#eq-apx-opt-centroid}
The chapter cites Sablayrolles et al. to explain why, in its chain @eq-threat-access-chain, access to the weights gives no better attack than the calibrated loss of each query.

### The idea

Two targets with the same loss show why the calibration matters. Take $\nu = 1$ and $\pi_1 = \frac12$, so that the prior term $\ln(\pi_1/(1 - \pi_1))$ is zero. A hard target has $\tau = 2$, because models trained without it give it a typical loss near 2. If the released model gives it loss 0.5, the posterior is $\sigma(2 - 0.5) = \sigma(1.5) = 0.8176$, likely a member. An easy target has $\tau = 0.3$, because every model fits it well, and the same loss 0.5 gives $\sigma(0.3 - 0.5) = \sigma(-0.2) = 0.4502$, slightly less likely than not. So a low loss is evidence of membership only relative to what the target's loss would be without it.

### Derivation

1. **Bayes' rule with the others fixed.** Write $A = \pi_1\,p(\theta \mid m_1 = 1, m_{-1})$ and $B = (1 - \pi_1)\,p(\theta \mid m_1 = 0, m_{-1})$ for the two joint weights. Bayes' rule says $\Pr[m_1 = 1 \mid \theta, m_{-1}] = A/(A + B)$. A ratio of this form equals $\sigma(\ln(A/B))$, because $1/(1 + B/A) = 1/(1 + e^{-\ln(A/B)})$. So
$$\Pr[m_1 = 1 \mid \theta, m_{-1}] = \sigma\Big(\ln\frac{p(\theta \mid m_1 = 1, m_{-1})}{p(\theta \mid m_1 = 0, m_{-1})} + \ln\frac{\pi_1}{1 - \pi_1}\Big).$$
2. **The density ratio.** The two exponents differ by the target's term only, so the density ratio is $e^{-\ell(\theta; z_1)/\nu}\,Z_0/Z_1$, where $Z_1$ and $Z_0$ are the normalizing constants with and without the target. Write $Z_1$ as the integral of $e^{-\ell(\theta'; z_1)/\nu}$ times the unnormalized density without the target. Dividing by $Z_0$ turns that integral into an average over $\theta'$ drawn from the posterior without the target:
$$\frac{Z_1}{Z_0} = \mathbb E_{\theta' \sim p(\cdot \mid m_1 = 0,\ m_{-1})}\big[e^{-\ell(\theta'; z_1)/\nu}\big] = e^{-\tau(z_1, m_{-1})/\nu}.$$
So the log density ratio is $-(\ell(\theta; z_1) - \tau(z_1, m_{-1}))/\nu$, a function of $\theta$ only through the loss.
3. **Average over the unknown others.** By the law of total probability, $\Pr[m_1 = 1 \mid \theta]$ is the average of $\Pr[m_1 = 1 \mid \theta, m_{-1}]$ over $m_{-1}$, weighted by $\Pr[m_{-1} \mid \theta]$. Together with steps 1 and 2 that is @eq-apx-opt-gibbs. The weights depend on $\theta$, which is why replacing $\tau(z_1, m_{-1})$ by a single $\tau(z_1)$ is an approximation.
4. **The centroid: two Gaussians.** If $z$ is a member, $\theta = (z + S)/n$, where $S$, the sum of the other $n - 1$ records, is $\mathcal N(0, (n - 1)I_d)$. So $\theta$ is Gaussian with mean $z/n$ and variance $(n - 1)/n^2$ per coordinate. If it is not, all $n$ records are fresh draws, so $\theta$ is Gaussian with mean 0 and variance $1/n$ per coordinate. The log of each density is minus the squared distance to the mean over twice the variance, plus $-\frac d2\ln$ of the variance. Their difference is
$$-\frac{n^2}{2(n - 1)}\Big\|\theta - \frac zn\Big\|^2 + \frac n2\|\theta\|^2 + \frac d2\ln\frac{n}{n - 1}.$$
5. **Expand the squares.** The first square is $\|\theta\|^2 - \frac2n\langle z, \theta\rangle + \frac1{n^2}\|z\|^2$. Collecting terms, $\|\theta\|^2$ has coefficient $\frac n2 - \frac{n^2}{2(n - 1)} = -\frac{n}{2(n - 1)}$, and $\langle z, \theta\rangle$ has coefficient $\frac{n}{n - 1}$. So the log ratio is $\frac{n}{2(n - 1)}(2\langle z, \theta\rangle - \|\theta\|^2) - \frac{\|z\|^2}{2(n - 1)} + \frac d2\ln\frac{n}{n - 1}$, the chapter's formula with its constant made explicit.
6. **Write it with the loss.** Since $\ell(\theta; z) = \frac12\|z\|^2 - \frac12(2\langle z, \theta\rangle - \|\theta\|^2)$, the bracket of step 5 equals $\|z\|^2 - 2\ell(\theta; z)$. Substituting, the $\|z\|^2$ terms are $\frac{n}{2(n - 1)}\|z\|^2 - \frac{1}{2(n - 1)}\|z\|^2 = \frac12\|z\|^2$, which is $\frac{n}{n - 1}\tau(z)$. The result is @eq-apx-opt-centroid. It has the Gibbs form exactly, with temperature $(n - 1)/n$, because the other records enter only through their sum $S$, whose distribution does not depend on which records they are.

Step 5 also explains why the chapter's white-box and calibrated black-box attacks agree to four decimals. The calibrated score $\|z\|^2 - \|z - \theta\|^2$ is the bracket $2\langle z, \theta\rangle - \|\theta\|^2$, so the log ratio is a fixed multiple of it minus the term $\|z\|^2/(2(n - 1))$. That term changes from target to target, so the two scores do not rank the games (pairs of model and target) in exactly the same order. But it changes very little: at $n = 100$ and $d = 50$ its standard deviation is 0.0504, against 0.757 for the log ratio itself.

### Checking it

The script checks @eq-apx-opt-gibbs exactly on a small Gibbs posterior: 10 candidates $z_i \sim \mathcal N(0, 1)$, the loss $\frac12(\theta - z)^2$, $\nu = 1$, $\pi_1 = \frac12$, and $\theta$ on a grid of 801 points in $[-4, 4]$. Enumerating all $2^9$ memberships of the other candidates, the exact posterior and the right side of @eq-apx-opt-gibbs differ by less than $10^{-12}$. Replacing $\tau(z_1, m_{-1})$ by the single value $\bar\tau(z_1) = -\nu\ln\mathbb E_{m_{-1}}[e^{-\tau(z_1, m_{-1})/\nu}]$, averaged over memberships drawn from their prior, changes the posterior by up to 0.054 where $\theta$ carries 99.9% of its probability. So in this small example the approximation is reasonable but visible.

For the centroid, @eq-apx-opt-centroid matches the log ratio computed from the two densities to within $10^{-10}$. Over 100,000 games per world, the best advantage is 0.2804 for the exact likelihood ratio, 0.2800 for the calibrated score and 0.0422 for the raw loss. @fig-apx-opt-gibbs shows why calibration matters. Against the raw loss the exact log ratio is a cloud, because a target's loss mixes its membership with how typical it is, while against $\ell - \tau(z)$ it is a line of slope $-n/(n - 1)$.

![For 1,500 member and 1,500 non-member games of the centroid detector ($n = 100$, $d = 50$), the exact log likelihood ratio against the raw loss $\ell(\theta; z)$ (left) and against the calibrated loss $\ell(\theta; z) - \tau(z)$ (right).](figures/apx_opt_gibbs.svg){#fig-apx-opt-gibbs}

### What it does not cover

Real training is not a Gibbs posterior. SGD with a constant learning rate behaves like one only roughly, and an exactly trained model, like the centroid, is deterministic given its data. The centroid escapes the problem because, from the attacker's view, the unknown other records make $\theta$ random, and the result has the Gibbs form anyway. In general the per-target calibration $\tau(z)$ must be estimated, which is what shadow models do in the per-example attacks of [Membership inference](../part02-attacks/02-membership-inference.md).

## Code

The script [apx_optimization.py](../../code/appendix-derivations/apx_optimization.py) computes every number on this page and draws every figure. It checks the inequalities on random pairs and on gradient-descent runs. It runs the subgradient method against the resisting gradients and computes the exact nearest-neighbour distances on the sphere. It also refits logistic regressions thousands of times, compares leave-one-out refits with the influence formula, and enumerates a small Gibbs posterior. From the repository root:

```bash
uv run python code/appendix-derivations/apx_optimization.py
```

@tbl-apx-opt-code collects the main numbers it prints.

| Check | Result |
| --- | --- |
| Strong convexity of the ridge logistic risk, $\lambda = 0.01$ | smallest ratio 0.0372 over 2,000 pairs |
| GD, largest gap ratio per step against $1 - 1/\kappa = 0.9615$ | 0.7131 |
| GD gap after 50 steps, against @eq-apx-opt-gd-values | $2.59\times10^{-9}$ against $1.99\times10^{-2}$ |
| Co-coercivity, smallest ratio of the two sides | 1.0006 (logistic loss), 5.3704 (logistic risk) |
| Worst distance ratio after one step, $\eta = 8$ and $\eta = 10$ | 0.9988 and 1.4985 |
| Early stopping, $n = 100$, $\Vert\theta^*\Vert = 5$, $L = 1$, $\eta = 0.25$: $u^*$ and $T^*$ | 25 (also the grid minimum) and 100 steps |
| Early stopping: the two terms at $u^*$, and the bound | 0.5 each; $1 + 0.125 = 1.125$ |
| Hard function, $T = 10{,}000$: lower bound, method, SGD bound | 0.0050, 0.0058, 0.0100 |
| 1-NN limit and Bayes error | 0.338 and 0.250 |
| 1-NN error and plug-in average at $n = 4{,}000$ | 0.356 and 0.358 |
| Sandwich diagonal at $t = \frac12$, ridge $10^{-3}$ | $(11.83, 6.76, 5.89)$ |
| $n\operatorname{Var}$ over 2,000 refits at $n = 1{,}000$ | $(11.72, 6.95, 5.92)$ |
| Influence, median measured/predicted: typical points, outliers | 1.003, 0.671 |
| Gibbs identity, largest error | below $10^{-12}$ |
| Centroid advantage: likelihood ratio, calibrated score, raw loss | 0.2804, 0.2800, 0.0422 |

: The main numbers printed by the appendix's script. {#tbl-apx-opt-code}

Every bound holds where it should, and the checks also show how loose each one is. The linear rate of GD is exact in form but pessimistic in its curvature, while co-coercivity is tight where the loss is most curved, and the lower bound is within 17% of what the subgradient method achieves. The Gaussian picture of a fitted model is accurate to sampling error at $n = 1{,}000$, while its first-order influence formula fails exactly for the records that matter most to membership attacks, the outliers.

## References

- Yurii Nesterov, *Lectures on Convex Optimization*, 2nd ed. (2018), §2.1. The characterizations of convex and strongly convex functions, co-coercivity (Theorem 2.1.5), and the inequality that gives the distance rate of GD without second derivatives (Theorem 2.1.12).
- Boris Polyak, "Gradient methods for minimizing functionals" (1963), and Hamed Karimi, Julie Nutini & Mark Schmidt, "Linear convergence of gradient and proximal-gradient methods under the Polyak–Łojasiewicz condition" (2016). The inequality of step 5 in @sec-apx-opt-convexity and how far it reaches beyond convexity.
- Moritz Hardt, Benjamin Recht & Yoram Singer, "Train faster, generalize better: Stability of stochastic gradient descent" (2016). The stability of SGD on convex and non-convex losses, which @sec-apx-opt-early-stop balances against its optimization error.
- Shai Shalev-Shwartz & Shai Ben-David, *Understanding Machine Learning* (2014), ch. 14. The optimization error of SGD on convex Lipschitz losses, used in @sec-apx-opt-early-stop.
- Sébastien Bubeck, "Convex Optimization: Algorithms and Complexity" (2015), Theorem 3.13. The hard function of @sec-apx-opt-lower, and the cutting-plane methods of ch. 2.
- Arkadi Nemirovski & David Yudin, *Problem Complexity and Method Efficiency in Optimization* (1983). The lower bound for every gradient method, not only span-restricted ones.
- Thomas Cover & Peter Hart, "Nearest neighbor pattern classification" (1967). The original 1-NN limit.
- Luc Devroye, László Györfi & Gábor Lugosi, *A Probabilistic Theory of Pattern Recognition* (1996), ch. 5. The 1-NN limit without continuity, and nearest-neighbour rules in general.
- Charles Stone, "Consistent nonparametric regression" (1977). $k$-NN rules that reach the Bayes error.
- Aad van der Vaart, *Asymptotic Statistics* (1998), ch. 2 and ch. 5. The Cramér–Wold device, Slutsky's lemma, and the asymptotic normality of M-estimators in general.
- Peter Huber, "The behavior of maximum likelihood estimates under nonstandard conditions" (1967). The sandwich covariance for a misspecified model.
- Pang Wei Koh & Percy Liang, "Understanding black-box predictions via influence functions" (2017). The influence formula of step 10 as a tool for explaining predictions.
- Pragya Sur & Emmanuel Candès, "A modern maximum-likelihood theory for high-dimensional logistic regression" (2019). What replaces the sandwich when $d/n$ does not go to zero.
- Alexandre Sablayrolles, Matthijs Douze, Yann Ollivier, Cordelia Schmid & Hervé Jégou, "White-box vs black-box: Bayes optimal strategies for membership inference" (2019). The Gibbs-posterior result of @sec-apx-opt-gibbs and its approximation.
