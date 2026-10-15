# Optimization

## Why this matters

Training a model minimizes the average loss over the training set, $F(w) = \frac1n\sum_i\ell(w, z_i)$, over parameters $w$, almost always with stochastic gradient descent (SGD). Two questions come up every time. How fast does SGD get there, and how should you set its step size when its gradients are noisy? And how much does the answer depend on any single training example: if you replace one and retrain, does the minimizer barely move, or can it jump?

Both answers come from a few constants of the loss: how steep it can be, how fast its slope can change, how strongly it curves upward, and how much the random gradients scatter. Every result here is a statement about them, so know which ones a result assumes, and whether your loss has them.

## Lipschitz, smooth and strongly convex losses {#sec-opt-definitions}

Four properties of the loss matter. A loss is **$L$-Lipschitz** when its gradient norm never exceeds $L$, so moving the parameters by $d$ changes the loss by at most $Ld$. It is **$\beta$-smooth** when its gradient changes by at most $\beta$ per unit of distance; then the tangent is a trustworthy guide for a step of about $1/\beta$, and that is the largest safe step size. It is convex when its curvature is never negative, and **$\lambda$-strongly convex** when the curvature is at least $\lambda$ in every direction; adding $\frac\lambda2\|w\|^2$ to a convex loss makes it so.

The logistic loss $\ln(1 + e^{-y\langle w, x\rangle})$, with features scaled to $\|x\| \le 1$, is the standard example. Its slope is a probability times $x$, so it is 1-Lipschitz; its curvature is $p(1 - p) \le \frac14$, so it is $\frac14$-smooth. It is convex but not strongly convex, because far from the data it flattens out. A ridge term makes it $\lambda$-strongly convex and $(\frac14 + \lambda)$-smooth.

Lipschitz and smooth bound different derivatives, so neither implies the other: $|w|$ is 1-Lipschitz but its slope jumps at zero, and $w^2/2$ is 1-smooth but its slope is unbounded. The parabola is the general case: a strongly convex function curves upward everywhere, so its slope grows at least in proportion to the distance from the minimizer, and it cannot be Lipschitz on all of $\mathbb R^d$. The ridge-regularized logistic loss is therefore Lipschitz only on a ball $\|w\| \le R$, with gradient at most $1 + \lambda R$ there. Algorithms keep the iterates in the ball by **projection**, $\Pi(u) = u\min(1, R/\|u\|)$, which never moves a point farther from anything inside the ball.

## SGD convergence {#sec-opt-sgd}

SGD steps against the gradient of one random example, or of a small random batch: cheap, right on average, but noisy. Even at the minimizer the individual gradients are not zero, so with a fixed step the iterates keep jumping around it. Averaging the iterates cancels the jumps; shrinking the step makes them smaller.

For a convex loss, the error of the averaged iterate after $T$ steps of size $\eta$, starting within $R$ of the minimizer, with random gradients of mean squared size at most $G^2$, is
$$\mathbb E[F(\bar w_T)] - F^* \le \frac{R^2}{2\eta T} + \frac{\eta G^2}{2}.$$
The two terms are a trade-off. The first is the cost of travelling the distance $R$ with steps of size $\eta$: small steps take long. The second is the jitter each step adds: large steps overshoot. Balancing them gives the best step $\eta = R/(G\sqrt T)$ and an error of $RG/\sqrt T$, with $G = L$ for an $L$-Lipschitz loss. The bound holds for the average of the iterates, not the last one, so average.

@tbl-opt-sgd-convex tests the bound on logistic regression with $n = 1{,}000$, $d = 5$, $R = 5$ and $L = 1$.

| Steps $T$ | Step $\eta$ | Observed $F(\bar w_T) - F^*$ | Bound $RL/\sqrt T$ |
| --- | --- | --- | --- |
| 100 | 0.5000 | $4.44\times10^{-2}$ | 0.500 |
| 1,000 | 0.1581 | $6.46\times10^{-3}$ | 0.158 |
| 10,000 | 0.0500 | $7.71\times10^{-4}$ | 0.050 |

: Projected SGD on logistic regression against the convex Lipschitz bound, averaged over 20 runs. {#tbl-opt-sgd-convex}

The bound holds, loose by a factor of 11 at $T = 100$ and 65 at $T = 10{,}000$. The gap grows because the observed error falls almost like $1/T$: the data curve the loss near the minimizer, and the convex bound does not know it. Every constant in a bound is a worst case the run does not reach, so trust how a bound scales, not its size.

Curvature is what speeds SGD up. On a $\lambda$-strongly convex loss, the farther the iterate is from the minimizer, the harder the gradient pulls it back, and with steps $\eta_t = 2/(\lambda(t + 1))$ and an average that weights later iterates more, the error falls to $2G^2/(\lambda(T + 1))$. The price is the $1/\lambda$: a weak regularizer gives a fast rate only after many steps.

On a non-convex loss, such as a neural network's, no gradient method can promise the global minimum. What smoothness guarantees is a point where the gradient is small: for a $\beta$-smooth loss that can fall at most $\Delta$ from the start, with gradient variance $\sigma^2$ and $\eta \le 1/\beta$,
$$\frac1T\sum_{t=1}^T\mathbb E\|\nabla F(w_t)\|^2 \le \frac{2\Delta}{\eta T} + \beta\eta\sigma^2.$$
It is the same trade-off, progress against noise, and a small gradient is all you get: the point may be a saddle or a poor local minimum.

## Gradient noise and minibatches {#sec-opt-noise}

Every bound above pays for the mean squared size of the random gradients, which is the squared true gradient plus the variance of the estimate. Averaging a batch of $B$ independent gradients divides the variance by $B$, from $\sigma_1^2$ to $\sigma_1^2/B$, a little less when drawn without replacement. Noise of mean zero added to each gradient adds its mean squared size $\sigma^2$: independent of the gradient and zero on average, it neither reinforces nor cancels it. So the convex bound holds with $G^2 = L^2 + \sigma^2$:
$$\eta^* = \frac{R}{\sqrt{(L^2 + \sigma^2)T}}, \qquad \frac{R\sqrt{L^2 + \sigma^2}}{\sqrt T}.$$
To reach a fixed error, $T$ must grow in proportion to $L^2 + \sigma^2$, so noise multiplies the steps needed by $1 + \sigma^2/L^2$. Noise of size $\sigma = 3L$ makes that ten times as many.

| Noise $\sigma$ | $\eta^*$ from the bound | Observed at $\eta^*$ | Best $\eta$ on the grid | Observed there |
| --- | --- | --- | --- | --- |
| 0 | 0.0500 | $4.99\times10^{-4}$ | 0.178 | $1.85\times10^{-4}$ |
| 1 | 0.0354 | $1.58\times10^{-3}$ | 0.056 | $1.38\times10^{-3}$ |
| 3 | 0.0158 | $9.58\times10^{-3}$ | 0.056 | $5.10\times10^{-3}$ |

: Projected SGD with added gradient noise, $T = 10{,}000$: error of the averaged iterate at the bound's step and at the best step on the grid, 10 runs per setting. {#tbl-opt-noise}

@tbl-opt-noise shows both effects on the logistic problem. Noise pushed the best step down from 0.178 to 0.056, so retune the step size whenever the gradient noise changes. And the bound's $\eta^*$ is conservative, by a factor of 3.6 without noise, for the same reason the bound is loose: the real gradients and starting distance are smaller than $L$ and $R$. Larger batches are not free either: a batch of $B$ costs $B$ gradient evaluations and divides the variance by $B$, so per evaluation the gain cancels, and batches pay mainly through parallelism.

## Sensitivity of regularized ERM {#sec-opt-erm}

The last question is about the problem, not the optimizer. Regularized **empirical risk minimization (ERM)** returns the minimizer of $J(w; D) = \frac1n\sum_i\ell(w, z_i) + \frac\lambda2\|w\|^2$. Replace one example: how far can the minimizer move?

The argument has two steps. Each example carries weight $1/n$, and its gradient is at most $L$ long, so swapping it changes the gradient of $J$ at any point by at most $2L/n$, the old one's $L/n$ out and the new one's $L/n$ in. At the old minimizer the new objective's gradient is therefore at most $2L/n$. Strong convexity says the gradient grows by at least $\lambda$ per unit of distance from the minimizer, so a point where it is that small lies within $(2L/n)/\lambda$ of the new minimizer:
$$\|w^*(D) - w^*(D')\| \le \frac{2L}{\lambda n}.$$
The loss at any point then changes by at most $L$ times that, $2L^2/(\lambda n)$. Later parts call this movement a **sensitivity**.

With $n = 1{,}000$ and $\lambda = 0.1$ the bound is $2/(0.1 \times 1{,}000) = 0.02$, and over 20 replacements in regularized logistic regression the largest observed move is $6.66\times10^{-3}$: smaller, because real data add curvature of their own and two gradients rarely point in opposite directions at full length. Because one example weighs $1/n$, ten times the data gives a tenth of the movement (@fig-opt-erm-sensitivity).

![Left: largest observed movement of the regularized logistic minimizer when one of $n$ examples is replaced (solid), against $2L/(\lambda n)$ (dashed), for three values of $\lambda$. Right: the same at $n = 1000$ as a function of $\lambda$: the bound blows up as $\lambda \to 0$, while the observed movement levels off.](figures/optimization_erm_sensitivity.svg){#fig-opt-erm-sensitivity}

The argument shows what the bound needs. It is about the exact minimizer, so an optimizer stopped early adds its own error. And it needs the curvature: without a regularizer, the absolute loss gives the median, which one replaced value can move by 1 whatever $n$ is. On a non-convex loss with two valleys, one label can flip which valley is deeper, and the minimizer of a double-well loss jumps by 2.0854 at $n = 11$ and still by 2.0010 at $n = 1{,}001$. For non-convex models, bound one example's influence on the *algorithm* instead, such as a fixed number of SGD steps ([Algorithmic stability and generalization](07-stability-generalization.md)).

## Code

The formulas are one line each, so there is no snippet. The full edition's script, [optimization.py](../../code/part01-prerequisites/optimization.py), runs every experiment above:

```bash
uv run python code/part01-prerequisites/optimization.py
```

## Cheat sheet

| Result | Assumptions | Bound |
| --- | --- | --- |
| Logistic loss | $\lVert x\rVert \le 1$ | 1-Lipschitz, $\frac14$-smooth, convex, not strongly convex |
| Ridge term $\frac\lambda2\lVert w\rVert^2$ | Added to a convex loss | $\lambda$-strongly convex; Lipschitz only on a ball |
| SGD, convex | $\mathbb E\lVert g\rVert^2 \le G^2$, start within $R$, $\eta = R/(G\sqrt T)$ | $RG/\sqrt T$, averaged iterate |
| SGD, strongly convex | $\eta_t = 2/(\lambda(t+1))$, $t$-weighted average | $2G^2/(\lambda(T + 1))$ |
| SGD, non-convex | $\beta$-smooth, variance $\sigma^2$, $\eta \le 1/\beta$ | $\frac1T\sum_t\mathbb E\lVert\nabla F(w_t)\rVert^2 \le \frac{2\Delta}{\eta T} + \beta\eta\sigma^2$ |
| Minibatch variance | $B$ of $n$ | $\sigma_1^2/B$; times $\frac{n - B}{n - 1}$ without replacement |
| Added gradient noise | Mean zero, $\mathbb E\lVert\xi\rVert^2 = \sigma^2$ | $G^2 = L^2 + \sigma^2$; $\eta^* = R/\sqrt{(L^2 + \sigma^2)T}$ |
| Regularized ERM | Convex $L$-Lipschitz loss | Minimizer moves $\le 2L/(\lambda n)$; loss $\le 2L^2/(\lambda n)$ |

: The results of this chapter. {#tbl-opt-cheat}

**Rules of thumb**

- Without curvature SGD's error falls like $1/\sqrt T$, with it like $1/(\lambda T)$. Average the iterates either way.
- Noise multiplies the steps needed by $1 + \sigma^2/L^2$; lower the step size when gradients get noisier.
- One example moves a $\lambda$-regularized minimizer by at most $2L/(\lambda n)$. Without strong convexity there is no such bound.

## Further reading

- Shalev-Shwartz & Ben-David, *Understanding Machine Learning* (2014), ch. 12–14: convexity, regularization and stability, and SGD, at the level of this chapter.
- Bottou, Curtis & Nocedal, "Optimization Methods for Large-Scale Machine Learning" (SIAM Review 2018): noise, step sizes and batch sizes in practice.
- Bubeck, "Convex Optimization: Algorithms and Complexity" (2015): the rates for each combination of assumptions, with lower bounds.

**Derivations.** Every result in this chapter is derived step by step in [the full edition's chapter](../../book-full/part01-prerequisites/06-optimization.md).
