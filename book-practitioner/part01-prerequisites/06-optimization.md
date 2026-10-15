# Optimization

## Why this matters

Training a model minimizes the average loss over the training set, $F(w) = \frac1n\sum_i\ell(w, z_i)$, over parameters $w$. Two questions come up every time. **How fast does stochastic gradient descent (SGD) get there, and how should you set its step size and batch size when its gradients are noisy?** And **how much does the answer depend on any one training example?** If you replace one example and retrain, does the minimizer barely move, or can it jump?

The one idea: a few constants decide everything. They are how steep the loss can be (the Lipschitz constant $L$), how fast its slope can change (the smoothness $\beta$), how much it curves upward at least (the strong convexity $\lambda$), and how much SGD's random gradients scatter (their variance $\sigma^2$). Know which constants a result uses, and you know when it applies. @tbl-opt-overview maps the results to the sections.

| Question | Result | Section |
| --- | --- | --- |
| Does my loss qualify? | Lipschitz, smooth, strongly convex; logistic loss is 1-Lipschitz and $\frac14$-smooth | @sec-opt-definitions |
| How fast does SGD converge? | $RL/\sqrt T$ convex; $O(G^2/(\lambda T))$ strongly convex; stationary points otherwise | @sec-opt-sgd |
| What do noise and batch size cost? | Variance $\sigma_1^2/B$; step size shrinks as noise grows | @sec-opt-noise |
| How far can one example move the minimizer? | $2L/(\lambda n)$ for regularized empirical risk minimization | @sec-opt-erm |

: The questions of this chapter, and where each is answered. {#tbl-opt-overview}

## Lipschitz, smooth and strongly convex losses {#sec-opt-definitions}

**Idea.** Two properties bound a derivative from above: Lipschitz continuity bounds the slope, smoothness bounds the curvature. Two bound the curvature from below: convexity says it is never negative, strong convexity says it is at least $\lambda > 0$.

**Result.** A loss is **$L$-Lipschitz** when its gradient norm never exceeds $L$, so moving the parameters by $d$ changes it by at most $Ld$. It is **$\beta$-smooth** when its gradient changes by at most $\beta$ per unit of distance, which makes $1/\beta$ the largest safe step. It is **$\lambda$-strongly convex** when its curvature is at least $\lambda$ in every direction; adding $\frac\lambda2\|w\|^2$ to a convex loss makes it so. To find the constants, bound the gradient norm, and the eigenvalues of the matrix of second derivatives.

**Example.** The logistic loss $\ln(1 + e^{-y\langle w, x\rangle})$ with features scaled to $\|x\| \le 1$ is convex, 1-Lipschitz and $\frac14$-smooth, but not strongly convex. For $x = (0.6, 0.8)$ at $w = 0$, the loss is 0.6931, the gradient norm 0.5000 and the curvature 0.2500, the maximum. Adding a ridge term $\frac\lambda2\|w\|^2$ makes it $\lambda$-strongly convex and $(\frac14 + \lambda)$-smooth.

**Pitfalls.** Lipschitz is not smooth: $|w|$ is 1-Lipschitz but its slope jumps at 0, and $w^2/2$ is 1-smooth but its slope is unbounded. A strongly convex function is never Lipschitz on all of $\mathbb R^d$, because its slope grows with the distance from the minimizer. So the ridge-regularized logistic loss is Lipschitz only on a ball $\|w\| \le R$, where its gradient is at most $1 + \lambda R$. Algorithms keep the iterates in the ball by **projection**, $\Pi(u) = u\min(1, R/\|u\|)$, which never moves a point farther from any point of the ball.

## SGD convergence {#sec-opt-sgd}

**Idea.** SGD steps against the gradient of one random example, or the average over a small random batch, which is right on average but noisy. Its error after $T$ steps is a trade-off: small steps take long to travel the distance $R$ to the minimizer, large steps overshoot by an amount set by the size $G$ of the random gradients. Curvature pulls the iterates back toward the minimizer and makes the trade-off cheaper.

**Result.** For a convex loss, with $\mathbb E\|g_t\|^2 \le G^2$, a start within $R$ of the minimizer and a constant step $\eta$, projected SGD's averaged iterate $\bar w_T$ has error
$$\mathbb E[F(\bar w_T)] - F^* \le \frac{R^2}{2\eta T} + \frac{\eta G^2}{2},$$
which is $RG/\sqrt T$ at the best step $\eta = R/(G\sqrt T)$; for an $L$-Lipschitz loss, $G = L$. For a $\lambda$-strongly convex loss, steps $\eta_t = 2/(\lambda(t + 1))$ and an average that weights iterate $t$ by $t$ give $2G^2/(\lambda(T + 1))$; the plain average with steps $1/(\lambda t)$ loses a factor $\ln T$. For a $\beta$-smooth non-convex loss that can fall at most $\Delta$ from the start, with gradient variance $\sigma^2$ and $\eta \le 1/\beta$,
$$\frac1T\sum_{t=1}^T\mathbb E\|\nabla F(w_t)\|^2 \le \frac{2\Delta}{\eta T} + \beta\eta\sigma^2.$$

**Example.** Projected SGD on logistic regression with $n = 1{,}000$, $d = 5$, $R = 5$ and $L = 1$, averaged over 20 runs, gives @tbl-opt-sgd-convex.

| Steps $T$ | Step $\eta$ | Observed $F(\bar w_T) - F^*$ | Bound $RL/\sqrt T$ |
| --- | --- | --- | --- |
| 100 | 0.5000 | $4.44\times10^{-2}$ | 0.500 |
| 1,000 | 0.1581 | $6.46\times10^{-3}$ | 0.158 |
| 10,000 | 0.0500 | $7.71\times10^{-4}$ | 0.050 |

: Projected SGD on logistic regression against the convex Lipschitz bound, averaged over 20 runs. {#tbl-opt-sgd-convex}

The bound holds, loose by a factor of 11 at $T = 100$ and 65 at $T = 10{,}000$. The observed error falls almost like $1/T$, because the data curve the loss near the minimizer, which the convex bound does not know.

**Pitfalls.** The guarantees are for the averaged iterate, so average. Worst-case constants make the bounds loose by factors of 10 to 130: trust how they scale with $T$, $n$, $\lambda$ and $\sigma$, not their size. On non-convex losses a small gradient is all you get; the point can be a saddle or a poor local minimum.

## Gradient noise and minibatches {#sec-opt-noise}

**Idea.** Averaging $B$ random gradients divides their variance by $B$. Independent noise with mean zero added to each gradient adds its mean squared size to $G^2$ and changes nothing else.

**Result.** If a single-example gradient has variance $\sigma_1^2$, a batch of $B$ has variance $\sigma_1^2/B$ when drawn with replacement, and $\frac{\sigma_1^2}{B}\cdot\frac{n - B}{n - 1}$ without. With added noise of mean squared size $\sigma^2$, the convex bound holds with $G^2 = L^2 + \sigma^2$, so the best step and the bound are
$$\eta^* = \frac{R}{\sqrt{(L^2 + \sigma^2)T}}, \qquad \frac{R\sqrt{L^2 + \sigma^2}}{\sqrt T}.$$
Reaching a target error therefore takes $1 + \sigma^2/L^2$ times as many steps as without noise.

**Example.** Noise of size $\sigma = 3L$ makes $L^2 + \sigma^2 = 10L^2$: ten times the steps. On the logistic problem with $T = 10{,}000$, @tbl-opt-noise compares the bound's step with the best step on a grid.

| Noise $\sigma$ | $\eta^*$ from the bound | Observed at $\eta^*$ | Best $\eta$ on the grid | Observed there |
| --- | --- | --- | --- | --- |
| 0 | 0.0500 | $4.99\times10^{-4}$ | 0.178 | $1.85\times10^{-4}$ |
| 1 | 0.0354 | $1.58\times10^{-3}$ | 0.056 | $1.38\times10^{-3}$ |
| 3 | 0.0158 | $9.58\times10^{-3}$ | 0.056 | $5.10\times10^{-3}$ |

: Projected SGD with added gradient noise: error of the averaged iterate at the bound's step and at the best step on the grid, 10 runs per setting; the row without noise differs from @tbl-opt-sgd-convex only by the number of runs averaged. {#tbl-opt-noise}

**Pitfalls.** Retune the step size whenever the gradient noise changes: noise pushed the best step down from 0.178 to 0.056 here. The bound's $\eta^*$ is conservative (by a factor of 3.6 without noise), because the real gradients and starting distance are smaller than $L$ and $R$. Larger batches cost proportionally more gradient evaluations, so they pay mainly through parallelism.

## Sensitivity of regularized ERM {#sec-opt-erm}

**Idea.** Regularized **empirical risk minimization (ERM)** returns the minimizer of $J(w; D) = \frac1n\sum_i\ell(w, z_i) + \frac\lambda2\|w\|^2$. Replacing one example moves the gradient by at most $2L/n$, and curvature $\lambda$ turns that into a small move of the minimizer.

**Result.** If each loss is convex and $L$-Lipschitz, and $D$ and $D'$ differ in one example,
$$\|w^*(D) - w^*(D')\| \le \frac{2L}{\lambda n},$$
and the loss at any point changes by at most $2L^2/(\lambda n)$. The bound is tight for a loss with no curvature of its own. Later parts call this movement a **sensitivity**.

**Example.** Regularized logistic regression with $n = 1{,}000$ and $\lambda = 0.1$ has bound $2/(0.1 \times 1{,}000) = 0.02$. Over 20 replacements the largest observed move is $6.66\times10^{-3}$. Ten times the data gives a tenth of the movement (@fig-opt-erm-sensitivity).

![Left: largest observed movement of the regularized logistic minimizer when one of $n$ examples is replaced (solid), against $2L/(\lambda n)$ (dashed), for three values of $\lambda$. Right: the same at $n = 1000$ as a function of $\lambda$: the bound blows up as $\lambda \to 0$, while the observed movement levels off.](figures/optimization_erm_sensitivity.svg){#fig-opt-erm-sensitivity}

**Pitfalls.** The bound is about the exact minimizer; an optimizer stopped early adds its own error. Strong convexity is essential. Without the regularizer, the absolute loss gives the median, which one replaced value can move by 1 for every $n$. On a non-convex loss with two valleys, one label can flip which is deeper: the minimizer of a double-well loss jumps by 2.0854 at $n = 11$ and still by 2.0010 at $n = 1{,}001$. For non-convex models, bound one example's influence on the *algorithm*, such as a fixed number of SGD steps ([Algorithmic stability and generalization](07-stability-generalization.md)).

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

- Check the constants first: Lipschitz bounds the gradient, smooth bounds its change, strong convexity bounds the curvature from below.
- Without curvature SGD's error falls like $1/\sqrt T$, with it like $1/(\lambda T)$. Average the iterates either way.
- Noise multiplies the steps needed by $1 + \sigma^2/L^2$; lower the step size when gradients get noisier.
- One example moves a $\lambda$-regularized minimizer by at most $2L/(\lambda n)$. Without strong convexity there is no such bound.

## Further reading

- Shalev-Shwartz & Ben-David, *Understanding Machine Learning* (2014), ch. 12–14: convexity, regularization and stability, and SGD, at the level of this chapter.
- Bottou, Curtis & Nocedal, "Optimization Methods for Large-Scale Machine Learning" (SIAM Review 2018): noise, step sizes and batch sizes in practice.
- Bubeck, "Convex Optimization: Algorithms and Complexity" (2015): the rates for each combination of assumptions, with lower bounds.

**Derivations.** Every result in this chapter is derived step by step in [the full edition's chapter](../../book-full/part01-prerequisites/06-optimization.md).
