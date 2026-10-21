# Optimization

## Why this matters

Almost all training is done by gradient descent (GD), or by its cheaper and noisier variant, stochastic gradient descent (SGD), and two practical questions come up every time. The first is how to set the step size and the number of steps, and how much noise in the gradients slows training down. The second is how much the trained model depends on any one training example, which is the quantity that later parts of the book have to control.

Both are answered by a few constants of the loss and by the variance of the random gradients. So the first job is to know these constants for your loss, or to enforce them.

## The constants of a loss, and how to enforce them {#sec-opt-definitions}

Throughout, the model is a parameter vector $w$, and the training set has $n$ examples. Write $\ell(w, z)$ for the loss on an example $z$, and $z_1, \dots, z_n$ for the training examples. Training minimizes the **empirical risk**, the average loss
$$F(w) = \frac1n\sum_{i=1}^n\ell(w, z_i),$$
and this chapter writes $w^*$ for its minimizer and $F^*$ for its smallest value.

A loss is **$L$-Lipschitz** when its gradient never has length more than $L$, so it caps how far one example can push a model. It is **smooth** when its gradient changes by at most a constant $\beta$ per unit of distance. That bounds how far you can trust a step, and it sets the safe step size $1/\beta$.

A loss is **$\lambda$-strongly convex** when it curves upward by at least $\lambda$ in every direction, so that a point with a small gradient must lie close to the minimizer. The ratio $\kappa = \beta/\lambda$ is the **condition number**, the ratio of the steepest curvature to the flattest. @tbl-opt-losses gives the constants for the losses this book uses, for features of length at most 1.

| Loss | Lipschitz | Smooth | Strongly convex |
| --- | --- | --- | --- |
| Logistic, $\ln(1 + e^{-y\langle w, x\rangle})$ | $L = 1$ | $\beta = \frac14$ | No |
| Hinge, $\max(0, 1 - y\langle w, x\rangle)$ | $L = 1$ | No, it has a kink | No |
| Squared, $\frac12(\langle w, x\rangle - y)^2$ | No, the gradient grows with $w$ | $\beta = 1$ | Only if the feature vectors span every direction |
| Any of these plus $\frac\lambda2\|w\|^2$ | Only on a bounded set: $L + \lambda r$ within radius $r$ | $\beta + \lambda$ | $\lambda$ |

: The constants of common losses, for feature vectors of length at most 1. {#tbl-opt-losses}

The table shows that the constants come from three choices you control. Normalizing every feature vector to length at most 1 is what makes the logistic and hinge losses 1-Lipschitz, so a single unnormalized outlier breaks the guarantee. A ridge term $\frac\lambda2\|w\|^2$ is the usual source of strong convexity, because the logistic loss has none of its own. But the ridge term makes the gradient grow with the parameters. So you also keep $w$ in a ball by **projection**, which pulls it back to the nearest point of the ball after any step that leaves it.

For a neural network, whose constants nobody can compute, you instead **clip** each example's gradient to a chosen length $C$. Clipping enforces the Lipschitz bound by construction, with $L = C$, but only if it happens per example. In PyTorch, `torch.nn.utils.clip_grad_norm_` clips the gradient of a whole batch, which bounds nothing about any single example. Per-example clipping, which Part 5 uses, needs per-example gradients.

## Gradient descent and SGD {#sec-opt-gd}

**Gradient descent** repeatedly steps against the full gradient, with a step size written $\eta$. The safe step is $1/\beta$, because smoothness guarantees that such a step never overshoots. On a strongly convex risk the error then shrinks by a factor $1 - 1/\kappa$ per step. So a badly conditioned problem is slow, which is why rescaling features to similar ranges speeds training up.

SGD replaces the full gradient with that of one example, or the average over a small random **minibatch**. The random gradient equals the true gradient on average, but only if the examples are sampled uniformly, and a sampler that favours some examples minimizes a reweighted risk instead. Its scatter around the true gradient, the variance, is what every SGD rate pays for. With a constant step it keeps the iterates jittering around the minimizer, and averaging them or shrinking the step removes the jitter.

## Convex losses: the step size and the averaged iterate {#sec-opt-sgd}

The basic guarantee covers convex losses with bounded gradients, including the hinge loss. It runs **projected SGD** inside a closed convex set that contains the minimizer. It returns the **averaged iterate** $\bar w_T$, the average of the parameters over a run of $T$ steps. Write $G^2$ for a bound on the average squared length of the random gradients, and $R$ for a bound on the distance from the start to the minimizer. With the step size
$$\eta = \frac{R}{G\sqrt T}, \qquad \mathbb E[F(\bar w_T)] - F^* \le \frac{RG}{\sqrt T}.$$ {#eq-opt-sgd-convex}
For an $L$-Lipschitz loss, $G = L$.

To use it, take $R$ as the radius of the projection ball and $L$ from @tbl-opt-losses, choose $T$, and start from the step the formula gives. For logistic regression on the ball of radius 5, a run of 10,000 steps gives a step of $0.05$. No method does better than this $1/\sqrt T$ rate on this class.

Treat the bound as a guarantee and the step as a starting point. The formula assumes the worst gradients and the farthest start, so on the full edition's logistic problem the best step on a grid is $3.6$ times larger. So tune the step on a grid around the formula's value, and return the averaged iterate, which the guarantee covers, rather than the last one.

## Strongly convex losses: a decaying step {#sec-opt-strong}

With a ridge term the rate improves from $1/\sqrt T$ to $1/T$, so halving the error takes twice the steps rather than four times. Use the steps $2/(\lambda(t + 1))$ at step $t$, and average the iterates with weights proportional to $t$. Then
$$\mathbb E[F(\bar w_T)] - F^* \le \frac{2G^2}{\lambda(T + 1)}.$$ {#eq-opt-sgd-strong}
The simpler schedule $1/(\lambda t)$ with a uniform average costs an extra factor that grows like $\ln T$. Here $G$ bounds the gradients on the projection ball, since a ridge term makes them grow with the weights.

In practice, scikit-learn's `SGDClassifier` with `learning_rate="optimal"` uses a step of this $1/(\lambda t)$ shape, with its `alpha` equal to $\lambda$, and `average=True` returns an averaged iterate. Its `LogisticRegression` is parameterized differently. It multiplies the summed loss by `C` instead of penalizing the weights, so its ridge strength in this chapter's terms is $\lambda = 1/(Cn)$. A large `C` therefore means a weak ridge term.

The schedule needs a value of $\lambda$ no larger than the true curvature. If you overestimate it, the steps shrink too fast and the iterates can stall far from the minimizer.

## Gradient noise and minibatches {#sec-opt-noise}

Gradients get noisier from small batches, low-precision arithmetic or compression, and later parts add noise to them on purpose. Write $\sigma_1^2$ for the variance of a single-example gradient. A batch of $B$ examples divides it by $B$, and drawing the batch without replacement does slightly better:
$$\mathbb E\|g_B - \nabla F(w)\|^2 = \frac{\sigma_1^2}{B}\cdot\frac{n - B}{n - 1}.$$ {#eq-opt-minibatch}
Noise of a fixed size added to the summed gradient of a batch is also divided by $B$ when the sum is averaged. So larger batches dilute added noise, which is why noisy training prefers them.

Now add noise of mean zero whose average squared length is $\sigma^2$. It simply adds to $G^2$, so the convex recipe becomes
$$\eta^* = \frac{R}{\sqrt{(L^2 + \sigma^2)T}}, \qquad \mathbb E[F(\bar w_T)] - F^* \le \frac{R\sqrt{L^2 + \sigma^2}}{\sqrt T}.$$
To reach the same bound, the number of steps must grow by the factor $1 + \sigma^2/L^2$, so noise of size $\sigma = 3L$ multiplies the steps you need by 10. On the full edition's logistic problem with 10,000 steps, the formula's step falls from 0.0500 without noise to 0.0158 at $\sigma = 3$, and the error at that step grows about 20-fold. The best step on a grid falls too, from 0.178 to 0.056. So a step tuned without noise is too large once noise is added, and the formula's step is conservative throughout. Retune on a grid whenever the noise level changes.

## Non-convex losses {#sec-opt-nonconvex}

Neural networks have non-convex losses, and no gradient method can promise to find their global minimum. With a smooth loss and a step of at most $1/\beta$, SGD only guarantees a point where the gradient is small. Noise slows that down, just as it does for convex losses. A small gradient can sit at a saddle point or a poor local minimum, so nothing here says how far the risk is from its best value. A step above $2/\beta$ can even raise the risk, which is what a diverging loss curve usually means.

## How far one example moves the model {#sec-opt-erm}

Regularized **empirical risk minimization (ERM)** minimizes the average loss plus a ridge term,
$$J(w; D) = \frac1n\sum_{i=1}^n\ell(w, z_i) + \frac\lambda2\|w\|^2,$$
where $D$ is the training set and $w^*(D)$ its minimizer. If each loss is convex and $L$-Lipschitz, replacing one example moves the minimizer by at most
$$\|w^*(D) - w^*(D')\| \le \frac{2L}{\lambda n}.$$ {#eq-opt-erm-sensitivity}
One example tilts the gradient by at most $2L/n$, and strong convexity turns a small tilt into a small move. The loss on any example then changes by at most $2L^2/(\lambda n)$, a property called **uniform stability**, which [the next chapter](07-stability-generalization.md) turns into a generalization bound.

Later parts calibrate noise to this per-example influence, so read the bound as a design rule. Ten times more data gives a tenth of the movement, and halving $\lambda$ can double it. For the logistic loss with $\lambda = 0.1$ and 1,000 examples, the bound is $0.02$, while the largest movement observed on the full edition's data is $6.66\times10^{-3}$.

The bound needs strong convexity, and without it no bound of this kind exists. The absolute loss is convex and 1-Lipschitz, but its minimizer is the median, and replacing one example can move the median from 0 to 1 however large $n$ is. For a non-convex loss, one example can decide which of two valleys is deeper. So for neural networks, any statement about the influence of one example must be about the training algorithm, not about the minimizer.

## Code

This calculator gives the step size of convex SGD, with or without gradient noise, and the movement bound for regularized ERM.

```python
import math


def sgd_step(radius: float, lipschitz: float, steps: int, noise: float = 0.0) -> float:
    """Step size that balances the convex SGD bound."""
    return radius / math.sqrt((lipschitz**2 + noise**2) * steps)


def erm_movement(lipschitz: float, lam: float, n: int) -> float:
    """Largest move of the minimizer when one example is replaced."""
    return 2 * lipschitz / (lam * n)


print(f"{sgd_step(5, 1, 10_000):.4f}")  # 0.0500
print(f"{sgd_step(5, 1, 10_000, noise=1):.4f}")  # 0.0354
print(f"{sgd_step(5, 1, 10_000, noise=3):.4f}")  # 0.0158
print(f"{erm_movement(1, 0.1, 1000):.3f}")  # 0.020
```

## Cheat sheet

| Result | Conditions | Bound or rule |
| --- | --- | --- |
| SGD, convex | Gradients bounded by $G$, start within $R$ | Step $R/(G\sqrt T)$; error $\le RG/\sqrt T$, averaged iterate |
| SGD, strongly convex | Steps $2/(\lambda(t+1))$, $t$-weighted average | $2G^2/(\lambda(T+1))$ |
| Minibatch of $B$ out of $n$ | Uniform, without replacement | Variance $\frac{\sigma_1^2}{B}\cdot\frac{n-B}{n-1}$ |
| Added gradient noise $\sigma$ | Mean zero | $G^2 = L^2 + \sigma^2$; steps needed grow by $1 + \sigma^2/L^2$ |
| Regularized ERM | Convex, $L$-Lipschitz loss | Minimizer moves $\le 2L/(\lambda n)$ |

: The results of this chapter. {#tbl-opt-cheat}

**Rules of thumb**

- Normalize features to length at most 1, add a ridge term, and project onto a ball. That gives you $L$, $\lambda$ and $R$, and without the normalization $L$ is not 1.
- For a model whose constants you cannot compute, clip each example's gradient. Clipping the batch's gradient bounds nothing per example.
- Never keep a step size tuned without noise once noise is added. Retune it on a grid, and return the averaged iterate.
- Read scikit-learn's `C` as $\lambda = 1/(Cn)$, not as $\lambda$.
- One example moves a regularized minimizer by at most $2L/(\lambda n)$. Without strong convexity, as in neural networks, there is no such bound.

## Further reading

- Bottou, Curtis & Nocedal, "Optimization Methods for Large-Scale Machine Learning" (SIAM Review 2018): noise, step sizes and batch sizes in practice.
- Bubeck, "Convex Optimization: Algorithms and Complexity" (2015): the rates and their lower bounds.

**Derivations.** Every result here is derived step by step in [the full edition's chapter](../../book-full/part01-prerequisites/06-optimization.md).
