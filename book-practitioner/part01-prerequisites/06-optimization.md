# Optimization

## What it is {#sec-opt-what}

Training a model minimizes the average loss over the $n$ records of its training set $D$, usually by stochastic gradient descent (SGD). Each step moves the parameters $\theta$ against the gradient of a random minibatch, scaled by a learning rate $\eta$, also called the step size. A loss is **$L$-Lipschitz** when its gradient never has length more than $L$, and **smooth** when its gradient changes by at most a constant $\beta$ per unit of distance.

A loss is also **strongly convex** when its curvature is at least a constant $\lambda$ in every direction, which pins down a single minimizer. @tbl-opt-losses gives all three for common losses.

| Loss | Lipschitz | Smooth | Strongly convex |
| --- | --- | --- | --- |
| Logistic, $\ln(1 + e^{-y\langle \theta, x\rangle})$ | $L = 1$ | $\beta = \frac14$ | No |
| Hinge, $\max(0, 1 - y\langle \theta, x\rangle)$ | $L = 1$ | No, it has a kink | No |
| Squared, $\frac12(\langle \theta, x\rangle - y)^2$ | No, the gradient grows with $\theta$ | $\beta = 1$ | Only if the features span every direction |
| Any of these plus $\frac\lambda2\|\theta\|^2$ | Only on a bounded set: $L + \lambda r$ within radius $r$ | $\beta + \lambda$ | $\lambda$ |

: The constants of common losses, for labels $\pm1$ and features of length at most 1. {#tbl-opt-losses}

## When you meet it {#sec-opt-when}

You meet these results when you choose a learning rate, a batch size and a number of steps, and when you read scikit-learn's regularization arguments. They also bound one training record's influence, which [the next chapter](07-stability-generalization.md) turns into a bound on the generalization gap.

## How to use it {#sec-opt-use}

For a convex loss, run SGD for a number $T$ of steps inside a ball of radius $R$ that contains the minimizer. Return the **averaged iterate** $\bar\theta_T$, the average of the parameters over all steps. The minibatch gradients have an average squared length of at most $G^2$, which is $L^2$ for an $L$-Lipschitz loss. The step that minimizes the guarantee, and the error it guarantees on the average training loss $\mathcal L(\theta; D)$ with smallest value $\mathcal L^*$, are
$$\eta = \frac{R}{G\sqrt T}, \qquad \mathbb E[\mathcal L(\bar\theta_T; D)] - \mathcal L^* \le \frac{RG}{\sqrt T}.$$ {#eq-opt-sgd-convex}
Logistic regression on normalized features has $G = 1$. In a ball of radius 5, $T = 10{,}000$ steps then get a learning rate of $5/100 = 0.05$. The formula assumes the worst case, so on the full edition's logistic data the best learning rate on a grid is 3.6 times larger. Start there and tune on a grid around it.

A ridge term $\frac\lambda2\|\theta\|^2$ makes the error fall about like $1/T$ instead of $1/\sqrt T$. It needs a learning rate that shrinks like $1/(\lambda t)$ at step $t$, which `SGDClassifier(alpha=λ, learning_rate="optimal", average=True)` uses. `LogisticRegression` instead minimizes `C` times the summed loss plus $\frac12\|\theta\|^2$. Dividing by $n$ times `C` gives a ridge of strength $\lambda$ equal to the reciprocal of that product, so a large `C` means a weak ridge.

If the variance of one record's gradient around the full gradient is $\sigma_1^2$, a minibatch of $B$ records drawn without replacement has a gradient variance of
$$\mathbb E\|g_B - \nabla \mathcal L(\theta; D)\|^2 = \frac{\sigma_1^2}{B}\cdot\frac{n - B}{n - 1}.$$ {#eq-opt-minibatch}
The constant $G^2$ is the squared length of the full gradient plus this variance. So a larger batch lowers $G$, which allows a larger learning rate.

**Gradient-norm clipping** caps the length of each per-example gradient $g_i$ at a chosen clipping norm $C$, replacing it by $g_i\min(1, C/\|g_i\|)$. Suppose a training step subtracts $\eta/B$ times the sum of the clipped gradients, with $B$ fixed in advance. Then adding or removing one record changes the step by at most the first amount below, and replacing one by at most the second, whatever the loss:
$$\frac{\eta C}{B}, \qquad \frac{2\eta C}{B}.$$ {#eq-opt-step-influence}
Dividing by the actual batch size would let a removal rescale every other gradient too. It covers one step from the same starting point; over a run the changes can accumulate.

Regularized **empirical risk minimization (ERM)** minimizes the average loss plus a ridge term of strength $\lambda$. Write $\theta^*(D)$ for the minimizer on a training set $D$, and $D'$ for the same set with one record replaced. If each loss is convex and $L$-Lipschitz,
$$\|\theta^*(D) - \theta^*(D')\| \le \frac{2L}{\lambda n}.$$ {#eq-opt-erm-sensitivity}
For logistic regression on 1,000 records, $\lambda = 0.1$ gives a bound of 0.02. On the full edition's data the largest movement observed is $6.66\times10^{-3}$, so the bound is loose.

## What goes wrong {#sec-opt-wrong}

A single unnormalized feature vector breaks $L = 1$, so normalize every feature vector to length 1 or less. The decaying learning rate needs a value of $\lambda$ no larger than the true curvature, because an overestimate shrinks the steps too fast and training stalls. And a learning rate above $2/\beta$ can raise the loss, the usual cause of a diverging loss curve.

On a neural network's non-convex loss, SGD guarantees only a point with a small gradient, perhaps a saddle point. The movement bound needs the ridge term's strong convexity: without it, one replaced record can move the median, the absolute loss's minimizer, from 0 to 1. So for a neural network, a statement about one record's influence must be about the training steps, as in @eq-opt-step-influence, not about the minimizer.

**Derivations.** Every result here is derived step by step in [the full edition's chapter](../../book-full/part01-prerequisites/06-optimization.md).
