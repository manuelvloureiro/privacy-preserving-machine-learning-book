# Optimization

## What it is {#sec-opt-what}

Training minimizes the average loss on the $n$ records of a training set $D$, usually with stochastic gradient descent (SGD). Each step moves the parameters $\theta$ against the gradient of a random minibatch, times a learning rate $\eta$. A loss is **$L$-Lipschitz** when its gradient never has a length of more than $L$. It is **smooth** when its gradient changes by at most a constant $\beta$ per unit of distance.

It is **strongly convex** when its curvature is at least a constant $\lambda$ in every direction, so that it has a single minimizer. @tbl-opt-losses gives them for common losses.

| Loss | Lipschitz | Smooth | Strongly convex |
| --- | --- | --- | --- |
| Logistic, $\ln(1 + e^{-y\langle \theta, x\rangle})$ | $L = 1$ | $\beta = \frac14$ | No |
| Hinge, $\max(0, 1 - y\langle \theta, x\rangle)$ | $L = 1$ | No, it has a kink | No |
| Squared, $\frac12(\langle \theta, x\rangle - y)^2$ | No, the gradient grows with $\theta$ | $\beta = 1$ | Only if the features span every direction |
| Any of these plus $\frac\lambda2\|\theta\|^2$ | Only on a bounded set: $L + \lambda r$ within radius $r$ | $\beta + \lambda$ | $\lambda$ |

: The constants of common losses, for labels $\pm1$ and features of length at most 1. {#tbl-opt-losses}

## When you meet it {#sec-opt-when}

You meet these results when you choose a learning rate, a batch size and a number of steps, or the regularization of a scikit-learn model. They also bound the effect of one record, which [the next chapter](07-stability-generalization.md) turns into a bound on the generalization gap.

## How to use it {#sec-opt-use}

For a convex loss, run SGD for a number $T$ of steps inside a ball of radius $R$ that contains the minimizer. Return the **averaged iterate** $\bar\theta_T$, the average of the parameters across all steps. Let $G^2$ bound the average squared length of the minibatch gradients, which is $L^2$ for an $L$-Lipschitz loss. Then the best fixed step, and the error it guarantees on the average training loss $\mathcal L(\theta; D)$ above its smallest value $\mathcal L^*$, are
$$\eta = \frac{R}{G\sqrt T}, \qquad \mathbb E[\mathcal L(\bar\theta_T; D)] - \mathcal L^* \le \frac{RG}{\sqrt T}.$$ {#eq-opt-sgd-convex}
Logistic regression on normalized features has $L = 1$, so $G = 1$. In a ball of radius 5, a run of $T = 10{,}000$ steps then gets the learning rate $5/\sqrt{10{,}000} = 0.05$. This rate is for the worst case, so use it only to start a grid search. On the full edition's data, the best rate was 3.6 times larger.

A ridge term $\frac\lambda2\|\theta\|^2$ makes the loss strongly convex, so the error falls about like $1/T$ instead of $1/\sqrt T$. For that, the learning rate at each step $t$ must shrink like $1/(\lambda t)$, as in `SGDClassifier(alpha=λ, learning_rate="optimal", average=True)`. Scikit-learn's `LogisticRegression` minimizes `C` times the summed loss plus $\frac12\|\theta\|^2$. On the average loss, that is a ridge of strength $\lambda$ equal to one over $n$ times `C`, so a large `C` means a weak ridge.

If the gradient of one record has a variance, written $\sigma_1^2$, around the full gradient, a minibatch of $B$ records drawn without replacement has the variance
$$\mathbb E\|g_B - \nabla \mathcal L(\theta; D)\|^2 = \frac{\sigma_1^2}{B}\cdot\frac{n - B}{n - 1}.$$ {#eq-opt-minibatch}
The constant $G^2$ of @eq-opt-sgd-convex is the squared length of the full gradient plus this variance, so a larger batch lowers $G$ and allows a larger learning rate.

**Gradient-norm clipping** limits the length of each per-example gradient $g_i$ to a clipping norm $C$, because it replaces $g_i$ by $g_i\min(1, C/\|g_i\|)$. Suppose a training step subtracts $\eta/B$ times the sum of the clipped gradients. Then, for every loss, one added or removed record changes the step by at most the first amount below, and a replaced record by at most the second:
$$\frac{\eta C}{B}, \qquad \frac{2\eta C}{B}.$$ {#eq-opt-step-influence}
Fix $B$ in advance, because if you divide by the actual batch size, a removal changes the scale of every other gradient. The bound is for one step, so across a run the changes can add up.

Regularized **empirical risk minimization (ERM)** minimizes the average loss plus a ridge term of strength $\lambda$. Write $\theta^*(D)$ for its minimizer, and $D'$ for $D$ with one record replaced. If each loss is convex and $L$-Lipschitz,
$$\|\theta^*(D) - \theta^*(D')\| \le \frac{2L}{\lambda n}.$$ {#eq-opt-erm-sensitivity}
For logistic regression on 1,000 records, $\lambda = 0.1$ gives a bound of 0.02, three times the largest movement on the full edition's data.

## What goes wrong {#sec-opt-wrong}

One feature vector longer than 1 breaks $L = 1$, so normalize all of them to length 1 or less. The decreasing learning rate needs a $\lambda$ no larger than the true curvature, or the steps shrink too fast and training stops early. A learning rate above $2/\beta$ can increase the loss, the usual cause of a loss curve that diverges.

On the non-convex loss of a neural network, SGD guarantees only a point with a small gradient, such as a saddle point. The bound of @eq-opt-erm-sensitivity also needs the strong convexity of the ridge term. Without it, one replaced record can move the median, the minimizer of the absolute loss, from 0 to 1. So for a neural network, bound the effect of one record on the steps, as in @eq-opt-step-influence, not on the minimizer.

**Derivations.** Every result here is derived step by step in [the full edition's chapter](../../book-full/part01-prerequisites/06-optimization.md).
