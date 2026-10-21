# Private empirical risk minimization

## Why this matters

Most of machine learning is **empirical risk minimization (ERM)**, which chooses the weights that minimize the average training loss, usually plus a penalty that keeps them short. The trained weights can give records away ([Membership inference](../part02-attacks/02-membership-inference.md)). Differential privacy (DP) bounds how much replacing one record can change what you release, through a privacy level $\varepsilon$ and a small slack $\delta$. So randomness has to enter somewhere between the data and the released weights.

There are three places to put it, and each gives a classic algorithm. Output perturbation trains as usual and adds noise to the final weights, while objective perturbation adds a random term to the training objective and minimizes it exactly. Gradient perturbation goes furthest back and adds noise to every gradient step. Each one hides with noise how far one record can move some quantity, its sensitivity. For a convex model with a ridge penalty, objective perturbation is usually the most accurate. For anything else only gradient perturbation works, and with clipping and random batches it becomes stochastic gradient descent (SGD) with noise, [DP-SGD](02-dp-sgd.md).

## The setting and the noise {#sec-dperm-setting}

A training set $D$ holds a number $n$ of records, each with $d$ features and a label of plus or minus one. Write $\ell(w, z)$ for the loss of the weights $w$ on a record $z$. With a ridge of strength $\lambda$, training minimizes the objective $J$,
$$J(w; D) = \frac1n\sum_{i=1}^n\ell(w, z_i) + \frac\lambda2\|w\|^2,$$
and its minimizer is written $w^*$. Two datasets are neighbours here when one **replaces** a record of the other. Every sensitivity is built from one more number $L$, a bound on the length of each record's loss gradient. For logistic regression with features of length at most 1, $L = 1$. A private release $\tilde w$ is judged by its **excess risk**, the amount by which its objective exceeds the smallest possible value.

The running example is logistic regression with ten features and a ridge of 0.01, at a target of $(1, 10^{-5})$. Every Gaussian method here calibrates its noise exactly ([The Laplace and Gaussian mechanisms](../part03-dp-foundations/03-laplace-gaussian.md)). For this target the noise must be 3.7306 times the sensitivity, and the classic formula asks for 30% more. The snippet in @sec-dperm-code computes the factor for any target.

For pure $\varepsilon$-DP, with no slack, the noise vector instead has a density that falls exponentially with its length, with a scale equal to the sensitivity divided by $\varepsilon$.

The guarantee holds only if your pipeline enforces the bounds it assumes. Clip or normalize every feature vector to length at most 1, with a bound fixed from public knowledge, because a norm computed from the private data leaks. Keep logistic labels at plus or minus one, and clip the labels of a linear regression to a public range, because $L$ grows with them. Fix $\lambda$ in advance, since choosing it by validation on private data is a search that must be paid for ([Hyperparameter tuning](03-hyperparameter-tuning.md)). Finally, the ridge must cover every weight, the intercept included. Scikit-learn's default solver leaves the intercept unpenalized, so drop it and append a constant feature instead, counted in the norm bound.

## Output perturbation {#sec-dperm-output}

Output perturbation treats training as a black box (Chaudhuri & Monteleoni 2008). When the loss is convex, the ridge makes the objective **strongly convex**, and that pins down how far one record can move the minimizer,
$$\|w^*(D) - w^*(D')\| \le \frac{2L}{\lambda n}.$$ {#eq-dperm-sens}
Write $\mu(\varepsilon, \delta)$ for the largest safe ratio of sensitivity to noise, which is 0.2681 for the running example's target. The release adds to the minimizer a Gaussian vector $Z$, with a standard deviation $\sigma$ on each coordinate,
$$\tilde w = w^*(D) + Z, \qquad Z \sim \mathcal N\big(0, \sigma^2I\big), \qquad \sigma = \frac{2L}{\lambda n\,\mu(\varepsilon, \delta)}.$$ {#eq-dperm-output}
With ten thousand records the sensitivity is 0.02, and the noise is 0.07461 per coordinate.

What goes wrong is the solver. The bound holds for the exact minimizer, and an optimizer that stops where the gradient has length $g$ adds $2g/\lambda$ to the sensitivity. In the running example, a tolerance of $10^{-3}$ raises the sensitivity to 0.22, eleven times the exact value. So stop on a gradient length below $L/n$, checked at the end, never on a fixed step count.

## Objective perturbation {#sec-dperm-objective}

Objective perturbation (Chaudhuri, Monteleoni & Sarwate 2011) adds a random linear term to the objective and minimizes the result exactly. All of the objective's curvature resists that term, so data that add curvature also reduce the damage.

It needs a loss of a particular form. The loss must depend on the weights only through the margin, the label times the inner product of the weights and the features. Its slope must be at most 1, its curvature between 0 and a known constant $c$, and the features of length at most 1. Logistic regression qualifies with $c = \frac14$. A small share of the budget pays for the curvature, and the noise gets what is left,
$$\varepsilon' = \varepsilon - 2\ln\Big(1 + \frac{c}{n\lambda}\Big).$$ {#eq-dperm-eps-prime}
The noise $b$ is the pure-DP noise with scale $2/\varepsilon'$, and the release is
$$\tilde w = \arg\min_w\Big[J(w; D) + \frac1n\langle b, w\rangle\Big].$$ {#eq-dperm-objective}
This is $\varepsilon$-DP. If $\varepsilon'$ comes out zero or negative, raise the ridge until the curvature share takes at most half of the budget. With ten thousand records in the running example, the noise keeps 0.9950 of the budget.

When the data's curvature near the minimizer is about $h$ in every direction, objective perturbation beats output perturbation by a factor of about $(h/\lambda)^2$. In the running example $h$ is about three times the ridge, and the excess risk is ten times smaller.

The price is exact minimization, which no solver delivers, and floating-point rounding can itself leak ([Discrete mechanisms](../part03-dp-foundations/08-discrete-mechanisms.md)). The loss must also be smooth, so a hinge loss must be smoothed first.

## Gradient perturbation: noisy gradient descent {#sec-dperm-gradient}

The third method makes each training step private and composes the steps. Its simplest form is full-batch gradient descent from zero weights, with a step size $\eta$ and a fixed number of steps, written $T$. At every step $t$, fresh Gaussian noise, written $\xi_t$, is added to the gradient, so the update is
$$w_{t+1} = w_t - \eta\big(\nabla J(w_t; D) + \xi_t\big), \qquad \xi_t \sim \mathcal N(0, \sigma^2I), \qquad \sigma = \frac{2L}{n}\cdot\frac{\sqrt T}{\mu(\varepsilon, \delta)}.$$ {#eq-dperm-noisy-gd}
Releasing every iterate is $(\varepsilon, \delta)$-DP for any loss whose per-record gradients have length at most $L$, convex or not. The noise grows like $\sqrt T$ because Gaussian steps compose exactly, by the rules of [Gaussian differential privacy](../part04-privacy-accounting/04-gaussian-dp.md).

So **the step count is a privacy parameter**. Too few steps leave optimization error, and too many drown the solution in noise. For a strongly convex objective you can choose $T$ from public quantities only, such as the largest curvature the logistic loss allows, and with ten thousand records this rule picks 218 steps. Choosing $T$ by its validation score instead is a computation on the data that must be paid for.

## Noisy gradient descent without a ridge {#sec-dperm-convex}

Many convex models have no useful ridge. Then keep the weights in a ball whose radius $R$ is fixed in advance from public knowledge. Shrink them back onto it after each noisy step, and release the average of the iterates. The noise is the same as in @eq-dperm-noisy-gd, and the excess risk falls like $RL\sqrt d/(n\varepsilon)$, which is optimal up to logarithms. The cost is computation, since ten thousand records already need 179,629 full passes, so practice runs the same idea on random minibatches.

## How the three compare {#sec-dperm-compare}

For strongly convex objectives, the best excess risk that any $(\varepsilon, \delta)$-DP algorithm can guarantee is of order
$$\frac{L^2\,d}{\lambda\,n^2\varepsilon^2},$$ {#eq-dperm-rate-strong}
up to logarithms. So a model with more parameters loses more accuracy at the same privacy level. And $n$ and $\varepsilon$ appear only as their product, so halving $\varepsilon$ costs the same accuracy as halving the data. On the running example, objective perturbation has the lowest excess risk at every dataset size.

Pure-DP noise also beats Gaussian noise there, but only because the dimension is small, since its mean squared length grows like $d^2$ rather than $d$, so prefer Gaussian noise for large models.

How much data do you need before the noise stops hurting? Predictions are more forgiving than excess risk. With a thousand records, objective perturbation lowers held-out accuracy from 0.747 to 0.736, and output perturbation lowers it to 0.681. With ten thousand records, all four releases come within 0.002 of non-private accuracy, so a low-dimensional convex model can be made private almost for free. As a rule, privacy adds little to the sampling error once the number of records is well above $d\ln(1/\delta)/\varepsilon^2$ (Bassily et al. 2019).

## When the model is not convex {#sec-dperm-nonconvex}

A neural network's loss is not convex, and that breaks output and objective perturbation, which both need a unique minimizer that moves little. Gradient perturbation keeps its guarantee, because its proof used only the gradient bound. But a network's gradients have no bound, so you **clip** each record's gradient to a length $C$ ([Beyond global sensitivity](../part03-dp-foundations/07-sensitivity.md#sec-sens-clip)), which biases what is optimized. Clipping, noise, random minibatches and an accountant together make [DP-SGD](02-dp-sgd.md).

## Libraries {#sec-dperm-libraries}

For a linear model you rarely need to write any of this. IBM's diffprivlib offers `LogisticRegression(epsilon=..., data_norm=...)`, modelled on scikit-learn's, which implements objective perturbation. Set `data_norm`, the largest feature-vector length, yourself and clip to it, because if you leave it unset diffprivlib computes it from the data and warns that this leaks. Its `LinearRegression` takes `epsilon` with `bounds_X` and `bounds_y`, the public ranges of the features and labels. Both stop an iterative solver at a tolerance, so the exact-minimizer condition holds only approximately.

Output perturbation needs only an ordinary solver and a noise draw. Scikit-learn's `LogisticRegression` minimizes $C$ times the summed loss plus half the squared weights, so $C = 1/(\lambda n)$ gives the objective of this chapter.

```text
from sklearn.linear_model import LogisticRegression
clf = LogisticRegression(C=1 / (lam * n), fit_intercept=False, tol=tol, max_iter=max_iter)
clf.fit(X, y)                                     # rows of X with length <= 1
sigma = 2 * L / (lam * n * mu)                    # mu from the snippet below
w = clf.coef_.ravel() + np.random.default_rng().normal(0, sigma, X.shape[1])
```

Choose `tol` well below $L/n$ and check the final gradient length, since the default tolerance voids the sensitivity bound. For a model trained by gradient steps, use Opacus as in [DP-SGD](02-dp-sgd.md). Its accountant assumes add/remove neighbours rather than replacement, so state the relation next to each $\varepsilon$.

## Code {#sec-dperm-code}

The snippet finds the largest safe Gaussian ratio by bisection and sizes the noise of output perturbation for the running example.

```python
from math import erfc, exp, sqrt


def phi(x: float) -> float:
    """Standard normal cumulative distribution function."""
    return 0.5 * erfc(-x / sqrt(2))


def gaussian_mu(eps: float, delta: float) -> float:
    """Largest sensitivity-to-noise ratio mu for which the Gaussian mechanism is (eps, delta)-DP."""
    lo, hi = 1e-6, 50.0
    for _ in range(200):
        mid = (lo + hi) / 2
        d = phi(mid / 2 - eps / mid) - exp(eps) * phi(-mid / 2 - eps / mid)
        lo, hi = (lo, mid) if d > delta else (mid, hi)
    return lo


mu = gaussian_mu(1.0, 1e-5)
print(f"{mu:.4f} {1 / mu:.4f}")  # 0.2681 3.7306
print(f"{2 / (0.01 * 10_000) / mu:.5f}")  # 0.07461
```

## Cheat sheet

| Method | Noise | Needs | Use when |
| --- | --- | --- | --- |
| Output perturbation | $\sigma = 2L/(\lambda n\mu)$ on $w^*$ | Convex loss, ridge, near-exact minimizer | Training must stay a black box |
| Objective perturbation | Scale $2/\varepsilon'$, $\varepsilon' = \varepsilon - 2\ln(1 + c/(n\lambda))$ | Margin loss with bounded slope and curvature, $\|x\| \le 1$, exact minimizer | Convex linear model; the data add curvature |
| Noisy gradient descent | $\sigma = (2L/n)\sqrt T/\mu$ | Bounded or clipped gradients; $T$ chosen privately | Any model; the basis of DP-SGD |

: Which private ERM method to use, with $\mu = \mu(\varepsilon, \delta)$ the largest safe Gaussian ratio, 0.2681 at $(1, 10^{-5})$. {#tbl-dperm-cheat}

**Rules of thumb**

- Stop exact-minimizer methods on a gradient tolerance well below $L/n$, never on a step count.
- The number of steps of noisy gradient descent is part of the privacy budget, and so is a $\lambda$ tuned on private data.
- Set feature and label bounds from public knowledge, and include the intercept in the ridge or drop it.

## Further reading

- Chaudhuri, Monteleoni & Sarwate, "Differentially Private Empirical Risk Minimization" (JMLR 2011): output and objective perturbation.
- Iyengar et al., "Towards Practical Differentially Private Convex Optimization" (IEEE S&P 2019): objective perturbation that tolerates inexact solvers, and a benchmark of the three methods.

**Derivations.** Every result here is derived step by step in [the full edition's chapter](../../book-full/part05-dp-ml/01-dp-erm.md).
