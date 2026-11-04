# Private empirical risk minimization

## What it is {#sec-dperm-what}

**Empirical risk minimization (ERM)** selects weights $\theta$ that minimize average loss on a training set. Let $D$ be the training set and $n$ its record count. Let $\lambda$ be the strength of the ridge penalty. Training minimizes
$$\mathcal L(\theta; D) = \frac1n\sum_{i=1}^n\ell(\theta; z_i) + \frac\lambda2\|\theta\|^2,$$
where $\ell(\theta; z)$ is the loss of the weights on one record $z$. Differential privacy (DP) requires that replacing one record has limited effect on released weights. Add noise to the final weights, the objective or each gradient step.

**Output perturbation** trains normally, then adds Gaussian noise to the final weights, with slack $\delta$. Calibrate noise to the maximum change in trained weights $\theta^*$ between neighbouring datasets $D$ and $D'$. For a convex loss, ridge regularization bounds that change:
$$\|\theta^*(D) - \theta^*(D')\| \le \frac{2L}{\lambda n}.$$
Here, $L$ bounds the maximum norm of one record's loss gradient. For logistic regression, labels are $\pm1$ and feature norms are at most 1. These conditions give L of 1.

**Objective perturbation** adds a random linear term to the objective and minimizes it exactly. Part of privacy budget $\varepsilon$ accounts for loss curvature, bounded by constant $c$ (one quarter for logistic loss). The remaining budget is
$$\varepsilon' = \varepsilon - 2\ln\Big(1 + \frac{c}{n\lambda}\Big).$$
As $n\lambda$ decreases, the subtracted term increases. With little data, increase the ridge strength to keep $\varepsilon'$ positive.

**Gradient perturbation** adds noise at every gradient step. With random batches and clipped per-record gradients, this is noisy stochastic gradient descent (SGD), or [DP-SGD](02-dp-sgd.md), which also applies to non-convex models.

## When you meet it {#sec-dperm-when}

Use these methods for linear models, such as a linear probe on features from a public pretrained network ([Public data and private fine-tuning](06-public-data.md)). Objective perturbation often gives higher accuracy because the objective's full curvature reduces its noise; output perturbation relies only on ridge curvature. Use output perturbation when training is a black box.

Required data depends on $n$ and $\varepsilon$. In the training objective, they enter through their product, so halving $\varepsilon$ has the same effect as halving the sample size. On new data with $d$ features, privacy adds less error than sampling once $n$ is much larger than $d\ln(1/\delta)/\varepsilon^2$.

## How to use it {#sec-dperm-use}

IBM's diffprivlib offers objective perturbation as `LogisticRegression(epsilon=..., data_norm=...)`, with `data_norm` bounding each row's length. Its version 0.6.6 fails with scikit-learn 1.8, so the snippet calls the mechanism it wraps, `mechanisms.Vector`, with the curvature bound $c$ as `function_sensitivity`. The breast-cancer data has no public data dictionary for feature bounds. So the snippet treats 500 rows as private and 69 as a public stand-in. It clips the private rows to the stand-in's log-transformed ranges and maps them to $[-1, 1]$. A constant 1 replaces the intercept, and dividing by $\sqrt{31}$ keeps each row's length at most 1.

```python
import numpy as np
from diffprivlib.mechanisms import Vector
from scipy.optimize import minimize
from sklearn.datasets import load_breast_cancer
data = load_breast_cancer()
public, private = np.log1p(data.data[:69]), np.log1p(data.data[69:])  # 69 stand-in rows fix the bounds
lo, hi = public.min(axis=0), public.max(axis=0)
X = np.hstack([2 * (np.clip(private, lo, hi) - lo) / (hi - lo) - 1, np.ones((500, 1))]) / np.sqrt(31)
y = 2 * data.target[69:] - 1
Xt, yt, n = X[::2], y[::2], 250  # even rows train, odd rows test
lam = 1 / n  # diffprivlib's default ridge, fixed in advance

def objective(w):  # J(w) and its gradient
    m = yt * (Xt @ w)
    return np.logaddexp(0, -m).mean() + lam / 2 * w @ w, lam * w - Xt.T @ (yt / (1 + np.exp(m))) / n

def accuracy(eps, seed):  # eps = inf adds no noise
    vector = Vector(epsilon=eps, function_sensitivity=0.25, dimension=31, alpha=n * lam, n=n, random_state=seed)
    w = minimize(vector.randomise(objective), np.zeros(31), jac=True, options={"gtol": 1e-9}).x
    return np.mean(np.sign(X[1::2] @ w) == y[1::2])

print(" ".join(f"{np.mean([accuracy(e, s) for s in range(20)]):.3f}" for e in [1, 4, np.inf]))  # 0.783 0.946 0.964
```

Across 20 noise draws, test accuracy is 0.783 at $\varepsilon = 1$ and 0.946 at $\varepsilon = 4$. Without noise, it is 0.964. With 250 training records, accuracy is close to the non-private result only at $\varepsilon = 4$. No parameter was tuned on the private rows, so the guarantee is pure DP for the 250 training rows under replace-one adjacency. The 69 stand-in rows receive no protection.

## What goes wrong {#sec-dperm-wrong}

The common failure is deriving a bound from private data. If omitted, `data_norm` is calculated from training rows and raises `PrivacyLeakWarning`; the guarantee is then invalid. A range table computed from those rows also leaks through its minima and maxima. Set bounds from public knowledge or public rows, then clip private rows to them.

The solver can also violate assumptions: output and objective perturbation require the exact minimizer. If optimization stops with gradient norm $g$, sensitivity increases by $2g/\lambda$. This increase is negligible relative to exact sensitivity $2L/(\lambda n)$ only when $g$ is much less than $L/n$. Set a corresponding tolerance and inspect the final gradient norm, not just the iteration count.

The sensitivity bound also requires ridge curvature in every direction. Scikit-learn's loss, reused by diffprivlib, leaves the intercept unpenalized, so the snippet adds a constant feature. Set $\lambda$ by a fixed rule or public data; private-data tuning must be accounted for ([Hyperparameter tuning](03-hyperparameter-tuning.md)). These guarantees use replace-one neighbours, unlike later add/remove accounting, and the bounds are not directly interchangeable. State the relation with every $\varepsilon$.

**Derivations.** Every result here is derived step by step in [the full edition's chapter](../../book-full/part05-dp-ml/01-dp-erm.md).
