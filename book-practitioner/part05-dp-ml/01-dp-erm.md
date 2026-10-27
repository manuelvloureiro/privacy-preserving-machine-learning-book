# Private empirical risk minimization

## What it is {#sec-dperm-what}

Most of machine learning is **empirical risk minimization (ERM)**, which picks the weights $\theta$ with the least average loss over a training set. Write $D$ for that set and $n$ for its number of records. A ridge penalty usually keeps the weights small. Its strength is a number $\lambda$, and training minimizes
$$\mathcal L(\theta; D) = \frac1n\sum_{i=1}^n\ell(\theta; z_i) + \frac\lambda2\|\theta\|^2,$$
where $\ell(\theta; z)$ is the loss of the weights on one record $z$. Differential privacy (DP) asks that replacing one record barely change the released weights, so noise must enter the trained weights, the objective or each gradient step.

**Output perturbation** trains as usual and adds Gaussian noise to the final weights, with a small slack $\delta$. The noise is scaled to the most that the trained weights $\theta^*$ can move between datasets $D$ and $D'$ that differ in one record. For a convex loss, the ridge limits that move to
$$\|\theta^*(D) - \theta^*(D')\| \le \frac{2L}{\lambda n}.$$
Here the constant $L$ bounds the length of each record's loss gradient. It is 1 for logistic regression with labels $\pm1$ and features of length at most 1.

**Objective perturbation** adds a random linear term to the objective and minimizes the result exactly. Part of the privacy budget $\varepsilon$ pays for the loss's curvature, which is at most a constant $c$, one quarter for the logistic loss. The noise is scaled to what is left,
$$\varepsilon' = \varepsilon - 2\ln\Big(1 + \frac{c}{n\lambda}\Big).$$
The subtracted term grows as $n\lambda$ shrinks, so with little data the ridge must grow to keep $\varepsilon'$ positive.

**Gradient perturbation** adds noise to every gradient step instead. With random batches and clipped per-record gradients, it becomes noisy stochastic gradient descent (SGD), [DP-SGD](02-dp-sgd.md), the only option for a model that is not convex.

## When you meet it {#sec-dperm-when}

You meet these methods for linear models, such as a linear probe on features from a public pretrained network ([Public data and private fine-tuning](06-public-data.md)). Objective perturbation is usually the most accurate, because the objective's full curvature damps its noise, while only the ridge damps output perturbation's. So use output perturbation only when training is a black box.

Whether you have enough data depends on $n$ and $\varepsilon$. On the training objective they matter only through their product, so halving $\varepsilon$ costs as much as halving the data. On new data, with $d$ features, privacy adds less error than sampling does once $n$ is well above $d\ln(1/\delta)/\varepsilon^2$.

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

Averaged over 20 noise draws, test accuracy is 0.783 at $\varepsilon = 1$ and 0.946 at $\varepsilon = 4$, against 0.964 without noise. So with only 250 training records, accuracy comes close only at $\varepsilon = 4$. Nothing was tuned on the private rows, so the guarantee is pure DP for the 250 training rows, against replacing one record. The 69 stand-in rows get none.

## What goes wrong {#sec-dperm-wrong}

The most common failure is a bound taken from the data. Left unset, `data_norm` is computed from the training rows with a `PrivacyLeakWarning`, and the guarantee is gone. A range table computed from the same rows is no better, because each minimum and maximum is one record's value. So take every bound from public knowledge or public rows, and clip the private rows to it.

The second failure is the solver, because output and objective perturbation assume the exact minimizer. A solver that stops where the gradient still has length $g$ adds $2g/\lambda$ to the sensitivity. That is small beside the exact sensitivity $2L/(\lambda n)$ only when $g$ is well below $L/n$. So set the tolerance there, and check the final gradient rather than the number of steps.

The third failure is an unpenalized weight, because the sensitivity bound needs the ridge's curvature in every direction. Scikit-learn's loss, which diffprivlib reuses, leaves the intercept out of the ridge, so the snippet adds a constant feature. Fix $\lambda$ by a rule or on public data, since tuning it on private data is a search that must be paid for ([Hyperparameter tuning](03-hyperparameter-tuning.md)). Finally, these guarantees assume replaced records, unlike the add/remove accountants of later chapters, and the two do not convert freely. So state the relation next to every $\varepsilon$.

**Derivations.** Every result here is derived step by step in [the full edition's chapter](../../book-full/part05-dp-ml/01-dp-erm.md).
