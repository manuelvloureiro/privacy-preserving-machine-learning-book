# Private empirical risk minimization

## What it is {#sec-dperm-what}

Most of machine learning is **empirical risk minimization (ERM)**, which picks the weights $\theta$ with the least average loss over a training set. Write $D$ for that set and $n$ for its number of records. Usually a ridge penalty keeps the weights small. Its strength is a number $\lambda$, and training minimizes
$$\mathcal L(\theta; D) = \frac1n\sum_{i=1}^n\ell(\theta; z_i) + \frac\lambda2\|\theta\|^2,$$
where $\ell(\theta; z)$ is the loss of the weights on one record $z$. Differential privacy (DP) asks that replacing one record barely change the released weights. So noise must enter in the trained weights, in the objective or in each gradient step.

**Output perturbation** trains as usual and adds Gaussian noise to the final weights, which gives a guarantee with a small slack $\delta$. The noise is scaled to the most one record can move the minimizer $\theta^*$. Write $D$ and $D'$ for two datasets that differ in one record. For a convex loss, the ridge limits that move to
$$\|\theta^*(D) - \theta^*(D')\| \le \frac{2L}{\lambda n}.$$
Here the constant $L$ bounds the length of each record's loss gradient. It is 1 for logistic regression with labels $\pm1$ and features of length at most 1.

**Objective perturbation** adds a random linear term to the objective and minimizes the result exactly. Part of the privacy budget $\varepsilon$ pays for the loss's curvature, which is at most a constant $c$, one quarter for the logistic loss. The noise is scaled to what is left,
$$\varepsilon' = \varepsilon - 2\ln\Big(1 + \frac{c}{n\lambda}\Big).$$
This must stay positive, so with little data the ridge $\lambda$ has to grow.

**Gradient perturbation** adds noise to every gradient step instead. With random batches and each record's gradient clipped to a fixed length, it becomes noisy stochastic gradient descent (SGD), [DP-SGD](02-dp-sgd.md), the only option for a model that is not convex.

## When you meet it {#sec-dperm-when}

You meet these methods for linear models, such as a logistic regression or a linear probe on features from a public pretrained network ([Public data and private fine-tuning](06-public-data.md)). IBM's diffprivlib offers `LogisticRegression(epsilon=..., data_norm=...)` for objective perturbation, with `data_norm` bounding each row's length.

The first decision is the method, and with a convex loss and a ridge, objective perturbation is usually the most accurate. Its noise is damped by all the objective's curvature, while output perturbation's is sized by the ridge alone. Use output perturbation when training is a black box, DP-SGD otherwise.

The second decision is whether you have enough data. On a ridge-penalized training objective, $n$ and $\varepsilon$ matter only through their product, so halving $\varepsilon$ costs as much as halving the data. On new data, with $d$ features and an $(\varepsilon, \delta)$ guarantee, privacy adds less error than sampling once $n$ is well above $d\ln(1/\delta)/\varepsilon^2$.

## How to use it {#sec-dperm-use}

diffprivlib 0.6.6's `LogisticRegression` fails with scikit-learn 1.8, so call the mechanism it wraps, `mechanisms.Vector`, directly. The bundled breast-cancer data has no public data dictionary, so the snippet sets 69 rows aside as a public stand-in and treats the other 500 as private. It takes the logarithm $\ln(1 + x)$ of each feature value $x$, reads each feature's range from the stand-in rows only, clips the private rows into that range and maps it to $[-1, 1]$. A constant 1 replaces the intercept, and dividing by $\sqrt{31}$ keeps each row's length at most 1. In the call, `function_sensitivity` is the curvature bound $c$.

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

Averaged over 20 noise draws, test accuracy at $\varepsilon = 1$ is 0.783, against 0.964 without noise. With only 250 training records, the budget has to rise to 4 before accuracy comes close, at 0.946. Nothing was tuned on the private rows, since all 30 features are used with diffprivlib's default ridge. So the guarantee is pure DP with no slack, for the 250 private training rows and for neighbours that replace one record. The 69 stand-in rows get no guarantee, because their ranges are printed in the code.

## What goes wrong {#sec-dperm-wrong}

The most common failure is a bound taken from the data. For example, left unset, `data_norm` is computed from the training rows with a `PrivacyLeakWarning`, and the guarantee is gone. A table of ranges in a dataset's description is no better when it was computed from the same rows. Breast cancer's is, so each minimum and maximum is one record's value. So take every bound from public knowledge or public rows, and clip the private rows to it.

The second failure is the solver, because output and objective perturbation assume the exact minimizer. A solver that stops where the gradient still has length $g$ adds $2g/\lambda$ to the sensitivity, which must stay small beside the exact $2L/(\lambda n)$. So set the tolerance well below $L/n$, and check the final gradient rather than the number of steps.

The third failure is an unpenalized weight, because the sensitivity bound needs the ridge's curvature in every direction. Scikit-learn's loss, which diffprivlib reuses, leaves the intercept out, so the snippet appends a constant feature instead. Fix $\lambda$ by a rule or on public data, since tuning it on private data is a search that must be paid for ([Hyperparameter tuning](03-hyperparameter-tuning.md)). Finally, these guarantees assume replaced records, unlike the add/remove accountants of later chapters, and the two do not convert freely, so state the relation next to every $\varepsilon$.

**Derivations.** Every result here is derived step by step in [the full edition's chapter](../../book-full/part05-dp-ml/01-dp-erm.md).
