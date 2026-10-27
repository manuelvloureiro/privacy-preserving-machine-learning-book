# Probability distributions

## What it is

Most random numbers in a private system are counts or draws of noise. A count of independent yes-or-no outcomes with one probability is **binomial**. When the outcomes are many and each is rare, the count is nearly **Poisson**, with a variance equal to its mean.

Noise usually comes from one of two continuous families. The **Laplace distribution** with scale $b$ is an exponential density on each side of zero. The normal distribution is the bell curve, and its spread is the standard deviation $\sigma$.

## When you meet it

You meet them when you size a test set, call a sampler, or put an error bar on a noisy number.

## How to use it

An error rate measured on a test set is a binomial count divided by $n$, the number of test records. If the true rate is $p$, the measured rate has the standard deviation
$$\sqrt{\frac{p(1-p)}{n}}.$$
With 2,000 records and a true rate of 5%, that is about half a percentage point. Divided by $p$, it gives the relative error $\sqrt{(1-p)/(np)}$, which shrinks only as the expected number of errors $np$ grows. So a relative target $r$ needs $n = (1-p)/(p\,r^2)$ records. At $r = 10\%$, a 5% error rate needs 1,900 records. A rate fifty times smaller needs 99,900.

A sampled batch is the Poisson case. Suppose each of 60,000 records joins with probability 0.01. Then the batch size is nearly Poisson with mean and variance 600, so its standard deviation is about 24.5.

For noise, choose the spread so that a draw stays within $t$ of zero, except with a failure probability $\beta$. A Laplace draw with scale $b$ lands at least $t$ from zero with probability
$$\Pr[|X| \ge t] = e^{-t/b}.$$
Make this tail, or the normal one, equal to $\beta$ to get the largest spreads that meet the target,
$$b = \frac{t}{\ln(1/\beta)}, \qquad \sigma = \frac{t}{z_{1-\beta/2}}.$$
Here $z_{1-\beta/2}$ is the standard normal quantile. The two formulas split $\beta$ between the two sides, so for one side only, replace $\beta$ by $2\beta$. For example, Laplace noise of scale 2 stays within $2\ln 20 = 5.99$ of zero 95% of the time. As $\beta$ shrinks, the Laplace multiplier grows much faster than the normal one (@tbl-dist-multipliers), because the Laplace tails fall along straight lines on a log scale (@fig-laplace-vs-normal).

| Failure probability $\beta$ | 0.05 | $10^{-3}$ | $10^{-6}$ |
| --- | --- | --- | --- |
| Normal, $z_{1-\beta/2}$ standard deviations | 1.96 | 3.29 | 4.89 |
| Laplace, $\ln(1/\beta)$ scales | 3.00 | 6.91 | 13.82 |

: How many standard deviations or scales a draw stays within, except with probability $\beta$. {#tbl-dist-multipliers}

![Normal and Laplace densities with variance 2 (Laplace scale 1), on linear and log scales.](figures/laplace_vs_normal.svg){#fig-laplace-vs-normal}

On a vector, the noise adds up. Put independent normal noise of standard deviation $\sigma$ on each of $d$ coordinates. Then its length is close to $\sigma\sqrt d$, which is about 10 for a million coordinates at $\sigma = 0.01$. The snippet checks the numbers above and the integer case below.

```python
import math

import numpy as np
from scipy import optimize, stats

rng = np.random.default_rng(0)
draws = rng.laplace(loc=0.0, scale=2.0, size=1_000_000)  # scale is b, not the standard deviation
print(f"{draws.std():.3f} {stats.laplace(scale=2.0).std():.4f}")  # 2.832 2.8284
print(f"{stats.laplace(scale=2.0).isf(0.025):.2f}")  # 5.99
print(f"{stats.norm.isf(0.05 / 2):.2f} {stats.norm.isf(1e-3 / 2):.2f} {stats.norm.isf(1e-6 / 2):.2f}")  # 1.96 3.29 4.89
print(f"{math.log(1 / 0.05):.2f} {math.log(1 / 1e-3):.2f} {math.log(1 / 1e-6):.2f}")  # 3.00 6.91 13.82
print(math.ceil((1 - 0.05) / (0.05 * 0.1**2)), math.ceil((1 - 0.001) / (0.001 * 0.1**2)))  # 1900 99900
b = 10 / math.log(1 / 0.01)  # t = 10, beta = 0.01
tail = lambda a: 2 * stats.dlaplace(a).sf(9)  # integer version: P[|K| >= 10]
a = optimize.brentq(lambda a: tail(a) - 0.01, 0.01, 5)
print(f"{b:.4f} {tail(1 / b):.4f} {a:.4f} {tail(a):.4f}")  # 2.1715 0.0123 0.4817 0.0100
```


## What goes wrong

The most common bug is the wrong spread. NumPy's `rng.laplace(0, b)` and SciPy's `stats.laplace(scale=b)` take the scale $b$, whose standard deviation is $\sqrt2\,b$. But `rng.normal(0, sigma)` and `stats.norm(scale=sigma)` take the standard deviation, never the variance. So compare the spread of many draws with the one you want. For very small tail probabilities, call the tail function `sf` or its inverse `isf`, because `1 - cdf` rounds to zero.

For integer data, draw from an integer distribution, not a rounded continuous one. In SciPy's `stats.dlaplace(a)`, each step away from zero divides the probability by $e^{a}$, so $a$ takes the part of $1/b$. But the integer tail is heavier, so $a = 1/b$ gives 0.0123 where the snippet wants 0.01. Solve for $a$ numerically instead.

These formulas need independent records, so when one user has several rows, use the number of users as $n$. The normal approximation to a binomial count needs $np(1-p)$ of about 10 or more, so it fails for rare events.

**Derivations.** Every result here is derived step by step in [the full edition's chapter](../../book-full/part01-prerequisites/01-distributions.md).
