# Probability distributions

## What it is

Most random numbers in a machine-learning system are counts or draws of noise. A count of independent yes-or-no outcomes, each with the same probability, is **binomial**. When the outcomes are many and each is rare, the count becomes **Poisson**, whose variance equals its mean. Noise usually comes from one of two continuous families. The **Laplace distribution** with scale $b$ is an exponential density mirrored around zero, and the normal distribution is the bell curve, set by its mean and its standard deviation $\sigma$.

## When you meet it

You meet these families when you size a test set, check which argument of a sampler sets its spread, or put an error bar on a noisy number.

## How to use it

An error rate $p$ measured on a test set of $n$ records is a binomial count of errors divided by $n$, so its standard deviation is
$$\sqrt{\frac{p(1-p)}{n}}.$$
With 2,000 records and a true error rate of 5%, this is about half a percentage point. Divided by $p$, it becomes the relative standard deviation $\sqrt{(1-p)/(np)}$, which shrinks only as the expected number of errors $np$ grows. A relative target $r$ thus needs $n = (1-p)/(p\,r^2)$ records. At $r = 10\%$, a 5% error rate needs 1,900 records, and a rate fifty times smaller needs 99,900.

A sampled batch is the Poisson case. If each of 60,000 records joins it with probability 0.01, its size is nearly Poisson with mean 600. So its variance is also 600, and its standard deviation about 24.5.

For noise, you pick the spread so that a draw stays within $t$ of zero except with a failure probability $\beta$. A Laplace draw with scale $b$ lands at least $t$ from zero with probability
$$\Pr[|X| \ge t] = e^{-t/b}.$$
Setting this tail, or the normal one, equal to $\beta$ gives the largest spreads that meet the target,
$$b = \frac{t}{\ln(1/\beta)}, \qquad \sigma = \frac{t}{z_{1-\beta/2}}.$$
Here $z_{1-\beta/2}$ is the standard normal quantile. Both formulas split $\beta$ between the two sides, so to bound one side only, replace $\beta$ by $2\beta$. For example, Laplace noise of scale 2 stays within $2\ln 20 = 5.99$ of zero 95% of the time. As $\beta$ shrinks, the Laplace multiplier grows much faster than the normal one (@tbl-dist-multipliers).

| Failure probability $\beta$ | 0.05 | $10^{-3}$ | $10^{-6}$ |
| --- | --- | --- | --- |
| Normal, $z_{1-\beta/2}$ standard deviations | 1.96 | 3.29 | 4.89 |
| Laplace, $\ln(1/\beta)$ scales | 3.00 | 6.91 | 13.82 |

: How many standard deviations or scales a draw stays within, except with probability $\beta$. {#tbl-dist-multipliers}

The log scale of @fig-laplace-vs-normal shows why: the Laplace tails fall along straight lines, far above the normal density's parabola.

![Normal and Laplace densities with variance 2 (Laplace scale 1), on linear and log scales.](figures/laplace_vs_normal.svg){#fig-laplace-vs-normal}

Noise on a vector adds up over its coordinates. With $d$ independent normal coordinates of standard deviation $\sigma$, the length is close to $\sigma\sqrt d$. A million coordinates at $\sigma = 0.01$ give a length of about 10. The snippet checks these numbers and the integer case below.

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

The commonest bug is a mismatched spread. NumPy's `rng.laplace(0, b)` and SciPy's `stats.laplace(scale=b)` take the scale $b$, whose standard deviation is $\sqrt2\,b$. But `rng.normal(0, sigma)` and `stats.norm(scale=sigma)` take the standard deviation, never the variance. So compare the standard deviation of many draws with the intended one. For tiny tail probabilities, call the tail function `sf` or its inverse `isf`, because `1 - cdf` rounds to zero.

For integer data, draw from an integer distribution rather than rounding. In SciPy's `stats.dlaplace(a)`, the probability of an integer falls by a factor $e^{-a}$ per step away from zero, so $a$ plays the part of $1/b$. Its tail is heavier, so $a = 1/b$ gives 0.0123 where the snippet targets 0.01. Solve for $a$ numerically instead.

These formulas assume independent records, so when a user has several rows, take the test-set size $n$ to be the number of users. The normal approximation to a binomial count needs $np(1-p)$ of about 10 or more, so it fails for rare events.

**Derivations.** Every result here is derived step by step in [the full edition's chapter](../../book-full/part01-prerequisites/01-distributions.md).
