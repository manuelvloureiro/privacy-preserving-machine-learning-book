"""The small pure functions that most chapter scripts need: the normal and Laplace densities and tails, the
normal quantile, a bisection, a stable log-sum-exp and the logistic function.

Each is written exactly as the scripts wrote their own copies, so replacing a copy by an import changes no
number: the normal CDF goes through ``math.erfc`` and the quantile through 200 bisection steps on [-40, 40].
"""

import math
from collections.abc import Callable, Sequence

import numpy as np
from numpy.typing import NDArray

Array = NDArray[np.float64]


def normal_cdf(x: float) -> float:
    """Probability that a standard normal variable is at most x."""
    return 0.5 * math.erfc(-x / math.sqrt(2))


def normal_sf(x: float) -> float:
    """Probability that a standard normal variable exceeds x: the upper tail."""
    return 0.5 * math.erfc(x / math.sqrt(2))


def normal_pdf(x: float, mu: float = 0.0, sigma: float = 1.0) -> float:
    """Density of the normal distribution with mean mu and standard deviation sigma at x."""
    return math.exp(-((x - mu) ** 2) / (2 * sigma**2)) / (sigma * math.sqrt(2 * math.pi))


def normal_quantile(p: float) -> float:
    """The point below which a standard normal variable falls with probability p, by bisection."""
    lo, hi = -40.0, 40.0
    for _ in range(200):
        mid = (lo + hi) / 2
        if normal_cdf(mid) < p:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def laplace_pdf(x: float, mu: float = 0.0, b: float = 1.0) -> float:
    """Density of the Laplace distribution centred at mu with scale b at x."""
    return math.exp(-abs(x - mu) / b) / (2 * b)


def binomial_pmf(k: int, n: int, p: float) -> float:
    """Probability of exactly k successes in n independent trials that each succeed with probability p."""
    return math.comb(n, k) * p**k * (1 - p) ** (n - k)


def bisect(f: Callable[[float], float], lo: float, hi: float, steps: int = 200) -> float:
    """A root of f in [lo, hi], where f changes sign, after `steps` halvings of the interval."""
    f_lo = f(lo)
    for _ in range(steps):
        mid = (lo + hi) / 2
        if (f(mid) > 0) == (f_lo > 0):
            lo, f_lo = mid, f(mid)
        else:
            hi = mid
    return (lo + hi) / 2


def logsumexp(values: Sequence[float] | Array) -> float:
    """log(sum(exp(values))), computed without overflow by factoring out the largest value."""
    arr = np.asarray(values, dtype=np.float64)
    top = float(np.max(arr))
    return top + math.log(float(np.sum(np.exp(arr - top))))


def sigmoid(s: Array) -> Array:
    """The logistic function 1 / (1 + exp(-s)), through tanh so that large |s| does not overflow."""
    return 0.5 * (1.0 + np.tanh(0.5 * s))
