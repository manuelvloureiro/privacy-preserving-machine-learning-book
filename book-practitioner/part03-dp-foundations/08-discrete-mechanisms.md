# Discrete and floating-point-safe mechanisms

## Why this matters

The proof of the [Laplace mechanism](03-laplace-gaussian.md) is about densities on the real line. A computer has only floating-point numbers, a finite set with uneven gaps, so the code that claims to sample Laplace noise produces some other distribution, and the proof says nothing about it. Mironov (2012) showed that the difference matters, because the textbook Laplace sampler, run in double precision, leaks the true answer through the last few bits of its output.

The rule of this chapter is that **differential privacy (DP) is a property of the distribution the code actually produces.** There are two ways to make the code match the proof. The first keeps floating point and repairs the output, by rounding it to a coarse grid and clamping it to a fixed range. The second makes the query integer-valued and samples integer noise exactly, which gives the discrete Laplace and discrete Gaussian mechanisms. In practice you use a vetted library that does one of the two. So the answer to "is `x + np.random.laplace(0, b)` safe to ship?" is no, and the same goes for any numpy or scipy sampler.

## Floating-point Laplace breaks differential privacy {#sec-discrete-float}

DP lets one record raise the probability of any set of outputs by at most a factor $e^\varepsilon$, plus a small slack $\delta$ ([The definition of differential privacy](01-definition.md)). Now suppose that a set $S$ of outputs can come from a dataset $D$ but never from its neighbour $D'$, so that seeing one of them names $D$ with certainty. The factor $e^\varepsilon$ then multiplies zero, and the whole probability of $S$ has to be paid for by the slack. This is the **support argument**, which says that for every $\varepsilon$,
$$\delta \;\ge\; \Pr[M(D) \in S].$$ {#eq-support-delta}
So one output that only one input can reach rules out pure DP. Finding such outputs needs only a list of what each input can produce, so this is the quickest audit of an implementation.

The textbook sampler fails this audit. It adds to the true answer $f(D)$ the noise $\pm b\ln(1/U)$, where $U$ is uniform between 0 and 1 and $b$ is the noise scale. In exact arithmetic that is Laplace noise, but in floating point the logarithm and the sum are both rounded. Floats are dense near 0 and sparse far from it, because their spacing doubles at each power of 2. Take true answers 0 and 1. The outputs near 0 then lie on a fine grid under the first input and on a coarser one under the second, so most of them can come from only one input.

The full edition enumerates every output of a toy sampler with 12-bit floats and $b = 1$. There the outputs that only $D$ can produce carry probability 0.4791, so the guarantee promised at $\varepsilon = 1$ is gone. More precision does not help, since about 48% of the probability sits on such outputs at every precision the full edition tries. Mironov found the same effect at the 53 bits of a double, in several DP libraries of the time.

## The snapping mechanism {#sec-discrete-snapping}

The leak lives in which float the output is, not in where it lands on the real line. So if the output must be a float, the low-order bits have to go. Mironov's **snapping mechanism** takes a query with sensitivity 1, a noise scale $\lambda$ and a clamping bound $B$. Write $\Lambda$ for the smallest power of 2 at or above $\lambda$. The mechanism clamps the true answer to the range from $-B$ to $B$, adds floating-point Laplace noise of scale $\lambda$, rounds the result to the nearest multiple of $\Lambda$, and clamps it again. Every input then reaches the same grid points.

Mironov proves that the snapped output is $\varepsilon$-DP at essentially the exact Laplace mechanism's level, $1/\lambda$, plus a floating-point term that is negligible for doubles and grows with $B$. Both clamps are needed, because rounding without clamping leaves the extreme outputs reachable from one input only.

Clamping biases answers near $\pm B$, so take $B$ from the range the answer can take, knowing that $\varepsilon$ grows with it.

## The discrete Laplace mechanism {#sec-discrete-laplace}

Most DP releases are counts and histograms, which take integer values, and integer noise for them has no low-order bits to leak. The **discrete Laplace** distribution keeps the Laplace density at the integers and normalizes it. Each step away from 0 multiplies the probability by the ratio $r = e^{-1/b}$, for a scale $b > 0$. Its probability mass function (PMF), the probability of each integer $z$, is
$$\Pr[Z = z] = \frac{1 - r}{1 + r}\,r^{|z|}, \qquad r = e^{-1/b}.$$ {#eq-dlap-pmf}

Adding this noise to an integer-valued query is the **geometric mechanism** of Ghosh, Roughgarden and Sundararajan (2009). Write $\Delta_1 f$ for the query's $\ell_1$ sensitivity, the largest total change across its coordinates when one record changes. With independent noise on each coordinate, the mechanism is pure $\varepsilon$-DP with
$$\varepsilon = \frac{\Delta_1 f}{b},$$ {#eq-dlap-eps}
the same calibration as the continuous Laplace mechanism. The query must be integer-valued, or the two inputs would reach different outputs, which is the support problem again.

For example, a count released at $\varepsilon = 1$ needs the scale $b = 1$. The exact count then comes out with probability 0.4621, and the noise has variance 1.8413, just below the 2 of continuous noise.

## The discrete Gaussian mechanism {#sec-discrete-gaussian}

The US Census Bureau used the integer analogue of the Gaussian mechanism (Canonne, Kamath and Steinke 2020) for the 2020 Census redistricting data. Pick a positive width $\sigma$. The **discrete Gaussian** $\mathcal N_{\mathbb Z}(0, \sigma^2)$ gives each integer $x$ a probability proportional to $e^{-x^2/(2\sigma^2)}$. The mechanism adds a draw of it to an integer-valued query whose answer moves by an integer $\Delta$ between neighbours.

Its exact privacy curve has the continuous Gaussian's form. Write $X$ for a draw of the noise. The smallest $\delta$ at a given $\varepsilon$ is
$$\delta(\varepsilon) = \Pr\Big[X > \frac{\varepsilon\sigma^2}{\Delta} - \frac\Delta2\Big] - e^\varepsilon\Pr\Big[X > \frac{\varepsilon\sigma^2}{\Delta} + \frac\Delta2\Big].$$ {#eq-dgauss-delta}
The values differ, though. At $\varepsilon = 1$ and $\sigma = 1$, the discrete $\delta$ is 0.1414 where the continuous formula gives 0.1269, and in general the discrete curve wiggles around the continuous one. So compute @eq-dgauss-delta directly, as the snippet in @sec-discrete-code does, and never report the continuous number for discrete noise.

Accounting by Rényi divergence is simpler. The discrete Gaussian's Rényi divergences never exceed the continuous ones, so it satisfies the same [zero-concentrated DP](../part04-privacy-accounting/03-concentrated-dp.md) guarantee. An accountant that composes Rényi divergences can therefore use the continuous Gaussian's bound unchanged. For $\sigma \ge 1$, treat its accuracy as the continuous Gaussian's.

## Making a real-valued query integer {#sec-discrete-integer}

Discrete noise needs an integer query, but gradients and sums of real values are not integers. The standard recipe follows.

1. Clip each record's contribution, so that the real-valued query has a known sensitivity ([Beyond global sensitivity](07-sensitivity.md)).
2. Divide by a step size $s$, small next to the noise, and round each value to an integer. Randomized rounding, which rounds up with probability equal to the fractional part, keeps the result unbiased.
3. Recompute the sensitivity after rounding, in units of $s$, because rounding can push two neighbours' values further apart. Calibrate to that larger value.
4. Add discrete Laplace or discrete Gaussian noise for that sensitivity, then multiply the result by $s$.

## Exact samplers, and choosing a library {#sec-discrete-sampler}

The discrete mechanisms are only as exact as their samplers. A coin flipped as `random() < exp(-gamma)` is inexact, because `exp` rounds and `random()` has only 53 bits. Canonne, Kamath and Steinke publish exact samplers built only from uniform random integers, and you call such samplers rather than write them.

An exact distribution is still not enough, because the running time is a second output. The number of rejection rounds depends on the value drawn, and Jin et al. (2022) show that such timing leaks are exploitable. When you review a library or a pipeline, ask these questions.

- Is the noise drawn by an exact sampler using only integer or rational arithmetic, or by a snapping-style repair, with no floating-point `exp`, `log` or `random()` inside the sampler?
- Do the random bits come from a cryptographic source, such as `secrets.SystemRandom()` in Python, rather than a seeded generator?
- Are the inputs and outputs clamped to a fixed range?
- Is the running time padded or fixed, so that rejection loops are hidden from the adversary?
- Does each release draw fresh noise?
- Does the reported guarantee match the mechanism actually used, including the discrete Gaussian's own $\delta(\varepsilon)$ and the sensitivity of the rounded query?

OpenDP's noise samplers, such as those behind `dp.m.make_laplace` and `dp.m.make_gaussian`, are designed to be exact, so they resist floating-point attacks, and on integer data they give the discrete Laplace and discrete Gaussian. The diffprivlib package offers discrete mechanisms as `diffprivlib.mechanisms.Geometric` and `diffprivlib.mechanisms.GaussianDiscrete`. Google's differential privacy library, which PyDP wraps for Python, also generates its noise with floating-point attacks in mind.

Private stochastic gradient descent adds continuous Gaussian noise to clipped gradients in floating point, so its proof ([Private stochastic gradient descent](../part05-dp-ml/02-dp-sgd.md)) is about idealized code. Opacus offers `secure_mode=True` when you create its `PrivacyEngine`, which draws the noise from a cryptographically secure generator and is meant to blunt floating-point attacks. Turn it on for any model you release, and record in the privacy statement that the guarantee is for the idealized mechanism.

## Code {#sec-discrete-code}

Even with a vetted library, check the calibration. The snippet computes the discrete Gaussian's $\delta$ from @eq-dgauss-delta by summing over the integers. Floating point is fine here, since it releases no data.

```python
from math import ceil, exp


def discrete_gaussian_delta(eps: float, sigma: float, sens: int = 1) -> float:
    """Exact delta(eps) of the discrete Gaussian mechanism, summed over the integers."""
    reach = ceil(40 * sigma) + sens
    weight = {y: exp(-y * y / (2 * sigma**2)) for y in range(-reach - sens, reach + 1)}
    total = sum(weight[y] for y in range(-reach, reach + 1))
    delta = 0.0
    for y in range(-reach, reach + 1):  # outputs of f(D) = 0 and f(D') = sens
        p, q = weight[y] / total, weight[y - sens] / total
        delta += max(p - exp(eps) * q, 0.0)
    return delta


print(f"{discrete_gaussian_delta(1.0, 1.0):.4f}")  # 0.1414
print(f"{discrete_gaussian_delta(1.0, 2.0):.4f}")  # 0.0072
```

## Cheat sheet

| Situation | Use | Guarantee |
| --- | --- | --- |
| Any production release | A vetted library, such as OpenDP | The library's documented guarantee |
| Any floating-point sampler | The support argument as an audit | $\delta \ge \Pr[M(D) \in S]$, with $S$ reachable from $D$ only |
| A real-valued output that must stay a float | Snapping: clamp, add noise, round to $\Lambda \ge \lambda$, clamp | About $1/\lambda$, the exact Laplace level |
| Counts, histograms, pure DP | Discrete Laplace, $\Pr[z] \propto e^{-\lvert z\rvert/b}$ | $\varepsilon = \Delta_1 f/b$ |
| Counts and sums, $\ell_2$ sensitivity, secure aggregation | Discrete Gaussian, $\Pr[x] \propto e^{-x^2/(2\sigma^2)}$ | $\delta(\varepsilon)$ by @eq-dgauss-delta; Rényi bounds as continuous |
| Gradients or other real values | Scale, round, recompute the sensitivity, add discrete noise | As the discrete mechanism used |
| Private training in Opacus | `secure_mode=True` | The idealized Gaussian guarantee |

: Which mechanism to use, and what it guarantees. {#tbl-discrete-cheat}

**Rules of thumb**

- Never ship `f(D) + laplace(b)` in floating point, including numpy or scipy samplers. Snap and clamp the output, or make the query integer-valued and use an exact sampler.
- Compute the discrete Gaussian's $\delta$ with @eq-dgauss-delta, because the continuous formula can understate it.
- Pad or fix the running time of rejection samplers.
- Turn on Opacus's `secure_mode` for any model you release.

## Further reading

- Mironov, "On Significance of the Least Significant Bits for Differential Privacy" (CCS 2012): the floating-point attack and the snapping mechanism.
- Canonne, Kamath & Steinke, "The Discrete Gaussian for Differential Privacy" (NeurIPS 2020): the discrete Gaussian and the exact samplers, the reference implementation to compare a library against.

**Derivations.** Every result here is derived step by step in [the full edition's chapter](../../book-full/part03-dp-foundations/08-discrete-mechanisms.md).
