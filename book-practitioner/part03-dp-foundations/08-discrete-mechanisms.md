# Discrete and floating-point-safe mechanisms

## What it is {#sec-discrete-float}

The proof of the [Laplace mechanism](03-laplace-gaussian.md) is about densities on the real line, but a computer samples floating-point numbers, a finite set with uneven gaps, which the proof says nothing about. The textbook sampler adds $\pm b\ln(1/U)$ to the true answer, for a noise scale $b$ and a uniform draw $U$. Mironov (2012) showed that in double precision it leaks the true answer through the last bits of its output. Floats are dense near 0 and sparse far from it, so many outputs can come from a true answer of 0 or of 1, but not from both.

Such an output names its input with certainty. Differential privacy (DP) caps how much one record can change the chance of any output, through a privacy level $\varepsilon$ and a slack $\delta$ ([The definition of differential privacy](01-definition.md)). Now take a set $S$ of outputs that a dataset $D$ can reach but its neighbour never can. No factor $e^\varepsilon$ can lift the neighbour's probability of $S$ above 0, so for a mechanism $M$ the slack must cover
$$\delta \ge \Pr[M(D) \in S].$$ {#eq-support-delta}
In a toy sampler with 12-bit floats, such outputs carry probability 0.4791, and more bits do not shrink it.

So **DP is a property of the distribution the code actually produces.** There are two repairs. The **snapping mechanism** keeps floats, but clamps the answer, adds noise, rounds to a coarse power-of-two grid and clamps again. The other makes the query integer-valued and adds exact discrete Laplace or discrete Gaussian noise.

## When you meet it

You meet this when you review code that adds noise itself, such as `x + np.random.laplace(0, b)`, unsafe to ship, and when you choose a library. OpenDP's samplers are exact, and on integer data they give discrete Laplace and discrete Gaussian noise. diffprivlib offers `mechanisms.Geometric`, its name for the discrete Laplace, `mechanisms.GaussianDiscrete` and `mechanisms.Snapping`. Its continuous `Laplace`, which its `tools` functions use, draws from four uniform variates to resist this attack (Holohan & Braghin 2021), a mitigation rather than an exact sampler. Secure aggregation needs integer noise too ([Local, shuffle and distributed trust models](09-trust-models.md)).

In private training, Opacus adds continuous Gaussian noise in floating point, so the proof of [Private stochastic gradient descent](../part05-dp-ml/02-dp-sgd.md) covers idealized code. Its `secure_mode=True` draws the noise from a cryptographic generator to blunt these attacks, so check that a released model used it and states its guarantee for the idealized mechanism.

## How to use it {#sec-discrete-laplace}

Use a library on integer data. Its **discrete Laplace** noise gives each integer $z$ a probability proportional to $e^{-|z|/b}$, and you calibrate it like continuous noise. For an integer query that moves by at most $\Delta f$ between neighbours, it gives pure $\varepsilon$-DP with $\varepsilon = \Delta f/b$. For example, at $b = 1$ a count comes out exact with probability 0.4621.

The **discrete Gaussian** gives each integer $x$ a probability proportional to $e^{-x^2/(2\sigma^2)}$, for a width $\sigma$. Its exact privacy curve differs slightly: for a sensitivity of 1 at $\sigma = 1$, its $\delta$ at $\varepsilon = 1$ is 0.1414, where the continuous formula gives 0.1269. So either report the discrete curve, or account in [zero-concentrated DP](../part04-privacy-accounting/03-concentrated-dp.md). There the discrete Gaussian has the continuous Gaussian's privacy parameter $\rho = (\Delta f)^2/(2\sigma^2)$, and OpenDP accounts this way. Here `map(1)` gives the guarantee at sensitivity 1.

```python
import opendp.prelude as dp

dp.enable_features("contrib")
integers, distance = dp.atom_domain(T=int), dp.absolute_distance(T=int)
laplace = dp.m.make_laplace(integers, distance, scale=1.0)
print(laplace.map(1))  # 1.0
gauss = dp.m.make_gaussian(integers, distance, scale=2.0)
print(gauss.map(1))  # 0.125
print(gauss.output_measure)  # ZeroConcentratedDivergence
```

A real-valued query, such as a sum of clipped values, becomes integer in three steps. Divide by a step much smaller than the noise and round randomly, up with probability equal to the fractional part, so the result stays unbiased. Recompute the sensitivity in those units, since rounding can push neighbours slightly apart. Then add discrete noise and multiply back.

## What goes wrong {#sec-discrete-sampler}

Ask these questions of any implementation.

- Is every sensitivity computed from bounds fixed in advance? A `max()`, `quantile()` or `unique()` on the private data that feeds a bound, a scale or a key list leaks ([Beyond global sensitivity](07-sensitivity.md)).
- Under add/remove, is a mean a noisy sum over a noisy count, not over the true $n$ ([The Laplace and Gaussian mechanisms](03-laplace-gaussian.md#sec-lg-recipe))?
- Do all releases about the same people draw on one budget, so that five queries at $\varepsilon = 1$ are not claimed as $\varepsilon = 1$ ([Post-processing, group privacy and composition](02-properties.md#sec-dpprop-comp))?
- Is an exact statistic of the same data, such as a maximum, published beside the private ones? It has no guarantee.
- Is the noise drawn by a vetted library, or by an exact sampler with no floating-point `exp`, `log` or `random()` inside? Search for `np.random`, `random.`, `scipy.stats`, `default_rng`, `.laplace(` and `.normal(`.
- Does integer data get integer noise, and is every floating-point output snapped and clamped to a fixed range?
- Does each release draw fresh noise from a cryptographic source, such as `secrets.SystemRandom()`? A fixed seed makes the noise reproducible, and so removable.
- Is the running time fixed or padded? Rejection samplers' rounds depend on the value drawn, which Jin et al. (2022) exploit.
- Does the stated guarantee match the mechanism, including the discrete Gaussian's own curve and the sensitivity after rounding?

**Derivations.** Every result here is derived step by step in [the full edition's chapter](../../book-full/part03-dp-foundations/08-discrete-mechanisms.md).
