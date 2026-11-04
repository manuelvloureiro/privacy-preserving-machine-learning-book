# Discrete and floating-point-safe mechanisms

## What it is {#sec-discrete-float}

The proof of the [Laplace mechanism](03-laplace-gaussian.md) assumes densities on the real line, but computers sample from a finite set of floating-point numbers with uneven spacing. The textbook sampler adds $\pm b\ln(1/U)$ to the true answer, where $b$ is the noise scale and $U$ is uniform. Mironov (2012) showed that this sampler leaks the true answer through low-order bits in double precision. Floating-point values are dense near 0 and sparse at large magnitudes. Some outputs can therefore occur for a true answer of 0 or 1 but not for intermediate values, identifying the input exactly.

Differential privacy (DP) does not hide this implementation leak. DP limits how much one record can change an output probability. The privacy level $\varepsilon$ bounds the multiplicative change, and the slack $\delta$ allows an additive failure probability ([The definition of differential privacy](01-definition.md)).

Let dataset $D$ denote the data and let $S$ denote a set of possible outputs. Suppose $D$ can produce outputs in $S$, but a neighbouring dataset cannot. The neighbour assigns probability zero to $S$, so multiplying that probability by $e^\varepsilon$ still gives zero. The mechanism therefore requires slack at least
$$\delta \ge \Pr[M(D) \in S].$$ {#eq-support-delta}
In a toy sampler with 12-bit floats, these outputs have probability 0.4791. Using more bits does not reduce this probability.

**DP applies to the output distribution that the code actually produces.** One repair is the **snapping mechanism**: clamp the answer, add noise, round to a coarse power-of-two grid, then clamp again. Another repair is to make the query integer-valued and add exact discrete Laplace or discrete Gaussian noise.

## When you meet it

This issue arises when reviewing code that samples noise directly, such as `x + np.random.laplace(0, b)`, or when choosing a library. OpenDP provides exact samplers, including discrete Laplace and discrete Gaussian noise for integer data. diffprivlib provides `mechanisms.Geometric`, `mechanisms.GaussianDiscrete` and `mechanisms.Snapping`. Its continuous `Laplace` mechanism, used by its `tools` functions, reduces this risk but is not exact. It draws four uniform variates (Holohan & Braghin 2021). Secure aggregation also requires integer noise ([Local, shuffle and distributed trust models](09-trust-models.md)).

Opacus adds continuous Gaussian noise in floating point during private training, so the proof of [Private stochastic gradient descent](../part05-dp-ml/02-dp-sgd.md) applies to idealized code. Its `secure_mode=True` uses a cryptographic generator, making these attacks harder. Verify that released models use this mode and state which idealized mechanism their guarantee assumes.

## How to use it {#sec-discrete-laplace}

Use a library to add noise to integer data. **Discrete Laplace** noise assigns a noise value $z$ probability proportional to $e^{-|z|/b}$ and uses the same calibration as continuous Laplace noise. Let $\Delta f$ denote the sensitivity of the integer-valued query $f$. Its privacy level is $\varepsilon = \Delta f/b$. At $b = 1$, a count is exact with probability 0.4621.

**Discrete Gaussian** noise assigns each integer $x$ probability proportional to $e^{-x^2/(2\sigma^2)}$, where $\sigma$ is the noise width. Its exact privacy curve is slightly weaker than the continuous Gaussian's. For sensitivity 1 and noise width 1, the discrete mechanism has privacy level $\varepsilon = 1$ with slack $\delta = 0.1414$. The corresponding slack for continuous noise is 0.1269. Report the discrete curve or use OpenDP's accounting in [zero-concentrated DP](../part04-privacy-accounting/03-concentrated-dp.md). Under that accounting, both Gaussian mechanisms have the same budget $\rho$, where $\rho = (\Delta f)^2/(2\sigma^2)$. In the snippet, `map(1)` returns each guarantee at sensitivity 1.

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

To integerize a real-valued query, such as a sum of clipped values, divide by a step much smaller than the noise scale. Round up with probability equal to the fractional part and down otherwise; this preserves the expectation. Recompute sensitivity in these units because rounding can increase the distance between neighbours. Then add discrete noise and multiply by the step.

## What goes wrong {#sec-discrete-sampler}

Check each implementation against the following conditions.

- Is every sensitivity computed from bounds fixed in advance, with no `max()`, `quantile()` or `unique()` on the private data ([Beyond global sensitivity](07-sensitivity.md))?
- Does a vetted library draw the noise, or an exact sampler with no floating-point `exp`, `log` or `random()` inside? Search for `np.random`, `random.`, `scipy.stats`, `default_rng`, `.laplace(` and `.normal(`.
- Does integer data get integer noise, and is every floating-point output snapped and clamped to a fixed range?
- Does each release draw fresh noise from a cryptographic source, such as `secrets.SystemRandom()`, rather than a fixed seed that makes it removable?
- Is the running time fixed or padded? The rounds of a rejection sampler depend on the value it draws, and Jin et al. (2022) use that time as a leak.
- Does the stated guarantee match the mechanism, including the discrete Gaussian's own curve and the sensitivity after rounding?

**Derivations.** Every result here is derived step by step in [the full edition's chapter](../../book-full/part03-dp-foundations/08-discrete-mechanisms.md).
