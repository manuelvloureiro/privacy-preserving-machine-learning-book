# Discrete and floating-point-safe mechanisms

## What it is {#sec-discrete-float}

The proof of the [Laplace mechanism](03-laplace-gaussian.md) is about densities on the real line. But a computer samples floating-point numbers, a finite set with uneven gaps. The textbook sampler adds $\pm b\ln(1/U)$ to the true answer, for a noise scale $b$ and a uniform draw $U$. Mironov (2012) showed that in double precision this sampler leaks the true answer through the last bits of its output. Floats are dense near 0 and sparse far from it. So many outputs can come from a true answer of 0 or of 1, but not from the two. Such an output names its input with certainty.

Differential privacy (DP) cannot cover that leak. DP limits how much one record can change the chance of an output, by a privacy level $\varepsilon$ and a slack $\delta$ ([The definition of differential privacy](01-definition.md)). Take a set $S$ of outputs that a dataset $D$ can reach but its neighbour never can. No factor $e^\varepsilon$ lifts the neighbour's probability of $S$ above 0, so for a mechanism $M$ the slack must cover all of
$$\delta \ge \Pr[M(D) \in S].$$ {#eq-support-delta}
In a toy sampler with 12-bit floats, such outputs carry probability 0.4791, and more bits do not shrink it.

So **DP is a property of the distribution that the code actually produces.** There are two repairs. The **snapping mechanism** keeps floats. It clamps the answer, adds noise, rounds to a coarse power-of-two grid and clamps again. The other repair makes the query integer-valued and adds exact discrete Laplace or discrete Gaussian noise.

## When you meet it

You meet this problem when you review code that adds noise itself, such as the unsafe `x + np.random.laplace(0, b)`, and when you choose a library. OpenDP's samplers are exact, and on integer data they give discrete Laplace and discrete Gaussian noise. diffprivlib has `mechanisms.Geometric` (its name for the discrete Laplace), `mechanisms.GaussianDiscrete` and `mechanisms.Snapping`. Its continuous `Laplace`, which its `tools` functions use, is a mitigation, not an exact sampler: it draws four uniform variates against this attack (Holohan & Braghin 2021). Secure aggregation needs integer noise too ([Local, shuffle and distributed trust models](09-trust-models.md)).

In private training, Opacus adds continuous Gaussian noise in floating point, so the proof of [Private stochastic gradient descent](../part05-dp-ml/02-dp-sgd.md) covers only idealized code. Its `secure_mode=True` takes the noise from a cryptographic generator, which makes these attacks harder. For a released model, make sure that it used this mode and that its guarantee names the idealized mechanism.

## How to use it {#sec-discrete-laplace}

Use a library on integer data. Its **discrete Laplace** noise gives each integer $z$ a probability proportional to $e^{-|z|/b}$, and you calibrate it like continuous noise. For an integer query with sensitivity $\Delta f$, it gives $\varepsilon = \Delta f/b$. At $b = 1$, a count comes out exact with probability 0.4621.

The **discrete Gaussian** gives each integer $x$ a probability proportional to $e^{-x^2/(2\sigma^2)}$, for a width $\sigma$. Its exact privacy curve is a little worse than the continuous one. At sensitivity 1 and $\sigma = 1$, its $\delta$ at $\varepsilon = 1$ is 0.1414, against 0.1269. So report the discrete curve, or account as OpenDP does in [zero-concentrated DP](../part04-privacy-accounting/03-concentrated-dp.md). In that accounting, the discrete Gaussian has the same privacy parameter as the continuous one, $\rho = (\Delta f)^2/(2\sigma^2)$. In the snippet, `map(1)` gives each guarantee at sensitivity 1.

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

To make a real-valued query integer, such as a sum of clipped values, divide it by a step much smaller than the noise. Round up with probability equal to the fractional part, and down otherwise, so that the result has no bias. Compute the sensitivity again in those units, because rounding can move neighbours a little further apart. Then add discrete noise and multiply back.

## What goes wrong {#sec-discrete-sampler}

Ask these questions of each implementation.

- Is every sensitivity computed from bounds fixed in advance, with no `max()`, `quantile()` or `unique()` on the private data ([Beyond global sensitivity](07-sensitivity.md))?
- Does a vetted library draw the noise, or an exact sampler with no floating-point `exp`, `log` or `random()` inside? Search for `np.random`, `random.`, `scipy.stats`, `default_rng`, `.laplace(` and `.normal(`.
- Does integer data get integer noise, and is every floating-point output snapped and clamped to a fixed range?
- Does each release draw fresh noise from a cryptographic source, such as `secrets.SystemRandom()`, rather than a fixed seed that makes it removable?
- Is the running time fixed or padded? The rounds of a rejection sampler depend on the value it draws, and Jin et al. (2022) use that time as a leak.
- Does the stated guarantee match the mechanism, including the discrete Gaussian's own curve and the sensitivity after rounding?

**Derivations.** Every result here is derived step by step in [the full edition's chapter](../../book-full/part03-dp-foundations/08-discrete-mechanisms.md).
