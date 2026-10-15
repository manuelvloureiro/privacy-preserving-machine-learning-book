# Discrete and floating-point-safe mechanisms

## Why this matters

The proof of the [Laplace mechanism](03-laplace-gaussian.md) compares two densities on the real line. Your computer has no real numbers. It has floating-point numbers, a finite set with uneven gaps, and random bits, and the function that "samples $\mathrm{Lap}(b)$" produces some distribution on that finite set that the proof says nothing about. Mironov (2012) showed that the textbook sampler, in double precision, leaks the true answer through the low-order bits of its output. How do you make the noise your code draws match the noise the proof is about?

The one idea: **differential privacy (DP) is a property of the output distribution the code actually produces.** There are two routes to the right one. Keep floating point and repair the output, by rounding it to a coarse grid and clamping it. Or make the query integer-valued and sample integer noise exactly, with integer and rational arithmetic only; that gives the discrete Laplace and discrete Gaussian mechanisms, whose guarantees match the continuous ones. @tbl-discrete-map maps the decisions to the sections.

| Decision | Answer | Section |
| --- | --- | --- |
| Can I add `numpy`-style Laplace or Gaussian noise to a float? | No: the output's low-order bits leak the input | @sec-discrete-float |
| My output must be a float | Snap it to a coarse grid and clamp it | @sec-discrete-snapping |
| Counts and histograms with pure DP | The discrete Laplace (geometric) mechanism | @sec-discrete-laplace |
| Dense integer vectors, or noise before secure aggregation | The discrete Gaussian mechanism | @sec-discrete-gaussian |
| How do I know a library does it right? | Exact integer samplers from a cryptographic source, with constant-time behaviour | @sec-discrete-sampler |

: The decisions behind floating-point-safe noise, and where this chapter makes each one. {#tbl-discrete-map}

## Floating-point Laplace breaks differential privacy {#sec-discrete-float}

**Idea.** If some output can be produced from $D$ but never from its neighbour $D'$, seeing it reveals $D$ with certainty, and no factor $e^\varepsilon$ covers that. The textbook sampler computes $f(D) + \xi\cdot b\ln(1/U)$, with $U$ uniform on $(0, 1)$ and $\xi$ a random sign, rounding at every step. Adding the same finite set of noise values to $f(D)$ or to $f(D')$ and rounding lands on two different sets of floats. Floats are packed more densely near 0 than near 1, so most fine-grained outputs from $f(D) = 0$ can't come from $f(D') = 1$.

**Result.** Let $S$ be the outputs reachable from $D$ but not from $D'$. Any $(\varepsilon, \delta)$-DP mechanism must have
$$\delta \;\ge\; \Pr[M(D) \in S]$$ {#eq-support-delta}
for every $\varepsilon$, however large. This support argument is the quickest audit of an implementation: outputs that only one input reaches rule out every finite $\varepsilon$, and $\delta$ is at least their probability.

**Example.** A hospital releases a count with the textbook sampler at $b = 1$, which would be 1-DP in exact arithmetic. An analyst who knows the count is 0 or 1 checks whether the released float can be produced from 1. In a toy model with 12-bit floats, small enough to enumerate exactly, $\Pr[M(D) \in S] = 0.4791$: the check identifies the count 0 with certainty 47.91% of the time, where the proof promised odds moved by at most a factor $e$ (@fig-discrete-float-support).

![The toy 12-bit floating-point Laplace sampler with $b = 1$. Left: the outputs that $f(D) = 0$ (top) and $f(D') = 1$ (bottom) can produce in a narrow window, coloured by whether the other input can produce them too. Right: the probability that $M(D)$ produces an output $M(D')$ cannot, against the number of significand bits, for the textbook sampler (flat, near 48%) and the snapped one (zero).](figures/discrete_float_support.svg){#fig-discrete-float-support}

**Pitfalls.** More precision does not help: from 6 to 14 bits about 48% of the mass stays on outputs the other input can't produce, because extra bits refine both grids together. Mironov found the same effect at the 53 bits of a double in several DP libraries of the time, and Jin et al. (2022) found floating-point and timing leaks in libraries released after. The tails leak too: a 53-bit uniform is never below $2^{-53}$, so the noise never exceeds about $36.74\,b$, and outputs just beyond that come from one input only. So `numpy.random.laplace`, or any `f + scale * log(u)`, is not a DP mechanism, whatever $\varepsilon$ the surrounding code claims.

## The snapping mechanism {#sec-discrete-snapping}

**Idea.** If the output must be a float, its low-order bits have to go. The **snapping mechanism** of Mironov (2012) rounds the noisy answer to a grid at least as coarse as the noise scale, so every input reaches the same outputs, and clamps input and output to $[-B, B]$ so that floating-point errors stay small next to the grid spacing. Rounding exact Laplace noise would cost no privacy; what remains is to pay for the floating-point error at the grid's edges.

**Result.** For a query with sensitivity 1 and noise scale $\lambda$, let $\Lambda$ be the smallest power of 2 at or above $\lambda$. Clamp $f(D)$ to $[-B, B]$, add floating-point Laplace noise of scale $\lambda$, round to the nearest multiple of $\Lambda$, and clamp again. For $\lambda < B < 2^{46}\lambda$ the result is $\varepsilon$-DP with
$$\varepsilon = \frac{1 + 12B\eta}{\lambda} + \frac{2^{-49}B}{\lambda}, \qquad \eta = 2^{-53},$$
the exact mechanism's $1/\lambda$ plus two terms for rounding error, which grow with the clamping range. For $\lambda = 1$ and $B = 10$ the excess is about $3.1\times10^{-14}$.

**Example.** Snap the toy sampler to integers and clamp to $[-10, 10]$. Both inputs now reach the same 21 outputs at any precision (@fig-discrete-float-support, right), and the largest privacy loss is 1.0010 at 12 bits and 1.0002 at 14 bits, against 1 for the exact mechanism.

**Pitfalls.** Rounding adds up to $\Lambda/2 \le \lambda$ of error, and clamping biases answers near $\pm B$; choose $B$ from the domain, since $\varepsilon$ grows with it. A grid finer than the noise scale keeps bits that depend on the input. The guarantee assumes an ideal uniform and a correct logarithm, which the library must supply. And snapping post-processes a mechanism that isn't DP, so its proof needs Mironov's bookkeeping: don't improvise your own variant.

## The discrete Laplace mechanism {#sec-discrete-laplace}

**Idea.** Most DP queries are counts and histograms: integer-valued, with sensitivity 1. Give them integer noise, which has no low-order bits to leak and can be sampled exactly. The **discrete Laplace** distribution keeps the Laplace density's values at the integers and renormalizes them; adding it is the **geometric mechanism** of Ghosh, Roughgarden and Sundararajan (2009).

**Result.** With scale $b$, each step away from 0 multiplies the probability by $r = e^{-1/b}$, as for the continuous Laplace. The probability mass function (PMF) is
$$\Pr[Z = z] = \frac{1 - r}{1 + r}\,r^{|z|}, \qquad z \in \mathbb Z, \qquad r = e^{-1/b}.$$ {#eq-dlap-pmf}
For an integer-valued query with $\ell_1$ sensitivity $\Delta_1 f$, adding independent $\mathrm{Lap}_{\mathbb Z}(b)$ noise to each coordinate is $\varepsilon$-DP with $\varepsilon = \Delta_1 f/b$: the same calibration as the continuous mechanism, because shifting by an integer only relabels the terms of the normalizing sum. The variance is $2r/(1 - r)^2$, slightly below $2b^2$; the gap tends to $1/6$ as $b$ grows.

**Example.** A count at $\varepsilon = 1$ has $b = 1$ and $r = e^{-1} = 0.3679$. The exact count comes out with probability 0.4621, and the noise variance is 1.8413 against 2 for continuous $\mathrm{Lap}(1)$. The output is an integer, sampled exactly, so the analysis of the mechanism is the analysis of the code.

**Pitfalls.** The query must be integer-valued; a non-integer shift moves the noise off the lattice and brings back the support problem. Rounding $f(D) + \mathrm{Lap}(b)$ to an integer is not a substitute, because it needs a floating-point sampler. For a single count nothing beats it: every user, whatever their loss function and prior, can post-process the geometric mechanism's output to do as well as with any other $\varepsilon$-DP mechanism.

## The discrete Gaussian mechanism {#sec-discrete-gaussian}

**Idea.** The Gaussian mechanism is the workhorse for dense vectors, because its noise follows the $\ell_2$ sensitivity ([The Laplace and Gaussian mechanisms](03-laplace-gaussian.md)). Its integer analogue, from Canonne, Kamath and Steinke (2020), keeps the Gaussian bell curve's values at the integers and renormalizes. Integer shifts leave the normalizer unchanged, so the privacy loss between neighbours is the continuous Gaussian's, evaluated at integers. The US Census Bureau used it for the 2020 Census, and it is the noise of choice before secure aggregation, which works with integers.

**Result.** The **discrete Gaussian** $\mathcal N_{\mathbb Z}(0, \sigma^2)$ gives each integer $x$ probability proportional to $e^{-x^2/(2\sigma^2)}$. Here $\sigma^2$ is the variance of the underlying bell curve, not quite of $X$. Add it to an integer query whose value differs by $\Delta$ between neighbours. The smallest $\delta$ at each $\varepsilon$ has the same form as the continuous Gaussian's analytic curve, with the discrete tails in place of the continuous ones:
$$\delta(\varepsilon) = \Pr\Big[X > \frac{\varepsilon\sigma^2}{\Delta} - \frac\Delta2\Big] - e^\varepsilon\Pr\Big[X > \frac{\varepsilon\sigma^2}{\Delta} + \frac\Delta2\Big], \qquad X \sim \mathcal N_{\mathbb Z}(0, \sigma^2).$$ {#eq-dgauss-delta}
The first term is the probability of the outputs that favour one dataset by more than $e^\varepsilon$; the second subtracts $e^\varepsilon$ times their probability under the neighbour. In Rényi divergence, which [Rényi differential privacy](../part04-privacy-accounting/01-renyi-dp.md) uses for composition, the discrete Gaussian is never worse than the continuous one, so Gaussian accountants apply unchanged. Its variance is slightly below $\sigma^2$.

**Example.** For sensitivity 1 at $\varepsilon = 1$, @eq-dgauss-delta gives $\delta = 0.6032$ against 0.5099 for the continuous Gaussian at $\sigma = 0.5$, 0.1414 against 0.1269 at $\sigma = 1$, and 0.0072 against 0.0068 at $\sigma = 2$ (@sec-discrete-code computes them). The discrete curve has a kink wherever the integer-valued privacy loss crosses $\varepsilon$, so it oscillates around the continuous one, and the gap closes as $\sigma$ grows (@fig-discrete-gaussian-delta). The variance deficit is a relative $2.11\times10^{-7}$ at $\sigma = 1$, but at $\sigma = 0.5$ the variance is $0.8601\,\sigma^2$. A histogram of 1,000 bins with $\mathcal N_{\mathbb Z}(0, 100)$ noise per bin has, under add/remove neighbours, essentially the continuous Gaussian's $\delta(\varepsilon)$ at $\sigma = 10$.

![Discrete against continuous Gaussian for sensitivity 1: $\delta(\varepsilon)$ of the discrete Gaussian (solid) and of the continuous Gaussian (dashed), for $\sigma = 0.5$, 1 and 2. The discrete curve has a kink at each value of its lattice-valued privacy loss, and approaches the continuous one as $\sigma$ grows.](figures/discrete_gaussian_delta.svg){#fig-discrete-gaussian-delta}

**Pitfalls.** Don't reuse the continuous formula for the discrete Gaussian's $\delta$: it is close but can be smaller than the true value, 0.1269 against 0.1414 at $\sigma = 1$, and the gap grows as $\sigma$ shrinks. Compute $\delta(\varepsilon)$ with @eq-dgauss-delta instead, as @sec-discrete-code does. The variance is safe to treat as $\sigma^2$ from $\sigma = 1$ on. A real-valued query, such as a clipped gradient, must be scaled by $1/\gamma$ for a small step $\gamma$ and rounded to integers. Randomized rounding (up with probability equal to the fractional part) keeps it unbiased, but the rounded query can have a slightly larger sensitivity, and that is the one to calibrate to. A sum of discrete Gaussians is not exactly a discrete Gaussian; for $\sigma \ge 1$ it is very close, which schemes where each client adds a share of the noise rely on.

## Sampling exactly, and choosing a library {#sec-discrete-sampler}

**Idea.** A discrete mechanism is only as good as its sampler. Inverting a cumulative distribution function (CDF) with a floating-point uniform brings back every problem of @sec-discrete-float. Canonne, Kamath and Steinke (2020) give samplers that use only integer and rational arithmetic and uniform random integers. They build a coin with probability $e^{-\gamma}$ from coins with rational probabilities, a discrete Laplace sampler from such coins, and a discrete Gaussian sampler that proposes discrete Laplace draws and rejects some with an exact coin. The algorithms are short but subtle; the full edition walks through each.

**Result.** The exact samplers are cheap. A discrete Laplace draw takes about 10 to 12 uniform integers. A discrete Gaussian draw takes fewer than two proposals on average (the acceptance rate is 0.5185 at $\sigma^2 = \frac14$ and 0.7532 at $\sigma^2 = 100$), or 15 to 27 uniform integers. But the number of rejection rounds depends on the output, so the running time leaks information about the noise, and Jin et al. (2022) show such leaks are exploitable.

**Example.** The histogram of 1,000 bins with $\mathcal N_{\mathbb Z}(0, 100)$ noise takes about 15.6 uniform integers per bin, about 15,600 in all, and no floating-point operation.

**Pitfalls.** When you review a DP library or a pipeline, check these points:

- The noise is drawn by an exact integer sampler or a snapping-style repair, never by a floating-point `exp`, `log` or `random()` inside the sampler.
- The random bits come from a cryptographic source (in Python, `secrets.SystemRandom()`), not a seeded generator.
- The running time doesn't depend on the noise drawn: the library pads it or runs a fixed number of rounds.
- The calibration reported matches the mechanism actually used: the discrete Gaussian's $\delta(\varepsilon)$, the rounded query's sensitivity, and the clamping range.

## Code {#sec-discrete-code}

Take the noise itself from a vetted library. What you can check yourself is the calibration: that the $\delta$ reported for a discrete Gaussian matches @eq-dgauss-delta. The snippet sums the PMF over the integers and compares it with the continuous Gaussian's exact curve.

```python
from math import ceil, erfc, exp, sqrt


def discrete_gaussian_delta(eps: float, sigma: float, sens: int = 1) -> float:
    """Exact delta(eps) of the discrete Gaussian mechanism, summed over the integers."""
    reach = ceil(40 * sigma) + sens  # the mass beyond 40 sigma is far below double precision
    weight = {y: exp(-y * y / (2 * sigma**2)) for y in range(-reach - sens, reach + 1)}
    total = sum(weight[y] for y in range(-reach, reach + 1))
    delta = 0.0
    for y in range(-reach, reach + 1):  # outputs of f(D) = 0 and f(D') = sens
        p, q = weight[y] / total, weight[y - sens] / total
        delta += max(p - exp(eps) * q, 0.0)
    return delta


def normal_cdf(x: float) -> float:
    return 0.5 * erfc(-x / sqrt(2))


def gaussian_delta(eps: float, sigma: float, sens: float = 1.0) -> float:
    """Exact delta(eps) of the continuous Gaussian mechanism, for comparison."""
    mu = sens / sigma
    return normal_cdf(mu / 2 - eps / mu) - exp(eps) * normal_cdf(-mu / 2 - eps / mu)


print(f"{discrete_gaussian_delta(1.0, 0.5):.4f} {gaussian_delta(1.0, 0.5):.4f}")  # 0.6032 0.5099
print(f"{discrete_gaussian_delta(1.0, 1.0):.4f} {gaussian_delta(1.0, 1.0):.4f}")  # 0.1414 0.1269
print(f"{discrete_gaussian_delta(1.0, 2.0):.4f} {gaussian_delta(1.0, 2.0):.4f}")  # 0.0072 0.0068
```

Floating point is fine here: this computes a guarantee, not a release. The full edition's script implements the exact samplers with `fractions.Fraction` and integers, checks each against its exact distribution, and enumerates the toy floating-point sampler:

```bash
uv run python code/part03-dp-foundations/discrete_mechanisms.py
```

## Cheat sheet

| Result | Formula | Notes |
| --- | --- | --- |
| Support argument | $\delta \ge \Pr[M(D) \in S]$, $S$ = outputs reachable from $D$ only | Any such output rules out pure DP; the quickest audit |
| Floating-point Laplace | $f(D) + \xi\,b\ln(1/U)$, rounded | Not DP: about 48% of the mass identifies the input in the toy, at every precision |
| Snapping | Clamp, add noise, round to $\Lambda \ge \lambda$, clamp | $\varepsilon = \frac{1 + 12B\eta}{\lambda} + \frac{2^{-49}B}{\lambda}$ for sensitivity 1 |
| Discrete Laplace | $\frac{1 - r}{1 + r}r^{\lvert z\rvert}$, $r = e^{-1/b}$ | $\varepsilon = \Delta_1 f/b$; variance $\frac{2r}{(1 - r)^2}$, just below $2b^2$ |
| Discrete Gaussian | $\propto e^{-x^2/(2\sigma^2)}$ on the integers | $\delta(\varepsilon)$ by @eq-dgauss-delta; same Rényi bound as the continuous Gaussian |
| Exact samplers | Integers, rationals, uniform integers | Fewer than two proposals per discrete Gaussian draw; pad the running time |

: The results of this chapter. {#tbl-discrete-cheat}

**Rules of thumb**

- Never ship `f(D) + laplace(b)` in floating point. Snap and clamp, or make the query integer-valued and use an exact integer sampler.
- Any `exp`, `log` or `random()` inside a sampler is a place where the analysed distribution and the real one can part.
- Always compute the discrete Gaussian's $\delta(\varepsilon)$ with @eq-dgauss-delta; the continuous formula can understate it. Its variance and Rényi bound match the continuous Gaussian's from $\sigma = 1$ on.
- An exact distribution is not enough: running time that depends on the output is a separate side channel.

## Further reading

- Mironov, "On Significance of the Least Significant Bits for Differential Privacy" (CCS 2012): the floating-point attack and the snapping mechanism.
- Canonne, Kamath & Steinke, "The Discrete Gaussian for Differential Privacy" (NeurIPS 2020): the discrete Gaussian's guarantees and the exact samplers; the reference implementation to compare a library against.
- Jin, McMurtry, Rubinstein & Ohrimenko, "Are We There Yet? Timing and Floating-Point Attacks on Differential Privacy Systems" (IEEE S&P 2022): what still goes wrong in modern libraries, including timing.
- Kairouz, Liu & Steinke, "The Distributed Discrete Gaussian Mechanism for Federated Learning with Secure Aggregation" (ICML 2021): the discrete Gaussian when each client adds its own noise.

**Derivations.** Every result in this chapter is derived step by step in [the full edition's chapter](../../book-full/part03-dp-foundations/07-discrete-mechanisms.md).
