# Rényi differential privacy

## Why this matters

You train a model with 1000 steps of full-batch noisy gradient descent. Each step computes a quantity whose **sensitivity** is 1, meaning one person's record moves it by at most 1, and adds Gaussian noise whose standard deviation is $\sigma = 20$ times that sensitivity; $\sigma$ is the **noise multiplier**. What is the total differential privacy (DP) guarantee at $\delta = 10^{-5}$?

Turn each step into one $(\varepsilon_0, \delta_0)$ pair and apply advanced composition ([Advanced and optimal composition](../part03-dp-foundations/04-composition.md)): with the best split of $\delta$ you get $\varepsilon = 68.01$. The true answer is $\varepsilon = 7.51$, nine times smaller. A Gaussian step has a whole curve of valid $(\varepsilon_0, \delta_0)$ pairs, and collapsing it to one point throws away exactly the information composition needs.

The one idea of this chapter is to **track a curve that adds**. **Rényi differential privacy (RDP)** records, for each **order** $\alpha > 1$, an exponential moment of the privacy loss. These curves add up over the steps of a run, with no $\delta$ spent along the way, and you convert the total to $(\varepsilon, \delta)$ once, at the end. For the 1000 Gaussian steps this gives $\varepsilon = 8.84$, 18% above the exact 7.51. @tbl-rdp-overview maps the decisions to the sections.

| Decision or question | What to know | Section |
| --- | --- | --- |
| What does an RDP guarantee say? | A curve $\varepsilon(\alpha)$ between the average loss and the worst loss | @sec-rdp-definition |
| What is the curve of my mechanism? | Closed forms for Gaussian, Laplace and randomized response | @sec-rdp-curves |
| How do thousands of steps combine? | Add the curves order by order | @sec-rdp-composition |
| Releasing a model; correlated records | Post-processing is free; groups cost order as well as $\varepsilon$ | @sec-rdp-properties |
| Is a "moments accountant" something else? | No, the same numbers in other units | @sec-rdp-moments |
| What $(\varepsilon, \delta)$ do I report? | Convert once, at the best order | @sec-rdp-conversion |

: The decisions behind RDP accounting, and where this chapter covers each. {#tbl-rdp-overview}

## The RDP curve {#sec-rdp-definition}

**Idea.** Run a mechanism $M$ on neighbouring datasets $D$ and $D'$, which differ in one person's record, and get output distributions $P = M(D)$ and $Q = M(D')$. The **privacy loss** of an output $y$ is $L(y) = \ln\frac{p(y)}{q(y)}$, the log of how much more likely $y$ is under $D$ than under $D'$. Pure $\varepsilon$-DP bounds the largest loss. RDP summarizes the loss by an exponential moment instead: the larger the order $\alpha$, the more heavily large losses count, so small orders look at the typical loss and large orders at the worst case.

**Result.** The **Rényi divergence** of order $\alpha$ is the log of the average of $e^{(\alpha - 1)L}$ over outputs drawn from $P$, divided by $\alpha - 1$:
$$D_\alpha(P\|Q) = \frac{1}{\alpha - 1}\ln\mathbb E_P\big[e^{(\alpha - 1)L}\big] = \frac{1}{\alpha - 1}\ln\int p^\alpha q^{1-\alpha}\,dy.$$
A mechanism is **$(\alpha, \varepsilon)$-RDP** if $D_\alpha(M(D)\|M(D')) \le \varepsilon$ for every pair of neighbours. The **RDP curve** $\varepsilon(\alpha)$ is the smallest such $\varepsilon$ at each order, the divergence of the worst pair. It never decreases in $\alpha$. As $\alpha \to 1$ it tends to the average loss, the Kullback–Leibler (KL) divergence; as $\alpha \to \infty$ it tends to the largest loss, so a mechanism is $\varepsilon$-DP exactly when $\varepsilon(\infty) \le \varepsilon$.

**Example.** **Randomized response** reports one person's bit truthfully with probability $p = 0.75$ and flips it otherwise, so every output is 3 times or $1/3$ as likely under one value of the bit as under the other: the mechanism is $\ln 3$-DP. Its RDP curve rises from the average loss, 0.5493 as $\alpha \to 1$, through 0.8473 at order 2 and 1.0666 at order 10, to the worst loss $\ln 3 = 1.0986$.

**Pitfalls.** An RDP guarantee is a curve, not a number. "$(2, 0.8473)$-RDP" is one point of it, and a point at one order says nothing about higher orders: the curve may keep rising there. Accountants store the curve on a grid of orders.

## RDP curves of the basic mechanisms {#sec-rdp-curves}

**Idea.** A mechanism that adds noise to a query's answer produces the same noise distribution centred at two answers. Its curve is the Rényi divergence between the noise and a shifted copy, at the largest shift one record can cause, the sensitivity.

**Result.** With sensitivity 1, the **Gaussian mechanism** with noise multiplier $\sigma$ has a straight-line curve,
$$\varepsilon(\alpha) = \frac{\alpha}{2\sigma^2}.$$ {#eq-rdp-gauss}
For a vector query, $\sigma$ is the noise standard deviation in units of the $\ell_2$ sensitivity. The **Laplace mechanism**, which adds noise of density $\frac{1}{2b}e^{-|y|/b}$ to a real-valued query, has
$$\varepsilon(\alpha) = \frac{1}{\alpha - 1}\ln\left(\frac{\alpha}{2\alpha - 1}e^{(\alpha - 1)/b} + \frac{\alpha - 1}{2\alpha - 1}e^{-\alpha/b}\right),$$
and randomized response with keep-probability $p$ has
$$\varepsilon(\alpha) = \frac{1}{\alpha - 1}\ln\Big(p^\alpha(1-p)^{1-\alpha} + (1-p)^\alpha p^{1-\alpha}\Big).$$
These are exact (Mironov 2017). The Laplace and randomized-response curves level off at their pure-DP levels $1/b$ and $\ln\frac{p}{1-p}$, because their loss is bounded. The Gaussian loss is unbounded, so its curve grows without limit: the Gaussian mechanism is not $\varepsilon$-DP for any $\varepsilon$, but it is RDP at every finite order, and that is all composition and conversion need (@fig-rdp-curves).

![RDP curves of the Gaussian mechanism ($\sigma = 1$), the Laplace mechanism ($b = 1$) and randomized response ($p = 0.75$), all with sensitivity 1, on a log scale in $\alpha$ (so the straight line $\alpha/2$ looks curved). The Laplace and randomized response curves level off at their pure DP levels $1/b$ and $\ln 3$ (dotted); the Gaussian curve keeps growing.](figures/renyi_dp_curves.svg){#fig-rdp-curves}

**Example.** @tbl-rdp-curves reads the three curves at a few orders. The Gaussian row is $\alpha/2$. The Laplace value at order 2 is $\ln\big(\frac23e + \frac13e^{-2}\big) = 0.6191$.

| Mechanism | $\varepsilon(1.5)$ | $\varepsilon(2)$ | $\varepsilon(10)$ | $\varepsilon(64)$ | $\varepsilon(\infty)$ |
| --- | --- | --- | --- | --- | --- |
| Gaussian, $\sigma = 1$ | 0.7500 | 1.0000 | 5.0000 | 32.0000 | $\infty$ |
| Laplace, $b = 1$ | 0.5129 | 0.6191 | 0.9287 | 0.9891 | 1 |
| Randomized response, $p = 0.75$ | 0.7340 | 0.8473 | 1.0666 | 1.0940 | 1.0986 |

: RDP curves of the three basic mechanisms with sensitivity 1. {#tbl-rdp-curves}

**Pitfalls.** The Gaussian formula assumes $\sigma$ is the noise standard deviation divided by the $\ell_2$ sensitivity. In the differentially private version of stochastic gradient descent (SGD), DP-SGD, which clips each example's gradient and adds Gaussian noise, that sensitivity is the clipping norm, so the $\sigma$ you plug in is the noise standard deviation over the clipping norm, not the raw standard deviation. These are the curves of a step that uses the whole dataset. A step on a random subset has a different curve, derived in [Amplification by subsampling](05-subsampling.md).

## Composition adds curves {#sec-rdp-composition}

**Idea.** The order-$\alpha$ moment of independent steps is a product of the steps' moments, and the logarithm turns it into a sum. The bound survives when each step is chosen after seeing earlier outputs, as in training.

**Result.** Running mechanisms with curves $\varepsilon_1, \dots, \varepsilon_k$ in sequence, adaptively, gives
$$\varepsilon_M(\alpha) \le \sum_{i=1}^k \varepsilon_i(\alpha) \qquad\text{for every } \alpha > 1,$$ {#eq-rdp-compose}
with equality for independent steps (Mironov 2017). For Gaussian steps this has a convenient form: $k$ steps with noise multiplier $\sigma$ have the curve $k\alpha/(2\sigma^2)$, which is the curve of a single Gaussian step with noise multiplier $\sigma/\sqrt k$.

**Example.** With $\sigma = 20$ each step has curve $\alpha/800$, and 1000 steps have the curve of one Gaussian with $\sigma = 0.6325$. Different mechanisms add order by order: a Laplace step with $b = 1$ then a Gaussian step with $\sigma = 1$ have $\varepsilon(2) = 0.6191 + 1 = 1.6191$.

Now the part's running example, 10,000 steps of DP-SGD with $\sigma = 1$, but with every step using the whole dataset. The composed curve is $5000\,\alpha$, and even the exact guarantee at $\delta = 10^{-5}$ is $\varepsilon = 5425.5$: no guarantee at all. DP-SGD is private only because each step touches a random 1% of the data, which [Amplification by subsampling](05-subsampling.md) accounts for.

**Pitfalls.** Never convert each step to $(\varepsilon_0, \delta_0)$ and then compose: that is where the factor of nine in the opening example comes from. The steps may be chosen adaptively, but each step's curve bound must be fixed in advance. If the noise level or the number of steps is chosen on the fly from the data, plain composition does not apply; use the filters of [Adaptive composition](08-adaptive-composition.md).

## Post-processing and group privacy {#sec-rdp-properties}

**Idea.** Anything computed from a private output without looking at the data again cannot raise the curve. Groups are a different story: the Rényi divergence has no plain triangle inequality, so going from one record to two costs order as well as $\varepsilon$.

**Result.** Post-processing is free: $\varepsilon_{g\circ M}(\alpha) \le \varepsilon_M(\alpha)$ for any map $g$ that ignores the data. For groups, if $M$ is $(\alpha, \varepsilon)$-RDP with $\alpha \ge 2^{c+1}$, then on datasets that differ in $2^c$ records it satisfies the RDP bound of order $\alpha/2^c$ with $\varepsilon$ replaced by $3^c\varepsilon$ (Mironov 2017, Proposition 2).

**Example.** For the Gaussian mechanism a group of $2^c$ records shifts the answer by $2^c$, so the exact group curve is @eq-rdp-gauss with sensitivity $2^c$. At $\sigma = 1$ and group order 2 the generic bound is 1.5 times too large for two records and 3.375 times for eight. For the Laplace mechanism with $b = 1$ at order 2, two records apart, the exact value is 1.5958, the generic bound 2.4411, and the pure-DP group bound $2/b = 2$.

**Pitfalls.** Use the generic group bound only when you cannot compute the mechanism's curve on far-apart inputs. For a mechanism that is already $\varepsilon$-DP, the pure-DP group bound $k\varepsilon$ for $k$ records holds at every order, and it can beat the RDP route, as the Laplace example shows.

## The moments accountant is RDP {#sec-rdp-moments}

**Idea.** Abadi et al. (2016) trained neural networks with DP by tracking the log of the moment generating function (MGF) of the privacy loss, $K(\lambda) = \ln\mathbb E_P[e^{\lambda L}]$, for the worst pair of neighbours, and called it the **moments accountant**. By the definition of @sec-rdp-definition, that is the RDP curve one order up, in different units.

**Result.**
$$K(\lambda) = \lambda\,\varepsilon(\lambda + 1), \qquad \varepsilon(\alpha) = \frac{K(\alpha - 1)}{\alpha - 1}.$$
Both add over steps, and both convert to $(\varepsilon, \delta)$ by the same tail bound.

**Example.** For randomized response with $p = 0.75$, $K(1) = \ln(0.75 \times 3 + 0.25 \times \frac13) = 0.8473 = 1 \cdot \varepsilon(2)$.

**Pitfalls.** Libraries that say "moments accountant" and libraries that say "RDP accountant" compute the same thing; a difference in their reported $\varepsilon$ comes from the curve of the subsampled step, the order grid or the conversion, not from the bookkeeping.

## From RDP to $(\varepsilon, \delta)$ {#sec-rdp-conversion}

**Idea.** Only outputs whose privacy loss exceeds $\varepsilon$ can break the $e^\varepsilon$ bound, so charge their total probability to $\delta$. An exponential moment, which each point of the RDP curve provides, bounds that tail probability. Every order gives a valid pair; pick the best one after composing.

**Result.** **Mironov's conversion**: an $(\alpha, \varepsilon(\alpha))$-RDP mechanism is $(\varepsilon_\delta, \delta)$-DP for every $0 < \delta < 1$, with
$$\varepsilon_\delta = \varepsilon(\alpha) + \frac{\ln(1/\delta)}{\alpha - 1}.$$ {#eq-rdp-mironov}
The first term grows with the order and the second shrinks, so minimize over the grid of orders. For a Gaussian curve $\alpha/(2\sigma^2)$ the best real order and the result are
$$\alpha^* = 1 + \sigma\sqrt{2\ln(1/\delta)}, \qquad \varepsilon_\delta = \frac{1}{2\sigma^2} + \frac{\sqrt{2\ln(1/\delta)}}{\sigma}.$$ {#eq-rdp-gauss-opt}
For $k$ steps, put $\sigma/\sqrt k$ in place of $\sigma$.

**Example.** One Gaussian step with $\sigma = 1$ at $\delta = 10^{-5}$, where $\ln(1/\delta) = 11.5129$: order 2 gives $1 + 11.5129 = 12.5129$, and the best grid order, 5.8, gives 5.2985. The exact $\varepsilon$ of the Gaussian mechanism is 4.3772, so the conversion is 21% too large; at $\varepsilon = 5.2985$ the true $\delta$ is $1.300 \times 10^{-7}$, 76.9 times smaller than promised. @tbl-rdp-compare repeats the comparison for $k$ steps with $\sigma = 20$, and @fig-rdp-conversion shows the trade-off over orders.

| $k$ | Basic | Advanced | RDP + Mironov (best order) | Exact |
| --- | --- | --- | --- | --- |
| 1 | 0.160 | 0.847 | 0.251 (128) | 0.160 |
| 10 | 1.892 | 3.270 | 0.771 (31) | 0.561 |
| 100 | 21.510 | 13.423 | 2.524 (10.6) | 1.993 |
| 1000 | 238.578 | 68.008 | 8.838 (4) | 7.511 |

: Total $\varepsilon$ at $\delta = 10^{-5}$ for $k$ Gaussian steps with $\sigma = 20$: basic and advanced composition of per-step $(\varepsilon_0, \delta_0)$ points, RDP with Mironov's conversion, and the exact value. {#tbl-rdp-compare}

![Left: converting the curve of 1000 Gaussian steps with $\sigma = 20$ at $\delta = 10^{-5}$. The converted $\varepsilon$ (black) is the RDP curve (dashed) plus the tail term $\ln(1/\delta)/(\alpha - 1)$ (dotted); its minimum over the grid, 8.84 at $\alpha = 4$, sits above the exact 7.51 (green). Right: total $\varepsilon$ against the number of steps, for advanced composition, RDP with Mironov's conversion, and the exact value.](figures/renyi_dp_conversion.svg){#fig-rdp-conversion}

RDP is 18% to 57% above exact here, while advanced composition is 5 to 9 times above. Most of the gain comes from composing curves rather than points.

**Pitfalls.** The best order moves down as the composed curve grows: one step with $\sigma = 20$ wants $\alpha \approx 97$, a thousand steps want $\alpha \approx 4$, and the full-batch run of @sec-rdp-composition wants 1.048, below the grid's smallest order 1.1. So keep a grid from just above 1 to a few hundred, and if the best order sits at an end of the grid, extend it. Mironov's conversion is loose: 1.07 times exact for a composed Gaussian at $\mu = 10$, 1.21 at $\mu = 1$ and 1.76 at $\mu = 0.01$, where $\mu = \sqrt k/\sigma$. Don't report it; use the improved conversion of [From Rényi DP to (ε, δ) and to tests](02-rdp-conversions.md), or a numerical accountant.

## Code {#sec-rdp-code}

RDP accounting for full-batch Gaussian steps is a curve on a grid of orders, a sum over steps and a minimum over orders. This snippet uses the part's grid of 154 orders.

```python
from math import log

ORDERS = [1 + x / 10 for x in range(1, 100)] + list(range(12, 64)) + [128, 256, 512]


def gaussian_rdp(alpha: float, sigma: float, steps: int = 1) -> float:
    """RDP curve of `steps` Gaussian steps with noise multiplier sigma (sensitivity 1)."""
    return steps * alpha / (2 * sigma**2)


def mironov_eps(curve: list[float], delta: float) -> tuple[float, float]:
    """Best (epsilon, order) from Mironov's conversion over the grid."""
    return min((eps + log(1 / delta) / (a - 1), a) for a, eps in zip(ORDERS, curve))


one = [gaussian_rdp(a, sigma=1.0) for a in ORDERS]
run = [gaussian_rdp(a, sigma=20.0, steps=1000) for a in ORDERS]
full_batch = [gaussian_rdp(a, sigma=1.0, steps=10_000) for a in ORDERS]
print("%.4f at order %g" % mironov_eps(one, 1e-5))  # 5.2985 at order 5.8
print("%.3f at order %g" % mironov_eps(run, 1e-5))  # 8.838 at order 4
print("%.1f at order %g" % mironov_eps(full_batch, 1e-5))  # 5615.1 at order 1.1
```

The last line shows the best order stuck at the bottom of the grid. The full edition's script, [renyi_dp.py](../../code/part04-privacy-accounting/renyi_dp.py), computes every number in this chapter, including the exact Gaussian values:

```bash
uv run python code/part04-privacy-accounting/renyi_dp.py
```

## Cheat sheet

| Result | Formula | Notes |
| --- | --- | --- |
| $(\alpha, \varepsilon)$-RDP | $D_\alpha(M(D)\Vert M(D')) \le \varepsilon$ for all neighbours | Non-decreasing in $\alpha$; $\varepsilon(\infty)$ is pure DP |
| Gaussian, noise multiplier $\sigma$ | $\varepsilon(\alpha) = \alpha/(2\sigma^2)$ | Exact; $\sigma$ in units of the $\ell_2$ sensitivity |
| Laplace, scale $b$ | $\frac{1}{\alpha - 1}\ln\big(\frac{\alpha}{2\alpha - 1}e^{(\alpha - 1)/b} + \frac{\alpha - 1}{2\alpha - 1}e^{-\alpha/b}\big)$ | Exact; levels off at $1/b$ |
| Randomized response, keep $p$ | $\frac{1}{\alpha - 1}\ln\big(p^\alpha(1-p)^{1-\alpha} + (1-p)^\alpha p^{1-\alpha}\big)$ | Exact; levels off at $\ln\frac{p}{1-p}$ |
| Composition | $\varepsilon_M(\alpha) \le \sum_i\varepsilon_i(\alpha)$ | Adaptive steps; equality for independent ones |
| $k$ Gaussian steps | One Gaussian with $\sigma/\sqrt k$ | Full-batch only |
| Group of $2^c$ records | Order $\alpha/2^c$, $3^c\varepsilon(\alpha)$, if $\alpha \ge 2^{c+1}$ | Loose; for the Gaussian use sensitivity $2^c$ |
| Moments accountant | $K(\lambda) = \lambda\,\varepsilon(\lambda + 1)$ | Same numbers as RDP |
| Mironov's conversion | $\varepsilon(\alpha) + \ln(1/\delta)/(\alpha - 1)$, best $\alpha$ | 1.07 to 1.76 times exact for a composed Gaussian |
| Gaussian optimum | $\alpha^* = 1 + \sigma\sqrt{2\ln(1/\delta)}$ | $\sigma$ is the composed noise multiplier |

: The results of this chapter. {#tbl-rdp-cheat}

**Rules of thumb**

- Compose in RDP and convert once at the end. Never convert each step to $(\varepsilon_0, \delta_0)$ first.
- $k$ full-batch Gaussian steps with noise multiplier $\sigma$ are one Gaussian step with $\sigma/\sqrt k$.
- Keep an order grid from just above 1 to a few hundred, and check that the best order is not at its edge.
- Report the noise multiplier, the number of steps and the sampling scheme along with $\varepsilon$, so others can recompute it.
- Use the improved conversion of the next chapter, not Mironov's, for the number you publish.

## Further reading

- Mironov, "Rényi Differential Privacy" (IEEE CSF 2017): the definition, composition, the curves of the basic mechanisms, group privacy and the conversion. Short and readable.
- Abadi et al., "Deep Learning with Differential Privacy" (CCS 2016): DP-SGD and the moments accountant, with the experiments that made private deep learning practical.
- Mironov, Talwar & Zhang, "Rényi Differential Privacy of the Sampled Gaussian Mechanism" (arXiv 2019): the curves of subsampled steps, and the reference behind most RDP accountants.

**Derivations.** Every result in this chapter is derived step by step in [the full edition's chapter](../../book-full/part04-privacy-accounting/01-renyi-dp.md).
