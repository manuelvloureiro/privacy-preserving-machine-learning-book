# Concentrated differential privacy

## Why this matters

You run a pipeline of Gaussian releases, such as noisy counts, answers in a query system or noisy marginals for synthetic data, and you want to keep the differential privacy (DP) budget as one number. [Rényi DP](01-renyi-dp.md) describes each mechanism by its **Rényi differential privacy (RDP) curve** $\varepsilon(\alpha)$: the largest Rényi divergence of order $\alpha > 1$ between the outputs $M(D)$ and $M(D')$ on **neighbouring** datasets, which differ in one person's record. Under composition the curves add. For the Gaussian mechanism with noise multiplier $\sigma$, the noise standard deviation in units of the sensitivity, the curve is a straight line through the origin, $\varepsilon(\alpha) = \alpha/(2\sigma^2)$, and a line through the origin is fixed by its slope.

That slope is **zero-concentrated differential privacy (zCDP)**, the most widely used form of concentrated differential privacy (CDP). The one idea of this chapter: **zCDP is RDP restricted to a line $\rho\alpha$ above the curve.** Lines add, so composition is addition of $\rho$, and the conversion to $(\varepsilon, \delta)$ has a closed form. The line is exact for the Gaussian, loose for pure DP mechanisms with large $\varepsilon$, and useless for the subsampled steps of differentially private stochastic gradient descent (SGD), DP-SGD for short. @tbl-cdp-overview maps the decisions to the sections.

| Decision | Rule | Section |
| --- | --- | --- |
| Charge a Gaussian step | $\rho = \Delta^2/(2\sigma^2)$; the $\rho$'s add | @sec-cdp-definition |
| Charge a Laplace, randomized-response or exponential-mechanism step | $\varepsilon^2/2$, sharply $\varepsilon\tanh(\varepsilon/2)$; $\varepsilon^2/8$ for the exponential mechanism | @sec-cdp-pure |
| Report $(\varepsilon, \delta)$, or calibrate $\sigma$ | Improved conversion; closed-form inverse | @sec-cdp-conversion |
| Protect users who contribute $m$ records | Charge $m^2\rho$ | @sec-cdp-group |
| Account for DP-SGD | Not in zCDP: subsampling is invisible to it. Use RDP | @sec-cdp-tcdp |
| Include steps that are only $(\varepsilon, \delta)$-DP | Approximate zCDP, a second column for $\delta$ | @sec-cdp-approx |

: The decisions behind a zCDP budget, and where this chapter makes each one. {#tbl-cdp-overview}

## The definition of zCDP {#sec-cdp-definition}

**Idea.** Describe a mechanism by the slope of a line through the origin that lies above its whole RDP curve. Composition adds curves, so the budget of a pipeline is a running sum of slopes. In terms of the **privacy loss**, the log of how much more likely an output is under $D$ than under $D'$, zCDP says the loss has tails no heavier than a normal variable with mean $\rho$ and variance $2\rho$. For the Gaussian mechanism the loss is exactly that normal variable, so the line fits perfectly.

**Result.** A mechanism $M$ is **$\rho$-zCDP** if, for all neighbours $D \simeq D'$ and all orders $\alpha > 1$,
$$D_\alpha\big(M(D)\,\|\,M(D')\big) \le \rho\alpha.$$ {#eq-cdp-def}
Three properties make it a ledger. The Gaussian mechanism, which adds $\mathcal N(0, \sigma^2)$ to a query with $\ell_2$ sensitivity $\Delta$, is $\Delta^2/(2\sigma^2)$-zCDP, and with the noise multiplier convention ($\Delta = 1$) it is $1/(2\sigma^2)$-zCDP. Composition adds the parameters, even when each mechanism is chosen after seeing the earlier outputs: $\rho_1$-zCDP followed by $\rho_2$-zCDP is $(\rho_1 + \rho_2)$-zCDP. And post-processing is free.

**Example.** One Gaussian step with $\sigma = 1$ is $0.5$-zCDP, ten of them are $10 \times 0.5 = 5$-zCDP, and one step with $\sigma = 2$ is $1/8 = 0.125$-zCDP. The 2020 United States Census TopDown algorithm is the best-known system built this way (Abowd et al. 2022). It measures many noisy queries with the discrete Gaussian mechanism ([Discrete and floating-point-safe mechanisms](../part03-dp-foundations/07-discrete-mechanisms.md)), adds up their $\rho$'s, and converts to $(\varepsilon, \delta)$ once, at the end.

**Pitfalls.** $\rho$ is not $\varepsilon$. A $\rho$ of 0.5 is about $\varepsilon = 4.7$ at $\delta = 10^{-5}$ (@sec-cdp-conversion), so never compare one system's $\rho$ with another's $\varepsilon$. Halving $\sigma$ multiplies $\rho$ by four. The accounted $\rho$ covers only what was released with noise. The Census Bureau released some statistics, the **invariants**, exactly, outside the accounting, and they sit outside the guarantee.

## Pure DP steps in a zCDP budget {#sec-cdp-pure}

**Idea.** A Laplace count, a randomized-response bit and an exponential-mechanism selection are pure $\varepsilon$-DP: the probability of any output changes by at most a factor $e^\varepsilon$ between neighbours. Their privacy loss is therefore confined to $[-\varepsilon, \varepsilon]$, and a bounded variable has light, normal-like tails, so each such step has a zCDP slope of order $\varepsilon^2$. The RDP curve of a pure DP step is also capped at $\varepsilon$, and a line cannot follow a cap.

**Result.** Every $\varepsilon$-DP mechanism is $(\varepsilon^2/2)$-zCDP (Bun & Steinke 2016), and its RDP curve lies under both the line and the cap:
$$\varepsilon(\alpha) \le \min\Big(\varepsilon,\ \frac{\alpha\varepsilon^2}{2}\Big).$$ {#eq-cdp-pure}
The sharp constant is $\varepsilon\tanh(\varepsilon/2)$, slightly below $\varepsilon^2/2$. Randomized response, which reports a bit truthfully with probability $e^\varepsilon/(1 + e^\varepsilon)$, attains it and is the worst $\varepsilon$-DP mechanism at every order. An $\varepsilon$-DP exponential mechanism is $(\varepsilon^2/8)$-zCDP, four times cheaper, because its privacy loss lies in an interval of length $\varepsilon$ rather than $2\varepsilon$ (Cesar & Rogers 2021).

**Example.** A Laplace count at $\varepsilon = 0.1$ costs $\varepsilon^2/2 = 0.005$. At $\varepsilon = 1$ the sharp slope is $0.4621$ against $\varepsilon^2/2 = 0.5000$. @tbl-cdp-rr compares the exact RDP curve of randomized response at $\varepsilon = 1$ with the line and the capped line. Up to $\alpha = 2$ the line is within 36% of the exact value; at $\alpha = 50$ it is 25 times too large, while the cap is within 1%.

| $\alpha$ | Exact $\varepsilon(\alpha)$ | Line $\alpha\varepsilon^2/2$ | Capped $\min(\varepsilon, \alpha\varepsilon^2/2)$ |
| --- | --- | --- | --- |
| 1.5 | 0.6273 | 0.7500 | 0.7500 |
| 2 | 0.7353 | 1.0000 | 1.0000 |
| 10 | 0.9652 | 5.0000 | 1.0000 |
| 50 | 0.9936 | 25.0000 | 1.0000 |

: The RDP curve of randomized response at $\varepsilon = 1$ against the zCDP line and the line capped at $\varepsilon$. {#tbl-cdp-rr}

![The RDP curve of randomized response (black) against the zCDP line $\alpha\varepsilon^2/2$ (dashed), the sharp line $\alpha\varepsilon\tanh(\varepsilon/2)$ (dash-dotted) and the pure-DP cap $\varepsilon$ (dotted), for $\varepsilon = 1$ (left) and $\varepsilon = 3$ (right). At $\varepsilon = 3$ even the sharp line crosses the cap almost at once.](figures/concentrated_dp_rr_line.svg){#fig-cdp-rr-line}

**Pitfalls.** The translation only pays for small $\varepsilon$. At $\varepsilon = 3$ the slope $\varepsilon^2/2 = 4.5$ already exceeds the cap at $\alpha = 1$, so the line is worse than plain $\varepsilon$ at every order (@fig-cdp-rr-line). A $5$-DP step charged $12.5$ converts to a worse $\varepsilon$ than the 5 you started with. For large $\varepsilon$, as in local DP, where each user randomizes their own data, keep the $\varepsilon$ or the RDP curve; an RDP accountant can charge a pure-DP step the exact curve of randomized response, which lies below both bounds.

## From zCDP to (ε, δ) {#sec-cdp-conversion}

**Idea.** Whatever you account in, you report $(\varepsilon, \delta)$. The RDP conversion of [From Rényi DP to (ε, δ)](02-rdp-conversions.md) gives a valid $\varepsilon$ at every order, and you take the best. For a line the best order can be found by hand; the improved conversion, which modern RDP accountants use, is a one-line numerical minimum.

**Result.** Write $\ell = \ln(1/\delta)$. A $\rho$-zCDP mechanism is $(\varepsilon, \delta)$-DP for every $\delta \in (0, 1)$ with the **simple conversion**
$$\varepsilon = \rho + 2\sqrt{\rho\,\ell},$$ {#eq-cdp-simple}
reached at the order $\alpha^* = 1 + \sqrt{\ell/\rho}$ (Bun & Steinke 2016). The **improved conversion** (Canonne, Kamath & Steinke 2020) is smaller:
$$\varepsilon = \min_{\alpha > 1}\left[\rho\alpha + \frac{\ell}{\alpha - 1} + \ln\Big(1 - \frac1\alpha\Big) - \frac{\ln\alpha}{\alpha - 1}\right].$$ {#eq-cdp-improved}
To calibrate, invert the simple conversion: a target $(\varepsilon, \delta)$ is met by any
$$\rho \le \Big(\sqrt{\varepsilon + \ell} - \sqrt{\ell}\Big)^2,$$ {#eq-cdp-inverse}
and the Gaussian noise multiplier is then $\sigma = 1/\sqrt{2\rho}$.

**Example.** @tbl-cdp-conversion converts a range of $\rho$ at $\delta = 10^{-5}$, where $\ell = 11.5129$. One Gaussian step with $\sigma = 1$ is $\rho = 0.5$: the simple formula gives $0.5 + 2\sqrt{0.5 \times 11.5129} = 5.299$, the improved conversion 4.728, and the exact $\varepsilon$ of that Gaussian mechanism is 4.377. The simple formula is 21% too large, the improved one 8%. To make a single Gaussian step $(1, 10^{-5})$-DP through @eq-cdp-inverse, take $\rho = (\sqrt{12.5129} - \sqrt{11.5129})^2 = 0.02082$, that is $\sigma = 4.901$.

| $\rho$ | Simple @eq-cdp-simple | Improved @eq-cdp-improved | Exact, Gaussian with the same $\rho$ |
| --- | --- | --- | --- |
| 0.01 | 0.689 | 0.546 | 0.497 |
| 0.1 | 2.246 | 1.914 | 1.760 |
| 0.5 | 5.299 | 4.728 | 4.377 |
| 1 | 7.786 | 7.077 | 6.573 |
| 5 | 20.174 | 19.047 | 17.857 |
| 50 | 97.985 | 96.035 | 91.817 |

: $\varepsilon$ at $\delta = 10^{-5}$ for a $\rho$-zCDP mechanism: simple and improved conversions, and the exact $\varepsilon$ of the Gaussian mechanism with the same $\rho$. {#tbl-cdp-conversion}

![Left: $\varepsilon$ at $\delta = 10^{-5}$ for a $\rho$-zCDP mechanism, from the simple formula, the improved conversion, and the exact value for the Gaussian mechanism with $\mu = \sqrt{2\rho}$. Right: each conversion divided by the exact Gaussian value; both gaps shrink in relative terms as $\rho$ grows.](figures/concentrated_dp_conversion.svg){#fig-cdp-conversion}

**Pitfalls.** $\varepsilon$ does not add; $\rho$ does. If your budget is $\rho = 1$ and you spend 0.5 on one query, there is no clean "$\varepsilon$ left": the whole run is $7.077$-DP by the improved conversion however you split it. For a pipeline made only of Gaussian mechanisms the line is exact, so all the slack sits in the conversion; the exact Gaussian column of @tbl-cdp-conversion, which [f-DP and Gaussian differential privacy](04-gaussian-dp.md) computes, is the number to report. Two systems with the same $\rho$ can quote different values of $\varepsilon$, 5.299 and 4.728 at $\rho = 0.5$, only because one uses the simple formula; say which conversion you used.

## Group privacy {#sec-cdp-group}

**Idea.** A household or a user with several records differs from a neighbouring dataset in $m$ records. For the Gaussian, a group moves the mean $m$ times as far, and the RDP grows with the square of the shift. zCDP inherits exactly this.

**Result.** A $\rho$-zCDP mechanism is $m^2\rho$-zCDP for datasets that differ in $m$ records (Bun & Steinke 2016), and the Gaussian mechanism attains it.

**Example.** A user who contributes up to 4 records to a $0.1$-zCDP release is protected at $\rho = 4^2 \times 0.1 = 1.6$. For the Gaussian with $\sigma = 1$, a group of 2 costs $\rho = 2$, or $\varepsilon = 10.725$ at $\delta = 10^{-5}$; a group of 8 costs $\rho = 32$, or $\varepsilon = 68.616$.

**Pitfalls.** With $\rho \approx \varepsilon^2/2$, the group cost $m^2\rho \approx (m\varepsilon)^2/2$ matches pure DP's $m\varepsilon$: a user-level guarantee is much weaker than a record-level one. If users contribute many records, bound each user's contribution before adding noise and account at the user level.

## Subsampling: why DP-SGD is not accounted in zCDP {#sec-cdp-tcdp}

**Idea.** DP-SGD runs each step on a Poisson sample of the data: each record enters the batch with probability $q$. Because a record is usually not in the batch, the step's RDP curve is tiny at small orders. But high orders look at the rare event that the record was sampled, where it pays the full Gaussian loss, so at large orders the curve bends up towards the unsubsampled line $\alpha/(2\sigma^2)$. A line above the whole curve must be that steep: zCDP throws away all of the amplification. **Truncated concentrated differential privacy (tCDP)** asks for the line only on orders up to a cut-off $\omega$.

**Result.** For the Poisson-subsampled Gaussian, at any $q$, the smallest valid $\rho$ is $1/(2\sigma^2)$, the same as without subsampling. A mechanism is $(\rho, \omega)$-tCDP if $D_\alpha \le \rho\alpha$ for all $\alpha \in (1, \omega)$ (Bun, Dwork, Rothblum & Steinke 2018). Steps compose to $(\sum_i\rho_i, \min_i\omega_i)$, and the conversion is the simple one when its best order is below $\omega$, and uses order $\omega$ otherwise:
$$\varepsilon = \begin{cases}\rho + 2\sqrt{\rho\,\ell} & \text{if } \ell \le (\omega - 1)^2\rho,\\[2pt] \rho\omega + \dfrac{\ell}{\omega - 1} & \text{otherwise.}\end{cases}$$ {#eq-cdp-tcdp-conv}

**Example.** Take the part's running example, $q = 0.01$, $\sigma = 1$, $k = 10{,}000$ steps and $\delta = 10^{-5}$. zCDP must charge $\rho = 0.5$ per step, so $\rho = 5000$ in total and $\varepsilon = 5479.9$ by the simple formula, or $5475.8$ with the improved conversion. RDP accounting of the same run, on integer orders, gives $7.469$ with Mironov's conversion and $6.719$ with the improved one. On orders up to $\omega = 8$ the per-step slope is 4476 times smaller than 0.5 (@fig-cdp-subsampled). tCDP with $\omega = 4$ gives $7.469$, the same as RDP with Mironov's conversion. With $\omega = 10$ it gives 80.252, and with $\omega = 16$ it gives 2228.027: once $\omega$ passes the bend near $\alpha = 10$, tCDP collapses towards zCDP.

![The RDP curve of one Poisson-subsampled Gaussian step with $q = 0.01$, $\sigma = 1$, on integer orders (black), against the best zCDP line $\rho\alpha$ with $\rho = 0.5$ (red dashed), which is the unsubsampled Gaussian's line, and the tCDP line on orders up to $\omega = 8$ (blue). Both axes are logarithmic.](figures/concentrated_dp_subsampled.svg){#fig-cdp-subsampled}

**Pitfalls.** Never account a subsampled mechanism in zCDP. If a pipeline mixes Gaussian releases on the full data with a DP-SGD run, keep the whole pipeline in RDP: a $\rho$-zCDP step is just the RDP curve $\rho\alpha$, so it joins an RDP ledger, while a subsampled step cannot join a zCDP one. tCDP's general subsampling theorem is loose (with $\sigma = \sqrt5$ and a 1% sample it gives $\varepsilon = 9.037$ where RDP gives $2.046$) and assumes fixed-size batches, a different sampling scheme ([Amplification by subsampling](05-subsampling.md)). RDP keeps the whole curve and does at least as well as any $\omega$, so use it.

## Approximate zCDP {#sec-cdp-approx}

**Idea.** Some useful mechanisms, such as stability-based histograms, are $(\varepsilon, \delta)$-DP but not zCDP at all. Approximate zCDP lets zCDP fail on an event of probability $\delta_a$, so one ledger holds two columns, $\rho$ and $\delta_a$.

**Result.** An $(\varepsilon, \delta)$-DP step enters as $\rho = \varepsilon^2/2$ and $\delta_a = \delta$. The $\rho$ column adds, and the $\delta_a$ column adds up to at most $\sum_i\delta_i$. At the end the pipeline is $(\varepsilon, \delta_a + (1 - \delta_a)\delta')$-DP, where $\varepsilon$ is any zCDP conversion of $\rho$ at $\delta'$ (Bun & Steinke 2016).

**Example.** Ten stability-based steps, each $(0.5, 10^{-7})$-DP, and ten Gaussian steps with $\sigma = 1$ give $\rho = 10 \times 0.125 + 10 \times 0.5 = 6.25$ and $\delta_a = 1.0000 \times 10^{-6}$, so $\varepsilon = 22.100$ at total $\delta = 10^{-5}$. Composing the stability steps separately, by basic composition, gives $19.119 + 5 = 24.119$.

**Pitfalls.** Use the $\delta_a$ column only for steps that are genuinely not zCDP; a Gaussian step put there wastes $\delta$.

## Code {#sec-cdp-code}

The whole zCDP ledger is a few lines: charge each Gaussian step its $\rho$, add, convert once. The improved conversion is minimized over the same grid of 20,001 orders that the full edition uses.

```python
from math import log, sqrt


def zcdp_gaussian(sigma: float, sensitivity: float = 1.0) -> float:
    """rho of one Gaussian step; compose steps by adding their rho."""
    return sensitivity**2 / (2 * sigma**2)


def zcdp_to_eps_simple(rho: float, delta: float) -> float:
    return rho + 2 * sqrt(rho * log(1 / delta))


def zcdp_to_eps_improved(rho: float, delta: float) -> float:
    """Improved conversion, minimized over 20,001 log-spaced orders in (1, 1 + 1e4]."""
    best = float("inf")
    for i in range(20001):
        alpha = 1 + 10 ** (-4 + 8 * i / 20000)
        eps = rho * alpha + log(1 / delta) / (alpha - 1) + log(1 - 1 / alpha) - log(alpha) / (alpha - 1)
        best = min(best, eps)
    return best


def rho_for_target(eps: float, delta: float) -> float:
    """Largest rho whose simple conversion stays within eps at delta."""
    return (sqrt(eps + log(1 / delta)) - sqrt(log(1 / delta))) ** 2


rho = 10 * zcdp_gaussian(sigma=1.0)  # ten Gaussian steps with sigma = 1
print(f"{rho:.1f}")  # 5.0
print(f"{zcdp_to_eps_simple(rho, 1e-5):.3f}")  # 20.174
print(f"{zcdp_to_eps_improved(rho, 1e-5):.3f}")  # 19.047
r = rho_for_target(1.0, 1e-5)
print(f"{r:.5f} {1 / sqrt(2 * r):.3f}")  # 0.02082 4.901
```

The full edition's script, [concentrated_dp.py](../../code/part04-privacy-accounting/concentrated_dp.py), computes every number in this chapter, including the subsampled curve and the tCDP table:

```bash
uv run python code/part04-privacy-accounting/concentrated_dp.py
```

## Cheat sheet

Here $\ell = \ln(1/\delta)$.

| Result | Formula | Notes |
| --- | --- | --- |
| $\rho$-zCDP | $D_\alpha \le \rho\alpha$ for all $\alpha > 1$ | RDP restricted to a line |
| Gaussian mechanism | $\rho = \Delta^2/(2\sigma^2)$ | Exact: the line is the curve |
| Composition, post-processing | $\rho$'s add; processing keeps $\rho$ | Also for adaptive steps |
| Pure DP step | $\varepsilon^2/2$, sharply $\varepsilon\tanh(\varepsilon/2)$; curve $\le \min(\varepsilon, \alpha\varepsilon^2/2)$ | Poor fit for large $\varepsilon$ |
| Exponential mechanism | $\varepsilon^2/8$ | Bounded range |
| To $(\varepsilon, \delta)$, simple | $\rho + 2\sqrt{\rho\ell}$ | 21% above exact at $\rho = 0.5$ |
| To $(\varepsilon, \delta)$, improved | $\min_\alpha[\rho\alpha + \ell/(\alpha-1) + \ln(1 - 1/\alpha) - \ln\alpha/(\alpha-1)]$ | 8% above exact at $\rho = 0.5$ |
| Calibration | $\rho = (\sqrt{\varepsilon + \ell} - \sqrt\ell)^2$, $\sigma = 1/\sqrt{2\rho}$ | Through the simple conversion |
| Group of $m$ | $m^2\rho$ | Exact for the Gaussian |
| Subsampled Gaussian | $\rho = 1/(2\sigma^2)$ whatever $q$ | No amplification |
| $(\rho, \omega)$-tCDP | Line on $\alpha < \omega$; compose to $(\sum\rho_i, \min\omega_i)$ | Recovers amplification for small $\omega$ |
| Approximate zCDP | $\rho$ adds, $\delta_a$ adds; $(\varepsilon(\rho, \delta'), \delta_a + \delta')$-DP | For steps that are only $(\varepsilon, \delta)$-DP |

: The results of this chapter. {#tbl-cdp-cheat}

**Rules of thumb**

- Keep budgets in $\rho$, which adds, and convert to $(\varepsilon, \delta)$ once, at the end, with the improved conversion.
- One Gaussian step with $\sigma = 1$ costs $\rho = 0.5$, about $\varepsilon = 4.7$ at $\delta = 10^{-5}$.
- Charge a small-$\varepsilon$ pure DP step $\varepsilon^2/2$; for $\varepsilon$ of 2 or more, keep its $\varepsilon$ or its RDP curve.
- Never account a subsampled mechanism, such as a DP-SGD step, in zCDP. Use RDP or a privacy loss distribution.
- A user with $m$ records costs $m^2\rho$.

## Further reading

- Bun & Steinke, "Concentrated Differential Privacy: Simplifications, Extensions, and Lower Bounds" (TCC 2016): the definition, the Gaussian and pure-DP conversions, group privacy and approximate zCDP. Sections 1–3 are the core.
- Canonne, Kamath & Steinke, "The Discrete Gaussian for Differential Privacy" (NeurIPS 2020): the discrete Gaussian with its zCDP guarantee, and the improved conversion with a stable way to compute it.
- Bun, Dwork, Rothblum & Steinke, "Composable and Versatile Privacy via Truncated CDP" (STOC 2018): truncated CDP and its subsampling theorem.
- Abowd et al., "The 2020 Census Disclosure Avoidance System TopDown Algorithm" (Harvard Data Science Review, 2022): zCDP accounting at national scale, and the role of invariants.

**Derivations.** Every result in this chapter is derived step by step in [the full edition's chapter](../../book-full/part04-privacy-accounting/03-concentrated-dp.md).
