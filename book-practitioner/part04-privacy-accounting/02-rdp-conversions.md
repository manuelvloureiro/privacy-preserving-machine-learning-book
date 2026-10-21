# From Rényi DP to (ε, δ) and to tests

## Why this matters

An accountant that tracks Rényi differential privacy (RDP) ends a training run holding a curve, with one value for each **order** $\alpha$ above 1 ([Rényi differential privacy](01-renyi-dp.md)). Nobody reports the curve, though. A paper, a model card or a regulator asks for a differential privacy (DP) guarantee, a privacy level $\varepsilon$ with a small slack $\delta$. So every RDP analysis ends with a conversion, and the conversion decides the number you publish.

The short answer is to use the **improved conversion**, which every maintained library already uses, and to check that the best order is inside the grid. For the Gaussian mechanism it still sits 5 to 10% above the exact $\varepsilon$, and only a different kind of accountant removes that slack.

## What δ means for your users {#sec-conv-delta}

An $(\varepsilon, \delta)$ guarantee says that one person's record changes the chance of any outcome by at most a factor $e^\varepsilon$, except for an extra probability $\delta$. Strictly, $\delta$ is an average over the privacy loss of each output, defined in [Numerical accounting](06-numerical-accounting.md). In practice, treat $\delta$ as the chance that the guarantee fails for a person, possibly badly. Write $n$ for the number of people in the data. Then $\delta$ must be much smaller than $1/n$, which is why [The definition of differential privacy](../part03-dp-foundations/01-definition.md) recommends a value such as $1/n^{1.5}$.

## Mironov's and the improved conversion {#sec-conv-improved}

The original conversion is **Mironov's conversion** (Mironov 2017). For a mechanism with RDP curve $\varepsilon(\alpha)$ and any order $\alpha$, it says that the mechanism is $(\varepsilon', \delta)$-DP with
$$\varepsilon' = \varepsilon(\alpha) + \frac{\ln(1/\delta)}{\alpha - 1}.$$
You minimize it over the grid of orders. The result is valid but loose, because the conversion charges every output whose privacy loss exceeds $\varepsilon'$ with its whole probability, and then bounds that tail with Markov's inequality.

The **improved conversion** recovers most of that loss. Balle and coauthors, Canonne, Kamath and Steinke, and Asoodeh and coauthors found it independently in 2020. A mechanism with RDP curve $\varepsilon(\alpha)$ is $(\varepsilon', \delta)$-DP, at every order, with
$$\varepsilon' = \varepsilon(\alpha) + \ln\frac{\alpha - 1}{\alpha} - \frac{\ln\delta + \ln\alpha}{\alpha - 1}.$$
You again minimize over the grid, and if the result comes out negative, you report 0, since a guarantee at a negative $\varepsilon$ also holds at 0. At any fixed order the improved value is lower by an amount that depends only on the order, 0.3612 at order 10, and more at smaller orders.

For a worked example, release the mean of a bounded statistic once with the Gaussian mechanism at noise multiplier 1, so that its curve is $\alpha/2$. At $\delta = 10^{-5}$, Mironov's conversion gives 5.299. The improved conversion gives 4.729, against an exact value of 4.377 ([f-DP and Gaussian differential privacy](04-gaussian-dp.md)).

This explains why your library's $\varepsilon$ can be lower than an older paper's for the same settings, since older papers and code used Mironov's conversion. Their numbers are loose, not wrong. So if two RDP accountants report values about 0.4 apart for the same run, with a best order near 10, one of them uses Mironov's conversion. On this part's running example, private stochastic gradient descent (SGD), called DP-SGD, Mironov's conversion gives 7.43 and the improved one 6.71.

## Which conversion your library uses {#sec-conv-libraries}

Both widely used libraries use the improved conversion, as of Opacus 1.6.0 and `dp_accounting` 0.6.0. In Opacus, `RDPAccountant` calls the function `get_privacy_spent(orders=..., rdp=..., delta=...)` of `opacus.accountants.analysis.rdp`, and in `dp_accounting`, `RdpAccountant.get_epsilon(target_delta)` calls `compute_epsilon`. But their default grids of orders differ, and the grid matters as much as the formula. The Opacus list `RDPAccountant.DEFAULT_ALPHAS` stops at order 63, while the `DEFAULT_RDP_ORDERS` of `dp_accounting` go on to 1024.

So set the orders yourself when the default does not fit. In Opacus, pass `alphas=[...]` to the accountant's `get_privacy_spent` or `get_epsilon`. In `dp_accounting`, pass `orders=[...]` to the `RdpAccountant` and read the best order from `get_epsilon_and_optimal_order`. Opacus warns when the best order is at an end of its list, but `dp_accounting` does not, so there you compare the returned order with the ends of the list yourself.

One composed curve gives a guarantee at every $\delta$, so you can report several pairs at no extra privacy cost. In `dp_accounting`, call `get_epsilon(target_delta)` once per $\delta$, or `get_delta(target_epsilon)` for the reverse question. For the single release above, the improved conversion gives 4.729 at $\delta = 10^{-5}$ and 6.086 at $10^{-8}$, as the snippet at the end prints.

## Choosing the order {#sec-conv-order}

For a run of $k$ Gaussian steps with noise multiplier $\sigma$, the composed curve is a straight line in the order. Its slope, written $\rho$, is $k/(2\sigma^2)$. Then the improved conversion's best order is close to
$$\alpha^* - 1 \approx \sqrt{\frac{\ln(1/\delta)}{\rho}} = \sigma\sqrt{\frac{2\ln(1/\delta)}{k}}.$$
More steps or less noise make the curve steeper, so they push the best order towards 1. But a smaller $\delta$ pushes it up only through the square root of a logarithm, so $\delta$ matters little. @tbl-conv-orders shows the range a grid must cover.

| $\sigma$ | $k$ | $\rho$ | Best order (continuous / grid) | $\varepsilon$ (continuous / grid) |
| --- | --- | --- | --- | --- |
| 10 | 1 | 0.005 | 40.525 / 41 | 0.375 / 0.375 |
| 1 | 1 | 0.5 | 5.432 / 5.4 | 4.728 / 4.729 |
| 1 | 100 | 50 | 1.472 / 1.5 | 96.035 / 96.116 |
| 1 | 10,000 | 5000 | 1.048 / 1.1 | 5475.791 / 5611.778 |

: Best orders of the improved conversion for the composed Gaussian at $\delta = 10^{-5}$. {#tbl-conv-orders}

Long runs with little noise need orders close to 1, as the last row shows, and short runs with much noise need large orders. Subsampled DP-SGD steps have tiny curves at small orders, so a run with few steps or much noise can want orders above 63, the top of the Opacus default. So whenever the reported best order sits at either end of the grid, extend the grid in that direction and convert again.

## The slack no conversion removes {#sec-conv-slack}

No cleverer formula closes the gap between 4.729 and the exact 4.377 for the single release, because the curve itself lacks the information. The full edition computes the best conversion any method can get from the curve, and it matches the improved one to three decimals at every run length in @tbl-conv-slack.

| $k$ | $\mu$ | Exact | Improved | Mironov |
| --- | --- | --- | --- | --- |
| 1 | 0.100 | 0.341 | 0.375 | 0.485 |
| 100 | 1.000 | 4.377 | 4.729 | 5.299 |
| 10,000 | 10.000 | 91.817 | 96.116 | 98.026 |

: $\varepsilon$ at $\delta = 10^{-5}$ for $k$ Gaussian steps with $\sigma = 10$, where $\mu = \sqrt k/\sigma$. {#tbl-conv-slack}

Every number in the table depends on the run only through $\mu$, so the row with 100 steps is the single release again. The improved conversion exceeds the exact value by 10% for one step and by 5% for 10,000 steps, while Mironov's excess shrinks from 42% to 7%.

That 5 to 10% holds for the plain Gaussian. Subsampled DP-SGD steps have curves that are not straight lines, and the gap can be much larger ([Choosing an accountant](09-choosing-an-accountant.md) measures it for the running example). When the slack matters, change the accountant rather than the formula. A privacy loss distribution (PLD) accountant ([Numerical accounting](06-numerical-accounting.md)) tracks the whole distribution of the privacy loss, and it reaches the exact 4.377 for the single release.

## RDP, attacks and other guarantees {#sec-conv-tests}

An RDP curve also limits attacks. Each order bounds the power of any test that tries to tell whether your record was used. Write $a$ for the test's false-positive rate. An $(\varepsilon, \delta)$ pair only caps the power at $e^\varepsilon a + \delta$, while at low false-positive rates the whole curve caps it far more tightly. So read attack risk from the curve, or better from a trade-off function ([f-DP and Gaussian differential privacy](04-gaussian-dp.md)). Then compare it with what an audit measures ([Auditing private training](../part05-dp-ml/10-auditing.md)).

Mixing RDP with other guarantees works in one direction only. A pure $\varepsilon$-DP component has a curve, which [Concentrated differential privacy](03-concentrated-dp.md#sec-cdp-pure) gives. But an $(\varepsilon, \delta)$ component with $\delta$ above 0 has none, because a mechanism that publishes your record with probability $\delta$ is $(0, \delta)$-DP yet has infinite Rényi divergence. So never feed an $(\varepsilon, \delta)$ pair into an RDP accountant, and take the curve from the mechanism behind it instead.

## Code {#sec-conv-code}

This snippet converts the single release's curve on the part's grid of orders at three values of $\delta$. To convert another mechanism, replace the curve by its RDP values on the same grid.

```python
from math import log

ORDERS = [1 + x / 10 for x in range(1, 100)] + list(range(12, 64)) + [128, 256, 512]


def mironov(rdp: float, alpha: float, delta: float) -> float:
    return rdp + log(1 / delta) / (alpha - 1)


def improved(rdp: float, alpha: float, delta: float) -> float:
    """Balle et al. (2020); Canonne, Kamath & Steinke (2020). Clipped at 0."""
    return max(0.0, rdp + log((alpha - 1) / alpha) - (log(delta) + log(alpha)) / (alpha - 1))


def report(mu: float, delta: float = 1e-5) -> str:
    """Both conversions for a composed Gaussian with mu = sqrt(steps) / sigma."""
    curve = [a * mu**2 / 2 for a in ORDERS]
    mir = min((mironov(e, a, delta), a) for a, e in zip(ORDERS, curve))
    imp = min((improved(e, a, delta), a) for a, e in zip(ORDERS, curve))
    return "Mironov %.3f at order %g, improved %.3f at order %g" % (*mir, *imp)


print(report(1.0))  # Mironov 5.299 at order 5.8, improved 4.729 at order 5.4
print(report(1.0, 1e-6))  # Mironov 5.757 at order 6.3, improved 5.222 at order 5.9
print(report(1.0, 1e-8))  # Mironov 6.570 at order 7.1, improved 6.086 at order 6.7
```

The full edition's script, [rdp_conversions.py](../../code/part04-privacy-accounting/rdp_conversions.py), also computes the exact Gaussian values, the best possible conversion and the limits on tests.

## Cheat sheet

| Result | Formula | Notes |
| --- | --- | --- |
| What $\delta$ means | The chance the guarantee fails for a person | Keep it well below $1/n$ |
| Mironov's conversion | $\varepsilon' = \varepsilon(\alpha) + \ln(1/\delta)/(\alpha - 1)$, best $\alpha$ | Valid but loose; older papers and code |
| Improved conversion | $\varepsilon' = \varepsilon(\alpha) + \ln\frac{\alpha - 1}{\alpha} - \frac{\ln\delta + \ln\alpha}{\alpha - 1}$, best $\alpha$ | Use this; clip at 0; Opacus and `dp_accounting` use it |
| Best order, composed Gaussian | $\alpha^* - 1 \approx \sigma\sqrt{2\ln(1/\delta)/k}$ | Falls towards 1 as $k$ grows |
| Unavoidable slack | 5 to 10% of $\varepsilon$ for the Gaussian | Larger for DP-SGD; use a PLD accountant |
| $(\varepsilon, \delta)$ to RDP | Impossible for $\delta > 0$ | Take the curve from the mechanism |

: The results of this chapter. {#tbl-conv-cheat}

**Rules of thumb**

- Always use the improved conversion, which costs nothing.
- Check the best order. If it sits at either end of the grid, extend the grid and convert again.
- Never compare values of $\varepsilon$ from different conversions or grids as if they were the same quantity.
- Choose $\delta$ well below $1/n$, and report $\varepsilon$ at several values of $\delta$ from the same curve.
- Expect an RDP accountant to overstate the plain Gaussian's $\varepsilon$ by 5 to 10%, and DP-SGD's by more. If that matters, use a PLD accountant.

## Further reading

- Balle, Barthe, Gaboardi, Hsu & Sato, "Hypothesis Testing Interpretations and Rényi Differential Privacy" (AISTATS 2020): the improved conversion, and the region an RDP curve puts on tests.
- Steinke, "Composition of Differential Privacy & Privacy Amplification by Subsampling" (2022 survey): privacy loss, RDP and conversions in one place, with their tightness.

**Derivations.** Every result here is derived step by step in [the full edition's chapter](../../book-full/part04-privacy-accounting/02-rdp-conversions.md).
