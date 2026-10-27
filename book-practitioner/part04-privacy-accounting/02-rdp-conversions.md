# From Rényi DP to (ε, δ) and to tests

## What it is {#sec-conv-improved}

An accountant that tracks Rényi differential privacy (RDP) ends a run with a curve ([Rényi differential privacy](01-renyi-dp.md)). The curve gives a privacy level, written $\varepsilon(\alpha)$, at each order $\alpha$ above 1. A report needs a differential privacy (DP) guarantee, a privacy level $\varepsilon'$ with a small slack $\delta$. A conversion turns each point of the curve into such a pair, and you keep the smallest $\varepsilon'$.

Two conversions are in use. The original one, **Mironov's conversion** (Mironov 2017), gives at each order
$$\varepsilon' = \varepsilon(\alpha) + \frac{\ln(1/\delta)}{\alpha - 1}.$$
The **improved conversion** (Balle et al. 2020) gives
$$\varepsilon' = \varepsilon(\alpha) + \ln\frac{\alpha - 1}{\alpha} - \frac{\ln\delta + \ln\alpha}{\alpha - 1}.$$
Both are valid, but at the same order the improved one is always lower, by 0.3612 at order 10 and more at smaller orders. If the improved value is negative, report 0.

For example, one Gaussian release with noise multiplier 1 has the curve $\alpha/2$. At $\delta = 10^{-5}$ its exact $\varepsilon$ is 4.377, Mironov's conversion gives 5.299 and the improved one 4.729.

## When you meet it

You meet the conversion when you compare RDP numbers from two sources. Opacus 1.6.0 and `dp_accounting` 0.6.0 use the improved one, and older papers and code use Mironov's. This part's running example is 10,000 steps of the private form of stochastic gradient descent (SGD), called DP-SGD, each on a random 1% of the data with noise multiplier 1. There, Mironov's conversion gives 7.43 and the improved one 6.71.

The saving depends only on the order. So two RDP accountants that report values about 0.36 apart for one run, with a best order near 10, use different conversions. Recompute both with one library before you compare them.

## How to use it

In `dp_accounting`, `get_epsilon_and_optimal_order` converts at the best order of the grid and returns both. One curve gives every $\delta$, so `get_epsilon` can report several pairs at no extra privacy cost. This snippet converts the single Gaussian release above.

```python
from dp_accounting import GaussianDpEvent
from dp_accounting.rdp import RdpAccountant

accountant = RdpAccountant()
accountant.compose(GaussianDpEvent(1.0))  # one release, noise multiplier 1
eps, order = accountant.get_epsilon_and_optimal_order(1e-5)
print(f"{eps:.4f} {order:g}")  # 4.7285 5.4
print(f"{accountant.get_epsilon(1e-8):.4f}")  # 6.0864
```

A $\delta$ a thousand times smaller raises $\varepsilon$ only from 4.7285 to 6.0864, because $\delta$ enters the conversion through a logarithm. The best order comes from a fixed grid. Short runs with much noise can need orders above 63, where the Opacus default grid stops. So if the returned order is at an end of the grid, extend the grid and convert again ([Rényi differential privacy](01-renyi-dp.md#sec-rdp-orders)).

## What goes wrong

No conversion closes the whole gap to the exact value, because the curve is only a summary. Other mechanisms with the same curve have a larger exact $\varepsilon$, and the conversion must hold for them too. For the plain Gaussian, the improved conversion is 5 to 10% above the exact value, and more for the subsampled steps of DP-SGD. A privacy loss distribution (PLD) accountant keeps the whole distribution instead, and gets the exact 4.377 for the single release ([Numerical accounting](06-numerical-accounting.md)).

The conversion works in one direction only, because a guarantee with $\delta$ above 0 gives no RDP curve. For example, a mechanism that publishes your record with probability $\delta$ is $(0, \delta)$-DP, but its Rényi divergence is infinite. So never put an $(\varepsilon, \delta)$ pair into an RDP accountant. Take the curve from the mechanism instead.

A conversion also loses what the curve says about attacks. Take a membership attack on the single release above, at a false-positive rate of $10^{-3}$. The curve limits the attack's true-positive rate to $2.4910 \times 10^{-2}$. But the pair at $\varepsilon = 4.729$ limits it only to 0.1131 ([f-DP and Gaussian differential privacy](04-gaussian-dp.md)).

## References

- Balle, Barthe, Gaboardi, Hsu & Sato, "Hypothesis Testing Interpretations and Rényi Differential Privacy" (AISTATS 2020): the improved conversion, and what an RDP curve says about tests.

**Derivations.** Every result here is derived step by step in [the full edition's chapter](../../book-full/part04-privacy-accounting/02-rdp-conversions.md).
