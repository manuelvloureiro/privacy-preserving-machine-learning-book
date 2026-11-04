# From Rényi DP to (ε, δ) and to tests

## What it is {#sec-conv-improved}

An accountant that tracks Rényi differential privacy (RDP) returns a curve ([Rényi differential privacy](01-renyi-dp.md)). At each order $\alpha > 1$, it gives a privacy level $\varepsilon(\alpha)$. A report needs a differential privacy (DP) guarantee $(\varepsilon',\delta)$. Conversion maps each point on the curve to such a pair; report the smallest resulting $\varepsilon'$. These conversions are used to report the privacy cost of stochastic gradient descent (SGD).

Two conversions are in use. The original one, **Mironov's conversion** (Mironov 2017), gives at each order
$$\varepsilon' = \varepsilon(\alpha) + \frac{\ln(1/\delta)}{\alpha - 1}.$$
The **improved conversion** (Balle et al. 2020) gives
$$\varepsilon' = \varepsilon(\alpha) + \ln\frac{\alpha - 1}{\alpha} - \frac{\ln\delta + \ln\alpha}{\alpha - 1}.$$
Both conversions are valid. At a fixed order, the improved conversion is always lower: by 0.3612 at order 10, and by more at smaller orders. Report 0 if the improved value is negative.

For example, one Gaussian release with noise multiplier 1 has the curve $\alpha/2$. At $\delta = 10^{-5}$ its exact $\varepsilon$ is 4.377, Mironov's conversion gives 5.299 and the improved one 4.729.

## When you meet it

Use these conversions when comparing RDP results from different sources. Opacus 1.6.0 and `dp_accounting` 0.6.0 use the improved conversion; older papers and code often use Mironov's. In this part's running example, 10,000 steps of private stochastic gradient descent (DP-SGD) each sample 1% of the data and use noise multiplier 1. Mironov's conversion gives 7.43, compared with 6.71 for the improved conversion.

The difference depends only on the order. Two RDP accountants reporting values about 0.36 apart for the same run, with a best order near 10, likely use different conversions. Recompute both with one library before comparing them.

## How to use it

In `dp_accounting`, `get_epsilon_and_optimal_order` converts at the best grid order and returns both values. The same curve supports multiple $\delta$ values, so `get_epsilon` can report several pairs without additional privacy cost. The snippet converts the Gaussian release above.

```python
from dp_accounting import GaussianDpEvent
from dp_accounting.rdp import RdpAccountant

accountant = RdpAccountant()
accountant.compose(GaussianDpEvent(1.0))  # one release, noise multiplier 1
eps, order = accountant.get_epsilon_and_optimal_order(1e-5)
print(f"{eps:.4f} {order:g}")  # 4.7285 5.4
print(f"{accountant.get_epsilon(1e-8):.4f}")  # 6.0864
```

Reducing $\delta$ by a factor of 1,000 raises $\varepsilon$ from 4.7285 to 6.0864 because the conversion depends logarithmically on $\delta$. The best order is selected from a fixed grid. Short, high-noise runs may require orders above 63, beyond Opacus's default grid. If the returned order is at a grid endpoint, extend the grid and convert again ([Rényi differential privacy](01-renyi-dp.md#sec-rdp-orders)).

## What goes wrong

No conversion reaches the exact value because the curve is only a summary. Other mechanisms with the same curve may have a larger exact $\varepsilon$, so the conversion must cover them too. For the plain Gaussian mechanism, the improved conversion is 5–10% above the exact value, and the gap is larger for subsampled DP-SGD steps. A privacy loss distribution (PLD) accountant tracks the full distribution and returns the exact 4.377 for the single release ([Numerical accounting](06-numerical-accounting.md)).

Conversion is one-way: an $(\varepsilon,\delta)$ guarantee with $\delta>0$ does not determine an RDP curve. For example, a mechanism that publishes a record with probability $\delta$ is $(0,\delta)$-DP but has infinite Rényi divergence. Do not enter an $(\varepsilon,\delta)$ pair into an RDP accountant; obtain the curve from the mechanism.

A conversion also loses what the curve says about attacks. Take a membership attack on the single release above, at a false-positive rate of $10^{-3}$. The curve limits the attack's true-positive rate to $2.4910 \times 10^{-2}$. But the pair at $\varepsilon = 4.729$ limits it only to 0.1131 ([f-DP and Gaussian differential privacy](04-gaussian-dp.md)).

## References

- Balle, Barthe, Gaboardi, Hsu & Sato, "Hypothesis Testing Interpretations and Rényi Differential Privacy" (AISTATS 2020): the improved conversion, and what an RDP curve says about tests.

**Derivations.** Every result here is derived step by step in [the full edition's chapter](../../book-full/part04-privacy-accounting/02-rdp-conversions.md).
