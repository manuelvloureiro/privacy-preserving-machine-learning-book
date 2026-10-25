# From Rényi DP to (ε, δ) and to tests

## What it is {#sec-conv-improved}

An accountant that tracks Rényi differential privacy (RDP) ends a run holding a curve, which gives a privacy cost $\varepsilon(\alpha)$ at each order $\alpha$ above 1 ([Rényi differential privacy](01-renyi-dp.md)). A report needs a differential privacy (DP) guarantee instead, a privacy level $\varepsilon'$ with a small additive slack $\delta$. A conversion turns each point of the curve into such a pair, and you keep the order that gives the smallest $\varepsilon'$.

There are two conversions in use. The original one, **Mironov's conversion** (Mironov 2017), gives at each order
$$\varepsilon' = \varepsilon(\alpha) + \frac{\ln(1/\delta)}{\alpha - 1}.$$
The **improved conversion**, found independently by three groups in 2020, gives
$$\varepsilon' = \varepsilon(\alpha) + \ln\frac{\alpha - 1}{\alpha} - \frac{\ln\delta + \ln\alpha}{\alpha - 1}.$$
Both are valid, but at the same order the improved one is always lower. The saving depends only on the order, 0.3612 at order 10 and more at smaller orders. If the improved value comes out negative, report 0.

For example, take one Gaussian release with noise multiplier 1, whose curve is $\alpha/2$, at $\delta = 10^{-5}$. Mironov's conversion gives 5.299 and the improved one 4.729. The exact value is 4.377.

## When you meet it

You meet the conversion whenever you compare an RDP number with someone else's. Both widely used libraries, Opacus 1.6.0 for PyTorch and `dp_accounting` 0.6.0, use the improved conversion. Older papers and code use Mironov's, including the moments accountant of the original paper on the private form of stochastic gradient descent (SGD), called DP-SGD. This part's running example is a DP-SGD run of 10,000 steps, each on a random 1% of the data with noise multiplier 1. Its curve gives 7.43 by Mironov's conversion and 6.71 by the improved one.

The fixed saving gives a reviewer a quick diagnosis. One of two RDP accountants uses Mironov's conversion if they report values about 0.36 apart for the same run, with a best order near 10. Before comparing such numbers, recompute both with one library, so that the conversion and the grid of orders are the same.

## How to use it

In `dp_accounting`, `get_epsilon_and_optimal_order` converts at the best order of the grid and returns both. One curve answers every $\delta$, so `get_epsilon` can report several pairs at no extra privacy cost. This snippet converts the single Gaussian release above, so you never type the formula in.

```python
from dp_accounting import GaussianDpEvent
from dp_accounting.rdp import RdpAccountant

accountant = RdpAccountant()
accountant.compose(GaussianDpEvent(1.0))  # one release, noise multiplier 1
eps, order = accountant.get_epsilon_and_optimal_order(1e-5)
print(f"{eps:.4f} {order:g}")  # 4.7285 5.4
print(f"{accountant.get_epsilon(1e-8):.4f}")  # 6.0864
```

A $\delta$ a thousand times smaller raises $\varepsilon$ only from 4.7285 to 6.0864, because $\delta$ enters the conversion through a logarithm. The best order depends on the run, and an accountant can only pick it from its fixed list, or grid, of orders. Long runs with little noise want orders close to 1, while short runs with much noise can want orders above 63, where the Opacus default grid stops. So whenever the returned order sits at an end of the grid, extend the grid and convert again ([Rényi differential privacy](01-renyi-dp.md#sec-rdp-orders)).

## What goes wrong

No conversion closes the whole gap to the exact value, because the curve is only a summary, and other mechanisms that satisfy the same curve have a larger exact $\varepsilon$. The improved conversion sits 5 to 10% above the exact $\varepsilon$ for the plain Gaussian, and the gap can be larger for the subsampled steps of DP-SGD. A privacy loss distribution (PLD) accountant keeps the mechanism's whole distribution instead of a summary, and it reaches the exact 4.377 for the single release ([Numerical accounting](06-numerical-accounting.md)).

The conversion also works in one direction only. A guarantee with $\delta$ above 0 gives no RDP curve, because $\delta$ allows a rare output that reveals the record outright. A mechanism that publishes your record with probability $\delta$ is $(0, \delta)$-DP, yet its Rényi divergence is infinite. So never feed an $(\varepsilon, \delta)$ pair into an RDP accountant, and take the curve from the mechanism behind it instead.

Converting also throws away what the curve says about attacks. Take a membership attack on the single release above, run at a false-positive rate of $10^{-3}$. The curve caps its true-positive rate at $2.4910 \times 10^{-2}$, while the pair at $\varepsilon = 4.729$ only caps it at 0.1131 ([f-DP and Gaussian differential privacy](04-gaussian-dp.md)).

## References

- Balle, Barthe, Gaboardi, Hsu & Sato, "Hypothesis Testing Interpretations and Rényi Differential Privacy" (AISTATS 2020): the improved conversion, and what an RDP curve says about tests.

**Derivations.** Every result here is derived step by step in [the full edition's chapter](../../book-full/part04-privacy-accounting/02-rdp-conversions.md).
