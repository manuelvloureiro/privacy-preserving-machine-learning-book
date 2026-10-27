# Concentrated differential privacy

## What it is {#sec-cdp-definition}

Zero-concentrated differential privacy (zCDP) keeps a differential privacy (DP) budget as one number that adds up. It builds on the curve of [Rényi differential privacy](01-renyi-dp.md) (RDP), which bounds how far apart a mechanism's outputs on neighbouring datasets can be at each order $\alpha$. A mechanism is **$\rho$-zCDP** when that curve lies under the straight line $\rho\alpha$, so $\rho$ is a slope. Composition adds the slopes, even when each step is chosen after seeing the earlier outputs.

The Gaussian mechanism's curve is exactly such a line. Write $\sigma$ for the standard deviation of its noise. The query $f$ has an $\ell_2$ sensitivity $\Delta_2 f$, the most one record can move it in Euclidean distance. One release then costs
$$\rho = \frac{(\Delta_2 f)^2}{2\sigma^2}.$$
So a release with noise multiplier 1, where $\sigma = \Delta_2 f$, costs $\rho = 0.5$, and doubling the noise cuts $\rho$ to a quarter.

## When you meet it

You meet zCDP in releases that add Gaussian noise to the whole dataset, such as noisy counts or the marginals behind a synthetic dataset. The 2020 United States Census used it with the discrete Gaussian mechanism ([Discrete and floating-point-safe mechanisms](../part03-dp-foundations/08-discrete-mechanisms.md)), and Tumult Analytics and OpenDP keep budgets in $\rho$.

## How to use it {#sec-cdp-pure}

Keep a ledger, a running list of releases, and charge each its $\rho$ from @tbl-cdp-charge.

| Step | Charge in $\rho$ | Notes |
| --- | --- | --- |
| Gaussian, sensitivity $\Delta_2 f$, noise $\sigma$ | $(\Delta_2 f)^2/(2\sigma^2)$ | Exact |
| Any pure $\varepsilon$-DP step | $\varepsilon^2/2$, or $\varepsilon^2/8$ for an exponential mechanism | Above the plain $\varepsilon$ once $\varepsilon \ge 2$: keep such a step in RDP |
| An $(\varepsilon, \delta)$-DP step | $\varepsilon^2/2$, plus $\delta$ in a second column that also adds | **Approximate zCDP** |
| Poisson-subsampled Gaussian | $(\Delta_2 f)^2/(2\sigma^2)$, as without sampling | No amplification |

: What each kind of step costs in a zCDP ledger. {#tbl-cdp-charge}

Add the charges and convert the total once, at the end, to a privacy level $\varepsilon$ and a small slack $\delta$. Then you pay the conversion's slack only once. An all-Gaussian total $\rho$ costs the same as one Gaussian release with noise multiplier $1/\sqrt{2\rho}$, so `get_epsilon_gaussian` converts it exactly. The snippet does this, then calibrates a $(1, 10^{-5})$-DP release.

```python
from math import sqrt

import dp_accounting

rho = 1 / (2 * 1.0**2)  # one Gaussian release, noise multiplier 1
print(f"{rho:.1f} {dp_accounting.get_epsilon_gaussian(1 / sqrt(2 * rho), 1e-5):.4f}")  # 0.5 4.3772
sigma = dp_accounting.get_sigma_gaussian(1.0, 1e-5)  # one release that must be (1, 1e-5)-DP
budget = 1 / (2 * sigma**2)
quarter = budget / 4  # four releases share it
print(f"{budget:.5f} {sigma:.3f} {1 / sqrt(2 * quarter):.3f}")  # 0.03593 3.731 7.461
```

So the budget for $(1, 10^{-5})$ is $\rho = 0.03593$. A single query meets it with noise multiplier 3.731, so four queries sharing it each get a quarter and need twice the noise. Split the budget in $\rho$, which adds exactly, never in $\varepsilon$. A histogram in which each person sits in one cell is charged once, not once per cell ([Parallel composition](../part03-dp-foundations/02-properties.md#sec-dpprop-parallel)).

A ledger with steps that are not Gaussian has no exact conversion. The simple conversion of Bun and Steinke (2016) gives
$$\varepsilon = \rho + 2\sqrt{\rho\ln(1/\delta)}.$$
It is safe but loose, overshooting the exact Gaussian value by 21% at $\rho = 0.5$. The improved conversion of Canonne, Kamath and Steinke (2020) overshoots by only 8%, and the RDP accountant in `dp_accounting` applies it. So use it, or the exact one for an all-Gaussian ledger, and name it in the report.

## What goes wrong

The commonest error is reading $\rho$ as $\varepsilon$. A Gaussian release at $\rho = 0.5$ is $(4.3772, 10^{-5})$-DP, so reading it as $\varepsilon = 0.5$ badly overrates the protection. And $\rho$ covers only what passed through the noise. The Census Bureau released some counts exactly, and such **invariants** sit outside the guarantee, so list them next to the budget.

Subsampled steps belong outside zCDP, which gives them no amplification. On this part's [running example](00-introduction.md), a run of the private form of stochastic gradient descent (SGD), DP-SGD, zCDP reports $\varepsilon = 5475.8$ at $\delta = 10^{-5}$ where RDP gives 6.719. So keep a pipeline that includes DP-SGD in RDP or a numerical accountant, where a $\rho$-zCDP release joins without loss as a `GaussianDpEvent` with noise multiplier $1/\sqrt{2\rho}$.

Several records of one user cost more, because a group of $m$ records costs $m^2\rho$. Four records in a 0.1-zCDP release cost $\rho = 1.6$.

A budget never renews, because next year's releases about the same people compose with this year's, so fix the period and its total in advance ([Release and governance](../part05-dp-ml/12-release-governance.md#sec-relgov-after)). And keep a long ledger in $\rho$, not in diffprivlib's `BudgetAccountant`, which adds $\varepsilon$ by plain composition ([Keeping a budget](../part03-dp-foundations/02-properties.md#sec-dpprop-budget)).

## References

- Bun & Steinke, "Concentrated Differential Privacy: Simplifications, Extensions, and Lower Bounds" (TCC 2016): the definition.
- Canonne, Kamath & Steinke, "The Discrete Gaussian for Differential Privacy" (NeurIPS 2020): the improved conversion.

**Derivations.** Every result here is derived step by step in [the full edition's chapter](../../book-full/part04-privacy-accounting/03-concentrated-dp.md).
