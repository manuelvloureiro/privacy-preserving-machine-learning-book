# Concentrated differential privacy

## What it is {#sec-cdp-definition}

Zero-concentrated differential privacy (zCDP) represents a differential privacy (DP) budget by one composable number. It uses the curve of [Rényi differential privacy](01-renyi-dp.md) (RDP), which bounds divergence between outputs on neighbouring datasets at each order $\alpha$, where $\alpha$ denotes the order parameter. A mechanism is **$\rho$-zCDP** when its curve lies below the line $\rho\alpha$. Composition adds these slopes, including for adaptive steps.

The Gaussian mechanism has a linear RDP curve. Let $f$ denote the query. Let $\Delta_2 f$ denote its $\ell_2$ sensitivity, the largest Euclidean change caused by replacing one record. Let $\sigma$ denote the standard deviation of the added noise. A release costs
$$\rho = \frac{(\Delta_2 f)^2}{2\sigma^2}.$$
With noise multiplier 1, $\sigma = \Delta_2 f$ and $\rho = 0.5$. Doubling the noise reduces $\rho$ to one quarter.

## When you meet it

Use zCDP for whole-dataset releases with Gaussian noise, such as noisy counts or the marginals used to generate synthetic data. The 2020 United States Census used zCDP with the discrete Gaussian mechanism ([Discrete and floating-point-safe mechanisms](../part03-dp-foundations/08-discrete-mechanisms.md)). Tumult Analytics and OpenDP also represent budgets with $\rho$. For stochastic gradient descent (SGD) with differential privacy, use RDP or numerical accounting instead.

## How to use it {#sec-cdp-pure}

Track releases in a ledger and charge each according to @tbl-cdp-charge.

| Step | Charge in $\rho$ | Notes |
| --- | --- | --- |
| Gaussian, sensitivity $\Delta_2 f$, noise $\sigma$ | $(\Delta_2 f)^2/(2\sigma^2)$ | Exact |
| Any pure $\varepsilon$-DP step | $\varepsilon^2/2$, or $\varepsilon^2/8$ for an exponential mechanism | Above the plain $\varepsilon$ once $\varepsilon \ge 2$: keep such a step in RDP |
| An $(\varepsilon, \delta)$-DP step | $\varepsilon^2/2$, plus $\delta$ in a second column that also adds | **Approximate zCDP** |
| Poisson-subsampled Gaussian | $(\Delta_2 f)^2/(2\sigma^2)$, as without sampling | No amplification |

: What each kind of step costs in a zCDP ledger. {#tbl-cdp-charge}

Add the charges, then convert the total once. Let $\varepsilon$ denote the resulting privacy level, and let $\delta$ denote the slack. For an all-Gaussian ledger, total $\rho$ is equivalent to one Gaussian release with noise multiplier $1/\sqrt{2\rho}$, so `get_epsilon_gaussian` converts it exactly. The snippet performs this conversion and calibrates a $(1, 10^{-5})$-DP release.

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

For a target privacy level of 1 and slack $10^{-5}$, the required budget is $\rho = 0.03593$. One query meets it with noise multiplier 3.731. If four queries share this budget equally, each receives one quarter and requires twice the noise. Allocate in $\rho$, which composes exactly, not in $\varepsilon$. A histogram with each person in one cell incurs one charge, not one per cell ([Parallel composition](../part03-dp-foundations/02-properties.md#sec-dpprop-parallel)).

For a ledger containing non-Gaussian steps, conversion is not exact. At $\rho = 0.5$, Bun and Steinke's (2016) simple conversion is 21% above the exact Gaussian value. Canonne, Kamath and Steinke's (2020) improved conversion is 8% above. The RDP accountant in `dp_accounting` uses this improved conversion. Use it, or the exact conversion for an all-Gaussian ledger, and identify the method in the report.

## What goes wrong

Do not confuse $\rho$ with $\varepsilon$. A Gaussian release at $\rho = 0.5$ converts to privacy level $\varepsilon = 4.3772$. This conversion uses slack $10^{-5}$; it is not privacy level $\varepsilon = 0.5$. Also, $\rho$ covers only data processed through the private mechanism. The Census Bureau released some counts exactly; list these **invariants** separately because they are outside the guarantee.

Do not account for subsampled steps with zCDP because it does not capture subsampling amplification. In this part's [running example](00-introduction.md), zCDP gives privacy level $\varepsilon = 5475.8$ at slack $\delta = 10^{-5}$ for DP-SGD. RDP gives 6.719 for the same run. Use RDP or a numerical accountant for DP-SGD pipelines. Add a $\rho$-zCDP release without loss as a `GaussianDpEvent` with noise multiplier $1/\sqrt{2\rho}$.

Multiple records from one user increase cost: a group of $m$ records costs $m^2\rho$. Four records in a 0.1-zCDP release therefore cost $\rho = 1.6$.

Budgets do not renew automatically: releases about the same people compose across years. Set the accounting period and total in advance ([Release and governance](../part05-dp-ml/12-release-governance.md#sec-relgov-after)). Track long-term budgets in $\rho$, not diffprivlib's `BudgetAccountant`, which adds $\varepsilon$ by basic composition ([Keeping a budget](../part03-dp-foundations/02-properties.md#sec-dpprop-budget)).

## References

- Bun & Steinke, "Concentrated Differential Privacy: Simplifications, Extensions, and Lower Bounds" (TCC 2016): the definition.
- Canonne, Kamath & Steinke, "The Discrete Gaussian for Differential Privacy" (NeurIPS 2020): the improved conversion.

**Derivations.** Every result here is derived step by step in [the full edition's chapter](../../book-full/part04-privacy-accounting/03-concentrated-dp.md).
