# Advanced and optimal composition

## What it is {#sec-comp-advanced}

A dashboard may refresh a hundred counts, and a training run may take thousands of noisy steps. Both require one differential privacy (DP) guarantee for the full release. Each step has privacy level $\varepsilon$ and slack $\delta$ ([The definition of differential privacy](01-definition.md)). **Basic composition** adds these values, so $k$ steps cost $k\varepsilon$ and $k\delta$ ([Post-processing, group privacy and composition](02-properties.md)).

This total is valid but can be loose. Each step's [privacy loss](01-definition.md#sec-dpdef-divergences) depends on the noise and may be negative, so the sum rarely reaches the worst-case value $k\varepsilon$.

**Advanced composition** (Dwork, Rothblum & Vadhan 2010) accounts for these cancellations. With an additional slack $\delta'$ for the run, the steps together satisfy
 $(\varepsilon_{\mathrm{total}}, k\delta + \delta')$-DP, with
$$\varepsilon_{\mathrm{total}} = \varepsilon\sqrt{2k\ln(1/\delta')} + k\varepsilon\tanh(\varepsilon/2).$$ {#eq-comp-advanced}
The first term is the spread of total privacy loss, which grows like $\sqrt k$. The second term is its mean, or drift.

**Optimal composition** (Kairouz, Oh & Viswanath 2015) gives the tightest total guarantee when each step is known only by its pair $(\varepsilon,\delta)$. For 100 counts on 10,000 people, each with $\varepsilon = 0.1$ and no slack, basic composition gives total $\varepsilon=10$. Advanced and optimal composition use slack $\delta' = 10^{-5}$, the [book's default](01-definition.md#sec-dpdef-small-delta) for 10,000 people, and give totals 5.30 and 4.31.

## When you meet it {#sec-comp-choose}

These rules appear in papers, privacy reviews and libraries that report one total $\varepsilon$ for multiple steps (@tbl-comp-choose).

| What you know about the steps | Rule | Total $\varepsilon$ for 100 steps at 0.1 |
| --- | --- | --- |
| Each step's $\varepsilon$, and $\delta$ must stay 0 | Basic composition | 10 |
| Each step's $(\varepsilon, \delta)$, to plan by hand | Advanced composition | 5.30 |
| Each step's $(\varepsilon, \delta)$, for the number you report | Optimal composition | 4.31 |
| The mechanism itself, such as Gaussian noise | An accountant (Part 4), which composes whole privacy curves | Often far smaller |
| Budgets chosen during the run | A privacy filter, which enforces a budget fixed in advance | — |

: Which composition rule to use, at a slack of $10^{-5}$ (none for basic composition). {#tbl-comp-choose}

Optimal composition is tight only when each step is represented by one privacy pair. A mechanism's full privacy curve contains a $\delta$ for each $\varepsilon$, and reducing it to one pair discards information. For example, 1,000 Gaussian steps with noise standard deviation 20 on a query of sensitivity 1 have an exact curve of $(7.51, 10^{-5})$-DP. Composing the best single pair for each step gives 62.7, about eight times larger. When the mechanism is known, compose its curve with an accountant ([Privacy accounting](../part04-privacy-accounting/00-introduction.md)).

## How to use it {#sec-comp-optimal}

To allocate total budget $\varepsilon_{\mathrm{total}}$ across $k$ steps, invert advanced composition. When $\varepsilon\sqrt k$ is small, the drift term in @eq-comp-advanced is negligible, so each step can use approximately
$$\varepsilon \approx \frac{\varepsilon_{\mathrm{total}}}{\sqrt{2k\ln(1/\delta')}}.$$
When $\varepsilon\sqrt k$ approaches 1, the drift is no longer negligible and this approximation overestimates the per-step budget.

For a known mechanism, report the accountant's result at the run's total $\delta$. Otherwise, report optimal composition. The snippet allocates a total budget of $(1, 10^{-5})$ across 100 counts, placing all slack in $\delta'$. It then accounts for 1,000 Gaussian steps with Google's `dp_accounting` using privacy loss distribution (PLD) and Rényi differential privacy (RDP) accountants.

```python
import math

import dp_accounting

print(f"{1.0 / math.sqrt(2 * 100 * math.log(1e5)):.4f}")  # 0.0208
step = dp_accounting.GaussianDpEvent(noise_multiplier=20.0)
run = dp_accounting.SelfComposedDpEvent(step, 1000)
pld = dp_accounting.pld.PLDAccountant()
pld.compose(run)
print(f"{pld.get_epsilon(target_delta=1e-5):.2f}")  # 7.51
rdp = dp_accounting.rdp.RdpAccountant()
rdp.compose(run)
print(f"{rdp.get_epsilon(target_delta=1e-5):.2f}")  # 8.08
```

Each count receives 0.0208, twice the $1/100$ under basic composition. The PLD accountant returns 7.51; the RDP accountant returns the valid but larger value 8.08 ([Choosing an accountant](../part04-privacy-accounting/09-choosing-an-accountant.md)).

## What goes wrong {#sec-comp-adaptive}

For a small number of steps, advanced composition may give a larger bound than basic composition; use the smaller bound. Every rule also composes the per-step slacks, so include each $\delta$.

These rules allow each query to depend on earlier answers, provided each query's budget is fixed before the run. If budgets depend on the answers, use a privacy filter ([Adaptive composition](../part04-privacy-accounting/08-adaptive-composition.md)). Stopping rules also matter. A rule based only on noisy answers is post-processing, so the run costs no more than its planned length. A rule that stops when a true group count exceeds a threshold reveals information through the run length. In that case, each answer may be $\varepsilon$-DP while the full run has no finite DP guarantee.

In a review, a total below the optimal-composition bound is a warning unless an accountant derived it from the mechanism's full privacy curve.


**Derivations.** Every result here is derived step by step in [the full edition's chapter](../../book-full/part03-dp-foundations/04-composition.md).
