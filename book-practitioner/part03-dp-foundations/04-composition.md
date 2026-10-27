# Advanced and optimal composition

## What it is {#sec-comp-advanced}

A dashboard refreshes a hundred counts, and a training run takes thousands of noisy steps, but you report one differential privacy (DP) guarantee for the whole run. Each step has its own privacy level $\varepsilon$ and slack $\delta$ ([The definition of differential privacy](01-definition.md)). **Basic composition** adds them, so a run of $k$ steps costs $k\varepsilon$ and $k\delta$ ([Post-processing, group privacy and composition](02-properties.md)). That total is correct but too large, because each step's [privacy loss](01-definition.md#sec-dpdef-divergences) depends on the noise and is often negative. So the sum almost never
 reaches $k\varepsilon$.

**Advanced composition** (Dwork, Rothblum & Vadhan 2010) uses these cancellations. With an extra slack $\delta'$ that you pick for the whole run, the steps are together
 $(\varepsilon_{\mathrm{total}}, k\delta + \delta')$-DP, with
$$\varepsilon_{\mathrm{total}} = \varepsilon\sqrt{2k\ln(1/\delta')} + k\varepsilon\tanh(\varepsilon/2).$$ {#eq-comp-advanced}
The first term is the spread of the total loss, which grows only like $\sqrt k$. The second term, the drift, is its average.

**Optimal composition** (Kairouz, Oh & Viswanath 2015) gives the smallest total that is correct when you know each step only by its pair $(\varepsilon, \delta)$. For a hundred counts on 10,000 people, each at $\varepsilon = 0.1$ with no slack, basic composition gives a total of 10. Advanced and optimal composition spend a slack $\delta' = 10^{-5}$, the [book's default](01-definition.md#sec-dpdef-small-delta) for 10,000 people, and give 5.30 and 4.31.

## When you meet it {#sec-comp-choose}

You meet these rules in papers, privacy reviews and libraries that give one total $\varepsilon$ for many steps (@tbl-comp-choose).

| What you know about the steps | Rule | Total $\varepsilon$ for 100 steps at 0.1 |
| --- | --- | --- |
| Each step's $\varepsilon$, and $\delta$ must stay 0 | Basic composition | 10 |
| Each step's $(\varepsilon, \delta)$, to plan by hand | Advanced composition | 5.30 |
| Each step's $(\varepsilon, \delta)$, for the number you report | Optimal composition | 4.31 |
| The mechanism itself, such as Gaussian noise | An accountant (Part 4), which composes whole privacy curves | Often far smaller |
| Budgets chosen during the run | A privacy filter, which enforces a budget fixed in advance | — |

: Which composition rule to use, at a slack of $10^{-5}$ (none for basic composition). {#tbl-comp-choose}

Optimal composition is optimal only for steps that you know by one pair each. A real mechanism has a whole privacy curve, with one $\delta$ for each $\varepsilon$, and one pair loses the rest of it. For example, a thousand Gaussian steps with noise standard deviation 20 on a query of sensitivity 1 are exactly $(7.51, 10^{-5})$-DP as curves. With the best single pair of each step, optimal composition gives 62.7, about eight times worse. So when you know the mechanism, compose its curve with an accountant ([Privacy accounting](../part04-privacy-accounting/00-introduction.md)).

## How to use it {#sec-comp-optimal}

To divide a total budget $\varepsilon_{\mathrm{total}}$ among $k$ steps, use advanced composition in reverse. When $\varepsilon\sqrt k$ is small, the drift term of @eq-comp-advanced is very small, so each step can have about
$$\varepsilon \approx \frac{\varepsilon_{\mathrm{total}}}{\sqrt{2k\ln(1/\delta')}}.$$
When $\varepsilon\sqrt k$ comes near 1, the drift is no longer small, and this shortcut gives each step too much.

To report a run, give the accountant's number at the run's total $\delta$ when you know the mechanism. Otherwise, give the optimal composition. The snippet below plans the hundred counts within a total of $(1, 10^{-5})$, with all of the slack in $\delta'$. Then it reports the thousand Gaussian steps with two accountants of Google's `dp_accounting`, one on the privacy loss distribution (PLD) and one on Rényi differential privacy (RDP).

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

Each count gets 0.0208, twice the $1/100$ of basic composition. The PLD accountant gives the exact 7.51, and the RDP accountant a correct but larger 8.08 ([Choosing an accountant](../part04-privacy-accounting/09-choosing-an-accountant.md)).

## What goes wrong {#sec-comp-adaptive}

For a few steps, advanced composition can give more than basic composition, so always take the smaller of the two. Every rule also adds the slacks of the steps, so count each $\delta$.

All these rules hold when each query depends on earlier answers, but only if you fix the budget of each step before the run. If you choose the budgets from the answers, you need a privacy filter ([Adaptive composition](../part04-privacy-accounting/08-adaptive-composition.md)). Early stops also need care. A stop rule that reads only the noisy answers is post-processing, so the run costs no more than its planned length. But if a run stops when the true count of a group goes above a threshold, its length shows that count. Then each answer is $\varepsilon$-DP, but the whole run is not DP for any finite $\varepsilon$.

So in a review, a total below optimal composition is a red flag, unless an accountant calculated it from the curve of the mechanism.


**Derivations.** Every result here is derived step by step in [the full edition's chapter](../../book-full/part03-dp-foundations/04-composition.md).
