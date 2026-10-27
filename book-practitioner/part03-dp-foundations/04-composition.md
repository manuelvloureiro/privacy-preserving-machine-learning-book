# Advanced and optimal composition

## What it is {#sec-comp-advanced}

A dashboard refreshes a hundred counts, and a training run takes thousands of noisy steps, yet you report one differential privacy (DP) guarantee for the whole run. Each step has its own privacy level $\varepsilon$ and slack $\delta$ ([The definition of differential privacy](01-definition.md)). **Basic composition** adds them, so a run of $k$ steps costs $k\varepsilon$ and $k\delta$ ([Post-processing, group privacy and composition](02-properties.md)). That is correct but pessimistic. The steps' [privacy losses](01-definition.md#sec-dpdef-divergences) add up, but each depends on the noise drawn and is often negative, so their sum almost never reaches $k\varepsilon$.

**Advanced composition** (Dwork, Rothblum & Vadhan 2010) uses this cancellation. For an extra slack $\delta'$ that you pick for the whole run, the steps are together $(\varepsilon_{\mathrm{total}}, k\delta + \delta')$-DP, with
$$\varepsilon_{\mathrm{total}} = \varepsilon\sqrt{2k\ln(1/\delta')} + k\varepsilon\tanh(\varepsilon/2).$$ {#eq-comp-advanced}
The first term is the spread of the total loss, which grows only like $\sqrt k$. The second is its average, the drift.

**Optimal composition** (Kairouz, Oh & Viswanath 2015) is the exact best bound for steps known only by their $(\varepsilon, \delta)$. Take a hundred counts at $\varepsilon = 0.1$ each, with no slack, on 10,000 people. Basic composition gives a total of 10, still with no slack. The other two rules spend the book's default slack for 10,000 people, $\delta' = 10^{-5}$, and give 5.30 and 4.31.

## When you meet it {#sec-comp-choose}

You meet these rules in papers, privacy reviews and any library that quotes a total $\varepsilon$ for many steps. They decide which number you plan with and which you report (@tbl-comp-choose).

| What you know about the steps | Rule | Total $\varepsilon$ for 100 steps at 0.1 |
| --- | --- | --- |
| Each step's $\varepsilon$, and $\delta$ must stay 0 | Basic composition | 10 |
| Each step's $(\varepsilon, \delta)$, to plan by hand | Advanced composition | 5.30 |
| Each step's $(\varepsilon, \delta)$, for the number you report | Optimal composition | 4.31 |
| The mechanism itself, such as Gaussian noise | An accountant (Part 4), which composes whole privacy curves | Often far smaller |
| Budgets chosen during the run | A privacy filter, which enforces a budget fixed in advance | — |

: Which composition rule to use, at a slack of $10^{-5}$ (none for basic composition). {#tbl-comp-choose}

"Optimal" means optimal for steps described by one pair each. A real mechanism has a whole privacy curve, one $\delta$ for every $\varepsilon$, and a single pair forgets the rest of it. A thousand Gaussian steps with noise standard deviation 20 on a query of sensitivity 1 are exactly $(7.51, 10^{-5})$-DP when composed as curves. Summarizing each step by its best single pair and composing optimally gives 62.7, about eight times worse. So whenever you know the mechanism, compose its curve with an accountant ([Privacy accounting](../part04-privacy-accounting/00-introduction.md)).

## How to use it {#sec-comp-optimal}

Composition does two jobs. The first is planning: to split a total budget $\varepsilon_{\mathrm{total}}$ over $k$ steps, run advanced composition backwards. When $\varepsilon\sqrt k$ is small the second term, the drift, is negligible, so each step can have about
$$\varepsilon \approx \frac{\varepsilon_{\mathrm{total}}}{\sqrt{2k\ln(1/\delta')}}.$$
Once $\varepsilon\sqrt k$ nears 1, the drift takes over and this shortcut gives each step too much.

The second job is reporting. When you know the mechanism, report the accountant's number at the run's total $\delta$, and otherwise report optimal composition. This snippet plans the hundred counts within a total of $(1, 10^{-5})$, giving the whole slack to $\delta'$. It then reports the thousand Gaussian steps with two accountants of Google's `dp_accounting`, based on the privacy loss distribution (PLD) and on Rényi differential privacy (RDP).

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

Each count gets 0.0208, twice basic composition's $1/100$. The PLD accountant matches the exact 7.51 quoted above, and the RDP accountant gives a valid but looser 8.08 ([Choosing an accountant](../part04-privacy-accounting/09-choosing-an-accountant.md)).

## What goes wrong {#sec-comp-adaptive}

For a few steps, advanced composition can lose to basic composition, so always take the smaller of the two. Every rule also adds up the steps' slacks, so count each $\delta$. And when the steps have different privacy levels, optimal composition is computationally hard, one more reason to use an accountant.

All these rules hold when each query depends on earlier answers, provided each step's budget is fixed in advance. Choosing the budgets from the answers needs a privacy filter ([Adaptive composition](../part04-privacy-accounting/08-adaptive-composition.md)), and stopping early needs care. A stopping rule that reads only the noisy answers is post-processing, so the run costs no more than its planned length. But a run that stops when the true count of some group crosses a threshold reveals that count through its length. Every answer can then be $\varepsilon$-DP while the whole run is not DP for any finite $\varepsilon$.

So, for a reviewer, a total below optimal composition is a red flag unless an accountant computed it from the mechanism's curve.

**Derivations.** Every result here is derived step by step in [the full edition's chapter](../../book-full/part03-dp-foundations/04-composition.md).
