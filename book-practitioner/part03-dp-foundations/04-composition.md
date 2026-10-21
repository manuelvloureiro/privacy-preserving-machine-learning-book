# Advanced and optimal composition

## Why this matters

A differentially private system rarely answers just one query. A dashboard refreshes a hundred counts, and a training run takes thousands of noisy gradient steps. Each step on its own satisfies differential privacy (DP), with a privacy level $\varepsilon$ and a small slack $\delta$, as in [The definition of differential privacy](01-definition.md). But what you report, and what you calibrate the noise to, is one guarantee for the whole run.

[Post-processing, group privacy and composition](02-properties.md) gave the first answer, **basic composition**, which adds up the $\varepsilon$ of the steps. A hundred queries at $\varepsilon = 0.1$ then cost 10. Yet if you allow the whole run a slack of $\delta = 10^{-5}$, the same hundred queries are $(4.31, 10^{-5})$-DP. The gap exists because the privacy losses of the steps add up, and each loss is a random amount. Over $k$ steps their total has a mean of only about $k\varepsilon^2/2$ and a spread of about $\sqrt k\,\varepsilon$, so it almost never reaches its maximum $k\varepsilon$.

## Which rule to use {#sec-comp-choose}

@tbl-comp-choose sorts the common situations by what you know about the steps, with the total each rule gives for the hundred queries at $\varepsilon = 0.1$ and an overall $\delta = 10^{-5}$.

| What you know | Use | Total $\varepsilon$ for 100 steps at 0.1 |
| --- | --- | --- |
| Each step's $\varepsilon$, and you need $\delta = 0$ | Basic composition | 10 |
| Each step's $(\varepsilon, \delta)$; you want a quick budget | Advanced composition | 5.30 |
| Each step's $(\varepsilon, \delta)$; you want the number to report | Optimal composition | 4.31 |
| The mechanism itself, such as Gaussian noise | An accountant from Part 4 | Often far smaller |
| Privacy parameters chosen during the run | A privacy filter | — |

: Which composition rule to use. {#tbl-comp-choose}

For few steps, advanced composition can lose to basic composition, so always take the smaller of the two. To an auditor, report the accountant's number when you know the mechanism, and the optimal rule's number otherwise, with the full $\delta$.

## Basic and advanced composition {#sec-comp-advanced}

Basic composition says that $k$ steps that are each $(\varepsilon, \delta)$-DP are together $(k\varepsilon, k\delta)$-DP. It is the only rule that keeps $\delta = 0$.

**Advanced composition** (Dwork, Rothblum & Vadhan 2010) trades a little slack for a much smaller $\varepsilon$. Pick an extra slack $\delta'$ for the whole run. Then $k$ steps that are each $(\varepsilon, \delta)$-DP are together $(\varepsilon', k\delta + \delta')$-DP, with
$$\varepsilon' = \varepsilon\sqrt{2k\ln(1/\delta')} + k\varepsilon\tanh(\varepsilon/2).$$ {#eq-comp-advanced}
The first term is the spread of the total privacy loss, and the second is its average, the **drift**. The original theorem has $e^\varepsilon - 1$ in place of $\tanh(\varepsilon/2)$, which is about twice as large. Use the $\tanh$ form. The original form is useless for large steps. At $\varepsilon = 1$ per step its drift alone is $1.7183k$, more than basic composition's total $k$.

For the hundred queries, the spread term is 4.80 and the drift adds 0.50, which gives 5.30. Advanced composition only helps once $\sqrt k$ outgrows the factor $\sqrt{2\ln(1/\delta')}$. At $\varepsilon = 0.1$, the original form beats basic composition only from $k = 29$ steps on (@tbl-comp-vs-k).

| $k$ | Basic $k\varepsilon$ | Advanced, original form | Advanced, @eq-comp-advanced | Optimal |
| --- | --- | --- | --- | --- |
| 10 | 1.000 | 1.623 | 1.567 | 0.994 |
| 100 | 10.000 | 5.850 | 5.298 | 4.307 |
| 1,000 | 100.000 | 25.691 | 20.170 | 17.787 |
| 10,000 | 1,000.000 | 153.156 | 97.944 | 91.691 |

: Total $\varepsilon'$ at $\delta' = 10^{-5}$ for $k$ mechanisms that are each $0.1$-DP. {#tbl-comp-vs-k}

The most common use runs the theorem backwards. You have a total budget $(\varepsilon_{\text{total}}, \delta')$ and $k$ steps, and you want each step's share. When $\varepsilon\sqrt k$ is small, the drift is negligible next to the spread, so each step can have about
$$\varepsilon = \frac{\varepsilon_{\text{total}}}{\sqrt{2k\ln(1/\delta')}},$$
while basic composition would allow only $\varepsilon_{\text{total}}/k$. Once $\varepsilon\sqrt k$ is near 1 or above, the drift takes over and this shortcut gives too much.

## Optimal composition {#sec-comp-optimal}

Kairouz, Oh & Viswanath (2015) found the exact best rule for steps that are known only by their $(\varepsilon, \delta)$. It is stated through randomized response at level $\varepsilon$, which tells the truth with probability $t = e^\varepsilon/(1 + e^\varepsilon)$ and is the worst case. Write $j$ for the number of lies among $k$ reports, and $(x)_+$ for the larger of $x$ and zero. Then the smallest $\delta$ at a total $\varepsilon'$ is
$$\delta(\varepsilon') = 1 - (1 - \delta)^k\big(1 - \delta_{\mathrm{RR}}(\varepsilon')\big), \qquad \delta_{\mathrm{RR}}(\varepsilon') = \sum_{j=0}^k\binom kj t^{k-j}(1 - t)^j\big(1 - e^{\varepsilon' - (k - 2j)\varepsilon}\big)_+.$$ {#eq-comp-kov}
No rule that sees only each step's $(\varepsilon, \delta)$ can do better.

To use it, evaluate the sum for a candidate $\varepsilon'$ and search for the smallest one whose $\delta$ meets your target. For the hundred queries that gives 4.3068.

Two things go wrong when you compute it. The closed form in the paper divides two huge numbers, and for a thousand steps at $\varepsilon = 1$ it overflows double precision. The fix is to add the terms of @eq-comp-kov through their logarithms, as the code below does. And when the steps have different privacy levels, the exact bound is computationally hard (Murtagh & Vadhan 2016), so use an accountant instead.

The bigger limit is that "optimal" means optimal for the class. A real mechanism has a whole curve of $(\varepsilon, \delta)$ pairs, and summarizing each step by one point forgets the rest of the curve. A thousand Gaussian steps with noise standard deviation 20, on a query of sensitivity 1, are exactly $(7.51, 10^{-5})$-DP. Converting each step to its best single pair and composing optimally gives 62.7, about 8 times worse. So whenever you know the mechanism, compose its whole curve with an accountant from [Privacy accounting](../part04-privacy-accounting/00-introduction.md). In Google's `dp_accounting` library that looks like this:

```text
import dp_accounting

step = dp_accounting.GaussianDpEvent(noise_multiplier=20.0)
accountant = dp_accounting.pld.PLDAccountant()
accountant.compose(dp_accounting.SelfComposedDpEvent(step, 1000))
print(accountant.get_epsilon(target_delta=1e-5))
```

It should print a value close to the exact 7.51. The same library's `dp_accounting.rdp.RdpAccountant` takes the same events, and Microsoft's `prv_accountant` package is a second numerical accountant to check against.

## Adaptive composition {#sec-comp-adaptive}

In practice the next query often depends on the last answer, as when a training loop computes each gradient at parameters set by earlier noisy ones. All the rules above still hold, as long as each step is $(\varepsilon, \delta)$-DP whatever happened before it.

Two kinds of adaptivity break the rules. The first is choosing the privacy parameters during the run, so that each step's $\varepsilon$ or the number of steps depends on earlier answers. A rule like "keep querying until the budget is spent" is of this kind. It needs a **privacy filter**, which is a stopping rule with its own theorem. Reporting the budget spent so far at every moment needs a **privacy odometer** instead. [Adaptive composition](../part04-privacy-accounting/08-adaptive-composition.md) covers both.

The second is letting the raw data steer the run. A stopping rule that reads only the noisy answers is post-processing and costs nothing. But a system that answers private counts until the true count of some group crosses a threshold reveals that count through how many answers it gives. Every answer can then be $\varepsilon$-DP, while the whole run is not DP for any finite $\varepsilon$.

## Code {#sec-comp-code}

The snippet computes advanced and optimal composition, and inverts the optimal rule to give each step its share of a total budget. It prints the two totals quoted above, then plans a dashboard of 100 counts under a total budget of $(1, 10^{-5})$.

```python
from math import exp, expm1, lgamma, log, log1p, sqrt, tanh


def advanced_epsilon(eps: float, k: int, delta_prime: float) -> float:
    """Advanced composition with the sharp drift term k * eps * tanh(eps / 2)."""
    return eps * sqrt(2 * k * log(1 / delta_prime)) + k * eps * tanh(eps / 2)


def optimal_delta(eps_total: float, eps: float, k: int, delta: float = 0.0) -> float:
    """Kairouz-Oh-Viswanath: the exact delta(eps_total) of k (eps, delta)-DP steps, in log space."""
    log_t, log_1mt = -log1p(exp(-eps)), -log1p(exp(eps))  # log of truthful / flipped probability
    logs: list[float] = []
    for j in range(k + 1):  # j of the k reports are flipped
        loss = (k - 2 * j) * eps
        if loss <= eps_total:
            break
        log_binom = lgamma(k + 1) - lgamma(j + 1) - lgamma(k - j + 1)
        logs.append(log_binom + (k - j) * log_t + j * log_1mt + log(-expm1(eps_total - loss)))
    top = max(logs, default=0.0)
    delta_rr = exp(top) * sum(exp(x - top) for x in logs) if logs else 0.0
    return 1 - (1 - delta) ** k * (1 - delta_rr)


def optimal_epsilon(eps: float, k: int, target_delta: float, delta: float = 0.0) -> float:
    """The smallest total epsilon whose optimal delta is at most target_delta, by bisection."""
    lo, hi = 0.0, k * eps
    for _ in range(100):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if optimal_delta(mid, eps, k, delta) > target_delta else (lo, mid)
    return hi


def per_step_epsilon(eps_total: float, k: int, target_delta: float) -> float:
    """The largest per-step epsilon whose k-fold optimal composition fits in (eps_total, target_delta)."""
    lo, hi = eps_total / k, eps_total
    for _ in range(50):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if optimal_epsilon(mid, k, target_delta) <= eps_total else (lo, mid)
    return lo


print(f"{advanced_epsilon(0.1, 100, 1e-5):.4f}")  # 5.2981
print(f"{optimal_epsilon(0.1, 100, 1e-5):.4f}")  # 4.3068
eps0 = per_step_epsilon(1.0, 100, 1e-5)
print(f"{eps0:.4f} {1 / eps0:.1f}")  # 0.0271 37.0
print(f"{1.0 / sqrt(2 * 100 * log(1e5)):.4f}")  # 0.0208
```

In the dashboard line, basic composition gives each count $\varepsilon = 0.01$, so a count of sensitivity 1 gets Laplace noise of scale 100. Optimal composition allows each count 0.0271, which cuts the scale to 37.0, while the rule of thumb from advanced composition, 0.0208, sits in between.

## Cheat sheet

| Rule | Total for $k$ steps that are each $(\varepsilon, \delta)$-DP | When to use it |
| --- | --- | --- |
| Basic | $(k\varepsilon,\ k\delta)$ | Few steps, or $\delta = 0$ required |
| Advanced | $\big(\varepsilon\sqrt{2k\ln(1/\delta')} + k\varepsilon\tanh(\varepsilon/2),\ k\delta + \delta'\big)$ | Quick budgets; per-step share $\approx \varepsilon_{\text{total}}/\sqrt{2k\ln(1/\delta')}$ |
| Optimal | @eq-comp-kov, in log space, inverted by bisection | The number you report, when you know only $(\varepsilon, \delta)$ |
| Accountant | The whole privacy curve of each step | Whenever you know the mechanism |

: The composition rules of this chapter. All of them hold for adaptively chosen queries. {#tbl-comp-cheat}

**Rules of thumb**

- Take the smaller of basic composition and whichever other rule you use, and never report basic composition for thousands of steps.
- Count every step's $\delta$, because the slacks add up to $k\delta$.
- If you know the mechanism, compose its curve, because a single $(\varepsilon, \delta)$ per step can be 8 times too pessimistic.
- Fix the number of steps and their budgets before the run, and never change a step's $\varepsilon$ after seeing results unless a privacy filter covers it.
- Let only released outputs, never the raw data, decide what runs next.

## Further reading

- Dwork & Roth, *The Algorithmic Foundations of Differential Privacy* (2014), §3.5: basic and advanced composition, in the form most libraries cite.
- Kairouz, Oh & Viswanath, "The Composition Theorem for Differential Privacy" (ICML 2015): the worst-case mechanism and the optimal rule.

**Derivations.** Every result here is derived step by step in [the full edition's chapter](../../book-full/part03-dp-foundations/04-composition.md).
