# Advanced and optimal composition

## Why this matters

A differential privacy (DP) system rarely answers one query. A dashboard refreshes a hundred counts, an analyst explores a dataset with dozens of queries, and a training run takes thousands of noisy gradient steps. Each step is $(\varepsilon, \delta)$-DP: for any two **neighbouring** datasets $D$ and $D'$, which differ in one person's record, the probability of any set of outputs changes by at most a factor $e^\varepsilon$, plus a small slack $\delta$. You need one guarantee for the whole run, as small as the truth allows, because it sets how much noise each step can afford.

[Post-processing, group privacy and composition](02-properties.md) gave the first answer, **basic composition**: $k$ steps that are each $\varepsilon$-DP are together $k\varepsilon$-DP. It is correct and, for large $k$, far too pessimistic. A hundred counts at $\varepsilon = 0.1$ give $k\varepsilon = 10$, which lets one output multiply an attacker's odds by up to $e^{10} \approx 22{,}000$: nothing worth stating. Yet at $\delta = 10^{-5}$ the same hundred counts are $(4.31, 10^{-5})$-DP, and no analysis that knows only each step's $\varepsilon$ can say less.

The one idea behind the gap: the privacy loss of a composition is a sum of per-step losses, and a sum of many small random terms almost never reaches its maximum. Basic composition charges for the maximum. Every result in this chapter charges for the tail of the sum instead, and pays for the rare outputs beyond it with $\delta$. @tbl-comp-overview maps the decisions to the sections.

| Decision | Rule | Section |
| --- | --- | --- |
| Why $k\varepsilon$ overstates the cost | The total loss is a sum that concentrates around $k\varepsilon^2/2$ | @sec-comp-loss |
| A quick budget for $k$ steps | Advanced composition: $\varepsilon$ grows like $\sqrt k$ | @sec-comp-advanced |
| The best bound from per-step $(\varepsilon, \delta)$ | Optimal composition, computed exactly in log space | @sec-comp-optimal |
| Queries chosen after seeing earlier answers | Covered by every theorem; adaptive privacy parameters are not | @sec-comp-adaptive |

: The decisions behind composing private steps, and where this chapter makes each one. {#tbl-comp-overview}

## The privacy loss of a composition {#sec-comp-loss}

**Idea.** Run a mechanism $M$ on $D$ and look at its output $Y$. The log of how much more likely $Y$ was under $D$ than under $D'$ is the **privacy loss**: positive when $Y$ favours $D$, negative when it favours $D'$. Its distribution is the **privacy loss distribution**. Pure $\varepsilon$-DP says the loss never exceeds $\varepsilon$ in absolute value. When you run $k$ mechanisms in sequence, the probability of the whole output sequence is a product of per-step probabilities, so its log ratio is a sum: the total loss is the sum of the $k$ per-step losses. This holds even when each step is chosen after seeing the earlier outputs.

**Result.** Three facts about this sum carry the whole chapter. First, the smallest $\delta$ that works at a given $\varepsilon$, written $\delta(\varepsilon)$, is an average over outputs of a function of the loss,
$$\delta(\varepsilon) = \mathbb E\big[(1 - e^{\varepsilon - L})_+\big] \le \Pr[L > \varepsilon],$$ {#eq-comp-delta}
where $L$ is the loss, $(x)_+ = \max\{x, 0\}$, and outputs with loss below $\varepsilon$ contribute nothing. So bounding $\delta$ only needs a bound on the tail of the total loss. Second, the loss of an $\varepsilon$-DP step can reach $\varepsilon$, but its mean is at most $\varepsilon\tanh(\varepsilon/2)$, about $\varepsilon^2/2$ for small $\varepsilon$. Third, the worst case is **randomized response**, which reports a person's bit truthfully with probability $t = e^\varepsilon/(1 + e^\varepsilon)$: each report has loss exactly $+\varepsilon$ or $-\varepsilon$. Over $k$ rounds its total loss has mean $k\varepsilon\tanh(\varepsilon/2)$ and standard deviation about $\sqrt k\,\varepsilon$, and it reaches $k\varepsilon$ only when every report is truthful, with probability $t^k$.

**Example.** A dashboard releases 100 counts, each at $\varepsilon = 0.1$. For randomized response, $t = 0.5250$. The total loss has mean 0.4996 and standard deviation 0.9988. Basic composition's $k\varepsilon = 10$ sits 9.5 standard deviations above the mean, reached with probability $t^{100} = 1.0\times10^{-28}$. The optimal $\varepsilon'$ at $\delta' = 10^{-5}$, 4.31, sits far enough into the tail that only a sliver of the distribution lies beyond it (@fig-comp-loss-distribution).

![The distribution of the total privacy loss of 100 rounds of randomized response at $\varepsilon = 0.1$. The vertical lines mark its mean, the $\varepsilon'$ of optimal composition at $\delta' = 10^{-5}$, the $\varepsilon'$ of advanced composition, and basic composition's $k\varepsilon = 10$, which the loss reaches with probability $10^{-28}$.](figures/comp_loss_distribution.svg){#fig-comp-loss-distribution}

**Pitfalls.** The mean of the total loss grows like $k\varepsilon^2/2$ and its spread like $\sqrt k\,\varepsilon$; once $\varepsilon\sqrt k$ is of order 1 or more, the mean matters as much as the spread, and the savings over $k\varepsilon$ shrink. The concentration argument needs $\delta > 0$: with $\delta = 0$ you are stuck with $k\varepsilon$, because the worst output does have positive probability.

## Advanced composition {#sec-comp-advanced}

**Idea.** Dwork, Rothblum & Vadhan (2010) use the concentration of the total loss. The total is a drift, the $k$ per-step means added up, plus a fluctuation around it. The fluctuation of $k$ terms each in $[-\varepsilon, \varepsilon]$ is of order $\varepsilon\sqrt k$, and Azuma's inequality, a version of Hoeffding's inequality for terms that may depend on earlier terms, says how far above the drift it goes with probability $\delta'$. Adding the drift gives the bound.

**Result.** For every $\delta' > 0$, $k$ steps that are each $(\varepsilon, \delta)$-DP, chosen adaptively, are together $(\varepsilon', k\delta + \delta')$-DP with
$$\varepsilon' = \varepsilon\sqrt{2k\ln(1/\delta')} + k\varepsilon\tanh(\varepsilon/2).$$ {#eq-comp-advanced}
The first term is the fluctuation and the second the drift. The original theorem has $k\varepsilon(e^\varepsilon - 1)$ as the drift, about twice as large for small $\varepsilon$; both are valid. The bound beats basic composition only once $\sqrt k$ outgrows the factor $\sqrt{2\ln(1/\delta')}$, which is 4.80 at $\delta' = 10^{-5}$: at $\varepsilon = 0.1$, from $k = 29$ on. For budgeting, invert the leading term: to be $(\varepsilon_{\text{total}}, \delta')$-DP after $k$ steps, give each step roughly $\varepsilon = \varepsilon_{\text{total}}/\sqrt{2k\ln(1/\delta')}$, much more than the $\varepsilon_{\text{total}}/k$ basic composition allows.

**Example.** For the 100 counts at $\varepsilon = 0.1$ and $\delta' = 10^{-5}$, the fluctuation term is 4.7985 and the drift $100 \times 0.1\tanh(0.05) = 0.4996$, so $\varepsilon' = 5.2981$; with the original drift, 5.8502. @tbl-comp-vs-k compares the bounds over $k$, and @fig-comp-epsilon-vs-k plots them.

| $k$ | Basic $k\varepsilon$ | Advanced, $e^\varepsilon - 1$ drift | Advanced, @eq-comp-advanced | Optimal |
| --- | --- | --- | --- | --- |
| 10 | 1.000 | 1.623 | 1.567 | 0.994 |
| 100 | 10.000 | 5.850 | 5.298 | 4.307 |
| 1,000 | 100.000 | 25.691 | 20.170 | 17.787 |
| 10,000 | 1,000.000 | 153.156 | 97.944 | 91.691 |

: Total $\varepsilon'$ at $\delta' = 10^{-5}$ for $k$ mechanisms that are each $0.1$-DP. {#tbl-comp-vs-k}

![Total $\varepsilon'$ at $\delta = 10^{-5}$ after $k$ compositions of $\varepsilon$-DP mechanisms, for $\varepsilon = 0.1$ (left) and $\varepsilon = 0.5$ (right), on log axes. Basic composition grows like $k$; advanced and optimal composition grow like $\sqrt k$ until the drift term takes over. The optimal curve is never above any other.](figures/comp_epsilon_vs_k.svg){#fig-comp-epsilon-vs-k}

**Pitfalls.** At small $k$ advanced composition is worse than basic composition (1.567 against 1 at $k = 10$), so always take the smaller of the two. At large $k$ the drift dominates: at $k = 10{,}000$ the original drift term costs 105 of the 153, which is why the $\tanh$ form matters. And the bound is conservative: at $k = 100$, $\varepsilon = 0.1$, the exact $\delta$ at the advanced $\varepsilon' = 5.85$ is $1.96\times10^{-9}$, not $10^{-5}$. Use it as a quick estimate, and the next section's bound for the number you report.

## Optimal composition {#sec-comp-optimal}

**Idea.** Kairouz, Oh & Viswanath (2015) found the best possible bound for $k$ steps known only by their $(\varepsilon, \delta)$. Every $(\varepsilon, \delta)$-DP mechanism can be obtained from one particular mechanism by **post-processing**, further randomized steps that do not look at the data, and post-processing can never increase $\delta(\varepsilon')$. That mechanism is a four-output randomized response: with probability $\delta$ it announces which dataset it ran on, and otherwise it runs randomized response at level $\varepsilon$. So the worst composition of $k$ steps is $k$ copies of it, whose loss you already know.

**Result.** Either some step announces the dataset, which happens with probability $1 - (1 - \delta)^k$ and counts in full, or the run is $k$ rounds of plain randomized response. Then exactly $j$ of the $k$ reports lie with binomial probability, and the loss is $(k - 2j)\varepsilon$. Plugging that loss into @eq-comp-delta gives, for every composition of $k$ steps that are each $(\varepsilon, \delta)$-DP, adaptive or not,
$$\delta(\varepsilon') \le 1 - (1 - \delta)^k\big(1 - \delta_{\mathrm{RR}}(\varepsilon')\big), \qquad \delta_{\mathrm{RR}}(\varepsilon') = \sum_{j=0}^k\binom kj t^{k-j}(1 - t)^j\big(1 - e^{\varepsilon' - (k - 2j)\varepsilon}\big)_+,$$ {#eq-comp-kov}
with $t = e^\varepsilon/(1 + e^\varepsilon)$. The bound is attained, so no better one holds for the whole class. To get the total $\varepsilon'$ for a target $\delta$, solve @eq-comp-kov for $\varepsilon'$ by bisection; $\delta(\varepsilon')$ falls as $\varepsilon'$ grows. It costs one pass over $j$ per evaluation (@sec-comp-code).

**Example.** For the 100 counts at $\varepsilon = 0.1$, the optimal $\varepsilon'$ at $\delta' = 10^{-5}$ is 4.3068: 19% below the $\tanh$ advanced bound, 26% below the original one and 57% below basic composition. If each step also has $\delta = 5\times10^{-8}$, the announcements use up $5.0\times10^{-6}$ of a total $\delta = 10^{-5}$, and the optimal $\varepsilon'$ is 4.4469, against 5.9926 for advanced composition with the remaining $5\times10^{-6}$.

**Pitfalls.** Evaluate the bound in log space. Kairouz, Oh & Viswanath's closed form divides numbers that overflow: at $\varepsilon = 1$ and $k = 1{,}000$ it raises `OverflowError`, while the log-space sum gives $\delta(550) = 6.094\times10^{-4}$. "Optimal" means optimal for mechanisms you know only by $(\varepsilon, \delta)$; with different $\varepsilon_i$ per step, computing it exactly is #P-complete, the class of hard counting problems (Murtagh & Vadhan 2016), so use an accountant instead. The bigger trap is summarizing a mechanism by one point at all. A Gaussian mechanism has a whole curve $\delta(\varepsilon)$, and $k$ Gaussian steps with noise $\sigma$ compose exactly into one Gaussian step with shift $\sqrt k/\sigma$. With $\sigma = 20$ on a sensitivity-1 query, 1,000 steps are exactly $(7.51, 10^{-5})$-DP, but converting each step to its best $(\varepsilon, \delta)$ point and composing optimally gives 62.7, a factor of about 8 (@fig-comp-gaussian). If you know the mechanism, compose its curve with the accountants of [Privacy accounting](../part04-privacy-accounting/00-introduction.md).

![Total $\varepsilon$ at $\delta = 10^{-5}$ after $k$ Gaussian mechanisms with $\sigma = 20$ on a query of sensitivity 1. The first three curves convert each step to one $(\varepsilon, \delta)$ point, with the best split of $\delta$, then compose with basic, advanced or optimal composition. The dashed curve composes the exact Gaussian curves, a single Gaussian with shift $\sqrt k/\sigma$, and lies far below.](figures/comp_gaussian.svg){#fig-comp-gaussian}

## Adaptive composition {#sec-comp-adaptive}

**Idea.** In practice the next query depends on the last answer. An analyst zooms into the largest bin of a noisy histogram; a training loop takes its next gradient at parameters set by earlier noisy gradients. The proofs above look at one step at a time given the history so far, and once the history is fixed, so is the choice of step. So adaptivity costs nothing extra.

**Result.** In the **adaptive composition experiment** of Dwork, Rothblum & Vadhan, an adversary picks each mechanism from a class of $(\varepsilon, \delta)$-DP mechanisms as any function of the earlier outputs. Basic composition, @eq-comp-advanced and @eq-comp-kov all hold for the resulting view, for every adversary. The condition is that $k$, $\varepsilon$ and $\delta$ are fixed before the interaction starts; only the choice of mechanisms adapts.

**Example.** An analyst runs 100 steps at $\varepsilon = 0.1$, choosing a Laplace query after any output that pointed to $D$ and a randomized-response query otherwise. In 100,000 simulated runs its $\delta$ at $\varepsilon' = 2$ is 0.0196, below the optimal bound of 0.0201, as the theorem requires (@fig-comp-delta-curves).

![The privacy curve $\delta(\varepsilon')$ of 100 steps at $\varepsilon = 0.1$: the exact curve of composed randomized response (optimal composition), simulated curves for composed Laplace mechanisms and for the adaptive analyst, the advanced-composition pairs $(\varepsilon', \delta')$ for several $\delta'$, and basic composition at $(10, 0)$. The three compositions nearly coincide; advanced composition lies far to the right.](figures/comp_delta_curves.svg){#fig-comp-delta-curves}

**Pitfalls.** Adaptivity does not let you choose privacy parameters on the fly. A rule such as "keep querying until the budget is spent", with each $\varepsilon_i$ picked after seeing earlier outputs, falls outside these theorems and needs a **privacy filter**; reporting the spend so far at every moment needs a **privacy odometer**, which costs more ([Adaptive composition](../part04-privacy-accounting/08-adaptive-composition.md)). For pure DP under basic composition the naive sum $\sum_i\varepsilon_i$ is still fine. Nor may a choice depend on the raw data. A stopping rule that looks only at the noisy answers is post-processing and free. A system that answers private counts until the *true* count of some group crosses a threshold reveals that count through the number of answers it gives, and the whole run is not DP for any finite $\varepsilon$, even though every answer is.

## Code {#sec-comp-code}

The optimal bound needs only a binomial sum, evaluated in log space so that nothing overflows, and a bisection. This is the whole of it, with advanced composition alongside:

```python
from math import exp, expm1, lgamma, log, log1p, sqrt, tanh


def advanced_epsilon(eps: float, k: int, delta_prime: float) -> float:
    """Advanced composition with the sharp drift term k * eps * tanh(eps / 2)."""
    return eps * sqrt(2 * k * log(1 / delta_prime)) + k * eps * tanh(eps / 2)


def optimal_delta(eps_total: float, eps: float, k: int, delta: float = 0.0) -> float:
    """Kairouz-Oh-Viswanath: the exact delta(eps_total) of k (eps, delta)-DP steps, in log space."""
    log_t, log_1mt = -log1p(exp(-eps)), -log1p(exp(eps))  # log of truthful / lying probability
    logs: list[float] = []
    for j in range(k + 1):  # j of the k reports lie
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


print(f"{advanced_epsilon(0.1, 100, 1e-5):.4f}")  # 5.2981
print(f"{optimal_epsilon(0.1, 100, 1e-5):.4f}")  # 4.3068
print(f"{optimal_epsilon(0.1, 100, 1e-5, delta=5e-8):.4f}")  # 4.4469
print(f"{optimal_epsilon(0.1, 1000, 1e-5):.3f}")  # 17.787
print(f"{optimal_delta(550, 1.0, 1000):.3e}")  # 6.094e-04
```

It reproduces the numbers of @sec-comp-advanced and @sec-comp-optimal, including the case where the closed form overflows. The full edition's script, [composition.py](../../code/part03-dp-foundations/composition.py), checks the bound against brute-force enumeration and simulates the Laplace and adaptive compositions:

```bash
uv run python code/part03-dp-foundations/composition.py
```

## Cheat sheet

| Result | Formula | Notes |
| --- | --- | --- |
| Privacy loss of a composition | $L = L_1 + \dots + L_k$ | Adaptive or not |
| $\delta$ from the loss | $\delta(\varepsilon) = \mathbb E[(1 - e^{\varepsilon - L})_+] \le \Pr[L > \varepsilon]$ | Only the tail of $L$ matters |
| Mean loss of an $\varepsilon$-DP step | $\le \varepsilon\tanh(\varepsilon/2) \approx \varepsilon^2/2$ | Equality for randomized response |
| Basic composition | $(k\varepsilon, k\delta)$ | Adaptive; parameters may adapt for pure DP |
| Advanced composition | $\big(\varepsilon\sqrt{2k\ln(1/\delta')} + k\varepsilon\tanh(\varepsilon/2),\ k\delta + \delta'\big)$ | Beats basic from $k = 29$ at $\varepsilon = 0.1$, $\delta' = 10^{-5}$ |
| Optimal composition | $1 - (1 - \delta)^k(1 - \delta_{\mathrm{RR}}(\varepsilon'))$, @eq-comp-kov | Tight for mechanisms known only by $(\varepsilon, \delta)$; compute in log space |
| Known mechanism, such as Gaussian | Compose the whole curve | Point summaries can be off by a factor of 8 |

: The composition results of this chapter, for $k$ steps that are each $(\varepsilon, \delta)$-DP. {#tbl-comp-cheat}

**Rules of thumb**

- Under composition the privacy loss adds up; its mean grows like $k\varepsilon^2/2$ and its spread like $\sqrt k\,\varepsilon$.
- For a quick budget, give each of $k$ steps $\varepsilon \approx \varepsilon_{\text{total}}/\sqrt{2k\ln(1/\delta')}$.
- If all you know is each step's $(\varepsilon, \delta)$, compute the optimal bound exactly; it costs nothing.
- If you know the mechanism, compose its curve with an accountant instead of composing points.
- Adaptive choice of queries is free; adaptive choice of privacy parameters, or stopping rules that look at the raw data, are not.

## Further reading

- Dwork & Roth, *The Algorithmic Foundations of Differential Privacy* (2014), §3.5: basic and advanced composition, the version most libraries cite.
- Kairouz, Oh & Viswanath, "The Composition Theorem for Differential Privacy" (ICML 2015): the hypothesis-testing view, the worst-case mechanism and the optimal bound.
- Rogers, Roth, Ullman & Vadhan, "Privacy Odometers and Filters: Pay-as-you-Go Composition" (NeurIPS 2016): why adaptively chosen privacy parameters need new theorems.
- Dong, Roth & Su, "Gaussian Differential Privacy" (JRSS Series B, 2022): composing whole privacy curves, and why the total loss looks normal.

**Derivations.** Every result in this chapter is derived step by step in [the full edition's chapter](../../book-full/part03-dp-foundations/04-composition.md).
