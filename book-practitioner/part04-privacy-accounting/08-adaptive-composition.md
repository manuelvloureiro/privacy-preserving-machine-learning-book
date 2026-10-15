# Adaptive composition: filters, odometers and individual accounting

## Why this matters

You train a model with differential privacy (DP) using stochastic gradient descent (SGD) with clipped per-example gradients and Gaussian noise, called DP-SGD. You want to stop when the validation loss stalls, lower the noise mid-run, or let well-fitted records keep training. What $\varepsilon$ can you claim?

Every composition theorem so far assumes that the per-step budgets $\varepsilon_1, \dots, \varepsilon_k$, and the number of steps $k$, are fixed before the run starts ([Advanced and optimal composition](../part03-dp-foundations/04-composition.md)). Data-dependent stopping, adaptive noise and per-record budgets make them random. Plugging the parameters you actually used into a fixed-parameter theorem after the run is not valid.

The one idea is to split "what is the $\varepsilon$ of this run?" in two. A **privacy filter** answers "may I run one more step without exceeding a budget fixed in advance?" A **privacy odometer** answers "how much have I spent so far?", and must hold at every step at once, since you may stop at any time. In Rényi differential privacy (RDP), a filter at a fixed order costs nothing, and an odometer costs a small union bound. @tbl-adapt-overview maps the decisions to the sections.

| Decision | Rule | Section |
| --- | --- | --- |
| Can I report a fixed-time bound at a time I chose from the outputs? | No: that bound fails more often than its $\delta$ | @sec-adapt-why-breaks |
| A global budget, pure-DP queries with $\varepsilon_t$ chosen on the fly | Whitehouse filter: advanced composition, valid under adaptivity | @sec-adapt-dp-filters |
| Adaptive noise or stopping in an RDP accountant | Rényi filter at an order fixed before the run | @sec-adapt-renyi-filter |
| No budget; report the spend so far | Ville odometer over a small grid of orders | @sec-adapt-odometers |
| Let well-fitted records keep training | Individual filter on each record's own spend | @sec-adapt-individual |

: The decisions behind adaptive privacy budgets, and where this chapter makes each one. {#tbl-adapt-overview}

## Why plugging in the realized parameters fails {#sec-adapt-why-breaks}

**Idea.** Advanced composition bounds the total privacy loss at a time fixed in advance: except with probability $\delta$, the loss after $t$ steps is below a bound $U_t$. An analyst who stops when the loss happens to be high picks exactly the paths where the bound fails. Stopping on anything correlated with the loss, such as "the answers look unusual", has the same effect.

**Result.** A filter with budget $(\varepsilon_g, \delta)$ looks, before each step, at the parameters so far, including the step about to run, and says continue or halt; it is valid if the stopped run is $(\varepsilon_g, \delta)$-DP however the steps were chosen. An odometer is a running bound $U_t$; it is valid if the probability that the loss *ever* exceeds it is at most $\delta$. That "ever" is what plugging in misses, and paying for it costs an extra $\ln\ln n$ inside the logarithm for $n$ steps (Rogers et al. 2016). A filter makes no running claim, so it need not pay.

**Example.** Take randomized response with $\varepsilon = 0.1$ on one person's bit, every step, and the plug-in advanced-composition bound at $\delta = 0.01$. At any single fixed step up to 1,000, the bound fails with probability at most 0.0016. But an analyst who stops at the first crossing makes it fail with the probability in @tbl-adapt-plugin: by $10^4$ steps, 0.0318, so a report "$\varepsilon = U_T$ with $\delta = 0.01$" is wrong 3.18 times as often as it claims.

| Horizon $n$ | 100 | 1000 | 10,000 |
| --- | --- | --- | --- |
| $\delta = 0.05$ | 0.0575 | 0.1004 | 0.1424 |
| $\delta = 0.01$ | 0.0090 | 0.0202 | 0.0318 |
| $\delta = 10^{-5}$ | $3.12\times10^{-6}$ | $1.80\times10^{-5}$ | $3.69\times10^{-5}$ |

: Exact probability that randomized response with $\varepsilon = 0.1$ crosses the plug-in advanced-composition bound by step $n$. It exceeds $\delta$ and keeps growing with the horizon. {#tbl-adapt-plugin}

![Left: forty runs of randomized response with $\varepsilon = 0.1$, the loss in units of $\varepsilon$, and the plug-in bound at $\delta = 0.01$; an analyst who stops at the first crossing freezes the red paths there. Right: the probability that the bound is crossed by step $n$, for $\delta = 0.05$ and $0.01$ (dashed lines); it passes $\delta$ and keeps growing.](figures/adaptive_composition_plugin_failure.svg){#fig-adapt-plugin-failure}

**Pitfalls.** A smaller $\delta$ does not fix it: at $\delta = 10^{-5}$ the crossing probability by step $10^4$ is $3.69\times10^{-5}$, 3.69 times $\delta$. Basic composition is immune, because the sum of the realized $\varepsilon_t$ bounds the loss of every output; summing what you spent is a valid filter and odometer for pure DP. Every high-probability bound, the kind you want for long runs, is exposed.

## Filters for pure differential privacy {#sec-adapt-dp-filters}

**Idea.** An interactive system has a global budget, and each query brings its own $\varepsilon_t$, chosen after earlier answers. Summing the $\varepsilon_t$ is valid but grows like $k$. The best filter that grows like $\sqrt k$ turns out to be advanced composition itself, with a slightly different mean term.

**Result.** For steps that are each $\varepsilon_t$-DP given the past, the **Whitehouse filter** (Whitehouse et al. 2023) continues while
$$\sqrt{2\ln(1/\delta)\textstyle\sum_t\varepsilon_t^2} + \tfrac12\textstyle\sum_t\varepsilon_t^2 \le \varepsilon_g,$$ {#eq-adapt-whitehouse}
where the sums run over every step so far, including the one about to run. The first term is the typical spread of the total loss, the second its mean, and the stopped run is $(\varepsilon_g, \delta)$-DP. Equivalently, keep $\frac12\sum_t\varepsilon_t^2 \le \rho^* = (\sqrt{\ln(1/\delta) + \varepsilon_g} - \sqrt{\ln(1/\delta)})^2$; equal steps of $\varepsilon_0$ then number $\lfloor 2\rho^*/\varepsilon_0^2\rfloor$. The Rogers et al. (2016) filter has the same shape with worse constants; the **basic filter**, $\sum_t\varepsilon_t \le \varepsilon_g$, gives $(\varepsilon_g, 0)$-DP.

**Example.** At $\delta = 10^{-6}$ and $\varepsilon_g = 1$, the Whitehouse filter has $\rho^* = 0.017469$ and allows 349 steps of 0.01, slightly more than the 337 of advanced composition with $k$ fixed in advance (@tbl-adapt-steps). Conversely, 1,000 steps of 0.01 need a budget of 1.712 under the Whitehouse filter, 1.763 under fixed-$k$ advanced composition and 10.000 under the basic filter.

| $\varepsilon_0$ | Basic | Advanced (fixed $k$) | Rogers filter | Whitehouse filter |
| --- | --- | --- | --- | --- |
| 0.01 | 100 | 337 | 154 | 349 |
| 0.001 | 1000 | 33,785 | 15,481 | 34,937 |

: Steps of $\varepsilon_0$ allowed by a global budget $\varepsilon_g = 1$ at $\delta = 10^{-6}$. {#tbl-adapt-steps}

![Global budget needed to run $k$ steps of $\varepsilon_0 = 0.01$ at $\delta = 10^{-6}$: basic composition, advanced composition with $k$ fixed in advance, and the Rogers and Whitehouse filters. The Whitehouse filter lies on or below fixed-$k$ advanced composition; the Rogers filter sits a constant factor above.](figures/adaptive_composition_filter_budgets.svg){#fig-adapt-filter-budgets}

**Pitfalls.** The Rogers et al. filter needs 1.40 to 1.51 times the budget of the Whitehouse filter and lets through fewer than half as many steps; use the Whitehouse form. For steps with $\delta_t > 0$, add a second rule, $\sum_t\delta_t \le \delta''$, and the guarantee becomes $(\varepsilon_g, \delta + \delta'')$. Check the filter *before* each step, with that step's $\varepsilon_t$ included.

## The Rényi filter {#sec-adapt-renyi-filter}

**Idea.** RDP describes each step by a curve $\varepsilon_t(\alpha)$ over orders $\alpha > 1$, and the curves of a run add ([Rényi DP](01-renyi-dp.md)). This still holds when each curve depends on earlier outputs, provided you fix one order and check a budget there on every path. In words: a rescaled exponential of the privacy loss never grows on average from one step to the next, and no stopping rule can raise its average.

**Result.** Fix an order $\alpha$ and an RDP budget $B$ before the run. Let step $t$ be $(\alpha, \varepsilon_t(\alpha))$-RDP given the past, with $\varepsilon_t(\alpha)$ known before the step runs. Run step $t$ only if
$$\sum_{j \le t}\varepsilon_j(\alpha) \le B.$$ {#eq-adapt-renyi-filter}
Then the stopped run is $(\alpha, B)$-RDP, exactly as if the spends had been fixed. For a DP target $(\varepsilon_g, \delta)$, set $B = \varepsilon_g - \ln(1/\delta)/(\alpha - 1)$, Mironov's conversion run backwards. A full-batch Gaussian step with noise multiplier $\sigma_t$ spends $\alpha/(2\sigma_t^2)$.

**Example.** You train with full-batch noisy gradient descent and a budget of $(5.299, 10^{-5})$ at order $\alpha = 5.8$, so $B = 5.299 - \ln(10^5)/4.8 = 2.900$. With $\sigma_t = 10$ throughout, each step spends $5.8/200 = 0.0290$ and you get 100 steps. If you switch to $\sigma_t = 5$ after a plateau at step 60, each later step costs $5.8/50 = 0.1160$, and the filter allows $(2.900 - 60\times0.0290)/0.1160 = 10$ more. Either way the run is $(5.8, 2.900)$-RDP, whatever rule decided the switch. When branches trade costs, the filter even beats fixed-curve composition, which charges every step its worst branch: 1.8487 against 2.5419 in the full edition's three-step example.

**Pitfalls.** A step's spend must be known before it runs; a price that depends on the step's own output breaks the theorem. The budget must hold on every path, not on average. Fix the order before the run: you may track several orders and halt when any one is exhausted, but continuing while *some* order still fits costs a union bound (@sec-adapt-odometers). And the per-step spend must match the sampling the code actually runs ([Choosing an accountant](09-choosing-an-accountant.md)).

## Odometers {#sec-adapt-odometers}

**Idea.** Early stopping and open-ended analyses have no budget to stop at; they need a running "how much have I spent?" that stays valid however the stopping time is chosen. At one order fixed in advance, Mironov's conversion of the running spend is such an answer, for free. The cost comes from picking the best order after the fact.

**Result.** Track the running spend $S_t(\alpha) = \sum_{j\le t}\varepsilon_j(\alpha)$ at each order $\alpha$ in a grid $\Lambda$ of $|\Lambda|$ orders fixed in advance. The **Ville odometer** reports
$$U_t = \min_{\alpha\in\Lambda}\Big[S_t(\alpha) + \frac{\ln(|\Lambda|/\delta)}{\alpha - 1}\Big],$$ {#eq-adapt-ville-odometer}
and with probability at least $1 - \delta$ the loss stays below $U_t$ at every step simultaneously. It is Mironov's conversion with $\delta$ replaced by $\delta/|\Lambda|$: the union bound over orders is the whole price. Lécuyer (2021) gives an alternative from Rényi filters with doubling budgets, slightly looser below.

**Example.** For full-batch Gaussian steps with $\sigma = 10$ at $\delta = 10^{-5}$ and the part's 154 orders (@tbl-adapt-odometers), the Ville odometer costs 15–19% over the fixed-time value: 6.253 against 5.299 at $t = 100$. A single order fixed in advance (5.8, best for $t = 100$) is free at its design point and poor elsewhere: 31.399 at $t = 1000$. For the part's running example (DP-SGD with $q = 0.01$, $\sigma = 1$, $10^4$ steps, $\delta = 10^{-5}$), the fixed-time RDP value is 7.43, and an odometer valid whenever the run stops costs 8.82 (Ville) or 9.08 (Lécuyer).

| Steps $t$ | Fixed-time conversion | Ville odometer (154 orders) | Lécuyer odometer | Ville at $\alpha = 5.8$ only |
| --- | --- | --- | --- | --- |
| 10 | 1.568 | 1.869 | 1.916 | 2.689 |
| 100 | 5.299 | 6.253 | 6.507 | 5.299 |
| 1000 | 20.175 | 23.194 | 24.633 | 31.399 |

: Running $\varepsilon$ at $\delta = 10^{-5}$ for $t$ full-batch Gaussian steps with $\sigma = 10$. Only the last three columns are valid when $t$ is chosen from the outputs. {#tbl-adapt-odometers}

![Running privacy bounds at $\delta = 10^{-5}$ for full-batch Gaussian steps with $\sigma = 10$: the fixed-time conversion (dashed, not a valid odometer), the Ville odometer over 154 orders, Lécuyer's odometer, and a Ville odometer at the single order 5.8.](figures/adaptive_composition_odometers.svg){#fig-adapt-odometers}

**Pitfalls.** An odometer is a tail bound, not a DP guarantee: "$\varepsilon = U_T$" says the loss exceeds the random $U_T$ with probability at most $\delta$. For an $(\varepsilon_g, \delta)$-DP guarantee, stop by a rule that keeps $U_T \le \varepsilon_g$, a filter over the grid. Keep the grid small: 154 orders raise $\ln(10^5) = 11.513$ to 16.550. If you have a budget, a filter is cheaper.

## Individual privacy accounting in DP-SGD {#sec-adapt-individual}

**Idea.** DP-SGD's usual analysis charges every record the worst case at every step, as if its clipped gradient always had norm $C$, the clipping norm. Once the model fits a record, its gradient shrinks and it moves the noisy sum less. That cost is known before the step (it depends on the record and the current model), so the Rényi filter applies record by record: track each record's spend and drop it from the sum once its budget is used.

**Result.** With full-batch steps and noise of standard deviation $\sigma C$ on the summed clipped gradients, record $i$'s spend at step $t$ is
$$\varepsilon_{i,t}(\alpha) = \frac{\alpha\|\bar g_{i,t}\|^2}{2\sigma^2C^2} \le \frac{\alpha}{2\sigma^2},$$ {#eq-adapt-individual-rdp}
where $\bar g_{i,t}$ is its clipped gradient. The **individual filter** (Feldman & Zrnic 2021) lets record $i$ take part only while its own spends sum to at most $B$, and the run is $(\alpha, B)$-RDP. No step costs more than the worst case, so every record trains at least as long as in the worst-case run.

**Example.** The full edition simulates 10,000 records with $C = 1$, $\sigma = 10$ and full-batch steps: 90% typical records whose gradients shrink as the model fits them, and 10% hard records never fitted. The worst-case run has 100 steps and $\varepsilon = 5.299$ at $\delta = 10^{-5}$, best at order 5.8, where $B = 2.900$. After 100 steps, individual $\varepsilon$ values (@tbl-adapt-individual) have typical median 3.17 and hard median 5.02, and 16.3% of all records are below half the worst case. Over 300 steps under the individual filter, hard records are dropped from step 101 (the median one at step 111), no typical record ever is, and the typical median spend at step 300 is 0.778, a quarter of the budget. Every record keeps $(5.8, 2.900)$-RDP.

| Records | 10th percentile | Median | 90th percentile |
| --- | --- | --- | --- |
| All | 2.56 | 3.25 | 3.81 |
| Typical | 2.55 | 3.17 | 3.67 |
| Hard | 4.23 | 5.02 | 5.26 |

: Individual $\varepsilon$ at $\delta = 10^{-5}$ after 100 full-batch steps with $\sigma = 10$, at order 5.8. The worst case is 5.299. {#tbl-adapt-individual}

![Left: cumulative spend at order 5.8 under the individual filter for four typical records (blue) and three hard records (red), against the worst case (dotted) and the budget $B = 2.90$ (dashed). Hard records reach the budget and stop; typical records level off far below it. Right: the fraction of each group taking part at each step.](figures/adaptive_composition_individual_spend.svg){#fig-adapt-individual-spend}

**Pitfalls.** The filter certifies one guarantee for everyone. A per-record $\varepsilon$ from the realized spend is an estimate, not a guarantee (@sec-adapt-why-breaks), and at a fixed order it cannot fall below $\ln(1/\delta)/(\alpha - 1) = 2.399$ here. Individual $\varepsilon$ is itself sensitive, since a small spend says "this record is well fitted": show it to the record's owner, never publish all of them. The records the model fits worst, often from under-represented groups, spend first and are dropped first, so the model sees even less of them. With Poisson subsampling the argument still works, using the subsampled Gaussian at noise multiplier $\sigma C/\|\bar g_{i,t}\|$ ([Amplification by subsampling](05-subsampling.md)), but it needs every record's gradient norm at every step, sampled or not.

## Code {#sec-adapt-code}

Filters and odometers are a sum of spends and a comparison. This snippet implements the Whitehouse filter, the Rényi filter for a noise schedule and the Ville odometer for full-batch Gaussian steps, and reproduces the numbers above.

```python
from math import log, sqrt


def whitehouse_statistic(eps_steps: list[float], delta: float) -> float:
    """Advanced composition of pure-DP budgets with mean term eps^2 / 2: valid under adaptivity."""
    v = sum(e * e for e in eps_steps)
    return sqrt(2 * log(1 / delta) * v) + v / 2


def steps_allowed(eps_step: float, eps_global: float, delta: float) -> int:
    """Run steps of eps_step while the Whitehouse filter, checked before each step, says continue."""
    spent: list[float] = []
    while whitehouse_statistic([*spent, eps_step], delta) <= eps_global:
        spent.append(eps_step)
    return len(spent)


def gaussian_rdp(alpha: float, sigma: float) -> float:
    """RDP at order alpha of one full-batch Gaussian step with noise multiplier sigma."""
    return alpha / (2 * sigma**2)


def renyi_filter(sigmas: list[float], alpha: float, budget: float) -> int:
    """Number of steps of a noise schedule that a Renyi filter at order alpha lets through."""
    spent, steps = 0.0, 0
    for sigma in sigmas:
        cost = gaussian_rdp(alpha, sigma)
        if spent + cost > budget + 1e-12:  # tolerance for floating-point sums
            break
        spent, steps = spent + cost, steps + 1
    return steps


ORDERS = [1 + x / 10 for x in range(1, 100)] + [float(x) for x in range(12, 64)] + [128.0, 256.0, 512.0]


def ville_odometer(t: int, sigma: float, delta: float, orders: list[float]) -> float:
    """Running epsilon after t full-batch Gaussian steps, valid whenever the run stops."""
    return min(t * gaussian_rdp(a, sigma) + log(len(orders) / delta) / (a - 1) for a in orders)


print(f"{whitehouse_statistic([0.01] * 1000, 1e-6):.3f}")  # 1.712
print(steps_allowed(0.01, 1.0, 1e-6))  # 349
alpha, budget = 5.8, 2.9
print(renyi_filter([10.0] * 1000, alpha, budget))  # 100
print(renyi_filter([10.0] * 60 + [5.0] * 1000, alpha, budget) - 60)  # 10
print(f"{budget + log(1 / 1e-5) / (alpha - 1):.3f}")  # 5.299
print(len(ORDERS))  # 154
print(f"{ville_odometer(100, 10.0, 1e-5, ORDERS):.3f}")  # 6.253
print(f"{ville_odometer(100, 10.0, 1e-5, [alpha]):.3f}")  # 5.299
print(f"{ville_odometer(1000, 10.0, 1e-5, [alpha]):.3f}")  # 31.399
```

`ORDERS` is the part's grid of orders. The full edition's script, [adaptive_composition.py](../../code/part04-privacy-accounting/adaptive_composition.py), also computes the plug-in failure exactly, checks the Rényi filter by simulation and runs the individual-accounting experiment:

```bash
uv run python code/part04-privacy-accounting/adaptive_composition.py
```

## Cheat sheet

| Object | Rule or bound | Guarantee |
| --- | --- | --- |
| Basic filter and odometer | Continue while $\sum_t\varepsilon_t \le \varepsilon_g$; report $\sum_{j\le t}\varepsilon_j$ | $(\varepsilon_g, 0)$-DP; holds on every path |
| Whitehouse filter (pure DP) | $\sqrt{2\ln(1/\delta)\sum_t\varepsilon_t^2} + \frac12\sum_t\varepsilon_t^2 \le \varepsilon_g$ | $(\varepsilon_g, \delta)$-DP |
| Rényi filter | $\sum_{j\le t}\varepsilon_j(\alpha) \le B$ on every path, $\alpha$ fixed in advance | $(\alpha, B)$-RDP; DP target: $B = \varepsilon_g - \ln(1/\delta)/(\alpha - 1)$ |
| Ville odometer | $U_t = \min_{\alpha\in\Lambda}[S_t(\alpha) + \ln(\lvert\Lambda\rvert/\delta)/(\alpha - 1)]$ | Loss above $U_t$ at some $t$ with probability at most $\delta$ |
| Individual spend, Gaussian sum | $\varepsilon_{i,t}(\alpha) = \alpha\lVert\bar g_{i,t}\rVert^2/(2\sigma^2C^2)$, at most $\alpha/(2\sigma^2)$ | Per-record filter gives $(\alpha, B)$-RDP for every record |

: Filters, odometers and individual accounting. {#tbl-adapt-cheat}

**Rules of thumb**

- Never report a fixed-time bound at a time chosen from the outputs. Use a filter if you have a budget, an odometer if you do not.
- For adaptive noise or early stopping in RDP, fix the order before the run and use a Rényi filter: it costs nothing.
- An odometer costs a union over orders: $\ln(1/\delta)$ becomes $\ln(|\Lambda|/\delta)$, about 15–19% in $\varepsilon$ on the examples here. Keep the grid small.
- Individual accounting gives every record the same guarantee and well-fitted records more training. Per-record $\varepsilon$ values are sensitive estimates, not guarantees.

## Further reading

- Rogers, Roth, Ullman & Vadhan, "Privacy Odometers and Filters: Pay-as-you-Go Composition" (NeurIPS 2016): the definitions, and why odometers cost more.
- Whitehouse, Ramdas, Rogers & Wu, "Fully-Adaptive Composition in Differential Privacy" (ICML 2023): the filter that matches advanced composition.
- Feldman & Zrnic, "Individual Privacy Accounting via a Rényi Filter" (NeurIPS 2021): the Rényi filter and per-record accounting.
- Koskela, Tobaben & Honkela, "Individual Privacy Accounting with Gaussian Differential Privacy" (ICLR 2023): tighter individual accounting, including subsampled DP-SGD.

**Derivations.** Every result in this chapter is derived step by step in [the full edition's chapter](../../book-full/part04-privacy-accounting/08-adaptive-composition.md).
