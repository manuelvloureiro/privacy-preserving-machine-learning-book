# Adaptive composition: filters, odometers and individual accounting

## Why this matters

Every composition theorem so far assumes that each step's privacy parameter, and the number of steps, are fixed before the run ([Advanced and optimal composition](../part03-dp-foundations/04-composition.md)). Real training breaks this whenever you stop on the validation loss or lower the noise as training stalls. Stochastic gradient descent (SGD) with differential privacy (DP), called DP-SGD, adds a third case, since a record with a small gradient costs less than the worst case. Plugging the values that occurred into a fixed-parameter theorem gives a report that can be wrong several times as often as it claims.

Fixed-step DP-SGD with a schedule chosen in advance needs none of this. Otherwise, which tool you need depends on what you fixed in advance. If only the stopping time adapts within a planned maximum, report $\varepsilon$ for the maximum. If you fixed a budget and want to spend it adaptively, a **privacy filter** decides before each step whether it still fits. If you fixed nothing, a **privacy odometer** reports what you have spent so far, at a small extra price.

## Why plugging in the realized parameters fails {#sec-adapt-why-breaks}

Advanced composition bounds the total privacy loss of a run ([Numerical accounting](06-numerical-accounting.md#sec-num-pld)), except with a failure probability $\delta$, when the number of steps is fixed in advance. Applying it **ex post**, at a stopping time chosen from the outputs, is a different claim that Rogers et al. (2016) showed to be invalid.

A simple rule breaks it. Run randomized response with $\varepsilon = 0.1$ on one person's bit, step after step, so that the total loss is a random walk. The plug-in bound is correct at every fixed step, but an analyst may stop the first time the walk crosses it and report the bound at that step. At $\delta = 0.01$, the full edition finds this report wrong with probability 0.0318 within 10,000 steps, over three times its claim. Given enough steps, the walk crosses the bound with probability 1.

Two kinds of report escape the problem. For pure DP, basic composition, the plain sum of the per-step parameters, bounds the loss of every output, so it is valid however you stop. And a filter escapes, because it fixes its budget in advance and makes no running claim that a stopping rule could catch out.

## Early stopping within a planned maximum {#sec-adapt-early-stop}

The commonest adaptive run needs no new theory. You fix the noise, the sampling rate and a maximum number of steps before training, and you stop early when the validation loss stops improving. **Then report the $\varepsilon$ of the planned maximum, not of the steps you ran.** The run that stopped early releases a prefix of the full planned run. Anything computed from that prefix, including the decision to stop, is post-processing of the full run, so the full run's guarantee covers it.

For example, plan 100 full-batch Gaussian steps whose noise multiplier is 10. That run is $(5.299, 10^{-5})$-DP, so if you stop at step 60, you still report 5.299.

The rule has three conditions. First, the stopping decision must look only at released outputs, or at data you account for separately, because a decision that reads the private data leaks through the stopping time. Second, the maximum, the noise and the sampling rate must really be fixed. If you might extend training past the plan, you need a filter or an odometer. Third, report with the same accountant and $\delta$ you calibrated with. Choosing hyperparameters from several runs is a separate cost, which [Hyperparameter tuning](../part05-dp-ml/03-hyperparameter-tuning.md) accounts for.

Library accountants are not odometers. In Opacus, `PrivacyEngine.make_private_with_epsilon` calibrates the noise for a target $\varepsilon$ and $\delta$ over a planned number of `epochs`. Its `get_epsilon(delta)`, like the one in Google's `dp_accounting`, composes the steps taken so far and converts them once. That is valid only at a time fixed in advance, so call it at the planned end, or report the target you calibrated for.

## The Rényi filter {#sec-adapt-renyi-filter}

When the noise itself changes during training, use a Rényi filter. Rényi differential privacy (RDP) measures each step by a curve over orders $\alpha$ ([Rényi DP](01-renyi-dp.md)). Fix one order and an RDP budget $B$ before the run. Each step then has a **spend**, its RDP at that order, which may depend on earlier outputs but must be known before the step runs, so it is called **predictable**. Write $\varepsilon_t(\alpha)$ for the spend of the $t$-th step. The filter runs that step only if the spends so far, its own included, fit in the budget,
$$\sum_{j \le t}\varepsilon_j(\alpha) \le B.$$ {#eq-adapt-renyi-filter}
Then the stopped run is $(\alpha, B)$-RDP, as if the spends had been fixed in advance (Feldman and Zrnic 2021).

Each condition matters, because a spend that depended on its own step's output could be small exactly where the output reveals the most. The budget must also hold on every path, not on average. And the order must be fixed before the run, although you may track several and stop when any one runs out.

To aim at a DP target, write $\varepsilon_g$ for the global privacy level you want. Then set the budget from the conversion of [RDP conversions](02-rdp-conversions.md), run backwards,
$$B = \varepsilon_g - \frac{\ln(1/\delta)}{\alpha - 1}.$$
For example, a target of $(5.299, 10^{-5})$ at order 5.8 gives $B = 2.900$. A full-batch Gaussian step spends $\alpha/(2\sigma_t^2)$, where $\sigma_t$ is its noise multiplier, so with $\sigma_t$ at 10 the budget lasts 100 steps. Now suppose training plateaus at step 60 and you halve the noise. Each later step then costs four times as much, so the filter allows 10 more steps. Either way the run is $(5.8, 2.900)$-RDP, whatever rule decided the switch, provided that rule looks only at released outputs.

## Filters for pure differential privacy {#sec-adapt-dp-filters}

Interactive query systems give each query a pure-DP parameter $\varepsilon_t$, chosen from earlier answers. The **basic filter** of Rogers et al. (2016) continues while they sum to at most the global $\varepsilon_g$. It is exact, but its budget grows like the number of steps.

For square-root growth, use the **Whitehouse filter** of Whitehouse et al. (2023). It continues while
$$\sqrt{2\ln(1/\delta)\textstyle\sum_t\varepsilon_t^2} + \tfrac12\textstyle\sum_t\varepsilon_t^2 \le \varepsilon_g,$$ {#eq-adapt-whitehouse}
summing over the steps so far and the one about to run, and its stopped run is $(\varepsilon_g, \delta)$-DP. It reads like advanced composition because it is the Rényi filter applied to zero-concentrated differential privacy (zCDP) ([Concentrated DP](03-concentrated-dp.md)), so adaptivity costs essentially nothing.

For example, a budget of 1 at $\delta = 10^{-6}$ lets the Whitehouse filter run 349 steps of 0.01. That is slightly more than advanced composition allows with the number of steps fixed in advance. So use the Whitehouse form, checked before each step with that step's parameter included.

For an analytics service, use a library that enforces the budget for you. OpenDP's `make_privacy_filter` wraps an odometer built with `make_fully_adaptive_composition` and rejects any query that would push the total past its bound. Tumult Analytics tracks a `Session`'s remaining budget, such as a `RhoZCDPBudget`, and exposes it as `Session.remaining_privacy_budget`.

## Odometers {#sec-adapt-odometers}

Open-ended analyses and training to a target accuracy have no budget to stop at. They need a running answer to "how much have I spent?" that holds at every step at once. The **Ville odometer** gives one. Fix a grid $\Lambda$ of orders before the run, with $|\Lambda|$ orders in it, and write $S_t(\alpha)$ for the RDP spend so far at order $\alpha$. With probability at least $1 - \delta$, simultaneously for every step $t$, the loss so far is at most
$$U_t = \min_{\alpha\in\Lambda}\Big[S_t(\alpha) + \frac{\ln(|\Lambda|/\delta)}{\alpha - 1}\Big].$$ {#eq-adapt-ville-odometer}
This is the usual RDP conversion at the best order of the grid, with $\delta$ replaced by $\delta/|\Lambda|$, and that replacement is the whole price. So keep the grid small.

Over the part's grid of orders, the price for full-batch Gaussian steps is 15 to 19%. For example, the 100-step run above gets 6.253 instead of 5.299. Lécuyer (2021) builds a slightly looser odometer from doubling Rényi filters.

An odometer is a tail bound, not a DP guarantee. To turn it into an $(\varepsilon_g, \delta)$-DP guarantee, stop by a rule that keeps the running bound at most $\varepsilon_g$. If you have a budget anyway, a Rényi filter at one order fixed in advance is cheaper, because it needs no union over orders.

## Individual privacy accounting in DP-SGD {#sec-adapt-individual}

A DP-SGD step clips each record's gradient to a norm of at most the clipping norm $C$, and adds Gaussian noise with standard deviation $\sigma C$. The usual analysis charges every record as if its clipped gradient always had norm $C$, but a record that the model fits moves the noisy sum less. Write $\bar g_{i,t}$ for the clipped gradient of record $i$ at step $t$. With full-batch steps, that record's spend is
$$\varepsilon_{i,t}(\alpha) = \frac{\alpha\|\bar g_{i,t}\|^2}{2\sigma^2C^2} \le \frac{\alpha}{2\sigma^2}.$$ {#eq-adapt-individual-rdp}
This spend is predictable, because the gradient depends only on the record and the current model. So the **individual filter** of Feldman and Zrnic (2021) runs one Rényi filter per record and drops a record once its own spend would pass $B$. Every record then has the guarantee of the worst-case run, while records with small gradients keep training long after.

In the full edition's simulation with the order and budget above, the median typical record has an individual $\varepsilon$ of 3.17 after 100 steps, against 5.299 for the worst case.

Read the per-record numbers with care. A per-record $\varepsilon$ from the realized spend is an estimate, not a guarantee, because it is a fixed-time bound at a random quantity. It is also sensitive, since a small spend says that the model fits that record well, so show it only to the record's owner. There is a fairness cost too, because the records the model serves poorly are dropped first, and these often come from under-represented groups (Yu et al. 2023; [Disparate impact](../part05-dp-ml/08-disparate-impact.md)). Under Poisson subsampling, it needs every record's gradient norm at every step, sampled or not (Koskela et al. 2023).

## Code {#sec-adapt-code}

A filter or an odometer is a sum of spends and a test. This snippet codes the Rényi filter for a noise schedule, and prints the numbers of the example above.

```python
from math import log


def renyi_filter(sigmas: list[float], alpha: float, budget: float) -> int:
    """Steps of a full-batch Gaussian noise schedule that a Renyi filter lets through."""
    spent, steps = 0.0, 0
    for sigma in sigmas:
        cost = alpha / (2 * sigma**2)
        if spent + cost > budget + 1e-12:  # tolerance for floating-point sums
            break
        spent, steps = spent + cost, steps + 1
    return steps


alpha, budget = 5.8, 2.9
print(renyi_filter([10.0] * 1000, alpha, budget))  # 100
print(renyi_filter([10.0] * 60 + [5.0] * 1000, alpha, budget) - 60)  # 10
print(f"{budget + log(1 / 1e-5) / (alpha - 1):.3f}")  # 5.299
```

## Cheat sheet

| Situation | Use | Report |
| --- | --- | --- |
| Early stopping within a planned maximum | Nothing new | $\varepsilon$ of the planned maximum |
| Noise chosen during training, budget fixed | Rényi filter at a fixed order | $(\alpha, B)$-RDP, with $B = \varepsilon_g - \ln(1/\delta)/(\alpha - 1)$ |
| Pure-DP queries chosen adaptively | Whitehouse filter, @eq-adapt-whitehouse | $(\varepsilon_g, \delta)$-DP |
| No budget, stop whenever | Ville odometer, @eq-adapt-ville-odometer | The running bound, valid at every step |
| Per-record spend in DP-SGD | Individual filter, @eq-adapt-individual-rdp | The run's $(\alpha, B)$; per-record values as estimates |

: Which adaptive tool for which situation. Spends may depend on earlier outputs but must be known before their step runs. {#tbl-adapt-cheat}

**Rules of thumb**

- If you early-stop within a planned maximum number of steps, report $\varepsilon$ for the maximum.
- Never report a fixed-time bound at a time chosen from the outputs, and that includes a library's `get_epsilon` called when you decide to stop.
- An odometer must fix its grid of orders in advance. Picking the best order after the run without paying $\ln|\Lambda|$ is the plug-in mistake again.
- Compute per-record spends from the clipped gradient norms, keep them out of logs, and remember that the records dropped first are the ones the model already serves worst.

## Further reading

- Whitehouse, Ramdas, Rogers & Wu, "Fully-Adaptive Composition in Differential Privacy" (ICML 2023): the filter that matches advanced composition, and odometers for pure-DP steps.
- Feldman & Zrnic, "Individual Privacy Accounting via a Rényi Filter" (NeurIPS 2021): the Rényi filter and per-record accounting.

**Derivations.** Every result here is derived step by step in [the full edition's chapter](../../book-full/part04-privacy-accounting/08-adaptive-composition.md).
