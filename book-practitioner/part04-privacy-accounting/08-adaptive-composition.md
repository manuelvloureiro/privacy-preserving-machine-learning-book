# Adaptive composition: filters, odometers and individual accounting

## What it is {#sec-adapt-why-breaks}

The composition theorems of differential privacy (DP) assume that you fix each step's privacy parameter and the number of steps before the run ([Composition](../part03-dp-foundations/04-composition.md)). Actual runs often choose them from the outputs, for example when training stops on the validation loss. Then the theorems do not hold for the actual values (Rogers et al. 2016). This **plug-in mistake** can make the bound fail about three times as often as its $\delta$ claims.

A **privacy filter** fixes a total budget in advance and refuses any step that would go past it, so the run keeps the budget's guarantee for any stopping rule. A **privacy odometer**, for open-ended analyses, reports the spend so far instead. Its value holds at every step, at a higher $\varepsilon$.

## When you meet it

Early stopping with a planned maximum needs neither. Fix the noise, sampling rate and maximum number of steps in advance, then stop when the validation loss stops falling. **Report the $\varepsilon$ of the planned maximum, not of the steps you ran.** The steps you ran are the start of the planned run, so the planned guarantee covers them. If 100 planned full-batch Gaussian steps with noise multiplier 10 are $(5.299, 10^{-5})$-DP, a run that stops at step 60 reports that.

The rule has three conditions. The stop signal must come from the released models or from held-out data, so that the stop is post-processing. To train past the maximum, make a new plan and account it from the first step. A resumed run must restore its accountant ([the calibration workflow](10-calibrate-dp-sgd.md#sec-calib-train)).

You meet filters when training changes the noise in response to the outputs, and in analytics services, such as OpenDP's `make_privacy_filter` or Tumult Analytics' `Session`.

## How to use it {#sec-adapt-dp-filters}

The Rényi filter of Feldman and Zrnic (2021) works in Rényi differential privacy (RDP) at one order, written $\alpha$ and fixed in advance. It charges each step its spend at that order ([Rényi DP](01-renyi-dp.md)). The spend can depend on earlier outputs, but the filter must know it before the step runs. The filter refuses a step that would take the total past a budget $B$. For a target privacy level $\varepsilon_g$ with slack $\delta$, use the [conversion from RDP](02-rdp-conversions.md) in reverse, which sets
$$B = \varepsilon_g - \frac{\ln(1/\delta)}{\alpha - 1}.$$

A Gaussian step with a noise multiplier, written $\sigma$, spends $\alpha/(2\sigma^2)$ at order $\alpha$. At order 5.8, the best order for this target, the budget is $B = 2.900$, enough for 100 full-batch steps at $\sigma = 10$. Suppose that the validation loss stops falling at step 60 and you halve the noise. Each later step then costs four times as much, so the budget left for 40 steps pays for 10, and the run still meets the target.

OpenDP's filter works in zero-concentrated DP (zCDP). Its single budget number, written $\rho$, bounds the RDP spend at every order at once ([Concentrated DP](03-concentrated-dp.md)).

```python
import opendp.prelude as dp

dp.enable_features("contrib")
domain, metric = dp.atom_domain(T=float, nan=False), dp.absolute_distance(T=float)
odometer = dp.c.make_fully_adaptive_composition(domain, metric, dp.zero_concentrated_divergence())
queryable = dp.c.make_privacy_filter(odometer, d_in=1.0, d_out=0.5)(0.0)  # rho = 0.5, i.e. 2.9 at order 5.8

answered = 0
try:
    for _ in range(1000):  # sigma = 10: rho = 1 / (2 sigma^2) = 0.005 each
        queryable(dp.m.make_gaussian(domain, metric, scale=10.0))
        answered += 1
except dp.OpenDPException:  # the release that would exceed the budget
    pass
print(answered)  # 99
```

OpenDP lets 99 releases through, not 100, because it rounds each charge up for floating-point safety. For pure-DP queries, the **Whitehouse filter** (Whitehouse et al. 2023) holds many more small queries than a plain sum of their $\varepsilon$.

## What goes wrong

An odometer costs more than a filter. The **Ville odometer** converts the RDP spend at the best order of a grid fixed in advance. It pays for that late choice with $\delta$ divided by the grid's size, which raises $\varepsilon$ from 5.299 to 6.253 on the 100-step run above.

**Individual accounting** charges each training record by its own clipped gradient norm, because a small gradient moves the model less. It drops a record when its spend would pass the budget. In the full edition's simulation, the median record ends at an individual $\varepsilon$ of 3.17, against 5.299 for the worst case. But it comes from the actual norms, so it is an estimate, not a guarantee. It is also sensitive, because a small value says that the model fits that record well. And the records with large gradients, which go first, are often those the model serves worst ([Disparate impact](../part05-dp-ml/08-disparate-impact.md)).

## References

- Whitehouse, Ramdas, Rogers & Wu, "Fully-Adaptive Composition in Differential Privacy" (ICML 2023).
- Feldman & Zrnic, "Individual Privacy Accounting via a Rényi Filter" (NeurIPS 2021).

**Derivations.** Every result here is derived step by step in [the full edition's chapter](../../book-full/part04-privacy-accounting/08-adaptive-composition.md).
