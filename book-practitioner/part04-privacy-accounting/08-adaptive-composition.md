# Adaptive composition: filters, odometers and individual accounting

## What it is {#sec-adapt-why-breaks}

Differential privacy (DP) composition theorems assume that each step's privacy parameter and the total step count are fixed before the run ([Composition](../part03-dp-foundations/04-composition.md)). Adaptive runs violate this assumption when later choices depend on earlier outputs; for example, training may stop based on validation loss. The fixed-parameter theorems do not cover the realized values (Rogers et al. 2016). This **plug-in mistake** can make the bound fail about three times as often as its stated $\delta$.

A **privacy filter** fixes a total budget and rejects any step that would exceed it, preserving the guarantee under any stopping rule. A **privacy odometer** reports cumulative spend for open-ended analyses. Its anytime guarantee has a higher $\varepsilon$.

## When you meet it

Early stopping with a fixed maximum needs neither tool. Set noise, sampling rate and maximum steps in advance. You can then stop when validation loss stops improving, because the executed steps are a prefix of the plan. **Report the planned maximum's $\varepsilon$, not the value for the steps actually run.** The plan's guarantee is $(5.299, 10^{-5})$-DP. It covers 100 full-batch Gaussian steps at noise multiplier 10. Training may stop at step 60 and still use that guarantee.

The stop signal must use released models or held-out data, so the decision is post-processing. To exceed the maximum, create a new plan and account for the run from its first step. When resuming, restore the accountant state ([the calibration workflow](10-calibrate-dp-sgd.md#sec-calib-train)).

Use filters when training changes noise in response to outputs. Analytics services also provide them, including OpenDP's `make_privacy_filter` and Tumult Analytics' `Session`.

## How to use it {#sec-adapt-dp-filters}

Feldman and Zrnic's (2021) Rényi filter uses one Rényi divergence order $\alpha$, the order parameter fixed in advance. It charges each step's Rényi differential privacy (RDP) spend at that order ([Rényi DP](01-renyi-dp.md)). A step's spend may depend on earlier outputs, but must be known before that step runs. The filter rejects any step that would exceed budget $B$. Let $\varepsilon_g$ denote the target privacy level and $\delta$ the slack. Invert the [RDP conversion](02-rdp-conversions.md) to set
$$B = \varepsilon_g - \frac{\ln(1/\delta)}{\alpha - 1}.$$

Let $\sigma$ denote the Gaussian noise multiplier, the standard deviation relative to sensitivity. A Gaussian step then spends $\alpha/(2\sigma^2)$ at order $\alpha$. For this target, the best order is 5.8 and the budget is $B = 2.900$. This allows 100 full-batch steps at noise multiplier 10. If validation loss stops improving at step 60, halving the noise makes each later step cost four times as much. The remaining budget covers 10 more steps and preserves the target.

OpenDP's filter uses zero-concentrated DP (zCDP). Its budget $\rho$ bounds RDP spend at every order ([Concentrated DP](03-concentrated-dp.md)).

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

OpenDP permits 99 releases rather than 100 because it rounds each charge up for floating-point safety. For pure-DP queries, the **Whitehouse filter** (Whitehouse et al. 2023) allows more small queries than basic summation of $\varepsilon$.

## What goes wrong

An odometer gives a looser bound than a filter. The **Ville odometer** converts RDP spend at the best order on a grid fixed in advance. It accounts for this later choice by dividing $\delta$ by the grid size. For the 100-step run above, this raises $\varepsilon$ from 5.299 to 6.253.

**Individual accounting** charges each record according to its clipped gradient norm, since smaller gradients move the model less. It excludes a record when its spend would exceed the budget. In the full edition's simulation, median individual $\varepsilon$ is 3.17, compared with worst-case 5.299. Because this accounting uses observed norms, it is an estimate rather than a guarantee. It also reveals which records the model fits well. High-gradient records are excluded first and may be the records the model serves worst ([Disparate impact](../part05-dp-ml/08-disparate-impact.md)).

## References

- Whitehouse, Ramdas, Rogers & Wu, "Fully-Adaptive Composition in Differential Privacy" (ICML 2023).
- Feldman & Zrnic, "Individual Privacy Accounting via a Rényi Filter" (NeurIPS 2021).

**Derivations.** Every result here is derived step by step in [the full edition's chapter](../../book-full/part04-privacy-accounting/08-adaptive-composition.md).
