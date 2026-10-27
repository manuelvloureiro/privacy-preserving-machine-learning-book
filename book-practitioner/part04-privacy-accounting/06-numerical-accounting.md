# Numerical accounting with privacy loss distributions

## What it is {#sec-num-pld}

A privacy loss distribution (PLD) accountant computes the privacy level $\varepsilon$ of a whole differential privacy (DP) run numerically. Take a dataset and a neighbour that differs in one record, and run the mechanism on the first. The **privacy loss** $L$ of that run is the logarithm of how much likelier the output is on the first dataset than on the second. Because the output is random, $L$ is too, and its distribution is the PLD. Only outputs whose loss exceeds $\varepsilon$ add to the slack $\delta$, and with $(\cdot)_+$ turning a negative value into 0, the smallest slack is
$$\delta(\varepsilon) = \mathbb E\big[(1 - e^{\varepsilon - L})_+\big].$$

When steps draw their noise independently, their losses add up, so the PLD of a run is the convolution of the steps' PLDs. All the error comes from storing each PLD on a grid of losses, with a spacing $h$ between grid points. Rounding every loss up can only raise $\delta$, so the $\varepsilon$ it gives is an upper bound, the **pessimistic** side. Rounding down gives a lower bound, the optimistic side. Each rounding moves a step's loss by at most $h$. So after a number $T$ of steps the two values are at most $Th$ apart, a width known in advance.

## When you meet it

You meet a PLD accountant every time you train with Opacus, whose default `"prv"` is the privacy loss random variable (PRV) accountant of Gopi, Lee and Wutschitz (2021). It rounds each loss to the nearest grid point, so the errors partly cancel, then adds a proven error bound and reports the upper value. Google's `dp_accounting` has `PLDAccountant`, which by default rounds up with **connect the dots**, the tightest pessimistic rounding a grid allows.

@tbl-num-dpsgd compares a PLD with Rényi differential privacy (RDP) and with the central limit theorem (CLT) estimate of Gaussian differential privacy (GDP). The run is the part's example of private stochastic gradient descent (SGD), DP-SGD, with sampling rate 0.01 and noise multiplier 1.

| Steps | PLD upper | PLD lower | RDP | GDP CLT |
| --- | --- | --- | --- | --- |
| 1 | 0.1995 | 0.1994 | 0.9555 | 0.0368 |
| 1000 | 1.8332 | 1.8232 | 2.1014 | 1.6177 |
| 10000 | 6.2377 | 6.1377 | 6.7127 | 6.0071 |

: $\varepsilon$ at $\delta = 10^{-5}$, for the running example. The PLD columns round up and down on the grid $h = 10^{-5}$, without connect the dots. {#tbl-num-dpsgd}

RDP gives a valid but loose bound ([Rényi DP](01-renyi-dp.md)). The CLT estimate falls below the PLD's lower bound at every length, so it claims a guarantee the run lacks ([f-DP and Gaussian DP](04-gaussian-dp.md)).

## How to use it

In `dp_accounting` you describe the run as an event, compose it, and read off $\varepsilon$ at your $\delta$. The method `get_delta` runs the other way, from a claimed $\varepsilon$ to a bound on its $\delta$.

```python
from dp_accounting import GaussianDpEvent, PoissonSampledDpEvent, SelfComposedDpEvent
from dp_accounting.pld import PLDAccountant
from opacus.accountants import PRVAccountant

run = SelfComposedDpEvent(PoissonSampledDpEvent(0.01, GaussianDpEvent(1.0)), 10_000)
pld = PLDAccountant(value_discretization_interval=1e-4).compose(run)  # h, 10x the table's
print(f"{pld.get_epsilon(1e-5):.4f}")  # 6.1877
print(f"{pld.get_delta(6.24):.1e}")  # 8.5e-06

prv = PRVAccountant()
prv.history = [(1.0, 0.01, 10_000)]  # (noise multiplier, sample rate, steps)
print(f"{prv.get_epsilon(delta=1e-5):.4f}")  # 6.1980
```

Both answers are upper bounds inside the bracket of @tbl-num-dpsgd, so you can report either. Connect the dots reaches 6.1877 on a grid ten times coarser than the table's. And `get_delta` lets a reviewer confirm the claim $(6.24, 10^{-5})$, since it bounds the run's slack at that $\varepsilon$ by $8.5\times10^{-6}$.

Each library has one accuracy setting. In `dp_accounting` it is the grid spacing `value_discretization_interval`, and a tenfold finer grid narrows the bracket tenfold for about ten times the memory. In Opacus it is `eps_error`, the error allowed in $\varepsilon$. The error grows with the steps, so set either for your longest run.

## What goes wrong

The commonest mistake is reporting the wrong side of the bracket. Only the pessimistic value, or the PRV accountant's upper value, is a guarantee. The optimistic value and the bracket's midpoint are estimates.

Under add-or-remove neighbours, a subsampled step has a different PLD for adding a record than for removing one, and neither is larger in general. So an accountant must compose both and report the larger, as `dp_accounting` does by default. Opacus's PRV accountant models only removal, the larger direction in the running example. Both assume Poisson sampling, which shuffled or fixed-size batches break ([Amplification by subsampling](05-subsampling.md#sec-sub-schemes)).

A run whose steps or noise are chosen from the outputs needs a filter, which enforces a budget fixed in advance ([Adaptive composition](08-adaptive-composition.md)).

## References

- Gopi, Lee & Wutschitz, "Numerical Composition of Differential Privacy" (NeurIPS 2021): the PRV accountant, with explicit error bounds.

**Derivations.** Every result here is derived step by step in [the full edition's chapter](../../book-full/part04-privacy-accounting/06-numerical-accounting.md).
