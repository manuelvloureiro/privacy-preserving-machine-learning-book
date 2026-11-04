# Numerical accounting with privacy loss distributions

## What it is {#sec-num-pld}

A privacy loss distribution (PLD) accountant numerically computes the privacy level $\varepsilon$ of a complete differential privacy (DP) run. Compare a dataset with a neighbour differing in one record. The **privacy loss** $L$ is the logarithm of the output likelihood ratio between these datasets. Because the output is random, $L$ has a distribution called the PLD. Outputs with loss above $\varepsilon$ contribute to slack $\delta$; the minimum slack is given below, where $(\cdot)_+$ sets negative values to 0:
$$\delta(\varepsilon) = \mathbb E\big[(1 - e^{\varepsilon - L})_+\big].$$

With independent noise, step losses add, so the run's PLD is the convolution of the step PLDs. Numerical error comes from representing losses on a grid with spacing $h$. Rounding losses up can only increase $\delta$, so the resulting $\varepsilon$ is an upper bound, the **pessimistic** value. Rounding down gives the optimistic lower bound. Each step's loss shifts by at most $h$, so after $T$ steps the two bounds differ by at most $Th$.

## When you meet it

Opacus uses a PLD accountant by default for training: `"prv"` is the privacy loss random variable (PRV) accountant of Gopi, Lee and Wutschitz (2021). It rounds each loss to a grid point, adds a proven error bound and reports the upper value. Google's `dp_accounting` provides `PLDAccountant`; its default **connect the dots** method gives the tightest pessimistic rounding on a grid.

@tbl-num-dpsgd compares PLD accounting with Rényi differential privacy (RDP) and the central limit theorem (CLT) estimate of Gaussian differential privacy (GDP). The example is stochastic gradient descent (SGD) with differential privacy, per-step clipping and noise, using sampling rate 0.01 and noise multiplier 1.

| Steps | PLD upper | PLD lower | RDP | GDP CLT |
| --- | --- | --- | --- | --- |
| 1 | 0.1995 | 0.1994 | 0.9555 | 0.0368 |
| 1000 | 1.8332 | 1.8232 | 2.1014 | 1.6177 |
| 10000 | 6.2377 | 6.1377 | 6.7127 | 6.0071 |

: $\varepsilon$ at $\delta = 10^{-5}$, for the running example. The PLD columns round up and down on the grid $h = 10^{-5}$, without connect the dots. {#tbl-num-dpsgd}

RDP gives a valid but loose bound ([Rényi DP](01-renyi-dp.md)). At every run length, the CLT estimate is below the PLD lower bound and therefore overstates privacy ([f-DP and Gaussian DP](04-gaussian-dp.md)).

## How to use it

In `dp_accounting`, compose the run as an event and request $\varepsilon$ at the desired $\delta$. The `get_delta` method instead returns a bound on $\delta$ for a chosen $\varepsilon$.

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

Both answers are upper bounds within @tbl-num-dpsgd's bracket. A reviewer can use `get_delta` to verify the $(6.24,10^{-5})$ claim: it bounds the run's slack at that $\varepsilon$ by $8.5\times10^{-6}$.

Each library exposes one accuracy setting. In `dp_accounting`, `value_discretization_interval` sets grid spacing; making the grid ten times finer narrows the bracket tenfold and uses about ten times more memory. In Opacus, `eps_error` sets the allowed error in $\varepsilon$. Since error grows with the number of steps, configure either setting for the longest planned run.

## What goes wrong

The main reporting error is to use the wrong side of the bracket. Only the pessimistic value, or the PRV accountant's upper value, is a guarantee; the optimistic value and midpoint are estimates.

Under add/remove adjacency, a subsampled step has separate PLDs for adding and removing a record; neither direction is always larger. An accountant must compose both and report the larger, as `dp_accounting` does by default. Opacus's PRV accountant models removal only, which is the larger direction in this example. Both libraries assume Poisson sampling; shuffled or fixed-size batches violate that assumption ([Amplification by subsampling](05-subsampling.md#sec-sub-schemes)).

For a run that adapts its steps or noise to earlier outputs, use a filter ([Adaptive composition](08-adaptive-composition.md)).

## References

- Gopi, Lee & Wutschitz, "Numerical Composition of Differential Privacy" (NeurIPS 2021): the PRV accountant, with explicit error bounds.

**Derivations.** Every result here is derived step by step in [the full edition's chapter](../../book-full/part04-privacy-accounting/06-numerical-accounting.md).
