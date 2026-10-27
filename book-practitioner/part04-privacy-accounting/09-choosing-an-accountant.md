# Choosing an accountant

## Why this matters

A model trained with differential privacy (DP) is reported with a privacy level $\varepsilon$ and a small slack $\delta$. The procedure that computes $\varepsilon$ is an **accountant**, and the same run can get 6.24 or 24,379.55 depending on which one you use. @tbl-choose-main shows three, advanced composition, Rényi differential privacy (RDP) and the privacy loss distribution (PLD), on one run of private stochastic gradient descent (SGD), called DP-SGD. Each record joins each batch with probability $q = 0.01$, the sampling rate, and the noise multiplier $\sigma$, the noise as a multiple of the clipping norm, is 1.

| Steps | Advanced composition | RDP, improved | PLD upper | PLD lower |
| --- | --- | --- | --- | --- |
| 1,000 | 1258.64 | 2.1014 | 1.8332 | 1.8232 |
| 10,000 | 24379.55 | 6.7127 | 6.2377 | 6.1377 |

: One DP-SGD run, $q = 0.01$ and $\sigma = 1$, under three accountants: $\varepsilon$ at $\delta = 10^{-5}$. The PLD columns round the privacy loss up and down to a grid of width $10^{-5}$, so they bracket the true value. {#tbl-choose-main}

Only the PLD's lower column is a lower bound, so the true $\varepsilon$ lies between the two PLD columns. Advanced composition, which adds up each step's guarantee, is useless here, and RDP still overstates $\varepsilon$, by 14.6% on the shorter run and 7.6% on the longer one. Hence the chapter's rule. **Report the tightest upper bound whose assumptions hold for the code you ran, and give enough detail that someone else can recompute it.**

## Which accountant {#sec-choose-decision}

An accountant is tight only when the summary it keeps of each step fits the mechanism, so the mechanism decides (@tbl-choose-decision).

| Mechanism | Report with | Avoid |
| --- | --- | --- |
| Gaussian noise, no subsampling | Gaussian differential privacy (GDP), converted exactly ([f-DP and Gaussian DP](04-gaussian-dp.md)) | Per-step composition |
| DP-SGD with Poisson sampling | PLD, or the privacy loss random variable (PRV) accountant ([Numerical accounting](06-numerical-accounting.md)), with the PLD's lower value as the error bar ([its code](10-calibrate-dp-sgd.md#sec-calib-crosscheck)) | Zero-concentrated DP (zCDP); the GDP central limit theorem (CLT) |
| Pure-DP steps (Laplace, randomized response) | PLD, exact when the privacy loss takes few values; Laplace with scale $b$ is `LaplaceDpEvent(b/Δ)`, $\Delta$ its $\ell_1$ sensitivity | Advanced composition |
| Many statistical releases | A zCDP ledger ([Concentrated DP](03-concentrated-dp.md)) or PLD | Adding up each release's $\varepsilon$ |
| Noise or budget chosen during the run | A Rényi filter ([Adaptive composition](08-adaptive-composition.md)) | Plugging realized values into a fixed theorem |
| Correlated noise across steps: matrix factorization, or the follow the regularized leader (FTRL) optimizer on tree sums, DP-FTRL | The mechanism's own accountant ([Matrix mechanisms](../part05-dp-ml/04-matrix-mechanisms.md)) | The Poisson DP-SGD accountant |
| A shuffler of locally randomized reports, or only the last model released | [Shuffling and iteration](07-shuffling-iteration.md), with its conditions; a loader that shuffles is a sampler (@tbl-choose-checklist) | The Poisson accountant for a shuffler; a last-model claim for non-convex training |

: Which accountant to report for which mechanism. {#tbl-choose-decision}

The GDP CLT is an approximation, not a bound. After 1,000 steps it gives 1.6177, below the PLD's certified lower value of 1.8232, a guarantee the run does not have. Opacus's `"gdp"` accountant is this CLT, so never report it; its default PRV accountant is safe.

## The inverse problem: noise for a target ε {#sec-choose-inverse}

Usually you fix the target $\varepsilon$, batch size and epochs, and ask for the smallest $\sigma$ that meets them. A looser accountant overstates $\varepsilon$, so it asks for more noise. For a target of 3 on the 10,000-step run, RDP needs 1.662 and the PLD only 1.584. So calibrate with the accountant you will report, passing `accountant="prv"` to Opacus's `get_noise_multiplier`, whose default is `"rdp"` ([Calibrate and account a DP-SGD run](10-calibrate-dp-sgd.md)).

## Choosing δ {#sec-choose-delta}

Treat $\delta$ roughly as the chance that the guarantee fails for someone, possibly badly. On a dataset of $n$ people, publishing one random record in full meets a $\delta$ of about $1/n$ with $\varepsilon$ zero, yet exposes someone completely. So $\delta$ must sit well below $1/n$. This book follows Ponomareva et al. (2023) and chooses
$$\delta \le \frac{1}{n^{1.1}}, \quad \text{rounded down to a power of ten.}$$ {#eq-choose-delta}
Count $n$ in the unit the claim protects, which at the user level means users, not records.

For example, run the 10,000 steps of @tbl-choose-main on 50,000 records, which at $q = 0.01$ is 100 epochs. Then $1/n^{1.1}$ is $6.8\times10^{-6}$, which rounds down to $10^{-6}$. The table's $10^{-5}$ is half of $1/n$, so it fails the rule. The US National Institute of Standards and Technology (NIST) recommends the stricter $1/n^2$, so report $\varepsilon$ there too.

## The statement {#sec-choose-report}

@tbl-choose-statement fills in the items that Ponomareva et al. (2023) and NIST ask for, for the run on 50,000 records.

| Item | The run on 50,000 records |
| --- | --- |
| Setting | DP-SGD in the central model, covering every checkpoint and the final model |
| Unit and neighbours | One record per person; neighbours add or remove one record |
| Training | Poisson sampling at $q = 0.01$, 10,000 steps (100 epochs), clipping norm 1, noise multiplier 1 |
| Accountant | PLD, upper value, on the grid $10^{-5}$, with the library's version |
| Guarantee | $(6.96, 10^{-6})$ and, at $1/n^2$, $(9.06, 4.0\times10^{-10})$; the PLD's lower values, 6.8574 and 8.9378, show how far each could still fall |
| Other data | No pretrained weights; no preprocessing fitted on private rows; validation rows held out and not protected |
| Not covered | Nothing: hyperparameters were tuned on a public proxy dataset, so no tuning run touched the private data |

: A filled privacy statement. {#tbl-choose-statement}

Round an upper bound up, because a lower value is one the accountant has not certified. So the PLD's upper value at $10^{-6}$, 6.9574, becomes 6.96. And publish the value of the accountant you name: with `dp_accounting`'s PLD, which gives 6.9074 (@sec-choose-check), the claim is 6.91. [Calibrate and account a DP-SGD run](10-calibrate-dp-sgd.md#sec-calib-statement) prints such a statement from code.

## Matching the accountant to the code {#sec-choose-match}

An accountant's number holds only if the code does what its theorem assumes. Many papers have reported the Poisson-sampling number while training with shuffled fixed-size batches, a different algorithm (Ponomareva et al. 2023). So check @tbl-choose-checklist against the code before you report.

| Item | The accountant assumes | Check in the code | If it does not match |
| --- | --- | --- | --- |
| Sampler | Poisson sampling at rate $q$ | Do batch sizes vary from step to step? | The sampler's row of [Subsampling](05-subsampling.md#sec-sub-schemes): fixed-size batches at $\sigma/2$; shuffle and partition unamplified, for zero-out neighbours |
| Neighbours | Add or remove one record | Which relation does the claim state? | Replace-one costs more per step; for Poisson sampling, compose the same steps in `PLDAccountant(NeighboringRelation.REPLACE_ONE)` |
| Unit | One record | Can one person give several records? | Clip and sample per user, or pay for the group ([User-level DP](../part05-dp-ml/07-user-level.md)) |
| Tuning, adaptivity and steps | Settings and steps fixed in advance | Tuning on private data, early stopping, restarts? | Account for the tuning runs ([Hyperparameter tuning](../part05-dp-ml/03-hyperparameter-tuning.md)); report the planned maximum; keep the accountant's state across restarts |
| Other data | Only the training rows are private | Pretraining, fitted preprocessing, validation rows? | Name them; preprocessing fitted on private rows needs its own budget |

: A checklist for matching the accountant to the training code. {#tbl-choose-checklist}

The sampler costs the most. At $10^{-5}$, the 6.19 that `dp_accounting`'s PLD gives under Poisson sampling becomes 43.37 with fixed-size batches, and 91.82, for zero-out neighbours only, with shuffle and partition ([Subsampling](05-subsampling.md#sec-sub-dpsgd)).

## Reviewing someone else's claim {#sec-choose-review}

A reviewer works in this order.

1. Ask for the inputs of @tbl-choose-statement, and reject a claim that cannot be recomputed.
2. Check $\delta$ against @eq-choose-delta, with $n$ counted in the claim's unit.
3. Recompute $\varepsilon$ with a different library, and expect agreement within a few percent. A PLD accountant's `get_delta` at the claimed $\varepsilon$ gives the $\delta$ at which it holds. A claim well above your value is valid but loose.
4. Read the training code against @tbl-choose-checklist, starting with the sampler.
5. Ask what the claim leaves out, such as tuning runs, pretraining data and preprocessing fitted on the private rows.

For example, a team claims $(2, 10^{-5})$-DP for a model trained on one million users with noise multiplier 0.65. Batches of 1,000 users make the sampling rate $q$ 0.001, and 10 epochs 10,000 steps, but the loader shuffles and takes fixed batches. This snippet does steps 2 and 3, then accounts for the loader as it ran.

```python
from math import floor, log10

from dp_accounting import GaussianDpEvent, PoissonSampledDpEvent, SelfComposedDpEvent
from dp_accounting.pld import PLDAccountant

n, q, sigma, steps, epochs = 1_000_000, 0.001, 0.65, 10_000, 10  # the claim's parameters
delta = 10.0 ** floor(log10(n**-1.1))  # the book's rule for one million users
print(f"{delta:.0e}")  # 1e-07
poisson = SelfComposedDpEvent(PoissonSampledDpEvent(q, GaussianDpEvent(sigma)), steps)
pld = PLDAccountant(value_discretization_interval=1e-4).compose(poisson)
print(f"{pld.get_epsilon(1e-5):.2f}")  # 1.72
print(f"{pld.get_delta(2.0):.1e}")  # 3.0e-06
print(f"{pld.get_epsilon(delta):.2f}")  # 2.84
# What the loader ran: shuffle and partition, zero-out neighbours, no amplification
shuffled = SelfComposedDpEvent(GaussianDpEvent(sigma), epochs)
print(f"{PLDAccountant(value_discretization_interval=1e-4).compose(shuffled).get_epsilon(delta):.2f}")  # 36.45
```

The arithmetic holds up, since the claim's parameters give 1.72 at its $\delta$. But that $\delta$ is a hundred times the rule's $10^{-7}$, at which the run gives 2.84. And the loader voids both numbers, which assume Poisson sampling. What the code ran is certified only at 36.45, for zero-out neighbours, so the review asks for a Poisson loader and a $\delta$ of $10^{-7}$.

## Checking an accountant {#sec-choose-check}

An accountant is code, so test it on known answers. With sampling rate $q = 1$, each step is a plain Gaussian mechanism, whose composition is exact ([f-DP and Gaussian DP](04-gaussian-dp.md)). Ten such steps at noise multiplier 1 are exactly $(17.8566, 10^{-5})$-DP, so an upper bound must lie above 17.8566 and a PLD bracket must contain it. Also, $\varepsilon$ must never fall as steps grow or rise as noise grows.

Then compare two maintained libraries, with pinned versions. This snippet recomputes @tbl-choose-main with Google's `dp_accounting`.

```python
import dp_accounting
from dp_accounting.pld import PLDAccountant
from dp_accounting.rdp import RdpAccountant

step = dp_accounting.PoissonSampledDpEvent(0.01, dp_accounting.GaussianDpEvent(1.0))
run = dp_accounting.SelfComposedDpEvent(step, 10_000)  # q = 0.01, sigma = 1, 10,000 steps
pld = PLDAccountant(value_discretization_interval=1e-4).compose(run)
rdp = RdpAccountant(orders=list(range(2, 65))).compose(run)  # integer orders only
print(f"{pld.get_epsilon(1e-5):.4f} {rdp.get_epsilon(1e-5):.4f}")  # 6.1877 6.7194
print(f"{pld.get_epsilon(1e-6):.4f} {rdp.get_epsilon(1e-6):.4f}")  # 6.9074 7.4869
```

Each PLD value lies inside the table's bracket, and each RDP value sits just above the table's, which also tries fractional orders. When libraries disagree, look for three causes. RDP values that differ by an amount depending only on the order, 0.3612 at order 10, come from different conversions ([RDP conversions](02-rdp-conversions.md)). PLD brackets that do not overlap mean a bug. And a factor of two often means different neighbouring relations.

## Cheat sheet

| Question | Default | Check |
| --- | --- | --- |
| Accountant for DP-SGD | PLD or PRV, upper value | RDP lies above it; never report the CLT |
| $\delta$ | $1/n^{1.1}$ rounded down, $n$ counted in the claim's unit | Also report $\varepsilon$ at $1/n^2$ |
| Fixed-size batches | Poisson accountant at $\sigma/2$ (add/remove); replace-one: see [Subsampling](05-subsampling.md#sec-sub-schemes) | Poisson batch sizes vary from step to step |
| Shuffle and partition | One Gaussian step per epoch, zero-out neighbours | Better, switch to Opacus's Poisson loader |

: Reporting and reviewing an $\varepsilon$ claim. {#tbl-choose-cheat}

## References

- Ponomareva et al., "How to DP-fy ML: A Practical Guide to Machine Learning with Differential Privacy" (JAIR 2023): matching the accounting to the pipeline, and what to report.
- Near, Darais, Lefkovitz & Howarth, "Guidelines for Evaluating Differential Privacy Guarantees" (NIST SP 800-226, 2025): what an evaluator will ask of your claim.

**Derivations.** Every result here is derived step by step in [the full edition's chapter](../../book-full/part04-privacy-accounting/09-choosing-an-accountant.md).
