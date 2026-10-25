# Choosing an accountant

## Why this matters

You have trained a model with differential privacy (DP), and a paper or a model card needs one sentence. It says that the model is DP with a privacy level $\varepsilon$ and a small slack $\delta$. The same run can be reported as 6.24 or as 24,379.55, depending on the procedure that computed it, and nobody can check either value without knowing what that procedure assumed.

The run in @tbl-choose-main is private stochastic gradient descent (SGD), called DP-SGD. Each record joins each batch with probability $q = 0.01$, the sampling rate, and the noise multiplier $\sigma$, the noise as a multiple of the clipping norm, is 1. The run has one true $\varepsilon$ that nobody can compute exactly, and a procedure that bounds or estimates it is an **accountant**. The table compares advanced composition, Rényi differential privacy (RDP) and the privacy loss distribution (PLD).

| Steps | Advanced composition | RDP, improved | PLD upper | PLD lower |
| --- | --- | --- | --- | --- |
| 1,000 | 1258.64 | 2.1014 | 1.8332 | 1.8232 |
| 10,000 | 24379.55 | 6.7127 | 6.2377 | 6.1377 |

: One DP-SGD run, $q = 0.01$ and $\sigma = 1$, under three accountants: $\varepsilon$ at $\delta = 10^{-5}$. The PLD columns round the privacy loss up and down to a grid of width $10^{-5}$, so they bracket the true value. {#tbl-choose-main}

The PLD's lower column is a proved lower bound, and the other columns are proved upper bounds. Adding up each step's guarantee is useless after a few steps. RDP is far closer, but it still overstates $\varepsilon$, by 14.6% on the shorter run and 7.6% on the longer one. The $\delta$ of $10^{-5}$ is only for comparison (@sec-choose-delta). So the rule of this chapter is short. **Report the tightest upper bound whose assumptions hold for the code you ran, and give enough detail that someone else can recompute it.**

## Which accountant {#sec-choose-decision}

An accountant is tight, close to the true $\varepsilon$, only when what it keeps of each step fits the mechanism, so the mechanism decides (@tbl-choose-decision).

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

The CLT is missing from @tbl-choose-main because it is an approximation with no proof. After 1,000 steps of the same run it gives 1.6177. That is below the PLD's certified lower value of 1.8232, so a paper that quotes it claims a guarantee the run does not have. Opacus's default PRV accountant is safe, but its `"gdp"` accountant is this CLT, so never report it.

## The inverse problem: noise for a target ε {#sec-choose-inverse}

Usually you fix the budget, batch size and epochs, and ask for the smallest $\sigma$ that meets them. A looser accountant overstates $\varepsilon$, so it asks for more noise. For a target of 3 on the longer run, RDP needs a noise multiplier of 1.662, while the PLD needs only 1.584. So calibrate with the accountant you will report, and pass `accountant="prv"` to Opacus's `get_noise_multiplier`, whose default is `"rdp"`. [Calibrate and account a DP-SGD run](10-calibrate-dp-sgd.md) does this for a real run.

## Choosing δ {#sec-choose-delta}

Roughly, treat $\delta$ as the chance that the guarantee fails for someone, possibly badly. With $n$ people, a mechanism that publishes one randomly chosen record in full meets a $\delta$ of about $1/n$ with an $\varepsilon$ of zero, yet it exposes someone completely. So $\delta$ must sit well below $1/n$.

Following the convention of Ponomareva et al. (2023), this book chooses
$$\delta \le \frac{1}{n^{1.1}}, \quad \text{rounded down to a power of ten.}$$ {#eq-choose-delta}
Count $n$ in the unit the claim protects, which at the user level means users, not records.

For example, put the longer run of @tbl-choose-main on 50,000 records, which makes it 100 epochs. Then $1/n^{1.1}$ is $6.8\times10^{-6}$, which rounds down to $10^{-6}$. The comparison point $10^{-5}$ is half of $1/n$, so it fails the rule. The US National Institute of Standards and Technology (NIST) recommends the stricter $1/n^2$, so report $\varepsilon$ at both values of $\delta$.

## The statement {#sec-choose-report}

Ponomareva et al. (2023) and NIST ask for the items of @tbl-choose-statement. The table fills them in for the run on 50,000 records, with the full edition's PLD, whose upper value is the claim.

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

Round an upper bound up, never to the nearest value. The PLD's upper value at $10^{-6}$ is 6.9574, so the claim is 6.96, because a lower value is one the accountant has not certified. Publish the value of the accountant you name. `dp_accounting`'s PLD gives 6.9074 on its own grid (@sec-choose-check), so a report made with it claims 6.91. [Calibrate and account a DP-SGD run](10-calibrate-dp-sgd.md#sec-calib-statement) prints such a statement from code.

## Matching the accountant to the code {#sec-choose-match}

An accountant trusts that the code does what its theorem assumes. Yet many papers have reported the Poisson-sampling number while training with shuffled fixed-size batches, a statement about a different algorithm (Ponomareva et al. 2023). So check each row of @tbl-choose-checklist against the code before you report.

| Item | The accountant assumes | Check in the code | If it does not match |
| --- | --- | --- | --- |
| Sampler | Poisson sampling at rate $q$ | Do batch sizes vary from step to step? | The sampler's row of [Subsampling](05-subsampling.md#sec-sub-schemes): fixed-size batches at $\sigma/2$; shuffle and partition unamplified, for zero-out neighbours |
| Neighbours | Add or remove one record | Which relation does the claim state? | Replace-one costs more per step; for Poisson sampling, compose the same steps in `PLDAccountant(NeighboringRelation.REPLACE_ONE)` |
| Unit | One record | Can one person give several records? | Clip and sample per user, or pay for the group ([User-level DP](../part05-dp-ml/07-user-level.md)) |
| Tuning, adaptivity and steps | Settings and steps fixed in advance | Tuning on private data, early stopping, restarts? | Account for the tuning runs ([Hyperparameter tuning](../part05-dp-ml/03-hyperparameter-tuning.md)); report the planned maximum; keep the accountant's state across restarts |
| Other data | Only the training rows are private | Pretraining, fitted preprocessing, validation rows? | Name them; preprocessing fitted on private rows needs its own budget |

: A checklist for matching the accountant to the training code. {#tbl-choose-checklist}

The sampler costs the most. On the run above, Google's `dp_accounting` library gives 6.19 at $10^{-5}$ with its PLD. Fixed-size batches give 43.37, accounted at $\sigma/2$. Shuffle and partition gives 91.82 over the 100 epochs, and only for zero-out neighbours ([Subsampling](05-subsampling.md#sec-sub-dpsgd)).

## Reviewing someone else's claim {#sec-choose-review}

A reviewer works in this order.

1. Ask for the inputs of @tbl-choose-statement, and reject a claim that cannot be recomputed.
2. Check $\delta$ against @eq-choose-delta, with $n$ counted in the claim's unit.
3. Recompute $\varepsilon$ with a different library, and expect agreement within a few percent. A PLD accountant's `get_delta` at the claimed $\varepsilon$ gives the $\delta$ at which it holds. A claim well above your tight value is valid but loose, and was usually calibrated with Opacus's default RDP accountant.
4. Read the training code against @tbl-choose-checklist, starting with the sampler.
5. Ask what the claim leaves out, such as tuning runs, pretraining data and preprocessing fitted on the private rows.

[Release and governance](../part05-dp-ml/12-release-governance.md#sec-relgov-review) runs these steps on a made-up claim and lists what fails.

For example, a team claims $(2, 10^{-5})$-DP for a model trained on one million users. Its batches hold 1,000 users, so $q = 0.001$ and 10 epochs are 10,000 steps, with a noise multiplier of 0.65 and a loader that shuffles and takes fixed batches. This snippet does steps 2 and 3, then accounts for the loader as it ran.

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

The arithmetic holds up, because the claim's own parameters give 1.72 at its $\delta$. Its $\delta$ is the problem, since it is a hundred times the rule's $10^{-7}$, at which the same run gives 2.84. The loader then voids both numbers, because the Poisson accountant assumed a sampler that the code never ran. What the code did run is certified only at 36.45, and only for zero-out neighbours, so the review asks for a Poisson loader and a $\delta$ of $10^{-7}$.

## Checking an accountant {#sec-choose-check}

An accountant is code, so test it on known answers. Setting $q = 1$ turns each step into a plain Gaussian mechanism whose composition is exact ([f-DP and Gaussian DP](04-gaussian-dp.md)). Ten steps at $\sigma = 1$ are then exactly $(17.8566, 10^{-5})$-DP. Any upper bound must lie above that, and a PLD bracket on the same steps must contain it. Also, $\varepsilon$ must never fall as the steps grow or rise as the noise grows, and no upper bound may fall below the PLD's lower end.

Then compare two maintained libraries, and pin their versions. This snippet recomputes the run of @tbl-choose-main with Google's `dp_accounting`.

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

The libraries agree. Each PLD value lies inside the full edition's bracket, and each RDP value sits just above the full edition's, which also tries fractional orders. When libraries do disagree, there are three usual causes. Two RDP accountants that differ by an amount depending only on the order, 0.3612 at order 10, use different conversion formulas from RDP to $(\varepsilon, \delta)$ ([RDP conversions](02-rdp-conversions.md)). Brackets that do not overlap mean a bug in one library, and values about a factor of two apart often mean different neighbouring relations.

## Cheat sheet

| Question | Default | Check |
| --- | --- | --- |
| Accountant for DP-SGD | PLD or PRV, upper value | RDP lies above it; never report the CLT |
| $\delta$ | $1/n^{1.1}$ rounded down, $n$ counted in the claim's unit | Also report $\varepsilon$ at $1/n^2$ |
| Fixed-size batches | Poisson accountant at $\sigma/2$ (add/remove); replace-one: see [Subsampling](05-subsampling.md#sec-sub-schemes) | Poisson batch sizes vary from step to step |
| Shuffle and partition | One Gaussian step per epoch, zero-out neighbours | Better, switch to Opacus's Poisson loader |

: Reporting and reviewing an $\varepsilon$ claim. {#tbl-choose-cheat}

**Rules of thumb**

- Never publish $\varepsilon$ without $\delta$, the unit, the relation, the accountant and its version, and the inputs to recompute it.

## References

- Ponomareva et al., "How to DP-fy ML: A Practical Guide to Machine Learning with Differential Privacy" (JAIR 2023): matching the accounting to the pipeline, and what to report.
- Near, Darais, Lefkovitz & Howarth, "Guidelines for Evaluating Differential Privacy Guarantees" (NIST SP 800-226, 2025): what an evaluator will ask of your claim.

**Derivations.** Every result here is derived step by step in [the full edition's chapter](../../book-full/part04-privacy-accounting/09-choosing-an-accountant.md).
