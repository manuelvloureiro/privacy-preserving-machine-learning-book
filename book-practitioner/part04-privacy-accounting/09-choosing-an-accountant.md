# Choosing an accountant

## Why this matters

You report a model trained with differential privacy (DP) with a privacy level $\varepsilon$ and a small slack $\delta$. The code that computes $\varepsilon$ is an **accountant**, and the same run can get 6.24 from one accountant and 24,379.55 from another. @tbl-choose-main runs three accountants on one run of private stochastic gradient descent (SGD), called DP-SGD. They are advanced composition, Rényi differential privacy (RDP) and the privacy loss distribution (PLD). Each record joins each batch with probability $q = 0.01$, the sampling rate. The noise multiplier $\sigma$, the noise as a multiple of the clipping norm, is 1.

| Steps | Advanced composition | RDP, improved | PLD upper | PLD lower |
| --- | --- | --- | --- | --- |
| 1,000 | 1258.64 | 2.1014 | 1.8332 | 1.8232 |
| 10,000 | 24379.55 | 6.7127 | 6.2377 | 6.1377 |

: One DP-SGD run, $q = 0.01$ and $\sigma = 1$, under three accountants: $\varepsilon$ at $\delta = 10^{-5}$. The PLD columns round the privacy loss up and down to a grid of width $10^{-5}$, so they bracket the true value. {#tbl-choose-main}

The true $\varepsilon$ lies between the two PLD columns, because the lower PLD column is a lower bound. Advanced composition, which adds up the guarantees of the steps, is useless here. RDP is much closer, but it still overstates $\varepsilon$, by 14.6% on the shorter run and by 7.6% on the longer one. This gives the rule of the chapter. **Report the tightest upper bound whose assumptions hold for the code you ran, and give enough detail that someone else can recompute it.**

## Which accountant {#sec-choose-decision}

An accountant is tight only when its summary of each step fits the mechanism. So let the mechanism choose the accountant (@tbl-choose-decision).

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

The GDP CLT is an approximation, not a bound. After 1,000 steps it gives 1.6177, which is below the certified lower value of the PLD, 1.8232. So it claims a guarantee that the run does not have. Opacus's `"gdp"` accountant is this CLT, so do not report it. The default PRV accountant of Opacus is safe.

## The inverse problem: noise for a target ε {#sec-choose-inverse}

Usually you fix the target $\varepsilon$, the batch size and the number of epochs, and you ask for the smallest $\sigma$ that meets them. A looser accountant overstates $\varepsilon$, so it asks for more noise. For a target of 3 on the 10,000-step run, RDP needs a noise multiplier of 1.662 and the PLD only 1.584. So calibrate with the accountant that you will report. Opacus's `get_noise_multiplier` uses `"rdp"` by default, so give it `accountant="prv"` ([Calibrate and account a DP-SGD run](10-calibrate-dp-sgd.md)).

## Choosing δ {#sec-choose-delta}

Choose $\delta$ by the rule of this book, which for $n$ people is at most $1/n^{1.1}$, rounded down to a power of ten ([How small δ must be](../part03-dp-foundations/01-definition.md#sec-dpdef-small-delta)). Count $n$ in the unit that the claim protects, which at the user level means users, not records. For example, run the 10,000 steps of @tbl-choose-main on 50,000 records. At $q = 0.01$, that is 100 epochs. For 50,000 records the rule gives $10^{-6}$, so the $10^{-5}$ of the table fails it. The US National Institute of Standards and Technology (NIST) recommends the stricter $1/n^2$, so report $\varepsilon$ there too.

Publish the upper value of the accountant that you name, rounded up, because the accountant does not certify a lower value. [Write the statement](../part05-dp-ml/12-release-governance.md#sec-relgov-statement) shows the full privacy statement that carries these numbers.

## Matching the accountant to the code {#sec-choose-match}

The number of an accountant holds only if the code does what its theorem assumes. Many papers reported the number for Poisson sampling but trained with shuffled batches of fixed size, a different algorithm (Ponomareva et al. 2023). So check the code against @tbl-choose-checklist before you report.

| Item | The accountant assumes | Check in the code | If it does not match |
| --- | --- | --- | --- |
| Sampler | Poisson sampling at rate $q$ | Do batch sizes vary from step to step? | The sampler's row of [Subsampling](05-subsampling.md#sec-sub-schemes): fixed-size batches at $\sigma/2$; shuffle and partition unamplified, for zero-out neighbours |
| Neighbours | Add or remove one record | Which relation does the claim state? | Replace-one costs more per step; for Poisson sampling, compose the same steps in `PLDAccountant(NeighboringRelation.REPLACE_ONE)` |
| Unit | One record | Can one person give several records? | Clip and sample per user, or pay for the group ([User-level DP](../part05-dp-ml/07-user-level.md)) |
| Tuning, adaptivity and steps | Settings and steps fixed in advance | Tuning on private data, early stopping, restarts? | Account for the tuning runs ([Hyperparameter tuning](../part05-dp-ml/03-hyperparameter-tuning.md)); report the planned maximum; keep the accountant's state across restarts |
| Other data | Only the training rows are private | Pretraining, fitted preprocessing, validation rows? | Name them; preprocessing fitted on private rows needs its own budget |

: A checklist for matching the accountant to the training code. {#tbl-choose-checklist}

The sampler has the largest effect. Under Poisson sampling, the PLD of `dp_accounting` gives 6.19 at $\delta = 10^{-5}$, but batches of fixed size raise this to 43.37. Shuffle and partition raise it to 91.82, and that number holds only for zero-out neighbours ([Subsampling](05-subsampling.md#sec-sub-dpsgd)).

A review of someone else's claim also uses this checklist, in the order that [Review someone else's claim](../part05-dp-ml/12-release-governance.md#sec-relgov-review) gives.

## Checking an accountant {#sec-choose-check}

An accountant is code, so test it on known answers. With sampling rate $q = 1$, each step is a plain Gaussian mechanism, and the composition of such steps is exact ([f-DP and Gaussian DP](04-gaussian-dp.md)). Ten such steps at noise multiplier 1 are exactly $(17.8566, 10^{-5})$-DP. So an upper bound must be above 17.8566, and a PLD bracket must contain it. Also, $\varepsilon$ must not decrease when you add steps, and it must not increase when you add noise.

Then compare two maintained libraries at pinned versions. This snippet recomputes @tbl-choose-main with Google's `dp_accounting`.

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

Each PLD value is inside the bracket of the table. Each RDP value is just above the value in the table, because the table also tries fractional orders. When libraries disagree, look for three causes. If two RDP values differ by an amount that depends only on the order, such as 0.3612 at order 10, the libraries use different conversions ([RDP conversions](02-rdp-conversions.md)). If two PLD brackets do not overlap, one library has a bug. A factor of two often comes from different neighbouring relations.

## Cheat sheet

| Question | Default | Check |
| --- | --- | --- |
| Accountant for DP-SGD | PLD or PRV, upper value | RDP lies above it; never report the CLT |
| $\delta$ | The rule of [How small δ must be](../part03-dp-foundations/01-definition.md#sec-dpdef-small-delta), $n$ counted in the claim's unit | Also report $\varepsilon$ at $1/n^2$ |
| Fixed-size batches | Poisson accountant at $\sigma/2$ (add/remove); replace-one: see [Subsampling](05-subsampling.md#sec-sub-schemes) | Poisson batch sizes vary from step to step |
| Shuffle and partition | One Gaussian step per epoch, zero-out neighbours | Better, switch to Opacus's Poisson loader |

: Choosing and checking an accountant. {#tbl-choose-cheat}

## References

- Ponomareva et al., "How to DP-fy ML: A Practical Guide to Machine Learning with Differential Privacy" (JAIR 2023): matching the accounting to the pipeline, and what to report.
- Near, Darais, Lefkovitz & Howarth, "Guidelines for Evaluating Differential Privacy Guarantees" (NIST SP 800-226, 2025): what an evaluator will ask of your claim.

**Derivations.** Every result here is derived step by step in [the full edition's chapter](../../book-full/part04-privacy-accounting/09-choosing-an-accountant.md).
