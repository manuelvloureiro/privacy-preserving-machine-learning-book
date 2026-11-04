# Choosing an accountant

## Why this matters

Report a model protected by differential privacy (DP), with privacy level $\varepsilon$ and slack $\delta$. The code computing $\varepsilon$ is an **accountant**. Different accountants can give very different results for the same run: one reports 6.24, another 24,379.55. @tbl-choose-main compares advanced composition, Rényi differential privacy (RDP) and privacy loss distribution (PLD) accounting for one stochastic gradient descent (SGD) run with DP. Each record is sampled independently with probability $q=0.01$, the sampling rate. The noise multiplier $\sigma$ is the noise standard deviation relative to the clipping norm; here it is 1.

| Steps | Advanced composition | RDP, improved | PLD upper | PLD lower |
| --- | --- | --- | --- | --- |
| 1,000 | 1258.64 | 2.1014 | 1.8332 | 1.8232 |
| 10,000 | 24379.55 | 6.7127 | 6.2377 | 6.1377 |

: One DP-SGD run, $q = 0.01$ and $\sigma = 1$, under three accountants: $\varepsilon$ at $\delta = 10^{-5}$. The PLD columns round the privacy loss up and down to a grid of width $10^{-5}$, so they bracket the true value. {#tbl-choose-main}

The true $\varepsilon$ lies between the PLD bounds. Advanced composition gives a bound too loose to use here. RDP is tighter but still overstates $\varepsilon$ by 14.6% for the shorter run and 7.6% for the longer one. **Report the tightest upper bound whose assumptions match the implementation, with enough detail for independent recomputation.**

## Which accountant {#sec-choose-decision}

An accountant is tight only when its per-step model matches the mechanism. Choose it accordingly (@tbl-choose-decision).

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

The central-limit approximation gives 1.6177 for 1,000 steps, but it is not a bound. This falls below the PLD's certified lower bound of 1.8232, so it claims a guarantee the run does not satisfy. Opacus's `"gdp"` accountant uses this approximation; do not report it. Opacus's default PRV accountant is safe.

## The inverse problem: noise for a target ε {#sec-choose-inverse}

Usually, fix target privacy level $\varepsilon$, batch size and epoch count, then solve for the smallest acceptable noise multiplier $\sigma$. A looser accountant requires more noise to meet the same target. For a target $\varepsilon=3$, RDP requires noise multiplier 1.662 on the 10,000-step run. PLD requires 1.584. Calibrate with the accountant used for reporting. Opacus's `get_noise_multiplier` defaults to `"rdp"`, so specify `accountant="prv"` when needed ([Calibrate and account a DP-SGD run](10-calibrate-dp-sgd.md)).

## Choosing δ {#sec-choose-delta}

Let $n$ denote the number of people in the protected unit. For user-level privacy, count users rather than records. Set $\delta$ using the book's rule: at most $1/n^{1.1}$, rounded down to a power of ten ([How small δ must be](../part03-dp-foundations/01-definition.md#sec-dpdef-small-delta)). The run in @tbl-choose-main has 10,000 steps. The sampling rate is 0.01, and the dataset has 50,000 records. Together, these settings give 100 epochs. The rule sets $\delta=10^{-6}$, so the table's $10^{-5}$ is too large. The US National Institute of Standards and Technology (NIST) recommends the stricter $1/n^2$; also report $\varepsilon$ at that value.

Publish the named accountant's upper bound, rounded up; it does not certify the lower bound. [Write the statement](../part05-dp-ml/12-release-governance.md#sec-relgov-statement) gives a complete privacy statement.

## Matching the accountant to the code {#sec-choose-match}

An accountant's result is valid only if the code satisfies its assumptions. Many papers reported Poisson-sampling bounds but trained with fixed-size shuffled batches, a different algorithm (Ponomareva et al. 2023). Check the implementation against @tbl-choose-checklist before reporting.

| Item | The accountant assumes | Check in the code | If it does not match |
| --- | --- | --- | --- |
| Sampler | Poisson sampling at rate $q$ | Do batch sizes vary from step to step? | The sampler's row of [Subsampling](05-subsampling.md#sec-sub-schemes): fixed-size batches at $\sigma/2$; shuffle and partition unamplified, for zero-out neighbours |
| Neighbours | Add or remove one record | Which relation does the claim state? | Replace-one costs more per step; for Poisson sampling, compose the same steps in `PLDAccountant(NeighboringRelation.REPLACE_ONE)` |
| Unit | One record | Can one person give several records? | Clip and sample per user, or pay for the group ([User-level DP](../part05-dp-ml/07-user-level.md)) |
| Tuning, adaptivity and steps | Settings and steps fixed in advance | Tuning on private data, early stopping, restarts? | Account for the tuning runs ([Hyperparameter tuning](../part05-dp-ml/03-hyperparameter-tuning.md)); report the planned maximum; keep the accountant's state across restarts |
| Other data | Only the training rows are private | Pretraining, fitted preprocessing, validation rows? | Name them; preprocessing fitted on private rows needs its own budget |

: A checklist for matching the accountant to the training code. {#tbl-choose-checklist}

The sampler has the largest effect. At $\delta=10^{-5}$, `dp_accounting`'s PLD gives 6.19 for Poisson sampling and 43.37 for fixed-size batches. Shuffle and partition give 91.82. The last bound applies only to zero-out neighbours ([Subsampling](05-subsampling.md#sec-sub-dpsgd)).

Use the same checklist when reviewing another claim; [Review someone else's claim](../part05-dp-ml/12-release-governance.md#sec-relgov-review) gives the review order.

## Checking an accountant {#sec-choose-check}

Test accountants against a known exact case. The sampling rate $q$ is the probability that a record enters a batch; at $q=1$, every record enters. Each step is then a full Gaussian mechanism, whose composition is exact ([f-DP and Gaussian DP](04-gaussian-dp.md)). Ten steps at noise multiplier 1 are $(17.8566,10^{-5})$-DP. An upper bound must be at least 17.8566, and a PLD bracket must contain it. Also verify that $\varepsilon$ does not decrease with more steps or increase with more noise.

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

Each PLD result falls inside the table's bracket. RDP results are slightly higher because the table also evaluates fractional orders. When libraries disagree, compare their assumptions and calculations. RDP differences can result from using different orders; at order 10, that accounts for 0.3612 ([RDP conversions](02-rdp-conversions.md)). Non-overlapping PLD brackets may indicate a library bug. A factor-of-two difference often indicates different neighbouring relations.

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
