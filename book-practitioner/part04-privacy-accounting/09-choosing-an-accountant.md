# Choosing an accountant

## Why this matters

You have trained a model with differential privacy (DP), using private stochastic gradient descent (SGD), called DP-SGD ([Amplification by subsampling](05-subsampling.md)). Each record joined each batch with probability $q$, the sampling rate, here 0.01, and the noise multiplier $\sigma$ was 1. After 10,000 steps, a number this chapter writes $k$, you must state the $\varepsilon$ at which the model is $(\varepsilon, 10^{-5})$-DP, in a paper or a model card.

A procedure that bounds or estimates the true $\varepsilon$ is an **accountant**, and on this run the accountants disagree by a factor of over a thousand. That is because they make different kinds of statement. An **upper bound** comes with a proof, so the run really is as private as it reports, while an **approximation** has none and may land below the truth. A certified **lower bound** is no guarantee, but it exposes any reported $\varepsilon$ that is too small. So the rule of this chapter is short. **Report the tightest upper bound whose assumptions hold for the code you ran, and report enough that someone else can recompute it.**

## Every accountant on one run {#sec-choose-compare}

Accountants differ in how much of each step's privacy they keep, filling in the rest with a worst case. From least to most, they are [per-step composition](../part03-dp-foundations/04-composition.md) and [Rényi differential privacy (RDP)](02-rdp-conversions.md), the central limit theorem (CLT) of [Gaussian differential privacy (GDP)](04-gaussian-dp.md), and the [privacy loss distribution (PLD)](06-numerical-accounting.md).

| $k$ | Basic | Advanced | RDP, Mironov | RDP, improved | GDP CLT | PLD upper | PLD lower |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Kind | Upper | Upper | Upper | Upper | Approximation | Upper | Lower |
| 1 | 0.20 | 1.13 | 1.3175 | 0.9555 | 0.0368 | 0.1995 | 0.1994 |
| 1,000 | 839.39 | 1258.64 | 2.5380 | 2.1014 | 1.6177 | 1.8332 | 1.8232 |
| 10,000 | 11296.93 | 24379.55 | 7.4287 | 6.7127 | 6.0071 | 6.2377 | 6.1377 |

: DP-SGD with $q = 0.01$ and $\sigma = 1$: $\varepsilon$ at $\delta = 10^{-5}$ after $k$ steps under each accountant. The PLD columns use the grid $h = 10^{-5}$. {#tbl-choose-main}

The proofs put the RDP bounds in a fixed order. With the optimal conversion, the smallest valid value for the curve, at the end,
$$\varepsilon_{\text{RDP, Mironov}} \;\ge\; \varepsilon_{\text{RDP, improved}} \;\ge\; \varepsilon_{\text{RDP, optimal}} \;\ge\; \varepsilon_{\text{true}}.$$ {#eq-choose-order}
Per-step composition, basic or advanced, is useless beyond a handful of steps, off by a factor of over a thousand after the full run. RDP is safe, and with the improved conversion it is 7.6% above the PLD at 10,000 steps, but far more on short runs. The CLT falls below the certified lower bound at every $k$, so a paper that quotes it, such as 1.62 after 1,000 steps, claims a guarantee the run does not have.

## The inverse problem: noise for a target ε {#sec-choose-inverse}

Practitioners usually fix the budget, the steps and the batch size, and then ask how much noise they need. For an accountant $A$, write $\varepsilon_A(\sigma)$ for the value it reports at noise multiplier $\sigma$. The noise it requires for a target $\varepsilon^*$ is then
$$\sigma_A(\varepsilon^*) = \min\{\sigma > 0 : \varepsilon_A(\sigma) \le \varepsilon^*\}.$$ {#eq-choose-inverse}
More noise never raises a correct accountant's $\varepsilon$, so a bisection finds it. Return the end of the bracket where the target was checked, because that end is safe.

A loose accountant turns its slack directly into extra noise. For a target of 3 over 10,000 steps, the pessimistic PLD needs $\sigma = 1.584$, while RDP with the improved conversion needs 1.662. Across targets, the improved conversion costs up to 4.9% more noise than the PLD, and Mironov's conversion far more. The CLT asks for less noise than the PLD, which at a target of 8 is provably too little. So calibrate $\sigma$ with the accountant you will report.

## Where each accountant is tight {#sec-choose-decision}

An accountant is tight only when its summary describes the mechanism exactly, so work through these questions in order.

1. **Does the code match the accountant's assumptions?** Check this first (@sec-choose-match), because a mismatch costs more than any choice of accountant.
2. **Are any parameters chosen during training?** If only the stopping time adapts within a planned maximum, account for the maximum. If the noise or budget adapts, use a Rényi filter ([Adaptive composition](08-adaptive-composition.md)).
3. **Is it the Gaussian mechanism without subsampling?** Then the composed pair is exactly Gaussian, and GDP gives the exact $\varepsilon$. RDP with the improved conversion is 8.0% above it on a hundred releases at $\sigma = 10$, and zCDP gives the same as RDP here.
4. **Is it DP-SGD, the Poisson-subsampled Gaussian?** Then use the pessimistic PLD or a privacy loss random variable (PRV) accountant, with the optimistic value as the error bar. Avoid zero-concentrated differential privacy (zCDP), because its single slope cannot express amplification by subsampling.
5. **Are the steps pure DP, such as randomized response or the Laplace mechanism?** The PLD is exact when the loss takes few values. A hundred randomized-response steps at 0.1 give 4.3068, against 4.6154 from RDP.
6. **Do local reports pass through a shuffler, or is only the last model released?** See [Amplification by shuffling and by iteration](07-shuffling-iteration.md).

## Matching the accountant to the code {#sec-choose-match}

Every number in @tbl-choose-main assumes a model of the training run that the accountant cannot check against your code. Many papers reported the Poisson-sampling number while training with shuffled fixed-size batches, a statement about a different algorithm (Ponomareva et al. 2023). So check each item of @tbl-choose-checklist before you report.

| Item | The accountant assumes | Check in the code | If it does not match |
| --- | --- | --- | --- |
| Sampling scheme | Poisson sampling at rate $q$ | Is each record included independently, so batch sizes vary? | Fixed-size random batches: account with $\sigma/2$. Shuffle and partition: report the unamplified guarantee |
| Neighbouring relation | Add or remove one record | Which relation does the claim state? | Replace-one costs more per step |
| Privacy unit | One example | Can one person contribute several examples? | Clip and sample per user, or pay for group privacy |
| Tuning and adaptivity | Settings fixed in advance | Tuning on private data, early stopping, noise schedules? | Account for the tuning runs; planned maximum, or a filter |
| Steps counted | Every step on private data | Warm-up runs, restarts, evaluation passes that release results? | Count them all |
| Noise scale | Noise standard deviation $\sigma C$ | Is $\sigma$ measured against the clipping norm actually used? | Recompute $\sigma$ from the noise added |

: A checklist for matching the accountant to the training code. {#tbl-choose-checklist}

The sampler is the most expensive item. Lebeda et al. (2024) show that such a run is covered by the Poisson accountant at noise multiplier $\sigma/2$. After the full run, the PLD upper bound rises from 6.24 to 43.42. That factor of 6.96 dwarfs any choice of accountant. Shuffle and partition is worse, because no amplification theorem covers it. What remains certain is one Gaussian step per epoch for zero-out neighbours, which gives $\varepsilon = 91.8173$ over a hundred epochs at $\sigma = 1$.

In a code review, a data loader with `shuffle=True` and a fixed `batch_size` is shuffle and partition, whatever `sample_rate` the accountant was given. Opacus samples by the Poisson scheme by default (`poisson_sampling=True`), so keep that default.

## What to report {#sec-choose-report}

An $\varepsilon$ without its assumptions cannot be checked or compared, so Ponomareva et al. (2023) and the US National Institute of Standards and Technology (NIST) ask for the items below. Choose the slack $\delta$ with care, because with $n$ records, publishing one random record in full meets a $\delta$ of about $1/n$. So Ponomareva et al. ask for a $\delta$ much smaller than $1/n$, and NIST gives $1/n^2$ as a common recommendation.

Reading $\varepsilon$ off the composed PLD at another $\delta$ is cheap, so report several values, not a single point. For the running example with 50,000 records, the pessimistic PLD gives 9.0502 at $1/n^2$, against 6.2377 at $10^{-5}$. Round upper bounds up, never to the nearest value, so 9.0502 becomes 9.06. A filled-in statement for the running example reads as follows.

- **Setting.** DP-SGD in the central model. It covers every checkpoint and the final model.
- **Unit and neighbours.** One training example per person, with neighbours that add or remove one example.
- **Training.** Poisson sampling at rate 0.01 over 50,000 examples for 100 epochs, with clipping norm 1 and noise multiplier 1.
- **Accountant.** The pessimistic PLD on the grid $10^{-5}$, in both directions, with the library and its version.
- **Guarantee.** The model is $(6.24, 10^{-5})$-DP and $(9.06, 4.0\times10^{-10})$-DP. The optimistic values, 6.1377 and 8.9378, bound the accountant's own error.
- **Not covered.** Say what the guarantee leaves out, such as hyperparameters tuned on a public proxy dataset, pretraining on public data, or preprocessing statistics computed from the private data.

## Checking your accountant {#sec-choose-check}

An accountant is code, and code has bugs, so run checks whose answers you know. First, set $q = 1$, which makes each step a plain Gaussian mechanism. Ten steps at $\sigma = 1$ are then exactly $(17.8566, 10^{-5})$-DP, and every upper bound must lie above it, as RDP's 19.0536 does. Second, the PLD bracket on plain Gaussian steps must contain the exact value, which the calculator in [Numerical accounting](06-numerical-accounting.md) gives. Third, the order of @eq-choose-order must hold, with no upper bound below the optimistic PLD. Fourth, $\varepsilon$ must never decrease as the number of steps grows, and never increase as the noise grows.

Then compare two maintained libraries, and pin their versions in your report. The full edition's numbers agree with Opacus 1.6.0, `dp_accounting` 0.6.0 and `prv_accountant` 0.2.0. Disagreements usually have one of three causes. If two RDP accountants differ at each order by a fixed amount, 0.3612 at order 10, one of them uses Mironov's conversion. Fractional orders are a weak spot of `dp_accounting` 0.6.0, where it returns a loose bound, so compare at integer orders. And if two PLD or PRV brackets fail to overlap, one library has a bug.

## Code {#sec-choose-code}

In Google's `dp_accounting`, you describe the run as an event and compose it into an accountant. Restricting RDP to integer orders avoids the fractional-order weakness.

```text
import dp_accounting

step = dp_accounting.PoissonSampledDpEvent(
    sampling_probability=0.01, event=dp_accounting.GaussianDpEvent(noise_multiplier=1.0))
run = dp_accounting.SelfComposedDpEvent(event=step, count=10_000)

pld = dp_accounting.PLDAccountant(value_discretization_interval=1e-5)
pld.compose(run)
eps_pld = pld.get_epsilon(target_delta=1e-5)       # should lie in [6.1377, 6.2377]
rdp = dp_accounting.RdpAccountant(orders=list(range(2, 65)))   # same compose / get_epsilon
```

For the inverse problem, `dp_accounting.calibrate_dp_mechanism` searches for the noise multiplier that meets a target $\varepsilon$ and $\delta$. Opacus has `opacus.accountants.utils.get_noise_multiplier`, whose `accountant` argument defaults to `"rdp"`, so pass `"prv"` to match the default of `PrivacyEngine`. Opacus's `"gdp"` accountant uses the CLT, so never report it as a guarantee. For a cross-check, Microsoft's `prv_accountant` installs a command-line tool that prints a lower bound, an estimate and an upper bound.

```bash
compute-dp-epsilon --sampling-probability 0.01 --noise-multiplier 1.0 --delta 1e-5 --num-compositions 10000
```

## Cheat sheet

| Mechanism | Use | Avoid |
| --- | --- | --- |
| Gaussian, no subsampling | GDP, exact | Per-step $(\varepsilon, \delta)$ composition |
| Poisson-subsampled Gaussian (DP-SGD) | Pessimistic PLD or PRV, with the lower bound as error bar | zCDP; the CLT as a guarantee |
| Pure-DP steps | PLD, exact for finite outputs, or RDP with the mechanism's own curve | Advanced composition |
| Fixed-size or shuffled batches | Poisson accountant at $\sigma/2$, or the unamplified bound | The Poisson number at $\sigma$ |

: Which accountant for which mechanism. {#tbl-choose-cheat}

**Rules of thumb**

- Report the pessimistic PLD, with the optimistic value beside it, and calibrate $\sigma$ with the same accountant you report.
- Use RDP for quick bounds, filters and many heterogeneous mechanisms.
- Never report a CLT number as a guarantee, and check any reported number against the PLD lower bound.
- Check the sampler first, because accounting fixed-size or shuffled batches as Poisson sampling is the most expensive error in practice.
- Never publish an $\varepsilon$ without its accountant and its $\delta$. Give the training parameters and the library version, and $\varepsilon$ at several $\delta$, including one well below $1/n$.

## Further reading

- Ponomareva et al., "How to DP-fy ML: A Practical Guide to Machine Learning with Differential Privacy" (JAIR 2023): matching the accounting to the pipeline, and what to report.
- Near, Darais, Lefkovitz & Howarth, "Guidelines for Evaluating Differential Privacy Guarantees" (NIST SP 800-226, 2025): what an evaluator will ask of your claim.

**Derivations.** Every result here is derived step by step in [the full edition's chapter](../../book-full/part04-privacy-accounting/09-choosing-an-accountant.md).
