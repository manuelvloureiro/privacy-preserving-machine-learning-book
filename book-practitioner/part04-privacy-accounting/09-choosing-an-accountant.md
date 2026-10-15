# Choosing an accountant

## Why this matters

You have trained a model with differential privacy (DP), and now you must write one sentence about it in a paper or a model card. The training loop was stochastic gradient descent (SGD) with per-example clipping and Gaussian noise, called DP-SGD ([Amplification by subsampling](05-subsampling.md)). At each step, every record joined the batch independently with a fixed probability, the sampling rate $q$, which is called Poisson sampling. Each record's gradient was then clipped to a norm of at most the clipping norm $C$, and Gaussian noise was added to the sum of the clipped gradients. The noise had standard deviation $\sigma C$, where the **noise multiplier** $\sigma$ says how large the noise is compared with one record's largest possible contribution. Your run used $q = 0.01$ and $\sigma = 1$. This chapter writes $k$ for the number of steps, which here goes up to 10,000.

The sentence you must write is "the model is $(\varepsilon, 10^{-5})$-DP", for some privacy level $\varepsilon$. It promises something about any two datasets that differ in one record ([The definition of differential privacy](../part03-dp-foundations/01-definition.md)). Any event about what training released is at most $e^\varepsilon$ times more likely on one dataset than on the other, plus a small slack $\delta$, here $10^{-5}$. A small $\varepsilon$ means that one record barely changes what anyone can see.

For a fixed run, the true $\varepsilon$ is a single number, but nobody can compute it exactly. A procedure that bounds or estimates it is called an **accountant**, and the earlier chapters of this part give you several. On this run they disagree widely. Adding up per-step guarantees with basic composition gives 11296.93 after 10,000 steps. Rényi differential privacy (RDP), a central-limit approximation and a privacy loss distribution (PLD) accountant all land far lower, between 6.0071 and 7.4287. Which number do you report, and what else must you say so that someone else can check it?

The numbers differ because the accountants make different kinds of statement. An **upper bound** comes with a proof that holds for every pair of neighbouring datasets. It may be loose, but the run really is $(\varepsilon, \delta)$-DP with the $\varepsilon$ it reports. An **approximation** has no such proof, so it may land below the truth, and then the claim it supports is false. A certified **lower bound** proves that the true $\varepsilon$ is at least some value. That is not a privacy guarantee, but it exposes any reported $\varepsilon$ that is too small.

So the rule of this chapter is short. **Report the tightest upper bound whose assumptions hold for the code you ran, and report enough that someone else can recompute it.**

## Every accountant on one run {#sec-choose-compare}

All accountants start from what one step of DP-SGD reveals about one record. Look at the noisy gradient sum along the direction of that record's clipped gradient, in units of the clipping norm. [Amplification by subsampling](05-subsampling.md) explains why this one direction is the pair to account for. Write $\mathcal N(m, \sigma^2)$ for a normal distribution with mean $m$ and standard deviation $\sigma$. Without the record, the output is pure noise, $\mathcal N(0, \sigma^2)$. With the record, it is in the batch only with probability $q$, and then its gradient shifts the output by 1. So with the record, the output follows the mixture
$$(1 - q)\,\mathcal N(0, \sigma^2) + q\,\mathcal N(1, \sigma^2).$$
The accountants differ in how much of this pair of distributions, without and with the record, they keep, and in how they combine $k$ copies of it. Whatever an accountant throws away has to be filled in with a worst case, so the more it keeps, the tighter its number.

The **privacy loss** of an output is the logarithm of how much more likely that output is on one of the two datasets than on the other. Over $k$ steps the losses add up, and since each step's loss can be positive or negative, much of the sum cancels. The crudest accountant ignores this. It keeps one $(\varepsilon, \delta)$ pair per step, shrunk for subsampling because the record is missing from most batches, and composes $k$ of them ([Advanced and optimal composition](../part03-dp-foundations/04-composition.md)). Basic composition adds up the $\varepsilon$'s, while advanced composition grows like $\sqrt k$ at the price of a little extra $\delta$. Both are valid upper bounds, even when each step depends on the earlier outputs, but they are very loose, because they assume that every step leaks its worst case.

RDP keeps a whole curve per step ([Rényi differential privacy](01-renyi-dp.md)). At each order $\alpha$ greater than 1, the curve gives the Rényi divergence of the two output distributions. Up to a factor, that is the logarithm of an exponential moment of the privacy loss. Curves of composed steps simply add, so $k$ steps have $k$ times the curve, and for the subsampled Gaussian the curve is exact.

At the end, a conversion turns the composed curve into a single guarantee ([From Rényi DP to (ε, δ) and to tests](02-rdp-conversions.md)). Mironov's conversion is the original one, and the improved conversion of Balle et al. (2020) is smaller at every order. The optimal conversion gives the smallest $\varepsilon$ that holds for every pair of distributions with the composed curve. All three are upper bounds, because each is proved for every pair with the curve, and the real pair is one of them. Evaluating the curve on fewer orders can only give a larger $\varepsilon$, so a coarse grid of orders costs tightness, never validity.

The central-limit approximation of Gaussian differential privacy (GDP) takes a different route ([f-DP and Gaussian differential privacy](04-gaussian-dp.md)). A mechanism is **$\mu$-GDP** if telling its two output distributions apart is no easier than telling $\mathcal N(0, 1)$ from a normal distribution shifted by $\mu$. The privacy loss of $k$ steps is a sum of $k$ small losses, so by the central limit theorem (CLT) it should be close to Gaussian. The approximation therefore treats the run as $\mu$-GDP with
$$\mu = q\sqrt{k\,(e^{1/\sigma^2} - 1)}.$$
It then reads $\varepsilon$ off the exact privacy curve of a mechanism with that $\mu$.

That formula is a limit for many steps with a small sampling rate, so it holds only when $q\sqrt k$ stays fixed as $k$ grows. It also keeps only the mean and the variance of each step's loss. Neither simplification comes with an error bound in the tail of the loss distribution, yet a $\delta$ of $10^{-5}$ asks exactly about that tail. So the CLT number is an approximation, not a bound.

The PLD accountant keeps everything ([Numerical accounting with privacy loss distributions](06-numerical-accounting.md)). The PLD is the distribution of the privacy loss over outputs. Under composition the losses add, so the PLD of $k$ steps is that of one step convolved with itself $k$ times. A computer does this quickly with the fast Fourier transform (FFT), after it rounds the losses to a grid of width $h$, here $10^{-5}$. Rounding every loss up can only make the pair easier to tell apart, and that stays true through all $k$ convolutions. So rounding up gives the **pessimistic PLD**, an upper bound. Rounding down gives the **optimistic PLD**, a lower bound.

Each step's rounding moves its loss by less than $h$, so after $k$ steps the two bounds are at most $kh$ apart, which is 0.1 here. That holds as long as the accountant cuts off no tail of the loss distribution, and @sec-choose-report shows where the cut-off starts to matter. Since the guarantee must hold whichever dataset contains the record, the accountant computes the PLD in both directions and keeps the larger $\varepsilon$.

@tbl-choose-main gives every accountant's $\varepsilon$ on the running example.

| $k$ | Basic | Advanced | RDP, Mironov | RDP, improved | RDP, optimal | GDP CLT | PLD upper | PLD lower |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 0.20 | 1.13 | 1.3175 | 0.9555 | 0.4899 | 0.0368 | 0.1995 | 0.1994 |
| 10 | 3.64 | 7.88 | 1.4145 | 1.0353 | 0.7951 | 0.1304 | 0.3800 | 0.3799 |
| 100 | 58.01 | 78.10 | 1.6118 | 1.2141 | 1.1538 | 0.4575 | 0.7185 | 0.7175 |
| 1,000 | 839.39 | 1258.64 | 2.5380 | 2.1014 | 2.1009 | 1.6177 | 1.8332 | 1.8232 |
| 10,000 | 11296.93 | 24379.55 | 7.4287 | 6.7127 | 6.7127 | 6.0071 | 6.2377 | 6.1377 |

: DP-SGD with $q = 0.01$ and $\sigma = 1$: $\varepsilon$ at $\delta = 10^{-5}$ after $k$ steps under every accountant. Basic and advanced are per-step guarantees, amplified by subsampling and then composed. The PLD columns use the grid $h = 10^{-5}$. {#tbl-choose-main}

The proofs put the RDP bounds in a fixed order. Mironov's conversion is the largest, because the improved one is smaller at every order, while the optimal one is by definition the smallest valid value for the curve:
$$\varepsilon_{\text{RDP, Mironov}} \;\ge\; \varepsilon_{\text{RDP, improved}} \;\ge\; \varepsilon_{\text{RDP, optimal}} \;\ge\; \varepsilon_{\text{true}}.$$ {#eq-choose-order}
The theory does not order RDP against the pessimistic PLD, which can exceed the truth by up to $kh$. On this run RDP stays above it at every $k$, but that is an observation, not a theorem.

![Left: $\varepsilon$ at $\delta = 10^{-5}$ against the number of steps $k$ for DP-SGD with $q = 0.01$ and $\sigma = 1$, under every accountant (the two PLD curves overlap). Right: each accountant divided by the PLD upper bound. Ratios above 1 are valid but loose. The shaded region below 1 is where only the PLD lower bound may lie, and the CLT lies there at every $k$.](figures/choosing_an_accountant_eps_vs_k.svg){#fig-choose-eps-k}

@tbl-choose-main shows that per-step composition is useless beyond a handful of steps. It is right only at $k = 1$, where there is nothing to compose, and at 10,000 steps it is off by a factor of over a thousand. Advanced composition is even worse here, because its second term grows like $k$ and this run's per-step $\varepsilon$ is too large for that term to be negligible.

RDP is a safe bound, and its excess over the pessimistic PLD has two sources. The first is the conversion, which at order $\alpha$ pays a toll of about $\ln(1/\delta)/(\alpha - 1)$, however small the curve is. After one step the curve is tiny, so the toll is most of the answer, and the improved conversion overstates the PLD by 379.1%.

The second source is the curve itself. On this run even the optimal conversion stays above the PLD, because the curve keeps only a family of moments of the privacy loss, not its whole distribution. In a long run the composed curve is large, the best order moves up, and the toll shrinks in comparison. From 1,000 steps on, almost all of the excess is in the curve. At 10,000 steps the excess is 7.6%, and only an accountant that keeps the whole distribution removes it.

The CLT under-reports on this run. It falls below the certified lower bound at every $k$, for example 6.0071 against 6.1377 after the full run. Its ratio to the PLD climbs toward 1 as $k$ grows, as a central limit theorem should, but at no $k$ in the table is it a valid guarantee.

Suppose you ran 1,000 steps. Your paper says $\varepsilon = 2.54$ with Mironov's conversion, or 2.10 with the improved conversion that the Opacus library uses. A PLD accountant says 1.83, and the CLT says 1.62. The first three are true upper bounds, and the PLD's is the most informative because it is the smallest. The CLT's is false, because the optimistic PLD proves that $\varepsilon$ is at least 1.8232.

## The inverse problem: noise for a target ε {#sec-choose-inverse}

Practitioners usually fix a budget, such as $\varepsilon = 3$, together with the number of steps and the batch size, and then ask how much noise they must add. The noise multiplier is then the knob that sets the model's utility. Write $n$ for the number of records. The averaged gradient carries noise with standard deviation $\sigma C/(qn)$, where $qn$ is the expected batch size. A loose accountant overstates $\varepsilon$ at every $\sigma$, so it asks for a larger $\sigma$, and its slack turns directly into extra noise.

For an accountant $A$, write $\varepsilon_A(\sigma)$ for the privacy level it reports when the noise multiplier is $\sigma$. The noise it requires for a target $\varepsilon^*$ is then the smallest noise multiplier at which its reported value meets the target:
$$\sigma_A(\varepsilon^*) = \min\{\sigma > 0 : \varepsilon_A(\sigma) \le \varepsilon^*\}.$$ {#eq-choose-inverse}

This is easy to solve because more noise never raises a correct accountant's $\varepsilon$. A noisier run can be made from a less noisy one by adding independent Gaussian noise, without looking at the data. By the data-processing inequality ([Divergences](../part01-prerequisites/03-divergences.md)), such a map can only make the two output distributions harder to tell apart, so every accountant's $\varepsilon$ can only fall. For the PLD this holds up to grid effects of order $kh$.

So the required $\sigma$ is where a decreasing function crosses a level, and a bracketing search finds it. The search shrinks an interval from a $\sigma$ that misses the target to one that meets it. At the end it returns the upper end, because that is where $\varepsilon_A(\sigma) \le \varepsilon^*$ was actually checked, so for an upper-bound accountant the returned $\sigma$ is safe. The CLT needs no search, because its formula for $\mu$ can be solved for $\sigma$ directly.

| Target $\varepsilon$ | Advanced | RDP, Mironov | RDP, improved | GDP CLT | PLD upper | PLD lower |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 24.682 | 4.974 | 4.126 | 3.796 | 3.988 | 3.654 |
| 3 | 10.505 | 1.854 | 1.662 | 1.549 | 1.584 | 1.547 |
| 8 | 5.711 | 0.962 | 0.917 | 0.868 | 0.885 | 0.880 |

: The noise multiplier $\sigma$ each accountant requires for a target $\varepsilon$ at $\delta = 10^{-5}$, with $k = 10{,}000$ steps at $q = 0.01$. The PLD uses the grid $h = 10^{-5}$. The PLD lower column is the $\sigma$ at which the optimistic bound reaches the target. {#tbl-choose-sigma}

![For each target $\varepsilon$ at $\delta = 10^{-5}$ ($k = 10{,}000$, $q = 0.01$), the noise multiplier each accountant requires, divided by the one the PLD upper bound requires. RDP with Mironov's conversion asks for up to 25% more noise, and RDP with the improved conversion about 4% more. The CLT and the PLD lower bound fall in the shaded region: they ask for less noise than any certified accountant can justify. Advanced composition (6.19 to 6.63) is off the chart.](figures/choosing_an_accountant_noise.svg){#fig-choose-noise}

@fig-choose-noise divides each accountant's $\sigma$ by the one the pessimistic PLD needs. Advanced composition needs about six times as much, a noise level at which DP-SGD rarely trains a useful model. The RDP conversions cost only a few percent. Mironov's conversion costs 8.6% to 24.7% more noise, most at the strictest target, where its toll is the largest share of a small $\varepsilon$. The improved conversion costs only 3.4% to 4.9% more. That is less than its 7.6% excess in $\varepsilon$, because in this range a small rise in $\sigma$ buys a larger fall in $\varepsilon$. At the target of 8, for example, RDP asks for 3.6% more noise in every averaged gradient than the PLD does.

The CLT asks for 2.0% to 4.8% less noise than the certified bound needs. At a target of 8 that is provably too little. At the CLT's $\sigma = 0.8676$, the optimistic PLD already gives 8.2594. At targets of 1 and 3 the grid is too coarse to tell, since the PLD bracket at the CLT's $\sigma$ contains the target.

The width of the PLD bracket matters most at small targets, because $kh$ is a fixed amount of $\varepsilon$. At a target of 1 it is 10% of the budget. So the PLD pins $\sigma$ down only to the interval from 3.654 to 3.988. For a tight $\sigma$ at a small target, refine the grid.

## Where each accountant is tight {#sec-choose-decision}

DP-SGD is only one mechanism, and you will also use the Gaussian mechanism without subsampling, pure-DP mechanisms, whose $\delta$ is zero, and adaptive training. The ranking of the accountants changes with the mechanism, because an accountant is tight only when its summary describes that mechanism's privacy loss exactly.

The summaries form a ladder, from the one that keeps least to the one that keeps most. At the bottom, an $(\varepsilon, \delta)$ pair keeps two numbers per step. Next, zero-concentrated differential privacy (zCDP) keeps a single slope $\rho$ ([Concentrated differential privacy](03-concentrated-dp.md)). It promises that the RDP curve lies below the straight line $\rho\alpha$ through the origin. RDP keeps the whole curve.

GDP keeps more still, the whole trade-off between an attacker's two errors when testing which dataset was used, wrongly accusing and wrongly clearing. It is exact only when that trade-off is Gaussian. At the top, the PLD keeps the entire distribution of the privacy loss. Each rung loses information that the next one keeps, and lost information has to be filled in with a worst case.

For the Gaussian mechanism without subsampling, the composed pair is exactly Gaussian, so GDP describes it exactly. A Gaussian step with sensitivity 1 is $\mu$-GDP with $\mu = 1/\sigma$, and under composition the squares of $\mu$ add, just as the variances of independent Gaussians do. So a hundred Gaussian releases at $\sigma = 10$ compose to exactly $\mu = 1$, and GDP then gives the exact $\varepsilon$, which is 4.3772. RDP keeps this curve exactly too, but its conversion still costs something. The improved conversion is 8.0% above the truth, and Mironov's is 21.0% above it. Here zCDP gives the same as RDP, because the Gaussian's RDP curve is itself the straight line $\alpha\mu^2/2$.

For pure-DP steps, the privacy loss takes only a few values. In randomized response with per-step privacy level $\varepsilon_0 = 0.1$, each step's loss is either $+\varepsilon_0$ or $-\varepsilon_0$. The composed loss then takes only a few values, so the PLD needs no rounding and is exact, at 4.3068 for a hundred steps. Every summary that keeps less pays more. For example, RDP with the exact randomized-response curve gives 4.6154, while basic composition gives 10.0000.

For DP-SGD, subsampling makes each step a mixture, which no closed form describes exactly, so only the PLD is both certified and tight. RDP with the improved conversion is 7.6% above it at 10,000 steps, and the CLT falls below it. zCDP is the wrong tool here. Its line $\rho\alpha$ must lie above the subsampled RDP curve at every order. At high orders the ratio of that curve to $\alpha$ approaches $1/(2\sigma^2)$, the slope without subsampling, so $\rho$ must take that unsubsampled value. The line then sits far above the curve at the small orders where subsampling helps, and zCDP throws that gain away.

Adaptivity is the one place where RDP is the natural tool. The PLD accountant assumes that the per-step pair is fixed in advance, which fails when the noise or the stopping time is chosen during training from what has been released. A Rényi **filter** handles this case and loses nothing to the adaptivity ([Adaptive composition: filters, odometers and individual accounting](08-adaptive-composition.md)). It lets a step run only while the composed curve at one fixed order stays within the budget. An odometer, which reports the spending so far, tracks several orders and pays a little for that.

Two other cases change the pair you account for ([Amplification by shuffling and by iteration](07-shuffling-iteration.md)). Reports from local randomizers that pass through a shuffler gain privacy because each report hides among similar reports from others, and that gain needs its own accountant. When only the last model is released, amplification by iteration can give more privacy. But it needs a convex and smooth loss, steps that never push two models apart, and no released checkpoints, so it rarely applies to deep learning. @tbl-choose-decision sums up the choices.

| Mechanism | Use | Why | Avoid |
| --- | --- | --- | --- |
| Gaussian, no subsampling | GDP, or zCDP converted exactly | The squares of $\mu$ add, and the Gaussian trade-off is exact | Per-step $(\varepsilon, \delta)$ composition |
| Poisson-subsampled Gaussian (DP-SGD) | Pessimistic PLD, with the optimistic value as error bar | Certified bracket of width $kh$; RDP is 7.6% to 69% above it here for $k \ge 100$ | zCDP, which cannot express amplification; the CLT as a guarantee |
| Pure-DP steps (randomized response, Laplace, exponential mechanism) | PLD (exact for finite outputs), or RDP with the mechanism's own curve | The privacy loss takes few values | Advanced composition |
| Adaptive noise, stopping or per-record budgets | Rényi filter at a fixed order | Free at a fixed order; an odometer pays a union bound over orders | Plugging the $\varepsilon$ spent so far into a theorem that assumes fixed parameters |
| Local randomizers behind a shuffler | Exact computation for binary randomized response, a numerical bound otherwise | The closed-form bound is 4.47 to 7.55 times the exact value | The Poisson-subsampling accountant |
| Only the last iterate released | Amplification by iteration | Needs a convex, smooth loss, steps that do not push models apart, and no released checkpoints | Any hidden-state claim for non-convex training |

: Which accountant to use for which mechanism. {#tbl-choose-decision}

## Matching the accountant to the code {#sec-choose-match}

Every number in @tbl-choose-main assumes a model of the training run, and the accountant cannot check that model against your code. It takes $q$, $\sigma$ and $k$, and it trusts that the code does what its theorem assumes. Ponomareva et al. (2023) note that many papers reported the Poisson-sampling number while training with shuffled fixed-size batches. The habit goes back to Abadi et al. (2016). Such a mismatch does not make the number a little loose. It makes the number a statement about a different algorithm, so check each item of @tbl-choose-checklist against the code before you report $\varepsilon$.

| Item | What the accountant assumes | What to check in the code | If it does not match |
| --- | --- | --- | --- |
| Sampling scheme | Poisson sampling at rate $q$ | Is each record included independently, so batch sizes vary? | Fixed-size random batches: account with noise multiplier $\sigma/2$. Shuffle and partition: report the unamplified guarantee for zero-out neighbours (see below) |
| Neighbouring relation | Add or remove one record | Is the guarantee stated for add/remove, replace-one or zero-out? | Replace-one under Poisson sampling costs more per step, about twice the add/remove cost for a pure-DP step with small $\varepsilon$ |
| Privacy unit | One example | Can one person contribute several examples? | Clip and sample per user, or pay the group-privacy cost of protecting several records at once |
| Released outputs | Every noisy gradient | Are checkpoints, logs or gradients exposed? | Nothing to fix: the guarantee covers them all |
| Hyperparameter tuning | Not counted | Were hyperparameters chosen from results on the private data? | Account for the tuning runs, or tune on public data |
| Adaptive parameters | $\sigma$, $q$ and $k$ fixed in advance | Early stopping, noise schedules, per-record budgets? | Use a filter |
| Implementation | Exact real-valued Gaussian noise, exact clipping | Floating-point noise sampling, accumulation across devices | Outside the accountant ([Discrete and floating-point-safe mechanisms](../part03-dp-foundations/07-discrete-mechanisms.md)) |

: A checklist for matching the accountant to the training code. {#tbl-choose-checklist}

The sampling scheme is the most expensive item, for a mechanical reason. Suppose each step draws a fresh random batch of fixed size. Then adding a record has to push another one out. In effect one record is swapped for another, so the batch sum can move by up to $2C$ instead of $C$. Lebeda et al. (2024) show that such a run is covered by the Poisson pair with noise multiplier $\sigma/2$, the same noise measured against a shift twice as large. After the full run, the PLD upper bound then rises from 6.24 to 43.42. RDP rises in the same way, from 6.71 to 47.42. Read the other way, if your code draws fixed-size batches and you want the Poisson accountant's 6.24 to be honest, train with $\sigma = 2$.

Shuffle and partition is worse. Each epoch shuffles the data once and cuts it into batches, so each record is used exactly once per epoch, and no amplification theorem holds for every dataset. What remains certain is one Gaussian step per epoch for the record. It holds for **zero-out neighbours**, two datasets that are identical except that one record's gradient is replaced by zero in one of them. The batch that holds the record then moves by at most $C$, so each epoch is $\mu$-GDP with $\mu = 1/\sigma$. Over a hundred such epochs at $\sigma = 1$, the squares of $\mu$ add up, giving a total $\mu$ of 10. That gives $\varepsilon = 91.8173$. Under add/remove there is no clean bound at all, because inserting a record shifts the partition and changes every later batch. Under replace-one, the batch sum moves by up to $2C$, so $\mu$ doubles and $\varepsilon$ rises to 284.39.

The third mismatch is tuning on the private data. Suppose you train ten configurations on it and release the best. Which one is best depends on all ten runs, so the release costs all ten runs composed, which RDP puts at $\varepsilon = 27.23$. Papernot & Steinke (2022) give much better bounds for tuning procedures that try a random number of configurations.

So the sampler matters far more than the accountant. A one-line change in the data loader moves the PLD bound by a factor of 6.96. At the same number of steps, the choice between RDP and PLD changes $\varepsilon$ only by a factor of 1.19 with Mironov's conversion, and 1.08 with the improved one (@fig-choose-eps-k). The Opacus library, for one, uses Poisson sampling by default when you make a model private.

The privacy unit decides what is protected, because the guarantee is about datasets that differ in one unit. Example-level DP-SGD on a dataset with many examples per user protects each example, not each user.

In a code review, suppose a script uses a data loader with `shuffle=True` and `batch_size=500` on 50,000 records for 100 epochs. Its RDP accountant, called with `sample_rate=0.01`, reports 6.71. But the sampler is shuffle and partition, so the certified guarantee is the unamplified 91.8173. Switching the loader to Poisson sampling takes a few lines and makes the PLD accountant's 6.24 honest.

## What to report {#sec-choose-report}

An $\varepsilon$ without its assumptions cannot be checked or compared. The same number means different things for one example or one user, with or without tuning, and at different $\delta$. Two guidance documents say what a statement should contain. One is SP 800-226 from the US National Institute of Standards and Technology (NIST). The other is the machine-learning guide of Ponomareva and colleagues. It asks first for the setting, meaning who is trusted with the raw data, and then for what the guarantee covers, such as one training run or the tuning too. It also asks for the unit of privacy, the neighbouring relation, the accountant with its assumptions, and code when possible.

For DP-SGD, add what an accountant needs to recompute $\varepsilon$. That is the sampling scheme with the sampling rate $q$, the number of steps, the noise multiplier $\sigma$, and the accountant's version and discretization. Report the clipping norm $C$ too, although $\varepsilon$ does not depend on it, because $\sigma C$ is the actual noise level.

Choose $\delta$ with its meaning in mind, since it is the probability with which the promise may fail. For example, take a dataset of $n$ records and a mechanism that publishes one record, chosen at random, in full. When one record changes, the output distribution changes by only about $1/n$, because that record is published only with probability $1/n$. So the mechanism meets a $\delta$ of that size with $\varepsilon = 0$, and yet it exposes someone completely. A $\delta$ near $1/n$ therefore allows guarantees that protect nobody, with $n$ counted in units of privacy. That is why Ponomareva et al. ask for a $\delta$ much smaller than $1/n$, for example $n^{-1.1}$. The NIST guidelines go further, giving $1/n^2$ as a common recommendation and calling values above $10^{-5}$ suspicious.

Take the running example with $n = 50{,}000$ records. Then $1/n$ is $2.0\times10^{-5}$, and the customary $\delta$ is only half of that, not much smaller. Reading $\varepsilon$ off the composed PLD at another $\delta$ is cheap, so @tbl-choose-profile gives several.

| $\delta$ | PLD upper | PLD lower | RDP, improved |
| --- | --- | --- | --- |
| $2.0\times10^{-5}$ ($1/n$) | 6.0068 | 5.9068 | 6.4884 |
| $1.0\times10^{-5}$ | 6.2377 | 6.1377 | 6.7127 |
| $6.8\times10^{-6}$ ($n^{-1.1}$) | 6.3641 | 6.2641 | 6.8353 |
| $1.0\times10^{-6}$ | 6.9574 | 6.8574 | 7.4140 |
| $4.0\times10^{-10}$ ($1/n^2$) | 9.0502 | 8.9378 | 9.4540 |

: The running example's privacy profile: $\varepsilon$ at several $\delta$ for $k = 10{,}000$, $q = 0.01$, $\sigma = 1$ and $n = 50{,}000$. {#tbl-choose-profile}

At $1/n^2$, as NIST suggests, the certified $\varepsilon$ rises to 9.0502. There the bracket widens slightly beyond $kh$, to 0.1124, because the tails the accountant cuts off are no longer negligible next to so small a $\delta$.

Report the profile, not a single point. A single $(\varepsilon, \delta)$ pair cannot be converted to a smaller $\delta$, because many different mechanisms share that one pair and behave differently at smaller $\delta$. For the same reason, NIST asks you to report the original parameters of an RDP, zCDP or GDP guarantee alongside any conversion, which for the RDP accountant means its curve. Here the composed curve at the best order, 4.1, is 3.7336. Report the accountant's error bar as well. For a PLD accountant the optimistic value gives it, while an RDP number has no matching lower bound. Finally, if any item of @tbl-choose-checklist is in doubt, also give the guarantee that does not rely on it.

@tbl-choose-statement is a filled-in statement for the running example.

| Item | Statement |
| --- | --- |
| Setting | DP-SGD in the central model, with the training server trusted to run the code below. The guarantee covers every intermediate noisy gradient, so every checkpoint and the final model. |
| Unit and neighbours | One training example, and each person contributes at most one. Neighbours differ by adding or removing one example. |
| Training | Poisson sampling at rate $q = 0.01$ over $n = 50{,}000$ examples, so the expected batch size is 500. Training ran for $k = 10{,}000$ steps (100 epochs), with per-example clipping norm $C = 1$ and Gaussian noise multiplier $\sigma = 1$. |
| Accountant | Privacy loss distributions, with pessimistic discretization on the grid $h = 10^{-5}$, in both the add and the remove direction. |
| Guarantee | $(6.24, 10^{-5})$-DP and $(9.06, 4.0\times10^{-10})$-DP. The optimistic discretization gives 6.1377 and 8.9378, so the accountant's own error is at most 0.1000 and 0.1124. |
| Not covered | Hyperparameters were tuned on a public proxy dataset, not on the private data. |
| If the sampling assumption failed | If the data were instead shuffled once per epoch and cut into fixed-size batches, the guarantee without amplification is $(91.82, 10^{-5})$-DP, for zero-out neighbours. |

: A privacy statement for the running example, with $n = 50{,}000$ examples. {#tbl-choose-statement}

Every number in it comes from @tbl-choose-profile or @sec-choose-match. Upper bounds are rounded up, never to the nearest value, so 9.0502 becomes 9.06, because rounding down would claim more than the accountant certified.

## Checking your accountant {#sec-choose-check}

An accountant is code, and code has bugs. So before you trust a number, run checks whose answers you know, either cases where the truth is known exactly or properties that every correct accountant must have.

The first check sets $q = 1$. Without subsampling each step is the plain Gaussian mechanism, which is exactly $\mu$-GDP with $\mu = 1/\sigma$. So $k$ steps are exactly $\sqrt k/\sigma$-GDP, which gives the exact $\varepsilon$, and every upper bound must lie above it. For ten steps at $\sigma = 1$ the exact $\varepsilon$ is 17.8566. The RDP accountant gives 19.0536, above it as it must be. The CLT formula, which is not a bound, misses the exact value, because its derivation assumes that $q$ is small. It gives $\mu = 4.1452$ instead of the exact 3.1623, and so an $\varepsilon$ far from the truth.

The second check runs the PLD machinery on the same plain Gaussian pair, where the closed form is the truth. The bracket must contain it, and with $\mu = 1$ per step, ten steps and a grid of $10^{-3}$ it does. The bracket runs from 17.8516 to 17.8616, so its width is exactly $kh$.

The third check is that the orders of @eq-choose-order hold at every $k$, with no upper bound below the optimistic PLD, since a violation is a bug, not slack. The fourth check is a round trip. Evaluating the composed pessimistic PLD at its own $\varepsilon$ must give back the $\delta$ it was solved at, and it does. Its $\varepsilon$ must also never decrease with the number of steps, because a longer run releases everything a shorter one does.

Production code should use a maintained library and compare it against a second one. Google's `dp_accounting` has RDP and PLD accountants. The default in Opacus is a privacy loss random variable (PRV) accountant, a numerical accountant of the same kind as the PLD. Microsoft's `prv_accountant` returns a lower bound, an estimate and an upper bound.

Two RDP accountants differ only in the orders they try and in the conversion they use. On the running example the best orders lie between 4.1 and 9.9, inside the default grid of Opacus, which also uses the improved conversion, so the results should agree closely. If two RDP accountants differ at each order by a fixed amount, 0.3612 at order 10, one of them uses Mironov's conversion. On the running example, for instance, an RDP accountant that reports 7.43 instead of 6.71 uses Mironov's conversion (@tbl-choose-main).

Fractional orders are a known weak spot. At a fractional order, unlike an integer one, the subsampled Gaussian's RDP needs an infinite series whose terms have both signs. Version 0.6.0 of `dp_accounting` counts the negative terms as positive, so it returns a loose upper bound rather than the exact value. For the fixed-size-batch run, for example, it gives 49.43 where the full edition computes 47.42. So compare at integer orders, or against Opacus, which keeps the signs.

PLD and PRV accountants should agree within their error bars. Every certified upper bound lies above the truth and every certified lower bound below it, so if two libraries' brackets do not overlap, one of them has a bug. In the full edition's cross-check, run outside its scripts, the libraries' tight PLD estimate at 10,000 steps is about 6.19, inside the bracket.

## Code {#sec-choose-code}

The snippet below is a small RDP accountant for DP-SGD, with exact subsampled-Gaussian RDP at integer orders and the improved conversion. It also contains the exact GDP conversion that the checks use, and the noise calibration of @eq-choose-inverse.

```python
from math import e, erfc, exp, lgamma, log, sqrt


def normal_cdf(x: float) -> float:
    return 0.5 * erfc(-x / sqrt(2))


def gdp_epsilon(mu: float, delta: float) -> float:
    """Exact epsilon at delta of a mu-GDP mechanism, by bisection on its delta(eps) curve."""

    def gdp_delta(eps: float) -> float:
        return normal_cdf(mu / 2 - eps / mu) - exp(eps) * normal_cdf(-mu / 2 - eps / mu)

    lo, hi = 0.0, 1000.0
    for _ in range(200):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if gdp_delta(mid) > delta else (lo, mid)
    return hi


def subsampled_gaussian_rdp(alpha: int, q: float, sigma: float) -> float:
    """Exact RDP at an integer order of one Poisson-subsampled Gaussian step."""
    if q == 1.0:
        return alpha / (2 * sigma**2)
    terms = [  # log of each term of the binomial sum, combined with log-sum-exp
        lgamma(alpha + 1) - lgamma(j + 1) - lgamma(alpha - j + 1)
        + (alpha - j) * log(1 - q) + j * log(q) + (j * j - j) / (2 * sigma**2)
        for j in range(alpha + 1)
    ]
    top = max(terms)
    return (top + log(sum(exp(t - top) for t in terms))) / (alpha - 1)


def rdp_epsilon(q: float, sigma: float, steps: int, delta: float) -> float:
    """Upper bound on DP-SGD's epsilon: RDP on integer orders 2..64, improved conversion."""
    return min(
        steps * subsampled_gaussian_rdp(a, q, sigma) + log((a - 1) / a) - (log(delta) + log(a)) / (a - 1)
        for a in range(2, 65)
    )


def sigma_for_target(eps_target: float, q: float, steps: int, delta: float) -> float:
    """Smallest noise multiplier meeting the target, by bisection; returns the end that was checked."""
    lo, hi = 0.3, 50.0
    while hi - lo > 1e-4:
        mid = (lo + hi) / 2
        lo, hi = (lo, mid) if rdp_epsilon(q, mid, steps, delta) <= eps_target else (mid, hi)
    return hi


q, sigma, steps, delta = 0.01, 1.0, 10_000, 1e-5
print(f"{rdp_epsilon(q, sigma, steps, delta):.4f}")  # 6.7194
print(f"{gdp_epsilon(q * sqrt(steps * (e - 1)), delta):.4f}")  # 6.0071
print(f"{sigma_for_target(3.0, q, steps, delta):.3f}")  # 1.665
print(f"{gdp_epsilon(sqrt(10) / sigma, delta):.4f}")  # 17.8566
print(f"{rdp_epsilon(1.0, sigma, 10, delta):.4f}")  # 19.8017
print(f"{gdp_epsilon(sqrt(10 * (e - 1)), delta):.4f}")  # 25.5853
print(f"{gdp_epsilon(sqrt(100) / sigma, delta):.4f}")  # 91.8173
```

The first three lines print the running example's RDP bound, CLT value and noise for a target of 3. The rest print the $q = 1$ checks and the shuffle-and-partition fallback.

The snippet searches only integer orders, so its bounds come out a little larger than the full edition's but stay valid. For example, its RDP bound is 6.7194 instead of 6.7127. The full edition's script, [choosing_an_accountant.py](../../code/part04-privacy-accounting/choosing_an_accountant.py), runs every accountant, the PLD included. It takes about two and a half minutes to run from the repository root.

```bash
uv run python code/part04-privacy-accounting/choosing_an_accountant.py
```

## Cheat sheet

| Accountant | Kind | $\varepsilon$ at $k = 10{,}000$ | $\sigma$ for $\varepsilon = 3$ | When to use it |
| --- | --- | --- | --- | --- |
| Per step + advanced composition | Upper bound | 24379.55 | 10.505 | Never for DP-SGD |
| RDP, Mironov conversion | Upper bound | 7.4287 | 1.854 | Never: the improved conversion is free |
| RDP, improved conversion | Upper bound | 6.7127 | 1.662 | Quick bounds, adaptive filters, many heterogeneous mechanisms |
| PLD, pessimistic | Upper bound, within $kh$ of the truth | 6.2377 | 1.584 | The number to report |
| PLD, optimistic | Lower bound | 6.1377 | 1.547 | The error bar, and the referee for other accountants |
| GDP CLT | Approximation | 6.0071 | 1.549 | Intuition only; under-reports here |

: The accountants on the running example ($q = 0.01$, $\sigma = 1$, $\delta = 10^{-5}$), the kind of statement each makes, and when to use each. {#tbl-choose-cheat}

**Rules of thumb**

- Report the pessimistic PLD, with the optimistic value beside it. Use RDP with the improved conversion for adaptivity or a quick bound, and expect it to overstate $\varepsilon$, by 7.6% on this run and far more on short runs.
- Never report a CLT number as a guarantee. Check any reported number against the PLD lower bound.
- Calibrate $\sigma$ with the accountant you will report.
- Check the sampling scheme first. Accounting fixed-size or shuffled batches as Poisson sampling is the most expensive error in practice.
- Report the privacy unit, the neighbouring relation and the training parameters, with the accountant and its discretization. Give $\varepsilon$ at several $\delta$, including one well below $1/n$, and say what was not accounted for.

## Further reading

- Ponomareva et al., "How to DP-fy ML: A Practical Guide to Machine Learning with Differential Privacy" (JAIR 2023): matching the accounting to the pipeline, setting $\delta$, and what to report.
- Near, Darais, Lefkovitz & Howarth, "Guidelines for Evaluating Differential Privacy Guarantees" (NIST SP 800-226, 2025): what an evaluator will ask of your claim.
- Lebeda, Regehr, Kamath & Steinke, "Avoiding Pitfalls for Privacy Accounting of Subsampled Mechanisms under Composition" (IEEE SaTML 2025): fixed-size batches versus Poisson sampling.
- Gopi, Lee & Wutschitz, "Numerical Composition of Differential Privacy" (NeurIPS 2021): the PRV accountant with certified error bounds, and examples where the CLT under-reports.

**Derivations.** The ordering of the accountants, the monotonicity behind the noise calibration and the checks are derived step by step in [the full edition's chapter](../../book-full/part04-privacy-accounting/09-choosing-an-accountant.md), and each accountant is derived in the earlier chapter it links to. The full edition cites rather than derives the fixed-size-batch result (Lebeda et al. 2024), the reporting and $\delta$ conventions (Ponomareva et al. 2023, NIST SP 800-226) and the tuning bounds of Papernot & Steinke (2022), and it quotes the library values 49.43 and 6.19 from runs outside its scripts.
