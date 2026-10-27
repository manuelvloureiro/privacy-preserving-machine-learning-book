# Privacy accounting

This part answers the question that decides whether a trained model's privacy claim means anything: **after thousands of noisy steps, each on a random subset of the data, how large is $\varepsilon$?** The previous part defined differential privacy (DP), built the basic mechanisms and proved how $(\varepsilon, \delta)$ guarantees compose, but no single theorem of it answers this question. The answer comes from an **accountant**, a procedure that tracks a summary of each step's privacy loss, combines the summaries, and converts the result into the $(\varepsilon, \delta)$ you report. The running case is stochastic gradient descent (SGD) with per-example clipping and Gaussian noise, called DP-SGD, and the main summary is Rényi differential privacy (RDP). Two accountants can report values of $\varepsilon$ for the same run that differ by a large factor. Both can be correct upper bounds, but only the tighter one tells you what the noise bought.

## Who this part is for

This part assumes the [prerequisites](../part01-prerequisites/00-introduction.md) and the [DP foundations](../part03-dp-foundations/00-introduction.md). From [Divergences](../part01-prerequisites/03-divergences.md) you need the Rényi divergence $D_\alpha$, the hockey-stick divergence $E_\gamma$ and the data-processing inequality. From [Hypothesis testing](../part01-prerequisites/04-hypothesis-testing.md) you need trade-off functions $T(P, Q)$ and the Gaussian trade-off curve. From [Concentration inequalities](../part01-prerequisites/02-concentration.md) you need Hoeffding's lemma, and from [Optimization](../part01-prerequisites/06-optimization.md) the contraction of a gradient step. From the DP foundations you need [randomized response](../part03-dp-foundations/01-definition.md#sec-dpdef-rr), [the hypothesis-testing reading of DP](../part03-dp-foundations/01-definition.md#sec-dpdef-testing) and [basic composition](../part03-dp-foundations/02-properties.md#sec-dpprop-comp). You also need the privacy loss random variable with advanced and optimal composition from [Composition](../part03-dp-foundations/04-composition.md), the Gaussian mechanism and its exact $\delta(\varepsilon)$ from [the Laplace and Gaussian mechanisms](../part03-dp-foundations/03-laplace-gaussian.md#sec-lg-analytic), and the local and shuffle models from [Trust models](../part03-dp-foundations/09-trust-models.md). Each chapter restates these facts in one line where it uses them, with a link.

## The common thread

Composition is easy for the privacy loss itself, but hard for $(\varepsilon, \delta)$. If you run $k$ mechanisms, the privacy loss of the whole run is the sum of the $k$ per-step losses, so its distribution is a convolution. A pair $(\varepsilon, \delta)$ forgets almost everything about that distribution, and advanced composition recovers only part of what is lost. So every accountant in this part makes one move: **track a richer summary of the privacy loss, one that composes simply, and convert to $(\varepsilon, \delta)$ only once, at the end.** The summaries differ in how much they keep, and @tbl-acct-map lists them.

| Chapter | The question it answers | Summary it tracks | Builds on |
| --- | --- | --- | --- |
| [Rényi DP](01-renyi-dp.md) | Which summary of the privacy loss adds up under composition? | The curve $\varepsilon(\alpha)$ of Rényi divergences | 2, 3, 18–21 |
| [From Rényi DP to (ε, δ) and to tests](02-rdp-conversions.md) | How much does converting an RDP curve to $(\varepsilon, \delta)$ lose, and what is unavoidable? | The same curve, read as a constraint on tests | 3, 4, 18, 21, 29 |
| [Concentrated differential privacy](03-concentrated-dp.md) | When does one number $\rho$ describe the whole curve, and when not? | A line $\rho\alpha$ above the curve | 2, 3, 19–21, 25, 29, 30; one formula from 33 |
| [f-DP and Gaussian differential privacy](04-gaussian-dp.md) | What is the most informative description of a mechanism's privacy? | The full trade-off function; $\mu$ for Gaussian ones | 3, 4, 18, 21, 29, 30 |
| [Amplification by subsampling](05-subsampling.md) | How much privacy does using each record only with probability $q$ buy? | RDP of the subsampled Gaussian | 3, 18–21, 29, 30 |
| [Numerical accounting with privacy loss distributions](06-numerical-accounting.md) | How do you compute the exact $\varepsilon$ of a composition, with error bars? | The discretized distribution of the privacy loss | 3, 18, 20, 21, 29, 30, 32, 33 |
| [Amplification by shuffling and by iteration](07-shuffling-iteration.md) | What other sources of uncertainty amplify privacy? | Bounds for shuffled reports and hidden iterates | 3, 6, 7, 18, 26, 29, 30, 32, 33 |
| [Adaptive composition: filters, odometers and individual accounting](08-adaptive-composition.md) | What changes when the privacy parameters are chosen on the fly, or differ per record? | Running sums checked by filters and odometers | 2, 21, 29, 31, 33 |
| [Choosing an accountant](09-choosing-an-accountant.md) | Which accountant should you use for DP-SGD, and what should you report? | All of the above, side by side | 3, 21, 29–36 |

: The chapters of this part, the question each one answers, the summary of the privacy loss it tracks, and the earlier chapters it uses. {#tbl-acct-map}

The summaries form a ladder, in which each one keeps less than the one before. An RDP curve is a lossy summary of the trade-off function. A zero-concentrated DP (zCDP) parameter $\rho$ is a lossy summary of the RDP curve, and $(\varepsilon, \delta)$ is a lossy summary of all three. Numerical accounting keeps the whole privacy loss distribution and loses only a discretization error, which it bounds. Amplification (chapters 33 and 35) is a separate question: which pair of distributions you account for in the first place. Every accountant must use the pair that matches the sampling scheme that the training code runs.

## The running example

Every chapter that can compute it reports the same number: the $\varepsilon$ at $\delta = 10^{-5}$ of DP-SGD with Poisson sampling rate $q = 0.01$, noise multiplier $\sigma = 1$ and up to $T = 10{,}000$ steps, which is one hundred epochs. A single Gaussian mechanism with $\sigma = 1$ is the small example. As the number moves from chapter to chapter, it shows what each tool gains, and [Choosing an accountant](09-choosing-an-accountant.md) puts all the values side by side.

## How to read this part

**Order.** Read chapters 29 and 30 first, because every later chapter uses the RDP curve and its conversion. Chapter 30 ends by reading the curve as a constraint on tests, and chapter 32 develops that view in full, so read it next. Then read chapters 33 and 34, the practical core, in that order. Chapter 31 can come any time after chapter 30, but its section on subsampling uses one formula from chapter 33. Read chapter 31 before chapter 36, which uses its conversion. Chapter 35 needs chapter 33, chapter 36 needs chapters 31 and 33, and chapter 37 uses all the others, so read it last.

**Run the code.** Each chapter has a script in `code/` that prints every number its text quotes, except values from an external library or paper, and regenerates its figures, for example

```bash
uv run python code/part04-privacy-accounting/renyi_dp.py
```

No script uses an accounting library. Each accountant is a few dozen lines of NumPy written from its definition, so you can read what it computes.

## Notation

The part uses the notation of the DP foundations part, extended in @tbl-acct-notation.

| Symbol | Meaning |
| --- | --- |
| $M$, $D \sim D'$ | A mechanism; neighbouring datasets (add/remove-one unless replace-one is named) |
| $y$, $P = M(D)$, $Q = M(D')$ | An output; the two output distributions being compared |
| $D' = D \cup \{x\}$; $\mathrm{FPR}$, $\mathrm{FNR}$ | Under add/remove, $D'$ holds the record $x$ and $D$ does not. A test of $P$ against $Q$ has false positive rate $\mathrm{FPR}$, the rate of saying $D'$ when the data were $D$, and false negative rate $\mathrm{FNR}$, the rate of saying $D$ when they were $D'$. A guarantee covers both orders of the pair. A chapter that needs one order says so, as the subsampling chapters do for the remove direction, which puts the dataset with the record first |
| $L_{D,D'}(y) = \ln\frac{p_{M(D)}(y)}{p_{M(D')}(y)}$ | The privacy loss of output $y$; $L_{D,D'}(Y)$ with $Y \sim M(D)$ is the privacy loss random variable |
| $\delta(\varepsilon) = E_{e^\varepsilon}(P\|Q)$ | The smallest $\delta$ for a given $\varepsilon$, as a hockey-stick divergence |
| $D_\alpha$, $\varepsilon(\alpha)$ | Rényi divergence of order $\alpha > 1$; the RDP curve of a mechanism |
| $\rho$ | The zCDP parameter |
| $T(P, Q)$, $G_\mu$, $\mu$ | Trade-off function; the Gaussian trade-off function $T(\mathcal N(0,1), \mathcal N(\mu, 1))$; the parameter of Gaussian differential privacy, so a Gaussian mechanism with noise multiplier $\sigma$ has $\mu = 1/\sigma$ |
| $\Delta_2 f$, $\sigma$ | $\ell_2$ sensitivity; noise multiplier (noise standard deviation per unit of sensitivity) |
| $q$, $k$, $T$ | Sampling rate; number of mechanisms composed; number of steps of one iterative algorithm, such as DP-SGD |
| $h$ | Grid width of a discretized privacy loss distribution |

: Notation used throughout this part, in addition to that of the earlier parts. {#tbl-acct-notation}

Two symbols could have two meanings, so the part avoids the clash. The Rényi order is $\alpha$, but the f-DP literature uses $\alpha$ for the type I error, so this part writes the two errors as $\mathrm{FPR}$ and $\mathrm{FNR}$. The sampling rate is $q$, so the chapters with a sampling rate write output densities with capital letters, such as $Q(y)$ or $P_q(y)$, never $q(y)$.

## What you will be able to do

At the end of this part you can:

- compute the RDP curves of the Gaussian, Laplace and randomized-response mechanisms, compose them, and convert the result to $(\varepsilon, \delta)$ with the classic and the improved conversion;
- say what an RDP curve, a zCDP parameter and a trade-off function each keep and lose about the privacy loss;
- derive amplification by subsampling, and compute the RDP of the Poisson-subsampled Gaussian;
- build a privacy-loss-distribution accountant with pessimistic and optimistic bounds, and use it to check other accountants;
- say when shuffling or hidden iterates give extra privacy;
- run a privacy filter, and compute per-record privacy spend in DP-SGD;
- pick an accountant for a DP-SGD run, state its assumptions, and report $\varepsilon$ so that someone else can check it.
