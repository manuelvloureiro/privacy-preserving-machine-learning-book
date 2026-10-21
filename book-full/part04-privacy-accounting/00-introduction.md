# Privacy accounting

The previous part defined differential privacy (DP), built the basic mechanisms and proved how $(\varepsilon, \delta)$ guarantees compose. This part answers the question that decides whether a trained model's privacy claim means anything: **after thousands of noisy steps, each on a random subset of the data, how large is $\varepsilon$, really?** The answer is never read off a single theorem. It comes from an **accountant**, a procedure that tracks some summary of each step's privacy loss, combines the summaries, and converts the result into the $(\varepsilon, \delta)$ you report. The running case throughout is stochastic gradient descent (SGD) with per-example clipping and Gaussian noise, called DP-SGD, and the workhorse summary is Rényi differential privacy (RDP). Two accountants given the same training run can report values of $\varepsilon$ that differ by a large factor. Both can be correct upper bounds, but only the tighter one tells you what the noise actually bought.

## Who this part is for

You are expected to have read the [prerequisites](../part01-prerequisites/00-introduction.md) and the part on [DP foundations](../part03-dp-foundations/00-introduction.md). From [Divergences](../part01-prerequisites/03-divergences.md) you need the Rényi divergence $D_\alpha$, the hockey-stick divergence $E_\gamma$ and the data-processing inequality. From [Hypothesis testing](../part01-prerequisites/04-hypothesis-testing.md) you need trade-off functions $T(P, Q)$ and the Gaussian trade-off curve. From [Concentration inequalities](../part01-prerequisites/02-concentration.md) you need Hoeffding's lemma, and from [Optimization](../part01-prerequisites/06-optimization.md) the contraction of a gradient step. From the DP foundations you need [randomized response](../part03-dp-foundations/01-definition.md#sec-dpdef-rr) and the hypothesis-testing reading of DP, [basic composition](../part03-dp-foundations/02-properties.md#sec-dpprop-comp), the privacy loss random variable with advanced and optimal composition from [Composition](../part03-dp-foundations/04-composition.md), the Gaussian mechanism and its exact $\delta(\varepsilon)$ from [The Laplace and Gaussian mechanisms](../part03-dp-foundations/03-laplace-gaussian.md), and the local and shuffle models from [Trust models](../part03-dp-foundations/09-trust-models.md). Each chapter restates the facts it uses in one line and links to them.

## The common thread

Composition is easy for the privacy loss itself. If you run $k$ mechanisms, the privacy loss of the whole run is the sum of the $k$ per-step losses, and its distribution is a convolution. Composition is hard for $(\varepsilon, \delta)$, because a pair of numbers forgets almost everything about that distribution. Advanced composition recovers part of what is lost. The rest of the field rests on one move: **track a richer summary of the privacy loss, one that composes simply, and convert to $(\varepsilon, \delta)$ only once, at the end.**

The summaries differ in how much they keep, and @tbl-acct-map lists them.

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

Read the table from top to bottom as a ladder. An RDP curve is a lossy summary of the trade-off function, a zero-concentrated DP (zCDP) parameter $\rho$ is a lossy summary of the RDP curve, and $(\varepsilon, \delta)$ is a lossy summary of all three. Numerical accounting keeps the whole privacy loss distribution and loses only discretization error, which it bounds. Amplification (chapters 33 and 35) is a separate axis. It is about which pair of distributions you account for in the first place, and every accountant above must use the pair that matches the sampling scheme the training code actually runs.

## The running example

Every chapter that can compute it reports the same number: the $\varepsilon$ at $\delta = 10^{-5}$ of DP-SGD with Poisson sampling rate $q = 0.01$, noise multiplier $\sigma = 1$ and up to $k = 10{,}000$ steps. That is about one hundred epochs at a typical setting. A single Gaussian mechanism with $\sigma = 1$ serves as the small example. Watching the same number move from chapter to chapter shows what each tool gains. [Choosing an accountant](09-choosing-an-accountant.md) puts all the values side by side.

## How to read this part

**Order.** Read chapters 29 and 30 first: every later chapter uses the RDP curve and its conversion. Chapter 30 ends by reading the curve as a constraint on tests, and chapter 32 develops that trade-off view in full, so read it next. Chapters 33 and 34 form the practical core and should be read in order. Chapter 31 can be read any time after chapter 30, except its section on subsampling, which uses one formula from chapter 33; read it before chapter 36, which uses its conversion. Chapter 35 needs chapter 33. Chapter 36 needs chapters 31 and 33. Chapter 37 closes the part and uses all the others, so read it last.

**Structure.** The chapters follow the shape of the earlier parts: why the topic matters, a toolkit, one section per result with a derivation, interpretation, worked example and two check-yourself questions, then code, references and a cheat sheet. Exercises with worked solutions are collected in the exercise book, a companion website, with a link at the end of each chapter.

**Run the code.** Each chapter has a script in `code/` that prints every number its text quotes, except values attributed to an external library or paper, and regenerates its figures, for example

```bash
uv run python code/part04-privacy-accounting/renyi_dp.py
```

None of the scripts uses an accounting library. Each accountant is implemented from its definition in a few dozen lines of NumPy, so you can read exactly what it computes. As an exercise, compare them with Google's `dp_accounting` or Opacus.

## Notation

The part uses the notation of the DP foundations part, extended in @tbl-acct-notation.

| Symbol | Meaning |
| --- | --- |
| $M$, $D \simeq D'$ | A mechanism; neighbouring datasets (add/remove-one unless replace-one is named) |
| $y$, $P = M(D)$, $Q = M(D')$ | An output; the two output distributions being compared |
| $D' = D \cup \{x\}$; $a$, $b$ | As in the DP foundations part, under add/remove $D'$ is the dataset that holds the record $x$ and $D$ the one without it. A test of $P$ against $Q$ has type I error $a$, the false-positive rate of saying $D'$ when the data were $D$, and type II error $b$, the rate of saying $D$ when they were $D'$. A guarantee covers both orders of the pair; a chapter that needs one order says so, as the subsampling chapters do for the remove direction, which puts the dataset with the record first |
| $L_{D,D'}(y) = \ln\frac{p_{M(D)}(y)}{p_{M(D')}(y)}$ | The privacy loss of output $y$; $L_{D,D'}(Y)$ with $Y \sim M(D)$ is the privacy loss random variable |
| $\delta(\varepsilon) = E_{e^\varepsilon}(P\|Q)$ | The smallest $\delta$ for a given $\varepsilon$, as a hockey-stick divergence |
| $D_\alpha$, $\varepsilon(\alpha)$ | Rényi divergence of order $\alpha > 1$; the RDP curve of a mechanism |
| $\rho$ | The zCDP parameter |
| $T(P, Q)$, $G_\mu$, $\mu$ | Trade-off function; the Gaussian trade-off function $T(\mathcal N(0,1), \mathcal N(\mu, 1))$; the parameter of Gaussian differential privacy, so a Gaussian mechanism with noise multiplier $\sigma$ has $\mu = 1/\sigma$ |
| $\Delta_2 f$, $\sigma$ | $\ell_2$ sensitivity; noise multiplier (noise standard deviation per unit of sensitivity) |
| $q$, $k$ | Sampling rate; number of composed steps |
| $h$ | Grid width of a discretized privacy loss distribution |

: Notation used throughout this part, in addition to that of the earlier parts. {#tbl-acct-notation}

Two symbols do double duty, and each chapter warns you when it happens. $\alpha$ is the Rényi order throughout this part, but the f-DP literature uses it for the type I error, so [f-DP and Gaussian differential privacy](04-gaussian-dp.md) renames the errors. $q$ is the sampling rate, so the chapters that subsample write output densities with capital letters, such as $Q(y)$ or $P_q(y)$, never $q(y)$.

## What you will be able to do

By the end of this part you should be able to:

- compute the RDP curve of the Gaussian, Laplace and randomized-response mechanisms, compose them, and convert the result to $(\varepsilon, \delta)$ with both the classic and the improved conversion;
- explain what an RDP curve, a zCDP parameter and a trade-off function each keep and lose about the privacy loss;
- derive privacy amplification by subsampling, and compute the RDP of the Poisson-subsampled Gaussian;
- build a privacy-loss-distribution accountant with pessimistic and optimistic error bounds, and use it to check any other accountant;
- say when shuffling or hiding intermediate iterates gives extra privacy, and when it does not apply;
- run a privacy filter, and compute per-record privacy spend in DP-SGD;
- pick an accountant for a DP-SGD run, state its assumptions, and report $\varepsilon$ in a way that someone else can check.
