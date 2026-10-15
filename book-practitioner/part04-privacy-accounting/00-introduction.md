# Privacy accounting

The [previous part](../part03-dp-foundations/00-introduction.md) defined differential privacy (DP), built the basic mechanisms and showed how $(\varepsilon, \delta)$ guarantees add up. This part answers the question that decides whether a trained model's privacy claim means anything: **after thousands of noisy steps, each on a random subset of the data, how large is $\varepsilon$, really?** No single theorem gives the answer. It comes from an **accountant**, a procedure that tracks a summary of each step's privacy loss, combines the summaries, and converts the result into the $(\varepsilon, \delta)$ you report. Two accountants given the same training run can report values of $\varepsilon$ that differ by a large factor, both correct upper bounds; only the tighter one tells you what the noise bought.

## Who this part is for

Read it if you train models with DP, or review or audit a model that claims an $\varepsilon$. You should come away able to compute or check the $\varepsilon$ of a training run yourself, to choose and configure an accountant, and to say which assumptions each accountant makes about the training code. The part assumes [Differential privacy foundations](../part03-dp-foundations/00-introduction.md): the definition, the privacy loss of an output, the Gaussian mechanism and its exact curve $\delta(\varepsilon)$, and composition. Each chapter restates in a sentence what it uses.

## The common thread

The privacy loss of an output is the log of how much more likely that output is with your record than without it. Over a run of $k$ steps, the privacy loss of the whole run is the sum of the per-step losses, so composition is easy for the loss itself. It is hard for $(\varepsilon, \delta)$, because a pair of numbers forgets almost everything about the distribution of that sum. The whole field rests on one move: **track a richer summary of the privacy loss, one that composes simply, and convert to $(\varepsilon, \delta)$ only once, at the end.**

The summaries differ in how much they keep. A Rényi differential privacy (RDP) curve keeps the exponential moments of the loss and adds up step by step. A single zero-concentrated DP parameter $\rho$ keeps less, a trade-off function keeps more, and a numerical accountant keeps the whole distribution of the loss, losing only a discretization error that it bounds. Amplification is a separate question, which pair of output distributions you account for, and it must match how the training code actually samples its data. @tbl-acct-map lists the chapters.

| Chapter | The practical question it answers |
| --- | --- |
| [Rényi differential privacy](01-renyi-dp.md) | Which summary of the privacy loss adds up over thousands of steps? |
| [From Rényi DP to (ε, δ) and to tests](02-rdp-conversions.md) | Which conversion turns an RDP curve into the $(\varepsilon, \delta)$ you report, and what does it cost? |
| [Concentrated differential privacy](03-concentrated-dp.md) | When does one number $\rho$ describe a mechanism, and when does it mislead? |
| [f-DP and Gaussian differential privacy](04-gaussian-dp.md) | What is the most informative description of a mechanism's privacy, and how does it read as attack power? |
| [Amplification by subsampling](05-subsampling.md) | How much privacy does using each record in only a random fraction of the steps buy? |
| [Numerical accounting with privacy loss distributions](06-numerical-accounting.md) | How do you compute the tightest $\varepsilon$ of a run, with error bars? |
| [Amplification by shuffling and by iteration](07-shuffling-iteration.md) | When do shuffling or hiding intermediate models give extra privacy, and when not? |
| [Adaptive composition: filters, odometers and individual accounting](08-adaptive-composition.md) | What changes when budgets are chosen on the fly, or differ per record? |
| [Choosing an accountant](09-choosing-an-accountant.md) | Which accountant should you use for a training run, and what should you report? |

: The chapters of this part and the practical question each one answers. {#tbl-acct-map}

## The running example

Every chapter that can compute it reports the same number: the $\varepsilon$ at $\delta = 10^{-5}$ of the differentially private version of stochastic gradient descent (SGD), called DP-SGD, which clips each example's gradient and adds Gaussian noise. It runs with Poisson sampling rate $q = 0.01$, noise multiplier $\sigma = 1$ and up to $k = 10{,}000$ steps. That is about one hundred epochs at a typical setting. A single Gaussian mechanism with $\sigma = 1$ serves as the small example. Watching the same number move from chapter to chapter shows what each tool gains, and [Choosing an accountant](09-choosing-an-accountant.md) puts all the values side by side.

## How to read this part

Read the first two chapters first: every later chapter uses the RDP curve and its conversion. Then read the chapters on subsampling and numerical accounting, in that order; they are the practical core, because they are what a modern DP-SGD accountant computes. The chapters on concentrated DP and on f-DP explain the other summaries you will meet in papers and libraries. Read the chapter on shuffling and iteration if your training code shuffles instead of sampling or releases only the final model, and the chapter on adaptive composition if your budgets are chosen while training runs. Read the last chapter last.

Every chapter opens with the practical question and a table of the decisions it covers, gives each topic as an idea, a result, a worked example and the pitfalls, and closes with a cheat sheet. The derivations are in the full edition, linked at the end of each chapter.

## What you will be able to do

By the end of this part you will be able to compute the RDP curve of a Gaussian training step, compose a run of them, and convert the result to $(\varepsilon, \delta)$ with the right conversion. You will know what an RDP curve, a zero-concentrated DP parameter and a trade-off function each keep and lose, how much subsampling buys and which sampling scheme it assumes, and how to use a numerical accountant to check any other. You will be able to tell when shuffling or hidden iterates give extra privacy, run a privacy filter, and, for a DP-SGD run, pick an accountant, state its assumptions, and report $\varepsilon$ in a way someone else can check.
