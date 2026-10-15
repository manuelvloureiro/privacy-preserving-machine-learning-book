# Privacy accounting

The previous part defined differential privacy (DP), built the basic mechanisms and showed how guarantees combine. This part answers the question that decides whether a trained model's privacy claim means anything. **After thousands of noisy training steps, each on a random subset of the data, how large is ε, really?** No single theorem gives the answer. It comes from an **accountant**, a procedure that tracks a summary of each step's privacy loss, combines the summaries, and converts the result into the ε and δ you report.

The running case is stochastic gradient descent (SGD) with per-example clipping and Gaussian noise, called DP-SGD. Two accountants given the same training run can report values of ε that differ by a large factor. Both can be correct upper bounds, but only the tighter one tells you what the noise actually bought.

You need the [Prerequisites](../part01-prerequisites/00-introduction.md) and [Differential privacy foundations](../part03-dp-foundations/00-introduction.md). In particular, this part uses the divergences and trade-off curves of Part 1, and the Gaussian mechanism and composition of Part 3. Each chapter restates in a sentence what it uses and links back.

## The common thread

Composition is easy for the privacy loss itself. If you run many mechanisms, the privacy loss of the whole run is the sum of the per-step losses. Composition is hard for a pair (ε, δ), because two numbers forget almost everything about the distribution of that sum. The rest of the field rests on one move. **Track a richer summary of the privacy loss, one that composes simply, and convert it to (ε, δ) only once, at the end.**

The summaries differ in how much they keep. A Rényi DP curve keeps more than a single (ε, δ) pair, but less than the full trade-off curve between an attacker's two errors. A single number for concentrated DP keeps less than the Rényi curve. Numerical accounting keeps the whole distribution of the privacy loss, and loses only a discretization error that it bounds. Each step down this ladder is cheaper to compute and to compose, and gives a looser ε.

Amplification is a separate question. It asks which pair of output distributions you should account for in the first place. Sampling a random batch, shuffling the data, or hiding the intermediate models can all make each step more private than its noise alone suggests. Every accountant must use the amplification result that matches the sampling the training code actually runs, or its ε is wrong.

## The running example

Every chapter that can compute it reports the same number, the ε at δ = 10⁻⁵ of one DP-SGD run. The run samples each example with probability 0.01 per step and uses a noise multiplier of 1. It lasts up to 10,000 steps, which is about one hundred epochs. Watching that number move from chapter to chapter shows what each tool gains. [Choosing an accountant](09-choosing-an-accountant.md) puts all the values side by side.

## Why the chapters come in this order

[Rényi differential privacy](01-renyi-dp.md) and [From Rényi DP to (ε, δ) and to tests](02-rdp-conversions.md) come first, because every later chapter uses the Rényi curve and its conversion. The second ends by reading the curve as a limit on hypothesis tests. [f-DP and Gaussian differential privacy](04-gaussian-dp.md) develops that view in full, and is best read next. [Concentrated differential privacy](03-concentrated-dp.md), the one-number summary, can be read any time after the conversions.

[Amplification by subsampling](05-subsampling.md) and [Numerical accounting with privacy loss distributions](06-numerical-accounting.md) are the practical core of the part. The first says how much a random batch helps, and the second computes the resulting ε as tightly as possible. Read them in that order. [Amplification by shuffling and by iteration](07-shuffling-iteration.md) builds on subsampling, and covers two other ways a training pipeline can gain privacy. [Adaptive composition](08-adaptive-composition.md) handles budgets chosen during training and accounting per record. [Choosing an accountant](09-choosing-an-accountant.md) closes the part and uses all the others.

## How to read a chapter

Each chapter starts from the practical question it answers, reasons its way to each result, and shows it at work on the running example. Every formula you need to compute or check an ε is kept and explained in words, together with the conditions it needs and what goes wrong without them. A cheat sheet at the end collects the results.

The chapters contain no proofs. Each one ends with a link to the matching chapter of the full edition. That chapter derives every result, and its script implements each accountant from its definition, without an accounting library, so you can see exactly what it computes.

## What you will be able to do

After this part, you will be able to compute the Rényi curve of the common mechanisms, combine curves across many steps, and convert the result to (ε, δ). You will know what a Rényi curve, a concentrated-DP parameter and a trade-off curve each keep and lose about the privacy loss.

You will be able to say how much random batching amplifies privacy, and under which sampling scheme. You will be able to compute a tight ε with a privacy-loss-distribution accountant, and use it to check any other accountant. You will also know when shuffling or hiding intermediate models gives extra privacy, and when it does not apply.

Finally, you will be able to stop training when a privacy budget runs out, track how much privacy each record has spent, and choose an accountant for a DP-SGD run. Above all, you will be able to report ε in a way that someone else can check.
