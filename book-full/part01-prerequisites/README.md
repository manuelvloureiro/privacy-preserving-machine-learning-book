# Part 1 — Prerequisites

The mathematics the rest of the book stands on: probability distributions, concentration, divergences, hypothesis testing, optimization, stability, information theory and the basics of cryptography. The part teaches mathematics only. Privacy definitions and mechanisms start in later parts, which build on exactly these tools.

Start with the [introduction](00-introduction.md), which is part of the book. The goal is not a full course in each area. Refresh only what you can't derive on demand. Most machine-learning backgrounds are weakest on divergences and hypothesis testing (chapters 3 and 4), so spend most of your time there.

This file is the GitHub overview. It is not part of the rendered book.

## Chapters

| Ch. | Chapter | Covers | Script |
|-----|---------|--------|--------|
| 1 | [Probability distributions](01-distributions.md) | Bernoulli, binomial, Poisson, exponential, Laplace, normal; moments and moment generating functions; normal vectors in high dimension; a density against its shifted copy | [distributions.py](../../code/part01-prerequisites/distributions.py) |
| 2 | [Concentration inequalities](02-concentration.md) | Markov, Chebyshev, Chernoff, sub-Gaussian variables, Hoeffding, Bernstein; Gaussian and Laplace tails; McDiarmid and Azuma | [concentration.py](../../code/part01-prerequisites/concentration.py) |
| 3 | [Divergences](03-divergences.md) | Kullback–Leibler, Rényi, total variation, hockey-stick, f-divergences; data processing; Pinsker | [divergences.py](../../code/part01-prerequisites/divergences.py) |
| 4 | [Hypothesis testing](04-hypothesis-testing.md) | Type I and II errors, Neyman–Pearson, trade-off functions and ROC curves, Gaussian and Laplace curves, AUC at low false positive rates, estimating rates from trials | [hypothesis_testing.py](../../code/part01-prerequisites/hypothesis_testing.py) |
| 5 | [Information theory](05-information-theory.md) | Entropy, mutual information, channel capacity, min-entropy and guessing, Shannon against min-entropy leakage, Fano | [information_theory.py](../../code/part01-prerequisites/information_theory.py) |
| 6 | [Optimization](06-optimization.md) | Convexity, Lipschitz and smooth losses, SGD convergence (convex, strongly convex, non-convex), gradient noise, how far a minimizer moves when one example changes | [optimization.py](../../code/part01-prerequisites/optimization.py) |
| 7 | [Algorithmic stability and generalization](07-stability-generalization.md) | Uniform stability, the expected and high-probability generalization bounds, stability of regularized ERM and of SGD, memorization | [stability.py](../../code/part01-prerequisites/stability.py) |
| 8 | [Cryptography basics](08-crypto-basics.md) | Negligible functions, indistinguishability and hybrids, PRGs and PRFs, IND-CPA, the simulation paradigm, learning with errors | [crypto_basics.py](../../code/part01-prerequisites/crypto_basics.py) |

Each script prints the numbers its chapter quotes and regenerates the chapter's figures in `_build/figures/part01-prerequisites/` (the build copies them into each book):

```bash
uv run python code/part01-prerequisites/<script>.py
```

Every chapter's exercises, each with a worked solution, are in the exercise book, [book-exercises/part01-prerequisites/](../../book-exercises/part01-prerequisites/), one file per chapter. The exercise book is its own website, and `code/part01-prerequisites/solutions_<script>.py` prints the numbers they quote.

## Suggested order

1. Chapters 1 → 2 → 3 → 4, in order. Each uses the previous one: distributions give the moment generating functions that concentration needs, concentration bounds tails, divergences summarize how different two distributions are, and hypothesis testing says what an observer can do with that difference.
2. Then chapter 5, information theory, which builds on divergences.
3. Then chapters 6 → 7. Stability uses the optimization results directly.
4. Chapter 8 last. It can wait until chapter 9 (threat models, which uses its security games), and chapters 15 (linkage, which uses the birthday bound) and 25 (trust models, which use secret sharing) build on it too, as will a planned later part on cryptographic approaches.

## Core readings (from the study plan)

- Vershynin, *High-Dimensional Probability* (2018), ch. 2 → chapter 2
- van Erven & Harremoës, "Rényi Divergence and Kullback–Leibler Divergence" (2014), sections 1–3 → chapter 3
- Lindell, "How to Simulate It — A Tutorial on the Simulation Proof Technique" (2017) → chapter 8

## Further readings

- Bousquet & Elisseeff, "Stability and Generalization" (JMLR 2002) → chapter 7
- Cover & Thomas, *Elements of Information Theory*, ch. 2 → chapter 5
- Boneh & Shoup, *A Graduate Course in Applied Cryptography*, ch. 2–3 → chapter 8

Each chapter ends with its own references, with what each is good for.

## Study-plan exercises

1. Derive the Chernoff bound for a sum of Bernoullis: worked step by step in [Concentration inequalities](02-concentration.md), section "The Chernoff method".
2. Compute the KL and Rényi divergence between two Gaussians with equal variance by hand: worked two ways in [Divergences](03-divergences.md), section "Two Gaussians with equal variance, by hand".

Try both before reading the solutions. Every chapter also has its own exercises in the exercise book, most with check values.

## Exit check

You are done with this part when you can do each of these on paper, without notes:

- [ ] Derive the mean, variance and moment generating function of the binomial, Poisson, Laplace and normal, and show that a difference of two exponentials is Laplace.
- [ ] Explain why the log ratio of a Laplace density and its shifted copy is bounded, while for the normal it is not.
- [ ] Prove Markov's inequality from an indicator, then derive Chebyshev and the Chernoff bound for a sum of Bernoullis.
- [ ] Prove Hoeffding's lemma, and say when Bernstein or Chernoff beats Hoeffding.
- [ ] Derive $D_\alpha(\mathcal N(\mu_0,\sigma^2)\,\|\,\mathcal N(\mu_1,\sigma^2)) = \alpha(\mu_0-\mu_1)^2/(2\sigma^2)$.
- [ ] Write total variation and the hockey-stick divergence as suprema over events, and prove the data-processing inequality for one of them.
- [ ] Prove the Neyman–Pearson lemma, and derive the trade-off curve between $\mathcal N(0,1)$ and $\mathcal N(\mu,1)$.
- [ ] Explain why AUC can hide large differences between two classifiers at a false positive rate of $10^{-3}$.
- [ ] Prove the $RL/\sqrt T$ rate of projected SGD for convex Lipschitz objectives.
- [ ] Prove that the minimizer of a $\lambda$-strongly convex regularized ERM moves by at most $2L/(\lambda n)$ when one example is replaced.
- [ ] Prove that uniform stability $\beta$ implies an expected generalization gap of at most $\beta$.
- [ ] Compute the min-entropy leakage of a small channel, and give a channel where Shannon and min-entropy leakage disagree badly.
- [ ] Define a negligible function and IND-CPA security, explain why encryption must be randomized, and write a simulator for additive secret sharing.
- [ ] State the learning with errors problem, and say why it is easy without the noise.
