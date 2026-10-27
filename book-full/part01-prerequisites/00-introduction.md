# Prerequisites

This part collects the mathematics that the rest of the book uses. It is standard probability, statistics, optimization, information theory and cryptography, with no ideas about privacy yet. Each topic is here because a later part needs it, and each goes only as far as that need, but always with full derivations.

## Who this part is for

You need undergraduate calculus and linear algebra, the basics of probability (random variables, expectation, variance, independence, conditional probability), and enough machine learning to recognise a loss function, a training set and stochastic gradient descent. You also need to read short Python programs that use NumPy. This part derives everything else that later parts use. If you already know a topic well, read the Summary table at the end of its chapter, and move on if every result in it is familiar.

## The common thread

Later parts study algorithms whose output is random, and they keep asking the same few questions about them. How far is a random output from its expected value? How different are the output distributions on two slightly different inputs, and how would anyone tell them apart? How much does a trained model depend on one training record? How much does an observation reveal about something hidden, and what can an observer with limited computing power learn from it? Each chapter of this part gives the tools for one of these questions (@tbl-prereq-map).

| Chapter | The question it answers | Builds on |
| --- | --- | --- |
| [Probability distributions](01-distributions.md) | Which few distributions model counts, waiting times and additive noise, and what are their moments and tails? | — |
| [Concentration inequalities](02-concentration.md) | How far can a random quantity be from its mean, and with what probability? | 1 |
| [Divergences](03-divergences.md) | How different are two probability distributions, as a single number? | 1, 2 |
| [Hypothesis testing](04-hypothesis-testing.md) | Given one observation, how well can you tell which of two distributions produced it? | 1, 2, 3 |
| [Information theory](05-information-theory.md) | How much does an observation tell you about a hidden value, on average and for a single best guess? | 3 |
| [Optimization](06-optimization.md) | How fast does (stochastic) gradient descent converge, and how far does the minimizer move when one record changes? | 1, 2 |
| [Algorithmic stability and generalization](07-stability-generalization.md) | Why does a model that depends little on each training record generalize? | 2, 6 |
| [Cryptography basics](08-crypto-basics.md) | What does it mean that an efficient observer learns nothing it was not supposed to learn? | 2, 3, 4 |

: The chapters of this part, the question each one answers, and the earlier chapters it uses. {#tbl-prereq-map}

Two ideas run through all eight chapters, and most later parts are built from them. The first is the **likelihood ratio** $p(x)/q(x)$, which says how many times more likely an observation is under one distribution than under another. A divergence is an average of a function of it, the best test compares it with a threshold, and mutual information is a divergence too. The second is the **effect of one input**, which is how much one record moves an average, a minimizer, a loss or a test statistic. Concentration inequalities bound it for sums, and stability bounds it for learning algorithms. The sensitivity of regularized empirical risk minimization makes it a concrete number.

## How to read this part

**Order.** Read chapters 1 to 4 in order, because each one uses the ones before. They are the core of this part. Tail bounds come first, then a number for how different two distributions are, then what that number means for anyone who tries to tell them apart. Chapter 5, information theory, continues that thread and measures how much an observation reveals. Chapters 6 and 7 go together: how training moves when one record changes, and why that matters for generalization. You can leave chapter 8 until chapter 9, which uses its security games. Chapter 15 on linkage uses its birthday bound, and chapter 26 on trust models builds secure aggregation from its secret sharing.

**Structure of a chapter.** Every chapter opens with its question and a running example, then gives one section per topic. Its definitions, theorems and examples are numbered so that later chapters can point to them. Each theorem has its proof right after it, in numbered steps. A **Code** section runs the script that checks every quoted number, and a **Summary** table lists each result with where it is used.

**Abstract steps are made concrete.** An abstract step, such as writing a probability as the average of an indicator, is first worked out on a small case, usually a die or a coin with a table of every outcome. If a step feels like a trick, look for that table.

**Run the code.** Each chapter has a script in `code/` that reproduces every number and figure in its text. From the repository root, run, for example,

```bash
uv run python code/part01-prerequisites/concentration.py
```

The scripts use fixed random seeds, so your output matches the tables in the text. To get a feel for how loose each bound is, change the parameters and watch how the bounds respond.

## Notation

The chapters share the conventions in @tbl-prereq-notation, and a chapter that departs from them says so at its start.

| Symbol | Meaning |
| --- | --- |
| $\Pr[A]$, $\mathbb E[X]$, $\operatorname{Var}(X)$ | Probability, expectation, variance |
| $\mathbf 1[A]$ | Indicator of the event $A$: 1 if $A$ happens, 0 otherwise |
| $\mathcal N(\mu, \sigma^2)$, $\mathrm{Lap}(b)$ | Normal distribution with mean $\mu$ and variance $\sigma^2$; Laplace distribution with scale $b$ |
| $P, Q$ and $p, q$ | Two distributions and their probability mass functions or densities |
| $\ln$, $\log_2$ | Natural logarithm (in nats); base-2 logarithm (in bits), used in information theory |
| $\|\cdot\|$ | Euclidean norm, $\|x\| = \|x\|_2$ |
| $D = (z_1, \dots, z_n)$, $z = (x, y)$ | A dataset of $n$ records; a labelled record with features $x$ and label $y$ |
| $\theta \in \mathbb R^d$, $\ell(\theta; z)$ | Model parameters of dimension $d$; the loss of $\theta$ on the record $z$ |
| $\mathcal L(\theta; D)$, $\mathcal L_{\mathcal D}(\theta)$ | Empirical risk on $D$; population risk under the data distribution $\mathcal D$ |
| $\eta$, $T$ | Learning rate; number of steps |
| $L$, $\beta$, $\lambda$ | Lipschitz, smoothness and strong convexity constants |

: Notation shared by the chapters of this part. {#tbl-prereq-notation}

## What you will be able to do

By the end of this part you can do each of the following on paper, because each is a result derived in one of the chapters that follow:

- Derive exponential tail bounds for sums of independent variables, and for normal and Laplace noise.
- Compute the Kullback–Leibler (KL), Rényi, total-variation and hockey-stick divergences between two normal or two Laplace distributions. Explain why processing data cannot increase them.
- Derive the optimal test between two distributions and its trade-off curve between the two kinds of error.
- Prove convergence rates for stochastic gradient descent, and bound how far a regularized minimizer moves when one training record is replaced.
- Prove that a uniformly stable algorithm generalizes.
- Compute entropy, mutual information and min-entropy leakage, and say when the last two disagree.
- State a security definition as a game between two worlds, and prove a simple scheme secure with a hybrid argument or a simulator.
