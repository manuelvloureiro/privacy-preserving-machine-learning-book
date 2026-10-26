# Prerequisites

This part collects the mathematics the rest of the book stands on. It contains no new ideas about privacy. Everything here is standard probability, statistics, optimization, information theory and cryptography, and you may have met most of it before. What is different is the selection and the angle: each topic is chosen because a later part needs it, and each is developed only as far as that need goes, but always with full derivations.

## Who this part is for

You are expected to know undergraduate calculus and linear algebra, the basics of probability (random variables, expectation, variance, independence, conditional probability), and enough machine learning to recognise a loss function, a training set and stochastic gradient descent. You should be able to read short Python programs that use NumPy. Nothing beyond that is assumed. Every inequality, divergence or definition that the later parts rely on is stated and derived here, so you can check each step for yourself instead of taking it on trust.

If you already know a topic well, read the Summary table at the end of its chapter and try its exercises in the exercise book. If they go smoothly, move on.

## The common thread

Later parts study algorithms whose output is random, and keep asking the same few questions about them. How far is a random output from its expected value? How different are the output distributions on two slightly different inputs, and how would anyone tell them apart? How much does a trained model depend on a single training record? How much does an observation reveal about something hidden, and what can an observer with limited computing power learn from it? Each chapter of this part supplies the vocabulary and tools for one of these questions (@tbl-prereq-map).

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

Two ideas run through all eight chapters. The first is the **likelihood ratio** $p(x)/q(x)$: how many times more likely an observation is under one distribution than under another. Divergences are averages of functions of it, the best test thresholds it, and mutual information is a divergence in disguise. The second is the **effect of one input**: how much one record shifts an average, a minimizer, a loss or a test statistic. Concentration inequalities bound it for sums, stability bounds it for learning algorithms, and the sensitivity of regularized empirical risk minimization makes it a concrete number. Watch for both as you read. Most of the later parts are built from them.

## How to read this part

**Order.** Read chapters 1 to 4 in order, since each one uses the previous ones. Together they form the core of this part: tail bounds, then a number for how different two distributions are, then what that number means for anyone trying to tell them apart. Chapter 5, information theory, continues that thread: it measures how much an observation reveals, on average and for a single best guess. Then read chapters 6 and 7, which go together: how training moves when one record changes, and why that matters for generalization. Chapter 8 is used by three later chapters: chapter 9 on threat models uses its security games and advantage, chapter 15 on linkage uses its birthday bound, and chapter 26 on trust models builds secure aggregation from its secret sharing. A planned later part on cryptographic approaches will use all of it. You can defer it until you reach chapter 9.

**Structure of a chapter.** Every chapter has the same shape, so you always know where to look. It opens with the question it answers, the one idea that organizes the answer, and a running example with its numbers. Then comes one section per topic, in teaching order. Each section opens with its question, and its definitions, theorems and worked examples are numbered so that later chapters can point to them. Every theorem is followed at once by its proof in numbered steps, and then by what it means and how tight it is. A **Code** section runs the script that checks every quoted number, a **Summary** table lists each result with where it is used, and **References** close the chapter. The exercises, each with a worked solution, are in the exercise book, a companion website, and the last line of each chapter links to them.

**Abstract steps are made concrete.** Where a derivation takes an abstract step, such as writing a probability as the expectation of an indicator or changing the distribution an expectation is taken under, the text works the step out on a small case first, usually a die or a coin with a table of every outcome. If a step feels like a trick, look for that table. It is there to show that the step is just bookkeeping.

**Run the code.** Each chapter has a script in `code/` that reproduces every number and figure in its text. From the repository root, run, for example,

```bash
uv run python code/part01-prerequisites/concentration.py
```

The scripts use fixed random seeds, so your output matches the tables in the text. Change the parameters and watch how the bounds respond. That is the quickest way to develop a feel for how loose each bound is.

## Notation

The chapters share the conventions in @tbl-prereq-notation. Where a chapter departs from them, it says so at its start.

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

By the end of this part you should be able to do each of the following on paper:

- derive exponential tail bounds for sums of independent variables, and for normal and Laplace noise;
- compute the Kullback–Leibler (KL), Rényi, total-variation and hockey-stick divergences between two normal or two Laplace distributions, and explain why processing data cannot increase any of them;
- derive the optimal test between two distributions and its trade-off curve between the two kinds of error;
- prove convergence rates for stochastic gradient descent, and bound how far a regularized minimizer moves when one training record is replaced;
- prove that a uniformly stable algorithm generalizes;
- compute entropy, mutual information and min-entropy leakage, and say when the last two disagree;
- state a security definition as a game between two worlds, and prove a simple scheme secure with a hybrid argument or a simulator.

Each of these is a result derived in one of the chapters that follow.
