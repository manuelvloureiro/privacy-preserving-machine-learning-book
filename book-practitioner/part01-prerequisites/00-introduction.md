# Prerequisites

This part is a mathematical toolkit, and it contains no privacy yet. You do not need to read it first. Later chapters link back to the facts they use, so come here when a link sends you.

## What each chapter is for

Two ideas run through the part. The first is the **likelihood ratio**, how many times more likely an output is under one dataset than under a neighbouring one. The second is the **effect of one input**, how far a single example can move a result.

[Probability distributions](01-distributions.md) and [Concentration inequalities](02-concentration.md) give you the noise families, their scales and the error bars you publish with noisy numbers, which Parts 3 and 4 use constantly. [Divergences](03-divergences.md) compares the output distributions on two datasets with a single number, the language of Parts 3 and 4. [Hypothesis testing](04-hypothesis-testing.md) shows how to read a detector's trade-off curve at low false positive rates, which Parts 2 and 5 rely on. [Information theory](05-information-theory.md) bounds what an output reveals about a hidden value, which Part 2 uses.

[Optimization](06-optimization.md) gives the constants, step sizes and clipping that Part 5 needs. [Algorithmic stability and generalization](07-stability-generalization.md) explains why regularization and early stopping limit how much a model depends on one example, and why a reused holdout set misleads you. [Cryptography basics](08-crypto-basics.md) teaches you to read a security claim stated as a game or a simulator, which later parts need.

## Notation

| Symbol | Meaning |
| --- | --- |
| $\Pr$, $\mathbb E$, $\mathrm{Var}$ | Probability, expectation, variance |
| $\mathcal N(\mu, \sigma^2)$, $\mathrm{Lap}(b)$ | Normal and Laplace distributions |
| $\ln$, $\log_2$ | Natural logarithm (nats) and base-2 logarithm (bits) |
| $\lVert w\rVert$ | Euclidean length of a vector |
| $D$, $\ell(w, z)$ | A dataset, and the loss of a model $w$ on an example $z$ |

: Notation used throughout the book. {#tbl-prereq-notation}

## How to use the chapters

Each chapter states its results, shows how to use them with a worked number, and links to its derivations in the full edition. Every number comes from a script you can rerun, such as `uv run python code/part01-prerequisites/concentration.py`. The scripts draw their randomness from seeded NumPy generators, `numpy.random.default_rng`, so reruns give the same numbers.
