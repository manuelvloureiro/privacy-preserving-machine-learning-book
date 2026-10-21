# Prerequisites

This part is a mathematical toolkit, and it contains no privacy yet. You do not need to read it first. Later chapters link back to the facts they use, so come here when a link sends you.

## What each chapter is for

Two ideas run through the part. The first is the **likelihood ratio**, how many times more likely an observation is under one explanation than under another. The second is the **effect of one input**, how far a single example can move a result.

[Hypothesis testing](04-hypothesis-testing.md) is the one full chapter, because the rest of the book leans on it most. It shows how to read a detector's trade-off curve at a low false positive rate. It also shows how to put an honest interval on a rate measured from a few hundred cases.

The other seven are short reference pages, each with the result, when you meet it, how to use it and what goes wrong. [Probability distributions](01-distributions.md) and [Concentration inequalities](02-concentration.md) give you the noise families, their scales in NumPy and SciPy, and the error bars and sample sizes you attach to a measured number. [Divergences](03-divergences.md) compares two distributions with a single number. [Information theory](05-information-theory.md) bounds what an output reveals about a hidden value. [Optimization](06-optimization.md) gives the constants and step sizes of gradient descent, and [Algorithmic stability and generalization](07-stability-generalization.md) explains why a reused holdout set misleads you. [Cryptography basics](08-crypto-basics.md) teaches you to read a security claim stated as a game.

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

Each page states its results, shows how to use them with a worked number, and links to its derivations in the full edition. Every number comes from a snippet on the page or from the full edition's script for the same chapter, which you can rerun, such as `uv run python code/part01-prerequisites/optimization.py`. The code draws its randomness from seeded NumPy generators, `numpy.random.default_rng`, so reruns give the same numbers.
