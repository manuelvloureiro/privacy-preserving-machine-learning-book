# Prerequisites

This part is a mathematical toolkit with no privacy in it yet. You do not need to read it first, because later chapters link back to each fact they use.

## What each chapter is for

Two ideas run through the part. The first is the **likelihood ratio**, how many times more likely an observation is under one explanation than under another, and divergences and tests are built on it. The second is the **effect of one input**, how far a single record can move a result, which concentration and stability bound.

[Hypothesis testing](04-hypothesis-testing.md) is the one full chapter, because the rest of the book leans on it most. It shows how to read a detector's trade-off curve at a low false positive rate, and how to put an honest interval on a measured rate.

The other seven are short reference pages. [Probability distributions](01-distributions.md) and [Concentration inequalities](02-concentration.md) give the noise families and the error bars and sample sizes for a measured number. [Divergences](03-divergences.md) compares two distributions with one number, and [Information theory](05-information-theory.md) bounds what an output reveals about a hidden value. [Optimization](06-optimization.md) gives the learning rates of gradient descent, and [Algorithmic stability and generalization](07-stability-generalization.md) explains why a reused holdout set misleads. [Cryptography basics](08-crypto-basics.md) shows how to read a security claim stated as a game.

## Notation

| Symbol | Meaning |
| --- | --- |
| $\Pr$, $\mathbb E$, $\operatorname{Var}$ | Probability, expectation, variance |
| $\mathcal N(\mu, \sigma^2)$, $\mathrm{Lap}(b)$ | Normal and Laplace distributions |
| $\ln$, $\log_2$ | Natural logarithm (nats) and base-2 logarithm (bits) |
| $\|x\|$ | Euclidean length of a vector $x$ |
| $D$, $\theta$, $\ell(\theta; z)$ | A dataset, the parameters of a model, and its loss on one record $z$ |

: Notation used throughout the book. {#tbl-prereq-notation}

## How to use the chapters

Each page states its results, works one example with real numbers, and links to its derivations in the full edition. Every number comes from a snippet on the page or from the full edition's seeded script for the chapter, such as `uv run python code/part01-prerequisites/optimization.py`, so reruns give the same numbers.
