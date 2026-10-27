# Prerequisites

This part is a toolkit of mathematics, with no privacy in it yet. You need not read it first, because later chapters link to each fact they use.

## What each chapter is for

Two ideas run through the part. The first is the **likelihood ratio**: how many times more likely an observation is under one explanation than under another. Divergences and tests use it. The second is the **effect of one input**: how far one record can move a result, which concentration and stability bound.

[Hypothesis testing](04-hypothesis-testing.md) is the one full chapter, because the rest of the book uses it most. It shows how to read a detector's trade-off curve at a low false positive rate, and how to put an honest interval on a measured rate.

The other seven are short reference pages. [Probability distributions](01-distributions.md) gives the noise families, and [Concentration inequalities](02-concentration.md) the error bars and sample sizes. [Divergences](03-divergences.md) compares two distributions with one number, and [Information theory](05-information-theory.md) bounds what an output shows about a hidden value. [Optimization](06-optimization.md) gives the learning rates of gradient descent. [Algorithmic stability and generalization](07-stability-generalization.md) explains why a holdout set used again misleads. [Cryptography basics](08-crypto-basics.md) shows how to read a security claim stated as a game.

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

Each page states its results, works one example with real numbers, and links to its derivations in the full edition. Every number comes from a snippet on the page or from the full edition's seeded script for the chapter, so a new run gives the same numbers.
