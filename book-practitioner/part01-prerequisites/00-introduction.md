# Prerequisites

This part introduces two mathematical tools used throughout the book. Later chapters link to each result where they use it, so you can read this part as needed.

## What each chapter is for

Both chapters use the **likelihood ratio**, the relative probability of an observation under two explanations. Later chapters use it to compare privacy guarantees and attacks.

[Hypothesis testing](04-hypothesis-testing.md) shows how to read a detector's trade-off curve at a low false positive rate and compute an interval for a measured rate. Attacks, audits and the Gaussian view of accounting use these results. [Divergences](03-divergences.md) is a short reference page on measures that compare two distributions; the definitions in Parts 3 and 4 use them.

The rest of the mathematics is in the full edition's Part 1, which a page here links to where it uses a fact: [probability distributions](../../book-full/part01-prerequisites/01-distributions.md), [concentration inequalities](../../book-full/part01-prerequisites/02-concentration.md), [information theory](../../book-full/part01-prerequisites/05-information-theory.md), [optimization](../../book-full/part01-prerequisites/06-optimization.md), [algorithmic stability and generalization](../../book-full/part01-prerequisites/07-stability-generalization.md) and [cryptography basics](../../book-full/part01-prerequisites/08-crypto-basics.md).

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

Each chapter states its results, works an example with real numbers and links to the derivations in the full edition. A snippet or a seeded script in the full edition checks every number, so each run reproduces it.
