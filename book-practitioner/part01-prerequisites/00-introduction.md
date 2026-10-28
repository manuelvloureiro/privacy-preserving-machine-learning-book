# Prerequisites

This part holds the two pieces of mathematics that the rest of the book uses most, with no privacy in them yet. You need not read it first, because later chapters link to each fact they use.

## What each chapter is for

Both chapters rest on one idea, the **likelihood ratio**: how many times more likely an observation is under one explanation than under another. The rest of the book measures privacy with it, so these are the two pieces of mathematics a practitioner uses most.

[Hypothesis testing](04-hypothesis-testing.md) is a full chapter. It shows how to read a detector's trade-off curve at a low false positive rate, and how to put an honest interval on a measured rate. Attacks, audits and the Gaussian view of accounting all rest on it. [Divergences](03-divergences.md) is a short reference page. It compares two distributions with one number, and the definitions of Part 3 and Part 4 are such numbers.

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

Each chapter states its results, works one example with real numbers, and links to its derivations in the full edition. Every number comes from a snippet on the page or from the full edition's seeded script for the chapter, so a new run gives the same numbers.
