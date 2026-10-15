# Differential privacy foundations

The previous part showed that models and statistics leak: membership, attributes, training text, whole datasets. Each attack found a difference between what a system releases with your record and without it. This part introduces the defence the rest of the book is built on, **differential privacy (DP)**, which makes that difference provably small, and takes it far enough for you to build a private release, check one somebody else built, and say what it costs. The finer tools for thousands of training steps come in the [next part](../part04-privacy-accounting/00-introduction.md).

## Who this part is for

Read it if you build, review or audit a system that publishes statistics or trains models on personal data and claims a privacy guarantee. A claim such as "trained with $\varepsilon = 8$, $\delta = 10^{-5}$" is only useful if you can say what it promises and to whom, check the noise behind it, and spot the mistakes that silently void it. The part assumes the attack games of [Threat models and attacks](../part02-attacks/00-introduction.md) and a little probability from the [Prerequisites](../part01-prerequisites/00-introduction.md): the Laplace and normal distributions, and the error rates of a test.

## The common thread

A randomized algorithm is differentially private when changing one person's record changes the probability of any output by at most a factor $e^\varepsilon$, up to a small slack $\delta$. The whole part reads that sentence in three ways, each answering a practical question.

Read as a **bound on every attack**, it tells you what the guarantee is worth. An attacker asking whether you are in the data ends by saying yes or no, and the definition caps how much more often they can say yes when it is true than when it is false. So a membership attack's true positive rate is at most $e^\varepsilon$ times its false positive rate, plus $\delta$, whatever the attacker knows and however much they compute. That is how you turn $\varepsilon$ into something a risk owner understands, and, run backwards, how an attack can prove a claimed $\varepsilon$ wrong.

Read as a **recipe for mechanisms**, it tells you how to build a release. If one record can move an answer by at most some amount, its **sensitivity**, then noise that is wide compared with that amount makes the two worlds look alike. The size of the noise follows from the sensitivity and $\varepsilon$, so most of the work is bounding how far one record can move the answer and choosing the cheapest noise that hides it.

Read as a **budget to spend**, it tells you how guarantees combine. Anything computed from a private output is free, because computing cannot add evidence the output didn't contain. But every new look at the data adds its own evidence, so privacy losses add up over releases, and over the many records one person may own.

## How to read this part

The chapters follow the order in which those questions come up when you build something. [The definition](01-definition.md) comes first, because every later claim is checked against it. [Post-processing, group privacy and composition](02-properties.md) comes next: they let you prove each building block once and assemble a system from blocks. Then [the Laplace and Gaussian mechanisms](03-laplace-gaussian.md) give the two building blocks almost everything else uses. [Advanced and optimal composition](04-composition.md) then revisits the budget, because adding up losses is safe but far too pessimistic for many releases; its exact accounting is what the next part builds on.

The next two chapters cover what noise on a number does not: [selection](05-selection.md), where the output is a choice among options, and [data whose worst case is far from typical](06-sensitivity.md), where noise sized for the worst case would drown the answer. [Discrete and floating-point-safe mechanisms](07-discrete-mechanisms.md) is short and practical: a mechanism private on paper can leak through how a computer draws random numbers, so read it before implementing one. [Local, shuffle and distributed trust models](08-trust-models.md) asks who sees the raw data. [Lower bounds](09-lower-bounds.md) comes last because it closes the loop with the previous part's attacks: it says how much accuracy any private release must lose, and so when a privacy target and an accuracy target cannot both be met.

Each chapter starts from the practical question, explains every result with the reason it holds and a worked example, and ends with a cheat sheet. The derivations are in the full edition, linked at the end of each chapter.

## What you will be able to do

By the end of this part you will be able to state a DP guarantee with its neighbouring relation and unit of privacy, and translate a claimed $(\varepsilon, \delta)$ into a cap on what an attacker can achieve. You will be able to compute the sensitivity of a query, calibrate Laplace and Gaussian noise exactly, add up the cost of a pipeline, and choose a $\delta$ that doesn't hide a catastrophe. You will know which mechanism fits a selection problem or data with a heavy worst case, why naive floating-point noise is not private, what the local and shuffle models buy, and what accuracy no private mechanism can beat.
