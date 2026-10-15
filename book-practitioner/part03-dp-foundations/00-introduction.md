# Differential privacy foundations

The previous part showed that models and statistics leak: membership, attributes, training text, whole datasets. This part introduces the defence the rest of the book is built on, **differential privacy (DP)**, and takes it as far as you need to build a private release, check one somebody else built, and say what it costs. Keeping track of privacy over thousands of training steps needs finer tools (Rényi DP, numerical accountants, amplification by subsampling). They come in the [next part](../part04-privacy-accounting/00-introduction.md), which builds directly on this one.

## Who this part is for

Read it if you build, review or audit a system that publishes statistics or trains models on personal data and claims a privacy guarantee. You need to be able to read a claim such as "trained with $\varepsilon = 8$, $\delta = 10^{-5}$" and say what it promises and to whom, to calibrate the noise of a release yourself, to add up the cost of a pipeline, and to recognise the implementation mistakes that silently void a guarantee. The part assumes the attack games of [Threat models and attacks](../part02-attacks/00-introduction.md), and a little probability from the [Prerequisites](../part01-prerequisites/00-introduction.md): the Laplace and normal distributions, and false positive and false negative rates of a test. Each chapter restates what it uses.

## The common thread

A randomized algorithm is differentially private when changing one person's record changes the distribution of its output by at most a factor $e^\varepsilon$, up to a small slack $\delta$. Every chapter of this part reads that one sentence in one of three ways.

As a **bound on every attack**, it caps what any adversary can achieve, whatever it knows and however much it computes. A membership attack's true positive rate can be at most $e^\varepsilon$ times its false positive rate, plus $\delta$, and you can compute that cap before the data are touched. This is how you translate $\varepsilon$ into something a risk owner understands, and how you audit a trained model by attacking it.

As a **recipe for mechanisms**, it says how much noise a release needs. The noise is calibrated to how far one record can move the answer, the **sensitivity**, and the chapters on the Laplace and Gaussian mechanisms, on selection, and on data-dependent sensitivity give the calibrations for the releases you will meet in practice, along with the samplers that keep them private on real computers.

As a **budget to spend**, it says how guarantees combine. Anything you compute from a private output is free, groups of records cost more than one record, and privacy losses add up over the releases made from the same data. The last chapter says how much accuracy no mechanism can avoid paying.

@tbl-dp-map lists the chapters and the practical question each one answers.

| Chapter | The practical question it answers |
| --- | --- |
| [The definition of differential privacy](01-definition.md) | What does a claimed $(\varepsilon, \delta)$ promise an attacker can and cannot do, and to whom? |
| [Post-processing, group privacy and composition](02-properties.md) | Which steps of a pipeline cost privacy, and how do the costs add up? |
| [The Laplace and Gaussian mechanisms](03-laplace-gaussian.md) | How much noise does a count, a mean or a gradient need? |
| [Advanced and optimal composition](04-composition.md) | What is the total cost of hundreds or thousands of releases, without overpaying? |
| [Selection: exponential mechanism, report-noisy-max, sparse vector](05-selection.md) | How do you privately pick the best option, or the few queries above a threshold? |
| [Beyond global sensitivity](06-sensitivity.md) | What if the worst case is far from the data you actually have? |
| [Discrete and floating-point-safe mechanisms](07-discrete-mechanisms.md) | Why does textbook noise break on real computers, and what should you use instead? |
| [Local, shuffle and distributed trust models](08-trust-models.md) | What changes when no one is trusted with the raw data? |
| [Lower bounds: what privacy costs](09-lower-bounds.md) | How much accuracy must any private release give up, and when is a target unreachable? |

: The chapters of this part and the practical question each one answers. {#tbl-dp-map}

## How to read this part

The first four chapters are the core, and they are best read in order: the definition, the properties that make it usable, the two mechanisms almost everything else is built from, and the composition rules that turn many releases into one guarantee. The next two add mechanisms for selection and for data whose worst case is far from typical. Read the chapter on discrete and floating-point-safe mechanisms before you implement any mechanism yourself. The chapter on trust models matters if you collect data from devices or across organisations, and the chapter on lower bounds tells you when a privacy target and an accuracy target cannot both be met.

Every chapter has the same shape. It opens with the practical question and a table of the decisions it covers, then gives each topic as an idea, a result, a worked example with real numbers and the pitfalls, and closes with a cheat sheet. The derivations are in the full edition, linked at the end of each chapter.

## What you will be able to do

By the end of this part you will be able to state a DP guarantee with its neighbouring relation and unit of privacy, and turn a claimed $(\varepsilon, \delta)$ into a cap on membership attacks and on how far an adversary's belief can move. You will be able to calibrate Laplace and Gaussian noise exactly, add up the cost of a pipeline of releases, and pick a $\delta$ that does not hide a catastrophe. You will know when to use the exponential mechanism, report-noisy-max or the sparse vector technique, how to handle data whose sensitivity depends on the data, why naive floating-point noise is not private, what the local and shuffle models buy, and what accuracy no private mechanism can beat.
