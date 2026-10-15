# Prerequisites

This part collects the mathematics the rest of the book stands on. It has no privacy in it: it is standard probability, statistics, optimization, information theory and cryptography, chosen because a later part needs it and taken only as far as that need goes. Each chapter gives you the facts in the form you will use them, such as how wide a noisy answer is, how to compute a divergence between two noise distributions, or why a detector should be judged by its true positive rate at a low false positive rate.

## Who this part is for

Read it if a later chapter quotes a tail bound, a divergence or a trade-off curve and you want to know what it says and how to compute it. You need the basics of probability (expectation, variance, independence), enough machine learning to recognise a loss function and stochastic gradient descent (SGD), and enough Python to read a short script. If you already know a topic, skim its cheat sheet and move on; each later chapter restates in a sentence what it uses and links back here.

## The common thread

Later parts study algorithms whose output is random, and keep asking the same few questions about them. How far is a random output from its expected value? How different are the output distributions on two inputs that differ in one example, and how well could anyone tell them apart? How much does a trained model depend on a single training example? How much does an observation reveal about something hidden, and what can an observer with limited computing power learn?

Two ideas answer most of them. The first is the **likelihood ratio** $p(x)/q(x)$: how many times more likely an observation is under one distribution than under another. Divergences summarize it, the best test thresholds it, and a bounded ratio caps what any observer can conclude from one observation. The second is the **effect of one input**: how far one example can move an average, a minimizer or a loss. Concentration bounds turn it into error bars, and stability turns it into generalization. Watch for both. @tbl-prereq-map lists the chapters and the practical question each one answers.

| Chapter | The practical question it answers |
| --- | --- |
| [Probability distributions](01-distributions.md) | What do counts, waiting times and normal or Laplace noise look like, and how heavy are their tails? |
| [Concentration inequalities](02-concentration.md) | How wide must an error bar be to fail with probability at most $\beta$? |
| [Divergences](03-divergences.md) | How different are two distributions, which number answers which question, and how do I compute it? |
| [Hypothesis testing](04-hypothesis-testing.md) | What is the best test between two distributions, which summary of it should I report, and how many trials does it take to measure? |
| [Information theory](05-information-theory.md) | How much does an observation reveal about a hidden value, on average and for a single best guess? |
| [Optimization](06-optimization.md) | How fast does gradient descent converge, and how far does the minimizer move when one example changes? |
| [Algorithmic stability and generalization](07-stability-generalization.md) | Why does a model that depends little on each training example generalize? |
| [Cryptography basics](08-crypto-basics.md) | What does it mean that an efficient observer learns nothing it was not supposed to learn? |

: The chapters of this part and the practical question each one answers. {#tbl-prereq-map}

## How to read this part

Chapters 1 to 4 are the core and build on one another: the distributions, then error bars, then a number for how different two distributions are, then what that number means for anyone trying to tell them apart. Chapter 5 continues that thread with what an observation reveals. Chapters 6 and 7 go together, on how training moves when one example changes and why that matters for generalization. Chapter 8 is needed for the security games of the threat-model chapter and for the cryptographic tools of later parts, and can wait until then.

Every chapter has the same shape. It opens with the practical question and a table of what it covers, gives each topic as an idea, a result, a worked example and the pitfalls, and closes with a cheat sheet. The derivations, the exercises and the scripts that check every number are in the full edition, linked at the end of each chapter.

## What you will be able to do

By the end of this part you will be able to put an error bar on a noisy answer, a measured rate or the maximum of many noise draws, and choose the inequality that gives the narrowest valid one. You will be able to compute the KL (Kullback–Leibler), Rényi, total variation and hockey-stick divergences between two normal or two Laplace distributions, and explain why processing data cannot increase any of them. You will know the best test between two distributions, read its trade-off curve, report a detector by its true positive rate at a low false positive rate, and size a test set to measure that rate. You will also be able to say how fast gradient descent converges and how much a trained model moves when one example changes, compute what an observation leaks about a hidden value, and state a security property as a game between an attacker and two worlds.
