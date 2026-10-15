# Prerequisites

This part collects the mathematics that the rest of the book stands on. It contains no privacy yet. Everything here is standard probability, statistics, optimization, information theory and cryptography. What makes it different from a textbook is the selection. Each topic is here because a later part needs it, and each is taken only as far as that need goes.

You need the basics of probability, such as expectation, variance and independence. You need enough machine learning to recognise a loss function and stochastic gradient descent (SGD), and enough Python to read a short script. If you already know a topic, skim its cheat sheet and move on. Each later chapter restates in a sentence what it uses from here, and links back.

## The common thread

The later parts study algorithms whose output is random, and they keep asking the same few questions about them. How far can a random output be from its expected value? How different are the outputs on two inputs that differ in one example, and how well could anyone tell them apart? How much does a trained model depend on a single training example? How much does an observation reveal about something hidden?

Two ideas answer most of these questions. The first is the **likelihood ratio**: how many times more likely an observation is under one distribution than under another. If you want to know which of two distributions produced an observation, this ratio is all the observation tells you. So every way of comparing two distributions comes back to it. When the ratio can never be large, no single observation is strong evidence either way, and that is the property privacy will later demand.

The second idea is the **effect of one input**: how far one example can move an average, a trained model or a loss. A quantity that no single input can move much has little room to vary. So a small effect of one input turns into narrow error bars, and in learning, into a model that does as well on new data as on its training data.

## Why the chapters come in this order

The first four chapters build one argument, each on the one before. [Probability distributions](01-distributions.md) introduces the few families that model counts and noise. It ends by comparing a noise distribution with a copy of itself moved sideways. The likelihood ratio of normal noise grows without limit far from the centre, while that of Laplace noise stays below a fixed bound.

That raises the question of how often extreme values occur. [Concentration inequalities](02-concentration.md) answers it. It turns what you know about a random quantity into an error bar that fails only with a small probability you choose. [Divergences](03-divergences.md) then compresses the difference between two distributions into a single number. It shows that there are several such numbers, and that each answers a different question. [Hypothesis testing](04-hypothesis-testing.md) says what those numbers mean for anyone who tries to tell the two distributions apart. It finds the best possible test, draws the curve of its two kinds of error, and explains why a detector should be judged where false alarms are rare.

[Information theory](05-information-theory.md) follows the same thread from another angle. It asks how much an observation reveals about a hidden value, both on average and for a single best guess. [Optimization](06-optimization.md) and [Algorithmic stability and generalization](07-stability-generalization.md) take up the second idea. They show how fast training converges, how far its result moves when one example changes, and why a model that depends little on each example generalizes.

[Cryptography basics](08-crypto-basics.md) stands somewhat apart. It defines what it means that an efficient observer learns nothing it should not, by asking whether an attacker can tell two worlds apart. You can leave it until a later chapter needs it. Threat models, linkage attacks and trust models are the first chapters that do.

## How to read a chapter

Each chapter opens with the practical question it answers and the idea behind the answer. Each section then starts from a problem and reasons its way to a result. Every formula is explained in words first, every condition a result needs is stated with its reason, and a worked example with real numbers shows the result in use. A cheat sheet at the end collects the results on one page.

The chapters contain no proofs. Each one ends with a link to the matching chapter of the full edition, which derives every result step by step and has a script that checks every number quoted.

## What you will be able to do

After this part, you will be able to put an error bar on a noisy answer, a measured error rate, or the largest of many noise draws. You will know which inequality gives the narrowest valid error bar, and why.

You will be able to compute the Kullback–Leibler (KL), Rényi, total variation and hockey-stick divergences between two normal or two Laplace distributions. You will also be able to explain why processing data can never increase any of them.

You will know the best test between two distributions and how to read its trade-off curve. You will be able to report a detector by how often it catches positives at a low false-positive rate. You will also be able to size a test set so that the rate you measure is precise enough.

Finally, you will be able to say how fast gradient descent converges and how far a trained model moves when one example changes. You will be able to compute how much an observation leaks about a hidden value, and state a security property as a game between an attacker and two worlds.
