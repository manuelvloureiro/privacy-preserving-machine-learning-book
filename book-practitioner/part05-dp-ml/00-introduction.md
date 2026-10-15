# Differential privacy in machine learning

The previous parts built the guarantee and the tools to account for it. This part trains models with it, and asks the questions that decide whether a private model is worth deploying. **How do you train with differential privacy (DP), and how much accuracy does it cost? Which settings move that cost, and how do you check that the trained model is as private as the accountant claims?**

Most of the answers are about one algorithm. It is stochastic gradient descent (SGD) with each example's gradient clipped to a fixed length and Gaussian noise added, called DP-SGD. The rest of the part is about everything around it: how to choose its hyperparameters, what replaces it when its assumptions fail, how public data and the unit of privacy change it, who pays for its noise, which ε to aim for, and how to audit the result.

You need the earlier parts. In particular, this part uses gradient descent from [Optimization](../part01-prerequisites/06-optimization.md) and the membership attacks of [Part 2](../part02-attacks/00-introduction.md). It also uses the Gaussian mechanism and clipping of [Part 3](../part03-dp-foundations/00-introduction.md), and the DP-SGD accountants of [Part 4](../part04-privacy-accounting/00-introduction.md). Each chapter restates in a sentence what it uses and links back.

## The common thread

A private training algorithm has two halves. The **privacy half** says which quantity is limited in how much one record can change it, and how much noise covers that change. It is a theorem, and the accountants of the previous part turn it into a number. The **utility half** says how much accuracy the noise and the limits cost. It is mostly measured, and rarely proved. Every chapter of this part moves one of the two halves.

This split also explains why private training is hard to judge. A privacy claim can be checked against a proof, but an accuracy claim depends on the data, the model and the settings. So the part keeps asking two separate questions about every method: is the ε correct, and is the accuracy at that ε good enough?

## The running examples

Two models recur. The first is regularized logistic regression on synthetic data in 10 dimensions. Its exact best fit can be computed, so every private method can be compared with the best possible non-private answer. The second is a small neural network trained with DP-SGD on synthetic data. It is small enough to train on a laptop in seconds, but it is not convex, so clipping and noise behave as they do in deep learning. The full-scale experiment, DP-SGD on an image benchmark audited with a membership attack, needs a GPU, and the chapters describe how to run it.

## Why the chapters come in this order

The first two chapters build the algorithm, and every later chapter uses it. [Private empirical risk minimization](01-dp-erm.md) compares three places to add noise to training, and shows why only noise on the gradients works for models that are not convex. [DP-SGD](02-dp-sgd.md) adds clipping, random batches and accounting.

The next three chapters ask what else spends the privacy budget, or could spend it better. [Hyperparameter tuning and private selection](03-hyperparameter-tuning.md) counts the cost of trying many settings, and is best read first. Training methods built on follow the regularized leader (FTRL) do not rely on random batches, and [DP-FTRL and matrix factorization](04-matrix-mechanisms.md) shows when they do better. By contrast, [Private prediction](05-private-prediction.md) protects the answers a model gives instead of the model itself.

The next three chapters change the setting, and can be read in any order. [Public data and private fine-tuning](06-public-data.md) starts from a model trained on public data, so that private training only has to adjust it. [User-level differential privacy](07-user-level.md) widens the protection to everything one person contributes, not just one example. Finally, [Disparate impact](08-disparate-impact.md) asks who pays for the noise, and finds that small groups pay most.

The last two chapters close the loop between theory and practice. [Choosing ε](09-choosing-epsilon.md) uses the vocabulary of all the earlier chapters to pick a target and justify it. [Auditing private training](10-auditing.md) comes last, because it tests empirically what the others claim.

## How to read a chapter

Each chapter starts from the practical question it answers, reasons its way to each result, and shows it at work on the running examples. Every formula you need to set up or check a training run is kept and explained in words. Each comes with the conditions it needs and what goes wrong without them. A cheat sheet at the end collects the results.

The chapters contain no proofs. Each one ends with a link to the matching chapter of the full edition. That chapter derives every result, and its script implements every model in NumPy, gradients included, so you can see exactly what each algorithm computes.

## What you will be able to do

After this part, you will be able to train a convex model privately in three ways and predict how much accuracy each costs. You will be able to implement DP-SGD with clipping, random batches and an accountant, and explain what each of its settings trades. You will know how to account for a hyperparameter search, and when a method without random batches beats DP-SGD.

You will be able to answer queries privately with an ensemble of models, and decide how to fine-tune a public model on private data. You will be able to protect each user rather than each example, and measure who loses most accuracy under private training.

Finally, you will be able to choose an ε and justify it to someone who will hold you to it. You will also be able to audit a private training run with planted records, and turn the attack's success into an empirical lower bound on ε.
