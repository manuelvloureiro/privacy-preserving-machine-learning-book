# Differential privacy in machine learning

The previous parts built the guarantee and the tools to account for it. This part trains models with it, and asks the questions that decide whether a private model is worth deploying: how do you train with differential privacy (DP), how much accuracy does it cost, which knobs move that cost, and how do you check that the trained model is as private as the accountant claims? The answers are mostly about one algorithm, stochastic gradient descent (SGD) with per-example clipping and Gaussian noise, called DP-SGD, and about everything around it.

Read this part if you train, review or sign off on a model that claims an $\varepsilon$. It assumes the Gaussian mechanism and clipping from [Differential privacy foundations](../part03-dp-foundations/00-introduction.md), and amplification by subsampling and the accountant for DP-SGD from [Privacy accounting](../part04-privacy-accounting/00-introduction.md). Each chapter restates what it uses.

## The common thread

A private training algorithm has two halves. The **privacy half** says which quantity has bounded sensitivity, meaning how far one record can move it, and how much noise covers that move; it is a theorem, and the accountants of Part 4 turn it into a number. The **utility half** says how much the noise and the bounding cost in accuracy; it is mostly measured, rarely proved. Every chapter moves one of the two, and keeping them apart is the most useful habit here: a choice that only affects accuracy can be tuned freely, one that touches the privacy half must be accounted for.

The first two chapters build the algorithm. [Private empirical risk minimization](01-dp-erm.md) compares three places to put the noise, on the final weights, inside the objective and on every gradient, and shows that only noise on the gradients survives non-convex models, because it needs nothing but a bound on each record's gradient. [DP-SGD](02-dp-sgd.md) supplies that bound by clipping, adds random minibatches and an accountant, and works out what each knob does. Every later chapter uses it, so these two come first.

The next three ask what else spends the budget, or could spend it better. [Hyperparameter tuning](03-hyperparameter-tuning.md) shows that picking the best of several private runs is itself a computation on the data, and how to pay for it honestly. Mechanisms built on the follow the regularized leader (FTRL) algorithm, [DP-FTRL and matrix factorization](04-matrix-mechanisms.md), replace the random batches that DP-SGD's amplification relies on with correlated noise, for settings where sampling cannot be trusted or controlled. [Private prediction](05-private-prediction.md) privatizes answers instead of weights, which can be cheaper when you only need a limited number of predictions.

The next three change the setting rather than the algorithm. [Public data and private fine-tuning](06-public-data.md) explains why a public pretrained model helps so much: the noise grows with the number of parameters you train, so fine-tuning fewer of them can be more accurate. [User-level differential privacy](07-user-level.md) changes the unit of protection from one record to one person, which is usually what you meant to promise. [Disparate impact](08-disparate-impact.md) asks who pays for the noise and the clipping, and finds that small groups and hard examples pay most.

The last two close the loop between theory and practice. [Choosing ε](09-choosing-epsilon.md) turns a budget into what it means for an attacker and asks how to justify one. [Auditing private training](10-auditing.md) runs attacks with planted records against a trained model and turns their success into an empirical lower bound on $\varepsilon$, which catches implementations that do not deliver what the accountant says. It comes last because it checks what the others claim.

## The running examples

Two models recur. The first is ridge-regularized logistic regression in ten dimensions on synthetic data, where the exact minimizer can be computed, so every private method can be compared with the non-private optimum. The second is a small neural network trained with DP-SGD on synthetic data, small enough that every experiment runs on a laptop in seconds, yet non-convex, so clipping and noise behave as they do in deep learning. It runs exactly the configuration whose $\varepsilon$ Part 4 computed, so you can see what that number costs in accuracy.

## How to read this part

Read the first two chapters in order. Hyperparameter tuning and DP-FTRL change how the algorithm is run and can be read in either order. The chapters on private prediction, public data, user-level privacy and disparate impact are independent of each other. Read choosing $\varepsilon$ after them, and auditing last. The derivations are in the full edition.

## What you will be able to do

By the end of this part you will be able to train a convex model privately and predict what it costs, and run DP-SGD knowing which knobs set $\varepsilon$ and which only set accuracy. You will be able to check an implementation for the details that silently break its guarantee, account for tuning, decide when correlated noise, private prediction or a public pretrained model is the better route, and state the privacy unit you protect. And you will be able to say who pays for the noise, choose an $\varepsilon$ you can defend, and audit a trained model to test whether it is as private as claimed.
