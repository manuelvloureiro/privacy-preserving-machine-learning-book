# Differential privacy in machine learning

The previous parts built the guarantee and the tools to account for it. This part trains models with it. It asks the questions that decide whether a private model is worth deploying: **how do you train with differential privacy (DP), how much accuracy does it cost, which knobs move that cost, and how do you check that the trained model is as private as the accountant claims?** The answers are mostly about one algorithm, stochastic gradient descent (SGD) with per-example clipping and Gaussian noise, called DP-SGD, and about everything around it: how its hyperparameters are chosen, what replaces it when its assumptions fail, how public data and the choice of privacy unit change it, who pays for its noise, which $\varepsilon$ to target, and how to audit the result. The last chapter turns the guarantee to a different use: the same noise that protects people also keeps a reused validation or test set honest.

## Who this part is for

You are expected to have read the earlier parts. From [Optimization](../part01-prerequisites/06-optimization.md) you need SGD, strong convexity and the sensitivity of regularized empirical risk minimization (ERM). From [Membership inference](../part02-attacks/02-membership-inference.md) you need the likelihood-ratio attack (LiRA) and the true and false positive rates of an attack. From the DP foundations you need the Gaussian mechanism ([The Laplace and Gaussian mechanisms](../part03-dp-foundations/03-laplace-gaussian.md)), clipping ([Beyond global sensitivity](../part03-dp-foundations/07-sensitivity.md)), the exponential mechanism and report-noisy-max ([Selection](../part03-dp-foundations/05-selection.md)) and the lower bounds of [Lower bounds](../part03-dp-foundations/10-lower-bounds.md). From the accounting part you need Gaussian differential privacy (GDP) and trade-off functions ([f-DP and Gaussian DP](../part04-privacy-accounting/04-gaussian-dp.md)), amplification by subsampling and the Rényi differential privacy (RDP) accountant for DP-SGD ([Amplification by subsampling](../part04-privacy-accounting/05-subsampling.md)), and what to report ([Choosing an accountant](../part04-privacy-accounting/09-choosing-an-accountant.md)). The last chapter also uses the replace-one argument of [Algorithmic stability and generalization](../part01-prerequisites/07-stability-generalization.md) and the sparse vector technique of [Selection](../part03-dp-foundations/05-selection.md). Each chapter restates the facts it uses and links to them.

## The common thread

A private training algorithm has two halves. The **privacy half** says which quantity has bounded sensitivity and how much noise covers it; it is a theorem, and the accountants of the previous part turn it into a number. The **utility half** says how much the noise and the bounding cost; it is mostly measured, rarely proved. Every chapter of this part moves one of the two.

The first two chapters build the algorithm: [Private ERM](01-dp-erm.md) compares three places to inject noise into training and shows why, in practice, only noise on the gradients survives non-convex models, and [DP-SGD](02-dp-sgd.md) adds clipping, minibatches and accounting. The next three ask what else spends the budget or could spend it better: hyperparameter search; mechanisms that do not rely on random batches, such as the follow the regularized leader (FTRL) algorithm run on noisy gradient sums, DP-FTRL; and private prediction instead of private training, such as private aggregation of teacher ensembles (PATE). The next three change the setting: public data, a different privacy unit, and the effect on under-represented groups. The next two close the loop between theory and practice: which $\varepsilon$ to choose, and how to test a claim empirically. The last chapter uses the privacy half for a different purpose. A mechanism that reveals little about any one record cannot be overfitted, so DP also lets one holdout set answer many adaptively chosen questions, the situation of every hyperparameter search and leaderboard. @tbl-dpml-map lists them.

| Chapter | The question it answers | Builds on |
| --- | --- | --- |
| [Private empirical risk minimization](01-dp-erm.md) | Where can noise enter training, what does each choice cost, and what rate can a convex model reach? | 6, 7, 20, 24, 27, 32 |
| [DP-SGD](02-dp-sgd.md) | How do clipping, noise and sampling make deep-learning training private, and what does each knob do? | 6, 20, 24, 33, 37, 38 |
| [Hyperparameter tuning and private selection](03-hyperparameter-tuning.md) | How much privacy does tuning leak, and how do you pay for it honestly? | 22, 29, 39 |
| [DP-FTRL and matrix factorization](04-matrix-mechanisms.md) | How do you train privately without trusting random batches? | 20, 32, 39 |
| [Private prediction: PATE and in-context learning](05-private-prediction.md) | When is it better to privatize answers than weights? | 22, 24, 29, 39 |
| [Public data and private fine-tuning](06-public-data.md) | How much does a public pretrained model buy, and why does fine-tuning fewer parameters help? | 39, 40 |
| [User-level differential privacy](07-user-level.md) | What changes when one person contributes many records? | 19, 39 |
| [Disparate impact](08-disparate-impact.md) | Who pays for the noise and the clipping? | 39 |
| [Choosing ε](09-choosing-epsilon.md) | Which $\varepsilon$ should you target, and how do you justify it? | 18, 32, 37, 39 |
| [Auditing private training](10-auditing.md) | Is the model as private as the accountant says? | 10, 32, 39, 46 |
| [Adaptive data analysis](11-adaptive-data-analysis.md) | How many adaptively chosen questions can one holdout set answer, and how do you reuse it safely? | 7, 19, 20, 22, 23, 31, 40 |

: The chapters of this part, the question each answers, and the earlier chapters it uses. {#tbl-dpml-map}

## The running examples

Two models recur. The first is **regularized logistic regression** in $d = 10$ dimensions on synthetic data, where the exact minimizer can be computed and every method can be compared with the non-private optimum; [Private ERM](01-dp-erm.md) introduces it. The second is a **small neural network** trained with DP-SGD on synthetic data, small enough that every experiment of the part runs on a laptop in seconds, yet non-convex, so that clipping and noise behave as they do in deep learning; [DP-SGD](02-dp-sgd.md) introduces it. The full-scale experiment, DP-SGD on CIFAR-10 audited with LiRA, needs a GPU and PyTorch, which are not dependencies of the repository; the chapters scale it down and say how.

## How to read this part

**Order.** Read chapters 38 and 39 first, in order: every later chapter uses DP-SGD. Chapters 40 and 41 change how the algorithm is run, and can be read in either order. Chapters 42 to 45 are independent of each other. Chapter 46 uses the vocabulary of all the earlier ones, and chapter 47 should come after the others, because it checks what the others claim. Chapter 48 can be read at any point after chapter 40, whose hyperparameter search is its motivating example. It needs DP-SGD only for that example.

**Structure.** The chapters follow the shape of the earlier parts: why the topic matters, a toolkit, one section per result with a derivation, interpretation, worked example and two check-yourself questions, then code, references and a cheat sheet. Exercises with worked solutions are in the exercise book, a companion website, with a link at the end of each chapter.

**Run the code.** Each chapter has a script in `code/` that prints every number its text quotes, except values attributed to a paper, and regenerates its figures, for example

```bash
uv run python code/part05-dp-ml/dp_erm.py
```

Every model is written in NumPy, gradients included, so you can see exactly what each algorithm computes.

## Notation

The part uses the notation of the earlier parts, extended in @tbl-dpml-notation.

| Symbol | Meaning |
| --- | --- |
| $D = (z_1, \dots, z_n)$, $z_i = (x_i, y_i)$ | A training set of $n$ records with features $x_i$ and labels $y_i$; a neighbouring dataset $D'$ adds or removes one record, $D \sim D'$ |
| $\theta \in \mathbb R^d$, $\ell(\theta; z)$, $\mathcal L(\theta; D)$ | The model parameters; the loss on one record; the empirical risk, the training objective (with its regularizer, when there is one) |
| $\theta_t$, $\theta^*$, $\tilde\theta$ | The iterate at step $t$; the non-private minimizer; a private release |
| $g_i = \nabla_\theta\ell(\theta; z_i)$, $\bar g_i$ | The gradient of record $i$'s loss, a per-example gradient; the clipped one |
| $C$ | The clipping norm: per-example gradients are rescaled to length at most $C$ |
| $\sigma$ | The noise multiplier: the noise standard deviation in units of $C$ |
| $q$, $B = qn$, $T$ | The sampling rate; the expected batch size; the number of steps |
| $\eta$ | The learning rate |
| $\mu(\varepsilon, \delta)$ | The largest ratio of sensitivity to noise standard deviation for which a Gaussian mechanism is $(\varepsilon, \delta)$-DP |

: Notation used throughout this part, in addition to that of the earlier parts. {#tbl-dpml-notation}

## What you will be able to do

By the end of this part you should be able to:

- train a convex model privately by output, objective or gradient perturbation, and predict its excess risk;
- implement DP-SGD from scratch, with per-example clipping, Poisson sampling and an RDP accountant, and explain what each hyperparameter trades;
- account for hyperparameter search, and run a private selection procedure;
- explain when DP-FTRL or a matrix-factorization mechanism beats DP-SGD, and why;
- train a teacher ensemble and answer queries with PATE or a private in-context learner;
- decide whether to fine-tune all parameters or a few, with a public pretrained model;
- account for user-level privacy, and bound each user's contribution;
- measure and explain the disparate impact of private training on small groups;
- choose an $\varepsilon$ and justify it to someone who will hold you to it;
- audit a private training run with canaries and turn the attack's success into an empirical lower bound on $\varepsilon$;
- explain why a reused holdout set overfits, size a holdout set for adaptively chosen questions, and reuse one safely with Gaussian noise or Thresholdout.
