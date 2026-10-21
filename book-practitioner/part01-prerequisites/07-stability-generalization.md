# Algorithmic stability and generalization

## Why this matters

What matters is a model's loss on new data, not on its training data. The difference between the two is the **generalization gap**, and you measure it on data the model has not seen. This chapter tells you in advance which knobs shrink it, which examples carry it, and when a held-out measurement stops being honest.

The one idea is that **a model generalizes when it does not depend much on any single training example.** For regularized training and for stochastic gradient descent (SGD), that dependence shrinks with a stronger regularizer and a shorter run. It can also be measured example by example, and Part 2 shows why it matters beyond accuracy.

## Training loss, test loss and the gap {#sec-stability-risk}

Take a training set $S$ of examples drawn independently from one distribution $\mathcal D$. Write $n$ for the number of examples, and $A(S)$ for the model that a learning algorithm $A$ returns. A loss $\ell(w, z)$ that is never negative measures how badly a model $w$ does on an example $z$. The **test loss** $R(w)$ is the average loss on a new example from $\mathcal D$, and the **training loss** $\hat R_S(w)$ is the average over the training set. The gap of the trained model is
$$\mathrm{gap} = R(A(S)) - \hat R_S(A(S)).$$

For a model fitted to the data, the training loss is biased downward, because the algorithm chose the model that does well on these particular examples. So estimate the test loss on a held-out set that played no part in training and comes from the same distribution as future data.

The chapter's running example is logistic regression on features in 10 dimensions with noisy labels, plus a ridge penalty whose strength is written $\lambda$.

## The gap and uniform stability {#sec-stability-gap}

The tool here is the **replace-one neighbour** $S^{(i)}$, the training set with example $i$ replaced by a fresh draw from $\mathcal D$. The expected gap is then exactly the average damage that training without an example does to that example's loss:
$$\mathbb E[\mathrm{gap}] = \frac1n\sum_{i=1}^n\mathbb E\Big[\ell\big(A(S^{(i)}), z_i\big) - \ell\big(A(S), z_i\big)\Big].$$ {#eq-stability-identity}
It holds for any algorithm, because a training example and a fresh draw are interchangeable. In other words, overfitting is dependence on individual examples.

To bound that dependence, call an algorithm **$\beta$-uniformly stable** when replacing any one training example by any other moves the loss by at most $\beta$. That must hold at every test point $z$:
$$\big|\ell(A(S), z) - \ell(A(S^{(i)}), z)\big| \le \beta.$$
Every term of the identity then lies within $\beta$ of zero, so the expected gap does too:
$$\big|\mathbb E[\mathrm{gap}]\big| \le \beta.$$ {#eq-stability-expected}
It works for large models too, but $\beta$ is a worst case over training sets, replacements and test points, so it can sit far above the true gap.

## High probability: the Bousquet–Elisseeff bound {#sec-stability-high-probability}

You train once, so you may want a high-probability bound. The bound of Bousquet & Elisseeff gives one for losses bounded by some maximum, but only when $\beta$ falls faster than $1/\sqrt n$, and it is loose at realistic sizes. At 95% confidence, for regularized logistic regression with 10,000 examples at $\lambda = 0.1$, it adds 0.614 to the training loss, which says nothing useful. So measure the gap on held-out data, and use the bounds only to compare settings.

## Regularized ERM is stable {#sec-stability-erm}

Regularized **empirical risk minimization (ERM)** covers ridge regression and regularized logistic regression. It picks the weights that minimize the training loss plus a ridge penalty,
$$w_S = \arg\min_w \Big(\hat R_S(w) + \frac\lambda2\|w\|^2\Big).$$
Suppose each loss is convex and $L$-Lipschitz, which means that its gradient never has length more than $L$ ([Optimization](06-optimization.md)). Then replacing one example moves the weights by at most $2L/(\lambda n)$, and the algorithm is stable with
$$\beta = \frac{2L^2}{\lambda n}.$$ {#eq-stability-erm}

To compute $\beta$, normalize the features to length at most 1, which makes $L = 1$ for the logistic and hinge losses, and plug in $\lambda$ and $n$. Logistic regression with $\lambda = 0.1$ and 1,000 examples then has $\beta = 0.02$. In scikit-learn, `LogisticRegression(C=...)` sets $\lambda = 1/(Cn)$, so a large `C` means a weak penalty and a large $\beta$.

On the full edition's running example, the measured gap falls like $1/n$, as the bound predicts, but it stays between about 3% and 13% of the bound. So the bound tells you how the gap scales with $\lambda$ and $n$, not how large it is. Doubling $\lambda$ halves $\beta$, but a larger penalty also pulls the weights away from the best fit. The test loss is therefore U-shaped in $\lambda$, so choose $\lambda$ on held-out data and use the bound to know which direction helps.

Both conditions matter. Squared loss is Lipschitz only on a bounded region, so ridge regression needs bounded features and labels. Without convexity, one flipped label can move the minimizer from one valley to another, however large $n$ is.

## SGD is stable if you stop early {#sec-stability-sgd}

Large models are trained by SGD rather than to an exact minimizer. Write $\eta_t$ for the step size at step $t$, and $T$ for the number of steps. Hardt, Recht & Singer (2016) showed that
$$\beta \le \frac{2L^2}{n}\sum_{t=1}^T\eta_t.$$ {#eq-stability-hrs}
With a constant step $\eta$ over a number of epochs $E$, the bound becomes $2L^2\eta E$. So for a fixed number of epochs, more data does not improve it.

The bound needs a convex, $L$-Lipschitz and smooth loss, with steps small enough not to overshoot. It also needs examples picked uniformly at random, so shuffling once per epoch, as most implementations do, is not covered as stated.

The practical reading is that training time is a stability knob, which makes early stopping a trade-off rather than a trick. Take the running example without regularization, with 100 examples and a step of 0.25. One epoch then gives $\beta \le 0.5$. The gap grows from 0.036 after one epoch to 0.121 after 40. The test loss is lowest after about 5 epochs and rises afterwards, while the training loss keeps falling. The bound gets the direction right but the size wrong, by a factor of 165 after 40 epochs. For non-convex losses the bound grows polynomially in the number of steps, so treat it as a direction, not a number.

## An unstable algorithm: 1-nearest neighbour {#sec-stability-nn}

The nearest neighbour (NN) rule with one neighbour, 1-NN, predicts the label of the closest training point. Like a lookup table that stores the training set, it has training error 0 and, under the 0-1 loss, stability $\beta = 1$. That is because replacing a point by the same point with the opposite label flips the prediction there. So with noisy labels its gap never closes, and in the running example its test error never falls below 0.338. In scikit-learn, `KNeighborsClassifier(n_neighbors=1)` is this rule, and its perfect training score is a warning, not a result.

## Overfitting is memorization of individual examples {#sec-stability-memorization}

Each term of @eq-stability-identity measures how much one example depends on its own presence, and it is called the example's **memorization score**, or **self-influence**,
$$\mathrm{mem}_i(S) = \ell\big(A(S^{(i)}), z_i\big) - \ell\big(A(S), z_i\big).$$
Their average is the expected gap, so the scores show which examples carry it.

To measure a score, train on $S$ and record the example's loss. Then replace the example by a fresh draw, retrain with the same settings, and record its loss again. That costs one retraining per example, which is affordable only for small models. For a large model, score a sample of examples, or train many models on random subsets and compare each example's loss with and without it.

The scores are uneven in a predictable way. A typical example is fitted well whether or not it is present, because its neighbours carry the same information. An atypical one, such as a mislabelled point or a member of a rare subgroup, is fitted only if it is there itself. In the running example, the examples whose label goes against the trend have three times the average score of the others. Regularization flattens that profile. A small average gap therefore leaves room for a few heavily memorized examples, so a small train-test gap does not mean that no example was memorized. Look at the largest scores, not only the mean, and see Part 3 for a stronger, worst-case guarantee. Feldman (2020) shows that on data with a long tail of rare subgroups some memorization is needed for accuracy, so a large score is not always a defect.

## Choosing what to measure after looking {#sec-stability-adaptive}

A held-out set is honest only for questions fixed before you look at it. Call a **question** a rule that gives each example a score between 0 and 1, such as whether a model classifies it correctly. Questions fixed before the holdout set is drawn are safe. For some number $k$ of them on $n$ examples, their largest error is on average at most $\sqrt{\ln(2k)/(2n)}$. But you usually look at a verdict, change something, and ask again. Each new question then depends on the holdout set, so you have become a learning algorithm trained on it, and the bound no longer holds.

Freedman's paradox shows the size of the effect. A holdout set has 1,000 examples, each with many random features and a label independent of them all, so no rule beats 50% on new data. An analyst keeps the features that agree with the label noticeably more or less than half the time, and measures their majority vote on the same holdout set. With 1,000 features, the questions fixed in advance stay within the bound of 0.0616. But the vote, chosen after looking, reports an accuracy of 0.8053.

The fix is **sample splitting**, which means never letting an example answer a question it helped to choose. Choosing the features on one half and measuring the vote on the other brings the accuracy back to 0.5018. You need a fresh part of the data for every round of choices, not for every question. So split the data with scikit-learn's `train_test_split` before you start, and tune with `cross_val_score` on the training part. Then use the test set once, for the final verdict.

## Code

This calculator gives the stability of regularized ERM and of SGD, and prints the numbers quoted above.

```python
import math


def erm_beta(lipschitz: float, lam: float, n: int) -> float:
    """Uniform stability of ridge-regularized ERM."""
    return 2 * lipschitz**2 / (lam * n)


def sgd_beta(lipschitz: float, step: float, epochs: float) -> float:
    """Stability bound for SGD with a constant step on a convex, smooth, Lipschitz loss."""
    return 2 * lipschitz**2 * step * epochs


print(f"{erm_beta(1, 0.1, 1000):.3f}")  # 0.020
print(sgd_beta(1, 0.25, 1))  # 0.5
print(sgd_beta(1, 0.25, 40))  # 20.0
```

## Cheat sheet

| Result | Conditions | Statement |
| --- | --- | --- |
| Gap identity | Examples drawn independently from one distribution | Expected gap = average memorization score |
| Expected gap | $\beta$-uniformly stable | $\lvert\mathbb E[\mathrm{gap}]\rvert \le \beta$ |
| Regularized ERM | Convex, $L$-Lipschitz loss | $\beta = 2L^2/(\lambda n)$ |
| SGD | Convex, smooth, $L$-Lipschitz, small steps | $\beta \le 2L^2\eta E$ for $E$ epochs |
| Fixed questions | $k$ questions fixed before the holdout set is drawn | Largest error $\le \sqrt{\ln(2k)/(2n)}$ on average |

: The results of this chapter. {#tbl-stability-cheat}

**Rules of thumb**

- Measure the gap on held-out data. The two knobs that shrink it are the regularization strength and the training time. For ERM, $\beta$ is proportional to $1/\lambda$, and for SGD to $\eta E$.
- A training error of 0 is a warning, not a result.
- A small train-test gap does not rule out memorized outliers. The atypical examples are memorized most, so inspect the largest scores.
- A holdout set that helped choose a question is a training set for it. Never tune on the test set, give each round of choices its own data, and use the final test set once.
- Do not report a generalization bound without checking that it is below the loss you measure.

## Further reading

- Bousquet & Elisseeff, "Stability and Generalization" (JMLR 2002): uniform stability and the high-probability bound.
- Hardt, Recht & Singer, "Train Faster, Generalize Better: Stability of Stochastic Gradient Descent" (ICML 2016): SGD stability and early stopping.

**Derivations.** Every result here is derived step by step in [the full edition's chapter](../../book-full/part01-prerequisites/07-stability-generalization.md).
