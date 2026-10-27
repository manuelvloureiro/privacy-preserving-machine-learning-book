# Algorithmic stability and generalization

## What it is {#sec-stability-what}

A model's **generalization gap** is its average loss on new data minus its average loss on the training data. A model generalizes when it does not depend much on one training record. To measure that dependence, replace the $i$-th training record by a new draw from the same source, and train again. On the record taken out, the new model's loss minus the old model's loss is the **memorization score** $\mathrm{mem}_i$. The average of the scores across the $n$ training records and across random training sets is exactly the expected gap:
$$\mathbb E[\mathrm{gap}] = \frac1n\sum_{i=1}^n\mathbb E[\mathrm{mem}_i].$$ {#eq-stability-identity}

To bound the scores, call a learning algorithm **$\beta$-uniformly stable** when a change of one training record moves the loss at each test point by at most $\beta$. Then every score is at most $\beta$, so the expected gap is too:
$$\big|\mathbb E[\mathrm{gap}]\big| \le \beta.$$ {#eq-stability-expected}
Regularized training and stochastic gradient descent (SGD) with early stopping are stable in this sense. A model that stores its training set is not.

## When you meet it {#sec-stability-when}

You meet stability when you choose a regularization strength and a number of epochs, and when a training error of 0 says nothing about the test error. Memorization scores show which records a model has learned by heart.

## How to use it {#sec-stability-use}

For regularized **empirical risk minimization (ERM)**, which minimizes the training loss plus a ridge penalty of strength $\lambda$, the stability is
$$\beta = \frac{2L^2}{\lambda n}.$$ {#eq-stability-erm}
It needs a convex loss whose gradients have length at most $L$ ([Optimization](06-optimization.md)). For the logistic loss, features of length at most 1 give $L = 1$. So with 1,000 records and $\lambda = 0.1$, logistic regression has $\beta = 0.02$. In scikit-learn's `LogisticRegression`, $\lambda$ is one over $n$ times `C`, so a large `C` gives a large $\beta$. The intercept has no penalty, so the bound is strictly true only with `fit_intercept=False`.

The bound gets the trend right but not the size. On the full edition's simulated data, the measured gap falls like $1/n$, as the bound predicts, but it is only about 3% to 13% of the bound. @eq-stability-identity is an average across training sets, so average the measured gap across several random splits. Use the bound to see which direction helps. Choose $\lambda$ itself on held-out data, because a strong ridge also pulls the weights away from the best fit.

SGD on a convex and smooth loss is also stable if it stops early. Hardt, Recht and Singer (2016) proved that with a constant step $\eta$, its stability after $E$ epochs is at most $2L^2\eta E$. This bound grows with every epoch, so a longer run gives a better fit but less stability, and early stopping balances the two. In the full edition's example without a ridge, the gap does grow with the epochs. But after 40 epochs, the bound is 165 times the measured gap of 0.121.

To measure a memorization score, train again with a held-out point in place of the record, and compare its loss. That costs one training run per record. So for a large model, score a sample, or train many models on random subsets and compare each record's loss when it is in and out. Expect uneven scores, because similar records fit a typical record anyway, while a mislabelled or rare record fits only when it is in the training set.

## What goes wrong {#sec-stability-wrong}

A perfect training score is a warning, not a result. The 1-nearest-neighbour rule, `KNeighborsClassifier(n_neighbors=1)`, has training error 0. Under the 0–1 loss its $\beta$ is 1, so the bound says nothing, and with noisy labels its gap never closes.

A small average gap does not mean that the model memorized no record, because a few large scores can hide in the average. So look at the largest scores, not only the mean. But a large score is not always a defect. On data with a long tail of rare subpopulations, a model must memorize some rare records to be accurate (Feldman, 2020).

A bound for one run, such as the Bousquet–Elisseeff bound at 95% confidence, is even looser. For logistic regression on 10,000 records with $\lambda = 0.1$, it adds 0.614 to the training loss, close to the coin-flip loss of 0.69.

Last, a held-out set is honest only for questions fixed before you look at it. If you look at a result, change something and ask again, you have trained on the holdout. Freedman's paradox shows the size of the effect on a holdout of 1,000 records with 1,000 random features, all independent of the label. Each feature's rate of agreement with the label is a question fixed in advance, and each is within 0.0616 of one half, the bound for such questions. But a majority vote of the features that agree most shows an accuracy of 0.8053 on the same data, against 0.5018 when you choose them on a separate half.

So split with `train_test_split` before you start, tune with `cross_val_score` on the training part, and use the test set once.

**Derivations.** Every result here is derived step by step in [the full edition's chapter](../../book-full/part01-prerequisites/07-stability-generalization.md).
