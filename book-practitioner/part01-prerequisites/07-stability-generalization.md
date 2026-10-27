# Algorithmic stability and generalization

## What it is {#sec-stability-what}

A model's **generalization gap** is its average loss on new data minus its average loss on the training data. A model generalizes when it does not depend much on any single training record. To measure that dependence, replace the $i$-th training record by a fresh draw from the same source and retrain. How much higher the new model's loss is than the old one's, on the record taken out, is its **memorization score** $\mathrm{mem}_i$. Averaged over the $n$ training records and over random training sets, the scores give the expected gap exactly:
$$\mathbb E[\mathrm{gap}] = \frac1n\sum_{i=1}^n\mathbb E[\mathrm{mem}_i].$$ {#eq-stability-identity}

To bound the scores, call a learning algorithm **$\beta$-uniformly stable** when replacing any one training record moves the loss at any test point by at most $\beta$. Every score is then at most $\beta$, so the expected gap is too:
$$\big|\mathbb E[\mathrm{gap}]\big| \le \beta.$$ {#eq-stability-expected}
Regularized training and stochastic gradient descent (SGD) stopped early are stable in this sense, while a model that stores its training set is not.

## When you meet it {#sec-stability-when}

You meet stability when you choose a regularization strength and a number of epochs, and when a training error of 0 says nothing about the test error. Memorization scores tell you which records a model has learned by heart.

## How to use it {#sec-stability-use}

For regularized **empirical risk minimization (ERM)**, which minimizes the training loss plus a ridge penalty of strength $\lambda$, the stability is
$$\beta = \frac{2L^2}{\lambda n}.$$ {#eq-stability-erm}
It needs a convex loss whose gradients have length at most $L$ ([Optimization](06-optimization.md)), and features normalized to length at most 1 give $L = 1$ for the logistic loss. So with 1,000 records and $\lambda = 0.1$, logistic regression has $\beta = 0.02$. Scikit-learn's `LogisticRegression(C=...)` sets $\lambda$ to one over $n$ times `C`, so a large `C` means a large $\beta$. Its intercept is not penalized, so strictly the bound covers only `fit_intercept=False`.

The bound gets the trend right but not the size. On the full edition's simulated data, the measured gap falls like $1/n$, as predicted, but is only about 3% to 13% of the bound. Because @eq-stability-identity is an average over training sets, average the measured gap over several random splits. Use the bound to see which direction helps, and choose $\lambda$ itself on held-out data, since a strong ridge also pulls the weights away from the best fit.

SGD on a convex and smooth loss is also stable if it stops early. Hardt, Recht and Singer (2016) showed that with a constant step $\eta$, its stability after $E$ epochs is at most $2L^2\eta E$. This grows with every epoch, so training longer trades a better fit for less stability, which is what early stopping balances. In the full edition's unregularized example, the gap does grow with the epochs, but after 40 of them the bound is 165 times the measured gap of 0.121.

To measure a memorization score, retrain without the record, or with a held-out point in its place, and compare its loss. That costs one retraining per record. So for a large model, score a sample, or train many models on random subsets and compare each record's loss when it is in and out. Expect uneven scores. A typical record is fitted anyway through similar records, while a mislabelled or rare one is fitted only when it is there itself.

## What goes wrong {#sec-stability-wrong}

A perfect training score is a warning, not a result. The 1-nearest-neighbour rule, `KNeighborsClassifier(n_neighbors=1)`, has training error 0, and under the 0–1 loss its $\beta$ is 1, so the bound says nothing. With noisy labels its gap never closes.

A small average gap does not mean that no record was memorized, because the average leaves room for a few large scores. So look at the largest scores, not only the mean. Some memorization of rare records is needed for accuracy on data with a long tail of rare subpopulations (Feldman, 2020), so a large score is not always a defect.

A bound for a single run, rather than for the average, is looser still. The Bousquet–Elisseeff bound holds with 95% confidence for one run. For logistic regression on 10,000 records with $\lambda = 0.1$, it adds 0.614 to the training loss, close to the 0.69 loss of a coin flip.

Finally, a held-out set is honest only for questions fixed before you look at it. Once you look at a result, change something and ask again, you have trained on the holdout. Freedman's paradox shows the size of the effect on a holdout of 1,000 records with 1,000 random features, all independent of the label. Each feature's agreement rate with the label is a question fixed in advance, and each strays from one half by less than the bound for such questions, 0.0616. But a majority vote of the features that agree most reports an accuracy of 0.8053 on the same data, against 0.5018 when they are chosen on a separate half. So split with `train_test_split` before you start, tune with `cross_val_score` on the training part, and use the test set once.

**Derivations.** Every result here is derived step by step in [the full edition's chapter](../../book-full/part01-prerequisites/07-stability-generalization.md).
