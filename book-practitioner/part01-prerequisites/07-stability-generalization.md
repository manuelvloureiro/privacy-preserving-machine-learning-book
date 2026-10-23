# Algorithmic stability and generalization

## What it is {#sec-stability-what}

A model's **generalization gap** is its average loss on new data minus its average loss on the training data. A model generalizes when it does not depend much on any single training record. Replace one training record by a fresh draw from the same source, retrain, and see how much higher the new model's loss is than the old one's on the record you took out. This difference is the record's **memorization score**. Averaged over the $n$ training records and over random training sets, the scores give the expected gap exactly:
$$\mathbb E[\mathrm{gap}] = \frac1n\sum_{i=1}^n\mathbb E[\mathrm{mem}_i].$$ {#eq-stability-identity}
Here $\mathrm{mem}_i$ is the score of the $i$-th record.

To bound the scores, call a learning algorithm **$\beta$-uniformly stable** when replacing any one training record moves the loss at any test point by at most $\beta$. Every score is then at most $\beta$, so the expected gap is too:
$$\big|\mathbb E[\mathrm{gap}]\big| \le \beta.$$ {#eq-stability-expected}
Regularized training and stochastic gradient descent (SGD) stopped early are stable in this sense, while a model that stores its training set is not.

## When you meet it {#sec-stability-when}

You meet stability when you choose a regularization strength and a number of epochs, and when a training error of 0 says nothing about the test error. Memorization scores tell you which records a model has learned by heart.

## How to use it {#sec-stability-use}

For regularized **empirical risk minimization (ERM)**, which minimizes the training loss plus a ridge penalty of strength $\lambda$, the stability is
$$\beta = \frac{2L^2}{\lambda n}.$$ {#eq-stability-erm}
It needs a convex loss whose gradients have length at most $L$ ([Optimization](06-optimization.md)). Normalizing the features to length at most 1 gives $L = 1$ for the logistic loss. So with 1,000 records and $\lambda = 0.1$, logistic regression has $\beta = 0.02$. In scikit-learn, `LogisticRegression(C=...)` sets $\lambda$ to the reciprocal of $n$ times `C`, so a large `C` means a large $\beta$. Its intercept is not penalized, so strictly the bound covers only `fit_intercept=False`.

On the full edition's simulated data, the measured gap falls like $1/n$, as the bound predicts. But it stays between about 3% and 13% of the bound, depending on $\lambda$. @eq-stability-identity is an average over training sets, so average the measured gap over several random splits. Use the bound to see which direction helps, and choose $\lambda$ itself on held-out data, since a strong ridge also pulls the weights away from the best fit.

SGD on a convex and smooth loss is also stable if it stops early. Hardt, Recht and Singer (2016) showed that with a constant step $\eta$, its stability after $E$ epochs is at most $2L^2\eta E$. So training longer trades a better fit for less stability, which is what early stopping balances. In the full edition's unregularized example, the gap grows from 0.036 after one epoch to 0.121 after 40. The bound gets the direction right, but after 40 epochs it is 165 times the measured gap.

To measure a memorization score, retrain with the record removed, or replaced by a held-out point in place of a fresh draw, and compare its loss. That costs one retraining per record. So for a large model, score a sample, or train many models on random subsets and compare each record's loss when it is in and out. The scores are uneven, because a typical record is fitted anyway through similar records, while a mislabelled or rare one is fitted only when it is there itself. In the full edition's example at $\lambda = 0.01$, records with the less likely label for their features score three times higher on average than the others.

## What goes wrong {#sec-stability-wrong}

A perfect training score is a warning, not a result. The 1-nearest-neighbour rule, `KNeighborsClassifier(n_neighbors=1)`, has training error 0, and under the 0–1 loss its $\beta$ is 1, so the bound is empty. With noisy labels its gap never closes, and in the full edition's example its test error stays above 0.338.

A small average gap does not mean that no record was memorized, because the average leaves room for a few large scores. So look at the largest scores, not only the mean. Some memorization of rare records is needed for accuracy on data with a long tail of rare subpopulations (Feldman, 2020), so a large score is not always a defect.

A bound that holds with 95% confidence for one run is looser still. For logistic regression on 10,000 records with $\lambda = 0.1$, the Bousquet–Elisseeff bound adds 0.614 to the training loss. That is close to the 0.69 loss of a coin flip.

Finally, a held-out set is honest only for questions fixed before you look at it. Once you look at a result, change something and ask again, you have trained on the holdout. Freedman's paradox shows how large the effect is. On 1,000 records whose labels are independent of every feature, each feature's agreement with the label is a question fixed in advance, and their errors stay below the bound, 0.0616. But a majority vote of the features that agree most reports an accuracy of 0.8053 on the same data, against 0.5018 when the features are chosen on a separate half. So split with `train_test_split` before you start, tune with `cross_val_score` on the training part, and use the test set once.

**Derivations.** Every result here is derived step by step in [the full edition's chapter](../../book-full/part01-prerequisites/07-stability-generalization.md).
