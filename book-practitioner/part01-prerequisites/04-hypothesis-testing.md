# Hypothesis testing

## Why this matters

A fraud detector must call each transaction normal or fraudulent, so it decides which of two distributions an observation came from. Every detector and every binary classifier is such a hypothesis test. To evaluate one, measure its two kinds of error, false alarms and missed cases, at the low rate of false alarms that you need. Accuracy hides that rate.

## Tests and their two errors {#sec-ht-setup}

The default assumption, the **null hypothesis** $H_0$, is that the observation comes from a distribution $P$, such as the fraud model's scores for normal transactions. The **alternative hypothesis** $H_1$ is that it comes from a distribution $Q$, such as its scores for fraudulent ones. A **test** decides whether to reject the null, which here means to flag the transaction.

A test can make two kinds of mistake. A flag on a normal case is a false positive, and its probability is the **false positive rate (FPR)**, written $\alpha$. A missed real case is a false negative, with probability $\beta$, the **false negative rate (FNR)**. The share of real cases that the test catches, $1 - \beta$, is the **true positive rate (TPR)**. A lower threshold raises the TPR but also the FPR. So first fix a budget for the FPR, the **level**, and then look for the test with the highest TPR, or **power**, within it.

## The Neyman–Pearson lemma {#sec-ht-np}

Write $p(x)$ and $q(x)$ for the probabilities, or densities, of an observation $x$ under the two hypotheses. Their quotient is the **likelihood ratio**
$$\Lambda(x) = \frac{q(x)}{p(x)}.$$
A **likelihood ratio (LR) test** rejects wherever this ratio exceeds a threshold $\tau$, which you set so that the FPR of the test equals the budget. The **Neyman–Pearson lemma** says that no other test at that level catches more real cases.

So the best test uses only the likelihood ratio of an observation. A threshold on a raw score is that test only when a higher score always means a higher ratio. To estimate the ratio from labelled scores, fit a logistic regression of the label on the score. Its log-odds, minus the log prior odds, estimate $\ln\Lambda$. If a known group shifts the scores, the ratio depends on the group, so the best test sets a threshold within each group.

## Trade-off curves {#sec-ht-tradeoff}

The best FNR at every level says how hard $P$ is to tell from $Q$. This curve is the **trade-off function** $T(P, Q)$, which you get when you move the LR threshold across its range. The **receiver operating characteristic (ROC)** curve holds the same information upside down, as TPR against FPR. Each real detector has its own ROC curve, on or below the best one.

## The Gaussian trade-off curve {#sec-ht-gaussian}

A common pair is a score that is normal under the two hypotheses with the same spread, scaled so that normal cases have mean 0 and standard deviation 1. Write $\mu$ for the distance between the two centres in standard deviations, and $\Phi$ for the standard normal cumulative distribution function. The best test has its threshold at $\Phi^{-1}(1 - \alpha)$, and its FNR at level $\alpha$ is the Gaussian trade-off function
$$G_\mu(\alpha) = \Phi\big(\Phi^{-1}(1 - \alpha) - \mu\big).$$ {#eq-ht-gaussian-tradeoff}
In SciPy this is `norm.cdf(norm.isf(alpha) - mu)`. Use `isf`, because `norm.ppf(1 - alpha)` loses the digits that matter at a very small $\alpha$. The single number $\mu$ fixes the best TPR at every FPR. At $\mu = 2$, for example, the best TPR is 0.6388 at an FPR of 5%, and only 0.1378 at an FPR of $10^{-3}$. In the other direction, a measured pair gives $\mu = \Phi^{-1}(1 - \alpha) + \Phi^{-1}(\mathrm{TPR})$, or `norm.isf(alpha) + norm.ppf(tpr)`. If that estimate changes with $\alpha$, the scores do not have this shape.

## The Laplace trade-off curve {#sec-ht-laplace}

Take a Laplace distribution and a copy shifted by $\kappa$ times its scale. Their likelihood ratio stays between $e^{-\kappa}$ and $e^{\kappa}$ ([Divergences](03-divergences.md)). So each outcome that a test rejects is at most $e^\kappa$ times more likely under $Q$ than under $P$, and every test has
$$\mathrm{TPR} \le e^\kappa\,\mathrm{FPR}.$$
At a low FPR the best test reaches this cap. At $\kappa = 1$ and an FPR of 0.05, its TPR is $e \times 0.05 = 0.1359$.

This cap makes the Laplace pair different from the normal one. A Laplace pair at $\kappa = 1$ and a normal pair at $\mu = 1.0301$ have the same best advantage, the largest TPR minus FPR of all tests (@sec-ht-tv). But at a level of $10^{-6}$, the normal pair allows 36 times the TPR. Its ratio has no cap, so a test can find rare observations that give strong evidence. So compare two pairs at the level you will use.

## Best advantage and total variation {#sec-ht-tv}

The **advantage** of a test is its TPR minus its FPR. The best advantage over all tests is the **total variation (TV)** distance of [Divergences](03-divergences.md):
$$\max(\mathrm{TPR} - \mathrm{FPR}) = \mathrm{TV}(P, Q) = \frac12\sum_x |q(x) - p(x)|.$$
For the normal pair it is $2\Phi(\mu/2) - 1$, or 0.3829 at a shift of one standard deviation. Accuracy on a balanced set is one half plus half the advantage. So accuracy measures the same single point of the curve, and it says nothing about the low-FPR end.

## ROC summaries: why AUC hides the low-FPR regime {#sec-ht-auc}

People often compare detectors by the **area under the curve (AUC)** of their ROC curves, the chance that a random positive scores higher than a random negative. But the area gives the same weight to every FPR from 0 to 1. So the low FPRs where a strict detector works add almost nothing to it. For example, take two detectors with standard normal scores on negatives. The positives of detector A have mean 1 and standard deviation 1. Those of B have mean 1.5811 and standard deviation 2, so the two detectors have the same AUC of 0.7602.

But at an FPR of $10^{-3}$, A catches 0.0183 of the positives and B about 12 times as many (@fig-ht-auc-low-fpr). A few positives of B score far above all negatives, and a strict threshold catches exactly those.

![Two detectors with the same AUC of 0.760, on linear axes (left) and log axes (right).](figures/hypothesis_testing_auc_low_fpr.svg){#fig-ht-auc-low-fpr}

So plot the ROC curve on log–log axes, and report the TPR at fixed low FPRs. Drop the first point of `roc_curve`, because its FPR is 0. Give `roc_curve(y_true, y_score)` scores, not the labels from `predict`. At a target FPR, take the highest TPR among the points at or below it, because no single threshold reaches a value from `np.interp`. And do not read a TPR at an FPR below one over the number of negatives, because there the curve rests on one false positive or none.

## The Bayesian view {#sec-ht-bayes}

Often you know a base rate $\pi$, for example that one transaction in a thousand is fraud. Bayes' rule says that the odds of $H_1$ after you see $x$ are the odds before, multiplied by the likelihood ratio:
$$\frac{\Pr[H_1 \mid x]}{\Pr[H_0 \mid x]} = \frac{\pi}{1 - \pi} \times \Lambda(x).$$ {#eq-ht-posterior-odds}
The decision with the fewest errors says $H_1$ when these odds exceed 1, so it is again an LR test.

The same rule applies to an alarm, whose likelihood ratio is the TPR divided by the FPR. Take the test with $\mu = 2$ in a population where 1% are positive. At an FPR of 0.05, a flagged case is positive with probability only 0.114, so most alarms are false. At an FPR of $10^{-3}$, the ratio of the alarm is much larger, and that probability rises to 0.582. So when positives are rare, the FPR decides whether you can believe an alarm.

## Measuring rates from finite trials {#sec-ht-estimation}

To measure the FPR of a detector, count the false positives among $n$ test negatives. If the detector flags each negative independently with the same small probability, the count $k$ is binomial and varies by about $\sqrt k$.

The **Clopper–Pearson interval** holds the true rates under which the observed count is not surprising. It is exact, because it uses the binomial itself, and conservative, because a 95% interval covers the true rate at least 95% of the time. In SciPy, it is `binomtest(k, n).proportion_ci(method="exact")`. A claim that an FPR is below a target needs only the upper end, which `alternative="less"` gives. With no events at all, the one-sided 95% bound is about $3/n$, the **rule of three**. The exact computation shows that an FPR below $10^{-3}$ needs 2,995 negatives with no false positive.

These intervals are wide. Even 10 false positives in 10,000 negatives give the interval $[0.00048, 0.00184]$, from about half to almost twice the observed rate. About 100 events fix a rate within about 20%, so an FPR of $p$ needs about $100/p$ negatives. The **Wald interval** treats the count as normal and fails here, because it is too narrow and becomes $[0, 0]$ with no events.

## Evaluating a detector at a low false positive rate {#sec-ht-recipe}

1. Fix the target FPRs, such as $10^{-2}$ and $10^{-3}$.
2. Plan the trials with about $100/p$ negatives for a target FPR $p$.
3. Choose the threshold on the negatives of one split, and measure the rates on another. A threshold tuned on the evaluation set looks better there than it is. The calibration split needs as many negatives, because below one over their number, `np.quantile` only interpolates between the top scores.
4. Report the TPR and FPR measured at that threshold, with Clopper–Pearson intervals on the two, and the full ROC curve on log–log axes. A TPR read off the ROC curve of the test split is too high, because it picks its threshold on the test data.
5. Compare two detectors at the same FPR, never by AUC or accuracy alone. On a shared evaluation set, check the gap with McNemar's test, `binomtest(m1, m1 + m2, 0.5)`. Here $m_1$ and $m_2$ count the positives that only one of the two detectors catches.

The snippet runs steps 3 and 4 on scikit-learn's breast cancer dataset, with a logistic regression as the detector and malignant tumours as positives. It keeps 30% of the data for calibration and 30% for testing. For a target FPR of 1%, it puts the threshold at the 99th percentile of the benign scores in the calibration split.

```python
import numpy as np
from scipy.stats import binomtest
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

X, y = load_breast_cancer(return_X_y=True)
y = 1 - y  # make malignant (coded 0) the positive class
X_tr, X_ev, y_tr, y_ev = train_test_split(X, y, test_size=0.6, stratify=y, random_state=0)
X_cal, X_te, y_cal, y_te = train_test_split(X_ev, y_ev, test_size=0.5, stratify=y_ev, random_state=0)
model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)).fit(X_tr, y_tr)
threshold = np.quantile(model.decision_function(X_cal[y_cal == 0]), 0.99)  # target FPR 1%, set on calibration
s = model.decision_function(X_te)
pos, neg = s[y_te == 1] > threshold, s[y_te == 0] > threshold  # flagged positives, negatives
tpr = binomtest(int(pos.sum()), pos.size).proportion_ci(method="exact")
fpr_upper = binomtest(int(neg.sum()), neg.size, alternative="less").proportion_ci(method="exact").high
print(pos.sum(), pos.size, f"{pos.mean():.3f} [{tpr.low:.3f}, {tpr.high:.3f}]")  # 56 63 0.889 [0.784, 0.954]
print(neg.sum(), neg.size, f"{neg.mean():.4f} <= {fpr_upper:.4f}")  # 2 108 0.0185 <= 0.0571
```

The detector catches 56 of the 63 malignant test cases, a TPR of 0.889 with a 95% interval from 0.784 to 0.954. But the threshold for 1% flags 2 of the 108 benign test cases. From those counts, the one-sided 95% bound shows only that the FPR is below 0.0571. Step 2 asks for about 10,000 negatives at this target, so on a dataset this small, report the interval and do not claim the target.

## Cheat sheet

| Question | Answer |
| --- | --- |
| What is the best test? | Threshold the likelihood ratio, per group if groups score differently (Neyman–Pearson) |
| How good can any test be? | The trade-off curve; normal pair $G_\mu(\alpha) = \Phi(\Phi^{-1}(1 - \alpha) - \mu)$ |
| What does a capped ratio allow? | $\mathrm{TPR} \le e^\kappa\,\mathrm{FPR}$ |
| What do accuracy and advantage measure? | Best advantage $=$ TV, one point of the curve |
| Why is AUC not enough? | Equal AUC can hide a 12-fold gap in TPR at an FPR of $10^{-3}$ |
| What does a base rate do? | Posterior odds $=$ prior odds $\times$ TPR/FPR |
| How sure is a measured rate? | Clopper–Pearson; zero events give about $3/n$ at 95% |

: The questions of this chapter and their answers. {#tbl-ht-cheat-sheet}

**Rules of thumb**

- Plan for $100/p$ negatives, and use Clopper–Pearson, never Wald.
- Report TPR at fixed low FPRs on log–log axes, never AUC alone.

## References

- Fawcett, "An introduction to ROC analysis" (Pattern Recognition Letters, 2006): ROC curves and their pitfalls.
- Brown, Cai & DasGupta, "Interval Estimation for a Binomial Proportion" (Statistical Science, 2001): why Wald fails and what to use.

**Derivations.** Every result here is derived step by step in [the full edition's chapter](../../book-full/part01-prerequisites/04-hypothesis-testing.md).
