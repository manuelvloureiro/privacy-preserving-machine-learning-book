# Hypothesis testing

## Why this matters

A spam filter must call an email legitimate or spam, and a fraud detector must call a transaction normal or fraudulent. A monitoring job must likewise decide from a batch of readings whether a sensor has drifted. In each case an observation came from one of two distributions, and you must decide which. So every detector and every binary classifier is a hypothesis test, and evaluating one means measuring a test's two error rates.

This chapter gives you the best possible test, the curve that says how good any test can be, and a way to measure a real detector's error rates on a finite set of trials. Throughout, a single summary number such as accuracy or the area under the curve hides how a detector behaves when false alarms must be rare. That regime is often the one that matters.

## Tests and their two errors {#sec-ht-setup}

The default assumption, called the **null hypothesis** $H_0$, is that the observation comes from a distribution $P$, such as the scores of normal transactions. The **alternative hypothesis** $H_1$ is that it comes from a distribution $Q$, such as the scores of fraudulent ones. A **test** decides whether to reject the null, which here means flagging the transaction.

A test can make two kinds of mistake. Flagging a normal case is a false positive, and its probability is the **false positive rate (FPR)**, written $\alpha$. Missing a real case is a false negative, with probability $\beta$, the **false negative rate (FNR)**. The **power** $1 - \beta$ is the **true positive rate (TPR)**, the share of real cases the test catches. Catching more of them always means more false alarms. So the convention is to fix a budget for the FPR first, called the **level**, and then to find the most powerful test within it.

## The Neyman–Pearson lemma {#sec-ht-np}

Write $p(x)$ and $q(x)$ for the probabilities, or densities, of an observation $x$ under the two hypotheses. Their quotient is the **likelihood ratio**
$$\Lambda(x) = \frac{q(x)}{p(x)},$$
which puts the alternative on top, unlike [Divergences](03-divergences.md). A **likelihood ratio (LR) test** picks a threshold $\eta$ and rejects wherever the ratio exceeds it. The **Neyman–Pearson lemma** says that this is the most powerful test at its level. The reason is that rejecting $x$ costs $p(x)$ of false-alarm budget and buys $q(x)$ of power, so the best test spends its budget on the best exchange rates first. On discrete data the LR test may have to reject a boundary value only with some probability, a **randomized test**, to hit a budget exactly.

So the best test looks at an observation only through its likelihood ratio. A single global threshold on a raw score is not that test when the score behaves differently for different items. For example, some merchants' normal transactions score high anyway, while for others a high score is rare unless something is wrong. The better test models each merchant's own two score distributions and thresholds the ratio between them. With a number $n$ of independent observations, the LR test thresholds a sum of log ratios. Its FNR at a fixed FPR then falls like
$$e^{-n\,\mathrm{KL}(P\|Q)},$$
where $\mathrm{KL}(P\|Q)$ is the Kullback–Leibler (KL) divergence. This is the Chernoff–Stein lemma.

## Trade-off curves {#sec-ht-tradeoff}

To say how hard it is to tell $P$ from $Q$, record the best FNR at every level. As a function of the budget, this is the **trade-off function** $T(\alpha)$, and you compute it by sweeping the LR threshold. The **receiver operating characteristic (ROC)** curve holds the same information upside down, as TPR against FPR. Any particular detector traces its own ROC curve, which lies on or below the best one. Every trade-off curve lies on or below the coin-flip line $1 - \alpha$ and is convex, because randomizing between two tests reaches every point between them.

## The Gaussian trade-off curve {#sec-ht-gaussian}

The most common pair in practice is a score that is normal under both hypotheses, with the same spread. Write $\mu$ for the distance between the two centres in standard deviations, and $\Phi$ for the standard normal cumulative distribution function. The best test thresholds at $\Phi^{-1}(1 - \alpha)$, and its FNR is
$$T(\alpha) = \Phi\big(\Phi^{-1}(1 - \alpha) - \mu\big).$$ {#eq-ht-gaussian-tradeoff}
In SciPy this is `norm.cdf(norm.ppf(1 - alpha) - mu)`, and at a tiny $\alpha$ use `norm.isf(alpha)` for the threshold, because $1 - \alpha$ loses the digits that matter. So a single number $\mu$ fixes the best TPR at every FPR.

For example, take $\mu = 2$. At an FPR of 5%, the best test has a TPR of 0.6388. If false alarms must be rare, at a rate of $10^{-3}$, the TPR drops to 0.1378.

## The Laplace trade-off curve {#sec-ht-laplace}

For a Laplace distribution and a copy shifted by $\kappa$ scales, the likelihood ratio is capped between $e^{-\kappa}$ and $e^{\kappa}$ ([Probability distributions](01-distributions.md)). No test can buy power faster than the cap per unit of FPR, so every test has
$$\mathrm{TPR} \le e^\kappa\,\mathrm{FPR}.$$
At $\kappa = 1$ and an FPR of 0.05, the best TPR is 0.1359. The full curve has two more pieces, which the full edition gives. The same cap holds for any pair whose likelihood ratio is bounded by $e^\kappa$, so at a low FPR every test on such a pair has a very small TPR.

A Laplace pair at $\kappa = 1$ and a normal pair at $\mu = 1.0301$ have the same best advantage. At a level of $10^{-6}$, though, the best test on the normal pair has 36 times the power, because the unbounded normal ratio lets it find rare, overwhelming observations. So plan for the level you will actually use.

## Best advantage and total variation {#sec-ht-tv}

The **advantage** of a test is its TPR minus its FPR, called Youden's index in medicine. The best advantage over all tests is the **total variation (TV)** distance of [Divergences](03-divergences.md):
$$\max(\mathrm{TPR} - \mathrm{FPR}) = \mathrm{TV}(P, Q) = \frac12\sum_x |q(x) - p(x)|.$$
It is reached by the test that rejects exactly where $q$ exceeds $p$. For the normal pair it is $2\Phi(\mu/2) - 1$, or 0.3829 at a shift of one standard deviation. Accuracy on a balanced set of positives and negatives measures the same point. But that point says nothing about the ends of the curve, as the Laplace and normal pairs above show. So never report only accuracy or advantage for a detector.

## ROC summaries: why AUC hides the low-FPR regime {#sec-ht-auc}

Detectors are often compared by the **area under the curve (AUC)** of their ROC curves. It is the chance that a random positive scores higher than a random negative:
$$\mathrm{AUC} = \Pr[S_+ > S_-].$$
But the AUC weighs mostly false-alarm rates nobody would operate at, while the question is often whether a detector can flag some cases with confidence, as when a fraud team can review only a few transactions.

Two detectors with standard normal scores on negatives make the point. Detector A's positives score normal with mean 1 and standard deviation 1. Detector B's positives are more spread out, with mean 1.5811 and standard deviation 2. Both have an AUC of 0.7602. At an FPR of $10^{-3}$, though, A catches 0.0183 of the positives, while B catches about 12 times as many. A few of B's spread-out positives score far above every negative, and those are exactly the cases a strict threshold catches.

So plot the full ROC curve on log–log axes, and report the TPR at fixed low FPRs such as $10^{-2}$ and $10^{-3}$. In scikit-learn, `sklearn.metrics.roc_curve(y_true, y_score)` returns the FPRs and TPRs to read them from, and `roc_auc_score(y_true, y_score)` gives the AUC for comparison with older work. Don't read a TPR at an FPR below one over the number of negatives, though, because there the curve rests on a single false positive or none, and it is noise.

## The Bayesian view {#sec-ht-bayes}

Often you know a base rate $\pi$, such as one transaction in a thousand being fraud. Bayes' rule says that the odds of $H_1$ after seeing $x$ are the odds before, multiplied by the likelihood ratio:
$$\frac{\Pr[H_1 \mid x]}{\Pr[H_0 \mid x]} = \frac{\pi}{1 - \pi} \times \Lambda(x).$$ {#eq-ht-posterior-odds}
So the decision with the fewest errors says $H_1$ when the ratio exceeds $(1 - \pi)/\pi$, which is again an LR test.

The same rule applies to an alarm, whose likelihood ratio is the TPR divided by the FPR. Take the test with $\mu = 2$ in a population where 1% are positive. At an FPR of 0.05, a flagged case is positive with probability only 0.114, so most alarms are false. At an FPR of $10^{-3}$, the test flags fewer cases, but a flagged case is positive with probability 0.582. So when positives are rare, the FPR decides whether an alarm is worth believing.

## Measuring rates from finite trials {#sec-ht-estimation}

You measure a detector's FPR by counting false positives among $n$ test negatives. If each negative is flagged independently with the same probability, the count $k$ is binomial, and its relative uncertainty is about $1/\sqrt k$. So the precision of a measured rate depends on how many events you see, not on $n$ alone.

The **Clopper–Pearson interval** gives the range of true rates that make the observed count unsurprising. Write $1 - a$ for the confidence. Its lower end $l$ and upper end $u$ solve
$$\Pr[\mathrm{Bin}(n, l) \ge k] = \frac a2, \qquad \Pr[\mathrm{Bin}(n, u) \le k] = \frac a2.$$
It is exact, because it uses the binomial itself, and conservative, because it covers the true rate at least $1 - a$ of the time. In SciPy, `scipy.stats.binomtest(k, n).proportion_ci(confidence_level=0.95, method="exact")` computes it, and statsmodels' `proportion_confint(k, n, alpha=0.05, method="beta")` does the same.

To claim that an FPR is below a target, use the upper end alone. The upper end of a two-sided 90% interval is a one-sided 95% bound, which is 0.0047 for 1 false positive in 1,000 negatives. With no events at all, the one-sided 95% bound is about $3/n$, the **rule of three**. So certifying an FPR below $10^{-3}$ takes 2,995 negatives with no false positive.

Even 10 false positives in 10,000 negatives give the interval $[0.00048, 0.00184]$, from about half to almost twice the target. As a rule of thumb, measuring an FPR of $\alpha$ to within about 20% takes on the order of $100/\alpha$ negatives. A common shortcut fails here, because the **Wald interval**, a normal approximation, is too narrow and collapses to $[0, 0]$ with no events.

## Evaluating a detector at a low false positive rate {#sec-ht-recipe}

These results give a recipe for evaluating any detector or binary classifier.

1. Fix the target FPRs, such as $10^{-2}$ and $10^{-3}$.
2. Plan the trials with about $100/\alpha$ negatives for a target FPR $\alpha$.
3. Choose the threshold on one split and measure the rates on another, because a threshold tuned on the evaluation set looks better there than it is.
4. Report the TPR at each target FPR, with Clopper–Pearson intervals on both rates, and the full ROC curve on log–log axes.
5. Compare two detectors at the same FPR, never by AUC or accuracy alone. On a shared evaluation set, a paired test such as McNemar's (`statsmodels.stats.contingency_tables.mcnemar`) compares the positives each one catches.

## Cheat sheet

| Question | Answer |
| --- | --- |
| What is the best test? | Threshold the likelihood ratio, per item if items differ (Neyman–Pearson) |
| How good can any test be? | The trade-off curve; normal pair $\Phi(\Phi^{-1}(1 - \alpha) - \mu)$ |
| What does a capped ratio allow? | $\mathrm{TPR} \le e^\kappa\,\mathrm{FPR}$ |
| What do accuracy and advantage measure? | Best advantage $=$ TV, one point of the curve |
| Why is AUC not enough? | Equal AUC can hide a 12-fold gap in TPR at an FPR of $10^{-3}$ |
| What does a base rate do? | Posterior odds $=$ prior odds $\times$ TPR/FPR |
| How sure is a measured rate? | Clopper–Pearson; zero events give about $3/n$ at 95%|

: The questions of this chapter and their answers. {#tbl-ht-cheat-sheet}

**Rules of thumb**

- Use Clopper–Pearson intervals, never the Wald interval, for rare rates.

## Further reading

- Fawcett, "An introduction to ROC analysis" (Pattern Recognition Letters, 2006): ROC curves and AUC as a ranking probability.
- Brown, Cai & DasGupta, "Interval Estimation for a Binomial Proportion" (Statistical Science, 2001): why the Wald interval fails and what to use instead.

**Derivations.** Every result here is derived step by step in [the full edition's chapter](../../book-full/part01-prerequisites/04-hypothesis-testing.md).
