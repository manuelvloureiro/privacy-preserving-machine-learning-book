# Adaptive data analysis

## What it is {#sec-ada-what}

Tuning on a held-out validation set uses scores to choose later settings. This **adaptive** reuse overfits the holdout, making its scores overstate performance on new data.

Each question is a **statistical query**: the average of a per-record value in $[0,1]$ over $n$ held-out records, where $n$ is the holdout size. Classifier accuracy is one example; its population value is the corresponding average on new data. If records are independent and $k$ denotes the number of queries fixed before any answers, the expected largest gap between empirical and population values is at most
$$\sqrt{\frac{\ln(2k)}{2n}}.$$ {#eq-ada-fixed}
The query count affects this bound only logarithmically.

Adaptive queries invalidate this fixed-query bound because later queries can fit the sample's noise. Blum & Hardt's boosting attack demonstrates this on a leaderboard with coin-flip labels, where population accuracy cannot exceed 0.5. After 1,000 submissions, the leaderboard reports 0.7182. This score uses 1,000 holdout records.

Differential privacy (DP) limits how much answers reveal about each record, reducing their ability to overfit the sample.

Let $\alpha$ be the average distance between answers and their empirical values. Let $\varepsilon$ denote the privacy level and $\delta$ the slack. Suppose the complete exchange is differentially private under replace-one neighbours. The **transfer theorem** bounds expected maximum error against population values by
$$\alpha + e^\varepsilon - 1 + \delta,$$ {#eq-ada-transfer}
regardless of how the analyst uses the answers. For small $\varepsilon$, the middle term is approximately $\varepsilon$, so the bound is useful mainly when $\varepsilon\ll1$. It protects answer accuracy, not the people represented in the holdout.

## When you meet it {#sec-ada-when}

Adaptivity arises when a validation set, test set or leaderboard answers questions in multiple rounds, as in hyperparameter search. A round is a batch fixed before receiving its answers. The bounds below depend on the round count; @tbl-ada-choose summarizes the options.

| Situation | What to do |
| --- | --- |
| A final evaluation | Keep a test set and use it once |
| Few adaptive rounds, plenty of data | Give each round its own slice of data |
| Many adaptive rounds, scarce data | Answer through noise, or use Thresholdout |
| A search that is private in the holdout set | Nothing more, since the transfer theorem covers the winner |
| A public leaderboard | Show a new score only when it beats the best by a margin (Blum & Hardt's Ladder) |

: How to reuse a holdout set, by situation. {#tbl-ada-choose}

## How to use it {#sec-ada-use}

The simplest method is **sample splitting**: assign each of $r$ rounds its own $n/r$ records. For $k$ total queries, expected maximum error is at most $\sqrt{r\ln(2k)/(2n)}$, so required data grow proportionally with $r$. Suppose 1,000 rounds evaluate 1,000 adaptively selected configurations. The holdout has 10,000 validation records, and the bound is 0.6165. Accuracy scores lie in $[0,1]$, so that bound is too large. To reduce error to 0.05, sample splitting would require 1,520,181 records. Fixed queries require 1,521 records for the same error.

The transfer theorem implies that Gaussian-noisy answers require data growing only as $\sqrt{k}$. Its constants are large, however, so sample splitting remains cheaper for up to a few hundred adaptive queries.

**Thresholdout** (Dwork et al. 2015) answers using training-set values while they agree with holdout values; these answers reveal nothing about the holdout. When the difference exceeds a noisy threshold, it records a **detection** and returns a noisy holdout value. Privacy cost depends on detections rather than total queries. However, the full edition's experiment gives $\varepsilon\approx29$. No mainstream library implements the method, so treat it as a practical limit on reuse rather than a strong guarantee.

## What goes wrong {#sec-ada-wrong}

The common failure is a long search that selects each setting from earlier validation scores. A fixed grid evaluated once, as in scikit-learn's `GridSearchCV`, is one round, so @eq-ada-fixed bounds the selected score's overstatement. Count and report the rounds of validation reuse. Typical overfitting may be smaller than the worst-case bound, but risk remains high with few records and many rounds.

## References

- Dwork et al., "The Reusable Holdout" (*Science* 2015): Thresholdout.
- Blum & Hardt, "The Ladder" (ICML 2015): the boosting attack and a leaderboard that resists it.

**Derivations.** Every result here is derived step by step in [the full edition's chapter](../../book-full/part05-dp-ml/11-adaptive-data-analysis.md).
