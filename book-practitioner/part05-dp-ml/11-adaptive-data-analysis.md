# Adaptive data analysis

## What it is {#sec-ada-what}

When you tune on a held-out validation set, you read the scores and try new settings, so each question depends on earlier answers. Such **adaptive** reuse overfits the holdout set, whose scores then overstate how the model does on new data.

Each such question is a **statistical query**, which averages a per-record answer between 0 and 1 over the $n$ records you hold. A classifier's accuracy is an example. Its population value is the same average over new data. Suppose the records are drawn independently and a number $k$ of queries are fixed before any answer is seen. Then the largest gap between the two values is on average at most
$$\sqrt{\frac{\ln(2k)}{2n}}.$$ {#eq-ada-fixed}
The number of queries enters through a logarithm, so it hardly matters.

Adaptive queries break this, because a later query can be built to fit the sample's own noise. For example, Blum & Hardt's boosting attack runs on a leaderboard whose labels are coin flips, where nothing can beat 0.5 on new data. Yet after 1,000 submissions on as many holdout records, it reports 0.7182.

Differential privacy (DP) is the remedy, because answers that reveal little about any record cannot be used to overfit it. Let a mechanism's answers be on average within a distance $\alpha$ of the empirical values. Let the whole exchange be DP at a privacy level $\varepsilon$, with a small slack $\delta$, for samples that differ in one replaced record. The **transfer theorem** then bounds the expected largest error against the population by
$$\alpha + e^\varepsilon - 1 + \delta,$$ {#eq-ada-transfer}
whatever the analyst does with the answers. For small $\varepsilon$ the middle term is about $\varepsilon$, so the bound is useful only well below $\varepsilon = 1$. It protects the validity of the answers, not the people in the holdout set.

## When you meet it {#sec-ada-when}

You meet adaptivity when a validation set, a test set or a leaderboard answers more than one round of questions, as in hyperparameter search. A round is a batch of questions fixed before any of its answers is seen, and the bounds below grow with the number of rounds. @tbl-ada-choose gives the choice for each situation.

| Situation | What to do |
| --- | --- |
| A final evaluation | Keep a test set and use it once |
| Few adaptive rounds, plenty of data | Give each round its own slice of data |
| Many adaptive rounds, scarce data | Answer through noise, or use Thresholdout |
| A search that is private in the holdout set | Nothing more, since the transfer theorem covers the winner |
| A public leaderboard | Show a new score only when it beats the best by a margin (Blum & Hardt's Ladder) |

: How to reuse a holdout set, by situation. {#tbl-ada-choose}

## How to use it {#sec-ada-use}

The simplest fix is **sample splitting**, which answers each of $r$ rounds on its own $n/r$ records. The expected largest error is then at most $\sqrt{r\ln(2k)/(2n)}$ for $k$ queries in all, so the data you need grow in proportion to $r$. Take 10,000 validation records and a search over 1,000 configurations, each chosen from the earlier scores, so $r = k = 1{,}000$. Splitting then gives a bound of 0.6165, useless for scores between 0 and 1. A bound of 0.05 would need 1,520,181 records, where fixed queries would need 1,521.

Answering every query with Gaussian noise, under the transfer theorem, needs records that grow only like $\sqrt k$. But its constants are large, so splitting stays cheaper up to a few hundred adaptive queries.

**Thresholdout** (Dwork et al. 2015) answers from a training set while the training value agrees with the holdout value. Such answers reveal nothing about the holdout set. When they differ by more than a noisy threshold, it records a **detection** and returns a noisy holdout value. So its privacy cost grows with the number of detections, not of queries. But its formal $\varepsilon$ in the full edition's experiment is about 29, and no mainstream library ships it. So treat it as a brake, not a guarantee.

## What goes wrong {#sec-ada-wrong}

The common failure is not an attack but a long search that picks each setting from earlier validation scores. A fixed grid scored once, as in scikit-learn's `GridSearchCV`, is a single round, so @eq-ada-fixed bounds its winner's overstatement. So count rounds, and report how many a validation number has been through. Practice is often milder than the worst case, but the risk is real with few records and many rounds.

## References

- Dwork et al., "The Reusable Holdout" (*Science* 2015): Thresholdout.
- Blum & Hardt, "The Ladder" (ICML 2015): the boosting attack and a leaderboard that resists it.

**Derivations.** Every result here is derived step by step in [the full edition's chapter](../../book-full/part05-dp-ml/11-adaptive-data-analysis.md).
