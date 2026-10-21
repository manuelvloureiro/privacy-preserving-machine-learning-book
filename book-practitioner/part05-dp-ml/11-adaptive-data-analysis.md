# Adaptive data analysis

## Why this matters

Machine learning reuses its evaluation data all the time. For example, you tune hyperparameters on a validation set, look at the scores and try new settings, and a competition shows a leaderboard score after every submission. Each time, the next question depends on the answers to the earlier ones, which makes the analysis **adaptive**. The practical question is how many such questions one holdout set can answer before its answers stop telling you anything about new data.

The answer depends on how the questions are answered. Exact scores on a holdout set break quickly. Giving each question its own fresh slice of data keeps the answers valid, but it needs data in proportion to the number of questions. Differential privacy (DP) does better. Answers that reveal little about any single record cannot be used to overfit the records behind them. So with noisy answers, the data you need grows only like the square root of the number of questions.

## What to do in practice {#sec-ada-practice}

What decides the cost is the number of adaptive **rounds**, not the number of queries. A round is a batch of questions fixed before any of its answers is seen. @tbl-ada-choose gives the method for each situation.

| Situation | What to do | Cost in records |
| --- | --- | --- |
| Questions fixed before any answer is seen | Answer them exactly on one holdout set | Grows like $\ln k$ in the number $k$ of questions |
| A final evaluation | Keep a test set and use it once | One round |
| Few adaptive rounds, plenty of data | Give each round its own slice of data | Grows like the number of rounds |
| Many adaptive rounds, scarce data | Answer through Gaussian noise (@sec-ada-gauss) | Grows like $\sqrt k$ |
| A training set to explore and a holdout set to confirm | Thresholdout (@sec-ada-thresholdout) | Pays only when the two disagree |
| Tuning that is already private | Nothing more to do, since privacy protects the winner's score | Free |
| A public leaderboard | Release improvements, not scores, as in the Ladder | Resists the boosting attack |

: How to reuse a holdout set, by situation. {#tbl-ada-choose}

Choosing the best of $k$ models trained without the validation set is one round. With a number $n$ of validation records, the winner's score then overstates its true score by at most $\sqrt{\ln(2k)/(2n)}$ on average. Your search may already be private in the validation set, at a privacy level $\varepsilon$ with slack $\delta$ ([Hyperparameter tuning and private selection](03-hyperparameter-tuning.md)). Then the winner's score overstates its true score by at most $e^\varepsilon - 1 + \delta$ on average (@sec-ada-transfer). On a leaderboard, Blum & Hardt's **Ladder** reports a new score only when a submission beats the best so far by more than a fixed step.

The classical fix inside one dataset is nested cross-validation, which keeps the folds that score the search away from the folds that choose it. Whatever you use, report how many evaluations a validation number has survived. Practice is often milder than the worst case. The risk is real when the records are few, the adaptive rounds are many, and someone has an incentive to climb a leaderboard.

## Reusing a holdout set overfits {#sec-ada-overfit}

A **statistical query** asks a question about each record whose answer is a number between 0 and 1, such as "does this classifier get this record right?". Its empirical value is its average over the $n$ records you hold, and its population value is its average over new data. If all $k$ queries are fixed before the data are drawn, then on average the largest error among their exact answers is at most
$$\sqrt{\frac{\ln(2k)}{2n}}.$$ {#eq-ada-fixed}
The number of queries hardly matters, because it enters only through a logarithm.

That guarantee fails as soon as a query depends on earlier answers. Blum & Hardt's **boosting attack** shows it on a leaderboard whose holdout labels are fair coin flips, so that nothing can beat 50% on new data. The attacker submits $k$ random labellings, reads their exact scores, and submits the majority vote of those that scored above 50%. With 1,000 submissions on 1,000 holdout records, the final submission reports 0.7182. The overfitting is close to
$$\frac{1}{\pi}\sqrt{\frac{k}{2n}},$$
so it depends only on the ratio of submissions to holdout records.

What goes wrong in practice is rarely a deliberate attack. A long hyperparameter search, run with scikit-learn's `GridSearchCV` on one validation split, does the same thing, because the winner is the setting that happened to fit that split's noise best. So its validation score overstates its test score.

## Sample splitting and its cost {#sec-ada-split}

The simplest fix never lets a record answer a question it helped to choose. Split the sample into $k$ equal parts and answer each query on its own part. Then, however the queries were chosen, the largest error is on average at most
$$\sqrt{\frac{k\ln(2k)}{2n}}.$$ {#eq-ada-split}
The data you need now grows linearly with $k$. With queries in $r$ rounds, one part per round is enough, and the $k$ in front of the logarithm becomes $r$.

Take a validation set of 10,000 records and a sequential search over 1,000 configurations, each chosen from the earlier scores. Splitting restores validity but leaves the bound at 0.6165, useless for scores between 0 and 1. Bringing it down to 0.05 takes 1,520,181 records, against 1,521 for fixed queries.

## Differential privacy implies generalization {#sec-ada-transfer}

Both results below use $(\varepsilon, \delta)$-DP for samples that differ by *replacing* one record, which doubles an add-or-remove $\varepsilon$.

The **transfer lemma** covers a single query chosen by a private algorithm. If the algorithm is $(\varepsilon, \delta)$-DP, the chosen query's empirical and population values differ on average by at most
$$e^\varepsilon - 1 + \delta.$$ {#eq-ada-lemma}
The **transfer theorem** covers a whole interaction. Suppose a mechanism answers $k$ adaptive queries, the whole exchange is $(\varepsilon, \delta)$-DP, and its answers are within $\alpha$ of the empirical values on average. Then its largest error against the population is on average at most
$$\alpha + e^\varepsilon - 1 + \delta.$$ {#eq-ada-transfer}
The bound holds whatever the analyst does with the answers. For small $\varepsilon$, privacy at level $\varepsilon$ costs about $\varepsilon$ in accuracy, so answering to within 0.05 needs $\varepsilon$ around 0.05 or less. Bassily et al. (2016) proved versions of the theorem that hold with high probability rather than on average.

This DP protects the validity of the answers, not the people in the holdout set, so a holdout set of personal data needs its own privacy accounting.

## Answering with Gaussian noise {#sec-ada-gauss}

The simplest private mechanism adds Gaussian noise with a standard deviation $\sigma$ to each empirical value. Composing the $k$ answers gives a privacy level $\varepsilon$ ([Concentrated differential privacy](../part04-privacy-accounting/03-concentrated-dp.md)), and with this level the transfer theorem bounds the expected largest error by
$$\sigma\sqrt{2\ln(2k)} + e^\varepsilon - 1 + \delta.$$ {#eq-ada-gauss}
The first term is the noise, and the rest is the transfer term. More noise shrinks the transfer term but grows the noise term, so there is a best $\sigma$. At that balance the records you need grow like $\sqrt k$, because the privacy loss of $k$ Gaussian answers grows like $\sqrt k$.

To use it, choose the $\sigma$ and $\delta$ that minimize the bound, as the snippet below does. @tbl-ada-rates compares the three approaches.

| Queries $k$ | Fixed in advance | Sample splitting | Gaussian noise |
| --- | --- | --- | --- |
| 1 | 139 | 139 | 7,300 |
| 1,000 | 1,521 | 1,520,181 | 764,337 |
| 1,000,000 | 2,902 | 2,901,731,548 | 33,393,779 |

: Records needed to bring each bound on the expected largest error down to 0.05. The fixed-query column is valid only for queries chosen in advance. {#tbl-ada-rates}

The constants are large, so even a single query needs 7,300 records with noise, because the transfer term must itself be below 0.05. So splitting stays cheaper up to a few hundred adaptive queries, and only beyond that does the square root win.

In the other direction, when each record has many features, no efficient mechanism can answer much more than about $n^2$ adaptive queries (Hardt & Ullman 2014, assuming one-way functions exist).

## The reusable holdout: Thresholdout {#sec-ada-thresholdout}

**Thresholdout** (Dwork et al. 2015) answers from the training set while they agree and pays privacy only when they do not. It takes a threshold $T$, a noise scale $\sigma$ and a budget of $B$ detections. Below the noisy threshold it returns the training value at no cost, and above it, it counts an **overfitting detection** and returns a noisy holdout value. This is the sparse vector technique of [Selection](../part03-dp-foundations/05-selection.md), and no mainstream library ships it, but it takes a few lines of NumPy:

```text
rng = np.random.default_rng()
noisy_t = T + rng.laplace(scale=2 * sigma)
def answer(train_value, holdout_value):
    global noisy_t, detections
    if abs(holdout_value - train_value) + rng.laplace(scale=4 * sigma) <= noisy_t:
        return train_value
    detections += 1                     # stop answering once detections reaches B
    noisy_t = T + rng.laplace(scale=2 * sigma)
    return holdout_value + rng.laplace(scale=sigma)
```

Write $\Delta$ for the most one holdout record can move a query. A run that stops at its $B$-th detection is DP in the holdout records at
$$\varepsilon = \frac{2B\Delta}{\sigma},$$ {#eq-ada-tho}
however many queries were answered below the threshold. The threshold sets the resolution of every answer, and when the budget is spent the holdout set must stop answering.

In the full edition's experiment, an analyst screens noise features on a training set and confirms them on a holdout set. A holdout set answered exactly then reports 0.6842 for a classifier whose true accuracy is 0.5, while Thresholdout reports 0.5217. But its formal guarantee is weak, because the run triggered 325.8 detections on average, about $\varepsilon = 29.1$.

## Code

This snippet computes the three bounds for 1,000 queries on 10,000 records, and checks one entry of @tbl-ada-rates.

```python
from math import expm1, log, sqrt


def fixed_queries(n: int, k: int) -> float:
    """Expected largest error of k queries chosen before the data were drawn."""
    return sqrt(log(2 * k) / (2 * n))


def splitting(n: int, k: int) -> float:
    """Expected largest error when each adaptive query gets its own n/k records."""
    return sqrt(k * log(2 * k) / (2 * n))


def gaussian(n: int, k: int, sigma: float, delta: float) -> float:
    """Transfer-theorem bound for k adaptive queries answered with Gaussian noise sigma."""
    rho = k / (2 * n**2 * sigma**2)
    eps = rho + 2 * sqrt(rho * log(1 / delta))
    return sigma * sqrt(2 * log(2 * k)) + expm1(min(eps, 50.0)) + delta  # cap avoids overflow


def best_gaussian(n: int, k: int) -> float:
    """The smallest Gaussian bound over a grid of noise levels and deltas."""
    grid = [10 ** (e / 100) for e in range(-600, 1)]
    return min(gaussian(n, k, s, d) for s in grid for d in grid)


print(f"{fixed_queries(10_000, 1000):.4f}")  # 0.0195
print(f"{splitting(10_000, 1000):.4f}")  # 0.6165
print(f"{best_gaussian(10_000, 1000):.4f}")  # 0.4095
print(f"{best_gaussian(764_337, 1000):.4f}")  # 0.0500
```

The full edition's script runs the attacks and the reusable-holdout experiment:

```bash
uv run python code/part05-dp-ml/adaptive_data_analysis.py
```

## Cheat sheet

| Method | Expected largest error | Conditions |
| --- | --- | --- |
| Fixed queries | @eq-ada-fixed | Queries chosen before the data were drawn |
| Sample splitting | @eq-ada-split, with $k$ the number of rounds | Each round on its own records |
| Transfer theorem | $\alpha + e^\varepsilon - 1 + \delta$ | Whole interaction DP under replacement |
| Gaussian noise | @eq-ada-gauss, records growing like $\sqrt k$ | Noise $\sigma$ on every answer |
| Thresholdout | Privacy $\varepsilon = 2B\Delta/\sigma$ for $B$ detections | Answers below the threshold are free |

: The results of this chapter. {#tbl-ada-cheat}

**Rules of thumb**

- Count adaptive rounds, not queries.
- Keep a test set that is used once, at the end, and never for model selection.
- Privacy at level $\varepsilon$ costs about $\varepsilon$ in accuracy, so validity needs a small $\varepsilon$.
- Below a few hundred adaptive queries, splitting is cheaper than noise.
- A large holdout set does not make many rounds of tuning safe, since adaptive error grows with the number of rounds.
- On a leaderboard, release improvements, never exact scores.

## Further reading

- Dwork, Feldman, Hardt, Pitassi, Reingold & Roth, "The Reusable Holdout: Preserving Validity in Adaptive Data Analysis" (*Science* 2015): Thresholdout and the experiment this chapter scales down.
- Blum & Hardt, "The Ladder: A Reliable Leaderboard for Machine Learning Competitions" (ICML 2015): the boosting attack and a leaderboard that resists it.

**Derivations.** Every result here is derived step by step in [the full edition's chapter](../../book-full/part05-dp-ml/11-adaptive-data-analysis.md).
