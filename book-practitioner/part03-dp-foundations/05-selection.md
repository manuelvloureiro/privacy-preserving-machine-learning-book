# Selection: exponential mechanism, report-noisy-max, sparse vector

## Why this matters

Many analyses select an item, such as the best model on a validation set, a median or the first statistic above a threshold. Publishing every candidate score with Laplace noise spends budget on every score. The mechanisms in this chapter **release the selected position, not the scores**. Under differential privacy (DP), selecting one item costs one query regardless of the number of candidates; the candidate count affects accuracy through its logarithm.

## Which mechanism to use {#sec-sel-choose}

@tbl-sel-choose maps common tasks to mechanisms. In each case, **sensitivity** $\Delta u$ is the maximum change one person's record can cause to a candidate's score.

| Task | Mechanism | Privacy cost |
| --- | --- | --- |
| Pick the best of a fixed list | Exponential mechanism or report-noisy-max | $\varepsilon$ |
| Pick the top $c$, such as the most frequent items | Gumbel noise, keep the $c$ largest, or noisy max $c$ times | $c\varepsilon$ |
| Pick a number from a range, such as a quantile or a clipping norm | Exponential mechanism over the range | $\varepsilon$ |
| Find the first query in a stream above a threshold | AboveThreshold | $\varepsilon$, however long the stream |

: Which selection mechanism to use. {#tbl-sel-choose}

## The exponential mechanism {#sec-sel-exp}

Let $\mathcal R$ be a fixed candidate set. Each candidate $r$ has a **score** $u(D,r)$ on dataset $D$, such as the number of validation records a model classifies correctly. The **exponential mechanism** of McSherry and Talwar (2007) samples candidates with probability proportional to an exponential of their scores. With privacy budget $\varepsilon$, it outputs candidate $r$ with probability
$$\Pr[M_E(D) = r] = \frac{\exp\big(\varepsilon\,u(D, r)/(2\Delta u)\big)}{\sum_s\exp\big(\varepsilon\,u(D, s)/(2\Delta u)\big)}.$$ {#eq-sel-em}
It is $\varepsilon$-DP because one record changes each weight, and also their sum, by at most a factor $e^{\varepsilon/2}$, which makes $e^\varepsilon$ for their ratio.

Two conditions determine whether the mechanism is valid. First, fix the candidate list independently of the private data; a data-derived list may itself reveal records, such as a word used by one person. Second, remove the factor 2 in the exponent only for **monotone** scores, where one record changes all scores in the same direction. Counts under add/remove are monotone. A histogram under replace is not: replacing a record lowers one count and raises another, doubling the privacy cost if the factor 2 is omitted.


## Accuracy, and the cost of many candidates {#sec-sel-utility}

The number of candidates $|\mathcal R|$ does not affect privacy but can reduce accuracy by increasing the number of low-scoring choices. Define the **gap** as the best score minus the selected score. Its expected value is at most
$$\frac{2\Delta u}{\varepsilon}(\ln|\mathcal R| + 1),$$ {#eq-sel-em-utility}
so doubling the candidate count adds only $\ln 2$ to the bracket.

Suppose 50 models are evaluated on a private validation set of 1,000 records. Model $r$, numbered from 1, correctly classifies $900-r$ records, so model 1 is best. A single record changes this count by at most 1, giving $\Delta u = 1$. At $\varepsilon = 1$, the mechanism selects the best model 39% of the time but loses only 1.54 correct predictions on average, or 0.15% accuracy, because errors are mostly near-ties. [Hyperparameter tuning](../part05-dp-ml/03-hyperparameter-tuning.md) uses this method.

## A private quantile or clipping norm {#sec-sel-apps}

A median or clipping norm computed directly from private data can leak information. Use the exponential mechanism and fix the candidate range before accessing the data, then clip records to that range. Score each candidate by how closely the fraction of records below it matches the target quantile; for the median, the target is one half. Adding or removing one record changes each score by at most 1. Scores are constant between adjacent sorted records, so the mechanism selects one such interval, called a **piece**, and returns a uniform point within it. Longer pieces near the target quantile receive higher probability.

diffprivlib's `tools.quantile` and `tools.median` accept `epsilon` and a `bounds` range. OpenDP's `.dp.quantile` and `.dp.median` operate on Polars data frames and select from a finite candidate list. The script [practitioner_adult_statistics.py](../../code/part03-dp-foundations/practitioner_adult_statistics.py) estimates the median age in Adult, a census dataset with 48,842 people and one row per person, at $\varepsilon = 1$. The survey documentation supplies the range 17 to 90.

```{.python excerpt="adult-median-diffprivlib"}
age = adult["age"].to_numpy()
private = tools.median(age, epsilon=1.0, bounds=(17, 90), random_state=seed)
print(f"true median age: {np.median(age):.1f}  diffprivlib median: {private:.2f}")
```

With `random_state=0`, it prints a true median of 37.0 and a private median of 37.72. Ages are integers, so points between 37 and 38 have the same score; the mechanism samples a point from that piece. OpenDP requires a public upper bound on row count, specified by a `Margin`. Choose a bound from public information that is at least the true row count; a larger bound has little cost.

```{.python excerpt="adult-median-opendp"}
context = dp.Context.compositor(
    data=adult.lazy(),
    privacy_unit=dp.unit_of(contributions=1),  # one row per person
    privacy_loss=dp.loss_of(epsilon=1.0),
    split_evenly_over=1,
    margins=[dp.polars.Margin(max_length=100_000)],  # a public bound on the number of rows
)
age_col: Any = pl.col("age")  # OpenDP adds .dp to Polars expressions when it is imported
query = context.query().select(age_col.dp.median(candidates=list(range(17, 91))))
print(query.summarize(alpha=0.05))
released = query.release().collect()  # changes on every run: OpenDP's noise has no seed
```

The summary reports `ExponentialMin` with scale 2.0 because OpenDP 0.16 adds exponential noise to each candidate score and selects the best. This is **permute-and-flip** (McKenna & Sheldon 2020), which has the exponential mechanism's guarantee and no larger expected error. Accuracy is `null` because OpenDP does not provide a quantile error bar. The noise is unseeded, so the released value changes each time `uv run python code/part03-dp-foundations/practitioner_adult_statistics.py` runs.

The error bound concerns rank, not value. With probability $1 - \beta$, the exponential-mechanism gap is at most $(2\Delta u/\varepsilon)(\ln|\mathcal R| + \ln(1/\beta))$, the tail bound corresponding to @eq-sel-em-utility. For a median with $\Delta u = 1$, the gap is the number of records between the output and the middle rank. At $\beta = 0.05$, this gives a 95% rank interval. A value interval requires the number of records near the median, which the release does not reveal. A dashboard needing value accuracy can spend budget on noisy counts below selected candidates. A wide range adds little cost because the gap depends on the logarithm of the candidate count.

Two limitations matter in practice. If data are sparse near the quantile, the best piece is long, so a small rank error may produce a large value error. The range and candidates must also be independent of private data. Setting the upper bound to the observed maximum reintroduces leakage. diffprivlib raises a `PrivacyLeakWarning` when `bounds` is omitted ([Sensitivity](07-sensitivity.md)).

## Report-noisy-max and Gumbel noise {#sec-sel-rnm}

**Report-noisy-max** adds independent Laplace noise to the score of each candidate and outputs the candidate with the largest noisy score. It is $\varepsilon$-DP for any number of candidates when the noise has scale $2\Delta u/\varepsilon$, or $\Delta u/\varepsilon$ when one record pushes all scores the same way. Release only the index of the winner, and pay for a fresh noisy answer if you need its score.

Gumbel noise of scale $2\Delta u/\varepsilon$ in place of the Laplace noise gives exactly the exponential mechanism. Libraries sample it with this **Gumbel-max trick**, because the exponential of a score in the thousands overflows, but noisy scores do not. The $c$ largest scores with Gumbel noise give the top $c$ at once (Durfee & Rogers 2019).

## The sparse vector technique {#sec-sel-svt}

Sometimes queries come one at a time, and you want the first one that is high enough. The **sparse vector technique** (Dwork et al. 2009) compares each query with one noisy threshold and pays only for the query that fires. Its basic form, **AboveThreshold**, takes a threshold $\tau$ and a stream of queries $f$, all with the same sensitivity $\Delta f$. It divides the budget into $\varepsilon_1$ for the threshold and $\varepsilon_2$ for the queries.

1. Draw one threshold noise, written $\nu$, from $\mathrm{Lap}(\Delta f/\varepsilon_1)$ for the whole run.
2. For each query, draw a fresh noise from $\mathrm{Lap}(2\Delta f/\varepsilon_2)$ and add it to the answer of the query. If the result is at least $\tau + \nu$, output "above" and stop. If not, output "below" and continue.

AboveThreshold is $(\varepsilon_1 + \varepsilon_2)$-DP for any number of "below" answers, even when each query is chosen after the earlier answers. All the "below" answers together pay $\varepsilon_1$, through the one threshold noise, and the single "above" pays $\varepsilon_2$.

But the accuracy depends on the stream length $k$, which you must fix before the run. Write $\beta$ for the chance that the accuracy guarantee fails. With an equal split, $\varepsilon_1 = \varepsilon_2 = \varepsilon/2$, every query gets the correct answer unless its true value is within a margin
$$\alpha = \frac{8\Delta f\big(\ln k + \ln(2/\beta)\big)}{\varepsilon}$$ {#eq-sel-svt-alpha}
of the threshold. Monitor 1,000 counts, each with $\Delta f = 1$, against a threshold of 500 at $\varepsilon = 1$. At $\beta = 0.05$, the margin is 84.77. So with probability 0.95, only a count that near the threshold can get the wrong answer.

To find the first $c$ queries above the threshold, start AboveThreshold again after each "above", with fresh threshold noise and $\varepsilon/c$ per run. Stop after the $c$-th. The margin then grows with $c$, so ask for as few alarms as you can. Lyu, Su and Li keep one threshold for all $c$ alarms and multiply the query noise by $c$.

## When the sparse vector technique goes wrong {#sec-sel-svt-bugs}

Several published variants do not have the privacy they claim (@tbl-sel-svt-variants).

| Variant (as numbered by Lyu, Su & Li) | What it changes | Actual privacy |
| --- | --- | --- |
| Algorithm 3 | Outputs the noisy value of each "above" | $\infty$ |
| Algorithm 4 | Query noise does not grow with $c$ | $\frac{1 + 6c}{4}\varepsilon$ |
| Algorithm 5 | No noise on the queries | $\infty$ |
| Algorithm 6 | Never halts | $\infty$ |

: Variants of the sparse vector technique and their actual privacy, from Lyu, Su & Li (2017). {#tbl-sel-svt-variants}

The variant that never stops is the one to remember. It adds exactly as much noise per query as AboveThreshold, so a reviewer who checks noise scales does not find it. But at $\varepsilon = 1$, its privacy loss gets to 18.1246 on 80 queries.

So review the accounting, not only the noise. Draw one threshold noise per run. If you draw the threshold again for each query,
 every "below" answer costs budget that nobody counted. Draw fresh noise for every query, because one noise draw for all of them is the same as Algorithm 5. Stop after the permitted number of "above" answers. Never publish the noisy values that you compared.


## Cheat sheet

| Mechanism | Privacy | Accuracy |
| --- | --- | --- |
| Exponential mechanism, @eq-sel-em | $\varepsilon$; drop the 2 only for monotone scores | Mean gap $\le \frac{2\Delta u}{\varepsilon}(\ln|\mathcal R| + 1)$ |
| Report-noisy-max, noise $\mathrm{Lap}(2\Delta u/\varepsilon)$ | $\varepsilon$, index only | Similar to the exponential mechanism |
| AboveThreshold | $\varepsilon_1 + \varepsilon_2$ | Margin $8\Delta f(\ln k + \ln\frac2\beta)/\varepsilon$, failure probability $\beta$ |

: The selection mechanisms of this chapter, with $|\mathcal R|$ candidates or $k$ queries. {#tbl-sel-cheat}

**Rules of thumb**

- Release positions, not values.
- Fix the candidate list, and the range of any number you pick, before you look at the data.

## References

- Dwork & Roth, *The Algorithmic Foundations of Differential Privacy* (2014), §3.3, §3.4 and §3.6: the three mechanisms in their standard form.
- Lyu, Su & Li, "Understanding the Sparse Vector Technique for Differential Privacy" (VLDB 2017): the budget split and the survey of broken variants.

**Derivations.** Every result here is derived step by step in [the full edition's chapter](../../book-full/part03-dp-foundations/05-selection.md).
