# Selection: exponential mechanism, report-noisy-max, sparse vector

## Why this matters

Many questions ask you to pick something rather than compute a number, such as the best model on a validation set, a median, or the first alarm among many statistics. Publishing every candidate's score with Laplace noise would charge the budget once per candidate, because the budgets of separate releases add. The mechanisms of this chapter instead **release the position, not the values**. One record can change which candidate wins only a little, however many candidates there are. So under differential privacy (DP), a selection costs as much as a single query, and the number of candidates enters only the accuracy, through its logarithm.

## Which mechanism to use {#sec-sel-choose}

@tbl-sel-choose matches common tasks to the mechanisms below. Throughout, the **sensitivity** $\Delta u$ is the most that one person's record can change any candidate's score.

| Task | Mechanism | Privacy cost |
| --- | --- | --- |
| Pick the best of a fixed list | Exponential mechanism or report-noisy-max | $\varepsilon$ |
| Pick the top $c$, such as the most frequent items | Gumbel noise, keep the $c$ largest, or noisy max $c$ times | $c\varepsilon$ |
| Pick a number from a range, such as a quantile or a clipping norm | Exponential mechanism over the range | $\varepsilon$ |
| Find the first query in a stream above a threshold | AboveThreshold | $\varepsilon$, however long the stream |

: Which selection mechanism to use. {#tbl-sel-choose}

## The exponential mechanism {#sec-sel-exp}

The candidates form a fixed set, written $\mathcal R$. Each candidate $r$ has a **score** on the dataset $D$, written $u(D, r)$, such as the number of validation records a model gets right. The **exponential mechanism** of McSherry and Talwar (2007) picks each candidate with probability proportional to an exponential of its score. With a privacy budget $\varepsilon$, it outputs candidate $r$ with probability
$$\Pr[M_E(D) = r] = \frac{\exp\big(\varepsilon\,u(D, r)/(2\Delta u)\big)}{\sum_s\exp\big(\varepsilon\,u(D, s)/(2\Delta u)\big)}.$$ {#eq-sel-em}
It is $\varepsilon$-DP because one record changes each weight, and also their sum, by at most a factor $e^{\varepsilon/2}$, which makes $e^\varepsilon$ for their ratio.

Two conditions decide whether a use is correct. First, fix the list of candidates before you look at the data, because a list built from the data reveals records by itself, as a word in a corpus's vocabulary does when only one person used it. Second, drop the 2 in the exponent only when the scores are **monotone**, meaning that one record pushes all of them the same way. Counts under add/remove neighbours are monotone. A histogram under replace neighbours is not, because moving one record lowers one count and raises another, so dropping the 2 there doubles the real cost.

## Accuracy, and the cost of many candidates {#sec-sel-utility}

The number of candidates $|\mathcal R|$ is free for privacy but not for accuracy, because more candidates means more bad ones to pick. Call the best score minus the chosen candidate's score the **gap**. On average it is at most
$$\frac{2\Delta u}{\varepsilon}(\ln|\mathcal R| + 1),$$ {#eq-sel-em-utility}
so doubling the candidates adds only $\ln 2$ to the bracket.

Suppose you trained 50 models and want to release the best one, judged on a private validation set of 1,000 records. Model $r$, numbered from 1, gets $900 - r$ records right, so model 1 is the best. The score is that count, which one record moves by at most 1, so $\Delta u = 1$. At $\varepsilon = 1$ the mechanism picks the best model only 39% of the time. Yet it loses just 1.54 correct predictions on average, or 0.15% accuracy, because its mistakes are mostly near-ties. [Hyperparameter tuning](../part05-dp-ml/03-hyperparameter-tuning.md) picks models this way.

## A private quantile or clipping norm {#sec-sel-apps}

Reading a median or a clipping norm off the data leaks it, so pick it with the exponential mechanism. First fix a range before looking at the data, and clip every record into it. The mechanism then scores each point of the range by how close the share of records below it is to the quantile you want, such as one half for the median, and one added or removed record moves any score by at most 1. The score stays the same between consecutive sorted records. So the mechanism picks one of these intervals, called **pieces**, favouring long pieces near the quantile, and returns a uniform point inside it.

diffprivlib's `tools.quantile` and `tools.median` take an `epsilon` and the range as `bounds`. OpenDP's `.dp.quantile` and `.dp.median`, which work on Polars data frames, pick one value from a finite list of candidates instead. The script [practitioner_adult_statistics.py](../../code/part03-dp-foundations/practitioner_adult_statistics.py) takes the median age of the Adult census data, 48,842 people with one row each, at $\varepsilon = 1$. The range of 17 to 90 comes from the survey's documentation.

```{.python excerpt="adult-median-diffprivlib"}
age = adult["age"].to_numpy()
private = tools.median(age, epsilon=1.0, bounds=(17, 90), random_state=seed)
print(f"true median age: {np.median(age):.1f}  diffprivlib median: {private:.2f}")
```

With `random_state=0` it prints a true median of 37.0 and a private one of 37.72. Ages are whole numbers, so every point between 37 and 38 has the same score, and the mechanism picked that piece and a point in it at random. OpenDP refuses to run the median without a public upper bound on the number of rows, declared as a `Margin`. Choose it generously from public facts, since it must be at least the true number of rows and a larger one costs almost nothing.

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

Its summary reports `ExponentialMin` with a scale of 2.0, because OpenDP 0.16 adds exponential noise to every candidate's score and reports the best. That is **permute-and-flip** (McKenna & Sheldon 2020), with the exponential mechanism's guarantee and never a worse expected error. The accuracy is `null`, because OpenDP gives no error bar for a quantile. Its noise has no seed, so the released value changes from run to run, and `uv run python code/part03-dp-foundations/practitioner_adult_statistics.py` prints one at the end.

The error bar you can state is about rank, not value. With probability $1 - \beta$, the exponential mechanism's gap is at most $(2\Delta u/\varepsilon)(\ln|\mathcal R| + \ln(1/\beta))$, the tail form of @eq-sel-em-utility. For a median, with $\Delta u = 1$, the gap is the number of records by which the output misses the middle rank, so $\beta = 0.05$ gives its 95% bar in ranks. A bar in value would need the number of records near the median, which the release does not reveal. If a dashboard needs values, spend budget on a few noisy counts below candidate values.

A generous range is cheap, because its width enters the gap only through the logarithm of the number of candidates.

Two things go wrong in practice. Where the data near the quantile are sparse, the best piece is long and the output can land anywhere in it. A small error in rank is then a large error in value. And the range or the candidates must not come from the data, since setting the upper end to the observed maximum reintroduces the leak. diffprivlib warns with a `PrivacyLeakWarning` when you leave out `bounds` ([Sensitivity](07-sensitivity.md)).

## Report-noisy-max and Gumbel noise {#sec-sel-rnm}

**Report-noisy-max** adds independent Laplace noise to each candidate's score and outputs the candidate with the largest noisy score. It is $\varepsilon$-DP for any number of candidates when the noise has scale $2\Delta u/\varepsilon$, or $\Delta u/\varepsilon$ when one record pushes all scores the same way. Release only the winner's index, and pay for a fresh noisy answer if you need its score. Its accuracy is close to the exponential mechanism's, and neither is always better.

Gumbel noise of scale $2\Delta u/\varepsilon$ in place of the Laplace noise gives exactly the exponential mechanism. Libraries sample it with this **Gumbel-max trick**, because the exponential of a score in the thousands overflows, while noisy scores do not. Keeping the $c$ largest Gumbel-noised scores picks the top $c$ at once (Durfee & Rogers 2019).

## The sparse vector technique {#sec-sel-svt}

Sometimes queries arrive one at a time, and you want the first one that is high enough. The **sparse vector technique** (Dwork et al. 2009) compares each with one noisy threshold and pays only for the one that fires. Its basic form, **AboveThreshold**, takes a threshold $\tau$ and a stream of queries $f$, all with the same sensitivity $\Delta f$. It splits the budget into $\varepsilon_1$ for the threshold and $\varepsilon_2$ for the queries.

1. Draw one threshold noise, written $\nu$, from $\mathrm{Lap}(\Delta f/\varepsilon_1)$ for the whole run.
2. For each query, draw a fresh noise from $\mathrm{Lap}(2\Delta f/\varepsilon_2)$ and add it to the query's answer. If the result is at least $\tau + \nu$, output "above" and halt. Otherwise output "below" and go on.

AboveThreshold is $(\varepsilon_1 + \varepsilon_2)$-DP for any number of "below" answers, even when each query is chosen after the earlier answers. All the "below" answers together pay $\varepsilon_1$, through the one threshold noise, and the single "above" pays $\varepsilon_2$.

The accuracy, unlike the privacy, depends on the stream length $k$, which must be fixed before the run. Write $\beta$ for the chance that the accuracy guarantee fails. With an equal split, $\varepsilon_1 = \varepsilon_2 = \varepsilon/2$, every query is classified correctly unless its true value is within a margin
$$\alpha = \frac{8\Delta f\big(\ln k + \ln(2/\beta)\big)}{\varepsilon}$$ {#eq-sel-svt-alpha}
of the threshold. Monitor 1,000 counts, each with $\Delta f = 1$, against a threshold of 500 at $\varepsilon = 1$. At $\beta = 0.05$ the margin is 84.77, so with probability 0.95 only a count that close to the threshold can be misclassified.

To find the first $c$ queries above the threshold, restart AboveThreshold with fresh threshold noise after each "above", at $\varepsilon/c$ per run, and halt after the $c$-th. The margin then grows with $c$, so ask for as few alarms as you can. Lyu, Su and Li instead keep one threshold for all $c$ alarms and multiply the query noise by $c$.

## When the sparse vector technique goes wrong {#sec-sel-svt-bugs}

Several published variants do not have the privacy they claim (@tbl-sel-svt-variants).

| Variant (as numbered by Lyu, Su & Li) | What it changes | Actual privacy |
| --- | --- | --- |
| Algorithm 3 | Outputs the noisy value of each "above" | $\infty$ |
| Algorithm 4 | Query noise does not grow with $c$ | $\frac{1 + 6c}{4}\varepsilon$ |
| Algorithm 5 | No noise on the queries | $\infty$ |
| Algorithm 6 | Never halts | $\infty$ |

: Variants of the sparse vector technique and their actual privacy, from Lyu, Su & Li (2017). {#tbl-sel-svt-variants}

The no-halt variant is the one to remember. It adds exactly as much noise per query as AboveThreshold, so a reviewer who checks noise scales will miss it. Yet at $\varepsilon = 1$ its privacy loss reaches 18.1246 on 80 queries.

So review the accounting, not just the noise. An implementation must draw one threshold noise per run, since a threshold redrawn for each query makes every "below" answer cost budget that nobody counted. It must draw fresh noise for every query, since one query-noise draw reused for all of them amounts to Algorithm 5. And it must halt after the allowed number of "above" answers and never publish the noisy values it compared.

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
