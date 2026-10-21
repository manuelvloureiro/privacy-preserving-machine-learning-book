# Selection: exponential mechanism, report-noisy-max, sparse vector

## Why this matters

Many questions about a dataset ask you to pick something rather than to compute a number. You might want the model that scores best on a validation set, a clipping bound for private training, or the first alarm among a thousand statistics.

Adding Laplace noise to every score and publishing them all would cost one release per candidate. The mechanisms of this chapter follow one rule instead, which is to **release the position, not the values**. Each one publishes only which candidate won, and one record can change that only a little, however many candidates there are. So under differential privacy (DP), a selection costs as much as a single query, and the number of candidates enters only the accuracy, through its logarithm.

## Which mechanism to use {#sec-sel-choose}

@tbl-sel-choose matches common tasks to the mechanisms of this chapter. Throughout, the **sensitivity** $\Delta$ is the most that one person's record can change any candidate's score.

| Task | Mechanism | Privacy cost |
| --- | --- | --- |
| Pick the best of a fixed list | Exponential mechanism, sampled with Gumbel noise | $\varepsilon$ |
| Pick the top $c$, such as the most frequent items | Gumbel noise, keep the $c$ largest, or noisy max $c$ times | $c\varepsilon$ |
| Pick a number from a range, such as a quantile or a clipping bound | Exponential mechanism, piece by piece | $\varepsilon$ |
| Find the first query in a stream above a threshold | AboveThreshold | $\varepsilon$, however long the stream |
| Find the first $c$ such queries | AboveThreshold restarted $c$ times | $\varepsilon$, with $\varepsilon/c$ per run |

: Which selection mechanism to use. {#tbl-sel-choose}

## The exponential mechanism {#sec-sel-exp}

Each candidate $r$ has a **score** $u(D, r)$ on the dataset $D$, such as the number of validation examples a model gets right. For a split in a private decision tree, it can be how well the split separates the labels. The **exponential mechanism** of McSherry and Talwar (2007) picks each candidate with probability proportional to an exponential of its score. With a privacy budget $\varepsilon$, it outputs candidate $r$ with probability
$$\Pr[M_E(D) = r] = \frac{\exp\big(\varepsilon\,u(D, r)/(2\Delta)\big)}{\sum_s\exp\big(\varepsilon\,u(D, s)/(2\Delta)\big)}.$$ {#eq-sel-em}
It is $\varepsilon$-DP because one record shifts each weight, and their sum, by at most a factor $e^{\varepsilon/2}$.

Two conditions decide whether a use is correct. First, fix the list of candidates before you look at the data, because a list built from the data, such as a corpus's vocabulary, reveals records by itself. Second, drop the 2 in the exponent only when the scores are **monotone**, meaning that one record pushes all of them the same way. Counts under add/remove neighbours are monotone, but a histogram under replace neighbours is not, because moving one record lowers one count and raises another, and dropping the 2 there doubles the real cost.

## Accuracy, and the cost of many candidates {#sec-sel-utility}

The number of candidates $k$ is free for privacy but not for accuracy, because more candidates means more bad ones to pick by accident. Write $\mathrm{OPT}(D)$ for the best score, and call the best score minus the chosen candidate's score the **gap**. For any $t > 0$, the gap stays below a threshold except with probability $e^{-t}$:
$$\Pr\Big[\mathrm{OPT}(D) - u(D, M_E(D)) \ge \frac{2\Delta}{\varepsilon}(\ln k + t)\Big] \le e^{-t}.$$ {#eq-sel-em-utility}
On average the gap is at most $\frac{2\Delta}{\varepsilon}(\ln k + 1)$, so doubling the candidates adds only $\ln 2$ to the bracket.

For example, suppose you trained 50 models and want to release the best one, judged on a private validation set of 1,000 examples. Model $r$ gets $900 - r$ examples right, and the score is that count, so $\Delta = 1$. At $\varepsilon = 1$ the mechanism picks the best model only 39% of the time. Yet it loses just 1.54 correct predictions on average, or 0.15% accuracy, because its mistakes are mostly near-ties.

Scoring each classifier in a finite class by minus its training mistakes makes the mechanism a private learner, which [Hyperparameter tuning](../part05-dp-ml/03-hyperparameter-tuning.md) uses.

## A private quantile or clipping bound {#sec-sel-apps}

A common clipping bound for private training is the median of the gradient norms, and reading it off the data leaks it, so pick it with the exponential mechanism. First fix a range from 0 to an upper end $B$ before looking at the data, and clip every record into it. For the quantile $q$, such as the median at $q = 0.5$, score each point $r$ of the range by
$$u(D, r) = -\big|(1 - q)\,\#\{x_i < r\} - q\,\#\{x_i > r\}\big|,$$
where $\#\{x_i < r\}$ counts the records below $r$. One added or removed record moves this score by at most the larger of $q$ and $1 - q$, which is its sensitivity. The score is constant between consecutive sorted records, so the range splits into **pieces**. Pick a piece with probability proportional to its length times the weight of @eq-sel-em, and return a uniform point inside it.

For the median, the guarantee bounds the **imbalance** of the output, the gap between the numbers of records on its two sides. Except with probability $e^{-t}$, it is at most $\frac2\varepsilon(\ln(B/L^\star) + t)$, where $L^\star$ is the total length of the best pieces. For example, take 1,000 lognormal gradient norms capped at $B = 10$, whose median is 0.9804. At $\varepsilon = 1$ the private median lands on average 0.0013 from it, and widening the range to $B = 1000$ leaves that unchanged, because $B$ enters only through a logarithm. IBM's diffprivlib provides this as `diffprivlib.tools.quantile` and `diffprivlib.tools.median`, which take an `epsilon` and the range as `bounds`. Tumult Analytics has the same aggregations on its `QueryBuilder`. For a general list of candidates, diffprivlib has the `mechanisms.Exponential` class.

Two things go wrong in practice. The guarantee bounds the output's rank, not its value, so where the data near the quantile are sparse the output can land anywhere in a long piece. And the range must not come from the data, because setting $B$ to the observed maximum reintroduces the leak.

## Report-noisy-max and Gumbel noise {#sec-sel-rnm}

**Report-noisy-max** adds independent Laplace noise to each candidate's score, and outputs the candidate with the largest noisy score. With noise of scale $2\Delta/\varepsilon$ it is $\varepsilon$-DP for any number of candidates, and with monotone scores a scale of $\Delta/\varepsilon$ is enough. Release only the index of the winner, and spend extra budget on a fresh Laplace answer if you need its score.

On the 50 models at $\varepsilon = 1$, it picks the best with probability 0.4241, against 0.3935 for the exponential mechanism, but neither mechanism is always better.

Gumbel noise of scale $2\Delta/\varepsilon$ in place of the Laplace noise gives exactly the exponential mechanism. This **Gumbel-max trick** is the stable way to sample the mechanism, because computing the weights overflows for scores in the thousands. Keeping the $c$ largest Gumbel-noised scores picks the top $c$ in one shot, at the cost of $c$ exponential mechanisms (Durfee & Rogers 2019).

## The sparse vector technique {#sec-sel-svt}

Now the queries arrive one at a time, and you want the first one that is high enough. The **sparse vector technique** (Dwork et al. 2009) compares each query with one noisy threshold and pays only for the query that fires. Its basic form, **AboveThreshold**, takes a threshold $T$ and a stream of queries that share a sensitivity $\Delta$. It splits the budget into $\varepsilon_1$ for the threshold and $\varepsilon_2$ for the queries.

1. Draw one threshold noise, written $\rho$, from $\mathrm{Lap}(\Delta/\varepsilon_1)$ for the whole run.
2. For each query, draw a fresh noise from $\mathrm{Lap}(2\Delta/\varepsilon_2)$ and add it to the query's answer. If the result is at least $T + \rho$, output "above" and halt. Otherwise output "below" and go on.

AboveThreshold is $(\varepsilon_1 + \varepsilon_2)$-DP for any number of "below" answers, even when each query is chosen after the earlier answers.

The accuracy does depend on the stream length $k$, which must be fixed before the run. With an equal split and failure probability $\beta$, every query is classified correctly unless its true value is within a margin
$$\alpha = \frac{8\Delta\big(\ln k + \ln(2/\beta)\big)}{\varepsilon}$$ {#eq-sel-svt-alpha}
of the threshold. For example, monitor 1,000 counts against $T = 500$, at $\varepsilon = 1$. With $\beta = 0.05$ the margin is 84.77, so with probability 0.95 only counts within 84.77 of 500 can be misclassified.

To find the first $c$ queries above the threshold, restart AboveThreshold with fresh threshold noise after each "above", at $\varepsilon/c$ per run, and halt after the $c$-th. The margin then grows with $c$, so ask for as few alarms as you can. The split between $\varepsilon_1$ and $\varepsilon_2$ affects only accuracy, and Lyu, Su and Li recommend $\varepsilon_1 : \varepsilon_2 = 1 : (2c)^{2/3}$.

## When the sparse vector technique goes wrong {#sec-sel-svt-bugs}

Several published variants are not private (@tbl-sel-svt-variants).

| Variant (as numbered by Lyu, Su & Li) | What it changes | Actual privacy |
| --- | --- | --- |
| Algorithm 3 | Outputs the noisy value of each "above" | $\infty$ |
| Algorithm 4 | Query noise does not grow with $c$ | $\frac{1 + 6c}{4}\varepsilon$ |
| Algorithm 5 | No noise on the queries | $\infty$ |
| Algorithm 6 | Never halts | $\infty$ |

: Variants of the sparse vector technique and their actual privacy, from Lyu, Su & Li (2017). {#tbl-sel-svt-variants}

The no-halt variant is the one to remember. It adds exactly as much noise per query as AboveThreshold, so a reviewer who checks noise scales will miss it. Yet at $\varepsilon = 1$, on neighbours with 40 low and then 40 high queries, its privacy loss is 18.1246.

So review the accounting, not just the noise. An implementation must draw fresh noise for every query and one threshold noise per run. A threshold redrawn for each query makes every "below" answer cost budget that nobody counted. It must halt after the allowed number of "above" answers, and it must never publish the noisy values it compared. Reusing one noise draw for all queries quietly turns it into Algorithm 5.

## Code {#sec-sel-code}

The snippet picks a private quantile piece by piece, as a clipping bound would.

```python
import random
from math import exp, log
from typing import Sequence


def private_quantile(data: Sequence[float], q: float, upper: float, eps: float, rng: random.Random) -> float:
    """Exponential mechanism for the q-quantile over [0, upper], add/remove neighbours."""
    sensitivity = max(q, 1 - q)
    xs = sorted(min(max(x, 0.0), upper) for x in data)  # clip into the range fixed in advance
    edges = [0.0, *xs, upper]
    n = len(xs)
    logw = [  # piece j has j records below it and n - j above; empty pieces get no weight
        log(edges[j + 1] - edges[j]) - eps * abs((1 - q) * j - q * (n - j)) / (2 * sensitivity)
        if edges[j + 1] > edges[j]
        else -1e300
        for j in range(n + 1)
    ]
    top = max(logw)
    j = rng.choices(range(n + 1), weights=[exp(w - top) for w in logw])[0]
    return edges[j] + rng.random() * (edges[j + 1] - edges[j])


gen = random.Random(0)
norms = [gen.lognormvariate(0.0, 0.75) for _ in range(1000)]  # gradient norms, median about 1
print(f"{sorted(norms)[499]:.4f} {sorted(norms)[500]:.4f}")  # 1.0273 1.0296
print(f"{private_quantile(norms, 0.5, 10.0, 1.0, random.Random(1)):.4f}")  # 1.0292
```

The private median lands between the two middle norms. A floating-point sampler like this one is fine for checking a calculation but not for production, as [Discrete and floating-point-safe mechanisms](08-discrete-mechanisms.md) explains.

## Cheat sheet

| Mechanism | Privacy | Accuracy |
| --- | --- | --- |
| Exponential mechanism, @eq-sel-em | $\varepsilon$; drop the 2 only for monotone scores | Mean gap $\le \frac{2\Delta}{\varepsilon}(\ln k + 1)$ |
| Report-noisy-max, noise $\mathrm{Lap}(2\Delta/\varepsilon)$ | $\varepsilon$, index only | Similar to the exponential mechanism |
| AboveThreshold | $\varepsilon_1 + \varepsilon_2$ | Margin $8\Delta(\ln k + \ln\frac2\beta)/\varepsilon$, failure probability $\beta$ |

: The selection mechanisms of this chapter, with $k$ candidates or queries. {#tbl-sel-cheat}

**Rules of thumb**

- Release positions, not values, because every released noisy value needs its own budget.
- Fix the candidate list, and the range of any number you pick, before you look at the data.
- Keep the factor $2\Delta$ unless the scores are monotone under your neighbouring relation.
- A sparse vector implementation draws fresh noise per query and one threshold noise per run, halts, and publishes nothing it compared.

## Further reading

- Dwork & Roth, *The Algorithmic Foundations of Differential Privacy* (2014), §3.3, §3.4 and §3.6: the three mechanisms in their standard form.
- Lyu, Su & Li, "Understanding the Sparse Vector Technique for Differential Privacy" (VLDB 2017): the budget split and the survey of broken variants.

**Derivations.** Every result here is derived step by step in [the full edition's chapter](../../book-full/part03-dp-foundations/05-selection.md).
