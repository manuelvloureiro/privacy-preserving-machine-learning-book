# Beyond global sensitivity

## What it is {#sec-sens-local}

The [Laplace and Gaussian mechanisms](03-laplace-gaussian.md) set their noise by the **global sensitivity**, the most one record can move the answer over every pair of neighbouring datasets that could exist. The median of ages between 0 and 100 has global sensitivity 100. So under differential privacy (DP) at a privacy level of $\varepsilon = 1$, the Laplace mechanism's median error is $100\ln 2 = 69.31$ years. Yet on a simulated sample of 1,001 ages, one person moves the median by only 0.1386 years. That smaller number is the **local sensitivity**, the most one record can move the answer $f$ starting from the dataset $D$ you hold:
$$\mathrm{LS}_f(D) = \max_{D' \text{ a neighbour of } D}|f(D) - f(D')|.$$

The techniques of this page add noise on the scale of the data you hold, under one rule. **A noise scale, bound or threshold computed from the data is itself a statistic of the data, so it must be private too.** Throughout, datasets have a fixed size $n$, and neighbours replace one record.

## When you meet it {#sec-sens-smooth}

You meet these techniques whenever a statistic has no useful worst case (@tbl-sens-map).

| You release | Technique | Where you meet it |
| --- | --- | --- |
| A sum, mean or gradient of unbounded values | Clipping to a bound fixed in advance | Every library's `bounds`, and the clipping norm of private training |
| A median or quantile | Inverse sensitivity | `diffprivlib.tools.quantile`, OpenDP's `dp.quantile` |
| A discrete answer with a clear winner | Stability-based release | The threshold of private partition selection |
| A black-box model or estimator | Subsample-and-aggregate | Teacher ensembles for private prediction |

: Which technique releases what, and where you meet it. {#tbl-sens-map}

A **stability-based release** publishes a discrete answer, such as the most common value, exactly, but only when a noisy test shows that it leads the runner-up by about $2\ln(1/\delta)/\varepsilon$ records, for a slack $\delta$. Otherwise it refuses, so plan for refusals downstream. The same test decides which keys found in the data a group-by may release ([Release statistics with a library](12-release-statistics.md)).

**Subsample-and-aggregate** runs any estimator on disjoint blocks fixed in advance, clips each block's answer and averages them with noise. Private aggregation of teacher ensembles (PATE) is its voting form ([Private prediction](../part05-dp-ml/05-private-prediction.md)).

**Propose-test-release** proposes a bound on the local sensitivity, checks with noise that the data are many records away from breaking it, and refuses otherwise. **Smooth sensitivity** scales heavy-tailed noise to a bound that cannot jump between neighbours. PATE uses it to publish the $\varepsilon$ of its votes, but on the 1,001 ages it is about twelve times less accurate than inverse sensitivity (below).

## How to use it {#sec-sens-clip}

Most of the time you bound the inputs. **Clipping** replaces each value $x$ by $\min(x, C)$, for a clip bound $C$ fixed before seeing the data. For values that cannot be negative, one record then moves the sum by at most $C$ and the mean of $n$ records by at most $C/n$. Laplace noise of scale $C/(n\varepsilon)$ on the clipped mean gives $\varepsilon$-DP, and raising $C$ shrinks the bias but grows the noise.

Choose $C$ from domain knowledge, such as a legal maximum or a sensor range. Failing that, assume a distribution in advance and take the $C$ that minimizes bias plus noise under it, or spend a small slice of the budget, charged by [composition](02-properties.md), on a private high quantile and clip there. If you omit the bound, diffprivlib computes it from the data and issues a `PrivacyLeakWarning`. So turn that warning into an error with `warnings.simplefilter("error", PrivacyLeakWarning)`, as [practitioner_adult_statistics.py](../../code/part03-dp-foundations/practitioner_adult_statistics.py) does. Bounding the rows each person contributes is the same move for a whole person ([Release statistics with a library](12-release-statistics.md)).

For a median or quantile, use **inverse sensitivity**, which favours each candidate output by how few records must change before the median equals it. That is the [exponential mechanism](05-selection.md), with pure $\varepsilon$-DP, so pass it bounds or a candidate grid fixed in advance. On the 1,001 ages at $\varepsilon = 1$, its median error is 0.0783 years, below even the local sensitivity of 0.1386.

## What goes wrong

Noise scaled to the local sensitivity is not private. The five ages $(0, 0, 0, 0, 100)$ have median 0, and no single change moves it, so they get no noise. Their neighbour $(0, 0, 0, 100, 100)$ has a positive local sensitivity, so it gets continuous noise and its output is never exactly 0. An output of exactly 0 therefore names the first dataset. The same leak hides in a clip bound set to the observed maximum, a candidate grid built from the data's values, or keys read off the data. In review, look for a `max()`, `quantile()` or `unique()` on the private data that feeds a parameter.

Clipping from above always biases a mean downward, and a two-sided clip cancels the bias only when the range is centred on the data ([Private mean estimation](11-mean-estimation.md)). If neighbours add or remove a record, $n$ is not public, so release a noisy clipped sum and a noisy count and divide them.

**Derivations.** Every result here is derived step by step in [the full edition's chapter](../../book-full/part03-dp-foundations/07-sensitivity.md).
