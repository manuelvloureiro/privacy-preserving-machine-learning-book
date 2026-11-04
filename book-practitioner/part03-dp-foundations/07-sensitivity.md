# Beyond global sensitivity

## What it is {#sec-sens-local}

The [Laplace and Gaussian mechanisms](03-laplace-gaussian.md) calibrate noise to **global sensitivity**, the maximum change one record can cause across all neighbouring datasets. For ages bounded between 0 and 100, the median's global sensitivity is 100. Under differential privacy (DP) at $\varepsilon = 1$, the Laplace mechanism's median error is $100\ln 2 = 69.31$ years. On a simulated sample of 1,001 ages, however, one person changes the median by only 0.1386 years. This smaller dataset-specific value is **local sensitivity**, the largest change one record can cause to answer $f$ on dataset $D$:
$$\mathrm{LS}_f(D) = \max_{D' \text{ a neighbour of } D}|f(D) - f(D')|.$$

These techniques use the data's local scale under one condition: **a noise scale, bound or threshold computed from private data is itself a statistic and must also be private.** This chapter assumes fixed-size datasets and replace-one-record neighbours.

## When you meet it {#sec-sens-smooth}

Use these techniques when a statistic has no useful worst-case sensitivity (@tbl-sens-map).

| You release | Technique | Where you meet it |
| --- | --- | --- |
| A sum, mean or gradient of unbounded values | Clipping to a bound fixed in advance | Every library's `bounds`, and the clipping norm of private training |
| A median or quantile | Inverse sensitivity | `diffprivlib.tools.quantile`, OpenDP's `dp.quantile` |
| A discrete answer with a clear winner | Stability-based release | The threshold of private partition selection |
| A black-box model or estimator | Subsample-and-aggregate | Teacher ensembles for private prediction |

: Which technique releases what, and where you meet it. {#tbl-sens-map}

A **stability-based release** publishes a discrete answer, such as the most common value, when a noisy test shows that it exceeds the runner-up by about $2\ln(1/\delta)/\varepsilon$ records for slack $\delta$. Otherwise, it returns no answer; downstream steps must handle refusals. The same test selects keys for a group-by ([Release statistics with a library](12-release-statistics.md)).

**Subsample-and-aggregate** runs an estimator on disjoint blocks, clips their answers and releases a noisy average. Private aggregation of teacher ensembles (PATE) uses this method for voting ([Private prediction](../part05-dp-ml/05-private-prediction.md)).

**Propose-test-release** proposes a local-sensitivity bound and refuses to release unless a noisy test shows that the data are sufficiently far from the bound. **Smooth sensitivity** scales heavy-tailed noise to a bound that changes gradually between neighbours. PATE uses smooth sensitivity to calibrate its voting mechanism, but on the 1,001-age example it is about twelve times less accurate than inverse sensitivity.

## How to use it {#sec-sens-clip}

For unbounded inputs, use **clipping**. Replace each value $x$ by $\min(x,C)$, where clip bound $C$ is fixed independently of the data. If values are nonnegative, one record changes the mean of $n$ records by at most $C/n$. Laplace noise with scale $C/(n\varepsilon)$ therefore gives $\varepsilon$-DP for the clipped mean. A larger $C$ reduces clipping bias but increases noise.

Choose $C$ from public domain information, such as a legal maximum or sensor range. Otherwise, spend part of the budget on a private high quantile, account for it by [composition](02-properties.md), and clip at that quantile. Without a bound, diffprivlib uses the data to calculate one and raises `PrivacyLeakWarning`. Convert the warning to an error with `warnings.simplefilter("error", PrivacyLeakWarning)`, as in [practitioner_adult_statistics.py](../../code/part03-dp-foundations/practitioner_adult_statistics.py).

For a median or quantile, use **inverse sensitivity**. This [exponential mechanism](05-selection.md) favours candidate outputs that require fewer record changes to become the median. Set bounds or a candidate grid in advance. On the 1,001-age sample at $\varepsilon = 1$, median error is 0.0783 years, below the local sensitivity of 0.1386.

## What goes wrong

Noise calibrated to local sensitivity alone is not private. The five ages $(0, 0, 0, 0, 100)$ have median 0, and no single change moves it, so the mechanism adds no noise. A neighbouring dataset $(0, 0, 0, 100, 100)$ receives continuous noise and never outputs exactly 0; therefore, output 0 identifies the first dataset. Similar leaks arise from a clip bound based on the observed maximum, a candidate grid derived from private data or keys read from private data. During review, search for `max()`, `quantile()` and `unique()` calls on private data that set parameters.

Upper clipping biases a mean downward. Two-sided clipping removes that bias only when the range is centred on the data ([Private mean estimation](11-mean-estimation.md)). If neighbouring datasets differ by adding or removing a record, $n$ is not public. Release a noisy clipped sum and noisy count, then divide them.

**Derivations.** Every result here is derived step by step in [the full edition's chapter](../../book-full/part03-dp-foundations/07-sensitivity.md).
