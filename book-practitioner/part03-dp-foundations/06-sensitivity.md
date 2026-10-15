# Beyond global sensitivity

## Why this matters

The [Laplace and Gaussian mechanisms](03-laplace-gaussian.md) set their noise by the **global sensitivity**: the most one record can move the answer, over every pair of neighbouring datasets that could exist. For counts and clipped sums that is the right scale. For many other statistics it is not. The median of ages in $[0, 100]$ has global sensitivity 100, because a dataset of half zeros and half hundreds lets one person swing it end to end. The mean of unbounded data has infinite sensitivity, and for an estimator fitted by an optimizer nobody knows the sensitivity at all. Yet on a typical sample of 1,001 ages, one person moves the median by about a tenth of a year.

Can you add noise on the scale of your data, not the worst conceivable dataset, and keep a differential privacy (DP) guarantee? Datasets have a fixed size $n$, and neighbours differ by one **replaced** record. The one idea: **the noise scale is itself a function of the data, so it must be private too.** Each technique below makes the scale safe in a different way (@tbl-sens-overview).

| Your problem | Technique | Guarantee | Section |
| --- | --- | --- | --- |
| Tempted to scale noise to "this dataset's" sensitivity | Don't: it leaks | None | @sec-sens-local |
| Median, quantiles, robust statistics with a known formula | Smooth sensitivity | $\varepsilon$ (Cauchy noise) or $(\varepsilon, \delta)$ (Laplace noise) | @sec-sens-smooth |
| A statistic that is stable on most real data | Propose-test-release | $(2\varepsilon, \delta)$, may refuse to answer | @sec-sens-ptr |
| A black-box estimator or model | Subsample-and-aggregate | $\varepsilon$ | @sec-sens-saa |
| Means and sums of unbounded data | Clipping | $\varepsilon$, with a bias | @sec-sens-clip |

: The problems that global sensitivity handles badly, and the technique for each. {#tbl-sens-overview}

## Local sensitivity, and why it leaks {#sec-sens-local}

**Idea.** The obvious fix is to take the maximum only over the neighbours of the dataset you hold. That **local sensitivity** is often hundreds of times smaller than the global one. But noise scaled to it is not private: the local sensitivity can jump when one record changes, and the noise level then tells two neighbours apart even when their answers agree.

**Result.** The local sensitivity of $f$ at $D$ is the largest change one record can make starting from $D$:
$$\mathrm{LS}_f(D) = \max_{D':\ d(D, D') = 1}|f(D) - f(D')|,$$
where $d(D, D')$, the **Hamming distance**, counts the records in which two datasets differ. For the median of $n$ sorted values $x_1 \le \dots \le x_n$ ($n$ odd, median $x_m$), one change moves it at most one place, so $\mathrm{LS} = \max(x_m - x_{m-1},\ x_{m+1} - x_m)$, the larger gap next to it.

**Example.** Take ages in $[0, 100]$, $\varepsilon = 1$, and the two neighbours of @tbl-sens-leak. Both have median 30, but their local sensitivities are 10 and 60.

| Dataset | Values | Median | $\mathrm{LS}$ |
| --- | --- | --- | --- |
| $D$ | 10, 20, 30, 31, 90 | 30 | 10 |
| $D'$ (31 replaced by 90) | 10, 20, 30, 90, 90 | 30 | 60 |

: Two neighbouring datasets with the same median and very different local sensitivities. {#tbl-sens-leak}

The releases $30 + \mathrm{Lap}(10)$ and $30 + \mathrm{Lap}(60)$ have the same centre, but the narrow one falls off six times faster. The privacy loss grows without bound and exceeds 1 once the output is more than 33.50 from 30, which happens with probability 0.572 under $D'$ and 0.0351 under $D$. To be $(1, \delta)$-DP this pair would need $\delta = 0.477$, all of it from the spread of the noise (@fig-sens-local-leak).

![Left: the median of $D$ and of $D'$ released with Laplace noise scaled to their own local sensitivities, 10 and 60. Right: the privacy loss grows linearly with the distance from 30 and crosses $\varepsilon = 1$ at distance 33.5, so no $\varepsilon$ bounds it.](figures/sens_local_leak.svg){#fig-sens-local-leak}

**Pitfalls.** The same leak hides in any data-dependent scale: a clip bound set to the observed maximum, a noise level set from the empirical spread. Ask of every number that sets a noise scale whether it was computed from the private data. If it was, it needs one of the repairs below or its own share of the budget.

## Smooth sensitivity {#sec-sens-smooth}

**Idea.** Replace the local sensitivity by an upper bound that cannot jump: one that changes by at most a factor $e^\beta$ between neighbours. Rescaling the noise by a bounded factor shifts its log-density by a bounded amount, which is what a privacy loss measures. Nissim, Raskhodnikova and Smith (2007) found the smallest such bound and the noise it can be used with.

**Result.** The **$\beta$-smooth sensitivity** discounts the local sensitivity of every other dataset by $e^{-\beta}$ per record of distance from yours, and takes the largest:
$$S^*_\beta(D) = \max_{D''}\ \mathrm{LS}_f(D'')\,e^{-\beta\,d(D, D'')}.$$ {#eq-sens-smooth}
Grouping datasets by distance makes it computable: $S^*_\beta(D) = \max_k e^{-k\beta}A^{(k)}(D)$, where $A^{(k)}$ is the largest local sensitivity reachable by changing $k$ records. For the median, $A^{(k)}$ is the widest gap between order statistics $k + 1$ ranks apart that sits next to the median, $A^{(k)} = \max_{0 \le t \le k + 1}(x_{m+t} - x_{m+t-k-1})$, with ranks below 1 read as 0 and above $n$ as the upper bound $\Lambda$. That is an exact $O(n^2)$ computation (@sec-sens-code).

The release stretches noise of a fixed shape $Z$ by the smooth bound: $f(D) + (S^*_\beta(D)/\alpha)\,Z$, where $\alpha$ sets how many noise scales one record's shift is worth. **Pure DP needs heavy tails:** with $Z$ standard Cauchy and $\alpha = \beta = \varepsilon/6$, releasing $f(D) + \frac{6}{\varepsilon}S^*_{\varepsilon/6}(D)\cdot Z$ is $\varepsilon$-DP. **Laplace noise needs a $\delta$:** with $Z \sim \mathrm{Lap}(1)$, the release is $(\varepsilon, \delta)$-DP whenever $\beta + e^\beta\alpha + (e^\beta - 1)\ln(1/\delta) \le \varepsilon$. The last term pays for rare large draws and forces $\beta$ down to roughly $\varepsilon/(2\ln(1/\delta))$. The split exists because a light-tailed density and a stretched copy of it have an unbounded log-ratio far in the tail, as in @tbl-sens-leak.

**Example.** The full edition's script draws $n = 1001$ ages and releases their median, 36.96, at $\varepsilon = 1$, comparing median absolute errors. Global sensitivity gives 69.31 years: useless. The local sensitivity is 0.1386, but noise scaled to it is not private. Cauchy noise has $S^*_{1/6} = 0.1632$ and error 0.979 (simulation: 0.980). Laplace noise at $\delta = 10^{-6}$ and $\alpha = 1/2$ allows $\beta = 0.0322$, so $S^* = 0.5144$ and the error is 0.713. Both are 70 to 100 times more accurate than the global release, and the gain grows with $n$ (@fig-sens-methods-vs-n).

**Pitfalls.** The Cauchy has no mean and a heavy tail: about one release in 16 is off by more than ten times the typical error, so don't feed Cauchy-noised values into averages. For $n \lesssim 20$ smooth sensitivity loses to global sensitivity, because a few changes open a gap across nearly the whole range. Exact formulas for $A^{(k)}$ exist for only a few statistics; a bound on $S^*$ works only if it is itself $\beta$-smooth.

## Propose-test-release {#sec-sens-ptr}

**Idea.** When no smooth-sensitivity formula exists, Dwork and Lei (2009) take a cruder route. Propose a bound $b$ on the local sensitivity before looking at the data, test privately that the data are many records away from any dataset where $b$ is too small, and release with noise of scale $b/\varepsilon$ only if the test passes. The test is private because one record moves that distance by at most 1.

**Result.** Let $\operatorname{dist}_b(D)$ be the number of records you must change before the local sensitivity can exceed $b$. **Propose-test-release** outputs $\bot$ ("no answer") if $\operatorname{dist}_b(D) + \mathrm{Lap}(1/\varepsilon) \le \ln(1/\delta)/\varepsilon$, and $f(D) + \mathrm{Lap}(b/\varepsilon)$ otherwise. It is $(2\varepsilon, \delta)$-DP: one $\varepsilon$ for the test, one for the release, and $\delta$ for a lucky pass on a dataset where $b$ is wrong. For the median, $\operatorname{dist}_b(D)$ is the first $k$ with $A^{(k)}(D) > b$.

**Example.** On the 1,001 ages, propose $b = 1$ year with $\varepsilon = 1$ and $\delta = 10^{-6}$. The threshold is $\ln(10^6) = 13.82$ and the distance is 21, so the test passes with probability 0.999621 and the release has median error 0.693 years, for a total of $2\varepsilon = 2$. On 101 ages the distance is 1, and the test passes with probability $1.36\times10^{-6}$ (@fig-sens-methods-vs-n, right).

![Left: median absolute noise when releasing the median age at $\varepsilon = 1$, by global sensitivity, smooth sensitivity with Cauchy noise and with Laplace noise ($\delta = 10^{-6}$), and propose-test-release with $b = 1$ when it passes, as the number of people grows. The dotted line is the non-private local sensitivity. Right: the probability that propose-test-release with $b = 1$ answers.](figures/sens_methods_vs_n.svg){#fig-sens-methods-vs-n}

**Pitfalls.** Fix $b$ before you look at the data; choosing it from the data brings back the leak of @sec-sens-local. The test passes reliably only when the data are stable against more than a dozen changes. Plan for $\bot$: it is a legitimate private answer, and every retry with a larger $b$ spends budget again.

## Subsample-and-aggregate {#sec-sens-saa}

**Idea.** If you cannot analyse $f$ at all, run it on disjoint blocks of the data and privately combine the block answers. One record lives in one block, so it changes only one block answer. The privacy proof never looks inside $f$.

**Result.** Split the records into $B$ blocks by a partition fixed in advance (by position, or by a hash of a record identifier). Compute $z_j = f(\text{block } j)$, clip each to $[0, R]$, and release
$$M(D) = \frac1B\sum_{j=1}^B \operatorname{clip}_{[0,R]}(z_j) + \mathrm{Lap}\Big(\frac{R}{B\varepsilon}\Big),$$
which is $\varepsilon$-DP for any $f$. Its mean squared error has three parts: the squared bias of $f$ on blocks of only $n/B$ records, a sampling term that hardly depends on $B$, and the noise variance $2(R/(B\varepsilon))^2$. More blocks mean less noise and more bias; the best $B$ balances the two.

**Example.** Estimate the rate $\theta = 1$ of $n = 10{,}000$ exponential inter-arrival times at $\varepsilon = 1$. The estimate $1/\bar x$ has infinite global sensitivity. With $R = 5$, @tbl-sens-saa shows the trade-off.

| Blocks $B$ | Block size | Bias | Noise sd | Root-mean-square error |
| --- | --- | --- | --- | --- |
| 10 | 1,000 | 0.0010 | 0.7071 | 0.7072 |
| 100 | 100 | 0.0101 | 0.0707 | 0.0722 |
| 250 | 40 | 0.0256 | 0.0283 | 0.0396 |
| 1,000 | 10 | 0.1111 | 0.0071 | 0.1120 |

: Subsample-and-aggregate for the rate of an exponential distribution, $n = 10{,}000$, $\theta = 1$, $R = 5$, $\varepsilon = 1$. {#tbl-sens-saa}

The best is $B = 250$, with error 0.0396, where bias and noise are of similar size. Without privacy the error would be about 0.0100: a factor of four, for an estimator whose sensitivity you never had to bound.

**Pitfalls.** A partition that depends on the data values lets one record reshuffle many blocks. The range $R$ is still a global bound, now on the output of $f$, and a loose $R$ means large noise. Choose $B$ from the error formula, not by trying several on the data. With a noisy vote as the combiner, this is private aggregation of teacher ensembles (PATE), where models trained on disjoint shards vote.

## Clipping {#sec-sens-clip}

**Idea.** **Clipping** each value to $\min(x, C)$ brings the sensitivity of a mean of unbounded data down to $C/n$. The price is bias, and the choice of $C$ trades it against the noise. The same move applied to per-example gradients is the core of private model training.

**Result.** For non-negative data, the clipped mean $\frac1n\sum_i\min(x_i, C) + \mathrm{Lap}(C/(n\varepsilon))$ is $\varepsilon$-DP. Its mean squared error is the squared bias (the average mass above the clip, $\mathbb E[(X - C)_+]$), plus the sampling variance of the clipped values, plus the noise variance:
$$\operatorname{err}(C)^2 = \mathbb E[(X - C)_+]^2 + \frac{\operatorname{Var}(\min(X, C))}{n} + \frac{2C^2}{n^2\varepsilon^2}.$$ {#eq-sens-clip-error}
Raising $C$ shrinks the bias and grows the noise. The noise shrinks with $n$ and the bias does not, so the best $C$ grows with the dataset. Given a model of your data, minimize @eq-sens-clip-error numerically before seeing any records.

**Example.** Incomes, in thousands, are log-normal with median 40 and mean 55.085 (@tbl-sens-clip).

| Clip bound $C$ | $\Pr[X > C]$ | Bias | Noise sd | Root-mean-square error |
| --- | --- | --- | --- | --- |
| 40 (the median) | 0.5000 | 23.415 | 0.057 | 23.417 |
| 257.23 (99th percentile) | 0.0100 | 0.924 | 0.364 | 1.767 |
| 331.34 (optimal) | 0.0041 | 0.438 | 0.469 | 1.666 |
| 1000 | 0.0000 | 0.006 | 1.414 | 2.169 |

: The clipped private mean of log-normal incomes, $n = 1000$, $\varepsilon = 1$, for four clip bounds. {#tbl-sens-clip}

Clipping at the median is a disaster. The optimum, 331.34, cuts off 0.4% of the population and has error 1.666, only 1% above the 1.649 of the unclipped, non-private mean. It grows with $n$: 177.5 at $n = 100$, 557.3 at $n = 10{,}000$ (@fig-sens-clip-tradeoff).

![Left: root-mean-square error of the clipped private mean of log-normal incomes against the clip bound $C$, with its bias, noise and sampling parts; the dotted line marks the optimum near 331. Right: the optimal clip bound grows with the dataset size.](figures/sens_clip_tradeoff.svg){#fig-sens-clip-tradeoff}

**Pitfalls.** Setting $C$ to the observed maximum or the empirical 99th percentile brings back the leak of @sec-sens-local. Choose $C$ from domain knowledge, or spend part of the budget on a private quantile with the [exponential mechanism](05-selection.md) or the methods above. Clipping from above always underestimates; clipping to $[-C, C]$ is unbiased only for distributions symmetric about 0. Under add/remove neighbours $n$ is not public, so release a noisy clipped sum and a noisy count, divide, and charge both to the budget ([Post-processing, group privacy and composition](02-properties.md)).

## Code {#sec-sens-code}

The smooth sensitivity of the median is easy to get subtly wrong, mostly in the padding of the order statistics. The snippet computes it for $D = (1, 3, 4, 6, 9)$ in $[0, 10]$ with $\beta = 0.5$: the local sensitivity is 2, but one change opens a gap of 5, and $e^{-0.5}\times 5 = 3.03$ wins. It also solves for the $\beta$ of the age example.

```python
from math import exp, log


def gap_within(xs: list[float], upper: float, k: int) -> float:
    """A(k): the widest gap next to the median that changing k records can open."""
    n, m = len(xs), (len(xs) + 1) // 2  # n odd; m is the rank of the median

    def x(i: int) -> float:  # order statistic i, padded with 0 below and `upper` above
        return 0.0 if i < 1 else upper if i > n else xs[i - 1]

    return max(x(m + t) - x(m + t - k - 1) for t in range(k + 2))


def smooth_sensitivity_median(values: list[float], upper: float, beta: float) -> float:
    """The beta-smooth sensitivity of the median of values in [0, upper]."""
    xs = sorted(values)
    return max(exp(-k * beta) * gap_within(xs, upper, k) for k in range(len(xs) + 1))


def laplace_beta(eps: float, delta: float, alpha: float) -> float:
    """The largest beta with beta + e^beta * alpha + (e^beta - 1) * ln(1/delta) <= eps."""
    lo, hi = 0.0, eps
    for _ in range(100):
        mid = (lo + hi) / 2
        ok = mid + exp(mid) * alpha + (exp(mid) - 1) * log(1 / delta) <= eps
        lo, hi = (mid, hi) if ok else (lo, mid)
    return lo


data = [1.0, 3.0, 4.0, 6.0, 9.0]
print(f"{gap_within(sorted(data), 10.0, 0):.2f}")  # 2.00
print(f"{smooth_sensitivity_median(data, 10.0, 0.5):.2f}")  # 3.03
print(f"{laplace_beta(1.0, 1e-6, 0.5):.4f}")  # 0.0322
```

The distance for propose-test-release is the first `k` with `gap_within(xs, upper, k) > b`. The full edition's script computes every number in this chapter and checks the formula for $A^{(k)}$ by brute force:

```bash
uv run python code/part03-dp-foundations/sensitivity.py
```

## Cheat sheet

| Technique | Release | Privacy | Use when |
| --- | --- | --- | --- |
| Local sensitivity | $f(D) + \mathrm{Lap}(\mathrm{LS}_f(D)/\varepsilon)$ | **Not private** | Never |
| Smooth sensitivity, Cauchy | $f(D) + \frac{6}{\varepsilon}S^*_{\varepsilon/6}(D)\,Z$ | $\varepsilon$ | Median, quantiles; $n$ in the hundreds or more |
| Smooth sensitivity, Laplace | $f(D) + \frac{S^*_\beta(D)}{\alpha}\mathrm{Lap}(1)$, $\beta + e^\beta\alpha + (e^\beta - 1)\ln\frac1\delta \le \varepsilon$ | $(\varepsilon, \delta)$ | Same, when you want light tails |
| Propose-test-release | $\bot$ if $\operatorname{dist}_b + \mathrm{Lap}(\frac1\varepsilon) \le \frac{\ln(1/\delta)}{\varepsilon}$, else $f(D) + \mathrm{Lap}(\frac b\varepsilon)$ | $(2\varepsilon, \delta)$ | Stable statistics without a smooth-sensitivity formula |
| Subsample-and-aggregate | Mean of $B$ clipped block outputs $+ \mathrm{Lap}(\frac{R}{B\varepsilon})$ | $\varepsilon$ | Black-box estimators and models |
| Clipped mean | $\frac1n\sum\min(x_i, C) + \mathrm{Lap}(\frac{C}{n\varepsilon})$; bias $\mathbb E[(X - C)_+]$ | $\varepsilon$ | Means and sums of unbounded data |

: The techniques of this chapter, with neighbours that replace one record. {#tbl-sens-cheat}

**Rules of thumb**

- If a noise scale depends on the data, ask what the scale reveals. If you can't answer, use one of the repairs above.
- Pure DP with a data-dependent scale needs heavy-tailed noise; Laplace noise needs a $\delta$ and a $\beta$ near $\varepsilon/(2\ln(1/\delta))$.
- Smooth sensitivity pays off once the data are dense around the statistic: for the median, once $n$ is in the hundreds.
- Fix clip bounds, block counts and proposals before looking at the data, from an error formula or with a separate slice of the budget.

## Further reading

- Nissim, Raskhodnikova & Smith, "Smooth Sensitivity and Sampling in Private Data Analysis" (STOC 2007): smooth sensitivity, the admissible noise distributions, the median formula and the original sample-and-aggregate framework.
- Dwork & Lei, "Differential Privacy and Robust Statistics" (STOC 2009): propose-test-release, and robust private estimators built on it.
- Amin, Kulesza, Muñoz Medina & Vassilvitskii, "Bounding User Contributions: A Bias-Variance Trade-off in Differential Privacy" (ICML 2019): choosing clip bounds when users contribute many records.

**Derivations.** Every result in this chapter is derived step by step in [the full edition's chapter](../../book-full/part03-dp-foundations/06-sensitivity.md).
