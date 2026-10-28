# Lower bounds: what privacy costs

## What it is {#sec-lower-packing}

A **lower bound** says that no mechanism with differential privacy (DP) can be more accurate than some floor. It holds for all mechanisms at once, so you can use it as a feasibility check. Accuracy has two numbers here. The **error** $\alpha$ is how far the output may land from the true answer, and the **failure probability** $\beta$ is the chance that it lands further away.

The floor you meet most is for a histogram over $N$ bins under pure DP, with a privacy level $\varepsilon$ and no slack. Suppose that such a histogram is within $\alpha$ records of every count with probability at least $1 - \beta$. Then
$$\alpha \ge \frac12\left(\frac{\ln\big(N(1 - \beta)\big)}{\varepsilon} - 1\right).$$ {#eq-lower-histogram}
The floor grows only with the logarithm of the number of bins, but under pure DP it never stops. Under approximate DP, which allows a small slack $\delta$, it stops at about $\ln(1/\delta)/\varepsilon$ records for all numbers of bins.

## When you meet it {#sec-lower-sample}

You meet a lower bound when a design looks too good. Take a request for a histogram over a million locations at $\varepsilon = 1$ with no slack, accurate to five records in every bin with 95% probability.

First ask if one person moves one statistic, as in the disjoint cells of a histogram, or all of them, as in the means of many attributes. The histogram floor applies to the first case, and the fingerprinting and reconstruction floors below to the second. Then the floor tells you whether to tune more. A target below it needs more data, a coarser answer or a weaker guarantee. A mechanism within a small factor of the floor is done, because no algorithm can gain more than that factor.

## How to use it

For the million locations, all pure-DP histograms have an error of 6.38 records or more in some bin. So the five-record promise is impossible. The Laplace histogram of [the Laplace and Gaussian mechanisms](03-laplace-gaussian.md), with an error of 16.79, is about 2.6 times the floor and so already near the best. With $\delta = 10^{-6}$, the floor stays below about $\ln(10^6) = 13.82$ records. A stability-based histogram, which releases only non-empty bins, has an error that does not grow with $N$ ([Beyond global sensitivity](07-sensitivity.md)). So when a large, mostly empty universe puts the floor too high, change to approximate DP.

The same floor gives the size of a dataset, because an error in records divided by the number of records $n$ is an error in frequencies. A pure-DP histogram whose frequencies must all be within $\alpha$ needs
$$n \ge \frac{\ln\big(N(1 - \beta)\big) - \varepsilon}{2\alpha\varepsilon}.$$ {#eq-lower-n-histogram}
If the data are a sample from a population, sampling costs records too, about $\ln(2N/\beta)/(2\alpha^2)$ by [Hoeffding's inequality](../../book-full/part01-prerequisites/02-concentration.md). You need the larger of the two. At $\varepsilon = 1$ and 95% probability, a million cells at an error of 0.01 need at least 638 records for privacy, but 87,522 for sampling. So at $\varepsilon = 1$, fine answers run short of records because of sampling, which no mechanism can fix.

When one person moves many statistics, the cost grows faster. Under pure DP, the means of $d$ yes/no attributes cost records in proportion to $d$. Under approximate DP, the **fingerprinting** bound puts the cost at about $\sqrt{d\ln(1/\delta)}/(\alpha\varepsilon)$ records. The Gaussian mechanism is within a factor $\sqrt{\ln d}$ of it. So use approximate DP for outputs with many coordinates, such as a gradient. After you calibrate the Gaussian noise, search for more records, not a better mechanism. For about $n$ counting queries on $n$ records, the reconstruction attack of Dinur and Nissim puts the floor at an error of order $\sqrt n$ per answer ([Reconstruction attacks](../part02-attacks/09-reconstruction.md)).

## What goes wrong {#sec-lower-tracing}

Aggregates are not a defence. The attack behind the fingerprinting bound (Homer et al. 2008) found whether a given person took part in a genome study, from its published frequencies of genetic variants. It scores how far the released means lean towards the target's record and away from the population means, attribute by attribute. Members score higher, because each member pulls the means toward their own record. On the means of 50 people over 2,000 attributes, it reaches a true positive rate (TPR) of 0.242 at a false positive rate (FPR) of 1%. A $(1, 10^{-5})$-DP Gaussian mechanism cuts that TPR to 0.014.

So use the score to audit a release of many averages, with population means from a separate sample. Compare its TPR at a low FPR with the cap $\mathrm{TPR} \le e^\varepsilon\,\mathrm{FPR} + \delta$ from the [definition](01-definition.md). If an audit beats the cap, it has found a bug in the mechanism or its accounting. Passing proves nothing, because an attack only bounds $\varepsilon$ from below ([Auditing private training](../part05-dp-ml/10-auditing.md)).

Each approximate-DP floor here assumes a small $\delta$, such as the book's default ([How small δ must be](01-definition.md#sec-dpdef-small-delta)), or it means little. A mechanism that publishes each record with probability $\delta$ is $(0, \delta)$-DP, so a claim with $\delta$ near $1/n$ lets about one record out in full.

**Derivations.** Every result here is derived step by step in [the full edition's chapter](../../book-full/part03-dp-foundations/10-lower-bounds.md).
