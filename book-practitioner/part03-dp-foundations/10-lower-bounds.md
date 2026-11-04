# Lower bounds: what privacy costs

## What it is {#sec-lower-packing}

A **lower bound** states the best accuracy any mechanism satisfying differential privacy (DP) can achieve. It applies to all such mechanisms and provides a feasibility check. Accuracy has two parameters: **error** $\alpha$, the maximum distance between output and true answer, and **failure probability** $\beta$, the probability that the error exceeds $\alpha$.

One common lower bound applies to a histogram with $N$ bins under pure DP, at privacy level $\varepsilon$ and with no slack. If every count is within $\alpha$ records of its true value with probability at least $1 - \beta$, then
$$\alpha \ge \frac12\left(\frac{\ln\big(N(1 - \beta)\big)}{\varepsilon} - 1\right).$$ {#eq-lower-histogram}
The bound grows logarithmically with the number of bins and remains positive under pure DP. Under approximate DP with slack $\delta$, it is about $\ln(1/\delta)/\varepsilon$ records, independent of the number of bins.

## When you meet it {#sec-lower-sample}

Use a lower bound to check whether a proposed accuracy is feasible. For example, consider a histogram of one million locations at $\varepsilon = 1$, with no slack, that promises every bin within five records of its true count with 95% probability.

Determine whether one person affects one statistic, as in disjoint histogram cells, or many statistics, as in means of multiple attributes. The histogram bound applies to the first case; fingerprinting and reconstruction bounds apply to the second. If a target is below the lower bound, increase the data, coarsen the answer or weaken the guarantee. A mechanism near the bound is close to the best possible accuracy.

## How to use it

For one million locations, every pure-DP histogram has error of at least 6.38 records in some bin, so the five-record target is impossible. The Laplace histogram in [the Laplace and Gaussian mechanisms](03-laplace-gaussian.md) has error 16.79, about 2.6 times the lower bound. At $\delta = 10^{-6}$, the bound is below about $\ln(10^6) = 13.82$ records. A stability-based histogram that releases only non-empty bins has error independent of $N$ ([Beyond global sensitivity](07-sensitivity.md)). For a large, mostly empty universe, approximate DP may therefore provide better utility.

The same bound gives a minimum dataset size: dividing count error by the record count $n$ gives frequency error. A pure-DP histogram with all frequencies within $\alpha$ requires
$$n \ge \frac{\ln\big(N(1 - \beta)\big) - \varepsilon}{2\alpha\varepsilon}.$$ {#eq-lower-n-histogram}
If the data are sampled from a population, sampling error also requires records: about $\ln(2N/\beta)/(2\alpha^2)$ by [Hoeffding's inequality](../../book-full/part01-prerequisites/02-concentration.md). Use the larger of the privacy and sampling requirements. At $\varepsilon = 1$ and 95% probability, a million cells with error 0.01 require at least 638 records for privacy but 87,522 for sampling. Sampling, not privacy, therefore limits the accuracy; no mechanism can remove that limit.

When one person affects many statistics, the cost increases. Under pure DP, estimating means of $d$ yes/no attributes requires a number of records proportional to $d$. Under approximate DP, the **fingerprinting** bound requires about $\sqrt{d\ln(1/\delta)}/(\alpha\varepsilon)$ records. The Gaussian mechanism is within a factor $\sqrt{\ln d}$ of this bound. Use approximate DP for high-dimensional outputs such as gradients. Once Gaussian noise is calibrated, adding records is more effective than seeking a substantially better mechanism. For about $n$ counting queries on $n$ records, the reconstruction attack of Dinur and Nissim gives a lower bound of order $\sqrt n$ error per answer ([Reconstruction attacks](../part02-attacks/09-reconstruction.md)).

## What goes wrong {#sec-lower-tracing}

Aggregates do not prevent membership inference. The attack underlying the fingerprinting bound (Homer et al. 2008) identified participants in a genome study from published frequencies of genetic variants. It scores how much released means shift toward a target's record and away from population means, attribute by attribute. Members tend to score higher because their records affect the means. For the means of 50 people over 2,000 attributes, the attack reaches true positive rate (TPR) 0.242 at false positive rate (FPR) 1%. A $(1, 10^{-5})$-DP Gaussian mechanism reduces TPR to 0.014.

Use this score to audit releases of many averages, estimating population means from a separate sample. Compare TPR at low FPR with the cap $\mathrm{TPR} \le e^\varepsilon\,\mathrm{FPR} + \delta$ from the [definition](01-definition.md). An audit that exceeds the cap indicates a mechanism or accounting error. Passing does not establish privacy because an attack provides only a lower bound on $\varepsilon$ ([Auditing private training](../part05-dp-ml/10-auditing.md)).

Each approximate-DP bound assumes a small $\delta$, such as the book's default ([How small δ must be](01-definition.md#sec-dpdef-small-delta)). A mechanism that publishes each record with probability $\delta$ is $(0, \delta)$-DP; if $\delta$ is near $1/n$, it can publish about one record in full.

**Derivations.** Every result here is derived step by step in [the full edition's chapter](../../book-full/part03-dp-foundations/10-lower-bounds.md).
