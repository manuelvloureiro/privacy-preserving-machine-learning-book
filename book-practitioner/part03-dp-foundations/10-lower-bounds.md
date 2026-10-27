# Lower bounds: what privacy costs

## What it is {#sec-lower-packing}

A **lower bound** says that no mechanism satisfying differential privacy (DP) can be more accurate than some floor. It holds for every mechanism at once, so it works as a feasibility check. Accuracy has two numbers here. The **error** $\alpha$ is how far the output may land from the true answer, and the **failure probability** $\beta$ is the chance that it lands further away.

The floor you meet most is for a histogram over $N$ bins under pure DP, with a privacy level $\varepsilon$ and no slack. If such a histogram is within $\alpha$ records of every count with probability at least $1 - \beta$, then
$$\alpha \ge \frac12\left(\frac{\ln\big(N(1 - \beta)\big)}{\varepsilon} - 1\right).$$ {#eq-lower-histogram}
The floor grows only with the logarithm of the number of bins, but under pure DP it keeps growing. Under approximate DP, which allows a small slack $\delta$, it stops at about $\ln(1/\delta)/\varepsilon$ records, however many bins there are.

## When you meet it {#sec-lower-sample}

You meet a lower bound when a design looks too good. Take a request for a histogram over a million locations at $\varepsilon = 1$ with no slack, accurate to five records in every bin with 95% probability. First ask whether one person moves one of the statistics, as in the disjoint cells of a histogram, or all of them, as in the means of many attributes. The histogram floor governs the first case, and the fingerprinting and reconstruction floors below the second. The floor then decides whether to keep tuning. A target below it needs more data, a coarser answer or a weaker guarantee. A mechanism within a small factor of it is done, because no algorithm gains more than that factor.

## How to use it

For the million locations, every pure-DP histogram has an error of at least 6.38 records in some bin, so the five-record promise is impossible. The Laplace histogram of [the Laplace and Gaussian mechanisms](03-laplace-gaussian.md) has an error of 16.79, about 2.6 times the floor, so it is already near the best. With $\delta = 10^{-6}$, the floor never passes about $\ln(10^6) = 13.82$ records. A stability-based histogram, which releases only non-empty bins, has an error that does not grow with $N$ at all ([Beyond global sensitivity](07-sensitivity.md)). So when a huge, mostly empty universe puts the floor too high, switch to approximate DP.

The same floor sizes a dataset, because dividing an error in records by the number of records $n$ gives an error in frequencies. A pure-DP histogram whose frequencies must all be within $\alpha$ needs
$$n \ge \frac{\ln\big(N(1 - \beta)\big) - \varepsilon}{2\alpha\varepsilon}.$$ {#eq-lower-n-histogram}
If the data are a sample from a population, sampling costs records too, about $\ln(2N/\beta)/(2\alpha^2)$ by [Hoeffding's inequality](../part01-prerequisites/02-concentration.md), and you need the larger of the two. At $\varepsilon = 1$ and 95% probability, a million cells at an error of 0.1 need at least 64 records for privacy and at most 875 for sampling. At an error of 0.01, privacy needs 638 and sampling 87,522. So at $\varepsilon = 1$ it is sampling, which no mechanism fixes, that makes fine answers short of records.

When one person moves many statistics, the rate changes. Releasing the means of $d$ yes/no attributes costs records in proportion to $d$ under pure DP. Under approximate DP, the **fingerprinting** bound puts the cost at about $\sqrt{d\ln(1/\delta)}/(\alpha\varepsilon)$ records, and the Gaussian mechanism matches it to within $\sqrt{\ln d}$. So use approximate DP for outputs with many coordinates, such as a gradient, and once the Gaussian noise is calibrated, look for more records, not a better mechanism. For about $n$ counting queries on $n$ records, the reconstruction attack of Dinur and Nissim sets the floor at an error of order $\sqrt n$ per answer ([Reconstruction attacks](../part02-attacks/09-reconstruction.md)). With far more queries than records, answer them from one private synthetic dataset, whose error grows only slowly with their number ([Query release](06-query-release.md)).

## What goes wrong {#sec-lower-tracing}

Aggregates are not a defence. The attack behind the fingerprinting bound, from Homer and colleagues (2008), found from a genome study's published frequencies of genetic variants whether a given person took part. It scores how far the released means lean, attribute by attribute, towards the target's record and away from the population means. Members score higher, because each pulls the means toward their own record. On the means of 50 people over 2,000 attributes, it reaches a true positive rate (TPR) of 0.242 at a false positive rate (FPR) of 1%. A $(1, 10^{-5})$-DP Gaussian mechanism cuts that TPR to 0.014.

So use the score, with population means from a separate sample, to audit any release of many averages. Compare its TPR at a low FPR with the cap $\mathrm{TPR} \le e^\varepsilon\,\mathrm{FPR} + \delta$ from the [definition](01-definition.md). An audit that beats the cap has found a bug in the mechanism or its accounting. Passing proves nothing, because an attack only bounds $\varepsilon$ from below ([Auditing private training](../part05-dp-ml/10-auditing.md)).

Every approximate-DP floor here assumes a meaningful $\delta$, such as the book's default $\delta \le 1/n^{1.1}$ ([How small δ must be](01-definition.md#sec-dpdef-small-delta)). Otherwise they mean little: a mechanism that publishes each record outright with probability $\delta$ is $(0, \delta)$-DP, so a claim with $\delta$ near $1/n$ allows publishing about one record.

**Derivations.** Every result here is derived step by step in [the full edition's chapter](../../book-full/part03-dp-foundations/10-lower-bounds.md).
