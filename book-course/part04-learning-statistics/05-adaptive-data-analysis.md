# Adaptive data analysis

A lab tests hypothesis after hypothesis on one dataset, each suggested by the earlier results, until one is significant. Often, fresh data does not reproduce that result. Each test may be sound alone, but the sequence used the same data to choose the questions and to answer them. So the answers describe the sample instead of the population.

This chapter makes that failure precise for averages of predicates, called statistical queries, and shows how differential privacy (DP) prevents it. Queries fixed in advance need a sample size logarithmic in the number $k$ of queries, and adaptively chosen queries can overfit unless it is of order $k$. A mechanism that answers through a DP algorithm needs only about $\sqrt k$, because a private algorithm cannot find a query on which the sample and the population disagree.

## Statistical queries {#sec-ada-setup}

The data is a sample from an unknown distribution $\mathcal D$ on a universe $\mathcal X$, and the analyst cares about the distribution, not the sample. Each question asks for the fraction of records with a yes-or-no property, so it has a value on the sample and a value on the distribution.

::: {#def-ada-query name="Statistical query"}
A **statistical query** is a function $q \colon \mathcal X \to \{0,1\}$. Its value on the distribution and its empirical value on a dataset $D = (X_1, \dots, X_n)$ are

$$
q(\mathcal D) = \mathbb E_{x \sim \mathcal D}[q(x)], \qquad q(D) = \frac1n \sum_{j=1}^n q(X_j).
$$
:::

So $q(\mathcal D)$ is the fraction of the population with the property $q$, such as red hair, and $q(D)$ is the fraction in the sample. These are the counting queries of earlier chapters, divided by $n$. If the records are independent draws from $\mathcal D$, the empirical value is an unbiased estimate of the population value. For a fixed list of queries, a sample whose size is logarithmic in the length of the list is enough.

::: {#prp-ada-nonadaptive}
Let $\alpha > 0$ be the allowed error and $\beta \in (0,1)$ the allowed failure probability. Let $q_1, \dots, q_k$ be statistical queries fixed before the sample is drawn, and let $X_1, \dots, X_n$ be independent draws from $\mathcal D$. If

$$
n \ge \frac{\ln(2k/\beta)}{2\alpha^2},
$$

then some query has error at least $\alpha$ with probability at most $\beta$,

$$
\Pr\big[\exists i \colon |q_i(D) - q_i(\mathcal D)| \ge \alpha\big] \le \beta .
$$
:::

::: {.proof}
1. For a fixed $i$, the values $q_i(X_1), \dots, q_i(X_n)$ are independent, lie in $[0,1]$, and have mean $q_i(\mathcal D)$. Hoeffding's inequality gives $\Pr[|q_i(D) - q_i(\mathcal D)| \ge \alpha] \le 2e^{-2\alpha^2 n}$.
2. A union bound over the $k$ queries gives failure probability at most $2k e^{-2\alpha^2 n}$, which the condition on $n$ makes at most $\beta$.
:::

So a sample of size $n$ supports about $e^{2\alpha^2 n}$ queries. But step 1 needs the query to be independent of the sample, and adaptivity breaks that step.

## Adaptivity breaks the bound {#sec-ada-adaptivity}

In an adaptive analysis, each query may depend on the answers to the earlier ones. The definition sets up this interaction, and says when the mechanism that answers is private whatever the analyst does.

::: {#def-ada-interaction name="Adaptive analysis"}
An **analyst** $\mathcal A$ and a **mechanism** $M$ interact for $k$ rounds. In round $i$, the analyst chooses a statistical query $q_i$ as a function of the earlier answers $a_1, \dots, a_{i-1}$ and its own random bits, called its coins. The mechanism returns an answer $a_i$, computed from the dataset $D$, its own coins and the history. The mechanism is **$(\varepsilon,\delta)$-DP against adaptive analysts** if for every analyst $\mathcal A$ the map from $D$ to the transcript $(q_1, a_1, \dots, q_k, a_k)$ is $(\varepsilon, \delta)$-DP.
:::

The empirical mechanism $a_i = q_i(D)$ answers every query exactly on the sample, and an adaptive analyst makes it fail badly.

::: {#prp-ada-overfit}
Let $\mathcal X = \{1, \dots, N\}$, let $\mathcal D$ be uniform on $\mathcal X$, and let the mechanism answer $a_i = q_i(D)$. Some analyst with $k = N+1$ queries ends with a query $q_k$ such that, for every sample,

$$
q_k(D) - q_k(\mathcal D) \ge 1 - \frac{n}{k-1}.
$$
:::

::: {.proof}
1. For $i = 1, \dots, N$, the analyst asks $q_i(x) = \mathbf 1[x = i]$. The answer $a_i$ is the fraction of records equal to $i$, so after $N$ rounds the analyst knows the set $S = \{X_1, \dots, X_n\}$ of values in the sample.
2. The last query is $q_k(x) = \mathbf 1[x \in S]$. Every record lies in $S$, so $q_k(D) = 1$.
3. The set $S$ has at most $n$ elements, so $q_k(\mathcal D) = |S|/N \le n/N = n/(k-1)$.
:::

The first $N$ queries are harmless one at a time, and each could be fixed in advance. The analyst builds the last one from their answers, and on it the sample and the population disagree by almost $1$ unless $n$ is of order $k$. So the number of queries that a sample supports drops from exponential in $n$ to linear in $n$.

A simple defence is to give each query fresh data. The analyst can still choose adaptively, but each answer comes from records that no earlier answer has seen.

::: {#prp-ada-split name="Sample splitting"}
Let $k$ divide $n$. Split $D$ into $k$ disjoint parts $D^{(1)}, \dots, D^{(k)}$ of $n/k$ records each, and answer $a_i = q_i(D^{(i)})$. If the records are independent draws from $\mathcal D$ and

$$
n \ge \frac{k\ln(2k/\beta)}{2\alpha^2},
$$

then for every analyst $\Pr\big[\exists i \colon |a_i - q_i(\mathcal D)| \ge \alpha\big] \le \beta$.
:::

::: {.proof}
1. The query $q_i$ depends only on the coins of the analyst and on $a_1, \dots, a_{i-1}$, which depend only on the parts $D^{(1)}, \dots, D^{(i-1)}$. So $q_i$ is independent of $D^{(i)}$.
2. Conditioned on $q_i$, the part $D^{(i)}$ is still $n/k$ independent draws from $\mathcal D$. Step 1 of the proof of @prp-ada-nonadaptive gives $\Pr[|a_i - q_i(\mathcal D)| \ge \alpha] \le 2e^{-2\alpha^2 n/k}$.
3. A union bound over $i$ gives at most $2k e^{-2\alpha^2 n/k}$, which the condition makes at most $\beta$.
:::

Sample splitting is valid but expensive, because it needs $k$ times the data of the non-adaptive case. A DP mechanism does much better. With Gaussian noise, a sample of size

$$
n = O\!\left(\frac{\sqrt{k}\,\ln^{3/2}(k/(\alpha\beta))}{\alpha^2}\right)
$$

is enough for error $\alpha$ (@cor-ada-gaussian). Under a standard cryptographic assumption, no computationally efficient mechanism can do better than $\sqrt k$ in general.

## Privacy implies generalization {#sec-ada-transfer}

To overfit, the analyst must learn too much about individual records, as the analyst of @prp-ada-overfit did. If every answer passes through a DP mechanism, no answer reveals much about one record, so the analyst can learn only about the population. A **transfer theorem** makes this precise. It turns accuracy on the sample and privacy into accuracy on the population.

::: {#thm-ada-transfer name="Transfer theorem"}
Let $\alpha, \beta \in (0, 0.1)$, and let $M$ answer $k$ adaptively chosen statistical queries. Suppose that

1. the mechanism is $(\alpha/64, \alpha\beta/32)$-DP against adaptive analysts, and
2. the mechanism is accurate on the sample, that is, for every analyst and every dataset $D$, $\Pr\big[\exists i \colon |a_i - q_i(D)| > \alpha/8\big] \le \alpha\beta/16$, with probability over the coins of $M$ and of the analyst.

Then for every analyst and every distribution $\mathcal D$, if the $n$ records are independent draws from $\mathcal D$,

$$
\Pr\big[\exists i \colon |a_i - q_i(\mathcal D)| > \alpha\big] \le \beta .
$$
:::
<!-- proof: cited -->

This is Theorem 3.4 of Bassily, Nissim, Smith, Steinke, Stemmer and Ullman (2016), for queries of sensitivity $1/n$. The failure probability on the sample must be of order $\alpha\beta$, not $\beta$, for the conclusion to hold with failure probability $\beta$. The proof runs about $1/\beta$ independent copies of the interaction, each on its own sample. A monitor, which knows $\mathcal D$ and exists only in the proof, picks the copy and the query with the largest error on the population. Its choice post-processes DP outputs, so it is DP, and a version of @lem-ada-key below for several samples shows that it cannot find a large population error on average.

Two results from earlier chapters apply the theorem. The [Gaussian mechanism](../part02-dp-basics/03-approximate-dp.md) answers a query $f$ with Gaussian noise, and the sensitivity $\Delta f$ sets the standard deviation. With $\sigma = \Delta f\sqrt{2\ln(1/\delta_0) + \varepsilon_0}/\varepsilon_0$, the mechanism is $(\varepsilon_0, \delta_0)$-DP for every $\varepsilon_0 > 0$ and $\delta_0 \in (0,1)$. A corollary of [advanced composition](../part02-dp-basics/04-advanced-composition.md) says that, for $\varepsilon \in (0,1]$ and $\delta' \in (0, 1/2]$, $k$ adaptively chosen mechanisms that are each $(\varepsilon_0, \delta_0)$-DP with

$$
\varepsilon_0 = \frac{\varepsilon}{2\sqrt{2k\ln(1/\delta')}}
$$

are, together, $(\varepsilon, k\delta_0 + \delta')$-DP. A statistical query has sensitivity $1/n$, because a replaced record changes one term of the average by at most $1$. The corollary adds Gaussian noise to each empirical answer, at a scale that meets both hypotheses of @thm-ada-transfer, and gives the sample size that this needs.

::: {#cor-ada-gaussian name="Gaussian answers generalize"}
Let $\alpha, \beta \in (0, 0.1)$. Answer each query by $a_i = q_i(D) + Z_i$ with independent $Z_i \sim \mathcal N(0, \sigma^2)$, where

$$
\varepsilon_0 = \frac{\alpha}{128\sqrt{2k\ln(64/(\alpha\beta))}}, \qquad \sigma = \frac{\sqrt{2\ln(64k/(\alpha\beta)) + \varepsilon_0}}{\varepsilon_0\, n}.
$$

If

$$
n \ge \frac{2048\sqrt3}{\alpha^2}\sqrt{k\ln\frac{64}{\alpha\beta}}\;\ln\frac{64k}{\alpha\beta},
$$ {#eq-ada-gaussian-n}

then for every analyst, $\Pr\big[\exists i \colon |a_i - q_i(\mathcal D)| > \alpha\big] \le \beta$.
:::

::: {.proof}
1. *Each answer is private.* With $\delta_0 = \alpha\beta/(64k)$, the noise is the Gaussian mechanism's for sensitivity $1/n$ and budget $(\varepsilon_0, \delta_0)$. For a fixed analyst, the $i$-th answer is this mechanism applied to a query fixed by the earlier answers and the coins of the analyst.
2. *All answers are $(\alpha/64, \alpha\beta/32)$-DP.* The corollary of advanced composition applies with $\varepsilon = \alpha/64$ and $\delta' = \alpha\beta/64$. The total $\delta$ is $k\delta_0 + \delta' = \alpha\beta/32$. The queries in the transcript are a post-processing of the answers and the coins of the analyst, so the transcript has the same guarantee.
3. *Accuracy on the sample.* The error $a_i - q_i(D)$ is the noise $Z_i$, and a Gaussian variable exceeds $t$ standard deviations in absolute value with probability at most $2e^{-t^2/2}$. With $t = \sqrt{2\ln(32k/(\alpha\beta))}$, each $|Z_i|$ exceeds $t\sigma$ with probability at most $\alpha\beta/(16k)$, so by a union bound all $k$ answers are within $t\sigma$ of the sample values with probability at least $1 - \alpha\beta/16$.
4. *The threshold.* Write $\ell = \ln(64k/(\alpha\beta))$. Since $\varepsilon_0 < 1 < \ell$, we have $2\ell + \varepsilon_0 \le 3\ell$, and $t \le \sqrt{2\ell}$. So

$$
t\sigma \le \frac{\sqrt{6}\,\ell}{\varepsilon_0 n} = \frac{256\sqrt3}{\alpha n}\sqrt{k\ln\frac{64}{\alpha\beta}}\;\ell ,
$$

which @eq-ada-gaussian-n makes at most $\alpha/8$. Steps 2 to 4 are the hypotheses of @thm-ada-transfer, which gives the claim.
:::

The $\sqrt k$ in @eq-ada-gaussian-n is the $\sqrt k$ of advanced composition. The constant is large because the constants of the transfer theorem are. But the number of queries that a sample supports grows from linear in $n$, for sample splitting, to quadratic in $n$. For a finite universe and a very large $k$, [private multiplicative weights](../part03-query-release/01-multiplicative-weights.md) does better still. For the average error over the queries, it needs a sample only logarithmic in $k$ (Bassily et al., 2016, Corollary 6.3).

## An easier transfer theorem {#sec-ada-easy}

A weaker transfer theorem, with accuracy in expectation instead of with high probability, has a short complete proof. It turns a bound on the expected worst error on the sample into one on the population.

::: {#thm-ada-easy name="Transfer in expectation"}
Let $M$ be $(\varepsilon, \delta)$-DP against adaptive analysts. Suppose that for every analyst, if the records are independent draws from $\mathcal D$, then

$$
\mathbb E\Big[\max_{i \le k} |a_i - q_i(D)|\Big] \le \alpha .
$$

Then for every analyst,

$$
\mathbb E\Big[\max_{i \le k} |a_i - q_i(\mathcal D)|\Big] \le \alpha + e^{\varepsilon} - 1 + \delta .
$$
:::
<!-- proof: after @lem-ada-key -->

For small $\varepsilon$, the excess is about $\varepsilon + \delta$. Most of the work is in a lemma about one query. It says that a query chosen by a DP algorithm has on average about the same value on the sample as on the population. So a DP algorithm cannot produce the last query of @prp-ada-overfit, on which the two disagree.

::: {#lem-ada-key name="A private query generalizes"}
Let $W$ be an $(\varepsilon, \delta)$-DP algorithm that maps a dataset in $\mathcal X^n$ to a statistical query. Let $X_1, \dots, X_n$ be independent draws from $\mathcal D$ and $q = W(D)$. Then

$$
\big|\mathbb E[q(D)] - \mathbb E[q(\mathcal D)]\big| \le e^{\varepsilon} - 1 + \delta ,
$$

with expectations over the sample and the coins of $W$.
:::

::: {.proof}
1. *Linearity.* Since $q$ takes values in $\{0,1\}$,

$$
\mathbb E[q(D)] = \frac1n\sum_{i=1}^n \mathbb E[q(X_i)] = \frac1n\sum_{i=1}^n \Pr[q(X_i) = 1].
$$

2. *A fresh record.* Fix $i$, draw $X_i'$ from $\mathcal D$ independently of everything else, and let $D^{(i)}$ be $D$ with entry $i$ replaced by $X_i'$. The datasets $D$ and $D^{(i)}$ are neighbours, $D \sim D^{(i)}$, since they differ in one replaced record.
3. *Privacy.* Write $W(D)(X_i)$ for the value of the query $W(D)$ on the record $X_i$. For fixed values of $D$ and $X_i'$, the queries $q'$ with $q'(X_i) = 1$ form an event about the output of $W$. DP for the neighbours $D$ and $D^{(i)}$ gives

$$
\Pr_W\big[W(D)(X_i) = 1\big] \le e^{\varepsilon}\Pr_W\big[W(D^{(i)})(X_i) = 1\big] + \delta .
$$

The average over $D$ and $X_i'$ keeps the inequality.
4. *Swapping.* The pair $(D^{(i)}, X_i)$ has the same distribution as the pair $(D, X_i')$, since each is $n$ independent draws from $\mathcal D$ plus one more independent draw. Hence $\Pr[W(D^{(i)})(X_i) = 1] = \Pr[q(X_i') = 1]$. The record $X_i'$ is independent of $q = W(D)$, so this probability is $\mathbb E[q(\mathcal D)]$.
5. *One direction.* Steps 1, 3 and 4 give $\mathbb E[q(D)] \le e^\varepsilon\,\mathbb E[q(\mathcal D)] + \delta$. Since $\mathbb E[q(\mathcal D)] \le 1$,

$$
\mathbb E[q(D)] - \mathbb E[q(\mathcal D)] \le (e^\varepsilon - 1)\,\mathbb E[q(\mathcal D)] + \delta \le e^{\varepsilon} - 1 + \delta .
$$

6. *Other direction.* DP applied from $D^{(i)}$ to $D$ gives $\Pr[W(D^{(i)})(X_i) = 1] \le e^\varepsilon \Pr[W(D)(X_i) = 1] + \delta$. With steps 1 and 4, this is $\mathbb E[q(\mathcal D)] \le e^\varepsilon\,\mathbb E[q(D)] + \delta$, and the same argument as in step 5 bounds $\mathbb E[q(\mathcal D)] - \mathbb E[q(D)]$ by $e^{\varepsilon} - 1 + \delta$.
:::

The proof of @thm-ada-easy lets a private algorithm pick out the query with the worst error, and two tricks make that query fit the lemma.

::: {.proof}
1. *Removing the absolute value.* Whenever the analyst asks $q_i$ and receives $a_i$, add the query $1 - q_i$ with the answer $1 - a_i$. The extended transcript is a post-processing of the original, so it is still $(\varepsilon,\delta)$-DP. The errors on the sample are unchanged, because $|(1 - a_i) - (1 - q_i)(D)| = |a_i - q_i(D)|$. Since $(1-q_i)(\mathcal D) - (1 - a_i) = -(q_i(\mathcal D) - a_i)$, the $2k$ queries $q_j$ satisfy

$$
\max_{i \le k}|q_i(\mathcal D) - a_i| = \max_{j \le 2k}\big(q_j(\mathcal D) - a_j\big).
$$

2. *The selector.* Let $W$ run the interaction on $D$ and output the query $q^\ast = q_{j^\ast}$, with answer $a^\ast = a_{j^\ast}$, where $j^\ast$ maximizes $q_j(\mathcal D) - a_j$. The selector uses $\mathcal D$, which is fixed and is not the data, so $W$ is a post-processing of the transcript and is $(\varepsilon,\delta)$-DP. It is a device of the proof and is never run.
3. *Splitting the error.* By linearity of expectation,

$$
\mathbb E\Big[\max_{i}|q_i(\mathcal D) - a_i|\Big] = \mathbb E\big[q^\ast(\mathcal D) - q^\ast(D)\big] + \mathbb E\big[q^\ast(D) - a^\ast\big] .
$$

4. *The two terms.* @lem-ada-key, applied to $W$, bounds the first term by $e^\varepsilon - 1 + \delta$. The second is at most $\mathbb E[\max_j |q_j(D) - a_j|] \le \alpha$, by the accuracy hypothesis and step 1.
:::

The Gaussian mechanism applies this theorem too, with a proof that cites nothing. The only change from @cor-ada-gaussian is that the error on the sample is bounded in expectation.

::: {#cor-ada-expected}
Let $\alpha \in (0, 1]$. Answer each query by $a_i = q_i(D) + Z_i$ with independent $Z_i \sim \mathcal N(0, \sigma^2)$, where

$$
\varepsilon_0 = \frac{\alpha}{2\sqrt{2k\ln(2/\alpha)}}, \qquad \sigma = \frac{\sqrt{2\ln(2k/\alpha) + \varepsilon_0}}{\varepsilon_0\, n}.
$$

If

$$
n \ge \frac{4\sqrt3}{\alpha^2}\sqrt{k\ln\frac{2}{\alpha}}\;\ln\frac{2k}{\alpha},
$$

then for every analyst, $\mathbb E[\max_i |a_i - q_i(\mathcal D)|] \le 4\alpha$.
:::

::: {.proof}
1. *Privacy.* As in steps 1 and 2 of the proof of @cor-ada-gaussian, each answer is the Gaussian mechanism with budget $(\varepsilon_0, \alpha/(2k))$, and the corollary of advanced composition with $\varepsilon = \alpha$ and $\delta' = \alpha/2$ makes the mechanism $(\alpha, \alpha)$-DP against adaptive analysts.
2. *Expected worst noise.* For $\lambda > 0$, Jensen's inequality and $e^{\lambda\max_i|Z_i|} \le \sum_i e^{\lambda |Z_i|}$ give

$$
e^{\lambda\,\mathbb E[\max_i |Z_i|]} \le \sum_{i=1}^k \mathbb E\big[e^{\lambda|Z_i|}\big] \le 2k\,e^{\lambda^2\sigma^2/2},
$$

because $\mathbb E[e^{\lambda|Z|}] \le \mathbb E[e^{\lambda Z}] + \mathbb E[e^{-\lambda Z}] = 2e^{\lambda^2\sigma^2/2}$. The logarithm, with $\lambda = \sqrt{2\ln(2k)}/\sigma$, gives $\mathbb E[\max_i|Z_i|] \le \sigma\sqrt{2\ln(2k)}$.
3. *The threshold.* Write $\ell = \ln(2k/\alpha) \ge \ln 2$. Since $\varepsilon_0 \le 1/(2\sqrt{2\ln 2}) < \ln 2 \le \ell$, we have $2\ell + \varepsilon_0 \le 3\ell$, and $\ln(2k) \le \ell$. So

$$
\sigma\sqrt{2\ln(2k)} \le \frac{\sqrt6\,\ell}{\varepsilon_0 n} = \frac{4\sqrt3}{\alpha n}\sqrt{k\ln\frac{2}{\alpha}}\;\ell ,
$$

which the condition on $n$ makes at most $\alpha$. Hence the expected worst error on the sample is at most $\alpha$.
4. @thm-ada-easy with $\varepsilon = \delta = \alpha$ gives at most $\alpha + (e^\alpha - 1) + \alpha \le 4\alpha$, since $e^\alpha - 1 \le 2\alpha$ for $\alpha \le 1$.
:::

## Summary {#sec-ada-summary}

| Result | Statement |
| --- | --- |
| @prp-ada-nonadaptive | $k$ fixed queries generalize once $n \ge \ln(2k/\beta)/(2\alpha^2)$ |
| @prp-ada-overfit | exact answers to $k$ adaptive queries can be off by $1 - n/(k-1)$ on the population |
| @prp-ada-split | sample splitting generalizes once $n \ge k\ln(2k/\beta)/(2\alpha^2)$ |
| @thm-ada-transfer | an $(\alpha/64, \alpha\beta/32)$-DP mechanism with error $\alpha/8$ on the sample, except with probability $\alpha\beta/16$, has error $\alpha$ on the population, except with probability $\beta$ |
| @cor-ada-gaussian | Gaussian answers generalize with $n$ of order $\sqrt k$ up to logarithmic factors |
| @lem-ada-key | a query chosen by an $(\varepsilon,\delta)$-DP algorithm has $\lvert\mathbb E[q(D)] - \mathbb E[q(\mathcal D)]\rvert \le e^\varepsilon - 1 + \delta$ |
| @thm-ada-easy | expected worst error on the population is at most $\alpha + e^\varepsilon - 1 + \delta$ |
| @cor-ada-expected | Gaussian answers give expected worst error $4\alpha$ with $n$ of order $\sqrt k$ |

: The chapter's results. {#tbl-ada-summary}

## References {#sec-ada-notes}

- John P. A. Ioannidis, "Why Most Published Research Findings Are False", 2005, on the reproducibility problem that motivates the chapter.
- Cynthia Dwork, Vitaly Feldman, Moritz Hardt, Toniann Pitassi, Omer Reingold and Aaron Roth, "Preserving Statistical Validity in Adaptive Data Analysis", 2015, the first use of DP for adaptively chosen queries with guarantees on the population.
- Raef Bassily, Kobbi Nissim, Adam Smith, Thomas Steinke, Uri Stemmer and Jonathan Ullman, "Algorithmic Stability for Adaptive Data Analysis", 2016, with the transfer theorem (Theorem 3.4) and the bound of @lem-ada-key (Lemma 3.1).
- Christopher Jung, Katrina Ligett, Seth Neel, Aaron Roth, Saeed Sharifi-Malvajerdi and Moshe Shenfeld, "A New Analysis of Differential Privacy's Generalization Guarantees", 2020, with a simpler and stronger transfer theorem.
- Moritz Hardt and Jonathan Ullman, "Preventing False Discovery in Interactive Data Analysis Is Hard", 2014, and Thomas Steinke and Jonathan Ullman, "Interactive Fingerprinting Codes and the Hardness of Preventing False Discovery", 2015, which show that efficient mechanisms cannot answer many more than $n^2$ adaptive queries, if one-way functions with exponential hardness exist.
- Cynthia Dwork, Guy N. Rothblum and Salil Vadhan, "Boosting and Differential Privacy", 2010, which proves advanced composition.
- Cynthia Dwork and Aaron Roth, *The Algorithmic Foundations of Differential Privacy*, 2014, on the Gaussian mechanism and advanced composition (Theorem 3.20).
