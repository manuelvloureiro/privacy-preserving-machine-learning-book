# What differential privacy protects

Machine learning on sensitive data can produce invasive results, but not every such result violates privacy. This chapter distinguishes inferences that differential privacy (DP) must prevent from those it permits. A private learner can predict a sensitive label from population patterns, but should not memorize a secret found only in one person's records.

## What the definition promises {#sec-viol-checklist}

Recall the definition from [the randomized response chapter](../part02-dp-basics/01-randomized-response.md). Neighbours $D\sim D'$ differ in one record. A mechanism $M$ is $\varepsilon$-DP if, for every such pair and output event $S$,

$$
\Pr[M(D)\in S] \le e^\varepsilon \Pr[M(D')\in S].
$$

The output distribution changes little when one record changes. Two properties from [the Laplace mechanism chapter](../part02-dp-basics/02-laplace-mechanism.md) translate this guarantee to attackers. **Post-processing** preserves $\varepsilon$-DP when a function uses only the mechanism's output and independent randomness. **Group privacy** extends the bound to datasets differing in $k$ entries:

$$
\Pr[M(D)\in S] \le e^{k\varepsilon}\Pr[M(D')\in S].
$$

Write $D^{(i\leftarrow z)}$ for $D$ with record $i$ replaced by $z$. An attacker is any possibly randomized function $\mathcal A$ of the output that returns $1$ when it detects a specified harm. It may use all other records and any side information; these are fixed inputs to $\mathcal A$. The next proposition bounds the attacker's success when one record is replaced.

::: {#prp-viol-participation name="What an attacker gains from one record"}
Let $M$ be $\varepsilon$-DP, and let $\mathcal A$ be a randomized function of the output whose coins are independent of the coins of $M$. Then for every dataset $D$, every index $i$ and every record $z$,

$$
\Pr\big[\mathcal A(M(D)) = 1\big] \le e^{\varepsilon}\,\Pr\big[\mathcal A(M(D^{(i\leftarrow z)})) = 1\big].
$$
:::

::: {.proof}
1. The composed algorithm $\mathcal A \circ M$ is $\varepsilon$-DP by post-processing.
2. The datasets $D$ and $D^{(i\leftarrow z)}$ differ in at most one entry, so they are neighbours or equal. The definition, applied to the event $\{1\}$, gives the inequality.
:::

Thus, replacing a person's record with a placeholder changes the probability of an output-caused harm by at most the DP factor. DP does not prevent harms that remain likely after replacement. To assess an alleged violation, ask:

1. Is the result substantially different if one person's record is changed or left out?
2. Is the harmful inference the same thing as the accuracy of the learned model?
3. If the inference still feels wrong, at what point, and by whom, was the information given away?

The second question is central. If an inference is exactly what an accurate model predicts, preventing it requires releasing an inaccurate model, such as one that ignores the data. Such a model is uninformative.

## Learning a rule is not a violation {#sec-viol-rule}

A learning algorithm takes training pairs $(x_i,y_i)$, where $x_i$ is a **feature vector** and $y_i$ is a **label**. It outputs a **classifier**, which maps feature vectors to predicted labels. Under DP, the training set is private and the classifier is the protected output; test points used to measure accuracy are not protected.

Consider real-valued training pairs

$$
(0, 1),\ (2, 3),\ (5, 6),\ (-6, -5),\ (0.5, 1.5),\ (\pi, \pi+1),\ \dots
$$

Every label is one more than its feature. A learner can infer the rule $y=x+1$, predict $5$ at $x=4$, and predict $\pi+1$ at $x=\pi$. The latter prediction matches a training label, but it is not a privacy violation: the rule holds across the population, so the prediction does not depend on whether $(\pi,\pi+1)$ was collected.

A private algorithm makes this distinction explicit. Each record determines an offset $y_i-x_i$, and a rule $y=x+b$ specifies an offset. The stability-based histogram from [the chapter on beyond global sensitivity](../part03-query-release/03-beyond-global-sensitivity.md) can release frequently occurring offsets even though the domain is continuous. Its guarantee is approximate DP, defined in [the approximate DP chapter](../part02-dp-basics/03-approximate-dp.md) by adding slack $\delta$:

$$
\Pr[M(D)\in S] \le e^\varepsilon\Pr[M(D')\in S] + \delta \quad\text{for all } D \sim D' \text{ and all } S.
$$

The algorithm adds Laplace noise to each observed offset's count and releases offsets whose noisy counts exceed a threshold.

::: {#def-viol-histogram name="Stable histogram of offsets"}
Let $\varepsilon > 0$ and $\delta \in (0,1)$, and set the threshold

$$
\tau = 1 + \frac{2\ln(1/\delta)}{\varepsilon}.
$$

On input $D = ((x_1,y_1),\dots,(x_n,y_n))$, a list of $n$ records with real entries, let $c_b = |\{i : y_i - x_i = b\}|$ be the number of records whose offset is $b$. For each $b$ with $c_b > 0$, draw an independent $Z_b$ from the Laplace distribution $\mathrm{Lap}(2/\varepsilon)$ with scale $2/\varepsilon$, and output the set

$$
M(D) = \{\, b : c_b > 0 \text{ and } c_b + Z_b > \tau \,\}.
$$
:::

This applies the stability-based histogram to offsets $y_i-x_i$, then removes the noisy counts from the output. Zero-count bins are not considered, and threshold $\tau$ suppresses a bin supported by one record except with small probability. The resulting mechanism retains the histogram's privacy guarantee.

::: {#thm-viol-histogram}
The stable histogram of offsets is $(\varepsilon,\delta)$-DP.
:::

::: {.proof}
1. Replacing one record changes at most one offset, so when $D \sim D'$ the two lists of offsets are neighbours or equal.
2. The stability-based histogram is $(\varepsilon,\delta)$-DP on datasets of offsets, by the theorem of that chapter.
3. The output $M(D)$ is the set of elements on the histogram's list, a function of that output alone. So it is $(\varepsilon,\delta)$-DP by post-processing, which holds for approximate DP as well.
:::

The privacy cost does not depend on how many records follow the rule. Accuracy requires enough records for the offset count to exceed the threshold.

::: {#prp-viol-rule}
Suppose every record satisfies $y_i = x_i + 1$, let $\tau$ be the threshold of @def-viol-histogram, and let $\beta \in (0, 1/2)$ be a failure probability. If

$$
n \ge \tau + \frac{2}{\varepsilon}\ln\frac{1}{2\beta},
$$

then with probability at least $1 - \beta$ the output is exactly $\{1\}$, and the released rule predicts $y = x + 1$ at every feature value.
:::

::: {.proof}
1. Only the offset $1$ has a positive count, $c_1 = n$, because all $n$ records have that offset. So the output is either $\{1\}$ or empty.
2. The output is empty when the Laplace noise $Z_1$ on that count satisfies $n + Z_1 \le \tau$. A $\mathrm{Lap}(2/\varepsilon)$ variable $Z$ has $\Pr[Z \le -s] = \tfrac12 e^{-\varepsilon s/2}$ for every $s \ge 0$. With $s = n - \tau$, which is not negative because $n \ge \tau$, this gives
$$
\Pr[Z_1 \le -(n-\tau)] = \tfrac12 e^{-\varepsilon(n-\tau)/2}.
$$
3. The assumed bound on $n$ makes this at most $\beta$.
:::

::: {#exm-viol-rule}
Take $\varepsilon = 1$, $\delta = 10^{-6}$ and $\beta = 0.01$. With these values the threshold is $\tau = 1 + 2\ln(10^{6}) = 1 + 2 \times 13.82 = 28.63$. The bound on $n$ adds $2\ln 50 = 7.82$ to it, which gives $36.45$. So $37$ records that follow the rule are enough for the private algorithm to output it with probability at least $0.99$.
:::

Thus, @prp-viol-participation protects record $(\pi,\pi+1)$ even though the rule predicts its label exactly. The prediction follows from a population-level rule. The relevant disclosure occurred when the feature value $\pi$ became known; anyone who knows the rule can then calculate the label.

## Inferences from what people reveal {#sec-viol-inference}

The same distinction applies to a study of smoking and cancer. A smoker may fear higher insurance premiums if the study publishes an association. Here, "smokes" is the feature and "likely to develop cancer" is the label. The study would reach the same conclusion without this person. Preventing the resulting inference would require suppressing the population-level finding; the person's smoking status was disclosed separately.

## What a violation looks like {#sec-viol-violations}

A privacy violation is an inference that depends on the person's sensitive record being in the dataset.

**The Netflix Prize.** Netflix released dated movie ratings under pseudonymous identifiers. Narayanan and Shmatikov linked them to public reviews on the Internet Movie Database and identified ratings that matched users had not made public. This reidentification depends on the users' records. By contrast, inferring that someone watched the second film in a trilogy after seeing their public reviews of the first and third uses only information they chose to publish.

**Memorized secrets.** Carlini and coauthors trained language models on text containing secrets, such as a nine-digit identification number entered several times by one person. The model assigned the secret unusually high likelihood, enabling score-guided extraction. This is a violation because language patterns alone do not favour one nine-digit number; the preference comes from the person's records. DP and group privacy bound this risk, as the next proposition shows.

::: {#prp-viol-secret name="Secrets are not memorized"}
Let $\mathcal S$ be a finite set of $N$ possible secrets. For each $s \in \mathcal S$, let $D(s)$ be a training set in which $k$ entries depend on $s$ and the other entries do not. Let $M$ be $\varepsilon$-DP, and let $G$ be any guessing rule, with coins independent of $M$, that maps the trained model to an element of $\mathcal S$. If the secret $s$ is drawn uniformly from $\mathcal S$, then

$$
\Pr\big[G(M(D(s))) = s\big] \le \frac{e^{k\varepsilon}}{N}.
$$
:::

::: {.proof}
1. Let $D_0$ be the training set with the $k$ entries that depend on the secret replaced by fixed placeholder records. For every $s$, the datasets $D(s)$ and $D_0$ differ in at most $k$ entries.
2. By post-processing and group privacy, $\Pr[G(M(D(s))) = s] \le e^{k\varepsilon}\Pr[G(M(D_0)) = s]$ for every $s$.
3. The guess on $D_0$ does not depend on $s$, and it equals at most one secret, so $\sum_{s\in\mathcal S}\Pr[G(M(D_0)) = s] \le 1$.
4. The average of step 2 over the uniform secret, with step 3, gives
$$
\Pr\big[G(M(D(s))) = s\big] = \frac1N\sum_{s\in\mathcal S}\Pr\big[G(M(D(s))) = s\big] \le \frac{e^{k\varepsilon}}{N}.
$$
:::

For a nine-digit number, $N=10^9$. If $\varepsilon=1$ and the secret appears $k=4$ times, the attacker's success probability is at most $e^4/10^9=5.5\times10^{-8}$, compared with $10^{-9}$ for a blind guess. The bound grows exponentially with $k$, so repeated contributions require a user-level rather than record-level guarantee.

## Summary {#sec-viol-summary}

| Result | Statement |
| --- | --- |
| @prp-viol-participation | an attacker succeeds about as often when a person's record is replaced by any other |
| @thm-viol-histogram | releasing offsets whose noisy count exceeds $\tau$ is $(\varepsilon,\delta)$-DP |
| @prp-viol-rule | with $n \ge \tau + \frac2\varepsilon\ln\frac1{2\beta}$ records, the private algorithm finds $y = x+1$ |
| @prp-viol-secret | a secret in $k$ records of one person is guessed with probability at most $e^{k\varepsilon}/N$ |

: The chapter's results. {#tbl-viol-summary}

## References {#sec-viol-notes}

- Cynthia Dwork and Aaron Roth, *The Algorithmic Foundations of Differential Privacy*, 2014, whose Section 2.3 discusses what the definition promises and the smoking example.
- Arvind Narayanan and Vitaly Shmatikov, "Robust De-anonymization of Large Sparse Datasets", 2008, the linkage attack on the Netflix Prize data.
- Nicholas Carlini, Chang Liu, Úlfar Erlingsson, Jernej Kos and Dawn Song, "The Secret Sharer: Evaluating and Testing Unintended Memorization in Neural Networks", 2019, which extracts secrets that appear in few training records.
- Abhradeep Thakurta and Adam Smith, "Differentially Private Feature Selection via Stability Arguments, and the Robustness of the Lasso", 2013, and Salil Vadhan, "The Complexity of Differential Privacy", 2017, Section 3, on stability-based algorithms such as the histogram behind @thm-viol-histogram.
