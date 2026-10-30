# What differential privacy protects

Machine learning on sensitive data often gives results that feel invasive, but not each of them is a privacy violation. This chapter gives a test that separates the inferences that differential privacy (DP) must prevent from those it allows. Two facts support the test. A private learner can still find a rule that predicts a person's sensitive label, but it cannot memorize a secret that is only in one person's records.

## What the definition promises {#sec-viol-checklist}

Recall the definition from [the randomized response chapter](../part02-dp-basics/01-randomized-response.md). Two datasets $D \sim D'$ are neighbours if they differ by the replacement of one record. A mechanism $M$ is $\varepsilon$-DP if, for all such pairs and all sets $S$ of outputs,

$$
\Pr[M(D)\in S] \le e^\varepsilon \Pr[M(D')\in S].
$$

So the output changes little when one record changes, and the definition says nothing more. Two properties from [the Laplace mechanism chapter](../part02-dp-basics/02-laplace-mechanism.md) turn the definition into a statement about attackers. **Post-processing** says that a function of the output of an $\varepsilon$-DP mechanism, computed with coins independent of the data, is again $\varepsilon$-DP. **Group privacy** says that if $D$ and $D'$ differ in some number $k$ of entries, then

$$
\Pr[M(D)\in S] \le e^{k\varepsilon}\Pr[M(D')\in S].
$$

Write $D^{(i \leftarrow z)}$ for the dataset $D$ with entry $i$ replaced by a record $z$. An attacker is any function $\mathcal A$ of the output, possibly randomized, that answers $1$ when it believes it has learned something harmful. The attacker may know every other record and any side information, because all of that is fixed in $\mathcal A$. The first fact compares its chance to answer $1$ on the real dataset and on the dataset with one record replaced.

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

So a harm that the output causes a person is about as likely when a placeholder replaces their record. The definition does not try to stop that harm. This gives three questions to ask of an alleged violation.

1. Is the result substantially different if one person's record is changed or left out?
2. Is the harmful inference the same thing as the accuracy of the learned model?
3. If the inference still feels wrong, at what point, and by whom, was the information given away?

The second question matters most. If an inference about a person is exactly what an accurate model predicts, the only way to prevent it is to release an inaccurate model, for example one that ignores the data. That is always possible and always useless.

## Learning a rule is not a violation {#sec-viol-rule}

A learning algorithm takes a training set of pairs $(x_i, y_i)$, where $x_i$ is a **feature vector** and $y_i$ a **label**. It outputs a **classifier**, a function that maps a feature vector to a predicted label. In the private setting, the training set is the sensitive dataset, and the classifier is the output that must be differentially private. The test points that measure its accuracy are not protected.

Take a simple training set of real pairs,

$$
(0, 1),\ (2, 3),\ (5, 6),\ (-6, -5),\ (0.5, 1.5),\ (\pi, \pi+1),\ \dots
$$

in which every label is one more than its feature. Any reasonable learner finds the rule $y = x + 1$. At $x = 4$, which is not in the training set, it predicts $5$. At $x = \pi$, it predicts $\pi + 1$, which is the true label of a training record. That is not a violation, because the learner found a fact about the population. The prediction at $x = \pi$ is the same if the record $(\pi, \pi+1)$ was never collected.

A private algorithm makes this exact. Each record gives an offset $y_i - x_i$, and a rule $y = x + b$ is a choice of offset. The stability-based histogram of [the chapter on beyond global sensitivity](../part03-query-release/03-beyond-global-sensitivity.md) releases the offsets that many records share, although offsets are real numbers with infinitely many possible bins. Its guarantee is approximate DP, $(\varepsilon,\delta)$-DP, from [the approximate DP chapter](../part02-dp-basics/03-approximate-dp.md), which adds a slack $\delta$ to the bound of the definition:

$$
\Pr[M(D)\in S] \le e^\varepsilon\Pr[M(D')\in S] + \delta \quad\text{for all } D \sim D' \text{ and all } S.
$$

The algorithm adds Laplace noise to the count of each offset and releases the offsets whose noisy count is above a threshold.

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

This is the stability-based histogram on the offsets $y_i - x_i$, with the noisy counts dropped from its output. Bins with count zero are never released, and the threshold $\tau$ stops the release of a bin that one record holds, except with small probability. So the guarantee of the histogram carries over.

::: {#thm-viol-histogram}
The stable histogram of offsets is $(\varepsilon,\delta)$-DP.
:::

::: {.proof}
1. Replacing one record changes at most one offset, so when $D \sim D'$ the two lists of offsets are neighbours or equal.
2. The stability-based histogram is $(\varepsilon,\delta)$-DP on datasets of offsets, by the theorem of that chapter.
3. The output $M(D)$ is the set of elements on the histogram's list, a function of that output alone. So it is $(\varepsilon,\delta)$-DP by post-processing, which holds for approximate DP as well.
:::

The privacy cost does not depend on how many records follow the rule. Accuracy only needs enough of them to clear the threshold.

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

So @prp-viol-participation protects the record $(\pi, \pi+1)$, but the rule predicts the label of that record exactly. The prediction follows from the rule, and the rule is a fact about nature. If this still feels like a leak, the third question finds the point of the leak. It is the moment when someone revealed that the person's feature value is $\pi$, because after that anyone who knows the rule can calculate the label.

## Inferences from what people reveal {#sec-viol-inference}

The same pattern explains examples that people often present as privacy failures. Take a smoker who must decide whether to join a study of whether smoking causes cancer, and fears higher insurance premiums if the study publishes the link. This is the rule $y = x + 1$ in another form. The feature is "smokes", the label is "likely to develop cancer", and the study reaches the same conclusion without this person. The only way to prevent the harm is not to run the study. The person gave up privacy when their smoking became known.

## What a violation looks like {#sec-viol-violations}

A violation is an inference that depends on the presence of the person's own sensitive record in the dataset.

**The Netflix Prize.** Netflix released movie ratings with dates under pseudonymous user identifiers. Narayanan and Shmatikov linked them to public reviews on the Internet Movie Database, and so found films that the matched users had rated but never reviewed in public. Without a user's record, that reidentification is impossible, so it fails the first question. In contrast, an analyst who sees that a user reviewed the first and third films of a trilogy may guess that they watched the second. That guess uses only what the user chose to publish, so it is not a violation.

**Memorized secrets.** Carlini and coauthors trained language models on text with a secret, such as a nine-digit identification number, that one person had typed several times. The model gave the secret an unusually high likelihood, and a search guided by those scores could extract it. This is a violation. A model that learned only the language has no reason to prefer one nine-digit number to another, so the preference comes from that person's records. DP prevents it, with group privacy for the repetitions, as the next proposition shows.

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

For a nine-digit number, $N = 10^9$. With $\varepsilon = 1$ and a secret typed $k = 4$ times, the chance of the attacker is at most $e^4/10^9 = 5.5\times10^{-8}$, against $10^{-9}$ for a blind guess. The bound grows exponentially in $k$. So a person who contributes many records needs a guarantee per person, not per record.

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
