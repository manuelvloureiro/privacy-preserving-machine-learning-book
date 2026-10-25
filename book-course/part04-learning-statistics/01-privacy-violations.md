# What Differential Privacy Protects

Machine learning on sensitive data often produces results that feel invasive, and it is tempting to call each of them a privacy violation. This chapter separates the inferences that differential privacy (DP) is meant to prevent from those it is designed to allow. It gives a test for telling them apart, and proves the two facts behind it. A private learner can still discover a rule that predicts a person's sensitive label. A private learner cannot memorize a secret that appears only in one person's records.

## What the definition promises {#sec-viol-checklist}

Recall the definition from [the randomized response chapter](../part02-dp-basics/01-randomized-response.md). A mechanism $M$ is $\varepsilon$-DP if, for all pairs of neighbouring datasets $D \simeq D'$, which differ by replacing one record, and all sets of outputs $S$,

$$
\Pr[M(D)\in S] \le e^\varepsilon \Pr[M(D')\in S].
$$

The definition says that the output changes little when one record changes, and it says nothing else. In particular, it does not stop statistics or learning from working, nor anything that follows from their results.

Two properties from [the Laplace mechanism chapter](../part02-dp-basics/02-laplace-mechanism.md) turn the definition into a statement about attackers. **Post-processing** says that a function of the output of an $\varepsilon$-DP mechanism, computed with coins independent of the data, is again $\varepsilon$-DP. **Group privacy** says that if $D$ and $D'$ differ in some number $k$ of entries, then

$$
\Pr[M(D)\in S] \le e^{k\varepsilon}\Pr[M(D')\in S].
$$

Write $D^{(i \leftarrow z)}$ for the dataset $D$ with entry $i$ replaced by a record $z$. An attacker is any function $\mathcal A$ of the output, possibly randomized, that answers $1$ when it believes it has learned something harmful. The attacker may know every other record and any side information, since all of that is fixed in $\mathcal A$.

The first fact behind the test says how much such an attacker gains from one person's record. It compares the attacker's chance of answering $1$ on the real dataset with its chance when that person's record is replaced by any other.

::: {#prp-viol-participation name="What an attacker gains from one record"}
Let $M$ be $\varepsilon$-DP, and let $\mathcal A$ be a randomized function of the output whose coins are independent of the coins of $M$. Then for every dataset $D$, every index $i$ and every record $z$,

$$
\Pr\big[\mathcal A(M(D)) = 1\big] \le e^{\varepsilon}\,\Pr\big[\mathcal A(M(D^{(i\leftarrow z)})) = 1\big].
$$
:::

::: {.proof}
1. The composed algorithm $\mathcal A \circ M$ is $\varepsilon$-DP by post-processing.
2. The datasets $D$ and $D^{(i\leftarrow z)}$ differ in at most one entry, so they are neighbours or equal. Applying the definition to the event $\{1\}$ gives the inequality.
:::

So any harm the output can cause a person is about as likely when their record is swapped for an arbitrary placeholder. An inference that would succeed just as well without the record is not something the definition tries to stop. This suggests three questions to ask of an alleged violation.

1. Would the result be substantially different if one person's record were changed or left out?
2. Is the harmful inference the same thing as the learned model being accurate?
3. If the inference still feels wrong, at what point, and by whom, was the information given away?

The second question deserves a word. If an inference about a person is exactly what an accurate model predicts, the only way to prevent it is to release an inaccurate model, for instance one that ignores the data. That is always possible and always useless.

## Learning a rule is not a violation {#sec-viol-rule}

A learning algorithm takes a training set of pairs $(x_i, y_i)$, where $x_i$ is a **feature vector** and $y_i$ a **label**, and outputs a **classifier**, a function that maps a feature vector to a predicted label. Its quality is measured on test points, usually as the fraction of test labels it predicts correctly. In the private setting, the training set is the sensitive dataset and the classifier is the output that must be differentially private. Test points are not protected. Some work protects only the classifier's predictions instead of the classifier itself.

Consider a deliberately simple training set of real pairs,

$$
(0, 1),\ (2, 3),\ (5, 6),\ (-6, -5),\ (0.5, 1.5),\ (\pi, \pi+1),\ \dots
$$

in which every label is one more than its feature. Any reasonable learner finds the rule $y = x + 1$. Asked for the label of $x = 4$, which is not in the training set, it answers $5$, and no one would call that a violation. Asked for the label of $x = \pi$, it answers $\pi + 1$, which is the true label of a training record. That is not a violation either. The learner found a fact about the population, and the answer at $x = \pi$ would be the same if the record $(\pi, \pi+1)$ had never been collected.

The claim can be made exact. Each record determines an offset $y_i - x_i$, and a rule of the form $y = x + b$ is a choice of offset. The stability-based histogram of [the chapter on beyond global sensitivity](../part03-query-release/03-beyond-global-sensitivity.md) releases the offsets that many records share, even though offsets are real numbers and there are infinitely many possible bins. Its guarantee is approximate differential privacy, $(\varepsilon,\delta)$-DP, from [the approximate DP chapter](../part02-dp-basics/03-approximate-dp.md), which allows the bound of the definition an additive slack $\delta$:

$$
\Pr[M(D)\in S] \le e^\varepsilon\Pr[M(D')\in S] + \delta \quad\text{for all } D \simeq D' \text{ and all } S.
$$

The algorithm counts how many records share each offset, adds Laplace noise to each count, and releases the offsets whose noisy count clears a threshold. A threshold that is high enough keeps an offset that only one record holds from being released.

::: {#def-viol-histogram name="Stable histogram of offsets"}
Let $\varepsilon > 0$ and $\delta \in (0,1)$, and set $\tau = 1 + 2\ln(1/\delta)/\varepsilon$. On input $D = ((x_1,y_1),\dots,(x_n,y_n))$ with real entries, let $c_b = |\{i : y_i - x_i = b\}|$ be the number of records whose offset is $b$. For each $b$ with $c_b > 0$, draw an independent $Z_b$ from the Laplace distribution $\mathrm{Lap}(2/\varepsilon)$ with scale $2/\varepsilon$, and output the set

$$
M(D) = \{\, b : c_b > 0 \text{ and } c_b + Z_b > \tau \,\}.
$$
:::

This is the stability-based histogram run on the offsets $y_i - x_i$, with the noisy counts dropped from its output. Bins with count zero are never released, so the output only ever names offsets that occur in the data. The threshold $\tau$ keeps a bin held by a single record from being released, except with small probability. The privacy guarantee is inherited from the histogram.

::: {#thm-viol-histogram}
The stable histogram of offsets is $(\varepsilon,\delta)$-DP.
:::

::: {.proof}
1. Replacing one record changes at most one offset, so when $D \simeq D'$ the two lists of offsets are neighbours or equal.
2. The stability-based histogram is $(\varepsilon,\delta)$-DP on datasets of offsets, by the theorem of that chapter.
3. $M(D)$ is the set of elements on its list, a function of its output alone, so it is $(\varepsilon,\delta)$-DP by post-processing, which holds for approximate DP as well.
:::

The privacy cost does not depend on how many records follow the rule. Accuracy, on the other hand, needs only enough of them to clear the threshold.

::: {#prp-viol-rule}
Suppose every record satisfies $y_i = x_i + 1$, let $\tau$ be the threshold of @def-viol-histogram, and let $\beta \in (0, 1/2)$ be a failure probability. If

$$
n \ge \tau + \frac{2}{\varepsilon}\ln\frac{1}{2\beta},
$$

then with probability at least $1 - \beta$ the output is exactly $\{1\}$, and the released rule predicts $y = x + 1$ at every feature value.
:::

::: {.proof}
1. Only the offset $1$ has a positive count, $c_1 = n$, so the output is either $\{1\}$ or empty.
2. It is empty when $n + Z_1 \le \tau$, which has probability $\Pr[Z_1 \le -(n-\tau)] = \tfrac12 e^{-\varepsilon(n-\tau)/2}$, because $n \ge \tau$ and a $\mathrm{Lap}(2/\varepsilon)$ variable satisfies $\Pr[Z \le -s] = \tfrac12 e^{-\varepsilon s/2}$ for $s \ge 0$.
3. The assumed bound on $n$ makes this at most $\beta$.
:::

Concrete values show how few records the rule needs.

::: {#exm-viol-rule}
Take $\varepsilon = 1$, $\delta = 10^{-6}$ and $\beta = 0.01$. The threshold is $\tau = 1 + 2\ln(10^{6}) = 1 + 2 \times 13.82 = 28.63$. Adding $2\ln 50 = 7.82$ gives $36.45$, so $37$ records that follow the rule suffice for the private algorithm to output it with probability at least $0.99$.
:::

So the record $(\pi, \pi+1)$ is protected in the sense of @prp-viol-participation, and yet the released rule predicts its label exactly. The prediction is a consequence of the rule, and the rule is a fact about nature. If this still feels like a leak, ask the third question. The information was given away when someone revealed that a person's feature value is $\pi$. From then on, anyone who knows the rule can work out the label.

## Inferences from what people reveal {#sec-viol-inference}

The same pattern explains several examples that are often presented as privacy failures.

**Smoking and cancer.** A smoker decides whether to join a study of whether smoking causes cancer, and fears higher insurance premiums if the link is published. This is the rule $y = x + 1$ in another form. The feature is "smokes", the label is "likely to develop cancer", and the study would reach the same conclusion without this person. The only way to prevent the harm is not to run the study. Privacy was given up when the person's smoking became known.

**A cast.** Stripped of the study, the point is plainer. Someone who walks around with a cast on their foot probably has a broken foot. They may regard the injury as private, but they display the cast in public, and common knowledge links the two.

**Social network likes.** Kosinski, Stillwell and Graepel showed in 2013 that the pages a person likes on Facebook, which are public by default, predict sensitive traits such as sexual orientation and political views. The model was trained on public likes and on traits that many users state on their own profiles, so there was no sensitive dataset to protect, and the results would presumably hold under differential privacy too. Privacy was given up when the likes were made public. This example adds one lesson: whether a piece of information is revealing may become known only later, after it was shared in the belief that it was harmless.

**Targeted advertising.** A retailer is said to have mailed baby-product coupons to a teenager, having inferred her pregnancy from her purchases before her family knew. The story may be apocryphal. Even if it is true, the analysis was not the violation. The information was handed over when her purchases were linked to her identity through a loyalty programme, and paying anonymously would have prevented it.

## What a violation looks like {#sec-viol-violations}

A violation is an inference that depends on the person's own sensitive record being in the dataset.

**The Netflix Prize.** Netflix released movie ratings with dates under pseudonymous user identifiers. Narayanan and Shmatikov linked the records to public reviews on the Internet Movie Database and recovered, for the users they matched, films those users had rated but never reviewed in public. Had a user's record been withheld, that reidentification would have been impossible, so it fails the first question. Compare an analyst who sees only public reviews. If a user reviewed the first and third films of a trilogy, the analyst may guess that they watched the second. The guess may be right, but it uses only what the user chose to publish, so it is not a violation.

**Memorized secrets.** Carlini and coauthors trained language models on text that contained a secret, such as a nine-digit identification number, that one person had typed several times. The trained model assigned the secret an unusually high likelihood, and a search guided by those scores could extract it. This is a violation. A model that had learned only the language would have no reason to prefer one nine-digit number over another, so the preference comes from that person's records. Differential privacy rules it out, with group privacy accounting for the repetitions. The next proposition makes this exact. It bounds the chance that an attacker who sees the trained model guesses the secret, and compares it with blind guessing.

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
4. Averaging step 2 over the uniform secret and using step 3 gives
$$
\Pr\big[G(M(D(s))) = s\big] = \frac1N\sum_{s\in\mathcal S}\Pr\big[G(M(D(s))) = s\big] \le \frac{e^{k\varepsilon}}{N}.
$$
:::

For a nine-digit number, $N = 10^9$. With $\varepsilon = 1$ and a secret typed $k = 4$ times, the attacker's chance is at most $e^4/10^9 = 5.5\times10^{-8}$, against $10^{-9}$ for blind guessing. The bound degrades exponentially in $k$, which is why a person who contributes many records needs a guarantee per person rather than per record.

The three questions settle most cases. The most useful is the second. If an alleged violation is at the same time the model doing what it is supposed to do, then it is the notion of violation, not the model, that needs another look.

## Summary {#sec-viol-summary}

| Result | Statement |
| --- | --- |
| @prp-viol-participation | an attacker succeeds about as often when a person's record is replaced by any other |
| @thm-viol-histogram | releasing offsets whose noisy count exceeds $\tau$ is $(\varepsilon,\delta)$-DP |
| @prp-viol-rule | with $n \ge \tau + \frac2\varepsilon\ln\frac1{2\beta}$ records, the private algorithm finds $y = x+1$ |
| @prp-viol-secret | a secret in $k$ records of one person is guessed with probability at most $e^{k\varepsilon}/N$ |

: The chapter's results. {#tbl-viol-summary}

## References {#sec-viol-notes}

- Frank McSherry, "Statistical Inference Considered Harmful", 2016. A blog post arguing that inference from public facts, such as the cast, is not a privacy violation.
- Cynthia Dwork and Aaron Roth, *The Algorithmic Foundations of Differential Privacy*, 2014. Section 2.3 discusses what the definition promises and the smoking example.
- Michal Kosinski, David Stillwell and Thore Graepel, "Private Traits and Attributes Are Predictable from Digital Records of Human Behavior", 2013. Prediction of sensitive traits from public likes.
- Arvind Narayanan and Vitaly Shmatikov, "Robust De-anonymization of Large Sparse Datasets", 2008. The linkage attack on the Netflix Prize data.
- Nicholas Carlini, Chang Liu, Úlfar Erlingsson, Jernej Kos and Dawn Song, "The Secret Sharer: Evaluating and Testing Unintended Memorization in Neural Networks", 2019. Extraction of secrets that appear in few training records.
- Abhradeep Thakurta and Adam Smith, "Differentially Private Feature Selection via Stability Arguments, and the Robustness of the Lasso", 2013, and Salil Vadhan, "The Complexity of Differential Privacy", 2017, Section 3. Stability-based algorithms such as the histogram behind @thm-viol-histogram.
