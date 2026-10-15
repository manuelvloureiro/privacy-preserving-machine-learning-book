# Exercises: Memorization and training-data extraction

Exercises for [Memorization and training-data extraction](../../book-full/part02-attacks/05-memorization-extraction.md), each with a worked solution. Try each exercise before opening its solution. The script `code/part02-attacks/solutions_memorization.py` computes every number quoted here; run it with `uv run python code/part02-attacks/solutions_memorization.py`. With the default seed it rebuilds the chapter's contact list and its canary "my pin is 58745".

The notation is the chapter's: $\mathcal R$ is the canary's randomness space, the rank of a candidate counts the candidates at least as likely (itself included), and exposure is $\log_2|\mathcal R| - \log_2\operatorname{rank}$. $N(c, w)$ are the counts of an $n$-gram model, $\alpha = 0.1$ its smoothing constant and $V = 39$ its alphabet size. In a written-out context, `_` stands for a space.

## Expected exposure of a small space {#sec-sol-memo-1}

> **Exercise 1.** **Expected exposure of a small space.** For $|\mathcal R| = 10$ and a canary that was not inserted (no ties), compute $\mathbb E[\mathrm{exposure}]$ from @eq-memo-null-mean and $\Pr[\mathrm{exposure} \ge 2]$ exactly. *Check: 1.1428 bits and 0.2.* *(Null distribution of exposure)*

::: {.callout-tip collapse="true" title="Solution"}
1. **The rank is uniform.** The canary was not inserted, so the model does not depend on it, and with no ties its rank is uniform on $\{1, \dots, 10\}$ (step 2 of the null-distribution derivation).
2. **The mean.** $\mathbb E[\log_2\operatorname{rank}] = \frac{1}{10}\sum_{i=1}^{10}\log_2i = \frac{\log_2(10!)}{10} = \frac{\log_2 3{,}628{,}800}{10} = \frac{21.791}{10} = 2.1791$.
3. **Subtract from $\log_2 10 = 3.3219$:** $\mathbb E[\mathrm{exposure}] = 3.3219 - 2.1791 = 1.1428$ bits. It is below the limit $\log_2e = 1.4427$: the $O(\ln N/N)$ correction in Stirling's formula is still large at $N = 10$.
4. **The tail.** Exposure $\ge 2$ means $\log_2\operatorname{rank} \le \log_210 - 2$, that is $\operatorname{rank} \le 10/4 = 2.5$. Ranks 1 and 2 qualify, so the probability is $2/10 = 0.2$, below the bound $2^{-2} = 0.25$. The bound is loose here because $10 \cdot 2^{-2}$ is not an integer.

*Exposure has the same law for every model when the canary is absent and there are no ties: it depends only on $|\mathcal R|$.*
:::

## Samples for extraction {#sec-sol-memo-2}

> **Exercise 2.** **Samples for extraction.** A secret has probability $p = 0.0442$ after its prefix. How many samples does the bound @eq-memo-samples ask for to extract it with probability 99%, and how many does the exact condition $(1 - p)^m \le 0.01$ ask for? *Check: 105 and 102.* *(Samples needed)*

::: {.callout-tip collapse="true" title="Solution"}
1. **The bound.** $m \ge \ln(1/0.01)/p = 4.6052/0.0442 = 104.19$, so $m = 105$.
2. **The exact condition.** $(1 - p)^m \le 0.01$ is $m\ln(1 - p) \le \ln 0.01$. Dividing by the negative $\ln(1 - p) = -0.04521$ flips the inequality: $m \ge 4.6052/0.04521 = 101.9$, so $m = 102$.
3. **Check.** With 102 samples the success probability is $1 - 0.9558^{102} = 0.9901$; with 101 it is 0.9896, just short of 99%.
4. **Why the bound is close.** $1 - p \le e^{-p}$ loses only a factor $e^{-p^2/2}$ per sample for small $p$ ($\ln(1 - p) = -p - p^2/2 - \dots$). The relative error of the bound is about $p/2 = 2\%$, which is the gap between 104.2 and 101.9.

*For rare strings the bound and the exact count agree; the useful rule is "about $4.6/p$ samples for 99%".*
:::

## A canary in unique contexts has maximal exposure {#sec-sol-memo-3}

> **Exercise 3.** **A canary in unique contexts has maximal exposure.** Suppose the order $n$ is large enough that every context met while scoring a pin after "my pin is" contains part of that prefix (for five digits and the end-of-line symbol, $n \ge 8$), and that the prefix occurs in the training data only in the $k \ge 1$ copies of the canary. Show that the canary is the unique most likely pin, so its exposure is $\log_2|\mathcal R|$. *Hint: compare the canary with a pin that first differs at position $j$; after position $j$ that pin's contexts were never seen, so each of its later symbols has probability $1/V$, and use @eq-memo-unseen.* *Check: the order-8 model of the script gives the canary of @tbl-memo-exposure exposure 16.61 at $k = 1$.* *(Fact 2, rank)*

::: {.callout-tip collapse="true" title="Solution"}
Write the canary as $\hat r = \hat r_1\dots\hat r_5$ and let $\hat r_6$ be the end-of-line symbol; a candidate $r$ is scored on the same six steps (@eq-memo-chain).

1. **Which contexts have counts.** Every context met while scoring contains part of "my pin is", so it can only have been seen in the canary's copies. The canary's own contexts were seen exactly $k$ times each, always followed by the canary's next symbol. A context that is not on the canary's path was never seen.
2. **Take a candidate $r \ne \hat r$** and let $j$ be the first position where $r_j \ne \hat r_j$. Positions before $j$ contribute the same factors to both scores, so only positions $j, \dots, 6$ matter.
3. **Position $j$.** Both are scored in the canary's context $c_j$. By @eq-memo-unseen the canary gets $\frac{k + \alpha}{k + \alpha V}$ and $r$ gets $\frac{\alpha}{k + \alpha V}$, strictly less.
4. **Positions after $j$.** The candidate's context now contains $r_j \ne \hat r_j$, so it is not on the canary's path and was never seen: $N(c) = 0$ and @eq-memo-ngram gives $\alpha/(\alpha V) = 1/V$ for every symbol. The canary keeps getting $\frac{k + \alpha}{k + \alpha V}$.
5. **Compare the factors.** $\frac{k + \alpha}{k + \alpha V} \ge \frac1V$ is equivalent to $V(k + \alpha) \ge k + \alpha V$, that is $Vk \ge k$, which holds. At $k = 1$: $0.2245 \ge 1/39 = 0.0256$.
6. **Conclude.** Every factor of the canary is at least the candidate's, and the one at position $j$ is strictly larger, so $p(\hat r) > p(r)$. This holds for every $r \ne \hat r$, so the canary has rank 1 and exposure $\log_2|\mathcal R| = \log_2 10^5 = 16.61$.

**On the chapter's corpus.** No background line contains `s_`, because names end in a vowel, so the contexts of the order-8 model after "my pin is" all contain `s_` and the assumptions hold. The script gives exposure 16.61 at $k = 1$ for order 8, and also for order 6, where the assumption fails at the last steps but the canary still wins.

*A model with enough context to give a secret its own path memorizes it completely from one copy. Exposure saturates, and the only remaining question is whether an attacker can find the prefix.*
:::

## Copies needed by hand {#sec-sol-memo-4}

> **Exercise 4.** **Copies needed by hand.** A secret's five steps have counts (true digit, strongest rival) without the secret of $(1, 2), (0, 2), (0, 5), (0, 1), (0, 1)$, and every rival has a lower code than the true digit. What is $k^*$? Which step decides it? If a larger model split that context so that the rival count at the third step fell from 5 to 2, what would $k^*$ become? *Check: 6, then 3.* *(@eq-memo-kstar)*

::: {.callout-tip collapse="true" title="Solution"}
1. **Ties go to the rival.** Greedy decoding breaks ties toward the lowest code, and every rival has a lower code, so the true digit must win strictly at every step: $k \ge \text{rival} - \text{true} + 1$.
2. **Per step.** $2 - 1 + 1 = 2$, $2 - 0 + 1 = 3$, $5 - 0 + 1 = 6$, $1 - 0 + 1 = 2$, $1 - 0 + 1 = 2$.
3. **The maximum** is $k^* = 6$, set by the third step. With 5 copies the first two digits come out right, the third goes to the rival, and decoding leaves the secret's path.
4. **After the split** the third step needs $2 - 0 + 1 = 3$, so the steps need $2, 3, 3, 2, 2$ and $k^* = 3$. The bottleneck is now shared by the second and third steps.
5. **If ties went the other way** (rivals with higher codes), every requirement would drop by one and $k^*$ would be 5 before the split.

*A larger model helps the secret only where it splits the bottleneck context. Splitting an easy context changes nothing.*
:::

## Exposure as a membership test over many canaries {#sec-sol-memo-5}

> **Exercise 5.** **Exposure as a membership test over many canaries.** Plant 20 canaries "my pin is $r_i$", one copy each, in the training set of the order-3 model of the chapter, and draw 20 more pins that you do not plant. Declare "planted" when the exposure is at least 3 bits. Use @eq-memo-null to bound the false positive rate, then measure both rates. Why is the detection so much weaker than for the single canary of @tbl-memo-exposure, which reached 4.05 bits with one copy? *Check: the bound is $2^{-3} = 0.125$; the script measures a true positive rate of 0.25 and a false positive rate of 0.15 on the 20 unplanted pins, and 0.1250 over all 99,980 unplanted pins.* *(Exposure, hypothesis testing)*

::: {.callout-tip collapse="true" title="Solution"}
1. **The bound.** An unplanted pin is independent of the model, so by @eq-memo-null its exposure is at least 3 with probability at most $2^{-3} = 0.125$. That is the false positive rate of the test "exposure $\ge 3$".
2. **Measured.** All 40 exposures come from one model trained with the 20 planted canaries. Planted pins: 5 of 20 reach 3 bits, a true positive rate of 0.25, and their mean exposure is 1.82 bits. Unplanted pins: 3 of 20 reach it, 0.15, with mean exposure 1.64. Twenty pins is a small sample: over all 99,980 unplanted pins the fraction is exactly 0.1250, on the bound, because the scores have no ties and the rank of a random pin is uniform.
3. **Why the test is weak.** At order 3 only the first digit of a canary has a context that the background never uses, `s_`. With one canary, that context sends its whole count to the canary's first digit, worth 3.46 bits (@tbl-memo-unseen). With 20 canaries, the context `s_` is followed 20 times by 20 first digits spread over 10 values. Each canary's first digit gets about 2 of the 20 counts, and it competes with the first digits of the other 19 canaries instead of with nothing. Its advantage over a random first digit shrinks from a factor of 11 to about $(2 + \alpha)/(\text{average count} + \alpha) \approx 1$. The remaining digits share their contexts with 5,000 background pins, where one copy is a negligible vote.
4. **Reading it as a test.** At false positive rate 0.125 the test finds a quarter of the planted canaries: barely better than chance. Auditing with many canaries needs either canaries that do not share contexts (different prefixes) or more copies.

*Canaries planted with the same prefix compete for the same contexts. Their exposures measure memorization of the set, not of each one.*
:::

## Which secrets an unprompted sampler finds {#sec-sol-memo-6}

> **Exercise 6.** **Which secrets an unprompted sampler finds.** By the chain rule, one sample from the start of a line equals a given training line with probability $p(\text{line})$. Predict the fraction of the secrets of each duplication group that appear at least once among 20,000 samples from the order-6 model (trained without the public lines), and compare with a simulation. *Check: for the secrets with 16 copies the mean $p(\text{line})$ is $1.34 \times 10^{-4}$, the predicted fraction 0.91 and the simulated one 0.96; for 4 copies, 0.01 and 0.02.* *(Chain rule, samples needed)*

::: {.callout-tip collapse="true" title="Solution"}
1. **One sample.** A temperature-1 sample draws each symbol from $p(x_t \mid x_{<t})$ and stops at the end-of-line symbol, so it equals a given line, end of line included, with probability $p(\text{line})$ (@eq-memo-chain).
2. **Many samples.** The samples are independent, so a line with probability $p$ appears at least once among $m$ of them with probability $1 - (1 - p)^m$, as in the derivation of @eq-memo-samples.
3. **Average over each group.** For each secret compute $p$ under the model and average $1 - (1 - p)^{20{,}000}$ over the 100 secrets of the group. Averaging the probabilities, not plugging in the mean $p$, matters: the per-secret probabilities differ by orders of magnitude.
4. **Compare** (@tbl-sol-memo-sampling).

| Copies $k$ | Mean $p(\text{line})$ | Expected appearances, $20{,}000\,p$ | Predicted fraction | Simulated fraction |
| --- | --- | --- | --- | --- |
| 1 | $1.15 \times 10^{-10}$ | 0.000 | 0.00 | 0.00 |
| 4 | $4.81 \times 10^{-7}$ | 0.010 | 0.01 | 0.02 |
| 8 | $1.23 \times 10^{-5}$ | 0.246 | 0.22 | 0.25 |
| 16 | $1.34 \times 10^{-4}$ | 2.680 | 0.91 | 0.96 |
| 32 | $7.52 \times 10^{-4}$ | 15.049 | 1.00 | 1.00 |

: Secret lines generated by 20,000 unprompted samples of the order-6 model: predicted from the line probabilities, and simulated. {#tbl-sol-memo-sampling}

5. **Why the probability climbs so fast with $k$.** The line's probability is a product over its 13 symbols. The name's first symbols are drawn in contexts shared with every line (the start of a line, then one or two letters). Each copy is one vote in each of those contexts, so each factor grows with $k$, and the product of several growing factors grows faster than $k$: from 4 to 16 copies the mean probability rises by a factor of 280. Once the name reaches contexts of its own, the remaining factors approach 1 (@tbl-memo-unseen).

The predictions agree with the simulation within the sampling error of 100 secrets per group. They also explain the chapter's attack: the 4-copy secrets have a 1 in 100 chance each, so almost none appear, although 76% of them are discoverably memorized at order 6.

*An unprompted attacker meets a secret at the rate $p(\text{line})$, which is the product of every vote along its path, prefix included. Holding the prefix removes most of that product.*
:::
