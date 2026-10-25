# Exercises: Adaptive data analysis

Exercises for [Adaptive data analysis](../../book-full/part05-dp-ml/11-adaptive-data-analysis.md), each with a worked solution. Try each exercise before opening its solution. The script `code/part05-dp-ml/solutions_adaptive_data_analysis.py` computes every number quoted here; run it with `uv run python code/part05-dp-ml/solutions_adaptive_data_analysis.py`.

The solutions use the chapter's notation. The data are a sample of $n$ records, the dataset $D$, drawn independently from a distribution $\mathcal D$. A statistical query $q$ maps a record to $[0, 1]$. Its empirical value $q(D)$ is its average over the sample, and its population value $q(\mathcal D)$ is its average over the distribution. A mechanism answers $k$ queries, possibly chosen adaptively, and $a_j$ is its answer to query $j$.

Privacy is differential privacy (DP) with privacy level $\varepsilon$ and slack $\delta$, under the replace-one relation, where a neighbour $D^{(i)}$ replaces record $x_i$ by a fresh draw $x_i'$. Zero-concentrated differential privacy (zCDP) with parameter $\rho$ converts to $(\varepsilon, \delta)$-DP with
$$\varepsilon = \rho + 2\sqrt{\rho\ln(1/\delta)}.$$

## How many point queries break a sample {#sec-sol-ada-1}

> **Exercise 1.** **How many point queries break a sample.** In the point-query attack of @sec-ada-overfit, the records are uniform on $N$ values and the final query asks whether a record's value occurs in the sample. Show that the final query's expected population value is $1 - (1 - 1/N)^n$ and that it is at most $n/N$. For $n = 1{,}000$, find the smallest $N$ for which the final query's expected error exceeds $\frac12$, and compare it with $n/\ln 2$. *Check: $N = 1{,}444$, so 1,445 queries, against $n/\ln 2 = 1{,}442.7$.* *(Counting, and the inequality $1 - x \le e^{-x}$)*

::: {.callout-tip collapse="true" title="Solution"}
1. **The population value.** The final query is 1 on a value exactly when that value occurs in the sample. A fresh record has each value with probability $1/N$, so the population value is the fraction of the $N$ values that the sample covers.
2. **Its expectation.** A given value is missed by one draw with probability $1 - 1/N$, and by all $n$ independent draws with probability $(1 - 1/N)^n$. So it is covered with probability $1 - (1 - 1/N)^n$, and by linearity of expectation the expected covered fraction is the same number.
3. **The upper bound.** The sample has $n$ records, so it covers at most $n$ values, and the covered fraction is at most $n/N$ for every sample, hence also on average.
4. **The error.** The final query's empirical value is 1, so its expected error is $(1 - 1/N)^n$. This exceeds $\frac12$ when $n\ln(1 - 1/N) > -\ln 2$.
5. **Solve.** For large $N$, $\ln(1 - 1/N) \approx -1/N$, so the condition is about $N > n/\ln 2 = 1{,}442.7$. The script checks every $N$ and finds that the smallest is $N = 1{,}444$. With $N = 2{,}000$ the expected population value is 0.3935 and the error 0.6065, and with $N = 10{,}000$ the error is 0.9048.

*So 1,445 queries, of which only the last is adaptive, are enough to make one answer off by more than a half on 1,000 records. Exact empirical answers need $n$ to grow linearly with the number of queries.*
:::

## The reverse direction, and a selection from one record {#sec-sol-ada-2}

> **Exercise 2.** **The reverse direction, and a selection from one record.** (a) Prove the second inequality of the transfer lemma, $\mathbb E[q_W(\mathcal D)] \le e^\varepsilon\,\mathbb E[q_W(D)] + \delta$, by applying DP from $D^{(i)}$ to $D$. (b) Take $n = 1$ record, a uniform bit $x_1$. The algorithm $W$ keeps the bit with probability $e^\varepsilon/(1 + e^\varepsilon)$ and flips it otherwise, which gives a bit $y$, and outputs the query $q(x) = \mathbf 1[x = y]$. Show that $W$ is $\varepsilon$-DP and that its selection bias is $\tanh(\varepsilon/2)/2$, about $\varepsilon/4$. Compare it with the lemma's bounds. *Check: at $\varepsilon = 0.1$, a bias of 0.02498 against the sharper bound 0.05259 and the lemma 0.10517.* *(Transfer lemma; randomized response)*

::: {.callout-tip collapse="true" title="Solution"}
**(a)**

1. **DP in the other direction.** Hold $D$ and $x_i'$ fixed. The samples $D^{(i)}$ and $D$ are neighbours, and the replace-one relation is symmetric. So Fact 3 of the chapter, applied to $f(q) = q(x_i)$, gives $\mathbb E_W[q_{W(D^{(i)})}(x_i)] \le e^\varepsilon\,\mathbb E_W[q_{W(D)}(x_i)] + \delta$.
2. **Average.** Averaging over $D$ and $x_i'$ keeps the inequality. Averaging over $i$ turns the right side into $e^\varepsilon\,\mathbb E[q_W(D)] + \delta$, by step 1 of the lemma's derivation.
3. **The left side.** By the renaming of Fact 2, $\mathbb E[q_{W(D^{(i)})}(x_i)] = \mathbb E[q_{W(D)}(x_i')] = \mathbb E[q_W(\mathcal D)]$, because $x_i'$ is a record that $W$ never saw. This gives the claim.

**(b)**

1. **Privacy.** The output is determined by $y$. Changing $x_1$ changes the probability of each value of $y$ from $e^\varepsilon/(1 + e^\varepsilon)$ to $1/(1 + e^\varepsilon)$ or back, a ratio of $e^\varepsilon$. This is randomized response, which is $\varepsilon$-DP.
2. **The empirical value.** With one record, $q_W(D) = \mathbf 1[x_1 = y]$, which is 1 exactly when the bit was kept. Its expectation is $e^\varepsilon/(1 + e^\varepsilon)$.
3. **The population value.** A fresh uniform bit equals $y$ with probability $\frac12$, whatever $y$ is, so $q_W(\mathcal D) = \frac12$.
4. **The bias.** The difference is $e^\varepsilon/(1 + e^\varepsilon) - \frac12 = (e^\varepsilon - 1)/(2(e^\varepsilon + 1)) = \tanh(\varepsilon/2)/2$. For small $\varepsilon$, $\tanh(\varepsilon/2) \approx \varepsilon/2$, so the bias is about $\varepsilon/4$.
5. **Numbers.** At $\varepsilon = 0.1$ the bias is 0.02498, against the sharper bound $(e^\varepsilon - 1)\cdot\frac12 = 0.05259$ and the lemma's $e^\varepsilon - 1 = 0.10517$. At $\varepsilon = 1$ it is 0.23106, against 0.85914 and 1.71828.

*A selection bias linear in $\varepsilon$ is possible, so the lemma's order cannot be improved. Its constant is loose by a factor of 2 against the sharper form here, and by 8 for the exponential mechanism of the chapter's example.*
:::

## Why the complements are needed {#sec-sol-ada-3}

> **Exercise 3.** **Why the complements are needed.** In the transfer theorem's proof, suppose you skip step 1 and let $W$ choose, among the original $k$ queries only, the one with the largest signed error $q_j(\mathcal D) - a_j$. What does the proof then bound? Show that adding the complements costs no privacy, and that answering $1 - q_j$ with $1 - a_j$ keeps the accuracy on the sample. *(Post-processing)*

::: {.callout-tip collapse="true" title="Solution"}
1. **Without complements.** The same steps bound $\mathbb E[\max_j(q_j(\mathcal D) - a_j)]$, the largest error in one direction only: answers that are too low. Answers that are too high, with $a_j > q_j(\mathcal D)$, are invisible to this $W$. In @tbl-ada-complement, $W$ would pick $q_2$ with error 0.05 and miss $q_1$, whose answer is 0.12 too high.
2. **A second selector.** One could instead run the proof again with a selector that maximizes $a_j - q_j(\mathcal D)$, but that needs the lemma for queries of the form $1 - q$. That is exactly what the complements provide, in one pass.
3. **No privacy cost.** The complement answers $1 - a_j$ are computed from the answers $a_j$ alone, without touching the data. By post-processing, the extended transcript is as private as the original one, $(\varepsilon, \delta)$-DP.
4. **Same accuracy.** The error of a complement on the sample is $|(1 - a_j) - (1 - q_j(D))| = |a_j - q_j(D)|$. So the largest error on the sample over the $2k$ queries is the same as over the original $k$, at most $\alpha$ on average.
5. **The complement is a statistical query.** $1 - q_j$ maps each record to $[0, 1]$, so the lemma applies to it.

*The complements are a bookkeeping trick: they let one private choice carry the largest error in either direction, at no cost in privacy or accuracy.*
:::

## Gaussian noise for 1,000 queries {#sec-sol-ada-4}

> **Exercise 4.** **Gaussian noise for 1,000 queries.** Use the balancing argument of @sec-ada-gauss to show that the Gaussian mechanism with the transfer theorem reaches an expected largest error $\alpha$ with about $n = 4\sqrt{4k\ln(2k)\ln(1/\delta)}/\alpha^2$ records. Evaluate it for $k = 1{,}000$, $\alpha = 0.05$ and $\delta = 0.0019$, and compare it with the exact optimum of @tbl-ada-rates. At that optimum, compute $\rho$ from $\sigma$. *Check: the formula gives 697,435 records against 764,337; at the optimum $\sigma = 0.00620$, $\rho = 2.224\times10^{-5}$ and $\varepsilon = 0.0236$.* *(zCDP composition; optimizing a sum $\sigma A + B/\sigma$)*

::: {.callout-tip collapse="true" title="Solution"}
1. **The two terms.** By @eq-ada-gauss the bound is the expected largest noise, $\sigma\sqrt{2\ln(2k)}$, plus the transfer term $e^\varepsilon - 1 + \delta$. With $\rho = k/(2n^2\sigma^2)$ and small $\varepsilon$, the transfer term is about $2\sqrt{\rho\ln(1/\delta)} = \sqrt{2k\ln(1/\delta)}/(n\sigma)$, plus $\delta$.
2. **Balance.** Write the sum as $\sigma A + B/\sigma$ with $A = \sqrt{2\ln(2k)}$ and $B = \sqrt{2k\ln(1/\delta)}/n$. Its derivative in $\sigma$, $A - B/\sigma^2$, vanishes at $\sigma = \sqrt{B/A}$, where the sum is $2\sqrt{AB}$.
3. **Solve for $n$.** Setting $2\sqrt{AB} = \alpha$ gives $AB = \alpha^2/4$, that is $\sqrt{2\ln(2k)}\sqrt{2k\ln(1/\delta)}/n = \alpha^2/4$, so $n = 4\sqrt{4k\ln(2k)\ln(1/\delta)}/\alpha^2$.
4. **Numbers.** With $k = 1{,}000$, $\ln(2k) = 7.601$, and with $\delta = 0.0019$, $\ln(1/\delta)$ is about 6.27. The formula gives 697,435 records.
5. **Against the optimum.** The exact minimization of @eq-ada-gauss needs 764,337 records. The formula is about 9% low because it drops the $\rho$ term of the conversion, the $\delta$ term and the curvature of $e^\varepsilon$, all of which add to the bound.
6. **The parameters at the optimum.** There $\sigma = 0.00620$, so $\rho = 1{,}000/(2 \times 764{,}337^2 \times 0.00620^2) = 2.224\times10^{-5}$. The conversion with $\delta = 0.0019$ gives $\varepsilon = 0.0236$. The noise term is 0.0242 and the transfer term 0.0258, almost equal, as the balancing argument predicts.

*The rate grows like $\sqrt k$, but the constant is large: three quarters of a million records for 1,000 queries at an error of 0.05.*
:::

## Rounds of adaptivity {#sec-sol-ada-5}

> **Exercise 5.** **Rounds of adaptivity.** An analysis asks $k$ queries in $r$ rounds. The queries of each round are fixed before any of that round's answers are seen. Show that splitting the sample into $r$ parts, one per round, gives an expected largest error of at most $\sqrt{r\ln(2k)/(2n)}$. For $k = 1{,}000$ and $\alpha = 0.05$, compute the records needed for $r = 1$, 10 and 1,000, and explain why splitting is linear in $k$ when every query is its own round. *Check: 1,521, 15,202 and 1,520,181 records.* *(Conditioning; the maximal inequality)*

::: {.callout-tip collapse="true" title="Solution"}
1. **One round.** Fix the parts used by earlier rounds. The queries of round $\ell$ are then fixed, and part $\ell$, of $n/r$ records, is independent of them. Each query's error on part $\ell$ has a moment generating function at most $e^{\lambda^2r/(8n)}$, by Hoeffding's lemma with $n/r$ records.
2. **All rounds.** Averaging over the earlier parts keeps the bound, so every one of the $2k$ signed errors satisfies it. The maximal inequality with $v = r/(4n)$ and $m = 2k$ gives $\sqrt{2\cdot r/(4n)\cdot\ln(2k)} = \sqrt{r\ln(2k)/(2n)}$.
3. **Records needed.** Setting this equal to $\alpha$ gives $n = r\ln(2k)/(2\alpha^2) = r \times 7.601/0.005$. That is 1,521 records for $r = 1$, 15,202 for $r = 10$ and 1,520,181 for $r = 1{,}000$, after rounding up.
4. **Why linear in $k$.** When every query is chosen from the previous answers, each query is its own round, so $r = k$ and the cost is $k\ln(2k)/(2\alpha^2)$. Each record answers one query only, so the data must grow with the number of queries.

*The number of rounds, not the number of queries, sets the cost of splitting. That is why a single final test set is cheap, and why a long sequential search is expensive.*
:::

## The boosting attack's approximation {#sec-sol-ada-6}

> **Exercise 6.** **The boosting attack's approximation.** Derive the chapter's approximation $\sqrt{k/(2n)}/\pi$ for the overfitting of the boosting attack, and check it by simulation for $n = 1{,}000$, $k = 100$ and for $n = 4{,}000$, $k = 400$. *Check: the approximation is 0.0712 in both cases; the script's simulations, over 40 runs each, give 0.0681 ± 0.0021 and 0.0715 ± 0.0012.* *(Normal approximation)*

::: {.callout-tip collapse="true" title="Solution"}
1. **One submission.** A random labelling's holdout accuracy is $\frac12 + Z$, where $Z$ is approximately normal with mean 0 and standard deviation $s = 1/(2\sqrt n)$. Its average agreement with each holdout label is also $\frac12 + Z$, by symmetry between the records.
2. **The kept submissions.** Given $Z > 0$, the mean of $Z$ is $s\sqrt{2/\pi}$, the mean of a half-normal. So a kept submission agrees with each holdout label with probability about $\frac12 + s\sqrt{2/\pi}$. About $k/2$ are kept.
3. **The vote on one record.** Each kept submission votes $+1$ for the right label or $-1$, with mean $2s\sqrt{2/\pi}$. The sum of $k/2$ votes has mean $k s\sqrt{2/\pi}$ and standard deviation about $\sqrt{k/2}$, since each vote has variance close to 1.
4. **The ratio.** Mean over standard deviation is $k s\sqrt{2/\pi}/\sqrt{k/2} = 2s\sqrt k/\sqrt\pi = \sqrt{k/n}/\sqrt\pi$, using $2s = 1/\sqrt n$.
5. **The accuracy.** For a small ratio $z$, a normal sum is positive with probability about $\frac12 + z/\sqrt{2\pi}$. So the overfitting is about $\sqrt{k/n}/(\sqrt\pi\sqrt{2\pi}) = \sqrt{k/(2n)}/\pi$.
6. **Numbers.** Both cases have $k/n = 0.1$, so the approximation is $\sqrt{0.05}/\pi = 0.0712$. The simulations give 0.0681 ± 0.0021 and 0.0715 ± 0.0012, within about one and a half standard errors. The chapter's 20 runs at $n = 1{,}000$, $k = 100$ gave 0.0739.

*The overfitting depends on $k$ and $n$ only through $k/n$: ten times more holdout records buy ten times more submissions at the same overfitting, not a hundred times.*
:::

## Thresholdout's budget {#sec-sol-ada-7}

> **Exercise 7.** **Thresholdout's budget.** In the chapter's experiment, the queries are correlations with sensitivity $\Delta = 2/n$, and Thresholdout uses the Laplace noise scale $b = 1/\sqrt n$ with $n = 2{,}000$. How many overfitting detections fit within $\varepsilon = 1$? How large must the holdout set be to afford 300 detections at $\varepsilon = 0.1$ with the same $b = 1/\sqrt n$? *Check: 11 detections; $n = 1.44\times10^8$.* *(Sparse vector technique; basic composition)*

::: {.callout-tip collapse="true" title="Solution"}
1. **The cost of one detection.** By @eq-ada-tho, one detection costs $2\Delta/b = 2 \cdot (2/n) \cdot \sqrt n = 4/\sqrt n$. At $n = 2{,}000$ that is 0.0894.
2. **The budget.** Detections compose by adding their costs, so $\varepsilon = 1$ allows $\lfloor 1/0.0894\rfloor = 11$ detections.
3. **The experiment.** The chapter's runs recorded 325.8 detections on average, so their total cost was $\varepsilon = 29.1$, far above 1.
4. **A larger holdout set.** With $B$ detections, $\varepsilon = 4B/\sqrt n$. Setting $\varepsilon = 0.1$ and $B = 300$ gives $\sqrt n = 4 \times 300/0.1 = 12{,}000$, so $n = 1.44\times10^8$ records.

*Thresholdout's guarantee makes agreeing answers free, but each disagreement costs a fixed amount. A careless analyst who triggers many detections exhausts the formal budget quickly, which is why the experiment's good behaviour lies outside its theorem.*
:::
