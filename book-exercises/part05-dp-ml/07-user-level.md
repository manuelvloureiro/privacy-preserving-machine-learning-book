# Exercises: User-level differential privacy

Exercises for [the chapter on user-level differential privacy](../../book-full/part05-dp-ml/07-user-level.md), each with a worked solution. Try each exercise before opening its solution. The script `code/part05-dp-ml/solutions_user_level.py` computes every number quoted here; run it with `uv run python code/part05-dp-ml/solutions_user_level.py`.

The solutions use the chapter's notation. A guarantee of differential privacy (DP) is **example-level** when neighbouring datasets differ in one record and **user-level** when they differ in all the records of one user. Stochastic gradient descent (SGD) with per-example clipping and Gaussian noise is DP-SGD; $q$ is its Poisson sampling rate, $\sigma$ its noise multiplier, $C$ its clipping norm and $T$ its number of steps. A user has $k$ records, and $H \sim \mathrm{Bin}(k, q)$ of them land in a batch. Rényi differential privacy (RDP) is the accountant's currency. The running example's model is example-level $(6.71, 10^{-5})$-DP, from $q = 0.01$, $\sigma = 1$ and $T = 10{,}000$, and its population has 1,000 users with between 1 and 348 records each.

## Group privacy at a small δ {#sec-sol-ulvl-1}

> **Exercise 1.** **Group privacy at a small $\delta$.** A mechanism is example-level $(1, 10^{-6})$-DP. Using @eq-ulvl-group, give the user-level guarantee for users with 2, 5 and 10 records. For which of them is the guarantee meaningful if the dataset has 10,000 users? *Check: $\delta_2 = 3.718 \times 10^{-6}$, $\delta_5 = 8.579 \times 10^{-5}$, $\delta_{10} = 0.01282$.* *(The geometric sum of group privacy)*

::: {.callout-tip collapse="true" title="Solution"}
1. **The formula.** A chain of $k$ neighbouring steps gives $\varepsilon_k = k\varepsilon$ and $\delta_k = \delta(e^{k\varepsilon} - 1)/(e^\varepsilon - 1)$, because each step's $\delta$ is multiplied by the $e^\varepsilon$ of every later step.
2. **$k = 2$.** $\delta_2 = 10^{-6}(e^2 - 1)/(e - 1) = 10^{-6}(1 + e) = 3.718 \times 10^{-6}$, with $\varepsilon_2 = 2$.
3. **$k = 5$.** $\delta_5 = 10^{-6}(e^5 - 1)/(e - 1) = 8.579 \times 10^{-5}$, with $\varepsilon_5 = 5$.
4. **$k = 10$.** $\delta_{10} = 10^{-6}(e^{10} - 1)/(e - 1) = 0.01282$, with $\varepsilon_{10} = 10$.
5. **Which are meaningful.** $\delta$ is the probability that the guarantee fails outright, so for $U = 10{,}000$ users it should be well below $1/U = 10^{-4}$. Only $k = 2$ satisfies that comfortably; $k = 5$ is just below $10^{-4}$, and $k = 10$ allows a failure for about one user in 78.

*The $\delta$ of group privacy grows like $e^{(k-1)\varepsilon}$, so a guarantee designed for records must start from a much smaller $\delta$ to survive the step to users.*
:::

## When group privacy gives nothing {#sec-sol-ulvl-2}

> **Exercise 2.** **When group privacy gives nothing.** The running example's accountant can convert its RDP curve to $(\varepsilon_{\text{rec}}, \delta_{\text{rec}})$ at any $\delta_{\text{rec}}$. For $\delta_{\text{rec}} = 10^{-5}$, $10^{-10}$, $10^{-20}$, $10^{-50}$ and $10^{-100}$, compute $\varepsilon_{\text{rec}}$ and the user's $\delta_k$ from @eq-ulvl-group for $k = 2$, 5 and 10. Explain why group privacy yields a user-level $(\varepsilon, 10^{-5})$ guarantee for $k = 2$ and 5 but none for $k = 10$. *Check: $\varepsilon_{\text{rec}} = 6.71$, 9.78, 14.31, 24.94 and 41.02; for $k = 10$, $\log_{10}\delta_{10}$ grows from 21.2 to 60.3.* *(A race between $\delta$ and $e^{(k-1)\varepsilon}$)*

::: {.callout-tip collapse="true" title="Solution"}
1. **The record-level guarantees.** The accountant gives $\varepsilon_{\text{rec}} = 6.71$, 9.78, 14.31, 24.94 and 41.02 at the five values of $\delta_{\text{rec}}$. A smaller $\delta$ costs a larger $\varepsilon$, roughly through the term $\ln(1/\delta)/(\alpha - 1)$ of the conversion.
2. **The user's $\delta$.** $\delta_k \approx \delta_{\text{rec}}e^{(k-1)\varepsilon_{\text{rec}}}$, so $\log_{10}\delta_k \approx \log_{10}\delta_{\text{rec}} + (k - 1)\varepsilon_{\text{rec}}/\ln 10$. The script gives, in powers of 10:

| $\delta_{\text{rec}}$ | $10^{-5}$ | $10^{-10}$ | $10^{-20}$ | $10^{-50}$ | $10^{-100}$ |
| --- | --- | --- | --- | --- | --- |
| $\varepsilon_{\text{rec}}$ | 6.71 | 9.78 | 14.31 | 24.94 | 41.02 |
| $k = 2$ | $10^{-2.1}$ | $10^{-5.8}$ | $10^{-13.8}$ | $10^{-39.2}$ | $10^{-82.2}$ |
| $k = 5$ | $10^{6.7}$ | $10^{7.0}$ | $10^{4.9}$ | $10^{-6.7}$ | $10^{-28.7}$ |
| $k = 10$ | $10^{21.2}$ | $10^{28.2}$ | $10^{35.9}$ | $10^{47.5}$ | $10^{60.3}$ |

: The user's δ from group privacy, as a function of the record-level δ at which the accountant is converted. Values above 1 are no guarantee. {#tbl-sol-ulvl-race}

3. **Who wins the race.** Going from $\delta_{\text{rec}} = 10^{-50}$ to $10^{-100}$ lowers $\log_{10}\delta_{\text{rec}}$ by 50 and raises $\varepsilon_{\text{rec}}$ by 16.08, so $(k - 1)\varepsilon_{\text{rec}}/\ln 10$ rises by $6.98(k - 1)$. For $k = 2$ and 5 the rise, 7.0 and 27.9, is less than 50 and $\delta_k$ falls; it crosses $10^{-5}$ for $\delta_{\text{rec}}$ between $10^{-5}$ and $10^{-10}$ when $k = 2$, and between $10^{-20}$ and $10^{-50}$ when $k = 5$. For $k = 10$ the rise is 62.8, more than 50, and $\delta_{10}$ grows at every step of the table: it never comes near $10^{-5}$.
4. **Why.** At small $\delta_{\text{rec}}$ the conversion uses orders close to 1, where $\varepsilon_{\text{rec}}$ grows almost in proportion to $\ln(1/\delta_{\text{rec}})$, with a slope that the table puts at about $16.08/(50\ln 10) = 0.14$. The user's $\ln\delta_k$ then changes by $(k - 1) \times 0.14 - 1$ per unit of $\ln(1/\delta_{\text{rec}})$, positive once $k - 1 > 7$.

*This is why the chapter's group-privacy column reads "none" from $k = 10$: generic group privacy cannot turn this RDP curve into any user-level guarantee at $\delta = 10^{-5}$. The direct accountant, which knows the mechanism, gives $\varepsilon = 146.20$.*
:::

## The second moment and the rate-kq rule {#sec-sol-ulvl-3}

> **Exercise 3.** **The second moment and the rate-kq rule.** At order $\alpha = 2$, @eq-ulvl-pair says $A_2 = \mathbb E[e^{H_1H_2/\sigma^2}]$ for two independent copies of $H \sim \mathrm{Bin}(k, q)$. Compute $\varepsilon(2) = \ln A_2$ for $q = 0.01$, $\sigma = 1$ and $k = 2$, 5 and 10, and compare with @eq-ulvl-rate's single record at rate $kq$, $\ln(1 + (kq)^2(e^{1/\sigma^2} - 1))$. Explain the result for $k = 10$. *Check: ratios 1.0177, 1.0793 and 469.* *(Moments of a mixture)*

::: {.callout-tip collapse="true" title="Solution"}
1. **Exact.** $A_2 = \sum_{a,b}b_ab_be^{ab/\sigma^2}$ with $b_m = \binom km q^m(1 - q)^{k-m}$: a double sum over the possible numbers of drawn records in two independent batches. It gives $\varepsilon(2) = 6.992273 \times 10^{-4}$ for $k = 2$, $4.626224 \times 10^{-3}$ for $k = 5$ and $7.990343$ for $k = 10$, the same values as the chapter's numerical integral for $k = 2$ and 5.
2. **Rate $kq$.** $\ln(1 + (kq)^2(e - 1))$ gives $6.870766 \times 10^{-4}$, $4.286504 \times 10^{-3}$ and $1.703686 \times 10^{-2}$.
3. **Ratios.** 1.0177, 1.0793 and 469. For $k = 2$ and 5 the rule is good to 2% and 8%: the terms it drops, with $ab \ge 2$, have probability of order $(kq)^3$ or smaller.
4. **What breaks at $k = 10$.** The term $a = b = k$, both batches drawing all of the user's records, has probability $q^{2k}$ and factor $e^{k^2/\sigma^2}$. For $k = 10$ that is $10^{-40}e^{100} \approx e^{7.90}$, larger than everything else together: the exact $\ln A_2 = 7.99$ is almost entirely this one term. A tiny probability cannot cancel a factor that grows like $e^{k^2}$.
5. **Consequence.** At $\sigma = 1$, low orders of the user's curve are dominated by the rare steps that draw all of a heavy user's records together. The accountant then moves to orders close to 1, as @tbl-ulvl-k shows (best order 1.3 for $k = 10$).

*The all-records-drawn term $q^{2k}e^{k^2/\sigma^2}$ exceeds 1 when $k^2/\sigma^2 > 2k\ln(1/q)$, that is $k > 2\sigma^2\ln(1/q)$, which is 9.2 for $q = 0.01$ and $\sigma = 1$. Well below that size the rate-$kq$ rule is a good quick estimate; near or above it only the accountant is reliable.*
:::

## Per-record clipping inside user-level sampling {#sec-sol-ulvl-4}

> **Exercise 4.** **Per-record clipping inside user-level sampling.** User-level DP-SGD with $q = 0.1$, $T = 1{,}000$ and per-user clipping needs $\sigma = 3.78$ for $\varepsilon = 4$ at $\delta = 10^{-5}$. Instead, you clip each record to $C$ and sum at most $k$ records per user. Using @eq-ulvl-together, what noise multiplier gives the same guarantee for $k = 2$ and $k = 5$? Verify numerically at $\alpha = 3$. *Check: 7.56 and 18.9.* *(Rescaling a Gaussian mechanism)*

::: {.callout-tip collapse="true" title="Solution"}
1. **The sensitivity.** A user's sum of at most $k$ clipped records has length at most $kC$, and the user is sampled whole with probability $q$.
2. **Rescale.** Dividing the noisy sum by $k$ is invertible, so it changes no divergence. After it, the shift is at most $C$ and the noise is $\sigma C/k$: a mechanism with noise multiplier $\sigma/k$. That is @eq-ulvl-together.
3. **Match the guarantee.** To be as private as per-user clipping with $\sigma = 3.78$, the noise multiplier must satisfy $\sigma/k = 3.78$: $\sigma = 7.56$ for $k = 2$ and $18.9$ for $k = 5$.
4. **Check.** At $\alpha = 3$, the binomial sum with $k = 2$ and $\sigma = 7.56$, and with $k = 1$ and $\sigma = 3.78$, both give $1.094286 \times 10^{-3}$; so does $k = 5$ with $\sigma = 18.9$.

*Summing clipped records costs a factor $k$ in noise, where clipping the user's average costs nothing. Per-record clipping is convenient when the code for per-example gradients already exists, but its noise grows with the cap.*
:::

## Averaging or summing before clipping {#sec-sol-ulvl-5}

> **Exercise 5.** **Averaging or summing before clipping.** Three users have 1, 10 and 348 records, and every record of every user has the same gradient $g$ with $\|g\| = 0.01$. With $C = 0.5$, compute the length of each user's clipped contribution when the per-user vector is (i) the average of the user's gradients and (ii) their sum. Relate the answers to @eq-ulvl-expected. *Check: (i) 0.01 for all three; (ii) 0.01, 0.1 and 0.5.* *(Weights in the expected update)*

::: {.callout-tip collapse="true" title="Solution"}
1. **Average then clip.** Every user's average is $g$, of length $0.01 < C$, so nothing is clipped and each user contributes 0.01. Every user has the same weight, whatever their size.
2. **Sum then clip.** The sums have lengths 0.01, 0.1 and 3.48. The first two are below $C$ and are kept; the third is clipped to 0.5.
3. **Weights.** With averaging, the expected update is @eq-ulvl-expected: each user counts $1/U$. With summing, users below the clip count in proportion to their records, as in example-level training, and users above it count $C/\|g\|$ records' worth: here the 348-record user counts as 50 records, not 348.
4. **Privacy.** Both are valid user-level DP-SGD: each user's vector is computed from their own records and clipped to $C$, so the guarantee is the same.

*Between equal weights per user and equal weights per record there is a continuum, set by how a user's vector is formed and by $C$. It is a modelling choice, made inside the per-user computation, and it costs no privacy.*
:::

## Capping contributions to a count {#sec-sol-ulvl-6}

> **Exercise 6.** **Capping contributions to a count.** You release the number of records in the running example's population, $n = 10{,}000$, with user-level $(1, 10^{-5})$-DP. Cap each user at $k$ records, so the true capped count is $n_k = \sum_u\min(n_u, k)$ and its user-level sensitivity is $k$, and add Gaussian noise of standard deviation $k$ times the noise for sensitivity 1. Taking the error of the estimate of $n$ as $\sqrt{(n - n_k)^2 + \text{noise variance}}$, find the best cap. *Check: noise 3.731 per unit of sensitivity; best $k = 158$ with error 698.6; no cap ($k = 348$) gives 1,298.3.* *(The bias–variance trade-off of Amin et al.)*

::: {.callout-tip collapse="true" title="Solution"}
1. **The noise.** The exact calibration of the Gaussian mechanism for $(1, 10^{-5})$ and sensitivity 1 gives standard deviation 3.731, so a cap of $k$ needs $3.731k$.
2. **The bias.** Capping at $k$ removes $n - n_k$ records, all from users with more than $k$. From the population: $n_k = 1{,}000$ at $k = 1$ (bias $-9{,}000$), 5,063 at $k = 10$, 8,437 at $k = 50$, 9,214 at $k = 100$, 9,756 at $k = 200$ and 10,000 at $k = 348$.
3. **The error.** $\sqrt{(n - n_k)^2 + (3.731k)^2}$ is 9,000 at $k = 1$, 1,574.1 at $k = 50$, 870.0 at $k = 100$, 785.0 at $k = 200$ and 1,298.3 at $k = 348$, where the bias vanishes and the noise standard deviation is 1,298.3.
4. **The best cap.** Scanning every $k$ gives $k = 158$, with $n_k = 9{,}625$ and error 698.6. Below it, each extra unit of $k$ removes more bias than it adds noise, because many users still have more records; above it, only a few heavy users gain and the noise keeps growing in proportion to $k$.

*The best cap is set by the tail of the distribution of user sizes, as Amin et al. (2019) show in general. Choosing it from the data, as here, is itself a release that should use public information or be made privately.*
:::
