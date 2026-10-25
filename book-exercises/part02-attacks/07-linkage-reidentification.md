# Exercises: Linkage and re-identification

Exercises for [Linkage and re-identification](../../book-full/part02-attacks/07-linkage-reidentification.md), each with a worked solution. Try each exercise before opening its solution. The script `code/part02-attacks/solutions_linkage.py` computes every number quoted here; run it with `uv run python code/part02-attacks/solutions_linkage.py`.

The notation is the chapter's: $n$ people, cells $c$ with probabilities $p_c$, $K$ cells, $U$ the number of unique people, $q$ the sampling rate and $\lambda$ the expected number of other people in a cell.

## Linkage success and class sizes {#sec-sol-link-1}

> **Exercise 1.** **Linkage success and class sizes.** Show that for a fixed number of records $n$, @eq-link-success is at most $1/k$ when every class has at least $k$ records, and compute the success for a release of 8 records in classes of sizes 2, 2 and 4. When is the bound $1/k$ attained? *Check: $3/8 = 0.375$.* *(The linkage game)*

::: {.callout-tip collapse="true" title="Solution"}
1. **Count the classes.** The $J$ classes partition the $n$ records, so $n = \sum_jk_j$.
2. **Use the minimum size.** If every $k_j \ge k$, then $n \ge Jk$, so $J \le n/k$.
3. **Apply @eq-link-success.** $\Pr[\text{win}] = J/n \le 1/k$.
4. **Equality.** Step 2 is an equality exactly when every $k_j = k$: all classes have exactly $k$ records.
5. **The example.** Sizes 2, 2 and 4 give $J = 3$ classes for $n = 8$ records, so the success is $3/8 = 0.375$. The bound with $k = 2$ is $0.5$, not attained because the class of 4 wastes two records' worth of protection. A release of 9 records in three classes of 3 attains its bound, $1/3 = 0.333$.

The same formula in the other direction: the toy release of the chapter, with classes 1, 1, 1, 2, 3, has success $0.625$, and its smallest class is a singleton, so no bound below 1 applies.

*The linkage success is decided by the number of classes, so the best you can do against this adversary with classes of at least $k$ is to make them all exactly $k$.*
:::

## Bits for a target uniqueness {#sec-sol-link-2}

> **Exercise 2.** **Bits for a target uniqueness.** Under uniform cells, how many bits beyond $\log_2n$ does it take to make a fraction $f$ of the population unique? Evaluate for 8 billion people at $f = 0.5$ and $0.99$, and show that at exactly $\log_2n$ bits the fraction tends to $e^{-1}$ as $n \to \infty$. *Check: 33.4 and 39.5 bits.* *(Fact 2, uniform cells)*

::: {.callout-tip collapse="true" title="Solution"}
1. **The fraction.** With $K = 2^b$ uniform cells, @eq-link-uniform gives $f \approx e^{-n/K}$.
2. **Solve for $K$.** $n/K = \ln(1/f)$, so $K = n/\ln(1/f)$.
3. **Take $\log_2$.** $b = \log_2n - \log_2\ln(1/f) = \log_2 n + \log_2\frac{1}{\ln(1/f)}$. The excess over $\log_2n$ does not depend on $n$.
4. **Evaluate.** $\log_2(8\times10^9) = 32.9$. At $f = 0.5$, $\log_2(1/\ln 2) = 0.53$, so $b = 33.4$. At $f = 0.99$, $\ln(1/0.99) = 0.01005$ and $\log_2(1/0.01005) = 6.64$, so $b = 39.5$.
5. **At $b = \log_2 n$.** Then $K = n$ and the exact fraction is $(1 - 1/n)^{n-1}$. By Fact 2 it lies between $e^{-(n-1)/(n-1)} = e^{-1}$ and $e^{-(n-1)/n}$, and both tend to $e^{-1}$. Numerically: 0.3874 for $n = 10$, 0.3697 for $n = 100$, and 0.3679 $= e^{-1}$ to four decimals for $n = 10^4$ and $8\times10^9$.

*Six and a half bits beyond $\log_2 n$ make 99% of a population unique, whatever its size: about two extra attributes of the kind in the chapter's census.*
:::

## Collisions and the birthday bound {#sec-sol-link-3}

> **Exercise 3.** **Collisions and the birthday bound.** Show that the expected number of pairs of people who share a cell is $\binom n2\sum_cp_c^2$, and deduce a union bound on the probability that *somebody* shares a cell. For 30 people and the 730 cells of (birthday, sex), compare it with the exact probability. *Check: union bound $435/730 = 0.596$.* *(Indicators, birthday bound)*

::: {.callout-tip collapse="true" title="Solution"}
1. **Indicators.** Let $Y_{ij} = \mathbf 1[\text{persons } i \text{ and } j \text{ share a cell}]$ for each of the $\binom n2$ pairs. The number of sharing pairs is $\sum_{i<j}Y_{ij}$.
2. **One pair.** Persons $i$ and $j$ are independent, so $\Pr[Y_{ij} = 1] = \sum_c\Pr[i \in c]\Pr[j \in c] = \sum_cp_c^2$.
3. **Linearity** (Fact 1): the expected number of sharing pairs is $\binom n2\sum_cp_c^2$.
4. **Union bound.** "Somebody shares a cell" is the event that some $Y_{ij} = 1$, and $\Pr[\bigcup_{i<j}\{Y_{ij} = 1\}] \le \sum_{i<j}\Pr[Y_{ij} = 1] = \binom n2\sum_cp_c^2$. (Equivalently, Markov's inequality applied to the number of sharing pairs.)
5. **The example.** 730 equally likely cells give $\sum_cp_c^2 = 1/730$, and $\binom{30}{2} = 435$, so the bound is $435/730 = 0.596$. The exact probability, $1 - \prod_{i<30}(1 - i/730)$ as in the birthday bound of [Cryptography basics](../../book-full/part01-prerequisites/08-crypto-basics.md), is 0.4534, and the lower bound $1 - e^{-0.5959} = 0.4489$.

With birthdays alone (365 cells), the union bound is 1.192, useless as a probability, while the exact value is 0.7063.

*The expected number of colliding pairs is $\binom n2$ times the collision probability: the quantity in the chapter's lower bound @eq-link-collision, counted over pairs instead of felt by one person.*
:::

## The variance of the number of uniques {#sec-sol-link-4}

> **Exercise 4.** **The variance of the number of uniques.** For $K$ uniform cells, show that $\Pr[i \text{ and } j \text{ both unique}] = (1 - \frac1K)(1 - \frac2K)^{n-2}$ for $i \ne j$, and compute $\operatorname{Var}U$. For $n = K = 10{,}000$, compare the standard deviation of $U/n$ with a simulation and with McDiarmid. *Hint: $\operatorname{Var}U = \mathbb E[U(U - 1)] + \mathbb EU - (\mathbb EU)^2$.* *(Indicators, concentration)*

::: {.callout-tip collapse="true" title="Solution"}
1. **Both unique.** Persons $i$ and $j$ must be in different cells, probability $1 - \frac1K$. Given their two cells, each of the other $n - 2$ people must avoid both, probability $1 - \frac2K$ each, independently. Multiply.
2. **Second factorial moment.** $U(U - 1) = \sum_{i \ne j}\mathbf 1[i \text{ unique}]\mathbf 1[j \text{ unique}]$, a sum over the $n(n-1)$ ordered pairs. By linearity, $\mathbb E[U(U - 1)] = n(n-1)(1 - \frac1K)(1 - \frac2K)^{n-2}$.
3. **Mean.** $\mathbb EU = n(1 - \frac1K)^{n-1}$ (@eq-link-unique with uniform cells).
4. **Variance.** $\operatorname{Var}U = \mathbb E[U^2] - (\mathbb EU)^2 = \mathbb E[U(U-1)] + \mathbb EU - (\mathbb EU)^2$.
5. **Evaluate at $n = K = 10{,}000$.** $\mathbb EU/n = 0.3679$ and $\operatorname{Var}U = 2325.6$, so the standard deviation of $U/n$ is $0.00482$. 2,000 simulated populations give mean 0.3678 and standard deviation 0.00483.
6. **Compare with McDiarmid.** A normal approximation puts 95% of populations within $1.96 \times 0.00482 = 0.0095$ of the mean. McDiarmid guarantees 0.0272, about 2.9 times wider, because it charges every person the worst-case change of 2 in $U$.

**A detail about dependence.** If the indicators were independent, the variance would be $n f(1 - f)$ with $f = \mathbb EU/n$: 2325.5, almost exactly the true 2325.6. At $n = K$ the covariances between indicators happen to cancel. They do not in general: at $n = 10{,}000$ and $K = 2{,}500$, $\mathbb EU/n = 0.0183$ and $\operatorname{Var}U = 139.5$, against 179.7 for independent indicators. The covariance part is $-40.2$: there, one person being unique makes others slightly less likely to be, because the remaining people crowd into fewer cells. The standard deviation of $U/n$ is then 0.00118 (simulated 0.00117).

*Pairwise indicators give the exact variance, and it shows McDiarmid's bound to be about three times too wide for the fraction unique, but valid without any calculation.*
:::

## How many rare items {#sec-sol-link-5}

> **Exercise 5.** **How many rare items.** Each auxiliary item is rated by 1% of 480,189 users, and agreements are independent. What is the smallest number of items for which the expected number of unrelated records agreeing on all of them is below 0.01? With that many items, $\rho = 0$, and exactly one record agreeing on all of them, what is the posterior probability that this record is the target, if the target is in the release with prior probability $\frac12$? *Check: 4 items, 0.0048 expected false matches, posterior 0.9952.* *(Bits, Bayes)*

::: {.callout-tip collapse="true" title="Solution"}
1. **Expected false matches.** An unrelated record agrees on all $m$ items with probability $0.01^m$, so the expected number among $n = 480{,}189$ records is $n\cdot0.01^m$.
2. **Solve.** $m = 3$ gives 0.48 and $m = 4$ gives 0.0048, so $m = 4$. In bits: each item carries $\log_2 100 = 6.64$ bits, and four carry 26.6, which exceeds $\log_2 n = 18.9$ by more than the 6.64 bits needed to push the expected false matches below 0.01.
3. **Likelihood ratios.** With $\rho = 0$, a record that misses an item cannot be the target, so its likelihood ratio is 0. The one agreeing record has $\mathrm{LR} = \prod_i(1/s_i) = 100^4 = 10^8$.
4. **Posterior with a prior.** From the chapter's interpretation of @eq-link-posterior, with prior $\pi$ that the target is in the release,
$$\Pr[\text{target} = t \mid \text{data}] = \frac{\pi\,\mathrm{LR}(t)}{\pi\sum_j\mathrm{LR}(j) + (1 - \pi)n} = \frac{10^8/2}{10^8/2 + 480{,}189/2} = 0.9952.$$
5. **The role of the prior.** With $\pi = 0.01$, say if the adversary merely suspects the target used the service, the posterior is 0.6778. The same match is much weaker evidence when the target is unlikely to be in the data at all.

*Each rare item cuts the expected false matches by its rarity factor; the posterior then says how much a perfect match is worth, which depends on whether the target is in the data at all.*
:::

## Sampling and the Poisson approximation {#sec-sol-link-6}

> **Exercise 6.** **Sampling and the Poisson approximation.** Derive the Poisson form of @eq-link-kappa directly from Poisson thinning, and show $\kappa \ge 1/(1 + \lambda)$ for every $q$. For which $\lambda$ is $\kappa = \frac12$ in a 1% sample? *Check: $\lambda = 1/0.99 = 1.0101$.* *(Poisson thinning)*

::: {.callout-tip collapse="true" title="Solution"}
1. **Poisson thinning.** Let the number of other people in the target's cell be $J \sim \mathrm{Poisson}(\lambda)$. Each is sampled independently with probability $q$. By Exercise 2 of [Probability distributions](../../book-full/part01-prerequisites/01-distributions.md), the sampled count is $J_s \sim \mathrm{Poisson}(\lambda q)$, and (by the same moment generating function argument applied to the pair) it is independent of the unsampled count.
2. **Correct and one match.** The target is sampled and $J_s = 0$: probability $q e^{-\lambda q}$.
3. **Wrong and one match.** The target is not sampled and $J_s = 1$: probability $(1 - q)\lambda q e^{-\lambda q}$.
4. **Condition.** $\kappa = \dfrac{q e^{-\lambda q}}{q e^{-\lambda q} + (1 - q)\lambda q e^{-\lambda q}} = \dfrac{1}{1 + \lambda(1 - q)}$.
5. **The floor.** $\lambda(1 - q) \le \lambda$ for every $q \in [0, 1]$, so $\kappa \ge 1/(1 + \lambda)$, with equality only as $q \to 0$.
6. **Half.** $\kappa = \frac12$ when $\lambda(1 - q) = 1$; at $q = 0.01$, $\lambda = 1/0.99 = 1.0101$.
7. **Against the exact binomial.** With $n = 100{,}000$ and $p = \lambda/(n - 1)$, the exact @eq-link-kappa and the Poisson form agree to five decimals: 0.90992 for $\lambda = 0.1$ and 0.50251 for $\lambda = 1$, just above their floors 0.90909 and 0.50000.

*A 1% sample protects a target only if their cell is expected to hold about one other person or more; for rarer combinations the match is right most of the time.*
:::

## Eccentricity under an exponential tail {#sec-sol-link-7}

> **Exercise 7.** **Eccentricity under an exponential tail.** Suppose an outsider's scores against the $n$ records were independent exponential variables with mean $\mu$ (so standard deviation $\mu$). Show that the gap between the two largest is again exponential with mean $\mu$, whatever $n$, so the eccentricity test with threshold $\tau$ falsely matches with probability about $e^{-\tau}$. Compare with normally distributed scores by simulation. *Hint: memorylessness ([Probability distributions](../../book-full/part01-prerequisites/01-distributions.md)).* *Check: $e^{-1.5} = 0.2231$.* *(Exponential distribution, false positives)*

::: {.callout-tip collapse="true" title="Solution"}
1. **Condition on the second largest.** Let $x$ be the value of the second-largest score. Given $x$, exactly one score exceeds $x$, and (by independence and symmetry) it is distributed as a single exponential score conditioned to exceed $x$.
2. **Memorylessness.** For an exponential $S$ with mean $\mu$, $\Pr[S > x + g \mid S > x] = e^{-(x+g)/\mu}/e^{-x/\mu} = e^{-g/\mu}$. So, given $x$, the gap $s_1 - x$ is exponential with mean $\mu$.
3. **Remove the conditioning.** The conditional law in step 2 does not depend on $x$, so the gap is exponential with mean $\mu$ unconditionally, and independent of $x$. Nothing depended on $n$.
4. **The test.** For large $n$ the sample standard deviation of the scores is close to $\mu$, so $\Pr[(s_1 - s_2)/\sigma \ge \tau] \approx \Pr[\text{gap} \ge \tau\mu] = e^{-\tau}$. At $\tau = 1.5$ that is 0.2231. A simulation with $n = 1000$ and 20,000 trials gives 0.2257.
5. **Normal scores.** With standard normal scores, the gap between the top two shrinks with $n$, roughly like an exponential with mean $1/\sqrt{2\ln n}$ (the normal tail near its maximum behaves like an exponential with that rate). The approximation gives $e^{-1.5\sqrt{2\ln 1000}} = 0.0038$, and the simulation 0.0024.

So the same threshold $\tau = 1.5$ falsely matches about one outsider in five when null scores have an exponential tail, and about one in 400 when they are normal. The chapter's ratings experiment, with 26.3% of outsiders matched, sits on the exponential side: its scores are sums of a few weighted agreements, mostly zero, with a long right tail.

*The eccentricity threshold controls false matches only relative to the tail of the null score distribution, which is why it has to be calibrated on outsiders.*
:::

## Hashed identifiers and taxi trips {#sec-sol-link-8}

> **Exercise 8.** **Hashed identifiers and taxi trips.** Driver licence numbers in the taxi release of @sec-link-pseudonyms have six digits, or seven digits starting with 5. (a) How many values is that, how many bits, and how long does it take to hash them all at $10^7$ hashes per second? Compare a random 10-character password over 62 symbols. (b) Treat each trip's pickup as a uniformly random cell: one of 525,600 minutes of the year and one of 78,000 squares of 100 m. Use @eq-link-uniform to find the fraction of the 173 million trips that are unique on their pickup. Why is the true fraction lower, and why does that not save the passengers? *Check: $2\times10^6$ values (20.9 bits) take 0.2 seconds. A password carries 59.5 bits and takes about 2661 years. The unique fraction is 0.9958.* *(Bits, uniqueness)*

::: {.callout-tip collapse="true" title="Solution"}
1. **Count the licences.** There are $10^6$ six-digit numbers and $10^6$ seven-digit numbers starting with 5, so $2\times10^6$ values. That is $\log_2(2\times10^6) = 20.9$ bits.
2. **Time the dictionary.** At $10^7$ hashes per second, hashing every value takes $2\times10^6/10^7 = 0.2$ seconds. Sorting the hashes and looking up each pseudonym adds little.
3. **Compare a password.** A random 10-character password over 62 symbols has $62^{10} = 8.39\times10^{17}$ values, 59.5 bits. Hashing all of them at the same rate takes about 2661 years. The hash function is the same in both cases; what differs is how many bits the input carries.
4. **Count the pickup cells.** There are $525{,}600 \times 78{,}000 = 4.1\times10^{10}$ cells, so the average cell holds $n/K = 173\times10^6/4.1\times10^{10} = 0.00422$ trips.
5. **Apply @eq-link-uniform.** The fraction unique is about $e^{-0.00422} = 0.9958$.
6. **Why the true fraction is lower.** Trips are not spread uniformly. They crowd into busy hours and busy places, such as Midtown Manhattan at rush hour, where many trips share a minute and a square. The uniform model overstates uniqueness, as it did for Sweeney's attributes.
7. **Why that does not save the passengers.** An adversary who knows a pickup usually knows more than its minute and square: the drop-off area, the date, a photograph of the taxi. Each adds bits, and a celebrity photographed outside a hotel is rarely in the crowded cells anyway. Even in a crowded cell, the adversary learns that the trip is one of a handful.

*A hash hides as many bits as its input carries and no more. A single recorded action, like a taxi pickup, carries more bits than it takes to single out one trip among hundreds of millions.*
:::
