# Exercises: Syntactic anonymity and how it breaks

Exercises for [Syntactic anonymity and how it breaks](../../book-full/part02-attacks/08-syntactic-anonymity.md), each with a worked solution. Try each exercise before opening its solution. The script `code/part02-attacks/solutions_syntactic_anonymity.py` computes every number quoted here; run it with `uv run python code/part02-attacks/solutions_syntactic_anonymity.py`.

The notation is the chapter's: $n$ records, classes $E$, $P_E$ a class's distribution of the sensitive attribute and $P$ the table's, $r_1 \ge r_2 \ge \dots$ a class's sorted counts.

## Bits for 10-anonymity {#sec-sol-anon-1}

> **Exercise 1.** **Bits for 10-anonymity.** For a population of $n = 10^6$ in uniform cells, how many bits of quasi-identifier can a release keep if at most 1% of records may fall in classes smaller than 10? Compute $\lambda_k/k$ for $k = 2$, 10 and 100, and explain why it tends to 1. *Check: $\lambda_{10} = 17.40$, 15.81 bits.* *(Poisson tails)*

::: {.callout-tip collapse="true" title="Solution"}
1. **The condition.** By @prp-anon-k-violate, the fraction of records in classes smaller than 10 is $\Pr[\mathrm{Poisson}(\lambda) \le 8]$ with $\lambda = n/K$. It falls as $\lambda$ grows.
2. **Solve for $\lambda$.** Bisection on $\Pr[\mathrm{Poisson}(\lambda) \le 8] = 0.01$ gives $\lambda_{10} = 17.40$. It does not depend on $n$.
3. **Convert to bits.** $K = n/\lambda_{10}$, so $b = \log_2 n - \log_2\lambda_{10} = 19.93 - 4.12 = 15.81$ bits.
4. **The ratio.** $\lambda_2 = 4.61$, $\lambda_{10} = 17.40$ and $\lambda_{100} = 123.61$, so $\lambda_k/k = 2.30$, 1.74 and 1.24.
5. **Why it tends to 1.** A Poisson variable with mean $\lambda$ has standard deviation $\sqrt\lambda$, and for large $\lambda$ it is close to normal. Its lower 1% point is about $\lambda - 2.33\sqrt\lambda$. Setting this equal to $k - 2$ gives $\lambda_k \approx k + 2.33\sqrt k$, up to lower-order terms, so $\lambda_k/k \approx 1 + 2.33/\sqrt k \to 1$. The approximation gives 1.74 at $k = 10$ and 1.23 at $k = 100$. It is poor at $k = 2$ (2.65 against 2.30), where the Poisson distribution is far from normal.

*For large $k$, k-anonymity costs about $\log_2 k$ bits plus a margin that shrinks like $1/\sqrt k$ in relative terms. For small $k$ the margin is more than a bit.*
:::

## Entropy diversity and the top value {#sec-sol-anon-2}

> **Exercise 2.** **Entropy diversity and the top value.** A class has one value with share $\rho$ and the rest spread evenly over the other 7 of 8 values. Find the largest $\rho$ for which the class is entropy 3-diverse, and compare it with the bound that recursive $(c, 3)$-diversity would impose. *Check: $\rho = 0.733$.* *(Entropy)*

::: {.callout-tip collapse="true" title="Solution"}
1. **The entropy.** The class has $H(\rho) = -\rho\log_2\rho - (1 - \rho)\log_2\frac{1 - \rho}{7}$ bits. At $\rho = \frac18$ this is $\log_2 8 = 3$ bits, and it falls to 0 as $\rho \to 1$.
2. **Solve $H(\rho) = \log_2 3 = 1.5850$.** $H$ decreases in $\rho$ on $[\frac18, 1)$, so bisection gives the largest share, $\rho = 0.733$.
3. **Recursive diversity.** The tail beyond the two most common values is the 6 smallest counts, a share $6(1 - \rho)/7$. Recursive $(c, 3)$-diversity requires $\rho < c \cdot 6(1 - \rho)/7$. With $c = 2$ this is $7\rho < 12(1 - \rho)$, that is $\rho < 12/19 = 0.632$.
4. **The comparison.** The entropy 3-diverse class with $\rho = 0.733$ is recursive $(c, 3)$-diverse only for $c > 7\rho/(6(1 - \rho)) = 3.21$, where the bound of @prp-anon-recursive is $c/(1 + c) = 0.762$.

*Entropy 3-diversity sounds like "about as uncertain as three equally likely values", yet it allows one value in almost three quarters of the class. The entropy counts the spread of the rare values, which the adversary does not care about.*
:::

## Earth mover's distance on salaries {#sec-sol-anon-3}

> **Exercise 3.** **Earth mover's distance on salaries.** Salaries take the ordered values 3k, 4k, …, 11k, uniformly in the table. Compute the earth mover's distance (EMD) of @eq-anon-emd for the classes {3k, 4k, 5k} and {6k, 8k, 11k}. *Check: 0.375 and 0.167.* *(Cumulative sums)*

::: {.callout-tip collapse="true" title="Solution"}
1. **The table.** Nine values, each $\frac19$. Its cumulative sums at the first eight values are $\frac19, \frac29, \dots, \frac89$.
2. **The low class.** Each of 3k, 4k, 5k has $\frac13$. The cumulative sums are $\frac13, \frac23, 1, 1, 1, 1, 1, 1$. The absolute differences from the table are $\frac29, \frac49, \frac69, \frac59, \frac49, \frac39, \frac29, \frac19$, which sum to $\frac{27}{9} = 3$. Divide by $m - 1 = 8$: EMD $= 0.375$.
3. **The spread class.** 6k, 8k and 11k are values 4, 6 and 9. The cumulative sums are $0, 0, 0, \frac13, \frac13, \frac23, \frac23, \frac23$. The absolute differences are $\frac19, \frac29, \frac39, \frac19, \frac29, 0, \frac19, \frac29$, which sum to $\frac{12}{9}$. Divide by 8: EMD $= \frac16 = 0.167$.

These are the two classes of Li, Li & Venkatasubramanian's example. Both are distinct 3-diverse, but the first tells the adversary that the target earns little, and its EMD is more than twice as large.

*EMD on ordered values charges for how far the mass must move, so a class bunched at one end costs more than one spread across the range.*
:::

## Intersecting two diverse classes {#sec-sol-anon-4}

> **Exercise 4.** **Intersecting two diverse classes.** Each of two classes holds the true value and $\ell - 1$ wrong values, chosen uniformly without replacement from the $m - 1$ wrong ones, independently for the two classes. Show that the intersection pins down the true value with probability $\binom{m-\ell}{\ell-1}/\binom{m-1}{\ell-1}$, and evaluate it for $m = 8$ and $\ell = 2$, 3, and for $m = 20$, $\ell = 3$. *Check: 0.4762 for $m = 8$, $\ell = 3$.* *(Counting, hypergeometric)*

::: {.callout-tip collapse="true" title="Solution"}
1. **When it is pinned.** The intersection is the true value plus the wrong values the two classes share. It is pinned when they share none.
2. **Condition on the first class.** Its wrong values are some set $W$ of size $\ell - 1$. By symmetry the answer does not depend on which set.
3. **Count.** The second class's wrong set is one of $\binom{m-1}{\ell-1}$ equally likely sets. Those disjoint from $W$ are chosen from the $m - 1 - (\ell - 1) = m - \ell$ wrong values outside $W$: $\binom{m-\ell}{\ell-1}$ of them.
4. **Evaluate.** For $m = 8$, $\ell = 3$: $\binom52/\binom72 = 10/21 = 0.4762$ (simulated: 0.4777). For $m = 8$, $\ell = 2$: $6/7 = 0.8571$ (0.8565). For $m = 20$, $\ell = 3$: $\binom{17}{2}/\binom{19}{2} = 136/171 = 0.7953$ (0.7966).

The probability rises with $m$: with more possible values, two random wrong sets are less likely to overlap. In the hospital releases of the chapter, the wrong values are not uniform but drawn from similar patients, and two distinct 3-diverse releases pinned down far fewer diagnoses (0.27% of shared patients) because their classes held many records and several values each.

*Diversity in each release does not survive intersection. The more values the attribute can take, the more likely two diverse classes are to agree on only the truth.*
:::

## Mondrian class sizes {#sec-sol-anon-5}

> **Exercise 5.** **Mondrian class sizes.** Show that if no two records share a value on any quasi-identifier, every Mondrian class has between $k$ and $2k - 1$ records, and deduce that the linkage success lies in $[1/(2k-1), 1/k]$. Explain why the hospital table has classes larger than $2k - 1$. *Check: at $k = 5$, linkage between 0.1111 and 0.2000.* *(Medians, @prp-anon-k-link)*

::: {.callout-tip collapse="true" title="Solution"}
1. **Classes are at least $k$.** The whole table has at least $k$ records, and a split is made only when both halves have at least $k$. By induction, every class does.
2. **What a split does.** With no ties, splitting a box of $s$ records at the median puts $\lfloor s/2\rfloor$ records on one side and $\lceil s/2\rceil$ on the other (or $\lfloor s/2\rfloor + 1$ and $\lceil s/2\rceil - 1$, the other choice of side for the median, which is never more balanced).
3. **When a split is allowed.** It is allowed exactly when the smaller side has at least $k$, that is $\lfloor s/2\rfloor \ge k$, that is $s \ge 2k$. Every box of $2k$ or more records has a quasi-identifier with a positive range, because no two records tie, so it is split.
4. **So classes are at most $2k - 1$.** A box that is not split has $s \le 2k - 1$.
5. **The linkage success.** The upper end, $J/n \le 1/k$, is @prp-anon-k-link. The same count with classes of at most $2k - 1$ records gives $n \le J(2k - 1)$, so $J/n \ge 1/(2k - 1)$. The lower end is attained when every class has exactly $2k - 1$ records. At $k = 5$ that is $[0.1111, 0.2000]$.
6. **Ties.** Ages are whole years and sex has two values, so many records share values. A box in which, say, 12 patients share an age and sex, and whose ZIP codes cannot be split into two halves of at least 5, stays whole. On a fresh draw of the hospital table, Mondrian's classes have sizes 5 to 12 at $k = 5$ and 10 to 21 at $k = 10$. Still, 97.7% and 98.5% of classes are within $[k, 2k - 1]$, and the linkage success, 0.1525 and 0.0738, is inside the interval.

*Mondrian keeps classes between $k$ and $2k - 1$ when values are distinct, so its linkage success is within a factor of 2 of the bound $1/k$. With ties, classes can be larger than $2k - 1$.*
:::
