# Exercises: Reconstruction attacks

Exercises for [Reconstruction attacks](../../book-full/part02-attacks/09-reconstruction.md), each with a worked solution. Try each exercise before opening its solution. The script `code/part02-attacks/solutions_reconstruction.py` computes every number quoted here; run it with `uv run python code/part02-attacks/solutions_reconstruction.py`.

The solutions use the chapter's notation: $d \in \{0, 1\}^n$ is the secret database, a query is a subset $S$ with indicator $q$, the curator answers $a = Ad + e$, $E$ bounds every $|e_j|$, and $\beta$ is a fraction of wrong bits. Logarithms are natural unless written $\log_2$.

## A rounding curator {#sec-sol-reconstruction-1}

> **Exercise 1.** **A rounding curator.** A curator answers all $2^n$ subset sums rounded to the nearest multiple of $w$, so its error is at most $w/2$. Explain why the worst-case analysis allows $4E = 2w$ wrong bits while, on the database of @tbl-recon-exp with $w = 4$, only $d$ itself is consistent. *Hint: flipping bit $i$ changes the sum of every subset containing $i$ by 1, and rounding to multiples of 4 separates $s$ from $s + 1$ when $s \equiv 1 \pmod 4$. Check: at least 128 of the 1024 subsets separate $d$ from every one-bit change of it.* *(Exponential attack)*

::: {.callout-tip collapse="true" title="Solution"}
1. **What the worst case assumes.** The bound $4E$ holds for every curator whose error is at most $E$, including one that chooses its errors knowing both $d$ and a decoy $c$. The tightness argument used exactly such a curator: it answered the midpoint of $\sum_Sc_i$ and $\sum_Sd_i$, which depends on $c$.
2. **What a rounding curator does instead.** Its answer is $R(\sum_Sd_i)$ with the fixed function $R(s) = w\lfloor s/w + \frac12\rfloor$. The attacker knows $R$, so it can demand more than "within $w/2$": a candidate $c$ survives only if $R(\sum_Sc_i) = R(\sum_Sd_i)$ for every $S$. The error is no longer chosen against the attacker; it is a known function of the truth.
3. **When rounding separates $s$ from $s + 1$.** With $w = 4$, $R(s) = 4\lfloor s/4 + \frac12\rfloor$ jumps from $4k$ to $4k + 4$ between $s = 4k + 1$ and $s = 4k + 2$. So $R(s) \ne R(s + 1)$ exactly when $s \equiv 1 \pmod 4$.
4. **Count the separating subsets for a one-bit flip.** The database $d = (1, 1, 1, 0, 0, 0, 0, 0, 0, 1)$ has four ones. Only subsets containing the flipped bit $i$ change, and there are $2^9 = 512$ of them.
   - *Flipping a 0 to 1.* The sum moves from $s$ to $s + 1$, where $s$ counts the ones of $d$ in $S$. Those come from the four ones, each in $S$ with probability $\frac12$ independently, so $s \equiv 1 \pmod 4$ means $s = 1$: $\binom41 = 4$ of the 16 choices for the ones, times $2^5$ choices for the five other zeros. That is $4 \times 32 = 128$ separating subsets.
   - *Flipping a 1 to 0.* The sum moves from $s = 1 + j$ to $j$, where $j$ counts the other three ones in $S$, and it is separated when $j \equiv 1 \pmod 4$, that is $j = 1$: $\binom31 = 3$ of 8 choices, times $2^6$ for the six zeros, giving $192$.
   The fewest is 128, the check value. The script finds the same by brute force, and 256 for $w = 2$, where $R$ separates $s$ from $s + 1$ whenever $s$ is even.
5. **All other candidates.** The script checks every one of the 1024 candidates against all 1024 rounded answers: only $d$ survives, for $w = 2$ and for $w = 4$. Against a curator that picks its errors adversarially with $E = 2$, 912 candidates survive, as far as 8 bits from $d$.

*An error bound alone does not describe a curator. Deterministic errors that the attacker can recompute leak as much as exact answers; only errors the attacker cannot predict can hide anything.*
:::

## Another Paley–Zygmund threshold {#sec-sol-reconstruction-2}

> **Exercise 2.** **Another Paley–Zygmund threshold.** Redo the lemma with $\theta = \frac19$. Show that the escape probability becomes $(8/9)^2/3 = 64/243$ at threshold $\sqrt{\beta n}/18$, and compute the error tolerance and the number of queries the theorem then needs at $n = 1000$, $\beta = 0.1$, $\eta = 0.01$. *Check: escape 0.2634, $E \le 0.0278$, $m \ge 22\,617$.* Which choice is better? *(Paley–Zygmund, union bound)*

::: {.callout-tip collapse="true" title="Solution"}
1. **Steps 1–6 of the lemma do not involve $\theta$.** They give $\Pr[|Y| > \sqrt\theta\,\sigma] \ge (1 - \theta)^2/3$ with $\sigma \ge \sqrt{\beta n}/6$.
2. **Set $\theta = \frac19$.** Then $\sqrt\theta = \frac13$, so the threshold is $\frac13\cdot\frac{\sqrt{\beta n}}6 = \frac{\sqrt{\beta n}}{18}$, and the probability is $(8/9)^2/3 = \frac{64}{81 \cdot 3} = \frac{64}{243} = 0.2634$.
3. **Error tolerance.** The theorem needs $2E + \frac12 \le \sqrt{\beta n}/18 = 10/18 = 0.5556$, so $E \le (0.5556 - 0.5)/2 = 0.0278$.
4. **Queries.** Step 4 of the theorem now uses survival probability $1 - \frac{64}{243} = \frac{179}{243}$ per query, so
$$m \ge \frac{1000\ln1001 + \ln100}{\ln(243/179)} = \frac{6913.4}{0.3057} = 22\,617.$$
5. **Compare with $\theta = \frac14$:** escape 0.1875, $E \le 0.1667$, $m \ge 33\,295$.

| $\theta$ | Escape probability | Threshold at $n = 1000$, $\beta = 0.1$ | Error tolerance | Queries needed |
| --- | --- | --- | --- | --- |
| $\frac14$ | 0.1875 | 0.8333 | 0.1667 | 33 295 |
| $\frac19$ | 0.2634 | 0.5556 | 0.0278 | 22 617 |

: The lemma's threshold trades the error the theorem tolerates against the number of queries it needs. {#tbl-sol-reconstruction-theta}

*A smaller $\theta$ makes each query a likelier catch, so fewer queries are needed, but only for a much smaller error. Since the point of the theorem is the error scale, the larger $\theta = \frac14$ is the better choice; neither changes the $\sqrt{\beta n}$ scaling.*
:::

## The fourth moment {#sec-sol-reconstruction-3}

> **Exercise 3.** **The fourth moment.** Prove $\mathbb E\big(\sum_is_ia_i\big)^4 = 3(\sum_ia_i^2)^2 - 2\sum_ia_i^4$ by counting index patterns, and check it by enumeration for $a = (1, 2)$ and $a = (\frac12, \frac12, \frac12, \frac12)$. *Check: 41 and 2.5.* *(Fact 2)*

::: {.callout-tip collapse="true" title="Solution"}
1. **Expand.** $\big(\sum_is_ia_i\big)^4 = \sum_{i, j, k, l}s_is_js_ks_l\,a_ia_ja_ka_l$, a sum over all ordered quadruples of indices.
2. **Which terms survive the expectation.** The signs are independent with $\mathbb E s = \mathbb E s^3 = 0$ and $s^2 = s^4 = 1$. A quadruple in which some index appears once or three times has expectation 0. The survivors are the quadruples in which every index appears an even number of times.
3. **All four equal.** $i = j = k = l$: $n$ quadruples, contributing $\sum_ia_i^4$.
4. **Two distinct pairs.** The positions $\{1, 2, 3, 4\}$ split into two pairs in 3 ways ($\{12, 34\}$, $\{13, 24\}$, $\{14, 23\}$), and the two pairs carry distinct indices $i \ne k$ in order. Each contributes $a_i^2a_k^2$, so the total is $3\sum_{i \ne k}a_i^2a_k^2 = 3\big((\sum_ia_i^2)^2 - \sum_ia_i^4\big)$.
5. **Add:** $\sum_ia_i^4 + 3(\sum_ia_i^2)^2 - 3\sum_ia_i^4 = 3(\sum_ia_i^2)^2 - 2\sum_ia_i^4$.

**Check by enumeration.**

- $a = (1, 2)$: the four sign patterns give sums $3, -1, 1, -3$, whose fourth powers average $(81 + 1 + 1 + 81)/4 = 41$. The formula gives $3 \times 5^2 - 2 \times 17 = 75 - 34 = 41$.
- $a = (\frac12, \frac12, \frac12, \frac12)$: the sum is half the number of $+$ signs minus half the number of $-$ signs, so it is $\pm2$ (2 patterns), $\pm1$ (8 patterns) or 0 (6 patterns). The average fourth power is $(2 \times 16 + 8 \times 1)/16 = 2.5$. The formula gives $3 \times 1^2 - 2 \times \frac14 = 2.5$.

*The ratio of the fourth moment to the squared second moment is at most 3, the value for a normal variable. That ratio is what the Paley–Zygmund step needs, and it holds for every weight vector, not just for large $n$.*
:::

## The inverse of $I + b\mathbf 1\mathbf 1^\top$ {#sec-sol-reconstruction-4}

> **Exercise 4.** **The inverse of $I + b\mathbf 1\mathbf 1^\top$.** Show $(I + b\mathbf 1\mathbf 1^\top)^{-1} = I - \frac{b}{1 + nb}\mathbf 1\mathbf 1^\top$, and deduce that the diagonal of $\big(\mathbb E[A^\top A]\big)^{-1}$ is $\frac4m\cdot\frac{n}{n + 1}$. *Check: $\frac{n}{n + 1} = 0.9980$ at $n = 500$.* *(Linear algebra for least squares)*

::: {.callout-tip collapse="true" title="Solution"}
1. **Multiply out.** For any $c$, $(I + b\mathbf 1\mathbf 1^\top)(I - c\mathbf 1\mathbf 1^\top) = I + b\mathbf 1\mathbf 1^\top - c\mathbf 1\mathbf 1^\top - bc\,\mathbf 1(\mathbf 1^\top\mathbf 1)\mathbf 1^\top$.
2. **Use $\mathbf 1^\top\mathbf 1 = n$.** The product is $I + (b - c - bcn)\mathbf 1\mathbf 1^\top$.
3. **Choose $c$** to make the bracket zero: $c(1 + bn) = b$, so $c = \frac{b}{1 + nb}$. The product is then $I$, which proves the formula. (This is the Sherman–Morrison formula for a rank-one update of the identity.)
4. **Apply it.** $\mathbb E[A^\top A] = \frac m4(I + \mathbf 1\mathbf 1^\top)$, so with $b = 1$, $\big(\mathbb E[A^\top A]\big)^{-1} = \frac4m\big(I - \frac{1}{n + 1}\mathbf 1\mathbf 1^\top\big)$.
5. **The diagonal.** Each diagonal entry is $\frac4m\big(1 - \frac1{n + 1}\big) = \frac4m\cdot\frac{n}{n + 1}$. At $n = 500$ the factor is $500/501 = 0.9980$.
6. **Numerical check.** The script inverts $I + \mathbf 1\mathbf 1^\top$ for $n = 5$ directly and compares it with the formula; the entries agree to rounding error ($2.2 \times 10^{-16}$).

*The all-ones direction carries the total count, which every random query measures with a large weight, so it is estimated well. Every other direction is estimated with variance about $4\sigma^2/m$, and the correction $\frac{n}{n + 1}$ is negligible.*
:::

## Queries that beat noise {#sec-sol-reconstruction-5}

> **Exercise 5.** **Queries that beat noise.** Using @eq-recon-ls-random, find the number of queries per person at which least squares recovers 90% of the bits when $\sigma = 5\sqrt n$. Then check the prediction for $\sigma = 2\sqrt n$ by simulation at $n = 100$. *Check: $m/n = 657.9$; at $\sigma = 2\sqrt n$, $m = 10\,611$ queries give 0.9100 in simulation against 0.9011 predicted.* *(Least squares under random noise)*

::: {.callout-tip collapse="true" title="Solution"}
1. **Set the prediction to 0.9.** $\Phi\big(\sqrt{m - n}/(4\sigma)\big) = 0.9$ means $\sqrt{m - n}/(4\sigma) = \Phi^{-1}(0.9) = 1.2816$.
2. **Write $\sigma = c\sqrt n$ and solve.** $\sqrt{m - n} = 4c\sqrt n \times 1.2816$, so $m/n = 1 + (4c \times 1.2816)^2$.
3. **At $c = 5$:** with $\Phi^{-1}(0.9) = 1.28155$, $(20 \times 1.28155)^2 = 656.95$, so $m/n = 657.95$, which the script prints as 657.9. At $c = 2$: $(8 \times 1.28155)^2 = 105.11$, so $m/n = 106.1$.
4. **Simulate.** At $n = 100$ and $c = 2$ ($\sigma = 20$), $m = 10\,611$ queries. Over four repetitions, least squares recovers 0.9100 of the bits. The finite-$n$ prediction, with the factor $\frac{n}{n + 1}$ and $m - n - 1$ in place of $m - n$, is 0.9011.

*A curator that adds independent noise of fixed size to each answer protects nobody against an attacker with enough queries: the number needed grows only with the square of the noise.*
:::

## Information bound for rounded counts {#sec-sol-reconstruction-6}

> **Exercise 6.** **Information bound for rounded counts.** A curator rounds every count of $n = 1000$ bits to a multiple of 10. How many queries does any attack need to get at most 1% of the bits wrong? *Check: $h(0.01) = 0.0808$, $V = 102$, $m \ge 137.8$, so at least 138.* *(Fano, entropy)*

::: {.callout-tip collapse="true" title="Solution"}
1. **Values per answer.** A count is in $\{0, \dots, 1000\}$, and rounding to the nearest multiple of 10 gives one of $0, 10, \dots, 1000$: 101 values. The chapter's bound uses the safe count $\lfloor n/w\rfloor + 2 = 102$, so $\log_2V = \log_2102 = 6.6724$ bits.
2. **Bits needed.** $h(0.01) = -0.01\log_20.01 - 0.99\log_20.99 = 0.0664 + 0.0144 = 0.0808$, so recovering all but 1% of the bits needs $I(d; \text{answers}) \ge n(1 - h(0.01)) = 919.2$ bits.
3. **Divide.** $m \ge 919.2/6.6724 = 137.8$, so at least 138 queries.
4. **Compare.** With exact answers ($V = 1001$, $\log_2V = 9.97$), the bound is 92.2. Rounding to 10 makes each answer carry fewer bits, so more queries are needed; it does not stop the attack.

*Coarsening the answers raises the number of queries any attack needs, but only by the factor $\log_2(n + 1)/\log_2(n/w + 2)$; it does not change its order.*
:::

## A sampling curator that keeps more {#sec-sol-reconstruction-7}

> **Exercise 7.** **A sampling curator that keeps more.** Repeat the sampling-curator analysis with keep probability $p = 0.9$. Find the Hoeffding error bound at $n = 500$, $m = 2000$, $\beta = 0.05$, and the ceiling on the fraction any attacker recovers. *Check: 59.03 and 0.95; in simulation the largest error is 12.00 and least squares recovers 0.9420.* *(Hoeffding, union bound)*

::: {.callout-tip collapse="true" title="Solution"}
1. **The error terms.** Each term $d_i(t_i/p - 1)$ is 0 when $d_i = 0$, and otherwise $\frac1{0.9} - 1 = 0.111$ with probability 0.9 or $-1$ with probability 0.1. The mean is $0.9 \times 0.111 - 0.1 = 0$ and the range is $\frac1p = 1.111$.
2. **Hoeffding and the union bound,** exactly as in the chapter: $\Pr[\text{some error} \ge u] \le 2m\exp(-2p^2u^2/n)$. Setting it to $\beta = 0.05$:
$$u = \sqrt{\frac{n\ln(2m/\beta)}{2p^2}} = \sqrt{\frac{500\ln80\,000}{2 \times 0.81}} = \sqrt{\frac{500 \times 11.290}{1.62}} = 59.03.$$
3. **The ceiling.** The people outside the sample $T$ are hidden, so the attacker is right on at most all of the sampled people (fraction 0.9) and half of the others: $0.9 + 0.1/2 = 0.95$.
4. **Simulation.** The script keeps 445 of the 500 people. The largest error over 2000 queries is 12.00, well below the bound, and least squares recovers 0.9420: it gets the 445 sampled people right and, among the 55 others, exactly those whose bit is 0 (it estimates their bit as 0).

*Keeping more people lowers the error and exposes more of them. The curator's hiding power is exactly the fraction it leaves out, and it buys that by making each answer less accurate.*
:::

## Redundant statistics {#sec-sol-reconstruction-8}

> **Exercise 8.** **Redundant statistics.** Show that withholding any one of the total, female and male means from @tbl-recon-census-published leaves the attacker's information unchanged, and explain why withholding the married mean does not. *Check: withholding the married mean leaves 45 consistent blocks; withholding any of the other five means leaves 1.* *(Census reconstruction)*

::: {.callout-tip collapse="true" title="Solution"}
1. **Means are sums.** Given the counts, which are published, a mean is equivalent to the sum of ages in its group: total 240, female 121, male 119.
2. **The sex groups partition the block.** Every person is female or male, so the total sum is the female sum plus the male sum: $121 + 119 = 240$. Any two of the three sums determine the third. Withholding one of the three means removes nothing the attacker can't recompute.
3. **The same for race.** Every person is Black or White, so the Black, White and total sums satisfy the same relation, and withholding the Black or White mean changes nothing either. The script confirms one consistent block in all five cases.
4. **Marital status is different.** The table publishes the married group but not the single group. The married sum (191) is the only statistic that splits the total age between married and single people; the single sum $240 - 191 = 49$ comes from it. Withholding it removes a genuine equation. The married median still constrains the block, but 45 blocks now fit, with 7 distinct patterns of sex, race and marital status; 51% of them have the true pattern.

*Published statistics are rarely independent. Before withholding a number to protect a table, check whether the rest of the table implies it; most of the time it does.*
:::
