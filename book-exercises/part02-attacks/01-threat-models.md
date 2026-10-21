# Exercises: Threat models and privacy games

Exercises for [Threat models and privacy games](../../book-full/part02-attacks/01-threat-models.md), each with a worked solution. Try each exercise before opening its solution. The script `code/part02-attacks/solutions_threat_models.py` computes every number quoted here; run it with `uv run python code/part02-attacks/solutions_threat_models.py`.

The solutions use the chapter's notation. $b = 1$ is the world where the target is in the data, with view distribution $Q$; $b = 0$ is the other world, with view distribution $P$. An attack's true positive rate (TPR) is $\Pr[b' = 1 \mid b = 1]$, its false positive rate (FPR) is $\Pr[b' = 1 \mid b = 0]$, and its advantage is $\mathrm{TPR} - \mathrm{FPR}$. The total variation (TV) distance and the hockey-stick divergence $E_\gamma$ are those of [Divergences](../../book-full/part01-prerequisites/03-divergences.md).

## Winning with a prior {#sec-sol-threat-1}

> **Exercise 1.** **Winning with a prior.** In a two-world game with prior $\pi_1$ on $b = 1$, show that no attack wins with probability more than $\max(\pi_0, \pi_1) + \min(\pi_0, \pi_1)\,\mathrm{TV}(P, Q)$. *Hint: start from @eq-threat-bayes and use $E_\gamma \le E_1$ for $\gamma \ge 1$.* *Check: for the noisy count with $\sigma = 1$ and $\pi_1 = 0.1$, the bound is $0.9 + 0.1 \times 0.3829 = 0.9383$, against the exact best 0.9013.* *(Bayes-optimal guess, hockey-stick)*

::: {.callout-tip collapse="true" title="Solution"}
1. **Case $\pi_1 \le \frac12$.** By @eq-threat-bayes the best attack wins with probability $\pi_0 + \pi_1E_\gamma(Q\|P)$ with $\gamma = \pi_0/\pi_1 \ge 1$.
2. **$E_\gamma$ is non-increasing in $\gamma$.** For every event $A$ and $\gamma \ge 1$, $Q(A) - \gamma P(A) \le Q(A) - P(A)$, because $P(A) \ge 0$. Maximizing both sides over $A$ gives $E_\gamma(Q\|P) \le E_1(Q\|P) = \mathrm{TV}(P, Q)$.
3. **Combine.** The best winning probability is at most $\pi_0 + \pi_1\mathrm{TV}$, and here $\pi_0 = \max(\pi_0, \pi_1)$, $\pi_1 = \min(\pi_0, \pi_1)$.
4. **Case $\pi_1 > \frac12$.** Rename the worlds: call $b = 0$ the "positive" one. The game is the same, with prior $\pi_0 < \frac12$ on the renamed positive world and the roles of $P$ and $Q$ swapped. TV is symmetric, so step 3 gives $\pi_1 + \pi_0\mathrm{TV}$, again $\max + \min\cdot\mathrm{TV}$.
5. **Check.** With $\pi_1 = \frac12$ the bound is $\frac12 + \frac12\mathrm{TV} = 0.6915$ for the noisy count, and it is attained. With $\pi_1 = 0.1$ it is $0.9 + 0.1 \times 0.3829 = 0.9383$, while the exact best is $0.9013$.

*The bound is loose for skewed priors because it uses the likelihood-ratio threshold 1, while the best attack needs a ratio above 9 before it overrules the prior. Few views carry that much evidence.*
:::

## The exact count in an unknown crowd {#sec-sol-threat-2}

> **Exercise 2.** **The exact count in an unknown crowd.** Show that $\mathrm{TV}\big(\mathrm{Binomial}(m, \frac12), 1 + \mathrm{Binomial}(m, \frac12)\big) = \binom{m}{\lfloor m/2\rfloor}2^{-m}$, and use Stirling's formula to show it is about $\sqrt{2/(\pi m)}$. *Hint: write the sum $\sum_k(q_k - p_k)_+$ with $q_k = p_{k-1}$, and notice it telescopes.* *Check: 0.0796 for $m = 100$, against $\sqrt{2/(100\pi)} = 0.0798$.* *(TV, average case)*

::: {.callout-tip collapse="true" title="Solution"}
1. **The two distributions.** Let $p_k = \binom mk2^{-m}$ for $0 \le k \le m$ and $p_k = 0$ otherwise. Without the target the count is $k$ with probability $p_k$; with it, with probability $q_k = p_{k-1}$.
2. **Where $q$ beats $p$.** $p_k/p_{k-1} = (m - k + 1)/k$, which is below 1 exactly when $k > (m + 1)/2$. So $q_k > p_k$ exactly for $k \ge j := \lfloor (m + 1)/2\rfloor + 1$, up to $k = m + 1$.
3. **Telescope.** $\mathrm{TV} = \sum_{k \ge j}(p_{k-1} - p_k) = p_{j-1} - p_{m+1} = p_{\lfloor(m+1)/2\rfloor}$.
4. **The central weight.** For even $m$, $\lfloor (m+1)/2\rfloor = m/2$. For odd $m$ it is $(m+1)/2 = \lceil m/2\rceil$, and $\binom{m}{\lceil m/2\rceil} = \binom{m}{\lfloor m/2\rfloor}$. Either way $\mathrm{TV} = \binom{m}{\lfloor m/2\rfloor}2^{-m}$.
5. **Stirling.** For $m = 2j$, $n! \approx \sqrt{2\pi n}(n/e)^n$ gives $\binom{2j}{j} \approx \frac{\sqrt{4\pi j}(2j/e)^{2j}}{2\pi j(j/e)^{2j}} = \frac{4^j}{\sqrt{\pi j}}$, so $\mathrm{TV} \approx 1/\sqrt{\pi j} = \sqrt{2/(\pi m)}$.
6. **Check.** Summing $\frac12\sum_k|p_k - q_k|$ directly and taking the central weight both give 0.2461 for $m = 10$ and 0.0796 for $m = 100$. Stirling gives 0.2523 and 0.0798.

*Knowing the other records turns this TV from 0.08 into 1: the exact count reveals the target's presence completely. Only the adversary's ignorance hides it.*
:::

## How large a crowd? {#sec-sol-threat-3}

> **Exercise 3.** **How large a crowd?** With $\sigma = 1$ and $m$ unknown other bits, how large must $m$ be for the average-case membership advantage on the noisy count to drop to 0.05? Use the normal approximation, then check with the exact formula. *Check: about $m = 250$.* *(Mixing, normal approximation)*

::: {.callout-tip collapse="true" title="Solution"}
1. **Normal approximation.** The unknown count $K \sim \mathrm{Binomial}(m, \frac12)$ has variance $m/4$. Treating it as normal, the release is $\mathcal N(\frac m2, s^2)$ against $\mathcal N(\frac m2 + 1, s^2)$ with $s^2 = m/4 + \sigma^2$, and the TV is $2\Phi(1/(2s)) - 1$.
2. **Solve for $s$.** $2\Phi(1/(2s)) - 1 = 0.05$ means $\Phi(1/(2s)) = 0.525$, so $1/(2s) = \Phi^{-1}(0.525) = 0.06271$ and $s = 7.974$, $s^2 = 63.58$.
3. **Solve for $m$.** $m = 4(s^2 - \sigma^2) = 4(63.58 - 1) = 250.3$.
4. **Exact check.** Computing the exact average-case TV of the chapter for each $m$, the smallest $m$ with TV at most 0.05 is $m = 250$, where TV is 0.0500. The normal approximation also gives 0.0500 there.

*To get the same advantage against an adversary who knows the other records, the noise alone would need $\sigma = s = 7.97$: the 250 unknown records supply the rest of the variance. That protection disappears the moment the adversary learns them.*
:::

## Reconstruction from one count {#sec-sol-threat-4}

> **Exercise 4.** **Reconstruction from one count.** Prove that the best reconstruction of $m$ uniform bits from their exact sum recovers an expected fraction $\mathbb E[\max(K, m - K)]/m$ with $K \sim \mathrm{Binomial}(m, \frac12)$. Compute it by hand for $m = 1, 2, 3$. *Check: 1, 0.75, 0.75; and 0.6230 for $m = 10$.* *(Reconstruction game, posterior)*

::: {.callout-tip collapse="true" title="Solution"}
1. **Score by bits.** A reconstruction is a guess $x'_1, \dots, x'_m$ that may depend on $K$. Its expected gain is $\frac1m\sum_i\Pr[x'_i = x_i]$, so each bit can be optimized separately.
2. **Posterior of a bit.** Given $K = k$, all $\binom mk$ arrangements are equally likely, and bit $i$ is 1 in $\binom{m-1}{k-1}$ of them. So $\Pr[x_i = 1 \mid K = k] = \binom{m-1}{k-1}/\binom mk = k/m$.
3. **Best guess.** Guess 1 if $k/m > \frac12$, else 0 (at $k = m/2$ both are equally good). The expected number of correct bits given $K = k$ is $\max(k, m - k)$. Averaging over $K$ gives the claim.
4. **By hand.**

| $m$ | Distribution of $K$ | $\max(K, m - K)$ | Expected fraction |
| --- | --- | --- | --- |
| 1 | $0, 1$ with $\frac12, \frac12$ | $1, 1$ | 1 |
| 2 | $0, 1, 2$ with $\frac14, \frac12, \frac14$ | $2, 1, 2$ | $\frac12(\frac14 \cdot 2 + \frac12 \cdot 1 + \frac14 \cdot 2) = 0.75$ |
| 3 | $0, 1, 2, 3$ with $\frac18, \frac38, \frac38, \frac18$ | $3, 2, 2, 3$ | $\frac13 \cdot \frac{3 + 6 + 6 + 3}{8} = 0.75$ |

: The best reconstruction of $m$ uniform bits from their exact sum, for small $m$. {#tbl-sol-threat-recon}

5. **Check.** The script gives 1, 0.75, 0.75 and, for $m = 10$, 0.6230.

*For $m = 1$ the count is the record. From $m = 2$ on, one count says less and less about each bit, and the gain over guessing shrinks like $1/\sqrt{2\pi m}$.*
:::

## Releasing a record from a small domain {#sec-sol-threat-5}

> **Exercise 5.** **Releasing a record from a small domain.** Redo the random-record release when records are uniform on only $v$ values, so that a non-member can collide with the released record. Find TPR, FPR and the precision at a fair prior. *Check: for $n = v = 1000$, $\mathrm{TPR} = 0.001999$, $\mathrm{FPR} = 0.001$, precision 0.6666.* *(Membership game, precision)*

::: {.callout-tip collapse="true" title="Solution"}
1. **FPR.** If $z \notin D$, the released record is an independent uniform value, equal to $z$ with probability $1/v$.
2. **TPR.** If $z \in D$, the target itself is released with probability $1/n$. Otherwise (probability $1 - 1/n$) another record is released, which equals $z$ with probability $1/v$. So $\mathrm{TPR} = \frac1n + (1 - \frac1n)\frac1v$.
3. **Advantage.** $\mathrm{TPR} - \mathrm{FPR} = \frac1n - \frac1{nv} = \frac1n(1 - \frac1v)$.
4. **Precision at a fair prior.** By @eq-threat-precision, $\mathrm{TPR}/(\mathrm{TPR} + \mathrm{FPR})$.
5. **Check.** With $n = v = 1000$: $\mathrm{TPR} = 0.001 + 0.999 \times 0.001 = 0.001999$, $\mathrm{FPR} = 0.001$, advantage 0.000999, precision $0.001999/0.002999 = 0.6666$. Two million simulated games per world give $\mathrm{TPR} = 0.001945$ and $\mathrm{FPR} = 0.001004$.

*Collisions with common values give the target deniability: seeing their value released is now only 2-to-1 evidence. With distinct records ($v \to \infty$) the precision is 1.*
:::

## The inner-product attack {#sec-sol-threat-6}

> **Exercise 6.** **The inner-product attack.** For the centroid detector, the attack "say member when $\langle z, \theta\rangle$ is large" uses only the inner product. Compute the mean and variance of $\langle z, \theta\rangle$ in both worlds, approximate both by normals, and show that the best advantage is about $2\Phi\big(\frac12\sqrt{d/n}\big) - 1$. *Check: 0.2763 for $n = 100$, $d = 50$; the simulated attack gets 0.2771, and the likelihood-ratio attack of @tbl-threat-access 0.2789.* *(Gaussian approximation, access levels)*

::: {.callout-tip collapse="true" title="Solution"}
1. **Member world.** $\theta = (z + S)/n$ with $S \sim \mathcal N(0, (n - 1)I_d)$ independent of $z \sim \mathcal N(0, I_d)$, so $\langle z, \theta\rangle = \|z\|^2/n + \langle z, S\rangle/n$.
    - Mean: $\mathbb E\|z\|^2 = d$ and $\mathbb E\langle z, S\rangle = 0$, so the mean is $d/n$.
    - Variance: $\operatorname{Var}\|z\|^2 = 2d$ (a chi-square with $d$ degrees of freedom). Given $z$, $\langle z, S\rangle \sim \mathcal N(0, (n-1)\|z\|^2)$, so its variance is $(n - 1)d$, and it is uncorrelated with $\|z\|^2$ because its conditional mean is 0. Total: $(2d + (n-1)d)/n^2$.
2. **Non-member world.** $\theta \sim \mathcal N(0, I_d/n)$ independent of $z$. Given $z$, $\langle z, \theta\rangle \sim \mathcal N(0, \|z\|^2/n)$: mean 0, variance $d/n$.
3. **Numbers.** For $n = 100$, $d = 50$: mean 0.5 against 0; standard deviations 0.7106 and 0.7071.
4. **Normal approximation.** Both standard deviations are about $\sqrt{d/n}$ (the member variance is $(n+1)d/n^2 \approx d/n$). Two normals with equal spread $\sqrt{d/n}$ and means $d/n$ apart are a Gaussian pair with standardized shift $\mu = (d/n)/\sqrt{d/n} = \sqrt{d/n}$, whose TV is $2\Phi(\mu/2) - 1$ ([Divergences](../../book-full/part01-prerequisites/03-divergences.md)). The best threshold attack on the score reaches it.
5. **Check.** $2\Phi(\frac12\sqrt{0.5}) - 1 = 0.2763$. Simulating the attack over 100,000 games per world gives 0.2771. The likelihood-ratio attack (0.2789) also uses $\|\theta\|^2$ and the target-dependent threshold, which adds a little.

*The advantage depends on $d/n$, the number of parameters per training record. That ratio, not the model's accuracy, decides how much one record moves the model. [Lower bounds](../../book-full/part03-dp-foundations/10-lower-bounds.md) in Part 3 turn this attack into a limit on what any defence can achieve.*
:::

## Mixing hides {#sec-sol-threat-7}

> **Exercise 7.** **Mixing hides.** One other record with a uniform bit $\vartheta$; the mechanism releases the exact count $\vartheta + b$. Compute the best advantage of an adversary who knows $\vartheta$ and of one who does not, and check them against @eq-threat-mixing. *Check: 1 and 0.5.* *(Fact 3)*

::: {.callout-tip collapse="true" title="Solution"}
1. **Knowing $\vartheta$.** For each $\vartheta$, the release is $\vartheta$ when $b = 0$ and $\vartheta + 1$ when $b = 1$: two point masses at different places. $\mathrm{TV}(P_\vartheta, Q_\vartheta) = 1$, and the attack "say member when the count exceeds $\vartheta$" is always right.
2. **Not knowing $\vartheta$.** With $b = 0$ the count is 0 or 1 with probability $\frac12$ each; with $b = 1$ it is 1 or 2.

| Count | $\bar p$ ($b = 0$) | $\bar q$ ($b = 1$) | $\lvert\bar p - \bar q\rvert$ |
| --- | --- | --- | --- |
| 0 | $\frac12$ | 0 | $\frac12$ |
| 1 | $\frac12$ | $\frac12$ | 0 |
| 2 | 0 | $\frac12$ | $\frac12$ |

: The exact count with one unknown other bit. {#tbl-sol-threat-mixing}

3. **TV of the mixtures.** $\frac12(\frac12 + 0 + \frac12) = \frac12$. The best attack says "member" when the count is 2: $\mathrm{TPR} = \frac12$, $\mathrm{FPR} = 0$.
4. **Against @eq-threat-mixing.** $\mathrm{TV}(\bar P, \bar Q) = 0.5 \le \mathbb E_\vartheta\mathrm{TV}(P_\vartheta, Q_\vartheta) = 1 \le \max_\vartheta\mathrm{TV} = 1$. The script prints 0.5000 and 1.0000.

*Half the time the count is 1, which both worlds produce equally often, and the target is hidden. The other half the count gives the answer away, even to the adversary who knows nothing about the other record.*
:::
