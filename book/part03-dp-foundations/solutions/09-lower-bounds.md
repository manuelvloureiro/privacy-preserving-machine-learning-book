# Exercises: Lower bounds: what privacy costs

Exercises for [Lower bounds: what privacy costs](../09-lower-bounds.md), each with a worked solution. Try each exercise before opening its solution. The script `code/solutions_lower_bounds.py` computes every number quoted here; run it with `uv run python book/part03-dp-foundations/code/solutions_lower_bounds.py`.

The solutions use the chapter's notation. $M$ is a differential privacy (DP) mechanism, $\alpha$ an error and $\beta$ a failure probability, and the random dataset of the fingerprinting section has attribute means $p_j$ uniform on $[-1, 1]$ and records $x_i \in \{\pm1\}^d$ with $\mathbb E[x_{ij} \mid p] = p_j$. The examples use $\varepsilon = 1$, $\beta = 0.05$ and $\delta = 10^{-6}$.

## Histograms under replace neighbours {#sec-sol-lower-1}

> **Exercise 1.** **Histograms under replace neighbours.** Redo the histogram lower bound with datasets of a fixed size $n$: $D_0$ is $n$ copies of an element $x_0$, and $D_x$ replaces $k$ of them by $x \ne x_0$. Show $\alpha \ge \frac12\big(\ln((N - 1)(1 - \beta))/\varepsilon - 1\big)$, and show that the Laplace mechanism's error doubles. *Check: at $N = 10^6$, $\varepsilon = 1$, $\beta = 0.05$, the lower bound is still 6.38 and the Laplace error becomes 33.57.* *(Packing)*

::: {.callout-tip collapse="true" title="Solution"}
1. **The candidates.** For each $x \ne x_0$ there is one dataset $D_x$, so $m = N - 1$. Each is $k$ replacements away from $D_0$.
2. **Their histograms.** $h(D_x) = (n - k)e_{x_0} + k\,e_x$. For $x \ne y$, the two histograms agree in bin $x_0$ and differ by $k$ in bins $x$ and $y$, so $\|h(D_x) - h(D_y)\|_\infty = k$.
3. **Disjoint regions.** As in the chapter, the $\ell_\infty$ balls of radius $\alpha$ around them are disjoint when $k > 2\alpha$. Take $k = \lfloor2\alpha\rfloor + 1 \le 2\alpha + 1$.
4. **Packing lemma** with $m = N - 1$: $2\alpha + 1 \ge \ln((N - 1)(1 - \beta))/\varepsilon$, which rearranges to the claim.
5. **The number.** $\ln(N - 1)$ and $\ln N$ differ by about $1/N = 10^{-6}$, so the bound is 6.3821 under both relations.
6. **The upper bound doubles.** Replacing a record lowers one count by 1 and raises another by 1, so the $\ell_1$ sensitivity is 2 and the Laplace scale is $2/\varepsilon$. Every quantile of $\max_j|\eta_j|$ scales with it: $2\times16.79 = 33.57$ (the script computes $2 \times 16.785\ldots$ before rounding).

*The lower bound does not care about the neighbouring relation, because both relations put the packing datasets $k$ steps from $D_0$. The upper bound does, because sensitivity depends on it. So under replace neighbours the gap between Laplace and the lower bound is about twice as large.*
:::

## What packing can force under approximate DP {#sec-sol-lower-2}

> **Exercise 2.** **What packing can force under approximate DP.** Starting from @eq-lower-group, show that $m$ disjoint regions force $m\big(e^{-k\varepsilon}(1 - \beta) - k\,e^{-\varepsilon}\delta\big) \le 1$, and that as $m \to \infty$ this forces only $e^{-k\varepsilon}(1 - \beta) \le k\,e^{-\varepsilon}\delta$. *Check: at $\varepsilon = 1$, $\delta = 10^{-6}$, $\beta = 0.05$ the limit is $k = 12.26$, so the forced integer group distance is 13 for $m = 2^{20}$ and for $m = 2^{1000}$ alike, against $\ln(1/\delta)/\varepsilon = 13.82$.* *(Group privacy, packing)*

::: {.callout-tip collapse="true" title="Solution"}
1. **Group privacy for each region.** @eq-lower-group with $D = D_i$, $D'' = D_0$ and $S = G_i$: $1 - \beta \le e^{k\varepsilon}\Pr[M(D_0) \in G_i] + \delta_k$.
2. **Solve for the share of $D_0$:** $\Pr[M(D_0) \in G_i] \ge e^{-k\varepsilon}(1 - \beta) - e^{-k\varepsilon}\delta_k$.
3. **Bound the $\delta$ term.** $e^{-k\varepsilon}\delta_k = \delta\sum_{j=0}^{k-1}e^{(j - k)\varepsilon}$. Each of the $k$ terms is at most the largest, $e^{-\varepsilon}$ (at $j = k - 1$), so $e^{-k\varepsilon}\delta_k \le k\,e^{-\varepsilon}\delta$.
4. **Sum over the disjoint regions:** $1 \ge \sum_i\Pr[M(D_0) \in G_i] \ge m\big(e^{-k\varepsilon}(1 - \beta) - k\,e^{-\varepsilon}\delta\big)$.
5. **The limit.** Call the bracket the floor. If it is positive, step 4 fails for large enough $m$, so an accurate mechanism must have a $k$ where the floor is $\le 0$. As $m \to \infty$, that is all step 4 says: $e^{-k\varepsilon}(1 - \beta) \le k\,e^{-\varepsilon}\delta$.
6. **The numbers.** Bisection finds the floor's zero at $k = 12.26$. At $k = 12$ the floor is $0.95e^{-12} - 12e^{-1}\times10^{-6} = 5.83\times10^{-6} - 4.41\times10^{-6} = 1.42\times10^{-6}$, and $m = 2^{20} \approx 1.05\times10^6$ makes $m$ times it 1.49, more than 1. So $k \le 12$ is impossible and $k \ge 13$ is forced. At $k = 13$ the floor is already negative, so no larger $m$, not even $2^{1000}$, forces more.

*Under approximate DP, packing hits a ceiling near $\ln(1/\delta)/\varepsilon$ that no number of candidates can lift. The ceiling is 12.26 rather than 13.82 because the $\delta$ term carries the factor $k$, which costs a further $\ln k/\varepsilon$.*
:::

## The lemma for a shrunk mean {#sec-sol-lower-3}

> **Exercise 3.** **The lemma for a shrunk mean.** For $f = c\,\bar x$ with $0 \le c \le 1$, show that $\mathbb E[f\sum_i(x_i - p)] = 2c/3$ exactly, and that the bound in @eq-lower-fp-lemma is $\frac23 - 2(1 - c)\mathbb E|\bar x|$. Compute $\mathbb E|\bar x|$ for $n = 1$ and its limit as $n \to \infty$. *Check: for $n = 5$, $\mathbb E|\bar x| = 0.6$, so $c = 0.75$ gives correlation 0.5000 against the bound 0.3667.* *(Fingerprinting lemma)*

::: {.callout-tip collapse="true" title="Solution"}
1. **The correlation.** By the equality in @eq-lower-fp-lemma, $\mathbb E[f\sum_i(x_i - p)] = 2\mathbb E[p\,f] = 2c\,\mathbb E[p\,\bar x]$. Given $p$, $\mathbb E[\bar x \mid p] = p$, so $\mathbb E[p\,\bar x] = \mathbb E[p^2] = \int_{-1}^1\frac{p^2}2\,dp = \frac13$. The correlation is $2c/3$.
2. **The bound.** $|f - \bar x| = |c\bar x - \bar x| = (1 - c)|\bar x|$, so $\frac23 - 2\mathbb E|f - \bar x| = \frac23 - 2(1 - c)\mathbb E|\bar x|$.
3. **$n = 1$.** $\bar x = x_1 \in \{\pm1\}$, so $\mathbb E|\bar x| = 1$.
4. **$n \to \infty$.** $\bar x \to p$ by the law of large numbers, and $|\bar x| \le 1$, so $\mathbb E|\bar x| \to \mathbb E|p| = \int_{-1}^1\frac{|p|}2\,dp = \frac12$. The script gives 0.5050 at $n = 100$.
5. **$n = 5$.** Summing over the six possible counts of $+1$ and integrating over $p$ gives $\mathbb E|\bar x| = 0.6000$. @tbl-sol-lower-shrunk compares the two sides.

| $c$ | Correlation $2c/3$ | $\mathbb E\lvert f - \bar x\rvert = 0.6(1 - c)$ | Bound $\frac23 - 2\,\mathbb E\lvert f - \bar x\rvert$ |
| --- | --- | --- | --- |
| 0.25 | 0.1667 | 0.4500 | −0.2333 |
| 0.5 | 0.3333 | 0.3000 | 0.0667 |
| 0.75 | 0.5000 | 0.1500 | 0.3667 |
| 1 | 0.6667 | 0 | 0.6667 |

: The fingerprinting lemma for the shrunk mean $c\,\bar x$ with $n = 5$ records, computed exactly. {#tbl-sol-lower-shrunk}

*The bound always holds and is tight only at $c = 1$. Its slack is $\frac{2c}3 - \big(\frac23 - 1.2(1 - c)\big) = (1.2 - \frac23)(1 - c) = 0.5333(1 - c)$: the lemma charges an estimate twice its error, $1.2(1 - c)$, while shrinking costs only $\frac23(1 - c)$ in correlation.*
:::

## Padding recovers $1/\alpha$ {#sec-sol-lower-4}

> **Exercise 4.** **Padding recovers $1/\alpha$.** Let $M$ release the means of $n$ records with $\ell_\infty$ error $\alpha$. Pad a dataset of $n_0$ records with $n - n_0$ records, half all $+1$ and half all $-1$, run $M$, and rescale. Show that this gives an equally private mechanism for $n_0$ records with error $\alpha n/n_0$, choose $n_0$ so that this error is $\alpha_0$, and conclude $n \ge \frac{\alpha_0}{\alpha}\cdot\frac{(2/3 - 2\alpha_0 - 4\beta)d}{(e^\varepsilon - 1)\sqrt d + 4\delta d}$ (up to rounding). Find the best $\alpha_0$. *Check: $\alpha_0 = \frac16 - \beta$, and the constant $\alpha_0(\frac23 - 2\alpha_0 - 4\beta) = (\frac23 - 4\beta)^2/8 = 0.0272$ at $\beta = 0.05$; at $d = 10^4$ the bound is 15.8 records for $\alpha = 0.1$ and 158.4 for $\alpha = 0.01$.* *(Fingerprinting, reductions)*

::: {.callout-tip collapse="true" title="Solution"}
1. **The padded dataset.** Given a small dataset $S$ of $n_0$ records, form $\tilde D = S \cup P$, where $P$ has $(n - n_0)/2$ copies of $(+1, \dots, +1)$ and as many of $(-1, \dots, -1)$. The padding sums to 0 in every coordinate, so the mean of $\tilde D$ is $\frac{n_0}{n}\bar x_S$.
2. **The new mechanism.** $M'(S) = \operatorname{clip}\big(\frac{n}{n_0}M(\tilde D)\big)$. When $M$ is within $\alpha$ of the mean of $\tilde D$ in every coordinate, $\frac n{n_0}M(\tilde D)$ is within $\frac n{n_0}\alpha$ of $\bar x_S$, and clipping to $[-1, 1]$ cannot move it further away. So $M'$ has error $\alpha n/n_0$ with the same failure probability $\beta$.
3. **The same privacy.** Replacing one record of $S$ replaces one record of $\tilde D$, and the padding does not depend on $S$. So $M'$ is $(\varepsilon, \delta)$-DP whenever $M$ is; rescaling and clipping are post-processing.
4. **Choose $n_0$.** Take $n_0 = \lceil\alpha n/\alpha_0\rceil$, so that the error $\alpha n/n_0$ is at most $\alpha_0$. This needs $\alpha_0 \ge \alpha$, since padding can only add records.
5. **Apply @eq-lower-fp to $M'$** with $\alpha' = \alpha_0 + 2\beta$: $n_0 \ge (\frac23 - 2\alpha_0 - 4\beta)d/R$ with $R = (e^\varepsilon - 1)\sqrt d + 4\delta d$. Since $n_0 \le \alpha n/\alpha_0 + 1$, ignoring the $+1$ gives $n \ge \frac{\alpha_0}\alpha(\frac23 - 2\alpha_0 - 4\beta)\frac dR$.
6. **The best $\alpha_0$.** $\phi(\alpha_0) = \alpha_0(\frac23 - 4\beta - 2\alpha_0)$ is a downward parabola with $\phi'(\alpha_0) = \frac23 - 4\beta - 4\alpha_0 = 0$ at $\alpha_0 = \frac16 - \beta$. There $\frac23 - 4\beta - 2\alpha_0 = \frac13 - 2\beta = 2\alpha_0$, so $\phi = 2(\frac16 - \beta)^2 = (\frac23 - 4\beta)^2/8$. At $\beta = 0.05$: $\alpha_0 = 0.1167$ and $\phi = 0.0272$.
7. **The numbers.** At $d = 10^4$, $R = 1.7183\times100 + 0.04 = 171.87$. For $\alpha = 0.1$: $n \ge 0.0272\times10^4/(0.1\times171.87) = 15.8$. For $\alpha = 0.01$: ten times more, 158.4.

*Padding is a general trick: a lower bound for constant accuracy becomes one for accuracy $\alpha$ at a cost of a factor $1/\alpha$ in $n$. With it the fingerprinting bound has the shape $\sqrt d/(\alpha\varepsilon)$ of the Gaussian mechanism's upper bound.*
:::

## How many attributes does tracing need? {#sec-sol-lower-5}

> **Exercise 5.** **How many attributes does tracing need?** Using @eq-lower-tracing with $\sigma = 0$, find the number of attributes $d$ at which the tracing attack has advantage 0.5 for $n = 100$ and for $n = 200$. *Check: $d = 5678$ and $d = 22{,}274$; the ratio $d/n^2$ is about 0.56 in both cases.* *(Tracing, normal approximation)*

::: {.callout-tip collapse="true" title="Solution"}
1. **The advantage.** With $m = 2d/(3n)$ and $s = \sqrt{d(2/15 + 8/(15n))}$, the advantage is $2\Phi(m/(2s)) - 1$. Setting it to 0.5 gives $\Phi(m/(2s)) = 0.75$, so $m/(2s) = \Phi^{-1}(0.75) = 0.6745$.
2. **Solve for $d$.** $m/s = \frac{2\sqrt d}{3n\sqrt{2/15 + 8/(15n)}}$, so $\sqrt d = 1.349\times\frac{3n}2\sqrt{2/15 + 8/(15n)}$.
3. **$n = 100$:** $\sqrt{0.13333 + 0.00533} = 0.3724$, so $\sqrt d = 1.349\times150\times0.3724 = 75.35$ and $d = 5678$.
4. **$n = 200$:** $\sqrt{0.13333 + 0.00267} = 0.3688$, so $\sqrt d = 1.349\times300\times0.3688 = 149.2$ and $d = 22{,}274$.
5. **The ratio.** $d/n^2 = 0.5678$ and $0.5568$. Apart from the small $8/(15n)$ term, $d$ is proportional to $n^2$.

*Doubling the number of people quadruples the number of attributes the attacker needs. That is the $n \approx \sqrt d$ of the fingerprinting bound, read from the attacker's side.*
:::

## Tracing with a reference sample {#sec-sol-lower-6}

> **Exercise 6.** **Tracing with a reference sample.** The attacker does not know $p$ but has $m$ fresh records from the population, with mean $\hat p$. Show that the score $\langle\bar x, y - \hat p\rangle$ has the same mean gap $2d/(3n)$ and a variance larger by the factor $1 + 1/m$ (ignoring terms of order $1/(nm)$). *Check: at $n = 50$, $d = 2000$, $m = 50$, the predicted advantage is 0.563, and the simulation gives 0.563.* *(Tracing, robust traceability)*

::: {.callout-tip collapse="true" title="Solution"}
Work with one coordinate and drop the index $j$. Given $p$, the dataset, the reference sample and a fresh target are independent.

1. **Non-member mean.** $y$ and $\hat p$ both have mean $p$ and are independent of $\bar x$, so $\mathbb E[\bar x(y - \hat p)] = \mathbb E[\bar x]\cdot0 = 0$.
2. **Member mean.** For $y = x_i$: $\mathbb E[\bar x\,x_i] - \mathbb E[\bar x]\,\mathbb E[\hat p]$. The first term is $\frac1n\mathbb E[x_i^2] + \frac{n - 1}np^2 = \frac1n + \frac{n-1}np^2$; the second is $p^2$. The difference is $\frac{1 - p^2}n$, as with a known $p$. Averaging and summing over coordinates gives the same gap $2d/(3n)$.
3. **Non-member variance.** $y - \hat p$ has mean 0 and variance $(1 - p^2) + (1 - p^2)/m = (1 - p^2)(1 + \frac1m)$, and it is independent of $\bar x$. The coordinates are independent, so cross terms vanish, and $\operatorname{Var} = (1 + \frac1m)\sum_j\mathbb E\big[\bar x_j^2(1 - p_j^2)\big] = (1 + \frac1m)\,d\big(\frac2{15} + \frac8{15n}\big)$, the known-$p$ variance times $1 + \frac1m$. (The member's variance differs from this by terms of order $d/n$ and smaller, which the normal approximation ignores.)
4. **The numbers.** At $n = 50$, $d = 2000$: gap 26.67, standard deviation $16.97\times\sqrt{1.02} = 17.14$, so the predicted advantage is $2\Phi(26.67/(2\times17.14)) - 1 = 0.563$. The simulation over 200 datasets gives a member mean of 26.15, a non-member standard deviation of 17.11 and an advantage of 0.563.

*An estimated population costs the attacker almost nothing: with as many reference records as there are members, the advantage drops only from 0.568 (known $p$) to 0.563. This is the robustness that made genomic tracing practical, since public reference panels supply $\hat p$.*
:::

## Reconstruction implies $\varepsilon$ {#sec-sol-lower-7}

> **Exercise 7.** **Reconstruction implies $\varepsilon$.** A reconstruction attack recovers 90% of a database of uniform bits from a release claimed to be $(\varepsilon, 10^{-6})$-DP. What is the smallest $\varepsilon$ the claim can have? *Check: $\varepsilon \ge \ln\big((0.9 - 10^{-6})/0.1\big) = 2.1972$; at 99% it is 4.5951.* *(Fact 3)*

::: {.callout-tip collapse="true" title="Solution"}
1. **The bound on recovery.** By @eq-lower-dn, an $(\varepsilon, \delta)$-DP release lets any reconstruction recover a fraction of at most $(e^\varepsilon + \delta)/(e^\varepsilon + 1)$ of uniform bits, in expectation.
2. **Solve for $\varepsilon$.** $r(e^\varepsilon + 1) \le e^\varepsilon + \delta$ rearranges to $e^\varepsilon(1 - r) \ge r - \delta$, so $\varepsilon \ge \ln\frac{r - \delta}{1 - r}$.
3. **$r = 0.9$:** $\ln(0.899999/0.1) = \ln 9.0000 = 2.1972$.
4. **$r = 0.99$:** $\ln(0.989999/0.01) = \ln 99.0 = 4.5951$.
5. **Conversely,** at $\varepsilon = 1$ no attack can recover more than $(e + 10^{-6})/(e + 1) = 0.7311$ of the bits.

*Each extra "nine" in the recovery rate adds about $\ln 10 = 2.3$ to the $\varepsilon$ the attack proves. The $10^{-6}$ barely matters here; $\delta$ would need to be comparable to $1 - r$ to change the answer.*
:::
