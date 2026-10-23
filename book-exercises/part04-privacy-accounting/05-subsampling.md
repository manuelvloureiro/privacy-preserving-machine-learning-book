# Exercises: Amplification by subsampling

Exercises for [Amplification by subsampling](../../book-full/part04-privacy-accounting/05-subsampling.md), each with a worked solution. Try each exercise before opening its solution. The script `code/part04-privacy-accounting/solutions_subsampling.py` computes every number quoted here; run it with `uv run python code/part04-privacy-accounting/solutions_subsampling.py`.

The solutions use the chapter's notation. $q$ is the sampling rate, never a density. $P_0 = \mathcal N(0, \sigma^2)$ and $P_1 = \mathcal N(1, \sigma^2)$ are the outputs of one Gaussian step without and with the record, and $\varepsilon(\alpha)$ is the Rényi differential privacy (RDP) curve of the subsampled step. Stochastic gradient descent (SGD) with per-example clipping and Gaussian noise gives differential privacy (DP), and is called DP-SGD.

## Subsampling twice {#sec-sol-subsampling-1}

> **Exercise 1.** **Subsampling twice.** Show that a Poisson sample with rate $q_2$ of a Poisson sample with rate $q_1$ is a Poisson sample with rate $q_1q_2$, and that applying @eq-sub-amplify twice gives the same $(\varepsilon', \delta')$ as applying it once with $q_1q_2$. *Check: $\varepsilon = 1$, $q_1 = q_2 = 0.1$ gives 0.1586 after one stage and 0.01704 after two.* *(Composition of the amplification map)*

::: {.callout-tip collapse="true" title="Solution"}
1. **The two-stage sample.** A record survives the first stage with probability $q_1$ and, given that, the second with probability $q_2$, using fresh independent coins. So it survives both with probability $q_1q_2$, independently of every other record. That is a Poisson sample with rate $q_1q_2$.
2. **Write the map in terms of $e^\varepsilon - 1$.** @eq-sub-amplify says $e^{\varepsilon'} - 1 = q(e^\varepsilon - 1)$. The quantity $e^\varepsilon - 1$ is simply multiplied by $q$.
3. **Apply it twice.** After the first stage $e^{\varepsilon_1} - 1 = q_1(e^\varepsilon - 1)$, and after the second $e^{\varepsilon_2} - 1 = q_2(e^{\varepsilon_1} - 1) = q_1q_2(e^\varepsilon - 1)$. This is the one-stage formula with rate $q_1q_2$. The $\delta$ is multiplied by $q_1$ and then by $q_2$, so it is also $q_1q_2\delta$.
4. **Check.** With $\varepsilon = 1$ and $q_1 = 0.1$, $\varepsilon_1 = \ln(1 + 0.1(e - 1)) = 0.1586$. With $q_2 = 0.1$ the second stage gives 0.01704, and the direct formula with $q = 0.01$ also gives 0.01704.

*The amplification map is consistent: in the coordinate $e^\varepsilon - 1$ it is multiplication by $q$, so stages of sampling compose by multiplying their rates. This is consistent with @eq-sub-amplify being exact, though consistency alone does not prove it; exactness comes from randomized response attaining the formula (@sec-sub-eps-delta).*
:::

## The binomial sum at α = 3 {#sec-sol-subsampling-2}

> **Exercise 2.** **The binomial sum at $\alpha = 3$.** Write out the four terms of @eq-sub-binomial for $\alpha = 3$, $q = 0.01$, $\sigma = 1$, and compute $\varepsilon(3)$. *Check: $\varepsilon(3) = 2.6464 \times 10^{-4}$.* Which term carries most of the privacy cost? *(Binomial expansion, Gaussian moments)*

::: {.callout-tip collapse="true" title="Solution"}
1. **The terms.** The Gaussian moments are $\mathbb E_{P_0}[(P_1/P_0)^j] = e^{(j^2 - j)/(2\sigma^2)}$, which is $1, 1, e^{1/\sigma^2}, e^{3/\sigma^2}$ for $j = 0, 1, 2, 3$. So
$$A_3 = (1 - q)^3 + 3q(1 - q)^2 + 3q^2(1 - q)e^{1/\sigma^2} + q^3e^{3/\sigma^2}.$$
2. **The numbers** are in @tbl-sol-subsampling-a3.

| $j$ | Coefficient | Moment | Term |
| --- | --- | --- | --- |
| 0 | $(0.99)^3$ | 1 | 0.97029900 |
| 1 | $3 \times 0.01 \times 0.99^2$ | 1 | 0.02940300 |
| 2 | $3 \times 0.01^2 \times 0.99$ | $e$ | 0.00080733 |
| 3 | $0.01^3$ | $e^3$ | 0.00002009 |

: The four terms of $A_3$ for $q = 0.01$, $\sigma = 1$. {#tbl-sol-subsampling-a3}

3. **Sum and convert.** $A_3 = 1.00052942$, and $\varepsilon(3) = \ln(A_3)/(3 - 1) = 2.6464 \times 10^{-4}$, the same value the chapter's script computes.
4. **Which term costs.** Without the moments, the four coefficients sum to exactly 1 (the binomial theorem for $((1 - q) + q)^3$). So the privacy cost is the excess $3q^2(1 - q)(e - 1) + q^3(e^3 - 1)$. The $j = 2$ part, $0.00080733 - 0.00029700 = 0.00051033$ (the term minus its coefficient), dominates the $j = 3$ part, $0.00002009 - 0.000001 = 0.00001909$. The $j = 2$ term carries a share of 0.96 of the cost.

*At small orders the whole privacy cost of a subsampled Gaussian step is the $j = 2$ term, the chance that the record is drawn, squared, times the $\chi^2$ divergence $e^{1/\sigma^2} - 1$ of the unsubsampled step.*
:::

## Replace-one under Poisson sampling, exactly {#sec-sol-subsampling-3}

> **Exercise 3.** **Replace-one under Poisson sampling, exactly.** Let $M$ be $\varepsilon$-DP ($\delta = 0$) for add/remove neighbours, and run it on a Poisson sample with rate $q$. Show that for replace-one neighbours the subsampled mechanism is $\varepsilon'$-DP with $\varepsilon'$ from @eq-sub-replace-pure, and that the Laplace mechanism on a sum of values in $[-1, 1]$ attains it. Compare with the group-privacy bound $\ln(1 + q(e^{2\varepsilon} - 1))$ for small and large $\varepsilon$. *Hint: the ratio of two mixtures that share the part $R$ is largest when one component ratio is as large and the other as small as DP allows.* *Check: at $q = 0.01$, $\varepsilon = 1$, 0.02338 against 0.06193, and 0.01704 for add/remove.* *(Likelihood-ratio bounds on mixtures)*

::: {.callout-tip collapse="true" title="Solution"}
1. **Condition on the rest of the sample.** Let $D$ hold $x$ and $D'$ hold $x'$ in its place, and let $\tilde D$ be the Poisson sample of the other records. Given $\tilde D$, the outputs are $(1 - q)R + qP$ and $(1 - q)R + qP'$, with $R = M(\tilde D)$, $P = M(\tilde D \cup \{x\})$ and $P' = M(\tilde D \cup \{x'\})$.
2. **Bound the component ratios.** $\tilde D \cup \{x\}$ and $\tilde D$ are add/remove neighbours, and so are $\tilde D \cup \{x'\}$ and $\tilde D$. For pure DP the densities then satisfy $a = P(y)/R(y) \in [e^{-\varepsilon}, e^\varepsilon]$ and $b = P'(y)/R(y) \in [e^{-\varepsilon}, e^\varepsilon]$ at every output $y$.
3. **Bound the mixture ratio.** At each $y$, the ratio of the two mixtures is $\frac{(1 - q) + qa}{(1 - q) + qb}$. It increases in $a$ and decreases in $b$, so it is at most $\frac{1 - q + qe^\varepsilon}{1 - q + qe^{-\varepsilon}}$. Swapping $x$ and $x'$ gives the same bound for the other order.
4. **Average over $\tilde D$.** Each conditional density of the first run is at most $e^{\varepsilon'}$ times that of the second, with the same weights over $\tilde D$, so the same holds for the averages. Taking logarithms, $\varepsilon' = \ln(1 + q(e^\varepsilon - 1)) - \ln(1 - q(1 - e^{-\varepsilon}))$, which is @eq-sub-replace-pure.
5. **The Laplace mechanism attains it.** Let $M$ add Laplace noise of scale $1/\varepsilon$ to the sum. Values lie in $[-1, 1]$, so adding or removing one changes the sum by at most 1 and $M$ is $\varepsilon$-DP. Set every other record to 0, $x = 1$ and $x' = -1$. Then $R$, $P$, $P'$ are Laplace distributions centred at 0, 1 and $-1$ for every $\tilde D$. On the event $\{Y > 1\}$, $P(Y > 1) = \frac12$, $R(Y > 1) = \frac12e^{-\varepsilon}$ and $P'(Y > 1) = \frac12e^{-2\varepsilon}$. The two mixtures give this event probabilities in the ratio
$$\frac{(1 - q)e^{-\varepsilon} + q}{(1 - q)e^{-\varepsilon} + qe^{-2\varepsilon}} = \frac{1 - q + qe^\varepsilon}{1 - q + qe^{-\varepsilon}}$$
(multiply above and below by $e^\varepsilon$). An event with this ratio rules out any smaller $\varepsilon'$.
6. **Compare** (@tbl-sol-subsampling-replace). The script computes the three values and the log-ratio of the Laplace event, which equals the exact value in every row.

| $\varepsilon$ | Add/remove | Exact replace-one | Group-privacy bound | Exact / add/remove | Bound / add/remove |
| --- | --- | --- | --- | --- | --- |
| 0.1 | 0.00105 | 0.00200 | 0.00221 | 1.91 | 2.10 |
| 1 | 0.01704 | 0.02338 | 0.06193 | 1.37 | 3.64 |
| 5 | 0.90589 | 0.91587 | 5.39931 | 1.01 | 5.96 |

: Replace-one neighbours under Poisson sampling at $q = 0.01$, pure DP: the exact $\varepsilon'$ against the group-privacy bound and the add/remove value. {#tbl-sol-subsampling-replace}

For small $\varepsilon$ both are about twice the add/remove value. For large $\varepsilon$ the exact value is barely above it, while the bound charges the doubled base $2\varepsilon$ and lands near $2\varepsilon - \ln(1/q)$.

*Group privacy charges every part of the mixture the two-step price, but only the pair $(P, P')$ is two steps apart; each of them is one step from the shared part $R$. Replacing a record under Poisson sampling costs about twice as much as adding one when $\varepsilon$ is small, and almost nothing extra when $\varepsilon$ is large.*
:::

## Inverting the accountant {#sec-sol-subsampling-4}

> **Exercise 4.** **Inverting the accountant.** Find the smallest $\sigma$ for which DP-SGD with $q = 0.01$, 10,000 steps and $\delta = 10^{-5}$ has $\varepsilon \le 1$ under the RDP accountant with the improved conversion. *Hint: $\varepsilon$ decreases in $\sigma$; bisect.* *Check: $\sigma = 4.126$.* *(Monotonicity, bisection)*

::: {.callout-tip collapse="true" title="Solution"}
1. **Monotone in $\sigma$.** More noise is post-processing of less noise: $\mathcal N(0, \sigma_2^2)$ is $\mathcal N(0, \sigma_1^2)$ plus independent $\mathcal N(0, \sigma_2^2 - \sigma_1^2)$ for $\sigma_2 > \sigma_1$. By data processing every $D_\alpha$ can only fall, so each $\varepsilon(\alpha)$ and hence the converted $\varepsilon$ is non-increasing in $\sigma$.
2. **Bracket.** At $\sigma = 0.5$ the accountant reports $\varepsilon = 47.42$, and at $\sigma = 10$ it reports 0.377. So the answer lies in $[0.5, 10]$.
3. **Bisect.** Halve the interval 20 times, keeping the half where $\varepsilon \le 1$ starts. Each step computes the RDP curve on the grid of orders, multiplies by 10,000 and converts.
4. **Result** (@tbl-sol-subsampling-sigma). For $\varepsilon = 1$ the answer is $\sigma = 4.126$, and the accountant then reports $\varepsilon = 1.000$ at the order $\alpha = 18$.

| Target $\varepsilon$ | $\sigma$ | Reported $\varepsilon$ | Best order |
| --- | --- | --- | --- |
| 1 | 4.126 | 1.000 | 18 |
| 2 | 2.278 | 2.000 | 10.1 |
| 8 | 0.917 | 7.999 | 3.7 |

: The noise multiplier that reaches each target $\varepsilon$ at $\delta = 10^{-5}$, $q = 0.01$, 10,000 steps, under the RDP accountant with the improved conversion. {#tbl-sol-subsampling-sigma}

*Calibration runs the accountant backwards, and it inherits the accountant's slack: a tighter accountant reports a smaller $\varepsilon$ at each $\sigma$, so it allows a smaller $\sigma$ for the same target, and the model trains with less noise.*
:::

## Fixed-size batches, honestly {#sec-sol-subsampling-5}

> **Exercise 5.** **Fixed-size batches, honestly.** A team trains with a fresh random batch of exactly 1% of the data at every step, $\sigma = 1$, for 10,000 steps, and reports the Poisson value 6.71. Using @tbl-sub-matching, what is the honest RDP value, and what noise multiplier would make the honest value equal 6.71? *Check: 47.42; $\sigma = 2$.* *(Matching sampling scheme and neighbouring relation)*

::: {.callout-tip collapse="true" title="Solution"}
1. **Which row.** A fresh fixed-size batch at each step is the row "fixed size $m$ of $n$". Whether the team states add/remove or replace-one neighbours, the base mechanism needs a replace-one guarantee: a clipped sum moves by up to $2C$ when the record enters the batch and pushes another one out (Lebeda et al. 2024, Proposition 9).
2. **Rescale.** Noise $\sigma C$ against a shift of $2C$ is the same pair as noise $(\sigma/2)C$ against a shift of $C$. So the honest accounting is the Poisson-subsampled Gaussian with $q = 0.01$ and noise multiplier $\sigma/2 = 0.5$.
3. **Compute.** The RDP accountant gives $\varepsilon = 47.42$ at $\delta = 10^{-5}$, against the reported 6.71.
4. **Fix.** To make the honest value 6.71, the halved noise multiplier must be 1, so the team must train with $\sigma = 2$. The script confirms that fixed-size batches with $\sigma = 2$ give 6.71. With $\sigma = 2$ and Poisson sampling, the same accountant would give 2.35.

*Reporting the Poisson number for fixed-size batches is equivalent to claiming you added twice as much noise as you did. The fix is either to sample with Poisson, or to double $\sigma$, or to report the honest number.*
:::

## The first direction, exactly {#sec-sol-subsampling-6}

> **Exercise 6.** **The first direction, exactly.** Show that for any pair $(P_0, P_1)$ and $e^{\varepsilon'} = 1 + q(e^\varepsilon - 1)$, $E_{e^{\varepsilon'}}((1 - q)P_0 + qP_1\|P_0) = q\,E_{e^\varepsilon}(P_1\|P_0)$, and that the other direction is 0 whenever $e^{\varepsilon'} \ge 1/(1-q)$. Check it numerically for the subsampled Gaussian with $q = 0.01$, $\sigma = 1$. *Check: at $\varepsilon = 2$, both sides equal $2.09236 \times 10^{-4}$.* *(Hockey-stick divergence, the mixture lemma with equality)*

::: {.callout-tip collapse="true" title="Solution"}
1. **Expand the event-wise difference.** For any event $A$,
$$(1 - q)P_0(A) + qP_1(A) - \big(1 + q(e^\varepsilon - 1)\big)P_0(A) = qP_1(A) - qe^\varepsilon P_0(A) = q\big[P_1(A) - e^\varepsilon P_0(A)\big].$$
The $P_0(A)$ terms cancel except for $-qe^\varepsilon P_0(A)$.
2. **Maximize over $A$.** The difference is $q$ times the bracket for every event, so the two maxima agree: $E_{e^{\varepsilon'}}(\text{mixture}\|P_0) = q\max_A[P_1(A) - e^\varepsilon P_0(A)] = q\,E_{e^\varepsilon}(P_1\|P_0)$. This is the mixture lemma's step 1 with $P' = R$, where no inequality was needed.
3. **The other direction.** The mixture is at least $(1 - q)P_0$ everywhere, so $P_0/\text{mixture} \le 1/(1 - q)$. If $e^{\varepsilon'} \ge 1/(1-q)$, then $P_0 - e^{\varepsilon'}\cdot\text{mixture} \le 0$ everywhere and $E_{e^{\varepsilon'}}(P_0\|\text{mixture}) = 0$.
4. **Numerically** (@tbl-sol-subsampling-hockey). For the Gaussian, $E_{e^\varepsilon}(P_1\|P_0)$ is the exact $\delta(\varepsilon)$ of the Gaussian mechanism. The script integrates both hockey-stick divergences of the mixture on a fine grid.

| $\varepsilon$ | $\varepsilon'$ | $q\,\delta(\varepsilon)$ | $E_{e^{\varepsilon'}}(\text{mixture}\Vert P_0)$ | $E_{e^{\varepsilon'}}(P_0\Vert\text{mixture})$ |
| --- | --- | --- | --- | --- |
| 1 | 0.01704 | $1.26937 \times 10^{-3}$ | $1.26937 \times 10^{-3}$ | 0 |
| 2 | 0.06193 | $2.09236 \times 10^{-4}$ | $2.09236 \times 10^{-4}$ | 0 |
| 3 | 0.17467 | $1.53719 \times 10^{-5}$ | $1.53719 \times 10^{-5}$ | 0 |

: The hockey-stick divergences of the subsampled Gaussian at $e^{\varepsilon'} = 1 + q(e^\varepsilon - 1)$, $q = 0.01$, $\sigma = 1$. {#tbl-sol-subsampling-hockey}

The other direction is 0 in every row, because $1/(1 - q) = 1.0101$ while even the smallest $e^{\varepsilon'}$, at $\varepsilon = 1$, is $e^{0.01704} = 1.0172$.

*For the remove direction, with the mixture first (the name used in [Numerical accounting](../../book-full/part04-privacy-accounting/06-numerical-accounting.md)), amplification maps the whole privacy profile exactly: $\delta_{\text{sub}}(\varepsilon') = q\,\delta(\varepsilon)$. All the slack of per-step $(\varepsilon, \delta)$ accounting comes from composing these profiles through a single pair of numbers, not from the amplification step.*
:::
