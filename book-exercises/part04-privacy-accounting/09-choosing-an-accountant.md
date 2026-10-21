# Exercises: Choosing an accountant

Exercises for [Choosing an accountant](../../book-full/part04-privacy-accounting/09-choosing-an-accountant.md), each with a worked solution. Try each exercise before opening its solution. The script `code/part04-privacy-accounting/solutions_choosing_an_accountant.py` computes every number quoted here; run it with `uv run python code/part04-privacy-accounting/solutions_choosing_an_accountant.py`.

The solutions use the chapter's notation. Stochastic gradient descent (SGD) with per-example clipping and Gaussian noise gives differential privacy (DP), and is called DP-SGD. $q$ is the Poisson sampling rate, $\sigma$ the noise multiplier, $k$ the number of steps, and neighbours are add/remove. Rényi differential privacy (RDP) means the RDP accountant of [Amplification by subsampling](../../book-full/part04-privacy-accounting/05-subsampling.md); the privacy loss distribution (PLD) accountant is that of [Numerical accounting](../../book-full/part04-privacy-accounting/06-numerical-accounting.md), on the grid $h = 10^{-5}$; the Gaussian differential privacy (GDP) central limit theorem (CLT) is that of [f-DP and Gaussian DP](../../book-full/part04-privacy-accounting/04-gaussian-dp.md). The running example is $q = 0.01$, $\sigma = 1$, $\delta = 10^{-5}$.

## Where RDP's slack comes from {#sec-sol-choose-1}

> **Exercise 1.** **Where RDP's slack comes from.** For $k = 1$, 100 and 10,000, split the excess of RDP (improved conversion) over the PLD upper bound into the conversion slack and the curve slack. Explain why the conversion slack dominates at $k = 1$ by evaluating $\ln(1/\delta)/(\alpha - 1)$ at the best order. *Check: at $k = 1$, conversion slack 0.4656 and curve slack 0.2905; $\ln(1/\delta)/(\alpha - 1) = 1.2936$ at $\alpha = 9.9$; at $k = 10{,}000$, 0.0000 and 0.4750.* *(Decomposing a bound)*

::: {.callout-tip collapse="true" title="Solution"}
1. **Three numbers.** The improved conversion is an upper bound on what the RDP curve certifies, the optimal conversion is exactly what it certifies, and the PLD upper bound is what the real pair certifies, up to the grid. Always
$$\varepsilon_{\text{improved}} - \varepsilon^\uparrow_{\text{PLD}} = \underbrace{(\varepsilon_{\text{improved}} - \varepsilon_{\text{optimal}})}_{\text{conversion slack}} + \underbrace{(\varepsilon_{\text{optimal}} - \varepsilon^\uparrow_{\text{PLD}})}_{\text{curve slack}},$$
The conversion slack is non-negative by @eq-choose-order. The curve slack is not covered by the theory: the pessimistic PLD may exceed the true $\varepsilon$ by up to $kh$, so in general the curve slack is only $\ge -kh$ (without truncation). On this example it is non-negative at every $k$, and the script of the chapter checks it.
2. **The numbers** are in @tbl-sol-choose-slack.

| $k$ | Best order | RDP, improved | RDP, optimal | PLD upper | Conversion slack | Curve slack |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 9.9 | 0.9555 | 0.4899 | 0.1995 | 0.4656 | 0.2905 |
| 100 | 8.8 | 1.2141 | 1.1538 | 0.7185 | 0.0603 | 0.4353 |
| 10,000 | 4.1 | 6.7127 | 6.7127 | 6.2377 | 0.0000 | 0.4750 |

: Splitting the excess of the RDP accountant over the PLD upper bound. {#tbl-sol-choose-slack}

3. **Why the conversion dominates at $k = 1$.** The improved conversion at order $\alpha$ is $k\,\varepsilon(\alpha) + \ln\frac{\alpha - 1}{\alpha} - \frac{\ln\delta + \ln\alpha}{\alpha - 1}$. Its last term contains $\ln(1/\delta)/(\alpha - 1)$, which does not depend on $k$ or on the mechanism. At the best order 9.9 it is $\ln(10^5)/8.9 = 1.2936$, far more than the true $\varepsilon$, which the PLD puts at 0.1995 (to four decimals). The curve term $\varepsilon(9.9)$ is tiny for one step. To shrink the fixed cost, the conversion would need a larger order, but at orders above about 10 the subsampled curve explodes (@sec-sub-small-q). The improved conversion's correction terms win back some of this, and the optimal conversion more, but every conversion from the curve must pay for not knowing the distribution beyond its moments.
4. **Why it vanishes at large $k$.** At $k = 10{,}000$ the best order is 4.1, and the curve $k\,\varepsilon(\alpha)$ is large. The optimal conversion then matches the improved one to four decimals, as chapter 30 found for the Gaussian at every $\mu$. What remains, 0.4750, is the curve slack: pairs other than the subsampled Gaussian have the same RDP curve and a larger $\delta$.

*For short runs, a better conversion would help; for long runs, only a better summary than the RDP curve helps, and the PLD is that summary.*
:::

## Steps under a budget {#sec-sol-choose-2}

> **Exercise 2.** **Steps under a budget.** With $q = 0.01$, $\sigma = 1$ and a budget of $\varepsilon = 3$ at $\delta = 10^{-5}$, find the largest number of steps each accountant allows: RDP with the improved conversion, the PLD upper bound, and the CLT. Is the CLT's answer safe? *Check: 2195, 2643 and 3009 steps; at 3009 steps the PLD lower bound is 3.1823.* *(Monotonicity in $k$, binary search)*

::: {.callout-tip collapse="true" title="Solution"}
1. **Monotonicity in $k$.** Running one more step is never more private: the $k$-step transcript is a post-processing (a projection) of the $(k + 1)$-step one. Every divergence, and so every bound in this part, is non-decreasing in $k$. The largest admissible $k$ is therefore well defined, and binary search over the integers finds it.
2. **RDP.** Binary search gives $k = 2195$: the accountant reports 3.0000 there (to four decimals, and at most 3) and 3.0006 at 2196 steps.
3. **PLD.** Binary search with the pessimistic PLD, recomposed at each trial $k$, gives $k = 2643$: 2.9995 there and 3.0001 at 2644.
4. **CLT.** The CLT has a closed form. Find the $\mu^*$ whose GDP profile gives $\varepsilon = 3$ at $\delta$, then $q^2k(e^{1/\sigma^2} - 1) \le \mu^{*2}$ gives $k = \lfloor(\mu^*/q)^2/(e^{1/\sigma^2} - 1)\rfloor = 3009$.
5. **Is 3009 safe?** No. At $k = 3009$ the optimistic PLD, a certified lower bound, is 3.1823. The run is not $(3, 10^{-5})$-DP.
6. **What the accountant buys.** In epochs, RDP allows 21.95 and the PLD 26.43. The tighter accountant buys 20% more training for the same certified budget.

*The inverse problems of the chapter come in two flavours, fixed steps and solve for $\sigma$, or fixed $\sigma$ and solve for $k$. Both are monotone, both are solved by bracketing, and in both the CLT's answer can overshoot the budget.*
:::

## Reporting at a smaller δ {#sec-sol-choose-3}

> **Exercise 3.** **Reporting at a smaller δ.** Show that a single $(\varepsilon, \delta)$ guarantee implies nothing at any smaller $\delta' < \delta$ beyond the trivial, by exhibiting a mechanism that is $(0, \delta)$-DP but not $(\varepsilon', \delta')$-DP for any $\varepsilon'$. Then compute the running example at $\delta = 10^{-6}$ and $10^{-8}$ from the PLD and the RDP curve. *Check: PLD upper 6.9574 and 8.2356; RDP 7.4140 and 8.6650.* *(Privacy profiles)*

::: {.callout-tip collapse="true" title="Solution"}
1. **The mechanism.** On a dataset with one record $x$, output $x$ with probability $\delta$ and the symbol $\bot$ otherwise. On the neighbouring dataset without $x$, always output $\bot$. With $P$ the first output distribution and $Q$ the second, $P(\{x\}) = \delta$ and $Q(\{x\}) = 0$.
2. **It is $(0, \delta)$-DP.** For any event $S$, $P(S) - Q(S) \le P(\{x\}) = \delta$, because the two distributions differ only by moving mass $\delta$ from $\bot$ to $x$. In the other direction, $Q(S) - P(S) \le \delta$ in the same way.
3. **It is not $(\varepsilon', \delta')$-DP for $\delta' < \delta$.** Take $S = \{x\}$: $P(S) = \delta > e^{\varepsilon'} \cdot 0 + \delta'$ for every $\varepsilon'$. Its privacy profile is $\delta(\varepsilon) = \delta$ for every $\varepsilon$, and $\varepsilon(\delta') = \infty$.
4. **Consequence.** From the pair $(\varepsilon, \delta)$ alone, a reader cannot tell this mechanism from one with a rapidly decaying profile. A report that gives only one pair cannot be re-read at a smaller $\delta$. That is why NIST SP 800-226 asks for the original parameters of RDP, zero-concentrated DP (zCDP) or GDP guarantees, and why the chapter recommends reporting the profile.
5. **The running example.** From the composed PLDs and the RDP curve, @tbl-sol-choose-profile.

| $\delta$ | PLD upper | RDP, improved |
| --- | --- | --- |
| $10^{-5}$ | 6.2377 | 6.7127 |
| $10^{-6}$ | 6.9574 | 7.4140 |
| $10^{-8}$ | 8.2356 | 8.6650 |

: The running example at $k = 10{,}000$ at three values of $\delta$. {#tbl-sol-choose-profile}

*For the subsampled Gaussian, $\varepsilon$ grows smoothly as $\delta$ falls, and both accountants recompute it from what they store. A single pair hides the profile, and the mechanism of step 1 shows that it has to.*
:::

## A noise schedule {#sec-sol-choose-4}

> **Exercise 4.** **A noise schedule.** A run uses 5,000 steps at $\sigma = 1$ and then 5,000 at $\sigma = 2$, with $q = 0.01$. Compute $\varepsilon$ at $\delta = 10^{-5}$ with RDP (adding the two curves) and with PLDs (convolving the two phases), and compare with adding the two phases' $\varepsilon$. *Check: RDP 4.9544; PLD $[4.4981, 4.5981]$; the phases alone are 4.2268 and 1.5024, which add to 5.7292.* *(Heterogeneous composition)*

::: {.callout-tip collapse="true" title="Solution"}
1. **RDP.** Curves add under composition (@eq-rdp-compose), whatever the steps are. The total curve is $5000\,\varepsilon_{\sigma=1}(\alpha) + 5000\,\varepsilon_{\sigma=2}(\alpha)$. The improved conversion of the sum, minimized over the grid, gives 4.9544.
2. **PLD.** The privacy loss of the run is the sum of the per-step losses, so its PLD is the convolution of the two phases' composed PLDs ([Numerical accounting](../../book-full/part04-privacy-accounting/06-numerical-accounting.md)). Convolve direction by direction: the remove-direction PLD of phase 1 with that of phase 2, and the same for add. Then solve for $\varepsilon$ and take the larger direction. Rounding up in both phases keeps the result an upper bound, and rounding down a lower bound. The bracket is $[4.4981, 4.5981]$, of width $kh = 10{,}000 \times 10^{-5} = 0.1$ as before.
3. **Adding the phases' $\varepsilon$.** The phases alone give 4.2268 and 1.5024 (PLD upper). Their sum, 5.7292, is worse than both accountants, and it is not even a guarantee at $\delta = 10^{-5}$. Basic composition of $(\varepsilon_1, \delta_1)$ and $(\varepsilon_2, \delta_2)$ gives $(\varepsilon_1 + \varepsilon_2, \delta_1 + \delta_2)$, so the sum is a guarantee only at $\delta = 2 \times 10^{-5}$.
4. **Comparison with a constant schedule.** Ten thousand steps at $\sigma = 1$ gave 6.2377. Doubling the noise for the second half brings the bound down to 4.5981. The second phase adds only $4.5981 - 4.2268 = 0.3713$ to the first phase's bound, much less than its own 1.5024, because composition adds privacy losses, not $\varepsilon$'s.

*Both RDP and the PLD handle heterogeneous steps without effort. The schedule must be fixed in advance; if the switch depended on the outputs, you would need a filter ([Adaptive composition](../../book-full/part04-privacy-accounting/08-adaptive-composition.md)).*
:::

## Whose number is it? {#sec-sol-choose-5}

> **Exercise 5.** **Whose number is it?** Four papers train the running example for 1,000 steps and report $\varepsilon = 1.62$, 1.83, 2.10 and 2.54 at $\delta = 10^{-5}$. Identify the accountant behind each, and say which reports are valid DP guarantees. *Check: the CLT gives 1.6177, the PLD 1.8332 (lower bound 1.8232), RDP 2.1014 (improved) and 2.5380 (Mironov).* *(Recomputing a reported guarantee)*

::: {.callout-tip collapse="true" title="Solution"}
1. **Recompute with every accountant.** At $k = 1000$: CLT 1.6177, PLD lower 1.8232, PLD upper 1.8332, RDP improved 2.1014, RDP Mironov 2.5380, and per-step advanced composition 1258.64, which no paper would report.
2. **Match.** 1.62 is the CLT, 1.83 the PLD upper bound, 2.10 RDP with the improved conversion, 2.54 RDP with Mironov's conversion.
3. **Validity.** The PLD lower bound 1.8232 is certified: the true $\varepsilon$ is at least that. So 1.62 is not a valid guarantee; the run is not $(1.62, 10^{-5})$-DP. The other three are upper bounds and valid. Note that 1.83 must be read as 1.8332 rounded *down*: a careful report would write 1.84.
4. **Ranking the valid ones.** All three describe the same run. The PLD's number is the most informative: it is within 0.0100 of the truth. Mironov's is 38% above it (the ratio 1.38 in @tbl-choose-ratio), purely because of the accountant.

*Two reports with different $\varepsilon$ can describe identical training. Before comparing two private models by their $\varepsilon$, recompute both with the same accountant.*
:::
