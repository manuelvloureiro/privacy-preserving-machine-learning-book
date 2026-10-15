# Exercises: The definition of differential privacy

Exercises for [The definition of differential privacy](../01-definition.md), each with a worked solution. Try each exercise before opening its solution. The script `code/solutions_definition.py` computes every number quoted here; run it with `uv run python book/part03-dp-foundations/code/solutions_definition.py`.

The solutions use the chapter's notation. Differential privacy (DP) is defined by @eq-dpdef-def: $\Pr[M(D) \in S] \le e^\varepsilon\Pr[M(D') \in S] + \delta$ for all neighbours $D \simeq D'$ and all events $S$. $P = M(D)$ and $Q = M(D')$, and $L_{D,D'} = \ln(p/q)$ is the privacy loss. Randomized response at level $\varepsilon$ reports the truth with probability $t = e^\varepsilon/(1 + e^\varepsilon)$.

## Randomized response with $k$ answers {#sec-sol-dpdef-1}

> **Exercise 1.** **Randomized response with $k$ answers.** A person holds one of $k$ values. Report the true value with probability $e^\varepsilon/(e^\varepsilon + k - 1)$ and each other value with probability $1/(e^\varepsilon + k - 1)$. Show that this is $\varepsilon$-DP, and that $k = 2$ gives randomized response. *Check: at $k = 4$, $\varepsilon = 1$, the truth is reported with probability 0.4754.* *(Fact 1)*

::: {.callout-tip collapse="true" title="Solution"}
1. **It is a distribution.** For a true value $v$, the probabilities are one value $e^\varepsilon/(e^\varepsilon + k - 1)$ and $k - 1$ values $1/(e^\varepsilon + k - 1)$. They sum to $(e^\varepsilon + k - 1)/(e^\varepsilon + k - 1) = 1$.
2. **Neighbours.** The dataset is one person's value, and a neighbour changes it from $v$ to $v' \ne v$.
3. **Ratios outcome by outcome.** For the report $y$:
    - $y = v$: probability $e^\varepsilon/c$ under $v$, $1/c$ under $v'$ (it is an "other" value for $v'$), with $c = e^\varepsilon + k - 1$. Ratio $e^\varepsilon$.
    - $y = v'$: $1/c$ against $e^\varepsilon/c$. Ratio $e^{-\varepsilon}$.
    - Any other $y$: $1/c$ against $1/c$. Ratio 1.
4. **Fact 1.** The largest ratio over outcomes, in either order, is $e^\varepsilon$, so the largest ratio over events is too. The mechanism is $\varepsilon$-DP, and no smaller $\varepsilon$ works, because the outcome $y = v$ attains $e^\varepsilon$.
5. **$k = 2$.** The truth has probability $e^\varepsilon/(e^\varepsilon + 1) = t$ and the other value $1/(e^\varepsilon + 1) = 1 - t$: randomized response.

@tbl-sol-dpdef-krr gives the probabilities, and the smallest $\varepsilon$ found by computing the max-divergence between the output distributions of every pair of true values.

| $k$ | $\varepsilon$ | Truth | Each other value | Smallest $\varepsilon$ over all pairs |
| --- | --- | --- | --- | --- |
| 2 | $\ln 3$ | 0.7500 | 0.2500 | 1.0986 |
| 4 | 1 | 0.4754 | 0.1749 | 1.0000 |
| 10 | 1 | 0.2320 | 0.0853 | 1.0000 |

: Randomized response with $k$ answers. {#tbl-sol-dpdef-krr}

*The privacy is the same for every $k$, but the truth gets rarer as $k$ grows: at $k = 10$ only 23% of reports are true, which is why frequency estimation over many categories is expensive in the local model.*
:::

## From add/remove to replace with $\delta$ {#sec-sol-dpdef-2}

> **Exercise 2.** **From add/remove to replace with $\delta$.** Show that $(\varepsilon, \delta)$-DP under add/remove implies $(2\varepsilon, (1 + e^\varepsilon)\delta)$-DP under replace. *Hint: chain @eq-dpdef-def through $D \setminus \{x\}$ as in @eq-dpdef-replace and keep track of the $\delta$ terms.* *Check: $\varepsilon = 1$, $\delta = 10^{-6}$ gives $(2, 3.718\times10^{-6})$.* *(Neighbouring relations)*

::: {.callout-tip collapse="true" title="Solution"}
1. **The path.** $D$ has record $x$ where $D'$ has $x'$. Let $D_0 = D \setminus \{x\} = D' \setminus \{x'\}$. Then $D \simeq D_0$ and $D_0 \simeq D'$ under add/remove.
2. **First step.** For any event $S$, $\Pr[M(D) \in S] \le e^\varepsilon\Pr[M(D_0) \in S] + \delta$.
3. **Second step.** $\Pr[M(D_0) \in S] \le e^\varepsilon\Pr[M(D') \in S] + \delta$.
4. **Substitute 3 into 2.** $\Pr[M(D) \in S] \le e^\varepsilon\big(e^\varepsilon\Pr[M(D') \in S] + \delta\big) + \delta = e^{2\varepsilon}\Pr[M(D') \in S] + (1 + e^\varepsilon)\delta$. The $\delta$ of the second step is multiplied by $e^\varepsilon$ because it sits inside the first step's ratio.
5. **The other order** follows from the same chain read from $D'$ to $D$.
6. **Check.** $\varepsilon = 1$, $\delta = 10^{-6}$: $(2, (1 + e)\times10^{-6}) = (2, 3.718\times10^{-6})$.

**How loose is it?** For the mechanism "release each record with probability $\delta$" with $n = 3$ and $\delta = 0.1$, the script enumerates the outputs. Under add/remove the smallest $\delta$ at $\varepsilon = 0$ is 0.1000. Under replace (record $x$ swapped for a new $x'$) it is also 0.1000, while the conversion promises only $(1 + e^0)\times0.1 = 0.2000$. Replacing a record changes two coins, but the event "$x$ published" already captures the whole difference in one direction.

*The conversion is a worst case over mechanisms. For pure DP it is tight (the histogram of the chapter); for this mechanism its $\delta$ is twice the true one.*
:::

## The $\delta(\varepsilon)$ curve of randomized response {#sec-sol-dpdef-3}

> **Exercise 3.** **The $\delta(\varepsilon)$ curve of randomized response.** Show that randomized response at level $\varepsilon_0$ is $(\varepsilon, \delta)$-DP for $\varepsilon < \varepsilon_0$ exactly when $\delta \ge (e^{\varepsilon_0} - e^\varepsilon)/(1 + e^{\varepsilon_0})$. *Check: $\varepsilon_0 = \ln 3$, $\varepsilon = 0.5$ gives $\delta = 0.3378$.* *(Hockey-stick form)*

::: {.callout-tip collapse="true" title="Solution"}
1. **The hockey-stick form.** By @eq-dpdef-hockey, the smallest $\delta$ is $\max\{E_{e^\varepsilon}(P\|Q), E_{e^\varepsilon}(Q\|P)\}$ over neighbours. For randomized response there is one pair, bit 1 against bit 0, and it is symmetric (swapping the bits swaps the reports), so both orders give the same value.
2. **The two outcomes.** With $t = e^{\varepsilon_0}/(1 + e^{\varepsilon_0})$, $P = (\Pr[\text{report }1], \Pr[\text{report }0]) = (t, 1 - t)$ and $Q = (1 - t, t)$.
3. **The terms.** For report 1: $(t - e^\varepsilon(1 - t))_+$. For report 0: $(1 - t - e^\varepsilon t)_+ = 0$, since $1 - t < t \le e^\varepsilon t$.
4. **Simplify.** $t - e^\varepsilon(1 - t) = \frac{e^{\varepsilon_0} - e^\varepsilon}{1 + e^{\varepsilon_0}}$, positive exactly when $\varepsilon < \varepsilon_0$.
5. **Check.** $\varepsilon_0 = \ln 3$: $\delta(\varepsilon) = (3 - e^\varepsilon)/4$. At $\varepsilon = 0.5$, $(3 - 1.6487)/4 = 0.3378$; at $\varepsilon = 1$, 0.0704; at $\varepsilon = 0$, 0.5000, the total variation between the two report distributions. The script gets the same values from the hockey-stick sum.

*The curve falls linearly in $e^\varepsilon$ and hits zero at $\varepsilon_0$: a mechanism with two outputs and bounded loss trades $\varepsilon$ for $\delta$ along a straight line.*
:::

## Asking twice {#sec-sol-dpdef-4}

> **Exercise 4.** **Asking twice.** Alice answers the same question $k$ times, with fresh coins each time, using randomized response at $\varepsilon_0 = \ln 3$. Show that the vector of $k$ reports has largest privacy loss $k\varepsilon_0$, and find the posterior that Alice's answer is "yes" after $k$ "yes" reports, from a prior of $\frac12$. *Check: for $k = 3$ the posterior is $27/28 = 0.9643$.* *(Bayesian form)*

::: {.callout-tip collapse="true" title="Solution"}
1. **The output distribution factors.** Given Alice's bit $x$, the $k$ reports are independent, so $p(y_1, \dots, y_k \mid x) = \prod_i p(y_i \mid x)$.
2. **The privacy loss adds.** $L(y) = \sum_i\ln\frac{p(y_i \mid 1)}{p(y_i \mid 0)}$, and each term is $+\varepsilon_0$ (report 1) or $-\varepsilon_0$ (report 0).
3. **The largest loss.** All $k$ reports equal to 1 give $L = k\varepsilon_0$. Unlike the $n$-person case of the chapter, the $k$ factors do not cancel, because all of them depend on the same bit. The script computes the max-divergence of the joint distributions: 1.0986, 2.1972 and 3.2958 for $k = 1, 2, 3$, equal to $k\ln 3$.
4. **The posterior.** By Fact 4, the posterior odds are the prior odds times $e^{L} = 3^k$. From prior odds 1, they are $3^k$, so the posterior is $3^k/(3^k + 1)$: 0.75, 0.9 and $27/28 = 0.9643$ for $k = 1, 2, 3$.

*Re-randomizing the same answer spends the budget again: the losses add, which is basic composition in [Post-processing, group privacy and composition](../02-properties.md). Practical randomized response memoizes one noisy answer per person and reuses it.*
:::

## Choosing $\delta$ {#sec-sol-dpdef-5}

> **Exercise 5.** **Choosing $\delta$.** For the mechanism "release each record with probability $\delta$" on $n = 10^6$ records, find the largest $\delta$ for which the probability of publishing any record is at most 0.01. *Check: $\delta = 1.005\times10^{-8}$.* *(Why $\delta \ll 1/n$)*

::: {.callout-tip collapse="true" title="Solution"}
1. **The requirement.** $1 - (1 - \delta)^n \le 0.01$, that is, $(1 - \delta)^n \ge 0.99$.
2. **Solve.** Take logarithms: $n\ln(1 - \delta) \ge \ln 0.99$, so $\delta \le 1 - 0.99^{1/n}$.
3. **Evaluate.** With $n = 10^6$: $\delta \le 1 - e^{\ln(0.99)/10^6} = 1.0050\times10^{-8}$.
4. **Approximation.** For small values, $1 - (1 - \delta)^n \approx n\delta$, so $\delta \approx 0.01/n = 10^{-8}$. The exact value is 0.5% larger, because $-\ln 0.99 = 0.01005$ is slightly above 0.01.

*Asking for a 1% chance that anyone is exposed gives $\delta \approx 0.01/n$: about a hundred times smaller than $1/n$.*
:::

## The advantage bound at $\varepsilon = 0$ {#sec-sol-dpdef-6}

> **Exercise 6.** **The advantage bound at $\varepsilon = 0$.** Show that the bound @eq-dpdef-advantage at $\varepsilon = 0$ is $\delta$, and that the mechanism "release each record with probability $\delta$" attains it. *Check: with $n = 3$ and $\delta = 0.1$, the total variation between the outputs on neighbours is 0.1.* *(Advantage, testing form)*

::: {.callout-tip collapse="true" title="Solution"}
1. **The bound.** At $\varepsilon = 0$, $(e^0 - 1 + 2\delta)/(e^0 + 1) = 2\delta/2 = \delta$.
2. **An adversary for the mechanism.** In the game between $D$ and $D' = D \cup \{x\}$, guess "$D'$" exactly when $x$ appears in the output.
3. **Its errors.** Under $D$, $x$ can never appear: the false positive rate (FPR) is $\alpha = 0$. Under $D'$, it appears with probability $\delta$: the true positive rate (TPR) is $\delta$.
4. **Its advantage.** $\mathrm{TPR} - \mathrm{FPR} = \delta$, equal to the bound. So the bound is attained, and the guessing adversary is optimal.
5. **Check by enumeration.** The best advantage is the total variation between the two output distributions (Part 2). For $n = 3$ and $\delta = 0.1$, the script computes it over all subsets of the four records: 0.1000, equal to the bound 0.1000.

*Pure DP forbids a positive TPR at zero FPR; approximate DP with $\varepsilon = 0$ allows exactly $\delta$ of it, and this mechanism uses all of it.*
:::

## How far below the bound is the Laplace count? {#sec-sol-dpdef-7}

> **Exercise 7.** **How far below the bound is the Laplace count?** Show that $1 - e^{-\varepsilon/2}$ and $\tanh(\varepsilon/2)$ both equal $\varepsilon/2 + O(\varepsilon^2)$, and compute their ratio at $\varepsilon = 0.1$ and $\varepsilon = 1$. For large $\varepsilon$, compare $1 - \tanh(\varepsilon/2)$ with $e^{-\varepsilon/2}$. *Check: the ratios are 0.9762 and 0.8514.* *(Advantage bound)*

::: {.callout-tip collapse="true" title="Solution"}
1. **Laplace.** $e^{-u} = 1 - u + u^2/2 - \dots$, so $1 - e^{-\varepsilon/2} = \varepsilon/2 - \varepsilon^2/8 + O(\varepsilon^3)$.
2. **The bound.** $\tanh u = u - u^3/3 + \dots$ (odd function, so no $u^2$ term), so $\tanh(\varepsilon/2) = \varepsilon/2 - \varepsilon^3/24 + O(\varepsilon^5)$.
3. **First order.** Both are $\varepsilon/2 + O(\varepsilon^2)$. Their ratio is $1 - \varepsilon/4 + O(\varepsilon^2)$: the Laplace count falls below the bound at second order.
4. **Values.** @tbl-sol-dpdef-laplace. At $\varepsilon = 0.1$, $1 - 0.1/4 = 0.975$ predicts the ratio 0.9762 well.
5. **Large $\varepsilon$.** $1 - \tanh(u) = 2e^{-2u}/(1 + e^{-2u}) \approx 2e^{-\varepsilon}$ with $u = \varepsilon/2$, while $1 - (1 - e^{-\varepsilon/2}) = e^{-\varepsilon/2}$. The Laplace count's distance from a perfect attacker decays like $e^{-\varepsilon/2}$, the bound's like $2e^{-\varepsilon}$, much faster. At $\varepsilon = 10$, $e^{-5} = 6.738\times10^{-3}$ against $9.080\times10^{-5}$.

| $\varepsilon$ | Laplace count $1 - e^{-\varepsilon/2}$ | Bound $\tanh(\varepsilon/2)$ | Ratio | $1 - $ Laplace | $1 - $ bound |
| --- | --- | --- | --- | --- | --- |
| 0.1 | 0.048771 | 0.049958 | 0.9762 | 0.9512 | 0.9500 |
| 1 | 0.393469 | 0.462117 | 0.8514 | 0.6065 | 0.5379 |
| 5 | 0.917915 | 0.986614 | 0.9304 | $8.209\times10^{-2}$ | $1.339\times10^{-2}$ |
| 10 | 0.993262 | 0.999909 | 0.9934 | $6.738\times10^{-3}$ | $9.080\times10^{-5}$ |

: The best membership advantage against the Laplace count and the bound for $\varepsilon$-DP. {#tbl-sol-dpdef-laplace}

*The ratio is smallest at moderate $\varepsilon$. For small $\varepsilon$ both are $\varepsilon/2$, and for large $\varepsilon$ both approach 1. In between, the script finds the smallest ratio, 0.8284, at $\varepsilon = 1.76$: there the Laplace count leaves 17% of the bound unused.*
:::
