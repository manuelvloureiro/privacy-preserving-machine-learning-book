# Beyond Global Sensitivity

The Laplace mechanism adds noise scaled to the global sensitivity of a function, the largest change one record can cause on any pair of neighbouring datasets. For many functions that worst case is far from the dataset at hand, or it is infinite. This chapter asks whether the noise can be scaled to the dataset actually held instead, while keeping differential privacy (DP).

Done naively, it cannot, because the amount of noise then depends on the data and leaks it. Four techniques make the idea work. Propose-test-release checks privately that the dataset is far from any dataset where the function is sensitive. A stability argument built on it finds the most frequent element, and estimates every count of a histogram, with no dependence on the size of the domain. The last two techniques estimate a bound on the sensitivity privately, or smooth it, before using it. All four can be accurate on well-behaved datasets even when no algorithm is accurate on the worst one.

## Local sensitivity {#sec-bgs-local}

Throughout the chapter, $f\colon \mathcal X^n \to \mathbb R$ is a real-valued function of the dataset. Its **global sensitivity** is the largest change that replacing one record can cause, over neighbouring datasets $D \simeq D'$ (the replace-one relation),

$$
\Delta f = \max_{D \simeq D'} |f(D) - f(D')| ,
$$

and the [Laplace mechanism](../part02-dp-basics/02-laplace-mechanism.md) adds to $f(D)$ a noise term drawn from $\mathrm{Lap}(\Delta f/\varepsilon)$, which makes the release $\varepsilon$-DP. The maximum ranges over all pairs of neighbours, including datasets that look nothing like the one held. Fixing the first dataset gives a local version.

::: {#def-bgs-local name="Local sensitivity"}
The **local sensitivity** of $f$ at a dataset $D \in \mathcal X^n$ is

$$
\mathrm{LS}_f(D) = \max_{D' \sim D} |f(D) - f(D')| .
$$
:::

The global sensitivity is the largest local sensitivity, $\Delta f = \max_D \mathrm{LS}_f(D)$, and the local sensitivity can be much smaller. The natural mechanism releases $f(D) + \mathrm{Lap}(\mathrm{LS}_f(D)/\varepsilon)$. It fails, because the scale of the noise is itself a function of the data. The distance between the two closest records shows how the local sensitivity can jump from one dataset to a neighbour.

::: {#exm-bgs-closest}
Let $\mathcal X = \mathbb R$, $n = 3$, and let $f(D)$ be the distance between the two closest records. On the real line $f$ has infinite global sensitivity. For example, $(0, c, 2c)$ and its neighbour $(0, 0, 2c)$ have $f$ equal to $c$ and $0$, for every $c > 0$, so the Laplace mechanism does not apply. Take $D = (0, 0, 0)$. Replacing any one record leaves two records at $0$, so $f$ stays $0$ and $\mathrm{LS}_f(D) = 0$. Now take the neighbour $D' = (0, 0, 5)$. It also has $f(D') = 0$, but replacing one of its zeros by $10$ gives $(0, 10, 5)$ with closest distance $5$, so $\mathrm{LS}_f(D') \ge 5$.
:::

On these two neighbours, the naive mechanism adds no noise to one and some noise to the other, and that difference alone gives the dataset away.

::: {#prp-bgs-naive}
The mechanism $M(D) = f(D) + \mathrm{Lap}(\mathrm{LS}_f(D)/\varepsilon)$, where $\mathrm{Lap}(0)$ means no noise, is not $\varepsilon$-DP for any finite $\varepsilon$ when $f$ is the closest-pair distance of @exm-bgs-closest.
:::

::: {.proof}
1. On $D = (0,0,0)$ the local sensitivity is $0$, so $M(D) = 0$ with probability $1$.
2. On the neighbour $D' = (0,0,5)$ the local sensitivity is positive, so $M(D')$ has a continuous distribution and $\Pr[M(D') = 0] = 0$.
3. So the event $\{0\}$ has probability $1$ on $D$ and $0$ on $D'$, and no factor $e^{\varepsilon}$ with finite $\varepsilon$ can bound the first by the second.
:::

The failure is that $D$ has low local sensitivity while a neighbour has high local sensitivity, and the noise level tells them apart. The local sensitivity is itself a statistic of the data, and releasing it, even only through the scale of the noise, leaks information. The techniques below each account for the sensitivity of nearby datasets, not just of $D$.

## Propose-test-release {#sec-bgs-ptr}

The first technique proposes a bound $b$ on the local sensitivity, tests privately that the bound holds at $D$ and at every dataset near $D$, and releases $f(D)$ with noise scaled to $b$ only if the test passes. The test measures how many records must change before the bound fails. The **Hamming distance** $d(D, D')$ between two datasets is the number of records in which they differ.

::: {#def-bgs-distance name="Distance to instability"}
For a proposed bound $b \ge 0$ on the local sensitivity $\mathrm{LS}_f$, the **distance to instability** of $D$ is

$$
\gamma_b(D) = \min\{\, d(D, D') : D' \in \mathcal X^n,\ \mathrm{LS}_f(D') > b \,\},
$$

the fewest records that must change to reach a dataset whose local sensitivity exceeds $b$, where $d$ is the Hamming distance. If no dataset has local sensitivity above $b$, set $\gamma_b(D) = \infty$ for every $D$.
:::

So $\gamma_b(D) = 0$ exactly when $\mathrm{LS}_f(D) > b$. The distance is a count of records, and that gives it sensitivity $1$ whatever $f$ is.

::: {#lem-bgs-distance}
If $\gamma_b$ is finite, then $|\gamma_b(D) - \gamma_b(D')| \le 1$ for all neighbours $D \simeq D'$.
:::

::: {.proof}
1. If some dataset $W$ has $\mathrm{LS}_f(W) > b$, every dataset is within distance $n$ of $W$, so $\gamma_b$ is finite everywhere.
2. Let $W$ attain the minimum for $D'$, so $d(D', W) = \gamma_b(D')$ and $\mathrm{LS}_f(W) > b$. The Hamming distance satisfies the triangle inequality, so $d(D, W) \le d(D, D') + d(D', W) = 1 + \gamma_b(D')$. Hence $\gamma_b(D) \le \gamma_b(D') + 1$.
3. Swapping the roles of $D$ and $D'$ gives the other inequality.
:::

Because the distance has sensitivity $1$, the Laplace mechanism can release it privately, and that noisy distance is the test. The threshold of the test is set so that a dataset whose local sensitivity exceeds $b$ passes only with probability $\delta/2$.

::: {#def-bgs-ptr name="Propose-test-release"}
The mechanism takes a proposed bound $b > 0$ and privacy parameters $\varepsilon > 0$ and $\delta \in (0,1)$. On input $D$, it does three steps.

1. compute $\hat\gamma = \gamma_b(D) + Z_1$ with $Z_1 \sim \mathrm{Lap}(1/\varepsilon)$;
2. if $\hat\gamma \le \ln(1/\delta)/\varepsilon$, output $\bot$, meaning "no answer";
3. otherwise output $f(D) + Z_2$ with $Z_2 \sim \mathrm{Lap}(b/\varepsilon)$, independent of $Z_1$.
:::

The test spends one budget of $\varepsilon$, and the release spends another. The test can wrongly pass on a sensitive dataset, and $\delta$ pays for that. The release step is the naive mechanism of @prp-bgs-naive with the scale $b$ in place of $\mathrm{LS}_f(D)$. It is safe now because $b$ is fixed in advance, and the decision to use it depends on the data only through the private estimate $\hat\gamma$.

::: {#thm-bgs-ptr}
Propose-test-release is $(2\varepsilon, \delta)$-DP.
:::

::: {.proof}
Fix neighbours $D \simeq D'$ and an event $S \subseteq \mathbb R \cup \{\bot\}$. Write $\tau = \ln(1/\delta)/\varepsilon$ for the threshold. If $\gamma_b$ is infinite at every dataset, the test always passes and $b \ge \Delta f$, so the output is the Laplace mechanism and is $\varepsilon$-DP. Assume from now on that $\gamma_b$ is finite.

1. *The test is $\varepsilon$-DP.* By @lem-bgs-distance, $\gamma_b$ has sensitivity $1$, so $\hat\gamma$ is the Laplace mechanism and is $\varepsilon$-DP. The events $\{\hat\gamma \le \tau\}$ and $\{\hat\gamma > \tau\}$ are post-processing of $\hat\gamma$, so each has probability on $D$ at most $e^{\varepsilon}$ times its probability on $D'$.
2. *Case $\mathrm{LS}_f(D) > b$.* Then $\gamma_b(D) = 0$, and the test passes only if $Z_1 > \tau$, which has probability $\frac12 e^{-\varepsilon\tau} = \delta/2$. Splitting on whether the output is $\bot$ and using step 1,

$$
\Pr[M(D) \in S] \le \Pr[M(D) = \bot]\,\mathbf 1[\bot \in S] + \Pr[M(D) \ne \bot] \le e^{\varepsilon}\Pr[M(D') \in S] + \tfrac\delta2 .
$$

3. *Case $\mathrm{LS}_f(D) \le b$.* Then $|f(D) - f(D')| \le \mathrm{LS}_f(D) \le b$, so the release step is the Laplace mechanism for a pair at distance at most $b$, with scale $b/\varepsilon$. Its density at any $y$ changes by a factor at most $e^{\varepsilon}$ between $D$ and $D'$. Write $S_{\mathbb R} = S \cap \mathbb R$. The two steps use independent noise, so

$$
\Pr[M(D) \in S] = \Pr[\hat\gamma \le \tau]\,\mathbf 1[\bot \in S] + \Pr[\hat\gamma > \tau]\,\Pr[f(D) + Z_2 \in S_{\mathbb R}] .
$$

Each probability on the right is at most $e^{\varepsilon}$ times the same probability on $D'$, by step 1 and by the Laplace mechanism. So $\Pr[M(D) \in S] \le e^{2\varepsilon}\Pr[M(D') \in S]$.

4. In both cases $\Pr[M(D) \in S] \le e^{2\varepsilon}\Pr[M(D') \in S] + \delta$. The pair was arbitrary, so this holds for every ordered pair of neighbours.
:::

The test passes with high probability on datasets far from instability. If $\gamma_b(D) \ge 2\ln(1/\delta)/\varepsilon$, then the output is $\bot$ only if $Z_1 \le -\ln(1/\delta)/\varepsilon$, which has probability $\delta/2$. The price is computation. Evaluating $\mathrm{LS}_f$ at one dataset takes up to $n(|\mathcal X| - 1)$ evaluations of $f$, one per neighbour, and $\gamma_b$ is a minimum over all $|\mathcal X|^n$ datasets. So the distance to instability has to be worked out by hand for each function. The median is the standard case where this can be done efficiently.

## Stability: the mode and the histogram {#sec-bgs-stability}

Propose-test-release is most striking with the bound $b = 0$. Then the test checks that the answer does not change at all near $D$, and the release step adds no noise. The mode, the most frequent element, is the standard example. If all $n$ records equal the same value $v$, the mode stays $v$ until about $n/2$ records change, so the mode is far from instability.

Let $\mathcal X$ be any set with a fixed total order, used only to break ties. Write $c_x(D)$ for the number of records of $D$ equal to $x$, and $c_{(1)}(D) \ge c_{(2)}(D)$ for the largest and second-largest of these counts, with $c_{(2)}(D) = 0$ if all records are equal. The mode $\mathrm{mode}(D)$ is the element with the largest count, the least one in the order among ties. To change the mode, records must move from the most frequent element to the runner-up until it overtakes. So half the gap between the two largest counts is, up to $1$, the distance to instability, and it is computed in one pass over the data whatever the domain.

::: {#lem-bgs-gap}
Let $g(D) = \big(c_{(1)}(D) - c_{(2)}(D)\big)/2$. For all neighbours $D \simeq D'$:

1. $|g(D) - g(D')| \le 1$;
2. if $g(D) > 1$, then $\mathrm{mode}(D') = \mathrm{mode}(D)$.
:::

::: {.proof}
1. Replacing one record lowers one count by $1$ and raises another by $1$, so every count changes by at most $1$.
2. The $j$-th largest count is the maximum, over sets $S$ of $j$ elements, of $\min_{x \in S} c_x$. If every count rises by at most $1$, each such minimum rises by at most $1$, and so does the maximum. The same holds for decreases, so $c_{(1)}$ and $c_{(2)}$ each change by at most $1$, and $g$ changes by at most $(1 + 1)/2 = 1$. This is claim 1.
3. For claim 2, let $a = \mathrm{mode}(D)$. In $D'$, the count of $a$ is at least $c_{(1)}(D) - 1$, and every other count is at most $c_{(2)}(D) + 1$. The condition $g(D) > 1$ means $c_{(1)}(D) - 1 > c_{(2)}(D) + 1$, so $a$ has the strictly largest count in $D'$.
:::

The stable mode is propose-test-release with $b = 0$, with the gap $g$ in place of the distance to instability. Its threshold has an extra $1$ because claim 2 protects the mode only when $g(D) > 1$.

::: {#def-bgs-mode name="Stable mode"}
Given $\varepsilon > 0$ and $\delta \in (0,1)$, on input $D$, compute $\hat g = g(D) + Z$ with $Z \sim \mathrm{Lap}(1/\varepsilon)$, where $g(D)$ is half the gap between the two largest counts. If $\hat g \le 1 + \ln(1/\delta)/\varepsilon$, output $\bot$, and otherwise output $\mathrm{mode}(D)$ exactly.
:::

The guarantee has two parts. The test costs $\delta$ in privacy, and it fails, by outputting $\bot$, only on datasets whose two largest counts are close.

::: {#thm-bgs-mode}
The stable mode is $(\varepsilon, \delta)$-DP. On every dataset whose two largest counts differ by at least $2 + 4\ln(1/\delta)/\varepsilon$, it outputs the mode with probability at least $1 - \delta/2$.
:::

::: {.proof}
Fix neighbours $D \simeq D'$ and an event $S$, and write $\tau = 1 + \ln(1/\delta)/\varepsilon$.

1. *The gap is $\varepsilon$-DP.* By @lem-bgs-gap, claim 1, $\hat g$ is the Laplace mechanism for a function of sensitivity $1$, so it is $\varepsilon$-DP.
2. *Case $\mathrm{mode}(D') = \mathrm{mode}(D)$.* The output on both datasets is the same function of $\hat g$, namely $\bot$ below the threshold and the common mode above it. Post-processing gives $\Pr[M(D) \in S] \le e^{\varepsilon}\Pr[M(D') \in S]$.
3. *Case $\mathrm{mode}(D') \ne \mathrm{mode}(D)$.* By @lem-bgs-gap, claim 2, $g(D) \le 1$. The mode is released only if $Z > \tau - g(D) \ge \ln(1/\delta)/\varepsilon$, which has probability at most $\delta/2$. As in the proof of @thm-bgs-ptr, $\Pr[M(D) \in S] \le e^{\varepsilon}\Pr[M(D') \in S] + \delta/2$.
4. *Accuracy.* If $c_{(1)} - c_{(2)} \ge 2 + 4\ln(1/\delta)/\varepsilon$, then $g(D) \ge \tau + \ln(1/\delta)/\varepsilon$. The output is $\bot$ only if $Z \le -\ln(1/\delta)/\varepsilon$, which has probability $\delta/2$.
:::

With typical parameters, the gap the theorem needs is a few dozen records.

::: {#exm-bgs-mode}
With $\varepsilon = 1$ and $\delta = 10^{-6}$, the threshold is $1 + \ln(10^6) = 14.82$. A gap of $2 + 4 \times 13.82 = 57.3$, so $58$ records, between the most frequent element and the runner-up is enough. The domain plays no part. It may be the set of all strings, which is infinite.
:::

The stable mode costs $\varepsilon$, not $2\varepsilon$, because the release step adds no noise and spends no budget. Its accuracy depends on the gap, not on the domain. Under pure DP no such bound is possible. The Laplace histogram has error of order $\ln|\mathcal X|/\varepsilon$, which is useless for an infinite domain. The [packing chapter](04-packing-lower-bounds.md) shows that an $\varepsilon$-DP algorithm that finds the mode of a dataset over $k$ elements needs at least $\ln(k/2)/\varepsilon$ records, even when all records are equal. The cost $\ln k$ of a large domain is replaced here by $\ln(1/\delta)$.

The same idea estimates every count at once. The algorithm adds noise only to the counts of elements that appear in the data, and it reports only those whose noisy count clears a threshold. An element that appears once in $D$ and not at all in a neighbour is the only danger, and the threshold hides it except with probability $\delta/2$.

::: {#def-bgs-histogram name="Stability-based histogram"}
Given $\varepsilon > 0$ and $\delta \in (0,1)$, let $\tau = 1 + 2\ln(1/\delta)/\varepsilon$. On input $D$, for each element $x$ whose count $c_x(D)$ of records is positive, draw an independent $Z_x \sim \mathrm{Lap}(2/\varepsilon)$ and set $\hat c_x = c_x(D) + Z_x$. Output the list of pairs $(x, \hat c_x)$ with $\hat c_x > \tau$. An element not on the list has estimate $\hat c_x = 0$.
:::

The noise has scale $2/\varepsilon$ because replacing one record changes two counts. The threshold costs $\delta$ in privacy. In accuracy, it adds at most $\tau$ to the noise on each count, because a count is reported as $0$ only when its noisy value is at most $\tau$.

::: {#thm-bgs-histogram}
The stability-based histogram is $(\varepsilon, \delta)$-DP. For every $\beta \in (0,1)$, with probability at least $1 - \beta$, every element $x \in \mathcal X$ has

$$
|\hat c_x - c_x(D)| \le \tau + \frac{2}{\varepsilon}\ln\frac{n}{\beta} .
$$ {#eq-bgs-histogram}
:::

::: {.proof}
*Privacy.* Let $D'$ be $D$ with one record equal to $a$ replaced by $a' \ne a$. Let $C$ be the set of elements that appear in both datasets. Any element that appears only in $D$ is $a$, with count $1$, and any element that appears only in $D'$ is $a'$, with count $1$. Write $O(D) = O_C(D) \cup O_A(D)$ for the output list on $D$, where $O_C$ is the part of the list from $C$ and $O_A$ the part from elements only in $D$, which can only be $a$. Define $O_C(D')$ and $O_{A'}(D')$ in the same way, where $O_{A'}$ can only contain $a'$. Fix an event $S$ and let $p_C = \Pr[O_C(D') \in S]$.

1. *The extra element of $D$.* An element with count $1$ is listed only if its noise exceeds $\tau - 1 = 2\ln(1/\delta)/\varepsilon$, which has probability $\frac12 e^{-\ln(1/\delta)} = \delta/2$. When $O_A(D)$ is empty, $O(D) = O_C(D)$, so $\Pr[O(D) \in S] \le \Pr[O_C(D) \in S] + \delta/2$.
2. *The common elements.* On $C$, the counts of $D$ and $D'$ differ by at most $1$ at $a$ and at $a'$, so in $\ell_1$ by at most $2$. Laplace noise of scale $2/\varepsilon$ makes the noisy counts on $C$ $\varepsilon$-DP, and the list is post-processing of them. So $\Pr[O_C(D) \in S] \le \min(1, e^{\varepsilon} p_C)$.
3. *The extra element of $D'$.* The parts $O_C(D')$ and $O_{A'}(D')$ are independent, and $O_{A'}(D')$ is empty with probability at least $1 - \delta/2$ by the computation of step 1. When it is empty, $O(D') = O_C(D')$. So $\Pr[O(D') \in S] \ge (1 - \delta/2)\, p_C$.
4. *Combining.* If $e^{\varepsilon}p_C \le 1$, then $\min(1, e^\varepsilon p_C) - e^{\varepsilon}(1-\delta/2)p_C = e^{\varepsilon}p_C\,\delta/2 \le \delta/2$. Otherwise $1 - e^{\varepsilon}(1-\delta/2)p_C < 1 - (1 - \delta/2) = \delta/2$. In both cases, steps 1 to 3 give

$$
\Pr[O(D) \in S] \le e^{\varepsilon}(1 - \delta/2)\,p_C + \tfrac\delta2 + \tfrac\delta2 \le e^{\varepsilon}\Pr[O(D') \in S] + \delta .
$$

*Accuracy.*

5. At most $n$ elements appear in $D$. A Laplace variable of scale $2/\varepsilon$ exceeds $s$ in absolute value with probability $e^{-\varepsilon s/2}$, so by a union bound all of them satisfy $|Z_x| \le (2/\varepsilon)\ln(n/\beta)$ with probability at least $1 - \beta$.
6. On that event, take any element $x$. If it is listed, its error is $|Z_x|$. If it appears in $D$ but is not listed, then $c_x + Z_x \le \tau$, so its error is $c_x \le \tau + |Z_x|$. If it does not appear, both its count and its estimate are $0$. In every case the error is at most @eq-bgs-histogram.
:::

In the usual regime $\delta \le 1/n$, the bound @eq-bgs-histogram is at most $1 + (2/\varepsilon)\big(2\ln(1/\delta) + \ln(1/\beta)\big)$. Every count is then accurate to within a number of order $\ln(1/\delta)/\varepsilon$, with no dependence on the size of the domain. The Laplace histogram under pure DP pays $\ln|\mathcal X|$ instead, and the [packing chapter](04-packing-lower-bounds.md) shows that pure DP cannot avoid that cost.

## Privately bounding the local sensitivity {#sec-bgs-bounding}

Propose-test-release guesses a bound and checks it. A second technique estimates the local sensitivity itself, privately, and uses the estimate as the noise scale. It applies when the local sensitivity does not change fast between neighbours, that is, when $\mathrm{LS}_f$ has bounded global sensitivity.

::: {#def-bgs-bounding name="Noisy local sensitivity"}
Suppose that the local sensitivity changes by at most $\lambda > 0$ between neighbours, that is, $|\mathrm{LS}_f(D) - \mathrm{LS}_f(D')| \le \lambda$ for all $D \simeq D'$. Given $\varepsilon > 0$ and $\delta \in (0,1)$, on input $D$:

1. compute $\hat s = \mathrm{LS}_f(D) + Z_1 + \lambda\ln(1/\delta)/\varepsilon$ with $Z_1 \sim \mathrm{Lap}(\lambda/\varepsilon)$;
2. if $\hat s \le 0$, output $\bot$, and otherwise output $f(D) + Z_2$ with $Z_2 \sim \mathrm{Lap}(\hat s/\varepsilon)$.
:::

The shift $\lambda\ln(1/\delta)/\varepsilon$ makes the estimate $\hat s$ an overestimate of $\mathrm{LS}_f(D)$ except with probability $\delta/2$. On that good event, the noise is large enough for privacy. Paying both for the noisy estimate and for the release gives the factor $2\varepsilon$ in the next theorem.

::: {#thm-bgs-bounding}
Let $\lambda > 0$ satisfy $|\mathrm{LS}_f(D) - \mathrm{LS}_f(D')| \le \lambda$ for all neighbours $D \simeq D'$, and let $\varepsilon > 0$ and $\delta \in (0,1)$. Then noisy local sensitivity with these parameters is $(2\varepsilon, \delta)$-DP.
:::

::: {.proof}
Fix neighbours $D \simeq D'$. It suffices to prove the guarantee for the pair $(\hat s, \text{output})$, since the output alone is post-processing of it.

1. *The good event.* Let $G = \{\hat s \ge \mathrm{LS}_f(D)\}$, computed with the data $D$. It fails only if $Z_1 < -\lambda\ln(1/\delta)/\varepsilon$, which has probability $\delta/2$.
2. *Density of the scale.* The function $\mathrm{LS}_f$ has sensitivity $\lambda$, so the density of $\hat s$ on $D$ is at most $e^{\varepsilon}$ times its density on $D'$, at every value $s$.
3. *Density of the release.* Fix a value $s$ with $s \ge \mathrm{LS}_f(D)$ and $s > 0$. Then $|f(D) - f(D')| \le \mathrm{LS}_f(D) \le s$, so the Laplace density of scale $s/\varepsilon$ at any $y$ changes by a factor at most $e^{\varepsilon |f(D) - f(D')|/s} \le e^{\varepsilon}$ between $D$ and $D'$.
4. *Combining.* On values of $\hat s$ in $G$, the joint density of $(\hat s, \text{output})$ on $D$ is at most $e^{2\varepsilon}$ times that on $D'$, by steps 2 and 3. For an event $S$, split on $G$:

$$
\Pr[(\hat s, M(D)) \in S] \le e^{2\varepsilon}\Pr[(\hat s, M(D')) \in S] + \Pr[\text{not } G] \le e^{2\varepsilon}\Pr[(\hat s, M(D')) \in S] + \tfrac\delta2 .
$$
:::

Two quantities must be worked out by hand for each function: the local sensitivity at $D$, and a bound $\lambda$ on how fast it changes. The first takes up to $n(|\mathcal X| - 1)$ evaluations of $f$, one per neighbour, and can sometimes be computed even for an infinite domain. For some functions it is much faster, such as the mode, whose local sensitivity depends only on the two largest counts. The technique is simpler to run than propose-test-release, which needs the distance to instability, but it applies to fewer functions, because it needs the local sensitivity to change slowly everywhere.

## Smooth sensitivity {#sec-bgs-smooth}

The last technique also lets the local sensitivity grow away from $D$, but discounts each dataset by its distance. Take the closest-pair function of @exm-bgs-closest on the bounded domain $\mathcal X = [0, 10]$, so that its global sensitivity is finite. The dataset $D = (0,0,0)$ still has local sensitivity $0$, while its neighbour $(0,0,5)$ has local sensitivity at least $5$. A safe noise scale at $D$ must account for that neighbour, but less than for $D$ itself.

::: {#def-bgs-smooth name="Smooth sensitivity"}
For a smoothing parameter $s > 0$, the **smooth sensitivity** of $f$ at $D$ discounts the local sensitivity of every dataset $D'$ by a factor $e^{-s}$ for each record in which $D'$ differs from $D$:

$$
S_{f,s}(D) = \sup_{D' \in \mathcal X^n} \mathrm{LS}_f(D')\, e^{-s\, d(D, D')} ,
$$

where $d$ is the Hamming distance.
:::

The supremum ranges over all datasets, not only neighbours. Every dataset is within distance $n$ of every other, so $e^{-sn}\Delta f \le S_{f,s}(D) \le \Delta f$. The smooth sensitivity is therefore finite exactly when the global sensitivity is, and it can be smaller by a factor up to $e^{sn}$. Two properties follow from the definition, and they are all that privacy needs.

::: {#lem-bgs-smooth}
For every $D$, $S_{f,s}(D) \ge \mathrm{LS}_f(D)$. For all neighbours $D \simeq D'$, $S_{f,s}(D) \le e^{s} S_{f,s}(D')$.
:::

::: {.proof}
1. The term $D' = D$ in the supremum is $\mathrm{LS}_f(D)$, so the supremum is at least that.
2. For any dataset $W$, the triangle inequality gives $d(D, W) \ge d(D', W) - 1$, so $\mathrm{LS}_f(W)e^{-s\,d(D,W)} \le e^{s}\,\mathrm{LS}_f(W)e^{-s\,d(D',W)} \le e^{s} S_{f,s}(D')$. Taking the supremum over $W$ gives the second claim.
:::

Laplace noise scaled to the smooth sensitivity is not enough for pure DP. The scale moves by a factor up to $e^{s}$ between neighbours, and far in the tail a Laplace density changes by much more than that when its scale changes. Noise with polynomial tails changes little under rescaling, and it works. The standard Cauchy distribution is such a noise, with density at a point $z$ equal to

$$
h(z) = \frac{1}{\pi(1+z^2)} .
$$

::: {#thm-bgs-smooth name="Smooth sensitivity with Cauchy noise"}
Let $f$ be non-constant with finite global sensitivity $\Delta f$, let $\varepsilon > 0$, and write $S^* = S_{f, \varepsilon/2}$. The mechanism

$$
M(D) = f(D) + \frac{2\,S^*(D)}{\varepsilon}\, Z, \qquad Z \sim \text{standard Cauchy},
$$

is $\varepsilon$-DP.
:::

::: {.proof}
1. *The scale is positive and finite.* Some dataset $W_0$ has $\mathrm{LS}_f(W_0) > 0$ because $f$ is not constant, and every $D$ is within distance $n$ of it, so $S^*(D) \ge \mathrm{LS}_f(W_0)e^{-\varepsilon n/2} > 0$. Every term of the supremum is at most $\Delta f$, so $S^*(D) \le \Delta f < \infty$.
2. *Two facts about $h$.* The derivative of $\ln(1+z^2)$ is $2z/(1+z^2)$, which lies in $[-1, 1]$, so $h(z)/h(z + u) \le e^{|u|}$ for all $z$ and $u$. For every real $\kappa$ and $v$, $e^{-\kappa}h(v)/h(e^{\kappa}v) = e^{-\kappa}(1 + e^{2\kappa}v^2)/(1+v^2) \le e^{-\kappa}\max(1, e^{2\kappa}) = e^{|\kappa|}$.
3. *The ratio of densities.* Fix neighbours $D \simeq D'$ and write $a = \varepsilon/2$. The density of $M(D)$ at $y$ is $(a/S^*(D))\,h\big(a(y - f(D))/S^*(D)\big)$. Set $u = a(y - f(D))/S^*(D)$, $w = a(f(D) - f(D'))/S^*(D)$ and $e^{\kappa} = S^*(D)/S^*(D')$. Then $a(y - f(D'))/S^*(D') = e^{\kappa}(u + w)$, and the ratio of the densities on $D$ and $D'$ is

$$
e^{-\kappa}\,\frac{h(u)}{h(e^{\kappa}(u+w))} = \frac{h(u)}{h(u+w)} \cdot \frac{e^{-\kappa}\,h(u+w)}{h(e^{\kappa}(u+w))} \le e^{|w|}\,e^{|\kappa|} ,
$$

by the two facts of step 2.

4. *Bounding the exponents.* By @lem-bgs-smooth, $|f(D) - f(D')| \le \mathrm{LS}_f(D) \le S^*(D)$, so $|w| \le a = \varepsilon/2$. By the same lemma applied in both directions, $|\kappa| \le \varepsilon/2$. The densities therefore differ by a factor at most $e^{\varepsilon}$, and integrating over any event gives $\varepsilon$-DP.
:::

The Cauchy distribution has such heavy tails that it has no mean. Its tail is $\Pr[|Z| > z] = \frac2\pi\arctan(1/z) \le 2/(\pi z)$, so the error exceeds $4S^*(D)/(\pi\varepsilon\beta)$ with probability at most $\beta$, which grows as $1/\beta$ rather than as $\ln(1/\beta)$. Laplace or Gaussian noise scaled to the smooth sensitivity, with a smaller smoothing parameter, gives approximate DP with lighter tails. Laplace and Gaussian tails decay exponentially, which is where the logarithms in earlier error bounds come from. Computing $S_{f,s}$ means a maximum over all datasets, which is hard in general. For the median it takes polynomial time, and near-linear time with more care.

## Summary {#sec-bgs-summary}

| Result | Statement |
| --- | --- |
| @prp-bgs-naive | Laplace noise scaled to the local sensitivity is not private |
| @lem-bgs-distance | the distance to instability has sensitivity $1$ |
| @thm-bgs-ptr | propose-test-release is $(2\varepsilon, \delta)$-DP |
| @thm-bgs-mode | the stable mode is $(\varepsilon,\delta)$-DP and exact when the top two counts differ by $2 + 4\ln(1/\delta)/\varepsilon$ |
| @thm-bgs-histogram | the stability-based histogram is $(\varepsilon,\delta)$-DP with every count within $\tau + (2/\varepsilon)\ln(n/\beta)$ |
| @thm-bgs-bounding | a noisy overestimate of a Lipschitz local sensitivity gives $(2\varepsilon,\delta)$-DP |
| @lem-bgs-smooth | smooth sensitivity bounds the local sensitivity and changes by at most $e^{s}$ between neighbours |
| @thm-bgs-smooth | Cauchy noise of scale $2S_{f,\varepsilon/2}(D)/\varepsilon$ gives $\varepsilon$-DP when $0 < \Delta f < \infty$ |

: The chapter's results. {#tbl-bgs-summary}

## Notes {#sec-bgs-notes}

- Kobbi Nissim, Sofya Raskhodnikova and Adam Smith, "Smooth Sensitivity and Sampling in Private Data Analysis", 2007. Local and smooth sensitivity, the failure of naive local noise, and noise distributions for smooth sensitivity under pure and approximate DP.
- Cynthia Dwork and Jing Lei, "Differential Privacy and Robust Statistics", 2009. The propose-test-release framework.
- Abhradeep Thakurta and Adam Smith, "Differentially Private Feature Selection via Stability Arguments, and the Robustness of the Lasso", 2013. The distance-to-instability approach behind the stable mode.
- Aleksandra Korolova, Krishnaram Kenthapadi, Nina Mishra and Alexandros Ntoulas, "Releasing Search Queries and Clicks Privately", 2009. Thresholded noisy counts over an unbounded domain, the idea of the stability-based histogram.
- Cynthia Dwork and Aaron Roth, *The Algorithmic Foundations of Differential Privacy*, 2014. Chapter 7 covers propose-test-release and stability-based algorithms.
- Salil Vadhan, "The Complexity of Differential Privacy", 2017. Section 3 treats local sensitivity, propose-test-release, the stability-based histogram (Theorem 3.5) and privately bounding the local sensitivity, with an example on graphs.
