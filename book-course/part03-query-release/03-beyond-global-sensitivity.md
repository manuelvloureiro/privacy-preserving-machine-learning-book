# Beyond Global Sensitivity

The Laplace mechanism adds noise scaled to the global sensitivity of a function, the largest change one record can cause on any pair of neighbouring datasets. For many functions that worst case is far from the dataset at hand, or it is infinite. This chapter asks whether the noise can be scaled to the dataset actually held instead, while keeping differential privacy (DP).

Done naively, it cannot, because the amount of noise then depends on the data and leaks it. Four techniques make the idea work. Propose-test-release checks privately that the dataset is far from any dataset where the function is sensitive. A stability argument built on it finds the most frequent element, and estimates every count of a histogram, with no dependence on the size of the domain. The last two techniques estimate a bound on the sensitivity privately, or smooth it, before using it. All four can be accurate on well-behaved datasets even when no algorithm is accurate on the worst one.

## Local sensitivity {#sec-bgs-local}

Throughout the chapter, $f\colon \mathcal X^n \to \mathbb R$ is a real-valued function of the dataset. Its **global sensitivity** is the largest change that replacing one record can cause,

$$
\Delta_f = \max_{X \sim X'} |f(X) - f(X')| ,
$$

and the [Laplace mechanism](../part02-dp-basics/02-laplace-mechanism.md) adds to $f(X)$ a noise term drawn from $\mathrm{Lap}(\Delta_f/\varepsilon)$, which makes the release $\varepsilon$-DP. The maximum ranges over all pairs of neighbours, including datasets that look nothing like the one held. Fixing the first dataset gives a local version.

::: {#def-bgs-local name="Local sensitivity"}
The **local sensitivity** of $f$ at a dataset $X \in \mathcal X^n$ is

$$
\mathrm{LS}_f(X) = \max_{X' \sim X} |f(X) - f(X')| .
$$
:::

The global sensitivity is the largest local sensitivity, $\Delta_f = \max_X \mathrm{LS}_f(X)$, and the local sensitivity can be much smaller. The natural mechanism releases $f(X) + \mathrm{Lap}(\mathrm{LS}_f(X)/\varepsilon)$. It fails, because the scale of the noise is itself a function of the data.

::: {#exm-bgs-closest}
Let $\mathcal X = \mathbb R$, $n = 3$, and let $f(X)$ be the distance between the two closest records. On the real line $f$ has infinite global sensitivity. For example, $(0, c, 2c)$ and its neighbour $(0, 0, 2c)$ have $f$ equal to $c$ and $0$, for every $c > 0$, so the Laplace mechanism does not apply. Take $X = (0, 0, 0)$. Replacing any one record leaves two records at $0$, so $f$ stays $0$ and $\mathrm{LS}_f(X) = 0$. Now take the neighbour $X' = (0, 0, 5)$. It also has $f(X') = 0$, but replacing one of its zeros by $10$ gives $(0, 10, 5)$ with closest distance $5$, so $\mathrm{LS}_f(X') \ge 5$.
:::

::: {#prp-bgs-naive}
The mechanism $M(X) = f(X) + \mathrm{Lap}(\mathrm{LS}_f(X)/\varepsilon)$, where $\mathrm{Lap}(0)$ means no noise, is not $\varepsilon$-DP for any finite $\varepsilon$ when $f$ is the closest-pair distance of @exm-bgs-closest.
:::

::: {.proof}
1. On $X = (0,0,0)$ the local sensitivity is $0$, so $M(X) = 0$ with probability $1$.
2. On the neighbour $X' = (0,0,5)$ the local sensitivity is positive, so $M(X')$ has a continuous distribution and $\Pr[M(X') = 0] = 0$.
3. So the event $\{0\}$ has probability $1$ on $X$ and $0$ on $X'$, and no factor $e^{\varepsilon}$ with finite $\varepsilon$ can bound the first by the second.
:::

The failure is that $X$ has low local sensitivity while a neighbour has high local sensitivity, and the noise level tells them apart. The local sensitivity is itself a statistic of the data, and releasing it, even only through the scale of the noise, leaks information. The techniques below each account for the sensitivity of nearby datasets, not just of $X$.

## Propose-test-release {#sec-bgs-ptr}

The first technique proposes a bound $b$ on the local sensitivity, tests privately that the bound holds at $X$ and at every dataset near $X$, and releases $f(X)$ with noise scaled to $b$ only if the test passes. The test measures how many records must change before the bound fails. The **Hamming distance** $d(X, X')$ between two datasets is the number of entries in which they differ.

::: {#def-bgs-distance name="Distance to instability"}
For a proposed bound $b \ge 0$, the **distance to instability** of $X$ is

$$
\gamma_b(X) = \min\{\, d(X, X') : X' \in \mathcal X^n,\ \mathrm{LS}_f(X') > b \,\},
$$

the fewest records that must change to reach a dataset whose local sensitivity exceeds $b$. If no dataset has local sensitivity above $b$, set $\gamma_b \equiv \infty$.
:::

So $\gamma_b(X) = 0$ exactly when $\mathrm{LS}_f(X) > b$. The distance is a count of records, and that gives it sensitivity $1$ whatever $f$ is.

::: {#lem-bgs-distance}
If $\gamma_b$ is finite, then $|\gamma_b(X) - \gamma_b(X')| \le 1$ for all neighbours $X \sim X'$.
:::

::: {.proof}
1. If some dataset $Z$ has $\mathrm{LS}_f(Z) > b$, every dataset is within distance $n$ of $Z$, so $\gamma_b$ is finite everywhere.
2. Let $Z$ attain the minimum for $X'$, so $d(X', Z) = \gamma_b(X')$ and $\mathrm{LS}_f(Z) > b$. The Hamming distance satisfies the triangle inequality, so $d(X, Z) \le d(X, X') + d(X', Z) = 1 + \gamma_b(X')$. Hence $\gamma_b(X) \le \gamma_b(X') + 1$.
3. Swapping the roles of $X$ and $X'$ gives the other inequality.
:::

::: {#def-bgs-ptr name="Propose-test-release"}
The mechanism takes a proposed bound $b > 0$ and privacy parameters $\varepsilon > 0$ and $\delta \in (0,1)$. On input $X$, it does three steps.

1. compute $\hat\gamma = \gamma_b(X) + Z_1$ with $Z_1 \sim \mathrm{Lap}(1/\varepsilon)$;
2. if $\hat\gamma \le \ln(1/\delta)/\varepsilon$, output $\perp$, meaning "no answer";
3. otherwise output $f(X) + Z_2$ with $Z_2 \sim \mathrm{Lap}(b/\varepsilon)$, independent of $Z_1$.
:::

The test spends one budget of $\varepsilon$, and the release spends another. The test can wrongly pass on a sensitive dataset, and $\delta$ pays for that. The release step is the naive mechanism of @prp-bgs-naive with the scale $b$ in place of $\mathrm{LS}_f(X)$. It is safe now because $b$ is fixed in advance, and the decision to use it depends on the data only through the private estimate $\hat\gamma$.

::: {#thm-bgs-ptr}
Propose-test-release is $(2\varepsilon, \delta)$-DP.
:::

::: {.proof}
Fix neighbours $X \sim X'$ and an event $T \subseteq \mathbb R \cup \{\perp\}$. Write $\tau = \ln(1/\delta)/\varepsilon$ for the threshold. If $\gamma_b \equiv \infty$, the test always passes and $b \ge \Delta_f$, so the output is the Laplace mechanism and is $\varepsilon$-DP. Assume from now on that $\gamma_b$ is finite.

1. *The test is $\varepsilon$-DP.* By @lem-bgs-distance, $\gamma_b$ has sensitivity $1$, so $\hat\gamma$ is the Laplace mechanism and is $\varepsilon$-DP. The events $\{\hat\gamma \le \tau\}$ and $\{\hat\gamma > \tau\}$ are post-processing of $\hat\gamma$, so each has probability on $X$ at most $e^{\varepsilon}$ times its probability on $X'$.
2. *Case $\mathrm{LS}_f(X) > b$.* Then $\gamma_b(X) = 0$, and the test passes only if $Z_1 > \tau$, which has probability $\frac12 e^{-\varepsilon\tau} = \delta/2$. Splitting on whether the output is $\perp$ and using step 1,

$$
\Pr[M(X) \in T] \le \Pr[M(X) = \perp]\,\mathbf 1\{\perp \in T\} + \Pr[M(X) \ne \perp] \le e^{\varepsilon}\Pr[M(X') \in T] + \tfrac\delta2 .
$$

3. *Case $\mathrm{LS}_f(X) \le b$.* Then $|f(X) - f(X')| \le \mathrm{LS}_f(X) \le b$, so the release step is the Laplace mechanism for a pair at distance at most $b$, with scale $b/\varepsilon$. Its density at any $y$ changes by a factor at most $e^{\varepsilon}$ between $X$ and $X'$. Write $S = T \cap \mathbb R$. The two steps use independent noise, so

$$
\Pr[M(X) \in T] = \Pr[\hat\gamma \le \tau]\,\mathbf 1\{\perp \in T\} + \Pr[\hat\gamma > \tau]\,\Pr[f(X) + Z_2 \in S] .
$$

Each probability on the right is at most $e^{\varepsilon}$ times the same probability on $X'$, by step 1 and by the Laplace mechanism. So $\Pr[M(X) \in T] \le e^{2\varepsilon}\Pr[M(X') \in T]$.

4. In both cases $\Pr[M(X) \in T] \le e^{2\varepsilon}\Pr[M(X') \in T] + \delta$. The pair was arbitrary, so this holds for every ordered pair of neighbours.
:::

The test passes with high probability on datasets far from instability. If $\gamma_b(X) \ge 2\ln(1/\delta)/\varepsilon$, then the output is $\perp$ only if $Z_1 \le -\ln(1/\delta)/\varepsilon$, which has probability $\delta/2$. The price is computation. Evaluating $\mathrm{LS}_f$ at one dataset takes up to $n(|\mathcal X| - 1)$ evaluations of $f$, one per neighbour, and $\gamma_b$ is a minimum over all $|\mathcal X|^n$ datasets. So the distance to instability has to be worked out by hand for each function. The median is the standard case where this can be done efficiently.

## Stability: the mode and the histogram {#sec-bgs-stability}

Propose-test-release is most striking with the bound $b = 0$. Then the test checks that the answer does not change at all near $X$, and the release step adds no noise. The mode, the most frequent element, is the standard example. If all $n$ records equal the same value $v$, the mode stays $v$ until about $n/2$ records change, so the mode is far from instability.

Let $\mathcal X$ be any set with a fixed total order, used only to break ties. Write $c_x(X)$ for the number of records of $X$ equal to $x$, and $c_{(1)}(X) \ge c_{(2)}(X)$ for the largest and second-largest of these counts, with $c_{(2)}(X) = 0$ if all records are equal. The mode $\mathrm{mode}(X)$ is the element with the largest count, the least one in the order among ties. To change the mode, records must move from the most frequent element to the runner-up until it overtakes. So half the gap between the two largest counts is, up to $1$, the distance to instability, and it is computed in one pass over the data whatever the domain.

::: {#lem-bgs-gap}
Let $g(X) = \big(c_{(1)}(X) - c_{(2)}(X)\big)/2$. For all neighbours $X \sim X'$:

1. $|g(X) - g(X')| \le 1$;
2. if $g(X) > 1$, then $\mathrm{mode}(X') = \mathrm{mode}(X)$.
:::

::: {.proof}
1. Replacing one record lowers one count by $1$ and raises another by $1$, so every count changes by at most $1$.
2. The $j$-th largest count is the maximum, over sets $S$ of $j$ elements, of $\min_{x \in S} c_x$. If every count rises by at most $1$, each such minimum rises by at most $1$, and so does the maximum. The same holds for decreases, so $c_{(1)}$ and $c_{(2)}$ each change by at most $1$, and $g$ changes by at most $(1 + 1)/2 = 1$. This is claim 1.
3. For claim 2, let $a = \mathrm{mode}(X)$. In $X'$, the count of $a$ is at least $c_{(1)}(X) - 1$, and every other count is at most $c_{(2)}(X) + 1$. The condition $g(X) > 1$ means $c_{(1)}(X) - 1 > c_{(2)}(X) + 1$, so $a$ has the strictly largest count in $X'$.
:::

::: {#def-bgs-mode name="Stable mode"}
Given $\varepsilon > 0$ and $\delta \in (0,1)$, on input $X$, compute $\hat g = g(X) + Z$ with $Z \sim \mathrm{Lap}(1/\varepsilon)$. If $\hat g \le 1 + \ln(1/\delta)/\varepsilon$, output $\perp$, and otherwise output $\mathrm{mode}(X)$ exactly.
:::

::: {#thm-bgs-mode}
The stable mode is $(\varepsilon, \delta)$-DP. On every dataset whose two largest counts differ by at least $2 + 4\ln(1/\delta)/\varepsilon$, it outputs the mode with probability at least $1 - \delta/2$.
:::

::: {.proof}
Fix neighbours $X \sim X'$ and an event $T$, and write $\tau = 1 + \ln(1/\delta)/\varepsilon$.

1. *The gap is $\varepsilon$-DP.* By @lem-bgs-gap, claim 1, $\hat g$ is the Laplace mechanism for a function of sensitivity $1$, so it is $\varepsilon$-DP.
2. *Case $\mathrm{mode}(X') = \mathrm{mode}(X)$.* The output on both datasets is the same function of $\hat g$, namely $\perp$ below the threshold and the common mode above it. Post-processing gives $\Pr[M(X) \in T] \le e^{\varepsilon}\Pr[M(X') \in T]$.
3. *Case $\mathrm{mode}(X') \ne \mathrm{mode}(X)$.* By @lem-bgs-gap, claim 2, $g(X) \le 1$. The mode is released only if $Z > \tau - g(X) \ge \ln(1/\delta)/\varepsilon$, which has probability at most $\delta/2$. As in the proof of @thm-bgs-ptr, $\Pr[M(X) \in T] \le e^{\varepsilon}\Pr[M(X') \in T] + \delta/2$.
4. *Accuracy.* If $c_{(1)} - c_{(2)} \ge 2 + 4\ln(1/\delta)/\varepsilon$, then $g(X) \ge \tau + \ln(1/\delta)/\varepsilon$. The output is $\perp$ only if $Z \le -\ln(1/\delta)/\varepsilon$, which has probability $\delta/2$.
:::

::: {#exm-bgs-mode}
With $\varepsilon = 1$ and $\delta = 10^{-6}$, the threshold is $1 + \ln(10^6) = 14.82$. A gap of $2 + 4 \times 13.82 = 57.3$, so $58$ records, between the most frequent element and the runner-up is enough. The domain plays no part. It may be the set of all strings, which is infinite.
:::

The stable mode costs $\varepsilon$, not $2\varepsilon$, because the release step adds no noise and spends no budget. Its accuracy depends on the gap, not on the domain. Under pure DP no such bound is possible. The Laplace histogram has error of order $\ln|\mathcal X|/\varepsilon$, which is useless for an infinite domain. The [packing chapter](04-packing-lower-bounds.md) shows that an $\varepsilon$-DP algorithm that finds the mode of a dataset over $k$ elements needs at least $\ln(k/2)/\varepsilon$ records, even when all records are equal. The cost $\ln k$ of a large domain is replaced here by $\ln(1/\delta)$.

The same idea estimates every count at once. The algorithm adds noise only to the counts of elements that appear in the data, and it reports only those whose noisy count clears a threshold. An element that appears once in $X$ and not at all in a neighbour is the only danger, and the threshold hides it except with probability $\delta/2$.

::: {#def-bgs-histogram name="Stability-based histogram"}
Given $\varepsilon > 0$ and $\delta \in (0,1)$, let $\tau = 1 + 2\ln(1/\delta)/\varepsilon$. On input $X$, for each element $x$ with $c_x(X) > 0$, draw an independent $Z_x \sim \mathrm{Lap}(2/\varepsilon)$ and set $\hat c_x = c_x(X) + Z_x$. Output the list of pairs $(x, \hat c_x)$ with $\hat c_x > \tau$. An element not on the list has estimate $\hat c_x = 0$.
:::

::: {#thm-bgs-histogram}
The stability-based histogram is $(\varepsilon, \delta)$-DP. For every $\beta \in (0,1)$, with probability at least $1 - \beta$, every element $x \in \mathcal X$ has

$$
|\hat c_x - c_x(X)| \le \tau + \frac{2}{\varepsilon}\ln\frac{n}{\beta} .
$$ {#eq-bgs-histogram}
:::

::: {.proof}
*Privacy.* Let $X'$ be $X$ with one record equal to $a$ replaced by $b \ne a$. Let $C$ be the set of elements that appear in both datasets. Any element that appears only in $X$ is $a$, with count $1$, and any element that appears only in $X'$ is $b$, with count $1$. Write $O(X) = O_C(X) \cup O_A(X)$, where $O_C$ is the part of the list from $C$ and $O_A$ the part from elements only in $X$. Define $O_C(X')$ and $O_B(X')$ in the same way. Fix an event $T$ and let $q = \Pr[O_C(X') \in T]$.

1. *The extra element of $X$.* An element with count $1$ is listed only if its noise exceeds $\tau - 1 = 2\ln(1/\delta)/\varepsilon$, which has probability $\frac12 e^{-\ln(1/\delta)} = \delta/2$. When $O_A(X)$ is empty, $O(X) = O_C(X)$, so $\Pr[O(X) \in T] \le \Pr[O_C(X) \in T] + \delta/2$.
2. *The common elements.* On $C$, the counts of $X$ and $X'$ differ by at most $1$ at $a$ and at $b$, so in $\ell_1$ by at most $2$. Laplace noise of scale $2/\varepsilon$ makes the noisy counts on $C$ $\varepsilon$-DP, and the list is post-processing of them. So $\Pr[O_C(X) \in T] \le \min(1, e^{\varepsilon} q)$.
3. *The extra element of $X'$.* The parts $O_C(X')$ and $O_B(X')$ are independent, and $O_B(X')$ is empty with probability at least $1 - \delta/2$ by the computation of step 1. When it is empty, $O(X') = O_C(X')$. So $\Pr[O(X') \in T] \ge (1 - \delta/2)\, q$.
4. *Combining.* If $e^{\varepsilon}q \le 1$, then $\min(1, e^\varepsilon q) - e^{\varepsilon}(1-\delta/2)q = e^{\varepsilon}q\,\delta/2 \le \delta/2$. Otherwise $1 - e^{\varepsilon}(1-\delta/2)q < 1 - (1 - \delta/2) = \delta/2$. In both cases, steps 1 to 3 give

$$
\Pr[O(X) \in T] \le e^{\varepsilon}(1 - \delta/2)\,q + \tfrac\delta2 + \tfrac\delta2 \le e^{\varepsilon}\Pr[O(X') \in T] + \delta .
$$

*Accuracy.*

5. At most $n$ elements appear in $X$. A Laplace variable of scale $2/\varepsilon$ exceeds $s$ in absolute value with probability $e^{-\varepsilon s/2}$, so by a union bound all of them satisfy $|Z_x| \le (2/\varepsilon)\ln(n/\beta)$ with probability at least $1 - \beta$.
6. On that event, take any element $x$. If it is listed, its error is $|Z_x|$. If it appears in $X$ but is not listed, then $c_x + Z_x \le \tau$, so its error is $c_x \le \tau + |Z_x|$. If it does not appear, both its count and its estimate are $0$. In every case the error is at most @eq-bgs-histogram.
:::

In the usual regime $\delta \le 1/n$, the bound @eq-bgs-histogram is at most $1 + (2/\varepsilon)\big(2\ln(1/\delta) + \ln(1/\beta)\big)$. Every count is then accurate to within a number of order $\ln(1/\delta)/\varepsilon$, with no dependence on the size of the domain. The Laplace histogram under pure DP pays $\ln|\mathcal X|$ instead, and the [packing chapter](04-packing-lower-bounds.md) shows that pure DP cannot avoid that cost.

## Privately bounding the local sensitivity {#sec-bgs-bounding}

Propose-test-release guesses a bound and checks it. A second technique estimates the local sensitivity itself, privately, and uses the estimate as the noise scale. It applies when the local sensitivity does not change fast between neighbours, that is, when $\mathrm{LS}_f$ has bounded global sensitivity.

::: {#def-bgs-bounding name="Noisy local sensitivity"}
Suppose that $\lambda > 0$ and $|\mathrm{LS}_f(X) - \mathrm{LS}_f(X')| \le \lambda$ for all neighbours $X \sim X'$. Given $\varepsilon > 0$ and $\delta \in (0,1)$, on input $X$:

1. compute $\hat s = \mathrm{LS}_f(X) + Z_1 + \lambda\ln(1/\delta)/\varepsilon$ with $Z_1 \sim \mathrm{Lap}(\lambda/\varepsilon)$;
2. if $\hat s \le 0$, output $\perp$, and otherwise output $f(X) + Z_2$ with $Z_2 \sim \mathrm{Lap}(\hat s/\varepsilon)$.
:::

The shift $\lambda\ln(1/\delta)/\varepsilon$ makes the estimate $\hat s$ an overestimate of $\mathrm{LS}_f(X)$ except with probability $\delta/2$. On that good event, the noise is large enough for privacy.

::: {#thm-bgs-bounding}
Let $\lambda > 0$ satisfy $|\mathrm{LS}_f(X) - \mathrm{LS}_f(X')| \le \lambda$ for all neighbours $X \sim X'$, and let $\varepsilon > 0$ and $\delta \in (0,1)$. Then noisy local sensitivity with these parameters is $(2\varepsilon, \delta)$-DP.
:::

::: {.proof}
Fix neighbours $X \sim X'$. It suffices to prove the guarantee for the pair $(\hat s, \text{output})$, since the output alone is post-processing of it.

1. *The good event.* Let $G = \{\hat s \ge \mathrm{LS}_f(X)\}$, computed with the data $X$. It fails only if $Z_1 < -\lambda\ln(1/\delta)/\varepsilon$, which has probability $\delta/2$.
2. *Density of the scale.* The function $\mathrm{LS}_f$ has sensitivity $\lambda$, so the density of $\hat s$ on $X$ is at most $e^{\varepsilon}$ times its density on $X'$, at every value $s$.
3. *Density of the release.* Fix a value $s$ with $s \ge \mathrm{LS}_f(X)$ and $s > 0$. Then $|f(X) - f(X')| \le \mathrm{LS}_f(X) \le s$, so the Laplace density of scale $s/\varepsilon$ at any $y$ changes by a factor at most $e^{\varepsilon |f(X) - f(X')|/s} \le e^{\varepsilon}$ between $X$ and $X'$.
4. *Combining.* On values of $\hat s$ in $G$, the joint density of $(\hat s, \text{output})$ on $X$ is at most $e^{2\varepsilon}$ times that on $X'$, by steps 2 and 3. For an event $T$, split on $G$:

$$
\Pr[(\hat s, M(X)) \in T] \le e^{2\varepsilon}\Pr[(\hat s, M(X')) \in T] + \Pr[\text{not } G] \le e^{2\varepsilon}\Pr[(\hat s, M(X')) \in T] + \tfrac\delta2 .
$$
:::

Two quantities must be worked out by hand for each function: the local sensitivity at $X$, and a bound $\lambda$ on how fast it changes. The first takes up to $n(|\mathcal X| - 1)$ evaluations of $f$, one per neighbour, and can sometimes be computed even for an infinite domain. For some functions it is much faster, such as the mode, whose local sensitivity depends only on the two largest counts. The technique is simpler to run than propose-test-release, which needs the distance to instability, but it applies to fewer functions, because it needs the local sensitivity to change slowly everywhere.

## Smooth sensitivity {#sec-bgs-smooth}

The last technique also lets the local sensitivity grow away from $X$, but discounts each dataset by its distance. Take the closest-pair function of @exm-bgs-closest on the bounded domain $\mathcal X = [0, 10]$, so that its global sensitivity is finite. The dataset $X = (0,0,0)$ still has local sensitivity $0$, while its neighbour $(0,0,5)$ has local sensitivity at least $5$. A safe noise scale at $X$ must account for that neighbour, but less than for $X$ itself.

::: {#def-bgs-smooth name="Smooth sensitivity"}
For a parameter $s > 0$, the **smooth sensitivity** of $f$ at $X$ is

$$
S_{f,s}(X) = \sup_{X' \in \mathcal X^n} \mathrm{LS}_f(X')\, e^{-s\, d(X, X')} .
$$
:::

The supremum ranges over all datasets, not only neighbours. Every dataset is within distance $n$ of every other, so $e^{-sn}\Delta_f \le S_{f,s}(X) \le \Delta_f$. The smooth sensitivity is therefore finite exactly when the global sensitivity is, and it can be smaller by a factor up to $e^{sn}$. Two properties follow from the definition, and they are all that privacy needs.

::: {#lem-bgs-smooth}
For every $X$, $S_{f,s}(X) \ge \mathrm{LS}_f(X)$. For all neighbours $X \sim X'$, $S_{f,s}(X) \le e^{s} S_{f,s}(X')$.
:::

::: {.proof}
1. The term $X' = X$ in the supremum is $\mathrm{LS}_f(X)$, so the supremum is at least that.
2. For any dataset $Z$, the triangle inequality gives $d(X, Z) \ge d(X', Z) - 1$, so $\mathrm{LS}_f(Z)e^{-s\,d(X,Z)} \le e^{s}\,\mathrm{LS}_f(Z)e^{-s\,d(X',Z)} \le e^{s} S_{f,s}(X')$. Taking the supremum over $Z$ gives the second claim.
:::

Laplace noise scaled to the smooth sensitivity is not enough for pure DP. The scale moves by a factor up to $e^{s}$ between neighbours, and far in the tail a Laplace density changes by much more than that when its scale changes. Noise with polynomial tails changes little under rescaling, and it works. The standard Cauchy distribution is such a noise, with density at a point $z$ equal to

$$
h(z) = \frac{1}{\pi(1+z^2)} .
$$

::: {#thm-bgs-smooth name="Smooth sensitivity with Cauchy noise"}
Let $f$ be non-constant with finite global sensitivity $\Delta_f$, let $\varepsilon > 0$, and let $S = S_{f, \varepsilon/2}$. The mechanism

$$
M(X) = f(X) + \frac{2\,S(X)}{\varepsilon}\, Z, \qquad Z \sim \text{standard Cauchy},
$$

is $\varepsilon$-DP.
:::

::: {.proof}
1. *The scale is positive and finite.* Some dataset $Z_0$ has $\mathrm{LS}_f(Z_0) > 0$ because $f$ is not constant, and every $X$ is within distance $n$ of it, so $S(X) \ge \mathrm{LS}_f(Z_0)e^{-\varepsilon n/2} > 0$. Every term of the supremum is at most $\Delta_f$, so $S(X) \le \Delta_f < \infty$.
2. *Two facts about $h$.* The derivative of $\ln(1+z^2)$ is $2z/(1+z^2)$, which lies in $[-1, 1]$, so $h(z)/h(z + u) \le e^{|u|}$ for all $z$ and $u$. For every $\lambda$ and $v$, $e^{-\lambda}h(v)/h(e^{\lambda}v) = e^{-\lambda}(1 + e^{2\lambda}v^2)/(1+v^2) \le e^{-\lambda}\max(1, e^{2\lambda}) = e^{|\lambda|}$.
3. *The ratio of densities.* Fix neighbours $X \sim X'$ and write $a = \varepsilon/2$. The density of $M(X)$ at $y$ is $(a/S(X))\,h\big(a(y - f(X))/S(X)\big)$. Set $u = a(y - f(X))/S(X)$, $w = a(f(X) - f(X'))/S(X)$ and $e^{\lambda} = S(X)/S(X')$. Then $a(y - f(X'))/S(X') = e^{\lambda}(u + w)$, and the ratio of the densities on $X$ and $X'$ is

$$
e^{-\lambda}\,\frac{h(u)}{h(e^{\lambda}(u+w))} = \frac{h(u)}{h(u+w)} \cdot \frac{e^{-\lambda}\,h(u+w)}{h(e^{\lambda}(u+w))} \le e^{|w|}\,e^{|\lambda|} ,
$$

by the two facts of step 2.

4. *Bounding the exponents.* By @lem-bgs-smooth, $|f(X) - f(X')| \le \mathrm{LS}_f(X) \le S(X)$, so $|w| \le a = \varepsilon/2$. By the same lemma applied in both directions, $|\lambda| \le \varepsilon/2$. The densities therefore differ by a factor at most $e^{\varepsilon}$, and integrating over any event gives $\varepsilon$-DP.
:::

The Cauchy distribution has such heavy tails that it has no mean. Its tail is $\Pr[|Z| > z] = \frac2\pi\arctan(1/z) \le 2/(\pi z)$, so the error exceeds $4S(X)/(\pi\varepsilon\beta)$ with probability at most $\beta$, which grows as $1/\beta$ rather than as $\ln(1/\beta)$. Laplace or Gaussian noise scaled to the smooth sensitivity, with a smaller smoothing parameter, gives approximate DP with lighter tails. Laplace and Gaussian tails decay exponentially, which is where the logarithms in earlier error bounds come from. Computing $S_{f,s}$ means a maximum over all datasets, which is hard in general. For the median it takes polynomial time, and near-linear time with more care.

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
| @thm-bgs-smooth | Cauchy noise of scale $2S_{f,\varepsilon/2}(X)/\varepsilon$ gives $\varepsilon$-DP when $0 < \Delta_f < \infty$ |

: The chapter's results. {#tbl-bgs-summary}

## Notes {#sec-bgs-notes}

- Kobbi Nissim, Sofya Raskhodnikova and Adam Smith, "Smooth Sensitivity and Sampling in Private Data Analysis", 2007. Local and smooth sensitivity, the failure of naive local noise, and noise distributions for smooth sensitivity under pure and approximate DP.
- Cynthia Dwork and Jing Lei, "Differential Privacy and Robust Statistics", 2009. The propose-test-release framework.
- Abhradeep Thakurta and Adam Smith, "Differentially Private Feature Selection via Stability Arguments, and the Robustness of the Lasso", 2013. The distance-to-instability approach behind the stable mode.
- Aleksandra Korolova, Krishnaram Kenthapadi, Nina Mishra and Alexandros Ntoulas, "Releasing Search Queries and Clicks Privately", 2009. Thresholded noisy counts over an unbounded domain, the idea of the stability-based histogram.
- Cynthia Dwork and Aaron Roth, *The Algorithmic Foundations of Differential Privacy*, 2014. Chapter 7 covers propose-test-release and stability-based algorithms.
- Salil Vadhan, "The Complexity of Differential Privacy", 2017. Section 3 treats local sensitivity, propose-test-release, the stability-based histogram (Theorem 3.5) and privately bounding the local sensitivity, with an example on graphs.
