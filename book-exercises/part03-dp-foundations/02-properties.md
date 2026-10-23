# Exercises: Post-processing, group privacy and composition

Exercises for [Post-processing, group privacy and composition](../../book-full/part03-dp-foundations/02-properties.md), each with a worked solution. Try each exercise before opening its solution. The script `code/part03-dp-foundations/solutions_properties.py` computes every number quoted here; run it with `uv run python code/part03-dp-foundations/solutions_properties.py`.

The solutions use the chapter's notation. $M$ is a mechanism, $D \simeq D'$ neighbouring datasets, $P = M(D)$ and $Q = M(D')$, and $L_{D,D'}(y) = \ln\frac{p(y)}{q(y)}$ the privacy loss. Differential privacy (DP) at level $(\varepsilon, \delta)$ means $\Pr_P[S] \le e^\varepsilon \Pr_Q[S] + \delta$ for every event $S$ and every ordered neighbouring pair. Randomized response at $\varepsilon_0 = \ln 3$ reports the true bit with probability $\frac34$. Logarithms are natural.

## Majority of three answers {#sec-sol-dpprop-1}

> **Exercise 1.** **Majority of three answers.** A person gives three randomized-response answers at $\varepsilon_0 = \ln 3$ each, and only the majority of the three is published. By composition and post-processing it is $3\ln 3$-DP. Find its exact $\varepsilon$. *Check: $\Pr[\text{majority} = 1 \mid \text{bit } 1] = \frac{54}{64}$, so $\varepsilon = \ln 5.4 = 1.6864$.* *(Post-processing, composition)*

::: {.callout-tip collapse="true" title="Solution"}
1. **The bound.** The three answers are $3\ln 3$-DP by basic composition, and the majority is a deterministic function of them, so it is $3\ln 3 = 3.2958$-DP by post-processing.
2. **Under bit 1** ($D$), each answer is 1 with probability $\frac34$, independently. The majority is 1 when at least two answers are 1:
$$\Pr[\text{maj} = 1 \mid 1] = \Big(\frac34\Big)^3 + 3\Big(\frac34\Big)^2\frac14 = \frac{27}{64} + \frac{27}{64} = \frac{54}{64} = 0.84375.$$
3. **Under bit 0** ($D'$), each answer is 1 with probability $\frac14$:
$$\Pr[\text{maj} = 1 \mid 0] = \Big(\frac14\Big)^3 + 3\Big(\frac14\Big)^2\frac34 = \frac{1}{64} + \frac{9}{64} = \frac{10}{64} = 0.15625.$$
4. **The privacy loss of each output.** For "majority 1" it is $\ln\frac{54}{10} = \ln 5.4$. For "majority 0" the probabilities swap ($\frac{10}{64}$ under bit 1, $\frac{54}{64}$ under bit 0), so the loss is $-\ln 5.4$. The exact $\varepsilon$ is $\ln 5.4 = 1.6864$.

*Majority voting merges transcripts with losses $3\ln 3$ (three 1s) and $\ln 3$ (two 1s), and the merged loss is in between. The worst transcript is not reported on its own, so the bound $3\ln 3$ is loose by almost a factor of 2.*
:::

## The geometric sum of group privacy {#sec-sol-dpprop-2}

> **Exercise 2.** **The geometric sum of group privacy.** Show that $\delta\sum_{j<k}e^{j\varepsilon} \le k e^{(k-1)\varepsilon}\delta$, that their ratio tends to 1 as $\varepsilon \to 0$, and that for the alarm mechanism every neighbouring pair is $(\varepsilon, \delta)$-DP as long as $a_0 \le 1 - e^{-\varepsilon}$. *Check: for $\varepsilon = 0.5$, $\delta = 10^{-5}$ and $k = 10$ the two forms are $2.272 \times 10^{-3}$ and $9.002 \times 10^{-3}$.* *(Group privacy)*

::: {.callout-tip collapse="true" title="Solution"}
**The inequality.**

1. The sum has $k$ terms $e^{0}, e^{\varepsilon}, \dots, e^{(k-1)\varepsilon}$. Since $\varepsilon \ge 0$, each is at most the largest, $e^{(k-1)\varepsilon}$.
2. So the sum is at most $k e^{(k-1)\varepsilon}$; multiply by $\delta$.

**The ratio.** As $\varepsilon \to 0$, every term $e^{j\varepsilon} \to 1$, so the sum tends to $k$, and $k e^{(k-1)\varepsilon} \to k$ too. The ratio of simple form to sum tends to 1. At $k = 10$ the script gives ratios 6.3215, 1.5054 and 1.0456 for $\varepsilon = 1, 0.1, 0.01$: the simple form is only harmless for small $\varepsilon$.

**The check.** With $\varepsilon = 0.5$, $\delta = 10^{-5}$, $k = 10$: $\delta\frac{e^{5} - 1}{e^{0.5} - 1} = 2.272 \times 10^{-3}$ and $10e^{4.5}\delta = 9.002 \times 10^{-3}$.

**The alarm mechanism.** It outputs "alarm" with probability $a_c$ on count $c$, where $a_c = e^\varepsilon a_{c+1} + \delta$ for $c < K$, $a_K = 0$ and $a_c = 0$ for $c > K$; so $a_0 \ge a_1 \ge \dots \ge a_K = 0$. For counts below 0, set $a_c = a_0$. A neighbouring pair has counts $c$ and $c + 1$. With two outputs, the events are $\emptyset$, $\{\text{alarm}\}$, $\{\text{quiet}\}$ and everything; the first and last satisfy the event form trivially. Check the other two in both orders.

1. **Alarm, $c$ against $c + 1$:** $a_c \le e^\varepsilon a_{c+1} + \delta$ holds with equality by construction (or trivially when both are 0, or both $a_0$).
2. **Alarm, $c + 1$ against $c$:** $a_{c+1} \le a_c \le e^\varepsilon a_c + \delta$.
3. **Quiet, $c$ against $c + 1$:** $1 - a_c \le 1 - a_{c+1} \le e^\varepsilon(1 - a_{c+1}) + \delta$, since $a_c \ge a_{c+1}$.
4. **Quiet, $c + 1$ against $c$:** we need $1 - a_{c+1} \le e^\varepsilon(1 - a_c) + \delta$. The left side is at most 1, so it suffices that $1 \le e^\varepsilon(1 - a_c)$, that is $a_c \le 1 - e^{-\varepsilon}$. Since $a_c \le a_0$, the condition $a_0 \le 1 - e^{-\varepsilon}$ is enough.

For $\varepsilon = 0.5$, $\delta = 10^{-5}$ and $K = 10$, $a_0 = 2.272 \times 10^{-3}$, far below $1 - e^{-0.5} = 0.3935$. The script computes the exact $\delta(0.5)$ of every neighbouring pair and finds the largest to be $1.000 \times 10^{-5}$, attained on "alarm".

*The condition only rules out absurd parameters: $a_0$ is the probability of an alarm, which must stay well below 1 for the mechanism to be useful at all.*
:::

## From records to users, then to releases {#sec-sol-dpprop-3}

> **Exercise 3.** **From records to users, then to releases.** Each user contributes at most 5 records, and a mechanism is $(0.2, 10^{-7})$-DP at the record level. Find the user-level guarantee, then the guarantee of 10 independent runs of the mechanism at the user level. *Check: $(1, 7.761 \times 10^{-7})$ per run and $(10, 7.761 \times 10^{-6})$ for ten runs.* *(Group privacy, composition)*

::: {.callout-tip collapse="true" title="Solution"}
1. **User-level neighbours are at record distance at most 5.** Removing one user removes at most 5 records, one at a time, so the two datasets are joined by a path of at most 5 record-level steps.
2. **Group privacy** with $k = 5$:
$$\varepsilon_5 = 5 \times 0.2 = 1, \qquad \delta_5 = 10^{-7}\,\frac{e^{1} - 1}{e^{0.2} - 1} = 10^{-7} \times \frac{1.71828}{0.22140} = 7.761 \times 10^{-7}.$$
(A user with fewer records is at a shorter distance, and the guarantee only gets better.)
3. **Composition at the user level.** Ten runs with fresh noise, each $(1, 7.761 \times 10^{-7})$-DP for user-level neighbours, are $(10, 7.761 \times 10^{-6})$-DP by @eq-dpprop-basic.

*The order matters. Composing first at the record level gives $(2, 10^{-6})$, and group privacy for 5 records then gives $\varepsilon = 10$ and $\delta = 10^{-6}\frac{e^{10} - 1}{e^{2} - 1} = 3.447 \times 10^{-3}$, 444 times worse. Apply group privacy to each run, then compose: group privacy inflates $\delta$ by powers of the per-step $\varepsilon$, so apply it where $\varepsilon$ is smallest.*
:::

## Parallel composition under replace {#sec-sol-dpprop-4}

> **Exercise 4.** **Parallel composition under replace.** For a histogram of $d$ bins with independent $\mathrm{Lap}(0, b)$ noise on each, show that the exact pure $\varepsilon$ is $1/b$ under add/remove and $2/b$ under replace (for a replacement that moves a record between bins). Then show that if records are assigned to bins by their position in the dataset, not their value, a replacement costs $1/b$. *Hint: under replace by value, find the output where both bins' losses take their largest value.* *(Parallel composition)*

::: {.callout-tip collapse="true" title="Solution"}
Write $c_i$ for the count in bin $i$ and $y_i = c_i + Z_i$ for the released noisy count. For two Laplace densities of scale $b$ with centres $c$ and $c'$, the privacy loss at $y$ is $\frac{|y - c'| - |y - c|}{b}$, which lies in $[-\frac{|c - c'|}{b}, \frac{|c - c'|}{b}]$ and reaches each end (Laplace section of [Divergences](../../book-full/part01-prerequisites/03-divergences.md)).

**Add/remove.**

1. Adding a record raises one count, say $c_1$, by 1. The other bins have the same distribution under $D$ and $D'$.
2. By Fact 3 of the chapter, $L(y) = \frac{|y_1 - c_1 - 1| - |y_1 - c_1|}{b} + 0$. Its largest absolute value is $1/b$, reached for every $y_1 \le c_1$.

**Replace across bins.**

1. Replacing a record in bin 1 by one in bin 2 changes $(c_1, c_2)$ under $D$ into $(c_1 - 1, c_2 + 1)$ under $D'$.
2. $L(y) = \frac{|y_1 - c_1 + 1| - |y_1 - c_1|}{b} + \frac{|y_2 - c_2 - 1| - |y_2 - c_2|}{b}$.
3. The first term equals $1/b$ for every $y_1 \ge c_1$, the second equals $1/b$ for every $y_2 \le c_2$. These regions can occur together, so the largest loss is $2/b$, and it is never larger than the sum of the two maxima. The exact pure $\varepsilon$ is $2/b$.

**Replace within a slot.** If bin $i$ holds a noisy version of record $i$'s value (one slot per record, values in $\{0, 1\}$), then replacing record $i$ changes slot $i$ only, by at most 1, and the analysis of add/remove applies: $\varepsilon = 1/b$.

The script computes the exact pure $\varepsilon$ of two bins with rounded Laplace noise: for $b = 1$, $1.0000$ under add/remove (and within a slot) and $2.0000$ across bins; for $b = 2$, $0.5000$ and $1.0000$.

*Which neighbouring relation the guarantee uses is part of the guarantee. A histogram with $\mathrm{Lap}(0, 1/\varepsilon)$ per bin is $\varepsilon$-DP under add/remove and $2\varepsilon$-DP under replace.*
:::

## Shared noise {#sec-sol-dpprop-5}

> **Exercise 5.** **Shared noise.** For the two counts $c_1(D) + Z$ and $c_2(D) + Z$ of @sec-dpprop-limits, show that $E_{e^\varepsilon}(P\|Q) = 1$ for every $\varepsilon$, where $P$ and $Q$ are the distributions of the pair under $D$ and $D'$. Then find the exact pure $\varepsilon$ of the pair when the noise is fresh. *Check: $\varepsilon = 1$ with fresh $\mathrm{Lap}(0, 1)$ noise.* *(Composition, hockey-stick)*

::: {.callout-tip collapse="true" title="Solution"}
**Shared noise.**

1. **Disjoint supports.** Under $D$ every released pair $(u, v)$ satisfies $u - v = c$, where $c = c_1(D) - c_2(D)$. Under $D'$ every pair satisfies $u - v = c - 1$. Let $A = \{(u, v) : u - v = c\}$. Then $\Pr_P[A] = 1$ and $\Pr_Q[A] = 0$.
2. **The hockey-stick is at least 1.** $E_{e^\varepsilon}(P\|Q) = \max_S[\Pr_P[S] - e^\varepsilon \Pr_Q[S]] \ge \Pr_P[A] - e^\varepsilon \Pr_Q[A] = 1$.
3. **And at most 1,** since $\sum(p - e^\varepsilon q)_+ \le \sum p = 1$. So it is exactly 1 for every $\varepsilon$, and the pair is not $(\varepsilon, \delta)$-DP for any $\delta < 1$.

The script represents the difference $u - v$ as a point mass at $c$ under $D$ and at $c - 1$ under $D'$ and gets $E_{e^\varepsilon} = 1.0000$ at $\varepsilon = 0, 1, 5, 20$.

**Fresh noise.** The pair is $(c_1(D) + Z_1, c_2(D) + Z_2)$ with independent $Z_1, Z_2$.

1. Removing Alice lowers $c_1$ by 1 and leaves $c_2$ unchanged, so the second coordinate has the same distribution under $D$ and $D'$.
2. By Fact 3 of the chapter, the privacy loss is the loss of the first coordinate, a $\mathrm{Lap}(0, 1)$ count against its shift by 1, whose largest absolute value is 1.
3. So the pair is exactly 1-DP; the script gets $1.0000$ with rounded Laplace noise.

*Each count alone is 1-DP in both cases. With shared noise, the pair is not DP at all; with fresh noise, it is 1-DP, and even 2-DP by basic composition if both counts depended on Alice.*
:::

## A preview of advanced composition {#sec-sol-dpprop-6}

> **Exercise 6.** **A preview of advanced composition.** Show that if $\Pr_{y \sim M(D)}[L_{D,D'}(y) > \varepsilon] \le \delta$ for every ordered neighbouring pair, then $M$ is $(\varepsilon, \delta)$-DP. For $k$ randomized-response answers at $\varepsilon_0$, use Hoeffding's inequality on $L$ to show that $M$ is $(\varepsilon, \delta)$-DP with $\varepsilon = k\varepsilon_0\tanh(\varepsilon_0/2) + \varepsilon_0\sqrt{2k\ln(1/\delta)}$. *Check: for $\varepsilon_0 = 0.1$, $k = 1000$, $\delta = 10^{-6}$ this gives $21.62$, against 100 for basic composition and the exact 19.345.* *(Composition, concentration)*

::: {.callout-tip collapse="true" title="Solution"}
**A tail bound on the loss gives $(\varepsilon, \delta)$.**

1. **The worst event** for $E_{e^\varepsilon}(P\|Q)$ is $B = \{y : p(y) > e^\varepsilon q(y)\} = \{L > \varepsilon\}$ (Fact 3 of [Divergences](../../book-full/part01-prerequisites/03-divergences.md)).
2. **Drop the negative term:** $E_{e^\varepsilon}(P\|Q) = \Pr_P[B] - e^\varepsilon \Pr_Q[B] \le \Pr_P[B] = \Pr_P[L > \varepsilon] \le \delta$.
3. This holds for every ordered pair, which is the hockey-stick form of $(\varepsilon, \delta)$-DP.

**The mean of the loss.** For one answer under $D$ (bit 1), the loss is $+\varepsilon_0$ with probability $t = \frac{e^{\varepsilon_0}}{1 + e^{\varepsilon_0}}$ and $-\varepsilon_0$ with probability $1 - t$. Its mean is $\varepsilon_0(2t - 1)$, and
$$2t - 1 = \frac{e^{\varepsilon_0} - 1}{e^{\varepsilon_0} + 1} = \tanh(\varepsilon_0/2).$$
For $k$ independent answers (Fact 3 of the chapter) the mean is $k\varepsilon_0\tanh(\varepsilon_0/2)$. At $\varepsilon_0 = 0.1$, $t = 0.5250$ and $2t - 1 = \tanh(0.05) = 0.04996$, so for $k = 1000$ the mean is 4.996.

**Hoeffding.**

1. $L = \sum_{i=1}^kL_i$ with independent $L_i \in [-\varepsilon_0, \varepsilon_0]$, each of range $2\varepsilon_0$. Hoeffding's inequality ([Concentration](../../book-full/part01-prerequisites/02-concentration.md)) gives, for $s > 0$,
$$\Pr[L - \mathbb E[L] \ge s] \le \exp\Big(-\frac{2s^2}{k(2\varepsilon_0)^2}\Big) = \exp\Big(-\frac{s^2}{2k\varepsilon_0^2}\Big).$$
2. **Set the right side to $\delta$:** $s = \varepsilon_0\sqrt{2k\ln(1/\delta)}$.
3. **Combine with the first part:** with $\varepsilon = \mathbb E[L] + s$, $\Pr[L > \varepsilon] \le \delta$. Randomized response is symmetric in the two bits, so the same holds for the reversed pair, and $M$ is $(\varepsilon, \delta)$-DP.

**The check.** For $\varepsilon_0 = 0.1$, $k = 1000$, $\delta = 10^{-6}$: $s = 0.1\sqrt{2000 \ln 10^6} = 16.623$, so $\varepsilon = 4.996 + 16.623 = 21.62$. The exact tail at that point is $\Pr_D[L > 21.62] = 5.319 \times 10^{-8}$, well below $10^{-6}$, which shows how much Hoeffding gives away. The exact $\varepsilon$ at $\delta = 10^{-6}$ is 19.345 and basic composition gives 100.

*The bound grows like $\sqrt k$, not $k$: the mean grows like $k\varepsilon_0^2/2$, small when $\varepsilon_0$ is small, and the fluctuations like $\varepsilon_0\sqrt k$. [Advanced composition](../../book-full/part03-dp-foundations/04-composition.md) proves the same shape for any $\varepsilon_0$-DP mechanisms.*
:::

## Ask until yes {#sec-sol-dpprop-7}

> **Exercise 7.** **Ask until yes.** A survey asks a person for randomized-response answers at $\varepsilon_0 = \ln 3$, one at a time, and stops at the first answer 1 or after 3 answers. Only the number of answers asked is published (1, 2, 3, or "3 and no 1"). Show it is $3\ln 3$-DP by adaptive composition, then compute its exact $\varepsilon$. *Check: the exact $\varepsilon$ is $\ln 27 = 3.2958$: the basic bound is attained.* *(Adaptive composition, post-processing)*

::: {.callout-tip collapse="true" title="Solution"}
**The bound.**

1. **Write it as an adaptive composition of three steps.** Step $i$ looks at the previous answers. If one of them is 1, it outputs "stopped", which does not depend on the data (0-DP). Otherwise it asks a fresh randomized-response question ($\ln 3$-DP). For every fixed previous output, step $i$ is at most $\ln 3$-DP.
2. **Adaptive composition** gives $3\ln 3$ for the transcript of three steps.
3. **The published count is a function of the transcript,** so post-processing keeps $3\ln 3$.

**The exact $\varepsilon$.** Let $s$ be the probability that one answer is 1: $s = \frac34$ under bit 1 and $\frac14$ under bit 0. Stopping after answer $i$ means $i - 1$ zeros and then a 1, with probability $(1 - s)^{i-1}s$; no 1 in three answers has probability $(1 - s)^3$. @tbl-sol-dpprop-ask lists the four outputs.

| Published | Under bit 1 | Under bit 0 | Ratio | Privacy loss |
| --- | --- | --- | --- | --- |
| 1 | $\frac34 = 0.750000$ | $\frac14 = 0.250000$ | 3 | $\ln 3$ |
| 2 | $\frac14 \cdot \frac34 = 0.187500$ | $\frac34 \cdot \frac14 = 0.187500$ | 1 | 0 |
| 3 | $\frac1{16} \cdot \frac34 = 0.046875$ | $\frac9{16} \cdot \frac14 = 0.140625$ | $\frac13$ | $-\ln 3$ |
| 3, no 1 | $\frac1{64} = 0.015625$ | $\frac{27}{64} = 0.421875$ | $\frac1{27}$ | $-3\ln 3$ |

: The distribution of the number of answers asked, under each value of the bit. {#tbl-sol-dpprop-ask}

The largest absolute loss is $\ln 27 = 3\ln 3 = 3.2958$, on the output "three answers and no 1". The basic bound is attained.

*Unlike the majority vote of Exercise 1, this processing keeps the worst transcript (three 0s) as its own output, so nothing is averaged away. Adaptivity does not cost more than the sum of the budgets, and here it does not cost less either.*
:::
