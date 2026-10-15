# Post-processing, group privacy and composition

## Why this matters

You have proved that one mechanism is differentially private. Now you use it. You round its output, plot it, or train a model on it. The dataset you protect has families in it, or users who each contributed fifty records. And you never run just one mechanism: an analyst asks a hundred queries, and a training run takes thousands of noisy gradient steps. Three questions come up at once:

- **After processing,** is the result still private, and do you need to prove anything new?
- **For a group** of $k$ records, such as one user's records or one household, what does a guarantee about one record give?
- **For many mechanisms** run on the same data, what is the guarantee of all their outputs together?

The answers are the three properties that make differential privacy (DP) usable as an engineering tool. **Post-processing** is free: nothing you compute from the output without touching the data can weaken the guarantee. **Group privacy** costs a factor $k$ in $\varepsilon$, and more than that in $\delta$. **Composition** adds: running mechanisms with budgets $(\varepsilon_i, \delta_i)$ costs at most $(\sum\varepsilon_i, \sum\delta_i)$. Without these three properties, every pipeline would need its own privacy proof. With them, you prove each building block once and add up the costs.

### The one idea: privacy losses add, processing only averages them

A mechanism $M$ is a randomized algorithm that takes a dataset and returns an output. Two datasets are **neighbouring**, written $D \simeq D'$, when they differ in one person's record. Differential privacy asks that $M$ behave almost the same on every neighbouring pair, so that its output reveals little about any one record. The way to measure "almost the same" at a single output $y$ is to ask how much more likely $y$ is when the data are $D$ than when they are $D'$. If $y$ is three times as likely under $D$, seeing $y$ is evidence for $D$; if it is equally likely under both, seeing it tells you nothing. The logarithm of that ratio is the **privacy loss** of $y$, from [The definition of differential privacy](01-definition.md). Writing $p_{M(D)}(y)$ for the probability (or density) that $M$ outputs $y$ on data $D$,
$$L_{D,D'}(y) = \ln\frac{p_{M(D)}(y)}{p_{M(D')}(y)}.$$
It is positive when $y$ favours $D$, negative when it favours $D'$, and zero when $y$ carries no evidence either way. Pure $\varepsilon$-DP says that no output carries much evidence: $|L_{D,D'}(y)| \le \varepsilon$ for every output and every neighbouring pair $D \simeq D'$. Equivalently, the likelihood ratio $e^{L}$ always lies in the band $[e^{-\varepsilon}, e^\varepsilon]$. All three properties of this chapter are statements about how $L$ behaves.

Post-processing replaces $y$ by $z = f(y)$. Several outputs $y$ can lead to the same $z$, and the probability of $z$ is the sum of their probabilities. So the likelihood ratio of $z$ is a weighted average of the likelihood ratios of those $y$, which is the content of the data processing inequality (DPI) of [Divergences](../part01-prerequisites/03-divergences.md): processing pools evidence, and pooling can only dilute it. An average of numbers that all lie in $[e^{-\varepsilon}, e^\varepsilon]$ stays in $[e^{-\varepsilon}, e^\varepsilon]$, so the processed output is just as private.

Group privacy compares $D$ with a dataset $D''$ that differs in $k$ records. You can walk from $D$ to $D''$ one record at a time, through $k$ neighbouring pairs. The likelihood ratio between the two ends is the product of the $k$ ratios along the way, so the privacy loss between the ends, its logarithm, is the sum of the $k$ step losses, each at most $\varepsilon$.

Composition releases $y = (y_1, \dots, y_k)$ from $k$ mechanisms that use independent coins. The probability of independent outcomes is the product of their probabilities, so the log-ratio of the whole release, its privacy loss, is the sum of the individual losses.

So the rule is: **along a path of datasets or across a sequence of mechanisms, privacy losses add; under processing, they only average.** @tbl-dpprop-overview lists the results and where each is used.

| Result | Where it is used |
| --- | --- |
| Post-processing | Publishing anything computed from a private output: rounded counts, plots, models trained on private statistics, synthetic data; every attack of [Part 2](../part02-attacks/01-threat-models.md) is a post-processing |
| Group privacy, pure and approximate | Turning a per-record guarantee into a per-user or per-household one; why correlated records weaken DP |
| Basic sequential composition, non-adaptive and adaptive | Budgeting an analysis of many queries; the starting point for every accountant in [Advanced and optimal composition](04-composition.md) and later |
| Parallel composition | Histograms, per-group statistics, local randomized response: disjoint data cost the maximum, not the sum |
| What composes and what doesn't | Fresh noise, matching neighbouring relations, not touching the data again |

: The results of this chapter and where each is used. {#tbl-dpprop-overview}

**How to read this chapter.** Each section follows the same arc: context, statement, derivation, interpretation, example, check yourself. Read the post-processing and composition sections first; they are used everywhere. The approximate-DP derivations (Fact 2 in the toolkit, the two-step lemma in the composition section) are short but abstract; walk through the concrete tables next to them. Basic composition is tight in the worst case but loose for most purposes. How much better you can do is the subject of [Advanced and optimal composition](04-composition.md); here you only see why there is room.

**Conventions.** $M$ is a randomized mechanism, $D \simeq D'$ neighbouring datasets, and $P = M(D)$, $Q = M(D')$ its two output distributions with mass functions or densities $p$ and $q$. Sums over outputs $\sum_y$ are integrals for continuous outputs. Logarithms are natural. The running mechanism is **randomized response** with $\varepsilon_0 = \ln 3$: a person reports their true bit with probability $\frac34$ and the other bit with probability $\frac14$. Comparing a person whose bit is 1 with one whose bit is 0, the report 1 has probability $\frac34$ against $\frac14$, a ratio of 3, and the report 0 has the ratio $\frac13$. So $L = \pm\ln 3$ on every output.

## Toolkit: three facts every derivation uses

### Fact 1 — Three forms of the definition

The definition of $(\varepsilon, \delta)$-DP can be written in three ways, and each derivation below uses whichever makes its step shortest. The first form talks about events: whatever set $S$ of outputs an observer cares about, its probability under $D$ is at most $e^\varepsilon$ times its probability under $D'$, plus a small slack $\delta$. The second form packs all events into one number, the **hockey-stick divergence**: scale $q$ up by the factor $e^\varepsilon$ and add up the probability mass of $p$ that still sticks out above it. Here $(x)_+ = \max(x, 0)$ keeps only the parts that stick out. The third form, for $\delta = 0$, is the bound on the privacy loss you already met. From [The definition of differential privacy](01-definition.md), these are equivalent for a pair $P = M(D)$, $Q = M(D')$:

1. **Events:** $P(S) \le e^\varepsilon Q(S) + \delta$ for every set of outputs $S$.
2. **Hockey-stick:** $E_{e^\varepsilon}(P\|Q) = \sum_y\big(p(y) - e^\varepsilon q(y)\big)_+ \le \delta$.
3. **Pure case ($\delta = 0$):** $|L_{D,D'}(y)| \le \varepsilon$ for every $y$ (on the common support).

$M$ is $(\varepsilon, \delta)$-DP when this holds for every ordered neighbouring pair.

Forms 1 and 2 are equivalent because the worst event is easy to find. The gap $P(S) - e^\varepsilon Q(S) = \sum_{y \in S}\big(p(y) - e^\varepsilon q(y)\big)$ is a sum of one term per output in $S$. It is largest when $S$ contains exactly the outputs whose term is positive, those with $p(y) > e^\varepsilon q(y)$, and its largest value is then the sum of the positive parts, which is $E_{e^\varepsilon}(P\|Q)$. This is Fact 3 of [Divergences](../part01-prerequisites/03-divergences.md). So "every event satisfies the bound" and "the worst event satisfies it" say the same thing. Form 3 follows from form 1 with $\delta = 0$ by taking single outputs, $S = \{y\}$: this gives $p(y) \le e^\varepsilon q(y)$, and the reversed pair gives $q(y) \le e^\varepsilon p(y)$, which together say $|L(y)| \le \varepsilon$. Conversely, summing $p(y) \le e^\varepsilon q(y)$ over the outputs $y \in S$ gives the event form back.

Since $\delta$ is the hockey-stick divergence at the scale $e^\varepsilon$, you can ask for every $\varepsilon$ at once how much slack is needed. The function $\varepsilon \mapsto \delta(\varepsilon) = \max\{E_{e^\varepsilon}(P\|Q), E_{e^\varepsilon}(Q\|P)\}$, maximized over neighbouring pairs, is the **privacy profile** of $M$. $M$ is $(\varepsilon, \delta)$-DP exactly when $\delta(\varepsilon) \le \delta$. The script computes it exactly for every mechanism in this chapter.

### Fact 2 — Tests instead of events

An event is an all-or-nothing rule: an output is in $S$ or it is not. Later steps of a pipeline are often softer than that. A randomized test rejects with some probability that depends on the output, and a later stage of an algorithm does something with a probability that depends on what it was given. Such a rule is a function $g$ from outputs to $[0, 1]$. It is a mixture of events, so it should obey the same bound, and it does. If $P(S) \le e^\varepsilon Q(S) + \delta$ for every event $S$, then for every function $g$ of the output with values in $[0, 1]$,
$$\mathbb E_P[g(Y)] \le e^\varepsilon\,\mathbb E_Q[g(Y)] + \delta.$$

The proof reuses the worst-event argument of Fact 1. The gap $\mathbb E_P[g] - e^\varepsilon\mathbb E_Q[g] = \sum_y g(y)\big(p(y) - e^\varepsilon q(y)\big)$ is a sum in which each value $g(y)$ multiplies a fixed bracket. To make it as large as possible, you set $g(y) = 1$ where the bracket is positive and $g(y) = 0$ where it is negative; any value in between only gives up part of a positive term or adds part of a negative one. That choice of $g$ is the indicator of the worst event, and the maximum is $E_{e^\varepsilon}(P\|Q) \le \delta$. This is the randomized version of Fact 3 in [Divergences](../part01-prerequisites/03-divergences.md).

A small case shows what the fact says. Take randomized response with $\varepsilon = \ln 3$, $\delta = 0$, and a later step that acts with probability $g(1) = 0.6$ on the report 1 and $g(0) = 0.2$ on the report 0. Under bit 1 the report is 1 with probability $\frac34$, so $\mathbb E[g] = \frac34(0.6) + \frac14(0.2) = 0.5$; under bit 0, $\frac14(0.6) + \frac34(0.2) = 0.3$. Indeed $0.5 \le 3 \times 0.3 = 0.9$. You will use this fact for post-processing, and in every approximate-DP proof where a later stage depends on an earlier output.

### Fact 3 — Privacy losses of independent outputs add

When two outputs come from independent coins, the probability of seeing both is the product of their probabilities. The logarithm turns that product into a sum, so the evidence carried by the pair is the evidence of the first plus the evidence of the second. In symbols, if $Y = (Y_1, Y_2)$ where $Y_1$ and $Y_2$ use independent coins, then $p(y_1, y_2) = p_1(y_1)p_2(y_2)$ and the same for $q$, so
$$L(y_1, y_2) = \ln\frac{p_1(y_1)p_2(y_2)}{q_1(y_1)q_2(y_2)} = L_1(y_1) + L_2(y_2).$$

Take two randomized-response answers from a person whose bit is 1 ($D$) or 0 ($D'$). The answers $(1, 1)$ have probability $\frac34 \cdot \frac34 = \frac9{16}$ under $D$ and $\frac14 \cdot \frac14 = \frac1{16}$ under $D'$, so $L = \ln 9 = \ln 3 + \ln 3$: two pieces of evidence pointing the same way. The answers $(1, 0)$ have $\frac34 \cdot \frac14 = \frac3{16}$ under both, so $L = 0 = \ln 3 - \ln 3$: the two pieces of evidence cancel.

The fact survives when the second mechanism depends on the first output (the **adaptive** case). The chain rule of probability writes any joint probability as "first, times second given first", $p(y_1, y_2) = p_1(y_1)\,p_2(y_2 \mid y_1)$, so the same sum holds with $p_2(y_2 \mid y_1)$ in place of $p_2(y_2)$. This is what composition and parallel composition use.

## Post-processing {#sec-dpprop-post}

### Context

A private output is rarely published as is. You round a noisy count to an integer, clip it at zero, threshold it, fit a model to a table of noisy marginals, or let an analyst look at it and decide what to do next. Each of these is a function of the output, possibly randomized with its own coins, that does not look at the data again. The question is whether such processing can undo the protection. It cannot: the data processing inequality of Part 1 says that processing averages likelihood ratios together, and averaging cannot push a ratio outside the band $[e^{-\varepsilon}, e^\varepsilon]$.

### Statement

To state the result for randomized processing too, describe the processing as a **channel** $K$: $K(z \mid y)$ is the probability that the processing outputs $z$ when its input is $y$. A deterministic $f$ is the case $K(z \mid y) = \mathbf 1[z = f(y)]$, which is 1 when $z = f(y)$ and 0 otherwise. The processed output on data $D$ then has the distribution $PK(z) = \sum_y p(y)K(z \mid y)$: sum over every $y$ the mechanism could give, weighted by how likely it is to become $z$. Write $L^f_{D,D'}(z)$ for the privacy loss of the processed output $z$.

The claim is that processing can never make the two worlds easier to tell apart. Let $M$ be $(\varepsilon, \delta)$-DP and let $f$ be any randomized function whose coins are independent of the data and of $M$'s coins. Then $f \circ M$, the mechanism $D \mapsto f(M(D))$, is $(\varepsilon, \delta)$-DP. Equivalently, for every channel $K$ and every neighbouring pair, the hockey-stick divergence and the largest privacy loss can only go down:
$$E_{e^\varepsilon}\big(PK\,\|\,QK\big) \le E_{e^\varepsilon}(P\|Q), \qquad \max_z\big|L^{f}_{D,D'}(z)\big| \le \max_y\big|L_{D,D'}(y)\big|,$$ {#eq-dpprop-post}
where $PK(z) = \sum_y p(y)K(z \mid y)$ is the distribution of $f(M(D))$. In words: the privacy profile of $f \circ M$ lies on or below that of $M$ at every $\varepsilon$.

### Derivation

The plan is to turn any event about the processed output into a soft test on the original output, then apply Fact 2. Recall that $K(z \mid y)$ is the probability that $f$ outputs $z$ on input $y$.

1. **Fix an event $S$ of processed outputs.** Let $g(y) = K(S \mid y) = \sum_{z \in S}K(z \mid y)$, the probability that processing $y$ lands in $S$. It is a probability, so it takes values in $[0, 1]$. It does not depend on the data, because $f$ does not look at the data: the same $g$ serves for $D$ and for $D'$.
2. **Average over $Y$.** To land in $S$, the mechanism first produces some $y$, and then processing sends it into $S$ with probability $g(y)$. Summing over $y$, $\Pr[f(M(D)) \in S] = \sum_y p(y)g(y) = \mathbb E_P[g(Y)]$, and the same with $q$ for $D'$.
3. **Apply Fact 2** to $g$: $\mathbb E_P[g] \le e^\varepsilon\mathbb E_Q[g] + \delta$, that is $\Pr[f(M(D)) \in S] \le e^\varepsilon\Pr[f(M(D')) \in S] + \delta$. This holds for every $S$ and every ordered neighbouring pair, which is the event form of $(\varepsilon, \delta)$-DP for $f \circ M$ (Fact 1). The hockey-stick inequality of @eq-dpprop-post is the same statement for the worst event $S$.
4. **The privacy-loss form.** This is step 1 of the DPI derivation in [Divergences](../part01-prerequisites/03-divergences.md). The likelihood ratio of $z$ is $PK(z)/QK(z)$. In the numerator, write each $p(y)$ as $q(y)$ times its likelihood ratio $e^{L(y)} = p(y)/q(y)$. That turns the ratio into a weighted average of the likelihood ratios of the $y$ that can produce $z$:
$$e^{L^f(z)} = \frac{PK(z)}{QK(z)} = \sum_y\frac{q(y)K(z \mid y)}{QK(z)}\,e^{L(y)}.$$
The weight of each $y$ is the share of the $D'$-probability of $z$ that comes through $y$. The weights are non-negative and sum to 1, because $\sum_y q(y)K(z \mid y) = QK(z)$. So $e^{L^f(z)}$ lies between the smallest and the largest $e^{L(y)}$, which is the second inequality in @eq-dpprop-post.

**Concretely, randomized response followed by a second coin.** You collect answers with $\varepsilon_0 = \ln 3$ (truth with probability $\frac34$), then, before publishing, flip each answer with probability $f = \frac14$ using a fresh coin. (Here $f$ is a number, the flip probability.) The published answer $z = 1$ can be reached in two ways: the first answer was 1 and was kept, or it was 0 and was flipped. Each path's probability is the product of its two steps. @tbl-dpprop-rr-flip follows the published answer $z = 1$.

| Path to $z = 1$ | Probability under bit 1 ($D$) | Probability under bit 0 ($D'$) | $e^{L(y)}$ of the first answer |
| --- | --- | --- | --- |
| $y = 1$, kept | $\frac34 \cdot \frac34 = \frac9{16}$ | $\frac14 \cdot \frac34 = \frac3{16}$ | 3 |
| $y = 0$, flipped | $\frac14 \cdot \frac14 = \frac1{16}$ | $\frac34 \cdot \frac14 = \frac3{16}$ | $\frac13$ |
| total | $\frac{10}{16} = 0.625$ | $\frac6{16} = 0.375$ | — |

: Randomized response with $\varepsilon_0 = \ln 3$, then a fresh coin that flips the answer with probability $\frac14$. The published $z = 1$ is reached by two paths. {#tbl-dpprop-rr-flip}

The likelihood ratio of $z = 1$ is $0.625/0.375 = \frac53$. Step 4 predicts it as the average of $3$ and $\frac13$, weighted by each path's share of the $D'$ column: each path contributes $\frac3{16}$ of the total $\frac6{16}$, a weight $\frac{3/16}{6/16} = \frac12$ each, so $\frac12(3 + \frac13) = \frac53$. The published $z = 0$ gets the remaining probability, $\frac6{16}$ under $D$ and $\frac{10}{16}$ under $D'$, a ratio of $\frac35$. So the published answers are $\ln\frac53 = 0.5108$-DP, less than $\ln 3 = 1.0986$. The right panel of @fig-dpprop-post shows $\varepsilon$ for every flip probability: it falls to 0 at $f = \frac12$, where the published bit is a fair coin whatever the first answer was.

### Interpretation

The first consequence is that nothing downstream needs a new proof. A model trained on DP statistics, synthetic data sampled from a DP model, a plot, a dashboard: all are functions of the private output that never read the data again, so all inherit the guarantee. This is why DP systems privatize at one narrow point (noisy counts, noisy gradients) and treat everything after it as public.

The same argument covers attackers. A membership-inference attacker of [Part 2](../part02-attacks/02-membership-inference.md), who tries to tell whether a given record was in the data, sees $y$ and outputs a guess $A(y)$, possibly using auxiliary knowledge and its own randomness. That guess is a channel applied to $y$, so it is a post-processing. So the bounds of [The definition of differential privacy](01-definition.md) on an attack's true and false positive rates hold for every attack, including attacks invented after the release. In hypothesis-testing terms: the **trade-off curve** of [Hypothesis testing](../part01-prerequisites/04-hypothesis-testing.md) gives, for each false positive rate, the smallest false negative rate any test can reach. Under processing it can only rise, because any test on $z$ is also a test on $y$: compute $z$ from $y$, then apply it.

How much does processing lose? Deterministic, invertible processing (relabelling, a change of units) keeps the privacy loss exactly, since each $z$ comes from a single $y$ and nothing is averaged. Processing that merges outputs with different privacy losses lowers it, as the coin flip does when it mixes a ratio of 3 with a ratio of $\frac13$. Most processing you do for utility (rounding, clipping) merges only a little, so do not count on it: the guarantee you report is the one of $M$.

The one condition is that $f$ must not touch the data, because step 1 of the derivation needed the same $g$ under $D$ and $D'$. "Compute the noisy count, then look up the exact count of the winning category" is not post-processing. The second step reads the data, so it is a second mechanism, and a non-private one (@sec-dpprop-limits).

### Example: rounding and thresholding a noisy count

Add Laplace noise of scale $b = 1$ to a count. Neighbouring datasets have counts 0 and 1, so the outputs are $\mathrm{Lap}(0, 1)$ and $\mathrm{Lap}(1, 1)$, with densities $\frac12e^{-|y|}$ and $\frac12e^{-|y - 1|}$. Their privacy loss at $y$ is $|y - 1| - |y|$. Two points 1 apart cannot differ in their distance to $y$ by more than 1 (the triangle inequality: $|y - 1| \le |y| + 1$ and $|y| \le |y - 1| + 1$), so this loss is never more than 1 in size, and it equals $(1 - y) - (-y) = 1$ for every $y \le 0$. The largest log-ratio, the max-divergence of [Divergences](../part01-prerequisites/03-divergences.md), is therefore 1: the noisy count is 1-DP. (The Laplace mechanism in general is in [The Laplace and Gaussian mechanisms](03-laplace-gaussian.md).) Now post-process it in two ways: round it to the nearest integer, or publish only whether it exceeds $\frac12$. @tbl-dpprop-laplace-post gives the exact privacy profiles from the script, and @fig-dpprop-post draws them.

| Published | Smallest pure $\varepsilon$ | $\delta(0.25)$ | $\delta(0.5)$ |
| --- | --- | --- | --- |
| Noisy count $c + Z$ | 1.0000 | 0.3127 | 0.2212 |
| Rounded to an integer | 1.0000 | 0.3073 | 0.1967 |
| Only "is $c + Z > \frac12$?" | 0.8318 | 0.3073 | 0.1967 |

: Exact privacy profiles of a count with $\mathrm{Lap}(0, 1)$ noise, before and after two post-processings. Each row is on or below the one above it, as @eq-dpprop-post requires. {#tbl-dpprop-laplace-post}

![Left: the privacy profile $\delta(\varepsilon)$ of a count with $\mathrm{Lap}(0, 1)$ noise (black), after rounding to an integer (blue), and after keeping only whether it exceeds $\frac12$ (orange). Post-processing can only lower the curve. Right: randomized response with $\varepsilon_0 = \ln 3$ followed by a fresh coin that flips the answer with probability $f$; the dot is $f = \frac14$, where $\varepsilon = \ln\frac53$.](figures/dpprop_postprocessing.svg){#fig-dpprop-post}

Three things can be read from the table and the figure, and each follows from the averaging picture of step 4.

Rounding keeps the pure $\varepsilon$. Every output $j \le -1$ of the rounded count collects the cell $[j - \frac12, j + \frac12]$, which lies entirely left of 0. There the two Laplace densities have the constant ratio $e$, so rounding merges outputs with the same loss, and an average of equal ratios is that ratio: nothing is averaged away.

Rounding still lowers $\delta(\varepsilon)$ for smaller $\varepsilon$, because of the cell around 0. That cell, $[-\frac12, \frac12]$, mixes outputs whose losses run from $-1$ to $1$. Its probability is $1 - e^{-1/2} = 0.3935$ when the count is 0 and $\frac12(e^{-1/2} - e^{-3/2}) = 0.1917$ when it is 1, so its loss is $\ln(0.3935/0.1917) = 0.7191$. For $\varepsilon$ below that value, the worst event of the rounded output is "the rounded count is at most 0". That is the same as "$c + Z < \frac12$", exactly the threshold event, so the rounded and thresholded curves coincide there.

The one-bit summary is the most private. Its "yes" event, $c + Z > \frac12$, needs $Z > \frac12$ when the count is 0, which has probability $\frac12e^{-1/2} = 0.3033$ under the Laplace tail, and $Z > -\frac12$ when the count is 1, which has probability $1 - 0.3033 = 0.6967$. Its privacy loss is $\ln(0.6967/0.3033) = 0.8318$, and the "no" event gives the same loss with the opposite sign.

### Check yourself

- A data scientist reads a DP histogram, notices that one bin looks odd, and decides to publish a second histogram with finer bins. Is the second histogram covered by post-processing? *(No. Choosing to publish is post-processing, but computing a finer histogram reads the data again. It is a second mechanism, and its cost adds by composition.)*
- Can post-processing turn a $(1, 10^{-6})$-DP mechanism into a $(0.5, 10^{-6})$-DP one? *(It can, as the coin flip shows, but only by destroying information. It can never do the reverse.)*

## Group privacy {#sec-dpprop-group}

### Context

The definition protects one record: datasets that differ in one row. But people are not rows. A user of a messaging app contributes hundreds of messages, a household shares an address and an income, and a disease runs in a family. If the neighbouring relation is example-level and you want to know what an adversary learns about all $k$ records of one user at once, you need a guarantee for datasets that differ in $k$ records. Group privacy gives it from the one-record guarantee, at a price.

### Statement

Say $D$ and $D''$ are at **distance $k$** if a path $D = D_0 \simeq D_1 \simeq \dots \simeq D_k = D''$ joins them, changing one record per step. The idea is to apply the one-record guarantee once per step and chain the results. Each step multiplies the probability bound by $e^\varepsilon$ and adds a slack $\delta$, so after $k$ steps the factor is $e^{k\varepsilon}$. The slacks pile up in a geometric sum, because a $\delta$ added early is multiplied by $e^\varepsilon$ again at every later step. If $M$ is $(\varepsilon, \delta)$-DP, then for every such pair and every event $S$,
$$\Pr[M(D) \in S] \le e^{k\varepsilon}\Pr[M(D'') \in S] + \delta_k, \qquad \delta_k = \delta\sum_{j=0}^{k-1}e^{j\varepsilon} = \delta\,\frac{e^{k\varepsilon} - 1}{e^\varepsilon - 1} \le k\,e^{(k-1)\varepsilon}\delta.$$ {#eq-dpprop-group}
In particular, pure $\varepsilon$-DP gives pure $k\varepsilon$-DP for groups of size $k$. For $\delta > 0$ the bound $\delta_k$ grows exponentially in $k$, and it is attained by some mechanisms.

### Derivation

**Pure DP, through the privacy loss.** The likelihood ratio between the two ends of the path factors into the ratios of the steps, so the losses add, and each one is at most $\varepsilon$.

1. **Telescope the likelihood ratio** along the path. Multiply and divide by the densities at the intermediate datasets $D_1, \dots, D_{k-1}$; each one appears once in a numerator and once in a denominator, so they cancel and only the two ends remain:
$$\frac{p_{M(D_0)}(y)}{p_{M(D_k)}(y)} = \prod_{i=1}^k\frac{p_{M(D_{i-1})}(y)}{p_{M(D_i)}(y)}, \qquad\text{so}\qquad L_{D_0,D_k}(y) = \sum_{i=1}^k L_{D_{i-1},D_i}(y).$$
2. **Bound each term** by Fact 1, form 3: each pair $D_{i-1} \simeq D_i$ is neighbouring, so $|L_{D_{i-1},D_i}(y)| \le \varepsilon$.
3. **Add:** a sum of $k$ terms, each between $-\varepsilon$ and $\varepsilon$, lies between $-k\varepsilon$ and $k\varepsilon$, so $|L_{D_0,D_k}(y)| \le k\varepsilon$ for every $y$.

**Concretely, a noisy count.** Take a count with $\mathrm{Lap}(0, 1)$ noise and the path of counts $0, 1, 2, 3$ ($k = 3$). The noisy count has density $\frac12e^{-|y - c|}$ when the true count is $c$. The ratio of the densities at counts $c$ and $c + 1$ is $e^{|y - c - 1| - |y - c|}$, so the privacy loss between counts $c$ and $c + 1$ at output $y$ is $L_{c,c+1}(y) = |y - c - 1| - |y - c|$ (the ratio of two Laplace densities): how much farther $y$ is from $c + 1$ than from $c$. @tbl-dpprop-path adds the steps at three outputs.

| Output $y$ | $L_{0,1}(y)$ | $L_{1,2}(y)$ | $L_{2,3}(y)$ | Sum | Direct: $\lvert y - 3\rvert - \lvert y\rvert$ |
| --- | --- | --- | --- | --- | --- |
| $-2$ | $3 - 2 = 1$ | $4 - 3 = 1$ | $5 - 4 = 1$ | 3 | $5 - 2 = 3$ |
| $0.5$ | $0.5 - 0.5 = 0$ | $1.5 - 0.5 = 1$ | $2.5 - 1.5 = 1$ | 2 | $2.5 - 0.5 = 2$ |
| $1.5$ | $0.5 - 1.5 = -1$ | $0.5 - 0.5 = 0$ | $1.5 - 0.5 = 1$ | 0 | $1.5 - 1.5 = 0$ |

: Privacy losses along a path of three neighbouring counts, with $\mathrm{Lap}(0, 1)$ noise. The step losses add up to the loss between the ends. {#tbl-dpprop-path}

At $y = -2$ the output lies left of every count on the path, so every step points the same way and the sum reaches $3\varepsilon$. At $y = 1.5$, in the middle of the path, the steps point in different directions and cancel. The script confirms that the rounded noisy count has pure $\varepsilon$ exactly $k$ for counts that differ by $k = 1, 2, 3, 5$: for a count, group privacy is tight.

**Approximate DP, through events.** With $\delta > 0$ the privacy loss is no longer bounded, so there is nothing to add up. Instead, chain the event form of the definition: bound the probability under $D_0$ by the one under $D_1$, that one by the one under $D_2$, and so on.

1. **One step.** For every event $S$, $\Pr[M(D_{i-1}) \in S] \le e^\varepsilon\Pr[M(D_i) \in S] + \delta$ (Fact 1, form 1).
2. **Substitute the next step** into the right side. Writing $P_i = \Pr[M(D_i) \in S]$, the bound on $P_1$ goes in place of $P_1$:
$$P_0 \le e^\varepsilon P_1 + \delta \le e^\varepsilon\big(e^\varepsilon P_2 + \delta\big) + \delta = e^{2\varepsilon}P_2 + \delta(1 + e^\varepsilon).$$
3. **Repeat $k$ times** (induction): $P_0 \le e^{k\varepsilon}P_k + \delta(1 + e^\varepsilon + \dots + e^{(k-1)\varepsilon})$. Each new $\delta$ enters before $k - 1, k - 2, \dots, 0$ more multiplications by $e^\varepsilon$, which is where the geometric sum comes from.
4. **Simplify:** the geometric sum is $\frac{e^{k\varepsilon} - 1}{e^\varepsilon - 1}$ (multiply it by $e^\varepsilon - 1$ and all terms but the first and last cancel). Each of its $k$ terms is at most the largest one, $e^{(k-1)\varepsilon}$, giving the simpler but looser $k\,e^{(k-1)\varepsilon}\delta$.

**The bound is attained.** To show that no better bound holds for all $(\varepsilon, \delta)$-DP mechanisms, build one for which every inequality in the chain is an equality. That needs an event on which each one-step bound is met exactly, so that step 2 loses nothing. Here is a mechanism on counts for which @eq-dpprop-group holds with equality. Fix a size $K$ and let $M$ output "alarm" with probability $a_c = \delta\sum_{j=0}^{K-c-1}e^{j\varepsilon}$ when the count is $c \le K$ (and $a_c = 0$ above $K$), and "quiet" otherwise. Pulling the first term out of the sum and $e^\varepsilon$ out of the rest shows that the $a_c$ satisfy $a_c = e^\varepsilon a_{c+1} + \delta$, so each neighbouring pair meets the event form with equality on "alarm", and the script checks that the other events are also within $(\varepsilon, \delta)$. For counts $0$ and $k$ the telescoping of step 3 runs with equality: $a_0 - e^{k\varepsilon}a_k = \delta\sum_{j=0}^{k-1}e^{j\varepsilon} = \delta_k$.

### Interpretation

Pure DP degrades linearly. A $0.1$-DP guarantee per record is a $1$-DP guarantee for a user with 10 records and a $10$-DP guarantee, almost nothing, for a user with 100: a likelihood ratio of up to $e^{10} \approx 22{,}000$ lets an adversary be nearly certain. This is the formal version of the "correlated records" caveat of [The definition of differential privacy](01-definition.md): if one person's data appear in $k$ rows, protecting one row does not protect the person.

Approximate DP degrades exponentially in $\delta$. With $\varepsilon = 1$ and $k = 10$, the geometric sum gives $\delta_k = \frac{e^{10} - 1}{e - 1}\,\delta = 12818.3\,\delta$. And $\delta$ must already be much smaller than $1/n$ for a dataset of $n$ records, because $\delta$ is a probability of failing outright: a mechanism that publishes each record in full with probability $\delta$ is $(0, \delta)$-DP and still leaks $n\delta$ records on average ([The definition of differential privacy](01-definition.md)). Multiply such a $\delta$ by thousands and group privacy for large groups is often vacuous.

The bound is generic; a specific mechanism can do much better. For a count with Gaussian noise, a group of $k$ records just shifts the count by $k$ instead of 1, so its exact group profile is that of a Gaussian pair with a larger shift, as the Gaussian column of @tbl-dpprop-group shows. The tool that tracks this exactly for the Gaussian is in [Part 4](../part04-privacy-accounting/00-introduction.md).

In practice, the better fix is to change the neighbouring relation. If you care about users, define neighbours as datasets that differ in one user's records, bound each user's contribution (clip it), and prove DP directly at that level. Group privacy is the fallback when you cannot.

### Example: from records to users

Each record is protected at $(\varepsilon, \delta) = (0.5, 10^{-5})$. @tbl-dpprop-group compares the bound with two exact answers. The first is the alarm mechanism, built to attain the bound. The second is a count with Gaussian noise of standard deviation $\sigma$, calibrated so that $\delta(0.5) = 10^{-5}$ exactly at shift 1. For two normal distributions with standard deviation $\sigma$ whose means differ by $s$, the privacy loss is a straight line in the output, so the event "loss above $\varepsilon$" is a half-line and the hockey-stick divergence is a difference of two normal tail probabilities. With $\mu = s/\sigma$ and $\Phi$ the standard normal cumulative distribution function, it is $\Phi(\frac\mu2 - \frac\varepsilon\mu) - e^\varepsilon\Phi(-\frac\mu2 - \frac\varepsilon\mu)$, the closed form from [Divergences](../part01-prerequisites/03-divergences.md). The script finds $\sigma = 7.0318$ by bisection on it, and for a group of $k$ records it evaluates the same formula with the shift $s = k$. @fig-dpprop-group plots the same numbers up to $k = 12$.

| Group size $k$ | $\varepsilon_k = k\varepsilon$ | $\delta_k$ from @eq-dpprop-group | Simple form $k e^{(k-1)\varepsilon}\delta$ | Alarm mechanism, exact | Gaussian count, exact |
| --- | --- | --- | --- | --- | --- |
| 1 | 0.5 | $1.000 \times 10^{-5}$ | $1.000 \times 10^{-5}$ | $1.000 \times 10^{-5}$ | $1.000 \times 10^{-5}$ |
| 2 | 1 | $2.649 \times 10^{-5}$ | $3.297 \times 10^{-5}$ | $2.649 \times 10^{-5}$ | $2.551 \times 10^{-5}$ |
| 3 | 1.5 | $5.367 \times 10^{-5}$ | $8.155 \times 10^{-5}$ | $5.367 \times 10^{-5}$ | $4.858 \times 10^{-5}$ |
| 5 | 2.5 | $1.724 \times 10^{-4}$ | $3.695 \times 10^{-4}$ | $1.724 \times 10^{-4}$ | $1.288 \times 10^{-4}$ |
| 10 | 5 | $2.272 \times 10^{-3}$ | $9.002 \times 10^{-3}$ | $2.272 \times 10^{-3}$ | $7.592 \times 10^{-4}$ |

: Group privacy from a $(0.5, 10^{-5})$ guarantee per record: the bound, its simple form, and the exact $\delta_k$ of two mechanisms at $\varepsilon_k = k\varepsilon$. {#tbl-dpprop-group}

![The $\delta$ of group privacy at $\varepsilon_k = k\varepsilon$, for a per-record guarantee of $(0.5, 10^{-5})$, on a log scale. The geometric-sum bound (dashed) is attained by the alarm mechanism (red dots); the simple form (dotted) is looser by a factor that grows with $k$; Gaussian noise on a count (blue squares) does better than the bound but still grows quickly with $k$.](figures/dpprop_group.svg){#fig-dpprop-group}

The geometric sum is exact for the alarm mechanism at every $k$, so no better bound holds for all $(\varepsilon, \delta)$-DP mechanisms. The simple form replaces every term of the sum by the largest one, so it is looser, by a factor $3.297/2.649 = 1.24$ at $k = 2$ and $9.002/2.272 = 3.96$ at $k = 10$; use the sum. The Gaussian count is a factor $2.272/0.7592 = 2.99$ below the bound at $k = 10$, and the gap widens with $k$ (@fig-dpprop-group). Even so, for a user with 10 records the record-level $(0.5, 10^{-5})$ becomes $(5, 7.6 \times 10^{-4})$: a weak guarantee.

### Check yourself

- A dataset has one row per hospital visit, and a mechanism is $0.2$-DP at the visit level. A patient made 7 visits. What guarantee does the patient get? *($7 \times 0.2 = 1.4$-DP, by group privacy, assuming nothing else about the mechanism.)*
- Why does the derivation for $\delta > 0$ not just add $k$ copies of $\delta$? *(Each step multiplies everything before it, including the $\delta$s already accumulated, by $e^\varepsilon$. A $\delta$ that entered early is inflated by the later steps.)*

## Basic sequential composition {#sec-dpprop-comp}

### Context

An analyst asks $k$ queries of the same dataset, each answered by a DP mechanism with fresh noise. A training run takes $k$ noisy gradient steps. A survey asks the same question twice. Each output alone is private, but an adversary sees all of them. The question is what guarantee holds for the whole transcript $(y_1, \dots, y_k)$, including when each mechanism is chosen after seeing the previous outputs. Fact 3 already gives the answer for pure DP: outputs with independent coins have probabilities that multiply, so their privacy losses add, and so do the budgets.

### Statement

Let $M_1, \dots, M_k$ run on the same dataset with independent coins. Each $M_i$ may also take the previous outputs $y_1, \dots, y_{i-1}$ as input, the way an analyst picks the next query after reading the last answer; this is **adaptive** composition, and **non-adaptive** composition is the special case where $M_i$ ignores them. The condition is that each step is private however the earlier outputs turned out. If, for every fixed value of the previous outputs, $D \mapsto M_i(D; y_1, \dots, y_{i-1})$ is $(\varepsilon_i, \delta_i)$-DP, then the mechanism that releases $(y_1, \dots, y_k)$ is
$$\Big(\sum_{i=1}^k\varepsilon_i,\ \sum_{i=1}^k\delta_i\Big)\text{-DP}.$$ {#eq-dpprop-basic}
For pure DP the bound $\sum\varepsilon_i$ is the exact worst-case privacy loss whenever each $M_i$ attains its own worst case at outputs that can occur together, as $k$ randomized-response answers about the same bit do.

### Derivation

**Pure DP, through the privacy loss.** Write the probability of a transcript as a product of one factor per step, take logs, and bound each term.

1. **Factor the probability of a transcript** by the chain rule: the probability of $(y_1, \dots, y_k)$ is the probability of $y_1$, times that of $y_2$ given $y_1$, and so on. Under $D$,
$$p_D(y_1, \dots, y_k) = \prod_{i=1}^k p_{M_i(D;\,y_{<i})}(y_i),$$
where $y_{<i} = (y_1, \dots, y_{i-1})$ and each factor uses fresh coins. The same holds under $D'$, with the same previous outputs plugged in.
2. **Take the log-ratio** (Fact 3, adaptive version): the log of a ratio of products is a sum of log-ratios, $L_{D,D'}(y_1, \dots, y_k) = \sum_{i=1}^k L^{(i)}_{D,D'}(y_i;\,y_{<i})$, where $L^{(i)}$ is the privacy loss of $M_i$ with its previous outputs fixed.
3. **Bound each term.** For the fixed $y_{<i}$, $M_i$ is $\varepsilon_i$-DP by assumption, so $|L^{(i)}| \le \varepsilon_i$.
4. **Add:** $|L_{D,D'}| \le \sum_i\varepsilon_i$ at every transcript, which is pure $\sum_i\varepsilon_i$-DP (Fact 1, form 3).

**Concretely, an adaptive survey.** Ask a person for a randomized-response answer with $\varepsilon_1 = \ln 3$. If the answer is 1, ask again with $\ln 3$; if it is 0, ask again with $0.5$, which means telling the truth with probability $e^{0.5}/(1 + e^{0.5}) = 0.6225$ (randomized response at level $\varepsilon$ tells the truth with probability $e^\varepsilon/(1 + e^\varepsilon)$, so that truth and lie have ratio $e^\varepsilon$). The second mechanism depends on the first output, but for each value of $y_1$ it is at most $\ln 3$-DP. @tbl-dpprop-adaptive lists the four transcripts, from the script. Each probability is the product of the two steps; for example $(0, 1)$ under bit 1 is a lie at the first step and the truth at the second, $\frac14 \times 0.6225 = 0.1556$.

| $(y_1, y_2)$ | Under bit 1 ($D$) | Under bit 0 ($D'$) | $L = L^{(1)} + L^{(2)}$ |
| --- | --- | --- | --- |
| $(1, 1)$ | 0.5625 | 0.0625 | $\ln 3 + \ln 3 = 2.1972$ |
| $(1, 0)$ | 0.1875 | 0.1875 | $\ln 3 - \ln 3 = 0$ |
| $(0, 1)$ | 0.1556 | 0.2832 | $-\ln 3 + 0.5 = -0.5986$ |
| $(0, 0)$ | 0.0944 | 0.4668 | $-\ln 3 - 0.5 = -1.5986$ |

: An adaptive composition of two randomized-response answers: the second one uses $\varepsilon = \ln 3$ after a 1 and $\varepsilon = 0.5$ after a 0. Each transcript's privacy loss is the sum of the two step losses. {#tbl-dpprop-adaptive}

The worst loss is $2.1972 = \ln 3 + \ln 3$, exactly the bound with $\varepsilon_2 = \max(\ln 3, 0.5)$, the largest budget the second step can have.

**Approximate DP, through a two-step lemma.** With $\delta > 0$ the losses are no longer bounded, so work with events and Fact 2. The plan is to split an event over the first output, use the second mechanism's guarantee for each value of that output, and then use the first mechanism's guarantee on what is left, which is a soft test of the first output. Take $k = 2$; induction on $k$ does the rest, since the first $k - 1$ steps form one mechanism.

1. **Split an event over the first output.** For a set $S$ of transcripts, let $S_{y_1} = \{y_2 : (y_1, y_2) \in S\}$, the second outputs that complete $y_1$ into $S$. The transcript lands in $S$ when the first output is some $y_1$ and the second lands in $S_{y_1}$, so
$$\Pr_D[(Y_1, Y_2) \in S] = \sum_{y_1}p_1(y_1)\,P_2^{y_1}(S_{y_1}),$$
where $p_1$ is the distribution of $M_1(D)$ and $P_2^{y_1}$ that of $M_2(D; y_1)$. Write $q_1$ and $Q_2^{y_1}$ for $D'$.
2. **Apply the second mechanism's guarantee inside the sum.** For each $y_1$, $P_2^{y_1}(S_{y_1}) \le e^{\varepsilon_2}Q_2^{y_1}(S_{y_1}) + \delta_2$, and also $P_2^{y_1}(S_{y_1}) \le 1$ because it is a probability. So $P_2^{y_1}(S_{y_1}) \le g(y_1) + \delta_2$ with
$$g(y_1) = \min\big(1,\ e^{\varepsilon_2}Q_2^{y_1}(S_{y_1})\big) \in [0, 1].$$
3. **Sum over $y_1$:** the weights $p_1(y_1)$ add up to 1, so $\delta_2$ comes out once, $\Pr_D[S] \le \sum_{y_1}p_1(y_1)g(y_1) + \delta_2 = \mathbb E_{p_1}[g] + \delta_2$.
4. **Apply the first mechanism's guarantee to $g$** (Fact 2, which needs values in $[0, 1]$; that is why $g$ was capped at 1): $\mathbb E_{p_1}[g] \le e^{\varepsilon_1}\mathbb E_{q_1}[g] + \delta_1$.
5. **Undo the cap:** $g(y_1) \le e^{\varepsilon_2}Q_2^{y_1}(S_{y_1})$, so $\mathbb E_{q_1}[g] \le e^{\varepsilon_2}\sum_{y_1}q_1(y_1)Q_2^{y_1}(S_{y_1}) = e^{\varepsilon_2}\Pr_{D'}[S]$, by step 1 under $D'$.
6. **Combine:** $\Pr_D[S] \le e^{\varepsilon_1 + \varepsilon_2}\Pr_{D'}[S] + \delta_1 + \delta_2$. Nothing used that $M_2$ ignores $y_1$, so this covers adaptive composition.

### Interpretation

Basic composition is exact for the worst case. For $k$ randomized-response answers about the same bit, the transcript "all answers equal the true bit" has loss exactly $k\ln 3$, since each answer contributes $+\ln 3$. No smaller pure $\varepsilon$ holds.

But the worst case is rare. That transcript has probability $(\frac34)^k$ under $D$. Typical transcripts mix truthful answers with lies, and their losses cancel in part. Each answer contributes $+\varepsilon_0$ with probability $t = e^{\varepsilon_0}/(1 + e^{\varepsilon_0})$ (it told the truth) and $-\varepsilon_0$ otherwise, so its mean is $\varepsilon_0(2t - 1) = \varepsilon_0\frac{e^{\varepsilon_0} - 1}{e^{\varepsilon_0} + 1} = \varepsilon_0\tanh(\varepsilon_0/2)$. For $\varepsilon_0 = \ln 3$ this is $\varepsilon_0\cdot\frac{3 - 1}{3 + 1} = \frac{\varepsilon_0}2$. The privacy loss of $k$ answers is a sum of $k$ independent terms of $\pm\varepsilon_0$, so it concentrates around its mean $k\varepsilon_0\tanh(\varepsilon_0/2)$ (half of $k\ln 3$ for $\varepsilon_0 = \ln 3$) with spread of order $\varepsilon_0\sqrt k$. That is the behaviour of any sum of bounded independent terms ([Concentration](../part01-prerequisites/02-concentration.md)): its standard deviation grows like $\sqrt k$, and deviations much larger than that are exponentially unlikely. Allowing a small $\delta$ lets you ignore that unlikely tail, and the $\varepsilon$ you need grows like $\sqrt k$ rather than $k$. That is advanced composition, derived in [Advanced and optimal composition](04-composition.md); [Exercise 6](../../book-exercises/part03-dp-foundations/02-properties.md#sec-sol-dpprop-6) previews it.

The $\delta$s add without inflation. Unlike group privacy, composition adds $\delta$s plainly: there is no chain of multiplications, because each mechanism compares the *same* pair $D, D'$. In group privacy each step moved to a new dataset, and the bound for the next step had to be multiplied through; here step 3 of the lemma just averages the second slack over the first output.

Adaptivity is free here, but only with fixed budgets. The theorem allows $M_i$ to depend on previous outputs, but its budget $(\varepsilon_i, \delta_i)$ must be fixed in advance. Choosing the budgets themselves from the outputs ("stop when the answer is good enough") needs privacy filters, also in [Advanced and optimal composition](04-composition.md).

### Example: asking the same question $k$ times

A survey asks the same yes/no question $k$ times, each time with fresh randomized response at $\varepsilon_0 = \ln 3$. Only the number $j$ of answers equal to 1 matters for the privacy loss, which is $(2j - k)\ln 3$: $j$ answers with loss $+\ln 3$ and $k - j$ with $-\ln 3$. When the true bit is 1, $j$ follows a binomial distribution with $k$ trials and success probability $\frac34$; when it is 0, success probability $\frac14$. @fig-dpprop-rr-loss shows the distribution of the privacy loss for $k = 10$, and @tbl-dpprop-rr-comp gives the exact numbers from the script. "Exact $\varepsilon$ at $\delta = 10^{-6}$" is the smallest $\varepsilon$ with $\delta(\varepsilon) \le 10^{-6}$ for the $k$ answers, computed from the binomial distribution of the number of truthful answers.

![Distribution of the privacy loss $L_{D,D'}$ of 10 randomized-response answers with $\varepsilon_0 = \ln 3$ each, when the true bit is 1 (blue) or 0 (orange). The loss takes the values $(2j - 10)\ln 3$, where $j$ is the number of answers equal to 1. The basic bound $10\ln 3$ is reached, but only with probability $(3/4)^{10} = 0.056$; the mean under $D$ is half of it.](figures/dpprop_rr_loss.svg){#fig-dpprop-rr-loss}

| $k$ | Basic bound $k\varepsilon_0$ | Exact worst-case loss | $\Pr_D[L = k\varepsilon_0] = (\frac34)^k$ | Mean loss under $D$ | Exact $\varepsilon$ at $\delta = 10^{-6}$ |
| --- | --- | --- | --- | --- | --- |
| 1 | 1.099 | 1.099 | 0.75 | 0.549 | 1.099 |
| 10 | 10.986 | 10.986 | $5.631 \times 10^{-2}$ | 5.493 | 10.986 |
| 20 | 21.972 | 21.972 | $3.171 \times 10^{-3}$ | 10.986 | 21.972 |
| 50 | 54.931 | 54.931 | $5.663 \times 10^{-7}$ | 27.465 | 52.680 |
| 100 | 109.861 | 109.861 | $3.207 \times 10^{-13}$ | 54.931 | 94.290 |

: $k$ randomized-response answers about the same bit, $\varepsilon_0 = \ln 3$ each: the basic bound, the exact worst case, how likely the worst case is, the mean privacy loss, and the exact $\varepsilon$ when $\delta = 10^{-6}$ is allowed. {#tbl-dpprop-rr-comp}

A slack $\delta$ only helps if some transcripts with large loss have total probability below $\delta$, so that they can be written off. Up to $k = 20$ the worst case has probability far above $10^{-6}$, so allowing $\delta = 10^{-6}$ buys almost nothing: the exact $\varepsilon$ equals the basic bound to three decimals. At $k = 50$ the worst case has probability $5.663 \times 10^{-7}$, below $10^{-6}$, and the exact $\varepsilon$ starts to drop. At $k = 100$ it buys $109.861 - 94.290 = 15.6$ nats. With a smaller per-answer budget the gain is much larger, because the $\sqrt k$ spread is then far below the $k\varepsilon_0$ worst case: at $\varepsilon_0 = 0.1$ the basic bound for $k = 1000$ answers is 100, while the exact $\varepsilon$ at $\delta = 10^{-6}$ is 19.345 (@fig-dpprop-composition). A simulation of 200,000 runs of 10 answers gives a mean loss of 5.500 (exact 5.493) and the worst case in a fraction 0.0559 of runs (exact 0.0563).

![The $\varepsilon$ of $k$ randomized-response answers: the basic bound $k\varepsilon_0$, which is also the exact worst-case loss (dashed), the exact $\varepsilon$ at $\delta = 10^{-6}$ and $10^{-3}$, and the mean privacy loss (dotted). Left: $\varepsilon_0 = \ln 3$; the exact curves leave the basic line only once $(3/4)^k$ drops below $\delta$. Right: $\varepsilon_0 = 0.1$; the exact curves grow like $\sqrt k$.](figures/dpprop_composition.svg){#fig-dpprop-composition}

**Composing approximate-DP mechanisms.** The same gap appears for the $\delta$ part. Take counts with Gaussian noise, each calibrated to $(0.5, 10^{-5})$ ($\sigma = 7.0318$). Changing one record shifts each of the $k$ noisy counts by 1, so the two output distributions are $k$-dimensional normals with independent coordinates of standard deviation $\sigma$, whose centres differ by the vector $(1, \dots, 1)$ of length $\sqrt k$. A normal with independent equal-variance coordinates looks the same in every direction, so you can rotate the axes until that shift points along the first one. The other $k - 1$ coordinates then have the same distribution under both datasets and carry no evidence. So the exact profile of $k$ such counts is that of one Gaussian pair with $\sqrt k$ times the shift, since independent Gaussian coordinates combine into one ([Divergences](../part01-prerequisites/03-divergences.md)). @tbl-dpprop-gauss-comp compares it with @eq-dpprop-basic at the same total $\delta$.

| $k$ | Basic: $(k\varepsilon, k\delta)$ | Exact $\varepsilon$ at the same $\delta = k \times 10^{-5}$ |
| --- | --- | --- |
| 1 | $(0.5, 10^{-5})$ | 0.500 |
| 2 | $(1, 2 \times 10^{-5})$ | 0.694 |
| 4 | $(2, 4 \times 10^{-5})$ | 0.966 |
| 10 | $(5, 10^{-4})$ | 1.504 |

: $k$ counts with Gaussian noise, each $(0.5, 10^{-5})$-DP: basic composition against the exact $\varepsilon$ at the same total $\delta$. {#tbl-dpprop-gauss-comp}

Basic composition is a valid bound, and the only one you need until the number of mechanisms grows. For ten Gaussian queries it already overstates $\varepsilon$ by a factor of $5/1.504 = 3.3$.

### Check yourself

- Two mechanisms are each $(1, 10^{-6})$-DP. What does basic composition give for running both, and what for running the first one twice with the same output reported twice? *($(2, 2 \times 10^{-6})$ for the two runs. Reporting one output twice is post-processing, so $(1, 10^{-6})$.)*
- In @tbl-dpprop-rr-comp, why is the exact $\varepsilon$ at $\delta = 10^{-6}$ equal to the basic bound for $k \le 20$? *(The worst transcript has probability $(3/4)^k \ge 3.171 \times 10^{-3}$ under $D$ and $e^{k\varepsilon_0}$ times less under $D'$. At $\varepsilon = k\varepsilon_0 - \eta$, slightly below the bound, its contribution to the hockey-stick divergence is its probability under $D$ minus $e^\varepsilon$ times its probability under $D'$, which is $(3/4)^k(1 - e^{-\eta}) \approx (3/4)^k\eta$. That alone must stay below $\delta$, so $\delta = 10^{-6}$ allows only $\eta \le 10^{-6}/(3.171 \times 10^{-3}) \approx 3 \times 10^{-4}$, invisible at three decimals.)*

## Parallel composition {#sec-dpprop-parallel}

### Context

A histogram counts people in disjoint bins. Per-region statistics use disjoint subsets of the data. In local randomized response, each person randomizes only their own bit. Sequential composition would charge the sum of the budgets, $d\varepsilon$ for $d$ bins, which is far too pessimistic: one person's record lands in one bin only, so only one output depends on it. Parallel composition makes that precise. It needs a careful look at the neighbouring relation, because there are two common ones. Under **add/remove**, neighbours differ by adding or removing one record. Under **replace**, neighbours have the same size and differ in the content of one record, which is a removal followed by an addition.

### Statement

The idea is that a changed record touches only the part of the data it falls in, so only that part's mechanism sees a difference, and you pay only that part's budget. Split the data universe into disjoint parts $U_1, \dots, U_d$ by a rule fixed in advance (it may depend on a record's content, but not on the rest of the dataset). Let $M_i$ run on $D \cap U_i$, the records of $D$ that fall in part $i$, with independent coins, and suppose each $M_i$ is $(\varepsilon_i, \delta_i)$-DP under the **add/remove** relation. Then the release $(M_1(D \cap U_1), \dots, M_d(D \cap U_d))$ is
$$\big(\max_i\varepsilon_i,\ \max_i\delta_i\big)\text{-DP under add/remove}.$$ {#eq-dpprop-parallel}
Under replace, a record that moves from part $a$ to part $b \ne a$ costs $(\varepsilon_a + \varepsilon_b, \delta_a + \delta_b)$, and one replaced within part $a$ costs that part's guarantee for two add/remove steps, at most group privacy with $k = 2$ (@eq-dpprop-group). For pure DP with equal budgets, both cases give $2\varepsilon$. If the parts are fixed slots (record $i$ always goes to part $i$, as in local randomized response, where each person is one slot), a replacement changes one part only and the cost is $(\max_i\varepsilon_i, \max_i\delta_i)$ again.

### Derivation

1. **Add/remove changes one part.** If $D' = D \cup \{x\}$, then $x$ lies in exactly one part $U_j$, because the parts are disjoint and cover the universe. For $i \ne j$, $D \cap U_i = D' \cap U_i$, so $M_i$ gets the same input and has the same output distribution on both.
2. **Pure DP: the other losses are zero.** By Fact 3 the privacy loss of the whole release is $\sum_iL^{(i)}$, and $L^{(i)} \equiv 0$ for $i \ne j$ because the two distributions are equal and their ratio is 1. So $|L| = |L^{(j)}| \le \varepsilon_j \le \max_i\varepsilon_i$.
3. **Approximate DP: condition on the unchanged parts.** Let $W$ be the outputs of the parts $i \ne j$. They have the same distribution under $D$ and $D'$ and are independent of $Y_j$. For an event $S$, fix $W$ first: given $W$, the release lands in $S$ when $Y_j$ falls in $S_W$, the set of $y_j$ completing $W$ into $S$. So $\Pr_D[S] = \mathbb E_W\big[\Pr_D[Y_j \in S_W]\big]$. Apply $M_j$'s guarantee inside the expectation, $\Pr_D[Y_j \in S_W] \le e^{\varepsilon_j}\Pr_{D'}[Y_j \in S_W] + \delta_j$, and average over $W$, which has the same distribution under $D'$: $\Pr_D[S] \le e^{\varepsilon_j}\Pr_{D'}[S] + \delta_j$.
4. **Replace changes up to two parts.** If $D'$ replaces a record $x \in U_a$ by $x' \in U_b$, then $D \cap U_a$ loses a record and $D \cap U_b$ gains one: two add/remove steps on two independent mechanisms. This is a composition of two mechanisms (@eq-dpprop-basic, with the other parts handled as in step 3), giving $(\varepsilon_a + \varepsilon_b, \delta_a + \delta_b)$. If $a = b$, part $a$ sees a removal and an addition, two neighbouring steps, which is group privacy with $k = 2$ inside one part.

### Interpretation

A histogram costs one $\varepsilon$, not $d$. Laplace noise of scale $1/\varepsilon$ on each of $d$ bins is $\varepsilon$-DP under add/remove, however many bins there are, because adding a record changes one bin by 1 and the Laplace noise on that bin covers it. Under replace it is $2\varepsilon$-DP, since moving a record changes two bins, which is why a histogram's sensitivity is often quoted as 2 under replace ([The Laplace and Gaussian mechanisms](03-laplace-gaussian.md)).

Local randomized response composes in parallel too. Each person's report depends only on their own bit, so the whole vector of $n$ reports is $\ln 3$-DP for any $n$, not $n\ln 3$. Each person is one slot, so replacing one person's bit changes only their own report.

The partition must not depend on the data. "Split the data at its median" changes when one record changes, which can move many records between parts, and then step 1 of the derivation fails: more than one part sees a difference. Use fixed bins, or account for the data-dependent split as a mechanism of its own.

### Example: a two-bin histogram

Take two bins, each with rounded $\mathrm{Lap}(0, 1)$ noise, and let the script compute the exact pure $\varepsilon$ of the pair of outputs under each relation. Adding or removing a record changes one bin by 1 and leaves the other alone, so only one bin contributes privacy loss, and $\varepsilon = 1.0000$, the budget of one bin. Replacing a record in bin 1 by one in bin 2 moves bin 1 up by 1 and bin 2 down by 1. Each bin then contributes a loss of at most 1, and the worst output has both noisy counts far out on the side that favours $D$, where both step losses equal 1 at once, so $\varepsilon = 2.0000$.

### Check yourself

- A dataset is split into 50 states, and each state's mean income is released with $0.5$-DP under add/remove. What is the guarantee for the whole release under add/remove, and under replace? *($0.5$ under add/remove; $1$ under replace, since a person who moves from one state to another changes two states' inputs.)*
- Why is it wrong to apply parallel composition to "the top 10% of earners" and "the rest"? *(The split depends on the data: adding one high earner can move a record across the boundary, changing both parts.)*

## What composes and what doesn't {#sec-dpprop-limits}

### Context

The three properties hold under weak assumptions, which makes it tempting to apply them where even those assumptions fail. Each assumption appeared in a derivation: post-processing needs $f$ not to read the data, composition needs independent coins and the same neighbouring relation for every step, and group privacy and composition both need the budgets fixed in advance. This section collects the ways those assumptions break in practice.

### Statement

@tbl-dpprop-rules lists what composes and what does not, with the assumption each row keeps or breaks.

| Situation | Composes? | Why |
| --- | --- | --- |
| Function of the output, own coins | Yes, at no cost | Post-processing (@sec-dpprop-post) |
| Second query on the same data, fresh noise | Yes, budgets add | Basic composition (@eq-dpprop-basic) |
| Second query chosen from the first answer | Yes, budgets add | Adaptive composition, budgets fixed in advance |
| Two queries sharing one noise draw | **No** | The noise cancels in a difference; Fact 3 needs independent coins |
| "Post-processing" that reads the data again | **No** | It is a new mechanism, often a non-private one |
| Budgets chosen from earlier outputs | Not by @eq-dpprop-basic | Needs privacy filters ([Advanced and optimal composition](04-composition.md)) |
| Mechanisms proved under add/remove, released under replace | Yes: $\varepsilon$ doubles to $2\varepsilon$ and $\delta$ grows to $(1 + e^\varepsilon)\delta$ | A replacement is a removal plus an addition, so group privacy with $k = 2$ (@eq-dpprop-group) applies, as in @eq-dpdef-replace of [The definition of differential privacy](01-definition.md) |
| A DP release plus an exact statistic of the same data | **No** | The exact statistic is not DP; the pair has no guarantee |
| Many records of one person | Yes, at $k\varepsilon$ and $\delta_k$ | Group privacy (@eq-dpprop-group) |

: What composes and what doesn't. {#tbl-dpprop-rules}

### Derivation: why shared noise breaks everything

Composition assumed that each mechanism draws fresh noise. Here is what happens when two releases share one draw: the noise can be subtracted away. Release two counts with the *same* Laplace draw $Z$: $c_1(D) + Z$, the number of smokers, and $c_2(D) + Z$, the number of smokers aged 40 or over. Alice is a smoker aged 30.

1. **The difference is noise-free:** $(c_1 + Z) - (c_2 + Z) = c_1(D) - c_2(D)$, the exact number of smokers under 40. It is a deterministic function of the data.
2. **It changes between neighbours.** Removing Alice lowers $c_1$ by 1 and leaves $c_2$, since she is under 40. So if the difference is $c$ under $D$, it is $c - 1$ under $D'$.
3. **So some event has probability 1 under $D$ and 0 under $D'$:** "the difference equals $c$". The event form $1 \le e^\varepsilon \cdot 0 + \delta$ fails for every $\varepsilon$ unless $\delta \ge 1$.

Each release alone is 1-DP. The pair is not DP at all. What went wrong is step 1 of the composition derivation: the transcript's probability is no longer a product of the two mechanisms' probabilities.

### Interpretation

Fresh randomness is part of the mechanism. Seeding two "independent" mechanisms with the same random seed, caching noisy answers and re-adding the same noise to a related query, or reusing noise across the shards of a distributed system are all instances of the shared-noise bug.

Exact releases are not "public information" you can ignore. If the exact total of a column is published elsewhere, a DP release of a related quantity may be differenced against it, just as the two counts were differenced above. The DP guarantee covers your mechanism, and adds nothing to a release that is not DP.

Post-processing must be data-blind. A pipeline that takes the noisy top category, then reports its exact count, has added a non-private mechanism.

The neighbouring relation is part of the budget. A guarantee proved under add/remove and reported under replace can be off by a factor of 2 ([The definition of differential privacy](01-definition.md)), because a replacement is two add/remove steps. Mixing the two inside one accountant is a bug, one that comes back with subsampling in [Subsampling](../part04-privacy-accounting/05-subsampling.md).

### Example: the differencing attack in simulation

The script simulates the attacker of the derivation: Alice is in the data with probability $\frac12$, and the attacker guesses "in" when the difference of the two released counts exceeds the midpoint. Over 200,000 runs with $\mathrm{Lap}(0, 1)$ noise, the attack with shared noise has accuracy $1.0000$: the difference is exact, so the attacker is always right.

With fresh noise the same attack has accuracy $0.6206$. Here $c_2 + Z_2$ does not depend on Alice at all, so the pair is just a 1-DP count plus independent noise. How well could any attack do? With Alice in or out with probability $\frac12$ each, the best accuracy is $\frac12$ plus half the **total variation distance** between what the attacker sees in the two cases, the largest difference in probability that the two distributions give to any one event ([Divergences](../part01-prerequisites/03-divergences.md)). For $\mathrm{Lap}(0, 1)$ and $\mathrm{Lap}(1, 1)$ the best event is "the output is above $\frac12$", which has probability $\frac12e^{-1/2}$ under $\mathrm{Lap}(0, 1)$ and $1 - \frac12e^{-1/2}$ under $\mathrm{Lap}(1, 1)$, a difference of $1 - e^{-1/2}$. So the best accuracy is $\frac12 + \frac12(1 - e^{-1/2}) = 0.6967$. The difference attack adds the noise of the second count, which only hurts it.

### Check yourself

- A developer adds Laplace noise to a query answer and caches it, so that repeated identical queries return the same noisy answer. Is that safe? *(Yes. Returning the same output again is post-processing of one release. The bug is reusing the noise draw for a different query.)*
- An analysis runs 10 queries at $\varepsilon = 0.1$ each, but the eleventh is run only if the first ten look interesting. Does basic composition bound the total by $1.1$? *(Yes, if the eleventh query's budget of $0.1$ was fixed in advance, because deciding whether to run it is adaptive composition with a mechanism that may output nothing. If its budget is chosen from the answers, you need a privacy filter.)*

## Code: check every property by exact computation and simulation

The script [code/properties.py](../../code/part03-dp-foundations/properties.py) computes every number in this chapter and draws every figure. Each mechanism is a small typed function that returns its exact output distributions on two neighbouring datasets; the privacy profile $\delta(\varepsilon)$ is the hockey-stick divergence of those distributions, and pure $\varepsilon$ is their largest log-ratio. Post-processing applies a channel; composition of randomized-response answers uses the binomial distribution of the number of truthful answers; the Gaussian numbers use the closed form from Part 1. From the repository root:

```bash
uv run python code/part03-dp-foundations/properties.py
```

**Output** (seed 0), summarized in @tbl-dpprop-code:

| Experiment | Result |
| --- | --- |
| Randomized response $\ln 3$, then flip with probability $f$ | $f = 0.1$: $\varepsilon = 0.8473$; $f = 0.25$: $0.5108 = \ln\frac53$; $f = 0.5$: 0 |
| Count + $\mathrm{Lap}(0, 1)$, post-processed | Raw: $\varepsilon = 1$, $\delta(0.5) = 0.2212$; rounded: $1.0000$, $0.1967$; threshold: $0.8318$, $0.1967$ |
| Group privacy, pure | Rounded Laplace count, counts $k$ apart: $\varepsilon = k$ for $k = 1, 2, 3, 5$ |
| Group privacy, $(0.5, 10^{-5})$ per record | @tbl-dpprop-group; the alarm mechanism attains the bound at every $k$ |
| $k$ randomized-response answers | @tbl-dpprop-rr-comp; $\varepsilon_0 = 0.1$, $k = 1000$: basic 100, exact 19.345 at $\delta = 10^{-6}$ |
| Simulation, 10 answers, 200,000 runs | Mean loss 5.500 (exact 5.493); worst case in 0.0559 of runs (exact 0.0563) |
| Adaptive two-step survey | @tbl-dpprop-adaptive; worst loss 2.1972 $= 2\ln 3$ |
| $k$ Gaussian counts, $(0.5, 10^{-5})$ each | @tbl-dpprop-gauss-comp |
| Two-bin histogram, rounded $\mathrm{Lap}(0, 1)$ | Add/remove: $\varepsilon = 1.0000$; replace across bins: $2.0000$ |
| Shared against fresh noise, 200,000 runs | Attack accuracy 1.0000 against 0.6206; best possible with fresh noise 0.6967 |

: The script's output with the default seed. {#tbl-dpprop-code}

[tests/test_properties.py](../../tests/test_properties.py) checks that the output still matches.

**What to take from it.** Post-processing never raises the profile, and sometimes lowers it: the threshold summary drops from $\varepsilon = 1$ to $0.8318$, because it averages outputs with different losses. Group privacy is tight for counts in pure DP, and tight in general for approximate DP through the alarm mechanism, though specific mechanisms like the Gaussian do better. Basic composition is exact for the worst case, and the worst case is exponentially rare. That gap, $k\varepsilon_0$ against roughly $\sqrt k\,\varepsilon_0$, is what the accountants of the later chapters recover. Parallel composition costs one budget under add/remove and two under replace, since a replacement touches two parts. And shared noise is a total failure, not a small loss: the attack's accuracy goes from 0.6206 to 1.0000.

## References

- Dwork & Roth, *The Algorithmic Foundations of Differential Privacy* (2014): Proposition 2.1 (post-processing), Theorem 2.2 (group privacy, pure), Theorem 3.16 (basic composition, with the proof for approximate DP in Appendix B), and §3.5 for composition in general. The standard reference for every statement here.
- Dwork, McSherry, Nissim & Smith, "Calibrating Noise to Sensitivity in Private Data Analysis" (TCC 2006): the paper that defined DP, where the guarantee for groups of records first appears.
- McSherry, "Privacy Integrated Queries" (SIGMOD 2009): parallel composition, and a system built on post-processing and composition as its only rules.
- Vadhan, "The Complexity of Differential Privacy" (2017), the opening sections: the basic properties of DP (post-processing, group privacy, composition) with careful proofs, at a more theoretical level than Dwork & Roth.
- Near & Abuah, *Programming Differential Privacy*, chapters on the properties of DP: sequential and parallel composition and post-processing, with code.
- Kifer & Machanavajjhala, "No Free Lunch in Data Privacy" (SIGMOD 2011): why correlated records defeat a per-record guarantee, the problem group privacy addresses.
- Dwork, Rothblum & Vadhan, "Boosting and Differential Privacy" (FOCS 2010), and Kairouz, Oh & Viswanath, "The Composition Theorem for Differential Privacy" (ICML 2015): advanced and optimal composition, the subject of [Advanced and optimal composition](04-composition.md).
- Rogers, Roth, Ullman & Vadhan, "Privacy Odometers and Filters: Pay-as-you-Go Composition" (NeurIPS 2016): what changes when the budgets themselves are chosen adaptively.

## Cheat sheet

$M$ is $(\varepsilon, \delta)$-DP: $\Pr[M(D) \in S] \le e^\varepsilon\Pr[M(D') \in S] + \delta$ for all $S$ and all $D \simeq D'$; equivalently $E_{e^\varepsilon}(M(D)\|M(D')) \le \delta$. The privacy loss is $L_{D,D'}(y) = \ln\frac{p_{M(D)}(y)}{p_{M(D')}(y)}$. @tbl-dpprop-cheat collects the results.

| Property | Statement | Proof idea | Tight? |
| --- | --- | --- | --- |
| Post-processing | $f \circ M$ is $(\varepsilon, \delta)$-DP for any $f$ that doesn't read the data | DPI: processing averages likelihood ratios; Fact 2 for $\delta > 0$ | For invertible $f$; processing that merges outputs can lower $\varepsilon$ |
| Group privacy, pure | $k\varepsilon$ for datasets $k$ records apart | Losses telescope along a path | Yes, for counts with Laplace noise |
| Group privacy, approximate | $\big(k\varepsilon,\ \delta\frac{e^{k\varepsilon} - 1}{e^\varepsilon - 1}\big)$, at most $(k\varepsilon, k e^{(k-1)\varepsilon}\delta)$ | Chain the event form $k$ times | Yes, for the alarm mechanism |
| Basic composition | $(\sum\varepsilon_i, \sum\delta_i)$, also adaptive with fixed budgets | Losses of independent outputs add; two-step lemma for $\delta > 0$ | Worst case yes; with $\delta > 0$ allowed, loose for large $k$ |
| Parallel composition | $(\max\varepsilon_i, \max\delta_i)$ under add/remove; twice that under replace | Only one part sees the changed record | Yes |

: Every result in this chapter. {#tbl-dpprop-cheat}

**Rules of thumb**

- Privatize once, as early as possible; everything after it is free.
- Budgets add across queries on the same data and take the maximum across disjoint data.
- Group privacy is cheap only for small groups and pure DP; for users with many records, define the neighbouring relation at the user level instead.
- Every composed mechanism needs its own fresh noise.
- Basic composition is exact for the worst case but the worst case is exponentially rare; when $k$ is large, use the accountants of the later chapters.

**Practice.** Exercises for this chapter, each with a worked solution, are in [the exercise book](../../book-exercises/part03-dp-foundations/02-properties.md).
