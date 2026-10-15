# Post-processing, group privacy and composition

## Why this matters

A private mechanism is never the whole system. Its output gets rounded, plotted or fed into a model, and none of those steps is part of the mechanism you proved private. The people in your data may each have contributed many records, while the guarantee speaks about one record at a time. And you rarely run only one mechanism, because an analyst asks a hundred queries and a training run takes thousands of noisy gradient steps. In each case you need to know what guarantee holds for what you actually publish.

Three properties of differential privacy (DP) answer that. Post-processing says that anything you compute from a private output, without reading the data again, is as private as the output itself. Group privacy says how a guarantee about one record extends to a group of records, at a cost that grows with the size of the group. Composition says that running several mechanisms on the same data costs at most the sum of their separate privacy costs. So you can prove each building block private once and then add up the costs, instead of writing a new proof for every pipeline.

All three follow from one quantity, the **privacy loss** of [The definition of differential privacy](01-definition.md). A mechanism $M$ is a randomized algorithm that takes a dataset and returns an output. Two datasets are neighbouring when they differ in one person's record, and this chapter calls them $D$ and $D'$. The privacy loss of an output is the logarithm of how much more likely that output is when the data are $D$ than when they are $D'$. It is positive when the output is evidence for $D$, negative when it is evidence for $D'$, and zero when it says nothing either way.

Pure DP fixes a privacy budget $\varepsilon$ and keeps the loss between $-\varepsilon$ and $\varepsilon$ for every output. The likelihood ratio itself then stays between $e^{-\varepsilon}$ and $e^\varepsilon$. A mechanism that meets this bound is called $\varepsilon$-DP. Approximate DP adds a small slack $\delta$, and such a mechanism is called $(\varepsilon, \delta)$-DP. The slack lets the ratio bound fail on rare outputs. Their probability on $D$ in excess of $e^\varepsilon$ times their probability on $D'$ must add up to at most $\delta$.

The one idea of this chapter is how the privacy loss behaves when you build systems. If you walk from one dataset to another a record at a time, or run several mechanisms one after another, the losses of the individual steps add. If you instead process an output, the losses of the raw outputs that lead to the same processed output are averaged, which can never make them larger. Each section applies this idea to one situation and names the condition its argument needs, because each property fails as soon as that condition is broken.

## Post-processing {#sec-dpprop-post}

A private output is rarely published as it is. You might round a noisy count to an integer, clip it at zero, or fit a model to a table of noisy statistics. An analyst might read it and decide what to do next. Each of these is a function of the output, which may use random coins of its own but never reads the data again. The question is whether such processing can undo the protection, and the answer is that it cannot.

The reason is that processing averages evidence. Several raw outputs can lead to the same processed output, and the probability of the processed output is the sum of their probabilities. So its likelihood ratio between $D$ and $D'$ is a weighted average of the likelihood ratios of the raw outputs that lead to it. Each raw output's weight is its share of the processed output's probability on $D'$. If each of those ratios lies between $e^{-\varepsilon}$ and $e^\varepsilon$, their average does too. This is the **data processing inequality** of [Divergences](../part01-prerequisites/03-divergences.md), which says that pooling evidence can only dilute it.

With a slack $\delta$, the argument works on events instead of single outputs. Write $f$ for the processing, and take any event about the processed output, such as "the rounded count is 5". Each raw output leads to that event with some probability, and this probability is the same on $D$ and on $D'$. That holds because $f$ never reads the data. Its coins are also independent of the data and of the mechanism's own coins, so nothing about the dataset can leak into them. The event's probability is then an average of these per-output probabilities. The definition's bound holds not only for plain events but also for such averages of probabilities between 0 and 1 (the full edition proves this). So it holds for the processed event too. If $M$ is $(\varepsilon, \delta)$-DP, then so is the processed mechanism:
$$f(M(D)) \text{ is } (\varepsilon, \delta)\text{-DP}.$$ {#eq-dpprop-post}

More than that, the whole **privacy curve** of the processed mechanism lies on or below the curve of $M$. The privacy curve, written $\delta(\varepsilon)$, gives the smallest $\delta$ that works at each $\varepsilon$ ([The definition of differential privacy](01-definition.md)).

A small example shows the averaging at work. Randomized response reports a person's true bit with probability $\frac34$, and the other bit otherwise ([The definition of differential privacy](01-definition.md)). So each report is 3 times more likely under one value of the true bit than under the other. The mechanism is therefore DP with a per-answer budget $\varepsilon_0$ of $\ln 3$. Now suppose that before publishing, you flip each answer with probability $\frac14$, using a fresh coin. A published 1 can arise in two ways. Either the first answer was 1 and was kept, or it was 0 and was flipped. @tbl-dpprop-rr-flip follows both paths, multiplying the probabilities of their two steps.

| Path to a published 1 | Probability when the bit is 1 ($D$) | Probability when the bit is 0 ($D'$) | Likelihood ratio of the first answer |
| --- | --- | --- | --- |
| First answer 1, kept | $\frac34 \cdot \frac34 = \frac9{16}$ | $\frac14 \cdot \frac34 = \frac3{16}$ | 3 |
| First answer 0, flipped | $\frac14 \cdot \frac14 = \frac1{16}$ | $\frac34 \cdot \frac14 = \frac3{16}$ | $\frac13$ |
| Total | $\frac{10}{16} = 0.625$ | $\frac6{16} = 0.375$ | — |

: Randomized response at $\varepsilon_0 = \ln 3$, then a fresh coin that flips the answer with probability $\frac14$. A published 1 is reached by two paths. {#tbl-dpprop-rr-flip}

The published 1 has likelihood ratio $0.625/0.375 = \frac53$. That is the plain average of 3 and $\frac13$, because each path contributes $\frac3{16}$ of the total $\frac6{16}$ when the bit is 0, so each carries half the weight. A published 0 gets the inverse ratio, $\frac35$. So the published answers are $\ln\frac53$-DP, which is 0.5108, well below the original 1.0986 of $\ln 3$. The coin lowered the loss because it mixed outputs that pointed in opposite directions. The right panel of @fig-dpprop-post repeats the calculation for every flip probability. The loss falls to 0 when the flip probability is $\frac12$, since the published bit is then a fair coin whatever the first answer was.

Processing that you do for utility rarely mixes outputs this much. Take a count whose value is 0 on one dataset and 1 on its neighbour. Add Laplace noise of scale 1, whose density falls by a factor $e$ for each unit of distance from the true count. The noisy count is then 1-DP ([The Laplace and Gaussian mechanisms](03-laplace-gaussian.md)). Now process it in two ways. You can round it to the nearest integer, or you can publish only whether it exceeds $\frac12$. @tbl-dpprop-laplace-post gives the exact privacy curves of the three releases at two values of $\varepsilon$, computed by the full edition's script.

| Published | Smallest pure $\varepsilon$ | $\delta(0.25)$ | $\delta(0.5)$ |
| --- | --- | --- | --- |
| Noisy count | 1.0000 | 0.3127 | 0.2212 |
| Rounded to an integer | 1.0000 | 0.3073 | 0.1967 |
| Only whether it exceeds $\frac12$ | 0.8318 | 0.3073 | 0.1967 |

: Exact privacy curves of a count with Laplace noise of scale 1, before and after two kinds of processing. Each row is on or below the one above it. {#tbl-dpprop-laplace-post}

![Left: the privacy curve $\delta(\varepsilon)$ of a count with Laplace noise of scale 1 (black), after rounding to an integer (blue), and after keeping only whether it exceeds $\frac12$ (orange); processing can only lower the curve. Right: randomized response at $\varepsilon_0 = \ln 3$ followed by a fresh coin that flips the answer with the probability on the horizontal axis; the dot marks a flip probability of $\frac14$, where $\varepsilon = \ln\frac53$.](figures/dpprop_postprocessing.svg){#fig-dpprop-post}

Rounding sends every output in an interval of length 1 around an integer, its rounding cell, to that integer. It keeps the pure $\varepsilon$ at 1.0000, because of the cells that lie entirely below 0. Every output in such a cell is exactly 1 further from the count 1 than from the count 0, so every output in it has the same loss of 1. An average of equal ratios is that same ratio, so nothing is averaged away. Rounding still lowers the curve at smaller $\varepsilon$, because the cell around 0 mixes outputs whose losses differ, from 0 up to 1. Averaged together, they give the cell a loss of 0.7191, and the curve drops at values of $\varepsilon$ below that.

The one-bit summary is the most private of the three. Its "yes" has probability 0.3033 when the count is 0. When the count is 1, it has the rest of the probability, 0.6967. The logarithm of their ratio is 0.8318, and the "no" answer gives the same loss with the opposite sign.

The general rule follows from these cases. Processing that maps each published output back to a single raw one, such as relabelling or a change of units, keeps the loss exactly, because nothing is averaged. Processing that merges outputs with different losses lowers it. Since the processing you do for utility merges only a little, report the guarantee of $M$ and treat any gain from processing as a bonus you don't count on.

The first consequence for practice is that nothing downstream needs a new proof. A model trained on DP statistics is a function of the private output that never reads the data again. So are synthetic data sampled from a DP model, a plot and a dashboard, and they all inherit the guarantee. This is why a DP system adds noise at one narrow point, such as noisy counts or noisy gradients, and treats everything after it as public.

The same argument covers attackers. A membership-inference attacker ([Membership inference](../part02-attacks/02-membership-inference.md)) looks at the output and returns a guess, possibly using outside knowledge and its own randomness. That guess is a function of the output, so it is a post-processing. The definition bounds every attack's true positive rate by $e^\varepsilon$ times its false positive rate plus $\delta$ ([The definition of differential privacy](01-definition.md)). So this bound holds even for attacks invented after the release. In the language of [Hypothesis testing](../part01-prerequisites/04-hypothesis-testing.md), processing can only raise the **trade-off curve**, the smallest false negative rate a test can reach at each false positive rate. That is because any test on the processed output is also a test on the raw output.

The argument has one condition. The averaging used the same processing on $D$ and on $D'$, which holds only if the processing never reads the data. "Take the noisy top category, then report its exact count" breaks it, because the second step reads the data. That step is a new mechanism, and a non-private one. The same goes for an analyst who notices that one bin of a DP histogram looks odd and publishes a second histogram with finer bins. Deciding to publish is post-processing, but computing the finer histogram reads the data again, so it is a second mechanism whose cost adds by composition.

## Group privacy {#sec-dpprop-group}

The definition protects one record, but people are not rows. A user of a messaging app contributes hundreds of messages, a household shares an address and an income, and a disease runs in a family. Suppose the guarantee is stated per record, and a user owns $k$ records. To bound what an adversary learns about all of them at once, you need a guarantee for datasets that differ in $k$ records. Group privacy derives it from the one-record guarantee, at a price.

The idea is to walk from $D$ to a dataset $D''$ that differs from it in $k$ records, changing one record per step. Each step joins two neighbouring datasets, so the one-record guarantee applies to it. For pure DP, the likelihood ratio between the two ends of the path is the product of the ratios along the way. That is because every dataset in the middle appears once in a numerator and once in a denominator, and cancels. Taking the logarithm turns the product into a sum, so the privacy loss between the ends is the sum of the $k$ step losses. Each step loss is at most $\varepsilon$ in size, so the total is at most $k\varepsilon$ in size. Pure DP at $\varepsilon$ per record therefore becomes pure DP at $k\varepsilon$ for a group of $k$ records.

A noisy count shows the steps adding and cancelling. Take a count with Laplace noise of scale 1, and walk from a true count of 0 up to 3, one step at a time. At the output $-2$, which lies below every count on the path, each of the three steps contributes a loss of 1, so the sum reaches 3. At the output 1.5, in the middle of the path, the steps point in different directions and cancel to 0. For counts, the full edition checks that this bound is met exactly, so it cannot be improved.

With approximate DP there is nothing to add up, because the slack $\delta$ lets rare outputs have a loss of any size. Instead you chain the event form of the definition. The first step bounds an event's probability on $D$ by $e^\varepsilon$ times its probability on the next dataset, plus $\delta$. That probability is in turn bounded through the following step, and so on along the path. The factor grows to $e^{k\varepsilon}$, but the slack grows faster. Each step adds its own $\delta$, and every later step then multiplies everything accumulated so far by $e^\varepsilon$ again, including the earlier $\delta$s. So the $\delta$s pile up in a geometric sum. For datasets $k$ records apart, an $(\varepsilon, \delta)$-DP mechanism is
$$\Big(k\varepsilon,\ \delta_k\Big)\text{-DP}, \qquad \delta_k = \delta\sum_{j=0}^{k-1}e^{j\varepsilon} = \delta\,\frac{e^{k\varepsilon} - 1}{e^\varepsilon - 1} \le k\,e^{(k-1)\varepsilon}\delta.$$ {#eq-dpprop-group}

The simpler form on the right replaces every term of the sum by the largest one, so it is looser. The sum itself cannot be improved in general. The full edition builds a mechanism that raises an "alarm" with a probability that meets every step's bound with equality, so it attains the sum exactly.

The price is steep in both cases. Pure DP degrades in proportion to the group size. A user with 100 records who is protected at $0.1$ per record is protected only at $10$. That guarantee protects almost nothing, because a likelihood ratio of up to $e^{10} \approx 22{,}000$ lets an adversary be nearly certain. This is the formal version of the warning in [The definition of differential privacy](01-definition.md) that protecting one row does not protect a person who appears in many rows.

With approximate DP, the factor on the slack grows exponentially with the group size. Take a larger budget of $\varepsilon = 1$ per record. The geometric sum for a group of 10 records then multiplies $\delta$ by 12818.3. That hurts because $\delta$ must already be much smaller than one over the number of records. Consider a mechanism that publishes each record in full with probability $\delta$. It is still $(0, \delta)$-DP, yet on average it leaks $\delta$ times the number of records ([The definition of differential privacy](01-definition.md)). Multiplying such a $\delta$ by thousands often leaves no meaningful guarantee for large groups.

A particular mechanism can do better than the general bound. For a count with Gaussian noise, a group of $k$ records only shifts the count by $k$ instead of by 1. Its exact guarantee for the group is that of two Gaussian distributions whose means are $k$ apart, which [Divergences](../part01-prerequisites/03-divergences.md) computes in closed form. @tbl-dpprop-group starts from a per-record guarantee of $(0.5, 10^{-5})$ and compares the bound with this exact $\delta$. The Gaussian noise has standard deviation 7.0318, chosen so that one record costs exactly that guarantee.

| Group size $k$ | $k\varepsilon$ | $\delta_k$, the geometric sum | Simple form $k e^{(k-1)\varepsilon}\delta$ | Gaussian count, exact |
| --- | --- | --- | --- | --- |
| 1 | 0.5 | $1.000 \times 10^{-5}$ | $1.000 \times 10^{-5}$ | $1.000 \times 10^{-5}$ |
| 2 | 1 | $2.649 \times 10^{-5}$ | $3.297 \times 10^{-5}$ | $2.551 \times 10^{-5}$ |
| 3 | 1.5 | $5.367 \times 10^{-5}$ | $8.155 \times 10^{-5}$ | $4.858 \times 10^{-5}$ |
| 5 | 2.5 | $1.724 \times 10^{-4}$ | $3.695 \times 10^{-4}$ | $1.288 \times 10^{-4}$ |
| 10 | 5 | $2.272 \times 10^{-3}$ | $9.002 \times 10^{-3}$ | $7.592 \times 10^{-4}$ |

: Group privacy from a $(0.5, 10^{-5})$ guarantee per record: the geometric-sum bound, which the alarm mechanism attains at every $k$, its simple form, and the exact $\delta$ of a Gaussian count, all at $\varepsilon = k \times 0.5$. {#tbl-dpprop-group}

![The $\delta$ of group privacy at $\varepsilon = k \times 0.5$, for a per-record guarantee of $(0.5, 10^{-5})$, on a log scale. The geometric-sum bound (dashed) is attained by the alarm mechanism (red dots); the simple form (dotted) is looser by a factor that grows with $k$; Gaussian noise on a count (blue squares) does better than the bound but still grows quickly with $k$.](figures/dpprop_group.svg){#fig-dpprop-group}

The simple form is looser than the sum by a factor that grows with the group, from 1.24 for two records to 3.96 for ten, so use the sum. The Gaussian count stays a factor of 2.99 below the bound for a group of 10, and the gap widens with $k$ (@fig-dpprop-group). Even so, a user with 10 records gets only $(5, 7.592 \times 10^{-4})$ from the Gaussian count, which is a weak guarantee.

So if you care about users, don't patch a per-record guarantee with group privacy. Define neighbours as datasets that differ in one user's records, and bound each user's contribution, for example by capping how many records a user may add. Then prove DP at that level directly. Group privacy is the fallback when you cannot.

## Basic sequential composition {#sec-dpprop-comp}

An analyst asks several queries of the same dataset, each answered by a DP mechanism with fresh noise. In the same way, a training run takes many noisy gradient steps, and a survey may ask the same question twice. Each output alone is private, but an adversary sees all of them together, as one **transcript**. So the question is what guarantee holds for the whole transcript.

The answer comes from how independent outputs combine. When two outputs use independent coins, the probability of seeing both is the product of their probabilities. The logarithm turns that product into a sum, so the privacy loss of the pair is the loss of the first output plus the loss of the second. Two randomized-response answers at $\ln 3$ from one person show both ways this can go. Two answers of 1 are 9 times more likely when the person's bit is 1 than when it is 0. Their loss is therefore $\ln 9$, which is $\ln 3$ twice, two pieces of evidence pointing the same way. An answer of 1 followed by an answer of 0 is equally likely under both bits, so its loss is 0, and the two pieces of evidence cancel.

For pure DP, each term of the sum is at most its mechanism's budget in size, so the transcript's loss is at most their total. With a slack the losses are not bounded, but the full edition proves the same rule through events, and the slacks simply add. Take several mechanisms, and write $M_i$ for the $i$-th of them and $(\varepsilon_i, \delta_i)$ for its budget. If there are $k$ of them, releasing all their outputs is
$$\Big(\sum_{i=1}^k\varepsilon_i,\ \sum_{i=1}^k\delta_i\Big)\text{-DP}.$$ {#eq-dpprop-basic}
This is **basic composition**. Unlike group privacy, the $\delta$s add plainly, without any factor of $e^\varepsilon$. Every mechanism here compares the same pair of datasets, so there is no chain of steps to multiply them through, and each slack is counted once. For the same reason, every mechanism must be private under the same neighbouring relation, or the losses being added would compare different pairs.

The rule also covers **adaptive** composition, where each mechanism may depend on the earlier outputs, the way an analyst picks the next query after reading the last answer. The probability of a transcript is then the probability of the first output, times that of the second given the first, and so on. The losses still add, with each step's loss computed as if the earlier outputs were fixed at the values they took. So the condition is that each $M_i$ is $(\varepsilon_i, \delta_i)$-DP for every possible value of the earlier outputs. For example, ask one person a randomized-response question at $\ln 3$. If the answer is 1, ask again at the same level. If it is 0, ask again at the smaller level 0.5. Either way the second step is at most $\ln 3$-DP, and the worst transcript has a loss of exactly $2\ln 3$, which is the bound.

The rule needs the budgets $(\varepsilon_i, \delta_i)$ to be fixed in advance, even when the queries are chosen adaptively. Suppose an analysis runs ten queries at $\varepsilon = 0.1$ each, and an eleventh only if the first ten look interesting. It still costs at most 1.1, provided the eleventh query's budget was fixed beforehand. That is because deciding whether to run it is adaptive composition with a mechanism that may output nothing. If the budgets themselves are chosen from the answers, as in "keep going until the estimate looks good", @eq-dpprop-basic does not cover it in general. That case needs privacy filters, rules that decide when to stop as the budget is spent, which [Adaptive composition](../part04-privacy-accounting/08-adaptive-composition.md#sec-adapt-dp-filters) covers.

Pure DP with basic composition is the exception to that need. Each step's loss is at most its own $\varepsilon_i$ at every output, not just on typical ones. So fix a total in advance and stop before the budgets spent so far exceed it. The whole run is then pure DP at that total, even when each budget was chosen from earlier answers ([Advanced and optimal composition](04-composition.md)).

Basic composition is the exact worst case, but the worst case is rare. Ask one person the same yes/no question $k$ times, each time with fresh randomized response at the per-answer budget $\varepsilon_0$, equal to $\ln 3$. The transcript in which every answer is truthful has loss exactly $k\ln 3$, because each answer contributes $+\ln 3$, so no smaller pure $\varepsilon$ holds. But that transcript has probability $(\frac34)^k$. Typical transcripts mix truthful answers, which add $\varepsilon_0$ to the loss, with lies, which subtract it, and the two partly cancel. Write $t$ for the probability that an answer is truthful. Each answer then adds $\varepsilon_0$ with probability $t$ and subtracts it otherwise, so its mean loss is $\varepsilon_0(2t - 1)$. For randomized response this works out to $\varepsilon_0\tanh(\varepsilon_0/2)$. At the budget $\ln 3$, where $t$ is $\frac34$, it is half of $\varepsilon_0$, so the mean loss of the transcript is half the bound (@fig-dpprop-rr-loss).

![Distribution of the privacy loss of 10 randomized-response answers at $\varepsilon_0 = \ln 3$ each, when the true bit is 1 (blue) or 0 (orange). The basic bound $10\ln 3$ is reached, but only with probability $(3/4)^{10} = 0.056$; the mean when the true bit is 1 is half of it.](figures/dpprop_rr_loss.svg){#fig-dpprop-rr-loss}

The loss of $k$ answers is a sum of $k$ independent terms, each equal to plus or minus $\varepsilon_0$. Because the terms are bounded and independent, the sum stays near its mean ([Concentration inequalities](../part01-prerequisites/02-concentration.md)). Its spread is of order $\varepsilon_0\sqrt k$, and deviations much larger than that are exponentially unlikely. Allowing a small $\delta$ lets you write off that unlikely tail. The $\varepsilon$ you then need is the mean loss plus a margin that grows like $\sqrt k$.

The mean still grows like $k$, but only by $\varepsilon_0\tanh(\varepsilon_0/2)$ per answer. That is about $\varepsilon_0^2/2$ when $\varepsilon_0$ is small. So for small budgets per answer the margin dominates, and $\varepsilon$ grows roughly like $\sqrt k$ instead of $k$. [Advanced and optimal composition](04-composition.md) turns this into a rule called advanced composition. @tbl-dpprop-rr-comp shows when the slack starts to help at the budget $\ln 3$.

| $k$ | Basic bound $k\varepsilon_0$ | Probability of the worst case, $(\frac34)^k$ | Mean loss | Exact $\varepsilon$ at $\delta = 10^{-6}$ |
| --- | --- | --- | --- | --- |
| 1 | 1.099 | 0.75 | 0.549 | 1.099 |
| 10 | 10.986 | $5.631 \times 10^{-2}$ | 5.493 | 10.986 |
| 20 | 21.972 | $3.171 \times 10^{-3}$ | 10.986 | 21.972 |
| 50 | 54.931 | $5.663 \times 10^{-7}$ | 27.465 | 52.680 |
| 100 | 109.861 | $3.207 \times 10^{-13}$ | 54.931 | 94.290 |

: $k$ randomized-response answers about the same bit at $\varepsilon_0 = \ln 3$ each. The basic bound is also the exact worst-case loss. The mean loss is computed when the true bit is 1. {#tbl-dpprop-rr-comp}

A slack $\delta$ helps only if the transcripts with large loss have a total probability below $\delta$, so that they can be written off. Up to 20 answers, the worst transcript is far more likely than $\delta$, so the slack buys nothing visible. Only from 50 answers, where its probability falls below $\delta$, does the exact $\varepsilon$ drop below the bound. At this large budget per answer the mean is half the bound, so the exact $\varepsilon$ still grows almost in proportion to $k$.

The gain is much larger with a smaller budget per answer, because the mean is then small and the $\sqrt k$ margin dominates. Take 1000 answers at a budget of 0.1 each. Basic composition gives an $\varepsilon$ of 100. With the same slack of $10^{-6}$, the exact $\varepsilon$ is only 19.345 (@fig-dpprop-composition).

![The $\varepsilon$ of $k$ randomized-response answers: the basic bound $k\varepsilon_0$, which is also the exact worst-case loss (dashed), the exact $\varepsilon$ at $\delta = 10^{-6}$ and $10^{-3}$, and the mean privacy loss (dotted). Left: $\varepsilon_0 = \ln 3$; the exact curves leave the basic line only once $(3/4)^k$ drops below $\delta$. Right: $\varepsilon_0 = 0.1$; the exact curves grow like $\sqrt k$.](figures/dpprop_composition.svg){#fig-dpprop-composition}

The same gap appears for mechanisms that need a slack. Take counts with Gaussian noise of standard deviation 7.0318, each calibrated to $(0.5, 10^{-5})$. Changing one record shifts each of the $k$ noisy counts by 1. Gaussian noise with independent coordinates of equal variance looks the same in every direction, so you can rotate the axes until the combined shift points along the first one. Its length is then $\sqrt k$, and the other coordinates have the same distribution on both datasets, so they carry no evidence. The exact guarantee of $k$ such counts is therefore that of one Gaussian pair with $\sqrt k$ times the shift, which @tbl-dpprop-gauss-comp compares with basic composition at the same total $\delta$.

| $k$ | Basic: $(k\varepsilon, k\delta)$ | Exact $\varepsilon$ at the same $\delta = k \times 10^{-5}$ |
| --- | --- | --- |
| 1 | $(0.5, 10^{-5})$ | 0.500 |
| 2 | $(1, 2 \times 10^{-5})$ | 0.694 |
| 4 | $(2, 4 \times 10^{-5})$ | 0.966 |
| 10 | $(5, 10^{-4})$ | 1.504 |

: $k$ counts with Gaussian noise, each $(0.5, 10^{-5})$-DP: basic composition against the exact $\varepsilon$ at the same total $\delta$. {#tbl-dpprop-gauss-comp}

So basic composition suits a handful of releases. For ten Gaussian counts it already overstates $\varepsilon$ by a factor of 3.3. For thousands of training steps it is useless, and you need the accountants of [Advanced and optimal composition](04-composition.md) and [Part 4](../part04-privacy-accounting/00-introduction.md), which exploit exactly this cancellation.

## Parallel composition {#sec-dpprop-parallel}

A histogram counts people in disjoint bins, per-region statistics use disjoint subsets of the data, and in local randomized response each person randomizes only their own bit. Sequential composition would charge the sum of the budgets, one per bin, which is far too pessimistic. One person's record lands in one bin only, so only one output depends on it. Parallel composition makes that precise.

It depends on the neighbouring relation, so recall the two common ones from [The definition of differential privacy](01-definition.md). Under **add/remove**, neighbouring datasets differ by one record added or taken away. Under **replace**, they have the same size and differ in the value of one record, which amounts to a removal followed by an addition.

Under add/remove, an added record falls into exactly one part of the data. The other parts get identical inputs on $D$ and $D'$, so their outputs have the same distribution in both worlds and contribute zero privacy loss. Only the part that changed costs anything, and you pay its budget alone. The same holds with a slack, which the full edition proves by fixing the outputs of the unchanged parts.

The result needs two conditions. First, the parts must split the set of all possible records by a rule fixed in advance. The rule may look at a record's own content, but not at the rest of the dataset. Second, each mechanism must use independent coins, so that the outputs of the unchanged parts carry no evidence about the changed one. So split the possible records into disjoint parts, and run the mechanism $M_i$ on the records that fall in the $i$-th part. If each $M_i$ is $(\varepsilon_i, \delta_i)$-DP under add/remove, the whole release is
$$\big(\max_i\varepsilon_i,\ \max_i\delta_i\big)\text{-DP under add/remove}.$$ {#eq-dpprop-parallel}

This is why a histogram costs one $\varepsilon$, not one per bin. Put Laplace noise of scale $1/\varepsilon$ on each bin. Adding a record changes one bin by 1, and the noise on that bin covers it, so the histogram is $\varepsilon$-DP under add/remove however many bins there are. For two bins with rounded Laplace noise of scale 1, the full edition's script finds an exact pure $\varepsilon$ of 1.0000 under add/remove, the budget of a single bin.

Under replace, a record can move from one part to another, and then two parts see a difference. Moving a record from part $a$ to part $b$ is a composition of two mechanisms, so it costs $(\varepsilon_a + \varepsilon_b, \delta_a + \delta_b)$. Replacing a record within one part is two add/remove steps inside that part, which costs at most its guarantee for a group of two (@eq-dpprop-group). For pure DP with equal budgets, both cases give $2\varepsilon$. In the two-bin histogram, moving a record from one bin to the other raises one count and lowers the other, and the exact $\varepsilon$ is 2.0000. That is why a histogram's sensitivity is often quoted as 2 under replace ([The Laplace and Gaussian mechanisms](03-laplace-gaussian.md)).

Under replace, the cost goes back to the maximum when each part is a fixed slot that belongs to one person. Replacing that person's data then changes only their own slot. Local randomized response is the main case, since each person's report depends only on their own bit. Replacing one person's bit changes only their own report, so the whole vector of reports is $\ln 3$-DP however many people answer, not $\ln 3$ times their number.

The rule fixed in advance is the condition that is easiest to break. For example, a split into "the top 10% of earners and the rest" moves its boundary when one high earner is added. Records then cross the boundary, so both parts change and more than one mechanism sees a difference. Use fixed bins, or account for a data-dependent split as a mechanism of its own.

## What composes and what doesn't {#sec-dpprop-limits}

The three properties hold under weak assumptions, which makes it tempting to apply them even where those assumptions fail. Post-processing needs the processing to ignore the data (@sec-dpprop-post). Composition needs independent coins, the same neighbouring relation for every step, and budgets fixed in advance. Most real failures break one of these, and the damage is usually total rather than a small loss.

Shared noise is the clearest case. Release the number of smokers and the number of smokers aged 40 or over, adding the same draw of Laplace noise of scale 1 to both. Each release alone is 1-DP. But subtracting one release from the other cancels the noise and leaves the exact number of smokers under 40. Alice is a smoker aged 30, so removing her lowers that difference by one. So the event "the difference equals its value with Alice in the data" is certain when she is in and impossible when she is out. No $\varepsilon$ can bound that unless $\delta$ is at least 1, so the pair of releases is not DP at all. Composition assumed that the transcript's probability is a product of independent parts, and one shared draw breaks that.

The full edition simulates this attack, with Alice in the data with probability $\frac12$. The attacker guesses that she is in when the difference of the two releases exceeds the midpoint between its noise-free values with and without her. Over 200,000 runs with shared noise, the attacker is right every time, an accuracy of 1.0000. With fresh noise for each count, the same attack reaches 0.6206. No attack on the fresh-noise release can beat 0.6967. That ceiling comes from the **total variation distance** between what the attacker sees with and without Alice. It is the largest difference in probability that the two cases give to any one event ([Divergences](../part01-prerequisites/03-divergences.md)), and the best accuracy is one half plus half that distance.

In code, the shared-noise bug hides in two mechanisms seeded with the same random seed. It also hides in a cached noise value re-added to a related query, and in noise reused across the shards of a distributed system. Returning the same noisy answer to a repeated identical query is fine, because that is post-processing of one release. The bug is reusing a noise draw for a different query.

An exact statistic published elsewhere is the same mistake in another form. If the exact total of a column is public, a DP release of a related quantity can be differenced against it, just as the two counts were. The DP guarantee covers only what your mechanism released, and adds nothing to a release that is not DP.

Finally, the neighbouring relation is part of the budget. A replacement is a removal plus an addition, so a guarantee proved under add/remove is a group guarantee for two records under replace. By @eq-dpprop-group with a group of two, its $\varepsilon$ doubles and its $\delta$ grows to $(1 + e^\varepsilon)\delta$. Mixing the two relations inside one accountant is therefore a bug, and it comes back with subsampling in [Subsampling](../part04-privacy-accounting/05-subsampling.md).

## Code {#sec-dpprop-code}

Group privacy is where hand calculations go wrong most often, because $\delta$ does not simply scale with the group size. This snippet computes the three budget rules of this chapter:

```python
from math import exp


def group_privacy(eps: float, delta: float, k: int) -> tuple[float, float]:
    """The guarantee for a group of k records, from an (eps, delta) guarantee per record."""
    delta_k = delta * sum(exp(j * eps) for j in range(k))
    return k * eps, delta_k


def basic_composition(budgets: list[tuple[float, float]]) -> tuple[float, float]:
    """The guarantee of several mechanisms run on the same data with fresh noise."""
    return sum(e for e, _ in budgets), sum(d for _, d in budgets)


def parallel_composition(budgets: list[tuple[float, float]]) -> tuple[float, float]:
    """The guarantee of mechanisms run on disjoint parts of the data, under add/remove."""
    return max(e for e, _ in budgets), max(d for _, d in budgets)


eps_k, delta_k = group_privacy(0.5, 1e-5, 10)
print(f"{eps_k:.1f} {delta_k:.3e}")  # 5.0 2.272e-03
print(f"{group_privacy(0.5, 1e-5, 2)[1]:.3e}")  # 2.649e-05
print(f"{group_privacy(1.0, 1.0, 10)[1]:.1f}")  # 12818.3
print(basic_composition([(0.5, 1e-5)] * 10))  # (5.0, 0.0001)
print(parallel_composition([(0.5, 0.0)] * 50))  # (0.5, 0.0)
```

The third printed line is the factor by which $\delta$ grows for a group of 10 at $\varepsilon = 1$. The full edition's script checks every number in this chapter by exact computation and simulation:

```bash
uv run python code/part03-dp-foundations/properties.py
```

## Cheat sheet

| Property | Rule | Condition and notes |
| --- | --- | --- |
| Post-processing | $f(M(D))$ is $(\varepsilon, \delta)$-DP | $f$ never reads the data and uses independent coins; it can lower $\varepsilon$, never raise it |
| Group privacy, pure | $k\varepsilon$ for datasets $k$ records apart | Tight for a count with Laplace noise |
| Group privacy, approximate | $\big(k\varepsilon,\ \delta\frac{e^{k\varepsilon} - 1}{e^\varepsilon - 1}\big)$, at most $(k\varepsilon, k e^{(k-1)\varepsilon}\delta)$ | Attained by some mechanisms; prefer user-level neighbours |
| Basic composition | $(\sum\varepsilon_i, \sum\delta_i)$ | Independent coins, one neighbouring relation; also adaptive, with budgets fixed in advance; exact worst case, loose for many releases |
| Parallel composition | $(\max\varepsilon_i, \max\delta_i)$ under add/remove | Partition fixed in advance; under replace, up to $2\varepsilon$ and $(1 + e^\varepsilon)\delta$, unless the parts are fixed slots |
| Add/remove guarantee under replace | $(2\varepsilon, (1 + e^\varepsilon)\delta)$ | Group privacy with $k = 2$ |
| Shared noise, or an exact release beside a DP one | No guarantee at all | Every mechanism needs fresh noise |

: The results of this chapter. {#tbl-dpprop-cheat}

**Rules of thumb**

- Add noise once, as early as possible, and everything computed from the noisy output after that is free.
- Budgets add across queries on the same data, and take the maximum across disjoint parts of the data.
- For users with many records, define neighbours at the user level instead of relying on group privacy.
- Use basic composition for a handful of releases, and an accountant when there are many.

## Further reading

- Dwork & Roth, *The Algorithmic Foundations of Differential Privacy* (2014): post-processing, group privacy and basic composition (Proposition 2.1, Theorems 2.2 and 3.16), the standard reference.
- McSherry, "Privacy Integrated Queries" (SIGMOD 2009): parallel composition, and a query system built on post-processing and composition as its only rules.
- Near & Abuah, *Programming Differential Privacy*, the chapters on the properties of DP: sequential and parallel composition and post-processing, with code.
- Rogers, Roth, Ullman & Vadhan, "Privacy Odometers and Filters: Pay-as-you-Go Composition" (NeurIPS 2016): what changes when the budgets themselves are chosen adaptively.

**Derivations.** Every result in this chapter is derived step by step in [the full edition's chapter](../../book-full/part03-dp-foundations/02-properties.md), with three exceptions. The growth like $\sqrt k$ under a small $\delta$ is derived in [Advanced and optimal composition](../../book-full/part03-dp-foundations/04-composition.md), which also explains why pure DP with basic composition tolerates budgets chosen on the fly. Privacy filters, including that simple one, are derived in [Adaptive composition](../../book-full/part04-privacy-accounting/08-adaptive-composition.md#sec-adapt-dp-filters).
