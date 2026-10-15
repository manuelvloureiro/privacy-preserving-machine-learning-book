# Beyond global sensitivity

## Why this matters

The [Laplace and Gaussian mechanisms](03-laplace-gaussian.md) answer a numeric query by adding random noise to the true answer. They set the size of that noise by the **global sensitivity**, the most that changing one person's record can move the answer. Two datasets that differ in one record are called neighbours. The global sensitivity takes its maximum over every pair of neighbours that could exist, not just over the dataset you hold. For counts this is the right scale, because one person changes a count by at most 1 whatever the data.

Recall what differential privacy (DP) asks for ([The definition of differential privacy](01-definition.md)). It has a privacy parameter $\varepsilon$ and a small slack $\delta$. Write $D$ and $D'$ for two neighbouring datasets, and $E$ for any set of possible outputs, an event. A randomized mechanism $M$ satisfies $(\varepsilon, \delta)$-DP when every such choice obeys
$$\Pr[M(D) \in E] \le e^\varepsilon\Pr[M(D') \in E] + \delta.$$
In words, one record can change the probability of any outcome by at most a factor $e^\varepsilon$, up to the slack $\delta$. With $\delta = 0$ the guarantee is called pure $\varepsilon$-DP. Throughout this chapter, datasets have a fixed size $n$, and two datasets are neighbours when one is obtained from the other by **replacing** one record.

For many statistics it fails, in three different ways. First, the worst case can be absurd. Take the median of ages between 0 and 100, and imagine a dataset in which half the people are 0 years old and the other half are 100. With one more person at 0, the median is 0. Replacing that one person by a 100-year-old moves the median to 100, so the global sensitivity of the median is the whole range. The Laplace mechanism must then add noise on the scale of 100 years, although no real dataset of ages looks like this.

Second, the worst case can be infinite. Incomes and gradient norms have no natural upper limit, so one record can move their mean as far as it likes, and no Laplace noise makes it private. Third, the worst case can be unknown. For an estimator fitted by an optimizer, such as the parameters of a trained model, nobody knows how to bound how far one record moves it.

Yet on the data you actually hold, the picture is often much better. In a sample of 1,001 ages, changing one person moves the median by only about a tenth of a year. That is because, with that many people, the ages near the middle are packed tightly together. The most that one record can move the answer, starting from the dataset you hold, is the **local sensitivity**. Here it is hundreds of times smaller than the global one, so the question is whether you can add noise on that scale and still satisfy DP.

The answer is yes, if you respect one idea: **the noise scale is itself a function of the data, so it must be private too.** Otherwise the size of the noise reveals something about the data, and that alone breaks the guarantee. The chapter repairs this in two ways. Smooth sensitivity makes the noise scale change only slowly between neighbours, and propose-test-release checks privately whether a fixed scale is safe before using it. Subsample-and-aggregate and clipping instead make the sensitivity small or bounded by construction.

## Local sensitivity, and why it leaks {#sec-sens-local}

Global sensitivity takes its maximum over all pairs of neighbours, including pathological ones that look nothing like your data. The natural fix is to take the maximum only over the neighbours of the dataset you hold. To say this precisely, the **Hamming distance** $d(D, D')$ counts the records in which two datasets differ, so neighbours are datasets at distance 1. The local sensitivity of a query $f$ at a dataset $D$ is the most that one record can move the answer, starting from that dataset itself:
$$\mathrm{LS}_f(D) = \max_{D':\ d(D, D') = 1}|f(D) - f(D')|.$$
The global sensitivity, written $\Delta f$, is the largest local sensitivity over all datasets, so the local sensitivity is never larger than the global one.

Take the median of values that lie between 0 and a top value $\Lambda$. Sort the values of $D$ from smallest to largest, as $x_1 \le x_2 \le \dots \le x_n$, which are called the **order statistics**. With $n$ odd, the median is the middle value $x_m$, at rank $m = (n+1)/2$. Replacing one record can move each sorted value by at most one rank, because it removes one value and inserts one. So the median of a neighbour lies between $x_{m-1}$ and $x_{m+1}$, and the local sensitivity of the median is the larger of the two gaps next to it:
$$\mathrm{LS}_{\mathrm{med}}(D) = \max\big(x_m - x_{m-1},\ x_{m+1} - x_m\big).$$ {#eq-sens-median-ls}
Both gaps are reached, by moving the median value to $\Lambda$ or to 0. A dataset with $m$ values at 0 and the rest at $\Lambda$ reaches the global sensitivity $\Lambda$, because there the gap above the median spans the whole range.

Now consider adding Laplace noise scaled to the local sensitivity of the data, releasing $f(D) + \mathrm{Lap}(\mathrm{LS}_f(D)/\varepsilon)$. Its noise scale is a function of the data, and that function can itself be very sensitive. Nissim, Raskhodnikova and Smith (2007) give a stark example with five ages and $\Lambda = 100$. The dataset $D = (0, 0, 0, 0, 100)$ has median 0 with no gap on either side, so the mechanism adds no noise and outputs exactly 0. Its neighbour $D' = (0, 0, 0, 100, 100)$ has the same median, but the gap above it now spans the whole range, so its output is spread out and is never exactly 0. The output 0 therefore identifies $D$ with certainty. No finite $\varepsilon$ covers that, and neither does any $\delta$ below 1, because the definition would need $1 \le e^\varepsilon \cdot 0 + \delta$.

The leak does not need a degenerate dataset. To measure it on an ordinary pair, use the **privacy loss** at an output $y$, which is the log of the ratio of the output densities on two neighbours:
$$L_{D,D'}(y) = \ln\frac{p_{M(D)}(y)}{p_{M(D')}(y)}.$$
It says how much more likely the output is under the first dataset than under the second, and pure $\varepsilon$-DP holds when it never exceeds $\varepsilon$. Now take $\varepsilon = 1$ and the two datasets of @tbl-sens-leak. They differ in one record and have the same median, but their local sensitivities differ by a factor of six.

| Dataset | Values | Median | $\mathrm{LS}$ |
| --- | --- | --- | --- |
| $D$ | 10, 20, 30, 31, 90 | 30 | $\max(30 - 20, 31 - 30) = 10$ |
| $D'$ (31 replaced by 90) | 10, 20, 30, 90, 90 | 30 | $\max(30 - 20, 90 - 30) = 60$ |

: Two neighbouring datasets with the same median and very different local sensitivities. {#tbl-sens-leak}

The two releases are $30 + \mathrm{Lap}(10)$ and $30 + \mathrm{Lap}(60)$. They have the same centre, so the privacy loss depends only on how far the output lands from that centre, a distance written $u = |y - 30|$. Far from the centre the wide density is much larger than the narrow one, because the narrow one decays six times faster. So the loss in favour of $D'$ grows with $u$, as the log of the ratio of the two Laplace densities confirms:
$$L_{D',D}(y) = -\ln 6 + \frac{u}{12}.$$
The slope $1/12$ is the narrow density's decay rate minus the wide one's, $1/10 - 1/60$. The loss grows without bound as $u$ grows, so no pure $\varepsilon$ can hold for this pair (@fig-sens-local-leak).

Setting the loss equal to 1 and solving for the distance gives $u = 12(1 + \ln 6) = 33.50$. A Laplace variable whose scale is $b$ lands more than $u$ from its centre with probability $e^{-u/b}$. So an output that far out has probability 0.572 under $D'$, but only 0.0351 under $D$.

![Left: the densities of the median of $D$ and of $D'$ released with Laplace noise scaled to their own local sensitivities, 10 and 60. Right: the privacy loss $L_{D',D}(y)$ grows linearly in $|y - 30|$ and crosses $\varepsilon = 1$ at distance 33.50.](figures/sens_local_leak.svg){#fig-sens-local-leak}

How large must $\delta$ be for this pair to satisfy the definition with $\varepsilon = 1$? The smallest such $\delta$ is the largest gap $\Pr[M(D') \in E] - e^{\varepsilon}\Pr[M(D) \in E]$ over all events $E$. That is the **hockey-stick divergence** of [Divergences](../part01-prerequisites/03-divergences.md), and the largest gap comes from taking $E$ to be exactly the outputs where the loss exceeds 1. The full edition's script computes it as 0.477 for $D'$ against $D$. So an adversary who sees an output more than 33.50 from 30 can bet on $D'$ and be right far more often than any reasonable guarantee allows. In the distinguishing game of [Threat models and privacy games](../part02-attacks/01-threat-models.md), the adversary wins by looking at the spread of the noise, not at its centre.

The same leak hides in any noise scale computed from the private data, such as a clip bound set to the largest value observed (@sec-sens-clip).

## Smooth sensitivity {#sec-sens-smooth}

The leak came from the noise scale jumping between neighbours. Nissim, Raskhodnikova and Smith (2007) proposed the first repair, a noise scale that cannot jump. It replaces the local sensitivity by an upper bound that changes by at most a factor $e^\beta$ between neighbours, where $\beta$ is a positive smoothness parameter you choose. A bounded *factor* is the right kind of control because the privacy loss is a log-ratio. Rescaling the noise by $e^\beta$ changes the log of the density's normalizing height by exactly $\beta$. Whether the rest of the density stays close too depends on its tails, as this section shows.

Formally, a function $S$ of the data is a **$\beta$-smooth upper bound** on the local sensitivity if two conditions hold for all neighbours:
$$S(D) \ge \mathrm{LS}_f(D) \quad\text{and}\quad S(D) \le e^\beta S(D').$$
So the bound is never below the local sensitivity, and it changes by at most the factor $e^\beta$ between neighbours.

What is the smallest such bound? Suppose that some dataset $D''$ has a large local sensitivity, and write $k$ for its distance from yours. A smooth bound must be at least that large at $D''$. On the walk from there to your dataset, the bound can drop by at most a factor $e^{-\beta}$ per step. So after $k$ steps it must still be at least $\mathrm{LS}_f(D'')e^{-\beta k}$. The **$\beta$-smooth sensitivity** takes exactly the largest of these discounted values:
$$S^*_\beta(D) = \max_{D''}\ \mathrm{LS}_f(D'')\,e^{-\beta\,d(D, D'')}.$$ {#eq-sens-smooth}
It is itself a $\beta$-smooth upper bound, and the argument above shows that it is the smallest one.

The release adds noise of a fixed shape, stretched by the smooth bound:
$$M(D) = f(D) + \frac{S(D)}{\alpha}\,Z.$$ {#eq-sens-smooth-mech}
Here the noise draw $Z$ comes from a fixed density $h$. The positive constant $\alpha$ caps how far one record can move the centre, measured in noise scales. Since one record moves $f$ by at most the local sensitivity, which is at most $S(D)$, the centre moves by at most $\alpha$ noise scales.

Which densities work depends on their tails. On two neighbours the output distributions differ in their centres and in their scales, and the change of scale enters the privacy loss twice. It changes the density's normalizing height, which costs at most $\beta$, and it stretches the density's shape. To produce the same output, the noise draw on one dataset must be the draw on the other stretched by up to $e^\beta$. So the density must stay within a bounded log-ratio of a stretched copy of itself, however far out in the tail.

That stretch decides between light and heavy tails. The log of the Laplace density at a point $z$ falls in a straight line as $|z|$ grows, so stretching changes it by an amount proportional to $|z|$. That amount grows without bound, which is the leak of @tbl-sens-leak in miniature. A density whose tails decay like a power of $|z|$ has a log-density that falls only like $\ln|z|$, so stretching changes it by a bounded amount everywhere. This is why **pure DP with a data-dependent noise scale needs heavy tails**. The normal density falls off even faster than the Laplace one, so it fails as well.

The densities that work for pure DP form a Cauchy-like family. Each member has a power $\gamma$ and the density $h_\gamma(z) \propto 1/(1 + |z|^\gamma)$, so its tails decay like $|z|^{-\gamma}$. The power must exceed 1, because otherwise the tails would hold infinite total probability, and with $\gamma = 2$ the density is the standard Cauchy. Now take $S$ to be any $\beta$-smooth upper bound. Then the release @eq-sens-smooth-mech is $\varepsilon$-DP when the costs fit within the budget:
$$\gamma\alpha + (\gamma + 1)\beta \le \varepsilon.$$ {#eq-sens-cauchy-cond}
The log of $h_\gamma$ never changes faster than $\gamma$ times its argument does. So the shift of the centre, by at most $\alpha$, costs at most $\gamma\alpha$. The change of scale costs $\beta$ through the normalizing height, plus $\gamma\beta$ through the stretch of the shape. Nissim, Raskhodnikova and Smith use $\alpha = \beta = \varepsilon/(2(\gamma + 1))$, which satisfies the condition with room to spare. For the Cauchy density this gives $\alpha = \beta = \varepsilon/6$, and the often-quoted release is
$$f(D) + \frac{6}{\varepsilon}\,S^*_{\varepsilon/6}(D)\,Z.$$

Laplace noise can still be used, if you pay for its rare large draws with a $\delta$. Take $Z$ from the Laplace distribution with scale 1, and write $z$ for the draw. The privacy loss is then at most $\beta + e^\beta\alpha + (e^\beta - 1)|z|$. Its terms are the change of normalizing height, the shift measured after the stretch, and the stretch of the shape, which grows with the draw. But large draws are rare. A draw exceeds $\ln(1/\delta)$ in absolute value with probability exactly $\delta$ ([Concentration inequalities](../part01-prerequisites/02-concentration.md)). Outside that rare event the loss is at most $\varepsilon$, as long as
$$\beta + e^\beta\alpha + (e^\beta - 1)\ln(1/\delta) \le \varepsilon,$$ {#eq-sens-laplace-cond}
and under this condition the release is $(\varepsilon, \delta)$-DP.

The rule behind this step, and behind every $\delta$ in this chapter, is that a privacy loss which rarely exceeds $\varepsilon$ still gives a guarantee. Outputs where the loss is small obey the factor $e^\varepsilon$, and the rest have total probability at most $\delta$, which becomes the slack.

The condition sets how small $\beta$ must be. For small $\beta$ the stretch term is about $\beta\ln(1/\delta)$, because the factor in front of the logarithm is then close to $\beta$. With about half the budget left for the shift $\alpha$, that term must fit in the other half, so $\beta$ is about $\varepsilon/(2\ln(1/\delta))$. That is much smaller than the Cauchy's $\varepsilon/6$. A small $\beta$ is not free, though, because it makes $S^*_\beta$ larger. The discount $e^{-\beta k}$ then falls more slowly with distance, so more distant datasets can raise the maximum.

Nissim, Raskhodnikova and Smith call these conditions on the noise **admissibility**. They bound the shift and the rescaling separately, so their constants differ slightly from the ones above.

Heavy tails have a price in the error. The Cauchy distribution has no mean, so its error is reported through the median. For the standard Cauchy, the median of $|Z|$ is 1, so the typical error equals the noise scale $S/\alpha$. But the tail is heavy, since $|Z|$ exceeds 10 with probability 0.0635. So about one release in 16 is off by more than ten times the typical error. With Laplace noise the same probability is only about $4.5\times10^{-5}$.

## The smooth sensitivity of the median {#sec-sens-smooth-median}

The definition @eq-sens-smooth is a maximum over all datasets, which you cannot enumerate. The way out is to group the datasets by their distance $k$ from yours. All datasets at distance $k$ get the same discount $e^{-k\beta}$, so within a group you only need the largest local sensitivity. Write $A^{(k)}(D)$ for the largest local sensitivity of any dataset within $k$ changes of $D$. Then the smooth sensitivity is the best discounted group:
$$S^*_\beta(D) = \max_{k = 0, \dots, n} e^{-k\beta}A^{(k)}(D).$$ {#eq-sens-smooth-by-k}

For the median, $A^{(k)}$ asks how wide a gap you can open next to the median by changing $k$ records. You open a gap by taking the values inside a stretch of the sorted data and moving them out of it, to 0 or to $\Lambda$. With $k$ changes you can empty the $k$ values that lie between two order statistics $k + 1$ ranks apart. Each value moved to 0 shifts the remaining values up one rank. So by choosing how many go down and how many go up, you can place the emptied stretch next to the median. The walkthrough below shows this on five values.

The counting argument for local sensitivity extends to $k$ changes, which move each sorted value by at most $k$ ranks. That is why only windows near rank $m$ count. A window can then reach ranks below 1 or above $n$. So pad the sorted list with zeros below and copies of $\Lambda$ above, since "below the smallest value" means "possibly as low as 0". With this padding, a result of the same authors (their Proposition 3.4) says that
$$A^{(k)}(D) = \max_{0 \le t \le k+1}\big(x_{m+t} - x_{m+t-k-1}\big).$$ {#eq-sens-median-ak}
Each window runs from rank $m + t - k - 1$ to rank $m + t$, and the offset $t$ slides it across the median. Each $A^{(k)}$ takes one subtraction per window, and the number of changes never needs to exceed $n$. So the exact computation takes a number of steps that grows like $n^2$, written $O(n^2)$.

@tbl-sens-walk works through a small case. Take the dataset $D = (1, 3, 4, 6, 9)$, with the top of the range at $\Lambda = 10$. The smoothness parameter is $\beta = 0.5$. The full edition checks the formula against brute force at every $k$.

| $k$ | Windows $x_{m+t} - x_{m+t-k-1}$, $t = 0, \dots, k+1$ | $A^{(k)}$ | $e^{-0.5k}A^{(k)}$ |
| --- | --- | --- | --- |
| 0 | $4 - 3,\ 6 - 4$ | 2 | 2.00 |
| 1 | $4 - 1,\ 6 - 3,\ 9 - 4$ | 5 | **3.03** |
| 2 | $4 - 0,\ 6 - 1,\ 9 - 3,\ 10 - 4$ | 6 | 2.21 |
| 3 | $4 - 0,\ 6 - 0,\ 9 - 1,\ 10 - 3,\ 10 - 4$ | 8 | 1.79 |
| 4 | $4 - 0,\ 6 - 0,\ 9 - 0,\ 10 - 1,\ 10 - 3,\ 10 - 4$ | 9 | 1.22 |
| 5 | $4 - 0,\ 6 - 0,\ 9 - 0,\ 10 - 0,\ 10 - 1,\ 10 - 3,\ 10 - 4$ | 10 | 0.82 |

: The smooth sensitivity of the median of $D = (1, 3, 4, 6, 9)$ in the range from 0 to 10, with $\beta = 0.5$: the windows of each row, the widest gap $A^{(k)}$, and the discounted values whose maximum is $S^*_{0.5}$. {#tbl-sens-walk}

The first row gives the local sensitivity, which is the gap of 2 above the median. But one change opens a wider gap, by moving the 6 up to the top of the range. The next value above the median is then the 9, so the gap has width 5. Even after the discount $e^{-0.5} = 0.6065$ for its one step, that gap is worth 3.03, which beats the local sensitivity. Further rows open wider gaps but pay larger discounts. So $S^*_{0.5}(D) = 3.03$, attained at $k = 1$ (@fig-sens-smooth-profile, left).

![Left: $A^{(k)}$ and $e^{-k\beta}A^{(k)}$ for the walkthrough dataset; the star marks $S^*_{0.5} = 3.03$. Right: the same for the median of 1,001 synthetic ages (a fresh draw) with $\beta = 1/6$ (Cauchy) and $\beta = 0.032$ (Laplace, $\delta = 10^{-6}$); the stars mark the maxima. The smaller $\beta$ reaches further out.](figures/sens_smooth_profile.svg){#fig-sens-smooth-profile}

Now return to a realistic sample. The full edition's script draws 1,001 synthetic ages, roughly normal around 38 years and clipped to at most $\Lambda$. It releases their median with $\varepsilon = 1$, and compares the options by their median absolute error. For noise that is a scale times a draw $Z$, that error is the scale times the median of $|Z|$. The median of $|Z|$ is 1 for the Cauchy, as shown above. For the Laplace distribution it is $\ln 2$, since that is the distance a draw exceeds in absolute value with probability $\frac12$.

With global sensitivity, the Laplace noise has scale $\Lambda/\varepsilon = 100$, so the median absolute error is 69.31 years and the release is useless. The local sensitivity is 0.1386 years. The Cauchy release uses $\alpha = \beta = 1/6$, and its smooth sensitivity is 0.1632. That is only slightly above the local sensitivity, because with this many people even the seven changes where the maximum is reached cannot open a wide gap near the middle. The noise scale is $6S^*/\varepsilon = 0.979$, and that is also the median absolute error.

The Laplace release gives half the budget to the shift, $\alpha = 1/2$, and uses $\delta = 10^{-6}$. With these values the largest $\beta$ that @eq-sens-laplace-cond allows is 0.0322. That smaller $\beta$ lets more distant datasets count, so the smooth sensitivity rises to 0.5144, and the noise scale is $S^*/\alpha = 1.029$. The two noise scales are close, yet the Laplace release is more accurate, with median absolute error 0.713. The reason is that the median of a Laplace $|Z|$ is $\ln 2$ rather than 1. Its guarantee, though, carries the slack $\delta = 10^{-6}$, which the Cauchy release does not need.

Both smooth releases are 70 to 100 times more accurate than the global one. The gain grows with $n$ (@fig-sens-methods-vs-n), because the smooth noise follows the spacing of the data near the median, which shrinks roughly like $1/n$. The same reasoning shows where the method fails. With only a few points, a few changes can open a gap across nearly the whole range. Then $S^*$ can come close to $\Lambda$, and the factor $1/\alpha$ multiplies it. In the full edition's comparison, the smooth noise is worse than the global noise below about 20 points.

## Propose-test-release {#sec-sens-ptr}

Smooth sensitivity needs a formula like @eq-sens-median-ak, and such formulas are known for only a few functions. Dwork and Lei (2009) take a cruder route that needs much less. You propose a bound $b$ on the local sensitivity before looking at the data. Then you check privately that the data are far from any dataset where it fails, and only then release with noise of scale $b/\varepsilon$.

The quantity to check is the number of records you must change before the local sensitivity can exceed $b$. Call it the distance to instability:
$$\operatorname{dist}_b(D) = \min\big\{d(D, D''):\ \mathrm{LS}_f(D'') > b\big\}.$$
One record changes this distance by at most 1, because a path from a neighbour to an unstable dataset can always go through $D$ first. So adding Laplace noise of scale $1/\varepsilon$ makes the distance $\varepsilon$-DP. For the median, the distance is the first number of changes $k$ at which $A^{(k)}(D)$ exceeds $b$.

**Propose-test-release** adds $\mathrm{Lap}(1/\varepsilon)$ noise to the distance and compares the result with the threshold $\ln(1/\delta)/\varepsilon$. If the noisy distance is at or below the threshold, it outputs $\bot$, which means "no answer". Otherwise it releases $f(D) + \mathrm{Lap}(b/\varepsilon)$. The threshold is chosen so that the noise alone pushes a distance of 0 over it only with probability $\delta/2$, and the mechanism is $(2\varepsilon, \delta/2)$-DP.

The reason is a split of neighbouring pairs into two cases. In the first case the proposal is right for the pair, so that $f(D)$ and $f(D')$ differ by at most $b$. Then the release is an ordinary Laplace mechanism with sensitivity $b$, and it is $\varepsilon$-DP. The test is also $\varepsilon$-DP, and deciding from it whether to release is post-processing. So by basic composition this case costs $2\varepsilon$ ([Post-processing, group privacy and composition](02-properties.md)).

In the second case the proposal is wrong, so $f(D)$ and $f(D')$ differ by more than $b$. Each dataset is a neighbour of the other, so both local sensitivities exceed $b$, and both datasets sit at distance 0. The test then passes only if the Laplace draw alone lands $\ln(1/\delta)/\varepsilon$ above zero, which happens with probability $\delta/2$. So a dataset whose local sensitivity exceeds the proposal can leak only through that rare draw, and the rare lucky pass is what $\delta$ pays for.

Propose-test-release is all or nothing. When the data are stable, the noise scale is set by your proposal rather than by the data, so it reveals nothing. When they are not, you get $\bot$, which is itself private, because it says only that the data were probably near an unstable dataset.

The threshold is large. A distance equal to it passes only half the time, because the Laplace noise is positive half the time. At $\varepsilon = 1$ the threshold is simply $\ln(1/\delta)$ changes. For $\delta = 10^{-6}$ that is 13.82, so the data must be stable against more than a dozen changes, not just one.

The proposal $b$ must not depend on the data. Suppose you choose $b$ as twice the observed local sensitivity. Then the noise scale of the release is a function of the data again, and the leak of @sec-sens-local comes back.

Compared with smooth sensitivity, propose-test-release needs only the distance to instability, which is often easier to compute or bound than $S^*$. It pays $2\varepsilon$ instead of $\varepsilon$, because it runs two Laplace mechanisms, where smooth sensitivity needs one noise draw. It can also refuse, which smooth sensitivity never does.

On the 1,001 ages, propose $b = 1$ year. The smallest number of changes that can push the local sensitivity above 1 year is 21, and that is the distance. It is comfortably above the threshold of 13.82, so the test fails only if the noise lowers the distance by more than the difference. The pass probability is therefore 0.999621. The release then has noise $\mathrm{Lap}(1)$, whose median absolute error is $\ln 2 = 0.693$ years, and the total budget is twice $\varepsilon$. On a sample of 101 ages, by contrast, the distance is only 1. There the test passes with probability $1.36\times10^{-6}$, because one year is far too optimistic for so few people. @fig-sens-methods-vs-n (right) shows the pass probability switching from 0 to 1 between 500 and 1,000 people.

![Left: median absolute noise when releasing the median age with $\varepsilon = 1$, by global sensitivity, smooth sensitivity with Cauchy noise and with Laplace noise ($\delta = 10^{-6}$), and propose-test-release with $b = 1$ when it passes, averaged over 10 synthetic datasets per $n$. The dotted line is the non-private local sensitivity. Right: the probability that propose-test-release with $b = 1$ answers.](figures/sens_methods_vs_n.svg){#fig-sens-methods-vs-n}

## Subsample-and-aggregate {#sec-sens-saa}

Sometimes you cannot analyse $f$ at all, because it is a trained model or a pipeline you did not write. **Subsample-and-aggregate** (Nissim, Raskhodnikova and Smith 2007; Smith 2011) treats $f$ as a black box. It runs $f$ on disjoint pieces of the data and privately combines the answers. One record lives in one piece, so it can change only one answer.

Split the $n$ records into $B$ disjoint blocks, using a partition fixed in advance, for example by a hash of a record identifier. Run $f$ on each block, and write $z_j$ for the answer on block $j$. Clip each answer to the range from 0 to a bound $R$ fixed in advance, and release their average plus Laplace noise:
$$M(D) = \frac1B\sum_{j=1}^B \operatorname{clip}_{[0,R]}(z_j) + \mathrm{Lap}\Big(\frac{R}{B\varepsilon}\Big).$$
Replacing one record changes only the block that contains it, so it moves one clipped answer by at most $R$, and the average by at most $R/B$. That is a Laplace mechanism with sensitivity $R/B$, so the release is $\varepsilon$-DP for any $f$. The proof never looks inside $f$, but it does rely on the partition. If the blocks depended on the values, replacing one record could reshuffle many records between blocks and change many answers.

As an estimate of a target $\theta$, the release has an error with three parts, which add in the mean squared error. Write $z$ for the output of $f$ on one block. When the clipping is inactive and the blocks are independent samples, the mean squared error is
$$\underbrace{\big(\mathbb E[z] - \theta\big)^2}_{\text{bias of } f \text{ on } n/B \text{ records}} + \underbrace{\frac{\operatorname{Var}(z)}{B}}_{\text{sampling}} + \underbrace{2\Big(\frac{R}{B\varepsilon}\Big)^2}_{\text{noise}}.$$ {#eq-sens-saa-error}
The bias is there because $f$ on a small block can be biased in a way that $f$ on the whole dataset is not. The three terms simply add, because the sampling error and the noise both average to zero and are independent, so their products average to zero. Both conditions are needed. Active clipping adds a bias that the formula leaves out, and the sampling term $\operatorname{Var}(z)/B$ assumes independent blocks.

The number of blocks sets the trade-off. On one side, the noise falls like $1/B$, because one record's influence on the average is diluted over more blocks. On the other side, the bias grows with $B$, because the blocks get smaller. The sampling term hardly moves, because smaller blocks make each estimate noisier by about the same factor that averaging more of them removes. So the best $B$ balances the bias against the noise.

For an example, take $n = 10{,}000$ inter-arrival times, drawn independently from an exponential distribution with rate $\theta = 1$. The privacy parameter is again $\varepsilon = 1$, and the maximum-likelihood estimate is one over the sample mean, $1/\bar x$. Its global sensitivity is infinite, because a dataset of tiny times has a huge $1/\bar x$, and one record can change it by an arbitrary amount. On a block of $n_b$ records, this estimate has bias $\theta/(n_b - 1)$, which the full edition derives from the gamma distribution of the block's sum. The block answers are clipped to the range from 0 to $R = 5$, set in advance well above the true rate, and @tbl-sens-saa shows the three terms for four block counts.

| Blocks $B$ | Block size | Bias | Sampling sd | Noise sd | Root-mean-square error |
| --- | --- | --- | --- | --- | --- |
| 10 | 1,000 | 0.0010 | 0.0100 | 0.7071 | 0.7072 |
| 100 | 100 | 0.0101 | 0.0102 | 0.0707 | 0.0722 |
| 250 | 40 | 0.0256 | 0.0105 | 0.0283 | 0.0396 |
| 1,000 | 10 | 0.1111 | 0.0124 | 0.0071 | 0.1120 |

: Error of subsample-and-aggregate for the rate of an exponential distribution, with $n = 10{,}000$, $\theta = 1$, $R = 5$ and $\varepsilon = 1$: the three terms of @eq-sens-saa-error and their combination. {#tbl-sens-saa}

Few blocks leave the noise dominant, and many small blocks make the bias dominant. The best block count on the script's grid is 250, where the bias and the noise are of similar size and the error is 0.0396. A non-private estimate has standard deviation about $\theta/\sqrt n = 0.0100$. So privacy at $\varepsilon = 1$ costs a factor of four here, for an estimator whose sensitivity you never had to bound (@fig-sens-saa-blocks). Choose the block count from an error model like @eq-sens-saa-error, before looking at the data, not by trying values on it.

![Root-mean-square error of subsample-and-aggregate for the exponential rate against the number of blocks $B$ (formula and simulation), with its three components: the bias of the estimator on small blocks, the Laplace noise, and the sampling error.](figures/sens_saa_blocks.svg){#fig-sens-saa-blocks}

The clip range $R$ is still a global bound, now on the output of $f$, and a loose $R$ means large noise. Nissim, Raskhodnikova and Smith therefore refine the combining step with smooth sensitivity, so that the noise adapts to how tightly the block answers agree. Smith (2011) shows that for estimators whose error is approximately normal, a version of the method costs nothing in the leading term of the error. In private aggregation of teacher ensembles (PATE), "teacher" models trained on disjoint shards of the data are combined by a noisy vote (Papernot et al. 2017). That is subsample-and-aggregate with a vote as the combining step.

## Clipping {#sec-sens-clip}

On unbounded data the global sensitivity of the mean is infinite. **Clipping** forces a bound, by replacing each value $x$ with $\min(x, C)$ before averaging, where $C$ is a clip bound you choose. Take non-negative data, so that each clipped value lies between 0 and $C$. Replacing one record then changes the sum by at most $C$, and the mean by at most $C/n$. So the clipped mean adds Laplace noise for that sensitivity, which makes it $\varepsilon$-DP:
$$M(D) = \frac1n\sum_{i=1}^n\min(x_i, C) + \mathrm{Lap}\Big(\frac{C}{n\varepsilon}\Big).$$

The same move applied to per-example gradients, rather than to values, is the heart of private model training in later parts. Every "assume the data lie between 0 and $\Lambda$" of earlier chapters is a clip in disguise, and it carries the cost described next.

That cost is a bias. Clipping removes exactly the part of each value above $C$, which is written
$$(x - C)_+ = \max(x - C, 0).$$
So the estimate is low by the average excess $\mathbb E[(X - C)_+]$, where $X$ is a random value drawn from the data's distribution.

Assume that the $n$ values are drawn independently from that distribution, because only then does averaging divide their variance by $n$. The error then has the same three parts as in subsample-and-aggregate, namely the bias, the sampling variation of the clipped values, and the noise. So the mean squared error of the clipped mean is
$$\operatorname{err}(C)^2 = \underbrace{\mathbb E[(X - C)_+]^2}_{\text{bias}^2} + \underbrace{\frac{\operatorname{Var}(\min(X, C))}{n}}_{\text{sampling}} + \underbrace{\frac{2C^2}{n^2\varepsilon^2}}_{\text{noise}}.$$ {#eq-sens-clip-error}

Raising $C$ shrinks the bias and grows the noise.

So the best $C$ is where the derivative of @eq-sens-clip-error in $C$ is zero. Raising $C$ a little raises a clipped value only when the value is above $C$, so the fraction $\Pr[X > C]$ of values that clipping touches sets how fast the bias falls. Working out the derivative shows that the best $C$ solves
$$\Pr[X > C]\Big(\mathbb E[(X - C)_+] - \frac{C - \mathbb E[\min(X, C)]}{n}\Big) = \frac{2C}{n^2\varepsilon^2}.$$ {#eq-sens-clip-opt}
The second term in the bracket comes from the sampling variance, and for large $n$ it drops out because of its extra $1/n$. Then the left side falls as $C$ grows, while the right side grows with $C$ and shrinks as $n\varepsilon$ grows. So with more data the two sides meet at a larger $C$, further out in the tail. In words, the bias does not shrink with $n$ but the noise does, so with more data you can afford to clip less.

For an example, take incomes in thousands that follow a log-normal distribution, which means that the log of an income is normal, here with mean $\ln 40$ and standard deviation 0.8. Half the logs lie below $\ln 40$, so the median income is 40. The mean is larger, 55.085, because of a heavy right tail. Each term of @eq-sens-clip-error then has a closed form, and @tbl-sens-clip evaluates them for $n = 1000$ and $\varepsilon = 1$.

| Clip bound $C$ | $\Pr[X > C]$ | Bias | Sampling sd | Noise sd | Root-mean-square error |
| --- | --- | --- | --- | --- | --- |
| 40 (the median) | 0.5000 | 23.415 | 0.335 | 0.057 | 23.417 |
| 257.23 (99th percentile) | 0.0100 | 0.924 | 1.462 | 0.364 | 1.767 |
| 331.34 (optimal) | 0.0041 | 0.438 | 1.538 | 0.469 | 1.666 |
| 1000 | 0.0000 | 0.006 | 1.645 | 1.414 | 2.169 |

: Error of the clipped private mean of log-normal incomes, with $n = 1000$ and $\varepsilon = 1$, for four clip bounds: the three terms of @eq-sens-clip-error and their combination. {#tbl-sens-clip}

Clipping at the median is a disaster, because half the incomes are cut back and the bias of 23 dwarfs everything else. The optimum, 331.34, cuts off only 0.4% of the population. Its total error is 1.666, against a standard deviation of 1.649 for the unclipped, non-private mean. So privacy and clipping together add only 1% to the error here.

Above the optimum the noise takes over, because its standard deviation $\sqrt2\,C/(n\varepsilon)$ grows in proportion to $C$. At $C = 1000$ it reaches 1.414 (@fig-sens-clip-tradeoff, left). The optimum also grows with $n$, as the condition predicted. With a hundred people it is 177.5, and with ten thousand people it moves out to 557.3 (@fig-sens-clip-tradeoff, right).

![Left: root-mean-square error of the clipped private mean of log-normal incomes against the clip bound $C$, with its bias, noise and sampling components and simulated points; the dotted line marks the optimum $C^* = 331$. Right: the optimal clip bound as a function of the dataset size.](figures/sens_clip_tradeoff.svg){#fig-sens-clip-tradeoff}

The bias also has a sign, because clipping only from above always lowers values, so it always underestimates the mean. Data on both sides of zero are clipped to the range from $-C$ to $C$, which cuts off mass on both sides. The two cuts cancel only for distributions symmetric about 0, so only those are estimated without bias.

Choosing $C$ is itself a decision that depends on the data. Setting $C$ to the observed maximum, or to the empirical 99th percentile, makes the noise scale a function of the data, and the leak of @sec-sens-local returns. So choose $C$ from domain knowledge or from an error model like @eq-sens-clip-error, before seeing any records. Otherwise spend part of the budget on a private quantile, which the [exponential mechanism](05-selection.md), smooth sensitivity or propose-test-release can provide.

Finally, the clipped mean as written relies on replace neighbours, where the size $n$ is public. Under add/remove neighbours, where one dataset has an extra record, the size $n$ itself changes, so the mean's denominator is not public. The fix is to release a noisy clipped sum and a noisy count and divide them, charging both releases to the budget ([Post-processing, group privacy and composition](02-properties.md)).

## Code {#sec-sens-code}

The smooth sensitivity of the median is easy to get subtly wrong, mostly in the padding of the order statistics. The snippet computes it for the walkthrough dataset of @tbl-sens-walk, with $\beta = 0.5$, and it also solves @eq-sens-laplace-cond for the $\beta$ of the age example.

```python
from math import exp, log


def gap_within(xs: list[float], upper: float, k: int) -> float:
    """A(k): the widest gap next to the median that changing k records can open."""
    n, m = len(xs), (len(xs) + 1) // 2  # n odd; m is the rank of the median

    def x(i: int) -> float:  # order statistic i, padded with 0 below and `upper` above
        return 0.0 if i < 1 else upper if i > n else xs[i - 1]

    return max(x(m + t) - x(m + t - k - 1) for t in range(k + 2))


def smooth_sensitivity_median(values: list[float], upper: float, beta: float) -> float:
    """The beta-smooth sensitivity of the median of values in [0, upper]."""
    xs = sorted(values)
    return max(exp(-k * beta) * gap_within(xs, upper, k) for k in range(len(xs) + 1))


def laplace_beta(eps: float, delta: float, alpha: float) -> float:
    """The largest beta with beta + e^beta * alpha + (e^beta - 1) * ln(1/delta) <= eps."""
    lo, hi = 0.0, eps
    for _ in range(100):
        mid = (lo + hi) / 2
        ok = mid + exp(mid) * alpha + (exp(mid) - 1) * log(1 / delta) <= eps
        lo, hi = (mid, hi) if ok else (lo, mid)
    return lo


data = [1.0, 3.0, 4.0, 6.0, 9.0]
print(f"{gap_within(sorted(data), 10.0, 0):.2f}")  # 2.00
print(f"{smooth_sensitivity_median(data, 10.0, 0.5):.2f}")  # 3.03
print(f"{laplace_beta(1.0, 1e-6, 0.5):.4f}")  # 0.0322
```

The distance for propose-test-release is the first `k` with `gap_within(xs, upper, k) > b`. The full edition's script computes every number in this chapter, and it checks the formula for $A^{(k)}$ by brute force:

```bash
uv run python code/part03-dp-foundations/sensitivity.py
```

## Cheat sheet

| Technique | Release | Privacy | Use when |
| --- | --- | --- | --- |
| Local sensitivity | $f(D) + \mathrm{Lap}(\mathrm{LS}_f(D)/\varepsilon)$ | **Not private** | Never |
| Median | $\mathrm{LS} = \max(x_m - x_{m-1}, x_{m+1} - x_m)$, global sensitivity $\Lambda$ | — | — |
| Smooth sensitivity | $S^*_\beta(D) = \max_k e^{-k\beta}A^{(k)}(D)$, the smallest $\beta$-smooth upper bound; $O(n^2)$ for the median | — | — |
| Smooth sensitivity, Cauchy-like | $f(D) + \frac{S(D)}{\alpha}Z$, $h \propto 1/(1 + \lvert z\rvert^\gamma)$, $\gamma\alpha + (\gamma + 1)\beta \le \varepsilon$; Cauchy: $f(D) + \frac{6}{\varepsilon}S^*_{\varepsilon/6}(D)\,Z$ | $\varepsilon$ | Median, quantiles; $n$ in the hundreds or more |
| Smooth sensitivity, Laplace | $f(D) + \frac{S(D)}{\alpha}\mathrm{Lap}(1)$, $\beta + e^\beta\alpha + (e^\beta - 1)\ln\frac1\delta \le \varepsilon$ | $(\varepsilon, \delta)$ | Same, when you want light tails |
| Propose-test-release | $\bot$ if $\operatorname{dist}_b + \mathrm{Lap}(\frac1\varepsilon) \le \frac{\ln(1/\delta)}{\varepsilon}$, else $f(D) + \mathrm{Lap}(\frac b\varepsilon)$ | $(2\varepsilon, \delta/2)$ | Stable statistics without a smooth-sensitivity formula |
| Subsample-and-aggregate | Mean of $B$ clipped block outputs $+ \mathrm{Lap}(\frac{R}{B\varepsilon})$; error = bias on small blocks + sampling + noise | $\varepsilon$ | Black-box estimators and models |
| Clipped mean | $\frac1n\sum\min(x_i, C) + \mathrm{Lap}(\frac{C}{n\varepsilon})$; bias $\mathbb E[(X - C)_+]$ | $\varepsilon$ | Means and sums of unbounded data |

: The techniques of this chapter, with neighbours that replace one record. {#tbl-sens-cheat}

**Rules of thumb**

- If the noise scale depends on the data, ask what the noise scale reveals. If you cannot answer, use one of the four repairs.
- Pure DP with a data-dependent scale needs heavy-tailed noise. Laplace noise needs a $\delta$ and a much smaller $\beta$, near $\varepsilon/(2\ln(1/\delta))$.
- Smooth sensitivity pays off once the data are dense around the statistic. For the median, that means $n$ in the hundreds.
- Fix clip bounds, block counts and proposals before looking at the data, from an error formula or with a separate slice of the budget.

## Further reading

- Nissim, Raskhodnikova & Smith, "Smooth Sensitivity and Sampling in Private Data Analysis" (STOC 2007): local and smooth sensitivity, the admissible noise distributions, the median formula and the original sample-and-aggregate framework.
- Dwork & Lei, "Differential Privacy and Robust Statistics" (STOC 2009): propose-test-release, and robust private estimators built on it.
- Amin, Kulesza, Muñoz Medina & Vassilvitskii, "Bounding User Contributions: A Bias-Variance Trade-off in Differential Privacy" (ICML 2019): choosing clip bounds when users contribute many records.

**Derivations.** Every result in this chapter is derived step by step in [the full edition's chapter](../../book-full/part03-dp-foundations/06-sensitivity.md), except the "center of attention" aggregator and Smith's (2011) result for asymptotically normal estimators, which it cites.
