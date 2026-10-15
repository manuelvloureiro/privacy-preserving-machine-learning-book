# Amplification by shuffling and by iteration

## Why this matters

[Amplification by subsampling](05-subsampling.md) showed that uncertainty is worth privacy. If an adversary cannot tell whether a step used your record, the step costs far less than its noise alone suggests. A step that touches your record only with some small probability $q$ leaks roughly $q$ times as much. Two other kinds of uncertainty matter in practice, and each needs its own accountant.

The first is uncertainty about which report is yours. In the shuffle model ([Local, shuffle and distributed trust models](../part03-dp-foundations/08-trust-models.md)), each of $n$ users randomizes their own data on their own device. Each user then sends the noisy report through a **shuffler**, a trusted component that strips identities and mixes the order. So the analyst sees only a bag of anonymous reports.

Each report on its own satisfies local differential privacy (DP) at some level $\varepsilon_0$. That means whoever sees the one report, even the collector, learns little about its sender. To keep the reports useful, $\varepsilon_0$ is usually large, which makes this guarantee weak. But the analyst does not know which report in the bag is yours, so your report hides among the others. The bag as a whole then satisfies central DP, the guarantee about everything the analyst sees, with an $\varepsilon$ that falls like $1/\sqrt n$. The shuffling here mixes anonymous reports that users have already randomized. It differs from the shuffle-and-partition batching that trains a model in the central model, where the shuffle only hides which batch each record is in. [Amplification by subsampling](05-subsampling.md#sec-sub-schemes) explains why the subsampling accountant does not apply to that case.

The second is uncertainty about what happened inside training. Noisy gradient descent (GD) on a convex loss forgets. Each later step adds fresh noise, and on a convex loss a gradient step never pushes two parameter vectors further apart. So the effect of your record on the final model is washed out step by step. Composition, as in [Rényi DP](01-renyi-dp.md), charges every step as if the adversary had seen every intermediate model. If only the final model is released, a record used early in training is much better hidden than composition says. For some problems, the privacy loss even stops growing with the number of steps.

Every result in this chapter rests on one idea. You show that what the adversary sees can be generated from a much simpler pair of distributions, by a procedure that does not depend on your record. Anything the adversary can learn from the real output, it could then learn from the simple pair, by running that procedure itself. This is the data processing inequality of [Divergences](../part01-prerequisites/03-divergences.md#sec-div-dpi), and it means that the simple pair's privacy holds for the real output. For shuffling, the simple pair counts how many reports look like each of your two possible values. For iteration, it is a pair of normal distributions, or a pair of distributions that a chain of noisy steps keeps close together.

All the results use replace-one neighbours. The two datasets $D$ and $D'$ have the same size and differ in the value of one record, yours. Every theorem in this chapter is stated for that relation.

## Shuffled binary randomized response, exactly {#sec-shuf-rr}

Start with the simplest local randomizer, because for it the privacy of the shuffled bag can be computed exactly. That exact answer is the yardstick for every general bound in the next section. Each user holds a bit and reports it with binary randomized response ([The definition of differential privacy](../part03-dp-foundations/01-definition.md#sec-dpdef-rr)). The user reports the true bit with a probability $\theta$ and the flipped bit otherwise, where
$$\theta = \frac{e^{\varepsilon_0}}{1 + e^{\varepsilon_0}}.$$
A single report is $\varepsilon_0$-DP, because any report is at most $\theta/(1 - \theta) = e^{\varepsilon_0}$ times likelier for one value of the bit than for the other.

Once the shuffler has removed the order, a bag of bits can tell the analyst only how many of them are 1. Because the shuffler picks the order uniformly at random, every arrangement of a given number of 1s is equally likely whatever the users hold. So the arrangement carries no information beyond the count. Your report moves the count by at most one. The other users' reports add random noise to it, and that noise is what hides you.

To write down what the analyst sees, let your bit be 0 in $D$ and 1 in $D'$, and suppose for now that the other users all hold 0. Each of them reports 1 only when their coin flips the bit, which happens with probability $1 - \theta$. So the number of 1-reports among the $n - 1$ others is binomial, and we write $b(y)$ for the probability that it equals $y$. A total count of $y$ then arises in two ways. Either your report is 0 and the others sent $y$ ones, or your report is 1 and the others sent $y - 1$. Weighting the two cases by the chance of your report gives the distribution $P$ of the count when you hold 0, and the distribution $Q$ when you hold 1:
$$P(y) = \theta\,b(y) + (1 - \theta)\,b(y - 1), \qquad Q(y) = (1 - \theta)\,b(y) + \theta\,b(y - 1).$$ {#eq-shuf-rr-pair}

The exact $\delta$ for a given $\varepsilon$ is the **hockey-stick divergence** between the two distributions ([Divergences](../part01-prerequisites/03-divergences.md#sec-div-hockey)). It is the largest gap, over all sets of outcomes, between the chance of the set under $P$ and $e^\varepsilon$ times its chance under $Q$. For each count $y$, write $L(y) = \ln(P(y)/Q(y))$ for the **privacy loss**, how much likelier that count is when you hold 0 than when you hold 1. Every count whose privacy loss exceeds $\varepsilon$ adds its excess to the divergence, and no other count does:
$$\delta(\varepsilon) = \sum_{y:\,L(y) > \varepsilon}\big(P(y) - e^\varepsilon Q(y)\big).$$
The adversary may also try to tell $D'$ from $D$, so you compute the same sum with $P$ and $Q$ swapped and keep the larger of the two. Each sum has only $n + 1$ terms, so the exact value is cheap to compute. The full edition's script takes milliseconds even for 100,000 users, and the plain snippet in @sec-shuf-code takes a few seconds.

The pair assumes that the other users all hold 0, but a guarantee must hold whatever they hold. The full edition checks the cases in which some of the others hold 1, at $\delta = 10^{-6}$. It checks all of them for 1000 users, and up to 30 such users for 10,000 and 100,000 users. In every case it checks, the largest $\varepsilon$ is within $10^{-4}$ of the all-zero value. So this chapter calls the all-zero value exact, although strictly it is exact only for that one pair, and the worst case may be slightly larger.

A small case shows the count at work. Take three users, and set $\varepsilon_0$ to $\ln 3$ so that each report tells the truth with probability $\theta = 3/4$. The two other users hold 0, so each of them reports 1 with probability $1/4$, and @tbl-shuf-n3 fills in @eq-shuf-rr-pair count by count.

| Count $y$ | 0 | 1 | 2 | 3 |
| --- | --- | --- | --- | --- |
| $b(y)$, others' 1-reports | 9/16 | 6/16 | 1/16 | 0 |
| $P(y)$, you hold 0 | 27/64 | 27/64 | 9/64 | 1/64 |
| $Q(y)$, you hold 1 | 9/64 | 33/64 | 19/64 | 3/64 |
| $P(y)/Q(y)$ | 3.0000 | 0.8182 | 0.4737 | 0.3333 |

: Shuffled binary randomized response with $n = 3$ users and $\varepsilon_0 = \ln 3$; the two other users hold 0. {#tbl-shuf-n3}

Now take $\varepsilon = 0.5$, so that the threshold $e^\varepsilon$ is 1.6487. Only the count of zero has a ratio $P/Q$ above that threshold, so only it enters the sum, which gives 0.1900. In the other direction, the two highest counts have a ratio $Q/P$ above the threshold, but together they give only 0.0861. So the shuffled reports have $\delta(0.5) = 0.1900$. A single report on its own has just two outcomes, and the same sum gives it $\delta(0.5) = 0.3378$, so even two other users nearly halve $\delta$.

The first column of @tbl-shuf-n3 also shows why shuffling never improves pure DP, the guarantee with $\delta = 0$. When nobody reports a 1, the ratio $P(0)/Q(0)$ is exactly $e^{\varepsilon_0}$, whatever the number of users. In words, if no report is a 1, you probably hold 0, and no number of other users changes how strongly that outcome points to you. What changes is how often it happens. When you hold 0, it requires every report, yours included, to be 0, which has probability $\theta^n$ and is tiny for large $n$. So shuffling amplifies by moving the rare revealing outcomes into $\delta$. Any shuffled guarantee with $\varepsilon$ below $\varepsilon_0$ therefore needs $\delta > 0$.

Why should the privacy fall like $1/\sqrt n$? Under both datasets the count is a sum of many independent bits, so by the central limit theorem it is close to a normal distribution. Your report shifts its mean by $2\theta - 1$, the difference between the chances that it is 1 when you hold 1 and when you hold 0. Every report has the same variance $\theta(1 - \theta)$, so the count has the standard deviation $\sqrt{n\theta(1 - \theta)}$ under both datasets.

That turns the question into one about two normal distributions with equal variance. Gaussian differential privacy (GDP) uses such pairs as its yardstick ([f-DP and Gaussian DP](04-gaussian-dp.md)). A mechanism is **$\mu$-GDP** when its outputs on neighbouring datasets are at most as easy to tell apart as two such normals whose means are $\mu$ standard deviations apart. So the shuffled count is approximately $\mu$-GDP, with $\mu$ the shift divided by the standard deviation. Writing $\theta$ in terms of $\varepsilon_0$ and simplifying that ratio gives
$$\mu \approx \frac{2\sinh(\varepsilon_0/2)}{\sqrt n}.$$ {#eq-shuf-rr-clt}
Here $\sinh(x) = (e^x - e^{-x})/2$, which is close to $e^x/2$ for large $x$. More users add more noise to hide one report in, which gives the $1/\sqrt n$. A larger $\varepsilon_0$ makes each report more faithful. Your report then shifts the count more, and the others' reports vary less, so they blur it less, which gives the growth like $e^{\varepsilon_0/2}$.

The normal approximation needs many 1-reports to hide among. With many users and a moderate $\varepsilon_0$, it comes close to the exact value (@tbl-shuf-compare in the next section). It fails when the others send few 1-reports. Take 1000 users and $\varepsilon_0 = 4$. Each of the others then reports a one only with probability $1/(1 + e^4)$, so together they send only 18.0 such reports on average. A count that small is far from normal. Converted to an $\varepsilon$ at $\delta = 10^{-6}$, the approximation gives 0.9667, while the exact value is far larger, at 1.6910. Since the exact computation is cheap, use it.

![Left: the distribution of the number of 1-reports among $n = 30$ shuffled randomized-response reports with $\varepsilon_0 = 2$, when you hold 0 (blue) and when you hold 1 (orange); the other users hold 0. Your report moves the count by at most one, against a spread of about $\sqrt{n\theta(1-\theta)}$. Right: the exact $\delta(\varepsilon)$ for one unshuffled report and for $n = 10, 100, 1000$ shuffled reports. All curves reach 0 only at $\varepsilon = \varepsilon_0$, but for large $n$ they are negligible far earlier.](figures/shuffling_iteration_counts.svg){#fig-shuf-counts}

In @fig-shuf-counts, your report moves the count by one against a much wider spread, and more users push $\delta(\varepsilon)$ down sharply, although every curve still reaches zero only at $\varepsilon_0$.

Here is what this buys in a deployment. A company asks 10,000 users whether they used a feature, through randomized response with $\varepsilon_0 = 2$ and a shuffler. On its own, each user's report gives a weak guarantee, since it tells the truth with probability $\theta = 0.8808$, about seven times out of eight. The shuffled bag of reports, though, is $(0.0870, 10^{-6})$-DP (@tbl-shuf-compare). The shuffler bought a strong central guarantee without anyone seeing the raw data.

## The clone bound for any local randomizer {#sec-shuf-clones}

Randomized response on bits is one randomizer among many. In practice users send histograms, answers with many categories or noisy vectors. Each user's randomizer may even be chosen adaptively from earlier outputs, as in shuffled stochastic gradient descent (SGD), where each user randomizes their own gradient before it is shuffled. So you want one bound that holds for every $\varepsilon_0$-DP **local randomizer**. That is any randomized map from one user's data to a report that is $\varepsilon_0$-DP in this sense. No report becomes more than $e^{\varepsilon_0}$ times likelier when the user's data changes from any value to any other. Erlingsson et al. (2019) and Balle et al. (2019) gave earlier bounds. The one to use today is that of Feldman, McMillan & Talwar (2021), and its idea is simple enough to follow without the proof.

Their argument finds copies of your report inside everyone else's. Fix the two values you might hold, one in $D$ and one in $D'$, and call the reports you would send your $D$-version and your $D'$-version. Now take any other user. Because the randomizer is $\varepsilon_0$-DP between any two values, that user's report gives each output at least a fraction $e^{-\varepsilon_0}$ of the chance your $D$-version gives it. The same holds for your $D'$-version, so each output also gets at least $e^{-\varepsilon_0}$ times the average of your two versions' chances. A share $e^{-\varepsilon_0}$ of the other user's report distribution can therefore be set aside as an even mix of your two versions.

That lets the report be generated in two stages. First the user flips a three-sided coin. On the first face, which comes up with probability $e^{-\varepsilon_0}/2$, the report is drawn exactly like your $D$-version. The second face has the same probability and draws exactly like your $D'$-version. Otherwise the report is drawn from a **leftover** distribution that depends only on that user's own data.

A user whose coin lands on one of the first two faces is a **clone** of you. Each of the other $n - 1$ users is a clone with probability $e^{-\varepsilon_0}$, independently of the others, so you can expect about $ne^{-\varepsilon_0}$ clones. The clones are split evenly between your two versions on purpose, to keep the situation symmetric. If they all looked like your $D$-version, a report that looks like your $D'$-version would stand out against them, and spotting it would point to $D'$.

@tbl-shuf-clone-split does the split for binary randomized response with $\varepsilon_0 = \ln 3$, for another user who holds 0, when your two values are 0 and 1. That user's report is a clone of each of your versions with probability $1/6$, and the leftover keeps the remaining $2/3$ of the probability.

| Output $y$ | User's report | Clone of your $D$-version | Clone of your $D'$-version | Leftover part | Leftover, rescaled |
| --- | --- | --- | --- | --- | --- |
| 0 | 3/4 | 1/8 | 1/24 | 7/12 | 7/8 |
| 1 | 1/4 | 1/24 | 1/8 | 1/12 | 1/8 |
| Total | 1 | 1/6 | 1/6 | 2/3 | 1 |

: The clone split for binary randomized response with $\varepsilon_0 = \ln 3$, for a user holding 0. With probability $\frac16$ the user's report is drawn like a report of 0, with probability $\frac16$ like a report of 1, and otherwise from the leftover. {#tbl-shuf-clone-split}

After the shuffle, nobody can tell a clone's report from yours. Imagine someone who knows every other user's data but not your value. That person could produce the whole shuffled bag from just two numbers, the number of reports drawn like each of your two versions. Your own report is counted too, as one of those drawn like the version you hold. They would draw that many reports of each kind, draw a leftover report for every other user, and shuffle. So by the data processing inequality, the shuffled output reveals no more than this pair of counts.

One refinement uses the fact that your own report is noisy too. Restricted to your two values, any $\varepsilon_0$-DP randomizer acts like binary randomized response followed by some processing (Kairouz, Oh & Viswanath 2015). Concretely, your report is drawn from a distribution on the side of your true value with probability $e^{\varepsilon_0}/(e^{\varepsilon_0} + 1)$, and from one on the other side otherwise. Say that your report **points to** whichever of your two versions it was drawn for.

Put together, these steps give Theorem 3.2 of Feldman, McMillan & Talwar. Write $C$ for the number of clones. It is binomial, with $n - 1$ trials and success probability $e^{-\varepsilon_0}$. Of these clones, a number $A$ look like your $D$-version, and each clone does so with probability $1/2$. Finally, write $\Delta$ for a bit that is 1 when your own report points to your true version. The three are drawn independently. Then the shuffled output on $D$ and on $D'$ is a processing, by one and the same procedure, of the pair of counts
$$\tilde P = (A + \Delta,\; C - A + 1 - \Delta), \qquad \tilde Q = (A + 1 - \Delta,\; C - A + \Delta).$$ {#eq-shuf-clone-pair}
The first coordinate counts the reports that look like your $D$-version, and the second those that look like your $D'$-version. Under $D$ your own report adds to the first count when $\Delta = 1$, and under $D'$ it adds to the second.

The exact $\delta(\varepsilon)$ of this pair, computed with the same sum as for randomized response, is the **numeric clone bound**. It holds for every $\varepsilon_0$-DP randomizer, including adaptively chosen ones. It needs one condition, which a deployment must actually enforce. The shuffler's order must be uniformly random and hidden from the analyst. With a fixed or leaked order, the analyst knows which report is yours, and the guarantee falls back to $\varepsilon_0$.

The pair also explains the shape of the guarantee. On average, the first counts of the two distributions differ by
$$2\Pr[\Delta = 1] - 1 = \frac{e^{\varepsilon_0} - 1}{e^{\varepsilon_0} + 1},$$
which is how much more often your report points the right way than the wrong way. The noise that hides this difference is the spread of $A$, a binomial count with about $ne^{-\varepsilon_0}$ trials and probability $1/2$, whose standard deviation is about $\sqrt{ne^{-\varepsilon_0}}/2$. Dividing the difference by that spread gives a $\mu$ that falls like $1/\sqrt n$, and grows like $e^{\varepsilon_0/2}$ because the number of clones shrinks like $e^{-\varepsilon_0}$.

Feldman, McMillan & Talwar also bound the pair in closed form, with Chernoff bounds on the binomial counts ([Concentration inequalities](../part01-prerequisites/02-concentration.md)). Their Theorem 3.1 holds for every $\delta$ between 0 and 1 under one condition, that
$$\varepsilon_0 \le \ln\frac{n}{16\ln(2/\delta)}.$$
When it holds, the shuffled output is $(\varepsilon, \delta)$-DP with
$$\varepsilon \le \ln\left(1 + \frac{e^{\varepsilon_0} - 1}{e^{\varepsilon_0} + 1}\left(\frac{8\sqrt{e^{\varepsilon_0}\ln(4/\delta)}}{\sqrt n} + \frac{8e^{\varepsilon_0}}{n}\right)\right).$$ {#eq-shuf-clone-bound}
The condition asks for enough clones to hide among. It is the same as asking that the expected number of clones, $ne^{-\varepsilon_0}$, be at least $16\ln(2/\delta)$. That keeps the number of clones close to its mean except with probability about $\delta$. With fewer clones expected, the closed form says nothing. At $\delta = 10^{-6}$, the condition caps $\varepsilon_0$ at 1.460 for 1000 users. The cap grows like $\ln n$, so 100,000 users may each use up to 6.066.

| $n$ | $\varepsilon_0$ | Exact, binary randomized response | Normal approx., @eq-shuf-rr-clt | Clones, numeric | Clones, closed form | Erlingsson et al. |
| --- | --- | --- | --- | --- | --- | --- |
| 1,000 | 1 | 0.1266 | 0.1212 | 0.1824 | 0.5662 | — |
| 1,000 | 2 | 0.3233 | 0.2890 | 0.5455 | n/a | — |
| 1,000 | 4 | 1.6910 | 0.9667 | 3.9900 | n/a | — |
| 10,000 | 1 | 0.0357 | 0.0352 | 0.0530 | 0.2140 | — |
| 10,000 | 2 | 0.0870 | 0.0844 | 0.1550 | 0.5009 | — |
| 10,000 | 4 | 0.3146 | 0.2816 | 0.6009 | n/a | — |
| 100,000 | 1 | 0.0101 | 0.0101 | 0.0153 | 0.0726 | 0.4285 |
| 100,000 | 2 | 0.0247 | 0.0244 | 0.0450 | 0.1862 | — |
| 100,000 | 4 | 0.0847 | 0.0822 | 0.1698 | 0.5346 | — |

: Central $\varepsilon$ at $\delta = 10^{-6}$ after shuffling $n$ reports that are each $\varepsilon_0$-DP. "n/a" marks where the condition of @eq-shuf-clone-bound fails, and "—" marks where the bound of Erlingsson et al. is at least $\varepsilon_0$. {#tbl-shuf-compare}

@tbl-shuf-compare shows what generality costs. The numeric clone bound is larger than the exact value for randomized response, because it hides you among fewer reports. In binary randomized response every other report is a bit that could be yours, while the clone argument counts only the fraction $e^{-\varepsilon_0}$ of reports guaranteed to look like yours. Dividing the clone argument's $\mu$ by that of randomized response, as the full edition does, puts the cost at a factor of $2/(1 + e^{-\varepsilon_0})$. That factor is 1.46 at $\varepsilon_0 = 1$, and it grows towards 2 as $\varepsilon_0$ grows.

Wherever many clones are expected, the table's ratios are 1.44 to 2.00, close to that prediction. Where few are expected, the ratio is larger. It reaches 2.36 in the row with 1000 users and $\varepsilon_0 = 4$, where only 18.3 clones are expected. No general bound can go below the exact column, because binary randomized response is one of the randomizers it must cover.

The closed form is looser again, because its Chernoff bounds are safe but not sharp. Where its condition holds, it is 3.10 to 4.75 times the numeric clone bound. Use it to reason about how the guarantee scales, and compute the numeric bound to deploy. The first general bound, by Erlingsson et al. (2019), was much weaker still, because its leading term grows with $\varepsilon_0$ like $e^{3\varepsilon_0}$ rather than $e^{\varepsilon_0/2}$. It amplifies at all only in the table's row with the most users and the smallest $\varepsilon_0$, where it gives 0.4285.

![Central $\varepsilon$ at $\delta = 10^{-6}$ against the number of users $n$, for $\varepsilon_0 = 1, 2, 4$: the closed-form clone bound (drawn only where its condition holds), the numeric clone bound, the bound of Erlingsson et al. (drawn where it is below 50) and the exact $\varepsilon$ of shuffled binary randomized response. The grey line is the local $\varepsilon_0$.](figures/shuffling_iteration_bounds.svg){#fig-shuf-bounds}

@fig-shuf-bounds draws the same comparison over the number of users. Once $n$ is large, all the curves fall like $1/\sqrt n$, and only their constants separate them.

Which accountant you use can decide whether a deployment is feasible. Suppose you collect a bit from 100,000 users behind a shuffler, and you want a central $\varepsilon$ of at most 0.2 at the table's $\delta$. Among the table's values of $\varepsilon_0$, the closed form allows at most 2, where it gives 0.1862. The numeric clone bound allows 4, where it gives 0.1698. For randomized response, the exact value at $\varepsilon_0 = 4$ is only 0.0847.

A larger $\varepsilon_0$ means less noise per report, so it is worth seeing what $\varepsilon_0 = 4$ buys. At that level each report is the truth with probability $\theta = 0.9820$. Suppose a proportion $p$ of the population holds 1. You estimate $p$ from the fraction of 1-reports, after undoing the known rate of flipping, and the estimate's variance is
$$\frac{1}{n}\left(p(1 - p) + \frac{\theta(1 - \theta)}{(2\theta - 1)^2}\right).$$
The first term is the ordinary sampling variance, which you would pay without any privacy. The second is the cost of the noise, enlarged by the division by $2\theta - 1$ that undoes the flipping. At this $\theta$ the noise term is only 0.0190, while the sampling term can be as large as 0.25, so the noise costs little accuracy.

## Amplification by iteration {#sec-shuf-iteration}

Projected noisy SGD makes one pass over the data. At each step, numbered $t$, it takes a gradient step on the record $x_t$, adds Gaussian noise, and projects the parameters back onto a convex set of allowed values. If every intermediate model, or **iterate**, is released, the record $x_t$ is protected only by the noise of its own step, as in a single Gaussian mechanism. If only the final iterate is released, the noise of every later step also covers it. Feldman, Mironov, Talwar & Thakurta (2018) turned this into a guarantee. The obstacle is that the later steps are not just additions of noise, so you cannot simply add up noise variances as you would for a sum of Gaussians.

The argument needs every step to be a **contraction**, a map that never pushes two points further apart. Without that, a step could stretch a small gap between two runs faster than the noise could hide it. Noisy SGD uses two kinds of maps, and each is a contraction under conditions on the loss or the set. The first is a gradient step ([Stability and generalization](../part01-prerequisites/07-stability-generalization.md)) on a convex loss that is $\beta$-smooth, meaning that its gradient changes by at most $\beta$ per unit of distance. Such a step is a contraction when the step size $\eta$ is at most $2/\beta$. The reason is that on a convex, smooth loss, the gradients at two points differ in a direction that pulls the points together, and a short enough step never overshoots.

The second kind is projection onto a closed convex set, which moves each point to the nearest point of the set. It is a contraction because the set is convex. The nearest point lies on the near side of the set, so two projected points are never further apart than the originals ([Optimization](../part01-prerequisites/06-optimization.md)). One contraction followed by another is still a contraction, so a gradient step followed by a projection is one too.

Now follow two runs of training, one on $D$ and one on $D'$. They agree until the step that uses your record. At that step the two runs apply different updates, which can send the same point to places up to some distance $s$ apart, so the runs are pushed apart by at most $s$. After that, the contractions keep the gap from growing, while the noise blurs it. Feldman et al. call the largest distance by which the runs might be apart the **shift**.

To measure privacy while the runs are apart, they use a **shifted divergence**. It compares the two runs' distributions after you have been allowed to move each point of one run by at most the shift, choosing the moves that make the two distributions as close as possible. The comparison uses the Rényi divergence of order $\alpha$, which defines Rényi differential privacy (RDP) ([Rényi DP](01-renyi-dp.md)). With no move allowed, the shifted divergence is the ordinary Rényi divergence, which is the privacy of the run so far.

Two facts make the argument work. First, each noisy step lets you trade some of the allowed shift for divergence, which Feldman et al. call paying off the shift. Lowering the allowed shift by an amount $a$ adds to the divergence the Rényi divergence between the noise and a copy of it moved by $a$. That price is small, because a small move hidden in noise is hard to see. For Gaussian noise with standard deviation $\sigma$, it is
$$R_\alpha(a) = \frac{\alpha a^2}{2\sigma^2}.$$ {#eq-shuf-gauss-cost}

Second, contractions cost nothing, because they never move points apart, so any move that matched the runs before a step still works after it. At the step that uses your record, the allowed shift must grow by $s$, but the divergence does not grow. By the end of the run, the whole shift must have been paid off, because only a shift of zero leaves the ordinary divergence, which is the privacy of the final iterate.

Since the price is quadratic, small equal payments are cheapest. Paying a gap $s$ in $m$ equal instalments of $s/m$ costs
$$m \cdot \frac{\alpha (s/m)^2}{2\sigma^2} = \frac{\alpha s^2}{2m\sigma^2},$$
which is $m$ times less than paying it all at once. For example, take a gap of 1 and noise with standard deviation 1. At order $\alpha = 2$, each payment then simply costs its square. Spread over four steps, the even schedule pays a quarter of the gap at each step, for a total cost of 0.25. That equals the exact divergence, because four noisy steps add up to a variance of 4, and hiding a shift of 1 under that variance costs exactly 0.25. Uneven schedules cost more, because squares punish large payments. For example, paying 0.8 and then 0.2 costs 0.68. Paying everything at once costs the most, 1 (@fig-shuf-shift).

![Left: the even schedule that pays off a shift over $T = 4$ noisy steps. Each red arrow is one payment of a quarter of the shift, and a payment $a$ costs $\alpha a^2/(2\sigma^2)$. Right: the total cost of the even schedule over $T$ steps (dots) against the exact Rényi divergence of order 2 between $\mathcal N(0, T)$ and $\mathcal N(1, T)$ (line), for a shift of 1 and $\sigma = 1$. The dotted line is the cost of paying the whole shift in one step.](figures/shuffling_iteration_shift.svg){#fig-shuf-shift}

For noisy SGD this gives Theorem 23 of Feldman et al. (2018). It needs convex, $\beta$-smooth losses with $\eta \le 2/\beta$, and it needs the losses to be $L_f$-Lipschitz, which means that every gradient has a norm of at most $L_f$. The gradients on your two possible records then differ by at most $2L_f$, which is the gradient's replace-one sensitivity. Write $\tilde\sigma$ for the **noise multiplier**, the noise's standard deviation per unit of that sensitivity. Releasing only the final iterate of one pass over $n$ records then gives the record used at step $t$ the RDP curve
$$\varepsilon_t(\alpha) = \frac{\alpha}{2\tilde\sigma^2}\cdot\frac{1}{n - t + 1}.$$ {#eq-shuf-fmtt}
This is one Gaussian step's curve, divided by the number of noisy steps from $t$ to the end. The noise enters the parameters multiplied by the step size, just like the gradient, so the step size scales the shift and the noise alike and cancels.

The theorem is exact in the worst case over contractions. That worst case is a step that uses your record right after mapping every point to a single point, which erases all the earlier noise. For typical runs it is loose, because it ignores two things that also hide your record. Noise added before step $t$ also helps when the later maps keep the shift as it is, because it reaches the final iterate along with the later noise. And a **strict** contraction, one that brings points closer together, shrinks the shift for free.

@fig-shuf-positions shows both effects on a one-dimensional run of 100 steps. Each step multiplies the parameter by a contraction factor $c$, then adds an offset and Gaussian noise. Your record changes only the offset of the step that uses it, by one noise standard deviation. The full edition computes the exact $\mu$ of the final iterate, in GDP terms, and in those terms Theorem 23 gives $\mu = 1/\sqrt{100 - t + 1}$. With $c = 1$, every later step keeps the shift as it is, so it persists to the end and all 100 noise draws pile up on top of it. The exact $\mu$ is then $1/\sqrt{100} = 0.1000$ at every position, while the bound rises to 1 for the last record. With a strict contraction by a factor of 0.95 per step, the shift also decays, so the exact $\mu$ is far smaller still. In the middle of the run it is 0.0240, against the bound's 0.1400.

![The $\mu$ of the last iterate against the step $t$ at which your record is used, for a one-pass run of $T = 100$ steps in which each step multiplies the parameter by a contraction factor $c$, and your record shifts its own step's offset by one noise standard deviation ($s/\sigma = 1$). Releasing all iterates gives $\mu = 1$ for every record (dotted). Theorem 23 (red, as $\mu = 1/\sqrt{T - t + 1}$) is exact when the run is reset just before step $t$; with $c = 1$ (blue) the earlier noise also helps, and with a strict contraction $c = 0.95$ (orange) the shift itself decays.](figures/shuffling_iteration_positions.svg){#fig-shuf-positions}

Early records are safer than late ones, and the last record is protected by one step only. A random stopping time spreads the protection out. Suppose the number of steps is drawn uniformly from 1 to $n$. Then Theorem 26 of the same paper gives every record the RDP curve $\alpha\ln n/(n\tilde\sigma^2)$, provided $\sigma$ is at least $L_f\sqrt{2\alpha(\alpha - 1)}$. That is $2\ln n$ times the curve of the first record in @eq-shuf-fmtt. The theorems cover one pass in a fixed order, with each record used once. Several passes need the general form of the argument, which pays off a new shift at each use of the record, and Altschuler & Talwar (2022) combine iteration with random sampling.

Here is a one-pass run over 10,000 records with noise multiplier 1. To get a guarantee at $\delta = 10^{-5}$, each record's curve goes through Mironov's conversion ([Rényi DP](01-renyi-dp.md#sec-rdp-conversion)). That conversion minimizes $\varepsilon(\alpha) + \ln(1/\delta)/(\alpha - 1)$ over the part's grid of orders, a fixed list of 154 orders from 1.1 to 512. Because the curve depends on position, so does the answer. The last record gets $\varepsilon = 5.2985$, the same as a single Gaussian step, and this is also what every record gets if all iterates are released. Records further from the end are much better hidden. For example, the record used 100 steps before the end gets 0.4849. The record in the middle gets just 0.0707, because the whole later half of the run's noise covers it.

The first record gets 0.0481, at the largest order of the grid. Its RDP curve is so small that the grid of orders is too short for it, and a finer conversion from [From RDP to (ε, δ)](02-rdp-conversions.md) would give less. But the RDP conversion is not tight at the other end either. A single Gaussian step with noise multiplier 1 has the exact $\varepsilon$ of 4.3772, below the 5.2985 that RDP gives. Since the guarantee depends on position, a report must state the worst record's value, or randomize the order and the stopping time.

## Hidden state: privacy that stops growing {#sec-shuf-hidden}

Full-batch noisy GD uses every record at every step. Composition charges every step, so the RDP of $T$ steps grows in proportion to $T$. Amplification by iteration does not help by itself, because every step adds a fresh shift $s$. The payments must then cover $Ts$ in total, and the cheapest schedule pays $s$ at every step, which is composition again. But suppose the parameters live in a convex set with **diameter** $d_K$, the largest distance between two of its points. Then the two runs can never be further apart than $d_K$, however long they run. So the shift never needs to exceed $d_K$, and only the last stretch of steps has to pay it off.

Altschuler & Talwar (2022, Theorem 3.1) prove that the privacy loss of the last iterate then stops growing. Their conditions are convex, $L_f$-Lipschitz, $\beta$-smooth losses on a convex set of diameter $d_K$, with $\eta \le 2/\beta$. They also need enough noise for the batch size, and orders $\alpha$ up to a threshold. The theorem covers noisy SGD with any batch size, full-batch GD included. Up to a constant factor, their bound on the RDP of the last iterate over $n$ records grows like composition for about
$$\bar T = \left\lceil\frac{d_K n}{L_f\eta}\right\rceil$$
steps, and then stays flat. The number of records $n$ appears because the gradient is averaged over the records, so replacing one record moves each step by at most $2\eta L_f/n$. Their Theorem 4.1 shows that this shape is tight up to constants.

The full edition specializes their key lemma to full-batch noisy GD, with a per-step shift $s$ and per-step noise $\sigma$, both measured in parameter units. Costs below are in units of $\alpha/(2\sigma^2)$. The accounting can start at the beginning, where both runs start at the same point. Each step then pays off its own shift, which is composition and costs $Ts^2$. Or the accounting can start $m$ steps before the end, when the two runs are at most $d_K$ apart. The last $m$ steps must then pay off that distance plus the $m$ new shifts, $d_K + ms$ in total, which costs $(d_K + ms)^2/m$ when paid evenly.

The choice of $m$ is a trade-off, because a short window means few, large payments, while a long window means more new shift to pay. The cost is lowest at $m = d_K/s$, which balances these two effects, and there it equals $4d_Ks$, whatever the number of steps $T$. Taking the better of the two starting points, and the best window, gives
$$\varepsilon(\alpha) \le \frac{\alpha}{2\sigma^2}\min\left\{Ts^2,\ \min_{m \in \{1, \dots, T\}}\frac{(d_K + ms)^2}{m}\right\} \le \frac{2\alpha d_Ks}{\sigma^2}\quad\text{when } d_K/s \text{ is an integer}.$$ {#eq-shuf-plateau}
The composition branch catches up with the plateau after $4d_K/s$ steps, and from then on the bound no longer grows.

For an example, take a linear loss on the unit interval, whose diameter is 1. Let each step shift the runs by $s = 0.01$ and add noise with $\sigma = 0.1$. At $\alpha = 2$, each composed step then costs 0.01, so composition charges 20.0000 for 2000 steps. The bound of @eq-shuf-plateau, by contrast, stops at its plateau of 4.0000, which it reaches at 400 steps. To see how loose that is, the full edition computes the exact divergence for two random walks that drift in opposite directions and are kept inside the interval. Their divergence flattens at 0.3786 after about 100 steps (@tbl-shuf-plateau). Because the walks are one pair of datasets, 0.3786 is a lower bound on the worst case, so the true worst case lies between it and the bound. The shape of the bound is right, and its constant is at most 10.57 times too large.

| Steps $T$ | 10 | 100 | 400 | 1000 | 2000 |
| --- | --- | --- | --- | --- | --- |
| Composition, $\alpha Ts^2/(2\sigma^2)$ | 0.1000 | 1.0000 | 4.0000 | 10.0000 | 20.0000 |
| Bounded-domain bound, @eq-shuf-plateau | 0.1000 | 1.0000 | 4.0000 | 4.0000 | 4.0000 |
| Projected random walks, numerical | 0.0961 | 0.3693 | 0.3786 | 0.3786 | 0.3786 |

: Rényi divergence of order 2 of the last iterate of full-batch noisy GD on a linear loss over $[0, 1]$, with per-step shift $s = 0.01$ and noise $\sigma = 0.1$, by composition, by the bounded-domain bound, and computed numerically for the two projected random walks. {#tbl-shuf-plateau}

Strong convexity gives a plateau without a bounded set. A loss is **$\lambda$-strongly convex** when, as a function of the parameters $w$, it curves upward at least as fast as $\frac\lambda2\|w\|^2$. Each gradient step is then a strict contraction by the factor $1 - \eta\lambda$, so the shift shrinks every step even when nobody pays for it. An old shift has faded after about $1/(\eta\lambda)$ steps, so the plateau arrives then, instead of after $\bar T$ steps. Chourasia, Ye & Shokri (2021) had shown earlier, by a different route, that the RDP of the last iterate of full-batch noisy GD levels off in this case. Ye & Shokri (2022) then extended their result to mini-batch SGD.

These results rest on two conditions. The first is that **only the last iterate is released**. Checkpoints, logged intermediate models or a model released after each epoch break the argument, because the adversary then sees the run before it has forgotten. That puts you back to composition. The same holds if you publish the best of several checkpoints, because the choice depends on all of them.

The second condition is that every step is a contraction, which requires a convex loss. For non-convex losses, Annamalai (2024) constructs a loss for which the last iterate of differentially private SGD (DP-SGD) leaks as much as all iterates together. So no hidden-state amplification holds for every non-convex loss. Nasr et al. (2021) found that their strongest attacks match the DP-SGD bound when the adversary sees every step or can craft gradients. When the adversary sees only the final model, their attacks fall short of the bound, but that gap is an observation about their attacks, not a guarantee. So do not report a hidden-state $\varepsilon$ for deep learning.

## An exact example: noisy GD on a quadratic {#sec-shuf-quadratic}

All the results above are bounds. In one dimension, with a quadratic loss, the privacy of the last iterate can be computed exactly. The computation shows the plateau directly, shows that it needs contraction, and gives a yardstick for the strongly convex bounds.

Take a model with a single parameter $w$. Give each record $x$ the quadratic loss $\frac\lambda2(w - x)^2$, whose curvature is set by $\lambda$. Full-batch noisy GD then pulls $w$ towards the mean of the records, which we write $\bar x$. Each step first shrinks the parameter by a **contraction factor** $c$, which equals $1 - \eta\lambda$ for a step size $\eta$. It then adds an offset that depends on $\bar x$, and Gaussian noise with standard deviation $\sigma$. Replacing your record changes $\bar x$, and so it changes every step's offset by the same amount $s$. A record may also join the training set while training is under way. If it enters at step $t$ and stays, the offsets on the two datasets agree before step $t$ and differ by $s$ from then on. Unlike the one-pass run of @fig-shuf-positions, the record here changes every step from $t$ on.

Unrolling the run, the last iterate is a weighted sum of all offsets and all noise draws, with the step-$k$ term weighted by $c^{T - k}$. So it is Gaussian on both datasets, with the same variance. Its $\mu$ is the weighted sum of the offset differences, divided by the standard deviation of the weighted noise. The last iterate of a run of $T$ steps is therefore exactly $\mu_T$-GDP, where
$$\mu_T = \frac{s}{\sigma}\cdot\frac{\sum_{k=t}^{T}c^{T-k}}{\sqrt{\sum_{j=0}^{T-1}c^{2j}}}, \qquad\text{and for } t = 1,\ c < 1:\qquad \lim_{T\to\infty}\mu_T = \frac{s}{\sigma}\sqrt{\frac{1 + c}{1 - c}}.$$ {#eq-shuf-last-iterate}
Releasing all iterates instead is GDP with $\mu = \sqrt{T - t + 1}\,s/\sigma$, the composition of one Gaussian step per step that sees your record.

Contraction is what forgets. With $c < 1$, the offset of each step reaches the end damped, while fresh noise keeps arriving. So the signal saturates at $s/(1 - c)$, and the noise's standard deviation saturates at $\sigma/\sqrt{1 - c^2}$. With $c = 1$ both keep growing, the signal like $T$ and the noise only like $\sqrt T$. Then $\mu$ grows like $\sqrt T$, exactly as under composition, and hiding the state buys nothing (@fig-shuf-hidden, left).

The plateau arrives after about $1/(1 - c)$ steps, which is how long it takes $c^T$ to become small. To see this, let the shift equal the noise's standard deviation, and compare two contraction factors. With $c = 0.9$ the run reaches its plateau within 100 steps, where $\mu$ is 4.3588, against a limit of 4.3589. A contraction factor closer to 1 forgets more slowly. With $c = 0.99$, $\mu$ has reached only 9.6101 after 100 steps, and its limit is 14.1067. It takes about 1000 steps to get close, to 14.1061. On a $\lambda$-strongly convex, $\beta$-smooth loss with step size $1/\beta$, the contraction factor is $1 - \lambda/\beta$. So $1/(1 - c)$ is the **condition number** $\beta/\lambda$, the ratio of the loss's largest to smallest curvature. That is the threshold, up to logarithmic factors, that Altschuler & Talwar give for strongly convex losses.

Recent records are hardly hidden. Take a run of 50 steps with $c = 0.9$, again with the shift equal to the noise's standard deviation, and a record that joins at step 41. From the final model, that record gets $\mu = 2.8391$. Releasing all of its 10 steps would give 3.1623, only a little more, because the record's signal lies in the last few steps and little earlier noise can hide it.

The example also measures the strongly convex bound of Chourasia, Ye & Shokri. Translated into this section's notation, their bound levels off at an RDP of
$$\frac{2\alpha s^2}{(1 - c)\sigma^2}.$$
The exact limit is the RDP of a $\mu$-GDP pair, $\alpha\mu^2/2$, with $\mu$ the limit in @eq-shuf-last-iterate, and it comes to $(1 + c)/4$ times the bound. So when $c$ is close to 1, the bound is about twice the exact value. At $c = 0.99$ and $\alpha = 2$, with the shift equal to the noise's standard deviation, the exact limit is 199.0000 and the bound is 400.0000.

![Left: the exact $\mu$ of the last iterate of one-dimensional full-batch noisy GD with per-step shift equal to the noise standard deviation, against the number of steps $T$, for $c = 1$ (dashed, on top of the black composition line), $c = 0.99$ and $c = 0.9$. Right: the Rényi divergence of order 2 of the last iterate for a linear loss on $[0, 1]$ with $s = 0.01$ and $\sigma = 0.1$ per step: composition, the bounded-domain bound @eq-shuf-plateau, and the numerical value for two projected random walks.](figures/shuffling_iteration_hidden_state.svg){#fig-shuf-hidden}

Here is the difference in a realistic run. Take 10,000 steps of full-batch noisy GD on a quadratic with $c = 0.99$, with noise ten times the per-step shift. Each released iterate is then a Gaussian step with $\mu = 0.1$. Composing 10,000 of them multiplies that by $\sqrt{10{,}000} = 100$, so releasing every iterate is 10-GDP. At $\delta = 10^{-5}$ that is $\varepsilon = 91.8173$, which is no guarantee at all. After this many steps, the final model alone is at the limit of @eq-shuf-last-iterate, which is 1.4107-GDP, or $\varepsilon = 6.5534$ at the same $\delta$. The noise is the same in both cases, and the difference lies entirely in what the adversary sees. This part's running example of DP-SGD gets its small $\varepsilon$ from subsampling instead, under add/remove neighbours.

## Code {#sec-shuf-code}

This standard-library snippet builds @eq-shuf-rr-pair in log space, takes the larger hockey-stick divergence over both directions, and bisects for $\varepsilon$:

```python
from math import exp, lgamma, log


def log_binom_pmf(k: int, n: int, p: float) -> float:
    return lgamma(n + 1) - lgamma(k + 1) - lgamma(n - k + 1) + k * log(p) + (n - k) * log(1 - p)


def shuffled_rr_pair(n: int, eps0: float) -> tuple[list[float], list[float]]:
    """Distribution of the number of 1-reports when you hold 0 (P) or 1 (Q); the n - 1 others hold 0."""
    theta = exp(eps0) / (1 + exp(eps0))
    b = [exp(log_binom_pmf(y, n - 1, 1 - theta)) for y in range(n)] + [0.0]
    P = [theta * b[y] + (1 - theta) * (b[y - 1] if y else 0.0) for y in range(n + 1)]
    Q = [(1 - theta) * b[y] + theta * (b[y - 1] if y else 0.0) for y in range(n + 1)]
    return P, Q


def hockey_stick(P: list[float], Q: list[float], eps: float) -> float:
    return sum(max(p - exp(eps) * q, 0.0) for p, q in zip(P, Q))


def shuffled_rr_epsilon(n: int, eps0: float, delta: float) -> float:
    P, Q = shuffled_rr_pair(n, eps0)
    lo, hi = 0.0, eps0  # shuffling never does worse than the local eps0
    for _ in range(50):
        mid = (lo + hi) / 2
        worst = max(hockey_stick(P, Q, mid), hockey_stick(Q, P, mid))
        lo, hi = (mid, hi) if worst > delta else (lo, mid)
    return hi


print(f"{shuffled_rr_epsilon(10_000, 2.0, 1e-6):.4f}")  # 0.0870
print(f"{shuffled_rr_epsilon(100_000, 4.0, 1e-6):.4f}")  # 0.0847
print(f"{shuffled_rr_epsilon(1_000, 4.0, 1e-6):.4f}")  # 1.6910
```

It reproduces the exact column of @tbl-shuf-compare. The full edition's script, [shuffling_iteration.py](../../code/part04-privacy-accounting/shuffling_iteration.py), also computes the numeric clone bound and every iteration result:

```bash
uv run python code/part04-privacy-accounting/shuffling_iteration.py
```

## Cheat sheet

| Result | Statement | Use |
| --- | --- | --- |
| Shuffled binary randomized response | Count pair @eq-shuf-rr-pair; $\mu \approx 2\sinh(\varepsilon_0/2)/\sqrt n$ | Compute exactly; the approximation fails with few 1-reports |
| Numeric clone bound | Exact $\delta(\varepsilon)$ of the count pair @eq-shuf-clone-pair | Deploy with it for any $\varepsilon_0$-DP randomizer; 1.44–2.36 times the exact value for randomized response |
| Closed-form clone bound | @eq-shuf-clone-bound, needs $\varepsilon_0 \le \ln\frac{n}{16\ln(2/\delta)}$ | Scaling only; 3.10–4.75 times the numeric bound |
| One-pass noisy SGD, last iterate | $\varepsilon_t(\alpha) = \frac{\alpha}{2\tilde\sigma^2(n - t + 1)}$ | Convex, smooth, Lipschitz losses, $\eta \le 2/\beta$; report the last record's value |
| Bounded domain | $\varepsilon(\alpha) \le \frac{2\alpha d_Ks}{\sigma^2}$ for full-batch GD ($d_K/s$ an integer) | Right shape; constant at most about 10 times too large in the example |
| Quadratic, exact | $\mu_T$ of @eq-shuf-last-iterate; limit $\frac s\sigma\sqrt{\frac{1+c}{1-c}}$ | Yardstick; plateau after about $1/(1 - c)$ steps |

: The results of this chapter. All use replace-one neighbours. {#tbl-shuf-cheat}

**Rules of thumb**

- Shuffling turns $\varepsilon_0$-local reports into a central $\varepsilon$ that falls like $1/\sqrt n$, with a constant that grows like $e^{\varepsilon_0/2}$. It never lowers the pure-DP level.
- For binary randomized response, compute the exact value. For other randomizers, compute the numeric clone bound, and use the closed form only to reason.
- Amplification by iteration needs contractive steps and a hidden state. The record used last gets no amplification.
- With a bounded domain or strong convexity, the last iterate's privacy plateaus. Without either, it grows like composition.
- Never claim hidden-state privacy for non-convex training or when checkpoints are released.

## Further reading

- Feldman, McMillan & Talwar, "Hiding Among the Clones: A Simple and Nearly Optimal Analysis of Privacy Amplification by Shuffling" (FOCS 2021): the clone reduction and bound, and the numerical method to compute it.
- Feldman, Mironov, Talwar & Thakurta, "Privacy Amplification by Iteration" (FOCS 2018): the shifted Rényi divergence and the one-pass noisy SGD guarantee.
- Altschuler & Talwar, "Privacy of Noisy Stochastic Gradient Descent: More Iterations without More Privacy Loss" (NeurIPS 2022): the bounded-domain plateau, with a matching lower bound.
- Annamalai, "It's Our Loss: No Privacy Amplification for Hidden State DP-SGD With Non-Convex Loss" (AISec 2024): why the hidden-state results do not extend to deep learning.

**Derivations.** The full edition derives the exact count pair, the clone reduction, the iteration bounds and the exact last-iterate privacy on a quadratic step by step in [its chapter](../../book-full/part04-privacy-accounting/07-shuffling-iteration.md). It cites, without deriving, the closed-form clone bound and the bound of Erlingsson et al., the general shift-reduction and contraction lemmas, the random-stopping-time bound, Altschuler & Talwar's theorem and lower bound, and the strongly convex bounds of Chourasia, Ye & Shokri.
