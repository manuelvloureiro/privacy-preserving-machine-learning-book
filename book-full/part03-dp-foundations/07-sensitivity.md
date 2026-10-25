# Beyond global sensitivity

## Why this matters

The [Laplace and Gaussian mechanisms](03-laplace-gaussian.md) answer a numeric query $f$ by adding random noise to the true answer, and they set the size of that noise by the **global sensitivity**: the most that changing one record can move $f$, looking at *every* pair of neighbouring datasets that could exist, not just at yours. In symbols, $\Delta f = \max_{D \simeq D'}|f(D) - f(D')|$, where $D \simeq D'$ means the two datasets are neighbours. For counts and bounded sums this is the right scale: one person changes a count by at most 1, whatever the data. For many other statistics it is hopeless, for three different reasons.

The first is that the worst case can be absurd. The **median** of ages in $[0, 100]$ has global sensitivity 100. To see it, take a dataset in which half the people are 0 and half are 100, with one more person at 0 so that the median is 0. Replacing that one person by a 100-year-old moves the median from one end to the other. So the Laplace mechanism must add noise of scale $100/\varepsilon$, which swamps any real answer, even though no realistic dataset of ages looks like this. The second is that the worst case can be infinite. The **mean of unbounded data** (incomes, session lengths, gradient norms) can be moved arbitrarily far by one record, because that record can be arbitrarily large, so no amount of Laplace noise makes it private. The trouble goes deeper than the Laplace mechanism. Under the strict form of differential privacy, with no slack, no mechanism at all can estimate the mean of arbitrary unbounded data with a finite worst-case error, because infinitely many datasets with means far apart all lie within $n$ changes of each other, which is the packing argument of [Lower bounds](10-lower-bounds.md). What rescues the mean is an assumption about how the data were drawn, and [Private mean estimation](11-mean-estimation.md) uses one to find the range of the data privately. The third is that the worst case can be unknown. For a **statistic you can only compute**, such as an estimator fitted by an algorithm, nobody knows how to bound how far one record moves it.

Yet on the dataset you actually hold, the median of 1,001 ages typically moves by a tenth of a year when one person changes, because with that many people the ages near the middle are packed tightly together. That is the **local sensitivity**, and it is hundreds of times smaller than the global one. This chapter asks: **can you add noise on the scale of the data you have, rather than the worst dataset there could be, and still satisfy differential privacy (DP)?**

Recall what DP asks for. A randomized mechanism $M$ is $(\varepsilon, \delta)$-DP if, for every pair of neighbouring datasets $D \simeq D'$ and every set $S$ of outputs, $\Pr[M(D) \in S] \le e^\varepsilon\Pr[M(D') \in S] + \delta$: one record can change the probability of any outcome by at most a factor $e^\varepsilon$, up to a small slack $\delta$. With $\delta = 0$ this is **pure** $\varepsilon$-DP. The answer to the question is yes, with one idea that organizes the whole chapter:

> **The noise scale is itself a function of the data, so it must be private too.**

If you calibrate noise to the local sensitivity of your dataset, the *size* of the noise reveals something about the dataset: an observer who sees a release with very wide noise learns that the data were of the kind that has a large local sensitivity. That is enough to break the guarantee. Each technique of this chapter repairs this in a different way:

- **Smooth sensitivity** replaces the local sensitivity by an upper bound that changes slowly between neighbours, so that the noise scale leaks little.
- **Propose-test-release** proposes a noise scale, and privately tests whether the dataset is far from any dataset where that scale would be too small.
- **Inverse sensitivity** drops the noise scale altogether: it samples each candidate answer with a probability that falls exponentially in the number of records you would have to change to make it the true answer.
- **Stability-based release** handles statistics with discrete answers: it tests privately that no few changes can flip the answer, and then releases the answer exactly.
- **Subsample-and-aggregate** makes the sensitivity small by construction: run the statistic on disjoint blocks and privately combine the block answers, each of which one record can affect only once.
- **Clipping** forces bounded sensitivity on unbounded data, at the price of a bias that you trade against the noise.

@tbl-sens-uses lists where each result is used.

| Result | What it gives you | Where it is used |
| --- | --- | --- |
| Local sensitivity and its leak | Why "noise proportional to how sensitive *this* dataset is" fails | The motivation for everything below |
| Smooth sensitivity and admissible noise | Data-dependent noise with a proof: Cauchy-like noise for $\varepsilon$-DP, Laplace for $(\varepsilon,\delta)$-DP | Private medians, quantiles and robust statistics |
| Smooth sensitivity of the median | An exact $O(n^2)$ formula | Private quantiles; choosing clipping norms |
| Propose-test-release | Release with small noise if the data are stable, refuse otherwise | Robust statistics (Dwork & Lei 2009) |
| Inverse sensitivity | Pure DP with no noise scale: the exponential mechanism with "changes needed" as the penalty | Private medians and quantiles; choosing clipping norms |
| Stability-based release | An exact answer for a discrete statistic when it is stable | The most common value; private selection on top of subsample-and-aggregate; stability-based histograms |
| Subsample-and-aggregate | Privacy for a black-box statistic | Private model training by ensembles of teachers; estimators without known sensitivity |
| Clipping | Bounded sensitivity for unbounded data, and the bias–variance trade-off of the clipping norm | Private means and sums; per-example gradient clipping in private training |

: The results of this chapter, what each gives you, and where it is used. {#tbl-sens-uses}

Throughout, datasets have a fixed size $n$ and $D \simeq D'$ means that $D'$ is obtained from $D$ by **replacing** one record (the bounded relation of [The definition of differential privacy](01-definition.md)). The **Hamming distance** $d(D, D')$ counts the records in which two datasets differ, so neighbours are datasets at distance 1, and $d(D, D'') = k$ means you can walk from $D$ to $D''$ by changing one record at a time in $k$ steps. Every proof below compares the output densities $p_{M(D)}$ and $p_{M(D')}$ of a mechanism on two neighbours, point by point. The natural quantity for that is the log of their ratio at an output $y$, the **privacy loss**: it says how much more likely $y$ is under $D$ than under $D'$, and pure $\varepsilon$-DP holds when it never exceeds $\varepsilon$. In symbols, $L_{D,D'}(y) = \ln\frac{p_{M(D)}(y)}{p_{M(D')}(y)}$.

## Toolkit: four facts every derivation uses

### Fact 1 — The Laplace mechanism with global sensitivity

If one record never moves $f$ by more than $\Delta$, then adding Laplace noise of scale $\Delta/\varepsilon$ is enough. Formally, if $|f(D) - f(D')| \le \Delta$ for all $D \simeq D'$, then $f(D) + \mathrm{Lap}(\Delta/\varepsilon)$ is $\varepsilon$-DP. This is the [Laplace mechanism](03-laplace-gaussian.md), and the reason is short. The Laplace density with centre $\mu$ and scale $b$ is $\frac1{2b}e^{-|y - \mu|/b}$, so with $b = \Delta/\varepsilon$ the normalizing constants cancel in the ratio of the two densities, and the privacy loss is $\big(|y - f(D')| - |y - f(D)|\big)\varepsilon/\Delta$. The triangle inequality says $|y - f(D')| \le |y - f(D)| + |f(D) - f(D')|$, so the bracket is at most $|f(D) - f(D')| \le \Delta$, and the loss is at most $\varepsilon$ at every $y$. A density that is at most $e^\varepsilon$ times another at every point has at most $e^\varepsilon$ times its probability on every set (integrate over the set), which is $\varepsilon$-DP. Every technique in this chapter ends by applying this fact, or a cousin of it, to something whose sensitivity *is* bounded.

### Fact 2 — A tail bound on the privacy loss gives $(\varepsilon, \delta)$

Some mechanisms in this chapter have a privacy loss that is at most $\varepsilon$ for almost every output, but not for all of them. This fact says that is enough for $(\varepsilon, \delta)$-DP, with $\delta$ the probability of the rare outputs where the loss is too large. For every set $S$ of outputs,
$$\Pr[M(D) \in S] \le e^\varepsilon \Pr[M(D') \in S] + \Pr_{y \sim M(D)}\big[L_{D,D'}(y) > \varepsilon\big].$$ {#eq-sens-loss-tail}

The proof splits $S$ into the outputs where the loss is small and those where it is large: $S_1 = S \cap \{L \le \varepsilon\}$ and $S_2 = S \cap \{L > \varepsilon\}$. On $S_1$ the loss is at most $\varepsilon$, which says exactly that the densities satisfy $p_{M(D)} \le e^\varepsilon p_{M(D')}$ there, so integrating over $S_1$ gives $\Pr[M(D) \in S_1] \le e^\varepsilon\Pr[M(D') \in S_1] \le e^\varepsilon\Pr[M(D') \in S]$. On $S_2$ you give up comparing and just bound the probability: $\Pr[M(D) \in S_2]$ is at most the probability that the loss exceeds $\varepsilon$ at all, $\Pr_{M(D)}[L > \varepsilon]$. Adding the two pieces gives @eq-sens-loss-tail.

So if the privacy loss exceeds $\varepsilon$ only with probability $\delta$, the mechanism is $(\varepsilon, \delta)$-DP. This "bad event" argument is behind every $\delta$ in this chapter.

### Fact 3 — Distances to a set have sensitivity 1

A question of the form "how many records would have to change before something bad happens?" has an answer that one record can change by at most 1. Formally, for any set $\mathcal B$ of datasets (the "bad" ones), let $\operatorname{dist}(D, \mathcal B) = \min_{D'' \in \mathcal B} d(D, D'')$ be the number of changes needed to reach the nearest of them. Then $|\operatorname{dist}(D, \mathcal B) - \operatorname{dist}(D', \mathcal B)| \le 1$ for $D \simeq D'$.

The reason is that a path from $D'$ to a bad dataset can always go through $D$ first, at the cost of one extra step. In terms of the Hamming distance, which satisfies the triangle inequality, $d(D', D'') \le d(D', D) + d(D, D'') = 1 + d(D, D'')$ for every $D''$. Minimizing both sides over $D'' \in \mathcal B$ gives $\operatorname{dist}(D', \mathcal B) \le 1 + \operatorname{dist}(D, \mathcal B)$, and the same argument with $D$ and $D'$ exchanged gives the other direction. So such a count can be released with $\mathrm{Lap}(1/\varepsilon)$ noise by Fact 1. Propose-test-release, inverse sensitivity and the stability-based release are all built on it.

### Fact 4 — Changing $k$ records moves each order statistic by at most $k$ places

If you change $k$ records, the $j$-th smallest value can move, but only as far as the $(j-k)$-th or $(j+k)$-th smallest value of the original data. To state it, sort the values of $D$ as $x_1 \le \dots \le x_n$ (the **order statistics**), all in $[0, \Lambda]$. Ranks $j - k$ or $j + k$ can fall outside $1, \dots, n$, so pad with $x_i = 0$ for $i \le 0$ and $x_i = \Lambda$ for $i > n$: "below the smallest" means "possibly as low as 0", and "above the largest" means "possibly as high as $\Lambda$". If $D''$ differs from $D$ in at most $k$ records and has sorted values $y_1 \le \dots \le y_n$, then
$$x_{j-k} \le y_j \le x_{j+k} \quad\text{for every } j.$$

For the upper bound, count the values below $x_{j+k}$. At least $j + k$ values of $D$ are $\le x_{j+k}$ (when $j + k > n$ the padding sets $x_{j+k} = \Lambda$, which bounds every value, so the claim is trivially true). Changing $k$ records removes at most $k$ of them, so at least $j$ values of $D''$ are still $\le x_{j+k}$, and that says the $j$-th smallest value of $D''$ is at most $x_{j+k}$: $y_j \le x_{j+k}$. The lower bound is the same argument counting values above $x_{j-k}$. This fact pins down how far the median of a modified dataset can travel.

## Local sensitivity, and why it leaks {#sec-sens-local}

### Context

Global sensitivity takes a maximum over all pairs of neighbours, including pathological ones that look nothing like your data. The natural fix is to take the maximum only over the neighbours of the dataset you hold. That gives a much smaller number. But adding noise proportional to it is not private, and seeing exactly why is the key to everything else in this chapter.

### Statement

The **local sensitivity** of $f$ at $D$ is the most that one record can move $f$ starting from $D$ itself, rather than from the worst dataset:
$$\mathrm{LS}_f(D) = \max_{D':\ d(D, D') = 1}|f(D) - f(D')|,$$
and the global sensitivity is its worst case over all datasets, $\Delta f = \max_D \mathrm{LS}_f(D)$.

For the median, one change can push the median at most one place up or down the sorted list, so its local sensitivity is the larger of the two gaps next to it. Take $n$ values in $[0, \Lambda]$, with $n$ odd and $m = (n+1)/2$ the rank of the median, so that the median is $x_m$. Then
$$\mathrm{LS}_{\mathrm{med}}(D) = \max\big(x_m - x_{m-1},\ x_{m+1} - x_m\big), \qquad \Delta\,\mathrm{med} = \Lambda.$$ {#eq-sens-median-ls}

### Derivation

1. **The median of a neighbour lies between $x_{m-1}$ and $x_{m+1}$.** This is Fact 4 with $k = 1$ change and rank $j = m$. So the median moves from $x_m$ by at most the larger of the gaps to its two sorted neighbours: $|\mathrm{med}(D') - x_m| \le \max(x_m - x_{m-1}, x_{m+1} - x_m)$.
2. **Both ends are reached.** To show the bound is attained, move the median value itself to an extreme. Replace $x_m$ by $\Lambda$: the sorted values become $x_1, \dots, x_{m-1}, x_{m+1}, \dots, x_n, \Lambda$, and the $m$-th of them is $x_{m+1}$. Replace $x_m$ by 0 instead, and everything above it keeps its rank while the new 0 goes first, so the $m$-th value becomes $x_{m-1}$. So the bound of step 1 is attained, which is @eq-sens-median-ls.
3. **The global sensitivity is $\Lambda$.** Take $m$ values equal to 0 and $m - 1$ equal to $\Lambda$. Then $x_m = 0$ and $x_{m+1} = \Lambda$, so @eq-sens-median-ls gives $\Lambda$, and no dataset can do worse because every value lies in $[0, \Lambda]$, so no gap exceeds $\Lambda$.

### Interpretation: the noise scale is a statistic

Consider the mechanism that adds Laplace noise scaled to the local sensitivity of the data it is given, $M_{\mathrm{LS}}(D) = f(D) + \mathrm{Lap}(\mathrm{LS}_f(D)/\varepsilon)$. Its noise scale is the function $\mathrm{LS}_f$ of the data, and **$\mathrm{LS}_f$ itself can have large sensitivity**: one record can change the local sensitivity of the median from 0 to $\Lambda$. The mechanism then releases $f(D)$ with no noise on one dataset and with enormous noise on its neighbour, and the noise level alone tells the two apart, even when $f$ is the same on both.

The degenerate case makes this stark (Nissim, Raskhodnikova & Smith 2007, Example 1). With $n = 5$ and $\Lambda = 100$, the dataset $D = (0, 0, 0, 0, 100)$ has median 0, and both gaps next to the median are 0, so $\mathrm{LS} = 0$ and $M_{\mathrm{LS}}(D) = 0$ exactly. Its neighbour $D' = (0, 0, 0, 100, 100)$ also has median 0, but the gap above the median is now 100, so $\mathrm{LS} = 100$ and $M_{\mathrm{LS}}(D')$ has a continuous distribution, which equals 0 with probability 0. The event "the output is exactly 0" has probability 1 under $D$ and 0 under $D'$. No finite $\varepsilon$ and no $\delta < 1$ cover that, because the DP inequality would need $1 \le e^\varepsilon \cdot 0 + \delta$.

### Example: two neighbours with the same median

The leak does not need a degenerate dataset. Take $\Lambda = 100$, $\varepsilon = 1$, and the two datasets of @tbl-sens-leak, which differ in one record and have the same median, but whose local sensitivities differ by a factor of six.

| Dataset | Values | Median | $\mathrm{LS}$ |
| --- | --- | --- | --- |
| $D$ | 10, 20, 30, 31, 90 | 30 | $\max(30 - 20, 31 - 30) = 10$ |
| $D'$ (31 replaced by 90) | 10, 20, 30, 90, 90 | 30 | $\max(30 - 20, 90 - 30) = 60$ |

: Two neighbouring datasets with the same median and very different local sensitivities. {#tbl-sens-leak}

The two releases are $30 + \mathrm{Lap}(10)$ and $30 + \mathrm{Lap}(60)$. They have the same centre, so the privacy loss depends only on the distance $u = |y - 30|$ from it. Far from the centre the wide density is much larger than the narrow one, because the narrow one decays six times faster, so the loss should grow with $u$. Writing the two Laplace densities $\frac1{2b}e^{-u/b}$ with $b = 60$ and $b = 10$ and taking the log of their ratio:
$$L_{D',D}(y) = \ln\frac{e^{-u/60}/120}{e^{-u/10}/20} = -\ln 6 + u\Big(\frac1{10} - \frac1{60}\Big) = -1.792 + \frac{u}{12}.$$
The constant is $\ln(20/120) = -\ln 6 = -1.792$, and the slope is $\frac{6 - 1}{60} = \frac1{12}$. The loss grows without bound in $u$ (@fig-sens-local-leak), so no pure $\varepsilon$ can hold. It exceeds $\varepsilon = 1$ once $-\ln 6 + u/12 > 1$, that is once $u > 12(1 + \ln 6) = 33.50$. A Laplace variable of scale $b$ lands more than $u$ from its centre with probability $e^{-u/b}$, so this happens with probability $e^{-33.5/60} = 0.572$ under $D'$ and $e^{-33.5/10} = 0.0351$ under $D$.

How large must $\delta$ be for this pair? The smallest $\delta$ for which the pair satisfies the $(1, \delta)$ inequality is the **hockey-stick divergence** of [Divergences](../part01-prerequisites/03-divergences.md): the largest gap $\Pr[M(D') \in S] - e^{1}\Pr[M(D) \in S]$ over all sets $S$, which is attained by taking $S$ to be exactly the outputs where the loss exceeds 1. The script computes it as $0.477$ in the direction $D'$ against $D$ (and $0.215$ in the other direction). An adversary who sees a value more than 34 away from 30 can bet on $D'$ and be right far more often than any reasonable $(\varepsilon, \delta)$ allows. This is the distinguishing game of [Threat models and privacy games](../part02-attacks/01-threat-models.md), in which an adversary who knows both candidate datasets sees one release and guesses which dataset produced it. Here it wins by looking at the spread of the noise, not its centre.

![Left: the densities of the median of $D$ and of $D'$ released with Laplace noise scaled to their own local sensitivities, 10 and 60. Right: the privacy loss $L_{D',D}(y)$ grows linearly in $|y - 30|$ and crosses $\varepsilon = 1$ at distance 33.5.](figures/sens_local_leak.svg){#fig-sens-local-leak}

### Check yourself

- Is $\mathrm{LS}_f(D) \le \Delta f$ always, and when are they equal? *(Always, since the global sensitivity is the maximum of the local one over all datasets. They are equal at the datasets where that maximum is attained, such as half zeros and half $\Lambda$ for the median.)*
- Why doesn't the Laplace proof of Fact 1 go through for $M_{\mathrm{LS}}$? *(It needs the same scale $\Delta$ on both sides. With scales $b \ne b'$ the loss contains $|y - f(D')|/b' - |y - f(D)|/b$, whose slopes no longer cancel, so it is unbounded in $y$.)*

## Smooth sensitivity {#sec-sens-smooth}

### Context

The leak came from the noise scale jumping between neighbours. So use a noise scale that cannot jump: an upper bound on the local sensitivity whose *logarithm* changes by at most $\beta$ between neighbours, so that the scale itself changes by at most a factor $e^\beta$. A change by a bounded *factor* is the right kind of control, because a privacy loss is a log-ratio: rescaling the noise by $e^\beta$ shifts log-densities by about $\beta$. Nissim, Raskhodnikova & Smith (2007) call such a bound smooth, find the smallest one, and show which noise distributions can be scaled by it. Two questions need answers: how to compute the bound, and which noise makes the result private.

### Statement

A function $S$ is a **$\beta$-smooth upper bound** on $\mathrm{LS}_f$ if it is never below the local sensitivity and changes by at most a factor $e^\beta$ between neighbours:
$$S(D) \ge \mathrm{LS}_f(D) \quad\text{and}\quad S(D) \le e^\beta S(D') \quad\text{for all } D \simeq D'.$$
The smallest such bound has an explicit form. If some dataset $D''$ at distance $k$ has a large local sensitivity, a smooth bound cannot be small at $D$, because it can only decrease by a factor $e^{-\beta}$ per step on the way from $D''$ to $D$. So the bound at $D$ must be at least $\mathrm{LS}_f(D'')e^{-\beta k}$ for every $D''$, and the **$\beta$-smooth sensitivity** takes exactly the largest of these:
$$S^*_{f,\beta}(D) = \max_{D''}\ \mathrm{LS}_f(D'')\,e^{-\beta\,d(D, D'')}$$ {#eq-sens-smooth}
It is a $\beta$-smooth upper bound, and it is the smallest one: $S^*_{f,\beta}(D) \le S(D)$ for every $\beta$-smooth upper bound $S$.

The mechanism adds noise from a fixed shape, stretched by $S(D)$:
$$M(D) = f(D) + \frac{S(D)}{\alpha}\,Z,$$ {#eq-sens-smooth-mech}
where $Z$ is drawn from a fixed density $h$ and $\alpha > 0$ is a constant that sets how many noise scales one record's shift is worth. Which $h$ work is the content of the next two results. They differ in how heavy the tail of $h$ is, and the derivation will show why that matters: the noise scale differs between neighbours, and a light-tailed density compared with a stretched copy of itself has a log-ratio that grows without bound in the tail, as in the example of @tbl-sens-leak.

**Pure DP with Cauchy-like noise.** Let $h_\gamma(z) \propto 1/(1 + |z|^\gamma)$ with $\gamma > 1$, a density whose tails decay like the power $|z|^{-\gamma}$ ($\gamma = 2$ is the standard Cauchy). If $S$ is a $\beta$-smooth upper bound and the sources of privacy loss together fit within the budget (the shift of the centre is worth $\gamma\alpha$, and the change of scale is worth $\beta$ through the density's normalizing factor plus $\gamma\beta$ through the stretch of its shape),
$$\gamma\alpha + (\gamma + 1)\beta \le \varepsilon,$$ {#eq-sens-cauchy-cond}
then @eq-sens-smooth-mech is $\varepsilon$-DP. Nissim, Raskhodnikova & Smith (2007, Lemma 2.7) use $\alpha = \beta = \varepsilon/(2(\gamma + 1))$, which satisfies @eq-sens-cauchy-cond with room to spare. For the Cauchy, $\gamma = 2$ gives $\alpha = \beta = \varepsilon/6$, the often-quoted release $f(D) + \frac{6}{\varepsilon}S^*_{f,\varepsilon/6}(D)\cdot Z$.

**Approximate DP with Laplace noise.** Let $Z \sim \mathrm{Lap}(1)$, the Laplace density $\frac12e^{-|z|}$. Its tail is too light for pure DP, so the condition gains a term that charges for the rare draws of size above $\ln(1/\delta)$. If $S$ is a $\beta$-smooth upper bound and
$$\beta + e^\beta\alpha + (e^\beta - 1)\ln(1/\delta) \le \varepsilon,$$ {#eq-sens-laplace-cond}
then @eq-sens-smooth-mech is $(\varepsilon, \delta)$-DP. Nissim, Raskhodnikova & Smith (2007, Lemma 2.9) prove a version with a slightly different split of the budget, $\alpha = \varepsilon/2$ and $\beta = \varepsilon/(2\ln(2/\delta))$. Because the third term is roughly $\beta\ln(1/\delta)$ for small $\beta$, here $\beta$ must shrink like $1/\ln(1/\delta)$, and no $\beta > 0$ gives pure DP with Laplace noise.

### Derivation

**$S^*$ is a smooth upper bound, and the smallest one.**

1. *Upper bound.* One of the datasets in the maximum of @eq-sens-smooth is $D'' = D$ itself, at distance 0, which contributes $\mathrm{LS}_f(D)e^0 = \mathrm{LS}_f(D)$. A maximum is at least each of its terms, so $S^*_{f,\beta}(D) \ge \mathrm{LS}_f(D)$.
2. *Smooth.* Moving from $D$ to a neighbour $D'$ makes every dataset at most one step further away: $d(D', D'') \le d(D, D'') + 1$, by the triangle inequality of Fact 3. So each term $\mathrm{LS}_f(D'')e^{-\beta d(D', D'')}$ of $S^*(D')$ is at least $e^{-\beta}$ times the term for the same $D''$ in $S^*(D)$. Taking the maximum over $D''$ on both sides, $S^*(D') \ge e^{-\beta}S^*(D)$. Nothing in this argument depended on which of the two neighbours was called $D$, so running it with $D$ and $D'$ exchanged gives $S^*(D) \ge e^{-\beta}S^*(D')$, the smoothness condition.
3. *Smallest.* This is the argument that motivated the definition. Let $S$ be any $\beta$-smooth upper bound and $D''$ any dataset at distance $k$ from $D$. Walk from $D''$ to $D$ through $k$ neighbours; smoothness lets $S$ drop by at most a factor $e^{-\beta}$ per step, so $S(D) \ge e^{-\beta k}S(D'')$, and $S(D'') \ge \mathrm{LS}_f(D'')$ because $S$ is an upper bound. So $S(D) \ge e^{-\beta k}\mathrm{LS}_f(D'')$ for every $D''$; maximize over $D''$.

**Pure DP for Cauchy-like noise.** The plan is to write the privacy loss at an output $y$ as a sum of three pieces, one for each way the two output distributions differ (their scales differ, their centres differ, and the noise draw that explains $y$ is stretched), and to bound each piece by smoothness or by the upper-bound property.

1. *Write out the privacy loss.* Stretching $Z$ by $S(D)/\alpha$ divides its density by that factor, so the density of $M(D)$ at $y$ is $\frac{\alpha}{S(D)}h\big(z\big)$, where $z = \alpha\frac{y - f(D)}{S(D)}$ is the noise draw that would produce $y$ from $D$. Likewise that of $M(D')$ is $\frac{\alpha}{S(D')}h(z')$ with $z' = \alpha\frac{y - f(D')}{S(D')}$. To compare the two draws, substitute $y = f(D) + S(D)z/\alpha$ into $z'$: $z' = \frac{S(D)}{S(D')}\big(z + \alpha\frac{f(D) - f(D')}{S(D)}\big)$. That is, $z'$ is $z$ shifted and then stretched:
$$z' = e^{\lambda}(z + v), \qquad e^\lambda = \frac{S(D)}{S(D')},\quad v = \alpha\,\frac{f(D) - f(D')}{S(D)}.$$
(Here $v$ is the shift of the centre, measured in noise scales.) The log of the ratio of the two densities splits into the ratio of the prefactors and the ratio of $h$ at $z$ and at $z'$, and inserting $h(z + v)$ in the middle splits the second part into the shift and the stretch:
$$L_{D,D'}(y) = \underbrace{\ln\frac{S(D')}{S(D)}}_{\text{scale}} + \underbrace{\ln\frac{h(z)}{h(z + v)}}_{\text{sliding}} + \underbrace{\ln\frac{h(z + v)}{h(e^\lambda(z + v))}}_{\text{dilation}}.$$
2. *Size of the shift and the stretch.* Smoothness says $S(D)/S(D')$ lies between $e^{-\beta}$ and $e^\beta$, so $|\lambda| \le \beta$. And one record moves $f$ by at most the local sensitivity, which is at most $S(D)$: $|f(D) - f(D')| \le \mathrm{LS}_f(D) \le S(D)$, so $|v| \le \alpha$. This is the only place where "$S$ is an upper bound on the local sensitivity" enters.
3. *Scale term:* at most $\beta$, by smoothness again.
4. *Sliding term.* The question is how fast $-\ln h$ can change when its argument shifts by $|v|$. Write $\varphi(u) = \ln(1 + u^\gamma)$ for $u \ge 0$, so $-\ln h_\gamma(z) = \varphi(|z|) + \text{const}$. Its slope is $\varphi'(u) = \gamma u^{\gamma-1}/(1 + u^\gamma) \le \gamma$, because $u^{\gamma-1} \le 1 + u^\gamma$ (if $u \le 1$ the left side is at most 1; if $u > 1$ it is below $u^\gamma$). The mean value theorem says a function whose slope is at most $\gamma$ in size changes by at most $\gamma$ times the change in its argument. The argument changes from $|z|$ to $|z + v|$, and these differ by at most $|v|$ (the reverse triangle inequality). So $|\varphi(|z + v|) - \varphi(|z|)| \le \gamma\big||z + v| - |z|\big| \le \gamma|v| \le \gamma\alpha$.
5. *Dilation term.* Here the question is how much $h$ changes when its argument is stretched by $e^\lambda$. With $w = |z + v|$ and $\lambda \ge 0$, $\frac{h(w)}{h(e^\lambda w)} = \frac{1 + e^{\lambda\gamma}w^\gamma}{1 + w^\gamma} \le e^{\lambda\gamma}$, since $(1 + ca)/(1 + a) \le c$ for $c \ge 1$ (multiply out: $1 + ca \le c + ca$). This bound holds for every $w$, however far in the tail, and that is what the power-law tail buys. For $\lambda < 0$ the stretch moves the argument towards 0, where $h$ is larger, so the ratio is at most 1. Either way the term is at most $\gamma|\lambda| \le \gamma\beta$.
6. *Add up:* $L_{D,D'}(y) \le \beta + \gamma\alpha + \gamma\beta$ for **every** $y$. That is $\gamma\alpha + (\gamma + 1)\beta$, so under @eq-sens-cauchy-cond it is at most $\varepsilon$, which is pure $\varepsilon$-DP. With $\alpha = \beta = \varepsilon/6$ and $\gamma = 2$ the bound is $\gamma\alpha + (\gamma + 1)\beta = \frac{2}{6} + \frac{3}{6} = 0.8333\,\varepsilon$.

**Approximate DP for Laplace noise.** Steps 1–3 are unchanged. For $h(z) = \frac12e^{-|z|}$ the log of a ratio of $h$ values is just a difference of absolute values, so the sliding and dilation terms combine into one. Using $|e^\lambda(z + v)| \le e^\lambda(|z| + |v|)$, then $e^\lambda \le e^\beta$ and $|v| \le \alpha$ from step 2:
$$\ln\frac{h(z)}{h(z')} = |e^\lambda(z + v)| - |z| \le e^\lambda(|z| + |v|) - |z| \le (e^\beta - 1)|z| + e^\beta\alpha.$$
Adding the scale term, $L_{D,D'}(y) \le \beta + e^\beta\alpha + (e^\beta - 1)|z|$, where $z$ is the noise actually drawn, $Z \sim \mathrm{Lap}(1)$. This is **not** bounded: the stretch makes the Laplace log-density change by an amount proportional to $|z|$, so far in the tail the dilation term wins. But large draws are rare. A $\mathrm{Lap}(1)$ variable exceeds $t$ in absolute value with probability $e^{-t}$ (integrate the two tails of $\frac12e^{-|z|}$; see [Concentration inequalities](../part01-prerequisites/02-concentration.md)), so $\Pr[|Z| > \ln(1/\delta)] = \delta$. Outside that event of probability $\delta$, $|z| \le \ln(1/\delta)$ and the loss is at most the left side of @eq-sens-laplace-cond, hence at most $\varepsilon$. Fact 2 turns that into $(\varepsilon, \delta)$-DP.

### Interpretation: why heavy tails, and what admissibility means

Pure DP forces polynomial tails. The dilation step needed $h(w)/h(e^\lambda w)$ to be bounded for all $w$, however large. Write $h(w) \approx e^{-g(w)}$; the condition says $g(e^\beta w) - g(w)$ stays below some constant $c$. Applying it $k$ times from $w = 1$ gives $g(e^{k\beta}) \le g(1) + kc$, so $g$ can grow at most like $\frac{c}{\beta}\ln w$, and $h$ can decay at most like a power of $w$: Cauchy-like. Laplace noise has $g(w) = w$, for which $g(e^\beta w) - g(w) = (e^\beta - 1)w$ grows without bound, and the Gaussian is worse. That is why both fail for pure DP and pay with $\delta$.

The two conditions are what Nissim, Raskhodnikova & Smith call **admissibility**. They call $h$ $(\alpha, \beta)$-admissible when shifting it by at most $\alpha$ (sliding) and rescaling it by at most a factor $e^{\pm\beta}$ (dilation) each change probabilities by at most a factor $e^{\varepsilon/2}$ plus $\delta/2$. The derivation above bounds the two together, which gives slightly different constants.

The price of heavy tails shows up in the error. The Cauchy has no mean, so its error is reported through the median. For a standard Cauchy, $\Pr[|Z| \le t] = \frac2\pi\arctan t$, which is $\frac12$ at $t = 1$ because $\arctan 1 = \pi/4$. So $|Z|$ has median 1, and the typical error is the noise scale $S/\alpha$. But the tail is heavy: $\Pr[|Z| > 10] = 1 - \frac2\pi\arctan 10 = 0.0635$, so about one release in 16 is off by more than ten times the typical error. With Laplace noise the same probability is $e^{-10} \approx 4.5\times10^{-5}$.

Finally, $\beta$ is a lever. A small $\beta$ makes $S^*$ larger, because the factor $e^{-\beta k}$ discounts distant datasets less and so lets more of them raise the maximum. In exchange it costs less privacy per unit of noise. Laplace noise needs $\beta \approx \varepsilon/(2\ln(1/\delta))$, much smaller than the Cauchy's $\varepsilon/6$, because in @eq-sens-laplace-cond the change of scale is charged once for every unit of $\ln(1/\delta)$.

### Computing $S^*$ for the median

@eq-sens-smooth is a maximum over all datasets, which is not something you can enumerate. The way out is to group the datasets by their distance $k$ from $D$: all datasets at distance $k$ get the same discount $e^{-k\beta}$, so within a group you only need the largest local sensitivity, $A^{(k)}(D)$, and then the best discounted group:
$$S^*_{\beta}(D) = \max_{k = 0, \dots, n} e^{-k\beta}A^{(k)}(D), \qquad A^{(k)}(D) = \max_{D'':\ d(D, D'') \le k}\mathrm{LS}(D'').$$ {#eq-sens-smooth-by-k}
(Using "$\le k$" instead of "$= k$" changes nothing: a dataset at distance $j < k$ is counted again with the smaller weight $e^{-k\beta}$, which never beats its own weight $e^{-j\beta}$.)

For the median, $A^{(k)}$ asks: by changing $k$ records, how wide a gap can you open next to the median? A gap is opened by moving the values that lie inside an interval of the sorted data out of it, to 0 or to $\Lambda$. With $k$ changes you can empty out the $k$ values between two order statistics $k + 1$ ranks apart, and you can choose how many to move down and how many up so that the emptied interval ends up next to the median. So the answer is the widest such window that sits close enough to rank $m$. With the padding of Fact 4, Nissim, Raskhodnikova & Smith (2007, Proposition 3.4) show
$$A^{(k)}(D) = \max_{0 \le t \le k+1}\big(x_{m+t} - x_{m+t-k-1}\big).$$ {#eq-sens-median-ak}
Each window runs from rank $m + t - k - 1$ to rank $m + t$, and $t$ slides it across the median. Each $A^{(k)}$ costs $O(k)$ to evaluate, one subtraction per window, and $k$ never needs to exceed $n$: at $k = n$ the window with $t = m$ spans $x_0 = 0$ to $x_{n+1} = \Lambda$, so $A^{(n)} = \Lambda$, the largest possible value, and larger $k$ only lowers the weight. So the exact $S^*_\beta$ takes $O(n^2)$ time. (An $O(n\log n)$ algorithm exists; the script uses the plain formula.)

**Derivation of @eq-sens-median-ak.** Steps 1–4 show that no dataset within distance $k$ has a gap next to its median wider than the widest window, and step 5 shows that every window is achieved.

1. *What the gap looks like.* Let $D''$ be within distance $k$ of $D$, with sorted values $y_i$. By @eq-sens-median-ls its local sensitivity is one of its two gaps next to the median. Take the upper one, $(y_m, y_{m+1})$. The lower one needs no separate argument: reflecting every value, $x \mapsto \Lambda - x$, turns lower gaps into upper gaps and maps the set of windows in @eq-sens-median-ak to itself.
2. *The gap contains at most $k$ original values.* Assume the gap is positive (otherwise there is nothing to bound), so exactly $m$ values of $D''$ are $\le y_m$. No value of $D''$ lies strictly inside $(y_m, y_{m+1})$, so every value of $D$ strictly inside it must have been one of the $k$ changed records. Call the values of $D$ inside the gap $x_{a+1}, \dots, x_{a+j}$ with $j \le k$. Then $x_a \le y_m$ and $x_{a+j+1} \ge y_{m+1}$, so the gap is at most $x_{a+j+1} - x_a$, a window of the original data.
3. *The gap sits near rank $m$.* Now pin down $a$ by counting values of $D''$ at or below $y_m$, of which there are exactly $m$. Each is either an unchanged value of $D$, and those are among $x_1, \dots, x_a$ (everything above $x_a$ is inside the gap or beyond it), or one of at most $k$ new values. So $m \le a + k$, that is $a \ge m - k$. In the other direction, at least $a - (k - j)$ of $x_1, \dots, x_a$ are unchanged (the $j$ values inside the gap used up $j$ of the $k$ changes), and all of them are $\le y_m$, so $a - (k - j) \le m$, that is $a \le m + k - j$.
4. *Fit it in a window.* Let $i = \min(a, m)$, the lower end of the window to use. If $a \le m$, the window $[x_a, x_{a+k+1}]$ has $i = a \ge m - k$, so it is one of the windows of @eq-sens-median-ak ($t = a + k + 1 - m \in [0, k+1]$), and it contains $[x_a, x_{a+j+1}]$ because $j \le k$. If $a > m$, the window $[x_m, x_{m+k+1}]$ ($t = k + 1$) starts below $x_a$ and contains $[x_a, x_{a+j+1}]$ because $a + j + 1 \le m + k + 1$ by step 3. Either way the gap is at most the right side of @eq-sens-median-ak.
5. *Every window is reached.* For a window $[x_\ell, x_u]$ with $\ell = m + t - k - 1$ and $u = m + t$, there are exactly $u - \ell - 1 = k$ values strictly between its ends. Change them: move $k + 1 - t$ of them to 0 and the rest to $\Lambda$ (for $t = 0$, move all $k$ to 0). Each value moved to 0 now sits below $x_\ell$ and raises its rank by one. So $x_\ell$ has rank $\ell + (k + 1 - t) = m$ and $x_u$ rank $m + 1$ for $t \ge 1$ (for $t = 0$, ranks $m - 1$ and $m$). Either way the median sits at one end of an empty interval of length $x_u - x_\ell$, which is then one of its two gaps.

**A walkthrough.** Take $n = 5$ values on $[0, 10]$, $D = (1, 3, 4, 6, 9)$, so $m = 3$ and the median is 4, and pad with $x_i = 0$ for $i \le 0$ and $x_i = 10$ for $i \ge 6$. @tbl-sens-walk evaluates @eq-sens-median-ak window by window, with $\beta = 0.5$, and checks it against brute force: the script enumerates all $11^5 = 161{,}051$ datasets with values in $\{0, \dots, 10\}$, computes the local sensitivity of each and its distance from $D$, and takes the maximum for each $k$.

| $k$ | Windows $x_{m+t} - x_{m+t-k-1}$, $t = 0, \dots, k+1$ | $A^{(k)}$ | Brute force | $e^{-0.5k}A^{(k)}$ |
| --- | --- | --- | --- | --- |
| 0 | $4 - 3,\ 6 - 4$ | 2 | 2 | 2.00 |
| 1 | $4 - 1,\ 6 - 3,\ 9 - 4$ | 5 | 5 | **3.03** |
| 2 | $4 - 0,\ 6 - 1,\ 9 - 3,\ 10 - 4$ | 6 | 6 | 2.21 |
| 3 | $4 - 0,\ 6 - 0,\ 9 - 1,\ 10 - 3,\ 10 - 4$ | 8 | 8 | 1.79 |
| 4 | $\dots,\ 9 - 0,\ 10 - 1,\ \dots$ | 9 | 9 | 1.22 |
| 5 | $\dots,\ 10 - 0,\ \dots$ | 10 | 10 | 0.82 |

: The smooth sensitivity of the median of $D = (1, 3, 4, 6, 9)$ on $[0, 10]$ with $\beta = 0.5$: the formula for $A^{(k)}$ against brute force over all datasets, and the discounted values whose maximum is $S^*_{0.5}$. {#tbl-sens-walk}

The local sensitivity (the row $k = 0$) is 2, the gap from 4 up to 6. But one change, moving the 6 up to 10, opens a gap of $9 - 4 = 5$ next to the median, and even after the discount $e^{-0.5} = 0.6065$ for one step, $e^{-0.5}\times 5 = 3.03$ beats 2. Further rows open wider gaps but pay a larger discount. So $S^*_{0.5}(D) = 3.03$, attained at $k = 1$ (@fig-sens-smooth-profile, left).

![Left: $A^{(k)}$ and $e^{-k\beta}A^{(k)}$ for the walkthrough dataset; the star marks $S^*_{0.5} = 3.03$. Right: the same for the median of 1,001 synthetic ages (a fresh draw) with $\beta = 1/6$ (Cauchy) and $\beta = 0.032$ (Laplace, $\delta = 10^{-6}$); the stars mark the maxima. The smaller $\beta$ reaches further out.](figures/sens_smooth_profile.svg){#fig-sens-smooth-profile}

### Example: the median age

The script draws $n = 1001$ ages from a normal distribution with mean 38 and standard deviation 14, clipped to $[0, 100]$, and releases the median (36.96) with $\varepsilon = 1$. Each option is compared by its median absolute error. For noise $bZ$ that is $b$ times the median of $|Z|$: 1 for the Cauchy, as shown above, and $\ln 2$ for the Laplace, because $\Pr[|Z| > t] = e^{-t}$ equals $\frac12$ at $t = \ln 2$.

With global sensitivity, Laplace noise has scale $\Lambda/\varepsilon = 100$ and median absolute value $100\ln 2 = 69.31$ years: useless. The local sensitivity is 0.1386 years, but releasing with it is not private, as @sec-sens-local showed.

Smooth sensitivity with Cauchy noise uses $\alpha = \beta = 1/6$. Then $S^*_{1/6} = 0.1632$, attained at $k = 7$: only slightly above the local sensitivity, because with this many people even seven changes cannot open a wide gap near the middle. The noise scale $6S^*/\varepsilon = 6 \times 0.1632 = 0.979$ is also the median absolute error (simulation: 0.980).

Smooth sensitivity with Laplace noise uses $\delta = 10^{-6}$ and $\alpha = 1/2$. The largest $\beta$ allowed by @eq-sens-laplace-cond is then $0.0322$ (close to the $0.0345$ of Nissim, Raskhodnikova & Smith). The smaller $\beta$ lets more distant datasets count, so $S^*_{0.0322} = 0.5144$, attained at $k = 21$. The noise scale is $S^*/\alpha = 2S^*/\varepsilon = 1.029$ and the median absolute error $1.029\ln 2 = 0.713$.

Both smooth releases are 70 to 100 times more accurate than the global one ($69.31/0.979$ and $69.31/0.713$). @fig-sens-methods-vs-n shows the gain grows with $n$: the smooth noise falls roughly like $1/n$, the spacing of the data near the median, while the global noise stays put. For $n \lesssim 20$ the smooth noise is *worse* than the global one, because with few points a handful of changes can open a gap of nearly the whole range, so $S^*$ can be close to $\Lambda$, and then the factor $1/\alpha$ multiplies it.

### Check yourself

- Why does $S^*_\beta(D) > 0$ even when $\mathrm{LS}(D) = 0$? *(Some dataset at finite distance $k$ has positive local sensitivity, and it contributes $e^{-k\beta}$ times that. For the median, $A^{(n)} = \Lambda$ gives $S^* \ge e^{-n\beta}\Lambda > 0$, which also means the scale in @eq-sens-smooth-mech is never zero.)*
- In the Cauchy derivation, where is "upper bound" used and where is "smooth" used? *(Upper bound bounds the shift: $|v| \le \alpha$. Smoothness bounds the scale term and the stretch $|\lambda| \le \beta$.)*

## Propose-test-release {#sec-sens-ptr}

### Context

Smooth sensitivity needs a formula like @eq-sens-median-ak, and those are known for only a few functions. Dwork & Lei (2009) take a cruder route that needs much less. Guess a bound $b$ on the local sensitivity, check privately that the data are not near any dataset where the guess fails, and only then release with noise of scale $b/\varepsilon$. If the check fails, refuse to answer. The check is private because it is a "how many changes until something bad" count, which has sensitivity 1 by Fact 3.

### Statement

Fix a proposed bound $b > 0$, and privacy parameters $\varepsilon > 0$ and $\delta \in (0, 1)$. The quantity to test is the number of records you must change before the local sensitivity can exceed $b$, that is, how far $D$ is from the nearest dataset where the proposal would be too small:
$$\operatorname{dist}_b(D) = \min\big\{d(D, D''):\ \mathrm{LS}_f(D'') > b\big\}.$$
**Propose-test-release** adds Laplace noise to this distance and compares it with a threshold $\ln(1/\delta)/\varepsilon$, chosen so that the noise alone pushes a distance of 0 over it only with probability $\delta/2$. It outputs one of two things:

- $\bot$ ("no answer") if $\operatorname{dist}_b(D) + \mathrm{Lap}(1/\varepsilon) \le \ln(1/\delta)/\varepsilon$;
- $f(D) + \mathrm{Lap}(b/\varepsilon)$ otherwise.

It is $(2\varepsilon, \delta/2)$-DP, so in particular $(2\varepsilon, \delta)$-DP. For the median, $\operatorname{dist}_b(D) = \min\{k : A^{(k)}(D) > b\}$, directly from @eq-sens-median-ak: $A^{(k)}$ is the largest local sensitivity within $k$ changes, so the first $k$ where it exceeds $b$ is the distance to the first dataset where $b$ fails.

### Derivation

The idea is to split neighbouring pairs into two cases. If the proposal $b$ is right for this pair, the release is an ordinary Laplace mechanism. If it is wrong, both datasets sit right at an unstable point, the distance is 0, and the test almost always refuses.

1. **The test is $\varepsilon$-DP.** $\operatorname{dist}_b$ is a distance to the set of datasets whose local sensitivity exceeds $b$, so it has sensitivity 1 (Fact 3), and adding $\mathrm{Lap}(1/\varepsilon)$ is the Laplace mechanism (Fact 1). Whether the test passes is a post-processing of that noisy value: a function of it that does not look at the data again, which cannot increase the privacy loss.
2. **Case 1: $|f(D) - f(D')| \le b$.** Then the release step is also $\varepsilon$-DP (Fact 1 with $\Delta = b$). The output is a function of two independent $\varepsilon$-DP outputs, the noisy distance and the noisy value. Basic composition says that running two mechanisms with independent noise costs the sum of their $\varepsilon$s, because the privacy loss of the pair is the sum of the two losses; post-processing says the function of them costs nothing more ([Post-processing, group privacy and composition](02-properties.md)). So the mechanism is $2\varepsilon$-DP on this pair.
3. **Case 2: $|f(D) - f(D')| > b$.** Then both $\mathrm{LS}_f(D) > b$ and $\mathrm{LS}_f(D') > b$, because each local sensitivity is the largest change over that dataset's neighbours, and each of $D$, $D'$ is a neighbour of the other. So $\operatorname{dist}_b(D) = 0$, and the test passes only if $\mathrm{Lap}(1/\varepsilon) > \ln(1/\delta)/\varepsilon$. A $\mathrm{Lap}(1/\varepsilon)$ variable exceeds $t$ with probability $\frac12e^{-\varepsilon t}$ (one tail of the density), so this has probability $\frac12e^{-\ln(1/\delta)} = \delta/2$.
4. **Bound case 2.** For any set $S$ of outputs, $M(D)$ lands in $S$ either by refusing (if $\bot \in S$) or by releasing, and the release happens with probability at most $\delta/2$ by step 3. So $\Pr[M(D) \in S] \le \Pr[M(D) = \bot]\,\mathbf 1[\bot \in S] + \delta/2$, where $\mathbf 1[\bot \in S]$ is 1 if $\bot$ is in $S$ and 0 otherwise. The refusal is a post-processing of the $\varepsilon$-DP test of step 1, so the first term is at most $e^\varepsilon\Pr[M(D') = \bot]\,\mathbf 1[\bot\in S] \le e^{2\varepsilon}\Pr[M(D') \in S]$.
5. **Combine:** in both cases $\Pr[M(D) \in S] \le e^{2\varepsilon}\Pr[M(D') \in S] + \delta/2$.

The "release the median for free" failure of the previous section is now confined to step 3: a dataset whose local sensitivity exceeds the proposal leaks only when a Laplace draw lands $\ln(1/\delta)/\varepsilon$ above zero.

### Interpretation

Propose-test-release is all or nothing. When the data are stable, you get noise of scale $b/\varepsilon$, set by your proposal rather than by the data, so the noise scale reveals nothing. When they are not, you get $\bot$, and $\bot$ itself is a private answer: it says only that the data were probably near an unstable dataset.

The threshold the distance must clear is large. The test passes with probability $\frac12$ when the distance equals the threshold, because the Laplace noise is positive half the time. With $\varepsilon = 1$ and $\delta = 10^{-6}$ that threshold is $\ln(10^6) = 13.82$, so the test needs $\operatorname{dist}_b(D) \gtrsim 13.82$ to pass with probability $\frac12$. The data must be stable against more than a dozen changes, not just one.

The proposal $b$ must not depend on the data. It is a proposal made before looking. Choosing $b$ as, say, twice the observed local sensitivity makes the noise scale of the release a function of the data again, and reintroduces the leak of the previous section through the back door.

Compared with smooth sensitivity, propose-test-release needs only the distance to instability, which is often easier to compute or bound than $S^*$. It pays $2\varepsilon$ instead of $\varepsilon$, and it can refuse.

### Example: the median age

On the 1,001 ages, propose $b = 1$ year. The first $k$ with $A^{(k)} > 1$ is $k = 21$, so $\operatorname{dist}_1(D) = 21$, comfortably above the threshold $13.82$. The test fails only if the noise drops the distance by more than $21 - 13.82$, which a $\mathrm{Lap}(1)$ draw does with probability $\frac12e^{-(21 - 13.82)}$. So the pass probability is $1 - \frac12e^{-(21 - 13.82)} = 0.999621$, and the release has noise $\mathrm{Lap}(1)$, with median absolute error $\ln 2 = 0.693$ years at a total budget of $2\varepsilon = 2$. On a sample of 101 ages, $\operatorname{dist}_1 = 1$, and the test passes only if the noise adds more than $13.82 - 1$, which has probability $\frac12e^{-(13.82 - 1)} = 1.36\times10^{-6}$: there, one year is far too optimistic a proposal. @fig-sens-methods-vs-n (right) shows the pass probability switching from 0 to 1 between $n = 500$ and $n = 1000$.

![Left: median absolute noise when releasing the median age with $\varepsilon = 1$, by global sensitivity, smooth sensitivity with Cauchy noise and with Laplace noise ($\delta = 10^{-6}$), propose-test-release with $b = 1$ when it passes, and inverse sensitivity (@sec-sens-inverse), averaged over 10 synthetic datasets per $n$. The dotted line is the non-private local sensitivity. Right: the probability that propose-test-release with $b = 1$ answers.](figures/sens_methods_vs_n.svg){#fig-sens-methods-vs-n}

### Check yourself

- Why does step 3 need *both* local sensitivities to exceed $b$, and why is that automatic? *(If either were at most $b$, the pair would fall into case 1. It is automatic because $|f(D) - f(D')|$ is one of the terms in the maximum defining each local sensitivity.)*
- Why does propose-test-release cost $2\varepsilon$ while smooth sensitivity costs $\varepsilon$? *(It runs two Laplace mechanisms, one for the test and one for the release, and composes them. Smooth sensitivity folds the cost of a data-dependent scale into the single noise draw.)*

## Inverse sensitivity {#sec-sens-inverse}

### Context

Smooth sensitivity and propose-test-release keep the shape "true answer plus noise", and all their effort goes into choosing the noise scale safely. Inverse sensitivity (Asi & Duchi 2020) drops that shape. Instead, for each candidate output $t$ it asks how many records you would have to change before the statistic equals $t$. Then it samples $t$ with the [exponential mechanism](05-selection.md), using that count as a penalty, so outputs that need many changes become exponentially unlikely. The count is a "how many changes" question, so by Fact 3 one record moves it by at most 1, and no smooth bound, test or refusal is needed.

### Statement

The **inverse sensitivity** of $f$ at $D$ toward an output $t$ is the number of records you must change to make $f$ equal to $t$:
$$\mathrm{len}_f(D, t) = \min\{d(D, D'') : f(D'') = t\}.$$
The **inverse-sensitivity mechanism** samples $t$ from the set of possible outputs, which for the statistics of this chapter is the interval $[0, \Lambda]$, with the density
$$p_D(t) = \frac{e^{-\varepsilon\,\mathrm{len}_f(D, t)/2}}{\int_0^\Lambda e^{-\varepsilon\,\mathrm{len}_f(D, s)/2}\,\mathrm ds}.$$ {#eq-sens-inverse}
It is $\varepsilon$-DP whenever every possible output is the value of $f$ on some dataset, so that every count is finite.

For the median the count has a closed form. Take $n$ odd, so that the median is the single value of rank $m = (n + 1)/2$, and sort and pad the data as in Fact 4, with $x_0 = 0$ and $x_{n+1} = \Lambda$. For a candidate $t$ in the open gap $(x_i, x_{i+1})$ between two consecutive sorted values, the count is the number of steps from that gap to the median:
$$\mathrm{len}_{\mathrm{med}}(D, t) = \begin{cases} m - i & \text{if } i < m,\\ i - m + 1 & \text{if } i \ge m.\end{cases}$$ {#eq-sens-inverse-median}
So the density is a staircase. It is flat on each gap and falls by a factor $e^{-\varepsilon/2}$ for each step away from the median, and the probability of a gap is its length times its height.

### Derivation

1. **The penalty has sensitivity 1.** The count $\mathrm{len}_f(D, t)$ is the distance from $D$ to the set of datasets on which $f$ equals $t$. By Fact 3 a distance to a set changes by at most 1 between neighbours, and this holds for every $t$ at once. The set must not be empty, which is why every output must be reachable. For the median, $n$ copies of $t$ reach $t$.
2. **Privacy.** @eq-sens-inverse is the exponential mechanism with score $u(D, t) = -\mathrm{len}_f(D, t)$, sensitivity $\Delta = 1$, and length on $[0, \Lambda]$ as the base measure. Its proof in [Selection](05-selection.md) bounds two factors. One record moves the weight $e^{\varepsilon u/2}$ of the output $t$ by at most $e^{\varepsilon/2}$, and it moves the total weight in the denominator by at most $e^{\varepsilon/2}$ too. So the ratio of the two densities at every $t$ is at most $e^\varepsilon$, and integrating over any set of outputs gives $\varepsilon$-DP, as in Fact 1.
3. **The count above the median.** Take $t$ in the gap $(x_i, x_{i+1})$ with $i \ge m$, so exactly $i$ values of $D$ lie below $t$. For the median of a dataset to be $t$, at most $m - 1$ of its values can lie below $t$, and one must equal $t$. Each change removes at most one value from below $t$, so at least $i - (m - 1) = i - m + 1$ changes are needed. That many are enough: move one of the values below $t$ to $t$ itself and $i - m$ others to $\Lambda$. Then $m - 1$ values lie below $t$ and the $m$-th smallest is $t$.
4. **The count below the median** follows by reflecting every value, $x \mapsto \Lambda - x$, which turns a gap below the median into one above it. The gap $(x_i, x_{i+1})$ with $i < m$ becomes the gap with index $n - i \ge m$, and step 3 gives $(n - i) - m + 1 = m - i$, using $n + 1 = 2m$.

### Interpretation

Smooth sensitivity had to keep its noise scale from leaking, but here there is no noise scale at all. The data shape the whole output density, through the lengths of the gaps, but by step 2 that density changes by at most a factor $e^\varepsilon$ between neighbours. Where the data are dense, the gaps near the median are short, so the mass that sits within a few steps of the median also sits within a short distance of it. The error is therefore a few gaps, the spacing of the data near the median, times a factor that grows with $1/\varepsilon$. Asi & Duchi (2020) show that, up to logarithmic factors, no $\varepsilon$-DP mechanism does much better on a dataset and its close neighbours together. So the method is near-optimal dataset by dataset, not only in the worst case.

The method has two limits. The first is the two outer gaps, $(0, x_1)$ and $(x_n, \Lambda)$, which can be long and are $m$ steps away. Each gets a weight of its length times $e^{-\varepsilon m/2}$, against a total weight set by the short gaps near the median, so its probability is about $e^{-\varepsilon(m-1)/2}$ times the ratio of its length to those gaps. That is negligible for a thousand points but not for eleven, which is the same failure on tiny datasets as smooth sensitivity's. The second is that the count must be computed. Closed forms like @eq-sens-inverse-median are known for medians, quantiles and a few other statistics. For a general $f$, Asi & Duchi study approximations of the count that keep the guarantee. [Exercise 8](../../book-exercises/part03-dp-foundations/07-sensitivity.md#sec-sol-sens-8) extends the count to any quantile.

### Example: the walkthrough and the median age

Take the walkthrough dataset $D = (1, 3, 4, 6, 9)$ on $[0, 10]$ again, with $\varepsilon = 1$, so $m = 3$. @tbl-sens-inverse lists its six gaps, the count from @eq-sens-inverse-median, and each gap's probability, its length times $e^{-\mathrm{len}/2}$ divided by the total. The two gaps next to the median, $(3, 4)$ and $(4, 6)$, are one change away. Together they get probability $0.1477 + 0.2955 = 0.4432$.

| Gap | $(0, 1)$ | $(1, 3)$ | $(3, 4)$ | $(4, 6)$ | $(6, 9)$ | $(9, 10)$ |
| --- | --- | --- | --- | --- | --- | --- |
| Length | 1 | 2 | 1 | 2 | 3 | 1 |
| $\mathrm{len}_{\mathrm{med}}(D, t)$ | 3 | 2 | 1 | 1 | 2 | 3 |
| Probability | 0.0544 | 0.1792 | 0.1477 | 0.2955 | 0.2688 | 0.0544 |

: The inverse-sensitivity release of the median of $D = (1, 3, 4, 6, 9)$ on $[0, 10]$ with $\varepsilon = 1$: each gap between sorted values, the number of changes that make a point of it the median, and its probability. {#tbl-sens-inverse}

On the 1,001 ages of the previous sections the result is striking. Computed exactly from the density, the median absolute error of the release is 0.0783 years, and 200,000 simulated draws give 0.0784. That is pure $\varepsilon$-DP with $\varepsilon = 1$, it never refuses, and it is twelve times more accurate than smooth sensitivity with Cauchy noise, whose median error is 0.979. It is even below the local sensitivity, 0.1386. That is no contradiction, because the local sensitivity is the larger of the two gaps next to the median, a worst case over one change, while a typical release lands only a few short gaps from the median. @fig-sens-methods-vs-n shows the same ordering at every dataset size. On 101 ages the error grows to 0.696 years, because the gaps are ten times wider.

### Check yourself

- Why does the exponential mechanism here keep the factor 2 in $e^{-\varepsilon\,\mathrm{len}/2}$? *(The scores are not monotone. One change can bring the median one step closer to some outputs and one step further from others, so the weight of an output and the total weight can move in opposite directions, and each can use up $e^{\varepsilon/2}$.)*
- What happens to the release if every value of $D$ is the same, so that most gaps have length 0? *(The empty gaps get probability 0, and the mass sits on the two outer gaps, $m$ steps away on either side of the common value. Both have count $m$, so the release is uniform on $[0, \Lambda]$ and says nothing about the common value.)*

## Stability-based release {#sec-sens-stable}

### Context

Some statistics take discrete values: the most common diagnosis, the model a selection procedure picks, the majority vote of the blocks in subsample-and-aggregate (the next section, @sec-sens-saa). On most datasets one record cannot change them at all, so their local sensitivity is 0. Propose-test-release with the proposal $b = 0$ handles them. You test privately that the data are far from any dataset where one change flips the answer, and if they are, you release the answer exactly, with no noise at all. This is the **stability-based release** (Dwork & Roth 2014, ch. 7; Thakurta & Smith 2013).

### Statement

Let $f$ be the discrete statistic to release, and let $s$ be a **stability score**: a function of the data with sensitivity 1 that equals 0 on every dataset that has a neighbour where $f$ gives a different answer. Fact 3 provides one: the distance to the set of such datasets. The **stability-based release** outputs
$$M(D) = \begin{cases}\bot & \text{if } s(D) + \mathrm{Lap}(1/\varepsilon) \le \ln(1/\delta)/\varepsilon,\\ f(D) & \text{otherwise,}\end{cases}$$ {#eq-sens-stable}
and it is $(\varepsilon, \delta/2)$-DP.

Here $\bot$ is a refusal to answer. For the **mode** of a categorical dataset, the most frequent value with ties broken by a fixed order of the values, write $g$ for its lead, the top count minus the second-highest count. Under replacement,
$$s(D) = \max\big(0,\ \lceil g/2\rceil - 1\big)$$ {#eq-sens-stable-mode}
satisfies both conditions on $s$, where $\lceil g/2\rceil$ rounds $g/2$ up to an integer. It halves the lead because one replacement can move the lead by 2, and it subtracts 1 because a lead of 2 or less can already be overturned by one change.

### Derivation

The proof is that of propose-test-release, with one difference: when the answers agree, nothing is released that depends on the data apart from the test.

1. **The test is $\varepsilon$-DP.** The score $s$ has sensitivity 1, so adding $\mathrm{Lap}(1/\varepsilon)$ is the Laplace mechanism (Fact 1), and comparing the noisy value with a threshold is post-processing.
2. **Neighbours with the same answer, $f(D) = f(D')$.** The release is the same function of the test's outcome on both: $\bot$ if the test fails, the common answer if it passes. So it is post-processing of an $\varepsilon$-DP output, and it is $\varepsilon$-DP on this pair. Propose-test-release paid a second $\varepsilon$ here for its noisy answer, which this release does not add.
3. **Neighbours with different answers.** Then $s(D) = 0$ and $s(D') = 0$ by assumption, since each has a neighbour with a different answer, and the test passes only if the noise exceeds $\ln(1/\delta)/\varepsilon$, which has probability $\frac12e^{-\ln(1/\delta)} = \delta/2$. Step 4 of the propose-test-release derivation then gives $\Pr[M(D) \in S] \le e^\varepsilon\Pr[M(D') \in S] + \delta/2$ for every set $S$ of outputs.
4. **The mode's $s$ has sensitivity 1.** One replacement lowers one count by at most 1 and raises another by at most 1, so the largest and the second-largest counts each move by at most 1, and the lead $g$ by at most 2. Halving and rounding up turns a move of 2 into a move of at most 1, since $\lceil (g + 2)/2\rceil = \lceil g/2\rceil + 1$, and taking the maximum with 0 keeps that.
5. **The mode's $s$ is 0 where the answer can flip.** Write $c_1$ for the count of the mode and $c_2$ for the second-highest count. If $g \ge 3$, then after one change the old mode still has at least $c_1 - 1$ records, and every other value has at most $c_2 + 1 \le c_1 - 2$. So the mode cannot change. A dataset whose neighbour has a different mode therefore has $g \le 2$, and then $\lceil g/2\rceil - 1 \le 0$.

### Interpretation

The release is exact. When the test passes, you get the true answer, and the whole cost of privacy is the chance of a refusal. That chance depends on the lead, not on the number of possible values. The test needs $s(D)$ to be about $\ln(1/\delta)/\varepsilon$, so the mode must lead by about $2\ln(1/\delta)/\varepsilon$ records, whether there are ten categories or a billion. Under pure DP, by contrast, a lower bound proved later by packing ([Lower bounds](10-lower-bounds.md)) says that choosing among $N$ possible values needs a lead of about $\ln N/\varepsilon$. So the stability-based release, which needs $\delta > 0$, removes the dependence on $N$, and this is how approximate DP gets around the packing bound. The stability-based histogram, described in the lower-bounds chapter, applies the same idea to every bin.

The release combines naturally with subsample-and-aggregate. Run a selection procedure on each of $B$ disjoint blocks of the data, split by a partition fixed in advance, and release the most common block answer by @eq-sens-stable. One replaced record changes one block answer, so it moves the counts of the block answers as a replaced record would. When the procedure is stable, most blocks agree and the answer comes out exact (Thakurta & Smith 2013 use this to select features).

### Example: the most common value

Take $\varepsilon = 1$ and $\delta = 10^{-6}$, so the threshold is $\ln(10^6) = 13.82$. A lead of 10 records gives $s = 4$, far below the threshold, so the release answers with probability only $2.73\times10^{-5}$. Raising the lead to 30 gives $s = 14$, just above the threshold, and an answer with probability 0.5842, while a lead of 40 gives $s = 19$ and probability 0.9972. So the smallest lead that answers with probability at least 0.95 is 35, whatever the number of categories. Under add/remove neighbours a record moves the lead by only 1, and the lead needed halves ([Exercise 9](../../book-exercises/part03-dp-foundations/07-sensitivity.md#sec-sol-sens-9)).

### Check yourself

- Why does the stability-based release cost $\varepsilon$ while propose-test-release costs $2\varepsilon$? *(When neighbours have the same answer, the release is a function of the test alone. Propose-test-release also releases a noisy value that differs between the neighbours, which costs a second $\varepsilon$.)*
- Is releasing $f(D)$ exactly safe on a dataset whose neighbour has a different answer? *(Only because it rarely happens: there $s = 0$, and the test passes with probability $\delta/2$. That rare exact release is what $\delta$ pays for.)*

## Subsample-and-aggregate {#sec-sens-saa}

### Context

Sometimes you cannot analyse $f$ at all: it is an estimator fitted by an optimizer, a trained model, or a pipeline you did not write. Subsample-and-aggregate (Nissim, Raskhodnikova & Smith 2007; Smith 2011) treats $f$ as a black box. It runs $f$ on disjoint pieces of the data and privately combines the answers. One record lives in one piece, so it can change only one answer, and the aggregator needs to be private only with respect to changing one of its inputs.

### Statement

Split the $n$ records into $B$ disjoint blocks of $n/B$ records, by a partition fixed in advance (for example by position, or by a hash of a record identifier). Compute the block answers $z_j = f(\text{block } j)$, clamp each to the range $[0, R]$ (replace values below 0 by 0 and above $R$ by $R$), and release their average plus Laplace noise. One record can move one clipped answer by at most $R$, so it moves the average by at most $R/B$, and that sets the noise:
$$M(D) = \frac1B\sum_{j=1}^B \operatorname{clip}_{[0,R]}(z_j) + \mathrm{Lap}\Big(\frac{R}{B\varepsilon}\Big).$$
This is $\varepsilon$-DP for any $f$.

How accurate is it, as an estimate of a target $\theta$? Three things go wrong, and they add up in the mean squared error. Each block sees only $n/B$ records, so $f$ on a block can be biased in a way that $f$ on the whole dataset is not; averaging the blocks keeps some random sampling error; and the Laplace noise has variance $2(R/(B\varepsilon))^2$. Writing $z$ for the output of $f$ on one block,
$$\underbrace{\big(\mathbb E[z] - \theta\big)^2}_{\text{bias of } f \text{ on } n/B \text{ records}} + \underbrace{\frac{\operatorname{Var}(z)}{B}}_{\text{sampling}} + \underbrace{2\Big(\frac{R}{B\varepsilon}\Big)^2}_{\text{noise}},$$ {#eq-sens-saa-error}
when the clipping is inactive and the blocks are independent samples.

### Derivation

1. **One record touches one block.** Replacing record $i$ changes only the block that contains it, because the partition does not depend on the values. So the vector $(z_1, \dots, z_B)$ changes in at most one coordinate.
2. **The aggregate has sensitivity $R/B$.** After clipping, that coordinate moves by at most $R$, the width of the clamping range, and dividing by $B$ in the average divides that by $B$.
3. **Privacy:** Fact 1 with $\Delta = R/B$.
4. **Error.** The released value minus $\theta$ is the sum of three pieces: $\mathbb E[z] - \theta$ (a constant, the bias), $\frac1B\sum_j(z_j - \mathbb E[z])$ (mean zero; it is an average of $B$ independent terms, each with variance $\operatorname{Var}(z)$, so its variance is $\operatorname{Var}(z)/B$) and the Laplace noise (mean zero, variance $2(R/(B\varepsilon))^2$, since a Laplace variable has variance twice the square of its scale). Squaring the sum gives the three squares plus cross terms. Each cross term is a product involving at least one mean-zero piece that is independent of the other factor, so it averages to zero. That leaves @eq-sens-saa-error.

### Interpretation

The number of blocks sets the trade-off: more blocks means less noise but more bias. The noise falls like $1/B$, because one record's influence on the average is diluted over more blocks. The sampling term hardly moves: for estimators whose variance scales like one over the sample size, a block estimate has variance about $B$ times that of $f$ on all $n$ records, and averaging $B$ of them divides it by $B$ again, so the term stays near $\operatorname{Var}(f \text{ on } n \text{ records})$. The bias of $f$ on small blocks grows with $B$, because the blocks get smaller. The best $B$ balances the first and the last.

The privacy proof never looks inside $f$. This is what makes the method useful for trained models: an ensemble of "teacher" models, each trained on a disjoint shard and combined by a noisy vote (private aggregation of teacher ensembles, Papernot et al. 2017), is subsample-and-aggregate with a vote as the aggregator.

The range $R$ is still a global bound, on the output of $f$ rather than on the data, and a loose $R$ means large noise. Nissim, Raskhodnikova & Smith replace the clipped average by a "center of attention" aggregator with smooth sensitivity, which adapts to how tightly the $z_j$ cluster. Smith (2011) shows that for asymptotically normal estimators a version of this method loses nothing in the leading-order error.

### Example: estimating a rate

You observe $n = 10{,}000$ inter-arrival times, independent and $\mathrm{Exp}(\theta)$-distributed with rate $\theta = 1$, and want to estimate $\theta$ privately with $\varepsilon = 1$. The maximum-likelihood estimate is one over the sample mean, $1/\bar x$, whose global sensitivity is infinite: a dataset of tiny times has a huge $1/\bar x$, and one record changes it by an arbitrary amount.

To fill in @eq-sens-saa-error you need the mean and variance of $1/\bar x$ on a block of $n_b = n/B$ records. The block sum $n_b\bar x$ is a sum of $n_b$ independent exponentials with rate $\theta$, which has the gamma distribution with shape $n_b$ and rate $\theta$: $n_b\bar x \sim \mathrm{Gamma}(n_b, \theta)$. The inverse moments of a gamma variable $G$ with shape $a$ are $\mathbb E[1/G] = \theta/(a - 1)$ and $\mathbb E[1/G^2] = \theta^2/((a - 1)(a - 2))$: integrate $g^{-1}$ or $g^{-2}$ against the gamma density $\theta^ag^{a-1}e^{-\theta g}/\Gamma(a)$, recognize what is left as a gamma density with shape $a - 1$ or $a - 2$, and use $\Gamma(a) = (a - 1)\Gamma(a - 1)$. Since $z = 1/\bar x = n_b/G$, multiplying by $n_b$ and $n_b^2$ gives
$$\mathbb E[z] = \frac{n_b\theta}{n_b - 1}, \qquad \text{bias } \frac{\theta}{n_b - 1}, \qquad \operatorname{Var}(z) = \frac{n_b^2\theta^2}{(n_b - 1)^2(n_b - 2)}.$$
The variance is $\mathbb E[z^2] - \mathbb E[z]^2 = n_b^2\theta^2\big(\frac1{(n_b-1)(n_b-2)} - \frac1{(n_b-1)^2}\big)$, which simplifies to the form shown. With the clamping range $[0, 5]$, so $R = 5$, @tbl-sens-saa and @fig-sens-saa-blocks show the trade-off. For example, with $B = 10$ blocks the noise standard deviation is $\sqrt2\,R/(B\varepsilon) = \sqrt2 \times 0.5 = 0.7071$, and the bias on blocks of 1,000 is $1/999 = 0.0010$.

| Blocks $B$ | Block size | Bias | Sampling sd | Noise sd | Root-mean-square error | Simulated |
| --- | --- | --- | --- | --- | --- | --- |
| 10 | 1,000 | 0.0010 | 0.0100 | 0.7071 | 0.7072 | 0.7205 |
| 100 | 100 | 0.0101 | 0.0102 | 0.0707 | 0.0722 | 0.0733 |
| 250 | 40 | 0.0256 | 0.0105 | 0.0283 | 0.0396 | 0.0401 |
| 1,000 | 10 | 0.1111 | 0.0124 | 0.0071 | 0.1120 | 0.1114 |

: Error of subsample-and-aggregate for the rate of an exponential distribution, $n = 10{,}000$, $\theta = 1$, $R = 5$, $\varepsilon = 1$: the three terms of @eq-sens-saa-error, their combination, and a simulation with 1,000 repetitions. {#tbl-sens-saa}

The best $B$ on the script's grid is 250, with error $0.0396$, where the bias and the noise are of similar size. A non-private estimate has standard deviation about $\theta/\sqrt n = 0.0100$, so privacy at $\varepsilon = 1$ costs a factor of four here, for an estimator whose sensitivity you never had to bound.

![Root-mean-square error of subsample-and-aggregate for the exponential rate against the number of blocks $B$ (formula and simulation), with its three components: the bias of the estimator on small blocks, the Laplace noise, and the sampling error.](figures/sens_saa_blocks.svg){#fig-sens-saa-blocks}

### Check yourself

- Why must the partition into blocks not depend on the data values? *(Otherwise replacing one record could reshuffle many records between blocks and change many $z_j$, breaking step 1.)*
- Why does the sampling term in @tbl-sens-saa barely move with $B$? *(The average of $B$ block estimates, each with variance about $\theta^2 B/n$, has variance about $\theta^2/n$ whatever $B$ is. Only the higher-order terms of the inverse-gamma variance depend on the block size.)*

## Clipping {#sec-sens-clip}

### Context

The mean is the most common statistic of all, and on unbounded data its global sensitivity is infinite. **Clipping** forces a bound: replace each value $x$ by $\min(x, C)$ before averaging. The sensitivity becomes $C/n$, and Laplace noise of scale $C/(n\varepsilon)$ makes the mean private. Nothing is free, though. Clipping changes the quantity you estimate, and the choice of $C$ trades that bias against the noise. The same move, applied to per-example gradients rather than to values, is the heart of private model training in later parts.

### Statement

Take non-negative data $X_1, \dots, X_n$ drawn independently from a distribution with mean $\mu$. The clipped mean averages the clipped values and adds noise for sensitivity $C/n$:
$$M(D) = \frac1n\sum_{i=1}^n\min(x_i, C) + \mathrm{Lap}\Big(\frac{C}{n\varepsilon}\Big)$$
It is $\varepsilon$-DP. Its error has the same three parts as in subsample-and-aggregate. Clipping cuts off the part of each value above $C$, written $(x - C)_+ = \max(x - C, 0)$, and its average is the bias. The clipped values still vary from sample to sample, and the noise adds its own variance. So the mean squared error is
$$\operatorname{err}(C)^2 = \underbrace{\mathbb E[(X - C)_+]^2}_{\text{bias}^2} + \underbrace{\frac{\operatorname{Var}(\min(X, C))}{n}}_{\text{sampling}} + \underbrace{\frac{2C^2}{n^2\varepsilon^2}}_{\text{noise}}.$$ {#eq-sens-clip-error}
Raising $C$ shrinks the bias and grows the noise, so the best $C$ is where the two rates of change balance, which is where the derivative of @eq-sens-clip-error in $C$ is zero. Here $\Pr[X > C]$ is the fraction of values that clipping touches. The best $C$ solves
$$\Pr[X > C]\Big(\mathbb E[(X - C)_+] - \frac{C - \mathbb E[\min(X, C)]}{n}\Big) = \frac{2C}{n^2\varepsilon^2}.$$ {#eq-sens-clip-opt}

### Derivation

1. **Sensitivity.** Each clipped value lies in $[0, C]$, so replacing one record changes the sum by at most $C$ and the mean by at most $C/n$. Fact 1 gives $\varepsilon$-DP.
2. **The bias.** Clipping removes exactly the excess above $C$: $x - \min(x, C) = (x - C)_+$. Taking expectations, $\mu - \mathbb E[\min(X, C)] = \mathbb E[(X - C)_+]$: the mass that clipping cuts off.
3. **Error decomposition.** As in the subsample-and-aggregate derivation, the error is a constant (minus the bias), a mean-zero sampling term with variance $\operatorname{Var}(\min(X, C))/n$, and independent Laplace noise with variance $2(C/(n\varepsilon))^2$. Squaring and averaging kills the cross terms, since each contains a mean-zero factor independent of the other: @eq-sens-clip-error.
4. **Derivatives in $C$.** Raising $C$ by a small amount $h$ raises $\min(X, C)$ by $h$ exactly when $X > C$ and leaves it alone otherwise, so $\frac{\mathrm d}{\mathrm dC}\mathbb E[\min(X, C)] = \Pr[X > C]$. By the same reasoning $\frac{\mathrm d}{\mathrm dC}\mathbb E[\min(X, C)^2] = 2C\Pr[X > C]$. The bias is $\mathbb E[(X - C)_+] = \mu - \mathbb E[\min(X, C)]$, whose derivative is $-\Pr[X > C]$, so by the chain rule the bias term has derivative $-2\,\mathbb E[(X - C)_+]\Pr[X > C]$. The variance is $\mathbb E[\min(X, C)^2] - \mathbb E[\min(X, C)]^2$, so the sampling term has derivative $\frac1n\big(2C\Pr[X > C] - 2\,\mathbb E[\min(X, C)]\Pr[X > C]\big) = \frac2n\Pr[X > C]\big(C - \mathbb E[\min(X, C)]\big)$. The noise term has derivative $\frac{4C}{n^2\varepsilon^2}$.
5. **Set the sum to zero** and divide by 2: @eq-sens-clip-opt.

### Interpretation

The bias does not shrink with $n$, but the noise does. So the best $C$ grows with $n$: with more data you can afford to clip less. For large $n$ the sampling part of @eq-sens-clip-opt, which has an extra $1/n$, drops out, and the condition reads $\Pr[X > C]\,\mathbb E[(X - C)_+] \approx 2C/(n\varepsilon)^2$. The right side shrinks as $n\varepsilon$ grows, so the left side must shrink too, which pushes $C$ out into the tail.

The bias has a sign. Clipping only from above always lowers values, so it always underestimates. For data on both sides of a centre $c$, clipping to $[c - C, c + C]$ cuts off mass on both sides. The two cuts cancel exactly when the distribution is symmetric about $c$, and they can also happen to balance for some asymmetric ones. What matters is where the clip interval sits, not where 0 is. A clip interval centred far from the bulk of the data cuts one side much more than the other, which is why the centre itself is usually estimated privately first, the subject of [Private mean estimation](11-mean-estimation.md).

Choosing $C$ is itself a data-dependent decision. Setting $C$ to the observed maximum, or to the empirical 99th percentile, makes the noise scale a function of the data, and the leak of @sec-sens-local returns. Choose $C$ from domain knowledge, or spend part of the budget on a private quantile: the [exponential mechanism](05-selection.md), inverse sensitivity (@sec-sens-inverse), or the smooth-sensitivity and propose-test-release methods above applied to an order statistic.

Clipping is the honest version of "bounded data". Every "assume $x \in [0, \Lambda]$" in earlier chapters is a clip in disguise, and its bias is the price of that assumption.

### Example: mean income

Incomes (in thousands) follow a log-normal distribution with $\ln X \sim \mathcal N(\ln 40, 0.8^2)$: the log of an income is normal. The median is $e^{\ln 40} = 40$, because the exponential preserves the median. The mean is larger, $e^{\ln 40 + 0.32} = 55.085$, because a log-normal with $\ln X \sim \mathcal N(\nu, s^2)$ has mean $e^{\nu + s^2/2}$ and here $s^2/2 = 0.64/2 = 0.32$; the gap reflects a heavy right tail.

Every term of @eq-sens-clip-error needs averages of $X$ and $X^2$ over the values below $C$. For a log-normal these partial moments have a closed form, $\mathbb E[X^j\mathbf 1[X \le c]] = e^{j\nu + j^2s^2/2}\,\Phi\big((\ln c - \nu - js^2)/s\big)$, where $\Phi$ is the standard normal cumulative distribution function and $\mathbf 1[X \le c]$ is 1 when $X \le c$ and 0 otherwise. To get it, write $X = e^Y$ with $Y$ normal, so the integral is $\int_{-\infty}^{\ln c}e^{jy}\varphi(y)\,\mathrm dy$ with $\varphi$ the normal density of $Y$; completing the square in the exponent turns $e^{jy}\varphi(y)$ into $e^{j\nu + j^2s^2/2}$ times a normal density with mean shifted to $\nu + js^2$, whose integral up to $\ln c$ is the $\Phi$ term (as in [Probability distributions](../part01-prerequisites/01-distributions.md)). @tbl-sens-clip uses these with $n = 1000$ and $\varepsilon = 1$.

| Clipping norm $C$ | $\Pr[X > C]$ | Bias | Sampling sd | Noise sd | Root-mean-square error | Simulated |
| --- | --- | --- | --- | --- | --- | --- |
| 40 (the median) | 0.5000 | 23.415 | 0.335 | 0.057 | 23.417 | 23.411 |
| 257.23 (99th percentile) | 0.0100 | 0.924 | 1.462 | 0.364 | 1.767 | 1.747 |
| 331.34 (optimal) | 0.0041 | 0.438 | 1.538 | 0.469 | 1.666 | 1.649 |
| 1000 | 0.0000 | 0.006 | 1.645 | 1.414 | 2.169 | 2.134 |

: Error of the clipped private mean of log-normal incomes, $n = 1000$, $\varepsilon = 1$, for four clipping norms: the three terms of @eq-sens-clip-error, their combination, and a simulation with 5,000 repetitions. {#tbl-sens-clip}

Clipping at the median is a disaster: half the incomes are cut back, and the bias of 23 dwarfs everything. The optimum, $C^* = 331.34$, cuts off only 0.4% of the population and has total error $1.666$, only 1% above the standard deviation $1.649$ of the unclipped, non-private mean (the ratio is $1.666/1.649 \approx 1.01$). The simulated error at the optimum happens to print as 1.649 too. That is a coincidence of the 5,000 repetitions, whose own standard error is about 1%, and the formula's 1.666 is the value to trust: privacy costs a little here, not nothing. Above the optimum, the noise takes over: its standard deviation $\sqrt2\,C/(n\varepsilon)$ grows linearly in $C$, reaching $1.414$ at $C = 1000$ (@fig-sens-clip-tradeoff, left). The optimum grows with $n$, as the interpretation predicted: $177.5$ at $n = 100$, $557.3$ at $n = 10{,}000$ and $876.3$ at $n = 100{,}000$ (@fig-sens-clip-tradeoff, right).

![Left: root-mean-square error of the clipped private mean of log-normal incomes against the clipping norm $C$, with its bias, noise and sampling components and simulated points; the dotted line marks the optimum $C^* = 331$. Right: the optimal clipping norm as a function of the dataset size.](figures/sens_clip_tradeoff.svg){#fig-sens-clip-tradeoff}

### Check yourself

- Under the add/remove relation, why is a clipped *sum* easier to release than a clipped mean? *(Adding or removing one record changes the sum by at most $C$, but it also changes $n$, so the mean's denominator is not public. One releases a noisy sum and a noisy count and divides.)*
- If you double $n$, roughly how does the noise term change, and which way does $C^*$ move? *(The noise standard deviation halves at fixed $C$, so the balance in @eq-sens-clip-opt shifts toward a larger $C^*$ with less bias.)*

## Code: check every number by computation

The script [code/sensitivity.py](../../code/part03-dp-foundations/sensitivity.py) computes every number in this chapter and draws its figures. Local and smooth sensitivity, the noise parameters, propose-test-release, inverse sensitivity, the stability-based release, subsample-and-aggregate and the clipped mean are each small typed functions; the table calls them once and the figures loop over them. From the repository root:

```bash
uv run python code/part03-dp-foundations/sensitivity.py
```

@tbl-sens-results shows the output with seed 0.

| Experiment | Result |
| --- | --- |
| Local-sensitivity noise on $D$, $D'$ of @tbl-sens-leak, $\varepsilon = 1$ | $\mathrm{LS} = 10$ and 60; $\delta(1) = 0.477$ ($D'$ against $D$) and $0.215$ |
| $A^{(k)}$ for $D = (1, 3, 4, 6, 9)$, $\Lambda = 10$ | Formula equals brute force over $11^5$ datasets for every $k$; $S^*_{0.5} = 3.03$ at $k = 1$ |
| Noise parameters, $\varepsilon = 1$, $\delta = 10^{-6}$ | Cauchy $\alpha = \beta = 0.1667$, loss bound 0.8333; Laplace $\alpha = 0.5$, $\beta = 0.0322$ |
| Median of 1,001 ages | local sensitivity 0.1386; Cauchy scale 0.979 (simulated median error 0.980); Laplace median error 0.713; global 69.31 |
| Propose-test-release, $b = 1$ | $n = 1001$: distance 21, pass probability 0.999621; $n = 101$: distance 1, pass probability $1.36\times10^{-6}$ |
| Subsample-and-aggregate, exponential rate | Best $B = 250$ on the grid, error 0.0396, against 0.0100 without privacy |
| Inverse sensitivity | Walkthrough gaps next to the median: probability 0.4432; 1,001 ages: median error 0.0783 (simulated 0.0784); 101 ages: 0.696 |
| Stability-based release of the mode | Pass probability 0.5842 at a lead of 30, 0.9972 at 40; lead 35 for probability 0.95 |
| Clipped mean of incomes, $n = 1000$ | $C^* = 331.34$, error 1.666 (simulated 1.649, within its 1% Monte Carlo error of 1.666); 99th percentile 1.767; median 23.417 |

: Results of the chapter's script with seed 0. {#tbl-sens-results}

[tests/test_sensitivity.py](../../tests/test_sensitivity.py) checks that the output still matches this table.

**What to take from it.** The first lesson is that a data-dependent noise scale can leak more than the answer: in the leak example both datasets have the same median, so the whole $\delta = 0.477$ comes from the spread of the noise. The repair for the median rests on an exact formula, and the $O(n^2)$ formula for $A^{(k)}$ matches brute force at every distance $k$, so the smooth sensitivities it produces can be trusted. With it, smooth sensitivity and propose-test-release buy about two orders of magnitude over global sensitivity for the median of a thousand points, though they lose to it on tiny datasets, where a few changes can open a gap across the whole range. Inverse sensitivity does better still, by another factor of twelve on the ages, because it never has to pay for a noise scale. Finally, every method ends in a bias–noise trade-off you can compute: the number of blocks and the clipping norm both have optima you can find in closed form before touching the data.

## References

- Nissim, Raskhodnikova & Smith, "Smooth Sensitivity and Sampling in Private Data Analysis" (STOC 2007; full version on the authors' pages): local and smooth sensitivity, admissible noise (Lemmas 2.6–2.9), the median (Proposition 3.4) and the original sample-and-aggregate framework
- Dwork & Lei, "Differential Privacy and Robust Statistics" (STOC 2009): propose-test-release, and private robust estimators of scale, location and regression built on it
- Dwork & Roth, *The Algorithmic Foundations of Differential Privacy* (2014), ch. 7: propose-test-release, stability-based release and subsample-and-aggregate in textbook form
- Near & Abuah, *Programming Differential Privacy*, chapter on local sensitivity: the same ideas with short code
- Asi & Duchi, "Instance-optimality in Differential Privacy via Approximate Inverse Sensitivity Mechanisms" (NeurIPS 2020): the inverse-sensitivity mechanism and its near-optimality on every dataset
- Thakurta & Smith, "Differentially Private Feature Selection via Stability Arguments, and the Robustness of the Lasso" (COLT 2013): stability-based release, used for model selection
- Smith, "Privacy-Preserving Statistical Estimation with Optimal Convergence Rates" (STOC 2011): subsample-and-aggregate for asymptotically normal estimators
- Bun & Steinke, "Average-Case Averages: Private Algorithms for Smooth Sensitivity and Mean Estimation" (NeurIPS 2019): better noise distributions for smooth sensitivity, under concentrated DP
- Amin, Kulesza, Muñoz Medina & Vassilvitskii, "Bounding User Contributions: A Bias-Variance Trade-off in Differential Privacy" (ICML 2019): the clipping trade-off when users contribute many records
- Papernot, Abadi, Erlingsson, Goodfellow & Talwar, "Semi-supervised Knowledge Transfer for Deep Learning from Private Training Data" (ICLR 2017): subsample-and-aggregate for model training

## Cheat sheet

@tbl-sens-cheat-sheet collects the results. Datasets have fixed size $n$ and neighbours differ by one replaced record.

| Result | Statement | Privacy |
| --- | --- | --- |
| Local sensitivity | $\mathrm{LS}_f(D) = \max_{d(D, D') = 1}\lvert f(D) - f(D')\rvert$; noise scaled by it is **not** private | — |
| Median | $\mathrm{LS} = \max(x_m - x_{m-1}, x_{m+1} - x_m)$; $\Delta = \Lambda$ | — |
| Smooth sensitivity | $S^*_\beta(D) = \max_{D''}\mathrm{LS}(D'')e^{-\beta d(D, D'')}$, the smallest $\beta$-smooth upper bound | — |
| Median, exact | $S^*_\beta = \max_{k \le n}e^{-k\beta}\max_{0 \le t \le k+1}(x_{m+t} - x_{m+t-k-1})$, $O(n^2)$ | — |
| Cauchy-like noise | $f + (S/\alpha)Z$, $h \propto 1/(1 + \lvert z\rvert^\gamma)$, $\gamma\alpha + (\gamma + 1)\beta \le \varepsilon$; Cauchy: $6S^*_{\varepsilon/6}/\varepsilon$ | $\varepsilon$ |
| Laplace noise | $f + (S/\alpha)\,\mathrm{Lap}(1)$, $\beta + e^\beta\alpha + (e^\beta - 1)\ln(1/\delta) \le \varepsilon$ | $(\varepsilon, \delta)$ |
| Propose-test-release | $\bot$ if $\operatorname{dist}_b + \mathrm{Lap}(1/\varepsilon) \le \ln(1/\delta)/\varepsilon$, else $f + \mathrm{Lap}(b/\varepsilon)$ | $(2\varepsilon, \delta/2)$ |
| Inverse sensitivity | Density $\propto e^{-\varepsilon\,\mathrm{len}_f(D, t)/2}$; median: $\mathrm{len}$ is the number of steps from the gap to the median | $\varepsilon$ |
| Stability-based release | $\bot$ if $s + \mathrm{Lap}(1/\varepsilon) \le \ln(1/\delta)/\varepsilon$, else $f(D)$ exactly; mode: $s = \max(0, \lceil g/2\rceil - 1)$ | $(\varepsilon, \delta/2)$ |
| Subsample-and-aggregate | Mean of $B$ clipped block outputs $+ \mathrm{Lap}(R/(B\varepsilon))$; error = bias of small blocks + sampling + noise | $\varepsilon$ |
| Clipped mean | $\frac1n\sum\min(x_i, C) + \mathrm{Lap}(C/(n\varepsilon))$; bias $\mathbb E[(X - C)_+]$ | $\varepsilon$ |

: Summary of the results in this chapter. {#tbl-sens-cheat-sheet}

**Rules of thumb**

- If the noise scale depends on the data, ask what the noise scale reveals. If you cannot answer, use one of the repairs of this chapter.
- Pure DP with a data-dependent scale needs polynomial tails; light tails need $\delta$ and a much smaller $\beta \approx \varepsilon/(2\ln(1/\delta))$.
- Smooth sensitivity pays off once the data are dense around the statistic: for the median, once $n$ is in the hundreds. For medians and quantiles, inverse sensitivity is simpler and more accurate.
- A discrete answer that is stable can be released exactly, at the cost of $\varepsilon$ and a refusal when the lead is below about $2\ln(1/\delta)/\varepsilon$.
- Choose clipping norms and block counts before looking at the data, by the error formula, or with a separate slice of the privacy budget.

**Practice.** Exercises for this chapter, each with a worked solution, are in [the exercise book](../../book-exercises/part03-dp-foundations/07-sensitivity.md).
