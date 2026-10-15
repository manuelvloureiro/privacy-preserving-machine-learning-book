# Advanced and optimal composition

## Why this matters

A differentially private system rarely answers just one query. For example, a dashboard may refresh a hundred counts, and a training run takes thousands of noisy gradient steps. Each step on its own satisfies differential privacy (DP), from [The definition of differential privacy](01-definition.md).

DP compares two **neighbouring** datasets $D$ and $D'$, which differ in one person's record. It has a privacy level $\varepsilon$ and a small slack $\delta$. A step is $(\varepsilon, \delta)$-DP if any set of outputs is at most $e^\varepsilon$ times likelier on one dataset than on the other, plus $\delta$.

What you need, though, is one guarantee for the whole run. It should be as small as the truth allows, because the total decides how much noise each step must add.

[Post-processing, group privacy and composition](02-properties.md) gave a first answer, called **basic composition**. It says that $k$ steps that are each $\varepsilon$-DP are together $k\varepsilon$-DP. The reason is that each step can multiply the probability of an output by at most $e^\varepsilon$, and the $k$ factors multiply. That answer is correct, but for many steps it is far too pessimistic. A hundred queries at $\varepsilon = 0.1$ add up to $k\varepsilon = 10$. Read literally, the run could then multiply an attacker's odds that the data are $D$ rather than $D'$ by up to $e^{10}$, which is about 22,000. A promise that weak is not worth making. Yet if you allow the whole run a slack of $\delta = 10^{-5}$, the same hundred queries are $(4.31, 10^{-5})$-DP, as @sec-comp-optimal computes. Moreover, no analysis that knows only each query's privacy parameters can give a smaller number.

The gap comes from the way basic composition counts. It assumes that every step produces its most revealing output in the same run. But the evidence each step leaks about one person is a number, its privacy loss, and a whole run leaks the sum of these numbers (@sec-comp-prv). A sum of many small random terms almost never reaches its largest possible value. So every result in this chapter bounds the likely range of that sum instead of its maximum, and uses $\delta$ to pay for the rare runs beyond it.

## The privacy loss adds up {#sec-comp-prv}

Run a mechanism $M$ on the dataset $D$ and look at its output $y$. Write $p_{M(D)}(y)$ for the probability, or the density, of that output when the mechanism runs on $D$. Similarly, $p_{M(D')}(y)$ is its probability on $D'$. The **privacy loss** of the output is the logarithm of how much more likely it was under $D$ than under $D'$,
$$L(y) = \ln\frac{p_{M(D)}(y)}{p_{M(D')}(y)}.$$ {#eq-comp-prv}
The loss is positive when $y$ favours $D$, negative when it favours $D'$, and zero when the output says nothing about which dataset was used. Since the output is random, so is its loss, whose distribution is the **privacy loss distribution**.

Pure $\varepsilon$-DP, the case $\delta = 0$, says exactly that no output is more than $e^\varepsilon$ times likelier under one dataset than under the other. In terms of the loss, every output has a loss between $-\varepsilon$ and $\varepsilon$.

Now run $k$ steps, one after the other. On either dataset, the probability of the whole sequence of outputs is the probability of the first output, times the probability of the second given the first, and so on. So the ratio of the two probabilities is the product of the step-by-step ratios, and the logarithm turns that product into a sum. The total privacy loss of the run is therefore the sum of the per-step losses. This needs no independence between the steps. Each factor is the probability of one step's output given the earlier outputs, so a step is allowed to depend on what came before.

@tbl-comp-two-rr shows the sum for the simplest case. Randomized response, from [The definition of differential privacy](01-definition.md), reports a person's bit truthfully with probability $\frac34$ and flips it otherwise, which makes it $\ln 3$-DP. Run it twice on a person whose bit is 0 in $D$ and 1 in $D'$. A report of 0 is then three times as likely under $D$ as under $D'$, so its loss is $\ln 3$. A report of 1 has the two probabilities swapped, so its loss is $-\ln 3$.

| Output $(y_1, y_2)$ | $(0, 0)$ | $(0, 1)$ | $(1, 0)$ | $(1, 1)$ |
| --- | --- | --- | --- | --- |
| Probability under $D$ (bit 0) | 9/16 | 3/16 | 3/16 | 1/16 |
| Probability under $D'$ (bit 1) | 1/16 | 3/16 | 3/16 | 9/16 |
| Privacy loss | $2\ln 3$ | 0 | 0 | $-2\ln 3$ |

: Two rounds of randomized response at $\varepsilon = \ln 3$. The privacy loss of each output is the sum of the two per-round losses. {#tbl-comp-two-rr}

The output $(0, 1)$ has a loss of zero, because the first report favours $D$ exactly as much as the second favours $D'$. Only the output $(0, 0)$, where both reports point to $D$, reaches the largest loss, $2\ln 3$. In general, if each step is $\varepsilon$-DP whatever happened before it, each term is at most $\varepsilon$ in absolute value, so the total is at most $k\varepsilon$. That is basic composition, and the rest of the chapter bounds the same sum more cleverly.

## From the privacy loss to $\delta$ {#sec-comp-delta}

To turn the total loss into a guarantee, you need to know which $\delta$ it implies. Look first at one direction, from $D$ to $D'$, with outputs drawn from the mechanism run on $D$. The definition allows each output to be up to $e^\varepsilon$ times likelier under $D$ than under $D'$. Only the excess beyond that factor has to be covered by $\delta$. So an output whose loss $L$ is at most $\varepsilon$ is within the allowed factor, and it costs nothing. An output with a larger loss has, under $D'$, a fraction $e^{-L}$ of its probability under $D$. The allowed amount is $e^\varepsilon$ times that, a fraction $e^{\varepsilon - L}$, so the excess is the remaining fraction $1 - e^{\varepsilon - L}$ of its probability.

That fraction is close to zero when the loss is just above $\varepsilon$. It is the whole probability when the loss is infinite, which happens for an output that is impossible under $D'$. Adding up these costs over all outputs gives the smallest $\delta$ that works at a given $\varepsilon$ in this direction. Write $(x)_+$ for the larger of a number $x$ and zero. Then
$$\delta(\varepsilon) = \mathbb E\big[(1 - e^{\varepsilon - L})_+\big] \le \Pr[L > \varepsilon],$$ {#eq-comp-delta}
where the average is over the outputs of the mechanism run on $D$. This average is the hockey-stick divergence of [Divergences](../part01-prerequisites/03-divergences.md), rewritten in terms of the loss. The inequality on the right holds because each output's cost is a fraction of its probability, never more, and it is zero unless the loss exceeds $\varepsilon$. DP asks for this bound in both directions, with $D$ and $D'$ swapped, and for every pair of neighbours.

The consequence is that a composition theorem only needs to know how the total loss is distributed. The probability that the loss exceeds $\varepsilon$ is the simple bound, and the average in @eq-comp-delta is the exact value.

## The average privacy loss {#sec-comp-mean}

Each step's loss can be as large as $\varepsilon$, but its average is much smaller. When $\varepsilon$ is small, an output that favours $D$ is only slightly more likely than one that favours $D'$, so positive and negative losses nearly cancel. What remains is of order $\varepsilon^2$, because it is a product of two small numbers. Each output's loss is at most $\varepsilon$, and the two output distributions differ in total by at most $e^\varepsilon - 1$, which is also about $\varepsilon$. The average loss over the outputs on $D$ is, by definition, the Kullback–Leibler (KL) divergence from [Divergences](../part01-prerequisites/03-divergences.md).

The cancellation argument needs a condition, namely that every output's loss lies between $-\varepsilon$ and $\varepsilon$, which is pure $\varepsilon$-DP. Without it, a single output with a huge loss could outweigh all the cancelling ones. Under that condition, the average loss is at most
$$\mathbb E[L] \le \varepsilon(e^\varepsilon - 1).$$ {#eq-comp-drv-mean}
At $\varepsilon = 0.1$ this bound is 0.0105, about a tenth of the largest loss. But the bound is not sharp. Its best possible version, which @sec-comp-optimal derives, is $\varepsilon\tanh(\varepsilon/2)$. That is about half as large for small $\varepsilon$, for example 0.0050 at $\varepsilon = 0.1$.

The average loss of a step is never negative, since it is a KL divergence. On average, each step points a little more towards the truth than away from it. Over $k$ steps these averages add up to a steady **drift**. With the sharp bound the drift is at most about $k\varepsilon^2/2$, which grows much more slowly than the worst case $k\varepsilon$.

## The loss of composed randomized response {#sec-comp-loss}

Randomized response composed $k$ times makes all of this concrete, and @sec-comp-optimal shows that it is also the worst case. Randomized response at level $\varepsilon$ tells the truth with probability $t = e^\varepsilon/(1 + e^\varepsilon)$, the value that makes a truthful report exactly $e^\varepsilon$ times likelier than a lie. Take a person whose bit is 0 in $D$ and 1 in $D'$. A report of 0 is then the truth under $D$, so it points to $D$ and has loss $+\varepsilon$. A report of 1 is a lie, so it points to $D'$ and has loss $-\varepsilon$.

So the whole loss is fixed by how many reports point to $D'$, and this chapter writes $J$ for that number. Each lie subtracts $\varepsilon$ from the loss and each truthful report adds it, so the total loss is $(k - 2J)\varepsilon$. Under $D$ each report lies with probability $1 - t$, independently of the others, so the count $J$ is binomial. Each report adds $\varepsilon$ with probability $t$ and subtracts it otherwise, so its average is $\varepsilon(2t - 1)$. Substituting the value of $t$ turns this average into $\varepsilon\tanh(\varepsilon/2)$. Adding up the $k$ reports gives the mean and the standard deviation of the loss:
$$\mathbb E[L] = k\varepsilon\tanh(\varepsilon/2), \qquad \operatorname{sd}(L) = \varepsilon\sqrt{k\big(1 - \tanh^2(\varepsilon/2)\big)}.$$
The mean is $k$ times the average loss of one report, while the standard deviation grows only like $\sqrt k$, because the reports are independent and their variances add. The largest loss, $k\varepsilon$, needs every report to be truthful, which happens with probability $t^k$.

Take the hundred queries at $\varepsilon = 0.1$, answered with randomized response, so that each report is truthful with probability $t = 0.5250$. The total loss has mean 0.4996 and standard deviation 0.9988. Both are far below basic composition's $k\varepsilon = 10$, which lies 9.5 standard deviations above the mean. Reaching it needs all 100 reports to be truthful, which happens with probability $1.0\times10^{-28}$. So basic composition guards against an event that will never happen in practice.

![The distribution of the total privacy loss of 100 rounds of randomized response at $\varepsilon = 0.1$, under $D$. The vertical lines mark its mean, the $\varepsilon'$ of optimal composition at $\delta' = 10^{-5}$, the $\varepsilon'$ of advanced composition, and basic composition's $k\varepsilon = 10$, which the loss reaches with probability $10^{-28}$.](figures/comp_loss_distribution.svg){#fig-comp-loss-distribution}

What does happen is set by the mean and the spread (@fig-comp-loss-distribution). A sum of many small, independent, bounded terms is close to a normal distribution, as the central limit theorem says. From here on, a prime marks the parameters of the whole run, $\varepsilon'$ and $\delta'$, to keep them apart from each step's parameters. The optimal $\varepsilon'$ at $\delta' = 10^{-5}$, which @sec-comp-optimal computes, is 4.31. It sits 3.8 standard deviations above the mean, where only a sliver of a normal distribution remains. The rest of the chapter turns this picture into guarantees.

Two consequences follow for practice. First, the spread grows like $\sqrt k\,\varepsilon$, but the drift grows like $k\varepsilon^2/2$. The drift divided by the spread is about $\varepsilon\sqrt k/2$. So once $\varepsilon\sqrt k$ is of order 1 or more, the drift is comparable to the spread or larger. The total then grows like $k$ again, but at the slower rate of about $\varepsilon^2/2$ per step (@fig-comp-epsilon-vs-k). Second, the saving needs $\delta > 0$. With $\delta = 0$ the guarantee must cover even the all-truthful output, which has a positive probability, so you are left with $k\varepsilon$.

## Advanced composition {#sec-comp-advanced}

Dwork, Rothblum & Vadhan (2010) were the first to turn this concentration into a theorem, which needs only each step's guarantee. Their idea is to split the total loss into the drift, which @eq-comp-drv-mean bounds, and the **fluctuation**, which is the total loss minus the drift.

Take pure $\varepsilon$-DP steps first. The fluctuation is then a sum of $k$ terms that each average zero given the past. Each term lies in a range of width $2\varepsilon$, whatever happened before, because each step's loss lies between $-\varepsilon$ and $\varepsilon$. The terms are not independent, since each step may depend on earlier outputs. That is where Azuma's inequality from [Concentration inequalities](../part01-prerequisites/02-concentration.md) comes in. It bounds how far a running total can wander when its steps average zero given the past and lie in intervals whose lengths are fixed in advance. For ranges of width $2\varepsilon$, it bounds the chance that the fluctuation reaches a level $s$:
$$\Pr[\text{fluctuation} \ge s] \le \exp\Big(-\frac{s^2}{2k\varepsilon^2}\Big).$$
Set the right side equal to a small probability $\delta'$ of your choice and solve for the level. The fluctuation then exceeds
$$s = \varepsilon\sqrt{2k\ln(1/\delta')}$$
with probability at most $\delta'$.

Adding the drift to this bound on the fluctuation gives the theorem. Take $k$ steps that are each $(\varepsilon, \delta)$-DP, even when each is chosen after seeing earlier outputs. Then for every $\delta' > 0$ the steps are together $(\varepsilon', k\delta + \delta')$-DP, with
$$\varepsilon' = \varepsilon\sqrt{2k\ln(1/\delta')} + k\varepsilon(e^\varepsilon - 1).$$ {#eq-comp-advanced}
Steps with $\delta > 0$ are handled through the worst-case mechanism of @sec-comp-optimal. Such a step behaves like a pure $\varepsilon$-DP step, except that with probability $\delta$ it may reveal the dataset outright. The chance that any of the $k$ steps does so is at most $k\delta$, and that is the extra $k\delta$ in the theorem. With the sharp average loss of @sec-comp-optimal, the drift term improves:
$$\varepsilon' = \varepsilon\sqrt{2k\ln(1/\delta')} + k\varepsilon\tanh(\varepsilon/2).$$ {#eq-comp-advanced-tanh}
Both forms are valid. The first is the original theorem, and the second gives a smaller total.

The gain is a total that grows like $\sqrt k$ instead of $k$. The price is the slack $\delta'$ and the factor $\sqrt{2\ln(1/\delta')}$, which is the cost of cutting far into the tail. At $\delta' = 10^{-5}$ that factor is 4.80, so advanced composition helps only once $k$ is large enough for $\sqrt k$ to outgrow it. With the original drift, at this $\delta'$ and $\varepsilon = 0.1$, advanced composition beats basic composition only from $k = 29$ on. Below that basic composition is smaller, so always take the smaller of the two.

The theorem also gives a rule of thumb for budgets. Suppose you want the whole run to be $(\varepsilon_{\text{total}}, \delta')$-DP after $k$ steps. When $\varepsilon\sqrt k$ is small, the drift is small next to the fluctuation term, so you can drop it. Setting the fluctuation term equal to the target and solving for the per-step level, you can give each step roughly
$$\varepsilon = \frac{\varepsilon_{\text{total}}}{\sqrt{2k\ln(1/\delta')}}.$$
Basic composition would allow each step only $\varepsilon_{\text{total}}/k$, which is much smaller when $k$ is large.

To see how the bounds behave as $k$ grows, @tbl-comp-vs-k lists them and @fig-comp-epsilon-vs-k plots them. For the hundred queries the fluctuation term alone is 4.80.

| $k$ | Basic $k\varepsilon$ | Advanced, @eq-comp-advanced | Advanced, @eq-comp-advanced-tanh | Optimal |
| --- | --- | --- | --- | --- |
| 10 | 1.000 | 1.623 | 1.567 | 0.994 |
| 100 | 10.000 | 5.850 | 5.298 | 4.307 |
| 1,000 | 100.000 | 25.691 | 20.170 | 17.787 |
| 10,000 | 1,000.000 | 153.156 | 97.944 | 91.691 |

: Total $\varepsilon'$ at $\delta' = 10^{-5}$ for $k$ mechanisms that are each $0.1$-DP. {#tbl-comp-vs-k}

At $k = 10$ both advanced bounds are worse than basic composition, because 10 is below the crossover. At $k = 10{,}000$ the drift has taken over. There the original drift alone costs 105 of the total of 153. The $\tanh$ form costs only 50, which is why it is worth using when $k$ is large.

![Total $\varepsilon'$ at $\delta = 10^{-5}$ after $k$ compositions of $\varepsilon$-DP mechanisms, for $\varepsilon = 0.1$ (left) and $\varepsilon = 0.5$ (right), on log axes. Basic composition grows like $k$; advanced and optimal composition grow like $\sqrt k$ until the drift term takes over. The optimal curve is never above any other.](figures/comp_epsilon_vs_k.svg){#fig-comp-epsilon-vs-k}

The bound is conservative for three reasons. Azuma's inequality uses only the width of each step's range, so it must allow for the largest spread that range permits. The original drift is about twice the true average loss for small $\varepsilon$. And the tail bound in @eq-comp-delta counts every output above $\varepsilon'$ in full. So for the hundred queries at the original advanced bound of 5.85, the exact $\delta$ of the worst case is $1.96\times10^{-9}$, far below the promised $\delta'$. So use advanced composition for quick estimates and budgets, and compute the number you report with the optimal bound.

## Optimal composition {#sec-comp-optimal}

The best bound is the smallest $\delta(\varepsilon')$ that holds for every composition of $k$ steps with a given $(\varepsilon, \delta)$. Kairouz, Oh & Viswanath (2015) found it exactly. Their argument rests on **post-processing**, from [Post-processing, group privacy and composition](02-properties.md). Passing an output through a further random step that never looks at the data can only make the two output distributions harder to tell apart. So post-processing never increases $\delta(\varepsilon')$, a fact known as the data processing inequality.

In this argument a mechanism stands for its two output distributions, on one fixed pair of neighbours $D$ and $D'$. Kairouz, Oh & Viswanath showed that every $(\varepsilon, \delta)$-DP pair of output distributions can be obtained by post-processing one particular pair. This worst pair spends both parameters to the full. Its output reveals which of the two datasets was used with probability $\delta$, and otherwise it is randomized response at level $\varepsilon$. @tbl-comp-four-point lists its four outputs.

| Output | "flag $D$" | 0 | 1 | "flag $D'$" |
| --- | --- | --- | --- | --- |
| Probability on $D$ | $\delta$ | $(1 - \delta)t$ | $(1 - \delta)(1 - t)$ | 0 |
| Probability on $D'$ | 0 | $(1 - \delta)(1 - t)$ | $(1 - \delta)t$ | $\delta$ |
| Privacy loss | $+\infty$ | $+\varepsilon$ | $-\varepsilon$ | $-\infty$ |

: The $(\varepsilon, \delta)$ randomized response of Kairouz, Oh & Viswanath, with $t = e^\varepsilon/(1 + e^\varepsilon)$. With probability $\delta$ it announces the dataset outright; otherwise it is randomized response at level $\varepsilon$. {#tbl-comp-four-point}

For $\delta = 0$ the post-processing step can be written down. Call it a **channel**, a random rule that takes the report of randomized response and turns it into an output of the target mechanism. For example, take a $\ln 3$-DP mechanism with three outputs. On $D$ it gives them with probabilities $(0.5, 0.3, 0.2)$, and on $D'$ it gives them in the reverse order. A channel turns randomized response at $\ln 3$ into this mechanism. When the report is 0, it draws the three outputs with probabilities $(0.65, 0.30, 0.05)$, and when the report is 1 it uses the reverse order. The same construction works for any $\varepsilon$-DP mechanism, and the condition is what makes it work. The channel's two lists of probabilities have no negative entries exactly because no output is more than $e^\varepsilon$ times likelier on one dataset than on the other (@fig-comp-tradeoff-region, right). The full edition gives the general formula.

For $\delta > 0$ there is no one-line formula, but there is a clear reason the channel must exist. Recall the **trade-off curve** from [Hypothesis testing](../part01-prerequisites/04-hypothesis-testing.md), which records, for each false positive rate, the smallest false negative rate any test can reach. A mechanism is $(\varepsilon, \delta)$-DP exactly when its trade-off curve lies above a fixed boundary made of two straight lines. The lines come from applying the definition to the set of outputs on which a test says $D$, and to its complement. The four-point mechanism sits exactly on that boundary (@fig-comp-tradeoff-region, left). So every $(\varepsilon, \delta)$-DP mechanism is at least as hard to test as the four-point one, at every false positive rate. By Blackwell's theorem (1953), that is exactly the condition for being a post-processing of it. Every $(\varepsilon, \delta)$-DP mechanism is therefore a post-processing of the four-point one, and Kairouz, Oh & Viswanath construct the channel explicitly.

![Left: the boundary trade-off curves at $\varepsilon = \ln 3$ with $\delta = 0$ and $\delta = 0.1$, attained by randomized response, and the trade-off curve of a Laplace mechanism with the same $\varepsilon$, which lies inside the allowed region. Right: a $\ln 3$-DP mechanism on three outputs, with probabilities $(0.5, 0.3, 0.2)$ on $D$ and $(0.2, 0.3, 0.5)$ on $D'$, and the channel's two lists of probabilities that produce it from randomized response.](figures/comp_tradeoff_region.svg){#fig-comp-tradeoff-region}

Since every step is a post-processing of the four-point mechanism, every composition of $k$ steps is a post-processing of $k$ copies of it. This holds even when each step is chosen after seeing the earlier outputs. You can draw all $k$ hidden reports first and then apply each step's channel in turn, with the channel chosen from the outputs so far. So no composition of $(\varepsilon, \delta)$-DP steps can be worse than $k$ copies of the four-point mechanism, and the loss of those $k$ copies is easy to work out.

The $\delta$ of the $k$ copies has two parts. In the first case, some step announces the dataset, which happens with probability $1 - (1 - \delta)^k$. That output is impossible under $D'$, so its loss is infinite and it counts in full. In the other case no step announces anything, and the run is $k$ rounds of randomized response. Its loss is then $(k - 2J)\varepsilon$, where $J$ is the number of lies, as in @sec-comp-loss. Putting that loss into @eq-comp-delta gives the $\delta$ of plain randomized response, written $\delta_{\mathrm{RR}}(\varepsilon')$.

The two cases then add up. An announcement contributes its whole probability, while the other case, which has probability $(1 - \delta)^k$, contributes that probability times $\delta_{\mathrm{RR}}(\varepsilon')$. So any composition of $k$ steps that are each $(\varepsilon, \delta)$-DP, adaptive or not, satisfies
$$\delta(\varepsilon') \le 1 - (1 - \delta)^k\big(1 - \delta_{\mathrm{RR}}(\varepsilon')\big), \qquad \delta_{\mathrm{RR}}(\varepsilon') = \sum_{j=0}^k\binom kj t^{k-j}(1 - t)^j\big(1 - e^{\varepsilon' - (k - 2j)\varepsilon}\big)_+.$$ {#eq-comp-kov}
The sum runs over the possible values $j$ of the number of lies. Each binomial weight is the probability that exactly $j$ of the $k$ reports are lies, and the bracket is that output's cost from @eq-comp-delta. The $k$ copies of the four-point mechanism are themselves a valid composition, and for them the bound holds with equality. So no better bound can hold for every mechanism in the class.

The same argument sharpens the average loss. The KL divergence also obeys the data processing inequality, so post-processing cannot increase it. Every $\varepsilon$-DP step is a post-processing of randomized response, so its average loss is at most that of randomized response:
$$\mathbb E[L] \le \varepsilon\tanh(\varepsilon/2),$$ {#eq-comp-tanh-mean}
with equality for randomized response itself. This is the bound that gives @eq-comp-advanced-tanh its smaller drift.

For the hundred queries the optimal $\varepsilon'$ is 4.3068, which is 19% below the sharp advanced bound and less than half of basic composition's total. Now suppose each step also has its own $\delta = 5\times10^{-8}$, while the total budget stays at $10^{-5}$. Then the chance of an announcement, $1 - (1 - \delta)^k$, uses up about $5.0\times10^{-6}$ of that budget. The optimal $\varepsilon'$ counts the announcements inside the full budget, and it rises to 4.4469. Advanced composition in its original form, by contrast, must fit in the remaining $5\times10^{-6}$, and it gives 5.9926.

The bound is optimal for the class of mechanisms you know only by their $(\varepsilon, \delta)$. A real mechanism usually does better, because its loss is not always at $+\varepsilon$ or $-\varepsilon$. Take a count with Laplace noise of scale $1/\varepsilon$. When the noisy answer lands between the two possible true counts, its loss lies strictly between $-\varepsilon$ and $\varepsilon$. Those outputs carry less evidence, so at $\varepsilon = 1$ the Laplace mechanism's average loss is 0.3679. That is below the 0.4621 of randomized response, the worst case.

The bigger waste is summarizing a mechanism by one $(\varepsilon, \delta)$ pair at all. A Gaussian mechanism has a whole curve of pairs, a smallest $\delta$ for each $\varepsilon$, and one point on it forgets how the rest of the curve runs. With independent noise the exact composition is easy, because the privacy loss of one Gaussian step is itself normal, and so is a sum of independent normal losses.

Take $k$ Gaussian steps on a query of sensitivity 1, and write $\sigma$ for the standard deviation of the noise each adds. They compose exactly into one Gaussian step whose **shift**, the distance between its two output means in units of the noise's standard deviation, is $\sqrt k/\sigma$. For example, a thousand steps with $\sigma = 20$ are exactly $(7.51, 10^{-5})$-DP. If instead you convert each step to one $(\varepsilon, \delta)$ point, splitting the overall $\delta$ between the steps as well as possible, optimal composition gives 62.7, about 8 times worse (@fig-comp-gaussian). When you know the mechanisms, compose their whole curves with the accountants of [Privacy accounting](../part04-privacy-accounting/00-introduction.md).

![Total $\varepsilon$ at $\delta = 10^{-5}$ after $k$ Gaussian mechanisms with $\sigma = 20$ on a query of sensitivity 1. The first three curves convert each step to one $(\varepsilon, \delta)$ point, with the best split of $\delta$, then compose with basic, advanced or optimal composition. The dashed curve composes the exact Gaussian curves, a single Gaussian with shift $\sqrt k/\sigma$.](figures/comp_gaussian.svg){#fig-comp-gaussian}

Computing the bound takes some care, because Kairouz, Oh & Viswanath state it in a closed form that divides two very large numbers. For a thousand steps at $\varepsilon = 1$ the denominator is about $10^{570}$, so the formula as printed overflows a double-precision number. The fix is to add the terms of @eq-comp-kov using their logarithms, subtracting the largest one before exponentiating. Because every term is non-negative, nothing cancels and the result stays accurate. Evaluated this way at the same $\varepsilon$ and $k$, the bound gives $\delta(550) = 6.094\times10^{-4}$, for instance. To find the $\varepsilon'$ for a target $\delta$, search for it by bisection (@sec-comp-code).

Steps with different privacy levels, written $\varepsilon_i$ for step $i$, are harder. Composed randomized response at those levels is still the worst case, but its loss now depends on which steps lied, not just on how many. Murtagh & Vadhan (2016) showed that computing the optimal bound exactly is then #P-complete. That is the class of hard counting problems, so no efficient exact algorithm is expected. You can still enumerate a handful of steps, but for many steps use an accountant from Part 4.

## Adaptive composition {#sec-comp-adaptive}

In practice the next query often depends on the last answer. An analyst looks at a noisy histogram and then zooms into its largest bin, and a training loop computes its next gradient at parameters set by earlier noisy gradients. Each step is private, but which step runs depends on earlier outputs.

Dwork, Rothblum & Vadhan (2010) made this precise as the **adaptive composition experiment**. An adversary interacts with either $D$ or $D'$ for $k$ rounds. In each round it picks a mechanism that is $(\varepsilon, \delta)$-DP, as any function of the outputs so far, and receives that mechanism's output. Basic, advanced and optimal composition all hold for what the adversary sees. The reason is that every argument above looked at one step at a time, given the outputs so far, and once those outputs are fixed, so is the adversary's choice.

The full edition checks this with a simulated analyst who runs 100 steps at $\varepsilon = 0.1$. After any output that pointed to $D$, meaning its loss was positive, the analyst asks a Laplace query next, and otherwise a randomized-response query. @tbl-comp-adaptive compares its $\delta$ at $\varepsilon' = 2$ with the non-adaptive compositions.

| Composition of 100 steps at $\varepsilon = 0.1$ | Mean total loss | $\delta(\varepsilon' = 2)$ |
| --- | --- | --- |
| Randomized response, exact (@eq-comp-kov) | 0.4996 | 0.0201 |
| Randomized response, simulated | — | 0.0198 |
| Laplace, simulated | 0.4838 (exact 0.4837) | 0.0189 |
| Adaptive mix, simulated | 0.4971 | 0.0196 |

: The privacy loss of 100 steps of $0.1$-DP mechanisms, composed non-adaptively and by an adaptive analyst. {#tbl-comp-adaptive}

The adaptive analyst's $\delta$ lies between the two non-adaptive compositions and below the optimal bound. The theorem requires this, since it covers every way of choosing among $0.1$-DP mechanisms. In @fig-comp-delta-curves the three compositions nearly coincide, and the advanced-composition points lie far to their right.

![The privacy curve $\delta(\varepsilon')$ of 100 steps at $\varepsilon = 0.1$: the exact curve of composed randomized response (optimal composition), simulated curves for composed Laplace mechanisms and for the adaptive analyst, the advanced-composition pairs $(\varepsilon', \delta')$ for several $\delta'$, and basic composition at $(10, 0)$.](figures/comp_delta_curves.svg){#fig-comp-delta-curves}

The same reasoning shows where adaptivity breaks. The theorems require the number of steps and each step's $\varepsilon$ and $\delta$ to be fixed before the interaction starts, and only the choice of mechanism may adapt. Suppose instead that the analyst picks each step's $\varepsilon_i$ after seeing earlier outputs. Then the width of each term's range depends on the history, and since Azuma's inequality needs those widths fixed in advance, advanced composition no longer applies. So a rule such as "keep querying until the budget is spent" needs its own theorem.

Rogers, Roth, Ullman & Vadhan (2016) call a stopping rule with such a theorem a **privacy filter**. Reporting the privacy spent so far, in a way that is valid at every moment, needs a **privacy odometer**, which costs more than a filter. [Adaptive composition](../part04-privacy-accounting/08-adaptive-composition.md) covers both. Pure DP under basic composition needs no new theorem, even when the $\varepsilon_i$ adapt. There each step's loss is bounded by its $\varepsilon_i$ on every output. So the sum of the $\varepsilon_i$ bounds the total loss whatever order they came in, and no probability argument is involved.

Adaptive choices must also be made from the outputs, never from the raw data. A stopping rule that looks only at the noisy answers is post-processing, so it costs nothing. A rule that looks at the raw data leaks on its own. For example, a system that answers private counts until the true count of some group crosses a threshold reveals that count through the number of answers it gives. Every answer can then be $\varepsilon$-DP, and yet the whole run is not DP for any finite $\varepsilon$.

## Code {#sec-comp-code}

The optimal bound needs only a binomial sum, evaluated with logarithms so that nothing overflows, and a bisection to invert it. This snippet is the whole computation, with advanced composition alongside:

```python
from math import exp, expm1, lgamma, log, log1p, sqrt, tanh


def advanced_epsilon(eps: float, k: int, delta_prime: float) -> float:
    """Advanced composition with the sharp drift term k * eps * tanh(eps / 2)."""
    return eps * sqrt(2 * k * log(1 / delta_prime)) + k * eps * tanh(eps / 2)


def optimal_delta(eps_total: float, eps: float, k: int, delta: float = 0.0) -> float:
    """Kairouz-Oh-Viswanath: the exact delta(eps_total) of k (eps, delta)-DP steps, in log space."""
    log_t, log_1mt = -log1p(exp(-eps)), -log1p(exp(eps))  # log of truthful / lying probability
    logs: list[float] = []
    for j in range(k + 1):  # j of the k reports lie
        loss = (k - 2 * j) * eps
        if loss <= eps_total:
            break
        log_binom = lgamma(k + 1) - lgamma(j + 1) - lgamma(k - j + 1)
        logs.append(log_binom + (k - j) * log_t + j * log_1mt + log(-expm1(eps_total - loss)))
    top = max(logs, default=0.0)
    delta_rr = exp(top) * sum(exp(x - top) for x in logs) if logs else 0.0
    return 1 - (1 - delta) ** k * (1 - delta_rr)


def optimal_epsilon(eps: float, k: int, target_delta: float, delta: float = 0.0) -> float:
    """The smallest total epsilon whose optimal delta is at most target_delta, by bisection.

    Assumes 1 - (1 - delta) ** k < target_delta, or no finite total epsilon exists.
    """
    lo, hi = 0.0, k * eps
    for _ in range(100):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if optimal_delta(mid, eps, k, delta) > target_delta else (lo, mid)
    return hi


print(f"{advanced_epsilon(0.1, 100, 1e-5):.4f}")  # 5.2981
print(f"{optimal_epsilon(0.1, 100, 1e-5):.4f}")  # 4.3068
print(f"{optimal_epsilon(0.1, 100, 1e-5, delta=5e-8):.4f}")  # 4.4469
print(f"{optimal_epsilon(0.1, 1000, 1e-5):.3f}")  # 17.787
print(f"{optimal_delta(550, 1.0, 1000):.3e}")  # 6.094e-04
```

It reproduces the numbers of @sec-comp-advanced and @sec-comp-optimal, including the overflow case. The full edition's script [composition.py](../../code/part03-dp-foundations/composition.py) also simulates the Laplace and adaptive compositions. It checks the log-space sum against brute-force enumeration on small cases, up to 4,096 output sequences, and the two agree within $4\times10^{-16}$, the level of rounding error.

## Cheat sheet

| Result | Formula | Notes |
| --- | --- | --- |
| Privacy loss of a composition | $L = L_1 + \dots + L_k$ | Adaptive or not; no independence needed |
| $\delta$ from the loss | $\delta(\varepsilon) = \mathbb E[(1 - e^{\varepsilon - L})_+] \le \Pr[L > \varepsilon]$ | Only the right tail of $L$ matters |
| Average loss of an $\varepsilon$-DP step | $\le \varepsilon\tanh(\varepsilon/2) \le \varepsilon(e^\varepsilon - 1)$ | Equality for randomized response; about $\varepsilon^2/2$ for small $\varepsilon$ |
| Randomized response, $k$ rounds | $L = (k - 2J)\varepsilon$, $J$ binomial with $k$ trials and probability $1 - t$ | The worst case for $\varepsilon$-DP steps |
| Basic composition | $(k\varepsilon, k\delta)$ | Adaptive |
| Advanced composition | $\big(\varepsilon\sqrt{2k\ln(1/\delta')} + k\varepsilon\tanh(\varepsilon/2),\ k\delta + \delta'\big)$ | Adaptive, parameters fixed in advance; $e^\varepsilon - 1$ in place of $\tanh(\varepsilon/2)$ is the original form |
| Optimal composition | $1 - (1 - \delta)^k(1 - \delta_{\mathrm{RR}}(\varepsilon'))$, @eq-comp-kov | Adaptive; tight for mechanisms known only by $(\varepsilon, \delta)$; compute in log space |

: The composition results of this chapter, for $k$ steps that are each $(\varepsilon, \delta)$-DP. {#tbl-comp-cheat}

**Rules of thumb**

- Under composition the privacy loss adds up. Its average grows like at most about $k\varepsilon^2/2$ and its spread like $\sqrt k\,\varepsilon$.
- Take the smaller of basic and advanced composition. For a quick budget, give each step about $\varepsilon_{\text{total}}/\sqrt{2k\ln(1/\delta')}$.
- If all you know is each step's $(\varepsilon, \delta)$, compute the optimal bound exactly in log space, since it costs almost nothing.
- If you know the mechanism, compose its whole privacy curve instead, because a single $(\varepsilon, \delta)$ point per step can be off by a factor of 8.
- Queries may be chosen adaptively from earlier outputs. Privacy parameters chosen on the fly, and stopping rules that read the raw data, need more.

## Further reading

- Dwork & Roth, *The Algorithmic Foundations of Differential Privacy* (2014), §3.5: basic and advanced composition, the version most libraries cite.
- Kairouz, Oh & Viswanath, "The Composition Theorem for Differential Privacy" (ICML 2015): the hypothesis-testing view, the worst-case mechanism and the optimal bound.
- Rogers, Roth, Ullman & Vadhan, "Privacy Odometers and Filters: Pay-as-you-Go Composition" (NeurIPS 2016): why adaptively chosen privacy parameters need new theorems.
- Dong, Roth & Su, "Gaussian Differential Privacy" (JRSS Series B, 2022): composing whole privacy curves, and why the total loss looks normal.

**Derivations.** Every result in this chapter is derived step by step in [the full edition's chapter](../../book-full/part03-dp-foundations/04-composition.md), except three that it cites: the channel for $\delta > 0$ (Kairouz, Oh & Viswanath, with Blackwell's theorem), the #P-completeness of optimal composition with different $\varepsilon_i$ (Murtagh & Vadhan), and privacy filters and odometers, which [Adaptive composition](../../book-full/part04-privacy-accounting/08-adaptive-composition.md) treats. The exact composition of Gaussian steps is stated in [Laplace and Gaussian mechanisms](../../book-full/part03-dp-foundations/03-laplace-gaussian.md) and derived in its exercises.
