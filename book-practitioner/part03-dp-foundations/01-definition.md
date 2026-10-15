# The definition of differential privacy

## Why this matters

Every attack in [Threat models and attacks](../part02-attacks/00-introduction.md) can be written as a game. In it, a challenger runs an algorithm on one of two datasets that differ in one person. The attacker then looks at the output and guesses which dataset was used ([Threat models and privacy games](../part02-attacks/01-threat-models.md)). For example, membership inference plays this game against a trained model, and reconstruction plays it against a table of statistics. Each attack succeeded because the output looked different enough on the two datasets for some test to tell them apart.

A defence therefore has to make the two datasets produce outputs that look alike. Beating today's attacks is not enough, because the next attack may run a test that nobody has tried yet. **Differential privacy (DP)** asks for exactly this. For every pair of datasets that differ in one person, the distribution of the output must barely change. It is a property of the algorithm that produces the release, not of the data or of a particular attack. So you can check it on paper before any data are touched, and the guarantee holds whatever the attacker knows and however much computing power it has.

The whole chapter rests on one quantity. For any output, you can ask how much more likely it was on one dataset than on the other. The logarithm of that ratio measures how much evidence the output gives about the person, on a scale where independent pieces of evidence add up. Differential privacy is a bound on this evidence. The rest of the chapter says what "differ in one person" means, states the bound, and builds the first mechanism that meets it. It then reads the bound as a limit on attacks and on beliefs, and ends with what the guarantee does not cover.

## Neighbouring datasets {#sec-dpdef-neighbours}

The guarantee compares two datasets that differ in one person, and before anything can be checked, that phrase needs a precise meaning. A dataset is a collection of **records**. The **neighbouring relation** says which pairs of datasets count as differing in one record, and so which pairs the guarantee must protect. Choosing it fixes the **unit of privacy**, the thing whose presence or value the output must not reveal. Most misunderstandings of a claimed guarantee come from not stating this choice.

There are two standard relations, and each protects something different. Under **add/remove**, two datasets are neighbours when one is the other with one record added or taken away. The attacker's question is then "is this person in the data at all?", which is exactly the membership question of Part 2. The guarantee must hold between datasets whose sizes differ by one, so the output cannot reveal the size either, and the number of records is itself protected. Under **replace**, two datasets are neighbours when they have the same size and differ in the value of one record. Replace never compares datasets of different sizes, so the size is treated as public. The attacker is assumed to know it already, and often to know that you took part, and the guarantee hides only what your record says.

Separately, you choose what one record is. At the **example level**, a record is one training example, one row or one event. At the **user level**, a record is all the data of one person, so a user who wrote 200 messages is one record, not 200. Under example-level DP, that user is a group of 200 records, and removing the user changes all of them at once. The guarantee does extend to groups, but it becomes much weaker as the group grows ([Post-processing, group privacy and composition](02-properties.md)). So ask of every claimed guarantee what one record is, and check that it is the unit you care about, which is usually the person.

Neighbours range over every record that the data format allows, not only over plausible ones. A private algorithm must hide even a person aged 900, because the guarantee takes the worst case over all neighbouring pairs. That is why private algorithms clip or bound their inputs before they add noise, for example by capping every age at a fixed maximum. After clipping, no single record can move the output by more than a known amount. Without it, one absurd record could move the output by an unbounded amount.

## The definition {#sec-dpdef-definition}

Call a randomized algorithm that takes a dataset and produces an output a **mechanism**, and write $M$ for it. Write $D$ for a dataset and $D'$ for one of its neighbours, and run the mechanism on each. Then look at one particular output, written $y$. The natural measure of what this output reveals is how much more likely it is on one dataset than on the other.

Write $p_{M(D)}(y)$ for the probability, or the density, of the output $y$ when the mechanism runs on $D$. The logarithm of the ratio of the two probabilities is the **privacy loss** of the output,
$$L_{D,D'}(y) = \ln\frac{p_{M(D)}(y)}{p_{M(D')}(y)}.$$ {#eq-dpdef-loss}
A loss of 0 means the output is equally likely on both datasets, so it says nothing about which one was used. A large positive loss is strong evidence for $D$, and a large negative loss is strong evidence for $D'$. Independent pieces of output multiply their ratios, and the logarithm turns that product into a sum, so their evidence adds up.

The measure is a ratio rather than a difference for a reason. Suppose that adding a person to the data raises one output's probability from 0.50 to 0.51. Suppose it also raises another output's probability from 0 to 0.01, which is the same difference. The second output can occur only when the person is present, so seeing it identifies them with certainty. Only a ratio, which is infinite here, catches that. The definition below adds a small slack $\delta$ that reopens this hole for outputs of probability up to $\delta$, which is why $\delta$ must be tiny (@sec-dpdef-small-delta). For the same reason, a privacy definition should not average the loss over outputs, because a small average can hide one rare output that exposes a person completely.

The definition asks that no event, meaning no set of outputs, become much more likely when one person's record changes. It works with sets rather than single outputs because an attacker's decision, such as "accuse whenever the output lands in this range", is a set of outputs. It measures "much more likely" by a multiplicative factor $e^\varepsilon$, where the privacy parameter $\varepsilon$ is a non-negative number, so that a small $\varepsilon$ means a ratio close to 1. On top of that, it allows a small additive slack $\delta$ between 0 and 1, so that events of tiny probability are not held to the ratio.

Formally, fix a neighbouring relation. A mechanism $M$ is **$(\varepsilon, \delta)$-differentially private** if every pair of neighbours $D$ and $D'$ under that relation, and every set $S$ of outputs, satisfy
$$\Pr[M(D) \in S] \le e^\varepsilon\,\Pr[M(D') \in S] + \delta.$$ {#eq-dpdef-def}
The neighbouring relation is symmetric, so the inequality also holds with $D$ and $D'$ swapped. The probabilities come only from the mechanism's own randomness, with the dataset held fixed. Nothing is assumed about how the data were generated.

Each condition of the definition is there because an attack would slip through without it. It must hold for every pair of neighbours, because the attacker may target the most unusual person in the data. It must hold for every set of outputs, because, as @sec-dpdef-testing shows, any attack can be described by the set of outputs on which it accuses someone. And it assumes nothing about what the attacker knows or can compute. This worst case in every direction is what makes the guarantee hold against attacks that have not been invented yet. It is also what makes it expensive, because hiding the worst case takes more noise, and so costs more accuracy, than hiding a typical one.

When $\delta = 0$, the definition is called **pure DP**, written $\varepsilon$-DP. With no slack, the probability of every event may grow by at most the factor $e^\varepsilon$. For small $\varepsilon$, that factor is about $1 + \varepsilon$, so no event's probability can rise by more than about $\varepsilon$ times its own value. If you take $\varepsilon = 0.1$, for example, an event of probability 0.5 on one dataset then has probability at most $0.5e^{0.1} = 0.553$ on the other. At $\varepsilon = 1$, the same bound is $0.5e = 1.36$. That is above 1, so the definition says nothing about such a likely event. Values used in practice range from below 1 to around 10, and @sec-dpdef-advantage shows how little the large ones promise.

Pure DP can be checked one output at a time. The probability of a set is the sum of its outputs' probabilities. So the ratio for a set is an average of its outputs' ratios, each weighted by that output's probability on $D'$. An average is never larger than its largest term, and so the worst set is a single output. So for discrete outputs, which can be listed one by one, pure DP requires the privacy loss of every single output to lie between $-\varepsilon$ and $\varepsilon$. For continuous outputs, the same holds for the density at almost every point.

Pure DP is too strict for some useful mechanisms. @fig-dpdef-ratio-band draws the definition for a count released with noise. Pure DP asks the output density on $D$ to stay inside the shaded band, which runs from $e^{-\varepsilon}$ to $e^\varepsilon$ times the density on $D'$. Laplace noise stays inside everywhere, because the log ratio of two Laplace densities is bounded. Gaussian noise leaves the band in the tail. The reason is that the log ratio of two normal densities with the same spread is a straight line in the output, so it grows without bound.

Those far-out outputs are very rare, and $\delta$ is the allowance for them. When $\delta > 0$, the definition is called **approximate DP**. The ratio bound may then fail on some outputs, as long as the probability those outputs carry beyond $e^\varepsilon$ times their probability on $D'$ adds up to at most $\delta$.

![The definition as a picture, for a count released with noise at $\varepsilon = 1$, with log vertical axes. The shaded band runs from $e^{-\varepsilon}$ to $e^\varepsilon$ times the output density on $D'$ (orange, true count 1). Left: with Laplace noise of scale $1/\varepsilon$, the density on $D$ (blue, true count 0) stays inside the band, so the release is $\varepsilon$-DP. Right: with Gaussian noise of standard deviation 2, it leaves the band in the left tail, and the red mass is what $\delta$ has to cover.](figures/dpdef_ratio_band.svg){#fig-dpdef-ratio-band}

Approximate DP, unlike pure DP, cannot be checked one output at a time. A per-output check would allow each output its own slack $\delta$, instead of one slack for a whole set. The slack would then add up across the outputs of a set. A counterexample shows how bad this gets. Let the output on $D$ be spread evenly over 1,000 values, and the output on $D'$ over 1,000 different values. Each output then has probability 0.001 on one dataset and 0 on the other. So it passes the per-output check with a slack of 0.0010, whatever $\varepsilon$ is. Yet one look at the output tells you which dataset was used, and the definition needs a slack of 1.0000 for the set of all outputs that $D$ can produce.

The choice of neighbouring relation also changes the guarantee by a known amount. Replacing Alice's record by Dan's is the same as removing Alice and then adding Dan, which is two add/remove steps. Under pure DP, each step can multiply the probability of any event by at most $e^\varepsilon$, so two steps can multiply it by at most $e^{2\varepsilon}$. That gives the conversion
$$\varepsilon\text{-DP under add/remove} \implies 2\varepsilon\text{-DP under replace}.$$ {#eq-dpdef-replace}
For approximate DP, each step also adds its own slack, and the second step multiplies the first step's slack by $e^\varepsilon$. So the same path gives $2\varepsilon$ with slack $(1 + e^\varepsilon)\delta$.

The reverse direction fails, because a guarantee under replace never compares datasets of different sizes. A mechanism that publishes the number of records, for example, is perfectly private under replace. Under add/remove it is not private for any $\varepsilon$, because the size it publishes always differs between the two datasets. One output then has probability 1 on one dataset and 0 on the other.

Whether the factor of 2 is actually paid depends on the mechanism. Take a survey of smokers and non-smokers, and release the two counts, each with its own Laplace noise of scale $1/\varepsilon$. Recall from [Divergences](../part01-prerequisites/03-divergences.md) that when two Laplace densities with the same scale have centres one unit apart, their log ratio is never more than one divided by the scale. With scale $1/\varepsilon$, a count that moves by one therefore has a largest privacy loss of $\varepsilon$. The two counts get independent noise, so their losses add. Removing Carol changes only the count of smokers, and at $\varepsilon = 1$ the largest loss is 1.0000. Changing Alice's answer from smoker to non-smoker moves one count down and the other up, so the largest loss doubles to 2.0000. If you release only the number of smokers, a replacement changes one count, and the cost under replace stays at $\varepsilon$.

## Randomized response {#sec-dpdef-rr}

The oldest private mechanism comes from survey design. Suppose you want the fraction of people who have done something embarrassing, such as cheating on an exam, but people asked directly tend to lie. So Warner (1965) proposed that each person toss a coin in private. On heads, they answer truthfully, and on tails, they toss again and answer "yes" on heads and "no" on tails. Any single "yes" might be the coin's doing, so it proves nothing about the person. But the coins behave predictably over many people, so the fraction can still be recovered.

The scheme tells the truth with probability $\frac34$. A person answers truthfully when the first coin lands heads, which has probability $\frac12$. They also answer truthfully when it lands tails and the second coin happens to match the truth, which has probability $\frac14$. Here the two datasets being compared differ only in this person's true answer. So a "yes" is 3 times as likely from someone whose true answer is yes as from someone whose answer is no, and the same holds for a "no". No report has a larger ratio, so the scheme is pure DP with $\varepsilon = \ln 3 = 1.0986$.

In general, **randomized response** at level $\varepsilon$ tells the truth with some probability $t$ a little above one half, and lies otherwise. A report is then $t/(1 - t)$ times as likely under the matching true answer as under the other. Setting that ratio to $e^\varepsilon$ and solving for $t$ gives
$$t = \frac{e^\varepsilon}{1 + e^\varepsilon},$$
With this choice, every report has a privacy loss of exactly $\varepsilon$ or $-\varepsilon$. For Warner's coins, $e^\varepsilon = 3$ gives $t = \frac34$ again.

When a group of $n$ people each answer this way, the guarantee is still $\varepsilon$ per person, provided that everyone tosses their own coins independently. Independence makes the probability of the whole list of reports a product of one factor per person. Under replace, two neighbouring datasets differ only in one person's true answer, so every other factor is the same on both and cancels from the ratio. What remains is that one person's loss. The guarantee holds under replace only. Under add/remove, the number of reports reveals how many people took part, and with it whether you did. Randomized response protects the value of your answer, not the fact that you answered.

The lying pulls the share of "yes" reports towards one half by a known amount, and undoing that pull recovers the true fraction. Write $\pi$ for the true fraction of people whose answer is yes, and $\bar y$ for the share of "yes" reports. A report is "yes" with probability $t$ when the true answer is yes and $1 - t$ when it is no. So on average, $\bar y$ equals $\pi t + (1 - \pi)(1 - t)$, which is $(1 - t) + \pi(2t - 1)$. Solving for $\pi$ gives an estimate that is right on average.

In symbols, the estimate and its standard deviation are
$$\hat\pi = \frac{\bar y - (1 - t)}{2t - 1}, \qquad \operatorname{sd}(\hat\pi) = \frac{e^{\varepsilon/2}}{(e^\varepsilon - 1)\sqrt n} \approx \frac{1}{\varepsilon\sqrt n}\ \text{ for small } \varepsilon.$$ {#eq-dpdef-rr-estimate}
The cost of privacy is in the denominator. When $\varepsilon$ is small, $t$ is close to one half, the reports are nearly pure coin tosses, and $2t - 1$ is small. Dividing by it blows up the noise in $\bar y$. So halving $\varepsilon$ roughly doubles the error, and since the error falls like $1/\sqrt n$, you need four times as many people to win the accuracy back.

Take a survey of 1,000 people with a true fraction of 0.3. With Warner's coins, the estimate's standard deviation is 0.0274. An ordinary survey of the same size, drawn from a large population, already has a sampling error of 0.0145. So randomized response at this level adds noise about twice as large as the sampling noise already there. At $\varepsilon = 0.5$, the standard deviation grows to 0.0626 (@fig-dpdef-rr). At $\varepsilon = 0.1$ it reaches 0.316, which is larger than the fraction being measured, so the survey is useless.

![Left: randomized response at $\varepsilon = \ln 3$ as output probabilities; each report is 3 times more likely under one true answer than under the other. Right: estimates of the fraction of "yes" from 4,000 simulated surveys of 1,000 people with true fraction 0.3, at $\varepsilon = \ln 3$ and $\varepsilon = 0.5$, with the normal curves that the standard-deviation formula predicts.](figures/dpdef_rr.svg){#fig-dpdef-rr}

Each person randomizes before their answer leaves them, so nobody, not even the analyst, ever sees a true answer. This is the **local** model, in which no one is trusted with the raw data, and the large error is its price. [Local, shuffle and distributed trust models](08-trust-models.md) develops it and compares it with models that trust someone.

## What δ measures {#sec-dpdef-divergences}

The definition quantifies over every set of outputs. That is fine for a mechanism with two outputs, but a mechanism with continuous outputs has infinitely many sets, and nobody can check them one by one. The way out is to find the worst set directly, which turns the definition into a single formula. The formula also shows what $\delta$ actually counts.

For pure DP, @sec-dpdef-definition showed that the worst set is a single output, so pure DP bounds the largest privacy loss over all outputs. The logarithm of the largest ratio between two distributions is called their **max-divergence** ([Divergences](../part01-prerequisites/03-divergences.md)). In these terms, $\varepsilon$-DP says that the max-divergence between the outputs on any two neighbours is at most $\varepsilon$.

For approximate DP, the definition bounds a difference instead. For every set, the probability on $D$ may exceed $e^\varepsilon$ times the probability on $D'$ by at most $\delta$. Each output $y$ adds its own term, $p_{M(D)}(y) - e^\varepsilon p_{M(D')}(y)$, to this difference, and the term is positive exactly when the output's privacy loss exceeds $\varepsilon$. So the worst set collects every output whose loss exceeds $\varepsilon$, and leaves out all the others. The resulting largest difference is called the **hockey-stick divergence**, and a mechanism is $(\varepsilon, \delta)$-DP exactly when it is at most $\delta$ for every ordered pair of neighbours.

The hockey-stick divergence has a useful reading in terms of the privacy loss. Take an output, and write $L$ for its privacy loss. Its probability on $D'$ is then $e^{-L}$ times its probability on $D$. Its term in the sum is its probability on $D$ minus $e^\varepsilon$ times its probability on $D'$. Substituting the second probability, the term is a fraction $1 - e^{\varepsilon - L}$ of the output's probability on $D$. So the sum is an average over the outputs of the mechanism on $D$, each weighted by this fraction, or by 0 when the fraction is negative. Taking the worst pair of neighbours gives the smallest valid $\delta$:
$$\delta(\varepsilon) = \max_{D, D'}\ \mathbb E_{y \sim M(D)}\Big[\max\big\{0,\ 1 - e^{\varepsilon - L_{D,D'}(y)}\big\}\Big].$$ {#eq-dpdef-hockey}
The weight is 0 when the loss is at most $\varepsilon$. It is close to 0 when the loss is just above $\varepsilon$, and it approaches 1 as the loss grows.

So $\delta$ is not "the probability that privacy fails". That probability, the chance that the loss exceeds $\varepsilon$, counts every output past the line in full, while $\delta$ counts an output just past the line for almost nothing. The tail probability is still a valid choice of $\delta$, because every weight is at most 1 and is 0 unless the loss exceeds $\varepsilon$. So if the loss exceeds $\varepsilon$ with probability at most $\delta$, the mechanism is $(\varepsilon, \delta)$-DP. The converse fails, and the gap can be large. Take a count released with Gaussian noise of standard deviation 2, at $\varepsilon = 1$.

Both numbers come from the shape of the privacy loss. On $D$ the output is normal around the true count, and on $D'$ it is the same normal shifted by one, a shift of $\mu = 1/2$ standard deviations. [Divergences](../part01-prerequisites/03-divergences.md) showed that for such a pair the privacy loss is a straight line in the output, so the loss is itself normal. When the output comes from $D$, the loss has mean $\mu^2/2 = 0.125$ and standard deviation $\mu = 0.5$, so it exceeds $\varepsilon = 1$ when a standard normal variable exceeds $(1 - 0.125)/0.5 = 1.75$. When the output comes from $D'$, the mean is $-0.125$, and the same event needs $(1 + 0.125)/0.5 = 2.25$. Write $\Phi$ for the cumulative distribution function of the standard normal, so that $\Phi(-z)$ is the chance that it exceeds $z$. The tail probability is then the first of these chances, and the true $\delta$ is the hockey-stick divergence of the pair, which subtracts $e^\varepsilon$ times the second (@eq-div-gauss-hockey with $\gamma = e^\varepsilon$):
$$\Pr[L > \varepsilon] = \Phi\Big(\frac\mu2 - \frac\varepsilon\mu\Big) = \Phi(-1.75) = 4.006\times10^{-2},$$
$$\delta(\varepsilon) = \Phi\Big(\frac\mu2 - \frac\varepsilon\mu\Big) - e^\varepsilon\,\Phi\Big(-\frac\mu2 - \frac\varepsilon\mu\Big) = 4.006\times10^{-2} - e \times 1.222\times10^{-2} = 6.830\times10^{-3}.$$
The subtracted term removes most of the tail, so the true $\delta$ is less than a sixth of the tail probability. @tbl-dpdef-gauss repeats the comparison at other values of $\varepsilon$.

| $\varepsilon$ | 0.5 | 1 | 2 | 3 |
| --- | --- | --- | --- | --- |
| Exact $\delta(\varepsilon)$ | $5.244\times10^{-2}$ | $6.830\times10^{-3}$ | $9.439\times10^{-6}$ | $3.401\times10^{-10}$ |
| Tail, chance that the loss exceeds $\varepsilon$ | $2.266\times10^{-1}$ | $4.006\times10^{-2}$ | $8.842\times10^{-5}$ | $4.462\times10^{-9}$ |
| Tail / exact | 4.32 | 5.87 | 9.37 | 13.12 |

: The exact $\delta(\varepsilon)$ of a count with Gaussian noise of standard deviation 2, against the tail probability of the privacy loss. {#tbl-dpdef-gauss}

The gap grows with $\varepsilon$, from a factor of 4 to a factor of 13 in the table. The reason is the discount in @eq-dpdef-hockey. Most outputs in the tail have a loss only a little above $\varepsilon$, so they count fully toward the tail but almost nothing toward $\delta$. Suppose you calibrate the noise with the tail probability, choosing its scale so that the tail meets your target $\delta$. Then you add noise you did not need, which is why [The Laplace and Gaussian mechanisms](03-laplace-gaussian.md) calibrates the Gaussian with the exact curve.

The table also shows that a mechanism has a whole curve of guarantees, not one. Every value of $\varepsilon$ has its own smallest $\delta$, and the mechanism satisfies all of these pairs at once. As you ask for a larger $\varepsilon$, fewer outputs exceed it, so the required $\delta$ shrinks. That function $\delta(\varepsilon)$ is the **privacy curve**, and any single pair you quote is one point on it. So report the mechanism's parameters along with the pair, which lets others recompute the whole curve. The privacy accountants of [Numerical accounting](../part04-privacy-accounting/06-numerical-accounting.md), tools that track the total cost of many releases, combine whole curves, not points.

Whether the curve reaches zero depends on whether the privacy loss is bounded (@fig-dpdef-delta-curves). Randomized response and the Laplace count have a largest loss, and at that value of $\varepsilon$ every weight is 0, so the curve reaches $\delta = 0$. The Laplace count with scale 1, for example, is pure DP at $\varepsilon = 1$. The Gaussian count has an unbounded loss, so some outputs always exceed any $\varepsilon$. It therefore has no pure guarantee, only approximate ones. For tracking many releases, it helps to treat the privacy loss as a random number, drawn by running the mechanism on $D$ and computing the loss of its output. Its distribution, the **privacy loss distribution (PLD)**, determines the whole curve through @eq-dpdef-hockey, and Part 4's accountants work with it directly.

![The smallest $\delta$ as a function of $\varepsilon$ for three mechanisms, with a log vertical axis: randomized response at $\varepsilon_0 = \ln 3$, a count with Laplace noise of scale 1, and a count with Gaussian noise of standard deviation 2. The first two reach $\delta = 0$ at their largest privacy loss, and the Gaussian curve never does. The dashed line is the tail probability of the Gaussian count's privacy loss, a valid but loose $\delta$.](figures/dpdef_delta_curves.svg){#fig-dpdef-delta-curves}

## What ε means for an attacker {#sec-dpdef-testing}

A membership attacker is a test. It sees the output and decides between two hypotheses, that the data were $D$ or that they were $D'$. To match Part 2, take $D'$ to be the dataset with the target's record and $D$ the one without. Saying "$D'$" then means accusing the target of being a member. These two datasets are add/remove neighbours, so this reading needs a guarantee under add/remove. Under replace, the same bounds apply to tests of what the target's record says. Whatever the attacker computes, it ends by saying "member" on some set of outputs and "not a member" on the rest. That set is exactly the kind of set the definition bounds, so the definition limits every attack.

As in Part 2, describe an attack by its error rates ([Hypothesis testing](../part01-prerequisites/04-hypothesis-testing.md)). Its **false positive rate (FPR)** is how often it says "member" when the target is out. Its **false negative rate (FNR)** is how often it says "not a member" when the target is in. Its **true positive rate (TPR)** is how often it says "member" when the target is in, which is one minus the FNR.

Applying the definition to the attacker's "member" set, and to the opposite set of outputs, gives two inequalities that every attack must satisfy:
$$\mathrm{FPR} + e^\varepsilon\,\mathrm{FNR} \ge 1 - \delta \qquad\text{and}\qquad e^\varepsilon\,\mathrm{FPR} + \mathrm{FNR} \ge 1 - \delta.$$ {#eq-dpdef-testing}
Each inequality is the definition applied to one of the attacker's two sets. Take first the set of outputs on which the attacker says "not a member". It has probability $1 - \mathrm{FPR}$ on $D$, the dataset without the target, and probability FNR on $D'$. So the definition says that $1 - \mathrm{FPR}$ is at most $e^\varepsilon$ times the FNR plus $\delta$, which rearranges to the first inequality. For the second, swap the two datasets and take the set on which the attacker says "member". Its probability is the TPR on $D'$ and the FPR on $D$, so the definition says that the TPR is at most $e^\varepsilon$ times the FPR plus $\delta$. Since the TPR is one minus the FNR, that is the second inequality.

An attacker that randomizes its decision is covered too, because, as @sec-dpdef-divergences showed for sets, nothing does better than the worst set. The reverse also holds. If every test satisfies both inequalities, the mechanism is $(\varepsilon, \delta)$-DP. So this form says exactly as much as the definition (Kairouz, Oh & Viswanath, 2015).

The smallest FNR any test can reach at a given FPR is the **trade-off function** of the two output distributions. Each inequality above is a line in the plane of the two error rates, and no attack can lie below either line. So the trade-off function of every pair of neighbours lies above the higher of the two lines, and never below 0:
$$f_{\varepsilon,\delta}(\mathrm{FPR}) = \max\big\{0,\ 1 - \delta - e^\varepsilon\,\mathrm{FPR},\ e^{-\varepsilon}(1 - \delta - \mathrm{FPR})\big\}.$$ {#eq-dpdef-tradeoff}
The two lines are mirror images of each other across the diagonal where the two error rates are equal, so they cross on it (@fig-dpdef-tradeoff-region).

Written in terms of the TPR, the second inequality is the form to remember. At every false positive rate, every attack has
$$\mathrm{TPR} \le e^\varepsilon\,\mathrm{FPR} + \delta.$$
This bites where it matters. Part 2 judged membership attacks by their TPR at a low FPR, because an attack that flags members while almost never flagging non-members is one that confidently identifies people. For example, take a mechanism with $\varepsilon = 1$ and $\delta = 10^{-5}$. At an FPR of $10^{-3}$, no attack against it reaches a TPR above 0.002728. Dividing the bound by the FPR shows that an attack can flag members at most $e^\varepsilon + \delta/\mathrm{FPR}$ times as often as it flags non-members. As the FPR falls toward 0, the second term takes over, so at a very low FPR the guarantee rests almost entirely on $\delta$.

![Left: in the plane of false positive rate and false negative rate, the lower boundary $f_{\varepsilon,\delta}$ for $\varepsilon = \ln 3$ and $\delta = 0$ (shaded below: no test can be there), and for $\varepsilon = 1$ with $\delta = 0$ and $\delta = 0.05$. The randomized response test at $\varepsilon = \ln 3$ sits exactly at the corner. Right: the same bound as the largest TPR against FPR on log axes, for $\varepsilon = 1$ with and without $\delta$, and the exact best TPR against a count with Gaussian noise of standard deviation 2, which is $(1, 0.00683)$-DP.](figures/dpdef_tradeoff_region.svg){#fig-dpdef-tradeoff-region}

The bound is tight for randomized response. Take Warner's coins, with two datasets that differ only in whether one person's true answer is yes or no, so that the test guesses the answer rather than membership. The best test simply believes the person's report, so each of its error rates is the chance of a lie, 0.25. That puts it exactly at the crossing point of the two lines. At that FPR, the bound allows a TPR of at most $3 \times 0.25 = 0.75$, and the test reaches exactly that.

Most mechanisms sit strictly inside the allowed region. The Gaussian count with standard deviation 2, for example, is $(1, 0.00683)$-DP. At an FPR of $10^{-3}$, the bound therefore allows a TPR of up to 0.0095. But the best test against it reaches only 0.0048. The $\delta$ term alone is already larger than that. The Gaussian count has a smooth trade-off curve that touches the two lines at one point each. Keeping that whole curve instead of the two lines is the idea behind f-DP, a definition that bounds the whole trade-off curve, in [Gaussian differential privacy](../part04-privacy-accounting/04-gaussian-dp.md).

Read backwards, the same inequality turns an attack into an audit. Measure an attack's FPR and TPR against a trained model, and solve the TPR bound for $\varepsilon$, which gives the smallest $\varepsilon$ consistent with what you measured. Suppose someone claims a model is $(2, 10^{-5})$-DP, and you run an attack that reaches a TPR of 0.05 at an FPR of 0.001. The bound allows a TPR of only about 0.0074 at that FPR, so either the claim is false or the rates were measured wrongly. The snippet in @sec-dpdef-code shows that, if the measured rates are exact, $\varepsilon$ must be at least 3.91.

Audits use a **canary**, a record planted in the training data to be as easy to detect as possible, such as a mislabelled example. The easiest record to detect is the right one to plant, because the definition must hold for the worst record. An ordinary record gives a much weaker lower bound ([Memorization and extraction](../part02-attacks/05-memorization-extraction.md)).

## Bound on the membership advantage {#sec-dpdef-advantage}

Part 2 also scored a membership attacker by a single number, its **advantage**. In the membership game, the challenger flips a fair coin to pick $D$ or $D'$, runs the mechanism, and the attacker guesses which dataset was used. Guessing blindly is right half the time, so the advantage measures how much better than half the attacker does, doubled so that a perfect attacker scores 1. With a fair coin, the chance of a correct guess is the average of the TPR and one minus the FPR, so the advantage equals the TPR minus the FPR. Its best value over all attackers is the **total variation (TV)** distance between the two output distributions ([Threat models and privacy games](../part02-attacks/01-threat-models.md)).

The testing form bounds the advantage. A high advantage needs both error rates small at once, so the attacker wants the smallest possible sum of the two. The two lines of @eq-dpdef-testing keep it away from zero. Along each line the sum of the error rates changes at a constant rate, so the smallest sum sits at a corner of the allowed region. Of those corners, the one with the smallest sum is where the two lines cross. Evaluating the advantage there gives the bound for every attacker against an $(\varepsilon, \delta)$-DP mechanism,
$$\mathrm{TPR} - \mathrm{FPR} \le \frac{e^\varepsilon - 1 + 2\delta}{e^\varepsilon + 1}.$$ {#eq-dpdef-advantage}

For pure DP, the right side equals the hyperbolic tangent of $\varepsilon/2$, which is about $\varepsilon/2$ when $\varepsilon$ is small. Randomized response attains the bound exactly, because its best test sits at the crossing point. Some of Part 2's games first draw the dataset at random. The bound still holds there, because it holds for every fixed pair of neighbours and so for their average.

A small $\varepsilon$ gives strong protection. At $\varepsilon = 0.1$ the bound is about 0.05. That leaves the best attacker barely better than a coin flip, since it is right with probability at most 0.525. A large $\varepsilon$, by contrast, promises little, because the bound at $\varepsilon = 5$ is 0.9866. Such a bound allows an attacker that guesses right 99.3% of the time. Models are sometimes trained at an even larger $\varepsilon$, such as 8. Such a model may resist real attacks far better than the bound suggests, but the guarantee itself no longer promises it.

An older bound, from Yeom et al. (2018), limits the advantage under pure DP by $e^\varepsilon - 1$. At small $\varepsilon$ it has the same order as the bound above, but it is about twice as large. And once $e^\varepsilon$ reaches 2, it reaches 1 and says nothing.

Most mechanisms do not reach the bound, because it describes the worst mechanism with a given $\varepsilon$. For a count with Laplace noise of scale $1/\varepsilon$, the best advantage is the total variation distance between two Laplace distributions one unit apart. [Divergences](../part01-prerequisites/03-divergences.md) computes it as $1 - e^{-\varepsilon/2}$, which stays below the bound at every $\varepsilon$ (@tbl-dpdef-advantage and @fig-dpdef-advantage).

| $\varepsilon$ | 0.1 | 0.5 | 1 | $\ln 3$ | 2 | 5 |
| --- | --- | --- | --- | --- | --- | --- |
| Bound $\tanh(\varepsilon/2)$ | 0.0500 | 0.2449 | 0.4621 | 0.5000 | 0.7616 | 0.9866 |
| Older bound $e^\varepsilon - 1$ | 0.1052 | 0.6487 | 1.7183 | 2.0000 | 6.3891 | 147.4 |
| Randomized response | 0.0500 | 0.2449 | 0.4621 | 0.5000 | 0.7616 | 0.9866 |
| Laplace count | 0.0488 | 0.2212 | 0.3935 | 0.4226 | 0.6321 | 0.9179 |

: The membership advantage bound for $\varepsilon$-DP, and the exact best advantage against two $\varepsilon$-DP mechanisms. {#tbl-dpdef-advantage}

![The largest membership advantage against an $\varepsilon$-DP mechanism (black), the looser bound $e^\varepsilon - 1$ (dotted), and the exact best advantage against a count with Laplace noise of scale $1/\varepsilon$ (orange). Markers are advantages measured in 100,000 simulated membership games with the likelihood-ratio adversary; randomized response sits on the bound.](figures/dpdef_advantage.svg){#fig-dpdef-advantage}

A small $\delta$ barely moves this bound. At $\varepsilon = 1$, the bound for pure DP is 0.4621. Adding a $\delta$ of $10^{-5}$ raises it only to 0.462123. The advantage is an average over the coin flip, and it hides harm that is concentrated on a few people. That harm shows up at a low FPR, where @sec-dpdef-testing found the guarantee resting on $\delta$, and @sec-dpdef-small-delta shows how bad it can get.

## The Bayesian interpretation {#sec-dpdef-bayes}

People often ask a different question: after seeing the output, how sure can someone be about me? That is a question about beliefs. Bayes' rule, recalled in [Hypothesis testing](../part01-prerequisites/04-hypothesis-testing.md), says that the odds after seeing an output are the odds before it times the likelihood ratio of the output. The likelihood ratio here is $e^L$, where $L$ is the privacy loss. Pure DP keeps that loss between $-\varepsilon$ and $\varepsilon$. So one output can multiply or divide the odds between two neighbouring datasets by at most $e^\varepsilon$, whatever anyone believed before (Kasiviswanathan & Smith, 2014).

Take an adversary who knows that the data are one of two neighbours. Write $\pi$ for its belief, before seeing the output, that the data are $D$. Its odds for $D$ are then $\pi/(1 - \pi)$. One output can raise these odds to at most $e^\varepsilon$ times that. Converting the largest posterior odds back into a probability gives the largest belief in $D$ after any output of an $\varepsilon$-DP mechanism:
$$\Pr[D \mid y] \le \frac{e^\varepsilon\pi}{e^\varepsilon\pi + 1 - \pi}.$$ {#eq-dpdef-posterior}
The bound holds for any prior, right or wrong, because it bounds only the likelihood ratio, which does not depend on the prior.

Go back to Warner's survey, which has a ratio of 3. The interviewer knows everyone's answer except Alice's, and thinks there is an even chance that her true answer is yes. She reports "yes", which is 3 times as likely from a true yes. The odds move from 1 to 3, so the interviewer can now be at most 75% sure. If the interviewer's prior was 1%, the posterior is at most 0.0294. @tbl-dpdef-posterior gives the bound for other values, and @fig-dpdef-posterior draws it against the prior.

| $\varepsilon$ | Prior 0.01 | Prior 0.1 | Prior 0.5 |
| --- | --- | --- | --- |
| 0.1 | 0.0110 | 0.1094 | 0.5250 |
| 1 | 0.0267 | 0.2320 | 0.7311 |
| $\ln 3$ | 0.0294 | 0.2500 | 0.7500 |
| 3 | 0.1687 | 0.6906 | 0.9526 |

: The largest posterior probability after one output of an $\varepsilon$-DP mechanism, from @eq-dpdef-posterior. {#tbl-dpdef-posterior}

![The largest (solid) and smallest (dashed) posterior probability of $D$ after one output of an $\varepsilon$-DP mechanism, against the prior, for four values of $\varepsilon$. The dotted diagonal is no change. Small priors stay small: even at $\varepsilon = 3$ a prior of 0.01 can rise only to 0.169.](figures/dpdef_posterior.svg){#fig-dpdef-posterior}

The guarantee bounds the change in a belief, not the belief itself. An adversary who is already 99% sure from other sources stays about that sure, and the release only promises not to make it much surer. The adversary in this statement is also as strong as possible, since it knows every record except the one in question. Against weaker adversaries, who are also unsure about other records, a version of the bound still holds. It compares their belief about one person after the release with the belief they would form from the data without that person's record.

Under approximate DP, the privacy loss can exceed $\varepsilon$ on some outputs, so the posterior bound cannot hold for every output. Instead, it holds at a slightly weaker level for all but a few outputs. To make that precise, pick a level $\varepsilon'$ above $\varepsilon$, and call an output exceptional if its loss exceeds $\varepsilon'$. On every other output, the odds move by at most $e^{\varepsilon'}$, as in the pure case.

By the weighting of @sec-dpdef-divergences, each exceptional output counts toward $\delta$ with a weight of at least $1 - e^{\varepsilon - \varepsilon'}$. Their weighted total is at most $\delta$, so their total probability on $D$ is at most
$$\frac{\delta}{1 - e^{\varepsilon - \varepsilon'}}.$$ {#eq-dpdef-bad-set}
Take the Gaussian count again, which is $(1, 0.00683)$-DP. With $\varepsilon' = 2$, the chance of an exceptional output is bounded by 0.0108. The exact chance is $8.842\times10^{-5}$, far smaller, because the bound uses only the pair $(\varepsilon, \delta)$ and not the shape of the Gaussian tail. A tiny $\delta$ keeps the exceptions tiny, and a large $\delta$ does not.

## How small δ must be {#sec-dpdef-small-delta}

The definition lets $\delta$ be anything between 0 and 1, and the advantage bound barely notices a small one. A deliberately terrible mechanism shows how small it must be. The mechanism goes through the dataset and publishes each record, in full, independently with probability $\delta$. Under add/remove, it is $(0, \delta)$-DP. Adding a record, call it $x$, to the data changes the output only if $x$'s own coin publishes it, which happens with probability $\delta$. Otherwise the output does not depend on $x$ at all, so every set of outputs changes probability by at most $\delta$.

Yet over a whole dataset of $n$ records, the mechanism publishes $n\delta$ records on average. To see how likely an exposure is, note that each record escapes with probability $1 - \delta$, independently of the others. So all of them escape with probability $(1 - \delta)^n$, and at least one is published with probability
$$1 - (1 - \delta)^n \approx 1 - e^{-n\delta}.$$
So with 10,000 records and $\delta = 1/n$, someone's full record is published with probability 0.6321 (@tbl-dpdef-release and @fig-dpdef-release). A guarantee of $(0, 1/n)$-DP is therefore compatible with publishing a random person's full record in most releases.

| $\delta$ | Expected records published, $n\delta$ | Probability that at least one is published |
| --- | --- | --- |
| $10^{-4} = 1/n$ | 1 | 0.6321 |
| $10^{-6}$ | 0.01 | $9.950\times10^{-3}$ |
| $10^{-8}$ | 0.0001 | $9.999\times10^{-5}$ |

: Exposure under the $(0, \delta)$-DP mechanism that releases each of 10,000 records with probability $\delta$. {#tbl-dpdef-release}

![The probability that the mechanism "release each record with probability $\delta$" publishes at least one of $n = 10{,}000$ records (red), against $n\delta$, on log axes. Below $n\delta \approx 0.1$ the probability is essentially $n\delta$; at $\delta = 1/n$ it is already 0.63.](figures/dpdef_release.svg){#fig-dpdef-release}

So think of $\delta$ as a per-person budget for catastrophe. The definition lets each person be exposed with probability up to about $\delta$, which over $n$ people means about $n\delta$ exposures on average. Only when $n\delta$ is much smaller than 1 is the expected number of fully exposed people small, so a meaningful guarantee needs $\delta$ much smaller than $1/n$.

The testing picture shows the same thing. The test that asks whether the record is in the output never fires when the record is absent, so its FPR is 0. Yet it fires with probability $\delta$ when the record is present. All of the $\delta$ is spent at an FPR of 0, where pure DP allows nothing. The advantage bound gives no warning, since at $\varepsilon = 0$ it allows an advantage of only $\delta$, an average that hides a few people exposed completely.

In practice, pick $\delta$ well below $1/n$, for example $1/n^{1.5}$. A fixed value is also common. When Abadi and coauthors first trained neural networks with DP, they used $10^{-5}$ for 60,000 training examples. That value is only just below one over the number of examples. When one person contributes many records, $n$ must count people, not records, and the group-privacy cost of [Post-processing, group privacy and composition](02-properties.md) also multiplies $\delta$. For example, a dataset of 10 million users needs $\delta$ well below one over 10 million, which is $10^{-7}$.

A small $\delta$ usually costs little. Reasonable mechanisms, like the Gaussian count, spread their $\delta$ over many outputs with a moderately large loss, rather than over a few outputs with an infinite loss. Their privacy curve therefore falls steeply, so a slightly larger $\varepsilon$ buys a much smaller $\delta$, as @tbl-dpdef-gauss showed.

## What differential privacy does not protect {#sec-dpdef-limits}

The definition compares two worlds, one with your record and one without it. Anything that can be learned in both worlds is outside the guarantee, and three kinds of inference fall there. People's intuitions about privacy often go beyond what DP promises in exactly these cases.

The first is what the output reveals about the population. Suppose a study that never used Alice's record shows that smokers get cancer at a high rate. Your belief that Alice, a smoker, will get cancer goes up, even though the study is 0-DP with respect to her. In both worlds that DP compares, with and without Alice, the study shows the same thing, so DP correctly says that nothing happened to Alice because she took part. Preventing this kind of inference would mean preventing the study from learning anything at all. Kifer & Machanavajjhala (2011) showed that no definition can protect against every inference without assumptions about how the data were generated.

The second is what the output reveals about correlated records. DP treats records as separate. It hides each record by making the output nearly the same with or without it, which works only if the other records do not give the same answer away. But if five people in a household always answer alike, the others carry the same information, and an attacker who learns one answer learns all five.

For example, take a count released with Laplace noise at $\varepsilon = 1$ per record. The household moves the count by 5, so its largest privacy loss is 5.0000, five times $\varepsilon$. In general, a group of $k$ records that move together can reach a loss of $k\varepsilon$. So the household is protected only as well as a 5-DP mechanism would protect it. Against this count, the exact best advantage of an attacker asking whether the household took part is 0.9179. That attacker can therefore be right with probability up to 0.959, although each record is "protected at $\varepsilon = 1$".

The remedy is to match the unit of privacy to the unit of correlation, and protect households or users rather than rows. Correlations you do not know about, however, stay outside the guarantee.

The third is anything outside the mathematical mechanism. A proof of DP covers an algorithm, not the code that runs it. Floating-point noise samplers can leak through their low-order bits ([Discrete and floating-point-safe mechanisms](07-discrete-mechanisms.md)), and timing can leak too. The raw data stored on a server, and releases of the same data through other channels, are also outside the guarantee. The implementation and everything around it need their own review.

## Code {#sec-dpdef-code}

This snippet turns a claimed $(\varepsilon, \delta)$ into what an attacker can achieve, and a measured attack into a lower bound on $\varepsilon$:

```python
from math import exp, log, sqrt


def advantage_bound(eps: float, delta: float = 0.0) -> float:
    """The largest membership advantage, TPR - FPR, against any (eps, delta)-DP mechanism."""
    return (exp(eps) - 1 + 2 * delta) / (exp(eps) + 1)


def tpr_cap(eps: float, delta: float, fpr: float) -> float:
    """The largest true positive rate any attack can reach at a given false positive rate."""
    return min(1.0, exp(eps) * fpr + delta)


def epsilon_lower_bound(tpr: float, fpr: float, delta: float) -> float:
    """The smallest eps consistent with an attack measured at (fpr, tpr)."""
    return log((tpr - delta) / fpr)


def posterior_bound(eps: float, prior: float) -> float:
    """The largest posterior probability after one output of an eps-DP mechanism."""
    return exp(eps) * prior / (exp(eps) * prior + 1 - prior)


def rr_sd(eps: float, n: int) -> float:
    """Standard deviation of the randomized-response estimate of a fraction from n people."""
    return exp(eps / 2) / ((exp(eps) - 1) * sqrt(n))


def exposure(delta: float, n: int) -> float:
    """Probability that 'publish each record with probability delta' exposes someone."""
    return 1 - (1 - delta) ** n


print(f"{advantage_bound(1.0):.4f}")  # 0.4621
print(f"{tpr_cap(1.0, 1e-5, 1e-3):.6f}")  # 0.002728
print(f"{tpr_cap(2.0, 1e-5, 1e-3):.4f}")  # 0.0074
print(f"{epsilon_lower_bound(0.05, 1e-3, 1e-5):.2f}")  # 3.91
print(f"{posterior_bound(1.0, 0.01):.4f}")  # 0.0267
print(f"{rr_sd(log(3), 1000):.4f}")  # 0.0274
print(f"{exposure(1e-4, 10_000):.4f}")  # 0.6321
```

A real audit must also account for the uncertainty in measured rates, since a TPR measured on a few canaries can be far from the true one ([Membership inference](../part02-attacks/02-membership-inference.md)). The full edition's script checks every number in this chapter by enumeration and simulation:

```bash
uv run python code/part03-dp-foundations/definition.py
```

## Cheat sheet

| Result | Formula | Notes |
| --- | --- | --- |
| $(\varepsilon, \delta)$-DP | $\Pr[M(D) \in S] \le e^\varepsilon\Pr[M(D') \in S] + \delta$ for all neighbours, all $S$ | State the neighbouring relation and the unit of privacy |
| Pure DP | $\delta = 0$; privacy loss within $\pm\varepsilon$ for every output | Can be checked output by output; approximate DP cannot |
| Add/remove vs replace | $\varepsilon$ under add/remove gives $2\varepsilon$ under replace | Only add/remove hides participation |
| Smallest $\delta$ | $\mathbb E_{M(D)}[\max\{0, 1 - e^{\varepsilon - L}\}]$, worst pair | Not the chance that the loss exceeds $\varepsilon$, which is larger |
| Randomized response | Truth with probability $e^\varepsilon/(1 + e^\varepsilon)$; error $\approx 1/(\varepsilon\sqrt n)$ | Local: no trusted collector; replace only |
| Attack cap | $\mathrm{TPR} \le e^\varepsilon\,\mathrm{FPR} + \delta$ | Audit: $\varepsilon \ge \ln((\mathrm{TPR} - \delta)/\mathrm{FPR})$ |
| Membership advantage | $\le (e^\varepsilon - 1 + 2\delta)/(e^\varepsilon + 1)$; $\tanh(\varepsilon/2)$ for pure DP | Tight for randomized response |
| Posterior | Odds multiplied by at most $e^\varepsilon$ | Bounds the change in belief, not the belief |
| Choosing $\delta$ | $\delta \ll 1/n$, with $n$ people | $\delta = 1/n$ can expose someone with probability 0.63 |

: The results of this chapter. {#tbl-dpdef-cheat}

**Rules of thumb**

- State the neighbouring relation and the unit of privacy with every $\varepsilon$.
- An $\varepsilon$ below 1 is strong, a few is moderate, and around 10 the worst-case guarantee says little.
- Pick $\delta$ much smaller than $1/n$, with $n$ the number of people, not records.
- Compare a measured membership attack with the cap $e^\varepsilon\,\mathrm{FPR} + \delta$ at a low FPR, not with the advantage bound.
- A bounded privacy loss gives pure DP at its largest value. An unbounded one gives only approximate DP, a curve $\delta(\varepsilon)$ that never reaches 0.
- DP protects your contribution, not secrets that other people's data reveal about you.

## Further reading

- Dwork & Roth, *The Algorithmic Foundations of Differential Privacy* (2014), ch. 2: the definition, randomized response, what DP does and does not promise, and the "release a random record" argument for small $\delta$.
- Kairouz, Oh & Viswanath, "The Composition Theorem for Differential Privacy" (ICML 2015): the hypothesis-testing reading behind the attack cap and auditing.
- Kasiviswanathan & Smith, "On the 'Semantics' of Differential Privacy: A Bayesian Formulation" (2014): the posterior reading for any prior, including against weaker adversaries.
- Kifer & Machanavajjhala, "No Free Lunch in Data Privacy" (SIGMOD 2011): why DP cannot protect correlated data without assumptions on how it was generated.

**Derivations.** Every result in this chapter is derived step by step in [the full edition's chapter](../../book-full/part03-dp-foundations/01-definition.md), except the posterior bound against weaker adversaries, the older advantage bound $e^\varepsilon - 1$ and the No-Free-Lunch result, which it cites.
