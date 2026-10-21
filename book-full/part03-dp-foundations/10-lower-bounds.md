# Lower bounds: what privacy costs

## Why this matters

The earlier chapters of this part built mechanisms and measured their error. The Laplace mechanism releases a histogram over $N$ bins with error about $\ln N/\varepsilon$ in every bin. The Gaussian mechanism releases $d$ attribute means with error about $\sqrt{d\ln(1/\delta)}/(n\varepsilon)$ each. These are **upper bounds**: they say what one mechanism achieves. They leave one question open. Is the error the price of privacy, or the price of a clumsy mechanism? A cleverer algorithm might do much better.

A **lower bound** answers it. It says that *no* mechanism satisfying the definition can be more accurate than some amount, whatever it does. When the lower bound matches an upper bound, the question is closed. You know the mechanism is optimal, and you know that asking for more accuracy means asking for less privacy or for more data. This chapter proves the three standard lower bounds of differential privacy (DP), and each one matches a mechanism you already know:

- **Packing** (pure $\varepsilon$-DP): an output that must pick one of $m$ candidates costs error growing like $\ln m/\varepsilon$. Laplace histograms are optimal up to a constant.
- **Fingerprinting** (approximate $(\varepsilon, \delta)$-DP): releasing $d$ attribute means needs $n$ growing like $\sqrt d$. The Gaussian mechanism has this rate.
- **Reconstruction** (any $\varepsilon$): answering about $n$ random counting queries needs error of order $\sqrt n$ per answer. This is the Dinur–Nissim attack of [Reconstruction attacks](../part02-attacks/09-reconstruction.md), read as a lower bound.

### The one idea: accuracy is an attack

How can you prove that *every* mechanism fails, when there are infinitely many mechanisms and you cannot try them all? You do not look at how a mechanism works at all. You only use the two things you know about it, that it is accurate and that it is private, and show that the two cannot hold together. Every proof in this chapter does this in the same three steps.

1. **Accuracy gives an attack.** If a mechanism is accurate, its output tells you something about the data, and an attacker can use that: to pick out which of many datasets was used, to tell whether a person is in the dataset, or to reconstruct secret bits. The more accurate the output, the better the attack works.
2. **DP limits every attack.** The definition says that no test can tell the outputs on two neighbouring datasets apart much better than chance. Group privacy extends that to datasets that differ in $k$ records, with a weaker limit that degrades with $k$.
3. **Compare.** If the attack built from an accurate mechanism would succeed more often than DP allows, the mechanism cannot be both accurate and private. Writing this as an inequality and solving it for the error, or for the number of records $n$, gives the lower bound.

So a lower bound on error is the same thing as an attack on accurate mechanisms. The fingerprinting proof makes this literal: its attack is a membership inference attack of the kind in [Membership inference](../part02-attacks/02-membership-inference.md), and you will run it.

@tbl-lower-overview lists the results and where each is used.

| Result | Privacy | Where it is used |
| --- | --- | --- |
| Packing lemma | Pure | Lower bounds for any output that must single out one of many candidates |
| Mode and histogram bounds | Pure | Showing Laplace histograms and report-noisy-max are optimal |
| Sample complexity | Pure and approximate | How many records a private analysis needs; when privacy is cheap |
| Packing under $\delta > 0$ | Approximate | Why the Gaussian mechanism can beat every pure-DP mechanism; the $\ln(1/\delta)/\varepsilon$ cost of stability-based releases |
| Packing for a mean in $[-R, R]$ | Pure and approximate | The $\ln R/\varepsilon$ cost of not knowing where the data lie |
| Fingerprinting lemma and the $\sqrt d$ bound | Approximate | Optimality of the Gaussian mechanism for many means; the cost of high-dimensional releases |
| Tracing attack | Any | Membership inference on released statistics; auditing noisy means |
| Dinur–Nissim as a lower bound | Any | Why every query answered privately must carry noise of order $\sqrt n$ |

: The results of this chapter and where each is used. {#tbl-lower-overview}

**How to read this chapter.** The Toolkit has three facts; the packing and fingerprinting sections each use two of them. On a first pass, read the packing section, the statement and interpretation of the fingerprinting section, and the tracing section, which is the fingerprinting proof turned into an experiment. The fingerprinting derivation is the hardest piece of mathematics in the chapter. It is short, but come back to it after the tracing simulation has made the score concrete.

**What this chapter assumes.** The chapter restates what it uses where it uses it, so the links below are for more detail. The definitions of DP, its hypothesis-testing form and its neighbouring relations are in [The definition of differential privacy](01-definition.md). Group privacy is in [Post-processing, group privacy and composition](02-properties.md), and the [Laplace and Gaussian mechanisms](03-laplace-gaussian.md) have their own chapter. Report-noisy-max is in [Selection](05-selection.md). From Part 1 the chapter uses Hoeffding's inequality ([Concentration](../part01-prerequisites/02-concentration.md)), the link between the advantage of a test and total variation (TV) ([Hypothesis testing](../part01-prerequisites/04-hypothesis-testing.md)) and Fano's method ([Information theory](../part01-prerequisites/05-information-theory.md)), which is the non-private ancestor of packing.

**Conventions.** $M$ is a randomized mechanism: it takes a dataset and returns a random output. Two datasets $D \simeq D'$ are **neighbours** when they differ in one record. Recall the definition: $M$ is **$(\varepsilon, \delta)$-DP** if for every pair of neighbours and every set $S$ of outputs,
$$\Pr[M(D) \in S] \le e^\varepsilon\Pr[M(D') \in S] + \delta.$$
In words, no event becomes more than $e^\varepsilon$ times more likely when one record changes, up to a small additive slack $\delta$. Because the neighbour relation is symmetric, the inequality also holds with $D$ and $D'$ swapped. **Pure** DP is the case $\delta = 0$, written $\varepsilon$-DP; **approximate** DP allows $\delta > 0$.

What "differ in one record" means depends on the setting. Histograms use **add/remove** neighbours: one dataset has one more record than the other, so one count changes by 1. The marginal, tracing and reconstruction sections use **replace** neighbours on datasets of a fixed size $n$: one record is swapped for another. They use it because they speak about the number of records $n$, which add/remove would change.

An accuracy guarantee has two parameters: the **error** $\alpha$ and the **failure probability** $\beta$. "$M$ is $(\alpha, \beta)$-accurate" means that, on *every* dataset, its error is at most $\alpha$ with probability at least $1 - \beta$. (In [Hypothesis testing](../part01-prerequisites/04-hypothesis-testing.md) $\alpha$ and $\beta$ were the error rates of a test. Here they are not, and tests are described instead by two rates: the **true positive rate** (TPR), how often a test fires when the data are $D'$, the dataset with the record, and the **false positive rate** (FPR), how often it fires when the data are $D$, the dataset without it. This is the convention of [The definition of differential privacy](01-definition.md), where a test that says "$D'$" fires.) The examples use $\varepsilon = 1$, $\beta = 0.05$ and, for approximate DP, $\delta = 10^{-6}$.

## Toolkit: three facts every derivation uses

The definition of DP talks about one event and one pair of neighbours at a time. The proofs below need three consequences of it: what happens when datasets are $k$ records apart (Fact 1), how much the average of a score can move (Fact 2), and how well any test or guess can do (Fact 3). Each fact is stated, proved once, and given a number.

### Fact 1 — Group privacy

DP only compares neighbours, but the packing proof needs to compare datasets that are $k$ records apart. The idea is to walk from one to the other one record at a time. Each step lets probabilities grow by a factor $e^\varepsilon$ and adds $\delta$, so after $k$ steps the factor is $e^{k\varepsilon}$ and the additive terms have piled up. If $M$ is $(\varepsilon, \delta)$-DP and $D$, $D''$ differ in $k$ records (a chain of $k$ neighbouring steps), then for every set of outputs $S$
$$\Pr[M(D) \in S] \le e^{k\varepsilon}\Pr[M(D'') \in S] + \delta_k, \qquad \delta_k = \delta\sum_{j=0}^{k-1}e^{j\varepsilon} \le k\,e^{(k-1)\varepsilon}\delta.$$ {#eq-lower-group}

*Why it holds.* Write the chain as $D = D_0 \simeq D_1 \simeq \dots \simeq D_k = D''$ and apply the definition at each link: $\Pr[M(D_{j}) \in S] \le e^\varepsilon\Pr[M(D_{j+1}) \in S] + \delta$. Substitute the second inequality into the first, the third into the result, and so on. For two steps this gives $\Pr[M(D_0) \in S] \le e^\varepsilon\big(e^\varepsilon\Pr[M(D_2) \in S] + \delta\big) + \delta = e^{2\varepsilon}\Pr[M(D_2) \in S] + \delta(1 + e^\varepsilon)$. In general, the $\delta$ that enters at link $j$ (counting from 0) is multiplied by $e^\varepsilon$ once for each of the $j$ links before it, which gives the sum $\delta\sum_{j=0}^{k-1}e^{j\varepsilon}$. Each of its $k$ terms is at most the largest one, $e^{(k-1)\varepsilon}\delta$, which gives the simpler bound. [Post-processing, group privacy and composition](02-properties.md) does this in full.

*How fast it degrades.* With $\varepsilon = 1$ and $\delta = 10^{-6}$, for $k = 5$ steps, $\delta_5 = 10^{-6}(1 + e + e^2 + e^3 + e^4) = 8.6\times10^{-5}$, still small. For $k = 20$, $\delta_{20} = 282$, a bound on a probability that is larger than 1 and so says nothing at all. The geometric growth of $e^{j\varepsilon}$ is what makes approximate DP useless for groups of more than about $\ln(1/\delta)/\varepsilon$ records. Under pure DP ($\delta = 0$) there is no additive term and so no such blow-up, only the factor $e^{k\varepsilon}$. The packing section uses this fact.

### Fact 2 — DP bounds the expectation of a bounded score

DP bounds the probability of every event. The fingerprinting proof needs something slightly different: a bound on the *average* of a score $h$ computed from the output. The bridge is that the average of a random variable can be written as a sum of probabilities of events, "$h$ exceeds $t$", over all levels $t$. DP bounds each of those probabilities, and so it bounds the average. The result is that a score's mean on $D$ can exceed its mean on a neighbour $D'$ only by a multiple $e^\varepsilon - 1$ of its typical size on $D'$, plus a small term from $\delta$. Precisely, if $M$ is $(\varepsilon, \delta)$-DP, $D \simeq D'$, and $h$ is a function of the output with $|h| \le T$, then
$$\mathbb E\,h(M(D)) \le \mathbb E\,h(M(D')) + (e^\varepsilon - 1)\,\mathbb E\,|h(M(D'))| + 2\delta T.$$ {#eq-lower-expectation}

*Derivation.*

1. **Write the mean as an integral of tail probabilities.** A variable $Y$ with $0 \le Y \le T$ has $\mathbb E\,Y = \int_0^T\Pr[Y > t]\,dt$. To see why, take $Y$ equal to 1, 2 or 3 with probability $\frac13$ each, so $\mathbb E\,Y = 2$. Then $\Pr[Y > t]$ is 1 for $t < 1$, $\frac23$ for $1 \le t < 2$ and $\frac13$ for $2 \le t < 3$, and the integral is $1 + \frac23 + \frac13 = 2$: each value $y$ is counted once for every level $t$ below it, which adds up to $y$. A score that can be negative is split into its positive part $h_+ = \max(h, 0)$ and negative part $h_- = \max(-h, 0)$, so $h = h_+ - h_-$ and
$$\mathbb E\,h = \int_0^T\Pr[h > t]\,dt - \int_0^T\Pr[h < -t]\,dt.$$
2. **Bound the positive part from above.** Apply the definition to the event $\{h(M(D)) > t\}$: $\Pr[h(M(D)) > t] \le e^\varepsilon\Pr[h(M(D')) > t] + \delta$.
3. **Bound the negative part from below.** It is subtracted, so you need a lower bound on it. Apply the definition the other way round, with $D'$ on the left, to the event $\{h < -t\}$: $\Pr[h(M(D')) < -t] \le e^\varepsilon\Pr[h(M(D)) < -t] + \delta$. Solved for the term on $D$, this reads $\Pr[h(M(D)) < -t] \ge e^{-\varepsilon}\big(\Pr[h(M(D')) < -t] - \delta\big)$.
4. **Integrate over $t \in [0, T]$.** Write $h' = h(M(D'))$. Integrating the bounds of steps 2 and 3 turns the tail probabilities back into means, by step 1, and each $\delta$ into $\delta T$:
$$\mathbb E\,h(M(D)) \le e^\varepsilon\,\mathbb E\,h_+' - e^{-\varepsilon}\,\mathbb E\,h_-' + (1 + e^{-\varepsilon})\delta T.$$
5. **Pull out the mean on $D'$.** Adding and subtracting $\mathbb E\,h_+'$ and $\mathbb E\,h_-'$ gives $e^\varepsilon\mathbb E h_+' - e^{-\varepsilon}\mathbb E h_-' = \mathbb E h' + (e^\varepsilon - 1)\mathbb E h_+' + (1 - e^{-\varepsilon})\mathbb E h_-'$, because $\mathbb E h' = \mathbb E h_+' - \mathbb E h_-'$. The coefficient $1 - e^{-\varepsilon}$ is at most $e^\varepsilon - 1$ (their difference is $e^\varepsilon + e^{-\varepsilon} - 2 \ge 0$), so the two middle terms are at most $(e^\varepsilon - 1)(\mathbb E h_+' + \mathbb E h_-') = (e^\varepsilon - 1)\mathbb E|h'|$. Finally $1 + e^{-\varepsilon} \le 2$.

*What it allows.* If $h(M(D'))$ has mean 0 and mean absolute value 1, then $h(M(D))$ can have mean at most $e - 1 = 1.72$ at $\varepsilon = 1$ (plus $2\delta T$). A score that is centred on one dataset can move only by a multiple of its typical size on the neighbour. The fingerprinting bound uses this fact, with $h$ the tracing score.

### Fact 3 — DP bounds every membership test and every guess of a bit

A test that tries to tell $M(D)$ from $M(D')$ is any rule that looks at the output and "fires" or not. Following [The definition of differential privacy](01-definition.md), the test fires when it says "$D'$", the dataset with the record. Its quality is its **advantage**, $\mathrm{TPR} - \mathrm{FPR}$: how much more often it fires on $M(D')$ than on $M(D)$. A test that ignores the output fires equally often on both and has advantage 0; a perfect test has advantage 1. DP caps the advantage, because it constrains how often a test can fire on one dataset given how often it fires on the other, and it does so in both directions. If $M$ is $(\varepsilon, \delta)$-DP and $D \simeq D'$, every test that tries to tell $M(D)$ from $M(D')$ has
$$\mathrm{TPR} - \mathrm{FPR} \le \frac{e^\varepsilon - 1 + 2\delta}{e^\varepsilon + 1}.$$ {#eq-lower-advantage}
The same cap gives a limit on guessing a secret. If one record holds a uniformly random secret bit and $D, D'$ differ only in that bit, every guess of the bit from the output is right with probability at most $(e^\varepsilon + \delta)/(e^\varepsilon + 1)$.

*Derivation.*

1. **Two constraints from the definition.** Let the test fire with probability $b$ on $M(D')$ (its TPR) and $a$ on $M(D)$ (its FPR). The definition applied to the pair $(D', D)$ and the event "fires" gives $b \le e^\varepsilon a + \delta$. Applied in the other direction, to the pair $(D, D')$ and the complement "does not fire", it gives $1 - a \le e^\varepsilon(1 - b) + \delta$.
2. **Each constraint caps the advantage.** The first gives $b - a \le (e^\varepsilon - 1)a + \delta$, a cap that grows with $a$. The second, solved for $b$, gives $b \le 1 - e^{-\varepsilon}(1 - a - \delta)$, so $b - a \le (1 - a)(1 - e^{-\varepsilon}) + e^{-\varepsilon}\delta$, a cap that shrinks with $a$. A test is limited by the smaller of the two caps.
3. **The best test sits where the caps meet.** The smaller of a rising and a falling line is largest at their crossing. Setting $(e^\varepsilon - 1)a + \delta = (1 - a)(1 - e^{-\varepsilon}) + e^{-\varepsilon}\delta$ and multiplying by $e^\varepsilon$ gives $(e^\varepsilon - 1)(e^\varepsilon + 1)a = (e^\varepsilon - 1)(1 - \delta)$, so $a = (1 - \delta)/(e^\varepsilon + 1)$. There the first cap is $(e^\varepsilon - 1)(1 - \delta)/(e^\varepsilon + 1) + \delta = (e^\varepsilon - 1 + 2\delta)/(e^\varepsilon + 1)$.
4. **From tests to guesses of a bit.** A guess of the bit is a test: it says "1" on some set $A$ of outputs. With the bit equal to 1 or 0 with probability $\frac12$ each, it is right with probability $\frac12\Pr[A \mid 1] + \frac12\big(1 - \Pr[A \mid 0]\big) = \frac12 + \frac12(b - a)$, where $b$ and $a$ are the probabilities of $A$ in the two worlds. Inserting the cap of step 3 gives $\frac12 + \frac12\cdot\frac{e^\varepsilon - 1 + 2\delta}{e^\varepsilon + 1} = \frac{e^\varepsilon + \delta}{e^\varepsilon + 1}$.

*What it allows.* At $\varepsilon = 1$, $\delta = 0$ the advantage is at most $(e - 1)/(e + 1) = 0.4621$, and a secret bit is guessed correctly at most $e/(e + 1) = 73.11\%$ of the time. The tracing attack and the reconstruction bound use this fact. [The definition of differential privacy](01-definition.md) derives the full trade-off curve, the largest TPR any test can reach at each FPR, of which this bound is the single point where the advantage is largest.

## Packing: pure DP pays the logarithm of the number of candidates {#sec-lower-packing}

### Context

Suppose the output of a mechanism must point to one of $m$ candidates: the most common ZIP code among $N = 10^6$, the bin of a histogram that holds a spike, the corner of a cube that a dataset sits at. For each candidate $i$ you can build a dataset $D_i$ on which a good answer must point to $i$. If the mechanism is accurate, $M(D_i)$ lands in its own region $G_i$ of outputs almost always.

Now run the mechanism on a single reference dataset $D_0$ instead. Its output must land somewhere, and its probabilities add up to 1, so on average each of the $m$ regions gets at most a $1/m$ share. But under pure DP, group privacy (Fact 1) says that no event is more than $e^{k\varepsilon}$ times more likely under $M(D_i)$ than under $M(D_0)$, where $k$ is the number of records that separate $D_0$ from $D_i$. Region $G_i$ has probability almost 1 under $M(D_i)$, so it must have probability at least about $e^{-k\varepsilon}$ under $M(D_0)$. That holds for *every* region at once. The two requirements clash when $m$ is large: a probability budget of 1 has to be shared among $m$ regions, each of which demands $e^{-k\varepsilon}$. This counting argument is **packing**. It is the private version of Fano's method in [Information theory](../part01-prerequisites/05-information-theory.md), which shows that an estimator choosing among many hypotheses whose data distributions are hard to tell apart must often choose wrong.

### Statement

The lemma turns the budget argument into one inequality. The $m$ demands of size $(1 - \beta)e^{-k\varepsilon}$ must fit in a total of 1, so $k$ must be at least the logarithm of the number of candidates, divided by $\varepsilon$.

**Packing lemma.** Let $D_0, D_1, \dots, D_m$ be datasets with $D_0$ within $k$ records of every $D_i$, and let $G_1, \dots, G_m$ be disjoint sets of outputs. If $M$ is $\varepsilon$-DP and $\Pr[M(D_i) \in G_i] \ge 1 - \beta$ for every $i$, then
$$m(1 - \beta)e^{-k\varepsilon} \le 1, \qquad\text{that is,}\qquad k \ge \frac{\ln\big(m(1 - \beta)\big)}{\varepsilon}.$$ {#eq-lower-packing}

Two consequences follow by choosing the datasets. The **mode** is the most frequent element of the data. To report it reliably, the most frequent element must lead the others by enough records, and the lemma says how many. Any $\varepsilon$-DP mechanism (add/remove) that returns the most frequent element of a universe of size $N$ with probability at least $1 - \beta$, whenever that element appears $k$ times and every other element appears 0 times, needs
$$k \ge \frac{\ln\big(N(1 - \beta)\big)}{\varepsilon}.$$ {#eq-lower-mode}

A **histogram** is the vector of counts of each of $N$ elements, and its $\ell_\infty$ error is the largest error over the bins. A histogram that is accurate to within $\alpha$ in every bin can be used to find a spike of height more than $2\alpha$, so its error inherits the mode's bound, halved. Any $\varepsilon$-DP mechanism (add/remove) that releases a histogram over $N$ bins with $\ell_\infty$ error at most $\alpha$ with probability at least $1 - \beta$ on every dataset has
$$\alpha \ge \frac12\left(\frac{\ln\big(N(1 - \beta)\big)}{\varepsilon} - 1\right).$$ {#eq-lower-histogram}

### Derivation

**Packing lemma.**

1. **Each region gets a share of $M(D_0)$.** Apply Fact 1 with $\delta = 0$ to $D_i$ and $D_0$ and the set $G_i$. Accuracy says $G_i$ is likely under $D_i$, and group privacy says it cannot be much less likely under $D_0$: $1 - \beta \le \Pr[M(D_i) \in G_i] \le e^{k\varepsilon}\Pr[M(D_0) \in G_i]$. So $\Pr[M(D_0) \in G_i] \ge (1 - \beta)e^{-k\varepsilon}$.
2. **The shares fit into one distribution.** The $G_i$ are disjoint, so no output is counted twice, and the probabilities add: $\sum_i\Pr[M(D_0) \in G_i] = \Pr[M(D_0) \in \cup_iG_i] \le 1$.
3. **Combine.** Adding the $m$ lower bounds of step 1 and using step 2 gives $m(1 - \beta)e^{-k\varepsilon} \le 1$. Multiplying by $e^{k\varepsilon}$ and taking logarithms gives $\ln\big(m(1 - \beta)\big) \le k\varepsilon$.

@fig-lower-packing-budget draws the two sides of step 3. The required share per region falls like $e^{-k\varepsilon}$; the available share is $1/m$. Accuracy is only possible once $k$ is large enough for the required share to fit.

![The packing argument at $\varepsilon = 1$, $\beta = 0.05$. The black line is the probability that $M(D_0)$ must put on each good region, $(1 - \beta)e^{-k\varepsilon}$. The dashed lines are the average share available when there are $m$ regions. Accuracy at group distance $k$ is impossible to the left of each dot, at $k = \ln(m(1 - \beta))/\varepsilon$.](figures/lower_packing_budget.svg){#fig-lower-packing-budget}

**Mode.** The task is to build the $m$ candidate datasets and their regions. Take $D_0 = \emptyset$, the empty dataset, and, for each element $x$ of the universe, let $D_x$ be $k$ copies of $x$. On $D_x$ the mode is $x$, so a correct answer lands in $G_x = \{x\}$, and these regions are disjoint. Each $D_x$ is $k$ additions away from $D_0$, and there are $m = N$ of them. The lemma gives @eq-lower-mode.

**Histogram.** Use the same datasets. The true histogram of $D_x$ is $k\,e_x$, which has $k$ in bin $x$ and 0 elsewhere.

1. **Disjoint regions.** A correct answer on $D_x$ lies in the ball $G_x = \{\hat h : \|\hat h - k\,e_x\|_\infty \le \alpha\}$ around the true histogram. For $x \ne y$ the centres differ by $\|k\,e_x - k\,e_y\|_\infty = k$. A point in both balls would be within $\alpha$ of each centre, so the centres would be within $2\alpha$ of each other, by the triangle inequality. Hence the balls are disjoint when $k > 2\alpha$.
2. **Pick the smallest such $k$.** The lemma's bound is strongest when the datasets are as close to $D_0$ as possible, so take the smallest integer above $2\alpha$: $k = \lfloor2\alpha\rfloor + 1 \le 2\alpha + 1$.
3. **Apply the lemma.** It gives $k \ge \ln(N(1 - \beta))/\varepsilon$, and with step 2, $2\alpha + 1 \ge \ln(N(1 - \beta))/\varepsilon$. Subtracting 1 and halving gives @eq-lower-histogram.

### Interpretation

**Laplace is optimal up to a factor of about 2.** The Laplace mechanism adds independent $\mathrm{Lap}(1/\varepsilon)$ noise to every count, which suits add/remove neighbours because one record moves one count by 1 (sensitivity 1). A $\mathrm{Lap}(b)$ variable, with density $\frac1{2b}e^{-|z|/b}$, exceeds $\alpha$ in absolute value with probability $e^{-\alpha/b}$, here $e^{-\alpha\varepsilon}$. The $N$ noises are independent, so all of them stay within $\alpha$ with probability $(1 - e^{-\alpha\varepsilon})^N$, and the $\ell_\infty$ error exceeds $\alpha$ with probability $1 - (1 - e^{-\alpha\varepsilon})^N \le Ne^{-\alpha\varepsilon}$, the right side being the union bound (the probability that one of $N$ events happens is at most the sum of their probabilities). Setting the exact expression equal to $\beta$ and solving for $\alpha$ gives the error at failure probability $\beta$, exactly $-\ln\big(1 - (1 - \beta)^{1/N}\big)/\varepsilon$. Setting the union bound equal to $\beta$ gives the simpler upper bound $\ln(N/\beta)/\varepsilon$. Both the lower bound and the upper bound grow like $\ln N/\varepsilon$ (@fig-lower-histogram). The lower bound has slope $1/(2\varepsilon)$ in $\ln N$ and Laplace has slope $1/\varepsilon$, so the gap between them is a factor of about 2 in the slope, plus the constants from $\beta$.

**Report-noisy-max is optimal for the mode.** Report-noisy-max adds $\mathrm{Lap}(1/\varepsilon)$ to every count and returns the bin with the largest noisy count ([Selection](05-selection.md)). On $D_x$, bin $x$ has count $k$ and every other bin has count 0. So the mechanism fails only if some other bin's noise exceeds bin $x$'s noise by $k$ or more. The difference of two independent $\mathrm{Lap}(1/\varepsilon)$ variables has density $\frac\varepsilon4(1 + \varepsilon|z|)e^{-\varepsilon|z|}$ (the convolution of the two Laplace densities, worked out in @sec-apx-prob-laplace-diff). Substituting $u = \varepsilon z$, its tail beyond $k$ is $\frac14\int_{k\varepsilon}^\infty(1 + u)e^{-u}\,du = \frac14\big((1 + k\varepsilon)e^{-k\varepsilon} + e^{-k\varepsilon}\big) = e^{-k\varepsilon}(2 + k\varepsilon)/4$. A union bound over the $N - 1$ other bins gives a failure probability at most $(N - 1)e^{-k\varepsilon}(2 + k\varepsilon)/4$. Up to the slowly growing factor $2 + k\varepsilon$, this falls like $Ne^{-k\varepsilon}$, the same rate at which the packing lemma allows failure.

**Where packing is loose.** Two steps of the lemma give something away. Step 2 throws away the probability that $M(D_0)$ puts outside the good regions, and step 1 uses the worst-case factor $e^{k\varepsilon}$ for every region. For the Laplace histogram, $M(\emptyset)$ is pure noise around the all-zero histogram, and almost all of its mass lies outside every $G_x$, because a ball $G_x$ needs bin $x$ to be near $k$. That wasted mass is the factor of 2.

**Approximate DP escapes.** Why does the lemma need $\delta = 0$? Repeat step 1 with $\delta > 0$. Fact 1 now gives $1 - \beta \le e^{k\varepsilon}\Pr[M(D_0) \in G_i] + \delta_k$, so $\Pr[M(D_0) \in G_i] \ge e^{-k\varepsilon}(1 - \beta - \delta_k)$. Using $\delta_k \le k\,e^{(k-1)\varepsilon}\delta$, the subtracted part $e^{-k\varepsilon}\delta_k$ is at most $k\,e^{-\varepsilon}\delta$, so $\Pr[M(D_0) \in G_i] \ge e^{-k\varepsilon}(1 - \beta) - k\,e^{-\varepsilon}\delta$. The first term shrinks exponentially in $k$ and the second grows. Once $e^{-k\varepsilon}$ is smaller than about $k\delta$, the right side is negative, step 1 demands nothing of $M(D_0)$, and the lemma says nothing, however many regions there are. So under approximate DP packing can force $k$ only up to about $\ln(1/\delta)/\varepsilon$. This is not a weakness of the proof, because a mechanism really does better. The stability-based histogram (Vadhan 2017) adds noise only to the bins that are non-empty and releases a bin only if its noisy count clears a threshold of order $\ln(1/\delta)/\varepsilon$. Empty bins are never released, so there is no union bound over all $N$ bins, and its error does not grow with $N$ at all. @sec-apx-mech-histogram in the appendices proves its privacy and an error bound of $1 + (\ln(1/(2\delta)) + \ln(k/\beta))/\varepsilon$ when $k$ bins are non-empty.

**The packing lemma under approximate DP.** The same argument still gives a lower bound, only a capped one. Let $D_0, D_1, \dots, D_m$ and the disjoint sets $G_1, \dots, G_m$ be as in the lemma, but let $M$ be $(\varepsilon, \delta)$-DP. Then at least one of two things holds:
$$e^{-k\varepsilon}(1 - \beta) \le 2k\delta \qquad\text{or}\qquad m\,e^{-k\varepsilon}(1 - \beta) \le 2.$$ {#eq-lower-packing-approx}
In words, the group distance $k$ must be at least about $\min\big(\ln m, \ln(1/\delta)\big)/\varepsilon$. The proof takes two lines. The share bound above, with $e^{-\varepsilon} \le 1$, says each region gets at least $e^{-k\varepsilon}(1 - \beta) - k\delta$ of the probability of $M(D_0)$. If the first condition fails, that share is more than half of $e^{-k\varepsilon}(1 - \beta)$, and the $m$ shares must still fit into a total of 1, which is the second condition. For the $2^d$ corners of the marginals section below, at $\varepsilon = 1$, $\delta = 10^{-6}$ and $\beta = 0.05$, @eq-lower-packing-approx forces $k \ge 7$ at $d = 10$ and $k \ge 11$ at $d = 100$ and $d = 1000$. The share bound itself, without the factor 2, forces 7, 13 and 13 ([Exercise 2](../../book-exercises/part03-dp-foundations/10-lower-bounds.md#sec-sol-lower-2)). Either way the cap sits a little below $\ln(10^6) = 13.82$, because the $\delta$ term carries the factor $k$. On the packing datasets of the mode, the [stability-based release](07-sensitivity.md) reaches this cap. It reports the most common value exactly once that value leads by about $2\ln(1/\delta)/\varepsilon$ records, however many values there are, and since one replacement moves the lead by 2, that lead is about $\ln(1/\delta)/\varepsilon$ replacements.

![$\ell_\infty$ error of $\varepsilon$-DP histograms over $N$ bins at $\varepsilon = 1$, failure probability 0.05. Orange: the Laplace mechanism, exact (line), union bound (dotted, indistinguishable) and simulated (dots). Blue: the packing lower bound; no $\varepsilon$-DP mechanism has an error in the shaded region. Both grow like $\ln N$.](figures/lower_histogram.svg){#fig-lower-histogram}

### Example: a million ZIP codes

@tbl-lower-histogram compares the bounds at $\varepsilon = 1$ and $\beta = 0.05$ for three universe sizes. For $N = 10^6$, a typical number of fine-grained locations, an $\varepsilon$-DP release must allow the most common location to lead by at least $\ln(10^6\times0.95)/1 = 13.76$ records before it can be reported reliably; report-noisy-max needs a lead of 18.44, the $k$ at which its failure bound $(N - 1)e^{-k}(2 + k)/4$ equals 0.05. Every $\varepsilon$-DP histogram has, at the 95% level, an error of at least $\frac12(13.76 - 1) = 6.38$ in some bin; Laplace has 16.79. The script's simulation of the Laplace histogram at $N = 1000$ gives 9.84 against the exact 9.88, and report-noisy-max at the lead 11.09 fails 2.28% of the time against the union bound's 5%.

| $N$ | Mode: lower bound on lead $k$ | Report-noisy-max needs | Histogram: lower bound on error | Laplace error (exact) | Laplace error (union bound) |
| --- | --- | --- | --- | --- | --- |
| $10^3$ | 6.86 | 11.09 | 2.93 | 9.88 | 9.90 |
| $10^6$ | 13.76 | 18.44 | 6.38 | 16.79 | 16.81 |
| $10^9$ | 20.67 | 25.65 | 9.84 | 23.69 | 23.72 |

: Packing lower bounds against the mechanisms that achieve them, $\varepsilon = 1$, $\beta = 0.05$, add/remove neighbours. Errors and leads are in records. {#tbl-lower-histogram}

Each factor of 1000 in $N$ adds $\ln 1000 = 6.9$ to every column that grows like $\ln N$, and half that to the histogram lower bound, which carries the factor $\frac12$.

### Example: a mean known to lie in $[-R, R]$

Packing also prices a question every analyst meets: what does it cost not to know where the data lie? Suppose the records are drawn from a normal distribution with unknown mean $\mu$ and variance 1, and all you know is $|\mu| \le R$. An $\varepsilon$-DP mechanism that returns an estimate closer than $\alpha$ to $\mu$, with probability at least $1 - \beta$ whatever $\mu$ is, needs
$$n \ge \frac{\ln\big(\lfloor R/\alpha\rfloor(1 - \beta)\big)}{\varepsilon},$$ {#eq-lower-mean-packing}
where $\lfloor R/\alpha\rfloor$ is $R/\alpha$ rounded down. The cost grows with the logarithm of the range, not the range itself.

The derivation is the packing lemma with random datasets.

1. **Candidates.** Place $m = \lfloor R/\alpha\rfloor$ means $\mu_1, \dots, \mu_m$ in $[-R, R]$, spaced $2R/m \ge 2\alpha$ apart, and let $G_j$ be the open interval of radius $\alpha$ around $\mu_j$. Neighbouring intervals at most touch, so they are disjoint.
2. **Every dataset is close to a reference.** Fix any reference dataset $D_0$ of size $n$. Any other dataset of size $n$ is at most $n$ replacements away, so group privacy (Fact 1 with $\delta = 0$) gives $\Pr[M(x) \in G_j] \le e^{n\varepsilon}\Pr[M(D_0) \in G_j]$ for every dataset $x$.
3. **Average over the data.** Draw $x$ from the normal distribution with mean $\mu_j$. Accuracy says that, averaged over $x$ and the mechanism, $\Pr[M(x) \in G_j] \ge 1 - \beta$. Averaging the inequality of step 2 over $x$ leaves its right side unchanged, so $\Pr[M(D_0) \in G_j] \ge (1 - \beta)e^{-n\varepsilon}$.
4. **Combine** as in the lemma: the $m$ shares fit into a total of 1, so $m(1 - \beta)e^{-n\varepsilon} \le 1$, which is @eq-lower-mean-packing.

At $\alpha = 0.1$, $\beta = 0.05$ and $\varepsilon = 1$, the bound is $n \ge 4.55$ for $R = 10$, $9.16$ for $R = 10^3$ and $16.07$ for $R = 10^6$. These numbers are small because the cost grows only with the logarithm of the range. In $d$ dimensions the same argument charges about $d\ln(R/\alpha)/\varepsilon$ records ([Exercise 8](../../book-exercises/part03-dp-foundations/10-lower-bounds.md#sec-sol-lower-8)).

The obvious estimator clips the data to $[-R, R]$ and adds noise proportional to $R$, so the records it needs for a given error grow linearly in $R$, while packing allows a cost that grows only like $\ln R$. [Private mean estimation](11-mean-estimation.md) closes the gap with a mechanism that first finds the data's rough location with a histogram, at a cost of order $\ln R/\varepsilon$ records. Under approximate DP, @eq-lower-packing-approx caps the bound: at $\delta = 10^{-6}$ it forces only $n \ge 11$ for $R = 10^6$, and a stability-based histogram can indeed find the location with a number of records that does not depend on $R$ at all.

### Check yourself

- Why do the regions $G_i$ have to be disjoint, and what happens to the lemma if they overlap? *(Disjointness is what lets the shares add up to at most 1 in step 2. If the regions overlap, one output can count for several candidates, and $M(D_0)$ can meet all the demands with less total mass.)*
- Under replace neighbours a histogram count moves by 1 in two bins. Does the lower bound change? *(Barely: the same datasets work with $D_0$ = $n$ copies of a fixed element, giving $N - 1$ candidates instead of $N$. The Laplace upper bound doubles, because the $\ell_1$ sensitivity is 2. [Exercise 1](../../book-exercises/part03-dp-foundations/10-lower-bounds.md#sec-sol-lower-1) does this.)*

## Sample complexity: how many records privacy costs {#sec-lower-sample}

### Context

An error in counts becomes an error in fractions once you divide by $n$. An error of 6 records is large in a dataset of 50 and negligible in a dataset of a million. So a lower bound on error is a lower bound on the number of records: to get every frequency right to within $\alpha$ (a fraction), you need enough people. This view also shows how the cost of privacy compares with a cost you pay anyway, sampling error. And it is where pure and approximate DP part ways, as soon as the output has many dimensions.

### Statement

**Histograms.** Translate the histogram bound of the previous section into fractions: an error of $\alpha$ in each frequency is an error of $\alpha n$ in each count, and the packing bound says this must be at least about $\ln N/(2\varepsilon)$. An $\varepsilon$-DP histogram whose $\ell_\infty$ error is at most $\alpha n$ with probability $1 - \beta$, on every dataset of up to $n$ records, needs
$$n \ge \frac{\ln\big(N(1 - \beta)\big) - \varepsilon}{2\alpha\varepsilon}.$$ {#eq-lower-n-histogram}
For comparison, without privacy, estimating all $N$ frequencies of a distribution to within $\alpha$ from a sample takes at most $n = \ln(2N/\beta)/(2\alpha^2)$ samples.

**One-way marginals.** Records are $x \in \{-1, +1\}^d$, a person's answers to $d$ yes/no attributes, and the release is the vector of attribute means $\bar x \in [-1, 1]^d$. The datasets made of copies of one corner of the cube $\{\pm1\}^d$ give $2^d$ candidates, so packing charges $\ln 2^d = d\ln 2$, a cost linear in $d$. Any $\varepsilon$-DP mechanism (replace) with $\ell_\infty$ error at most $\alpha$ with probability $1 - \beta$ needs
$$n \ge \frac1\alpha\left(\frac{\ln\big((2^d - 1)(1 - \beta)\big)}{\varepsilon} - 1\right) \approx \frac{d\ln 2}{\alpha\varepsilon}.$$ {#eq-lower-n-marginals}
The approximation holds for large $d$, where $\ln(2^d - 1) \approx d\ln 2$ and the terms $\ln(1 - \beta)$ and $-1$ are small next to it.

Two mechanisms come close to this. The Laplace mechanism achieves it with $n = 2d\ln(d/\beta)/(\alpha\varepsilon)$, and the Gaussian mechanism, which is only $(\varepsilon, \delta)$-DP, with $n = 2\sqrt d\,z/(\mu\alpha)$. Here $z$ is the normal quantile with tail $\beta/(2d)$, that is, the number with $\Phi(-z) = \beta/(2d)$, where $\Phi$ is the standard normal cumulative distribution function. The number $\mu = \mu(\varepsilon, \delta)$ is the largest ratio of $\ell_2$ sensitivity to noise standard deviation that the exact Gaussian calibration of [The Laplace and Gaussian mechanisms](03-laplace-gaussian.md) allows. That chapter shows that Gaussian noise of standard deviation $\sigma$ on a query of $\ell_2$ sensitivity $\Delta_2$ is $(\varepsilon, \delta)$-DP exactly when $\delta \ge \Phi(\frac\mu2 - \frac\varepsilon\mu) - e^\varepsilon\Phi(-\frac\mu2 - \frac\varepsilon\mu)$ with $\mu = \Delta_2/\sigma$, a quantity that grows with $\mu$; $\mu(\varepsilon, \delta)$ is the value where equality holds ($\mu = 0.2367$ at $\varepsilon = 1$, $\delta = 10^{-6}$).

### Derivation

**Histograms.** Apply @eq-lower-histogram with the error $\alpha n$ in counts: $\alpha n \ge \frac12\big(\ln(N(1 - \beta))/\varepsilon - 1\big)$. Multiplying by $2\varepsilon$ gives $2\alpha\varepsilon n \ge \ln(N(1 - \beta)) - \varepsilon$, and dividing by $2\alpha\varepsilon$ gives @eq-lower-n-histogram. The packing datasets have $k \le n$ records, so they are among the datasets the guarantee covers.

The sampling bound comes from Hoeffding's inequality ([Concentration](../part01-prerequisites/02-concentration.md)), which says that the average of $n$ independent variables with values in $[0, 1]$ is more than $\alpha$ from its mean with probability at most $2e^{-2n\alpha^2}$. A frequency $\hat\pi$ is such an average of indicators, so $\Pr[|\hat\pi - \pi| > \alpha] \le 2e^{-2n\alpha^2}$. A union bound over the $N$ bins puts the probability that any bin is off by more than $\alpha$ at $2Ne^{-2n\alpha^2}$. Setting this to $\beta$ and solving for $n$ gives $n = \ln(2N/\beta)/(2\alpha^2)$.

**Marginals, lower bound.** This is packing again, now at a fixed size $n$, so the candidates are made by replacing records rather than adding them.

1. **Candidates.** Fix a corner $u_0 \in \{\pm1\}^d$. $D_0$ is $n$ copies of $u_0$. For each other corner $v$ ($m = 2^d - 1$ of them), $D_v$ is $D_0$ with $k$ records replaced by $v$, so $D_0$ is $k$ replacements from each $D_v$.
2. **Disjoint regions.** $D_v$ has $n - k$ copies of $u_0$ and $k$ copies of $v$, so its mean is $\frac{(n - k)u_0 + kv}{n} = u_0 + \frac kn(v - u_0)$. Two corners $v \ne v'$ differ in some coordinate by 2 (one is $-1$ there, the other $+1$), so their means differ there by $2k/n$. As in the histogram derivation, the $\ell_\infty$ balls of radius $\alpha$ around them are disjoint when the centres are more than $2\alpha$ apart: $2k/n > 2\alpha$, that is $k > \alpha n$.
3. **Apply the lemma** with the smallest such $k$, $k = \lfloor\alpha n\rfloor + 1 \le \alpha n + 1$: $\alpha n + 1 \ge \ln\big((2^d - 1)(1 - \beta)\big)/\varepsilon$. Subtracting 1 and dividing by $\alpha$ gives @eq-lower-n-marginals.

**Marginals, upper bounds.** Both mechanisms add noise scaled to how much one record can move the answer. Replacing one record changes each coordinate of that record by at most 2 (from $-1$ to $+1$), so it moves each of the $d$ means by at most $2/n$. Adding up over the coordinates, the $\ell_1$ sensitivity is $\Delta_1 = d\cdot\frac2n = 2d/n$, and the $\ell_2$ sensitivity is $\Delta_2 = \sqrt{d(2/n)^2} = 2\sqrt d/n$.

For the Laplace mechanism, the noise in each coordinate is $\mathrm{Lap}(b)$ with scale $b = \Delta_1/\varepsilon = 2d/(n\varepsilon)$. Each coordinate's noise exceeds $\alpha$ with probability $e^{-\alpha/b}$, so a union bound over the $d$ coordinates gives $\Pr[\max_j|\eta_j| > \alpha] \le de^{-\alpha/b}$. Setting this to $\beta$ gives $\alpha/b = \ln(d/\beta)$, that is $\alpha n\varepsilon/(2d) = \ln(d/\beta)$, and solving for $n$ gives $n = 2d\ln(d/\beta)/(\alpha\varepsilon)$.

For the Gaussian mechanism, the noise in each coordinate has standard deviation $\sigma = \Delta_2/\mu$. A normal variable exceeds $z$ standard deviations in absolute value with probability $2\Phi(-z)$, so a union bound gives $\Pr[\max_j|\eta_j| > \sigma z] \le 2d\,\Phi(-z) = \beta$, which is how $z$ was chosen. Setting $\sigma z = \alpha$, that is $2\sqrt d\,z/(n\mu) = \alpha$, and solving for $n$ gives $n = 2\sqrt d\,z/(\mu\alpha)$.

### Interpretation

**Privacy adds a term, it does not multiply.** For histograms, the private requirement grows like $\ln N/(\alpha\varepsilon)$ and the statistical one like $\ln N/\alpha^2$. (Fano's method gives a matching statistical lower bound of the same order, so sampling really costs that much.) A data analyst needs the larger of the two. Their ratio, private over statistical, is about $\alpha/\varepsilon$, so the private term dominates only when $\alpha \gtrsim \varepsilon$, that is, for coarse answers at strong privacy. At $\varepsilon = 1$ and any accuracy finer than 0.1, the records you need for sampling error alone already pay for privacy.

**Pure DP pays linearly in the dimension.** With $d$ attributes the number of candidates is $2^d$, so packing charges $\ln 2^d = d\ln2$. The Laplace mechanism matches it up to the ratio of the two requirements, $\frac{2d\ln(d/\beta)/(\alpha\varepsilon)}{d\ln 2/(\alpha\varepsilon)} = 2\ln(d/\beta)/\ln2$, a factor that grows only logarithmically in $d$.

**Approximate DP pays only $\sqrt d$.** The Gaussian mechanism's requirement grows like $\sqrt{d\ln d}$, because its noise is calibrated to the $\ell_2$ sensitivity $2\sqrt d/n$ rather than the $\ell_1$ sensitivity $2d/n$, and the quantile $z$ grows like $\sqrt{\ln d}$. For large $d$ it needs fewer records than packing says *any* pure-DP mechanism needs. There is no contradiction: packing breaks down once $\delta > 0$, as the previous section showed, because the additive $\delta_k$ of group privacy swamps the share $e^{-k\varepsilon}$ once $k$ passes about $\ln(1/\delta)/\varepsilon$. At $\delta = 10^{-6}$ packing over $2^d$ corners can only force a group distance of 13, just below $\ln(10^6) = 13.82$, whatever $d$ is, against $\ln(2^{1000}\times0.95) = 693.10$ for pure DP at $d = 1000$. A lower bound for approximate DP needs a different idea, which is the next section.

### Example: a survey with many questions

@tbl-lower-sample gives the numbers at $\varepsilon = 1$ and $\beta = 0.05$. A histogram over $N = 10^6$ cells to within $\alpha = 0.1$ needs at least $(13.76 - 1)/(2\times0.1\times1) = 63.8$, so 64 records under $\varepsilon$-DP, and at most 875 for sampling error; at $\alpha = 0.01$ the numbers are 638 and 87,522. Privacy is the smaller cost in both cases.

| | Pure DP, packing (lower) | Pure DP, Laplace (upper) | Approximate DP, Gaussian (upper) |
| --- | --- | --- | --- |
| $d = 100$ | 683 | 15,202 | 2,941 |
| $d = 1000$ | 6,921 | 198,070 | 10,836 |
| $d = 10{,}000$ | 69,304 | 2,441,215 | 38,570 |
| $d = 100{,}000$ | 693,137 | 29,017,315 | 134,299 |

: Records needed to release $d$ attribute means with $\ell_\infty$ error 0.1 (on the scale $[-1, 1]$) with probability 0.95, at $\varepsilon = 1$ and, for the Gaussian, $\delta = 10^{-6}$. {#tbl-lower-sample}

The packing column grows tenfold with each tenfold increase in $d$, the Laplace column a little faster, and the Gaussian column only 3.7-, 3.6- and 3.5-fold, close to $\sqrt{10}$, so the Gaussian column must eventually drop below the packing column. It does so from $d = 2731$ attributes on: from there, the Gaussian mechanism needs fewer records than any pure-DP mechanism can. A survey of a few thousand yes/no items, or a model with a few thousand parameters, is already past that point. This is one reason why private machine learning, where $d$ is the number of parameters, is done with the Gaussian mechanism and approximate DP.

### Check yourself

- A histogram over $N = 10^6$ cells at $\varepsilon = 0.1$ instead of 1: roughly how does the private lower bound on $n$ change, and does it now dominate the sampling term at $\alpha = 0.1$? *(By @eq-lower-n-histogram it becomes $(\ln(950{,}000) - 0.1)/(2\times0.1\times0.1) = (13.76 - 0.1)/0.02 = 683$, about ten times more. That is still below the sampling term of 875, but the two are now comparable.)*
- Why does the marginal lower bound @eq-lower-n-marginals not apply to the Gaussian mechanism? *(It uses the packing lemma, which needs pure DP. With $\delta > 0$ the $\delta_k$ term of group privacy swamps the $e^{-k\varepsilon}$ share once $k \gtrsim \ln(1/\delta)/\varepsilon$.)*

## Fingerprinting: the $\sqrt d$ lower bound for approximate DP {#sec-lower-fingerprinting}

### Context

Packing looks for many datasets far from one reference, and approximate DP defeats it because group privacy breaks down at large distances. Fingerprinting only ever compares neighbours, so $\delta$ does not blow up. It looks at one random dataset and asks what an accurate answer reveals about each person in it. Draw a dataset at random, release its $d$ attribute means accurately, and look at a single record. If the released means agree with that record more than with a stranger's record, the release has "fingerprinted" the person.

The argument has two halves that pull against each other. The **fingerprinting lemma** shows that accuracy *forces* such agreement: an estimate that follows the true means must follow the data, and the data are the records, so the release must correlate with the records, by a total amount of order $d$. DP, through Fact 2, says that each single record's correlation must be small, no larger than a stranger's plus a little. Since there are only $n$ records to carry the total, $n$ must be large.

The idea comes from fingerprinting codes in cryptography (Boneh & Shaw 1998; Tardos 2003), which trace the source of a pirated copy; Bun, Ullman & Vadhan (2014) brought it to DP. The version here, with a uniform prior on the attribute means, follows Dwork et al. (2015) and Steinke & Ullman (2016).

### Statement

**The random dataset.** Draw attribute means $p_1, \dots, p_d$ independently and uniformly from $[-1, 1]$. Given $p$, draw $n$ records independently, each coordinate $x_{ij} \in \{\pm1\}$ independently with mean $p_j$, that is, $+1$ with probability $(1 + p_j)/2$ and $-1$ otherwise.

**Fingerprinting lemma.** The lemma is about one attribute, so drop the index $j$. Let $x = (x_1, \dots, x_n)$ be the column of that attribute, $\bar x$ its mean, and $f : \{\pm1\}^n \to \mathbb R$ any estimate of $p$ computed from the column. The quantity on the left below is the correlation between the estimate and the records' deviations $x_i - p$ from the true mean, summed over the records. The lemma says that this correlation equals $2\,\mathbb E[p\,f(x)]$, which measures how well $f$ follows $p$, and that it cannot be small unless the estimate is inaccurate:
$$\mathbb E\Big[f(x)\sum_{i=1}^n(x_i - p)\Big] = 2\,\mathbb E\big[p\,f(x)\big] \ge \frac23 - 2\,\mathbb E\,|f(x) - \bar x|.$$ {#eq-lower-fp-lemma}
The expectations are over $p$ and the column.

**Lower bound.** Summing the lemma over the $d$ attributes gives a total correlation of order $d$, and Fact 2 caps each record's share at about $(e^\varepsilon - 1)\sqrt d + 4\delta d$. So $n$ times the cap must cover the total. Let $M$ be $(\varepsilon, \delta)$-DP under replace neighbours, with outputs in $[-1, 1]^d$, and with average error $\mathbb E\,\|M(D) - \bar x\|_1 \le \alpha' d$ on the random dataset, where $\alpha'$ is the average error per attribute. Then
$$n \ge \frac{(2/3 - 2\alpha')\,d}{(e^\varepsilon - 1)\sqrt d + 4\delta d}.$$ {#eq-lower-fp}
The bound uses an average error, but the guarantees of the previous sections are of the $(\alpha, \beta)$ kind. They convert: an $(\alpha, \beta)$-accurate mechanism in $\ell_\infty$ has $\alpha' \le \alpha + 2\beta$. To see this, first clip its output to $[-1, 1]^d$. The true means lie in that cube, so clipping moves the output closer to them and never increases the error. Then each coordinate is off by at most $\alpha$, except with probability $\beta$, when it is off by at most 2, the width of $[-1, 1]$. For $\delta \ll 1/\sqrt d$ and constant accuracy, the term $4\delta d$ is small next to $\sqrt d$, and the bound reads $n \gtrsim \sqrt d/(e^\varepsilon - 1)$.

### Derivation

**The lemma.** The plan is to show that the correlation on the left is exactly the slope of the average estimate as a function of $p$, weighted by $1 - p^2$, and then to average that slope over $p$. An estimate that is accurate must rise from about $-1$ at $p = -1$ to about $+1$ at $p = 1$, so its slope cannot be small on average. Fix $p$ and write $g(p) = \mathbb E[f(x) \mid p]$, the average estimate when the true mean is $p$.

1. **How the probability of the data responds to $p$.** The probability of the column $x$ is $\prod_i\frac{1 + x_ip}{2}$, because $\Pr[x_i = 1] = \frac{1 + p}2$ and $\Pr[x_i = -1] = \frac{1 - p}2$. Its log-derivative in $p$, called the **score** in statistics, says how strongly the data favour a larger $p$. It is $\sum_i\frac{x_i}{1 + x_ip}$. For $x_i = 1$ the term is $\frac1{1 + p} = \frac{1 - p}{1 - p^2}$, and for $x_i = -1$ it is $\frac{-1}{1 - p} = \frac{-1 - p}{1 - p^2}$. In both cases it is $\frac{x_i - p}{1 - p^2}$. So the score is the sum of the records' deviations from $p$, divided by $1 - p^2$.
2. **Differentiate the average estimate.** Write $g(p) = \sum_x f(x)\Pr[x \mid p]$ and differentiate each term. Since $\frac{d}{dp}\Pr[x \mid p] = \Pr[x \mid p]\,\frac{d}{dp}\ln\Pr[x \mid p]$, this gives $g'(p) = \mathbb E\big[f(x)\,\frac{d}{dp}\ln\Pr[x \mid p]\ \big|\ p\big]$. Inserting the score of step 1 and multiplying by $1 - p^2$,
$$(1 - p^2)\,g'(p) = \mathbb E\Big[f(x)\sum_i(x_i - p)\ \Big|\ p\Big].$$
In words: an estimate that follows $p$ ($g' > 0$) must be correlated with the deviations $x_i - p$ of the records, because the deviations are the only way the data carry information about $p$.
3. **Average over $p$ and integrate by parts.** The density of $p$ is $\frac12$ on $[-1, 1]$, so averaging step 2 over $p$ gives $\frac12\int_{-1}^1(1 - p^2)g'(p)\,dp$. Integration by parts moves the derivative from $g$ to $1 - p^2$, whose derivative is $-2p$:
$$\mathbb E\Big[f\sum_i(x_i - p)\Big] = \frac12\int_{-1}^1(1 - p^2)g'(p)\,dp = \frac12\Big[(1 - p^2)g(p)\Big]_{-1}^1 + \frac12\int_{-1}^12p\,g(p)\,dp = 2\,\mathbb E[p\,g(p)].$$
The boundary term vanishes because $1 - p^2 = 0$ at $\pm1$ and $g$ is finite there: each $\Pr[x \mid p]$ is a polynomial in $p$, so $g$ is a polynomial. The last equality holds because $\frac12\int_{-1}^12p\,g(p)\,dp = \int_{-1}^1p\,g(p)\,dp$ is twice the average $\mathbb E[p\,g(p)] = \frac12\int_{-1}^1p\,g(p)\,dp$. Finally, $g(p)$ is the average of $f$ given $p$, so $\mathbb E[p\,g(p)] = \mathbb E[p\,f(x)]$. This is the equality in @eq-lower-fp-lemma.
4. **Compare with the sample mean.** The sample mean is the benchmark accurate estimate. Its average given $p$ is $\mathbb E[\bar x \mid p] = p$, so $\mathbb E[p\,\bar x] = \mathbb E[p^2] = \frac12\int_{-1}^1p^2\,dp = \frac13$. Any other estimate differs from it by $f - \bar x$, so $\mathbb E[p\,f] = \frac13 + \mathbb E[p(f - \bar x)] \ge \frac13 - \mathbb E|f - \bar x|$, because $|p| \le 1$. Multiply by 2.

**The lower bound.** The proof is an attack. Score each record $x_i$ by how well it agrees with the release, measured as a correlation with the record's deviations from the true means:
$$Z_i = \big\langle M(D),\ x_i - p\big\rangle = \sum_{j=1}^dM(D)_j\,(x_{ij} - p_j).$$
A member should score high, because the release follows the data. A stranger's record, independent of the release, should score around 0. The four steps make this precise and play the two facts against each other.

1. **Accuracy makes the scores large on average.** Fix an attribute $j$. Condition on the other columns and on the other means $p_{-j}$. Columns are independent given $p$, so column $j$ and $p_j$ still have the distribution of the lemma. Let $f$ be the average of $M(D)_j$ over the mechanism's random coins: with everything else fixed, it is a function of column $j$ alone. The lemma gives $\mathbb E\big[M(D)_j\sum_i(x_{ij} - p_j)\big] \ge \frac23 - 2\,\mathbb E|M(D)_j - \bar x_j|$. Two remarks justify this. The coins are independent of the data, so averaging $M(D)_j$ over them first does not change the left side. And Jensen's inequality, here simply the fact that the absolute value of an average is at most the average of the absolute values, gives $|f - \bar x_j| \le \mathbb E_{\text{coins}}|M(D)_j - \bar x_j|$. Now sum over $j$. On the left, swapping the sums over $j$ and $i$ gives $\sum_i\mathbb E Z_i$. On the right, the errors add up to the $\ell_1$ error:
$$\sum_{i=1}^n\mathbb E Z_i \ge \frac23d - 2\,\mathbb E\|M(D) - \bar x\|_1 \ge \Big(\frac23 - 2\alpha'\Big)d.$$
2. **A stranger's score is small.** Let $D^{(i)}$ be $D$ with record $i$ replaced by a fresh record from the same distribution, and $Z_i' = \langle M(D^{(i)}), x_i - p\rangle$ the score of $x_i$ against a release that never saw it. Given $p$, $x_i$ is independent of $M(D^{(i)})$ and each deviation $x_{ij} - p_j$ has mean 0, so $\mathbb E Z_i' = 0$. For the spread, expand $Z_i'^2$ as a double sum over coordinates $j, k$. Given $p$ and the release, the deviations $x_{ij} - p_j$ are independent with mean 0 and variance $1 - p_j^2 \le 1$, so the cross terms ($j \ne k$) average to 0 and the diagonal terms give $\mathbb E Z_i'^2 \le \sum_j\mathbb E\,M(D^{(i)})_j^2 \le d$, because each output coordinate lies in $[-1, 1]$. By the Cauchy–Schwarz inequality, $\mathbb E|Z_i'| \le \sqrt{\mathbb E Z_i'^2} \le \sqrt d$.
3. **DP keeps a member's score close to a stranger's.** $D$ and $D^{(i)}$ differ in exactly record $i$, so they are neighbours under replace. Fix the data and apply Fact 2 to the score $h(y) = \langle y, x_i - p\rangle$ of an output $y$. It is bounded: $|y_j| \le 1$ and $|x_{ij} - p_j| \le 2$, so $|h| \le \|x_i - p\|_1 \le 2d$, which makes $T = 2d$ and $2\delta T = 4\delta d$. Fact 2 then says that the member's score $Z_i$ exceeds the stranger's $Z_i'$ on average by at most $(e^\varepsilon - 1)\mathbb E|Z_i'| + 4\delta d$. Averaging over the data and inserting step 2: $\mathbb E Z_i \le 0 + (e^\varepsilon - 1)\sqrt d + 4\delta d$.
4. **Combine.** Summing step 3 over the $n$ records bounds the total from above, and step 1 bounds it from below: $(\frac23 - 2\alpha')d \le n\big((e^\varepsilon - 1)\sqrt d + 4\delta d\big)$. Dividing by the bracket gives @eq-lower-fp.

@fig-lower-fingerprinting shows the lemma at work for $n = 5$. The sample mean has $g(p) = p$, so $g' = 1$ and its correlation with the records is $(1 - p^2)$ at each $p$, averaging $1 - \frac13 = \frac23$. Shrinking the estimate toward 0, to half the sample mean, halves $g'$ and so halves the correlation, and a constant estimate has $g' = 0$ and no correlation. But those estimates pay in error, exactly as @eq-lower-fp-lemma says.

![The fingerprinting lemma for one attribute and $n = 5$ records. Left: the average estimate $g(p)$ for the sample mean, half the sample mean, and the constant 0. Right: the correlation of each estimate with the records' deviations, $(1 - p^2)g'(p)$. Averaged over $p$ it is $\frac23$ for the mean; an estimate can only lower it by being less accurate.](figures/lower_fingerprinting.svg){#fig-lower-fingerprinting}

### Interpretation

**The $\sqrt d$ is the gap between signal and noise.** Step 1 says the total correlation is of order $d$, shared among $n$ records, so a typical member has a score of order $d/n$. Step 2 says a stranger's score has spread of order $\sqrt d$. DP allows a member's score to exceed a stranger's by only a constant multiple of that spread. So $d/n \lesssim \sqrt d$, which is $n \gtrsim \sqrt d$. With pure DP the same bound holds with $\delta = 0$, but packing gives the stronger $d$.

**The role of $\delta$.** The term $4\delta d$ is harmless only while $\delta \ll 1/\sqrt d$, so that it stays small next to $\sqrt d$. It cannot be removed. A mechanism that publishes each record outright with probability $\delta$ satisfies $(0, \delta)$-DP, because one record's presence changes any event's probability by at most $\delta$ ([The definition of differential privacy](01-definition.md)). If $\delta$ were as large as $1/n$, it would publish about one record on average, and no lower bound of this kind could hold.

**How tight is it?** The $\sqrt d$ rate matches the Gaussian mechanism, but the constants in @eq-lower-fp are poor. At $\alpha = 0.1$, $\beta = 0.05$, $\varepsilon = 1$, $\delta = 10^{-6}$ the bound says $n \ge 5.0$ for $d = 1000$ and $n \ge 15.8$ for $d = 10{,}000$, against 10,836 and 38,570 records for the Gaussian mechanism (@fig-lower-sample-complexity). Three things separate the simple proof from the Gaussian rate. The first is a factor $1/\alpha$: [Exercise 4](../../book-exercises/part03-dp-foundations/10-lower-bounds.md#sec-sol-lower-4) recovers it with a padding argument, and the numbers just quoted already include it. The second is a factor $\sqrt{\ln(1/\delta)}$, which Steinke & Ullman (2016) obtain with a sharper choice of the distribution of $p$, proving $n = \Omega\big(\sqrt{d\ln(1/\delta)}/(\alpha\varepsilon)\big)$. The third is the $\sqrt{\ln d}$ from the union bound in the upper bound, which is a property of $\ell_\infty$ error, not of privacy. With all three, upper and lower bounds agree up to that logarithm.

![Records needed to release $d$ attribute means to $\ell_\infty$ error 0.1 with probability 0.95, at $\varepsilon = 1$ and $\delta = 10^{-6}$, on log–log axes. Orange: pure DP (Laplace upper bound, packing lower bound), both of slope 1. Blue: approximate DP (Gaussian upper bound, fingerprinting lower bound), both of slope ½. The constants of the fingerprinting bound are loose; the slopes are the point.](figures/lower_sample_complexity.svg){#fig-lower-sample-complexity}

### Example: the lemma, computed exactly

For $n = 5$ the script sums over all $2^5 = 32$ possible columns and integrates over $p$ exactly, so the numbers below have no simulation error. For the sample mean, the left side of @eq-lower-fp-lemma is 0.6667, equal to $2\mathbb E[pf]$, and the bound is also 0.6667, because the error term $\mathbb E|f - \bar x|$ is 0: no slack. For half the sample mean, the correlation is half as large, 0.3333, and its error is $\mathbb E|f - \bar x| = \frac12\mathbb E|\bar x| = 0.3000$, so the bound is $\frac23 - 0.6 = 0.0667$, well below the truth. For $f = 0$ the correlation is 0, the error is $\mathbb E|\bar x| = 0.6$, and the bound, $\frac23 - 1.2 = -0.5333$, is vacuous. The lemma is tight for unbiased estimators and loose for biased ones. A biased estimator $c\,\bar x$ loses correlation in proportion to its bias (its correlation is $2c/3$), while the bound charges it twice its absolute error.

### Check yourself

- Why does the proof draw the means $p_j$ at random instead of fixing them? *(For a fixed $p$, the constant estimate $f = p$ is accurate and uncorrelated with every record, so there is no lower bound. The mechanism must be accurate for every $p$ at once, and averaging over $p$ is what turns "follows $p$" into "correlates with the records" via the integration by parts.)*
- In step 3, why is the stranger's dataset $D^{(i)}$ a neighbour of $D$? *(It differs from $D$ in exactly record $i$, which is the replace relation. That is why this section uses replace neighbours.)*

## Tracing attacks: the lower bound is a membership inference attack {#sec-lower-tracing}

### Context

In 2008 Homer et al. showed that from the published allele frequencies of a genome-wide association study (summary statistics of a few hundred people over hundreds of thousands of single-nucleotide polymorphisms) you could tell whether a given person's DNA was in the study. The US National Institutes of Health took such summaries off public access the same year. The test they used is, up to details, the score $Z_i$ of the fingerprinting proof: compare the person's genome with the published frequencies, relative to the population frequencies, and see whether they agree more than chance would explain. This section runs that attack on synthetic data, computes its power, and compares it with what DP allows. Dwork et al. (2015) proved the attack is **robust** in a precise sense: it still works when the released means are noisy, and when the attacker knows the population only through a small reference sample.

### Statement

The attacker sees the release $\hat q = \bar x + \eta$, the true means with Gaussian noise $\eta \sim \mathcal N(0, \sigma^2I_d)$ of standard deviation $\sigma$ in each coordinate. It also has a target record $y$ and the population means $p$. It scores the target by the correlation of the release with the target's deviations from the population, $Z(y) = \langle\hat q, y - p\rangle$, and says "member" when the score is large. Each member pulls the release slightly toward its own record, by $1/n$ of its deviation, so a member's score is shifted up; a non-member's record is independent of the release, so its score is centred at 0. With the random dataset of the previous section, the shift and the spread of the scores are
$$\mathbb E[Z \mid \text{member}] - \mathbb E[Z \mid \text{non-member}] = \frac{2d}{3n}, \qquad \operatorname{Var}[Z \mid \text{non-member}] = d\left(\frac2{15} + \frac{8}{15n} + \frac{2\sigma^2}3\right).$$ {#eq-lower-tracing}
How well the attack works depends on the shift measured in units of the spread. If the score is approximately normal with this mean gap $m$ and standard deviation $s$ under both hypotheses, the best threshold has advantage $\mathrm{TPR} - \mathrm{FPR} = 2\Phi\big(m/(2s)\big) - 1$, where $\Phi$ is the standard normal cumulative distribution function.

### Derivation

All averages are over $p$ uniform on $[-1, 1]$, with density $\frac12$, where $\mathbb E[p^2] = \frac12\int_{-1}^1p^2\,dp = \frac13$, $\mathbb E[p^4] = \frac12\int_{-1}^1p^4\,dp = \frac15$ and so $\mathbb E[(1 - p^2)^2] = 1 - \frac23 + \frac15 = \frac8{15}$. Two facts about a $\pm1$ variable with mean $p$ are used throughout: its square is always 1, and its variance is $1 - p^2$.

1. **Non-member mean.** A fresh $y$ is independent of $\hat q$ given $p$, and $\mathbb E[y_j - p_j \mid p] = 0$, so each coordinate of $Z$ averages to 0 and $\mathbb E Z = 0$.
2. **Member mean.** For $y = x_i$, coordinate $j$ contributes $\mathbb E[\bar x_j(x_{ij} - p_j)]$; the noise is independent of everything and averages out. Write $\bar x_j = \frac1n\sum_kx_{kj}$. The terms $k \ne i$ are other people's records, independent of $x_{ij}$, and contribute 0. The term $k = i$ is the member's own record inside the mean, and contributes $\frac1n\mathbb E[x_{ij}(x_{ij} - p_j)] = \frac1n\mathbb E[1 - x_{ij}p_j] = \frac1n(1 - p_j^2)$, because $x_{ij}^2 = 1$ and $\mathbb E[x_{ij}] = p_j$. Averaging over $p_j$ gives $\frac1n\cdot\frac23$, and summing over the $d$ coordinates gives $\frac{2d}{3n}$.
3. **Non-member variance.** The mean is 0, so the variance is $\mathbb E Z^2$. The coordinates $y_j - p_j$ are independent of each other and of $\hat q$, with mean 0, so in the double sum $Z^2 = \sum_{j,k}\hat q_j\hat q_k(y_j - p_j)(y_k - p_k)$ the cross terms vanish and $\operatorname{Var}Z = \sum_j\mathbb E\big[\hat q_j^2(1 - p_j^2)\big]$. Given $p$, the mean square of $\hat q_j$ is its squared mean plus its variance: $\mathbb E[\hat q_j^2] = p_j^2 + \frac{1 - p_j^2}n + \sigma^2$ (squared mean, sampling variance of an average of $n$ records, noise variance). Multiply by $1 - p_j^2$ and average over $p_j$: $\mathbb E[p^2 - p^4] + \frac1n\mathbb E[(1 - p^2)^2] + \sigma^2\mathbb E[1 - p^2] = (\frac13 - \frac15) + \frac8{15n} + \frac{2\sigma^2}3 = \frac2{15} + \frac8{15n} + \frac{2\sigma^2}3$ per coordinate.
4. **Advantage of a threshold.** For $\mathcal N(0, s^2)$ against $\mathcal N(m, s^2)$, the two densities cross at the midpoint $m/2$, and by the Neyman–Pearson lemma (the best test thresholds the ratio of the two densities, which here is increasing in the score) the best threshold is that crossing point. The test fires above $m/2$. A member's score exceeds it with probability $\Phi(\frac m{2s})$, since $m/2$ is $\frac m{2s}$ standard deviations below the member mean, and a non-member's with probability $\Phi(-\frac m{2s})$, by the mirror argument. So $\mathrm{TPR} - \mathrm{FPR} = \Phi(\frac m{2s}) - \Phi(-\frac m{2s}) = 2\Phi(\frac m{2s}) - 1$, using $\Phi(-t) = 1 - \Phi(t)$. This is the total variation of the two normal distributions, as in [Hypothesis testing](../part01-prerequisites/04-hypothesis-testing.md), where the best advantage of any test is shown to equal the TV distance.

### Interpretation

**Tracing needs $d \gtrsim n^2$, and noise of order $\sqrt d/n$ stops it.** Without noise, and ignoring the small $8/(15n)$ term, the signal-to-noise ratio is $m/s \approx \frac{2d}{3n}\big/\sqrt{2d/15} = \frac{2\sqrt d}{3n}\big/\sqrt{2/15}$. The attack is strong once this ratio is a few units, that is, once $\sqrt d$ is a few times $n$: once there are many more attributes than the square of the number of people. Noise adds $2\sigma^2/3$ to the variance per coordinate, and dominates the sampling term $\frac2{15}$ once $\frac{2\sigma^2}3 > \frac2{15}$, that is, $\sigma \gtrsim 1/\sqrt{5}$. The ratio then behaves like $\frac{2d}{3n}\big/\sqrt{2d\sigma^2/3}$, which is of order $\sqrt d/(n\sigma)$. So the attack fails when $\sigma \gg \sqrt d/n$. The Gaussian mechanism adds $\sigma = \Delta_2/\mu = 2\sqrt d/(n\mu)$: the same order, as it must be for a mechanism that is both private and as accurate as fingerprinting allows.

**DP bounds the attack for each target.** The member-versus-stranger game is a test between $M(D)$ and $M(D^{(i)})$, neighbours under replace. Here the member's record is in $D$ rather than in the second dataset, but Fact 3 does not care which of two neighbours plays which role, because the definition holds for both orders of the pair. So by Fact 3 its advantage is at most $(e^\varepsilon - 1 + 2\delta)/(e^\varepsilon + 1)$. For the Gaussian mechanism there is a sharper bound. Its outputs on the two neighbours are Gaussians with the same $\sigma$ whose means differ by at most $\Delta_2$, and step 4 with $m = \Delta_2$ and $s = \sigma$ gives their TV distance, at most $2\Phi\big(\Delta_2/(2\sigma)\big) - 1$.

**This is the membership inference of Part 2, on statistics instead of models.** The score is a likelihood-ratio-style statistic, and, as in [Membership inference](../part02-attacks/02-membership-inference.md), the honest way to report it is the TPR at a low FPR, because an attack that identifies a few people with confidence is a privacy failure even if its average advantage is small. Released aggregates are not safe because they are aggregates: what protects members is the ratio of $n$ to $\sqrt d$, or noise.

### Example: 50 people, 2000 attributes

The script draws 200 datasets of $n = 50$ records with $d = 2000$ attributes, and scores the 50 members and 50 fresh non-members of each. Without noise, @eq-lower-tracing predicts a mean gap of $2\times2000/(3\times50) = 26.67$ and a spread of $\sqrt{2000(\frac2{15} + \frac8{750})} = 16.97$; the simulation gives 26.70 and 16.92. The normal approximation then gives the advantage $2\Phi\big(26.67/(2\times16.97)\big) - 1 = 2\Phi(0.786) - 1 = 0.568$. @tbl-lower-tracing shows the attack at three noise levels. The (1, $10^{-5}$)-DP Gaussian mechanism on these means has $\sigma = 6.67$.

| Noise $\sigma$ | Advantage (simulated) | Advantage (normal approximation) | Worst-case TV bound | TPR at 1% FPR |
| --- | --- | --- | --- | --- |
| 0 | 0.578 | 0.568 | 1.000 | 0.242 |
| 1.00 | 0.264 | 0.259 | 0.629 | 0.052 |
| 6.67 (DP, $\varepsilon = 1$) | 0.049 | 0.044 | 0.107 | 0.014 |

: The tracing attack on released attribute means, $n = 50$, $d = 2000$. The advantage is the best $\mathrm{TPR} - \mathrm{FPR}$ over thresholds. {#tbl-lower-tracing}

The worst-case TV column is $2\Phi\big(\Delta_2/(2\sigma)\big) - 1$ with $\Delta_2 = 2\sqrt{2000}/50 = 1.79$; without noise it is 1, because in the worst case the exact means of two neighbours differ, and an exact release then tells them apart perfectly. Without noise, a quarter of the members are identified at a 1% false-alarm rate, from means of 50 people. With $(1, 10^{-5})$-DP noise the advantage is 0.049, under both the Gaussian TV bound of 0.107 and the generic DP bound of Fact 3, 0.462. The TPR at 1% FPR is 0.014. DP bounds it too: applied to the event "the test fires", the definition says $\mathrm{TPR} \le e^\varepsilon\,\mathrm{FPR} + \delta = e^1\times0.01 + 10^{-5} = 0.0272$, and 0.014 is under it. The simulated advantage sits slightly above the normal approximation, because it takes the best of many thresholds on a finite sample, which picks up some luck.

@fig-lower-tracing-scores shows the score distributions and @fig-lower-tracing-advantage the whole curve. The attack fades out around $\sigma = \sqrt d/n = \sqrt{2000}/50 = 0.89$ and is almost gone at the DP noise level.

![Tracing scores of members (blue) and non-members (orange) for $n = 50$ records and $d = 2000$ attributes. Left: exact means; the members' scores are shifted by $2d/(3n) = 26.7$. Right: means released with the Gaussian noise of a $(1, 10^{-5})$-DP mechanism; the two distributions nearly coincide.](figures/lower_tracing_scores.svg){#fig-lower-tracing-scores}

![Advantage of the tracing attack against the noise standard deviation, $n = 50$, $d = 2000$. Dots: simulation; solid line: the normal approximation from @eq-lower-tracing; dashed: the TV distance between the outputs on neighbouring datasets in the worst case, which bounds every attack. The attack fails once $\sigma$ is well above $\sqrt d/n$; the DP Gaussian mechanism adds $\sigma = 6.7$.](figures/lower_tracing_advantage.svg){#fig-lower-tracing-advantage}

### Check yourself

- If you double both $n$ and $\sqrt d$, what happens to the tracing advantage without noise? *(Almost nothing: $m/s \approx \frac{2\sqrt d}{3n}/\sqrt{2/15}$ depends on $\sqrt d/n$, apart from the small $8/(15n)$ term.)*
- The attacker needs $p$. Why is it not enough to replace $p$ by the released $\hat q$ itself? *(Then the score is $\langle\hat q, y - \hat q\rangle$, and the member's own record is part of $\hat q$, so the subtraction cancels part of the signal. A separate reference sample keeps the mean gap $2d/(3n)$, at the cost of extra variance; see [Exercise 6](../../book-exercises/part03-dp-foundations/10-lower-bounds.md#sec-sol-lower-6).)*

## Dinur–Nissim as a lower bound {#sec-lower-dinur-nissim}

### Context

[Reconstruction attacks](../part02-attacks/09-reconstruction.md) showed that a database of $n$ secret bits can be almost entirely reconstructed from a number of random subset sums only polylogarithmically larger than $n$ (Dinur and Nissim used $n(\log n)^2$; later analyses need only $O(n)$), answered with error $o(\sqrt n)$, that is, error that grows more slowly than $\sqrt n$. Each query picks a random subset of the people and asks how many of them have bit 1. The attacker then finds the database of bits that best fits all the noisy answers, for example by least squares. If the errors are much smaller than $\sqrt n$, any database that fits them well must agree with the true one on almost every bit. There it was an attack. Here it is a lower bound: since DP limits how well any single bit can be guessed (Fact 3), a release from which most bits can be reconstructed cannot be private. Every mechanism that answers that many counting queries with $(\varepsilon, \delta)$-DP must therefore add error of order $\sqrt n$ to its answers. The Gaussian mechanism adds exactly that order, so the bound is tight.

### Statement

Let the database be $n$ independent uniform bits, and call two databases neighbours when they differ in one bit. Fact 3 says each bit can be guessed correctly with probability at most $(e^\varepsilon + \delta)/(e^\varepsilon + 1)$, and the fraction of bits recovered is an average of such guesses, so it obeys the same limit. Read the other way, recovering a fraction $r$ proves a lower bound on $\varepsilon$. If $M$ is $(\varepsilon, \delta)$-DP, then every reconstruction procedure applied to $M$'s output recovers, in expectation, a fraction of the bits of at most
$$r \le \frac{e^\varepsilon + \delta}{e^\varepsilon + 1}, \qquad\text{so}\qquad \varepsilon \ge \ln\frac{r - \delta}{1 - r}.$$ {#eq-lower-dn}
By the Dinur–Nissim theorem, in the sharper form of Dwork, McSherry and Talwar (2007) that needs only $k = O(n)$ queries, answering $k$ random subset-sum queries with error $o(\sqrt n)$ allows a reconstruction with $r \to 1$. The right side of @eq-lower-dn then grows without limit, so no such mechanism is $(\varepsilon, \delta)$-DP for any fixed $\varepsilon$ and $\delta < 1/2$. Error $\Omega(\sqrt n)$, at least a constant times $\sqrt n$, is necessary.

### Derivation

1. **One bit at a time.** Fix $i$ and all bits except $x_i$. Since the bits are independent, $x_i$ is still uniform. The two values of $x_i$ give databases that differ in one bit, which are neighbours. The reconstruction's guess of $x_i$ is computed from $M$'s output alone (a post-processing), so it is a test between the two neighbours, and by Fact 3 it is right with probability at most $(e^\varepsilon + \delta)/(e^\varepsilon + 1)$. Averaging over the other bits keeps the bound.
2. **Average over the bits.** The expected fraction recovered is $\frac1n\sum_i\Pr[\text{bit } i \text{ is guessed right}]$, by linearity of expectation, and each term obeys the bound of step 1, so the average does too. To solve $r \le (e^\varepsilon + \delta)/(e^\varepsilon + 1)$ for $\varepsilon$, multiply by $e^\varepsilon + 1$: $re^\varepsilon + r \le e^\varepsilon + \delta$, that is $e^\varepsilon(1 - r) \ge r - \delta$. Dividing by $1 - r$ and taking logarithms gives the second form of @eq-lower-dn.
3. **Insert the attack.** If the errors are $o(\sqrt n)$, the Dinur–Nissim decoder misses $o(n)$ bits, so $r \to 1$, the denominator $1 - r$ goes to 0, and the bound forces $\varepsilon \to \infty$.

### Interpretation

**The Gaussian mechanism matches it.** Changing one bit changes by 1 the answer of every query whose subset contains that person, and leaves the others alone. At most $k$ answers change, so the answer vector moves by at most $\sqrt k$ in $\ell_2$ (in fact by about $\sqrt{k/2}$, since each person is in about half of the random subsets). The Gaussian mechanism therefore needs $\sigma$ proportional to $\sqrt k/\mu(\varepsilon, \delta)$, which for $k = 4n$ is $2\sqrt n/\mu$, of order $\sqrt n$. Answering $k$ counting queries with independent noise costs error of order $\sqrt k$ per query under approximate DP, as the marginals section found for $d$ means. Under pure DP the Laplace mechanism must be calibrated to the $\ell_1$ sensitivity, which is $k$, and so pays error of order $k$ per query, as packing says it must for families of queries as rich as the marginals.

For random subset sums, these two rates are lower bounds only up to about $n$ queries. With more queries than records they stop being the right picture, for two reasons. An error of order $n$ says nothing, because answering $n/2$ to every query already achieves it, and the example below, with $k = 4n$, sits at that edge. More importantly, a mechanism need not noise each answer separately. Mechanisms that build a small private synthetic dataset and answer every query from it, such as SmallDB and private multiplicative weights ([Query release](06-query-release.md)), have per-query errors that grow only like a small power of $\log k$, times factors in $n$ and in the size of the data universe. For random subset sums under approximate DP, private multiplicative weights keeps the error of order $\sqrt n$ up to logarithmic factors for any number of queries polynomial in $n$, which is the Dinur–Nissim bound of this section. Other query families behave differently. For the one-way marginals, whose universe is far larger, the $\sqrt d$ rate of approximate DP stays optimal up to about $n^2$ queries.

**The bound is crude in one way.** It looks only at the fraction of bits recovered, and least squares is not the best attack. So the $\varepsilon$ it implies is far below the $\varepsilon$ the Gaussian mechanism guarantees at the same noise. The true privacy loss of the release lies somewhere between them. Auditing, in a later part, narrows this gap with stronger attacks.

### Example: 200 bits, 800 queries

The script answers $k = 800$ random subset sums of $n = 200$ uniform bits with Gaussian noise $\sigma = c\sqrt n$, decodes by least squares and rounds at $\frac12$, over 40 random databases. @tbl-lower-dn gives the fraction recovered, the $\varepsilon$ it implies by @eq-lower-dn at $\delta = 10^{-6}$, and the $\varepsilon$ that the Gaussian mechanism with that noise guarantees (with $\ell_2$ sensitivity $\sqrt k$, so $\mu = \sqrt k/\sigma = \sqrt{800}/(c\sqrt{200}) = 2/c$). For instance, recovering 0.9607 of the bits proves $\varepsilon \ge \ln\big((0.9607 - 10^{-6})/(1 - 0.9607)\big) = 3.20$.

| $\sigma/\sqrt n$ | Fraction recovered | $\varepsilon$ at least (attack) | $\varepsilon$ at most (Gaussian mechanism) |
| --- | --- | --- | --- |
| 0.1 | 1.0000 | $\infty$ | 294.17 |
| 0.25 | 0.9607 | 3.20 | 69.24 |
| 0.5 | 0.8058 | 1.42 | 26.36 |
| 1 | 0.6710 | 0.71 | 11.00 |
| 2 | 0.5810 | 0.33 | 4.89 |

: Reconstruction from 800 noisy subset sums of 200 bits, and the privacy loss it proves, at $\delta = 10^{-6}$. {#tbl-lower-dn}

At $\sigma = 0.1\sqrt n = 0.1\times\sqrt{200} = 1.4$ every bit is recovered, so $r = 1$ and the release is not $(\varepsilon, 10^{-6})$-DP for any finite $\varepsilon$. A mechanism that wants $\varepsilon = 1$ can let an attacker recover at most $e/(e + 1) = 73.11\%$ of the bits (Fact 3). In the table, 0.8058 at $0.5\sqrt n$ is above that and 0.6710 at $\sqrt n$ is below it, so such a mechanism needs noise between $0.5\sqrt n$ and $\sqrt n$ per answer. @fig-lower-dinur-nissim shows the whole range.

![Left: fraction of 200 secret bits recovered by least squares from 800 subset sums with Gaussian noise of standard deviation $\sigma$, against $\sigma/\sqrt n$. The dotted line is the most that any $\varepsilon = 1$ mechanism allows. Right: the $\varepsilon$ that the reconstruction proves (blue) and the $\varepsilon$ that the Gaussian mechanism guarantees (orange), at $\delta = 10^{-6}$; the true privacy loss of the release lies between them.](figures/lower_dinur_nissim.svg){#fig-lower-dinur-nissim}

### Check yourself

- Why does @eq-lower-dn need the secret bits to be uniform? *(Fact 3's bit-guessing bound is for a uniform prior. With a biased prior, guessing the more likely value already beats $\frac12$, and the bound must start from the prior instead.)*
- The attack recovers 67% of the bits at $\sigma = \sqrt n$. Does that mean the release is $(0.71, 10^{-6})$-DP? *(No. The attack proves only $\varepsilon \ge 0.71$. A lower bound from one attack is never a privacy guarantee; the guarantee comes from the mechanism's analysis, here $\varepsilon \le 11.00$.)*

## Code

The script [code/lower_bounds.py](../../code/part03-dp-foundations/lower_bounds.py) computes every number in this chapter and draws every figure. Each bound and each mechanism's error is a small function. The simulations check the Laplace histogram and report-noisy-max against their formulas, the fingerprinting lemma by exact summation, the tracing attack against @eq-lower-tracing, and the reconstruction attack against @eq-lower-dn. From the repository root:

```bash
uv run python code/part03-dp-foundations/lower_bounds.py
```

**Output** (seed 0), summarized in @tbl-lower-code:

| Experiment | Result |
| --- | --- |
| Toolkit, $\varepsilon = 1$ | Advantage at most 0.4621; bit guessed at most 0.7311 of the time; group $\delta_5 = 8.6\times10^{-5}$, $\delta_{20} = 282$ at $\delta = 10^{-6}$ |
| Mode and histogram, $N = 10^6$ | Lead $k \ge 13.76$ (noisy-max needs 18.44); error $\ge 6.38$ (Laplace 16.79); full table in @tbl-lower-histogram |
| Simulations at $N = 1000$ | Laplace 95% error 9.84 (exact 9.88); noisy-max failure 0.0228 at lead 11.09 (union bound 0.05) |
| Packing under $\delta = 10^{-6}$, $2^d$ candidates | Forces $k \ge 7$ at $d = 10$ and $k \ge 13$ at $d = 100$ and $1000$ (pure: 6.88, 69.26, 693.10); @eq-lower-packing-approx: 7, 11, 11; $\ln(1/\delta)/\varepsilon = 13.82$ |
| Packing for a mean, $\alpha = 0.1$ | $n \ge 4.55$, 9.16, 16.07 for $R = 10$, $10^3$, $10^6$; under $\delta = 10^{-6}$: 4, 9, 11 |
| Sample complexity, histogram $N = 10^6$ | $\alpha = 0.1$: 64 private, 875 sampling. $\alpha = 0.01$: 638 private, 87,522 sampling |
| Sample complexity, marginals | @tbl-lower-sample; the Gaussian beats the pure-DP lower bound from $d = 2731$; $\mu(1, 10^{-6}) = 0.2367$ |
| Fingerprinting lower bound | $n \ge 1.6$, 5.0, 15.8, 50.1 at $d = 10^2, 10^3, 10^4, 10^5$ |
| Fingerprinting lemma, $n = 5$ | Mean: 0.6667 (bound 0.6667). Half mean: 0.3333 (bound 0.0667). Zero: 0 (bound $-0.5333$) |
| Tracing, $n = 50$, $d = 2000$ | Gap 26.70 (predicted 26.67), spread 16.92 (16.97); @tbl-lower-tracing |
| Dinur–Nissim, $n = 200$, $k = 800$ | @tbl-lower-dn |

: The script's output with the default seed. {#tbl-lower-code}

[tests/test_lower_bounds.py](../../tests/test_lower_bounds.py) checks that the output still matches.

**What to take from it.** Packing is nearly tight under pure DP: the lower bounds and the Laplace and report-noisy-max mechanisms all grow like $\ln N/\varepsilon$, within a factor of about 2. Allowing $\delta > 0$ changes the rate, not just the constant. Packing cannot force more than $\ln(1/\delta)/\varepsilon$ under approximate DP, because group privacy degrades past that distance, and so the Gaussian mechanism overtakes every pure-DP mechanism from a few thousand dimensions on. The $\sqrt d$ lower bound that replaces packing is an attack you can run: the fingerprinting score identifies members of a 50-person dataset from its means, its power matches the formula, and DP noise of order $\sqrt d/n$ defeats it. Finally, lower bounds from attacks are lower bounds on $\varepsilon$. The reconstruction attack proves a release is not private below some $\varepsilon$; it never proves that it is.

## References

- Dwork & Roth, *The Algorithmic Foundations of Differential Privacy* (2014), ch. 8: packing lower bounds and reconstruction attacks as lower bounds, at textbook depth.
- Vadhan, "The Complexity of Differential Privacy" (2017), §5: packing, fingerprinting and reconstruction lower bounds side by side; also the stability-based histogram for approximate DP.
- Hardt & Talwar, "On the Geometry of Differential Privacy" (STOC 2010): packing generalized to any set of linear queries through volume arguments, with matching upper bounds for pure DP.
- Beimel, Brenner, Kasiviswanathan & Nissim, "Bounds on the Sample Complexity for Private Learning and Private Data Release" (Machine Learning, 2014): the sample-complexity view of packing.
- Bun, Ullman & Vadhan, "Fingerprinting Codes and the Price of Approximate Differential Privacy" (STOC 2014): the first $\sqrt d$ lower bound for approximate DP, through fingerprinting codes.
- Steinke & Ullman, "Between Pure and Approximate Differential Privacy" (Journal of Privacy and Confidentiality, 2016): the simplified fingerprinting lemma and the tight bound $\Omega(\sqrt{d\ln(1/\delta)}/(\alpha\varepsilon))$ for one-way marginals.
- Dwork, Smith, Steinke, Ullman & Vadhan, "Robust Traceability from Trace Amounts" (FOCS 2015): tracing attacks that tolerate noisy statistics and estimated population means.
- Dwork, Smith, Steinke & Ullman, "Exposed! A Survey of Attacks on Private Data" (Annual Review of Statistics and Its Application, 2017): reconstruction and tracing attacks for a statistics audience.
- Homer et al., "Resolving Individuals Contributing Trace Amounts of DNA to Highly Complex Mixtures Using High-Density SNP Genotyping Microarrays" (PLoS Genetics, 2008): the tracing attack on genomic summary statistics.
- Kamath & Ullman, "A Primer on Private Statistics" (2020, arXiv): fingerprinting lower bounds for private mean estimation, with the score-and-integrate-by-parts proof used here.
- Dinur & Nissim, "Revealing Information while Preserving Privacy" (PODS 2003): the reconstruction attack and the $\sqrt n$ threshold.
- Boneh & Shaw, "Collusion-Secure Fingerprinting for Digital Data" (IEEE Transactions on Information Theory, 1998) and Tardos, "Optimal Probabilistic Fingerprint Codes" (STOC 2003): the cryptographic codes that gave fingerprinting its name.

## Cheat sheet

$\alpha$ is the error, $\beta$ the failure probability, $d$ the number of attributes or queries, $N$ the universe size, $n$ the number of records. @tbl-lower-cheat collects the results.

| Result | Statement | Matching mechanism |
| --- | --- | --- |
| Packing lemma (pure) | $m$ disjoint good sets, all within $k$ of $D_0$: $k \ge \ln(m(1 - \beta))/\varepsilon$ | — |
| Mode over $N$ elements (pure) | Lead $k \ge \ln(N(1 - \beta))/\varepsilon$ | Report-noisy-max: $(N - 1)e^{-k\varepsilon}(2 + k\varepsilon)/4 \le \beta$ |
| Histogram, $\ell_\infty$ (pure) | $\alpha \ge \frac12(\ln(N(1 - \beta))/\varepsilon - 1)$ | Laplace: $\alpha \le \ln(N/\beta)/\varepsilon$ |
| Packing under $\delta > 0$ | $e^{-k\varepsilon}(1 - \beta) \le 2k\delta$ or $m e^{-k\varepsilon}(1 - \beta) \le 2$: $k \gtrsim \min(\ln m, \ln(1/\delta))/\varepsilon$ | Stability-based histogram and release |
| Mean with $\lvert\mu\rvert \le R$ (pure) | $n \ge \ln(\lfloor R/\alpha\rfloor(1 - \beta))/\varepsilon$ | Coarse histogram, then fine ([Private mean estimation](11-mean-estimation.md)) |
| Histogram, records | $n \ge (\ln(N(1 - \beta)) - \varepsilon)/(2\alpha\varepsilon)$; sampling needs $\approx\ln(2N/\beta)/(2\alpha^2)$ | Laplace |
| $d$ marginals (pure) | $n \gtrsim d\ln2/(\alpha\varepsilon)$ | Laplace: $n = 2d\ln(d/\beta)/(\alpha\varepsilon)$ |
| Fingerprinting lemma | $\mathbb E[f\sum_i(x_i - p)] = 2\mathbb E[pf] \ge \frac23 - 2\mathbb E\lvert f - \bar x\rvert$ | — |
| $d$ marginals (approximate) | $n \ge (\frac23 - 2\alpha')d/((e^\varepsilon - 1)\sqrt d + 4\delta d)$ | Gaussian: $n = 2\sqrt d\,z/(\mu\alpha)$ |
| Tracing score | Gap $2d/(3n)$; variance $d(\frac2{15} + \frac8{15n} + \frac{2\sigma^2}3)$ | Defeated by $\sigma \gg \sqrt d/n$ |
| Bit guessing / membership | Accuracy $\le \frac{e^\varepsilon + \delta}{e^\varepsilon + 1}$; advantage $\le \frac{e^\varepsilon - 1 + 2\delta}{e^\varepsilon + 1}$ | — |
| Dinur–Nissim | Recovering a fraction $r$ proves $\varepsilon \ge \ln\frac{r - \delta}{1 - r}$; error $o(\sqrt n)$ on $O(n)$ queries is never private | Gaussian with $\sigma \propto \sqrt k$ |

: Every lower bound in this chapter, with the mechanism that matches it. {#tbl-lower-cheat}

**Rules of thumb**

- A lower bound on error is an attack on accurate mechanisms plus DP's limit on attacks.
- Pure DP pays $\ln(\text{number of candidates})/\varepsilon$; for $d$-dimensional outputs that is linear in $d$.
- Approximate DP pays about $\sqrt d$ for $d$ statistics, and $\delta$ must stay far below $1/n$ for that to mean anything.
- Answering $k \lesssim n$ counting queries with independent noise costs error about $\sqrt k$ per query with $\delta > 0$, and about $k$ without. With many more queries than records, answering from a private synthetic dataset can do much better, with error that grows only like a small power of $\log k$ ([Query release](06-query-release.md)).
- Not knowing where the data lie costs only $\ln(R/\alpha)/\varepsilon$ records under pure DP, and at most about $\ln(1/\delta)/\varepsilon$ under approximate DP.
- Privacy is an additive cost in the number of records: it matters most for coarse answers at small $\varepsilon$.
- Aggregates over many attributes are not safe by being aggregates: without noise, tracing reaches advantage 0.5 once $d \approx 0.56\,n^2$ ([Exercise 5](../../book-exercises/part03-dp-foundations/10-lower-bounds.md#sec-sol-lower-5)). Noise of order $\sqrt d/n$ stops it.

**Practice.** Exercises for this chapter, each with a worked solution, are in [the exercise book](../../book-exercises/part03-dp-foundations/10-lower-bounds.md).
