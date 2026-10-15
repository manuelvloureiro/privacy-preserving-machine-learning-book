# Amplification by shuffling and by iteration

## Why this matters

[Amplification by subsampling](05-subsampling.md) showed that uncertainty is worth privacy. If an adversary cannot tell **whether** your record was used in a step, the step costs far less than its noise alone suggests: a step that touches your record only with probability $q$ leaks roughly $q$ times as much. That is not the only uncertainty an algorithm can create. Two others matter in practice, and each has its own accountant.

The first is uncertainty about **which report is yours**. In the shuffle model, $n$ users each randomize their own data on their own device and send the noisy report through a shuffler, a trusted component that strips identities and mixes the order. The analyst sees a bag of $n$ anonymous reports. Each report on its own satisfies local differential privacy (DP) with some level $\varepsilon_0$: whoever sees that one report, even the collector, learns little about its sender, and the smaller $\varepsilon_0$, the less. To keep the reports useful, $\varepsilon_0$ is usually large, which makes this a weak guarantee. But the analyst does not know which report in the bag is yours, so your report hides among the others. The result is central DP, the guarantee about everything the analyst sees, with an $\varepsilon$ that falls like $1/\sqrt n$. This is shuffling of anonymous, locally randomized reports. It is not the shuffle-and-partition batching used to train a model privately in the central model, where each record is used in exactly one batch per epoch and the shuffle only hides which one: [Amplification by subsampling](05-subsampling.md#sec-sub-schemes) shows that no comparable amplification theorem covers that case, and that the Poisson number is wrong for it.

The second is uncertainty about **when your record was used**. Noisy gradient descent (GD) on a convex loss forgets. Each later step adds fresh noise and pulls nearby parameters together, so the effect of your record on the final model is washed out step by step. If only the final model is released, a record used early in training is much better hidden than the composition accounting of [Rényi DP](01-renyi-dp.md) says, because composition charges every step as if the adversary had seen every intermediate model. For some problems, the privacy loss stops growing with the number of steps.

### The one idea: the adversary's view is a post-processing of something simpler

Every result in this chapter has the same shape. You show that what the adversary sees can be **generated from a much simpler pair of distributions** by a procedure that does not depend on your record. Anything the adversary can learn from the real output, it could then have learned from the simple pair, by running that procedure itself. This is the data-processing inequality, and it means the simple pair's privacy holds for the real output. For shuffling, the simple pair is a pair of **counts**: how many reports look like each of your two possible values. For iteration, it is a pair of **Gaussians**, or a pair of distributions that a short chain of noisy steps keeps close. The work is in finding the simple pair and computing how far apart its two distributions are. @tbl-shuf-overview lists the results.

| Result | Where it is used |
| --- | --- |
| Exact $\delta(\varepsilon)$ of shuffled binary randomized response | Ground truth for any shuffling bound; deploying randomized-response surveys |
| The clone reduction and bound of Feldman, McMillan & Talwar | Central guarantees for any local randomizer behind a shuffler |
| Shifted Rényi divergence, shift reduction, contraction | Privacy of the last iterate of contractive noisy iterations |
| Amplification by iteration for one-pass noisy gradient descent | Per-record guarantees that depend on when the record was used |
| Hidden-state bounds that plateau | Convex training run for many epochs, releasing only the final model |
| Exact last-iterate privacy of noisy GD on a quadratic | A worked case in which every quantity is in closed form |

: The results of this chapter and where each is used. {#tbl-shuf-overview}

**How to read this chapter.** The first two sections are about shuffling and the last three about iteration; the two halves are independent. In each half, read the exact computation first (shuffled randomized response; the quadratic loss), because it shows what the general bounds approximate. The derivations of the shift-reduction lemma and of the bounded-domain bound can wait for a second pass.

**What you need.** The Toolkit below restates every outside fact the derivations use, with a short reason for each, so you can read this chapter on its own. For more detail: the hockey-stick divergence $E_\gamma$, the Rényi divergence $D_\alpha$ and the data-processing inequality are in [Divergences](../part01-prerequisites/03-divergences.md). The non-expansiveness of a gradient step is Fact 4 of [Stability and generalization](../part01-prerequisites/07-stability-generalization.md), and projection onto a convex set is Fact 3 of [Optimization](../part01-prerequisites/06-optimization.md). Randomized response is in [The definition of differential privacy](../part03-dp-foundations/01-definition.md#sec-dpdef-rr), and the shuffle model in [Trust models](../part03-dp-foundations/08-trust-models.md). Rényi differential privacy (RDP) and its composition are in [Rényi DP](01-renyi-dp.md), and its conversion to $(\varepsilon, \delta)$ in [From RDP to (ε, δ)](02-rdp-conversions.md). Gaussian differential privacy (GDP) is in [f-DP and Gaussian DP](04-gaussian-dp.md).

**Conventions.** **Throughout this chapter the neighbouring relation is replace-one**: $D$ and $D'$ have the same size and differ in the value of one record, yours. Both shuffling theorems and both iteration theorems below are stated for it. $P$ and $Q$ are the distributions of the adversary's view on $D$ and on $D'$. A mechanism is $(\varepsilon, \delta)$-DP when $P(S) \le e^\varepsilon Q(S) + \delta$ and $Q(S) \le e^\varepsilon P(S) + \delta$ for every set $S$ of outputs: no event becomes more than $e^\varepsilon$ times likelier when your record changes, up to a small slack $\delta$. In the iteration sections, some symbols differ from the part's notation table. $\sigma$ is the noise standard deviation in parameter units, and $\tilde\sigma$ the noise multiplier. $L_f$ is the Lipschitz constant of the loss and $\beta$ its smoothness constant; $L$ stays the privacy loss. $d_K$ is the diameter of the convex set $K$ that the parameters live in, so that $D$ always means a dataset.

## Toolkit: three facts every derivation uses

### Fact 1 — $\delta(\varepsilon)$ of a discrete pair is a sorted tail sum

For a fixed $\varepsilon$, the smallest $\delta$ that makes $P(S) \le e^\varepsilon Q(S) + \delta$ true for every set $S$ is the largest gap $P(S) - e^\varepsilon Q(S)$ over all sets. That largest gap is the **hockey-stick divergence** $E_{e^\varepsilon}(P\|Q)$. To find it, look at what each outcome $y$ contributes to the gap: $P(y) - e^\varepsilon Q(y)$. This is positive exactly when $P(y)/Q(y) > e^\varepsilon$, that is, when the **privacy loss** $L(y) = \ln\frac{P(y)}{Q(y)}$, the log of how much likelier $y$ is under $D$ than under $D'$, exceeds $\varepsilon$. So the best set collects every outcome with $L(y) > \varepsilon$ and nothing else, and for distributions $P, Q$ on a finite set
$$\delta(\varepsilon) = E_{e^\varepsilon}(P\|Q) = \sum_{y:\,L(y) > \varepsilon}\big(P(y) - e^\varepsilon Q(y)\big).$$

This form makes the computation cheap. Sort the outcomes by $L(y)$ once and keep running sums of $P$ and $Q$ from the top. Then $\delta(\varepsilon)$ for any $\varepsilon$ is one lookup: find where $L$ drops below $\varepsilon$ and read off the two running sums. The $\varepsilon$ for a target $\delta$ is a bisection over these lookups. The script works this way, and builds the binomial probabilities in log space so that values of $10^{-300}$ cause no trouble.

The definition of $(\varepsilon, \delta)$-DP has two directions, because the adversary may try to tell $D$ from $D'$ or $D'$ from $D$. It needs $E_{e^\varepsilon}(P\|Q) \le \delta$ and $E_{e^\varepsilon}(Q\|P) \le \delta$, so the exact $\delta(\varepsilon)$ of a pair is the larger of the two sums.

### Fact 2 — Post-processing a simpler pair

Suppose there is a randomized map $f$, the same for both datasets, that turns a sample of $\tilde P$ into a sample of $P$ and a sample of $\tilde Q$ into a sample of $Q$; write $P = f(\tilde P)$ and $Q = f(\tilde Q)$. Then the real pair can be no further apart than the simple one: every divergence of $(P, Q)$ is at most that of $(\tilde P, \tilde Q)$,
$$E_\gamma(P\|Q) \le E_\gamma(\tilde P\|\tilde Q) \qquad\text{and}\qquad D_\alpha(P\|Q) \le D_\alpha(\tilde P\|\tilde Q).$$
This is the **data-processing inequality**. The reason, for $E_\gamma$, is that a test on the output can be pulled back to a test on the input. Take any set $S$ of outputs, and let $g(u) = \Pr[f(u) \in S]$, a number between 0 and 1 for each input $u$. Then $P(S) - \gamma Q(S) = \mathbb E_{\tilde P}[g] - \gamma\,\mathbb E_{\tilde Q}[g]$. The right side is a sum of $g(u)\big(\tilde P(u) - \gamma\tilde Q(u)\big)$ over $u$, and it is largest when $g = 1$ wherever the bracket is positive and $g = 0$ elsewhere, which gives exactly $E_\gamma(\tilde P\|\tilde Q)$. The Rényi divergence $D_\alpha$, defined in Fact 3, satisfies the same inequality, by Jensen's inequality applied to the convex function $t \mapsto t^\alpha$ inside its sum ([Divergences](../part01-prerequisites/03-divergences.md) proves it).

In this chapter the shuffled view will be a post-processing of counts, and the final model a post-processing of a coupled pair of noisy iterates.

### Fact 3 — Gaussian pairs

Gaussian pairs come up in both halves of the chapter, so it pays to know everything about them once. Suppose that on $D$ the adversary sees $Y \sim \mathcal N(m, v)$ and on $D'$ it sees $Y \sim \mathcal N(m', v)$: the same spread, a shifted centre. Subtracting $m$ and dividing by $\sqrt v$ is an invertible map that is the same for both datasets, so by Fact 2 applied in both directions it changes no divergence. It turns the pair into $\mathcal N(0, 1)$ against $\mathcal N(\mu, 1)$ with
$$\mu = \frac{|m - m'|}{\sqrt v},$$
the shift measured in standard deviations. A pair like this is **$\mu$-GDP**: exactly as hard to tell apart as two unit-variance normals $\mu$ apart. Everything else follows from the privacy loss of $\mathcal N(0, 1)$ against $\mathcal N(\mu, 1)$, which is the difference of the two log densities, $L(y) = \frac{(y - \mu)^2 - y^2}{2} = \frac{\mu^2}{2} - \mu y$. It is linear in $y$, so it is normal: when $y \sim \mathcal N(0, 1)$ it has mean $\mu^2/2$ and variance $\mu^2$, and when $y \sim \mathcal N(\mu, 1)$ it has mean $-\mu^2/2$ and the same variance.

The $\delta(\varepsilon)$ curve comes from the continuous form of Fact 1, $\delta(\varepsilon) = \Pr_P[L > \varepsilon] - e^\varepsilon\Pr_Q[L > \varepsilon]$. Write $\Phi$ for the standard normal cumulative distribution function, $\Phi(x) = \Pr[\mathcal N(0, 1) \le x]$. Under $P$, $L > \varepsilon$ means a $\mathcal N(\mu^2/2, \mu^2)$ variable exceeds $\varepsilon$, which has probability $\Phi\big((\mu^2/2 - \varepsilon)/\mu\big)$; under $Q$ the mean is $-\mu^2/2$. So $\delta(\varepsilon) = \Phi(-\frac\varepsilon\mu + \frac\mu2) - e^\varepsilon\Phi(-\frac\varepsilon\mu - \frac\mu2)$.

The RDP curve needs one more definition. The **Rényi divergence of order $\alpha > 1$** measures how far apart two distributions are by a moment of the likelihood ratio: $D_\alpha(P\|Q) = \frac{1}{\alpha - 1}\ln\mathbb E_Q\big[(P/Q)^\alpha\big] = \frac{1}{\alpha - 1}\ln\mathbb E_P\big[e^{(\alpha - 1)L}\big]$. A mechanism whose pair has $D_\alpha \le \varepsilon(\alpha)$ for all neighbours is $(\alpha, \varepsilon(\alpha))$-RDP. For a normal $L$ with mean $\mu^2/2$ and variance $\mu^2$, $\mathbb E[e^{(\alpha - 1)L}] = \exp\big((\alpha - 1)\frac{\mu^2}2 + (\alpha - 1)^2\frac{\mu^2}2\big) = \exp\big(\alpha(\alpha - 1)\frac{\mu^2}2\big)$, using the fact that a normal variable $X$ has $\mathbb E[e^{tX}] = e^{t\,\mathbb E[X] + t^2\mathrm{Var}(X)/2}$. So $\varepsilon(\alpha) = \alpha\mu^2/2$.

Two more properties are used throughout. Sums of independent Gaussians are Gaussian, with means and variances adding. Releasing $k$ independent Gaussian pairs with parameters $\mu_1, \dots, \mu_k$ is like seeing a $k$-dimensional normal shifted by $(\mu_1, \dots, \mu_k)$; only the direction of the shift matters, and along it the shift is $\sqrt{\mu_1^2 + \dots + \mu_k^2}$ standard deviations. So the composition is $\sqrt{\mu_1^2 + \dots + \mu_k^2}$-GDP ([f-DP and Gaussian DP](04-gaussian-dp.md) proves this for adaptively chosen steps too).

In RDP terms, hiding a shift of size $a$ under noise of standard deviation $\sigma$ is a Gaussian pair with $\mu = a/\sigma$, so it costs $\alpha\mu^2/2$:
$$R_\alpha(a) = D_\alpha\big(\mathcal N(0, \sigma^2)\,\|\,\mathcal N(a, \sigma^2)\big) = \frac{\alpha a^2}{2\sigma^2}.$$ {#eq-shuf-gauss-cost}

## Shuffled binary randomized response, exactly {#sec-shuf-rr}

### Context

Each of $n$ users holds a bit and reports it with binary randomized response: the true bit with probability $\theta = e^{\varepsilon_0}/(1 + e^{\varepsilon_0})$ (written $t$ in Part 3), the flipped bit otherwise. Each report on its own is $\varepsilon_0$-DP. A report of 1 has probability $\theta$ if you hold 1 and $1 - \theta$ if you hold 0, and the reverse for a report of 0, so the likelihood ratio of any report is at most $\theta/(1 - \theta) = e^{\varepsilon_0}$. A shuffler permutes the reports before the analyst sees them. How private is the shuffled bag? For this one randomizer the question has an exact answer, and that answer is the yardstick for every general bound in the next section.

### Statement

Once the order is gone, the only thing a bag of bits can tell the analyst is how many of them are 1. So the analyst's view is equivalent to the **number $Y$ of 1-reports**. Your report moves that count by at most one, and the other users' reports add random noise to it. Fix the other users' bits, and let your bit be 0 in $D$ and 1 in $D'$. If the other $n - 1$ users all hold 0, each of them reports 1 only when their coin flips the bit, so their number of 1-reports is $S \sim \mathrm{Bin}(n - 1, 1 - \theta)$, with probability mass function $b$. The count $Y = y$ arises either when your report is 0 and the others sent $y$ ones, or when your report is 1 and the others sent $y - 1$. Weighting the two cases by the probability of your report gives
$$P(y) = \theta\,b(y) + (1 - \theta)\,b(y - 1), \qquad Q(y) = (1 - \theta)\,b(y) + \theta\,b(y - 1), \qquad y = 0, \dots, n.$$ {#eq-shuf-rr-pair}
The exact $\delta(\varepsilon)$ of shuffled randomized response for this pair is the larger of $E_{e^\varepsilon}(P\|Q)$ and $E_{e^\varepsilon}(Q\|P)$, computed by Fact 1.

### Derivation

1. **The bag is the count.** A shuffled sequence of bits is a uniformly random arrangement of its 1s and 0s. Given the count $Y$, the arrangement carries no further information about anyone's data, because every arrangement with $Y$ ones is equally likely under both $D$ and $D'$. A summary with this property is a **sufficient statistic**. Fact 2 applies in both directions: the count is a function of the bag, and the bag can be rebuilt from the count by arranging $Y$ ones and $n - Y$ zeros in a random order, which uses randomness that does not depend on the data. So the divergences of the bag equal those of $Y$.
2. **Split the count.** Write $Y = S + B$, where $S$ is the number of 1-reports among the others and $B$ is your report. $S$ does not depend on your bit. $B \sim \mathrm{Bern}(1 - \theta)$ under $D$, because you hold 0 and report 1 only when you flip, and $B \sim \mathrm{Bern}(\theta)$ under $D'$.
3. **The others' count.** A user holding 0 reports 1 with probability $1 - \theta$, independently of everyone else, so the number of 1-reports among $n - 1$ such users is $S \sim \mathrm{Bin}(n - 1, 1 - \theta)$.
4. **Condition on $B$.** By the law of total probability, split on the value of your report: $P(Y = y) = \Pr[B = 0]\,b(y) + \Pr[B = 1]\,b(y - 1)$. With $\Pr[B = 0] = \theta$ under $D$ and $1 - \theta$ under $D'$, this is @eq-shuf-rr-pair.
5. **Other pairs.** The pair above assumes the other users all hold 0. If $m$ of them hold 1, those $m$ report 1 with probability $\theta$ each, so $S$ becomes $\mathrm{Bin}(m, \theta) + \mathrm{Bin}(n - 1 - m, 1 - \theta)$, the distribution of a sum of two independent counts (a convolution). A guarantee has to hold for the worst $m$. You only need to check half of them. Swapping 0 and 1 for every user, in both the bits and the reports, is a relabelling that the analyst could undo. It turns "$m$ others hold 1, you hold 0 or 1" into "$n - 1 - m$ others hold 1, you hold 1 or 0", the same pair with $D$ and $D'$ exchanged. Since $\delta(\varepsilon)$ already takes the larger of the two directions, $m$ and $n - 1 - m$ give the same $\varepsilon$, and $m \le (n - 1)/2$ covers every pair. The script scans every such $m$ at $n = 1000$ and $\delta = 10^{-6}$. For $\varepsilon_0 = 1$ and 4 the largest $\varepsilon$ is at $m = 0$. For $\varepsilon_0 = 2$ it is at $m = 2$: 0.323324 against 0.323279 at $m = 0$, because the discrete count makes $\varepsilon$ ripple slightly in $m$. Large $m$ is much better (0.2858 at $m = 499$), because then about half the others report 1 and the count is at its noisiest. At $n = 10^4$ and $10^5$ the script scans $m \le 30$, and the largest value it finds exceeds the $m = 0$ value by at most $4.1 \times 10^{-6}$. The rest of the chapter uses the pair $m = 0$ and calls its $\varepsilon$ the exact one. Strictly, it is exact for that pair and a lower bound on the worst case, within $10^{-4}$ of the largest value found: every entry of @tbl-shuf-compare is the same to four decimals.

**A walk-through with $n = 3$.** Take $\varepsilon_0 = \ln 3$, so $\theta = \frac{3}{1 + 3} = \frac34$. The two other users hold 0, so each reports 1 with probability $\frac14$, and $S \sim \mathrm{Bin}(2, \frac14)$ has $b = \big((\frac34)^2, 2\cdot\frac14\cdot\frac34, (\frac14)^2\big) = (\frac9{16}, \frac6{16}, \frac1{16})$. @tbl-shuf-n3 fills in @eq-shuf-rr-pair outcome by outcome. For example, $Q(1) = \frac14\cdot\frac6{16} + \frac34\cdot\frac9{16} = \frac{6 + 27}{64}$.

| Count $y$ | 0 | 1 | 2 | 3 |
| --- | --- | --- | --- | --- |
| $b(y)$, others' 1-reports | 9/16 | 6/16 | 1/16 | 0 |
| $P(y)$, you hold 0 | 27/64 | 27/64 | 9/64 | 1/64 |
| $Q(y)$, you hold 1 | 9/64 | 33/64 | 19/64 | 3/64 |
| $P(y)/Q(y)$ | 3.0000 | 0.8182 | 0.4737 | 0.3333 |

: Shuffled binary randomized response with $n = 3$ users and $\varepsilon_0 = \ln 3$; the two other users hold 0. {#tbl-shuf-n3}

Now apply Fact 1 at $\varepsilon = 0.5$, where $e^{0.5} = 1.6487$. Only $y = 0$ has a ratio $P/Q$ above 1.6487, so only it enters the sum: $E_{e^{0.5}}(P\|Q) = \frac{27 - 9e^{0.5}}{64} = 0.1900$. In the other direction, the ratios $Q/P$ are the reciprocals of the last row, and $y = 2$ and $y = 3$ exceed 1.6487, so $E_{e^{0.5}}(Q\|P) = \frac{(19 - 9e^{0.5}) + (3 - e^{0.5})}{64} = 0.0861$. The other direction is smaller, so $\delta(0.5) = 0.1900$. Compare a single unshuffled report, whose pair is $(\frac34, \frac14)$ against $(\frac14, \frac34)$: only one output has a ratio above $e^{0.5}$, and $\delta(0.5) = \frac34 - \frac14e^{0.5} = 0.3378$. Even two other users nearly halve $\delta$. At $\varepsilon = 1$ the two values are 0.0396 and 0.0704. A simulation of 200,000 shuffles under $D$ gives count frequencies 0.4226, 0.4203, 0.1413, 0.0158 against the exact 0.4219, 0.4219, 0.1406, 0.0156; the largest gap, 0.0015, is within sampling error.

### Interpretation

**Shuffling never improves pure DP.** Look at the count $y = 0$. Its ratio is $P(0)/Q(0) = \theta b(0)/((1 - \theta)b(0)) = e^{\varepsilon_0}$ for every $n$, because $b(-1) = 0$ removes the second term of both $P$ and $Q$. In words: if nobody reports a 1, you were probably the one holding 0, and no number of other users changes that ratio. What changes is how often it happens. That outcome has probability $\theta^n$ under $D$ (everyone, you included, reports 0), which is tiny for large $n$. So shuffling amplifies by moving the bad outcomes into $\delta$, as the first column of @tbl-shuf-n3 shows.

**A central limit view.** Why should the privacy fall like $1/\sqrt n$? Under both datasets $Y$ is the sum of many independent bits, so by the central limit theorem it is close to normal, and Fact 3 says a pair of normals with equal variance is $\mu$-GDP with $\mu$ equal to the mean difference divided by the standard deviation. The mean of $Y$ differs between $D$ and $D'$ only through your report, by $\mathbb E[B'] - \mathbb E[B] = \theta - (1 - \theta) = 2\theta - 1$, where $B'$ is your report under $D'$. The variance is $n\theta(1 - \theta)$ under both, since every report, yours included, is a Bernoulli variable that is 1 with probability $\theta$ or $1 - \theta$, and either way has variance $\theta(1 - \theta)$. Dividing,
$$\mu \approx \frac{2\theta - 1}{\sqrt{n\theta(1 - \theta)}} = \frac{e^{\varepsilon_0} - 1}{e^{\varepsilon_0/2}\sqrt n} = \frac{2\sinh(\varepsilon_0/2)}{\sqrt n},$$ {#eq-shuf-rr-clt}
where the middle form uses $2\theta - 1 = \frac{e^{\varepsilon_0} - 1}{e^{\varepsilon_0} + 1}$ and $\sqrt{\theta(1 - \theta)} = \sqrt{\frac{e^{\varepsilon_0}}{(e^{\varepsilon_0} + 1)^2}} = \frac{e^{\varepsilon_0/2}}{e^{\varepsilon_0} + 1}$, so that the factors $e^{\varepsilon_0} + 1$ cancel. The last form divides top and bottom by $e^{\varepsilon_0/2}$: $\frac{e^{\varepsilon_0} - 1}{e^{\varepsilon_0/2}} = e^{\varepsilon_0/2} - e^{-\varepsilon_0/2} = 2\sinh(\varepsilon_0/2)$. The privacy of shuffled randomized response falls like $1/\sqrt n$, with an $\varepsilon_0$-dependent constant that grows like $e^{\varepsilon_0/2}$.

**When the approximation fails.** The normal approximation is good when there are many 1-reports to hide among. At $n = 10^5$ and $\varepsilon_0 = 2$ it gives $\varepsilon = 0.0244$ against the exact 0.0247 (@tbl-shuf-compare). At $n = 1000$ and $\varepsilon_0 = 4$, each of the 999 others reports 1 with probability $1 - \theta = 1/(1 + e^4)$, so on average they send only $999/(1 + e^4) = 18.0$ 1-reports. A count that small is far from normal, and the approximation gives 0.9667 against the exact 1.6910. Use the exact computation, which costs milliseconds.

@fig-shuf-counts draws the two count distributions and the exact $\delta(\varepsilon)$ curves.

![Left: the distribution of the number of 1-reports among $n = 30$ shuffled randomized-response reports with $\varepsilon_0 = 2$, when you hold 0 (blue) and when you hold 1 (orange); the other users hold 0. Your report moves the count by at most one, against a spread of about $\sqrt{n\theta(1-\theta)}$. Right: the exact $\delta(\varepsilon)$ for one unshuffled report and for $n = 10, 100, 1000$ shuffled reports. All curves reach 0 only at $\varepsilon = \varepsilon_0$, but for large $n$ they are negligible far earlier.](figures/shuffling_iteration_counts.svg){#fig-shuf-counts}

### Example: a survey of 10,000 people

A company asks 10,000 users whether they used a feature, through randomized response with $\varepsilon_0 = 2$ and a shuffler. Each user's report alone is 2-DP, a weak guarantee: the truth is reported with probability $\theta = e^2/(1 + e^2) = 0.8808$, so a single report is the truth seven times out of eight. The shuffled bag is $(0.0870, 10^{-6})$-DP (@tbl-shuf-compare). The shuffler bought central-model privacy without anyone seeing raw data.

### Check yourself

- Why does the analysis of the count not depend on the order in which reports arrive? *(The shuffler outputs a uniformly random order, which carries no information. The count is sufficient for the bag.)*
- With $n = 3$ and $\varepsilon_0 = \ln 3$, what is the smallest $\varepsilon$ for which $\delta(\varepsilon) = 0$? *(It is $\ln 3$: the outcome $y = 0$ has likelihood ratio 3, so for any smaller $\varepsilon$ it contributes a positive amount to Fact 1's sum. Shuffling never lowers the pure-DP level.)*

## The clone bound for any local randomizer {#sec-shuf-clones}

### Context

Randomized response on bits is one randomizer. In practice users send histograms, $k$-ary responses or noisy vectors, and the shuffle is sometimes applied to adaptively chosen randomizers, as in shuffled stochastic gradient descent (SGD), where each user randomizes their own gradient before it is shuffled. You want one bound that holds for **every** $\varepsilon_0$-DP local randomizer, that is, every randomized map $R$ from a user's data to a report whose output probabilities change by at most a factor $e^{\varepsilon_0}$ when the input changes. Erlingsson et al. (2019) gave the first such bound; Balle et al. (2019) gave another with the "privacy blanket" idea, which writes each randomizer as a mixture of an input-independent part and the rest. The bound to use today is Feldman, McMillan & Talwar (2021), and its proof is short enough to follow.

### Statement

The bound says that shuffling turns $\varepsilon_0$-local privacy into a central $\varepsilon$ that shrinks like $1/\sqrt n$, with a constant that grows like $e^{\varepsilon_0/2}$ and a factor $\sqrt{\ln(1/\delta)}$ for the slack. It needs enough users relative to $e^{\varepsilon_0}$, for a reason the derivation makes clear.

**Theorem 3.1 of Feldman, McMillan & Talwar (2021).** Let each user apply an $\varepsilon_0$-DP local randomizer (possibly chosen adaptively from the previous outputs) to the data in a uniformly random order. For any $\delta \in [0, 1]$ with $\varepsilon_0 \le \ln\big(\frac{n}{16\ln(2/\delta)}\big)$, the output is $(\varepsilon, \delta)$-DP with
$$\varepsilon \le \ln\left(1 + \frac{e^{\varepsilon_0} - 1}{e^{\varepsilon_0} + 1}\left(\frac{8\sqrt{e^{\varepsilon_0}\ln(4/\delta)}}{\sqrt n} + \frac{8e^{\varepsilon_0}}{n}\right)\right).$$ {#eq-shuf-clone-bound}

The theorem rests on a reduction, which is the useful part. Any $\varepsilon_0$-DP randomizer, run on any other user's data, produces with probability $e^{-\varepsilon_0}$ a report that looks exactly like yours could, a **clone**. Once the order is hidden, the adversary is left counting how many reports look like your $D$-version and how many look like your $D'$-version, and the clones blur that count. To state it, let $C$ be the number of other users who act as clones, $A$ the number of those clones that look like your $D$-version, and $\Delta$ a bit that says whether your own report points the right way.

**Theorem 3.2 of the same paper (the reduction).** Let $C \sim \mathrm{Bin}(n - 1, e^{-\varepsilon_0})$, $A \sim \mathrm{Bin}(C, \frac12)$ and $\Delta \sim \mathrm{Bern}\big(\frac{e^{\varepsilon_0}}{e^{\varepsilon_0} + 1}\big)$, independent. The shuffled output on $D$ and on $D'$ is a post-processing, by one and the same randomized map, of
$$\tilde P = (A + \Delta,\; C - A + 1 - \Delta) \qquad\text{and}\qquad \tilde Q = (A + 1 - \Delta,\; C - A + \Delta).$$ {#eq-shuf-clone-pair}
In $\tilde P$ your report adds to the first count when $\Delta = 1$, and in $\tilde Q$ it adds to the second. By Fact 2, any $(\varepsilon, \delta)$ that holds for this pair of two-dimensional counts holds for the shuffled output. Computing the pair's $\delta(\varepsilon)$ exactly, with Fact 1, gives the **numeric clone bound**; @eq-shuf-clone-bound is an analytic upper bound on it.

### Derivation

Steps 1 to 4 prove the reduction, and step 1 carries the idea. Step 5 bounds the resulting pair.

1. **Every report contains a copy of yours.** Fix the two values $x^0$ (in $D$) and $x^1$ (in $D'$) that you might hold, and let $R(x)$ be the distribution of the report of a user holding $x$. Being $\varepsilon_0$-DP means that no output is more than $e^{\varepsilon_0}$ times likelier for one input than for another. Read from the other side, every output of $R(x)$ keeps at least a fraction $e^{-\varepsilon_0}$ of its probability under $R(x^0)$, and likewise for $x^1$: $R(x)(y) \ge e^{-\varepsilon_0}R(x^0)(y)$ and $R(x)(y) \ge e^{-\varepsilon_0}R(x^1)(y)$ for every output $y$. Average the two inequalities: $R(x) \ge \frac{e^{-\varepsilon_0}}2\big(R(x^0) + R(x^1)\big)$ pointwise. The difference between the two sides is non-negative, and its total mass is $1 - \frac{e^{-\varepsilon_0}}2(1 + 1) = 1 - e^{-\varepsilon_0}$, so it is $(1 - e^{-\varepsilon_0})$ times a distribution $\mathrm{LO}(x)$, the **leftover**:
$$R(x) = \frac{e^{-\varepsilon_0}}{2}R(x^0) + \frac{e^{-\varepsilon_0}}{2}R(x^1) + \big(1 - e^{-\varepsilon_0}\big)\mathrm{LO}(x).$$ {#eq-shuf-clone-split}
2. **Read the mixture as a three-sided coin.** A mixture is sampled by first choosing a component with its weight, then drawing from that component. So sampling from @eq-shuf-clone-split means: flip a coin that says "0" with probability $e^{-\varepsilon_0}/2$, "1" with probability $e^{-\varepsilon_0}/2$, "leftover" otherwise; then report a draw from $R(x^0)$, $R(x^1)$ or $\mathrm{LO}(x)$. A user whose coin says "0" sends a report distributed exactly like yours on $D$: a **clone**. Each of the other $n - 1$ users is a clone (face "0" or "1") with probability $e^{-\varepsilon_0}$, independently, so the number of clones is $C \sim \mathrm{Bin}(n - 1, e^{-\varepsilon_0})$. Each clone is a 0-clone or a 1-clone with probability $\frac12$, so $A \sim \mathrm{Bin}(C, \frac12)$ of them are 0-clones.
3. **After shuffling, only the counts of the coin faces matter.** Generate the output in two stages. First flip every other user's three-sided coin; then draw each report given its face. A user whose face is 0 or 1 draws from $R(x^0)$ or $R(x^1)$, which does not depend on who they are. A user whose face is "leftover" draws from $\mathrm{LO}(x_i)$, which depends only on that user's own data $x_i$, not on yours. Give yourself the face 0 on $D$ and 1 on $D'$, since your report is a draw from $R(x^0)$ or $R(x^1)$. After the shuffle nobody can tell whose report is whose. So someone who knows every other user's data, but not your bit, can produce the shuffled bag from just the number of 0-faces and 1-faces: draw that many reports from $R(x^0)$ and $R(x^1)$, draw a leftover report for every other user, and shuffle (Lemma 3.3 of the paper makes this precise). Those numbers are $(A + 1, C - A)$ on $D$ and $(A, C - A + 1)$ on $D'$. By Fact 2, the output is no more distinguishable than these counts.
4. **Your own report is a randomized response.** The counts of step 3 still give your face away exactly, by one. The last improvement uses the fact that your own report is itself noisy. Any $\varepsilon_0$-DP randomizer restricted to two inputs is a post-processing of binary randomized response (Kairouz, Oh & Viswanath 2015, Lemma 3.4 of the paper). Concretely, $R(x^0)$ and $R(x^1)$ are mixtures of two distributions $Q_0, Q_1$ with weights $\big(\frac{e^{\varepsilon_0}}{e^{\varepsilon_0}+1}, \frac1{e^{\varepsilon_0}+1}\big)$ and the reverse: your report is drawn from the distribution "on your side" with probability $\frac{e^{\varepsilon_0}}{e^{\varepsilon_0}+1}$ and from the other side otherwise, exactly like a randomized-response bit. Replacing your face by that randomized-response bit $\Delta$ gives @eq-shuf-clone-pair. This is where the factor $\frac{e^{\varepsilon_0} - 1}{e^{\varepsilon_0} + 1}$ in @eq-shuf-clone-bound comes from: it is $\Pr[\Delta = 1] - \Pr[\Delta = 0]$, how much more often randomized response points the right way than the wrong way. Equivalently, randomized response passes on your bit only with that probability, and outputs a fair coin otherwise.
5. **Bound the pair.** The last step bounds $E_{e^\varepsilon}(\tilde P\|\tilde Q)$ with binomial concentration (Chernoff bounds on $C$ and on $A$ given $C$; Lemma 3.5 and Appendix A of the paper). These bounds need many clones to hide among: the condition on $\varepsilon_0$ is the same as $ne^{-\varepsilon_0} \ge 16\ln(2/\delta)$, an expected number of clones large enough for $C$ to be close to its mean except with probability about $\delta$. We do not repeat the calculation; the script computes the pair's $\delta(\varepsilon)$ exactly instead.

**Step 1 on a concrete randomizer.** Binary randomized response with $\varepsilon_0 = \ln 3$ has $R(0) = (\frac34, \frac14)$ and $R(1) = (\frac14, \frac34)$ on outputs $(0, 1)$, and $e^{-\varepsilon_0} = \frac13$, so each clone weight is $\frac{e^{-\varepsilon_0}}2 = \frac16$ and the leftover weight is $1 - \frac13 = \frac23$. Take another user who holds $x = 0$, with your values $x^0 = 0$, $x^1 = 1$. @tbl-shuf-clone-split decomposes this user's report.

| Output $y$ | $R(0)(y)$ | $\frac16R(x^0)(y)$ | $\frac16R(x^1)(y)$ | $\frac23\mathrm{LO}(0)(y)$ | $\mathrm{LO}(0)(y)$ |
| --- | --- | --- | --- | --- | --- |
| 0 | 3/4 | 1/8 | 1/24 | 7/12 | 7/8 |
| 1 | 1/4 | 1/24 | 1/8 | 1/12 | 1/8 |
| Total | 1 | 1/6 | 1/6 | 2/3 | 1 |

: The clone decomposition @eq-shuf-clone-split for binary randomized response with $\varepsilon_0 = \ln 3$, for a user holding 0. With probability $\frac16$ the user's report is a draw from $R(0)$, with probability $\frac16$ a draw from $R(1)$, and otherwise a draw from the leftover. {#tbl-shuf-clone-split}

The fourth column is what remains of $R(0)$ after the two clone parts are taken out, and the last column rescales it by $\frac32$ to a distribution. Each row adds up: $\frac18 + \frac1{24} + \frac7{12} = \frac34$. The script prints the clone part $(0.1667, 0.1667)$ and $\mathrm{LO}(0) = (0.875, 0.125)$.

### Interpretation

**Why $1/\sqrt n$ and $e^{\varepsilon_0/2}$.** The central limit view of the previous section explains the shape of the bound. Given $C = c$ clones, the first coordinates of $\tilde P$ and $\tilde Q$ are $A + \Delta$ and $A + 1 - \Delta$. Their means differ by $\mathbb E[\Delta] - \mathbb E[1 - \Delta] = 2\Pr[\Delta = 1] - 1 = \frac{e^{\varepsilon_0}-1}{e^{\varepsilon_0}+1}$. The noise that hides this difference is the spread of $A \sim \mathrm{Bin}(c, \frac12)$, a standard deviation of $\sqrt{c\cdot\frac12\cdot\frac12} = \sqrt c/2$. The expected number of clones is $c \approx ne^{-\varepsilon_0}$, and dividing the mean difference by the spread, as Fact 3 does for a Gaussian pair, gives $\mu \approx \frac{e^{\varepsilon_0}-1}{e^{\varepsilon_0}+1}\cdot\frac{2}{\sqrt{ne^{-\varepsilon_0}}}$. That is the shape of @eq-shuf-clone-bound: $\varepsilon = O\big(\frac{e^{\varepsilon_0}-1}{e^{\varepsilon_0}+1}\sqrt{e^{\varepsilon_0}\ln(1/\delta)/n}\big)$. For $\varepsilon_0 \le 1$ the first factor is about $\varepsilon_0/2$ and $e^{\varepsilon_0}$ is at most $e$, so this is of order $\varepsilon_0\sqrt{\ln(1/\delta)/n}$; above 1 the first factor tends to 1, leaving $\sqrt{e^{\varepsilon_0}\ln(1/\delta)/n}$.

**What the general bound gives up.** Divide the clone $\mu$ by the randomized-response $\mu$ of @eq-shuf-rr-clt, written as $\frac{e^{\varepsilon_0} - 1}{e^{\varepsilon_0/2}\sqrt n}$. The factors $e^{\varepsilon_0} - 1$ and $\sqrt n$ cancel, and what is left is $\frac{2e^{\varepsilon_0/2}}{e^{\varepsilon_0} + 1}\cdot e^{\varepsilon_0/2} = \frac{2e^{\varepsilon_0}}{e^{\varepsilon_0} + 1} = \frac{2}{1 + e^{-\varepsilon_0}}$. The ratio is above 1 for a clear reason. In binary randomized response every other report is a bit that could be yours, while the clone argument counts only the $e^{-\varepsilon_0}$ fraction of reports guaranteed to look like yours, split into two halves. The script gives this ratio as 1.46, 1.76 and 1.96 for $\varepsilon_0 = 1, 2, 4$; for instance $2e/(e + 1) = 1.46$. The measured ratios of the numeric clone bound to the exact $\varepsilon$ in @tbl-shuf-compare are 1.44 to 2.00 wherever the expected number of clones $(n - 1)e^{-\varepsilon_0}$ is large, close to this prediction. At $n = 1000$ and $\varepsilon_0 = 4$ there are only about $999e^{-4} = 18.3$ expected clones, far from the central limit regime, and the ratio is 2.36. No general bound can go below the exact column, since binary randomized response is one of the randomizers it must cover.

**The closed form is loose by another factor of 3 to 5.** @eq-shuf-clone-bound comes from Chernoff bounds with constant 8, which are safe but not sharp. Where its condition holds in @tbl-shuf-compare, it is 3.10 to 4.75 times the numeric clone bound and 4.47 to 7.55 times the exact $\varepsilon$. Use it to reason about scaling; compute the numeric bound to deploy.

**The condition matters.** Below $ne^{-\varepsilon_0} = 16\ln(2/\delta)$ there may be too few clones, and @eq-shuf-clone-bound says nothing. Solving for $\varepsilon_0$ gives the largest allowed level, $\ln\big(n/(16\ln(2/\delta))\big)$. At $\delta = 10^{-6}$, $16\ln(2 \times 10^6) = 232.1$, so $\varepsilon_0$ must be at most $\ln(1000/232.1) = 1.460$ for $n = 1000$, 3.763 for $n = 10^4$ and 6.066 for $n = 10^5$.

**The first bound was much weaker.** Erlingsson et al. (2019, Theorem 7) prove $\varepsilon \le \varepsilon_1\sqrt{2n\ln(1/\delta)} + n\varepsilon_1(e^{\varepsilon_1} - 1)$ with $\varepsilon_1 = 2e^{2\varepsilon_0}(e^{\varepsilon_0} - 1)/n$, through a swap argument, subsampling and advanced composition. Substituting $\varepsilon_1$ into the first term, its leading term is $2e^{2\varepsilon_0}(e^{\varepsilon_0} - 1)\sqrt{2\ln(1/\delta)/n}$, so its dependence on $\varepsilon_0$ is $e^{3\varepsilon_0}$ instead of $e^{\varepsilon_0/2}$, a factor $e^{2.5\varepsilon_0}$ worse. In @tbl-shuf-compare it is below $\varepsilon_0$, that is, it amplifies at all, only for $n = 10^5$ and $\varepsilon_0 = 1$, where it gives 0.4285.

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

: Central $\varepsilon$ at $\delta = 10^{-6}$ after shuffling $n$ reports that are each $\varepsilon_0$-DP. "n/a" marks where the condition of @eq-shuf-clone-bound fails; "—" marks where the bound of Erlingsson et al. is at least $\varepsilon_0$. The $n = 10^4$ rows agree with the table in [Trust models](../part03-dp-foundations/08-trust-models.md). {#tbl-shuf-compare}

@fig-shuf-bounds draws the same comparison over $n$. All curves fall like $1/\sqrt n$ once $n$ is large; the constants are what separates them. Multiplying by $\sqrt n$ removes the common $1/\sqrt n$ and leaves the constant: the exact $\varepsilon$ times $\sqrt n$ is 3.566, 8.702 and 31.464 at $n = 10^4$ for $\varepsilon_0 = 1, 2, 4$, growing with $\varepsilon_0$ roughly as $2\sinh(\varepsilon_0/2)$ in @eq-shuf-rr-clt does.

![Central $\varepsilon$ at $\delta = 10^{-6}$ against the number of users $n$, for $\varepsilon_0 = 1, 2, 4$: the closed-form clone bound (drawn only where its condition holds), the numeric clone bound of Theorem 3.2, the bound of Erlingsson et al. (drawn where it is below 50, so it appears only for large $n$ when $\varepsilon_0 = 2$ and not at all when $\varepsilon_0 = 4$, where it exceeds $7 \times 10^6$ for every $n$ shown) and the exact $\varepsilon$ of shuffled binary randomized response. The grey line is the local $\varepsilon_0$.](figures/shuffling_iteration_bounds.svg){#fig-shuf-bounds}

### Example: how large may each user's $\varepsilon_0$ be?

You collect a bit from $n = 10^5$ users behind a shuffler and want central $\varepsilon \le 0.2$ at $\delta = 10^{-6}$. Of the values in @tbl-shuf-compare, the closed form allows $\varepsilon_0 = 2$ (0.1862), the numeric clone bound allows $\varepsilon_0 = 4$ (0.1698), and for binary randomized response the exact value at $\varepsilon_0 = 4$ is 0.0847. A larger $\varepsilon_0$ means less noise per report, so it matters what $\varepsilon_0 = 4$ buys. Each report is the truth with probability $\theta = e^4/(1 + e^4) = 0.9820$. Suppose the users are a random sample from a population in which a proportion $p$ hold 1. A report is then 1 with probability $q = p\theta + (1 - p)(1 - \theta) = (1 - \theta) + p(2\theta - 1)$, so the fraction $\bar Y$ of 1-reports estimates $q$, and solving for $p$ gives the debiased estimate $\hat p = (\bar Y - (1 - \theta))/(2\theta - 1)$. Its variance is $q(1 - q)/\big(n(2\theta - 1)^2\big)$, and expanding $q(1 - q) = (2\theta - 1)^2p(1 - p) + \theta(1 - \theta)$ gives $\big(p(1 - p) + \frac{\theta(1 - \theta)}{(2\theta - 1)^2}\big)/n$. The first term is the ordinary sampling variance, which you would pay without any privacy. The term added by the noise is $\frac{0.9820 \times 0.0180}{0.9640^2} = 0.0190$, against up to 0.25 for the sampling term $p(1 - p)$, so the noise costs little accuracy. Which accountant you use decides whether a deployment is feasible.

### Check yourself

- In step 1, why average the two inequalities instead of using just one? *(Using only $x^0$ would give clones of $D$'s version of you but none of $D'$'s. The analyst could then count reports that look like $x^1$. Splitting the clones evenly between the two makes the situation symmetric.)*
- Why does the analysis need a uniformly random order, rather than a fixed one? *(With a fixed order the analyst knows which report is yours, and the guarantee falls back to $\varepsilon_0$. The random order is what lets the faces be described by their counts.)*

## Amplification by iteration {#sec-shuf-iteration}

### Context

Projected noisy SGD makes one pass over the data: at step $t$ it takes a gradient step on record $x_t$, adds Gaussian noise, and projects back onto a convex set of allowed parameters. If every iterate is released, the record $x_t$ is protected only by the noise of step $t$, as in one Gaussian mechanism: a value with bounded sensitivity plus Gaussian noise. If only the final iterate $w_n$ is released, the noise of steps $t + 1, \dots, n$ also covers $x_t$. Feldman, Mironov, Talwar & Thakurta (2018) turned this into a guarantee. The obstacle is that the steps in between are arbitrary maps applied to random points, not additions, so you cannot just add up variances as for a sum of Gaussians. Their tool is a divergence that allows for a bounded shift.

### Statement

The argument needs every step to be a map that never pushes two points further apart. A **contractive noisy iteration** is $X_{t} = \psi_{t}(X_{t-1}) + Z_{t}$, where each $\psi_t$ is a **contraction**, $\|\psi_t(u) - \psi_t(v)\| \le \|u - v\|$, and $Z_t \sim \mathcal N(0, \sigma^2 I)$ is fresh noise. Two kinds of maps in noisy SGD have this property.

A gradient step $w \mapsto w - \eta\nabla f(w)$ on a convex $\beta$-smooth $f$ is a contraction when $\eta \le 2/\beta$ ([Stability and generalization](../part01-prerequisites/07-stability-generalization.md), Fact 4). The reason is that for such $f$ the gradient is **co-coercive**: $\langle\nabla f(u) - \nabla f(v), u - v\rangle \ge \frac1\beta\|\nabla f(u) - \nabla f(v)\|^2$, so the gradients of two points differ in a direction that brings them together. Writing $g = \nabla f(u) - \nabla f(v)$ and expanding the square, $\|u - v - \eta g\|^2 = \|u - v\|^2 - 2\eta\langle g, u - v\rangle + \eta^2\|g\|^2 \le \|u - v\|^2 - \eta(\frac2\beta - \eta)\|g\|^2$, which is at most $\|u - v\|^2$ when $\eta \le 2/\beta$.

The projection $\Pi$ onto a closed convex set $K$, which maps each point to the nearest point of $K$, is a contraction too. The proof of Fact 3 of [Optimization](../part01-prerequisites/06-optimization.md) shows that the nearest point makes an obtuse angle with every other point of $K$: $\langle u - \Pi u, w - \Pi u\rangle \le 0$ for every $w \in K$. Take $w = \Pi v$, then exchange $u$ and $v$, and add the two inequalities: $\|\Pi u - \Pi v\|^2 \le \langle u - v, \Pi u - \Pi v\rangle \le \|u - v\|\,\|\Pi u - \Pi v\|$ by Cauchy–Schwarz, so $\|\Pi u - \Pi v\| \le \|u - v\|$. A composition of contractions is a contraction, so a projected gradient step is one too.

Now the divergence. The two runs, on $D$ and on $D'$, are driven apart at the step that uses your record, and after that the contractions keep the gap from growing while the noise blurs it. The quantity to track is therefore how far apart the two runs might be (a **shift**) and how much divergence would remain if you were allowed to move one of them by that shift. To move a random variable by at most $z$ means to pair each of its values with a value at most $z$ away; such a pairing of two random variables is a **coupling**. The **shifted Rényi divergence** allows the first distribution to be moved by at most $z$ in this way before it is compared, with the Rényi divergence $D_\alpha$ of Fact 3:
$$D_\alpha^{(z)}(X\|X') = \inf\big\{D_\alpha(Y\|X') : \text{some coupling of } X \text{ and } Y \text{ has } \|X - Y\| \le z \text{ almost surely}\big\}.$$
With $z = 0$ no move is allowed, and it is $D_\alpha$. Two lemmas drive everything (Feldman et al. 2018, Lemmas 20 and 21): noise lets you pay off part of the shift, and contractions only add the gap between the two maps to the shift, at no cost.

**Shift reduction.** Adding noise lets you buy down the shift. Let $R_\alpha(a)$ be the largest $D_\alpha$ between the noise and a copy of it shifted by at most $a$; for Gaussian $Z$ it is @eq-shuf-gauss-cost. Adding $Z$ to both sides lowers the allowed shift by $a$ at a price $R_\alpha(a)$: for any $a \ge 0$,
$$D_\alpha^{(z)}(X + Z\,\|\,X' + Z) \le D_\alpha^{(z + a)}(X\|X') + R_\alpha(a).$$ {#eq-shuf-shift-reduction}

**Contraction.** Applying contractions $\psi, \psi'$ to the two sides cannot move points apart, but the two maps themselves may differ (they do at the step that uses your record). If they differ by at most $s$ at every point, $\sup_u\|\psi(u) - \psi'(u)\| \le s$, the shift needed grows by at most $s$ and the divergence does not grow:
$$D_\alpha^{(z + s)}\big(\psi(X)\,\|\,\psi'(X')\big) \le D_\alpha^{(z)}(X\|X').$$ {#eq-shuf-contraction}

Chaining the two lemmas over $T$ steps gives their Theorem 22. Let the two runs differ by $s_t = \sup_u\|\psi_t(u) - \psi'_t(u)\|$ at step $t$, and choose payments $a_1, \dots, a_T \ge 0$ such that the unpaid shift $z_t = \sum_{i \le t}(s_i - a_i)$ never goes negative and is 0 at the end, $z_T = 0$. Then $D_\alpha(X_T\|X'_T) \le \sum_tR_\alpha(a_t)$: the divergence of the last iterate is at most the total price of the payments. For noisy SGD this becomes the following.

**Theorem 23 of Feldman et al. (2018).** Let the losses $f(\cdot, x)$ be convex, $L_f$-Lipschitz and $\beta$-smooth on a convex set $K$, and run projected noisy SGD $w_{t} = \Pi_K\big(w_{t-1} - \eta(\nabla f(w_{t-1}, x_t) + Z_t)\big)$, $Z_t \sim \mathcal N(0, \sigma^2 I)$, $\eta \le 2/\beta$, once over $x_1, \dots, x_n$. Releasing only $w_n$ is $\big(\alpha, \frac{2\alpha L_f^2}{\sigma^2(n - t + 1)}\big)$-RDP for the record $x_t$, for every $\alpha > 1$.

The guarantee is easier to read in terms of the **noise multiplier** $\tilde\sigma = \sigma/(2L_f)$, the noise per unit of the gradient's replace-one sensitivity. That sensitivity is $2L_f$ because an $L_f$-Lipschitz loss has gradients of norm at most $L_f$, so the gradients on $x_t$ and $x'_t$ differ by at most $2L_f$. Substituting $\sigma^2 = 4L_f^2\tilde\sigma^2$, the RDP of the record at step $t$ reads
$$\varepsilon_t(\alpha) = \frac{\alpha}{2\tilde\sigma^2}\cdot\frac{1}{n - t + 1}:$$ {#eq-shuf-fmtt}
one step's Gaussian RDP curve, divided by the number of noisy steps from $t$ to the end.

### Derivation

**Shift reduction for one Gaussian step, concretely.** The lemma is easiest to see on two point masses. Let $X = 0$ and $X' = z$, a distance $z$ apart, and add $Z \sim \mathcal N(0, \sigma^2)$ to both.

1. With no shift allowed, the two outputs are $\mathcal N(0, \sigma^2)$ and $\mathcal N(z, \sigma^2)$, a Gaussian pair with $\mu = z/\sigma$, so $D_\alpha = \frac{\alpha z^2}{2\sigma^2}$ (Fact 3).
2. Now allow a shift of $z - a$, and use it: move the first output by $z - a$ towards the second. That is, $Y = X + Z + (z - a) \sim \mathcal N(z - a, \sigma^2)$, coupled to $X + Z$ at distance exactly $z - a$. Only $a$ of the gap is left, so $D_\alpha(Y\|X' + Z) = \frac{\alpha a^2}{2\sigma^2}$ by Fact 3 again. Hence $D_\alpha^{(z - a)}(X + Z\|X' + Z) \le R_\alpha(a)$.
3. Now read @eq-shuf-shift-reduction with shift $z - a$ on the left. Its right side is $D_\alpha^{(z)}(X\|X') + R_\alpha(a)$, and $D_\alpha^{(z)}(X\|X') = 0$ because moving $X$ by $z$ turns it into $X'$. So the lemma gives exactly the bound of step 2. The noise **pays for $a$ of the shift at price $\alpha a^2/(2\sigma^2)$**; the rest of the shift is carried forward. The general proof (Lemma 20 of the paper) does the same with a coupling $U + W$, $\|W\| \le a$, and bounds the Rényi divergence of the joint pair $(U + W, Z - W)$.

**Spreading the shift.** Now let the two runs differ only at step $t$, by $s$, and apply identity maps afterwards ($\psi_k(u) = u$, a contraction). You decide how much shift to pay at each later step, $a_t, \dots, a_T$ with $\sum_ka_k = s$. Since the price is quadratic in each payment, small equal payments should be cheapest.

1. By @eq-shuf-contraction at step $t$, the shift grows from 0 to $s$. By @eq-shuf-shift-reduction at each step $k \ge t$ it shrinks by $a_k$ at cost $R_\alpha(a_k)$. At step $T$ it is 0, so the shifted divergence is the ordinary $D_\alpha$.
2. The total cost is $\frac{\alpha}{2\sigma^2}\sum_ka_k^2$. Cauchy–Schwarz, applied to the vectors $(a_t, \dots, a_T)$ and $(1, \dots, 1)$ of length $T - t + 1$, gives $(\sum_ka_k)^2 \le (T - t + 1)\sum_ka_k^2$, that is, $\sum_ka_k^2 \ge \frac{(\sum_ka_k)^2}{T - t + 1}$, with equality for the even schedule $a_k = s/(T - t + 1)$.
3. So $D_\alpha(X_T\|X'_T) \le \frac{\alpha s^2}{2\sigma^2(T - t + 1)}$.
4. **The general induction (Theorem 22).** With arbitrary contractions in between, write $z_k = \sum_{i\le k}(s_i - a_i)$ for the shift still unpaid after step $k$. The claim is that the divergence, with the unpaid shift allowed, is at most what has been paid so far: $D_\alpha^{(z_k)}(X_k\|X'_k) \le \sum_{i\le k}R_\alpha(a_i)$. It holds at $k = 0$, where both runs start at the same point. For the step from $k$ to $k + 1$, write the new iterates as the maps plus noise, peel off the noise with shift reduction (paying $a_{k+1}$), rewrite the shift using $z_{k+1} + a_{k+1} = z_k + s_{k+1}$, and peel off the maps with the contraction lemma:
$$\begin{aligned}
D_\alpha^{(z_{k+1})}(X_{k+1}\|X'_{k+1}) &= D_\alpha^{(z_{k+1})}\big(\psi_{k+1}(X_k) + Z\,\|\,\psi'_{k+1}(X'_k) + Z\big)\\
&\le D_\alpha^{(z_{k+1} + a_{k+1})}\big(\psi_{k+1}(X_k)\,\|\,\psi'_{k+1}(X'_k)\big) + R_\alpha(a_{k+1}) &&\text{(shift reduction)}\\
&= D_\alpha^{(z_k + s_{k+1})}\big(\psi_{k+1}(X_k)\,\|\,\psi'_{k+1}(X'_k)\big) + R_\alpha(a_{k+1}) &&\text{(definition of } z_{k+1})\\
&\le D_\alpha^{(z_k)}(X_k\|X'_k) + R_\alpha(a_{k+1}) &&\text{(contraction)}.
\end{aligned}$$
Applying the claim for step $k$ to the last line proves it for step $k + 1$. At $k = T$ the shift is $z_T = 0$, and the shifted divergence is the ordinary one. The contractions never enter the cost: they can only keep the shift from growing beyond $z_k + s_{k+1}$.
5. For noisy SGD, the steps on $x_t$ and $x'_t$ differ by $s = \eta\|\nabla f(w, x_t) - \nabla f(w, x'_t)\| \le 2\eta L_f$, since each gradient has norm at most $L_f$ (Lipschitz). The update subtracts $\eta Z_t$, so the noise in parameter space has standard deviation $\eta\sigma$. Step 3, with the induction of step 4 in place of the identity maps and $n$ in place of $T$, gives $\frac{\alpha(2\eta L_f)^2}{2\eta^2\sigma^2(n - t + 1)} = \frac{2\alpha L_f^2}{\sigma^2(n - t + 1)}$, which is Theorem 23. The step size $\eta$ cancels, because it scales the shift and the noise alike.

@tbl-shuf-schedule walks through steps 1 to 3 with $s = 1$, $\sigma = 1$, $\alpha = 2$ and $T - t + 1 = 4$ noisy steps. Each step pays $a = \frac14$ at price $2 \cdot (\frac14)^2/2 = 0.0625$.

| Step | 1 | 2 | 3 | 4 | Total |
| --- | --- | --- | --- | --- | --- |
| Shift before the step | 1 | 0.75 | 0.5 | 0.25 | |
| Shift paid, $a_k$ | 0.25 | 0.25 | 0.25 | 0.25 | 1 |
| Cost $\alpha a_k^2/(2\sigma^2)$ | 0.0625 | 0.0625 | 0.0625 | 0.0625 | 0.2500 |

: Paying off a shift of 1 over four Gaussian steps with $\sigma = 1$, at $\alpha = 2$. The total equals $D_2(\mathcal N(0, 4)\|\mathcal N(1, 4)) = 0.25$. {#tbl-shuf-schedule}

The total matches the exact divergence because four independent steps of variance 1 add up to variance 4, and by Fact 3 a shift of 1 under variance 4 costs $\frac{2 \cdot 1^2}{2 \cdot 4} = 0.25$. Uneven schedules cost more, because squares punish large payments: $(0.8, 0.2, 0, 0)$ costs $0.8^2 + 0.2^2 = 0.6800$ (with $\alpha/(2\sigma^2) = 1$), and paying everything at once costs 1. @fig-shuf-shift draws the schedule and its cost as the number of steps grows.

![Left: the even shift schedule over $T = 4$ noisy steps; each red arrow is a payment of $a = z/T$ at price $\alpha a^2/(2\sigma^2)$. Right: the total cost of the even schedule (dots) against the exact $D_2(\mathcal N(0, T)\|\mathcal N(1, T))$ (line), for a shift $z = 1$ and $\sigma = 1$; the dotted line is the cost of paying the whole shift in one step.](figures/shuffling_iteration_shift.svg){#fig-shuf-shift}

### Interpretation

**Exact for translations from a point.** The even schedule loses nothing when the maps after step $t$ are translations (they move every point by the same vector, so they keep the shift exactly as it is) and the two runs start from the same point at step $t$. Then the last iterate on $D$ and $D'$ is a pair of Gaussians shifted by $s$ with variance $(T - t + 1)\sigma^2$ added after the shift, whose exact $D_\alpha$ is $\frac{\alpha s^2}{2(T - t + 1)\sigma^2}$ by Fact 3. That is exactly the even schedule's cost, for every $T$ (@fig-shuf-shift, right; the script checks $T = 1, 2, 4, 10$).

**Two things Theorem 23 ignores.** Noise added **before** step $t$ also hides the shift when the maps preserve it, and a **strict** contraction, one that brings points closer, shrinks the shift for free. @fig-shuf-positions compares the bound with the exact $\mu$ of a one-dimensional run, $w_k = c\,w_{k-1} + b_k + \mathcal N(0, \sigma^2)$, in which $c \le 1$ is the contraction factor and your record shifts only the offset $b_t$, by $s = \sigma$. Unrolling the run as in @sec-shuf-quadratic, the last iterate is Gaussian, your shift reaches the end damped to $c^{T-t}s$, and the noise has standard deviation $\sigma\sqrt{\sum_{j=0}^{T-1}c^{2j}}$; their ratio is the exact $\mu$ (Fact 3). For $c = 1$ the shift persists unchanged to the end and all $T$ noise draws pile up on top of it, so the exact $\mu$ is $\frac{s}{\sigma\sqrt T} = 1/\sqrt T = 0.1000$ at every position, with $T = 100$. The bound, $1/\sqrt{T - t + 1}$, counts only the noise after step $t$ and rises to 1 at $t = T$. For $c = 0.95$ the exact $\mu$ is far smaller (0.0240 at $t = 50$ against the bound's $1/\sqrt{51} = 0.1400$).

**When it is exact.** The bound is attained when step $t$ first maps every point to a single point, which is a contraction, so that the earlier noise is erased. Then only the noise from step $t$ on hides the shift, and the exact $\mu$ is $s/(\sigma\sqrt{T - t + 1})$, which matches Theorem 23 in its $\mu$ form. So Theorem 23 is exact in the worst case over contractions, and loose for typical ones.

**Early records are safer, late ones are not.** The last record is protected by one step only. A randomly chosen stopping time spreads this out: Theorem 26 of the paper gives $\big(\alpha, \frac{4\alpha L_f^2\ln n}{n\sigma^2}\big)$-RDP for every record when the number of steps is uniform on $\{1, \dots, n\}$, provided $\sigma \ge L_f\sqrt{2(\alpha - 1)\alpha}$. Substituting $\sigma^2 = 4L_f^2\tilde\sigma^2$, that is $\alpha\ln n/(n\tilde\sigma^2)$ in noise-multiplier form. The first record gets $\frac{\alpha}{2\tilde\sigma^2n}$ from @eq-shuf-fmtt with $t = 1$, so the random stopping time gives every record $2\ln n$ times the guarantee of the first record.

**One pass, no sampling.** Each record is used once, at a fixed position. Several passes need the general Theorem 22 with several shifts; with random sampling, Altschuler & Talwar (2022) combine iteration with subsampling.

![The $\mu$ of the last iterate against the step $t$ at which your record is used, for a one-pass run of $T = 100$ steps with per-step shift equal to the noise standard deviation ($s/\sigma = 1$). Releasing all iterates gives $\mu = 1$ for every record (dotted). Theorem 23 (red, as $\mu = 1/\sqrt{T - t + 1}$) is exact when the run is reset just before step $t$; with $c = 1$ (blue) the earlier noise also helps, and with a strict contraction $c = 0.95$ (orange) the shift itself decays.](figures/shuffling_iteration_positions.svg){#fig-shuf-positions}

### Example: a one-pass run over 10,000 records

Projected noisy SGD makes one pass over $n = 10{,}000$ records with noise multiplier $\tilde\sigma = 1$. An RDP curve becomes an $(\varepsilon, \delta)$ guarantee through Mironov's conversion: at each order $\alpha$, the probability that the privacy loss exceeds $\varepsilon$ is bounded by Markov's inequality applied to $e^{(\alpha - 1)L}$, whose mean the RDP curve controls, and this gives $\varepsilon = \varepsilon(\alpha) + \frac{\ln(1/\delta)}{\alpha - 1}$. The first term grows with $\alpha$ and the second shrinks, so you minimize over the part's grid of orders. At $\delta = 10^{-5}$ the second term is $\ln(10^5)/(\alpha - 1) = 11.51/(\alpha - 1)$. Converting @eq-shuf-fmtt this way gives a different answer for each position.

The last record ($t = 10{,}000$) has $n - t + 1 = 1$ and gets $\varepsilon = 5.2985$. Its RDP curve is $\alpha/2$, the same as a single Gaussian mechanism with $\tilde\sigma = 1$, which is also what every record gets if all iterates are released. (That mechanism is a Gaussian pair with $\mu = 1$, and its exact $\varepsilon$, read from the $\delta(\varepsilon)$ curve of Fact 3, is 4.3772: the RDP conversion is not tight.) The record used 100 steps before the end ($t = 9901$) gets $\varepsilon = 0.4849$, and the record in the middle ($t = 5000$) gets $\varepsilon = 0.0707$. The first record gets $\varepsilon = 0.0481$, at the largest order $\alpha = 512$ of the grid. Its RDP curve is $\alpha/20{,}000$, so small ($\varepsilon(2) = 0.000100$) that the conversion term $\ln(1/\delta)/(\alpha - 1)$ stays as large as the RDP term even at the end of the grid: 0.0225 against $512/20{,}000 = 0.0256$ at $\alpha = 512$. A finer conversion from [From RDP to (ε, δ)](02-rdp-conversions.md) would give less.

The guarantee is per record and depends on position. A report must state the worst record's value, or randomize the order and the stopping time.

### Check yourself

- Why does the theorem need the loss to be convex and smooth, with $\eta \le 2/\beta$? *(So that each gradient step is a contraction. An expanding map could amplify a small shift faster than the noise can pay it off.)*
- If every iterate is released, what does the record at step 1 get? *(Only the noise of its own step: $\varepsilon(\alpha) = \alpha/(2\tilde\sigma^2)$, as for the last record. The later iterates reveal $w_1$ directly.)*

## Hidden state: privacy that stops growing {#sec-shuf-hidden}

### Context

Full-batch noisy GD uses every record at every step. Composition charges every step, so the RDP of $T$ steps grows linearly in $T$. Amplification by iteration does not help by itself: with every step adding a fresh shift $s$, the payments of Theorem 22 must cover $Ts$ in total, and the cheapest schedule pays $s$ at every step, which is composition again. Yet if the parameters live in a bounded set, two runs can never be further apart than its diameter, however long they run. So the shift never has to exceed the diameter, and only the last stretch of steps can matter. Can the privacy loss of the last iterate then stop growing with $T$? Altschuler & Talwar (2022) show that it can.

### Statement

The idea is to ignore everything that happened before some late step $\tau$: at that point the two runs are somewhere inside the set, at most $d_K$ apart, and only the steps after $\tau$ need to pay off that distance plus the new shifts. The cost then depends on how many steps are needed to pay, not on $T$.

**Theorem 3.1 of Altschuler & Talwar (2022), in shape.** For convex, $L_f$-Lipschitz, $\beta$-smooth losses on a convex set $K$ of diameter $d_K$, step size $\eta \le 2/\beta$, noise $\mathcal N(0, \sigma^2 I)$ added to the batch gradient, any batch size $b$ and $T$ steps, noisy SGD releasing only the last iterate is $(\alpha, \varepsilon)$-RDP with
$$\varepsilon \lesssim \frac{\alpha L_f^2}{n^2\sigma^2}\min\{T, \bar T\}, \qquad \bar T = \left\lceil\frac{d_Kn}{L_f\eta}\right\rceil,$$
for $\alpha$ up to a threshold and $\sigma > 8\sqrt2L_f/b$; $\lesssim$ hides a constant, and $\lceil\cdot\rceil$ rounds up. Their Theorem 4.1 shows the shape is tight up to constants. The bound grows like composition until $\bar T$ and then stays flat.

The key lemma is their Proposition 3.2. It is Theorem 22 started at step $\tau$ instead of 0, with an initial shift of $d_K$ instead of 0. In a contractive noisy iteration confined to a set of diameter $d_K$, with noise standard deviation $\sigma_t$ at step $t$, for any $\tau < T$ and payments $a_{\tau+1}, \dots, a_T$ whose unpaid shift $z_t = d_K + \sum_{i=\tau+1}^t(s_i - a_i)$ stays non-negative and ends at $z_T = 0$,
$$D_\alpha(X_T\|X'_T) \le \sum_{t = \tau + 1}^{T}\frac{\alpha a_t^2}{2\sigma_t^2}.$$
Specializing it to **full-batch noisy GD** with per-step shift $s$ and per-step noise standard deviation $\sigma$ (both in parameter units) gives the bound below. It takes the better of two strategies, starting the accounting at the beginning (cost $Ts^2$, composition) or $m$ steps before the end (cost $(d_K + ms)^2/m$), each times $\frac{\alpha}{2\sigma^2}$. This specialization is our derivation, below; the paper proves Theorem 3.1 by a different route.
$$\varepsilon(\alpha) \le \frac{\alpha}{2\sigma^2}\min\left\{Ts^2,\ \min_{m \in \{1, \dots, T\}}\frac{(d_K + ms)^2}{m}\right\} \le \frac{2\alpha d_Ks}{\sigma^2}\quad\text{when } d_K/s \text{ is an integer}.$$ {#eq-shuf-plateau}
For other $d_K/s$ the right-hand bound is $\frac{\alpha}{2\sigma^2}(4d_Ks + s^2)$ (step 5).

### Derivation

1. **Start the accounting late.** At step $\tau$ both runs lie in a set of diameter $d_K$, so they are at most $d_K$ apart, and $D_\alpha^{(d_K)}(X_\tau\|X'_\tau) = 0$: shifting $X_\tau$ by at most $d_K$ can turn it into $X'_\tau$ (couple the two runs and move each point onto its partner). This is the only place where boundedness enters.
2. **Run Theorem 22 from $\tau$.** The induction of Theorem 22, started at $\tau$ with shift $d_K$ and divergence 0 instead of shift 0, gives Proposition 3.2. Projections onto $K$ are contractions (@sec-shuf-iteration), so by @eq-shuf-contraction they cost nothing.
3. **Pay evenly over the last $m = T - \tau$ steps.** The shift to pay off is the initial $d_K$ plus $s$ per step, $d_K + ms$ in total. The even schedule $a = (d_K + ms)/m$ pays $s$ more than each step adds, so the unpaid shift falls linearly from $d_K$ to 0 and $z_t \ge 0$ holds. It costs $m\cdot\frac{\alpha a^2}{2\sigma^2} = \frac{\alpha(d_K + ms)^2}{2m\sigma^2}$.
4. **Or start at the beginning.** With $\tau = 0$ both runs start at the same point, the starting shift is 0, and the even schedule over all $T$ steps pays $s$ per step: $\frac{\alpha Ts^2}{2\sigma^2}$, exactly the composition of $T$ Gaussian steps. Take the better of the two choices.
5. **Optimize $m$.** A short window means few payments, each large; a long window means more shift to pay. Expanding the square, $(d_K + ms)^2/m = d_K^2/m + 2d_Ks + ms^2$, the sum of a term falling in $m$ and one growing in $m$. Over real $m > 0$ it is minimized where the two balance, at $m = d_K/s$ (setting the derivative $-d_K^2/m^2 + s^2$ to 0), where it equals $d_Ks + 2d_Ks + d_Ks = 4d_Ks$. When $d_K/s$ is an integer and $T \ge d_K/s$, that $m$ is allowed, so the bound never exceeds $\frac{\alpha}{2\sigma^2}\cdot 4d_Ks = \frac{2\alpha d_Ks}{\sigma^2}$, whatever $T$. Otherwise take $m = \lceil d_K/s\rceil$: writing $m = d_K/s + r$ with $0 < r < 1$, $d_K + ms = 2d_K + rs$, and $(d_K + ms)^2/m = 4d_Ks + r^2s^2/m < 4d_Ks + s^2$. The bound reaches its plateau when the composition branch catches up with it, $Ts^2 = 4d_Ks$, at $T = 4d_K/s$.
6. **Back to their parameters.** With $s = 2\eta L_f/n$ (the full-batch gradient is an average over $n$ records, so replacing one moves it by at most $2L_f/n$) and noise $\eta\sigma$ in parameter space, the plateau is $\frac{2\alpha d_K\cdot 2\eta L_f/n}{\eta^2\sigma^2} = \frac{\alpha L_f^2}{n^2\sigma^2}\cdot\frac{4d_Kn}{L_f\eta}$, the shape of Theorem 3.1 with $\bar T \propto d_Kn/(L_f\eta)$.

### Interpretation

**Why the bounded domain caps the loss.** Composition pays for every step because it assumes the adversary sees every iterate. The last iterate forgets: whatever happened before step $T - m$ is summarized by a starting point inside the set, and the starting points of the two runs differ by at most $d_K$. The last $m$ noisy steps can hide that difference for a cost that does not depend on $T$.

**The constant is loose.** To see how loose, take a linear loss on $K = [0, 1]$, with $s = 0.01$ and $\sigma = 0.1$ per step. The two runs are then random walks drifting in opposite directions and projected onto the interval (the construction behind Altschuler & Talwar's lower bound). The script computes their $D_2$ on a grid of 1000 cells (500 cells give 0.3779 against 0.3786 at $T = 2000$, so the discretization error is small). @tbl-shuf-plateau compares it with composition and with @eq-shuf-plateau. The walk's divergence flattens at 0.3786 after about 100 steps, the bound at 4.0000 after 400 steps (@fig-shuf-hidden, right). This walk is one pair of datasets with one starting point, so its divergence is a lower bound on the worst case: the shape of the bound is right, and its constant is at most $4.0000/0.3786 = 10.57$ times too large.

**Strongly convex losses converge faster.** A loss is $\lambda$-strongly convex when it curves upward at least as fast as $\frac\lambda2\|w\|^2$. Then each gradient step is a strict contraction by $1 - \eta\lambda$, so the shift shrinks by that factor every step even when nobody pays for it. The plateau is reached after about $1/(\eta\lambda)$ steps instead of $d_Kn/(L_f\eta)$, with no diameter needed. Chourasia, Ye & Shokri (2021, Corollary 1) had shown earlier, by a different route, that for full-batch noisy GD (Langevin dynamics) the RDP of the last iterate is at most $\frac{\alpha S_g^2}{\lambda\sigma^2n^2}(1 - e^{-\lambda\eta T/2})$ after $T$ steps, where $S_g$ is the sensitivity of the summed gradient and their noise is $\sqrt{2\eta}\sigma$ per step. As $T$ grows the bracket tends to 1, so the bound levels off. Ye & Shokri (2022) extended this to mini-batch SGD, concurrently with Altschuler & Talwar. The next section checks their limit against an exact computation.

**Only the last iterate.** All these results need **only the last iterate to be released**. Checkpoints, logged intermediate models or a model released after each epoch break the argument, because the adversary then sees the run before it has forgotten, and you are back to composition.

**Contractive steps.** They also need **contractive steps**, which require convexity. For non-convex losses, Annamalai (2024) constructs a loss for which the last iterate of DP-SGD leaks as much as all iterates together, so no hidden-state amplification holds for all non-convex losses. Empirically, Nasr et al. (2021) found that their strongest attacks match the DP-SGD bound when the adversary sees every step or can craft gradients, and fall short of it when the adversary only sees the final model. That gap is an observation about their attacks, not a guarantee.

Do not report a hidden-state $\varepsilon$ for deep learning.

| Steps $T$ | 10 | 100 | 400 | 1000 | 2000 |
| --- | --- | --- | --- | --- | --- |
| Composition, $\alpha Ts^2/(2\sigma^2)$ | 0.1000 | 1.0000 | 4.0000 | 10.0000 | 20.0000 |
| Bounded-domain bound, @eq-shuf-plateau | 0.1000 | 1.0000 | 4.0000 | 4.0000 | 4.0000 |
| Projected random walks, numerical | 0.0961 | 0.3693 | 0.3786 | 0.3786 | 0.3786 |

: $D_2$ of the last iterate of full-batch noisy GD on a linear loss over $K = [0, 1]$ ($d_K = 1$), with per-step shift $s = 0.01$ and noise $\sigma = 0.1$, by composition, by the bounded-domain bound, and computed numerically for the two projected random walks. {#tbl-shuf-plateau}

### Example: two thousand steps on the unit interval

Take the setting of @tbl-shuf-plateau: diameter $d_K = 1$, $s = 0.01$, $\sigma = 0.1$, so $d_K/s = 100$ is an integer. Each composed step costs $\frac{\alpha s^2}{2\sigma^2} = \frac{2 \times 0.0001}{2 \times 0.01} = 0.01$ at $\alpha = 2$, so composition charges $0.01T$. Step 5 gives the plateau $2\alpha d_Ks/\sigma^2 = 2\alpha \cdot \frac{0.01}{0.01} = 2\alpha$, which is 4.0000 at $\alpha = 2$, reached at $T = 4d_K/s = 400$ steps. After 2000 steps, composition charges 20.0000; the bounded-domain bound charges 4.0000, five times less, and for the hard instance of the projected random walks the value is below 0.4. Running ten times longer does not change the bound; under composition it would cost ten times more.

### Check yourself

- Why does an unconstrained linear loss get no hidden-state amplification? *(Without a bounded set or a strict contraction, the runs drift apart by $s$ per step forever: after $T$ steps the means differ by $Ts$, the variances are $T\sigma^2$, and $\mu = Ts/(\sigma\sqrt T) = \sqrt T s/\sigma$, the same as composition. The next section computes this.)*
- A training pipeline saves a checkpoint every epoch and publishes the best one. Does the plateau apply? *(No. The adversary may see any of the checkpoints, so the analysis must cover all of them; the published model is selected using all of them.)*

## An exact example: noisy GD on a quadratic {#sec-shuf-quadratic}

### Context

All the results above are bounds. In one dimension, with a quadratic loss, each step of noisy GD is a linear map plus Gaussian noise, so the last iterate is Gaussian on both datasets and its privacy can be computed exactly with Fact 3. The computation shows the plateau directly, shows that it needs contraction, and gives a yardstick for the strongly convex bounds.

### Statement

Consider the run $w_k = c\,w_{k-1} + b_k + \xi_k$, $\xi_k \sim \mathcal N(0, \sigma^2)$, for $k = 1, \dots, T$ from a fixed $w_0$. Here $c$ is the contraction factor, $b_k$ is a data-dependent offset and $\xi_k$ is the noise of step $k$. The offsets on $D$ and $D'$ differ by $s$ at every step $k \ge t$ and agree before: your record enters at step $t$ and stays. Unrolling the run, the last iterate is a weighted sum of all offsets and all noise draws, the step-$k$ term weighted by $c^{T-k}$. So it is Gaussian with the same variance on both datasets, and by Fact 3 its $\mu$ is the weighted sum of the offset differences divided by the standard deviation of the weighted noise. The last iterate is exactly $\mu$-GDP with
$$\mu_T = \frac{s}{\sigma}\cdot\frac{\sum_{k=t}^{T}c^{T-k}}{\sqrt{\sum_{j=0}^{T-1}c^{2j}}}, \qquad\text{and for } t = 1,\ c < 1:\qquad \lim_{T\to\infty}\mu_T = \frac{s}{\sigma}\sqrt{\frac{1 + c}{1 - c}}.$$ {#eq-shuf-last-iterate}
Releasing all iterates instead is $\sqrt{T - t + 1}\,s/\sigma$-GDP. For $c = 1$ and $t = 1$ the two agree.

This is full-batch noisy GD on the quadratic loss $\ell(w; x) = \frac\lambda2(w - x)^2$, whose gradient is $\lambda(w - x)$. Averaged over the records, the gradient is $\lambda(w - \bar x)$, where $\bar x$ is the mean of the records. So the step $w - \eta(\lambda(w - \bar x) + \text{noise})$ is $(1 - \eta\lambda)w + \eta\lambda\bar x$ plus noise: it has $c = 1 - \eta\lambda$ and offset $b = \eta\lambda\bar x$. Replacing one of $n$ records by one that differs by $\Delta x$ changes $\bar x$ by $\Delta x/n$, so $s = \eta\lambda\Delta x/n$. Starting the record at step $t > 1$ models a record that joins the training set late.

### Derivation

1. **Unroll.** Substituting each step into the next, by induction on $T$, $w_T = c^Tw_0 + \sum_{k=1}^Tc^{T-k}(b_k + \xi_k)$: each step's offset and noise are multiplied by $c$ once for every later step.
2. **The noise part** is a sum of independent Gaussians, so it is Gaussian with mean 0 and variance $\sigma^2\sum_kc^{2(T-k)} = \sigma^2\sum_{j=0}^{T-1}c^{2j}$, the same on $D$ and $D'$ (Fact 3).
3. **The mean part** differs between the datasets by $\sum_{k\ge t}c^{T-k}s$; the start $w_0$ and the offsets before step $t$ are the same in both runs and cancel.
4. **Gaussian pair.** By Fact 3 the pair is exactly $\mu$-GDP with $\mu$ = (mean difference)/(standard deviation), which is @eq-shuf-last-iterate.
5. **The limit.** For $t = 1$ and $c < 1$, both sums are geometric: $\sum_{k=1}^Tc^{T-k} = (1 - c^T)/(1 - c)$ and $\sum_{j=0}^{T-1}c^{2j} = (1 - c^{2T})/(1 - c^2)$. So $\mu_T = \frac{s(1 - c^T)/(1 - c)}{\sigma\sqrt{(1 - c^{2T})/(1 - c^2)}} \to \frac{s}{\sigma}\cdot\frac{\sqrt{1 - c^2}}{1 - c}$, since $c^T \to 0$. Writing $1 - c^2 = (1 - c)(1 + c)$, this is $\frac{s}{\sigma}\sqrt{\frac{(1 - c)(1 + c)}{(1 - c)^2}} = \frac{s}{\sigma}\sqrt{\frac{1 + c}{1 - c}}$.
6. **Composition.** Releasing all iterates is $T - t + 1$ Gaussian mechanisms, each $s/\sigma$-GDP given the past (the offset shifts that step's output by $s$ under noise $\sigma$), so by the composition rule of Fact 3 it is $\sqrt{T - t + 1}\,s/\sigma$-GDP. For $c = 1$, every power of $c$ is 1, and step 4 gives $\frac{(T - t + 1)s}{\sigma\sqrt T}$, which for $t = 1$ is $\sqrt Ts/\sigma$: hiding the state buys nothing without contraction.

### Interpretation

**Contraction is what forgets.** With $c < 1$, the offset of step $k$ reaches the end damped by $c^{T-k}$, while fresh noise keeps arriving. The signal saturates at the geometric limit $s/(1 - c)$ and the noise standard deviation at $\sigma/\sqrt{1 - c^2}$. With $c = 1$ both keep growing, the signal like $T$ and the noise standard deviation like $\sqrt T$, so $\mu$ grows like $\sqrt T$ (@fig-shuf-hidden, left).

**The plateau arrives after about $1/(1 - c)$ steps.** That is how long it takes $c^T$ to become small. With $c = 0.9$, $\mu$ is 4.3588 at $T = 100$ and its limit is $\sqrt{1.9/0.1} = \sqrt{19} = 4.3589$; with $c = 0.99$ it is 9.6101 at $T = 100$, 14.1061 at $T = 1000$, and the limit is $\sqrt{1.99/0.01} = \sqrt{199} = 14.1067$. In general, a gradient step with $\eta = 1/\beta$ on a $\lambda$-strongly convex, $\beta$-smooth loss contracts by $c = 1 - \lambda/\beta$, so $1/(1 - c)$ is the condition number $\beta/\lambda$, which is the threshold, up to logarithmic factors, that Altschuler & Talwar give for strongly convex losses.

**Recent records are hardly hidden.** A record that joins at step 41 of 50, with $c = 0.9$ and $s/\sigma = 1$, gets $\mu = 2.8391$ against $\sqrt{10} = 3.1623$ for releasing its 10 steps: only the last steps carry its signal, and a little earlier noise helps.

**How tight is the strongly convex bound?** In this example the limit of Chourasia, Ye & Shokri's Corollary 1, translated into the $s$, $\sigma$ and $c$ of this section, becomes $\frac{2\alpha s^2}{(1 - c)\sigma^2}$, while the exact limiting RDP is $\alpha\mu^2/2$ (Fact 3) with the limiting $\mu$ of @eq-shuf-last-iterate, that is, $\frac{\alpha}{2}\cdot\frac{s^2}{\sigma^2}\cdot\frac{1 + c}{1 - c}$. Their ratio is $\frac{1 + c}{4}$: at $c = 0.99$, $\alpha = 2$ and $s/\sigma = 1$ the exact value is $199 = 199.0000$ and the bound $4/0.01 = 400.0000$, a ratio of $1.99/4 = 0.4975$. The bound is within a factor of about 2.

**A check by simulation.** 200,000 runs with $c = 0.9$, $T = 50$ and $s/\sigma = 1$ give an estimated $\mu$ of 4.3381 against the exact 4.3365.

![Left: the exact $\mu$ of the last iterate of one-dimensional full-batch noisy GD with per-step shift equal to the noise standard deviation, against the number of steps $T$, for $c = 1$ (dashed, on top of the black composition line), $c = 0.99$ and $c = 0.9$. Right: $D_2$ of the last iterate for a linear loss on $[0, 1]$ with $s = 0.01$ and $\sigma = 0.1$ per step: composition, the bounded-domain bound @eq-shuf-plateau, and the numerical value for two projected random walks.](figures/shuffling_iteration_hidden_state.svg){#fig-shuf-hidden}

### Example: ten thousand steps with noise multiplier 10

Run $k = 10{,}000$ steps of full-batch noisy GD with noise multiplier $\sigma/s = 10$ on a quadratic with $c = 0.99$. Releasing every iterate is $\mu = \sqrt{10{,}000}/10 = 10.0000$-GDP, which is $\varepsilon = 91.8173$ at $\delta = 10^{-5}$ (from the $\delta(\varepsilon)$ curve of Fact 3): no guarantee at all. Releasing only the final model is, after this many steps, at the limit of @eq-shuf-last-iterate, $\mu = 0.1 \times 14.1067 = 1.4107$-GDP, $\varepsilon = 6.5534$. The noise is the same; the difference is entirely in what the adversary is allowed to see. Compare the DP-SGD running example of this part (under add/remove neighbours), where subsampling rather than hidden state does the work: there each step uses only 1% of the data, which is why noise multiplier 1 suffices.

### Check yourself

- In @eq-shuf-last-iterate, why does the start $w_0$ not matter? *(It enters both runs as the same constant $c^Tw_0$, so it cancels in the mean difference and adds no variance.)*
- Doubling the noise $\sigma$ in this example: what happens to the plateau $\mu$? *(It halves: $\mu$ is proportional to $s/\sigma$ at every $T$.)*

## Code: check every number by computation {#sec-shuf-code}

The script [code/shuffling_iteration.py](code/shuffling_iteration.py) computes every number in this chapter and draws every figure. Shuffled randomized response and the clone pair are computed exactly as discrete distributions in log space, with $\delta(\varepsilon)$ read from sorted prefix sums (Fact 1). The iteration results are closed forms for Gaussian pairs, checked by Monte Carlo, plus one grid computation for the projected random walks. From the repository root:

```bash
uv run python book/part04-privacy-accounting/code/shuffling_iteration.py
```

**Output** (seed 0), summarized in @tbl-shuf-code:

| Experiment | Result |
| --- | --- |
| Shuffled randomized response, $n = 3$, $\varepsilon_0 = \ln 3$ | $\delta(0.5) = 0.1900$ against 0.3378 for one report; simulation within 0.0015 of the exact count distribution |
| Worst dataset pair, all $m$ at $n = 1000$, $m \le 30$ at $n = 10^4, 10^5$ | The pair where the others all hold 0 is the worst for $\varepsilon_0 = 1, 4$ at $n = 1000$, and within $4.5 \times 10^{-5}$ of the largest $\varepsilon$ found everywhere |
| Clone decomposition, $\varepsilon_0 = \ln 3$ | Clone part $(0.1667, 0.1667)$, leftover $(0.875, 0.125)$ |
| Shuffling at $\delta = 10^{-6}$ | @tbl-shuf-compare; numeric clone bound 1.44 to 2.36 times the exact value |
| Shift reduction, $z = 1$, $\sigma = 1$, $\alpha = 2$ | Even schedule equals the exact $D_2$ at $T = 1, 2, 4, 10$ |
| One pass over $10^4$ records, $\tilde\sigma = 1$ | $\varepsilon$ from 0.0481 (first record) to 5.2985 (last) at $\delta = 10^{-5}$ |
| Bounded domain, $d_K = 1$, $s = 0.01$, $\sigma = 0.1$ | @tbl-shuf-plateau; plateau 4.0000 against 0.3786 for the projected random walks |
| Quadratic, $s/\sigma = 1$ | Limits 4.3589 ($c = 0.9$) and 14.1067 ($c = 0.99$); Monte Carlo 4.3381 against 4.3365 |

: The script's output with the default seed. {#tbl-shuf-code}

[tests/test_shuffling_iteration.py](../../tests/test_shuffling_iteration.py) checks that the output still matches.

**What to take from it.** **Shuffling amplifies like $1/\sqrt n$,** and the constant matters: the exact value for binary randomized response is about half the best general bound and a quarter to an eighth of the closed form. Since **exact computations are cheap** (the count distributions have $n + 1$ points, and even $n = 10^5$ takes milliseconds), do not settle for a closed form when the exact value is at hand. On the iteration side, **iteration amplifies only what is hidden**: the same noisy run is worth $\mu = 10$ or $\mu = 1.41$ depending on whether intermediate iterates are released. And **hidden-state gains need contraction or a bounded domain**; without either, the last iterate is as revealing as the whole trajectory.

## References

- Feldman, McMillan & Talwar, "Hiding Among the Clones: A Simple and Nearly Optimal Analysis of Privacy Amplification by Shuffling" (FOCS 2021; arXiv:2012.12803). Theorems 3.1 and 3.2 and Lemmas 3.3–3.5 are the clone bound and reduction used here; Section 6 gives the numerical method, and their code computes it. Also covers approximate-DP randomizers (Theorem 3.8) and RDP of shuffled mechanisms.
- Erlingsson, Feldman, Mironov, Raghunathan, Talwar & Thakurta, "Amplification by Shuffling: From Local to Central Differential Privacy via Anonymity" (SODA 2019; arXiv:1811.12469). The first amplification-by-shuffling theorem (Theorem 7 of the arXiv version), for adaptive randomizers; read it for the swap argument and for the encode–shuffle–analyze motivation.
- Balle, Bell, Gascón & Nissim, "The Privacy Blanket of the Shuffle Model" (CRYPTO 2019). Writes each randomizer as a mixture with an input-independent "blanket" and derives amplification bounds from it, including a closed form tailored to $k$-ary randomized response.
- Cheu, Smith, Ullman, Zeber & Zhilyaev, "Distributed Differential Privacy via Shuffling" (EUROCRYPT 2019). An early formal study of the shuffle model; its analysis of shuffled binary randomized response already has the $e^{\varepsilon_0/2}/\sqrt n$ dependence.
- Kairouz, Oh & Viswanath, "The Composition Theorem for Differential Privacy" (ICML 2015). Shows that every $\varepsilon_0$-DP randomizer on two inputs is a post-processing of binary randomized response, the fact used in step 4 of the clone derivation.
- Feldman, Mironov, Talwar & Thakurta, "Privacy Amplification by Iteration" (FOCS 2018; arXiv:1808.06651). Shifted Rényi divergence, the shift-reduction lemma (Lemma 20), the contraction lemma (Lemma 21), Theorem 22, and noisy SGD (Theorems 23, 24 and 26).
- Altschuler & Talwar, "Privacy of Noisy Stochastic Gradient Descent: More Iterations without More Privacy Loss" (NeurIPS 2022; arXiv:2205.13710). The bounded-domain plateau (Proposition 3.2, Theorem 3.1; the full-batch form @eq-shuf-plateau is this chapter's specialization of Proposition 3.2), a matching lower bound (Theorem 4.1), and extensions to strongly convex losses and cyclic batches.
- Chourasia, Ye & Shokri, "Differential Privacy Dynamics of Langevin Diffusion and Noisy Gradient Descent" (NeurIPS 2021; arXiv:2102.05855). Converging RDP of the last iterate of noisy GD for smooth strongly convex losses (Corollary 1), proved through Langevin diffusion and log-Sobolev inequalities.
- Ye & Shokri, "Differentially Private Learning Needs Hidden State (Or Much Faster Convergence)" (NeurIPS 2022; arXiv:2203.05363). Extends the converging bound to mini-batch noisy SGD with shuffle-and-partition and sampling without replacement.
- Nasr, Song, Thakurta, Papernot & Carlini, "Adversary Instantiation: Lower Bounds for Differentially Private Machine Learning" (IEEE S&P 2021; arXiv:2101.04535). Attacks that match the DP-SGD bound for strong adversaries and fall short when the adversary sees only the final model.
- Annamalai, "It's Our Loss: No Privacy Amplification for Hidden State DP-SGD With Non-Convex Loss" (ACM Workshop on Artificial Intelligence and Security, AISec 2024; arXiv:2407.06496). A non-convex loss for which the last iterate leaks as much as all iterates.

## Cheat sheet

@tbl-shuf-cheat collects the results. $\varepsilon_0$ is the local level, $n$ the number of users or records, $\tilde\sigma$ the noise multiplier, $s$ and $\sigma$ the per-step shift and noise in parameter units, $c$ the contraction factor, $d_K$ the diameter. All use replace-one neighbours.

| Result | Statement | Tightness |
| --- | --- | --- |
| Shuffled binary randomized response | Count pair @eq-shuf-rr-pair; $\mu \approx 2\sinh(\varepsilon_0/2)/\sqrt n$ | Exact for the pair where the others hold 0, within $10^{-4}$ of the worst pair found; normal approximation fails for few 1-reports |
| Clone reduction (Thm 3.2) | Output is a post-processing of $(A + \Delta, C - A + 1 - \Delta)$ vs $(A + 1 - \Delta, C - A + \Delta)$, $C \sim \mathrm{Bin}(n-1, e^{-\varepsilon_0})$ | Numeric: 1.44–2.36 × exact for binary randomized response (≈ $2/(1 + e^{-\varepsilon_0})$ with many clones) |
| Clone bound (Thm 3.1) | @eq-shuf-clone-bound, needs $\varepsilon_0 \le \ln\frac{n}{16\ln(2/\delta)}$ | 4.47–7.55 × exact |
| Shift reduction | $D_\alpha^{(z)}(X + Z\|X' + Z) \le D_\alpha^{(z+a)}(X\|X') + \frac{\alpha a^2}{2\sigma^2}$ | Exact for translations from a common point |
| One-pass noisy SGD (Thm 23) | $\varepsilon_t(\alpha) = \frac{\alpha}{2\tilde\sigma^2(n - t + 1)}$ for the record at step $t$, last iterate only | Exact after a reset; loose for strict contractions |
| Bounded domain (from Prop. 3.2) | $\varepsilon(\alpha) \le \frac{\alpha}{2\sigma^2}\min\{Ts^2, \min_m(d_K + ms)^2/m\} \le \frac{2\alpha d_Ks}{\sigma^2}$ ($d_K/s$ an integer) | Right shape; constant at most about 10 × too large in the example |
| Quadratic, exact | $\mu_T$ of @eq-shuf-last-iterate; limit $\frac s\sigma\sqrt{\frac{1+c}{1-c}}$ | Exact; strongly convex bound within a factor $\frac{4}{1+c}$ |

: The results of this chapter. {#tbl-shuf-cheat}

**Rules of thumb**

- Shuffling turns $\varepsilon_0$-local into central $\varepsilon \approx (\text{a constant growing like } e^{\varepsilon_0/2})\cdot\sqrt{\ln(1/\delta)/n}$. It never lowers the pure-DP level.
- For binary randomized response, compute the exact value. For other randomizers, compute the numeric clone bound; use the closed form only to reason.
- Amplification by iteration needs contractive steps (convex, smooth, $\eta \le 2/\beta$) and a hidden state. The record used last gets no amplification.
- With a bounded domain or strong convexity, the last iterate's privacy plateaus; without them it grows like composition.
- Never claim hidden-state privacy for non-convex training or when checkpoints are released.

**Practice.** Exercises for this chapter, each with a worked solution, are in [the exercise book](solutions/07-shuffling-iteration.md).
