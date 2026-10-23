# Amplification by subsampling

## Why this matters

Differential privacy (DP) for deep learning would be useless without one trick. Stochastic gradient descent (SGD) with per-example clipping and Gaussian noise, DP-SGD, runs ten thousand steps or more. Each step is a Gaussian mechanism: it clips every example's gradient, sums them and adds Gaussian noise whose standard deviation is the **noise multiplier** $\sigma$ times the clipping norm. Composing ten thousand such steps with $\sigma = 1$, and paying for every record at every step, gives $\varepsilon = 5425.5$ at $\delta = 10^{-5}$, as [Rényi DP](01-renyi-dp.md) computes: a privacy loss nobody would report.

The trick is that each step only looks at a small random batch. If a record is in the batch only with probability $q$, an adversary who sees the step's output must first guess whether the record was used at all. That uncertainty is worth a lot. With $q = 0.01$, it turns a step that costs $\varepsilon = 1$ into one that costs about $0.017$. In the language of Rényi differential privacy (RDP), which describes a mechanism by a curve $\varepsilon(\alpha)$ giving its privacy cost at each order $\alpha > 1$, it turns the curve of a Gaussian step from $\alpha/2$ into something of order $10^{-4}\alpha$ for small orders.

This chapter answers three questions. How much privacy does random sampling buy, exactly? Which sampling scheme does the answer hold for? And how do you compute it for the Gaussian steps of DP-SGD?

**Notation.** In this chapter $q$ is always the **sampling rate**, never a density. Output distributions are capital letters, as in the rest of this part, and the same letter names a distribution and its density. $P_0$ is a step's output without the record and $P_1$ its output with it. For the subsampled Gaussian, $P_0 = \mathcal N(0, \sigma^2)$, $P_1 = \mathcal N(1, \sigma^2)$, and the output with the record sampled at rate $q$ is the mixture $P_q = (1 - q)P_0 + qP_1$: with probability $1 - q$ the record is left out and the output is drawn from $P_0$, with probability $q$ it is used and the output is drawn from $P_1$. The mixture lemma of the toolkit mixes three generic distributions $R$, $P$, $P'$. Its weight is $q$, the probability that the record is used. An output is $y$.

### The one idea: the output is a mixture, and mixtures are closer

Fix a dataset $D$ and add one record $x$ to get $D' = D \cup \{x\}$. Under **Poisson sampling**, each record enters the batch independently with probability $q$, by its own coin flip. Condition on which records of $D$ were drawn. On $D$, the mechanism then outputs some distribution $P_0$. On $D'$, it outputs $P_1$ if $x$ was drawn and $P_0$ otherwise. Since $x$ is drawn with probability $q$, the output on $D'$ is the **mixture**, compared with the output on $D$:
$$(1 - q)P_0 + qP_1 \quad\text{against}\quad P_0.$$
With probability $1 - q$ the two runs are identical. The difference between the runs is diluted by a factor $q$, and every likelihood ratio, the ratio of the two densities at an output $y$ that measures how much more likely $y$ is in one run than in the other, is pulled toward 1. Divide the mixture by $P_0$ to see it: the ratio $P_1/P_0$ becomes $(1 - q) + q\,P_1/P_0$, a weighted average of 1 and the old ratio. All of DP-SGD's privacy rests on this dilution, and the dilution depends on how the batch was drawn. A different sampling scheme gives a different mixture, and an accountant that assumes one scheme while the training code runs another reports a number that does not apply.

@tbl-sub-overview lists the results of the chapter and where each is used.

| Result | Where it is used |
| --- | --- |
| The mixture lemma | Every amplification statement in this chapter |
| $(\varepsilon, \delta)$ amplification, Poisson sampling, add/remove | One-step guarantees; the naive DP-SGD accountant |
| The same for a fixed-size batch, replace-one | Accounting for fixed-size batches; why the scheme must match the neighbouring relation |
| RDP of the Poisson-subsampled Gaussian | The RDP accountant for DP-SGD, and the input of [numerical accounting](06-numerical-accounting.md) |
| Small-$q$ asymptotics, the moments accountant | Quick estimates; reading Abadi et al. (2016) |
| The RDP accountant for DP-SGD, against per-step composition | The running example; [Choosing an accountant](09-choosing-an-accountant.md) |
| No tight generic RDP amplification | Why accountants special-case the Gaussian |

: The results of this chapter and where each is used. {#tbl-sub-overview}

**How to read this chapter.** Read the toolkit and @sec-sub-eps-delta, then @sec-sub-schemes on sampling schemes. It holds the warning that matters most in practice. @sec-sub-rdp-gaussian is the computation every DP-SGD accountant performs; do its $\alpha = 2$ walk-through by hand. @sec-sub-small-q and @sec-sub-generic can wait for a second reading.

The chapter uses the Rényi divergence $D_\alpha$ and the hockey-stick divergence $E_\gamma$ from [Divergences](../part01-prerequisites/03-divergences.md), RDP and its composition from [Rényi DP](01-renyi-dp.md), and the conversions to $(\varepsilon, \delta)$ from [From RDP to (ε, δ)](02-rdp-conversions.md). Each is restated where it is used.

## Toolkit: three facts every derivation uses

### Fact 1 — $(\varepsilon, \delta)$ is a statement about events, and averaging preserves it

A mechanism is $(\varepsilon, \delta)$-DP when, for every pair of neighbouring datasets taken in both orders, the output distributions $P$ and $Q$ satisfy $P(A) \le e^\varepsilon Q(A) + \delta$ for every event $A$ ([Definition](../part03-dp-foundations/01-definition.md)): no set of outputs becomes more than $e^\varepsilon$ times more likely, except for a slack $\delta$. The **hockey-stick divergence** packs this into one number. It is the largest amount by which $P$ exceeds $\gamma$ times $Q$ on any event, $E_\gamma(P\|Q) = \max_A[P(A) - \gamma Q(A)]$, so the DP condition for the pair is exactly $E_{e^\varepsilon}(P\|Q) \le \delta$.

Because the condition is a bound on each event separately, it survives averaging. Suppose $P = \sum_t w_tP_t$ and $Q = \sum_t w_tQ_t$ are averages with the same weights $w_t$, for instance over some random choice $t$ that both runs share. For every event the excess is the average of the excesses, $P(A) - \gamma Q(A) = \sum_tw_t[P_t(A) - \gamma Q_t(A)]$. If each bracket is at most $\delta$, so is their weighted average. So $E_\gamma(P\|Q) \le \max_tE_\gamma(P_t\|Q_t)$. This is **joint convexity** of $E_\gamma$, restricted to a shared mixing variable.

The subsampled mechanism is exactly such an average, over which other records were drawn. Fact 1 lets you fix those records and prove the bound for one draw.

### Fact 2 — The mixture lemma

This is the fact that does the amplifying. Picture two runs whose outputs are mixtures with the same weight $q$: with probability $1 - q$ both draw from a common distribution $R$, where the runs agree, and with probability $q$ the first draws from $P$ and the second from $P'$, where they may differ. Where they agree the likelihood ratio is 1; where they differ it is at most about $e^\varepsilon$. The mixture's ratio should then be at most the weighted average $(1 - q)\cdot 1 + q e^\varepsilon = 1 + q(e^\varepsilon - 1)$, and the slack $\delta$, which only enters through the differing part, should shrink to $q\delta$. The lemma makes this precise, with one subtlety: $P$ must be close not only to $P'$ but also to $R$, because a high-probability event of $P$ could otherwise be one that $R$ never produces.

Let $R$, $P$, $P'$ be three distributions, $\varepsilon \ge 0$, $\delta \in [0, 1]$ and $q \in [0, 1]$. Suppose that for every event $A$,
$$P(A) \le e^\varepsilon P'(A) + \delta \qquad\text{and}\qquad P(A) \le e^\varepsilon R(A) + \delta.$$
Then for every event $A$,
$$(1 - q)R(A) + qP(A) \le e^{\varepsilon'}\big[(1 - q)R(A) + qP'(A)\big] + q\delta, \qquad e^{\varepsilon'} = 1 + q(e^\varepsilon - 1).$$ {#eq-sub-mixture-lemma}

So the $\varepsilon$ shrinks to $\varepsilon'$ and the $\delta$ to $q\delta$. This is an $(\varepsilon, \delta)$ form of what Balle, Barthe & Gaboardi (2018, Theorem 2) call **advanced joint convexity**.

*Proof.* The plan is to treat the three probabilities of $A$ as unknowns, see that the gap between the two sides gets larger only when $P'(A)$ and $R(A)$ get smaller, and push them down to the smallest values the assumptions allow. Fix $A$ and write $a = P(A)$, $b = P'(A)$, $c = R(A)$, and $F$ for the left side minus the right side without the $q\delta$. The claim is $F \le q\delta$.

1. *Expand.* The $(1-q)c$ on the left and the $e^{\varepsilon'}(1 - q)c$ on the right combine to $(1 - q)c(1 - e^{\varepsilon'})$, and $1 - e^{\varepsilon'} = -q(e^\varepsilon - 1)$. So
$$F = qa - q(1 - q)(e^\varepsilon - 1)\,c - q\big(1 + q(e^\varepsilon - 1)\big)\,b.$$
2. *Signs.* The coefficients of $b$ and $c$ are negative, so $F$ is largest when $b$ and $c$ are as small as the assumptions allow. Solving each assumption for the smaller side gives $b \ge e^{-\varepsilon}(a - \delta)$ and $c \ge e^{-\varepsilon}(a - \delta)$, and both are also $\ge 0$, being probabilities. Which of the two lower bounds binds depends on whether $a$ exceeds $\delta$.
3. *Case $a \le \delta$.* Drop the negative terms: $F \le qa \le q\delta$.
4. *Case $a > \delta$.* Put $b = c = e^{-\varepsilon}(a - \delta)$ into step 1. The two coefficients add up to $(1 - q)(e^\varepsilon - 1) + 1 + q(e^\varepsilon - 1) = e^\varepsilon$, so $F \le qa - q e^{-\varepsilon}(a - \delta)e^\varepsilon = q\delta$.

The script checks the lemma on random examples. It draws 20,000 random triples $(R, P, P')$ on four outcomes, computes the smallest $\delta$ for which the two assumptions hold in both directions at $\varepsilon = 1$, and checks @eq-sub-mixture-lemma with $q = 0.1$. For each triple it divides the left side's excess over $e^{\varepsilon'}$ times the bracket by $q\delta$. The largest value of this ratio is 1.0000: the lemma holds, and some triples meet it with equality to four digits, so it cannot be improved.

### Fact 3 — Group privacy

A guarantee for datasets one record apart also covers datasets two records apart, at a price. If a mechanism is $(\varepsilon, \delta)$-DP for datasets one step apart, apply the definition twice along the chain $D_0, D_1, D_2$: $P_0(A) \le e^\varepsilon P_1(A) + \delta \le e^\varepsilon\big(e^\varepsilon P_2(A) + \delta\big) + \delta = e^{2\varepsilon}P_2(A) + (1 + e^\varepsilon)\delta$. So it is $(2\varepsilon, (1 + e^\varepsilon)\delta)$-DP for datasets two steps apart ([Properties](../part03-dp-foundations/02-properties.md)). Replacing a record is a removal followed by an addition, so a guarantee for add/remove neighbours gives one for replace-one neighbours at this price.

## Amplification for (ε, δ)-DP under Poisson sampling {#sec-sub-eps-delta}

### Context

You have a mechanism $M$ that is $(\varepsilon, \delta)$-DP for add/remove neighbours, datasets that differ by one record being present in one and absent from the other. You run it on a Poisson sample of the data, each record kept with probability $q$. What are the privacy parameters of the combined mechanism? This is the oldest amplification result. Kasiviswanathan et al. (2008) used it to learn privately from a sample, and the tight form below is due to Li, Qardaji & Su (2012) and Balle, Barthe & Gaboardi (2018).

### Statement

The answer is the mixture lemma with the sampling rate as the weight: the likelihood ratio becomes a weighted average of 1 and $e^\varepsilon$, and $\delta$ is paid only when the record is drawn. Let $M$ be $(\varepsilon, \delta)$-DP for add/remove neighbours and let $M \circ S_q$ run $M$ on a Poisson sample with rate $q$. Then $M \circ S_q$ is $(\varepsilon', \delta')$-DP for add/remove neighbours with
$$\varepsilon' = \ln\big(1 + q(e^\varepsilon - 1)\big), \qquad \delta' = q\delta.$$ {#eq-sub-amplify}
The bound is tight: for every $\varepsilon$ and $q$ some mechanism attains it (Balle, Barthe & Gaboardi 2018, Theorems 8 and 13).

### Derivation

The idea is to fix every random choice except whether $x$ is drawn, apply the mixture lemma once for each order of the pair, and then average the fixed choices back in with Fact 1. Let $D' = D \cup \{x\}$.

1. **Condition on the rest of the sample.** Let $\tilde D$ be the Poisson sample of $D$. Because each record has its own coin, the sample of $D'$ is $\tilde D$ or $\tilde D \cup \{x\}$, the second with probability $q$, independently of $\tilde D$. Given $\tilde D$, write $P_0 = M(\tilde D)$ and $P_1 = M(\tilde D \cup \{x\})$. Then $M \circ S_q(D)$ given $\tilde D$ is $P_0$, and $M \circ S_q(D')$ given $\tilde D$ is $(1 - q)P_0 + qP_1$.
2. **Direction "$D'$ against $D$".** Here the mixture is on the left. Apply the mixture lemma (Fact 2) with $R = P_0$, $P = P_1$, $P' = P_0$ and weight $q$. Both of its assumptions then read $P_1(A) \le e^\varepsilon P_0(A) + \delta$, which holds because $\tilde D$ and $\tilde D \cup \{x\}$ are neighbours and $M$ is DP. The right side of @eq-sub-mixture-lemma becomes $e^{\varepsilon'}[(1 - q)P_0(A) + qP_0(A)] + q\delta = e^{\varepsilon'}P_0(A) + q\delta$. This is the **remove** direction in the naming of [Numerical accounting](06-numerical-accounting.md): the dataset with the record comes first, and its neighbour is obtained by removing $x$.
3. **Direction "$D$ against $D'$".** Now the mixture is on the right. Apply the lemma with $R = P_0$, $P = P_0$, $P' = P_1$. The assumptions are $P_0(A) \le e^\varepsilon P_1(A) + \delta$, which is DP of $M$ again, and $P_0(A) \le e^\varepsilon P_0(A) + \delta$, which is always true. The left side is $(1 - q)P_0(A) + qP_0(A) = P_0(A)$ and the right side $e^{\varepsilon'}[(1 - q)P_0(A) + qP_1(A)] + q\delta$. This is the **add** direction.
4. **Average over $\tilde D$.** Both bounds hold for every $\tilde D$ with the same $(\varepsilon', q\delta)$, and both runs average over $\tilde D$ with the same weights, so they hold for the averages over $\tilde D$ (Fact 1).

**Tightness.** To show $\varepsilon'$ cannot be smaller, you need a mechanism whose likelihood ratio actually reaches $1 + q(e^\varepsilon - 1)$. Randomized response does it. It reports a bit truthfully with probability $e^\varepsilon/(1 + e^\varepsilon)$ and flipped otherwise ([Randomized response](../part03-dp-foundations/01-definition.md#sec-dpdef-rr)), so the probability of any report changes by a factor of exactly $e^\varepsilon$ when the true bit changes. Let $M$ apply it to the bit "the sample contains a flagged record". Take $D$ with no flagged record and $x$ flagged. Then the bit is 0 whenever $x$ is not drawn and 1 whenever it is, so $P_0$ and $P_1$ are the same for every $\tilde D$, and $P_1(1)/P_0(1) = e^\varepsilon$. The likelihood ratio of the mixture to $P_0$ at the output "1" is
$$\frac{(1 - q)P_0(1) + qP_1(1)}{P_0(1)} = (1 - q) + q\frac{P_1(1)}{P_0(1)} = 1 + q(e^\varepsilon - 1).$$
With $\delta = 0$ the event $\{1\}$ forces $e^{\varepsilon'} \ge 1 + q(e^\varepsilon - 1)$, so $\varepsilon'$ in @eq-sub-amplify cannot be improved.

### Interpretation

The formula behaves differently for small and large $\varepsilon$. For small $\varepsilon$, amplification is linear in $q$. When $\varepsilon \ll 1$, $e^\varepsilon - 1 \approx \varepsilon$ and $\ln(1 + y) \approx y$ for small $y$, so $\varepsilon' \approx q\varepsilon$. A 1% sample makes each step a hundred times cheaper.

For large $\varepsilon$, amplification only subtracts. When $qe^\varepsilon \gg 1$, the 1 inside the logarithm is negligible and $1 + q(e^\varepsilon - 1) \approx qe^\varepsilon$, so $\varepsilon' \approx \ln(qe^\varepsilon) = \varepsilon - \ln(1/q)$. Sampling can never remove more than $\ln(1/q)$ nats, because in the worst case the record was drawn, and that happens with probability $q$. At $q = 0.01$ and $\varepsilon = 10$, @tbl-sub-amp gives 5.3993, close to $10 - \ln 100 = 5.395$. @fig-sub-amp shows the switch between the two regimes around $\varepsilon \approx \ln(1/q)$, where $qe^\varepsilon$ reaches 1.

The $\delta$ always shrinks by $q$, because the failure event of $M$ only matters when the record was drawn. Finally, @eq-sub-amplify is a statement about one step, and it is tight for one step. Composing per-step $(\varepsilon', \delta')$ pairs is not tight, and the running example shows by how much (@sec-sub-dpsgd).

![The subsampled $\varepsilon'$ against the base $\varepsilon$ for three sampling rates, with the small-$\varepsilon$ approximation $q\varepsilon$ dotted and no sampling dashed. Amplification is a factor $q$ for small $\varepsilon$ and only a shift by $\ln(1/q)$ for large $\varepsilon$.](figures/subsampling_amplification.svg){#fig-sub-amp}

### Example: randomized response, sampled at 10%

Take $\varepsilon = 1$ and $q = 0.1$, and the randomized-response mechanism of the tightness argument. When the record is not drawn the true bit is 0, which is reported as 0 with probability $e/(1 + e) = 0.7311$ and as 1 with probability $1/(1 + e) = 0.2689$; when it is drawn the two probabilities swap. The mixture weights these by 0.9 and 0.1: for the output 1, $0.9 \times 0.2689 + 0.1 \times 0.7311 = 0.3152$. @tbl-sub-rr shows both outputs and the likelihood ratios in both directions.

| Output $y$ | $P_0(y)$, record not drawn | $P_1(y)$, record drawn | Mixture $0.9P_0 + 0.1P_1$ | Mixture $/\ P_0$ | $P_0\ /$ mixture |
| --- | --- | --- | --- | --- | --- |
| 0 | 0.7311 | 0.2689 | 0.6848 | 0.9368 | 1.0675 |
| 1 | 0.2689 | 0.7311 | 0.3152 | 1.1718 | 0.8534 |

: Randomized response with $\varepsilon = 1$ on a 10% Poisson sample. The likelihood ratios of the unsampled mechanism are $e^{\pm1}$; sampling pulls them to at most 1.1718. {#tbl-sub-rr}

Without sampling the ratios are $e = 2.718$ and $1/e$. With sampling the largest is $1.1718 = 1 + 0.1(e - 1)$, so $\varepsilon' = \ln 1.1718 = 0.1586$, as @eq-sub-amplify says, against $q\varepsilon = 0.1$. The other direction's largest ratio, 1.0675, is smaller: removing the record from a mixture changes less than adding it, because the mixture never puts less than $0.9$ times $P_0$'s probability on any output. @tbl-sub-amp gives $\varepsilon'$ for more cases.

| $q$ | $\varepsilon = 0.1$ | $\varepsilon = 1$ | $\varepsilon = 5$ | $\varepsilon = 10$ |
| --- | --- | --- | --- | --- |
| 0.01 | 0.0011 | 0.0170 | 0.9059 | 5.3993 |
| 0.1 | 0.0105 | 0.1586 | 2.7563 | 7.6978 |
| 0.5 | 0.0512 | 0.6201 | 4.3136 | 9.3069 |

: The amplified $\varepsilon' = \ln(1 + q(e^\varepsilon - 1))$ for three sampling rates. {#tbl-sub-amp}

### Check yourself

- At $q = 0.01$, roughly how large must $\varepsilon$ be before amplification stops being a factor of 100? *(Around $\ln(1/q) = 4.6$, where $q(e^\varepsilon - 1)$ reaches 1. At $\varepsilon = 5$, @tbl-sub-amp gives $\varepsilon' = 0.9059$, not 0.05.)*
- Why does the derivation need $\tilde D$ and $x$ to be drawn independently? *(So that, given $T$, the record is included with probability exactly $q$. The mixture weights must not depend on the rest of the sample.)*

## The sampling scheme must match the neighbouring relation {#sec-sub-schemes}

### Context

Poisson sampling gives batches of random size. Most training code draws fixed-size batches: either a random batch of $m$ out of $n$ records at each step, or a random permutation of the data cut into $n/m$ consecutive batches per epoch (**shuffle and partition**). Does the amplification of @eq-sub-amplify still apply? Only if you change what you compare. A fixed batch size pairs naturally with **replace-one** neighbours, datasets of the same size that differ in one record's value, because with a fixed batch size adding a record would change the size. Accounting for one scheme while running another is a bug that can multiply $\varepsilon$ several times over.

### Statement

**Fixed-size batches, replace-one.** A uniformly random batch of $m$ out of $n$ contains a given record with probability $m/n$, so $m/n$ plays the role of $q$, and the same formula holds, as long as the base guarantee is for replace-one neighbours. Draw a uniformly random batch of $m$ records from $n$, and let $M$ be $(\varepsilon, \delta)$-DP for replace-one neighbours on datasets of size $m$. Then the subsampled mechanism is $(\varepsilon', \delta')$-DP for replace-one neighbours on datasets of size $n$, with
$$\varepsilon' = \ln\Big(1 + \frac mn(e^\varepsilon - 1)\Big), \qquad \delta' = \frac mn\delta$$ {#eq-sub-wor}
(Balle, Barthe & Gaboardi 2018, Theorem 9). The formula is that of @eq-sub-amplify with $q = m/n$, but the base guarantee is a different one.

**The matching rule.** Each combination of sampling scheme and neighbouring relation produces its own pair of output distributions, and needs its own base guarantee. @tbl-sub-matching lists them. Its last column applies them to DP-SGD's noisy gradient sum, where every gradient is clipped to norm $C$ and the noise has standard deviation $\sigma C$; what matters is how far one record can move the sum, measured in units of the noise. The zero-out relation in the last row compares a dataset with the same dataset in which the target's gradient is set to 0.

| Sampling | Neighbours | Pair to account for | Base guarantee needed | Gaussian sum, clipping norm $C$ |
| --- | --- | --- | --- | --- |
| Poisson, rate $q$ | add/remove | $(1-q)P_0 + qP_1$ against $P_0$ | add/remove | means differ by $C$: noise multiplier $\sigma$ |
| Fixed size $m$ of $n$ | replace-one | $(1-q)R + qP$ against $(1-q)R + qP'$, $q = m/n$ | replace-one | means differ by $2C$: noise multiplier $\sigma/2$ |
| Fixed size $m$ of $n$ | add/remove | dominated by the Poisson pair with noise multiplier $\sigma/2$ (Lebeda et al. 2024, Proposition 9) | replace-one | $\sigma/2$ |
| Poisson, rate $q$ | replace-one | both sides are mixtures, base pair two steps apart | $(2\varepsilon, (1 + e^\varepsilon)\delta)$ by group privacy | means differ by up to $2C$ |
| Shuffle and partition | zero-out (the target's gradient set to 0) | the record is in exactly one batch per epoch | no simple formula; see below | measured, not derived |

: What to account for under each combination of sampling scheme and neighbouring relation. {#tbl-sub-matching}

### Derivation

**@eq-sub-wor by a coupling.** A fixed-size batch is not drawn record by record, so there is no independent coin for the differing record. The trick, called a **coupling**, is to build the batch in a way that has such a coin, and then to use the same coin and the same other records for both datasets. Let $D$ and $D'$ differ in record $i$: $x$ in $D$, $x'$ in $D'$.

1. **Build a batch that avoids $i$ in two stages.** Draw $B$, a uniform $(m-1)$-subset of the other $n - 1$ records, then $z$, a uniform record from the $(n - 1) - (m - 1) = n - m$ others not in $B$. The batch $B \cup \{z\}$ is a uniform $m$-subset that avoids $i$: each such subset arises from $m$ choices of which element is $z$, equally often.
2. **Write both outputs as mixtures over $(B, z)$.** A uniform $m$-subset of $n$ contains $i$ with probability $q = m/n$. Given that it does, the rest of it is a uniform $(m-1)$-subset of the others, which is $B$; given that it does not, it is a uniform $m$-subset avoiding $i$, which is $B \cup \{z\}$. So flip an independent coin with probability $q$: on heads the batch is $B \cup \{x\}$ in $D$ and $B \cup \{x'\}$ in $D'$, on tails it is $B \cup \{z\}$ in both. Given $(B, z)$, the outputs are $(1 - q)R + qP$ and $(1 - q)R + qP'$ with $R = M(B\cup\{z\})$, $P = M(B\cup\{x\})$, $P' = M(B\cup\{x'\})$.
3. **Check the lemma's assumptions.** The three batches $B\cup\{x\}$, $B\cup\{x'\}$, $B\cup\{z\}$ have size $m$ and differ pairwise in one element, so they are pairwise replace-one neighbours. Hence $P \le e^\varepsilon P' + \delta$ and $P \le e^\varepsilon R + \delta$ on every event, and the same with $P$ and $P'$ swapped.
4. **Apply the mixture lemma and average.** Fact 2 with weight $q$ gives the bound in one direction, and again with $P$ and $P'$ swapped in the other. Both runs average over $(B, z)$ with the same weights, so Fact 1 carries the bound to the full mechanism.

**The Gaussian column.** For a sum of gradients clipped to norm $C$, adding or removing a record moves the sum by at most $C$, and replacing one moves it by up to $2C$: the old gradient leaves and a new one, possibly pointing the opposite way, comes in. A Gaussian mechanism's privacy depends only on the shift divided by the noise's standard deviation, and $2C/(\sigma C) = C/((\sigma/2)C)$, so noise $\sigma C$ with shift $2C$ is the same as shift $C$ with noise multiplier $\sigma/2$. Lebeda, Regehr, Kamath & Steinke (2024, Proposition 9) show, using Zhu, Dong & Wang (2022, Theorem 11), that for fixed-size batches this shift of $2C$ is the worst case under the add relation and under the remove relation, each taken separately, because a record entering the batch pushes another one out.

**Poisson with replace-one.** Given the rest of the sample $\tilde D$, the target is drawn with probability $q$ in both runs, so the outputs are $(1 - q)R + qP$ and $(1 - q)R + qP'$ with $R = M(\tilde D)$, $P = M(\tilde D\cup\{x\})$, $P' = M(\tilde D\cup\{x'\})$. The pair $(P, R)$ is one add/remove step apart, but $(P, P')$ is two: remove $x$, then add $x'$. The mixture lemma needs one guarantee for both pairs, so use the weaker one. Fact 3 gives the base guarantee $(2\varepsilon, (1 + e^\varepsilon)\delta)$ for both, and Fact 2 then gives $\varepsilon' = \ln(1 + q(e^{2\varepsilon} - 1))$ and $\delta' = q(1 + e^\varepsilon)\delta$. For small $q$ both logarithms are close to their arguments' excess over 1, and since $e^{2\varepsilon} - 1 = (e^\varepsilon - 1)(e^\varepsilon + 1)$, this bound is about $e^\varepsilon + 1$ times the add/remove value.

The bound is not tight: the pair $(P, R)$ is only one step apart, but the lemma charges it the two-step guarantee. For pure DP ($\delta = 0$) the exact value follows from likelihood ratios. Pure DP of $M$ says that every likelihood ratio between neighbours lies in $[e^{-\varepsilon}, e^\varepsilon]$, so both $P/R$ and $P'/R$ do. Divide the two mixtures by $R$: their ratio is $\frac{(1 - q) + qP/R}{(1 - q) + qP'/R}$. It is largest when the numerator is largest and the denominator smallest, that is with $P/R = e^\varepsilon$ and $P'/R = e^{-\varepsilon}$, which gives at most $\frac{1 - q + qe^\varepsilon}{1 - q + qe^{-\varepsilon}}$. Taking the logarithm, and writing $1 - q + qe^{-\varepsilon} = 1 - q(1 - e^{-\varepsilon})$,
$$\varepsilon' = \ln\big(1 + q(e^\varepsilon - 1)\big) - \ln\big(1 - q(1 - e^{-\varepsilon})\big).$$ {#eq-sub-replace-pure}
The Laplace mechanism on a sum attains it ([Exercise 3](../../book-exercises/part04-privacy-accounting/05-subsampling.md#sec-sol-subsampling-3)). For small $\varepsilon$ each of the two logarithms is about $q\varepsilon$, so the value is about twice the add/remove value. At $\varepsilon = 1$ and $q = 0.01$ it is 0.02338, against 0.06193 for the group-privacy bound and 0.01704 for add/remove: a ratio of 1.37, not 3.64.

### Interpretation

**Why mixing them is a bug.** Each row of @tbl-sub-matching is correct on its own. The bug is to train with one row and report the number from another. Two cases are common.

The first is fixed-size sampled batches with a Poisson accountant. The accountant assumes a shift of $C$; the training run has a shift of $2C$. You are reporting the privacy of twice as much noise as you added. Lebeda et al. (2024, Table 1) give an example with $q = 0.001$, $\sigma = 0.8$ and 10,000 steps: at $\delta = 10^{-5}$ they report $\varepsilon = 0.80$ for Poisson sampling and $\varepsilon = 12.98$ for fixed-size batches, from a numerical accountant. (The tight privacy-loss-distribution accountants of the `dp_accounting` and `prv_accountant` libraries give 0.78 for Poisson sampling; that value comes from those libraries, not from this chapter's script.) The RDP accountant of this chapter reproduces the gap: 1.38 against 15.30.

The second is shuffle and partition with a Poisson accountant, and it is worse, because the mechanism of amplification is missing altogether. In one epoch the record is in exactly one batch, never absent, so no step's output is a mixture with "not used". The only uncertainty is *which* batch held the record, and the other records can give that away.

Chua et al. (2024) build a bad case. Every other record has gradient $-1$, the target has $+1$, and the neighbouring dataset zeroes the target out. Recentre each batch sum by adding the batch size, which removes the $-1$ contributions of a full batch of other records. Batches without the target then output $\mathcal N(0, \sigma^2)$. The target's batch has one $-1$ fewer and a $+1$ instead, so it outputs $\mathcal N(2, \sigma^2)$, or $\mathcal N(1, \sigma^2)$ when the target is zeroed. You don't know which batch holds it, so with $b$ batches the epoch's output is a mixture, over the batch index $t$, of $\mathcal N(2e_t, \sigma^2I)$ against the same mixture of $\mathcal N(e_t, \sigma^2I)$, where $e_t$ is the vector with 1 in position $t$ and 0 elsewhere.

A natural test looks at the largest of the $b$ batch outputs and says "target present" when it exceeds a threshold $\tau$. The largest output stays below $\tau$ only if every batch output does, and the batches are independent. The target's batch stays below $\tau$ with probability $\Phi(\frac{\tau-2}\sigma)$, where $\Phi$ is the standard normal distribution function, and each of the $b - 1$ others with probability $\Phi(\frac\tau\sigma)$. This does not depend on which batch holds the target, so mixing over $t$ changes nothing, and the test fires with probability $1 - \Phi(\frac{\tau-2}\sigma)\Phi(\frac\tau\sigma)^{b-1}$ under the first distribution and $1 - \Phi(\frac{\tau-1}\sigma)\Phi(\frac\tau\sigma)^{b-1}$ under the second. Recall that $\delta(\varepsilon)$, the smallest $\delta$ that works with a given $\varepsilon$, is the largest value of $P(A) - e^\varepsilon Q(A)$ over events $A$ (Fact 1). Any one event, such as "the largest output exceeds $\tau$", gives a value no larger, so it gives a lower bound on $\delta(\varepsilon)$. Maximizing over $\tau$ gives the best such lower bound, and the smallest $\varepsilon$ it does not rule out at a given $\delta$ is a lower bound on the true $\varepsilon$.

This is the zero-out relation, not add/remove. Under zero-out, a Poisson step's clipped sum also moves by at most $C$, as under add/remove, so the Poisson accountant's number applies to the same relation, and the comparison below is like with like.

The script reproduces Chua et al.'s headline check: with $\sigma = 0.4$ and $b = 10{,}000$ batches, the test gives $\delta(4) \ge 0.226$ for shuffling, against $\delta(4) = 0.244$ for a fixed, non-random batch order. They report that a Poisson accountant gives $\delta(4) \le 1.18 \times 10^{-5}$ for the same parameters.

### Example: one epoch of the running example

Take $q = 0.01$, so one epoch is $b = 100$ batches, and $\delta = 10^{-5}$. @tbl-sub-shuffle compares three batch samplers for one epoch. With deterministic batches the record is used once, in a known batch, so the epoch is a plain Gaussian mechanism and its $\varepsilon$ is exact. For shuffle and partition the table gives the lower bound from the test above, and for Poisson sampling the upper bound from the RDP accountant of @sec-sub-dpsgd.

| $\sigma$ | Deterministic batches (exact) | Shuffled batches (lower bound) | Poisson sampling (upper bound) |
| --- | --- | --- | --- |
| 0.5 | 10.00 | 10.00 | 8.03 |
| 1 | 4.38 | 4.06 | 1.21 |
| 1.5 | 2.75 | 1.46 | 0.47 |
| 2 | 1.99 | 0.48 | 0.26 |
| 2.5 | 1.55 | 0.20 | 0.17 |
| 3 | 1.27 | 0.10 | 0.14 |

: The $\varepsilon$ at $\delta = 10^{-5}$ of one epoch of 100 batches under three batch samplers. Shuffling is a lower bound and Poisson an upper bound, so wherever the shuffled column exceeds the Poisson column, the Poisson number is provably wrong for shuffled training. {#tbl-sub-shuffle}

At the running example's $\sigma = 1$, shuffled training has $\varepsilon \ge 4.06$ after one epoch, while the Poisson accountant reports at most 1.21. Shuffling buys almost nothing over a fixed order (4.38). The gap closes as $\sigma$ grows (@fig-sub-schemes). At $\sigma = 3$ the shuffled lower bound falls below the Poisson upper bound, which says only that this particular test is too weak there to show a gap. For fixed-size sampled batches over many epochs, the running example gives $\varepsilon = 47.42$ after 10,000 steps against 6.71 for Poisson (both by RDP; the first uses noise multiplier $\sigma/2 = 0.5$).

![The $\varepsilon$ at $\delta = 10^{-5}$ of one epoch of 100 batches as a function of the noise multiplier: deterministic batches (exact), shuffled fixed-size batches (a lower bound from the largest-batch-output test of Chua et al. 2024), and Poisson sampling with $q = 0.01$ (an RDP upper bound). For $\sigma$ up to about 2.5, shuffled training is provably less private than the Poisson accountant claims.](figures/subsampling_sampling_schemes.svg){#fig-sub-schemes}

### Check yourself

- Your code draws a fresh random batch of exactly 256 out of 25,600 records at each step. Which noise multiplier should you give a Poisson-sampling RDP accountant with $q = 0.01$? *(Half of the one you use, $\sigma/2$: the base mechanism's shift is $2C$.)*
- Why can't the mixture lemma be applied to one step of shuffled training? *(Given the other records' positions, the record's batch is fixed, so the step's output on $D'$ is not a mixture of "used" and "not used". The randomness is only in which step uses it.)*

## RDP of the Poisson-subsampled Gaussian {#sec-sub-rdp-gaussian}

### Context

Per-step $(\varepsilon, \delta)$ amplification composes badly (@sec-sub-dpsgd). RDP composes better. A mechanism is $(\alpha, \varepsilon(\alpha))$-RDP when the Rényi divergence of order $\alpha$ between its outputs on any two neighbouring datasets is at most $\varepsilon(\alpha)$. The Rényi divergence $D_\alpha(P\|Q) = \frac{1}{\alpha - 1}\ln\mathbb E_{y\sim Q}\big[(P(y)/Q(y))^\alpha\big]$ is a log-average of a power of the likelihood ratio; a larger order $\alpha$ weights large ratios more heavily. Under composition the RDP curves of the steps simply add ([Rényi DP](01-renyi-dp.md)), so you want the RDP curve of one subsampled Gaussian step. That is a Rényi divergence between a Gaussian mixture and a Gaussian. It has no closed form, but it has an exact series for integer orders and a stable numerical integral for the rest. @fig-sub-mixture shows the pair and its privacy loss, the logarithm of the likelihood ratio at each output.

![Left: one subsampled Gaussian step with $q = 0.1$. The output with the record is the mixture (dashed), barely distinguishable from $P_0$. Right: the privacy loss $\ln\big((1 - q) + q\,P_1/P_0\big)$. Where the record is unlikely to have been used it is flat near $\ln(1 - q)$; in the far right tail it runs parallel to the unsubsampled loss, $\ln(1/q)$ lower.](figures/subsampling_mixture.svg){#fig-sub-mixture}

### Statement

The statement reduces the whole problem to one dimension and one direction. Whatever the data, the worst case is a single coordinate in which the record moves the sum by the full sensitivity, and of the two orders of the pair the one with the mixture first is always the larger. That divergence is the average under $P_0$ of the mixture's likelihood ratio raised to the power $\alpha$, and for integer $\alpha$ the binomial theorem expands it into a finite sum.

Let $P_0 = \mathcal N(0, \sigma^2)$, $P_1 = \mathcal N(1, \sigma^2)$ and $P_q = (1 - q)P_0 + qP_1$. The Poisson-subsampled Gaussian mechanism with $\ell_2$ sensitivity 1 (one record moves the sum by Euclidean length at most 1) and noise multiplier $\sigma$ is $(\alpha, \varepsilon(\alpha))$-RDP for add/remove neighbours with
$$\varepsilon(\alpha) = \max\big\{D_\alpha(P_q\|P_0),\ D_\alpha(P_0\|P_q)\big\} = D_\alpha(P_q\|P_0) = \frac{\ln A_\alpha}{\alpha - 1},$$ {#eq-sub-rdp}
$$A_\alpha = \mathbb E_{y\sim P_0}\Big[\Big((1 - q) + q\frac{P_1(y)}{P_0(y)}\Big)^\alpha\Big], \qquad \frac{P_1(y)}{P_0(y)} = e^{(2y - 1)/(2\sigma^2)}.$$ {#eq-sub-a-alpha}
For integer $\alpha$,
$$A_\alpha = \sum_{j=0}^\alpha\binom\alpha j(1 - q)^{\alpha - j}q^j\exp\Big(\frac{j^2 - j}{2\sigma^2}\Big).$$ {#eq-sub-binomial}
The first direction, $D_\alpha(P_q\|P_0)$, puts the dataset with the record first: it is the remove direction of @sec-sub-eps-delta and of [Numerical accounting](06-numerical-accounting.md). The reduction to one dimension is Mironov, Talwar & Zhang (2019, Theorem 4), the dominance of the first direction their Theorem 5 and Corollary 7, and @eq-sub-binomial their Section 3.3. @sec-apx-amp-direction in the appendices derives the dominance.

### Derivation

The derivation first removes everything except the one coordinate along which the record moves the output, then evaluates the divergence of the remaining one-dimensional pair.

1. **Condition on the rest of the sample** (Mironov, Talwar & Zhang 2019, Theorem 4). Let $f$ be the clipped gradient sum, divided by $C$ so that the sensitivity is 1. Condition on the rest of the sample $\tilde D$. The outputs are $\mathcal N(f(\tilde D), \sigma^2I)$ and the mixture $(1 - q)\mathcal N(f(\tilde D), \sigma^2I) + q\mathcal N(f(\tilde D\cup\{x\}), \sigma^2I)$, and both runs average these over $\tilde D$ with the same weights.
2. **Averaging cannot increase the divergence.** This is the RDP counterpart of Fact 1. For $\alpha > 1$, $e^{(\alpha-1)D_\alpha(P\|Q)} = \int P(y)^\alpha Q(y)^{1-\alpha}\,\mathrm d y$. Write the integrand as $Q\,(P/Q)^\alpha$. The function $t^\alpha$ is convex, and a function of the form $Q\,g(P/Q)$ with $g$ convex, called the **perspective** of $g$, is jointly convex in the pair $(P, Q)$: the value at an average of pairs is at most the average of the values. So for averages with shared weights $w_t$, $\int(\sum_tw_tP_t)^\alpha(\sum_tw_tQ_t)^{1-\alpha} \le \sum_tw_t\int P_t^\alpha Q_t^{1-\alpha} \le \max_t\int P_t^\alpha Q_t^{1-\alpha}$, in either order of the pair. The divergence of the averages over $\tilde D$ is at most the largest divergence for one $\tilde D$. This is the **quasi-convexity** of the Rényi divergence (van Erven & Harremoës 2014, Theorem 13; @sec-apx-tail-perspective in the appendices derives it).
3. **Shift, rotate and rescale.** Moving both distributions by the same amount or rotating both does not change any likelihood ratio, so it does not change the divergence. Shift by $f(\tilde D)$ and rotate so the difference $f(\tilde D\cup\{x\}) - f(\tilde D)$ lies along the first axis. The other coordinates have the same distribution in both runs and are independent of the first, and the Rényi divergence of independent coordinates is the sum of theirs (additivity, [Divergences](../part01-prerequisites/03-divergences.md)), so the identical ones contribute 0 and drop out. A difference of length $c < 1$ is, after rescaling by $1/c$, the same as length 1 with noise $\sigma/c > \sigma$. That pair is the length-1 pair with extra independent noise added, and no processing of the outputs can increase a divergence (data processing), so it cannot be worse. What is left is the pair $(P_q, P_0)$ in both orders.
4. **Write the first direction as an expectation under $P_0$.** By the definition of $D_\alpha$ given in the context, $D_\alpha(P_q\|P_0) = \frac{1}{\alpha - 1}\ln\mathbb E_{P_0}[(P_q/P_0)^\alpha]$, and dividing the mixture by $P_0$ gives $P_q/P_0 = (1 - q) + q\,P_1/P_0$. This is @eq-sub-a-alpha. The ratio $P_1/P_0$ is the Gaussian likelihood ratio: the normalizing constants cancel and $\exp\big(\frac{-(y-1)^2 + y^2}{2\sigma^2}\big) = e^{(2y-1)/(2\sigma^2)}$.
5. **Expand for integer $\alpha$.** Write $r = P_1/P_0$. The binomial theorem gives $(1 - q + qr)^\alpha = \sum_j\binom\alpha j(1-q)^{\alpha-j}q^jr^j$, and the expectation of a sum is the sum of the expectations, so $A_\alpha$ is this sum with each $r^j$ replaced by $\mathbb E_{P_0}[r^j]$.
6. **Compute each moment.** $\mathbb E_{P_0}[r^j] = \mathbb E[e^{j(2y-1)/(2\sigma^2)}]$ for $y \sim \mathcal N(0, \sigma^2)$. The moment generating function of this normal is $\mathbb E[e^{\lambda y}] = e^{\lambda^2\sigma^2/2}$, and with $\lambda = j/\sigma^2$ it gives $e^{j^2/(2\sigma^2)}$. Multiplying by the constant factor $e^{-j/(2\sigma^2)}$, the $j$-th moment is $e^{(j^2 - j)/(2\sigma^2)}$, which gives @eq-sub-binomial. Equivalently, $\mathbb E_{P_0}[(P_1/P_0)^j] = e^{(j-1)D_j(P_1\|P_0)}$ by the definition of $D_j$, and $D_j(P_1\|P_0) = j/(2\sigma^2)$ for two Gaussians with unit shift ([Divergences](../part01-prerequisites/03-divergences.md), @eq-gauss-renyi).
7. **Fractional $\alpha$: integrate numerically.** For non-integer $\alpha$ the binomial sum does not terminate, so compute @eq-sub-a-alpha as an integral. Its integrand is $P_0(y)$ times a power of the mixture ratio. Huge and tiny factors would overflow, so work with its logarithm, $\ln P_0(y) + \alpha\,\mathrm{logaddexp}\big(\ln(1-q), \ln q + \frac{2y-1}{2\sigma^2}\big)$, where $\mathrm{logaddexp}(u, v) = \ln(e^u + e^v)$ is computed without forming $e^u$ or $e^v$. Sum on a grid of step $\sigma/100$ with log-sum-exp, the same trick for a sum of many exponentials. The integrand has a bump near $y = 0$ and, for large $\alpha$, a second one near $y = \alpha$. The second comes from the part $q^\alpha e^{\alpha(2y-1)/(2\sigma^2)}$: multiplied by $P_0(y) \propto e^{-y^2/(2\sigma^2)}$, it gives an exponent $(-y^2 + 2\alpha y)/(2\sigma^2)$, a Gaussian centred at $y = \alpha$ (the tilted distributions of [Divergences](../part01-prerequisites/03-divergences.md)). So the grid runs from $-12\sigma$ to $\alpha + 12\sigma$.
8. **The other direction.** By the definition, $D_\alpha(P_0\|P_q) = \frac{1}{\alpha - 1}\ln\mathbb E_{P_q}[(P_0/P_q)^\alpha]$. Changing the expectation from $P_q$ to $P_0$ multiplies the integrand by $P_q/P_0$, which turns $(P_0/P_q)^\alpha$ into $(P_0/P_q)^{\alpha - 1}$, so $D_\alpha(P_0\|P_q) = \frac{1}{\alpha - 1}\ln\mathbb E_{P_0}[(P_q/P_0)^{-(\alpha-1)}]$. The power is negative, so there is no finite binomial sum; integrate it numerically. Mironov, Talwar & Zhang (2019, Theorem 5) prove $D_\alpha(P_q\|P_0) \ge D_\alpha(P_0\|P_q)$ for every $\alpha \ge 1$. Their argument uses the reflection $y \mapsto 1 - y$, which swaps $P_0$ and $P_1$. The reflection cuts the pair into randomized-response pairs, and @sec-apx-amp-direction in the appendices derives the inequality that way, step by step. The script computes both and takes the larger, which is always the first. This dominance is for Rényi divergences. For $\delta(\varepsilon)$, @sec-num-subsampled computes both directions, because neither dominates in general.

**Walk-through at $\alpha = 2$.** With $q = 0.01$ and $\sigma = 1$, @eq-sub-binomial has three terms, one for each possible number $j$ of factors $r$ (@tbl-sub-alpha2).

| $j$ | $\binom 2j(1-q)^{2-j}q^j$ | $e^{(j^2 - j)/2}$ | Term |
| --- | --- | --- | --- |
| 0 | $(0.99)^2 = 0.9801$ | 1 | 0.980100 |
| 1 | $2 \times 0.99 \times 0.01 = 0.0198$ | 1 | 0.019800 |
| 2 | $(0.01)^2 = 0.0001$ | $e = 2.718$ | 0.000272 |

: The binomial sum for $A_2$ with $q = 0.01$, $\sigma = 1$. Without the exponential factors the terms would sum to exactly 1. {#tbl-sub-alpha2}

Without the exponential factors the three coefficients add up to $(1 - q + q)^2 = 1$. The exponential factor only changes the last term, from $q^2$ to $q^2e^{1/\sigma^2}$, so the terms sum to $A_2 = 1 + q^2(e^{1/\sigma^2} - 1) = 1 + 0.0001 \times 1.71828 = 1.00017183$, and $\varepsilon(2) = \ln A_2/(2 - 1) = 1.7181 \times 10^{-4}$. Without sampling, $\varepsilon(2) = 2/(2\sigma^2) = 1$. Only the $j = 2$ term carries any privacy cost, because the terms $j = 0$ and $1$ have the same factor 1 that a mixture of identical distributions would have.

### Interpretation

**Two regimes.** @fig-sub-rdp shows the curve for $q = 0.01$, $\sigma = 1$. Up to about $\alpha = 8$ the curve stays within 30% of the small-$q$ line, the $j = 2$ term alone, worked out in @sec-sub-small-q (1.30 times it at $\alpha = 8$, @tbl-sub-small-q): $\varepsilon(8) = 8.9364 \times 10^{-4}$, against 4 without sampling. Between $\alpha = 8$ and 16 it climbs steeply: $\varepsilon(16) = 3.0879$ and $\varepsilon(32) = 11.246$, against 8 and 16 without sampling. The reason is in the right panel of @fig-sub-mixture. Low orders average the privacy loss over the bulk, where sampling flattens it. High orders look at the far tail, where the loss is the unsubsampled one minus $\ln(1/q)$. In the tail, sampling saves only $\ln(1/q)$, as for $(\varepsilon, \delta)$ at large $\varepsilon$. The one-term lower bound @eq-cdp-sg-lower of [Concentrated DP](03-concentrated-dp.md) makes this precise. Every term of @eq-sub-binomial is positive, so $A_\alpha$ is at least its last term, $q^\alpha e^{\alpha(\alpha - 1)/(2\sigma^2)}$, and dividing its logarithm by $\alpha - 1$ gives $\varepsilon(\alpha) \ge \frac{\alpha\ln q}{\alpha - 1} + \frac{\alpha}{2\sigma^2}$: the unsubsampled curve minus about $\ln(1/q)$.

**Which orders matter.** An accountant turns the RDP curve into an $(\varepsilon, \delta)$ guarantee by a conversion that holds at every order, such as $\varepsilon = T\,\varepsilon(\alpha) + \ln(1/\delta)/(\alpha - 1)$ for $T$ steps, and it only uses the order that makes this smallest. For long DP-SGD runs that order sits in the quadratic regime. For a few steps it sits at the foot of the steep rise, where $T\,\varepsilon(\alpha)$ is still small next to the conversion term (@sec-sub-dpsgd).

![The RDP curve of the Poisson-subsampled Gaussian with $q = 0.01$, $\sigma = 1$ (black), against the unsubsampled Gaussian $\alpha/(2\sigma^2)$, the small-$q$ term $\alpha q^2(e^{1/\sigma^2} - 1)/2$ and the leading term of Abadi et al.'s bound. The vertical line marks where Abadi et al.'s bound stops applying. The dashed vertical line marks $\alpha = 2\sigma^2\ln(1/q)$, where the quadratic expansion breaks. The exact curve stays near the quadratic term up to about $\alpha = 8$, then rises toward the unsubsampled line.](figures/subsampling_rdp_curve.svg){#fig-sub-rdp}

### Example: checking the computation

The binomial sum is exact but only exists at integer orders; the numerical integral covers every order but could be wrong if the grid were too coarse or too short. Comparing them where both exist tests the integral. @tbl-sub-check does this at integer orders, and also compares the two directions at several orders, for $q = 0.01$ and $\sigma = 1$.

| $\alpha$ | Binomial sum | Integral | $D_\alpha(P_q\Vert P_0)$ | $D_\alpha(P_0\Vert P_q)$ |
| --- | --- | --- | --- | --- |
| 1.5 | — | — | $1.2725 \times 10^{-4}$ | $1.2148 \times 10^{-4}$ |
| 2 | $1.718134 \times 10^{-4}$ | $1.718134 \times 10^{-4}$ | $1.7181 \times 10^{-4}$ | $1.6022 \times 10^{-4}$ |
| 3 | $2.646376 \times 10^{-4}$ | $2.646376 \times 10^{-4}$ | — | — |
| 8 | — | — | $8.9364 \times 10^{-4}$ | $5.7239 \times 10^{-4}$ |
| 10 | $3.827042 \times 10^{-2}$ | $3.827042 \times 10^{-2}$ | — | — |
| 32 | $1.124628 \times 10^{1}$ | $1.124628 \times 10^{1}$ | $1.1246 \times 10^{1}$ | $1.6923 \times 10^{-3}$ |

: The two ways of computing $D_\alpha(P_q\|P_0)$ agree to seven digits, and the reverse direction is always smaller. $q = 0.01$, $\sigma = 1$. {#tbl-sub-check}

The integral agrees with the exact sum to all printed digits, from $10^{-4}$ to $10^1$, so the grid is fine enough. The reverse direction is smaller at every order. At $\alpha = 32$ it is nearly four orders of magnitude smaller ($1.6923 \times 10^{-3}$ against 11.246), because it raises the mixture ratio to a negative power and so looks at the left tail, where the ratio $P_0/P_q$ is bounded by $1/(1-q)$: the mixture always keeps at least $1 - q$ of $P_0$. The script also checks every order of the grid for $\sigma = 0.5$ and $\sigma = 2$: the largest excess of the reverse direction is negative in both cases ($-5.09 \times 10^{-4}$ and $-5.67 \times 10^{-8}$).

### Check yourself

- In @eq-sub-binomial, why do the terms $j = 0$ and $j = 1$ never cost privacy? *(Their exponential factor is $e^0 = 1$. They are the terms in which the record's influence enters at most linearly, and a linear term averages to the same value under both distributions, since $\mathbb E_{P_0}[P_1/P_0] = 1$.)*
- Why does the numerical grid for $D_\alpha(P_q\|P_0)$ extend to $y = \alpha + 12\sigma$, far beyond where $P_0$ has any mass? *(For large $\alpha$ the integrand is dominated by $q^\alpha P_0(y)e^{\alpha(2y-1)/(2\sigma^2)}$, which is a Gaussian bump centred at $y = \alpha$. Cutting the grid at $12\sigma$ would miss it.)*

## Small q and the moments accountant {#sec-sub-small-q}

### Context

Abadi et al. (2016) introduced DP-SGD together with the **moments accountant**. Its central quantity is built from the **privacy loss**, $L = \ln\big(P(y)/Q(y)\big)$ for an output $y$ drawn from $P$: the log-likelihood ratio of the output, which is large when the output gives the record away. The accountant tracks the logarithm of the moment generating function of $L$, which Abadi et al. call $\alpha_M(\lambda)$ and [Rényi DP](01-renyi-dp.md) renames $K_M(\lambda) = \ln\mathbb E[e^{\lambda L}]$, because $\alpha$ is our order. It adds $K_M$ over steps and converts the total to $(\varepsilon, \delta)$ with a tail bound.

This is RDP in disguise, as [Rényi DP](01-renyi-dp.md) shows. Since $e^{\lambda L} = (P/Q)^\lambda$, the expectation under $P$ is $\mathbb E_P[(P/Q)^\lambda] = \mathbb E_Q[(P/Q)^{\lambda + 1}] = e^{\lambda D_\alpha(P\|Q)}$ with $\alpha = \lambda + 1$, so $K_M(\lambda) = \lambda\,\varepsilon(\lambda + 1)$. The tail bound is Markov's inequality applied to $e^{\lambda L}$: $\Pr[L > \varepsilon] \le \mathbb E[e^{\lambda L}]e^{-\lambda\varepsilon}$, which gives $\delta = \min_\lambda e^{K_M(\lambda) - \lambda\varepsilon}$, Mironov's conversion. Abadi et al. did not have @eq-sub-binomial in general. They proved an asymptotic bound for small $q$. How good is it, and where does it stop holding?

### Statement

When $q$ is small, the walk-through at $\alpha = 2$ generalizes: the terms $j = 0$ and $1$ of @eq-sub-binomial cost nothing, every term from $j = 3$ on carries $q^3$ or a higher power, and the $j = 2$ term dominates. For $q \to 0$ at fixed $\alpha$ and $\sigma$, the leading term of the subsampled Gaussian's RDP curve is
$$\varepsilon(\alpha) \approx \frac{\alpha q^2\big(e^{1/\sigma^2} - 1\big)}{2} \approx \frac{\alpha q^2}{2\sigma^2}\quad\text{for large }\sigma.$$ {#eq-sub-small-q}
Abadi et al. (2016, Lemma 3) prove a bound of the same shape, under conditions that limit the order. For $\sigma \ge 1$, $q < 1/(16\sigma)$ and integer $\lambda \le \sigma^2\ln\frac{1}{q\sigma}$,
$$K_M(\lambda) \le \frac{q^2\lambda(\lambda + 1)}{(1 - q)\sigma^2} + O\Big(\frac{q^3\lambda^3}{\sigma^3}\Big), \quad\text{that is}\quad \varepsilon(\alpha) \lesssim \frac{q^2\alpha}{(1 - q)\sigma^2}\ \text{ for }\ \alpha \le 1 + \sigma^2\ln\frac{1}{q\sigma}.$$ {#eq-sub-abadi}
The second form divides by $\lambda$ and substitutes $\alpha = \lambda + 1$.

### Derivation

1. **Subtract what costs nothing.** Without the exponential factors, @eq-sub-binomial is $\sum_j\binom\alpha j(1-q)^{\alpha-j}q^j = (1 - q + q)^\alpha = 1$ by the binomial theorem. Subtracting this 1 from $A_\alpha$ leaves each term multiplied by its exponential factor minus 1:
$$A_\alpha = 1 + \sum_{j=2}^\alpha\binom\alpha j(1 - q)^{\alpha - j}q^j\big(e^{(j^2 - j)/(2\sigma^2)} - 1\big).$$
The $j = 0$ and $j = 1$ terms are gone, because their factor is $e^0 - 1 = 0$.
2. **Keep the lowest power of $q$.** The $j = 2$ term is $\binom\alpha2q^2(1-q)^{\alpha-2}(e^{1/\sigma^2} - 1)$. Every other term carries $q^3$ or more, so for small $q$ it is negligible next to this one.
3. **Take the logarithm.** For small $y$, $\ln(1 + y) = y + O(y^2)$, and $(1 - q)^{\alpha - 2} = 1 + O(q)$. So $(\alpha - 1)\varepsilon(\alpha) = \ln A_\alpha = \frac{\alpha(\alpha-1)}{2}q^2(e^{1/\sigma^2} - 1) + O(q^3)$.
4. **Divide by $\alpha - 1$** to get @eq-sub-small-q. For large $\sigma$, $e^{1/\sigma^2} - 1 \approx 1/\sigma^2$, since $e^u \approx 1 + u$ for small $u$.
5. **Where does the expansion break?** The expansion assumes the $j = 2$ term dominates, so look at how the terms change with $j$. Going from $j$ to $j + 1$, the binomial coefficient is multiplied by $\frac{\alpha - j}{j + 1}$, the power of $q$ by $q$, and the exponential by $e^{((j+1)^2 - (j+1) - j^2 + j)/(2\sigma^2)} = e^{j/\sigma^2}$. So the terms first fall, and grow once $q\,e^{j/\sigma^2}$ passes 1, near $j \approx \sigma^2\ln(1/q)$. The expansion fails only later, when the last term, $q^\alpha e^{\alpha(\alpha-1)/(2\sigma^2)}$, overtakes the $j = 2$ term, about $q^2e^{1/\sigma^2}$. Ignoring the binomial coefficient and comparing logarithms, that happens when $\alpha\ln q + \frac{\alpha(\alpha - 1)}{2\sigma^2} > 2\ln q + \frac{1}{\sigma^2}$. Move everything to one side: the terms in $\ln q$ give $-(\alpha - 2)\ln\frac1q$, and the others give $\frac{\alpha^2 - \alpha - 2}{2\sigma^2}$. Since $\alpha^2 - \alpha - 2 = (\alpha - 2)(\alpha + 1)$, the condition is $(\alpha - 2)\big(\frac{\alpha + 1}{2\sigma^2} - \ln\frac1q\big) > 0$, that is $\alpha > 2\sigma^2\ln(1/q) - 1$. So the expansion holds up to about
$$\alpha \approx 2\sigma^2\ln(1/q),$$ {#eq-sub-breakdown}
and the curve leaves the quadratic regime abruptly there. At $q = 0.01$, $\ln(1/q) = 4.605$, so this is 9.2 for $\sigma = 1$ and 147.4 for $\sigma = 4$. The exact curve first exceeds twice the leading term at 8.9 and 147.4 (@tbl-sub-small-q). Abadi et al.'s condition $\lambda \le \sigma^2\ln\frac{1}{q\sigma}$ has the same form with about half the constant, so it is conservative.

### Interpretation

The first thing to notice is that the cost is quadratic, not linear, in $q$. The $(\varepsilon, \delta)$ amplification of @sec-sub-eps-delta gives a per-step $\varepsilon'$ linear in $q$. In RDP the cost per step is quadratic in $q$, about $\alpha q^2/(2\sigma^2)$. Since $T = E/q$ steps make $E$ epochs, the total RDP after $E$ epochs is about $(E/q) \times \alpha q^2/(2\sigma^2) = E\,q\,\alpha/(2\sigma^2)$. It shrinks as batches get smaller at a fixed number of epochs.

Abadi et al.'s bound is about twice the leading term for large $\sigma$: $q^2\alpha/\sigma^2$ against $q^2\alpha/(2\sigma^2)$. It is a valid but loose upper bound, and it is the source of the "$\varepsilon(\alpha) \approx q^2\alpha/\sigma^2$" you will see quoted.

The real limit of the expansion is not its constant but its order range. Up to $\alpha = \sigma^2\ln(1/q)$ the leading term is within 8% (1.08 at $\alpha = 4.6$ for $\sigma = 1$, 1.05 at $\alpha = 73.7$ for $\sigma = 4$). Beyond $\alpha \approx 2\sigma^2\ln(1/q)$ (@eq-sub-breakdown) it is wrong by orders of magnitude, 2246 times too small at $\alpha = 16$ for $\sigma = 1$, and an accountant that uses it there under-reports $\varepsilon$.

### Example: where the expansion breaks

@tbl-sub-small-q compares the exact curve with @eq-sub-small-q and @eq-sub-abadi at $q = 0.01$. For instance, at $\alpha = 2$ and $\sigma = 1$ the Abadi column is $q^2\alpha/((1 - q)\sigma^2) = 0.0002/0.99 = 2.0202 \times 10^{-4}$.

| $\alpha$ | Exact, $\sigma = 1$ | Leading term, $\sigma = 1$ | Abadi, $\sigma = 1$ | Exact / leading, $\sigma = 1$ | Exact / leading, $\sigma = 4$ |
| --- | --- | --- | --- | --- | --- |
| 2 | $1.7181 \times 10^{-4}$ | $1.7183 \times 10^{-4}$ | $2.0202 \times 10^{-4}$ | 1.00 | 1.00 |
| 4 | $3.6315 \times 10^{-4}$ | $3.4366 \times 10^{-4}$ | $4.0404 \times 10^{-4}$ | 1.06 | 1.00 |
| 6 | $5.8350 \times 10^{-4}$ | $5.1548 \times 10^{-4}$ | $6.0606 \times 10^{-4}$ | 1.13 | 1.00 |
| 8 | $8.9364 \times 10^{-4}$ | $6.8731 \times 10^{-4}$ | $8.0808 \times 10^{-4}$ | 1.30 | 1.00 |
| 10 | $3.8270 \times 10^{-2}$ | $8.5914 \times 10^{-4}$ | $1.0101 \times 10^{-3}$ | 44.54 | 1.01 |
| 32 | $1.1246 \times 10^{1}$ | $2.7493 \times 10^{-3}$ | $3.2323 \times 10^{-3}$ | 4090.67 | 1.02 |

: The exact RDP of the subsampled Gaussian against its small-$q$ expansion and Abadi et al.'s leading term, $q = 0.01$. {#tbl-sub-small-q}

At $\sigma = 1$, Abadi et al.'s range ends at $\alpha = 1 + \sigma^2\ln\frac{1}{q\sigma} = 1 + \ln 100 = 5.6$, and the exact curve first exceeds twice the leading term at the grid order $\alpha = 8.9$ (8.9 also by bisection over fractional orders), just below $2\sigma^2\ln(1/q) = 9.2$. Inside its range Abadi et al.'s value is above the exact one, as an upper bound should be. At $\alpha = 8$, outside it, the leading term of their bound ($8.0808 \times 10^{-4}$) is already below the exact $8.9364 \times 10^{-4}$.

Step 5 of the derivation predicted terms that first fall and then grow; at $\alpha = 10$ you can see it. The binomial terms for $j = 2, \dots, 10$ are $1.1 \times 10^{-2}$, $2.2 \times 10^{-3}$, $8.0 \times 10^{-4}$, $5.3 \times 10^{-4}$, $6.6 \times 10^{-4}$, $1.5 \times 10^{-3}$, $6.4 \times 10^{-3}$, $4.3 \times 10^{-2}$ and $3.5 \times 10^{-1}$: they first fall and then grow, and the last one dominates. At $\sigma = 4$ the range ends at $\alpha = 52.5$ and the leading term is within 2% of the exact value up to $\alpha = 32$; the exact curve first exceeds twice the leading term between the grid orders 128 and 256, at $\alpha = 147.4$ by bisection, where @eq-sub-breakdown puts it.

**Abadi et al.'s headline number.** With $q = 0.01$, $\sigma = 4$, $\delta = 10^{-5}$ and 10,000 steps they report $\varepsilon \approx 1.26$ from the moments accountant. The exact RDP curve, composed over the 10,000 steps and converted with Mironov's conversion, gives 1.2586, at $\alpha = 20$, which matches their number to the digits they report. The improved conversion of [From RDP to (ε, δ)](02-rdp-conversions.md) gives 1.0355, at $\alpha = 17$.

### Check yourself

- At a fixed number of epochs, does halving the batch size (and so $q$) raise or lower the RDP accountant's total, to first order? *(Lower it. There are twice as many steps, each costing a quarter as much, so the total halves, as long as the orders used stay in the quadratic regime.)*
- Why does a small $\sigma$ shrink the range where @eq-sub-small-q holds? *(The higher terms grow like $q^je^{j^2/(2\sigma^2)}$. A small $\sigma$ makes the exponential overtake the power of $q$ at a smaller $j$, and so at a smaller $\alpha$.)*

## The running example: DP-SGD {#sec-sub-dpsgd}

### Context

Everything is now in place to account for the running example: DP-SGD with Poisson rate $q = 0.01$, noise multiplier $\sigma = 1$, $\delta = 10^{-5}$, and $T$ up to 10,000 steps. How large is $\varepsilon$, and how much does the accounting route matter?

### Statement

The RDP accountant adds the per-step curve of @eq-sub-rdp over the $T$ steps and converts the total to $(\varepsilon, \delta)$ once. A conversion takes an RDP guarantee at one order $\alpha$ and returns an $\varepsilon$ for the target $\delta$. It holds at every order, so the accountant takes the order that gives the smallest $\varepsilon$. Mironov's conversion adds $\ln(1/\delta)/(\alpha - 1)$ to the RDP value; the improved conversion subtracts two small corrections from that amount. DP-SGD is $(\varepsilon, \delta)$-DP for add/remove neighbours with
$$\varepsilon = \min_{\alpha}\Big[T\,\varepsilon(\alpha) + \frac{\ln(1/\delta)}{\alpha - 1}\Big] \quad\text{(Mironov)}, \qquad \varepsilon = \min_\alpha\Big[T\,\varepsilon(\alpha) + \ln\frac{\alpha - 1}{\alpha} - \frac{\ln\delta + \ln\alpha}{\alpha - 1}\Big] \quad\text{(improved)},$$ {#eq-sub-dpsgd}
with $\varepsilon(\alpha)$ from @eq-sub-rdp and the minimum over the grid of orders shared by this part (@sec-apx-sgd-rdp and @sec-apx-sgd-accounting in the appendices derive both from scratch, with worked numbers).

### Derivation

1. **One step is a subsampled Gaussian.** Clipping each per-example gradient to norm $C$ and adding $\mathcal N(0, \sigma^2C^2I)$ to the sum makes one step the mechanism of @sec-sub-rdp-gaussian, after dividing by $C$. Its RDP curve is $\varepsilon(\alpha)$.
2. **Compose.** Each step may depend on the previous outputs, through the model it starts from. RDP curves add under such adaptive composition ([Rényi DP](01-renyi-dp.md)), so $T$ steps have curve $T\,\varepsilon(\alpha)$.
3. **Convert.** Mironov's conversion is from [Rényi DP](01-renyi-dp.md): it is the Markov bound on the privacy loss recalled in @sec-sub-small-q, solved for $\varepsilon$. The improved one (Balle et al. 2020, Theorem 21) is from [From RDP to (ε, δ)](02-rdp-conversions.md). Each holds at every order, so take the smallest.

**The naive routes**, for comparison, amplify each step in $(\varepsilon, \delta)$ form with @eq-sub-amplify and then compose the per-step pairs.

1. **Per step.** Give each step a share $\delta_0$ of $\delta$ before amplification. The plain Gaussian mechanism with sensitivity 1 and noise $\sigma$ is $(\varepsilon_0, \delta_0)$-DP exactly when $\delta_0 = \Phi(\frac1{2\sigma} - \varepsilon_0\sigma) - e^{\varepsilon_0}\Phi(-\frac1{2\sigma} - \varepsilon_0\sigma)$, the hockey-stick divergence of the two Gaussians, with $\Phi$ the standard normal distribution function ([The Laplace and Gaussian mechanisms](../part03-dp-foundations/03-laplace-gaussian.md)). Solve this for $\varepsilon_0$, then amplify with @eq-sub-amplify to $(\varepsilon_1, q\delta_0)$.
2. **Basic composition.** The $\varepsilon$s and $\delta$s of the steps add, so $T$ steps are $(T\varepsilon_1, Tq\delta_0)$-DP ([Composition](../part03-dp-foundations/04-composition.md)). Setting $Tq\delta_0 = \delta$ fixes $\delta_0 = \delta/(Tq)$.
3. **Advanced composition.** Advanced composition trades a little extra $\delta'$ for an $\varepsilon$ that grows like $\sqrt T$ instead of $T$: $T$ steps are $(\varepsilon_1\sqrt{2T\ln(1/\delta')} + T\varepsilon_1(e^{\varepsilon_1} - 1),\ Tq\delta_0 + \delta')$-DP ([Composition](../part03-dp-foundations/04-composition.md)). The script splits $\delta$ between $Tq\delta_0$ and $\delta'$, trying shares of $\delta'$ from 0.01 to 0.99, and keeps the best.

### Interpretation

RDP is the only usable route for long runs. After 10,000 steps it gives $\varepsilon = 6.71$; the per-step routes give numbers in the thousands (@tbl-sub-dpsgd). The example below shows where the per-step routes lose.

For one step, though, RDP loses. At $T = 1$ the per-step route gives 0.20 and RDP 0.96. The reason is the conversion: it costs $\ln(1/\delta)/(\alpha - 1)$ no matter how small the curve is. RDP pays this once, so it wins once there are a handful of steps to compose: from $T = 4$ here (@fig-sub-dpsgd).

The improved conversion saves from 27.5% at $T = 1$ to 9.6% at $T = 10{,}000$: 0.96 against 1.32, and 6.71 against 7.43.

The best order falls as $T$ grows, from 9.9 to 4.1, because the curve term $T\,\varepsilon(\alpha)$ grows with $T$ and pulls the minimum toward smaller orders, where the curve is lower. For a few steps the best order sits at the foot of the steep rise in @fig-sub-rdp: at the best orders for $T = 1$, 10 and 100, the exact curve is 30.61, 5.44 and 1.82 times the quadratic term. It is chosen there because $T\,\varepsilon(\alpha)$ is still small next to the conversion term (0.026, 0.044 and 0.138), so a larger order, which shrinks the conversion term, is worth it. Only for long runs does the best order move into the quadratic regime: the ratio is 1.26 at $T = 1000$ and 1.06 at $T = 10{,}000$.

How tight is 6.71? It is an upper bound with two sources of slack: the conversion from RDP, and the RDP curve itself, which keeps only one summary of the privacy loss distribution. [Numerical accounting](06-numerical-accounting.md) computes the tight value for the same run, and [Choosing an accountant](09-choosing-an-accountant.md) compares all the accountants of this part on it.

### Example: the running example at T = 10,000

@tbl-sub-dpsgd and @fig-sub-dpsgd give $\varepsilon$ for the running example under the four routes.

| Steps $T$ | RDP, Mironov | RDP, improved | Best order (improved) | Per step, basic | Per step, advanced |
| --- | --- | --- | --- | --- | --- |
| 1 | 1.32 | 0.96 | 9.9 | 0.20 | 1.13 |
| 10 | 1.41 | 1.04 | 9.4 | 3.64 | 7.88 |
| 100 | 1.61 | 1.21 | 8.8 | 58.01 | 78.10 |
| 1,000 | 2.54 | 2.10 | 7.8 | 839.39 | 1258.64 |
| 10,000 | 7.43 | 6.71 | 4.1 | 11296.93 | 24379.55 |

: The $\varepsilon$ at $\delta = 10^{-5}$ of DP-SGD with $q = 0.01$, $\sigma = 1$ under four accounting routes. {#tbl-sub-dpsgd}

![The $\varepsilon$ at $\delta = 10^{-5}$ of DP-SGD with $q = 0.01$, $\sigma = 1$ against the number of steps, under four accounting routes. The RDP accountant grows slowly; per-step amplification followed by basic or advanced composition grows almost linearly and is useless beyond a few steps.](figures/subsampling_dpsgd.svg){#fig-sub-dpsgd}

Follow the per-step route at $T = 10{,}000$ to see where it loses. The first loss is weak amplification. With basic composition the route must spend $\delta_0 = \delta/(Tq) = 10^{-5}/100 = 10^{-7}$ per step. That forces $\varepsilon_0 = 5.35$, in the regime where amplification only subtracts (@fig-sub-amp): $\ln\big(1 + 0.01(e^{5.35} - 1)\big)$ gives the amplified $\varepsilon_1 = 1.130$, only about five times smaller, and basic composition then multiplies it by 10,000.

Advanced composition does not rescue it; it is worse than basic at every $T$ here. Its first term, $\varepsilon_1\sqrt{2T\ln(1/\delta')}$, beats $T\varepsilon_1$ only when $T \gg 2\ln(1/\delta')$. Its second term, $T\varepsilon_1(e^{\varepsilon_1} - 1)$, alone exceeds $T\varepsilon_1$ once $e^{\varepsilon_1} - 1 > 1$, that is $\varepsilon_1 > \ln 2$, as here. The sharper form of [Composition](../part03-dp-foundations/04-composition.md), with $\tanh(\varepsilon_1/2)$ in place of $e^{\varepsilon_1} - 1$, gives 6433: better than basic composition, and still useless.

The RDP accountant, by contrast, composes the whole curve, where each step costs almost nothing at moderate orders, and converts once, at the order 4.1.

### Check yourself

- At $T = 10{,}000$ the best order is about 4. If you only had orders $\alpha \ge 8$ available, would the reported $\varepsilon$ go up or down? *(Up. The conversion is a minimum over orders, and removing orders can only raise it.)*
- Why does the RDP column barely move between $T = 1$ and $T = 100$? *(For small $T$ the conversion term $\ln(1/\delta)/(\alpha - 1)$ dominates and $T\varepsilon(\alpha)$ is negligible at the chosen order.)*

## No tight amplification theorem for every mechanism {#sec-sub-generic}

### Context

For $(\varepsilon, \delta)$, @eq-sub-amplify is one formula, and it is tight over the class of $(\varepsilon, \delta)$-DP mechanisms: for every $\varepsilon$ and $q$, randomized response attains it. Is there an RDP analogue: a function that takes any mechanism's RDP curve and $q$ and returns the subsampled curve, tightly? No. The best general results are tight only up to small constant factors, and accountants special-case the Gaussian because it is the case where the exact answer is known.

### Statement

The general bound starts from the same binomial expansion as the Gaussian case, with each moment of the likelihood ratio bounded by the mechanism's own RDP curve at that order, and pays a factor 3 on the higher terms to cover the direction the expansion does not reach. Let $M$ be any mechanism with RDP curve $\varepsilon(\alpha)$ for add/remove neighbours, and let $q$ be the Poisson rate. For integer $\alpha \ge 2$, Zhu & Wang (2019, Theorem 5) show that the subsampled mechanism has RDP at most
$$\frac{1}{\alpha - 1}\ln\Big[(1 - q)^{\alpha-1}(\alpha q - q + 1) + \binom\alpha2q^2(1 - q)^{\alpha-2}e^{\varepsilon(2)} + 3\sum_{j=3}^\alpha\binom\alpha j(1 - q)^{\alpha-j}q^je^{(j-1)\varepsilon(j)}\Big].$$ {#eq-sub-zhu-wang}
The first term collects $j = 0$ and $j = 1$: $(1 - q)^\alpha + \alpha q(1 - q)^{\alpha - 1} = (1 - q)^{\alpha - 1}(\alpha q - q + 1)$. Without the factor 3 the same expression is a lower bound for some mechanisms (their Theorem 6). It is exact under a condition on the privacy loss that the Gaussian and Laplace mechanisms satisfy (their Theorem 8 and Proposition 10). For sampling a fixed batch without replacement under replace-one neighbours, Wang, Balle & Kasiviswanathan (2019, Theorem 9) give a bound of the form $\frac{1}{\alpha-1}\ln\big(1 + \sum_{j\ge2}\binom\alpha jq^jc_j\big)$ with $q = m/n$, where $c_2 \le 4(e^{\varepsilon(2)} - 1)$ and $c_j \le 2e^{(j-1)\varepsilon(j)}$ for $j \ge 3$, and a lower bound (Proposition 11) that matches it up to constant factors.

### Derivation (why the Gaussian is special)

1. **The first direction expands for any mechanism.** The binomial expansion (step 5 of the derivation in @sec-sub-rdp-gaussian) never used the Gaussian: for integer $\alpha$, $D_\alpha((1 - q)P_0 + qP_1\|P_0)$ expands into moments $\mathbb E_{P_0}[(P_1/P_0)^j]$. By the definition of the Rényi divergence each moment equals $e^{(j-1)D_j(P_1\|P_0)}$, and RDP of $M$ bounds it by $e^{(j-1)\varepsilon(j)}$. Bounding each term this way gives @eq-sub-zhu-wang without the factor 3.
2. **But each term may come from a different pair.** $\varepsilon(j)$ is a maximum over neighbouring pairs, taken separately for each $j$. The bound in step 1 is attained only if one pair attains $\varepsilon(j)$ for every $j$ at once. For the Gaussian it does: the pair whose sums differ by the full sensitivity is the worst for every order.
3. **The second direction does not expand.** $D_\alpha(P_0\|(1 - q)P_0 + qP_1)$ has the mixture in the denominator, raised to a negative power, as in step 8 of the derivation in @sec-sub-rdp-gaussian, so there is no finite binomial sum. For a general mechanism this direction can exceed the first. A general bound must cover it too, and Zhu & Wang's bound carries a factor 3 that the first-direction expansion of step 1 does not need. For the Gaussian, the reflection argument of Mironov, Talwar & Zhang (2019, Theorem 5) shows that the first direction always dominates in Rényi divergence.

### Interpretation

For the Gaussian, the general bound @eq-sub-zhu-wang is close but not exact, and the factor 3 costs most at the orders in between (see the example). That is why accountants use @eq-sub-binomial for the Gaussian and keep the general bounds for other mechanisms.

Lebeda et al. (2024) add two further warnings. For add/remove neighbours, the worst pair of datasets for one step need not be the worst for the composition of many steps; they give a counterexample with the subsampled Laplace mechanism, so a numerical accountant must check its worst-case pair, not assume it. And under replace-one neighbours there may be several candidate worst-case pairs, all of which an accountant must consider.

### Example: the general bound on the Gaussian

@tbl-sub-generic evaluates @eq-sub-zhu-wang with the Gaussian's curve $\varepsilon(j) = j/(2\sigma^2)$ against the exact curve.

| $\alpha$ | 2 | 4 | 8 | 16 | 32 |
| --- | --- | --- | --- | --- | --- |
| General bound @eq-sub-zhu-wang | $1.7181 \times 10^{-4}$ | $4.1880 \times 10^{-4}$ | $1.4036 \times 10^{-3}$ | 3.1611 | 11.282 |
| Exact @eq-sub-rdp | $1.7181 \times 10^{-4}$ | $3.6315 \times 10^{-4}$ | $8.9364 \times 10^{-4}$ | 3.0879 | 11.246 |
| Ratio | 1.00 | 1.15 | 1.57 | 1.02 | 1.00 |

: Zhu & Wang's general bound applied to the Gaussian against the exact RDP, $q = 0.01$, $\sigma = 1$. {#tbl-sub-generic}

The factor 3 costs nothing at $\alpha = 2$, where there are no $j \ge 3$ terms. It costs little at high orders, where the last term dominates the sum inside the logarithm, so multiplying it by 3 adds only about $\ln 3$ to the logarithm and $\ln 3/(\alpha - 1)$ to the RDP value. It costs most in between, a factor 1.57 at $\alpha = 8$, near where DP-SGD's best orders lie for $T \approx 100$ to 1000 (8.8 and 7.8 in @tbl-sub-dpsgd).

### Check yourself

- Why is the $(\varepsilon, \delta)$ amplification of @eq-sub-amplify tight in the worst case, when no RDP formula is? *(An $(\varepsilon, \delta)$ guarantee is one number per event, and randomized response attains the worst case of the mixture lemma for every $\varepsilon$. An RDP curve constrains infinitely many orders at once, and different mechanisms with the same curve can subsample differently.)*
- In @tbl-sub-generic, why is the ratio 1.00 at $\alpha = 2$? *(For $\alpha = 2$ the sum over $j \ge 3$ is empty, so the factor 3 never enters, and the remaining terms are exact for the Gaussian.)*

## Code: check every number by computation

The script [code/subsampling.py](../../code/part04-privacy-accounting/subsampling.py) computes every number in this chapter and draws every figure. The amplification formulas are small functions. The RDP of the subsampled Gaussian is computed by the binomial sum for integer orders and by integration on a grid for the rest, and the two are compared. The comparison of batch samplers reproduces checks from Chua et al. (2024) and Lebeda et al. (2024). From the repository root:

```bash
uv run python code/part04-privacy-accounting/subsampling.py
```

Its public functions are used by later chapters: `amplify_eps_delta(eps, delta, q)`, `subsampled_gaussian_rdp(alpha, q, sigma)`, `subsampled_gaussian_rdp_curve(q, sigma, orders)` (cached per $(q, \sigma)$) and `dpsgd_eps_rdp(q, sigma, steps, delta, conversion)`.

**Output** (seed 0), summarized in @tbl-sub-code:

| Experiment | Result |
| --- | --- |
| Mixture lemma, 20,000 random triples | Largest ratio of the left side's excess to $q\delta$: 1.0000 |
| Randomized response, $\varepsilon = 1$, $q = 0.1$ | Exact $\varepsilon' = 0.1586$; formula 0.1586 |
| Replace-one under Poisson, $\varepsilon = 1$, $q = 0.01$ | Group-privacy bound 0.06193 (ratio 3.64 to add/remove's 0.01704); exact pure-DP value 0.02338 (ratio 1.37) |
| One epoch, 100 batches, $\sigma = 1$ | Poisson $\le 1.21$; shuffled $\ge 4.06$; deterministic 4.38 |
| Chua et al. check, $\sigma = 0.4$, 10,000 batches | Shuffled $\delta(4) \ge 0.226$; deterministic 0.244 |
| Lebeda et al. check, $q = 0.001$, $\sigma = 0.8$, 10,000 steps | RDP: Poisson 1.38, fixed-size 15.30 |
| Subsampled Gaussian RDP, $q = 0.01$, $\sigma = 1$ | Binomial and integral agree to seven digits; reverse direction always smaller (@tbl-sub-check) |
| Small-$q$ breakdown, $q = 0.01$ | Exact first exceeds twice the leading term at $\alpha = 8.9$ ($\sigma = 1$) and 147.4 ($\sigma = 4$); $2\sigma^2\ln(1/q) = 9.2$ and 147.4 |
| Abadi et al. example, $\sigma = 4$ | Mironov 1.2586 at $\alpha = 20$; improved 1.0355 at $\alpha = 17$ |
| Running example, $T = 10{,}000$ | RDP improved 6.71 at $\alpha = 4.1$; Mironov 7.43; fixed-size batches 47.42; RDP beats per-step basic composition from $T = 4$ |

: The script's output with the default seed. {#tbl-sub-code}

[tests/test_subsampling.py](../../tests/test_subsampling.py) checks that the output still matches.

**What to take from it.** Three things. First, the numerical integral is trustworthy: it matches the exact binomial sum at every integer order checked, from $10^{-4}$ to $10^{1}$, so the fractional orders of the grid can use it. Second, published numbers reproduce: Abadi et al.'s 1.26, Chua et al.'s 0.226 and the direction of Lebeda et al.'s gap all come out of a few dozen lines. Third, the sampling scheme moves $\varepsilon$ more than the accountant does. Switching from Mironov's to the improved conversion saves about a tenth at 10,000 steps (7.43 to 6.71), while accounting for the batch sampler you actually run can multiply $\varepsilon$ by seven ($47.42/6.71 \approx 7.1$).

## References

- Kasiviswanathan, Lee, Nissim, Raskhodnikova & Smith, "What Can We Learn Privately?" (FOCS 2008; SIAM Journal on Computing 2011). The first use of sampling to amplify privacy, in private learning. Read it for the origin of the idea; the bound there is not tight.
- Li, Qardaji & Su, "On Sampling, Anonymization, and Differential Privacy: Or, k-Anonymization Meets Differential Privacy" (ACM Symposium on Information, Computer and Communications Security, ASIACCS 2012). The first proof of the tight $(\varepsilon, \delta)$ amplification for Poisson sampling.
- Balle, Barthe & Gaboardi, "Privacy Amplification by Subsampling: Tight Analyses via Couplings and Divergences" (NeurIPS 2018). Advanced joint convexity (Theorem 2), the tight $(\varepsilon, \delta)$ amplification for Poisson sampling (Theorem 8), fixed-size sampling with replace-one neighbours (Theorem 9) and sampling with replacement, and tightness via randomized response (Theorem 13). The reference for @sec-sub-eps-delta and @sec-sub-schemes.
- Abadi, Chu, Goodfellow, McMahan, Mironov, Talwar & Zhang, "Deep Learning with Differential Privacy" (CCS 2016). DP-SGD and the moments accountant; Lemma 3 is the small-$q$ bound of @sec-sub-small-q. Read it to see where the practice started and which assumptions it made, including Poisson sampling.
- van Erven & Harremoës, "Rényi Divergence and Kullback-Leibler Divergence" (IEEE Transactions on Information Theory, 2014). The properties of the Rényi divergence, including joint quasi-convexity (Theorem 13), used in step 2 of @sec-sub-rdp-gaussian.
- Mironov, Talwar & Zhang, "Rényi Differential Privacy of the Sampled Gaussian Mechanism" (arXiv 2019). The reduction to one dimension (Theorem 4), the dominance of the first direction (Theorem 5 and Corollary 7), and the numerically stable computation of $A_\alpha$ (Section 3.3). The reference for @sec-sub-rdp-gaussian and the basis of most RDP accountants.
- Wang, Balle & Kasiviswanathan, "Subsampled Rényi Differential Privacy and Analytical Moments Accountant" (AISTATS 2019). General RDP amplification for sampling without replacement (Theorem 9) and a lower bound (Proposition 11).
- Zhu & Wang, "Poisson Subsampled Rényi Differential Privacy" (ICML 2019). General RDP amplification for Poisson sampling (Theorem 5), the matching lower bound (Theorem 6), and exactness under a condition that the Gaussian and Laplace mechanisms satisfy (Theorem 8 and Proposition 10).
- Steinke, "Composition of Differential Privacy & Privacy Amplification by Subsampling" (arXiv 2022). A self-contained survey chapter with clean proofs of composition and of subsampling; the best second reading after this chapter.
- Chua, Ghazi, Kamath, Kumar, Manurangsi, Sinha & Zhang, "How Private are DP-SGD Implementations?" (ICML 2024). Shows that shuffle-and-partition batching can be much less private than the Poisson accountant reports, with the largest-batch-output test used in @tbl-sub-shuffle.
- Lebeda, Regehr, Kamath & Steinke, "Avoiding Pitfalls for Privacy Accounting of Subsampled Mechanisms under Composition" (arXiv:2405.20769, 2024; IEEE SaTML 2025). Fixed-size against Poisson sampling (the factor two in the shift, and $\varepsilon \approx 1$ against $\varepsilon > 10$), and worst-case datasets that fail under composition.
- Zhu, Dong & Wang, "Optimal Accounting of Differential Privacy via Characteristic Function" (AISTATS 2022). Dominating pairs for subsampled mechanisms (Theorem 11), used by Lebeda et al. and by [numerical accounting](06-numerical-accounting.md).
- Balle, Barthe, Gaboardi, Hsu & Sato, "Hypothesis Testing Interpretations and Rényi Differential Privacy" (AISTATS 2020). The improved conversion used in @eq-sub-dpsgd (Theorem 21).

## Cheat sheet

$q$ is the sampling rate; $P_0 = \mathcal N(0, \sigma^2)$ and $P_1 = \mathcal N(1, \sigma^2)$ are a step's output without and with the record, and $q$ is also the mixing weight of the mixture lemma. @tbl-sub-cheat collects the results.

| Result | Formula | Conditions | Tight? |
| --- | --- | --- | --- |
| Mixture lemma | $(1-q)R + qP \le e^{\varepsilon'}[(1-q)R + qP'] + q\delta$, $e^{\varepsilon'} = 1 + q(e^\varepsilon - 1)$ | $P \le e^\varepsilon P' + \delta$ and $P \le e^\varepsilon R + \delta$ on every event | Yes |
| Poisson, add/remove | $\varepsilon' = \ln(1 + q(e^\varepsilon - 1))$, $\delta' = q\delta$ | Base $(\varepsilon, \delta)$ for add/remove | Yes, per step |
| Fixed size $m$ of $n$, replace-one | Same with $q = m/n$ | Base $(\varepsilon, \delta)$ for replace-one | Yes, per step |
| Poisson, replace-one | $\varepsilon' = \ln(1 + q(e^{2\varepsilon} - 1))$, $\delta' = q(1 + e^\varepsilon)\delta$ | Group privacy on the base | No: an upper bound. For pure DP the exact value is @eq-sub-replace-pure, about twice the add/remove cost for small $\varepsilon$ |
| Fixed size, add/remove | Account the Poisson pair with noise multiplier $\sigma/2$ | Clipped Gaussian sums (Lebeda et al. 2024) | Dominating pair, each relation separately |
| Shuffle and partition | Max test: $1 - \Phi(\frac{\tau-2}\sigma)\Phi(\frac\tau\sigma)^{b-1}$ against $1 - \Phi(\frac{\tau-1}\sigma)\Phi(\frac\tau\sigma)^{b-1}$ | $b$ batches, zero-out | A lower bound on $\delta(\varepsilon)$ |
| Subsampled Gaussian RDP | $\varepsilon(\alpha) = \frac{1}{\alpha-1}\ln\mathbb E_{P_0}[(1 - q + qP_1/P_0)^\alpha]$ | Poisson, add/remove, sensitivity 1 | Exact |
| Integer orders | $A_\alpha = \sum_j\binom\alpha j(1-q)^{\alpha-j}q^je^{(j^2-j)/(2\sigma^2)}$ | Integer $\alpha$ | Exact |
| Small $q$ | $\varepsilon(\alpha) \approx \alpha q^2(e^{1/\sigma^2} - 1)/2$ | $\alpha \lesssim 2\sigma^2\ln(1/q)$ (@eq-sub-breakdown) | Leading term; within 8% up to $\sigma^2\ln(1/q)$ here |
| Abadi et al. | $\varepsilon(\alpha) \lesssim q^2\alpha/((1-q)\sigma^2)$ | $\alpha \le 1 + \sigma^2\ln\frac1{q\sigma}$, $q < \frac1{16\sigma}$, $\sigma \ge 1$ | About twice the exact value for large $\sigma$ ($2.0202/1.7181 = 1.18$ times at $\sigma = 1$, $\alpha = 2$); below it outside its order range |
| High orders | $\varepsilon(\alpha) \ge \frac{\alpha\ln q}{\alpha-1} + \frac{\alpha}{2\sigma^2}$ (@eq-cdp-sg-lower) | Any $\alpha$ | Nearly exact once $\alpha \gtrsim 2\sigma^2\ln(1/q)$; negative, so useless, well below it |
| DP-SGD, RDP accountant | @eq-sub-dpsgd: $T\,\varepsilon(\alpha)$, converted at the best order | Poisson, add/remove, adaptive steps | Upper bound; 6.71 for the running example |
| Any mechanism, Poisson | @eq-sub-zhu-wang | Integer $\alpha$ | Within a factor 3 inside the logarithm |

: Every result of the chapter. {#tbl-sub-cheat}

**Rules of thumb**

- Sampling at rate $q$ multiplies a small $\varepsilon$ by $q$, but takes at most $\ln(1/q)$ off a large one.
- In RDP the per-step cost is quadratic in $q$: about $\alpha q^2/(2\sigma^2)$ for large $\sigma$, valid up to $\alpha \approx 2\sigma^2\ln(1/q)$, and it breaks abruptly there.
- Never compose per-step $(\varepsilon, \delta)$ for DP-SGD. Compose RDP curves, or privacy loss distributions, and convert once.
- Account for the sampler you run. Poisson pairs with add/remove. Fixed-size batches need the base mechanism's replace-one guarantee, which for clipped Gaussian sums means half the noise multiplier. Shuffle and partition has no simple amplification theorem, and the Poisson number can be far too small for it.
- For the Gaussian, use the exact binomial sum and the numerical integral; keep the general bounds for other mechanisms.

**Practice.** Exercises for this chapter, each with a worked solution, are in [the exercise book](../../book-exercises/part04-privacy-accounting/05-subsampling.md).
