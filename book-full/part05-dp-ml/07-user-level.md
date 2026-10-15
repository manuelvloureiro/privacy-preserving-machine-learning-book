# User-level differential privacy

## Why this matters

Every differential privacy (DP) guarantee in this part so far protects one record. Stochastic gradient descent (SGD) with per-example clipping and Gaussian noise, [DP-SGD](02-dp-sgd.md), clips each record's gradient, adds noise scaled to that clip, and reports $\varepsilon = 6.71$ at $\delta = 10^{-5}$ for the part's running example: an adversary who sees the trained model learns little about whether any single training example was used. But the people behind the data do not contribute one record each. A user of a messaging app contributes hundreds of messages, a patient dozens of visits, a photographer thousands of photos. Each of those records is protected by DP, and the person is not, because what the adversary learns about all of a person's records together can be much larger than what they learn about one.

This chapter asks two questions. What does a record-level $\varepsilon$ say about a person who contributed $k$ records? And how do you train a model with a guarantee for the person instead?

### The one idea: the neighbouring relation is the privacy unit

A DP guarantee compares the outputs on two **neighbouring** datasets, and whatever differs between neighbours is what is protected. If neighbours differ in one record, records are protected; if they differ in all of one person's records, people are. Nothing else in the definition changes. The work is to bound how much the *unit*, now a person, can move the computation, and to account for that bound.

There are two ways to get a guarantee for a person. The first keeps the record as the unit and pays for the person afterwards, by group privacy or by an accountant that knows the person has $k$ records. It is cheap to implement and expensive in privacy, because the noise was calibrated to one record while the person moves the output by up to $k$. The second makes the person the unit from the start: sample people instead of records, bound each person's total contribution by clipping it, and run the usual accountant with people in place of records. On the running example the first route gives a useless model at any reasonable budget, and the second gives one within a few points of the non-private model.

@tbl-ulvl-overview lists the results and where each is used.

| Result | Where it is used |
| --- | --- |
| The privacy unit and group privacy | Reading a record-level guarantee as a statement about people |
| The accountant for a user with $k$ records under example-level sampling | Auditing what an existing DP-SGD model guarantees to heavy users |
| User-level DP-SGD | Training with a guarantee per person; federated learning |
| Contribution bounding and its bias–variance trade-off | Keeping an example-level pipeline while protecting people |
| Heterogeneous users | Deciding how to weight heavy and light users |
| The strategies compared on the running example | Choosing among the three in practice |

: The results of this chapter and where each is used. {#tbl-ulvl-overview}

**Notation.** The training set has $n$ records belonging to $U$ users; user $u$ owns $n_u$ records. Two datasets are **example-level neighbours** if they differ by adding or removing one record, and **user-level neighbours** if they differ by adding or removing all the records of one user. $k$ is the number of records a user contributes, or the cap on it. In a batch drawn by Poisson sampling at rate $q$, $M$ is the number of a given user's records that are drawn. As in [DP-SGD](02-dp-sgd.md), $C$ is the clipping norm, $\sigma$ the noise multiplier, $T$ the number of steps and $B$ the expected batch size, now counted in whichever unit is sampled.

**The running example.** The part's network (one hidden layer of 64 tanh units, 3,329 weights) learns labels produced by a fixed teacher network from 10,000 training records of 50 features, and is tested on 20,000 more. This chapter assigns the 10,000 records to $U = 1{,}000$ users. Each user gets one record, and the other 9,000 are spread over users with probabilities proportional to log-normal weights, so user sizes are very uneven: the mean is 10 records, the median 4, and the largest user holds 348 (@fig-ulvl-population). A user's records are a nearest-neighbour cluster around a random record, so they resemble each other the way one person's messages do: the label balance $|\bar y|$ is 0.21 on average in the ten largest users, against 0.06 in the whole set. The population is fixed, like the dataset.

![Left: the number of records per user in the running example's population of 1,000 users, on log–log axes; 193 users have one record and the largest has 348. Right: the share of all records held by the largest users. The largest 10 users hold 17.9% of the records and the smallest half of the users 10.3%.](figures/ulvl_population.svg){#fig-ulvl-population}

**How to read this chapter.** The toolkit restates three facts from earlier parts. @sec-ulvl-unit and @sec-ulvl-group-sgd say what a record-level guarantee gives a person; the second has the chapter's main derivation, the accountant for a user with $k$ records. @sec-ulvl-dpsgd and @sec-ulvl-bounding are the two ways to train with a user-level guarantee, @sec-ulvl-hetero asks what they do to heavy users, and @sec-ulvl-run compares everything on the running example.

## Toolkit: three facts every derivation uses

### Fact 1 — Group privacy

If a mechanism is $(\varepsilon, \delta)$-DP for neighbours that differ in one record, then for two datasets that differ in $k$ records it satisfies, for every event $S$,
$$\Pr[M(D) \in S] \le e^{k\varepsilon}\Pr[M(D'') \in S] + \delta_k, \qquad \delta_k = \delta\sum_{j=0}^{k-1}e^{j\varepsilon} = \delta\,\frac{e^{k\varepsilon} - 1}{e^\varepsilon - 1}.$$ {#eq-ulvl-group}
The reason is a chain: $D$ and $D''$ are joined by a path of $k$ steps, each between neighbouring datasets. Applying the one-record guarantee once per link multiplies the probability bound by $e^\varepsilon$ each time, and each link's slack $\delta$ is then multiplied by the $e^\varepsilon$ of every later link, which is where the geometric sum comes from. The bound is attained by some mechanisms, so no better one holds for every $(\varepsilon, \delta)$-DP mechanism ([Post-processing, group privacy and composition](../part03-dp-foundations/02-properties.md)). For Rényi differential privacy (RDP) the chain is weaker still, because the Rényi divergence has no triangle inequality: an $(\alpha, \varepsilon)$-RDP mechanism is $(\alpha/2^c, 3^c\varepsilon)$-RDP for groups of $2^c$ records, when $\alpha \ge 2^{c+1}$ (Mironov 2017; [Rényi DP](../part04-privacy-accounting/01-renyi-dp.md)). Each doubling of the group halves the order and triples $\varepsilon$.

### Fact 2 — The accountant of DP-SGD

One step of DP-SGD adds Gaussian noise of standard deviation $\sigma C$ to a sum of clipped per-example gradients computed on a Poisson batch: each record enters independently with probability $q$. In units of $C$, the worst case for one record is the pair $P_0 = \mathcal N(0, \sigma^2)$, the output without the record, and the mixture $(1 - q)P_0 + q\mathcal N(1, \sigma^2)$, the output with it. Its RDP curve at order $\alpha$, the Rényi divergence of order $\alpha$ between the two, is $\varepsilon_{q,\sigma}(\alpha) = \frac{1}{\alpha - 1}\ln A_\alpha$, and for integer $\alpha$
$$A_\alpha = \sum_{j=0}^\alpha\binom\alpha j(1 - q)^{\alpha - j}q^j\exp\Big(\frac{j^2 - j}{2\sigma^2}\Big).$$ {#eq-ulvl-binomial}
The sum comes from expanding $\big((1 - q) + q\,r\big)^\alpha$, with $r$ the Gaussian likelihood ratio, by the binomial theorem and averaging each power $r^j$ under $P_0$ ([Amplification by subsampling](../part04-privacy-accounting/05-subsampling.md)). RDP curves add over the $T$ steps, even when each step depends on earlier outputs, and the total converts to $(\varepsilon, \delta)$ at the best order:
$$\varepsilon = \min_\alpha\Big[T\varepsilon_{q,\sigma}(\alpha) + \ln\frac{\alpha - 1}\alpha - \frac{\ln\delta + \ln\alpha}{\alpha - 1}\Big].$$ {#eq-ulvl-convert}
The terms $j = 0$ and $1$ carry no privacy cost, since their exponential factor is 1. For small $q$ and moderate orders only the $j = 2$ term matters beyond them, so $A_\alpha \approx 1 + \binom\alpha2q^2(e^{1/\sigma^2} - 1)$ and $\varepsilon_{q,\sigma}(\alpha) \approx \frac\alpha2q^2(e^{1/\sigma^2} - 1)$.

### Fact 3 — The Gaussian moment generating function

If $Y \sim \mathcal N(0, \sigma^2)$, then $\mathbb E[e^{sY}] = e^{s^2\sigma^2/2}$ for every real $s$ ([Distributions](../part01-prerequisites/01-distributions.md)). Every Rényi divergence between Gaussians with the same variance reduces to it, because their likelihood ratio is the exponential of a linear function of the output: for $P_m = \mathcal N(m, \sigma^2)$,
$$\frac{P_m(y)}{P_0(y)} = \exp\Big(\frac{-(y - m)^2 + y^2}{2\sigma^2}\Big) = \exp\Big(\frac{2my - m^2}{2\sigma^2}\Big).$$ {#eq-ulvl-ratio}

## The privacy unit {#sec-ulvl-unit}

### Context

A model trained with record-level DP-SGD at $\varepsilon = 6.71$ is released. One of the people in the training set contributed $k$ of its records. What, exactly, does the guarantee promise that person? The question has to be answered before a deployment can say "your data are protected", because the promise is about whatever the neighbouring relation changes, and that relation was chosen by the engineer, not by the person.

### Statement

A mechanism is **user-level** $(\varepsilon, \delta)$-DP if for every pair of datasets $D$, $D'$ that differ by adding or removing all the records of one user, and every event $S$,
$$\Pr[M(D) \in S] \le e^\varepsilon\Pr[M(D') \in S] + \delta.$$
It is **example-level** DP if the same holds for datasets that differ in one record. A mechanism that is example-level $(\varepsilon, \delta)$-DP is user-level $(k\varepsilon, \delta_k)$-DP for users with at most $k$ records, with $\delta_k$ from @eq-ulvl-group. Without more information about the mechanism, nothing better can be said.

### Derivation

1. **User-level neighbours are $k$ example-level steps apart.** If $D'$ is $D$ plus the $k$ records of one user, add them one at a time. The datasets $D, D \cup\{x_1\}, D\cup\{x_1, x_2\}, \dots, D'$ form a chain in which each is an example-level neighbour of the next.
2. **Apply group privacy.** Fact 1 along this chain gives @eq-ulvl-group with $D'' = D'$, in both directions, since the chain can be walked either way.
3. **Nothing better holds in general.** The bound is attained by a mechanism built for the purpose, which outputs an alarm with probabilities chosen so that every link's inequality is an equality ([Post-processing, group privacy and composition](../part03-dp-foundations/02-properties.md)). A guarantee that is only known as "$(\varepsilon, \delta)$-DP" could come from that mechanism.

### Interpretation

**The definition is the same; the unit is not.** User-level DP is not a new privacy notion. It is DP with a coarser neighbouring relation, so every property proved for DP holds for it: post-processing, composition and the accountants of Part 4. The only thing that changes is the sensitivity: how far one unit can move the computation. Records, users, households, devices and organizations are all possible units, and the choice should follow the question "whose participation do I want to hide?" A deployment that says "DP" without naming the unit leaves that question open, and [Choosing an accountant](../part04-privacy-accounting/09-choosing-an-accountant.md) lists the unit among the things a guarantee must report.

**Group privacy is linear in $\varepsilon$ and exponential in $\delta$.** For pure DP a user with $k$ records gets $k\varepsilon$, already weak: at $\varepsilon = 1$ per record, a user with 10 records gets $\varepsilon = 10$, a likelihood ratio of $e^{10} \approx 22{,}000$. With $\delta > 0$ it is worse. In @eq-ulvl-group each $\delta$ is multiplied by up to $e^{(k-1)\varepsilon}$, so to keep the user's $\delta_k$ at $10^{-5}$ the per-record $\delta$ must be tiny, and a smaller $\delta$ means a larger per-record $\varepsilon$. For DP-SGD this trade can be made at any $\delta$, because the accountant's RDP curve converts to $(\varepsilon, \delta)$ at every $\delta$: choose the per-record $\delta$ so that $\delta_k = 10^{-5}$ exactly.

**For DP-SGD it can fail altogether.** The per-record $\varepsilon$ grows as the per-record $\delta$ shrinks, so $\delta_k = \delta\frac{e^{k\varepsilon} - 1}{e^\varepsilon - 1}$ is a race between a falling $\delta$ and a rising $e^{(k-1)\varepsilon}$. For small $k$ the falling $\delta$ wins. For larger $k$ the rising $\varepsilon$ wins at every $\delta$, and no choice brings $\delta_k$ down to $10^{-5}$. Group privacy then gives no $(\varepsilon, 10^{-5})$ guarantee at all ([Exercise 2](../../book-exercises/part05-dp-ml/07-user-level.md#sec-sol-ulvl-2) traces the race).

### Example: the running example's users

The running example's model is example-level $(6.71, 10^{-5})$-DP. Read naively as a user-level statement, a user with two records gets $\varepsilon = 2 \times 6.71 = 13.42$ with $\delta_2 = 10^{-5}(1 + e^{6.71}) = 0.0082$: a probability of almost one in a hundred that the guarantee fails outright, which is no guarantee. Re-converting the accountant's RDP curve at a smaller per-record $\delta$ does better. The per-record $\delta$ that makes $\delta_2 = 10^{-5}$ gives a per-record $\varepsilon$ of 9.30, so the user gets $(18.61, 10^{-5})$. For five records the same procedure gives a per-record $\varepsilon$ of 23.62 and a user-level $\varepsilon$ of 118.12. For ten records it gives nothing: no per-record $\delta$ makes $\delta_{10}$ as small as $10^{-5}$. In the running example's population, 244 of the 1,000 users have at least ten records (the median is 4), and 30 have at least 50.

### Check yourself

- A dataset has one row per hospital visit, and a mechanism is $0.2$-DP (pure) at the visit level. What does a patient with 7 visits get? *($1.4$-DP, by group privacy with $k = 7$; for pure DP the bound is just $k\varepsilon$.)*
- Why does a user-level guarantee still compose and survive post-processing? *(Because it is DP with a different neighbouring relation, and those proofs hold for any fixed relation: they never use what a neighbour is.)*

## A user with k records in example-level DP-SGD {#sec-ulvl-group-sgd}

### Context

Group privacy treats DP-SGD as a black box. It is not one: we know exactly how a user with $k$ records enters each step. Each of the user's records is drawn into the batch independently with probability $q$, so in each step a random number of them, between 0 and $k$, contributes to the noisy sum. An accountant that knows this can compute the user's guarantee directly, as [Amplification by subsampling](../part04-privacy-accounting/05-subsampling.md) did for one record. How large is that guarantee, and how does it grow with $k$?

### Statement

Let $M \sim \mathrm{Binomial}(k, q)$ be the number of the user's records in a Poisson batch, and $b_m = \binom km q^m(1 - q)^{k - m}$ its probabilities. In units of $C$, one step of example-level DP-SGD compares
$$P_0 = \mathcal N(0, \sigma^2) \qquad\text{and}\qquad P_{(k)} = \sum_{m=0}^k b_m\,\mathcal N(m, \sigma^2),$$ {#eq-ulvl-mixture}
the output without the user and with the user, whose drawn records all push the sum by the full $C$ in the same direction. Charles et al. (2024, Theorem 3) prove that this pair, composed over the $T$ steps, bounds the user's privacy loss, and that the bound is attained by a loss for which the user's gradients are aligned. The user's RDP curve is therefore $\varepsilon_{k}(\alpha) = \max\{D_\alpha(P_{(k)}\|P_0), D_\alpha(P_0\|P_{(k)})\}$. The first divergence is $\frac{1}{\alpha - 1}\ln A_\alpha$ with
$$A_\alpha = \mathbb E\Big[\exp\Big(\frac{1}{\sigma^2}\sum_{1\le i<j\le\alpha}M_iM_j\Big)\Big], \qquad M_1, \dots, M_\alpha \text{ independent copies of } M,$$ {#eq-ulvl-pair}
for integer $\alpha$. For small $kq$ the curve is close to that of a single record sampled at rate $kq$:
$$\varepsilon_k(\alpha) \approx \frac\alpha2(kq)^2\big(e^{1/\sigma^2} - 1\big).$$ {#eq-ulvl-rate}
The user's $(\varepsilon, \delta)$ after $T$ steps is @eq-ulvl-convert applied to $T\varepsilon_k(\alpha)$.

### Derivation

The derivation follows the one for a single record: find the distribution of one step's output with and without the user, write the Rényi divergence as an average of a power of the likelihood ratio, and evaluate the average. The new ingredient is that the power of a mixture over $m$ becomes a product of independent copies of $M$, which makes the average computable by Fact 3.

1. **How many of the user's records are drawn.** Poisson sampling draws each record independently with probability $q$, so the number of the user's $k$ records in the batch is a sum of $k$ independent coins: $M \sim \mathrm{Binomial}(k, q)$, independent of which other records are drawn.
2. **What they do to the sum.** Fix the other records' contribution, which is the same with and without the user. Given $M = m$, the user adds $m$ clipped gradients, each of length at most $C$, so their sum has length at most $mC$, with equality when they all point the same way. In units of $C$ and along that direction, the output is $\mathcal N(m, \sigma^2)$ with probability $b_m$: the mixture of @eq-ulvl-mixture. That the aligned case is the worst over all configurations of the user's gradients, and that the reduction to one dimension is valid, is the content of Charles et al.'s Theorem 3; the intuition is that gradients pointing in different directions add in squares rather than in length, and spread the user's influence over several coordinates where it is harder to detect.
3. **The divergence as an average.** By the definition of the Rényi divergence, $e^{(\alpha-1)D_\alpha(P_{(k)}\|P_0)} = \mathbb E_{y\sim P_0}\big[(P_{(k)}(y)/P_0(y))^\alpha\big]$. Dividing the mixture by $P_0$ term by term with @eq-ulvl-ratio,
$$\frac{P_{(k)}(y)}{P_0(y)} = \sum_m b_m\exp\Big(\frac{2my - m^2}{2\sigma^2}\Big) = \mathbb E_M\Big[\exp\Big(\frac{2My - M^2}{2\sigma^2}\Big)\Big],$$
an average over the number of drawn records.
4. **The power of an average is an average over independent copies.** For integer $\alpha$, $(\mathbb E_M[f(M)])^\alpha = \mathbb E[f(M_1)f(M_2)\cdots f(M_\alpha)]$ with $M_1, \dots, M_\alpha$ independent copies of $M$: multiplying out $\alpha$ copies of the sum $\sum_mb_mf(m)$ gives one term for every choice $(m_1, \dots, m_\alpha)$, with weight $b_{m_1}\cdots b_{m_\alpha}$, which is the joint distribution of independent copies. The product of exponentials is the exponential of the sum, so with $S = M_1 + \dots + M_\alpha$,
$$\Big(\frac{P_{(k)}(y)}{P_0(y)}\Big)^\alpha = \mathbb E\Big[\exp\Big(\frac{2Sy - \sum_iM_i^2}{2\sigma^2}\Big)\Big].$$
5. **Average over $y$.** Exchange the two averages, which is allowed because everything is positive. For fixed copies, Fact 3 with $s = S/\sigma^2$ gives $\mathbb E_{y\sim P_0}[e^{Sy/\sigma^2}] = e^{S^2/(2\sigma^2)}$, so
$$A_\alpha = \mathbb E\Big[\exp\Big(\frac{S^2 - \sum_iM_i^2}{2\sigma^2}\Big)\Big].$$
Expanding the square, $S^2 = \sum_iM_i^2 + 2\sum_{i<j}M_iM_j$, and the squares cancel: this is @eq-ulvl-pair. Only products of *different* copies remain.
6. **Check against one record.** For $k = 1$, each $M_i$ is a coin, 0 or 1. If $J$ of the $\alpha$ coins are 1, the sum $\sum_{i<j}M_iM_j$ counts the pairs among them, $\binom J2 = (J^2 - J)/2$, and $J \sim \mathrm{Binomial}(\alpha, q)$. The average of $e^{(J^2 - J)/(2\sigma^2)}$ over $J$ is @eq-ulvl-binomial.
7. **Small $kq$.** When $kq$ is small, most copies are 0. The terms of @eq-ulvl-pair with at most one non-zero copy have $\sum_{i<j}M_iM_j = 0$ and contribute their probability times 1. The first term that costs privacy has exactly two copies equal to 1, the most likely way for a product to be non-zero: there are $\binom\alpha2$ pairs of copies, each copy is 1 with probability about $kq$, and the factor is $e^{1/\sigma^2}$. So $A_\alpha \approx 1 + \binom\alpha2(kq)^2(e^{1/\sigma^2} - 1)$, the same as @eq-ulvl-binomial's first three terms with $q$ replaced by $kq$. Taking $\ln(1 + x) \approx x$ and dividing by $\alpha - 1$ gives @eq-ulvl-rate.
8. **Other orders and the other direction.** For non-integer $\alpha$, and for $D_\alpha(P_0\|P_{(k)})$, there is no finite expansion. Compute both as one-dimensional integrals of $P_0(y)$ times a power of the mixture ratio of step 3, on a grid of step $\sigma/10$ wide enough to contain the bump near $y = \alpha k$ that large orders create, with log-sum-exp to avoid overflow, exactly as for one record.

### Interpretation

**A user with $k$ records looks like one record sampled $k$ times as often.** @eq-ulvl-rate says that, in the regime where amplification works well, the only thing that matters is the probability that some record of the user is in the batch, about $kq$. Halving the sampling rate exactly compensates doubling the user's records. Since the per-step RDP grows like $(kq)^2$, and $\varepsilon$ after conversion grows roughly like the square root of the RDP slope, the user's $\varepsilon$ grows roughly linearly in $k$ while $kq$ is small.

**Beyond that regime it grows faster.** Two things break the approximation. First, $kq$ is no longer small: for $k = 50$ and $q = 0.01$, half of all steps include one of the user's records, and subsampling, which helps only because the unit is usually absent, stops helping. Second, the higher terms of @eq-ulvl-pair grow like $e^{M_iM_j/\sigma^2}$, and $M_iM_j$ can be as large as $k^2$. A step that draws all $k$ records shifts the sum by $kC$, a mechanism with noise multiplier $\sigma/k$, whose privacy loss has a tail that no amount of rarity can hide at high orders. The accountant then uses orders close to 1, where the conversion is expensive.

**Group privacy is the wrong tool for DP-SGD.** It ignores both the structure of the mechanism and the fact that the RDP curve is known. The direct accountant gives a smaller $\varepsilon$ for every $k$ and a finite one for every $k$.

**The accountant is a worst case over gradients, not over users.** It assumes the user's drawn records push the sum in the same direction by the full $C$. Real users' gradients are partly aligned, which is precisely why their records are correlated, and the theorem says some loss attains the bound. An auditor should therefore treat it as the guarantee, not as a pessimistic estimate.

![The user-level $\varepsilon$ at $\delta = 10^{-5}$ of the running example's DP-SGD ($q = 0.01$, $\sigma = 1$, 10,000 steps; $\varepsilon = 6.71$ per record) for a user with $k$ records. The direct accountant of @eq-ulvl-mixture (black) is close to a single record sampled at rate $kq$ (blue, dotted) and grows faster than linearly (grey). Group privacy from the record-level guarantee (red, dashed) is larger, and gives no guarantee at all for $k = 10$ and above.](figures/ulvl_user_epsilon.svg){#fig-ulvl-user-eps}

### Example: the running example's guarantee for its users

@tbl-ulvl-k applies the accountant to the running example's DP-SGD run, $q = 0.01$, $\sigma = 1$, $T = 10{,}000$, $\delta = 10^{-5}$.

| Records of the user $k$ | 1 | 2 | 5 | 10 | 50 |
| --- | --- | --- | --- | --- | --- |
| User accountant, @eq-ulvl-mixture | 6.72 | 15.54 | 52.36 | 146.20 | 2161.43 |
| Best order $\alpha$ | 4 | 2.5 | 1.65 | 1.3 | 1.05 |
| Group privacy, @eq-ulvl-group | 6.71 | 18.61 | 118.12 | none | none |
| One record at rate $kq$ | 6.71 | 15.32 | 50.11 | 134.50 | 1673.08 |

: The user-level $\varepsilon$ at $\delta = 10^{-5}$ of the running example's example-level DP-SGD, for a user with $k$ records. "None" means that group privacy gives no $(\varepsilon, 10^{-5})$ guarantee. {#tbl-ulvl-k}

The first column checks the accountant: with one record it is the DP-SGD accountant, 6.72 against 6.71, the difference coming from the coarser grid of orders this chapter's script uses (its best order is 4, against 4.1 on the finer grid). At $\alpha = 4.1$ the two curves agree to all printed digits, $3.733633 \times 10^{-4}$. For $k = 2$ and 5 at $\alpha = 2$, the numerical integral and the exact formula @eq-ulvl-pair, $A_2 = \mathbb E[e^{M_1M_2/\sigma^2}]$, agree to seven digits ($6.992273 \times 10^{-4}$ and $4.626224 \times 10^{-3}$), and the reverse direction is smaller ($6.119238 \times 10^{-4}$ and $3.412123 \times 10^{-3}$).

Already at two records, the person behind them is protected at $\varepsilon = 15.54$, more than twice the record's 6.71. At ten, $\varepsilon = 146.20$ means no protection at all. The rate-$kq$ approximation is within 1.5% at $k = 2$ (15.32 against 15.54) and 8% at $k = 10$ (134.50 against 146.20), so it is a quick estimate of the user's exposure: rerun the record-level accountant with $q$ replaced by $kq$. In every row it errs on the low side, because it drops the terms in which several of the user's records are drawn together.

### Check yourself

- Your model was trained with example-level DP-SGD at $q = 0.001$, and its heaviest user has 20 records. Which single-record accountant run gives a quick estimate of that user's $\varepsilon$, and in which direction is it off? *(The run with $q = 0.02$ and the same $\sigma$ and $T$. It underestimates slightly, because it ignores steps that draw several of the user's records at once.)*
- In @eq-ulvl-pair, why do only products of different copies appear? *(Because the $\sum_iM_i^2$ in the exponent of the likelihood ratio cancels the squares in $S^2$ after the Gaussian average. Each copy alone averages to a ratio whose expectation under $P_0$ is 1, as for one record.)*

## User-level DP-SGD {#sec-ulvl-dpsgd}

### Context

Accounting for users after the fact charges a heavy user for every one of their records. The alternative changes the algorithm, so that the unit DP-SGD samples, clips and accounts for is the user. The idea is due to McMahan, Ramage, Talwar & Zhang (2018), who used it to train a language model with federated learning, where each user's device computes an update on its own data and the server only sees the updates. What does the algorithm compute, and what does it guarantee?

### Statement

Start from weights $w_0$ chosen independently of the data. For $t = 0, \dots, T - 1$:

1. **Sample users.** Put each user in the set $S_t$ independently with probability $q$.
2. **Per-user update.** For each $u \in S_t$, compute one vector from user $u$'s records alone, for example the average gradient $\bar g_u = \frac1{n_u}\sum_{i \in u}\nabla_w\ell(w_t, z_i)$, or the change in weights after a few local steps of SGD on those records.
3. **Clip per user.** $\tilde g_u = \bar g_u\min(1, C/\|\bar g_u\|)$.
4. **Noisy sum.** $G_t = \sum_{u\in S_t}\tilde g_u + \xi_t$ with $\xi_t \sim \mathcal N(0, \sigma^2C^2I)$.
5. **Step.** $w_{t+1} = w_t - \eta\,G_t/(qU)$.

Releasing $w_1, \dots, w_T$ is user-level $(\varepsilon, \delta)$-DP with $\varepsilon$ given by @eq-ulvl-convert for the curve $T\varepsilon_{q,\sigma}(\alpha)$ of Fact 2: exactly the accountant of example-level DP-SGD, with users in place of records and $q$ the user sampling rate.

A common variant replaces steps 2 and 3 by **per-record clipping and a cap**: keep at most $k$ records per user, clip each record's gradient to $C$, and sum them. One user then moves the sum by at most $kC$, and the guarantee is that of noise multiplier $\sigma/k$:
$$A_\alpha = \sum_{j=0}^\alpha\binom\alpha j(1 - q)^{\alpha - j}q^j\exp\Big(\frac{k^2(j^2 - j)}{2\sigma^2}\Big),$$ {#eq-ulvl-together}
@eq-ulvl-binomial with $\sigma$ replaced by $\sigma/k$.

### Derivation

The proof of [DP-SGD](02-dp-sgd.md) used four properties of the clipped sum. Each holds with users in place of records.

1. **One user moves the clipped sum by at most $C$.** Fix the weights $w_t$ and the other sampled users. Each $\tilde g_v$ is computed from user $v$'s records alone, so adding user $u$ to the dataset leaves every other term unchanged and adds $\tilde g_u$, of length at most $C$ by step 3. However many records $u$ has, they enter only through $\tilde g_u$.
2. **One step is a subsampled Gaussian mechanism over users.** User $u$ enters $S_t$ with probability $q$, independently of the other users. So the step's output with $u$ is a mixture: with probability $1 - q$ the same as without $u$, with probability $q$ shifted by $\tilde g_u$. After rescaling by $C$ and rotating the shift onto one axis, the worst case is the pair of Fact 2, and the step's RDP curve is $\varepsilon_{q,\sigma}(\alpha)$.
3. **Dividing by $qU$ is post-processing**, since the number of users $U$ is treated as public, as $n$ is in example-level DP-SGD.
4. **Compose and convert.** RDP composition over the $T$ adaptive steps and @eq-ulvl-convert finish the proof, as before.
5. **The capped-sum variant.** With per-record clipping, user $u$'s contribution is a sum of at most $k$ vectors of length at most $C$, so its length is at most $kC$, and the user is still sampled as a whole with probability $q$. In units of $C$ the pair is $\mathcal N(0, \sigma^2)$ against $(1 - q)\mathcal N(0, \sigma^2) + q\mathcal N(k, \sigma^2)$. Dividing the output by $k$, an invertible map that changes no divergence, gives the pair with shift 1 and noise $\sigma/k$, whose curve is @eq-ulvl-binomial with $\sigma/k$: that is @eq-ulvl-together. The script checks it at $k = 2$, $\alpha = 3$: both give $8.219438 \times 10^{-2}$.

### Interpretation

**The count that matters is the number of units.** User-level DP-SGD has the privacy of example-level DP-SGD on a dataset of $U$ records. To keep the same expected batch of $B$ units, the sampling rate must be $B/U$, larger than $B/n$ by the average number of records per user, and amplification is correspondingly weaker. On the running example, with 1,000 steps and an expected batch of 100 units, a record-level $\varepsilon = 4$ needs $\sigma = 0.778$ at $q = 0.01$, and a user-level $\varepsilon = 4$ needs $\sigma = 3.78$ at $q = 0.1$. The noise on the averaged update, $\sigma C/B$, is 0.0189 against 0.00389, about five times larger. This is the honest price of protecting people: there are ten times fewer of them than records, and DP's cost per unit is fixed.

**Records inside a user are free.** The guarantee does not depend on $n_u$. A user's records can be averaged, used for several local steps, or subsampled, all without changing $\varepsilon$, as long as the result is one clipped vector computed from that user's data alone. More records per user make each $\bar g_u$ a better estimate of the user's gradient, so the signal improves while the noise stays fixed. Levy et al. (2021) show that, when each user's records come from the same distribution, user-level private learning gains from having more records per user.

**Average, or sum then clip?** Averaging before clipping gives every user a vector of the same kind regardless of size, which keeps the clipping norm meaningful across users. Clipping each record and summing at most $k$ of them uses per-example gradient code that already exists, but pays a factor $k$ in noise multiplier (@eq-ulvl-together). Summing all of a user's records and then clipping the sum is also valid (sensitivity $C$), and gives heavy users more weight than averaging does, as @sec-ulvl-hetero shows.

**What the proof assumes.** Step 1 needs every user's contribution to be computed from that user alone, which rules out cross-user normalization and requires that one user's records are not spread across several "users" in the data. If one person has several accounts, the unit is the account, and the person is again a group. Step 2 needs users to be Poisson-sampled; an implementation that iterates over users in a fixed shuffled order is a different mechanism, as for records.

### Example: the noise each strategy needs

@tbl-ulvl-sigma fixes 1,000 steps with an expected batch of 100 sampled units and gives the noise multiplier for four user-level budgets. The column "record-level" is not a user-level guarantee; it shows the noise the same schedule would need to protect only records. Column (a) is example-level DP-SGD on all 10,000 records ($q = 0.01$), accounted for the largest user with $k = 348$ by @eq-ulvl-mixture; (b) is user-level DP-SGD ($q = 0.1$); (c) keeps at most 2 records per user, 1,807 records in all, and runs example-level DP-SGD on them with $q = 100/1{,}807 = 0.0553$, accounted for $k = 2$ (@sec-ulvl-bounding).

| User-level $\varepsilon$ | Record-level (not user-level) | (a) All records, $k = 348$ | (b) User-level DP-SGD | (c) At most 2 records per user |
| --- | --- | --- | --- | --- |
| 1 | 1.52 | 449 | 12.9 | 14.4 |
| 2 | 1.03 | 237 | 6.89 | 7.62 |
| 4 | 0.778 | 128 | 3.78 | 4.18 |
| 8 | 0.616 | 70.3 | 2.18 | 2.4 |

: Noise multipliers for 1,000 steps with an expected batch of 100 units, at $\delta = 10^{-5}$. {#tbl-ulvl-sigma}

Column (a) is the cost of keeping the record as the unit when one user has 348 records: in each step this user has on average $348 \times 0.01 = 3.48$ records in the batch, so subsampling hardly hides them, and the noise must cover a shift of several clipping norms in nearly every step. At $\varepsilon = 4$ the noise on the averaged gradient has standard deviation $128 \times 0.5/100 = 0.64$ per coordinate, so length about $0.64 \times \sqrt{3{,}329} = 36.9$, against a clipped average of length at most $0.5$.

### Check yourself

- In user-level DP-SGD, a user with 300 records runs five local steps of SGD on them and sends the clipped change in weights. Does that change $\varepsilon$? *(No. The user's contribution is still one vector computed from their records alone, clipped to $C$; the local computation is invisible to the accountant.)*
- Why does user-level DP-SGD on 1,000 users need more noise than example-level DP-SGD on 10,000 records, at the same expected batch, steps and $\varepsilon$? *(To keep the batch at 100 units it must sample at rate 0.1 instead of 0.01. Amplification by subsampling is weaker at a higher rate, so the same $\varepsilon$ needs a larger $\sigma$.)*

## Contribution bounding {#sec-ulvl-bounding}

### Context

User-level DP-SGD needs a data pipeline that groups records by user and computes one vector per user. Many training pipelines are built around records: shuffled files of examples, per-example gradient code, fixed-length batches. Can you keep that pipeline and still protect users? You can, if you limit how many records each user contributes. The question is what limit to choose, because the limit trades the noise the accountant requires against the data you throw away.

### Statement

**Contribution bounding** keeps at most $k$ records of each user, chosen without looking at other users' data, for example at random, and discards the rest. Run example-level DP-SGD on the $n_k = \sum_u\min(n_u, k)$ remaining records with sampling rate $q$. The result is user-level $(\varepsilon, \delta)$-DP with $\varepsilon$ given by the accountant of @eq-ulvl-mixture for a user with $k$ records. The trade-off:

- the **noise** grows with $k$: in the small-$kq$ regime the user's RDP is that of a record sampled at rate $kq$ (@eq-ulvl-rate), so the noise multiplier must grow roughly in proportion to $k$ at a fixed $q$;
- the **bias** shrinks with $k$: the training set loses $n - n_k$ records, all of them from users with more than $k$ records, so the data are both smaller and reweighted toward light users.

### Derivation

1. **Bounding is applied per user.** Write the bounded dataset as $\Pi(D) = \bigcup_u\pi(D_u)$, where $D_u$ is user $u$'s records and $\pi$ keeps at most $k$ of them using randomness that is independent of everything else. Adding user $u$ to $D$ adds $\pi(D_u)$ to $\Pi(D)$ and changes nothing else, so user-level neighbours map to datasets that differ in at most $k$ records, all of one user.
2. **Account for those records.** Example-level DP-SGD on $\Pi(D)$ sees the user's at most $k$ records, each drawn independently with probability $q$. By @sec-ulvl-group-sgd, the pair for one step is @eq-ulvl-mixture with $k$. A user with fewer records is covered too: they are a $k$-record user some of whose clipped gradients happen to be 0, and the worst case of @sec-ulvl-group-sgd is over all gradients of length at most $C$, including 0. Composition and conversion are as before.
3. **The noise grows with $k$.** At fixed $q$, @eq-ulvl-rate depends on $k$ and $q$ through $kq$ only. Keeping the user's RDP fixed while $k$ doubles therefore requires either halving $q$, which halves the batch, or raising $\sigma$. In the regime where the per-step RDP is about $\frac\alpha2(kq)^2/\sigma^2$, holding it fixed means $\sigma \propto kq$.
4. **The bias.** Before bounding, the expected update of DP-SGD averages over all $n$ records; after it, over $n_k$ records in which each user has weight $\min(n_u, k)/n_k$ instead of $n_u/n$. A user with $n_u > k$ loses weight, and their records beyond $k$ are not used at all.

### Interpretation

**Choosing $k$ is a bias–variance trade-off.** Amin, Kulesza, Muñoz Medina & Vassilvitskii (2019) studied exactly this choice for private empirical risk minimization, and showed that the best cap depends on the distribution of user sizes: a cap above most users' sizes buys little data at a large noise cost, and a cap below them throws away most of the data. Their framing is the same as that of choosing the clipping norm in [Beyond global sensitivity](../part03-dp-foundations/06-sensitivity.md): a smaller bound means less noise and more distortion of what is computed. Because the cap depends on the data, it should be chosen from public knowledge of the population or privately; choosing it by trying several caps and looking at test accuracy, as the example below does to show the trade-off, spends privacy that the accountant does not see ([Hyperparameter tuning](03-hyperparameter-tuning.md)).

**Which records to keep matters.** Random selection keeps the proof simple. Chua et al. (2024) compare selection rules for fine-tuning language models under contribution bounding and report that keeping each user's longest records worked best on one of their datasets. Any rule that looks only at the user's own records is allowed, by step 1.

**Bounding plus user-level DP-SGD.** The two techniques combine: cap each user at $k$ records to bound compute, then compute one clipped vector per user. With user sampling, the cap no longer costs noise, only data, and users can contribute their full records again by subsampling them inside each step.

### Example: the cap on the running example

@tbl-ulvl-cap fixes the user-level budget at $\varepsilon = 4$, $\delta = 10^{-5}$ and 1,000 steps with an expected batch of 100 records. For each cap, the sampling rate is $q = 100/n_k$, so $kq$ is the probability scale at which a full user enters a batch.

| Cap $k$ | Records kept | Sampling rate $q$ | $kq$ | Noise multiplier $\sigma$ | Test accuracy |
| --- | --- | --- | --- | --- | --- |
| 1 | 1,000 (10.0%) | 0.1 | 0.100 | 3.78 | 0.8917 |
| 2 | 1,807 (18.1%) | 0.0553 | 0.111 | 4.18 | 0.8920 |
| 5 | 3,506 (35.1%) | 0.0285 | 0.143 | 5.34 | 0.8790 |
| 10 | 5,063 (50.6%) | 0.0198 | 0.198 | 7.34 | 0.8276 |
| 20 | 6,631 (66.3%) | 0.0151 | 0.302 | 11.2 | 0.7562 |
| 50 | 8,437 (84.4%) | 0.0119 | 0.593 | 21.8 | 0.6401 |

: Contribution bounding at user-level $\varepsilon = 4$: example-level DP-SGD on at most $k$ records per user, accounted for $k$ records, one run each. {#tbl-ulvl-cap}

Raising the cap from 1 to 50 multiplies the data by 8.4 and the noise multiplier by 5.8, and accuracy falls from 0.8917 to 0.6401. On this task the noise dominates: a 1,000-record dataset, one record per user, already contains most of what the network can learn privately at this budget, and every extra record per user costs more in noise than it adds in signal. The caps 1 and 2 are tied. The cap $k = 1$ row is user-level DP-SGD on one record per user in disguise: with one record per user, sampling records at rate 0.1 is sampling users at rate 0.1, and the accountant is the same, $\sigma = 3.78$.

### Check yourself

- Why must the cap be applied with randomness independent of other users' data? *(The derivation needs adding one user to change only that user's kept records. If, say, the cap kept the $k$ records most similar to the rest of the dataset, adding a user could change which records other users keep, and neighbours would differ in more than $k$ records.)*
- In @tbl-ulvl-cap, $kq$ rises from 0.100 to 0.593 as the cap goes from 1 to 50. What does @eq-ulvl-rate predict for the noise multiplier, and why is the actual one smaller? *(Proportionality to $kq$ predicts about $3.78 \times 5.93 = 22.4$. The rule is only approximate: it holds where $\sigma$ is large and $kq$ small, and at $kq = 0.593$ neither the rate-$kq$ approximation nor the large-$\sigma$ law is accurate, so the accountant's 21.8 differs from it.)*

## Heterogeneous users {#sec-ulvl-hetero}

### Context

User sizes are rarely equal. In the running example's population, 30 users with at least 50 records each hold 30.6% of the records, while the smallest half of the users hold 10.3% (@fig-ulvl-population). Example-level training weights each user by their number of records; user-level training does not. What objective does user-level DP-SGD optimize, and what does that mean for heavy users?

### Statement

Averaged over the user sampling and the noise, the update direction of user-level DP-SGD is
$$\mathbb E\Big[\frac{G_t}{qU}\Big] = \frac1U\sum_{u=1}^U\min\Big(1, \frac{C}{\|\bar g_u\|}\Big)\bar g_u,$$ {#eq-ulvl-expected}
a sum in which each user has weight at most $1/U$ whatever their number of records. Example-level DP-SGD gives user $u$ weight $n_u/n$ in its expected update, $\frac1n\sum_u\sum_{i\in u}\min(1, C/\|g_i\|)g_i$.

### Derivation

1. **Average over the sampling.** User $u$ is in $S_t$ with probability $q$, so $\mathbb E\sum_{u\in S_t}\tilde g_u = q\sum_u\tilde g_u$. The noise has mean zero.
2. **Divide by $qU$.** The $q$ cancels, leaving $\frac1U\sum_u\tilde g_u$, which is @eq-ulvl-expected by the definition of $\tilde g_u$.
3. **The weight of one user.** $\|\tilde g_u\| \le C$ for every user, so each term has length at most $C/U$, and the share of any group of users in the update is at most their share of the users.
4. **Example-level.** The same computation for DP-SGD over records ([DP-SGD](02-dp-sgd.md)) gives each record weight $1/n$, so user $u$'s records together have weight $n_u/n$.

### Interpretation

**User-level training changes the objective, not only the noise.** With averaged per-user gradients, the model minimizes (a clipped version of) the average over users of each user's average loss, $\frac1U\sum_u\frac1{n_u}\sum_{i\in u}\ell(w, z_i)$, rather than the average over records. Whether that is what you want depends on the application. For a keyboard model, weighting people equally keeps a few very active users from defining everyone's suggestions. For a model meant to fit the overall distribution of records, it under-weights heavy users' content, and the bias can matter when heavy users differ from light ones, for example by language, topic or device.

**Clipping caps heavy users twice.** A heavy user's average gradient is a good estimate, typically short when the model already fits their data, and is rarely clipped. But if the user's data are unusual, their gradient is long and is clipped to $C$, so their influence is capped at that of any other user. This is the user-level analogue of what clipping does to records with large gradients ([Disparate impact](08-disparate-impact.md)).

**Sum-then-clip sits in between.** Summing a user's gradients before clipping makes a heavy user's vector up to $n_u$ times longer than one record's, so heavy users are clipped first, and their influence rises with $n_u$ until it reaches $C$. It interpolates between record weighting, for users whose sum stays below $C$, and user weighting, for those above it ([Exercise 5](../../book-exercises/part05-dp-ml/07-user-level.md#sec-sol-ulvl-5)).

### Example: heavy users in the running example

The 30 users with at least 50 records hold 30.6% of the records, and so 30.6% of the weight in the expected example-level update. Under user-level DP-SGD their weight is at most $30/1{,}000 = 3.0\%$. Does that hurt the model where heavy users' data are? The script splits the test set by the owner of each test point's nearest training record. Because a heavy user's records were chosen as the nearest neighbours of one seed in 50 dimensions, they spread widely, and 68.3% of test points have a heavy user's record as their nearest neighbour. At user-level $\varepsilon = 4$, user-level DP-SGD reaches 0.9361 on those points and 0.9319 on the others; the non-private model, 0.9607 and 0.9622. On this task, down-weighting heavy users costs nothing measurable, because one teacher network labels every record, so the light users' records teach the same function. On data where heavy users' records follow their own patterns, the same reweighting would cost accuracy on those patterns.

### Check yourself

- A user-level model will be used to predict for each record equally, and you know that heavy users' records differ from light users'. What can you change without changing $\varepsilon$? *(How each user's vector is computed: summing instead of averaging before clipping, or a larger $C$ relative to typical averaged gradients. Both are inside the per-user computation and leave the sensitivity at $C$.)*
- Why is each user's share of the user-level update at most $1/U$ rather than exactly $1/U$? *(Because a user's clipped vector has length at most $C$; users whose average gradient is shorter than $C$, typically those the model already fits, contribute less.)*

## Comparing the strategies on the running example {#sec-ulvl-run}

### Context

Three ways to train the running example with a user-level guarantee are now on the table: (a) example-level DP-SGD on all records, with the user accountant for the heaviest user; (b) user-level DP-SGD; (c) contribution bounding to two records per user, then example-level DP-SGD. At the same user-level $(\varepsilon, \delta)$, which model is best?

### Statement

@tbl-ulvl-run trains each strategy for 1,000 steps with an expected batch of 100 units, $C = 0.5$ and $\eta = 0.5$, at the noise multipliers of @tbl-ulvl-sigma, one run each.

| User-level $\varepsilon$ | (a) Example-level, all records | (b) User-level DP-SGD | (c) At most 2 records per user |
| --- | --- | --- | --- |
| 1 | 0.5318 | 0.8065 | 0.7084 |
| 2 | 0.5582 | 0.9028 | 0.8374 |
| 4 | 0.5713 | 0.9347 | 0.8810 |
| 8 | 0.5478 | 0.9442 | 0.9333 |
| Non-private, 1,000 steps | 0.9612 | | |

: Test accuracy at the same user-level budget, $\delta = 10^{-5}$, one run each. {#tbl-ulvl-run}

### Derivation

Each strategy's guarantee was derived above: (a) and (c) by the accountant of @sec-ulvl-group-sgd with $k = 348$ and $k = 2$, and (b) by @sec-ulvl-dpsgd. The cap $k = 2$ is the best cap at $\varepsilon = 4$ in @tbl-ulvl-cap, tied with $k = 1$; choosing it from test accuracy is a leak that a real deployment would have to pay for. Run-to-run variation is of the order of a point at these budgets: the cap $k = 1$ and user-level DP-SGD restricted to one random record per user are the same algorithm with the same $\sigma$, and two runs of it gave 0.8917 and 0.9070.

### Interpretation

**Keeping the record as the unit fails.** Strategy (a) is at chance, between 0.53 and 0.57, at every budget up to $\varepsilon = 8$. The heaviest user dictates the noise for everyone: $\sigma$ between 70.3 and 449 drowns every gradient. That is the practical meaning of @fig-ulvl-user-eps: an example-level pipeline on data with heavy users needs either an enormous $\varepsilon$ or a cap.

**Changing the unit wins.** User-level DP-SGD is the best strategy at every budget, 0.9347 at $\varepsilon = 4$ against 0.9612 for the non-private model. It needs a little less noise than the cap of 2 (@tbl-ulvl-sigma) and, unlike the cap, it uses every record, through the per-user average. Its advantage is largest at small budgets: 9.8 points at $\varepsilon = 1$, 1.1 points at $\varepsilon = 8$. Its cost is compute: each step processes all the records of about 100 users, about 1,000 records, ten times what the example-level strategies process.

**Bounding is the fallback.** Strategy (c) keeps the example-level pipeline and loses 5 to 10 points against (b) below $\varepsilon = 8$. It comes close at $\varepsilon = 8$, where the noise is small enough that the discarded data, not the noise, limits accuracy.

These findings agree with published comparisons on language models, which use different data, models and accountants. Chua et al. (2024) compare contribution bounding with group-privacy accounting ("Group Privacy") against user-level DP-SGD with per-user averaging ("User-wise DP-SGD") for fine-tuning, and report that user-wise DP-SGD generally does better, especially at small budgets. Charles et al. (2024), who call the two sampling schemes example-level and user-level sampling, report that user-level sampling usually does better when each user's data are varied and when strong privacy is required, and that example-level sampling sometimes wins. Note that Chua et al. account for the cap with the generic group privacy of @eq-ulvl-group; the direct accountant of @sec-ulvl-group-sgd would give them less noise.

![Test accuracy of the running example against the user-level $\varepsilon$ at $\delta = 10^{-5}$, for the three strategies (1,000 steps, expected batch 100, one run each; the runs differ from @tbl-ulvl-run in their random seeds). User-level DP-SGD is best at every budget; example-level DP-SGD on all records, accounted for its heaviest user, stays near chance.](figures/ulvl_strategies.svg){#fig-ulvl-strategies}

### Example: what the user-level guarantee costs

At $\varepsilon = 4$ the running example loses 2.7 points to protect users (0.9347 against 0.9612). The record-level model of [DP-SGD](02-dp-sgd.md) at the same $\varepsilon$, trained for 10,000 steps, reached 0.9548, and this chapter's 1,000-step schedule would need only $\sigma = 0.778$ for a record-level guarantee (@tbl-ulvl-sigma). The difference is the price of the unit: 1,000 people instead of 10,000 records, and a guarantee that now covers the person with 348 records as well as the one with a single record.

### Check yourself

- Strategy (a) uses every record and the tightest available accountant. Why does it do worst? *(Because the guarantee must hold for the heaviest user, whose 348 records put about 3.5 of them into every batch; the noise has to hide a shift of several clipping norms in almost every step, which takes $\sigma$ in the hundreds.)*
- Would strategy (a) look better if the heaviest user were removed from the data? *(Only a little: the accountant would be for the next-largest user, who has 258 records, so about 2.6 of them would still be in every batch. Removing a user because of their size is itself a data-dependent choice that must be made with public information or paid for, and it is a crude form of contribution bounding.)*

## Code: check every number by computation

The script [code/user_level.py](../../code/part05-dp-ml/user_level.py) builds the user population, implements the user accountant of @eq-ulvl-mixture by numerical integration and checks it against @eq-ulvl-pair and @eq-ulvl-binomial, compares it with group privacy, and trains the three strategies with the part's network. It imports the model, DP-SGD and the record-level accountant from [code/dp_sgd.py](../../code/part05-dp-ml/dp_sgd.py). From the repository root:

```bash
uv run python code/part05-dp-ml/user_level.py
```

It takes about a minute and a half, most of it spent in user-level DP-SGD, which processes about 1,000 records per step, and in finding the noise multiplier for the user with 348 records.

**Output** (seed 0), summarized in @tbl-ulvl-code:

| Experiment | Result |
| --- | --- |
| Population | 1,000 users; mean 10, median 4, largest 348 records; 193 users with one record |
| Accountant checks | $k = 1$ matches the subsampled Gaussian; integral matches @eq-ulvl-pair to seven digits |
| Running example, user with $k$ records | $\varepsilon = 15.54$ at $k = 2$, 146.20 at $k = 10$; group privacy 18.61 and none |
| Noise at user-level $\varepsilon = 4$ | $\sigma = 128$ (all records), 3.78 (user-level), 4.18 (cap 2) |
| Contribution bounding at $\varepsilon = 4$ | 0.8917 at $k = 1$ down to 0.6401 at $k = 50$ |
| Strategies at $\varepsilon = 4$ | 0.5713 (all records), 0.9347 (user-level), 0.8810 (cap 2); non-private 0.9612 |
| Heavy users | 30.6% of records, at most 3.0% of the user-level update |

: The script's output with the default seed. {#tbl-ulvl-code}

[tests/test_user_level.py](../../tests/test_user_level.py) checks that the output still matches.

**What to take from it.** The record-level accountant answers the wrong question when people contribute many records, and asking it the right question, about a user with $k$ records, gives answers that grow faster than $k$. The fix is in the algorithm, not the accountant: make the person the unit that is sampled and clipped. Implementation needs only a per-user aggregation before clipping; the accountant is unchanged.

## References

- McMahan, Ramage, Talwar & Zhang, "Learning Differentially Private Recurrent Language Models" (ICLR 2018). User-level DP-SGD in federated learning: sample users, clip each user's update, add Gaussian noise, account with users as the unit.
- Charles, Ganesh, McKenna, McMahan, Mitchell, Pillutla & Rush, "Fine-Tuning Large Language Models with User-Level Differential Privacy" (arXiv 2407.07737, 2024). The tight accountant for example-level sampling (their Theorem 3, the mixture of @eq-ulvl-mixture), and an empirical comparison of example-level and user-level sampling for language models.
- Chua, Ghazi, Huang, Kamath, Kumar, Liu, Manurangsi, Sinha & Zhang, "Mind the Privacy Unit! User-Level Differential Privacy for Language Model Fine-Tuning" (Conference on Language Modeling 2024; arXiv 2406.14322). Contribution bounding with group privacy against user-wise DP-SGD, with record-selection rules and the choice of the cap.
- Amin, Kulesza, Muñoz Medina & Vassilvitskii, "Bounding User Contributions: A Bias-Variance Trade-off in Differential Privacy" (ICML 2019). The cap on records per user as a bias–variance trade-off, and how the best cap depends on the distribution of user sizes.
- Levy, Sun, Amin, Kale, Kulesza, Mohri & Suresh, "Learning with User-Level Privacy" (NeurIPS 2021). Theory of user-level private learning when each user holds several samples.
- Mironov, "Rényi Differential Privacy" (CSF 2017). Group privacy for RDP, Fact 1.
- Dwork & Roth, *The Algorithmic Foundations of Differential Privacy* (Foundations and Trends in Theoretical Computer Science, 2014). Group privacy for pure and approximate DP.

## Cheat sheet

$U$ users, user $u$ with $n_u$ records, cap $k$, sampling rate $q$, noise multiplier $\sigma$, clipping norm $C$, $T$ steps. @tbl-ulvl-cheat collects the results.

| Result | Formula | Conditions |
| --- | --- | --- |
| Group privacy | $(k\varepsilon, \delta\frac{e^{k\varepsilon} - 1}{e^\varepsilon - 1})$ | Any example-level $(\varepsilon, \delta)$ mechanism |
| User with $k$ records, example-level DP-SGD | $\mathcal N(0, \sigma^2)$ against $\sum_m\binom km q^m(1 - q)^{k-m}\mathcal N(m, \sigma^2)$ | Tight (Charles et al. 2024, Theorem 3) |
| Its moments | $A_\alpha = \mathbb E[\exp(\sigma^{-2}\sum_{i<j}M_iM_j)]$, $M_i \sim \mathrm{Binomial}(k, q)$ | Integer $\alpha$ |
| Quick estimate | Record-level accountant at rate $kq$ | Small $kq$; slightly optimistic |
| User-level DP-SGD | DP-SGD's accountant with users as units, $q$ = user sampling rate | One clipped vector per user, from that user's data alone |
| Capped sum per user | Noise multiplier $\sigma/k$ | Per-record clipping, at most $k$ records, users sampled whole |
| Contribution bounding | Accountant for $k$ records on $n_k = \sum_u\min(n_u, k)$ records | Cap chosen per user, independently of others |
| Expected user-level update | $\frac1U\sum_u\min(1, C/\|\bar g_u\|)\bar g_u$ | Each user weighs at most $1/U$ |
| Running example | Records: $\varepsilon = 6.71$; user with 2 records 15.54, with 10 records 146.20 | $q = 0.01$, $\sigma = 1$, $T = 10{,}000$ |

: Every result of the chapter. {#tbl-ulvl-cheat}

**Rules of thumb**

- Name the privacy unit with every $\varepsilon$. A record-level $\varepsilon$ says little about a person with many records.
- To estimate what an example-level model gives a user with $k$ records, rerun the accountant at rate $kq$, and expect the truth to be a little worse.
- Don't use group privacy for DP-SGD; account for the user's records directly, or better, sample and clip users.
- Prefer user-level DP-SGD with per-user averaging. It pays for the number of people, not their records, and uses all their data.
- If the pipeline must stay record-based, cap records per user, choose the cap from public knowledge, and account for the cap with the user accountant.
- Decide deliberately how heavy users should count: averaging weights people equally, summing before clipping weights records until the clip.

**Practice.** Exercises for this chapter, each with a worked solution, are in [the exercise book](../../book-exercises/part05-dp-ml/07-user-level.md).
