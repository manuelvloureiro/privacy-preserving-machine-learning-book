# Amplification by subsampling

## Why this matters

Most private models are trained with stochastic gradient descent (SGD) made to satisfy differential privacy (DP), a method called DP-SGD. Each step clips every example's gradient to a fixed length, sums the clipped gradients and adds Gaussian noise to the sum. The standard deviation of that noise is the clipping length times a number called the **noise multiplier**, written $\sigma$.

A training run takes ten thousand steps or more, and every step reads the data again. As always, the guarantee is a privacy parameter $\varepsilon$ together with a small slack $\delta$ ([the definition of DP](../part03-dp-foundations/01-definition.md)). Suppose every record paid the full cost of a plain Gaussian mechanism at every step. With $\sigma = 1$ and a slack of $10^{-5}$, ten thousand steps would then cost $\varepsilon = 5425.5$, as [Rényi DP](01-renyi-dp.md) computes, and nobody would report such a number.

DP-SGD escapes this cost because each step looks only at a small random batch of the data. Write $q$ for the **sampling rate**, the probability that a given record lands in the batch. An attacker who sees the output of a step must then first guess whether the record was used at all, and that uncertainty is worth a lot. At a sampling rate of 0.01, for example, a step that costs $\varepsilon = 1$ on all the data costs only about 0.017. This effect is called **amplification by subsampling**.

One idea explains every result in this chapter. With probability $q$ the step used the record, and otherwise it ran exactly as it would have without it. So the output with the record is a mixture of the two cases, and mixing in the unchanged run pulls every likelihood ratio towards 1. A likelihood ratio here is the factor by which an output is more likely with the record than without it. The mixture comes from how the batch was drawn, so a different sampling scheme gives a different mixture. A privacy accountant, the code that adds up the cost of a training run, must therefore assume the sampler that the training code actually runs. Otherwise it reports a number that does not apply.

## Amplification for (ε, δ)-DP under Poisson sampling {#sec-sub-eps-delta}

The simplest way to draw a batch is **Poisson sampling**. Each record enters the batch by its own coin flip, which comes up heads with probability $q$, independently of every other record. To measure what this buys, use the add/remove neighbouring relation of [the definition of DP](../part03-dp-foundations/01-definition.md). Under it, two datasets are neighbours when one of them has a record that the other lacks. Write $D$ for the dataset without the target record $x$, and $D'$ for the dataset with it.

Fix which of the other records were drawn, and compare the two runs for that draw. On $D$, the step then has some output distribution, written $P_0$. On $D'$, it outputs from the same $P_0$ when the target was not drawn, and from another distribution $P_1$ when it was. Since the target is drawn with probability $q$, the output on $D'$ is the mixture
$$(1 - q)P_0 + qP_1.$$
To compare the runs at one output, divide the mixture by $P_0$. The likelihood ratio of the two runs is then
$$\frac{(1 - q)P_0 + qP_1}{P_0} = (1 - q) + q\,\frac{P_1}{P_0},$$
which is a weighted average of 1 and $P_1/P_0$, the ratio the step would have without sampling.

That average gives the result. Suppose the step on its own is $(\varepsilon, \delta)$-DP under add/remove. Its likelihood ratio is then at most $e^\varepsilon$, apart from an excess of probability of at most $\delta$. So the averaged ratio is at most $1 + q(e^\varepsilon - 1)$. The slack, too, matters only in the runs where the record was drawn, which happen with probability $q$. Li, Qardaji & Su (2012) first proved the tight result, and Balle, Barthe & Gaboardi (2018) gave it in this form. A step run on a Poisson sample with rate $q$ is $(\varepsilon', \delta')$-DP under add/remove, with
$$\varepsilon' = \ln\big(1 + q(e^\varepsilon - 1)\big), \qquad \delta' = q\delta.$$ {#eq-sub-amplify}
The full edition shows that the bound also holds with the run without the record on top. And it holds for the whole step, because it holds for every draw of the other records.

The argument has two conditions. The first is that each record's coin is independent of the others. Only then is the target drawn with probability exactly $q$ whatever the rest of the batch looks like. If its chance of being drawn depended on the other records, the weights of the mixture would change from batch to batch, and the formula would not apply. The second condition is that the batch stays secret, so that the step publishes only its output and not the list of records drawn. If the batch were published, the attacker would know whether the record was used, and the uncertainty that pays for the amplification would be gone.

The formula cannot be improved, and [randomized response](../part03-dp-foundations/01-definition.md#sec-dpdef-rr) shows why. Mark the target as the only "flagged" record in the data, and sample at rate 0.1. Let the step apply randomized response with $\varepsilon = 1$ to one bit, "the batch contains a flagged record", which is 1 exactly when the target was drawn. When the target is not drawn, the true bit is 0, and the step reports 1 with probability $1/(1 + e) = 0.2689$. When it is drawn, the true bit is 1, and the step reports 1 with probability 0.7311. @tbl-sub-rr shows both outputs and the likelihood ratios in both directions.

| Output | Record not drawn, $P_0$ | Record drawn, $P_1$ | Mixture $0.9P_0 + 0.1P_1$ | Mixture $/\ P_0$ | $P_0\ /$ mixture |
| --- | --- | --- | --- | --- | --- |
| 0 | 0.7311 | 0.2689 | 0.6848 | 0.9368 | 1.0675 |
| 1 | 0.2689 | 0.7311 | 0.3152 | 1.1718 | 0.8534 |

: Randomized response with $\varepsilon = 1$ on a 10% Poisson sample. Without sampling the likelihood ratios are $e$ and $1/e$; sampling pulls them to at most 1.1718. {#tbl-sub-rr}

Without sampling the largest likelihood ratio is $e$, about 2.718. With sampling it is 1.1718, which is exactly $1 + 0.1(e - 1)$. This mechanism therefore reaches the bound, so no smaller $\varepsilon'$ could hold for every mechanism. Its amplified cost is $\ln 1.1718 = 0.1586$, as @eq-sub-amplify says, which is more than the 0.1 you might guess by multiplying $\varepsilon$ by $q$. The other direction, with the run without the record on top, has a smaller largest ratio, 1.0675. It stays small because the mixture always keeps at least 0.9 of the probability that $P_0$ puts on each output.

The formula behaves differently for small and for large $\varepsilon$. For small $\varepsilon$, $e^\varepsilon - 1$ is close to $\varepsilon$, and the logarithm of a number close to 1 is close to that number minus 1. So the amplified cost is about $q\varepsilon$, and a 1% sample makes each step a hundred times cheaper. For large $\varepsilon$, the 1 inside the logarithm no longer matters. The amplified cost is then about $\ln(qe^\varepsilon)$, which is
$$\varepsilon' \approx \varepsilon - \ln(1/q).$$
In that regime sampling only subtracts a fixed amount.

Sampling helps so little at large $\varepsilon$ because of the worst case. With probability $q$ the record was drawn, and when it was, the step reveals as much as it would without sampling. Sampling makes that event rare, but it cannot make it less revealing. @tbl-sub-amp shows both regimes, and its row for a sampling rate of 0.01 makes the contrast plain. There, a step with $\varepsilon = 0.1$ costs 0.0011, close to $q\varepsilon$. But a step with $\varepsilon = 5$ still costs 0.9059, far more than a hundredth of its cost. And a step with $\varepsilon = 10$ costs 5.3993, which is close to the large-$\varepsilon$ approximation. @fig-sub-amp shows that the switch between the regimes happens where $qe^\varepsilon$ reaches 1, that is, where the cost $\varepsilon$ without sampling is near $\ln(1/q)$.

| $q$ | $\varepsilon = 0.1$ | $\varepsilon = 1$ | $\varepsilon = 5$ | $\varepsilon = 10$ |
| --- | --- | --- | --- | --- |
| 0.01 | 0.0011 | 0.0170 | 0.9059 | 5.3993 |
| 0.1 | 0.0105 | 0.1586 | 2.7563 | 7.6978 |
| 0.5 | 0.0512 | 0.6201 | 4.3136 | 9.3069 |

: The amplified $\varepsilon' = \ln(1 + q(e^\varepsilon - 1))$ for three sampling rates. {#tbl-sub-amp}

![The subsampled $\varepsilon'$ against the base $\varepsilon$ for three sampling rates, with the small-$\varepsilon$ approximation $q\varepsilon$ dotted and no sampling dashed. Amplification is a factor $q$ for small $\varepsilon$ and only a shift by $\ln(1/q)$ for large $\varepsilon$.](figures/subsampling_amplification.svg){#fig-sub-amp}

The slack always shrinks by the factor $q$, because the step's failure matters only when the record was drawn. Finally, @eq-sub-amplify is a statement about one step, and it is tight for one step only. Composing the per-step pairs over thousands of steps is far from tight, and @sec-sub-dpsgd shows by how much.

## The sampling scheme must match the neighbouring relation {#sec-sub-schemes}

Poisson sampling gives batches of random size, but most training code draws batches of a fixed size $m$ from the $n$ records. Some code draws a fresh random batch of $m$ records at each step. Other code shuffles the data once per epoch and cuts the shuffled list into consecutive batches, which is called **shuffle and partition**. Does @eq-sub-amplify still apply to these samplers? For fresh random batches it does under replace-one neighbours, and under add/remove it holds for DP-SGD only at half the noise multiplier. For shuffle and partition it does not apply at all.

### Fixed-size batches

A fixed batch size pairs naturally with **replace-one** neighbours, two datasets of the same size that differ in the value of one record. (The [definition of DP](../part03-dp-foundations/01-definition.md) calls this relation replace.) A uniformly random batch of $m$ out of $n$ records contains any given record with probability $m/n$, and that fraction plays the role of the sampling rate. Balle, Barthe & Gaboardi (2018) show that the formula of @eq-sub-amplify then holds with $q = m/n$. The condition is different, though. The step must be $(\varepsilon, \delta)$-DP under replace-one, and the result is a guarantee under replace-one.

A fixed-size batch has no coin of its own for each record, but you can draw it in a way that has one. First flip a coin that comes up heads with probability $m/n$. On heads, the batch is the target plus $m - 1$ other records chosen at random. On tails, it is $m$ other records chosen at random. Both datasets share the coin and the other records, so the mixture dilutes their difference as Poisson sampling does. Here, though, both outputs are mixtures, and that needs one more condition. The batch with the target's old value must be close to the batch with its new value. It must also be close to the batch drawn on tails, which lacks the target. All three batches have size $m$ and differ pairwise in one record, so the step's replace-one guarantee covers both comparisons.

The catch is in what replace-one costs the step itself. DP-SGD sums gradients clipped to a length $C$, so adding or removing a record moves the sum by at most $C$. Replacing a record can move it by up to $2C$, because the old gradient leaves and a new one, possibly pointing the opposite way, comes in. A Gaussian mechanism's privacy depends only on the shift divided by the noise's standard deviation. So twice the shift with the same noise has the same privacy as the original shift with half the noise multiplier.

Under add/remove, fixed-size batches pay the same factor of two. A record that enters a fixed-size batch pushes another record out, so the sum can again move by $2C$. A Poisson accountant compares the Poisson mixture with the run without the record. Lebeda, Regehr, Kamath & Steinke (2024) prove that this same comparison covers the fixed-size case. The accountant must only be given rate $m/n$ and noise multiplier $\sigma/2$. Their proof treats the add direction and the remove direction separately. So for fixed-size batches under add/remove, run a Poisson accountant with half the noise multiplier you actually used.

Under replace-one there is no such shortcut. Each step's guarantee is @eq-sub-amplify with $q = m/n$, applied to the step's replace-one guarantee, which uses noise multiplier $\sigma/2$. No Poisson accountant gives the composed cost under replace-one, because there may be several candidate worst-case pairs (@sec-sub-generic).

Halving the noise multiplier is not a small change, as the running example of this part shows. It uses a sampling rate of 0.01 and a noise multiplier of 1, and runs for ten thousand steps. With Poisson sampling, the accountant of @sec-sub-dpsgd gives $\varepsilon = 6.71$ at $\delta = 10^{-5}$. With fixed-size batches under add/remove, the same accountant must be given noise multiplier 0.5, and it then gives $\varepsilon = 47.42$, about seven times more.

### Poisson sampling with replace-one neighbours

Poisson sampling can also be paired with replace-one neighbours, and then both outputs are mixtures. Given the other records, the target is drawn with probability $q$ in both runs. When it is drawn, one run uses its old value and the other its new value. Those two runs differ by a replacement, which is a removal followed by an addition. By group privacy ([Properties of DP](../part03-dp-foundations/02-properties.md#sec-dpprop-group)), an add/remove guarantee also covers two such changes, but at a price. The $\varepsilon$ doubles, and the slack grows to $(1 + e^\varepsilon)\delta$. Using this weaker guarantee for the base step gives
$$\varepsilon' = \ln\big(1 + q(e^{2\varepsilon} - 1)\big), \qquad \delta' = q(1 + e^\varepsilon)\delta.$$
For small $q$ this is about $e^\varepsilon + 1$ times the add/remove cost, because multiplying $e^\varepsilon - 1$ by that factor gives $e^{2\varepsilon} - 1$.

This bound is not tight, because it also charges the weaker guarantee to pairs that differ by a single addition. The run that used the target and the run that did not are such a pair. For pure DP, with $\delta = 0$, the full edition computes the exact cost, which is about twice the add/remove cost when $\varepsilon$ is small. At $\varepsilon = 1$ and a sampling rate of 0.01, for example, the exact replace-one cost is 0.02338. That is 1.37 times the add/remove cost of 0.01704, since the factor of two holds only for small $\varepsilon$. The group-privacy bound gives 0.06193, and so overstates that ratio as 3.64.

### Shuffle and partition

Shuffle and partition is worse, because the source of amplification is missing altogether. In one epoch the record is in exactly one batch, and never absent. So no step's output is a mixture with "not used". The only uncertainty is which batch held the record, and the other records can give that away.

Chua et al. (2024) build a case where they do. Every other record has gradient $-1$ and the target has gradient $+1$. The neighbouring dataset is the same with the target's gradient set to 0, which is the **zero-out** neighbouring relation. A Poisson step's clipped sum moves by at most $C$ under zero-out, as under add/remove, so the Poisson accountant's number applies to this relation too. In this dataset, the batch that holds the target stands out from all the others, because the target's gradient pushes its sum the other way. A test that takes the largest of the batches' noisy sums and compares it with a threshold then tells the two datasets apart. Every test's error rates are limited by $\varepsilon$, as [the definition of DP](../part03-dp-foundations/01-definition.md#sec-dpdef-testing) explains. So how well this one test works gives a lower bound on the true $\varepsilon$ of shuffled training.

Take the running example's sampling rate of 0.01, so that one epoch has 100 batches. @tbl-sub-shuffle compares three samplers over one epoch, at $\delta = 10^{-5}$. With deterministic batches, the record is used once in a known batch, so the epoch is a plain Gaussian mechanism and its $\varepsilon$ is exact. For shuffled batches the table gives the lower bound from the test, and for Poisson sampling the upper bound from the accountant of @sec-sub-dpsgd.

| $\sigma$ | Deterministic batches (exact) | Shuffled batches (lower bound) | Poisson sampling (upper bound) |
| --- | --- | --- | --- |
| 0.5 | 10.00 | 10.00 | 8.03 |
| 1 | 4.38 | 4.06 | 1.21 |
| 1.5 | 2.75 | 1.46 | 0.47 |
| 2 | 1.99 | 0.48 | 0.26 |
| 2.5 | 1.55 | 0.20 | 0.17 |
| 3 | 1.27 | 0.10 | 0.14 |

: The $\varepsilon$ at $\delta = 10^{-5}$ of one epoch of 100 batches under three batch samplers. Shuffling is a lower bound and Poisson an upper bound, so wherever the shuffled column exceeds the Poisson column, the Poisson number is provably wrong for shuffled training. {#tbl-sub-shuffle}

At the running example's noise multiplier, shuffled training has $\varepsilon$ of at least 4.06 after one epoch, while the Poisson accountant reports at most 1.21. Shuffling also buys almost nothing over a fixed batch order, which has $\varepsilon = 4.38$. The gap closes as $\sigma$ grows (@fig-sub-schemes). At $\sigma = 3$ the shuffled lower bound even falls below the Poisson upper bound, but that says only that this particular test is too weak there to show a gap.

![The $\varepsilon$ at $\delta = 10^{-5}$ of one epoch of 100 batches as a function of the noise multiplier: deterministic batches (exact), shuffled fixed-size batches (a lower bound from the largest-batch-output test of Chua et al. 2024), and Poisson sampling with $q = 0.01$ (an upper bound from a Rényi accountant). For $\sigma$ up to about 2.5, shuffled training is provably less private than the Poisson accountant claims.](figures/subsampling_sampling_schemes.svg){#fig-sub-schemes}

@tbl-sub-matching collects what to account for under each pairing, for DP-SGD's sum of gradients clipped to length $C$ with noise of standard deviation $\sigma C$.

| Sampling | Neighbours | What to account for |
| --- | --- | --- |
| Poisson, rate $q$ | add/remove | The Poisson mixture against the run without the record, with noise multiplier $\sigma$ |
| Fixed size $m$ of $n$ | replace-one | Per step, @eq-sub-amplify with $q = m/n$; the step's replace-one guarantee uses noise multiplier $\sigma/2$ (no Poisson-accountant shortcut) |
| Fixed size $m$ of $n$ | add/remove | The Poisson pair with noise multiplier $\sigma/2$, each direction separately (Lebeda et al. 2024) |
| Poisson, rate $q$ | replace-one | Mixtures on both sides, with used parts that differ by a replacement; about twice the add/remove cost for pure DP at small $\varepsilon$ |
| Shuffle and partition | zero-out | No simple formula; the record is in exactly one batch per epoch, never absent |

: What to account for under each combination of sampling scheme and neighbouring relation. {#tbl-sub-matching}

Each row of the table is correct on its own. The bug is to train with one row and report the number from another. This is the largest single risk in DP-SGD accounting, larger than the choice of accountant, so read the data loader before you trust a reported $\varepsilon$. If it draws fixed-size batches and you report under add/remove, give a Poisson accountant half the noise multiplier you used. If it shuffles and partitions, the Poisson number does not apply at all. Then either switch to Poisson sampling, or say plainly that the reported $\varepsilon$ assumes a sampler you did not run. In every case, state the neighbouring relation next to $\varepsilon$.

## RDP of the Poisson-subsampled Gaussian {#sec-sub-rdp-gaussian}

Per-step $(\varepsilon, \delta)$ pairs compose badly, as @sec-sub-dpsgd will show, so accountants for DP-SGD use Rényi differential privacy (RDP) instead. [Rényi DP](01-renyi-dp.md) describes a mechanism by a curve. At each **order** $\alpha$, a number above 1, the **RDP curve** $\varepsilon(\alpha)$ bounds the Rényi divergence between the outputs on any two neighbours. That divergence is an average of the likelihood ratio raised to the power $\alpha$, so larger orders weigh large ratios more. Under composition the curves of the steps simply add, and you convert the total to $(\varepsilon, \delta)$ once. What remains is the RDP curve of one subsampled Gaussian step.

Mironov, Talwar & Zhang (2019) reduce that question to one dimension. Divide the clipped gradient sum by the clipping length, so that one record moves it by at most 1. The other records shift both outputs by the same amount, and a common shift changes no likelihood ratio. The noise is independent and equally large in every direction. So every coordinate along which the record does not move the sum has the same distribution in both runs, and adds nothing. And a move shorter than 1 is the same as a move of exactly 1 with more noise, and more noise can only make the runs harder to tell apart. What is left is one coordinate.

In that coordinate, the step without the record outputs $P_0$, a normal distribution with mean 0 and standard deviation $\sigma$. The step with the record draws from $P_0$ with probability $1 - q$. Otherwise it draws from $P_1$, a normal distribution with mean 1 and the same standard deviation. So its output is the mixture
$$P_q = (1 - q)P_0 + qP_1.$$

The divergence can be taken in two directions, with the mixture first or second. Mironov, Talwar & Zhang prove that the direction with the mixture first is always the larger. It puts the dataset with the record first, and [Numerical accounting](06-numerical-accounting.md) calls it the remove direction. So under Poisson sampling and add/remove neighbours, the curve of one step is the Rényi divergence of order $\alpha$ of $P_q$ from $P_0$. This dominance holds for Rényi divergences only. For the $\delta$ at a given $\varepsilon$, neither direction dominates in general, and [Numerical accounting](06-numerical-accounting.md) computes both.

That divergence is the average under $P_0$ of the mixture's likelihood ratio raised to the power $\alpha$. Write $r$ for the ratio $P_1/P_0$ of the two normal densities, so that the mixture's ratio is $(1 - q) + qr$, as in @sec-sub-eps-delta. For an integer order, the binomial theorem expands the power of this ratio into a finite sum. Each term of the sum then needs an average of a power of $r$, and for two normal distributions that average has a closed form. With sensitivity 1 and noise multiplier $\sigma$, the result is
$$\varepsilon(\alpha) = \frac{\ln A_\alpha}{\alpha - 1}, \qquad A_\alpha = \sum_{j=0}^\alpha\binom\alpha j(1 - q)^{\alpha - j}q^j\exp\Big(\frac{j^2 - j}{2\sigma^2}\Big).$$ {#eq-sub-binomial}

The terms are indexed by a whole number $j$ from 0 to $\alpha$. One way to read term $j$ is as the case in which the record's contribution is present in $j$ of $\alpha$ imagined copies of the run. It is weighted by the binomial chance of that case, and its exponential is the privacy cost of that many copies.

A worked case shows how the sum behaves. Take order 2 and the running example's sampling rate and noise multiplier, so that the sum has the three terms of @tbl-sub-alpha2.

| $j$ | $\binom 2j(1-q)^{2-j}q^j$ | $e^{(j^2 - j)/2}$ | Term |
| --- | --- | --- | --- |
| 0 | $(0.99)^2 = 0.9801$ | 1 | 0.980100 |
| 1 | $2 \times 0.99 \times 0.01 = 0.0198$ | 1 | 0.019800 |
| 2 | $(0.01)^2 = 0.0001$ | $e = 2.718$ | 0.000272 |

: The binomial sum for $A_2$ with $q = 0.01$, $\sigma = 1$. Without the exponential factors the terms would sum to exactly 1. {#tbl-sub-alpha2}

Without the exponential factors, the three weights would add up to exactly 1, and the divergence would be 0. The exponential factor is 1 for the first two terms, so only the last term carries any privacy cost. The terms sum to $A_2 = 1.00017183$, which gives $\varepsilon(2) = 1.7181 \times 10^{-4}$. The same step without sampling has the Gaussian curve $\alpha/(2\sigma^2)$, which is 1 at this order.

For fractional orders the binomial sum does not end, so accountants compute the same average as a numerical integral, which the full edition checks against the exact sum.

The curve has a low regime and a steep one, and @fig-sub-mixture shows why. Its right panel plots the **privacy loss**, the logarithm of the likelihood ratio, at each output. Where the record was unlikely to have been used, the loss is flat and close to $\ln(1 - q)$. In the far right tail, the loss runs parallel to the loss without sampling, lower by $\ln(1/q)$.

![Left: one subsampled Gaussian step with $q = 0.1$. The output with the record is the mixture (dashed), barely distinguishable from $P_0$. Right: the privacy loss $\ln\big((1 - q) + q\,P_1/P_0\big)$. Where the record is unlikely to have been used it is flat near $\ln(1 - q)$; in the far right tail it runs parallel to the unsubsampled loss, $\ln(1/q)$ lower.](figures/subsampling_mixture.svg){#fig-sub-mixture}

Low orders average the loss over the bulk of the outputs, where sampling flattens it. High orders look at the far tail, where sampling saves only $\ln(1/q)$, as it does for $(\varepsilon, \delta)$ at large $\varepsilon$. For the running example (@fig-sub-rdp), the curve stays tiny up to order 8, where it is $8.9364 \times 10^{-4}$ against 4 without sampling. Beyond that order it climbs steeply. By order 16 it already reaches 3.0879, against 8 without sampling.

## Small q and the moments accountant {#sec-sub-small-q}

Abadi et al. (2016) introduced DP-SGD together with an accountant they called the **moments accountant**. It tracks the logarithm of the average of $e^{\lambda L}$, where $L$ is the privacy loss and $\lambda$ is a positive number. This quantity is the RDP curve in disguise, as [Rényi DP](01-renyi-dp.md) shows, because at $\lambda$ it equals $\lambda$ times the RDP curve at order $\lambda + 1$. Its conversion to $(\varepsilon, \delta)$ is the simple conversion of [Rényi DP](01-renyi-dp.md), which this chapter calls Mironov's conversion (@eq-sub-dpsgd). Abadi et al. did not have the exact sum of @eq-sub-binomial. They proved a bound for small sampling rates instead.

The worked case at order 2 shows where that bound comes from. As there, the weights of @eq-sub-binomial alone add up to 1, and the first two terms have exponential factor 1. So everything above 1 comes from the terms from $j = 2$ on. Every term after $j = 2$ carries at least $q^3$, so for a small sampling rate the $j = 2$ term dominates. Keeping only that term, and using that the logarithm of a number close to 1 is close to that number minus 1, gives the leading term of the curve,
$$\varepsilon(\alpha) \approx \frac{\alpha q^2\big(e^{1/\sigma^2} - 1\big)}{2} \approx \frac{\alpha q^2}{2\sigma^2}\ \text{for large } \sigma.$$ {#eq-sub-small-q}
The second form holds because $e^{1/\sigma^2} - 1$ is close to $1/\sigma^2$ when $\sigma$ is large.

The cost per step is quadratic in $q$, while the $(\varepsilon, \delta)$ amplification of @sec-sub-eps-delta was linear. That gives a design rule. A training run of $E$ epochs takes $E/q$ steps, so its total RDP is about
$$\frac{E}{q}\cdot\frac{\alpha q^2}{2\sigma^2} = \frac{E\,q\,\alpha}{2\sigma^2}.$$
At a fixed number of epochs, smaller batches are therefore cheaper. This holds as long as the orders the accountant uses stay in the range where the leading term is accurate.

Abadi et al.'s bound has the same shape with about twice the constant, roughly $q^2\alpha/\sigma^2$ for large $\sigma$. It is the source of the approximation you will often see quoted. Their bound also has conditions. It needs $\sigma$ of at least 1 and a sampling rate below $1/(16\sigma)$. It covers only orders up to about $1 + \sigma^2\ln(1/(q\sigma))$, which is 5.6 for the running example. Inside that range it lies above the exact curve, as an upper bound should, but outside it the bound can fall below the exact value.

Their headline example keeps the running example's sampling rate and ten thousand steps, but uses $\sigma = 4$. At $\delta = 10^{-5}$ they report $\varepsilon \approx 1.26$. The exact curve with Mironov's conversion gives 1.2586, at order 20, which matches their number to the digits they report. The improved conversion of [From RDP to (ε, δ)](02-rdp-conversions.md) does better and gives 1.0355, at order 17.

The leading term fails at high orders, because of the last term of the sum. That term is the case in which the record is present in all $\alpha$ imagined copies, and its weight $q^\alpha$ is tiny. But its exponential grows like $e^{\alpha^2/(2\sigma^2)}$, so at some order it overtakes the $j = 2$ term. Comparing the two terms shows that this happens near
$$\alpha \approx 2\sigma^2\ln(1/q).$$
For the running example that order is about 9.2. A larger noise multiplier pushes it out, to near 147.4 at $\sigma = 4$. Below about half that order, the leading term is within 8% of the exact curve, but near the breakdown the gap opens fast (@fig-sub-rdp). At order 8 the exact curve of the running example is 1.30 times the leading term. Two orders later it is already 44.54 times as large.

![The RDP curve of the Poisson-subsampled Gaussian with $q = 0.01$, $\sigma = 1$ (black), against the unsubsampled Gaussian $\alpha/(2\sigma^2)$, the small-$q$ term $\alpha q^2(e^{1/\sigma^2} - 1)/2$ and the leading term of Abadi et al.'s bound. The vertical line marks where Abadi et al.'s bound stops applying. The dashed vertical line marks $\alpha = 2\sigma^2\ln(1/q)$, where the quadratic expansion breaks. The exact curve stays near the quadratic term up to about $\alpha = 8$, then rises toward the unsubsampled line.](figures/subsampling_rdp_curve.svg){#fig-sub-rdp}

So use @eq-sub-small-q for quick estimates, and never inside an accountant. It is 2246 times too small at order 16 for the running example, so an accountant that used it at such orders would under-report $\varepsilon$.

## The RDP accountant for DP-SGD {#sec-sub-dpsgd}

Everything is now in place for the running example of this part. It runs DP-SGD with Poisson sampling at rate 0.01 and noise multiplier 1, for up to ten thousand steps. Its target slack is $\delta = 10^{-5}$. Each step is the subsampled Gaussian of @sec-sub-rdp-gaussian, and it also depends on the earlier steps through the model it starts from. RDP curves add under such adaptive composition too, so after a number of steps $k$, the curve of the whole run is $k\,\varepsilon(\alpha)$.

The accountant converts this total to $(\varepsilon, \delta)$ once. A conversion takes the RDP value at one order and returns an $\varepsilon$ for the target $\delta$. It holds at every order, so the accountant takes the order that gives the smallest $\varepsilon$. Mironov's conversion from [Rényi DP](01-renyi-dp.md) gives
$$\varepsilon = \min_{\alpha}\Big[k\,\varepsilon(\alpha) + \frac{\ln(1/\delta)}{\alpha - 1}\Big].$$ {#eq-sub-dpsgd}
The improved conversion of [From RDP to (ε, δ)](02-rdp-conversions.md) keeps the term $\ln(1/\delta)/(\alpha - 1)$ and adds two small negative corrections to it. Written out, it is
$$\varepsilon = \min_\alpha\Big[k\,\varepsilon(\alpha) + \ln\frac{\alpha - 1}{\alpha} - \frac{\ln\delta + \ln\alpha}{\alpha - 1}\Big].$$
The result is an $(\varepsilon, \delta)$ guarantee under add/remove neighbours.

The alternative is the per-step route. It amplifies each step with @eq-sub-amplify and then composes the per-step pairs, by basic or by advanced composition ([Composition](../part03-dp-foundations/04-composition.md)). @tbl-sub-dpsgd compares the routes.

| Steps $k$ | RDP, Mironov | RDP, improved | Best order (improved) | Per step, basic | Per step, advanced |
| --- | --- | --- | --- | --- | --- |
| 1 | 1.32 | 0.96 | 9.9 | 0.20 | 1.13 |
| 10 | 1.41 | 1.04 | 9.4 | 3.64 | 7.88 |
| 100 | 1.61 | 1.21 | 8.8 | 58.01 | 78.10 |
| 1,000 | 2.54 | 2.10 | 7.8 | 839.39 | 1258.64 |
| 10,000 | 7.43 | 6.71 | 4.1 | 11296.93 | 24379.55 |

: The $\varepsilon$ at $\delta = 10^{-5}$ of DP-SGD with $q = 0.01$, $\sigma = 1$ under four accounting routes. {#tbl-sub-dpsgd}

![The $\varepsilon$ at $\delta = 10^{-5}$ of DP-SGD with $q = 0.01$, $\sigma = 1$ against the number of steps, under four accounting routes. The RDP accountant grows slowly; per-step amplification followed by basic or advanced composition grows almost linearly and is useless beyond a few steps.](figures/subsampling_dpsgd.svg){#fig-sub-dpsgd}

After ten thousand steps, the RDP accountant with the improved conversion gives $\varepsilon = 6.71$. The per-step routes give numbers in the thousands (@fig-sub-dpsgd). They fail because of the large-$\varepsilon$ regime of @sec-sub-eps-delta. Basic composition adds up the slacks of the steps, and amplification multiplies each step's slack by $q$. So to end at the target $\delta$, each step before amplification may have a slack of only $\delta/(kq)$, which is $10^{-7}$ here. For a slack that small, the Gaussian mechanism's exact $(\varepsilon, \delta)$ curve ([The Laplace and Gaussian mechanisms](../part03-dp-foundations/03-laplace-gaussian.md)) forces a large per-step $\varepsilon$, here 5.35. That is above $\ln(1/q)$, where sampling only subtracts, so the amplified step still costs 1.130. Basic composition then multiplies that by ten thousand.

Advanced composition does not rescue the per-step route, and here it is worse than basic composition at every $k$. The reason is a second term in its bound, $k\varepsilon_1(e^{\varepsilon_1} - 1)$, where $\varepsilon_1$ is the amplified per-step cost. Once that cost exceeds $\ln 2$, this term alone is larger than $k\varepsilon_1$, the whole of basic composition. The RDP accountant avoids the trap altogether, because each step costs almost nothing at moderate orders and the conversion term is paid only once, at the end.

That one-time term is also why RDP loses for a single step, 0.96 against 0.20. The conversion costs $\ln(1/\delta)/(\alpha - 1)$ however small the curve is. RDP pays it once, so it wins as soon as there are a few steps to compose, from four steps in this example.

The best order falls from 9.9 to 4.1 as the number of steps grows. A larger $k\,\varepsilon(\alpha)$ pulls the minimum towards smaller orders, where the curve is lower. For a few steps, the best order sits at the foot of the steep rise in @fig-sub-rdp, because the curve term is still small next to the conversion term there. Make sure the accountant's grid of orders includes small and fractional ones. The conversion is a minimum over the orders, so removing orders can only raise the reported $\varepsilon$.

The value 6.71 is an upper bound with two sources of slack. One is the conversion from RDP, and the other is the RDP curve itself, which keeps only one summary of the privacy loss. [Numerical accounting](06-numerical-accounting.md) computes the tight value for the same run, and [Choosing an accountant](09-choosing-an-accountant.md) compares the accountants of this part on it.

## No tight amplification theorem for every mechanism {#sec-sub-generic}

For $(\varepsilon, \delta)$-DP, @eq-sub-amplify is one formula for every mechanism, and it is tight, because randomized response attains it for every $\varepsilon$ and sampling rate. For RDP, no rule takes any mechanism's curve and returns its subsampled curve exactly. Mechanisms with the same curve can subsample differently, and the best general bounds are tight only up to small constant factors.

The general bound for Poisson sampling, due to Zhu & Wang (2019), starts from the same binomial expansion as @eq-sub-binomial. It bounds term $j$ by the mechanism's own RDP curve at order $j$. It holds for integer orders and add/remove neighbours, and it multiplies every term from $j = 3$ on by a factor 3. Two things make the general bound loose. First, each term is bounded by the worst pair of datasets for its own order. Different orders may have different worst pairs, so the term-by-term bound may not be attained. Second, the expansion covers only the direction with the mixture first. For a general mechanism the other direction can be larger, and the factor 3 is there to cover it. Wang, Balle & Kasiviswanathan (2019) give a bound of the same kind for fixed-size batches under replace-one neighbours.

For the Gaussian, neither problem arises. The pair whose sums differ by the full sensitivity is the worst at every order, and the mixture-first order always dominates. That is why accountants use the exact sum of @eq-sub-binomial for the Gaussian. Applied to the running example's Gaussian step, the general bound is exact at order 2, because the sum then has no terms with the factor 3. At order 8, though, it is 1.57 times the exact value, and that order lies close to the best orders for runs of a hundred to a thousand steps. So keep the general bounds for other mechanisms.

Lebeda et al. (2024) add two warnings for numerical accountants. Under add/remove, the worst pair of datasets for one step need not be the worst for many steps, and they give a counterexample with the subsampled Laplace mechanism. An accountant must therefore check its worst-case pair, not assume it. Under replace-one there may be several candidate worst-case pairs, and an accountant must consider all of them.

## Code {#sec-sub-code}

An RDP accountant for DP-SGD under Poisson sampling and add/remove neighbours fits in a few lines. This one computes @eq-sub-binomial in log space, so that large orders do not overflow, and it uses the integer orders 2 to 63 only. With fractional orders, as in @tbl-sub-dpsgd, the improved value drops slightly, from 6.719 to 6.71.

```python
from math import exp, lgamma, log


def amplify(eps: float, delta: float, q: float) -> tuple[float, float]:
    """(eps, delta) of an add/remove-DP mechanism run on a Poisson sample with rate q."""
    return log(1 + q * (exp(eps) - 1)), q * delta


def subsampled_gaussian_rdp(alpha: int, q: float, sigma: float) -> float:
    """RDP at integer order alpha of one Poisson-subsampled Gaussian step (sensitivity 1)."""
    logs = [
        lgamma(alpha + 1) - lgamma(j + 1) - lgamma(alpha - j + 1)
        + (alpha - j) * log(1 - q) + j * log(q) + (j * j - j) / (2 * sigma**2)
        for j in range(alpha + 1)
    ]
    top = max(logs)
    log_a = top + log(sum(exp(t - top) for t in logs))  # log-sum-exp avoids overflow
    return log_a / (alpha - 1)


def dpsgd_eps(q: float, sigma: float, steps: int, delta: float, improved: bool = True) -> tuple[float, int]:
    """Compose the RDP curve over the steps, convert to (eps, delta) at the best integer order."""
    best = (float("inf"), 0)
    for alpha in range(2, 64):
        rdp = steps * subsampled_gaussian_rdp(alpha, q, sigma)
        if improved:
            eps = rdp + log((alpha - 1) / alpha) - (log(delta) + log(alpha)) / (alpha - 1)
        else:
            eps = rdp + log(1 / delta) / (alpha - 1)
        best = min(best, (eps, alpha))
    return best


print(f"{amplify(1.0, 0.0, 0.01)[0]:.4f}")  # 0.0170
print(f"{subsampled_gaussian_rdp(2, 0.01, 1.0):.4e}")  # 1.7181e-04
print("%.3f %d" % dpsgd_eps(0.01, 1.0, 10_000, 1e-5))  # 6.719 4
print(f"{dpsgd_eps(0.01, 1.0, 10_000, 1e-5, improved=False)[0]:.3f}")  # 7.469
print("%.4f %d" % dpsgd_eps(0.01, 4.0, 10_000, 1e-5, improved=False))  # 1.2586 20
print("%.4f %d" % dpsgd_eps(0.01, 4.0, 10_000, 1e-5))  # 1.0355 17
```

The last two lines reproduce Abadi et al.'s example. For fixed-size batches under add/remove neighbours, call `dpsgd_eps` with $q = m/n$ and half the noise multiplier. In production, use a maintained library and check it against a second one ([Choosing an accountant](09-choosing-an-accountant.md)). The full edition's script, [subsampling.py](../../code/part04-privacy-accounting/subsampling.py), computes every number in this chapter, including the fractional orders and the sampler comparisons. Run it from the repository root.

```bash
uv run python code/part04-privacy-accounting/subsampling.py
```

## Cheat sheet

| Result | Formula | Conditions |
| --- | --- | --- |
| One step, Poisson | $\varepsilon' = \ln(1 + q(e^\varepsilon - 1))$, $\delta' = q\delta$ | Base guarantee and result under add/remove; independent coins; tight for one step |
| One step, fixed size $m$ of $n$ | The same with $q = m/n$ | Base guarantee and result under replace-one; tight for one step; no Poisson-accountant shortcut |
| Small and large $\varepsilon$ | $\varepsilon' \approx q\varepsilon$, or $\varepsilon' \approx \varepsilon - \ln(1/q)$ | Poisson, add/remove; switch near $\varepsilon = \ln(1/q)$ |
| One step, Poisson, replace-one | $\varepsilon' = \ln(1 + q(e^{2\varepsilon} - 1))$, $\delta' = q(1 + e^\varepsilon)\delta$ | Group privacy on the base step; an upper bound, not tight |
| Fixed size, clipped Gaussian sum | Poisson accountant with rate $m/n$ and noise multiplier $\sigma/2$ | Add/remove, each direction separately (Lebeda et al. 2024) |
| Shuffle and partition | No simple amplification; the Poisson number can be far too small | Zero-out; lower bounds by attack |
| Subsampled Gaussian RDP | $\varepsilon(\alpha) = \ln A_\alpha/(\alpha - 1)$, with $A_\alpha$ from @eq-sub-binomial | Poisson, add/remove, sensitivity 1, integer $\alpha$; exact |
| Small $q$ | $\varepsilon(\alpha) \approx \alpha q^2(e^{1/\sigma^2} - 1)/2$ | Poisson, add/remove; only for $\alpha$ below about $2\sigma^2\ln(1/q)$ |
| DP-SGD | $\min_\alpha[k\,\varepsilon(\alpha) + \text{conversion}]$ | Poisson, add/remove, adaptive steps; an upper bound, 6.71 for the running example |
| Any mechanism, Poisson | Binomial expansion with a factor 3 on the terms from $j = 3$ on | Integer $\alpha$, add/remove (Zhu & Wang 2019); not tight |

: The results of this chapter. {#tbl-sub-cheat}

**Rules of thumb**

- Sampling multiplies a small $\varepsilon$ by the sampling rate $q$, but takes at most $\ln(1/q)$ off a large one.
- In RDP the per-step cost is quadratic in $q$, but only up to orders near $2\sigma^2\ln(1/q)$.
- Never compose per-step $(\varepsilon, \delta)$ pairs for DP-SGD. Compose RDP curves, or privacy loss distributions, and convert once.
- Account for the sampler you run, and state the neighbouring relation with $\varepsilon$. Fixed-size batches under add/remove mean half the noise multiplier in a Poisson accountant, and shuffled batches have no Poisson guarantee.
- The sampling scheme can move $\varepsilon$ by a factor of seven, while the choice of conversion moves it by about 10%.

## Further reading

- Balle, Barthe & Gaboardi, "Privacy Amplification by Subsampling: Tight Analyses via Couplings and Divergences" (NeurIPS 2018): tight $(\varepsilon, \delta)$ amplification for each sampling scheme and neighbouring relation.
- Mironov, Talwar & Zhang, "Rényi Differential Privacy of the Sampled Gaussian Mechanism" (2019): the subsampled Gaussian's RDP curve and its stable computation, the basis of most RDP accountants.
- Lebeda, Regehr, Kamath & Steinke, "Avoiding Pitfalls for Privacy Accounting of Subsampled Mechanisms under Composition" (IEEE SaTML 2025): fixed-size against Poisson sampling, and worst-case pairs that fail under composition.
- Chua, Ghazi, Kamath, Kumar, Manurangsi, Sinha & Zhang, "How Private are DP-SGD Implementations?" (ICML 2024): why shuffle-and-partition batching can be much less private than the Poisson accountant reports.

**Derivations.** Every result in this chapter is derived step by step in [the full edition's chapter](../../book-full/part04-privacy-accounting/05-subsampling.md), except those it cites: the fixed-size result under add/remove and the warnings about worst-case pairs (Lebeda et al.), the dominance of the mixture-first order for the subsampled Gaussian (Mironov, Talwar & Zhang), Abadi et al.'s small-$q$ bound, and the general bounds of Zhu & Wang and of Wang, Balle & Kasiviswanathan.
