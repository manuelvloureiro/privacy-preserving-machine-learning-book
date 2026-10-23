# Accounting for DP-SGD

This appendix follows one training run from the clipped gradient to its final guarantee of differential privacy (DP). The run is stochastic gradient descent (SGD) with per-example clipping and Gaussian noise, which is called DP-SGD. A training algorithm is **differentially private** when adding or removing one record changes the probability of any set of outcomes by at most a factor $e^\varepsilon$, plus a small slack $\delta$. The **sensitivity** of a computation is the most that one record can move its result. The **Gaussian mechanism** adds normal noise to such a result, and its privacy depends only on the sensitivity divided by the noise's standard deviation. Appendix @sec-apx-cal-gauss-exact derives the Gaussian mechanism's exact guarantee, and @sec-apx-ccomp-rdp derives the Rényi divergence, the tool that adds up the cost of many steps. This page uses both and restates each where it is needed.

The five sections answer the questions a practitioner asks, in the order a training run raises them. Clipping fixes the sensitivity, and the noise then turns each step into a Gaussian mechanism. Random batches make each step cheaper, and a curve built from the Rényi divergence measures that cheaper step exactly. Finally, an accountant adds the steps up and converts the total into one $(\varepsilon, \delta)$. Every number uses the book's running example, so you can check each one against the chapters. It trains on 10,000 records for 100 epochs. Each batch is drawn at random and holds 100 records on average, and the run and reports its guarantee at $\delta = 10^{-5}$. @tbl-apx-sgd-map maps each section to the chapter sections that use its formula.

| Section | Formula it derives | Used in |
| --- | --- | --- |
| @sec-apx-sgd-clipping | the clipped sum has sensitivity $C$ | @sec-dpsgd-algorithm, @sec-dpsgd-microbatch, @sec-sub-schemes |
| @sec-apx-sgd-step | one step has ratio $1/\sigma$; $T$ unsampled steps have $\sqrt T/\sigma$ | @sec-dpsgd-algorithm, @sec-dpsgd-batch, @sec-lg-analytic |
| @sec-apx-sgd-subsampling | $\varepsilon' = \ln\big(1 + q(e^\varepsilon - 1)\big)$ | @sec-sub-eps-delta |
| @sec-apx-sgd-rdp | the binomial sum for the subsampled Gaussian's Rényi curve | @sec-sub-rdp-gaussian, @sec-sub-small-q, @eq-dpsgd-rdp |
| @sec-apx-sgd-accounting | $\varepsilon = \min_\alpha\big[T\varepsilon(\alpha) + \ln(1/\delta)/(\alpha - 1)\big]$ | @sec-sub-dpsgd, @sec-rdp-conversion, @sec-choose-inverse |

: The five derivations of this appendix and the chapter sections that use them. {#tbl-apx-sgd-map}

The notation is that of the DP-SGD chapter, and each symbol is introduced where it first appears. Two datasets are **neighbours** when one of them is the other with one record added. Write $D$ for a dataset and $D' = D \cup \{x\}$ for the same dataset with the record $x$ added. This **add/remove** relation is the one the book uses for DP-SGD. The reason is the way DP-SGD draws its batches, by an independent coin flip for each record, so that adding a record simply adds one more coin (@sec-dpdef-neighbours, and @sec-apx-sgd-subsampling).

## Clipping gives the sum a sensitivity {#sec-apx-sgd-clipping}

### The question

The noise has to hide any one record's contribution, so you need a bound on how far one record can move the sum of gradients. But a gradient can have any length. A mislabelled record far from the decision boundary can have a gradient a thousand times longer than a typical one, and nothing about the model bounds it. How do you get a bound that holds for every model and every record?

### The answer

Write $g_i$ for the **per-example gradient** of record $i$, the gradient of that record's own loss at the current weights. You rescale each one so that its length is at most a number $C$ that you choose, the **clipping norm**. Rescaling keeps the direction and caps the length:
$$\bar g_i = g_i\min\Big(1, \frac{C}{\|g_i\|}\Big), \qquad \bar G(D) = \sum_{i \in D}\bar g_i.$$ {#eq-apx-sgd-clip}
Here $\|g_i\|$ is the Euclidean length of the gradient and $\bar G(D)$ is the **clipped sum** over the records of a dataset $D$. Its $\ell_2$ sensitivity is the largest length of $\bar G(D') - \bar G(D)$ over neighbouring datasets, and it is
$$\Delta_2 \bar G = C \quad\text{under add/remove}, \qquad \Delta_2 \bar G = 2C \quad\text{under replace-one}.$$ {#eq-apx-sgd-sens}
Under **replace-one**, the other common relation, neighbouring datasets have the same size and differ in the value of one record. The book uses the add/remove relation for DP-SGD, so the sensitivity is $C$.

### A worked example

Take $C = 1$ and a dataset $D$ of two records with two-dimensional gradients. The first is $(3, 4)$, of length 5, so clipping divides it by 5 and gives $(0.60, 0.80)$. The second is $(0.3, -0.4)$, of length 0.5, which is shorter than $C$ and stays as it is. The clipped sum is $\bar G(D) = (0.90, 0.40)$.

Now add a third record $x$ with gradient $(-2.4, 0.7)$, of length 2.5. Clipping divides it by 2.5 and gives $(-0.96, 0.28)$, so $\bar G(D') = (-0.06, 0.68)$. The sum moved by $(-0.96, 0.28)$, which is exactly the new record's clipped gradient. Its length is $\sqrt{0.96^2 + 0.28^2} = 1.00$, which is $C$.

Under the replace-one relation, change the first record's gradient from $(3, 4)$ to $(-3, -4)$ instead. Its clipped term goes from $(0.60, 0.80)$ to $(-0.60, -0.80)$, so the sum becomes $(-0.30, -1.20)$. It moved by $(-1.20, -1.60)$, of length 2.00, which is $2C$.

### Why it works

1. **A clipped gradient has length at most $C$.** If $\|g_i\| \le C$, the factor $\min(1, C/\|g_i\|)$ is 1 and the gradient is unchanged. Otherwise the factor is $C/\|g_i\|$, and the length becomes $\|g_i\| \cdot C/\|g_i\| = C$. Either way $\|\bar g_i\| \le C$, and the direction is the same because the factor is positive.
2. **The other records' terms do not change.** Each record's gradient is computed from that record alone and from the current weights. The weights come from the earlier noisy steps, which are already released, so they are fixed when this step runs. Adding $x$ therefore leaves every other term of the sum exactly as it was.
3. **Add/remove: the sum moves by one clipped gradient.** By step 2, $\bar G(D') - \bar G(D) = \bar g_x$, and by step 1 its length is at most $C$. That is the first half of @eq-apx-sgd-sens.
4. **Replace-one: the sum moves by a difference of two.** If record $x$ is replaced by $x'$, the sum moves by $\bar g_{x'} - \bar g_x$. The triangle inequality says that the length of a difference is at most the sum of the lengths, so it is at most $C + C = 2C$. The worked example reaches $2C$, because two gradients that point in opposite directions give the largest difference.

Nothing in these steps used the loss, the architecture or the data. That is why clipping gives a sensitivity for every model.

### Checking it

This appendix's script, described in the Code section at the end, draws 20,000 random datasets of five two-dimensional gradients, with lengths that are often far above $C$, and a random new record for each. The largest change of the clipped sum, in units of $C$, is 1.0000 when the record is added and 2.0000 when it replaces another. Both bounds are reached, so neither can be improved.

@fig-apx-sgd-clipping shows the mechanism. On the left, each gradient outside the circle of radius $C$ is pulled back onto it, along its own direction, and those inside are left alone. On the right, adding one record moves the clipped sum by that record's clipped gradient, so the new sum always lands inside a circle of radius $C$ around the old one.

![Left: nine random per-example gradients (grey) and the same gradients after clipping to norm $C = 1$ (blue), with the ball of radius $C$ dashed. Right: the clipped sum on $D$ and on $D$ plus one record, with the circle of radius $C$ around the first, inside which the second must land.](figures/sgd_clipping.svg){#fig-apx-sgd-clipping}

### Recipe and pitfalls

Clip every per-example gradient to a norm $C$ that is fixed before training, and the sum has sensitivity $C$ under add/remove. The clipping norm is not a privacy parameter. The next section sets the noise to $\sigma C$, a fixed multiple $\sigma$ of the clipping norm. So a larger $C$ means more noise in the same proportion, and the two changes cancel in $\varepsilon$ (@sec-apx-sgd-step). The clipping norm affects only accuracy, which @sec-dpsgd-clipping studies.

Step 2 is where implementations break the argument. Batch normalization makes each record's gradient depend on the other records in the batch, so one record can move every term of the sum. A clipping norm computed from the current batch, such as its median gradient length, has the same flaw in another place. The noise level then depends on the data, so it can itself reveal something about the records.

Two other cases double the sensitivity. Clipping the mean gradient of a group of records, a microbatch, has sensitivity $2C$. The group's clipped mean is a vector of length at most $C$ both before and after one member is added, and the two vectors can point in opposite directions, as in the replace-one example above (@sec-dpsgd-microbatch). The replace-one relation also has sensitivity $2C$. In both cases the program that adds up the privacy cost, the **accountant**, must be told, because it assumes $C$ (@sec-sub-schemes).

## One noisy step is a Gaussian mechanism {#sec-apx-sgd-step}

### The question

Each step releases the clipped sum plus noise. How much noise makes that one step private, and what do the rest of the update and the later steps cost?

### The answer

DP-SGD adds normal noise to the clipped sum, independently in every coordinate. Its standard deviation is $\sigma C$, where the number $\sigma$ is the **noise multiplier**, the noise's standard deviation in units of the sensitivity. So step number $t$ of training releases
$$G_t = \bar G + \xi_t, \qquad \xi_t \sim \mathcal N(0, \sigma^2C^2I).$$
Here $I$ is the identity matrix, which says that the noise $\xi_t$ has the same spread in every direction.

A Gaussian mechanism is described by one ratio, the sensitivity divided by the noise's standard deviation, which the book calls $\mu$. For one step it is
$$\mu = \frac{\Delta_2 \bar G}{\sigma C} = \frac{C}{\sigma C} = \frac1\sigma.$$
A larger ratio means the two output distributions overlap less, so an observer can tell them apart more easily. The exact trade-off between $\varepsilon$ and $\delta$ is the probability mass where the distribution with the record exceeds $e^\varepsilon$ times the one without it. For the mechanism with ratio $\mu$ that mass gives the condition
$$\delta \ge \delta_\mu(\varepsilon) = \Phi\Big(\frac\mu2 - \frac\varepsilon\mu\Big) - e^\varepsilon\,\Phi\Big(-\frac\mu2 - \frac\varepsilon\mu\Big),$$ {#eq-apx-sgd-gauss-delta}
where $\Phi(z)$ is the probability that a standard normal variable is at most $z$ (@sec-lg-analytic, and @sec-apx-cal-gauss-exact for a derivation from scratch). Without sampling, every record is used in every step. If the run has $T$ steps, they act together like one Gaussian mechanism with ratio
$$\mu_T = \frac{\sqrt T}{\sigma}.$$

### A worked example

Take $\sigma = 1$, so one step has $\mu = 1$. Solving @eq-apx-sgd-gauss-delta for $\delta = 10^{-5}$ gives $\varepsilon = 4.38$ for a single step, which is already a large privacy cost. The cost grows quickly with the number of steps if every record is used in each one. For example, 100 steps have $\mu_T = \sqrt{100} = 10$ and $\varepsilon = 91.82$, and the running example's 10,000 steps have $\mu_T = 100$ and $\varepsilon = 5425.51$. Nobody would report such a number, and the next two sections show how random batches bring it down to single digits.

The DP-SGD chapter has a full-batch row in @tbl-dpsgd-batch, where every batch is the whole dataset, so its 100 epochs are 100 steps. For a target of $\varepsilon = 4$ that row uses $\sigma = 10.8$. So $\mu_T = \sqrt{100}/10.8 = 0.926$, and the script gives $\varepsilon = 4.005$, the target up to the rounding of $\sigma$.

### Why it works

1. **Only the ratio matters.** Compare the noisy sum on $D$ and on $D'$. The two are normal distributions with the same spread, centred $s$ apart, where the shift $s$ is the length of $\bar G(D') - \bar G(D)$. Rotate the space so that this shift lies along the first axis. The noise has the same spread in every direction, so rotating it leaves it unchanged. Every other coordinate then has the same noise on both datasets and tells an observer nothing. What is left is one coordinate, $\mathcal N(0, \sigma^2C^2)$ against $\mathcal N(s, \sigma^2C^2)$. Dividing the output by $\sigma C$ changes no probability of any event, so this is the pair $\mathcal N(0, 1)$ against $\mathcal N(s/(\sigma C), 1)$.
2. **The worst shift gives ratio $1/\sigma$.** By @sec-apx-sgd-clipping the shift is at most $C$, so the ratio is at most $C/(\sigma C) = 1/\sigma$. A smaller ratio is never less private, because the two distributions then overlap more. Notice that the clipping norm cancels, which is why it is not a privacy parameter.
3. **The rest of the update is free.** The learning rate, the division by the expected batch size $B$, momentum, Adam and the new weights are all computed from $G_t$ and from earlier released values, without looking at the data again. Such a computation is called **post-processing**, and it cannot reveal more than its input. The reason is that an attacker who sees $G_t$ could run the same computation herself, so any event about the processed output is an event about $G_t$, and its probability obeys the same bound (@sec-dpprop-post).
4. **Unsampled steps add up like one Gaussian.** The **privacy loss** of an output $y$ is the logarithm of how much more likely $y$ is on $D'$ than on $D$. For the pair $\mathcal N(\mu, 1)$ against $\mathcal N(0, 1)$, the loss is
$$\ln\frac{e^{-(y - \mu)^2/2}}{e^{-y^2/2}} = \mu y - \frac{\mu^2}2.$$
With $y \sim \mathcal N(\mu, 1)$, this is a normal variable with mean $\mu^2/2$ and variance $\mu^2$. Each step draws fresh noise, independent of the earlier steps. So over $T$ steps the joint likelihood ratio is the product of the steps' ratios, and the losses add. Each step's loss has this same distribution whatever happened before, so the sum is normal with mean $T\mu^2/2$ and variance $T\mu^2$. That is exactly the loss of one Gaussian mechanism with ratio $\sqrt T\mu = \sqrt T/\sigma$. Since $\delta$ for a given $\varepsilon$ is computed from the distribution of the loss alone, the two have the same guarantee, and @sec-apx-ccomp-gaussian proves it in full.

### Checking it

The full-batch check above recovers the chapter's $\varepsilon = 4$ to three digits. It also confirms step 4, because the chapter solved its row for 100 composed steps, while here the 100 steps are one Gaussian with $\mu_T = 10/\sigma$. The script solves @eq-apx-sgd-gauss-delta by bisection on $\varepsilon$, working with logarithms, because at $\mu_T = 100$ both terms underflow to zero in ordinary arithmetic. The value at 10,000 steps, 5425.51, agrees with the 5425.5 quoted in [Amplification by subsampling](../part04-privacy-accounting/05-subsampling.md).

### Recipe and pitfalls

Set the noise to $\sigma C$, and one step is a Gaussian mechanism with ratio $1/\sigma$, whatever $C$ is. Keep everything after the noise a function of the noisy sum and of public numbers. Dividing by the number of records actually in the batch is the common mistake here, because that count depends on the data and changes when $x$ is added. Divide by the expected batch size $B$ instead, the sampling rate times the number of records, which is public. And never compose unsampled steps for a long run, because $\varepsilon$ grows at least like $\mu_T^2/2 = T/(2\sigma^2)$, the average privacy loss, and reaches the thousands.

## Random batches make each step cheaper {#sec-apx-sgd-subsampling}

### The question

DP-SGD does not use every record in every step. Each step draws a random batch, and with $q = 0.01$ your record is in it only one time in a hundred. How much cheaper is a step that probably did not see you?

### The answer

Under **Poisson sampling**, each record enters the batch independently with probability $q$, the **sampling rate**, by its own coin flip. If a step is $\varepsilon$-DP when it runs on whatever batch it is given, then the step run on a Poisson sample is $\varepsilon'$-DP under add/remove, with
$$\varepsilon' = \ln\big(1 + q(e^\varepsilon - 1)\big).$$ {#eq-apx-sgd-amplify}
This is **amplification by subsampling**. When $\varepsilon$ is small, $e^\varepsilon - 1$ is close to $\varepsilon$, and the logarithm of a number close to 1 is close to that number minus 1. So for small $\varepsilon$
$$\varepsilon' \approx q\varepsilon,$$
and the step becomes about $1/q$ times cheaper. When the step also has a slack $\delta$, the same argument shrinks that slack to $q\delta$ (@eq-sub-amplify).

### A worked example

Take $q = 0.01$ and a step with $\varepsilon = 1$. Then $e - 1 = 1.71828$, so $1 + q(e^\varepsilon - 1) = 1.017183$ and $\varepsilon' = \ln 1.017183 = 0.017037$. The step costs about 0.017 instead of 1, almost $q\varepsilon = 0.01$. @tbl-apx-sgd-amp shows how the saving shrinks for larger $\varepsilon$.

| Base $\varepsilon$ | 0.1 | 1 | 5 | 10 |
| --- | --- | --- | --- | --- |
| Sampled $\varepsilon'$, $q = 0.01$ | 0.0011 | 0.0170 | 0.9059 | 5.3993 |
| $q\varepsilon$ | 0.0010 | 0.0100 | 0.0500 | 0.1000 |

: Amplification by Poisson sampling at rate $q = 0.01$. For small $\varepsilon$ the cost shrinks by about $q$; for large $\varepsilon$ it only drops by about $\ln(1/q) = 4.6$. {#tbl-apx-sgd-amp}

### Why it works

The idea is that with probability $1 - q$ your record is not used, and then the output has exactly the distribution it would have without you. Only the remaining fraction $q$ of the time can the output differ. So every likelihood ratio becomes a weighted average of 1, with weight $1 - q$, and the old ratio, with weight $q$.

A tiny case shows it. Suppose a step's output $y$ is three times as likely when your record is used. Under sampling, $y$'s probability with you in the dataset is $(1 - q) \cdot p + q \cdot 3p$, where $p$ is its probability without you. With $q = 0.01$ that is $0.99p + 0.03p = 1.02p$, so the ratio has fallen from 3 to 1.02. Here are the steps in general.

1. **Fix everyone else's coin.** Let $U$ be the set of other records that were drawn. Each record has its own coin, so given $U$, record $x$ is drawn with probability $q$, independently of $U$.
2. **Write both outputs given $U$.** Let $P_0$ be the output distribution of the step run on $U$, and $P_1$ that of the step run on $U \cup \{x\}$. On $D$ the output is drawn from $P_0$. On $D'$ it is drawn from $P_0$ when the coin of $x$ says no, and from $P_1$ when it says yes. So for every set of outputs $S$,
$$P_{D'}(S \mid U) = (1 - q)P_0(S) + qP_1(S).$$
3. **Use the step's own guarantee.** The sets $U$ and $U \cup \{x\}$ are neighbours, and the step is $\varepsilon$-DP, so $P_1(S) \le e^\varepsilon P_0(S)$ for every $S$.
4. **Put it into the mixture.** Replacing $P_1(S)$ by the larger $e^\varepsilon P_0(S)$ gives
$$P_{D'}(S \mid U) \le (1 - q)P_0(S) + qe^\varepsilon P_0(S) = \big(1 + q(e^\varepsilon - 1)\big)P_0(S).$$
5. **Average over $U$.** Both datasets draw the other records' coins with the same probabilities. The bound of step 4 holds for every $U$ with the same factor, so it holds for the averages over $U$ too. This is @eq-apx-sgd-amplify in the direction "$D'$ against $D$".
6. **Check the other direction.** DP also needs $P_D(S) \le e^{\varepsilon'}P_{D'}(S)$. The step's guarantee holds with either dataset first, so it also gives $P_1(S) \ge e^{-\varepsilon}P_0(S)$, so the mixture is at least $(1 - q + qe^{-\varepsilon})P_0(S)$. Dividing, $P_0(S) \le P_{D'}(S)/(1 - q + qe^{-\varepsilon})$. To compare the two factors, write $a = e^\varepsilon$ and multiply out:
$$\big(1 - q + q/a\big)\big(1 + q(a - 1)\big) = 1 + q(1 - q)\frac{(a - 1)^2}{a} \ge 1.$$
So $1/(1 - q + qe^{-\varepsilon}) \le 1 + q(e^\varepsilon - 1)$, and this direction is never the worse one.

### Checking it

The script takes a count with Laplace noise, whose density falls like $e^{-\varepsilon|y - \text{count}|}$, with $\varepsilon = 1$. Adding the record moves the count by 1, so the noisy count's density ratio between "used" and "not used" ranges from $e^{-1}$ to $e$. It computes the mixture's likelihood ratio on a grid of outputs. At $q = 0.01$ the largest log ratio, with the record against without it, is 0.017037, equal to @eq-apx-sgd-amplify, so the bound is reached and cannot be improved. The other direction's largest log ratio is 0.006341, smaller, as step 6 says. @fig-apx-sgd-subsampling shows the same at $q = 0.1$, where the effect is easier to see. The ratio without sampling climbs to $e$, while the mixture's ratio stays below $1 + 0.1(e - 1) = 1.172$.

![Left: a Laplace count with $\varepsilon = 1$ without the record ($P_0$), with the record used ($P_1$), and the Poisson-sampled output with the record, the mixture $0.9P_0 + 0.1P_1$ (dashed). Right: the likelihood ratios $P_1/P_0$ and mixture$/P_0$ against the output, with the bounds $e^\varepsilon$ and $1 + q(e^\varepsilon - 1)$ dotted.](figures/sgd_subsampling_ratio.svg){#fig-apx-sgd-subsampling}

### Recipe and pitfalls

For a step that is $\varepsilon$-DP on its batch, Poisson sampling at rate $q$ makes it $\ln(1 + q(e^\varepsilon - 1))$-DP. For small $\varepsilon$ this is about $q\varepsilon$. For large $\varepsilon$ it is about $\varepsilon - \ln(1/q)$, because when the record is drawn, which happens with probability $q$, it is fully exposed.

The proof needs Poisson sampling, because step 1 needs an independent coin for your record. A batch of exactly $B$ records drawn at random has no such coin. When your record enters such a batch it pushes another one out, which is a replacement, so the analysis needs the replace-one relation and twice the sensitivity. Shuffling the data and cutting it into batches is worse, because then your record is in exactly one batch per epoch and is never "not used" (@sec-sub-schemes).

Finally, do not add up these per-step numbers over a long run. The slacks add up too, so with 10,000 steps each step may only have a tiny share of $\delta$. A Gaussian step with a tiny $\delta$ needs a large $\varepsilon$, and there amplification only subtracts $\ln(1/q)$. @sec-sub-dpsgd shows that the total runs into the thousands, and the next section gives the quantity that composes well.

## The Rényi curve of a sampled Gaussian step {#sec-apx-sgd-rdp}

### The question

A single $(\varepsilon, \delta)$ pair says how often the privacy loss is large, but not how large it gets when it is, so pairs add up badly over 10,000 steps. A curve that summarizes the whole distribution of the loss composes by simple addition. What is that curve for one sampled Gaussian step?

### The answer

The **Rényi divergence** of order $\alpha > 1$ between two distributions $P$ and $Q$ is a log-average of a power of their likelihood ratio,
$$D_\alpha(P\|Q) = \frac{1}{\alpha - 1}\ln\mathbb E_{y \sim Q}\Big[\Big(\frac{P(y)}{Q(y)}\Big)^\alpha\Big].$$
A mechanism has **Rényi differential privacy (RDP)** curve $\varepsilon(\alpha)$ when this divergence between its outputs on any two neighbouring datasets is at most $\varepsilon(\alpha)$, for every order $\alpha$. A larger order weights large ratios more heavily. For the plain Gaussian mechanism with noise multiplier $\sigma$ the curve is the straight line $\alpha/(2\sigma^2)$ (@sec-apx-ccomp-rdp).

For one DP-SGD step, divide the noisy sum by $C$. Then the output without the record is $P_0 = \mathcal N(0, \sigma^2)$, and with the record it is the mixture $P_q = (1 - q)P_0 + qP_1$ with $P_1 = \mathcal N(1, \sigma^2)$. This one-dimensional pair is the worst case, as the derivation below explains. Its RDP curve is $\varepsilon_{q,\sigma}(\alpha) = D_\alpha(P_q\|P_0)$, and for integer $\alpha$ it is a finite sum,
$$\varepsilon_{q,\sigma}(\alpha) = \frac{1}{\alpha - 1}\ln A_\alpha, \qquad A_\alpha = \sum_{j=0}^\alpha\binom\alpha j(1 - q)^{\alpha - j}q^je^{(j^2 - j)/(2\sigma^2)}.$$ {#eq-apx-sgd-rdp}
Here $\binom\alpha j$ is the binomial coefficient, the number of ways to choose $j$ of $\alpha$ factors. For small $q$ the curve is close to its leading term,
$$\varepsilon_{q,\sigma}(\alpha) \approx \frac{\alpha q^2\big(e^{1/\sigma^2} - 1\big)}{2} \approx \frac{\alpha q^2}{2\sigma^2}\quad\text{for large }\sigma.$$ {#eq-apx-sgd-small-q}

### A worked example

Take the running example, $q = 0.01$ and $\sigma = 1$, at order $\alpha = 2$. The sum has three terms, one for each $j$ (@tbl-apx-sgd-alpha2).

| $j$ | $\binom 2j(1 - q)^{2-j}q^j$ | $e^{(j^2 - j)/2}$ | Term |
| --- | --- | --- | --- |
| 0 | $0.99^2 = 0.9801$ | 1 | 0.980100 |
| 1 | $2 \times 0.99 \times 0.01 = 0.0198$ | 1 | 0.019800 |
| 2 | $0.01^2 = 0.0001$ | $e = 2.718$ | 0.000272 |

: The three terms of $A_2$ for $q = 0.01$ and $\sigma = 1$. {#tbl-apx-sgd-alpha2}

They add up to $A_2 = 1.00017183$, so $\varepsilon_{q,\sigma}(2) = \ln A_2/(2 - 1) = 1.7181 \times 10^{-4}$. Without sampling the curve at $\alpha = 2$ is $2/(2\sigma^2) = 1$, so sampling made this order almost six thousand times cheaper.

### Why it works

First, why one dimension and one shift of exactly 1 is enough. Given which other records were drawn, the step's output with the record is a mixture of "not drawn" and "drawn", as in @sec-apx-sgd-subsampling. As in @sec-apx-sgd-step, rotating the space leaves one coordinate in which the two components differ, by at most $C$, and a smaller shift is never worse. Averaging over the other records' coins cannot increase the divergence. The reason is that the average of $(P/Q)^\alpha$ weighted by $Q$ is a convex function of the pair $(P, Q)$, so a mixture of pairs is never further apart than the worst pair in it. @sec-sub-rdp-gaussian gives these reduction steps in full, so what remains is the pair $P_q$ and $P_0$. The derivation then takes five steps.

1. **Write the mixture's ratio.** Dividing the mixture by $P_0$ gives $P_q/P_0 = (1 - q) + q\,r$, where $r = P_1/P_0$ is the Gaussian likelihood ratio. The normalizing constants cancel, and
$$r(y) = \frac{e^{-(y - 1)^2/(2\sigma^2)}}{e^{-y^2/(2\sigma^2)}} = e^{(2y - 1)/(2\sigma^2)}.$$
So by the definition, $A_\alpha = \mathbb E_{y \sim P_0}\big[(1 - q + q\,r(y))^\alpha\big]$.
2. **Expand the power.** For a whole number $\alpha$, the binomial theorem expands the power of a sum of two numbers into $\alpha + 1$ terms. At $\alpha = 2$ it is the familiar $(a + b)^2 = a^2 + 2ab + b^2$. Here the two numbers are $1 - q$ and $qr$, so
$$(1 - q + qr)^\alpha = \sum_{j=0}^\alpha\binom\alpha j(1 - q)^{\alpha - j}q^jr^j.$$
3. **Average term by term.** The average of a sum is the sum of the averages, and $(1 - q)^{\alpha - j}q^j$ is a constant. So $A_\alpha$ is the same sum with each $r^j$ replaced by its average $\mathbb E[r^j]$ under $P_0$.
4. **Each average is a Gaussian moment generating function (MGF).** The MGF of a random variable $y$ is the average of $e^{\lambda y}$, as a function of a number $\lambda$. For $y \sim \mathcal N(0, \sigma^2)$ it is
$$\mathbb E[e^{\lambda y}] = e^{\lambda^2\sigma^2/2}.$$
To see why, complete the square in the exponent of the integrand, $\lambda y - y^2/(2\sigma^2) = -(y - \lambda\sigma^2)^2/(2\sigma^2) + \lambda^2\sigma^2/2$. After you take out the constant $e^{\lambda^2\sigma^2/2}$, what is left is a normal density centred at $\lambda\sigma^2$, and it integrates to 1. Now $r^j = e^{-j/(2\sigma^2)}e^{(j/\sigma^2)y}$, so taking $\lambda = j/\sigma^2$ gives
$$\mathbb E[r^j] = e^{-j/(2\sigma^2)}\,e^{j^2/(2\sigma^2)} = e^{(j^2 - j)/(2\sigma^2)}.$$
For example, at $j = 2$ and $\sigma = 1$ this is $e^{-1} \cdot e^{2} = e$, the factor in the last row of @tbl-apx-sgd-alpha2.
5. **Assemble.** Putting each average back into the sum gives $A_\alpha$ of @eq-apx-sgd-rdp, and dividing its logarithm by $\alpha - 1$ gives the curve.

The small-$q$ form comes from looking at which terms matter. Without the exponential factors, the terms add up to $(1 - q + q)^\alpha = 1$. The factors for $j = 0$ and $j = 1$ are $e^0 = 1$, so those two terms cost nothing, and
$$A_\alpha = 1 + \sum_{j=2}^\alpha\binom\alpha j(1 - q)^{\alpha - j}q^j\big(e^{(j^2 - j)/(2\sigma^2)} - 1\big).$$
For small $q$ the $j = 2$ term dominates, because every later term carries $q^3$ or a higher power. That term is about $\frac{\alpha(\alpha - 1)}2q^2(e^{1/\sigma^2} - 1)$, so $A_\alpha$ is 1 plus a small number. The logarithm of 1 plus a small number is close to that small number, so dividing by $\alpha - 1$ gives @eq-apx-sgd-small-q. Its second form follows because $e^{1/\sigma^2} - 1$ is close to $1/\sigma^2$ when $\sigma$ is large. Both forms are approximations, not bounds. You will also see the curve quoted as $q^2\alpha/\sigma^2$, which is the leading term of the upper bound Abadi et al. (2016) proved for small $q$, and about twice @eq-apx-sgd-small-q for large $\sigma$ (@sec-sub-small-q).

Last, the direction. The guarantee must hold with either dataset first, so the curve is the larger of $D_\alpha(P_q\|P_0)$ and $D_\alpha(P_0\|P_q)$. The first one puts the dataset with the record first, and it is called the **remove** direction. It is always the larger for the subsampled Gaussian, which is why accountants compute only it. Appendix @sec-apx-amp-direction derives this step by step, and the script checks it below.

### Checking it

The curve can be computed in two independent ways, and the script does both: by the binomial sum, and by averaging $(1 - q + qr)^\alpha$ numerically on a fine grid of outputs. At $\alpha = 2$, 8 and 32 the two agree to all seven printed digits: $1.718134 \times 10^{-4}$, $8.936439 \times 10^{-4}$ and $11.24628$. A Monte Carlo average of $r^2$ over two million normal draws gives 2.7084, against the exact $e = 2.7183$. The other direction is smaller at each of the three orders: $1.6022 \times 10^{-4}$, $5.7239 \times 10^{-4}$ and $1.6923 \times 10^{-3}$.

@tbl-apx-sgd-small-q compares the exact curve with its small-$q$ form at the running example's $q$ and $\sigma$.

| $\alpha$ | 2 | 4 | 8 | 16 | 32 |
| --- | --- | --- | --- | --- | --- |
| Exact @eq-apx-sgd-rdp | $1.7181 \times 10^{-4}$ | $3.6315 \times 10^{-4}$ | $8.9364 \times 10^{-4}$ | 3.0879 | 11.246 |
| Small-$q$ form @eq-apx-sgd-small-q | $1.7183 \times 10^{-4}$ | $3.4366 \times 10^{-4}$ | $6.8731 \times 10^{-4}$ | $1.3746 \times 10^{-3}$ | $2.7493 \times 10^{-3}$ |
| Exact / small-$q$ | 1.00 | 1.06 | 1.30 | 2246.32 | 4090.67 |
| No sampling, $\alpha/(2\sigma^2)$ | 1 | 2 | 4 | 8 | 16 |

: The RDP curve of one DP-SGD step with $q = 0.01$, $\sigma = 1$, against its small-$q$ form and against no sampling. {#tbl-apx-sgd-small-q}

Up to $\alpha = 8$ the approximation is within 30%, but then it fails abruptly, and at $\alpha = 16$ it is 2246 times too small. @fig-apx-sgd-rdp shows why. The terms with large $j$ grow like $q^je^{j^2/(2\sigma^2)}$, and the last one overtakes the $j = 2$ term near $\alpha = 2\sigma^2\ln(1/q)$ (@sec-sub-small-q). So the exact curve follows the small-$q$ line, then climbs steeply near $\alpha = 2\sigma^2\ln(1/q) = 9.2$ and runs toward the unsampled line. At high orders the divergence looks at the rare outputs where the record was clearly used, and there sampling saves only about $\ln(1/q)$. With more noise the approximation holds further: at $\alpha = 8$ the exact curve is 1.019 times the small-$q$ form for $\sigma = 2$ and 1.004 times for $\sigma = 4$.

![The RDP curve of one DP-SGD step against the order $\alpha$, on log scales, for $q = 0.01$ and $\sigma = 1$: exact (black), the small-$q$ term $\alpha q^2(e^{1/\sigma^2} - 1)/2$ (dashed) and no sampling, $\alpha/(2\sigma^2)$. The dotted line marks $\alpha = 2\sigma^2\ln(1/q)$, where the small-$q$ form breaks.](figures/sgd_rdp_curve.svg){#fig-apx-sgd-rdp}

### Recipe and pitfalls

For integer orders, compute @eq-apx-sgd-rdp with logarithms, since the terms span many orders of magnitude. For orders between the integers, average $(1 - q + qr)^\alpha$ numerically, as the chapters' accountants do. Use @eq-apx-sgd-small-q only for estimates at orders below $2\sigma^2\ln(1/q)$, never as the curve itself.

The curve assumes Poisson sampling and the add/remove relation, like @sec-apx-sgd-subsampling. It also assumes the sensitivity is exactly the clipping norm. If something in your code makes the sensitivity $2C$, such as microbatches or the replace-one relation, give the accountant the noise multiplier $\sigma/2$ instead of $\sigma$, because the ratio is then $2C/(\sigma C) = 1/(\sigma/2)$.

## From batch, noise and epochs to the final ε {#sec-apx-sgd-accounting}

### The question

You train with $n$ records, an expected batch of $B$, noise multiplier $\sigma$ and a number of epochs $E$. What $(\varepsilon, \delta)$ do you report? And the reverse question, which a practitioner asks more often: for a target $\varepsilon$, how much noise do you need?

### The answer

The sampling rate is $q = B/n$, and one epoch is $n/B = 1/q$ steps, so there are $T = E\,n/B$ steps in all. The RDP accountant multiplies the per-step curve by $T$ and converts it to $(\varepsilon, \delta)$ at the best order, using Mironov's conversion:
$$\varepsilon = \min_{\alpha}\Big[T\,\varepsilon_{q,\sigma}(\alpha) + \frac{\ln(1/\delta)}{\alpha - 1}\Big].$$ {#eq-apx-sgd-convert}
The second term is the price of converting, which this page calls the conversion **toll**. The minimum runs over a grid of orders. The book's grid has the orders from 1.1 to 10.9 in steps of 0.1, followed by whole orders up to 512. It takes $\varepsilon_{q,\sigma}(\alpha)$ from @eq-apx-sgd-rdp at whole orders and from the numerical average between them.

### A worked example

The running example has $n = 10{,}000$ records and $B = 100$, so the sampling rate is $q = 0.01$. Each of its 100 epochs has 100 steps, so $T = 10{,}000$. The script finds that the best order is $\alpha = 4.3$, where one step costs $\varepsilon_{q,\sigma}(4.3) = 3.9400 \times 10^{-4}$. Multiplying by the 10,000 steps gives a composed curve of 3.9400 at that order. The conversion then adds $\ln(10^5)/(4.3 - 1) = 11.5129/3.3 = 3.4888$. So $\varepsilon = 3.9400 + 3.4888 = 7.4287$, which you would report as $(7.43, 10^{-5})$. @tbl-apx-sgd-epochs gives $\varepsilon$ after each number of epochs.

| Epochs | Steps $T$ | Every record in every step | RDP, Mironov | Best order for Mironov | RDP, improved |
| --- | --- | --- | --- | --- | --- |
| 1 | 100 | 91.8 | 1.61 | 8.9 | 1.21 |
| 5 | 500 | 344.5 | 2.07 | 8.3 | 1.65 |
| 10 | 1,000 | 633.9 | 2.54 | 7.9 | 2.10 |
| 20 | 2,000 | 1189.8 | 3.34 | 7.2 | 2.87 |
| 50 | 5,000 | 2800.6 | 5.18 | 5.5 | 4.59 |
| 100 | 10,000 | 5425.5 | 7.43 | 4.3 | 6.71 |

: The $\varepsilon$ at $\delta = 10^{-5}$ of the running example ($q = 0.01$, $\sigma = 1$) after each number of epochs. "Every record in every step" composes unsampled Gaussian steps exactly (@sec-apx-sgd-step); the RDP columns use Mironov's conversion, @eq-apx-sgd-convert, and the improved conversion of Balle et al. (2020). {#tbl-apx-sgd-epochs}

Sampling turns thousands into single digits. The **improved conversion** of [From Rényi DP to (ε, δ)](../part04-privacy-accounting/02-rdp-conversions.md) subtracts two small corrections from the conversion term $\ln(1/\delta)/(\alpha - 1)$ and gives 6.71 at 100 epochs, the number the chapters report. A **privacy loss distribution (PLD)** accountant tracks the whole distribution of the loss instead of one curve, and @sec-num-dpsgd reports 6.2377 for the same run as its upper bound.

### Why it works

1. **The curves add.** By the definition, the Rényi divergence of the whole run is set by the average of the $\alpha$-th power of its likelihood ratio, and step 2 below writes that as the average of $e^{(\alpha - 1)L}$ for the privacy loss $L$. The run's likelihood ratio is the product of the steps' ratios, so $e^{(\alpha - 1)L}$ is a product of per-step factors. Average this product one step at a time, starting from the last. With everything before it held fixed, the last step is a subsampled Gaussian step, so its factor averages to at most $e^{(\alpha - 1)\varepsilon_{q,\sigma}(\alpha)}$, whatever the earlier steps did. That number is a constant, so it comes out of the average, and the next step is treated the same way. After $T$ steps the average is at most $e^{(\alpha - 1)T\varepsilon_{q,\sigma}(\alpha)}$, so the composed curve is $T\varepsilon_{q,\sigma}(\alpha)$, even though each step depends on the earlier ones (@sec-apx-ccomp-rdp).
2. **The average of a power is the Rényi divergence.** Let $L$ be the privacy loss, $\ln(P(y)/Q(y))$ with $y$ drawn from $P$, where $P$ is the output distribution on the dataset with the record. Then
$$\mathbb E_{y \sim P}\big[e^{(\alpha - 1)L}\big] = \int P\Big(\frac PQ\Big)^{\alpha - 1}\mathrm dy = \int Q\Big(\frac PQ\Big)^\alpha\,\mathrm dy = e^{(\alpha - 1)D_\alpha(P\|Q)}.$$
3. **Markov's inequality bounds the chance of a large loss.** For a variable that cannot be negative, the chance that it is at least $a$ is at most its average divided by $a$. Apply it to $e^{(\alpha - 1)L}$ at the level $a = e^{(\alpha - 1)\varepsilon}$. With step 2 and the composed curve of step 1, this gives
$$\Pr[L > \varepsilon] \le e^{(\alpha - 1)(T\varepsilon_{q,\sigma}(\alpha) - \varepsilon)}.$$
4. **A rare large loss is all that $\delta$ has to cover.** Split any set of outputs $S$ into the outputs where the loss is at most $\varepsilon$ and those where it is larger. On the first part, $P$ is at most $e^\varepsilon$ times $Q$, by the definition of $L$. The second part has probability at most the chance of a loss above $\varepsilon$. Adding the two parts,
$$P(S) \le e^\varepsilon Q(S) + \Pr[L > \varepsilon],$$
so the run is $(\varepsilon, \delta)$-DP as soon as the chance of a loss above $\varepsilon$ is at most $\delta$.
5. **Solve for $\varepsilon$.** Set the bound of step 3 equal to $\delta$, take logarithms and divide by $\alpha - 1$. This gives
$$\varepsilon = T\varepsilon_{q,\sigma}(\alpha) + \frac{\ln(1/\delta)}{\alpha - 1}.$$
It holds at every order, so the smallest value over the orders also holds, which is @eq-apx-sgd-convert. Appendix @sec-apx-conv-rdp derives the conversion, and its improved form, in more detail.

The minimum sits between the extremes because the two terms pull in opposite directions. @fig-apx-sgd-conversion shows them for the running example. The composed curve $T\varepsilon_{q,\sigma}(\alpha)$ rises with the order, slowly at first and then steeply past $\alpha \approx 8$, where the curve leaves its small-$q$ form. The conversion toll $\ln(1/\delta)/(\alpha - 1)$ falls. Their sum is smallest at $\alpha = 4.3$, where it is 7.43. A longer run makes the first term larger and pushes the best order down, from 8.9 after one epoch to 4.3 after 100.

![The two terms of @eq-apx-sgd-convert for the running example after $T = 10{,}000$ steps, against the order $\alpha$: the composed curve $T\varepsilon(\alpha)$, the conversion toll $\ln(1/\delta)/(\alpha - 1)$, and their sum, with its minimum, $\varepsilon = 7.43$ at $\alpha = 4.3$, marked.](figures/sgd_conversion.svg){#fig-apx-sgd-conversion}

### Checking it

The script's accountant reproduces the chapters' numbers to every printed digit: 7.4287 with Mironov's conversion and 6.71 with the improved one, as in @sec-sub-dpsgd. Restricting the grid to whole orders, where @eq-apx-sgd-rdp is exact, costs little, giving 7.4692 at $\alpha = 4$. @fig-apx-sgd-epochs plots $\varepsilon$ against the number of epochs. Without sampling it grows nearly in proportion to the number of steps, while the RDP accountant grows slowly. The PLD upper bounds from @sec-num-dpsgd, marked as crosses, sit below the RDP curves. A PLD bound is proved for every pair of neighbouring datasets, like the RDP bound, and it is the tightest such number for this run. Both are therefore valid guarantees, and with the improved conversion RDP overstates $\varepsilon$ by about 8% at 100 epochs, since $6.71/6.24 = 1.08$.

![The $\varepsilon$ at $\delta = 10^{-5}$ of the running example against the number of epochs, on log scales: every record in every step (exact Gaussian composition), the subsampled run with the RDP accountant under Mironov's and the improved conversion, and the PLD upper bound that Part 4's numerical accountant reports (crosses).](figures/sgd_eps_epochs.svg){#fig-apx-sgd-epochs}

### Recipe and pitfalls

To get $\varepsilon$ from a training configuration, follow these steps.

1. Compute the sampling rate $q$, which is $B/n$, and the number of steps $T = E\,n/B$.
2. Compute the per-step curve $\varepsilon_{q,\sigma}(\alpha)$ on a grid of orders, with @eq-apx-sgd-rdp at whole orders.
3. Multiply the curve by $T$.
4. At each order add the toll $\ln(1/\delta)/(\alpha - 1)$, or use the improved conversion, and report the smallest value with its $\delta$.

To go the other way, from a target $\varepsilon$ to a noise multiplier, search over $\sigma$. More noise never makes a step less private, so $\varepsilon$ falls as $\sigma$ grows, and bisection finds the smallest $\sigma$ that meets the target. @tbl-apx-sgd-sigma shows the result for the running example's $q$ and $T$. The improved column matches @tbl-dpsgd-sigma of the DP-SGD chapter.

| Target $\varepsilon$ at $\delta = 10^{-5}$ | 1 | 2 | 4 | 8 |
| --- | --- | --- | --- | --- |
| $\sigma$, Mironov's conversion | 4.974 | 2.617 | 1.484 | 0.962 |
| $\sigma$, improved conversion | 4.126 | 2.278 | 1.358 | 0.917 |

: The noise multiplier needed for each target $\varepsilon$ with $q = 0.01$ and $T = 10{,}000$ steps, found by bisection on $\sigma$. {#tbl-apx-sgd-sigma}

Most mistakes in reported numbers come from a mismatch between the accountant and the code, not from the accountant itself. Five are common.

- **Sampling must be Poisson, or the analysis must match the sampler.** The whole saving rests on the independent coin of @sec-apx-sgd-subsampling. If the code draws batches of exactly $B$ records, a new record pushes another one out, so the sum has sensitivity $2C$ and the accountant needs $\sigma/2$. For the running example that gives $\varepsilon = 47.42$ instead of 6.71, with the improved conversion. Shuffling and cutting into batches has no simple amplification at all (@sec-sub-schemes).
- **State the neighbouring relation.** The numbers here are for adding or removing one record. A replace-one guarantee is a different statement and needs twice the sensitivity.
- **The clipping norm is not a privacy parameter.** Changing $C$ changes accuracy, not $\varepsilon$, because the noise scales with it. A paper that reports $C$ in place of $\sigma$, $q$ and $T$ has not said how private the model is.
- **Report $\varepsilon$ with $\delta$.** The same run is $(5.89, 10^{-3})$, $(7.43, 10^{-5})$ and $(8.71, 10^{-7})$ under Mironov's conversion. An $\varepsilon$ without its $\delta$ means nothing, and $\delta$ should be well below $1/n = 10^{-4}$. A mechanism that published one random record in full would satisfy $\delta = 1/n$, so a larger $\delta$ allows a real leak.
- **Count every run, including tuning.** Each training run that looks at the private data spends budget, including the runs you discard. Choosing the best of several runs by a score computed on private data leaks more than one run does (@sec-hpt-leak, and @sec-apx-learn-tuning for the derivation).

## Code

The script `apx_core_dp_sgd.py` computes every number on this page and draws its figures. It rewrites the accountant in a few dozen lines rather than importing the chapters' code, so it is an independent check. It clips random gradients to test the sensitivity, solves the exact Gaussian curve with logarithms, checks amplification on a Laplace count, computes the subsampled Gaussian's curve by the binomial sum and by numerical averaging, and converts it to $\varepsilon$ and back to $\sigma$. From the repository root, run:

```bash
uv run python code/appendix-derivations/apx_core_dp_sgd.py
```

The script is [code/appendix-derivations/apx_core_dp_sgd.py](../../code/appendix-derivations/apx_core_dp_sgd.py), and [tests/test_appendix_core_dp_sgd.py](../../tests/test_appendix_core_dp_sgd.py) checks that its output still matches this page. @tbl-apx-sgd-code summarizes the output with the default seed.

| Check | Result |
| --- | --- |
| Clipped sum, 20,000 random datasets | Largest change: $1.0000\,C$ (add/remove), $2.0000\,C$ (replace-one) |
| Unsampled Gaussian steps, $\sigma = 1$ | $\varepsilon = 4.38$, 91.82 and 5425.51 after 1, 100 and 10,000 steps |
| Full batch, $\sigma = 10.8$, 100 steps | $\varepsilon = 4.005$, the chapter's target of 4 |
| Amplification, Laplace count, $\varepsilon = 1$, $q = 0.01$ | Largest log ratio 0.017037, equal to the formula; other direction 0.006341 |
| Subsampled Gaussian, $q = 0.01$, $\sigma = 1$ | Binomial sum and numerical average agree to seven digits; remove direction larger |
| Small-$q$ form | Within 30% up to $\alpha = 8$; 2246 times too small at $\alpha = 16$ |
| Running example, 100 epochs | $\varepsilon = 7.43$ (Mironov, $\alpha = 4.3$), 6.71 (improved) |
| Noise for $\varepsilon = 1$ | $\sigma = 4.974$ (Mironov), 4.126 (improved) |
| Fixed-size batches | $\varepsilon = 47.42$ against 6.71 for Poisson |

: The script's output with the default seed. {#tbl-apx-sgd-code}

The main thing to take from it is how much each idea is worth on the running example. Clipping and the noise make one step cost $\varepsilon = 4.38$, so 10,000 steps without sampling cost 5425.51. Sampling, measured by the Rényi curve and converted once, brings the same run down to 7.43, and the improved conversion to 6.71. Two independent ways of computing the curve agree to seven digits, so the remaining gap to the PLD accountant comes from the conversion and from keeping only a curve, not from numerical error.

## References

- Abadi, Chu, Goodfellow, McMahan, Mironov, Talwar & Zhang, "Deep Learning with Differential Privacy" (CCS 2016). DP-SGD with per-example clipping, and the moments accountant, which is the RDP accountant of @sec-apx-sgd-accounting in another notation.
- Mironov, "Rényi Differential Privacy" (CSF 2017). The definition of RDP, its composition and the conversion of @eq-apx-sgd-convert.
- Mironov, Talwar & Zhang, "Rényi Differential Privacy of the Sampled Gaussian Mechanism" (arXiv 2019). The reduction to one dimension, the dominance of the remove direction and the binomial sum of @eq-apx-sgd-rdp.
- Balle, Barthe & Gaboardi, "Privacy Amplification by Subsampling: Tight Analyses via Couplings and Divergences" (NeurIPS 2018). The tight amplification of @eq-apx-sgd-amplify, with $\delta$, and the other sampling schemes.
- Balle, Barthe, Gaboardi, Hsu & Sato, "Hypothesis Testing Interpretations and Rényi Differential Privacy" (AISTATS 2020). The improved conversion behind the 6.71.
- Ponomareva et al., "How to DP-fy ML: A Practical Guide to Machine Learning with Differential Privacy" (JAIR 2023). What to clip, how to choose the knobs and what to report, for practitioners.
