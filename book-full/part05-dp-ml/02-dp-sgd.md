# DP-SGD

[Private empirical risk minimization](01-dp-erm.md) ended with one method that still works for non-convex models: gradient descent with Gaussian noise on every step. What exactly does that method compute, why is it private for any model, and what does its privacy cost in accuracy? The method is stochastic gradient descent (SGD) with per-example clipping and Gaussian noise. Abadi et al. (2016) introduced it to give deep learning a guarantee of differential privacy (DP), under the name DP-SGD, and nearly every private deep-learning system runs it, through libraries such as Opacus, TensorFlow Privacy and JAX Privacy.

The one idea is that the privacy analysis never looks inside the model. Each step releases a noisy sum of clipped per-example gradients. Clipping rescales each record's gradient to length at most $C$, the clipping norm, so adding or removing one record changes the sum by at most $C$, whatever the network, the loss or the current parameters. The privacy proof needs nothing more, so the knobs fall into two groups. The noise multiplier $\sigma$, the noise's standard deviation in units of $C$, sets the privacy parameter $\varepsilon$ through an accountant, together with the sampling rate and the number of steps. The clipping norm and the learning rate leave $\varepsilon$ the same, and affect only accuracy. Privacy costs accuracy in two ways: a **bias** from clipping, which changes the direction of the average gradient, and a **variance** from the noise, which makes each step partly random.

The running example is a configuration whose privacy Part 4 computed without training anything: a sampling rate of 0.01, a noise multiplier of 1 and 10,000 steps. [Amplification by subsampling](../part04-privacy-accounting/05-subsampling.md) reports $\varepsilon = 6.71$ at $\delta = 10^{-5}$ for it, and this chapter uses it to train a network. Inputs are 50-dimensional standard normal vectors. Each label is the sign of the output of a fixed random **teacher** network with 8 hidden tanh units, so a perfect classifier exists. The **student** has one hidden layer of 64 tanh units and a logistic loss, with $d = 3{,}329$ parameters including biases. It trains on $n = 10{,}000$ records and is tested on 20,000 more, with a clipping norm of 0.5 and a learning rate of 0.5. Each step draws on average 100 records, so the 10,000 steps make 100 epochs. The dataset is fixed, and only the training randomness changes between runs. @tbl-dpsgd-symbols lists the symbols of the chapter.

| Symbol | Meaning |
| --- | --- |
| $D = (z_1, \dots, z_n)$ | The training set: $n$ records $z_i = (x_i, y_i)$ |
| $\theta \in \mathbb R^d$ | The model's $d$ parameters; $\theta_t$ after step $t$ |
| $\ell(\theta; z)$ | The loss of the parameters $\theta$ on the record $z$ |
| $g_i = \nabla_\theta\ell(\theta; z_i)$ | The per-example gradient of record $i$; $\bar g_i$ after clipping |
| $C$ | The clipping norm |
| $\sigma$ | The noise multiplier: the noise standard deviation in units of $C$ |
| $q$; $B = qn$ | The sampling rate; the expected batch size |
| $T$; $\eta$ | The number of steps; the learning rate |
| $S_t$; $G_t$ | The batch of step $t$; its noisy sum of clipped gradients |
| $\varepsilon_{q,\sigma}(\alpha)$ | The Rényi DP curve of one step at order $\alpha$ |

: The symbols of this chapter. {#tbl-dpsgd-symbols}

## The algorithm and its guarantee {#sec-dpsgd-algorithm}

What exactly does DP-SGD compute, and which $(\varepsilon, \delta)$ does it satisfy? The answer must be precise, because each detail of the algorithm is a hypothesis of the proof. An implementation that changes a detail can lose the guarantee with no visible symptom. Two datasets are neighbours here when one is the other with one record added or removed. Write $D \sim D'$ for such a pair, with $D' = D \cup \{z\}$ the dataset $D$ with the record $z$ added. The previous chapter used replacement instead. Add/remove suits DP-SGD because an independent coin for each record draws its batches, so adding a record just adds one more coin.

The algorithm changes three things in the noisy gradient descent of the previous chapter. Each step uses a random batch instead of the whole dataset, it clips each record's gradient, and it scales the noise to the clipping norm.

::: {#def-dpsgd-algorithm name="DP-SGD"}
Fix in advance the clipping norm $C > 0$, the noise multiplier $\sigma > 0$ and the sampling rate $q \in (0, 1]$, as well as the number of steps $T$ and the learning rate $\eta$. Write $B = qn$ for the expected batch size. Start from initial parameters $\theta_0$ chosen independently of the data. For each step $t = 0, \dots, T - 1$:

1. *Sample.* Put each record in the batch $S_t$ independently with probability $q$. This is **Poisson sampling**.
2. *Per-example gradients.* For each record $i$ of the batch, compute its **per-example gradient** $g_i = \nabla_\theta\ell(\theta_t; z_i)$ from the record $z_i$ and $\theta_t$ alone.
3. *Clip.* Rescale each gradient to length at most $C$,
$$\bar g_i = g_i\min\Big(1, \frac{C}{\|g_i\|}\Big).$$
4. *Noisy sum.* Add fresh Gaussian noise $\xi_t$, with standard deviation $\sigma C$ in every coordinate, to the clipped sum, and call the result $G_t$:
$$G_t = \sum_{i \in S_t}\bar g_i + \xi_t, \qquad \xi_t \sim \mathcal N(0, \sigma^2C^2I_d).$$
5. *Step.* Set $\theta_{t+1} = \theta_t - \eta\,G_t/B$. The divisor is the expected batch size, a public number, not the size of the batch actually drawn.

The output is the sequence of parameters $\theta_1, \dots, \theta_T$.
:::

The privacy proof needs one fact about step 4: how far one record can move the clipped sum. The answer is why the guarantee ignores the model.

::: {#lem-dpsgd-sensitivity name="The clipped sum has sensitivity C"}
Fix the parameters $\theta_t$ and the set of other records drawn into the batch. If a record $z$ is added to the batch, the clipped sum $\sum_{i \in S_t}\bar g_i$ changes by a vector of length at most $C$, whatever the model, the loss and $\theta_t$.
:::

::: {.proof}
1. *The other terms do not move.* Each term $\bar g_i$ is computed from its own record and $\theta_t$ alone, by step 2 of @def-dpsgd-algorithm, so every term of the other records is the same with or without $z$.
2. *The new term is short.* The only change is the extra term $\bar g_z$. Clipping multiplies the length of $g_z$ by $\min(1, C/\|g_z\|)$, so the new term's length is
$$\|\bar g_z\| = \min(\|g_z\|, C) \le C.$$
:::

With the sensitivity fixed at $C$ and the noise at $\sigma C$, a step that draws $z$ is a Gaussian mechanism whose shift, in noise standard deviations, is $C/(\sigma C) = 1/\sigma$. Its privacy depends on that ratio alone ([The Laplace and Gaussian mechanisms](../part03-dp-foundations/03-laplace-gaussian.md), @thm-lg-analytic), so the clipping norm cancels. A record whose clipped gradient is shorter than $C$ is no worse. Its pair of outputs is a post-processing of the pair with shift $C$, by adding extra noise, as in step 4 of the proof of @thm-lg-analytic.

Poisson sampling then makes the step's output on $D'$ a mixture. With probability $1 - q$ the record $z$ is left out, and the output has the same distribution as on $D$. With probability $q$ it is drawn, and the output is shifted. [Amplification by subsampling](../part04-privacy-accounting/05-subsampling.md#sec-sub-rdp-gaussian) computes the Rényi differential privacy (RDP) curve of that mixture against the unshifted Gaussian. That curve is the Rényi divergence between the two distributions as a function of its order $\alpha > 1$, which the accountant chooses later. At order $\alpha$ the divergence is the logarithm of the $\alpha$-th moment of the ratio of the two densities, divided by $\alpha - 1$. For integer orders the mixture's ratio is $1 - q$ plus $q$ times a Gaussian ratio. So its $\alpha$-th power expands by the binomial theorem, with one term for each number $j$ of factors that take the shifted component:
$$\varepsilon_{q,\sigma}(\alpha) = \frac{1}{\alpha - 1}\ln\sum_{j=0}^\alpha\binom\alpha j(1 - q)^{\alpha - j}q^je^{(j^2 - j)/(2\sigma^2)},$$ {#eq-dpsgd-rdp}
A numerical integral gives the other orders. Abadi et al. (2016) introduced this curve as the moments accountant.

The accountant tracks a curve rather than one $(\varepsilon, \delta)$ pair because curves add under composition. A pair only bounds the probability that the privacy loss exceeds $\varepsilon$, and says nothing about how large the loss is on that rare event, so pairs compose loosely. @thm-sub-dpsgd of [Amplification by subsampling](../part04-privacy-accounting/05-subsampling.md) composes the curves of the $T$ steps and converts the total once, with the conversion of Balle et al. (2020). It shows that releasing all of $\theta_1, \dots, \theta_T$ is $(\varepsilon, \delta)$-DP for add/remove neighbours, for every $0 < \delta < 1$, with
$$\varepsilon = \min_{\alpha > 1}\Big[T\varepsilon_{q,\sigma}(\alpha) + \ln\frac{\alpha - 1}\alpha - \frac{\ln\delta + \ln\alpha}{\alpha - 1}\Big].$$ {#eq-dpsgd-convert}
The parameter update is post-processing, because it uses $\theta_t$, $G_t$ and the public $B$ alone. @sec-apx-sgd-accounting derives the same result from scratch, with worked numbers. The guarantee does not depend on the loss, the model, $C$ or $\eta$, but it holds only under three conditions:

1. each per-example gradient depends only on its own record and the current parameters;
2. the initial parameters $\theta_0$ and the values of $C$, $\sigma$, $q$, $T$ and the denominator $B$ are fixed before the data are read;
3. every batch is drawn by Poisson sampling.

Each condition is a condition on the code, and each has a common way to fail. Batch normalization computes means and variances over the batch, so one record changes every other record's gradient, and the sum can move by much more than $C$. So private models use normalizations that act on one example at a time, such as group or layer normalization. To choose $C$ as the median gradient norm of the current batch makes the noise depend on the data, so $C$ must come from public data or a private estimate. To divide by the realized batch size $|S_t|$ instead of $B$ also breaks the proof, because $|S_t|$ changes when $z$ is added. Batches cut from a shuffled dataset are a different mechanism, which can be much less private than the Poisson accountant reports ([Amplification by subsampling](../part04-privacy-accounting/05-subsampling.md#sec-sub-schemes)). Finally, the accountant must see every step, including the steps of runs discarded during tuning ([Hyperparameter tuning](03-hyperparameter-tuning.md)).

The guarantee is for one record. A person with many records needs [user-level privacy](07-user-level.md). Preprocessing that uses the data, such as normalizing features by their mean, must itself be private or use public statistics.

::: {#exm-dpsgd-epsilon}
For the running example's configuration, @eq-dpsgd-convert gives $\varepsilon = 6.71$ at $\delta = 10^{-5}$, the same value Part 4 reports. The best order is $\alpha = 4.1$. @tbl-dpsgd-sigma gives the noise multiplier that the same $q$ and $T$ need for other budgets.

| $\varepsilon$ at $\delta = 10^{-5}$ | 0.5 | 1 | 2 | 4 | 6.71 | 8 |
| --- | --- | --- | --- | --- | --- | --- |
| Noise multiplier $\sigma$ | 7.72 | 4.13 | 2.28 | 1.36 | 1 | 0.917 |

: The noise multiplier that DP-SGD with $q = 0.01$ and $T = 10{,}000$ steps needs for each budget, from the RDP accountant. {#tbl-dpsgd-sigma}

How large is the noise? The averaged update $G_t/B$ carries noise with standard deviation $\sigma C/B = 1 \times 0.5/100 = 0.005$ on each of the 3,329 coordinates. A Gaussian vector with $d$ independent coordinates has length close to $\sqrt d$ times their standard deviation, so the noise has length about $0.005 \times \sqrt{3{,}329} = 0.288$. The signal is the averaged clipped gradient, whose length is at most $C = 0.5$. In the run its length is 0.070 at the first step, and between 0.015 and 0.027 from step 1,000 on. So from then on the noise is about ten times longer than the signal.

Training works anyway, because the signal points in roughly the same direction from one step to the next, while the noise is independent. Independent vectors add in squares, so over $t$ steps the signals add up to about $t$ times their length, and the noise to only about $\sqrt t$ times its length. With a typical signal length of 0.026, signal overtakes noise when $t \cdot 0.026 > \sqrt t \cdot 0.288$, that is after $t > (0.288/0.026)^2 \approx 123$ steps. The final model reaches test accuracy 0.9589, against 0.9627 for non-private SGD.
:::

## Clipping: bias, noise and the choice of C {#sec-dpsgd-clipping}

The clipping norm does not affect privacy, but it controls both forms of cost. A small $C$ clips more gradients, which biases the update, and a large $C$ adds more noise, because the noise is $\sigma C$. To choose $C$ you need to know what the update estimates on average, and how a run changes when $C$ changes. On average over the sampling and the noise, each record enters the update with a weight that clipping sets, and the denominator $B$ makes the sampling rate cancel.

::: {#prp-dpsgd-expected name="The expected update"}
In DP-SGD (@def-dpsgd-algorithm), with $\theta_t$ fixed, the averaged update is an unbiased estimate of the average clipped gradient,
$$\mathbb E\Big[\frac{G_t}{B}\Big] = \frac1n\sum_{i=1}^n\min\Big(1, \frac{C}{\|g_i\|}\Big)g_i,$$ {#eq-dpsgd-expected}
and its noise has standard deviation $\sigma C/B$ in each coordinate.
:::

::: {.proof}
1. *Average over the sampling.* Each record is in the batch with probability $q$, so the expected clipped sum is $\sum_{i=1}^n q\,\bar g_i$. The noise has mean zero.
2. *Divide by $B = qn$.* The $q$ cancels and leaves $\frac1n\sum_i\bar g_i$, which is @eq-dpsgd-expected by the definition of $\bar g_i$.
3. *The noise.* The noise $\xi_t$ has standard deviation $\sigma C$ in each coordinate, and dividing by $B$ divides it by $B$.
:::

So the update follows the gradient not of the training loss but of a reweighted loss. Each record's weight $\min(1, C/\|g_i\|)$ falls as its gradient grows. So records the model fits badly, such as rare examples, members of small groups or mislabelled points, get less weight, while well-fitted records keep their full weight. The weights change during training, and [Disparate impact](08-disparate-impact.md) measures who pays for them. To divide by $|S_t|$ instead of $B$ would not give this average, because the expectation of a ratio of random quantities is not the ratio of their expectations.

When $C$ is below every gradient norm, clipping turns each gradient into $C$ times a unit vector, and the clipping norm then has the same role as the learning rate.

::: {#prp-dpsgd-scale name="Scale invariance"}
Suppose that every per-example gradient of a run is clipped, that is $\|g_i\| \ge C$ at every step. Write the noise as $\xi_t = \sigma C\zeta_t$ with $\zeta_t \sim \mathcal N(0, I_d)$. Then the update is
$$\theta_{t+1} = \theta_t - \frac{\eta C}{B}\Big(\sum_{i\in S_t}\frac{g_i}{\|g_i\|} + \sigma\zeta_t\Big),$$ {#eq-dpsgd-normalized}
which depends on $\eta$ and $C$ only through their product $\eta C$. So two runs with the same $\eta C$ and the same seed, in both of which every gradient is clipped, produce the same parameters.
:::

::: {.proof}
1. *Clipped gradients are scaled unit vectors.* If $\|g_i\| \ge C$, then $\min(1, C/\|g_i\|) = C/\|g_i\|$, so $\bar g_i = Cg_i/\|g_i\|$.
2. *Factor out $C$.* Every term of the noisy sum now carries the factor $C$, so $G_t = C\big(\sum_{i \in S_t}g_i/\|g_i\| + \sigma\zeta_t\big)$.
3. *Substitute into the step.* Step 5 of @def-dpsgd-algorithm subtracts $\eta G_t/B$, which gives @eq-dpsgd-normalized. The batch, the unit vectors and $\zeta_t$ involve neither $\eta$ nor $C$, so the run depends on them only through $\eta C$.
:::

So below the gradient norms, DP-SGD is normalized SGD, which averages unit vectors, plus noise. A lower $C$ changes nothing once $\eta$ grows to match. This is why practitioners often clip aggressively and tune only the learning rate, and why De et al. (2022) find that small clipping norms work well. With the Adam optimizer, $C$ drops out of the step length altogether, because Adam divides each coordinate of the update by its own running scale, which carries the same factor $C$.

Above the gradient norms, noise grows and signal does not. Once $C$ exceeds most gradient norms, the clipped average stops growing with $C$, but the noise $\sigma C/B$ keeps growing in proportion. So the signal-to-noise ratio falls like $1/C$, and a $C$ that clips almost nothing only adds noise.

::: {#exm-dpsgd-clip}
Gradient norms change during training, so a fixed $C$ visits both regimes. In the DP-SGD run of the running example the median per-example gradient norm is 3.592 at the first step and 0.006 at step 1,000, and it rounds to 0.000 from step 3,000 on. Non-private training behaves the same way (@fig-dpsgd-clipping, right). The reason is that the logistic loss flattens on records classified with a wide margin, so their gradients vanish. With $C = 0.5$, every gradient is clipped at the start, and later only 9% to 25% of the batch are, the few records still near the decision boundary. So a fixed $C$ is in the scale-invariant regime early, and later above most gradient norms, where clipping changes little.

@tbl-dpsgd-clip trains the running example at $\varepsilon = 6.71$ with seven clipping norms, once with the learning rate fixed at 0.5 and once with $\eta C$ fixed at $0.5 \times 0.5 = 0.25$.

| $C$ | 0.01 | 0.03 | 0.1 | 0.3 | 1 | 3 | 10 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Fixed $\eta = 0.5$ | 0.9244 | 0.9510 | 0.9567 | 0.9607 | 0.9583 | 0.9523 | 0.8347 |
| Fixed $\eta C = 0.25$ | 0.9627 | 0.9623 | 0.9605 | 0.9599 | 0.9590 | 0.9558 | 0.9494 |

: Test accuracy of the running example at $\sigma = 1$ for seven clipping norms, one run each. With $\eta C$ fixed, small clipping norms all give the same accuracy. {#tbl-dpsgd-clip}

With the learning rate fixed, a small $C$ is too slow, with 0.9244 at $C = 0.01$, because every step has length proportional to $\eta C = 0.005$. A large $C$ is too noisy, with 0.8347 at $C = 10$. With $\eta C$ fixed, accuracy is flat for $C \le 0.3$, as scale invariance predicts. It falls slowly for large $C$, where the noise grows while the clipped gradients do not. The best accuracy, 0.9627 at $C = 0.01$ with a learning rate of 25, equals the non-private model's.
:::

![Left: test accuracy of the running example at $\sigma = 1$ ($\varepsilon = 6.71$) against the clipping norm, with the learning rate fixed at $\eta = 0.5$ (circles) and with $\eta C$ fixed at 0.25 (squares). Right: the median per-example gradient norm during non-private training, on a log scale. It falls by about five orders of magnitude.](figures/dpsgd_clipping.svg){#fig-dpsgd-clipping}

## Batch size and the number of steps at a fixed budget {#sec-dpsgd-batch}

Suppose the privacy budget $(\varepsilon, \delta)$ and the compute budget, a number of epochs $E$, are both fixed. Each epoch is $1/q$ steps, so $T = E/q$. Should you take many small steps or few large ones? The accountant says how much noise each choice needs, and the noise on the averaged gradient decides how useful each step is.

For small $q$, the terms $j = 0$ and $j = 1$ of @eq-dpsgd-rdp cost nothing. Their exponent $j^2 - j$ is zero, and with every exponential set to 1 the sum is the binomial expansion of $\big((1 - q) + q\big)^\alpha = 1$, whose logarithm is zero. Of the other terms, the term $j = 2$, which carries $q^2$, is the largest ([Amplification by subsampling](../part04-privacy-accounting/05-subsampling.md#sec-sub-small-q) derives the approximation), and it is enough to show the trade.

::: {#prp-dpsgd-batch name="Batch size at a fixed budget"}
Assume that the per-step RDP curve is approximately
$$\varepsilon_{q,\sigma}(\alpha) \approx \frac\alpha2q^2\big(e^{1/\sigma^2} - 1\big),$$
as it is for small $q$ and orders that are not too large. Fix $\varepsilon$, $\delta$ and the number of epochs $E$, so that $T = E/q$. Where $\sigma$ is well above 1, keeping $\varepsilon$ fixed as $q$ changes requires
$$\sigma \propto \sqrt q, \qquad \text{so} \qquad \frac{\sigma C}{B} \propto \frac{C}{\sqrt{qn}\,\sqrt n} = \frac{C}{\sqrt{Bn}},$$ {#eq-dpsgd-batch}
and the noise on the averaged gradient falls like $1/\sqrt B$. Where $\sigma$ is near or below 1, the per-step cost is larger by the factor $\sigma^2(e^{1/\sigma^2} - 1) > 1$, so small batches need more noise than @eq-dpsgd-batch says.
:::

::: {.proof}
The plan is to approximate the per-step cost and add it over all the steps. The total gives the noise multiplier that keeps $\varepsilon$ fixed, and the last step measures what the approximation leaves out.

1. *Per-step cost.* For $\sigma$ well above 1, the exponent $1/\sigma^2$ is small and $e^{1/\sigma^2} - 1 \approx 1/\sigma^2$. So the per-step curve is about $\frac{\alpha q^2}{2\sigma^2}$.
2. *Total cost.* Curves add under composition, so $T = E/q$ steps have the curve
$$\frac{E}{q}\cdot\frac{\alpha q^2}{2\sigma^2} = \frac{\alpha Eq}{2\sigma^2}.$$
The conversion @eq-dpsgd-convert depends on the run only through this curve, so $\varepsilon$ stays fixed when $q/\sigma^2$ does, that is when $\sigma \propto \sqrt q$.
3. *Noise per averaged gradient.* The noise on the averaged gradient is $\sigma C/B$ with $B = qn$. With $\sigma \propto \sqrt q$ and $n$ fixed it is proportional to $\sqrt q/q = 1/\sqrt q$, and since $q = B/n$ that is @eq-dpsgd-batch.
4. *Small $\sigma$.* The exponential grows faster than its tangent at 0, so $e^{1/\sigma^2} - 1$ exceeds the approximation $1/\sigma^2$ of step 1. The ratio of the two is the factor $\sigma^2(e^{1/\sigma^2} - 1)$, which is above 1 and grows as $\sigma$ falls. By step 2, small batches need small $\sigma$, so they are the ones that pay this factor, as extra noise beyond the $\sqrt q$ law.
:::

The accountant's numbers, in @tbl-dpsgd-batch of the example below, show that amplification is most efficient at large $\sigma$. The ratio $\sigma/\sqrt q$, which @eq-dpsgd-batch says is constant, falls from 22.3 at $B = 10$ to 10.8 at $B = 10{,}000$. Over the same range the factor $\sigma^2(e^{1/\sigma^2} - 1)$ falls from 3.23 to 1, so the small batches pay most of the extra noise through it. The rest comes from the high orders that the conversion uses, where the approximation of @prp-dpsgd-batch fails and sampling buys little.

Large batches help, but not for free. For a fixed number of epochs they give less noise per step and fewer steps, and fewer steps make less progress unless the learning rate grows in proportion. In @tbl-dpsgd-batch-train, below, the best learning rate grows from 0.25 at $B = 30$ to between 1 and 2 at $B = 100$ to $B = 1{,}000$. De et al. (2022) and Anil et al. (2022) train with batches of thousands to hundreds of thousands, and report that large batches are a main ingredient of accurate private models.

![DP-SGD at $\varepsilon = 4$, $\delta = 10^{-5}$, for 100 epochs on $n = 10{,}000$ records, as a function of the expected batch size. Left: the noise multiplier the accountant requires, against the $\sqrt B$ law of @eq-dpsgd-batch ($\sigma \propto \sqrt q$ at fixed $n$). Right: the noise standard deviation on the averaged gradient, in units of $C$, against the $1/\sqrt B$ law. Small batches need more noise than the law predicts.](figures/dpsgd_batch_size.svg){#fig-dpsgd-batch}

::: {#exm-dpsgd-batch}
@tbl-dpsgd-batch fixes $\varepsilon = 4$, $\delta = 10^{-5}$ and 100 epochs on $n = 10{,}000$ records, and finds the noise multiplier for each sampling rate. Full batches, $q = 1$, have no sampling, so the 100 Gaussian steps compose exactly into a single Gaussian mechanism with ratio $\sqrt{100}/\sigma$ ([f-DP and Gaussian DP](../part04-privacy-accounting/04-gaussian-dp.md)).

| $q$ | $B$ | Steps $T$ | $\sigma$ | Noise per averaged gradient $\sigma/B$ | $\sigma/\sqrt q$ | $\sigma^2(e^{1/\sigma^2} - 1)$ |
| --- | --- | --- | --- | --- | --- | --- |
| 0.001 | 10 | 100,000 | 0.704 | 0.0704 | 22.3 | 3.23 |
| 0.003 | 30 | 33,333 | 0.909 | 0.0303 | 16.6 | 1.95 |
| 0.01 | 100 | 10,000 | 1.36 | 0.0136 | 13.6 | 1.33 |
| 0.03 | 300 | 3,333 | 2.16 | 0.0072 | 12.5 | 1.12 |
| 0.1 | 1,000 | 1,000 | 3.78 | 0.00378 | 12 | 1.04 |
| 0.3 | 3,000 | 333 | 6.44 | 0.00215 | 11.8 | 1.01 |
| 1 | 10,000 | 100 | 10.8 | 0.00108 | 10.8 | 1 |

: Noise needed for $\varepsilon = 4$ at $\delta = 10^{-5}$ over 100 epochs, for seven sampling rates. The noise on the averaged gradient, in units of $C$, falls faster than $1/\sqrt B$. {#tbl-dpsgd-batch}

From $B = 10$ to $B = 1{,}000$, a hundredfold increase, the noise per averaged gradient falls from 0.0704 to 0.00378. That is a factor of 18.6, against the factor of 10 that $1/\sqrt B$ predicts. @tbl-dpsgd-batch-train trains four of these configurations with three learning rates each.

| $B$ | Learning rates and test accuracy |
| --- | --- |
| 30 | $\eta = 0.25$: 0.9544; 0.5: 0.9487; 1: 0.9453 |
| 100 | $\eta = 0.5$: 0.9522; 1: 0.9569; 2: 0.9465 |
| 300 | $\eta = 1$: 0.9591; 2: 0.9565; 4: 0.9577 |
| 1,000 | $\eta = 2$: 0.9589; 4: 0.9565; 8: 0.9576 |

: Test accuracy of the running example at $\varepsilon = 4$ and 100 epochs, for four batch sizes and three learning rates each, one run each. {#tbl-dpsgd-batch-train}

At its best learning rate, each batch size reaches between 0.9544 and 0.9591. The two largest batches, 300 and 1,000, do best, with 30 and 100 times fewer steps than $B = 10$ would take. To choose the best of three learning rates by test accuracy is itself a leak. This example does it only to compare batch sizes, and [Hyperparameter tuning](03-hyperparameter-tuning.md) shows how to do it privately.
:::

## Per-example gradient norms without per-example gradients {#sec-dpsgd-ghost}

Ordinary backpropagation computes the gradient of the *sum* of the batch's losses, never the gradient of each record. Clipping needs each record's gradient norm, and all $B$ per-example gradients cost $B$ times the memory of the model's gradient, too much for large models. Can you get the norms, and the clipped sum, more cheaply?

The tool is the shape of a dense layer's gradient. Take a dense layer that maps an input $a$ to $Wa$, and write $v$ for the gradient of the loss with respect to the layer's output, the **back-propagated error**. The gradient with respect to the weights $W$ is then the outer product $va^\top$ ([Gradient inversion](../part02-attacks/06-gradient-inversion.md), @prp-gradinv-outer). That chapter uses the outer product to reconstruct inputs from gradients, and here it makes per-example norms cheap.

::: {#prp-dpsgd-ghost name="Ghost norms"}
Consider a dense layer, and for record $i$ write $a_i$ for its input to the layer and $v_i$ for its back-propagated error, so that the record's gradient for the layer is $v_ia_i^\top$. Then
$$\big\|v_ia_i^\top\big\|_F = \|v_i\|\,\|a_i\|,$$ {#eq-dpsgd-ghost}
where the Frobenius norm $\|\cdot\|_F$ is the square root of the sum of squared entries. The whole network's per-example gradient norm is the square root of the sum of these squared norms over the layers. Write $c_i$ for the clipping factor of record $i$,
$$c_i = \min\Big(1, \frac{C}{\|g_i\|}\Big).$$
Stack the errors as the rows of a matrix $V$ and the inputs as the rows of a matrix $A$. If $c \circ V$ is $V$ with row $i$ multiplied by $c_i$, the clipped sum for the layer is $\sum_ic_iv_ia_i^\top = (c \circ V)^\top A$. Now let the layer act on a sequence of $N$ tokens, with input $a_{is}$ and error $v_{is}$ at token $s$. Record $i$'s gradient is then $\sum_sv_{is}a_{is}^\top$, and
$$\Big\|\sum_sv_{is}a_{is}^\top\Big\|_F^2 = \sum_{s, s'}(a_{is}\cdot a_{is'})(v_{is}\cdot v_{is'}).$$ {#eq-dpsgd-ghost-seq}
:::

::: {.proof}
The plan is to compute the norm of one outer product, add over layers, write the clipped sum as one matrix product, and expand the square for sequences.

1. *One token.* The entry $(j, k)$ of $va^\top$ is $v_ja_k$, so the squared Frobenius norm factors into two sums,
$$\sum_{j,k}v_j^2a_k^2 = \Big(\sum_jv_j^2\Big)\Big(\sum_ka_k^2\Big) = \|v\|^2\|a\|^2.$$
2. *Layers.* The full gradient is the concatenation of the layers' gradients, so its squared length is the sum of theirs.
3. *The clipped sum.* The sum $\sum_ic_iv_ia_i^\top$ is linear in the errors, so it is the usual batch product of errors and inputs with each record's error scaled by $c_i$. One backward pass computes the errors, the norms come from @eq-dpsgd-ghost, and one more matrix product per layer gives the clipped sum.
4. *Sequences.* Drop the record index $i$. The squared Frobenius norm of a matrix $M$ is the trace of $M^\top M$, so the square of a sum of outer products expands into one trace per pair of tokens,
$$\Big\|\sum_sv_sa_s^\top\Big\|_F^2 = \sum_{s, s'}\operatorname{tr}\big(a_sv_s^\top v_{s'}a_{s'}^\top\big).$$
Inside each trace, $v_s^\top v_{s'}$ is the number $v_s\cdot v_{s'}$, and what remains is $\operatorname{tr}(a_sa_{s'}^\top) = a_s\cdot a_{s'}$. That gives @eq-dpsgd-ghost-seq, the entrywise product of two $N \times N$ Gram matrices, summed.
:::

This is **ghost clipping**, in which the per-example gradients are "ghosts" that are never formed (Goodfellow 2015, Lee & Kifer 2021, and Li et al. 2022 for sequences). For one token per record, ghost clipping stores $B(d_{\text{in}} + d_{\text{out}})$ numbers per layer instead of $Bd_{\text{in}}d_{\text{out}}$, where $d_{\text{in}}$ and $d_{\text{out}}$ are the layer's input and output widths. For sequences the Gram matrices cost $2BN^2$, which beats the explicit gradients only when $2N^2 < d_{\text{in}}d_{\text{out}}$. So long sequences in narrow layers favour explicit gradients, and Bu, Mao & Xu (2022) choose per layer. A convolutional layer's per-example gradient is a sum of outer products over spatial positions, so @eq-dpsgd-ghost-seq applies with positions in place of tokens. Ghost norms are exact computations of the same quantities, so the clipped sum, and with it the guarantee of @thm-sub-dpsgd, is unchanged.

::: {#exm-dpsgd-ghost}
For the 256 records of a test batch of the running example, the norms from @eq-dpsgd-ghost agree with explicitly formed per-example gradients to a relative difference of $2.2 \times 10^{-16}$, which is rounding error. For a random 16-token sequence, @eq-dpsgd-ghost-seq gives 33.144228, and the explicit gradient's norm is the same. @tbl-dpsgd-ghost compares the memory for several layers.

| Layer | Explicit $Bd_{\text{in}}d_{\text{out}}$ | Ghost | Ratio |
| --- | --- | --- | --- |
| Running example, first layer: $B = 100$, $51 \to 64$ (50 inputs and a bias) | 326,400 | 11,500 | 28.4 |
| $B = 32$, one token, $768 \to 768$ | 18,874,368 | 49,152 | 384.0 |
| $B = 32$, 512 tokens, $768 \to 768$ | 18,874,368 | 16,777,216 | 1.1 |
| $B = 32$, 512 tokens, $768 \to 3{,}072$ | 75,497,472 | 16,777,216 | 4.5 |
| $B = 8$, 4,096 tokens, $4{,}096 \to 4{,}096$ | 134,217,728 | 268,435,456 | 0.5 |

: Numbers stored to clip one dense layer, explicitly and with ghost norms. {#tbl-dpsgd-ghost}

:::

## Microbatches and vectorized per-example gradients {#sec-dpsgd-microbatch}

Deep-learning frameworks compute the gradient of a batch's summed loss in one parallel pass, and per-example clipping breaks that pattern. One record at a time, DP-SGD can run tens or hundreds of times slower than ordinary training. Ghost norms (@sec-dpsgd-ghost) are one remedy, and two others are common. Vectorization computes the per-example gradients in parallel after all, while **microbatching** clips groups of records instead of single records. Vectorization changes nothing about privacy or accuracy, so the question is what microbatching costs.

::: {#def-dpsgd-microbatch name="Microbatched DP-SGD"}
Choose a group size $m$, and fix in advance a rule that assigns every record to one of $B/m$ groups, for example a hash of a record identifier. A Poisson batch of expected size $B$ then splits into about $B/m$ **microbatches** of about $m$ records each. Microbatched DP-SGD runs @def-dpsgd-algorithm, but clips the mean gradient of each microbatch instead of each record's gradient. It adds Gaussian noise to the sum of the clipped means and divides the result by $B/m$.
:::

Under per-example clipping, the added record's term goes from absent to present, a change of at most $C$. A microbatch's clipped mean instead goes from one vector of length at most $C$ to another, and that change can be twice as long.

::: {#prp-dpsgd-microbatch name="What microbatches cost"}
Under add/remove neighbours, the sum of clipped microbatch means has sensitivity $2C$ as soon as a record can share its group with others. So the accountant's $\varepsilon$ for the same $q$, $\sigma$ and $T$ holds only with noise of standard deviation $2\sigma C$. The update's noise then has standard deviation
$$\frac{2\sigma C}{B/m} = 2m\cdot\frac{\sigma C}{B}$$ {#eq-dpsgd-microbatch}
per coordinate, $2m$ times that of per-example clipping.
:::

::: {.proof}
The plan is to locate the one microbatch a record can change, bound how far its clipped mean moves, and then scale the noise to match the accountant.

1. *One record touches one microbatch.* The assignment of records to groups is fixed before training and does not depend on which other records are present. A position in the dataset would not do, because adding a record shifts the positions of the records after it. So adding or removing $z$ changes only the microbatch of $z$'s group, and every other clipped mean is the same on both datasets.
2. *That clipped mean moves by up to $2C$.* If the group has other sampled records, its clipped mean changes from one vector of length at most $C$ to another. The two can point in opposite directions, so they differ by up to $2C$. Only when $z$ is alone in its group does the change shrink to $C$, the per-example case, and a worst-case bound must take $2C$.
3. *The accountant.* The accountant of @thm-sub-dpsgd assumes noise $\sigma$ times the sensitivity, so the noise must be $\sigma \cdot 2C$. The rest of its argument is unchanged.
4. *Per coordinate.* The update divides the noisy sum by the public number $B/m$, and dividing $2\sigma C$ by it gives @eq-dpsgd-microbatch.
:::

Microbatching trades accuracy for speed at a steep rate. A batch now has only $B/m$ clipped terms to carry signal instead of $B$, while the total noise has doubled to $2\sigma C$. A microbatch's mean gradient is also shorter than its records' gradients when they disagree, so clipping removes less of it, but that does not make up for a factor $2m$ in the noise. Some libraries offer microbatches as a speed option, so check what yours assumes. The reported $\varepsilon$ is valid only if its accounting uses sensitivity $2C$ and its microbatches are fixed in advance. If they are cut by position from the sampled batch, adding one record shifts every later record into a different microbatch, and step 1 fails.

Vectorization costs nothing, because it keeps per-example clipping. JAX's `vmap` vectorizes over the examples of a batch, and Opacus attaches hooks that record the per-example quantities during backpropagation. Both compute the same per-example gradients exactly, so they change neither the privacy nor the accuracy. Subramani, Vadivelu & Kamath (2021) show that vectorization with just-in-time compilation often brings DP-SGD's running time close to that of ordinary training.

::: {#exm-dpsgd-microbatch}
At the running example's $\sigma = 1$, $C = 0.5$ and $B = 100$, the noise on the averaged update is 0.0050 per coordinate with per-example clipping. With microbatches of $m = 4$ records it is 0.0400, eight times more, and with $m = 16$ it is 0.1600, thirty-two times more, at the same $\varepsilon = 6.71$. The test accuracy falls with it, from 0.9589 for per-example clipping (@exm-dpsgd-epsilon) to 0.9336 with microbatches of 4 and 0.7459 with microbatches of 16, one run each.
:::

## The running example: privacy against accuracy {#sec-dpsgd-run}

For the running example, how much accuracy does each budget cost, and where does the loss come from?

::: {#exm-dpsgd-run}
@tbl-dpsgd-run trains the running example with $q = 0.01$, $C = 0.5$, $\eta = 0.5$ and $T = 10{,}000$ steps, at each budget's noise multiplier from @tbl-dpsgd-sigma, three times each from a fresh random initialization. The non-private baseline uses the same sampling and learning rate with neither clipping nor noise. Its accuracy is at its best after 1,000 steps, 0.9627 in a single run, and it stays there, at 0.9617 after 10,000 steps. So the baseline runs 1,000 steps, and its three runs average 0.9610.

| | Noise multiplier | Test accuracy (mean ± standard deviation over 3 runs) |
| --- | --- | --- |
| $\varepsilon = 0.5$ | 7.72 | $0.8209 \pm 0.0180$ |
| $\varepsilon = 1$ | 4.13 | $0.9175 \pm 0.0161$ |
| $\varepsilon = 2$ | 2.28 | $0.9441 \pm 0.0051$ |
| $\varepsilon = 4$ | 1.36 | $0.9548 \pm 0.0019$ |
| $\varepsilon = 8$ | 0.917 | $0.9604 \pm 0.0019$ |
| Clipped, no noise | 0 | $0.9649 \pm 0.0009$ |
| Non-private SGD, 1,000 steps | — | $0.9610 \pm 0.0005$ |

: Test accuracy of the running example under DP-SGD at five budgets, with $\delta = 10^{-5}$. {#tbl-dpsgd-run}

Part 4 found that the running configuration has $\varepsilon = 6.71$. This example adds the other half: on this task that budget costs 0.4 points of accuracy, 0.9589 against 0.9627 in the single runs of @exm-dpsgd-epsilon. To halve $\varepsilon$ to about 3 needs $\sigma$ between 1.36 and 2.28, and by the table costs about one more point of accuracy, while $\varepsilon = 1$ costs four points. [Choosing ε](09-choosing-epsilon.md) asks which of these trades is worth making, and [Auditing](10-auditing.md) checks whether the trained model is as private as the accountant says.
:::

Clipping costs nothing here, and noise costs everything. Clipping without noise reaches 0.9649, slightly above non-private SGD, because on this problem clipping only limits the influence of the few records with large gradients. So the whole loss comes from the noise, and it is steeply non-linear in $\varepsilon$. From $\varepsilon = 8$ to 4 the accuracy drops by half a point, from 2 to 1 by 2.7 points, and from 1 to 0.5 by almost 10. The signal-against-noise argument of @exm-dpsgd-epsilon explains the collapse. At $\varepsilon = 1$ the noise multiplier is 4.13, so the noise on the averaged gradient has length about $4.13 \times 0.5/100 \times \sqrt{3{,}329} = 1.19$. If the signal stays near its length of 0.026 at $\sigma = 1$, signal overtakes noise only after about $(1.19/0.026)^2 \approx 2{,}100$ steps, so the 10,000 steps contain only a few independent useful moves. @fig-dpsgd-training shows the result: with $\sigma = 4$ the model reaches its accuracy early and then wanders.

Run-to-run variation grows with the noise. The standard deviation over runs is 0.0019 at $\varepsilon = 8$ and 0.018 at $\varepsilon = 0.5$. So a comparison of two configurations by one run each, as in @tbl-dpsgd-clip, is reliable only for differences well above that spread.

Architecture choices also change under privacy. The running example uses tanh units because Papernot, Thakurta, Song, Chien & Erlingsson (2021) find that bounded activations such as tanh train better than ReLU under DP-SGD. Model size changes too. The noise vector has length about $\sigma C\sqrt d/B$, which grows with the number of parameters, while the clipped signal cannot exceed $C$. So the best private model is often smaller than the best non-private one, unless pretraining on public data has already done most of the work ([Public data and private fine-tuning](06-public-data.md)). Larger models can still win when you tune the batch size and the clipping norm with them, as De et al. (2022) do with very large batches and small clipping norms.

![Test accuracy of DP-SGD on the running example at $\delta = 10^{-5}$ against $\varepsilon$ (log scale), with the non-private model dashed. Accuracy is close to non-private above $\varepsilon = 4$ and falls steeply below $\varepsilon = 1$.](figures/dpsgd_privacy_accuracy.svg){#fig-dpsgd-privacy-accuracy}

![Left: test accuracy during training with clipping alone, with $\sigma = 1$ ($\varepsilon = 6.71$) and with $\sigma = 4$. Right: the length of the averaged clipped gradient at each step (solid, log scale), and the length of the noise on it (dotted). The noise is about ten times longer than the signal at $\sigma = 1$, and about forty times at $\sigma = 4$.](figures/dpsgd_training.svg){#fig-dpsgd-training}

## Code {#sec-dpsgd-code}

The script [code/dp_sgd.py](../../code/part05-dp-ml/dp_sgd.py) implements the network and its per-example gradients by ghost norms. It also implements DP-SGD with Poisson sampling and optional microbatches, and the RDP accountant of Part 4, rewritten in a few dozen lines so the part stands on its own. Later chapters import its model and accountant: `make_task()`, `train_dpsgd(rng, data, config)`, `accuracy`, `clipped_grad_sum`, `dpsgd_epsilon(q, sigma, steps, delta)` and `noise_multiplier_for(q, steps, eps, delta)`. From the repository root:

```bash
uv run python code/part05-dp-ml/dp_sgd.py
```

| Experiment | Result |
| --- | --- |
| Accountant, $q = 0.01$, $\sigma = 1$, 10,000 steps | $\varepsilon = 6.71$ at $\alpha = 4.1$, as in Part 4 |
| Noise on the averaged gradient | 0.005 per coordinate, length about 0.288; signal 0.015 to 0.027 |
| Ghost norms | Agree with explicit gradients to $2.2 \times 10^{-16}$ |
| Microbatches of 4 and 16 | Noise 8 and 32 times larger; accuracy 0.9336 and 0.7459 |
| Running example, $\varepsilon = 6.71$ | 0.9589 against 0.9627 non-private |
| Clipping sweep | With $\eta C$ fixed, 0.9599 to 0.9627 for $C \le 0.3$ |
| Batch sweep, $\varepsilon = 4$ | Noise per averaged gradient falls 18.6 times from $B = 10$ to 1,000 |
| Privacy against accuracy | From 0.8209 at $\varepsilon = 0.5$ to 0.9604 at $\varepsilon = 8$ |

: The script's output with the default seed. {#tbl-dpsgd-code}

[tests/test_dp_sgd.py](../../tests/test_dp_sgd.py) checks that the output still matches.

## Summary {#sec-dpsgd-summary}

| Result | Statement | Use |
| --- | --- | --- |
| @def-dpsgd-algorithm | Poisson batches; clip each $g_i$ to length $C$; add $\mathcal N(0, \sigma^2C^2I)$ to the sum; step by $\eta G_t/B$ | Every chapter of this part; check an implementation for Poisson sampling, per-example gradients, no batch statistics, a public denominator and a fixed $C$ |
| @lem-dpsgd-sensitivity | One record moves the clipped sum by at most $C$ | Why the guarantee ignores the model |
| @thm-sub-dpsgd | RDP curve $T\varepsilon_{q,\sigma}(\alpha)$, converted by @eq-dpsgd-convert, under the three conditions of @sec-dpsgd-algorithm | Only $\sigma$, $q$ and $T$ set $\varepsilon$, so $C$ and $\eta$ are free for accuracy; the running example has $\varepsilon = 6.71$ at $\delta = 10^{-5}$ |
| @exm-dpsgd-epsilon | Noise $\sigma C/B$ per coordinate, length about $\sigma C\sqrt d/B$; signal overtakes noise after about $(\text{noise length}/\text{signal length})^2$ steps | Judging whether a configuration can learn at all |
| @prp-dpsgd-expected | $\mathbb E[G_t/B] = \frac1n\sum_i\min(1, C/\Vert g_i\Vert)g_i$ | Divide by $B$, not $\lvert S_t\rvert$; the reweighting is the source of [disparate impact](08-disparate-impact.md) |
| @prp-dpsgd-scale | With every gradient clipped, the run depends on $\eta C$ only | Clip small and tune $\eta C$: below the typical gradient norm, $C$ stops mattering |
| @prp-dpsgd-batch | $\sigma \propto \sqrt q$ and noise $\propto 1/\sqrt B$ at fixed $\varepsilon$ and epochs, for small $q$ and $\sigma$ well above 1 | Prefer large batches at a fixed number of epochs, and scale the learning rate up with them |
| @prp-dpsgd-ghost | $\Vert v_ia_i^\top\Vert_F = \Vert v_i\Vert\,\Vert a_i\Vert$; for sequences, the summed entrywise product of two Gram matrices | Per-example norms at scale; ghost norms per layer while $2N^2 < d_{\text{in}}d_{\text{out}}$ |
| @prp-dpsgd-microbatch | Sensitivity $2C$; noise $2m\sigma C/B$ per coordinate at the same $\varepsilon$ | Speed up per-example clipping by vectorizing or by ghost norms, not by microbatches |
| @exm-dpsgd-run | Accuracy from 0.8209 at $\varepsilon = 0.5$ to 0.9604 at $\varepsilon = 8$, against 0.9610 non-private | Expect accuracy close to non-private at large $\varepsilon$ and a collapse below some threshold; locate it for your task before promising a budget |

: The results of this chapter, and where each is used. {#tbl-dpsgd-summary}

## References {#sec-dpsgd-references}

- Abadi, Chu, Goodfellow, McMahan, Mironov, Talwar & Zhang, "Deep Learning with Differential Privacy" (CCS 2016). DP-SGD and the moments accountant.
- Mironov, Talwar & Zhang, "Rényi Differential Privacy of the Sampled Gaussian Mechanism" (arXiv 2019). The accountant of @eq-dpsgd-rdp.
- De, Berrada, Hayes, Smith & Balle, "Unlocking High-Accuracy Differentially Private Image Classification through Scale" (arXiv 2022). Large batches, small clipping norms and other practices for accurate private image classification.
- Anil, Ghazi, Gupta, Kumar & Manurangsi, "Large-Scale Differentially Private BERT" (Findings of EMNLP 2022). Very large batches for private language-model pretraining.
- Goodfellow, "Efficient Per-Example Gradient Computations" (arXiv 2015). The norm trick of @eq-dpsgd-ghost.
- Lee & Kifer, "Scaling up Differentially Private Deep Learning with Fast Per-Example Gradient Clipping" (PoPETs 2021). Clipping with the norm trick and a second backward pass, without forming per-example gradients.
- Subramani, Vadivelu & Kamath, "Enabling Fast Differentially Private SGD via Just-in-Time Compilation and Vectorization" (NeurIPS 2021). How vectorization removes most of the cost of per-example clipping.
- Papernot, Thakurta, Song, Chien & Erlingsson, "Tempered Sigmoid Activations for Deep Learning with Differential Privacy" (AAAI 2021). Bounded activations train better than ReLU under DP-SGD.
- Li, Tramèr, Liang & Hashimoto, "Large Language Models Can Be Strong Differentially Private Learners" (ICLR 2022). Ghost clipping for sequence models, @eq-dpsgd-ghost-seq.
- Bu, Mao & Xu, "Scalable and Efficient Training of Large Convolutional Neural Networks with Differential Privacy" (NeurIPS 2022). Choosing per layer between ghost norms and explicit gradients.
- Balle, Barthe, Gaboardi, Hsu & Sato, "Hypothesis Testing Interpretations and Rényi Differential Privacy" (AISTATS 2020). The conversion of @eq-dpsgd-convert.
