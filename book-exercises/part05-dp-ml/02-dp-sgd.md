# Exercises: DP-SGD

Exercises for [the chapter on private stochastic gradient descent](../../book-full/part05-dp-ml/02-dp-sgd.md), each with a worked solution. Try each exercise before opening its solution. The script `code/part05-dp-ml/solutions_dp_sgd.py` computes every number quoted here; run it with `uv run python code/part05-dp-ml/solutions_dp_sgd.py`.

The solutions use the chapter's notation. To give differential privacy (DP), stochastic gradient descent (SGD) is run with per-example clipping and Gaussian noise, as DP-SGD. $C$ is the clipping norm, $\sigma$ the noise multiplier, $q$ the Poisson sampling rate, $B = qn$ the expected batch size, $T$ the number of steps, $\eta$ the learning rate and $d$ the number of parameters $\theta$. The running example has $n = 10{,}000$, $d = 3{,}329$, $q = 0.01$ and $C = 0.5$, and its averaged clipped gradient has length about 0.026 during training. Rényi differential privacy (RDP) is the accountant's currency.

## When does the signal beat the noise? {#sec-sol-dpsgd-1}

> **Exercise 1.** **When does the signal beat the noise?** In the running example the averaged clipped gradient has length about 0.026. Using the argument of @sec-dpsgd-algorithm, compute the length of the noise on the averaged gradient and the number of steps after which accumulated signal exceeds accumulated noise, at $\varepsilon = 8$, 4 and 1 ($\delta = 10^{-5}$, $T = 10{,}000$). *Check: 104, 228 and 2,100 steps.* *(Drift against random walk)*

::: {.callout-tip collapse="true" title="Solution"}
1. **The noise multipliers.** From the accountant, $\sigma = 0.917$, 1.36 and 4.13 for the three budgets (@tbl-dpsgd-sigma).
2. **Noise length.** The noise on the averaged gradient has standard deviation $\sigma C/B$ per coordinate and length about $\sigma C\sqrt d/B$: $0.917 \times 0.5 \times \sqrt{3{,}329}/100 = 0.265$, then 0.392 and 1.191.
3. **Accumulation.** Over $t$ steps a signal of roughly constant direction adds up to length $0.026t$, and independent noise to about $\sqrt t$ times its per-step length. They are equal when $\sqrt t = \text{noise}/0.026$, so $t = (\text{noise}/0.026)^2$.
4. **Numbers.** $(0.265/0.026)^2 = 104$, $(0.392/0.026)^2 = 228$ and $(1.191/0.026)^2 = 2{,}100$.

*At $\varepsilon = 1$ a run of 10,000 steps contains only about five break-even intervals; at $\varepsilon = 8$ it contains about a hundred. This is a rough measure, since the signal's direction changes as the model learns, but it explains why accuracy collapses below some budget instead of declining smoothly.*
:::

## Dividing by the realized batch size {#sec-sol-dpsgd-2}

> **Exercise 2.** **Dividing by the realized batch size.** Under Poisson sampling the batch size $|S|$ is random, with mean $B = qn$. For $n = 10{,}000$ and $q = 0.01$, compute its relative standard deviation, the probability that the batch is empty, and (by a normal approximation) the probability that it is less than half its mean. Repeat for $n = 1{,}000$. Why does dividing by $|S|$ break the proof even when these probabilities are tiny? *Check: relative standard deviation 0.0995 and 0.3146.* *(Binomial batch sizes)*

::: {.callout-tip collapse="true" title="Solution"}
1. **The distribution.** $|S|$ is a sum of $n$ independent coins with probability $q$, so it is binomial with mean $nq$ and variance $nq(1 - q)$.
2. **Relative spread.** $\sqrt{nq(1 - q)}/(nq) = \sqrt{(1 - q)/(nq)}$: $\sqrt{0.99/100} = 0.0995$ for $n = 10{,}000$, and $\sqrt{0.99/10} = 0.3146$ for $n = 1{,}000$.
3. **Empty batch.** $(1 - q)^n$: $10^{-43.6}$ for $n = 10{,}000$, and $10^{-4.36}$, about 1 in 23,000, for $n = 1{,}000$.
4. **Less than half.** The normal approximation gives $\Phi(-\frac{B/2}{\sqrt{nq(1-q)}})$: $2.5 \times 10^{-7}$ for $B = 100$, and 0.056 for $B = 10$.
5. **Why it breaks the proof anyway.** The derivation needs the update to be a function of the noisy sum $G_t$ alone. Dividing by $|S_t|$ uses a second data-dependent quantity, which changes by one when the target record is drawn. The accountant has not paid for releasing it, and the sensitivity of $G_t/|S_t|$ is no longer $C$ divided by a constant. A proof does not hold "with high probability" unless it was written that way.

*Small batch sizes vary a lot, and a correct implementation divides by the public $B$; the update is then unbiased (@eq-dpsgd-expected), at the price of a step length that varies a little with $|S_t|$.*
:::

## Adam does not see the clipping norm {#sec-sol-dpsgd-3}

> **Exercise 3.** **Adam does not see the clipping norm.** Suppose every per-example gradient is clipped, so each noisy sum is $C$ times a quantity that does not depend on $C$ (@eq-dpsgd-normalized). Show that Adam's update is then independent of $C$ up to its small constant $c_{\text{Adam}}$ in the denominator, so with Adam the clipping norm does not even need to be matched by the learning rate. Check numerically with 20 random gradients, $C = 0.01$ and $C = 10$. *Check: the updates agree to a relative difference of $7.3 \times 10^{-5}$.* *(Scale invariance of adaptive optimizers)*

::: {.callout-tip collapse="true" title="Solution"}
1. **Adam's update.** Adam keeps running averages $m_t$ of the gradients and $v_t$ of their squares, coordinate by coordinate, corrects them for their start at zero, and steps by $-\eta\hat m_t/(\sqrt{\hat v_t} + c_{\text{Adam}})$.
2. **Scale the gradients by $C$.** Every gradient is multiplied by $C$, so $m_t$ is multiplied by $C$ and $v_t$ by $C^2$. The bias corrections do not depend on the gradients.
3. **The ratio.** $\hat m_t/\sqrt{\hat v_t}$ is unchanged, since $C/\sqrt{C^2} = 1$. Only $c_{\text{Adam}}$ breaks the invariance, and only when $\sqrt{\hat v_t}$ is comparable to it.
4. **Check.** With 100 coordinates, 20 standard normal gradients scaled by 0.01 and by 10, and $c_{\text{Adam}} = 10^{-8}$, the summed updates differ by at most $7.3 \times 10^{-5}$ relative to their size. At $C = 0.01$ the gradients have entries of about 0.01, and $10^{-8}$ is a relative perturbation of about $10^{-6}$ per step, which accumulates over the 20 steps.

*With Adam, DP-SGD's clipping norm only decides which gradients are clipped, not the step length. This is one reason DP training with Adam is less sensitive to $C$. Privacy is unaffected: Adam is post-processing of the noisy sums.*
:::

## Where ghost clipping stops paying {#sec-sol-dpsgd-4}

> **Exercise 4.** **Where ghost clipping stops paying.** For a dense layer from $d_{\text{in}}$ to $d_{\text{out}}$ applied to sequences of length $N$, the number of tokens, ghost norms store $2BN^2$ numbers and explicit per-example gradients $Bd_{\text{in}}d_{\text{out}}$. Find the sequence length above which explicit gradients are cheaper, for $768 \to 768$, $768 \to 3{,}072$ and $4{,}096 \to 4{,}096$. *Check: 543, 1,086 and 2,896 tokens.* *(Comparing costs)*

::: {.callout-tip collapse="true" title="Solution"}
1. **The condition.** Explicit gradients are cheaper when $Bd_{\text{in}}d_{\text{out}} < 2BN^2$, that is $N > \sqrt{d_{\text{in}}d_{\text{out}}/2}$. The batch size cancels.
2. **Numbers.** $\sqrt{768^2/2} = 768/\sqrt2 = 543$; $\sqrt{768 \times 3{,}072/2} = 1{,}086$; $4{,}096/\sqrt2 = 2{,}896$.
3. **Reading it.** Wider layers favour ghost norms up to longer sequences. In a transformer, the projections inside the feed-forward block, $768 \to 3{,}072$, stay ghost-friendly up to twice the sequence length of the attention projections, $768 \to 768$.

*The choice is per layer and per sequence length; implementations that make it automatically, such as the mixed ghost clipping of Bu, Mao & Xu (2022), use exactly this comparison.*
:::

## The small-q law for the noise multiplier {#sec-sol-dpsgd-5}

> **Exercise 5.** **The small-q law for the noise multiplier.** Approximate the composed RDP curve of DP-SGD by a line, $\rho\alpha$, and convert it with $\varepsilon = \rho + 2\sqrt{\rho\ln(1/\delta)}$ (the zero-concentrated DP conversion of [Concentrated differential privacy](../../book-full/part04-privacy-accounting/03-concentrated-dp.md)). Find $\rho$ for $\varepsilon = 4$, $\delta = 10^{-5}$. Then compute $\sigma$ for $q = 0.01$, $T = 10{,}000$ in two ways, from $Tq^2/(2\sigma^2) = \rho$ and from $Tq^2(e^{1/\sigma^2} - 1)/2 = \rho$, and compare with the accountant. Repeat for $q = 0.001$, $T = 100{,}000$. *Check: $\rho = 0.2977$; $\sigma = 1.296$ and 1.463 against the accountant's 1.36; at $q = 0.001$, 0.410 and 0.718 against 0.704.* *(Approximating an accountant)*

::: {.callout-tip collapse="true" title="Solution"}
1. **Solve for $\rho$.** With $s = \sqrt\rho$ and $c = \ln(10^5) = 11.513$, $s^2 + 2s\sqrt c - 4 = 0$, so $s = -\sqrt c + \sqrt{c + 4} = 0.5456$ and $\rho = 0.2977$.
2. **Quadratic law.** $\sigma^2 = Tq^2/(2\rho) = 10{,}000 \times 10^{-4}/0.5954 = 1.680$, so $\sigma = 1.296$.
3. **With the exact factor.** $e^{1/\sigma^2} - 1 = 2\rho/(Tq^2) = 0.5954$, so $1/\sigma^2 = \ln 1.5954 = 0.4671$ and $\sigma = 1.463$.
4. **Compare at $q = 0.01$.** The accountant's 1.36 lies between the two. The quadratic law ignores the factor $\sigma^2(e^{1/\sigma^2} - 1) = 1.33$ and needs too little noise; the second estimate includes it but uses a looser conversion than the accountant's, and needs too much.
5. **At $q = 0.001$.** The required $\sigma$ is near 0.7, where the factor is about 3: the quadratic law gives 0.410, far too small, while the second estimate, 0.718, is within 2% of the accountant's 0.704.

*The quadratic law is good for orientation at large $\sigma$ and wrong by almost a factor of two at small $\sigma$. Use it to see trends, and the accountant for numbers.*
:::

## How ε grows with the number of steps {#sec-sol-dpsgd-6}

> **Exercise 6.** **How $\varepsilon$ grows with the number of steps.** With $q = 0.01$, $\sigma = 1$ and $\delta = 10^{-5}$, compute $\varepsilon$ after 100, 1,000, 10,000 and 100,000 steps, and the number of steps at which $\varepsilon$ first reaches 1, 2 and 4. Explain the shape. *Check: 1.21, 2.10, 6.71 and 27.23; 5, 882 and 3,859 steps.* *(Composition in RDP)*

::: {.callout-tip collapse="true" title="Solution"}
1. **The values.** The accountant gives $\varepsilon = 1.21$, 2.10, 6.71 and 27.23.
2. **The thresholds.** Bisection over $T$ gives 5 steps for $\varepsilon = 1$, 882 for 2 and 3,859 for 4.
3. **Why the first steps cost so much.** The conversion of @eq-dpsgd-convert adds $\ln(1/\delta)/(\alpha - 1)$ and related terms at any order, about 1 for the orders that are best for short runs. Even a few steps report $\varepsilon \approx 1$, because the conversion, not the composition, dominates. The first step alone has $\varepsilon = 0.96$ by this accountant.
4. **The middle.** From 1,000 to 10,000 steps, $\varepsilon$ grows by a factor of 3.2, close to $\sqrt{10}$, as in the quadratic regime: the composed RDP is $T$ times a curve linear in $\alpha$, and the conversion of a linear curve grows like the square root of its slope for small slopes.
5. **The end.** From 10,000 to 100,000 steps the factor is 4.1, more than $\sqrt{10}$: the best order falls toward 1, where the curve is no longer linear and the conversion grows faster.

*Privacy cost is not proportional to training time. It is front-loaded by the conversion, grows like $\sqrt T$ over the useful range, and faster for very long runs.*
:::

## What microbatches cost {#sec-sol-dpsgd-7}

> **Exercise 7.** **What microbatches cost.** Microbatched DP-SGD (@eq-dpsgd-microbatch) clips the mean gradient of groups of $m$ records fixed in advance, and adds noise $2\sigma C$ to keep the accountant's $\varepsilon$. For the running example ($q = 0.01$, $B = 100$, $C = 0.5$, $\sigma = 1$, 10,000 steps, $\varepsilon = 6.71$), find by what factor microbatches of $m = 2$ and $m = 4$ multiply the noise on the averaged update. Per-example DP-SGD with that much more noise would have what $\varepsilon$? Explain the factor 2 in the sensitivity with a two-record example. *Check: factors 4 and 8; per-example DP-SGD with $\sigma = 4$ and 8 has $\varepsilon = 1.035$ and 0.481.* *(Sensitivity of clipped means)*

::: {.callout-tip collapse="true" title="Solution"}
1. **The factor.** The noise per coordinate is $2\sigma C/(B/m) = 2m\,\sigma C/B$, against $\sigma C/B$ for per-example clipping, a factor $2m$: 4 for $m = 2$ and 8 for $m = 4$.
2. **The same noise, spent per example.** Per-example DP-SGD with noise multiplier 4 or 8 has exactly as much noise on its averaged update as these microbatched runs. The accountant gives it $\varepsilon = 1.035$ and 0.481. So microbatches of 4 buy the accuracy of a run at $\varepsilon \approx 0.5$ while paying $\varepsilon = 6.71$.
3. **The factor 2.** Take a microbatch with one other record whose gradient is $Cu$ for a unit vector $u$, and add a record whose gradient is $-3Cu$. The mean changes from $Cu$ to $-Cu$, both of length $C$ after clipping, so the clipped mean moves by $2C$. Under per-example clipping the added record contributes its own clipped term, $-Cu$, and the sum moves by only $C$.

*Microbatches make clipping cheaper by making it coarser, and the privacy accounting charges for that twice: fewer clipped terms carry the signal, and each moves by up to $2C$. Vectorized per-example gradients give the speed without the charge.*
:::
