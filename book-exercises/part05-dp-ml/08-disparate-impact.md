# Exercises: Disparate impact

Exercises for [the chapter on disparate impact](../../book-full/part05-dp-ml/08-disparate-impact.md), each with a worked solution. Try each exercise before opening its solution. The script `code/part05-dp-ml/solutions_disparate_impact.py` computes every number quoted here; run it with `uv run python code/part05-dp-ml/solutions_disparate_impact.py`.

The solutions use the chapter's notation. Training with differential privacy (DP) runs stochastic gradient descent (SGD) with per-example clipping and Gaussian noise, DP-SGD. $C$ is the clipping norm, $\sigma$ the noise multiplier, $q$ the Poisson sampling rate, $B = qn$ the expected batch size, $T$ the number of steps and $\eta$ the step size. A group $g$ has a share $s_g$ of the $n$ training records, and its **clipping weight** $w_g$ is the average of $\min(1, C/\|g_i\|)$ over its records. The running example uses $q = 0.01$, $C = 0.5$, $\eta = 0.5$, $T = 10{,}000$ and $\delta = 10^{-5}$; at $\varepsilon = 4$ the noise multiplier is $\sigma = 1.36$ and the noise on the averaged gradient has standard deviation $\tau = \sigma C/B = 0.0068$ per coordinate.

## Upweighting does not survive clipping {#sec-sol-fair-1}

> **Exercise 1.** **Upweighting does not survive clipping.** A common fix for an under-represented group is to multiply its records' losses by a factor $u > 1$. Take the chapter's two-dimensional example: 95 majority records with gradient $(0.2, 0)$ and 5 minority records with gradient $(0, 4)$, clipped at $C = 0.5$. Define the minority's share of the update as its component divided by the sum of the two components. Compute it without clipping, and with clipping for $u = 1$, 4 and 20. Then compute the effective share of @eq-fair-effective when the majority keeps weight 1 and the minority has weight $C/4 = 0.125$. *Check: 0.5128 without clipping; 0.1163 with clipping for every $u$; effective share 0.0065.* *(Clipping as reweighting)*

::: {.callout-tip collapse="true" title="Solution"}
1. **Without clipping.** The average gradient is $(0.95 \times 0.2, 0.05 \times 4) = (0.19, 0.2)$, and the minority's share is $0.2/0.39 = 0.5128$: five records with large gradients pull as hard as ninety-five with small ones.
2. **With clipping, $u = 1$.** The majority's gradients have length $0.2 < C$ and are untouched. The minority's have length 4 and are rescaled to length 0.5, so the average is $(0.19, 0.05 \times 0.5) = (0.19, 0.025)$, and the share is $0.025/0.215 = 0.1163$.
3. **Upweighting.** Multiplying the minority's loss by $u$ multiplies its gradient by $u$, to $(0, 4u)$. Its length $4u$ is still above $C$, so clipping rescales it back to length 0.5, pointing the same way. The clipped average is again $(0.19, 0.025)$, and the share 0.1163, for $u = 4$ and for $u = 20$ alike.
4. **Effective share.** @eq-fair-effective gives $0.05 \times 0.125/(0.05 \times 0.125 + 0.95 \times 1) = 0.00625/0.95625 = 0.0065$. This is the minority's share of the records, weighted by how much of each record's gradient survives; step 2's 0.1163 is larger because the minority's gradients were twenty times longer to begin with.

*Clipping bounds each record's pull at $C$ whatever the loss says, so a weight on the loss is undone exactly for records that are clipped. To give a group more pull you must give it more records in the batch (oversampling, which costs privacy) or a larger share of the clipping budget.*
:::

## The noise floor, exactly {#sec-sol-fair-2}

> **Exercise 2.** **The noise floor, exactly.** In the quadratic model of @sec-fair-noise, each of a group's $k$ weights follows $x \leftarrow (1 - a)x - \eta\tau\xi$ with $a = \eta s_g\lambda$ and $\xi$ standard normal. Derive the stationary variance of $x$ without the small-$a$ approximation, and evaluate the group's excess loss $\frac\lambda2k\operatorname{Var}x$ for $k = 51$, $\lambda = 1$, $\eta = 0.5$, $\tau = 0.0068$ and shares $0.95$ and $0.05$. Compare with the approximation $k\eta\tau^2/(4s_g)$. *Check: $4.069 \times 10^{-4}$ against $3.103 \times 10^{-4}$ for the majority, $5.970 \times 10^{-3}$ against $5.896 \times 10^{-3}$ for the minority.* *(Stationary variance of a linear recursion)*

::: {.callout-tip collapse="true" title="Solution"}
1. **One step of variance.** The new value is $(1 - a)$ times the old plus independent noise of variance $\eta^2\tau^2$. Variances of independent terms add, and a constant factor multiplies the variance by its square: $V_{t+1} = (1 - a)^2V_t + \eta^2\tau^2$.
2. **The fixed point.** For $0 < a < 2$, $(1 - a)^2 < 1$, so the recursion converges to the $V$ with $V = (1 - a)^2V + \eta^2\tau^2$, that is $V = \eta^2\tau^2/(1 - (1 - a)^2) = \eta^2\tau^2/(a(2 - a))$.
3. **The majority, $s_g = 0.95$.** $a = 0.5 \times 0.95 = 0.475$, so $V = 0.25 \times 0.0068^2/(0.475 \times 1.525) = 1.596 \times 10^{-5}$, and the excess loss is $0.5 \times 51 \times V = 4.069 \times 10^{-4}$. The approximation $51 \times 0.5 \times 0.0068^2/(4 \times 0.95) = 3.103 \times 10^{-4}$ is 24% low, because $2 - a = 1.525$ is far from 2.
4. **The minority, $s_g = 0.05$.** $a = 0.025$, $V = 2.341 \times 10^{-4}$, excess loss $5.970 \times 10^{-3}$; the approximation gives $5.896 \times 10^{-3}$, 1% low.
5. **The ratio.** Exactly, the minority's floor is $5.970 \times 10^{-3}/4.069 \times 10^{-4} = 14.7$ times the majority's, against $0.95/0.05 = 19$ from the approximation. The exact form is @eq-fair-floor.

*The approximation is accurate for small groups, the case that matters, and overstates the gap between a small group and a large one by the factor $(2 - a_{\text{major}})/(2 - a_{\text{minor}})$.*
:::

## Why waiting does not help a small group {#sec-sol-fair-3}

> **Exercise 3.** **Why waiting does not help a small group.** By @eq-fair-breakeven, a drift that is a fraction $s_g$ of the population's needs $1/s_g^2$ times as many steps to emerge from the noise. Evaluate the factor for $s_g = 0.2$, 0.05 and 0.01. Then explain why training longer at a fixed $\varepsilon$ does not fix this. *Check: 25, 400 and 10,000.* *(Drift against random walk)*

::: {.callout-tip collapse="true" title="Solution"}
1. **The factors.** $1/0.2^2 = 25$, $1/0.05^2 = 400$, $1/0.01^2 = 10{,}000$.
2. **What they mean.** In the running example the population's signal emerges after about 228 steps at $\varepsilon = 4$. A group with 1% of the records, pulling with its share only, would need about $2.3$ million steps, two hundred times the whole run.
3. **Why more steps do not help.** At a fixed budget, more steps need more noise per step. In the regime of the DP-SGD chapter, $\varepsilon$ stays fixed when $Tq^2/\sigma^2$ does, so multiplying $T$ by $m$ multiplies $\sigma$ by $\sqrt m$ and the per-step noise variance by $m$. The break-even count $(\text{noise}/\text{signal})^2$ also grows by $m$: the run gets longer and the target moves away at the same rate.
4. **What does help.** A larger share of the signal (more records from the group, or a larger effective share after clipping), less noise per unit of signal (larger batches, @eq-dpsgd-batch), or a starting point that already serves the group (a public pretrained model).

*The $1/s_g^2$ law is about directions the rest of the data does not teach. Where the minority's task overlaps the majority's, the majority's gradient does part of the work, and the penalty is smaller.*
:::

## How many test records does a gap need? {#sec-sol-fair-4}

> **Exercise 4.** **How many test records does a gap need?** A model has accuracy 0.70 on 100 minority test records. Give the 95% Clopper–Pearson interval. Then, with the majority measured on 9,500 test records at accuracy 0.95, find the smallest number of minority test records for which the normal 95% interval of the gap, @eq-fair-gap, excludes zero when the minority's true accuracy is 0.90, and when it is 0.85. *Check: $[0.600, 0.788]$; 140 and 50 records.* *(Binomial intervals)*

::: {.callout-tip collapse="true" title="Solution"}
1. **The interval.** With $k = 70$ correct of $m = 100$, the lower end solves $\Pr[\mathrm{Bin}(100, l) \ge 70] = 0.025$ and the upper end $\Pr[\mathrm{Bin}(100, u) \le 70] = 0.025$. Bisection gives $[0.600, 0.788]$, about 19 points wide.
2. **The gap's standard error.** The two accuracies are measured on disjoint test records, so their variances add: $\mathrm{se}^2 = 0.95 \times 0.05/9{,}500 + p(1 - p)/m$. The first term is $5 \times 10^{-6}$ and negligible.
3. **Minority at 0.90.** The gap is 0.05. The interval excludes zero when $1.96\sqrt{5 \times 10^{-6} + 0.09/m} < 0.05$, that is $0.09/m < 6.46 \times 10^{-4}$, so $m \ge 140$.
4. **Minority at 0.85.** The gap is 0.10, and $1.96\sqrt{5 \times 10^{-6} + 0.1275/m} < 0.10$ needs $m \ge 50$.
5. **Reading it.** The number of minority records you need grows like one over the gap squared. A test split that is 5% minority and has 1,000 records holds 50 of them: enough to see a 10-point gap, not a 5-point one.

*This is the expected gap at the true accuracies; with an observed gap the same calculation gives the interval to report. Disjoint groups make the variances add; the majority's term almost never matters.*
:::

## What oversampling costs {#sec-sol-fair-5}

> **Exercise 5.** **What oversampling costs.** Sampling the minority's records at rate $vq$ instead of $q$ gives them more pull, but their privacy is that of a subsampled Gaussian at rate $vq$ (@eq-fair-oversample). With $\sigma = 1.36$, $q = 0.01$, $T = 10{,}000$ and $\delta = 10^{-5}$, compute the minority records' $\varepsilon$ for $v = 2$, 4 and 8, and the noise multiplier that brings them back to $\varepsilon = 4$. *Check: $\varepsilon = 8.92$, 20.96 and 53.04; $\sigma = 2.44$, 4.71 and 9.31.* *(Amplification by subsampling)*

::: {.callout-tip collapse="true" title="Solution"}
1. **Whose privacy changes.** In the add/remove analysis of one record $x$, only $x$'s own inclusion probability enters the mixture of the subsampled Gaussian. Majority records are still sampled at $q$; minority records at $vq$. The guarantee of the whole algorithm is the worst over records, so it is the minority's.
2. **The accountant.** At $\sigma = 1.36$ and rate $vq$ the Rényi differential privacy (RDP) accountant gives $\varepsilon = 8.92$ for $v = 2$, 20.96 for $v = 4$ and 53.04 for $v = 8$, against 4 at $v = 1$.
3. **Paying it back.** To keep every record at $\varepsilon = 4$, the noise multiplier must be 2.44, 4.71 and 9.31. The number of steps is fixed, and in the small-$q$ regime of the DP-SGD chapter $\varepsilon$ stays fixed when $Tq^2(e^{1/\sigma^2} - 1)$ does. For large $\sigma$ that is $Tq^2/\sigma^2$, so $\sigma$ grows in proportion to the rate: $1.36v$ would be 2.72, 5.44 and 10.88. The accountant needs a little less, because at $\sigma = 1.36$ the factor $\sigma^2(e^{1/\sigma^2} - 1)$ is still well above 1 and at the larger $\sigma$ it is not.
4. **Who gains.** The minority's pull grows by $v$, and the noise by $\sigma(vq)/\sigma(q)$: for $v = 4$, by $4.71/1.36 = 3.46$. The minority's signal-to-noise ratio improves by $4/3.46 = 1.16$; the majority's falls by 3.46.

*Oversampling is honest only if the accountant uses the largest sampling rate. Paid for that way, it buys the minority little and costs the majority much; the chapter's experiment shows the same.*
:::

## Global scaling is a large clipping norm {#sec-sol-fair-6}

> **Exercise 6.** **Global scaling is a large clipping norm.** Global scaling multiplies every per-example gradient by $C/Z$, where $Z$ is an upper bound on all gradient norms, and adds noise $\sigma C$ to the sum. Show that if no gradient exceeds $Z$, one step equals a DP-SGD step with clipping norm $Z$, noise multiplier $\sigma$ and step size $\eta C/Z$, as @eq-fair-global states. Check numerically on 200 records of the running example at initialization with $C = 0.5$, $Z = 50$ and the same noise draw. *Check: the two updates differ by at most $1.7 \times 10^{-18}$.* *(Scale invariance)*

::: {.callout-tip collapse="true" title="Solution"}
1. **Global scaling.** Each gradient becomes $(C/Z)g_i$, of length at most $C$ because $\|g_i\| \le Z$. The step is $-\frac\eta B\big(\frac CZ\sum_ig_i + \sigma C\xi\big)$ with $\xi$ standard normal.
2. **Factor out $C/Z$.** $\frac CZ\sum_ig_i + \sigma C\xi = \frac CZ\big(\sum_ig_i + \sigma Z\xi\big)$.
3. **Recognize DP-SGD.** With clipping norm $Z$, no gradient is clipped, so the clipped sum is $\sum_ig_i$ and the noise is $\sigma Z\xi$. The step $-\frac{\eta C/Z}B\big(\sum_ig_i + \sigma Z\xi\big)$ is step 2's expression: DP-SGD with clipping norm $Z$ and step size $\eta C/Z$.
4. **Check.** At initialization the 200 gradient norms are well below 50, and the two computed updates agree to $1.7 \times 10^{-18}$, rounding error.
5. **Privacy.** Both have sensitivity $C$ to noise $\sigma C$ (or $Z$ to $\sigma Z$), the same ratio $1/\sigma$, so the same accountant applies.

*Global scaling does not escape the trade-off of the clipping norm: it is the point where nothing is clipped, so the update is unbiased in direction, and the noise is as large relative to the signal as it can be. Esipova et al.'s adaptive variant moves $Z$ during training to sit just above most gradient norms.*
:::
