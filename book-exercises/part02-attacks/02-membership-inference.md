# Exercises: Membership inference

Exercises for [Membership inference](../../book-full/part02-attacks/02-membership-inference.md), each with a worked solution. Try each exercise before opening its solution. The script `code/part02-attacks/solutions_membership_inference.py` computes every number quoted here; run it with `uv run python code/part02-attacks/solutions_membership_inference.py`.

The solutions use the chapter's notation. $\phi = y\langle \theta, x\rangle$ is the logit of the model's confidence in the true label, IN and OUT are the distributions of a record's $\phi$ over models trained with and without it, and an attack's true positive rate (TPR) and false positive rate (FPR) are measured over (target model, pool point) pairs of the running example. The likelihood ratio attack (LiRA) is the per-example test of the chapter, and attacks are compared by their receiver operating characteristic (ROC) curves and the area under the curve (AUC). Exercises 1 and 7 reuse the chapter's experiment with seed 0, so their numbers match the chapter's tables; Exercise 6 draws a fresh pool.

## Accuracy with an unbalanced prior {#sec-sol-mia-1}

> **Exercise 1.** **Accuracy with an unbalanced prior.** Suppose a fraction $\pi_1$ of the queried records are members. Show that "member if correctly classified" has accuracy $\pi_1(1 - e_{\mathrm{train}}) + (1 - \pi_1)e_{\mathrm{test}}$. With the running example's errors, compare it with the trivial attack "always non-member" at $\pi_1 = 0.5$ and $\pi_1 = 0.1$. *Check: 0.5584 against 0.5, and 0.3247 against 0.9.* What does this say about reporting accuracy? *(Fact 1, Yeom's attack)*

::: {.callout-tip collapse="true" title="Solution"}
1. **Condition on membership** (Fact 1 of [Threat models and privacy games](../../book-full/part02-attacks/01-threat-models.md)): accuracy $= \pi_1\,\mathrm{TPR} + (1 - \pi_1)(1 - \mathrm{FPR})$.
2. **The rates of this attack.** A member is flagged when it is classified correctly, so $\mathrm{TPR} = 1 - e_{\mathrm{train}}$. A non-member is flagged when it is classified correctly, so $\mathrm{FPR} = 1 - e_{\mathrm{test}}$ and $1 - \mathrm{FPR} = e_{\mathrm{test}}$.
3. **Substitute:** accuracy $= \pi_1(1 - e_{\mathrm{train}}) + (1 - \pi_1)e_{\mathrm{test}}$.
4. **The running example** has $e_{\mathrm{train}} = 0.1495$ and $e_{\mathrm{test}} = 0.2663$.
   - At $\pi_1 = 0.5$: $0.5 \times 0.8505 + 0.5 \times 0.2663 = 0.5584$, against 0.5 for "always non-member". This is $\frac12 + \frac12 \times 0.1168$, the advantage of the chapter.
   - At $\pi_1 = 0.1$: $0.1 \times 0.8505 + 0.9 \times 0.2663 = 0.3247$, against 0.9 for "always non-member".

The attack has the same TPR and FPR in both cases; only the prior changed. At $\pi_1 = 0.1$ it says "member" to 73% of non-members, and most queries are non-members, so its accuracy collapses below the trivial rule.

*Accuracy mixes the attack with the prior. TPR and FPR do not depend on the prior, which is one more reason to report them, and to report them at a low FPR, where a "member" verdict stays reliable even when members are rare.*
:::

## The Gaussian likelihood ratio {#sec-sol-mia-2}

> **Exercise 2.** **The Gaussian likelihood ratio.** For $H_0: \mathcal N(0, 1)$ against $H_1: \mathcal N(s, 1)$, show $\ln\Lambda(t) = st - s^2/2$. Use it to find, for the two kinds of record of @tbl-mia-two-types, the z-score cut that the threshold $\ln\Lambda \ge 1.497$ puts on each kind, and check that the pooled FPR is $10^{-3}$. *Check: cuts 3.244 (typical) and 2.374 (atypical).* *(Neyman–Pearson, per-example calibration)*

::: {.callout-tip collapse="true" title="Solution"}
1. **Write the log-densities.** $\ln\mathcal N(t; s, 1) = -\frac12(t - s)^2 - \frac12\ln(2\pi)$ and $\ln\mathcal N(t; 0, 1) = -\frac12t^2 - \frac12\ln(2\pi)$.
2. **Subtract.** The constants cancel: $\ln\Lambda(t) = -\frac12(t^2 - 2st + s^2) + \frac12t^2 = st - \frac{s^2}2$.
3. **Spot checks** (from the script, computing both forms): at $t = 3.0902$, $s = 0.5$ gives $1.4201$ and $s = 4$ gives $4.3608$; at $t = 1$, $s = 4$ gives $-4.0000$. The same z-score is strong evidence for a record with a large separation and weak evidence for one with a small separation.
4. **Invert.** For $s > 0$, $\ln\Lambda \ge \tau$ is the same as $t \ge (\tau + s^2/2)/s$. With $\tau = 1.497$:
   - typical records, $s = 0.5$: $t \ge (1.497 + 0.125)/0.5 = 3.244$;
   - atypical records, $s = 4$: $t \ge (1.497 + 8)/4 = 2.374$.
5. **FPR of each kind.** Under OUT, $t \sim \mathcal N(0, 1)$, so the FPRs are $1 - \Phi(3.244) = 5.90\times10^{-4}$ and $1 - \Phi(2.374) = 0.0088$.
6. **Pool.** $0.95 \times 5.90\times10^{-4} + 0.05 \times 0.0088 = 0.0010$. The TPR is $0.95\,(1 - \Phi(3.244 - 0.5)) + 0.05\,(1 - \Phi(2.374 - 4)) = 0.0503$, as in the chapter.

*A single threshold on the likelihood ratio becomes a different z-score cut for each kind of record: stricter where the separation is small, looser where it is large. That is how the Neyman–Pearson test spends its FPR budget where it buys the most TPR.*
:::

## Same AUC, different exposure {#sec-sol-mia-3}

> **Exercise 3.** **Same AUC, different exposure.** Derive the AUC, best advantage and TPR at FPR $\alpha$ of attacks A and B in @sec-mia-eval, and repeat the comparison at $r = 0.01$. *Check: AUC 0.5050, $\mu = 0.0177$, TPR at $10^{-3}$ 0.01099 for A and 0.00106 for B.* *(AUC as a ranking probability, Gaussian trade-off)*

::: {.callout-tip collapse="true" title="Solution"}
**Attack A.**

1. **AUC.** The AUC is $\Pr[\text{member's score} > \text{non-member's score}]$, with ties counted as one half. With probability $r$ the member is exposed and wins outright; otherwise its score has the same distribution as the non-member's and it wins half the time. So $\mathrm{AUC} = r + \frac12(1 - r) = \frac12 + \frac r2$.
2. **TPR.** At a threshold with FPR $\alpha$, the exposed members always pass and the rest pass with probability $\alpha$: $\mathrm{TPR} = r + (1 - r)\alpha$.
3. **Best advantage.** $\mathrm{TPR} - \mathrm{FPR} = r(1 - \alpha)$, largest as $\alpha \to 0$, where it tends to $r$.

**Attack B.**

4. **AUC.** A member's score minus a non-member's is $\mathcal N(\mu, 2)$, positive with probability $\Phi(\mu/\sqrt2)$. Setting this equal to $\frac12 + \frac r2$ gives $\mu = \sqrt2\,\Phi^{-1}(\frac12 + \frac r2)$.
5. **TPR** is @eq-mia-gauss-tpr with $s = \mu$, and the **best advantage** is the total variation between $\mathcal N(0, 1)$ and $\mathcal N(\mu, 1)$, $2\Phi(\mu/2) - 1$.

**Numbers** at $\alpha = 10^{-3}$ (@tbl-sol-mia-same-auc):

| $r$ | AUC | A: advantage | A: TPR | B: $\mu$ | B: advantage | B: TPR |
| --- | --- | --- | --- | --- | --- | --- |
| 0.05 | 0.5250 | 0.0500 | 0.05095 | 0.0887 | 0.0354 | 0.00134 |
| 0.01 | 0.5050 | 0.0100 | 0.01099 | 0.0177 | 0.0071 | 0.00106 |

: Attacks A and B with equal AUC, at two exposure levels. TPRs are at FPR $10^{-3}$. {#tbl-sol-mia-same-auc}

At $r = 0.01$, the AUC is 0.505, which most papers would call "no better than chance". Yet A identifies one member in a hundred with certainty, ten times the TPR of B at FPR $10^{-3}$.

*An AUC near one half says that the attack is weak on average, not that it is weak for everyone. A small exposed group barely moves the AUC and dominates the low-FPR end.*
:::

## How many non-members? {#sec-sol-mia-4}

> **Exercise 4.** **How many non-members?** You want to claim, at 95% confidence, that an attack's FPR is below $\alpha$, and you see no false positives at all. How many non-members do you need for $\alpha = 10^{-3}, 10^{-4}, 10^{-5}$? *Check: 2,995, 29,956 and 299,572.* *Hint: $(1 - \alpha)^n \le 0.05$.* *(Estimating rates from trials)*

::: {.callout-tip collapse="true" title="Solution"}
1. **The event to rule out.** If the true FPR were $\alpha$ or more, then seeing zero false positives among $n$ independent non-members would have probability at most $(1 - \alpha)^n$.
2. **Make it unlikely.** The one-sided 95% upper bound from zero counts ([Hypothesis testing](../../book-full/part01-prerequisites/04-hypothesis-testing.md)) is below $\alpha$ exactly when $(1 - \alpha)^n \le 0.05$.
3. **Solve.** Take logarithms (both sides negative): $n \ge \ln 0.05/\ln(1 - \alpha) = \ln 20/(-\ln(1 - \alpha))$.
4. **Plug in.** $\alpha = 10^{-3}$: $n \ge 2{,}994.2$, so 2,995. $\alpha = 10^{-4}$: $n \ge 29{,}955.8$, so 29,956. $\alpha = 10^{-5}$: $n \ge 299{,}571.7$, so 299,572. Each is about $3/\alpha$, the rule of three.

The running example has 64,000 non-member pairs, so a threshold at FPR $10^{-3}$ leaves about 64 false positives, enough for an interval of about $\pm 25\%$ on the FPR (the chapter's $[0.00077, 0.00128]$). At FPR $10^{-5}$ the same data could not even certify a bound. The pairs are also not fully independent: the same 1,000 points recur across the 128 target models, which is one more reason the interval is too narrow.

*Every factor of ten lower in FPR costs a factor of ten more non-members. Reporting a TPR at FPR $10^{-5}$ needs hundreds of thousands of non-member evaluations.*
:::

## Membership in a Gaussian mean {#sec-sol-mia-5}

> **Exercise 5.** **Membership in a Gaussian mean.** A "model" is the mean $\hat\theta$ of $n$ points drawn from $\mathcal N(0, I_d)$. The attacker sees $\hat\theta$ and a target $z$, and uses $T = \langle z, \hat\theta\rangle/(\|z\|/\sqrt n)$. Show that $T \sim \mathcal N(0, 1)$ when $z$ is not in the sample, and approximately $\mathcal N(\sqrt{d/n}, 1)$ when it is. Deduce the TPR at FPR $10^{-3}$ for $d = 1{,}000$ and $n = 100$, and check it by simulation. *Check: separation 3.1623, TPR 0.5287.* *Hint: given $z$, $\langle z, \sum_{j \ne i}x_j\rangle \sim \mathcal N(0, (n - 1)\|z\|^2)$, and $\|z\|^2 \approx d$.* *(Per-example separation)*

::: {.callout-tip collapse="true" title="Solution"}
1. **OUT.** If $z$ is not in the sample, $\hat\theta = \frac1n\sum_jx_j$ is independent of $z$ and $\hat\theta \sim \mathcal N(0, I_d/n)$. Given $z$, $\langle z, \hat\theta\rangle$ is a linear combination of independent Gaussians, so it is $\mathcal N(0, \|z\|^2/n)$. Dividing by $\|z\|/\sqrt n$ gives $T \sim \mathcal N(0, 1)$, exactly, whatever $z$ is.
2. **IN.** If $z = x_i$, then $\hat\theta = \frac1n\big(z + \sum_{j \ne i}x_j\big)$ and
$$\langle z, \hat\theta\rangle = \frac{\|z\|^2}n + \frac1n\Big\langle z, \sum_{j \ne i}x_j\Big\rangle.$$
Given $z$, the second term is $\mathcal N\big(0, (n - 1)\|z\|^2/n^2\big)$ by the hint.
3. **Standardize.** Divide by $\|z\|/\sqrt n$: $T = \|z\|/\sqrt n + \mathcal N\big(0, (n - 1)/n\big)$.
4. **Approximate.** $\|z\|^2$ is a sum of $d$ squared standard normals, with mean $d$ and relative spread $\sqrt{2/d}$, so $\|z\|/\sqrt n \approx \sqrt{d/n}$. And $(n - 1)/n \approx 1$. So $T \approx \mathcal N(\sqrt{d/n}, 1)$.
5. **TPR.** By Fact 2 of the chapter with separation $s = \sqrt{d/n}$, $\mathrm{TPR} = 1 - \Phi(3.0902 - s)$.

| $d$ | $n$ | Separation $\sqrt{d/n}$ | Formula TPR | Simulated TPR | Simulated FPR |
| --- | --- | --- | --- | --- | --- |
| 1,000 | 100 | 3.1623 | 0.5287 | 0.5268 | 0.0010 |
| 100 | 100 | 1.0000 | 0.0183 | 0.0181 | 0.0009 |
| 250 | 25 | 3.1623 | 0.5287 | 0.5350 | 0.0013 |

: Membership in a Gaussian mean at FPR $10^{-3}$, from 20,000 simulated targets in each world. {#tbl-sol-mia-mean}

The simulation (@tbl-sol-mia-mean) matches the formula within its sampling error at $n = 100$. At $n = 25$ the approximation in step 4 is cruder: the IN variance is $24/25$, not 1, and $\|z\|/\sqrt n$ varies from target to target. The simulated FPR stays at about $10^{-3}$ because step 1 is exact.

*Exposure depends on the ratio of parameters to records, $d/n$. With as many dimensions as records, a member is flagged 1.8% of the time at FPR $10^{-3}$; with ten times more dimensions, more than half the time. The tracing attacks of [Lower bounds](../../book-full/part03-dp-foundations/10-lower-bounds.md) build on exactly this statistic.*
:::

## The onion effect {#sec-sol-mia-6}

> **Exercise 6.** **The onion effect.** Remove the 50 outliers from the pool, retrain all target and shadow models, and rerun online LiRA. Before running, predict whether the TPR at FPR $10^{-3}$ *on typical points* goes up or down. Then run it (the solutions script does) and explain the result with self-influence. *(Self-influence, outliers)*

::: {.callout-tip collapse="true" title="Solution"}
1. **Setup.** Draw a new pool as in the chapter (seed 0 stream, after the earlier exercises), train 128 target and 64 shadow models on it, and compute online LiRA with per-example variances. Restrict the evaluation to typical points: build the ROC curve from the (target model, typical point) pairs only and read off the TPR at FPR $10^{-3}$. Then delete the 50 outliers, leaving a pool of 950, retrain all 192 models on random halves of it, and repeat.
2. **Result.** On typical points the TPR at FPR $10^{-3}$ goes from 0.0027 with the outliers to 0.0037 without them, with 60,800 typical member pairs behind each number. (The first number differs from the 0.0020 in @tbl-mia-outliers because there the threshold was set on all points, outliers included, and the pool is a fresh draw.)
3. **Why it goes up.** Each model now trains on 475 points instead of 500, all typical, and the outliers no longer take up the model's capacity. The typical points that are hardest to predict, those whose labels go against the trend, become the most atypical points in the data. Their self-influence rises, and with it their separation and their exposure. The mean separation of typical points, estimated from the shadow models, goes from 0.502 to 0.621.

*Privacy risk is relative to the rest of the dataset. Removing the most exposed records does not make the others safe; it promotes the next layer. Carlini et al. call this the privacy onion effect. A defence has to bound the exposure of every record, not remove the current worst ones.*
:::

## The shadow attack model is a threshold {#sec-sol-mia-7}

> **Exercise 7.** **The shadow attack model is a threshold.** Show that if the attack model for class $c$ is $\sigma(\kappa_c + \lambda_c\phi)$ with $\lambda_c > 0$, then within class $c$ it ranks pairs exactly as the loss does, and it says "member" with probability above $\frac12$ when $\phi > -\kappa_c/\lambda_c$. Compute these cut points for the running example. *Check: 0.751 for $y = -1$ and 0.813 for $y = +1$.* *(Shadow models, Fact 3)*

::: {.callout-tip collapse="true" title="Solution"}
1. **Monotone in $\phi$.** $\sigma$ is increasing and $\lambda_c > 0$, so $\phi \mapsto \sigma(\kappa_c + \lambda_c\phi)$ is increasing. Within class $c$, a larger $\phi$ always gives a larger attack score.
2. **Monotone in the loss.** By Fact 3 of the chapter, $\ell = \ln(1 + e^{-\phi})$ is decreasing in $\phi$. So within the class, the attack score is a decreasing function of the loss, and it orders pairs exactly as "lower loss means member" does. Any threshold on the score is a threshold on the loss, and the two attacks have the same ROC curve on the pairs of that class.
3. **The cut point.** $\sigma(u) > \frac12$ exactly when $u > 0$, so the attack model favours "member" when $\kappa_c + \lambda_c\phi > 0$, that is, $\phi > -\kappa_c/\lambda_c$.
4. **The running example.** The fitted models are $\kappa_{-1} = -0.179$, $\lambda_{-1} = 0.238$ and $\kappa_{+1} = -0.128$, $\lambda_{+1} = 0.158$. The cut points are $0.179/0.238 = 0.751$ and $0.128/0.158 = 0.813$ (computed from the unrounded coefficients).
5. **Across classes.** Pooling the two classes with one threshold on the attack score puts slightly different thresholds on $\phi$ in each class, so the pooled ROC differs from the loss attack's only through that offset. The chapter's @tbl-mia-attacks shows how little that buys: AUC 0.577 for both, and TPR at FPR 1% of 0.0104 against 0.0103 for the loss attack.

*An attack model that sees only the record-blind output $(y, \phi)$ can do no more than choose a loss threshold per class. Its value is calibration, not power.*
:::
