# Auditing private training

Every chapter of this part ended with a number. Train with stochastic gradient descent (SGD) under differential privacy (DP), that is, with per-example clipping and Gaussian noise (DP-SGD), and an accountant turns the configuration into a guarantee. That guarantee concludes a proof about an algorithm on paper, but what ships is code, and the code can quietly differ from the algorithm in a dozen places. The noise may be scaled for the averaged gradient but added to the sum, or the accountant may be handed the wrong dataset size. None of these changes the accuracy much, so the usual tests miss them. This chapter asks whether the trained model is as private as the accountant says.

An **audit** answers by attacking. It plants records of its own, called **canaries**, trains with and without them, and runs a membership test. It measures the test's true positive rate (TPR), how often it flags a canary that was trained on, and its false positive rate (FPR), how often it flags one that was not. The one idea is that DP caps how often any test can be right, so measured error rates imply a **lower bound** on $\varepsilon$. Such a bound cannot prove privacy, but it disproves any claim below it. The chapter then makes the bound bite. It puts confidence intervals on the measured rates, chooses the strongest attack, and gets many trials out of one or two training runs.

The running example is the part's network, a two-layer tanh network trained on the 10,000 records of [DP-SGD](02-dp-sgd.md) with sampling rate $q = 0.01$, clipping norm $C = 0.5$, noise multiplier $\sigma = 1$, learning rate 0.5 and $T = 10{,}000$ steps. The Rényi differential privacy (RDP) accountant reports $\varepsilon = 6.71$ at $\delta = 10^{-5}$, and this is the **claim** every audit below tests. Neighbours $D \sim D'$ differ by adding or removing one record, and @tbl-audit-symbols lists the symbols.

| Symbol | Meaning |
| --- | --- |
| $D$, $D' = D \cup \{z^*\}$ | the dataset without and with the canary $z^*$ |
| $\mathrm{FPR}$, $\mathrm{TPR}$ | how often the attack says "canary in" without and with the canary |
| $n$; $k_0$, $k_1$ | trials per hypothesis; false and true positives |
| $\beta$; $\varepsilon_{\mathrm{lo}}$ | the probability that an audit's bound is wrong (5%); the bound |
| $q$, $\sigma$, $C$, $B$, $\eta$, $T$ | sampling rate, noise multiplier, clipping norm, expected batch size (100), learning rate, steps |
| $x_t$; $\Lambda$ | the canary's projection at step $t$; the likelihood-ratio statistic |
| $\mu$ | the parameter of Gaussian differential privacy |
| $m$, $r$, $W$ | canaries, guesses and correct guesses of a one-run audit |

: The symbols of this chapter. {#tbl-audit-symbols}

## From an attack to a lower bound on ε {#sec-audit-bound}

You ran an attack and counted how often it was right. This section asks what that lets you conclude about $\varepsilon$, with what confidence, and how much $n$ trials could ever conclude. Every later audit reports its result through this bound.

::: {#def-audit-audit name="Audit trials and error rates"}
Let $M$ be a training algorithm, $D$ a dataset and $z^*$ a **canary** record, and write $D' = D \cup \{z^*\}$. A **trial** runs $M$ on $D$ or on $D'$ with fresh randomness and computes a score $s$ from the output. The attack says "canary in" when $s \ge \tau$, for a threshold $\tau$. Its **false positive rate** and **true positive rate** are
$$\mathrm{FPR} = \Pr[s \ge \tau \mid D], \qquad \mathrm{TPR} = \Pr[s \ge \tau \mid D'],$$
and its false negative rate is $\mathrm{FNR} = 1 - \mathrm{TPR}$. An audit runs $n$ trials on each dataset and counts $k_0$ false positives and $k_1$ true positives.
:::

The definition of DP limits these rates. By @thm-dpdef-testing in [The definition of differential privacy](../part03-dp-foundations/01-definition.md), if $M$ is $(\varepsilon, \delta)$-DP then every test, randomized or not, satisfies both
$$\mathrm{TPR} \le e^\varepsilon\,\mathrm{FPR} + \delta, \qquad 1 - \mathrm{FPR} \le e^\varepsilon\,(1 - \mathrm{TPR}) + \delta.$$ {#eq-audit-region}
The first inequality caps how often the test flags present canaries against how often it flags absent ones. The second swaps the two datasets and applies the cap to the verdict "canary out".

The audit sees only counts, not rates. If the trials are independent, $k_0$ is a binomial count whose success probability is the FPR, and @thm-ht-clopper-pearson in [Hypothesis testing](../part01-prerequisites/04-hypothesis-testing.md) turns such a count into an interval. Its proof shows that each end, on its own, misses the true rate with probability at most half the interval's failure probability. So write $\overline{\mathrm{FPR}}$ for the **one-sided Clopper–Pearson upper bound** on the FPR and $\underline{\mathrm{TPR}}$ for the lower bound on the TPR, each at level $\beta/2$. They solve
$$\Pr\big[\mathrm{Bin}(n, \overline{\mathrm{FPR}}) \le k_0\big] = \frac\beta2, \qquad \Pr\big[\mathrm{Bin}(n, \underline{\mathrm{TPR}}) \ge k_1\big] = \frac\beta2.$$
Putting these pessimistic ends into @eq-audit-region, and taking the larger of the two caps, gives a bound that holds with probability $1 - \beta$.

::: {#thm-audit-bound name="A certified lower bound on ε"}
Suppose the $2n$ trials are independent and the threshold $\tau$ was chosen without looking at them. Let $\overline{\mathrm{FPR}}$ and $\underline{\mathrm{TPR}}$ be the one-sided Clopper–Pearson bounds at level $\beta/2$, and let
$$\varepsilon_{\mathrm{lo}} = \max\Big\{\ln\frac{\underline{\mathrm{TPR}} - \delta}{\overline{\mathrm{FPR}}},\ \ln\frac{1 - \overline{\mathrm{FPR}} - \delta}{1 - \underline{\mathrm{TPR}}}\Big\}.$$ {#eq-audit-eps-lo}
Then, with probability at least $1 - \beta$, $M$ is not $(\varepsilon, \delta)$-DP for any $\varepsilon < \varepsilon_{\mathrm{lo}}$.
:::

::: {.proof}
The plan is to solve the two caps for $\varepsilon$, check that pessimistic rates can only lower the result, and then bound the probability that the rates are not pessimistic.

1. *The true rates bound $\varepsilon$.* Dividing each inequality of @eq-audit-region by the factor in front of $e^\varepsilon$ gives
$$e^\varepsilon \ge \frac{\mathrm{TPR} - \delta}{\mathrm{FPR}}, \qquad e^\varepsilon \ge \frac{1 - \mathrm{FPR} - \delta}{1 - \mathrm{TPR}}.$$
2. *Pessimistic rates lower both fractions.* Both fractions grow with the TPR and shrink with the FPR. So a larger FPR and a smaller TPR can only lower them.
3. *The two bounds hold together with probability $1 - \beta$.* Each bound fails with probability at most $\beta/2$, so by the union bound at least one fails with probability at most $\beta$. This needs binomial counts, which is why the trials must be independent and the threshold fixed before the counting.
4. *Combine.* When both bounds hold, step 2 says that $\varepsilon_{\mathrm{lo}}$ is at most the larger logarithm of the two fractions at the true rates. Step 1 says that this is at most the $\varepsilon$ of any valid guarantee. So, over the randomness of the trials, the bound exceeds the $\varepsilon$ of a valid guarantee with probability at most $\beta$.
:::

The bound says that no smaller $\varepsilon$ is possible, never that this $\varepsilon$ is enough. A weak attack gives a small $\varepsilon_{\mathrm{lo}}$ for a leaky model as easily as for a private one. So a bound above the claim refutes it, while a bound below it says only that this attacker did no better. The second form in the maximum rewards tests that rarely miss present canaries, and [Exercise 1](../../book-exercises/part05-dp-ml/10-auditing.md#sec-sol-audit-1) shows an attack for which only that form is informative.

The hypothesis on the threshold matters in practice. Sweeping the threshold over the same trials and reporting the best bound makes the confidence level wrong, because among many thresholds one is lucky. Every audit in this chapter chooses its threshold on half of the trials and counts on the other half.

Even an attack that never errs certifies only a limited $\varepsilon$, because the bound uses the pessimistic end of each interval. The next result computes that ceiling, which tells you how many trials to budget.

::: {#cor-audit-ceiling name="The ceiling of an audit"}
Under the hypotheses of @thm-audit-bound, the largest bound that $n$ trials per hypothesis can certify is that of a perfect attack, with $k_0 = 0$ and $k_1 = n$:
$$\varepsilon_{\max}(n) = \ln\frac{(\beta/2)^{1/n} - \delta}{1 - (\beta/2)^{1/n}} \approx \ln\frac{n}{\ln(2/\beta)}.$$ {#eq-audit-ceiling}
:::

::: {.proof}
1. *A perfect attack gives the best ends.* Fewer successes push both Clopper–Pearson ends down, so $k_0 = 0$ gives the smallest $\overline{\mathrm{FPR}}$ and $k_1 = n$ the largest $\underline{\mathrm{TPR}}$. By step 2 of the proof of @thm-audit-bound, these give the largest bound.
2. *Solve for the ends.* With $k_0 = 0$ the upper bound solves $(1 - \overline{\mathrm{FPR}})^n = \beta/2$, so $\overline{\mathrm{FPR}} = 1 - (\beta/2)^{1/n}$. With $k_1 = n$ the lower bound solves $\underline{\mathrm{TPR}}^n = \beta/2$, so $\underline{\mathrm{TPR}} = (\beta/2)^{1/n}$. Substituting into either form of @eq-audit-eps-lo gives the exact expression.
3. *Approximate.* Write the power as an exponential, $(\beta/2)^{1/n} = e^{-\ln(2/\beta)/n}$, which is about $1 - \ln(2/\beta)/n$ for large $n$. So $\overline{\mathrm{FPR}}$ is about $\ln(2/\beta)/n$ while $\underline{\mathrm{TPR}}$ is about 1, and their ratio is about $n/\ln(2/\beta)$. With $\beta = 0.05$, $\ln(2/\beta) = \ln 40 = 3.69$.
:::

So each factor of $e$ in the bound costs a factor of $e$ in trials. The reason is that a false positive rate cannot be told apart from zero with fewer than about $3/\mathrm{FPR}$ trials, the **rule of three**. For example, with no false positive in 3,000 trials the 95% upper bound is $9.9808 \times 10^{-4}$. @tbl-audit-ceiling shows what that means. Certifying the running example's 6.71 needs at least 3,038 trials per hypothesis, so 6,076 training runs, even for an attack that never errs. For a real network each run is a full training job.

| Trials per hypothesis $n$ | 100 | 1,000 | 10,000 | 100,000 | 1,000,000 |
| --- | --- | --- | --- | --- | --- |
| Largest certifiable $\varepsilon$, @eq-audit-ceiling | 3.28 | 5.60 | 7.90 | 10.21 | 12.51 |

: The most a perfect attack can certify with $n$ trials per hypothesis, at 95% confidence and $\delta = 10^{-5}$. Conversely, certifying $\varepsilon = 1$, 2, 4, 6.71 and 8 takes at least 12, 30, 204, 3,038 and 10,999 trials per hypothesis. {#tbl-audit-ceiling}

::: {#exm-audit-confidence}
Suppose 1,000 runs with the canary and 1,000 without give 500 true positives and 1 false positive. The point estimate is $\ln\frac{0.5 - 10^{-5}}{0.001} = 6.21$. The Clopper–Pearson bounds at $\beta/2 = 0.025$ are $\overline{\mathrm{FPR}} = 0.00556$ and $\underline{\mathrm{TPR}} = 0.4685$, so one false positive in 1,000 is consistent with a true FPR five times larger. The certified bound is $\ln\frac{0.4685 - 10^{-5}}{0.00556} = 4.43$, almost two less than the point estimate. With ten times the trials and the same rates, the bounds are $\overline{\mathrm{FPR}} = 0.00184$ and $\underline{\mathrm{TPR}} = 0.4902$, and the certified bound rises to 5.59. The price of confidence falls, slowly, as the counts grow.
:::

## Canaries and threat models {#sec-audit-canaries}

A bound is only as strong as the attack behind it, and DP is a promise about the worst pair of datasets and the worst test. A typical training record scored by its loss gives a valid but nearly useless bound, since [Membership inference](../part02-attacks/02-membership-inference.md) shows that most records are barely exposed. This section asks which canary and which view of training make the strongest audit of DP-SGD, and what that audit faces.

::: {#def-audit-canaries name="Canaries and threat models"}
An **input-space canary** is a record $(x, y)$ inserted into the training data like any other, such as a mislabelled example, chosen so that its gradient is large and points where few others do. A **gradient-space canary** is a gradient the auditor writes directly, as if it controlled the training code. A **Dirac canary** (Nasr et al. 2023) is a gradient-space canary equal to $C$ on one coordinate and zero elsewhere, on a coordinate where the other records' gradients are zero. In the **black-box** threat model the attacker sees only the final model, and in the **white-box** model it sees every iterate $\theta_0, \dots, \theta_T$.
:::

A white-box gradient audit tests the *mechanism*, that is, whether the code clips and adds the noise the proof assumes. A black-box input audit tests the *deployment*, what an attacker with a released model can learn. A white-box attacker can recover every step's noisy gradient sum $G_t$, which the lemma below defines. DP-SGD moves the weights by the learning rate $\eta$ times $G_t$ divided by the expected batch size $B$, so $G_t = -B(\theta_{t+1} - \theta_t)/\eta$. The proof of the next theorem needs to know what this sum looks like along one direction, and the lemma below works it out.

::: {#lem-audit-step name="One DP-SGD step seen along one direction"}
Let one DP-SGD step release $G_t = \sum_i\bar g_i + \xi_t$, the sum of the clipped gradients $\bar g_i$ of the records sampled at step $t$ plus noise $\xi_t \sim \mathcal N(0, \sigma^2C^2I)$, with Poisson sampling at rate $q$. Let $u$ be a fixed unit vector. Suppose one record's clipped gradient is exactly $Cu$ at every step, while no other record's gradient has a component along $u$. Then the projection $x_t = u^\top G_t/C$ satisfies
$$x_t \sim \mathcal N(0, \sigma^2) \text{ without the record}, \qquad x_t \sim (1 - q)\,\mathcal N(0, \sigma^2) + q\,\mathcal N(1, \sigma^2) \text{ with it},$$ {#eq-audit-step}
independently over the steps.
:::

::: {.proof}
1. *The noise and the other records.* The projection of an isotropic Gaussian vector on a unit vector is a one-dimensional Gaussian with the same standard deviation, so $u^\top\xi_t/C$ is $\mathcal N(0, \sigma^2)$. The other records add nothing, because their gradients have no component along $u$.
2. *The record.* Its gradient $Cu$ adds exactly 1 to $x_t$ when it is sampled, which happens with probability $q$, independently of the noise.
3. *Independence.* The sampling coin and the noise are drawn afresh at every step, and the record's contribution does not depend on the weights, so different steps are independent.
:::

This is the pair that the accountant of [Amplification by subsampling](../part04-privacy-accounting/05-subsampling.md) analyses, the worst case for one step under add/remove neighbours. A Dirac canary meets the lemma's hypotheses exactly. The strength of a test against this pair is measured on the scale of **Gaussian differential privacy (GDP)**. A mechanism is **$\mu$-GDP** (@def-gdp-gdp) when no test separates its neighbours better than one separates two unit-variance normals whose means are $\mu$ apart. That pair's best true positive rate at a given false positive rate is
$$\mathrm{TPR}_\mu(\mathrm{FPR}) = 1 - \Phi\big(\Phi^{-1}(1 - \mathrm{FPR}) - \mu\big),$$ {#eq-audit-gdp}
where $\Phi$ is the standard normal distribution function.

::: {#thm-audit-dirac name="The white-box Dirac audit"}
Let a Dirac canary sit on a coordinate $j$ that no other record's gradient touches, let the attacker see every iterate, and let $x_t = G_{t,j}/C$. Then $x_1, \dots, x_T$ follow @eq-audit-step independently, and the most powerful test at every FPR thresholds the **likelihood-ratio statistic**
$$\Lambda = \sum_{t=1}^T\ln\Big(1 - q + q\,e^{(x_t - 1/2)/\sigma^2}\Big).$$ {#eq-audit-lr}
No attacker who sees the whole training run does better. For large $T$ with $q\sqrt T$ fixed, the trade-off curve of $\Lambda$ approaches that of $\mu$-GDP with $\mu = q\sqrt{T(e^{1/\sigma^2} - 1)}$. The **sum of projections** $\sum_tx_t$ has a curve close to $\mu$-GDP with $\mu_{\mathrm{sum}} = q\sqrt T/\sigma$.
:::

::: {.proof}
The plan is to show that the projections follow the lemma's pair, to multiply the steps' likelihood ratios, to check that the rest of the run adds no evidence, and then to read off both statistics' strength.

1. *The canary is the only difference.* Every other record contributes 0 to coordinate $j$. The canary's gradient has norm exactly $C$, so clipping leaves it unchanged, and it is the same at every step whatever the weights are. So @lem-audit-step applies with $u$ the unit vector of coordinate $j$.
2. *The likelihood ratio of one step.* The squares $x^2$ cancel in the ratio of two normal densities with the same variance, so $\mathcal N(1, \sigma^2)$ over $\mathcal N(0, \sigma^2)$ at $x$ is $e^{(x - 1/2)/\sigma^2}$. The mixture puts weight $1 - q$ on the null density and $q$ on the shifted one, so its ratio to the null is
$$1 - q + q\,e^{(x - 1/2)/\sigma^2}.$$
3. *Multiply over the steps.* Independence makes the joint likelihood ratio the product of the per-step ratios, and its logarithm is @eq-audit-lr. By the Neyman–Pearson lemma (@thm-ht-np), thresholding it is the most powerful test at every FPR.
4. *The rest of the run carries no evidence.* The whole run is a function of the $x_t$ and of randomness with the same distribution under both hypotheses, namely the other records' sampling and the noise on the other coordinates. The weights depend on past $x_t$, but given those, everything else is drawn the same way with or without the canary.
5. *The two statistics in Gaussian terms.* For $\Lambda$, the subsampled Gaussian central limit theorem (CLT), @thm-gdp-sgd in [f-DP and Gaussian DP](../part04-privacy-accounting/04-gaussian-dp.md), gives the curve of $T$ steps of this pair, which the most powerful test attains. For the sum, the mean shifts by the expected number of times the canary is sampled, $qT$, while the noise has standard deviation $\sigma\sqrt T$, and their ratio is $q\sqrt T/\sigma$. With the canary the sum also has a little extra variance, $Tq(1 - q)$.
:::

The likelihood ratio beats the sum because it weighs each step by its evidence. A step with $x_t = 3$ is much stronger evidence than three steps with $x_t = 1$, since a value three noise standard deviations out is very unlikely without the canary. The ratio of the two $\mu$ is $\sigma\sqrt{e^{1/\sigma^2} - 1}$, which is 1.311 at $\sigma = 1$ and tends to 1 as $\sigma$ grows ([Exercise 4](../../book-exercises/part05-dp-ml/10-auditing.md#sec-sol-audit-4)).

Black-box audits with input canaries are weaker for three reasons. The final model is one function of all the noisy sums, so by post-processing it carries at most as much information as they do, and usually much less. An input canary's gradient shrinks or turns once the model fits it or gives up on it. And it shares directions with the other records' gradients, which act as extra noise. Nasr, Song, Thakurta, Papernot & Carlini (2021) built a ladder of adversaries for DP-SGD and found the bound tight only for the strongest, which sees every iterate and chooses a worst-case dataset and canary.

::: {#exm-audit-statistics}
At $q = 0.01$, $\sigma = 1$ and $T = 10{,}000$, the likelihood-ratio statistic has $\mu = 0.01 \times \sqrt{10{,}000 \times 1.7183} = 1.3108$. The sum of projections has $\mu_{\mathrm{sum}} = 0.01 \times 100 = 1.00$, or 0.9951 with the extra variance. The Gaussian privacy profile of @cor-gdp-profile converts these to $\varepsilon = 6.01$ and $4.35$ at $\delta = 10^{-5}$. So choosing the sum lowers the $\varepsilon$ the audit can ever reach from 6.01 to 4.35, before a single trial is run.
:::

## Tight audits of the mechanism {#sec-audit-tight}

The white-box Dirac audit faces the accountant's own pair. This section asks whether, with enough trials, it recovers the accountant's 6.71, and how close it gets with the trials an auditor can afford. The answer explains why audits stop short of their claims.

The smallest $\varepsilon$ that a pair of output distributions allows can be read straight off the definition. Only likelihood-ratio thresholds need to be checked, because the Neyman–Pearson lemma says they are the best tests.

::: {#thm-audit-exact name="A perfect audit measures the true ε"}
Let $P$ and $Q$ be the distributions of the training output without and with the canary. The smallest $\varepsilon$ such that $Q(S) \le e^\varepsilon P(S) + \delta$ for every event $S$ is
$$\varepsilon^*(\delta) = \sup_S\ \ln\frac{Q(S) - \delta}{P(S)},$$ {#eq-audit-sup}
the supremum over events with $Q(S) > \delta$, and it may be taken over likelihood-ratio thresholds only. So an audit that thresholds the likelihood ratio, and knows its rates exactly, computes $\varepsilon^*(\delta)$.
:::

::: {.proof}
1. *The supremum is the definition solved for $\varepsilon$.* If $\varepsilon$ is at least the supremum, every event's log-ratio is at most $\varepsilon$, which rearranges to $Q(S) \le e^\varepsilon P(S) + \delta$. If $\varepsilon$ is below it, some event violates the inequality.
2. *Only likelihood-ratio thresholds matter.* For a fixed $P(S)$, the ratio grows with $Q(S)$. By the Neyman–Pearson lemma (@thm-ht-np), among events with a given $P(S)$ the one with the largest $Q(S)$ is a likelihood-ratio threshold, randomized on ties.
3. *A perfect audit.* Reading off exact rates at every likelihood-ratio threshold evaluates the ratio of step 1 at every candidate of step 2, so it computes the supremum.
:::

For the Dirac canary this pair is the one the accountant bounds, so a perfect audit converges to the true $\varepsilon$ of DP-SGD, not to what an accountant reports. The RDP accountant's 6.71 is a loose upper bound. The numerical accountant of [f-DP and Gaussian DP](../part04-privacy-accounting/04-gaussian-dp.md) puts the true value between 6.14 and 6.24, and the CLT approximation gives 6.01. So even an audit with infinitely many trials stops about half a unit below 6.71, and that gap belongs to the accountant.

Real audits fall much further short, because of where on the curve the supremum is reached. For a $\mu$-GDP curve that point is explicit.

::: {#prp-audit-touch name="Where ε lives on a Gaussian curve"}
Let a mechanism's trade-off curve be the $\mu$-GDP curve @eq-audit-gdp, and let $\varepsilon = \varepsilon^*(\delta)$. The supremum @eq-audit-sup is reached at the threshold whose rates are
$$\mathrm{FPR}^* = \Phi\Big(-\frac{\varepsilon}{\mu} - \frac{\mu}{2}\Big), \qquad \mathrm{TPR}^* = \Phi\Big(-\frac{\varepsilon}{\mu} + \frac{\mu}{2}\Big).$$ {#eq-audit-touch}
:::

::: {.proof}
1. *The supremum is a tangent.* By @eq-audit-sup, $\varepsilon$ is set by the steepest line through $(0, \delta)$ that still meets the concave curve of TPR against FPR. At the optimum that line touches the curve, so the curve's slope there is $e^\varepsilon$.
2. *The slope is the likelihood ratio.* By @prp-ht-slope in [Hypothesis testing](../part01-prerequisites/04-hypothesis-testing.md), the slope at a threshold is the likelihood ratio there. For $\mathcal N(\mu, 1)$ against $\mathcal N(0, 1)$ at threshold $\tau$ it is $e^{\mu\tau - \mu^2/2}$.
3. *Solve for the threshold.* Setting the slope to $e^\varepsilon$ gives $\tau = \varepsilon/\mu + \mu/2$. Its FPR is $1 - \Phi(\tau) = \Phi(-\tau)$, and its TPR is $1 - \Phi(\tau - \mu) = \Phi(\mu - \tau)$.
:::

For the running example, $\mu = 1.3108$ and $\varepsilon = 6.01$ give $\mathrm{FPR}^* = 8.1 \times 10^{-8}$ and $\mathrm{TPR}^* = 4.3 \times 10^{-5}$. So certifying the full $\varepsilon$ at $\delta = 10^{-5}$ means measuring a false positive rate below one in ten million. With $n$ trials you can measure rates down to about $3/n$, and there the curve's log-ratio of TPR to FPR is far below $\varepsilon$. For $\mu = 1.3108$, @eq-audit-gdp gives 2.74 at FPR $10^{-2}$, 3.63 at $10^{-3}$ and 4.38 at $10^{-4}$. Seeing ten false positives near $10^{-7}$ takes about $1.2\times10^{8}$ trials without the canary ([Exercise 3](../../book-exercises/part05-dp-ml/10-auditing.md#sec-sol-audit-3)). @fig-audit-region shows the measured curve well inside the $(6.71, 10^{-5})$ limit.

![Left: the trade-off of the white-box audit of the running example on log axes, from 30,000 trials per hypothesis. The black line is the most any test can achieve under $(6.71, 10^{-5})$-DP; the dashed grey curve is $\mu$-GDP with the CLT's $\mu = 1.31$. The blue curve is the likelihood-ratio statistic, the orange one the sum of projections; boxes are 95% Clopper–Pearson regions at four thresholds. Right: the $\varepsilon$ a threshold would certify if its rates were known exactly, $\ln((\mathrm{TPR} - \delta)/\mathrm{FPR})$, against its false positive rate. The Gaussian curve reaches its $\varepsilon$ of 6.01 only at FPR $8 \times 10^{-8}$ (dotted line), far below the rates a feasible audit can measure.](figures/audit_region.svg){#fig-audit-region}

Nasr, Hayes, Steinke, Balle, Tramèr, Jagielski, Carlini & Terzis (2023) avoid the tail by assuming the curve's shape. If the mechanism is $\mu$-GDP, one test at a moderate FPR, where trials are cheap, bounds $\mu$ from below, and the Gaussian shape carries that bound into the tail.

::: {#prp-audit-mu name="A lower bound on μ from one test"}
If $M$ is $\mu$-GDP, every test satisfies
$$\mu \ge \Phi^{-1}(\mathrm{TPR}) - \Phi^{-1}(\mathrm{FPR}).$$ {#eq-audit-mu}
Under the hypotheses of @thm-audit-bound, the value $\mu_{\mathrm{lo}}$ of the right-hand side at $\underline{\mathrm{TPR}}$ and $\overline{\mathrm{FPR}}$ satisfies $\mu \ge \mu_{\mathrm{lo}}$ with probability at least $1 - \beta$.
:::

::: {.proof}
1. *Rearrange the curve.* Under $\mu$-GDP every test's TPR is at most @eq-audit-gdp. The normal distribution is symmetric, so $1 - \Phi(a) = \Phi(-a)$ and $\Phi^{-1}(1 - p) = -\Phi^{-1}(p)$, and the cap becomes
$$\mathrm{TPR} \le \Phi\big(\Phi^{-1}(\mathrm{FPR}) + \mu\big).$$
Applying the increasing function $\Phi^{-1}$ to both sides gives @eq-audit-mu.
2. *Pessimistic rates.* The right-hand side of @eq-audit-mu grows with the TPR and shrinks with the FPR, so the Clopper–Pearson ends can only lower it. They hold together with probability at least $1 - \beta$, as in step 3 of the proof of @thm-audit-bound.
:::

The bound on $\mu$ is rigorous, but its conversion to $\varepsilon$ is valid only if the curve really is Gaussian. The subsampled Gaussian of DP-SGD has a heavier tail, so the $\mu$ that @eq-audit-mu gives depends on where you read it. In the example below, the measured rates give 1.300 at FPR $10^{-1}$, 1.303 at $10^{-2}$, 1.371 at $10^{-3}$ and 1.492 at $2 \times 10^{-4}$, against the CLT's 1.3108. For this reason Nasr et al. mostly insert the canary at every step, which makes each step an exact Gaussian mechanism, as @sec-audit-bugs does.

::: {#exm-audit-tight}
The script simulates the white-box Dirac audit directly from @eq-audit-step. Each trial draws 10,000 projections, adds the canary on a $\mathrm{Bin}(10{,}000, 0.01)$ number of steps when it is present, and computes $\Lambda$. By @thm-audit-dirac this is statistically identical to training the network with a Dirac canary, and thousands of times cheaper.

With 50,000 trials per hypothesis, the measured curve is heavier-tailed than the Gaussian. At FPR $10^{-2}$ the point value of $\ln((\mathrm{TPR} - \delta)/\mathrm{FPR})$ is 2.73, as the Gaussian predicts, but at $10^{-4}$ it is 4.84, above the Gaussian's 4.38. @tbl-audit-tight gives the certified bounds, with thresholds chosen on the first half of the trials and counts taken on the second.

| Trials per hypothesis | 1,000 | 3,000 | 10,000 | 50,000 |
| --- | --- | --- | --- | --- |
| $\varepsilon_{\mathrm{lo}}$, @eq-audit-eps-lo | 1.64 | 1.88 | 2.84 | 3.63 |
| $\mu_{\mathrm{lo}}$, @eq-audit-mu with Clopper–Pearson | 0.795 | 1.064 | 1.147 | 1.185 |
| $\varepsilon$ of $\mu_{\mathrm{lo}}$-GDP (assumes the Gaussian shape) | 3.36 | 4.70 | 5.13 | 5.33 |
| Ceiling, @eq-audit-ceiling, at half the trials | 4.91 | 6.01 | 7.21 | 8.82 |

: The white-box Dirac audit of the running example with the likelihood-ratio statistic, against the accountant's $\varepsilon = 6.71$. All bounds hold with probability 95%. {#tbl-audit-tight}

The rigorous bound reaches 3.63 with 50,000 trials per hypothesis, about half the claim, while the Gaussian-shape estimate climbs to 5.33. The sum of projections needs far more trials for less, 1.97 with 10,000 trials per hypothesis and 3.26 with 100,000. With 1,000,000 it falls to 2.27 in this run, because bounds read in the far tail are noisy, as @fig-audit-trials shows. Every number in @tbl-audit-tight is below the claim, as it must be for a correct implementation.
:::

![The certified $\varepsilon$ lower bound of the white-box audit of the running example against the number of trials per hypothesis, at 95% confidence and $\delta = 10^{-5}$: the likelihood-ratio statistic (circles), the sum of projections (triangles) and the Gaussian-shape estimate from the likelihood ratio (squares), with the accountant's 6.71 and the ceiling of a perfect attack. The bounds grow roughly like the logarithm of the number of trials.](figures/audit_trials.svg){#fig-audit-trials}

## One-run auditing {#sec-audit-onerun}

The audits so far need one training run per trial, thousands in all, which is out of the question for a model that takes a day to train. Can a single training run give a valid bound? It can, if each canary is a trial.

::: {#def-audit-onerun name="One-run audit"}
Insert $m$ canaries, each independently with probability $\frac12$, and write $S_i = +1$ if canary $i$ is in the training set and $S_i = -1$ if not. Train once and score every canary. Guess $\hat S_i = +1$ ("in") for the $r/2$ highest scores and $\hat S_i = -1$ ("out") for the $r/2$ lowest, and abstain, $\hat S_i = 0$, on the rest. Let $W$ be the number of correct guesses among the $r$ made.
:::

Each guess is about one fair coin, made from an output that is private in that coin. DP lets the output make one value of the coin at most $e^\varepsilon$ times as likely as the other, so no guess can be right more often than $e^\varepsilon/(1 + e^\varepsilon)$. Steinke, Nasr & Jagielski (2023) turn that into a bound on the total count.

::: {#thm-audit-onerun name="One-run auditing under pure DP"}
If training is $\varepsilon$-DP, then for every $v$
$$\Pr[W \ge v] \le \Pr\big[\mathrm{Bin}(r, p_\varepsilon) \ge v\big], \qquad p_\varepsilon = \frac{e^\varepsilon}{1 + e^\varepsilon}.$$ {#eq-audit-onerun}
:::

::: {.proof}
The plan is to show that each guess is right with probability at most $p_\varepsilon$, even given the earlier guesses, and then to compare the count with a binomial.

1. *Each inclusion bit is protected.* Fix every inclusion bit except $S_i$. Changing $S_i$ adds or removes one record, so the run is $\varepsilon$-DP as a function of $S_i$, and by post-processing (@thm-dpprop-post) so are the guesses. For every vector of guesses $s$,
$$\Pr[\hat S = s \mid S_i = +1, S_{-i}] \le e^\varepsilon\Pr[\hat S = s \mid S_i = -1, S_{-i}].$$
2. *Average over the later bits.* Order the guessed canaries in a fixed order, and condition only on the earlier bits $S_{<i}$. The later bits are independent of $S_i$, so averaging step 1 over them puts the same weights on both sides, and the inequality holds given $S_{<i}$.
3. *No guess is right more than $p_\varepsilon$ of the time.* Given $S_{<i}$, write $a$ for the probability of $s$ when $S_i = +1$ and $b$ for it when $S_i = -1$. With the prior $\frac12$ on $S_i$, Bayes' rule gives $\Pr[S_i = +1 \mid \hat S = s, S_{<i}] = a/(a + b)$. The fraction grows with $a$, and step 2 says $a \le e^\varepsilon b$, so
$$\frac{a}{a + b} \le \frac{e^\varepsilon b}{e^\varepsilon b + b} = p_\varepsilon.$$
The same holds for $S_i = -1$. The earlier bits fix whether the earlier guesses were right, so each guess is right with probability at most $p_\varepsilon$ given the earlier outcomes.
4. *The count is dominated by a binomial.* Write the $j$-th guess's correctness as $\mathbf 1[U_j \le p_j]$, with $U_j$ independent uniform numbers and $p_j \le p_\varepsilon$ its conditional probability from step 3. Then $\mathbf 1[U_j \le p_j] \le \mathbf 1[U_j \le p_\varepsilon]$, and the right-hand indicators sum to a $\mathrm{Bin}(r, p_\varepsilon)$ count, so @eq-audit-onerun follows. @sec-apx-audit-onerun in the appendices builds this coupling step by step and proves the domination lemma it relies on.
:::

Under $(\varepsilon, \delta)$-DP each guess can be right a little more often. Write $F(v) = \Pr[\mathrm{Bin}(r, p_\varepsilon) \ge v]$ for the binomial tail of @eq-audit-onerun.

::: {#thm-audit-onerun-delta name="One-run auditing under approximate DP"}
If training is $(\varepsilon, \delta)$-DP, then for every $v$
$$\Pr[W \ge v] \le F(v) + 2m\delta\max_{i = 1, \dots, m}\frac{F(v - i) - F(v)}{i}.$$ {#eq-audit-onerun-delta}
:::
<!-- proof: cited -->

The proof is Steinke, Nasr & Jagielski (2023, Corollary 5.4), which tracks how the additive $\delta$ of step 1 accumulates over the guesses. Either theorem gives an audit in the same way. Observing $v$ correct guesses rejects every $\varepsilon$ whose right-hand side is below $\beta$, and the largest such $\varepsilon$ is the bound.

The binomial is exactly the count of $\varepsilon$-randomized response on the true bits (@thm-dpdef-rr), so the bound is tight for that mechanism ([Exercise 6](../../book-exercises/part05-dp-ml/10-auditing.md#sec-sol-audit-6)). In the paper's worked example, 75 correct guesses out of 100 give the lower bound 0.702 at 95% confidence. The script reproduces it, and the paper's 0.699 and 0.673 for $\delta = 10^{-4}$ with 100 and 1,000 canaries. The ceiling is again logarithmic, about $\ln(r/\ln(1/\beta))$ when all guesses are right, with $\ln(1/\beta)$ because the test is one-sided. That is 3.49 for 100 canaries, 5.81 for 1,000 and 8.11 for 10,000 at $\delta = 0$, and certifying 6.71 needs about 2,500 perfect guesses ([Exercise 5](../../book-exercises/part05-dp-ml/10-auditing.md#sec-sol-audit-5)). So a one-run audit saves runs, not trials.

The $\delta$ term grows with the number of canaries, since the correction is at most $2m\delta$. With 10,000 canaries all guessed right it lowers the bound from 8.11 to 7.83, and when $2m\delta$ approaches $\beta$ it swallows the whole failure probability. Even with white-box Dirac canaries, one run is not tight for DP-SGD. Each canary is guessed once, so the audit reads the trade-off curve near FPR $1/m$ at best. And the binomial is calibrated to randomized response, whose curve has a sharp corner, while the curve of DP-SGD is smooth.

::: {#exm-audit-onerun}
Simulate one run of the running example with 1,000 Dirac canaries on distinct untouched coordinates, about half of them included, each scored by $\Lambda$. The auditor tries six list lengths $r$ and pays for trying them with a Bonferroni correction. Each test runs at level $\beta/6$, so that the chance that any of the six is wrong is at most $\beta$. @tbl-audit-onerun gives the result.

| Guesses $r$ | 20 | 50 | 100 | 200 | 500 | 1,000 |
| --- | --- | --- | --- | --- | --- | --- |
| Correct, white-box Dirac canaries | 20 | 50 | 92 | 185 | 425 | 745 |
| $\varepsilon_{\mathrm{lo}}$ (Bonferroni over the six) | 1.21 | 2.22 | 1.58 | 1.88 | 1.43 | 0.90 |

: One-run audit of the running example's mechanism with 1,000 white-box Dirac canaries, at 95% overall confidence and $\delta = 10^{-5}$. {#tbl-audit-onerun}

All 50 most confident guesses are right, which certifies 2.22, while the 100-guess list makes 8 mistakes and certifies less. With 5,000 canaries the best bound is 2.53, from 479 correct out of 500. There the correction $2m\delta = 0.1$ exceeds the per-test level $\beta/8$ of its eight list lengths, and 20 correct guesses out of 20 certify nothing at all. One run certifies 2.22, against 2.84 for 10,000 runs per hypothesis in @tbl-audit-tight, at a twenty-thousandth of the cost.
:::

## Black-box audit of the trained network {#sec-audit-blackbox}

The white-box Dirac audit tests the mechanism. What does an attacker learn from the released network itself, with canaries planted as ordinary records? This section runs the one-run audit of @sec-audit-onerun, so each audit costs one training run. The canaries must be records the model can learn only by memorizing them, since a correctly labelled record has nearly the same loss whether or not it was trained on.

::: {#def-audit-blackbox name="Black-box audit with mislabelled canaries"}
Take 1,000 records from the held-out test set of the running example and flip their labels, which makes them **mislabelled canaries**. Include each in the 10,000 training records with probability $\frac12$, train once with DP-SGD at the claimed configuration, and score each canary by minus its loss on the final model. Guess with the six list lengths of @exm-audit-onerun, and bound $\varepsilon$ with @eq-audit-onerun-delta, Bonferroni-corrected. As controls, audit a model trained with clipping but no noise, and one trained without clipping or noise.
:::

The bound is valid, because including a canary is adding a record, and DP-SGD's guarantee holds for every dataset. But it should be far below 6.71. The attacker sees one function of the 10,000 noisy sums, the canary's gradient changes as it is fitted, and the 10,000 real records pull the other way. The no-noise control separates the attacker's weakness from the noise. A model trained without noise has no finite $\varepsilon$, so if its audit also gives a small bound, the audit is what keeps the bound small.

::: {#exm-audit-blackbox}
@tbl-audit-blackbox gives the three audits, one training run each.

| Training | Test accuracy | Mean loss, in | Mean loss, out | Correct of $r = 100$ | Correct of $r = 1{,}000$ | Best $\varepsilon_{\mathrm{lo}}$ |
| --- | --- | --- | --- | --- | --- | --- |
| DP-SGD, $\sigma = 1$, claim $\varepsilon = 6.71$ | 0.9645 | 5.565 | 6.013 | 59 | 556 | 0.13 |
| Clipping only, $\sigma = 0$, $\varepsilon = \infty$ | 0.9638 | 5.511 | 6.155 | 67 | 589 | 0.24 |
| Non-private, no clipping | 0.9155 | 0.023 | 9.041 | 91 | 967 | 2.96 |

: Black-box one-run audit of the running example's network with 1,000 mislabelled canaries, one training run each, at 95% overall confidence and $\delta = 10^{-5}$. Test accuracy is on the 19,000 test records not used as canaries. {#tbl-audit-blackbox}

The non-private model memorizes the canaries it trained on. Their mean loss is 0.023, against 9.041 for those it did not see, so 967 of 1,000 guesses are right and the audit certifies 2.96. A thousand perfect guesses would certify 5.78 even without the Bonferroni correction, so 33 mistakes cost almost three units. Memorizing also costs almost five points of test accuracy, 0.9155 against 0.9645.

DP-SGD does not memorize. The mean losses of included and excluded canaries differ by less than half a unit, 59 of the 100 most confident guesses are right, and the certified bound is 0.13, from 293 of 500. The clipping-only model is barely easier, at 0.24. Clipping alone bounds each canary's pull on the weights to $C$ per step, against the combined pull of 10,000 correctly labelled records, and that is what defeats this attacker. It does not make clipping private, since a white-box Dirac canary would identify membership in that model perfectly. @fig-audit-one-run shows the loss distributions, and how the fraction of correct guesses falls as the list grows.
:::

![One-run audits with 1,000 canaries. Left and middle: the loss of each mislabelled canary on the final model, for canaries in the training set (red) and left out (blue), without privacy and with DP-SGD at $\varepsilon = 6.71$. Right: the fraction of the $r$ guesses that are correct, against $r$, for the non-private network (black box), the DP-SGD network (black box) and the mechanism with white-box Dirac canaries; the dashed line is the most a $6.71$-DP algorithm allows on average, $e^\varepsilon/(1 + e^\varepsilon)$.](figures/audit_one_run.svg){#fig-audit-one-run}

Nothing here contradicts 6.71 or confirms it. The black-box bound of 0.13 says that a realistic attacker with one model extracts very little, while the white-box audits of @sec-audit-tight say that an attacker who sees every step extracts much more. An honest privacy statement reports both, with the threat model of each, and [Choosing an accountant](../part04-privacy-accounting/09-choosing-an-accountant.md) lists what else it must contain.

## Finding bugs {#sec-audit-bugs}

An audit bound below the claim proves nothing, but a bound above it proves, with confidence $1 - \beta$, that the claim is false. That makes audits a test suite for private training code, which passes silently on a correct implementation and fails loudly on a broken one. This section asks which bugs such a test catches, and how cheaply.

::: {#def-audit-perstep name="Per-step and end-to-end audits"}
A **per-step audit** inserts a Dirac canary at every step in one run and leaves it out in another, and treats each step's projection $x_t$ as one trial, so it gets $T$ trials per hypothesis from two runs. An **end-to-end audit** runs the many-trial protocol of @sec-audit-tight, with one training run per trial.
:::

With the canary always present there is no sampling coin to blur it, so each step of a correct implementation is a plain Gaussian mechanism. Its privacy is then known exactly, and a measured $\mu$ above it is evidence of a bug.

::: {#prp-audit-perstep name="A per-step audit tests the noise"}
In a per-step audit of a correct implementation, the projections are independent across steps, with
$$x_t \sim \mathcal N(0, \sigma^2) \text{ without the canary}, \qquad x_t \sim \mathcal N(1, \sigma^2) \text{ with it},$$ {#eq-audit-perstep}
so each step is exactly $\mu$-GDP with $\mu = 1/\sigma$. If the bound $\mu_{\mathrm{lo}}$ of @prp-audit-mu, computed from these trials with a threshold chosen on other trials, exceeds $1/\sigma$, the implementation does not add the noise it claims, with confidence $1 - \beta$.
:::

::: {.proof}
1. *One step is a Gaussian mechanism.* With the canary in every batch, @lem-audit-step with $q = 1$ gives @eq-audit-perstep. The means differ by 1 in noise of standard deviation $\sigma$, so by @thm-ht-gaussian the trade-off curve is @eq-audit-gdp with $\mu = 1/\sigma$.
2. *The steps are independent trials.* The noise is fresh at every step and the canary's gradient is fixed, so the counts at a fixed threshold are binomial and @prp-audit-mu applies.
3. *A refutation.* By @prp-audit-mu, a correct implementation gives $\mu_{\mathrm{lo}} \le 1/\sigma$ with probability at least $1 - \beta$. So observing $\mu_{\mathrm{lo}} > 1/\sigma$ rejects, at level $\beta$, the hypothesis that the code adds noise $\sigma C$ to a sum with sensitivity $C$.
:::

A per-step audit tests the noise and the clipping, but it never exercises the sampling code, and it compares the measured noise with the claimed $\sigma$, not with the claimed $\varepsilon$. So a bug in how batches are drawn, or in what the accountant is told, passes it. Nasr et al. (2023) used per-step audits to catch clipping applied after averaging, noise drawn from a seed with only 100 possible values, and a wrong noise scale. An end-to-end audit tests the claimed $\varepsilon$ directly, so it catches an accounting bug as readily as a noise bug, but only when the true $\varepsilon$ exceeds the claim by more than the audit's shortfall. For the running example that shortfall is about three units at 100,000 runs. Tramèr, Terzis, Steinke, Song, Jagielski & Carlini (2022) used such an audit to refute the guarantee of an open-source implementation, at a confidence they report as $1 - 10^{-10}$.

No canary audit catches everything, because it tests one dataset, one canary and one attack. Batches drawn by shuffling but accounted as Poisson can have an $\varepsilon$ several times the reported one, by a lower bound in [Amplification by subsampling](../part04-privacy-accounting/05-subsampling.md). Yet a per-step audit never sees the sampling, and an end-to-end audit sees the gap only where it exceeds the shortfall. Audits also catch errors only in the unsafe direction. Noise $\sigma C$ added to the averaged gradient is $B$ times more noise than claimed, so every audit passes, although the bug wastes accuracy. Passing an audit is evidence, never a certificate.

::: {#exm-audit-bugs}
@tbl-audit-bugs audits three implementations of the running example's noise, each with two runs of 10,000 steps.

| Implementation | Noise on the sum | Claimed per-step $\mu$ | Audited $\mu_{\mathrm{lo}}$ | Accountant at $\sigma = 1/\mu_{\mathrm{lo}}$ | Verdict |
| --- | --- | --- | --- | --- | --- |
| Correct | $\sigma C$ | 1 | 0.886 | 5.38 | consistent |
| Variance passed as a standard deviation | $\sigma^2C^2 = 0.5\,C$ | 1 | 1.745 | 29.27 | refuted |
| Noise scaled for the mean, added to the sum | $\sigma C/B = 0.01\,C$ | 1 | 6.359 | over 4,700 | refuted |

: Per-step audits of three implementations of the running example's DP-SGD step, from 10,000 steps with and 10,000 without a Dirac canary, at 95% confidence. The last column is what the RDP accountant would report for the noise the audit measured. {#tbl-audit-bugs}

The correct implementation passes, with 0.886 below the claimed 1. The two noise bugs fail decisively. Passing $\sigma^2C^2 = 0.25$ where the code expects a standard deviation halves the noise when $C = 0.5$, and the audit measures 1.745. Scaling the noise for the averaged gradient, whose sensitivity is $C/B$, and adding it to the sum makes the noise $B = 100$ times too small. Then 10,000 steps separate the hypotheses perfectly, and 6.359 is the most @eq-audit-mu can certify from the 5,000 error-free trials per hypothesis of the evaluation half. The same bug also fails an end-to-end audit with the sum of projections, which with 10,000 runs per hypothesis gives $\varepsilon_{\mathrm{lo}} = 7.21 > 6.71$.

In the accounting bug the code is correct, but the accountant is told that the dataset has 20,000 records instead of 10,000, as can happen when augmented copies are counted. It computes $q = 0.005$ and reports $\varepsilon = 3.06$. The per-step audit measures the noise, which is right, and composing its $\mu_{\mathrm{lo}}$ at the claimed $q = 0.005$ gives 2.47, which does not contradict 3.06. The end-to-end audit of @tbl-audit-tight does, because the code really samples at $q = 0.01$ and its 3.63 exceeds 3.06. @fig-audit-bugs shows why the two noise bugs are easy to see.
:::

![Per-step audits of three implementations: the projections of 10,000 steps on a Dirac canary, without it (blue) and with it at every step (red), in units of $C$. A correct implementation has noise of standard deviation $\sigma C$; the two bugs shrink it by a factor of 2 and of 100. The titles give the certified lower bound on the per-step $\mu$; the claim is 1.](figures/audit_bugs.svg){#fig-audit-bugs}

## Code {#sec-audit-code}

The script [code/auditing.py](../../code/part05-dp-ml/auditing.py) computes every bound of this chapter, simulates the white-box audits from @eq-audit-step, and trains the black-box networks with the model and DP-SGD of [code/dp_sgd.py](../../code/part05-dp-ml/dp_sgd.py). It takes a few minutes to run from the repository root:

```bash
uv run python code/part05-dp-ml/auditing.py
```

| Experiment | Result |
| --- | --- |
| Clopper–Pearson, $k = 10$ of 10,000 | $[0.00048, 0.00184]$, as in Part 1 |
| One-run bound, the paper's examples | 0.702, 0.699 and 0.673, as published |
| Ceiling of a perfect attack | 5.60 with 1,000 trials, 7.90 with 10,000; 3,038 needed for 6.71 |
| White-box Dirac audit, likelihood ratio | $\varepsilon_{\mathrm{lo}} = 3.63$ with 50,000 trials per hypothesis |
| Same, Gaussian-shape estimate | 5.33, against the CLT's 6.01 |
| One run, 1,000 white-box Dirac canaries | 2.22 |
| One run, black box, mislabelled canaries | 0.13 with DP-SGD, 0.24 with clipping only, 2.96 non-private |
| Per-step audits | Correct 0.886 against a claim of 1; bugs 1.745 and 6.359 |

: The script's output with the default seed. {#tbl-audit-code}

[tests/test_auditing.py](../../tests/test_auditing.py) checks that the output still matches.

## Summary {#sec-audit-summary}

| Result | Statement | Use |
| --- | --- | --- |
| @def-audit-audit | Trials on $D$ and $D' = D \cup \{z^*\}$; the attack's FPR and TPR | Every audit |
| @thm-audit-bound | $\varepsilon_{\mathrm{lo}}$ from the Clopper–Pearson ends of both rates, with probability $1 - \beta$ | Choose the threshold on other trials; it refutes a claim, never proves one |
| @cor-audit-ceiling | $\varepsilon_{\max}(n) \approx \ln(n/\ln(2/\beta))$: 5.60 at $n = 1{,}000$ | Budget at least $e^\varepsilon\ln(2/\beta)$ trials per hypothesis, in practice far more |
| @def-audit-canaries | Input, gradient and Dirac canaries; black-box and white-box attackers | Report the threat model with the number |
| @lem-audit-step | $\mathcal N(0, \sigma^2)$ against $(1 - q)\mathcal N(0, \sigma^2) + q\,\mathcal N(1, \sigma^2)$ per step | The pair the accountant analyses |
| @thm-audit-dirac | $\Lambda$ is optimal, with $\mu = q\sqrt{T(e^{1/\sigma^2} - 1)}$; the sum has $q\sqrt T/\sigma$ | Test code with the strongest attack |
| @thm-audit-exact | $\varepsilon^*(\delta) = \sup_S\ln\frac{Q(S) - \delta}{P(S)}$ | A perfect audit finds the true $\varepsilon$, not the accountant's bound |
| @prp-audit-touch | Reached at FPR $\Phi(-\varepsilon/\mu - \mu/2)$, $8.1\times10^{-8}$ here | Why audits at small $\delta$ stop short |
| @prp-audit-mu | $\mu \ge \Phi^{-1}(\mathrm{TPR}) - \Phi^{-1}(\mathrm{FPR})$ | Rigorous for $\mu$; gives $\varepsilon$ only if the curve is Gaussian |
| @def-audit-onerun | $m$ canaries in by fair coins; $W$ correct of $r$ guesses | Auditing one real training run |
| @thm-audit-onerun | $W$ is dominated by $\mathrm{Bin}(r, \frac{e^\varepsilon}{1 + e^\varepsilon})$ | Independent fair coins; tight for randomized response |
| @thm-audit-onerun-delta | The same plus at most $2m\delta$ | Pay for trying several list lengths with Bonferroni |
| @def-audit-blackbox | Mislabelled canaries scored by their loss | A small bound describes the attacker, not the model |
| @def-audit-perstep | Canary at every step; or one run per trial | Audit the noise per step and the claim end to end |
| @prp-audit-perstep | Correct code gives $\mu = 1/\sigma$ per step | A cheap test of noise and clipping, blind to sampling and accounting |

: The results of this chapter, and where each is used. {#tbl-audit-summary}

## References {#sec-audit-references}

- Jagielski, Ullman & Oprea, "Auditing Differentially Private Machine Learning: How Private is Private SGD?" (NeurIPS 2020). An early audit of DP-SGD: poisoning canaries, Clopper–Pearson intervals, and the lower bound of @eq-audit-eps-lo.
- Nasr, Song, Thakurta, Papernot & Carlini, "Adversary Instantiation: Lower Bounds for Differentially Private Machine Learning" (IEEE S&P 2021). A ladder of threat models from black box to gradient canaries with every iterate visible; tight bounds for the strongest.
- Nasr, Hayes, Steinke, Balle, Tramèr, Jagielski, Carlini & Terzis, "Tight Auditing of Differentially Private Machine Learning" (USENIX Security 2023). Dirac canaries, per-step audits with the canary at every step, Gaussian-shape bounds, and three bugs found.
- Steinke, Nasr & Jagielski, "Privacy Auditing with One (1) Training Run" (NeurIPS 2023). The one-run audit of @sec-audit-onerun; Corollary 5.4 is @eq-audit-onerun-delta, and its appendix gives code for the bound.
- Tramèr, Terzis, Steinke, Song, Jagielski & Carlini, "Debugging Differential Privacy: A Case Study for Privacy Auditing" (arXiv 2022). An audit that refuted the guarantee of a published implementation.
- Carlini, Chien, Nasr, Song, Terzis & Tramèr, "Membership Inference Attacks From First Principles" (IEEE S&P 2022). The likelihood ratio attack (LiRA) and the case for reporting TPR at a low FPR, which an audit's bound formalizes.
- Dong, Roth & Su, "Gaussian Differential Privacy" (JRSS B 2022). Trade-off curves, $\mu$-GDP and the CLT for composition used in @sec-audit-tight.
- Kairouz, Oh & Viswanath, "The Composition Theorem for Differential Privacy" (ICML 2015). The hypothesis-testing reading of $(\varepsilon, \delta)$-DP behind @eq-audit-region.

**Practice.** Exercises for this chapter, each with a worked solution, are in [the exercise book](../../book-exercises/part05-dp-ml/10-auditing.md).
