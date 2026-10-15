# Auditing private training

## Why this matters

Take this part's running example, a small network trained with stochastic gradient descent (SGD) under differential privacy (DP). Each step clips every example's gradient and adds Gaussian noise to their sum, which is the algorithm called DP-SGD ([DP-SGD](02-dp-sgd.md)). It trains on 10,000 records for 10,000 steps. Its noise multiplier $\sigma$, the noise's standard deviation in units of the clipping norm, is 1. At each step it samples every record independently with probability 0.01, which is called Poisson sampling.

The Rényi differential privacy (RDP) accountant turns this configuration into a privacy level $\varepsilon$ of 6.71. It does so at a slack $\delta$ of $10^{-5}$, the small probability with which the guarantee may fail. That number is the conclusion of a proof about an algorithm on paper, while what ships is code. An implementation can quietly differ from the algorithm. The noise may be scaled for the averaged gradient but added to the sum, or the batches may be shuffled but accounted as if Poisson-sampled. None of these changes the accuracy much, so the usual tests don't catch them, and the practical question is whether the trained model is as private as the accountant says.

An **audit** attacks the model to find out. It plants records of its own, called **canaries**, trains with and without them, and counts how often a membership test guesses right. DP limits how often any test can be right, so a test that is right too often shows that the algorithm cannot be private below some $\varepsilon$. That is the idea of this chapter: **privacy caps every test, so a measured test bounds privacy from below.**

## From an attack to a lower bound {#sec-audit-bound}

Write $D$ for the training set without the canary and $D'$ for the same set with it, so the two are neighbours that differ by one record. A **trial** is one training run on $D$ or on $D'$, with fresh randomness. The attack then computes a score for the canary from what it sees, such as minus the canary's loss, and says "in" when the score is above a threshold.

An attack is judged by two rates, one for each world. Its **true positive rate (TPR)** is the probability that it says "in" when the canary was trained on. Its **false positive rate (FPR)** is the probability that it says "in" when the canary was left out.

Both rates are probabilities of the same event under two neighbouring datasets, so DP bounds one by the other ([The definition of differential privacy](../part03-dp-foundations/01-definition.md)). The TPR can be at most $e^\varepsilon$ times the FPR, plus $\delta$. The same reasoning applied to the opposite event, the test saying "out", with the two datasets swapped, gives a second inequality. Together, for every test and every pair of neighbours, DP requires
$$\mathrm{TPR} \le e^\varepsilon\,\mathrm{FPR} + \delta, \qquad 1 - \mathrm{FPR} \le e^\varepsilon\,(1 - \mathrm{TPR}) + \delta.$$ {#eq-audit-region}
Solved for $\varepsilon$, the two inequalities turn any test with known rates into a lower bound,
$$\varepsilon \ge \ln\frac{\mathrm{TPR} - \delta}{\mathrm{FPR}} \quad\text{and}\quad \varepsilon \ge \ln\frac{1 - \mathrm{FPR} - \delta}{1 - \mathrm{TPR}}.$$

The rates are not known, though. They are estimated from $n$ trials with the canary and $n$ without. So the bound must use the pessimistic end of each estimate, or a lucky count would overstate the attack. Both expressions grow with the TPR and shrink with the FPR, so the pessimistic ends are an upper bound on the FPR and a lower bound on the TPR.

The standard tool for these ends is the one-sided **Clopper–Pearson bound** ([Hypothesis testing](../part01-prerequisites/04-hypothesis-testing.md)). Suppose that out of $n$ independent trials, each with the same unknown probability of success, a number $k$ succeed. Choose a small probability $a$, called the level. The upper bound at level $a$ is the rate at which $k$ or fewer successes would have probability exactly $a$. Any true rate above it would make the observed count implausibly small, which happens with probability at most $a$. The lower bound mirrors it.

Write $\beta$ for the probability that the audit's conclusion is wrong, 5% throughout this chapter. Take $\bar\alpha$, the Clopper–Pearson upper bound on the FPR, and $\underline\tau$, the lower bound on the TPR, each at level $\beta/2$. The chance that at least one of two events happens is at most the sum of their chances, which is the union bound. So both bounds hold together with probability at least $1 - \beta$. On that event, the algorithm is not $(\varepsilon, \delta)$-DP for any privacy level below
$$\varepsilon_{\mathrm{lo}} = \max\Big\{\ln\frac{\underline\tau - \delta}{\bar\alpha},\ \ln\frac{1 - \bar\alpha - \delta}{1 - \underline\tau}\Big\}.$$ {#eq-audit-eps-lo}

The bound has two conditions, and both come from the Clopper–Pearson step, which needs each count to be binomial. The trials must be independent, and the threshold must be chosen without looking at the trials that are counted. Sweeping the threshold over the same trials and reporting the best bound makes the confidence level wrong. The reported count is then the luckiest of many counts, not the count at one fixed threshold, so it is not binomial. Every audit in this chapter therefore chooses its threshold on half of the trials and counts on the other half.

Confidence is expensive. Suppose an audit makes 1,000 runs with the canary and 1,000 without. The attack catches the canary in 500 of the first runs and raises one false alarm in the second, so the point estimate is $\ln(0.5/0.001) = 6.21$. But one false positive in 1,000 runs is consistent with a true FPR of 0.00556, five times larger. So the certified bound is only 4.43. Ten times the trials at the same rates lift it only to 5.59.

That slowness has a hard limit, even for a perfect attack with no false positives and no misses. With no successes at all in $n$ trials, the one-sided 95% upper bound is about $3/n$, a shortcut called the **rule of three**. This bound works at level $\beta/2$, where the same reasoning gives about $\ln(2/\beta)/n$. The constant there is 3.69 instead of 3. When the TPR is near 1, $\varepsilon$ is about the logarithm of one over the FPR. So $n$ trials per hypothesis can certify at most about
$$\varepsilon_{\max}(n) \approx \ln\frac{n}{\ln(2/\beta)}.$$ {#eq-audit-ceiling}
Because $n$ sits inside a logarithm, each extra unit of $\varepsilon$ costs a factor of $e$ in trials (@tbl-audit-ceiling). Certifying the running example's 6.71 needs at least 3,038 trials per hypothesis, even for an attack that never errs. For a real network, each trial is a full training run.

| Trials per hypothesis $n$ | 100 | 1,000 | 10,000 | 100,000 | 1,000,000 |
| --- | --- | --- | --- | --- | --- |
| Largest certifiable $\varepsilon$ | 3.28 | 5.60 | 7.90 | 10.21 | 12.51 |

: The most a perfect attack can certify with $n$ trials per hypothesis, at 95% confidence and $\delta = 10^{-5}$. Conversely, certifying $\varepsilon = 1$, 2, 4, 6.71 and 8 takes at least 12, 30, 204, 3,038 and 10,999 trials per hypothesis. {#tbl-audit-ceiling}

Two consequences follow for reading audit results. First, $\varepsilon_{\mathrm{lo}}$ says that no smaller $\varepsilon$ is possible, never that this $\varepsilon$ is enough. That is because a weak attack gives a small bound for a leaky model as easily as for a private one. A bound above the claim refutes the claim, while a bound below it only says that this attacker, on this canary, could not do better. Second, compute both forms of @eq-audit-eps-lo. Take a test that says "in" almost always, with an FPR of 0.5 and a TPR of 0.999. It gets only 0.69 from the first form, but 6.21 from the second.

## Canaries and threat models {#sec-audit-canaries}

DP is a promise about the worst pair of neighbouring datasets and the worst test, so a strong audit has to build both. A typical record attacked by thresholding its loss gives a valid but nearly useless bound, because most records are barely exposed ([Membership inference](../part02-attacks/02-membership-inference.md)).

The auditor first chooses a canary. An **input-space canary** is an ordinary record designed to stand out, such as a mislabeled example or one far from the data. A **gradient-space canary** lets the auditor write the canary's gradient directly, as if it controlled the training code. The strongest is a **Dirac canary** (Nasr et al. 2023). Its gradient has length exactly $C$, the clipping norm, all on a single coordinate that the auditor picks so that the other records' gradients there are zero.

The auditor also chooses a threat model, meaning what the attacker sees. In the **black-box** model the attacker sees only the final model, and in the **white-box** model it sees every intermediate model. With plain SGD and a known step size, each update is the step's noisy gradient sum times a known factor. So a white-box attacker can recover every noisy sum that training produced.

A Dirac canary seen white-box gives the audit an exact picture of each step. Write $x_t$ for step $t$'s noisy sum on the canary's coordinate, divided by $C$. The other records contribute nothing there, so without the canary $x_t$ is pure noise, normal with mean 0 and standard deviation $\sigma$. With the canary, $x_t$ gains 1 whenever the canary is sampled, which happens with probability $q$, the sampling rate. So
$$x_t \sim \mathcal N(0, \sigma^2) \text{ without the canary}, \qquad x_t \sim (1 - q)\,\mathcal N(0, \sigma^2) + q\,\mathcal N(1, \sigma^2) \text{ with it}.$$ {#eq-audit-step}

The values are independent from step to step, even though training adapts. That is because the sampling coin and the noise are drawn afresh at every step, and the Dirac canary's gradient is the same whatever the weights are. This is exactly the pair that the accountant of [Amplification by subsampling](../part04-privacy-accounting/05-subsampling.md) analyses. The untouched coordinate is a condition of this picture. If other records' gradients touched it, they would add an unknown amount to each $x_t$. That amount acts as extra noise and as a bias that changes from step to step, so the audit's bound would drop.

The best test for this pair comes from the Neyman–Pearson lemma ([Hypothesis testing](../part01-prerequisites/04-hypothesis-testing.md)). Among all tests with a given FPR, the one with the largest TPR thresholds the **likelihood ratio**, how much more likely the observation is with the canary than without. The best TPR at each FPR traces the pair's **trade-off curve**.

For one step, the likelihood ratio of the mixture against pure noise is $1 - q + q\,e^{(x_t - 1/2)/\sigma^2}$. The steps are independent, so the ratio for the whole run is a product, and its logarithm is the **likelihood-ratio statistic**
$$\Lambda = \sum_{t=1}^T\ln\Big(1 - q + q\,e^{(x_t - 1/2)/\sigma^2}\Big),$$ {#eq-audit-lr}
where $T$ is the number of steps. Thresholding $\Lambda$ is the most powerful test at every FPR, and no attacker can do better against this canary, whatever it sees. The rest of the run depends only on the $x_t$ and on randomness that is the same in both worlds.

A simpler score is the **sum of projections**, $\sum_t x_t$. It needs only the first and last models when the step size is constant, because it is proportional to the total change in the canary's coordinate. But it treats every step alike, while the likelihood ratio does not. A step with $x_t = 3$ is three noise standard deviations out, which is very unlikely without the canary. So it is much stronger evidence than three steps at 1, and the exponential in $\Lambda$ weighs it accordingly.

To compare the two scores, measure each by one number. Take two unit-variance normal distributions whose means are a distance $\mu$ apart, and write $\Phi$ for the standard normal distribution function. At an FPR of $a$, the best test between them has
$$\mathrm{TPR}_\mu(a) = 1 - \Phi\big(\Phi^{-1}(1 - a) - \mu\big).$$ {#eq-audit-gdp}
A mechanism whose curve is this one or better satisfies Gaussian differential privacy (GDP) with parameter $\mu$, written $\mu$-GDP ([f-DP and Gaussian DP](../part04-privacy-accounting/04-gaussian-dp.md)). Each score adds up one small piece per step, so by the central limit theorem (CLT) its curve is close to a GDP curve. That chapter derives the likelihood-ratio statistic's $\mu$ as $q\sqrt{T(e^{1/\sigma^2} - 1)}$.

On the running example the likelihood-ratio statistic has $\mu = 1.3108$. The sum of projections has 1.00, or 0.9951 once the randomness in how often the canary is sampled is counted. The Gaussian privacy profile of that chapter converts $\mu$ into $\varepsilon$ at a given $\delta$. At the running example's $\delta$, the likelihood ratio's $\mu$ becomes 6.01 and the sum's only 4.35. So choosing the sum lowers the most the audit can ever reach by about 1.7, before a single trial is run.

Black-box audits with input canaries are weaker, for three reasons. First, the final model is one function of all the noisy sums, so it carries at most as much information as all of them, and usually much less. Second, an input canary's gradient does not keep length $C$ at every step, since it shrinks or turns once the model fits the canary or gives up on it. Third, it shares directions with the other records' gradients, whose contributions act as extra noise.

Published audits show this gap. Nasr et al. (2021) found the bound tight for an attacker who sees every intermediate model and chooses a worst-case dataset and gradient canary. It became much weaker as the attacker was restricted towards a realistic one.

So the two threat models answer different questions. A white-box gradient audit tests the mechanism, whether the code clips and adds noise as the proof assumes, and its bound should approach the accountant's. A black-box input audit tests the deployment, what an attacker with a released model and a planted record can do.

## Tight audits of the mechanism {#sec-audit-tight}

The white-box Dirac audit faces the accountant's own pair, so with enough trials it should approach the true $\varepsilon$. Write $P$ for the distribution of what the attacker sees without the canary, and $Q$ for the distribution with it. Solve the definition of DP for $\varepsilon$, and take the worst set $S$ of possible outcomes. This gives the smallest level the pair allows,
$$\varepsilon^*(\delta) = \sup_S\ \ln\frac{Q(S) - \delta}{P(S)},$$ {#eq-audit-sup}
over the sets to which $Q$ gives a probability above $\delta$. Only likelihood-ratio thresholds need trying, because they give the largest $Q(S)$ for each $P(S)$. So an audit that thresholds $\Lambda$ and knew its rates exactly would recover the pair's true $\varepsilon$. For the Dirac canary this pair is the accountant's worst case, so that is DP-SGD's true $\varepsilon$ for neighbours that differ by adding the canary.

The accountant's number is not that target, because the RDP accountant's 6.71 is a loose upper bound. The numerical accountant of [f-DP and Gaussian DP](../part04-privacy-accounting/04-gaussian-dp.md) puts the true value of the running example between 6.14 and 6.24. So even an audit with infinitely many trials stops about half a unit below 6.71, and that gap belongs to the accountant, not to the audit.

Real audits fall much further short, because of where on the trade-off curve $\varepsilon$ is decided. On a plot of TPR against FPR, the first inequality of @eq-audit-region is a straight line of slope $e^\varepsilon$. The ratio in @eq-audit-sup is largest where that line just touches the curve. The curve's slope at each threshold is the likelihood ratio there, so the touching point is the threshold where the likelihood ratio equals $e^\varepsilon$. For two normals at distance $\mu$, that threshold lies $\varepsilon/\mu + \mu/2$ standard deviations out. The FPR $a^*$ and the TPR at that threshold are then
$$a^* = \Phi\Big(-\frac{\varepsilon}{\mu} - \frac{\mu}{2}\Big), \qquad \mathrm{TPR}^* = \Phi\Big(-\frac{\varepsilon}{\mu} + \frac{\mu}{2}\Big).$$ {#eq-audit-touch}

For the running example, the Gaussian approximation has $\mu = 1.3108$ and $\varepsilon = 6.01$. Its line touches the curve at an FPR of $8.1 \times 10^{-8}$, where the TPR is only $4.3 \times 10^{-5}$. So certifying the full $\varepsilon$ at $\delta = 10^{-5}$ means measuring a false positive rate below one in ten million. With $n$ trials, though, an audit measures false positive rates only down to about $3/n$. On the running example's curve, $\ln(\mathrm{TPR}/\mathrm{FPR})$ is still only 4.38 at a false positive rate of one in ten thousand. Reaching 6 needs a rate near $10^{-7}$. Seeing even ten false positives at that rate would take about $1.2\times10^{8}$ trials without the canary (@fig-audit-region).

![Left: the trade-off of the white-box audit of the running example on log axes, from 30,000 trials per hypothesis. The black line is the most any test can achieve under $(6.71, 10^{-5})$-DP; the dashed grey curve is $\mu$-GDP with the CLT's $\mu = 1.31$. The blue curve is the likelihood-ratio statistic, the orange one the sum of projections; boxes are 95% Clopper–Pearson regions at four thresholds. Right: the $\varepsilon$ a threshold would certify if its rates were known exactly, $\ln((\mathrm{TPR} - \delta)/\mathrm{FPR})$, against its false positive rate. The Gaussian curve reaches its $\varepsilon$ of 6.01 only at FPR $8 \times 10^{-8}$ (dotted line), far below the rates a feasible audit can measure.](figures/audit_region.svg){#fig-audit-region}

Nasr et al. (2023) avoid the tail by assuming the curve's shape. If the mechanism is $\mu$-GDP, the TPR of any test is at most $\mathrm{TPR}_\mu$ at its FPR, by @eq-audit-gdp. Rearranged, a single test bounds $\mu$ from below:
$$\mu \ge \Phi^{-1}(\mathrm{TPR}) - \Phi^{-1}(\mathrm{FPR}).$$ {#eq-audit-mu}
The rates can be measured at a moderate FPR, where trials are cheap, and the Gaussian privacy profile then turns the bound on $\mu$ into a bound on $\varepsilon$.

The two steps differ in how far they can be trusted. The bound on $\mu$ is rigorous, but the conversion to $\varepsilon$ is valid only if the curve really is Gaussian. The subsampled Gaussian of DP-SGD has a heavier tail, so the $\mu$ that @eq-audit-mu gives depends on where you read it. In the simulation below, @eq-audit-mu applied to the measured rates, without confidence intervals, gives 1.300 at an FPR of one in ten. Further into the tail, at $2\times10^{-4}$, it gives 1.492. So those authors mostly put the canary into every step, which makes each step exactly Gaussian, as @sec-audit-bugs does.

The full edition's script simulates the white-box Dirac audit directly from @eq-audit-step, which is equivalent to training with a Dirac canary and far cheaper. @tbl-audit-tight gives the certified bounds, each with the threshold chosen on the first half of the trials and the counts taken on the second.

| Trials per hypothesis | 1,000 | 3,000 | 10,000 | 50,000 |
| --- | --- | --- | --- | --- |
| $\varepsilon_{\mathrm{lo}}$, @eq-audit-eps-lo | 1.64 | 1.88 | 2.84 | 3.63 |
| $\mu_{\mathrm{lo}}$, @eq-audit-mu with Clopper–Pearson | 0.795 | 1.064 | 1.147 | 1.185 |
| $\varepsilon$ of $\mu_{\mathrm{lo}}$-GDP (assumes the Gaussian shape) | 3.36 | 4.70 | 5.13 | 5.33 |
| Ceiling, @eq-audit-ceiling, at half the trials | 4.91 | 6.01 | 7.21 | 8.82 |

: The white-box Dirac audit of the running example with the likelihood-ratio statistic, against the accountant's $\varepsilon = 6.71$. All bounds hold with probability 95%. {#tbl-audit-tight}

The rigorous bound reaches 3.63 with 50,000 trials per hypothesis, about half the claim. The Gaussian-shape estimate climbs faster, to 5.33, and approaches the CLT's 6.01 from below. The sum of projections needs far more trials for less. It certifies 1.97 with 10,000 trials per hypothesis, and only 3.26 with ten times as many. Bounds read in the far tail are also noisy, as @fig-audit-trials shows, and all of them stay below the claim, as a correct implementation requires.

![The certified $\varepsilon$ lower bound of the white-box audit of the running example against the number of trials per hypothesis, at 95% confidence and $\delta = 10^{-5}$: the likelihood-ratio statistic (circles), the sum of projections (triangles) and the Gaussian-shape estimate from the likelihood ratio (squares), with the accountant's 6.71 and the ceiling of a perfect attack. The bounds grow roughly like the logarithm of the number of trials.](figures/audit_trials.svg){#fig-audit-trials}

## One-run auditing {#sec-audit-onerun}

Thousands of training runs are out of the question for a model that takes a day to train. Steinke, Nasr & Jagielski (2023) get many trials out of a single run by planting many canaries at once.

The protocol works as follows. Choose a number of canaries $m$, and include each one in the training set by its own independent fair coin. Train once and score every canary. Then choose the length $r$ of a list of guesses. Guess "in" for the $r/2$ canaries with the highest scores and "out" for the $r/2$ with the lowest, and abstain on the rest. Write $W$ for the number of correct guesses.

Write $p_\varepsilon$ for the largest chance of a correct guess that $\varepsilon$-DP allows, which is $e^\varepsilon/(1 + e^\varepsilon)$. The reason is as follows.

Order the guessed canaries, and look at the canary in position $i$. Keep the coins of the earlier canaries fixed and average over the later ones, which are independent of canary $i$'s coin. Flipping that coin adds or removes one record, so the guesses are $\varepsilon$-DP as a function of it, and averaging over independent coins keeps that guarantee.

Before training, the coin is a fair bet, at odds of one to one. Seeing the guesses can multiply those odds by at most $e^\varepsilon$. So the guess about canary $i$ is right with probability at most $p_\varepsilon$, even given whether the earlier guesses were right. A count of such guesses is no more likely to exceed any value than a binomial count with $r$ trials and success probability $p_\varepsilon$. For every count $v$,
$$\Pr[W \ge v] \le \Pr\big[\mathrm{Bin}(r, p_\varepsilon) \ge v\big].$$ {#eq-audit-onerun}
Observing $v$ correct guesses therefore rejects every privacy level whose binomial tail at $v$ is below $\beta$, and the largest rejected level is the certified bound.

The coins must be independent and fair, because the argument uses both the prior of one half and the independence of the later coins. Suppose instead that the auditor included the first half of the canaries. If the scores happened to rise with a canary's position, the guesses could then be right for reasons that have nothing to do with training.

The binomial is exactly what randomized response produces ([The definition of differential privacy](../part03-dp-foundations/01-definition.md)), so the bound is tight for that mechanism. The paper's worked example sets the privacy level to $\ln 3$. On average, that level produces 75 correct guesses out of 100. From exactly that count, @eq-audit-onerun certifies 0.702 at 95% confidence.

Under $(\varepsilon, \delta)$-DP each coin's guarantee may fail with probability $\delta$, and the bound adds a term to pay for it. The paper gives it as its Corollary 5.4. Write $f(v)$ for the binomial tail on the right of @eq-audit-onerun. Then
$$\Pr[W \ge v] \le f(v) + 2m\delta\max_{i = 1, \dots, m}\frac{f(v - i) - f(v)}{i}.$$ {#eq-audit-onerun-delta}
Each ratio in the maximum is at most 1, so the extra term is at most $2m\delta$, and it grows with the number of canaries. When it approaches $\beta$, it can use up the whole chance of error the audit allows, and then no number of correct guesses rejects anything.

The ceiling is again logarithmic. If all $r$ guesses are right, the bound is about $\ln(r/\ln(1/\beta))$ with $\delta = 0$. The constant is $\ln(1/\beta)$ rather than $\ln(2/\beta)$ because the test is a single one-sided tail. So a thousand canaries, all guessed and all right, certify 5.81. Certifying 6.71 needs about 2,500. Each canary is a trial, so a one-run audit has the same ceiling as a many-run audit with as many trials. It saves training runs, not trials.

One run is also not tight for DP-SGD, for two reasons. Each canary is guessed only once, so with $m$ canaries the audit cannot read the trade-off curve at an FPR much below $1/m$. And the binomial is calibrated to randomized response, whose curve has a sharp corner, while the curve of DP-SGD is smooth.

The full edition simulates one run of the running example with 1,000 white-box Dirac canaries, each scored by $\Lambda$. The auditor tries six list lengths and pays for trying them with a Bonferroni correction. Each test runs at level $\beta/6$, so that the chance that any of the six is wrong is at most $\beta$ (@tbl-audit-onerun).

| Guesses $r$ | 20 | 50 | 100 | 200 | 500 | 1,000 |
| --- | --- | --- | --- | --- | --- | --- |
| Correct, white-box Dirac canaries | 20 | 50 | 92 | 185 | 425 | 745 |
| $\varepsilon_{\mathrm{lo}}$ (Bonferroni over the six) | 1.21 | 2.22 | 1.58 | 1.88 | 1.43 | 0.90 |

: One-run audit of the running example's mechanism with 1,000 white-box Dirac canaries, at 95% overall confidence and $\delta = 10^{-5}$. {#tbl-audit-onerun}

All 50 most confident guesses are right, which certifies 2.22. That compares with 2.84 from 10,000 runs per hypothesis in @tbl-audit-tight, at a twenty-thousandth of the cost.

With 5,000 canaries the auditor also tries two longer lists, so it runs eight tests, each at level $\beta/8$. Its best bound is 2.53, from the list of 500, of which 479 are right. There the $\delta$ term shows its teeth. Its largest possible size, $2m\delta = 0.1$, is far above each test's level. The actual term for the list of 500 stays below that level, but a list of 20 guesses, all correct, certifies nothing at all.

## Black-box audit of the trained network {#sec-audit-blackbox}

To see what the released network itself gives away, the full edition runs a one-run audit with **mislabeled canaries**. It takes 1,000 held-out test records and flips their labels, so that the model can fit them only by memorizing them. It includes each in the 10,000 training records with probability one half and trains once with DP-SGD at the claimed configuration. It then scores each canary by minus its loss on the final model, and bounds $\varepsilon$ as in @sec-audit-onerun. The bound is valid because DP-SGD's guarantee holds for every dataset, including one with planted records.

Two control runs make the result readable. A model trained without clipping or noise shows what the attack achieves unprotected. A model trained with clipping but no noise has no finite $\varepsilon$ at all. So if its audit also gives a small bound, then the audit's weakness, not the noise, is what keeps the bound small. All three weaknesses of black-box audits from @sec-audit-canaries apply here, so the bound should be far below 6.71 (@tbl-audit-blackbox).

| Training | Test accuracy | Mean loss, in | Mean loss, out | Correct of $r = 100$ | Correct of $r = 1{,}000$ | Best $\varepsilon_{\mathrm{lo}}$ |
| --- | --- | --- | --- | --- | --- | --- |
| DP-SGD, $\sigma = 1$, claim $\varepsilon = 6.71$ | 0.9645 | 5.565 | 6.013 | 59 | 556 | 0.13 |
| Clipping only, $\sigma = 0$, $\varepsilon = \infty$ | 0.9638 | 5.511 | 6.155 | 67 | 589 | 0.24 |
| Non-private, no clipping | 0.9155 | 0.023 | 9.041 | 91 | 967 | 2.96 |

: Black-box one-run audit of the running example's network with 1,000 mislabeled canaries, one training run each, at 95% overall confidence and $\delta = 10^{-5}$. Test accuracy is on the 19,000 test records not used as canaries. {#tbl-audit-blackbox}

The non-private model memorizes the canaries it trained on. Their mean loss is 0.023, against 9.041 for those it did not see. Of its thousand guesses, 967 are right, which certifies 2.96. DP-SGD does not memorize, since the losses of included and excluded canaries differ by less than half a unit on average. Its best list, 293 right out of 500, certifies only 0.13. The clipping-only model is barely easier, at 0.24. So even with the noise removed, this attacker learns almost nothing. Clipping alone limits each canary's pull on the weights to $C$ per step, against the combined pull of 10,000 correctly labelled records, and that is what defeats it (@fig-audit-one-run).

![One-run audits with 1,000 canaries. Left and middle: the loss of each mislabeled canary on the final model, for canaries in the training set (red) and left out (blue), without privacy and with DP-SGD at $\varepsilon = 6.71$. Right: the fraction of the $r$ guesses that are correct, against $r$, for the non-private network (black box), the DP-SGD network (black box) and the mechanism with white-box Dirac canaries; the dashed line is the most a $6.71$-DP algorithm allows on average, $e^\varepsilon/(1 + e^\varepsilon)$.](figures/audit_one_run.svg){#fig-audit-one-run}

The clipping-only bound of 0.24 does not mean that clipping alone is 0.24-DP, since a white-box Dirac canary would identify membership in that model perfectly. So nothing here contradicts the claim of 6.71, and nothing confirms it. A realistic attacker with one model and planted records extracts very little, while an attacker who sees every step and controls a gradient extracts much more. An honest privacy statement reports both, each with its threat model ([Choosing an accountant](../part04-privacy-accounting/09-choosing-an-accountant.md) lists what else it must contain).

## Finding bugs {#sec-audit-bugs}

An audit bound below the claim proves nothing, but a bound above it shows, with confidence $1 - \beta$, that the claim is false. That makes audits a test suite for private training code, silent on a correct implementation and loud on a broken one.

The cheapest test is a **per-step audit**. It puts a Dirac canary into every step of one run and leaves it out of another. Each step's value on the canary coordinate is one trial, so two runs of $T$ steps give $T$ trials per hypothesis. With no sampling coin for the canary, each step of a correct implementation is a plain Gaussian mechanism, with sensitivity $C$ and noise $\sigma C$:
$$x_t \sim \mathcal N(0, \sigma^2) \text{ without the canary}, \qquad x_t \sim \mathcal N(1, \sigma^2) \text{ with it}.$$ {#eq-audit-perstep}
That pair is exactly $\mu$-GDP with $\mu = 1/\sigma$. The noise is fresh at every step and the canary's gradient is fixed, so the counts are binomial. Then @eq-audit-mu with Clopper–Pearson rates gives $\mu_{\mathrm{lo}}$, a rigorous lower bound on the per-step $\mu$. If $\mu_{\mathrm{lo}}$ exceeds $1/\sigma$, then with 95% confidence the code does not add the noise it claims.

A per-step audit checks the noise and the clipping, but it cannot see the sampling or the accountant. Because the canary is in every step, the audit never exercises the sampling code. It also compares the measured noise with the claimed $\sigma$, not with the claimed $\varepsilon$, so a bug in what the accountant is told passes it. With such audits, Nasr et al. (2023) caught three bugs in real code. One applied clipping after averaging, one drew noise from a seed with only 100 possible values, and one used a wrong noise scale.

An **end-to-end audit** uses the many-trial protocol of @sec-audit-tight and tests the claimed $\varepsilon$ directly. It catches an accounting bug as readily as a noise bug, but only a large one. A bug shows only when it raises $\varepsilon$ above the claim by more than the audit's shortfall, the gap between the true $\varepsilon$ and what the audit can certify. For the running example that shortfall is about three units at 100,000 runs. Tramèr et al. (2022) used such an audit to show that an open-source implementation of a private deep-learning algorithm violated its claimed guarantee.

@tbl-audit-bugs audits three implementations of the running example's noise, each from two runs of 10,000 steps. The running example has $\sigma = 1$ and $C = 0.5$. Its accountant column is what the RDP accountant would report for the noise the audit measured.

| Implementation | Noise on the sum | Claimed per-step $\mu$ | Audited $\mu_{\mathrm{lo}}$ | Accountant at $\sigma = 1/\mu_{\mathrm{lo}}$ | Verdict |
| --- | --- | --- | --- | --- | --- |
| Correct | $\sigma C$ | 1 | 0.886 | 5.38 | consistent |
| Variance passed as a standard deviation | $\sigma^2C^2 = 0.5\,C$ | 1 | 1.745 | 29.27 | refuted |
| Noise scaled for the mean, added to the sum | $\sigma C/B = 0.01\,C$ | 1 | 6.359 | over 4,700 | refuted |

: Per-step audits of three implementations of the running example's DP-SGD step, from 10,000 steps with and 10,000 without a Dirac canary, at 95% confidence. $B = 100$ is the expected batch size, and the clipping norm is $C = 0.5$. {#tbl-audit-bugs}

The correct implementation passes, at 0.886 against the claimed 1. Through the accountant that is 5.38, below 6.71. The two noise bugs fail decisively. Passing the variance $\sigma^2C^2 = 0.25$ where the code expects a standard deviation halves the noise, and the audit measures 1.745. Write $B$ for the expected batch size, 100 here. Scaling the noise for the averaged gradient, whose sensitivity is $C/B$, and adding it to the sum makes it $B$ times too small. Then 10,000 steps separate the two worlds perfectly, and 6.359 is the most @eq-audit-mu can certify from error-free counts. @fig-audit-bugs shows why both bugs are easy to see.

![Per-step audits of three implementations: the projections of 10,000 steps on a Dirac canary, without it (blue) and with it at every step (red), in units of $C$. A correct implementation has noise of standard deviation $\sigma C$; the two bugs shrink it by a factor of 2 and of 100. The titles give the certified lower bound on the per-step $\mu$; the claim is 1.](figures/audit_bugs.svg){#fig-audit-bugs}

An accounting bug shows the division of labour. Suppose the code is correct, but the accountant is told the dataset has 20,000 records instead of 10,000, as can happen when augmented copies are counted. It then reports $\varepsilon = 3.06$ at a sampling rate of 0.005. The per-step audit cannot object, because the noise it measures is right. Feeding its $\mu_{\mathrm{lo}}$ to the accountant at that sampling rate gives 2.47, which does not contradict 3.06. The end-to-end audit of @tbl-audit-tight does, because its bound of 3.63 exceeds the claim, so the claim is false with 95% confidence.

Some bugs no canary audit catches. A bug that shows only on datasets or records the audit did not try passes. Batches drawn by shuffling but accounted as Poisson can have an $\varepsilon$ several times the reported one ([Amplification by subsampling](../part04-privacy-accounting/05-subsampling.md)). Yet a per-step audit never exercises the sampling, and an end-to-end audit sees the gap only where it exceeds its shortfall.

Audits also catch errors only in the unsafe direction. Noise $\sigma C$ added to the averaged gradient instead of to the sum is $B$ times more noise than claimed. That implementation is more private than its claim, so every audit passes. Nor does passing show that the code is correct. A lower bound such as $\mu_{\mathrm{lo}} = 0.97$ for a claimed $\mu = 1$ can show that $\mu$ is too large, never that it is small enough. Passing an audit is evidence, never a certificate.

## Code

This snippet computes the chapter's bounds from counts alone. It gets the Clopper–Pearson bounds from exact binomial sums, and it prints numbers quoted in the text.

```python
from collections.abc import Callable
from math import exp, lgamma, log
from statistics import NormalDist

N = NormalDist()


def binom_cdf(k: int, n: int, p: float) -> float:
    """Pr[Bin(n, p) <= k], summed in logs so that large n is safe."""
    if p <= 0.0:
        return 1.0
    if p >= 1.0:
        return 1.0 if k >= n else 0.0
    log_terms = (lgamma(n + 1) - lgamma(i + 1) - lgamma(n - i + 1) + i * log(p) + (n - i) * log(1 - p) for i in range(k + 1))
    return min(1.0, sum(exp(t) for t in log_terms))


def bisect(f: Callable[[float], float], target: float, lo: float, hi: float) -> float:
    """The x in [lo, hi] with f(x) = target, for f decreasing in x."""
    for _ in range(100):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if f(mid) > target else (lo, mid)
    return lo


def cp_upper(k: int, n: int, a: float) -> float:
    """One-sided Clopper-Pearson upper bound: the rate at which k or fewer successes has probability a."""
    return bisect(lambda p: binom_cdf(k, n, p), a, 0.0, 1.0)


def eps_lower_bound(fp: int, tp: int, n: int, delta: float, beta: float = 0.05) -> float:
    """The eps certified with confidence 1 - beta from fp false and tp true positives in n trials each."""
    fpr_hi = cp_upper(fp, n, beta / 2)
    tpr_lo = 1 - cp_upper(n - tp, n, beta / 2)
    return max(log((tpr_lo - delta) / fpr_hi), log((1 - fpr_hi - delta) / (1 - tpr_lo)))


def ceiling(n: int, delta: float, beta: float = 0.05) -> float:
    """The most a perfect attack can certify with n trials per hypothesis."""
    r = (beta / 2) ** (1 / n)
    return log((r - delta) / (1 - r))


def one_run_bound(correct: int, guesses: int, beta: float = 0.05) -> float:
    """The eps a one-run audit certifies under pure DP: the largest eps the binomial tail rejects."""

    def tail(eps: float) -> float:
        return -(1 - binom_cdf(correct - 1, guesses, exp(eps) / (1 + exp(eps))))

    return bisect(tail, -beta, 0.0, 20.0)


def gdp_eps(mu: float, delta: float) -> float:
    """The eps of mu-GDP at a given delta, from the Gaussian privacy profile."""

    def profile(eps: float) -> float:
        return N.cdf(-eps / mu + mu / 2) - exp(eps) * N.cdf(-eps / mu - mu / 2)

    return bisect(profile, delta, 0.0, 50.0)


print(f"{eps_lower_bound(1, 500, 1_000, 1e-5):.2f}")  # 4.43
print(f"{eps_lower_bound(10, 5_000, 10_000, 1e-5):.2f}")  # 5.59
print(f"{ceiling(1_000, 1e-5):.2f}")  # 5.60
print(f"{ceiling(10_000, 1e-5):.2f}")  # 7.90
print(f"{one_run_bound(75, 100):.3f}")  # 0.702
print(f"{gdp_eps(1.3108, 1e-5):.2f}")  # 6.01
print(f"{gdp_eps(0.9951, 1e-5):.2f}")  # 4.35
```

The full edition's script runs every audit in this chapter:

```bash
uv run python code/part05-dp-ml/auditing.py
```

## Cheat sheet

| Result | Formula or value | Conditions |
| --- | --- | --- |
| DP caps every test | $\mathrm{TPR} \le e^\varepsilon\mathrm{FPR} + \delta$, $1 - \mathrm{FPR} \le e^\varepsilon(1 - \mathrm{TPR}) + \delta$ | Any test, any neighbours |
| Certified bound | $\max\{\ln\frac{\underline\tau - \delta}{\bar\alpha}, \ln\frac{1 - \bar\alpha - \delta}{1 - \underline\tau}\}$ with Clopper–Pearson bounds at $\beta/2$ | Independent trials; threshold chosen on other trials |
| Ceiling | $\approx\ln(n/\ln(2/\beta))$: 5.60 at $n = 1{,}000$, 7.90 at 10,000 | Perfect attack |
| Best statistic, Dirac canary | $\Lambda = \sum_t\ln(1 - q + qe^{(x_t - 1/2)/\sigma^2})$ | White box, untouched coordinate |
| Gaussian approximations | $\mu = q\sqrt{T(e^{1/\sigma^2} - 1)}$; sum of projections $q\sqrt T/\sigma$ | Many steps (CLT) |
| Where $\varepsilon$ lives | FPR $\Phi(-\varepsilon/\mu - \mu/2)$: $8.1\times10^{-8}$ for the running example | $\mu$-GDP |
| GDP bound from one test | $\mu \ge \Phi^{-1}(\mathrm{TPR}) - \Phi^{-1}(\mathrm{FPR})$ | Rigorous for $\mu$; for $\varepsilon$ only if the curve is Gaussian |
| One run | Correct guesses no larger than $\mathrm{Bin}(r, e^\varepsilon/(1 + e^\varepsilon))$, plus a correction of at most $2m\delta$ | Canaries included by independent fair coins |
| Per-step audit | $\mathcal N(0, \sigma^2)$ against $\mathcal N(1, \sigma^2)$; claim $\mu = 1/\sigma$ | Canary at every step; tests noise and clipping only |

: The results of this chapter. {#tbl-audit-cheat}

**Rules of thumb**

- An audit can refute a claim, never prove one. Report its threat model with its number.
- Budget trials before you start. Certifying a privacy level $\varepsilon$ needs at least $e^\varepsilon\ln(2/\beta)$ trials per hypothesis. In practice it needs far more, because at small $\delta$ the level lives at false positive rates you cannot measure.
- To test an implementation, use the strongest attack: white box, a Dirac gradient canary and the likelihood ratio. Audit the noise per step and the claim end to end.
- Choose thresholds and list lengths on separate data, or pay for the choice with a Bonferroni correction.
- Read a small black-box bound as a statement about the attacker, not as the model's privacy.

## Further reading

- Nasr, Hayes, Steinke, Balle, Tramèr, Jagielski, Carlini & Terzis, "Tight Auditing of Differentially Private Machine Learning" (USENIX Security 2023): Dirac canaries, per-step audits, Gaussian-shape bounds, and real bugs found.
- Nasr, Song, Thakurta, Papernot & Carlini, "Adversary Instantiation: Lower Bounds for Differentially Private Machine Learning" (IEEE S&P 2021): a ladder of threat models from black box to gradient canaries with every intermediate model visible, and tight bounds for the strongest.
- Steinke, Nasr & Jagielski, "Privacy Auditing with One (1) Training Run" (NeurIPS 2023): the one-run audit, with code for the bound.
- Tramèr, Terzis, Steinke, Song, Jagielski & Carlini, "Debugging Differential Privacy: A Case Study for Privacy Auditing" (arXiv 2022): an audit that refuted the guarantee of a published implementation.

**Derivations.** Every result in this chapter is derived step by step in [the full edition's chapter](../../book-full/part05-dp-ml/10-auditing.md) or in the chapters it links, except the $(\varepsilon, \delta)$ version of the one-run bound, which it cites from Steinke, Nasr & Jagielski (2023), and the audit findings it reports from the papers it cites.
