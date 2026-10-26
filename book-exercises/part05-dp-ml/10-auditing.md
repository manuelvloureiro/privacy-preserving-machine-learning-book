# Exercises: Auditing private training

Exercises for [Auditing private training](../../book-full/part05-dp-ml/10-auditing.md), each with a worked solution. Try each exercise before opening its solution. The script `code/part05-dp-ml/solutions_auditing.py` computes every number quoted here; run it with `uv run python code/part05-dp-ml/solutions_auditing.py`.

The solutions use the chapter's notation. An audit tests whether a training algorithm $M$ satisfies differential privacy (DP) as claimed. It trains with a canary record, on the dataset written $D'$, and without it, on $D$, and an attack says "canary in" when its score passes a threshold. The attack's true positive rate (TPR) is the fraction of runs with the canary that it flags, and its false positive rate (FPR) is the fraction of runs without the canary that it flags.

The training algorithm is stochastic gradient descent (SGD) with clipping and Gaussian noise, DP-SGD. Each step samples a fraction $q$ of the records, the sampling rate, and adds noise with multiplier $\sigma$. The run takes $T$ steps. The running example has $q = 0.01$, $\sigma = 1$ and $T = 10{,}000$. For it the accountant claims a privacy level $\varepsilon = 6.71$, at a slack $\delta = 10^{-5}$. Every bound holds with probability $1 - \beta = 95\%$, where $\beta$ is the failure probability. Gaussian differential privacy (GDP) with parameter $\mu$ is the trade-off curve of two unit-variance normals at distance $\mu$.

## Both inequalities {#sec-sol-audit-1}

> **Exercise 1.** **Both inequalities.** For three attacks with exactly known rates, (FPR, TPR) $= (0.05, 0.95)$, $(0.001, 0.05)$ and $(0.5, 0.999)$, compute the two lower bounds on $\varepsilon$ that @eq-audit-region gives at $\delta = 10^{-5}$, and say which one matters for each attack and why. *Check: 2.94 and 2.94; 3.91 and 0.05; 0.69 and 6.21.* *(Reading the privacy region)*

::: {.callout-tip collapse="true" title="Solution"}
1. **The two bounds.** @eq-audit-region gives $\varepsilon \ge \ln\frac{\mathrm{TPR} - \delta}{\mathrm{FPR}}$ (the direct form, from "canary in" events) and $\varepsilon \ge \ln\frac{1 - \mathrm{FPR} - \delta}{1 - \mathrm{TPR}}$ (the mirror form, from "canary out" events).
2. **A symmetric attack.** At $(0.05, 0.95)$, both are $\ln(0.95/0.05) = \ln 19 = 2.94$, up to the $\delta$: the attack errs equally often in both directions.
3. **A low-FPR attack.** At $(0.001, 0.05)$, the direct form is $\ln(0.05/0.001) = \ln 50 = 3.91$, and the mirror form $\ln(0.999/0.95) = 0.05$. The attack is confident when it says "in" and useless when it says "out", so the direct form carries the information.
4. **An attack that rarely misses.** At $(0.5, 0.999)$, the direct form is $\ln(0.999/0.5) = 0.69$ and the mirror form $\ln(0.5/0.001) = 6.21$. The attack almost never misses a present canary, so its "out" verdicts are confident: when it says "out", the canary was absent with odds of about 500 to 1.

*An audit that only computes the direct form throws away attacks of the third kind. The certified version, @eq-audit-eps-lo, applies Clopper–Pearson bounds to both forms.*
:::

## The ceiling of an audit {#sec-sol-audit-2}

> **Exercise 2.** **The ceiling of an audit.** A perfect attack with $n$ trials per hypothesis certifies the exact ceiling of @cor-audit-ceiling, about $\ln(n/\ln(2/\beta))$. Compare the approximation with the exact value for $n = 100$, 1,000, 10,000 and 100,000, and estimate the number of trials needed to certify the running example's 6.71. *Check: 3.28 against 3.30, 5.60 against 5.60, 7.90 against 7.91, 10.21 against 10.21; 3,038 trials exactly, 3,035 by the approximation.* *(The rule of three)*

::: {.callout-tip collapse="true" title="Solution"}
1. **The two formulas.** @cor-audit-ceiling gives the exact ceiling $\ln\frac{(\beta/2)^{1/n} - \delta}{1 - (\beta/2)^{1/n}}$, and its proof approximates it by $\ln(n/\ln 40)$ at $\beta = 0.05$, using $e^{-x} \approx 1 - x$ for the small exponent $x = \ln(2/\beta)/n$.
2. **Numbers.** The exact values are 3.28, 5.60, 7.90 and 10.21, against 3.30, 5.60, 7.91 and 10.21 from the approximation. The approximation is a little high at $n = 100$, where the exponent is largest and $e^{-x} \approx 1 - x$ is cruder.
3. **Inverting.** Setting $\ln(n/\ln 40) = 6.71$ gives $n = e^{6.71}\ln 40 = 3{,}035$, and the exact search gives 3,038.

*Each unit of $\varepsilon$ multiplies the trials by $e$. The ceiling is reached only by an attack that never errs; real audits need many more trials (@sec-audit-tight).*
:::

## Where ε lives {#sec-sol-audit-3}

> **Exercise 3.** **Where $\varepsilon$ lives.** For a $\mu$-GDP mechanism, @prp-audit-touch gives the false positive rate at which the line $\mathrm{TPR} = e^\varepsilon\mathrm{FPR} + \delta$ touches the trade-off curve. Evaluate it at $\delta = 10^{-5}$ for $\mu = 1.3108$ (the running example), 0.5 and 3, and compute how many trials without the canary are needed to see ten false positives there. *Check: $\varepsilon = 6.01$, FPR $8.1\times10^{-8}$, $1.2\times10^{8}$ trials; $\varepsilon = 1.99$, $1.1\times10^{-5}$, $8.8\times10^{5}$; $\varepsilon = 16.68$, $8.4\times10^{-13}$, $1.2\times10^{13}$.* *(Duality between $(\varepsilon, \delta)$ and trade-off curves)*

::: {.callout-tip collapse="true" title="Solution"}
1. **The touching point.** By @prp-audit-touch, the line touches the curve at the threshold with $\mathrm{FPR}^* = \Phi(-\varepsilon/\mu - \mu/2)$ and $\mathrm{TPR}^* = \Phi(-\varepsilon/\mu + \mu/2)$, where $\Phi$ is the standard normal distribution function and $\varepsilon$ is the curve's value at $\delta$. That $\varepsilon$ comes from the Gaussian privacy profile, @cor-gdp-profile, solved for $\varepsilon$ at $\delta = 10^{-5}$.
2. **The running example.** $\mu = 1.3108$ gives $\varepsilon = 6.01$ at $\delta = 10^{-5}$ through the Gaussian privacy profile, and then $\mathrm{FPR}^* = 8.1\times10^{-8}$, $\mathrm{TPR}^* = 4.3\times10^{-5}$. Ten false positives at that rate need $10/(8.1\times10^{-8}) = 1.2\times10^{8}$ trials.
3. **The other two.** $\mu = 0.5$ gives $\varepsilon = 1.99$, $\mathrm{FPR}^* = 1.1\times10^{-5}$, $\mathrm{TPR}^* = 9.3\times10^{-5}$ and $8.8\times10^{5}$ trials. $\mu = 3$ gives $\varepsilon = 16.68$, $\mathrm{FPR}^* = 8.4\times10^{-13}$, $\mathrm{TPR}^* = 2.5\times10^{-5}$ and $1.2\times10^{13}$ trials.

*In all three the touching TPR is between 2.5 and 10 times $\delta$: at $\delta = 10^{-5}$ the optimal test detects the canary only a few times in $10^5$, and its false positive rate is smaller still. Auditing the full $\varepsilon$ at a small $\delta$ is a measurement in the extreme tail, which is why the audits of the chapter stop short.*
:::

## What the sum of projections loses {#sec-sol-audit-4}

> **Exercise 4.** **What the sum of projections loses.** By @thm-audit-dirac, for a white-box Dirac canary the likelihood-ratio statistic of @eq-audit-lr behaves like $\mu$-GDP with $\mu = q\sqrt{T(e^{1/\sigma^2} - 1)}$ and the sum of projections like $\mu_{\mathrm{sum}} = q\sqrt T/\sigma$. Compute their ratio for $\sigma = 0.5$, 1, 2 and 4, and how many times more steps the sum would need to reach the likelihood ratio's $\mu$. Explain why the ratio tends to 1 for large $\sigma$. *Check: 3.661, 1.311, 1.066, 1.016; step factors 13.40, 1.72, 1.14, 1.03.* *(Linearizing a likelihood ratio)*

::: {.callout-tip collapse="true" title="Solution"}
1. **The ratio.** $\mu/\mu_{\mathrm{sum}} = \sqrt{e^{1/\sigma^2} - 1}\cdot\sigma$, since $q$ and $\sqrt T$ cancel.
2. **Numbers.** $0.5\sqrt{e^4 - 1} = 3.661$; $\sqrt{e - 1} = 1.311$; $2\sqrt{e^{0.25} - 1} = 1.066$; $4\sqrt{e^{1/16} - 1} = 1.016$.
3. **Steps.** Both $\mu$ grow like $\sqrt T$, so matching them needs the ratio squared in steps: 13.40, 1.72, 1.14 and 1.03.
4. **Large $\sigma$.** $e^{1/\sigma^2} - 1 = 1/\sigma^2 + 1/(2\sigma^4) + \dots$, so the ratio is $\sqrt{1 + 1/(2\sigma^2) + \dots} \to 1$. In words: when the noise is large compared with the canary's shift of 1, each step's log-likelihood ratio $\ln(1 - q + qe^{(x - 1/2)/\sigma^2})$ is nearly linear in $x$ over the range where $x$ falls, so summing the $x_t$ is nearly optimal. At small $\sigma$ the exponential is strongly curved and a single large $x_t$ outweighs many moderate ones.

*Small noise multipliers are where DP-SGD leaks most per step, and where a naive statistic wastes most of the evidence: at $\sigma = 0.5$ the sum would need thirteen times as many steps.*
:::

## How many canaries one run needs {#sec-sol-audit-5}

> **Exercise 5.** **How many canaries one run needs.** In a one-run audit where every one of $m = r$ guesses is correct, find the smallest $m$ that certifies the running example's $\varepsilon = 6.71$: first for pure DP with @eq-audit-onerun, then with the $\delta$ correction of @eq-audit-onerun-delta at $\delta = 10^{-6}$ and $10^{-5}$. Give the bound for 2,000, 2,500 and 5,000 canaries at $\delta = 10^{-5}$. *Check: 2,467; 2,483 and 2,648; 6.45, 6.66 and 7.28.* *(Binomial tails)*

::: {.callout-tip collapse="true" title="Solution"}
1. **Pure DP.** All $m$ guesses correct has probability at most $p^m$ under $\varepsilon$-DP, with $p = e^\varepsilon/(1 + e^\varepsilon)$. The claim is rejected when $p^m < \beta$, that is $m > \ln\beta/\ln p$.
2. **Numbers.** At $\varepsilon = 6.71$, $1 - p = 1/(1 + e^{6.71})$ is about $1.2\times10^{-3}$. Since $\ln p \approx -(1 - p)$ for $p$ near 1, the condition is about $m > \ln(1/\beta)(1 + e^\varepsilon)$, with $\ln(1/\beta) = \ln 20 = 3.00$; the smallest integer that satisfies the exact condition is 2,467.
3. **With $\delta$.** The correction term of @eq-audit-onerun-delta grows with the number of guesses $m$. At $\delta = 10^{-6}$ it barely matters, and 2,483 canaries suffice; at $\delta = 10^{-5}$ the requirement rises to 2,648.
4. **The bounds.** 2,000 perfect guesses certify 6.45, 2,500 certify 6.66 and 5,000 certify 7.28 at $\delta = 10^{-5}$.

*Two and a half thousand canaries, every one guessed right, just to reach the claim. A real attack errs, and @tbl-audit-onerun shows that 1,000 white-box Dirac canaries certify 2.22. In one run as in many, the bound grows like the logarithm of the number of trials.*
:::

## Randomized response meets the bound {#sec-sol-audit-6}

> **Exercise 6.** **Randomized response meets the bound.** Suppose each of 100 guesses is produced by $\varepsilon$-randomized response on the true inclusion bit: it reports the true bit with probability $e^\varepsilon/(1 + e^\varepsilon)$ and the other one otherwise. At $\varepsilon = \ln 3$, compute the expected number of correct guesses, check it by simulation, and compute the one-run p-value (@eq-audit-onerun with $\delta = 0$) of 75 and of 85 correct guesses. What does this say about the tightness of @eq-audit-onerun? *Check: 75; simulated mean 75.01; p-values 0.553 and 0.0111.* *(Tightness of a stochastic-dominance bound)*

::: {.callout-tip collapse="true" title="Solution"}
1. **Randomized response is $\varepsilon$-DP.** Its two output distributions differ by a factor of $e^\varepsilon$ on each output, by construction.
2. **Its correct guesses.** Each guess is right with probability $p = 3/(1 + 3) = 0.75$, independently, so the count is $\mathrm{Bin}(100, 0.75)$, with mean 75. Simulating 100,000 audits gives a mean of 75.01.
3. **The p-values.** @eq-audit-onerun bounds $\Pr[W \ge v]$ by $\Pr[\mathrm{Bin}(100, 0.75) \ge v]$: 0.553 for $v = 75$ and 0.0111 for $v = 85$.
4. **Tightness.** For randomized response, $W$ *is* the binomial, so the inequality holds with equality: no smaller bound is valid for every $\varepsilon$-DP algorithm. Seeing 75 correct guesses is entirely consistent with $\varepsilon = \ln 3$ (p-value 0.553), and is what this mechanism produces on average; seeing 85 would reject $\varepsilon = \ln 3$ at the 5% level.

*The one-run bound is exact for randomized response and conservative for mechanisms whose trade-off curve is smooth, like the Gaussian. This is why one-run audits of DP-SGD fall further short of the accountant than many-run audits.*
:::
