# Exercises: Hyperparameter tuning and private selection

Exercises for [Hyperparameter tuning and private selection](../../book-full/part05-dp-ml/03-hyperparameter-tuning.md), each with a worked solution. Try each exercise before opening its solution. The script `code/part05-dp-ml/solutions_hyperparameter_tuning.py` computes every number quoted here; run it with `uv run python code/part05-dp-ml/solutions_hyperparameter_tuning.py`.

The solutions use the chapter's notation. A run $Q$ is a private training algorithm, satisfying differential privacy (DP), that outputs a model and a score; the best-of-$K$ mechanism $M$ runs it $K$ times and releases the best output, and $f(x) = \mathbb E[x^K]$ is the generating function of $K$. Guarantees are Rényi differential privacy (RDP) curves $\varepsilon(\alpha)$, with $D_\alpha$ the Rényi divergence of order $\alpha$, which Papernot & Steinke write $\lambda$. A geometric number of runs stops after each run with a probability $\omega$, and a Poisson number of runs is described by its mean $\mathbb E[K]$. Randomized response with parameter $\varepsilon$ is the run with two outcomes, "good" and "bad", where a run is bad with probability $s = e^\varepsilon/(1 + e^\varepsilon)$ on $D$ and $1 - s$ on its neighbour $D'$. The running example trains with stochastic gradient descent (SGD) with clipping and noise, DP-SGD, at a sampling rate $q$ of 0.01.

## Composition is nearly tight at large orders {#sec-sol-hpt-1}

> **Exercise 1.** **Composition is nearly tight at large orders.** For randomized response with $\varepsilon = 0.5$ and the best of $k = 5$ runs, compute the exact Rényi divergence $D_\alpha(M(D)\|M(D'))$ at orders 2, 10 and 100, and compare it with the lower bound $k\varepsilon - k\ln(1 + e^{-\varepsilon})/(\alpha - 1)$ of @sec-hpt-leak and with composition, $k$ times one run's divergence. *Check: at order 2 the three numbers are 0.6763, 0.1296 and 1.1367; at order 10, 2.2366, 2.2366 and 2.2367.* *(Rényi divergence of discrete distributions)*

::: {.callout-tip collapse="true" title="Solution"}
1. **The two distributions.** The best of 5 is bad only if all five runs are bad. On $D$ that has probability $s^5$ with $s = e^{0.5}/(1 + e^{0.5}) = 0.6225$, on $D'$ probability $(1 - s)^5$; the good outcome takes the rest.
2. **The divergence.** $D_\alpha = \frac{1}{\alpha - 1}\ln\big[(s^5)^\alpha((1 - s)^5)^{1 - \alpha} + (1 - s^5)^\alpha(1 - (1 - s)^5)^{1 - \alpha}\big]$. At order 2 this is 0.6763, at order 10 it is 2.2366 and at order 100 it is 2.4761.
3. **The lower bound.** It keeps only the bad term of the sum: $5 \times 0.5 - 5\ln(1 + e^{-0.5})/(\alpha - 1)$, which is 0.1296 at order 2, 2.2366 at order 10 and 2.4761 at order 100.
4. **Composition.** Five times one run's divergence (@eq-hpt-compose): 1.1367 at order 2, 2.2367 at order 10 and 2.4761 at order 100.
5. **Reading it.** At low orders the good outcome contributes most of the sum, so the lower bound is loose and composition overstates the exact value by 68%. At high orders the bad outcome, with its likelihood ratio $e^{5\varepsilon}$, dominates the sum, and all three agree.

*The best of a fixed number of runs is as expensive as all the runs exactly where conversions to $(\varepsilon, \delta)$ at small $\delta$ look: at the large orders that weight rare events.*
:::

## Geometric stopping on randomized response {#sec-sol-hpt-2}

> **Exercise 2.** **Geometric stopping on randomized response.** Run randomized response with $\varepsilon = 0.5$ a geometric number of times, stopping after each run with probability $\omega$, and release the best outcome. Using Fact 3 of the chapter, show that the log ratio of the probabilities of "bad" is $\varepsilon + \ln\frac{\omega + (1 - \omega)s}{\omega + (1 - \omega)(1 - s)}$, find its limits as $\omega \to 1$ and $\omega \to 0$, and evaluate it for $\omega = 0.5$, 0.1, 0.01 and 0.001. *Check: 0.6636, 0.9063, 0.9897 and 0.9990, approaching $2\varepsilon = 1$.* *(Generating functions)*

::: {.callout-tip collapse="true" title="Solution"}
1. **The probability of "bad".** The best is bad only if every run is bad, so by @eq-hpt-pgf $\Pr[M = \text{bad}] = f(\Pr[\text{run is bad}])$ with the geometric generating function $f(x) = \omega x/(1 - (1 - \omega)x)$.
2. **The ratio.** On $D$ the argument is $s$, on $D'$ it is $1 - s$:
$$\frac{f(s)}{f(1 - s)} = \frac{s}{1 - s}\cdot\frac{1 - (1 - \omega)(1 - s)}{1 - (1 - \omega)s} = e^\varepsilon\cdot\frac{\omega + (1 - \omega)s}{\omega + (1 - \omega)(1 - s)},$$
using $s/(1 - s) = e^\varepsilon$ and $1 - (1 - \omega)x = \omega + (1 - \omega)(1 - x)$.
3. **Limits.** As $\omega \to 1$ the second factor tends to 1, and the log ratio to $\varepsilon$: one run. As $\omega \to 0$ it tends to $s/(1 - s) = e^\varepsilon$, and the log ratio to $2\varepsilon$.
4. **Numbers.** With $\varepsilon = 0.5$: 0.6636, 0.9063, 0.9897 and 0.9990. The good outcome has a smaller log ratio, so these are the exact $\varepsilon$ of the release, as the script confirms.

*Two outcomes give only $2\varepsilon$, as the chapter said: the good outcome has nothing better than it, so one of the three factors of @eq-hpt-lt cannot move. The third factor of Liu and Talwar's $3\varepsilon$ needs an outcome with something above and something below it.*
:::

## The expected quantile of the best run {#sec-sol-hpt-3}

> **Exercise 3.** **The expected quantile of the best run.** If one run's score has a continuous distribution, its quantile $U$ (the fraction of runs it beats) is uniform on $[0, 1]$, and the best of $K$ runs has quantile $\max(U_1, \dots, U_K)$. Show that its expectation is $\mathbb E[K/(K + 1)] = 1 - \int_0^1f(x)\,\mathrm dx$, and evaluate it for four distributions with mean 8: fixed $K = 8$, Poisson, geometric and logarithmic. *Check: 0.8889, 0.8750, 0.8034 and 0.7360.* *(Order statistics and generating functions)*

::: {.callout-tip collapse="true" title="Solution"}
1. **Fixed $k$.** The maximum of $k$ uniforms is at most $x$ with probability $x^k$, so its mean is $\int_0^1(1 - x^k)\,\mathrm dx = 1 - \frac{1}{k + 1} = \frac{k}{k + 1}$.
2. **Random $K$.** Average over $K$: $\mathbb E[K/(K + 1)] = 1 - \mathbb E\int_0^1x^K\,\mathrm dx = 1 - \int_0^1\mathbb E[x^K]\,\mathrm dx = 1 - \int_0^1f(x)\,\mathrm dx$, exchanging the expectation and the integral because the integrand is non-negative.
3. **Fixed 8.** $8/9 = 0.8889$.
4. **Poisson.** $\int_0^1e^{\mathbb E[K](x - 1)}\,\mathrm dx = (1 - e^{-\mathbb E[K]})/\mathbb E[K]$, so the expected quantile is $1 - (1 - e^{-8})/8 = 0.8750$, almost that of a fixed 8.
5. **Geometric and logarithmic.** Numerical integration of their generating functions gives 0.8034 (geometric, $\omega = 1/8$) and 0.7360 (logarithmic, $\omega = 0.0363$).

*Papernot and Steinke use this measure of utility. Poisson random search is nearly as good as running a fixed 8, while the skewed distributions, which the chapter showed are cheaper in privacy, lose a lot: the logarithmic distribution's best run is, on average, at the 74th percentile instead of the 89th.*
:::

## The Poisson theorem for a pure-DP run {#sec-sol-hpt-4}

> **Exercise 4.** **The Poisson theorem for a pure-DP run.** An $\varepsilon$-DP run is $(\hat\varepsilon, \hat\delta)$-DP with $\hat\varepsilon = \varepsilon$ and $\hat\delta = 0$. Find the largest order $\alpha^*$ at which @eq-hpt-poisson applies with $\hat\delta = 0$, and show that the bound there is $\varepsilon(\alpha^*) + (e^\varepsilon - 1)\ln\mathbb E[K]$. Evaluate it for randomized response with $\varepsilon = 0.1$ and $\mathbb E[K] = 100$, and compare it with the exact divergence and with composition of 100 runs at the same order. *Check: $\alpha^* = 10.5083$; bound 0.5299, exact 0.0729, composition 4.5556.* *(Applying a theorem's side condition)*

::: {.callout-tip collapse="true" title="Solution"}
1. **The side condition.** @eq-hpt-poisson needs $e^{\hat\varepsilon} \le 1 + \frac{1}{\alpha - 1}$. With $\hat\varepsilon = \varepsilon$ this is $\alpha - 1 \le \frac{1}{e^\varepsilon - 1}$, so $\alpha^* = 1 + \frac{1}{e^\varepsilon - 1}$; for $\varepsilon = 0.1$, $\alpha^* = 10.5083$.
2. **The bound.** With $\hat\delta = 0$, $\varepsilon_M(\alpha^*) \le \varepsilon(\alpha^*) + \frac{\ln\mathbb E[K]}{\alpha^* - 1} = \varepsilon(\alpha^*) + (e^\varepsilon - 1)\ln\mathbb E[K]$.
3. **Numbers.** Randomized response with $\varepsilon = 0.1$ has $\varepsilon(10.5083) = 0.0456$, and $(e^{0.1} - 1)\ln 100 = 0.1052 \times 4.6052 = 0.4843$, so the bound is 0.5299.
4. **Compare.** The exact divergence of the best of a Poisson number of runs, from @eq-hpt-pgf, is 0.0729 at this order; composition of 100 runs gives $100 \times 0.0456 = 4.5556$. The bound is about nine times below composition and seven times above the exact value.

*For very private runs the Poisson bound is excellent, because the condition $e^{\hat\varepsilon} \le 1 + 1/(\alpha - 1)$ allows large orders. For a DP-SGD run with $\varepsilon$ of several units, the same condition forces either small orders or a $\hat\delta$ at a small $\hat\varepsilon$, which is why the Poisson bound grows almost linearly in @tbl-hpt-overhead.*
:::

## Calibrating the noise of a search {#sec-sol-hpt-5}

> **Exercise 5.** **Calibrating the noise of a search.** You want to tune the running example's DP-SGD runs of 10,000 steps with $q = 0.01$ over about 10 configurations, and the whole search must satisfy $\varepsilon = 6.71$ at $\delta = 10^{-5}$, the budget of one run at $\sigma = 1$. Find the noise multiplier each run needs under composition of 10 runs, and with a Poisson, geometric or logarithmic number of runs with mean 10. *Check: 2.45, 1.65, 1.47 and 1.31.* *(Inverting an accountant)*

::: {.callout-tip collapse="true" title="Solution"}
1. **The total as a function of $\sigma$.** For each $\sigma$ the accountant gives one run's RDP curve, $10{,}000\,\varepsilon_{q,\sigma}(\alpha)$. Composition multiplies it by 10 (@eq-hpt-compose); the random-$K$ bounds transform it by @eq-hpt-poisson or @eq-hpt-tnb, with the best second order $\hat\alpha$; @eq-hpt-convert converts the result.
2. **Monotonicity.** More noise lowers the curve at every order, and each of the transformations is increasing in the curve, so the total $\varepsilon$ falls as $\sigma$ grows. Bisection over $\sigma$ (to three significant digits, rounding up) finds the smallest $\sigma$ within budget.
3. **Numbers.** Composition needs $\sigma = 2.45$ (as in @tbl-hpt-compose); Poisson 1.65; geometric 1.47; logarithmic 1.31.
4. **Reading it.** With the logarithmic distribution each run is only 31% noisier than the untuned run, against 145% more noise for composition. Whether that buys accuracy depends on how often the skewed distribution stops early (Exercise 3).

*Search and noise are one budget. Fix the search procedure first, then solve for the noise multiplier that keeps the whole procedure within the target.*
:::

## How large a validation set? {#sec-sol-hpt-6}

> **Exercise 6.** **How large a validation set?** You select among the fixed candidates in $\mathcal R$, the candidate set, with the exponential mechanism at $\varepsilon = 0.5$, scoring each candidate by its number $u$ of correct predictions on $m$ private validation records. Using the tail bound $\Pr[\text{gap} \ge \frac{2\Delta u}{\varepsilon}(\ln|\mathcal R| + t)] \le e^{-t}$ of [Selection](../../book-full/part03-dp-foundations/05-selection.md), find the smallest $m$ that keeps the accuracy gap below 0.5 percentage points with probability at least 0.95, for $|\mathcal R| = 8$ and $|\mathcal R| = 1{,}000$. *Check: 4,061 and 7,923 records.* *(Utility of the exponential mechanism)*

::: {.callout-tip collapse="true" title="Solution"}
1. **From counts to accuracy.** With $\Delta u = 1$ the gap in correct predictions exceeds $\frac{2}{\varepsilon}(\ln|\mathcal R| + t)$ with probability at most $e^{-t}$. Dividing by $m$, the accuracy gap exceeds $\frac{2(\ln|\mathcal R| + t)}{\varepsilon m}$ with probability at most $e^{-t}$.
2. **Choose $t$.** A failure probability of 0.05 means $t = \ln 20 = 2.9957$.
3. **Solve for $m$.** $\frac{2(\ln|\mathcal R| + \ln 20)}{\varepsilon m} \le 0.005$ gives $m \ge \frac{2(\ln|\mathcal R| + \ln 20)}{0.5 \times 0.005}$. For $|\mathcal R| = 8$: $2 \times (2.0794 + 2.9957)/0.0025 = 4{,}060.1$, so 4,061. For $|\mathcal R| = 1{,}000$: $2 \times (6.9078 + 2.9957)/0.0025 = 7{,}922.8$, so 7,923.
4. **Reading it.** A 125-fold increase in candidates needs less than twice the validation data, because the cost is logarithmic in $|\mathcal R|$.

*Private selection among fixed candidates is cheap in data. The chapter's running example has a validation set of 5,000 records, enough for eight candidates at this precision; what tuning costs is the training of the candidates, not the choice among them.*
:::
