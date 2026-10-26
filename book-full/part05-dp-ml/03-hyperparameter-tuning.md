# Hyperparameter tuning and private selection

[The previous chapter](02-dp-sgd.md) proved a guarantee for one training run. Stochastic gradient descent (SGD) with per-example clipping and Gaussian noise, run once with a fixed learning rate, clipping norm, batch size and number of steps, satisfies differential privacy (DP) with the $(\varepsilon, \delta)$ that its accountant, the routine that adds up the privacy cost of the steps, reports; this algorithm is DP-SGD. But in practice nobody trains once. A practitioner tries a grid of learning rates, scores each trained model on a validation set, and ships the best one. The validation set and the training set are private data, and every run touched them. So this chapter asks how much the search leaks, and how to pay for it honestly.

The answer turns on one idea. The model you ship was chosen by comparing all the runs, so it is a function of all of them, and with a fixed number of runs it can cost as much as all of them together. If instead the *number* of runs is random, drawn before the search starts, then the event "every run came out badly" is mostly explained by "there were few runs" rather than by the data. The cost then falls to a constant multiple of one run, plus a term that grows only with the logarithm of the expected number of runs.

The running example is that of [DP-SGD](02-dp-sgd.md): a network with 3,329 parameters trained on $n = 10{,}000$ records, with a sampling rate $q$ of 0.01, a clipping norm $C$ of 0.5, a noise multiplier $\sigma$ of 1 and 10,000 steps. One run costs $\varepsilon = 6.71$ at $\delta = 10^{-5}$. Charging for every run in Rényi differential privacy (RDP) turns a grid of 10 runs into $\varepsilon = 27.23$, less than ten times 6.71 because the conversion to $(\varepsilon, \delta)$ pays its $\delta$ term once. A random number of runs with mean 10 costs between 10.08 and 13.67, depending on its distribution (@sec-hpt-rdp).

## The cost of releasing the best run {#sec-hpt-leak}

A choice among eight learning rates carries at most three bits, which seems harmless. But the released model is a whole private run, picked *because* of how it came out, and the comparison depends on every run. This section measures how much that can cost, because the rest of the chapter is about avoiding it.

Call one **run** a randomized algorithm $Q$ that trains a model on the private data with hyperparameters taken from a fixed set of **candidates**, and outputs the model with its **score**, such as a validation accuracy. Outputs are ordered by score, and $Q(D)$ also denotes the distribution of one run's output on the dataset $D$.

The **best-of-$K$** algorithm $M$ draws a number of runs $K$, independently of the data, runs $Q$ independently $K$ times, and releases the best output. Each run draws its candidate independently from a fixed distribution, as random search does. Neighbouring datasets $D \sim D'$ differ by adding or removing one record. Every result of this chapter needs the distribution of the best of $K$ draws, which the next lemma gives.

::: {#lem-hpt-pgf name="The best of a random number of draws"}
Let one draw have a distribution $Q$ on outcomes ordered from best to worst. Write $Q(\preceq y)$ for the probability that a draw is no better than $y$, and $Q(\prec y)$ for the probability that it is strictly worse. Let the number of draws $K$ be random and independent of the draws, with generating function
$$f(x) = \mathbb E[x^K] = \sum_k \Pr[K = k]\,x^k.$$
If $K = 0$, the release is a fixed output that does not depend on the data. Then for every other outcome $y$, the best of the $K$ draws is $y$ with probability
$$\Pr[M = y] = f\big(Q(\preceq y)\big) - f\big(Q(\prec y)\big).$$ {#eq-hpt-pgf}
A fixed $K = k$ has $f(x) = x^k$, and a Poisson number of draws with mean $\mathbb E[K]$ has $f(x) = e^{\mathbb E[K](x - 1)}$. A geometric number on $\{1, 2, \dots\}$, with $\Pr[K = k] = \omega(1 - \omega)^{k - 1}$ for a stopping probability $\omega \in (0, 1)$, has $f(x) = \omega x/(1 - (1 - \omega)x)$.
:::

::: {.proof}
The plan is to count outcomes for a fixed number of draws, then average over $K$, then sum the three series.

1. *A fixed number $k$ of draws.* The best of $k$ independent draws is no better than $y$ exactly when every draw is, which has probability $Q(\preceq y)^k$.
2. *The best is exactly $y$.* That happens when every draw is no better than $y$ but not every draw is strictly worse. The second event is contained in the first, so subtracting its probability gives
$$\Pr[\text{best of } k = y] = Q(\preceq y)^k - Q(\prec y)^k.$$
3. *Average over $K$.* The number of draws is independent of the draws, so averaging step 2 over $K$ replaces each power $x^k$ by its average $\sum_k \Pr[K = k]\,x^k = f(x)$. The term for $K = 0$ is the same constant in both generating functions and cancels.
4. *The three generating functions.* A fixed $k$ gives one term, the Poisson weights sum by the exponential series, and the geometric weights by the geometric series.
:::

A die shows the lemma at work. The best of two rolls is a 6 with probability $1 - (5/6)^2 = 0.3056$. If you roll once or twice with probability $\frac12$ each, then $f(x) = \frac12x + \frac12x^2$, and the best is a 6 with probability $f(1) - f(5/6) = 0.2361$, the average of the one-roll and two-roll answers.

The simplest run that shows the cost has two outcomes, as in [randomized response](../part03-dp-foundations/01-definition.md) (@def-dpdef-rr). With "bad" the worse outcome, @lem-hpt-pgf says that the best of $k$ is bad only when all $k$ runs are bad, so the likelihood ratios of the $k$ independent runs multiply.

The proposition below also bounds the cost in Rényi terms, to show that large orders see it too. For distributions $P$ and $P'$ on a finite set, the **Rényi divergence** of order $\alpha > 1$ is
$$D_\alpha(P\|P') = \frac{1}{\alpha - 1}\ln\sum_y P(y)^\alpha P'(y)^{1 - \alpha},$$
and an RDP curve $\varepsilon(\alpha)$ bounds it between outputs on neighbours ([Rényi differential privacy](../part04-privacy-accounting/01-renyi-dp.md), @def-rdp-rdp).

::: {#prp-hpt-rr-k name="The best of k randomized responses"}
Let one run be randomized response with two outcomes, "good" and "bad", with good the better. On $D$ the run comes out good with probability $\frac{1}{1 + e^\varepsilon}$, and on its neighbour $D'$ with probability $\frac{e^\varepsilon}{1 + e^\varepsilon}$, so one run is exactly $\varepsilon$-DP. Releasing the best of $k$ runs is exactly $k\varepsilon$-DP, because
$$\frac{\Pr_D[\text{best of } k \text{ is bad}]}{\Pr_{D'}[\text{best of } k \text{ is bad}]} = \frac{\big(e^\varepsilon/(1 + e^\varepsilon)\big)^k}{\big(1/(1 + e^\varepsilon)\big)^k} = e^{k\varepsilon}.$$ {#eq-hpt-rr-k}
At every order $\alpha > 1$ the release also has (Papernot & Steinke 2022, Proposition 17)
$$D_\alpha\big(M(D)\|M(D')\big) \ge k\varepsilon - \frac{k\ln(1 + e^{-\varepsilon})}{\alpha - 1}.$$
:::

::: {.proof}
We compute the probability of "bad" on each dataset, compare the other outcome, and then keep only the bad term of the Rényi sum.

1. *Bad needs every run bad.* No outcome is worse than "bad", so @lem-hpt-pgf with $f(x) = x^k$ gives $\Pr[\text{best is bad}] = Q(\text{bad})^k$ on each dataset.
2. *The ratio for bad.* One run is bad with probability $e^\varepsilon/(1 + e^\varepsilon)$ on $D$ and $1/(1 + e^\varepsilon)$ on $D'$. Raising both to the power $k$ gives @eq-hpt-rr-k.
3. *The ratio for good.* The good outcome has the ratio $(1 - Q_D(\text{bad})^k)/(1 - Q_{D'}(\text{bad})^k)$, which lies between $e^{-\varepsilon}$ and 1. The two datasets play symmetric roles, so the largest log ratio over outcomes and over both orders of the pair is $k\varepsilon$.
4. *The Rényi version.* Every term of the Rényi sum is positive, so the sum is at least its bad term. With the probabilities of step 2,
$$e^{(\alpha - 1)D_\alpha} \ge \Pr_D[\text{bad}]^\alpha\,\Pr_{D'}[\text{bad}]^{1 - \alpha} = e^{k\varepsilon\alpha}(1 + e^\varepsilon)^{-k}.$$
Taking logarithms, dividing by $\alpha - 1$ and writing $\ln(1 + e^\varepsilon) = \varepsilon + \ln(1 + e^{-\varepsilon})$ gives the bound.
:::

The whole loss sits on one event, that every run failed, whose ratio is the product of $k$ per-run ratios. @fig-hpt-leak (right) shows the privacy loss, the log ratio of an output's probabilities on $D$ and $D'$, for $\varepsilon = 1$. One run has losses $\pm1$, while the best of 10 has a loss of 10 with probability 0.0436. A DP-SGD run that sometimes diverges is such a run, with a "bad" outcome whose probability depends a little on the data.

The bound for a fixed $k$ is therefore tight. Composition says that the best of $k$ costs at most $k\varepsilon$ (@sec-hpt-composition), and @prp-hpt-rr-k exhibits a run that attains it. No analysis that knows only each run's guarantee can promise less, so a better guarantee needs a different procedure. The next sections change one thing about it: the number of runs.

![Left: the exact $\varepsilon$ of releasing the best of $K$ randomized-response runs, each $1$-DP, against the expected number of runs, for a fixed number of runs, a Poisson, a geometric and a logarithmic number. A fixed number grows linearly; the geometric and logarithmic numbers stay below their bounds $3\varepsilon$ and $2\varepsilon$. Right: the privacy loss of the released outcome under $D$ for one run, the best of 10 runs and the best of a Poisson number with mean 10; the best of 10 puts a loss of 10 on an event of probability 0.0436.](figures/hpt_leak.svg){#fig-hpt-leak}

::: {#exm-hpt-rr-k}
With $\varepsilon = 1$, a run is good with probability 0.2689 on $D$ and 0.7311 on $D'$. The largest log ratio is 2.0000 for $k = 2$, 3.0000 for $k = 3$ and 10.0000 for $k = 10$, exactly $k\varepsilon$. At order $\alpha = 2$, the Rényi divergence of the best of 10 is 6.8683. That lies between the lower bound of @prp-hpt-rr-k, 6.8674, and the composition bound, ten times one run's 0.7353, which is 7.3533. Composition overstates the exact value by only 7%, so the best of 10 costs almost as much as all 10 runs.
:::

## Paying by composition {#sec-hpt-composition}

The safe answer is to charge for every run. This section says what that costs, because it is the baseline every cleverer method must beat. The best output is computed from the $k$ outputs alone, so it leaks no more than all of them, and they together cost the sum of their RDP curves.

::: {#prp-hpt-compose name="Composition of k runs"}
Let each of $k$ runs have RDP curve $\varepsilon(\alpha)$, and let the runs use independent randomness. Then releasing anything computed from the $k$ outputs alone, in particular the best, has RDP curve
$$\varepsilon_M(\alpha) \le k\,\varepsilon(\alpha),$$ {#eq-hpt-compose}
even if each candidate is chosen after seeing the earlier results. For DP-SGD, $k$ runs of $T$ steps at noise multiplier $\sigma$ have the curve of one run of $kT$ steps.
:::

::: {.proof}
1. *The runs compose.* The curves of adaptively chosen mechanisms add (@thm-rdp-composition), so the $k$ outputs together have curve $k\varepsilon(\alpha)$.
2. *The release is post-processing.* The best output is computed from the $k$ outputs without reading the data, so its curve is no larger than theirs (@thm-rdp-post).
3. *DP-SGD.* Write $\varepsilon_{q,\sigma}(\alpha)$ for the curve of one subsampled Gaussian step. One run of $T$ steps then has curve $T\varepsilon_{q,\sigma}(\alpha)$, so $k$ runs have $kT\varepsilon_{q,\sigma}(\alpha)$, the curve of one run of $kT$ steps.
:::

Every $\varepsilon$ in this chapter comes from a curve by the conversion of [From RDP to (ε, δ)](../part04-privacy-accounting/02-rdp-conversions.md) (@thm-conv-improved), minimized over the order. In $(\varepsilon, \delta)$ terms, $k$ runs cost less than $k$ times one run's $\varepsilon$, because the conversion pays its $\ln(1/\delta)$ term once. So 10 runs of the running example cost 4.06 times one run, not 10. But the curve grows linearly, and so does $\varepsilon$ once the budget is large: from 30 to 100 runs, $\varepsilon$ grows by a factor of 2.5.

Raising the noise keeps the budget: at a total of $\varepsilon = 6.71$, each of 10 runs needs $\sigma = 2.45$ instead of 1, which costs accuracy ([DP-SGD](02-dp-sgd.md)). For two or three candidates, composition costs little more than the alternatives below. @fig-hpt-overhead shows that the methods with a random number of runs pull away from composition only from a few runs on.

::: {#exm-hpt-compose}
@tbl-hpt-compose takes the running example, $q = 0.01$, $\sigma = 1$ and 10,000 steps, for which one run costs $\varepsilon = 6.71$ at $\delta = 10^{-5}$, and composes $k$ runs.

| Runs $k$ | 1 | 3 | 10 | 30 | 100 |
| --- | --- | --- | --- | --- | --- |
| $\varepsilon$ of $k$ runs at $\sigma = 1$ | 6.71 | 12.73 | 27.23 | 58.25 | 145.17 |
| Ratio to one run | 1.00 | 1.90 | 4.06 | 8.68 | 21.63 |
| $\sigma$ per run for a total of 6.71 | 1 | 1.46 | 2.45 | 4.12 | 7.44 |

: Composition of $k$ DP-SGD runs of the running example, at $\delta = 10^{-5}$. {#tbl-hpt-compose}

:::

## Random stopping {#sec-hpt-stopping}

Liu & Talwar (2019) asked whether the best of many $\varepsilon$-DP runs can cost a constant multiple of $\varepsilon$, whatever the number of runs. They found that it can, with one change to the procedure: the number of runs becomes random. This section proves their result for pure DP runs, where the mechanism behind the Rényi versions of @sec-hpt-rdp is easiest to see.

::: {#def-hpt-stopping name="Random stopping"}
Fix a stopping probability $\omega \in (0, 1)$. Repeat the following: run $Q$ on the data and keep its output, then, with probability $\omega$, stop and release the best output kept so far. The number of runs $K$ is then geometric, with $\Pr[K = j] = \omega(1 - \omega)^{j - 1}$ for $j \ge 1$ and mean $1/\omega$.
:::

The probability of each output turns out to be a ratio built from three probabilities of one run. Each moves by at most a factor $e^\varepsilon$ between neighbours, and the stopping probability cannot amplify those moves.

::: {#thm-hpt-lt name="Random stopping is 3ε-DP (Liu & Talwar 2019, Theorem 3.2)"}
Let $Q$ be $\varepsilon$-DP, and let $M$ be random stopping with probability $\omega$. For an outcome $y$, write $p$ for the probability that one run outputs $y$, $p_>$ for the probability that one run is strictly better than $y$, and $p_\ge = p_> + p$. Then
$$\Pr[M = y] = \frac{\omega\,p}{\big(\omega + (1 - \omega)p_>\big)\big(\omega + (1 - \omega)p_\ge\big)},$$ {#eq-hpt-lt}
and $M$ is $3\varepsilon$-DP for every $\omega$. For every fraction $\pi \in (0, 1]$, the release falls below the top $\pi$-quantile of one run's score with probability
$$\Pr[\text{miss}] = \frac{\omega(1 - \pi)}{\omega + (1 - \omega)\pi} \le \frac{\omega}{\pi}.$$ {#eq-hpt-lt-utility}
:::

::: {.proof}
The plan is to apply @lem-hpt-pgf with the geometric generating function, simplify the result to @eq-hpt-lt, bound each of its three factors between neighbours, and finally sum a geometric series for the utility.

1. *Apply the lemma.* One run is no better than $y$ with probability $1 - p_>$ and strictly worse with probability $1 - p_\ge$. With $f(x) = \omega x/(1 - (1 - \omega)x)$, and since $1 - (1 - \omega)(1 - p_>) = \omega + (1 - \omega)p_>$, @eq-hpt-pgf gives
$$\Pr[M = y] = \frac{\omega(1 - p_>)}{\omega + (1 - \omega)p_>} - \frac{\omega(1 - p_\ge)}{\omega + (1 - \omega)p_\ge}.$$
2. *Simplify.* Over the common denominator, the numerator is $\omega$ times $(1 - p_>)(\omega + (1 - \omega)p_\ge) - (1 - p_\ge)(\omega + (1 - \omega)p_>)$. The terms with $\omega$ give $\omega(p_\ge - p_>) = \omega p$, and the terms with $1 - \omega$ give $(1 - \omega)(p_\ge - p_>) = (1 - \omega)p$. Their sum is $p$, which gives @eq-hpt-lt.
3. *Bound each factor.* Write primes for the same quantities on $D'$. Each of $p$, $p_>$ and $p_\ge$ is the probability of an event about one run's output, so $\varepsilon$-DP gives $p \le e^\varepsilon p'$, $p'_> \le e^\varepsilon p_>$ and $p'_\ge \le e^\varepsilon p_\ge$. The constant $\omega$ does not depend on the data, so $\omega + (1 - \omega)p'_> \le e^\varepsilon\big(\omega + (1 - \omega)p_>\big)$, and the same holds with $p_\ge$.
4. *Multiply.* The ratio of output probabilities is the product of the numerator ratio and the two inverted denominator ratios, each at most $e^\varepsilon$ by step 3:
$$\frac{\Pr[M(D) = y]}{\Pr[M(D') = y]} = \frac{p}{p'}\cdot\frac{\omega + (1 - \omega)p'_>}{\omega + (1 - \omega)p_>}\cdot\frac{\omega + (1 - \omega)p'_\ge}{\omega + (1 - \omega)p_\ge} \le e^{3\varepsilon}.$$
The same argument with $D$ and $D'$ swapped bounds the inverse ratio, and this holds for every outcome, so the release is $3\varepsilon$-DP.
5. *Utility.* The release is below the top $\pi$-quantile only if every run is. Summing over the number of runs with the geometric series gives
$$\Pr[\text{miss}] = \sum_{j \ge 1}\omega(1 - \omega)^{j - 1}(1 - \pi)^j = \frac{\omega(1 - \pi)}{1 - (1 - \omega)(1 - \pi)},$$
and the denominator equals $\omega + (1 - \omega)\pi$.
6. *The simple bound.* The miss probability is at most $\omega/\pi$ exactly when $\pi(1 - \pi) \le \omega(1 - \pi) + \pi$, which holds because $\pi(1 - \pi) \le \pi$.
:::

This is why the disaster of a fixed $k$ disappears. With $f(x) = x^k$ the ratio for the worst outcome is a $k$-th power, while here it is a product of three bounded ratios. The event "every run was bad" is now mostly the event "the coin stopped early", which does not depend on the data. The factor 3 is attained in the limit, by an outcome with something above it and something below it (@exm-hpt-stopping). On randomized response the good outcome has nothing above it, so one factor cannot move and the cost tends to $2\varepsilon$ ([Exercise 2](../../book-exercises/part05-dp-ml/03-hyperparameter-tuning.md#sec-sol-hpt-2) derives it).

The utility guarantee is about quantiles, and it costs runs. @eq-hpt-lt-utility does not promise the best candidate, only a release in the top $\pi$ fraction of one run's outcomes with probability at least $1 - \omega/\pi$. To land in the top $1/m$ of a grid of $m$ with probability 0.9, set $\pi = 1/m$ and $\omega/\pi = 0.1$, which gives $\omega = 0.1/m$, a mean of $10m$ runs.

::: {#exm-hpt-stopping}
On randomized response with $\varepsilon = 1$, geometric stopping has an exact $\varepsilon$ of 1.3106 at $\omega = 0.5$, 1.7957 at $\omega = 0.1$, 1.9769 at $\omega = 0.01$ and 1.9977 at $\omega = 0.001$, approaching 2. To reach 3, take three outcomes, best first, with probabilities $(a, b, 1 - a - b)$ on $D$ and $(ae^\varepsilon, be^{-\varepsilon}, \dots)$ on $D'$, and let $\omega \ll b \ll a$ be small. For the middle outcome, the numerator ratio of @eq-hpt-lt is $e^\varepsilon$. Since $\omega$ is negligible next to $p_>$, and $p_\ge$ is close to $p_>$, both denominator ratios approach $e^\varepsilon$ too. @tbl-hpt-three shows the exact $\varepsilon$ approaching 3, and that of the logarithmic number of runs of @sec-hpt-rdp approaching 2.

| $a$ | $b$ | $\omega$ | Geometric stopping | Logarithmic number, same $\omega$ |
| --- | --- | --- | --- | --- |
| 0.01 | 0.001 | 0.1 | 1.2598 | 1.1299 |
| 0.01 | 0.0001 | 0.01 | 2.2316 | 1.6158 |
| 0.01 | $10^{-6}$ | $10^{-4}$ | 2.9874 | 1.9937 |
| 0.001 | $10^{-9}$ | $10^{-6}$ | 2.9987 | 1.9994 |

: Exact $\varepsilon$ of the best-of release on a three-outcome mechanism that is exactly 1-DP, computed from @eq-hpt-pgf. The middle outcome attains the maximum, and @eq-hpt-lt gives the same value. {#tbl-hpt-three}

For utility, take a grid of 8 candidates with one best. With $\omega = 1/8$, a mean of 8 runs, the chance of never trying the top eighth is 0.4667. With $\omega = 1/80$, a mean of 80 runs, it falls to 0.0805 against a bound of 0.1. A target ten times finer, the top $1/80$ of a finer grid, is then still missed with probability 0.4969.
:::

## Rényi bounds for a random number of runs {#sec-hpt-rdp}

DP-SGD is not usefully $\varepsilon$-DP for any small $\varepsilon$, because its guarantee is an RDP curve that converts to $(\varepsilon, \delta)$. Papernot & Steinke (2022) extended random stopping to RDP and to a family of distributions for the number of runs, which lets you trade more runs against a lower privacy cost. This section proves their generic bound and its Poisson case, and states a third case. (They write $\lambda$ for the order, $\mu$ for the Poisson mean and $\gamma$ for the stopping probability.)

The generic bound separates one run's divergence from a factor that depends only on the derivative $f'$ of the generating function at two points. Those points are acceptance probabilities of a test that reads only a run's output, which is what makes the bound useful: any privacy guarantee of the run keeps them close.

::: {#thm-hpt-generic name="The generic bound (Papernot & Steinke 2022, Lemma 7)"}
Let $K$ have generating function $f$ and be independent of the runs, and let $M$ release the best of $K$ independent runs of $Q$. For a pair of neighbours, write $Q, Q'$ for the two run distributions, and $M, M'$ also for the two best-of distributions. Then for every order $\alpha > 1$ there are numbers $s, s' \in [0, 1]$ such that
$$D_\alpha(M\|M') \le D_\alpha(Q\|Q') + \frac{1}{\alpha - 1}\ln\Big[f'(s)^\alpha f'(s')^{1 - \alpha}\Big],$$ {#eq-hpt-generic}
where $s$ and $s'$ are the probabilities that one and the same randomized test (step 5 of the proof) accepts a run's output, on $D$ and on $D'$.
:::

::: {.proof}
Take the outputs finite and $\Pr[K = 0] = 0$ to keep the steps short; the paper handles the general case, including the Poisson distribution, for which $K$ can be 0. We write each best-of probability as one run's probability times an average of $f'$, put that into the Rényi sum, bound the averages by their worst case, and recognize the worst case as a test.

1. *The best-of probability is an average of $f'$.* By @eq-hpt-pgf and the fundamental theorem of calculus,
$$M(y) = f\big(Q(\preceq y)\big) - f\big(Q(\prec y)\big) = \int_{Q(\prec y)}^{Q(\preceq y)}f'(x)\,\mathrm dx.$$
The interval has length $Q(y)$, so $M(y) = Q(y)\,c(y)$, where $c(y)$ is the average of $f'$ over that interval, the expectation of $f'(X)$ for $X$ uniform on it. Likewise $M'(y) = Q'(y)\,c'(y)$.
2. *Put it in the Rényi sum.* Substituting step 1 into the definition of the divergence gives
$$e^{(\alpha - 1)D_\alpha(M\|M')} = \sum_yM(y)^\alpha M'(y)^{1 - \alpha} = \sum_yQ(y)^\alpha Q'(y)^{1 - \alpha}\,c(y)^\alpha c'(y)^{1 - \alpha}.$$
3. *Bound the averages by a worst case.* Couple $X$ and $X'$ at the same relative position $t \in [0, 1]$ in their two intervals. The function $(u, v) \mapsto u^\alpha v^{1 - \alpha}$ is jointly convex for $\alpha > 1$, so by Jensen's inequality $c(y)^\alpha c'(y)^{1 - \alpha}$ is at most the average of $f'(X)^\alpha f'(X')^{1 - \alpha}$. That average is at most the largest value of $f'(s)^\alpha f'(s')^{1 - \alpha}$ over $t$, with $s = Q(\prec y) + tQ(y)$ and $s' = Q'(\prec y) + tQ'(y)$.
4. *Pull the worst case out of the sum.* Bounding every factor $c^\alpha c'^{1 - \alpha}$ by the largest one over $y$ and $t$ leaves $\sum_yQ(y)^\alpha Q'(y)^{1 - \alpha} = e^{(\alpha - 1)D_\alpha(Q\|Q')}$ times that maximum. Taking logarithms and dividing by $\alpha - 1$ gives @eq-hpt-generic.
5. *The pair comes from one test.* The test "accept an output worse than $y$, accept $y$ itself with probability $t$, reject anything better" accepts with probability $s$ under $Q$ and $s'$ under $Q'$. The test reads only the run's output, so any privacy guarantee of $Q$ constrains the pair $(s, s')$, by post-processing (@thm-dpprop-post, @thm-rdp-post).
:::

The first term is one run's divergence. The second is what the distribution of $K$ adds, and each bound below comes from choosing $f$ and bounding it.

For a Poisson number of runs, $f'$ is an exponential, so the second term is linear in $s$ and $s'$. An $(\hat\varepsilon, \hat\delta)$-DP guarantee of the run, applied to the event that the test rejects, then bounds it.

::: {#thm-hpt-poisson name="A Poisson number of runs (Papernot & Steinke 2022, Theorem 6)"}
Let $K$ be Poisson with mean $\mathbb E[K]$. Let $Q$ be $(\alpha, \varepsilon(\alpha))$-RDP, and also $(\hat\varepsilon, \hat\delta)$-DP for some $\hat\varepsilon$ with $e^{\hat\varepsilon} \le 1 + \frac{1}{\alpha - 1}$. Then
$$\varepsilon_M(\alpha) \le \varepsilon(\alpha) + \mathbb E[K]\hat\delta + \frac{\ln\mathbb E[K]}{\alpha - 1}.$$ {#eq-hpt-poisson}
:::

::: {.proof}
1. *Compute $f'$.* With $f(x) = e^{\mathbb E[K](x - 1)}$, the derivative is $f'(x) = \mathbb E[K]e^{\mathbb E[K](x - 1)}$. Taking the logarithm of the second term of @eq-hpt-generic and regrouping the exponents gives
$$\ln\Big[f'(s)^\alpha f'(s')^{1 - \alpha}\Big] = \ln\mathbb E[K] + \mathbb E[K]\big[(\alpha - 1)(1 - s') - \alpha(1 - s)\big].$$
2. *Use the DP guarantee on the rejection event.* The numbers $1 - s'$ and $1 - s$ are the probabilities that the test of @thm-hpt-generic rejects, on $D'$ and on $D$. The run is $(\hat\varepsilon, \hat\delta)$-DP, so $1 - s' \le e^{\hat\varepsilon}(1 - s) + \hat\delta$, and the bracket is at most
$$(1 - s)\big[(\alpha - 1)e^{\hat\varepsilon} - \alpha\big] + (\alpha - 1)\hat\delta.$$
3. *The condition removes the first term.* The condition $e^{\hat\varepsilon} \le 1 + \frac{1}{\alpha - 1}$ is the same as $(\alpha - 1)e^{\hat\varepsilon} \le \alpha$, so the first term is not positive and the bracket is at most $(\alpha - 1)\hat\delta$. Dividing by $\alpha - 1$, and adding $D_\alpha(Q\|Q') \le \varepsilon(\alpha)$ in @eq-hpt-generic, gives @eq-hpt-poisson.
:::

The Poisson bound pays $\mathbb E[K]\hat\delta$ for the chance of many runs. That is linear in the expected number of runs, but multiplied by the run's $\hat\delta$ at a small $\hat\varepsilon$, so it is small only when one run is very private. On randomized response with $\varepsilon = 0.5$ and mean 10, the run has $\hat\delta = 0$ at $\hat\varepsilon = 0.5$, so @eq-hpt-poisson applies at the order where $e^{0.5} = 1 + 1/(\alpha - 1)$, which is $\alpha = 2.5415$. There the exact divergence is 0.4421, the bound 1.7654 and composition of 10 runs 2.7165; [Exercise 4](../../book-exercises/part05-dp-ml/03-hyperparameter-tuning.md#sec-sol-hpt-4) works out the pure-DP case.

A bound that stays a constant multiple of one run, whatever the run's $\hat\delta$, needs another distribution for $K$, from a family that contains the geometric distribution.

::: {#def-hpt-tnb name="Truncated negative binomial number of runs"}
For a shape $\nu > -1$ and $\omega \in (0, 1)$, the **truncated negative binomial** distribution on $\{1, 2, \dots\}$ has
$$\Pr[K = k] = \frac{(1 - \omega)^k}{\omega^{-\nu} - 1}\prod_{i = 0}^{k - 1}\frac{i + \nu}{i + 1}, \qquad \mathbb E[K] = \frac{\nu(1 - \omega)}{\omega(1 - \omega^\nu)}.$$
The shape $\nu$ sets how skewed $K$ is: the smaller $\nu$, the more often $K = 1$. The case $\nu = 1$ is the geometric distribution. The limit $\nu = 0$ is the **logarithmic distribution**, with $\Pr[K = k] = \frac{(1 - \omega)^k}{k\ln(1/\omega)}$ and mean $\frac{1/\omega - 1}{\ln(1/\omega)}$.
:::

The theorem for this family uses the run's RDP guarantee at a second order $\hat\alpha$, chosen freely, to keep the rejection probabilities close.

::: {#thm-hpt-tnb name="A truncated negative binomial number of runs (Papernot & Steinke 2022, Theorem 2)"}
Let $K$ follow the distribution of @def-hpt-tnb, and let $Q$ be $(\alpha, \varepsilon(\alpha))$-RDP and $(\hat\alpha, \varepsilon(\hat\alpha))$-RDP for a second order $\hat\alpha \ge 1$. Then
$$\varepsilon_M(\alpha) \le \varepsilon(\alpha) + (1 + \nu)\Big(1 - \frac1{\hat\alpha}\Big)\varepsilon(\hat\alpha) + \frac{(1 + \nu)\ln(1/\omega)}{\hat\alpha} + \frac{\ln\mathbb E[K]}{\alpha - 1}.$$ {#eq-hpt-tnb}
If $Q$ is $\varepsilon$-DP, letting both orders grow gives $(2 + \nu)\varepsilon$-DP (their Corollary 3). That is $3\varepsilon$ for the geometric distribution, as in @thm-hpt-lt, and $2\varepsilon$ for the logarithmic.
:::
<!-- proof: in @sec-apx-learn-tuning -->

The proof follows that of @thm-hpt-poisson, now with $f'(x) = \omega^{\nu + 1}\mathbb E[K]\,(1 - (1 - \omega)x)^{-\nu - 1}$. Step 2 now bounds the rejection probabilities through the run's RDP guarantee at the order $\hat\alpha$. Papernot & Steinke (2022, Appendix B.2) carry out the algebra, and Appendix @sec-apx-learn-tuning does it step by step, with their Corollaries 3 and 4.

Each term of @eq-hpt-tnb pays for something. The first is one run. The last is the price of the number of runs, and it is logarithmic, so ten times more runs in expectation adds only $\ln 10/(\alpha - 1)$. The middle terms pay for how far the distribution of $K$ lets an adversary tell "many runs, all bad" from "few runs", and they are a constant times the run's own privacy at another order. For a run with a linear RDP curve, $\varepsilon(\alpha) = \rho\alpha$, Papernot & Steinke's Corollary 4 turns the bound into a closed form whose overhead grows like $\sqrt{\ln\mathbb E[K]}$.

The bounds hold for every run with the given curve, so they can be loose for one run. On randomized response with $\varepsilon = 0.5$ and a logarithmic number of runs with mean 10, @tbl-hpt-rr-check compares the exact Rényi divergence of the release with @eq-hpt-tnb and with composition of 10 runs. Randomized response is pure DP, so the bound keeps falling as $\hat\alpha$ grows, and the table gives its limit, in which the two middle terms become $(1 + \nu)\varepsilon = 0.5$. At order 2, @eq-hpt-tnb is worse than composition, because its constant terms dominate at small orders. At orders 5 and 20 it is 2.6 and 4.3 times below composition, and 3.7 and 1.7 times above the exact value.

| Order $\alpha$ | Exact $D_\alpha$ | Bound @eq-hpt-tnb | Composition of 10 runs |
| --- | --- | --- | --- |
| 2 | 0.1438 | 3.0299 | 2.2734 |
| 5 | 0.3912 | 1.4599 | 3.8424 |
| 20 | 0.6379 | 1.0962 | 4.7505 |

: Randomized response with $\varepsilon = 0.5$, best of a logarithmic number of runs with mean 10: the exact Rényi divergence against the bound and against composition. {#tbl-hpt-rr-check}

![The $\varepsilon$ at $\delta = 10^{-5}$ of releasing the best of several DP-SGD runs of the running example ($q = 0.01$, $\sigma = 1$, 10,000 steps, one run $\varepsilon = 6.71$), against the expected number of runs, by composition, by the Poisson bound and by the truncated negative binomial bound with $\nu = 1$ (geometric), $0.5$ and $0$ (logarithmic). Both axes are logarithmic.](figures/hpt_overhead.svg){#fig-hpt-overhead}

![Four distributions of the number of runs, each with mean 8: Poisson, geometric, logarithmic and a fixed 8. The logarithmic distribution stops after one run with probability 0.29.](figures/hpt_k_distributions.svg){#fig-hpt-k}

::: {#exm-hpt-overhead}
The chapter's script (@sec-hpt-code) evaluates @eq-hpt-poisson and @eq-hpt-tnb on the RDP curve of one DP-SGD run of the running example, with the best $\hat\alpha$, and with the $\hat\delta$ that @eq-conv-improved-delta gives at $\hat\varepsilon = \ln(1 + 1/(\alpha - 1))$. It uses that a bound at a larger order holds at a smaller one (@prp-rdp-ends), and converts with @eq-conv-improved (@tbl-hpt-overhead, @fig-hpt-overhead).

| Expected runs | 3 | 10 | 30 | 100 | 1,000 |
| --- | --- | --- | --- | --- | --- |
| Composition | 12.73 | 27.23 | 58.25 | 145.17 | 1035.93 |
| Poisson | 8.93 | 13.67 | 26.43 | 67.77 | 302.57 |
| Geometric ($\nu = 1$) | 9.22 | 11.31 | 12.85 | 14.32 | 16.76 |
| $\nu = 0.5$ | 9.03 | 10.75 | 11.98 | 13.15 | 15.09 |
| Logarithmic ($\nu = 0$) | 8.75 | 10.08 | 11.04 | 11.95 | 13.46 |

: The $\varepsilon$ at $\delta = 10^{-5}$ of tuning with DP-SGD runs of the running example (one run: $\varepsilon = 6.71$), by the number of runs and how it is chosen. {#tbl-hpt-overhead}

For 10 runs in expectation, the logarithmic distribution costs 10.08, 1.5 times one run, against 27.23 for composition, and from 10 to 1,000 runs it adds only 3.38 more. The Poisson bound starts well, at 8.93 for 3 runs, but grows nearly linearly, because a run at $\varepsilon = 6.71$ has a large $\hat\delta$ at the small $\hat\varepsilon$ the condition forces.

The cheaper bounds come with a more skewed number of runs. With mean 10, the logarithmic distribution has $\omega = 0.02692$ and stops after a single run with probability 0.2692, against 0.1000 for the geometric and 0.1562 for $\nu = 0.5$. The Poisson distribution, by contrast, is concentrated near its mean: with mean 10, $\Pr[K \le 1] = 4.99 \times 10^{-4}$. @fig-hpt-k shows the four distributions with mean 8.
:::

## Selecting by a noisy validation score {#sec-hpt-select}

The methods so far treat each run as a black box that outputs a model and its score. Often, though, the candidates are already fixed, such as public pretrained checkpoints, and only the choice among them uses private data, a held-out validation set. That is a selection problem, which [Selection](../part03-dp-foundations/05-selection.md) solved. This section says how those tools apply to tuning, and what they leave out: the cost of training the candidates, and the protection of the validation set.

Let the score of candidate $r$ be its number of correct predictions $u(r)$ on a private validation set of $m$ records. Adding or removing one validation record adds or removes one prediction, which is right or wrong, so every count moves by at most $\Delta u = 1$. The **exponential mechanism** picks $r$ from the candidate set $\mathcal R$ with probability proportional to $e^{\varepsilon u(r)/(2\Delta u)}$. It is $\varepsilon$-DP in the validation records (@thm-sel-em). Its expected gap to the best count is at most $\frac{2\Delta u}{\varepsilon}(\ln|\mathcal R| + 1)$ (@thm-sel-em-utility), an accuracy gap of $\frac{2}{\varepsilon m}(\ln|\mathcal R| + 1)$.

When the candidates were themselves trained on private data, the cost has two parts, which fall on different records.

::: {#prp-hpt-select name="Trained candidates and a disjoint validation set"}
Let $k$ candidates be trained on a private training set by runs with RDP curves $\varepsilon_1(\alpha), \dots, \varepsilon_k(\alpha)$. Let a selection mechanism choose among them on a private validation set disjoint from the training set, with RDP curve $\varepsilon_{\text{sel}}(\alpha)$ in the validation records for every fixed set of candidates. Then releasing the selected model costs
$$\varepsilon_M(\alpha) \le \max\Big(\sum_{j = 1}^k\varepsilon_j(\alpha),\ \varepsilon_{\text{sel}}(\alpha)\Big).$$ {#eq-hpt-select}
:::

::: {.proof}
1. *A training record.* If the neighbours differ in a training record, the validation set and the selection's noise are the same on both and do not depend on it. The release is then a post-processing (@thm-rdp-post) of the $k$ trained models, which compose (@thm-rdp-composition) to $\sum_j\varepsilon_j(\alpha)$.
2. *A validation record.* If the neighbours differ in a validation record, the trained models have the same distribution on both. Given the models, the release is the selection applied to fixed candidates, with curve at most $\varepsilon_{\text{sel}}(\alpha)$, and mixing both sides over the same distribution of models keeps the divergence below that worst case.
3. *Combine.* Neighbours differ in one record, which lies in one of the two sets, so the larger bound covers every pair, as in parallel composition (@thm-dpprop-parallel).
:::

Selection is cheap, and training is not. The expensive part is that each candidate was *trained* on private data, which @eq-hpt-select still charges by composition. If the candidates do not depend on private data, as with public checkpoints, that term vanishes, and choosing among them costs $\ln|\mathcal R|$ in utility and no linear cost in privacy.

The validation set needs protection under every method. The best-of methods of @sec-hpt-stopping and @sec-hpt-rdp assume that the score is part of each run's private output. If the score is an exact validation accuracy, the validation records are not protected at all. Adding Gaussian noise of standard deviation $\sigma_{\text{val}}$ to each validation count makes each score a Gaussian mechanism with curve $\alpha/(2\sigma_{\text{val}}^2)$ (@thm-rdp-curves). By step 3 of the proof of @prp-hpt-select, the counts then cost the larger of the training curve and their own composed curve. With $\sigma_{\text{val}} = 20$ correct predictions, one noisy count alone has $\varepsilon = 0.182$ at $\delta = 10^{-5}$, and the eight counts of a search, composed, have $0.546$, far below eight training runs.

::: {#exm-hpt-em}
In the experiment of @sec-hpt-run, eight DP-SGD runs at $\sigma = 2.07$, with learning rates from 0.0625 to 8, get these counts out of $m = 5{,}000$: 4,486, 4,604, 4,681, 4,694, 4,600, 4,470, 4,232 and 4,123. @tbl-hpt-em selects among them with the exponential mechanism.

| $\varepsilon_{\text{sel}}$ | $\Pr[\text{best}]$ | Expected gap (correct predictions) | Accuracy gap | Bound $2(\ln 8 + 1)/\varepsilon_{\text{sel}}$ |
| --- | --- | --- | --- | --- |
| 0.01 | 0.2478 | 87.62 | 1.752% | 615.9 |
| 0.1 | 0.6484 | 5.61 | 0.112% | 61.6 |
| 1 | 0.9985 | 0.02 | 0.000% | 6.2 |

: The exponential mechanism selecting among eight trained models by their correct predictions on 5,000 private validation records. {#tbl-hpt-em}

At $\varepsilon_{\text{sel}} = 0.1$ the selection costs about a tenth of a point of accuracy. The training costs much more. Each run at $\sigma = 2.07$ and 1,000 steps has $\varepsilon = 0.657$, and the eight together 1.994. Selecting instead by the eight noisy counts of the best-of methods leaves the total at 1.994, because the counts' composed curve is below the training curve at the orders the conversion uses.
:::

## Tuning the running example {#sec-hpt-run}

This section puts the strategies side by side on the running example, at the same total budget, and asks what each delivers in accuracy, which is the comparison a practitioner faces.

::: {#def-hpt-strategies name="Six ways to tune the learning rate"}
Each run trains the running example's network with DP-SGD ($q = 0.01$, $C = 0.5$) for 1,000 steps, with one of eight learning rates $\eta \in \{0.0625, 0.125, \dots, 8\}$, and is scored by its correct predictions on 5,000 private validation records plus Gaussian noise with $\sigma_{\text{val}} = 20$. Each strategy releases one model and chooses its noise multiplier $\sigma$ so that the whole procedure meets $\varepsilon = 2$ at $\delta = 10^{-5}$.

1. *No tuning:* one run with the learning rate 0.5 that [DP-SGD](02-dp-sgd.md) used.
2. *The reporting bug:* the eight runs at the $\sigma$ of one run, the best released, and the single-run $\varepsilon$ reported, though the search really spends more.
3. *Proxy data:* the grid run on a related public task, then one private run with its best learning rate.
4. *Composition:* the eight runs at the $\sigma$ for which eight runs together cost 2 (@prp-hpt-compose).
5. *Poisson random search:* $K$ runs with learning rates drawn uniformly from the grid, $K$ Poisson with mean 8 (@thm-hpt-poisson).
6. *Logarithmic random search:* the same with $K$ logarithmic (@thm-hpt-tnb).
:::

Runs take 1,000 steps instead of 10,000 to keep the script fast. Of the running example's 20,000 held-out records, 5,000 form the validation set and 15,000 the test set. The proxy task is labelled by a different random network and touches no private data, so it costs nothing (@thm-dpprop-post). The script trains three runs per learning rate for each $\sigma$ and repeats each procedure 4,000 times by drawing from this pool, with fresh validation noise and a fresh $K$, which understates the variety of a real search a little.

::: {#exm-hpt-run}
@tbl-hpt-run gives the result. Suppose one paper reports "test accuracy 0.948 at $\varepsilon = 1.97$" for a model chosen from eight learning rates, and another 0.943 at $\varepsilon = 1.99$ after an honest grid search. The first is the reporting bug of @tbl-hpt-run, which spends 5.60, so the difference is the accounting. At its honest cost of 5.60, the first row should be compared with methods at $\varepsilon = 5.60$.

| Strategy | $\sigma$ | $\varepsilon$ of one run | Total $\varepsilon$ | Test accuracy | Runs on private data |
| --- | --- | --- | --- | --- | --- |
| No tuning: learning rate 0.5 from [DP-SGD](02-dp-sgd.md) | 1.03 | 1.97 | 1.97 | $0.9509 \pm 0.0015$ | 1 |
| Grid of 8, single-run $\varepsilon$ reported | 1.03 | 1.97 | 5.60 | $0.9479 \pm 0.0030$ | 8 |
| Tune on proxy data, then one run | 1.03 | 1.97 | 1.97 | $0.9487 \pm 0.0025$ | 1 |
| Grid of 8, composition | 2.07 | 0.66 | 1.99 | $0.9430 \pm 0.0022$ | 8 |
| Random search, Poisson, mean 8 | 1.53 | 0.98 | 1.99 | $0.9433 \pm 0.0060$ | 8.01 |
| Random search, logarithmic, mean 8 | 1.36 | 1.17 | 2.00 | $0.9372 \pm 0.0200$ | 7.95 |

: Tuning the learning rate of the running example (1,000 steps per run) at a total budget of $\varepsilon = 2$, $\delta = 10^{-5}$. Test accuracy: mean and standard deviation over repetitions of the whole procedure; runs: the average number of private training runs. {#tbl-hpt-run}

:::

The honest strategies cost about a point of accuracy here, and the bug hides it. At the same total budget, composition reaches 0.9430 and Poisson random search 0.9433, against 0.9509 without tuning, while the bug claims 1.97 and spends up to 5.60. Even with its budget misreported, the grid search, at 0.9479, is below the untuned learning rate. The learning rate 0.5 is the best of the grid, and the noisy validation count sometimes picks a neighbour. It is the best because [DP-SGD](02-dp-sgd.md) chose its settings with knowledge of this very task, a leak that no accountant in the book has charged for. Tuning on proxy data is nearly free, reaching 0.9487 for the cost of one private run.

Among the honest searches, Poisson random search allows $\sigma = 1.53$ instead of composition's 2.07, so each run costs 0.98 instead of 0.66. But each learning rate is drawn a Poisson number of times with mean 1, so it is missed with probability $e^{-1} = 0.3679$, and its accuracy varies more from one search to the next, 0.0060 against 0.0022, so at this small grid the lower noise and the missed learning rates roughly cancel. The logarithmic distribution allows the least noise, $\sigma = 1.36$, but with mean 8 it stops after a single run with probability 0.2907 and then releases a random learning rate. That is why its accuracy is the lowest and most variable, $0.9372 \pm 0.0200$. @fig-hpt-accuracy repeats the comparison at $\varepsilon = 1$ and 4.

![Test accuracy of the model released by each tuning strategy against the total $\varepsilon$ at $\delta = 10^{-5}$ it costs (1,000 DP-SGD steps per run). The red dashed line places the grid search at the single-run $\varepsilon$ that the reporting bug claims, and the arrows move each point to its honest cost by composition. Points at $\varepsilon = 2$ use three runs per learning rate; those at 1 and 4 one run.](figures/hpt_accuracy.svg){#fig-hpt-accuracy}

## What to do and what to report {#sec-hpt-practice}

The results of this chapter turn into a short procedure, and into a rule for what a reported $\varepsilon$ must cover.

::: {#def-hpt-procedure name="An honest tuning procedure"}
1. *Spend private runs only where you must.* Tune on public or proxy data when it exists, and reuse hyperparameters from similar tasks. Shrink the search: below the typical gradient norm only the product of learning rate and clipping norm matters ([DP-SGD](02-dp-sgd.md#sec-dpsgd-clipping)).
2. *Fix the search before touching the private data.* Write down the candidate set, the distribution of the number of runs, the score and the validation set.
3. *Make the score private.* Use a held-out validation set and a noisy count.
4. *Account for the whole search.* For a few candidates, compose (@prp-hpt-compose). For more, use random search with @thm-hpt-poisson or @thm-hpt-tnb. Count every run on the private data, including crashed runs, debugging runs and evaluations for early stopping.
5. *Report both numbers.* Give the $\varepsilon$ of the released run, and the $\varepsilon$ of the whole procedure with how it was computed: the candidate set, the distribution of $K$, the accountant, $\delta$ and the neighbouring relation.
:::

The cheapest fix is usually upstream: in @tbl-hpt-run, the strategies that spend the whole budget on one private run beat every honest search. Tuning without noise does not escape the problem, because the hyperparameters are then a non-private function of the data. Papernot & Steinke (2022, Section 2) build a support vector machine whose best regularization weight reveals whether a few outliers were in the training set, a [membership inference](../part02-attacks/02-membership-inference.md) through a single number. Retraining the chosen configuration with fresh noise does not make the tuning free either, because the configuration is still a function of the tuning runs.

Fixing the search in advance is part of the guarantee. The theorems of @sec-hpt-stopping and @sec-hpt-rdp require that each run draws its candidate independently from a fixed distribution, which is random search, and that $K$ is drawn independently of the data. Adaptive methods such as Bayesian optimization, which pick the next candidate from the scores so far, are not covered and need [Adaptive composition](../part04-privacy-accounting/08-adaptive-composition.md). An engineer who looks at the first results and decides to "try a few more" has changed the procedure, and the theorem no longer applies.

Both numbers in step 5 matter. The released run's $\varepsilon$ says what the model alone would leak if its hyperparameters were public, while the procedure's $\varepsilon$ is what the release guarantees. Ponomareva et al. (2023, Section 5.3.3) ask for both. [Auditing](10-auditing.md) cannot catch the bug, because an audit tests the trained model, not how it was chosen.

::: {#exm-hpt-report}
For the Poisson row of @tbl-hpt-run, a complete report reads: "DP-SGD with Poisson sampling rate 0.01, clipping norm 0.5, noise multiplier 1.53, 1,000 steps; one run satisfies $(0.98, 10^{-5})$-DP. The learning rate was selected by random search over $\{0.0625, 0.125, \dots, 8\}$ with a Poisson number of runs with mean 8, scored by correct predictions on a disjoint validation set of 5,000 records with Gaussian noise of standard deviation 20; the whole procedure satisfies $(1.99, 10^{-5})$-DP by the Rényi bound of Papernot & Steinke (2022, Theorem 6), add/remove neighbours, record level."
:::

## Code {#sec-hpt-code}

The script [code/hyperparameter_tuning.py](../../code/part05-dp-ml/hyperparameter_tuning.py) computes the exact best-of-$K$ distributions of @eq-hpt-pgf, evaluates the bounds on the RDP accountant it imports from [code/dp_sgd.py](../../code/part05-dp-ml/dp_sgd.py), and runs the tuning experiment, about 200 DP-SGD runs, in two to four minutes. From the repository root:

```bash
uv run python code/part05-dp-ml/hyperparameter_tuning.py
```

| Experiment | Result |
| --- | --- |
| Best of 10 randomized responses, $\varepsilon = 1$ | Exactly $\varepsilon = 10$; loss 10 with probability 0.0436 |
| Composition, running example | 27.23 for 10 runs, 145.17 for 100 |
| Geometric stopping, three outcomes | Exact $\varepsilon$ up to 2.9987, below the bound 3 |
| Logarithmic number of runs, running example | 10.08 for 10 runs, 13.46 for 1,000 |
| Exponential mechanism over 8 models | $\Pr[\text{best}] = 0.6484$ at $\varepsilon = 0.1$ |
| Tuning at $\varepsilon = 2$ | Honest search 0.9430 to 0.9433; bug claims 1.97 and spends 5.60 |

: The script's output with the default seed. {#tbl-hpt-code}

[tests/test_hyperparameter_tuning.py](../../tests/test_hyperparameter_tuning.py) checks that the output still matches.

## Summary {#sec-hpt-summary}

| Result | Statement | Use |
| --- | --- | --- |
| @lem-hpt-pgf | $\Pr[M = y] = f(Q(\preceq y)) - f(Q(\prec y))$, with $f(x) = \mathbb E[x^K]$ | Every best-of computation; needs $K$ independent of the runs |
| @prp-hpt-rr-k | The best of $k$ randomized responses is exactly $k\varepsilon$-DP | The best of $k$ runs is no cheaper than $k$ runs unless $k$ is random |
| @prp-hpt-compose | $\varepsilon_M(\alpha) \le k\varepsilon(\alpha)$, valid always, even adaptively | Two or three candidates; report the cost of the search, not of the run |
| @thm-hpt-lt | Random stopping is $3\varepsilon$-DP; it misses the top $\pi$ fraction with probability at most $\omega/\pi$ | Pure-DP runs; $10m$ runs on average to reach the top $1/m$ with probability 0.9 |
| @thm-hpt-generic | $\varepsilon(\alpha) + \frac{1}{\alpha - 1}\ln[f'(s)^\alpha f'(s')^{1 - \alpha}]$, with $s, s'$ from one test | The source of every bound for a random number of runs |
| @thm-hpt-poisson | $\varepsilon(\alpha) + \mathbb E[K]\hat\delta + \frac{\ln\mathbb E[K]}{\alpha - 1}$, if $e^{\hat\varepsilon} \le 1 + \frac{1}{\alpha - 1}$ | Small searches at small per-run budgets |
| @thm-hpt-tnb | One run, plus a multiple of one run at order $\hat\alpha$, plus $\frac{\ln\mathbb E[K]}{\alpha - 1}$; pure DP: $(2 + \nu)\varepsilon$ | Large searches, at the price of often running few candidates |
| @prp-hpt-select | $\max(\sum_j\varepsilon_j(\alpha), \varepsilon_{\text{sel}}(\alpha))$ for a disjoint validation set | Validation scores must be noisy; a few thousand records make that cheap |
| @def-hpt-strategies | Six strategies at a total $\varepsilon = 2$ | Honest search costs about a point; the bug claims 1.97 and spends 5.60 |
| @def-hpt-procedure | Tune off the private data, fix the search, noise the score, count every run, report both numbers | Adaptive searches need adaptive composition |

: The results of this chapter, and where each is used. {#tbl-hpt-summary}

## References {#sec-hpt-references}

- Liu & Talwar, "Private Selection from Private Candidates" (STOC 2019; arXiv:1811.07971). Random stopping and the $3\varepsilon$ bound of @sec-hpt-stopping; threshold and $(\varepsilon, \delta)$ variants; the fixed-$k$ counterexample (their Appendix B).
- Papernot & Steinke, "Hyperparameter Tuning with Renyi Differential Privacy" (ICLR 2022; arXiv:2110.03620). The generic bound, the Poisson and truncated negative binomial theorems of @sec-hpt-rdp, the lower bound of @sec-hpt-leak, and an experiment on image classification that recommends the Poisson distribution for small searches.
- Chaudhuri & Vinterbo, "A Stability-Based Validation Procedure for Differentially Private Machine Learning" (NeurIPS 2013). The first improvement over composition for tuning, for learning algorithms that are stable.
- Mohapatra, Sasy, He, Kamath & Thakkar, "The Role of Adaptive Optimizers for Honest Private Hyperparameter Selection" (arXiv:2111.04906, 2021). Honest tuning with composition, and optimizers that need less tuning.
- McSherry & Talwar, "Mechanism Design via Differential Privacy" (FOCS 2007). The exponential mechanism of @sec-hpt-select.
- Papernot, Thakurta, Song, Chien & Erlingsson, "Tempered Sigmoid Activations for Deep Learning with Differential Privacy" (arXiv:2007.14191, 2020). Architecture choices that are best without privacy are not best with it, so tuning cannot simply be done non-privately.
- Ponomareva et al., "How to DP-fy ML: A Practical Guide to Machine Learning with Differential Privacy" (JAIR 2023). What to tune, how, and what to report.
- Bergstra & Bengio, "Random Search for Hyper-Parameter Optimization" (JMLR 2012). Why random search, which the random-$K$ theorems require, is a sound tuning method in its own right.

**Practice.** Exercises for this chapter, each with a worked solution, are in [the exercise book](../../book-exercises/part05-dp-ml/03-hyperparameter-tuning.md).
