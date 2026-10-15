# Concentrated differential privacy

## Why this matters

[The first chapter of this part](01-renyi-dp.md) describes a mechanism by a whole curve. For two neighbouring datasets $D$ and $D'$, which differ in one person's record, the mechanism's outputs $M(D)$ and $M(D')$ are two probability distributions, and the **Rényi divergence** $D_\alpha$ of order $\alpha > 1$ measures how far apart they are; larger orders put more weight on the outputs where one distribution is much more likely than the other. The **Rényi differential privacy (RDP) curve** $\varepsilon(\alpha)$ is the largest $D_\alpha$ over all neighbouring pairs, one value per order. That is a lot to carry around. For the Gaussian mechanism, which adds normal noise of standard deviation $\sigma$ to a query whose value moves by at most 1 between neighbours, the curve is a straight line through the origin, $\varepsilon(\alpha) = \alpha/(2\sigma^2)$, and a line through the origin is fixed by its slope. So for the Gaussian you could keep one number instead of a curve. The question of this chapter is: **when does one number describe the whole RDP curve, and what does that number buy you?**

The answer is **zero-concentrated differential privacy (zCDP)**, the most widely used form of concentrated differential privacy (CDP). A mechanism is $\rho$-zCDP when its RDP curve lies under the line $\rho\alpha$. Three things follow at once, and each comes from a property of the RDP curve that the line inherits. First, composition is addition. When you run several mechanisms one after another, their RDP curves add, so lines through the origin add to a line whose slope is the sum of the slopes: $k$ steps of $\rho$-zCDP are $k\rho$-zCDP. Second, the privacy loss is "concentrated", which gives the definition its name. The privacy loss is the log of how much more likely an output is under $D$ than under $D'$, and a bound on every Rényi divergence turns out to be a bound on its moment generating function (MGF), the function $s \mapsto \mathbb E[e^{sL}]$ that controls how heavy the tails of a random variable $L$ can be. The line $\rho\alpha$ says that the privacy loss has a tail no heavier than a normal variable with mean $\rho$ and variance $2\rho$. Third, the conversion to the usual $(\varepsilon, \delta)$ guarantee has a closed form, $\varepsilon = \rho + 2\sqrt{\rho\ln(1/\delta)}$, because the optimization over orders that the conversion requires can be done by hand for a line.

### The one idea: a line under the curve

Everything in this chapter is about **how well a line $\rho\alpha$ fits a mechanism's RDP curve**. For the Gaussian the line is the curve, and nothing is lost. A pure differential privacy (DP) mechanism, one whose output probabilities change by at most a factor $e^\varepsilon$ between neighbours, has a curve that bends over and flattens at $\varepsilon$, because no Rényi divergence can exceed the largest log-ratio of probabilities. A line through the origin fits such a curve only near $\alpha = 1$; the fit costs a little there and a lot at large orders. For the subsampled Gaussian of differentially private stochastic gradient descent (SGD), DP-SGD for short, where each step runs on a small random sample of the data, the curve is tiny at small orders and then bends sharply upwards. Any line that stays above it everywhere is as steep as the line of the Gaussian without subsampling, so zCDP throws away all of the privacy that subsampling gained. **Truncated CDP** fixes this by asking for the line only on orders up to a cut-off $\omega$.

@tbl-cdp-overview lists the results and where each is used.

| Result | Where it is used |
| --- | --- |
| Definition of $\rho$-zCDP; Gaussian is $\Delta^2/(2\sigma^2)$-zCDP; composition adds $\rho$ | Accounting for any pipeline of Gaussian mechanisms with one number, as in the 2020 US Census TopDown algorithm |
| $\varepsilon$-DP implies $(\varepsilon^2/2)$-zCDP, sharply $(\varepsilon\tanh(\varepsilon/2))$-zCDP; exponential mechanism is $(\varepsilon^2/8)$-zCDP | Mixing Laplace, randomized response and selection steps with Gaussian steps in one budget |
| $\rho$-zCDP to $(\varepsilon, \delta)$: simple and improved conversions | Reporting the final guarantee; calibrating $\sigma$ to a target $\varepsilon$ |
| Group privacy $k^2\rho$ | Protecting households or users with several records; lower bounds |
| Approximate zCDP | Pipelines that include mechanisms that are only $(\varepsilon, \delta)$-DP, such as stability-based ones |
| No amplification by subsampling; truncated CDP | Why DP-SGD is accounted in RDP, not zCDP; a one-number summary that does amplify |

: The results of this chapter and where each is used. {#tbl-cdp-overview}

**How to read this chapter.** The definition and the conversion sections are the core; read them first. The pure-DP section shows how other mechanisms enter a zCDP budget. The subsampling section explains why the rest of this part works with RDP curves and privacy loss distributions instead of a single $\rho$.

## Toolkit: four facts used below

Each fact is stated here once, with the short reason it holds and a link to where it is derived in full. Throughout, $P = M(D)$ and $Q = M(D')$ are the output distributions on two neighbouring datasets, and $P(y)$, $Q(y)$ are their probabilities (or densities) at an output $y$.

### Fact 1 — Rényi divergence is a log-MGF of the privacy loss

The **privacy loss** at an output $y$ is $\ln(P(y)/Q(y))$: how much more likely $y$ is when the data are $D$ than when they are $D'$, on a log scale. Drawing the output from $P$ makes it a random variable, the **privacy loss random variable** $L = L_{D,D'}(Y)$ with $Y \sim P$. The Rényi divergence of order $\alpha$ is defined as $D_\alpha(P\|Q) = \frac{1}{\alpha - 1}\ln\sum_y P(y)^\alpha Q(y)^{1-\alpha}$. Write $P(y)^\alpha Q(y)^{1-\alpha} = P(y)\,(P(y)/Q(y))^{\alpha - 1} = P(y)\,e^{(\alpha - 1)L(y)}$: the sum is then an average over $Y \sim P$ of $e^{(\alpha - 1)L}$. So the Rényi divergence, times $\alpha - 1$, is the log of the MGF of the privacy loss at the point $\alpha - 1$:
$$(\alpha - 1)D_\alpha(P\|Q) = \ln\mathbb E_P\big[e^{(\alpha - 1)L}\big].$$ {#eq-cdp-psi}
This is @eq-renyi-psi in [Divergences](../part01-prerequisites/03-divergences.md), with the log-likelihood ratio renamed the privacy loss.

### Fact 2 — RDP composes by adding curves, and survives post-processing

If $M_1$ has RDP curve $\varepsilon_1(\alpha)$ and $M_2$, possibly chosen after seeing the output of $M_1$, has curve $\varepsilon_2(\alpha)$, the pair has curve at most $\varepsilon_1(\alpha) + \varepsilon_2(\alpha)$. The reason is that the sum $\sum P^\alpha Q^{1-\alpha}$ over pairs of outputs $(y_1, y_2)$ factors: for each first output $y_1$, the inner sum over $y_2$ is at most $e^{(\alpha - 1)\varepsilon_2(\alpha)}$, whatever $M_2$ was chosen, and the remaining sum over $y_1$ is at most $e^{(\alpha - 1)\varepsilon_1(\alpha)}$. Exponents multiply, so the divergences add. Also, any function of the output has curve at most $\varepsilon(\alpha)$: this is the data-processing inequality, which says that passing two distributions through the same random map can only make them harder to tell apart. Both are derived in [Rényi DP](01-renyi-dp.md), from additivity and the data-processing inequality for $D_\alpha$.

### Fact 3 — Mironov's conversion from RDP to $(\varepsilon, \delta)$

A mechanism is $(\varepsilon, \delta)$-DP when, for every set $S$ of outputs, $P(S) \le e^\varepsilon Q(S) + \delta$. One way to get there is to show that the privacy loss exceeds $\varepsilon$ with probability at most $\delta$: on outputs where $L \le \varepsilon$ you have $P(y) \le e^\varepsilon Q(y)$, and the rest of $S$ has $P$-probability at most $\delta$. Markov's inequality applied to $e^{(\alpha - 1)L}$ bounds that probability, $\Pr[L > \varepsilon] \le \mathbb E[e^{(\alpha - 1)L}]e^{-(\alpha - 1)\varepsilon} \le e^{(\alpha - 1)(\varepsilon(\alpha) - \varepsilon)}$ by Fact 1, and setting the right side equal to $\delta$ and solving for $\varepsilon$ gives the conversion. A mechanism with $\varepsilon(\alpha)$ at order $\alpha$ is $(\varepsilon(\alpha) + \ln(1/\delta)/(\alpha - 1), \delta)$-DP, for every $\delta \in (0, 1)$. Every order gives a valid guarantee, and you pick the order that makes $\varepsilon$ smallest. An improved conversion, which avoids the slack of Markov's inequality, subtracts $\ln(\alpha)/(\alpha - 1) - \ln(1 - 1/\alpha)$ from this. Both are derived in [From Rényi DP to (ε, δ) and to tests](02-rdp-conversions.md).

### Fact 4 — Hoeffding's lemma

A random variable confined to an interval cannot have heavy tails, so its MGF is no larger than that of a normal variable with the same mean and a variance set by the interval's length. Precisely: if $X \in [a, b]$, then $\ln\mathbb E[e^{\lambda X}] \le \lambda\mathbb E[X] + \lambda^2(b - a)^2/8$ for every real $\lambda$. (A normal variable with variance $v$ has $\ln\mathbb E[e^{\lambda X}] = \lambda\mathbb E[X] + \lambda^2 v/2$, so this is the normal bound with $v = (b - a)^2/4$.) Derived in [Concentration](../part01-prerequisites/02-concentration.md).

## The definition of zCDP {#sec-cdp-definition}

### Context

You add Gaussian noise to a gradient, a count or a mean, again and again, and you want one number that says how much privacy you have spent so far. The RDP curve of the Gaussian is a line through the origin, and lines add. Bun and Steinke turned this observation into a definition: describe a mechanism by the slope of a line that lies above its RDP curve. The definition also has a reading in terms of the privacy loss itself, and that reading gives the name.

### Statement

A mechanism $M$ is **$\rho$-zero-concentrated differentially private** ($\rho$-zCDP) if its RDP curve lies below the line of slope $\rho$: for all neighbouring $D \simeq D'$ and all orders $\alpha > 1$,
$$D_\alpha\big(M(D)\,\|\,M(D')\big) \le \rho\alpha.$$ {#eq-cdp-def}
Multiplying both sides by $\alpha - 1$ and using Fact 1 turns this into a statement about the privacy loss random variable $L$: its MGF at every point $\alpha - 1 > 0$ is bounded,
$$\ln\mathbb E\big[e^{(\alpha - 1)L}\big] \le (\alpha - 1)\alpha\rho \quad\text{for all } \alpha > 1.$$ {#eq-cdp-mgf}
A bounded MGF limits how often $L$ can be large, so the privacy loss has a Gaussian-type upper tail: the chance that it exceeds its mean $\rho$ by more than $\lambda$ falls off like a normal tail, as $e^{-\lambda^2/(4\rho)}$,
$$\Pr[L > \rho + \lambda] \le e^{-\lambda^2/(4\rho)} \quad\text{for all } \lambda > 0.$$ {#eq-cdp-tail}

Three properties make the definition useful. The Gaussian mechanism, which adds $\mathcal N(0, \sigma^2)$ to a query with $\ell_2$ sensitivity $\Delta = \Delta_2 f$ (the largest Euclidean distance between the query's values on neighbouring datasets), is $\Delta^2/(2\sigma^2)$-zCDP. With the part's noise multiplier convention, which measures $\sigma$ in units of the sensitivity ($\Delta = 1$), $\rho = 1/(2\sigma^2)$. Composition adds the parameters: if $M_1$ is $\rho_1$-zCDP and $M_2$ is $\rho_2$-zCDP, possibly chosen adaptively, the pair is $(\rho_1 + \rho_2)$-zCDP. And post-processing is free: any function of a $\rho$-zCDP output is $\rho$-zCDP.

### Derivation

The first three steps show that the three displays say the same thing at increasing strength; the last two prove the three properties.

1. **The two forms agree.** Multiply @eq-cdp-def by $\alpha - 1$, which is positive, so the inequality keeps its direction. By Fact 1 the left side becomes $\ln\mathbb E[e^{(\alpha-1)L}]$, and the right side is $(\alpha - 1)\alpha\rho$.
2. **The tail bound is a Chernoff bound.** To bound the chance that $L$ is large, apply Markov's inequality to the positive variable $e^{(\alpha-1)L}$, which is large exactly when $L$ is. For any threshold $t$, $\Pr[L > t] = \Pr[e^{(\alpha-1)L} > e^{(\alpha-1)t}] \le \mathbb E[e^{(\alpha-1)L}]\,e^{-(\alpha-1)t}$. By @eq-cdp-mgf this is at most $\exp\big((\alpha - 1)(\alpha\rho - t)\big)$.
3. **Optimize the order.** Step 2 holds for every order, so use the one that gives the smallest bound. Put $t = \rho + \lambda$ and write $s = \alpha - 1$. The exponent is $s(\rho + s\rho - \rho - \lambda) = s^2\rho - s\lambda$, a parabola in $s$ that is smallest at $s = \lambda/(2\rho)$, where it equals $\lambda^2/(4\rho) - \lambda^2/(2\rho) = -\lambda^2/(4\rho)$. This is @eq-cdp-tail.
4. **Gaussian mechanism.** The output distributions are $\mathcal N(f(D), \sigma^2)$ and $\mathcal N(f(D'), \sigma^2)$ with $|f(D) - f(D')| \le \Delta$. Two normal distributions with the same variance $\sigma^2$ and means $s$ apart have Rényi divergence $D_\alpha = \alpha s^2/(2\sigma^2)$ (@eq-gauss-renyi in [Divergences](../part01-prerequisites/03-divergences.md); the sum in Fact 1 becomes a Gaussian integral that you evaluate by completing the square). So $D_\alpha = \alpha(f(D) - f(D'))^2/(2\sigma^2) \le \alpha\Delta^2/(2\sigma^2)$. The same holds in $d$ dimensions with $\ell_2$ sensitivity, because the Rényi divergence of two isotropic Gaussians depends only on the distance between their means: rotate the coordinates so the means differ along one axis, and the other axes contribute nothing.
5. **Composition and post-processing.** By Fact 2, the composed curve is at most $\rho_1\alpha + \rho_2\alpha = (\rho_1 + \rho_2)\alpha$, a line of slope $\rho_1 + \rho_2$, and post-processing does not raise the curve.

**Why "concentrated": the Gaussian case by hand.** The tail bound says the privacy loss behaves no worse than a normal variable. For the Gaussian mechanism it *is* a normal variable, and the bound holds with equality in every exponential moment. To see this, take $\Delta = 1$, $P = \mathcal N(0, \sigma^2)$, $Q = \mathcal N(1, \sigma^2)$. The privacy loss at output $y$ is the log of the ratio of the two normal densities; the normalizing constants cancel and only the exponents remain:
$$L(y) = \frac{(y - 1)^2 - y^2}{2\sigma^2} = \frac{1 - 2y}{2\sigma^2}.$$
With $Y \sim \mathcal N(0, \sigma^2)$, $L$ is a linear function of a normal variable, so it is normal. Its mean is the constant term, $1/(2\sigma^2) = \rho$, and its variance is the square of the coefficient of $y$ times $\sigma^2$, that is $4\sigma^2/(4\sigma^4) = 1/\sigma^2 = 2\rho$. A normal variable with mean $m$ and variance $v$ has MGF $e^{sm + s^2v/2}$, so the MGF of $\mathcal N(\rho, 2\rho)$ at $s$ is $e^{s\rho + s^2\rho}$; at $s = \alpha - 1$ it is $e^{(\alpha - 1)(1 + \alpha - 1)\rho} = e^{(\alpha - 1)\alpha\rho}$. So @eq-cdp-mgf holds **with equality** for every $\alpha$. A $\rho$-zCDP mechanism is one whose privacy loss is, in every exponential moment, no worse than a normal with mean $\rho$ and variance $2\rho$.

**Where the definition came from.** Dwork and Rothblum (2016) introduced **concentrated differential privacy** with two parameters $(\mu, \tau)$. The mean of the privacy loss is at most $\mu$, and the privacy loss minus its mean is **sub-Gaussian** with parameter $\tau$, which means its MGF is no larger than that of a centred normal with variance $\tau^2$: $\mathbb E[e^{\lambda(L - \mathbb E L)}] \le e^{\lambda^2\tau^2/2}$ for every real $\lambda$. Bun and Steinke (2016) call this mean-concentrated DP and replaced it with @eq-cdp-def, for two reasons. First, the Rényi form follows from properties of $D_\alpha$ (data processing, additivity), so post-processing and composition are immediate; their Appendix A shows that the mean-concentrated definition is *not* closed under post-processing. Second, zCDP is a relaxation of mean-concentrated DP, so nothing that was mean-concentrated is lost. To see this, write $L = \mathbb E L + (L - \mathbb E L)$ and apply the sub-Gaussian bound with $\lambda = \alpha - 1$: this gives $\ln\mathbb E[e^{(\alpha-1)L}] \le (\alpha - 1)\mu + (\alpha - 1)^2\tau^2/2$. Dividing by $\alpha - 1$ and using Fact 1 gives $D_\alpha \le \mu + (\alpha - 1)\tau^2/2 = (\mu - \tau^2/2) + (\tau^2/2)\alpha$, with equality for the Gaussian mechanism (their Lemma 4.2), where $\mu = \rho$ and $\tau^2 = 2\rho$ make the constant term vanish. In general that is a line with an intercept; Bun and Steinke allow one in a two-parameter $(\xi, \rho)$-zCDP, but mostly use $\xi = 0$, the one-parameter $\rho$-zCDP that this book follows. (This $\mu$ is unrelated to the $\mu$ of Gaussian differential privacy later in this part.)

### Interpretation

**zCDP is "RDP restricted to lines".** Every $\rho$-zCDP mechanism has an RDP curve, and the curve is at most $\rho\alpha$. Going from a curve to $\rho$ loses information: the smallest valid $\rho$ is the largest ratio $\varepsilon(\alpha)/\alpha$, wherever it occurs, and the rest of the curve is forgotten. The rest of the chapter is about how much that costs.

**The tail bound is loose by a polynomial factor.** @fig-cdp-privacy-loss shows, for $\sigma = 1$ ($\rho = 0.5$), the simulated privacy loss against $\mathcal N(\rho, 2\rho)$ and the exact tail against @eq-cdp-tail. Since the loss is exactly $\mathcal N(\rho, 2\rho)$, the event $L > \rho + \lambda$ is the event that a standard normal exceeds $z = \lambda/\sqrt{2\rho}$, so the exact tail is $\Phi(-\lambda/\sqrt{2\rho})$, where $\Phi$ is the standard normal cumulative distribution function. Both tails fall off like $e^{-\lambda^2/(4\rho)} = e^{-z^2/2}$; the bound misses only the Gaussian tail's polynomial factor. For large $z$ the Mills ratio approximation $\Phi(-z) \approx e^{-z^2/2}/(z\sqrt{2\pi})$ says the exact tail is the bound divided by $z\sqrt{2\pi} = \lambda\sqrt{\pi/\rho}$, so the ratio of the exact tail to the bound is about $\sqrt{\rho/\pi}/\lambda$. At $\lambda = 4$, $z = 4$, the exact tail is $\Phi(-4) = 3.167 \times 10^{-5}$ and the bound $e^{-16/2} = 3.355 \times 10^{-4}$: their ratio is $0.094$, against $\sqrt{0.5/\pi}/4 = 0.100$ from the Mills ratio. That gap is one reason the simple conversion of the conversion section (@sec-cdp-conversion) is loose.

![The privacy loss of the Gaussian mechanism with $\sigma = 1$. Left: a histogram of $L(Y)$ from simulation against the normal density with mean $\rho = 0.5$ and variance $2\rho = 1$. Right: the exact upper tail $\Pr[L > \rho + \lambda]$ against the zCDP bound $e^{-\lambda^2/(4\rho)}$, on a log scale.](figures/concentrated_dp_privacy_loss.svg){#fig-cdp-privacy-loss}

**Where zCDP is used.** zCDP is the accounting of choice when a system is built from Gaussian mechanisms on exact, non-subsampled data: statistical releases, query systems and synthetic data from noisy marginals. It is not the accounting of choice for DP-SGD, for the reason of @sec-cdp-tcdp. An example is the 2020 US Census TopDown algorithm, which protects the redistricting data (Abowd et al., 2022). It measures noisy versions of a set of queries ("measurements") under zCDP, using the discrete Gaussian mechanism, which adds integer-valued noise shaped like a normal density so that counts stay integers. It then post-processes the measurements together with **invariants**, statistics that the Census Bureau excluded from the privacy accounting as a matter of policy, into a microdata file. The discrete Gaussian and its zCDP guarantee are from Canonne, Kamath and Steinke (2020), the same paper as the improved conversion; [Discrete mechanisms](../part03-dp-foundations/07-discrete-mechanisms.md) covers it. zCDP suits this setting for three reasons. The budget is split over many queries and geographic levels, and zCDP makes the split a sum of $\rho$'s. The Gaussian's RDP curve is a line, so zCDP loses nothing before the final conversion. And the final conversion to $(\varepsilon, \delta)$ can be done once, at whatever $\delta$ the publisher chooses to report. The invariants are a reminder that the accounted $\rho$ covers only the noisy measurements: anything released exactly, without noise, sits outside the guarantee.

### Example: the Gaussian with $\sigma = 1$

With $\rho = 1/(2\sigma^2)$, one Gaussian step with $\sigma = 1$ is $0.5$-zCDP; ten of them are $10 \times 0.5 = 5$-zCDP; one step with $\sigma = 2$ is $1/8 = 0.125$-zCDP. A million simulated privacy losses (seed 0) have mean $0.4990$ and variance $1.0013$, against $\rho = 0.5$ and $2\rho = 1$, and $\ln\mathbb E[e^L]$ is $1.0000$, against $(\alpha - 1)\alpha\rho = 1$ at $\alpha = 2$. @tbl-cdp-tail compares the tail with its bound. With $\rho = 0.5$ the exact column is $\Phi(-\lambda)$ and the bound is $e^{-\lambda^2/2}$: at $\lambda = 1$, $\Phi(-1) = 0.1587$ against $e^{-1/2} = 0.6065$.

| $\lambda$ | Exact $\Pr[L > \rho + \lambda]$ | Simulated | zCDP bound $e^{-\lambda^2/(4\rho)}$ |
| --- | --- | --- | --- |
| 1 | $1.587 \times 10^{-1}$ | $1.586 \times 10^{-1}$ | $6.065 \times 10^{-1}$ |
| 2 | $2.275 \times 10^{-2}$ | $2.261 \times 10^{-2}$ | $1.353 \times 10^{-1}$ |
| 4 | $3.167 \times 10^{-5}$ | $2.700 \times 10^{-5}$ | $3.355 \times 10^{-4}$ |

: Upper tail of the Gaussian mechanism's privacy loss, $\sigma = 1$, $\rho = 0.5$: exact, simulated with $10^6$ draws, and the zCDP bound @eq-cdp-tail. {#tbl-cdp-tail}

### Check yourself

- A mechanism adds $\mathcal N(0, 4)$ noise to a count with sensitivity 1, and then a second mechanism adds $\mathcal N(0, 1)$ noise to another count. What is the total $\rho$? *($1/8 + 1/2 = 0.625$. The first has variance 4, so $\sigma = 2$ and $\rho = 1/(2 \cdot 4)$; the second has $\sigma = 1$ and $\rho = 1/2$; composition adds them.)*
- Does $\rho$-zCDP bound the lower tail of the privacy loss, $\Pr[L < -t]$? *(Yes, through the other order of the pair. @eq-cdp-mgf only uses positive exponents, so it says nothing directly about large negative values. But the definition also holds with $D$ and $D'$ swapped, and a change of measure connects the two: for $s > 1$, $\mathbb E_P[e^{-sL}] = \sum_y P(y)(Q(y)/P(y))^s = \sum_y P(y)^{1-s}Q(y)^s = e^{(s-1)D_s(Q\|P)} \le e^{(s-1)s\rho}$. Chernoff, as in steps 2 and 3, applied to $-L$ then gives $\Pr[L < -t] \le e^{-(t+\rho)^2/(4\rho)}$ for $t \ge \rho$. A weaker bound holds for every pair, zCDP or not: $\Pr_P[L < -t] \le e^{-t}$, because $P(y) < e^{-t}Q(y)$ on that event, and the $Q$-probabilities sum to at most 1.)*

## Pure DP implies zCDP {#sec-cdp-pure}

### Context

A real pipeline mixes mechanisms. A Laplace count, a randomized-response survey bit and an exponential-mechanism selection are $\varepsilon$-DP, not Gaussian: for neighbouring datasets, the probability of any output changes by at most a factor $e^\varepsilon$. To put them in the same budget as Gaussian steps, you need their $\rho$. A pure $\varepsilon$-DP mechanism has a bounded privacy loss, $|L| \le \varepsilon$, because the log of a ratio between $e^{-\varepsilon}$ and $e^\varepsilon$ lies between $-\varepsilon$ and $\varepsilon$. A bounded variable has light, Gaussian-like tails (Hoeffding's lemma, Fact 4). So it should be zCDP with $\rho$ of order $\varepsilon^2$. The question is the constant.

### Statement

If $M$ is $\varepsilon$-DP, then $M$ is $(\varepsilon^2/2)$-zCDP (Bun & Steinke 2016, Proposition 1.4). The RDP curve of an $\varepsilon$-DP mechanism is also capped at $\varepsilon$, because every Rényi divergence is at most the max-divergence $D_\infty$, the largest log-ratio $\ln(P(y)/Q(y))$, which is at most $\varepsilon$: $D_\alpha \le D_\infty \le \varepsilon$. Together, its RDP curve lies under both the line and the cap:
$$\varepsilon(\alpha) \le \min\left(\varepsilon,\ \frac{\alpha\varepsilon^2}{2}\right).$$ {#eq-cdp-pure}
The constant can be sharpened. The worst $\varepsilon$-DP mechanism turns out to be randomized response, which reports a bit truthfully with probability $e^\varepsilon/(1 + e^\varepsilon)$ and flips it otherwise, and its slope gives the sharp constant. Every $\varepsilon$-DP mechanism is $(\varepsilon\tanh(\varepsilon/2))$-zCDP,
$$\varepsilon(\alpha) \le \alpha\,\varepsilon\tanh(\varepsilon/2),$$ {#eq-cdp-pure-sharp}
and randomized response shows that no smaller $\rho$ holds for every $\varepsilon$-DP mechanism. Since $\tanh x \le x$ for $x \ge 0$, $\varepsilon\tanh(\varepsilon/2) \le \varepsilon \cdot \varepsilon/2 = \varepsilon^2/2$, with ratio tending to 1 as $\varepsilon \to 0$, where $\tanh x \approx x$.

For the exponential mechanism, which selects one candidate from a list and favours candidates with higher scores, the constant is four times smaller: an $\varepsilon$-DP exponential mechanism is $(\varepsilon^2/8)$-zCDP (Cesar & Rogers 2021, the remark after Lemma 3.4). The reason is the bounded-range property of Durfee & Rogers (2019): its privacy loss lies in an interval of length $\varepsilon$, not $2\varepsilon$.

### Derivation

Fix neighbours, write $P = M(D)$, $Q = M(D')$, and $L = \ln(P(y)/Q(y))$ under $P$, where $P(y)$ and $Q(y)$ are the densities (capital letters, because $q$ is the sampling rate later in this chapter).

**The constant $\tfrac12$, by Hoeffding.** The plan is to bound the MGF of $L$ with Hoeffding's lemma. Hoeffding needs the mean of $L$, so the first job is to bound that mean, and Hoeffding does that too.

1. **The privacy loss is bounded.** $\varepsilon$-DP for both orders of the pair means $e^{-\varepsilon} \le P(y)/Q(y) \le e^\varepsilon$ wherever either density is positive. So $L \in [-\varepsilon, \varepsilon]$, and $P$, $Q$ have the same support: neither can be positive where the other is zero.
2. **The mean is at most $\varepsilon^2/2$.** Apply Hoeffding's lemma (Fact 4) to $X = -L$, which lies in an interval of length $2\varepsilon$, with $\lambda = 1$: $\ln\mathbb E_P[e^{-L}] \le -\mathbb E_P[L] + (2\varepsilon)^2/8$. The left side is known exactly. $e^{-L} = Q(Y)/P(Y)$, and its average under $P$ is $\sum_y P(y)\,Q(y)/P(y) = \sum_y Q(y) = 1$, because the supports agree (Fact 1 of [Divergences](../part01-prerequisites/03-divergences.md)). So $0 \le -\mathbb E_P[L] + \varepsilon^2/2$: the Kullback–Leibler (KL) divergence $\mathbb E_P[L]$, the average privacy loss, is at most $\varepsilon^2/2$.
3. **The MGF is sub-Gaussian.** Hoeffding's lemma again, now for $L$ itself with $\lambda = \alpha - 1$: $\ln\mathbb E_P[e^{(\alpha-1)L}] \le (\alpha - 1)\mathbb E_P[L] + (\alpha - 1)^2(2\varepsilon)^2/8 = (\alpha - 1)\mathbb E_P[L] + (\alpha - 1)^2\varepsilon^2/2$.
4. **Combine.** Put the mean bound of step 2 into step 3: the right side is at most $(\alpha - 1)\varepsilon^2/2 + (\alpha - 1)^2\varepsilon^2/2 = (\alpha - 1)(1 + \alpha - 1)\,\varepsilon^2/2 = (\alpha - 1)\alpha\,\varepsilon^2/2$. That is @eq-cdp-mgf with $\rho = \varepsilon^2/2$.
5. **The cap.** $D_\alpha$ is non-decreasing in $\alpha$ with limit $D_\infty \le \varepsilon$ ([Divergences](../part01-prerequisites/03-divergences.md)). Larger orders weight the largest likelihood ratios more, and in the limit only the largest one, at most $e^\varepsilon$, counts. That gives the other term of the minimum.

**The sharp constant, through randomized response.** Hoeffding treats $L$ as an arbitrary bounded variable, which wastes something. The sharper route shows that randomized response is the worst case, then computes its MGF exactly.

6. **Reduce to randomized response.** Every $\varepsilon$-DP pair is a post-processing of $\varepsilon$-randomized response. That is, if $P^\star$ and $Q^\star$ are the output distributions of randomized response on the bits 0 and 1, there is a single random map $K$ from $\{0, 1\}$ to outputs with $P = P^\star K$ and $Q = Q^\star K$. [Composition](../part03-dp-foundations/04-composition.md) (@sec-comp-optimal) gives the channel $K$ explicitly (@eq-comp-channel). By the data-processing inequality for $D_\alpha$ (Fact 2), passing both distributions through $K$ cannot increase their divergence, so $D_\alpha(P\|Q) \le D_\alpha(P^\star\|Q^\star)$ at every order. It is enough to bound the curve of randomized response.
7. **The MGF of randomized response.** Under $P^\star$ the truthful answer has probability $t = e^\varepsilon/(1 + e^\varepsilon)$, and its likelihood ratio is $t/(1 - t) = e^\varepsilon$. So the privacy loss is $+\varepsilon$ with probability $t$ and $-\varepsilon$ otherwise. With $s = \alpha - 1$, the MGF is $t\,e^{s\varepsilon} + (1 - t)\,e^{-s\varepsilon}$, which is
$$\mathbb E[e^{sL}] = \frac{e^{(s+1)\varepsilon} + e^{-s\varepsilon}}{1 + e^\varepsilon} = \frac{\cosh\big((s + \tfrac12)\varepsilon\big)}{\cosh(\varepsilon/2)}.$$
For the second equality, factor $e^{\varepsilon/2}$ out of the numerator and the denominator: the numerator is $e^{\varepsilon/2}\big(e^{(s+\frac12)\varepsilon} + e^{-(s+\frac12)\varepsilon}\big) = 2e^{\varepsilon/2}\cosh\big((s + \tfrac12)\varepsilon\big)$ and the denominator is $e^{\varepsilon/2}(e^{-\varepsilon/2} + e^{\varepsilon/2}) = 2e^{\varepsilon/2}\cosh(\varepsilon/2)$.
8. **A concavity bound.** You want the log of step 7 to be at most $\rho\,s(s + 1)$. Write $x = (s + \frac12)\varepsilon$ and $a = \varepsilon/2$. Then $x^2 - a^2 = \varepsilon^2\big((s + \tfrac12)^2 - \tfrac14\big) = \varepsilon^2(s^2 + s)$ is exactly a multiple of $s(s + 1)$, so a bound on $\ln\cosh x - \ln\cosh a$ that is linear in $x^2 - a^2$ gives the right shape. That suggests looking at $\ln\cosh$ as a function of $x^2$. Let $h(u) = \ln\cosh\sqrt u$ for $u > 0$. By the chain rule its derivative is $h'(u) = \tanh(\sqrt u)/(2\sqrt u)$, which is decreasing, because $\tanh(y)/y$ is decreasing for $y > 0$ (its derivative has the sign of $y - \sinh y\cosh y < 0$). A function with decreasing derivative is concave, and a concave function lies below its tangent line. The tangent at $u_0 = a^2$ has slope $h'(a^2) = \tanh(a)/(2a)$, so
$$\ln\cosh x - \ln\cosh a \le (x^2 - a^2)\frac{\tanh a}{2a}.$$
With $x^2 - a^2 = \varepsilon^2(s^2 + s)$ and $\tanh(a)/(2a) = \tanh(\varepsilon/2)/\varepsilon$, the right side is $s(s + 1)\,\varepsilon\tanh(\varepsilon/2) = (\alpha - 1)\alpha\,\varepsilon\tanh(\varepsilon/2)$. With step 7 that is @eq-cdp-mgf with $\rho = \varepsilon\tanh(\varepsilon/2)$, and step 6 carries it to every $\varepsilon$-DP pair: @eq-cdp-pure-sharp.
9. **No smaller $\rho$.** As $\alpha \to 1$, any RDP curve tends to the KL divergence, the mean privacy loss. For randomized response that mean is $\varepsilon t - \varepsilon(1 - t) = \varepsilon(2t - 1) = \varepsilon\frac{e^\varepsilon - 1}{e^\varepsilon + 1} = \varepsilon\tanh(\varepsilon/2)$. So $\varepsilon(\alpha)/\alpha$ tends to $\varepsilon\tanh(\varepsilon/2)$, and any smaller slope fails near $\alpha = 1$.

So the naive Hoeffding route already gives the constant $\tfrac12$; the only care needed is to bound the mean with Hoeffding too, through $\mathbb E_P[e^{-L}] = 1$. Bun and Steinke prove the same result differently: they replace the likelihood ratio by a two-point variable on $\{e^{-\varepsilon}, e^\varepsilon\}$ with the same mean (Jensen's inequality, which says that for a convex function the average of the function is at least the function of the average, so spreading the ratio to the endpoints can only increase the moments), compute its moments in closed form with $\sinh$, and finish with an inequality between hyperbolic sines. Earlier, Dwork and Rothblum bounded the mean by the slightly weaker $\tfrac12\varepsilon(e^\varepsilon - 1)$. You might hope to get the sharp constant just by putting the sharp mean bound into step 4. That bound, @eq-comp-tanh-mean, says the KL divergence of any $\varepsilon$-DP pair is at most $\varepsilon\tanh(\varepsilon/2)$, by the same data-processing argument as step 6. It is not enough on its own: it gives $\varepsilon(\alpha) \le \varepsilon\tanh(\varepsilon/2) + (\alpha - 1)\varepsilon^2/2$, a line with the same slope $\varepsilon^2/2$ that starts a little lower, and not of the form $\rho\alpha$. The slope comes from Hoeffding's variance term, which is what steps 6 to 8 avoid: they bound every exponential moment at once.

**The exponential mechanism.** Its privacy loss lies in an interval $[c - \varepsilon, c]$ of length $\varepsilon$, not $2\varepsilon$, for some $c$ that depends on the pair (the bounded-range property). Steps 2 to 4 go through with the shorter interval, and the constant $(b - a)^2/8$ in Hoeffding's lemma drops from $(2\varepsilon)^2/8$ to $\varepsilon^2/8$. Hoeffding with $\lambda = 1$ and $\mathbb E_P[e^{-L}] = 1$ give $0 \le -\mathbb E_P[L] + \varepsilon^2/8$, so $\mathbb E_P[L] \le \varepsilon^2/8$. (The same identity forces $c - \varepsilon \le 0 \le c$: if $L$ were always positive, $e^{-L}$ would always be below 1 and could not average to 1, and likewise if it were always negative. So the interval contains 0, the supports agree, and the mechanism is also $\varepsilon$-DP.) Step 3 with length $\varepsilon$ gives $(\alpha - 1)^2\varepsilon^2/8$, and the sum is $(\alpha - 1)\varepsilon^2/8 + (\alpha - 1)^2\varepsilon^2/8 = (\alpha - 1)\alpha\,\varepsilon^2/8$: the mechanism is $(\varepsilon^2/8)$-zCDP. Cesar and Rogers bound the mean with a result of Dong, Durfee & Rogers (2020) instead. The bound is nearly tight. The script computes $D_\alpha/(\alpha\varepsilon^2/8)$, the fraction of the bound actually used, for the two-outcome bounded-range mechanisms of Cesar & Rogers's Definition 2.4, over $c \in (0, \varepsilon)$, $\alpha \in (1, 10^2]$ and both directions. The largest value is $0.9965$ at $\varepsilon = 0.5$, $0.9864$ at $\varepsilon = 1$ and $0.8348$ at $\varepsilon = 4$, each at the smallest order checked: the worst case is the mean, as $\alpha \to 1$.

### Interpretation

**One curve covers every $\varepsilon$-DP mechanism.** Step 6 says more than @eq-cdp-pure-sharp. Since every $\varepsilon$-DP pair is a post-processing of randomized response, the exact RDP curve of $\varepsilon$-randomized response is a valid RDP curve for *every* $\varepsilon$-DP mechanism, at every order. An RDP accountant can charge each pure-DP step that curve, which is below both the cap $\varepsilon$ and the line $\alpha\varepsilon\tanh(\varepsilon/2)$, instead of $\min(\varepsilon, \alpha\varepsilon^2/2)$. Randomized response is the worst $\varepsilon$-DP mechanism at every order.

**$\varepsilon^2/2$ is sharp only as $\varepsilon \to 0$.** For randomized response the smallest valid $\rho$ is $\varepsilon\tanh(\varepsilon/2)$ (steps 8 and 9): $0.1225$ against $\varepsilon^2/2 = 0.1250$ at $\varepsilon = 0.5$, and $0.4621$ against $0.5000$ at $\varepsilon = 1$. The script confirms it numerically: the largest $\varepsilon(\alpha)/\alpha$ on a fine grid of orders is the KL divergence. Since $\tanh(x) \approx x$ for small $x$, the ratio tends to 1. Mechanisms other than randomized response can need less: the Laplace mechanism needs $0.3679$ at $\varepsilon = 1$ ([Exercise 1](solutions/03-concentrated-dp.md#sec-sol-cdp-1)).

**The line is a poor fit far from $\alpha = 1$.** @fig-cdp-rr-line draws the exact curve of randomized response, both zCDP lines and the cap. At $\varepsilon = 1$ the line $\alpha\varepsilon^2/2$ is close near $\alpha = 1$ but crosses the cap where $\alpha\varepsilon^2/2 = \varepsilon$, that is at $\alpha = 2/\varepsilon = 2$. Beyond that, the pure-DP cap $\varepsilon$ is the better bound, and the true curve is below both. At $\varepsilon = 3$ the line $\alpha\varepsilon^2/2$ starts above the cap: its slope $\varepsilon^2/2 = 4.5$ already exceeds $\varepsilon = 3$ at $\alpha = 1$, so it is worse than $\varepsilon$ at every order. The sharp line starts at the KL divergence, below the cap, but crosses it almost at once. An RDP accountant keeps the exact curve, or $\min(\varepsilon, \alpha\varepsilon^2/2)$; a zCDP accountant can keep only a line. For mechanisms with large $\varepsilon$, as in local DP, where each user randomizes their own data, describe them by their RDP curve or their $\varepsilon$, not by $\rho$.

![The RDP curve of randomized response (black) against the zCDP line $\alpha\varepsilon^2/2$ (dashed), the sharp zCDP line $\alpha\varepsilon\tanh(\varepsilon/2)$ (dash-dotted) and the pure-DP cap $\varepsilon$ (dotted), for $\varepsilon = 1$ (left) and $\varepsilon = 3$ (right). The grey line is the KL divergence $\varepsilon\tanh(\varepsilon/2)$, the curve's value at $\alpha \to 1$.](figures/concentrated_dp_rr_line.svg){#fig-cdp-rr-line}

### Example: randomized response at $\varepsilon = 1$

The truthful-report probability is $e/(1 + e) = 0.7311$. @tbl-cdp-rr compares the exact curve with the bounds. The second column is the Hoeffding bound with the sharp mean, $\varepsilon\tanh(\varepsilon/2) + (\alpha - 1)\varepsilon^2/2$ with $\varepsilon\tanh(1/2) = 0.4621$; it sits $0.5 - 0.4621 = 0.0379$ below the zCDP line at every order, because both have slope $\varepsilon^2/2 = 0.5$.

| $\alpha$ | Exact $\varepsilon(\alpha)$ | $\varepsilon\tanh(\varepsilon/2) + (\alpha-1)\varepsilon^2/2$ | $\alpha\varepsilon^2/2$ | $\min(\varepsilon, \alpha\varepsilon^2/2)$ |
| --- | --- | --- | --- | --- |
| 1.5 | 0.6273 | 0.7121 | 0.7500 | 0.7500 |
| 2 | 0.7353 | 0.9621 | 1.0000 | 1.0000 |
| 5 | 0.9217 | 2.4621 | 2.5000 | 1.0000 |
| 10 | 0.9652 | 4.9621 | 5.0000 | 1.0000 |
| 50 | 0.9936 | 24.9621 | 25.0000 | 1.0000 |

: Randomized response at $\varepsilon = 1$: the exact RDP curve against the Hoeffding bound with the sharp mean, the zCDP line and the line capped at $\varepsilon$. {#tbl-cdp-rr}

Up to $\alpha = 2$ the line is within 36% of the exact value ($1.0000/0.7353 = 1.360$); at $\alpha = 50$ it is 25 times too large ($25.0000/0.9936 = 25.2$), while the cap is within 1% ($1.0000/0.9936 = 1.0064$).

### Check yourself

- A Laplace mechanism is $0.1$-DP. What $\rho$ do you charge for it in a zCDP budget? *($\varepsilon^2/2 = 0.01/2 = 0.005$, or slightly less with the sharp $\varepsilon\tanh(\varepsilon/2)$.)*
- Why can't zCDP describe a $5$-DP mechanism well? *(Charged $\varepsilon^2/2 = 12.5$, its line lies above the cap $\varepsilon = 5$ at every order $\alpha > 1$ (because $12.5\alpha > 5$). The sharp $\rho = 5\tanh(2.5) = 4.9331$ helps little: the line $4.9331\alpha$ passes 5 at $\alpha = 5/4.9331 = 1.0136$. Converting either gives a worse $\varepsilon$ than the $5$ you started with.)*

## From zCDP to $(\varepsilon, \delta)$ {#sec-cdp-conversion}

### Context

Whatever you account in, you report $(\varepsilon, \delta)$. Fact 3 turns an RDP curve into $(\varepsilon, \delta)$: each order $\alpha$ gives a valid $\varepsilon = \varepsilon(\alpha) + \ln(1/\delta)/(\alpha - 1)$, and you take the best order. For a $\rho$-zCDP mechanism the RDP curve is known in closed form, $\rho\alpha$, so that optimization over $\alpha$ can be done by hand. The result is the formula most papers quote. The improved conversion of Fact 3 has no closed form but is easy to minimize numerically.

### Statement

Putting the line $\rho\alpha$ into Mironov's conversion and choosing the best order gives the simple conversion. A $\rho$-zCDP mechanism is $(\varepsilon, \delta)$-DP for every $\delta \in (0, 1)$ with
$$\varepsilon = \rho + 2\sqrt{\rho\ln(1/\delta)},$$ {#eq-cdp-simple}
attained at $\alpha^* = 1 + \sqrt{\ln(1/\delta)/\rho}$ (Bun & Steinke 2016, Proposition 1.3). Putting the same line into the improved conversion (Canonne, Kamath & Steinke 2020, Proposition 12 and Corollary 13) gives a smaller $\varepsilon$, as a minimum over orders that has to be computed numerically:
$$\varepsilon = \min_{\alpha > 1}\left[\rho\alpha + \frac{\ln(1/\delta)}{\alpha - 1} + \ln\Big(1 - \frac1\alpha\Big) - \frac{\ln\alpha}{\alpha - 1}\right].$$ {#eq-cdp-improved}
For calibration you need the reverse: the largest $\rho$ whose simple conversion stays within a target $\varepsilon$ at $\delta$. Solving @eq-cdp-simple for $\rho$, a target $\varepsilon$ at $\delta$ is met by any
$$\rho \le \Big(\sqrt{\varepsilon + \ln(1/\delta)} - \sqrt{\ln(1/\delta)}\Big)^2.$$ {#eq-cdp-inverse}

### Derivation

1. **Plug the line into Mironov's conversion.** By Fact 3 with $\varepsilon(\alpha) = \rho\alpha$, every $\alpha > 1$ gives a valid $\varepsilon$: $g(\alpha) = \rho\alpha + \ell/(\alpha - 1)$, where $\ell = \ln(1/\delta)$. The first term grows with the order and the second shrinks, so there is a best order in between.
2. **Minimize.** $g'(\alpha) = \rho - \ell/(\alpha - 1)^2$ is zero at $\alpha - 1 = \sqrt{\ell/\rho}$. Both terms of $g$ are convex on $\alpha > 1$ (a line, and a positive multiple of $1/(\alpha - 1)$), so $g$ is convex and this stationary point is the minimum.
3. **Evaluate.** At $\alpha^* = 1 + \sqrt{\ell/\rho}$, the first term is $\rho\alpha^* = \rho + \rho\sqrt{\ell/\rho} = \rho + \sqrt{\rho\ell}$ and the second is $\ell/\sqrt{\ell/\rho} = \ell\sqrt{\rho/\ell} = \sqrt{\rho\ell}$. Their sum is $\rho + 2\sqrt{\rho\ell}$, which is @eq-cdp-simple. Bun and Steinke reach the same formula through the tail bound: by @eq-cdp-tail with $\lambda = 2\sqrt{\rho\ell}$, $\Pr[L > \varepsilon] \le e^{-4\rho\ell/(4\rho)} = \delta$ for this $\varepsilon$, and a privacy loss that exceeds $\varepsilon$ with probability at most $\delta$ gives $(\varepsilon, \delta)$-DP (the argument of Fact 3; see also [Composition](../part03-dp-foundations/04-composition.md)).
4. **The improved form** is Fact 3's improved conversion with $\varepsilon(\alpha) = \rho\alpha$. Its derivation is in [From Rényi DP to (ε, δ) and to tests](02-rdp-conversions.md). The script minimizes it over a fine continuous grid of 20,001 log-spaced orders from $1 + 10^{-4}$ to $1 + 10^4$, and, for comparison, over the part's grid `ORDERS`, the fixed list of orders on which this part's accountants store RDP curves.
5. **Inversion.** Write $x = \sqrt\rho$. @eq-cdp-simple reads $x^2 + 2x\sqrt\ell = \varepsilon$. Adding $\ell$ to both sides completes the square: $(x + \sqrt\ell)^2 = \varepsilon + \ell$. Take the positive root, $x = \sqrt{\varepsilon + \ell} - \sqrt\ell$, and square it. Since the simple $\varepsilon$ grows with $\rho$, any smaller $\rho$ also meets the target.

### Interpretation

**For a Gaussian, every bit of slack is in the conversion.** Dividing the output of a Gaussian mechanism by $\sigma$ changes nothing about privacy, because the map can be undone. After the division the two output distributions are $\mathcal N(0, 1)$ and $\mathcal N(\mu, 1)$ with $\mu = \Delta/\sigma$, and $\rho = \Delta^2/(2\sigma^2) = \mu^2/2$. So a $\rho$-zCDP Gaussian mechanism is the pair $\mathcal N(0, 1)$ vs $\mathcal N(\mu, 1)$ with $\mu = \sqrt{2\rho}$, the $\mu$ of $\mu$-Gaussian differential privacy (GDP) in [f-DP and Gaussian differential privacy](04-gaussian-dp.md). For this pair the exact $(\varepsilon, \delta)$ trade-off is known. The **privacy profile** $\delta(\varepsilon)$ is the smallest $\delta$ for which the mechanism is $(\varepsilon, \delta)$-DP, the largest value of $P(S) - e^\varepsilon Q(S)$ over sets of outputs $S$. The best set is the outputs where the privacy loss exceeds $\varepsilon$, so $\delta(\varepsilon) = \Pr_P[L > \varepsilon] - e^\varepsilon\Pr_Q[L > \varepsilon]$. The privacy loss is normal with mean $\mu^2/2$ and variance $\mu^2$ under $P$ (as in @sec-cdp-definition), and with mean $-\mu^2/2$ under $Q$, so each probability is a value of $\Phi$ ([The Laplace and Gaussian mechanisms](../part03-dp-foundations/03-laplace-gaussian.md)):
$$\delta(\varepsilon) = \Phi\Big(\frac\mu2 - \frac\varepsilon\mu\Big) - e^\varepsilon\,\Phi\Big(-\frac\mu2 - \frac\varepsilon\mu\Big).$$ {#eq-cdp-gauss-exact}
The zCDP line is exact for the Gaussian, so the gap between @eq-cdp-simple and the exact $\varepsilon$ from @eq-cdp-gauss-exact is the price of the conversion alone. @fig-cdp-conversion shows it. The simple formula is 21% too large at $\rho = 0.5$ (ratio $5.299/4.377 = 1.210$), the improved conversion 8% (ratio $4.728/4.377 = 1.080$). Both gaps shrink, in relative terms, as $\rho$ grows.

**The grid hardly matters here.** On `ORDERS`, the improved conversion agrees with the fine grid to within $0.001$ for $\rho \le 1$. At $\rho = 50$ the best order on the fine grid is $1.472$, between the grid points $1.4$ and $1.5$, and `ORDERS` gives $96.116$ against $96.035$.

![Left: $\varepsilon$ at $\delta = 10^{-5}$ for a $\rho$-zCDP mechanism, from the simple formula @eq-cdp-simple, the improved conversion @eq-cdp-improved, and the exact value for the Gaussian mechanism with $\mu = \sqrt{2\rho}$. Right: each conversion divided by the exact Gaussian value.](figures/concentrated_dp_conversion.svg){#fig-cdp-conversion}

### Example: $\varepsilon$ for $k$ Gaussian steps

$k$ Gaussian steps with $\sigma = 1$ are $(k/2)$-zCDP, since each costs $1/2$ and composition adds. @tbl-cdp-conversion converts a range of $\rho$ at $\delta = 10^{-5}$, where $\ln(1/\delta) = 11.5129$. The row $\rho = 0.5$ is one step of the part's small example. For it the best order of the simple formula is $\alpha^* = 1 + \sqrt{11.5129/0.5} = 5.799$, and the simple $\varepsilon$ is $0.5 + 2\sqrt{0.5 \times 11.5129} = 5.299$. The row $\rho = 5$ is ten steps, and $\rho = 50$ a hundred.

| $\rho$ | $\mu = \sqrt{2\rho}$ | Simple @eq-cdp-simple | Improved, fine grid | Improved, `ORDERS` | Exact Gaussian |
| --- | --- | --- | --- | --- | --- |
| 0.01 | 0.1414 | 0.689 | 0.546 | 0.546 | 0.497 |
| 0.1 | 0.4472 | 2.246 | 1.914 | 1.914 | 1.760 |
| 0.5 | 1.0000 | 5.299 | 4.728 | 4.729 | 4.377 |
| 1 | 1.4142 | 7.786 | 7.077 | 7.077 | 6.573 |
| 5 | 3.1623 | 20.174 | 19.047 | 19.054 | 17.857 |
| 50 | 10.0000 | 97.985 | 96.035 | 96.116 | 91.817 |

: $\varepsilon$ at $\delta = 10^{-5}$ for $\rho$-zCDP: simple formula, improved conversion on a fine grid of orders and on the part's `ORDERS`, and the exact $\varepsilon$ of the Gaussian mechanism with the same $\rho$. {#tbl-cdp-conversion}

**Calibrating the noise.** To make a single Gaussian step $(1, 10^{-5})$-DP through @eq-cdp-inverse, take $\rho = (\sqrt{12.5129} - \sqrt{11.5129})^2 = 0.02082$, that is $\sigma = 1/\sqrt{2\rho} = 4.901$.

### Check yourself

- You have a budget of $\rho = 1$ and spend $0.5$ on one query. How much $\varepsilon$ at $\delta = 10^{-5}$ is left for the rest? *(The question has no clean answer: $\varepsilon$ does not add, because the conversion is a square root of $\rho$, not a multiple of it. Budgets are kept in $\rho$, which adds, and converted once at the end. The whole run is $7.077$-DP by the improved conversion whatever the split.)*
- Why does the simple formula's best order $\alpha^* = 1 + \sqrt{\ln(1/\delta)/\rho}$ grow as $\rho$ shrinks? *(A small $\rho$ makes the term $\rho\alpha$ cheap, so the conversion can afford a high order, which shrinks the $\ln(1/\delta)/(\alpha - 1)$ term.)*

## Group privacy {#sec-cdp-group}

### Context

A household, a user with many records, or a hospital contributes several records. Group privacy asks how the guarantee degrades for datasets that differ in $m$ records. (This section writes $m$ for the group size, because $k$ is the number of steps in this part.) Two such datasets are joined by a chain of $m$ neighbouring steps, each changing one record. For pure DP the answer is $m\varepsilon$: each step multiplies probabilities by at most $e^\varepsilon$, so $m$ steps multiply them by at most $e^{m\varepsilon}$ ([Properties of DP](../part03-dp-foundations/02-properties.md)). For $(\varepsilon, \delta)$-DP, $\delta$ blows up exponentially in $m$, because each step's $\delta$ is multiplied by the $e^\varepsilon$ factors of the later steps. For RDP the chain argument works only if you give up orders, because Rényi divergence has no triangle inequality: $D_\alpha(P\|Q)$ is not bounded by $D_\alpha(P\|R) + D_\alpha(R\|Q)$ at the same order. zCDP gets a clean answer because it constrains every order at once, so it can pay for the chain in higher orders.

### Statement

The answer is that the slope grows with the square of the group size, exactly as it does for the Gaussian, where a group shifts the mean $m$ times as far. If $M$ is $\rho$-zCDP and $D$, $D'$ are joined by a chain of $m$ neighbouring steps, then for every $\alpha > 1$
$$D_\alpha\big(M(D)\,\|\,M(D')\big) \le m^2\rho\,\alpha,$$ {#eq-cdp-group}
so $M$ is $m^2\rho$-zCDP for groups of size $m$ (Bun & Steinke 2016, Proposition 1.9). The Gaussian mechanism attains it. The tool is a **triangle-like inequality** for Rényi divergence (their Lemma 5.2). It lets you go from $P$ to $Q$ through an intermediate $R$, at the price of using higher orders on the two legs and a factor slightly above 1 on the first: for distributions $P$, $Q$, $R$ and $m > 1$,
$$D_\alpha(P\|Q) \le \frac{m\alpha}{m\alpha - 1}D_{\frac{m\alpha - 1}{m - 1}}(P\|R) + D_{m\alpha}(R\|Q).$$ {#eq-cdp-triangle}
For comparison, Mironov's group bound for plain RDP gives up orders instead: an $(\alpha, \varepsilon)$-RDP mechanism is $(\alpha/2^c, 3^c\varepsilon)$-RDP for groups of $2^c$, when $\alpha \ge 2^{c+1}$ (Mironov 2017, Proposition 2; see [Rényi DP](01-renyi-dp.md)).

### Derivation

**The triangle-like inequality, by Hölder.** Write $P(y)$, $Q(y)$, $R(y)$ for the densities. The idea is to write the sum that defines $D_\alpha(P\|Q)$ as an average under $R$ of a product of two ratios, $P/R$ and $R/Q$, and then split the product with Hölder's inequality. Each factor then becomes a Rényi divergence of one leg of the chain.

1. **Insert $R$.** $e^{(\alpha-1)D_\alpha(P\|Q)} = \sum_y P(y)^\alpha Q(y)^{1-\alpha} = \mathbb E_R\big[(P/R)^\alpha\,(R/Q)^{\alpha - 1}\big]$, where the ratios are evaluated at $Y \sim R$. To check it, multiply out one term: $R \cdot P^\alpha R^{-\alpha} \cdot R^{\alpha - 1}Q^{1 - \alpha} = P^\alpha Q^{1 - \alpha}$. Rewriting a sum under one distribution as an average under another like this is a change of measure (Fact 1 of [Divergences](../part01-prerequisites/03-divergences.md)).
2. **Hölder's inequality.** For positive random variables $X$, $Y$ and exponents $u, v > 1$ with $\frac1u + \frac1v = 1$, $\mathbb E[XY] \le (\mathbb E[X^u])^{1/u}(\mathbb E[Y^v])^{1/v}$. Choose $v$ so that the second factor becomes a divergence of order $m\alpha$, the order the induction below can afford, and $u$ to match: $u = \frac{m\alpha - 1}{\alpha(m - 1)}$ and $v = \frac{m\alpha - 1}{\alpha - 1}$, which satisfy $\frac1u + \frac1v = \frac{\alpha(m-1) + \alpha - 1}{m\alpha - 1} = 1$. Then
$$\mathbb E_R\big[(P/R)^\alpha(R/Q)^{\alpha-1}\big] \le \big(\mathbb E_R[(P/R)^{u\alpha}]\big)^{1/u}\big(\mathbb E_R[(R/Q)^{v(\alpha-1)}]\big)^{1/v}.$$
3. **Recognize two Rényi divergences.** By Fact 1 in its sum form, an average under $R$ of a power of $P/R$ is a Rényi divergence of $P$ from $R$: $\mathbb E_R[(P/R)^{\beta}] = \sum_y R(y)^{1-\beta}P(y)^\beta = e^{(\beta-1)D_\beta(P\|R)}$ with $\beta = u\alpha = \frac{m\alpha - 1}{m - 1}$. In the same way $\mathbb E_R[(R/Q)^{\gamma - 1}] = \sum_y R(y)^{\gamma}Q(y)^{1-\gamma} = e^{(\gamma - 1)D_\gamma(R\|Q)}$ with $\gamma = v(\alpha - 1) + 1 = m\alpha$.
4. **Take logarithms and divide by $\alpha - 1$.** The left side becomes $D_\alpha(P\|Q)$. The second factor contributes $\frac1v\cdot v(\alpha - 1)D_{m\alpha}(R\|Q)/(\alpha - 1) = D_{m\alpha}(R\|Q)$. The first contributes $\frac1u(u\alpha - 1)D_\beta(P\|R)/(\alpha - 1)$, a coefficient $\frac{u\alpha - 1}{u(\alpha - 1)}$. Here $u\alpha - 1 = \frac{m\alpha - 1 - (m - 1)}{m - 1} = \frac{m(\alpha - 1)}{m - 1}$, so the coefficient is $\frac{m(\alpha-1)}{m-1}\cdot\frac{\alpha(m-1)}{(m\alpha - 1)(\alpha - 1)} = \frac{m\alpha}{m\alpha - 1}$. This is @eq-cdp-triangle.

**Group privacy, by induction on $m$.** Now walk along the chain one record at a time, using the triangle-like inequality to add one step to a group of $m - 1$.

5. **Base case.** $m = 1$ is the definition.
6. **Step.** Let $D \simeq_{m-1} \hat D \simeq D'$: $\hat D$ is $m - 1$ steps from $D$ and one step from $D'$. Apply @eq-cdp-triangle with $P = M(D)$, $R = M(\hat D)$, $Q = M(D')$. The first term involves the group of $m - 1$, so by the induction hypothesis at order $\beta = \frac{m\alpha - 1}{m - 1}$ it is at most $\frac{m\alpha}{m\alpha - 1}(m-1)^2\rho\frac{m\alpha - 1}{m - 1} = m(m - 1)\rho\alpha$; the factor $\frac{m\alpha - 1}{m - 1}$ from the order cancels the coefficient's denominator. The second term is a single neighbouring step, so by $\rho$-zCDP at order $m\alpha$ it is at most $m\rho\alpha$. The sum is $(m^2 - m + m)\rho\alpha = m^2\rho\alpha$, which is @eq-cdp-group.
7. **Tightness.** For the Gaussian mechanism a group of $m$ shifts the mean by $m\Delta$, and the Gaussian Rényi divergence $\alpha s^2/(2\sigma^2)$ with $s = m\Delta$ (@eq-gauss-renyi) gives exactly $\alpha m^2\Delta^2/(2\sigma^2) = m^2\rho\alpha$. Along the chain $\mathcal N(0, 1)$, $\mathcal N(1, 1)$, $\mathcal N(2, 1)$, @eq-cdp-triangle holds with equality: at $\alpha = 3$, $m = 2$, the left side is $D_3$ for a shift of 2, $3 \cdot 4/2 = 6$; on the right, the first leg has order $(6 - 1)/1 = 5$ and coefficient $6/5$, giving $\frac65 \cdot \frac52 = 3$, and the second leg has order 6, giving $6/2 = 3$. Both sides are $6$.

### Interpretation

**Quadratic in $m$, and that is right.** With $\rho \approx \varepsilon^2/2$ for pure DP, $m^2\rho \approx (m\varepsilon)^2/2$: zCDP's group privacy matches pure DP's $m\varepsilon$ after the translation from $\varepsilon$ to $\rho$. The RDP route is weaker on two counts. It halves the order at each doubling of the group, so a curve known on all orders becomes a curve known only on orders up to $\alpha/2^c$. And it multiplies $\varepsilon$ by 3 per doubling, so the value $\rho\alpha$ at order $\alpha$ becomes $3^c\rho\alpha$ at order $\alpha' = \alpha/2^c$. In terms of the new order that is $3^c 2^c\rho\alpha' = 6^c\rho\alpha'$: a slope of $6^c\rho$, against zCDP's $m^2\rho = 4^c\rho$.

**Strong group privacy is also a weakness.** Bun and Steinke use exactly this property to prove lower bounds for zCDP that are as strong as those for pure DP. For example, stability-based algorithms, which are $(\varepsilon, \delta)$-DP with sample sizes independent of the domain size, cannot be zCDP. That is the motivation for the next two sections: approximate zCDP and truncated CDP both weaken group privacy for large groups to escape these bounds.

### Example: households under the Gaussian mechanism

@tbl-cdp-group takes the Gaussian mechanism with $\sigma = 1$ ($\rho = 0.5$) and groups of size $m$. The RDP route starts from the curve at order 64, where $\varepsilon(64) = 0.5 \times 64 = 32$. For a group of $m = 2^c$ it gives order $64/2^c$, $\varepsilon = 3^c \times 32$, and slope $6^c \times 0.5$: for $m = 2$ that is order 32, $\varepsilon = 96$ and slope 3.

| Group size $m$ | zCDP $m^2\rho$ | $\varepsilon$ at $10^{-5}$ (improved) | Exact Gaussian, shift $m$ | RDP route: order, $\varepsilon$, slope |
| --- | --- | --- | --- | --- |
| 1 | 0.5 | 4.728 | 4.377 | — |
| 2 | 2 | 10.725 | 9.997 | 32, 96, 3 |
| 4 | 8 | 25.919 | 24.382 | 16, 288, 18 |
| 8 | 32 | 68.616 | 65.319 | 8, 864, 108 |

: Group privacy of the Gaussian mechanism with $\sigma = 1$. zCDP gives slope $m^2\rho$ at every order, and its conversion is within about 8% of the exact value. Mironov's RDP group bound, started from order 64, gives slope $6^c\rho$ and only on orders up to $64/m$. {#tbl-cdp-group}

For comparison, a $1$-DP mechanism gives $8$-DP for groups of 8.

### Check yourself

- A user contributes up to 4 records to a $0.1$-zCDP release. What $\rho$ protects the user? *($4^2 \times 0.1 = 1.6$.)*
- Why does the induction in step 6 need $\rho$-zCDP at order $m\alpha$, larger than $\alpha$? *(Hölder trades one divergence of order $\alpha$ for two of higher orders. zCDP controls every order, so the trade is free; a bound known only up to some order would run out after a few steps, which is what happens to truncated CDP.)*

## Approximate zCDP {#sec-cdp-approx}

### Context

Some useful mechanisms are $(\varepsilon, \delta)$-DP but not zCDP at all. Propose-test-release and stability-based histograms check, privately, that the data are far from any dataset where the answer would change, release an exact answer when they are, and fail only with probability about $\delta$. The group-privacy lower bound above shows they cannot be zCDP. You still want to compose them with Gaussian steps in one budget. Bun and Steinke's answer is to allow zCDP to fail on an event of probability $\delta$.

### Statement

$M$ is **$\delta$-approximately $\rho$-zCDP** if, after throwing away a bad event of probability at most $\delta$ in each world, the rest is $\rho$-zCDP. Precisely: for all neighbours $D \simeq D'$ there are events $E$ (for $M(D)$) and $E'$ (for $M(D')$), each of probability at least $1 - \delta$, such that for every $\alpha > 1$
$$D_\alpha\big(M(D)|_E\,\|\,M(D')|_{E'}\big) \le \rho\alpha \quad\text{and}\quad D_\alpha\big(M(D')|_{E'}\,\|\,M(D)|_E\big) \le \rho\alpha,$$
where $M(D)|_E$ is the output distribution conditioned on $E$ (Bun & Steinke 2016, Definition 1.12). With $\delta = 0$ the events are certain and this is $\rho$-zCDP.

Three more properties make it a usable ledger. First, it absorbs $(\varepsilon, \delta)$-DP: an $(\varepsilon, \delta)$-DP mechanism is $\delta$-approximately $(\varepsilon^2/2)$-zCDP (their Proposition 1.13 and Corollary 8.7). Second, it composes column by column: $\delta_1$-approximate $\rho_1$-zCDP followed by $\delta_2$-approximate $\rho_2$-zCDP is $(\delta_1 + \delta_2 - \delta_1\delta_2)$-approximately $(\rho_1 + \rho_2)$-zCDP (their Lemma 8.2). For many steps the failure probability is $1 - \prod_i(1 - \delta_i) \le \sum_i\delta_i$. Third, it converts to $(\varepsilon, \delta)$ by adding the failure probability to the conversion's $\delta$: a $\delta_a$-approximately $\rho$-zCDP mechanism is $(\varepsilon, \delta_a + (1 - \delta_a)\delta')$-DP, where $\varepsilon$ is any conversion of $\rho$-zCDP at $\delta'$ (their Lemma 8.8, stated there with their own refinement of the conversion; the argument below works with any of them).

### Derivation

This derives the conversion, which is what you use at the end. The idea: on the good event the mechanism is $\rho$-zCDP and the usual conversion applies; the bad event can add at most its probability. Let $\Pr[E], \Pr[E'] \ge 1 - \delta_a$ and fix a set of outputs $S$.

1. **Split on the event.** $\Pr[M(D) \in S] \le \Pr[M(D) \in S \mid E]\Pr[E] + (1 - \Pr[E])$, because the probability of $S$ on the complement of $E$ is at most 1. Write $a = \Pr[M(D) \in S \mid E]$; the right side is $a\Pr[E] + 1 - \Pr[E] = 1 - \Pr[E](1 - a)$, which can only shrink as $\Pr[E]$ grows. So its value at the smallest allowed $\Pr[E]$ bounds it: it is at most $a(1 - \delta_a) + \delta_a$.
2. **Use the conditional zCDP.** The conditioned pair is $\rho$-zCDP, so by any conversion, $\Pr[M(D) \in S \mid E] \le e^\varepsilon\Pr[M(D') \in S \mid E'] + \delta'$.
3. **Remove the conditioning on $D'$.** Dropping the outputs outside $E'$ can only lower a probability: $\Pr[M(D') \in S] \ge \Pr[M(D') \in S \mid E']\Pr[E'] \ge \Pr[M(D') \in S \mid E'](1 - \delta_a)$.
4. **Combine.** Put step 2 into step 1 and use step 3 for the term with $e^\varepsilon$: $\Pr[M(D) \in S] \le (1 - \delta_a)\big(e^\varepsilon\Pr[M(D') \in S \mid E'] + \delta'\big) + \delta_a \le e^\varepsilon\Pr[M(D') \in S] + (1 - \delta_a)\delta' + \delta_a$.

The other two properties have shorter reasons. The conversion from $(\varepsilon, \delta)$-DP comes from a decomposition of any $(\varepsilon, \delta)$-DP pair: with probability $1 - \delta$ it behaves like an $\varepsilon$-DP pair, and @sec-cdp-pure turns that into $\varepsilon^2/2$. Composition holds because the conditioned mechanisms compose as zCDP (Fact 2), and the good events of both steps hold together with probability at least $(1 - \delta_1)(1 - \delta_2) = 1 - (\delta_1 + \delta_2 - \delta_1\delta_2)$.

### Interpretation

Approximate zCDP is a way to keep one ledger with two columns, $\rho$ and $\delta_a$. The $\rho$ column composes like zCDP, with its $\sqrt k$ behaviour after conversion; the $\delta_a$ column adds up, like $\delta$ in basic composition. It is only worth using when some steps are genuinely $(\varepsilon, \delta)$ and not zCDP. For a step that is zCDP, putting it in the $\delta_a$ column wastes $\delta$.

### Example: a pipeline with stability-based steps

A pipeline runs 10 stability-based steps, each $(0.5, 10^{-7})$-DP, and 10 Gaussian steps with $\sigma = 1$. Each stability step enters the $\rho$ column as $0.5^2/2 = 0.125$ and the $\delta_a$ column as $10^{-7}$. The total is $\rho = 10 \times 0.125 + 10 \times 0.5 = 6.25$ and $\delta_a = 1 - (1 - 10^{-7})^{10} = 1.0000 \times 10^{-6}$. Converting with the improved conversion at $\delta' = 10^{-5} - \delta_a$ gives $\varepsilon = 22.100$ at total $\delta = 10^{-5}$. Without approximate zCDP you would convert the Gaussians alone ($\varepsilon = 19.119$ at $\delta = 9 \times 10^{-6}$) and add the stability steps by basic composition ($\varepsilon = 10 \times 0.5 = 5$, $\delta = 10^{-6}$), for $19.119 + 5 = 24.119$. The unified ledger saves about 2 because the $\varepsilon = 0.5$ steps compose in $\rho$ as $\varepsilon^2/2$ each, not as $\varepsilon$.

### Check yourself

- Is $(\varepsilon, 0)$-DP the same as $0$-approximate $(\varepsilon^2/2)$-zCDP? *(No. $\varepsilon$-DP implies $(\varepsilon^2/2)$-zCDP, but zCDP does not imply pure DP: the Gaussian is zCDP and not $\varepsilon$-DP for any finite $\varepsilon$, because its privacy loss is normal and so unbounded.)*
- Why do the $\delta_a$ add while the $\rho$ add only under a square root in $\varepsilon$? *(The $\delta_a$ are failure probabilities of events and combine by the union bound. The $\rho$ add, and the conversion turns $\rho$ into $\varepsilon \approx 2\sqrt{\rho\ln(1/\delta)}$.)*

## Subsampling, and truncated CDP {#sec-cdp-tcdp}

### Context

DP-SGD, the running example of this part, runs each step on a Poisson sample of the data: each record is used independently with probability $q$. Subsampling makes a step much more private than the Gaussian it contains, because a record is usually not in the batch. Does zCDP see this? In this section $q$ is the sampling rate, never a density.

Take add/remove neighbours (two datasets differ by one record being present or absent), sensitivity 1 and noise multiplier $\sigma$, and reduce to one dimension. Without the record the step outputs $Q = \mathcal N(0, \sigma^2)$. With it, the record is in the sample with probability $q$, shifting the output's mean by 1, and absent otherwise, so the output is the mixture $P = (1 - q)\mathcal N(0, \sigma^2) + q\,\mathcal N(1, \sigma^2)$. The RDP curve compares $P$ against $Q$; the other order, $Q$ against $P$, is never larger (both facts are derived in [Amplification by subsampling](05-subsampling.md), @eq-sub-rdp). For integer orders the curve has a closed form, a binomial sum over how many of $\alpha$ copies of the record's contribution are present:
$$\varepsilon(\alpha) = \frac{1}{\alpha - 1}\ln A_\alpha, \qquad A_\alpha = \sum_{j=0}^{\alpha}\binom\alpha j(1 - q)^{\alpha - j}q^j\exp\Big(\frac{j^2 - j}{2\sigma^2}\Big)$$ {#eq-cdp-sg}
(Mironov, Talwar & Zhang 2019). To see where it comes from, let $r$ be the likelihood ratio of $\mathcal N(1, \sigma^2)$ to $Q$. Then $P/Q = 1 - q + q\,r$, and by Fact 1, $A_\alpha = e^{(\alpha - 1)D_\alpha(P\|Q)} = \mathbb E_{Q}[(P/Q)^\alpha] = \mathbb E_Q[(1 - q + q\,r)^\alpha]$. Expand the power binomially. Each term needs $\mathbb E_Q[r^j]$, which by Fact 1 is $e^{(j - 1)D_j}$ for the pair $\mathcal N(1, \sigma^2)$, $\mathcal N(0, \sigma^2)$; with $D_j = j/(2\sigma^2)$ (@eq-gauss-renyi at order $j$) that is $e^{(j^2 - j)/(2\sigma^2)}$.

### Statement

**No amplification for zCDP.** For the Poisson-subsampled Gaussian with any $q \in (0, 1]$, the smallest $\rho$ with $\varepsilon(\alpha) \le \rho\alpha$ for all $\alpha$ is $1/(2\sigma^2)$, the same as without subsampling.

**Truncated CDP.** Since the problem comes from large orders, ask for the line only on orders below a cut-off $\omega$. A mechanism is **$(\rho, \omega)$-tCDP** if for all neighbours and all $\alpha \in (1, \omega)$
$$D_\alpha\big(M(D)\,\|\,M(D')\big) \le \rho\alpha$$
(Bun, Dwork, Rothblum & Steinke 2018, Definition 1). $\rho$-zCDP is the case $\omega = \infty$, that is $(\rho, \infty)$-tCDP. Composition works as for zCDP on the orders both steps cover: $(\rho_1, \omega_1)$ and $(\rho_2, \omega_2)$ compose to $(\rho_1 + \rho_2, \min(\omega_1, \omega_2))$ (their Lemma 2). The conversion to $(\varepsilon, \delta)$ is the simple zCDP conversion when its best order is below $\omega$, and uses the order $\omega$ itself otherwise (their Lemma 6):
$$\varepsilon = \begin{cases}\rho + 2\sqrt{\rho\ln(1/\delta)} & \text{if } \ln(1/\delta) \le (\omega - 1)^2\rho,\\[2pt] \rho\omega + \dfrac{\ln(1/\delta)}{\omega - 1} & \text{otherwise.}\end{cases}$$ {#eq-cdp-tcdp-conv}
And, unlike zCDP, tCDP is amplified by subsampling (their Theorem 11): let $\rho, s \in (0, 0.1]$ with $\ln(1/s) \ge 3\rho(2 + \log_2(1/\rho))$, and let $M$ be $(\rho, \omega')$-tCDP with $\omega' \ge \ln(1/s)/(2\rho) \ge 3$. Running $M$ on a uniformly random subset of a fraction $s$ of the records (fixed size, without replacement) is $(13s^2\rho,\ \omega)$-tCDP with $\omega = \ln(1/s)/(4\rho)$.

### Derivation

**No amplification for zCDP.** Write $\rho_0 = 1/(2\sigma^2)$, the slope of the Gaussian without subsampling. The plan is to show that $\rho_0$ is enough (subsampling never hurts) and that nothing smaller is enough (at large orders, subsampling stops helping).

1. **Upper bound, by convexity.** $1 - q + q\,r$ is an average of the values 1 and $r$ with weights $1 - q$ and $q$. Since $x \mapsto x^\alpha$ is convex, the power of an average is at most the average of the powers: $(1 - q + q\,r)^\alpha \le (1 - q) + q\,r^\alpha$. Take $\mathbb E_Q$; with $\mathbb E_Q[r^\alpha] = e^{(\alpha - 1)\alpha\rho_0}$ from the Gaussian curve, $A_\alpha \le 1 - q + q\,e^{(\alpha - 1)\alpha\rho_0} \le e^{(\alpha - 1)\alpha\rho_0}$, the last step because an average of 1 and a number at least 1 is at most that number. So $D_\alpha(P\|Q) \le \rho_0\alpha$. The other order obeys the same bound, by a convexity argument on pairs. For any pair $(A, B)$, $e^{(\alpha - 1)D_\alpha(A\|B)} = \sum_y B(y)\,(A(y)/B(y))^\alpha$ is the perspective of the convex function $x^\alpha$, which makes it jointly convex in $(A, B)$: the value at an average of pairs is at most the average of the values. Writing the pair $(Q, P)$ as the mixture $(1 - q)\,(Q, Q) + q\,(Q, \mathcal N(1, \sigma^2))$ gives $e^{(\alpha - 1)D_\alpha(Q\|P)} \le (1 - q) + q\,e^{(\alpha - 1)\alpha\rho_0}$, the same bound. Subsampling never hurts.
2. **Lower bound, by one term.** Every term of @eq-cdp-sg is positive, so dropping terms gives a lower bound. Keep only $j = \alpha$, the event that the record is present in all $\alpha$ copies: $A_\alpha \ge q^\alpha e^{(\alpha^2 - \alpha)\rho_0}$. Taking logs and dividing by $\alpha - 1$,
$$\varepsilon(\alpha) \ge \rho_0\alpha + \frac{\alpha\ln q}{\alpha - 1}.$$ {#eq-cdp-sg-lower}
3. **Divide by $\alpha$ and let $\alpha \to \infty$.** $\varepsilon(\alpha)/\alpha \ge \rho_0 + \ln q/(\alpha - 1) \to \rho_0$: the penalty $\ln q$ for the record being rarely sampled is fixed, while the Gaussian part grows with the order, so the penalty's share of the slope vanishes. A line $\rho\alpha$ above the whole curve needs $\rho \ge \rho_0$, and by step 1, $\rho_0$ suffices.

**The tCDP conversion.** This is the zCDP conversion of @sec-cdp-conversion with the orders restricted to $(1, \omega]$.

4. **Restrict Mironov's conversion.** As in @sec-cdp-conversion, every $\alpha \in (1, \omega]$ gives a valid $g(\alpha) = \rho\alpha + \ell/(\alpha - 1)$ with $\ell = \ln(1/\delta)$. (The endpoint $\omega$ is allowed: $D_\alpha$ is non-decreasing in $\alpha$ and lower semicontinuous, as a supremum over finite partitions of functions continuous in $\alpha$, so $D_\omega \le \lim_{\alpha \uparrow \omega}D_\alpha \le \rho\omega$.)
5. **Constrained minimum of a convex function.** The unconstrained minimizer is $\alpha^* = 1 + \sqrt{\ell/\rho}$. If $\alpha^* \le \omega$, that is $\ell \le (\omega - 1)^2\rho$, use it and get @eq-cdp-simple. Otherwise $\alpha^*$ lies beyond $\omega$; a convex function is decreasing to the left of its minimizer, so $g$ is decreasing on $(1, \omega]$, the best allowed order is $\omega$, and it gives $\rho\omega + \ell/(\omega - 1)$.

The subsampling theorem has a longer proof (their Section 3): write the subsampled pair as mixtures $sP + (1 - s)R$ and $sQ + (1 - s)R$, expand $(1 + sZ)^\alpha$ with $Z = (P - Q)/(sQ + (1 - s)R)$, and bound the second moment of $Z$ through $D_2$ and the higher moments through the tCDP bound at orders up to $\omega'$. The second-moment term carries the factor $s^2$ that gives the amplification. The truncation is essential: the higher-order terms are controlled only for $\alpha$ up to about $\ln(1/s)/\rho$.

### Interpretation

**The curve bends, and a line cannot.** @fig-cdp-subsampled shows the curve of one DP-SGD step, $q = 0.01$, $\sigma = 1$, on integer orders. Up to $\alpha = 8$ it is tiny and almost linear. Its slope there is close to $q^2(e^{1/\sigma^2} - 1)/2 = 10^{-4} \times 1.71828/2 = 8.5914 \times 10^{-5}$. That value comes from the first terms of @eq-cdp-sg. If every exponential in the sum were 1, the sum would be $((1 - q) + q)^\alpha = 1$; the $j = 0$ and $j = 1$ exponentials are exactly 1, so the excess over 1 starts at $j = 2$, and for small $q$ it is $A_\alpha \approx 1 + \binom\alpha2q^2(e^{1/\sigma^2} - 1)$. Then $\ln A_\alpha/(\alpha - 1) \approx \alpha q^2(e^{1/\sigma^2} - 1)/2$. Between $\alpha = 8$ and $\alpha = 16$ the curve jumps by more than three orders of magnitude, from $8.9364 \times 10^{-4}$ to $3.0879$ (@tbl-cdp-sg). Beyond that it follows the lower bound @eq-cdp-sg-lower, which turns positive where $\rho_0(\alpha - 1) = \ln(1/q)$, at $\alpha = 1 + 2\sigma^2\ln(1/q) = 1 + 2\ln 100 = 10.21$. At large orders the term $j = \alpha$ dominates: it is the event that the record is in every one of $\alpha$ imagined samples, rare but with the full Gaussian loss. A line through the origin above the whole curve must have the slope at infinity, $\rho_0 = 0.5$. A line that only has to stay above the curve up to $\omega = 8$ can have slope $1.1171 \times 10^{-4}$, $0.5/(1.1171 \times 10^{-4}) = 4476$ times smaller.

**Small orders are what matter for DP-SGD.** After $k = 10{,}000$ steps the RDP curve is $k\varepsilon(\alpha)$, and the conversion picks the order where $k\varepsilon(\alpha) + \ln(1/\delta)/(\alpha - 1)$ is smallest. With per-step values around $10^{-4}$, the first term is small at small orders (from $10^4 \times 1.7181 \times 10^{-4} = 1.7181$ at order 2 to $8.9364$ at order 8), and the order is pushed up only until the curve's jump makes the first term expensive. That is a small order: on the integer orders of `ORDERS` the best is $\alpha = 4$, for Mironov's conversion and for the improved one. The large orders that set the zCDP slope are never used. So the right one-number summary of DP-SGD keeps $\rho$ only on small orders, which is what $(\rho, \omega)$-tCDP does. RDP accounting keeps the whole curve and does at least as well.

**tCDP's theorem is loose in its constant.** Theorem 11 holds for every tCDP mechanism, so its constant 13 is generous. With $\sigma = \sqrt5$ ($\rho = 1/(2 \cdot 5) = 0.1$, so the theorem applies) and $s = 0.01$, the side condition reads $3\rho(2 + \log_2(1/\rho)) = 1.597 \le \ln(1/s) = 4.605$, and the theorem gives $(13 \times 0.01^2 \times 0.1,\ 4.605/0.4) = (1.300 \times 10^{-4},\ 11.51)$-tCDP. The Poisson-subsampled Gaussian curve on integer orders up to $11.51$ needs only $\rho = 1.128 \times 10^{-5}$, about 11 times less. After $10{,}000$ steps the theorem's parameters convert to $\varepsilon = 9.037$ by @eq-cdp-tcdp-conv, while RDP accounting of the Poisson curve on the integer orders of `ORDERS`, with the improved conversion, gives $2.046$.

This comparison is only indicative. The theorem is for fixed-size sampling and replace-one neighbours, the curve for Poisson sampling and add/remove neighbours, and [Amplification by subsampling](05-subsampling.md) explains why the two must not be mixed in an accountant.

![The RDP curve of one Poisson-subsampled Gaussian step with $q = 0.01$, $\sigma = 1$, on integer orders (black), against the best zCDP line $\rho\alpha$ with $\rho = 1/(2\sigma^2) = 0.5$ (red dashed), which is the unsubsampled Gaussian's line $\alpha/2$, and the tCDP line on orders up to $\omega = 8$ (blue). Both axes are logarithmic.](figures/concentrated_dp_subsampled.svg){#fig-cdp-subsampled}

### Example: the running DP-SGD example in zCDP and tCDP

@tbl-cdp-sg lists the per-step curve. All values are on integer orders, which is what @eq-cdp-sg computes; a guarantee between grid points needs the fractional-order formula of [Amplification by subsampling](05-subsampling.md).

| $\alpha$ | $\varepsilon(\alpha)$ | $\varepsilon(\alpha)/\alpha$ | Lower bound @eq-cdp-sg-lower | No subsampling, $\alpha/2$ |
| --- | --- | --- | --- | --- |
| 2 | $1.7181 \times 10^{-4}$ | $8.5907 \times 10^{-5}$ | −8.2103 | 1 |
| 4 | $3.6315 \times 10^{-4}$ | $9.0789 \times 10^{-5}$ | −4.1402 | 2 |
| 8 | $8.9364 \times 10^{-4}$ | $1.1171 \times 10^{-4}$ | −1.2631 | 4 |
| 10 | $3.8270 \times 10^{-2}$ | $3.8270 \times 10^{-3}$ | −0.1169 | 5 |
| 12 | $9.7801 \times 10^{-1}$ | $8.1501 \times 10^{-2}$ | 0.9762 | 6 |
| 16 | $3.0879$ | $1.9299 \times 10^{-1}$ | 3.0878 | 8 |
| 64 | $2.7322 \times 10^{1}$ | $4.2690 \times 10^{-1}$ | 27.3217 | 32 |
| 512 | $2.5139 \times 10^{2}$ | $4.9099 \times 10^{-1}$ | 251.3858 | 256 |

: The RDP curve of one DP-SGD step ($q = 0.01$, $\sigma = 1$) on integer orders: its value, its slope from the origin, the lower bound @eq-cdp-sg-lower, and the Gaussian without subsampling. {#tbl-cdp-sg}

**zCDP.** The steepest ratio on the integer orders of `ORDERS` is $0.4910$, at $\alpha = 512$, but that is not a valid zCDP parameter: zCDP needs the line at every order, and the ratio tends to $0.5$ as $\alpha \to \infty$. So zCDP must charge $\rho = 0.5$ per step, the same as without subsampling. After $10{,}000$ steps, $\rho = 5000$, and $\varepsilon = 5000 + 2\sqrt{5000 \times 11.5129} = 5479.9$ at $\delta = 10^{-5}$ by @eq-cdp-simple, or $5475.8$ with the improved conversion. That is the number zCDP reports for a training run that RDP accounting, on the integer orders of `ORDERS`, puts at $\varepsilon = 7.469$ with Mironov's conversion and $6.719$ with the improved one. [Amplification by subsampling](05-subsampling.md) also uses the fractional orders of `ORDERS` and gets slightly lower values (@tbl-sub-dpsgd).

**tCDP.** @tbl-cdp-tcdp takes the per-step slope on orders up to $\omega$, composes $10{,}000$ steps by adding the slopes, and converts with @eq-cdp-tcdp-conv.

| $\omega$ | $\rho$ per step | $\rho$ after $10^4$ steps | $\varepsilon$ at $10^{-5}$ |
| --- | --- | --- | --- |
| 2 | $8.5907 \times 10^{-5}$ | 0.8591 | 13.231 |
| 3 | $8.8213 \times 10^{-5}$ | 0.8821 | 8.403 |
| 4 | $9.0789 \times 10^{-5}$ | 0.9079 | 7.469 |
| 5 | $9.3733 \times 10^{-5}$ | 0.9373 | 7.507 |
| 6 | $9.7250 \times 10^{-5}$ | 0.9725 | 7.665 |
| 8 | $1.1171 \times 10^{-4}$ | 1.1171 | 8.289 |
| 10 | $3.8270 \times 10^{-3}$ | 38.2704 | 80.252 |
| 16 | $1.9299 \times 10^{-1}$ | 1929.9067 | 2228.027 |

: tCDP accounting of the running example, $q = 0.01$, $\sigma = 1$, $k = 10{,}000$, $\delta = 10^{-5}$, for several truncation points $\omega$. Slopes are taken on integer orders up to $\omega$. {#tbl-cdp-tcdp}

The best integer choice, $\omega = 4$, reproduces RDP with Mironov's conversion, $7.469$. A smaller $\omega$ lowers the slope only a little but forbids the orders the conversion needs. With $\omega = 4$, $\ln(1/\delta) = 11.51 > (\omega - 1)^2\rho = 9 \times 0.9079 = 8.17$, so the second branch of @eq-cdp-tcdp-conv applies: $4 \times 0.9079 + 11.5129/3 = 7.469$, which is Mironov's conversion at order 4. Using the first branch anyway would give $7.374$, but that needs order $\alpha^* = 1 + \sqrt{11.5129/0.9079} = 4.561 > \omega$, where the tCDP bound says nothing. Once $\omega$ passes the bend near $\alpha = 10$, tCDP collapses towards zCDP.

### Check yourself

- Why do the largest orders decide zCDP's $\rho$ for a subsampled Gaussian, when no conversion ever uses them? *(zCDP requires the line at every order. The ratio $\varepsilon(\alpha)/\alpha$ keeps growing towards $1/(2\sigma^2)$ as $\alpha \to \infty$, so the limit sets the slope, and $\rho = 1/(2\sigma^2)$.)*
- Two DP-SGD runs, one $(10^{-4}, 4)$-tCDP per step and one $(10^{-5}, 20)$-tCDP per step, are composed. What is the result? *($(\rho, \omega) = (1.1 \times 10^{-4}, 4)$ per pair of steps: $\rho$ adds and $\omega$ is the minimum, because the line is guaranteed for both steps only on the orders both cover.)*

## Code: check every number

The script [code/concentrated_dp.py](code/concentrated_dp.py) computes every number in this chapter and draws every figure. The Gaussian privacy loss is checked against simulation, randomized response and the binary bounded-range mechanisms are evaluated exactly in log space, the conversions are minimized on a fine grid and on `ORDERS` and compared with the exact Gaussian $\delta(\varepsilon)$, and the subsampled-Gaussian curve is computed with @eq-cdp-sg by a log-sum-exp over the binomial terms. From the repository root:

```bash
uv run python book/part04-privacy-accounting/code/concentrated_dp.py
```

**Output** (seed 0), summarized in @tbl-cdp-code:

| Experiment | Result |
| --- | --- |
| Gaussian privacy loss, $\sigma = 1$ | Simulated mean 0.4990, variance 1.0013, $\ln\mathbb E[e^L] = 1.0000$; tails in @tbl-cdp-tail; exact/bound 0.094 at $\lambda = 4$ |
| Pure DP to zCDP | Smallest $\rho$ of randomized response on a fine grid of orders $= \varepsilon\tanh(\varepsilon/2)$ (@eq-cdp-pure-sharp): 0.1225 ($\varepsilon = 0.5$), 0.4621 ($\varepsilon = 1$), against $\varepsilon^2/2$ = 0.1250, 0.5000; curve in @tbl-cdp-rr |
| Bounded range, $\varepsilon^2/8$ | Largest $D_\alpha/(\alpha\varepsilon^2/8)$: 0.9965, 0.9864, 0.8348 at $\varepsilon$ = 0.5, 1, 4 |
| zCDP to $(\varepsilon, 10^{-5})$ | @tbl-cdp-conversion; at $\rho = 0.5$ the simple formula is 1.210 and the improved 1.080 times the exact 4.377 |
| Group privacy, $\sigma = 1$ | @tbl-cdp-group; triangle-like inequality on Gaussians: both sides 6 |
| Approximate zCDP pipeline | $\rho = 6.25$, $\delta_a = 1.0000 \times 10^{-6}$, $\varepsilon = 22.100$; without it 24.119 |
| DP-SGD, $q = 0.01$, $\sigma = 1$, $k = 10^4$ | zCDP: $\rho = 5000$, $\varepsilon = 5479.9$ (simple), 5475.8 (improved). tCDP, $\omega = 4$: $\varepsilon = 7.469$. RDP on integer orders: 7.469 (Mironov), 6.719 (improved), both at $\alpha = 4$ |

: The script's output with the default seed. {#tbl-cdp-code}

[tests/test_concentrated_dp.py](../../tests/test_concentrated_dp.py) checks that the output still matches.

**What to take from it.** For Gaussian mechanisms zCDP is exact, and all the slack is in the conversion: 8% with the improved conversion at $\rho = 0.5$. For pure DP mechanisms the sharp constant is $\varepsilon\tanh(\varepsilon/2)$, and $\varepsilon^2/2$ is close to it for small $\varepsilon$, but any line ignores the cap $\varepsilon$; for large $\varepsilon$ keep the RDP curve, for instance that of randomized response. For subsampled mechanisms zCDP is useless: it reports $\varepsilon$ in the thousands for a run that is $\varepsilon \approx 7$, because its slope is set by orders the conversion never uses. tCDP, or the full RDP curve, recovers the amplification.

## References

- Bun & Steinke, "Concentrated Differential Privacy: Simplifications, Extensions, and Lower Bounds" (TCC 2016; arXiv:1605.02065). The source for this chapter: the definition (1.1), the Gaussian (1.6), pure DP to zCDP (1.4, proved as 3.3), conversion (1.3, 3.5, 3.6), group privacy (1.9, 5.3), approximate zCDP (1.12, Section 8), and the lower bounds. Read Sections 1–3 first.
- Dwork & Rothblum, "Concentrated Differential Privacy" (2016, arXiv:1603.01887). The original two-parameter definition and its motivation from advanced composition.
- Bun, Dwork, Rothblum & Steinke, "Composable and Versatile Privacy via Truncated CDP" (STOC 2018). Truncated CDP: definition, composition, conversion (Lemma 6), amplification by subsampling (Theorem 11) and the sinh-normal mechanism.
- Canonne, Kamath & Steinke, "The Discrete Gaussian for Differential Privacy" (NeurIPS 2020). The discrete Gaussian and its zCDP guarantee, and the improved conversion from RDP and zCDP to $(\varepsilon, \delta)$ (Proposition 12, Corollary 13), with a numerically stable way to compute it.
- Cesar & Rogers, "Bounding, Concentrating, and Truncating: Unifying Privacy Loss Composition for Data Analytics" (Algorithmic Learning Theory 2021). Composition of pure DP, bounded-range (exponential mechanism) and CDP steps in one analysis, including the $\varepsilon^2/8$ bound for bounded-range mechanisms.
- Durfee & Rogers, "Practical Differentially Private Top-k Selection with Pay-what-you-get Composition" (NeurIPS 2019). Introduces the bounded-range property of the exponential mechanism.
- Dong, Durfee & Rogers, "Optimal Differential Privacy Composition for Exponential Mechanisms" (ICML 2020). Exact composition of bounded-range mechanisms, and the mean bound that Cesar & Rogers use.
- Mironov, "Rényi Differential Privacy" (CSF 2017). The RDP framework this chapter restricts to lines, and the RDP group-privacy bound used for comparison.
- Mironov, Talwar & Zhang, "Rényi Differential Privacy of the Sampled Gaussian Mechanism" (2019, arXiv:1908.10530). The subsampled-Gaussian curve @eq-cdp-sg.
- Abowd et al., "The 2020 Census Disclosure Avoidance System TopDown Algorithm" (Harvard Data Science Review, 2022; arXiv:2204.08986). zCDP accounting with the discrete Gaussian at national scale, and the role of invariants.

## Cheat sheet

@tbl-cdp-cheat collects the results. $\ell = \ln(1/\delta)$.

| Result | Formula | Tight? |
| --- | --- | --- |
| $\rho$-zCDP | $D_\alpha \le \rho\alpha$ for all $\alpha > 1$; $\ln\mathbb E[e^{(\alpha-1)L}] \le (\alpha-1)\alpha\rho$ | Equality for the Gaussian |
| Privacy-loss tail | $\Pr[L > \rho + \lambda] \le e^{-\lambda^2/(4\rho)}$ | Off by the Mills-ratio factor $\sqrt{\rho/\pi}/\lambda$ for the Gaussian |
| Gaussian mechanism | $\rho = \Delta^2/(2\sigma^2)$; privacy loss $\sim \mathcal N(\rho, 2\rho)$ | Exact |
| Composition, post-processing | $\rho$'s add; functions of the output keep $\rho$ | Exact for Gaussians |
| Pure DP | $\varepsilon$-DP $\Rightarrow$ $(\varepsilon^2/2)$-zCDP; curve $\le \min(\varepsilon, \alpha\varepsilon^2/2)$ | Sharp only as $\varepsilon \to 0$ |
| Pure DP, sharp | $\varepsilon$-DP $\Rightarrow$ $(\varepsilon\tanh(\varepsilon/2))$-zCDP; curve $\le$ that of $\varepsilon$-randomized response | Attained by randomized response |
| Exponential mechanism | $(\varepsilon^2/8)$-zCDP, by Hoeffding on an interval of length $\varepsilon$ | Within 0.4% at $\varepsilon = 0.5$ on binary pairs |
| To $(\varepsilon, \delta)$, simple | $\varepsilon = \rho + 2\sqrt{\rho\ell}$ at $\alpha^* = 1 + \sqrt{\ell/\rho}$ | 21% above exact at $\rho = 0.5$ |
| To $(\varepsilon, \delta)$, improved | $\min_\alpha[\rho\alpha + \ell/(\alpha-1) + \ln(1 - 1/\alpha) - \ln\alpha/(\alpha-1)]$ | 8% above exact at $\rho = 0.5$ |
| Calibration | $\rho = (\sqrt{\varepsilon + \ell} - \sqrt\ell)^2$, $\sigma = 1/\sqrt{2\rho}$ | Via the simple conversion |
| Group of $m$ | $m^2\rho$ | Exact for Gaussians |
| Approximate zCDP | $\rho$ adds, $\delta_a$ adds; $(\varepsilon(\rho, \delta'), \delta_a + \delta')$-DP | — |
| Subsampled Gaussian | Smallest $\rho = 1/(2\sigma^2)$: no amplification | Exact |
| $(\rho, \omega)$-tCDP | $D_\alpha \le \rho\alpha$ for $\alpha \in (1, \omega)$; compose to $(\sum\rho_i, \min\omega_i)$; convert by @eq-cdp-tcdp-conv | Matches RDP-Mironov on the running example |

: Every result of this chapter. {#tbl-cdp-cheat}

**Rules of thumb**

- Keep budgets in $\rho$, which adds; convert to $(\varepsilon, \delta)$ once, at the end, with the improved conversion.
- One Gaussian step with $\sigma$ costs $\rho = 1/(2\sigma^2)$: $\sigma = 1$ is $\rho = 0.5$, about $\varepsilon = 4.7$ at $\delta = 10^{-5}$.
- A pure $\varepsilon$-DP step costs $\varepsilon^2/2$, or $\varepsilon\tanh(\varepsilon/2)$ with the sharp bound; an exponential-mechanism step costs $\varepsilon^2/8$. For $\varepsilon \ge 2$ the line $\alpha\varepsilon^2/2$ is above the plain $\varepsilon$ at every order, and the sharp line crosses it soon after $\alpha = 1$; keep the RDP curve or the plain $\varepsilon$.
- Never account a subsampled mechanism in zCDP. Use RDP, tCDP with a small $\omega$, or a privacy loss distribution.
- zCDP's group privacy is quadratic in the group size, exactly as for the Gaussian.

**Practice.** Exercises for this chapter, each with a worked solution, are in [the exercise book](solutions/03-concentrated-dp.md).
