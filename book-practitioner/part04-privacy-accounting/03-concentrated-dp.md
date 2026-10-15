# Concentrated differential privacy

## Why this matters

Suppose you run a pipeline of Gaussian releases, such as noisy counts, the answers of a query system or the noisy marginals behind a synthetic dataset. You would like to keep its differential privacy (DP) budget as a single number that you add to after each release. [Rényi DP](01-renyi-dp.md) describes each mechanism by a curve instead, its **Rényi differential privacy (RDP) curve**. At each order $\alpha > 1$, the curve's value $\varepsilon(\alpha)$ is the largest Rényi divergence of that order between the mechanism's outputs on two neighbouring datasets. Neighbouring datasets differ in one person's record. Larger orders put more weight on the rare outputs that one dataset makes much more likely than the other.

Curves add when you run mechanisms one after another, but a curve is a lot to carry around, and for the Gaussian mechanism it is more than you need. Write $\sigma$ for its noise multiplier, the standard deviation of the noise in units of the query's sensitivity. Its RDP curve is then the straight line $\alpha/(2\sigma^2)$, and a line through the origin is fixed by its slope. So for Gaussian noise one number already carries the whole curve. This chapter asks when one number can describe an RDP curve, and what that number buys you.

The answer is **zero-concentrated differential privacy (zCDP)**, the most widely used form of concentrated differential privacy (CDP). It describes a mechanism by the slope $\rho$ of a line through the origin that lies above its RDP curve, and three things follow from that choice. First, lines add to a line, so composing steps adds their slopes. Second, the privacy loss, which measures how much more likely an output is under one dataset than under the other, has tails no heavier than a normal variable's. That property gives the definition its name. Third, there is a closed-form conversion to the usual guarantee, with a privacy parameter $\varepsilon$ and a slack $\delta$.

Everything else in the chapter is about how well a line fits a mechanism's curve. For the Gaussian the fit is exact, while for pure DP mechanisms with a large $\varepsilon$ it is poor. For the subsampled steps of differentially private stochastic gradient descent (SGD), DP-SGD for short, it is so poor that zCDP throws away all the privacy that subsampling gained. Truncated CDP repairs this by asking for the line only on small orders.

## The definition of zCDP {#sec-cdp-definition}

Start from the quantity that RDP measures. Run a mechanism $M$ on two neighbouring datasets, called $D$ and $D'$. The **privacy loss** of an output is the log of how much more likely that output is when the data are $D$ than when they are $D'$. If you draw the output from the mechanism run on $D$, the loss becomes a random variable, written $L$. Write $D_\alpha$ for the Rényi divergence of order $\alpha$, whose subscript sets it apart from the dataset $D$. [Rényi DP](01-renyi-dp.md) shows that, up to a factor, it is the log of the average of $e^{(\alpha - 1)L}$, which is called an exponential moment of the loss:
$$(\alpha - 1)\,D_\alpha\big(M(D)\,\|\,M(D')\big) = \ln\mathbb E\big[e^{(\alpha - 1)L}\big].$$
So an RDP curve is a list of bounds on the exponential moments of the privacy loss, one bound for each order.

zCDP asks that this whole list be bounded by a line. A mechanism $M$ is **$\rho$-zCDP** if its RDP curve lies under the line of slope $\rho$. In symbols, for all neighbouring datasets $D \simeq D'$ and all orders $\alpha > 1$,
$$D_\alpha\big(M(D)\,\|\,M(D')\big) \le \rho\alpha.$$ {#eq-cdp-def}

Two properties of RDP curves carry over to the line at once. First, RDP curves add when you run mechanisms one after another, even when each mechanism is chosen after seeing the earlier outputs. A sum of lines through the origin is again such a line, whose slope is the sum of the slopes. So a $\rho_1$-zCDP step followed by a $\rho_2$-zCDP step is $(\rho_1 + \rho_2)$-zCDP. Second, processing an output can only make two distributions harder to tell apart, so it cannot raise a Rényi divergence. Any function of a $\rho$-zCDP output is therefore still $\rho$-zCDP.

The Gaussian mechanism is the case the definition was built for. It adds normal noise with standard deviation $\sigma$ to a query whose $\ell_2$ sensitivity is $\Delta$, the largest Euclidean distance between the query's values on neighbouring datasets. Its RDP curve is exactly the line $\alpha\Delta^2/(2\sigma^2)$, so the mechanism is $\Delta^2/(2\sigma^2)$-zCDP and the line loses nothing. With the noise multiplier convention, which measures $\sigma$ in units of the sensitivity, the cost of one step is
$$\rho = \frac{1}{2\sigma^2}.$$
With $\sigma = 1$, one Gaussian step is $0.5$-zCDP, and since composition adds, ten such steps cost ten times as much. Doubling the noise multiplier to $\sigma = 2$ cuts the cost to a quarter, $0.125$. That is because $\rho$ grows with the square of the shift between the query's values on neighbouring datasets, measured in units of the noise.

The name comes from what the line says about the privacy loss. For the Gaussian mechanism the loss is itself normal. It is the log of the ratio of two normal densities with the same variance. That log is a difference of two squares in the output whose squared terms cancel, so the loss is a linear function of the output. A linear function of a normal variable is normal, and working out its coefficients shows that with sensitivity 1 the loss has mean $\rho$ and variance $2\rho$. A general $\rho$-zCDP mechanism is one whose privacy loss is, in every exponential moment, no worse than this normal variable.

Bounds on exponential moments limit how often a variable can be large. That is the idea behind the Chernoff bound of [Concentration inequalities](../part01-prerequisites/02-concentration.md), which applies Markov's inequality to an exponential and then picks the best exponent. Here each order gives one exponent, and so one bound on the chance that the loss exceeds a given threshold. The best order depends on the threshold, and choosing it gives a tail bound. Write $\lambda$ for how far the threshold lies above $\rho$. For every $\lambda > 0$, the chance that the loss exceeds $\rho + \lambda$ is at most
$$\Pr[L > \rho + \lambda] \le e^{-\lambda^2/(4\rho)}.$$ {#eq-cdp-tail}
The right side is what the same argument gives for a normal variable with mean $\rho$ and variance $2\rho$. So the privacy loss is concentrated, in the sense that it rarely rises far above $\rho$.

The tail bound is close to the truth in its rate but not in its constant. Take the Gaussian mechanism with $\sigma = 1$, so that $\rho = 0.5$. Its privacy loss is exactly normal, so you can compare the bound with the exact tail (@fig-cdp-privacy-loss, @tbl-cdp-tail). Both fall at the same rate, but the bound misses an extra factor, which shrinks like $1/\lambda$, that the normal tail carries in front of the exponential. At $\lambda = 4$, for example, the exact tail is only 0.094 times the bound (@tbl-cdp-tail). This slack is one reason why the simple conversion of @sec-cdp-conversion is loose.

![The privacy loss of the Gaussian mechanism with $\sigma = 1$. Left: a histogram of the privacy loss $L$ from simulation against the normal density with mean $\rho = 0.5$ and variance $2\rho = 1$. Right: the exact upper tail $\Pr[L > \rho + \lambda]$ against the zCDP bound $e^{-\lambda^2/(4\rho)}$, on a log scale.](figures/concentrated_dp_privacy_loss.svg){#fig-cdp-privacy-loss}

| $\lambda$ | Exact $\Pr[L > \rho + \lambda]$ | Simulated | zCDP bound $e^{-\lambda^2/(4\rho)}$ |
| --- | --- | --- | --- |
| 1 | $1.587 \times 10^{-1}$ | $1.586 \times 10^{-1}$ | $6.065 \times 10^{-1}$ |
| 2 | $2.275 \times 10^{-2}$ | $2.261 \times 10^{-2}$ | $1.353 \times 10^{-1}$ |
| 4 | $3.167 \times 10^{-5}$ | $2.700 \times 10^{-5}$ | $3.355 \times 10^{-4}$ |

: Upper tail of the Gaussian mechanism's privacy loss, $\sigma = 1$, $\rho = 0.5$: exact, simulated with $10^6$ draws, and the zCDP bound @eq-cdp-tail. {#tbl-cdp-tail}

The definition has a history that explains its form. Dwork and Rothblum (2016) introduced concentrated differential privacy with two parameters. One bounded the mean of the privacy loss, and the other said that the loss's deviation from its mean has tails no heavier than a centred normal variable's. Bun and Steinke (2016) call it mean-concentrated differential privacy, and they replaced that two-parameter definition with @eq-cdp-def for two reasons. First, the Rényi form inherits composition and post-processing from the Rényi divergence, whereas they showed that the two-parameter definition is not closed under post-processing. Second, every mechanism that met the original definition also meets zCDP, so nothing was lost. This book follows their one-parameter $\rho$-zCDP.

Going from a curve to a single $\rho$ does lose information. The smallest valid $\rho$ is the largest ratio $\varepsilon(\alpha)/\alpha$ over all orders, wherever it occurs, and the rest of the curve is forgotten. For the Gaussian nothing is forgotten, which is why zCDP is the accounting of choice for systems built from Gaussian mechanisms on the full data, with no subsampling. Statistical releases, query systems and synthetic data built from noisy marginals are of this kind.

The 2020 United States Census TopDown algorithm is the best-known example (Abowd et al. 2022). It measures noisy answers to many queries under zCDP, using the discrete Gaussian mechanism ([Discrete and floating-point-safe mechanisms](../part03-dp-foundations/07-discrete-mechanisms.md)). That mechanism adds integer noise shaped like a normal density, so that counts stay integers. zCDP suits this setting for three reasons. The budget is split over many queries and geographic levels, and zCDP turns that split into a sum of $\rho$'s. The Gaussian's curve is a line, so zCDP loses nothing before the final conversion. And that conversion is done once, at whatever $\delta$ the publisher chooses to report.

Two warnings follow from this example. First, $\rho$ is not $\varepsilon$. The Gaussian step with $\sigma = 1$, for example, is $0.5$-zCDP. The conversion of @sec-cdp-conversion reports that as about $\varepsilon = 4.7$ at $\delta = 10^{-5}$. Second, the accounted $\rho$ covers only what passed through the noise. The Census Bureau also released some statistics exactly, as a matter of policy. These **invariants** were never charged to $\rho$, so they sit outside the guarantee.

## Pure DP steps in a zCDP budget {#sec-cdp-pure}

A real pipeline mixes mechanisms. A Laplace count, a randomized-response bit and an exponential-mechanism selection are pure $\varepsilon$-DP. That means the probability of any output changes by at most a factor $e^\varepsilon$ between neighbouring datasets. To put such a step in the same budget as Gaussian steps, you need its $\rho$.

The guarantee bounds the privacy loss itself. The probability ratio behind the loss lies between $e^{-\varepsilon}$ and $e^\varepsilon$, so the loss, its logarithm, always lies within $\varepsilon$ of zero. A random variable confined to an interval cannot have heavy tails, because it cannot go past the ends at all. The reasoning behind Hoeffding's inequality makes this precise ([Concentration inequalities](../part01-prerequisites/02-concentration.md)). It says that the variable's exponential moments are no larger than those of a normal variable with the same mean and a variance set by the length of the interval. That variance grows with the square of the length, so a pure DP step should be zCDP with a $\rho$ that grows like $\varepsilon^2$.

Bun and Steinke (2016) show that every $\varepsilon$-DP mechanism is $(\varepsilon^2/2)$-zCDP. The line also needs the average loss to be small. As the order approaches 1, any RDP curve approaches the average privacy loss, while the line approaches $\rho$. The same bound on the interval shows that this average is at most $\varepsilon^2/2$. The line is not the whole story, though. No Rényi divergence can exceed the largest log-ratio of probabilities, which pure DP limits to $\varepsilon$. So as the order grows, the RDP curve of a pure DP step flattens out at $\varepsilon$, and it lies under both the line and this cap:
$$\varepsilon(\alpha) \le \min\Big(\varepsilon,\ \frac{\alpha\varepsilon^2}{2}\Big).$$ {#eq-cdp-pure}

The constant $\tfrac12$ can be improved by finding the worst case. **Randomized response** reports one bit truthfully with probability $e^\varepsilon/(1 + e^\varepsilon)$ and flips it otherwise. Every $\varepsilon$-DP mechanism can be produced from randomized response by post-processing ([Advanced and optimal composition](../part03-dp-foundations/04-composition.md)). For any pair of neighbouring datasets, some random map turns the two output distributions of randomized response into the mechanism's two output distributions. Post-processing cannot raise a Rényi divergence. So randomized response has the largest RDP curve of any $\varepsilon$-DP mechanism, at every order, and its curve gives the sharp constant. Every $\varepsilon$-DP mechanism is $(\varepsilon\tanh(\varepsilon/2))$-zCDP:
$$\varepsilon(\alpha) \le \alpha\,\varepsilon\tanh(\varepsilon/2).$$ {#eq-cdp-pure-sharp}

No smaller slope works for every $\varepsilon$-DP mechanism. Near order 1 the curve of randomized response approaches its average privacy loss, as every RDP curve does, and that average is exactly $\varepsilon\tanh(\varepsilon/2)$. Any line with a smaller slope therefore falls below the curve near $\alpha = 1$. For small $\varepsilon$ the sharp constant is close to $\varepsilon^2/2$, because the hyperbolic tangent of a small number is close to the number itself. Even at $\varepsilon = 1$ the sharp constant is 0.4621, only a little below the general constant. Mechanisms other than randomized response can need less. The Laplace mechanism at that same $\varepsilon$ needs only 0.3679, its average privacy loss, as an exercise in the full edition works out.

So an RDP accountant can charge each pure DP step the exact curve of randomized response. That curve lies below both the cap $\varepsilon$ and the sharp line, so it is the best charge available without knowing more about the mechanism.

The exponential mechanism, which selects one candidate from a list and favours those with higher scores, does four times better than the general bound. An $\varepsilon$-DP exponential mechanism is $(\varepsilon^2/8)$-zCDP (Cesar & Rogers 2021). The reason is its **bounded-range** property (Durfee & Rogers 2019). Its privacy loss lies in an interval of length $\varepsilon$ rather than $2\varepsilon$, and the Hoeffding-type bound grows with the square of the interval's length. For small $\varepsilon$ this bound is nearly tight. Among mechanisms with two possible outputs and this property, the worst one at $\varepsilon = 0.5$ has an RDP curve that reaches 0.9965 times the bound.

A line fits a pure DP curve well only near $\alpha = 1$, because the curve flattens and the line does not. Randomized response at $\varepsilon = 1$ shows this (@tbl-cdp-rr). At low orders its zCDP line stays close to the exact curve, within 36% up to order 2. At order 50, though, the line says 25 while the exact curve is only 0.9936. There the cap is the better bound, within 1% of the curve.

| $\alpha$ | Exact $\varepsilon(\alpha)$ | Line $\alpha\varepsilon^2/2$ | $\min(\varepsilon, \alpha\varepsilon^2/2)$ |
| --- | --- | --- | --- |
| 1.5 | 0.6273 | 0.7500 | 0.7500 |
| 2 | 0.7353 | 1.0000 | 1.0000 |
| 5 | 0.9217 | 2.5000 | 1.0000 |
| 10 | 0.9652 | 5.0000 | 1.0000 |
| 50 | 0.9936 | 25.0000 | 1.0000 |

: Randomized response at $\varepsilon = 1$: the exact RDP curve against the zCDP line and the line capped at $\varepsilon$. {#tbl-cdp-rr}

The line reaches the cap at order $2/\varepsilon$, so the larger $\varepsilon$ is, the sooner the line becomes the worse of the two bounds. At $\varepsilon = 3$ the line's slope is 4.5. Even at order 1, where the line's value equals its slope, it is therefore above the cap of 3. So the line is worse than plain $\varepsilon$ at every order (@fig-cdp-rr-line, right). The sharp line starts lower, at the average loss, but it too crosses the cap almost at once.

![The RDP curve of randomized response (black) against the zCDP line $\alpha\varepsilon^2/2$ (dashed), the sharp zCDP line $\alpha\varepsilon\tanh(\varepsilon/2)$ (dash-dotted) and the pure-DP cap $\varepsilon$ (dotted), for $\varepsilon = 1$ (left) and $\varepsilon = 3$ (right). The grey line is the average privacy loss $\varepsilon\tanh(\varepsilon/2)$, the curve's value as $\alpha \to 1$.](figures/concentrated_dp_rr_line.svg){#fig-cdp-rr-line}

So a pure DP step with a large $\varepsilon$ costs more in $\rho$ than it should. A $5$-DP step charged $\varepsilon^2/2 = 12.5$ has a line above its cap at every order. Converting that line back gives a worse $\varepsilon$ than the 5 you started with, and the sharp constant helps little. This is the usual situation in local DP, where each user randomizes their own data before sending it. Describe such steps by their $\varepsilon$ or their RDP curve, not by $\rho$.

## From zCDP to (ε, δ) {#sec-cdp-conversion}

Whatever you account in, you report $(\varepsilon, \delta)$. [From Rényi DP to (ε, δ)](02-rdp-conversions.md) turns an RDP curve into that guarantee. Each order gives a valid $\varepsilon$, and you keep the order that gives the smallest one. Mironov's conversion at order $\alpha$ adds a penalty to the curve's value $\varepsilon(\alpha)$, and the penalty shrinks as the order grows:
$$\varepsilon = \varepsilon(\alpha) + \frac{\ell}{\alpha - 1}, \qquad \ell = \ln\frac1\delta.$$

For a line, the curve's value $\rho\alpha$ grows with the order while the penalty shrinks, so the best order balances the two. That order is $\alpha^* = 1 + \sqrt{\ell/\rho}$, and putting it into the conversion gives the **simple conversion** (Bun & Steinke 2016). A $\rho$-zCDP mechanism is then $(\varepsilon, \delta)$-DP, for every $\delta$ below 1, with
$$\varepsilon = \rho + 2\sqrt{\rho\,\ell}.$$ {#eq-cdp-simple}
The same formula follows from the tail bound @eq-cdp-tail. Choosing $\lambda = 2\sqrt{\rho\ell}$ makes the tail probability in that bound exactly $\delta$. With that choice the threshold $\rho + \lambda$ equals the $\varepsilon$ above. So the privacy loss exceeds $\varepsilon$ with probability at most $\delta$, and [the previous chapter](02-rdp-conversions.md) shows that this is enough for $(\varepsilon, \delta)$-DP.

Mironov's conversion bounds the chance of a large privacy loss with Markov's inequality, applied to an exponential of the loss, and that step is loose. The **improved conversion** of Canonne, Kamath and Steinke (2020) removes some of this slack. It adds two negative correction terms at each order. The minimum over orders then has no closed form and is found numerically:
$$\varepsilon = \min_{\alpha > 1}\left[\rho\alpha + \frac{\ell}{\alpha - 1} + \ln\Big(1 - \frac1\alpha\Big) - \frac{\ln\alpha}{\alpha - 1}\right].$$ {#eq-cdp-improved}

How much do the two conversions lose? For the Gaussian mechanism the line is the exact curve, so any gap between the reported $\varepsilon$ and the true one comes from the conversion alone. The true trade-off between $\varepsilon$ and $\delta$ is also known exactly. Dividing the output by $\sigma$ changes nothing about privacy, because you could multiply it back. After that division the two output distributions are normal with variance 1, and the distance between their means, written $\mu$, is $\sqrt{2\rho}$. This $\mu$ is the parameter of Gaussian differential privacy in [f-DP and Gaussian differential privacy](04-gaussian-dp.md).

The **privacy profile** $\delta(\varepsilon)$ is the smallest $\delta$ for which a mechanism is $(\varepsilon, \delta)$-DP. For the Gaussian it can be written with the standard normal distribution function $\Phi$ ([The Laplace and Gaussian mechanisms](../part03-dp-foundations/03-laplace-gaussian.md)):
$$\delta(\varepsilon) = \Phi\Big(\frac\mu2 - \frac\varepsilon\mu\Big) - e^\varepsilon\,\Phi\Big(-\frac\mu2 - \frac\varepsilon\mu\Big).$$ {#eq-cdp-gauss-exact}
Solving it for $\varepsilon$ at a given $\delta$ gives the exact value that both conversions approximate.

Take again the Gaussian step with $\sigma = 1$, so that $\rho = 0.5$. Its exact $\varepsilon$ at $\delta = 10^{-5}$ is 4.377, and both conversions overshoot it. The simple formula gives 5.299, which is 21% too large. The improved conversion comes closer but still overshoots, at 4.728, or 8% too large. Both gaps shrink in relative terms as $\rho$ grows (@tbl-cdp-conversion, @fig-cdp-conversion).

| $\rho$ | Simple @eq-cdp-simple | Improved @eq-cdp-improved | Exact, Gaussian with the same $\rho$ |
| --- | --- | --- | --- |
| 0.01 | 0.689 | 0.546 | 0.497 |
| 0.1 | 2.246 | 1.914 | 1.760 |
| 0.5 | 5.299 | 4.728 | 4.377 |
| 1 | 7.786 | 7.077 | 6.573 |
| 5 | 20.174 | 19.047 | 17.857 |
| 50 | 97.985 | 96.035 | 91.817 |

: $\varepsilon$ at $\delta = 10^{-5}$ for a $\rho$-zCDP mechanism: the simple and improved conversions, and the exact $\varepsilon$ of the Gaussian mechanism with the same $\rho$. {#tbl-cdp-conversion}

![Left: $\varepsilon$ at $\delta = 10^{-5}$ for a $\rho$-zCDP mechanism, from the simple formula @eq-cdp-simple, the improved conversion @eq-cdp-improved, and the exact value for the Gaussian mechanism with $\mu = \sqrt{2\rho}$. Right: each conversion divided by the exact Gaussian value.](figures/concentrated_dp_conversion.svg){#fig-cdp-conversion}

Accountants store curves only on a fixed grid of orders, but for a line the grid hardly matters. On the fixed grid that this part's accountants use, the improved conversion agrees with a much finer grid to within 0.001 for $\rho$ up to 1.

The choice of conversion does matter. For the same Gaussian step, two systems could quote 5.299 and 4.728 only because they converted differently. A report should therefore say which conversion it used.

For calibration you need the conversion run backwards, from a target $(\varepsilon, \delta)$ to the largest $\rho$ that meets it. Solving the simple conversion for $\rho$ shows that the target is met by any
$$\rho \le \Big(\sqrt{\varepsilon + \ell} - \sqrt{\ell}\Big)^2.$$ {#eq-cdp-inverse}
For a single Gaussian step, the noise multiplier then follows from $\rho = 1/(2\sigma^2)$ as $\sigma = 1/\sqrt{2\rho}$. For example, a step that must be $(1, 10^{-5})$-DP may spend at most $\rho = 0.02082$. It therefore needs a noise multiplier of 4.901.

Because the conversion grows like the square root of $\rho$, $\varepsilon$ does not add. Suppose you have a total budget of $\rho = 1$ and have spent half of it on one query. You cannot convert each half to $\varepsilon$ and subtract, because each half alone converts to 4.728, and two of those are far more than the whole. By the improved conversion the whole run is $7.077$-DP at $\delta = 10^{-5}$, however you split it. So keep budgets in $\rho$, which adds, and convert once, at the end.

## Group privacy {#sec-cdp-group}

A household, a user with many records or a hospital contributes several records. Removing such a group changes a dataset in $m$ records rather than one, and group privacy asks how the guarantee weakens for such datasets. The two datasets are joined by a chain of $m$ neighbouring steps, each changing one record. For pure DP the cost is $m\varepsilon$, because each step multiplies probabilities by at most $e^\varepsilon$ ([Post-processing, group privacy and composition](../part03-dp-foundations/02-properties.md)).

For RDP the chain is harder to follow, because Rényi divergence has no plain triangle inequality ([Rényi DP](01-renyi-dp.md)). Bun and Steinke prove a triangle-like inequality that goes through an intermediate distribution, but at a price. Its two legs are measured at higher orders than the one you want. Since each link of the chain needs higher orders than the last, a bound known only up to some order would run out after a few links. Because zCDP bounds every order at once, it can pay that price at every link.

The result is that zCDP's slope grows with the square of the group size. Take a $\rho$-zCDP mechanism $M$ and two datasets that differ in $m$ records. Then (Bun & Steinke 2016)
$$D_\alpha\big(M(D)\,\|\,M(D')\big) \le m^2\rho\,\alpha.$$ {#eq-cdp-group}
So $M$ is $m^2\rho$-zCDP for groups of $m$. The square is the right answer, as the Gaussian shows. A group of $m$ records shifts the query's value $m$ times as far, and the Gaussian's RDP curve grows with the square of the shift. So the Gaussian mechanism meets the bound with equality.

The square also agrees with pure DP's $m\varepsilon$. An $\varepsilon$-DP step costs $\varepsilon^2/2$ in $\rho$. For a group of $m$, zCDP multiplies that cost by $m^2$, which gives exactly what an $m\varepsilon$-DP step costs. For example, take a user who contributes up to 4 records to a $0.1$-zCDP release. That user is protected at $\rho = 4^2 \times 0.1 = 1.6$.

[Rényi DP](01-renyi-dp.md) states Mironov's (2017) group bound for plain RDP, which is weaker because it gives up orders instead. Each doubling of the group halves the order the bound speaks about, and it triples $\varepsilon$. So the slope, which is $\varepsilon$ divided by the order, grows sixfold per doubling. The slope of zCDP grows only fourfold. @tbl-cdp-group compares the two for the Gaussian mechanism with $\sigma = 1$.

| Group size $m$ | zCDP $m^2\rho$ | $\varepsilon$ at $10^{-5}$ (improved) | Exact Gaussian, shift $m$ | RDP route: order, $\varepsilon$, slope |
| --- | --- | --- | --- | --- |
| 1 | 0.5 | 4.728 | 4.377 | — |
| 2 | 2 | 10.725 | 9.997 | 32, 96, 3 |
| 4 | 8 | 25.919 | 24.382 | 16, 288, 18 |
| 8 | 32 | 68.616 | 65.319 | 8, 864, 108 |

: Group privacy of the Gaussian mechanism with $\sigma = 1$. zCDP gives slope $m^2\rho$ at every order, and its conversion is within about 8% of the exact value. Mironov's RDP group bound, started from order 64, gives slope $6^c\rho$ for groups of $m = 2^c$, and only on orders up to $64/m$. {#tbl-cdp-group}

Strong group privacy is also a weakness. Bun and Steinke use it to prove lower bounds for zCDP that are as strong as those for pure DP. One consequence is that stability-based algorithms, which are $(\varepsilon, \delta)$-DP with sample sizes that do not depend on the size of the data domain, cannot be zCDP. The next two sections both escape these lower bounds by weakening group privacy for large groups.

## Approximate zCDP {#sec-cdp-approx}

Some useful mechanisms are $(\varepsilon, \delta)$-DP but cannot be zCDP, by the lower bounds just mentioned. Propose-test-release and stability-based histograms are two examples, and both work the same way. They check privately that the data are far from any dataset where the answer would change, and they release an exact answer when that is so. They fail only with probability about $\delta$. You still want to put them in the same budget as Gaussian steps, and Bun and Steinke's answer is to let zCDP fail on a rare event.

Write $\delta_a$ for the probability of that event. A mechanism is **$\delta_a$-approximately $\rho$-zCDP** if it becomes $\rho$-zCDP once you throw away a bad event. Throwing it away means looking at the outputs only when that event did not happen. For each pair of neighbouring datasets, that event may have probability at most $\delta_a$ under each of the two datasets. With $\delta_a = 0$ nothing is thrown away, and the definition is plain $\rho$-zCDP.

Approximate zCDP keeps one ledger with two columns. The $\rho$ column adds, as in zCDP. The $\delta_a$ column adds as well, like $\delta$ in basic composition. The reason is the union bound, since the good events of all the steps hold together except with probability at most the sum of their failure probabilities. An $(\varepsilon, \delta)$-DP step enters the ledger as $\rho = \varepsilon^2/2$ and $\delta_a = \delta$ (Bun & Steinke 2016). That is because, on any pair of neighbouring datasets, an $(\varepsilon, \delta)$-DP mechanism behaves like an $\varepsilon$-DP one except with probability $\delta$. @sec-cdp-pure charges an $\varepsilon$-DP step $\varepsilon^2/2$.

At the end, you convert the $\rho$ column as usual and add the $\delta_a$ column to the $\delta$ of that conversion. Write $\delta'$ for the $\delta$ at which you convert the $\rho$ column. Let $\varepsilon$ be the result of any zCDP conversion at $\delta'$. Then a $\delta_a$-approximately $\rho$-zCDP mechanism is
$$\big(\varepsilon,\ \delta_a + (1 - \delta_a)\,\delta'\big)\text{-DP}.$$
On the good event the usual conversion applies, and the factor $1 - \delta_a$ appears because the proof weighs that event by its probability. The bad event can add at most its own probability to the probability of any set of outputs.

Take a pipeline that runs ten stability-based steps, each $(0.5, 10^{-7})$-DP, alongside ten Gaussian steps with $\sigma = 1$. Each stability step enters the $\rho$ column at $0.5^2/2 = 0.125$, and each Gaussian step at 0.5. So the $\rho$ column totals 6.25, and the $\delta_a$ column totals $1.0000 \times 10^{-6}$. Converting the $\rho$ column with the improved conversion at $\delta' = 10^{-5} - \delta_a$ gives $\varepsilon = 22.100$. The total $\delta$ is then $10^{-5}$.

Without approximate zCDP, you would convert the Gaussian steps alone and add the stability steps by basic composition. The Gaussian steps alone give $\varepsilon = 19.119$ at $\delta = 9 \times 10^{-6}$. Basic composition then adds the stability steps' $\varepsilon = 5$ and $\delta = 10^{-6}$ on top. That brings the total to 24.119. The unified ledger saves about 2, because each stability step costs only $\varepsilon^2/2$ in the $\rho$ column, rather than its full $\varepsilon$.

Approximate zCDP is worth using only when some steps really are $(\varepsilon, \delta)$-DP and not zCDP. A step that is already zCDP gains nothing from the $\delta_a$ column, and putting it there would only waste $\delta$.

## Subsampling, and truncated CDP {#sec-cdp-tcdp}

DP-SGD, this part's running example, runs each step on a Poisson sample of the data. Each record enters the batch independently with probability $q$, the sampling rate. Subsampling makes a step much more private than the Gaussian mechanism it contains, because a record is usually not in the batch ([Amplification by subsampling](05-subsampling.md)). The question is whether zCDP sees this gain.

For the running example, with $q = 0.01$ and $\sigma = 1$, the RDP curve of one step is tiny at small orders. Up to order 8 it is almost a line, with a slope of about $10^{-4}$ (@fig-cdp-subsampled). Between orders 8 and 16, however, the curve jumps by more than three orders of magnitude. The jump comes from the way high orders weigh the rare outputs that are most revealing. For a subsampled step, those outputs come from the event that the record was in the batch, and on that event the step pays the full Gaussian loss. At large orders that event dominates, and the curve's slope approaches the slope $1/(2\sigma^2)$ of the Gaussian without subsampling.

![The RDP curve of one Poisson-subsampled Gaussian step with $q = 0.01$, $\sigma = 1$, on integer orders (black), against the best zCDP line $\rho\alpha$ with $\rho = 1/(2\sigma^2) = 0.5$ (red dashed), which is the unsubsampled Gaussian's line $\alpha/2$, and the tCDP line on orders up to $\omega = 8$ (blue). Both axes are logarithmic.](figures/concentrated_dp_subsampled.svg){#fig-cdp-subsampled}

A line through the origin that lies above the whole curve must be at least as steep as the curve's slope at infinity. So for the Poisson-subsampled Gaussian, at any sampling rate, the smallest valid $\rho$ is $1/(2\sigma^2)$, exactly what the step would cost without subsampling. Subsampling never makes a step less private, so this $\rho$ is also enough. A zCDP accountant therefore throws away all of the amplification.

Over a training run the cost is enormous. Write $k$ for the number of steps, and take a run of $k = 10{,}000$ steps reported at $\delta = 10^{-5}$. Because zCDP charges each step $\rho = 0.5$, the whole run costs $\rho = 5000$. That converts to $\varepsilon = 5479.9$ by the simple formula, or 5475.8 by the improved one. RDP accounting of the same run, on the integer orders of the grid that this part's accountants use, gives only 7.469 with Mironov's conversion and 6.719 with the improved one.

RDP does so much better because the conversion only ever uses small orders. The composed curve is $k\varepsilon(\alpha)$, and the conversion picks the order where $k\varepsilon(\alpha) + \ell/(\alpha - 1)$ is smallest. Multiplied by 10,000 steps, the curve is no longer small, even below the jump. At order 2 it is already 1.7181, and it keeps growing with the order up to the jump. Meanwhile the penalty $\ell/(\alpha - 1)$ keeps shrinking, so the two balance early, and the best integer order is 4, for both conversions. The large orders that set the zCDP slope are never used.

**Truncated concentrated differential privacy (tCDP)** keeps one number per step but asks for the line only on orders below a cut-off $\omega$. Bun, Dwork, Rothblum and Steinke (2018) introduced it. A mechanism is $(\rho, \omega)$-tCDP if, for all neighbouring datasets and all orders $\alpha$ between 1 and the cut-off,
$$D_\alpha\big(M(D)\,\|\,M(D')\big) \le \rho\alpha.$$
Plain zCDP is the case of an infinite cut-off. Steps compose as in zCDP, but only on the orders that both cover. So a $(\rho_1, \omega_1)$-tCDP step followed by a $(\rho_2, \omega_2)$-tCDP step is $(\rho_1 + \rho_2,\ \min(\omega_1, \omega_2))$-tCDP.

The conversion to $(\varepsilon, \delta)$ is the simple zCDP conversion whenever its best order $\alpha^*$ lies below the cut-off. Otherwise the bound says nothing at $\alpha^*$. The conversion's value keeps falling as the order rises towards $\alpha^*$, so the best order the bound allows is $\omega$ itself. The condition in the first branch below is just $\alpha^* \le \omega$, rewritten:
$$\varepsilon = \begin{cases}\rho + 2\sqrt{\rho\,\ell} & \text{if } \ell \le (\omega - 1)^2\rho,\\[2pt] \rho\omega + \dfrac{\ell}{\omega - 1} & \text{otherwise.}\end{cases}$$ {#eq-cdp-tcdp-conv}

@tbl-cdp-tcdp applies tCDP to the running example. It takes the per-step slope on integer orders up to each cut-off, adds the slopes over the 10,000 steps, and converts the total.

| $\omega$ | $\rho$ per step | $\rho$ after $10^4$ steps | $\varepsilon$ at $10^{-5}$ |
| --- | --- | --- | --- |
| 2 | $8.5907 \times 10^{-5}$ | 0.8591 | 13.231 |
| 3 | $8.8213 \times 10^{-5}$ | 0.8821 | 8.403 |
| 4 | $9.0789 \times 10^{-5}$ | 0.9079 | 7.469 |
| 5 | $9.3733 \times 10^{-5}$ | 0.9373 | 7.507 |
| 8 | $1.1171 \times 10^{-4}$ | 1.1171 | 8.289 |
| 10 | $3.8270 \times 10^{-3}$ | 38.2704 | 80.252 |
| 16 | $1.9299 \times 10^{-1}$ | 1929.9067 | 2228.027 |

: tCDP accounting of the running example, $q = 0.01$, $\sigma = 1$, $k = 10{,}000$, $\delta = 10^{-5}$, for several cut-offs $\omega$. Slopes are taken on integer orders up to $\omega$. {#tbl-cdp-tcdp}

The best integer cut-off is $\omega = 4$. It gives 7.469, the same as RDP with Mironov's conversion, because both end up converting at order 4. A smaller cut-off lowers the slope only a little but forbids the orders that the conversion needs. A larger one must cover more of the curve, and once $\omega$ passes the jump near order 10, tCDP collapses towards zCDP.

Unlike zCDP, tCDP is amplified by subsampling. Bun, Dwork, Rothblum and Steinke prove a general theorem for a uniformly random subset of a fixed fraction $s$ of the records, drawn without replacement. It requires $\rho$ and $s$ to be at most 0.1, and the mechanism must also satisfy the tCDP bound up to order $\ln(1/s)/(2\rho)$, twice the cut-off it gets back. That is because the proof controls the higher-order terms of an expansion only up to there. A further, mild condition ties $\ln(1/s)$ to $\rho$. Under those conditions the subsampled mechanism is
$$\Big(13s^2\rho,\ \frac{\ln(1/s)}{4\rho}\Big)\text{-tCDP}.$$
The factor $s^2$ is the amplification.

The theorem holds for every tCDP mechanism, so its constant 13 is generous. Take a Gaussian step with $\sigma = \sqrt5$, so that $\rho = 0.1$. Each step runs it on a sample of 1% of the records. The theorem gives a per-step $\rho$ of $1.300 \times 10^{-4}$, while the subsampled Gaussian's own curve needs only $1.128 \times 10^{-5}$ on the integer orders up to the theorem's cut-off. After 10,000 steps the theorem's parameters convert to $\varepsilon = 9.037$. RDP accounting of the same run, with the improved conversion, gives only 2.046. This comparison is only indicative. The theorem assumes fixed-size batches and neighbours that replace one record, while the RDP curve assumes Poisson sampling and neighbours that add or remove one. [Amplification by subsampling](05-subsampling.md) explains why the two must not be mixed in an accountant.

So never account a subsampled mechanism in zCDP. Use RDP, tCDP with a small cut-off, or a privacy loss distribution. A privacy loss distribution tracks the whole distribution of the privacy loss numerically ([Numerical accounting with privacy loss distributions](06-numerical-accounting.md)). RDP keeps the whole curve, so it does at least as well as tCDP with any cut-off. If a pipeline mixes Gaussian releases with a DP-SGD run, keep all of it in RDP. A $\rho$-zCDP step is just the RDP curve $\rho\alpha$, so it joins an RDP ledger without loss.

## Code {#sec-cdp-code}

The whole zCDP ledger is a few lines of code. You charge each Gaussian step its $\rho$, add the charges, and convert once. The improved conversion below is minimized over a fine grid of 20,001 orders, the one the full edition uses for its fine-grid values.

```python
from math import log, sqrt


def zcdp_gaussian(sigma: float, sensitivity: float = 1.0) -> float:
    """rho of one Gaussian step; compose steps by adding their rho."""
    return sensitivity**2 / (2 * sigma**2)


def zcdp_to_eps_simple(rho: float, delta: float) -> float:
    return rho + 2 * sqrt(rho * log(1 / delta))


def zcdp_to_eps_improved(rho: float, delta: float) -> float:
    """Improved conversion, minimized over 20,001 log-spaced orders in (1, 1 + 1e4]."""
    best = float("inf")
    for i in range(20001):
        alpha = 1 + 10 ** (-4 + 8 * i / 20000)
        eps = rho * alpha + log(1 / delta) / (alpha - 1) + log(1 - 1 / alpha) - log(alpha) / (alpha - 1)
        best = min(best, eps)
    return best


def rho_for_target(eps: float, delta: float) -> float:
    """Largest rho whose simple conversion stays within eps at delta."""
    return (sqrt(eps + log(1 / delta)) - sqrt(log(1 / delta))) ** 2


rho = 10 * zcdp_gaussian(sigma=1.0)  # ten Gaussian steps with sigma = 1
print(f"{rho:.1f}")  # 5.0
print(f"{zcdp_to_eps_simple(rho, 1e-5):.3f}")  # 20.174
print(f"{zcdp_to_eps_improved(rho, 1e-5):.3f}")  # 19.047
r = rho_for_target(1.0, 1e-5)
print(f"{r:.5f} {1 / sqrt(2 * r):.3f}")  # 0.02082 4.901
```

The full edition's script, [concentrated_dp.py](../../code/part04-privacy-accounting/concentrated_dp.py), computes every number in this chapter, including the subsampled curve and the tCDP values. Run it from the repository's root with

```bash
uv run python code/part04-privacy-accounting/concentrated_dp.py
```

## Cheat sheet

Here $\ell$ stands for the logarithm of $1/\delta$.

| Result | Formula | Notes |
| --- | --- | --- |
| $\rho$-zCDP | $D_\alpha \le \rho\alpha$ for all $\alpha > 1$ | RDP restricted to a line |
| Privacy-loss tail | $\Pr[L > \rho + \lambda] \le e^{-\lambda^2/(4\rho)}$ | Right rate, misses the normal tail's polynomial factor |
| Gaussian mechanism | $\rho = \Delta^2/(2\sigma^2)$; privacy loss normal with mean $\rho$, variance $2\rho$ | Exact: the line is the curve |
| Composition, post-processing | $\rho$'s add; processing keeps $\rho$ | Also for adaptively chosen steps |
| Pure DP step | $\varepsilon^2/2$, sharply $\varepsilon\tanh(\varepsilon/2)$; curve $\le \min(\varepsilon, \alpha\varepsilon^2/2)$ | Poor fit for large $\varepsilon$ |
| Exponential mechanism | $\varepsilon^2/8$ | From the bounded-range property |
| To $(\varepsilon, \delta)$, simple | $\rho + 2\sqrt{\rho\ell}$, at order $1 + \sqrt{\ell/\rho}$ | 21% above exact at $\rho = 0.5$ |
| To $(\varepsilon, \delta)$, improved | $\min_\alpha[\rho\alpha + \ell/(\alpha-1) + \ln(1 - 1/\alpha) - \ln\alpha/(\alpha-1)]$ | 8% above exact at $\rho = 0.5$ |
| Calibration | $\rho = (\sqrt{\varepsilon + \ell} - \sqrt\ell)^2$, $\sigma = 1/\sqrt{2\rho}$ | Through the simple conversion |
| Group of $m$ | $m^2\rho$ | Exact for the Gaussian |
| Approximate zCDP | $\rho$ adds, $\delta_a$ adds; $(\varepsilon, \delta_a + (1 - \delta_a)\delta')$-DP | For steps that are only $(\varepsilon, \delta)$-DP |
| Subsampled Gaussian | $\rho = 1/(2\sigma^2)$ whatever $q$ | No amplification in zCDP |
| $(\rho, \omega)$-tCDP | $D_\alpha \le \rho\alpha$ for $\alpha < \omega$; compose to $(\sum_i\rho_i, \min_i\omega_i)$ | Matches RDP with Mironov's conversion on the running example |

: The results of this chapter. {#tbl-cdp-cheat}

**Rules of thumb**

- Keep budgets in $\rho$, which adds, and convert to $(\varepsilon, \delta)$ once, at the end, with the improved conversion.
- One Gaussian step with $\sigma = 1$ costs $\rho = 0.5$. That is about $\varepsilon = 4.7$ at $\delta = 10^{-5}$.
- Charge a pure DP step with a small $\varepsilon$ either $\varepsilon^2/2$ or the sharp $\varepsilon\tanh(\varepsilon/2)$. Charge an exponential-mechanism step $\varepsilon^2/8$. For $\varepsilon$ of 2 or more, keep the step's $\varepsilon$ or its RDP curve, because the slope $\varepsilon^2/2$ is then at least the cap.
- Never account a subsampled mechanism, such as a DP-SGD step, in zCDP. Use RDP, tCDP with a small cut-off, or a privacy loss distribution.
- zCDP's group privacy is quadratic in the group size, exactly as for the Gaussian.

## Further reading

- Bun & Steinke, "Concentrated Differential Privacy: Simplifications, Extensions, and Lower Bounds" (TCC 2016): the definition, the Gaussian and pure-DP results, the conversion, group privacy, approximate zCDP and the lower bounds. Sections 1–3 are the core.
- Canonne, Kamath & Steinke, "The Discrete Gaussian for Differential Privacy" (NeurIPS 2020): the discrete Gaussian with its zCDP guarantee, and the improved conversion with a stable way to compute it.
- Bun, Dwork, Rothblum & Steinke, "Composable and Versatile Privacy via Truncated CDP" (STOC 2018): truncated CDP, its conversion and its subsampling theorem.
- Abowd et al., "The 2020 Census Disclosure Avoidance System TopDown Algorithm" (Harvard Data Science Review, 2022): zCDP accounting at national scale, and the role of invariants.

**Derivations.** The full edition's chapter derives each result step by step, or links to the chapter that does so, as for the improved conversion and for Mironov's group bound for RDP. It cites a few results and explains each in a sentence. These are the failure of the two-parameter definition under post-processing, the lower bounds that rule out stability-based algorithms, the bounded-range property of the exponential mechanism, and the conversion of $(\varepsilon, \delta)$-DP to approximate zCDP. They also include the composition of approximate zCDP and of truncated CDP. The chapter outlines the proof of truncated CDP's subsampling theorem, and its exercises work out the Laplace mechanism's value of 0.3679.
