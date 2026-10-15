# Rényi differential privacy

## Why this matters

Suppose you train a model with 1000 steps of full-batch noisy gradient descent. Each step computes a quantity whose **sensitivity** is 1, which means that changing one person's record moves it by at most 1. The step then releases that quantity with Gaussian noise added. The standard deviation of the noise is a multiple of the sensitivity, and that multiple is the **noise multiplier**, written $\sigma$. Here $\sigma = 20$. You want the total differential privacy (DP) guarantee of the whole run.

Recall what such a guarantee says ([The definition of differential privacy](../part03-dp-foundations/01-definition.md)). Two datasets are neighbours when they differ in one person's record. A mechanism is $(\varepsilon, \delta)$-DP, for a privacy level $\varepsilon$ and a small slack $\delta$, when the following holds for any two neighbours and any set of outputs. The probability of that set under one dataset is at most $e^\varepsilon$ times its probability under the other, plus $\delta$. Here the slack is $\delta = 10^{-5}$.

Part 3 gives one way to answer the question. You first turn each step into a single guarantee, written $(\varepsilon_0, \delta_0)$. Then you apply advanced composition ([Composition](../part03-dp-foundations/04-composition.md)), which says that a run of $k$ such steps costs roughly $\varepsilon_0\sqrt k$ rather than $k\varepsilon_0$. Advanced composition spends part of $\delta$ on the steps' slacks and the rest on a bound for the tail of the summed privacy loss. With the best split between the two, it gives $\varepsilon = 68.01$. The true answer is $\varepsilon = 7.51$, so the bound is about nine times too large. You can compute the true answer here because 1000 Gaussian steps behave exactly like one Gaussian step with noise multiplier $\sigma/\sqrt{1000}$, as this chapter will show.

The loss happens in the very first move. A Gaussian step does not have one $(\varepsilon_0, \delta_0)$ pair. For each $\varepsilon_0$ there is a smallest $\delta_0$ that works, so the step has a whole curve of valid pairs, its **privacy curve**. Collapsing that curve to one point throws away how the step behaves everywhere else. Composition needs exactly that information, because how much a run of steps leaks depends on how each step behaves at every level of $\varepsilon_0$, not at one. Once it is gone, no composition theorem can recover it.

**Rényi differential privacy (RDP)** keeps a whole curve instead of one point, and it chooses a curve that is easy to compose. Its curves add up over the steps of a run, with no $\delta$ spent along the way. You then convert the total into an $(\varepsilon, \delta)$ guarantee once, at the end. For the 1000 Gaussian steps this gives $\varepsilon = 8.84$. That is 18% above the true value, instead of nine times above it. The same bookkeeping, under the name *moments accountant*, is what made differentially private deep learning practical.

## The RDP curve {#sec-rdp-definition}

Every guarantee in this chapter is a statement about one random quantity, the privacy loss. Write $M$ for a mechanism and $D$ and $D'$ for two neighbouring datasets. Running the mechanism on $D$ gives a distribution of outputs, written $P$, and running it on $D'$ gives another, written $Q$.

Take an output, written $y$, and write $p(y)$ for its probability or density under $P$. Its probability under $Q$ is written $q(y)$ in the same way. The ratio of the two is the **likelihood ratio** of $y$, and its logarithm is the **privacy loss**,
$$L(y) = \ln\frac{p(y)}{q(y)}.$$
The loss is positive when $y$ is more likely under $D$ than under $D'$, and negative when it is less likely ([Properties of differential privacy](../part03-dp-foundations/02-properties.md)). The output is random, so the loss of an output drawn from $M(D)$ is a random variable.

Pure $\varepsilon$-DP bounds this loss at every output, which is a statement about its largest value. The weaker $(\varepsilon, \delta)$-DP allows the bound to be broken on outputs that carry a little probability, measured by $\delta$. So both describe the distribution of the loss by a single point far out in its upper tail.

RDP describes it differently, by an exponential moment, which is the average of $e^{\lambda L}$ for some positive number $\lambda$. An exponential weights large losses much more heavily than small ones, and the larger $\lambda$ is, the more heavily it weights them. So a small $\lambda$ looks at the typical loss, and a large $\lambda$ looks at the worst case.

The **Rényi divergence** of [Divergences](../part01-prerequisites/03-divergences.md) packages such a moment. It has a parameter called the **order**, written $\alpha$, which is any number above 1, and it uses the moment with $\lambda = \alpha - 1$. The divergence of order $\alpha$ takes the logarithm of that moment and divides it by $\alpha - 1$. Between $P$ and $Q$ it is
$$D_\alpha(P\|Q) = \frac{1}{\alpha - 1}\ln\mathbb E_P\big[e^{(\alpha - 1)L}\big] = \frac{1}{\alpha - 1}\ln\int p^\alpha q^{1-\alpha}\,dy.$$
The second form is the one you compute with, and it is the same average written as an integral. The quantity $e^{(\alpha - 1)L}$ is the likelihood ratio $p/q$ raised to the power $\alpha - 1$. Averaging under $P$ multiplies it by $p$ before integrating, which gives $p^\alpha q^{1-\alpha}$.

A mechanism is **$(\alpha, \varepsilon)$-RDP** if this divergence is at most $\varepsilon$ for every pair of neighbours,
$$D_\alpha\big(M(D)\,\|\,M(D')\big) \le \varepsilon.$$
At each order, the smallest $\varepsilon$ that works is the divergence of the worst pair of neighbours. Recorded as a function of the order, it is the mechanism's **RDP curve**, written $\varepsilon(\alpha)$.

The curve inherits three properties from the Rényi divergence, and they tie its two ends to quantities you already know. First, it never decreases as the order grows, because a higher order leans harder on the large losses. Second, as the order grows without limit, the curve tends to the largest loss of any output, so a mechanism is $\varepsilon$-DP exactly when $\varepsilon(\infty) \le \varepsilon$. Together these say that an $\varepsilon$-DP mechanism is $(\alpha, \varepsilon)$-RDP at every order. Third, as the order falls towards 1, the curve tends to a limit that is at least the largest Kullback–Leibler (KL) divergence between neighbours. The KL divergence is the average loss of an output drawn from $P$.

So the curve sits between the average case and the worst case. Think of a mechanism whose loss is usually small but occasionally large. Its KL divergence is small, because the rare large losses barely move the average. Its high-order divergences are large, because a high exponential moment is dominated by exactly those rare losses. Such a mechanism is caught at large orders, and pure DP, at order infinity, is the most demanding point of the curve.

Orders below 1 exist as divergences, but RDP does not use them. At the end of this chapter, the curve is turned into an $(\varepsilon, \delta)$ guarantee by bounding how often the loss is large, using the exponential moment with $\lambda = \alpha - 1$. That bound needs $\lambda$ to be positive, since only then does $e^{\lambda L}$ grow with the loss, so it needs orders above 1.

**Randomized response** shows the curve on numbers. It reports one person's bit truthfully with probability $p$, the keep-probability, which is above one half, and flips it otherwise. Written alone, $p$ means this keep-probability, while $p(y)$, with an argument, is still the probability of an output. Take $p = 0.75$. The two neighbouring inputs are the two values of the bit. So $P$ is the output distribution when the bit is 1, and $Q$ is the one when it is 0. @tbl-rdp-rr lists them.

| Output $y$ | $p(y)$ (bit is 1) | $q(y)$ (bit is 0) | Likelihood ratio | $L(y)$ | $p(y)^2/q(y)$ |
| --- | --- | --- | --- | --- | --- |
| 1 | 0.75 | 0.25 | 3 | 1.0986 | 2.2500 |
| 0 | 0.25 | 0.75 | 1/3 | −1.0986 | 0.0833 |

: Randomized response with keep-probability $p = 0.75$. The last column gives the two terms of the order-2 sum $\int p^2 q^{-1}$. {#tbl-rdp-rr}

Each output is three times as likely under one value of the bit as under the other, so the loss is always $\ln 3$ in size. The mechanism is therefore $\ln 3$-DP. At order 2, the integral in the definition becomes a sum over the two outputs of $p(y)^2/q(y)$, which is the last column. Their sum is 2.3333, and its logarithm, divided by $\alpha - 1 = 1$, gives $\varepsilon(2) = 0.8473$. The same recipe works at any order, and for a general keep-probability $p$ it gives the curve
$$\varepsilon(\alpha) = \frac{1}{\alpha - 1}\ln\Big(p^\alpha(1-p)^{1-\alpha} + (1-p)^\alpha p^{1-\alpha}\Big).$$ {#eq-rdp-rr}
The Rényi divergence is not symmetric in general, so the curve must take the worse of the two directions between neighbours. Here the choice does not matter. Swapping the two bits swaps the two rows of the table and leaves the sum unchanged, so both directions give the same curve. @tbl-rdp-rr-orders gives it at a few orders.

| $\alpha$ | 1 (KL) | 1.5 | 2 | 5 | 10 | 64 | $\infty$ |
| --- | --- | --- | --- | --- | --- | --- | --- |
| $\varepsilon(\alpha)$ | 0.5493 | 0.7340 | 0.8473 | 1.0267 | 1.0666 | 1.0940 | 1.0986 |

: The RDP curve of randomized response with $p = 0.75$. {#tbl-rdp-rr-orders}

The curve rises from the average loss to the largest loss (@fig-rdp-rr-curve). Under $P$ the loss is $\ln 3$ with probability 0.75 and $-\ln 3$ with probability 0.25, so the average loss is $0.75\ln 3 - 0.25\ln 3$. That is half of $\ln 3$, or 0.5493, while at the other end the largest loss is $\ln 3 = 1.0986$. Randomized response is already a pure DP mechanism, so here the curve says little that $\ln 3$ did not. The curve matters most for mechanisms whose largest loss is huge or infinite, such as the Gaussian mechanism.

![The RDP curve $\varepsilon(\alpha)$ of randomized response with $p = 0.75$, on a log scale in $\alpha$. It starts at the KL divergence 0.549 as $\alpha \to 1$ and rises to the pure DP level $\ln 3 = 1.099$ as $\alpha \to \infty$.](figures/renyi_dp_rr_curve.svg){#fig-rdp-rr-curve}

An RDP guarantee is a curve, not a number. A statement such as "randomized response with $p = 0.75$ is $(2, 0.8473)$-RDP" is one point of it, and it says nothing about higher orders, where the curve may keep rising. So accountants store the whole curve on a grid of orders. This part always uses the same grid of 154 orders, listed in the code at the end of the chapter. It runs from just above 1 to 512, dense near 1 and sparse at the top, and the last section shows why it needs that range.

## RDP curves of the basic mechanisms {#sec-rdp-curves}

To account for a mechanism you need its curve. A mechanism that adds noise to a query's answer produces the same noise distribution centred at two different answers on the two neighbours. Its curve is therefore the Rényi divergence between the noise distribution and a shifted copy of itself, which [Divergences](../part01-prerequisites/03-divergences.md) computed for the normal and the Laplace distributions. The only new question is which pair of neighbours is the worst. For both kinds of noise the divergence grows with the shift, so the worst pair is the one whose answers are furthest apart, at a distance equal to the sensitivity.

The **Gaussian mechanism** adds normal noise, and with sensitivity 1 the noise has standard deviation $\sigma$. The log of the ratio of two normal densities is a difference of two squares whose quadratic terms cancel. So the privacy loss is a straight-line function of the output, and is therefore itself normal. The exponential moments of a normal variable have a closed form, whose logarithm is a quadratic in $\lambda$. Dividing it by $\alpha - 1$ leaves a straight line in the order:
$$\varepsilon(\alpha) = \frac{\alpha}{2\sigma^2}.$$ {#eq-rdp-gauss}
For a query that returns a vector, the noise is added to each coordinate and looks the same in every direction. Only the Euclidean distance between the two answers then matters. So $\sigma$ is the noise standard deviation in units of the $\ell_2$ sensitivity, which is the largest Euclidean distance between the answers on neighbours.

The **Laplace mechanism** adds noise from the Laplace distribution. Its scale, written $b$, sets the typical size of the noise, and its density $\frac{1}{2b}e^{-|y|/b}$ falls off exponentially on both sides. Its curve for a real-valued query with sensitivity 1 is
$$\varepsilon(\alpha) = \frac{1}{\alpha - 1}\ln\left(\frac{\alpha}{2\alpha - 1}e^{(\alpha - 1)/b} + \frac{\alpha - 1}{2\alpha - 1}e^{-\alpha/b}\right).$$ {#eq-rdp-laplace}
Randomized response has the curve @eq-rdp-rr from the previous section. All three curves are exact, and they are the ones Mironov (2017) gives.

@fig-rdp-curves draws the three curves, and two shapes appear. Which one you get depends on whether the privacy loss is bounded.

![RDP curves of the Gaussian mechanism ($\sigma = 1$), the Laplace mechanism ($b = 1$) and randomized response ($p = 0.75$), all with sensitivity 1, on a log scale in $\alpha$ (so the straight line $\alpha/2$ looks curved). The Laplace and randomized response curves level off at their pure DP levels $1/b$ and $\ln 3$ (dotted). The Gaussian curve $\alpha/2$ grows linearly.](figures/renyi_dp_curves.svg){#fig-rdp-curves}

When the privacy loss is bounded, the curve levels off. Randomized response has a loss of size $\ln 3$ at every output. For Laplace noise centred at the two answers 0 and 1, the loss at an output $y$ is $(|y - 1| - |y|)/b$. The triangle inequality keeps it between $-1/b$ and $1/b$. Every Rényi divergence is at most the largest loss, so these two curves rise towards $1/b$ and $\ln 3$ and stay below them.

When the privacy loss is unbounded, the curve keeps growing. The Gaussian loss is a straight line in the output, so it has no maximum, and the curve $\alpha/(2\sigma^2)$ grows without limit. The Gaussian mechanism is therefore not $\varepsilon$-DP for any $\varepsilon$. It is still $(\alpha, \varepsilon)$-RDP at every finite order, and that is all that composition and conversion need.

The straight-line shape of the Gaussian curve has a meaning of its own. Take one fixed pair of neighbours, and suppose its divergence is $\rho\alpha$ at every order, for some constant $\rho$. Then the log of the exponential moment of its loss is $\rho\lambda + \rho\lambda^2$. That is exactly the log-moment of a normal variable with mean $\rho$ and variance $2\rho$, so the privacy loss of that pair is normal. [Concentrated differential privacy](03-concentrated-dp.md) builds a definition on this fact.

@tbl-rdp-curves gives the three curves at a few orders. As the order falls towards 1, the Laplace curve tends to its KL divergence, 0.3679.

| Mechanism | $\varepsilon(1.5)$ | $\varepsilon(2)$ | $\varepsilon(10)$ | $\varepsilon(64)$ | $\varepsilon(\infty)$ |
| --- | --- | --- | --- | --- | --- |
| Gaussian, $\sigma = 1$ | 0.7500 | 1.0000 | 5.0000 | 32.0000 | $\infty$ |
| Laplace, $b = 1$ | 0.5129 | 0.6191 | 0.9287 | 0.9891 | 1 |
| Randomized response, $p = 0.75$ | 0.7340 | 0.8473 | 1.0666 | 1.0940 | 1.0986 |

: RDP curves of the three basic mechanisms with sensitivity 1. {#tbl-rdp-curves}

These are the curves of a step that uses the whole dataset. A step that uses a random subset of the data has a smaller curve, which [Amplification by subsampling](05-subsampling.md) derives.

## Composition adds curves {#sec-rdp-composition}

Training a model runs thousands of mechanisms, one after another, and each is chosen after seeing the outputs of the earlier ones. A gradient step, for example, depends on the model that all the earlier steps produced. You need the curve of the whole run.

If two steps used independent randomness, their curves would add for a simple reason. The likelihood ratio of the pair of outputs is the product of the two ratios. For independent factors, the average of a product is the product of the averages, which the logarithm turns into a sum. The steps of a training run are not independent, but the curves still add. Write $y_1$ for the first output. The result assumes that, once you fix $y_1$, the second step is a mechanism whose curve is at most a bound $\varepsilon_2$, the same bound whatever $y_1$ was. Then the second step's exponential moment, averaged over its own output with $y_1$ fixed, is at most a constant that does not depend on $y_1$. Pull that constant out, and the average over $y_1$ of the remaining factor is the first step's moment. Taking the logarithm adds the two curves.

Write $\varepsilon_1$ for the curve of the first step and $\varepsilon_M$ for the curve of the composed mechanism, which releases both outputs. Then
$$\varepsilon_M(\alpha) \le \varepsilon_1(\alpha) + \varepsilon_2(\alpha) \qquad\text{for every } \alpha > 1.$$ {#eq-rdp-compose}
Treating the first steps as one mechanism and repeating the argument extends this to any number of steps. The curve of $k$ adaptively chosen mechanisms is at most the sum of their $k$ curves, order by order (Mironov 2017).

The bound gives something away only where the argument replaced the second step's moment by its worst case over $y_1$. To see how much, take two randomized responses about the same bit. The first keeps the bit with probability 0.75, and the second adapts to the first output. After a 1 it keeps the bit with probability 0.75, but after a 0 only with probability 0.6. At order 2, the divergence between the two joint output distributions is 1.6766, while the bound is 1.6946. The bound is larger because it charges the second step its worst curve, 0.8473, on both branches. After a first output of 0, the second step's true curve is only 0.1542, from @eq-rdp-rr with keep-probability 0.6. If the second step does not adapt, the two steps are independent and the divergence is exactly 1.6946. So the bound is tight for independent steps.

This is why composition works for RDP and not for $(\varepsilon, \delta)$ directly. An exponential moment is an average of a product, so it splits into one factor per step. The smallest $\delta$ that works for a given $\varepsilon$ is a different kind of quantity. It is the **hockey-stick divergence**, the total probability by which one output distribution exceeds $e^\varepsilon$ times the other. That total counts only the part of the difference that lies above zero, and such a cut-off part does not split into one factor per step. So composing $(\varepsilon, \delta)$ points needs a separate theorem with its own slack, and that is the route that lost a factor of nine in the opening example.

One condition remains. The steps can be chosen adaptively, but each step's curve bound must be fixed in advance. The argument needs the second step's bound to hold whatever $y_1$ was, so the bound cannot itself depend on the data or on the earlier outputs. The sum of curves is no longer valid if you choose the noise level or the number of steps on the fly from the data. That situation needs the tools of [Adaptive composition](08-adaptive-composition.md), which track a budget whose parts are chosen as the run goes.

For Gaussian steps the sum has a simple picture, and here it is exact. Each step has the curve $\alpha/(2\sigma^2)$, so $k$ steps have $k$ times that curve. That is the curve of a single Gaussian step with noise multiplier $\sigma/\sqrt k$. To see why, stack the $k$ answers into one vector. A change of one record moves each answer by at most 1, so it moves the vector by at most $\sqrt k$ in Euclidean length. A shift of $\sqrt k$ under noise $\sigma$ is the same as a shift of 1 under noise $\sigma/\sqrt k$. So the run is a single Gaussian mechanism, whose curve is exactly the sum. In the opening example, 1000 steps with $\sigma = 20$ act like one step with $\sigma = 0.6325$.

Different mechanisms compose just as easily, because their curves are added order by order. Take a Laplace step with $b = 1$ followed by a Gaussian step with $\sigma = 1$, and read both curves from @tbl-rdp-curves. At order 2 the composed curve is $0.6191 + 1 = 1.6191$. At order 10 it is $0.9287 + 5 = 5.9287$.

The same picture shows why private training needs more than noise. The differentially private version of stochastic gradient descent (SGD) is called DP-SGD. The running example of this part is a DP-SGD run of 10,000 steps, with noise multiplier 1 and sensitivity 1. If each step used the whole dataset, the run would act like one Gaussian step with noise multiplier $1/\sqrt{10{,}000} = 0.01$. Its exact guarantee at $\delta = 10^{-5}$ would be $\varepsilon = 5425.5$, which is no guarantee at all. Mironov's conversion, which the last section of this chapter introduces, gives 5615.1 at the grid's lowest order, 1.1. The best order for such a steep curve lies just above 1, below the grid. DP-SGD is private only because each step touches a random 1% of the data, and [Amplification by subsampling](05-subsampling.md) computes the curve of such a step.

## Post-processing and group privacy {#sec-rdp-properties}

Two properties of DP carry over to RDP. Post-processing matters because a released model, or the predictions it makes, must not leak more than the training procedure that produced it. Group privacy matters because records are correlated, as when a family shares a household or a user contributes many rows.

Post-processing cannot raise the curve. To state this, take any randomized map $g$ that you apply to the mechanism's output without looking at the data again. Write $g\circ M$ for the mechanism that runs $M$ and then applies $g$. At every order, its curve is no larger than that of $M$,
$$\varepsilon_{g\circ M}(\alpha) \le \varepsilon_M(\alpha).$$
The reason is that the likelihood ratio of a processed output is an average of the ratios of the raw outputs that could have produced it. Each raw output is weighted by how likely it was to lead there, and averaging pulls the ratios towards each other. A power above 1 of an average is at most the average of the powers, by Jensen's inequality. So averaging can only lower the exponential moment.

The condition that $g$ ignores the data is essential. A map that reads the data is not applied to the output alone, and it could simply print a record.

For an example, pass the output of randomized response with $p = 0.75$ through a second step that flips it with probability 0.2. The final output is then 1 with probability 0.65 when the bit is 1, but with probability 0.35 when the bit is 0. These probabilities are closer together than the original ones, so the order-2 divergence drops from 0.8473 to 0.3333.

Groups are harder. For pure DP, a group of $k$ records costs $k\varepsilon$ ([Properties of differential privacy](../part03-dp-foundations/02-properties.md)). You walk from one dataset to the other a record at a time, and the likelihood ratios along the way multiply, so the losses add. An RDP bound does not combine this way, because the Rényi divergence has no plain triangle inequality. To go through a dataset in between, you must control the average of a product of two likelihood ratios. Hölder's inequality bounds such an average by higher moments of each factor, so the detour costs higher orders.

The **weak triangle inequality** makes this precise. Take three output distributions, with $P$ and $Q$ at the two ends of a path and a third, written $R$, in between. In its simplest form, the inequality bounds the divergence across the whole path by divergences of about twice the order on its two legs,
$$D_\alpha(P\|Q) \le \frac{\alpha - 1/2}{\alpha - 1}D_{2\alpha}(P\|R) + D_{2\alpha - 1}(R\|Q).$$ {#eq-rdp-weak-triangle-2}
The order doubles on the first leg and nearly doubles on the second. The factor in front of the first leg is above 1, and it falls towards 1 as the order grows.

Applied to two records, the inequality turns a bound at order $\alpha$ for one record into a bound at order $\alpha/2$ for two. Repeating it gives **Mironov's group bound** (his Proposition 2, Mironov 2017). Write $c$ for the number of doublings, so the group has $2^c$ records. Suppose $M$ is $(\alpha, \varepsilon)$-RDP at an order of at least $2^{c+1}$. Then, on datasets that differ in $2^c$ records, $M$ satisfies a weaker guarantee at a lower order,
$$D_{\alpha/2^c}\big(M(D)\,\|\,M(D')\big) \le 3^c\varepsilon.$$
Each doubling of the group halves the order and multiplies $\varepsilon$ by 3. The condition on $\alpha$ keeps the order of the last doubling at least 2, and the reason is the factor in @eq-rdp-weak-triangle-2. Write $\beta$ for the order of the group guarantee, which for two records is half of $\alpha$. Only from $\beta = 2$ up is the factor $\frac{\beta - 1/2}{\beta - 1}$ of @eq-rdp-weak-triangle-2 at most $\frac32$. Two records then cost at most $\frac32\varepsilon + \varepsilon = \frac52\varepsilon$, which Mironov rounds up to $3\varepsilon$.

This bound is loose, and the Gaussian mechanism shows by how much. A group of $2^c$ records shifts a Gaussian query of sensitivity 1 by up to $2^c$. So the exact group curve is @eq-rdp-gauss with sensitivity $2^c$, which is $\alpha(2^c)^2/(2\sigma^2)$. @tbl-rdp-group compares it at $\sigma = 1$ with the weak triangle inequality, applied once per doubling, and with Mironov's group bound.

| Group size $2^c$, order | Exact | Weak triangle, iterated | Group bound $3^c\varepsilon(2^c\beta)$ | Group bound / exact |
| --- | --- | --- | --- | --- |
| 2, $\beta = 2$ | 4.0000 | 4.5000 | 6.0000 | 1.5000 |
| 2, $\beta = 4$ | 8.0000 | 8.1667 | 12.0000 | 1.5000 |
| 2, $\beta = 10$ | 20.0000 | 20.0556 | 30.0000 | 1.5000 |
| 4, $\beta = 2$ | 16.0000 | 18.5000 | 36.0000 | 2.2500 |
| 8, $\beta = 2$ | 64.0000 | 74.5000 | 216.0000 | 3.3750 |

: Group privacy for the Gaussian mechanism with $\sigma = 1$: the exact curve on inputs $2^c$ records apart, against the weak triangle inequality and Mironov's group bound. {#tbl-rdp-group}

For two records, the weak triangle inequality loses little once the group order $\beta$ is 4 or more. Mironov's group bound costs a factor of 1.5. The reason is that for two records the exact Gaussian value is $2\varepsilon(2\beta)$, while the bound charges $3\varepsilon(2\beta)$. That loss compounds with each doubling, which is why the last column grows to 3.375 for eight records. @fig-rdp-group draws the two-record case.

![RDP of the Gaussian mechanism ($\sigma = 1$) on inputs two records apart, as a function of the order $\beta$ of the group guarantee. Black: the exact curve $2\beta$. Dashed: the weak triangle inequality of @eq-rdp-weak-triangle-2. Dotted: Mironov's group bound (Proposition 2) $3\varepsilon(2\beta)$, drawn from $\beta = 2$, where its condition $\alpha = 2\beta \ge 4$ starts to hold.](figures/renyi_dp_group.svg){#fig-rdp-group}

So use the mechanism itself when you can. For the Gaussian, the exact group curve is just the curve with the group's sensitivity. The generic bound matters for mechanisms whose curve on far-apart inputs you cannot compute.

A mechanism that is already $\varepsilon$-DP has a second route. A group of $k$ records then gets $k\varepsilon$-DP, and since the curve never exceeds its value at order infinity, that bound holds at every order. It can beat Mironov's group bound. For the Laplace mechanism with $b = 1$, on inputs two records apart at order 2, the exact value is 1.5958. The weak triangle inequality gives 1.9674, and Mironov's group bound gives 2.4411. The pure DP bound for two records, $2/b = 2$, is better than the group bound, though not than the weak triangle inequality.

## The moments accountant is RDP {#sec-rdp-moments}

Abadi et al. (2016) introduced the **moments accountant** to train neural networks with DP, before RDP had a name. For each positive $\lambda$, in practice the integers up to 32, it tracks the logarithm of the moment generating function (MGF) of the privacy loss. The MGF is the average of $e^{\lambda L}$ as a function of $\lambda$. The accountant takes the worst pair of neighbours and writes $K(\lambda)$ for the logarithm of this average. It adds these numbers over the training steps and converts the total with a tail bound.

The definition of @sec-rdp-definition says that $(\alpha - 1)$ times the Rényi divergence of order $\alpha$ is the logarithm of exactly this average, at $\lambda = \alpha - 1$. So the moments accountant's number at $\lambda$ is $\lambda$ times the RDP curve at the order $\lambda + 1$, and the two convert into each other by
$$K(\lambda) = \lambda\,\varepsilon(\lambda + 1), \qquad\text{equivalently}\qquad \varepsilon(\alpha) = \frac{K(\alpha - 1)}{\alpha - 1}.$$ {#eq-rdp-moments}
Randomized response with $p = 0.75$ checks this on numbers. At $\lambda = 1$, the average of $e^{L}$ under $P$ adds the same two terms as the last column of @tbl-rdp-rr. Their sum is 2.3333, whose logarithm is $\varepsilon(2) = 0.8473$.

The two parts of the moments accountant's main theorem are then the two results of this chapter that RDP needs. Its composability says that the log-moments add over steps, which is @eq-rdp-compose at order $\lambda + 1$, multiplied by $\lambda$. Its tail bound says that the composed mechanism is $(\varepsilon, \delta)$-DP with
$$\delta = \min_\lambda\exp\big(K(\lambda) - \lambda\varepsilon\big).$$
That is the conversion that the next section derives, read the other way round, fixing $\varepsilon$ and solving for $\delta$.

The two accountants therefore give the same numbers with different bookkeeping. The moments accountant stores $(\alpha - 1)\varepsilon(\alpha)$ where RDP stores $\varepsilon(\alpha)$, and both add over steps. Since the identity is this simple, what was new in 2016 was not the bookkeeping. It was a bound on $K(\lambda)$ for the subsampled Gaussian step of DP-SGD, where each step uses a random fraction of the data.

In statistical terms, $K$ is the **cumulant generating function** of the worst pair's privacy loss. That is the name for the logarithm of an MGF, whose derivatives at 0 give the mean, the variance and the higher cumulants of the variable. A normal loss has no cumulants beyond its variance, which is why it gives the straight-line curve of @sec-rdp-curves.

## From RDP to $(\varepsilon, \delta)$ {#sec-rdp-conversion}

After composing, you hold a curve, but you report an $(\varepsilon, \delta)$ pair. That pair has a direct meaning for events: no set of outputs becomes more than $e^\varepsilon$ times likelier, up to an additive $\delta$. The conversion rests on one observation. Pick a threshold $c$. Every output whose privacy loss is at most $c$ is at most $e^c$ times likelier under one neighbour than under the other. A set made only of such outputs satisfies the same bound, because its probability is a sum over its outputs. So only outputs with loss above $c$ can break the bound, and you can charge their total probability to $\delta$.

That total probability is a tail of the privacy loss. The Chernoff method of [Concentration inequalities](../part01-prerequisites/02-concentration.md) bounds a tail by an exponential moment, by applying Markov's inequality to $e^{\lambda L}$. An exponential moment is exactly what each point of the RDP curve provides. Setting the tail bound equal to $\delta$ and solving for $c$ gives **Mironov's conversion** (Mironov 2017). Take a mechanism with RDP curve $\varepsilon(\alpha)$, pick an order, and pick a slack $\delta$ between 0 and 1. The mechanism is then $(\varepsilon_\delta, \delta)$-DP with
$$\varepsilon_\delta = \varepsilon(\alpha) + \frac{\ln(1/\delta)}{\alpha - 1}.$$ {#eq-rdp-mironov}
The converted $\varepsilon$ is the RDP value plus a tail term that pays for $\delta$.

Every order gives a valid pair, so you take the best one. The two terms pull in opposite directions, because the curve grows with the order while the tail term shrinks. Small orders therefore make the curve small but the tail term large, and large orders do the opposite. When you hold the whole curve, you minimize @eq-rdp-mironov over the grid of orders, once, after composing. For the Gaussian curve $\alpha/(2\sigma^2)$, setting the derivative in the order to zero gives the minimum in closed form, and at that minimum the two order-dependent terms are equal:
$$\alpha^* = 1 + \sigma\sqrt{2\ln(1/\delta)}, \qquad \varepsilon_\delta = \frac{1}{2\sigma^2} + \frac{\sqrt{2\ln(1/\delta)}}{\sigma}.$$ {#eq-rdp-gauss-opt}
For $k$ Gaussian steps, put the composed noise multiplier $\sigma/\sqrt k$ in place of $\sigma$. The second term then grows like $\sqrt k$, the same shape as advanced composition. The difference is that its constants come from the Gaussian curve instead of from a worst-case $(\varepsilon_0, \delta_0)$ point.

The conversion gives something away in two places. First, it charges the whole probability of every output above the threshold to $\delta$. But an output above the threshold breaks the $e^c$ bound only by its excess over $e^c$ times its probability under the other neighbour. For an output with loss $L$, that excess is the fraction $1 - e^{c - L}$ of its probability, so an output just over the line costs almost nothing. Second, the Chernoff bound overestimates the tail it bounds. [From Rényi DP to (ε, δ) and to tests](02-rdp-conversions.md) removes much of both losses with sharper conversions.

Take one Gaussian step with $\sigma = 1$ at $\delta = 10^{-5}$. Its curve is $\alpha/2$, and since $\ln(1/\delta) = 11.5129$, the conversion adds $11.5129/(\alpha - 1)$. Three orders show the trade-off, starting with order 2, where the converted $\varepsilon$ is 12.5129 and almost all of it is tail term. At order 20, by contrast, it is 10.6059, almost all of it curve. In between, at order 5.8, the two terms nearly balance and give 5.2985, the best value on the grid. That order is next to the continuous optimum of @eq-rdp-gauss-opt, the order 5.7985.

To see how good this is, compare it with the exact $\varepsilon$ of the Gaussian mechanism. Write $\mu$ for the distance between the centres of the two output distributions, measured in noise standard deviations, which is $1/\sigma$ for one step with sensitivity 1. Write $\Phi$ for the cumulative distribution function of the standard normal distribution, so $\Phi(t)$ is the probability that a standard normal variable is below $t$. The exact privacy curve of the Gaussian mechanism ([Laplace and Gaussian mechanisms](../part03-dp-foundations/03-laplace-gaussian.md)) is
$$\delta(\varepsilon) = \Phi\Big(\frac\mu2 - \frac\varepsilon\mu\Big) - e^\varepsilon\Phi\Big(-\frac\mu2 - \frac\varepsilon\mu\Big).$$
Solving it for $\varepsilon$ at $\delta = 10^{-5}$ gives 4.3772, so the conversion is 21% too large. Read the other way, at the converted $\varepsilon$ the mechanism's true $\delta$ is 76.9 times smaller than the conversion promises.

Now return to the opening example, $k$ Gaussian steps with $\sigma = 20$. @tbl-rdp-compare compares four ways to account for them. The exact value is available because $k$ steps act as one Gaussian with $\mu = \sqrt k/\sigma$. The two routes from Part 3 first turn each step into one $(\varepsilon_0, \delta_0)$ point of its exact curve, then compose the points. Basic composition adds both parts of the per-step guarantees. Advanced composition spends part of $\delta$ on the per-step slacks and the rest on a tail bound for the summed loss. It uses the best of the candidate splits that Part 3 tries.

| $k$ | Basic | Advanced | RDP + Mironov (best order) | Exact | RDP / exact | Advanced / exact |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 0.160 | 0.847 | 0.251 (128) | 0.160 | 1.566 | 5.29 |
| 10 | 1.892 | 3.270 | 0.771 (31) | 0.561 | 1.374 | 5.83 |
| 100 | 21.510 | 13.423 | 2.524 (10.6) | 1.993 | 1.267 | 6.73 |
| 1000 | 238.578 | 68.008 | 8.838 (4) | 7.511 | 1.177 | 9.05 |

: Total $\varepsilon$ at $\delta = 10^{-5}$ for $k$ Gaussian steps with $\sigma = 20$, four ways. {#tbl-rdp-compare}

Advanced composition is 5 to 9 times the exact value, and it gets looser as $k$ grows, because every step's curve was collapsed to one point before composing. For a single step it is even worse than basic composition, because its tail term is paid even when there is no sum to concentrate.

RDP is 18% to 57% above the exact value in this table, and it gets tighter as $k$ grows. Most of the gain comes from composing curves rather than points, so even a loose conversion done once at the end beats a good theorem applied to points. @fig-rdp-conversion shows the trade-off over orders for 1000 steps, and the three methods as the number of steps grows.

![Left: converting the curve of 1000 Gaussian steps with $\sigma = 20$ at $\delta = 10^{-5}$. The converted $\varepsilon$ (black) is the sum of the RDP curve (dashed) and the tail term $\ln(1/\delta)/(\alpha - 1)$ (dotted). Its minimum over the grid, 8.84 at $\alpha = 4$, sits above the exact value 7.51 (green). Right: total $\varepsilon$ against the number of steps $k$ on log axes, for advanced composition, RDP with Mironov's conversion, and the exact value.](figures/renyi_dp_conversion.svg){#fig-rdp-conversion}

The best order falls as the run grows, because a steeper composed curve makes large orders expensive. By @eq-rdp-gauss-opt, one step with $\sigma = 20$ wants an order of about 97. A thousand steps, whose composed noise multiplier is 0.6325, want an order of about 4. So a grid must run from just above 1 to a few hundred.

The remaining gap to the exact value is the conversion's slack, and it depends on the composed $\mu$. At the continuous optimum, Mironov's conversion is only 1.07 times the exact value at $\mu = 10$, but 1.21 times at $\mu = 1$, 1.42 times at $\mu = 0.1$ and 1.76 times at $\mu = 0.01$. A small $\mu$ means two output distributions that are hard to tell apart. Such mechanisms pay the most, because as $\mu$ shrinks, the exact $\varepsilon$ falls faster than the Chernoff bound does. So for reporting, use the sharper conversions of the next chapter or a numerical accountant ([Numerical accounting](06-numerical-accounting.md)).

## Code {#sec-rdp-code}

RDP accounting for full-batch Gaussian steps needs only three operations. You evaluate the curve on the grid of orders, add it over the steps, and minimize the conversion over the orders. This snippet uses the part's grid of 154 orders and reproduces three numbers from this chapter.

```python
from math import log

ORDERS = [1 + x / 10 for x in range(1, 100)] + list(range(12, 64)) + [128, 256, 512]


def gaussian_rdp(alpha: float, sigma: float, steps: int = 1) -> float:
    """RDP curve of `steps` Gaussian steps with noise multiplier sigma (sensitivity 1)."""
    return steps * alpha / (2 * sigma**2)


def mironov_eps(curve: list[float], delta: float) -> tuple[float, float]:
    """Best (epsilon, order) from Mironov's conversion over the grid."""
    return min((eps + log(1 / delta) / (a - 1), a) for a, eps in zip(ORDERS, curve))


one = [gaussian_rdp(a, sigma=1.0) for a in ORDERS]
run = [gaussian_rdp(a, sigma=20.0, steps=1000) for a in ORDERS]
full_batch = [gaussian_rdp(a, sigma=1.0, steps=10_000) for a in ORDERS]
print("%.4f at order %g" % mironov_eps(one, 1e-5))  # 5.2985 at order 5.8
print("%.3f at order %g" % mironov_eps(run, 1e-5))  # 8.838 at order 4
print("%.1f at order %g" % mironov_eps(full_batch, 1e-5))  # 5615.1 at order 1.1
```

The last line shows the best order stuck at the bottom of the grid, which is the sign that the grid is too short for that curve. The full edition's script, [renyi_dp.py](../../code/part04-privacy-accounting/renyi_dp.py), computes every number in this chapter, including the exact Gaussian values. Run it from the repository root.

```bash
uv run python code/part04-privacy-accounting/renyi_dp.py
```

## Cheat sheet

| Result | Formula | Notes |
| --- | --- | --- |
| $(\alpha, \varepsilon)$-RDP | $D_\alpha(M(D)\Vert M(D')) \le \varepsilon$ for all neighbours | Non-decreasing in $\alpha$; $\varepsilon(\infty)$ is pure DP; $\alpha \to 1$ gives at least the largest KL |
| Gaussian, noise multiplier $\sigma$ | $\varepsilon(\alpha) = \alpha/(2\sigma^2)$ | Exact; $\sigma$ in units of the $\ell_2$ sensitivity |
| Laplace, scale $b$ | $\frac{1}{\alpha - 1}\ln\big(\frac{\alpha}{2\alpha - 1}e^{(\alpha - 1)/b} + \frac{\alpha - 1}{2\alpha - 1}e^{-\alpha/b}\big)$ | Exact, sensitivity 1; levels off at $1/b$ |
| Randomized response, keep $p > \frac12$ | $\frac{1}{\alpha - 1}\ln\big(p^\alpha(1-p)^{1-\alpha} + (1-p)^\alpha p^{1-\alpha}\big)$ | Exact; levels off at $\ln\frac{p}{1-p}$ |
| Composition | $\varepsilon_M(\alpha) \le \sum_i\varepsilon_i(\alpha)$ | Adaptive steps, bounds fixed in advance; equality for independent steps |
| $k$ Gaussian steps | One Gaussian with $\sigma/\sqrt k$ | Full-batch only |
| Post-processing | $\varepsilon_{g\circ M}(\alpha) \le \varepsilon_M(\alpha)$ | $g$ must not read the data |
| Weak triangle | $D_\alpha(P\Vert Q) \le \frac{\alpha - 1/2}{\alpha - 1}D_{2\alpha}(P\Vert R) + D_{2\alpha - 1}(R\Vert Q)$ | Costs about twice the order |
| Mironov's group bound, $2^c$ records | Order $\alpha/2^c$, $3^c\varepsilon(\alpha)$, if $\alpha \ge 2^{c+1}$ | Loose; for the Gaussian use sensitivity $2^c$ |
| Moments accountant | $K(\lambda) = \lambda\,\varepsilon(\lambda + 1)$ | Same numbers as RDP |
| Mironov's conversion | $\varepsilon(\alpha) + \ln(1/\delta)/(\alpha - 1)$, best $\alpha$ | 1.07 ($\mu = 10$) to 1.76 ($\mu = 0.01$) times exact for a composed Gaussian; worse for weaker mechanisms |
| Gaussian optimum | $\alpha^* = 1 + \sigma\sqrt{2\ln(1/\delta)}$ | $\sigma$ is the composed noise multiplier |

: The results of this chapter. {#tbl-rdp-cheat}

**Rules of thumb**

- Compose in RDP and convert once at the end. Never convert each step to $(\varepsilon_0, \delta_0)$ first.
- Keep an order grid from just above 1 to a few hundred, and check that the best order is not at its edge.
- For group privacy of a Gaussian mechanism, use the curve with the group's sensitivity rather than the generic group bound.
- For reporting, prefer a sharper conversion or a numerical accountant to Mironov's conversion.

## Further reading

- Mironov, "Rényi Differential Privacy" (IEEE CSF 2017): the definition, composition, the curves of the basic mechanisms, group privacy and the conversion. Short and readable.
- Abadi et al., "Deep Learning with Differential Privacy" (CCS 2016): DP-SGD and the moments accountant, with the experiments that made private deep learning practical.
- Mironov, Talwar & Zhang, "Rényi Differential Privacy of the Sampled Gaussian Mechanism" (arXiv 2019): the curves of subsampled steps, and the reference behind most RDP accountants.

**Derivations.** Every result in this chapter is derived step by step in [the full edition's chapter](../../book-full/part04-privacy-accounting/01-renyi-dp.md). It takes the exact privacy curve of the Gaussian mechanism (Balle & Wang 2018) and the two composition theorems of the comparison from Part 3.
