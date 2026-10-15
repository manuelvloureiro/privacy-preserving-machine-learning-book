# From Rényi DP to (ε, δ) and to tests

## Why this matters

Rényi differential privacy (RDP) describes a mechanism by a curve, with one value for each **order** $\alpha$ greater than 1. Each value bounds how far apart the mechanism's outputs on two neighbouring datasets can be, measured by the Rényi divergence of that order. Accountants track this curve because the curves of the steps in a run simply add up ([Rényi differential privacy](01-renyi-dp.md)).

Nobody reports the curve, though. A paper, a model card or a regulator asks instead for a differential privacy (DP) guarantee with two numbers, a privacy parameter $\varepsilon$ and a small probability $\delta$. The guarantee says that one record can change the chance of any set of outputs by at most a factor $e^\varepsilon$, up to an extra probability $\delta$. Others ask for a statement about attacks, which says how much better than guessing a test can tell whether your record was used. So every RDP analysis ends with a conversion, and the conversion decides the number you publish.

This chapter answers three questions about that conversion. The first is how much the standard conversion loses, and where it loses it. The second is what the best $(\varepsilon, \delta)$ is that you can claim from an RDP curve, and which order gives it. The third is what an RDP curve says about tests, and what it can never say.

One idea answers all three. The privacy loss of an output measures how much more likely that output is without your record than with it. An RDP curve only summarizes how this loss is distributed, so a conversion must give a $\delta$ that holds for every mechanism with the same summary, not just for yours. Mironov's conversion, the original one, throws away most of what the summary says, while the improved conversion recovers almost all of it. For the Gaussian mechanism it agrees to three decimals with the best that any method could do from the curve alone. What remains is an $\varepsilon$ 5 to 10% above the exact value, and no formula can remove it. A mechanism with only two outputs stays within the Gaussian's curve at every order but has a larger $\delta$. A conversion that sees only the curve must cover it too.

## The privacy loss, δ and the RDP curve {#sec-conv-loss}

Every conversion in this chapter compares two averages over the same random quantity, so this section sets up that quantity first. Take two neighbouring datasets, which differ in one person's record. Write $P$ for the distribution of the mechanism's output on the dataset without your record, and $Q$ for its distribution on the dataset with it. The **privacy loss** of an output is the logarithm of how much more likely that output is under $P$ than under $Q$. It is positive for an output that is more likely without your record, and negative for one that is more likely with it. If you draw the output at random from $P$, its privacy loss is a random number, written $L$.

The $\delta$ of an $(\varepsilon, \delta)$ guarantee is an average over this loss ([The Laplace and Gaussian mechanisms](../part03-dp-foundations/03-laplace-gaussian.md)). The guarantee says that no set of outputs is more than $e^\varepsilon$ times as likely without your record as with it, up to an extra probability $\delta$. So the smallest $\delta$ that works is the largest excess of a set's probability without your record over $e^\varepsilon$ times its probability with it. The set with the largest excess collects exactly the outputs whose loss exceeds $\varepsilon$, because any other output would only lower the excess.

This lets you count the excess output by output. An output's loss $L$ says that it is $e^L$ times as likely without your record as with it. So its probability with your record is $e^{-L}$ times its probability without, and its excess is its probability without your record times $1 - e^{\varepsilon - L}$. That fraction is positive exactly when the loss exceeds $\varepsilon$. So averaging the fractions over outputs drawn from $P$ gives the smallest $\delta$ that works,
$$\delta(\varepsilon) = \mathbb E_P\big[(1 - e^{\varepsilon - L})_+\big].$$ {#eq-conv-delta}
Here $(x)_+$ stands for $x$ when it is positive and for 0 otherwise.

The formula is worth reading closely, because each conversion below loses by misreading it. An output contributes only when its loss exceeds $\varepsilon$, and then only part of its probability. For example, an output whose loss is $\varepsilon + \ln 2$ contributes half its probability, since $e^{-\ln 2}$ is one half. An output whose loss is just above $\varepsilon$ contributes almost nothing. Only an output with an infinite loss, one that is impossible under $Q$, contributes all of its probability.

The RDP curve is a different average over the same loss. The Rényi divergence of order $\alpha$ is the logarithm of the average of $e^{(\alpha - 1)L}$, divided by $\alpha - 1$ ([Rényi differential privacy](01-renyi-dp.md)). The **RDP curve** $\varepsilon(\alpha)$ bounds this divergence for every pair of neighbours. Undoing the logarithm and the division turns the bound into a bound on an average:
$$\mathbb E_P\big[e^{(\alpha - 1)L}\big] \le e^{(\alpha - 1)\varepsilon(\alpha)}.$$ {#eq-conv-mgf}
Such an average of $e^{\lambda L}$ is called an exponential moment of the loss, and as a function of $\lambda$ it is the moment generating function of the loss. So the RDP curve gives an upper bound on this function at the points $\alpha - 1$ for the orders you tracked, and nothing else about the loss. A conversion has to bound the average in @eq-conv-delta using only the averages in @eq-conv-mgf.

A guarantee must also hold with the two datasets swapped, and RDP bounds that direction by the same curve, since swapping two neighbours gives two neighbours. So every argument below works the same way in the other direction.

The running example is the Gaussian mechanism, because its exact $\varepsilon$ can be computed and every conversion can be measured against it. Each step adds normal noise to a query of sensitivity 1, and the **noise multiplier** $\sigma$ is the standard deviation of that noise. Now run a number $k$ of such steps, each with independent noise, and treat the outputs together as one vector. Each step moves its own coordinate of the output by $1/\sigma$ noise standard deviations, and $k$ such moves at right angles add up to a distance of $\sqrt k/\sigma$. The noise looks the same in every direction, so only this distance matters, and every number in this chapter depends on it alone. Write $\mu$ for it. One step has the curve $\alpha/(2\sigma^2)$, and composition adds $k$ copies of it, so the composed curve is
$$\varepsilon(\alpha) = \frac{\alpha\mu^2}{2}.$$

The privacy loss of this mechanism is normal, so the full edition computes its exact $\delta$ in closed form. All numbers use $\delta = 10^{-5}$ and the grid of 154 orders that every chapter of this part uses. The grid runs from 1.1 to 10.9 in steps of 0.1, then takes every integer from 12 to 63, and ends with 128, 256 and 512.

## Where Mironov's conversion loses {#sec-conv-mironov}

[Rényi differential privacy](01-renyi-dp.md) derived Mironov's conversion, and its proof has two steps. Seen through @eq-conv-delta, each step throws away a different piece of information, and the Gaussian shows how much each one costs.

The first step ignores how far past $\varepsilon$ each loss goes. It charges the whole probability of every output with a loss above $\varepsilon$ to $\delta$, instead of the fraction that @eq-conv-delta charges. That replaces $\delta$ by the tail probability $P[L > \varepsilon]$, which is larger. The second step bounds this tail with Markov's inequality, applied to $e^{(\alpha - 1)L}$, which is never negative, as Markov's inequality requires. Because $\alpha$ is above 1, this quantity exceeds $e^{(\alpha - 1)\varepsilon}$ exactly when the loss exceeds $\varepsilon$. So the tail is at most its average divided by $e^{(\alpha - 1)\varepsilon}$, and the RDP curve bounds the average. Together, at every order $\alpha$,
$$\delta(\varepsilon) \le P[L > \varepsilon] \le e^{(\alpha - 1)(\varepsilon(\alpha) - \varepsilon)}.$$ {#eq-conv-mironov-chain}
Setting the right side equal to $\delta$ and solving for $\varepsilon$ gives **Mironov's conversion**
$$\varepsilon' = \varepsilon(\alpha) + \frac{\ln(1/\delta)}{\alpha - 1},$$ {#eq-conv-mironov}
which you then minimize over the orders of the grid.

@tbl-conv-mironov-steps measures the two steps on the Gaussian. Its first three rows turn the sides of @eq-conv-mironov-chain into an $\varepsilon$ at $\delta = 10^{-5}$, and the last two give what each step adds.

| | $\mu = 1$ (one step, $\sigma = 1$) | $\mu = 10$ (100 steps, $\sigma = 1$) |
| --- | --- | --- |
| Exact $\varepsilon$ | 4.377 | 91.817 |
| After step 1, $P[L > \varepsilon] = \delta$ | 4.765 | 92.649 |
| After step 2, Mironov | 5.299 at $\alpha = 5.8$ | 98.026 at $\alpha = 1.5$ |
| Loss from step 1 | 0.388 | 0.832 |
| Loss from step 2 | 0.534 | 5.377 (0.041 of it from the grid) |

: Mironov's conversion for the Gaussian at $\delta = 10^{-5}$, split into its two steps. {#tbl-conv-mironov-steps}

Step 1 loses because of where the Gaussian's tail sits. The privacy loss is normal, and $\varepsilon$ lies several standard deviations above its mean, where its density falls steeply. So most of the tail lies just above $\varepsilon$, which is exactly where each output's true contribution is close to zero. Step 1 would lose nothing only if every loss above $\varepsilon$ were infinite, as for a mechanism that sometimes publishes the record outright, read with the two datasets swapped.

Step 2 loses more, because Markov's inequality is exact only for a quantity that puts all its probability on 0 and on the threshold. A normal loss is nothing like that. For the longer run, the second step accounts for 5.377 of the total excess of 6.209.

The losses look even larger on the $\delta$ side. For the single release, the exact $\delta$ at Mironov's $\varepsilon$ of 5.299 is only $1.300\times10^{-7}$. So you pay for a $\delta$ of $10^{-5}$ and get one of $1.3\times10^{-7}$.

As an example, release the mean of a bounded statistic with the Gaussian mechanism at $\sigma = 1$, with the noise measured in units of the statistic's sensitivity. Then $\mu$ is 1 as well, and the curve is $\alpha/2$. Mironov's conversion then gives 5.299 at order 5.8, which no order between the grid points would improve on. The exact answer is 4.377, so Mironov's number overstates $\varepsilon$ by 0.921. Such a conversion is still valid, only loose, so older code and papers that use it are not wrong. They include the original moments accountant, which tracked the same exponential moments ([Rényi differential privacy](01-renyi-dp.md)). So check which conversion each side used before you compare your $\varepsilon$ with a published one.

## The improved conversion {#sec-conv-improved}

Seen together, Mironov's two steps replace each output's contribution to $\delta$ by an exponential in its loss that lies above it. Write $z$ for the loss of an output, so that its contribution is $(1 - e^{\varepsilon - z})_+$. The exponential has the form $c\,e^{(\alpha - 1)z}$, for some constant factor $c$. The bound has to be an exponential, because that is the only function whose average the RDP curve bounds. Mironov's choice of the constant $c$ makes the exponential equal 1 at $z = \varepsilon$, where the true contribution is still 0. A lower exponential still stays above the contribution, because the contribution is well below 1 just above $\varepsilon$. Far to the right there is no risk either, since the contribution never exceeds 1 while the exponential keeps growing. The lowest exponential touches the contribution slightly to the right of $\varepsilon$ (@fig-conv-majorants), and using it is the whole improvement.

![The contribution $(1 - e^{\varepsilon - z})_+$ of an output with loss $z$ at $\varepsilon = 3$ (black), and the two exponentials of order 6 that bound it: Mironov's, which equals 1 at $z = \varepsilon$ (red), and the smallest one, which touches the contribution just to the right of $\varepsilon$ (blue). The shaded curve is the density of the Gaussian privacy loss at $\mu = 1$.](figures/rdp_conversions_majorants.svg){#fig-conv-majorants}

The result is the **improved conversion**, which Balle et al., Canonne, Kamath & Steinke, and Asoodeh et al. found independently in 2020. A mechanism with RDP curve $\varepsilon(\alpha)$ is $(\varepsilon', \delta)$-DP with
$$\varepsilon' = \varepsilon(\alpha) + \ln\frac{\alpha - 1}{\alpha} - \frac{\ln\delta + \ln\alpha}{\alpha - 1}.$$ {#eq-conv-improved}
Again you minimize over the orders of the grid. If the result comes out negative, report 0. The smallest valid $\delta$ can only fall as $\varepsilon$ grows, so a guarantee at a negative $\varepsilon$ also holds at 0. Read the other way, at a given $\varepsilon$ the improved $\delta$ is Mironov's multiplied by a factor below 1,
$$\delta = \frac{e^{(\alpha - 1)(\varepsilon(\alpha) - \varepsilon)}}{\alpha}\Big(1 - \frac1\alpha\Big)^{\alpha - 1}.$$ {#eq-conv-improved-delta}

The gain depends only on the order. If you compare @eq-conv-improved with @eq-conv-mironov at the same order, both the curve and the $\ln(1/\delta)$ term cancel. What is left is the saving in $\varepsilon$, which depends neither on the mechanism nor on $\delta$. Written $\Delta(\alpha)$, it is
$$\Delta(\alpha) = \frac{\ln\alpha}{\alpha - 1} + \ln\frac{\alpha}{\alpha - 1}.$$ {#eq-conv-gain}
On the $\delta$ side, the factor in @eq-conv-improved-delta behaves like $1/(e\alpha)$ at large orders, and @tbl-conv-gain gives the factor and the saving $\Delta$ at three orders.

| $\alpha$ | $\delta$ factor | $\varepsilon$ saving $\Delta(\alpha)$ |
| --- | --- | --- |
| 2 | 0.2500 | 1.3863 |
| 10 | 0.0387 | 0.3612 |
| 100 | 0.0037 | 0.0566 |

: What the improved conversion gains over Mironov's at the same order: the factor by which its $\delta$ is smaller, and the saving in $\varepsilon$. {#tbl-conv-gain}

Over a whole grid, the saving is at least $\Delta$ at the order that Mironov's conversion picks, for any RDP curve. The reason is that the improved conversion is free to pick its own best order. Its minimum is therefore at most its value at Mironov's order, which is Mironov's minimum minus $\Delta$ there. That gives you a diagnostic. If two accountants report values about 0.4 apart from the same curve, at a best order near 10, one of them is using Mironov's conversion. Opacus and Google's `dp_accounting`, two widely used libraries for private training, both use the improved one.

For the mean released at $\sigma = 1$, Mironov's conversion picks order 5.8. At that order the saving $\Delta$ is 0.555, so the improved conversion must do at least that much better. In fact it picks order 5.4 and gives $(4.729, 10^{-5})$-DP. That removes 0.570 of the 0.921 excess from the same curve. The saving is a little more than $\Delta$ at Mironov's order, because the improved conversion moves to a better order. It even beats the 4.765 of step 1 alone, because its exponential is allowed to dip below 1 just above $\varepsilon$, where the true contributions are small.

A single order shows how much the improved conversion leaves on the table. Take the release with $\mu = 1$, and keep one bound from its curve, the one at order 2, in one direction only, which is all the improved conversion uses. From that bound alone, Mironov's conversion gives 12.513 and the improved conversion gives 11.127. The best possible from the same bound, which the next section shows how to compute, is lower still, at 10.668. At orders 8 and 32, however, the improved conversion and the best possible agree to three decimals.

## RDP as a constraint on tests {#sec-conv-tests}

An attacker who wants to know whether your record was used runs a test. It looks at the output and decides which of two hypotheses is true ([Hypothesis testing](../part01-prerequisites/04-hypothesis-testing.md)). One says that the output came from $P$, so your record was not used. The other says that it came from $Q$, so your record was used. Its **type I error** $a$ is the chance that it says "used" when the output came from $P$, which for a membership test is its false-positive rate. Its **type II error** $b$ is the chance that it says "not used" when the output came from $Q$, so its power, the chance of correctly saying "used", is $1 - b$. Throughout, $\alpha$ is the Rényi order and never an error rate.

Whatever the test computes, its verdict is a coin, whose two sides have the probabilities of @tbl-conv-test-outcome in each world.

| Verdict | Probability under $P$ | Probability under $Q$ |
| --- | --- | --- |
| "used" | $a$ | $1 - b$ |
| "not used" | $1 - a$ | $b$ |

: The verdict of a test with errors $a$ and $b$, as a coin under each hypothesis. {#tbl-conv-test-outcome}

The verdict is computed from the output, so it is post-processing. Post-processing cannot increase a Rényi divergence ([Divergences](../part01-prerequisites/03-divergences.md)), because merging outputs into two verdicts can only lose the information that tells the worlds apart. So at every order, the Rényi divergence between the two coins is at most $\varepsilon(\alpha)$, in both directions. Writing out the divergence of the two coins turns this into two inequalities on the errors for each order:
$$a^\alpha(1 - b)^{1 - \alpha} + (1 - a)^\alpha b^{1 - \alpha} \le e^{(\alpha - 1)\varepsilon(\alpha)}, \qquad (1 - b)^\alpha a^{1 - \alpha} + b^\alpha(1 - a)^{1 - \alpha} \le e^{(\alpha - 1)\varepsilon(\alpha)}.$$ {#eq-conv-region}
The pairs of errors that satisfy these inequalities at every order of the grid form the **RDP privacy region** (Balle et al. 2020). Write $f_{\mathrm{RDP}}(a)$ for the smallest type II error in the region at type I error $a$. Every test has its errors in the region, so no test at false-positive rate $a$ has a power above $1 - f_{\mathrm{RDP}}(a)$.

The orders guard different parts of the region. A test with a tiny false-positive rate can say "used" only on rare outputs with extreme losses, and large orders weigh such outputs heavily. So large orders limit tests with small false-positive rates, while small orders limit the bulk. For example, take the Gaussian at $\mu = 1$, and solve the inequalities of @eq-conv-region exactly for one order at a time, starting with a test whose false-positive rate is $10^{-6}$. Order 2 alone allows it a power of at most $1.3118\times10^{-3}$. Order 8, by contrast, holds it to $1.8622\times10^{-4}$. At a false-positive rate of 0.1, however, the roles swap. There the smaller order allows a power of 0.4932, while the larger one allows 0.9838. The exact limits of the two orders cross near a false-positive rate of $1.87\times10^{-4}$.

How close is the region to the truth? For the Gaussian, the best test simply checks whether the output is above a threshold, by the Neyman–Pearson lemma. Its type II error at each false-positive rate is the Gaussian's **trade-off function**, which describes the most powerful attack possible. @fig-conv-tradeoff and @tbl-conv-power compare this best test with the region. The region allows 1.24 to 1.36 times the true power at every false-positive rate shown, so it is a good but loose summary of the trade-off function.

![Largest power $1 - b$ at type I error $a$ allowed by the Gaussian's RDP curve at $\mu = 1$ (red), by orders 2 and 8 alone (blue), and achieved by the Gaussian itself (black). The right panel uses log axes and adds the $(\varepsilon, \delta)$ line of the improved conversion (grey), which touches the red curve.](figures/rdp_conversions_tradeoff.svg){#fig-conv-tradeoff}

| Type I error $a$ | True power | Largest power allowed by the RDP curve | Ratio |
| --- | --- | --- | --- |
| 0.1 | $3.8914\times10^{-1}$ | $4.8376\times10^{-1}$ | 1.24 |
| $10^{-3}$ | $1.8298\times10^{-2}$ | $2.4910\times10^{-2}$ | 1.36 |
| $10^{-6}$ | $8.7218\times10^{-5}$ | $1.1636\times10^{-4}$ | 1.33 |

: Power of the best test against the Gaussian mechanism at $\mu = 1$, true and allowed by its RDP curve. {#tbl-conv-power}

An $(\varepsilon, \delta)$ guarantee constrains tests far more crudely. It says that every test has a power of at most $e^\varepsilon a + \delta$, which is one straight line in the plane of errors. A straight line can touch the curved region at only one point, so it lies far above the region everywhere else. That is the grey line in @fig-conv-tradeoff. Conversely, a whole trade-off function can be described by many such lines, one for each $\varepsilon$, which is the idea that [f-DP and Gaussian differential privacy](04-gaussian-dp.md) develops.

The region is what RDP implies about tests, but it is not an equivalent definition of RDP. A test squeezes the output into a single verdict, and information spread over many outputs is lost when they are merged. So a mechanism can keep every test inside the region and still break the RDP curve.

The region also gives the best possible conversion from the curve, because $\delta$ is itself a statement about tests. Recall that $\delta$ is the largest excess of a set's probability without your record over $e^\varepsilon$ times its probability with it. For any test, the outputs on which it says "not used" form a set with probability $1 - a$ without your record and $b$ with it. So $\delta$ is the largest value of $1 - a - e^\varepsilon b$ over all tests. Every test has $b$ at least $f_{\mathrm{RDP}}(a)$, so
$$\delta(\varepsilon) \le \delta_{\mathrm{opt}}(\varepsilon) = \max_a\big[1 - a - e^\varepsilon f_{\mathrm{RDP}}(a)\big].$$ {#eq-conv-opt}
No conversion from the same RDP values can certify less, because some mechanism with only two outputs reaches this bound. Its output is just the coin of the test that attains the maximum, and that coin satisfies the curve because its errors lie in the region. The full edition's script computes $\delta_{\mathrm{opt}}$ numerically, and @sec-conv-slack uses it to measure the improved conversion.

The gap between the region and a single $(\varepsilon, \delta)$ pair matters for audits. Suppose an auditor attacks the mean released at $\sigma = 1$ with a false-positive rate of $10^{-3}$. The RDP curve guarantees that no test then reaches a power above $2.4910\times10^{-2}$, and the best test against the Gaussian actually reaches $1.8298\times10^{-2}$. The improved conversion's guarantee of $(4.729, 10^{-5})$ says much less, because it only limits the power to $e^\varepsilon a + \delta$. Here that limit is 0.1131, more than four times the region's bound. So to translate an RDP result into attack risk, use the region of @eq-conv-region, or better the trade-off function, rather than one $(\varepsilon, \delta)$ pair. At the low false-positive rates of membership attacks, a single pair throws away most of what the curve says.

## Choosing the order {#sec-conv-order}

Both conversions hold at every order, so you take the order that gives the smallest $\varepsilon'$. Each converted value is the curve, which grows with the order, plus a term in $\ln(1/\delta)/(\alpha - 1)$, which shrinks with it. The best order balances the two (@fig-conv-orders).

![Converted $\varepsilon$ at $\delta = 10^{-5}$ for each order of the grid, for the Gaussian at $\mu = 1$, with Mironov's conversion (red) and the improved one (blue). Dots mark the best order of each. The dashed line is the exact $\varepsilon$, and the grey curve is the RDP curve $\varepsilon(\alpha) = \alpha/2$ that both conversions add to.](figures/rdp_conversions_orders.svg){#fig-conv-orders}

The composed Gaussian has a curve that is a straight line in the order. Its slope, written $\rho$, is $\mu^2/2$, which equals $k/(2\sigma^2)$. Setting the derivative of @eq-conv-improved to zero gives the best order of the improved conversion as the solution of
$$\rho(\alpha - 1)^2 = \ln\frac{1}{\alpha\delta}.$$ {#eq-conv-alpha-star}
Mironov's conversion leads to the same equation without the $\alpha$ inside the logarithm, which is why the improved conversion's best order is slightly lower. The logarithm of $\alpha$ changes slowly, so dropping it gives the approximation
$$\alpha^* - 1 \approx \sqrt{\frac{\ln(1/\delta)}{\rho}} = \sigma\sqrt{\frac{2\ln(1/\delta)}{k}}.$$

The approximation says how the best order moves. More steps or less noise make the curve steeper, so high orders become expensive and the best order moves towards 1. A smaller $\delta$ pushes it up, but only through the square root of a logarithm, so $\delta$ matters little. For example, squaring a $\delta$ of $10^{-5}$ doubles $\ln(1/\delta)$. Yet for the single release, that moves the best order only from 5.432 to 7.483.

@tbl-conv-orders shows the range of orders a grid must cover.

| $\sigma$ | $k$ | $\rho$ | Best order, improved (continuous / grid) | $\varepsilon$ (continuous / grid) | Mironov's best order |
| --- | --- | --- | --- | --- | --- |
| 10 | 1 | 0.005 | 40.525 / 41 | 0.375 / 0.375 | 48.985 |
| 1 | 1 | 0.5 | 5.432 / 5.4 | 4.728 / 4.729 | 5.799 |
| 1 | 100 | 50 | 1.472 / 1.5 | 96.035 / 96.116 | 1.480 |
| 1 | 10,000 | 5000 | 1.048 / 1.1 | 5475.791 / 5611.778 | 1.048 |

: Best orders for the composed Gaussian at $\delta = 10^{-5}$. {#tbl-conv-orders}

Long runs with little noise need orders close to 1. Take the third row of the table, 100 steps at $\sigma = 1$. Its best order is 1.472, and the grid's nearest order costs only 0.081 in $\varepsilon$. With 10,000 steps the best order falls to 1.048, which is below the grid's smallest order. There the grid's answer is 135.987 too large, in a setting whose $\varepsilon$ is huge anyway.

Short runs with much noise need large orders. A single release at $\sigma = 10$ already wants order 40.525. For a straight-line curve with a slope $\rho$ as small as $5\times10^{-5}$, the best order is 338.312. Subsampled steps ([Amplification by subsampling](05-subsampling.md)), which use each record in only a random fraction of the steps, also have tiny curves at small orders and therefore want large ones. That is why the grid ends with orders as large as 512.

So if an accountant reports its best order at either end of its grid, the true best order lies outside the grid. The reported $\varepsilon$ is then larger than necessary, and you should extend the grid in that direction.

## The slack no conversion removes {#sec-conv-slack}

The improved conversion still stays above the exact $\varepsilon$ of the Gaussian. The question is whether a cleverer formula could close the gap, or whether the RDP curve itself lacks the information. It is the curve, and a mechanism with only two outputs shows why (@tbl-conv-worst-pair).

| Output | Probability under $P$ | Probability under $Q$ | Privacy loss $L$ |
| --- | --- | --- | --- |
| accept | $2.2779\times10^{-4}$ | $2.2998\times10^{-6}$ | 4.5956 |
| reject | rest | rest | $-2.255\times10^{-4}$ |

: A two-outcome mechanism that satisfies the Gaussian's RDP curve at $\mu = 1$ at every order and has nearly the largest $\delta$ at $\varepsilon = 4.377$ that any such mechanism can have. {#tbl-conv-worst-pair}

This mechanism almost always produces an output with a loss close to 0, and very rarely one with a loss of 4.5956. The full edition checks numerically that its Rényi divergence stays below the Gaussian's curve at $\mu = 1$, at every real order. Yet its $\delta$ is much larger than the Gaussian's. Take $\varepsilon = 4.377$, the Gaussian's exact value at $\delta = 10^{-5}$. Only the rare output has a loss above $\varepsilon$, so by @eq-conv-delta only it contributes. Its contribution makes the two-outcome mechanism's $\delta$ equal to $4.4698\times10^{-5}$, about 4.5 times the Gaussian's.

A certificate computed from the curve alone cannot tell the two mechanisms apart, so it must hold for both. That means it cannot certify 4.377 for the Gaussian. The rare output sits near the point where the improved conversion's exponential touches the true contribution in @fig-conv-majorants, at the order that limits this mechanism, about 5.1. For a loss concentrated there, the improved bound is nearly exact, while the Gaussian spreads its tail smoothly over all large losses. The curve sees only exponential moments, and they cannot tell these two shapes apart.

@tbl-conv-slack and @fig-conv-slack compare all four answers for the Gaussian with $\sigma = 10$ per step, for runs of 1 to 10,000 steps. Because everything depends only on $\mu$, the row with 100 steps repeats the single release at $\sigma = 1$ of the earlier sections.

| $k$ | $\mu$ | Exact | Optimal from the curve | Improved | Mironov |
| --- | --- | --- | --- | --- | --- |
| 1 | 0.100 | 0.341 | 0.375 | 0.375 | 0.485 |
| 10 | 0.316 | 1.199 | 1.308 | 1.308 | 1.568 |
| 100 | 1.000 | 4.377 | 4.729 | 4.729 | 5.299 |
| 1000 | 3.162 | 17.857 | 19.054 | 19.054 | 20.175 |
| 10,000 | 10.000 | 91.817 | 96.116 | 96.116 | 98.026 |

: $\varepsilon$ at $\delta = 10^{-5}$ for the Gaussian with $\sigma = 10$ composed $k$ times: exact, the best any conversion from the RDP grid can give (@eq-conv-opt), the improved conversion and Mironov's. {#tbl-conv-slack}

![Left: $\varepsilon$ at $\delta = 10^{-5}$ for the Gaussian with $\sigma = 10$ composed $k$ times, exact (black), optimal from the RDP curve (wide green), improved (blue, on top of the green) and Mironov (red). Right: each divided by the exact value. The improved conversion is as good as any conversion from the curve. The remaining 5 to 10% belongs to the RDP summary itself.](figures/rdp_conversions_slack.svg){#fig-conv-slack}

The optimal conversion and the improved one agree to three decimals at every run length, and at $\mu = 1$ they differ by $2\times10^{-6}$. The optimal conversion may use all orders at once, but that helps only when several orders constrain the worst mechanism together. Here a single order does all the work. Optimal and improved both exceed the exact value by 10% for a single step and by 5% for the longest run. Mironov's excess shrinks faster, from 42% to 7%. All of them shrink in relative terms because, for large $\mu$, the curve value $\rho\alpha$, which every conversion shares, makes up most of every answer.

So suppose a reviewer asks whether a better conversion would lower your $(4.729, 10^{-5})$ for a Gaussian with $\mu = 1$. The answer is no. The best value any method can certify from the curve is 4.7284 to four decimals, which the improved conversion already reaches up to rounding. A lower number needs an accountant that sees the mechanism itself, and for the Gaussian that number is 4.377.

When the slack matters, change the summary rather than the formula. To reach the Gaussian's exact 4.377, an accountant must track more than exponential moments of the loss. It can track the trade-off function ([f-DP and Gaussian differential privacy](04-gaussian-dp.md)) or the whole distribution of the privacy loss ([Numerical accounting with privacy loss distributions](06-numerical-accounting.md)).

The 5 to 10% holds for the plain Gaussian, though. The differentially private version of stochastic gradient descent (SGD), DP-SGD, composes subsampled Gaussian steps, each of which uses only a random fraction of the records. Their curve is not a straight line, and the gap to the exact value can be much larger. For example, the DP-SGD run that serves as this part's running example uses a fraction of 0.01 in each of 10,000 steps. With a noise multiplier of 1, Mironov's conversion gives 7.43 for this run. The improved conversion, by contrast, gives 6.71, and [Choosing an accountant](09-choosing-an-accountant.md) measures how far both are from the exact value.

Exponential moments also decide what happens when you mix RDP with other kinds of guarantee. An $(\varepsilon, \delta)$ guarantee with $\delta$ above 0 implies no RDP bound at any order, and one mechanism shows why. It outputs nothing, except that with probability $\delta$ it publishes your record. This mechanism is $(0, \delta)$-DP, because moving a probability of $\delta$ can change the chance of any event by at most $\delta$. But the published record is an output that is impossible without you, so its likelihood ratio is infinite. Swap the roles of the two datasets, as the guarantee must also allow (@sec-conv-loss). The exponential moment is then infinite, and so is the Rényi divergence at every order.

RDP, like pure DP, rules out such catastrophic outputs however rare they are, while $(\varepsilon, \delta)$-DP allows them with probability $\delta$. So never feed an $(\varepsilon, \delta)$ component into an RDP accountant as if it had a curve. You need to know how its $\delta$ arises, for example from Gaussian noise, and take the curve from that.

A pure $\varepsilon$-DP guarantee does give a curve, because its privacy loss never exceeds $\varepsilon$. Then every exponential moment is at most $e^{(\alpha - 1)\varepsilon}$, so the curve is at most $\varepsilon$ at every order. Bun & Steinke (2016) also prove a second bound that is smaller at low orders, and together the two give
$$\varepsilon(\alpha) \le \min\Big(\varepsilon, \frac{\alpha\varepsilon^2}{2}\Big).$$
[Concentrated differential privacy](03-concentrated-dp.md) then sharpens the constant $\varepsilon^2/2$ to $\varepsilon\tanh(\varepsilon/2)$. For example, take randomized response with $\varepsilon = 0.5$, and look at order 2. There the bound of Bun & Steinke is 0.2500, and the true Rényi divergence is just under it, at 0.2273.

## Code {#sec-conv-code}

Both conversions fit in a few lines. This snippet converts the composed Gaussian's curve on the part's grid of orders. It does so for the two cases of @tbl-conv-mironov-steps, the single release and 100 steps at $\sigma = 1$.

```python
from math import log

ORDERS = [1 + x / 10 for x in range(1, 100)] + list(range(12, 64)) + [128, 256, 512]


def mironov(rdp: float, alpha: float, delta: float) -> float:
    return rdp + log(1 / delta) / (alpha - 1)


def improved(rdp: float, alpha: float, delta: float) -> float:
    """Balle et al. (2020); Canonne, Kamath & Steinke (2020). Clipped at 0."""
    return max(0.0, rdp + log((alpha - 1) / alpha) - (log(delta) + log(alpha)) / (alpha - 1))


def report(mu: float, delta: float = 1e-5) -> str:
    """Both conversions for a composed Gaussian with mu = sqrt(steps) / sigma."""
    curve = [a * mu**2 / 2 for a in ORDERS]
    mir = min((mironov(e, a, delta), a) for a, e in zip(ORDERS, curve))
    imp = min((improved(e, a, delta), a) for a, e in zip(ORDERS, curve))
    return "Mironov %.3f at order %g, improved %.3f at order %g" % (*mir, *imp)


print(report(1.0))  # Mironov 5.299 at order 5.8, improved 4.729 at order 5.4
print(report(10.0))  # Mironov 98.026 at order 1.5, improved 96.116 at order 1.5
```

To convert any other mechanism, replace the curve by its RDP values on the same grid. The full edition's script, [rdp_conversions.py](../../code/part04-privacy-accounting/rdp_conversions.py), also computes the exact Gaussian values and the optimal conversion. It builds the two-outcome mechanism and the bound on tests as well:

```bash
uv run python code/part04-privacy-accounting/rdp_conversions.py
```

## Cheat sheet

| Result | Formula | Notes |
| --- | --- | --- |
| $\delta$ as an average | $\delta(\varepsilon) = \mathbb E_P[(1 - e^{\varepsilon - L})_+]$ | Each output pays only its excess |
| RDP as exponential moments | $\mathbb E_P[e^{(\alpha - 1)L}] \le e^{(\alpha - 1)\varepsilon(\alpha)}$ | All a conversion can use |
| Mironov's conversion | $\varepsilon' = \varepsilon(\alpha) + \ln(1/\delta)/(\alpha - 1)$ | Valid but loose |
| Improved conversion | $\varepsilon' = \varepsilon(\alpha) + \ln\frac{\alpha - 1}{\alpha} - \frac{\ln\delta + \ln\alpha}{\alpha - 1}$ | Use this; clip at 0 |
| Saving at a fixed order | $\Delta(\alpha) = \frac{\ln\alpha}{\alpha - 1} + \ln\frac{\alpha}{\alpha - 1}$ | Largest at small orders |
| Test region | $a^\alpha(1 - b)^{1 - \alpha} + (1 - a)^\alpha b^{1 - \alpha} \le e^{(\alpha - 1)\varepsilon(\alpha)}$, and with $a$ and $1 - b$ swapped | Bounds every attack's power |
| Optimal conversion | $\delta_{\mathrm{opt}} = \max_a[1 - a - e^\varepsilon f_{\mathrm{RDP}}(a)]$ | Attained by a two-outcome mechanism |
| Best order, composed Gaussian | $\rho(\alpha - 1)^2 = \ln\frac{1}{\alpha\delta}$, with $\rho = k/(2\sigma^2)$ | Moves towards 1 as $k$ grows |
| Unavoidable slack | 5 to 10% of $\varepsilon$ for the Gaussian | Larger for subsampled DP-SGD |
| $(\varepsilon, \delta)$ to RDP | Impossible for $\delta > 0$ | Need the mechanism, not the pair |
| $\varepsilon$-DP to RDP | $\varepsilon(\alpha) \le \min(\varepsilon, \alpha\varepsilon^2/2)$ | Bun & Steinke 2016 |

: The results of this chapter. {#tbl-conv-cheat}

**Rules of thumb**

- Always use the improved conversion, which is free.
- Check the best order. If it sits at either end of the grid, extend the grid.
- Expect an RDP accountant to overstate the plain Gaussian's $\varepsilon$ by 5 to 10%, and subsampled DP-SGD's by more. If that matters, use a privacy loss distribution accountant.
- Read an RDP result as a bound on attacks through the whole curve, not through one $(\varepsilon, \delta)$ pair.

## Further reading

- Mironov, "Rényi Differential Privacy" (IEEE CSF 2017): the original conversion, Proposition 3.
- Balle, Barthe, Gaboardi, Hsu & Sato, "Hypothesis Testing Interpretations and Rényi Differential Privacy" (AISTATS 2020): the improved conversion as a tangent line, and the region an RDP curve puts on tests.
- Canonne, Kamath & Steinke, "The Discrete Gaussian for Differential Privacy" (NeurIPS 2020): the short proof of the improved conversion, Proposition 12.
- Steinke, "Composition of Differential Privacy & Privacy Amplification by Subsampling" (2022 survey): privacy loss, RDP and conversions in one place, with their tightness.

**Derivations.** Every result in this chapter is derived step by step in [the full edition's chapter](../../book-full/part04-privacy-accounting/02-rdp-conversions.md), except the two bounds for pure DP, $\alpha\varepsilon^2/2$ from Bun & Steinke (2016) and the sharper $\alpha\varepsilon\tanh(\varepsilon/2)$. It uses both without proof, and [the full edition's chapter on concentrated DP](../../book-full/part04-privacy-accounting/03-concentrated-dp.md#sec-cdp-pure) derives them.
