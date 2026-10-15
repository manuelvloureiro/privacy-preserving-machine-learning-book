# Discrete and floating-point-safe mechanisms

## Why this matters

The proof of the [Laplace mechanism](03-laplace-gaussian.md) compares two densities on the real line. A computer has no real numbers to work with. It has floating-point numbers, which form a finite set with uneven gaps between its members, and a source of random bits. So the function that claims to sample Laplace noise produces some distribution on that finite set, and the proof says nothing about that distribution. Mironov (2012) showed that this mismatch between proof and code is more than a rounding nuisance. The textbook Laplace sampler, run in double precision, leaks the true answer through the last few bits of its output.

The idea that organizes this chapter is that **differential privacy (DP) is a property of the output distribution that the code actually produces**. So the code must produce exactly the distribution that the proof is about, and there are two ways to get there. The first keeps floating point and repairs the output. It rounds the output to a coarse grid and clamps it to a fixed range, so that every output can be reached from every input. The next section shows why privacy needs that. The second leaves floating point behind. It makes the query integer-valued and samples integer noise exactly, using only integer and fraction arithmetic and uniform random integers. That route gives the discrete Laplace and discrete Gaussian mechanisms. Their privacy proofs are as clean as the continuous ones, and their samplers produce exactly the distribution in the proof.

## Floating-point Laplace breaks differential privacy {#sec-discrete-float}

Recall the definition from [The definition of differential privacy](01-definition.md). A mechanism $M$ is a randomized program that takes a dataset and returns an output. Write $D$ for a dataset and $D'$ for a neighbouring dataset, one that differs from it in one person's record. Write $\varepsilon$ for the privacy budget and $\delta$ for a small additive slack. The mechanism is $(\varepsilon, \delta)$-DP if, for every pair of neighbours and every set $S$ of outputs,
$$\Pr[M(D) \in S] \le e^\varepsilon\Pr[M(D') \in S] + \delta.$$
In words, no event becomes more than $e^\varepsilon$ times more likely when one record changes, apart from the slack $\delta$. With $\delta = 0$ the guarantee is called pure DP.

Now suppose that some outputs can be produced from $D$ but never from $D'$, and let $S$ be the set of those outputs. Seeing one of them tells an observer that the data were $D$, with certainty. For this set the right side of the inequality is $e^\varepsilon \cdot 0 + \delta$, because $M(D')$ never lands in $S$. The factor $e^\varepsilon$ multiplies zero, so it is useless however large $\varepsilon$ is, and the whole probability of those outputs has to be paid for by $\delta$. This is the **support argument**, which says that for every $\varepsilon$,
$$\delta \;\ge\; \Pr[M(D) \in S].$$ {#eq-support-delta}
So a single output that only one input can reach rules out pure DP. Finding such outputs needs no statistics at all, only a list of what each input can produce, which makes the support argument the quickest audit of an implementation.

The textbook sampler fails this audit. It draws a number $U$ uniformly between 0 and 1, and a fair random sign $\xi$, which is plus or minus one. It then outputs $f(D) + \xi\, b\ln(1/U)$, where $f(D)$ is the true answer and $b$ is the noise scale. The added term is exactly Laplace noise with scale $b$ in exact arithmetic, which is why textbooks use this sampler. In floating point, though, the logarithm and the sum are each rounded to the nearest float.

The rounding is where the leak comes from, because floats are not evenly spread. Between one power of 2 and the next they are evenly spaced, and the spacing doubles at each power of 2. So floats are dense near 0 and sparse far from it.

Take $f(D) = 0$ and $f(D') = 1$. Under $D$, an output near 0 is simply a noise value near 0, and it sits on the fine grid of floats there. Under $D'$, whose answer is 1, the same output needs a noise value near $-1$. Noise values that large sit on the coarser grid of floats near 1. The output, 1 minus such a value, is then a multiple of that coarser spacing. So most of the finely spaced outputs that $M(D)$ produces near 0 cannot come from $M(D')$. Away from the two inputs the two grids interleave, so there too the two sets of outputs overlap only in part.

Doubles have too many values to list, so the full edition studies a toy model with shorter floats. Their significand, the stored binary digits of the number, which is often called the mantissa, has 6 to 14 bits. These are small enough to enumerate every output and its probability exactly. @fig-discrete-float-support shows the result for 12 bits, with noise scale $b = 1$ and the two answers above. There the outputs that only $D$ can produce carry probability 0.4791. More precision does not help, because extra bits refine both grids together and the mismatch stays. At every precision the full edition tries, about 48% of the probability of $M(D)$ sits on outputs that $M(D')$ cannot produce. Mironov found the same effect at the 53 bits of a double, in the samplers of several DP libraries of the time. He also showed how to recover $f(D)$ from single outputs.

![The toy 12-bit floating-point Laplace sampler with $b = 1$. Left: the outputs that $f(D) = 0$ (top) and $f(D') = 1$ (bottom) can produce in a narrow window, coloured by whether the other input can produce them too. Right: the probability that $M(D)$ produces an output $M(D')$ cannot, against the number of significand bits, for the textbook sampler and for the snapped version of the next section.](figures/discrete_float_support.svg){#fig-discrete-float-support}

Here is what that means for a release. A hospital publishes the number of patients with a rare diagnosis, a count that one patient changes by at most 1. It uses the textbook sampler with $b = 1$, so the exact mechanism would be DP with $\varepsilon = 1$. An analyst knows that the count is either 0 or 1, depending on whether one particular patient is in the data. So the analyst checks whether the released float can be produced from a count of 1. Suppose the true count is 0. In the 12-bit toy, the check then identifies it with certainty 47.91% of the time. Yet the proof promised that one output could move the analyst's odds by at most a factor $e$.

The tails leak too, though far less. A 53-bit uniform $U$ is a whole multiple of its smallest step, $2^{-53}$, so it can never be smaller than that step. As a result the noise magnitude never exceeds $b\ln(2^{53})$, which is about $36.74\,b$. Outputs just beyond $f(D) + 36.74\,b$ can therefore be reached from only one of the two inputs. By @eq-support-delta their probability, about $2^{-53}$, is still a floor on $\delta$, and on its own it rules out pure DP.

The toy is not a real library, since each library draws $U$ and rounds the logarithm in its own way. Those details change the numbers but not the conclusion. Notice also where the leak lives. It is not in where the output lands on the real line but in which float it is, so printing the output to a few decimals would hide it. That observation is the idea behind the first repair.

## The snapping mechanism {#sec-discrete-snapping}

If the output must be a float, the low-order bits have to go. Mironov's **snapping mechanism** removes them by rounding the noisy answer to a grid at least as coarse as the noise scale. Every input then reaches the same finite set of grid points, so no output can be reached from one input only, and the support argument has nothing to find.

The mechanism takes a query with sensitivity 1, a noise scale $\lambda$ and a clamping bound $B$. Write $\Lambda$ for the smallest power of 2 at or above $\lambda$, which will be the spacing of the grid. The mechanism clamps the true answer into the range from $-B$ to $B$, which keeps floating-point errors bounded, and adds Laplace noise of scale $\lambda$, computed in floating point. It then rounds the result to the nearest multiple of $\Lambda$, and clamps it again.

Rounding costs no privacy, and the reason is that each grid point collects a whole interval of noise. All the noise values that round to one grid point form an interval of width $\Lambda$ on the real line. In exact arithmetic, the grid point's probability under each input is the chance that Laplace noise lands in that input's interval. The two intervals are shifted against each other by at most one unit, because the query has sensitivity 1. The Laplace density changes by at most a factor $e^{1/\lambda}$ under such a shift, so the two probabilities differ by at most that factor too. In other words, rounding exact Laplace noise is post-processing of the exact Laplace mechanism. [Post-processing](02-properties.md) never increases the **privacy loss**, the log of the ratio of an output's probabilities under two neighbours.

What remains is the floating-point error, which moves the ends of each interval slightly. Write $\eta = 2^{-53}$ for the relative error of one floating-point operation, the largest fraction of its result by which it can be off. Clamping keeps the numbers the mechanism works with bounded by $B$ in size, so these errors stay small next to the grid. Mironov (2012, Theorem 1) bounds their effect for a clamping bound between $\lambda$ and $2^{46}\lambda$, the range his analysis covers. Under that condition the snapping mechanism is $\varepsilon$-DP with
$$\varepsilon = \frac{1 + 12B\eta}{\lambda} + \frac{2^{-49}B}{\lambda}.$$
The main part, $1/\lambda$, is the guarantee of the exact Laplace mechanism, and the two extra terms pay for floating-point error. Both grow with the clamping bound, but for doubles they are tiny. With a noise scale of 1 and a clamping bound of 10, the excess over the exact mechanism's guarantee is about $3.1\times10^{-14}$.

The toy model shows the repair at work. Snap the toy sampler to the integers and clamp it to the range from $-10$ to 10. Both inputs then reach the same 21 outputs at every precision, namely the integers in that range (@fig-discrete-float-support, right). Over these outputs the largest privacy loss is 1.0010 at 12 bits, just above the exact mechanism's guarantee. At 14 bits it falls to 1.0002, because the excess shrinks as precision grows, as the $\eta$ terms predict.

The costs follow from the construction. Rounding adds at most half a grid spacing of error, which is at most $\lambda$ because the spacing $\Lambda$ is below $2\lambda$. Clamping biases answers near $\pm B$, so choose $B$ from the range the answer can take, knowing that $\varepsilon$ grows with it. The guarantee also assumes an ideal uniform $U$ and a logarithm as accurate as the proof requires, which the library has to supply. Finally, the post-processing argument above covers exact Laplace noise, not the floating-point sampler, because post-processing a mechanism that is not DP proves nothing. Mironov's proof instead bounds how far the snapped floating-point output can be from snapped exact noise. It needs the grid to be at least as coarse as $\lambda$ to do so.

## The discrete Laplace mechanism {#sec-discrete-laplace}

Most DP queries are counts and histograms. They take integer values, and one record changes a count by at most 1. For such queries the natural noise is integer-valued too. The output is then an integer with no low-order bits to leak, and the noise can be sampled exactly.

The **discrete Laplace** distribution is built the way its name suggests. Take the Laplace density, keep its values at the integers, and divide by their sum so that they add up to 1. Each step away from 0 then multiplies the probability by the same ratio $r$, equal to $e^{-1/b}$, just as the continuous density drops by that factor per unit. This distribution is written $\mathrm{Lap}_{\mathbb Z}(b)$, for a scale $b > 0$. Its probability mass function (PMF), which gives the probability of each integer $z$, is
$$\Pr[Z = z] = \frac{1 - r}{1 + r}\,r^{|z|}, \qquad r = e^{-1/b}.$$ {#eq-dlap-pmf}
The fraction in front is the normalizing constant, one over the sum of the weights $r^{|z|}$ over all integers.

Adding this noise to an integer-valued query is the **geometric mechanism** of Ghosh, Roughgarden and Sundararajan (2009). Its privacy argument is the continuous one, under one condition. The query must be integer-valued, because otherwise the noise needed to reach an output would not be an integer, as the next paragraph shows.

To see why, take an output $y$ and compare its probability under two neighbours. Write $a$ for the query's answer on the first and $a'$ for its answer on the second. The noise needed is $y - a$ under the first and $y - a'$ under the second. Both noise values are integers because $a$ and $a'$ are integers, so both probabilities come from @eq-dlap-pmf with the same normalizing constant. That constant cancels in their ratio, and the privacy loss is the difference between the two distances, $|y - a'| - |y - a|$, divided by $b$. Two numbers' distances to $y$ can differ by at most the distance between the two numbers, so the privacy loss is at most $|a - a'|/b$.

The same holds for a vector-valued query with independent noise on each coordinate, because the privacy loss of a vector is the sum of its coordinates' losses. Write $\Delta_1 f$ for the query's $\ell_1$ sensitivity, the largest total absolute change across its coordinates when one record changes ([Sensitivity](06-sensitivity.md)). The geometric mechanism is then $\varepsilon$-DP with
$$\varepsilon = \frac{\Delta_1 f}{b},$$ {#eq-dlap-eps}
the same calibration as the continuous Laplace mechanism. The bound is tight, since it is reached at every output at or above both answers. If an answer were not an integer, the shifted noise would leave the integers, and the two inputs would reach different sets of outputs. That is the support problem of @sec-discrete-float again.

![Left: the PMF of $\mathrm{Lap}_{\mathbb Z}(1.5)$ on a log scale, exact and from 10,000 draws of the exact sampler, with the $\mathrm{Lap}(1.5)$ density for comparison. Right: the privacy loss between counts 0 and 1, a step function bounded by $\varepsilon = 2/3$.](figures/discrete_laplace.svg){#fig-discrete-laplace}

The discrete Laplace gives the same guarantee with slightly less noise, because its variance $2r/(1 - r)^2$ is a little below the variance $2b^2$ of continuous Laplace noise at the same scale. For example, a count released at $\varepsilon = 1$ needs the scale $b = 1$. The ratio $r$ is then $e^{-1} = 0.3679$. The exact count comes out whenever the noise is 0, which by @eq-dlap-pmf happens with probability 0.4621. The noise has variance 1.8413 against 2 for continuous noise, so its standard deviation is 1.357. Because the output is an integer sampled exactly, this analysis is an analysis of the code itself.

For a single count, nothing does better. Ghosh, Roughgarden and Sundararajan (2009) show that the geometric mechanism is universally utility-maximizing. Every user, whatever their loss function and their prior beliefs, can post-process its output to do as well as with any other $\varepsilon$-DP mechanism.

You might instead round $f(D)$ plus continuous Laplace noise to the nearest integer. In exact arithmetic that is also $\varepsilon$-DP, as post-processing of the Laplace mechanism. But it is a different distribution, and computing it needs a floating-point Laplace sampler, which is the thing this chapter set out to avoid.

## The discrete Gaussian mechanism {#sec-discrete-gaussian}

The Gaussian mechanism is the workhorse of private machine learning ([The Laplace and Gaussian mechanisms](03-laplace-gaussian.md)). Its noise grows with the $\ell_2$ sensitivity rather than the $\ell_1$ sensitivity, and sums of Gaussians stay Gaussian, so noise added in pieces is still Gaussian noise. Canonne, Kamath and Steinke (2020) define its integer analogue and analyse its privacy. The US Census Bureau used it for the 2020 Census redistricting data (Abowd et al. 2022). It is also the noise of choice when clients add noise before secure aggregation. That is a protocol that reveals only the sum of the clients' values, and it works with integers modulo a fixed number (Kairouz, Liu & Steinke 2021).

The construction is the same as for the discrete Laplace. Take the Gaussian bell curve, keep its values at the integers and normalize them. The **discrete Gaussian** has one parameter, a width $\sigma$ greater than zero. It gives each integer $x$ a probability proportional to $e^{-x^2/(2\sigma^2)}$, and it is written $\mathcal N_{\mathbb Z}(0, \sigma^2)$. Here $\sigma^2$ is the variance of the continuous bell curve, which is not quite the variance of the discrete distribution, as the end of this section shows. The mechanism adds a draw $X$ of this noise to an integer-valued query. Write $\Delta$ for the integer amount by which the query's answer differs between two neighbours.

The privacy loss is the continuous Gaussian's, evaluated at the integers. The reason is the one that worked for the discrete Laplace. Integer shifts leave the normalizing constant unchanged, so the log-ratio of two shifted PMFs is the log-ratio of the two bell curves. Take the answers to be 0 under $D$ and $\Delta$ under $D'$. The privacy loss of an integer output $y$ is then
$$L(y) = \frac{\Delta^2 - 2\Delta y}{2\sigma^2},$$
a straight line in $y$, exactly as for the continuous Gaussian.

The smallest $\delta$ for a given $\varepsilon$ follows from this line. It is the probability under $D$ of the outputs whose privacy loss exceeds $\varepsilon$, minus $e^\varepsilon$ times their probability under the neighbour. This is the hockey-stick divergence of [Divergences](../part01-prerequisites/03-divergences.md).

Because the loss is a straight line that falls as $y$ grows, the outputs whose loss exceeds $\varepsilon$ are the integers below a threshold. The discrete Gaussian is symmetric around 0, so each of these lower tails can be written as an upper tail of $X$. The exact $\delta(\varepsilon)$ is therefore a difference of two tail probabilities. Theorem 7 of Canonne, Kamath and Steinke (2020) gives it as
$$\delta(\varepsilon) = \Pr\Big[X > \frac{\varepsilon\sigma^2}{\Delta} - \frac\Delta2\Big] - e^\varepsilon\Pr\Big[X > \frac{\varepsilon\sigma^2}{\Delta} + \frac\Delta2\Big], \qquad X \sim \mathcal N_{\mathbb Z}(0, \sigma^2).$$ {#eq-dgauss-delta}
The first term is the probability of the outputs that favour $D$ by more than $e^\varepsilon$, and the second subtracts the part of it that the factor $e^\varepsilon$ already covers. This is the same expression as the continuous Gaussian's exact curve, with the tails of the discrete distribution in place of the continuous ones.

Same form does not mean same value, though. Because $y$ moves in whole steps and the line's slope is $\Delta/\sigma^2$, the privacy loss takes only values spaced $\Delta/\sigma^2$ apart. As $\varepsilon$ crosses one of these values, a whole output enters or leaves the tail. So the discrete curve has a kink at each of them, and it oscillates around the continuous curve (@fig-discrete-gaussian-delta).

@tbl-discrete-delta-compare compares the two curves for sensitivity 1 at $\varepsilon = 1$. At all three widths the discrete Gaussian's $\delta$ is the larger one, and at this $\varepsilon$ the gap narrows as $\sigma$ grows. Just past a kink, though, the discrete curve dips below the continuous one, so neither curve dominates the other. The practical consequence is that the continuous formula can understate the discrete Gaussian's $\delta$, so compute @eq-dgauss-delta directly, as @sec-discrete-code does.

| Width $\sigma$ | Discrete Gaussian $\delta$ | Continuous Gaussian $\delta$ |
| --- | --- | --- |
| 0.5 | 0.6032 | 0.5099 |
| 1 | 0.1414 | 0.1269 |
| 2 | 0.0072 | 0.0068 |

: The smallest $\delta$ at $\varepsilon = 1$ for sensitivity 1, for the discrete and the continuous Gaussian. {#tbl-discrete-delta-compare}

![Discrete against continuous Gaussian for sensitivity 1: $\delta(\varepsilon)$ of the discrete Gaussian (solid) and of the continuous Gaussian (dashed), for $\sigma = 0.5$, 1 and 2. The discrete curve has a kink at each value of its lattice-valued privacy loss.](figures/discrete_gaussian_delta.svg){#fig-discrete-gaussian-delta}

Rényi divergence tells a cleaner story. The **Rényi divergence** of order $\alpha$, a number above 1, averages the privacy loss in a way that weights large losses more as the order grows. [Rényi differential privacy](../part04-privacy-accounting/01-renyi-dp.md) uses it to track composition. For two continuous Gaussians whose means differ by $\Delta$, it equals $\alpha\Delta^2/(2\sigma^2)$, as [Divergences](../part01-prerequisites/03-divergences.md) shows. For two discrete Gaussians whose integer locations differ by $\Delta$, it is never larger than that, and it is equal at every integer order.

The reason is one fact about bell curves summed over the integers. Such a sum is largest when the curve is centred on an integer. The discrete divergence equals the continuous value plus a correction that depends on the ratio of two such sums, one centred on an integer and one possibly off it. That ratio is at most 1, so the correction can only lower the divergence, and at integer orders both centres are integers and the correction vanishes. At order 2.5, which is not an integer, the discrete divergence does fall below the continuous value. For example, it is 4.9808 instead of 5 when $\sigma$ is one half.

The discrete divergence never exceeds the continuous one at any order. So the discrete Gaussian satisfies the same [zero-concentrated DP](../part04-privacy-accounting/03-concentrated-dp.md) guarantee as the continuous one (Canonne, Kamath & Steinke 2020, Theorem 4). That guarantee bounds every Rényi divergence by a constant times the order. So a privacy accountant, the tool that adds up the privacy cost of many releases, can use the continuous Gaussian's bound if it composes Rényi divergences or zero-concentrated DP. An accountant that tracks the exact $\delta(\varepsilon)$ curve cannot, for the reason @tbl-discrete-delta-compare shows.

Restricting the bell curve to the integers also pulls a little probability towards 0, so the variance of the discrete Gaussian is below $\sigma^2$. The deficit shrinks extremely fast as $\sigma$ grows (@fig-discrete-gaussian-variance). At $\sigma = 1$ the relative shortfall is only $2.11\times10^{-7}$, and at twice that width it is at the level of rounding error. The difference becomes visible only below $\sigma$ of about 0.6. At $\sigma = 0.5$, for example, the variance is $0.8601\,\sigma^2$. At a quarter, almost all the probability sits at 0.

![Left: the variance of $\mathcal N_{\mathbb Z}(0, \sigma^2)$ divided by $\sigma^2$, against 1 for the continuous Gaussian. Right: the relative deficit on a log scale.](figures/discrete_gaussian_variance.svg){#fig-discrete-gaussian-variance}

Two consequences matter in machine learning. First, the proof needs an integer-valued query, so a real-valued query such as a clipped gradient must be made integer before noise is added. Divide it by a small step size and round it to an integer. Randomized rounding, which rounds up with probability equal to the fractional part, keeps the result unbiased. But rounding can push two neighbours' values apart, so the rounded query can have a slightly larger sensitivity than the scaled original. That larger sensitivity is the one to calibrate to. Second, unlike continuous Gaussians, the sum of two independent discrete Gaussians is not exactly a discrete Gaussian. For $\sigma \ge 1$ it is very close, and distributed schemes rely on that when each client adds its own share of the noise and only the sum is released.

A histogram shows how the pieces fit. Release a histogram of 1,000 bins, under neighbours that add or remove one record. Such a change moves a single bin by one count, so the $\ell_2$ sensitivity is 1. Add independent $\mathcal N_{\mathbb Z}(0, 100)$ noise to each bin. The noise in the 999 unchanged bins has the same distribution under both neighbours, so those bins add nothing, and the privacy loss is that of one bin. Its $\delta(\varepsilon)$ therefore comes from @eq-dgauss-delta with $\Delta = 1$ and $\sigma = 10$, and at that width it is essentially the continuous Gaussian's.

## Exact coins from uniform integers {#sec-discrete-bernoulli-exp}

The discrete mechanisms are only as exact as their samplers. Both samplers accept or reject draws with probabilities of the form $e^{-\gamma}$, so both need a coin with that bias. Here the exponent $\gamma$ is a rational number, a ratio of two integers, and it is not negative. The obvious code, `random() < exp(-gamma)`, is inexact twice over, because `exp` rounds and `random()` has only 53 bits. Canonne, Kamath and Steinke (2020) build everything instead from one primitive, a uniform random integer below some bound $d$. Python's `random.Random.randrange(d)` provides it exactly, because it draws just enough random bits to cover the range and draws again whenever the result is $d$ or more. So every integer below $d$ comes out with probability exactly $1/d$. A deployment draws the bits from the operating system with `secrets.SystemRandom()`, which has the same interface.

Three simple facts turn this primitive into exact samplers. The first is that a coin with a rational bias is easy. To get 1 with probability $n/d$, draw a uniform integer below $d$ and report 1 when it is one of the first $n$ values. Nothing is rounded, because $n$ favourable values out of $d$ equally likely ones give exactly $n/d$. A coin with an irrational bias such as $e^{-1}$ cannot be made this way, because no ratio of integers equals it.

The second fact is **rejection sampling**, which reshapes one distribution into another by throwing some draws away. Suppose you can sample from a distribution $q$, called the proposal, but you want samples from a target $p$. If the target is proportional to $q$ times some acceptance probability, draw from $q$ and keep each draw with its acceptance probability, repeating until one is kept. Values that the target favours less than the proposal are thrown away more often. No acceptance probability may exceed 1, so the proposal must give every value enough probability relative to the target. As long as each acceptance coin is exact, the kept value then has exactly the target distribution. The method never needs the target's normalizing constant. That constant only sets how often a draw is kept, and the mean number of rounds is one over that rate.

The third fact gives a geometric variable. Flip independent coins that succeed with probability $r$, and count the successes before the first failure. That count is $v$ with probability $r^v(1 - r)$, because it needs $v$ successes and then one failure. This is the geometric distribution of [Probability distributions](../part01-prerequisites/01-distributions.md), counted from 0.

Algorithm 1 of Canonne, Kamath and Steinke makes the coin with probability $e^{-\gamma}$, using a method of von Neumann from 1951. It starts from the power series of the exponential,
$$e^{-\gamma} = 1 - \gamma + \frac{\gamma^2}{2!} - \frac{\gamma^3}{3!} + \cdots,$$
whose terms are probabilities that coins with rational biases can produce.

Take $\gamma$ between 0 and 1 for now. Flip coins whose success probabilities shrink step by step, as $\gamma/k$ for the $k$-th coin, until one fails. Surviving the first $k$ flips has probability $\gamma^k/k!$, the product of those coins' probabilities, and this is a term of the series. Output 1 if the first failure came at an odd-numbered flip, and 0 if it came at an even one. Stopping exactly at a given flip means surviving the flips before it but not that one, so its chance is the difference of two consecutive terms. Adding these differences over the odd flips gives the series with its alternating signs, and so exactly $e^{-\gamma}$.

For example, take $\gamma = 0.7$. The chance of stopping at the first flip is 0.3000, and at the third flip it is 0.1878. The chances of stopping at the fifth and later odd flips are smaller, and all of them together add up to 0.4966, which is $e^{-0.7}$.

When $\gamma$ is rational, every coin probability $\gamma/k$ is rational too, so every coin is exact by the first fact. For $\gamma$ above 1 the first coin would need a probability above 1. So split $\gamma$ into its whole part and a remainder below 1. Flip one $e^{-1}$ coin for each unit of the whole part and one coin for the remainder, and output 1 only if all of them come up 1. The cost is small. For $\gamma$ up to 1 the mean number of flips is $e^\gamma$, so a coin costs at most $e = 2.7183$ uniform integers on average. With the split, the cost stays below about 5.3 for any $\gamma$ (@fig-discrete-bernoulli-exp).

![Left: the distribution of the index of the first failure for $\gamma = 0.7$, exact and from 5,000 runs, with odd indices (output 1) and even indices (output 0) coloured. Right: the frequency of 1s against $e^{-\gamma}$, with the expected number of uniform draws per coin on the right axis.](figures/discrete_bernoulli_exp.svg){#fig-discrete-bernoulli-exp}

These coins give an exact discrete Laplace sampler (Algorithm 2) for any rational scale. The probability in @eq-dlap-pmf falls by the same factor at each step away from 0, which is the shape of a geometric variable. So the sampler builds a geometric variable with that factor out of $e^{-\gamma}$ coins and uniform integers, using the third fact, and then attaches a random sign. The sign must not count zero twice, because plus zero and minus zero are the same output. So the sampler throws away one of the two ways of producing zero and starts again. Over the scales the full edition tests, a sample costs about 10 to 12 uniform integers.

## Sampling the discrete Gaussian, and choosing a library {#sec-discrete-sampler}

Sampling the discrete Gaussian by inverting its cumulative distribution function (CDF) would need its normalizing constant, which is irrational, and a floating-point uniform, which brings back every problem of @sec-discrete-float. Algorithm 3 of Canonne, Kamath and Steinke (2020) avoids both with rejection sampling. It proposes from the discrete Laplace, which has heavier tails than the discrete Gaussian. So the acceptance probability that turns it into the target never needs to exceed 1, as the second fact requires.

The acceptance probability is chosen so that proposal times acceptance is proportional to the Gaussian weight $e^{-y^2/(2\sigma^2)}$. For this to stay exact, the sampler requires $\sigma^2$ to be rational. It sets the proposal's scale to an integer $t$, one more than the whole part of $\sigma$. A proposal $y$ is then accepted with probability
$$\exp\Big(-\frac{(|y| - \sigma^2/t)^2}{2\sigma^2}\Big).$$
Expanding the square produces a term $|y|/t$ that cancels the discrete Laplace's own factor $e^{-|y|/t}$ exactly, and what remains is the Gaussian weight times a constant. The exponent is rational, so one exact coin from Algorithm 1 decides each acceptance. For example, take $\sigma^2 = 4$, so the scale is $t = 3$. A proposal of 4 then gets the exponent $8/9$. It is therefore kept with probability $e^{-8/9} = 0.4111$.

Any integer scale of at least 1 would give an exact sampler, so this choice of $t$ matters only for cost, and it keeps the sampler cheap. Its acceptance rate is 0.5185 at the smallest width the full edition tests, $\sigma^2 = \frac14$. The rate tends to 0.7602 as $\sigma$ grows, so at these widths a draw takes fewer than two proposals on average. In the full edition's runs a draw costs between 15 and 27 uniform integers. For the histogram above, with $\sigma^2 = 100$ in every bin, the sampler uses about 15.6 uniform integers per bin and no floating-point operation.

![Algorithm 3 for $\sigma^2 = 4$. Left: the discrete Laplace proposal with $t = 3$ and the part of it that is accepted, proportional to the discrete Gaussian PMF. Right: 10,000 draws of Algorithm 3 against the exact PMF on a log scale, with the $\mathcal N(0, 4)$ density for comparison.](figures/discrete_gaussian_sampler.svg){#fig-discrete-gaussian-sampler}

An exact distribution is still not enough, because the running time is a second output. The number of rejection rounds depends on the value drawn, so the time a sampler takes leaks information about the noise. Jin et al. (2022) show that such timing leaks are exploitable. A deployment therefore pads the running time or runs a fixed number of rounds.

These points give the questions to ask when you review a DP library or pipeline. Is the noise drawn by an exact integer sampler or by a snapping-style repair, with no floating-point `exp`, `log` or `random()` inside the sampler? Do the random bits come from a cryptographic source, such as `secrets.SystemRandom()` in Python, rather than a seeded generator? Is the running time padded or fixed? And does the reported guarantee match the mechanism actually used? That last check covers the discrete Gaussian's own $\delta(\varepsilon)$, the sensitivity of the rounded query and the clamping range.

## Code {#sec-discrete-code}

Take the noise itself from a vetted library. What you can check yourself is the calibration, which means checking that the $\delta$ reported for a discrete Gaussian matches @eq-dgauss-delta. The snippet sums the discrete Gaussian's PMF over the integers and compares the result with the continuous Gaussian's exact curve.

```python
from math import ceil, erfc, exp, sqrt


def discrete_gaussian_delta(eps: float, sigma: float, sens: int = 1) -> float:
    """Exact delta(eps) of the discrete Gaussian mechanism, summed over the integers."""
    reach = ceil(40 * sigma) + sens  # the mass beyond 40 sigma is far below double precision
    weight = {y: exp(-y * y / (2 * sigma**2)) for y in range(-reach - sens, reach + 1)}
    total = sum(weight[y] for y in range(-reach, reach + 1))
    delta = 0.0
    for y in range(-reach, reach + 1):  # outputs of f(D) = 0 and f(D') = sens
        p, q = weight[y] / total, weight[y - sens] / total
        delta += max(p - exp(eps) * q, 0.0)
    return delta


def normal_cdf(x: float) -> float:
    return 0.5 * erfc(-x / sqrt(2))


def gaussian_delta(eps: float, sigma: float, sens: float = 1.0) -> float:
    """Exact delta(eps) of the continuous Gaussian mechanism, for comparison."""
    mu = sens / sigma
    return normal_cdf(mu / 2 - eps / mu) - exp(eps) * normal_cdf(-mu / 2 - eps / mu)


print(f"{discrete_gaussian_delta(1.0, 0.5):.4f} {gaussian_delta(1.0, 0.5):.4f}")  # 0.6032 0.5099
print(f"{discrete_gaussian_delta(1.0, 1.0):.4f} {gaussian_delta(1.0, 1.0):.4f}")  # 0.1414 0.1269
print(f"{discrete_gaussian_delta(1.0, 2.0):.4f} {gaussian_delta(1.0, 2.0):.4f}")  # 0.0072 0.0068
```

Floating point is fine here, because this code computes a guarantee rather than releasing noisy data. The full edition's script implements the three exact algorithms with `fractions.Fraction` and integers, and checks each sampler against its exact distribution. It also enumerates the toy floating-point sampler:

```bash
uv run python code/part03-dp-foundations/discrete_mechanisms.py
```

## Cheat sheet

| Result | Formula | Notes |
| --- | --- | --- |
| Support argument | $\delta \ge \Pr[M(D) \in S]$, with $S$ the outputs reachable from $D$ only | Any such output rules out pure DP; the quickest audit |
| Floating-point Laplace | $f(D) + \xi\,b\ln(1/U)$, rounded | Not DP: about 48% of the probability identifies the input in the toy, at every precision tested |
| Snapping | Clamp, add noise, round to a multiple of $\Lambda \ge \lambda$, clamp | $\varepsilon = \frac{1 + 12B\eta}{\lambda} + \frac{2^{-49}B}{\lambda}$ for sensitivity 1 |
| Discrete Laplace | $\frac{1 - r}{1 + r}r^{\lvert z\rvert}$, $r = e^{-1/b}$ | $\varepsilon = \Delta_1 f/b$, pure DP; variance $\frac{2r}{(1 - r)^2}$, just below $2b^2$ |
| Discrete Gaussian | $\propto e^{-x^2/(2\sigma^2)}$ on the integers | $\delta(\varepsilon)$ by @eq-dgauss-delta; Rényi divergence at most the continuous one; variance below $\sigma^2$ |
| Exact $e^{-\gamma}$ coin | Parity of the first failure among coins with probability $\gamma/k$ | $e^\gamma$ uniform integers on average for $\gamma \le 1$ (at most $e$); below about 5.3 for any $\gamma$ |
| Exact discrete Gaussian sampler | Propose $\mathrm{Lap}_{\mathbb Z}(\lfloor\sigma\rfloor + 1)$, accept with an $e^{-\gamma}$ coin | Fewer than two proposals per draw at the widths tested; pad the running time |

: The results of this chapter. {#tbl-discrete-cheat}

**Rules of thumb**

- Never ship `f(D) + laplace(b)` in floating point. Either snap the output to a grid at least as coarse as $b$ and clamp it, or make the query integer-valued and use an exact integer sampler.
- An exact sampler uses only integers, rational numbers and uniform integers from a cryptographic source. Any `exp`, `log` or `random()` inside the sampler is a place where the analysed distribution and the real one can part.
- Compute the discrete Gaussian's $\delta(\varepsilon)$ with @eq-dgauss-delta, because the continuous formula can understate it. Its variance matches $\sigma^2$ closely for $\sigma \ge 1$, and its Rényi divergences never exceed the continuous ones.
- An exact distribution is not the end, because running time that depends on the output is a separate side channel.

## Further reading

- Mironov, "On Significance of the Least Significant Bits for Differential Privacy" (CCS 2012): the floating-point attack and the snapping mechanism.
- Canonne, Kamath & Steinke, "The Discrete Gaussian for Differential Privacy" (NeurIPS 2020): the discrete Gaussian's guarantees and the exact samplers; the reference implementation to compare a library against.
- Jin, McMurtry, Rubinstein & Ohrimenko, "Are We There Yet? Timing and Floating-Point Attacks on Differential Privacy Systems" (IEEE S&P 2022): what still goes wrong in modern libraries, including timing.
- Kairouz, Liu & Steinke, "The Distributed Discrete Gaussian Mechanism for Federated Learning with Secure Aggregation" (ICML 2021): the discrete Gaussian when each client adds its own noise.

**Derivations.** Every result in this chapter is derived step by step in [the full edition's chapter](../../book-full/part03-dp-foundations/07-discrete-mechanisms.md), except three that it cites: the floating-point terms of the snapping mechanism's guarantee (Mironov 2012, which the full edition outlines), the optimality of the geometric mechanism (Ghosh, Roughgarden & Sundararajan 2009), and the closeness of sums of discrete Gaussians, which the exercise book checks numerically and Kairouz, Liu and Steinke (2021) bound.
