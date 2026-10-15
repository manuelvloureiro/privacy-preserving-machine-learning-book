# f-DP and Gaussian differential privacy

## Why this matters

Every privacy guarantee in this part answers the same question. An attacker sees the output of a mechanism and wants to tell which of two datasets it ran on. One is a dataset $D$, and the other is a neighbour $D'$ that differs from it in one record. The guarantee says how well the attacker can tell them apart.

Each definition in this part gives the answer in a different form. Differential privacy (DP) uses two numbers, a privacy level and a small slack. Rényi differential privacy (RDP) uses a curve of divergences instead ([Rényi DP](01-renyi-dp.md)), and zero-concentrated differential privacy (zCDP) a single number ([Concentrated differential privacy](03-concentrated-dp.md)). As this chapter shows, all three summarize one more complete answer.

The complete answer comes from hypothesis testing ([Hypothesis testing](../part01-prerequisites/04-hypothesis-testing.md)). The attacker's decision is a test, and a test can make two kinds of mistake. Its **type I error** is how often it claims $D'$ when the data was $D$. Its **type II error** is the reverse, how often it claims $D$ when the data was $D'$.

Part 1 wrote the type I error as $\alpha$, but in this part $\alpha$ is the Rényi order. So this chapter writes $a$ for the type I error and $b$ for the type II error. The **trade-off function** records, for each type I error $a$, the smallest type II error that any test can reach. It describes everything that any attacker can do. The idea of this chapter is that this curve is the privacy guarantee itself, and every other notion a summary that throws part of it away.

Two facts make the curve practical as well as complete. Gaussian noise produces a family of curves with a single parameter, and composing two curves of the family gives another one. So a pipeline of Gaussian steps has an exact guarantee in one number. And a central limit theorem (CLT) says that many steps that each leak little compose to a curve close to one of these Gaussian curves, whatever the steps were. Training a model with stochastic gradient descent (SGD) under differential privacy, called DP-SGD, is such a pipeline, because each of its many noisy steps leaks little. So the CLT gives a quick estimate of its privacy, though, as the chapter shows, not a guarantee.

## f-DP {#sec-gdp-fdp}

A privacy guarantee should say what an attacker cannot do. In testing terms, the strongest such statement is a floor under the attacker's type II error at every type I error. However the attacker tunes its test, it must miss at least this often. A floor at every $a$ is a curve, so the definition takes a curve $f$ as its parameter.

For a mechanism $M$, write $T\big(M(D), M(D')\big)$ for the trade-off function between its outputs on the two datasets. Dong, Roth and Su turned the floor into a definition in 2022, and this chapter cites them as Dong et al. from here on. A mechanism is **$f$-differentially private ($f$-DP)** if no pair of neighbours is easier to tell apart than $f$ allows:
$$T\big(M(D), M(D')\big) \ge f \quad\text{for all neighbours } D \text{ and } D'.$$
Here one curve is at least another when it is at least as high at every type I error. The curve $f$ must itself be a trade-off function. As [Hypothesis testing](../part01-prerequisites/04-hypothesis-testing.md) showed, that means it never increases, it is convex, it has no jumps, and it lies on or below the coin-flip line $b = 1 - a$.

Two properties follow from the definition. The first is symmetry, because being neighbours is a symmetric relation and so the guarantee also holds with $D$ and $D'$ swapped. Swapping the datasets reflects a trade-off curve across the diagonal where $a = b$. The reason is that a test which says "$D'$" on some outputs can be read as a test which says "$D$" there, and its two errors then trade places. So an $f$-DP mechanism also has the reflected curve as a floor. When two curves are both floors, so is the higher of them at each point. A curve that equals its own reflection is called **symmetric**. Most curves in this chapter are, and the results that need symmetry say so.

The second property is post-processing. Processing an output can only make two distributions harder to tell apart, because any test on the processed output is also a test on the raw one. The curve can therefore only rise, so $f$-DP survives any post-processing.

The familiar definition with privacy level $\varepsilon$ and slack $\delta$, written $(\varepsilon, \delta)$-DP, is a special case. As [The definition of differential privacy](../part03-dp-foundations/01-definition.md) showed, it holds exactly when every test's pair of errors lies above two straight lines. So it is $f$-DP for the curve
$$f_{\varepsilon,\delta}(a) = \max\big(0,\; 1 - \delta - e^\varepsilon a,\; e^{-\varepsilon}(1 - \delta - a)\big).$$ {#eq-gdp-feps}
The first line inside the maximum is the DP inequality applied to the outputs on which a test fires. The second line is its mirror image, with $D$ and $D'$ swapped. The curve starts at $1 - \delta$ rather than at 1 when $a$ is zero. That gap is what $\delta$ means, since a test may catch $D'$ with probability $\delta$ without ever raising a false alarm.

Two straight lines make a crude floor, and two mechanisms that both carry the label "$\varepsilon = 1$" show why (@tbl-gdp-curves, @fig-gdp-eps-delta). Binary randomized response with $\varepsilon = 1$ reports a bit truthfully with probability $e/(1 + e) = 0.7311$ ([randomized response](../part03-dp-foundations/01-definition.md#sec-dpdef-rr)). Take the bit to be 0 under $D$ and 1 under $D'$. The only useful test then fires when the report is 1, and both errors of that test are 0.2689. That point is exactly the corner of $f_{1,0}$. Mixing this test with the tests that never fire and always fire fills in the straight segments on either side, so randomized response sits on the floor at every $a$. It is the worst $(1, 0)$-DP mechanism.

The Laplace mechanism with sensitivity 1 and scale 1 is also $(1, 0)$-DP. It matches the floor at small $a$, where both curves fall with slope $-e$, but it is more private in the middle. At a type I error of a quarter, for example, its smallest type II error is 0.3679, against 0.3204 for randomized response. The label "$(1, 0)$-DP" cannot tell the two mechanisms apart, but their curves can. For comparison, the last column of the table is a Gaussian mechanism, whose curve has a different shape that the next section explains.

| Type I error $a$ | $f_{1,0}$ (randomized response) | Laplace, $\varepsilon = 1$ | Gaussian, $\sigma = 1$ ($G_1$) |
| --- | --- | --- | --- |
| 0.01 | 0.9728 | 0.9728 | 0.9076 |
| 0.1 | 0.7282 | 0.7282 | 0.6109 |
| 0.25 | 0.3204 | 0.3679 | 0.3724 |

: Smallest type II error at three type I errors for three mechanisms. {#tbl-gdp-curves}

![Trade-off curves of $(1, 0)$-DP (randomized response), $(1, 0.1)$-DP, the Laplace mechanism with $\varepsilon = 1$ and the Gaussian mechanism with $\sigma = 1$. A mechanism is $f$-DP when its curve lies on or above $f$; the dotted diagonal is perfect privacy.](figures/gaussian_dp_eps_delta.svg){#fig-gdp-eps-delta}

The curve is also the most informative description of a mechanism's privacy, in a precise sense. Take two pairs of distributions, a first and a second. By Blackwell's theorem (Dong et al., Theorem 2.10), the second pair's curve lies on or above the first's exactly when post-processing can produce the second pair from the first. So any measure of privacy that post-processing cannot increase, such as a divergence, must rank the two pairs as their curves do. It can only be a coarser version of the curve.

## Gaussian differential privacy {#sec-gdp-gdp}

The Gaussian mechanism is the workhorse of private machine learning, and its trade-off curve has a simple form. It adds noise to a query $g$ whose $\ell_2$ sensitivity is $\Delta_2$, the largest Euclidean distance between the answers on two neighbouring datasets. The noise is normal with independent coordinates. Its standard deviation is $\sigma\Delta_2$, where the **noise multiplier** $\sigma$ is the standard deviation in units of the sensitivity.

Dividing the output by $\sigma\Delta_2$ and subtracting the scaled answer on $D$ changes no test's errors, because both maps can be undone. What is left is telling a standard normal vector from the same vector shifted by the scaled difference between the two answers. The noise has the same spread in every direction. It therefore looks the same under both datasets in every direction perpendicular to the shift, and there it carries no evidence. Only the projection of the output onto the direction of the shift matters. That projection is a standard normal draw under $D$, and under $D'$ it is a normal draw shifted by the length of the scaled difference.

So every Gaussian mechanism reduces to telling a standard normal draw from a normal draw shifted by a distance $\mu$. Here $\mu$ is the distance between the two answers in units of the noise, and the two distributions are written $\mathcal N(0, 1)$ and $\mathcal N(\mu, 1)$.

[Hypothesis testing](../part01-prerequisites/04-hypothesis-testing.md) derived the curve of this pair. The likelihood ratio of the two normals grows with the output, so by the Neyman–Pearson lemma the best test fires above a threshold. Write $\Phi$ for the standard normal distribution function, so that $\Phi(t)$ is the probability that a standard normal draw is at most $t$. Its inverse is $\Phi^{-1}$, and at type I error $a$ the threshold is $\Phi^{-1}(1 - a)$. A draw from the shifted normal stays below it with probability
$$G_\mu(a) = \Phi\big(\Phi^{-1}(1 - a) - \mu\big).$$ {#eq-gdp-gmu}

A mechanism satisfies **$\mu$-Gaussian differential privacy (GDP)** if the Gaussian curve $G_\mu$ is a floor for it, that is, if it is $G_\mu$-DP. In words, no test can tell its outputs on neighbouring datasets apart better than one can tell $\mathcal N(0, 1)$ from $\mathcal N(\mu, 1)$. The Gaussian mechanism with noise multiplier $\sigma$ satisfies it with
$$\mu = \frac1\sigma.$$
The reason is that a larger shift is easier to detect, so $G_\mu$ falls as $\mu$ grows. On every pair of neighbours the answers are at most one sensitivity apart, which is a shift of at most $1/\sigma$ in units of the noise. Every pair therefore has a curve on or above $G_{1/\sigma}$.

The bound is tight whenever some pair of neighbours is a full sensitivity apart, because that pair has exactly the curve $G_{1/\sigma}$. So for the Gaussian mechanism, $\mu$ is not a bound with slack but the curve itself. And since $\mu$-GDP is $f$-DP for one particular curve, it survives post-processing too.

The parameter $\mu$ reads directly as attack errors. Take a count, which has sensitivity 1 because adding or removing one record changes it by 1. Released with $\sigma = 1$ it is 1-GDP, so @tbl-gdp-g1 evaluates $G_1$ at three type I errors. For example, the threshold at a 1% type I error is 2.3263 standard deviations. Subtracting the shift leaves 1.3263, and a standard normal draw falls below that with probability 0.9076.

| Type I error $a$ | Smallest type II error $G_1(a)$ | Best power $1 - G_1(a)$ |
| --- | --- | --- |
| 0.01 | 0.9076 | 0.0924 |
| 0.05 | 0.7405 | 0.2595 |
| 0.1 | 0.6109 | 0.3891 |

: The 1-GDP guarantee of the Gaussian mechanism with $\sigma = 1$, read as attack errors. {#tbl-gdp-g1}

Read the first row as a membership attack, with $D'$ the dataset that contains the target's record. At a 1% false-positive rate, and whatever the attacker knows about the rest of the data, it detects the presence of the record at most 9.24% of the time.

To get a feel for the scale, compare two values of $\mu$ at a 5% false-positive rate. The best test against a count released at $\mu = 1$ has power 0.2595. At $\mu = 2$ its power rises to 0.6388. As a rough guide, Dong et al. suggest that $\mu \le 0.5$ is a reasonable guarantee and that $\mu \ge 6$ promises almost nothing.

## Duality with (ε, δ) {#sec-gdp-duality}

Regulators, papers and libraries report $(\varepsilon, \delta)$, so you need to turn a curve into $(\varepsilon, \delta)$ pairs. The geometry of the curve shows how to do it without losing anything. An $(\varepsilon, \delta)$ guarantee is the line $b = 1 - \delta - e^\varepsilon a$ together with its mirror image, and a curve satisfies it when it lies above the line. Fix the slope $-e^\varepsilon$ of the line. The smallest $\delta$ that works is the one whose line just touches the curve, since any smaller $\delta$ would lift the line above part of the curve.

Doing this for every slope gives a function $\delta(\varepsilon)$, the **privacy profile** of the curve. The line must stay below the curve at every type I error. So the smallest $\delta$ is the largest gap between $1 - e^\varepsilon a$ and the curve:
$$\delta(\varepsilon) = \max_{0 \le a \le 1}\big(1 - f(a) - e^\varepsilon a\big).$$ {#eq-gdp-conjugate}

The formula also reads as an attack. Suppose an attacker earns one unit for each correct detection of $D'$ and pays $e^\varepsilon$ units for each false alarm. A test detects $D'$ with probability $1 - b$, so its net gain is $1 - b - e^\varepsilon a$. The privacy profile $\delta(\varepsilon)$ is therefore the best net gain of any test. The best test fires on exactly the outputs that are more than $e^\varepsilon$ times as likely under $D'$ as under $D$. Each of those outputs earns more than it costs, and every other output costs more than it earns.

The **privacy loss** of an output is the logarithm of how much more likely it is under $D$ than under $D'$. For a symmetric curve, the best net gain is the $\delta(\varepsilon)$ that [Advanced and optimal composition](../part03-dp-foundations/04-composition.md) computed from this loss.

The converse holds too, and it is what makes the profile lossless. A convex curve is the upper envelope of the lines that touch it, so the touching lines together rebuild the whole curve. For a symmetric curve, being $f$-DP is therefore the same as being $(\varepsilon, \delta(\varepsilon))$-DP for every $\varepsilon \ge 0$ at once (Dong et al.).

The curve must be symmetric for this to work. A symmetric curve has slope $-1$ where it crosses the diagonal. The lines with $\varepsilon \ge 0$ are at least that steep, so they rebuild only the part left of the diagonal. Their mirror images must cover the rest. Without symmetry, swapping $D$ and $D'$ can give a different profile.

For the Gaussian curve the touching point has a closed form. By the reasoning above, the touching test fires on the outputs that are more than $e^\varepsilon$ times as likely under the shifted normal as under the standard one. That likelihood ratio grows with the output, so those are the outputs above a threshold, here $t = \varepsilon/\mu + \mu/2$. Reading off that test's power and type I error gives the profile of $\mu$-GDP,
$$\delta(\varepsilon) = \Phi\Big(-\frac{\varepsilon}{\mu} + \frac{\mu}{2}\Big) - e^\varepsilon\,\Phi\Big(-\frac{\varepsilon}{\mu} - \frac{\mu}{2}\Big)$$ {#eq-gdp-delta}
(Dong et al., Corollary 2.13). The first term is the power of the touching test, and the second is $e^\varepsilon$ times its type I error. This is the same formula as the analytic Gaussian calibration of [The Laplace and Gaussian mechanisms](../part03-dp-foundations/03-laplace-gaussian.md) (Balle & Wang 2018). There it came from the bad event that the privacy loss exceeds $\varepsilon$, and here it comes out of the curve.

Take $\mu = 1$ and $\varepsilon = 1$ as an example. The touching threshold is $t = 1.5$, so the touching test has type I error $\Phi(-1.5) = 0.066807$. Its power is $\Phi(-0.5) = 0.308538$. Multiplying the type I error by $e$ gives 0.181601, and subtracting that from the power leaves $\delta(1) = 0.126937$.

At $\varepsilon = 0$ the profile is the total variation (TV) distance, which Part 1 showed is the best advantage $1 - a - b$ of any test. The reason is that the line of slope $-1$ touches a symmetric curve on the diagonal, at the test with the best advantage. For $G_\mu$ that test puts its threshold halfway between the two means. Setting $\varepsilon = 0$ in @eq-gdp-delta then gives
$$\delta(0) = 2\Phi(\mu/2) - 1.$$
At $\mu = 1$ this is 0.3829.

Going the other way, from a target $\delta$ to $\varepsilon$, has no closed form. But $\delta(\varepsilon)$ falls as $\varepsilon$ grows, so you can find the $\varepsilon$ that hits the target by bisection, which keeps halving an interval that contains the crossing point (@tbl-gdp-eps).

| $\mu$ | 0.5 | 1 | 2 |
| --- | --- | --- | --- |
| $\varepsilon$ at $\delta = 10^{-5}$ | 1.9931 | 4.3772 | 9.9973 |

: The $\varepsilon$ of $\mu$-GDP at $\delta = 10^{-5}$. {#tbl-gdp-eps}

The profile also explains what a small $\delta$ costs. The second term of @eq-gdp-delta is subtracted from the first, so the first term, a normal tail, sets how small $\delta$ can be. A normal tail falls roughly like $e^{-x^2/2}$ at a distance $x$ from the centre, so the distance must be about $\sqrt{2\ln(1/\delta)}$. Here the distance is $\varepsilon/\mu - \mu/2$, which means that $\varepsilon$ grows roughly like $\mu\sqrt{2\ln(1/\delta)}$.

So each further factor of ten in $\delta$ costs extra $\varepsilon$, roughly in proportion to $\mu$ (@tbl-gdp-delta-cost). At $\mu = 1$ each factor of ten costs about half a unit of $\varepsilon$, a little less each time because a square root grows ever more slowly. At $\mu = 2$ the steps are about twice as large (@fig-gdp-duality, left).

| $\mu$ | $\varepsilon$ at $\delta = 10^{-4}$ | $\varepsilon$ at $\delta = 10^{-5}$ | $\varepsilon$ at $\delta = 10^{-6}$ | The two steps |
| --- | --- | --- | --- | --- |
| 1 | 3.8044 | 4.3772 | 4.8866 | 0.57, 0.51 |
| 2 | 8.8769 | 9.9973 | 10.9972 | 1.12, 1.00 |

: What each factor of ten in $\delta$ costs in $\varepsilon$ for $\mu$-GDP. {#tbl-gdp-delta-cost}

![Left: the privacy profile $\delta(\varepsilon)$ of $\mu$-GDP for $\mu = 0.5, 1, 2$, with the level $\delta = 10^{-5}$ dotted. Right: $G_1$ and six of its $(\varepsilon, \delta(\varepsilon))$ curves. Each touches $G_1$ at two mirror-image points (dots); their upper envelope is $G_1$.](figures/gaussian_dp_duality.svg){#fig-gdp-duality}

The whole profile loses nothing, as the right panel of @fig-gdp-duality shows, but one point of it loses a lot. Take 1-GDP, which implies $(4.3772, 10^{-5})$-DP. That single pair is one touching line and its mirror image, and it permits far more than the curve does. At $a = 0.1$ the mirror-image line, of slope $-e^{-\varepsilon}$, is the higher of the two, and it allows a type II error as low as 0.0113. The curve itself requires 0.6109. So when you report a pair, report $\mu$ or the noise multiplier along with it.

## Composition {#sec-gdp-composition}

Training runs a mechanism thousands of times. An attacker who sees the outputs of two mechanisms, run with independent randomness, faces a pair of joint distributions. Either both outputs are drawn on $D$, or both are drawn on $D'$. The trade-off curve of this joint pair is the composed guarantee, and Dong et al. call it the **tensor product** $f \otimes g$ of the two curves. It depends only on the two curves, not on which mechanisms produced them.

Write $f_i$ for the curve of the $i$-th step. The composition theorem says that if each of $k$ steps is $f_i$-DP, then together they are $f_1 \otimes \dots \otimes f_k$-DP. The theorem holds even when each step is chosen after seeing the earlier outputs (Dong et al., Theorem 3.2). No better bound holds in general. The bound is reached when the steps run without seeing each other's outputs and a single pair of neighbours attains every $f_i$.

For Gaussian curves the tensor product has a closed form. Two Gaussian releases with independent noise form a two-dimensional problem, a shift by $\mu_1$ along one axis and by $\mu_2$ along the other. By the projection argument of @sec-gdp-gdp, only the direction of the shift carries evidence. So the problem is a one-dimensional shift by the length of that vector (@fig-gdp-composition, left). Repeating the argument for $k$ releases gives
$$G_{\mu_1} \otimes \dots \otimes G_{\mu_k} = G_\mu, \qquad \mu = \sqrt{\mu_1^2 + \dots + \mu_k^2}$$ {#eq-gdp-compose}
(Dong et al., Corollary 3.3).

Under composition, the quantity that adds up is $\mu^2$, just as the variances of independent noises add. With $k$ identical steps, $\mu$ therefore grows like $\sqrt k$. Full-batch noisy gradient descent with clipped gradients and noise multiplier $\sigma$ is a composition of $k$ Gaussian mechanisms, each $1/\sigma$-GDP. After $k$ steps it is $\sqrt k/\sigma$-GDP, and in the worst case it is no better than that. So no accountant can improve on this number, and none needs to do more.

Concentrated DP does the same arithmetic, since in zCDP the Gaussian mechanism has $\rho = \mu^2/2$ and the $\rho$'s add ([Concentrated differential privacy](03-concentrated-dp.md)). But at the end GDP gives the exact curve, while $\rho$ gives only an RDP bound on it (@sec-gdp-rdp).

For example, two 1-GDP releases compose to $G_{\sqrt 2} = G_{1.4142}$, while four compose to $G_2$. A hundred releases with $\sigma = 10$ each, so that each is 0.1-GDP, compose to $G_1$. Two releases sit between one release, with $G_1(0.05) = 0.7405$, and one release with twice the shift, with $G_2(0.05) = 0.3612$.

![Left: 400 draws each from $\mathcal N(0, I_2)$ and $\mathcal N((1, 1), I_2)$, the outputs of two Gaussian releases on $D$ and $D'$. The best test thresholds the projection on the shift (arrow); the dashed line is its boundary at $a = 0.05$. Right: $G_1$ for one release, $G_{\sqrt 2}$ for two, and simulated errors of the projected test (crosses, 20,000 draws per point).](figures/gaussian_dp_composition.svg){#fig-gdp-composition}

Curves that are not Gaussian can leave their family when composed. Run randomized response with $\varepsilon = 1$ twice, independently, on the same bit, which is 0 under $D$ and 1 under $D'$. Each run keeps the bit with probability 0.7311, and @tbl-gdp-rr2 lists the probabilities of the four output pairs under each dataset.

| Outputs | $p$ | $q$ | $q/p$ |
| --- | --- | --- | --- |
| 00 | 0.5344 | 0.0723 | 0.1353 |
| 01 | 0.1966 | 0.1966 | 1.0000 |
| 10 | 0.1966 | 0.1966 | 1.0000 |
| 11 | 0.0723 | 0.5344 | 7.3891 |

: Two independent runs of randomized response with $\varepsilon = 1$, with $p$ the probability under $D$ and $q$ the probability under $D'$. {#tbl-gdp-rr2}

By the Neyman–Pearson lemma of [Hypothesis testing](../part01-prerequisites/04-hypothesis-testing.md), the best tests add outputs to the set on which they fire in a fixed order. They take the outputs in decreasing order of the likelihood ratio, the probability under $D'$ divided by the probability under $D$. The first test fires only when both runs report a one. Its type I error is 0.0723. Its type II error is 0.4656, which is one minus its power of 0.5344. Adding one of the mixed outputs reaches the corner where both errors are 0.2689. Adding the other one as well swaps the first pair of errors. So the composed curve is a polygon with three slopes, $-e^2$, $-1$ and $-e^{-2}$, which are minus the likelihood ratios.

An $f_{\varepsilon,\delta}$ curve has only two slopes, so this polygon is not $f_{\varepsilon,\delta}$ for any single pair. It is the higher of two such curves, $f_{2,0}$ for the steep segment and $f_{0, 0.4621}$ for the middle one. The middle segment's $\delta$ of 0.4621 is one minus the two errors of the first test. This is the smallest example of a general fact, that composed $(\varepsilon, \delta)$ guarantees form a family of pairs, not one pair.

Composing curves rather than single pairs matters in practice. Four 1-GDP releases compose to 2-GDP, which @tbl-gdp-eps turns into $\varepsilon = 9.9973$ at $\delta = 10^{-5}$. Suppose instead you first turn each release into one $(\varepsilon, \delta)$ pair, giving each release a quarter of the $\delta$ budget, which yields $(4.6902, 2.5\times10^{-6})$. Then you compose the four pairs with basic composition, which adds the $\varepsilon$'s and the $\delta$'s. You get $\varepsilon = 18.7607$ at $\delta = 10^{-5}$, almost twice as large. Even optimal composition of the pairs loses, because each pair has already dropped most of its curve, and [Advanced and optimal composition](../part03-dp-foundations/04-composition.md) shows by how much over many steps.

For curves other than Gaussian ones, the tensor product is hard to compute. For $f_{\varepsilon,\delta}$ curves it is the optimal composition theorem of Kairouz, Oh & Viswanath (2015). Computing it exactly is hard when the pairs differ from step to step (Murtagh & Vadhan 2016). Numerical accountants therefore add up privacy losses instead, since the privacy loss of a composition is the sum of the per-step losses. They put each step's loss on a fine grid and combine the steps numerically ([Numerical accounting with privacy loss distributions](06-numerical-accounting.md)).

## The central limit theorem for composition {#sec-gdp-clt}

Most mechanisms are not Gaussian, so @eq-gdp-compose does not apply to them directly. The CLT for $f$-DP says that many steps that each leak little still compose to nearly a Gaussian curve. It rests on two facts about the privacy loss.

The first fact is that the trade-off curve depends only on the distribution of the privacy loss. By the Neyman–Pearson lemma, the best tests fire on the outputs that are most likely under $D'$ relative to $D$. Those are the outputs with the smallest privacy loss. So every best test fires when the loss falls below a threshold, and its two errors are the probabilities of that event under the two datasets. The loss also links those two probabilities. An output with loss $L$ is $e^L$ times as likely under $D$ as under the neighbour, so its probability under one dataset fixes the other.

For the Gaussian pair, the privacy loss under $D$ is normal, with mean $\mu^2/2$ and variance $\mu^2$ ([The Laplace and Gaussian mechanisms](../part03-dp-foundations/03-laplace-gaussian.md)). Since the curve depends only on the loss, any privacy loss that is normal, with a mean of half its variance, gives a Gaussian curve.

The second fact is that the privacy loss of a composition is the sum of the per-step losses. Independent outputs have probabilities that multiply, and the logarithm turns the product into a sum. A sum of many small independent terms is close to normal, by the ordinary CLT. So many steps that each leak little compose to a curve close to some $G_\mu$.

Dong et al. make this precise. Write $K$ for the mean of the summed privacy loss under $D$ and $V$ for its variance. Both are sums of the per-step values, because means add and the variances of independent terms add. The Gaussian pair's loss has a mean of $\mu^2/2$ and a standard deviation of $\mu$, so the ratio of the two is $\mu/2$. Matching that ratio to $K/\sqrt V$ gives the Gaussian curve that the composition is close to:
$$\mu = \frac{2K}{\sqrt V}.$$ {#eq-gdp-clt-mu}
Matching the ratio, rather than the variance alone, is Dong et al.'s choice. A privacy loss that is normal must have a mean of half its variance, and a sum of small losses has this only approximately. So other matches give slightly different values of $\mu$, and they all agree in the limit.

The theorem comes with an error band, whose width $\gamma$ comes from the Berry–Esseen bound on how far a sum of independent terms is from normal. The band needs, for each step, the average cube of the distance of the loss from its mean. Then $\gamma$ is 0.56 times the sum of these averages, divided by $V^{3/2}$. The theorem requires symmetric curves for which these averages are finite (Dong et al., Theorem 3.4).

The band applies when $\gamma$ is below one half, and only to type I errors between $\gamma$ and $1 - \gamma$. There the composed curve lies within $\gamma$ of $G_\mu$, both across and down:
$$G_\mu(a + \gamma) - \gamma \le f_1 \otimes \dots \otimes f_k(a) \le G_\mu(a - \gamma) + \gamma.$$

As the number of steps grows and no single step dominates, the band shrinks to zero (Dong et al., Theorem 3.5). Take steps that are each $\varepsilon_i$-DP, where $\varepsilon_i$ is the privacy level of the $i$-th step. Their composition tends to a Gaussian curve when the largest $\varepsilon_i$ goes to zero and the sum of their squares converges (Dong et al., Theorem 3.6). The limit is $G_\mu$ with
$$\mu = \sqrt{\textstyle\sum_i \varepsilon_i^2}.$$
For $k$ equal steps this grows like $\sqrt k$, the behaviour of advanced composition, but now with the exact constant.

This is why GDP is the natural family. Many small compositions always tend to some $G_\mu$, and composing two curves of the family stays in it, while the $(\varepsilon, \delta)$ family does not (@tbl-gdp-rr2).

The CLT is an approximation, not a bound. The Berry–Esseen band is a bound, but it is wide, 0.0562 at $k = 100$ in the example below. The point estimate $G_\mu$ can lie above or below the true curve, and its error is largest in the tails. That is exactly where small values of $\delta$ are decided, because they depend on rare, large values of the loss. A CLT controls the bulk of a distribution much better than its tails.

An example shows both effects. Take $k$ mechanisms, each $\varepsilon_0$-DP with $\varepsilon_0 = 1/\sqrt k$. The squares of the $\varepsilon$'s then add up to 1, so the limit is $G_1$. Its $\delta(1)$ is 0.1269. To judge the CLT you need the true $\delta(1)$ of the composition. The full edition computes it exactly for randomized response and brackets it for Laplace.

| $k$ | Randomized response, exact $\delta(1)$ | Randomized response: $\mu$, $\gamma$ | Randomized response: GDP $\delta(1)$ | Laplace $\delta(1)$ | Laplace: $\mu$, $\gamma$ | Laplace: GDP $\delta(1)$ |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 0.0000 | 1.0422, 0.7663 | 0.1420 | 0.0000 | 0.9075, 0.7348 | 0.0956 |
| 10 | 0.1323 | 1.0042, 0.1837 | 0.1284 | [0.1112, 0.1114] | 0.9569, 0.1889 | 0.1120 |
| 100 | 0.1257 | 1.0004, 0.0562 | 0.1271 | [0.1211, 0.1214] | 0.9844, 0.0569 | 0.1215 |
| 1000 | 0.1269 | 1.0000, 0.0177 | 0.1270 | [0.1250, 0.1252] | 0.9948, 0.0178 | 0.1251 |

: $\delta(1)$ for $k$-fold randomized response and Laplace with $\varepsilon_0 = 1/\sqrt k$, exactly (or bracketed) and from the CLT with the $\mu$ of @eq-gdp-clt-mu. The limit $G_1$ has $\delta(1) = 0.1269$. {#tbl-gdp-clt}

Three things stand out in @tbl-gdp-clt. First, from $k = 100$ on, the CLT is accurate in the bulk. For randomized response it gives 0.1271 against the exact 0.1257. For Laplace it gives 0.1215, just above the bracket. Second, at $k = 10$ the exact value for randomized response lies above the CLT's, so the CLT under-reports $\delta$ there. Third, the Laplace compositions stay on the private side of $G_1$ at every $k$ shown, while their CLT $\mu$ rises toward 1 from below.

In the tail the picture changes, as @tbl-gdp-clt-eps shows for $\delta = 10^{-5}$.

| $k$ | Randomized response, exact $\varepsilon$ | GDP with the $\mu$ of @eq-gdp-clt-mu | GDP with $\mu = 1$ | Basic composition $k\varepsilon_0$ |
| --- | --- | --- | --- | --- |
| 10 | 3.1599 | 4.3983 | 4.3772 | 3.1623 |
| 100 | 4.3068 | 4.3793 | 4.3772 | 10.0000 |
| 1000 | 4.3683 | 4.3774 | 4.3772 | 31.6228 |

: $\varepsilon$ at $\delta = 10^{-5}$ for $k$-fold randomized response with $\varepsilon_0 = 1/\sqrt k$. {#tbl-gdp-clt-eps}

At $k = 10$ the true loss can never exceed $k\varepsilon_0 = 3.1623$. So the exact $\varepsilon$ of 3.1599 is far below the Gaussian's 4.3983, because a Gaussian tail does not know that the loss is bounded. By $k = 1000$ the three values agree to within 0.01. So in the tail the CLT over-reported $\varepsilon$ at every $k$, which errs on the safe side. In the bulk it under-reported $\delta$ at $k = 10$, the unsafe side, and slightly over-reported it from $k = 100$ on (@fig-gdp-clt).

![Left: the exact curve $f_{\varepsilon_0,0}^{\otimes k}$ of $k$-fold randomized response with $\varepsilon_0 = 1/\sqrt k$, for $k = 1, 4, 16, 64$, approaching $G_1$. Right: their privacy profiles $\delta(\varepsilon)$ on a log scale for $k = 1, 10, 100$; each drops to 0 at $\varepsilon = k\varepsilon_0 = \sqrt k$ (off the plot for $k = 100$), while the GDP profile has a Gaussian tail.](figures/gaussian_dp_clt.svg){#fig-gdp-clt}

## The subsampled Gaussian and DP-SGD {#sec-gdp-sgd}

DP-SGD composes $k$ steps ([Amplification by subsampling](05-subsampling.md)). Each step forms a batch by Poisson sampling, which includes each record independently with a **sampling rate** $q$. It then clips each gradient to a norm of at most the sensitivity, and adds Gaussian noise with noise multiplier $\sigma$ to their sum. When $q$ is small, the target record is usually not even in the batch, so each step leaks little and the CLT applies.

The neighbouring relation matters here, so fix it first. Under add/remove neighbours, $D'$ is $D$ with one record added. The other records contribute the same to both outputs, so they can be subtracted. The added record is in the batch with probability $q$, and its clipped gradient then shifts the noisy sum by at most one sensitivity. Measure the output in units of the sensitivity, rather than of the noise as before, so that the noise has standard deviation $\sigma$. Since a full shift is the easiest to detect, the worst case of one step is a normal distribution $\mathcal N(0, \sigma^2)$ against a mixture. The mixture is the same normal with probability $1 - q$, and the normal shifted by 1 with probability $q$.

Under add/remove neighbours, Bu et al. (2020) show that $k$ such steps are asymptotically $\mu$-GDP with
$$\mu = q\sqrt{k\big(e^{1/\sigma^2} - 1\big)}.$$ {#eq-gdp-sgd-mu}
The limit lets the number of steps $k$ grow while $q\sqrt k$ and $\sigma$ stay fixed. In practice the formula is used as an approximation for a finite number of steps.

The formula comes from the CLT. For small $q$, one step's loss has a mean of about $q^2(e^{1/\sigma^2} - 1)/2$ and a variance of about twice that. The factor $q^2$ reflects that the record is present only with probability $q$. Putting $k$ times these into @eq-gdp-clt-mu gives the formula. Holding $q\sqrt k$ fixed keeps the summed mean and variance finite while each step leaks ever less, which is what the limit form of the CLT needs.

Subsampling shrinks $\mu$ roughly in proportion to $q$. Compared with full-batch training, which is $\sqrt k/\sigma$-GDP after $k$ steps, subsampling multiplies $\mu$ by about $q\sigma\sqrt{e^{1/\sigma^2} - 1}$. At $\sigma = 1$ this multiplier is 1.3108 times $q$.

The formula also depends strongly on $\sigma$, through the factor $e^{1/\sigma^2} - 1$, which is 1.7183 at $\sigma = 1$. Lowering the noise multiplier to 0.8 more than doubles the factor, to 3.7707. Raising it to 1.5 cuts the factor to 0.5596.

Like the CLT itself, the result is an approximation, and the expansion for small $q$ adds a second one. Neither is controlled in the tail where $\delta = 10^{-5}$ lives. Gopi, Lee & Wutschitz (2021) showed that the CLT value can lie below the true $\varepsilon$ of DP-SGD.

The running example of this part shows the CLT under-reporting $\varepsilon$ in this way. It samples with rate $q = 0.01$ and noise multiplier $\sigma = 1$, and it aims at $\delta = 10^{-5}$. With these values the formula gives $\mu = 0.01 \times 1.3108\sqrt k$, so that at $k = 100$ steps, for example, $\mu$ is 0.1311. @tbl-gdp-sgd turns each $\mu$ into $\varepsilon$ with @eq-gdp-delta and compares it with a numerical accountant.

| Steps $k$ | $q\sqrt k$ | CLT $\mu$ | CLT $\varepsilon$ at $10^{-5}$ | Numerical accountant $\varepsilon$ | Full batch, $\mu = \sqrt k$ |
| --- | --- | --- | --- | --- | --- |
| 100 | 0.10 | 0.1311 | 0.4575 | [0.7175, 0.7185] | 10.0 |
| 1,000 | 0.32 | 0.4145 | 1.6177 | [1.8232, 1.8332] | 31.6 |
| 10,000 | 1.00 | 1.3108 | 6.0071 | [6.1377, 6.2377] | 100.0 |

: The running DP-SGD example: the CLT's $\mu$ and $\varepsilon$, and a numerical bracket for the true $\varepsilon$. {#tbl-gdp-sgd}

The bracket comes from the same rounding method as the Laplace example, taking the worse of the two orders of neighbours. Its width is fixed by the grid, so its informative number is the lower end, which comes from rounding every loss down. In all three rows the CLT value lies below that lower end. The gap narrows as $k$ grows, and the ratio of the two rises from 0.638 to 0.979.

Using the exact per-step moments does not help. At $k = 10{,}000$ they give $\varepsilon = 5.9184$, or 5.8379 with the datasets swapped, both further below the bracket. So a "more exact" $\mu$ is not more conservative. Only a certified accountant is, one whose $\varepsilon$ is a proven upper bound. [Numerical accounting with privacy loss distributions](06-numerical-accounting.md) computes certified values, and [Choosing an accountant](09-choosing-an-accountant.md) compares all the accountants on this example.

![CLT approximation for DP-SGD with $q = 0.01$. Left: $\mu = q\sqrt{k(e^{1/\sigma^2} - 1)}$ against the number of steps for three noise multipliers. Right: the $\varepsilon$ at $\delta = 10^{-5}$ it implies through @eq-gdp-delta. Dots mark the running example at $k = 100$, 1,000 and 10,000.](figures/gaussian_dp_subsampled.svg){#fig-gdp-subsampled}

The formula is specific to Poisson sampling under add/remove neighbours. Under replace-one neighbours, $D'$ swaps one record of $D$ for another. Dong et al. (Corollary 5.4) analyse fixed-size batches under that relation and get a different constant, with $\mu = 1.7101$ at $k = 10{,}000$ in the running example. Do not mix the two ([Amplification by subsampling](05-subsampling.md) explains why).

## RDP as a lossy summary of the curve {#sec-gdp-rdp}

[Rényi DP](01-renyi-dp.md) tracks, for each order $\alpha$, a Rényi divergence $\varepsilon(\alpha)$ between the outputs on neighbouring datasets. Each value is a scaled logarithm of an exponential moment of the privacy loss. The trade-off curve fixes the distribution of the loss (@sec-gdp-clt), so it determines all of them. An RDP curve is therefore a summary of the trade-off curve, and the question is how much it keeps.

Going back from RDP to a curve rests on one observation, that a test turns any pair of distributions into a pair of coins. The test fires with probability $a$ under $D$, and with probability $1 - b$ under the neighbour. Its verdict is a function of the output, and post-processing cannot increase a Rényi divergence. So the two coins must stay within the RDP budget at every order.

Call the lowest type II error that passes this check at every order, in both directions, $f_{\mathrm{RDP}}(a)$. Here "every order" means the fixed grid of orders that this part's accountants evaluate. The full edition shows that $f_{\mathrm{RDP}}$ is a symmetric trade-off function, so a mechanism with the given RDP curve is $f_{\mathrm{RDP}}$-DP.

No higher curve follows from the RDP values alone. At each $a$, some mechanism with only two outputs satisfies the RDP values and reaches $f_{\mathrm{RDP}}(a)$, although different values of $a$ need different mechanisms. And the best $\delta(\varepsilon)$ that the RDP values can certify is the profile of $f_{\mathrm{RDP}}$, by @eq-gdp-conjugate. That is the optimal conversion of [From Rényi DP to (ε, δ) and to tests](02-rdp-conversions.md).

The results form a ladder of summaries. For the Gaussian mechanism, $G_\mu$ is the exact curve, $f_{\mathrm{RDP}}$ is what its RDP curve keeps, and an $(\varepsilon, \delta)$ pair is one touching line of $f_{\mathrm{RDP}}$. So the slack of an RDP conversion has two parts. The optimal conversion removes the gap between the line and $f_{\mathrm{RDP}}$, but no conversion removes the gap between $f_{\mathrm{RDP}}$ and the true curve.

RDP loses information because it keeps only the exponential moments of the loss, at a few orders. Different distributions of the loss share those moments, and $f_{\mathrm{RDP}}$ has to allow the worst of them. The gap in type II error is largest in the bulk of the curve. Yet the power that $f_{\mathrm{RDP}}$ allows stays roughly the same multiple of the Gaussian's from the bulk down to a type I error of $10^{-6}$, as [From Rényi DP to (ε, δ) and to tests](02-rdp-conversions.md) showed. So the loss is not confined to the bulk.

The Gaussian pair shows how much is lost even in the best case. Its RDP curve is $\varepsilon(\alpha) = \alpha\mu^2/2$, which RDP tracks exactly. For $\mu = 1$, the full edition computes $f_{\mathrm{RDP}}$ and finds it below $G_1$ everywhere it looked, a numerical observation rather than a proof. Take the type I error 0.3085, where $G_1$ crosses the diagonal. There a dense set of orders raises $f_{\mathrm{RDP}}$ from 0.2201 only to 0.2219, while the exact curve is at 0.3085.

@tbl-gdp-rdp compares the two curves. Its third row is the best advantage, which for a symmetric curve sits at the diagonal point where $a = b$ and equals $1 - 2a$. Its last row is the privacy profile of each curve at $\varepsilon = 1$.

| Quantity | Exact curve $G_1$ | $f_{\mathrm{RDP}}$ from $\varepsilon(\alpha) = \alpha/2$ |
| --- | --- | --- |
| $b$ at $a = \Phi(-1/2) = 0.3085$ | 0.3085 | 0.2201 |
| Common value of $a$ and $b$ at the diagonal point | 0.3085 | 0.2585 |
| Best advantage $\delta(0) = 1 - 2a$ at the diagonal point | 0.3829 | 0.4830 |
| $\delta(1)$ | 0.1269 | 0.2149 |

: The trade-off curve of 1-GDP against the best curve its RDP curve implies over the part's grid of orders, read at the diagonal and through the duality. {#tbl-gdp-rdp}

In the bulk, the RDP summary gives a lot away. It allows an attacker an advantage of 0.4830, where the best possible is 0.3829. The best $\delta(1)$ it can certify is 0.2149, against the exact 0.1269. @fig-gdp-rdp-lossy draws both curves with their touching lines at $\varepsilon = 1$. So for Gaussian noise without subsampling, account in GDP, where $\mu^2$ adds and the profile at the end is exact.

![$G_1$ (black) and the lowest curve $f_{\mathrm{RDP}}$ compatible with its RDP curve $\varepsilon(\alpha) = \alpha/2$ over the part's grid of orders (red, dashed). The thin lines are the $(\varepsilon, \delta)$ curves at $\varepsilon = 1$ that each one certifies, $f_{1, 0.1269}$ for $G_1$ and $f_{1, 0.2149}$ for $f_{\mathrm{RDP}}$; each touches its own curve at two mirror-image points (dots). An RDP conversion can only report lines under $f_{\mathrm{RDP}}$.](figures/gaussian_dp_rdp_lossy.svg){#fig-gdp-rdp-lossy}

## Code {#sec-gdp-code}

A GDP accountant needs only the normal distribution function, which Python's `math.erfc` gives. The snippet below computes the privacy profile, its inverse, composition and the CLT estimate for DP-SGD.

```python
from math import erfc, exp, sqrt


def normal_cdf(x: float) -> float:
    return 0.5 * erfc(-x / sqrt(2))


def gdp_delta(eps: float, mu: float) -> float:
    """The exact delta(eps) of mu-GDP."""
    return normal_cdf(-eps / mu + mu / 2) - exp(eps) * normal_cdf(-eps / mu - mu / 2)


def gdp_eps(mu: float, delta: float) -> float:
    """The eps at which mu-GDP reaches delta; delta(eps) falls as eps grows, so bisect."""
    lo, hi = 0.0, 500.0
    for _ in range(200):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if gdp_delta(mid, mu) > delta else (lo, mid)
    return hi


def compose_gdp(*mus: float) -> float:
    return sqrt(sum(m * m for m in mus))


def clt_mu_dpsgd(q: float, sigma: float, steps: int) -> float:
    """Approximate mu of Poisson-subsampled DP-SGD. An estimate, not a bound."""
    return q * sqrt(steps * (exp(1 / sigma**2) - 1))


print(f"{gdp_delta(1.0, 1.0):.6f}")  # 0.126937
print(f"{gdp_eps(1.0, 1e-5):.4f}")  # 4.3772
print(f"{gdp_eps(compose_gdp(1, 1, 1, 1), 1e-5):.4f}")  # 9.9973
mu = clt_mu_dpsgd(q=0.01, sigma=1.0, steps=10_000)
print(f"{mu:.4f} {gdp_eps(mu, 1e-5):.4f}")  # 1.3108 6.0071
```

The last line is the CLT estimate for the running example, which lies below the lower bound of 6.1377. The full edition's script, [gaussian_dp.py](../../code/part04-privacy-accounting/gaussian_dp.py), computes every number in this chapter. Run it from the repository root.

```bash
uv run python code/part04-privacy-accounting/gaussian_dp.py
```

## Cheat sheet

| Result | Formula | Notes |
| --- | --- | --- |
| $f$-DP | $T(M(D), M(D')) \ge f$ for all neighbours | Then also above its mirror image across the diagonal; survives post-processing |
| $(\varepsilon, \delta)$-DP as a curve | $f_{\varepsilon,\delta}(a) = \max(0, 1 - \delta - e^\varepsilon a, e^{-\varepsilon}(1 - \delta - a))$ | Two straight lines; randomized response sits on it |
| $\mu$-GDP | $G_\mu(a) = \Phi(\Phi^{-1}(1 - a) - \mu)$ | Gaussian mechanism: $\mu = 1/\sigma$, exact |
| Duality | $\delta(\varepsilon) = \max_a(1 - f(a) - e^\varepsilon a)$ for every $\varepsilon \ge 0$ | Lossless for a symmetric $f$; one pair is not |
| GDP profile | $\delta(\varepsilon) = \Phi(-\frac\varepsilon\mu + \frac\mu2) - e^\varepsilon\Phi(-\frac\varepsilon\mu - \frac\mu2)$ | Bisect for $\varepsilon$ at a target $\delta$ |
| Composition | $f_1 \otimes \dots \otimes f_k$; for Gaussian steps $\mu = \sqrt{\sum_i\mu_i^2}$ | Holds for adaptive steps; tight in general |
| CLT | $f_1 \otimes \dots \otimes f_k \approx G_\mu$, $\mu = 2K/\sqrt V$ | $K$, $V$: mean and variance of the summed loss; error band $\gamma$ |
| Subsampled Gaussian | $\mu \approx q\sqrt{k(e^{1/\sigma^2} - 1)}$ | Poisson sampling, add/remove; an approximation that under-reported $\varepsilon$ here |
| RDP from the curve | RDP implies $f_{\mathrm{RDP}}$-DP, below the true curve | Slack that no conversion removes |

: The results of this chapter. Type I error is $a$, type II error is $b$, and $\alpha$ is a Rényi order. {#tbl-gdp-cheat}

**Rules of thumb**

- The trade-off curve is the guarantee, and $(\varepsilon, \delta)$, RDP and $\rho$ are summaries of it. Summarize as late as possible.
- For Gaussian noise without subsampling, account in GDP. Then $\mu^2$ adds, and the $(\varepsilon, \delta)$ profile at the end is exact.
- At $\delta = 10^{-5}$, a guarantee of $\mu = 1$ means $\varepsilon \approx 4.4$. A guarantee of $\mu = 0.5$ means about 2.
- Use the CLT for intuition and quick estimates, never as the reported guarantee. Report a certified numerical accountant's value instead.

## Further reading

- Dong, Roth & Su, "Gaussian Differential Privacy" (Journal of the Royal Statistical Society Series B, 2022): the source of $f$-DP and GDP, with duality, composition and the CLTs. The discussion sections explain why the curve is the right object.
- Bu, Dong, Long & Su, "Deep Learning with Gaussian Differential Privacy" (Harvard Data Science Review, 2020): the CLT for Poisson-subsampled DP-SGD, with experiments.
- Gopi, Lee & Wutschitz, "Numerical Composition of Differential Privacy" (NeurIPS 2021): an accountant with error bounds, and examples where the CLT under-reports $\varepsilon$.
- Balle & Wang, "Improving the Gaussian Mechanism for Differential Privacy" (ICML 2018): the exact $(\varepsilon, \delta)$ profile of the Gaussian mechanism and its use for calibration.

**Derivations.** The full edition derives the results of this chapter step by step in [its chapter](../../book-full/part04-privacy-accounting/04-gaussian-dp.md), with these exceptions. It cites Blackwell's theorem, the adaptive case of the composition theorem, the error band and limit of the CLT, the hardness of exact composition and the replace-one constant of Dong et al. from their sources, and gives the CLT only as a sketch. It derives the subsampled-Gaussian limit heuristically, with the proof in Bu, Dong, Long & Su. That $f_{\mathrm{RDP}}$ lies below $G_1$ is a numerical observation.
