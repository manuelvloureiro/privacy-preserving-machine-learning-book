# Approximate Differential Privacy

Pure differential privacy (DP) bounds the ratio of two output probabilities by $e^\varepsilon$ on every event. This chapter adds a small additive slack $\delta$ to that bound. The relaxed definition, $(\varepsilon,\delta)$-DP, lets an algorithm fail with probability about $\delta$, and in exchange it admits the Gaussian mechanism, which adds less noise than the Laplace mechanism to statistics with many coordinates. It is also the setting in which the next chapter's advanced composition holds.

The chapter defines the relaxation and reads it through the privacy loss random variable. Two examples then show how badly a mechanism may behave with probability $\delta$, which forces $\delta$ to be much smaller than $1/n$. The Gaussian mechanism comes next, with a full privacy proof, and the chapter ends by checking that post-processing, group privacy and composition survive the relaxation.

## The definition {#sec-adp-definition}

An additive bound alone is the wrong notion of privacy. Suppose we asked only that, for all neighbours and all events,

$$
\Pr[M(D)\in S] \le \Pr[M(D')\in S] + \delta .
$$

This allows no useful accuracy when $\delta$ is small, and it allows harmful algorithms when $\delta$ is large (Vadhan, 2017, Section 1.6). The second failure appears in @sec-adp-delta. Combined with the multiplicative bound of pure DP, however, an additive term gives the right definition. Recall that two datasets in $\mathcal X^n$ are neighbours, written $D \simeq D'$, when they differ in exactly one record (the replace-one relation).

::: {#def-adp-dp name="Approximate differential privacy"}
Let $\varepsilon \ge 0$ and $\delta \in [0,1]$. A randomized algorithm $M\colon \mathcal X^n \to \mathcal Y$ is **$(\varepsilon,\delta)$-differentially private** if for all neighbours $D \simeq D'$ and all events $S \subseteq \mathcal Y$,

$$
\Pr[M(D) \in S] \le e^{\varepsilon}\Pr[M(D') \in S] + \delta .
$$ {#eq-adp-dp}
:::

At $\delta = 0$ this is the $\varepsilon$-DP of [the randomized response chapter](01-randomized-response.md), which from now on is called **pure DP**. A definition with $\delta > 0$ is called **approximate DP**. The rest of the chapter is about what the extra $\delta$ permits.

## The privacy loss random variable {#sec-adp-loss}

@def-adp-dp bounds probabilities of events, but it is easier to understand one outcome at a time. Fix neighbours $D \simeq D'$. An observer who sees an output $y$ and wonders whether the dataset was $D$ or $D'$ compares how likely $y$ is under each. Write $P$ and $Q$ for the distributions of $M(D)$ and $M(D')$, and $p(y)$ and $q(y)$ for the likelihood of the outcome $y$ under each. When $\mathcal Y$ is countable, these are the probabilities $p(y) = \Pr[M(D) = y]$ and $q(y) = \Pr[M(D') = y]$. When both outputs have ordinary densities, $p$ and $q$ are those densities. (In general, $p$ and $q$ are densities with respect to the measure $P + Q$, which always exist.)

The privacy loss is the logarithm of the ratio $p(y)/q(y)$. The ratio needs a convention in one case: an outcome that can occur under $D$ but never under $D'$, so that $p(y)$ is positive while $q(y)$ is zero. Seeing such an outcome rules $D'$ out entirely, and its loss is set to $+\infty$.

::: {#def-adp-loss name="Privacy loss"}
Let $p(y)$ and $q(y)$ be the probabilities, or densities, of an outcome $y$ under $M(D)$ and under $M(D')$. The **privacy loss** of an outcome $y$ with $p(y) > 0$ is

$$
L_{D,D'}(y) = \ln\frac{p(y)}{q(y)},
$$

where $L_{D,D'}(y) = +\infty$ if $q(y) = 0$. Once the pair is fixed we write it $L(y)$. The **privacy loss random variable** is $L = L(Y)$ with $Y = M(D)$: the privacy loss of the outcome the mechanism actually produces on $D$.
:::

The outcome $M(D)$ has $p(M(D)) > 0$ with probability $1$, so $L$ is always defined. A large value of $L$ means that the observed outcome is much more likely under $D$ than under $D'$, so it is evidence that the dataset is $D$. A negative value is evidence for $D'$, and an infinite value means that the outcome is impossible under $D'$.

The Laplace mechanism of [the previous chapter](02-laplace-mechanism.md) gives a first example. It applies to a function $f\colon\mathcal X^n \to \mathbb R$ whose value changes by at most $\Delta_1 f$ between neighbours. The mechanism releases $f(D) + Z$, where the noise $Z$ has the Laplace density $\frac{1}{2b}e^{-|z|/b}$ with scale $b = \Delta_1 f/\varepsilon$, and it is $\varepsilon$-DP.

::: {#exm-adp-laplace-loss}
Take neighbours with $f(D) = a$ and $f(D') = a - \Delta_1 f$, the largest gap allowed. The densities give

$$
L(y) = \frac{\varepsilon}{\Delta_1 f}\big(|y - a + \Delta_1 f| - |y - a|\big),
$$

which equals $\varepsilon$ for $y \ge a$, equals $-\varepsilon$ for $y \le a - \Delta_1 f$, and is linear in between.

1. The event $y \ge a$ has probability $\tfrac12$ under $P$, because the noise is symmetric. So $L$ has an atom of mass $\tfrac12$ at $\varepsilon$.
2. The event $y \le a - \Delta_1 f$ needs noise below $-\Delta_1 f = -\varepsilon b$, which has probability $\tfrac12 e^{-\varepsilon}$. So $L$ has an atom of mass $\tfrac12 e^{-\varepsilon}$ at $-\varepsilon$.
3. In between, write $y = a - s$ with $s \in (0, \Delta_1 f)$, so that $L = \varepsilon - 2\varepsilon s/(\Delta_1 f)$. The variable $s$ has density $\frac{\varepsilon}{2\Delta_1 f}e^{-\varepsilon s/(\Delta_1 f)}$, and the change of variables gives $L$ the density $u \mapsto \tfrac14 e^{(u - \varepsilon)/2}$ on $(-\varepsilon, \varepsilon)$, of total mass $\tfrac12(1 - e^{-\varepsilon})$.

The three masses add up to $1$. The privacy loss never leaves $[-\varepsilon, \varepsilon]$, as @fig-adp-loss shows on the left.
:::

![Privacy loss for two mechanisms on neighbours whose values differ by the full sensitivity. Left, the Laplace mechanism with $\varepsilon = 1$, from @exm-adp-laplace-loss: atoms of mass $0.5$ at $\varepsilon$ and $0.184$ at $-\varepsilon$, and a density in between. Right, the Gaussian mechanism with noise standard deviation $\sigma = \Delta_2 f$, from @lem-adp-gaussian-loss: the loss is Gaussian with mean $1/2$ and variance $1$, so it exceeds $\varepsilon = 1$ with probability $0.309$.](figures/adp-loss.svg){#fig-adp-loss}

The example is an instance of a general fact. Pure DP says that the privacy loss is bounded by $\varepsilon$ with probability $1$, and approximate DP says, roughly, that it is bounded by $\varepsilon$ except with probability $\delta$. Both follow from one identity, which computes the largest violation of the multiplicative bound over all events. That largest violation is the smallest $\delta$ that works for the pair. The identity writes it as an average over the privacy loss. It uses the positive part $(z)_+ = \max(z, 0)$, which keeps $z$ when it is positive and replaces it by $0$ otherwise. When the loss is infinite, the term $e^{\varepsilon - L}$ is read as $0$.

::: {#lem-adp-hockey}
For neighbours $D \simeq D'$ and $\varepsilon \ge 0$,

$$
\sup_{S \subseteq \mathcal Y}\Big(\Pr[M(D)\in S] - e^\varepsilon \Pr[M(D')\in S]\Big) = \mathbb E\Big[\big(1 - e^{\varepsilon - L}\big)_+\Big],
$$ {#eq-adp-hockey}

where $L$ is the privacy loss random variable of @def-adp-loss for this pair, $(z)_+ = \max(z, 0)$ is the positive part of $z$, and $e^{\varepsilon - L}$ is read as $0$ when $L = +\infty$. So $M$ is $(\varepsilon,\delta)$-DP if and only if the right-hand side is at most $\delta$ for every ordered pair of neighbours.
:::

::: {.proof}
1. For an event $S$, write the difference as an integral, $\Pr[M(D)\in S] - e^\varepsilon \Pr[M(D')\in S] = \int_S (p - e^\varepsilon q)$. The integrand is positive exactly on $B = \{y : p(y) > e^\varepsilon q(y)\}$, so the integral is largest for $S = B$, where it equals $\int (p - e^\varepsilon q)_+$.
2. Where $p(y) = 0$, the integrand $(p - e^\varepsilon q)_+$ is $0$. Where $p(y) > 0$, it equals $p(y)\big(1 - e^{\varepsilon}q(y)/p(y)\big)_+ = p(y)\big(1 - e^{\varepsilon - L(y)}\big)_+$, which is also right when $q(y) = 0$ and $L(y) = \infty$.
3. Integrating $p(y)$ times a function of $y$ is taking the expectation under $P$, so $\int(p - e^\varepsilon q)_+ = \mathbb E[(1 - e^{\varepsilon - L})_+]$.
4. @def-adp-dp asks that the left-hand side of @eq-adp-hockey be at most $\delta$ for every ordered pair of neighbours, which proves the last sentence.
:::

The right-hand side of @eq-adp-hockey is called the **hockey-stick divergence** of $P$ from $Q$, after the shape of the function $z \mapsto (1 - e^{\varepsilon - z})_+$. Its integrand is $0$ when $L \le \varepsilon$ and at most $1$ otherwise. So the lemma ties $(\varepsilon,\delta)$-DP to how often $L$ exceeds $\varepsilon$, and the next theorem makes that tie precise in both directions.

::: {#thm-adp-loss}
Let $M\colon\mathcal X^n \to \mathcal Y$ be a randomized algorithm, and let $L$ be its privacy loss random variable for an ordered pair of neighbours $D \simeq D'$.

1. $M$ is $\varepsilon$-DP if and only if $L \le \varepsilon$ with probability $1$ for every ordered pair of neighbours. In that case $|L| \le \varepsilon$ with probability $1$.
2. If $\Pr[L > \varepsilon] \le \delta$ for every ordered pair of neighbours, then $M$ is $(\varepsilon,\delta)$-DP.
3. If $M$ is $(\varepsilon,\delta)$-DP, then for every $t > 0$ and every ordered pair of neighbours,

$$
\Pr[L > \varepsilon + t] \le \frac{\delta}{1 - e^{-t}} .
$$
:::

::: {.proof}
1. With $\delta = 0$, @lem-adp-hockey says that $M$ is $\varepsilon$-DP exactly when $\mathbb E[(1 - e^{\varepsilon-L})_+] = 0$ for every pair. The integrand is nonnegative, so this holds exactly when it is $0$ with probability $1$, that is, when $L \le \varepsilon$ with probability $1$. For the lower bound, apply the same fact to the pair $(D', D)$. It gives $q \le e^\varepsilon p$ except on a set of $Q$-probability $0$, and that set also has $P$-probability $0$, because pure DP gives $\Pr[M(D)\in S] \le e^\varepsilon \Pr[M(D')\in S]$. So $L \ge -\varepsilon$ with probability $1$ under $P$.
2. The integrand $(1 - e^{\varepsilon - L})_+$ is $0$ when $L \le \varepsilon$ and at most $1$ when $L > \varepsilon$. So its expectation is at most $\Pr[L > \varepsilon] \le \delta$, and @lem-adp-hockey gives $(\varepsilon,\delta)$-DP.
3. On the event $L > \varepsilon + t$, the integrand is at least $1 - e^{-t}$. So $(1 - e^{-t})\Pr[L > \varepsilon + t] \le \mathbb E[(1 - e^{\varepsilon-L})_+] \le \delta$, by @lem-adp-hockey.
:::

The converse of part 2 fails with the same parameters, which is why part 3 pays in both $\varepsilon$ and $\delta$. For example, the Gaussian mechanism on the right of @fig-adp-loss has privacy loss above $1$ with probability $0.309$, yet @prp-adp-gaussian-exact below shows that it is $(1, 0.127)$-DP. Taking $t = \varepsilon$, an $(\varepsilon,\delta)$-DP mechanism has privacy loss above $2\varepsilon$ with probability at most $\delta/(1 - e^{-\varepsilon})$. Up to these constants, then, $(\varepsilon,\delta)$-DP says that the privacy loss is at most $\varepsilon$ except with probability about $\delta$. The definition says nothing about what happens on that rare event.

## How small δ must be {#sec-adp-delta}

On the event of probability $\delta$, the privacy loss may be infinite. The first example shows this in its crudest form.

::: {#exm-adp-release-all}
Let $\bot$ be a fixed symbol that carries no information about the data. The mechanism $M(D)$ outputs $\bot$ with probability $1 - \delta$ and outputs the whole dataset $D$ with probability $\delta$. For neighbours $D \ne D'$ and an event $S$,

$$
\Pr[M(D)\in S] \le (1-\delta)\,\mathbf 1[\bot \in S] + \delta \le \Pr[M(D')\in S] + \delta ,
$$

so $M$ is $(0,\delta)$-DP. With probability $1 - \delta$ the privacy loss is $\ln\frac{1-\delta}{1-\delta} = 0$, and with probability $\delta$ it is infinite, because the output $D$ is impossible under $D'$.
:::

This mechanism publishes everyone's data with probability $\delta$, so $\delta$ must be small. The next example shows how small. It publishes each record on its own, and that makes $\delta$ comparable to $1/n$.

::: {#def-adp-name-shame name="Name and shame"}
For $\delta \in [0,1]$, the mechanism $\mathrm{NS}_\delta$ draws independent coins $c_1, \dots, c_n$, each equal to $1$ with probability $\delta$, and outputs the set of pairs $\{(i, x_i) : c_i = 1\}$.
:::

Each record, which might be a person's medical file, is published in the clear with probability $\delta$. Even so, the mechanism meets the definition with $\varepsilon = 0$ and the same $\delta$, because a person's record affects the output only when that person's coin comes up $1$, which happens with probability $\delta$.

::: {#prp-adp-name-shame}
For every $\delta \in [0,1]$, $\mathrm{NS}_\delta$ is $(0,\delta)$-DP.
:::

::: {.proof}
1. Let $D$ and $D'$ differ only in record $i$, and run the mechanism on both datasets with the same coins. Let $E$ be the event $c_i = 1$, so $\Pr[E] = \delta$.
2. On the complement $\bar E$, record $i$ is not published, and the output is a function of the coins and of the records $j \ne i$, which the two datasets share. So on $\bar E$ the two outputs are equal.
3. For an event $S$, split on $E$ and use step 2:

$$
\Pr[\mathrm{NS}_\delta(D) \in S] = \Pr[\mathrm{NS}_\delta(D) \in S,\ \bar E] + \Pr[\mathrm{NS}_\delta(D)\in S,\ E] \le \Pr[\mathrm{NS}_\delta(D') \in S,\ \bar E] + \Pr[E] .
$$

4. The first term is at most $\Pr[\mathrm{NS}_\delta(D') \in S]$ and the second is $\delta$, which is @eq-adp-dp with $\varepsilon = 0$.
:::

So a mechanism that plainly harms people meets $(0,\delta)$-DP. How much harm it does depends on the chance that it publishes at least one record, which the next proposition computes.

::: {#prp-adp-name-shame-release}
The probability that $\mathrm{NS}_\delta$ publishes at least one record is $1 - (1-\delta)^n$, and

$$
n\delta - \frac{(n\delta)^2}{2} \le 1 - (1-\delta)^n \le n\delta .
$$
:::

::: {.proof}
1. The coins are independent, so no record is published with probability $(1-\delta)^n$, and the complement gives the formula.
2. The upper bound is the union bound over the $n$ records, each published with probability $\delta$.
3. For the lower bound, $1 - \delta \le e^{-\delta}$ gives $(1-\delta)^n \le e^{-n\delta}$. The inequality $e^{-x} \le 1 - x + x^2/2$ for $x \ge 0$ then gives $1 - (1-\delta)^n \ge n\delta - (n\delta)^2/2$.
:::

So the chance that someone is named is about $n\delta$ when $n\delta$ is small. At $\delta = 1/n$, step 3 of the proof gives a chance of at least $1 - e^{-1} \approx 0.632$ that some record is published in the clear. A guarantee that allows this is not privacy, so $\delta$ must be well below $1/n$. Even $\delta = n^{-1.1}$ allows a chance of up to $n\delta = n^{-0.1}$, which is $0.25$ for a million records. Common choices are $\delta \le 1/n^2$, or a $\delta$ that is **cryptographically small**, smaller than every inverse polynomial in $n$.

In both examples the privacy loss is $0$ or infinite, so the rare event is a catastrophe. Without more information about a mechanism, one has to assume this worst case. Many mechanisms degrade gracefully instead. Their privacy loss exceeds $\varepsilon$ with probability about $\delta$ but rarely exceeds it by much, and they satisfy a family of guarantees at once, such as $(1, 10^{-3})$-DP and $(2, 10^{-6})$-DP. The Gaussian mechanism of the next section is the main example, as @exm-adp-gaussian-numbers shows. A single pair $(\varepsilon,\delta)$ is a crude threshold on the distribution of the privacy loss. Rényi differential privacy and concentrated differential privacy describe the whole tail of that distribution, and they appear in later chapters.

The definition must be stated for events, not single outcomes. For pure DP the two are equivalent on countable outputs, by the pointwise lemma of [the randomized response chapter](01-randomized-response.md). For approximate DP the pointwise version is much weaker.

::: {#exm-adp-pointwise}
Let $m = \lceil 1/\delta\rceil$ for some $\delta \in (0,1)$. The mechanism $M(D) = (D, U)$ outputs the whole dataset together with a number $U$ drawn uniformly from $\{1, \dots, m\}$. Every single outcome has probability at most $1/m \le \delta$, so for all neighbours and all outcomes $y$,

$$
\Pr[M(D) = y] \le e^{\varepsilon}\Pr[M(D') = y] + \delta .
$$

Yet the event $S = \{(D, u) : 1 \le u \le m\}$ has probability $1$ under $D$ and $0$ under $D'$, so $M$ is not $(\varepsilon, \delta_0)$-DP for any $\varepsilon$ and any $\delta_0 < 1$.
:::

The privacy loss of this mechanism is infinite with probability $1$. So @thm-adp-loss, which looks at one outcome at a time through the privacy loss, catches the failure that the pointwise inequality misses.

## The Gaussian mechanism {#sec-adp-gaussian}

The Laplace mechanism calibrates its noise to the $\ell_1$ sensitivity. The Gaussian mechanism adds Gaussian noise instead, calibrated to the $\ell_2$ sensitivity, which can be much smaller.

::: {#def-adp-l2 name="ℓ2 sensitivity"}
The **$\ell_2$ sensitivity** of a function $f\colon \mathcal X^n \to \mathbb R^d$ is

$$
\Delta_2 f = \sup_{D \simeq D'} \|f(D) - f(D')\|_2 .
$$
:::

The $\ell_1$ sensitivity $\Delta_1 f$ of the previous chapter is the same supremum with the $\ell_1$ norm. The two norms, and so the two sensitivities, are within a factor $\sqrt d$ of each other.

::: {#lem-adp-norms}
For every $v \in \mathbb R^d$, $\|v\|_2 \le \|v\|_1 \le \sqrt d\,\|v\|_2$. Hence $\Delta_2 f \le \Delta_1 f \le \sqrt d\,\Delta_2 f$.
:::

::: {.proof}
1. Expanding the square, $\|v\|_1^2 = \sum_i v_i^2 + \sum_{i\ne j}|v_i||v_j| \ge \|v\|_2^2$.
2. The Cauchy–Schwarz inequality applied to $(|v_1|,\dots,|v_d|)$ and the all-ones vector gives $\|v\|_1 = \sum_i |v_i|\cdot 1 \le \|v\|_2\sqrt d$.
3. Applying both inequalities to $f(D) - f(D')$ and taking the supremum over neighbours gives the bounds on the sensitivities.
:::

The Gaussian distribution $\mathcal N(\mu, \sigma^2)$ with mean $\mu$ and variance $\sigma^2$ has density

$$
\frac{1}{\sqrt{2\pi\sigma^2}}\exp\!\left(-\frac{(x-\mu)^2}{2\sigma^2}\right).
$$

Its tails fall like $e^{-x^2/(2\sigma^2)}$, faster than the $e^{-|x|/b}$ of the Laplace distribution. We write $\mathcal N(0, \sigma^2 I_d)$ for the vector of $d$ independent $\mathcal N(0,\sigma^2)$ coordinates. The Gaussian mechanism adds such a vector to the true answer.

::: {#def-adp-gaussian name="Gaussian mechanism"}
For $f\colon\mathcal X^n \to \mathbb R^d$ and a noise level $\sigma > 0$, the Gaussian mechanism outputs

$$
M(D) = f(D) + Z, \qquad Z \sim \mathcal N(0, \sigma^2 I_d).
$$
:::

The theorem below says how large the noise level $\sigma$ must be for a given $\varepsilon$ and $\delta$.

::: {#thm-adp-gaussian name="Privacy of the Gaussian mechanism"}
Let $f$ have $0 < \Delta_2 f < \infty$, and let $\varepsilon > 0$ and $\delta \in (0,1)$. The Gaussian mechanism with

$$
\sigma = \frac{\Delta_2 f}{\varepsilon}\sqrt{2\ln(1/\delta) + \varepsilon}
$$ {#eq-adp-sigma}

is $(\varepsilon,\delta)$-DP.
:::
<!-- proof: in @sec-adp-gaussian-proof -->

The proof is in @sec-adp-gaussian-proof. Asymptotically the noise level is $\sigma = \frac{\Delta_2 f}{\varepsilon}\sqrt{2\ln(1/\delta)}\,(1 + o(1))$ as $\delta \to 0$. Compared with the Laplace mechanism, whose noise has scale $\Delta_1 f/\varepsilon$ in each coordinate, there are two differences. The Gaussian mechanism pays a factor $\sqrt{2\ln(1/\delta)}$ and gives only approximate DP, and it measures sensitivity in $\ell_2$, which can save a factor $\sqrt d$. The next example shows that the saving can be the larger effect.

::: {#exm-adp-mean}
Each of $n$ people answers $d$ yes-or-no questions, so the dataset is $D \in \{0,1\}^{n\times d}$ with rows $x_i \in \{0,1\}^d$. The analyst wants the vector of averages $f(D) = \frac1n\sum_i x_i$, with small error in $\ell_2$ norm.

1. Replacing row $x$ by row $x'$ changes $f$ by $(x - x')/n$. The largest change is from the all-ones row to the all-zeros row, so $\Delta_1 f = d/n$ and $\Delta_2 f = \sqrt d/n$.
2. The Laplace mechanism adds $\mathrm{Lap}(d/(\varepsilon n))$ to each coordinate, and each coordinate has variance $2d^2/(\varepsilon n)^2$. The root mean squared $\ell_2$ error is

$$
\sqrt{\mathbb E\|Z\|_2^2} = \sqrt{d \cdot \frac{2d^2}{\varepsilon^2n^2}} = \frac{\sqrt2\, d^{3/2}}{\varepsilon n}.
$$

3. The Gaussian mechanism with @eq-adp-sigma adds $\mathcal N(0,\sigma^2)$ to each coordinate with $\sigma = \frac{\sqrt d}{\varepsilon n}\sqrt{2\ln(1/\delta)+\varepsilon}$, so its root mean squared error is

$$
\sqrt d\,\sigma = \frac{d\sqrt{2\ln(1/\delta) + \varepsilon}}{\varepsilon n}.
$$

4. The ratio of the two errors is $\sqrt{2d/(2\ln(1/\delta)+\varepsilon)}$. So the Gaussian mechanism is more accurate as soon as $d > \ln(1/\delta) + \varepsilon/2$, and its advantage grows like $\sqrt d$.

With $d = 100$ questions, $n = 10{,}000$ people, $\varepsilon = 1$ and $\delta = 10^{-6}$, the Laplace error is $\sqrt2\cdot 1000/10{,}000 = 0.1414$ and the Gaussian error is $100\sqrt{27.63+1}/10{,}000 = 0.0535$, smaller by a factor $2.64$. A simulation of 20,000 runs of each mechanism gives $0.1413$ and $0.0535$.
:::

Which mechanism to use is a trade-off. The Laplace mechanism gives the stronger guarantee of pure DP, and the Gaussian mechanism gives a weaker guarantee with less noise in high dimension.

### Proof of privacy {#sec-adp-gaussian-proof}

The proof has two steps. First, the privacy loss of the Gaussian mechanism is itself Gaussian. Second, a Gaussian tail bound shows that it exceeds $\varepsilon$ with probability at most $\delta$, and part 2 of @thm-adp-loss concludes. The lemma below first collects four facts about Gaussians. The first two give the distribution of the privacy loss, and the last two bound its tail.

::: {#lem-adp-gaussian-facts}
Let $W \sim \mathcal N(0,1)$.

1. For constants $\mu$ and $a \ne 0$, $\mu + aW \sim \mathcal N(\mu, a^2)$.
2. If $V_1 \sim \mathcal N(0, a_1^2)$ and $V_2 \sim \mathcal N(0, a_2^2)$ are independent, then $V_1 + V_2 \sim \mathcal N(0, a_1^2 + a_2^2)$.
3. For every $u \ge 0$, $\Pr[W > u] \le e^{-u^2/2}$.
4. For every $u > 0$, $\Pr[W > u] \le e^{-u^2/2}/(u\sqrt{2\pi})$.
:::

::: {.proof}
1. Completing the square in the density gives the moment generating function $\mathbb E[e^{\lambda W}] = e^{\lambda^2/2}$. A distribution is determined by its moment generating function when that function is finite near $0$, as it is for all the variables here.
2. For part 1, $\mathbb E[e^{\lambda(\mu + aW)}] = e^{\lambda\mu + \lambda^2a^2/2}$, which is the moment generating function of $\mathcal N(\mu, a^2)$.
3. For part 2, independence makes the moment generating function of the sum the product $e^{\lambda^2a_1^2/2}e^{\lambda^2a_2^2/2} = e^{\lambda^2(a_1^2+a_2^2)/2}$.
4. For part 3, Markov's inequality applied to $e^{uW}$ gives $\Pr[W > u] \le e^{-u^2}\,\mathbb E[e^{uW}] = e^{-u^2/2}$.
5. For part 4, write $\varphi(z) = e^{-z^2/2}/\sqrt{2\pi}$ for the density, whose derivative is $-z\varphi(z)$. On $z \ge u$ the ratio $z/u$ is at least $1$, so $\Pr[W > u] \le \int_u^\infty \frac zu\varphi(z)\,\mathrm d z = \frac{\varphi(u)}{u}$.
:::

The first step of the plan is to compute the privacy loss. It turns out to be Gaussian, with a mean and a variance set by the distance between $f(D)$ and $f(D')$.

::: {#lem-adp-gaussian-loss}
Let $D \simeq D'$, let $M$ be the Gaussian mechanism with noise level $\sigma$, and write $v = f(D) - f(D')$. The privacy loss random variable of $M(D)$ against $M(D')$ is distributed as

$$
L \sim \mathcal N\!\left(\rho,\ 2\rho\right), \qquad \rho = \frac{\|v\|_2^2}{2\sigma^2}.
$$ {#eq-adp-gaussian-loss}
:::

::: {.proof}
1. The output is $f(D) + z$ with $z \sim \mathcal N(0, \sigma^2 I_d)$. Its density under $D$ is proportional to $\exp(-\|z\|_2^2/(2\sigma^2))$. Under $D'$, the same point is at distance $z + v$ from the mean $f(D')$, so its density is proportional to $\exp(-\|z + v\|_2^2/(2\sigma^2))$, with the same constant.
2. The constants cancel in the ratio, and the logarithm gives

$$
L = \frac{\|z+v\|_2^2 - \|z\|_2^2}{2\sigma^2} = \frac{1}{2\sigma^2}\sum_{j=1}^d\big((z_j+v_j)^2 - z_j^2\big) = \frac{1}{2\sigma^2}\sum_{j=1}^d \big(2z_jv_j + v_j^2\big).
$$

3. The constant part is $\sum_j v_j^2/(2\sigma^2) = \rho$, the claimed mean.
4. The random part is $\frac{1}{\sigma^2}\sum_j v_jz_j$, and the terms with $v_j = 0$ vanish. By part 1 of @lem-adp-gaussian-facts each remaining term $v_jz_j$ is $\mathcal N(0, \sigma^2v_j^2)$, and the terms are independent. By part 2 their sum is $\mathcal N(0, \sigma^2\|v\|_2^2)$. Dividing by $\sigma^2$ gives $\mathcal N(0, \|v\|_2^2/\sigma^2) = \mathcal N(0, 2\rho)$.
:::

The mean and the variance of the privacy loss are tied, with variance twice the mean. For a single coordinate, $d = 1$, the proof is just the identity $L = vz/\sigma^2 + v^2/(2\sigma^2)$. A Gaussian privacy loss is special to this mechanism. The Laplace mechanism, by @exm-adp-laplace-loss, has a loss with atoms.

::: {.proof}
*Proof of @thm-adp-gaussian.*

1. By part 2 of @thm-adp-loss, it suffices to show $\Pr[L > \varepsilon] \le \delta$ for every ordered pair of neighbours. If $v = 0$ then $L = 0$, so assume $v \ne 0$.
2. By @lem-adp-gaussian-loss and part 1 of @lem-adp-gaussian-facts, $L = \rho + \sqrt{2\rho}\,W$ with $W \sim \mathcal N(0,1)$. So

$$
\Pr[L > \varepsilon] = \Pr\left[W > g(\rho)\right], \qquad g(\rho) = \frac{\varepsilon - \rho}{\sqrt{2\rho}} = \frac{\varepsilon}{\sqrt{2\rho}} - \sqrt{\frac{\rho}{2}} .
$$

3. The function $g$ decreases in $\rho$, and $\rho \le (\Delta_2 f)^2/(2\sigma^2)$ by the definition of $\Delta_2 f$. Write $t = \sqrt{2\ln(1/\delta)+\varepsilon}$, so that @eq-adp-sigma reads $\sigma = t\,\Delta_2 f/\varepsilon$ and the largest value of $\rho$ is $\varepsilon^2/(2t^2)$. Then

$$
g(\rho) \ge g\!\left(\frac{\varepsilon^2}{2t^2}\right) = t - \frac{\varepsilon}{2t} =: u .
$$

4. The value $u$ is positive, because $t^2 \ge \varepsilon > \varepsilon/2$. Its square is $u^2 = t^2 - \varepsilon + \varepsilon^2/(4t^2) \ge t^2 - \varepsilon = 2\ln(1/\delta)$.
5. By part 3 of @lem-adp-gaussian-facts, $\Pr[L > \varepsilon] \le \Pr[W > u] \le e^{-u^2/2} \le e^{-\ln(1/\delta)} = \delta$.
:::

Only the upper tail of $L$ matters in this proof, because part 2 of @thm-adp-loss bounds $\Pr[L > \varepsilon]$, not $\Pr[|L| > \varepsilon]$. For $\varepsilon \le 1$ the noise level @eq-adp-sigma is at most $\frac{\Delta_2 f}{\varepsilon}\sqrt{2\ln(1.65/\delta)}$. The sharper tail bound of part 4 of @lem-adp-gaussian-facts lowers the constant $1.65$ to $1.25$, which is the form of the Gaussian mechanism that later chapters use.

::: {#cor-adp-gaussian-classic name="Gaussian mechanism for ε ≤ 1"}
Let $f$ have $0 < \Delta_2 f < \infty$, and let $\varepsilon \in (0,1]$ and $\delta \in (0,1)$. The Gaussian mechanism with

$$
\sigma = \frac{\Delta_2 f}{\varepsilon}\sqrt{2\ln(1.25/\delta)}
$$

is $(\varepsilon,\delta)$-DP.
:::

::: {.proof}
1. Write $t = \sqrt{2\ln(1.25/\delta)}$, so that $\sigma = t\,\Delta_2 f/\varepsilon$ and $e^{-t^2/2} = \delta/1.25$. Steps 1 to 3 of the proof of @thm-adp-gaussian used only $\sigma = t\,\Delta_2 f/\varepsilon$, so they give $\Pr[L > \varepsilon] \le \Pr[W > u]$ with $u = t - \varepsilon/(2t)$.
2. Since $\varepsilon \le 1$, the value $u$ is at least $t - 1/(2t)$, which increases in $t$. Also $u^2 = t^2 - \varepsilon + \varepsilon^2/(4t^2) \ge t^2 - 1$.
3. *Case $\delta \le 0.74$.* Then $t \ge \sqrt{2\ln(1.25/0.74)} \ge 1.0239$, and by step 2, $u \ge 0.5356$. By part 4 of @lem-adp-gaussian-facts and step 2,

$$
\Pr[W > u] \le \frac{e^{-u^2/2}}{u\sqrt{2\pi}} \le \frac{e^{1/2}\,e^{-t^2/2}}{0.5356\sqrt{2\pi}} = \frac{e^{1/2}}{1.25 \cdot 0.5356\sqrt{2\pi}}\,\delta \le 0.983\,\delta .
$$

4. *Case $\delta > 0.74$.* Then $t > \sqrt{2\ln 1.25} \ge 0.6680$, and by step 2, $u > -0.0805$. The density of $W$ is at most $1/\sqrt{2\pi}$, so $\Pr[W > u] \le \tfrac12 + 0.0805/\sqrt{2\pi} \le 0.533 < \delta$.
5. In both cases $\Pr[L > \varepsilon] \le \delta$, and part 2 of @thm-adp-loss gives $(\varepsilon,\delta)$-DP.
:::

This is the bound of Dwork and Roth (2014, Theorem A.1). On a grid of $\varepsilon$ from $0.01$ to $1$ and $\delta$ from $0.89$ down to $10^{-20}$, the exact $\delta$ of @prp-adp-gaussian-exact at this noise level is at most $0.233$ times the target. Both constants are loose, because the tail bounds of @lem-adp-gaussian-facts are loose. The exact $\delta$ comes from evaluating the identity of @lem-adp-hockey for the Gaussian privacy loss, instead of bounding its tail.

::: {#prp-adp-gaussian-exact}
Let $\Phi$ be the distribution function of $\mathcal N(0,1)$, let $f$ have $0 < \Delta_2 f < \infty$, and let $s = \Delta_2 f/\sigma$. For every $\varepsilon \ge 0$, the Gaussian mechanism with noise level $\sigma$ is $(\varepsilon, \delta(\varepsilon))$-DP for

$$
\delta(\varepsilon) = \Phi\!\left(\frac s2 - \frac\varepsilon s\right) - e^{\varepsilon}\,\Phi\!\left(-\frac s2 - \frac\varepsilon s\right),
$$ {#eq-adp-gaussian-exact}

and for no smaller $\delta$ when some pair of neighbours has $\|f(D) - f(D')\|_2 = \Delta_2 f$.
:::

::: {.proof}
1. Fix a pair of neighbours and let $r = \|v\|_2/\sigma \in (0, s]$, so that $\rho = r^2/2$ and $L = r^2/2 + rW$. By @lem-adp-hockey, the smallest $\delta$ for this pair is $\mathbb E[(1 - e^{\varepsilon - L})\mathbf 1[L > \varepsilon]]$.
2. The event $L > \varepsilon$ is $W > c$ with $c = \varepsilon/r - r/2$, so it has probability $\Phi(r/2 - \varepsilon/r)$.
3. Write $\varphi$ for the standard normal density. Completing the square gives $e^{-rz}\varphi(z) = e^{r^2/2}\varphi(z + r)$, so $\mathbb E[e^{-rW}\mathbf 1[W > c]] = e^{r^2/2}\,\Phi(-c - r)$. Since $e^{\varepsilon - L} = e^{\varepsilon - r^2/2}e^{-rW}$ and $-c-r = -r/2 - \varepsilon/r$,

$$
\mathbb E\big[e^{\varepsilon - L}\mathbf 1[L>\varepsilon]\big] = e^{\varepsilon - r^2/2}\,e^{r^2/2}\,\Phi\!\left(-\frac r2 - \frac\varepsilon r\right) = e^\varepsilon\,\Phi\!\left(-\frac r2 - \frac\varepsilon r\right).
$$
4. Subtracting step 3 from step 2 gives @eq-adp-gaussian-exact with $r$ in place of $s$. Call that function $h(r)$.
5. Write $c_1 = r/2 - \varepsilon/r$ and $c_2 = -r/2 - \varepsilon/r$. Then $(c_1^2 - c_2^2)/2 = (c_1-c_2)(c_1+c_2)/2 = -\varepsilon$, so $e^\varepsilon\varphi(c_2) = \varphi(c_1)$. Differentiating, $h'(r) = \varphi(c_1)(\tfrac12 + \varepsilon/r^2) + e^\varepsilon\varphi(c_2)(\tfrac12 - \varepsilon/r^2) = \varphi(c_1) > 0$.
6. So $h$ increases in $r$, the worst pair is the one with the largest $r$, and $h(r) \le h(s) = \delta(\varepsilon)$, with equality when some pair attains $\Delta_2 f$.
:::

The exact formula shows the graceful degradation promised in @sec-adp-delta. At a single noise level, the Gaussian mechanism meets a whole family of guarantees.

::: {#exm-adp-gaussian-numbers}
For $\varepsilon = 1$ and $\delta = 10^{-5}$, @eq-adp-sigma gives $\sigma = \sqrt{2\ln(10^5) + 1}\,\Delta_2 f = 4.9016\,\Delta_2 f$. At this noise level, @eq-adp-gaussian-exact gives the following guarantees, all at once:

- $(0.5,\ 6.1\times 10^{-4})$-DP,
- $(1,\ 3.0\times 10^{-8})$-DP,
- $(2,\ 3.0\times 10^{-24})$-DP.

The guarantee at $\varepsilon = 1$ is far better than the $\delta = 10^{-5}$ that was asked for, and the smallest noise level that gives $(1, 10^{-5})$-DP is $\sigma = 3.7306\,\Delta_2 f$. As $\varepsilon$ grows, $\delta$ falls faster than exponentially, so the privacy loss of this mechanism degrades gracefully.
:::

## Properties {#sec-adp-properties}

The properties of pure DP proved in [the Laplace mechanism chapter](02-laplace-mechanism.md) carry over to approximate DP. Post-processing and composition carry over unchanged, and group privacy carries over with a worse $\delta$. Post-processing comes first. It says that whatever an analyst computes from a private output, without going back to the data, is as private as the output.

::: {#thm-adp-post name="Post-processing"}
Let $M\colon\mathcal X^n \to \mathcal Y$ be $(\varepsilon,\delta)$-DP, and let $F\colon\mathcal Y \to \mathcal Z$ be a randomized map whose coins are independent of those of $M$. Then $F \circ M$ is $(\varepsilon,\delta)$-DP.
:::

::: {.proof}
1. Write $F(y) = g_R(y)$, where $R$ is the coins of $F$ and each $g_r$ is a deterministic function. So $F$ is a random choice of a deterministic function.
2. For neighbours $D \simeq D'$ and an event $S \subseteq \mathcal Z$, condition on $R$ and apply @eq-adp-dp to the event $g_R^{-1}(S)$:

$$
\Pr[F(M(D))\in S] = \mathbb E_R\Big[\Pr[M(D) \in g_R^{-1}(S)]\Big] \le \mathbb E_R\Big[e^\varepsilon\Pr[M(D') \in g_R^{-1}(S)] + \delta\Big] = e^\varepsilon\Pr[F(M(D'))\in S] + \delta .
$$
:::

For group privacy, recall that an $\varepsilon$-DP mechanism satisfies the bound with $e^{k\varepsilon}$ on datasets that differ in $k$ records. With approximate DP, the $\delta$ is multiplied by the factors $e^\varepsilon$ that it meets along the way.

::: {#thm-adp-group name="Group privacy"}
Let $M$ be $(\varepsilon,\delta)$-DP, and let $D, D' \in \mathcal X^n$ differ in $k \ge 1$ records. For every event $S$,

$$
\Pr[M(D)\in S] \le e^{k\varepsilon}\Pr[M(D')\in S] + \delta\sum_{j=0}^{k-1}e^{j\varepsilon} \le e^{k\varepsilon}\Pr[M(D')\in S] + k e^{(k-1)\varepsilon}\delta .
$$ {#eq-adp-group}
:::

::: {.proof}
1. Change the differing records one at a time to get a chain $D = D^{(0)} \simeq D^{(1)} \simeq \dots \simeq D^{(k)} = D'$ of neighbours. Write $P_j = \Pr[M(D^{(j)})\in S]$.
2. We show $P_0 \le e^{j\varepsilon}P_j + \delta\sum_{i=0}^{j-1}e^{i\varepsilon}$ by induction on $j$. The case $j = 1$ is @eq-adp-dp.
3. If the bound holds for $j$, apply @eq-adp-dp to $D^{(j)} \simeq D^{(j+1)}$, which gives $P_j \le e^{\varepsilon}P_{j+1} + \delta$. Substituting,

$$
P_0 \le e^{j\varepsilon}\big(e^\varepsilon P_{j+1} + \delta\big) + \delta\sum_{i=0}^{j-1}e^{i\varepsilon} = e^{(j+1)\varepsilon}P_{j+1} + \delta\sum_{i=0}^{j}e^{i\varepsilon} .
$$

4. At $j = k$ this is the first inequality in @eq-adp-group. Each of the $k$ terms of the sum is at most $e^{(k-1)\varepsilon}$, which gives the second.
:::

The additive term grows exponentially in the group size. It exceeds $1$, and the bound says nothing, once $k$ is a little over $\ln(1/\delta)/\varepsilon$.

Composition needs one more tool. The definition bounds probabilities of events, and the proof also needs averages of functions with values in $[0,1]$.

::: {#lem-adp-functions}
Let $M$ be $(\varepsilon,\delta)$-DP and let $h\colon\mathcal Y \to [0,1]$. For all neighbours $D \simeq D'$,

$$
\mathbb E[h(M(D))] \le e^\varepsilon\,\mathbb E[h(M(D'))] + \delta .
$$
:::

::: {.proof}
1. A variable with values in $[0,1]$ has mean equal to the integral of its tail, so $\mathbb E[h(M(D))] = \int_0^1\Pr[h(M(D)) > t]\,\mathrm d t$.
2. Each tail is the probability of the event $\{y : h(y) > t\}$, which @eq-adp-dp bounds by $e^\varepsilon\Pr[h(M(D')) > t] + \delta$.
3. Integrating over $t \in [0,1]$ gives $e^\varepsilon\,\mathbb E[h(M(D'))] + \delta$.
:::

In an adaptive composition, $M_1$ runs on the dataset, then $M_2$ is chosen in the light of the output $y_1$ of $M_1$ and runs on the same dataset, and so on. Formally, for each $i$ and each sequence of earlier outputs $y_{<i} = (y_1,\dots,y_{i-1})$, the mechanism $M_i(\cdot\,; y_{<i})$ is $(\varepsilon_i,\delta_i)$-DP, with fresh coins. The composition releases $(y_1, \dots, y_k)$.

::: {#thm-adp-composition name="Basic composition"}
If each $M_i(\cdot\,;y_{<i})$ is $(\varepsilon_i, \delta_i)$-DP for every $y_{<i}$, where the parameters $\varepsilon_i$ and $\delta_i$ are fixed in advance, then the adaptive composition of $M_1, \dots, M_k$ is $\big(\sum_i\varepsilon_i, \sum_i\delta_i\big)$-DP.
:::

::: {.proof}
1. The composition of the first $k-1$ mechanisms is itself a mechanism. So by induction on $k$ it suffices to treat $k = 2$.
2. Fix neighbours $D \simeq D'$ and an event $S$ in $\mathcal Y_1\times\mathcal Y_2$. For an output $y_1$, write $S_{y_1} = \{y_2 : (y_1,y_2)\in S\}$ and $g_D(y_1) = \Pr[M_2(D; y_1) \in S_{y_1}]$, and define $g_{D'}$ in the same way. Conditioning on the first output,

$$
\Pr[M(D)\in S] = \mathbb E_{y_1\sim M_1(D)}\big[g_D(y_1)\big].
$$

3. Since $M_2(\cdot\,;y_1)$ is $(\varepsilon_2,\delta_2)$-DP and $g_D \le 1$, we have $g_D(y_1) \le \min\{1, e^{\varepsilon_2}g_{D'}(y_1) + \delta_2\} \le h(y_1) + \delta_2$, where $h = \min\{1, e^{\varepsilon_2}g_{D'}\}$ takes values in $[0,1]$.
4. By @lem-adp-functions applied to $M_1$ and $h$,

$$
\Pr[M(D)\in S] \le \mathbb E_{M_1(D)}[h] + \delta_2 \le e^{\varepsilon_1}\,\mathbb E_{M_1(D')}[h] + \delta_1 + \delta_2 .
$$

5. Finally $h \le e^{\varepsilon_2}g_{D'}$, and $\mathbb E_{y_1\sim M_1(D')}[g_{D'}(y_1)] = \Pr[M(D')\in S]$ by step 2 applied to $D'$. So $\Pr[M(D)\in S] \le e^{\varepsilon_1+\varepsilon_2}\Pr[M(D')\in S] + \delta_1+\delta_2$.
:::

The parameters must be fixed in advance. If the analyst could choose $\varepsilon_i$ in the light of earlier outputs, the sum would no longer be a valid bound without further work. With $k$ mechanisms that are each $(\varepsilon,\delta)$-DP, basic composition gives $(k\varepsilon, k\delta)$-DP. [The next chapter](04-advanced-composition.md) proves that, for every $\delta' > 0$, the same composition is also $(\varepsilon', k\delta + \delta')$-DP with

$$
\varepsilon' = \varepsilon\sqrt{2k\ln(1/\delta')} + k\varepsilon\,(e^\varepsilon - 1).
$$

When $\varepsilon$ is of order $1/\sqrt k$ or smaller, the second term is of the same order as the first or smaller, so $\varepsilon'$ grows like $\sqrt k$ rather than $k$. That allows quadratically more analyses for the same privacy, and it requires $\delta' > 0$ even when every mechanism is pure.

## Summary {#sec-adp-summary}

| Result | Statement |
| --- | --- |
| @def-adp-dp | $\Pr[M(D)\in S] \le e^\varepsilon\Pr[M(D')\in S] + \delta$ for all neighbours and events |
| @lem-adp-hockey | the largest violation over events is $\mathbb E[(1-e^{\varepsilon-L})_+]$ |
| @thm-adp-loss | pure DP is $L \le \varepsilon$ almost surely; $\Pr[L > \varepsilon] \le \delta$ implies $(\varepsilon,\delta)$-DP; the converse holds with $\varepsilon + t$ and $\delta/(1-e^{-t})$ |
| @prp-adp-name-shame | publishing each record with probability $\delta$ is $(0,\delta)$-DP, so $\delta$ must be well below $1/n$ |
| @exm-adp-pointwise | for approximate DP, single outcomes do not suffice |
| @lem-adp-norms | $\Delta_2 f \le \Delta_1 f \le \sqrt d\,\Delta_2 f$ |
| @lem-adp-gaussian-loss | the Gaussian mechanism's privacy loss is $\mathcal N(\rho, 2\rho)$ with $\rho = \|f(D)-f(D')\|_2^2/(2\sigma^2)$ |
| @thm-adp-gaussian | $\sigma = \Delta_2 f\sqrt{2\ln(1/\delta)+\varepsilon}/\varepsilon$ gives $(\varepsilon,\delta)$-DP |
| @cor-adp-gaussian-classic | for $\varepsilon \le 1$, $\sigma = \Delta_2 f\sqrt{2\ln(1.25/\delta)}/\varepsilon$ suffices |
| @prp-adp-gaussian-exact | exact $\delta(\varepsilon) = \Phi(s/2 - \varepsilon/s) - e^\varepsilon\Phi(-s/2-\varepsilon/s)$ with $s = \Delta_2 f/\sigma$ |
| @thm-adp-post | post-processing keeps $(\varepsilon,\delta)$ |
| @thm-adp-group | groups of $k$: $(k\varepsilon,\ k e^{(k-1)\varepsilon}\delta)$ |
| @thm-adp-composition | adaptive composition adds the $\varepsilon_i$ and the $\delta_i$ |

: The chapter's results. {#tbl-adp-summary}

## References {#sec-adp-notes}

- Cynthia Dwork, Krishnaram Kenthapadi, Frank McSherry, Ilya Mironov and Moni Naor, "Our Data, Ourselves: Privacy via Distributed Noise Generation", 2006. The definition of $(\varepsilon,\delta)$-DP and the first analysis of Gaussian noise.
- Cynthia Dwork and Aaron Roth, *The Algorithmic Foundations of Differential Privacy*, 2014. The privacy loss random variable (Section 3.5), the Gaussian mechanism with the constant $\sqrt{2\ln(1.25/\delta)}$ (Theorem A.1), and the composition theorems (Theorems 3.16 and 3.20).
- Salil Vadhan, "The Complexity of Differential Privacy", 2017. Section 1.6 explains why an additive bound alone fails, and the tutorial proves the properties of approximate DP.
- Shiva Prasad Kasiviswanathan and Adam Smith, "On the 'Semantics' of Differential Privacy: A Bayesian Formulation", 2014. The relation between $(\varepsilon,\delta)$-DP and tail bounds on the privacy loss.
- Borja Balle and Yu-Xiang Wang, "Improving the Gaussian Mechanism for Differential Privacy: Analytical Calibration and Optimal Denoising", 2018. The exact $\delta$ of the Gaussian mechanism, @eq-adp-gaussian-exact, and its calibration.
- Cynthia Dwork, Guy Rothblum and Salil Vadhan, "Boosting and Differential Privacy", 2010. Advanced composition.
- Ryan Rogers, Aaron Roth, Jonathan Ullman and Salil Vadhan, "Privacy Odometers and Filters: Pay-as-you-Go Composition", 2016. What goes wrong, and what can be saved, when the privacy parameters are chosen adaptively.
- Ilya Mironov, "Rényi Differential Privacy", 2017, Cynthia Dwork and Guy Rothblum, "Concentrated Differential Privacy", 2016, and Mark Bun and Thomas Steinke, "Concentrated Differential Privacy: Simplifications, Extensions, and Lower Bounds", 2016. Definitions that describe the whole distribution of the privacy loss rather than one threshold.
