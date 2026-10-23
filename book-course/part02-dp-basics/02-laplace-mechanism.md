# The Laplace Mechanism

Randomized response privatizes one bit per person, and it extends to a value from a few categories, but no further. Most questions an analyst asks are numbers computed from the whole dataset, such as a count, an average or a histogram. This chapter answers them with the Laplace mechanism, which computes the true answer and adds noise scaled to how much one person can change it. On the fraction of ones that randomized response estimates, its error is of order $1/(\varepsilon n)$ rather than $1/(\varepsilon\sqrt n)$.

The second half of the chapter proves three properties of differential privacy (DP) that make it possible to build private algorithms from private parts. Post-processing says that a private output stays private whatever is done with it. Group privacy bounds what is revealed about several people at once, and basic composition bounds the privacy cost of running several private algorithms on the same data.

## Sensitivity {#sec-lap-sensitivity}

Differential privacy asks that the output hide the contribution of any one person. So the noise must be at least as large as the change that one person can cause in the answer, and the sensitivity measures the largest such change.

::: {#def-lap-sensitivity name="ℓ1 sensitivity"}
The **$\ell_1$ sensitivity** of a function $f\colon\mathcal X^n \to \mathbb R^k$ is

$$
\Delta_1(f) = \sup_{X \sim X'}\|f(X) - f(X')\|_1 ,
$$

where $\|z\|_1 = \sum_{j=1}^k|z_j|$. We write $\Delta_1$ when $f$ is clear.
:::

The choice of the $\ell_1$ norm is what makes the privacy proof of @thm-lap-privacy work. Other mechanisms use other norms, such as the $\ell_2$ norm for the Gaussian mechanism of [the next chapter](03-approximate-dp.md). For $k = 1$ all these norms are the absolute value of the difference. For $k > 1$ the $\ell_1$ and $\ell_2$ sensitivities can differ by a factor of up to $\sqrt k$, as the next chapter proves.

::: {#exm-lap-mean-sensitivity}
Let $\mathcal X = \{0,1\}$ and let $f(X) = \frac1n\sum_{i=1}^n X_i$ be the fraction of ones. Changing one bit changes the sum by at most $1$, so $\Delta_1(f) = 1/n$.
:::

## The Laplace distribution {#sec-lap-distribution}

::: {#def-lap-distribution name="Laplace distribution"}
The **Laplace distribution** $\mathrm{Lap}(b)$ with scale $b > 0$ has density

$$
\frac{1}{2b}\exp\!\left(-\frac{|x|}{b}\right), \qquad x \in \mathbb R.
$$
:::

It is a two-sided exponential distribution. The exponential distribution has density proportional to $e^{-x/b}$ on $[0,\infty)$, and the Laplace distribution reflects it to the negative half-line. Its tails decay like $e^{-|x|/b}$, more slowly than the $e^{-x^2/(2\sigma^2)}$ of a Gaussian, but still exponentially.

::: {#lem-lap-moments}
Let $Y \sim \mathrm{Lap}(b)$. Then $\mathbb E[Y] = 0$, $\operatorname{Var}[Y] = 2b^2$, and for every $t \ge 0$,

$$
\Pr[|Y| \ge tb] = e^{-t}.
$$ {#eq-lap-tail}
:::

::: {.proof}
1. The density is symmetric about $0$, so the mean is $0$.
2. The absolute value $|Y|$ has density $\frac1b e^{-y/b}$ on $[0,\infty)$, the exponential distribution with mean $b$. Its second moment is $\int_0^\infty y^2\frac1be^{-y/b}\,dy = 2b^2$, by integrating by parts twice. Since the mean of $Y$ is $0$, $\operatorname{Var}[Y] = \mathbb E[Y^2] = 2b^2$.
3. The tail of $|Y|$ is $\int_{tb}^\infty \frac1be^{-y/b}\,dy = e^{-t}$.
:::

## The Laplace mechanism {#sec-lap-mechanism}

The mechanism adds independent Laplace noise to each coordinate, with scale equal to the sensitivity divided by $\varepsilon$.

::: {#def-lap-mechanism name="Laplace mechanism"}
For $f\colon\mathcal X^n\to\mathbb R^k$ with $0 < \Delta_1(f) < \infty$ and for $\varepsilon > 0$, the Laplace mechanism outputs

$$
M(X) = f(X) + (Y_1, \dots, Y_k),
$$

where $Y_1, \dots, Y_k$ are independent with distribution $\mathrm{Lap}(\Delta_1(f)/\varepsilon)$.
:::

Before proving that the mechanism is private, consider its accuracy on the running example. The mechanism is unbiased, and its error follows from @lem-lap-moments.

::: {#exm-lap-mean}
For $f(X) = p$, the fraction of ones, @exm-lap-mean-sensitivity gives $\Delta_1 = 1/n$. The mechanism outputs $\tilde p = p + Y$ with $Y \sim \mathrm{Lap}(1/(\varepsilon n))$. So $\mathbb E[\tilde p] = p$, and $\operatorname{Var}[\tilde p] = 2/(\varepsilon n)^2$. Chebyshev's inequality already gives an error of order $1/(\varepsilon n)$: with probability at least $3/4$, the error is at most two standard deviations, $2\sqrt2/(\varepsilon n)$. The exact tail does better for small failure probabilities $\beta$. By @eq-lap-tail with $t = \ln(1/\beta)$, with probability at least $1 - \beta$,

$$
|\tilde p - p| \le \frac{\ln(1/\beta)}{\varepsilon n}.
$$

With $\varepsilon = 1$, $n = 10{,}000$ and $\beta = 0.05$, this is $\ln(20)/10{,}000 = 3.0\times 10^{-4}$. Randomized response with the same $\varepsilon$, $n$ and $\beta$ has the bound $0.0294$, which is $98$ times larger.
:::

The gap grows with $n$, because randomized response has error of order $1/(\varepsilon\sqrt n)$ and the Laplace mechanism of order $1/(\varepsilon n)$. The Laplace mechanism needs a trusted curator who sees the true data, while randomized response does not. That is the price of the smaller error. It remains to prove that the mechanism is private.

::: {#thm-lap-privacy name="Privacy of the Laplace mechanism"}
For every $f\colon\mathcal X^n\to\mathbb R^k$ with $0 < \Delta_1(f) < \infty$ and every $\varepsilon > 0$, the Laplace mechanism is $\varepsilon$-DP.
:::

::: {.proof}
1. Fix neighbours $X \sim X'$, and write $p$ and $q$ for the densities of $M(X)$ and $M(X')$ on $\mathbb R^k$. Let $b = \Delta_1/\varepsilon$. The coordinates are independent, so at a point $z \in \mathbb R^k$,

$$
\frac{p(z)}{q(z)} = \prod_{j=1}^k \frac{\exp(-|f(X)_j - z_j|/b)}{\exp(-|f(X')_j - z_j|/b)} = \exp\!\left(\frac{\varepsilon}{\Delta_1}\sum_{j=1}^k\big(|f(X')_j - z_j| - |f(X)_j - z_j|\big)\right).
$$

2. By the triangle inequality, $|f(X')_j - z_j| - |f(X)_j - z_j| \le |f(X')_j - f(X)_j|$ for each $j$. Summing over $j$ gives at most $\|f(X) - f(X')\|_1$, which is at most $\Delta_1$ by @def-lap-sensitivity. So $p(z) \le e^\varepsilon q(z)$ for every $z$.
3. For an event $T \subseteq \mathbb R^k$, integrate the pointwise bound over $T$: $\Pr[M(X)\in T] = \int_T p \le e^\varepsilon\int_T q = e^\varepsilon\Pr[M(X')\in T]$.
:::

The proof shows why the Laplace distribution fits the definition. Its density falls exponentially in the distance from the true value, and the definition bounds a ratio by an exponential, so moving the centre by the sensitivity changes the density by a factor of at most $e^\varepsilon$.

@fig-lap-mechanism shows the two densities for a one-dimensional statistic. They have the same shape, shifted by the sensitivity, and the ratio of their heights never exceeds $e^\varepsilon$.

![Left, the densities of the Laplace mechanism on two neighbours whose values differ by the sensitivity $\Delta_1 = 1$, at $\varepsilon = 1$: their ratio stays between $e^{-1}$ and $e$. Right, the error in estimating a fraction of ones, as the 95th percentile over 2,000 simulated runs with $p = 0.3$. Randomized response falls as $1/\sqrt n$ and the Laplace mechanism as $1/n$. The dashed line is the exact 95th percentile $\ln(20)/(\varepsilon n)$ from @eq-lap-tail.](figures/lap-mechanism.svg){#fig-lap-mechanism}

## Counting queries {#sec-lap-counting}

A **counting query** asks how many people in the dataset have some property. It is given by a predicate $\phi\colon\mathcal X \to \{0,1\}$ and has the value $f(X) = \sum_{i=1}^n\phi(X_i)$. Changing one record changes the count by at most $1$, so $\Delta_1 = 1$. The Laplace mechanism adds $\mathrm{Lap}(1/\varepsilon)$, whose error is at most $\ln(1/\beta)/\varepsilon$ with probability $1-\beta$, whatever the size of the dataset. On a large dataset, an error of a few units is small next to a count of thousands.

Many counting queries $f = (f_1, \dots, f_k)$, with predicates $\phi_1, \dots, \phi_k$, are answered together on the same data. For example, $\phi_1$ might ask whether a person smokes and $\phi_2$ whether their income is above some threshold. One person can then change all of the answers at once.

::: {#prp-lap-counting}
The vector of $k$ counting queries has $\ell_1$ sensitivity at most $k$, with equality when some record satisfies every predicate and another satisfies none. For every $\beta \in (0,1)$, with probability at least $1 - \beta$, the Laplace mechanism answers all $k$ queries with error at most

$$
\frac{k\ln(k/\beta)}{\varepsilon}.
$$
:::

::: {.proof}
1. Let $X$ and $X'$ differ in one entry, $x$ in $X$ and $x'$ in $X'$. In each query the other terms cancel, so $\|f(X) - f(X')\|_1 = \sum_{j=1}^k|\phi_j(x) - \phi_j(x')| \le k$. The sum equals $k$ when $\phi_j(x) = 1$ and $\phi_j(x') = 0$ for all $j$.
2. The noise in each answer is $\mathrm{Lap}(k/\varepsilon)$. By @eq-lap-tail with $t = \ln(k/\beta)$, each noise is at least $k\ln(k/\beta)/\varepsilon$ in absolute value with probability $\beta/k$.
3. The union bound over the $k$ answers gives a failure probability of at most $\beta$.
:::

This analysis needs all $k$ queries to be fixed in advance. The composition theorem of @sec-lap-properties gives the same privacy when each query is chosen after seeing the earlier answers.

The error of order $k/\varepsilon$ per query should be compared with the reconstruction attacks of [the reconstruction chapter](../part01-privacy-failures/02-reconstruction.md). There, about $n$ counting queries answered with error $o(\sqrt n)$ let an attacker recover almost the whole dataset. Here, $n$ counting queries are answered privately with error of order $n/\varepsilon$, up to a logarithm. Between $\sqrt n$ and $n$ there is a wide gap. Either better attacks exist or less noise suffices, and the second is true. [Advanced composition](04-advanced-composition.md), under a relaxation of the definition, lowers the error to about $\sqrt{k}/\varepsilon$ per query, up to logarithmic factors.

## Histograms {#sec-lap-histograms}

When the queries have structure, a sharper sensitivity bound is possible. In a **histogram**, each record falls into exactly one of $k$ bins, such as an age in whole years or an eye colour, and the query counts the records in each bin: $f_j(X) = |\{i : X_i \text{ is in bin } j\}|$.

::: {#prp-lap-histogram}
A histogram with $k \ge 2$ bins has $\ell_1$ sensitivity $2$. For every $\beta \in (0,1)$, with probability at least $1-\beta$, the Laplace mechanism, which adds $\mathrm{Lap}(2/\varepsilon)$ to each bin, answers every bin with error at most

$$
\frac{2\ln(k/\beta)}{\varepsilon}.
$$
:::

::: {.proof}
1. Let $X$ and $X'$ differ in one entry, which lies in bin $a$ in $X$ and in bin $b$ in $X'$. If $a = b$ the histograms are equal. Otherwise count $a$ drops by $1$ and count $b$ rises by $1$, so $f(X) - f(X') = e_a - e_b$, where $e_j$ is the $j$-th unit vector, and $\|e_a - e_b\|_1 = 2$.
2. By @eq-lap-tail with $t = \ln(k/\beta)$, each of the $k$ noises is at least $2\ln(k/\beta)/\varepsilon$ in absolute value with probability $\beta/k$. The union bound over the bins finishes the proof.
:::

The sensitivity is $2$ because a replacement moves a record from one bin to another. Under neighbours that add or remove a record, only one count changes, and the sensitivity is $1$. A single bin has error at most $2\ln(1/\beta)/\varepsilon$ with probability $1-\beta$, and the proposition pays only $\ln k$ more to control all bins at once. So the noise per bin does not depend on the number of bins, and the error over all bins grows only like $\ln k$, which makes private histograms over very large domains practical. For $k = 100$ bins, $\varepsilon = 1$ and $\beta = 0.05$, the bound is $2\ln(2000) = 15.20$, and in a simulation of 20,000 runs the 95th percentile of the largest error is $15.17$. The same $100$ bins treated as arbitrary counting queries would get the bound $100\ln(2000) = 760.1$ from @prp-lap-counting.

## Properties of differential privacy {#sec-lap-properties}

Three properties let one reason about a private algorithm piece by piece.

### Post-processing {#sec-lap-post}

Once an output is private, no computation on it can undo the privacy, as long as the computation does not look at the data again. The randomized response chapter already used this, when it turned the private reports into an estimate.

::: {#thm-lap-post name="Post-processing"}
Let $M\colon\mathcal X^n\to\mathcal Y$ be $\varepsilon$-DP, and let $F\colon\mathcal Y\to\mathcal Z$ be a randomized map whose coins are independent of those of $M$. Then $F\circ M$ is $\varepsilon$-DP.
:::

::: {.proof}
1. A randomized map is a random choice of a deterministic one. Write $F(y) = g_R(y)$, where $R$ is the coins of $F$ and each $g_r\colon\mathcal Y\to\mathcal Z$ is deterministic.
2. Fix neighbours $X\sim X'$ and an event $T\subseteq\mathcal Z$. For fixed coins, $g_R(M(X))\in T$ exactly when $M(X) \in g_R^{-1}(T)$, so

$$
\Pr[F(M(X))\in T] = \mathbb E_R\Big[\Pr\big[M(X)\in g_R^{-1}(T)\big]\Big].
$$

3. Inside the expectation, apply the definition of $\varepsilon$-DP to the event $g_R^{-1}(T)$, then undo step 2 for $X'$:

$$
\mathbb E_R\Big[\Pr\big[M(X)\in g_R^{-1}(T)\big]\Big] \le e^\varepsilon\,\mathbb E_R\Big[\Pr\big[M(X')\in g_R^{-1}(T)\big]\Big] = e^\varepsilon\Pr[F(M(X'))\in T].
$$
:::

### Group privacy {#sec-lap-group}

The definition protects one record. Datasets that differ in several records, such as the records of one family, are still protected, with a guarantee that degrades in proportion to their number.

::: {#thm-lap-group name="Group privacy"}
Let $M\colon\mathcal X^n\to\mathcal Y$ be $\varepsilon$-DP, and let $X, X'\in\mathcal X^n$ differ in $k$ entries. For every event $T\subseteq\mathcal Y$,

$$
\Pr[M(X)\in T] \le e^{k\varepsilon}\Pr[M(X')\in T].
$$
:::

::: {.proof}
1. The proof is a **hybrid argument**. Change the $k$ differing entries one at a time, from their values in $X$ to their values in $X'$. This gives datasets $X = X^{(0)}, X^{(1)}, \dots, X^{(k)} = X'$ in which each consecutive pair are neighbours.
2. Apply the definition to each consecutive pair, from the first to the last:

$$
\Pr[M(X^{(0)})\in T] \le e^\varepsilon\Pr[M(X^{(1)})\in T] \le e^{2\varepsilon}\Pr[M(X^{(2)})\in T] \le \dots \le e^{k\varepsilon}\Pr[M(X^{(k)})\in T].
$$
:::

So a group of $k$ people has the protection of $k\varepsilon$-DP, which is meaningful only while $k\varepsilon$ is small. The theorem also relates the two notions of neighbours in [the randomized response chapter](01-randomized-response.md). A replacement is a removal followed by an addition, so a mechanism that is $\varepsilon$-DP when neighbours add or remove a record is $2\varepsilon$-DP when neighbours replace one.

### Basic composition {#sec-lap-composition}

An analyst rarely runs one algorithm. In an **adaptive composition**, $M_1$ runs on the dataset, then $M_2$ is chosen in the light of the output $y_1$ and runs on the same dataset, and so on up to $M_k$. Formally, for each $i$ and each sequence of earlier outputs $y_{<i} = (y_1, \dots, y_{i-1})$, the mechanism $M_i(\cdot\,; y_{<i})$ is a mechanism on $\mathcal X^n$ with fresh coins, and the composition releases $(y_1,\dots,y_k)$.

::: {#thm-lap-composition name="Basic composition"}
If each $M_i(\cdot\,; y_{<i})$ is $\varepsilon_i$-DP for every $y_{<i}$, then the adaptive composition of $M_1, \dots, M_k$ is $\big(\sum_{i=1}^k\varepsilon_i\big)$-DP. In particular, $k$ mechanisms that are each $\varepsilon$-DP compose to a $k\varepsilon$-DP mechanism.
:::

The proof is simplest for countable outputs, where the probability of a sequence of outputs factors into conditional probabilities.

::: {.proof}
*Countable outputs.*

1. Fix neighbours $X\sim X'$ and outputs $y = (y_1,\dots,y_k)$. Each $M_i$ uses fresh coins, so the chain rule gives $\Pr[M(X) = y] = \prod_{i=1}^k\Pr[M_i(X; y_{<i}) = y_i]$, and likewise for $X'$.
2. Each factor for $X$ is at most $e^{\varepsilon_i}$ times the matching factor for $X'$, by the privacy of $M_i(\cdot\,; y_{<i})$ applied to the event $\{y_i\}$. Multiplying the $k$ bounds gives

$$
\Pr[M(X) = y] = \prod_{i=1}^k\Pr[M_i(X; y_{<i}) = y_i] \le \prod_{i=1}^k e^{\varepsilon_i}\Pr[M_i(X'; y_{<i}) = y_i] = \exp\!\Big(\sum_{i=1}^k\varepsilon_i\Big)\Pr[M(X') = y].
$$

3. The pointwise lemma of the randomized response chapter turns this bound on single outcomes into the definition.
:::

For outputs that are not countable, such as the real numbers that the Laplace mechanism releases, the same idea works with averages in place of products. It uses a fact proved in the randomized response chapter, as the first step of its hypothesis-testing theorem. If $M$ is $\varepsilon$-DP and a function $h$ takes values in $[0,1]$, then for all neighbours

$$
\mathbb E[h(M(X))] \le e^\varepsilon\,\mathbb E[h(M(X'))].
$$

::: {.proof}
*General outputs.*

1. The composition of the first $k-1$ mechanisms is itself a mechanism, so by induction on $k$ it suffices to treat $k = 2$.
2. Fix neighbours $X\sim X'$ and an event $T$ of pairs. For an output $y_1$, write $T_{y_1} = \{y_2 : (y_1,y_2)\in T\}$ and $g_D(y_1) = \Pr[M_2(D; y_1)\in T_{y_1}]$ for a dataset $D$. Conditioning on the first output gives $\Pr[M(D)\in T] = \mathbb E_{y_1\sim M_1(D)}[g_D(y_1)]$.
3. By the privacy of $M_2(\cdot\,; y_1)$, $g_X(y_1) \le e^{\varepsilon_2}g_{X'}(y_1)$ for every $y_1$.
4. The function $g_{X'}$ takes values in $[0,1]$, so the fact above, applied to $M_1$, gives

$$
\Pr[M(X)\in T] \le e^{\varepsilon_2}\,\mathbb E_{y_1\sim M_1(X)}[g_{X'}(y_1)] \le e^{\varepsilon_1+\varepsilon_2}\,\mathbb E_{y_1\sim M_1(X')}[g_{X'}(y_1)] = e^{\varepsilon_1+\varepsilon_2}\Pr[M(X')\in T].
$$
:::

Basic composition recovers the analysis of @prp-lap-counting. Answering each counting query with $\mathrm{Lap}(k/\varepsilon)$ noise is $(\varepsilon/k)$-DP, and $k$ of them compose to $\varepsilon$-DP, now even when the queries are chosen adaptively. The cost of $k$ analyses grows linearly in $k$. It can be made to grow like $\sqrt k$, but only under the relaxation of the definition that [the next chapter](03-approximate-dp.md) introduces.

## Summary {#sec-lap-summary}

| Result | Statement |
| --- | --- |
| @def-lap-sensitivity | $\Delta_1(f)$ is the largest $\ell_1$ change in $f$ between neighbours |
| @lem-lap-moments | $\mathrm{Lap}(b)$ has variance $2b^2$ and $\Pr[\lvert Y\rvert \ge tb] = e^{-t}$ |
| @thm-lap-privacy | adding $\mathrm{Lap}(\Delta_1/\varepsilon)$ to each coordinate is $\varepsilon$-DP |
| @exm-lap-mean | a fraction of ones with error $\ln(1/\beta)/(\varepsilon n)$ |
| @prp-lap-counting | $k$ counting queries: sensitivity $k$, error $k\ln(k/\beta)/\varepsilon$ |
| @prp-lap-histogram | a histogram: sensitivity $2$, error $2\ln(k/\beta)/\varepsilon$ over all bins |
| @thm-lap-post | post-processing keeps $\varepsilon$-DP |
| @thm-lap-group | datasets differing in $k$ entries: factor $e^{k\varepsilon}$ |
| @thm-lap-composition | adaptive composition adds the $\varepsilon_i$ |

: The chapter's results. {#tbl-lap-summary}

## Notes {#sec-lap-notes}

- Cynthia Dwork, Frank McSherry, Kobbi Nissim and Adam Smith, "Calibrating Noise to Sensitivity in Private Data Analysis", 2006. The Laplace mechanism and the notion of sensitivity.
- Irit Dinur and Kobbi Nissim, "Revealing Information while Preserving Privacy", 2003. The reconstruction attacks that the error of the Laplace mechanism is compared with.
- Cynthia Dwork and Aaron Roth, *The Algorithmic Foundations of Differential Privacy*, 2014. Section 3.3 covers the Laplace mechanism, counting queries and histograms, and Section 2.3 and Section 3.5 cover post-processing, group privacy and composition.
- Cynthia Dwork, Guy Rothblum and Salil Vadhan, "Boosting and Differential Privacy", 2010. Advanced composition, which improves the linear growth of basic composition.
