# Exercises: Information theory

Exercises for [Information theory](../../book-full/part01-prerequisites/05-information-theory.md), each with a worked solution. Try each exercise before opening its solution. The script `code/part01-prerequisites/solutions_information_theory.py` computes every number quoted here; run it from the repository root with `uv run python code/part01-prerequisites/solutions_information_theory.py`.

## Entropy of a geometric variable {#sec-sol-information-theory-1}

> **Exercise 1.** **Entropy of a geometric variable.** $X$ takes value $k \ge 1$ with probability $2^{-k}$. Show that $H(X) = 2$ bits, while $H_\infty(X) = 1$. *Hint: $\sum_k k2^{-k} = 2$.* *(Entropy definition)*

::: {.callout-tip collapse="true" title="Solution"}
1. **Surprisal of each outcome.** $p(k) = 2^{-k}$, so $\log\frac{1}{p(k)} = k$ bits. The outcome $k$ costs exactly $k$ bits, as in the code $1, 01, 001, \dots$
2. **Average it** (definition of entropy):
$$H(X) = \sum_{k \ge 1} p(k)\log\frac{1}{p(k)} = \sum_{k \ge 1} k\,2^{-k}.$$
3. **Sum the series** (the hint). Let $S = \sum_{k \ge 1} k\,2^{-k}$. Then $S/2 = \sum_{k \ge 1} k\,2^{-(k+1)} = \sum_{k \ge 2}(k - 1)2^{-k}$. Subtract term by term: $S - S/2 = \sum_{k \ge 1} 2^{-k} = 1$, so $S = 2$ and $H(X) = 2$ bits.
4. **Min-entropy.** The most likely value is $k = 1$, so $V(X) = \max_k 2^{-k} = \frac12$ and $H_\infty(X) = -\log\frac12 = 1$ bit.

The series converges fast. The partial sums up to $k = 5$, $10$ and $20$ are 1.7812, 1.9883 and 2.0000. As predicted by $H_\infty \le H$, the min-entropy is below the entropy. The bound $H \le \log\lvert\mathcal X\rvert$ says nothing here, because $X$ has infinitely many values.

*A variable with infinitely many values can still have a small finite entropy, and its best single guess (here $X = 1$) succeeds half the time.*
:::

## Information from independent observations {#sec-sol-information-theory-2}

> **Exercise 2.** **Information from independent observations.** If $Y_1$ and $Y_2$ are independent given $X$, show that $I(X; Y_1, Y_2) \le I(X; Y_1) + I(X; Y_2)$. Find an example where $Y_1, Y_2$ are *not* conditionally independent and the inequality fails. *Hint for the counterexample: $X = Y_1 \oplus Y_2$ for independent fair bits.* *(Chain rule)*

::: {.callout-tip collapse="true" title="Solution"}
*The inequality.* Use the symmetric form $I(X; Y_1, Y_2) = H(Y_1, Y_2) - H(Y_1, Y_2 \mid X)$.

1. **The noise part splits.** Apply @eq-chain-rule conditioned on $X$ throughout, as in the Fano derivation: $H(Y_1, Y_2 \mid X) = H(Y_1 \mid X) + H(Y_2 \mid Y_1, X)$. Given $X$, $Y_2$ is independent of $Y_1$, so the posterior of $Y_2$ given $(Y_1, X)$ is its posterior given $X$, and $H(Y_2 \mid Y_1, X) = H(Y_2 \mid X)$. So $H(Y_1, Y_2 \mid X) = H(Y_1 \mid X) + H(Y_2 \mid X)$.
2. **The total part is subadditive.** By the chain rule and "conditioning reduces entropy", $H(Y_1, Y_2) = H(Y_1) + H(Y_2 \mid Y_1) \le H(Y_1) + H(Y_2)$ (the second Check yourself question under Shannon entropy).
3. **Subtract.** Combining steps 1 and 2:
$$I(X; Y_1, Y_2) \le \big[H(Y_1) - H(Y_1 \mid X)\big] + \big[H(Y_2) - H(Y_2 \mid X)\big] = I(X; Y_1) + I(X; Y_2).$$

Two observations that are independent given $X$ can repeat each other's information, but they cannot create information together that neither has alone. Take for example a fair bit $X$ sent twice through a binary symmetric channel (BSC) with flip probability 0.1, with independent noise each time. Each copy carries $I(X; Y_i) = 0.5310$ bits, and the pair carries $I(X; Y_1, Y_2) = 0.7421 \le 1.0620$ bits.

*The counterexample.* Let $Y_1, Y_2$ be independent fair bits and $X = Y_1 \oplus Y_2$. All four pairs are equally likely:

| $Y_1$ | $Y_2$ | $X = Y_1 \oplus Y_2$ | probability |
| --- | --- | --- | --- |
| 0 | 0 | 0 | 1/4 |
| 0 | 1 | 1 | 1/4 |
| 1 | 0 | 1 | 1/4 |
| 1 | 1 | 0 | 1/4 |

: The exclusive-or counterexample: every row has probability 1/4. {#tbl-sol-information-theory-xor}

1. **One observation tells nothing.** In @tbl-sol-information-theory-xor, fix $Y_1 = 0$. The two remaining rows have $X = 0$ and $X = 1$, equally likely. The same happens for $Y_1 = 1$. So $X$ is a fair bit whatever $Y_1$ is, $X$ and $Y_1$ are independent, and $I(X; Y_1) = 0$. By symmetry, $I(X; Y_2) = 0$.
2. **Both observations tell everything.** Given $(Y_1, Y_2)$, $X$ is determined, so $H(X \mid Y_1, Y_2) = 0$ and $I(X; Y_1, Y_2) = H(X) - 0 = 1$ bit.
3. **The inequality fails:** $1 > 0 + 0$. The script confirms $I(X; Y_1) = I(X; Y_2) = 0.0000$ and $I(X; Y_1, Y_2) = 1.0000$.
4. **Why the proof breaks.** Given $X = 0$, the pair is $(0, 0)$ or $(1, 1)$: $Y_2 = Y_1$. So $Y_1$ and $Y_2$ are fully dependent given $X$, $H(Y_1, Y_2 \mid X) = 1$, while $H(Y_1 \mid X) + H(Y_2 \mid X) = 2$. Step 1 above fails.

*Information about $X$ can sit in the combination of features and in neither feature alone. This is why ranking features one at a time by information gain can miss useful pairs.*
:::

## Min-capacity bounds capacity for the binary symmetric channel {#sec-sol-information-theory-3}

> **Exercise 3.** **Min-capacity bounds capacity for the binary symmetric channel.** Show that $1 - h(p) \le \log_2\big(2\max(p, 1 - p)\big)$ for every $p$. *Hint: $H_\infty \le H$ applied to a $\mathrm{Bernoulli}(p)$ variable.* *Check: at $p = 0.25$, $0.1887 \le 0.5850$.* *(Min-entropy vs Shannon)*

::: {.callout-tip collapse="true" title="Solution"}
1. **Min-entropy of a coin.** For $B \sim \mathrm{Bernoulli}(p)$, $V(B) = \max(p, 1 - p)$, so $H_\infty(B) = -\log\max(p, 1 - p)$.
2. **Apply $H_\infty \le H$** (Min-entropy and guessing): $-\log\max(p, 1 - p) \le h(p)$.
3. **Rearrange.** Subtract both sides from 1: $1 - h(p) \le 1 + \log\max(p, 1 - p) = \log_2 2 + \log_2\max(p, 1 - p) = \log_2\big(2\max(p, 1 - p)\big)$.
4. **Read the two sides.** The left side is the capacity @eq-bsc-capacity. The right side is @eq-min-capacity for $\mathrm{BSC}(p)$, whose two column maxima are both $\max(p, 1 - p)$.
5. **Check.** At $p = 0.25$: $1 - h(0.25) = 0.1887$ and $\log_2(2 \cdot 0.75) = \log_2 1.5 = 0.5850$. At $p = 0.11$ the two are 0.5001 and 0.8319. On a grid of 1,001 values of $p$, the smallest gap between the two sides is 0.0000.

Equality holds exactly when $H_\infty(B) = H(B)$, that is when $B$ is uniform on its support: $p \in \{0, \frac12, 1\}$. That is where the grid finds the zero gap.

*For this channel the min-capacity is always at least the Shannon capacity. The two measures disagree, and the one-guess measure is the larger.*
:::

## The Z-channel {#sec-sol-information-theory-4}

> **Exercise 4.** **The Z-channel.** Input 0 always arrives as 0; input 1 arrives as 0 with probability $\frac12$. Show that $I(X; Y) = h(q/2) - q$ for $q = \Pr[X = 1]$, maximize over $q$, and confirm that the uniform input is *not* optimal. *Check: the optimum is $q = 0.4$ and the capacity is $\log_2(5/4) = 0.3219$ bits; the script confirms both with Blahut–Arimoto.* *(Capacity)*

::: {.callout-tip collapse="true" title="Solution"}
*The formula.* Use $I(X; Y) = H(Y) - H(Y \mid X)$, as for the binary symmetric channel.

1. **Output distribution.** $Y = 1$ only if $X = 1$ and the bit survives: $\Pr[Y = 1] = q \cdot \frac12 = q/2$. So $H(Y) = h(q/2)$.
2. **Noise entropy.** Given $X = 0$, $Y = 0$ for certain: $H(Y \mid X = 0) = 0$. Given $X = 1$, $Y$ is a fair bit: $H(Y \mid X = 1) = h(\frac12) = 1$. Averaging over $X$: $H(Y \mid X) = (1 - q)\cdot0 + q\cdot1 = q$.
3. **Subtract:** $I(X; Y) = h(q/2) - q$. The script compares this with the general joint-table formula on a grid of 1,001 values of $q$; they agree to $3.1\times10^{-16}$.

*The maximum.*

1. **Derivative of $h$.** Differentiating $h(t) = -t\log t - (1 - t)\log(1 - t)$ gives $h'(t) = \log\frac{1 - t}{t}$ (the $\pm 1/\ln 2$ terms cancel).
2. **Derivative of $I$.** By the chain rule of calculus, $\frac{d}{dq}h(q/2) = \frac12\log\frac{1 - q/2}{q/2} = \frac12\log\frac{2 - q}{q}$. So
$$\frac{dI}{dq} = \frac12\log_2\frac{2 - q}{q} - 1.$$
3. **Set it to zero.** $\log_2\frac{2 - q}{q} = 2$, so $\frac{2 - q}{q} = 4$ and $q = \frac25 = 0.4$.
4. **It is a maximum.** $h$ is concave and $q \mapsto q/2$ is linear, so $h(q/2)$ is concave, and subtracting the linear $q$ keeps it concave. A stationary point of a concave function is its maximum.
5. **The capacity.** $h(0.2) = 0.2\log_2 5 + 0.8\log_2\frac54 = \log_2 5 - 0.8\cdot 2 = \log_2 5 - 1.6$. So $I = \log_2 5 - 1.6 - 0.4 = \log_2 5 - 2 = \log_2\frac54 = 0.3219$ bits.
6. **Uniform input.** At $q = \frac12$, $I = h(\frac14) - \frac12 = 0.8113 - 0.5 = 0.3113$ bits, less than 0.3219. So the uniform input is not optimal.
7. **Numerical check.** A grid search over $q$ in steps of 0.001 peaks at $q = 0.4000$. Blahut–Arimoto gives capacity 0.3219 at $\Pr[X = 1] = 0.4000$.

The optimum puts less weight on input 1 because sending a 1 is expensive: half of the time it is lost, and that loss costs a full bit of noise entropy $H(Y \mid X = 1) = 1$.

*When a channel is not symmetric, the capacity-achieving input is not uniform.*
:::

## Capacities of the M-ary symmetric channel {#sec-sol-information-theory-5}

> **Exercise 5.** **Capacities of the $M$-ary symmetric channel.** Show that its Shannon capacity is $\log M - h(\varepsilon) - \varepsilon\log(M - 1)$ and its min-capacity is $\log\big(M(1 - \varepsilon)\big)$ when $1 - \varepsilon \ge \varepsilon/(M - 1)$. *Check: at $M = 8$ and $\varepsilon = 0.3$, 1.2765 and 2.4854 bits.* *(Capacity, min-capacity)*

::: {.callout-tip collapse="true" title="Solution"}
The channel keeps the input with probability $1 - \varepsilon$ and otherwise sends one of the other $M - 1$ symbols, each with probability $\varepsilon/(M - 1)$. Every row of $C$ holds one entry $1 - \varepsilon$ and $M - 1$ entries $\varepsilon/(M - 1)$, and so does every column.

*Shannon capacity.* Follow the binary symmetric channel derivation.

1. **Noise entropy does not depend on the input.** Row $x$ has entropy
$$H(Y \mid X = x) = -(1 - \varepsilon)\log(1 - \varepsilon) - (M - 1)\frac{\varepsilon}{M - 1}\log\frac{\varepsilon}{M - 1} = h(\varepsilon) + \varepsilon\log(M - 1),$$
using $-\varepsilon\log\frac{\varepsilon}{M - 1} = -\varepsilon\log\varepsilon + \varepsilon\log(M - 1)$. The rows are permutations of each other, so this holds for every $x$, and $H(Y \mid X) = h(\varepsilon) + \varepsilon\log(M - 1)$ for every prior.
2. **Maximize $H(Y)$.** $Y$ takes $M$ values, so $H(Y) \le \log M$, with equality iff $Y$ is uniform (the entropy upper bound).
3. **The uniform input attains it.** With $\pi(x) = 1/M$, $\Pr[Y = y] = \frac1M\sum_x C[x, y] = \frac1M\big[(1 - \varepsilon) + (M - 1)\frac{\varepsilon}{M - 1}\big] = \frac1M$.
4. **So** $\mathcal C = \log M - h(\varepsilon) - \varepsilon\log(M - 1)$.

At $M = 8$, $\varepsilon = 0.3$: $h(0.3) = 0.8813$ and $0.3\log_2 7 = 0.8422$, so $\mathcal C = 3 - 0.8813 - 0.8422 = 1.2765$ bits. Blahut–Arimoto gives 1.2765, and the input distribution it finds is exactly uniform.

*Min-capacity.*

1. **Column maxima.** Each column holds one entry $1 - \varepsilon$ and $M - 1$ entries $\varepsilon/(M - 1)$. Under the condition $1 - \varepsilon \ge \varepsilon/(M - 1)$, its maximum is $1 - \varepsilon$.
2. **Apply @eq-min-capacity:** $\mathcal{ML}(C) = \log\sum_{y=1}^{M}(1 - \varepsilon) = \log\big(M(1 - \varepsilon)\big)$.

At $M = 8$, $\varepsilon = 0.3$: $\varepsilon/(M - 1) = 0.0429 \le 0.7$, so the condition holds, and $\mathcal{ML} = \log_2 5.6 = 2.4854$ bits. The general column-maxima formula gives the same. If the condition fails, the column maximum is $\varepsilon/(M - 1)$ and $\mathcal{ML} = \log\frac{M\varepsilon}{M - 1}$.

*Under the same channel, a one-try guesser gains almost twice as many bits (2.4854) as the Shannon capacity (1.2765) suggests.*
:::

## Smith versus the average of logs {#sec-sol-information-theory-6}

> **Exercise 6.** **Smith versus the average of logs.** Prove that $H_\infty(X \mid Y) \le \sum_y p(y)H_\infty(X \mid Y = y)$, and find a joint table where the gap is large. *Check: the walk-through gives 0.5995 vs 0.6548.* *(Jensen)*

::: {.callout-tip collapse="true" title="Solution"}
*The inequality.*

1. **Write Smith's value as a log of an average.** By the derivation of @eq-posterior-vulnerability, $V(X \mid Y) = \sum_y p(y)\,V(X \mid Y = y)$, where $V(X \mid Y = y) = \max_x p(x \mid y)$. Let $g(y) = V(X \mid Y = y)$ and $Z = g(Y)$, a positive random variable. Then $H_\infty(X \mid Y) = -\log\mathbb E[Z]$.
2. **Write the other side as an average of logs.** $\sum_y p(y)H_\infty(X \mid Y = y) = \sum_y p(y)\big(-\log V(X \mid Y = y)\big) = \mathbb E[-\log Z]$.
3. **Apply Fact 1 in its mirror form** for the convex $-\log$: $-\log\mathbb E[Z] \le \mathbb E[-\log Z]$. That is the claim. Equality holds iff $Z$ is constant, that is, when the best guess is equally likely to be right after every observation.

In the walk-through, Smith's value is 0.5995 bits and the average of logs is 0.6548, as the check says.

*A large gap.* Reuse reveal-if-zero(1) from @sec-shannon-vs-min: $X$ is uniform on $n$-bit strings, and $Y$ is $X$ itself if its lowest bit is 0 (probability $\frac12$), and $\perp$ otherwise.

1. **If $Y$ is a string,** the posterior is a point mass: $V(X \mid Y = y) = 1$ and $H_\infty(X \mid Y = y) = 0$.
2. **If $Y = \perp$,** the posterior is uniform on the $2^{n-1}$ odd strings: $V(X \mid Y = \perp) = 2^{-(n-1)}$ and $H_\infty(X \mid Y = \perp) = n - 1$.
3. **Average of logs:** $\frac12\cdot0 + \frac12(n - 1) = \frac{n - 1}{2}$.
4. **Smith:** $V(X \mid Y) = \frac12\cdot1 + \frac12\cdot2^{-(n-1)} = \frac12 + 2^{-n}$, so $H_\infty(X \mid Y) = -\log\big(\tfrac12 + 2^{-n}\big) = n - \log\big(2^{n-1} + 1\big)$, just under 1 bit.

| $n$ | Smith $H_\infty(X \mid Y)$ | Average of $H_\infty(X \mid Y = y)$ |
| --- | --- | --- |
| 6 | 0.9556 | 2.5000 |
| 64 | 1.0000 | 31.5000 |

: Smith's conditional min-entropy against the average of logs for reveal-if-zero(1) on a uniform $n$-bit secret. The $n = 6$ row is computed from the explicit channel matrix and agrees with the closed forms. {#tbl-sol-information-theory-smith}

@tbl-sol-information-theory-smith shows that the gap grows without bound: about $\frac{n - 1}{2} - 1$ bits. The average of logs says the secret keeps 31.5 bits of guessing difficulty. But a guesser who sees $Y$ wins more than half the time, because half the time $Y$ hands over $X$, and the maximum a posteriori (MAP) guesser takes that half for free. Smith's value, about 1 bit, reflects exactly that.

*Averaging the logs lets one very uncertain outcome hide an outcome that reveals everything. Smith's definition keeps the meaning "log of a success probability".*
:::

## Fano is tight on the symmetric channel {#sec-sol-information-theory-7}

> **Exercise 7.** **Fano is tight on the symmetric channel.** Show that the MAP error of the $M$-ary symmetric channel with uniform input is $\varepsilon$ for $\varepsilon \le (M - 1)/M$, and that @eq-fano holds with equality. *(Fano)*

::: {.callout-tip collapse="true" title="Solution"}
*The MAP error.*

1. **Joint table.** With uniform input, $p(x, y) = \frac1M C[x, y]$: $\frac{1 - \varepsilon}{M}$ on the diagonal and $\frac{\varepsilon}{M(M - 1)}$ off it.
2. **When the diagonal wins.** $1 - \varepsilon \ge \frac{\varepsilon}{M - 1}$ iff $(M - 1)(1 - \varepsilon) \ge \varepsilon$ iff $M - 1 \ge M\varepsilon$ iff $\varepsilon \le \frac{M - 1}{M}$. Under this condition the MAP guess on observation $y$ is $x = y$.
3. **Success probability** by @eq-posterior-vulnerability: $V(X \mid Y) = \sum_{y=1}^{M}\frac{1 - \varepsilon}{M} = 1 - \varepsilon$. So the MAP error is $1 - V(X \mid Y) = \varepsilon$.

*Fano holds with equality.*

1. **The output is uniform.** Each column of $C$ sums to 1 (Exercise 5), so $p(y) = \frac1M$.
2. **The posterior is a column of $C$.** $p(x \mid y) = \frac{p(x, y)}{p(y)} = C[x, y]$: $1 - \varepsilon$ on $x = y$ and $\frac{\varepsilon}{M - 1}$ on each of the other $M - 1$ values.
3. **Its entropy** is the row entropy computed in Exercise 5: $H(X \mid Y = y) = h(\varepsilon) + \varepsilon\log(M - 1)$ for every $y$. Averaging over $y$ gives the same value for $H(X \mid Y)$.
4. **Compare with @eq-fano at $P_e = \varepsilon$.** The right side is $h(\varepsilon) + \varepsilon\log(M - 1)$, the same expression. So @eq-fano holds with equality for the MAP estimator.
5. **Why each step of the derivation is tight.** Step 3 there, $H(E \mid \hat X) \le H(E)$, is tight because the error $E$ has probability $\varepsilon$ whatever the guess is, so $E$ is independent of $\hat X$. Step 4 is tight because, given an error, the true value is uniform over the $M - 1$ wrong values. Step 6 is tight because the MAP guess is $\hat X = Y$, so $\hat X$ and $Y$ carry the same information.

Since the right side of @eq-fano increases in $P_e$ on $[0, (M - 1)/M]$, the Fano bound obtained by inverting it equals $\varepsilon$ exactly. @tbl-sol-information-theory-fano checks this at $M = 8$.

| $\varepsilon$ | MAP error | $H(X \mid Y)$ | $h(\varepsilon) + \varepsilon\log_2 7$ | Fano bound |
| --- | --- | --- | --- | --- |
| 0.1 | 0.1000 | 0.7497 | 0.7497 | 0.1000 |
| 0.2 | 0.2000 | 1.2834 | 1.2834 | 0.2000 |
| 0.5 | 0.5000 | 2.4037 | 2.4037 | 0.5000 |
| 0.875 | 0.8750 | 3.0000 | 3.0000 | 0.8750 |

: The $8$-ary symmetric channel with uniform input: MAP error, conditional entropy, the right side of Fano's inequality at $P_e = \varepsilon$, and the inverted Fano bound. The two entropy columns agree to floating-point precision. {#tbl-sol-information-theory-fano}

At $\varepsilon = \frac78$ the output is independent of the input, $H(X \mid Y) = \log_2 8 = 3$, and the best guess is right only by chance.

*Fano's inequality cannot be improved in general: this channel attains it at every noise level.*
:::

## A sample-size lower bound {#sec-sol-information-theory-8}

> **Exercise 8.** **A sample-size lower bound.** In Fano's method, how many samples must you have before the bound $P_e \ge 1 - (n\kappa + 1)/\log M$ allows $P_e \le 0.1$? Express $n$ in terms of $\kappa$ and $M$. *Check: with $M = 1024$ and $\kappa = 0.01$ bits, $n \ge 800$.* *(Fano's method)*

::: {.callout-tip collapse="true" title="Solution"}
1. **What "allows" means.** The bound rules out $P_e \le 0.1$ as long as its right side is above 0.1. It stops ruling it out when
$$1 - \frac{n\kappa + 1}{\log M} \le 0.1.$$
2. **Solve for $n$.** Rearrange to $\frac{n\kappa + 1}{\log M} \ge 0.9$, then $n\kappa \ge 0.9\log M - 1$, so
$$n \ge \frac{0.9\log M - 1}{\kappa}.$$
For a general target error $\delta$, replace 0.9 by $1 - \delta$.
3. **Check.** With $\log_2 1024 = 10$ and $\kappa = 0.01$: $n \ge (9 - 1)/0.01 = 800$. The script gives a bound of 0.1010 at $n = 799$ and exactly 0.1000 at $n = 800$.

This is a necessary condition, not a sufficient one. With fewer than 800 samples, every estimator has error above 0.1. With 800 or more, Fano no longer forbids it, but that does not mean an estimator reaches it; that depends on the problem. The sample size grows like $\log M/\kappa$: more candidates, or candidates that are harder to tell apart, push it up.

*Fano's method turns "each sample carries at most $\kappa$ bits" into a minimum sample size, roughly the number of bits needed to identify the candidate divided by the bits each sample carries.*
:::
