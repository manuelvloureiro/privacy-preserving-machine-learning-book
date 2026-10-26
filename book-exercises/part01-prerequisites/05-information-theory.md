# Exercises: Information theory

Exercises for [Information theory](../../book-full/part01-prerequisites/05-information-theory.md), each with a worked solution. Try each exercise before opening its solution. The script `code/part01-prerequisites/solutions_information_theory.py` computes every number quoted here; run it from the repository root with `uv run python code/part01-prerequisites/solutions_information_theory.py`.

## Entropy of a geometric variable {#sec-sol-min-1}

> **Exercise 1.** **Entropy of a geometric variable.** $X$ takes value $j \ge 1$ with probability $2^{-j}$. Show that $H(X) = 2$ bits, while $H_\infty(X) = 1$. *Hint: $\sum_j j2^{-j} = 2$.* *(Entropy definition)*

::: {.callout-tip collapse="true" title="Solution"}
1. **Surprisal of each outcome.** $p(j) = 2^{-j}$, so $\log_2\frac{1}{p(j)} = j$ bits. The outcome $j$ costs exactly $j$ bits, as in the code $1, 01, 001, \dots$
2. **Average it** (definition of entropy):
$$H(X) = \sum_{j \ge 1} p(j)\log_2\frac{1}{p(j)} = \sum_{j \ge 1} j\,2^{-j}.$$
3. **Sum the series** (the hint). Let $S = \sum_{j \ge 1} j\,2^{-j}$. Then $S/2 = \sum_{j \ge 1} j\,2^{-(j+1)} = \sum_{j \ge 2}(j - 1)2^{-j}$. Subtract term by term: $S - S/2 = \sum_{j \ge 1} 2^{-j} = 1$, so $S = 2$ and $H(X) = 2$ bits.
4. **Min-entropy.** The most likely value is $j = 1$, so $V(X) = \max_j 2^{-j} = \frac12$ and $H_\infty(X) = -\log_2\frac12 = 1$ bit.

The series converges fast. The partial sums up to $j = 5$, $10$ and $20$ are 1.7812, 1.9883 and 2.0000. As @prp-min-min-vs-shannon predicts, the min-entropy is below the entropy. The bound $H \le \log_2\lvert\mathcal X\rvert$ says nothing here, because $X$ has infinitely many values.

*A variable with infinitely many values can still have a small finite entropy, and its best single guess (here $X = 1$) succeeds half the time.*
:::

## Information from independent observations {#sec-sol-min-2}

> **Exercise 2.** **Information from independent observations.** If $Y_1$ and $Y_2$ are independent given $X$, show that $I(X; Y_1, Y_2) \le I(X; Y_1) + I(X; Y_2)$. Find an example where $Y_1, Y_2$ are *not* conditionally independent and the inequality fails. *Hint for the counterexample: $X = Y_1 \oplus Y_2$ for independent fair bits.* *(Chain rule)*

::: {.callout-tip collapse="true" title="Solution"}
*The inequality.* Use the symmetric form $I(X; Y_1, Y_2) = H(Y_1, Y_2) - H(Y_1, Y_2 \mid X)$.

1. **The noise part splits.** Apply @eq-min-chain-rule conditioned on $X$ throughout, as in the proof of @thm-min-fano: $H(Y_1, Y_2 \mid X) = H(Y_1 \mid X) + H(Y_2 \mid Y_1, X)$. Given $X$, $Y_2$ is independent of $Y_1$, so the posterior of $Y_2$ given $(Y_1, X)$ is its posterior given $X$, and $H(Y_2 \mid Y_1, X) = H(Y_2 \mid X)$. So $H(Y_1, Y_2 \mid X) = H(Y_1 \mid X) + H(Y_2 \mid X)$.
2. **The total part is subadditive.** By @cor-min-conditioning, $H(Y_1, Y_2) \le H(Y_1) + H(Y_2)$.
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

: The exclusive-or counterexample: every row has probability 1/4. {#tbl-sol-min-xor}

1. **One observation tells nothing.** In @tbl-sol-min-xor, fix $Y_1 = 0$. The two remaining rows have $X = 0$ and $X = 1$, equally likely. The same happens for $Y_1 = 1$. So $X$ is a fair bit whatever $Y_1$ is, $X$ and $Y_1$ are independent, and $I(X; Y_1) = 0$. By symmetry, $I(X; Y_2) = 0$.
2. **Both observations tell everything.** Given $(Y_1, Y_2)$, $X$ is determined, so $H(X \mid Y_1, Y_2) = 0$ and $I(X; Y_1, Y_2) = H(X) - 0 = 1$ bit.
3. **The inequality fails:** $1 > 0 + 0$. The script confirms $I(X; Y_1) = I(X; Y_2) = 0.0000$ and $I(X; Y_1, Y_2) = 1.0000$.
4. **Why the proof breaks.** Given $X = 0$, the pair is $(0, 0)$ or $(1, 1)$: $Y_2 = Y_1$. So $Y_1$ and $Y_2$ are fully dependent given $X$, $H(Y_1, Y_2 \mid X) = 1$, while $H(Y_1 \mid X) + H(Y_2 \mid X) = 2$. Step 1 above fails.

*Information about $X$ can sit in the combination of features and in neither feature alone. This is why ranking features one at a time by information gain can miss useful pairs.*
:::

## Min-capacity bounds capacity for the binary symmetric channel {#sec-sol-min-3}

> **Exercise 3.** **Min-capacity bounds capacity for the binary symmetric channel.** Show that $1 - h(p) \le \log_2\big(2\max(p, 1 - p)\big)$ for every $p$. *Hint: $H_\infty \le H$ applied to a $\mathrm{Bern}(p)$ variable.* *Check: at $p = 0.25$, $0.1887 \le 0.5850$.* *(Min-entropy vs Shannon)*

::: {.callout-tip collapse="true" title="Solution"}
1. **Min-entropy of a coin.** For $B \sim \mathrm{Bern}(p)$, $V(B) = \max(p, 1 - p)$, so $H_\infty(B) = -\log_2\max(p, 1 - p)$.
2. **Apply $H_\infty \le H$** (@prp-min-min-vs-shannon): $-\log_2\max(p, 1 - p) \le h(p)$.
3. **Rearrange.** Subtract both sides from 1: $1 - h(p) \le 1 + \log_2\max(p, 1 - p) = \log_2 2 + \log_2\max(p, 1 - p) = \log_2\big(2\max(p, 1 - p)\big)$.
4. **Read the two sides.** The left side is the capacity @eq-min-bsc-capacity. The right side is @eq-min-capacity for $\mathrm{BSC}(p)$, whose two column maxima are both $\max(p, 1 - p)$.
5. **Check.** At $p = 0.25$: $1 - h(0.25) = 0.1887$ and $\log_2(2 \cdot 0.75) = \log_2 1.5 = 0.5850$. At $p = 0.11$ the two are 0.5001 and 0.8319. On a grid of 1,001 values of $p$, the smallest gap between the two sides is 0.0000.

Equality holds exactly when $H_\infty(B) = H(B)$, that is when $B$ is uniform on its support: $p \in \{0, \frac12, 1\}$. That is where the grid finds the zero gap.

*For this channel the min-capacity is always at least the Shannon capacity. The two measures disagree, and the one-guess measure is the larger.*
:::

## The Z-channel {#sec-sol-min-4}

> **Exercise 4.** **The Z-channel.** Input 0 always arrives as 0; input 1 arrives as 0 with probability $\frac12$. Show that $I(X; Y) = h(\pi_1/2) - \pi_1$ for $\pi_1 = \Pr[X = 1]$, maximize over $\pi_1$, and confirm that the uniform input is *not* optimal. *Check: the optimum is $\pi_1 = 0.4$ and the capacity is $\log_2(5/4) = 0.3219$ bits; the script confirms both with Blahut–Arimoto.* *(Capacity)*

::: {.callout-tip collapse="true" title="Solution"}
*The formula.* Use $I(X; Y) = H(Y) - H(Y \mid X)$, as for the binary symmetric channel.

1. **Output distribution.** $Y = 1$ only if $X = 1$ and the bit survives: $\Pr[Y = 1] = \pi_1 \cdot \frac12 = \pi_1/2$. So $H(Y) = h(\pi_1/2)$.
2. **Noise entropy.** Given $X = 0$, $Y = 0$ for certain: $H(Y \mid X = 0) = 0$. Given $X = 1$, $Y$ is a fair bit: $H(Y \mid X = 1) = h(\frac12) = 1$. Averaging over $X$: $H(Y \mid X) = (1 - \pi_1)\cdot0 + \pi_1\cdot1 = \pi_1$.
3. **Subtract:** $I(X; Y) = h(\pi_1/2) - \pi_1$. The script compares this with the general joint-table formula on a grid of 1,001 values of $\pi_1$; they agree to $3.1\times10^{-16}$.

*The maximum.*

1. **Derivative of $h$.** Differentiating $h(t) = -t\log_2 t - (1 - t)\log_2(1 - t)$ gives $h'(t) = \log_2\frac{1 - t}{t}$ (the $\pm 1/\ln 2$ terms cancel).
2. **Derivative of $I$.** By the chain rule of calculus, $\frac{\mathrm d}{\mathrm d\pi_1}h(\pi_1/2) = \frac12\log_2\frac{1 - \pi_1/2}{\pi_1/2} = \frac12\log_2\frac{2 - \pi_1}{\pi_1}$. So
$$\frac{\mathrm dI}{\mathrm d\pi_1} = \frac12\log_2\frac{2 - \pi_1}{\pi_1} - 1.$$
3. **Set it to zero.** $\log_2\frac{2 - \pi_1}{\pi_1} = 2$, so $\frac{2 - \pi_1}{\pi_1} = 4$ and $\pi_1 = \frac25 = 0.4$.
4. **It is a maximum.** $h$ is concave and $\pi_1 \mapsto \pi_1/2$ is linear, so $h(\pi_1/2)$ is concave, and subtracting the linear $\pi_1$ keeps it concave. A stationary point of a concave function is its maximum.
5. **The capacity.** $h(0.2) = 0.2\log_2 5 + 0.8\log_2\frac54 = \log_2 5 - 0.8\cdot 2 = \log_2 5 - 1.6$. So $I = \log_2 5 - 1.6 - 0.4 = \log_2 5 - 2 = \log_2\frac54 = 0.3219$ bits.
6. **Uniform input.** At $\pi_1 = \frac12$, $I = h(\frac14) - \frac12 = 0.8113 - 0.5 = 0.3113$ bits, less than 0.3219. So the uniform input is not optimal.
7. **Numerical check.** A grid search over $\pi_1$ in steps of 0.001 peaks at $\pi_1 = 0.4000$. Blahut–Arimoto gives capacity 0.3219 at $\Pr[X = 1] = 0.4000$.

The optimum puts less weight on input 1 because sending a 1 is expensive: half of the time it is lost, and that loss costs a full bit of noise entropy $H(Y \mid X = 1) = 1$.

*When a channel is not symmetric, the capacity-achieving input is not uniform.*
:::

## Capacities of the k-ary symmetric channel {#sec-sol-min-5}

> **Exercise 5.** **Capacities of the $k$-ary symmetric channel.** Show that its Shannon capacity is $\log_2 k - h(p) - p\log_2(k - 1)$ and its min-capacity is $\log_2\big(k(1 - p)\big)$ when $1 - p \ge p/(k - 1)$. *Check: at $k = 8$ and $p = 0.3$, 1.2765 and 2.4854 bits.* *(Capacity, min-capacity)*

::: {.callout-tip collapse="true" title="Solution"}
The channel keeps the input with probability $1 - p$ and otherwise sends one of the other $k - 1$ symbols, each with probability $p/(k - 1)$. Every row of $K$ holds one entry $1 - p$ and $k - 1$ entries $p/(k - 1)$, and so does every column.

*Shannon capacity.* Follow the proof of @thm-min-bsc.

1. **Noise entropy does not depend on the input.** Row $x$ has entropy
$$H(Y \mid X = x) = -(1 - p)\log_2(1 - p) - (k - 1)\frac{p}{k - 1}\log_2\frac{p}{k - 1} = h(p) + p\log_2(k - 1),$$
using $-p\log_2\frac{p}{k - 1} = -p\log_2p + p\log_2(k - 1)$. The rows are permutations of each other, so this holds for every $x$, and $H(Y \mid X) = h(p) + p\log_2(k - 1)$ for every prior.
2. **Maximize $H(Y)$.** $Y$ takes $k$ values, so $H(Y) \le \log_2 k$, with equality iff $Y$ is uniform (@thm-min-entropy-bounds).
3. **The uniform input attains it.** With $\pi(x) = 1/k$, $\Pr[Y = y] = \frac1k\sum_x K[x, y] = \frac1k\big[(1 - p) + (k - 1)\frac{p}{k - 1}\big] = \frac1k$.
4. **So** $\mathcal C = \log_2 k - h(p) - p\log_2(k - 1)$.

At $k = 8$, $p = 0.3$: $h(0.3) = 0.8813$ and $0.3\log_2 7 = 0.8422$, so $\mathcal C = 3 - 0.8813 - 0.8422 = 1.2765$ bits. Blahut–Arimoto gives 1.2765, and the input distribution it finds is exactly uniform.

*Min-capacity.*

1. **Column maxima.** Each column holds one entry $1 - p$ and $k - 1$ entries $p/(k - 1)$. Under the condition $1 - p \ge p/(k - 1)$, its maximum is $1 - p$.
2. **Apply @eq-min-capacity:** $\mathcal{ML}(K) = \log_2\sum_{y=1}^{k}(1 - p) = \log_2\big(k(1 - p)\big)$.

At $k = 8$, $p = 0.3$: $p/(k - 1) = 0.0429 \le 0.7$, so the condition holds, and $\mathcal{ML} = \log_2 5.6 = 2.4854$ bits. The general column-maxima formula gives the same. If the condition fails, the column maximum is $p/(k - 1)$ and $\mathcal{ML} = \log_2\frac{kp}{k - 1}$.

*Under the same channel, a one-try guesser gains almost twice as many bits (2.4854) as the Shannon capacity (1.2765) suggests.*
:::

## Smith versus the average of logs {#sec-sol-min-6}

> **Exercise 6.** **Smith versus the average of logs.** Prove that $H_\infty(X \mid Y) \le \sum_y p(y)H_\infty(X \mid Y = y)$, and find a joint table where the gap is large. *Check: the walk-through gives 0.5995 vs 0.6548.* *(Jensen)*

::: {.callout-tip collapse="true" title="Solution"}
*The inequality.*

1. **Write Smith's value as a log of an average.** By @prp-min-map, $V(X \mid Y) = \sum_y p(y)\,V(X \mid Y = y)$, where $V(X \mid Y = y) = \max_x p(x \mid y)$. Let $g(y) = V(X \mid Y = y)$ and $Z = g(Y)$, a positive random variable. Then $H_\infty(X \mid Y) = -\log_2\mathbb E[Z]$.
2. **Write the other side as an average of logs.** $\sum_y p(y)H_\infty(X \mid Y = y) = \sum_y p(y)\big(-\log_2 V(X \mid Y = y)\big) = \mathbb E[-\log_2 Z]$.
3. **Apply Jensen's inequality** ([Divergences](../../book-full/part01-prerequisites/03-divergences.md)) to the convex function $-\log_2$: $-\log_2\mathbb E[Z] \le \mathbb E[-\log_2 Z]$. That is the claim. Equality holds iff $Z$ is constant, that is, when the best guess is equally likely to be right after every observation.

In the walk-through, Smith's value is 0.5995 bits and the average of logs is 0.6548, as the check says.

*A large gap.* Reuse reveal-if-zero(1) from @def-min-reveal: $X$ is uniform on $w$-bit strings, and $Y$ is $X$ itself if its lowest bit is 0 (probability $\frac12$), and $\bot$ otherwise.

1. **If $Y$ is a string,** the posterior is a point mass: $V(X \mid Y = y) = 1$ and $H_\infty(X \mid Y = y) = 0$.
2. **If $Y = \bot$,** the posterior is uniform on the $2^{w-1}$ odd strings: $V(X \mid Y = \bot) = 2^{-(w-1)}$ and $H_\infty(X \mid Y = \bot) = w - 1$.
3. **Average of logs:** $\frac12\cdot0 + \frac12(w - 1) = \frac{w - 1}{2}$.
4. **Smith:** $V(X \mid Y) = \frac12\cdot1 + \frac12\cdot2^{-(w-1)} = \frac12 + 2^{-w}$, so $H_\infty(X \mid Y) = -\log_2\big(\tfrac12 + 2^{-w}\big) = w - \log_2\big(2^{w-1} + 1\big)$, just under 1 bit.

| $w$ | Smith $H_\infty(X \mid Y)$ | Average of $H_\infty(X \mid Y = y)$ |
| --- | --- | --- |
| 6 | 0.9556 | 2.5000 |
| 64 | 1.0000 | 31.5000 |

: Smith's conditional min-entropy against the average of logs for reveal-if-zero(1) on a uniform $w$-bit secret. The $w = 6$ row is computed from the explicit channel matrix and agrees with the closed forms. {#tbl-sol-min-smith}

@tbl-sol-min-smith shows that the gap grows without bound: about $\frac{w - 1}{2} - 1$ bits. The average of logs says the secret keeps 31.5 bits of guessing difficulty. But a guesser who sees $Y$ wins more than half the time, because half the time $Y$ hands over $X$, and the maximum a posteriori (MAP) guesser takes that half for free. Smith's value, about 1 bit, reflects exactly that.

*Averaging the logs lets one very uncertain outcome hide an outcome that reveals everything. Smith's definition keeps the meaning "log of a success probability".*
:::

## Fano is tight on the symmetric channel {#sec-sol-min-7}

> **Exercise 7.** **Fano is tight on the symmetric channel.** Show that the MAP error of the $k$-ary symmetric channel with uniform input is $p$ for $p \le (k - 1)/k$, and that @eq-min-fano holds with equality. *(Fano)*

::: {.callout-tip collapse="true" title="Solution"}
*The MAP error.*

1. **Joint table.** With uniform input, $p(x, y) = \frac1k K[x, y]$: $\frac{1 - p}{k}$ on the diagonal and $\frac{p}{k(k - 1)}$ off it.
2. **When the diagonal wins.** $1 - p \ge \frac{p}{k - 1}$ iff $(k - 1)(1 - p) \ge p$ iff $k - 1 \ge kp$ iff $p \le \frac{k - 1}{k}$. Under this condition the MAP guess on observation $y$ is $x = y$.
3. **Success probability** by @eq-min-posterior-vulnerability: $V(X \mid Y) = \sum_{y=1}^{k}\frac{1 - p}{k} = 1 - p$. So the MAP error is $1 - V(X \mid Y) = p$.

*Fano holds with equality.*

1. **The output is uniform.** Each column of $K$ sums to 1 (Exercise 5), so $p(y) = \frac1k$.
2. **The posterior is a column of $K$.** $p(x \mid y) = \frac{p(x, y)}{p(y)} = K[x, y]$: $1 - p$ on $x = y$ and $\frac{p}{k - 1}$ on each of the other $k - 1$ values.
3. **Its entropy** is the row entropy computed in Exercise 5: $H(X \mid Y = y) = h(p) + p\log_2(k - 1)$ for every $y$. Averaging over $y$ gives the same value for $H(X \mid Y)$.
4. **Compare with @eq-min-fano at $P_e = p$.** The right side is $h(p) + p\log_2(k - 1)$, the same expression. So @eq-min-fano holds with equality for the MAP estimator.
5. **Why each step of the proof of @thm-min-fano is tight.** Step 2 there, $H(E \mid \hat X) \le H(E)$, is tight because the error $E$ has probability $p$ whatever the guess is, so $E$ is independent of $\hat X$. Step 3 is tight because, given an error, the true value is uniform over the $k - 1$ wrong values. Step 5 is tight because the MAP guess is $\hat X = Y$, so $\hat X$ and $Y$ carry the same information.

Since the right side of @eq-min-fano increases in $P_e$ on $[0, (k - 1)/k]$, the Fano bound obtained by inverting it equals $p$ exactly. @tbl-sol-min-fano checks this at $k = 8$.

| $p$ | MAP error | $H(X \mid Y)$ | $h(p) + p\log_2 7$ | Fano bound |
| --- | --- | --- | --- | --- |
| 0.1 | 0.1000 | 0.7497 | 0.7497 | 0.1000 |
| 0.2 | 0.2000 | 1.2834 | 1.2834 | 0.2000 |
| 0.5 | 0.5000 | 2.4037 | 2.4037 | 0.5000 |
| 0.875 | 0.8750 | 3.0000 | 3.0000 | 0.8750 |

: The $8$-ary symmetric channel with uniform input: MAP error, conditional entropy, the right side of Fano's inequality at $P_e = p$, and the inverted Fano bound. The two entropy columns agree to floating-point precision. {#tbl-sol-min-fano}

At $p = \frac78$ the output is independent of the input, $H(X \mid Y) = \log_2 8 = 3$, and the best guess is right only by chance.

*Fano's inequality cannot be improved in general: this channel attains it at every noise level.*
:::

## A sample-size lower bound {#sec-sol-min-8}

> **Exercise 8.** **A sample-size lower bound.** In Fano's method (@prp-min-fano-method), how many samples must you have before the bound $P_e \ge 1 - (n\kappa + 1)/\log_2 k$ allows $P_e \le 0.1$? Express $n$ in terms of $\kappa$ and $k$. *Check: with $k = 1024$ and $\kappa = 0.01$ bits, $n \ge 800$.* *(Fano's method)*

::: {.callout-tip collapse="true" title="Solution"}
1. **What "allows" means.** The bound rules out $P_e \le 0.1$ as long as its right side is above 0.1. It stops ruling it out when
$$1 - \frac{n\kappa + 1}{\log_2 k} \le 0.1.$$
2. **Solve for $n$.** Rearrange to $\frac{n\kappa + 1}{\log_2 k} \ge 0.9$, then $n\kappa \ge 0.9\log_2 k - 1$, so
$$n \ge \frac{0.9\log_2 k - 1}{\kappa}.$$
For a general target error $t$, replace 0.9 by $1 - t$.
3. **Check.** With $\log_2 1024 = 10$ and $\kappa = 0.01$: $n \ge (9 - 1)/0.01 = 800$. The script gives a bound of 0.1010 at $n = 799$ and exactly 0.1000 at $n = 800$.

This is a necessary condition, not a sufficient one. With fewer than 800 samples, every estimator has error above 0.1. With 800 or more, Fano no longer forbids it, but that does not mean an estimator reaches it; that depends on the problem. The sample size grows like $\log_2 k/\kappa$: more candidates, or candidates that are harder to tell apart, push it up.

*Fano's method turns "each sample carries at most $\kappa$ bits" into a minimum sample size, roughly the number of bits needed to identify the candidate divided by the bits each sample carries.*
:::
