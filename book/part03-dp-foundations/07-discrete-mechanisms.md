# Discrete and floating-point-safe mechanisms

## Why this matters

Every privacy proof so far was a statement about real numbers. The Laplace mechanism adds a real-valued $\mathrm{Lap}(b)$ draw to a real-valued $f(D)$, and the proof compares two densities on the real line. A computer has neither. It has floating-point numbers, a finite set with uneven gaps, and a source of random bits. The program that "samples $\mathrm{Lap}(b)$" outputs a distribution on that finite set, and nothing in the proof says anything about it.

The gap is not a rounding nuisance. Mironov (2012) showed that the textbook Laplace sampler, run in double precision, leaks $f(D)$ through the low-order bits of its output. Some outputs can be produced from $D$ and not from $D'$, and an observer who sees one knows which dataset was used. No $\varepsilon$ covers that.

The organizing idea of this chapter is that **differential privacy (DP) is a property of the output distribution the code actually produces**, so the code must produce exactly the distribution the proof is about. There are two ways to get there. You can keep floating point and repair the output: round it to a coarse grid and clamp it, so that every reachable output is reachable from every input. Or you can leave floating point behind: work with integer-valued queries and sample integer noise exactly, using only integer and rational arithmetic and uniform random integers. The second route gives the discrete Laplace and the discrete Gaussian mechanisms. Their privacy proofs are as clean as the continuous ones, and their samplers produce exactly the distribution in the proof. @tbl-discrete-map lists the results.

| Result | What it says | Where it is used |
| --- | --- | --- |
| The support argument (@eq-support-delta) | Outputs reachable from $D$ but not from $D'$ force $\delta \ge \Pr[M(D) \in S]$ | Auditing any implementation; counterexample tests |
| Floating-point Laplace | The textbook sampler's outputs depend on $f(D)$ through their low-order bits | Why libraries don't ship it |
| The snapping mechanism | Rounding to a grid $\Lambda \ge b$ and clamping restores a bounded privacy loss | Laplace noise when the output must be a float |
| Exact $\mathrm{Bernoulli}(e^{-\gamma})$ | A coin with probability $e^{-\gamma}$ from uniform integers, for rational $\gamma$ | The building block of both samplers below |
| The discrete Laplace mechanism | $\varepsilon = \Delta_1 f/b$ for integer-valued $f$, sampled exactly | Counts and histograms |
| The discrete Gaussian mechanism | Same privacy loss as the continuous Gaussian at integer points; $\delta(\varepsilon)$ close to it; variance below $\sigma^2$ | Census releases; noise added before secure aggregation |
| The exact discrete Gaussian sampler | Rejection from the discrete Laplace, at most about 2 proposals per draw | Any deployment of the discrete Gaussian |

: The results of this chapter, what each says, and where it is used. {#tbl-discrete-map}

**What this chapter assumes, restated.** $M$ is a mechanism: a randomized program that takes a dataset and returns an output. $D \simeq D'$ are neighbouring datasets, which differ in one person's record. A query $f$ has $\ell_1$ sensitivity $\Delta_1 f$ and $\ell_2$ sensitivity $\Delta_2 f$: the largest change in $f$, measured as a sum of absolute coordinate changes or as a Euclidean length, when one record changes. $M$ is **$(\varepsilon, \delta)$-DP** if, for every pair of neighbours and every set $S$ of outputs, $\Pr[M(D) \in S] \le e^\varepsilon\Pr[M(D') \in S] + \delta$: no event becomes more than $e^\varepsilon$ times more likely when one record changes, except for an additive slack $\delta$. With $\delta = 0$ this is **pure** $\varepsilon$-DP. The **privacy loss** of an output $y$ is the log of the ratio of its probabilities under the two neighbours,
$$L_{D,D'}(y) = \ln\frac{p_{M(D)}(y)}{p_{M(D')}(y)},$$
which measures how much evidence $y$ gives for $D$ over $D'$. For a discrete output, $p_{M(D)}$ is a probability mass function (PMF), the probability of each single output. Pure $\varepsilon$-DP is the statement $|L_{D,D'}(y)| \le \varepsilon$ for every $y$, because the probability of a set is the sum of its outputs' probabilities, and if each term grows by at most $e^\varepsilon$ so does the sum. The full treatment is in [The definition of differential privacy](01-definition.md), the Laplace and Gaussian mechanisms are in [The Laplace and Gaussian mechanisms](03-laplace-gaussian.md), and the Laplace and geometric distributions are in [Probability distributions](../part01-prerequisites/01-distributions.md).

## Toolkit: exact randomness from uniform integers

The exact samplers below need only one primitive: a uniform integer in $\{0, 1, \dots, d - 1\}$ for any $d \ge 1$. Python's `random.Random.randrange(d)` provides it exactly: it draws $\lceil\log_2 d\rceil$ random bits and redraws when the result is $d$ or more, so every integer below $d$ comes out with probability exactly $1/d$. A deployment draws the bits from the operating system with `secrets.SystemRandom()`, which has the same interface. Three facts turn this primitive into everything else: how to make a coin with a fraction as its bias, how to reshape one distribution into another by throwing some draws away, and how counting coin flips gives a geometric variable.

### Fact 1: an exact coin with rational bias {#sec-discrete-fact-coin}

To make a coin that comes up 1 with probability $n/d$, draw a uniform integer below $d$ and call it a success when it is one of the first $n$ values. For integers $0 \le n \le d$, let $U$ be uniform on $\{0, \dots, d - 1\}$. Then
$$\Pr[U < n] = \frac{n}{d},$$
because there are $n$ favourable values out of $d$ equally likely ones. So $\mathbf 1[U < n]$, the indicator that is 1 when $U < n$ and 0 otherwise, is an exact $\mathrm{Bernoulli}(n/d)$ draw, with no rounding anywhere. A coin with an *irrational* bias such as $e^{-1}$ can't be made this way, because no fraction $n/d$ equals it, which is why Algorithm 1 below exists.

### Fact 2: rejection sampling {#sec-discrete-fact-rejection}

Suppose you can sample from a PMF $q$, the **proposal**, but you want samples from a different PMF $p$. If $p(x)$ is proportional to $q(x)a(x)$ for some acceptance probability $0 \le a(x) \le 1$, you can draw from $q$ and keep each draw $x$ only with probability $a(x)$: values that $p$ favours less than $q$ get thrown away more often. Precisely, let $p(x) = q(x)a(x)/A$, where $A = \sum_x q(x)a(x)$ is the normalizer. Repeat: draw $Y \sim q$, then accept it with probability $a(Y)$; output the first accepted $Y$. Three steps show that this is exact and what it costs.

1. **Look at one round.** A round outputs $x$ when it proposes $x$ (probability $q(x)$) and accepts it (probability $a(x)$), so with probability $q(x)a(x)$. Adding over $x$, a round outputs something with probability $A$ and nothing with probability $1 - A$.
2. **Add up the rounds.** The rounds are independent and identical, so the output is $x$ when the first $j$ rounds output nothing and round $j + 1$ outputs $x$, for some $j \ge 0$. That has probability $\sum_{j \ge 0}(1 - A)^jq(x)a(x)$, a geometric series with ratio $1 - A < 1$, which sums to $q(x)a(x)/A = p(x)$.
3. **Count the cost.** Each round succeeds with probability $A$ independently, so the number of rounds is geometric with success probability $A$, and its mean is $1/A$.

The output is exact as long as each acceptance is an exact coin. What matters is that proposal times acceptance is proportional to the target. You never need to compute the normalizer $A$; it only sets how many rounds you wait.

### Fact 3: counting successes gives a geometric variable {#sec-discrete-fact-geometric}

Flip independent coins with success probability $r$ and let $V$ be the number of successes before the first failure. The event $V = v$ means exactly $v$ successes followed by one failure, and independent flips multiply, so
$$\Pr[V = v] = r^v(1 - r), \qquad v = 0, 1, 2, \dots$$
This is the geometric distribution of [Probability distributions](../part01-prerequisites/01-distributions.md), counted from 0. Its tail is $\Pr[V \ge v] = r^v$, since $V \ge v$ just says the first $v$ flips were successes.

## Floating-point Laplace breaks differential privacy {#sec-discrete-float}

### Context

The textbook way to sample $\mathrm{Lap}(b)$ is inverse cumulative distribution function (CDF) sampling: turn a uniform number into a draw with the right tail by solving for where the tail probability equals it. Draw $U$ uniform on $(0, 1)$ and a fair sign $\xi \in \{-1, +1\}$, and output $\xi\cdot b\ln(1/U)$. The magnitude $b\ln(1/U)$ is exponential with mean $b$, written $\mathrm{Exp}(b)$, because $b\ln(1/U) > t$ exactly when $U < e^{-t/b}$, so $\Pr[b\ln(1/U) > t] = \Pr[U < e^{-t/b}] = e^{-t/b}$. A fair sign turns it into $\mathrm{Lap}(b)$, whose density $\frac1{2b}e^{-|z|/b}$ is the exponential mirrored around 0. The Laplace mechanism then outputs $y = f(D) + \xi\cdot b\ln(1/U)$. Run in floating point, $U$, the logarithm and the sum are each rounded. The question is whether the rounded mechanism is still DP.

### Statement

The test is simple: if some outputs can only come from $D$, seeing one of them reveals $D$ with certainty, and no multiplicative factor $e^\varepsilon$ can account for that, so all of their probability has to be paid for by $\delta$. Write $\mathcal Y_D$ for the set of outputs $M(D)$ can produce and let $S = \mathcal Y_D \setminus \mathcal Y_{D'}$ be the outputs reachable from $D$ but not from $D'$. If $M$ is $(\varepsilon, \delta)$-DP, then for every $\varepsilon$
$$\delta \;\ge\; \Pr[M(D) \in S]. $$ {#eq-support-delta}
In particular, if $\Pr[M(D) \in S] > 0$, then $M$ is not $\varepsilon$-DP for any finite $\varepsilon$. The floating-point Laplace sampler has such outputs. In a toy model with a 12-bit significand that can be enumerated exactly, $\Pr[M(D) \in S] = 0.4791$ for $f(D) = 0$, $f(D') = 1$ and $b = 1$.

### Derivation

1. **Apply the definition to the event $S$.** $(\varepsilon, \delta)$-DP says $\Pr[M(D) \in S] \le e^\varepsilon\Pr[M(D') \in S] + \delta$ for every event. Here $\Pr[M(D') \in S] = 0$ by the choice of $S$, so the multiplicative term is $e^\varepsilon \cdot 0 = 0$ however large $e^\varepsilon$ is, and what remains is $\Pr[M(D) \in S] \le \delta$. That is @eq-support-delta.
2. **Read it as an attack.** Consider the test "say $D$ when $y \in S$". It never fires under $D'$, so its false positive rate (the chance of saying $D$ when the truth is $D'$) is 0. It fires under $D$ with probability $\Pr[M(D) \in S]$, which is its power. The hypothesis-testing form of DP in [The definition of differential privacy](01-definition.md) says that any test's power is at most $e^\varepsilon$ times its false positive rate plus $\delta$; this is the definition applied to the event "the test fires". With a false positive rate of 0 the cap is again $\delta$. And this test is never wrong when it fires.
3. **Floating-point outputs form a finite set that depends on $f(D)$.** A float with a $p$-bit significand (IEEE 754 doubles have $p = 53$) is $m\cdot2^e$ with an integer $m < 2^p$ and an integer exponent $e$. Between $2^e$ and $2^{e+1}$ the floats are spaced $2^{e - p + 1}$ apart, so they are dense near 0 and sparse far from it. The sampler computes $\mathrm{round}(f(D) + \xi\cdot\mathrm{round}(b\ln(1/U)))$. Its outputs are the rounded sums of $f(D)$ with a fixed, finite set of noise values. Shifting by $f(D') \ne f(D)$ before rounding lands on a different set of floats.
4. **Why the two sets differ.** Near 0 the noise values are fine-grained: the floats there are spaced far more closely than near 1. $M(D)$ with $f(D) = 0$ outputs those noise values as they are. $M(D')$ with $f(D') = 1$ must produce an output near 0 as $1 - (\text{noise near } 1)$, which is a multiple of the coarser spacing of the noise near 1. So most of the fine-grained outputs of $M(D)$ near 0 are impossible under $M(D')$. Away from the inputs the two grids interleave and coincide only in part. @fig-discrete-float-support (left) shows a narrow window: 3 of the 9 outputs of $M(D)$ there can't come from $M(D')$, and 4 of the 10 outputs of $M(D')$ can't come from $M(D)$.

**Concretely, in a toy model.** Doubles have too many values to enumerate, so the script uses $p$-bit floats with $p$ from 6 to 14. $U$ is an ideal uniform $p$-bit float: a real uniform rounded down to a $p$-bit float, so each float in $[2^{-40}, 1)$ has probability equal to its gap to the next one. The logarithm is computed and rounded to $p$ bits, then the sum is rounded to $p$ bits. Every output and its probability is computed exactly. @tbl-discrete-float lists the results.

| Significand bits $p$ | Outputs of $M(D)$ | Outputs of $M(D')$ | $\Pr[M(D) \in S]$ | $\Pr[M(D') \in S']$ |
| --- | --- | --- | --- | --- |
| 8 | 1,604 | 1,409 | 0.4795 | 0.2070 |
| 12 | 25,646 | 22,477 | 0.4791 | 0.2033 |
| 14 | 102,586 | 89,859 | 0.4826 | 0.2053 |

: The toy floating-point Laplace sampler with $b = 1$, $f(D) = 0$ and $f(D') = 1$. $S$ is the set of outputs of $M(D)$ that $M(D')$ can't produce, and $S'$ the reverse. {#tbl-discrete-float}

At $p = 12$, outputs with $|y| < 0.5$ carry probability 0.3933 under $M(D)$, and 0.3173 of it lands where $M(D')$ can't. Most of the leak is near $f(D)$, as step 4 predicts.

![The toy 12-bit floating-point Laplace sampler with $b = 1$. Left: every output that $f(D) = 0$ (top) and $f(D') = 1$ (bottom) can produce in a narrow window, coloured by whether the other input can produce it too. Right: the probability that $M(D)$ produces an output $M(D')$ cannot, as a function of the significand length $p$, for the textbook sampler and for the snapped one of @sec-discrete-snapping.](figures/discrete_float_support.svg){#fig-discrete-float-support}

### Interpretation

More precision does not help. The right panel of @fig-discrete-float-support is flat: from 6 to 14 bits, about 48% of the mass of $M(D)$ is on outputs $M(D')$ can't produce. Adding bits refines both grids together, so the mismatch stays. Mironov (2012) found the same effect at $p = 53$ in the samplers of several DP libraries of the time, and showed how to recover $f(D)$ from single outputs.

The leak sits in the low-order bits. It is not in where the output lands on the real line but in which float it is. Printing the output to a few decimals would hide it, and that is the intuition behind snapping in the next section.

There is a second, smaller leak: the tails are cut off. A 53-bit uniform $U = k\cdot2^{-53}$, with $k$ a positive integer, is never below $2^{-53}$, so the noise magnitude $b\ln(1/U)$ never exceeds $b\ln(2^{53}) = b\cdot53\ln2 \approx 36.74\,b$. Outputs between $f(D) + 36.74b$ and $f(D') + 36.74b$ are then reachable from only one input. Their probability is about $2^{-53}$, but by @eq-support-delta it is still a floor on $\delta$, and it rules out pure DP.

Finally, the model is a toy, but the effect is not. Real libraries differ from the toy in how they draw $U$ and round the logarithm. Those details change the numbers, not the conclusion. Jin et al. (2022) found floating-point and timing leaks in DP libraries released after Mironov's paper.

### Example: a count released with $\varepsilon = 1$

A hospital releases the number of patients with a rare diagnosis, a count with $\Delta_1 f = 1$, using the textbook sampler with $b = 1$, so that the exact mechanism would have $\varepsilon = \Delta_1 f/b = 1$. An analyst who knows that the count is either 0 or 1 (say, whether one particular patient is in the data) checks whether the released float can be produced from 1. In the 12-bit toy, the check identifies the count 0 with certainty 47.91% of the time: that is $\Pr[M(D) \in S] = 0.4791$ from @tbl-discrete-float. The proof of the [Laplace mechanism](03-laplace-gaussian.md) promised that one output could move the analyst's odds by at most a factor $e$; the code delivers certainty almost half the time.

### Check yourself

- Why doesn't the support argument need $\varepsilon$ to be small? *(Because $\Pr[M(D') \in S] = 0$, the term $e^\varepsilon\Pr[M(D') \in S]$ vanishes for every $\varepsilon$, however large.)*
- Would the problem go away if $f(D)$ and $f(D')$ were both integers and $b$ were a power of 2? *(No. The noise values $b\ln(1/U)$ are still rounded differently once added to different integers, because the spacing of floats depends on the magnitude of the sum. The toy uses $b = 1$ and integer $f$.)*

## The snapping mechanism {#sec-discrete-snapping}

### Context

If the output must be a float, the low-order bits have to go. The snapping mechanism of Mironov (2012) removes them by rounding the noisy answer to a coarse grid, and clamps both the input and the output to a bounded range so that floating-point rounding errors stay small relative to the grid. This section gives the mechanism and its guarantee in outline.

### Statement

In words: clamp the true answer into $[-B, B]$, add Laplace noise computed in floating point, round the result to the nearest multiple of a power of 2 at least as large as the noise scale, and clamp again. Let $f$ have sensitivity 1, let $\lambda$ be the noise scale, let $\Lambda$ be the smallest power of 2 with $\Lambda \ge \lambda$, and let $B$ be a clamping bound. The **snapping mechanism** outputs
$$M(D) = \mathrm{clamp}_B\Big(\big\lfloor \mathrm{clamp}_B(f(D)) \oplus \xi\otimes\lambda\otimes\mathrm{LN}(U^*)\big\rceil_\Lambda\Big),$$
where $\oplus$, $\otimes$ and $\mathrm{LN}$ are floating-point addition, multiplication and logarithm, $\xi$ is a fair sign, $U^*$ is an ideal uniform double, $\lfloor\cdot\rceil_\Lambda$ rounds to the nearest multiple of $\Lambda$, and $\mathrm{clamp}_B$ maps to $[-B, B]$. ($\mathrm{LN}(U^*)$ is negative, and the fair sign $\xi$ makes the sign of the noise irrelevant.)

Its guarantee is the exact Laplace mechanism's $\varepsilon = 1/\lambda$ plus two small terms that pay for floating-point error. The error of one floating-point operation is at most a fraction $\eta = 2^{-53}$ of the number it produces, and clamping keeps every number below $B$ in size, so the terms grow with $B\eta$. Mironov (2012, Theorem 1) proves that for $\lambda < B < 2^{46}\lambda$ the mechanism is $\varepsilon$-DP with
$$\varepsilon = \frac{1 + 12B\eta}{\lambda} + \frac{2^{-49}B}{\lambda}, \qquad \eta = 2^{-53}.$$

### Derivation (outline)

1. **Clamping the input** bounds $|f(D)| \le B$, so every sum is computed at magnitudes where the float spacing is at most about $B\eta$.
2. **Rounding to a multiple of $\Lambda \ge \lambda$** discards every bit finer than $\Lambda$. The output is one of the multiples of $\Lambda$ in $[-B, B]$, the same finite set for every input, so the support argument of @sec-discrete-float has nothing to bite on: no output is reachable from one input only.
3. **Each grid point collects an interval of noise values** of width $\Lambda$: all noise values that round to that point. In exact arithmetic, its probability under $D$ and under $D'$ is the probability that a Laplace variable lands in one of two intervals shifted by $|f(D) - f(D')| \le 1$. The Laplace density changes by at most a factor $e^{1/\lambda}$ under a shift of 1, because $e^{-|z|/\lambda}$ and $e^{-|z - 1|/\lambda}$ differ in the exponent by at most $1/\lambda$, so the ratio of the two interval probabilities is at most $e^{1/\lambda}$. In other words, rounding exact Laplace noise is post-processing of the exact Laplace mechanism, and post-processing never increases the privacy loss.
4. **Floating-point error moves each interval's endpoints** by a relative amount of order $\eta$, measured against magnitudes up to $B$. Bounding the effect of those moves on the interval probabilities gives the extra terms proportional to $B\eta/\lambda$. Mironov (2012, §5) does this bookkeeping.

### Interpretation

The price is accuracy: rounding to $\Lambda$ adds up to $\Lambda/2 \le \lambda$ of error (since $\Lambda$ is the smallest power of 2 at or above $\lambda$, it is below $2\lambda$), and clamping biases answers near $\pm B$. The extra terms in $\varepsilon$ are tiny for doubles. With $\lambda = 1$ and $B = 10$, the first term exceeds $1/\lambda$ by $12B\eta = 120\cdot2^{-53}$ and the second is $10\cdot2^{-49}$, so the excess is $120\cdot2^{-53} + 10\cdot2^{-49} \approx 3.1\times10^{-14}$. The guarantee also assumes a correctly behaved logarithm and an ideal uniform $U^*$, which a library has to supply.

### Example: the toy, snapped

Snap the toy sampler to integers ($\Lambda = 1$ for $b = 1$) and clamp to $[-10, 10]$. Both inputs now reach the same 21 outputs, the integers from $-10$ to 10, so $\Pr[M(D) \in S] = 0$ at every precision (@fig-discrete-float-support, right). The largest privacy loss over those outputs, the largest log-ratio of an output's probabilities under the two inputs, is $\max_y|L_{D,D'}(y)| = 1.0010$ at $p = 12$ and 1.0002 at $p = 14$, against $\varepsilon = 1$ for the exact mechanism. The excess shrinks with precision, as the $\eta$ terms predict.

### Check yourself

- Why must the grid be at least as coarse as $\lambda$? *(Finer grids keep bits whose pattern still depends on $f(D)$. The proof needs each grid cell to collect a whole interval of noise values, wide enough that floating-point errors at its edges move only a tiny fraction of its probability.)*
- Is snapping post-processing of the floating-point Laplace mechanism? *(It is post-processing, but post-processing of a mechanism that isn't DP proves nothing. The proof compares the snapped output with snapped exact Laplace noise and bounds the difference.)*

## Exact $\mathrm{Bernoulli}(e^{-\gamma})$ {#sec-discrete-bernoulli-exp}

### Context

Both exact samplers below need a coin that comes up 1 with probability $e^{-\gamma}$ for a rational $\gamma \ge 0$. The obvious code, `random() < exp(-gamma)`, is inexact twice over: `exp` rounds, and `random()` has only 53 bits. Algorithm 1 of Canonne, Kamath and Steinke (2020), based on a method of von Neumann (1951), makes the coin from rational coins alone.

### Statement

The idea is to build $e^{-\gamma}$ from its power series $e^{-\gamma} = 1 - \gamma + \frac{\gamma^2}{2!} - \frac{\gamma^3}{3!} + \cdots$, whose terms are probabilities you can produce with rational coins. Flip coins whose success probabilities shrink as $\gamma, \gamma/2, \gamma/3, \dots$ until one fails. The chance of surviving $k$ flips is then $\gamma \cdot \frac\gamma2 \cdots \frac\gamma k = \gamma^k/k!$, the $k$-th term of the series. The series alternates in sign, and the parity of the stopping time produces exactly that alternation.

For $0 \le \gamma \le 1$: draw $A_k \sim \mathrm{Bernoulli}(\gamma/k)$ for $k = 1, 2, \dots$, and stop at the first $A_K = 0$, so $K$ is the index of the first failure. Output 1 if $K$ is odd and 0 if $K$ is even. Then the survival probability, the output probability and the mean number of flips are
$$\Pr[K > k] = \frac{\gamma^k}{k!}, \qquad \Pr[K\ \text{odd}] = e^{-\gamma}, \qquad \mathbb E[K] = e^\gamma.$$ {#eq-bern-exp}
For $\gamma > 1$ the first coin would need a probability above 1, so split $\gamma$ into its integer part $\lfloor\gamma\rfloor$ and a remainder below 1: draw $\lfloor\gamma\rfloor$ coins with probability $e^{-1}$, stopping at the first 0, and then one coin with probability $e^{-(\gamma - \lfloor\gamma\rfloor)}$. Output 1 only if all of them are 1.

### Derivation

1. **The survival probabilities.** $K > k$ means the first $k$ coins all came up 1. They are independent with probabilities $\gamma/1, \gamma/2, \dots, \gamma/k$, so $\Pr[K > k]$ is their product, $\gamma^k/k!$. For $\gamma = n/d$, each $\gamma/j = n/(dj)$ is a ratio of integers, at most 1 since $\gamma \le 1$, so each coin is exact by Fact 1.
2. **The point probabilities.** $K = k$ means the coins survived $k - 1$ flips but not $k$, so $\Pr[K = k] = \Pr[K > k - 1] - \Pr[K > k] = \frac{\gamma^{k-1}}{(k-1)!} - \frac{\gamma^k}{k!}$.
3. **Add over odd $k$.** With $k = 1, 3, 5, \dots$ the terms are $(1 - \gamma) + (\frac{\gamma^2}{2!} - \frac{\gamma^3}{3!}) + (\frac{\gamma^4}{4!} - \frac{\gamma^5}{5!}) + \cdots$. Every power $\gamma^j$ appears exactly once: with a plus sign when $j = k - 1$ is even and a minus sign when $j = k$ is odd. So the sum is the series $\sum_{j \ge 0}(-\gamma)^j/j! = e^{-\gamma}$.
4. **The mean.** A variable $K$ on $\{1, 2, \dots\}$ can be written as a count, $K = \sum_{k \ge 0}\mathbf 1[K > k]$, since exactly $K$ of the indicators $\mathbf 1[K > 0], \mathbf 1[K > 1], \dots$ are 1. Taking expectations, $\mathbb E[K] = \sum_{k \ge 0}\Pr[K > k] = \sum_k\gamma^k/k! = e^\gamma$. Each round uses one uniform integer, so a coin costs $e^\gamma \le e$ uniform draws on average.
5. **Large $\gamma$.** $e^{-\gamma} = (e^{-1})^{\lfloor\gamma\rfloor}\,e^{-(\gamma - \lfloor\gamma\rfloor)}$, and a product of probabilities is the probability that independent coins all come up 1. Each coin has an exponent of at most 1, so steps 1–3 cover it. Stopping at the first 0 doesn't change the output, which is 0 as soon as one coin is 0.

**Concretely, for $\gamma = 0.7$** (@tbl-discrete-bern-k). The odd rows add up to $0.3000 + 0.1878 + 0.0086 + \cdots = 0.4966 = e^{-0.7}$.

| $K$ | Coins seen | $\Pr[K = k]$ | Output |
| --- | --- | --- | --- |
| 1 | $A_1 = 0$ | $1 - 0.7 = 0.3000$ | 1 |
| 2 | $A_1 = 1$, $A_2 = 0$ | $0.7 - 0.7^2/2 = 0.4550$ | 0 |
| 3 | $A_1 = A_2 = 1$, $A_3 = 0$ | $0.245 - 0.0572 = 0.1878$ | 1 |
| 4 | three 1s, then 0 | $0.0472$ | 0 |
| 5 | four 1s, then 0 | $0.0086$ | 1 |

: Algorithm 1 for $\gamma = 0.7$, outcome by outcome. The output is the parity of the index of the first 0. {#tbl-discrete-bern-k}

In row 3, $0.245 = 0.7^2/2!$ is the chance of surviving two flips and $0.0572 = 0.7^3/3!$ the chance of surviving three; the difference is the chance of stopping at exactly the third.

![Left: the distribution of $K$ for $\gamma = 0.7$, exact and from 5,000 runs, with odd $K$ (output 1) and even $K$ (output 0) coloured. Right: the frequency of 1s from Algorithm 1 against $e^{-\gamma}$, with the expected number of uniform draws per coin on the right axis.](figures/discrete_bernoulli_exp.svg){#fig-discrete-bernoulli-exp}

### Interpretation

The coin is exact: no step rounds, and the only randomness is uniform integers. The cost is small and bounded: at most $e$ draws for $\gamma \le 1$, and below $\frac{e}{1 - e^{-1}} + 1 \approx 5.3$ for any $\gamma$ ([Exercise 1](solutions/07-discrete-mechanisms.md#sec-sol-discrete-1)). The cost curve in @fig-discrete-bernoulli-exp jumps at each integer. At $\gamma = 1$ the algorithm runs one full $e^{-1}$ coin and then a coin for the fractional part 0, which costs one extra draw when the first coin comes up 1 (probability $e^{-1}$), so the cost goes from $e = 2.7183$ just below 1 to $e + e^{-1} = 3.0862$ at 1. Splitting $\gamma$ differently would smooth this out; it doesn't matter for exactness.

### Example: one coin per proposal

Algorithm 3 below accepts a proposal $y$ with probability $e^{-\gamma}$ for a rational $\gamma = (|y| - \sigma^2/t)^2/(2\sigma^2)$, where $\sigma^2$ is the target variance and $t$ an integer scale. With $\sigma^2 = 4$, $t = 3$ and $y = 4$, that is $\gamma = (4 - 4/3)^2/8 = (8/3)^2/8 = 8/9$, so the coin is 1 with probability $e^{-8/9} = 0.4111$ and costs $e^{8/9} \approx 2.43$ draws on average. Everything stays a fraction: $\gamma$ is computed with `fractions.Fraction`, and each $\gamma/k$ is a ratio of integers.

### Check yourself

- Why does the algorithm use $\gamma/k$ rather than a fixed probability $\gamma$ at every step? *(The decreasing probabilities make $\Pr[K > k] = \gamma^k/k!$, the terms of the exponential series. A fixed $\gamma$ would give $\gamma^k$, a geometric series, and the alternating sum would be $1/(1 + \gamma)$.)*
- What does the algorithm output for $\gamma = 0$? *(The first coin is $\mathrm{Bernoulli}(0)$, so $K = 1$, which is odd, and the output is 1 = $e^0$.)*

## The discrete Laplace mechanism {#sec-discrete-laplace}

### Context

Most queries in DP are counts or histograms: integer-valued, with sensitivity 1. For them the natural noise is integer-valued too. Then the output is an integer, there are no low-order bits to leak, and the noise can be sampled exactly. The discrete Laplace distribution is the integer analogue of the Laplace, and adding it is the **geometric mechanism** of Ghosh, Roughgarden and Sundararajan (2009).

### Statement

Take the Laplace density $e^{-|z|/b}$, keep only its values at the integers, and divide by their sum so they add up to 1. Each step away from 0 multiplies the probability by the same factor $r = e^{-1/b}$, just as the continuous density drops by $e^{-1/b}$ per unit. The **discrete Laplace** distribution with scale $b > 0$, written $\mathrm{Lap}_{\mathbb Z}(b)$, has PMF
$$\Pr[Z = z] = \frac{1 - r}{1 + r}\,r^{|z|}, \qquad z \in \mathbb Z, \qquad r = e^{-1/b}.$$ {#eq-dlap-pmf}
Its variance is $2r/(1 - r)^2$, slightly below the $2b^2$ of $\mathrm{Lap}(b)$. Because shifting the input by one unit changes each output's probability by at most the factor $r^{-1} = e^{1/b}$, as for the continuous Laplace, the privacy guarantee is the same. For an integer-valued $f$ with sensitivity $\Delta_1 f$, the mechanism $M(D) = f(D) + Z$ is $\varepsilon$-DP with
$$\varepsilon = \frac{\Delta_1 f}{b}.$$ {#eq-dlap-eps}
For rational $b = t/s$, Algorithm 2 of Canonne, Kamath and Steinke (2020) samples $\mathrm{Lap}_{\mathbb Z}(b)$ exactly.

### Derivation

**The PMF and variance.** Both come from sums of geometric series.

1. **Normalize.** The weights $r^{|z|}$ are 1 at $z = 0$ and appear twice, at $\pm z$, for each $z \ge 1$. So $\sum_{z \in \mathbb Z}r^{|z|} = 1 + 2\sum_{z \ge 1}r^z = 1 + \frac{2r}{1 - r} = \frac{1 + r}{1 - r}$, and dividing by it gives the constant in @eq-dlap-pmf.
2. **Variance.** The PMF gives $z$ and $-z$ the same probability, so their contributions $z\Pr[Z = z]$ and $-z\Pr[Z = -z]$ cancel and the mean is 0. The variance is then $\mathbb E Z^2$. Differentiating $\sum_{z \ge 0}r^z = \frac1{1 - r}$ and multiplying by $r$ gives $\sum_{z \ge 1}zr^z = \frac{r}{(1 - r)^2}$; doing it once more gives $\sum_{z \ge 1}z^2r^z = \frac{r(1 + r)}{(1 - r)^3}$. Counting $\pm z$ twice, $\mathbb E Z^2 = \frac{1 - r}{1 + r}\cdot2\cdot\frac{r(1 + r)}{(1 - r)^3} = \frac{2r}{(1 - r)^2}$.

**Privacy.** The plan is to compute the privacy loss of each output and bound it by $\varepsilon$, which is pure $\varepsilon$-DP.

1. **The privacy loss.** For neighbours with $f(D) = a$ and $f(D') = a'$, an output $y$ requires noise $y - a$ under $D$ and $y - a'$ under $D'$. The normalizing constant is the same for both, so it cancels in the ratio:
$$L_{D,D'}(y) = \ln\frac{r^{|y - a|}}{r^{|y - a'|}} = \frac{|y - a'| - |y - a|}{b},$$
using $\ln r = -1/b$.
2. **Bound it.** The triangle inequality says that two numbers' distances to $y$ differ by at most their distance to each other: $|y - a'| - |y - a| \le |a - a'| \le \Delta_1 f$. So $|L_{D,D'}(y)| \le \Delta_1 f/b$ for every integer $y$, which is pure DP with @eq-dlap-eps. The bound is reached for every $y \ge \max(a, a')$, so it is tight.
3. **Vectors.** For $f \in \mathbb Z^d$ with independent noise per coordinate, the probability of an output vector is a product over coordinates, so its log, the privacy loss, is a sum of per-coordinate losses. Each is at most $|a_i - a'_i|/b$, and $\sum_i|a_i - a'_i| \le \Delta_1 f$ gives the same $\varepsilon$ with the $\ell_1$ sensitivity.

Step 1 is where integers matter. The constant cancels because $\sum_z r^{|z - a|}$ is the same for every *integer* $a$: shifting by an integer only relabels the terms. Shifting by a non-integer would move the noise off the lattice, and the mechanism would have different supports for different inputs: the floating-point failure again.

**The exact sampler (Algorithm 2).** The plan is to build a one-sided geometric variable with ratio $e^{-1/b}$ out of exact pieces, then attach a random sign. A geometric with ratio $e^{-1/t}$ is assembled from a remainder $U$ below $t$ and a quotient $V$, the way an integer $x$ is $u + tv$; dividing by $s$ then turns ratio $e^{-1/t}$ into $e^{-s/t} = e^{-1/b}$. Input integers $s, t \ge 1$, so $b = t/s$. Repeat until a value is returned:

1. Draw $U$ uniform on $\{0, \dots, t - 1\}$ and accept it with probability $e^{-U/t}$ (Algorithm 1); on rejection, start over.
2. Count the successes $V$ of independent $e^{-1}$ coins before the first failure (Algorithm 1 with $\gamma = 1$).
3. Set $X = U + tV$ and $Y = \lfloor X/s\rfloor$.
4. Draw a fair sign bit $B$. If $B = 1$ and $Y = 0$, start over. Otherwise return $(1 - 2B)Y$.

Five steps show that the output is exactly $\mathrm{Lap}_{\mathbb Z}(b)$, following the algorithm's own order.

1. **$U$ given acceptance** has $\Pr[U = u] \propto e^{-u/t}$ on $\{0, \dots, t - 1\}$, by Fact 2 with a uniform proposal and acceptance $e^{-u/t}$.
2. **$V$ is geometric** with $\Pr[V = v] = (1 - e^{-1})e^{-v}$, by Fact 3 with $r = e^{-1}$.
3. **$X = U + tV$ is geometric with ratio $e^{-1/t}$.** Each $x \ge 0$ is $u + tv$ for exactly one pair ($u = x \bmod t$, $v = \lfloor x/t\rfloor$), and the independent weights multiply to $e^{-u/t}e^{-v} = e^{-(u + tv)/t} = e^{-x/t}$. So $\Pr[X = x] \propto e^{-x/t}$.
4. **$Y = \lfloor X/s\rfloor$ is geometric with ratio $e^{-s/t} = e^{-1/b}$.** $Y = y$ collects $x = sy, \dots, sy + s - 1$, with total weight $e^{-sy/t}(1 + e^{-1/t} + \dots + e^{-(s-1)/t})$. The bracket does not depend on $y$, so the weight is $\propto e^{-sy/t} = r^y$.
5. **The sign.** $\pm Y$ with a fair sign gives $\pm y$ with probability $\frac12\Pr[Y = y]$ each, but 0 would get both halves, since $+0$ and $-0$ are the same output. Rejecting $(B = 1, Y = 0)$ leaves $\frac12\Pr[Y = y]$ for every $z = \pm y$, including 0. After renormalizing (Fact 2), $\Pr[Z = z] \propto r^{|z|}$.

**Concretely, for $t = 3$ and $s = 2$** ($b = 1.5$). @tbl-discrete-dlap-walk follows steps 3 and 4.

| $x$ | $u = x \bmod 3$ | $v = \lfloor x/3\rfloor$ | $\Pr[X = x] = (1 - e^{-1/3})e^{-x/3}$ | $y = \lfloor x/2\rfloor$ |
| --- | --- | --- | --- | --- |
| 0 | 0 | 0 | 0.2835 | 0 |
| 1 | 1 | 0 | 0.2031 | 0 |
| 2 | 2 | 0 | 0.1455 | 1 |
| 3 | 0 | 1 | 0.1043 | 1 |
| 4 | 1 | 1 | 0.0747 | 2 |
| 5 | 2 | 1 | 0.0535 | 2 |

: Algorithm 2 with $t = 3$, $s = 2$. Each $x$ is one pair $(u, v)$; pairs of consecutive $x$ merge into one $y$. {#tbl-discrete-dlap-walk}

$\Pr[Y = 0] = 0.2835 + 0.2031 = 0.4866$ and $\Pr[Y = 1] = 0.1455 + 0.1043 = 0.2498$. Their ratio is $0.5134 = e^{-2/3}$, a geometric with ratio $e^{-s/t}$, as step 4 says.

![Left: the PMF of $\mathrm{Lap}_{\mathbb Z}(1.5)$ on a log scale, exact and from 10,000 draws of Algorithm 2, with the $\mathrm{Lap}(1.5)$ density for comparison. Right: the privacy loss between counts 0 and 1, a step function bounded by $\varepsilon = s/t = 2/3$.](figures/discrete_laplace.svg){#fig-discrete-laplace}

### Interpretation

The discrete Laplace gives the same guarantee with less noise. It has the same $\varepsilon$ as the continuous one at the same scale and a slightly smaller variance: 1.8413 against 2 at $b = 1$. [Exercise 3](solutions/07-discrete-mechanisms.md#sec-sol-discrete-3) shows the gap tends to $1/6$ as $b$ grows. Rounding $f(D) + \mathrm{Lap}(b)$ to the nearest integer would also be $\varepsilon$-DP in exact arithmetic, as post-processing, but it is not the same distribution, and it needs a floating-point sampler.

It is also the best possible mechanism for one count. Ghosh, Roughgarden and Sundararajan (2009) show that for a single count, the geometric mechanism is universally utility-maximizing: every user, whatever their loss function and prior, can post-process its output to do as well as with any other $\varepsilon$-DP mechanism.

The cost follows from Fact 2. A pass of Algorithm 2 survives step 1 with probability $A_U = \frac1t\sum_{u < t}e^{-u/t}$, the average acceptance over the $t$ uniform values. It then survives step 4 unless $B = 1$ and $Y = 0$; since $Y$ is geometric with ratio $e^{-s/t}$, $\Pr[Y = 0] = 1 - e^{-s/t}$, so that happens with probability $\frac12(1 - e^{-s/t})$. The mean number of passes is one over the product of the two survival probabilities, $1/\big(A_U(1 - \frac12(1 - e^{-s/t}))\big)$. That is 1.462 passes at $b = 1$, and about 10 to 12 uniform draws per sample over the scales in @tbl-discrete-code.

### Example: a count with $\varepsilon = 1$

Release a count with sensitivity 1 at $\varepsilon = 1$, so $b = \Delta_1 f/\varepsilon = 1$ and $r = e^{-1} = 0.3679$. The exact count is released when the noise is 0, which happens with probability $\Pr[Z = 0] = \frac{1 - r}{1 + r} = \frac{0.6321}{1.3679} = 0.4621$. The noise has variance $\frac{2r}{(1 - r)^2} = 1.8413$, so its standard deviation is $\sqrt{1.8413} = 1.357$. The output is an integer, sampled exactly, so the analysis in this section is the analysis of the code.

### Check yourself

- Why is the bound in step 2 of the privacy proof reached for $y \ge \max(a, a')$? *(There both $|y - a|$ and $|y - a'|$ are $y$ minus the input, so their difference is exactly $|a - a'|$.)*
- In Algorithm 2, why can't you skip the rejection of $(B = 1, Y = 0)$? *(Then $\Pr[Z = 0] = \Pr[Y = 0]$ while $\Pr[Z = \pm y] = \frac12\Pr[Y = y]$: zero would get double weight, and the PMF would not be @eq-dlap-pmf.)*

## The discrete Gaussian mechanism {#sec-discrete-gaussian}

### Context

The Gaussian mechanism is the workhorse of private machine learning, because its noise grows with the $\ell_2$ sensitivity rather than the $\ell_1$ sensitivity, and because sums of Gaussians stay Gaussian. Canonne, Kamath and Steinke (2020) define its integer analogue and show that it keeps the privacy of the continuous Gaussian. The US Census Bureau used it for the 2020 Census redistricting data (Abowd et al., 2022), and it is the noise of choice when clients add noise before secure aggregation, which works with integers modulo a fixed number (Kairouz, Liu & Steinke, 2021).

### Statement

As with the discrete Laplace, take the Gaussian bell curve, keep its values at the integers and normalize. The **discrete Gaussian** with parameter $\sigma > 0$ and integer location $\mu$, written $\mathcal N_{\mathbb Z}(\mu, \sigma^2)$, has PMF
$$\Pr[X = x] = \frac{e^{-(x - \mu)^2/(2\sigma^2)}}{\sum_{y \in \mathbb Z}e^{-y^2/(2\sigma^2)}}, \qquad x \in \mathbb Z.$$
The parameter $\sigma^2$ is the variance of the continuous bell curve, not quite the variance of $X$, as the last claim below shows. For an integer-valued $f$, consider $M(D) = f(D) + X$ with $X \sim \mathcal N_{\mathbb Z}(0, \sigma^2)$, and neighbours with $f(D) - f(D') = \Delta$. Four claims describe how it compares with the continuous Gaussian mechanism.

First, **the privacy loss is the continuous Gaussian's**, evaluated at integers. The reason is that the normalizer is the same for every integer location, so the log-ratio of two PMFs is the log-ratio of the two bell curves:
$$L_{D,D'}(y) = \frac{(y - f(D'))^2 - (y - f(D))^2}{2\sigma^2}, \qquad y \in \mathbb Z.$$

Second, the smallest $\delta$ at each $\varepsilon$ has the same formula as for the continuous Gaussian. That smallest $\delta$ is the probability mass by which $M(D)$ exceeds $e^\varepsilon$ times $M(D')$, summed over outputs, and since the privacy loss is linear in $y$ the outputs that contribute form a tail. The **exact $\delta(\varepsilon)$** (Canonne, Kamath & Steinke 2020, Theorem 7) is
$$\delta(\varepsilon) = \Pr\Big[X > \frac{\varepsilon\sigma^2}{\Delta} - \frac\Delta2\Big] - e^\varepsilon\Pr\Big[X > \frac{\varepsilon\sigma^2}{\Delta} + \frac\Delta2\Big], \qquad X \sim \mathcal N_{\mathbb Z}(0, \sigma^2),$$ {#eq-dgauss-delta}
the same expression as the continuous Gaussian's analytic curve with $X \sim \mathcal N(0, \sigma^2)$, only with the tails of the discrete distribution in place of the continuous ones.

Third, the discrete Gaussian is never worse than the continuous one in Rényi divergence. The **Rényi divergence** of order $\alpha > 1$, $D_\alpha(P\|Q) = \frac{1}{\alpha - 1}\ln\sum_xp(x)^\alpha q(x)^{1 - \alpha}$, is a way of averaging the privacy loss that weights large losses more as $\alpha$ grows; for continuous Gaussians with means $\mu$ and $\nu$ and variance $\sigma^2$ it equals $\alpha(\mu - \nu)^2/(2\sigma^2)$. For every $\alpha > 1$,
$$D_\alpha\big(\mathcal N_{\mathbb Z}(\mu, \sigma^2)\,\|\,\mathcal N_{\mathbb Z}(\nu, \sigma^2)\big) \le \frac{\alpha(\mu - \nu)^2}{2\sigma^2},$$ {#eq-dgauss-renyi}
with equality for integer $\alpha$.

Fourth, **the variance is below $\sigma^2$**. Restricting the bell curve to integers pulls a little mass towards 0, by an amount that the Poisson summation lemma in the derivation computes exactly:
$$\operatorname{Var}X = \sigma^2 - \frac{\sum_{k \ne 0}4\pi^2\sigma^4k^2e^{-2\pi^2\sigma^2k^2}}{1 + \sum_{k \ne 0}e^{-2\pi^2\sigma^2k^2}} < \sigma^2.$$ {#eq-dgauss-var}

### Derivation

**Privacy loss and $\delta(\varepsilon)$.** Take $f(D) = 0$ and $f(D') = \Delta$; the general case shifts everything by an integer, which changes nothing.

1. **The normalizer cancels.** Under $D$ the output $y$ has probability $e^{-y^2/(2\sigma^2)}/Z$, and under $D'$ it has $e^{-(y - \Delta)^2/(2\sigma^2)}/Z$, with the same $Z = \sum_ye^{-y^2/(2\sigma^2)}$, because shifting the sum by the integer $\Delta$ only relabels its terms. So the ratio is the ratio of the two exponentials, and
$$L_{D,D'}(y) = \frac{(y - \Delta)^2 - y^2}{2\sigma^2} = \frac{\Delta^2 - 2\Delta y}{2\sigma^2},$$
the formula of the continuous Gaussian, now only at integers $y$.
2. **$\delta$ as an excess.** $(\varepsilon, \delta)$-DP asks $\Pr[M(D) \in S] - e^\varepsilon\Pr[M(D') \in S] \le \delta$ for every set $S$. The left side is a sum of $p(y) - e^\varepsilon q(y)$ over $y \in S$, where $p$ is the PMF under $D$ and $q$ under $D'$, so it is largest when $S$ holds exactly the outputs with a positive term. The smallest valid $\delta$ is therefore $\delta(\varepsilon) = \sum_y\big(p(y) - e^\varepsilon q(y)\big)_+$, where $(\cdot)_+$ keeps positive values and replaces negative ones by 0. This is the hockey-stick divergence of [Divergences](../part01-prerequisites/03-divergences.md). The summand is positive exactly when $p(y)/q(y) > e^\varepsilon$, that is when $L_{D,D'}(y) > \varepsilon$, which by step 1 means $\Delta^2 - 2\Delta y > 2\sigma^2\varepsilon$, or $y < \frac\Delta2 - \frac{\varepsilon\sigma^2}{\Delta}$.
3. **The first sum.** $\sum_{y < \Delta/2 - \varepsilon\sigma^2/\Delta}p(y) = \Pr[X < \frac\Delta2 - \frac{\varepsilon\sigma^2}{\Delta}]$. The PMF of $X$ gives $x$ and $-x$ the same probability, so $X$ and $-X$ have the same distribution, and $\Pr[X < c] = \Pr[-X < c] = \Pr[X > -c]$. With $c = \frac\Delta2 - \frac{\varepsilon\sigma^2}{\Delta}$ this is $\Pr[X > \frac{\varepsilon\sigma^2}{\Delta} - \frac\Delta2]$.
4. **The second sum.** Under $D'$ the output is $X + \Delta$, so $q(y) = p(y - \Delta)$ and $\sum_{y < \Delta/2 - \varepsilon\sigma^2/\Delta}q(y) = \Pr[X + \Delta < \frac\Delta2 - \frac{\varepsilon\sigma^2}{\Delta}] = \Pr[X < -\frac\Delta2 - \frac{\varepsilon\sigma^2}{\Delta}] = \Pr[X > \frac{\varepsilon\sigma^2}{\Delta} + \frac\Delta2]$, by the same symmetry. Subtracting $e^\varepsilon$ times step 4 from step 3 gives @eq-dgauss-delta.

**A lemma from Poisson summation.** The remaining two claims rest on one question: if you sum a Gaussian bump over the integers, how does the sum depend on where the bump is centred? The answer is that it is largest when the centre is an integer. The tool is the Poisson summation formula, which turns a sum of a function over the integers into a sum of its Fourier transform over the integers: $\sum_{x \in \mathbb Z}g(x) = \sum_{k \in \mathbb Z}\hat g(k)$ for a smooth, fast-decaying $g$ with Fourier transform $\hat g(k) = \int g(x)e^{-2\pi ikx}dx$ (Stein & Shakarchi 2003, ch. 5). The Fourier transform of a Gaussian is again a Gaussian, and shifting $g$ by $c$ multiplies the transform by the phase $e^{-2\pi ikc}$: for $g(x) = e^{-(x - c)^2/(2\sigma^2)}$, $\hat g(k) = \sqrt{2\pi}\sigma e^{-2\pi^2\sigma^2k^2}e^{-2\pi ikc}$. Pairing $k$ with $-k$ adds $e^{-2\pi ikc} + e^{2\pi ikc} = 2\cos(2\pi kc)$, which gives
$$\sum_{x \in \mathbb Z}e^{-(x - c)^2/(2\sigma^2)} = \sqrt{2\pi}\,\sigma\sum_{k \in \mathbb Z}e^{-2\pi^2\sigma^2k^2}\cos(2\pi kc).$$ {#eq-poisson-theta}
Every coefficient is positive and $\cos \le 1$, with $\cos(2\pi kc) = 1$ for all $k$ exactly when $c$ is an integer. So the left side is largest at integer $c$: **shifting the lattice off the integers can only lower the sum.** At $\sigma = 0.5$ the script finds 1.271342 at $c = 0$ and 1.235287 at $c = 0.5$. At $\sigma = 1$ both are 2.506628, which is $\sqrt{2\pi}$ to six decimals: the $k \ne 0$ terms are below $10^{-8}$.

**Rényi divergences.** The idea is that the sum defining $D_\alpha$ is, after completing a square, a Gaussian bump summed over the integers with centre $c$, and the lemma says that sum is at most its value at an integer centre. Write $p_\mu(x) = e^{-(x - \mu)^2/(2\sigma^2)}/Z$, with $Z$ the same for every integer $\mu$.

1. **Write out the summand.** $D_\alpha = \frac{1}{\alpha - 1}\ln\sum_xp_\mu(x)^\alpha p_\nu(x)^{1 - \alpha}$. The powers of $Z$ combine to $Z^{-\alpha}Z^{\alpha - 1} = 1/Z$, and the exponent is $-\frac{1}{2\sigma^2}\big(\alpha(x - \mu)^2 + (1 - \alpha)(x - \nu)^2\big)$.
2. **Complete the square.** $\alpha(x - \mu)^2 + (1 - \alpha)(x - \nu)^2 = (x - c)^2 - \alpha(\alpha - 1)(\mu - \nu)^2$ with $c = \alpha\mu + (1 - \alpha)\nu$. (Expand: the $x^2$ and $x$ terms match by the choice of $c$, and the constant is what is left.)
3. **So** $\sum_xp_\mu^\alpha p_\nu^{1 - \alpha} = e^{\alpha(\alpha - 1)(\mu - \nu)^2/(2\sigma^2)}\cdot\frac1Z\sum_xe^{-(x - c)^2/(2\sigma^2)}$.
4. **Apply the lemma.** $Z$ is the same sum centred at 0, so the last factor is the sum centred at $c$ divided by the sum centred at an integer. By the lemma it is at most 1, with equality when $c$ is an integer, which it is for integer $\alpha$ because $\mu$ and $\nu$ are integers. Taking $\frac{1}{\alpha - 1}\ln$ gives @eq-dgauss-renyi. The continuous Gaussian has exactly $\alpha(\mu - \nu)^2/(2\sigma^2)$ ([Divergences](../part01-prerequisites/03-divergences.md)), so the discrete Gaussian is never worse in Rényi divergence. In the language of [Privacy accounting](../part04-privacy-accounting/00-introduction.md), where zero-concentrated DP is the guarantee that every Rényi divergence of order $\alpha$ is at most a constant times $\alpha$, this is the statement that the discrete Gaussian satisfies the same zero-concentrated DP as the continuous one (Canonne, Kamath & Steinke 2020, Theorem 4).

**Variance.** The idea is that the normalizer, seen as a function of $1/(2\sigma^2)$, encodes the variance in its derivative, and the lemma gives the normalizer in closed form. Let $\tau = 1/(2\sigma^2)$ and $Z(\tau) = \sum_xe^{-\tau x^2}$.

1. **Variance from the normalizer.** The mean is 0 because $x$ and $-x$ have the same probability, so $\operatorname{Var}X = \mathbb E X^2 = \sum_xx^2e^{-\tau x^2}/Z$. Differentiating $e^{-\tau x^2}$ in $\tau$ brings down a factor $-x^2$, so $\frac{d}{d\tau}Z = -\sum_xx^2e^{-\tau x^2}$, and $\mathbb E X^2 = -\frac{Z'(\tau)}{Z(\tau)} = -\frac{d}{d\tau}\ln Z(\tau)$.
2. **Poisson summation at $c = 0$.** @eq-poisson-theta, with $\sigma = 1/\sqrt{2\tau}$, gives $Z(\tau) = \sqrt{\pi/\tau}\,(1 + \theta(\tau))$ with $\theta(\tau) = \sum_{k \ne 0}e^{-\pi^2k^2/\tau}$.
3. **Differentiate.** The log splits into $\frac12\ln\pi - \frac12\ln\tau + \ln(1 + \theta(\tau))$, so $-\frac{d}{d\tau}\ln Z = \frac{1}{2\tau} - \frac{\theta'(\tau)}{1 + \theta(\tau)}$, where $\theta'(\tau) = \sum_{k \ne 0}\frac{\pi^2k^2}{\tau^2}e^{-\pi^2k^2/\tau} > 0$.
4. **Substitute $\tau = 1/(2\sigma^2)$:** $\frac{1}{2\tau} = \sigma^2$, $\frac{\pi^2k^2}{\tau} = 2\pi^2\sigma^2k^2$ and $\frac{\pi^2k^2}{\tau^2} = 4\pi^2\sigma^4k^2$. This is @eq-dgauss-var. The correction is positive, so $\operatorname{Var}X < \sigma^2$, and its leading term, from $k = \pm1$ with the denominator close to 1, is $8\pi^2\sigma^4e^{-2\pi^2\sigma^2}$.

### Interpretation

The privacy proof is the continuous one. The privacy loss is the same linear function, so every statement about it carries over, restricted to integer points. The only new ingredient is that integer shifts don't change the normalizer. That is why the query must be integer-valued. To use the discrete Gaussian for a real-valued query, scale it by $1/\gamma$ for a small step $\gamma$ and round it to an integer first. Randomized rounding, which rounds up with probability equal to the fractional part, keeps it unbiased. Then account for the sensitivity of the rounded query, which can be slightly larger than the scaled original.

$\delta(\varepsilon)$ is close to the continuous curve, not equal to it. The privacy loss takes only the values $(\Delta^2 - 2\Delta y)/(2\sigma^2)$ for integer $y$; moving $y$ by one changes the loss by $2\Delta/(2\sigma^2) = \Delta/\sigma^2$, so these values form a lattice with spacing $\Delta/\sigma^2$ ($1/\sigma^2$ for $\Delta = 1$). As $\varepsilon$ crosses one of these values, a whole output enters or leaves the sum for $\delta$, so $\delta(\varepsilon)$ has a kink at each lattice value and oscillates around the continuous curve (@fig-discrete-gaussian-delta). At $\varepsilon = 1$ and $\Delta = 1$ the script finds $\delta = 0.6032$ against 0.5099 at $\sigma = 0.5$, 0.1414 against 0.1269 at $\sigma = 1$, and 0.0072 against 0.0068 at $\sigma = 2$. At other $\varepsilon$, such as just past a kink, the discrete curve is below. Neither dominates, and the gap closes as $\sigma$ grows. In Rényi divergence, by contrast, the discrete Gaussian is never worse: at $\sigma = 0.5$, $D_{2.5} = 4.9808$ against 5.

The variance deficit is invisible in practice. Its relative size $1 - \operatorname{Var}X/\sigma^2$ has leading term $8\pi^2\sigma^2e^{-2\pi^2\sigma^2}$, the leading term of @eq-dgauss-var divided by $\sigma^2$. At $\sigma = 1$ that is $8\pi^2e^{-2\pi^2} = 78.96 \times 2.67\times10^{-9}$, so $1 - \operatorname{Var}X/\sigma^2 = 2.11\times10^{-7}$; at $\sigma = 2$ it is at the level of rounding error. Only below $\sigma \approx 0.6$ does the discrete Gaussian concentrate visibly more than $\sigma^2$ suggests: at $\sigma = 0.5$, $\operatorname{Var}X = 0.8601\,\sigma^2$, and at $\sigma = 0.25$ almost all the mass is at 0 (@fig-discrete-gaussian-variance).

Sums are not discrete Gaussians. Unlike the continuous case, the sum of two independent discrete Gaussians is not exactly a discrete Gaussian. For $\sigma \ge 1$ it is very close ([Exercise 6](solutions/07-discrete-mechanisms.md#sec-sol-discrete-6)), which is what distributed noise addition, where each client adds its own share of the noise, relies on.

![Discrete against continuous Gaussian for sensitivity 1: $\delta(\varepsilon)$ from @eq-dgauss-delta (solid) and the analytic curve of the continuous Gaussian (dashed), for $\sigma = 0.5$, 1 and 2. The discrete curve has a kink at each value of the lattice-valued privacy loss.](figures/discrete_gaussian_delta.svg){#fig-discrete-gaussian-delta}

![Left: $\operatorname{Var}X/\sigma^2$ for $X \sim \mathcal N_{\mathbb Z}(0, \sigma^2)$, by direct summation and from @eq-dgauss-var, against 1 for the continuous Gaussian. Right: the relative deficit $1 - \operatorname{Var}X/\sigma^2$ on a log scale, with its leading term $8\pi^2\sigma^2e^{-2\pi^2\sigma^2}$.](figures/discrete_gaussian_variance.svg){#fig-discrete-gaussian-variance}

### Example: the Rényi divergence, computed

The script sums the PMFs directly to get $D_\alpha(\mathcal N_{\mathbb Z}(0, \sigma^2)\|\mathcal N_{\mathbb Z}(1, \sigma^2))$ (@tbl-discrete-renyi). Integer orders match $\alpha/(2\sigma^2)$ exactly, because then the centre $c$ of the completed square is an integer and the lemma holds with equality. The order 2.5 at $\sigma = 0.5$ has $c = 2.5 \cdot 0 + (1 - 2.5) \cdot 1 = -1.5$, half-way between integers, and falls below the continuous value. At $\sigma = 1$ the lemma's slack is too small to show.

| $\sigma$ | $D_2$ (continuous) | $D_{2.5}$ (continuous) | $D_5$ (continuous) |
| --- | --- | --- | --- |
| 0.5 | 4.0000 (4.0000) | 4.9808 (5.0000) | 10.0000 (10.0000) |
| 1 | 1.0000 (1.0000) | 1.2500 (1.2500) | 2.5000 (2.5000) |
| 2 | 0.2500 (0.2500) | 0.3125 (0.3125) | 0.6250 (0.6250) |

: Rényi divergences between discrete Gaussians shifted by 1, by direct summation, with the continuous value $\alpha/(2\sigma^2)$ in parentheses. {#tbl-discrete-renyi}

### Check yourself

- Where does the proof of @eq-dgauss-renyi use that $\mu$ and $\nu$ are integers? *(In writing both PMFs with the same normalizer $Z$. For non-integer locations, the normalizers differ by the lemma, and the clean formula breaks.)*
- At $\sigma = 1$ and $\Delta = 1$, where are the kinks of the discrete $\delta(\varepsilon)$ curve? *(At the values $(1 - 2y)/2$ of the privacy loss that are positive: $\varepsilon = 0.5, 1.5, 2.5, \dots$, as in @fig-discrete-gaussian-delta.)*

## Sampling the discrete Gaussian exactly {#sec-discrete-gaussian-sampler}

### Context

The privacy analysis is only as good as the sampler. Inverse-CDF sampling of $\mathcal N_{\mathbb Z}(0, \sigma^2)$ needs the normalizer, which is irrational, and a floating-point uniform, which brings back the problems of @sec-discrete-float. Algorithm 3 of Canonne, Kamath and Steinke (2020) instead proposes from the discrete Laplace and thins with an exact $e^{-\gamma}$ coin.

### Statement

The discrete Laplace has heavier tails than the discrete Gaussian, so it can serve as the proposal of rejection sampling (Fact 2): draw from it, and throw draws away with a probability chosen so that proposal times acceptance is proportional to $e^{-y^2/(2\sigma^2)}$. The acceptance probability is a Gaussian factor in $|y|$, shifted so that it cancels the Laplace factor exactly.

Input a rational $\sigma^2 > 0$. Let $t = \lfloor\sigma\rfloor + 1$. Repeat: draw $Y \sim \mathrm{Lap}_{\mathbb Z}(t)$ with Algorithm 2 ($s = 1$), and accept it with probability
$$\exp\Big(-\frac{(|Y| - \sigma^2/t)^2}{2\sigma^2}\Big)$$
using Algorithm 1. Output the first accepted $Y$. The output is exactly $\mathcal N_{\mathbb Z}(0, \sigma^2)$. The acceptance probability per proposal is the proposal's normalizing constant times the constant left over from the cancellation times the discrete Gaussian's unnormalized total mass:
$$A = \frac{1 - r}{1 + r}\,e^{-\sigma^2/(2t^2)}\sum_{y \in \mathbb Z}e^{-y^2/(2\sigma^2)}, \qquad r = e^{-1/t},$$ {#eq-dgauss-accept}
which tends to $\sqrt{\pi/2}\,e^{-1/2} = 0.7602$ as $\sigma$ grows.

### Derivation

1. **Proposal times acceptance.** By Fact 2, the output is the target if proposal times acceptance is proportional to $e^{-y^2/(2\sigma^2)}$. The proposal has $\Pr[Y = y] \propto e^{-|y|/t}$. Multiply by the acceptance probability, add the exponents and expand the square:
$$-\frac{|y|}{t} - \frac{(|y| - \sigma^2/t)^2}{2\sigma^2} = -\frac{|y|}{t} - \frac{y^2}{2\sigma^2} + \frac{|y|}{t} - \frac{\sigma^2}{2t^2} = -\frac{y^2}{2\sigma^2} - \frac{\sigma^2}{2t^2}.$$
The $|y|/t$ terms cancel, and what is left is $e^{-y^2/(2\sigma^2)}$ times a constant. By Fact 2 the output is $\mathcal N_{\mathbb Z}(0, \sigma^2)$.
2. **The acceptance probability is valid.** The exponent $-(|y| - \sigma^2/t)^2/(2\sigma^2)$ is at most 0, so its exponential is at most 1 and is a probability.
3. **Everything is rational.** $t$ is an integer, computed as $\lfloor\sqrt{\lfloor\sigma^2\rfloor}\rfloor + 1$ with integer square roots (an integer $n$ has $n^2 \le \sigma^2$ exactly when $n^2 \le \lfloor\sigma^2\rfloor$, so this equals $\lfloor\sigma\rfloor + 1$ without computing $\sigma$). So $\gamma = (|y| - \sigma^2/t)^2/(2\sigma^2)$ is a ratio of integers, and Algorithm 1 applies.
4. **The acceptance rate.** $A = \sum_y\Pr[Y = y]\,e^{-(|y| - \sigma^2/t)^2/(2\sigma^2)}$, the average acceptance over proposals. With $\Pr[Y = y] = \frac{1 - r}{1 + r}r^{|y|}$, step 1 turns each term into $\frac{1 - r}{1 + r}e^{-\sigma^2/(2t^2)}e^{-y^2/(2\sigma^2)}$, and summing over $y$ gives @eq-dgauss-accept. [Exercise 5](solutions/07-discrete-mechanisms.md#sec-sol-discrete-5) derives the limit.

**Concretely, for $\sigma^2 = 4$** ($t = \lfloor 2\rfloor + 1 = 3$, $r = e^{-1/3}$, $\frac{1 - r}{1 + r} = 0.1651$). @tbl-discrete-dgauss-walk computes a few rows by hand.

| $y$ | Proposal $0.1651\,r^{\lvert y\rvert}$ | $\gamma = (\lvert y\rvert - 4/3)^2/8$ | Acceptance $e^{-\gamma}$ | Product |
| --- | --- | --- | --- | --- |
| 0 | 0.1651 | 2/9 | 0.8007 | 0.1322 |
| 1 | 0.1183 | 1/72 | 0.9862 | 0.1167 |
| 2 | 0.0848 | 1/18 | 0.9460 | 0.0802 |
| 4 | 0.0435 | 8/9 | 0.4111 | 0.0179 |
| 6 | 0.0223 | 49/18 | 0.0657 | 0.0015 |

: Algorithm 3 for $\sigma^2 = 4$, by hand. The product is $\propto e^{-y^2/8}$: divide each row by the first and you get $e^{-y^2/8}$, for instance $0.1167/0.1322 = 0.883 = e^{-1/8}$. {#tbl-discrete-dgauss-walk}

![Algorithm 3 for $\sigma^2 = 4$. Left: the discrete Laplace proposal with $t = 3$ and the part of it that is accepted, which is $A$ times the discrete Gaussian PMF. Right: 10,000 draws of Algorithm 3 against the exact PMF on a log scale, with the $\mathcal N(0, 4)$ density for comparison.](figures/discrete_gaussian_sampler.svg){#fig-discrete-gaussian-sampler}

### Interpretation

The sampler is exact and cheap. Every step is integer or rational arithmetic. The acceptance rate is 0.5185 at $\sigma^2 = \frac14$, 0.6629 at $\sigma^2 = 4$ and 0.7532 at $\sigma^2 = 100$, so a draw takes fewer than two proposals on average ($1/A$ by Fact 2, at most $1/0.5185 < 2$ here), and between 15 and 27 uniform integers in the script's runs (@tbl-discrete-code).

The choice $t = \lfloor\sigma\rfloor + 1$ only affects cost. Any integer $t \ge 1$ gives an exact sampler, because step 1 of the derivation works for every $t$. For large $\sigma$, $\frac{1 - r}{1 + r} \approx \frac{1}{2t}$ (since $r = e^{-1/t} \approx 1 - 1/t$) and the sum in @eq-dgauss-accept is about $\sqrt{2\pi}\sigma$ by @eq-poisson-theta, so $A \approx \frac{\sqrt{2\pi}\sigma}{2t}e^{-\sigma^2/(2t^2)}$. The derivative of $\ln A$ in $t$ is $-\frac1t + \frac{\sigma^2}{t^3}$, which is zero at $t = \sigma$, and $\lfloor\sigma\rfloor + 1$ is the nearest integer above it.

Exactness is not timing safety. The number of rounds depends on the output, so the running time leaks information about the noise. Jin et al. (2022) show such leaks are exploitable; a deployment pads the running time or runs a fixed number of rounds.

### Example: a histogram over 1,000 bins

Release a histogram of 1,000 bins under add/remove neighbours, where one record changes one bin by 1, so $\Delta_2 f = 1$. Add $\mathcal N_{\mathbb Z}(0, 100)$ to each bin. The noise in the 999 unchanged bins has the same distribution under both neighbours, so those bins add nothing to the privacy loss, and the privacy loss is that of one bin. @eq-dgauss-delta with $\Delta = 1$ and $\sigma = 10$ gives its $\delta(\varepsilon)$, essentially the continuous Gaussian's at $\sigma = 10$. The script's sampler at $\sigma^2 = 100$ uses about 15.6 uniform integers per bin, so the whole histogram needs about 15.6 × 1,000 = 15,600 uniform integers and no floating-point operation.

### Check yourself

- If you dropped the $\sigma^2/t$ inside the acceptance probability, accepting with $e^{-y^2/(2\sigma^2)}$ instead, what would the output be? *(Proposal times acceptance would be $\propto e^{-|y|/t - y^2/(2\sigma^2)}$, a discrete Gaussian tilted by a Laplace factor, not $\mathcal N_{\mathbb Z}(0, \sigma^2)$.)*
- Why does the sampler not need the normalizer $\sum_ye^{-y^2/(2\sigma^2)}$? *(Rejection sampling only needs proposal times acceptance to be proportional to the target, Fact 2. The normalizer sets the acceptance rate, not the output.)*

## Code: check every sampler against its exact distribution

The script [code/discrete_mechanisms.py](code/discrete_mechanisms.py) implements Algorithms 1–3 with `fractions.Fraction` and integers, drawing uniform integers from a seeded `random.Random` (swap in `secrets.SystemRandom()` to deploy). It also enumerates the toy floating-point sampler exactly, and computes the discrete Gaussian's PMF, variance, Rényi divergences and $\delta(\varepsilon)$ by summation. From the repository root:

```bash
uv run python book/part03-dp-foundations/code/discrete_mechanisms.py
```

**Output** (seed 0, 10,000 draws per sampler), summarized in @tbl-discrete-code. To compare a sampler with its exact distribution, the script uses the total variation (TV) distance: half the sum, over outputs, of the absolute difference between the exact probability and the observed frequency, which is the largest difference the two can show on any event. Even a perfect sampler shows some TV distance after 10,000 draws, from sampling noise alone. The "noise floor" measures how much: it is the TV distance between the exact PMF and the empirical PMF of 10,000 draws from it made with NumPy's sampler.

| Experiment | Result |
| --- | --- |
| Float Laplace, $p = 12$ | $\Pr[M(D) \in S] = 0.4791$; snapped to integers: 0, with $\max\lvert L\rvert = 1.0010$ |
| Algorithm 1, $\gamma = \frac12, 1, \frac52, 4$ | Frequencies 0.6024, 0.3709, 0.0869, 0.0168 against 0.6065, 0.3679, 0.0821, 0.0183; draws per coin 1.662, 3.096, 3.976, 4.218 against 1.649, 3.086, 3.941, 4.240 |
| Algorithm 2, $b = 1$ | Variance 1.8305 (exact 1.8413); TV to exact 0.0114 (noise floor 0.0074); 11.49 draws per sample |
| Algorithm 2, $b = 3/2$ | Variance 4.3834 (exact 4.3370); TV 0.0119 (noise floor 0.0243); 12.08 draws per sample |
| Algorithm 2, $b = 10$ | Variance 195.2985 (exact 199.8334); TV 0.0339 (noise floor 0.0331); 10.25 draws per sample |
| Algorithm 3, $\sigma^2 = \frac14$ | Variance 0.2189 (exact 0.2150); TV 0.0028 (noise floor 0.0045); acceptance 0.5185; 26.6 draws |
| Algorithm 3, $\sigma^2 = 1$ | Variance 0.9977 (exact 1.0000); TV 0.0076 (noise floor 0.0042); acceptance 0.5418; 24.8 draws |
| Algorithm 3, $\sigma^2 = 4$ | Variance 4.0826 (exact 4.0000); TV 0.0154 (noise floor 0.0142); acceptance 0.6629; 18.9 draws |
| Algorithm 3, $\sigma^2 = 100$ | Variance 99.4386 (exact 100.0000); TV 0.0243 (noise floor 0.0325); acceptance 0.7532; 15.6 draws |
| Discrete Gaussian variance | $\operatorname{Var}/\sigma^2$ = 0.010728, 0.860051, 1.000000 at $\sigma = 0.25, 0.5, 1$, by summation and by @eq-dgauss-var alike |

: The script's output with the default seed. {#tbl-discrete-code}

[tests/test_discrete_mechanisms.py](../../tests/test_discrete_mechanisms.py) checks that the output still matches.

**What to take from it**

The samplers are exact, up to sampling noise. Every TV distance is within about a factor 2 of the noise floor, sometimes below it, and every frequency and variance is within a few standard errors of its exact value. To see what a few standard errors means for a variance: a variance estimated from $n$ draws has a relative standard error of about $\sqrt{(\kappa - 1)/n}$, where $\kappa$ is the kurtosis, the fourth moment divided by the squared variance. The Gaussian has $\kappa = 3$, which gives $\sqrt{2/10{,}000} \approx$ 1.4%; the Laplace has $\kappa = 6$, which gives $\sqrt{5/10{,}000} \approx$ 2.2%. The largest deviation, at $b = 10$, is $(199.8334 - 195.2985)/199.8334 \approx$ 2.3% low, about one standard error. The noise floor varies a lot between runs, so only the order of magnitude matters.

The costs match the formulas. Algorithm 1's draws per coin match @eq-bern-exp and its extension to $\gamma > 1$ to within 1%.

The float leak doesn't shrink with precision, and snapping removes it: the first row reproduces @tbl-discrete-float and the snapped example.

The discrete Gaussian behaves like the continuous one for $\sigma \ge 1$, in privacy and in variance. Below that, its variance is smaller than $\sigma^2$ and its $\delta(\varepsilon)$ oscillates more around the continuous curve.

## References

- Mironov, "On Significance of the Least Significant Bits for Differential Privacy" (CCS 2012). The floating-point attack on the textbook Laplace sampler, the snapping mechanism and its proof (Theorem 1 and §5).
- Canonne, Kamath & Steinke, "The Discrete Gaussian for Differential Privacy" (NeurIPS 2020; journal version in the Journal of Privacy and Confidentiality, 2022). Algorithms 1–3, the exact $\delta(\varepsilon)$ (Theorem 7), the Rényi and concentrated-DP guarantee (Theorem 4), the variance and tail bounds via Poisson summation. Read §5 alongside the code here.
- Ghosh, Roughgarden & Sundararajan, "Universally Utility-Maximizing Privacy Mechanisms" (STOC 2009; SIAM Journal on Computing, 2012). The geometric mechanism and its optimality for a single count.
- von Neumann, "Various Techniques Used in Connection with Random Digits" (1951). The original trick behind Algorithm 1: exponential probabilities from comparisons of uniform variables.
- Jin, McMurtry, Rubinstein & Ohrimenko, "Are We There Yet? Timing and Floating-Point Attacks on Differential Privacy Systems" (IEEE S&P 2022). Floating-point and timing attacks on DP libraries released after Mironov's paper, including attacks on samplers whose running time depends on the output.
- Kairouz, Liu & Steinke, "The Distributed Discrete Gaussian Mechanism for Federated Learning with Secure Aggregation" (ICML 2021). The discrete Gaussian when each client adds its own noise, and how close sums of discrete Gaussians are to a discrete Gaussian.
- Abowd et al., "The 2020 Census Disclosure Avoidance System TopDown Algorithm" (Harvard Data Science Review, 2022). The largest deployment of the discrete Gaussian.
- Stein & Shakarchi, *Fourier Analysis: An Introduction* (2003), ch. 5. The Poisson summation formula and the theta function behind @eq-poisson-theta.

## Cheat sheet

$r = e^{-1/b}$ for the discrete Laplace; $X \sim \mathcal N_{\mathbb Z}(0, \sigma^2)$ for the discrete Gaussian; $\Delta$ is the integer change in $f$ between neighbours. @tbl-discrete-cheat collects the results.

| Object | Formula | Fact to remember |
| --- | --- | --- |
| Support argument | $\delta \ge \Pr[M(D) \in \mathcal Y_D \setminus \mathcal Y_{D'}]$ | Outputs reachable from one input only rule out every $\varepsilon$ |
| Floating-point Laplace | $\mathrm{round}(f(D) + \xi\,\mathrm{round}(b\ln(1/U)))$ | Not DP: about 48% of the mass identifies $f(D)$ in the toy, at every precision |
| Snapping | $\mathrm{clamp}_B(\lfloor\mathrm{clamp}_B(f) \oplus \xi\otimes\lambda\otimes\mathrm{LN}(U^*)\rceil_\Lambda)$ | $\varepsilon = \frac{1 + 12B\eta}{\lambda} + \frac{2^{-49}B}{\lambda}$ for sensitivity 1 |
| $\mathrm{Bernoulli}(e^{-\gamma})$, $\gamma \le 1$ | Parity of the first 0 among $\mathrm{Bernoulli}(\gamma/k)$ | $\Pr[K > k] = \gamma^k/k!$; $e^\gamma$ draws on average |
| $\mathrm{Lap}_{\mathbb Z}(b)$ | $\frac{1 - r}{1 + r}r^{\lvert z\rvert}$ | Variance $\frac{2r}{(1 - r)^2} \approx 2b^2 - \frac16$ |
| Geometric mechanism | $f(D) + \mathrm{Lap}_{\mathbb Z}(b)$ | $\varepsilon = \Delta_1 f/b$, pure DP |
| $\mathcal N_{\mathbb Z}(0, \sigma^2)$ | $\propto e^{-x^2/(2\sigma^2)}$ on $\mathbb Z$ | $\operatorname{Var} = \sigma^2 - O(\sigma^4e^{-2\pi^2\sigma^2})$ |
| Discrete Gaussian mechanism | $L(y) = \frac{\Delta^2 - 2\Delta y}{2\sigma^2}$ at integers | $\delta(\varepsilon)$ by @eq-dgauss-delta; $D_\alpha \le \frac{\alpha\Delta^2}{2\sigma^2}$ |
| Algorithm 3 | Propose $\mathrm{Lap}_{\mathbb Z}(\lfloor\sigma\rfloor + 1)$, accept w.p. $e^{-(\lvert y\rvert - \sigma^2/t)^2/(2\sigma^2)}$ | Acceptance $\to 0.7602$; exact |

: Every result in this chapter. {#tbl-discrete-cheat}

**Rules of thumb**

- Never ship `f(D) + laplace(b)` in floating point. Either snap the output to a grid at least as coarse as $b$ and clamp it, or make the query integer-valued and use an exact integer sampler.
- An exact sampler uses only integers, rationals and uniform integers from a cryptographic source. Any `exp`, `log` or `random()` in the sampler is a place where the analysed distribution and the real one can part.
- The privacy proofs of the discrete Laplace and Gaussian are the continuous proofs, because integer shifts leave the normalizer unchanged.
- For $\sigma \ge 1$, treat the discrete Gaussian as the continuous one for both accuracy and privacy; check $\delta(\varepsilon)$ with @eq-dgauss-delta when it matters.
- Exactness of the distribution is not the end: running time that depends on the output is a separate side channel.

**Practice.** Exercises for this chapter, each with a worked solution, are in [the exercise book](solutions/07-discrete-mechanisms.md).
