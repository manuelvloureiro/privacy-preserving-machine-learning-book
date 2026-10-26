# Cryptography basics

Some computations must hide something, such as a key, a message, or a value split among several holders. **What does it mean that an observer with limited computing power learns nothing it was not supposed to learn?** This chapter gives cryptography's answer as mathematics: definitions you can state, bounds you can derive and toy constructions you can run.

One idea organizes the answer: **security is a game**. A challenger flips a fair coin to pick one of two worlds, runs a procedure in that world, and shows the result to an adversary, who guesses the world. An adversary that learns nothing is right half the time, and its **advantage** measures how much better it does. A scheme is secure when every efficient adversary's advantage is tiny. Proofs change one thing at a time, through intermediate worlds called **hybrids**, and add up the advantage gained at each step.

Two examples run through the chapter. The first is a coin with heads probability 0.51, to be told apart from a fair coin. One flip gives an advantage of at most 0.01, and 10,000 flips give 0.683, far below the 1 that adding 0.01 per flip would suggest. The second is a 128-bit key, or a 128-bit block of data that a cipher encrypts, the sizes used in practice. Testing $10^{18}$ keys a second for a year finds a 128-bit key with probability $9.3\times10^{-14}$, yet a cipher on 128-bit blocks loses its guarantee after about $2^{64}$ blocks. Cryptography texts write the number of queries and the advantage as $q$ and $\varepsilon$, letters later parts of this book reserve for other uses, so @tbl-crypto-symbols writes them $Q$ and $\nu$. The chapter's script runs toy versions of every scheme, far too small to protect anything.

| Symbol | Meaning |
| --- | --- |
| $\mathcal A$; $\mathcal B$ | an adversary; one built from it by a reduction |
| $\nu$; $\lambda$ | an advantage; the security parameter |
| $Q$ | the number of samples, queries or draws |
| $P_0$, $P_1$ | the two worlds a game compares |
| $\xi$ | the coin's extra heads probability, 0.01 |
| $n$ | the bit length of messages and inputs |
| $K$; $m$; $y$ | a random key; a message; a ciphertext |
| $p$; $h$ | a modulus, not a density; the number of share holders |
| $d$, $s$, $e$ | the dimension, secret and error of learning with errors |

: The symbols of this chapter. {#tbl-crypto-symbols}

## Distinguishers and advantage {#sec-crypto-advantage}

Every definition in this chapter compares two worlds, so you first need a score for an observer who tries to tell them apart, as you would score a classifier of real against generated images. This section defines that score and shows that it measures the same thing as winning a guessing game.

::: {#def-crypto-advantage name="Distinguisher and advantage"}
A **distinguisher** $\mathcal A$ is any procedure that receives one sample and outputs a bit. Its **advantage** in telling a distribution $P_0$ from a distribution $P_1$ is
$$\mathrm{Adv}_{\mathcal A}(P_0, P_1) = \big|\Pr[\mathcal A(P_1) = 1] - \Pr[\mathcal A(P_0) = 1]\big|,$$
where $\Pr[\mathcal A(P_1) = 1]$ is the probability that $\mathcal A$ outputs 1 on a sample from $P_1$. In the **distinguishing game**, a challenger flips a fair bit $b$ and gives $\mathcal A$ a sample from $P_b$, and $\mathcal A$ **wins** if its guess $b'$ equals $b$.
:::

The coin in the game is fair on purpose, so that an adversary with nothing to go on wins exactly half the time, whatever it answers. The next lemma makes this exact, and later proofs use it to move between win rates and advantages.

::: {#lem-crypto-win name="Winning and advantage"}
For any distinguisher $\mathcal A$ in the distinguishing game,
$$\Pr[b' = b] = \frac12 + \frac12\big(\Pr[\mathcal A(P_1) = 1] - \Pr[\mathcal A(P_0) = 1]\big).$$ {#eq-crypto-win-advantage}
So a distinguisher with advantage $\nu$ yields one that wins with probability $\frac12 + \frac\nu2$.
:::

::: {.proof}
1. *Split the win by world.* Each world has probability $\frac12$, and $\mathcal A$ wins in world 1 by saying 1 and in world 0 by saying 0, so
$$\Pr[b' = b] = \tfrac12\Pr[\mathcal A(P_1) = 1] + \tfrac12\Pr[\mathcal A(P_0) = 0].$$
2. *Count only the outputs of 1.* Saying 0 is the complement of saying 1, so replacing $\Pr[\mathcal A(P_0) = 0]$ by $1 - \Pr[\mathcal A(P_0) = 1]$ and collecting terms gives @eq-crypto-win-advantage.
3. *Flip a losing distinguisher.* If the bracket is negative, the distinguisher that outputs the opposite bit negates it, keeps the advantage, and wins with probability $\frac12 + \frac\nu2$.
:::

Write $\mathrm{TV}(P_0, P_1)$ for the **total variation (TV) distance**, the total excess of probability that $P_1$ puts on outcomes it makes likelier than $P_0$ does. [Divergences](03-divergences.md) shows that the best guess, which says 1 exactly on those outcomes, wins with probability $\frac12 + \frac12\mathrm{TV}(P_0, P_1)$. Comparing with @lem-crypto-win, no advantage exceeds the TV distance, however much time the distinguisher uses:
$$\mathrm{Adv}_{\mathcal A}(P_0, P_1) \le \mathrm{TV}(P_0, P_1).$$ {#eq-crypto-tv}
A TV distance of 0 is **perfect** security and a small one is **statistical** security. Cryptography's main business is the third case, where the TV distance is large but no *efficient* distinguisher can exploit it.

::: {#exm-crypto-coin}
Let $P_0$ be a fair coin and $P_1$ a coin with heads probability 0.6, a larger bias than the running example's to keep the arithmetic visible. The distinguisher outputs 1 on heads. Each row of @tbl-crypto-coin-game is one outcome, with probability $\frac12$ for the challenger's bit times the probability of the coin's outcome.

| $b$ | Coin | Outcome | Probability | $\mathcal A$ outputs | $\mathcal A$ wins? |
| --- | --- | --- | --- | --- | --- |
| 0 | fair | tails | 0.25 | 0 | yes |
| 0 | fair | heads | 0.25 | 1 | no |
| 1 | biased | tails | 0.20 | 0 | no |
| 1 | biased | heads | 0.30 | 1 | yes |

: The distinguishing game for a fair against a 0.6-biased coin, with the distinguisher "output 1 on heads". {#tbl-crypto-coin-game}

The adversary wins with probability $0.25 + 0.30 = 0.55$. Its advantage is $0.6 - 0.5 = 0.1$, and indeed $0.55 = \frac12 + \frac{0.1}2$. The excess of heads, 0.1, is also the TV distance, so no distinguisher does better from one flip, and for the running example's coin the best is 0.01.
:::

## Negligible functions {#sec-crypto-negligible}

A key of 40 bits can be found by trying every key, and a key of 256 bits cannot. So security says how fast the attacker's success shrinks as a **security parameter** $\lambda$, the key length in bits, grows. Calling a scheme secure first needs a notion of efficient attackers and of tiny success probabilities.

::: {#def-crypto-negligible name="Efficient adversaries and negligible functions"}
An adversary is **probabilistic polynomial time (PPT)** if it is a randomized algorithm whose running time is bounded by a polynomial in $\lambda$. A function $\mu\colon \mathbb N \to [0, 1]$ is **negligible** if, for every exponent $c$, it eventually falls below $\lambda^{-c}$ and stays there:
$$\forall c > 0\ \ \exists \lambda_0\ \ \forall \lambda \ge \lambda_0:\quad \mu(\lambda) < \lambda^{-c}.$$ {#eq-crypto-negligible}
A scheme is **secure** if every PPT adversary has negligible advantage in its security game.
:::

An efficient adversary can repeat its attack a polynomial number of times, so "tiny" must stay tiny after multiplication by any polynomial. The next proposition says that it does, because a negligible function beats every inverse polynomial, including one steep enough to also cancel the polynomial.

::: {#prp-crypto-closure name="Negligible functions survive polynomial repetition"}
If $\mu$ is negligible and $\mathrm{poly}$ is a polynomial, then $\mathrm{poly}(\lambda)\mu(\lambda)$ is negligible. The sum of two negligible functions is negligible.
:::

::: {.proof}
1. *Bound the polynomial by a power.* A polynomial of degree $c' - 1$ is eventually below $\lambda^{c'}$, so $\mathrm{poly}(\lambda) \le \lambda^{c'}$ for all $\lambda \ge \lambda_1$.
2. *Ask $\mu$ for a steeper decay.* Fix any $c > 0$. Applying @eq-crypto-negligible with the exponent $c + c'$ gives $\mu(\lambda) < \lambda^{-(c+c')}$ for all $\lambda \ge \lambda_2$.
3. *Multiply the two bounds.* For $\lambda \ge \max(\lambda_1, \lambda_2)$,
$$\mathrm{poly}(\lambda)\mu(\lambda) < \lambda^{c'}\lambda^{-(c+c')} = \lambda^{-c},$$
and since $c$ was arbitrary, the product is negligible.
4. *Handle a sum.* With the exponent $c + 1$, each of two negligible functions is eventually below $\lambda^{-(c+1)}$, so their sum is below $2\lambda^{-(c+1)}$, which is at most $\lambda^{-c}$ once $\lambda \ge 2$.
:::

By the union bound of [Concentration inequalities](02-concentration.md), the chance that at least one of several tries succeeds is at most the sum of their chances. So an attack repeated polynomially many times still succeeds with negligible probability. Brute force, trying $t$ of the $2^\lambda$ keys, succeeds with probability $t/2^\lambda$, negligible for polynomial $t$ but never zero. Taking $\log_2$ turns each test into a comparison of exponents. So $2^{-\lambda}$, $\lambda^{-\log_2\lambda}$ and $2^{-\sqrt\lambda}$ are negligible, but $\lambda^{-10}$ is not, because it is never below $\lambda^{-11}$.

::: {#exm-crypto-negligible}
At a fixed key length, "negligible" can mislead. @tbl-crypto-negligible-128 evaluates the four functions at $\lambda = 128 = 2^7$.

| $\mu(\lambda)$ | Negligible? | Value at $\lambda = 128$ |
| --- | --- | --- |
| $2^{-\lambda}$ | yes | $2^{-128} = 2.9\times10^{-39}$ |
| $\lambda^{-\log_2\lambda}$ | yes | $2^{-49} = 1.8\times10^{-15}$ |
| $2^{-\sqrt\lambda}$ | yes | $2^{-11.31} = 3.9\times10^{-4}$ |
| $\lambda^{-10}$ | no | $2^{-70} = 8.5\times10^{-22}$ |

: Four functions at $\lambda = 128$. The negligible $2^{-\sqrt\lambda}$ is far larger than the non-negligible $\lambda^{-10}$. {#tbl-crypto-negligible-128}

The non-negligible $\lambda^{-10}$ is smaller than two of the negligible functions, and $2^{-\sqrt\lambda}$ drops below it for good only at $\lambda = 20{,}520$. @fig-crypto-negligible plots the **bits of security** $-\log_2\mu(\lambda)$, the number of halvings from 1 down to $\mu(\lambda)$.
:::

![Bits of security $-\log_2\mu(\lambda)$ for three negligible functions and the non-negligible $\lambda^{-10}$, on log–log axes. Each negligible curve eventually overtakes the dashed one; $2^{-\sqrt\lambda}$ does so only beyond $\lambda = 20{,}520$.](figures/crypto_basics_negligible.svg){#fig-crypto-negligible}

## Indistinguishability and the hybrid argument {#sec-crypto-hybrid}

Most security definitions say that two things look the same, such as a ciphertext and random bits. This section makes that precise and answers the first question every proof meets: if one sample cannot be told apart, can many? Its tool, the hybrid argument, is the skeleton of every later proof.

::: {#def-crypto-indist name="Computational indistinguishability"}
Two families of distributions $P_0 = \{P_{0,\lambda}\}$ and $P_1 = \{P_{1,\lambda}\}$, one distribution for each $\lambda$, are **computationally indistinguishable**, written $P_0 \approx_c P_1$, if every PPT distinguisher has negligible advantage $\mathrm{Adv}_{\mathcal A}(P_{0,\lambda}, P_{1,\lambda})$. If their TV distance is negligible they are **statistically indistinguishable**, written $\approx_s$, and if they are equal they are **perfectly indistinguishable**, written $\equiv$.
:::

By @eq-crypto-tv, $\equiv$ implies $\approx_s$, which implies $\approx_c$. The hybrid argument walks from one world to the other through intermediate worlds, so its proof needs to know how advantages combine along such a chain. The next lemma says they add at most.

::: {#lem-crypto-chain name="Advantages add along a chain"}
For any distributions $H_0, H_1, \dots, H_Q$ and any distinguisher $\mathcal A$,
$$\mathrm{Adv}_{\mathcal A}(H_0, H_Q) \le \sum_{i=1}^Q \mathrm{Adv}_{\mathcal A}(H_{i-1}, H_i).$$
:::

::: {.proof}
1. *Telescope.* In the sum over $i$ of $\Pr[\mathcal A(H_{i-1}) = 1] - \Pr[\mathcal A(H_i) = 1]$, each intermediate probability appears once with each sign, so the sum is $\Pr[\mathcal A(H_0) = 1] - \Pr[\mathcal A(H_Q) = 1]$.
2. *Take absolute values.* The absolute value of a sum is at most the sum of the absolute values.
:::

The hybrid lemma walks from $Q$ samples of $P_0$ to $Q$ samples of $P_1$ by swapping one sample at a time. If the two ends are $\nu$ apart, a typical step is worth $\nu/Q$, and a single step is a one-sample problem. Write $P^{\otimes Q}$ for $Q$ independent samples from $P$.

::: {#thm-crypto-hybrid name="The hybrid lemma"}
Let $P_0 \approx_c P_1$ be families that an efficient algorithm can sample from, and let $Q = Q(\lambda)$ be a polynomial. Then $P_0^{\otimes Q} \approx_c P_1^{\otimes Q}$. Quantitatively, a distinguisher with advantage $\nu$ on $Q$ samples yields a distinguisher with advantage $\nu/Q$ on one sample, with about the same running time.
:::

::: {.proof}
The proof builds a one-sample distinguisher that lands at a random link of a chain of hybrids, computes its advantage by telescoping, and concludes by contradiction.

1. *Define the hybrids.* Let $H_i$ be $Q$ independent entries, the first $i$ from $P_1$ and the rest from $P_0$, so $H_0 = P_0^{\otimes Q}$ and $H_Q = P_1^{\otimes Q}$.
2. *Build a one-sample distinguisher.* On input $w$, let $\mathcal B$ pick a position $i$ uniformly from $\{1, \dots, Q\}$, draw the other entries itself as in $H_i$, which needs efficient sampling, put $w$ in position $i$, and run $\mathcal A$. The tuple is distributed as $H_{i-1}$ if $w$ comes from $P_0$, and as $H_i$ if it comes from $P_1$.
3. *Average over the position.* Each $i$ has probability $\frac1Q$, so
$$\Pr[\mathcal B(P_0) = 1] - \Pr[\mathcal B(P_1) = 1] = \frac1Q\sum_{i=1}^Q\big(\Pr[\mathcal A(H_{i-1}) = 1] - \Pr[\mathcal A(H_i) = 1]\big).$$
4. *Telescope.* As in the proof of @lem-crypto-chain, the sum equals $\Pr[\mathcal A(H_0) = 1] - \Pr[\mathcal A(H_Q) = 1]$, so $\mathcal B$ has advantage $\nu/Q$.
5. *Conclude by contradiction.* If $\nu$ were not negligible, there would be a $c$ with $\nu(\lambda) \ge \lambda^{-c}$ for infinitely many $\lambda$. With $Q(\lambda) \le \lambda^{c'}$, the efficient $\mathcal B$ would then have advantage at least $\lambda^{-(c+c')}$ for those $\lambda$, which contradicts $P_0 \approx_c P_1$.
:::

::: {#exm-crypto-coin-hybrid}
Let $P_0$ be a fair coin and $P_1$ the running example's coin, with heads probability $\frac12 + \xi$. Read in reverse, the hybrid lemma allows at most $Q\xi$ from $Q$ flips, a factor $Q$ that is pessimistic because the best attacker pools its samples. @tbl-crypto-coin-hybrid and @fig-crypto-hybrid compare that bound with the exact best advantage and with majority vote, which outputs 1 when more than half the flips are heads.

| Flips $Q$ | Majority vote (simulation) | Best possible (TV) | Hybrid bound $\min(1, Q\xi)$ |
| --- | --- | --- | --- |
| 1 | 0.008 | 0.010 | 0.010 |
| 100 | 0.082 | 0.080 | 1.000 |
| 10,000 | 0.481 | 0.683 | 1.000 |

: Advantage in telling $Q$ flips of a $0.51$-coin from $Q$ fair flips (seed 0, 200,000 simulated games per entry). {#tbl-crypto-coin-hybrid}

Majority vote's 0.082 is simulation noise. Both counts of heads have a standard deviation of about $\sqrt Q/2$, and their means differ by $Q\xi$, so they are close to bell curves whose centres are $2\xi\sqrt Q$ standard deviations apart. So the best advantage is about $2\Phi(\xi\sqrt Q) - 1$, with $\Phi$ the standard normal distribution function, and it grows like $\xi\sqrt Q$ rather than $\xi Q$ ([Exercise 2](../../book-exercises/part01-prerequisites/08-crypto-basics.md#sec-sol-crypto-2)). Majority vote stalls near $\frac12$, because its threshold sits at the centre of the fair curve, which the fair count still exceeds half the time ([Exercise 3](../../book-exercises/part01-prerequisites/08-crypto-basics.md#sec-sol-crypto-3)).
:::

![Advantage of telling $Q$ flips of a coin with bias $\frac12 + 0.01$ from fair flips: majority vote (simulation), the best possible distinguisher (exact TV distance), and the hybrid bound $Q\xi$.](figures/crypto_basics_hybrid.svg){#fig-crypto-hybrid}

## Concrete security {#sec-crypto-concrete}

Real systems fix $\lambda$, and a statement for all large enough $\lambda$ does not say whether 128 is large enough. Concrete security replaces "efficient" and "negligible" by explicit numbers, so that a proof says how much a key can withstand.

::: {#def-crypto-concrete name="Concrete security and reductions"}
A primitive is **$(t, Q, \nu)$-secure** if every adversary that runs in time at most $t$ and makes at most $Q$ queries to its **oracles**, the procedures the game lets it call, has advantage at most $\nu$. It has **$\kappa$-bit security** if, roughly, every adversary with time $t$ has advantage at most $t/2^\kappa$, as for brute force over $\kappa$-bit keys. A **reduction** turns an adversary that breaks a scheme with $(t, Q, \nu)$ into one that breaks a primitive with $(t', Q', \nu')$, and it is **tight** when $\nu' \approx \nu$.
:::

Read in reverse, a reduction says that a hard primitive makes a hard scheme, and its loss costs bits: one that turns $\nu$ into $\nu/2^{20}$ leaves a 128-bit primitive only 108 bits.

::: {#exm-crypto-brute}
An exascale machine that tests $10^{18}$ keys per second for a year of about $3.16\times10^7$ seconds tries $2^{84.7}$ keys. Against a 128-bit key it succeeds with probability $2^{84.7 - 128} = 9.3\times10^{-14}$, and trying all $2^{128}$ keys would take $1.08\times10^{13}$ years.
:::

## Pseudorandom generators and functions {#sec-crypto-prf}

The schemes later in this chapter need many random-looking values from one short random key, because truly random bits are expensive to share. This section defines the two objects that supply them, generators and functions, and builds the second from the first.

::: {#def-crypto-prg name="Pseudorandom generator"}
Write $U_\ell$ for the uniform distribution on $\ell$-bit strings. An efficient deterministic map $G$ from $\lambda$-bit strings to $\ell(\lambda)$-bit strings, with $\ell(\lambda) > \lambda$, is a **pseudorandom generator (PRG)** if
$$G(U_\lambda) \approx_c U_{\ell(\lambda)}.$$
:::

Pseudorandomness is a statement about efficient distinguishers only, because the outputs of $G$ form a tiny subset of all strings.

::: {#prp-crypto-prg-statistical name="A pseudorandom generator is never statistically secure"}
For every $G$ as in @def-crypto-prg, $\mathrm{TV}(G(U_\lambda), U_\ell) \ge \frac12$.
:::

::: {.proof}
1. *An exhaustive distinguisher.* Let $\mathcal A$ output 1 exactly when its input equals $G(u)$ for some seed $u$. On $G(U_\lambda)$ it always outputs 1.
2. *Few strings are outputs.* At most $2^\lambda$ of the $2^\ell$ strings are outputs, so on $U_\ell$, $\mathcal A$ outputs 1 with probability at most $2^{\lambda - \ell} \le \frac12$.
3. *Bound the distance.* So $\mathcal A$ has advantage at least $\frac12$, and @eq-crypto-tv gives the bound.
:::

This distinguisher loops over $2^\lambda$ seeds, so it is not efficient. A **random function** from a set $\mathcal X$ to a set $\mathcal Y$ gives each input its own independent, uniform output. A pseudorandom function must look like one even to an adversary who chooses its questions.

::: {#def-crypto-prf name="Pseudorandom function"}
A keyed function $F_K\colon \mathcal X \to \mathcal Y$, with a $\lambda$-bit key $K$, is a **pseudorandom function (PRF)** if every PPT adversary has negligible advantage in this game. The challenger flips $b$, and its oracle is $F_K$ for a uniform key $K$ if $b = 0$ and a random function if $b = 1$. The adversary sends $Q$ queries, each of which may depend on earlier answers, and guesses $b$.
:::

A random function on 128-bit inputs is a table of $2^{128}$ entries, which nobody can store, and a PRF gives the same appearance from a 128-bit key. The toy code uses the hash-based message authentication code (HMAC), a keyed hash that practice treats as a PRF.

### The GGM tree {#sec-crypto-ggm}

A PRF can be built from any PRG that doubles its input's length. The construction of Goldreich, Goldwasser & Micali (1986), the GGM tree, uses the generator to grow a binary tree of keys, picking the left or right half of the output by the next bit of the input.

::: {#thm-crypto-ggm name="The GGM tree is a pseudorandom function"}
Let $G$ map $\lambda$ bits to $2\lambda$ bits, with output halves $G_0(u)$ and $G_1(u)$. For an input $x = x_1\cdots x_n$, start at the node $K$, replace the current node $v$ by $G_{x_i}(v)$ for $i = 1, \dots, n$, and let $F_K(x)$ be the leaf reached. Then for every adversary $\mathcal A$ making $Q$ queries there is a PRG adversary $\mathcal B$ with about the same running time such that
$$\mathrm{Adv}_{\mathrm{PRF}}(\mathcal A) \le nQ\cdot\mathrm{Adv}_{\mathrm{PRG}}(\mathcal B).$$
:::
<!-- proof: cited -->

The proof is in Goldreich (2001, §3.6), and Appendix @sec-apx-crypto-ggm derives it step by step. It makes the tree random one level at a time, and each of the $n$ levels costs at most $Q$ times the PRG advantage by @thm-crypto-hybrid.

::: {#exm-crypto-ggm}
On the 2-bit input $10$, the tree computes $G_1(K)$ and then outputs its left half $G_0(G_1(K))$. For 128-bit inputs, one evaluation walks 128 levels and costs 128 generator calls. [Exercise 6](../../book-exercises/part01-prerequisites/08-crypto-basics.md#sec-sol-crypto-6) builds the tree from SHA-256 and works out what the factor $nQ$ costs in bits.
:::

## The birthday bound {#sec-crypto-birthday}

Many arguments about random values fail only if two of them coincide, such as a repeated nonce, a random value meant to be used once, so you need the probability of a coincidence. Among $Q$ values there are about $Q^2/2$ pairs, and each coincides with probability $1/N$, so collisions become likely after about $\sqrt N$ draws rather than $N$.

::: {#thm-crypto-birthday name="The birthday bound"}
Draw $Q$ values independently and uniformly from a set of size $N$. Then
$$1 - e^{-Q(Q-1)/(2N)} \;\le\; \Pr[\text{collision}] \;\le\; \frac{Q(Q-1)}{2N} \le \frac{Q^2}{2N},$$ {#eq-crypto-birthday}
and the collision probability reaches $\frac12$ after about $1.18\sqrt N$ draws.
:::

::: {.proof}
The upper bound counts pairs. The lower bound bounds the exact probability of no collision by an exponential.

1. *Upper bound.* Each of the $Q(Q-1)/2$ pairs coincides with probability $1/N$, so the union bound gives the upper bound.
2. *Exact value.* Draw the values one at a time. If the first $i$ are distinct, the next avoids them with probability $1 - i/N$, so
$$\Pr[\text{no collision}] = \prod_{i=0}^{Q-1}\Big(1 - \frac iN\Big).$$
3. *Lower bound.* The inequality $1 - x \le e^{-x}$ of [Concentration inequalities](02-concentration.md) bounds each factor, and the exponents add to $Q(Q-1)/(2N)$. Subtracting from 1 gives the lower bound.
4. *The 50% point.* Setting the lower bound to $\frac12$, with $Q(Q-1) \approx Q^2$, gives $Q^2/(2N) = \ln 2$, so $Q \approx \sqrt{2N\ln 2} \approx 1.18\sqrt N$.
:::

The two bounds are close, because $1 - e^{-x}$ bends downwards and so lies above its chord $(1 - 1/e)x$ on $[0, 1]$. Random 128-bit values start to collide after about $2^{64}$ draws.

::: {#exm-crypto-birthday}
@tbl-crypto-birthday gives the exact probability and both bounds for three cases. For the class of 23, the union bound is $23 \times 22/730 = 0.693$.

| Case | $Q$ | $N$ | Exact | Union bound | $1 - e^{-Q(Q-1)/(2N)}$ |
| --- | --- | --- | --- | --- | --- |
| Shared birthday in a class | 23 | 365 | 0.507 | 0.693 | 0.500 |
| Hashing trick: 1,000 features into $2^{20}$ buckets | 1,000 | $2^{20}$ | 0.379 | 0.476 | 0.379 |
| Random 32-bit identifiers for 100,000 records | 100,000 | $2^{32}$ | 0.688 | 1.164 | 0.688 |

: Collision probability, exact against the two bounds of @eq-crypto-birthday. {#tbl-crypto-birthday}

@fig-crypto-birthday checks @eq-crypto-birthday for $N = 2^{16}$. At $Q = 256$ the exact probability is 0.3927, a simulation of 100,000 trials gives 0.3920, and the union bound gives 0.4980. The 50% point is $Q = 301$, against $1.18 \times 256 \approx 302$.
:::

![Probability of a collision among $Q$ uniform draws from $N = 2^{16}$ values: exact, simulated, the union bound $Q(Q-1)/(2N)$ and the lower bound $1 - e^{-Q(Q-1)/(2N)}$.](figures/crypto_basics_birthday.svg){#fig-crypto-birthday}

## Block ciphers and the switching lemma {#sec-crypto-switching}

A block cipher such as the Advanced Encryption Standard (AES) is a keyed *permutation*, which never gives two inputs the same output, while most proofs assume a PRF. This section bridges the gap and turns the birthday bound into a limit on how much data one key may encrypt. A random function sometimes repeats an output and a random permutation never does, so an adversary can tell them apart only about as often as $Q$ random values collide.

::: {#def-crypto-prp name="Pseudorandom permutation"}
A keyed permutation of $\mathcal X$ is a **pseudorandom permutation (PRP)** if it satisfies @def-crypto-prf with the random function replaced by a uniformly random permutation of $\mathcal X$.
:::

::: {#thm-crypto-switching name="The switching lemma"}
Let $\mathcal A$ make $Q$ queries, and write $\mathcal A^{\mathrm{RF}}$ and $\mathcal A^{\mathrm{RP}}$ for $\mathcal A$ run with a random function and with a random permutation of $\mathcal X$ as its oracle. Then
$$\big|\Pr[\mathcal A^{\mathrm{RF}} = 1] - \Pr[\mathcal A^{\mathrm{RP}} = 1]\big| \le \frac{Q(Q-1)}{2|\mathcal X|} \le \frac{Q^2}{2|\mathcal X|}.$$ {#eq-crypto-switching}
:::

::: {.proof}
The plan is to run both oracles from one pre-drawn list of answers and show that they behave identically unless the list has a repeat.

1. *Simplify the adversary.* Assume the $Q$ queries are distinct, since both oracles answer a repeated query the same way.
2. *Pre-sample the function's answers.* Draw $Y_1, \dots, Y_Q$ independently and uniformly from $\mathcal X$ and answer the $i$-th query with $Y_i$. On distinct queries this is exactly a random function.
3. *Do the same for the permutation.* A random permutation answers each new query with a value uniform over those not yet used, so it can be run from a uniform list of $Q$ *distinct* values.
4. *Compare the two.* Let $\mathrm{Col}$ be the event that the list of step 2 has a repeat. Every list is equally likely, so conditioning on no repeat gives a uniform list of distinct values, exactly the permutation's. Hence $\Pr[\mathcal A^{\mathrm{RF}} = 1 \mid \text{no } \mathrm{Col}] = \Pr[\mathcal A^{\mathrm{RP}} = 1]$, which we call $\rho$.
5. *Bound the difference.* Splitting on $\mathrm{Col}$ and subtracting $\rho$,
$$\Pr[\mathcal A^{\mathrm{RF}} = 1] - \rho = \Pr[\mathrm{Col}]\big(\Pr[\mathcal A^{\mathrm{RF}} = 1 \mid \mathrm{Col}] - \rho\big),$$
and the bracket is a difference of probabilities, so the whole is at most $\Pr[\mathrm{Col}]$ in absolute value.
6. *Count the repeats.* @eq-crypto-birthday with $N = |\mathcal X|$ bounds $\Pr[\mathrm{Col}]$.
:::

The event $\mathrm{Col}$ is fixed before the adversary acts, which keeps step 4 valid for adaptive queries (Bellare & Rogaway 2006). The bound is tight, because the adversary that outputs 1 when two answers collide does so with probability $\Pr[\mathrm{Col}]$ against a function and never against a permutation.

::: {#cor-crypto-prp-prf name="A pseudorandom permutation is a pseudorandom function"}
A PRP with advantage at most $\nu$ against $Q$-query adversaries is a PRF with advantage at most $\nu + Q^2/(2|\mathcal X|)$.
:::

::: {.proof}
Go from $F_K$ to a random permutation, at cost at most $\nu$ by assumption, and then to a random function, at cost at most $Q^2/(2|\mathcal X|)$ by @thm-crypto-switching. By @lem-crypto-chain the total is at most the sum.
:::

::: {#exm-crypto-data-limit}
For a cipher on 128-bit blocks, the extra term after $Q = 2^k$ blocks is $Q^2/2^{129} = 2^{2k - 129}$. After $2^{32}$ blocks of 16 bytes, which is 64 GiB, it is at most $2^{-65}$, and after $2^{48}$ blocks, about 4 PiB, at most $2^{-33}$. At $2^{64}$ blocks it reaches $\frac12$, so keys must be changed long before.
:::

## Perfect secrecy and the one-time pad {#sec-crypto-otp}

Encryption should hide the message. The strongest possible answer is a ciphertext that tells even an unbounded adversary nothing, and Shannon (1949) showed what it costs: a key as long as the data. That cost is why the computational definition of the next section exists.

::: {#def-crypto-perfect name="Perfect secrecy"}
An encryption scheme with a random key $K$ has **perfect secrecy** if, for all messages $m, m'$ and ciphertexts $y$,
$$\Pr[\mathrm{Enc}_K(m) = y] = \Pr[\mathrm{Enc}_K(m') = y].$$
:::

Equivalently, by Bayes' rule, seeing the ciphertext does not change the probability of any message, for any prior on messages that is independent of the key. The proof for the one-time pad rests on one counting fact, which secret sharing and learning with errors use again.

::: {#lem-crypto-bijection name="A bijection maps uniform to uniform"}
If $U$ is uniform on a finite set $S$ and $f\colon S \to S$ is a bijection, then $f(U)$ is uniform on $S$.
:::

::: {.proof}
Each $t \in S$ comes from exactly one element, $f^{-1}(t)$, so $\Pr[f(U) = t] = \Pr[U = f^{-1}(t)] = 1/|S|$.
:::

::: {#thm-crypto-otp name="The one-time pad has perfect secrecy"}
For $n$-bit messages, let $K$ be uniform on $n$-bit strings and $\mathrm{Enc}_K(m) = m \oplus K$, where $\oplus$ is bitwise exclusive or (XOR). Then every ciphertext has probability $2^{-n}$ for every message, so the scheme has perfect secrecy.
:::

::: {.proof}
1. *The pad is a bijection.* For fixed $m$, the map $k \mapsto m \oplus k$ is its own inverse, since $m \oplus m = 0$.
2. *Apply the lemma.* By @lem-crypto-bijection, $m \oplus K$ is uniform whatever $m$ is.
:::

No scheme can do better on key length, so a 128-bit message needs a fresh 128-bit key.

::: {#prp-crypto-key-length name="Perfect secrecy needs as many keys as messages"}
If a scheme with key set $\mathcal K$ and message set $\mathcal M$ has perfect secrecy and correct decryption, then $|\mathcal K| \ge |\mathcal M|$.
:::

::: {.proof}
1. *Few messages reach one ciphertext.* Fix a ciphertext $y$ that occurs. Each key decrypts $y$ to one message, so at most $|\mathcal K|$ messages can produce $y$.
2. *Fewer keys break secrecy.* If $|\mathcal K| < |\mathcal M|$, some message never produces $y$ while another does, so their ciphertext distributions differ.
:::

The pad must also never be reused, because XOR-ing two ciphertexts under one pad cancels it and reveals $m_1 \oplus m_2$. [Exercise 5](../../book-exercises/part01-prerequisites/08-crypto-basics.md#sec-sol-crypto-5) shows how English text then gives away both messages.

::: {#exm-crypto-otp}
@tbl-crypto-otp shows the pad on 2-bit messages. Each message reaches every ciphertext under exactly one key.

| Key $K$ | 00 | 01 | 10 | 11 |
| --- | --- | --- | --- | --- |
| Ciphertext for $m = 01$ | 01 | 00 | 11 | 10 |
| Ciphertext for $m = 10$ | 10 | 11 | 00 | 01 |

: The one-time pad on 2-bit messages. Each row contains every ciphertext once, so the ciphertext is uniform for every message. {#tbl-crypto-otp}

:::

## Chosen-plaintext security {#sec-crypto-cpa}

Practical encryption accepts a negligible advantage in exchange for a short key. This section gives the definition that matters in practice and its surprising consequence, that encryption must be randomized, and then proves a randomized scheme secure. In the game, the adversary picks two messages and must say which one a ciphertext encrypts, which Goldwasser & Micali (1984) showed is equivalent to their **semantic security**.

::: {#def-crypto-cpa name="Indistinguishability under chosen-plaintext attack"}
A scheme is indistinguishable under chosen-plaintext attack (CPA), or **IND-CPA secure**, if every PPT adversary has negligible advantage in this game. The challenger draws a key $K$ and a fair bit $b$. The adversary may ask for encryptions of messages of its choice at any time. It submits two messages $m_0$ and $m_1$ of equal length, receives the challenge $y^* = \mathrm{Enc}_K(m_b)$, and outputs a guess $b'$.
:::

By @lem-crypto-win this advantage is also $2|\Pr[b' = b] - \frac12|$, and the messages have equal length because ciphertexts usually reveal lengths. If encryption always maps a message to the same ciphertext, the adversary can encrypt $m_0$ itself and compare.

::: {#prp-crypto-deterministic name="Deterministic encryption is never IND-CPA secure"}
If $\mathrm{Enc}_K$ is a deterministic function of the message and decryption is correct, some adversary that makes one encryption query has advantage 1.
:::

::: {.proof}
1. *The attack.* Ask for $y_0 = \mathrm{Enc}_K(m_0)$, submit $m_0$ and $m_1$, and output 0 if $y^* = y_0$ and 1 otherwise.
2. *World 0.* If $b = 0$, determinism gives $y^* = y_0$, so the guess is 0.
3. *World 1.* If $b = 1$, correct decryption forces $\mathrm{Enc}_K(m_1) \ne \mathrm{Enc}_K(m_0)$, so the guess is 1.
:::

The fix is a fresh pad for every encryption. Draw a uniform $r$ from a set $\mathcal R$, use $F_K(r)$ as a one-time pad, and send $r$ along. The scheme fails only if the PRF is distinguishable from random or the challenge reuses a seen $r$, and the bound adds the two.

::: {#thm-crypto-prf-cpa name="A randomized scheme from a PRF"}
Let $\mathrm{Enc}_K(m) = (r,\ m \oplus F_K(r))$ with $r$ uniform on $\mathcal R$. For every adversary $\mathcal A$ making $Q$ encryption queries there is a PRF adversary $\mathcal B$ with about the same running time such that
$$\Pr[\mathcal A \text{ wins}] \le \frac12 + \mathrm{Adv}_{\mathrm{PRF}}(\mathcal B) + \frac{Q}{2|\mathcal R|}.$$
:::

::: {.proof}
The proof replaces the PRF by a random function at the cost of the PRF advantage, and then shows that the challenge is a one-time pad unless one bad event happens.

1. *Two hybrids.* Hybrid 0 is the real game, and hybrid 1 replaces $F_K$ by a random function $f$.
2. *The step costs the PRF advantage.* $\mathcal B$ runs the game for $\mathcal A$ with its own oracle in place of $F_K$ and outputs 1 when $\mathcal A$ wins. With oracle $F_K$ it simulates hybrid 0, and with $f$ hybrid 1, so the win probabilities differ by at most $\mathrm{Adv}_{\mathrm{PRF}}(\mathcal B)$.
3. *The bad event.* Let $E$ be the event that the challenge's $r^*$ equals one of the $Q$ values the oracle used. Each equals it with probability $1/|\mathcal R|$, so the union bound gives $\Pr[E] \le Q/|\mathcal R|$.
4. *Without it, a one-time pad.* Then $f(r^*)$ is uniform and independent of everything else $\mathcal A$ sees, so by @lem-crypto-bijection the challenge is uniform whatever $b$ is, and $\mathcal A$ wins with probability $\frac12$.
5. *Combine.* Bounding the win probability by 1 when $E$ happens,
$$\Pr[\text{win in hybrid 1}] \le \tfrac12\Pr[\text{not } E] + \Pr[E] = \tfrac12 + \tfrac12\Pr[E] \le \tfrac12 + \frac{Q}{2|\mathcal R|},$$
and adding the loss of step 2 gives the bound.
:::

In advantage units the bound is $2\,\mathrm{Adv}_{\mathrm{PRF}} + Q/|\mathcal R|$, and the term $Q/|\mathcal R|$ is real, not an artefact of the proof.

::: {#exm-crypto-cpa}
Against the deterministic scheme $m \oplus F_K(0)$, the attack of @prp-crypto-deterministic has measured advantage 1.000 over 20,000 games for each $b$. Against the randomized scheme with 8-bit nonces, $|\mathcal R| = 256$, the adversary asks for $Q = 16$ encryptions of $m_0$, and XOR-ing each reply with $m_0$ reveals its pad $F_K(r_i)$. If $r^*$ matches one of those nonces it reads the challenge, and otherwise it outputs 1. So it always outputs 1 when $b = 1$, and when $b = 0$ only without a match, which makes its advantage $\Pr[E]$. Since $r^*$ misses each nonce with probability $255/256$,
$$\Pr[E] = 1 - (255/256)^{16} = 0.0607.$$
The simulation measures 0.0605, and the bound $Q/|\mathcal R|$ is 0.0625. With 128-bit nonces the same attack has advantage below $16/2^{128} = 2^{-124}$.
:::

## The simulation paradigm {#sec-crypto-simulation}

Encryption hides everything, but many procedures must reveal *something*, such as a participant's own share of a secret, and nothing else. The simulation paradigm defines "nothing else" indirectly: whatever the participant sees, it could have produced by itself from what it is allowed to know (Lindell 2017).

::: {#def-crypto-simulation name="Revealing nothing beyond a leakage"}
A randomized procedure on inputs $x$ gives a participant a **view** $\mathrm{View}(x)$, everything it sees during the run. The participant is allowed to learn a function $L(x)$, its **leakage**. The procedure **reveals nothing beyond $L$** if an efficient **simulator** $S$ given only $L(x)$ reproduces the view's distribution:
$$S(L(x)) \equiv \mathrm{View}(x) \quad\text{for every input } x.$$ {#eq-crypto-simulation}
Replacing $\equiv$ by $\approx_s$ or $\approx_c$ gives statistical or computational security.
:::

The definition captures "nothing else" because anything computed from the view has the same distribution when computed from the simulator's output, which uses $L(x)$ alone. The simplest procedure it covers splits a number into random pieces that add up to it, in $\mathbb Z_p$, the integers $\{0, \dots, p - 1\}$ with arithmetic mod $p$.

::: {#def-crypto-sharing name="Additive secret sharing"}
To share a secret $x \in \mathbb Z_p$ among $h$ holders, draw the shares $z_1, \dots, z_{h-1}$ independently and uniformly from $\mathbb Z_p$, and set
$$z_h = x - (z_1 + \dots + z_{h-1}) \bmod p.$$
:::

Any $h - 1$ shares should look uniform whatever the secret is, because the missing share is what ties them to $x$.

::: {#thm-crypto-sharing name="Any h − 1 shares reveal nothing"}
For every secret $x$ and every coalition of $h - 1$ holders, the coalition's shares are independent and uniform on $\mathbb Z_p^{h-1}$. So a simulator that outputs $h - 1$ uniform elements of $\mathbb Z_p$, with no leakage, satisfies @eq-crypto-simulation with $\equiv$.
:::

::: {.proof}
1. *The randomness is uniform.* Each of the $p^{h-1}$ tuples $(z_1, \dots, z_{h-1})$ has probability $p^{-(h-1)}$.
2. *The coalition's shares determine the randomness.* Fix $x$ and a coalition that misses holder $j$. If $j = h$, the coalition holds $z_1, \dots, z_{h-1}$ themselves. If $j < h$, it holds $z_h$ and every other $z_i$, and the definition of $z_h$ gives back $z_j = x - z_h - \sum_{i \ne j, h} z_i$.
3. *So the map is a bijection.* The map from the randomness to the coalition's shares can be undone, so by @lem-crypto-bijection every tuple of shares has probability $p^{-(h-1)}$ for every $x$.
:::

::: {#exm-crypto-shares}
@tbl-crypto-shares makes step 3 concrete for $h = 2$ and $p = 5$, and the script checks the count for $h = 3$ exhaustively.

| First share $z_1$ | 0 | 1 | 2 | 3 | 4 |
| --- | --- | --- | --- | --- | --- |
| Second share $x - z_1 \bmod 5$ for $x = 2$ | 2 | 1 | 0 | 4 | 3 |
| Second share $x - z_1 \bmod 5$ for $x = 4$ | 4 | 3 | 2 | 1 | 0 |

: Two-holder additive sharing in $\mathbb Z_5$. Each row is a permutation of $\mathbb Z_5$, so the share is uniform for every secret. {#tbl-crypto-shares}

Without the reduction mod $p$ the proof fails. With a first share uniform on $\{0, \dots, R - 1\}$, the second share is uniform on the $R$ integers ending at $x$, so the ranges for secrets $x$ and $x'$ differ in $|x - x'|$ values and the TV distance is $|x - x'|/R$. @fig-crypto-shares compares the secrets 0 and 7. With $p = 11$ the empirical TV distance over 100,000 shares each is 0.0045, sampling noise around the exact 0, while over the integers with $R = 16$ it is 0.4423, against the exact $7/16 = 0.4375$.
:::

![Histograms of the second share for secrets $x = 0$ and $x = 7$. Left: reduced mod $p = 11$, both are uniform. Right: over the integers with a first share below 16, the support shifts with the secret.](figures/crypto_basics_shares.svg){#fig-crypto-shares}

## Learning with errors {#sec-crypto-lwe}

Every primitive so far is defined by what it hides rather than by how. Learning with errors (LWE), introduced by Regev (2005), is a specific hard problem with no known efficient quantum attack, and encryption built on it lets you add ciphertexts. It is linear algebra with noise, and this section shows that the noise is everything.

::: {#def-crypto-lwe name="Learning with errors"}
Fix a dimension $d$, a modulus $p$ and an error distribution $\chi$ on small integers. Draw a secret $s$ uniformly from $\mathbb Z_p^d$. An **LWE sample** is a uniform $a \in \mathbb Z_p^d$ together with the inner product $\langle a, s\rangle$ plus an error $e$ drawn from $\chi$:
$$\big(a,\ \langle a, s\rangle + e \bmod p\big).$$
**Search LWE** asks for $s$ from many samples. **Decision LWE** asks whether given pairs are LWE samples or uniform pairs.
:::

Solving search solves decision, since you can check whether the residuals of a candidate $s$ are small. Regev shows the converse for a prime $p$ polynomial in $d$, and Appendix @sec-apx-crypto-lwe derives it step by step. For suitably wide errors drawn from a rounded normal distribution, Regev (2009) proves LWE as hard as the hardest instances of well-studied geometric problems on lattices, and Peikert (2016) surveys later results. These stay cited, and they do not cover the uniform errors used in the toy scheme below.

::: {#exm-crypto-elimination}
Without noise, LWE is a linear system that Gaussian elimination solves. Take $d = 2$, $p = 7$, $s = (3, 5)$, $a_1 = (1, 2)$ and $a_2 = (3, 1)$, so the right-hand sides are $13 \equiv 6$ and $14 \equiv 0$. The second equation gives $s_2 = -3s_1 \equiv 4s_1$, and the first then gives $9s_1 \equiv 2s_1 = 6$, so $s = (3, 5)$. With the error $(1, 0)$ both right-hand sides are 0, and the same steps give $(0, 0)$, completely wrong, because elimination multiplies the error by the inverse matrix, which mod 7 sends $(1, 0)$ to $(4, 2)$. At $d = 16$ and $p = 4096$ with 32 samples, the script's elimination recovers all 16 coordinates of $s$ without noise, and 0 with errors in $\{-1, 0, 1\}$.
:::

## Regev encryption and its noise budget {#sec-crypto-regev}

The hardness of LWE becomes encryption once a message bit hides in a sample. Picture $\mathbb Z_p$ as a clock, and move the sample's second component halfway round if the bit is 1. Whoever knows $s$ subtracts $\langle a, s\rangle$ and sees whether the rest is near 0 or near halfway. This section proves the scheme correct and secure, and measures how many ciphertexts can be added before the errors pile up.

::: {#def-crypto-regev name="Regev's bit encryption"}
Take $p$ even and a key $s \in \mathbb Z_p^d$. To encrypt a bit $m$, draw a fresh uniform $a$ and an error $e$ from $\chi$, and output
$$\mathrm{Enc}_s(m) = \big(a,\ w = \langle a, s\rangle + e + m\cdot\tfrac p2 \bmod p\big).$$
To decrypt, compute $v = w - \langle a, s\rangle \bmod p$ and output 1 if $p/4 < v < 3p/4$.
:::

::: {#prp-crypto-regev-correct name="Correctness"}
Decryption returns $m$ whenever $|e| < p/4$.
:::

::: {.proof}
1. *Remove the inner product.* By the definition of $w$, $v = e + m\frac p2 \bmod p$.
2. *A 0 stays near 0.* If $m = 0$, then $v \equiv e$ lies in $[0, p/4)$ or, for negative $e$, in $(3p/4, p)$, and the output is 0.
3. *A 1 stays near halfway.* If $m = 1$, then $v = e + p/2$ lies in $(p/4, 3p/4)$, and the output is 1.
:::

Without $s$, an LWE sample looks uniform, so the bit is added to something that looks uniform, as in a one-time pad.

::: {#prp-crypto-regev-secure name="Security under decision LWE"}
If an IND-CPA adversary $\mathcal A$ against Regev's scheme has advantage $\nu$, there is a decision-LWE distinguisher $\mathcal B$ with about the same running time and advantage $\nu/2$.
:::

::: {.proof}
1. *The distinguisher.* $\mathcal B$ runs the IND-CPA game for $\mathcal A$, builds each ciphertext from one of its given pairs by adding $m\cdot\frac p2$ to the second component, and outputs 1 when $\mathcal A$ wins.
2. *On LWE samples.* The ciphertexts are real, so by @lem-crypto-win $\mathcal B$ outputs 1 with probability $\frac12 + \frac\nu2$.
3. *On uniform pairs.* Adding a fixed value is a bijection of $\mathbb Z_p$, so by @lem-crypto-bijection each second component is uniform whatever $m$ is, and $\mathcal A$ wins with probability $\frac12$. The difference is $\frac\nu2$.
:::

A ciphertext is linear in $a$, the error and the encoded bit, so adding ciphertexts adds all three, and two half-turns make a full turn.

::: {#prp-crypto-regev-sum name="Adding ciphertexts"}
The componentwise sum of encryptions of $m_1, \dots, m_k$ under one key is an encryption of $m_1 \oplus \dots \oplus m_k$ with error $e_1 + \dots + e_k$. If every $|e_i| \le B$, it decrypts correctly whenever
$$k < \frac{p}{4B}.$$ {#eq-crypto-regev-budget}
:::

::: {.proof}
1. *Linearity.* By linearity of the inner product,
$$\sum_i w_i - \Big\langle\sum_i a_i, s\Big\rangle = \sum_i e_i + \Big(\sum_i m_i\Big)\frac p2 \bmod p.$$
2. *Only the parity survives.* Because $p$ is even, $2\cdot\frac p2 \equiv 0$, so only $m_1 \oplus \dots \oplus m_k$ remains in the last term.
3. *Bound the error.* By the triangle inequality the summed error is at most $kB$ in absolute value, and by @prp-crypto-regev-correct $kB < p/4$ suffices.
:::

@eq-crypto-regev-budget assumes that every error is as large as possible, with the same sign. Independent errors with mean zero mostly cancel, so $k$ errors of standard deviation $\sigma$ sum to a standard deviation of $\sigma\sqrt k$. For $\chi$ uniform on $\{-B, \dots, B\}$ the variance is $B(B+1)/3$, and by the central limit theorem, with $Z$ a standard normal variable,
$$\Pr[\text{failure after } k \text{ additions}] \approx \Pr\Big[\big|Z \sqrt{kB(B+1)/3}\big| > p/4 \text{ after wrapping mod } p\Big].$$

::: {#exm-crypto-regev}
Take $d = 3$, $p = 4096$ and $s = (5, 1000, 3000)$. Encrypting $m = 1$ with $a_1 = (17, 2, 1)$ and $e_1 = -7$ gives $\langle a_1, s\rangle = 5085 \equiv 989$ and $w_1 = 989 - 7 + 2048 = 3030$, which decrypts to $v = 2041$, near 2048, so to 1. Encrypting $m = 1$ with $a_2 = (1, 1, 1)$ and $e_2 = 12$ gives $w_2 \equiv 1969$. The sum has $a_1 + a_2 = (18, 3, 2)$, whose inner product with $s$ is $9090 \equiv 898$, and $w_1 + w_2 \equiv 903$, so decryption finds $v = 5 = e_1 + e_2$ and outputs $0 = 1 \oplus 1$.

@tbl-crypto-regev and @fig-crypto-regev compare the worst case with the typical case for $p = 4096$ and $B = 32$, where @eq-crypto-regev-budget guarantees $k < 32$. The simulation adds $k$ toy ciphertexts of random bits, with $d = 4$, 20,000 times.

| Ciphertexts added $k$ | Empirical failure rate | Normal approximation |
| --- | --- | --- |
| 1 | 0.0000 | 0.0000 |
| 31 | 0.0000 | 0.0000 |
| 256 | 0.0007 | 0.0006 |
| 1,024 | 0.0868 | 0.0881 |
| 4,096 | 0.3814 | 0.3833 |

: Decryption failure rate of a sum of $k$ Regev ciphertexts, $p = 4096$, errors uniform on $\{-32, \dots, 32\}$ (seed 0, 20,000 trials). {#tbl-crypto-regev}

At $k = 1024$ the summed error has standard deviation $\sqrt{1024 \times 32 \times 33/3} \approx 600$, so $p/4 = 1024$ is about 1.7 standard deviations, exceeded with probability about 0.088. Failures become measurable only around $k = 256$, eight times the guaranteed 32, but the budget is finite, and the failure rate tends to $\frac12$ as the error spreads evenly round the clock.
:::

![Decryption failure rate of a sum of $k$ Regev ciphertexts ($p = 4096$, errors uniform on $\{-32, \dots, 32\}$): simulation against the normal approximation. The dashed line is the worst-case guarantee $k < p/(4B) = 32$.](figures/crypto_basics_regev_noise.svg){#fig-crypto-regev}

## Code {#sec-crypto-code}

The script [code/crypto_basics.py](../../code/part01-prerequisites/crypto_basics.py) computes every number quoted in this chapter and draws its figures. From the repository root:

```bash
uv run python code/part01-prerequisites/crypto_basics.py
```

| Experiment | Result |
| --- | --- |
| Negligible functions | $2^{-\sqrt\lambda} < \lambda^{-10}$ from $\lambda = 20{,}520$ |
| Brute force, $10^{18}$ keys/s for a year | Success $9.3\times10^{-14}$ |
| Birthday, $N = 2^{16}$, $Q = 256$ | Exact 0.3927; simulated 0.3920; union bound 0.4980; 50% at $Q = 301$ |
| Biased coin, @tbl-crypto-coin-hybrid | At $Q = 10{,}000$: majority 0.481, best 0.683, hybrid 1.000 |
| IND-CPA, deterministic scheme | Advantage 1.000 |
| IND-CPA, PRF scheme, $Q = 16$, 8-bit nonce | Simulated 0.0605; exact 0.0607; bound 0.0625 |
| Secret shares, $x = 0$ against $x = 7$ | Mod 11: TV 0.0045. Integers, $R = 16$: TV 0.4423 (exact 0.4375) |
| Gaussian elimination, $d = 16$, $p = 4096$ | 16/16 coordinates without noise; 0/16 with noise |

: Output of the chapter's script (seed 0). {#tbl-crypto-code}

[tests/test_crypto_basics.py](../../tests/test_crypto_basics.py) checks that the output still matches this table.

## Summary {#sec-crypto-summary}

| Result | Statement | Use |
| --- | --- | --- |
| @lem-crypto-win | Win probability $\frac12 + \frac12\mathrm{Adv}$, at most $\frac12 + \frac12\mathrm{TV}$ | Scoring every game |
| @prp-crypto-closure | Negligible survives $\times\,\mathrm{poly}$ | Repeated attacks stay negligible |
| @thm-crypto-hybrid | $Q$ samples cost a factor $Q$ | Often loose by $\sqrt Q$ |
| @def-crypto-concrete | $\kappa$ bits: advantage $t/2^\kappa$ | Loose reductions cost bits |
| @thm-crypto-ggm | GGM is a PRF, loss $nQ$ | Many values from one key |
| @thm-crypto-birthday | Collisions after $1.18\sqrt N$ draws | $\kappa$-bit nonces collide after $2^{\kappa/2}$ |
| @thm-crypto-switching | Permutation versus function: $Q^2/(2\lvert\mathcal X\rvert)$ | Data per cipher key |
| @prp-crypto-key-length | Perfect secrecy needs $\lvert\mathcal K\rvert \ge \lvert\mathcal M\rvert$ | A fresh key per message |
| @thm-crypto-prf-cpa | Fresh nonces cost $Q/\lvert\mathcal R\rvert$ | Randomize encryption |
| @thm-crypto-sharing | Any $h - 1$ shares are uniform | Reduce mod $p$ |
| @prp-crypto-regev-sum | Correct if $kB < p/4$ | Noise is a budget |

: The results of this chapter, and where each is used. {#tbl-crypto-summary}

## References {#sec-crypto-references}

- Katz & Lindell, *Introduction to Modern Cryptography* (3rd ed., 2020), ch. 2–3: perfect secrecy, computational security, PRGs, PRFs and CPA security, with complete proofs at the level of this chapter
- Boneh & Shoup, *A Graduate Course in Applied Cryptography*, ch. 2–5: semantic security, PRGs and the hybrid argument, block ciphers, PRFs, the switching lemma and the GGM tree, CPA security; written in the concrete-security style
- Goldreich, *Foundations of Cryptography*, Vol. 1 (2001), ch. 3: pseudorandomness in depth, including the GGM proof (§3.6)
- Shannon, "Communication Theory of Secrecy Systems" (1949): perfect secrecy and the key-length bound
- Goldwasser & Micali, "Probabilistic Encryption" (JCSS 1984): semantic security and the need for randomized encryption
- Goldreich, Goldwasser & Micali, "How to Construct Random Functions" (JACM 1986): the GGM construction
- Håstad, Impagliazzo, Levin & Luby, "A Pseudorandom Generator from any One-way Function" (SIAM J. Computing 1999)
- Bellare & Rogaway, "The Security of Triple Encryption and a Framework for Code-Based Game-Playing Proofs" (EUROCRYPT 2006): game-playing proofs and a careful switching lemma
- Lindell, "How to Simulate It — A Tutorial on the Simulation Proof Technique" (2017): the simulation paradigm, step by step
- Regev, "On Lattices, Learning with Errors, Random Linear Codes, and Cryptography" (STOC 2005; JACM 2009): LWE, its quantum hardness reduction, and the original encryption scheme
- Peikert, "A Decade of Lattice Cryptography" (2016): survey of LWE, its variants and reductions
- Lyubashevsky, Peikert & Regev, "On Ideal Lattices and Learning with Errors over Rings" (EUROCRYPT 2010): ring-LWE

**Practice.** Exercises for this chapter, each with a worked solution, are in [the exercise book](../../book-exercises/part01-prerequisites/08-crypto-basics.md).
