# Exercises: Cryptography basics

Exercises for [Cryptography basics](../../book-full/part01-prerequisites/08-crypto-basics.md), each with a worked solution. Try each exercise before opening its solution. The numbers below, from the total variation (TV) distances of the coin distinguisher to the toy pseudorandom function (PRF) and the learning with errors (LWE) noise budget, come from `uv run python code/part01-prerequisites/solutions_crypto_basics.py` (seed 0).

## Which are negligible? {#sec-sol-crypto-basics-1}

> **Exercise 1.** **Which are negligible?** Decide for $\lambda^{-\log_2\log_2\lambda}$, $2^{-(\log_2\lambda)^2}$, $1/\lambda^{100}$ and $\mu(\lambda) = 2^{-\lambda}$ for even and $\lambda^{-2}$ for odd $\lambda$. *Check: yes, yes, no, no.* *(Negligible functions)*

::: {.callout-tip collapse="true" title="Solution"}
Each case compares $\mu(\lambda)$ with $\lambda^{-c}$, as @eq-crypto-negligible asks: is $\mu(\lambda) < \lambda^{-c}$ for all large $\lambda$, for every $c > 0$?

**$\lambda^{-\log_2\log_2\lambda}$: yes.**

1. Both sides are powers of $\lambda$. For $\lambda > 1$, $\lambda^{-a} < \lambda^{-c}$ exactly when $a > c$.
2. So the condition is $\log_2\log_2\lambda > c$, that is $\lambda > 2^{2^c}$.
3. For every $c$ there is such a $\lambda_0 = 2^{2^c} + 1$, so the function is negligible.

It is negligible only barely. Against $\lambda^{-10}$ it needs $\lambda > 2^{1024}$, and at $\lambda = 128$ it equals $128^{-\log_2 7} = 2^{-19.65}$, larger than the non-negligible $\lambda^{-10} = 2^{-70}$.

**$2^{-(\log_2\lambda)^2}$: yes.** Write $2^{-(\log_2\lambda)^2} = (2^{\log_2\lambda})^{-\log_2\lambda} = \lambda^{-\log_2\lambda}$. This is example 2 of the chapter: it is below $\lambda^{-c}$ for all $\lambda > 2^c$.

**$1/\lambda^{100}$: no.** Take $c = 101$. Then $\lambda^{-100} < \lambda^{-101}$ would need $\lambda < 1$, which never holds for $\lambda \ge 1$. One exponent $c$ that fails is enough.

**$2^{-\lambda}$ for even, $\lambda^{-2}$ for odd $\lambda$: no.**

1. Take $c = 3$. The definition needs $\mu(\lambda) < \lambda^{-3}$ for *every* $\lambda \ge \lambda_0$.
2. For every odd $\lambda \ge 1$, $\mu(\lambda) = \lambda^{-2} \ge \lambda^{-3}$.
3. Odd numbers exceed every $\lambda_0$, so no $\lambda_0$ works.

*A function is negligible only if it is small for all large $\lambda$, not just for most of them: one bad subsequence is enough for an adversary to win on those parameters.*
:::

## Hybrids for TV {#sec-sol-crypto-basics-2}

> **Exercise 2.** **Hybrids for TV.** Prove $\mathrm{TV}(P^{\otimes m}, Q^{\otimes m}) \le m\,\mathrm{TV}(P, Q)$ directly with the chain $H_i$ and Fact 3. Then show that for two coins the true value grows like $\sqrt m$ when $\mathrm{TV}(P, Q)$ is small. *(Hybrid argument)*

::: {.callout-tip collapse="true" title="Solution"}
**The bound.** $\mathrm{TV}$ is the TV distance of Fact 2.

1. **Hybrids.** Let $H_i = (Q_1, \dots, Q_i, P_{i+1}, \dots, P_m)$: $m$ independent entries, the first $i$ drawn from $Q$ and the rest from $P$. Then $H_0 = P^{\otimes m}$ and $H_m = Q^{\otimes m}$.
2. **Fix a distinguisher.** Let $A$ be any distinguisher on $m$-tuples, randomized or not, with no limit on its running time.
3. **One step changes one entry.** $H_{i-1}$ and $H_i$ differ only in entry $i$: $P$ in $H_{i-1}$, $Q$ in $H_i$. The other $m - 1$ entries have the same distribution in both, and are independent of entry $i$. Condition on their values $w$. Then $A(\dots)$ becomes a distinguisher $A_w$ that sees only entry $i$, so by Fact 2
$$\big|\Pr[A(H_{i-1}) = 1 \mid w] - \Pr[A(H_i) = 1 \mid w]\big| = \mathrm{Adv}_{A_w}(P, Q) \le \mathrm{TV}(P, Q).$$
4. **Average over $w$.** The two conditional probabilities are averaged with the same weights in both hybrids, so the triangle inequality gives $\mathrm{Adv}_A(H_{i-1}, H_i) \le \mathrm{TV}(P, Q)$.
5. **Chain.** By Fact 3, $\mathrm{Adv}_A(H_0, H_m) \le \sum_{i=1}^m\mathrm{Adv}_A(H_{i-1}, H_i) \le m\,\mathrm{TV}(P, Q)$.
6. **Take the best $A$.** By Fact 2 the best unbounded distinguisher attains $\mathrm{TV}(H_0, H_m)$. Step 5 holds for it too, so $\mathrm{TV}(P^{\otimes m}, Q^{\otimes m}) \le m\,\mathrm{TV}(P, Q)$.

**The true growth for two coins.** Let $P$ be a coin with heads probability $\frac12 + \delta$ and $Q$ a fair coin, so $\mathrm{TV}(P, Q) = \delta$.

1. **Only the count matters.** A sequence $\omega$ with $k$ heads has $P^{\otimes m}(\omega) = (\frac12 + \delta)^k(\frac12 - \delta)^{m - k}$ and $Q^{\otimes m}(\omega) = 2^{-m}$, which depend on $\omega$ only through $k$. In $\mathrm{TV} = \frac12\sum_\omega|P^{\otimes m}(\omega) - Q^{\otimes m}(\omega)|$ (@eq-tv), group the $\binom mk$ sequences with $k$ heads. Each group contributes $\frac12|\mathrm{Bin}(m, \frac12 + \delta)(k) - \mathrm{Bin}(m, \frac12)(k)|$, so the TV distance between the tuples equals the TV distance between the two head counts.
2. **Normal approximation.** The biased count has mean $\frac m2 + m\delta$ and variance $m(\frac14 - \delta^2) \approx \frac m4$. The fair count has mean $\frac m2$ and variance $\frac m4$. For large $m$ both are close to normal with standard deviation $\sigma = \sqrt m/2$.
3. **TV between two normals.** In [Divergences](../../book-full/part01-prerequisites/03-divergences.md), two normals with the same $\sigma$ and means $s$ apart have $\mathrm{TV} = 2\Phi(\mu/2) - 1$ with $\mu = s/\sigma$. Here $s = m\delta$, so $\mu/2 = m\delta/\sqrt m = \delta\sqrt m$ and
$$\mathrm{TV}(P^{\otimes m}, Q^{\otimes m}) \approx 2\Phi(\delta\sqrt m) - 1.$$
4. **Small argument.** Near 0, $\Phi(x) \approx \frac12 + x\varphi(0)$ with $\varphi(0) = 1/\sqrt{2\pi}$. So while $\delta\sqrt m$ is small, $\mathrm{TV} \approx \sqrt{2/\pi}\,\delta\sqrt m \approx 0.80\,\delta\sqrt m$: it grows like $\sqrt m$, not like $m$.

@tbl-sol-crypto-basics-tv compares the exact TV distance with both approximations and with the hybrid bound, for $\delta = 0.01$.

| Flips $m$ | Exact TV | $2\Phi(\delta\sqrt m) - 1$ | $\sqrt{2/\pi}\,\delta\sqrt m$ | Hybrid bound $\min(1, m\delta)$ |
| --- | --- | --- | --- | --- |
| 1 | 0.0100 | 0.0080 | 0.0080 | 0.0100 |
| 100 | 0.0799 | 0.0797 | 0.0798 | 1.0000 |
| 400 | 0.1584 | 0.1585 | 0.1596 | 1.0000 |
| 2,500 | 0.3830 | 0.3829 | 0.3989 | 1.0000 |
| 10,000 | 0.6827 | 0.6827 | 0.7979 | 1.0000 |

: TV distance between $m$ flips of a $0.51$-coin and $m$ fair flips: exact, the normal approximation, its small-argument form, and the hybrid bound. {#tbl-sol-crypto-basics-tv}

Each fourfold increase in $m$ (100 to 400) doubles the TV distance, as $\sqrt m$ predicts. The normal approximation is poor only at $m = 1$, where the count is a single coin flip and the bound $m\delta$ is exact. The linear form overshoots once $\delta\sqrt m$ is no longer small, because $2\Phi(x) - 1$ bends over towards 1.

*The hybrid bound is tight for one sample and loose by a factor of order $\sqrt m$ afterwards, because the per-step losses in Fact 3 do not add up in the same direction.*
:::

## The best threshold {#sec-sol-crypto-basics-3}

> **Exercise 3.** **The best threshold.** For $m$ flips of a $(\frac12 + \delta)$-coin against a fair coin, show that the likelihood ratio is increasing in the number of heads, so the best test is a threshold, and find where the two binomial probabilities cross. *Check: about $m/2 + m\delta/2$; for $m = 10{,}000$ and $\delta = 0.01$ the best advantage is 0.683.* *(Distinguishers and TV)*

::: {.callout-tip collapse="true" title="Solution"}
1. **Likelihood ratio.** For a sequence with $k$ heads, divide the two probabilities from Exercise 2:
$$L(k) = \frac{(\frac12 + \delta)^k(\frac12 - \delta)^{m - k}}{2^{-m}} = (1 + 2\delta)^k(1 - 2\delta)^{m - k}.$$
2. **It increases in $k$.** One more head multiplies it by $L(k + 1)/L(k) = (1 + 2\delta)/(1 - 2\delta) > 1$ for $\delta > 0$.
3. **The best test is a threshold.** By Fact 2, the best distinguisher outputs 1 exactly where the biased coin's probability is larger, that is where $L(k) > 1$. Since $L$ increases, this set is $\{k > k^*\}$ for the $k^*$ with $L(k^*) = 1$.
4. **Where the binomials cross.** The binomial probabilities are $\binom mk$ times the sequence probabilities. The factor $\binom mk$ is the same for both, so they cross at the same $k^*$. Take logarithms of $L(k^*) = 1$:
$$k^*\ln(1 + 2\delta) + (m - k^*)\ln(1 - 2\delta) = 0 \quad\Longrightarrow\quad k^* = m\,\frac{-\ln(1 - 2\delta)}{\ln(1 + 2\delta) - \ln(1 - 2\delta)}.$$
5. **Expand for small $\delta$.** With $u = 2\delta$, $-\ln(1 - u) = u + \frac{u^2}2 + \frac{u^3}3 + \dots$ and $\ln(1 + u) - \ln(1 - u) = 2u + \frac{2u^3}3 + \dots$. The ratio is
$$\frac{1 + \frac u2 + \frac{u^2}3}{2(1 + \frac{u^2}3)} = \frac12 + \frac u4 + O(u^3) = \frac12 + \frac\delta2 + O(\delta^3).$$
So $k^* = \frac m2 + \frac{m\delta}2 + O(m\delta^3)$: halfway between the fair mean $\frac m2$ and the biased mean $\frac m2 + m\delta$.
6. **The numbers.** For $m = 10{,}000$ and $\delta = 0.01$, the exact crossing is $k^* = 5050.0033$, against $m/2 + m\delta/2 = 5050$. The test "output 1 iff $k > k^*$" has advantage 0.6827, equal to the exact TV distance 0.6827, which rounds to the check value 0.683.
7. **Normal check.** The threshold sits $\frac{m\delta}2 = 50 = 1\cdot\sigma$ from each mean ($\sigma = \sqrt m/2 = 50$). So the advantage is about $\Phi(1) - (1 - \Phi(1)) = 2\Phi(1) - 1 = 0.6827$.

Majority vote is the threshold at $\frac m2$ instead of $k^*$. Its exact advantage is 0.4807, which matches the 0.481 simulated in @tbl-crypto-coin-hybrid. The fair count lands above $\frac m2$ almost half the time, so moving the threshold up by one standard deviation gains a lot.

*The best distinguisher is a likelihood-ratio test, and for coins that test is a threshold on the head count, placed midway between the two means.*
:::

## Birthday for 32 bits {#sec-sol-crypto-basics-4}

> **Exercise 4.** **Birthday for 32 bits.** How many random 32-bit identifiers can you draw before a collision has probability $\frac12$? *Check: $\sqrt{2\cdot2^{32}\ln 2} \approx 77{,}000$.* *(Birthday bound)*

::: {.callout-tip collapse="true" title="Solution"}
1. **Set up.** $N = 2^{32}$ possible identifiers, $q$ independent uniform draws.
2. **Use the lower bound as an approximation.** By @eq-crypto-birthday, $\Pr[\text{collision}] \ge 1 - e^{-q(q-1)/(2N)}$. For $q \ll N$ this is close to exact: the only step that lost anything was $1 - i/N \le e^{-i/N}$, and $i/N$ is tiny here.
3. **Solve for $\frac12$.** $1 - e^{-q^2/(2N)} = \frac12$ gives $q^2/(2N) = \ln 2$, so
$$q = \sqrt{2N\ln 2} = \sqrt{2\cdot 2^{32}\cdot 0.6931} = 77{,}162.7.$$
This is $1.18\sqrt N$ with $\sqrt N = 65{,}536$.
4. **Exact check.** The script evaluates the exact product $1 - \prod_{i<q}(1 - i/N)$ and finds that it first reaches $\frac12$ at $q = 77{,}164$.

So about 77,000 identifiers, in agreement with the check. A table of 100,000 records with random 32-bit keys is already past this point, which is why @tbl-crypto-birthday finds a collision probability of 0.688.

*Random $\kappa$-bit values collide after about $2^{\kappa/2}$ draws: 32-bit identifiers are safe only for tables of a few thousand rows.*
:::

## Two-time pad {#sec-sol-crypto-basics-5}

> **Exercise 5.** **Two-time pad.** Two messages of English text are encrypted with the same pad. Show what an eavesdropper learns, and why guessing a common word in one message reveals the other at the same position. *(Perfect secrecy)*

::: {.callout-tip collapse="true" title="Solution"}
**What the eavesdropper learns.**

1. It sees $c_1 = m_1 \oplus K$ and $c_2 = m_2 \oplus K$, byte by byte.
2. Combine them with $\oplus$. Since $K \oplus K = 0$, $c_1 \oplus c_2 = m_1 \oplus m_2$. The pad cancels.
3. That is all it learns. By Fact 4, $c_1$ alone is uniform whatever $m_1$ is. Given $c_1$, the second ciphertext is $c_2 = c_1 \oplus (m_1 \oplus m_2)$, fixed. So the pair is $(U, U \oplus (m_1 \oplus m_2))$ with $U$ uniform. A simulator that receives $L = m_1 \oplus m_2$ outputs exactly this distribution, so by @eq-crypto-simulation the pair reveals nothing beyond $m_1 \oplus m_2$.

But $m_1 \oplus m_2$ is a lot. Perfect secrecy needed a fresh uniform key per message. With the pad reused, the "key" of the second message is the first message, which is English text and far from uniform.

**Why a guessed word reveals the other message.** Suppose you guess that $m_1$ contains the word $w$ at position $j$. Combine $w$ with $c_1 \oplus c_2$ at that position:
$$(c_1 \oplus c_2)[j..] \oplus w = m_1[j..] \oplus m_2[j..] \oplus w = m_2[j..] \quad\text{if } m_1[j..] = w.$$
If the guess is right, you read the other message at the same position. If it is wrong, you get $m_2 \oplus m_1 \oplus w$, which is usually gibberish. You slide the guess along every position (**crib dragging**) and keep the positions where readable text comes out. The guess works in both directions: if $w$ is in $m_2$ instead, the same step reveals $m_1$.

The script does this with $m_1$ set to "meet me at the north gate" and $m_2$ to "the password is blue fish", both encrypted under one random pad, and the guess "the " (with its trailing space). Only three positions give lowercase letters and spaces; @tbl-sol-crypto-basics-crib lists them.

| Position $j$ | $(c_1 \oplus c_2)[j..] \oplus$ "the " | Explanation |
| --- | --- | --- |
| 0 | "meet" | $m_2$ starts with "the ", so this is $m_1$ |
| 11 | "d is" | $m_1$ has "the " at 11, so this is $m_2$ ("passwor**d is**") |
| 17 | "jih " | neither message has "the " here: a false alarm |

: Crib dragging the guess "the " across $c_1 \oplus c_2$ for two 25-byte messages under one pad (seed 0). {#tbl-sol-crypto-basics-crib}

From "d is" an attacker guesses "password is" and extends both messages a few letters at a time. The false alarm at 17 dies as soon as it is extended, because "jih" does not continue into English.

*Perfect secrecy holds for one message per key; reuse leaks $m_1 \oplus m_2$, and the redundancy of natural language turns that into both plaintexts.*
:::

## Toy GGM {#sec-sol-crypto-basics-6}

> **Exercise 6.** **Toy GGM.** Implement the GGM tree with SHA-256 as a length-doubling generator ($G_0(s) = \mathrm{SHA256}(s\,\|\,0)$, $G_1(s) = \mathrm{SHA256}(s\,\|\,1)$). How many generator calls does one evaluation on $n$-bit inputs cost, and what does the proof lose? *Check: $n$ calls; the loss factor is $nq$.* *(PRFs)*

::: {.callout-tip collapse="true" title="Solution"}
**Implementation.** A seed is 32 bytes and each half $G_b$ returns 32 bytes, so $G = G_0\,\|\,G_1$ maps 32 bytes to 64: a length-doubling pseudorandom generator (PRG), in the toy model where the hash is treated as one. The tree walk is three lines:

```python
def ggm_eval(gen: CountingGenerator, key: bytes, bits: list[int]) -> bytes:
    node = key
    for b in bits:
        node = gen.half(node, b)  # G_b(node) = SHA256(node || b), counted
    return node
```

**Cost.** The walk visits one node per level and computes only the half $G_{x_i}$ it follows. So one evaluation costs $n$ generator calls (one hash each). The script counts them: on $n = 8$ bits, each evaluation makes 8 calls. Evaluating all 256 inputs gives 256 distinct outputs, with a fraction 0.5001 of 1 bits, and re-evaluating an input gives the same output, as a function must.

**What the proof loses.** Follow the proof sketch in the chapter:

1. There are $n + 1$ hybrids $H_0, \dots, H_n$, where $H_j$ makes every node at depth $j$ uniform. $H_0$ is $F_k$ and $H_n$ is a random function.
2. A $q$-query adversary touches at most $q$ nodes at each depth. Telling $H_{j-1}$ from $H_j$ is telling $q$ outputs of $G$ from $q$ uniform strings, which by the hybrid lemma costs at most $q\cdot\mathrm{Adv}_{\mathrm{PRG}}$.
3. Summing the $n$ steps with Fact 3 gives $\mathrm{Adv}_{\mathrm{PRF}} \le nq\cdot\mathrm{Adv}_{\mathrm{PRG}}$: a loss factor $nq$.

Concretely, with $n = 128$ input bits and $q = 2^{40}$ queries, $nq = 2^7\cdot2^{40} = 2^{47}$. For a PRF advantage of at most $2^{-64}$ the generator must have advantage at most $2^{-111}$: the reduction costs 47 bits of security.

*The GGM tree turns any length-doubling generator into a PRF at $n$ generator calls per input, but the reduction is loose by $nq$, so the generator needs a large security margin.*
:::

## Threshold sharing {#sec-sol-crypto-basics-7}

> **Exercise 7.** **Threshold sharing.** In Shamir's 2-out-of-3 scheme over prime $\mathbb Z_p$, the dealer picks a uniform $c$ and gives holder $i$ the value $x + ci$. Show by counting that any single share is uniform, and that any two shares determine $x$. *(Simulation by counting)*

::: {.callout-tip collapse="true" title="Solution"}
Holders are numbered $i = 1, 2, 3$, and $p > 3$ so that these are distinct and nonzero mod $p$.

**Any single share is uniform.**

1. Fix the secret $x$ and a holder $i$. The share $s_i = x + ci \bmod p$ is a function of the randomness $c$ alone.
2. Because $p$ is prime and $i \ne 0 \bmod p$, $i$ has an inverse mod $p$. For any value $v$, the equation $x + ci = v$ has exactly one solution, $c = (v - x)\,i^{-1}$.
3. So $c \mapsto x + ci$ is a bijection of $\mathbb Z_p$, and by Fact 4 the share is uniform: each value has probability $1/p$, whatever $x$ is.
4. The simulator for one holder, with no leakage, outputs a uniform element of $\mathbb Z_p$, and @eq-crypto-simulation holds with $\equiv$.

@tbl-sol-crypto-basics-shamir walks the count for $p = 5$ and $x = 2$: each holder's column is a permutation of $\mathbb Z_5$.

| $c$ | $s_1 = 2 + c$ | $s_2 = 2 + 2c$ | $s_3 = 2 + 3c$ |
| --- | --- | --- | --- |
| 0 | 2 | 2 | 2 |
| 1 | 3 | 4 | 0 |
| 2 | 4 | 1 | 3 |
| 3 | 0 | 3 | 1 |
| 4 | 1 | 0 | 4 |

: Shamir 2-out-of-3 shares of $x = 2$ in $\mathbb Z_5$. Each column contains every value once, so each single share is uniform. {#tbl-sol-crypto-basics-shamir}

**Any two shares determine $x$.**

1. Holders $i \ne j$ have $s_i = x + ci$ and $s_j = x + cj$. Subtract: $s_i - s_j = c(i - j)$.
2. $i - j \ne 0 \bmod p$, so $c = (s_i - s_j)(i - j)^{-1}$.
3. Then $x = s_i - ci$.

With $p = 5$, take holders 1 and 3 with $s_1 = 4$ and $s_3 = 3$ (the row $c = 2$). Then $s_3 - s_1 = -1 \equiv 4$ and $(3 - 1)^{-1} = 2^{-1} = 3$ (since $2\cdot3 = 6 \equiv 1$), so $c = 4\cdot 3 = 12 \equiv 2$ and $x = s_1 - c = 4 - 2 = 2$. The script checks exhaustively for $p = 5$ and $p = 7$ that every single share is uniform for every $x$ and that every pair of shares gives back $x$ for every $x$ and $c$.

Geometrically, the shares are points on the line $y = x + ct$ at $t = 1, 2, 3$, and $x$ is its value at $t = 0$. One point lies on exactly one line through each possible intercept, so it says nothing. Two points fix the line.

*Counting shows perfect security for one share; two shares are not uniform on $\mathbb Z_p^2$, since they lie on a line through the secret.*
:::

## Regev's budget {#sec-sol-crypto-basics-8}

> **Exercise 8.** **Regev's budget.** Using the normal approximation, find the $k$ at which the failure rate for $q = 4096$, $B = 32$ reaches 1%. Then change the script's error distribution to a rounded normal with $\sigma = 3.2$ and redo it. *Check for the first part: $1024^2/(2.576^2\cdot352) \approx 450$.* *(Noise growth)*

::: {.callout-tip collapse="true" title="Solution"}
**Uniform errors on $\{-32, \dots, 32\}$.**

1. **Variance of one error.** $B(B + 1)/3 = 32\cdot33/3 = 352$, as in the chapter.
2. **Variance of the sum.** The $k$ errors are independent, so their sum has variance $352k$ and is close to normal.
3. **Failure event.** Decryption fails when the summed error leaves $(-q/4, q/4) = (-1024, 1024)$. Ignoring the wrap-around mod $q$, which only matters near failure rate $\frac12$,
$$\Pr[\text{failure}] \approx \Pr\big[|Z| > 1024/\sqrt{352k}\big], \qquad Z \sim \mathcal N(0, 1).$$
4. **Set it to 1%.** $\Pr[|Z| > z] = 0.01$ at $z = 2.576$. So $1024/\sqrt{352k} = 2.576$, and
$$k = \frac{1024^2}{2.576^2\cdot352} = 449.0.$$
The wrapped approximation used by the chapter's script, solved over the integers, gives the same $k = 449$. Adding 449 real toy ciphertexts 20,000 times, the script measures a failure rate of 0.0109, within sampling noise (about 0.0007) of 1%.

**Rounded normal errors with $\sigma = 3.2$.** In the solutions script the error sampler is a parameter. The new one rounds $\mathcal N(0, 3.2^2)$ to the nearest integer.

1. **Variance.** Rounding adds roughly $\frac1{12}$ to the variance. Summing $j^2\Pr[\mathrm{round}(X) = j]$ exactly gives $10.3233$, against $3.2^2 + \frac1{12} = 10.3233$ to this precision.
2. **Same formula.** $k = 1024^2/(2.576^2\cdot10.3233) = 15{,}309$, and the wrapped approximation gives the same integer.
3. **Simulation.** Adding 15,309 ciphertexts, 20,000 times, gives a failure rate of 0.0111.

The smaller error buys 34 times as many additions, which is the ratio of the variances, $352/10.3233$. The budget scales like $(q/\sigma)^2$: halving the error standard deviation buys four times as many additions.

There is no worst-case analogue of @eq-crypto-regev-budget for a rounded normal, because its support is unbounded; one bounds it by a tail cut instead, such as $|e| \le 6\sigma$, at a tiny failure probability.

*Under random errors the noise budget is set by the variance, not the bound: $k \approx (q/4)^2/(z^2\sigma^2)$ additions for failure rate $\Pr[|Z| > z]$.*
:::

## Odd modulus {#sec-sol-crypto-basics-9}

> **Exercise 9.** **Odd modulus.** Modify the addition experiment to use a prime $q$ and the encoding $\lfloor q/2\rfloor$. Measure the extra drift and compare with the prediction of about $-k/4$. *(LWE arithmetic)*

::: {.callout-tip collapse="true" title="Solution"}
**Prediction.**

1. **Write $q = 2h + 1$**, so the encoding is $h = \lfloor q/2\rfloor$ and $2h = q - 1 \equiv -1$.
2. **The sum.** As in the derivation of @eq-crypto-regev-budget, the summed ciphertext decrypts to $d = \sum_i e_i + wh \bmod q$, where $w = \sum_i m_i$ is the number of 1-bits.
3. **Split $w$.** $w = (w \bmod 2) + 2\lfloor w/2\rfloor$, so
$$wh = (w \bmod 2)\,h + \lfloor w/2\rfloor\cdot 2h \equiv (w \bmod 2)\,h - \lfloor w/2\rfloor.$$
The first term is the correct encoding of the parity. The second is an extra error of $-\lfloor w/2\rfloor$: every pair of 1-bits adds $-1$.
4. **Its mean.** For random bits, $w \sim \mathrm{Bin}(k, \frac12)$, with $\mathbb E w = k/2$. Also $\Pr[w \text{ even}] - \Pr[w \text{ odd}] = \mathbb E(-1)^w = (\frac12 - \frac12)^k = 0$ for $k \ge 1$, so $\Pr[w \text{ odd}] = \frac12$. Hence
$$\mathbb E\lfloor w/2\rfloor = \frac{\mathbb E w - \Pr[w \text{ odd}]}{2} = \frac{k - 1}4, \qquad \text{drift} = -\frac{k - 1}4.$$
This is the "about $-k/4$" of the exercise, exact up to the $\frac14$.

**Measurement.** The script repeats the addition experiment with the prime $q = 4093$ (the largest prime below 4096), errors uniform on $\{-32, \dots, 32\}$, $n = 4$ and 20,000 trials. It computes the drift as $d - \sum_i e_i - (w \bmod 2)h$, centred in $(-q/2, q/2]$, and also reruns $q = 4096$ for comparison. The normal approximation for the odd modulus shifts the mean of the summed error by the predicted drift. @tbl-sol-crypto-basics-odd has the results.

| $k$ | Drift, $q = 4093$ | Predicted $-(k-1)/4$ | Failure, $q = 4093$ | Normal approximation with drift | Failure, $q = 4096$ |
| --- | --- | --- | --- | --- | --- |
| 1 | 0.00 | 0.00 | 0.0000 | 0.0000 | 0.0000 |
| 31 | $-7.51$ | $-7.50$ | 0.0000 | 0.0000 | 0.0000 |
| 256 | $-63.79$ | $-63.75$ | 0.0010 | 0.0008 | 0.0003 |
| 1,024 | $-255.81$ | $-255.75$ | 0.1167 | 0.1171 | 0.0873 |
| 4,096 | $-1023.82$ | $-1023.75$ | 0.5007 | 0.5001 | 0.3780 |

: Sums of $k$ Regev ciphertexts with the odd modulus $q = 4093$ and encoding $\lfloor q/2\rfloor$, against the even modulus $q = 4096$ (errors uniform on $\{-32, \dots, 32\}$, seed 0, 20,000 trials). {#tbl-sol-crypto-basics-odd}

- **The drift matches the prediction** to within 0.1 at every $k$. With the even modulus it is exactly 0.
- **It costs correctness.** The drift grows like $k$, the random part only like $\sqrt k$. At $k = 1024$ the drift ($-256$) is already 0.43 standard deviations of the random part ($\sqrt{352\cdot1024} = 600$), and the failure rate rises from 0.0873 to 0.1167. At $k = 4096$ the drift reaches $-q/4$ by itself, and decryption is a coin flip.

*With an odd modulus the encoding is not exactly additive: each pair of 1-bits leaks $-1$ into the noise, a deterministic drift of $-(k-1)/4$ that eventually dominates the random noise.*
:::
