# Cryptography basics

## Why this matters

Some computations must hide a key, a message, or a value split among several holders. What does it mean that an observer learns nothing it should not? Against unlimited time nothing can be hidden, since an attacker could try every key, so cryptography asks that every attacker with realistic computing power learns almost nothing.

The precise version is a game. A challenger picks one of two worlds by a fair coin, such as the encryptions of two messages, and shows the result to an adversary, who guesses the world. The adversary's **advantage** is how much better than the coin it does, and a scheme is secure when every efficient adversary's advantage is tiny.

## Advantage and security levels {#sec-crypto-advantage}

A **distinguisher** sees one sample and outputs a bit. Its skill at telling $X$ from $Y$ is how much more often it says 1 on one than on the other,
$$\mathrm{Adv}_A(X, Y) = \big|\Pr[A(X) = 1] - \Pr[A(Y) = 1]\big|,$$
and in the coin game it wins with probability $\frac12 + \frac12\mathrm{Adv}$. A classifier of real against generated images is one, with advantage its true positive rate minus its false positive rate.

The best distinguisher says 1 exactly on the outcomes more likely under $X$ than under $Y$. Its advantage is the probability $X$ puts on those outcomes in excess of $Y$, the **total variation (TV) distance**, and no distinguisher, however much time it has, can beat it. TV distance 0 is **perfect** security, since there is nothing to find; a tiny one is **statistical** security. **Computational** security allows a large TV distance, as long as no efficient adversary can find the outcomes that differ; this is what makes short keys possible.

"Tiny" needs a definition. The asymptotic one is **negligible**: eventually below one over every polynomial in the key length. But that is about the limit, and says nothing at a fixed key length: $2^{-\sqrt\lambda}$ is negligible, yet still $3.9\times10^{-4}$ at $\lambda = 128$. Real systems fix the key length, so ask for concrete numbers; the usual one is **$\kappa$-bit security**: an attacker running for time $t$ gets advantage at most about $t/2^\kappa$, which is what brute force over $2^\kappa$ keys achieves. A year at $10^{18}$ keys per second tries $2^{84.7}$ keys and finds a 128-bit key with probability $9.3\times10^{-14}$.

## Hybrid arguments {#sec-crypto-hybrid}

Proofs rarely compare the two worlds directly. They walk from one to the other through intermediate **hybrids**, changing one thing at a time. Any gap between the ends must show up somewhere along the chain, so the advantages of the steps add up:
$$\mathrm{Adv}(H_0, H_m) \le \sum_{i=1}^m\mathrm{Adv}(H_{i-1}, H_i).$$
Swapping samples one at a time, $m$ independent samples give at most $m$ times the one-sample advantage.

That factor is safe but pessimistic, because it reflects how the proof is organized, not how the best attacker works. Take $m$ flips of a coin with heads probability 0.51 against $m$ fair flips (@tbl-crypto-coin-hybrid). The hybrid bound $m\delta$, with $\delta = 0.01$, is right for one flip and useless by 100. The best attacker pools the flips; its advantage grows like $\delta\sqrt m$, because the bias in the count grows like $m\delta$ while its random spread grows like $\sqrt m$.

| Flips $m$ | Majority vote (simulation) | Best possible (TV) | Hybrid bound $\min(1, m\delta)$ |
| --- | --- | --- | --- |
| 1 | 0.008 | 0.010 | 0.010 |
| 100 | 0.082 | 0.080 | 1.000 |
| 10,000 | 0.481 | 0.683 | 1.000 |

: Advantage in telling $m$ flips of a $0.51$-coin from $m$ fair flips; the majority vote is simulated over 200,000 games per entry, so at $m = 100$ it exceeds the best possible by sampling noise. {#tbl-crypto-coin-hybrid}

Majority vote, a natural test, stalls near 0.48: its threshold $m/2$ sits at the centre of the fair coin's count, so the fair count lands above it half the time however large $m$ is. A reasonable-looking test can be far from the best, which is why security is measured against every adversary.

## The birthday bound {#sec-crypto-birthday}

Among $q$ random values there are about $q^2/2$ pairs, and each pair collides with probability $1/N$, so collisions become likely once $q^2/2$ approaches $N$, that is after about $\sqrt N$ draws, not $N$. Precisely, for $q$ independent uniform draws from $N$ values,
$$1 - e^{-q(q-1)/(2N)} \le \Pr[\text{collision}] \le \frac{q(q-1)}{2N},$$
with the 50% point at $q \approx 1.18\sqrt N$. The lower bound is the accurate one (@tbl-crypto-birthday).

| Case | $q$ | $N$ | Exact | Union bound | $1 - e^{-q(q-1)/(2N)}$ |
| --- | --- | --- | --- | --- | --- |
| Shared birthday in a class | 23 | 365 | 0.507 | 0.693 | 0.500 |
| Hashing trick: 1,000 features into $2^{20}$ buckets | 1,000 | $2^{20}$ | 0.379 | 0.476 | 0.379 |
| Random 32-bit identifiers for 100,000 records | 100,000 | $2^{32}$ | 0.688 | 1.164 | 0.688 |

: Collision probability, exact against the two bounds. {#tbl-crypto-birthday}

So anything random of $\kappa$ bits starts colliding after about $2^{\kappa/2}$ draws: random 32-bit identifiers for 100,000 records are more likely than not to clash. The same counting limits how much data one key may protect. A block cipher on $n$-bit blocks never maps two inputs to the same output, while a random function sometimes does, so after $q$ blocks an observer can tell them apart only through a collision, with advantage at most $q^2/2^{n+1}$. For 128-bit blocks, $2^{32}$ blocks (64 GiB) cost at most $2^{-65}$, but at $2^{64}$ blocks the bound reaches $\frac12$. A $q$-query bound is a data limit: change keys long before it.

## Pseudorandomness and encryption {#sec-crypto-encryption}

A short random key can stand in for a lot of randomness. A **pseudorandom generator (PRG)** stretches a short seed into a long string that no efficient distinguisher tells from uniform. A **pseudorandom function (PRF)** $F_k$ turns a key into a huge table of random-looking values: no efficient adversary, choosing its own queries, can tell it from a truly random function. In proofs, both let you replace "pseudorandom" by "truly random" at the cost of a small advantage.

The **one-time pad** $m \oplus K$, with a uniform key as long as the message, has **perfect secrecy**: XOR with a uniform key gives a uniform ciphertext whatever the message. But the key is as long as the data, and it must never be reused, since $(m_1 \oplus K) \oplus (m_2 \oplus K) = m_1 \oplus m_2$ cancels the key.

Practical schemes aim for indistinguishability under **chosen-plaintext attack (CPA)**: an adversary who can get any message encrypted picks two, receives the encryption of one, and must not tell which. This forces encryption to be randomized. If encrypting $m_0$ always gives the same ciphertext, the adversary asks for the encryption of $m_0$ and compares, winning with advantage 1.000: deterministic encryption of a column of labels reveals how often each label occurs. The scheme $(r, m \oplus F_k(r))$, with a fresh random $r$ from a set $\mathcal R$, works because each message gets a fresh pad as long as $r$ never repeats. Its advantage after $q$ encryptions is at most twice the PRF advantage plus $q/\lvert\mathcal R\rvert$, and the second term is real: with 8-bit nonces and 16 encryptions, the attack that waits for a repeated $r$ wins with advantage 0.0607, against the bound 0.0625. Size nonces with the birthday bound.

## Simulation and secret sharing {#sec-crypto-simulation}

Many procedures must reveal something, such as the final result, and nothing else. "Nothing else" is hard to state directly, since the participant may combine what it sees cleverly. **Simulation** sidesteps this: if the participant could have faked everything it sees from what it is allowed to learn, then what it sees contains nothing more. Formally, a procedure reveals nothing beyond $L(x)$ if an efficient **simulator**, given only $L(x)$, produces output distributed like the participant's real **view**, perfectly, statistically or computationally.

**Additive secret sharing** is the simplest case. To split a secret $x$ in $\mathbb Z_p$, the integers mod $p$, among $h$ holders, draw $h - 1$ uniform random shares and set the last to $x$ minus their sum, mod $p$. Shares of two secrets add to shares of their sum. Any $h - 1$ shares are uniform whatever $x$ is, because subtracting a uniform number mod $p$ wraps around and hits every value equally often. The simulator outputs uniform numbers: perfect security.

The wrap-around is essential. Over the integers, with $r$ uniform below 16 and the share $x - r$ unreduced, the share's range moves with the secret: values possible for one secret are impossible for the other, and the TV distance between $x = 0$ and $x = 7$ is $7/16 = 0.4375$ (@fig-crypto-shares). In general it is $|x - x'|/R$ for a mask range $R$.

![Histograms of the second share for secrets $x = 0$ and $x = 7$. Left: reduced mod $p = 11$, both are uniform. Right: over the integers with $r < 16$, the support shifts with the secret.](figures/crypto_basics_shares.svg){#fig-crypto-shares}

Simulation proves only that nothing beyond $L(x)$ leaks; whether $L(x)$ itself is acceptable to reveal is a separate question.

## Learning with errors {#sec-crypto-lwe}

**Learning with errors (LWE)** hides a secret vector $s$ behind linear equations whose right-hand sides are slightly wrong: samples $(a, \langle a, s\rangle + e \bmod p)$ with $a$ uniform and a small error $e$. Without the errors, Gaussian elimination recovers $s$. With them, elimination solves the wrong system exactly and multiplies the small errors by large numbers mod $p$: on a 16-dimensional secret it recovers all 16 coordinates without noise, and 0 of 16 with errors in $\{-1, 0, 1\}$. LWE is as hard as worst-case lattice problems, with no known efficient quantum attack.

**Regev encryption** of a bit $m$ adds $m\cdot\frac p2$ to a sample. The key holder subtracts $\langle a, s\rangle$ and is left with $m\cdot\frac p2 + e$, which rounds to the right bit as long as $|e| < p/4$. Adding $k$ ciphertexts adds their bits (that is, their XOR) and also their errors, so with errors bounded by $B$, decryption is guaranteed when
$$k < \frac{p}{4B}.$$
With $p = 4096$ and $B = 32$ that is $k < 32$. But independent errors partly cancel, so the summed error typically grows like $\sigma\sqrt k$, not $kB$, and @tbl-crypto-regev shows that failures stay rare well beyond the guarantee.

| Ciphertexts added $k$ | Empirical failure rate | Normal approximation |
| --- | --- | --- |
| 31 | 0.0000 | 0.0000 |
| 256 | 0.0007 | 0.0006 |
| 1,024 | 0.0868 | 0.0881 |
| 4,096 | 0.3814 | 0.3833 |

: Decryption failure rate of a sum of $k$ Regev ciphertexts, $p = 4096$, errors uniform on $\{-32, \dots, 32\}$. {#tbl-crypto-regev}

Noise is a budget: every operation spends some, and once it nears $p/4$ decryption becomes a coin flip. Going beyond the worst-case budget trades a guarantee for a failure probability.

## Code {#sec-crypto-code}

The birthday bound needs one product:

```python
from math import exp, prod


def collision_probability(q: int, n: int) -> float:
    """Exact probability that q uniform draws from n values contain a repeat."""
    return 1 - prod(1 - i / n for i in range(q))


def union_bound(q: int, n: int) -> float:
    return q * (q - 1) / (2 * n)


def lower_bound(q: int, n: int) -> float:
    return 1 - exp(-union_bound(q, n))


print(f"{collision_probability(23, 365):.3f} {union_bound(23, 365):.3f}")  # 0.507 0.693
print(f"{collision_probability(1_000, 2**20):.3f} {lower_bound(1_000, 2**20):.3f}")  # 0.379 0.379
print(f"{collision_probability(100_000, 2**32):.3f} {union_bound(100_000, 2**32):.3f}")  # 0.688 1.164
```

The lines are the three rows of @tbl-crypto-birthday.

The full edition's script, [crypto_basics.py](../../code/part01-prerequisites/crypto_basics.py), computes every number in this chapter. Its schemes are toys: never use them to protect anything.

```bash
uv run python code/part01-prerequisites/crypto_basics.py
```

## Cheat sheet

| Result | Statement |
| --- | --- |
| Advantage | $\lvert\Pr[A(X) = 1] - \Pr[A(Y) = 1]\rvert = 2\lvert\Pr[\text{win}] - \frac12\rvert \le \mathrm{TV}(X, Y)$ |
| $\kappa$-bit security | Time $t$ buys advantage at most about $t/2^\kappa$ |
| Hybrids | Advantages add along a chain; $m$ samples cost at most a factor $m$ |
| Birthday bound | $\Pr[\text{collision}] \le q(q-1)/(2N)$; 50% at $q \approx 1.18\sqrt N$ |
| Encryption | One-time pad: perfect, never reuse. CPA security needs randomness: $(r, m \oplus F_k(r))$ |
| Simulation | Secure if a simulator fakes the view from the allowed leakage |
| Additive sharing mod $p$ | Any $h - 1$ shares uniform: perfect security |
| Regev sums | Correct while $kB < p/4$; typical noise grows like $\sigma\sqrt k$ |

: The results of this chapter. {#tbl-crypto-cheat}

**Rules of thumb**

- Random values of $\kappa$ bits collide after about $2^{\kappa/2}$ draws: size identifiers, nonces and key lifetimes accordingly.
- In LWE schemes, noise is a budget that each operation spends.

## Further reading

- Katz & Lindell, *Introduction to Modern Cryptography* (3rd ed., 2020), ch. 2–3: perfect secrecy, pseudorandomness and CPA security.
- Lindell, "How to Simulate It — A Tutorial on the Simulation Proof Technique" (2017): the simulation paradigm step by step.
- Peikert, "A Decade of Lattice Cryptography" (2016): LWE, its variants and their parameters.

**Derivations.** Every result in this chapter is derived step by step in [the full edition's chapter](../../book-full/part01-prerequisites/08-crypto-basics.md).
