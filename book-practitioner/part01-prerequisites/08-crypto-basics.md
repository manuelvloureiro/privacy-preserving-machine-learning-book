# Cryptography basics

## Why this matters

To build or review a system that hides data, you need to know which tool does the job, how much data one key can protect, and what a security claim promises. The last question has a precise answer, because security is a game. A challenger secretly flips a fair coin to pick one of two worlds, such as encrypting one message or another. An **adversary** sees the result and guesses which world it is in. A scheme is secure when no efficient adversary beats the coin by more than a tiny amount.

## Which tool for which job {#sec-crypto-tools}

Never build your own scheme. Use a vetted library, and pick the construction by the job, as in @tbl-crypto-tools.

| Job | Use | In Python | What goes wrong |
| --- | --- | --- | --- |
| Encrypt data | Authenticated encryption with associated data (AEAD): AES in GCM (Galois/Counter Mode), or ChaCha20-Poly1305 | `AESGCM` or `ChaCha20Poly1305` from the `cryptography` package | A nonce reused under one key reveals data |
| A keyed pseudonym or tag | A pseudorandom function, in practice the hash-based message authentication code (HMAC) | `hmac.new(key, msg, hashlib.sha256)` | A plain hash of an identifier is reversed by hashing every candidate |
| Keys, nonces, identifiers | A cryptographic random number generator | `secrets.token_bytes`, `secrets.token_hex`, `secrets.randbelow` | The `random` module is predictable |
| Split a value among holders | Additive secret sharing modulo a prime | `secrets.randbelow(p)` | Shares not reduced modulo $p$ leak |
| Compute on encrypted values | Lattice-based homomorphic encryption | Microsoft SEAL, its Python wrapper TenSEAL, or OpenFHE | Every operation spends a noise budget |

: The construction to use for each job, and the mistake to avoid. {#tbl-crypto-tools}

## Advantage {#sec-crypto-advantage}

The simplest adversary sees one sample and answers with a bit. It is called a **distinguisher**, written $A$, and its answer 1 means "I think this sample came from the second world". Write $X$ and $Y$ for the distributions of the sample in the two worlds. The **advantage** of the distinguisher is the gap between how often it says 1 in the two worlds:
$$\mathrm{Adv}_A(X, Y) = \big|\Pr[A(X) = 1] - \Pr[A(Y) = 1]\big|.$$
A distinguisher whose answer 1 points to the second world wins the coin game with probability $\frac12 + \frac12\mathrm{Adv}_A$, so the advantage is twice its edge over a coin flip. For example, take a fair coin against a coin that lands heads with probability 0.6. Saying 1 on heads has advantage 0.1, and so it wins with probability 0.55.

No distinguisher, however long it runs, can do better than the **total variation (TV) distance** between the two distributions ([Divergences](03-divergences.md)).

So if that distance is 0, security is **perfect**, and if it is tiny, it is **statistical**. Most of cryptography is **computational** instead, where no efficient adversary can find where the two worlds differ.

## Efficient adversaries and concrete security {#sec-crypto-negligible}

Textbooks define security asymptotically, requiring a **negligible** advantage, one that eventually falls below one over every polynomial in the key length. But "eventually" says nothing about the key length you use, so never read an asymptotic claim as a concrete one.

Practitioners use **concrete security** instead. A scheme has **$\kappa$-bit security** if no attack does better than trying keys by brute force over $2^\kappa$ keys. An attacker that runs for time $t$ then gets an advantage of at most about
$$\frac{t}{2^\kappa}.$$
Take an attacker who tests $10^{18}$ keys per second for a year. Against a 128-bit key, it succeeds with probability only $9.3\times10^{-14}$. That is why AES uses keys of 128 bits or more.

## Hybrid arguments {#sec-crypto-hybrid}

Proofs compare two worlds through a chain of intermediate worlds called **hybrids**, each differing from the next in one thing, and the advantages of the steps add up. So if one sample cannot be told apart, a number $m$ of independent samples cost at most $m$ times the one-sample advantage, a safe but pessimistic factor.

## Pseudorandom generators and functions {#sec-crypto-prf}

A **pseudorandom generator (PRG)** stretches a short random seed into a long string that looks random to any efficient observer. Stream ciphers such as ChaCha20 are treated as PRGs. A **pseudorandom function (PRF)** is a keyed function, written $F_k$ for the key $k$, whose outputs look like a random table to anyone without the key. HMAC is assumed to be one, which is what makes a keyed pseudonym safe. A plain hash has no key, so anyone can hash every possible identifier and match the results.

## The birthday bound and the data limits of a key {#sec-crypto-birthday}

Many arguments fail only if two random values coincide. Write $q$ for the number of values drawn independently and uniformly from a set of size $N$. The **birthday bound** says that the chance of a repeat is
$$1 - e^{-q(q-1)/(2N)} \le \Pr[\text{collision}] \le \frac{q(q-1)}{2N}.$$ {#eq-crypto-birthday}
A collision becomes as likely as not after about $1.18\sqrt N$ draws, so random values of $\kappa$ bits start to collide after about $2^{\kappa/2}$ draws. This needs uniform values, because others collide more often.

Collisions come early in practice. With the hashing trick, for example, 1,000 features hashed into about a million buckets collide with probability 0.379. Random 32-bit identifiers reach even odds of a clash at 77,164 records, as the snippet computes. So size identifiers until $q^2/(2N)$ is tiny for the largest $q$ you expect.

The same bound limits how much data one key may protect, because treating a block cipher as a PRF costs the birthday bound on the blocks processed. With 128-bit blocks, this **switching lemma** reaches one half at $2^{64}$ blocks, so change keys long before that, and ciphers with 64-bit blocks need new keys far sooner.

## Encryption {#sec-crypto-encryption}

The **one-time pad** encrypts a message $m$ as $m \oplus K$, the bitwise exclusive or (XOR) with a uniformly random key $K$ as long as the message. That gives **perfect secrecy**, but only with a key as long as all the data. It must also never be reused, because the XOR of two ciphertexts under one pad is the XOR of the two messages.

Practical schemes use short keys and aim for security under **chosen-plaintext attack (CPA)**. In its game, the adversary may ask for encryptions of any messages it likes, then guesses which of two messages of equal length a challenge encrypts. Ciphertexts usually reveal length, so pad messages when their length is sensitive.

The game forces encryption to be randomized. If the same message always gave the same ciphertext, the adversary would ask for the encryption of its first message and compare, with advantage 1. Deterministic tokens of email addresses can even be reversed, because anyone who can compute the token reverses it by running a dictionary of addresses through it. The fix is a fresh random value for every encryption, called a **nonce** and written $r$, drawn from a set $\mathcal R$ of possible values. The PRF's output on $r$ then serves as a one-time pad, and the ciphertext is $(r, m \oplus F_k(r))$. After $q$ requested encryptions, its advantage is at most
$$2\,\mathrm{Adv}_{\mathrm{PRF}} + \frac{q}{\lvert\mathcal R\rvert}.$$
The second term is a real attack, because a repeated nonce reuses a pad.

Over the life of a key, any two nonces may repeat, so the birthday bound sets the limit. Take the 96-bit nonces of AES in GCM. If they are random, the chance of any repeat after $2^{32}$ messages is about $1.2\times10^{-10}$, and it grows with the square of the count. So rotate keys well before that, or use a counter as the nonce. And since a CPA-secure scheme does nothing against tampering, use authenticated encryption.

## Simulation and secret sharing {#sec-crypto-simulation}

Many procedures among several participants must reveal the result and nothing else, and **simulation** makes "nothing else" precise. Write $\mathrm{View}(x)$ for everything a participant sees on the inputs $x$, and $L(x)$ for the leakage, what it is allowed to learn. The procedure reveals nothing more if an efficient **simulator** $S$, given only $L(x)$, can fake a view with the same distribution:
$$S(L(x)) \equiv \mathrm{View}(x).$$ {#eq-crypto-simulation}
Whether revealing $L(x)$ is acceptable is a separate decision.

**Additive secret sharing** is the basic example. Take a number $h$ of holders and a prime $p$. To split a secret $x$, draw $h - 1$ shares uniformly at random modulo $p$, and choose the last share so that all of them add up to $x$:
$$s_h = x - (s_1 + \dots + s_{h-1}) \bmod p.$$
Any $h - 1$ shares are then independent uniform numbers whatever the secret is, so a simulator just outputs random numbers and the security is perfect. Holders can add their shares of two secrets to get shares of the sum. Part 3 builds on this, and the sum itself is never hidden.

The wrap-around modulo $p$ is a condition, not a detail, because without it the range of a share moves with the secret. With a mask uniform on 0 to 15 and no reduction, the shares of the secrets 0 and 7 have a TV distance of $7/16$. So choose $p$ larger than any sum you will compute, and reduce every result modulo $p$.

## Learning with errors {#sec-crypto-lwe}

**Learning with errors (LWE)** is the hard problem behind post-quantum key exchange and homomorphic encryption. It hides a secret vector $s$ behind slightly wrong linear equations modulo $p$, and each sample pairs a uniformly random vector $a$ with a number
$$b = \langle a, s\rangle + e \bmod p.$$
Here $\langle a, s\rangle$ is the inner product, and $e$ is a small random error drawn fresh for each sample.

**Regev encryption** shows how such schemes compute on ciphertexts. It hides a bit in a fresh sample, and decryption works while the error stays below a quarter of $p$. Adding ciphertexts adds the bits, but it also adds the errors. So if every error lies within an error bound $B$ of zero, a sum of $k$ ciphertexts is guaranteed to decrypt when
$$k < \frac{p}{4B}.$$ {#eq-crypto-regev-budget}
With $p = 4096$ and $B = 32$, the guarantee covers 32 ciphertexts, but independent errors partly cancel, so failures become measurable only around 256.

This **noise budget** is the central parameter of homomorphic encryption. SEAL, TenSEAL and OpenFHE implement schemes such as BFV for exact integers and CKKS for approximate real numbers. In each, a larger modulus buys more operations at the cost of bigger, slower ciphertexts.

## Code {#sec-crypto-code}

This snippet shares and adds secrets, and sizes identifiers and nonces.

```python
import secrets


def union_bound(q: int, n: int) -> float:
    """Upper bound on the chance that q uniform draws from n values repeat."""
    return q * (q - 1) / (2 * n)


def share(secret: int, holders: int, p: int) -> list[int]:
    """Split a secret into additive shares modulo p."""
    shares = [secrets.randbelow(p) for _ in range(holders - 1)]
    return [*shares, (secret - sum(shares)) % p]


def reconstruct(shares: list[int], p: int) -> int:
    return sum(shares) % p


def even_odds_draws(n: int) -> int:
    """Smallest number of uniform draws from n values with a repeat at least half the time."""
    no_repeat, q = 1.0, 0
    while no_repeat > 0.5:
        no_repeat *= 1 - q / n
        q += 1
    return q


p = 2**61 - 1
a, b = share(42, 3, p), share(100, 3, p)
print(reconstruct(a, p))  # 42
print(reconstruct([x + y for x, y in zip(a, b)], p))  # 142
print(even_odds_draws(2**32))  # 77164
print(f"{union_bound(2**32, 2**96):.1e}")  # 1.2e-10
```

## Cheat sheet

| Result | Statement |
| --- | --- |
| Advantage | Wins with probability $\frac12 + \frac12\mathrm{Adv}$; at most the TV distance |
| $\kappa$-bit security | Time $t$ buys an advantage of at most about $t/2^\kappa$ |
| Birthday bound | Collisions likely after about $\sqrt N$ draws |
| CPA security | Deterministic schemes fail; so does a repeated nonce |
| Additive sharing mod $p$ | Any $h - 1$ shares are uniform; without reduction mod $p$ they leak |

: The results of this chapter. {#tbl-crypto-cheat}

**Rules of thumb**

- Never build your own scheme, and draw keys, nonces and shares from `secrets`, never from `random`.
- Random values of $\kappa$ bits collide after about $2^{\kappa/2}$ draws. Size identifiers, nonces and key lifetimes accordingly.
- Pseudonymize identifiers with a keyed function such as HMAC. A plain hash of an email or a phone number is not anonymization.
- Never read an asymptotic security claim as a concrete one.
- Secret sharing hides the inputs, never the result.

## Further reading

- Katz & Lindell, *Introduction to Modern Cryptography* (3rd ed., 2020), ch. 2–3: perfect secrecy, pseudorandomness and CPA security.
- Lindell, "How to Simulate It — A Tutorial on the Simulation Proof Technique" (2017): the simulation paradigm step by step.

**Derivations.** Every result here is derived step by step in [the full edition's chapter](../../book-full/part01-prerequisites/08-crypto-basics.md).
