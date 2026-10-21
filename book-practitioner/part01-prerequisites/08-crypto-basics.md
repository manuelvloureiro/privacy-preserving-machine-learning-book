# Cryptography basics

## What it is {#sec-crypto-what}

A security claim is a statement about a guessing game. A challenger secretly picks one of two worlds, such as encrypting one of two messages, and an adversary sees the result and guesses. Write $X$ and $Y$ for what it sees in each world. If an adversary $A$ answers 1 for the second, its **advantage** is the gap between how often it says 1 in each:
$$\mathrm{Adv}_A(X, Y) = \big|\Pr[A(X) = 1] - \Pr[A(Y) = 1]\big|.$$
If each world is equally likely, it wins with probability $\frac12 + \frac12\mathrm{Adv}_A$, and no advantage exceeds the total variation distance between $X$ and $Y$ ([Divergences](03-divergences.md)).

A scheme has **$\kappa$-bit security** when no attack beats trying keys. An attacker who tries $t$ keys then gets an advantage of at most about $t/2^\kappa$. For example, testing $10^{18}$ keys per second for a year breaks a 128-bit key with probability $9.3\times10^{-14}$.

## When you meet it {#sec-crypto-when}

You meet these tools whenever a pipeline pseudonymizes identifiers, generates keys, encrypts data or splits a value among holders. @tbl-crypto-tools gives the construction for each job; never implement your own.

| Job | Use | In Python | What goes wrong |
| --- | --- | --- | --- |
| Encrypt data | Authenticated encryption, such as AES in Galois/Counter Mode | `AESGCM` from the `cryptography` package | A reused nonce reveals data |
| A keyed pseudonym or tag | The hash-based message authentication code (HMAC) | `hmac.new(key, msg, hashlib.sha256)` | A plain hash is reversed by hashing every candidate |
| Keys, nonces, identifiers | A cryptographic random generator | `secrets.token_bytes` | `random` and seeded NumPy generators are predictable |
| Split a value among holders | Additive secret sharing modulo a prime | `secrets.randbelow(p)` | Shares not reduced modulo $p$ leak |
| Compute on encrypted values | Lattice-based homomorphic encryption | SEAL, TenSEAL or OpenFHE | Every operation spends a noise budget |

: The construction to use for each job, and the mistake to avoid. {#tbl-crypto-tools}

## How to use it {#sec-crypto-use}

The **birthday bound** says that when a number $q$ of values are drawn uniformly from a number $N$ of possible values, the chance of a repeat is at most
$$\Pr[\text{collision}] \le \frac{q(q-1)}{2N}.$$ {#eq-crypto-birthday}
A repeat becomes as likely as not after about $1.18\sqrt N$ draws, so random $\kappa$-bit values start to collide after about $2^{\kappa/2}$. For example, 1,000 features hashed into $2^{20}$ buckets collide with exact probability 0.379.

To split a secret among several holders, draw all shares but one uniformly modulo a prime $p$, and choose the last so that they add up to the secret. Adding two secrets' shares holder by holder gives shares of their sum. The snippet shares 42 and 100 and adds them. For 10 million identifiers, its birthday bounds show why a pseudonym keeps 128 bits of the HMAC. At 64 bits the bound is $2.7\times10^{-6}$.

```python
import hashlib
import hmac
import secrets

p = 2**61 - 1  # a prime larger than any secret or sum of secrets
a, b = [secrets.randbelow(p) for _ in range(2)], [secrets.randbelow(p) for _ in range(2)]
a, b = [*a, (42 - sum(a)) % p], [*b, (100 - sum(b)) % p]  # three shares each of 42 and 100
print(sum(a) % p, sum(x + y for x, y in zip(a, b)) % p)  # 42 142

key = secrets.token_bytes(32)  # kept apart from the data
pseudonym = hmac.new(key, b"alice@example.com", hashlib.sha256).hexdigest()[:32]  # 128 bits
q = 10**7  # identifiers
print(f"{q * (q - 1) / (2 * 2**128):.1e} {q * (q - 1) / (2 * 2**64):.1e}")  # 1.5e-25 2.7e-06
q, n = 2**32, 2**96  # messages under one key, 96-bit nonces
print(f"{q * (q - 1) / (2 * n):.1e}")  # 1.2e-10
```

The `cryptography` package is not installed here, so this encryption sketch was not run:

```text
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
key = AESGCM.generate_key(bit_length=256)
nonce = secrets.token_bytes(12)             # 96 bits, fresh for every message
ct = AESGCM(key).encrypt(nonce, data, aad)  # aad: authenticated, not encrypted
```

HMAC is secure if SHA-256's compression function is a pseudorandom function. Encryption with AES assumes that AES is a pseudorandom permutation, and lattice schemes that learning with errors is hard. Secret sharing needs none: a holder's shares are random numbers whatever the secret, which **simulation** makes precise.

Homomorphic encryption hides each value behind a small random error modulo a number $Q$. Decryption works while the total error is below $Q/4$. Adding ciphertexts adds their errors, so if each error is at most a bound $B$ in size, a sum of $k$ ciphertexts is guaranteed to decrypt when $k < Q/(4B)$. With $Q = 4096$ and $B = 32$, that guarantees 32 ciphertexts. A larger modulus buys a larger **noise budget**, at the cost of slower arithmetic.

## What goes wrong {#sec-crypto-wrong}

A repeated nonce reveals the XOR of the two messages, so rotate keys before the birthday bound bites.

Shares must be reduced modulo $p$, or their range moves with the secret. An unreduced share equal to the secret plus a random number from 0 to 15 tells the secrets 0 and 7 apart with advantage $7/16$.

Finally, "no efficient adversary gets more than a negligible advantage" is an asymptotic claim and says nothing about the key length you use.

**Derivations.** Every result here is derived step by step in [the full edition's chapter](../../book-full/part01-prerequisites/08-crypto-basics.md).
