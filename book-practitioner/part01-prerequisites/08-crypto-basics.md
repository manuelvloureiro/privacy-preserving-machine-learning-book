# Cryptography basics

## What it is {#sec-crypto-what}

A security claim is a statement about a guessing game. A challenger picks one of two worlds, such as which of two messages to encrypt, and an adversary sees the result and guesses the world. Write $X$ and $Y$ for what it sees in the two worlds. If an adversary, written $\mathcal A$, answers 1 for the second world, its **advantage** is the difference between how often it says 1 in each:
$$\mathrm{Adv}_{\mathcal A}(X, Y) = \big|\Pr[\mathcal A(X) = 1] - \Pr[\mathcal A(Y) = 1]\big|.$$
If the two worlds are equally likely, it wins with probability $\frac12 + \frac12\mathrm{Adv}_{\mathcal A}$. No advantage is more than the total variation distance between $X$ and $Y$ ([Divergences](03-divergences.md)).

A scheme has **$\kappa$-bit security** when no attack beats a search through the keys, so an attacker who tries $t$ keys gets an advantage of at most about $t/2^\kappa$. For example, $10^{18}$ keys per second for a year break a 128-bit key with probability $9.3\times10^{-14}$.

## When you meet it {#sec-crypto-when}

You meet these tools if a pipeline pseudonymizes identifiers, makes keys, encrypts data or splits a value among holders. @tbl-crypto-tools gives the construction for each job. Never write your own.

| Job | Use | In Python | What goes wrong |
| --- | --- | --- | --- |
| Encrypt data | Authenticated encryption, such as AES in Galois/Counter Mode | `AESGCM` from the `cryptography` package | A reused nonce reveals data |
| A keyed pseudonym or tag | The hash-based message authentication code (HMAC) | `hmac.new(key, msg, hashlib.sha256)` | A plain hash is reversed by hashing every candidate |
| Keys, nonces, identifiers | A cryptographic random generator | `secrets.token_bytes` | `random` and seeded NumPy generators are predictable |
| Split a value among holders | Additive secret sharing modulo a prime | `secrets.randbelow(p)` | Shares not reduced modulo $p$ leak |
| Compute on encrypted values | Lattice-based homomorphic encryption | SEAL, TenSEAL or OpenFHE | Every operation spends a noise budget |

: The construction to use for each job, and the mistake to avoid. {#tbl-crypto-tools}

## How to use it {#sec-crypto-use}

Draw a number $m$ of values uniformly from a number $N$ of possible values. The **birthday bound** says that the chance of a repeat is at most
$$\Pr[\text{collision}] \le \frac{m(m-1)}{2N}.$$ {#eq-crypto-birthday}
A repeat is as likely as not after about $1.18\sqrt N$ draws. So random $\kappa$-bit values start to collide after about $2^{\kappa/2}$.

To split a secret among several holders, draw all shares but one uniformly modulo a prime $p$, and choose the last so that they add up to the secret. When each holder adds its shares of two secrets, it gets a share of their sum, as the snippet shows for 42 and 100. Its birthday bounds show why pseudonyms of 10 million identifiers keep 128 bits of the HMAC. At 64 bits, a collision has probability up to $2.7\times10^{-6}$.

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

HMAC is secure if the compression function of SHA-256 is a pseudorandom function, and AES if AES is a pseudorandom permutation. Lattice schemes need learning with errors to be hard. Secret sharing needs no assumption, because a holder's shares are random numbers for every secret, which **simulation** makes precise.

Homomorphic encryption hides each value behind a small random error modulo a number $Q$. Decryption works while the total error is below $Q/4$. A sum of ciphertexts adds their errors, so if each error is at most a bound $B$, a sum of $k$ ciphertexts decrypts when $k < Q/(4B)$. A larger modulus gives a larger **noise budget** and slower arithmetic.

## What goes wrong {#sec-crypto-wrong}

A repeated nonce shows the XOR of the two messages, so change keys before random nonces collide. With 96-bit nonces, the snippet's last line puts that chance at $1.2\times10^{-10}$ after $2^{32}$ messages.

Reduce shares modulo $p$, or their range moves with the secret. An unreduced share equal to the secret plus a random number from 0 to 15 tells the secrets 0 and 7 apart with advantage $7/16$.

"No efficient adversary gets more than a negligible advantage" is an asymptotic claim, which says nothing about the key length you use.

**Derivations.** Every result here is derived step by step in [the full edition's chapter](../../book-full/part01-prerequisites/08-crypto-basics.md).
