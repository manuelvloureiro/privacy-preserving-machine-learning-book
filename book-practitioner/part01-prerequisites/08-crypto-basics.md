# Cryptography basics

## Why this matters

Some computations must hide something, such as the message in an encrypted file, or a value split among several holders that no single holder may learn. To build or review such a system, you need a precise answer to one question. What does it mean that an observer learns nothing it was not supposed to learn?

Against unlimited computing power, a key shorter than the message cannot hide it perfectly, because seeing the ciphertext rules some messages out. Perfect hiding is possible, but it needs a key as long as all the data. Practical cryptography therefore asks for less, namely that every attacker with realistic computing power learns almost nothing. This chapter makes "realistic" and "almost nothing" precise, and shows the few tools that prove such statements.

The whole subject rests on one idea, which is that security is a game. A challenger secretly flips a fair coin to pick one of two worlds, for example one world where it encrypts one message and one where it encrypts another. The challenger shows the result to an **adversary**, whose task is to guess which world it is in. An adversary who learns nothing from what it sees is right half the time, whatever it answers, so only a win rate above one half shows that something leaked. A scheme is secure when no efficient adversary beats the coin by more than a tiny amount, where "efficient" means roughly "feasible in practice" and is defined below. Proofs reach that conclusion in small steps, each of which changes one thing in what the adversary sees, and the rest of the chapter shows how.

## Advantage {#sec-crypto-advantage}

The simplest adversary sees one sample and answers with a single bit. It is called a **distinguisher**, written $A$, and its answer 1 means "I think this sample came from the second world". Let $X$ be the distribution of the sample in the first world, and let $Y$ be its distribution in the second. A distinguisher that is any good then says 1 more often on samples from $Y$ than on samples from $X$. Its **advantage** is the gap between how often it says 1 in the two worlds. To write this in symbols, let $\Pr[A(X) = 1]$ be the probability that it says 1 when its sample comes from $X$, and likewise for $Y$. The advantage is then
$$\mathrm{Adv}_A(X, Y) = \big|\Pr[A(X) = 1] - \Pr[A(Y) = 1]\big|.$$

The absolute value also credits a distinguisher whose answers point the wrong way, since flipping its answers turns it into one that points the right way. A distinguisher that ignores its sample says 1 equally often in both worlds, so its advantage is 0, while one that is always right has advantage 1.

The advantage and the chance of winning the coin game measure the same thing. The distinguisher wins when it says 1 in the second world or 0 in the first. Since each world comes up half the time, its chance of winning is the average of these two success rates:
$$\Pr[\text{win}] = \tfrac12\Pr[A(Y) = 1] + \tfrac12\Pr[A(X) = 0] = \tfrac12 + \tfrac12\big(\Pr[A(Y) = 1] - \Pr[A(X) = 1]\big).$$
So a distinguisher whose answer 1 points to the second world wins with probability one half plus half its advantage. For example, let the first world be a fair coin and the second a coin that lands heads with probability 0.6. A distinguisher that says 1 on heads says 1 with probability 0.6 in the second world and 0.5 in the first. Its advantage is therefore 0.1, and it wins the game with probability 0.55, which is one half plus 0.05.

You already know distinguishers from machine learning, since a classifier trained to tell real images from generated ones is one. If it outputs 1 for "real", it says 1 on real images at its true positive rate and on generated images at its false positive rate. Its advantage is therefore the difference between the two rates. In statistics, deciding which of two distributions produced a sample is a hypothesis test, the subject of [Hypothesis testing](04-hypothesis-testing.md).

How large can the advantage be? Some outcomes are more likely under one distribution than under the other. A distinguisher, even a randomized one, can only choose for each outcome how often to say 1 on it. Saying 1 on an outcome that is more likely under $Y$ widens the gap, while saying 1 on one that is more likely under $X$ narrows it. So the best distinguisher says 1 exactly on the outcomes that are more likely under $Y$, however long it may run. Its advantage is then the extra probability that $Y$ puts on those outcomes, compared with $X$. That quantity is the **total variation (TV) distance** between the two distributions, which [Divergences](03-divergences.md) studies. It is 0 when the two distributions are equal, and 1 when they never produce the same outcome.

This limit splits security into three levels. If the TV distance is 0, the two worlds look identical and no adversary has any advantage, which is **perfect** security. If the TV distance is tiny, even an adversary with unlimited time gains almost nothing, which is **statistical** security. The third level, **computational** security, is the main business of cryptography. There the TV distance can be large, as long as no efficient adversary can find the outcomes where the two worlds differ. A short key cannot make the two worlds identical, but it can make their difference too hard to find, and that is how a short key protects a long message.

## Efficient adversaries and tiny advantages {#sec-crypto-negligible}

No key resists every attacker, but the key's length decides how much an attack costs. An attacker who tries keys one after another finds a 40-bit key quickly and a 256-bit key never. Security is therefore a statement about how fast the attacker's success shrinks as the key grows. Cryptographers measure the size of the key with a **security parameter**, written $\lambda$, which you can think of as the key length in bits.

The asymptotic view describes what happens as $\lambda$ grows, and it needs two definitions. The first says which attackers count as efficient, namely those whose running time grows at most like a polynomial in $\lambda$, such as the cube of $\lambda$. Such an attacker may also flip coins as it runs. It is therefore called **probabilistic polynomial time (PPT)**, probabilistic because of the coins and polynomial time because of the bound on its running time.

The second definition says which advantages count as tiny. Think of the adversary's advantage as a function of $\lambda$. That function is **negligible** if it falls below one over every polynomial in $\lambda$ once $\lambda$ is large enough, and stays below it for all larger $\lambda$. For example, $2^{-\lambda}$ is negligible, because $2^\lambda$ outgrows every polynomial. But $\lambda^{-10}$ is not, because it never falls below $\lambda^{-11}$. A scheme is **secure** if every PPT adversary has a negligible advantage.

The two definitions fit together because of repetition. An efficient attacker can repeat its attack a polynomial number of times, and the union bound from [Concentration inequalities](02-concentration.md) limits what that buys. It says that the chance that at least one of several events happens is at most the sum of their chances. So the chance that at least one try succeeds is at most the number of tries times the chance per try. A negligible chance per try falls below one over any polynomial, including the number of tries times any other polynomial you pick. Multiplying it by the number of tries therefore still leaves it below one over that other polynomial, so repetition does not help.

A small chance that is not negligible does not survive repetition. Suppose an attack succeeds with probability $\lambda^{-3}$ on each try, and the tries are independent. Independence lets the chances of failure multiply, so $\lambda^4$ tries all fail with probability at most $e^{-\lambda}$, and the attack almost surely succeeds.

Brute force shows why the best one can ask for is "negligible", never zero. Suppose an attacker tries $t$ different keys against a key drawn uniformly from all $2^\lambda$ keys. Each try is right with probability $2^{-\lambda}$, so the attacker succeeds with probability $t/2^\lambda$. That chance is negligible when $t$ is a polynomial in $\lambda$, but it is never zero.

The weakness of these definitions is that "negligible" describes a limit, not a value at the key length you actually use. @tbl-crypto-negligible-128 evaluates four functions at $\lambda = 128$. The negligible $2^{-\sqrt\lambda}$ is still $3.9\times10^{-4}$ there, far larger than the non-negligible $\lambda^{-10}$. It drops below $\lambda^{-10}$ for good only from $\lambda = 20{,}520$ (@fig-crypto-negligible). A statement that holds "for all large enough $\lambda$" does not tell you whether 128 is large enough.

| Function | Negligible? | Value at $\lambda = 128$ |
| --- | --- | --- |
| $2^{-\lambda}$ | yes | $2.9\times10^{-39}$ |
| $\lambda^{-\log_2\lambda}$ | yes | $1.8\times10^{-15}$ |
| $2^{-\sqrt\lambda}$ | yes | $3.9\times10^{-4}$ |
| $\lambda^{-10}$ | no | $8.5\times10^{-22}$ |

: Four functions at $\lambda = 128$. The negligible $2^{-\sqrt\lambda}$ is far larger than the non-negligible $\lambda^{-10}$. {#tbl-crypto-negligible-128}

![Bits of security, minus the base-2 logarithm of each function, for three negligible functions and the non-negligible $\lambda^{-10}$, on log–log axes. Each negligible curve eventually overtakes the dashed one; $2^{-\sqrt\lambda}$ does so only beyond $\lambda = 20{,}520$.](figures/crypto_basics_negligible.svg){#fig-crypto-negligible}

Real systems fix the key length. A block cipher encrypts data in blocks of fixed size under a key, and the standard one, AES, uses keys of 128 bits or more. **Concrete security** therefore replaces "efficient" and "negligible" by explicit numbers, naming the attacker's resources and bounding the advantage they can buy. Those resources are the attacker's running time and the number of questions it may ask an oracle. An oracle is a box the game lets it query, such as one that encrypts messages of its choice. The benchmark is brute force. A scheme has **$\kappa$-bit security**, where $\kappa$ is a number of bits, if no attack does better than brute force over $2^\kappa$ keys. In other words, an attacker that runs for time $t$ gets an advantage of at most about
$$\frac{t}{2^\kappa}.$$

Here is what 128-bit security buys. Suppose an attacker tests $10^{18}$ keys per second for a whole year, which comes to about $2^{84.7}$ keys. Against a 128-bit key, that attack succeeds with probability only $9.3\times10^{-14}$. At that rate, trying all $2^{128}$ keys would take $1.08\times10^{13}$ years.

Security proofs are **reductions**. A scheme is built from a **primitive**, a simpler building block such as a block cipher. Its security proof turns any adversary who breaks the scheme into one that breaks the primitive. Read in reverse, this says that if the primitive is hard to break, so is the scheme. But the conversion usually loses something. Suppose a reduction turns an advantage against the scheme into one $2^{20}$ times smaller against the primitive. Then, if the primitive has 128-bit security, the scheme has only about 108 bits. A **tight** reduction loses almost nothing, while a loose one forces larger keys for the same guarantee.

## Hybrid arguments {#sec-crypto-hybrid}

Proofs rarely compare the two worlds of a game directly. They walk from one to the other through intermediate worlds called **hybrids**, and each world differs from the next in one thing. Write $m$ for the number of steps and $H_i$ for the world reached after $i$ steps. Then $H_0$ is the first world of the game and $H_m$ is the second.

Look at how often a distinguisher says 1 in each world along the way. The gap between the two ends is the sum of the gaps between neighbours, since every intermediate frequency is added once and subtracted once. The size of a sum is at most the sum of the sizes, so the advantages add up:
$$\mathrm{Adv}_A(H_0, H_m) \le \sum_{i=1}^m\mathrm{Adv}_A(H_{i-1}, H_i).$$
So any gap between the two ends must show up in at least one step of the chain.

The most common use is for many samples. Suppose that one sample from $X$ cannot be told from one sample from $Y$. To compare $m$ independent samples from each, walk from all samples drawn from $X$ to all drawn from $Y$, swapping one sample at a time. Each step changes a single sample, so it is a one-sample problem, and the total advantage is at most $m$ times the one-sample advantage.

The argument needs two conditions. First, the number of samples $m$ must be a polynomial in $\lambda$, because otherwise $m$ times a negligible advantage need not be negligible, and the bound says nothing. Second, both distributions must be efficiently samplable, because of how the proof works. It turns an attack on $m$ samples into an attack on one sample. That attack places its one sample at a random position, and fills the other positions with samples it draws itself, from $Y$ before that position and from $X$ after it. If drawing samples were slow, the one-sample attack would no longer be efficient.

The factor $m$ is safe but pessimistic, because it reflects how the proof is organized, not how the best attacker works. Take a coin that lands heads with probability 0.51, so that its bias, written $\delta$, is 0.01, and compare $m$ of its flips with $m$ fair flips. One flip gives an advantage of at most $\delta$, so the hybrid bound for $m$ flips is $m\delta$. @tbl-crypto-coin-hybrid compares it with the best possible advantage and with a simple test, majority vote, which says 1 if more than half the flips are heads.

| Flips $m$ | Majority vote (simulation) | Best possible (TV) | Hybrid bound $\min(1, m\delta)$ |
| --- | --- | --- | --- |
| 1 | 0.008 | 0.010 | 0.010 |
| 100 | 0.082 | 0.080 | 1.000 |
| 10,000 | 0.481 | 0.683 | 1.000 |

: Advantage in telling $m$ flips of a $0.51$-coin from $m$ fair flips. The majority vote is simulated over 200,000 games per entry, so at $m = 100$ it exceeds the best possible by sampling noise. {#tbl-crypto-coin-hybrid}

The hybrid bound is exact for one flip but useless by 100 flips. To see why, note that the best attacker counts the heads, and its advantage depends on how far apart the two average counts are compared with their random spread. The average counts differ by $m\delta$, while each count spreads by about $\sqrt m/2$ around its average. Their distance in units of the spread is therefore $2\delta\sqrt m$, so the best advantage grows like $\delta\sqrt m$, not like $\delta m$ (@fig-crypto-hybrid). To reach a constant advantage, such as one half, the attacker needs about $1/\delta^2$ flips, while the hybrid bound would allow it after $1/\delta$.

![Advantage of telling $m$ flips of a coin with bias $\frac12 + 0.01$ from fair flips: majority vote (simulation), the best possible distinguisher (exact TV distance), and the hybrid bound $m\delta$.](figures/crypto_basics_hybrid.svg){#fig-crypto-hybrid}

At 10,000 flips, majority vote reaches an advantage of only 0.481, while the best possible is 0.683. The trouble is its threshold of half the flips, which sits at the centre of the fair coin's count. For many flips, the biased count lands above that threshold almost always, but the fair count still lands above it about half the time. So the advantage of majority vote can never exceed about one half. The best test puts its threshold halfway between the two average counts instead. A reasonable-looking test can thus be far from the best one, which is why security is measured against every adversary.

## Pseudorandom generators and functions {#sec-crypto-prf}

Truly random bits are expensive to produce and to share, while a short random key is cheap. Two tools stretch a short key into a lot of randomness that looks real to any efficient observer.

A **pseudorandom generator (PRG)** is a fast, deterministic procedure that stretches a short random seed into a longer string. It is secure if no efficient distinguisher can tell its output from truly random bits of the same length.

A PRG can never be statistically secure. Suppose its output is at least one bit longer than its seed. Then there are at most half as many seeds as output strings, so at most half of all strings of that length are outputs of the PRG. A distinguisher that says 1 exactly on those outputs always says 1 on the PRG's output, and at most half the time on random bits. Its advantage is therefore at least one half. But checking whether a string is an output means trying every seed, which takes exponential time. That is why pseudorandomness is only a promise against efficient distinguishers.

PRGs exist if one-way functions exist, which are functions that are easy to compute but hard to invert. In practice, a stream cipher such as ChaCha20 encrypts by combining the data with a long keystream generated from a short key. Designers treat stream ciphers as PRGs, which means assuming that no efficient distinguisher can tell the keystream from random bits.

A **random function** is a table that gives each input its own independent, uniform output. A table on 128-bit inputs has $2^{128}$ entries, so nobody can store one.

A **pseudorandom function (PRF)** is a keyed function, written $F_k$ for the key $k$, whose outputs look like those of a random function to anyone without the key. Its game uses an oracle, a box that a fair coin fills either with $F_k$ under a freshly drawn random key or with a truly random function. The adversary may ask the box for its output on any inputs it likes, each one chosen after seeing the earlier answers. The PRF is secure if no efficient adversary can tell which function is in the box. In practice, the hash-based message authentication code (HMAC) is assumed to be a PRF.

A PRF can be built from any PRG whose output is twice as long as its seed, with a construction called the GGM tree, after Goldreich, Goldwasser and Micali. It grows a binary tree of keys with the secret key at the root, and the PRG applied to a node gives two halves, which become its two children. To evaluate the PRF, each bit of the input picks the left or the right child, and the leaf reached at the end is the output.

The proof of the GGM tree is a chain of hybrids that replaces the tree by random values, one level at a time. Write $n$ for the number of input bits, which is also the number of levels, and $q$ for the number of queries the adversary makes. At each level the $q$ queries touch at most $q$ nodes, so each step of the chain costs at most $q$ times the PRG's advantage. Write $\mathrm{Adv}_{\mathrm{PRG}}$ for that advantage and $\mathrm{Adv}_{\mathrm{PRF}}$ for an efficient adversary's advantage against the PRF. Over the $n$ levels, the steps add up to
$$\mathrm{Adv}_{\mathrm{PRF}} \le nq\cdot\mathrm{Adv}_{\mathrm{PRG}}.$$
The full edition sketches this proof and cites the complete one.

In proofs, both tools play the same role, since they let you replace "pseudorandom" by "truly random" at the cost of a small advantage.

## The birthday bound {#sec-crypto-birthday}

Many arguments about random values fail only if two of the values happen to coincide, and the **birthday bound** says how likely that is. Draw $q$ values independently and uniformly from a set of size $N$. They form about $q^2/2$ pairs. Each pair coincides with probability $1/N$, because whatever the first value is, the second matches it with probability $1/N$. So collisions become likely once the number of pairs approaches $N$, which happens after about $\sqrt N$ draws, far fewer than $N$. Precisely,
$$1 - e^{-q(q-1)/(2N)} \le \Pr[\text{collision}] \le \frac{q(q-1)}{2N}.$$ {#eq-crypto-birthday}

The upper bound is the union bound over all pairs. The lower bound instead comes from the exact chance of no collision, which multiplies the chances that each new value misses all earlier ones. The two bounds are close. The full edition shows that, while the upper bound is below 1, it overstates the true chance by a factor of at most about 1.6. It also computes that a collision becomes as likely as not at about $1.18\sqrt N$ draws.

The bound needs the values to be uniform and independent, because that is what makes each pair coincide with probability $1/N$. Values that are not uniform coincide more often, so the upper bound can then fail.

| Case | $q$ | $N$ | Exact | Union bound | $1 - e^{-q(q-1)/(2N)}$ |
| --- | --- | --- | --- | --- | --- |
| Shared birthday in a class | 23 | 365 | 0.507 | 0.693 | 0.500 |
| Hashing trick: 1,000 features into $2^{20}$ buckets | 1,000 | $2^{20}$ | 0.379 | 0.476 | 0.379 |
| Random 32-bit identifiers for 100,000 records | 100,000 | $2^{32}$ | 0.688 | 1.164 | 0.688 |

: Collision probability, exact against the two bounds of @eq-crypto-birthday. {#tbl-crypto-birthday}

@tbl-crypto-birthday shows how early collisions come. With the hashing trick, 1,000 features hashed into about a million buckets already collide with probability 0.379. Even random 32-bit identifiers for 100,000 records clash more often than not. There the union bound is above 1 and says nothing, while the lower bound is accurate. @fig-crypto-birthday checks the bounds against simulation for $N = 2^{16}$. In general, random values of $\kappa$ bits start to collide after about $2^{\kappa/2}$ draws.

![Probability of a collision among $q$ uniform draws from $N = 2^{16}$ values: exact, simulated, the union bound $q(q-1)/(2N)$ and the lower bound $1 - e^{-q(q-1)/(2N)}$.](figures/crypto_basics_birthday.svg){#fig-crypto-birthday}

The birthday bound also limits how much data one key may protect, and the reason runs through block ciphers. A block cipher is a keyed permutation, which never maps two different inputs to the same output. It is modelled as a **pseudorandom permutation (PRP)**, a keyed function that no efficient adversary can tell from a random shuffle of the inputs. Many proofs, such as the encryption proof in the next section, are written for a PRF instead, whose outputs on different inputs are independent. But a permutation's outputs are not quite independent, since they can never repeat.

The only visible difference between a random function and a random permutation is that the function sometimes repeats an output and the permutation never does. So an adversary can tell them apart only about as often as the answers collide, and this observation is the **switching lemma**. Write $n$ for the number of bits in a block, so that there are $2^n$ possible blocks. Treating a PRP as a PRF then costs an extra advantage, on top of its advantage as a PRP. After $q$ queries, the lemma bounds that extra advantage by
$$\frac{q^2}{2^{n+1}},$$
which is the birthday bound for $2^n$ values. The bound is tight, because an adversary that says 1 when it sees a repeated answer has an advantage equal to the exact chance of a collision.

So a bound in terms of queries is really a limit on data. Take blocks of 128 bits, so that $2^{32}$ blocks hold 64 GiB of data. That much data costs an extra advantage of at most $2^{-65}$. At $2^{48}$ blocks the bound has grown to $2^{-33}$. By $2^{64}$ blocks it reaches one half and says nothing, because the guarantee degrades with the square of the data under one key. That is why you should change keys long before that point. Ciphers with 64-bit blocks fare much worse. To keep the extra advantage below $2^{-20}$, they can process only about 6 million blocks under one key.

## Encryption {#sec-crypto-encryption}

An encryption scheme has **perfect secrecy** if the ciphertext's distribution is the same for every message. Seeing the ciphertext then does not change how likely any message is.

The **one-time pad** achieves this with a key $K$ that is a uniformly random string as long as the message $m$. The ciphertext is $m \oplus K$, where $\oplus$ is the bitwise exclusive or (XOR), which flips each bit of the message where the key has a 1. For a fixed message, each key gives a different ciphertext, and each ciphertext comes from exactly one key. A uniform key therefore gives a uniform ciphertext, whatever the message is. @tbl-crypto-otp shows this for 2-bit messages.

| Key $K$ | 00 | 01 | 10 | 11 |
| --- | --- | --- | --- | --- |
| Ciphertext for $m = 01$ | 01 | 00 | 11 | 10 |
| Ciphertext for $m = 10$ | 10 | 11 | 00 | 01 |

: The one-time pad on 2-bit messages. Each row contains every ciphertext once, so the ciphertext is uniform for every message. {#tbl-crypto-otp}

Perfect secrecy has a price, because it needs at least as many keys as messages. Fix a ciphertext that occurs. Because decryption must recover the message, each key decrypts it to one message, so it can come from at most as many messages as there are keys. With fewer keys than messages, some message could never produce that ciphertext, and seeing it would rule that message out. The pad must also be used only once, because combining two ciphertexts under the same pad with XOR cancels the pad and reveals the XOR of the two messages.

Practical schemes use short keys, so they settle for computational security. The standard goal is security under **chosen-plaintext attack (CPA)**, defined by a game. The adversary may ask an oracle for encryptions of any messages it likes, at any time. It then submits two messages of equal length, receives the encryption of one of them, chosen by a fair coin, and must guess which one it got. The lengths must be equal because a ciphertext usually reveals the length of its message. The encryption oracle models reality, since an attacker can often get chosen data encrypted, such as a request it sends or a value it plants in a database. Passing this game is equivalent to a scheme hiding everything an efficient adversary could compute about the message, apart from its length.

The game has a consequence that surprises many programmers, which is that encryption must be randomized. To see why, suppose encrypting the same message always gives the same ciphertext. The adversary can then ask for the encryption of its first message and compare it with the challenge. That works because two different messages never share a ciphertext, since decryption could not then tell which one to return. So if the two ciphertexts match, the adversary got the first message, and otherwise the second. Since it is always right, its advantage is 1, and the full edition's script measures exactly 1.000.

The same weakness shows up without any game. Deterministic encryption of a column of labels maps equal labels to equal ciphertexts, so the frequency of each ciphertext reveals the frequency of each label.

The fix is to use a fresh pad for every encryption. To encrypt, draw a random value $r$, called a nonce, uniformly from a set of possible values, written $\mathcal R$. The PRF's output on $r$, $F_k(r)$, then serves as a one-time pad, and $r$ is sent along so that the receiver can rebuild the pad. The ciphertext is the pair
$$(r,\ m \oplus F_k(r)).$$

This scheme can fail in only two ways. Either the PRF can be told from a random function, or the challenge reuses a nonce that the adversary has already seen. A reused nonce is fatal because the adversary already knows its pad. Each encryption it requested showed it a nonce together with the ciphertext of a message it chose, and combining the two with XOR reveals the pad for that nonce. So if the challenge reuses that nonce, the adversary removes the pad and reads the message.

The bound adds the two failures. Write $q$ for the number of encryptions the adversary requests, and $\lvert\mathcal R\rvert$ for the number of possible nonces. By the union bound, the challenge's nonce matches one of the $q$ earlier ones with probability at most $q/\lvert\mathcal R\rvert$. Writing $\mathrm{Adv}_{\mathrm{PRF}}$ for the PRF's advantage, the scheme's advantage is then at most
$$2\,\mathrm{Adv}_{\mathrm{PRF}} + \frac{q}{\lvert\mathcal R\rvert}.$$
The factor 2 comes from converting a win probability into an advantage, by the rule in @sec-crypto-advantage.

The second term is real, not a weakness of the proof. Take 8-bit nonces and 16 encryptions, for example. Then an attacker who waits for a repeated nonce wins with advantage 0.0607, against a bound of 0.0625. This matches the simulation, which measures 0.0605. With 128-bit nonces, though, the same attack has an advantage below $2^{-124}$.

The proof needs only the challenge's nonce to differ from the earlier ones, and that costs at most $q/\lvert\mathcal R\rvert$, which grows in proportion to $q$. Over the life of a key, though, any two nonces may repeat, and each repeat reuses a pad. By the birthday bound, repeats become likely after about $\sqrt{\lvert\mathcal R\rvert}$ encryptions, so size the nonce set for that.

## Simulation and secret sharing {#sec-crypto-simulation}

Encryption hides everything. Many procedures instead run among several participants, each holding its own input, and must reveal something to each of them, such as the final result, and nothing else. "Nothing else" is hard to state directly, because a participant may combine what it sees in clever ways. **Simulation** sidesteps the problem. Suppose a participant could have faked everything it sees from what it is allowed to learn. Then what it sees contains nothing more.

A participant's **view**, written $\mathrm{View}(x)$, is everything it sees during a run, which depends on all the inputs $x$ and on the procedure's randomness. Write $L(x)$ for what it is allowed to learn, such as its own input and the output. A **simulator**, written $S$, is an efficient procedure that receives only $L(x)$ and produces a fake view. The procedure reveals nothing beyond $L(x)$ if the fake view has the same distribution as the real one, for every input $x$:
$$S(L(x)) \equiv \mathrm{View}(x).$$ {#eq-crypto-simulation}
This exact equality is perfect security. If the two distributions are only close in TV distance, the security is statistical, and if no efficient distinguisher can tell them apart, it is computational.

The definition captures "nothing else" because anything the participant could compute from its real view, it could compute equally well from the simulator's output. That output was made from $L(x)$ alone, so whatever the view reveals, $L(x)$ already revealed. The simulator must be efficient, because in computational security an unbounded simulator could compute things the participant cannot. The definition would then accept views that reveal facts that are easy to read from the view but hard to compute from $L(x)$. Finally, the definition compares the view only with $L(x)$, so it says nothing about whether revealing $L(x)$ itself is acceptable. That is decided when you choose $L$.

**Additive secret sharing** is the simplest example. It works with the integers from 0 to $p - 1$, where $p$ is the modulus and arithmetic wraps around mod $p$. To split a secret $x$ among $h$ holders, draw the first $h - 1$ shares, written $s_1$ to $s_{h-1}$, independently and uniformly at random. The last share is $x$ minus the sum of the others,
$$s_h = x - (s_1 + \dots + s_{h-1}) \bmod p,$$
so that all the shares add up to $x$. Suppose two secrets are shared among the same holders, and each holder adds its share of one to its share of the other. The results are shares of the sum of the secrets, so the holders can compute a sum without ever putting their shares together.

Any $h - 1$ of the shares look like independent uniform numbers, whatever the secret is, because the one missing share is what ties them to $x$. The reason is that subtracting a uniform number mod $p$ wraps around and hits every value equally often. @tbl-crypto-shares shows this for two holders and $p = 5$. As the random first share $r$ runs through all five values, the second share $x - r$ takes each value exactly once, for either secret. The simulator for a group of $h - 1$ holders therefore just outputs uniform random numbers, and its fake view has exactly the real distribution, so the security is perfect.

| $r$ | 0 | 1 | 2 | 3 | 4 |
| --- | --- | --- | --- | --- | --- |
| Share $x - r \bmod 5$ for $x = 2$ | 2 | 1 | 0 | 4 | 3 |
| Share $x - r \bmod 5$ for $x = 4$ | 4 | 3 | 2 | 1 | 0 |

: Two-holder additive sharing modulo 5. Each row is a reordering of the values 0 to 4, so the share is uniform for every secret. {#tbl-crypto-shares}

Suppose the group of $h - 1$ holders also learns the secret, for example because it is published later. The simulator then receives $x$ as well. It draws the group's $h - 1$ shares uniformly, and these shares together with $x$ fix the one missing share as $x$ minus their sum, exactly as in the real sharing. This is the typical form of a simulator, random where the view is random and consistent with the leakage where the leakage pins the view down.

The wrap-around is a condition of the result, not a detail, because without it the share's range moves with the secret. For example, suppose the mask $r$ is uniform on the 16 values from 0 to 15, and the share $x - r$ is not reduced. Call the number of possible masks, here 16, the mask width $W$. The share then lies in a window of 16 values that ends at the secret itself. So the window for the secret 7 starts seven places above the window for the secret 0. Seven of the 16 values in each window are therefore impossible under the other secret, which makes the TV distance $7/16 = 0.4375$. A simulation with 100,000 shares for each secret measures 0.4423 (@fig-crypto-shares).

For comparison, the simulation also draws $r$ uniformly from 0 to 10 and reduces the share $x - r$ modulo 11. The measured TV distance is then 0.0045, which is sampling noise, since the exact distance is zero. In general, for two secrets $x$ and $x'$ whose difference is smaller than the mask width $W$, the TV distance between their shares without the wrap-around is
$$\frac{\lvert x - x'\rvert}{W}.$$
A much wider mask makes it tiny, which gives statistical rather than perfect security.

![Histograms of the second share for secrets $x = 0$ and $x = 7$. Left: reduced mod $p = 11$, both are uniform. Right: over the integers with $r < 16$, the support shifts with the secret.](figures/crypto_basics_shares.svg){#fig-crypto-shares}

## Learning with errors {#sec-crypto-lwe}

Every tool so far is defined by a property, such as looking random, and works with any primitive that has it. **Learning with errors (LWE)** is instead a specific hard problem, with three attractions. Under a condition on its errors, it is as hard as worst-case problems on lattices, regular grids of points in many dimensions. No efficient quantum attack on it is known. And encryption built on it lets you add ciphertexts without decrypting them.

LWE hides a secret vector behind linear equations whose right-hand sides are slightly wrong. Here the secret $s$ is a vector of numbers mod $p$, and its length $d$ is called the dimension. The problem gives out samples, each of which pairs a uniformly random vector $a$ with a number $b$. To compute $b$, take the inner product $\langle a, s\rangle$, the sum of the products of matching entries. Then add the error $e$, a small random integer drawn fresh for each sample, and reduce mod $p$ to get
$$b = \langle a, s\rangle + e \bmod p.$$

The search version of the problem asks for $s$, while the decision version asks whether a list of pairs consists of LWE samples or of uniform random pairs.

The errors are what make the problem hard. Without them, the samples form a linear system, and Gaussian elimination solves it from $d$ independent equations. With them, elimination solves the wrong system exactly, because undoing the equations multiplies the small errors by large numbers mod $p$, which scrambles the answer. The full edition's script shows this with a 16-dimensional secret and $p = 4096$. Elimination recovers all 16 coordinates when there are no errors, but none of them once small errors of $-1$, 0 or 1 are added.

Hardness holds under a condition on the errors. Regev proved it for errors drawn from a normal distribution rounded to integers, with a standard deviation large enough relative to $\sqrt d$. For such errors, solving LWE is at least as hard as worst-case lattice problems. His proof used a quantum algorithm, though later work gave classical proofs. Other small error distributions, such as the uniform errors in the toy below, need separate arguments, and the full edition cites these results rather than proving them.

**Regev encryption** of a single bit uses LWE directly. It is a secret-key scheme, in which the same secret $s$ encrypts and decrypts. Picture the numbers mod $p$ as a clock with $p$ positions, where $p$ is even. To encrypt a bit $m$, take a fresh LWE sample and move its $b$ halfway round the clock if the bit is 1, so that the ciphertext is
$$\big(a,\ \langle a, s\rangle + e + m\cdot\tfrac p2 \bmod p\big).$$
Whoever knows $s$ subtracts $\langle a, s\rangle$, which leaves the error plus the encoded bit. That is near 0 for the bit 0 and near halfway for the bit 1, and decryption reads off whichever is closer. It is correct as long as the error is smaller than a quarter of the clock,
$$\lvert e\rvert < \frac p4.$$
A larger error could push the value past the boundary between the two halves.

Without the key, the sample looks uniform, by the decision version of LWE, so the bit is added to something that looks uniform, which works like a one-time pad. That is why the scheme is secure against chosen-plaintext attack, provided that decision LWE is hard.

Adding ciphertexts entry by entry adds everything inside them. The bits add, and two half-turns round the clock make a full turn, so the sum decrypts to the XOR of the bits. Two half-turns make exactly a full turn only when $p$ is even, which is why the scheme needs an even $p$. The errors add as well. Suppose every error lies between $-B$ and $B$, where $B$ is the error bound. A sum of $k$ ciphertexts then has an error of at most $kB$, and it decrypts correctly whenever
$$k < \frac{p}{4B}.$$ {#eq-crypto-regev-budget}
With $p = 4096$ and $B = 32$, the budget guarantees that sums of fewer than 32 ciphertexts decrypt correctly.

That guarantee is a worst case, which assumes every error is as large as possible and all have the same sign. Independent errors with mean zero mostly cancel instead. Their variances add, so the summed error has a standard deviation that grows like $\sqrt k$, not like $k$. @tbl-crypto-regev compares a simulation with a normal approximation built on this fact.

| Ciphertexts added $k$ | Empirical failure rate | Normal approximation |
| --- | --- | --- |
| 1 | 0.0000 | 0.0000 |
| 31 | 0.0000 | 0.0000 |
| 256 | 0.0007 | 0.0006 |
| 1,024 | 0.0868 | 0.0881 |
| 4,096 | 0.3814 | 0.3833 |

: Decryption failure rate of a sum of $k$ Regev ciphertexts, $p = 4096$, errors uniform on $\{-32, \dots, 32\}$ (20,000 trials). {#tbl-crypto-regev}

Take a sum of 1,024 ciphertexts as an example. The full edition computes the standard deviation of the summed error from the variance of the individual errors, and gets about 600. A quarter of the clock is 1,024, or about 1.7 standard deviations. A normal value lands that far from zero with probability about 0.088. Overall, failures become measurable only for sums of around 256 ciphertexts, eight times the guaranteed 32 (@fig-crypto-regev).

![Decryption failure rate of a sum of $k$ Regev ciphertexts ($p = 4096$, errors uniform on $\{-32, \dots, 32\}$): simulation against the normal approximation. The dashed line is the worst-case guarantee $k < p/(4B) = 32$.](figures/crypto_basics_regev_noise.svg){#fig-crypto-regev}

But the budget is still finite. Once the error is comparable to a quarter of the clock, the decrypted value is spread round the whole clock, and decryption becomes a coin flip. A larger modulus, relative to the error, buys more additions. That trade-off sets the parameters of every LWE-based scheme that computes on ciphertexts.

A Regev ciphertext carries one bit in $d + 1$ numbers, which is wasteful. Practical lattice schemes use structured variants of LWE, such as ring-LWE and module-LWE, which pack many values into each ciphertext and compute faster. Module-LWE underlies ML-KEM, the standardized scheme for sending a fresh shared key, published as FIPS 203 in 2024.

## Code {#sec-crypto-code}

The exact collision probability needs one product. Each new value must miss all the earlier ones, and the code multiplies those chances:

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

The three print lines show two numbers from each of the three rows of @tbl-crypto-birthday.

The full edition's script, [crypto_basics.py](../../code/part01-prerequisites/crypto_basics.py), computes every number in this chapter. Its schemes are toys, and their parameters are far too small. Nothing in it runs in constant time, so its running time can depend on secret values. Its random numbers also come from a generator that is not designed to be unpredictable. Never use it to protect anything.

```bash
uv run python code/part01-prerequisites/crypto_basics.py
```

## Cheat sheet

| Result | Statement |
| --- | --- |
| Advantage | $\lvert\Pr[A(X) = 1] - \Pr[A(Y) = 1]\rvert$, and $\Pr[\text{win}] = \frac12 + \frac12\mathrm{Adv}$ |
| Best possible distinguisher | Advantage equal to the TV distance, whatever its running time |
| Negligible | Eventually below one over every polynomial in $\lambda$; survives polynomially many repetitions |
| $\kappa$-bit security | Time $t$ buys an advantage of at most about $t/2^\kappa$ |
| Hybrid argument | Advantages add along a chain; $m$ samples cost at most a factor $m$ |
| Birthday bound | $1 - e^{-q(q-1)/(2N)} \le \Pr[\text{collision}] \le q(q-1)/(2N)$; 50% at $q \approx 1.18\sqrt N$ |
| PRP to PRF switching | Extra advantage at most $q^2/2^{n+1}$ for $n$-bit blocks, and tight |
| GGM tree | PRF advantage at most $nq$ times the PRG advantage |
| One-time pad | Perfect secrecy; needs as many keys as messages; never reuse the pad |
| CPA security | Deterministic schemes have advantage 1; $(r, m \oplus F_k(r))$ has advantage at most $2\,\mathrm{Adv}_{\mathrm{PRF}} + q/\lvert\mathcal R\rvert$ |
| Simulation | Secure if a simulator fakes the view from the allowed leakage $L(x)$ |
| Additive sharing mod $p$ | Any $h - 1$ shares are uniform: perfect security |
| Regev encryption | Correct if $\lvert e\rvert < p/4$; sums of $k$ ciphertexts correct if $kB < p/4$; typical error grows like $\sqrt k$ |

: The results of this chapter. {#tbl-crypto-cheat}

**Rules of thumb**

- Security of $\kappa$ bits means an attacker with time $t$ succeeds with probability about $t/2^\kappa$.
- Random values of $\kappa$ bits collide after about $2^{\kappa/2}$ draws. Size identifiers, nonces and key lifetimes accordingly.
- Encryption that hides whether two messages are equal must be randomized.
- In LWE schemes, the error is a budget that each operation spends.

## Further reading

- Katz & Lindell, *Introduction to Modern Cryptography* (3rd ed., 2020), ch. 2–3: perfect secrecy, pseudorandomness and CPA security, with complete proofs.
- Boneh & Shoup, *A Graduate Course in Applied Cryptography*, ch. 2–5: the same material in the concrete-security style, including the switching lemma.
- Lindell, "How to Simulate It — A Tutorial on the Simulation Proof Technique" (2017): the simulation paradigm step by step.
- Peikert, "A Decade of Lattice Cryptography" (2016): LWE, its variants and their parameters.

**Derivations.** Every result in this chapter is derived step by step in [the full edition's chapter](../../book-full/part01-prerequisites/08-crypto-basics.md), with a few exceptions. It cites the existence of PRGs, the hardness of LWE, and the equivalence between passing the CPA game and hiding everything an efficient adversary could compute. It also only sketches the proof for the GGM tree.
