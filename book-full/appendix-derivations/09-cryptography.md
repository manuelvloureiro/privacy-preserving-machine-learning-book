# Pseudorandom functions and learning with errors

This appendix proves two results that [Cryptography basics](../part01-prerequisites/08-crypto-basics.md) states without a full proof. The first is that the GGM tree, named after Goldreich, Goldwasser and Micali, is a pseudorandom function (PRF) whenever the pseudorandom generator (PRG) it is built from is secure. The second is about learning with errors (LWE), a problem in which a secret vector is hidden behind noisy linear equations. When the arithmetic is done modulo a prime, an algorithm that can only *recognize* such equations can be turned into one that *finds* the secret. Both proofs are reductions: each turns an algorithm that solves one problem into an algorithm that solves another. Both also rest on the same move. The new algorithm takes the input it must judge, called its challenge, and uses it to build exactly the situation the old algorithm expects to see. So the heart of each proof is a check that two distributions are identical. @tbl-apx-crypto-map lists where each result comes from.

| Derivation | Where the chapter states it | What the chapter gives |
| --- | --- | --- |
| @sec-apx-crypto-ggm: the GGM tree is a PRF, with advantage at most $nq$ times the PRG's | @sec-crypto-ggm | A proof sketch, citing Goldreich for the full proof |
| @sec-apx-crypto-lwe: search LWE reduces to decision LWE for prime $p$ polynomial in $d$ | @sec-crypto-lwe, Definition | One sentence describing Regev's idea |

: The two derivations in this appendix and the chapter section that states each one. {#tbl-apx-crypto-map}

The notation is the chapter's. The security parameter $\lambda$ is the length of keys and seeds in bits, and $U_k$ stands for the uniform distribution on strings of $k$ bits. An **adversary** is an algorithm that tries to break a scheme, usually by asking it questions called queries, and $q$ is the number of queries it makes. The modulus in learning with errors is $p$.

A **distinguisher** receives input from one of two distributions, its two "worlds", and outputs a bit. Its **advantage** is the absolute difference between how often it outputs 1 in the two worlds. A procedure is **efficient** if its running time grows at most polynomially in $\lambda$, and a quantity is **negligible** if it shrinks faster than one over any polynomial in $\lambda$.

## The GGM tree is a pseudorandom function {#sec-apx-crypto-ggm}

### What the chapter uses

A pseudorandom generator stretches a short random seed into a longer string that efficient algorithms cannot tell from random. Here the generator doubles the length: $G: \{0,1\}^\lambda \to \{0,1\}^{2\lambda}$. Write its output as a left half $G_0(s)$ followed by a right half $G_1(s)$, each as long as the seed, so that $G(s) = (G_0(s), G_1(s))$. A PRG distinguisher $D$ receives one $2\lambda$-bit string and outputs a bit, and its advantage is
$$\mathrm{Adv}_{\mathrm{PRG}}(D) = \big|\Pr[D(G(U_\lambda)) = 1] - \Pr[D(U_{2\lambda}) = 1]\big|.$$

The GGM tree turns $G$ into a keyed function on $n$-bit inputs. Picture a binary tree of depth $n$ in which every node carries a $\lambda$-bit **label**. The root's label is the key $k$. A node's left child gets the left half $G_0$ of $G$ applied to the node's label, and its right child gets the right half $G_1$. An input $x = x_1 x_2 \cdots x_n$ is then read as directions from the root to a leaf. At level $i$, the bit $x_i$ says go left if it is 0 and right if it is 1, and the output is the label of the leaf you reach,
$$F_k(x) = G_{x_n}\big(\cdots G_{x_2}\big(G_{x_1}(k)\big)\cdots\big).$$
A node is named by the bits that lead to it, so node $01$ is the right child of the left child of the root, and its **depth** is the length of its name.

In the PRF game, a fair coin decides what sits inside a box. It is either $F_k$ with a uniform key, or a truly random function $f$ from $n$-bit inputs to $\lambda$-bit outputs, which gives each input an independent uniform output. An adversary $A$ sends up to $q$ queries, each chosen after seeing the answers to the earlier ones, and then outputs a bit. The box that answers the queries is called an oracle, and $A^O$ means $A$ run with oracle $O$. The adversary's advantage is
$$\mathrm{Adv}_{\mathrm{PRF}}(A) = \big|\Pr[A^{F_k} = 1] - \Pr[A^{f} = 1]\big|,$$
where the probabilities are also over the adversary's own random choices.

The result says that any such adversary can be turned into a PRG distinguisher whose advantage is smaller by exactly the factor $nq$, and which is about as fast. Precisely, take any adversary $A$ that makes at most $q$ queries. Then there is a PRG distinguisher $B$ that runs $A$ once and, apart from that, evaluates $G$ fewer than $nq$ times, such that
$$\mathrm{Adv}_{\mathrm{PRF}}(A) = nq\cdot\mathrm{Adv}_{\mathrm{PRG}}(B).$$ {#eq-apx-crypto-ggm}
The running time matters because PRG security only limits fast distinguishers. So if no distinguisher with $B$'s running time has PRG advantage above some $\varepsilon$, then no $q$-query adversary has PRF advantage above $nq\,\varepsilon$, which is the chapter's $\mathrm{Adv}_{\mathrm{PRF}} \le nq\cdot\mathrm{Adv}_{\mathrm{PRG}}$. The chapter gives a proof sketch and cites Goldreich, *Foundations of Cryptography*, Vol. 1, §3.6, for the full proof.

### The idea

The proof walks from the real function to a random one in $n$ steps, one level of the tree at a time. Each intermediate world is called a **hybrid**, because it is part real and part random. The $j$-th hybrid, $H_j$, gives every node at depth $j$ a fresh uniform label and computes everything below it with $G$ as usual. The first hybrid, $H_0$, is the real $F_k$, because the only node at depth 0 is the root and its uniform label is the key. The last, $H_n$, is a random function, because every leaf gets its own uniform label. Going from $H_{j-1}$ to $H_j$ changes a single thing. In $H_{j-1}$, the two children of a node at depth $j - 1$ carry the two halves of $G$ applied to a uniform label, while in $H_j$ they carry two uniform labels. Telling these apart is the PRG's game of output against uniform string, played once for every node at depth $j - 1$ that the adversary's queries pass through. @fig-apx-crypto-ggm-hybrids shows the four hybrids for $n = 3$ and three queries.

![The GGM tree of depth 3 in the hybrids $H_0$ to $H_3$, for the queries 000, 001 and 100. Orange nodes get independent uniform labels, blue nodes are computed with $G$ from their parent, and grey nodes lie on a query path but are never used. A red $y_c$ marks the node at depth $j - 1$ to which the reduction $B_j$ assigns its $c$-th challenge string.](figures/apx_crypto_ggm_hybrids.svg){#fig-apx-crypto-ggm-hybrids}

The subtle part is that the adversary chooses its queries as it goes. A reduction $B_j$ that wants to use the adversary to break the PRG receives $q$ challenge strings, which are either all outputs of $G$ on independent uniform seeds or all uniform. But it cannot know in advance which nodes at depth $j - 1$ the queries will pass through. So it hands out its challenge strings like cloakroom tickets. The first time a query passes through a new node at depth $j - 1$, that node gets the next unused string and keeps it for the rest of the game. @tbl-apx-crypto-tickets follows $B_2$ through the three queries of @fig-apx-crypto-ggm-hybrids.

| Query | Node at depth 1 | First visit? | String used | Node at depth 2 | Its label |
| --- | --- | --- | --- | --- | --- |
| 000 | 0 | yes | $y_1$ | 00 | left half of $y_1$ |
| 001 | 0 | no | $y_1$ | 00 | left half of $y_1$ |
| 100 | 1 | yes | $y_2$ | 10 | left half of $y_2$ |

: How $B_2$ assigns its challenge strings $y_1, y_2, y_3$ to the nodes at depth 1 as the queries arrive. The third string is never needed. {#tbl-apx-crypto-tickets}

If each $y_c$ is $G$ applied to its own uniform seed, then the labels at depth 2 are halves of $G$ applied to independent uniform labels at depth 1, which is $H_1$. If each $y_c$ is uniform, the labels at depth 2 are uniform, which is $H_2$. The one thing left to check is that handing out values on first use gives the adversary the same distribution of answers as drawing them all in advance. It does, because a value nobody has looked at yet is still a fresh draw, and step 4 below proves this carefully.

### Derivation

The proof has four parts. Steps 1 to 3 set up the hybrids and show that two neighbouring hybrids differ only in the distribution of one table. Steps 4 and 5 build the reduction $B_j$, which judges $q$ strings at once, and show that it recreates either neighbour exactly. Steps 6 and 7 turn it into a distinguisher that judges a single string, and add up the levels. Step 8 restates the result for efficient adversaries.

1. **Define the hybrids.** For $j = 0, 1, \dots, n$, hybrid $H_j$ gives every node at depth $j$ an independent uniform $\lambda$-bit label $L(x_1\cdots x_j)$, all drawn before the game starts. It answers the query $x$ by walking down from the depth-$j$ node on its path:
   $$H_j(x) = G_{x_n}\big(\cdots G_{x_{j+1}}\big(L(x_1\cdots x_j)\big)\cdots\big).$$
   With $j = 0$ the only label is the root's, which is a uniform key, so $H_0$ is $F_k$. With $j = n$ the answer is the leaf's own label, independent and uniform for each input, so $H_n$ is a random function. The table of $2^j$ labels can be huge, but that does not matter. The hybrids are thought experiments, and only the reduction has to be efficient.

2. **Split the advantage into levels.** Write $h_j = \Pr[A^{H_j} = 1]$ for how often the adversary says 1 in hybrid $H_j$. In the sum $\sum_{j=1}^n (h_{j-1} - h_j)$, each intermediate $h_j$ appears once with a plus sign and once with a minus sign, so the sum telescopes:
   $$h_0 - h_n = \sum_{j=1}^n (h_{j-1} - h_j).$$ {#eq-apx-crypto-telescope}
   The chapter calls this telescoping Fact 3, where it is used with absolute values. The left side is $\Pr[A^{F_k} = 1] - \Pr[A^f = 1]$, whose absolute value is $\mathrm{Adv}_{\mathrm{PRF}}(A)$.

3. **Write two neighbouring hybrids as one table.** Fix $j \ge 1$ and look at a node $w$ at depth $j - 1$. In $H_{j-1}$, its two children $w0$ and $w1$ get the two halves of $G(L(w))$, so together they carry the string $G(L(w))$. Different nodes $w$ have independent uniform labels, so these strings are independent, each distributed as $G(U_\lambda)$. In $H_j$, the children carry two independent uniform labels, so together they carry a string distributed as $U_{2\lambda}$, again independently for each $w$. Below depth $j$ both hybrids compute with $G$ in the same way. So the two hybrids are one and the same world, except for the distribution $\mathcal D$ that the children's strings come from. In that world, a table $T$ gives each node $w$ at depth $j - 1$ its own independent draw $T(w)$ from $\mathcal D$. The query $x$ is answered by taking the half of $T(x_1\cdots x_{j-1})$ that bit $x_j$ selects and walking down with $G$:
   $$W_j^{\mathcal D}(x) = G_{x_n}\big(\cdots G_{x_{j+1}}\big(\text{half } x_j \text{ of } T(x_1\cdots x_{j-1})\big)\cdots\big).$$
   With $\mathcal D = G(U_\lambda)$ this world is $H_{j-1}$, and with $\mathcal D = U_{2\lambda}$ it is $H_j$. For $j = 1$ the only node at depth 0 is the root, so the table has a single entry.

4. **Lazy sampling gives the same view.** There are two ways to fill the table $T$. In the *advance* version, every entry is drawn before the game starts. In the *lazy* version, each entry is drawn from $\mathcal D$ the first time a query needs it, and reused afterwards. The adversary's **view** is everything it sees: its own random choices (its coins) and all the answers it receives. The claim is that the view has the same distribution in both versions. Think of dice rolled earlier and hidden under cups: lifting a cup now shows a number that is exactly as random as rolling the die now, provided nobody peeked.

   The proof goes one query at a time. Call the **history** after $i$ queries the adversary's coins together with the table entries used so far. Every answer, and therefore every next query, is a function of the history. If the next query needs an entry that was already used, both versions answer with that same entry, which is part of the history. If it needs a new entry, then in the advance version that entry was drawn independently of the other entries and of the adversary's coins, so given the history it is still distributed as $\mathcal D$. In the lazy version it is a fresh draw from $\mathcal D$, so the same holds. So in both versions, the history after $i + 1$ queries is the history after $i$ queries plus one entry with the same conditional distribution. The probability of a whole history is the product of these conditional probabilities, one per query (the chain rule), so the histories have the same distribution after every query. The same then holds for anything computed from them, including the adversary's output.

5. **Build $B_j$, which judges $q$ strings at once.** The reduction $B_j$ receives $q$ strings $y_1, \dots, y_q$ of $2\lambda$ bits each. They are drawn independently from one distribution $\mathcal D$, which is either $G(U_\lambda)$ or $U_{2\lambda}$. It runs $A$ and keeps a ticket table, which starts empty. When a query $x$ arrives, $B_j$ looks at the node $w = x_1\cdots x_{j-1}$ at depth $j - 1$ on its path, and if $w$ has no ticket yet, gives it the next unused index $c$. It then answers with the half of $y_c$ that $x_j$ selects, walked down with $G$ exactly as in step 3. At the end, $B_j$ outputs whatever $A$ outputs. Each query passes through one node at depth $j - 1$, so at most $q$ nodes ever need a ticket and the strings never run out.

   Now compare this with step 4. The string $y_c$ is first read when the $c$-th new node appears. Until then, everything $B_j$ has done depends only on $y_1, \dots, y_{c-1}$ and on $A$'s coins, and $y_c$ is independent of all of them. So $B_j$ fills the table of step 3 lazily, in the sense of step 4, and $A$'s view inside $B_j$ is exactly its view in $W_j^{\mathcal D}$. With step 3, this gives
   $$\Pr\big[B_j(G(U_\lambda)^q) = 1\big] = h_{j-1}, \qquad \Pr\big[B_j(U_{2\lambda}^q) = 1\big] = h_j,$$
   where $G(U_\lambda)^q$ means $q$ independent outputs of $G$ on uniform seeds, and $U_{2\lambda}^q$ means $q$ independent uniform strings.

6. **Go from $q$ strings to one.** The chapter's hybrid lemma for many samples does this, and here is its argument for this case. Write $g_i$ for the probability that $B_j$ outputs 1 on a list of $q$ independent strings whose first $i$ are uniform and whose other $q - i$ are outputs of $G$. The two ends are the cases of step 5: $g_0 = h_{j-1}$ (all from $G$) and $g_q = h_j$ (all uniform). The mixed lists in between match no hybrid, and they do not need to. The one-string distinguisher $B'_j$ takes a challenge string $z$ and picks a position $i$ uniformly from 1 to $q$. It puts uniform strings before position $i$, puts $z$ at position $i$, and after it puts outputs of $G$ on seeds it draws itself, which is possible because $G$ is public. Then it outputs $B_j$'s answer on the whole list. If $z$ comes from $G$, the list starts with $i - 1$ uniform strings, so $B'_j$ says 1 with probability $g_{i-1}$; if $z$ is uniform, the probability is $g_i$. Averaging over $i$ gives $\frac1q\sum_{i=1}^q (g_{i-1} - g_i)$, which telescopes as in step 2 to $\frac1q(g_0 - g_q)$:
   $$\Pr[B'_j(G(U_\lambda)) = 1] - \Pr[B'_j(U_{2\lambda}) = 1] = \frac1q\,(h_{j-1} - h_j).$$

7. **Pick the level at random and add up.** The final distinguisher $B$ picks $j$ uniformly from $\{1, \dots, n\}$ and runs $B'_j$. Averaging step 6 over $j$ and using @eq-apx-crypto-telescope,
   $$\Pr[B(G(U_\lambda)) = 1] - \Pr[B(U_{2\lambda}) = 1] = \frac1n\sum_{j=1}^n \frac1q\,(h_{j-1} - h_j) = \frac{h_0 - h_n}{nq}.$$
   Taking absolute values gives @eq-apx-crypto-ggm. As for the cost, $B$ runs $A$ once. It computes at most $q - 1$ outputs of $G$ for the list, and at most $n - 1$ evaluations of $G$ per query to walk down the tree. That is at most $(q - 1) + q(n - 1) = nq - 1$ evaluations, fewer than $nq$.

8. **Conclude for efficient adversaries.** Suppose $n$ and $q$ grow at most polynomially in $\lambda$. Then an efficient $A$ gives an efficient $B$. If $A$'s advantage were not negligible, it would be at least $\lambda^{-c}$ for some exponent $c$ and infinitely many $\lambda$. Dividing by the polynomial $nq$ keeps it above one over a polynomial, so $B$'s advantage $\mathrm{Adv}_{\mathrm{PRF}}(A)/(nq)$ would not be negligible either, and $G$ would not be a PRG. So if $G$ is a PRG, the GGM tree is a PRF.

Adaptivity matters only in step 5. A reduction that fixed in advance which nodes get challenge strings would have to cover all $2^{j-1}$ nodes at depth $j - 1$. That can be exponentially many strings, and the loss factor would grow with it, from $q$ to $2^{j-1}$. The ticket rule lets the adversary's own queries decide where the $q$ strings go, and step 4 is why that changes nothing in what the adversary sees.

### Checking it

The script runs the whole reduction on a toy with $\lambda = 8$, $n = 3$ and $q = 3$. Its PRG is broken on purpose, so that every advantage is large enough to measure: $G(s) = (s, f(s))$ for a fixed random permutation $f$ of the 8-bit strings. So the left half is the seed itself, and the right half is a public function of it. In the real $F_k$, every left step keeps the label and every right step applies $f$. So $F_k(000) = k$, $F_k(001) = f(k)$ and $F_k(100) = f(k)$, while $F_k(110) = f(f(k))$ because it takes two right steps.

The adversary uses these relations, and it chooses its third query adaptively so that the check also covers the subtle part of step 5. It asks 000 and then 001, and for its third query it asks 100 if the first answer starts with a 0 bit, and 110 otherwise. Check 1 asks whether the second answer is $f$ of the first. Check 2 asks whether the third answer is $f$ of the first, after 100, or $f$ applied twice to the first, after 110. The adversary flips a coin to choose one check, and outputs 1, meaning "random function", if that check fails. A uniform label passes a fixed equality only with probability $2^{-8}$, so a check on labels that are independent fails with probability $1 - 2^{-8}$. Both checks pass in $H_0$, so $h_0 = 0$. In $H_1$ and $H_2$, the first two leaves still hang below one node and are computed from it with $G$, so check 1 passes. But check 2 compares labels from different branches of the tree, which are independent, so it fails with probability $1 - 2^{-8}$. That gives $h_1 = h_2 = \frac12(1 - 2^{-8}) = 0.4980$. In $H_3$ both checks fail, so $h_3 = 1 - 2^{-8} = 0.9961$. @tbl-apx-crypto-ggm-check compares these with simulation.

| $j$ | Exact $h_j$ | Simulated $h_j$ | $B_j$ on 3 outputs of $G$ (should be $h_{j-1}$) | $B_j$ on 3 uniform strings (should be $h_j$) |
| --- | --- | --- | --- | --- |
| 0 | 0.0000 | 0.0000 | | |
| 1 | 0.4980 | 0.4994 | 0.0000 | 0.5005 |
| 2 | 0.4980 | 0.4985 | 0.4979 | 0.4993 |
| 3 | 0.9961 | 0.9962 | 0.4983 | 0.9964 |

: How often the adversary says "random" in each hybrid, and in each reduction $B_j$ fed outputs of the toy $G$ or uniform strings (seed 0, 200,000 runs per entry). {#tbl-apx-crypto-ggm-check}

Each $B_j$ lands on the hybrid that step 5 predicts, to within simulation noise of about 0.001, even though the adversary's third query depends on what it has seen. @fig-apx-crypto-ggm-check draws the same comparison. The single-sample distinguisher $B$ of step 7 says 1 with probability 0.5535 on outputs of $G$ and 0.6641 on uniform strings. Its advantage is therefore 0.1105, against the predicted $0.9961/9 = 0.1107$. Multiplying back by $nq = 9$ gives 0.995, which matches the adversary's PRF advantage $h_3 - h_0 = 0.9961$ as @eq-apx-crypto-ggm says it should.

![How often the toy adversary says "random" in the hybrids $H_0$ to $H_3$, exactly and by simulation, and in each reduction $B_j$. Red triangles are $B_j$ fed outputs of $G$, placed at $H_{j-1}$; orange squares are $B_j$ fed uniform strings, placed at $H_j$. Every marker sits on the curve.](figures/apx_crypto_ggm_check.svg){#fig-apx-crypto-ggm-check}

Since @eq-apx-crypto-ggm is an equality, the factor $nq$ between the two advantages is lost inside $B$ itself, in the averaging over levels and positions. The toy shows why. The adversary's rate jumps by 0.498 at levels 1 and 3 and does not move at level 2, but $B$ spreads its effort evenly over all levels and all $q$ positions, most of which carry no signal. So the factor $nq$ comes from how the proof is organized, just like the factor for $m$ samples in the chapter's hybrid lemma.

### What it does not cover

The proof needs every input to have the same length $n$. With inputs of different lengths the GGM tree is not a PRF, because $F_k(0)$ is the label of the node that $F_k(00)$ hangs from. An adversary then checks whether $F_k(00) = G_0(F_k(0))$, which always holds for the real function and almost never for a random one. The proof also assumes that $G$ is deterministic, because a node's label must give the same children every time a query passes through it. It assumes as well that $G$ doubles its input exactly; a generator that stretches by only one bit can be applied repeatedly to its own output to reach double length. Finally, @eq-apx-crypto-ggm compares advantages. To turn it into a concrete guarantee, such as "no attacker running for $2^{128}$ steps succeeds", you need a concrete bound on the advantage of PRG distinguishers with $B$'s running time.

## Search LWE reduces to decision LWE {#sec-apx-crypto-lwe}

### What the chapter uses

Learning with errors hides a secret vector behind linear equations with small errors. Fix a dimension $d$, a modulus $p$ and an error distribution $\chi$ on integers that are small compared with $p$. Write $\mathbb Z_p$ for the integers $0, \dots, p - 1$ with arithmetic mod $p$, and $\mathbb Z_p^d$ for vectors of $d$ such numbers. The secret $s$ is a vector in $\mathbb Z_p^d$. An **LWE sample** is a pair made of a uniform random vector $a$ in $\mathbb Z_p^d$ and a number $b$. To make $b$, draw an error $e$ from $\chi$ and set
$$b = \langle a, s\rangle + e \bmod p, \qquad \langle a, s\rangle = \sum_{j=1}^d a_j s_j.$$
Every sample has its own fresh $a$ and $e$, independent of the other samples, and the secret is the same in all of them.

**Search LWE** asks for $s$, given as many samples as the algorithm needs. **Decision LWE** asks whether $m$ given pairs are LWE samples or uniform pairs in $\mathbb Z_p^d \times \mathbb Z_p$. The chapter shows that search is at least as hard as decision. For the converse it says only this: Regev, who introduced LWE, proves it for a prime $p$ that is at most polynomial in $d$, by guessing the coordinates of $s$ one at a time and testing each guess with the decision algorithm.

Here is that converse in a form this appendix proves. Assume that $p$ is prime. Assume also that a decision algorithm $D$, given $m$ pairs, has advantage at least $\varepsilon$ for every secret, where the probabilities are over the pairs and over $D$'s own random choices:
$$\big|\Pr[D(m \text{ LWE samples with secret } s) = 1] - \Pr[D(m \text{ uniform pairs}) = 1]\big| \ge \varepsilon \quad\text{for every } s \in \mathbb Z_p^d.$$
Then, for any failure probability $\beta$, there is a search algorithm that finds all of $s$ with probability at least $1 - \beta$. It calls $D$ at most $(dp + 1)N$ times and uses $dpNm$ LWE samples, where the number of repetitions per test is
$$N = \left\lceil \frac{8}{\varepsilon^2}\ln\frac{2(dp + 1)}{\beta} \right\rceil.$$ {#eq-apx-crypto-repetitions}
The advantage $\varepsilon$ is **noticeable** if it is at least one over some polynomial in $d$. In that case, and when $p$, $m$ and $D$'s running time are also polynomial in $d$, the search runs in polynomial time. The value of $N$ comes from step 6 of the derivation.

### The idea

The test for one guess works like this. Take an LWE sample and a guess $g$ for the first coordinate $s_1$. Pick a random number $r$, add it to the first coordinate of $a$, and add $r g$ to $b$. If the guess is right, the change to $b$ is exactly the change to $\langle a, s\rangle$. So the new pair is again an LWE sample with the same secret and the same small error. If the guess is wrong, the error picks up the extra term $r(g - s_1)$. When $p$ is prime, that term is a random multiple of a nonzero number, so it is uniform. It drowns the error, and together with the shifted first coordinate it makes the whole pair uniform. Feeding the new pairs to $D$ therefore asks $D$ "LWE or uniform?", and the true answer is "LWE" exactly when the guess is right. Repeating the test many times measures how often $D$ says 1, and comparing that rate with $D$'s rate on uniform pairs, which the search draws itself, tells the right guess from the wrong ones. Trying all $p$ guesses for each of the $d$ coordinates recovers $s$.

@tbl-apx-crypto-small makes the wrong-guess case concrete with $p = 5$, $s_1 = 3$ and error $e = 1$. After the change, the new $b$ minus the inner product of the new $a$ with $s$ is $e + r(g - s_1) \bmod 5$. With the right guess $g = 3$ it stays at $e = 1$ for every $r$. With the wrong guess $g = 1$ it becomes $1 - 2r \bmod 5$, which takes each of the five values exactly once as $r$ runs through $0, \dots, 4$. Since $r$ is uniform, the result is uniform.

| $r$ | 0 | 1 | 2 | 3 | 4 |
| --- | --- | --- | --- | --- | --- |
| Right guess $g = 3$: $1 + 0 \cdot r$ | 1 | 1 | 1 | 1 | 1 |
| Wrong guess $g = 1$: $1 - 2r \bmod 5$ | 1 | 4 | 2 | 0 | 3 |

: The value $b - \langle a, s\rangle$ after re-randomizing with guess $g$, for $p = 5$, $s_1 = 3$, $e = 1$ and each value of $r$. The wrong guess hits every element of $\mathbb Z_5$ once. {#tbl-apx-crypto-small}

### Derivation

Fix a coordinate $\ell$ and a guess $g \in \mathbb Z_p$. Steps 1 to 4 show what one re-randomized sample looks like, and steps 5 to 8 turn this into a search with a failure probability you choose.

1. **Re-randomize.** Take an LWE sample $(a, b)$ with $b = \langle a, s\rangle + e$. Draw a fresh $r$ uniform in $\mathbb Z_p$, independent of everything else. Let $a'$ be $a$ with $r$ added to its $\ell$-th coordinate, and let $b' = b + rg \bmod p$. Adding $r$ to $a_\ell$ adds $r s_\ell$ to the inner product, so $\langle a', s\rangle = \langle a, s\rangle + r s_\ell$. Subtracting this from $b'$ leaves the **residual**
   $$b' - \langle a', s\rangle = e + r(g - s_\ell) \bmod p.$$ {#eq-apx-crypto-residual}

2. **The new vector is still uniform.** For each fixed $r$, adding $r$ to one coordinate is a bijection of $\mathbb Z_p^d$. By the chapter's Fact 4, a bijection maps the uniform distribution to itself. Since $a$ is uniform and independent of $r$ and $e$, the vector $a'$ is uniform and independent of $e$.

3. **A right guess gives an LWE sample.** If $g = s_\ell$, the residual @eq-apx-crypto-residual is just $e$, so $b' = \langle a', s\rangle + e$. With step 2, $a'$ is uniform and $e$ is an independent draw from $\chi$, so $(a', b')$ is an LWE sample with the same secret $s$.

4. **A wrong guess gives a uniform pair.** Now let $c = g - s_\ell \ne 0$. This is where $p$ must be prime. Every nonzero number mod a prime has an inverse; for example, $2 \cdot 3 = 6 \equiv 1 \pmod 5$. Fix the error $e$ and write $v = e + rc$ for the residual. The map that sends $(a, r)$ to $(a', v)$ can be undone. From the residual you recover $r = (v - e)c^{-1}$, and then you recover $a$ by subtracting $r$ from coordinate $\ell$ of $a'$. So this map is a bijection of $\mathbb Z_p^d \times \mathbb Z_p$. The pair $(a, r)$ is uniform there and independent of $e$, so by Fact 4 the pair $(a', v)$ is uniform, whatever $e$ is. Finally, $b' = \langle a', s\rangle + v$. The map from $(a', v)$ to $(a', b')$ is another bijection, undone by subtracting $\langle a', s\rangle$ from $b'$. So by Fact 4 once more, $(a', b')$ is uniform on $\mathbb Z_p^d \times \mathbb Z_p$.

5. **The test feeds $D$ exactly one of its two worlds.** Each re-randomized pair uses its own fresh sample and its own $r$, so $m$ of them are independent. By steps 3 and 4, they are $m$ LWE samples with secret $s$ when the guess is right, and $m$ uniform pairs when it is wrong. Write $\alpha_{\mathrm{LWE}}$ and $\alpha_{\mathrm U}$ for the probabilities that $D$ outputs 1 in these two cases. Then a run of the test outputs 1 with probability $\alpha_{\mathrm{LWE}}$ for the right guess and $\alpha_{\mathrm U}$ for every wrong one, and $|\alpha_{\mathrm{LWE}} - \alpha_{\mathrm U}| \ge \varepsilon$ by assumption.

6. **Estimate the rates.** The search does not know $\alpha_{\mathrm{LWE}}$ or $\alpha_{\mathrm U}$, so it estimates them by repetition. For the baseline, it draws $m$ uniform pairs itself, runs $D$ on them, and repeats $N$ times. The fraction of 1s is its estimate $\hat\alpha_{\mathrm U}$. For each coordinate $\ell$ and guess $g$, it repeats the test $N$ times with fresh samples and fresh $r$, and the fraction of 1s is $\hat\alpha_{\ell,g}$. Each estimate is an average of $N$ independent 0/1 values, so it concentrates around its mean. Hoeffding's inequality, from [Concentration inequalities](../part01-prerequisites/02-concentration.md), says that such an average lands at least $\tau$ away from its mean with probability at most $2e^{-2N\tau^2}$. Step 7 needs every estimate within $\varepsilon/4$, so take $\tau = \varepsilon/4$, which gives $2e^{-N\varepsilon^2/8}$. There are $dp + 1$ estimates in all. The union bound says that the chance that at least one of several events happens is at most the sum of their chances. So some estimate is off by $\varepsilon/4$ or more with probability at most
   $$2(dp + 1)\,e^{-N\varepsilon^2/8} \le \beta,$$
   Taking logarithms, this holds when $N\varepsilon^2/8 \ge \ln(2(dp + 1)/\beta)$, which is @eq-apx-crypto-repetitions.

7. **Decide each guess.** The search keeps the guess $g$ for coordinate $\ell$ when $|\hat\alpha_{\ell,g} - \hat\alpha_{\mathrm U}| > \varepsilon/2$. Suppose every estimate is within $\varepsilon/4$ of its mean, which happens with probability at least $1 - \beta$. For the right guess, the true gap is at least $\varepsilon$ and the two estimation errors together are less than $\varepsilon/2$, so the measured gap is more than $\varepsilon/2$ and the guess is kept. For a wrong guess the true gap is 0, so the measured gap is less than $\varepsilon/2$ and the guess is dropped. So exactly one guess survives for each coordinate, and it is $s_\ell$.

8. **Count the cost.** The search calls $D$ $N$ times for the baseline and $N$ times for each of the $dp$ pairs of coordinate and guess, which is $(dp + 1)N$ calls. Each test call uses $m$ fresh LWE samples, so the search uses $dpNm$ samples. By @eq-apx-crypto-repetitions, $N$ grows like $1/\varepsilon^2$ and only like the logarithm of $dp/\beta$. So the cost is polynomial in $d$ when $p$, $m$ and $1/\varepsilon$ are. Even a failure probability as small as $\beta = 2^{-d}$ is affordable, because then $\ln(1/\beta) = d\ln 2$ is only linear in $d$.

Each assumption has its place. Primality is used in step 4, a polynomial $p$ in step 8 (the search tries every element of $\mathbb Z_p$), a noticeable $\varepsilon$ in step 8 as well, and "for every secret" in step 5. The coordinates are tested independently of one another, which is the "one at a time" of the chapter.

### Checking it

The script first checks steps 3 and 4 directly, with $d = 4$, $p = 97$ and errors drawn from a normal distribution with standard deviation 3, rounded to the nearest integer. It re-randomizes 20,000 samples with the right guess for $s_1$ and with a guess that is off by 1. @fig-apx-crypto-lwe-residuals shows the residuals. With the right guess they form the narrow bump of the error, and 0.760 of them lie within 3 of zero, against the exact 0.757 for the rounded normal. With the wrong guess they are flat, and 0.071 of them lie within 3 of zero. A uniform residual lands on each of the 7 values from $-3$ to 3 with probability $1/97$, so the exact share is $7/97 = 0.072$.

The third panel shows why the modulus must be prime. With $p = 96$ and a guess off by 48, the term $48r \bmod 96$ is only ever 0 or 48, because $48r$ is a multiple of 96 when $r$ is even. So the residual is the error, or the error shifted by half the modulus, and it forms two bumps instead of a flat line. A decision algorithm that looks for small residuals would accept this wrong guess about half as often as the right one, and the test of step 7 could keep it.

![Histograms of the residual $b' - \langle a', s\rangle$ after re-randomizing 20,000 LWE samples ($d = 4$, rounded normal errors with standard deviation 3). Left: prime $p = 97$ and the right guess, which keeps the error's bump. Middle: $p = 97$ and a guess off by 1, which is uniform. Right: composite $p = 96$ and a guess off by 48, which leaves two bumps.](figures/apx_crypto_lwe_residuals.svg){#fig-apx-crypto-lwe-residuals}

It then runs the whole search. A real decision algorithm for these parameters would be hard to write, so the script uses a stand-in. The stand-in sees one pair per call ($m = 1$) and is built with the secret, but the search treats it as a black box and never reads $s$. On 90% of calls it ignores its input and answers with a fair coin flip. On the other 10% it says 1 when the residual is within 3 of zero, which happens with probability 0.7567 for an LWE sample and $7/97$ for a uniform pair. Its advantage is therefore $\varepsilon = 0.1 \times (0.7567 - 7/97) = 0.0684$. With $\beta = 0.01$, @eq-apx-crypto-repetitions asks for $N = 19{,}230$ repetitions per test, which makes 7,480,470 calls to the stand-in in total.

The search recovers the secret $(6, 27, 27, 28)$ exactly. The smallest gap for a right guess is 0.0708, and the largest gap for a wrong one is 0.0166, both measured against the threshold $\varepsilon/2 = 0.0342$. @fig-apx-crypto-lwe-search shows a second run with another secret. Run on ten more random secrets, the search recovers every one of them.

![Acceptance rate of the stand-in decision algorithm on re-randomized samples, for every guess $g$ of each of the four coordinates of a secret in $\mathbb Z_{97}^4$. Grey dots are wrong guesses, the red dot is the right one, the black line is the rate on uniform pairs and the shaded band is $\pm\varepsilon/2$ around it.](figures/apx_crypto_lwe_search.svg){#fig-apx-crypto-lwe-search}

The bound behind @eq-apx-crypto-repetitions is conservative. It asks every estimate to be within $\varepsilon/4$ with probability $1 - \beta$ at once, but in practice the largest gap for a wrong guess, 0.0166, is only about half the threshold. The right guess clears the threshold with room to spare too. A search that stops testing once a guess is found, or that adds repetitions only for guesses near the threshold, would need fewer calls.

### What it does not cover

The proof assumes that $D$ works for every secret. Some decision algorithms work only on average over a random secret. The first step in handling them is to make the secret random: pick a uniform vector $u$ and replace each $b$ by $b + \langle a, u\rangle$, which gives samples with the random secret $s + u$. But for a given $u$ the algorithm may still fail, so the search must try several $u$ and check which ones work. Regev (2009) carries this out, and those details stay cited. The proof also assumes that a lower bound $\varepsilon$ on $D$'s advantage is known. Without one, the search can try $\varepsilon = 1/2, 1/4, 1/8, \dots$ in turn, which multiplies the cost by a logarithmic factor.

For moduli that are not prime, or are too large to try every guess, other reductions exist. Peikert (2009) handles moduli that are products of distinct small primes, and Micciancio & Mol (2011) give reductions that use no more samples than the decision algorithm, for a wide class of moduli. Brakerski, Langlois, Peikert, Regev & Stehlé (2013) show that decision LWE is hard for any modulus by relating moduli to one another rather than by guessing. These stay cited, because each needs lattice and number theory well beyond this appendix.

## Code

The script [apx_cryptography.py](../../code/appendix-derivations/apx_cryptography.py) computes every number on this page and draws its figures. For the GGM tree it builds the toy PRG, runs the adaptive adversary in each hybrid, runs each reduction $B_j$ on outputs of $G$ and on uniform strings, and measures the single-sample distinguisher of step 7. For LWE it prints @tbl-apx-crypto-small, re-randomizes samples with right and wrong guesses, and runs the full search with the stand-in decision algorithm. From the repository root:

```bash
uv run python code/appendix-derivations/apx_cryptography.py
```

| Quantity | Value |
| --- | --- |
| GGM: exact $h_0, h_1, h_2, h_3$ | 0.0000, 0.4980, 0.4980, 0.9961 |
| GGM: single-sample $B$, rate on $G(U_\lambda)$ and on $U_{2\lambda}$ | 0.5535, 0.6641 |
| GGM: advantage of $B$, measured and predicted $\mathrm{Adv}_{\mathrm{PRF}}(A)/(nq)$ | 0.1105, 0.1107 |
| LWE: residual within 3 of zero, right guess (exact) | 0.760 (0.757) |
| LWE: residual within 3 of zero, wrong guess (exact) | 0.071 (0.072) |
| LWE search: $\varepsilon$, repetitions $N$, calls to the stand-in | 0.0684, 19,230, 7,480,470 |
| LWE search: smallest right gap, largest wrong gap, threshold $\varepsilon/2$ | 0.0708, 0.0166, 0.0342 |
| LWE search: random secrets recovered exactly | 10 of 10 |

: The numbers the script prints (seed 0; 200,000 runs per GGM entry, 20,000 samples for the residuals). {#tbl-apx-crypto-code}

Two things are worth taking from these numbers. In the GGM check, the reductions recreate the hybrids exactly even against an adaptive adversary, so the factor $nq$ is lost only in the averaging of steps 6 and 7. In the LWE check, re-randomizing turns a right guess into LWE samples and a wrong one into exactly uniform pairs, and the only real cost is the number of repetitions needed to see a weak decision algorithm's small advantage.

## References

- Goldreich, Goldwasser & Micali, "How to Construct Random Functions", *Journal of the ACM*, 1986. The GGM construction and its original proof.
- Goldreich, *Foundations of Cryptography*, Vol. 1: *Basic Tools*, 2001, §3.6. The careful textbook proof of the GGM theorem, with the hybrids over the levels of the tree.
- Katz & Lindell, *Introduction to Modern Cryptography*, 3rd ed., 2020. A readable proof of the GGM theorem, in its chapter on theoretical constructions of symmetric-key primitives.
- Bellare & Rogaway, "The Security of Triple Encryption and a Framework for Code-Based Game-Playing Proofs", *EUROCRYPT*, 2006. Game-playing proofs, including lazy sampling of random tables.
- Regev, "On Lattices, Learning with Errors, Random Linear Codes, and Cryptography", *Journal of the ACM*, 2009 (conference version 2005). Defines LWE, and its search-to-decision lemma is the reduction proved here, including the step for decision algorithms that work only on average.
- Peikert, "Public-Key Cryptosystems from the Worst-Case Shortest Vector Problem", *STOC*, 2009. Search-to-decision for moduli that are products of small primes.
- Micciancio & Mol, "Pseudorandom Knapsacks and the Sample Complexity of LWE Search-to-Decision Reductions", *CRYPTO*, 2011. Search-to-decision reductions that preserve the number of samples, for a wide class of moduli.
- Brakerski, Langlois, Peikert, Regev & Stehlé, "Classical Hardness of Learning with Errors", *STOC*, 2013. Hardness of decision LWE for any modulus.
- Hoeffding, "Probability Inequalities for Sums of Bounded Random Variables", *Journal of the American Statistical Association*, 1963. The concentration bound behind the number of repetitions.
