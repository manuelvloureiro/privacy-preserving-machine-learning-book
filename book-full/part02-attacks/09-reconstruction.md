# Reconstruction attacks

## Why this matters

A hospital keeps one sensitive bit per patient: whether they tested positive. It never releases the bits. It only answers counting questions: "how many of patients 3, 17, 42, … tested positive?" To be safe, it adds a little noise to every answer. Is that enough? The answer, due to Dinur and Nissim (2003), is a hard no unless the noise is large. If every answer is within $o(\sqrt n)$ of the truth, where $n$ is the number of patients, an attacker who asks a number of random questions only slightly larger than $n$ can recover almost every bit by solving a linear program. If the attacker may ask all $2^n$ questions, an error of $o(n)$ is already too small. (Here $o(\sqrt n)$ means an amount that grows more slowly than $\sqrt n$, so that its ratio to $\sqrt n$ goes to 0 as $n$ grows; $O(\cdot)$, used below, means "at most a constant times".) A table of aggregates is a system of equations whose unknowns are the people in it, and enough equations determine the unknowns.

This is the **reconstruction game** of [Threat models and privacy games](01-threat-models.md). In that game the secret is the whole dataset (or one column of it), and the adversary is scored by the fraction of entries it recovers, compared with the baseline of guessing from the population, which for fair bits is $\frac12$. That chapter showed that one exact count of $n$ fair bits lets the adversary recover only about $\frac12 + 1/\sqrt{2\pi n}$ of them: knowing the total tells you which value is the majority, and nothing about who holds it. This chapter shows what happens with many counts.

The one idea that organizes the chapter is this. **Every answer is a linear equation in the secret bits, and the error of the answers is the only thing standing between the attacker and the solution.** To hide a wrong guess, the curator's error has to be as large as the amount by which that guess changes the answers. For a random query that amount is about $\sqrt n$ when the guess is wrong on a constant fraction of the bits, and for the query aimed at exactly the wrong bits it is about $n$; errors well below those scales cannot hide the solution. Errors that are independent from answer to answer, and do not grow with the number of queries, are averaged away by asking more. Dwork and Roth call this the **Fundamental Law of Information Recovery**: overly accurate answers to too many questions destroy privacy.

@tbl-recon-uses lists the results and where they are used.

| Result | Where it is used |
| --- | --- |
| Exponential attack: error $E$ on all $2^n$ subset sums leaves at most $4E$ wrong bits | Why per-answer error must be of order $n$ when queries are unlimited |
| Anti-concentration of a random subset sum (Paley–Zygmund) | The step that makes a random query informative about every bad candidate |
| Linear-programming decoder: error $o(\sqrt n)$ on $O(n\log n)$ random queries leaves $o(n)$ wrong bits | The Dinur–Nissim attack; the $\sqrt n$ threshold |
| Least squares under worst-case and random error | The attack you would actually run; how noise and the number of queries trade off |
| Sampling curator, via Hoeffding and the union bound | Error of order $\sqrt{n\log m}$ can stop reconstruction, yet still be far from private |
| Information bound, via Fano | No attack can succeed with much fewer than $n/\log n$ queries |
| Census block reconstruction by constraint search | Why publishing many overlapping tables reveals the microdata |

: The results of this chapter and where each is used. {#tbl-recon-uses}

**How to read this chapter.** Read the toolkit, then the exponential attack, which is short and exact. The linear-programming section is the heart of the chapter; on a first pass read its statement, the lemma and the figure, and come back for the union-bound steps. The least-squares section is what the script runs. The Fundamental Law section puts the results together with two converses, and the census section shows the same idea on real-looking tables.

The concentration tools (Hoeffding's inequality, the union bound) are in [Concentration inequalities](../part01-prerequisites/02-concentration.md). Entropy, mutual information, the data processing inequality (DPI) and Fano's inequality are in [Information theory](../part01-prerequisites/05-information-theory.md). Each is restated below where it is used. No privacy definition is used: [Part 3](../part03-dp-foundations/01-definition.md) introduces the one that defends against these attacks, and [Lower bounds](../part03-dp-foundations/09-lower-bounds.md) reads this chapter's results as lower bounds on the error any defence must add.

**Conventions.** The secret database is a bit vector $d = (d_1, \dots, d_n) \in \{0, 1\}^n$, one bit per person. A **subset-sum query** is a set $S \subseteq [n] = \{1, \dots, n\}$ of people, and it asks how many of them have bit 1. It is written as its 0/1 indicator vector $q$, with $q_i = 1$ when person $i$ is in $S$; its true answer is $\sum_{i \in S}d_i = \langle q, d\rangle$. The attacker asks $m$ queries, the rows of an $m \times n$ matrix $A$. The **curator** (the party holding $d$) returns $a = Ad + e$, where $e$ is its error, one entry per query. "Error at most $E$" means $|e_j| \le E$ for every query $j$. $\beta$ is a fraction of wrong bits, $\eta$ a failure probability and $\sigma$ the standard deviation of random noise. Logarithms are natural unless written $\log_2$. [Threat models and privacy games](01-threat-models.md) wrote $m$ for the number of bits; here $n$ is the number of bits and $m$ the number of queries.

## Toolkit: three facts every derivation uses

### Fact 1 — Subset sums are inner products

A subset sum adds up the bits of the people in the subset, so it is the inner product of the subset's indicator vector with the database: $\langle q, d\rangle = \sum_i q_id_i$, which is linear in $d$. Stacking $m$ queries as the rows of $A$ gives $a = Ad + e$: a linear system with $n$ unknowns (the bits) and $m$ equations, each off by its error. @tbl-recon-subsets shows a few of the $2^4 = 16$ subset sums of the database $d = (1, 0, 1, 1)$.

| Subset $S$ | $\{1\}$ | $\{2\}$ | $\{1, 2\}$ | $\{3, 4\}$ | $\{2, 3, 4\}$ | $\{1, 2, 3, 4\}$ |
| --- | --- | --- | --- | --- | --- | --- |
| Indicator $q$ | 1000 | 0100 | 1100 | 0011 | 0111 | 1111 |
| $\langle q, d\rangle$ | 1 | 0 | 1 | 2 | 2 | 3 |

: Six subset sums of $d = (1, 0, 1, 1)$. Each is a linear equation in the four secret bits. {#tbl-recon-subsets}

With exact answers there is nothing to attack. The singletons $\{1\}, \dots, \{4\}$ alone reveal $d$ exactly, one bit per answer, and any $n$ queries whose indicator vectors are linearly independent do the same, because Gaussian elimination solves the system. Everything interesting in this chapter is about answers with error.

Linearity also tells you how an attacker rules out a wrong guess. If the attacker considers a candidate database $c$, the answers $c$ would produce differ from the true ones by $\langle q, c\rangle - \langle q, d\rangle = \langle q, c - d\rangle$, the subset sum of the difference vector $c - d$. If, for some query, that difference is larger than the curator's error could explain, the candidate is ruled out. Every attack below is a way of finding such a query for every bad candidate.

### Fact 2 — Moments of a random-sign sum

A random query includes each person with probability $\frac12$, so its answer behaves like a sum of terms that are switched on or off by fair coins. The cleanest form of such a sum uses signs instead of on/off switches. Let $s_1, \dots, s_n$ be independent fair signs ($+1$ or $-1$ with probability $\frac12$ each) and $X = \sum_is_ia_i$ for fixed numbers $a_i$. The sum is centred at 0, and its spread is set by $\sum_ia_i^2$, because the variances of independent terms add. What the later proofs need is that $X$ has no heavy tails: its fourth moment is at most 3 times the square of its second moment, the same ratio as for a normal variable. In symbols,
$$\mathbb E X = 0, \qquad \mathbb E X^2 = \sum_ia_i^2, \qquad \mathbb E X^4 = 3\Big(\sum_ia_i^2\Big)^2 - 2\sum_ia_i^4 \le 3\Big(\sum_ia_i^2\Big)^2.$$

The proof is a count of which terms survive when you take expectations. Expand $X^4 = \sum_{i,j,k,l}s_is_js_ks_la_ia_ja_ka_l$. A term in which some index appears an odd number of times (once or three times) contains a factor $\mathbb E s_i = 0$ or $\mathbb E s_i^3 = 0$, and vanishes, because the signs are independent and $s_i^3 = s_i$ has mean 0. Two kinds of terms survive. If all four indices are equal, $s_i^4 = 1$, and these terms add up to $\sum_ia_i^4$. If the indices form two distinct pairs, each $s_i^2 = 1$; the pairs can be matched as $i = j \ne k = l$ or in two other ways, so these terms add up to $3\sum_{i \ne k}a_i^2a_k^2 = 3\big((\sum_ia_i^2)^2 - \sum_ia_i^4\big)$. Adding the two kinds gives the formula, and dropping the negative term $-2\sum_ia_i^4$ gives the bound. The second moment is the same computation with two indices: only the terms with $i = j$ survive. The mean is 0, and every odd moment is 0, because flipping all the signs turns $X$ into $-X$ without changing its distribution.

*A concrete case.* Take $a = (1, 1, 1)$. Of the eight sign patterns, two give $X = \pm3$ (all signs equal) and six give $X = \pm1$, so $\mathbb E X^4 = (2 \cdot 81 + 6 \cdot 1)/8 = 21$. The formula gives $3 \cdot 9 - 2 \cdot 3 = 21$, and the bound $3 \cdot 9 = 27$.

This is how the fact is used. A random subset has $q_i = (1 + s_i)/2$: the person is in when the sign is $+1$ and out when it is $-1$. So a random subset sum $\sum_iq_iz_i$ is the constant $\frac12\sum_iz_i$ plus half a random-sign sum, and its fourth moment is at most 3 times its squared second moment. That ratio is what Fact 3 needs.

### Fact 3 — Paley–Zygmund: a variable with a light fourth moment is not concentrated near 0

Markov's inequality ([Concentration inequalities](../part01-prerequisites/02-concentration.md)) says that a non-negative variable is rarely *far above* its mean. The attacks need the opposite: that a variable is not too often *far below* its mean, so that a random query does not usually give a tiny answer. That cannot hold for every variable. A variable that is 0 most of the time and huge once in a while has a large mean and is still usually 0. What rules this out is a second moment that is not much larger than the squared mean, since a rare huge value makes $\mathbb E[W^2]$ enormous. The **Paley–Zygmund inequality** makes this precise. For a random variable $W \ge 0$ with finite second moment and $0 \le \theta < 1$,
$$\Pr\big[W > \theta\,\mathbb E W\big] \ge (1 - \theta)^2\frac{(\mathbb E W)^2}{\mathbb E[W^2]}.$$ {#eq-recon-pz}
A lower bound on how often a variable is not close to zero is called **anti-concentration**.

The proof splits the mean $\mathbb E W$ into the part that comes from the event $B = \{W > \theta\,\mathbb E W\}$ and the part that comes from its complement: $\mathbb E W = \mathbb E[W\mathbf 1_{B^c}] + \mathbb E[W\mathbf 1_B]$. On the complement $W \le \theta\,\mathbb E W$, so the first part is at most $\theta\,\mathbb E W$. For the second part, use the Cauchy–Schwarz inequality $\mathbb E[UV] \le \sqrt{\mathbb E[U^2]\,\mathbb E[V^2]}$ with $U = W$ and $V = \mathbf 1_B$; since $\mathbf 1_B^2 = \mathbf 1_B$ has mean $\Pr[B]$, this gives $\mathbb E[W\mathbf 1_B] \le \sqrt{\mathbb E[W^2]\Pr[B]}$. Together, $\mathbb E W \le \theta\,\mathbb E W + \sqrt{\mathbb E[W^2]\Pr[B]}$, that is, $(1 - \theta)\mathbb E W \le \sqrt{\mathbb E[W^2]\Pr[B]}$. Squaring both sides and dividing by $\mathbb E[W^2]$ gives @eq-recon-pz.

The linear-programming decoder uses this with $W$ the square of a random subset sum: it needs a random query to move a bad candidate's answer by a lot with constant probability, and Fact 2 supplies the light fourth moment.

## The exponential attack {#sec-recon-exponential}

### Context

Start with the strongest attacker: it may ask every one of the $2^n$ subset sums. How large must the curator's error be before the attacker cannot pin $d$ down? The answer is linear in $n$, and the proof is three lines.

### Statement

The attacker cannot know $d$, but it can throw away every candidate that contradicts an answer. Call a candidate $c \in \{0, 1\}^n$ **consistent** if each of its subset sums is within $E$ of the published answer, $\big|a_S - \sum_{i \in S}c_i\big| \le E$ for every $S$; the true $d$ is always consistent. The attacker outputs any consistent candidate. The reason it cannot be far from $d$ is that the bits where $c$ is wrong form two subsets, the ones it missed and the ones it invented, and the attacker has asked about both. On each of them $c$ and $d$ disagree by the size of the subset, while both have to be within $E$ of the same answer, so each subset has at most $2E$ people. The number of wrong bits, the **Hamming distance** between $c$ and $d$, therefore satisfies
$$\mathrm{dist}(c, d) = \#\{i : c_i \ne d_i\} \le 4E.$$ {#eq-recon-exp}
The bound is tight: if $d$ has at least $\lfloor 2E\rfloor$ ones and $\lfloor 2E\rfloor$ zeros, some curator with error at most $E$ makes a candidate at distance $2\lfloor 2E\rfloor$ consistent.

So a curator with error $E = o(n)$ is **blatantly non-private** in the sense of Dinur and Nissim: the attacker recovers all but $o(n)$ of the bits.

### Derivation

1. **A consistent candidate exists**, because $d$ itself is one: the true answers are within $E$ of $d$'s sums by assumption.
2. **Count the ones that the candidate missed.** Let $S_0 = \{i : d_i = 1, c_i = 0\}$. On $S_0$, $d$ sums to $|S_0|$ and $c$ sums to 0. Both sums are within $E$ of the one answer $a_{S_0}$, so by the triangle inequality they are within $2E$ of each other: $|S_0| = |S_0| - 0 \le 2E$.
3. **Count the zeros that the candidate invented.** Let $S_1 = \{i : d_i = 0, c_i = 1\}$. Now $c$ sums to $|S_1|$ and $d$ to 0 on $S_1$, and the same argument gives $|S_1| \le 2E$.
4. **Add.** Every disagreement is in $S_0$ or in $S_1$, so $\mathrm{dist}(c, d) = |S_0| + |S_1| \le 4E$.

**Tightness.** To show a curator can really be this misleading, build a bad candidate and a curator that covers for it. Let $k = \lfloor 2E\rfloor$ and let $c$ flip $k$ of the ones of $d$ to 0 and $k$ of the zeros to 1, so $c$ is at distance $2k$. For any $S$, $\sum_S(c_i - d_i)$ is the number of flipped zeros in $S$ minus the number of flipped ones in $S$, so it lies in $[-k, k]$: the two databases' sums are never more than $k$ apart. A curator that answers the midpoint $a_S = \frac12\big(\sum_Sc_i + \sum_Sd_i\big)$ is then within $k/2 \le E$ of both, so its error is at most $E$ and $c$ is consistent.

### Interpretation

The proof uses only two queries, $S_0$ and $S_1$: the people the candidate got wrong in each direction. Small, well-aimed subsets do all the work. But the attacker does not know which people it got wrong, so it does not know these two subsets in advance, and that is why it needs all $2^n$ answers.

This is not an efficient attack. The attacker checks $2^n$ candidates against $2^n$ answers. The next section replaces "all subsets" by a few times $n$ random ones and "check every candidate" by a linear program. The price is that a random subset cannot aim at the wrong bits the way $S_0$ and $S_1$ do, so the error must be $o(\sqrt n)$ instead of $o(n)$.

The bound is a worst case over curators. It holds for every curator, including one that picks its errors to fool this attacker. A specific curator can be much weaker than the bound suggests. One that rounds every answer to a multiple of 4 has error at most 2, but its rounding is a known function of the true sum, and an attacker who simulates the rounding on each candidate recovers $d$ exactly ([Exercise 1](../../book-exercises/part02-attacks/09-reconstruction.md#sec-sol-reconstruction-1)).

### Example: ten people, all 1024 subsets

The script draws $d = (1, 1, 1, 0, 0, 0, 0, 0, 0, 1)$, with four ones, and checks all $2^{10}$ candidates against all $2^{10}$ subsets. It needs a test for whether a candidate $c$ can be made consistent by *some* curator with error at most $E$. The tightness argument gives it: if every subset sum of $c - d$ is at most $2E$ in absolute value, the midpoint curator has error at most $E$ and keeps $c$; if some subset sum of $c - d$ exceeds $2E$, no single answer can be within $E$ of both $c$'s sum and $d$'s, so no curator keeps $c$. The script counts the candidates that pass the test (@tbl-recon-exp).

| Error $E$ | 0 | 0.5 | 1 | 1.5 | 2 |
| --- | --- | --- | --- | --- | --- |
| Candidates that survive | 1 | 35 | 242 | 630 | 912 |
| Largest distance to $d$ | 0 | 2 | 4 | 6 | 8 |
| Bound $4E$ | 0 | 2 | 4 | 6 | 8 |

: The exponential attack on $n = 10$ bits. Every surviving candidate is within $4E$ of $d$, and the bound is attained. {#tbl-recon-exp}

The bound is attained at every $E$, as the tightness argument predicts: that argument needs $\lfloor 2E\rfloor$ ones and $\lfloor 2E\rfloor$ zeros to flip, at most 4 of each here, and $d$ has four ones and six zeros. At $E = 2$, 912 of the 1024 candidates survive, so the attacker learns little: the bound $4E = 8$ is close to $n = 10$, and a guarantee of at most 8 wrong bits out of 10 says almost nothing. @fig-recon-exponential draws both rows over a finer grid of $E$, for another random database of ten bits, so its counts can differ from those of @tbl-recon-exp.

![The exponential attack on ten bits. Left: the largest Hamming distance between the true database and a candidate that some curator with error $E$ can make consistent with all 1024 subset sums, against the bound $4E$. Right: the number of surviving candidates (log scale); at $E = 0$ only $d$ survives.](figures/recon_exponential.svg){#fig-recon-exponential}

### Check yourself

- Why does the derivation not need the curator's errors to be random? *(It only uses $|e_S| \le E$ for the two subsets $S_0$ and $S_1$. The errors can be chosen by an adversarial curator, even after seeing the attacker's candidate.)*
- With $n = 10^6$ people and every subset sum answered with error at most 1000, how many bits can the attacker get wrong? *(At most $4E = 4000$, or 0.4% of the bits, although each answer has error up to 1000.)*

## The linear-programming decoder {#sec-recon-lp}

### Context

Asking $2^n$ questions is not realistic. Dinur and Nissim showed that a number of *random* subsets only a polylogarithmic factor above $n$ suffices, with an efficient decoder: their proof asks $n(\log n)^2$ of them, and the version proved below needs $O(n\log n)$. Instead of checking every bit vector, the decoder lets each unknown be any number between 0 and 1, finds a fractional vector consistent with the answers by linear programming (LP), and rounds it. A linear program finds a point that satisfies a list of linear inequalities, and it can be solved in time polynomial in the number of unknowns and inequalities.

The price is that the error must now be $o(\sqrt n)$ instead of $o(n)$, and the reason is that a random query cannot aim at the wrong coordinates. Suppose a candidate differs from $d$ by a vector with $k$ entries of constant size. A random subset picks up each of those entries with probability $\frac12$, so its sum of the difference behaves like a sum of $k$ coin-flip terms, which fluctuates by about $\sqrt k$, not $k$. An error below $\sqrt k$ cannot hide that fluctuation. The key step is to show that the sum does fluctuate this much with constant probability, whatever the signs of the entries, and that is where the anti-concentration of Fact 3 comes in.

### Statement

Let the $m$ queries be independent uniformly random subsets (each person included with probability $\frac12$). The true $d$ satisfies every answer to within $E$, so the set of vectors that do so is not empty. The **linear-programming decoder** searches that set, after relaxing each bit to the interval $[0, 1]$ so that the search becomes a linear program. It solves the feasibility problem
$$\text{find } c \in [0, 1]^n \text{ with } \big|\langle q_j, c\rangle - a_j\big| \le E \text{ for } j = 1, \dots, m,$$
which is a linear program: each absolute value is two linear inequalities, which makes $2m$ of them, and each $0 \le c_i \le 1$ is a box constraint, one per person. The decoder then rounds each $c_i$ to the nearest bit.

The analysis rests on one lemma about a single random query. Think of $z$ as the difference between a bad candidate and $d$. If $z$ has many coordinates of at least a constant size (here $\frac13$, the size left after a rounding step in the proof), then a random subset sum of $z$ lands far from 0 with probability at least a constant, whatever the signs of the coordinates. "Far" means on the scale $\sqrt{\beta n}$, the square root of the number of large coordinates, as the coin-flip picture above predicts.

**Lemma (one random query catches a bad difference).** Let $z \in \mathbb R^n$ have at least $\beta n$ coordinates with $|z_i| \ge \frac13$, and let $q$ be a uniformly random subset. Then
$$\Pr\left[|\langle q, z\rangle| > \frac{\sqrt{\beta n}}{12}\right] \ge \frac{3}{16}.$$ {#eq-recon-anti}

The theorem turns this into a guarantee for the decoder. A bad candidate survives the linear program only if every one of the $m$ queries fails to catch it. Each query catches it with probability at least $\frac3{16}$, so it survives all $m$ independent queries with probability at most $(13/16)^m$, which shrinks exponentially in $m$. There are many bad candidates, but after rounding them to a grid there are only finitely many, about $(n + 1)^n$. Taking $m$ large enough that $(n + 1)^n(13/16)^m$ falls below a chosen failure probability $\eta$ makes every bad candidate fail at once.

**Theorem (Dinur–Nissim, in the form proved here).** Fix $d$, $\beta$ and a failure probability $\eta$. If the curator's error is at most $E$ with $2E + \frac12 \le \sqrt{\beta n}/12$, and
$$m \ge \frac{n\ln(n + 1) + \ln(1/\eta)}{\ln(16/13)},$$ {#eq-recon-lp}
then with probability at least $1 - \eta$ over the queries, every solution of the linear program rounds to a vector with at most $\beta n$ wrong bits, whatever errors the curator chose.

In words: error $o(\sqrt n)$ on $O(n\log n)$ random subset sums lets the attacker recover all but $o(n)$ bits in polynomial time.

### Derivation

**The lemma.** The plan is to apply Paley–Zygmund (@eq-recon-pz) to $W = Y^2$, where $Y = \langle q, z\rangle$ is the random subset sum. That needs the second and fourth moments of $Y$, and Fact 2 provides them once $Y$ is written as a constant plus a random-sign sum. So write the random subset as $q_i = (1 + s_i)/2$ with independent fair signs $s_i$.

1. **Split the sum into a constant and a random-sign part.** $Y = \langle q, z\rangle = \mu + X$, with the constant $\mu = \frac12\sum_iz_i$ and $X = \frac12\sum_is_iz_i$, a random-sign sum with coefficients $a_i = z_i/2$.
2. **Compute the second moment.** By Fact 2, $\mathbb E X = 0$, and the variance of the random-sign part, written $\tau^2$ so that $\tau$ is its standard deviation, is $\tau^2 := \mathbb E X^2 = \sum_ia_i^2 = \frac14\sum_iz_i^2$. Expanding $(\mu + X)^2$, the cross term $2\mu\,\mathbb E X$ vanishes, so $\mathbb E Y^2 = \mu^2 + \tau^2$.
3. **Bound the fourth moment by 3 times the squared second moment.** Expand $(\mu + X)^4 = \mu^4 + 4\mu^3X + 6\mu^2X^2 + 4\mu X^3 + X^4$. The terms with $\mathbb E X$ and $\mathbb E X^3$ vanish, since odd moments of a random-sign sum are 0. Fact 2 gives $\mathbb E X^4 \le 3(\sum_ia_i^2)^2 = \frac{3}{16}(\sum_iz_i^2)^2 = 3\tau^4$. So
$$\mathbb E Y^4 = \mu^4 + 6\mu^2\tau^2 + \mathbb E X^4 \le \mu^4 + 6\mu^2\tau^2 + 3\tau^4 \le 3(\mu^2 + \tau^2)^2,$$
where the last step holds because $3(\mu^2 + \tau^2)^2 = 3\mu^4 + 6\mu^2\tau^2 + 3\tau^4$ is larger by $2\mu^4 \ge 0$.
4. **Apply Paley–Zygmund** (@eq-recon-pz) with $W = Y^2$, whose mean is $\mu^2 + \tau^2$. Step 3 says $(\mathbb E Y^2)^2/\mathbb E Y^4 \ge \frac13$, so $\Pr[Y^2 > \theta(\mu^2 + \tau^2)] \ge (1 - \theta)^2(\mathbb E Y^2)^2/\mathbb E Y^4 \ge (1 - \theta)^2/3$.
5. **Drop the mean.** Since $\mu^2 + \tau^2 \ge \tau^2$, the event $Y^2 > \theta(\mu^2 + \tau^2)$ implies $Y^2 > \theta\tau^2$, that is $|Y| > \sqrt\theta\,\tau$. So $\Pr[|Y| > \sqrt\theta\,\tau] \ge (1 - \theta)^2/3$. A non-zero mean only helps: it pushes $Y$ away from 0.
6. **Use the large coordinates to bound $\tau$ from below.** At least $\beta n$ coordinates have $z_i^2 \ge \frac19$, and the others contribute $z_i^2 \ge 0$, so $\tau^2 = \frac14\sum_iz_i^2 \ge \frac14\cdot\frac{\beta n}{9}$ and $\tau \ge \sqrt{\beta n}/6$.
7. **Choose $\theta = \frac14$.** Then $\sqrt\theta\,\tau \ge \frac12\cdot\sqrt{\beta n}/6 = \sqrt{\beta n}/12$, and the probability is at least $(3/4)^2/3 = (9/16)/3 = 3/16$. This gives @eq-recon-anti.

**The theorem.** The idea is to show that a solution with many wrong bits would give a difference vector that no random query catches, and that, with enough queries, no such vector exists.

1. **Consistent candidates differ from $d$ by little on every query.** Both $c$ and $d$ are within $E$ of every answer, so by the triangle inequality $|\langle q_j, c - d\rangle| \le 2E$ for all $j$, as in the exponential attack.
2. **A wrong bit is a large coordinate.** Rounding $c_i$ gives the wrong bit only if $c_i$ is on the far side of $\frac12$ from $d_i$, that is, if $|c_i - d_i| \ge \frac12$. So if the rounding has more than $\beta n$ wrong bits, $z = c - d$ has more than $\beta n$ coordinates of size at least $\frac12$.
3. **Move to a finite set.** The linear program can return any point of the cube, and the union bound below needs a finite list of candidates. Round each $c_i$ to the nearest multiple of $1/n$, giving $c'$. Each coordinate moves by at most $\frac1{2n}$, and a subset sum adds at most $n$ coordinates, so $|\langle q, c - c'\rangle| \le n \cdot \frac{1}{2n} = \frac12$ for every subset $q$. Hence $z' = c' - d$ satisfies $|\langle q_j, z'\rangle| \le 2E + \frac12$ for all $j$, and still has more than $\beta n$ coordinates of size at least $\frac12 - \frac1{2n} \ge \frac13$ (for $n \ge 3$, since $\frac12 - \frac16 = \frac13$).
4. **One grid point survives all queries only with small probability.** Fix one grid point $c'$ whose $z'$ has more than $\beta n$ large coordinates. By the lemma (@eq-recon-anti), each random query has $|\langle q_j, z'\rangle| > \sqrt{\beta n}/12$ with probability at least $\frac3{16}$, and by the assumption $2E + \frac12 \le \sqrt{\beta n}/12$ such a query violates the bound of step 3. So each query lets $z'$ pass with probability at most $1 - \frac3{16} = \frac{13}{16}$. The queries are independent, so $z'$ passes all $m$ of them with probability at most $(13/16)^m$.
5. **Union bound over the grid.** The union bound ([Concentration inequalities](../part01-prerequisites/02-concentration.md), Fact 2) says that the probability that at least one of several events happens is at most the sum of their probabilities. Each coordinate of a grid point takes one of the $n + 1$ values $0, \frac1n, \dots, 1$, so there are at most $(n + 1)^n$ grid points, and the probability that *some* bad grid point passes every query is at most $(n + 1)^n(13/16)^m$. Taking logarithms, this is at most $\eta$ exactly when $n\ln(n + 1) - m\ln(16/13) \le \ln\eta$, that is, when $m\ln(16/13) \ge n\ln(n + 1) + \ln(1/\eta)$, which is @eq-recon-lp.
6. **Conclude.** Outside that event, no bad grid point passes, so by step 3 no linear-program solution with more than $\beta n$ wrong bits exists. The event depends only on the queries, not on the curator's errors, so the guarantee holds for every curator at once.

### Interpretation

The threshold is $\sqrt n$ because a random query cannot aim at the wrong coordinates. Its inner product with a difference vector that has $k$ coordinates of constant size fluctuates on the scale $\sqrt k$, as step 2 of the lemma computes. An error below that scale cannot hide $k = \beta n$ wrong bits. An error of order $\sqrt n$ can, and @sec-recon-law constructs a curator that does.

Among difference vectors, the balanced ones are the hard case. If the wrong coordinates of $z$ all have the same sign, $\langle q, z\rangle$ has a large mean and escapes any small band almost surely. If they cancel, the mean is near 0 and only the fluctuation helps. Step 5 of the lemma covers both cases, because it drops the mean and relies on the fluctuation alone. @fig-recon-anticoncentration shows the two cases.

The constants are poor, but the scaling is right. The theorem asks for $E$ below about $\sqrt{\beta n}/24$, which is what $2E + \frac12 \le \sqrt{\beta n}/12$ gives after dividing by 2, and for $m \approx 33n$ at $n = 1000$ (the example below). The union bound over $(n + 1)^n$ grid points costs the $\ln(n + 1)$ factor. Sharper analyses need only $m = O(n)$ queries. Dwork, McSherry and Talwar (2007) go further with a variant of the decoder: instead of requiring every answer to be within $E$, which a single wildly wrong answer makes impossible, it minimises the total absolute error $\lVert Ac - a\rVert_1$, and it tolerates a constant fraction of answers with arbitrarily large error. In practice, the least-squares decoder of the next section with $m = 4n$ already works at errors of about $0.2\sqrt n$.

Random queries are not essential either. Dinur and Nissim's queries were random to make the proof work. Real attacks use whatever queries the system answers. Cohen and Nissim (2020) ran a linear-programming reconstruction against a commercial query system that added noise to its answers, and recovered most of a secret column.

![A random subset sum of a difference vector $z$ with $\beta n = 100$ entries of size $\frac13$ ($n = 1000$), over 20 000 random subsets, against the band $|\langle q, z\rangle| \le \sqrt{\beta n}/12 = 0.83$ (red). Left: half the entries $+\frac13$ and half $-\frac13$, so the sum is centred at 0, yet it still leaves the band most of the time. Right: all entries $+\frac13$, so the sum is centred far from the band. The lemma guarantees at least $\frac3{16} = 0.1875$ in both cases.](figures/recon_anticoncentration.svg){#fig-recon-anticoncentration}

### Example: what the theorem promises at $n = 1000$

Take $n = 1000$ people and target $\beta = 0.1$, so at most $0.1 \times 1000 = 100$ wrong bits. The lemma's threshold is $\sqrt{\beta n}/12 = \sqrt{100}/12 = 0.8333$. The script estimates the escape probability, the chance that a random subset sum lands outside the band $\pm0.8333$, over 100 000 random subsets for the two difference vectors of @fig-recon-anticoncentration: 0.6161 for the balanced one and 1.0000 for the aligned one, against the guaranteed 0.1875. The lemma is far from tight for both.

The theorem's condition $2E + \frac12 \le \sqrt{\beta n}/12$ then allows an error of $E \le \sqrt{\beta n}/24 - \frac14 = 0.4167 - 0.25 = 0.1667$. With failure probability $\eta = 0.01$, so that $\ln(1/\eta) = \ln 100$, @eq-recon-lp asks for
$$m \ge \frac{1000\ln1001 + \ln100}{\ln(16/13)} = 33\,295.04,$$
so at least 33 296 queries, about $33n$. Both constants are far from what the attack needs in practice, but they come with a complete proof.

### Check yourself

- Why does the derivation round the linear program's solution to a grid before using the union bound? *(The solution can be any point in the cube, an infinite set. The union bound needs a finite list of candidates; the grid of step $1/n$ has $(n + 1)^n$ points and moves every answer by at most $\frac12$.)*
- If the curator could see the queries before choosing $d$, would the theorem still hold? *(The proof fixes $d$ before the queries. With $d$ chosen after the queries, add a union bound over the $2^n$ databases: the count of bad grid points grows to at most $2^n(n + 1)^n$, which only adds $n\ln2$ to the numerator of @eq-recon-lp.)*

## Least squares under worst-case and random error {#sec-recon-ls}

### Context

The linear program needs a solver and a value of $E$. A simpler decoder needs neither. **Least squares** picks the vector whose answers come closest to the published ones, measured by the sum of squared differences, $\hat x = \arg\min_x\|Ax - a\|^2$. Setting the gradient $2A^\top(Ax - a)$ to zero gives the closed form $\hat x = (A^\top A)^{-1}A^\top a$. The decoder clips each coordinate to $[0, 1]$ and rounds it. Because $\hat x$ is a linear function of the answers, its error is a linear function of the curator's error, so you can analyse it exactly in two cases: errors chosen by an adversary, and independent random noise.

### Statement

How much least squares amplifies the curator's error depends on how much $A$ can shrink a vector. The **smallest singular value** $s_{\min}(A)$ is the smallest length $\|Ax\|$ over unit vectors $x$: the factor by which $A$ shrinks the direction it shrinks most. Assume $m \ge n$ and that $A$ has full column rank, so that $s_{\min}(A) > 0$.

**Worst-case error.** The decoder's error vector has length at most $\|e\|/s_{\min}(A)$, because undoing $A$ can stretch a vector by at most $1/s_{\min}$. Each wrong bit is off by at least $\frac12$, so it uses up at least $\frac14$ of the squared length. If $|e_j| \le E$ for every $j$, so that $\|e\|^2 \le mE^2$, the least-squares decoder therefore gets at most
$$\frac{4\|e\|^2}{s_{\min}(A)^2} \le \frac{4mE^2}{s_{\min}(A)^2}$$ {#eq-recon-ls-worst}
bits wrong. For random subsets, $s_{\min}(A) \approx \frac12(\sqrt m - \sqrt n)$, so $s_{\min}^2 \approx \frac m4(1 - \sqrt{n/m})^2$ and the bound becomes $16E^2/(1 - \sqrt{n/m})^2$; at $m = 4n$, $1 - \sqrt{1/4} = \frac12$ and it is $64E^2$. So error $E = o(\sqrt n)$ gives $o(n)$ wrong bits with only $m = 4n$ queries.

**Random error.** Now let the curator's errors be independent $\mathcal N(0, \sigma^2)$, normal with mean 0 and standard deviation $\sigma$. Each coordinate of the decoder's error is then also normal, and, as the derivation shows, its variance is about $\frac{4\sigma^2}{m - n}$: the noise $\sigma^2$ divided by roughly the number of equations beyond the $n$ needed to determine $d$, times 4 because a random query includes each person only half the time. A bit is recovered when its error does not push it across $\frac12$. Write $\Phi$ for the standard normal cumulative distribution function, so $\Phi(t)$ is the probability that a standard normal variable is below $t$. Each coordinate of $\hat x - d$ is approximately $\mathcal N\big(0, \frac{4\sigma^2}{m - n}\big)$, and the expected fraction of bits recovered is approximately
$$\Phi\left(\frac{\sqrt{m - n}}{4\sigma}\right).$$ {#eq-recon-ls-random}
The argument of $\Phi$ is the distance $\frac12$ to the wrong side, measured in standard deviations $2\sigma/\sqrt{m - n}$.

### Derivation

**Worst case.**

1. **The decoder's error is linear in the curator's.** Substituting $a = Ad + e$, $\hat x = (A^\top A)^{-1}A^\top(Ad + e) = d + A^+e$, with $A^+ = (A^\top A)^{-1}A^\top$ the pseudo-inverse. The $d$ comes back exactly, because $(A^\top A)^{-1}A^\top A = I$.
2. **Bound its size.** Write $A$ in its singular value decomposition, $A = U\,\mathrm{diag}(s_1, \dots, s_n)\,V^\top$ with $U$ and $V$ having orthonormal columns. Then $A^+ = V\,\mathrm{diag}(1/s_1, \dots, 1/s_n)\,U^\top$, whose largest singular value is $1/s_{\min}(A)$. So $\|\hat x - d\| = \|A^+e\| \le \|e\|/s_{\min}(A)$.
3. **Clipping only helps.** Each $d_i$ is in $[0, 1]$, and moving $\hat x_i$ to the nearest point of $[0, 1]$ cannot move it away from $d_i$.
4. **Count the wrong bits.** A wrong bit has $|\hat x_i - d_i| \ge \frac12$, so it contributes at least $\frac14$ to $\|\hat x - d\|^2$. Hence the number of wrong bits is at most $4\|\hat x - d\|^2 \le 4\|e\|^2/s_{\min}^2$, and $\|e\|^2 \le mE^2$ because each of the $m$ errors is at most $E$ in size. This is @eq-recon-ls-worst.
5. **Estimate the singular value.** For a random matrix with $m$ independent rows, $s_{\min}$ concentrates near $\sqrt m - \sqrt n$ times the scale of the rows (Vershynin, *High-Dimensional Probability*, ch. 4, for the general statement). Here the rows are $\frac12(\mathbf 1 + s)$ with a random-sign vector $s$. The all-ones part sends every vector orthogonal to $\mathbf 1$ to 0, so on those directions $A$ acts as $\frac12$ times a random-sign matrix, which gives the approximation $\frac12(\sqrt m - \sqrt n)$. The script checks it: at $n = 500$, $m = 2000$, $s_{\min} = 11.43$ against $\frac12(\sqrt{2000} - \sqrt{500}) = \frac12(44.72 - 22.36) = 11.18$. Substituting gives $4mE^2/\big(\frac14(\sqrt m - \sqrt n)^2\big) = 16E^2/(1 - \sqrt{n/m})^2$.

**Random noise.** The plan is to find the variance of each coordinate of $\hat x - d$, then ask how often that normal error crosses $\frac12$.

1. **The error is Gaussian.** $\hat x - d = A^+e$ is a linear map of a Gaussian vector, so given $A$ it is Gaussian. A linear map $M$ turns covariance $\sigma^2I$ into $\sigma^2MM^\top$, so its covariance is $\sigma^2A^+(A^+)^\top = \sigma^2(A^\top A)^{-1}A^\top A(A^\top A)^{-1} = \sigma^2(A^\top A)^{-1}$.
2. **Compute the average Gram matrix.** $A^\top A = \sum_jq_jq_j^\top$ is a sum over the $m$ queries. For a random subset $q$, $q_i^2 = q_i$ because $q_i$ is 0 or 1, so $\mathbb E q_i^2 = \frac12$; and for $i \ne k$ the two memberships are independent, so $\mathbb E q_iq_k = \frac14$. Hence $\mathbb E[qq^\top] = \frac14(I + \mathbf 1\mathbf 1^\top)$ and $\mathbb E[A^\top A] = \frac m4(I + \mathbf 1\mathbf 1^\top)$.
3. **Invert it.** $(I + \mathbf 1\mathbf 1^\top)^{-1} = I - \frac{1}{n + 1}\mathbf 1\mathbf 1^\top$. To check, multiply out and use $\mathbf 1^\top\mathbf 1 = n$: the product is $I + \mathbf 1\mathbf 1^\top - \frac{1}{n + 1}\mathbf 1\mathbf 1^\top - \frac{n}{n + 1}\mathbf 1\mathbf 1^\top = I$ ([Exercise 4](../../book-exercises/part02-attacks/09-reconstruction.md#sec-sol-reconstruction-4)). Its diagonal is $1 - \frac1{n + 1} = \frac{n}{n + 1} \approx 1$, so replacing $A^\top A$ by its mean gives a per-coordinate variance of about $\sigma^2 \cdot \frac4m = 4\sigma^2/m$.
4. **Correct for finite $m$.** $A^\top A$ fluctuates around its mean, and the inverse of a fluctuating matrix is on average larger than the inverse of its mean, just as $\mathbb E[1/Z] \ge 1/\mathbb E Z$ for a positive number $Z$. For Gaussian rows the exact mean is known. A sum of outer products of $m$ independent Gaussian rows with covariance $\Sigma$ is called a Wishart matrix, and if $A^\top A$ is Wishart with $m$ rows and scale $\Sigma$, then $\mathbb E[(A^\top A)^{-1}] = \Sigma^{-1}/(m - n - 1)$ (the inverse-Wishart mean; Muirhead 1982, ch. 3). Using it with $\Sigma = \mathbb E[qq^\top]$ replaces $m$ by $m - n - 1$: the variance is $4\sigma^2\frac{n}{n + 1}\cdot\frac{1}{m - n - 1} \approx \frac{4\sigma^2}{m - n}$. The rows here are 0/1, not Gaussian, so this is an approximation; the simulations below check it.
5. **Find when a bit goes wrong.** A 0 rounds to 1 when its error exceeds $+\frac12$; a 1 rounds to 0 when its error is below $-\frac12$. Either way only one tail counts. The error has standard deviation $\frac{2\sigma}{\sqrt{m - n}}$, so the probability is $\Phi\big(-\frac12\big/\frac{2\sigma}{\sqrt{m - n}}\big) = \Phi\big(-\frac{\sqrt{m - n}}{4\sigma}\big)$. The fraction recovered is one minus that, which by the symmetry $1 - \Phi(-t) = \Phi(t)$ of the normal curve is @eq-recon-ls-random.

### Interpretation

At a fixed ratio of queries to people, only $\sigma/\sqrt n$ matters. With $m = \kappa n$, $\sqrt{m - n} = \sqrt{\kappa - 1}\sqrt n$, so @eq-recon-ls-random is $\Phi\big(\sqrt{\kappa - 1}/(4\sigma/\sqrt n)\big)$. Plotted against $\sigma/\sqrt n$, the curves for $n = 100$, 300 and 900 fall on top of each other (@fig-recon-ls-noise). The scale of the threat is $\sqrt n$, as in the theorem.

Noise of order $\sqrt n$ is not automatically safe. At $\sigma = 0.25\sqrt n$ and $m = 4n$, the argument of $\Phi$ is $\sqrt 3/(4 \times 0.25) = \sqrt3 \approx 1.73$, and least squares recovers about 96% of the bits. Recovery degrades smoothly as $\sigma/\sqrt n$ grows; it does not switch off at a threshold.

More queries average independent noise away. At a fixed $\sigma$, @eq-recon-ls-random tends to 1 as $m$ grows, because $m - n$ independent noisy equations average to an error of order $\sigma/\sqrt{m - n}$ per coordinate. To keep the attack at a fixed accuracy, the argument $\sqrt{m - n}/(4\sigma)$ must stay fixed, so the noise must grow like $\sqrt m$ (@fig-recon-ls-queries). This is the second half of the Fundamental Law: a bound on the error of *each* answer is not enough when the number of answers is not bounded.

Worst-case errors hurt more than random ones of the same size. A curator that chooses its errors along the decoder's weakest direction, the right singular vector of $s_{\min}$, which least squares amplifies the most, does much more damage than Gaussian noise with the same scale, whose errors point in all directions equally. The bound @eq-recon-ls-worst is loose by more than a factor of 10 in the example below, but it is the only statement that holds for every curator.

![Fraction of bits recovered by least squares from $m = 4n$ noisy random subset sums, against the noise scale in units of $\sqrt n$, for three database sizes. The black curve is @eq-recon-ls-random. The curves for different $n$ coincide: the attack's success depends on $\sigma/\sqrt n$ only.](figures/recon_ls_noise.svg){#fig-recon-ls-noise}

![Fraction of bits recovered by least squares at $n = 200$ as the number of queries grows, for three noise levels. Lines are @eq-recon-ls-random; dots are simulations. Near $m = n$ the system is barely determined and the noise is amplified; beyond that, more queries average the noise away, and even at $\sigma = 2\sqrt n$ the attack recovers most bits with $64n = 12\,800$ queries.](figures/recon_ls_queries.svg){#fig-recon-ls-queries}

### Example: $n = 500$ people, $m = 2000$ queries

The script averages over four random databases and query matrices (@tbl-recon-ls). Here $\sqrt n = \sqrt{500} = 22.36$, which gives the second row.

| $\sigma/\sqrt n$ | 0.1 | 0.25 | 0.5 | 1 | 2 |
| --- | --- | --- | --- | --- | --- |
| $\sigma$ | 2.24 | 5.59 | 11.18 | 22.36 | 44.72 |
| Recovered, simulated | 1.0000 | 0.9655 | 0.8130 | 0.6550 | 0.5875 |
| Recovered, @eq-recon-ls-random | 1.0000 | 0.9585 | 0.8069 | 0.6676 | 0.5858 |

: Least-squares reconstruction of $n = 500$ bits from $m = 2000$ random subset sums with Gaussian noise. {#tbl-recon-ls}

The prediction is within 0.013 of every simulated value. By hand at $\sigma = 0.5\sqrt n$: $\sqrt{m - n}/(4\sigma) = \sqrt{1500}/44.72 = 0.866$, and $\Phi(0.866) = 0.807$.

For worst-case errors, the script measures $s_{\min}(A) = 11.43$ and runs a curator that adds $E\cdot\mathrm{sign}(Av)$, with $v$ the weakest right singular vector. That curator pushes every answer by the full $E$ in the direction that least squares turns into the largest error along $v$. At $E = 1$, the bound @eq-recon-ls-worst allows $4 \times 2000 \times 1/11.43^2 = 61.2$ wrong bits, and this curator causes 0. At $E = 2$, the bound is four times larger, 245.0, and this curator causes 20. At $E = 0.25\sqrt n = 5.59$, the bound (1914.0) exceeds $n$ and says nothing, while this curator causes 143 wrong bits, against about 17 for Gaussian noise with $\sigma = 5.59$ (1 − 0.9655 of 500).

### Check yourself

- At $m = n$ with exact answers, least squares recovers $d$ exactly. Why does the prediction @eq-recon-ls-random fail there with noise? *(At $m = n$ the system is square and $s_{\min}$ is close to 0, so tiny errors are amplified enormously. The factor $1/(m - n - 1)$ in the variance blows up; @fig-recon-ls-queries shows recovery near $\frac12$ there.)*
- A curator adds independent noise with $\sigma = 2\sqrt n$ to every answer and lets anyone ask as many questions as they want. How many queries per person does least squares need to recover 90% of the bits? *(Solve $\Phi(\sqrt{m - n}/(4\sigma)) = 0.9$. The standard normal is below 1.2816 with probability 0.9, so $\sqrt{m - n}/(4\sigma) = 1.2816$; dividing through by $\sqrt n$ gives $\sqrt{m/n - 1} = 4 \times 2 \times 1.2816$, so $m/n = 106.1$.)*

## The Fundamental Law of Information Recovery {#sec-recon-law}

### Context

The attacks so far say when answers are too accurate. Two questions remain. Is error of order $\sqrt n$ really enough to stop reconstruction, or is it just where the proofs stop? And how many queries does *any* attack need? This section answers both, with one construction and one counting argument, and collects the results in the form Dwork and Roth call the Fundamental Law of Information Recovery: overly accurate answers to too many questions destroy privacy, and they do so spectacularly.

### Statement

@tbl-recon-law collects the regimes. The first three rows are the attacks of the previous sections; the last two are the converses proved in this section.

| Queries | Error per answer | Outcome | Source |
| --- | --- | --- | --- |
| All $2^n$ subsets | at most $E$ | at most $4E$ wrong bits; $E = o(n)$ is blatantly non-private | @eq-recon-exp |
| $O(n\log n)$ random subsets | at most $E$, $E = o(\sqrt n)$ | $o(n)$ wrong bits, by a linear program | @eq-recon-lp |
| $m$ random subsets | independent $\mathcal N(0, \sigma^2)$ | fraction recovered $\approx \Phi(\sqrt{m - n}/(4\sigma))$, tending to 1 as $m$ grows | @eq-recon-ls-random |
| $m$ arbitrary | $O(\sqrt{n\log m})$, sampling curator | no attack recovers more than $\frac34$ of the bits | @eq-recon-hoeffding |
| $m$, each answer one of $V$ values | any | at least $n(1 - h(\beta))/\log_2V$ queries are needed for $\beta n$ wrong bits | @eq-recon-info |

: The regimes of reconstruction from subset sums. The first three rows are attacks; the last two are limits on attacks. {#tbl-recon-law}

**A sampling curator.** To show that error of order $\sqrt n$ can stop reconstruction, you need a curator whose errors are that small yet hide many bits completely. The idea is to answer from a random sample of the people and scale up. The curator draws a secret set $T$, keeping each person independently with probability $p$, and answers every query by $a_j = \frac1p\sum_{i \in S_j \cap T}d_i$. Dividing by $p$ makes each answer correct on average, and its error is a sum of independent zero-mean terms, one per person, so it is of order $\sqrt n$. People outside $T$ never affect any answer, so nothing reveals their bits. Then, for a failure probability $\eta$, with probability at least $1 - \eta$ all $m$ answers have error at most
$$\max_j|a_j - \langle q_j, d\rangle| \le \sqrt{\frac{n\ln(2m/\eta)}{2p^2}},$$ {#eq-recon-hoeffding}
and if $d$ is uniform, no attacker recovers an expected fraction of more than $p + \frac{1 - p}{2}$ of the bits; at $p = \frac12$, $\frac12 + \frac14$, three quarters.

**An information bound.** To bound the number of queries any attack needs, count bits of information. The secret is $n$ uniform bits, so it carries $n$ bits of information. An answer that takes at most $V$ values can carry at most $\log_2V$ bits. A guess that is wrong on a fraction $\beta$ of the bits still leaves some uncertainty, measured by the **binary entropy** $h(\beta) = -\beta\log_2\beta - (1 - \beta)\log_2(1 - \beta)$, the uncertainty in bits of a coin that comes up heads with probability $\beta$. So the answers must carry at least $n(1 - h(\beta))$ bits. Precisely, let $d$ be uniform on $\{0, 1\}^n$, and let each of the $m$ answers take at most $V$ values (for example, exact counts take $V = n + 1$ values). If an attacker's guess $\hat d$ has an expected fraction $\beta \le \frac12$ of wrong bits, then
$$m \ge \frac{n\,(1 - h(\beta))}{\log_2V},$$ {#eq-recon-info}
where $h$ is the binary entropy function in bits.

### Derivation

**The sampling curator's error (Hoeffding and the union bound).** The plan is to bound one answer's error with Hoeffding's inequality, then all $m$ answers at once with the union bound.

1. **Write the error as a sum.** With $t_i = \mathbf 1[i \in T]$, the answer is $\sum_{i \in S_j}d_it_i/p$, so the error of query $j$ is $\sum_{i \in S_j}d_i\big(\frac{t_i}{p} - 1\big)$. Only the $k \le n$ people with $d_i = 1$ in $S_j$ contribute. Their terms are independent, have mean $p\cdot\frac1p - 1 = 0$, and lie in $[-1, \frac1p - 1]$, an interval of length $\frac1p$.
2. **Apply Hoeffding's inequality to the sum.** Hoeffding's inequality ([Concentration inequalities](../part01-prerequisites/02-concentration.md)) says that the mean of independent bounded terms is unlikely to stray far from its expectation, with a tail that falls like a Gaussian's. For a mean $\bar X$ of $k$ independent terms with range $\frac1p$ and expectation $\mu$, it reads $\Pr[\bar X - \mu \ge t] \le \exp(-2kt^2p^2)$. A sum deviates by $u$ when the mean deviates by $u/k$, so substituting $t = u/k$ and counting both tails, $\Pr[|\text{error}| \ge u] \le 2\exp(-2p^2u^2/k) \le 2\exp(-2p^2u^2/n)$, the last step because $k \le n$.
3. **Union bound over the $m$ queries.** The probability that some answer has error at least $u$ is at most the sum over the $m$ queries: $\Pr[\text{some error} \ge u] \le 2m\exp(-2p^2u^2/n)$. Set this equal to $\eta$ and solve: $2p^2u^2/n = \ln(2m/\eta)$, so $u^2 = n\ln(2m/\eta)/(2p^2)$, which is @eq-recon-hoeffding.

**What the attacker can learn.** The idea is that a person outside $T$ is invisible to every answer.

1. **The answers ignore the people outside $T$.** They are a function of $T$ and of $d_i$ for $i \in T$ only.
2. **So a person outside $T$ is a fair coin to the attacker.** Given $i \notin T$, the answers are independent of $d_i$, which is a fair bit independent of everything else. Any guess of $d_i$ is right with probability $\frac12$.
3. **Average over who is in $T$.** Each person is in $T$ with probability $p$, and at best the attacker gets all of those right. Each person is outside with probability $1 - p$, and then the attacker is right half the time. The expected fraction recovered is at most $p \cdot 1 + (1 - p)\cdot\frac12 = p + \frac{1 - p}2$.

**The information bound (entropy and Fano).** Let $Y = (a_1, \dots, a_m)$ be the answers, with the queries fixed. The **mutual information** $I(d; Y)$ is the number of bits the answers reveal about $d$: the uncertainty about $d$ before seeing $Y$ minus the uncertainty after, $H(d) - H(d \mid Y)$, where $H$ is entropy. The plan is to bound it from above by what the answers can carry and from below by what a good guess requires.

1. **Bound what the answers can carry.** $I(d; Y) \le H(Y) \le \sum_jH(a_j) \le m\log_2V$. Mutual information is at most the entropy of the output, because it is $H(Y)$ minus the non-negative $H(Y \mid d)$. The entropy of a tuple is at most the sum of the entropies, with equality only when the entries are independent. And a variable with $V$ values has entropy at most $\log_2V$ bits, attained when all values are equally likely.
2. **Processing cannot add information.** $\hat d$ is computed from $Y$, so $I(d; \hat d) \le I(d; Y)$ by the DPI, which says that no function of $Y$, random or not, can tell you more about $d$ than $Y$ itself.
3. **Write what a good guess requires.** $I(d; \hat d) = H(d) - H(d \mid \hat d) = n - H(d \mid \hat d)$, since $n$ uniform bits have entropy $n$. To bound the remaining uncertainty bit by bit, use the chain rule, $H(d \mid \hat d) = \sum_iH(d_i \mid \hat d, d_1, \dots, d_{i-1})$, and the fact that conditioning on less cannot reduce entropy: dropping $d_1, \dots, d_{i-1}$ and then all of $\hat d$ except $\hat d_i$ gives $H(d \mid \hat d) \le \sum_iH(d_i \mid \hat d) \le \sum_iH(d_i \mid \hat d_i)$.
4. **Apply Fano for one bit.** With two values, Fano's inequality reads $H(d_i \mid \hat d_i) \le h(P_i)$, where $P_i = \Pr[\hat d_i \ne d_i]$. The reason is short: if you knew both the guess $\hat d_i$ and whether it was wrong, you would know $d_i$. So the uncertainty about $d_i$ given $\hat d_i$ is at most the uncertainty about "was the guess wrong?", a coin that comes up "wrong" with probability $P_i$, whose entropy is at most $h(P_i)$.
5. **Average with concavity.** $h$ is concave (its graph bends downward), so by Jensen's inequality the average of its values is at most its value at the average: $\frac1n\sum_ih(P_i) \le h(\beta)$, where $\beta = \frac1n\sum_iP_i \le \frac12$ is the expected fraction of wrong bits.
6. **Combine.** Steps 3 to 5 give $I(d; \hat d) \ge n - nh(\beta)$, and steps 1 and 2 give $I(d; \hat d) \le m\log_2V$. So $n - nh(\beta) \le I(d; \hat d) \le m\log_2V$, which is @eq-recon-info.

### Interpretation

Error of order $\sqrt n$ is necessary to stop reconstruction, and it can be enough. The sampling curator's error is $O(\sqrt{n\log m})$, only a logarithm above the $o(\sqrt n)$ that the attacks need, and it caps every attacker at $\frac34$ of the bits.

But it is not privacy. The same curator answers exactly for the people in $T$: an attacker with enough queries learns their bits perfectly, because least squares recovers $d_i/p$ for every $i \in T$. Half the people are fully exposed so that the other half can be fully hidden. Stopping blatant reconstruction is a much weaker goal than protecting everyone, and [Part 3](../part03-dp-foundations/01-definition.md) defines a guarantee that holds for every individual.

The attacks use about as few queries as possible. With exact counts, @eq-recon-info asks for at least $n(1 - h(\beta))/\log_2(n + 1)$ queries, of order $n/\log n$, and least squares with exact answers uses $n$. Answers with error, published to a coarser precision, take fewer values, carry fewer bits each and need more queries.

The law has two halves, and a defence must control both. "Overly accurate" is the error scale: below $\sqrt n$ for random queries, below $n$ for all queries. "Too many" is the count: independent noise per answer is averaged away unless it grows with the number of answers. That is why Part 3 tracks a budget that is spent across queries.

### Example: three curators

**The sampling curator at $n = 500$, $m = 2000$.** With $p = \frac12$ the script keeps 250 of the 500 people. The largest error over the 2000 queries is 28.00, or $28.00/\sqrt{500} = 1.25\sqrt n$. That is well under the Hoeffding bound at $\eta = 0.05$: with $p = \frac12$, $2p^2 = \frac12$, so @eq-recon-hoeffding is $\sqrt{2 \cdot 500\ln(4000/0.05)} = 106.25$. (Hoeffding assumes the worst-case spread of each term; the actual terms have a smaller variance.) Least squares on these answers recovers 0.7560 of the bits, right at the ceiling of $\frac34$.

**The information bound at $n = 1000$.** For at most 5% wrong bits, $h(0.05) = 0.2864$, so each person still needs $1 - 0.2864 = 0.7136$ bits. Exact answers ($V = 1001$) carry at most $\log_21001 \approx 9.97$ bits each and need $m \ge 1000 \times 0.7136/\log_21001 = 71.6$ queries, so at least 72. If the curator rounds every count to a multiple of 10, an answer takes at most $\lfloor1000/10\rfloor + 2 = 102$ values (the 101 multiples of 10 from 0 to 1000, plus one to be safe; the script uses $\lfloor n/w\rfloor + 2$ for every width $w$), and the bound rises to 106.9. Rounding to multiples of 32 leaves $\lfloor 1000/32\rfloor + 2 = 33$ values and a bound of 141.5. These are lower bounds for any attack, whatever its running time.

**Independent noise with unlimited queries.** From @tbl-recon-ls, a curator with $\sigma = 2\sqrt n$ lets least squares recover 0.5875 of the bits at $m = 4n$. If it answers any number of queries with the same noise level, an attacker who asks $m = 106.1\,n$ queries recovers 90% (the second Check yourself question of @sec-recon-ls).

### Check yourself

- Why does the information bound need the answers to take finitely many values? *(Step 1 bounds $H(a_j)$ by $\log_2V$. A real-valued answer can carry unbounded information in its low-order digits, so the count of values is what limits it. Any published table has finite precision.)*
- A curator answers each query with an independent error of size $\sqrt{n\log m}$ chosen at random. Is it as safe as the sampling curator? *(No. Its errors are independent across queries, so least squares averages them away as $m$ grows. The sampling curator's errors come from one secret set $T$ shared by all answers, so asking more never reveals the people outside $T$.)*

## Reconstructing a census block {#sec-recon-census}

### Context

National statistical offices publish many tables computed from the same confidential records: counts by age, sex and race for every block, the same counts by household type, medians and means, and so on. Each table cell is a query. Garfinkel, Abowd and Martindale (2019) showed with a toy block of seven people that a modest set of published statistics can determine every person's age, sex, race and marital status, and that a constraint solver finds the solution in seconds. By then the US Census Bureau had already run the attack on its own 2010 publications at national scale, an experiment Abowd (2021) later described in detail.

This section repeats the toy experiment with a block of our own. It is a reconstruction game where the queries are not random subsets but the tables an agency actually publishes, and where the decoder is a search over all databases consistent with them.

### Statement

A fictional block has seven people, each with an age (a whole number from 0 to 99), a sex, a race (Black or White) and a marital status, and nobody under 18 is married. @tbl-recon-census-published is everything the agency publishes about the block. Means are given to two decimals, which is enough to recover the sum of ages in each group exactly: the sum is a whole number, and $34.29 \times 7 = 240.03$, so the total is 240.

| Group | Count | Median age | Mean age |
| --- | --- | --- | --- |
| Total | 7 | 33 | 34.29 |
| Female | 4 | 31 | 30.25 |
| Male | 3 | 33 | 39.67 |
| Black | 4 | 33.5 | 36.25 |
| White | 3 | 33 | 31.67 |
| Married | 4 | 43.5 | 47.75 |

: The published statistics for a fictional census block of seven people. A median over an even count is the average of the two middle ages. {#tbl-recon-census-published}

Each statistic is a constraint on the unknown ages and labels, and the claim is that together they leave no freedom at all.

**Claim.** Exactly one block of seven people is consistent with @tbl-recon-census-published, and it is the true one (@tbl-recon-census-block).

| Person | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Age | 7 | 15 | 27 | 33 | 35 | 52 | 71 |
| Sex | F | M | F | M | F | F | M |
| Race | B | B | W | W | W | B | B |
| Married | no | no | no | yes | yes | yes | yes |

: The block reconstructed from @tbl-recon-census-published, which is the true block. {#tbl-recon-census-block}

### Derivation

The decoder enumerates candidate blocks and discards those that contradict a published number. Each step uses the statistics in a way you can check by hand.

1. **Turn means into sums.** Mean × count gives the sum of ages: total 240, female 121, male 119, Black 145, White 95, married 191. The sums for female and male add to 240, so the male mean is implied by the other two; the same holds for Black and White. Single people number $7 - 4 = 3$ and their ages sum to $240 - 191 = 49$.
2. **Enumerate the women's ages.** Sort them $a \le b \le c \le d$. With four women the median is the average of the two middle ages, so $b + c = 2 \times 31 = 62$, and the sum gives $a + d = 121 - 62 = 59$. Choose $b \le 31$ (so that $c = 62 - b \ge b$) and $a \le b$; then $d = 59 - a$ must be at least $c = 62 - b$, that is $a \le b - 3$. For each $b$ from 3 to 31 that leaves $a = 0, \dots, b - 3$, which is $b - 2$ choices, so there are $1 + 2 + \dots + 29 = \sum_{k=1}^{29}k = 435$ age lists in all.
3. **Enumerate the men's ages.** Three sorted ages with median 33 and sum 119: the middle one is 33, so $x_1 + x_3 = 119 - 33 = 86$ with $x_1 \le 33 \le x_3 \le 99$. Every $x_1$ from 0 to 33 gives $x_3 = 86 - x_1$ between 53 and 86, which is allowed, so $x_1$ ranges over $0, \dots, 33$: 34 lists.
4. **Combine and check the total median.** The $435 \times 34 = 14\,790$ combined age vectors are all distinct; 12 906 have overall median 33.
5. **Check the race split.** For each age vector, try every way of choosing 4 of the 7 people as Black ($\binom74 = 35$ ways), and keep it if the Black ages have median 33.5 and sum 145 and the White ages have median 33 and sum 95. Only 387 age vectors admit such a split.
6. **Check the marital split** in the same way (married median 43.5, sum 191, everyone married at least 18): 33 age vectors admit one.
7. **Require all three at once.** Exactly one age vector admits a valid sex split, race split and marital split, and its splits are unique. Assigning each person the labels of the splits gives @tbl-recon-census-block.

Each published statistic removes most of the remaining candidates. The block is not determined by any one table, but by their intersection.

### Interpretation

Withholding one number rarely helps. @fig-recon-census repeats the search with each of the twelve medians and means withheld in turn. Eight of the twelve leave the block uniquely determined, because the remaining statistics imply the missing one or pin the block down without it ([Exercise 8](../../book-exercises/part02-attacks/09-reconstruction.md#sec-sol-reconstruction-8)). The worst case for the attacker, withholding the married mean, still leaves only 45 consistent blocks, and 51% of them have the true sex, race and marital pattern of all seven people.

Real publications are far more exposed than this toy, because published tables overlap. Real tables cross-classify many variables at many geographic levels, so the same people appear in thousands of cells. That is exactly the "many queries" condition of the Fundamental Law. The main protection of the 2010 Census, swapping a small fraction of households between blocks, changes some answers but leaves most constraints exact. Coarsening (age bands instead of single years) and suppressing small cells, common in other tables, have the same weakness: they remove some constraints and leave the rest exact.

The attack also works at scale. The Census Bureau's own reconstruction of the 2010 Census, described by Abowd (2021), rebuilt a record for each of the 308,745,538 people counted, with block, sex, age, race and ethnicity. About 46% of the reconstructed records matched a real person exactly on all five fields, and linking them to commercial data confirmed the identity of about 17% of the population. The Bureau's response was to change its disclosure-avoidance method for the 2020 Census to one based on the privacy definition of [Part 3](../part03-dp-foundations/01-definition.md).

![Number of seven-person blocks consistent with @tbl-recon-census-published when one median or mean is withheld (log scale). Blue bars: the block is still unique. Red bars: a few candidates remain, still far fewer than the possible blocks.](figures/recon_census.svg){#fig-recon-census}

### Example: what each withheld statistic leaves

@tbl-recon-census-withheld lists, for each statistic whose removal leaves more than one block, how many blocks remain, how many different sex/race/marital patterns they show, and what share of them has the true pattern.

| Withheld | Consistent blocks | Distinct sex/race/marital patterns | Share with the true pattern |
| --- | --- | --- | --- |
| Female median | 7 | 6 | 0.14 |
| Black median | 36 | 3 | 0.64 |
| Married median | 26 | 4 | 0.46 |
| Married mean | 45 | 7 | 0.51 |
| Any of the other eight | 1 | 1 | 1.00 |

: Blocks consistent with the published table when one statistic is withheld. {#tbl-recon-census-withheld}

Withholding the married mean hurts the attacker most, because it is the only statistic about the single people's ages: without it, the married and single sums are no longer known. Withholding the female mean changes nothing, because the total and male sums imply it: it equals $(240 - 119)/4$.

### Check yourself

- Why can the attacker turn a mean published to two decimals into an exact sum? *(The sum is a whole number and the mean is the sum over a count of at most 7. Two decimals locate the sum to within $7 \times 0.005 = 0.035$, which leaves one whole number.)*
- If the agency published only counts and no ages, could the attacker still reconstruct something? *(Yes: counts of every cross-classification by sex, race and marital status fix how many people fall in each combination, which is the attribute pattern. Ages need age statistics, which real tables publish in single years for small groups.)*

## Code: check every attack by computation and simulation

The script [code/reconstruction.py](../../code/part02-attacks/reconstruction.py) computes every number in this chapter and draws every figure. Each attack, bound and curator is a small typed function, and the figures loop over them. It uses NumPy only and runs in well under a minute. From the repository root:

```bash
uv run python code/part02-attacks/reconstruction.py
```

**Output** (seed 0), summarized in @tbl-recon-code:

| Experiment | Result |
| --- | --- |
| Fact 1, $d = (1, 0, 1, 1)$ | The 16 subset sums, in binary order: 0, 1, 0, 1, 1, 2, 1, 2, 1, 2, 1, 2, 2, 3, 2, 3 |
| Exponential attack, $n = 10$ | Survivors 1, 35, 242, 630, 912 and largest distance 0, 2, 4, 6, 8 for $E = 0, 0.5, 1, 1.5, 2$; rounding to multiples of 2 or 4: one survivor |
| Fact 2 | $\mathbb E(s_1 + s_2 + s_3)^4 = 21$ by enumeration and by formula |
| Lemma, $n = 1000$, $\beta = 0.1$ | Threshold 0.8333; escape probability 0.6161 (balanced), 1.0000 (aligned), bound 0.1875 |
| Theorem, $n = 1000$, $\beta = 0.1$, $\eta = 0.01$ | $E \le 0.1667$; $m \ge 33\,296$ ($33.3n$) |
| Least squares, Gaussian noise, $n = 500$, $m = 2000$ | @tbl-recon-ls; $m/n$ for 90% recovery: 106.1 at $\sigma = 2\sqrt n$, 657.9 at $5\sqrt n$ |
| Least squares, worst case, $n = 500$, $m = 2000$ | $s_{\min} = 11.43$ (approximation 11.18); wrong bits 0, 20, 143 for $E = 1, 2, 5.59$, bounds 61.2, 245.0, 1914.0 |
| Sampling curator, $p = \frac12$ | Kept 250 of 500; max error 28.00 ($1.25\sqrt n$); Hoeffding bound 106.25; recovered 0.7560 |
| Information bound, $n = 1000$ | $\beta = 0.05$: 71.6 (exact), 106.9 (rounded to 10), 141.5 (rounded to 32); $\beta = 0.01$: 92.2 (exact) |
| Census block | 435 female and 34 male age lists; 14 790 age vectors; 12 906 after the total median; 387 with a race split; 33 with a marital split; 1 block; @tbl-recon-census-withheld |

: The script's output with the default seed. {#tbl-recon-code}

[tests/test_reconstruction.py](../../tests/test_reconstruction.py) checks that the output still matches.

**What to take from it.** Five things stand out. First, the exponential attack's bound is exact: every surviving candidate is within $4E$ of $d$, and some candidate is at exactly $4E$, so no better guarantee is possible for that attack. Second, the proved constants are pessimistic, but the scaling is not. The theorem tolerates only $E \le 0.17$ at $n = 1000$, yet least squares with $m = 4n$ recovers 96.55% of 500 bits at $\sigma = 5.59$, because both are governed by the same $\sqrt n$ scale and only the proof's constants differ. Third, the least-squares prediction is accurate: @eq-recon-ls-random is within 0.013 of the simulation at every noise level, once the $m - n$ correction is included. Fourth, the sampling curator meets its ceiling: its answers have error $1.25\sqrt n$, and the best decoder gets exactly the people in its sample, plus half of the rest by chance. Fifth, published tables are systems of equations: eighteen numbers determine seven people's ages, sexes, races and marital statuses.

## References

- Dinur & Nissim, "Revealing Information while Preserving Privacy" (PODS 2003). The original attacks: all subset sums with error $o(n)$, and polynomially many random subset sums with error $o(\sqrt n)$ decoded by a linear program. The "blatantly non-private" definition is theirs.
- Dwork & Roth, *The Algorithmic Foundations of Differential Privacy* (2014), ch. 8. The reconstruction attacks as lower bounds, and the Fundamental Law of Information Recovery.
- Dwork, McSherry & Talwar, "The Price of Privacy and the Limits of LP Decoding" (STOC 2007). The linear-programming decoder tolerates a constant fraction of arbitrarily wrong answers, with $O(n)$ queries.
- Dwork & Yekhanin, "New Efficient Attacks on Statistical Disclosure Control Mechanisms" (CRYPTO 2008). Attacks with structured queries and fast decoders.
- Kasiviswanathan, Rudelson & Smith, "The Power of Linear Reconstruction Attacks" (SODA 2013). Least-squares and linear-programming reconstruction from released statistics such as marginal tables, with the role of the smallest singular value made precise.
- Dwork, Smith, Steinke & Ullman, "Exposed! A Survey of Attacks on Private Data" (Annual Review of Statistics and Its Application, 2017). A readable survey of reconstruction and tracing attacks; the best next step after this chapter.
- Cohen & Nissim, "Linear Program Reconstruction in Practice" (Journal of Privacy and Confidentiality, 2020). A linear-programming reconstruction against a deployed query system that added noise to its answers.
- Garfinkel, Abowd & Martindale, "Understanding Database Reconstruction Attacks on Public Data" (CACM 2019). The seven-person block reconstructed with a solver; the model for @sec-recon-census.
- Abowd, declaration in *State of Alabama v. US Department of Commerce* (2021), Appendix B. The Census Bureau's reconstruction and re-identification experiment on the 2010 Census.
- Vershynin, *High-Dimensional Probability* (2018), ch. 4. Singular values of random matrices, used in the worst-case least-squares bound. Muirhead, *Aspects of Multivariate Statistical Theory* (1982), ch. 3, for the inverse-Wishart mean.

## Cheat sheet

$d \in \{0, 1\}^n$ is the secret, $A$ the $m \times n$ 0/1 query matrix, $a = Ad + e$ the answers, $E$ a bound on every $|e_j|$, $\sigma$ the scale of random noise, $\beta$ a fraction of wrong bits, $\eta$ a failure probability. @tbl-recon-cheat collects the results.

| Result | Statement | Proof idea |
| --- | --- | --- |
| Exponential attack | All $2^n$ subsets, error $E$: any consistent $c$ has $\mathrm{dist}(c, d) \le 4E$; tight | Triangle inequality on $S_0$ and $S_1$ |
| Paley–Zygmund | $\Pr[W > \theta\,\mathbb E W] \ge (1 - \theta)^2(\mathbb E W)^2/\mathbb E W^2$ | Split $\mathbb E W$; Cauchy–Schwarz |
| Anti-concentration lemma | $\beta n$ entries of $z$ at least $\frac13$: $\Pr[\lvert\langle q, z\rangle\rvert > \sqrt{\beta n}/12] \ge \frac3{16}$ | Fourth moment $\le 3\times$ squared second moment |
| Linear-programming decoder | $2E + \frac12 \le \sqrt{\beta n}/12$ and $m \ge \frac{n\ln(n+1) + \ln(1/\eta)}{\ln(16/13)}$: at most $\beta n$ wrong bits | Grid of step $1/n$, lemma, union bound |
| Least squares, worst case | Wrong bits $\le 4mE^2/s_{\min}^2 \approx 16E^2/(1 - \sqrt{n/m})^2$ | $\lVert A^+e\rVert \le \lVert e\rVert/s_{\min}$ |
| Least squares, Gaussian noise | Fraction recovered $\approx \Phi(\sqrt{m - n}/(4\sigma))$ | Covariance $\sigma^2(A^\top A)^{-1}$; inverse-Wishart mean |
| Sampling curator | Error $\le \sqrt{n\ln(2m/\eta)/(2p^2)}$; recovery $\le p + \frac{1 - p}2$ | Hoeffding, union bound; unsampled bits are independent of the answers |
| Information bound | $m \ge n(1 - h(\beta))/\log_2V$ | DPI, chain rule, Fano for one bit, concavity of $h$ |

: Every result in this chapter. {#tbl-recon-cheat}

**Rules of thumb**

- The threat scale of a random subset-sum query is $\sqrt n$; of the worst query, $n$. Errors well below that scale reveal the database.
- Independent noise per answer is averaged away by more answers: to keep reconstruction at bay, the noise must grow like $\sqrt m$.
- A curator that stops blatant reconstruction can still expose many individuals completely; stopping reconstruction is not the same as privacy.
- Exact counts carry at most $\log_2(n + 1)$ bits each, so about $n/\log n$ of them can already determine $n$ bits.
- Every published table cell is a constraint. Many overlapping tables of exact or nearly exact statistics are a solvable system.

**Practice.** Exercises for this chapter, each with a worked solution, are in [the exercise book](../../book-exercises/part02-attacks/09-reconstruction.md).
