# Reconstruction Attacks

This chapter asks when answers to counting questions about a dataset, each only slightly wrong, let an attacker rebuild almost the whole dataset. It does not yet define privacy. Instead it defines a failure so flagrant that nobody would call it private, and shows that a wide class of algorithms fails in this way. If a curator answers every possible counting query, it must make errors of order $n$, the number of people, to stop the attack. If it answers only about $n$ random queries, errors of order $\sqrt n$ are needed.
## Reconstruction from exact statistics {#sec-rec-census}

Exact statistics give the data away. A census collects **microdata**, one record per person with fields such as age, sex and race. It publishes tables of counts, medians and means for each small area, such as a city block. It suppresses cells about fewer than three people, but suppression alone protects nobody, because tables overlap. If three of a group's four residents are women, a published median age for its men is the age of its one man.

Every published number is a constraint on the unknown microdata, and enough constraints leave only one possibility. The next example counts the age combinations that one line of a census table leaves for three people.

::: {#exm-rec-ages name="One line of a census table"}
A block's table says that it has three men, with median age $35$ and mean age $47$. Call the youngest man's age $u_1$, the middle one's $u_2$ and the oldest one's $u_3$, so that $u_1 \le u_2 \le u_3$. Ages are whole numbers between $0$ and $125$.

1. Without the statistics, the number of such triples is the number of multisets of size $3$ from $126$ values, $\binom{128}{3} = 341{,}376$.
2. The median of three numbers is the middle one, so $u_2 = 35$. Then $u_1$ has $36$ values and $u_3$ has $91$, which leaves $36 \times 91 = 3{,}276$ triples.
3. The mean gives $u_1 + u_2 + u_3 = 141$, so $u_3 = 106 - u_1$. Each $u_1$ from $0$ to $35$ gives a $u_3$ between $71$ and $106$, and every such $u_3$ lies between $u_2$ and $125$. So exactly $36$ triples remain, from $(0, 35, 106)$ to $(35, 35, 71)$.
:::

So one line of the table cuts the possibilities by a factor of almost $10{,}000$. The other lines group the same people differently and cut them further. A reconstruction attack searches for microdata that satisfies every constraint. This search is an integer program, which is hard in the worst case, but solvers handle such instances quickly. For a fictional block of seven people, Garfinkel, Abowd and Martindale found that the constraints of its full table have exactly one solution, which a solver finds in a fraction of a second.

The attack scales. The United States Census Bureau ran it on its 2010 tables, and the rebuilt records matched the confidential ones exactly for 46% of the population. They matched for 71% if ages could be off by one year. A link of these records with commercial databases then gave the names of more than 50 million people. So the Bureau adopted differential privacy for the 2020 census.

## Subset queries and blatant non-privacy {#sec-rec-model}

To analyse such attacks, strip the setting to its core. There are $n$ people. Each has public attributes that identify them, such as a name, and one secret bit, such as whether they have a disease. The secret bits form the vector $d = (d_1, \dots, d_n) \in \{0,1\}^n$. An analyst asks how many people who satisfy a condition on the identifiers have secret bit $1$. A condition such as "the name is Alice, Bob or Eve" can single out any set of people, so a query is a subset $S \subseteq [n] = \{1, \dots, n\}$. We identify it with its indicator vector $S \in \{0,1\}^n$, whose entry $i$ is $1$ when person $i$ is in the set and $0$ otherwise.

A curator holds the secret bits and answers the queries. The definition assumes only a bound $E$ on how far each answer may be from the truth.

::: {#def-rec-curator name="Subset queries and a curator with error $E$"}
Let $d \in \{0,1\}^n$ be the vector of secret bits. The true answer to a **subset query** $S \in \{0,1\}^n$ is the number of people in $S$ whose bit is $1$,

$$
a(S) = \sum_{i \in S} d_i = \langle S, d \rangle ,
$$

where $\langle S, d \rangle = \sum_i S_i d_i$ is the inner product of the indicator vector $S$ with $d$.

A **curator with error $E$** answers each query $S$ with a response $r(S)$, a number within $E$ of the truth, $|r(S) - a(S)| \le E$. Within that bound, the curator may choose its answers in any way, depending on $d$ and on all the queries.
:::

The error need not be random. The curator may pick the most misleading answer in the allowed range, so every attack below also works against random noise. Some error is necessary, because the exact answer to the query $S = \{i\}$ is the secret bit of person $i$.

To measure how much an attacker learns, count the bits it gets wrong. The attacker outputs a guess $c \in \{0,1\}^n$ of the secret vector $d$, and the **Hamming distance** $\operatorname{dist}(c, d) = |\{i : c_i \ne d_i\}|$ is the number of positions where the guess is wrong. The next definition names the clearest failure of privacy: for every secret vector, some attacker gets all but a vanishing fraction of the bits right, except with a vanishing probability.

::: {#def-rec-blatant name="Blatant non-privacy"}
Consider a family of curators, one for each number of people $n$. The family is **blatantly non-private** if three things exist. The first is an attacker that interacts with the curator and outputs a guess $c \in \{0,1\}^n$. The second is a bound $g(n)$ on the number of wrong bits, with $g(n)/n \to 0$. The third is a failure probability $\beta(n)$, with $\beta(n) \to 0$. They must satisfy, for every $n$ and every secret vector $d \in \{0,1\}^n$,

$$
\Pr\left[\operatorname{dist}(c, d) \le g(n)\right] \ge 1 - \beta(n).
$$
:::

Such an attacker, with $o(n)$ wrong bits (a vanishing fraction of $n$), learns the secret of almost everyone. Whatever privacy means, this violates it.

## Asking every query {#sec-rec-exponential}

The first attack ignores cost. It asks all $2^n$ subset queries and outputs any candidate that agrees with every answer.

::: {#thm-rec-exponential name="Reconstruction from all subset queries"}
Let $E \ge 0$, and let a curator with error $E$ answer all $2^n$ subset queries, giving the answer $r(S)$ to the query $S$. An attacker that outputs any $c \in \{0,1\}^n$ satisfying

$$
\left|\langle S, c \rangle - r(S)\right| \le E \quad \text{for every } S \subseteq [n]
$$ {#eq-rec-consistent}

finds such a $c$, and every such $c$ differs from the secret vector $d$ in at most $4E$ positions, $\operatorname{dist}(c, d) \le 4E$.
:::

::: {.proof}
1. The true vector $d$ satisfies @eq-rec-consistent by @def-rec-curator, so a candidate exists.
2. Let $I_0 = \{i : d_i = 0\}$. The attacker asked the query $I_0$, so @eq-rec-consistent gives $\langle I_0, c\rangle \le r(I_0) + E$. The curator's bound gives $r(I_0) \le \langle I_0, d\rangle + E = E$. Hence $\langle I_0, c \rangle \le 2E$. This sum counts the positions of $I_0$ where $c_i = 1 \ne d_i$, so $c$ and $d$ differ in at most $2E$ positions of $I_0$.
3. Let $I_1 = \{i : d_i = 1\}$. In the same way, $\langle I_1, c \rangle \ge r(I_1) - E \ge \langle I_1, d \rangle - 2E = |I_1| - 2E$. The number of positions of $I_1$ where $c_i = 0$ is $|I_1| - \langle I_1, c\rangle \le 2E$.
4. The sets $I_0$ and $I_1$ partition $[n]$, so $\operatorname{dist}(c, d) \le 4E$.
:::

The attacker does not know $I_0$ and $I_1$, which is why it asks every query. With errors that are a vanishing fraction of $n$, the bound meets @def-rec-blatant.

::: {#cor-rec-exponential}
If $E = n/401$, the attacker of @thm-rec-exponential recovers more than 99% of the bits. If $E = E(n)$ with $E(n)/n \to 0$, the curator is blatantly non-private, with $g(n) = 4E(n)$ and $\beta(n) = 0$.
:::

::: {.proof}
For the first claim, $4n/401 < n/100$, since $4 \times 100 < 401$. The second claim is @thm-rec-exponential with @def-rec-blatant, because the attacker never fails.
:::

Errors of order $n$ are the most a counting query can bear, since every answer lies between $0$ and $n$. The next result shows that error $E$ always hides about $E/2$ bits, however many queries the attacker asks.

::: {#prp-rec-limit name="Error $E$ hides $E/2$ bits"}
Let $E \ge 1$ and $k = \min(\lfloor E \rfloor, n)$. For every attacker, which may ask any queries in any adaptive order, there are a vector $d \in \{0,1\}^n$ and a curator with error $E$ for which $\Pr[\operatorname{dist}(c, d) \ge k/2] \ge 1/2$.
:::

::: {.proof}
1. Let $d$ be all zeros, and let $d'$ have ones in its first $k$ positions and zeros elsewhere, so $\operatorname{dist}(d, d') = k$.
2. On the data $d'$, let the curator answer every query exactly, $r(S) = \langle S, d' \rangle$. On the data $d$, let it give the same answers. Their error is $|\langle S, d' - d\rangle| \le k \le E$, so both are curators with error $E$.
3. The answers to every query are the same on $d$ and on $d'$. So the attacker's output $c$ has the same distribution in the two cases.
4. For every $c$, the triangle inequality gives $\operatorname{dist}(c, d) + \operatorname{dist}(c, d') \ge k$, so at least one of the two distances is at least $k/2$. Hence $\Pr[\operatorname{dist}(c,d) \ge k/2] + \Pr[\operatorname{dist}(c,d') \ge k/2] \ge 1$, and one of the two probabilities is at least $1/2$.
:::

So for the attacker who asks everything, safety needs error of order $n$, and @thm-rec-exponential is tight up to a factor of $8$. But an attack with $2^n$ queries is not a practical threat once $n$ is more than a few dozen.

## Random queries {#sec-rec-random}

The second attack asks about $n$ queries, each a uniformly random subset of the people. It defeats errors of order $\sqrt n$, much smaller than the first attack can handle but still large next to the error a useful statistic can tolerate.

The idea is to fix a wrong candidate $c$ and show that a random query is likely to expose it. If $c$ differs from $d$ in $k$ positions, the answers $\langle S, c \rangle$ and $\langle S, d \rangle$ differ by a sum of about $k/2$ random signs, which is typically of size $\sqrt k$. A curator with error $E$ can hide a difference of at most $2E$, so when $\sqrt k$ is much larger than $E$, a random query rules $c$ out with constant probability. To make "typically" precise, the proof needs a lower bound on the size of a random sum, the opposite of a concentration inequality.

::: {#lem-rec-anticoncentration name="A random subset sees a fixed vector"}
Let $w \in \mathbb R^n$, with Euclidean length $\|w\|_2 = (\sum_i w_i^2)^{1/2}$. Let $S$ be a uniformly random subset of $[n]$, so that each $i$ is in $S$ independently with probability $1/2$. Then the sum $\langle S, w\rangle$ of the entries of $w$ in $S$ satisfies

$$
\Pr\left[\,|\langle S, w \rangle| \ge \tfrac14 \|w\|_2\,\right] \ge \frac{3}{32}.
$$
:::

::: {.proof}
1. If $w = 0$ the event always holds, so let $w \ne 0$. Write $S_i = (1 + \xi_i)/2$, where the signs $\xi_i \in \{-1, +1\}$ are independent and uniform. Then $\langle S, w\rangle = \tfrac12(a + Z)$, with the constant $a = \sum_i w_i$ and the random sum $Z = \sum_i \xi_i w_i$.
2. *Moments of $Z$.* Expand the powers and use independence. A product of the $\xi_i$ has mean $0$ unless each index appears an even number of times. So $\mathbb E[Z^2] = \sum_i w_i^2 = \|w\|_2^2$, and

$$
\mathbb E[Z^4] = \sum_i w_i^4 + 3\sum_{i \ne j} w_i^2 w_j^2 = 3\|w\|_2^4 - 2\sum_i w_i^4 \le 3\|w\|_2^4 .
$$

3. *Paley–Zygmund inequality.* Let $Y \ge 0$ have $\mathbb E[Y] > 0$, and let $s \in [0,1]$. Split $\mathbb E[Y]$ according to whether $Y > s\,\mathbb E[Y]$, and apply the Cauchy–Schwarz inequality to the second part:

$$
\mathbb E[Y] \le s\,\mathbb E[Y] + \mathbb E\left[Y \mathbf 1[Y > s\,\mathbb E[Y]]\right] \le s\,\mathbb E[Y] + \sqrt{\mathbb E[Y^2]\,\Pr[Y > s\,\mathbb E[Y]]}.
$$

Rearranging gives $\Pr[Y > s\,\mathbb E[Y]] \ge (1-s)^2\,\mathbb E[Y]^2/\mathbb E[Y^2]$.

4. Apply step 3 to $Y = Z^2$ with $s = 1/4$, using step 2. The event $|Z| > \|w\|_2/2$ has probability at least $(3/4)^2/3 = 3/16$.
5. *Removing the constant $a$.* For every $z$, $|a + z| + |a - z| \ge 2|z|$, so $|a + Z| \ge |Z|$ or $|a - Z| \ge |Z|$. Flipping all signs maps $Z$ to $-Z$ and preserves the distribution and the event of step 4. Hence the events $\{|Z| > \|w\|_2/2,\ |a+Z| \ge |Z|\}$ and $\{|Z| > \|w\|_2/2,\ |a-Z| \ge |Z|\}$ have the same probability. Together they cover the event of step 4, so each has probability at least $3/32$.
6. On the first of these events, $|\langle S, w\rangle| = \tfrac12|a + Z| \ge \tfrac12|Z| > \tfrac14\|w\|_2$.
:::

The lemma rules out one wrong candidate with constant probability per query. A union bound over the $2^n$ candidates in $\{0,1\}^n$ then shows that about $n$ queries rule out all the wrong ones at once.

::: {#thm-rec-random name="Reconstruction from random queries"}
Let $E \ge 0$ and $\beta \in (0, 1)$, and let

$$
m \ge \frac{32}{3}\left(n \ln 2 + \ln\frac1\beta\right).
$$

The attacker draws $m$ independent, uniformly random subsets $S_1, \dots, S_m$ of $[n]$ before seeing any answer. A curator with error $E$ answers them with $r_1, \dots, r_m$. The attacker outputs any $c \in \{0,1\}^n$ with $|\langle S_j, c\rangle - r_j| \le E$ for all $j$. Then such a $c$ exists and, with probability at least $1-\beta$ over the queries, every such $c$ has

$$
\operatorname{dist}(c, d) \le 64E^2 .
$$ {#eq-rec-random}
:::

::: {.proof}
1. The true vector $d$ satisfies all $m$ constraints by @def-rec-curator, so a candidate exists.
2. *One far candidate.* Fix $c$ with $k = \operatorname{dist}(c, d) > 64E^2$, and let $w = c - d$. Its entries are in $\{-1, 0, 1\}$ and exactly $k$ are nonzero, so $\|w\|_2 = \sqrt k > 8E$. By @lem-rec-anticoncentration, each query has $|\langle S_j, w\rangle| \ge \sqrt k/4 > 2E$ with probability at least $3/32$. The queries are independent, so the probability that $|\langle S_j, w\rangle| \le 2E$ for every $j$ is at most $(1 - 3/32)^m \le e^{-3m/32}$.
3. *All far candidates.* There are at most $2^n$ candidates. By the union bound, the event $B$ that some candidate with $\operatorname{dist}(c, d) > 64E^2$ has $|\langle S_j, c - d\rangle| \le 2E$ for every $j$ has probability at most $2^n e^{-3m/32}$, which is at most $\beta$ by the choice of $m$.
4. *Consistent candidates are close.* Suppose $B$ does not occur. If $c$ is consistent with the answers, the triangle inequality gives $|\langle S_j, c - d\rangle| \le |\langle S_j, c\rangle - r_j| + |r_j - \langle S_j, d \rangle| \le 2E$ for every $j$. So $c$ is not far, which is @eq-rec-random. The event $B$ depends only on the queries, so this holds whatever answers the curator chose.
:::

The number of queries is about $7.4n$: for $n = 1{,}000$ and $\beta = 0.01$ the theorem asks for $m = 7{,}443$. Setting $\beta = 2^{-n}$ gives the asymptotic statement.

::: {#cor-rec-random}
If $E = E(n)$ with $E(n)/\sqrt n \to 0$, the curator is blatantly non-private against $15n$ random queries, with $g(n) = 64E(n)^2$ and $\beta(n) = 2^{-n}$. If $E = \alpha\sqrt n$, the same attacker gets at most $64\alpha^2 n$ bits wrong with probability at least $1 - 2^{-n}$.
:::

::: {.proof}
With $\beta = 2^{-n}$, the bound on $m$ in @thm-rec-random is $\frac{32}{3}\cdot 2n\ln 2 \approx 14.79\,n \le 15n$. Then $g(n) = 64E(n)^2 = 64 n\,(E(n)/\sqrt n)^2$, and $g(n)/n \to 0$.
:::

### An efficient attack {#sec-rec-lp}

The attacker of @thm-rec-random asks few queries but searches $2^n$ candidates. A search over fractional vectors is a linear program, so this attack is called linear programming (LP) decoding. It looks for $x \in [0,1]^n$ with $|\langle S_j, x\rangle - r_j| \le E$ for all $j$, which takes time polynomial in $n$ and $m$, and rounds $x$ to the nearest bit vector. The analysis is the same as before with one extra step: a fractional candidate is not one of $2^n$ points, so a fine grid replaces the set of candidates.

::: {#thm-rec-lp name="Reconstruction by linear programming"}
Let $E > 0$, $\beta \in (0,1)$, $K = \lceil n/(2E) \rceil$, and

$$
m \ge \frac{32}{3}\left(n \ln(2K+1) + \ln\frac1\beta\right).
$$

The attacker draws $m$ random subsets and receives answers from a curator with error $E$, as in @thm-rec-random. It finds any $x \in [0,1]^n$ with $|\langle S_j, x\rangle - r_j| \le E$ for all $j$, and it outputs $c_i = 1$ if $x_i \ge 1/2$ and $c_i = 0$ otherwise. With probability at least $1 - \beta$ over the queries,

$$
\operatorname{dist}(c, d) \le 676E^2 .
$$
:::

::: {.proof}
1. The vector $d$ is feasible, so the linear program has a solution $x$. Let $z = x - d \in [-1, 1]^n$. As in step 4 of @thm-rec-random, $|\langle S_j, z\rangle| \le 2E$ for every $j$.
2. *Rounding errors.* If $c_i \ne d_i$, then either $d_i = 0$ and $x_i \ge 1/2$, or $d_i = 1$ and $x_i < 1/2$. In both cases $|z_i| \ge 1/2$, so $(2z_i)^2 \ge 1$. Summing over $i$ gives $\operatorname{dist}(c, d) \le 4\|z\|_2^2$.
3. *The grid.* Let $h = 1/K \le 2E/n$, and let $G$ be the set of vectors whose entries are multiples of $h$ in $[-1, 1]$. It has $(2K+1)^n$ points. Round each entry of $z$ to the nearest multiple of $h$ to get $z' \in G$, with $|z_i - z'_i| \le h/2 \le E/n$. Then $|\langle S_j, z - z'\rangle| \le \sum_i |z_i - z'_i| \le E$, so $|\langle S_j, z'\rangle| \le 3E$ for every $j$. Also $\|z - z'\|_2 \le \sqrt n \cdot E/n \le E$.
4. *Far grid points are ruled out.* Fix $g \in G$ with $\|g\|_2 > 12E$. By @lem-rec-anticoncentration, each query has $|\langle S_j, g\rangle| \ge \|g\|_2/4 > 3E$ with probability at least $3/32$. As in @thm-rec-random, the probability that $|\langle S_j, g\rangle| \le 3E$ for every $j$ is at most $e^{-3m/32}$. A union bound over the $(2K+1)^n$ grid points bounds the probability that this happens for some far grid point by $(2K+1)^n e^{-3m/32} \le \beta$.
5. Outside that event, step 3 shows that $z'$ is not far, so $\|z'\|_2 \le 12E$ and $\|z\|_2 \le 13E$. Step 2 gives $\operatorname{dist}(c,d) \le 4 \cdot 169 E^2 = 676E^2$.
:::

For $1 \le E \le n$, the factor $\ln(2K+1)$ is at most $\ln(n+3)$, so the attack runs in polynomial time with $O(n \log n)$ queries and gets at most $676E^2$ bits wrong. So error $o(\sqrt n)$ again lets an attacker recover all but a vanishing fraction of the bits. The constants $64$ and $676$ are far from the truth, as @fig-rec-attack shows: on simulated data the attack recovers almost every bit at errors the theorem cannot handle.

![The linear-programming attack on simulated data, with $m = 8n$ random subset queries and an error drawn uniformly from $[-E, E]$ for each answer. The attacker finds a point of the feasible set of @thm-rec-lp, up to a violation of at most $0.01$, by projected gradient descent and rounds it. Each point averages $1600/n$ runs. The curves for different $n$ nearly coincide, so the error the attack tolerates grows like $\sqrt n$. For $n = 400$, a single run gets none of the bits wrong at $E = \sqrt n/2$ and 17% of them at $E = 2\sqrt n$.](figures/rec-attack.svg){#fig-rec-attack}

Bounds on random matrices, which this course does not develop, show that $O(n)$ queries suffice for the linear program (Dwork and Roth, 2014, Section 8.1). They also show that a linear program that minimizes the total error tolerates a constant fraction of arbitrary answers (Dwork, McSherry and Talwar, 2007). The argument above gives an inefficient version of that tolerance.

::: {#prp-rec-robust name="Tolerating some arbitrary answers"}
Let $E \ge 0$, let $\beta \in (0,1)$ be a failure probability, let $\rho \in [0, 3/128)$ be the fraction of answers that may be arbitrary, and let

$$
m \ge \frac{2048}{9}\left(n\ln 2 + \ln\frac1\beta\right).
$$

The attacker draws $m$ random subsets as in @thm-rec-random. The curator answers at least $(1-\rho)m$ of them with error at most $E$ and the others arbitrarily, choosing which after seeing the queries. The attacker outputs any $c \in \{0,1\}^n$ with $|\langle S_j, c\rangle - r_j| \le E$ for at least $(1-\rho)m$ of the queries. With probability at least $1 - \beta$, every such $c$ has $\operatorname{dist}(c, d) \le 64E^2$.
:::

::: {.proof}
1. The true vector $d$ meets the condition, so a candidate exists.
2. Fix $c$ with $\operatorname{dist}(c,d) > 64E^2$ and let $V_j = \mathbf 1[|\langle S_j, c - d\rangle| > 2E]$ indicate that query $j$ exposes $c$. As in step 2 of @thm-rec-random, the $V_j$ are independent with mean at least $3/32$. Hoeffding's inequality gives $\Pr\left[\sum_j V_j < 3m/64\right] \le \exp(-2m(3/64)^2) = e^{-9m/2048}$.
3. A union bound over at most $2^n$ candidates shows that, with probability at least $1 - 2^n e^{-9m/2048} \ge 1-\beta$, every far candidate has $\sum_j V_j \ge 3m/64$.
4. Suppose this holds, and let $c$ be an output. The queries where $c$ is consistent and those answered with error at most $E$ each number at least $(1-\rho)m$, so at least $(1-2\rho)m$ queries are of both kinds. On each of them, the triangle inequality gives $|\langle S_j, c - d\rangle| \le 2E$, so $V_j = 0$. Hence $\sum_j V_j \le 2\rho m < 3m/64$, and $c$ is not far.
:::

These attacks are nearly tight. The chapters on [approximate differential privacy](../part02-dp-basics/03-approximate-dp.md) and [advanced composition](../part02-dp-basics/04-advanced-composition.md) show that a curator can answer $m$ subset queries with error of order $\sqrt m$ and still meet the definition of differential privacy, up to logarithmic factors and factors that depend on the privacy parameters. So for $n$ queries, the line between blatant non-privacy and a formal guarantee runs at error $\sqrt n$.

## Reconstruction in practice {#sec-rec-practice}

The random-query attack seems theoretical, because it asks about arbitrary sets of people, but it broke a deployed system. The company Aircloak built Diffix, an interface that let analysts ask an unlimited number of counting queries on sensitive data. It promised privacy with less error than differential privacy adds.

Diffix had defences against the attacks above. Each condition in a query added its own Gaussian noise with standard deviation $1$, so the error grew with the number of conditions. The interface also rejected conditions joined by OR, suppressed small counts and altered extreme values. A random subset of $n$ people needs about $n/2$ conditions joined by OR to describe person by person, so both defences block that route.

Cohen and Nissim reversed the order of the attack. Instead of a random set that they then had to describe, they drew a random single condition from a family whose sets look random. Each condition tests a digit of a power of each person's numeric identifier. One query they sent kept a row when

$$
\left\lfloor 100\,(2\cdot\text{identifier})^{0.7} + 0.5 \right\rfloor = \left\lfloor 100\,(2\cdot\text{identifier})^{0.7} \right\rfloor ,
$$

which holds exactly when the third decimal digit of $(2\cdot\text{identifier})^{0.7}$ is at most $4$. Of the $1{,}001$ identifiers from $2000$ to $3000$, it keeps $501$, about half, as a random subset would.

Each such query has few conditions, so its error does not grow with the number of rows. Diffix added noise of standard deviation $4$ to it. Cohen and Nissim targeted four ranges of identifiers with $73$ to $142$ rows each and asked $3{,}500$ queries about each range. Gaussian noise is not bounded, so they used the linear program that minimizes the total error, which tolerates a few large errors as @prp-rec-robust does. It recovered the secret attribute of every row in all four ranges.

The theorems of this chapter do not apply to these queries as stated, because a fixed formula is not a random subset. But the proofs use the randomness of the queries only through @lem-rec-anticoncentration, by which each query exposes a fixed wrong candidate with constant probability. The digit tests behave that way on real identifiers, as an empirical fact and not a theorem.

## Summary {#sec-rec-summary}

| Result | Statement |
| --- | --- |
| @exm-rec-ages | a median and a mean cut $341{,}376$ possible ages to $36$ |
| @def-rec-blatant | an attacker recovers all but $o(n)$ secret bits with probability tending to $1$ |
| @thm-rec-exponential | all $2^n$ queries with error $E$ reveal all but $4E$ bits |
| @prp-rec-limit | error $E$ can hide $E/2$ bits from any attacker |
| @lem-rec-anticoncentration | a random subset has $\lvert\langle S, w\rangle\rvert \ge \|w\|_2/4$ with probability at least $3/32$ |
| @thm-rec-random | $\frac{32}{3}(n\ln 2 + \ln\frac1\beta)$ random queries with error $E$ reveal all but $64E^2$ bits |
| @thm-rec-lp | a linear program does the same in polynomial time with $O(n\log n)$ queries, all but $676E^2$ bits |
| @prp-rec-robust | a fraction below $3/128$ of arbitrary answers does not stop the attack |

: The chapter's results. {#tbl-rec-summary}

## References {#sec-rec-notes}

- Simson Garfinkel, John M. Abowd and Christian Martindale, "Understanding Database Reconstruction Attacks on Public Data", 2019. Reconstructs census blocks from published tables with a solver.
- John M. Abowd, "Staring Down the Database Reconstruction Theorem", 2019. Reports the Census Bureau's reconstruction of the 2010 census.
- Irit Dinur and Kobbi Nissim, "Revealing Information while Preserving Privacy", 2003. Defines blatant non-privacy and gives both attacks.
- Cynthia Dwork, Frank McSherry and Kunal Talwar, "The Price of Privacy and the Limits of LP Decoding", 2007. Gives the LP attack with $O(n)$ queries that tolerates some arbitrary answers.
- Cynthia Dwork and Aaron Roth, *The Algorithmic Foundations of Differential Privacy*, 2014. Presents both attacks in Section 8.1.
- Raymond Paley and Antoni Zygmund, "A Note on Analytic Functions in the Unit Circle", 1932. Gives the inequality in @lem-rec-anticoncentration.
- Aloni Cohen and Kobbi Nissim, "Linear Program Reconstruction in Practice", 2020. Breaks Diffix with digit-test queries.
