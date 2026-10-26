# Syntactic anonymity and how it breaks

[Linkage and re-identification](07-linkage-reidentification.md) showed that deleting names does not anonymize a table. The columns that remain, such as age, ZIP code and sex, are **quasi-identifiers**. None of them names anyone, but together they single out most people, so anyone with a list of names and the same columns can join the two tables. Between 1998 and 2008, researchers answered with definitions that constrain the released table itself. k-anonymity hides every record in a crowd of at least $k$ records that look alike. l-diversity asks each crowd to hold several different values of the sensitive column, such as a diagnosis, and t-closeness asks each crowd to have the same mix of those values as the whole table. They are called **syntactic** because each is a property of the published table that you can check by reading it. Statistics offices and anonymization tools such as ARX still use them, so you need to know exactly where each promise ends.

The one idea of this chapter is that a syntactic definition fixes the adversary's knowledge in advance. For k-anonymity the adversary knows the quasi-identifiers and nothing else. For l-diversity it also knows a few facts about the target, and for t-closeness the table's overall distribution. Each attack below gives the adversary one more piece of knowledge, and the guarantee then says nothing. A second release, the most ordinary auxiliary information there is, breaks all three definitions.

The running example is a synthetic hospital table of 10,000 patients, with age, ZIP code and sex as quasi-identifiers and one of eight diagnoses as the sensitive attribute. Released with exact quasi-identifiers, it lets an adversary who knows a random patient's three columns find that patient's record with probability 0.933. The chapter anonymizes it under each definition, and finally shows two independent anonymized releases revealing more than three times as many diagnoses as either alone.

| Symbol | Meaning |
| --- | --- |
| $n$; $S$, $m$ | records; the sensitive attribute and its number of values |
| $E$, $\lvert E\rvert$; $J$ | a class and its size; the number of classes |
| $n_E(v)$; $P_E(v)$, $P(v)$ | records of $E$ with $S = v$; their fraction in $E$, and in the table |
| $k$; $K$, $\lambda_k$ | the smallest class size; cells of a model, and people per cell needed |
| $\ell$, $c$; $r_1 \ge \dots \ge r_m$ | the parameters of l-diversity; a class's sorted value counts |
| $t$; $P_E(A)$ | the parameter of t-closeness; the probability of a set $A$ of values |

: The symbols of this chapter. {#tbl-anon-symbols}

## k-anonymity {#sec-anon-k}

Sweeney and Samarati proposed the first answer to the linkage attack in 1998. Their idea is to make sure that every combination of quasi-identifiers in the release is shared by at least $k$ people. A custodian gets there by generalizing, so that age 37 becomes 35–39 and ZIP code 02139 becomes 021\*\*, and by suppressing the few records that still stand out. This section asks what the definition buys against linkage, and what it costs in the detail the quasi-identifiers can keep.

The definition needs words for what the custodian, the data holder, does to the table and for what the result looks like.

::: {#def-anon-k name="k-anonymity"}
A **generalization** replaces each quasi-identifier of a record by a coarser value, such as an age by a range or a ZIP code by a prefix. **Suppression** deletes records. An **equivalence class** $E$ of the released table is a set of records with the same generalized quasi-identifiers, and $\lvert E\rvert$ is its number of records. A release is **k-anonymous** if every equivalence class has at least $k$ records.
:::

The linkage game of [Linkage and re-identification](07-linkage-reidentification.md) draws a target uniformly from a release of $n$ records and gives the adversary the target's quasi-identifiers. The adversary wins if it picks out the target's record. In a class of $\lvert E\rvert$ records it is right with probability $1/\lvert E\rvert$, and the target lands there with probability $\lvert E\rvert/n$. So each class contributes $1/n$, and @eq-link-success gives a success of $J/n$ for $J$ classes. Large classes mean few classes, and that is the whole guarantee.

::: {#prp-anon-k-link name="k-anonymity caps linkage"}
If a release of $n$ records is k-anonymous, then the adversary's success in the linkage game is at most $1/k$:
$$\Pr[\text{win}] = \frac{J}{n} \le \frac1k.$$ {#eq-anon-k-link}
:::

::: {.proof}
1. *Count records by class.* The $J$ classes split the $n$ records between them, each record in exactly one class, so $n = \sum_j\lvert E_j\rvert$.
2. *Use the definition.* Every class has at least $k$ records, so the sum is at least $Jk$. Then $n \ge Jk$, which is $J/n \le 1/k$.
3. *Read off the linkage success.* @eq-link-success says that the linkage success is $J/n$, so it is at most $1/k$.
:::

The guarantee is about names only. @eq-anon-k-link bounds the chance of picking out the target's *record*, and says nothing about the sensitive attribute, which the next two sections attack. It also needs the right columns treated as quasi-identifiers. In data with many columns and few filled entries, such as movie ratings, every column is one (@sec-link-sparse), and k-anonymity then leaves little of the table (Aggarwal, 2005).

Coarse quasi-identifiers protect more but say less, so the next question is how coarse they must be. Suppose the generalized quasi-identifiers split $n$ people into $K$ cells, each person falling into a cell independently and every cell equally likely, and no record is suppressed, so each cell is a class. Naming a person's cell then takes $\log_2 K$ bits, which measures how much the quasi-identifiers say. Each of the other $n - 1$ people lands in a given person's cell with probability $1/K$, so the number of others there is a $\mathrm{Bin}(n - 1, 1/K)$ count. With many people and many cells, @thm-dist-poisson-limit in [Probability distributions](../part01-prerequisites/01-distributions.md) shows that this count is close to a Poisson count with the same mean, $(n - 1)/K \approx n/K$.

::: {#prp-anon-k-violate name="The cost of k-anonymity for uniform cells"}
Suppose $n$ people fall independently into $K$ equally likely cells, and treat the number of other people in a person's cell as a $\mathrm{Poisson}(n/K)$ count. Then the expected fraction of people in classes smaller than $k$ is
$$\Pr[\mathrm{Poisson}(n/K) \le k - 2].$$ {#eq-anon-k-violate}
Suppose the custodian accepts that at most 1% of records fall in such classes, to be suppressed. Let $\lambda_k$ be the mean at which the fraction is exactly 1%, that is $\Pr[\mathrm{Poisson}(\lambda_k) \le k - 2] = 0.01$. The fraction is at most 1% exactly when $n/K \ge \lambda_k$, so the quasi-identifiers may carry at most
$$\log_2 n - \log_2\lambda_k \text{ bits.}$$
:::

::: {.proof}
The plan is to compute the chance that one person's class is too small, note that this chance is also the expected fraction of such people, and then solve for the coarseness.

1. *The size of a person's class.* A person's class is their cell, which holds the person plus $N$ others, with $N$ a $\mathrm{Poisson}(n/K)$ count by assumption.
2. *When it is too small.* The class is smaller than $k$ when $1 + N \le k - 1$, that is when $N \le k - 2$, which has probability $\Pr[\mathrm{Poisson}(n/K) \le k - 2]$.
3. *From one person to the fraction.* The expected number of people in small classes is the sum, over people, of each one's chance of being in one, because the expected count of events is the sum of their probabilities. Every person has the same chance, so the sum is $n$ times it, and the expected fraction is that chance.
4. *Solve for the coarseness.* A Poisson count with a larger mean is more likely to exceed any fixed level, so the chance that it is at most $k - 2$ falls as the mean grows. The fraction is therefore at most 1% exactly when $n/K \ge \lambda_k$, that is $K \le n/\lambda_k$. Taking $\log_2$ of both sides gives $\log_2 K \le \log_2 n - \log_2\lambda_k$.
:::

So k-anonymity costs more than $\log_2 k$ bits below $\log_2 n$, the naive cost of exactly $k$ people per cell. Cells vary in size around their mean, and the unlucky small cells violate the definition, so you need $\lambda_k$ people per cell *on average*, more than $k$. For $k = 2$ a person violates the definition when $N = 0$, which has probability $e^{-\lambda}$. Setting $e^{-\lambda} = 0.01$ gives $\lambda_2 = \ln 100 = 4.61$, more than twice $k$.

::: {#exm-anon-bits}
@tbl-anon-bits applies @eq-anon-k-violate to a population of $n = 10{,}000$, with $\log_2 n = 13.29$ bits. For each $k$ it finds $\lambda_k$, then subtracts $\log_2 \lambda_k$ from $\log_2 n$. For $k = 2$, $\lambda_2 = 4.61$ from the calculation above, $\log_2 4.61 = 2.20$, and $13.29 - 2.20 = 11.08$ bits are allowed.

| $k$ | People per cell $\lambda_k$ | Bits allowed | Below $\log_2 n$ by | $\log_2 k$ |
| --- | --- | --- | --- | --- |
| 2 | 4.61 | 11.08 | 2.20 | 1.00 |
| 5 | 10.05 | 9.96 | 3.33 | 2.32 |
| 10 | 17.40 | 9.17 | 4.12 | 3.32 |
| 20 | 30.58 | 8.35 | 4.93 | 4.32 |
| 50 | 66.74 | 7.23 | 6.06 | 5.64 |

: Uniform cells, $n = 10{,}000$: how many bits of quasi-identifier k-anonymity allows if at most 1% of records may fall in classes smaller than $k$. {#tbl-anon-bits}

The last two columns compare the true cost with the naive $\log_2 k$. The gap is about 1.2 bits at $k = 2$ ($2.20 - 1.00$) and shrinks to about 0.4 bits at $k = 50$ ($6.06 - 5.64$), because a Poisson count with a large mean varies little relative to its mean. @fig-anon-bits draws the whole curve for each $k$.

At the other extreme, 99% of people are unique once the quasi-identifiers carry $\log_2n + 6.64$ bits (@sec-link-unique). Only $6.64 + 2.20 = 8.84$ bits, about 9, separate "almost everyone is unique" from "almost everyone is hidden in a pair", and every published attribute uses some of them.
:::

![The fraction of records in classes smaller than $k$, against the bits of quasi-identifier beyond $\log_2 n$, for uniform cells. Each curve is a Poisson CDF (cumulative distribution function); the dotted line is 1% of records.](figures/anon_bits.svg){#fig-anon-bits}

Real tables are not uniform, so real anonymizers search for a generalization that works on the data at hand. This chapter's script uses the algorithm of LeFevre, DeWitt & Ramakrishnan (2006).

::: {#def-anon-mondrian name="Mondrian"}
Given a table and a criterion that a class must meet, such as having at least $k$ records, **Mondrian** starts with the whole table as one box. It splits a box at the median of the quasi-identifier whose values in the box span the largest share of their full range, as long as both halves still meet the criterion. It stops when no split is allowed. Each final box is a class, published as a range per quasi-identifier.
:::

Because Mondrian splits until a half would fail the criterion, its classes stay close to size $k$, and [Exercise 5](../../book-exercises/part02-attacks/08-syntactic-anonymity.md#sec-sol-anon-5) shows that with no tied values they have between $k$ and $2k - 1$ records.

::: {#exm-anon-mondrian}
In the hospital table of $n = 10{,}000$ patients, age runs from 0 to 89, the ZIP code takes 800 values with a few areas far more common than the rest, and sex has two values. The diagnosis depends on age. Asthma peaks in childhood, fractures around 18 and depression around 40, while diabetes, hypertension and cancer come later in life. Flu is equally common at every age, and HIV is rare (1.0% of records). The exact linkage success of 0.933 means, by @eq-link-success, that the patients fall into $0.933 \times 10{,}000 = 9{,}330$ distinct combinations, so most are alone in theirs. @tbl-anon-k gives the result of Mondrian with k-anonymity as the criterion.

| $k$ | Classes | Sizes | Mean width | Linkage $J/n$ | Certain disclosure | Guess accuracy |
| --- | --- | --- | --- | --- | --- | --- |
| 2 | 4,091 | 2–5 | 0.005 | 0.4091 | 0.2797 | 0.675 |
| 5 | 1,542 | 5–11 | 0.016 | 0.1542 | 0.0232 | 0.593 |
| 10 | 748 | 10–20 | 0.028 | 0.0748 | 0.0000 | 0.562 |
| 20 | 366 | 20–41 | 0.043 | 0.0366 | 0.0000 | 0.540 |
| 50 | 129 | 50–99 | 0.075 | 0.0129 | 0.0000 | 0.506 |

: Mondrian k-anonymization of a synthetic hospital table of 10,000 patients. Mean width is the average, over records and quasi-identifiers, of the class's range as a fraction of the full range. Certain disclosure is the fraction of records in a class with a single diagnosis. Guess accuracy is the fraction of records whose diagnosis is the most common one in their class. {#tbl-anon-k}

The linkage success sits between $1/(2k)$ and $1/k$ for every $k$ (@fig-anon-tradeoff, left), because $J/n$ is one over the average class size. No class is smaller than $k$, so the upper bound $1/k$ is guaranteed by @prp-anon-k-link. The lower bound is not guaranteed, because ties in age and sex let Mondrian leave classes larger than $2k$, up to 41 records at $k = 20$, but they are rare enough that the average size stays below $2k$ in every row. At $k = 2$, for instance, $0.4091$ lies between $0.25$ and $0.5$. The cost in precision is small, since at $k = 5$ the average range covers 1.6% of each quasi-identifier's span. The last two columns are about the diagnosis, and the next section explains them.
:::

![Left: the linkage success of Mondrian releases of the synthetic hospital table against $k$, between $1/(2k)$ and the bound $1/k$. Right: how often the diagnosis is the most common one in the record's class, and how often the class holds a single diagnosis, against two baselines that use no release.](figures/anon_tradeoff.svg){#fig-anon-tradeoff}

## Homogeneity, background knowledge and l-diversity {#sec-anon-l}

Machanavajjhala, Kifer, Gehrke & Venkitasubramaniam (2006) pointed out that k-anonymity hides who a record belongs to but not what it says. The release exists to show a **sensitive attribute** $S$, such as a diagnosis, and @prp-anon-k-link says nothing about it. This section asks what an adversary learns about a target's sensitive value from a k-anonymous release, and what condition on the classes limits it. @tbl-anon-toy is a 3-anonymous release that shows the two attacks.

| ZIP code | Age | Diagnosis |
| --- | --- | --- |
| 130\*\* | < 30 | heart disease |
| 130\*\* | < 30 | heart disease |
| 130\*\* | < 30 | heart disease |
| 148\*\* | 4\* | cancer |
| 148\*\* | 4\* | flu |
| 148\*\* | 4\* | heart disease |
| 130\*\* | 3\* | cancer |
| 130\*\* | 3\* | cancer |
| 130\*\* | 3\* | flu |

: A 3-anonymous release with three classes of three records. {#tbl-anon-toy}

Both attacks, and the guarantee that answers them, compute the adversary's **posterior**, its belief about the target's sensitive value after it has seen the release. The adversary can find the target's class but not the target's record within it, so it believes the class's own frequency of each value.

::: {#lem-anon-posterior name="Within a class, the posterior is the frequency"}
Suppose the adversary knows the target's quasi-identifiers, knows that the target is in the release, and knows nothing else about the target. Let $E$ be the target's class, and let $n_E(v)$ be the number of records of $E$ with sensitive value $v$. Then the adversary's posterior is the class's frequency of $v$,
$$\Pr[S_{\text{target}} = v \mid \text{release}] = \frac{n_E(v)}{\lvert E\rvert} = P_E(v).$$ {#eq-anon-posterior}
If the adversary also knows that the target's value is not in a set $B$ of values, then for every value $v$ outside $B$,
$$\Pr[S_{\text{target}} = v \mid \text{release}, S_{\text{target}} \notin B] = \frac{n_E(v)}{\lvert E\rvert - \sum_{u \in B}n_E(u)}.$$
:::

::: {.proof}
1. *Every record of the class is equally likely to be the target's.* All records of $E$ show the same generalized quasi-identifiers and differ only in the sensitive value, which the adversary does not know. The target was drawn at random, so the adversary has no reason to favour any record, and each has probability $1/\lvert E\rvert$. The derivation of @eq-link-success uses the same step.
2. *Count the records with value $v$.* The target has value $v$ exactly when its record is one of the $n_E(v)$ records with that value. Adding $1/\lvert E\rvert$ over them gives $n_E(v)/\lvert E\rvert$.
3. *Cross out the excluded values.* Knowing that the target's value is not in $B$ rules out the records with values in $B$. The records that are left still look alike, so the adversary spreads its belief evenly over them. There are $\lvert E\rvert - \sum_{u \in B}n_E(u)$ of them, and step 2 applied to these records gives the second formula.
:::

Alice is 26 and lives in ZIP code 13053, so her class is the first. Every record in it says heart disease, and by @lem-anon-posterior her posterior for heart disease is $\frac33 = 1$. This is the **homogeneity attack**, which discloses the diagnosis without re-identifying anyone. Bob is 35, lives in 13068, and is in the third class, where two of three records say cancer, so his posterior for cancer is $\frac23$. His neighbour, the adversary here, knows that Bob never had flu that season and rules it out. Crossing out the flu record leaves two cancer records, and cancer then has posterior $\frac22 = 1$. This is the **background knowledge attack**.

Both attacks work because one value dominates the class, either from the start or after a few values are ruled out. l-diversity therefore asks for enough different values in each class, and its three versions differ in how they measure "enough". Sort a class's value counts from largest to smallest as $r_1 \ge r_2 \ge \dots \ge r_m$, so that $r_1$ is the count of the most common value. The entropy $H(P_E) = -\sum_v P_E(v)\log_2 P_E(v)$ measures in bits how spread out the values are, and $\ell$ equally common values have entropy exactly $\log_2 \ell$.

::: {#def-anon-l name="l-diversity"}
A class $E$ is

- **distinct $\ell$-diverse** if it holds at least $\ell$ different values;
- **entropy $\ell$-diverse** if $H(P_E) \ge \log_2\ell$, so that the class is at least as spread out as $\ell$ equally common values;
- **recursive $(c, \ell)$-diverse** if $r_1 < c\,(r_\ell + r_{\ell+1} + \dots + r_m)$, so that the most common value is less than $c$ times the total of all but the $\ell - 1$ most common values.

A release is $\ell$-diverse in one of these senses if every class is.
:::

The right-hand side of the recursive condition is the tail left after the $\ell - 1$ biggest values are set aside. That tail survives the background knowledge attack. An adversary who rules out $\ell - 2$ values can at worst remove big values, and the remaining values other than the top one still add up to at least the original tail. The remaining top value is less than $c$ times that, so its share is less than $c$ parts out of $c + 1$.

::: {#prp-anon-recursive name="What recursive diversity guarantees"}
Let a class be recursive $(c, \ell)$-diverse, and let the adversary know, besides the target's quasi-identifiers, that the target's value is not one of some $\ell - 2$ values. Then every remaining value $v$ has
$$\Pr[S_{\text{target}} = v \mid \text{release}, \text{background}] < \frac{c}{1 + c}.$$ {#eq-anon-recursive}
:::

::: {.proof}
The plan is to compare the class after elimination with the original class. Its top count cannot be larger and its tail cannot be smaller, so the defining inequality survives, and @lem-anon-posterior turns it into a posterior.

1. *What is left.* Rule out a set $B$ of $\ell - 2$ values. The remaining $m - \ell + 2$ values keep their original counts. Call the largest of them $r'_1$, and the total of the other $m - \ell + 1$ remaining values $R$.
2. *The top does not grow.* Ruling out values removes records but never adds any, so $r'_1$ is one of the original counts. No original count exceeds $r_1$, so $r'_1 \le r_1$.
3. *The tail does not shrink below the original tail.* $R$ is a sum of $m - \ell + 1$ of the original counts. The smallest such sum is that of the $m - \ell + 1$ smallest counts, which is $r_\ell + \dots + r_m$. So $R \ge r_\ell + \dots + r_m$.
4. *Combine with the definition.* Chaining step 2, the definition and step 3 gives
$$r'_1 \le r_1 < c\,(r_\ell + \dots + r_m) \le c\,R.$$
5. *Turn counts into a posterior.* By @lem-anon-posterior, after elimination the posterior of a remaining value is its count over the total count left, $r'_1 + R$, so it is at most $r'_1/(r'_1 + R)$. The share $x/(x + R)$ grows with $x$, like a larger slice of a pie whose other part is fixed. With $r'_1 < cR$ from step 4, the share is therefore below
$$\frac{cR}{cR + R} = \frac{c}{1 + c}.$$
:::

l-diversity rules out certainty but still allows high likelihood. The bound $c/(1 + c)$ is $\frac12$ at $c = 1$, $\frac23$ at $c = 2$ and $\frac34$ at $c = 3$, so even recursive diversity lets the adversary be fairly sure. Distinct $\ell$-diversity only says that the adversary must rule out $\ell - 1$ values to be certain, while one value can still hold all but $\ell - 1$ of the records. Entropy $\ell$-diversity does not limit the top value's share directly either. An entropy 3-diverse class over 8 values can have one value in 73% of its records, because the other 27% spread over seven values add enough entropy ([Exercise 2](../../book-exercises/part02-attacks/08-syntactic-anonymity.md#sec-sol-anon-2)).

The adversary's knowledge is assumed, not bounded. The proof allows $\ell - 2$ ruled-out values and no more, and the count is tight. A class with exactly $\ell$ values loses all but one of them when $\ell - 1$ are ruled out, and the last one then has posterior 1 whatever $c$ is. An adversary who knows something softer, such as "probably not flu", is not covered either.

Some tables cannot be made diverse at all. The whole table's distribution $P$ is the average of its classes' distributions $P_E$, each weighted by the class's share of the records, and averaging distributions cannot make them less spread out.

::: {#prp-anon-entropy-table name="A table with low entropy has no entropy-diverse release"}
If the whole table's distribution $P$ of the sensitive attribute has entropy below $\log_2\ell$, then no partition of the table into classes is entropy $\ell$-diverse.
:::

::: {.proof}
1. *The table is a mixture of its classes.* Pick a record uniformly at random, and let $C$ be its class and $S$ its sensitive value. Then $S$ has distribution $P$, and given that $C$ is the class $E$, it has distribution $P_E$.
2. *Conditioning lowers entropy on average.* [Information theory](../part01-prerequisites/05-information-theory.md) proves in @cor-min-conditioning that $H(S \mid C) \le H(S)$. Here $H(S \mid C)$ is the average of the class entropies $H(P_E)$, weighted by the class shares, and $H(S)$ is $H(P)$.
3. *Conclude.* If every class had $H(P_E) \ge \log_2\ell$, their weighted average would too, and step 2 would give $H(P) \ge \log_2\ell$, against the hypothesis.
:::

::: {#exm-anon-l}
In @tbl-anon-toy the three classes hold 1, 3 and 2 different values, so they are distinct 1-, 3- and 2-diverse. The largest $\ell$ for which a class is entropy $\ell$-diverse is $2^{H(P_E)}$, the number of equally common values with the same entropy. The first class has one value, entropy 0 and $\ell = 1.00$. The second has three equally common values and $\ell = 3.00$. The third, $\frac23$ cancer and $\frac13$ flu, has entropy $\frac23\log_2\frac32 + \frac13\log_2 3 = 0.918$ bits and $\ell = 2^{0.918} = 1.89$. It has two values, but they are unevenly spread, so they are worth fewer than two. The release as a whole is only 1-diverse, because of Alice's class.

In the hospital table, the "certain disclosure" column of @tbl-anon-k counts the patients in homogeneous classes. At $k = 2$ it is 28.0% of all patients. At $k = 5$ it is 2.3%, or 232 patients whose diagnosis the release states outright. The population never gives that certainty, since at no age does the population's distribution give any diagnosis more than 0.705.

A posterior has to be read against what the adversary would believe with no release, since 0.6 for a diagnosis is no leak if the population already gives 0.6 to people of the target's age. The right comparison is the **imputation baseline** of [Attribute inference and model inversion](03-attribute-inference.md), the best guess from the target's own quasi-identifiers under the true population, here from age alone, since the diagnosis depends only on age. Guessing the most common diagnosis of the whole table is right for 18.5% of patients. Imputation, which guesses the most likely diagnosis for each patient's exact age, is right for 53.6% (@fig-anon-tradeoff, right). A 5-anonymous release is right for 59.3%. So most of what the release teaches the adversary is the age–diagnosis pattern, which any useful study of this table would report. The excess over imputation, $0.593 - 0.536 = 0.057$ at $k = 5$, comes from the small classes, and it is down to $0.540 - 0.536 = 0.004$ at $k = 20$.

@tbl-anon-l adds diversity to $k = 5$.

| Criterion | Classes | Sizes | Mean width | Linkage $J/n$ | Certain disclosure | Guess accuracy |
| --- | --- | --- | --- | --- | --- | --- |
| $k = 5$ | 1,542 | 5–11 | 0.016 | 0.1542 | 0.0232 | 0.593 |
| $k = 5$, distinct $\ell = 3$ | 1,132 | 5–33 | 0.022 | 0.1132 | 0.0000 | 0.574 |
| $k = 5$, recursive $(c = 2, \ell = 3)$ | 389 | 5–641 | 0.113 | 0.0389 | 0.0000 | 0.417 |

: Mondrian with l-diversity on the same hospital table: each split is allowed only if both halves satisfy the criterion. {#tbl-anon-l}

Distinct 3-diversity removes the certain disclosures for little cost. Recursive $(2, 3)$-diversity forces some classes to cover hundreds of patients and pushes the guess accuracy below imputation. The reason is that diagnosis depends on age. To keep any diagnosis from dominating a class, Mondrian must mix ages, and that erases the pattern the data were released to show.
:::

## Skewness, similarity and t-closeness {#sec-anon-t}

l-diversity counts values, but what matters is how far the adversary's belief moves. Li, Li & Venkatasubramanian (2007) found two attacks that diversity does not see. In the **skewness attack**, 1% of the table has HIV, and a class that is half HIV is 2-diverse in every sense, yet by @lem-anon-posterior it raises the belief that a member has HIV fifty-fold. In the **similarity attack**, a class holds gastric ulcer, gastritis and stomach cancer, so it is 3-diverse, and still reveals a stomach disease. This section measures a class by its distance from the whole table's distribution, and asks what a bound on that distance guarantees.

::: {#def-anon-t name="t-closeness"}
Let $d$ be a distance between distributions on the values of the sensitive attribute. A release is **t-close** if every class's distribution is within distance $t$ of the whole table's, that is $d(P_E, P) \le t$ for every class $E$.
:::

The first choice of distance is total variation. @thm-div-tv in [Divergences](../part01-prerequisites/03-divergences.md) proves that the **total variation (TV)** distance, half the total absolute difference between two probability tables, equals the largest difference in probability that any question about the value can show:
$$\mathrm{TV}(P_E, P) = \frac12\sum_v\lvert P_E(v) - P(v)\rvert = \max_A\,\lvert P_E(A) - P(A)\rvert.$$
Here the maximum is over all sets of values $A$, and $P_E(A)$ is the probability under $P_E$ that the value lands in $A$. By @lem-anon-posterior the adversary's belief about a target in class $E$ is $P_E$, so a bound on the distance is a bound on every question the adversary can ask.

::: {#prp-anon-tv name="What t-closeness in total variation guarantees"}
If a release is t-close with the total variation distance, then for every class $E$ and every set of values $A$,
$$\lvert P_E(A) - P(A)\rvert \le t.$$ {#eq-anon-tv}
In particular, guessing the most common value of the target's class is right with probability at most $\max_vP(v) + t$, the best guess from the table alone plus $t$.
:::

::: {.proof}
1. *A distance bound is an event bound.* By @thm-div-tv, $\mathrm{TV}(P_E, P)$ is the largest of $\lvert P_E(A) - P(A)\rvert$ over all sets $A$, so a bound $t$ on the distance bounds every one of them by $t$.
2. *The adversary's belief is the class distribution.* By @lem-anon-posterior, the adversary's belief about a target in class $E$ is $P_E$. So every question it can ask about the target's value, "is it in $A$?", gets an answer within $t$ of the answer from the table as a whole.
3. *Apply it to the best guess.* Take $A = \{v\}$ for the class's most common value $v$. Then step 1 gives $P_E(v) \le P(v) + t$, and $P(v)$ is at most $\max_uP(u)$.
:::

The bound is additive, and rare values need a ratio. @eq-anon-tv lets a 1% value rise to $1\% + t$. At $t = 0.15$ that is 16%, a sixteen-fold increase for the people who most need protection. A difference of 0.15 looks small next to a common value such as 40%, and enormous next to 1%. A guarantee for rare values has to bound the *ratio* of the adversary's beliefs with and without the release, not their difference, and that is the shape of the definition in the next part.

Total variation treats every pair of values as equally different, so it does not see the similarity attack. The **earth mover's distance (EMD)**, the choice of Li et al., makes similar values close. Think of each distribution as piles of sand on the values, and of a ground distance between values as the cost of moving a unit of sand from one to the other.

::: {#def-anon-emd name="Earth mover's distance"}
Given a ground distance between values, the **earth mover's distance** $\mathrm{EMD}(P_E, P)$ is the least total cost of moving the mass of $P_E$ until it becomes $P$, where moving mass $f$ from value $i$ to value $j$ costs $f$ times their ground distance.
:::

For ordered values, such as salary bands or age groups, the least cost has a closed form. On a line, sand can only get from one side of a gap to the other by crossing it, so the work can be counted gap by gap.

::: {#prp-anon-emd name="Earth mover's distance on ordered values"}
Number $m$ ordered values $1$ to $m$, and place them at ground distance $\lvert i - j\rvert/(m - 1)$, so that the two ends are distance 1 apart. Then
$$\mathrm{EMD}(P_E, P) = \frac{1}{m - 1}\sum_{i=1}^{m-1}\Big\lvert\sum_{j \le i}\big(P_E(j) - P(j)\big)\Big\rvert.$$ {#eq-anon-emd}
:::

::: {.proof}
1. *Cost per gap.* A move from $i$ to $j$ crosses each of the $\lvert i - j\rvert$ gaps between them once, so its cost is $1/(m - 1)$ per unit of mass per gap crossed.
2. *How much must cross each gap.* Look at the gap between values $i$ and $i + 1$. The values up to $i$ hold $\sum_{j \le i}P_E(j)$ under the class and must end up holding $\sum_{j \le i}P(j)$. Their excess, or shortfall, is $\sum_{j \le i}(P_E(j) - P(j))$, and that net mass has to cross the gap, rightwards if it is positive and leftwards if it is negative. So any plan moves at least this much across the gap.
3. *A plan that does no more.* A plan that only moves mass between neighbours, and never in both directions across the same gap, moves exactly the net amount across each gap. So the least cost is $1/(m - 1)$ times the sum, over the $m - 1$ gaps, of the absolute net mass that crosses each one, which is @eq-anon-emd.
:::

The choice of distance is a choice of what counts as similar. EMD with a ground distance from a hierarchy of diagnoses treats gastritis and gastric ulcer as close, so a class full of stomach diseases is far from the table and fails the definition. On salaries, a class bunched at the low end is farther from a uniform table than a class spread across the range, as [Exercise 3](../../book-exercises/part02-attacks/08-syntactic-anonymity.md#sec-sol-anon-3) computes.

::: {#exm-anon-t}
In @tbl-anon-toy, the table's distribution is $\frac49$ heart disease, $\frac39$ cancer and $\frac29$ flu. Both distributions sum to 1, so the excesses and shortfalls are equal, and the half-sum of absolute differences is the total excess of the class over the table. Alice's class is all heart disease, an excess of $1 - \frac49 = \frac59 = 0.556$. The second class is $\frac13 = \frac39$ of each value, an excess of $\frac39 - \frac29 = \frac19 = 0.111$ on flu only. Bob's class, $\frac69$ cancer and $\frac39$ flu, has excesses $\frac39$ and $\frac19$, together $\frac49 = 0.444$. So the release is $t$-close only for $t \ge 0.556$.

For the skewness attack (@fig-anon-skew), the class that is 50% HIV is at total variation $0.50 - 0.01 = 0.49$ from a table with 1% HIV, the excess on HIV. A class with 10% HIV is at $0.10 - 0.01 = 0.09$, which passes $t = 0.1$, and still makes HIV ten times as likely for its members.

@tbl-anon-t adds t-closeness in total variation to $k = 5$ on the hospital table.

| Criterion | Classes | Sizes | Mean width | Largest TV | Guess accuracy | Bound $\max_vP(v) + t$ |
| --- | --- | --- | --- | --- | --- | --- |
| $k = 5$ | 1,542 | 5–11 | 0.016 | 0.937 | 0.593 | — |
| $k = 5$, $t = 0.3$ | 495 | 6–60 | 0.291 | 0.300 | 0.275 | 0.485 |
| $k = 5$, $t = 0.15$ | 136 | 14–304 | 0.324 | 0.149 | 0.222 | 0.335 |

: Mondrian with t-closeness (total variation) on the hospital table. The table's most common diagnosis has frequency 0.185. {#tbl-anon-t}

The bound column is $0.185 + t$, which is $0.185 + 0.3 = 0.485$ and $0.185 + 0.15 = 0.335$, and the guess accuracy respects it, as @prp-anon-tv says it must. But the classes now span about 30% of each quasi-identifier's range, and the guess accuracy of 0.222 to 0.275 is far below the 0.536 of imputation from age. Making every class t-close mixed ages until the age–diagnosis pattern, a fact about the population rather than a secret about anyone, was gone.
:::

![The skewness attack: a table in which 1% of patients have HIV, and a class of it that is half HIV. The class is 2-diverse but at total variation 0.49 from the table.](figures/anon_skew.svg){#fig-anon-skew}

## The composition attack {#sec-anon-composition}

Every definition so far is about one release, and Ganta, Kasiviswanathan & Smith (2008) asked what happens when there are two. Two hospitals in one city publish their patient tables, each properly k-anonymous or $\ell$-diverse, with no coordination, and a patient treated at both appears in a class of each. This section defines the attack that combines the two classes, computes how often it succeeds in a model, and runs it on the hospital table.

::: {#def-anon-composition name="The composition attack"}
Let a target appear in two releases, in class $E_1$ of the first and class $E_2$ of the second. The attack's **candidates** are the values that appear in both classes,
$$\{v : n_{E_1}(v) > 0\} \cap \{v : n_{E_2}(v) > 0\}.$$
The target's value is **pinned down** when a single candidate is left.
:::

The true value is always a candidate, because the target's own record carries it in each class, so a single candidate is the truth. Each release satisfies its definition, yet the combination can reveal the diagnosis outright. None of the three definitions **composes**, that is, survives the combination of two releases, because each is a condition on one table, and nobody checked the intersection of two classes. A hospital would need to know every other release to prevent it. A definition that survives must bound what two releases leak together, whatever the adversary knows, and the next part starts there.

To see how often the attack succeeds, model the classes as random. Let the target's class have $k_1$ records in the first release and $k_2$ in the second, and let the values of the other members be independent draws from the population, with $p_v$ the probability of value $v$. Given the true value $v$, let $W_1$ be the set of other values that appear among the $k_1 - 1$ other members of the first class, and $W_2$ the same for the second. These are the wrong values each release leaves open, and the target is pinned exactly when no wrong value is left open by both. Write $p(W) = \sum_{u \in W}p_u$ for the probability of a set of values $W$.

::: {#prp-anon-pinned name="How often two releases pin down a value"}
In this model, the probability that the two releases pin down the target's value is
$$\Pr[\text{pinned}] = \sum_vp_v\,\mathbb E\big[(1 - p(W_1))^{k_2 - 1}\big],$$ {#eq-anon-pinned}
where the expectation is over the random set $W_1$ for the given $v$. Its distribution follows by inclusion–exclusion over the values of a set $W$ that the draws might miss, so for every set $W$ of values other than $v$,
$$\Pr[W_1 = W] = \sum_{T \subseteq W}(-1)^{\lvert W\rvert - \lvert T\rvert}\big(p_v + p(T)\big)^{k_1 - 1}.$$
:::

::: {.proof}
The plan is to fix the true value and what the first class contains, compute the chance that the second class avoids it, and then average.

1. *Condition on the true value.* The target's true value is $v$ with probability $p_v$. Everything below is for a fixed $v$, and the outer sum $\sum_v p_v(\cdot)$ averages over it.
2. *When the value is pinned.* The candidates are $v$ itself together with the wrong values that appear in both classes, the set $W_1 \cap W_2$. So the value is pinned exactly when $W_1$ and $W_2$ are disjoint.
3. *Fix what the first class leaves open.* Condition on $W_1 = W$ for a particular set $W$. The $k_2 - 1$ others of the second class are independent draws, and each avoids $W$ with probability $1 - p(W)$, so
$$\Pr[W_2 \cap W = \emptyset] = (1 - p(W))^{k_2 - 1}.$$
Averaging over the possible sets $W$, weighted by $\Pr[W_1 = W]$, gives the expectation in @eq-anon-pinned.
4. *The distribution of the first class's set.* $W_1 = W$ means two things. Every one of the $k_1 - 1$ draws lands in $W \cup \{v\}$, and each value of $W$ is hit at least once. For any set $T$, the event that all draws land in $T \cup \{v\}$ has probability $(p_v + p(T))^{k_1 - 1}$. **Inclusion–exclusion** then handles the second requirement. Start from "all draws land in $W \cup \{v\}$", subtract the cases where some value of $W$ is missed, which means all draws land in a smaller set, add back the cases subtracted twice, and continue alternating over every number of missed values. Each term is indexed by the set $T \subseteq W$ that the draws may still hit, so the missed values are those of $W$ outside $T$, and the sign $(-1)^{\lvert W\rvert - \lvert T\rvert}$ alternates with their number.
:::

For a single wrong value $u$, the sum has two terms,
$$\Pr[W_1 = \{u\}] = (p_v + p_u)^{k_1 - 1} - p_v^{k_1 - 1}.$$
That is the probability that every draw is $v$ or $u$, minus the probability that every draw is $v$, in which case $u$ was never hit.

The model understates the risk, because it draws the other members of a class from the whole population. In real releases a class collects people of similar ages, who are more likely to share a diagnosis, so each class holds fewer distinct wrong values and the intersection is smaller.

The sum over subsets grows quickly with the number of values, so a simpler bound helps. The target escapes only if some wrong value turns up in both classes, and the chance of "some" is at most the sum of the chances of each.

::: {#prp-anon-union name="A union bound on pinning"}
In the same model,
$$\Pr[\text{pinned}] \ge 1 - \sum_vp_v\sum_{u \ne v}\big(1 - (1 - p_u)^{k_1 - 1}\big)\big(1 - (1 - p_u)^{k_2 - 1}\big).$$ {#eq-anon-union}
:::

::: {.proof}
1. *When it fails.* Given $v$, the value is not pinned when some $u \ne v$ is in both $W_1$ and $W_2$.
2. *The chance for one wrong value.* The value $u$ is missing from the first class when all $k_1 - 1$ others avoid it, which has probability $(1 - p_u)^{k_1 - 1}$. So $u$ is in $W_1$ with probability $1 - (1 - p_u)^{k_1 - 1}$, and likewise for $W_2$. The two classes are drawn independently, so the probability that $u$ is in both is the product.
3. *Add over wrong values.* The expected number of wrong values in both sets is the sum of these products, because the expected count of events is the sum of their probabilities. A count that is at least 1 whenever the event happens has an expectation at least the event's probability, so the chance that there is at least one shared wrong value is at most this sum. Subtracting from 1 and averaging over $v$ gives @eq-anon-union.
:::

The union bound charges every wrong value separately, so it is accurate only when two wrong values rarely turn up in both classes at once. With classes of 2, each class holds at most one wrong value, and the bound is exact. With classes of 5, the expected number of shared wrong values exceeds 1, the bound is negative, and @eq-anon-pinned is needed.

::: {#exm-anon-composition}
With the population's diagnosis distribution of the hospital table, the model's probability that two releases pin down the diagnosis is 0.8734 for classes of 2, 0.1615 for classes of 5, 0.0288 for classes of 7 and 0.0015 for classes of 10. It falls fast because a larger class contains more wrong values, and each one is another chance for the two classes to share one. Simulation agrees to within 0.001 (@fig-anon-composition, left). The union bound gives 0.8734 at $k = 2$ and is negative from $k = 5$ on.

The script then runs the attack. A city has 15,000 people drawn from the model of the hospital table. Two hospitals each treat a random 10,000 of them, so 6,649 people are treated at both. Each hospital publishes a 5-anonymous Mondrian release of its own patients. @tbl-anon-composition gives what the adversary learns about the 6,649.

| Releases | Pinned by release 1 alone | Pinned by release 2 alone | Pinned by both | Mean candidates, release 1 alone | Mean candidates, both |
| --- | --- | --- | --- | --- | --- |
| $k = 5$ | 0.0289 | 0.0262 | 0.0985 | 2.79 | 2.38 |
| $k = 5$, distinct $\ell = 3$ | 0.0000 | 0.0000 | 0.0027 | 3.22 | 2.92 |

: The composition attack on two independent hospital releases, for the 6,649 people treated at both. "Pinned" is the fraction left with a single possible diagnosis. {#tbl-anon-composition}

Two 5-anonymous releases pin down the diagnosis of 9.9% of the shared patients, more than three times either release alone (0.0985 against 0.0289 and 0.0262). At $k = 5$ a patient's class has 6.8 records on average over patients, so the nearest model case is classes of 7, and it predicts only 2.9%. Real classes are more alike than random draws. With distinct 3-diversity, no single release pins anyone down, and the two together pin down 18 patients (0.27% of 6,649).
:::

![Left: the probability that two releases pin down the diagnosis, against the class size, when the other members of each class are drawn from the population: exact (@eq-anon-pinned), simulated, and the union bound (@eq-anon-union). Right: the number of diagnoses still possible for a patient treated at both hospitals, from either 5-anonymous release alone and from their intersection.](figures/anon_composition.svg){#fig-anon-composition}

## Code {#sec-anon-code}

The script [code/syntactic_anonymity.py](../../code/part02-attacks/syntactic_anonymity.py) computes every number in this chapter and draws every figure. It implements Mondrian with each criterion of the chapter, the uniform-cells calculation, the attacks on the sensitive attribute, and the composition attack, exactly, by simulation and on two hospital releases. From the repository root:

```bash
uv run python code/part02-attacks/syntactic_anonymity.py
```

| Experiment | Result |
| --- | --- |
| Toy release | Posteriors 1 (homogeneous class), $\frac23$ cancer (Bob), 1 after ruling out flu; entropy $\ell$ = 1.00, 3.00, 1.89; TV to the table 0.556, 0.111, 0.444 |
| Recursive $(c, \ell)$ bound | $c/(1 + c)$ = 0.500, 0.667, 0.750 for $c$ = 1, 2, 3 |
| Skewness | 50% HIV against 1%: TV 0.49, ratio 50. 10% HIV: TV 0.09, ratio 10 |
| Uniform cells, $n = 10^4$ | @tbl-anon-bits |
| Hospital table, $n = 10^4$ | Exact quasi-identifiers: linkage 0.933. Baselines: most common diagnosis 0.185, imputation from age 0.536, largest population posterior 0.705 |
| Mondrian | @tbl-anon-k, @tbl-anon-l, @tbl-anon-t |
| Two hospitals | @tbl-anon-composition |
| Composition model | Pinned: 0.8734 ($k = 2$), 0.1615 ($k = 5$), 0.0288 ($k = 7$), 0.0015 ($k = 10$); simulation within 0.001 |

: Output of the chapter's script (seed 0). {#tbl-anon-code}

[tests/test_syntactic_anonymity.py](../../tests/test_syntactic_anonymity.py) checks that the output still matches.

## Summary {#sec-anon-summary}

| Result | Statement | Use |
| --- | --- | --- |
| @def-anon-k, @prp-anon-k-link | Every class has at least $k$ records, so linkage $J/n \le 1/k$ | Protects names, not attributes: ask what a class's values reveal |
| @prp-anon-k-violate | Uniform cells: a fraction $\Pr[\mathrm{Poisson}(n/K) \le k - 2]$ is in classes below $k$ | 99% 2-anonymous at $\log_2n - 2.20$ bits; more than $\log_2 k$ bits |
| @def-anon-mondrian | Split at medians while both halves meet the criterion | Linkage between $1/(2k)$ and $1/k$ at little cost in precision |
| @lem-anon-posterior | The posterior is the class's frequency; ruling out values renormalizes | Every attribute attack; measure it against imputation, or findings count as leaks |
| @def-anon-l, @prp-anon-recursive | Recursive $(c, \ell)$-diversity keeps the posterior below $c/(1 + c)$ after $\ell - 2$ values are ruled out | Distinct and entropy diversity do not cap the top value's share |
| @prp-anon-entropy-table | Table entropy below $\log_2\ell$ rules out entropy $\ell$-diversity | Check the table's entropy before choosing $\ell$ |
| @def-anon-t, @prp-anon-tv | Total variation $t$ moves every event's probability by at most $t$ | A 1% value may rise to $1\% + t$: rare values need a bound on the ratio |
| @def-anon-emd, @prp-anon-emd | On ordered values, EMD sums the net mass crossing each gap | Similar values are close, which catches the similarity attack |
| @def-anon-composition, @prp-anon-pinned, @prp-anon-union | Candidates are the values in both classes | No syntactic definition prevents it: assume another release exists |

: The results of this chapter, and where each is used. {#tbl-anon-summary}

## References {#sec-anon-references}

- Samarati & Sweeney, "Protecting Privacy when Disclosing Information: k-Anonymity and Its Enforcement through Generalization and Suppression" (SRI technical report, 1998), and Sweeney, "k-Anonymity: A Model for Protecting Privacy" (International Journal of Uncertainty, Fuzziness and Knowledge-Based Systems, 2002): the definition, generalization and suppression, and the linkage attack it answers.
- LeFevre, DeWitt & Ramakrishnan, "Mondrian Multidimensional k-Anonymity" (ICDE, 2006): the median-split algorithm the script uses, and why multidimensional recoding loses less than one generalization per column.
- Machanavajjhala, Kifer, Gehrke & Venkitasubramaniam, "l-Diversity: Privacy Beyond k-Anonymity" (ACM Transactions on Knowledge Discovery from Data, 2007; ICDE 2006): the homogeneity and background knowledge attacks, and the three versions of l-diversity with their Bayesian justification.
- Li, Li & Venkatasubramanian, "t-Closeness: Privacy Beyond k-Anonymity and l-Diversity" (ICDE, 2007): the skewness and similarity attacks, and the earth mover's distance with ordered and hierarchical ground distances.
- Ganta, Kasiviswanathan & Smith, "Composition Attacks and Auxiliary Information in Data Privacy" (KDD, 2008): the intersection attack on independent releases, experiments on public census data, and the argument that a definition must hold up against auxiliary information such as other releases.
- Aggarwal, "On k-Anonymity and the Curse of Dimensionality" (VLDB, 2005): why high-dimensional data cannot be k-anonymized without destroying it.
- Prasser, Kohlmayer and colleagues, the ARX data anonymization tool (arx.deidentifier.org): an open implementation of these criteria and many more, with risk models and utility measures.
- Article 29 Working Party, "Opinion 05/2014 on Anonymisation Techniques": how European regulators judge these techniques against singling out, linkability and inference, the three attacks of this chapter.

**Practice.** Exercises for this chapter, each with a worked solution, are in [the exercise book](../../book-exercises/part02-attacks/08-syntactic-anonymity.md).
