# Attempts at data privacy

This chapter examines five releases of sensitive data that their owners considered safe. Attackers defeated each protection by joining the release with information from another source. The protections lacked a formal guarantee against a defined class of attackers. The final section shows that $k$-anonymity, an early formal response, fails for the same reason.

## Pseudonyms: the taxi trips {#sec-att-taxi}

In 2014, a public-records request obtained records of every New York City taxi trip in 2013. Each record included pickup and drop-off times and locations, the fare, the cab's medallion number, and the driver's licence number. Because the identifiers revealed drivers' work locations and earnings, the regulator replaced each with a string of 32 hexadecimal digits. This pseudonymization preserves one key property: the same identifier always receives the same code.

::: {#def-att-pseudonym name="Pseudonymization"}
Let $\mathcal U$ be a set of identifiers, such as licence numbers. A **pseudonymization** replaces each identifier $u \in \mathcal U$ in a table by $h(u)$, where $h$ is a fixed injective function, so that the same person gets the same pseudonym in every row.
:::

An analyst found that one licence pseudonym earned more per day than any driver could. It was the MD5 hash of `0`, the placeholder for a missing licence; thus $h$ was unkeyed MD5. A hash can be hard to invert when its input space is large. Here the identifier formats allowed only about 22 million strings: about 3 million licence numbers and 19 million medallion numbers. Anyone could hash every candidate and look up each pseudonym.

::: {#prp-att-dictionary name="Dictionary attack"}
Let the set of identifiers $\mathcal U$ be finite, and let the pseudonymization function $h$ be injective and computable by anyone. Then $|\mathcal U|$ evaluations of $h$ recover the identifier behind every pseudonym in the table.
:::

::: {.proof}
1. Compute $h(u)$ for every $u \in \mathcal U$ and store the pairs $(h(u), u)$ in a table sorted by $h(u)$. This takes $|\mathcal U|$ evaluations of $h$.
2. For each pseudonym $v$ in the released table, look up $v$. Since $h$ is injective, exactly one stored pair has first entry $v$, and its second entry is the identifier.
:::

A licence number does not identify a name by itself. The attacker joined the decoded table with a public list of licensed drivers, which mapped each licence number to a name. This revealed every driver's trips and income. The same kind of join with a named auxiliary table underlies the attacks in this chapter.

::: {#def-att-linkage name="Linkage attack"}
Let $T$ be a released table and $T'$ an auxiliary table that names people, and let $Q$ be a set of attributes that both tables contain. A **linkage attack** joins $T$ and $T'$ on $Q$. A row of $T$ is **re-identified** when its values on $Q$ match exactly one row of $T'$ and no other row of $T$.
:::

Random pseudonyms would stop the dictionary attack but not linkage. A passenger knows the time and place of their pickup and can read the driver's licence number in the cab. The matching row reveals the driver's pseudonym, which links to every other trip by that driver. Pickup time and location also identify a passenger's trip; the drop-off may be their home.

## Rich records: the Netflix Prize {#sec-att-netflix}

In 2006, Netflix released about 100 million film ratings from about 480,000 subscribers for a contest. Each rating included a user pseudonym, film, score, and date. Such histories can reveal sensitive traits, including a person's politics or sexuality.

Narayanan and Shmatikov showed in 2008 that the pseudonyms protected little, because many subscribers also post reviews on the Internet Movie Database (IMDb) under their name or a stable handle. A linkage attack in the sense of @def-att-linkage, with $Q$ the films, scores and dates, matches the two sources. The match need not be exact. The authors scored each candidate pair of users by how many films both rated with similar scores on nearby dates, and they gave rarely rated films more weight. They reported that 8 ratings, 2 of which may be wrong, with dates known only to within 14 days, single out 99% of the records.

In a sparse table with many columns, records are often unique. A small set of ratings can identify a person.

## Releasing a model: memorization {#sec-att-memorization}

A trained model may seem safer to release than a table. In 2019, Carlini and coauthors tested this assumption on language models. A language model predicts each word from the preceding words, using parameters $\theta$ learned from a text corpus. The authors measured how surprising a string was to the model, in bits; strings the model considered likely received lower scores.

::: {#def-att-perplexity name="Log-perplexity"}
Write $\Pr_\theta$ for the probability that the model with parameters $\theta$ assigns to a sequence of words, or to the next word given the words before it. The **log-perplexity** of a sequence $x_1, \dots, x_\ell$ of $\ell$ words under this model is

$$
P_\theta(x_1, \dots, x_\ell) = -\log_2 \Pr_\theta[x_1, \dots, x_\ell] = \sum_{i=1}^{\ell} -\log_2 \Pr_\theta\left[x_i \mid x_1, \dots, x_{i-1}\right].
$$
:::

The second form follows from the chain rule of probability. An English model assigns a low score to a common sentence and a high score to an unlikely sequence. The test asks whether a rare training string, such as a password or identity number, receives a much lower score than comparable strings. A low score indicates memorization and may let anyone querying the model recover the string.

To measure this, the authors put a **canary** into the training text: a sentence of a fixed format with a random secret, such as "my identity number is 314-15-9265". They then compared its score with those of all other sentences of that format. Exposure is large when the model ranks the true canary above most of them.

::: {#def-att-exposure name="Exposure"}
Let $\mathcal R$ be the finite set of candidate secrets, all sentences of the canary's format, and let $N = |\mathcal R|$. Sort $\mathcal R$ by increasing log-perplexity under the model, and let $\operatorname{rank}(r) \in \{1, \dots, N\}$ be the position of $r$, so that rank $1$ is the most likely candidate. The **exposure** of $r$ is

$$
\operatorname{exposure}(r) = \log_2 N - \log_2 \operatorname{rank}(r).
$$
:::

Exposure ranges from $0$ for the least likely candidate to $\log_2 N$ for the most likely. An attacker can test candidates in rank order to recover a high-ranked canary. The next result bounds exposure when the model did not see the canary.

::: {#prp-att-exposure name="Exposure without memorization"}
Suppose the canary $r$ is drawn uniformly from $\mathcal R$, independently of the model, and the ranking breaks ties in a fixed way. Then for every $t \ge 0$,

$$
\Pr[\operatorname{exposure}(r) \ge t] \le 2^{-t}, \qquad \mathbb E[\operatorname{exposure}(r)] < \frac{1}{\ln 2} \approx 1.443 .
$$
:::

::: {.proof}
1. The ranking depends only on the model, and $r$ is uniform and independent of it. So $\operatorname{rank}(r)$ is uniform on $\{1, \dots, N\}$.
2. The exposure is at least $t$ exactly when $\operatorname{rank}(r) \le N 2^{-t}$. By step 1 this has probability $\lfloor N 2^{-t} \rfloor / N \le 2^{-t}$.
3. By step 1, the average exposure is $\log_2 N - \frac1N\sum_{k=1}^N \log_2 k = \frac{1}{N\ln 2}\left(N\ln N - \ln N!\right)$.
4. Since $\ln$ is increasing, $\ln k$ is at least the integral of $\ln x$ over $[k-1, k]$ for $k \ge 2$. Summing over $k$ gives

$$
\ln N! \ge \int_1^N \ln x\,\mathrm d x = N\ln N - N + 1.
$$

5. Steps 3 and 4 give $\mathbb E[\operatorname{exposure}(r)] \le \frac{N-1}{N\ln 2} < \frac{1}{\ln 2}$.
:::

So if the model never saw the canary, an exposure of $t$ bits is as surprising as an event of probability $2^{-t}$. With nine-digit secrets, $N = 10^9$, and the largest possible exposure is $\log_2 10^9 \approx 29.9$.

The experiments showed different levels of memorization. A translation model trained on about 100,000 sentence pairs assigned maximum exposure to a canary containing a nine-digit identity number after four insertions. Gmail Smart Compose, trained on billions of word sequences, used a candidate space of order $10^{12}$; after $10{,}000$ insertions, canary exposure reached about $10$ out of $39.9$. Dropout and regularization did not reduce exposure. The authors found that differential privacy, the subject of the next part, did.

## Releasing statistics: genomic studies {#sec-att-genomics}

Medical studies must publish findings, and aggregate statistics can appear safer than individual records. But if a study enrolls only people with a stigmatized disease, participation reveals a diagnosis. A test that infers membership from published statistics is then an attack.

Homer and coauthors showed in 2008 that such a test can use genetic data. It compares a person's genotype with published allele frequencies for the study group and a reference population. Given sufficient genetic markers and a suitable group size, the test can infer whether the person's DNA was included. In practice, the test has more false positives than the analysis predicts because its threshold depends on how closely the reference population matches the study group. The United States National Institutes of Health nevertheless placed aggregate genomic statistics behind an approval process.

## Removing identifiers: $k$-anonymity {#sec-att-kanon}

In the mid-1990s, the Massachusetts Group Insurance Commission gave researchers hospital records for state employees after removing names and Social Security numbers. Each record retained the patient's ZIP code, birth date, sex, and diagnosis. Latanya Sweeney bought the Cambridge voter roll, which listed each voter's name, ZIP code, birth date, and sex. A linkage attack (@def-att-linkage) on the first three attributes identified the governor's hospital records. The same attributes identify many people: Sweeney later estimated that 87% of the United States population was unique by five-digit ZIP code, sex, and birth date.

Sweeney and Samarati classified table attributes into three types. A release removes **identifiers**, such as names, and retains **sensitive attributes**, such as diagnoses. **Quasi-identifiers**, such as ZIP code, age, and sex, can identify people in combination. The release therefore generalizes them, for example by shortening ZIP codes, until each person shares their quasi-identifier values with at least $k$ rows.

::: {#def-att-kanon name="$k$-anonymity"}
Let $k \ge 1$ be an integer. A table is **$k$-anonymous** with respect to a set of quasi-identifiers if every combination of quasi-identifier values that appears in the table appears in at least $k$ rows. The rows that share a combination form an **equivalence class**.
:::

A $k$-anonymous table stops the attack on the governor, because a linkage on the quasi-identifiers matches every person to at least $k$ rows. But the definition says nothing about the sensitive values in a class, or about other releases. Two examples show what goes wrong, with tables from two hospitals in the same town.

| ZIP code | Age | Diagnosis |
| --- | --- | --- |
| 021\*\* | 20–29 | influenza |
| 021\*\* | 20–29 | asthma |
| 021\*\* | 20–29 | diabetes |
| 021\*\* | 30–39 | hepatitis |
| 021\*\* | 30–39 | hepatitis |
| 021\*\* | 30–39 | hepatitis |
| 024\*\* | 40–59 | influenza |
| 024\*\* | 40–59 | migraine |
| 024\*\* | 40–59 | asthma |

: Hospital A's release, $3$-anonymous with respect to ZIP code and age. {#tbl-att-hospital-a}

| ZIP code | Age | Diagnosis |
| --- | --- | --- |
| 02\*\*\* | 18–24 | influenza |
| 02\*\*\* | 18–24 | asthma |
| 02\*\*\* | 18–24 | fracture |
| 02\*\*\* | 25–34 | diabetes |
| 02\*\*\* | 25–34 | migraine |
| 02\*\*\* | 25–34 | fracture |

: Hospital B's release, $3$-anonymous with respect to ZIP code and age. {#tbl-att-hospital-b}

The first example shows that a class whose rows all share one sensitive value hides nothing.

::: {#exm-att-homogeneity name="Homogeneity"}
A neighbour aged 35, living in ZIP code 02139, was a patient of hospital A. In @tbl-att-hospital-a, the neighbour must be in the class with ZIP code 021\*\* and age 30–39. All three of its rows say hepatitis, so the attacker learns the diagnosis without knowing which row is the neighbour's.
:::

The second example has varied diagnoses in each class, but combines the two releases.

::: {#exm-att-intersection name="Intersection"}
A colleague aged 27, living in ZIP code 02139, was treated at both hospitals for the same condition. In @tbl-att-hospital-a the colleague's class allows influenza, asthma or diabetes. In @tbl-att-hospital-b the class with age 25–34 allows diabetes, migraine or fracture. Only diabetes is in both lists, so the two releases together reveal the diagnosis, although each is $3$-anonymous.
:::

The second example is a failure of composition: each hospital checked its own table and could not know what the other released. Ganta, Kasiviswanathan and Smith showed in 2008 that such attacks succeed in practice, also against refinements of $k$-anonymity.

These cases motivate three requirements for a privacy definition. It must allow arbitrary side information, apply to any output, and degrade gracefully under composition. Differential privacy, defined in [the chapter on randomized response](../part02-dp-basics/01-randomized-response.md), meets all three.

## Summary {#sec-att-summary}

| Result | Statement |
| --- | --- |
| @def-att-pseudonym | replace each identifier $u$ by $h(u)$ for a fixed injective $h$ |
| @prp-att-dictionary | a public $h$ on a set of $N$ identifiers is inverted by $N$ evaluations |
| @def-att-linkage | join a release with named auxiliary data on shared attributes |
| @def-att-exposure | $\log_2 N - \log_2 \operatorname{rank}(r)$ for a canary among $N$ candidates |
| @prp-att-exposure | without memorization, exposure $\ge t$ has probability at most $2^{-t}$ and mean below $1/\ln 2$ |
| @def-att-kanon | every quasi-identifier combination appears in at least $k$ rows |
| @exm-att-homogeneity | a class with one sensitive value reveals it |
| @exm-att-intersection | two $k$-anonymous releases together can reveal a value |

: The chapter's results. {#tbl-att-summary}

## References {#sec-att-notes}

- Vijay Pandurangan, "On Taxis and Rainbows", 2014. Inverts the hashed medallion and licence numbers.
- Arvind Narayanan and Vitaly Shmatikov, "Robust De-anonymization of Large Sparse Datasets", 2008. Links the Netflix Prize data with public reviews.
- Nicholas Carlini, Chang Liu, Úlfar Erlingsson, Jernej Kos and Dawn Song, "The Secret Sharer: Evaluating and Testing Unintended Memorization in Neural Networks", 2019. Defines canaries and exposure and measures them on real models.
- Nils Homer and coauthors, "Resolving Individuals Contributing Trace Amounts of DNA to Highly Complex Mixtures Using High-Density SNP Genotyping Microarrays", 2008. Gives the membership test on aggregate genetic statistics.
- Rosemary Braun, William Rowe, Carl Schaefer, Jinghui Zhang and Kenneth Buetow, "Needles in the Haystack: Identifying Individuals Present in Pooled Genomic Data", 2009. Shows that the test's false-positive rate depends on the reference population.
- Latanya Sweeney, "Simple Demographics Often Identify People Uniquely", 2000. Estimates how often ZIP code, sex and date of birth identify a person.
- Pierangela Samarati and Latanya Sweeney, "Generalizing Data to Provide Anonymity when Disclosing Information", 1998. Defines $k$-anonymity.
- Srivatsava Ranjit Ganta, Shiva Prasad Kasiviswanathan and Adam Smith, "Composition Attacks and Auxiliary Information in Data Privacy", 2008. Gives intersection attacks on independently anonymized releases.
