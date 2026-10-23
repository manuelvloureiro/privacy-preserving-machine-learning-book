# Attempts at Data Privacy

Before defining privacy, it helps to see how hard it is to get right. This chapter goes through five releases of sensitive data whose owners believed them safe. They are a pseudonymized table of taxi trips, a table of film ratings, a language model, the summary statistics of genetic studies and hospital records with names removed. Each was undone by an attacker who combined the release with information from somewhere else.

The cases share a pattern. A protection was chosen by intuition, and nobody could say exactly what it guaranteed or against whom. The last section looks at $k$-anonymity, the first formal answer to these attacks, which asks every record to look like at least $k - 1$ others. It fails in the same way. The next chapter makes the attacks quantitative.

## Pseudonyms: the taxi trips {#sec-att-taxi}

In 2014, a public-records request obtained from New York City's taxi regulator the record of every taxi trip and fare in the city in 2013, about 19 gigabytes of data. A trip record holds the pickup and drop-off times, their coordinates, the distance, the duration and the number of passengers. It also holds two identifiers, the number of the cab's medallion and the number of the driver's licence. These two would reveal where each driver worked and how much they earned, so the regulator did not publish them as they were. It replaced each by a string of 32 hexadecimal digits.

Replacing each identifier by a code in this way is called pseudonymization. The definition below names the one property such a code must have, that the same identifier always gets the same code, because the attack depends on exactly that.

::: {#def-att-pseudonym name="Pseudonymization"}
Let $\mathcal U$ be a set of identifiers, such as licence numbers. A **pseudonymization** replaces each identifier $u \in \mathcal U$ in a table by $h(u)$, where $h$ is a fixed injective function, so that the same person gets the same pseudonym in every row.
:::

The pseudonyms did not hold for long. An analyst noticed that one licence pseudonym earned far more per day than any driver could. That pseudonym turned out to be the MD5 hash of the string `0`, a placeholder for a missing licence, so many drivers' trips had been pooled under it. The discovery revealed the function $h$, which was MD5 with no secret key. A hash function is hard to invert on inputs drawn from a huge set, but the identifier formats allowed only about 22 million strings, about 3 million licence numbers and 19 million medallion numbers. With so few possible inputs, anyone can apply the function to all of them and look each pseudonym up, as the next result says.

::: {#prp-att-dictionary name="Dictionary attack"}
Let the set of identifiers $\mathcal U$ be finite, and let the pseudonymization function $h$ be injective and computable by anyone. Then $|\mathcal U|$ evaluations of $h$ recover the identifier behind every pseudonym in the table.
:::

::: {.proof}
1. Compute $h(u)$ for every $u \in \mathcal U$ and store the pairs $(h(u), u)$ in a table sorted by $h(u)$. This takes $|\mathcal U|$ evaluations of $h$.
2. For each pseudonym $v$ in the released table, look up $v$. Since $h$ is injective, exactly one stored pair has first entry $v$, and its second entry is the identifier.
:::

A licence number is not yet a name. The second step joined the decoded table with a public list of licensed drivers, which gives each licence number its holder's name. Every driver's trips, home area and income were then public. Joining a release with a second table that names people is the step that turns every attack in this chapter into a disclosure, so it gets its own definition.

::: {#def-att-linkage name="Linkage attack"}
Let $T$ be a released table and $T'$ an auxiliary table that names people, and let $Q$ be a set of attributes that both tables contain. A **linkage attack** joins $T$ and $T'$ on $Q$. A row of $T$ is **re-identified** when its values on $Q$ match exactly one row of $T'$ and no other row of $T$.
:::

Replacing MD5 by random pseudonyms would stop the dictionary attack but not linkage. A passenger who takes one ride knows the time and place of the pickup and can read the driver's licence number in the cab. That single row of the table carries the driver's pseudonym, which then points to every other trip by the same driver. Passengers are exposed too. Someone who sees a colleague hail a cab outside the office at a known time can find that trip, and its drop-off point is likely the colleague's home. The fare table even shows the tip. The attacker needs no identifier at all, because the time and place of a pickup identify the trip.

## Rich records: the Netflix Prize {#sec-att-netflix}

From 2006 to 2009, Netflix ran a contest with a prize of one million dollars for the team that most improved its film recommendations. The winning entry blended many models, among them matrix factorization. To train their models, contestants received about 100 million ratings from about 480,000 subscribers. Each rating was a tuple of a user pseudonym, a film, a score and a date. Viewing histories are sensitive, since they can hint at a person's politics or sexuality, and in the United States the Video Privacy Protection Act of 1988 restricts their disclosure.

Narayanan and Shmatikov showed in 2008 that the pseudonyms protected little. Many subscribers also post reviews on the Internet Movie Database (IMDb) under their name or a stable handle. A linkage attack in the sense of @def-att-linkage, with $Q$ the films, scores and dates, matches the two sources. The match need not be exact. They scored each candidate pair of users by how many films both rated with similar scores on nearby dates, and they weighted rarely rated films more heavily. They reported that 8 ratings, 2 of which may be wrong, with dates known only to within 14 days, single out 99% of the records in the dataset.

The lesson is that in a sparse table with many columns, almost every record is unique, so no attribute is harmless. Even a handful of a person's ratings identifies them. The release also had concrete consequences, because a class action over it was settled out of court and Netflix cancelled a planned second contest.

## Releasing a model: memorization {#sec-att-memorization}

The first two releases were whole tables. Releasing only a function of the data seems safer, such as a model trained on it. Carlini and coauthors tested this in 2019 on language models, which assign a probability to every sequence of words. Such a model predicts each word from the ones before it, using a vector of parameters $\theta$ learned from a corpus of text. To test whether a model has memorized a string, the authors measured how surprised the model is by it, counted in bits. A string the model finds likely gets a small score.

::: {#def-att-perplexity name="Log-perplexity"}
Write $\Pr_\theta$ for the probability that the model with parameters $\theta$ assigns to a sequence of words, or to the next word given the words before it. The **log-perplexity** of a sequence $x_1, \dots, x_\ell$ of $\ell$ words under this model is

$$
P_\theta(x_1, \dots, x_\ell) = -\log_2 \Pr_\theta[x_1, \dots, x_\ell] = \sum_{i=1}^{\ell} -\log_2 \Pr_\theta\left[x_i \mid x_1, \dots, x_{i-1}\right].
$$
:::

The second form is the chain rule of probability. Low log-perplexity means high probability. A model of English gives "the cat sat on the mat" a low score and a string of unrelated words a high one. The question is whether a rare string that appears in the training text, such as a password or an identity number, gets a much lower score than it otherwise would. If it does, the model has memorized it, and anyone who can query the model may find it.

To measure this, the authors inserted a **canary** into the training text: a sentence of a fixed format with a random secret, such as "my identity number is 314-15-9265". They then compared its score with those of all the other sentences of the same format. Exposure turns that comparison into a number, which is large when the model ranks the true canary above most of the alternatives.

::: {#def-att-exposure name="Exposure"}
Let $\mathcal R$ be the finite set of candidate secrets, all sentences of the canary's format, and let $N = |\mathcal R|$. Sort $\mathcal R$ by increasing log-perplexity under the model, and let $\operatorname{rank}(r) \in \{1, \dots, N\}$ be the position of $r$, so that rank $1$ is the most likely candidate. The **exposure** of $r$ is

$$
\operatorname{exposure}(r) = \log_2 N - \log_2 \operatorname{rank}(r).
$$
:::

Exposure ranges from $0$, for the least likely candidate, to $\log_2 N$, for the most likely one. A canary at the top of the list can be read off the model by trying the candidates in order. The next result says how much exposure to expect when the model has not seen the canary, which calibrates the measurement.

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

So an exposure of $t$ bits is as surprising as an event of probability $2^{-t}$ under the hypothesis that the model never saw the canary. With nine-digit secrets, $N = 10^9$, and the largest possible exposure is $\log_2 10^9 \approx 29.9$.

The experiments gave both answers. A translation model trained on about 100,000 pairs of sentences was given a canary with a nine-digit identity number, and after four insertions the canary reached the largest possible exposure. Gmail's Smart Compose, a model trained on billions of word sequences, was given canaries whose secret part was a few words from a vocabulary of tens of thousands, so $N$ was of order $10^{12}$. Even after $10{,}000$ insertions their exposure reached only about $10$, far from the roughly $\log_2 10^{12} \approx 39.9$ that extraction would need. Trying all $N$ candidates is not needed to extract a secret. A shortest-path search over the digits, in the manner of Dijkstra's algorithm with $-\log_2$ of each conditional probability as the edge length, finds the most likely candidates first. Standard defences against overfitting, such as dropout and regularization, did not reduce exposure. Training with differential privacy, the subject of the next part, was the only method the authors found that did.

## Releasing statistics: genomic studies {#sec-att-genomics}

Medical and genetic studies need sensitive data and are expected to publish their findings. Aggregate statistics seemed a safe middle ground. Suppose a study enrols only people with a stigmatized disease. Then revealing that someone took part reveals their diagnosis, and a test that decides membership from the published statistics is an attack.

Homer and coauthors showed in 2008 that such a test exists for genetic data. They took a person's genotype, the published allele frequencies of a study group and those of a reference population. From these alone, they could tell whether the person's DNA was in the group, under conditions on the number of genetic markers and the size of the group. Others found that the test makes many more false positives in practice than this analysis suggests, because its threshold depends on how well the reference population matches the study group. Even so, the National Institutes of Health in the United States withdrew open access to aggregate genomic statistics, including allele frequencies, odds ratios and $p$-values, and placed them behind an approval process. The attack cost open science. A method that releases such statistics with a guarantee would let them be public again.

## Removing identifiers: $k$-anonymity {#sec-att-kanon}

In the mid-1990s, the Group Insurance Commission of Massachusetts let researchers request the hospital records of every state employee. Each record held the patient's name, social security number, ZIP code, date of birth, sex and diagnosis. The commission removed the names and social security numbers, and the governor, William Weld, assured the public that patients' privacy was protected.

Latanya Sweeney, then a graduate student, bought the voter roll of Cambridge, Massachusetts, for $20. It lists every registered voter's name, address, ZIP code, date of birth and sex. A linkage attack on ZIP code, date of birth and sex found the governor's hospital records, and she mailed them to him. The attack works at scale. Sweeney later estimated that 87% of the population of the United States is the only person with their combination of five-digit ZIP code, sex and date of birth.

Her answer, with Samarati, was a property of the released table. Its attributes fall into three kinds. **Identifiers**, such as names, are removed. **Quasi-identifiers**, such as ZIP code, age and sex, are harmless alone but identify people in combination. **Sensitive attributes**, such as the diagnosis, are what the release is for, so they stay. The quasi-identifiers are coarsened, for example by replacing a ZIP code by its first three digits or an age by a range, until no combination is rare. The goal is that every person hides in a crowd of at least $k$ rows that look the same on the quasi-identifiers, and the definition states that goal.

::: {#def-att-kanon name="$k$-anonymity"}
Let $k \ge 1$ be an integer. A table is **$k$-anonymous** with respect to a set of quasi-identifiers if every combination of quasi-identifier values that appears in the table appears in at least $k$ rows. The rows that share a combination form an **equivalence class**.
:::

A $k$-anonymous table stops the attack on the governor: a linkage on the quasi-identifiers matches every person to at least $k$ rows, not one. But the definition constrains only the quasi-identifiers. It says nothing about the sensitive values within a class, and nothing about other releases. Two examples show what goes wrong, using the two tables below, released by two hospitals that serve the same town.

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

The first example uses only hospital A's table. It shows that a class whose rows all share one sensitive value hides nothing.

::: {#exm-att-homogeneity name="Homogeneity"}
A neighbour aged 35, living in ZIP code 02139, was a patient of hospital A. In @tbl-att-hospital-a, the neighbour must be in the class with ZIP code 021\*\* and age 30–39. All three of its rows say hepatitis, so the attacker learns the diagnosis without knowing which row is the neighbour's.
:::

The second example needs classes with varied diagnoses, so homogeneity is not the problem. Instead, it combines the two hospitals' releases.

::: {#exm-att-intersection name="Intersection"}
A colleague aged 27, living in ZIP code 02139, was treated at both hospitals for the same condition. In @tbl-att-hospital-a the colleague's class allows influenza, asthma or diabetes. In @tbl-att-hospital-b the class with age 25–34 allows diabetes, migraine or fracture. Only diabetes is in both lists, so the two releases together reveal the diagnosis, although each is $3$-anonymous.
:::

The second example is a failure of composition. Each hospital checked its own table, and neither could know what the other released. Ganta, Kasiviswanathan and Smith showed in 2008 that such attacks succeed in practice, and that they defeat refinements of $k$-anonymity as well.

The five cases point to three requirements for a definition. It must hold whatever side information an attacker has, since every attack above used some. It must apply to any output, a table, a model or a statistic, and not only to tables. And it must degrade gracefully when several releases are combined. Differential privacy, defined in [the chapter on randomized response](../part02-dp-basics/01-randomized-response.md), meets all three.

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

## Notes {#sec-att-notes}

- Vijay Pandurangan, "On Taxis and Rainbows", 2014. The inversion of the hashed medallion and licence numbers.
- Arvind Narayanan and Vitaly Shmatikov, "Robust De-anonymization of Large Sparse Datasets", 2008. The linkage of the Netflix Prize data with public reviews, and the statistics of how few ratings identify a record.
- Nicholas Carlini, Chang Liu, Úlfar Erlingsson, Jernej Kos and Dawn Song, "The Secret Sharer: Evaluating and Testing Unintended Memorization in Neural Networks", 2019. Canaries, exposure, the extraction search, the experiments on a translation model and on Smart Compose, and the experiments on defences.
- Nils Homer and coauthors, "Resolving Individuals Contributing Trace Amounts of DNA to Highly Complex Mixtures Using High-Density SNP Genotyping Microarrays", 2008. The membership test on aggregate genetic statistics.
- Rosemary Braun, William Rowe, Carl Schaefer, Jinghui Zhang and Kenneth Buetow, "Needles in the Haystack: Identifying Individuals Present in Pooled Genomic Data", 2009. A study showing that the test's false-positive rate is sensitive to its assumptions about the reference population.
- Latanya Sweeney, "Simple Demographics Often Identify People Uniquely", 2000. The estimate that ZIP code, sex and date of birth identify most of the population of the United States.
- Pierangela Samarati and Latanya Sweeney, "Generalizing Data to Provide Anonymity when Disclosing Information", 1998. The definition of $k$-anonymity.
- Srivatsava Ranjit Ganta, Shiva Prasad Kasiviswanathan and Adam Smith, "Composition Attacks and Auxiliary Information in Data Privacy", 2008. Intersection attacks on independently anonymized releases.
