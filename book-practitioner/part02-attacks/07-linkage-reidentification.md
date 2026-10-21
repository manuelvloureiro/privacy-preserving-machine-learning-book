# Linkage and re-identification

## Why this matters

A hospital wants to share its discharge records with researchers. It deletes the names but keeps the postcode, the date of birth, the sex and the diagnosis, because the research needs them. But anyone who holds a voter roll, with names next to those same three columns, can join the two tables and put a name back on most rows. This is a **linkage attack**, and putting a name back on a "de-identified" record is **re-identification**.

To judge whether a release is identifying, measure identifying information in bits, because bits add up. Write $n$ for the number of people in a population. Singling out one of them takes $\log_2 n$ bits, which is 27.9 bits for the United States of 1990 and 32.9 bits for the whole world. An attribute with $v$ equally likely values carries $\log_2 v$ bits, and the bits of independent attributes add. Once the bits an adversary knows exceed $\log_2 n$ by a few, almost everyone is unique.

## Quasi-identifiers and the linkage attack {#sec-link-quasi}

**Direct identifiers**, like a name or an email address, name a person on their own, while **sensitive attributes**, like a diagnosis, are what the release is about. **Quasi-identifiers** sit in between. None of them names anyone alone, but together they often single out one person, and they appear in other datasets next to names. The most common "de-identification" removes the direct identifiers and keeps the quasi-identifiers, which is exactly the gap the linkage attack uses.

To score a release you hold, group its $n$ records into **equivalence classes**, the sets of records with identical quasi-identifiers, and write $J$ for the number of classes. An adversary who knows the target is in the release, and knows the target's quasi-identifiers, finds a target alone in its class for sure and a target in a class of $k$ records one time in $k$. Averaged over targets, its success probability is
$$\Pr[\text{win}] = \frac{J}{n},$$ {#eq-link-success}
and no adversary that sees only the quasi-identifiers can do better.

For example, the hospital release in the Code section uses ZIP code, birth date and sex, and its records fall into 5 classes for a success of 0.625. Three of its records are unique, so anyone on a voter roll with one of those combinations has their diagnosis exposed with certainty. Latanya Sweeney estimated from the 1990 census that 87% of Americans were likely unique on those three attributes, because together they carry at most 31.1 bits. Coarser geography helps, and with the county in place of the ZIP code the figure fell to 18%.

If every class holds at least $k$ records, the success is at most $1/k$, but a class whose records share one diagnosis still leaks it ([Syntactic anonymity and how it breaks](08-syntactic-anonymity.md)).

## Pseudonyms and hashed identifiers {#sec-link-pseudonyms}

Many releases replace each identifier by a **pseudonym**, a code that stays the same across a person's records, and such codes fail in two ways. The first is that an unkeyed hash, such as MD5, can be reversed whenever the possible identifiers can be listed. The adversary hashes every candidate once and looks each pseudonym up, which takes seconds to hours for phone, licence or identity numbers. The second failure survives any key. Because the pseudonym ties a person's records together, linking any one record to them reveals all the others.

New York City's 2013 taxi release showed both. Driver licences had been replaced by unkeyed MD5 hashes, and there were only about 2 million possible licences. So Vijay Pandurangan hashed every one in minutes, and public lists of licence holders put a name on every driver's trips and income. Passengers were exposed by the trips themselves, because a pickup minute and place single out one trip.

So treat a pseudonymized release as identifiable, as the GDPR does, since it counts pseudonymized data as personal data. If you must keep a join key, derive it with a keyed hash, a hash-based message authentication code (HMAC) such as Python's `hmac.new(key, msg, hashlib.sha256)`, and keep the key away from the data. That stops the dictionary attack, but the key still groups each person's records into a history that is itself identifying. Use a fresh key for each release, because one pseudonym reused across releases joins them.

## Predicting uniqueness before a release {#sec-link-unique}

For a release you are still designing, you need a model rather than a count. Call a person's combination of attribute values their **cell**, write $p_c$ for the probability of cell $c$, and assume that people fall in cells independently. The expected fraction of the $n$ people who share their cell with nobody is then
$$\frac{\mathbb E U}{n} = \sum_c p_c(1 - p_c)^{n-1}.$$ {#eq-link-unique}

When all cells are equally likely and there are $2^b$ of them, so that the attributes carry $b$ bits, the formula becomes
$$\frac{\mathbb E U}{n} \approx \exp\big(-2^{\log_2 n - b}\big).$$ {#eq-link-uniform}
This fraction depends only on the bits beyond $\log_2 n$, and it is steep. Half the people are unique at 0.53 bits beyond it, and 99% at 6.64 bits.

Use @eq-link-uniform for a first estimate. For the Sweeney attributes it predicts that 0.899 of Americans are unique, close to her 87%. Real attributes are skewed, and skew usually lowers uniqueness, so the uniform estimate is pessimistic. The full edition also gives a floor on uniqueness from the collision probability, the chance that two random people share a cell. Correlated attributes carry fewer bits than their sum. They need a model of the joint distribution, such as the Gaussian copula of Rocher, Hendrickx and de Montjoye (2019).

## Sparse data: the Netflix attack {#sec-link-sparse}

Ratings, purchases and locations have no quasi-identifier columns, but they are even more identifying. In 2006 Netflix released about 100 million movie ratings, with subscribers' names replaced by random numbers. Narayanan and Shmatikov (2008) matched its records to public reviews. Their attack scores how well each record $r$ agrees with what the adversary knows, a few items $i$ with approximate values $\mathrm{aux}_i$:
$$\mathrm{Score}(\mathrm{aux}, r) = \sum_{i \in \mathrm{aux}} w_i\,\mathrm{Sim}(\mathrm{aux}_i, r_i).$$ {#eq-link-score}
Here the agreement $\mathrm{Sim}$ is 1 when the record's rating $r_i$ is within one star of the known value, and 0 otherwise. The weight $w_i$ favours rare items, because agreeing on an obscure film says much more than agreeing on a hit. Write $s_i$ for the fraction of records that rated item $i$. The weight $\ln(1/s_i)$ works well, and the item carries $\log_2(1/s_i)$ bits.

On Netflix, eight ratings identified 99% of records, even with two of them wrong and dates known only to within two weeks. Bits explain it again, since a film that 1% of subscribers rated carries 6.64 bits, and three such films single out one subscriber. Errors in the auxiliary data cost little because the score adds up evidence rather than demanding a perfect match.

The top score is the best guess only if the target is in the release. When it may not be, the **eccentricity test** declares a match only if the best score stands out. Write $S_1$ and $S_2$ for the two largest scores, and $\sigma$ for the standard deviation of all scores. The test requires the gap to clear a threshold $\varphi$,
$$\frac{S_1 - S_2}{\sigma} \ge \varphi.$$ {#eq-link-eccentricity}
Calibrate the threshold on people known not to be in the data, because the published threshold of 1.5 falsely matched 26.3% of outsiders on the full edition's synthetic ratings. So always report the rate at a stated false positive rate ([Hypothesis testing](../part01-prerequisites/04-hypothesis-testing.md)).

## Does releasing a sample protect? {#sec-link-sampling}

Statistical agencies often release only a sample, arguing that a sample-unique record may not be unique in the population. Write $\pi$ for the **sampling fraction** and $\lambda$ for the expected number of other people in the target's cell. Start from a known person who has exactly one match in the sample, and write $\kappa$ for the probability that the match is right. Or start from a record alone in its cell in the sample, and write $\xi$ for the probability that it is alone in the population too. For a rare cell,
$$\kappa \approx \frac{1}{1 + \lambda(1 - \pi)}, \qquad \xi \approx e^{-\lambda(1 - \pi)}.$$ {#eq-link-kappa}
Since $1 - \pi$ is at most 1, sampling can never push them below $1/(1 + \lambda)$ and $e^{-\lambda}$, so it helps only when cells are crowded. For a rare cell, a match from a person is right with probability 0.910 even in a 1% sample.

In the full edition's synthetic population, 0.8301 of the sample-unique records in a 1% sample were still unique in the population, and the adversary can tell which ones from the sample itself. So estimate uniqueness for the population, never just for the sample. Rocher, Hendrickx and de Montjoye predicted that 99.98% of Americans would be correctly re-identified in any dataset with 15 demographic attributes, and that sampling changes this very little. So a small sample protects only people who were not unique anyway.

## A pre-release risk check {#sec-link-check}

1. **List the quasi-identifiers.** Include every column an adversary could find elsewhere next to a name, such as demographics, dates, locations, employers and rare events like a hospital stay in the news.
2. **Measure the release.** Group by the quasi-identifiers and report $J/n$, the fraction of unique records and the fraction in classes smaller than your $k$. In pandas, `df.groupby(qi_cols).size()` gives the class sizes.
3. **Estimate the bits for planned releases.** Add $\log_2$ of each attribute's number of values, and if the total is several bits above $\log_2 n$, expect most people to be unique.
4. **Reduce the risk.** Coarsen geography and dates first, since they carry the most bits, then suppress the records that remain unique.
5. **Know when to stop.** If the analysis needs fine-grained or sparse data, release aggregates or a differentially private output instead ([The definition of differential privacy](../part03-dp-foundations/01-definition.md)).

Several tools automate these steps. ARX, an open-source Java application, measures risk and searches for generalizations, and the R package sdcMicro does the same from an object built with `createSdcObj`. To test a release by actually linking it, use the Python `recordlinkage` package, Splink, or Anonymeter's `SinglingOutEvaluator` and `LinkabilityEvaluator`. No tool knows what auxiliary data exist, so the choice of quasi-identifiers stays your judgement.

## Code

The snippet computes the risk of a release you hold, and the uniform estimate of @eq-link-uniform for Sweeney's attributes.

```python
from collections import Counter
from math import exp, log2


def linkage_report(rows: list[tuple[str, ...]]) -> tuple[float, float, int]:
    """Linkage success J/n, fraction of unique records, and smallest class size of a release."""
    sizes = Counter(rows).values()  # one count per distinct quasi-identifier combination
    n = len(rows)
    return len(sizes) / n, sum(1 for k in sizes if k == 1) / n, min(sizes)



release = [
    ("02138", "1961-03-14", "F"), ("02138", "1972-11-02", "M"), ("02138", "1972-11-02", "M"),
    ("02139", "1958-06-21", "F"), ("02139", "1990-01-01", "F"), ("02139", "1990-01-01", "F"),
    ("02139", "1990-01-01", "F"), ("02141", "1983-09-30", "M"),
]
success, unique, k = linkage_report(release)
print(f"{success:.3f} {unique:.3f} {k}")  # 0.625 0.375 1
cells = 40_000 * 365 * 80 * 2  # ZIP code x birth date over 80 years x sex
print(f"{log2(cells):.1f} {exp(-248.7e6 / cells):.3f}")  # 31.1 0.899
```

The full edition's script, [linkage.py](../../code/part02-attacks/linkage.py), computes every number in this chapter.

## Cheat sheet

| Question | Result | Use it for |
| --- | --- | --- |
| Bits to name one person | $\log_2 n$ | 33 for the world, 28 for the US, 13 for 10,000 people |
| Risk of a release you hold | Success $J/n$; at most $1/k$ if every class has $k$ records | Target known to be in the release |
| Fraction unique, uniform cells | $\exp(-2^{\log_2 n - b})$ | 50% at $\log_2 n + 0.53$ bits, 99% at $\log_2 n + 6.64$ |
| Fraction unique, known frequencies | $\sum_c p_c(1 - p_c)^{n-1}$ | Planned releases with independent attributes |
| Sparse data | Score with rarity weights, then the eccentricity test | Calibrate $\varphi$ on outsiders at a small false positive rate |
| Releasing a sample | $\xi \approx e^{-\lambda(1 - \pi)}$ | Sampling helps only in crowded cells |

: The results of this chapter. {#tbl-link-cheat}

**Rules of thumb**

- Add up the bits of every attribute an adversary might know. Above $\log_2 n$, assume most people are unique, and about 7 bits more make 99% unique.
- Estimate uniqueness for the population, not just the sample, and assume the adversary holds auxiliary data you have not seen.
- Hashing a licence, phone or identity number is not anonymization, because anyone can hash every possible value. Key the hash, and never reuse a pseudonym across releases.

## Further reading

- Sweeney, "Simple Demographics Often Identify People Uniquely" (2000): the 87% estimate.
- Narayanan & Shmatikov, "Robust De-anonymization of Large Sparse Datasets" (2008): the Netflix attack.

**Derivations.** Every result here is derived step by step in [the full edition's chapter](../../book-full/part02-attacks/07-linkage-reidentification.md).
