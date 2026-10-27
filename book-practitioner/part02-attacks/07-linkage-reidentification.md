# Linkage and re-identification

## Why this matters

A hospital deletes the names from its discharge records and shares them with researchers. It keeps the postcode, the date of birth, the sex and the diagnosis. A voter roll lists names next to the same three columns. Anyone who has one can join the two tables and put a name back on most rows. This is a **linkage attack**, and putting a name back on a "de-identified" record is **re-identification**.

To judge whether a table can be linked, count its identifying information in bits. Each bit halves the candidates. So it takes $\log_2 n$ bits to single out one person in a population of $n$ people. For the United States of 1990, that is 27.9 bits. Almost everyone is unique once the attributes an adversary knows carry a few bits more than that, and a unique record can be linked.

## Quasi-identifiers and the linkage attack {#sec-link-quasi}

**Direct identifiers**, like a name or an email address, name a person on their own. **Sensitive attributes**, such as a diagnosis, are what the release is about. **Quasi-identifiers** sit in between. None of them names anyone alone, but together they often single out one person, and they appear next to names in other datasets.

To score a release you hold, group its $n$ records into **equivalence classes**. A class is a set of records with identical quasi-identifiers. Write $J$ for the number of classes. Suppose an adversary knows the target's quasi-identifiers and knows that the target is in the release. If the target's class holds $k$ records, the adversary names the right one with probability $1/k$. The class holds $k$ possible targets, so every class adds one win whatever its size. Over all $n$ targets, the chance of naming the right record is therefore
$$\Pr[\text{win}] = \frac{J}{n},$$ {#eq-link-success}
and no adversary that sees only the quasi-identifiers can do better. This is the **prosecutor model**, the worst case, because it assumes the adversary knows the target is in the data.

The script [practitioner_table_risk.py](../../code/part02-attacks/practitioner_table_risk.py) shows how fast uniqueness grows on a real table. The table is Adult, the 48,842 people of the 1994 US Current Population Survey in the UCI repository. Adult has no ZIP code or birth date, so the script starts from age, sex and race and adds one quasi-identifier at a time. For each set it prints the share of unique records, the share in classes smaller than 5, the smallest class size and $J/n$.

```{.python excerpt="qi-uniqueness"}
n = df.height
for qis in QI_SETS:
    # One row per equivalence class (records that agree on every quasi-identifier), with its size.
    sizes = df.group_by(qis).len()
    unique = sizes.filter(pl.col("len") == 1).height / n  # share of records alone in their class
    small = sizes.filter(pl.col("len") < 5)["len"].sum() / n  # share in classes smaller than 5
    min_k = sizes["len"].min()  # the table is k-anonymous for this k
    print(
        f"{' + '.join(qis)}: unique={unique:.2%}  in classes <5={small:.2%}  "
        f"min k={min_k}  J/n={sizes.height / n:.3f}"
    )
```

Run it with `uv run python code/part02-attacks/practitioner_table_risk.py`, which prints:

```text
age + sex + race: unique=0.12%  in classes <5=0.75%  min k=1  J/n=0.012
age + sex + race + native-country: unique=3.07%  in classes <5=7.83%  min k=1  J/n=0.060
age + sex + race + native-country + marital-status: unique=5.88%  in classes <5=11.99%  min k=1  J/n=0.100
age + sex + race + native-country + marital-status + education: unique=14.64%  in classes <5=26.83%  min k=1  J/n=0.227
age + sex + race + native-country + marital-status + education + hours-per-week: unique=36.55%  in classes <5=57.05%  min k=1  J/n=0.480
```

Three columns that look harmless leave 0.12% of the records unique, and six leave 14.64%. The hours worked per week, which nobody would list as an identifier, then make 36.55% unique. So add every column an adversary could know to the test, one at a time. The column that breaks the table may be one that nobody suspected. A minimum class size of 5 is a common rule, yet every set here has classes of a single record.

## Pseudonyms and hashed identifiers {#sec-link-pseudonyms}

Many releases replace each identifier by a **pseudonym**, a code that stays the same across a person's records. Such codes fail in two ways. First, an adversary can reverse a hash without a secret key, such as plain MD5, whenever the possible identifiers can be listed. It hashes every candidate and looks each pseudonym up. New York City hashed its taxi drivers' licences this way in its 2013 trip release. There were only about 2 million possible licences, so in 2014 Vijay Pandurangan hashed them all in minutes and named each driver.

The second failure survives any key. The pseudonym ties a person's records together, so a link to any one of them reveals all the others. In the taxi data, a passenger who knows the pickup minute and place can find one trip and its driver. The driver's pseudonym, even a random one, then reveals every trip that driver made. For this reason the General Data Protection Regulation (GDPR) still counts pseudonymized data as personal data.

If you must keep a join key, make it with a keyed hash, a hash-based message authentication code (HMAC). Keep the key away from the data.

```python
import hashlib
import hmac
import secrets

licence = "482913"  # a six-digit licence number, the identifier being pseudonymized
pseudonym = hashlib.md5(licence.encode()).hexdigest()  # what an unkeyed release prints

# Dictionary attack: hash all 10**6 possible licences once, then look the pseudonym up.
lookup = {hashlib.md5(f"{i:06d}".encode()).hexdigest(): f"{i:06d}" for i in range(10**6)}
print(lookup[pseudonym])  # 482913

# A keyed hash still gives one code per person, but without the key nobody can build the table.
key = secrets.token_bytes(32)  # store it apart from the data; draw a new one for each release
keyed = hmac.new(key, licence.encode(), hashlib.sha256).hexdigest()
print(keyed in lookup)  # False
```

The key stops the dictionary attack but not the second failure, because each person's records still form an identifying history. The same key in two releases would join that history across them, so draw a new key for each release.

## How many people are unique? {#sec-link-unique}

When you design a release, there is no table to count yet, so estimate uniqueness by adding bits. An attribute with $v$ equally likely values carries $\log_2 v$ bits, and the bits of independent attributes add. Attributes that carry $b$ bits in total split people into $2^b$ cells. Suppose each person lands in a random cell, every cell equally likely. Then the expected share of the $n$ people who are alone in their cell is about
$$\frac{\text{unique people}}{n} \approx \exp\big(-2^{\log_2 n - b}\big).$$ {#eq-link-uniform}
The share depends only on the bits beyond $\log_2 n$, and it rises steeply with them. Half the people are unique at only 0.53 bits beyond $\log_2 n$.

For example, ZIP code, birth date and sex carry 31.1 bits, a few more than the 27.9 bits that single out one American. The formula then predicts that 0.899 of Americans are unique, close to the 87% that Latanya Sweeney estimated from the 1990 census. Real attributes are skewed and correlated, which usually lowers uniqueness, so the estimate is cautious.

## Sparse data: the Netflix attack {#sec-link-sparse}

Ratings, purchases and locations have no quasi-identifier columns, yet they identify people even better. In 2006 Netflix released about 100 million movie ratings, with random numbers in place of the subscribers' names. Narayanan and Shmatikov (2008) matched its records to public reviews. Their score adds up the items on which a record agrees with what the adversary knows. It weights each item by its rarity, because an obscure film says more than a hit. Eight ratings with dates known to within 14 days identified 99% of the records, even when two of them were wrong.

The best score always points at some record, even when the target is not in the release. So the **eccentricity test** declares a match only if the best score beats the second best by a margin. It counts the margin in standard deviations of all the scores. Set that margin on people who are not in the data. On the full edition's simulated ratings, the original attack's margin also matched 26.3% of such outsiders. So report the attack's success at a stated false positive rate ([Hypothesis testing](../part01-prerequisites/04-hypothesis-testing.md)).

## Does releasing a sample protect? {#sec-link-sampling}

Agencies often release a sample, as Adult is, because a record unique in the sample may share its attributes with someone left out. This protects less than it seems. Suppose the release holds a share $q$ of the population, the **sampling rate**, and the adversary finds exactly one record that matches a person. Write $\lambda$ for the expected number of other people in the population who share the target's cell. The record can be a look-alike's only if the target was left out, which happens with probability $1 - q$. So for a rare cell, the probability $\kappa$ that the match is right is about
$$\kappa \approx \frac{1}{1 + \lambda(1 - q)}.$$ {#eq-link-kappa}

For example, take a cell expected to hold $\lambda = 0.1$ other people. Even in a 1% sample, a match is right with probability 0.910. The sample size enters only through $1 - q$, which can never exceed 1. So the chance of a right match never falls below $1/(1 + \lambda)$, and a smaller sample barely helps. A sample protects only people whose cells are crowded anyway.

So the Adult figures above are **sample uniqueness**, exact when the adversary knows the target is in the release. The **population uniqueness** behind them is lower, but not by much for rare cells. Estimate it with a model, and do not assume the risk away. ARX and sdcMicro offer such estimates, and Rocher, Hendrickx and de Montjoye (2019) model correlated attributes for it.

## A pre-release risk check {#sec-link-check}

First list every column an adversary could find next to a name elsewhere, such as a date, a location or a rare event. No tool knows what other data exist, so no tool can make this list. To lower the risk, coarsen geography and dates first, because they carry the most bits. Then suppress the records that remain unique ([Syntactic anonymity and how it breaks](08-syntactic-anonymity.md)). If the analysis needs fine-grained or sparse data, release a differentially private output instead ([Release statistics with a library](../part03-dp-foundations/12-release-statistics.md)). [Pre-release privacy risk assessment](10-risk-assessment.md) runs this check with thresholds fixed before the counts are seen.

## Cheat sheet

| Question | Result | Use it for |
| --- | --- | --- |
| Bits to name one person | $\log_2 n$ | 27.9 for the US of 1990 |
| Risk of a release you hold | Success $J/n$; at most $1/k$ if every class has at least $k$ records | Prosecutor model |
| Share unique, uniform cells | $\exp(-2^{\log_2 n - b})$ | 50% at $\log_2 n + 0.53$ bits, 99% at $\log_2 n + 6.64$ |
| Releasing a sample | $\kappa \approx 1/(1 + \lambda(1 - q))$ | 0.910 at $\lambda = 0.1$ in a 1% sample |

: The results of this chapter. {#tbl-link-cheat}

**Rules of thumb**

- Add quasi-identifiers one at a time, because one innocent column can more than double the unique share.
- Estimate uniqueness for the population, not just the sample.
- Hashing an identifier is not anonymization. Key the hash, and use a new key for each release.

## References

- Sweeney, "Simple Demographics Often Identify People Uniquely" (2000): the 87% estimate and its variants with coarser geography.
- Narayanan & Shmatikov, "Robust De-anonymization of Large Sparse Datasets" (2008): the Netflix attack.
- Rocher, Hendrickx & de Montjoye (2019), in *Nature Communications*: population uniqueness from a sample.

**Derivations.** Every result here is derived step by step in [the full edition's chapter](../../book-full/part02-attacks/07-linkage-reidentification.md).
