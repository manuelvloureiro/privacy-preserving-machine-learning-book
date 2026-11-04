# Linkage and re-identification

## Why this matters

A hospital removes names from discharge records but retains postcode, date of birth, sex and diagnosis. A voter roll lists names with the same quasi-identifiers. Joining the tables can identify many records. This is a **linkage attack**; identifying a person in a supposedly de-identified record is **re-identification**.

Estimate linkage risk by measuring identifying information in bits. Each bit halves the candidate set, so $\log_2 n$ bits can distinguish one person among $n$. For the 1990 United States, this is 27.9 bits. When known attributes carry slightly more information, many people have unique combinations that can be linked.

## Quasi-identifiers and the linkage attack {#sec-link-quasi}

**Direct identifiers**, such as names or email addresses, identify a person on their own. **Sensitive attributes**, such as diagnoses, are the subject of the release. **Quasi-identifiers** do not identify people individually but can do so in combination; they also appear beside names in other datasets.

To assess a release, group its $n$ records into **equivalence classes**, each containing records with identical quasi-identifiers. Let $J$ be the number of classes. Suppose an adversary knows the target's quasi-identifiers and knows that the target is in the release. If the target's class contains $k$ records, the adversary identifies the target with probability $1/k$. Each class contributes one expected correct identification across its $k$ records. Thus, across all $n$ targets, the success probability is
$$\Pr[\text{win}] = \frac{J}{n},$$ {#eq-link-success}
No adversary using only quasi-identifiers can do better. This is the **prosecutor model**, a worst-case assumption in which the adversary knows that the target is in the release.

The script [practitioner_table_risk.py](../../code/part02-attacks/practitioner_table_risk.py) measures uniqueness on Adult, a dataset of 48,842 people from the 1994 US Current Population Survey in the UCI repository. Because Adult lacks ZIP codes and birth dates, the script starts with age, sex and race, then adds quasi-identifiers one at a time. For each set, it prints the unique-record share, the share in classes smaller than 5, the minimum class size and $J/n$.

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

Three columns leave 0.12% of records unique; six leave 14.64%. Adding hours worked per week, which is not usually treated as an identifier, raises the unique share to 36.55%. Test every column an adversary could know, adding them one at a time. Every set here contains single-record classes, despite the common minimum class-size rule of 5.

## Pseudonyms and hashed identifiers {#sec-link-pseudonyms}

Many releases replace identifiers with **pseudonyms**, codes that remain constant across one person's records. These codes can fail in two ways. First, an adversary can reverse an unkeyed hash such as MD5 when the possible identifiers can be enumerated: hash each candidate and look for a match. New York City used this approach for taxi-driver licenses in its 2013 trip release. The set contained about 2 million possible licenses; in 2014, Vijay Pandurangan hashed them all and identified the drivers.

The second risk remains even with a key. A pseudonym links one person's records, so identifying one record reveals the others. In the taxi data, a passenger who knows a trip's pickup time and place can identify the driver; the driver's pseudonym then links all their trips. The General Data Protection Regulation (GDPR) therefore still treats pseudonymized data as personal data.

If a join key is necessary, generate it with a keyed hash, such as a hash-based message authentication code (HMAC), and store the key separately from the data.

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

The key prevents dictionary attacks but does not prevent linkage through a person's record history. Reusing the key across releases enables cross-release linkage, so generate a new key for each release.

## How many people are unique? {#sec-link-unique}

Before the table exists, estimate uniqueness by summing information across attributes. An attribute with $v$ equally likely values carries $\log_2 v$ bits; independent attributes' bits add. Attributes with $b$ total bits divide the population into $2^b$ cells. If each person is assigned uniformly at random to a cell, the expected fraction of $n$ people alone in their cell is approximately
$$\frac{\text{unique people}}{n} \approx \exp\big(-2^{\log_2 n - b}\big).$$ {#eq-link-uniform}
The fraction depends on the number of bits above $\log_2 n$ and increases rapidly with that excess. Half the people are unique at 0.53 bits above $\log_2 n$.

ZIP code, birth date and sex carry 31.1 bits, slightly more than the 27.9 bits needed to single out one American. The formula predicts that 0.899 of Americans are unique, close to Latanya Sweeney's 87% estimate from the 1990 census. Real attributes are skewed and correlated, which usually lowers uniqueness; the uniform-cell estimate is therefore cautious.

## Sparse data: the Netflix attack {#sec-link-sparse}

Ratings, purchases and locations can identify people without quasi-identifier columns. In 2006, Netflix released about 100 million movie ratings with random identifiers instead of subscriber names. Narayanan and Shmatikov (2008) matched these records to public reviews. Their score sums agreement with known items, weighting rare items more heavily than popular ones. Eight ratings with dates known within 14 days identified 99% of records, even when two ratings were incorrect.

The highest score always identifies some record, even when the target is absent. The **eccentricity test** declares a match only when the highest score exceeds the second-highest by a specified margin, measured in standard deviations of all scores. Calibrate this margin using people known to be outside the release. In the full edition's simulated ratings, the original attack's margin also matched 26.3% of outsiders. Report attack success at a specified false positive rate ([Hypothesis testing](../part01-prerequisites/04-hypothesis-testing.md)).

## Does releasing a sample protect? {#sec-link-sampling}

Agencies often release samples, as with Adult, because a record unique in the sample may match someone outside it. Let $q$ be the **sampling rate**, the fraction of the population in the release. Suppose the adversary finds exactly one matching record, and let $\lambda$ be the expected number of other people in the population who share the target's cell. A look-alike can account for the match only if the target was not sampled, with probability $1-q$. For a rare cell, the probability $\kappa$ that the match is correct is approximately
$$\kappa \approx \frac{1}{1 + \lambda(1 - q)}.$$ {#eq-link-kappa}

For a cell with $\lambda = 0.1$ expected other people, a match is correct with probability 0.910 even in a 1% sample. Since $1-q$ cannot exceed 1, the probability of a correct match is at least $1/(1+\lambda)$. Reducing the sample size therefore offers little protection for rare cells. Sampling mainly helps when cells are already crowded.

The Adult figures above measure **sample uniqueness**, which is exact under the assumption that the target is in the release. **Population uniqueness** is lower but remains close for rare cells. Estimate it with a model; ARX and sdcMicro provide such estimates, and Rocher, Hendrickx and de Montjoye (2019) model attribute correlations.

## A pre-release risk check {#sec-link-check}

List every field an adversary could find beside a name in another source, such as a date, location or rare event. No tool can identify all available auxiliary data. Reduce risk by first coarsening geography and dates, which often carry many bits, then suppressing records that remain unique ([Syntactic anonymity and how it breaks](08-syntactic-anonymity.md)). For analyses that require fine-grained or sparse data, release differentially private outputs instead ([Release statistics with a library](../part03-dp-foundations/12-release-statistics.md)). [Pre-release privacy risk assessment](10-risk-assessment.md) applies thresholds set before examining the counts.

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
