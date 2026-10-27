# Linkage and re-identification

## Why this matters

A hospital shares its discharge records with researchers after deleting the names, but keeps the postcode, the date of birth, the sex and the diagnosis. Anyone with a voter roll, which lists names next to those same three columns, can join the two tables and put a name back on most rows. This is a **linkage attack**, and putting a name back on a "de-identified" record is **re-identification**.

Whether a table can be linked is judged by counting identifying information in bits. Each bit halves the candidates, so singling out one person in a population of $n$ people takes $\log_2 n$ bits, 27.9 bits for the United States of 1990. Almost everyone is unique once the attributes an adversary knows carry a few bits more than that, and a unique record can be linked.

## Quasi-identifiers and the linkage attack {#sec-link-quasi}

**Direct identifiers**, like a name or an email address, name a person on their own. **Sensitive attributes**, such as a diagnosis, are what the release is about. **Quasi-identifiers** sit in between: none of them names anyone alone, but together they often single out one person, and they appear next to names in other datasets.

To score a release you hold, group its $n$ records into **equivalence classes**, the sets of records with identical quasi-identifiers, and write $J$ for the number of classes. Suppose an adversary knows the target's quasi-identifiers and that the target is in the release. If the target is alone in its class, the adversary names its record for sure, and if the class holds $k$ look-alikes, one time in $k$. A class of $k$ records holds $k$ targets, each named with probability $1/k$, so every class adds one win whatever its size. Averaged over all $n$ targets, the chance of naming the right record is therefore
$$\Pr[\text{win}] = \frac{J}{n},$$ {#eq-link-success}
and no adversary that sees only the quasi-identifiers can do better. This is the **prosecutor model**, the worst case, because it assumes the adversary knows the target is in the data.

The script [practitioner_table_risk.py](../../code/part02-attacks/practitioner_table_risk.py) shows how fast uniqueness grows on a real table, Adult, the 48,842 people of the 1994 US Current Population Survey in the UCI repository. Adult has no ZIP code or birth date, so the script starts from age, sex and race. It adds one quasi-identifier at a time, with native country as the only geography. For each set it prints the share of unique records, the share in classes smaller than 5, the smallest class size and $J/n$.

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

Three columns that look harmless leave 0.12% of the records unique, and six leave 14.64%. Adding the hours worked per week, which nobody would list as an identifier, makes 36.55% unique and puts more than half the records in classes smaller than 5. So add every column an adversary could know to the test, one at a time, because the one that breaks the table may be one nobody suspected. A minimum class size of 5 is a common convention, yet every set here has classes of a single record.

## Pseudonyms and hashed identifiers {#sec-link-pseudonyms}

Many releases replace each identifier by a **pseudonym**, a code that stays the same across a person's records, and such codes fail in two ways. First, a hash without a secret key, such as plain MD5, can be reversed whenever the possible identifiers can be listed, because the adversary can hash every candidate and look each pseudonym up. New York City hashed its taxi drivers' licences this way in its 2013 trip release. There were only about 2 million possible licences, so in 2014 Vijay Pandurangan hashed them all in minutes, and public lists of licence holders then named each driver.

The second failure survives any key. The pseudonym ties a person's records together, so linking any one of them reveals all the others. In the taxi data, a pickup minute and place single out one trip, which a passenger who knows who drove it can find. The driver's pseudonym, even a random one, then reveals every trip that driver made. This is why the General Data Protection Regulation (GDPR) still counts pseudonymized data as personal data.

If you must keep a join key, derive it with a keyed hash, a hash-based message authentication code (HMAC), and keep the key away from the data.

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

The key stops the dictionary attack but not the second failure, since each person's records still form an identifying history. Reusing the key would join that history across releases, so draw a fresh key for each release.

## How many people are unique? {#sec-link-unique}

For a release still being designed, there is no table to count, so estimate uniqueness by adding bits. An attribute with $v$ equally likely values carries $\log_2 v$ bits, and the bits of independent attributes add. Attributes that carry $b$ bits in total split people into $2^b$ cells. If each person lands in a cell at random, every cell equally likely, the expected share of the $n$ people who are alone in their cell is about
$$\frac{\text{unique people}}{n} \approx \exp\big(-2^{\log_2 n - b}\big).$$ {#eq-link-uniform}
The share depends only on the bits beyond $\log_2 n$, and it rises steeply with them. Half the people are unique at 0.53 bits beyond $\log_2 n$, and 99% at 6.64 bits.

For example, ZIP code, birth date and sex carry 31.1 bits, a few more than the 27.9 bits that single out one American. The formula then predicts that 0.899 of Americans are unique, close to the 87% that Latanya Sweeney estimated from the 1990 census. Real attributes are skewed and correlated, which usually lowers uniqueness, so the estimate errs on the side of caution.

## Sparse data: the Netflix attack {#sec-link-sparse}

Ratings, purchases and locations have no quasi-identifier columns, yet they identify people even better. Netflix released about 100 million movie ratings in 2006, with the subscribers' names replaced by random numbers. Narayanan and Shmatikov (2008) matched its records to public reviews with a score that adds up the items on which a record agrees with what the adversary knows. Each item is weighted by its rarity, because agreeing on an obscure film says more than agreeing on a hit. With this score, eight ratings with dates known to within 14 days identified 99% of the records, even when two of them were wrong.

The best score always points at some record, even when the target is not in the release. So the **eccentricity test** declares a match only if the best score beats the second best by a margin, counted in standard deviations of all the scores. Calibrate that margin on people known not to be in the data: on the full edition's simulated ratings, the original attack's margin also matched 26.3% of such outsiders. So report the attack's success at a stated false positive rate ([Hypothesis testing](../part01-prerequisites/04-hypothesis-testing.md)).

## Does releasing a sample protect? {#sec-link-sampling}

Agencies often release a sample, as Adult is, arguing that a record unique in the sample may share its attributes with someone left out. The argument is weaker than it sounds. Suppose the adversary knows a person and finds exactly one matching record in a release that holds a share $q$ of the population, the **sampling rate**. Write $\lambda$ for the expected number of other people in the population who share the target's cell. The record found can be a look-alike's only if the target was left out, which happens with probability $1 - q$. So for a rare cell, the probability $\kappa$ that the match is right is about
$$\kappa \approx \frac{1}{1 + \lambda(1 - q)}.$$ {#eq-link-kappa}

Take a rare cell, one expected to hold $\lambda = 0.1$ other people. Even a 1% sample leaves a match right with probability 0.910. Sampling enters only through $1 - q$, which can never exceed 1, so the chance of a right match never falls below $1/(1 + \lambda)$, and shrinking the sample barely helps. Sampling protects only people whose cells are crowded anyway.

So the Adult figures above are **sample uniqueness**, exact when the target is known to be in the release. The **population uniqueness** behind them is lower, but not by much for rare cells, so estimate it with a model rather than assume the risk away. ARX and sdcMicro offer such estimates, and Rocher, Hendrickx and de Montjoye (2019) model correlated attributes for it.

## A pre-release risk check {#sec-link-check}

A quasi-identifier is any column an adversary could find next to a name elsewhere, such as a date, a location or a rare event, so list them all. No tool can do this for you, because no tool knows what auxiliary data exist. To lower the risk, coarsen geography and dates first, since they carry the most bits. Then suppress the records that remain unique ([Syntactic anonymity and how it breaks](08-syntactic-anonymity.md)). If the analysis needs fine-grained or sparse data, release a differentially private output instead ([Release statistics with a library](../part03-dp-foundations/12-release-statistics.md)). [Pre-release privacy risk assessment](10-risk-assessment.md) runs this check with thresholds fixed before the counts are seen.

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
