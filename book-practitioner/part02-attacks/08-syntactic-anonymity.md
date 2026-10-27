# Syntactic anonymity and how it breaks

## Why this matters

[Linkage and re-identification](07-linkage-reidentification.md) showed that a table without names is not anonymous. Quasi-identifiers such as age, postcode and sex single out many people. The classic answer is to coarsen the table until every person hides in a crowd. The definitions of this answer are **syntactic**, because you check each one by reading the published table. You will meet them in audits, because statistics offices and tools such as ARX and sdcMicro still use them. But each one holds for one table and for an adversary whose knowledge is fixed in advance. One unexpected fact, or a second release of the same people, breaks it.

## What the adversary believes about a class {#sec-anon-posterior}

To anonymize a table, the custodian, the organization that holds the data, applies a **generalization**. It turns an age into a range or a postcode into a prefix. Records with the same generalized quasi-identifiers form an **equivalence class**. An adversary who knows the target's quasi-identifiers finds the target's class $E$ but cannot tell its records apart. So its belief about the target's sensitive value, its **posterior**, is the share of the class with that value. Write $\lvert E\rvert$ for the size of the class and $n_E(v)$ for the number of its records with a sensitive value $v$. Then the posterior that the target's sensitive value, written $S_{\text{target}}$, equals $v$ is

$$\Pr[S_{\text{target}} = v \mid \text{release}] = \frac{n_E(v)}{\lvert E\rvert}.$$ {#eq-anon-posterior}
The adversary's background knowledge about the target works by elimination. A fact that rules out a value crosses out its records, and the adversary computes the posterior again over the records that remain. Every attack in this chapter is this calculation.

## k-anonymity and what it costs {#sec-anon-k}

**k-anonymity** (Sweeney and Samarati, 1998) asks that every class hold at least $k$ records, for a crowd size $k$ that the custodian chooses. Generalization gets there, together with **suppression**, which deletes the records that still stand out. In the previous chapter, an adversary who knows the target is in the release links it with probability $J/n$, the number of classes $J$ over the number of records $n$. So with at least $k$ records per class, there are at most $n/k$ classes, and
$$\Pr[\text{win}] = \frac{J}{n} \le \frac1k.$$ {#eq-anon-k-link}
The guarantee covers names only, and only against an adversary who knows nothing beyond the quasi-identifiers. The choice of $k$ is a policy, and 5 is a common minimum.

Anonymizers such as **Mondrian** search for a generalization that loses little, but you can also generalize by hand and check again. The script [practitioner_table_risk.py](../../code/part02-attacks/practitioner_table_risk.py) takes the previous chapter's Adult census extract and its six quasi-identifiers. It puts age into 10-year bands, native country into US or other, and education into four levels. Then it counts the records alone in their class, and the different occupations in each class, with occupation as the sensitive attribute. Both are Polars counts, because no maintained Python package computes k-anonymity or distinct $\ell$.

```{.python excerpt="generalize-and-check"}
general = df.with_columns(
    (pl.col("age") // 10 * 10).alias("age"),  # 10-year bands: 30 stands for 30-39
    pl.when(pl.col("native-country") == "United-States")
    .then(pl.lit("US"))
    .otherwise(pl.lit("other"))
    .alias("native-country"),
    # education-num runs from 1 (preschool) to 16 (doctorate); 9 is high-school graduate.
    pl.col("education-num")
    .cut([8, 9, 12], labels=["no diploma", "HS-grad", "some college", "bachelor+"])
    .alias("education"),
)
# Recheck, and count the occupations in each class (distinct l).
classes = general.group_by(QIS).agg(pl.len(), pl.col("occupation").n_unique().alias("l"))
unique = classes.filter(pl.col("len") == 1).height / general.height
one_job = classes.filter(pl.col("l") == 1)["len"].sum() / general.height
print(f"after generalizing: unique={unique:.2%}  in a class with one occupation={one_job:.2%}")
```

Run it with `uv run python code/part02-attacks/practitioner_table_risk.py`. Its first two output lines use the six generalized quasi-identifiers, and the third adds the raw hours worked per week.

```text
after generalizing: unique=0.94%  in a class with one occupation=1.20%
age + sex + race + native-country + marital-status + education: unique=0.94%  in classes <5=3.31%  min k=1  J/n=0.032
age + sex + race + native-country + marital-status + education + hours-per-week: unique=10.09%  in classes <5=21.78%  min k=1  J/n=0.175
  records in classes of size >=5 with l=1: 0.03%
```

Generalization cut the unique share to 0.94%, from 14.64% before. Yet the smallest class still holds one record. So suppress the 3.31% of records in classes smaller than 5, or make the bands coarser. The result is fragile, because an adversary who also knows the hours worked finds 10.09% of the records unique again.

## Homogeneity, background knowledge and l-diversity {#sec-anon-l}

k-anonymity hides who a record belongs to, but not what it says (Machanavajjhala and colleagues, 2006). In the **homogeneity attack**, all three records of Alice's class say heart disease. Her posterior is then 1, and nobody has to find her record. In the **background knowledge attack**, Bob's class says cancer, cancer and flu. A neighbour who knows that Bob did not have flu crosses out that record and learns that Bob has cancer.

**l-diversity** asks each class to hold enough different sensitive values. The distinct version asks for at least $\ell$ different values. It does not cap the most common value's share, so a class can be 3-diverse with one value in most of its records. The recursive version adds that cap, with a constant $c$ that the custodian picks. An adversary who rules out $\ell - 2$ values then keeps a posterior below $c/(1 + c)$. That cap is already one half at $c = 1$, so even this version protects against certainty, not likelihood.

Suppression of the small classes also removes most homogeneous ones. On Adult, 1.20% of the generalized records sit in a class where everyone has the same occupation. Only 0.03% sit in such a class of 5 or more. On a synthetic hospital table of the full edition, distinct 3-diversity removed the certain disclosures at little cost. The recursive version forced some classes to grow to 641 patients.

## Measuring what a release teaches {#sec-anon-baseline}

Compare every posterior with the **imputation baseline** of [Attribute inference and model inversion](03-attribute-inference.md), the best guess of a sensitive value from the population alone. Only the success beyond it is about the individual. On the hospital table, a guess of each patient's diagnosis from age alone is right for 53.6% of patients, and a 5-anonymous release raises that to 59.3%. Most of that gain is the link between age and diagnosis, which any useful study reports. Only the excess of 5.7 percentage points is a leak.

## Skewness, similarity and t-closeness {#sec-anon-t}

l-diversity misses two attacks (Li, Li and Venkatasubramanian, 2007). In the **skewness attack**, a class that is half HIV is 2-diverse, yet in a table where 1% have HIV it raises its members' posterior fifty-fold. In the **similarity attack**, a class of three different stomach diseases is 3-diverse but still reveals a stomach disease.

**t-closeness** answers both. It asks that the distribution of values in every class $E$ be within a distance $t$ of the whole table's. Write $P_E(A)$ and $P(A)$ for the chance that a value drawn from the class, or from the table, lands in a set $A$ of values. With the **total variation (TV)** distance, no such chance, and so no belief, moves by more than $t$,
$$\lvert P_E(A) - P(A)\rvert \le t.$$ {#eq-anon-tv}
The bound is a difference, not a ratio, so it protects rare values badly. At $t = 0.15$, a value held by 1% of the table may rise to 16%. Against the similarity attack, Li and colleagues used the earth mover's distance instead, which counts similar values as close.

t-closeness costs a lot. At $t = 0.15$, a guess of diagnoses from the hospital release was right for only 22.2% of patients, below the 53.6% from age alone. So the release told less about the population than no release.

## The composition attack {#sec-anon-composition}

Ganta, Kasiviswanathan and Smith (2008) asked what happens when two hospitals each publish a k-anonymous or l-diverse table. A patient treated at both sits in a class $E_A$ of one release and a class $E_B$ of the other. The adversary keeps only the diagnoses that appear in both classes,
$$\{v : n_{E_A}(v) > 0\} \cap \{v : n_{E_B}(v) > 0\}.$$
When one candidate is left, the diagnosis is **pinned down**.

In the full edition's simulation, two hospitals share 6,649 patients. Together, their two 5-anonymous releases pin down the diagnosis of 9.9% of them, over three times what either pins alone. With distinct 3-diversity, neither release pins anyone, yet together they pin 18.

No definition for a single table can stop this, because the intersection of two tables is a table that neither custodian checked. A definition that survives must compose across releases and hold whatever the adversary knows. It must also bound the ratio of beliefs, so that it protects rare values. [The definition of differential privacy](../part03-dp-foundations/01-definition.md) meets these needs, and statistics offices such as the US Census Bureau have moved towards it ([Reconstruction attacks](09-reconstruction.md)).

## Choosing and running an anonymization {#sec-anon-workflow}

Start from the quasi-identifiers of [Linkage and re-identification](07-linkage-reidentification.md), because a list that is too short is the most common failure. Use k-anonymity against linkage. Add distinct l-diversity when a class with one sensitive value is a disclosure. Use t-closeness, the costliest, only when the sensitive attribute is skewed or has similar values.

Then audit the anonymizer's output yourself, in Polars as the excerpt does, or with pycanon, which reports $k$, $\ell$ and $t$ for a pandas table. [Pre-release privacy risk assessment](10-risk-assessment.md) turns these checks into a release decision. Move to differential privacy if other releases of the same people exist, if the sensitive attribute is rare, or if the data are sparse. For tables of counts and sums, [Release statistics with a library](../part03-dp-foundations/12-release-statistics.md) does the job end to end.

## Cheat sheet

| Definition or attack | Condition | What it gives, and where it ends |
| --- | --- | --- |
| k-anonymity | Every class has at least $k$ records | Linkage at most $1/k$; says nothing about attributes |
| Distinct, entropy l-diversity | At least $\ell$ values, or $\log_2 \ell$ bits of entropy | No cap on the top value's share |
| Recursive $(c, \ell)$-diversity | Top value below $c$ times the values ranked $\ell$-th and lower | Posterior below $c/(1 + c)$ after ruling out $\ell - 2$ values |
| t-closeness (TV) | Each class within $t$ of the table | A 1% value may reach $1\% + t$ |
| Composition | Intersect the target's classes in two releases | No single-table definition prevents it |

: The syntactic definitions and the attacks on them. {#tbl-anon-cheat}

**Rules of thumb**

- k-anonymity protects names, not attributes, so check what each class's sensitive values reveal.
- Assume another release exists, because a guarantee about one table is not a guarantee about a person.

## References

- Machanavajjhala and colleagues, "l-Diversity" (2006), and Li, Li & Venkatasubramanian, "t-Closeness" (2007): the attacks and the definitions that answer them.
- Ganta, Kasiviswanathan & Smith, "Composition Attacks and Auxiliary Information in Data Privacy" (2008): the intersection attack.

**Derivations.** Every result here is derived step by step in [the full edition's chapter](../../book-full/part02-attacks/08-syntactic-anonymity.md).
