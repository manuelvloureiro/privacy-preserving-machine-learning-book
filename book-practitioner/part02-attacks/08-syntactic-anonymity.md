# Syntactic anonymity and how it breaks

## Why this matters

[Linkage and re-identification](07-linkage-reidentification.md) showed that deleting names does not anonymize a table, because quasi-identifiers such as age, postcode and sex single out many people. The classic answer is to coarsen the table until every person hides in a crowd. The definitions that make this precise are called **syntactic**, because you check each one by reading the published table. Statistics offices and tools such as ARX and sdcMicro still use them, so you will meet them in audits. But each holds for one table against an adversary whose knowledge is fixed in advance, so one unanticipated fact, or a second release of the same people, voids it.

## What the adversary believes about a class {#sec-anon-posterior}

To anonymize a table, the custodian, the organization that holds the data, applies a **generalization**, which turns an age into a range or a postcode into a prefix. Records with the same generalized quasi-identifiers form an **equivalence class**. An adversary who knows the target's quasi-identifiers finds the target's class $E$ but cannot tell its records apart. So its belief about the target's sensitive value, its **posterior**, is the share of the class with that value. Write $\lvert E\rvert$ for the size of the class and $n_E(v)$ for the number of its records with a sensitive value $v$. The posterior that the target's sensitive value, written $S_{\text{target}}$, equals $v$ is

$$\Pr[S_{\text{target}} = v \mid \text{release}] = \frac{n_E(v)}{\lvert E\rvert}.$$ {#eq-anon-posterior}
Background knowledge works by elimination: ruling out a value crosses out its records, and the posterior is recomputed over those that remain. Every attack in this chapter is this calculation.

## k-anonymity and what it costs {#sec-anon-k}

**k-anonymity** (Sweeney and Samarati, 1998) asks that every class hold at least $k$ records, for a crowd size $k$ the custodian chooses. Generalization gets there, together with **suppression**, which deletes the records that still stand out. In the previous chapter, an adversary who knows the target is in the release links it with probability $J/n$, the number of classes $J$ over the number of records $n$. At least $k$ records per class means at most $n/k$ classes, so
$$\Pr[\text{win}] = \frac{J}{n} \le \frac1k.$$ {#eq-anon-k-link}
The guarantee covers names only, and only against an adversary who knows nothing beyond the quasi-identifiers. The choice of $k$ is a policy: a minimum of 5 is common, and some health-data rules ask for more.

Anonymizers such as **Mondrian** search for a generalization that loses little, but you can also generalize by hand and recheck. The script [practitioner_table_risk.py](../../code/part02-attacks/practitioner_table_risk.py) takes the previous chapter's Adult census extract and its six quasi-identifiers. It puts age into 10-year bands, native country into US or other, and education into four levels. Then it counts the records alone in their class, and the different occupations in each class, with occupation standing in for a sensitive attribute. Both are Polars counts, since no maintained Python package computes k-anonymity or distinct $\ell$.

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

Generalizing cut the unique share to 0.94%, from 14.64% before. Yet the smallest class still holds one record, so the 3.31% of records in classes smaller than 5 must be suppressed, or the bands made coarser. And the result is fragile, because an adversary who also knows the hours worked finds 10.09% of the records unique again.

## Homogeneity, background knowledge and l-diversity {#sec-anon-l}

k-anonymity hides who a record belongs to, but not what it says (Machanavajjhala and colleagues, 2006). If all three records of Alice's class say heart disease, her posterior is 1 without anyone finding her record, which is the **homogeneity attack**. If Bob's class says cancer, cancer and flu, a neighbour who knows he did not have flu crosses out that record. The neighbour learns that Bob has cancer, which is the **background knowledge attack**.

**l-diversity** asks each class to hold enough different sensitive values. The distinct version asks for at least $\ell$ different values. It does not cap the most common value's share, so a class can be 3-diverse with one value in most of its records. The recursive version adds that cap, with a constant $c$ the custodian picks, so that an adversary who rules out $\ell - 2$ values keeps a posterior below $c/(1 + c)$. That cap is already one half at $c = 1$, so even this version protects against certainty, not likelihood.

Suppressing the small classes also removes most homogeneous ones. On Adult, 1.20% of the generalized records sit in a class where everyone has the same occupation, but only 0.03% sit in such a class of 5 or more. On a synthetic hospital table of the full edition, distinct 3-diversity removed the certain disclosures at little cost, while the recursive version forced some classes to grow to 641 patients.

## Measuring what a release teaches {#sec-anon-baseline}

Compare every posterior with the **imputation baseline** of [Attribute inference and model inversion](03-attribute-inference.md), the best guess of a sensitive value from the population alone. Only the success beyond it is about the individual. On the hospital table, guessing each patient's diagnosis from age alone is right for 53.6% of patients, and a 5-anonymous release raises that to 59.3%. Most of what the release teaches is the link between age and diagnosis, which any useful study would report. Only the excess of 5.7 percentage points is a leak.

## Skewness, similarity and t-closeness {#sec-anon-t}

l-diversity misses two attacks (Li, Li and Venkatasubramanian, 2007). In the **skewness attack**, a class that is half HIV is 2-diverse, yet in a table where 1% have HIV it raises its members' posterior fifty-fold. In the **similarity attack**, a class of three different stomach diseases is 3-diverse but still reveals a stomach disease.

**t-closeness** answers both, by asking that the distribution of values in every class $E$ be within a distance $t$ of the whole table's. Write $P_E(A)$ and $P(A)$ for the chance that a value drawn from the class, or from the table, lands in a set $A$ of values. With the **total variation (TV)** distance, no such chance, and so no belief, moves by more than $t$,
$$\lvert P_E(A) - P(A)\rvert \le t.$$ {#eq-anon-tv}
The bound is a difference, not a ratio, so it protects rare values badly. At $t = 0.15$, a value held by 1% of the table may rise to 16%. Against the similarity attack, Li and colleagues used the **earth mover's distance (EMD)** instead, which counts similar values as close.

t-closeness costs a lot. At $t = 0.15$, guessing diagnoses from the hospital release was right for only 22.2% of patients, below the 53.6% of imputation from age alone. The release told less about the population than no release.

## The composition attack {#sec-anon-composition}

Ganta, Kasiviswanathan and Smith (2008) asked what happens when two hospitals each publish a k-anonymous or l-diverse table. A patient treated at both sits in a class $E_A$ of one release and a class $E_B$ of the other. The adversary keeps only the diagnoses that appear in both classes,
$$\{v : n_{E_A}(v) > 0\} \cap \{v : n_{E_B}(v) > 0\}.$$
When one candidate is left, the diagnosis is **pinned down**.

In the full edition's simulation, two hospitals share 6,649 patients. Their two 5-anonymous releases together pin down the diagnosis of 9.9% of them, over three times what either pins alone. With distinct 3-diversity neither release pins anyone, yet together they pin 18.

No single-table definition can stop this, because the intersection of two tables is a table that neither custodian checked. A definition that survives must compose across releases, hold whatever the adversary knows, and bound the ratio of beliefs, so that rare values are protected. [The definition of differential privacy](../part03-dp-foundations/01-definition.md) starts from these requirements, which is why statistics offices such as the US Census Bureau have moved towards it ([Reconstruction attacks](09-reconstruction.md)).

## Choosing and running an anonymization {#sec-anon-workflow}

Start from the quasi-identifiers of [Linkage and re-identification](07-linkage-reidentification.md), since a list that is too short is the commonest failure. Use k-anonymity against linkage, and add distinct l-diversity when a class with one sensitive value would be a disclosure. Use t-closeness, the costliest, only when the sensitive attribute is skewed or has similar values.

Then audit the anonymizer's output yourself, in Polars as the excerpt does, or with pycanon, which reports the crowd size $k$, $\ell$ and $t$ for a pandas table. [Pre-release privacy risk assessment](10-risk-assessment.md) turns these checks into a release decision. Move to differential privacy when other releases of the same people exist, when the sensitive attribute is rare, or when the data are sparse. For tables of counts and sums, [Release statistics with a library](../part03-dp-foundations/12-release-statistics.md) does the job end to end.

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
