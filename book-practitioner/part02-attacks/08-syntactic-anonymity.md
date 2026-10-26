# Syntactic anonymity and how it breaks

## Why this matters

[Linkage and re-identification](07-linkage-reidentification.md) showed that deleting names does not anonymize a table, because the remaining quasi-identifiers, such as age, postcode and sex, single out many people. The classic answer is to coarsen the table until every person hides in a crowd. Between 1998 and 2008, researchers turned this into definitions that constrain the released table itself. They are called **syntactic** because you can check each one by reading the published table.

Statistics offices, clinical data sharing and open-source anonymization tools such as ARX and sdcMicro still use these definitions, so you will meet them in audits. They are not enough, because each is a property of one table against an adversary whose knowledge is fixed in advance. One unanticipated fact, or a second release of the same people, voids the guarantee.

## What the adversary believes about a class {#sec-anon-posterior}

To anonymize a table, the custodian, the organization that holds the data, applies a **generalization**, which turns an age into a range or a postcode into a prefix. Records with the same generalized quasi-identifiers form an **equivalence class**. An adversary who knows the target's quasi-identifiers can find the target's class $E$, but the records in it look alike. So if it knows nothing else, its belief about the target's sensitive value is the share of the class with that value.

Call that belief its **posterior**. Write $\lvert E\rvert$ for the size of the class, and $n_E(v)$ for the number of its records with a sensitive value $v$. Then the posterior that the target's sensitive value, written $S_{\text{target}}$, equals $v$ is
$$\Pr[S_{\text{target}} = v \mid \text{release}] = \frac{n_E(v)}{\lvert E\rvert}.$$ {#eq-anon-posterior}
Background knowledge works by elimination. Ruling out a value crosses out its records, and the posterior is recomputed over the records that remain, so the other values' shares grow. Every attack in this chapter is this calculation.

## k-anonymity and what it costs {#sec-anon-k}

Sweeney and Samarati proposed **k-anonymity** in 1998. A release is k-anonymous if every class has at least $k$ records, for a crowd size $k$ the custodian chooses. Generalization gets there, together with **suppression**, which deletes the records that still stand out. The previous chapter's linkage success, for an adversary who knows the target is in the release, is the number of classes $J$ over the number of records $n$. Classes of at least $k$ records number at most $n/k$, so that success is capped,
$$\Pr[\text{win}] = \frac{J}{n} \le \frac1k.$$ {#eq-anon-k-link}
The guarantee is about names only, and it assumes the adversary knows the quasi-identifiers and nothing more.

The choice of $k$ is a policy. A minimum of 5 is a common convention, and some health-data rules ask for more. The US Centers for Medicare & Medicaid Services, for example, forbid publishing any count from 1 to 10 drawn from their data, in effect a minimum of 11. The HIPAA Privacy Rule also offers two routes to de-identification. Safe Harbor removes a fixed list of identifiers, while expert determination has an expert measure the risk with methods like these.

Anonymizers search for a generalization that loses little. **Mondrian**, for example, splits groups of records at the median of one quasi-identifier while both halves keep $k$ records. The full edition tries it on a synthetic table of 10,000 hospital patients. There, at $k = 5$, a class spans on average only 1.6% of each quasi-identifier's range, so k-anonymity itself costs little precision.

On a real table, you can also generalize by hand and recheck. The script [practitioner_table_risk.py](../../code/part02-attacks/practitioner_table_risk.py) takes the Adult census extract of the previous chapter, with its six quasi-identifiers. It puts age into 10-year bands, native country into US or other, and education into four levels. It then counts the different occupations in each class, with occupation standing in for a sensitive attribute. Both checks are Polars counts, since no maintained Python package computes k-anonymity or distinct $\ell$. ARX in Java and sdcMicro in R do.

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

Run it with `uv run python code/part02-attacks/practitioner_table_risk.py`. The four lines below come from its output. The first two use the six generalized quasi-identifiers, and the third adds the raw hours worked per week. The last counts records in classes of 5 or more with a single occupation.

```text
after generalizing: unique=0.94%  in a class with one occupation=1.20%
age + sex + race + native-country + marital-status + education: unique=0.94%  in classes <5=3.31%  min k=1  J/n=0.032
age + sex + race + native-country + marital-status + education + hours-per-week: unique=10.09%  in classes <5=21.78%  min k=1  J/n=0.175
  records in classes of size >=5 with l=1: 0.03%
```

Generalizing cut the unique share to 0.94%, from 14.64% on the same six columns before generalizing. Yet the table is still not even 2-anonymous, since its smallest class holds one record. So the remaining 3.31% of records in classes smaller than 5 must be suppressed, or the bands made coarser. The third line shows how fragile the result is, because an adversary who also knows the hours worked finds 10.09% of the records unique again.

## Homogeneity, background knowledge and l-diversity {#sec-anon-l}

Machanavajjhala and colleagues (2006) pointed out that k-anonymity hides who a record belongs to, but not what it says. If all three records of Alice's class say heart disease, her posterior is 1 without anyone finding her record. This is the **homogeneity attack**. If Bob's class says cancer, cancer and flu, a neighbour who knows he did not have flu learns that he has cancer, which is the **background knowledge attack**. On the hospital table at $k = 5$, 2.3% of patients sit in classes with a single diagnosis.

**l-diversity** asks each class to hold enough different sensitive values. The distinct version asks for at least $\ell$ different values. The entropy version asks that the class's values carry at least $\log_2 \ell$ bits of entropy, as much as $\ell$ equally common values. Neither caps the share of the most common value, so a class can be 3-diverse with one value in most of its records. The recursive version adds that cap, with a constant $c$ that the custodian picks. It requires the most common value's count to stay below $c$ times the combined count of the values ranked $\ell$-th and lower. Then even an adversary who rules out $\ell - 2$ values keeps a posterior below $c/(1 + c)$. That cap is already one half at $c = 1$, so even the strongest version protects against certainty, not likelihood.

On Adult, 1.20% of the generalized records sit in a class where everyone has the same occupation. Only 0.03% of records sit in such a class of 5 or more. So almost all of them are in small classes, and suppressing those also removes most homogeneous ones. On the hospital table, distinct 3-diversity removed the certain disclosures at little cost, while the recursive version forced some classes to grow to 641 patients.

## Measuring what a release teaches {#sec-anon-baseline}

Compare every posterior with what the adversary would believe without the release. That is the **imputation baseline** of [Attribute inference and model inversion](03-attribute-inference.md), the best guess of a sensitive value from the population alone. Only the success beyond it is about the individual.

On the hospital table, guessing the most likely diagnosis from each patient's age, with no release, is right for 53.6% of patients. A 5-anonymous release raises that to 59.3%. So most of what the release teaches is the link between age and diagnosis, which any useful study would report. Only the excess of 5.7 percentage points is a leak.

## Skewness, similarity and t-closeness {#sec-anon-t}

Li, Li and Venkatasubramanian (2007) found two attacks that l-diversity misses. In the **skewness attack**, a class that is half HIV is 2-diverse, yet in a table where 1% have HIV it raises its members' posterior fifty-fold. In the **similarity attack**, a class of three different stomach diseases is 3-diverse but still reveals a stomach disease.

**t-closeness** answers both, by asking that every class's distribution be within a distance $t$ of the whole table's. Write $P_E(A)$ and $P(A)$ for the chance that a value drawn from the class, or from the whole table, lands in a set $A$ of values. With the **total variation (TV)** distance, the condition says that no such chance, and so no belief, moves by more than $t$,
$$\lvert P_E(A) - P(A)\rvert \le t.$$ {#eq-anon-tv}
The bound is a difference, not a ratio, so it protects rare values badly. At $t = 0.15$, a value held by 1% of the table may rise to 16%.

Against the similarity attack, Li and colleagues used the **earth mover's distance (EMD)**, which counts similar values, such as two stomach diseases, as close. t-closeness costs a lot. At that $t$, guessing diagnoses from the hospital release was right for only 22.2% of patients, below the 53.6% of imputation from age alone. The release then told less about the population than no release.

## The composition attack {#sec-anon-composition}

Ganta, Kasiviswanathan and Smith (2008) asked what happens with two releases. Suppose two hospitals each publish a k-anonymous or l-diverse table. A patient treated at both sits in a class $E_A$ of one release and a class $E_B$ of the other. The adversary then keeps only the diagnoses that appear in both classes,
$$\{v : n_{E_A}(v) > 0\} \cap \{v : n_{E_B}(v) > 0\}.$$
When one candidate is left, the diagnosis is **pinned down**.

In the full edition's simulation, two hospitals share 6,649 patients. Their two 5-anonymous releases together pin down the diagnosis of 9.9% of them, more than three times what either pins alone. With distinct 3-diversity neither release pins anyone, yet together they still pin 18 of them.

No single-table definition can stop this, because the intersection of two tables is a table that neither custodian checked. A definition that survives must answer all three failures. It must compose across releases, hold whatever the adversary knows, and bound the ratio of beliefs rather than their difference, so that rare values are protected. [The definition of differential privacy](../part03-dp-foundations/01-definition.md) starts from these requirements, which is why statistics offices such as the US Census Bureau have moved towards it ([Reconstruction attacks](09-reconstruction.md)).

## Choosing and running an anonymization {#sec-anon-workflow}

Start from the quasi-identifiers of [Linkage and re-identification](07-linkage-reidentification.md), since a list that is too short is the commonest failure. Use k-anonymity against linkage, and add distinct l-diversity when a class with one sensitive value would be a disclosure. Use t-closeness only when the sensitive attribute is skewed or has similar values, because it costs the most.

Run the anonymizer, then audit its output yourself, as the excerpt does in Polars. The Python package pycanon reports $k$, $\ell$ and $t$ for a pandas table, with calls such as `pycanon.anonymity.k_anonymity(df.to_pandas(), qis)`. [Pre-release privacy risk assessment](10-risk-assessment.md) turns these checks into a release decision, with the blocking thresholds fixed before the counts are seen.

Move to differential privacy when other releases of the same people exist, when the sensitive attribute is rare, or when the data are sparse. For tables of counts and sums, [Release statistics with a library](../part03-dp-foundations/12-release-statistics.md) does the job end to end.

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
