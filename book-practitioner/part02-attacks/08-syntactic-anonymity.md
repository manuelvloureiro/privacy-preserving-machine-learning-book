# Syntactic anonymity and how it breaks

## Why this matters

[Linkage and re-identification](07-linkage-reidentification.md) showed that removing names does not make a table anonymous. Combinations of age, postcode and sex can still identify people. A common response is to generalize the table until each person belongs to a group. These definitions are **syntactic** because they can be checked directly from the released table. Statistics offices and tools such as ARX and sdcMicro still use them. Each definition applies to one table and assumes fixed adversarial knowledge; additional information or another release can defeat it.

## What the adversary believes about a class {#sec-anon-posterior}

To anonymize a table, the custodian—the organization holding the data—applies **generalization**, such as replacing an age with a range or a postcode with a prefix. Records with identical generalized quasi-identifiers form an **equivalence class**. An adversary who knows the target's quasi-identifiers can identify the target's class $E$ but cannot distinguish its records. The adversary's **posterior** for a sensitive value is therefore the fraction of the class with that value. Let $\lvert E\rvert$ be the class size and $n_E(v)$ the number of records in it with sensitive value $v$. The posterior that the target's sensitive value $S_{\text{target}}$ equals $v$ is

$$\Pr[S_{\text{target}} = v \mid \text{release}] = \frac{n_E(v)}{\lvert E\rvert}.$$ {#eq-anon-posterior}
An adversary can use background knowledge to eliminate values, then recompute the posterior over the remaining records. The attacks in this chapter use this calculation.

## k-anonymity and what it costs {#sec-anon-k}

**k-anonymity** (Sweeney and Samarati, 1998) requires each class to contain at least $k$ records, where the custodian chooses $k$. Generalization and **suppression**—deleting records that remain distinguishable—can enforce the requirement. In the previous chapter, an adversary who knows the target is in the release links it with probability $J/n$, where $J$ is the number of classes and $n$ the number of records. If each class contains at least $k$ records, there are at most $n/k$ classes, so
$$\Pr[\text{win}] = \frac{J}{n} \le \frac1k.$$ {#eq-anon-k-link}
This bound concerns linkage and assumes the adversary knows only the quasi-identifiers. The choice of $k$ is a policy decision; 5 is a common minimum.

Anonymizers such as **Mondrian** search for a low-cost generalization. Alternatively, generalize the table manually and recheck it. The script [practitioner_table_risk.py](../../code/part02-attacks/practitioner_table_risk.py) uses the Adult census extract and six quasi-identifiers from the previous chapter. It groups age into 10-year bands, native country into US or other, and education into four levels. It then counts records alone in their class and the distinct occupations in each class, treating occupation as the sensitive attribute. The script uses Polars because no maintained Python package computes k-anonymity or distinct $\ell$.

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

Generalization reduced the unique share from 14.64% to 0.94%, but the smallest class still contains one record. Suppress the 3.31% of records in classes smaller than 5 or use broader bands. The result remains fragile: an adversary who also knows hours worked finds 10.09% of records unique.

## Homogeneity, background knowledge and l-diversity {#sec-anon-l}

k-anonymity obscures record identity but not sensitive values (Machanavajjhala et al., 2006). In a **homogeneity attack**, all three records in Alice's class report heart disease, so her posterior probability of heart disease is 1. In a **background knowledge attack**, Bob's class contains two cancer records and one flu record. A neighbour who knows Bob did not have flu can eliminate that record and infer cancer.

**l-diversity** requires each class to contain enough distinct sensitive values. Its distinct version requires at least $\ell$ values but does not limit the most common value's share; a class can be 3-diverse even if one value dominates. The recursive version also caps that share using a custodian-selected constant $c$. If an adversary rules out $\ell-2$ values, the posterior is below $c/(1+c)$. At $c=1$, this bound is one half, so the guarantee prevents certainty but not a high posterior.

Suppressing small classes also removes most homogeneous classes. In Adult, 1.20% of generalized records are in a class where every record has the same occupation; only 0.03% are in such a class of at least 5 records. In the full edition's synthetic hospital table, distinct 3-diversity removed certain disclosures at little cost. The recursive version required some classes to grow to 641 patients.

## Measuring what a release teaches {#sec-anon-baseline}

Compare each posterior with the **imputation baseline** from [Attribute inference and model inversion](03-attribute-inference.md), the best population-only prediction of a sensitive value. Only improvement over this baseline reflects information about an individual. On the hospital table, age alone predicts each patient's diagnosis correctly for 53.6% of patients; a 5-anonymous release raises this to 59.3%. Most of the increase reflects the age-diagnosis association that the study reports. The additional 5.7 percentage points are leakage.

## Skewness, similarity and t-closeness {#sec-anon-t}

l-diversity does not prevent two attacks (Li, Li and Venkatasubramanian, 2007). In a **skewness attack**, a class that is half HIV-positive is 2-diverse, but if 1% of the full table is HIV-positive, membership in the class raises the posterior fifty-fold. In a **similarity attack**, a class containing three distinct stomach diseases is 3-diverse but still reveals that the person has a stomach disease.

**t-closeness** addresses both attacks by requiring the sensitive-value distribution in every class $E$ to be within distance $t$ of the full table's distribution. Let $P_E(A)$ and $P(A)$ be the probabilities that a value drawn from the class or full table lies in set $A$. Under **total variation (TV)** distance, no such probability—and therefore no posterior belief—changes by more than $t$:
$$\lvert P_E(A) - P(A)\rvert \le t.$$ {#eq-anon-tv}
The bound controls an absolute difference, not a ratio, so it gives weak protection for rare values. At $t = 0.15$, a value present in 1% of the table may reach 16% in a class. To address similarity attacks, Li and colleagues used earth mover's distance, which accounts for distances between values.

t-closeness reduces utility. At $t = 0.15$, a guess of diagnoses from the hospital release was correct for 22.2% of patients, below the 53.6% accuracy from age alone. The release therefore provided less information than age alone.

## The composition attack {#sec-anon-composition}

Ganta, Kasiviswanathan and Smith (2008) asked what happens when two hospitals each publish a k-anonymous or l-diverse table. A patient treated at both sits in a class $E_A$ of one release and a class $E_B$ of the other. The adversary keeps only the diagnoses that appear in both classes,
$$\{v : n_{E_A}(v) > 0\} \cap \{v : n_{E_B}(v) > 0\}.$$
When one candidate is left, the diagnosis is **pinned down**.

In the full edition's simulation, two hospitals share 6,649 patients. Together, their two 5-anonymous releases pin down the diagnosis of 9.9% of them, over three times what either pins alone. With distinct 3-diversity, neither release pins anyone, yet together they pin 18.

A single-table definition cannot prevent this attack because neither custodian checks the intersection of two releases. A robust definition must compose across releases, hold under arbitrary adversarial knowledge and bound belief ratios to protect rare values. [The definition of differential privacy](../part03-dp-foundations/01-definition.md) has these properties; statistical agencies such as the US Census Bureau have adopted it ([Reconstruction attacks](09-reconstruction.md)).

## Choosing and running an anonymization {#sec-anon-workflow}

Start with the quasi-identifiers described in [Linkage and re-identification](07-linkage-reidentification.md); an incomplete list is a common failure. Use k-anonymity to limit linkage. Add distinct l-diversity when a class containing one sensitive value is a disclosure. Use t-closeness when the sensitive attribute is skewed or contains similar values, because it can reduce utility substantially.

Audit the anonymizer's output in Polars as the excerpt does, or use pycanon, which reports $k$, $\ell$ and $t$ for a pandas table. [Pre-release privacy risk assessment](10-risk-assessment.md) uses these results to inform the release decision. Use differential privacy when other releases of the same people exist, the sensitive attribute is rare or the data are sparse. For counts and sums, [Release statistics with a library](../part03-dp-foundations/12-release-statistics.md) provides an end-to-end workflow.

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

- k-anonymity limits linkage but does not protect sensitive attributes; inspect the values in each class.
- Check for other releases of the same records; a guarantee for one table does not protect a person across releases.

## References

- Machanavajjhala and colleagues, "l-Diversity" (2006), and Li, Li & Venkatasubramanian, "t-Closeness" (2007): the attacks and the definitions that answer them.
- Ganta, Kasiviswanathan & Smith, "Composition Attacks and Auxiliary Information in Data Privacy" (2008): the intersection attack.

**Derivations.** Every result here is derived step by step in [the full edition's chapter](../../book-full/part02-attacks/08-syntactic-anonymity.md).
