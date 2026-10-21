# Syntactic anonymity and how it breaks

## Why this matters

[Linkage and re-identification](07-linkage-reidentification.md) showed that deleting names does not anonymize a table, because the remaining quasi-identifiers, such as age, ZIP code and sex, single out most people. Between 1998 and 2008, researchers answered with definitions that constrain the released table itself. They are called **syntactic** because you can check each one by reading the published table.

Statistics offices, clinical data sharing under the HIPAA expert-determination route, and tools such as ARX and sdcMicro still use these definitions, so you will meet them in audits. The short answer to whether they are enough is that each is a property of one table against an adversary whose knowledge is fixed in advance. One unanticipated fact, or a second release, voids the guarantee.

## What the adversary believes about a class {#sec-anon-posterior}

To anonymize a table, the custodian applies a **generalization**, which turns an age into a range, a ZIP code into a prefix, and so on. Records with the same generalized quasi-identifiers form an **equivalence class**, or class. An adversary who knows the target's quasi-identifiers can find the target's class $E$, but the records in it look alike. So if it knows nothing else, its belief about the target's sensitive value is the share of the class with that value.

Call that belief its **posterior**. Write $\lvert E\rvert$ for the size of the class, and $n_E(v)$ for the number of its records with a sensitive value $v$. Then
$$\Pr[S_{\text{target}} = v \mid \text{release}] = \frac{n_E(v)}{\lvert E\rvert} = P_E(v).$$ {#eq-anon-posterior}

Background knowledge works by elimination. Ruling out a value crosses out its records, and the remaining shares grow, which is the calculation behind every attack in this chapter.

## k-anonymity and what it costs {#sec-anon-k}

Sweeney and Samarati proposed **k-anonymity** in 1998. A release is k-anonymous if every class has at least $k$ records, for a crowd size $k$ the custodian chooses. Generalizing gets there, with **suppression** of the records that still stand out. The linkage success of [Linkage and re-identification](07-linkage-reidentification.md) is the number of classes $J$ over the number of records $n$, and it is then capped:
$$\Pr[\text{win}] = \frac{J}{n} \le \frac1k.$$ {#eq-anon-k-link}
The guarantee is about names only, and it assumes the adversary knows the quasi-identifiers and nothing more. In sparse data every column is a quasi-identifier, so k-anonymity destroys a ratings matrix (Aggarwal, 2005).

What $k$ to use is a policy choice that the mathematics does not settle, so data-sharing agreements fix it. What the mathematics does say is how coarse the quasi-identifiers must be. Counted in bits as in the linkage chapter, hiding 99% of people in classes of at least 2 allows about 9 bits fewer than making 99% unique.

Anonymizers search for a generalization that fits the data. For example, **Mondrian** keeps splitting groups of records at the median of one quasi-identifier, as long as both halves keep at least $k$ records. @tbl-anon-criteria gives what each criterion costs on the full edition's synthetic hospital table, where mean width is the average share of each quasi-identifier's range that a class covers.

| Criterion | Class sizes | Mean width | Certain disclosure | Guess accuracy |
| --- | --- | --- | --- | --- |
| $k = 5$ | 5–11 | 0.016 | 0.0232 | 0.593 |
| $k = 5$, distinct $\ell = 3$ | 5–33 | 0.022 | 0.0000 | 0.574 |
| $k = 5$, recursive $(2, 3)$ | 5–641 | 0.113 | 0.0000 | 0.417 |
| $k = 5$, $t = 0.15$ | 14–304 | 0.324 | — | 0.222 |

: Mondrian on a synthetic hospital table of 10,000 patients. Certain disclosure is the fraction of records in a single-diagnosis class, and guess accuracy the fraction whose diagnosis is the most common one in their class. Imputation from age alone guesses right for 0.536. {#tbl-anon-criteria}

At $k = 5$ a class covers only 1.6% of each range, so k-anonymity itself is cheap. The last two columns are about the diagnosis, which it does not protect.

## Homogeneity, background knowledge and l-diversity {#sec-anon-l}

Machanavajjhala and colleagues (2006) pointed out that k-anonymity hides who a record belongs to, but not what it says. If all three records of Alice's class say heart disease, her posterior is 1 without anyone finding her record, which is the **homogeneity attack**. If Bob's class says cancer, cancer and flu, a neighbour who knows he did not have flu learns he has cancer, which is the **background knowledge attack**. In the hospital table at $k = 5$, 2.3% of patients sit in single-diagnosis classes.

**l-diversity** asks each class to hold enough different values. Its distinct version asks for at least $\ell$ different values, and its entropy version asks for at least $\log_2 \ell$ bits of entropy. Neither caps the share of the most common value, and a class over eight values can be entropy 3-diverse with one value in 73% of its records.

The recursive version targets background knowledge directly. Sort a class's value counts from largest to smallest, and write $r_1$ for the largest count and $r_\ell$ for the $\ell$-th largest. A class is **recursive $(c, \ell)$-diverse**, for a constant $c$ the custodian picks, if
$$r_1 < c\,(r_\ell + r_{\ell+1} + \dots).$$
That condition caps the posterior even after the adversary rules out $\ell - 2$ values:
$$\Pr[S_{\text{target}} = v \mid \text{release}, \text{background}] < \frac{c}{1 + c}.$$ {#eq-anon-recursive}

So l-diversity protects against certainty, not likelihood, since the cap is already one half at $c = 1$. On the hospital table, distinct 3-diversity removes the certain disclosures at little cost, while recursive diversity forces some classes to hold hundreds of patients.

## Measuring what a release teaches {#sec-anon-baseline}

Compare every posterior with what the adversary would believe with no release, not with zero. The comparison is the **imputation baseline** of [Attribute inference and model inversion](03-attribute-inference.md), the most likely sensitive value given the target's own quasi-identifiers under the population distribution. Only success beyond it is about the individual.

On the hospital table, guessing the most likely diagnosis for each patient's age, with no release, is right for 53.6%, and a 5-anonymous release is right for 59.3%. So most of what the release teaches is the link between age and diagnosis, which any useful study would report, and only the excess of 0.057 is a leak.

## Skewness, similarity and t-closeness {#sec-anon-t}

Li, Li and Venkatasubramanian (2007) found two attacks that l-diversity misses. In the **skewness attack**, a class that is half HIV is 2-diverse, yet in a table with 1% HIV it raises its members' posterior fifty-fold. In the **similarity attack**, a class of three stomach diseases is 3-diverse but reveals a stomach disease. **t-closeness** answers both. A release is t-close if every class's distribution is within a distance $t$ of the whole table's.

With the **total variation (TV)** distance ([Divergences](../part01-prerequisites/03-divergences.md)), no belief moves by more than $t$. Write $P_E(A)$ and $P(A)$ for the chance that a value from the class, or from the table, lands in a set $A$. Then
$$\lvert P_E(A) - P(A)\rvert \le t.$$ {#eq-anon-tv}
The bound is additive, which protects rare values badly. A value held by 1% of the table may rise to 16% at $t = 0.15$. Total variation also treats all diagnoses as equally different, so against the similarity attack Li and colleagues used the **earth mover's distance (EMD)**, under which similar values are close.

t-closeness also fights the purpose of the release, because at $t = 0.15$ the hospital table's guess accuracy falls far below what imputation from age achieves.

## The composition attack {#sec-anon-composition}

Ganta, Kasiviswanathan and Smith (2008) asked what happens with two releases. Suppose that two hospitals in one city each publish a properly k-anonymous or $\ell$-diverse table, and a patient was treated at both. Write $E_A$ and $E_B$ for the patient's classes in the two releases. The adversary keeps only the diagnoses that appear in both,
$$\{v : n_{E_A}(v) > 0\} \cap \{v : n_{E_B}(v) > 0\}.$$
When one candidate is left, the diagnosis is **pinned down**.

The full edition simulates two hospitals that each publish a 5-anonymous release. Together, the releases pin down the diagnosis of 9.9% of their shared patients, more than three times what either pins alone. With distinct 3-diversity neither release pins anyone, yet together they still pin 18 patients.

No single-table definition can stop this, because the intersection of two tables is a table that neither custodian checked. A definition that survives must compose, hold whatever the adversary knows, and bound the ratio of beliefs to protect rare values. [The definition of differential privacy](../part03-dp-foundations/01-definition.md) starts from these requirements, and that is why regulators and the Census Bureau have moved towards it ([Reconstruction attacks](09-reconstruction.md)).

## Choosing and running an anonymization {#sec-anon-workflow}

1. **Fix the quasi-identifiers** with the risk check of [Linkage and re-identification](07-linkage-reidentification.md). Choosing them too narrowly is the commonest failure, since every column an adversary could know counts.
2. **Choose the criteria.** Use k-anonymity against linkage, add distinct l-diversity when a single-diagnosis class would be a disclosure, and use t-closeness only when the sensitive attribute is skewed or has similar values, since it costs the most.
3. **Run the anonymizer.** ARX, an open-source Java tool, takes a generalization hierarchy for each quasi-identifier, the privacy models above and a suppression limit, and searches for the generalization that loses the least information.
4. **Audit the output yourself.** In pandas, `df.groupby(qi_cols)[sensitive].agg(["size", "nunique"])` gives each class's size and number of distinct values, and the Code section computes entropy $\ell$ and distance from the table. The Python package pycanon also reports $k$, $\ell$ and $t$ for a table. Report guess accuracy next to the imputation baseline.
5. **Move to differential privacy** when overlapping views or other releases of the same people exist, when the sensitive attribute is rare, or when the data are sparse.

## Code

The snippet audits a 3-anonymous release and prints the smallest class size, the fewest distinct values, and each class's entropy-equivalent $\ell$ and distance from the table.

```python
from collections import Counter, defaultdict
from math import log2


def classes(rows: list[tuple[str, str, str]]) -> list[Counter[str]]:
    """Count the diagnoses in each class of (ZIP, age, diagnosis) rows."""
    groups: defaultdict[tuple[str, str], Counter[str]] = defaultdict(Counter)
    for zip_code, age, diagnosis in rows:
        groups[(zip_code, age)][diagnosis] += 1
    return list(groups.values())


def entropy_l(counts: Counter[str]) -> float:
    """The number of equally common values with the same entropy."""
    total = sum(counts.values())
    return 2 ** -sum(c / total * log2(c / total) for c in counts.values())


def tv_to_table(counts: Counter[str], table: Counter[str]) -> float:
    """Total variation between a class's distribution and the whole table's."""
    nc, nt = sum(counts.values()), sum(table.values())
    return sum(abs(counts[v] / nc - table[v] / nt) for v in table) / 2


rows = [
    ("130**", "<30", "heart disease"), ("130**", "<30", "heart disease"), ("130**", "<30", "heart disease"),
    ("148**", "4*", "cancer"), ("148**", "4*", "flu"), ("148**", "4*", "heart disease"),
    ("130**", "3*", "cancer"), ("130**", "3*", "cancer"), ("130**", "3*", "flu"),
]
table = Counter(diagnosis for _, _, diagnosis in rows)
found = classes(rows)
print(min(sum(e.values()) for e in found))  # 3
print(min(len(e) for e in found))  # 1
print(" ".join(f"{entropy_l(e):.2f}" for e in found))  # 1.00 3.00 1.89
print(" ".join(f"{tv_to_table(e, table):.3f}" for e in found))  # 0.556 0.111 0.444
```

The full edition's script, [syntactic_anonymity.py](../../code/part02-attacks/syntactic_anonymity.py), implements Mondrian and the attacks.

## Cheat sheet

| Definition or attack | Condition | What it gives, and where it ends |
| --- | --- | --- |
| k-anonymity | Every class has at least $k$ records | Linkage at most $1/k$; nothing about the sensitive attribute |
| Distinct, entropy l-diversity | At least $\ell$ values, or entropy at least $\log_2 \ell$ | No cap on the top value's share |
| Recursive $(c, \ell)$-diversity | $r_1 < c(r_\ell + r_{\ell+1} + \dots)$ | Posterior below $c/(1 + c)$ after ruling out $\ell - 2$ values |
| t-closeness (TV) | Each class within $t$ of the table | Every belief moves by at most $t$, so a 1% value may reach $1\% + t$ |
| Composition | Intersect the target's classes in two releases | No single-table definition prevents it |

: The syntactic definitions and the attacks on them. {#tbl-anon-cheat}

**Rules of thumb**

- k-anonymity protects names, not attributes, so check what each class's sensitive values reveal.
- Measure attribute disclosure against imputation from the population, or you will count the study's findings as leaks.
- A rare sensitive value needs a bound on the ratio of beliefs, not on their difference.
- Assume another release exists, because a guarantee about one table is not a guarantee about a person.

## Further reading

- Machanavajjhala and colleagues, "l-Diversity" (2006), and Li, Li & Venkatasubramanian, "t-Closeness" (2007).
- Ganta, Kasiviswanathan & Smith, "Composition Attacks and Auxiliary Information in Data Privacy" (2008): the intersection attack.

**Derivations.** Every result here is derived step by step in [the full edition's chapter](../../book-full/part02-attacks/08-syntactic-anonymity.md).
