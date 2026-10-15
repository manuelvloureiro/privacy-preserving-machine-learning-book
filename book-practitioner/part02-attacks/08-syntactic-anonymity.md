# Syntactic anonymity and how it breaks

## Why this matters

[Linkage and re-identification](07-linkage-reidentification.md) showed that deleting names does not anonymize a table: the remaining **quasi-identifiers**, such as age, ZIP code and sex, single out most people. Between 1998 and 2008 a family of definitions answered by constraining the released table itself, each one proposed to stop an attack on the one before. **k-anonymity** hides every record in a crowd of at least $k$ look-alikes. **l-diversity** asks each crowd to hold several different sensitive values. **t-closeness** asks each crowd to look like the whole table. They are **syntactic**: each is a property of the published table that you can check by reading it, with no reference to how it was produced or what else the adversary knows. Statistics offices, regulators' guidance and tools such as ARX still use them.

What they share is also what breaks them: each fixes the adversary's knowledge in advance. k-anonymity assumes the adversary knows the quasi-identifiers and nothing else, l-diversity allows a few facts about the target, t-closeness the table's overall distribution. Each attack in this chapter hands the adversary one piece of knowledge the definition did not anticipate, and the guarantee then says nothing. A second release, the most ordinary auxiliary information there is, breaks all three. Those failures are the specification for differential privacy.

One fact drives every attack below. Records are grouped into **equivalence classes**, sets of records with the same (generalized) quasi-identifiers, and each record has one **sensitive attribute** $S$, such as a diagnosis. An adversary who knows the target's quasi-identifiers can find the target's class $E$ but not its record, so the best it can believe, its **posterior**, is that the target looks like a random member of the class:
$$\Pr[S_{\text{target}} = v \mid \text{release}] = \frac{n_E(v)}{\lvert E\rvert} = P_E(v),$$ {#eq-anon-posterior}
where $n_E(v)$ counts the records of $E$ with value $v$. If the adversary also knows the target's value is not $u$, it crosses out the records with $u$ and renormalizes over the rest.

## k-anonymity {#sec-anon-k}

Sweeney and Samarati's answer to linkage was to make sure no combination of quasi-identifiers singles out fewer than $k$ people. A release is **k-anonymous** if every equivalence class has at least $k$ records. A custodian gets there by **generalization** (age 37 becomes 35–39, ZIP code 02139 becomes 021\*\*) and by **suppression**, deleting the few records that still stand out. What this buys follows from the linkage game: an adversary who finds the target's class must guess among its records, so each class yields one record's worth of wins, and the success is the number of classes $J$ over the number of records $n$. Classes of at least $k$ records cannot number more than $n/k$, so
$$\Pr[\text{win}] = \frac{J}{n} \le \frac{1}{k}.$$ {#eq-anon-k-link}

The price is precision, and it is larger than it looks. Cells of the generalized quasi-identifiers vary in size around their average, and the unlucky small ones violate the definition, so the average must be well above $k$. For 10,000 people, keeping 99% of records in classes of at least 2 needs 4.61 people per cell on average, more than twice $k$, which allows the quasi-identifiers 11.08 bits, 2.20 below $\log_2 n$. Since 99% of people are unique 6.64 bits above it, the window between "almost everyone is unique" and "almost everyone is hidden" is only about 9 bits wide, and every column you publish spends some of it.

The **Mondrian** algorithm (LeFevre, DeWitt & Ramakrishnan, 2006) starts from the whole table as one box and repeatedly splits a box at the median of its widest quasi-identifier, as long as both halves keep $k$ records. Because it recodes locally, classes stay close to size $k$. @tbl-anon-k applies it to a synthetic hospital table of 10,000 patients, with age, ZIP code and sex as quasi-identifiers and one of eight diagnoses that depend on age. With exact quasi-identifiers the linkage success is 0.933.

| $k$ | Classes | Sizes | Mean width | Linkage $J/n$ | Certain disclosure | Guess accuracy |
| --- | --- | --- | --- | --- | --- | --- |
| 2 | 4,091 | 2–5 | 0.005 | 0.4091 | 0.2797 | 0.675 |
| 5 | 1,542 | 5–11 | 0.016 | 0.1542 | 0.0232 | 0.593 |
| 10 | 748 | 10–20 | 0.028 | 0.0748 | 0.0000 | 0.562 |
| 20 | 366 | 20–41 | 0.043 | 0.0366 | 0.0000 | 0.540 |
| 50 | 129 | 50–99 | 0.075 | 0.0129 | 0.0000 | 0.506 |

: Mondrian k-anonymization of a synthetic hospital table of 10,000 patients. Mean width is the class's range as a fraction of each quasi-identifier's full range. Certain disclosure is the fraction of records in a class with a single diagnosis. Guess accuracy is the fraction of records whose diagnosis is the most common one in their class. {#tbl-anon-k}

Linkage stays below $1/k$ at little cost in precision: at $k = 5$ a class's range covers 1.6% of each quasi-identifier's span on average. But the guarantee bounds the chance of picking out the target's record, not of learning the diagnosis, which is the point of the release. At $k = 5$, 2.3% of patients sit in a class with a single diagnosis, and the release states their diagnosis outright, no linkage needed.

## Homogeneity, background knowledge and l-diversity {#sec-anon-l}

Machanavajjhala and colleagues (2006) made the gap concrete. @tbl-anon-toy is 3-anonymous.

| ZIP code | Age | Diagnosis |
| --- | --- | --- |
| 130\*\* | < 30 | heart disease |
| 130\*\* | < 30 | heart disease |
| 130\*\* | < 30 | heart disease |
| 148\*\* | 4\* | cancer |
| 148\*\* | 4\* | flu |
| 148\*\* | 4\* | heart disease |
| 130\*\* | 3\* | cancer |
| 130\*\* | 3\* | cancer |
| 130\*\* | 3\* | flu |

: A 3-anonymous release with three classes of three records. {#tbl-anon-toy}

Alice, 26, is in the first class, where every record says heart disease, so by @eq-anon-posterior her posterior is 1: the **homogeneity attack**, full disclosure without re-identification. Bob, 35, is in the third class, with posterior $\frac23$ for cancer. His neighbour knows he did not have flu, crosses out the flu record, and cancer has posterior 1: the **background knowledge attack**. Both work because one value dominates the class, so l-diversity asks each class for enough different values.

Its three versions differ in what they bound. A class is **distinct $\ell$-diverse** if it holds at least $\ell$ different values, and **entropy $\ell$-diverse** if its distribution is as spread out as $\ell$ equally common values (entropy at least $\log_2 \ell$). Neither limits the share of the most common value, so neither caps the posterior: an entropy 3-diverse class over 8 values can have one value in 73% of its records, because the other 27%, spread over seven values, add enough entropy. The third version targets the background knowledge attack directly. Sort a class's value counts as $r_1 \ge r_2 \ge \dots \ge r_m$; the class is **recursive $(c, \ell)$-diverse** if
$$r_1 < c\,(r_\ell + \dots + r_m).$$
The right side is the tail left after setting aside the $\ell - 1$ most common values. An adversary who rules out $\ell - 2$ values cannot make the top count larger, and whatever it removes, the values that remain besides the top one add up to at least that tail. So the top value still holds less than $c$ parts out of $c + 1$. Every remaining value therefore keeps a posterior below $c/(1 + c)$: 0.500, 0.667 and 0.750 for $c = 1, 2, 3$. Even that cap lets the adversary be fairly sure. l-diversity protects against certainty, not likelihood, and only against an adversary who rules out at most $\ell - 2$ values; one who knows something softer ("probably not flu") is not covered.

To judge what a release teaches, compare with what the adversary would believe without it. On the hospital table (@tbl-anon-l), the most common diagnosis overall is right for 18.5% of patients, but **imputation**, guessing the most likely diagnosis for each patient's age from the population, is right for 53.6%. So most of the 59.3% that a 5-anonymous release achieves is the age–diagnosis pattern any study would report, a fact about the population rather than a secret about anyone. Measure attribute disclosure against imputation, or you will count the study's findings as leaks.

| Criterion | Classes | Sizes | Mean width | Linkage $J/n$ | Certain disclosure | Guess accuracy |
| --- | --- | --- | --- | --- | --- | --- |
| $k = 5$ | 1,542 | 5–11 | 0.016 | 0.1542 | 0.0232 | 0.593 |
| $k = 5$, distinct $\ell = 3$ | 1,132 | 5–33 | 0.022 | 0.1132 | 0.0000 | 0.574 |
| $k = 5$, recursive $(c = 2, \ell = 3)$ | 389 | 5–641 | 0.113 | 0.0389 | 0.0000 | 0.417 |

: Mondrian with l-diversity on the same hospital table: each split is allowed only if both halves satisfy the criterion. {#tbl-anon-l}

Against that baseline, distinct 3-diversity removes certain disclosure at almost no cost. Recursive $(2, 3)$-diversity forces some classes to hold hundreds of patients and pushes accuracy below imputation: it has started destroying the pattern the release exists to show.

## Skewness, similarity and t-closeness {#sec-anon-t}

Li, Li and Venkatasubramanian (2007) found two attacks that diversity does not see. In the **skewness attack**, 1% of the table has HIV, and a class that is half HIV is 2-diverse by every definition, yet it raises the belief that a member has HIV from 1% to 50% (@fig-anon-skew). In the **similarity attack**, a class holding gastric ulcer, gastritis and stomach cancer is 3-diverse and still says "stomach disease". So counting values is the wrong yardstick: what matters is how far a class moves the adversary from what it would believe anyway, the whole table's distribution $P$.

![The skewness attack: a table in which 1% of patients have HIV, and a class of it that is half HIV. The class is 2-diverse but at total variation 0.49 from the table.](figures/anon_skew.svg){#fig-anon-skew}

A release is **t-close** if every class's distribution is within distance $t$ of the table's, $d(P_E, P) \le t$. With the **total variation (TV)** distance, $\mathrm{TV}(P, Q) = \frac12\sum_v\lvert P(v) - Q(v)\rvert$, the guarantee has a direct reading. TV is the largest gap between the two distributions' probabilities of any set of values, so every question the adversary can ask about the target's value gets an answer within $t$ of the table's:
$$\lvert P_E(A) - P(A)\rvert \le t \quad \text{for every set of values } A.$$ {#eq-anon-tv}
In particular, guessing the class's top value is right with probability at most $\max_v P(v) + t$, and on the hospital table (@tbl-anon-t) the accuracy stays under that bound.

| Criterion | Classes | Sizes | Mean width | Largest TV | Guess accuracy | Bound $\max_vP(v) + t$ |
| --- | --- | --- | --- | --- | --- | --- |
| $k = 5$ | 1,542 | 5–11 | 0.016 | 0.937 | 0.593 | — |
| $k = 5$, $t = 0.3$ | 495 | 6–60 | 0.291 | 0.300 | 0.275 | 0.485 |
| $k = 5$, $t = 0.15$ | 136 | 14–304 | 0.324 | 0.149 | 0.222 | 0.335 |

: Mondrian with t-closeness (total variation) on the hospital table. The table's most common diagnosis has frequency 0.185. {#tbl-anon-t}

The bound is additive, and that is its weakness. A difference of 0.15 is small next to a value held by 40% of patients and enormous next to one held by 1%: at $t = 0.15$ a 1% value may rise to 16%, a sixteen-fold increase for exactly the people who most need protection. The half-HIV class sits at TV 0.49; a class with 10% HIV sits at 0.09, passes $t = 0.1$, and still makes HIV ten times as likely. A guarantee for rare values has to bound the ratio of beliefs, not their difference.

t-closeness also fights the purpose of the release. If age predicts diagnosis, a class of similar ages has a different diagnosis distribution from the whole table, so making every class close to the table means mixing ages until the pattern is gone. In @tbl-anon-t the classes span about 30% of each quasi-identifier's range, and guess accuracy falls to 0.222–0.275, far below the 0.536 of imputation. The release now hides a population fact that was never a secret.

The distance also decides what counts as similar, and TV, which treats all values as equally different, misses the similarity attack. Li et al. use the **earth mover's distance (EMD)**, the least work to reshape one distribution into the other when moving mass between two values costs their distance. With a ground distance that puts gastritis near gastric ulcer, a class full of stomach diseases is far from the table and fails. For $m$ ordered values (salary bands, age groups) at spacing $1/(m - 1)$, the work is the net mass that has to cross each gap between neighbouring values:
$$\mathrm{EMD}(P_E, P) = \frac{1}{m - 1}\sum_{i=1}^{m-1}\Big\lvert\sum_{j \le i}\big(P_E(j) - P(j)\big)\Big\rvert.$$ {#eq-anon-emd}

## The composition attack {#sec-anon-composition}

Every definition so far concerns one table. Ganta, Kasiviswanathan and Smith (2008) asked about two. Two hospitals in one city publish their tables, each properly k-anonymous or $\ell$-diverse, without coordinating. A patient treated at both sits in a class of each, and the true diagnosis is in both classes, because the patient's own record carries it. So the adversary intersects the two sets of values, $\{v : n_{E_A}(v) > 0\} \cap \{v : n_{E_B}(v) > 0\}$, and the diagnosis is **pinned down** when one candidate is left. Neither hospital did anything wrong.

How often this works depends on class size, for a clear reason: every other member of a class brings a wrong value, and each wrong value is another chance for the two classes to share one. In a model where the other members are drawn from the population, the probability of pinning is 0.8734 for classes of 2 and 0.1615 for classes of 5, falling to 0.0015 for classes of 10. On data, take a city of 15,000 people in which two hospitals each treat a random 10,000, so 6,649 are treated at both, and each publishes a 5-anonymous Mondrian release (@tbl-anon-composition).

| Releases | Pinned by A alone | Pinned by B alone | Pinned by A and B | Mean candidates, A alone | Mean candidates, A and B |
| --- | --- | --- | --- | --- | --- |
| $k = 5$ | 0.0289 | 0.0262 | 0.0985 | 2.79 | 2.38 |
| $k = 5$, distinct $\ell = 3$ | 0.0000 | 0.0000 | 0.0027 | 3.22 | 2.92 |

: The composition attack on two independent hospital releases, for the 6,649 people treated at both. "Pinned" is the fraction left with a single possible diagnosis. {#tbl-anon-composition}

Together, two 5-anonymous releases pin down the diagnosis of 9.9% of shared patients, more than three times either alone. With distinct 3-diversity neither release pins anyone, and together they pin 18 patients. The model understates the risk: a patient's class here holds 6.8 records on average, and the model predicts only 2.9% pinned for classes of 7, because real classes group people of similar age, who share diagnoses more often than random draws.

No custodian can prevent this by checking its own table harder, since it cannot know every other release its patients appear in. A better definition must therefore do three things. It has to hold whatever the adversary knows, since another release is just auxiliary information. It has to say how leakage adds up when releases are combined. And it has to bound the ratio of beliefs with and without a person's data, so that rare values are protected. [The definition of differential privacy](../part03-dp-foundations/01-definition.md) starts from these three requirements.

## Code

An auditor can check every definition of this chapter by grouping the rows. The snippet audits @tbl-anon-toy: smallest class, fewest distinct values, entropy-equivalent $\ell$, recursive $(2, 2)$-diversity and TV distance from the table.

```python
from collections import Counter, defaultdict
from math import log2


def classes(rows: list[tuple[str, str, str]]) -> list[Counter[str]]:
    """Group (ZIP, age, diagnosis) rows by their quasi-identifiers; count the diagnoses in each class."""
    groups: defaultdict[tuple[str, str], Counter[str]] = defaultdict(Counter)
    for zip_code, age, diagnosis in rows:
        groups[(zip_code, age)][diagnosis] += 1
    return list(groups.values())


def entropy_l(counts: Counter[str]) -> float:
    """2 ** entropy: the number of equally common values with the same spread."""
    total = sum(counts.values())
    return 2 ** -sum(c / total * log2(c / total) for c in counts.values())


def recursive_diverse(counts: Counter[str], c: float, l: int) -> bool:
    """Recursive (c, l)-diversity: the top count is below c times the counts from the l-th largest on."""
    r = sorted(counts.values(), reverse=True)
    return r[0] < c * sum(r[l - 1 :])


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
print(all(recursive_diverse(e, 2, 2) for e in found))  # False
print(" ".join(f"{tv_to_table(e, table):.3f}" for e in found))  # 0.556 0.111 0.444
```

The full edition's script, [syntactic_anonymity.py](../../code/part02-attacks/syntactic_anonymity.py), implements Mondrian with each criterion and runs every attack of this chapter:

```bash
uv run python code/part02-attacks/syntactic_anonymity.py
```

## Cheat sheet

| Definition or attack | Condition | What it gives, and where it ends |
| --- | --- | --- |
| Posterior in a class | $P_E(v) = n_E(v)/\lvert E\rvert$ | Ruling out values renormalizes over the rest |
| k-anonymity | Every $\lvert E\rvert \ge k$ | Linkage $J/n \le 1/k$; nothing about the sensitive attribute |
| Distinct, entropy l-diversity | $\ge \ell$ values; entropy $\ge \log_2 \ell$ | No cap on the top value's share |
| Recursive $(c, \ell)$-diversity | $r_1 < c(r_\ell + \dots + r_m)$ | Posterior $< c/(1 + c)$ after ruling out $\ell - 2$ values |
| t-closeness (TV) | $\mathrm{TV}(P_E, P) \le t$ | Every belief moves by $\le t$; a 1% value may rise to $1\% + t$ |
| t-closeness (EMD) | $\mathrm{EMD}(P_E, P) \le t$ | Similar values are close; catches the similarity attack |
| Composition | Intersect the values of the target's classes | Breaks all three; no single-table check prevents it |

: The syntactic definitions and the attacks on them. {#tbl-anon-cheat}

**Rules of thumb**

- Of a "k-anonymous" release, ask what each class's sensitive values reveal, not just how big it is.
- Measure attribute disclosure against imputation from the population.
- Rare values need a bound on the ratio of beliefs, not their difference.
- Assume another release exists.

## Further reading

- Sweeney, "k-Anonymity: A Model for Protecting Privacy" (2002): the definition, generalization and suppression.
- Machanavajjhala, Kifer, Gehrke & Venkitasubramaniam, "l-Diversity: Privacy Beyond k-Anonymity" (2007) and Li, Li & Venkatasubramanian, "t-Closeness: Privacy Beyond k-Anonymity and l-Diversity" (ICDE, 2007): the attacks on the sensitive attribute and the definitions that answer them.
- Ganta, Kasiviswanathan & Smith, "Composition Attacks and Auxiliary Information in Data Privacy" (KDD, 2008): the intersection attack and why a definition must survive auxiliary information.
- Article 29 Working Party, "Opinion 05/2014 on Anonymisation Techniques": how European regulators judge these techniques against singling out, linkability and inference.

**Derivations.** Every result in this chapter is derived step by step in [the full edition's chapter](../../book-full/part02-attacks/08-syntactic-anonymity.md).
