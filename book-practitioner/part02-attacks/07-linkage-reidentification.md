# Linkage and re-identification

## Why this matters

A hospital shares its discharge records with researchers after deleting names and addresses, keeping postcode, date of birth, sex and diagnosis. Is anyone identifiable? Usually, yes. Most people are the only person with their postcode, birth date and sex, so anyone holding a table with names and those three columns, such as a voter roll, can join the two and put a name back on most rows. This is a **linkage attack**, and naming a "de-identified" record is **re-identification**. It needs only the release and some auxiliary data. If you sign off on data sharing or audit a release, it is the first attack to run.

The one idea is that identifying information is measured in bits, and bits add up. Picking one person out of $n$ takes $\log_2 n$ bits: 27.9 bits for the 248.7 million people in the United States in 1990, 32.9 for 8 billion. An attribute with $v$ equally likely values is worth at most $\log_2 v$ bits. **Once the bits an adversary knows exceed $\log_2 n$ by a few, almost everyone is unique.** @tbl-link-uses maps the questions an auditor asks to the sections that answer them.

| Question | What to compute | Section |
| --- | --- | --- |
| How often does a join name the right record? | Distinct quasi-identifier combinations over records, $J/n$ | @sec-link-quasi |
| How many people are unique on these columns? | The bits of the columns against $\log_2 n$, or the exact expected fraction | @sec-link-unique |
| The data are sparse (ratings, baskets, locations) | A rarity-weighted score and a calibrated match threshold | @sec-link-sparse |
| We release only a sample. Does that protect? | The probability that a match in the sample is correct | @sec-link-sampling |

: The questions behind a re-identification audit, and where this chapter answers each. {#tbl-link-uses}

## Quasi-identifiers and the linkage attack {#sec-link-quasi}

**Idea.** **Direct identifiers** name a person on their own: name, email address. **Sensitive attributes** are what the release is about, such as a diagnosis. **Quasi-identifiers** are in between: columns that name nobody alone but single people out in combination, and that appear in other datasets next to names, like ZIP code, birth date and sex. Deleting direct identifiers and keeping quasi-identifiers is the most common "de-identification", and linkage breaks it. Group the records of a release into **equivalence classes**, sets of records with identical quasi-identifiers. A target alone in its class is found for sure; a target in a class of $k$ look-alikes is found one time in $k$.

**Result.** Against an adversary who knows the target is in the release and knows its quasi-identifiers, the best possible success is
$$\Pr[\text{win}] = \frac{J}{n},$$ {#eq-link-success}
the number $J$ of distinct quasi-identifier combinations divided by the number $n$ of records. Each class, however large, contributes one record's worth of wins, so the fraction of unique records is a floor on the success, and a release whose classes all have at least $k$ records keeps it at most $1/k$.

**Example.** @tbl-link-release is a de-identified release and @tbl-link-voters part of a voter roll.

| ZIP code | Birth date | Sex | Diagnosis |
| --- | --- | --- | --- |
| 02138 | 1961-03-14 | F | asthma |
| 02138 | 1972-11-02 | M | diabetes |
| 02138 | 1972-11-02 | M | hypertension |
| 02139 | 1958-06-21 | F | depression |
| 02139 | 1990-01-01 | F | migraine |
| 02139 | 1990-01-01 | F | asthma |
| 02139 | 1990-01-01 | F | fracture |
| 02141 | 1983-09-30 | M | pneumonia |

: A de-identified hospital release: no names, three quasi-identifiers, one sensitive attribute. {#tbl-link-release}

| Name | ZIP code | Birth date | Sex |
| --- | --- | --- | --- |
| Ana Silva | 02138 | 1961-03-14 | F |
| Ben Cole | 02138 | 1972-11-02 | M |
| Carla Diaz | 02139 | 1958-06-21 | F |
| Dan Eto | 02141 | 1983-09-30 | M |

: Part of a voter roll: names with the same three quasi-identifiers. {#tbl-link-voters}

The join gives Ana's asthma, Carla's depression and Dan's pneumonia with certainty. Ben matches two records, so the adversary is right half the time. The classes have sizes 1, 1, 1, 2 and 3, so $J = 5$ for $n = 8$ and the success is $5/8 = 0.625$. This is how Sweeney re-identified the governor of Massachusetts in the late 1990s. From 1990 census tables she estimated that 87% of Americans were likely unique on {5-digit ZIP code, date of birth, sex}, 53% on {city, date of birth, sex} and 18% on {county, date of birth, sex}.

**Pitfalls.** Which columns are quasi-identifiers is a judgement about what auxiliary data exist, and in sparse data every column is one (@sec-link-sparse). @eq-link-success counts only record linkage: if a whole class shares one diagnosis, the adversary learns it without picking a record, the gap studied in [Syntactic anonymity and how it breaks](08-syntactic-anonymity.md). And without knowing the target is in the release, a unique match may belong to someone outside it (@sec-link-sampling).

## How many people are unique? {#sec-link-unique}

**Idea.** To predict uniqueness for columns you have not released yet, use a model of the population. Call a person's combination of attribute values their **cell**, and suppose each person falls in cell $c$ with probability $p_c$, independently. A person is unique when every one of the other $n - 1$ people avoids their cell.

**Result.** The expected fraction of unique people is
$$\frac{\mathbb E U}{n} = \sum_c p_c(1 - p_c)^{n-1}.$$ {#eq-link-unique}
If all $M$ cells are equally likely, with $b = \log_2 M$ bits, this is about $\exp(-2^{\log_2 n - b})$: one curve in the bits beyond $\log_2 n$, the same for a town and for the planet (@fig-link-uniqueness, right). Half the people are unique once $b$ exceeds $\log_2 n$ by 0.53 bits, 90% at 3.25 bits more, and 99% at 6.64 bits more. The **collision probability** $\sum_c p_c^2$, the chance that two random people share a cell, gives a floor, $\mathbb E U/n \ge (1 - \sum_c p_c^2)^{n-1}$; for independent attributes it is a product over attributes, so the per-attribute frequencies suffice.

**Example.** Sweeney's three attributes carry at most $15.3 + 14.8 + 1.0 = 31.1$ bits (about 40,000 ZIP codes, 29,200 birthdays over 80 years, two sexes) against the 27.9 bits that single out one American. That is $M = 2.34 \times 10^9$ cells for 248.7 million people, $n/M = 0.106$, and a uniform-cells uniqueness of $e^{-0.106} = 0.899$. Sweeney's 87% is lower because people crowd into popular cells. @tbl-link-census reveals the attributes of a synthetic census of 10,000 people one at a time ($\log_2 n = 13.29$ bits).

| Attributes known | Bits $H$ | Exact | Collision bound | Uniform cells |
| --- | --- | --- | --- | --- |
| sex, age band, region | 9.31 | 0.0158 | 0.0000 | 0.0004 |
| + birth month | 12.90 | 0.3328 | 0.0834 | 0.5215 |
| + marital status | 14.60 | 0.5789 | 0.4237 | 0.8498 |
| + children | 16.48 | 0.7939 | 0.7443 | 0.9732 |
| + occupation | 20.39 | 0.9747 | 0.9736 | 0.9986 |
| + blood group | 22.57 | 0.9925 | 0.9924 | 0.9998 |

: Fraction of 10,000 people who are unique as attributes are revealed: the exact expectation, the collision floor, and the uniform-cells value for the same number of cells. $H$ is the entropy of the attributes known so far. {#tbl-link-census}

With 12.90 bits, just under $\log_2 n$, a third of people are unique; with 20.39 bits, 7.1 beyond it, 97.5% are.

**Pitfalls.** The often-quoted "33 bits identify anyone" is where the average cell holds one person, not where everyone is unique: with $2^{33}$ uniform cells for 8 billion people only a fraction 0.394 is unique, and 99% needs 39.5 bits. For correlated attributes (ZIP code and income) the sum of bits is only an upper bound. For skewed attributes the uniform model usually overstates uniqueness, by up to 0.27 in @tbl-link-census, and the collision floor understates it when a few cells hold most people. Compute @eq-link-unique from a model of the joint distribution when you can, such as the Gaussian copula of Rocher, Hendrickx and de Montjoye (2019).

![Left: the fraction of people who are unique in the synthetic census as attributes are added, exact against one simulated population, with the collision floor and an entropy-based rule of thumb. Right: with uniform cells, the fraction unique depends only on the bits beyond $\log_2 n$; the curves for $n = 10^4$, $10^6$ and $8\times10^9$ coincide.](figures/link_uniqueness.svg){#fig-link-uniqueness}

## Sparse, high-dimensional data: the Netflix attack {#sec-link-sparse}

**Idea.** In 2006 Netflix released about 100 million ratings by 480,189 subscribers, names replaced by random numbers and some ratings and dates perturbed. There are no columns to join on, and the adversary's knowledge ("she watched these films around these dates") is partial and noisy. Narayanan and Shmatikov (2008) showed that this does not help: most items are rated by few people, so each is worth many bits, and a few approximate facts match one record far better than any other. Score every record by how well it agrees, item by item, giving rare items more weight.

**Result.** For auxiliary information $\mathrm{aux}$, a few items $i$ with approximate values,
$$\mathrm{Score}(\mathrm{aux}, r) = \sum_{i \in \mathrm{aux}} w_i\,\mathrm{Sim}(\mathrm{aux}_i, r_i),$$ {#eq-link-score}
where $\mathrm{Sim}$ is 1 if record $r$ agrees on item $i$ (say, within one star) and 0 otherwise. Narayanan and Shmatikov use $w_i = 1/\ln\lvert\mathrm{supp}(i)\rvert$, with $\lvert\mathrm{supp}(i)\rvert$ the number of records that rated item $i$. The likelihood ratio of "this record is the target" against "unrelated" gives $w_i \approx \ln(1/s_i) + \ln\frac{1 - \varepsilon}{\varepsilon}$, where $s_i$ is the chance that an unrelated record agrees on item $i$ and $\varepsilon$ the chance that the auxiliary information about it is wrong. The top-scoring record is the best guess; to decide whether to trust it, the **eccentricity test** declares a match only if the gap between the two best scores $s_1$ and $s_2$, in units of the standard deviation $\sigma$ of all scores, clears a threshold, $(s_1 - s_2)/\sigma \ge \varphi$. Each agreeing item is worth $\log_2(1/s_i)$ bits: three films each rated by 1% of subscribers carry 19.9 bits, more than the 18.9 bits that single out one Netflix subscriber.

**Example.** On Netflix, 8 ratings of which 2 may be wrong, with dates known within 14 days, uniquely identify 99% of records; 2 ratings with dates within 3 days identify 68%. A synthetic matrix of 5,000 users, with density close to Netflix's, gives @tbl-link-netflix.

| Auxiliary information | Equal weights | $1/\ln\lvert\mathrm{supp}(i)\rvert$ | $\ln(n/\lvert\mathrm{supp}(i)\rvert)$ |
| --- | --- | --- | --- |
| 2 items | 0.428 | 0.440 | 0.402 |
| 4 items | 0.936 | 0.941 | 0.924 |
| 8 items | 1.000 | 1.000 | 1.000 |
| 8 items, 2 of them wrong | 0.943 | 0.989 | 0.978 |

: Re-identification rate in a synthetic ratings matrix with 5,000 users: the fraction of 400 targets whose record gets the top score, for three weightings of @eq-link-score. {#tbl-link-netflix}

An item carries a median of 5.83 bits against $\log_2 5000 = 12.29$ needed, so two items re-identify about 43% and four 93%. With two wrong items equal weights drop to 94%, because a wrong popular item helps many unrelated records.

**Pitfalls.** The popular threshold $\varphi = 1.5$ is not a safe default. Scoring 400 outsiders, people not in the release, with 8 items of which 2 are wrong: $\varphi = 1.5$ correctly matches 94.8% of insiders but also "matches" 26.3% of outsiders. Holding false matches to 1% of outsiders takes $\varphi = 4.83$, which still re-identifies 73.0% of insiders. Calibrate the threshold on records known not to be the target, and read the attack, like any other, at a small false positive rate (@fig-link-eccentricity). As a defender, don't expect perturbing a few values to protect sparse data: the score needs a margin, not a perfect match.

![True against false positive rates of the eccentricity test as the threshold $\varphi$ is swept, for 4, 6 and 8 auxiliary items with 1, 1 and 2 of them wrong. The dots mark $\varphi = 1.5$. A false positive rate of 0 is drawn at the left edge.](figures/link_eccentricity_roc.svg){#fig-link-eccentricity}

## Does releasing a sample protect? {#sec-link-sampling}

**Idea.** Agencies often release a sample, say 1% of a census, arguing plausible deniability: a record unique in the sample need not be unique in the population. That only helps people in crowded cells. Let each person be in the release with probability $\pi$, the **sampling fraction**, and let $\lambda$ be the expected number of other people in the target's cell.

**Result.** Matching from a known person to a unique record in the release is correct with probability
$$\kappa \approx \frac{1}{1 + \lambda(1 - \pi)},$$ {#eq-link-kappa}
and a record unique in the sample is unique in the whole population, so that a register such as a voter roll names it, with probability
$$\xi \approx e^{-\lambda(1 - \pi)}.$$ {#eq-link-xi}
Sampling enters only through $1 - \pi$, so whatever the sampling fraction, $\kappa \ge 1/(1 + \lambda)$ and $\xi \ge e^{-\lambda}$. Over a whole release, the fraction of sample uniques that are population uniques tends to the population's uniqueness rate as $\pi \to 0$, not to 0.

**Example.** @tbl-link-kappa gives both probabilities for three crowding levels. Going from a 50% to a 1% sample lowers $\kappa$ by 0.04 for a rare cell and by 0.16 for a crowded one.

| $\lambda$ (expected others in the cell) | $\kappa$, $\pi = 0.01$ | $\kappa$, $\pi = 0.5$ | $\xi$, $\pi = 0.01$ | $\xi$, $\pi = 0.5$ |
| --- | --- | --- | --- | --- |
| 0.1 | 0.910 | 0.952 | 0.906 | 0.951 |
| 0.5 | 0.669 | 0.800 | 0.610 | 0.779 |
| 2 | 0.336 | 0.500 | 0.138 | 0.368 |

: Probability that a match in a sample is correct: from a person to the release ($\kappa$) and from a sample-unique record to the population ($\xi$). {#tbl-link-kappa}

In a synthetic population of 100,000 with seven attributes, 82.8% of people are unique. In a 1% sample, 0.8301 of the sample-unique records are population-unique: the sample protects almost nobody who was not already protected.

**Pitfalls.** The adversary can tell which matches to trust. An adversary who estimates each attribute's frequencies from one 1% sample, and from them each record's $\xi$, predicts 0.817 on average against an actual 0.824, and the records it rates above 0.95 are population-unique 99.3% of the time. So "any match might be wrong" is no defence. A defence has to hold whatever auxiliary data exist. The classic attempts, k-anonymity and its successors, are tested against that requirement in [Syntactic anonymity and how it breaks](08-syntactic-anonymity.md), and [The definition of differential privacy](../part03-dp-foundations/01-definition.md) gives the guarantee that meets it.

## Code

Three numbers summarise the linkage risk of a release you hold: $J/n$, the fraction of unique records and the smallest class size. For a planned release, the uniform-cells formula and @eq-link-kappa give quick estimates.

```python
from collections import Counter
from math import exp, log2


def linkage_report(rows: list[tuple[str, ...]]) -> tuple[float, float, int]:
    """Linkage success J/n, fraction of unique records, and smallest class size of a release."""
    sizes = Counter(rows).values()  # one count per distinct quasi-identifier combination
    n = len(rows)
    return len(sizes) / n, sum(1 for k in sizes if k == 1) / n, min(sizes)


def uniform_unique(n: float, cells: float) -> float:
    """Expected fraction of n people who are unique when they fall in `cells` equally likely cells."""
    return exp(-n / cells)


def match_correct(lam: float, pi: float) -> tuple[float, float]:
    """In a sample of fraction pi, with lam expected others in the cell: (kappa, xi)."""
    return 1 / (1 + lam * (1 - pi)), exp(-lam * (1 - pi))


release = [
    ("02138", "1961-03-14", "F"), ("02138", "1972-11-02", "M"), ("02138", "1972-11-02", "M"),
    ("02139", "1958-06-21", "F"), ("02139", "1990-01-01", "F"), ("02139", "1990-01-01", "F"),
    ("02139", "1990-01-01", "F"), ("02141", "1983-09-30", "M"),
]
success, unique, k = linkage_report(release)
print(f"{success:.3f} {unique:.3f} {k}")  # 0.625 0.375 1
cells = 40_000 * 365 * 80 * 2  # ZIP code x birth date over 80 years x sex
print(f"{log2(cells):.1f} {uniform_unique(248.7e6, cells):.3f}")  # 31.1 0.899
kappa, xi = match_correct(0.1, 0.01)
print(f"{kappa:.3f} {xi:.3f}")  # 0.910 0.906
```

The full edition's script, [linkage.py](../../code/part02-attacks/linkage.py), computes every number in this chapter, including the synthetic Netflix experiment and the sampling simulations:

```bash
uv run python code/part02-attacks/linkage.py
```

## Cheat sheet

| Result | Formula | Notes |
| --- | --- | --- |
| Bits to name one person | $\log_2 n$ | 33 for the world, 28 for the US, 13 for 10,000 people |
| Linkage success | $J/n$ | Target known to be in the release; at most $1/k$ if every class has $\ge k$ records |
| Expected fraction unique | $\sum_c p_c(1 - p_c)^{n-1}$ | Needs a model of the joint distribution |
| Uniform cells, $b$ bits | $\exp(-2^{\log_2 n - b})$ | 50% unique at $\log_2 n + 0.53$ bits, 99% at $\log_2 n + 6.64$ |
| Collision floor | $(1 - \sum_c p_c^2)^{n-1}$ | Needs only the marginals for independent attributes |
| Sparse-data score | $\sum_i w_i\,\mathrm{Sim}(\mathrm{aux}_i, r_i)$, $w_i \approx \ln(1/s_i) + \ln\frac{1-\varepsilon}{\varepsilon}$ | Each agreeing item is worth $\log_2(1/s_i)$ bits |
| Eccentricity | $(s_1 - s_2)/\sigma \ge \varphi$ | Calibrate $\varphi$ on outsiders at a small false positive rate |
| Match in a sample | $\kappa \approx 1/(1 + \lambda(1 - \pi))$, $\xi \approx e^{-\lambda(1 - \pi)}$ | At least $1/(1 + \lambda)$ and $e^{-\lambda}$ for every $\pi$ |

: The results of this chapter. {#tbl-link-cheat}

**Rules of thumb**

- Add up the bits of every attribute an adversary might know. If the sum exceeds $\log_2 n$, assume most people are unique; about 7 bits more make 99% unique.
- Measure a release you hold by its linkage success $J/n$ and its unique records, not by whether names were removed.
- In sparse data, a handful of rare items is enough, noise or not.
- When you run the attack, calibrate the match threshold on people known not to be in the data.
- Removing names, releasing a sample or perturbing a few values does not anonymize a dataset.

## Further reading

- Sweeney, "Simple Demographics Often Identify People Uniquely" (2000): the 87% estimate and its variants with coarser geography.
- Narayanan & Shmatikov, "Robust De-anonymization of Large Sparse Datasets" (IEEE Symposium on Security and Privacy, 2008): the scoring algorithm, the eccentricity test and the Netflix experiments.
- Rocher, Hendrickx & de Montjoye, "Estimating the success of re-identifications in incomplete datasets using generative models" (Nature Communications, 2019): per-record re-identification risk from a copula model, and why sampling does not protect.
- Ohm, "Broken Promises of Privacy: Responding to the Surprising Failure of Anonymization" (UCLA Law Review, 2010): the legal side, with the Massachusetts, search-log and Netflix cases.

**Derivations.** Every result in this chapter is derived step by step in [the full edition's chapter](../../book-full/part02-attacks/07-linkage-reidentification.md).
