# Linkage and re-identification

## Why this matters

A hospital wants to share its discharge records with researchers. It deletes the names, the addresses and the insurance numbers, and keeps what the research needs: postcode, date of birth, sex, diagnosis. Nobody's name is in the file. Is anyone in it identifiable?

Usually, yes. Most people are the only person with their combination of postcode, birth date and sex. Anyone who holds a second table with names and those same three columns, such as a voter roll, can join the two tables and put a name back on most rows. This is a **linkage attack**, and putting a name back on a "de-identified" record is **re-identification**. It needs no model, no gradients and no queries: only a release and some auxiliary data about the people in it. The other chapters of this part attack trained models. This one attacks the data itself, and it explains why removing names was never the same as anonymizing.

This chapter follows three questions, each building on the one before. The first is how many people are unique on a given set of attributes, and how you can predict that number without having the data. Once that is answered for tidy tables, the second question is what happens when the data are high-dimensional and sparse, like movie ratings or shopping baskets, so that no record matches the adversary's information exactly. The third is whether releasing only a sample helps, since a record that is unique in the sample need not be unique in the population.

### The one idea: identifying information is measured in bits, and bits add up

Picking one person out of $n$ takes $\log_2 n$ bits of information, because each bit halves the number of candidates and you need to halve $n$ down to 1. That is 27.9 bits for the 248.7 million people in the United States in 1990, and 32.9 bits for 8 billion people. Each attribute you learn about someone carries some bits too: an attribute with $v$ equally likely values is worth $\log_2 v$ bits. A sex is worth $\log_2 2 = 1$ bit, a birth month $\log_2 12 \approx 3.6$, and a birth date in a span of 80 years, one of $365 \times 80 = 29{,}200$ days, about $\log_2 29{,}200 = 14.8$. For independent attributes, the bits add: the chance of a given combination is the product of the chances of its values, and the logarithm turns that product into a sum. **Once the bits of the attributes you know exceed $\log_2 n$ by a few, almost everyone is unique.** Every result in this chapter makes that sentence precise: the expected fraction of unique people as a function of the bits (@sec-link-unique), the score that adds up bits from rare movie ratings (@sec-link-sparse), and the correction for a sample (@sec-link-sampling).

@tbl-link-uses lists the results and where each is used.

| Result | Where it is used |
| --- | --- |
| Linkage success is the number of distinct quasi-identifier combinations over $n$ | Scoring a release against an adversary who knows who is in it |
| Expected fraction unique, $\sum_c p_c(1 - p_c)^{n-1}$ | Predicting re-identification risk from a model of the population |
| Uniform cells: $e^{-n/M}$, a single curve in bits minus $\log_2 n$ | The "33 bits" rule and its correction |
| Collision lower bound $(1 - \sum_c p_c^2)^{n-1}$ | A guaranteed uniqueness rate from marginals alone |
| Concentration of the fraction unique (McDiarmid) | Why one sample of the population is enough |
| Narayanan–Shmatikov score as a likelihood ratio | De-anonymizing sparse data with noisy auxiliary information |
| Eccentricity and its false-match rate | Deciding when the best match is the right one |
| Correctness of a match in a sample, $1/(1 + \lambda(1 - \pi))$ and $e^{-\lambda(1 - \pi)}$ | Why releasing a small sample does not protect the people in it |

: The results of this chapter and where each is used. {#tbl-link-uses}

**What you need.** Every outside fact this chapter uses is restated where it is used; these links are for more detail. Entropy, the chain rule and the conversion between probabilities and bits are in [Information theory](../part01-prerequisites/05-information-theory.md). The binomial and Poisson distributions, and Poisson thinning ([Exercise 2](../part01-prerequisites/solutions/01-distributions.md#sec-sol-distributions-2) of that chapter), are in [Probability distributions](../part01-prerequisites/01-distributions.md). McDiarmid's inequality is in [Concentration](../part01-prerequisites/02-concentration.md). The Neyman–Pearson lemma, the Bayesian view of a test, and why you should read a curve of true against false positive rates at small false positive rates are in [Hypothesis testing](../part01-prerequisites/04-hypothesis-testing.md). The birthday bound is in [Cryptographic basics](../part01-prerequisites/08-crypto-basics.md). The vocabulary of adversaries, auxiliary knowledge and games is in [Threat models](01-threat-models.md).

**Notation.** A population has $n$ people. Each person has a combination of attribute values, their **cell**. Under the model of the population, a person falls in cell $c$ with probability $p_c$, independently of everyone else. There are $M$ cells. A person is **unique** when nobody else is in their cell. $U$ is the number of unique people. Logarithms are base 2 when the unit is bits and natural otherwise.

## Toolkit: three facts every derivation uses

### Fact 1 — Counting with indicators

To find how many people are unique on average, you do not need the distribution of $U$. Give each person a counter, an **indicator**, that is 1 if that person is unique and 0 otherwise. $U$ is the sum of these counters, $U = \sum_{i=1}^n\mathbf 1[\text{person } i \text{ is unique}]$. The average of a sum is the sum of the averages, and the average of a 0/1 counter is the probability that it is 1. So

$$\mathbb E U = \sum_{i=1}^n\Pr[\text{person } i \text{ is unique}].$$

The rule "the average of a sum is the sum of the averages" is **linearity of expectation**, and it holds even when the terms depend on each other. Here they do: knowing that person 1 is unique changes the chance that person 2 is. Linearity does not care, because it only adds up how often each counter is 1, outcome by outcome. Walk through the smallest case to see it: $n = 3$ people, $M = 2$ equally likely cells A and B, so 8 equally likely outcomes (@tbl-link-indicator).

| Cells of (person 1, 2, 3) | `AAA` | `AAB` | `ABA` | `BAA` | `ABB` | `BAB` | `BBA` | `BBB` |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Unique people | none | 3 | 2 | 1 | 1 | 2 | 3 | none |
| $U$ | 0 | 1 | 1 | 1 | 1 | 1 | 1 | 0 |

: Three people, two equally likely cells: who is unique in each of the 8 outcomes. {#tbl-link-indicator}

Directly, six of the eight outcomes have $U = 1$, so $\mathbb E U = 6/8 = 0.75$. Through the indicators, person 1 is unique when persons 2 and 3 are both in the other cell, which happens with probability $\frac12 \times \frac12 = \frac14$; the same holds for each person, so $\mathbb E U = 3 \times \frac14 = 0.75$. The two counts agree. This chapter uses the same move for the expected number of unique people, of sample uniques, and of false matches.

### Fact 2 — $(1 - p)^k$ is $e^{-pk}$, up to a factor you can bound

The question "does each of the other $k$ people avoid my cell?" has the answer $(1 - p)^k$ when each avoids it with probability $1 - p$. Powers like this are awkward, and an exponential is easier to read: $(1 - p)^k$ is close to $e^{-pk}$ when $p$ is small, because $1 - p$ is close to $e^{-p}$ for small $p$. The fact pins down how close, with bounds on both sides. For $0 \le p < 1$ and $k \ge 0$,

$$e^{-pk/(1-p)} \le (1 - p)^k \le e^{-pk}.$$

To prove it, take logarithms, so that $(1 - p)^k$ becomes $k\ln(1 - p)$ and the claim becomes $-\frac{p}{1 - p} \le \ln(1 - p) \le -p$. Both sides come from one inequality, $\ln x \le x - 1$ for all $x > 0$, which holds because $\ln$ is concave and $x - 1$ is its tangent line at $x = 1$, and a concave curve lies below its tangents. With $x = 1 - p$ it gives the upper bound $\ln(1 - p) \le -p$. With $x = \frac1{1 - p}$ it gives $\ln\frac1{1-p} \le \frac1{1-p} - 1 = \frac{p}{1-p}$, and flipping the sign gives the lower bound $\ln(1 - p) \ge -\frac{p}{1-p}$. The two bounds differ only by the factor $\frac1{1-p}$ in the exponent, which is close to 1 when $p$ is small.

To see how tight this is, take $p = 10^{-4}$ and $k = 9999$. Then $pk = 0.9999$ and $pk/(1 - p) = 0.9999/0.9999 = 1$, and $(1-p)^k = 0.36790$ sits between the lower bound $e^{-pk/(1-p)} = e^{-1} = 0.36788$ and the upper bound $e^{-pk} = e^{-0.9999} = 0.36792$. This is how the chapter turns "every one of the other $n - 1$ people avoids my cell" into $e^{-np}$.

### Fact 3 — Probabilities multiply, bits add

If attributes $A_1, \dots, A_k$ are independent, the probability of a cell is the product of the probabilities of its values, $p_c = \prod_j p_j(c_j)$, where $p_j(c_j)$ is the probability that attribute $j$ takes the value $c_j$ that the cell has. Three quantities that this chapter uses follow from that one product.

The **surprisal** of a cell, $\log_2(1/p_c)$, is the number of bits it takes to single out that cell: a cell with probability $2^{-10}$ has a surprisal of 10 bits. Since the logarithm of a product is the sum of the logarithms, surprisal adds up over attributes, $\log_2(1/p_c) = \sum_j\log_2(1/p_j(c_j))$.

The **entropy** is the average surprisal, $H = \sum_c p_c\log_2(1/p_c)$: how many bits a typical person's cell carries. Averaging the sum above term by term, it adds up too, $H(A_1, \dots, A_k) = \sum_jH(A_j)$. This is the chain rule of [Information theory](../part01-prerequisites/05-information-theory.md), which says that the entropy of a pair is the entropy of the first plus that of the second given the first, in the special case where knowing the first tells you nothing about the second.

The **collision probability** $\sum_cp_c^2$ is the chance that two independent people land in the same cell: both land in cell $c$ with probability $p_c \cdot p_c$, and you add over cells. It multiplies over attributes, because two people share a cell exactly when they share the value of every attribute, and for independent attributes those agreements are independent events. In symbols, $\sum_c\prod_jp_j(c_j)^2 = \prod_j\sum_vp_j(v)^2$, which you can check by expanding the product of sums on the right: each term of the expansion picks one value for each attribute, that is, one cell.

Real attributes are not independent (age and number of children, ZIP code and income), and then the pair carries fewer bits than the sum of their separate bits, because part of what the second attribute says was already said by the first. The sum is therefore an upper bound on the joint entropy, and adding up bits overstates how identifying correlated attributes are.

## Quasi-identifiers and the linkage attack {#sec-link-quasi}

### Context

Data protection practice sorts the columns of a table into three kinds. **Direct identifiers** name a person on their own: name, social security number, email address. **Sensitive attributes** are what the release is about and what must not be tied to a person: a diagnosis, a salary. **Quasi-identifiers** are everything in between: attributes that do not name anyone alone but that, combined, often single out one person, and that also appear in other datasets with names attached. ZIP code, birth date and sex are the classic example. Removing direct identifiers and keeping quasi-identifiers is the most common form of "de-identification", and the linkage attack is what breaks it. This section asks how often the attack names the right record.

### Statement

**The linkage attack.** The release $R$ holds records (quasi-identifiers, sensitive attribute). The adversary holds an auxiliary table with (name, quasi-identifiers), for example a voter roll. For each name, it finds the records of $R$ with the same quasi-identifiers and picks one of them uniformly at random.

**The game.** In the language of [Threat models](01-threat-models.md), an attack is scored by a game. A challenger draws a target uniformly from the $n$ people in $R$ and gives the adversary the target's quasi-identifiers. The adversary wins if it outputs the target's record. Group the records of $R$ into **equivalence classes**, sets of records with identical quasi-identifiers, with sizes $k_1, \dots, k_J$; there are $J$ classes, one per distinct combination. A target alone in its class is found for sure, and a target in a class of $k$ look-alikes is found one time in $k$. A class of $k$ records is therefore hit $k$ times as often as a single record but won only a $1/k$ share of those times, so each class contributes the same amount, one record's worth, to the wins. Averaging over the target gives the success probability of the linkage attack,

$$\Pr[\text{win}] = \frac1n\sum_{j=1}^J k_j\cdot\frac{1}{k_j} = \frac{J}{n}:$$ {#eq-link-success}

the number of distinct quasi-identifier combinations divided by the number of records. No adversary that sees only the quasi-identifiers can do better.

**Sweeney's measurement.** Sweeney (2000) estimated from 1990 census tables that 87% of the US population, 216 of 248 million people, were likely unique on {5-digit ZIP code, date of birth, sex}. On {city, date of birth, sex} the figure was 53%, and on {county, date of birth, sex} 18%.

### Derivation

The idea is to fix the class the target falls in, find the best guess within it, and then average over classes.

1. **Condition on the target's class.** A target in a class of size $k_j$ shares its quasi-identifiers with $k_j - 1$ other records. To an adversary who sees only quasi-identifiers, those records look the same as the target's. Picking one of the $k_j$ at random wins with probability $1/k_j$.
2. **Show that nothing beats this.** Within a class, the target is equally likely to be any of the $k_j$ records, because it was drawn uniformly and the records differ only in their sensitive attribute, which the adversary does not know. So every guess within the class wins with probability exactly $1/k_j$, and a guess outside the class never wins, since a record there has different quasi-identifiers from the target.
3. **Average over the target.** The target lands in class $j$ with probability $k_j/n$, the share of records in that class. Weighting each class's win probability by how often the target lands in it gives $\Pr[\text{win}] = \sum_j\frac{k_j}{n}\cdot\frac{1}{k_j} = \frac Jn$.

### Interpretation

For this adversary, uniqueness is the whole story. A unique record is re-identified with certainty and a record in a class of 2 with probability $\frac12$. Each unique record adds a full win and no class adds more, so the fraction of unique people is a lower bound on the success, and the success is at most 1. That is why the next section is about predicting uniqueness.

The game assumes the adversary knows the target is in the release, as a curious neighbour or an insurer with a list of patients might. Without that knowledge, a unique match in the release may belong to someone else who happens to share the quasi-identifiers but is not in $R$. Whether that uncertainty protects anyone is the question of @sec-link-sampling.

Re-identification is also not the only harm. If all records of a class share one diagnosis, the adversary learns the diagnosis without knowing which record is the target's. In the other direction, @eq-link-success shows that a release in which every class has at least $k$ records keeps the linkage success at most $1/k$: each class then contributes one win for at least $k$ records, so $J \le n/k$. That guarantee holds against this adversary only. [Syntactic anonymity and how it breaks](08-syntactic-anonymity.md) studies such releases and the attacks on the sensitive attribute that they leave open.

### Example: one hospital release, one voter roll

@tbl-link-release is a hospital release with names removed, and @tbl-link-voters part of a public voter roll.

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

Joining on the three columns gives Ana's asthma, Carla's depression and Dan's pneumonia with certainty, because each of them matches exactly one record. Ben matches two records, so the adversary gets diabetes or hypertension and is right with probability $\frac12$. The release has classes of sizes 1, 1, 1, 2 and 3, so $J = 5$ classes for $n = 8$ records, and @eq-link-success gives a success of $5/8 = 0.625$ for a target drawn from the release.

This is, in miniature, how Sweeney re-identified the governor of Massachusetts in the late 1990s. The state's Group Insurance Commission had released hospital records of about 135,000 state employees and their families without names. Sweeney bought the Cambridge voter roll for twenty dollars. Six people on it shared the governor's birth date, three of them were men, and only one of those lived in his ZIP code.

**In bits** (@fig-link-bits). Now count the bits of Sweeney's three attributes, treating each as uniform over its values, which gives an upper bound. There are about 40,000 ZIP codes, $365 \times 80 = 29{,}200$ birthdays in about 80 birth years, and two sexes, so the three attributes carry at most $\log_2 40{,}000 + \log_2 29{,}200 + 1 = 15.3 + 14.8 + 1.0 = 31.1$ bits, against $\log_2(248.7\times10^6) = 27.9$ bits needed to single out one American. The same count in cells: there are $M = 40{,}000 \times 29{,}200 \times 2 = 2.34\times10^9$ cells for $n = 248.7$ million people, so $n/M = 0.106$ people per cell. If everyone fell into a uniformly random cell, the next section shows that a fraction $e^{-n/M} = e^{-0.106} = 0.899$ would be unique. Sweeney's 87% is a little lower because people are not spread uniformly: ZIP codes range from a few residents to tens of thousands, and birth years are not equally common, so people crowd into the popular cells. Golle (2006) repeated the estimate with the 2000 census and found 63%. Both estimates have to model how birth dates spread within the age groups that census tables publish, and the answer is sensitive to that modelling.

![The three classic quasi-identifiers, in bits, against the bits needed to single out one person in the US population of 1990 and in the world population. The bits are upper bounds that treat each attribute as uniform over its values.](figures/link_bits.svg){#fig-link-bits}

### Check yourself

- A release of 1,000 records has 900 distinct quasi-identifier combinations. What is the linkage success for a target drawn from the release, and what is the smallest possible number of unique records? *(0.9 by @eq-link-success. At least 800: with $s$ singletons, the other $900 - s$ classes hold at least 2 records each, so $s + 2(900 - s) \le 1000$, that is $s \ge 800$, attained with 100 pairs.)*
- Why does it not matter for @eq-link-success what the sensitive attribute is? *(The adversary chooses only from the quasi-identifiers. The sensitive attribute is what it learns after winning, not what it uses to win.)*

## How many people are unique? A birthday calculation {#sec-link-unique}

### Context

A data custodian who wants to know the re-identification risk of a release can count unique records in it. But counting needs the data, and it says nothing about a new release or a different set of columns. A model of the population answers both. If each person falls in cell $c$ with probability $p_c$, the expected fraction of unique people has a closed form. It is the birthday problem asked the other way round. The birthday problem, in [Cryptographic basics](../part01-prerequisites/08-crypto-basics.md), asks whether *any* two of $n$ people share a birthday; here you ask how many people share their cell with *nobody*.

### Statement

**Exact expected fraction.** A person is unique when each of the other $n - 1$ people lands outside their cell. If the person is in cell $c$, each other person avoids it with probability $1 - p_c$, independently, so all of them avoid it with probability $(1 - p_c)^{n-1}$. Averaging over the person's own cell, which is $c$ with probability $p_c$, gives the expected fraction of unique people, for $n$ people falling in cells independently with probabilities $(p_c)$:

$$\frac{\mathbb E U}{n} = \sum_cp_c(1 - p_c)^{n-1}.$$ {#eq-link-unique}

**Uniform cells.** If all $M$ cells are equally likely, every term has $p_c = 1/M$, and the formula becomes $(1 - 1/M)^{n-1} \approx e^{-n/M}$: the chance that nobody else lands in your cell when the average cell holds $n/M$ people. Now count the cells in bits, $b = \log_2M$, so that $M = 2^b$ and $n = 2^{\log_2 n}$. The average occupancy is $n/M = 2^{\log_2n - b}$, and

$$\frac{\mathbb E U}{n} \approx \exp\big(-2^{\log_2n - b}\big),$$ {#eq-link-uniform}

a single curve in $b - \log_2n$, the bits beyond those needed to name one person. Solving $e^{-n/M} = f$ for the bits says how many you need for a fraction $f$ of people to be unique: $b = \log_2n + \log_2\frac{1}{\ln(1/f)}$. That is half once $b$ exceeds $\log_2n$ by 0.53 bits, 90% at 3.25 bits more and 99% at 6.64 bits more.

**A lower bound from the collision probability.** The exact formula needs every $p_c$, and there can be billions of cells. A cruder question needs much less: how likely is a random person to share a cell with one other random person? That is the collision probability $\sum_cp_c^2$ of Fact 3. If you pretend every person's cell is as crowded as average in this sense, you get a guaranteed floor on uniqueness. For any $(p_c)$,

$$\frac{\mathbb E U}{n} \ge \Big(1 - \sum_cp_c^2\Big)^{n-1},$$ {#eq-link-collision}

and for independent attributes $\sum_cp_c^2 = \prod_j\sum_vp_j(v)^2$ by Fact 3, so the bound needs only the per-attribute frequencies (the **marginals**).

**Concentration.** The expectation is an average over many imagined populations, but you have only one. Moving a single person to another cell changes the number of unique people by at most 2, so in a large population no one person can move the fraction much, and the fraction in the population you have is close to its expectation: for every $t > 0$,

$$\Pr\Big[\Big|\frac Un - \frac{\mathbb EU}{n}\Big| \ge t\Big] \le 2e^{-nt^2/2}.$$ {#eq-link-mcdiarmid}

### Derivation

**The exact formula** (@eq-link-unique).

1. **Count with indicators.** By Fact 1, $\mathbb EU = \sum_i\Pr[\text{person } i \text{ is unique}]$. Every person draws their cell from the same probabilities $(p_c)$, independently of the others, so nothing distinguishes person $i$ from person $j$ and every term of the sum is the same. It is enough to find it for one person.
2. **Condition on the person's own cell.** Call it $c$; it has probability $p_c$.
3. **Ask the others to stay away.** Given $c$, person $i$ is unique if each of the other $n - 1$ people lands outside $c$. They are independent and each avoids $c$ with probability $1 - p_c$, so this has probability $(1 - p_c)^{n-1}$.
4. **Average over the cell.** $\Pr[\text{person } i \text{ is unique}] = \sum_cp_c(1 - p_c)^{n-1}$. Since all $n$ terms are equal, $\mathbb EU$ is $n$ times this; divide by $n$.

**Uniform cells** (@eq-link-uniform).

1. **Put $p_c = 1/M$ in @eq-link-unique.** There are $M$ terms, each $\frac1M(1 - \frac1M)^{n-1}$, so the sum is $(1 - \frac1M)^{n-1}$.
2. **Replace the power by an exponential.** Fact 2 with $p = 1/M$ and $k = n - 1$ squeezes $(1 - \frac1M)^{n-1}$ between $e^{-(n-1)/M}$ and $e^{-(n-1)/(M-1)}$. These exponents differ by $\frac{n-1}{M(M-1)}$, of order $n/M^2$, and replacing $n - 1$ by $n$ changes the exponent by only $1/M$. So $(1 - \frac1M)^{n-1} \approx e^{-n/M}$ with relative error of order $n/M^2$, negligible for $M$ in the thousands or more.
3. **Write the exponent in bits.** $n/M = 2^{\log_2n}/2^b = 2^{\log_2n - b}$.
4. **Find the bits for a target fraction.** Solve $e^{-n/M} = f$ for $M$: taking logarithms, $n/M = \ln(1/f)$, so $M = n/\ln(1/f)$ and $b = \log_2n - \log_2\ln(1/f)$. At $f = \frac12$, $\ln 2 = 0.693$ and $-\log_2 0.693 = 0.53$. At $f = 0.9$, $\ln\frac{1}{0.9} = 0.105$ and $-\log_2 0.105 = 3.25$. At $f = 0.99$, $\ln\frac1{0.99} = 0.0101$ and $-\log_2 0.0101 = 6.64$.

**The collision lower bound** (@eq-link-collision). The idea is that @eq-link-unique is an average of the function $(1 - x)^{n-1}$ over the cell probabilities people find themselves in, and for a curve that bends upwards the average of the function is at least the function of the average.

1. **Write @eq-link-unique as an average.** Draw a person's cell $C$, so $C = c$ with probability $p_c$, and look at the probability $p_C$ of the cell they landed in. Then $\frac{\mathbb EU}{n} = \mathbb E[g(p_C)]$ with $g(x) = (1 - x)^{n-1}$.
2. **Check that $g$ bends upwards.** $g$ is convex on $[0, 1]$, because its second derivative $g''(x) = (n-1)(n-2)(1 - x)^{n-3}$ is never negative there.
3. **Apply Jensen's inequality.** For a convex function, the average of the function is at least the function of the average, $\mathbb E[g(X)] \ge g(\mathbb E X)$. The reason is that a convex curve lies above its tangent line at the point $\mathbb E X$; averaging the curve gives at least the average of the tangent line, which is the tangent's value at $\mathbb E X$, which is $g(\mathbb E X)$. So $\mathbb E[g(p_C)] \ge g(\mathbb E[p_C])$.
4. **Compute the inner average.** $\mathbb E[p_C] = \sum_cp_c\cdot p_c = \sum_cp_c^2$, the collision probability. For independent attributes it factorizes by Fact 3.

Equality in step 3 holds when $p_C$ is constant, that is, when all cells with positive probability are equally likely, because then there is nothing to average. The bound is tight for uniform cells and loose when a few cells hold most of the people: those crowded cells push the collision probability up and the bound down, while the many people in rare cells are still unique.

**Concentration** (@eq-link-mcdiarmid). The tool is McDiarmid's inequality ([Concentration](../part01-prerequisites/02-concentration.md)). It applies to any function $f$ of $n$ independent inputs such that changing input $i$ alone, with the others fixed, moves $f$ by at most $c_i$. It says that $f$ is unlikely to be far from its mean: $\Pr[f - \mathbb Ef \ge t] \le \exp(-2t^2/\sum_ic_i^2)$, and the same for the lower tail. So the work is to find $c_i$.

1. **Change one input.** $U$ is a function of the $n$ independent cells. Change one person's cell from $a$ to $b$; only cells $a$ and $b$ change their counts, so only they can change $U$.
2. **Bound the change.** A cell with count $k$ contributes $\mathbf 1[k = 1]$ to $U$: it holds a unique person exactly when it holds one person. Cell $a$'s count drops by one and its contribution changes by $\mathbf 1[k_a - 1 = 1] - \mathbf 1[k_a = 1] \in \{-1, 0, 1\}$. Cell $b$'s count rises by one, and its contribution likewise changes by $-1$, 0 or 1. So $U$ changes by at most 2, and $U/n$ by at most $c_i = 2/n$. (It can change by exactly 2: move a person out of a pair and into an empty cell, and both the person and their former partner become unique.)
3. **Apply McDiarmid.** With $c_i = 2/n$ for all $n$ people, $\sum_ic_i^2 = n \cdot 4/n^2 = 4/n$, so each tail is at most $\exp(-2t^2/(4/n)) = e^{-nt^2/2}$. Adding the upper and lower tails gives the factor 2.

### Interpretation

For uniform cells, the bits picture is exact. @eq-link-uniform depends on $n$ and $M$ only through $b - \log_2n$, so the curve is the same for a town and for the planet (@fig-link-uniqueness, right). The often-quoted "33 bits identify anyone" is the point $b = \log_2 n$, where the average cell holds one person and the curve is at $e^{-1}$. With $2^{33}$ uniform cells for 8 billion people, $n/M = 8\times10^9/2^{33} = 0.93$ and only a fraction $e^{-0.93} = 0.394$ is unique. 33 bits are enough to *give* everyone a distinct label, but random labels of 33 bits collide often; you need $32.9 + 6.64 = 39.5$ bits for 99% of the world to be unique.

Skew usually lowers uniqueness. When a few values are common (a big city, a common occupation), many people crowd into a few cells and are not unique, while many rare cells stay empty. In the census below, and for Sweeney's attributes, the uniform model with the same number of cells overstates uniqueness for this reason. This is not a theorem, though: with far fewer cells than people, a uniform spread leaves nobody unique, while a few rare cells would hold some uniques.

The entropy $H$ gives a rule of thumb, not a bound. The rule replaces the real cells by $2^H$ equally likely cells, as if every cell carried the average surprisal, and gives $e^{-n2^{-H}}$. The collision bound is always below it (up to replacing $n$ by $n - 1$), because the collision probability is at least $2^{-H}$. To see why, note that $\log_2\sum_cp_c^2$ is the logarithm of an average of $p_c$ (weighted by $p_c$), while $-H = \sum_cp_c\log_2p_c$ is the average of the logarithm, and for the concave logarithm Jensen's inequality runs the other way: the log of an average is at least the average of the log. So $\log_2\sum_cp_c^2 \ge -H$. The exact value can sit on either side of the entropy rule: in @tbl-link-census it is above it with four attributes and below it with five or more.

If attributes are independent, the prediction needs only the marginals. In practice they are not, and a model of the joint distribution is needed. That is what Rocher, Hendrickx & de Montjoye (2019) fit, a **Gaussian copula**: a model that keeps each attribute's own frequencies and ties the attributes together through the correlations of a multivariate normal distribution. It is how @sec-link-sampling estimates risk from a sample.

Finally, one population is enough. Setting the right side of @eq-link-mcdiarmid equal to a failure probability $\beta$ and solving for $t$ says that the fraction unique in the population you have is within $\sqrt{2\ln(2/\beta)/n}$ of its expectation with probability $1 - \beta$. For $n = 10{,}000$ and $\beta = 0.05$ that is $\sqrt{2\ln 40/10^4} = 0.0272$. That is about five times the simulated standard deviation (at most 0.0053, @tbl-link-census), where a normal approximation would put 95% of populations within two standard deviations. McDiarmid is loose here because it charges every person the worst case, a change of 2 in $U$, while most changes move it by 0 or 1.

### Example: a synthetic census

The script builds a population of $n = 10{,}000$ people with eight independent attributes, made up but shaped like real ones: sex (uniform), age in 16 five-year bands (flat, then thinning out), region (40 values with Zipf frequencies, so a few regions are large), birth month (uniform), marital status, number of children, occupation (20 values, Zipf) and blood group (real-world frequencies). It reveals them one at a time. @tbl-link-census gives, for each step, the entropy $H$ of the attributes known so far and the number of cells $M$, then the exact expected fraction unique from @eq-link-unique, the average over 20 simulated populations, the collision lower bound of @eq-link-collision, the entropy rule $e^{-n2^{-H}}$ and the uniform-cells value $e^{-n/M}$ for the same number of cells.

| Attributes known | Bits $H$ | Cells $M$ | Exact | Simulated | Collision bound | Entropy rule | Uniform cells |
| --- | --- | --- | --- | --- | --- | --- | --- |
| sex, age band, region | 9.31 | 1,280 | 0.0158 | 0.0160 | 0.0000 | 0.0000 | 0.0004 |
| + birth month | 12.90 | 15,360 | 0.3328 | 0.3332 | 0.0834 | 0.2692 | 0.5215 |
| + marital status | 14.60 | 61,440 | 0.5789 | 0.5797 | 0.4237 | 0.6690 | 0.8498 |
| + children | 16.48 | 368,640 | 0.7939 | 0.7918 | 0.7443 | 0.8967 | 0.9732 |
| + occupation | 20.39 | 7,372,800 | 0.9747 | 0.9737 | 0.9736 | 0.9928 | 0.9986 |
| + blood group | 22.57 | 58,982,400 | 0.9925 | 0.9926 | 0.9924 | 0.9984 | 0.9998 |

: Fraction of 10,000 people who are unique as attributes are revealed: exact, simulated (mean of 20 populations), and three approximations. $\log_2n = 13.29$ bits. {#tbl-link-census}

![Left: the fraction of people who are unique in the synthetic census as attributes are added, exact against one simulated population, with the collision lower bound and the entropy rule. Right: with uniform cells, the fraction unique depends only on the bits beyond $\log_2 n$; the curves for $n = 10^4$, $10^6$ and $8\times10^9$ coincide.](figures/link_uniqueness.svg){#fig-link-uniqueness}

Read the table along the bits column, against $\log_2n = 13.29$. With 12.90 bits, just below it, a third of the people are unique. With 20.39 bits, $20.39 - 13.29 = 7.1$ bits beyond it, 97.5% are, close to the 99% that the uniform curve predicts at 6.64 bits beyond. Uniform cells overstate uniqueness at every step, by up to $0.8498 - 0.5789 = 0.27$ (with five attributes), because the attributes are skewed. The collision bound is loose with few attributes, where region and age band put most people in a few large cells, and becomes tight once most cells are nearly empty. Simulation and the exact formula agree to within 0.003.

### Check yourself

- With $n = 1000$ people and $M = 1000$ equally likely cells, what fraction is unique, and how many bits beyond $\log_2n$ is that? *($(1 - 0.001)^{999} = 0.368 \approx e^{-1}$, at zero bits beyond $\log_2 n$, since $M = n$.)*
- Why can the indicators "person $i$ is unique" be dependent and still give $\mathbb EU = n\Pr[\text{person 1 is unique}]$? *(Linearity of expectation holds for any random variables, dependent or not. Dependence matters for the variance, which is why the concentration step needs McDiarmid rather than a sum of independent terms.)*

## Sparse, high-dimensional data: the Netflix attack {#sec-link-sparse}

### Context

In 2006 Netflix released about 100 million movie ratings by 480,189 subscribers for a public competition. Names were replaced by random numbers, and some ratings and dates were perturbed. The data look nothing like a census table: each record is a set of (movie, rating, date) triples, 209 on average out of 17,770 movies. There are no quasi-identifier columns to join on, and the adversary's information about a person ("she watched these films around these dates and liked them") is partial and noisy. Narayanan & Shmatikov (2008) showed that this does not help. In a sparse, high-dimensional dataset, a handful of approximate facts about a person typically matches one record far better than any other. They then matched a small sample of Netflix records to public reviews on the Internet Movie Database, which carry names or pseudonyms. This section explains why a scoring rule that favours rare items works, and when to trust its answer.

### Statement

**The scoring algorithm.** The adversary has auxiliary information $\mathrm{aux}$: a few items $i$ (films) with approximate values $\mathrm{aux}_i$ (ratings). Since no record will match exactly, it scores how well each record $r$ of the release agrees, item by item, and adds up the agreements. An agreement on a film that half the subscribers rated says little, while an agreement on an obscure film says a lot, so each item gets a weight $w_i$ that is larger for rarer items:

$$\mathrm{Score}(\mathrm{aux}, r) = \sum_{i \in \mathrm{aux}}w_i\,\mathrm{Sim}(\mathrm{aux}_i, r_i),$$ {#eq-link-score}

where $r_i$ is the record's value for item $i$ and $\mathrm{Sim} \in [0, 1]$ measures agreement (here: 1 if $r$ rated $i$ within one star of $\mathrm{aux}_i$, else 0). Narayanan and Shmatikov use $w_i = 1/\ln\lvert\mathrm{supp}(i)\rvert$, where $\lvert\mathrm{supp}(i)\rvert$, the **support** of item $i$, is the number of records that rated it.

**The eccentricity test.** The top-scoring record is the best guess, but the adversary also needs to know whether to trust it. A real match should stand out: its score should be well above the second-best one, measured against how spread out all the scores are. Let $s_1 \ge s_2$ be the two largest scores and $\sigma$ the standard deviation of all scores. The adversary declares a match only if the gap, in units of $\sigma$, clears a threshold $\varphi$,

$$\frac{s_1 - s_2}{\sigma} \ge \varphi,$$ {#eq-link-eccentricity}

and otherwise says "no match". Narayanan and Shmatikov used $\varphi = 1.5$.

**Their findings on Netflix.** With 8 ratings, of which 2 may be completely wrong, and dates known to within 14 days, 99% of records are uniquely identified. With 2 ratings and dates within 3 days, 68% are.

**The likelihood-ratio form of the score.** The weighting can be derived rather than guessed. For each record, the adversary is weighing two explanations: "this record is the target" against "this record is someone unrelated". The standard way to weigh two explanations of the same observations is the **likelihood ratio**, the probability of what you observed under the first explanation divided by its probability under the second. Its logarithm is a sum over items when items behave independently, so it has exactly the form of @eq-link-score. To make this precise, suppose an unrelated record agrees with the auxiliary value of item $i$ with probability $s_i$, independently across items, and the target's record agrees with probability $1 - \varepsilon$, where $\varepsilon$ covers wrong or stale auxiliary information. Then ranking records by the likelihood ratio is ranking them by @eq-link-score with

$$w_i = \ln\frac{1 - s_i}{s_i} + \ln\frac{1 - \varepsilon}{\varepsilon} \approx \ln\frac{1}{s_i} + \ln\frac{1 - \varepsilon}{\varepsilon},$$ {#eq-link-lr-weights}

where the approximation uses $1 - s_i \approx 1$ for rare items. The record with the highest score is the most likely to be the target.

### Derivation

**The weights.** The plan is to write the probability of a record's pattern of agreements under each explanation, take the log of the ratio, and collect the terms that differ from record to record.

1. **Record the agreements.** Write $a_i(r) = \mathrm{Sim}(\mathrm{aux}_i, r_i) \in \{0, 1\}$: 1 if record $r$ agrees with the auxiliary information on item $i$.
2. **Write the probability of the pattern under each explanation.** If $r$ is the target, each item agrees with probability $1 - \varepsilon$ and disagrees with probability $\varepsilon$, independently, so the pattern has probability $\Pr[a(r)] = \prod_i(1 - \varepsilon)^{a_i}\varepsilon^{1 - a_i}$ (the exponents pick the right factor for each item). If $r$ is unrelated, each item agrees with probability $s_i$, so $\Pr[a(r)] = \prod_is_i^{a_i}(1 - s_i)^{1 - a_i}$.
3. **Take the log of the ratio.** The logarithm turns the ratio of products into a sum over items: $\sum_i\big[a_i\ln\frac{1 - \varepsilon}{s_i} + (1 - a_i)\ln\frac{\varepsilon}{1 - s_i}\big]$.
4. **Keep only what depends on the record.** Collect the terms with $a_i$: $\sum_ia_i\big[\ln\frac{1-\varepsilon}{s_i} - \ln\frac{\varepsilon}{1 - s_i}\big] + \sum_i\ln\frac{\varepsilon}{1 - s_i}$. The last sum is the same for every record, so it does not change the ranking. The bracket, rearranged, is @eq-link-lr-weights: the log-likelihood ratio is the score with those weights, plus a constant.
5. **Why the likelihood ratio is the right thing to threshold.** For any single record, deciding between "target" and "unrelated" is a hypothesis test. The Neyman–Pearson lemma of [Hypothesis testing](../part01-prerequisites/04-hypothesis-testing.md) says that, among all tests that wrongly accept an unrelated record with a given probability, the one that accepts when the likelihood ratio exceeds a threshold accepts the true target most often. The intuition: if you can afford only so many false accepts, spend them on the patterns that are most typical of the target relative to an unrelated record.

**The best guess is the top score.** Step 5 is about one record at a time. When the adversary knows the target is one of the $n$ records and must pick one, Bayes' rule gives the answer. Suppose each record is equally likely a priori to be the target, and records are independent given who the target is.

1. **Write the probability of everything you see if $t$ is the target.** Record $t$'s pattern follows the target distribution and every other record's follows the unrelated one, so all the agreement patterns together have probability $P_1(a(t))\prod_{j \ne t}P_0(a(j))$, with $P_1$ and $P_0$ the two products of step 2 above.
2. **Turn it into a posterior.** Bayes' rule says the posterior probability of $t$ is proportional to its prior times this probability, and the prior is the same for every $t$. Divide by $\prod_jP_0(a(j))$, which does not depend on $t$ either: what remains is $P_1(a(t))/P_0(a(t)) = \mathrm{LR}(t)$, the likelihood ratio of record $t$. Normalizing so the probabilities sum to 1,
$$\Pr[\text{target} = t \mid \text{data}] = \frac{\mathrm{LR}(t)}{\sum_j\mathrm{LR}(j)}.$$ {#eq-link-posterior}
3. **Pick the most probable record.** The maximum a posteriori (MAP) guess is the record with the largest likelihood ratio, which is the top score.

**Why rare items carry the attack.** A false match is an unrelated record that happens to agree on every item the adversary knows. Take the clean case, $\varepsilon \to 0$ and exact agreement required. An unrelated record agrees on all of a set $I$ of items with probability $\prod_{i\in I}s_i$, and by Fact 1 the expected number of false matches among $n$ records is $n$ times that. Writing both factors as powers of 2 shows the bits:

$$n\prod_{i\in I}s_i = 2^{\log_2n - \sum_{i\in I}\log_2(1/s_i)}.$$

Each item contributes $\log_2(1/s_i)$ bits, its surprisal, and the false matches vanish once the bits exceed $\log_2n$: the same picture as in @sec-link-unique. A film that 1% of subscribers rated carries $\log_2100 = 6.64$ bits; three of them carry $3 \times 6.64 = 19.9$ bits, more than the $\log_2480{,}189 = 18.9$ bits needed to single out one Netflix subscriber.

### Interpretation

Sparsity is the vulnerability. In a sparse dataset most items are rated by few people, so each item a person rated is worth many bits, and nobody's record is close to anyone else's. Narayanan and Shmatikov formalize this as most records having no near neighbour. A census table has few columns with few values each; a ratings matrix has thousands of columns, and a person's "row" is a long sequence of rare events.

Because the score adds up evidence rather than demanding a perfect match, noise in the auxiliary information costs little. With a wrong item, the target loses one term of the score, while unrelated records rarely gain one. The score needs a margin, not a perfect match.

The eccentricity test answers a different question from the score. The top score is the best guess *if the target is in the release*. If it may not be, the adversary also needs to decide whether to trust the guess, and the eccentricity asks whether the top score stands out from the crowd of scores. How often it stands out by chance depends on the shape of the score distribution, so $\varphi$ has to be calibrated on records known not to be the target. It should be read at a small false positive rate, as for any attack, because a match is only useful if the adversary can be confident in it ([Hypothesis testing](../part01-prerequisites/04-hypothesis-testing.md)). @eq-link-posterior gives the principled version. Add one more explanation, "the target is not in the release", with prior $1 - q$, and give each record prior $q/n$. Under the new explanation every record is unrelated, so its likelihood, divided by $\prod_jP_0(a(j))$ as before, is 1. Bayes' rule then gives the posterior that record $t$ is the target as $q\,\mathrm{LR}(t)/\big(q\sum_j\mathrm{LR}(j) + (1 - q)n\big)$.

The weights matter less than the bits. Any weighting that favours rare items does about as well as any other; what decides success is whether the auxiliary information carries more bits than $\log_2n$.

### Example: a synthetic ratings matrix

The script draws a sparse matrix of 5,000 users and 4,000 items. Item popularity follows a Zipf law, the number of ratings per user is log-normal (median 30, mean 40.6), and ratings are 1 to 5 stars from a user bias, an item quality and noise. The result has 203,029 ratings and density 0.0102, close to Netflix's 0.0118. An item drawn from a random user's record carries a median of 5.83 bits, against $\log_2 5000 = 12.29$ bits needed. The auxiliary information about a target is $m$ items from their record, with the rating off by one star half the time; in the noisy setting, a quarter of the items are replaced by random items drawn by popularity, which the target has almost never rated. @tbl-link-netflix gives the fraction of 400 targets whose record gets the top score (ties broken at random), for three weightings. The first is equal weights. The second is the Narayanan–Shmatikov weights. The third is the likelihood-ratio weights of @eq-link-lr-weights, with $s_i = \lvert\mathrm{supp}(i)\rvert/n$, the fraction of users who rated the item, and $\varepsilon = \frac12$, which makes $\ln\frac{1-\varepsilon}{\varepsilon} = \ln 1 = 0$ and so drops the constant term.

| Auxiliary information | Equal weights | $1/\ln\lvert\mathrm{supp}(i)\rvert$ | $\ln(n/\lvert\mathrm{supp}(i)\rvert)$ |
| --- | --- | --- | --- |
| 2 items | 0.428 | 0.440 | 0.402 |
| 4 items | 0.936 | 0.941 | 0.924 |
| 8 items | 1.000 | 1.000 | 1.000 |
| 8 items, 2 of them wrong | 0.943 | 0.989 | 0.978 |

: Re-identification rate in a synthetic ratings matrix with 5,000 users: the fraction of 400 targets whose record gets the top score, for three weightings of @eq-link-score. {#tbl-link-netflix}

Read the table with the bits in mind. Two items carry about $2 \times 5.83 = 11.7$ bits, a little under $\log_2n$, and re-identify about 43% of targets. Four items carry about $4 \times 5.83 = 23$ bits and re-identify 93%. With 8 items of which 2 are wrong, the rarity weightings re-identify 98–99% of targets, matching the 99% of Narayanan and Shmatikov for 8 ratings with 2 wrong. Equal weights fall to 94% because a wrong popular item helps many unrelated records. The two rarity weightings differ by less than two standard errors; the standard error of a rate near 0.98 over 400 targets is $\sqrt{0.98 \times 0.02/400}$, which is 0.007 at these rates.

@fig-link-netflix (left and centre) shows what the eccentricity measures. When the target is in the release, its record scores far above every other. When the target is not, the top score is just the high end of a long tail.

![Left and centre: the scores of all 5,000 records for one target's auxiliary information (8 items, 2 wrong), on a log scale. When the target is in the release its record stands far out; when it is not, the best score is the edge of the crowd. Right: re-identification rate against the number of auxiliary items, for three weightings, and for the likelihood-ratio weights when a quarter of the items are wrong.](figures/link_netflix.svg){#fig-link-netflix}

To test the eccentricity rule, the script also scores 400 outsiders, people drawn from the same model who are not in the release, so that any match it declares for them is false. With 8 items, 2 of them wrong, and $\varphi = 1.5$, 94.8% of insiders are correctly re-identified, but 26.3% of outsiders are also "matched" to someone. The median eccentricity is 7.32 for insiders and 0.72 for outsiders, but the outsiders' distribution has a long tail. Holding the false-match rate to 1% means raising $\varphi$ until only 1% of outsiders clear it, which happens at $\varphi = 4.83$, and that threshold still re-identifies 73.0% of insiders. @fig-link-eccentricity shows the whole trade-off, the true positive rate (insiders matched) against the false positive rate (outsiders matched) as $\varphi$ varies, with false positives on a log axis.

![True against false positive rates of the eccentricity test as the threshold $\varphi$ is swept, for 4, 6 and 8 auxiliary items with a quarter wrong. The dots mark $\varphi = 1.5$. A false positive rate of 0 is drawn at the left edge.](figures/link_eccentricity_roc.svg){#fig-link-eccentricity}

### Check yourself

- Why do the likelihood-ratio weights in @eq-link-lr-weights contain a term in $\varepsilon$ that is the same for every item, and what does it do? *(It is the evidence from agreeing at all, independent of rarity. When the auxiliary information is reliable ($\varepsilon$ small), every agreement counts a lot and a record that misses one item is heavily penalized relative to one that matches all of them.)*
- A record agrees with 3 auxiliary items, each rated by 1% of 480,189 users. How many unrelated records would you expect to agree with all three, if agreements were independent? *($480{,}189 \times 10^{-6} = 0.48$, since each unrelated record agrees on all three with probability $0.01^3 = 10^{-6}$: about half a false match, so three such items are close to, but not quite, enough.)*

## Does releasing a sample protect? {#sec-link-sampling}

### Context

Statistical agencies and companies often release a sample: 1% of a census, 5% of the patients. The argument is plausible deniability. A record that is unique in the sample need not be unique in the population: the person you have in mind may share their attributes with someone who was not sampled, so a match is not proof. Rocher, Hendrickx & de Montjoye (2019) measured how much this argument is worth. They fitted a Gaussian copula model (each attribute's own frequencies, tied together by the correlations of a multivariate normal) to a sample and used it to estimate, for each record, how likely it is to be unique in the population and so how likely a match is to be correct. Their model predicted that 99.98% of Americans would be correctly re-identified in any dataset with 15 demographic attributes, and that releasing a sample does little to change that. This section derives the two probabilities that matter and checks them.

### Statement

Each of $n$ people is in the release independently with probability $\pi$, the **sampling fraction**. Consider a cell with probability $p$, and let $\lambda = (n - 1)p$ be the expected number of *other* people in it. There are two ways to attack, depending on which side the adversary starts from.

**From a person to the release.** The adversary knows a target's cell and finds exactly one record in that cell in the release. There are two ways that can happen. Either the target was sampled and none of the others in the cell were, and the match is right. Or the target was not sampled and exactly one of the others was, and the match is wrong. If the cell holds $J$ others, the wrong story is $J$ times as likely as the right one, because swapping who was sampled between the target and any one of the others leaves the probability unchanged. How many others should you expect? The release shows no second record in the cell, so the others who are there were all left out of the sample, and each of the $n - 1$ others is in the cell and unsampled with probability about $p(1 - \pi)$. So the cell is expected to hold about $\lambda(1 - \pi)$ others, not $\lambda$, and the wrong story is about $\lambda(1 - \pi)$ times as likely as the right one. A match is correct with probability "right over right plus wrong":

$$\kappa = \Pr[\text{correct} \mid \text{one match}] = \frac{1}{1 + \frac{(n-1)p(1 - \pi)}{1 - p\pi}} \approx \frac{1}{1 + \lambda(1 - \pi)}.$$ {#eq-link-kappa}

**From a record to the population.** The adversary takes a record that is alone in its cell in the release and looks the cell up in a register of the whole population, such as a voter roll. The record is identified if it is alone in its cell in the population too. Knowing the record is alone in the sample already tells you that nobody else in the cell was sampled. What remains is whether anyone in the cell was left out of the sample. Each of the $n - 1$ others is in the cell but unsampled with probability about $p(1 - \pi)$, so about $\lambda(1 - \pi)$ people are expected to hide there, and the chance that nobody does is

$$\xi = \Pr[\text{population-unique} \mid \text{sample-unique}] = \left(\frac{1 - p}{1 - p\pi}\right)^{n-1} \approx e^{-\lambda(1 - \pi)}.$$ {#eq-link-xi}

**The whole release.** Averaged over the release, the fraction of sample-unique records that are unique in the population is the expected number of records that are unique in both, divided by the expected number of sample-unique records:

$$\frac{\sum_cp_c(1 - p_c)^{n-1}}{\sum_cp_c(1 - p_c\pi)^{n-1}},$$ {#eq-link-sample-unique}

which equals the population uniqueness rate @eq-link-unique as $\pi \to 0$ and rises to 1 at $\pi = 1$.

### Derivation

**From a person** (@eq-link-kappa). The plan is to compute the two stories, "right match" and "wrong match", given the number of others in the cell, and then average over that number.

1. **Name the crowd.** Let $J$ be the number of other people in the target's cell. Each of the $n - 1$ others is in it with probability $p$, independently, so $J \sim \mathrm{Binomial}(n - 1, p)$.
2. **The right match.** Given $J$, "exactly one match and it is the target" means the target is sampled and none of the $J$ others are: probability $\pi(1 - \pi)^J$.
3. **The wrong match.** Given $J$, "exactly one match and it is someone else" means the target is not sampled and exactly one of the $J$ others is. There are $J$ choices of which other, each sampled with probability $\pi$ while the remaining $J - 1$ are not: probability $(1 - \pi)\cdot J\pi(1 - \pi)^{J-1} = J\pi(1 - \pi)^J$.
4. **Form the conditional probability.** The probability of a correct match given one match is the right story over the sum of both, each averaged over $J$. The common factor $\pi$ cancels, leaving $\Pr[\text{correct} \mid \text{one match}] = \dfrac{\mathbb E[(1 - \pi)^J]}{\mathbb E[(1 + J)(1 - \pi)^J]}$.
5. **Compute the first average.** The **probability generating function** of $J$ is $\mathbb E[z^J]$. For a binomial it is $(1 - p + pz)^{n-1}$, because $z^J$ is a product of one factor per other person, $z$ if they are in the cell and 1 if not, and each factor averages to $1 - p + pz$ independently. At $z = 1 - \pi$: $\mathbb E[(1 - \pi)^J] = (1 - p\pi)^{n-1}$.
6. **Compute the second average.** Differentiating $z^J$ in $z$ and multiplying by $z$ gives $Jz^J$, so $\mathbb E[Jz^J] = z\frac{d}{dz}(1 - p + pz)^{n-1} = (n - 1)pz(1 - p + pz)^{n-2}$. At $z = 1 - \pi$: $\mathbb E[J(1 - \pi)^J] = (n-1)p(1 - \pi)(1 - p\pi)^{n-2}$.
7. **Simplify.** Divide numerator and denominator of step 4 by $(1 - p\pi)^{n-1}$ to get @eq-link-kappa. For small $p$, $1 - p\pi \approx 1$ and $(n-1)p = \lambda$, which gives the approximation.

**From a record** (@eq-link-xi).

1. **Sort the others into three groups.** Each other person is, independently, in the cell and sampled (probability $p\pi$), in the cell and not sampled ($p(1 - \pi)$), or outside the cell ($1 - p$).
2. **Condition on sample-uniqueness.** "The record is sample-unique" says that no other person is in the first group. Because each person's group is independent of the others', this condition acts on each person separately: it rules out the first group and rescales the other two by $1 - p\pi$, their total probability. So, given it, each other person is outside the cell with probability $\frac{1 - p}{1 - p\pi}$, still independently.
3. **Ask for population-uniqueness.** Population-unique means all $n - 1$ others are outside the cell: probability $\big(\frac{1-p}{1-p\pi}\big)^{n-1}$. The ratio $\frac{1 - p}{1 - p\pi}$ is $1 - \frac{p(1 - \pi)}{1 - p\pi}$, close to $1 - p(1 - \pi)$ for small $p$, so by Fact 2 the whole power is about $e^{-(n-1)(p - p\pi)} = e^{-\lambda(1 - \pi)}$.

**The whole release** (@eq-link-sample-unique).

1. **Count the records unique in both.** By Fact 1, the expected number of records that are sample-unique and population-unique is $\sum_i\Pr[i \text{ sampled and population-unique}]$. Population-unique implies sample-unique, and sampling is independent of the cells, so each term is $\pi$ times the probability of being population-unique, which is the per-person term of @eq-link-unique. The total is $n\pi\sum_cp_c(1 - p_c)^{n-1}$.
2. **Count the sample-unique records.** Their expected number is $n\pi\sum_cp_c(1 - p_c\pi)^{n-1}$: person $i$ is sampled, and each other person is either outside $i$'s cell or not sampled, which fails only with probability $p_c\pi$.
3. **Divide.** The factor $n\pi$ cancels. Each cell's term in the numerator is its term in the denominator times $\big(\frac{1 - p_c}{1 - p_c\pi}\big)^{n-1}$, its $\xi$ from @eq-link-xi, so the ratio is an average of $\xi$ over cells weighted by how often each is sample-unique. As $\pi \to 0$ the denominator tends to $\sum_cp_c = 1$, leaving the population uniqueness rate. At $\pi = 1$ the two sums are equal and the ratio is 1.

### Interpretation

Sampling only helps when the cell is crowded. In both formulas, sampling enters only through the factor $1 - \pi$, which multiplies $\lambda$. For rare combinations, $\lambda \ll 1$, both are close to 1 whatever $\pi$ is. Because $1 - \pi \le 1$, both are at least their values at $\pi = 0$: $\kappa \ge 1/(1 + \lambda)$ and $\xi \ge e^{-\lambda}$ for every sampling fraction. Below a few percent, shrinking the sample further barely changes anything, because $1 - \pi$ is already close to 1.

Matching from a person is more forgiving than matching from a record: $\kappa \ge \xi$, since $1/(1 + x) \ge e^{-x}$ (which is $e^x \ge 1 + x$, the exponential lying above its tangent line at 0). A person-to-release match can be right even when the target is not unique in the population: the others in the cell may simply not have been sampled.

At small sampling fractions, a sample unique is a population unique at the population rate. By @eq-link-sample-unique, a 1% sample gives plausible deniability exactly to the people who were not unique in the population anyway. With enough attributes, that is almost nobody.

Finally, the adversary can tell which matches to trust. $\lambda$ depends on the cell, and the sample is a good sample of the population, so the adversary can estimate $p$ from it and compute $\kappa$ or $\xi$ for each record. Rocher et al. showed this with a copula model; the script does it with the product of the marginals estimated from a 1% sample.

### Example: a 1% sample of a synthetic census

For a single cell, @tbl-link-kappa gives both probabilities in their Poisson forms, the approximations on the right of @eq-link-kappa and @eq-link-xi, for three values of $\lambda$. For instance, with $\lambda = 0.1$ and $\pi = 0.01$, $\kappa = 1/(1 + 0.1 \times 0.99) = 1/1.099 = 0.910$.

| $\lambda$ (expected others in the cell) | $\kappa$, $\pi = 0.01$ | $\kappa$, $\pi = 0.5$ | $\xi$, $\pi = 0.01$ | $\xi$, $\pi = 0.5$ |
| --- | --- | --- | --- | --- |
| 0.1 | 0.910 | 0.952 | 0.906 | 0.951 |
| 0.5 | 0.669 | 0.800 | 0.610 | 0.779 |
| 2 | 0.336 | 0.500 | 0.138 | 0.368 |

: Probability that a match in a sample is correct: from a person to the release ($\kappa$) and from a sample-unique record to the population ($\xi$), for three crowding levels and two sampling fractions. {#tbl-link-kappa}

Going from a 50% to a 1% sample lowers $\kappa$ by $0.952 - 0.910 = 0.04$ for a rare cell and by $0.500 - 0.336 = 0.16$ for a crowded one.

The script then draws a population of $n = 100{,}000$ with the first seven attributes of @tbl-link-census; 82.8% of its people are unique. @eq-link-sample-unique gives 0.8301 of sample uniques being population-unique at $\pi = 0.01$ (simulated: 0.8334 over 19,951 sample uniques in 20 samples), 0.8493 at $\pi = 0.1$ and 0.9235 at $\pi = 0.5$. As the formula predicts, the 1% value sits just above the population rate of 0.828. A 1% sample leaves 83% of its unique records identifiable against a population register.

The adversary's own estimate works too. From one 1% sample with 993 sample-unique records, it estimates each attribute's frequencies, multiplies them to get an estimate $\hat p$ of each record's cell probability, and computes $\xi$ by @eq-link-xi with $n$ and $\pi$, which are public. Its mean prediction is 0.817 against an actual 0.824. The predictions also sort the records well. The 415 records it rates above 0.95 average a prediction of 0.980 and are population-unique 99.3% of the time; the 119 it rates below 0.5 average 0.284 and are unique 30.3% of the time. @fig-link-sampling shows both experiments.

![Left: the fraction of sample-unique records that are unique in a population of 100,000, against the sampling fraction, for 5 to 8 attributes; lines are @eq-link-sample-unique, dots are simulations. As $\pi \to 0$ each curve levels off at the population's uniqueness rate. Right: the adversary's predicted probability that a sample-unique record is unique in the population, estimated from a 1% sample, against the observed fraction, in bins of width 0.1.](figures/link_sampling.svg){#fig-link-sampling}

### Check yourself

- A combination is expected to be shared by $\lambda = 0.05$ other people. What is $\kappa$ for a 1% sample, and for a full release? *($1/(1 + 0.05 \times 0.99) = 0.953$ and $1/(1 + 0) = 1$.)*
- Why does @eq-link-sample-unique tend to the population uniqueness rate, not to 0, as $\pi \to 0$? *(A tiny sample has almost no collisions, so nearly every sampled record is sample-unique. The fraction of them that are population-unique is then just the fraction of the population that is unique.)*

## Code: check every formula by simulation

The script [code/linkage.py](code/linkage.py) computes every number in this chapter and draws every figure. The uniqueness formulas are computed by summing over all cells (grouped by probability, so even 59 million cells are summed quickly) and checked against simulated populations. The ratings experiment is a scaled-down Netflix: a NumPy matrix of 5,000 users, scored with @eq-link-score. From the repository root:

```bash
uv run python book/part02-attacks/code/linkage.py
```

**Output** (seed 0), summarized in @tbl-link-code:

| Experiment | Result |
| --- | --- |
| Bits | $\log_2(8\times10^9) = 32.9$; $\log_2(248.7\times10^6) = 27.9$; ZIP 15.3 + birth date 14.8 + sex 1.0 = 31.1 bits |
| Sweeney, uniform cells | $M = 2.34\times10^9$; $n/M = 0.106$; fraction unique 0.899 |
| World, uniform cells | 33 bits: fraction unique 0.394. Bits needed for 50%, 90%, 99%: 33.4, 36.1, 39.5 |
| Toy release | Classes of sizes 1, 1, 1, 2, 3: linkage success 0.625 |
| Synthetic census, $n = 10^4$ | @tbl-link-census; McDiarmid 95% half-width 0.0272 |
| Synthetic ratings, 5,000 users | 203,029 ratings, density 0.0102; median bits per item 5.83; rates in @tbl-link-netflix |
| Eccentricity, 8 items, 2 wrong | $\varphi = 1.5$: insiders 0.948, outsiders 0.263. $\varphi = 4.83$: insiders 0.730, outsiders 0.010 |
| Sampling, $n = 10^5$, 7 attributes | Population unique 0.828. Sample uniques that are unique: 0.8301 ($\pi = 0.01$; simulated 0.8334), 0.8493 ($\pi = 0.1$), 0.9235 ($\pi = 0.5$) |
| $\kappa$ and $\xi$ | @tbl-link-kappa |
| Adversary's estimate of $\xi$, 1% sample | Mean predicted 0.817, actual 0.824; above 0.95: predicted 0.980, actual 0.993; below 0.5: predicted 0.284, actual 0.303 |

: The script's output with the default seed. {#tbl-link-code}

[tests/test_linkage.py](../../tests/test_linkage.py) checks that the output still matches.

**What to take from it.** First, the formulas are right: every simulated fraction agrees with its exact expectation to within 0.004. Second, the bits beyond $\log_2n$ decide uniqueness. A few bits short, a minority is unique; seven bits beyond, almost everyone is, in a census table and in a ratings matrix alike. Third, the shortcuts err in known directions. For real, skewed attributes the uniform model overstates uniqueness and the collision bound understates it, so compute the exact sum; it needs only a model of the population. Fourth, the decision rule needs calibration. Picking the top score is nearly always right when the target is in the data, but the popular threshold $\varphi = 1.5$ matches a quarter of outsiders in this toy, so choose $\varphi$ from the false positive rate you can accept. Last, sampling is not a defence. It protects the people in crowded cells, who were not at much risk, and the adversary can tell who they are.

## References

- Sweeney, "Simple Demographics Often Identify People Uniquely" (Carnegie Mellon University, Data Privacy Working Paper 3, 2000): the 87% estimate for {ZIP code, birth date, sex} and its variants with coarser geography. Sweeney, "k-Anonymity: A Model for Protecting Privacy" (International Journal of Uncertainty, Fuzziness and Knowledge-Based Systems, 2002) tells the Massachusetts story in its introduction.
- Golle, "Revisiting the Uniqueness of Simple Demographics in the US Population" (Workshop on Privacy in the Electronic Society, 2006): the same estimate from the 2000 census, and how it depends on the granularity of the attributes.
- Narayanan & Shmatikov, "Robust De-anonymization of Large Sparse Datasets" (IEEE Symposium on Security and Privacy, 2008): the scoring algorithm, the eccentricity test, the formal notion of sparsity, and the Netflix Prize experiments.
- Rocher, Hendrickx & de Montjoye, "Estimating the success of re-identifications in incomplete datasets using generative models" (Nature Communications, 2019): the Gaussian copula model, per-record correctness, and why sampling and incompleteness do not protect.
- de Montjoye, Hidalgo, Verleysen & Blondel, "Unique in the Crowd: The privacy bounds of human mobility" (Scientific Reports, 2013): four spatio-temporal points identify 95% of 1.5 million mobile phone users, the same sparsity argument for location traces.
- Ohm, "Broken Promises of Privacy: Responding to the Surprising Failure of Anonymization" (UCLA Law Review, 2010): the legal side, with the Massachusetts, search-log and Netflix cases.
- Dwork, McSherry, Nissim & Smith, "Calibrating Noise to Sensitivity in Private Data Analysis" (2006): where the response to these attacks starts. The next part introduces differential privacy, a guarantee that holds whatever auxiliary information the adversary has. The next chapter first looks at the definitions that came before it.

## Cheat sheet

$n$ people, cells $c$ with probabilities $p_c$, $M$ cells, $U$ the number of unique people, $\pi$ the sampling fraction, $\lambda$ the expected number of others in the target's cell. @tbl-link-cheat collects the results.

| Result | Formula | Note |
| --- | --- | --- |
| Linkage success | $J/n$, $J$ = number of distinct quasi-identifier combinations | Target known to be in the release; at most $1/k$ if all classes have $\ge k$ records |
| Expected fraction unique | $\sum_cp_c(1 - p_c)^{n-1}$ | Exact for independent people |
| Uniform cells | $(1 - 1/M)^{n-1} \approx \exp(-2^{\log_2n - b})$ | 50% at $b = \log_2n + 0.53$; 99% at $\log_2n + 6.64$ |
| Collision bound | $\ge(1 - \sum_cp_c^2)^{n-1}$; $\sum_cp_c^2 = \prod_j\sum_vp_j(v)^2$ | Jensen; tight for uniform cells |
| Concentration | $\Pr[\lvert U/n - \mathbb EU/n\rvert \ge t] \le 2e^{-nt^2/2}$ | McDiarmid with $c_i = 2/n$ |
| Netflix score | $\sum_{i\in\mathrm{aux}}w_i\,\mathrm{Sim}(\mathrm{aux}_i, r_i)$ | $w_i \approx \ln(1/s_i) + \ln\frac{1-\varepsilon}{\varepsilon}$ is the log-likelihood ratio |
| Eccentricity | $(s_1 - s_2)/\sigma \ge \varphi$ | Calibrate $\varphi$ on outsiders at a small false positive rate |
| Expected false matches | $n\prod_{i\in I}s_i = 2^{\log_2n - \sum_i\log_2(1/s_i)}$ | Each item is worth its surprisal in bits |
| Correct match, person to release | $\kappa \approx 1/\big(1 + \lambda(1 - \pi)\big)$ | $\ge 1/(1 + \lambda)$ for every $\pi$ |
| Sample-unique is population-unique | $\xi \approx e^{-\lambda(1 - \pi)}$ | $\ge e^{-\lambda}$ for every $\pi$ |
| Sample uniques that are real | $\sum_cp_c(1 - p_c)^{n-1}/\sum_cp_c(1 - p_c\pi)^{n-1}$ | Population uniqueness rate as $\pi \to 0$ |

: Every result in this chapter. {#tbl-link-cheat}

**Rules of thumb**

- $\log_2n$ bits name one person: 33 for the world, 28 for the US, 13 for a town of 10,000. About 7 bits more make 99% of people unique.
- Add up the bits of the attributes an adversary might know. If the sum exceeds $\log_2n$, assume most people are unique; if attributes are correlated, the true sum is lower.
- In sparse data every rare item is worth $\log_2(1/s_i)$ bits, so a handful of facts is enough, noise or not.
- Removing names, releasing a sample, or perturbing a few values does not anonymize a dataset. The defence has to hold whatever auxiliary data exist. The next chapter tests the classic attempts, k-anonymity and its successors, against that requirement.

**Practice.** Exercises for this chapter, each with a worked solution, are in [the exercise book](solutions/07-linkage-reidentification.md).
