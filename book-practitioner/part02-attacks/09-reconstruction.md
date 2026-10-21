# Reconstruction attacks

## Why this matters

A hospital keeps one sensitive bit per patient, recording whether that patient tested positive. It only answers counting questions such as "how many of the patients on this list tested positive?", with a little noise added to each answer. Dinur and Nissim (2003) showed that this is not safe unless the noise is large. Each answer is a linear equation in the patients' bits, and enough equations with small enough errors determine the unknowns.

The protection can fail in two ways. The error can be too small, or there can be too many answers, because independent errors are averaged away as the attacker asks more. Dwork and Roth call this the **Fundamental Law of Information Recovery**: overly accurate answers to too many questions destroy privacy. Any aggregate release must be checked against both.

## Counts are linear equations {#sec-recon-toolkit}

Write the secret as a list of bits $d = (d_1, \dots, d_n)$, one for each of $n$ people. A **subset-sum query** names a set of people and asks how many of them have bit 1. The attacker's $m$ queries form a matrix $A$ of zeros and ones, with one row per query and one column per person, so the true answers are $Ad$.

The party that answers, the **curator**, publishes the answers with an error vector $e$,
$$a = Ad + e.$$
Write $E$ for a bound on the size of every error. The attacker is scored by the fraction of bits it recovers, against one half from guessing ([Threat models and privacy games](01-threat-models.md)).

## Exact and random queries: how much error is enough {#sec-recon-exponential}

An attacker who could ask all $2^n$ subset sums would get at most $4E$ bits wrong, whatever errors the curator chose. So a curator whose error is a vanishing fraction of $n$ is **blatantly non-private**. A distortion the attacker can simulate, such as rounding to a multiple of 4, is not noise at all, because the attacker can round each candidate's answers too.

Nobody can ask $2^n$ questions, but random ones suffice. Each random query includes every person with probability one half. The Dinur–Nissim decoder solves a **linear program**, searching for fractional bits whose answers lie within $E$ of the published ones, and then rounds. The price is that the error must now be well below $\sqrt n$, not just below $n$, because a random query picks up each wrong bit only half the time. A number of queries of order $n \log n$ then suffices. The proved constants are pessimistic, and in practice least squares already works at errors of about $0.2\sqrt n$ with $4n$ queries.

## Least squares: the attack to run on your system {#sec-recon-ls}

**Least squares** needs no solver and no value of $E$. It picks the vector whose answers come closest to the published ones in squared error,
$$\hat x = (A^\top A)^{-1}A^\top a,$$
and then clips and rounds each entry. Suppose the curator adds independent normal noise, and write $\sigma$ for its standard deviation. A bit then comes out wrong when its error crosses one half, so write $\Phi$ for the standard normal cumulative distribution function. The expected fraction of bits recovered is then about
$$\Phi\left(\frac{\sqrt{m - n}}{4\sigma}\right).$$ {#eq-recon-ls-random}

Plug your system's numbers into it. At a fixed number of queries per person only $\sigma/\sqrt n$ matters, so $\sqrt n$ is the scale of the threat, and recovery degrades smoothly rather than switching off at a threshold. On 500 people with 2000 queries, the prediction is within 0.013 of every simulated value in @tbl-recon-ls.

| $\sigma/\sqrt n$ | 0.1 | 0.25 | 0.5 | 1 | 2 |
| --- | --- | --- | --- | --- | --- |
| Recovered, simulated | 1.0000 | 0.9655 | 0.8130 | 0.6550 | 0.5875 |
| Recovered, @eq-recon-ls-random | 1.0000 | 0.9585 | 0.8069 | 0.6676 | 0.5858 |

: Least-squares reconstruction of $n = 500$ bits from $m = 2000$ random subset sums with Gaussian noise. {#tbl-recon-ls}

The last column shows how a curator can look safe and still fail. With noise of twice $\sqrt n$, four queries per person recover only 0.5875 of the bits. But if the curator answers any number of queries at that noise level, 106.1 queries per person are predicted to recover 90%. So the noise must grow like $\sqrt m$ to hold the attack back.

Errors chosen against the decoder hurt more than random noise of the same size. Least squares amplifies errors most along the direction its queries see least clearly. A curator that pushed every error that way caused 143 wrong bits, where normal noise of the same size caused about 17. So test a system with errors aimed at the decoder, too.

## The Fundamental Law of Information Recovery {#sec-recon-law}

Noise of order $\sqrt n$ can stop reconstruction without giving privacy. A curator that answers from a secret random sample of half the people hides the others completely, yet enough queries reveal every sampled person's bit exactly. [The definition of differential privacy](../part03-dp-foundations/01-definition.md) protects every individual instead.

Exact answers are also cheap to exploit. An exact count of up to $n$ people carries at most $\log_2(n + 1)$ bits. So any attack needs at least about $n/\log_2(n + 1)$ counts to pin down the $n$ bits, and least squares with exact answers needs only $n$, close to that floor. Coarser answers carry fewer bits each, but they are no defence, because an attacker can simply ask more.

Weaker goals need far fewer aggregates. For example, a genome study's published allele frequencies reveal whether a given person took part, as Homer and colleagues showed in 2008 ([Lower bounds](../part03-dp-foundations/10-lower-bounds.md)).

## Attacking a deployed query system {#sec-recon-diffix}

In 2017 Aircloak offered a bounty for breaking Diffix, its commercial system for counting queries in a restricted form of SQL. Diffix added noise for each condition in a query, so the noise grew with the length of a query but not with the number of queries. The noise was sticky, so repeating a query gained nothing. Its language forbade OR, so naming an arbitrary subset took many noisy conditions.

Cohen and Nissim (2020) picked random short descriptions instead, and took whatever subset each one selected. A query multiplied a record's public identifier by a prime, raised it to a fractional power, and kept the record when a chosen digit was divisible by 2 or 5. Diffix counted that as one condition, so each query carried little noise. On a real banking dataset, 3,500 such queries recovered every secret bit, with reported noise of standard deviation 4.

Least squares predicts this in advance. Write $\rho$ for the fraction of records a query includes, and $v$ for the average of $\rho(1 - \rho)$ over the queries. From $m$ queries, least squares then recovers about
$$\Phi\left(\frac{\sqrt{v\,(m - n - 1)}}{2\sigma}\right)$$ {#eq-recon-diffix}
of the $n$ bits. For the smallest range in the attack, of 73 records, the prediction is near-total recovery. Holding the attack to 75% would have needed noise 4.9 times larger. The lesson applies to any system, because many distinct queries average away noise that is fixed per query. A restricted language does not help either, since any short formula that splits records independently of their secrets acts as a random query.

## Reconstructing a census block {#sec-recon-census}

Each cell of a published table is a query. Garfinkel, Abowd and Martindale (2019) showed that a constraint solver rebuilds a census block from its statistics in seconds. The full edition repeats this with a fictional block of seven people. For six groups of them it publishes a count, a median age and a mean age, and means to two decimals give exact sums of ages.

A decoder lists candidate blocks and discards each one that contradicts a published number, and exactly one block survives. So eighteen numbers determine the age, sex, race and marital status of all seven people.

Real publications are more exposed, because the same people appear in thousands of overlapping cells. The usual protections also leave most constraints exact. For example, swapping changes a few households and leaves the rest, while coarsening and suppression remove some constraints but keep the others. A suppressed cell even adds an inequality, because it says the count is below the threshold.

The Census Bureau ran this attack on its own 2010 publications. It rebuilt a record for every person counted, and about 46% of the records matched a real person exactly on block, sex, age, race and ethnicity. Linking the records to commercial data then confirmed the identity of about 17% of the population. So the Bureau moved the 2020 Census to differential privacy, using its TopDown algorithm, and published the source code of its disclosure avoidance system.

## Auditing an aggregate release {#sec-recon-audit}

1. **Count the equations.** If the published cells or allowed queries outnumber the people, reconstruction is possible in principle.
2. **Classify the error.** Rounding, swapping and suppression are not noise, because the attacker can simulate them or work around them. Fresh noise per answer is averaged away, and sticky noise only stops repeats, while restricting or auditing the allowed queries fails because ordinary-looking formulas act as random queries. Only noise calibrated to a budget over all answers holds up.
3. **Predict least squares.** Plug the noise and the number of queries into @eq-recon-ls-random or @eq-recon-diffix. A prediction far above one half means the system leaks.
4. **Run the attack.** For noisy counts, solve with numpy's `np.linalg.lstsq`, as below, or pose the linear program with SciPy's `scipy.optimize.linprog`, bounding each unknown between 0 and 1. For tables of counts, medians and means, write each cell as a constraint on whole-number unknowns. Then solve the integer program with SciPy's `scipy.optimize.milp`, PuLP or Gurobi, and count the solutions.
5. **Fix it with a budget.** If the attack works, move to noise that grows with the number of answers, which is what differential privacy provides.

```text
A = np.random.default_rng(0).integers(0, 2, size=(m, n))   # random subset-sum queries
x = np.linalg.lstsq(A, answers(A), rcond=None)[0]           # least-squares estimate
recovered = np.mean(np.clip(x, 0, 1).round() == d)          # against a test secret d
```

## Code

The functions below predict least-squares recovery at your noise level and the queries per person that reach a target.

```python
from math import sqrt
from statistics import NormalDist

PHI = NormalDist()


def ls_recovered(n: int, m: int, sigma: float) -> float:
    """Predicted fraction of n secret bits that least squares recovers from m noisy random subset sums."""
    sd = 2 * sigma * sqrt(n / ((n + 1) * (m - n - 1)))  # spread of each decoded bit's error
    return PHI.cdf(0.5 / sd)  # a bit comes out wrong when its error crosses 1/2


def queries_per_person(target: float, sigma_over_sqrt_n: float) -> float:
    """m/n at which least squares recovers a fraction `target` of the bits, for noise sigma = c * sqrt(n)."""
    return 1 + (4 * sigma_over_sqrt_n * PHI.inv_cdf(target)) ** 2



n, m = 500, 2000
print(" ".join(f"{ls_recovered(n, m, c * sqrt(n)):.4f}" for c in (0.1, 0.25, 0.5, 1, 2)))  # 1.0000 0.9585 0.8069 0.6676 0.5858
print(f"{queries_per_person(0.9, 2.0):.1f}")  # 106.1
```

The full edition's script, [reconstruction.py](../../code/part02-attacks/reconstruction.py), runs the attacks themselves.

## Cheat sheet

| Setting | Result | Use it for |
| --- | --- | --- |
| All $2^n$ subset sums | At most $4E$ wrong bits | Against the worst queries, error must be of order $n$ |
| Random queries | About $n \log n$ suffice if the error is well below $\sqrt n$ | Error must be of order $\sqrt n$ or more |
| Least squares, noise $\sigma$ | Recovered $\approx \Phi(\sqrt{m - n}/(4\sigma))$ | Predicting an attack at your noise level |
| Diffix-style fixed noise | Averaged away by many distinct queries | Syntax limits and sticky noise are no defence |
| Exact counts | At most $\log_2(n + 1)$ bits each | Attacks need only about $n/\log n$ to $n$ of them |
| Published tables | Each cell is a constraint | Run a solver and count the solutions |

: The results of this chapter. {#tbl-recon-cheat}

**Rules of thumb**

- The threat scale of a random query is $\sqrt n$, and errors well below it reveal the database.
- Budget the error over all answers, because more answers average away independent noise.
- Aggregates, small-cell suppression and swapping are not safe by themselves, so try to reconstruct your own release before publishing it.
- Stopping reconstruction is not privacy, so ask what the system guarantees for each individual.

## Further reading

- Cohen & Nissim, "Linear Program Reconstruction in Practice" (Journal of Privacy and Confidentiality, 2020): the attack on Diffix.
- Garfinkel, Abowd & Martindale, "Understanding Database Reconstruction Attacks on Public Data" (2019): a census block rebuilt with a solver.

**Derivations.** Every result here is derived step by step in [the full edition's chapter](../../book-full/part02-attacks/09-reconstruction.md).
