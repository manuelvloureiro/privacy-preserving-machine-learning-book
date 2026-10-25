# Reconstruction attacks

## What it is

A **reconstruction attack** rebuilds the records behind published aggregates. It works because every count is a linear equation in the secret values of the people counted. Suppose each of $n$ people has one secret bit. A **subset-sum query** names a set of people and asks how many of them have bit 1. Write $d$ for the list of secret bits and $A$ for the matrix of zeros and ones that lists the attacker's $m$ queries. The party that answers, the **curator**, publishes the true answers plus an error vector $e$,
$$a = Ad + e.$$

**Least squares** finds the real numbers whose answers come closest to the published ones, then clips and rounds each to 0 or 1. Suppose each query includes every person with probability one half. Suppose too that the curator adds independent normal noise, and write $\sigma$ for its standard deviation and $\Phi$ for the standard normal cumulative distribution function. Least squares then recovers about a fraction
$$\Phi\left(\frac{\sqrt{m - n}}{4\sigma}\right)$$ {#eq-recon-ls-random}
of the bits. Compare it with the **majority guess**, which gives everyone the commoner value and scores its share, one half only for balanced bits. With $m$ a few times $n$, noise well below $\sqrt n$ therefore reveals the data. A curator who answers all $2^n$ subset sums needs error of order $n$ to stop reconstruction (Dinur and Nissim 2003).

These results are the **Fundamental Law of Information Recovery**: answering too many questions too accurately destroys privacy. Stopping reconstruction is not privacy, though. A curator that answers from a secret random half of the people hides the other half, yet gives the sampled half away exactly. And learning only whether someone took part needs far fewer statistics, as a genome study's published frequencies of genetic variants showed (Homer et al. 2008, [Lower bounds](../part03-dp-foundations/10-lower-bounds.md)).

## When you meet it

You meet it whenever you publish counts, cross-tabulations, dashboards or a query interface, because each cell of a table is a query. The Census Bureau attacked its 2010 publications and rebuilt a record for every person. About 46% of the rebuilt records matched a real person exactly on block, sex, age, race and ethnicity. Linking them to commercial data then confirmed the identity of about 17% of the population, so the Bureau moved the 2020 Census to differential privacy. Query systems fall too. The noise of Diffix, a commercial one, did not grow with the number of queries. So Cohen and Nissim (2020) averaged it away with 3,500 queries per group of loans in a banking dataset, recovering every bit they targeted.

## How to use it

To audit an aggregate release, first count the equations. If the published cells or allowed queries outnumber the people, exact answers can pin down every secret, and only the noise protects them. Then plug your noise level and number of queries into @eq-recon-ls-random, and run the attack if the prediction is far above the majority guess, as the snippet does. The release fails when either decoder beats that guess.

```python
from statistics import NormalDist

import numpy as np

rng = np.random.default_rng(0)
n, m = 500, 2000  # people with a secret bit each, and random subset-sum queries
sigma = 0.25 * np.sqrt(n)  # noise sd added to each answer
secret = rng.integers(0, 2, n)
queries = rng.integers(0, 2, (m, n))
answers = queries @ secret + rng.normal(0, sigma, m)
estimate = np.linalg.lstsq(queries, answers, rcond=None)[0]
recovered = np.mean(np.clip(estimate, 0, 1).round() == secret)
predicted = NormalDist().cdf(np.sqrt(m - n) / (4 * sigma))
print(f"recovered {recovered:.4f}, predicted {predicted:.4f}")  # recovered 0.9660, predicted 0.9584
```

The number of queries matters as much as the noise. With noise of twice $\sqrt n$, four queries per person recovered only 0.5875 of the bits in the full edition's simulation, so the curator looks safe. But solving @eq-recon-ls-random at that noise shows that 106.1 queries per person recover 90%. So the noise must grow with the number of answers, which is what differential privacy provides.

For queries that are not random, Dinur and Nissim's linear-programming decoder fits unknowns between 0 and 1 to the answers with `scipy.optimize.linprog`. For published tables, write each cell as a constraint on whole-number unknowns, solve with `scipy.optimize.milp`, and count the solutions, since a single one rebuilds the records.

## What goes wrong

The usual mistake is to treat rounding, swapping households or small-cell suppression as noise. An attacker can round the answers of each dataset it tries, so rounding hides nothing. Swapping leaves most constraints exact, and a suppressed cell even adds one, since it says the count is below the threshold. Sticky noise, which repeats the same answer to a repeated query, only stops repeats, which the attack never needs.

A second is to test one table at a time, when the risk lies in the union of all releases. A third is to trust least squares alone, because errors chosen against it hurt it more than random noise does. So reconstruct your own release with both decoders before publishing it, and record the result in [Pre-release privacy risk assessment](10-risk-assessment.md).

## References

- Cohen & Nissim, "Linear Program Reconstruction in Practice" (Journal of Privacy and Confidentiality, 2020): the attack on Diffix.
- Garfinkel, Abowd & Martindale, "Understanding Database Reconstruction Attacks on Public Data" (2019): a census block rebuilt with a solver.

**Derivations.** Every result here is derived step by step in [the full edition's chapter](../../book-full/part02-attacks/09-reconstruction.md).
